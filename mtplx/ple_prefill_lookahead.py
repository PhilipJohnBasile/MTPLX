"""One-slot lookahead for the PLE n-gram prefill gather.

Pure Python + NumPy: no MLX import, no MLX array is created or evaluated
anywhere in this module, and nothing here runs on the GPU.  The worker thread
this module owns produces RAW NUMPY ROWS only; the generation owner thread
turns them into MLX arrays.

The problem
-----------
Every prefill chunk opens with a synchronous PLE n-gram gather:
``NGramEmbedding.stage`` hashes the chunk's 2,048 token ids into 32,768 sidecar
row ids and calls ``_SidecarGather.gather_np``.  The census (scratchpad
J-prefill-attribution, cb-sorted i in [432, 7151)) finds eight host-late GPU
stalls, one immediately before each chunk's ``gather_frontbfloat16_int32_int_2``,
totalling **2,313 ms** -- 96% host-late, GPU fully idle -- against 158 ms of
in-chunk idle for everything else.  The gather's own GPU cost is 3.5 ms.

What the stall actually is
--------------------------
Measured host-only, at the production row geometry (16 ngram heads/token,
head_dim 160, q4/g32 -> three maps of 80/10/10 bytes per row), on a
page-cache-warm file (production runs prewarm the table):

===============================  ==========  =========  ==============
formulation                      pread warm  memmap fx  per map, total
===============================  ==========  =========  ==============
shipped (step<=64, 512 tasks)      164.8 ms    0.62 ms       165.5 ms
16 big batches                     175.6 ms    0.52 ms       176.2 ms
one pread per distinct page        183.6 ms    0.92 ms       184.6 ms
**no warm, memmap only**           **0 ms**    0.44 ms      **0.44 ms**
===============================  ==========  =========  ==============

``np.unique`` on the 32,768 ids is 1.4 ms.  So the stall is **not I/O and not
NumPy**: it is ~5 us of GIL-contended Python per ``os.pread``, times 3 maps
times the chunk's unique rows.  No batching, offset-precomputation or
page-dedup formulation moves it -- the syscall count is the cost.  Dropping
the warm pass is not an option either: it exists for the COLD case (the
``_SidecarGather`` docstring's 2026-08-26 receipt: 36 s serial faults vs 7.5 s
pooled on a cold 100k-token prefill), and warmth is not knowable cheaply.

So the fix is to move the whole preparation off the critical path.  Every
prompt token is known before chunk 0 runs, so chunk k+1's row ids, unique set,
page warm and raw row read can all happen on a worker thread while chunk k's
48-layer forward occupies the GPU for ~1.2 s.  The owner thread then only
builds the MLX arrays.

Exactness
---------
The worker runs the same expression the owner would have run --
``np.ascontiguousarray(map[unique])[inverse]`` over the same read-only memmaps
with the same ids -- so the bytes are identical by construction.  The prepared
payload is accepted only after its span's token ids compare equal to the ids
``stage`` was actually called with; a mismatch discards the payload and takes
the ordinary path, counted, never silently.

The hot-row LRU is owner-thread-only state, so the worker path is restricted
to gathers that bypass it (``len(unique) > _SidecarGather._HOT_PATH_MAX_ROWS``)
-- which is every real prefill chunk, and is checked rather than assumed.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import time
from functools import lru_cache
from typing import Any, Callable, Sequence

__all__ = [
    "COUNTERS",
    "EARLY_ENV_FLAG",
    "ENV_FLAG",
    "EarlyFirstGather",
    "PrefillLookahead",
    "active_early_first_gather",
    "active_lookahead",
    "count",
    "early_enabled",
    "enabled",
    "first_gather_early_scope",
    "last_early_status",
    "last_scope_status",
    "prefill_lookahead_scope",
    "reject_unwired_prefill_loop",
    "reset_counters",
    "snapshot_counters",
]

ENV_FLAG = "MTPLX_FABLE_PLE_PREFILL_LOOKAHEAD"

#: The first chunk's gather is the one the lookahead cannot hide: it has no
#: previous chunk to run behind, so `prefill_lookahead_scope` submits it and
#: `stage()` blocks on it microseconds later (measured on the 16K prefill-stack cell:
#: chunk 1 `ple_gather_s` 0.627 s against 0.0006 s for chunks 2-4).  This flag
#: starts it at REQUEST ARRIVAL instead -- as soon as the prompt ids exist,
#: before the session-bank lookup and the prefill graph setup -- so the host
#: work between tokenisation and the first forward pays for it.  It lives in
#: `mtplx.ple_row_gather` because the vectorised gather it also turns on has
#: to be importable without this module.
EARLY_ENV_FLAG = "MTPLX_FABLE_PLE_FIRST_GATHER_EARLY"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})

#: Engagement receipts.  A lane with no counter is a lane whose benchmark
#: cannot be read (the A/B law); every branch below bumps exactly one.
COUNTERS: dict[str, int] = {}


def count(name: str, n: int = 1) -> None:
    COUNTERS[name] = COUNTERS.get(name, 0) + int(n)


def snapshot_counters() -> dict[str, int]:
    return dict(COUNTERS)


def reset_counters() -> None:
    COUNTERS.clear()
    _LAST_EARLY.update(_EARLY_RECEIPT_ZERO)
    _LAST_SCOPE.update(
        {
            "armed": None,
            "reason": None,
            "spans": 0,
            "required": 0,
            "span_tokens": [],
        }
    )


#: What the most recent :func:`prefill_lookahead_scope` did with the lane.
#: ``COUNTERS`` is int-only and process-cumulative; this carries the one
#: non-numeric fact a receipt needs -- whether THIS prefill ran the candidate
#: at all, and if not, why.  It never contradicts the driver's
#: ``prefill_lookahead_armed``, which stays a statement about the environment
#: flag (the 2026-09-01 blind spot) rather than about one request.
_LAST_SCOPE: dict[str, Any] = {
    "armed": None,
    "reason": None,
    "spans": 0,
    "required": 0,
    "span_tokens": [],
}


def last_scope_status() -> dict[str, Any]:
    return dict(_LAST_SCOPE)


#: The first-gather-early receipt.  ``started_at_ms_before_layer2`` is the head
#: start the lane actually bought: milliseconds between the worker submit at
#: request arrival and the moment the owner thread first NEEDS the rows.  On
#: this architecture that moment is `NGramEmbedding.stage`, called at the top
#: of `Model._forward` -- staging deliberately hoists the gather OUT of the PLE
#: layer (index 2) so no mid-forward GPU sync is needed, so the row tensor is
#: first needed *before* layer 0, not at layer 2.  The name is kept because it
#: is what the number means: how far ahead of the consuming forward the gather
#: started.
_EARLY_RECEIPT_ZERO: dict[str, Any] = {
    "armed": None,
    "reason": None,
    "started_at_ms_before_layer2": None,
    "rows": 0,
    "path": None,
    "resident_fraction": None,
    "span": None,
    "outcome": None,
    "prefetch_rest_rows": 0,
}
_LAST_EARLY: dict[str, Any] = dict(_EARLY_RECEIPT_ZERO)


def last_early_status() -> dict[str, Any]:
    return dict(_LAST_EARLY)


@lru_cache(maxsize=1)
def early_enabled() -> bool:
    """Resolve :data:`EARLY_ENV_FLAG` once.  Unparseable values raise."""

    from mtplx.ple_row_gather import enabled as _row_gather_enabled

    return _row_gather_enabled()


@lru_cache(maxsize=1)
def enabled() -> bool:
    """Resolve :data:`ENV_FLAG` once.  Unparseable values raise."""

    raw = (os.environ.get(ENV_FLAG) or "").strip().lower()
    if raw in _FALSE:
        return False
    if raw in _TRUE:
        return True
    accepted = sorted((_TRUE | _FALSE) - {""})
    raise ValueError(
        f"{ENV_FLAG} must be one of {accepted}, got {os.environ.get(ENV_FLAG)!r}"
    )


def reject_unwired_prefill_loop(loop: str) -> None:
    """Refuse to run an armed lane through a prefill loop it is not wired to.

    The lane only overlaps gathers issued from the loop that opened a
    :func:`prefill_lookahead_scope`.  Any other chunked prefill loop would run
    the ordinary synchronous gather while the arm still called itself the
    candidate -- the 2026-09-01 failure, in a different disguise.  Every such
    loop calls this first, so "armed but inert" cannot be reached at all.
    """

    if not enabled():
        return
    raise RuntimeError(
        f"{ENV_FLAG}=1 but this request takes the {loop!r} prefill loop, "
        "which the lookahead is not wired to; it would measure the control "
        "under the candidate's label. Wire that loop or clear the flag."
    )


_ACTIVE: contextvars.ContextVar["PrefillLookahead | None"] = (
    contextvars.ContextVar("mtplx_ple_prefill_lookahead", default=None)
)


def active_lookahead() -> "PrefillLookahead | None":
    return _ACTIVE.get()


_EARLY_ACTIVE: contextvars.ContextVar["EarlyFirstGather | None"] = (
    contextvars.ContextVar("mtplx_ple_first_gather_early", default=None)
)

#: A process-wide pool, not one per request.  The whole point of the lane is
#: that the submit happens on the request's critical path, and creating a
#: thread there costs more than the first layers it is trying to overlap.  It
#: is the SAME mechanism the lookahead uses (an executor whose tasks produce
#: raw NumPy rows only); the lookahead cannot simply be built earlier because
#: its spans come from the prefill loop, which does not exist yet at request
#: arrival.
#:
#: Two workers, not one: the pre-touch task (c) can run for as long as the
#: prompt is large, and on a single worker the NEXT request's first chunk
#: would queue behind it -- head-of-line blocking that would show up as
#: exactly the TTFT this lane exists to remove.  Ordering within a request
#: does not depend on the worker count: the pre-touch is chained off the
#: first chunk's future, never merely submitted after it.
_EARLY_POOL = None


def _early_pool():
    global _EARLY_POOL
    if _EARLY_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _EARLY_POOL = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="ple-first-gather"
        )
    return _EARLY_POOL


def active_early_first_gather() -> "EarlyFirstGather | None":
    return _EARLY_ACTIVE.get()


class EarlyFirstGather:
    """The first prefill chunk's PLE rows, started at request arrival.

    Two consumers, never both: the ordinary chunked prefill builds a
    :class:`PrefillLookahead` and ADOPTS this future as its slot 0 (so the
    hit/miss/engagement bookkeeping is unchanged, and nothing is prepared
    twice), while a single-chunk prefill -- where the lookahead is inert by
    construction, the 1K cell -- consumes it directly through :meth:`take`.

    Exactness is the lookahead's, unchanged: the worker runs the same
    expression over the same read-only memmaps with the same ids, and the
    payload is accepted only after its span's token ids compare equal to the
    ids `stage` was actually called with.
    """

    def __init__(
        self,
        token_ids,
        span: tuple[int, int],
        prepare: Callable[..., Any],
        *,
        submit: Callable[..., Any] | None = None,
        prefetch_rest: Callable[..., Any] | None = None,
    ) -> None:
        import numpy as np

        self._ids = np.ascontiguousarray(
            np.asarray(token_ids, dtype=np.int64).reshape(-1)
        )
        self._ids.setflags(write=False)
        self.span = (int(span[0]), int(span[1]))
        self.record: dict[str, Any] = {"path": None, "rows": 0}
        self._submit = submit if submit is not None else _early_pool().submit
        self._closed = False
        self.adopted = False
        self.outcome: str | None = None
        self.needed_at_ms: float | None = None
        self._started = time.perf_counter()
        self._future = self._submit(
            prepare, self._ids, self.span[0], self.span[1], self.record
        )
        count("early_submitted")
        # (c) The rest of the prompt, page-warmed BEHIND the first chunk:
        # chained off its future rather than merely submitted after it, so it
        # cannot delay the take whatever the pool's worker count.  It leaves
        # chunks 2..n page-warm for the vectorised gather instead of making
        # each of them re-pread its own rows.
        self._rest_future = None
        self._prefetch_rest = prefetch_rest
        if prefetch_rest is not None and self.span[1] < int(self._ids.shape[0]):
            self._future.add_done_callback(self._submit_prefetch_rest)

    def _submit_prefetch_rest(self, _future) -> None:
        if self._closed or self._prefetch_rest is None:
            return
        self._rest_future = self._submit(
            self._prefetch_rest, self._ids, self.span[1], self.record
        )
        count("early_prefetch_rest_submitted")

    # -- identity -----------------------------------------------------------

    @property
    def token_ids(self):
        return self._ids

    def matches_plan(self, plan_ids) -> bool:
        """Whether ``plan_ids`` is the prompt this gather was started for."""

        import numpy as np

        other = np.asarray(plan_ids, dtype=np.int64).reshape(-1)
        return bool(
            other.shape == self._ids.shape and np.array_equal(other, self._ids)
        )

    def matches_chunk(self, chunk_ids) -> bool:
        """Whether ``chunk_ids`` are exactly this gather's span's tokens."""

        import numpy as np

        ids = np.asarray(chunk_ids, dtype=np.int64).reshape(-1)
        start, end = self.span
        return bool(
            ids.shape[0] == end - start
            and np.array_equal(self._ids[start:end], ids)
        )

    # -- consumption --------------------------------------------------------

    def note_needed(self) -> None:
        """Stamp the head start at the moment the owner first needs the rows."""

        if self.needed_at_ms is None:
            self.needed_at_ms = (time.perf_counter() - self._started) * 1000.0
            _LAST_EARLY["started_at_ms_before_layer2"] = self.needed_at_ms

    def note_outcome(self, outcome: str) -> None:
        self.outcome = outcome
        _LAST_EARLY["outcome"] = outcome

    def _publish(self) -> None:
        _LAST_EARLY["rows"] = int(self.record.get("rows") or 0)
        _LAST_EARLY["path"] = self.record.get("path")
        _LAST_EARLY["resident_fraction"] = self.record.get("resident_fraction")
        _LAST_EARLY["prefetch_rest_rows"] = int(
            self.record.get("prefetch_rest_rows") or 0
        )

    def adopt(self):
        """Hand the in-flight future to the lookahead's slot 0."""

        self.adopted = True
        return self._future

    def take(self, chunk_ids):
        """Payload for a chunk that IS this gather's span, else None."""

        if self._closed or self.adopted:
            return None
        self.note_needed()
        if not self.matches_chunk(chunk_ids):
            count("early_miss_wrong_span")
            self.note_outcome("miss_wrong_span")
            return None
        try:
            prepared = self._future.result()
        except BaseException:
            count("early_worker_error")
            raise
        self._publish()
        if prepared is None:
            # The sidecar declined these ids to the owner-thread hot-row LRU
            # (the aa20bf11 servability rule); the owner pays the ordinary
            # gather, exactly, and that is not an inert lane.
            count("early_miss_ineligible")
            self.note_outcome("miss_ineligible")
            return None
        count("early_hit")
        self.note_outcome("hit")
        return prepared

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._publish()
        if self.outcome is None:
            # Armed, prepared, and never consumed: a bank restore served the
            # whole prompt, or the prefill took a loop this lane is not wired
            # to.  Named in the receipt rather than left to be read as a hit.
            self.note_outcome(
                "never_needed" if self.needed_at_ms is None else "unused"
            )
        if not self.adopted:
            self._future.cancel()
            try:
                self._future.result()
            except Exception:
                pass
        if self._rest_future is not None:
            # Never waited on: it exists to leave pages warm, and a prefill
            # that finished without it is simply a prefill that did not need
            # it.  Cancelling only bites while it is still queued.
            self._rest_future.cancel()


@contextlib.contextmanager
def first_gather_early_scope(early: "EarlyFirstGather | None", reason: str | None = None):
    """Make ``early`` the request-scoped first-chunk gather."""

    _LAST_EARLY.update(_EARLY_RECEIPT_ZERO)
    if early is None:
        _LAST_EARLY["armed"] = False
        _LAST_EARLY["reason"] = reason
        yield None
        return
    _LAST_EARLY["armed"] = True
    _LAST_EARLY["span"] = list(early.span)
    token = _EARLY_ACTIVE.set(early)
    try:
        yield early
    finally:
        _EARLY_ACTIVE.reset(token)
        early.close()


class PrefillLookahead:
    """One prepared prefill chunk, at most, at a time.

    ``prepare`` runs on a single dedicated worker thread; the owner thread
    never blocks on it except in :meth:`take`, and only for the span it is
    about to consume anyway.
    """

    def __init__(
        self,
        token_ids: Sequence[int],
        spans: Sequence[tuple[int, int]],
        prepare: Callable[[int, int], Any],
        *,
        submit: Callable[..., Any] | None = None,
        rows_per_token: int | None = None,
        min_servable_rows: int = 0,
    ) -> None:
        import numpy as np

        self._ids = np.ascontiguousarray(
            np.asarray(token_ids, dtype=np.int64).reshape(-1)
        )
        self._ids.setflags(write=False)
        self._spans = [(int(a), int(b)) for a, b in spans]
        self._prepare = prepare
        self._submit = submit
        # The sidecar's OWN eligibility rule, handed down by the model
        # builder rather than restated here: ``prepare_rows_np`` declines a
        # gather whose unique row count is ``<= _SidecarGather.
        # _HOT_PATH_MAX_ROWS`` (that LRU is owner-thread-only state), and a
        # span hashes to ``tokens * NGramEmbedding.ngram_heads`` rows.  The
        # comparison below is strictly-greater for the same reason the
        # sidecar's is ``<=``: a span of exactly _HOT_PATH_MAX_ROWS rows is
        # declined, and the GDN tail grid cuts 256-token spans -- 256 * 16 =
        # 4,096 rows exactly.  ``rows_per_token=None`` means the geometry is
        # unknown, and then every span is required (the strict contract).
        self._rows_per_token = (
            None if rows_per_token is None else max(0, int(rows_per_token))
        )
        self._min_servable_rows = max(0, int(min_servable_rows))
        self._pool = None
        if submit is None:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ple-lookahead"
            )
            self._submit = self._pool.submit
        self._pending: tuple[int, Any] | None = None
        self._early: "EarlyFirstGather | None" = None
        self._hint: int | None = 0
        self._closed = False
        # Per-scope engagement, kept off the module-global COUNTERS so one
        # request's invariant cannot be satisfied by an earlier request's work.
        self.hits = 0
        self.misses = 0
        self.submits = 0
        self.ineligible = 0
        self.ineligible_small = 0
        #: index -> "hit" | "ineligible" | "ineligible_small" | "miss_empty"
        #: | "miss_wrong_span".  Scalars alone cannot say WHICH span went
        #: unserved, and the engagement verdict below is a per-span statement.
        self._outcomes: dict[int, str] = {}
        #: The spans the worker is DESIGNED to serve.  Everything else is
        #: exempt by construction, not by tolerance: a span the sidecar
        #: declines on sight can never be evidence that the lane ran or did
        #: not.  Two live 500s came from treating a decline as inertness --
        #: a one-chunk HumanEval prompt, then the GDN tail grid cutting the
        #: same prompt into 256 + 144 tokens (4,096 + 2,304 rows, both
        #: declined, reported as {'spans': 2, 'hits': 0, 'ineligible': 2}).
        self.required = [
            index
            for index in range(len(self._spans))
            if self.span_is_servable(index)
        ]
        # The lane overlaps chunk k+1's gather with chunk k's forward, so a
        # prefill of one chunk has nothing to look ahead FROM, and a prefill
        # with no servable span has nothing to look ahead WITH.
        self.armed = len(self._spans) > 1 and bool(self.required)
        self.inert_reason: str | None = None
        if not self._spans:
            self.inert_reason = "no_spans"
        elif len(self._spans) == 1:
            self.inert_reason = "single_span"
        elif not self.required:
            self.inert_reason = "no_servable_spans"

    # -- span bookkeeping ---------------------------------------------------

    @property
    def token_ids(self):
        return self._ids

    @property
    def spans(self) -> list[tuple[int, int]]:
        return list(self._spans)

    @property
    def span_tokens(self) -> list[int]:
        return [end - start for start, end in self._spans]

    def span_rows(self, index: int) -> int | None:
        """Sidecar rows this span hashes to, or None if geometry is unknown."""

        if self._rows_per_token is None:
            return None
        start, end = self._spans[index]
        return (end - start) * self._rows_per_token

    def span_is_servable(self, index: int) -> bool:
        """Whether the worker is DESIGNED to prepare this span.

        Mirrors ``_SidecarGather.prepare_rows_np`` exactly: it declines when
        the gather's unique rows are ``<= _HOT_PATH_MAX_ROWS``, so servable
        is strictly greater.  ``tokens * ngram_heads`` is an upper bound on
        those unique rows, so this can only ever over-require, never exempt a
        span the worker would in fact have served.
        """

        rows = self.span_rows(index)
        if rows is None:
            return True
        return rows > self._min_servable_rows

    def span_index_of(self, chunk_ids) -> int | None:
        """Index of the span whose tokens equal ``chunk_ids``, else None.

        Prefill chunks arrive in order, so the hint hits first every time on
        a plain chunked prefill; the scan behind it is what keeps the answer
        correct for a restored suffix or a re-issued chunk.
        """

        import numpy as np

        ids = np.asarray(chunk_ids, dtype=np.int64).reshape(-1)
        rows = int(ids.shape[0])
        order = range(len(self._spans))
        if self._hint is not None:
            order = [
                *range(self._hint, len(self._spans)),
                *range(0, self._hint),
            ]
        for index in order:
            start, end = self._spans[index]
            if end - start != rows:
                continue
            if np.array_equal(self._ids[start:end], ids):
                self._hint = index + 1
                return index
        return None

    def next_index(self, index: int) -> int | None:
        nxt = int(index) + 1
        return nxt if 0 <= nxt < len(self._spans) else None

    # -- one-slot lifecycle -------------------------------------------------

    def adopt_early(self, early: "EarlyFirstGather | None") -> bool:
        """Take the request-arrival first-chunk gather as this lane's slot 0.

        The early lane and the lookahead prepare the SAME span 0, so without
        this they would prepare it twice: the early payload would be dropped,
        the duplicate would occupy the worker, and the receipt would show a
        hit that the head start did not produce.  Adoption instead installs
        the in-flight future as the pending slot, so `take(0)` -- and with it
        every hit/miss/ineligible/engagement statement below -- is unchanged.

        Refused, counted, and harmless whenever the plan the prefill actually
        chose is not the plan the early lane guessed at request arrival (a
        session-bank restore that prefills a suffix, a mandatory stable-prefix
        edge, a different chunk width): the lane then submits span 0 itself.
        """

        if self._closed or early is None or self._pending is not None:
            return False
        if not self._spans:
            return False
        if tuple(self._spans[0]) != tuple(early.span):
            count("early_span_mismatch")
            return False
        if not early.matches_plan(self._ids):
            count("early_plan_mismatch")
            return False
        self._pending = (0, early.adopt())
        self._early = early
        self.submits += 1
        count("early_adopted")
        return True

    def submit(self, index: int) -> None:
        """Prepare span ``index`` on the worker unless a slot is already held."""

        if self._closed or index is None:
            return
        if self._pending is not None:
            return
        if not (0 <= index < len(self._spans)):
            return
        start, end = self._spans[index]
        self._pending = (index, self._submit(self._prepare, start, end))
        self.submits += 1
        count("submitted")

    def take(self, index: int):
        """Result prepared for ``index``, or None; the slot is always released."""

        if int(index) == 0 and self._early is not None:
            self._early.note_needed()
        pending = self._pending
        if pending is None:
            self.misses += 1
            self._outcomes[int(index)] = "miss_empty"
            count("miss_empty")
            return None
        self._pending = None
        pending_index, future = pending
        if pending_index != index:
            self._discard(future)
            self.misses += 1
            self._outcomes[int(index)] = "miss_wrong_span"
            count("miss_wrong_span")
            return None
        try:
            prepared = future.result()
        except BaseException:
            count("worker_error")
            raise
        if self._early is not None and pending_index == 0:
            self._early._publish()
            self._early.note_outcome(
                "adopted_hit" if prepared is not None else "adopted_ineligible"
            )
        if prepared is None:
            # The worker ran and declined: these ids belong to the owner's
            # hot-row LRU, which is owner-thread-only state.  That is a
            # property of the span, not evidence of an inert lane -- counted
            # as a miss (the owner does pay the ordinary gather) but kept
            # apart from the engagement verdict.
            self.misses += 1
            self.ineligible += 1
            small = not self.span_is_servable(int(index))
            outcome = "ineligible_small" if small else "ineligible"
            if small:
                self.ineligible_small += 1
            self._outcomes[int(index)] = outcome
            count(f"miss_{outcome}")
            return None
        self.hits += 1
        self._outcomes[int(index)] = "hit"
        count("hit")
        return prepared

    def engagement(self) -> dict[str, int]:
        """Per-scope receipt: what this prefill's lookahead actually did."""

        return {
            "spans": len(self._spans),
            "submits": int(self.submits),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "ineligible": int(self.ineligible),
            "ineligible_small": int(self.ineligible_small),
            "required": len(self.required),
        }

    def verify_full_engagement(self) -> None:
        """Every span the worker was designed to serve must have hit it.

        An armed lane that quietly served zero chunks is the one outcome an
        A/B cannot detect afterwards -- it measures the control while wearing
        the candidate's label, which is exactly what happened on 2026-09-01
        (``lookahead_batches`` 0, ``prefill_lookahead`` {}, a 2 s "regression"
        that was arm-position drift). So this raises instead.

        What is NOT that failure, and must not raise, is a span the sidecar
        declines by construction: a gather at or below
        ``_HOT_PATH_MAX_ROWS`` belongs to the owner-thread hot-row LRU, which
        the worker is forbidden to touch.  Those spans are exempt from the
        requirement, not tolerated within it -- the difference is that an
        empty slot, a wrong span, or a REQUIRED span that missed or was
        declined still raises with the offending indices named, whatever the
        rest of the prefill did.
        """

        if not self.armed:
            return
        unserved = [
            index for index in self.required
            if self._outcomes.get(index) != "hit"
        ]
        if not unserved:
            return
        raise RuntimeError(
            f"{ENV_FLAG}=1 did not engage on every prefill chunk: "
            f"{self.engagement()}; spans the worker was designed to serve "
            f"but did not: "
            f"{[(index, self._outcomes.get(index, 'never_taken')) for index in unserved]}"
            f", span sizes {self.span_tokens}. The lane is armed but (partly) "
            "inert; refusing to report a measurement that did not run the "
            "candidate."
        )

    def _discard(self, future) -> None:
        future.cancel()
        try:
            future.result()
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        pending, self._pending = self._pending, None
        if pending is not None:
            self._discard(pending[1])
            count("discarded_on_close")
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


@contextlib.contextmanager
def prefill_lookahead_scope(lookahead: "PrefillLookahead | None"):
    """Make ``lookahead`` the request-scoped instance for this prefill."""

    if lookahead is None:
        yield None
        return
    if not lookahead.armed:
        # Nothing to look ahead from (one chunk), or nothing the worker is
        # designed to serve (every span at or below the sidecar's hot-row
        # threshold).  The construction-bound checks in the model builder
        # already ran -- an armed flag on a model that cannot serve the lane
        # still raises, whatever the prompt length -- so what is skipped here
        # is only the worker, which would pay a thread handoff for zero
        # overlap and then report the hot-LRU decline as a non-engagement.
        _LAST_SCOPE.update(
            {
                "armed": False,
                "reason": lookahead.inert_reason,
                "spans": len(lookahead.spans),
                "required": len(lookahead.required),
                "span_tokens": lookahead.span_tokens,
            }
        )
        count(f"scope_skipped_{lookahead.inert_reason}")
        lookahead.close()
        yield None
        return
    _LAST_SCOPE.update(
        {
            "armed": True,
            "reason": None,
            "spans": len(lookahead.spans),
            "required": len(lookahead.required),
            "span_tokens": lookahead.span_tokens,
        }
    )
    token = _ACTIVE.set(lookahead)
    # Start the first chunk's preparation before the caller does anything
    # else: chunk 0 is the one with the largest measured stall (450 ms vs
    # 157-346 ms, first-touch on top of the gather) and nothing else has
    # claimed the worker yet.  Submitting it HERE still leaves it exposed --
    # `stage()` blocks on it microseconds later (0.627 s on chunk 1 against
    # 0.0006 s on chunks 2-4, the 16K prefill-stack receipt) -- so when
    # MTPLX_FABLE_PLE_FIRST_GATHER_EARLY armed the same span at request
    # arrival, adopt that in-flight future instead of preparing it twice.
    if not lookahead.adopt_early(active_early_first_gather()):
        lookahead.submit(0)
    clean = False
    try:
        yield lookahead
        clean = True
    finally:
        _ACTIVE.reset(token)
        lookahead.close()
        # Only on the clean path: an aborted prefill legitimately consumed
        # fewer chunks than it planned, and masking its exception with an
        # engagement complaint would hide the real failure.
        if clean:
            lookahead.verify_full_engagement()
