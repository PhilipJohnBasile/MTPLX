"""Shape-stable QSA replay metadata for speculative MTP cycles.

SGLang's DSA multi-step backend computes request/page-table metadata once and
copies it into each CUDA-graph backend.  Qwen4Exp in MTPLX has no page table
and no per-draft-step backend: it serves one sequence through one positional
``QSACache``.  The metadata worth staging here is consequently smaller:

* the raw/pooled backing capacities needed by a captured draft or verify
  window, and
* the exact frontier to retain when a persistent MTP-history cache is repaired
  after speculative drafting.

This module deliberately has no MLX import.  The plans are ordinary immutable
Python values, so the generation layer can compute them outside a compiled
region and a cache implementation can reserve the resulting buckets before a
trace.  Keeping the policy independent from the array implementation also
makes all boundary and rollback rules testable without constructing a model.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

QSA_MTP_PRECOMPUTE_ENV = "MTPLX_QSA_MTP_PRECOMPUTE"


def qsa_mtp_precompute_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve the Phase-3 gate; unset and unknown values are safely off."""

    source = os.environ if environ is None else environ
    return str(source.get(QSA_MTP_PRECOMPUTE_ENV, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def qsa_mtp_outer_device_core_supported(
    caches: Sequence[Any] | None,
) -> bool:
    """Reject QSA caches from MTPLX's outer stateful draft compilation.

    The QSA indexer's own compiled graph is safe because it threads full
    power-of-two backings and moving frontiers as explicit inputs/outputs.
    MTPLX's older whole-draft ``device``/``device-d2`` cores instead discover
    cache leaves through ``cache.state``.  The v2.10 ``QSACache.state`` surface
    exposes logical ``:offset``/``:pooled_len`` slices and owns a Python integer
    offset, so its outer signature changes every cycle and that offset cannot be
    replayed dynamically.  A capacity reservation alone cannot make that
    boundary safe.

    ``reserve_indexer_capacity`` is the deliberately narrow, model-independent
    marker for the current QSA cache.  Future support should replace this guard
    only after a full-backing, tensor-frontier outer compile adapter exists.
    Ordinary cache stacks remain eligible byte-for-byte.
    """

    if caches is None:
        return True
    return not any(
        callable(getattr(cache, "reserve_indexer_capacity", None)) for cache in caches
    )


def _nonnegative(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0; got {value}")
    return value


def _positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0; got {value}")
    return value


def qsa_indexer_is_bucket_capacity(capacity: int, *, minimum: int = 256) -> bool:
    """Return whether ``capacity`` is an allowed compiled-indexer bucket."""

    capacity = _nonnegative("capacity", capacity)
    minimum = _positive("minimum", minimum)
    if minimum & (minimum - 1):
        raise ValueError(f"minimum must be a power of two; got {minimum}")
    return capacity == 0 or (capacity >= minimum and (capacity & (capacity - 1)) == 0)


def qsa_indexer_capacity_bucket(required: int, *, minimum: int = 256) -> int:
    """Promote a required extent to a bounded compiled-indexer capacity."""

    required = _nonnegative("required", required)
    minimum = _positive("minimum", minimum)
    if minimum & (minimum - 1):
        raise ValueError(f"minimum must be a power of two; got {minimum}")
    if required == 0:
        return 0
    return max(minimum, 1 << (required - 1).bit_length())


@dataclass(frozen=True, slots=True)
class QSAReplayCapacity:
    """Backing geometry for one shape-stable QSA replay window.

    ``end_offset`` is the token frontier after the widest forward that may run
    in the replay.  ``complete_blocks`` follows QSA's exact pooling rule:
    incomplete tails are not part of the logical pooled-key frontier.  The
    physical pooled capacity can include one extra staging row for the
    compiled core's fixed ceil(window/ratio) preparation window.

    The two capacity fields never shrink an existing allocation.  They are
    independently promoted to power-of-two graph buckets (minimum
    ``allocation_step``), even when the live v2.10 cache arrived with an
    additive 256-step capacity such as 768.  This bounds the graph bank instead
    of creating one compiled signature for every additive cache growth.
    """

    start_offset: int
    window_tokens: int
    end_offset: int
    complete_blocks: int
    raw_capacity: int
    pooled_capacity: int
    compress_ratio: int
    allocation_step: int

    @property
    def graph_key(self) -> tuple[int, int]:
        """The backing-shape portion of an ``mx.compile`` graph-bank key."""

        return (self.raw_capacity, self.pooled_capacity)


def precompute_qsa_replay_capacity(
    *,
    start_offset: int,
    window_tokens: int,
    compress_ratio: int,
    allocation_step: int = 256,
    current_raw_capacity: int = 0,
    current_pooled_capacity: int = 0,
) -> QSAReplayCapacity:
    """Compute the backing buckets needed before a captured QSA forward.

    The live v2.10 cache grows additively; replay promotion deliberately uses
    power-of-two buckets so the number of compiled signatures is logarithmic.
    A caller can compute the plan on the host, reserve once, and keep every
    replay inside the same backing shapes.
    """

    start = _nonnegative("start_offset", start_offset)
    width = _nonnegative("window_tokens", window_tokens)
    ratio = _positive("compress_ratio", compress_ratio)
    step = _positive("allocation_step", allocation_step)
    if step & (step - 1):
        raise ValueError(
            f"allocation_step must be a power of two; got {allocation_step}"
        )
    current_raw = _nonnegative("current_raw_capacity", current_raw_capacity)
    current_pooled = _nonnegative("current_pooled_capacity", current_pooled_capacity)

    end = start + width
    blocks = end // ratio
    # The compiled core pools one fixed ceil(window/ratio)-block staging
    # window.  An unaligned first prefill can need one physical row beyond
    # the logical complete-block frontier (S=1025, ratio=4: 257 staging rows
    # versus 256 complete blocks).  Stage that physical requirement here so
    # Phase-3 precompute actually prevents a bucket transition in the replay.
    staging_blocks = (width + ratio - 1) // ratio
    staging_tokens = staging_blocks * ratio
    raw_capacity = qsa_indexer_capacity_bucket(
        max(current_raw, end, staging_tokens), minimum=step
    )
    pooled_capacity = qsa_indexer_capacity_bucket(
        max(current_pooled, blocks, staging_blocks), minimum=step
    )
    return QSAReplayCapacity(
        start_offset=start,
        window_tokens=width,
        end_offset=end,
        complete_blocks=blocks,
        raw_capacity=raw_capacity,
        pooled_capacity=pooled_capacity,
        compress_ratio=ratio,
        allocation_step=step,
    )


@runtime_checkable
class QSAReplayReservable(Protocol):
    """Cache hook consumed by :func:`stage_qsa_replay_capacity`."""

    def reserve_indexer_capacity(
        self, *, raw_capacity: int, pooled_capacity: int
    ) -> None: ...


def stage_qsa_replay_capacity(cache: Any, plan: QSAReplayCapacity) -> bool:
    """Apply a precomputed capacity plan when a cache exposes the hook.

    Returning ``False`` is an intentional compatibility path for cache types
    that do not own a QSA indexer.  A callable hook is allowed to raise: a
    failed reservation must not be hidden immediately before graph capture.
    """

    reserve = getattr(cache, "reserve_indexer_capacity", None)
    if not callable(reserve):
        return False
    reserve(
        raw_capacity=plan.raw_capacity,
        pooled_capacity=plan.pooled_capacity,
    )
    return True


def precompute_and_stage_qsa_replay(
    cache: Any,
    *,
    window_tokens: int,
) -> QSAReplayCapacity | None:
    """Plan and reserve one duck-typed QSA cache, or ignore other caches.

    The reservation hook is checked first so ordinary KV/recurrent cache
    entries remain a zero-work compatibility path.  A QSA cache supplies the
    same ``offset``, ``ratio``, ``step``, ``raw_keys`` and ``pooled`` surface
    as v2.10's implementation.  Shape reads are host metadata only.
    """

    if not callable(getattr(cache, "reserve_indexer_capacity", None)):
        return None
    ratio = _positive("cache.ratio", cache.ratio)
    step = _positive("cache.step", getattr(cache, "step", 256))
    raw = getattr(cache, "raw_keys", None)
    pooled = getattr(cache, "pooled", None)
    current_raw = 0 if raw is None else int(raw.shape[1])
    current_pooled = 0 if pooled is None else int(pooled.shape[1])
    plan = precompute_qsa_replay_capacity(
        start_offset=int(cache.offset),
        window_tokens=window_tokens,
        compress_ratio=ratio,
        allocation_step=step,
        current_raw_capacity=current_raw,
        current_pooled_capacity=current_pooled,
    )
    stage_qsa_replay_capacity(cache, plan)
    return plan


def precompute_and_stage_qsa_replay_caches(
    caches: Sequence[Any],
    *,
    window_tokens: int,
) -> tuple[QSAReplayCapacity, ...]:
    """Reserve every QSA entry in a heterogeneous model cache list."""

    plans = []
    for cache in caches:
        plan = precompute_and_stage_qsa_replay(
            cache,
            window_tokens=window_tokens,
        )
        if plan is not None:
            plans.append(plan)
    return tuple(plans)


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class MTPIndexerReplayPlan:
    """Exact persistent-history repair plan for one speculative cycle.

    The first MTP draft row is special: it consumes the authoritative trunk
    hidden state and the cycle's primary token, so it is safe to retain.
    Every later row consumes recursively predicted MTP hidden state and must be
    overwritten from target-verify hidden rows even when its token was
    accepted.  If a non-MTP draft source (for example prompt lookup) skipped
    the first head forward, ``observed_offset == cycle_offset`` and no row may
    be retained.
    """

    cycle_offset: int
    observed_offset: int
    speculative_rows: int
    primary_staged: bool
    reusable_rows: int
    rollback_offset: int
    reappend_start: int

    def reappend_tokens(self, committed: Sequence[T]) -> tuple[T, ...]:
        """Return the committed suffix that still needs an MTP-head write."""

        return tuple(committed[self.reappend_start :])

    def authoritative_hidden_rows(self, committed_count: int) -> int:
        """Number of hidden input rows required for :meth:`reappend_tokens`.

        Each MTP history token consumes one corresponding hidden input row.
        When the primary is retained, the authoritative target-verify rows
        begin at the hidden state after that primary.  When it was not staged,
        the caller must prepend the pre-cycle trunk hidden row.
        """

        count = _nonnegative("committed_count", committed_count)
        if count < self.reappend_start:
            raise ValueError(
                "committed_count cannot be smaller than the retained prefix; "
                f"got {count} < {self.reappend_start}"
            )
        return count - self.reappend_start


def precompute_mtp_indexer_replay(
    *, cycle_offset: int, observed_offset: int
) -> MTPIndexerReplayPlan:
    """Precompute the exact rollback/reappend frontier after MTP drafting.

    ``observed_offset`` is read after the draft source has run.  An offset
    advance proves that the normal MTP head staged the authoritative primary
    row.  No advance means a substitute draft source skipped the head, so the
    primary must be included in the authoritative history append.
    """

    base = _nonnegative("cycle_offset", cycle_offset)
    observed = _nonnegative("observed_offset", observed_offset)
    if observed < base:
        raise ValueError(
            f"observed_offset cannot precede cycle_offset; got {observed} < {base}"
        )
    speculative_rows = observed - base
    reusable = 1 if speculative_rows else 0
    return MTPIndexerReplayPlan(
        cycle_offset=base,
        observed_offset=observed,
        speculative_rows=speculative_rows,
        primary_staged=bool(reusable),
        reusable_rows=reusable,
        rollback_offset=base + reusable,
        reappend_start=reusable,
    )


__all__ = [
    "QSA_MTP_PRECOMPUTE_ENV",
    "MTPIndexerReplayPlan",
    "QSAReplayCapacity",
    "QSAReplayReservable",
    "precompute_and_stage_qsa_replay",
    "precompute_and_stage_qsa_replay_caches",
    "precompute_mtp_indexer_replay",
    "precompute_qsa_replay_capacity",
    "qsa_indexer_capacity_bucket",
    "qsa_indexer_is_bucket_capacity",
    "qsa_mtp_outer_device_core_supported",
    "qsa_mtp_precompute_enabled",
    "stage_qsa_replay_capacity",
]
