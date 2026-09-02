"""``MTPLX_FABLE_DRAFT_K20_PRESCATTER`` -- draft K20 support on the FR-Spec row.

Read ONCE at import, default OFF.  Flag-off, :func:`is_enabled` is False, no
plan is ever built, and the single call site in ``generation.py`` stays behind
a module-level constant, so the retained stock draft lane evaluates exactly the
expressions it evaluated before this module existed.

The problem
-----------
With ``MTPLX_FRSPEC_DRAFT=1`` the draft head is
``frspec_draft._FullVocabDraftHead``: it runs the pruned q8 head over the
65,536 frequency-ranked rows and then *scatters* those logits into a
248,320-wide row padded with ``-1e30``
(``mtplx/frspec_draft.py`` ``_FullVocabDraftHead.__call__``).  The stock draft
reader then builds its K20 support over that padded row
(``generation._fixed_width_draft_reader`` -> ``_sample_draft_from_logits`` ->
``_sample_from_logits`` -> ``_distribution_from_mlx_logits`` ->
``fast_sampling.sparse_distribution_from_mlx_logits`` ->
``fast_sampling._device_serial_support_arrays``), which costs, per draft step:

* the scatter itself: one ``mx.full`` + one ``put_along_axis`` over 248,320
  float32 lanes (~0.99 MB written, and the ``mx.full`` fill before it),
* one ``argpartition`` to an 80-candidate superset over 248,320 lanes,
* one full-vocabulary ``logsumexp`` over 248,320 lanes,

when 73.6% of those lanes are a constant sentinel that cannot win any of it.

What this module does
---------------------
It builds the SAME K20 support from the 65,536-row head output, maps the
selected LOCAL rows back to real token ids through the ranked id table, and
hands the stock draft reader's caller the same ``(token, SparseDistribution)``
pair.  Nothing evaluates the scattered array, and because MLX is lazy the
scatter graph node is therefore never executed -- ``put_along_axis`` is built
and dropped, not run.

Exactness argument
------------------
Write ``ids`` for the ranked table (``_mtplx_frspec_ids``), ``sub`` for the
compact head row, ``dense`` for the scattered row, ``T`` for the draft
temperature and ``scaled = row.astype(float32) * (1/T)``.

1. **The table is strictly ascending.**  ``mtplx/data/qwen38_code_ranked_64k.json``
   is row-sorted (``frspec_draft`` module docstring: "row-sorted for efficient
   gathering"); :func:`claim_draft_route` re-proves it per request and refuses
   otherwise.  So ``local -> ids[local]`` is a strictly increasing bijection
   from ``[0, 65536)`` onto the occupied lanes of ``dense``, and it preserves
   BOTH the relative order of the values in the row and every ``id asc``
   tie-break.

2. **No sentinel can enter the support.**  The pad is ``-1e30`` in the head's
   own dtype; the row carries 65,536 real logits and the support is 20 wide,
   so the 80-candidate superset (``m = 4k = 80 <= 65536``) is entirely real on
   both rows, and the multiset of the top-80 VALUES is identical.  The stock
   builder's ``spill`` condition (``nanmin(cand) >= cutoff``) is a function of
   those 80 sorted values alone, so it fires on both rows or neither.

3. **Selection is identical.**  ``argpartition`` leaves the superset
   unordered, and the stock builder then imposes a total order with
   ``np.lexsort((cand_ids, -cand_val_rows))`` -- value desc, id asc.  This
   module maps local rows to real ids BEFORE that lexsort, so it sorts the
   same 80 (value, real id) pairs and takes the same top 20 in the same order.
   The spill fallback re-derives with ``_deterministic_mlx_top_k_support``,
   whose ``higher``/``tied``/``cumsum`` rank is likewise order-preserving under
   a strictly increasing id map, and its ids are mapped before ITS lexsort.
   The greedy branch is ``argmax``, whose lowest-index tie-break maps to the
   lowest real id for the same reason.

4. **The normaliser is value-identical.**  The only quantity the compact row
   cannot reproduce structurally is ``log_total = logsumexp(scaled)`` over the
   full row.  ``logsumexp`` is ``M + log(sum(exp(x - M)))`` with
   ``M = max(scaled)`` a real logit (tens, not -1e30).  A sentinel lane
   contributes ``exp(-1e30/T - M)``, and float32 ``exp`` underflows to
   *exactly* ``+0.0`` below about ``-103.97``; ``-1.67e30`` is 28 orders of
   magnitude past that, in float32 AND after a bfloat16 head output widens to
   float32 (bfloat16 holds ``-1e30`` as a finite ``-9.9964e29``; it is not an
   inf and it is not a NaN).  Since ``x + 0.0 == x`` exactly for every finite
   ``x``, every sentinel lane contributes exactly nothing to the sum, and the
   two rows' logsumexps are equal *as real numbers computed from the same
   float32 terms in the same order*.

   What this does NOT prove is bit-identity across the two reduction SHAPES.
   Floating-point addition is not associative, and a 248,320-lane row reduce
   partitions the 65,536 nonzero terms across threads/blocks differently than a
   65,536-lane one does; a residual of a few ULP in ``log_total`` is admissible
   in principle.  Measured on the CPU stream (which this repo's tests can run)
   the two are bit-identical -- ``tests/test_fable_draft_k20_prescatter.py``
   pins that -- and ``scripts/fable/micro_draft_k20.py`` re-measures it on the
   Metal stream and prints the differing-row and ULP counters rather than
   assuming.  A residual ULP would scale every q entry by ``1 +- 2**-24`` and
   could only change behaviour by flipping a knife-edge top-p ``cumulative_before``
   comparison; it cannot change the SUPPORT (point 3 is exact and independent
   of ``log_total``).

5. **No consumer needs the dense draft row.**  Under this route the only
   readers of ``draft_logits`` in ``generation.generate_mtpk``'s draft loop are
   ``int(draft_logits.shape[-1])`` (a static shape, forces no evaluation) and
   ``_tree_nbytes(draft_logits)`` under ``trace.enabled`` (also static).  Every
   reader that would materialise the row -- ``_draft_confidence_metrics`` /
   ``_top2_margin`` (``draft_margin_threshold`` or the policy metrics),
   ``_greedy_draft_token_and_metrics`` (``combine_greedy_draft_read``), the
   ``_draft_conf_needed`` block, the ``mtp_topk_reranker``, the adapter
   ensemble, the correction cache, the A3B target-prefix route, and the
   ``MTPLX_FRSPEC_LEGACY`` local-id remap -- is a construction-time REFUSAL in
   :func:`claim_draft_route`, not a fallback.

Anything this lane does not implement raises
:class:`DraftK20PrescatterIneligible` at construction.  An armed flag that
quietly ran the stock selector would make every receipt a lie about which
selector produced it.

NO device work happens at import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .sampling import SamplerConfig, SparseDistribution, sample_from_distribution


_ENV_VAR = "MTPLX_FABLE_DRAFT_K20_PRESCATTER"

#: The only ranked-table width this route admits.  The built-in artifact
#: ``mtplx/data/qwen38_code_ranked_64k.json`` is exactly this wide; a different
#: width means a different (unproven) artifact, so the claim refuses instead of
#: generalising.
FRSPEC_ROWS = 65_536

#: ``_device_serial_support_arrays``'s candidate-superset multiplier.  Mirrored
#: here only to prove ``m <= FRSPEC_ROWS`` at claim time.
_SUPERSET_MULTIPLIER = 4


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


#: Read exactly once, at import.
_ENABLED = _env_truthy(_ENV_VAR)


def is_enabled() -> bool:
    """True when ``MTPLX_FABLE_DRAFT_K20_PRESCATTER`` was set at import."""

    return _ENABLED


def _configure_for_test(enabled: bool) -> None:
    """Flip the import-time gate (tests only)."""

    global _ENABLED
    _ENABLED = bool(enabled)


class DraftK20PrescatterIneligible(RuntimeError):
    """The armed flag met a request this lane does not implement.

    Raised at construction, never mid-decode.  There is no silent fallback.
    """


# ---------------------------------------------------------------------------
# Construction-time plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftK20PrescatterPlan:
    """One request's bound pre-scatter draft route."""

    head: Any
    """The live ``frspec_draft._FullVocabDraftHead``."""

    ids_np: np.ndarray
    """The ranked table as int64, strictly ascending, shape ``[rows]``."""

    rows: int
    """Ranked-table width (the pruned head's output domain)."""

    vocab_rows: int
    """Full vocabulary width -- the domain every emitted distribution spans."""

    top_k: int
    temperature: float
    top_p: float

    def to_dict(self) -> dict[str, object]:
        """The receipt this route writes."""

        return {
            "installed": True,
            "rows": int(self.rows),
            "vocab_rows": int(self.vocab_rows),
            "top_k": int(self.top_k),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
        }


def _refuse(reason: str) -> None:
    raise DraftK20PrescatterIneligible(
        f"MTPLX_FABLE_DRAFT_K20_PRESCATTER: {reason}"
    )


def claim_draft_route(
    rt: Any,
    *,
    draft_sampler: SamplerConfig,
    draft_core: str,
    target_prefix_verify: bool,
    a3b_target_prefix_route: Any,
    pr391_route: Any,
    device_k20_route: Any,
    frspec_legacy_ids: Any,
    adaptive_width_policy: Any,
    combine_greedy_draft_read: bool,
    draft_confidence_needed: bool,
    draft_margin_threshold: float | None,
    wants_policy_metrics: bool,
    correction_cache_enabled: bool,
    adapter_ensemble_q: bool,
    mtp_topk_reranker: Any,
    relaxed_draft_ties: bool,
    penalties_active: bool,
    steer_active: bool,
) -> DraftK20PrescatterPlan | None:
    """Bind the pre-scatter draft route to one generation construction.

    Returns ``None`` when the flag is off.  Raises
    :class:`DraftK20PrescatterIneligible` when the flag is ON and the request
    is not one this route can serve bit-for-bit.
    """

    if not _ENABLED:
        return None

    text = getattr(getattr(rt, "model", None), "language_model", None)
    if text is None:
        text = getattr(rt, "model", None)
    head = getattr(text, "_mtplx_frspec_draft_head", None)
    if head is None:
        _refuse("no FR-Spec draft head is installed (need MTPLX_FRSPEC_DRAFT=1)")
    live = getattr(text, "_mtplx_draft_lm_head", None)
    if live is not head:
        _refuse(
            "the FR-Spec full-vocabulary head is not the live draft head "
            f"(live={type(live).__name__})"
        )
    ids = getattr(head, "_ids", None)
    if ids is None:
        _refuse("the FR-Spec head carries no ranked id table")
    if not hasattr(head, "arm_prescatter_capture"):
        _refuse(
            "the live draft head has no prescatter capture surface "
            f"({type(head).__name__})"
        )
    ids_np = np.asarray(ids, dtype=np.int64).reshape(-1)
    rows = int(ids_np.shape[0])
    if rows != FRSPEC_ROWS:
        _refuse(
            f"ranked table is {rows} rows; this route is proven only at "
            f"{FRSPEC_ROWS}"
        )
    if not bool(np.all(ids_np[1:] > ids_np[:-1])):
        _refuse(
            "ranked id table is not strictly ascending; the local->real id "
            "map must be monotone for the tie-break contract to hold"
        )
    vocab_rows = int(getattr(head, "_vocab_rows", 0))
    if vocab_rows <= rows or int(ids_np[-1]) >= vocab_rows:
        _refuse(f"full vocabulary width {vocab_rows} does not admit the table")

    top_k = int(draft_sampler.top_k)
    if top_k <= 0:
        _refuse("this route requires a top-k draft sampler")
    superset = min(max(_SUPERSET_MULTIPLIER * top_k, top_k), rows)
    if superset > rows:  # pragma: no cover - unreachable given the min() above
        _refuse(f"candidate superset {superset} exceeds the ranked table")
    if (
        float(draft_sampler.presence_penalty) != 0.0
        or float(draft_sampler.frequency_penalty) != 0.0
    ):
        _refuse("draft sampler penalties index by real token id")
    if penalties_active or steer_active:
        _refuse("steering/penalty overlays index by real token id")
    if str(draft_core) != "stock":
        _refuse(f"this route requires the stock draft selector (got {draft_core!r})")
    if relaxed_draft_ties:
        _refuse("MTPLX_QWEN4_RELAXED_DRAFT_TIES installs a different builder")
    if frspec_legacy_ids is not None:
        _refuse("MTPLX_FRSPEC_LEGACY already remaps local draft ids")
    if device_k20_route is not None:
        _refuse("MTPLX_FABLE_DEVICE_K20 owns the draft selector")
    if pr391_route is not None:
        _refuse("the PR391 float32 D3 route owns the draft chain")
    if a3b_target_prefix_route is not None or target_prefix_verify:
        _refuse("target-prefix verification samples drafts on device")
    if adaptive_width_policy is not None:
        _refuse("adaptive-width readers own the draft read")
    if combine_greedy_draft_read:
        _refuse("the joint greedy confidence read materialises the dense row")
    if draft_confidence_needed:
        _refuse("draft-confidence tracing materialises the dense row")
    if draft_margin_threshold is not None or wants_policy_metrics:
        _refuse("draft confidence metrics materialise the dense row")
    if correction_cache_enabled:
        _refuse("the online/prompt correction cache bypasses the draft read")
    if adapter_ensemble_q:
        _refuse("the adapter ensemble reads two dense draft rows")
    if mtp_topk_reranker is not None:
        _refuse("the top-k reranker reads the dense draft row")

    head.arm_prescatter_capture(True)
    return DraftK20PrescatterPlan(
        head=head,
        ids_np=ids_np,
        rows=rows,
        vocab_rows=vocab_rows,
        top_k=top_k,
        temperature=float(draft_sampler.temperature),
        top_p=float(draft_sampler.top_p),
    )


def release_draft_route(plan: DraftK20PrescatterPlan | None) -> None:
    """Disarm the head capture and drop the stashed row."""

    if plan is None:
        return
    plan.head.arm_prescatter_capture(False)


# ---------------------------------------------------------------------------
# Per-step read
# ---------------------------------------------------------------------------


def take_compact_row(
    plan: DraftK20PrescatterPlan,
    draft_logits: Any,
) -> Any:
    """Return the 65,536-wide pre-scatter row behind ``draft_logits``.

    Identity-checked against the array the head returned for this very call,
    then consumed, so a stale or mismatched stash raises instead of silently
    scoring the wrong step.
    """

    stashed = plan.head.take_prescatter_row(draft_logits)
    if stashed is None:
        raise DraftK20PrescatterIneligible(
            "the FR-Spec head did not capture a pre-scatter row for this "
            "draft step (the live draft head changed mid-request)"
        )
    row = stashed.reshape(-1)
    if int(row.shape[0]) != plan.rows:
        raise DraftK20PrescatterIneligible(
            f"pre-scatter row is {int(row.shape[0])} wide, expected {plan.rows}"
        )
    return row


def prescatter_serial_support_arrays(
    plan: DraftK20PrescatterPlan,
    compact_row: Any,
    config: SamplerConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """``fast_sampling._device_serial_support_arrays`` on the compact row.

    Every arithmetic step is the shipped builder's, in the shipped order and
    the shipped dtypes; the only change is that ``argpartition`` /
    ``logsumexp`` run over ``plan.rows`` lanes instead of ``plan.vocab_rows``,
    and the selected LOCAL rows are mapped to real token ids through the ranked
    table before the deterministic ``lexsort``.  Returns
    ``(token_rows [N,k] int64, prob_rows [N,k] float64, vocab_size)`` with
    ``vocab_size`` the FULL vocabulary, because that is the domain the emitted
    ``SparseDistribution`` spans.
    """

    import mlx.core as mx

    from .fast_sampling import _fixed_top_k_support

    rows = compact_row.reshape(-1, compact_row.shape[-1]).astype(mx.float32)
    row_width = int(rows.shape[-1])
    if row_width != plan.rows:
        raise DraftK20PrescatterIneligible(
            f"pre-scatter row is {row_width} wide, expected {plan.rows}"
        )
    ids_np = plan.ids_np
    vocab_size = int(plan.vocab_rows)
    k = min(int(config.top_k), row_width)
    scaled = rows * (1.0 / float(config.temperature))

    m = min(max(_SUPERSET_MULTIPLIER * k, k), row_width)
    cand_idx = mx.argpartition(-scaled, kth=m - 1, axis=-1)[:, :m]
    cand_vals = mx.take_along_axis(scaled, cand_idx, axis=-1)
    top_p_active = 0.0 < float(config.top_p) < 1.0
    if top_p_active:
        log_total = mx.logsumexp(scaled, axis=-1, keepdims=True)
        cand_probs = mx.exp(cand_vals - log_total)
        mx.eval(cand_idx, cand_vals, cand_probs)
        cand_prob_rows = np.asarray(cand_probs, dtype=np.float64)
    else:
        mx.eval(cand_idx, cand_vals)
        cand_prob_rows = None
    # LOCAL rows -> real token ids.  Done here, before the lexsort, so the
    # deterministic (value desc, id asc) order is imposed on the same pairs the
    # full-vocabulary builder would have sorted.
    cand_ids = ids_np[np.asarray(cand_idx, dtype=np.int64)]
    cand_val_rows = np.asarray(cand_vals, dtype=np.float32)

    order = np.lexsort((cand_ids, -cand_val_rows), axis=1)
    cand_ids = np.take_along_axis(cand_ids, order, axis=1)
    cand_val_rows = np.take_along_axis(cand_val_rows, order, axis=1)
    if cand_prob_rows is not None:
        cand_prob_rows = np.take_along_axis(cand_prob_rows, order, axis=1)
    token_rows = cand_ids[:, :k]
    if m > k:
        cutoff = cand_val_rows[:, k - 1]
        spill = np.nanmin(cand_val_rows, axis=1) >= cutoff
    else:
        spill = np.zeros(cand_ids.shape[0], dtype=bool)

    if top_p_active:
        prob_rows = cand_prob_rows[:, :k].copy()
        cumulative_before = np.concatenate(
            (
                np.zeros((prob_rows.shape[0], 1), dtype=np.float64),
                np.cumsum(prob_rows[:, :-1], axis=1),
            ),
            axis=1,
        )
        prob_rows = np.where(
            cumulative_before < float(config.top_p), prob_rows, 0.0
        )
    else:
        vals64 = cand_val_rows[:, :k].astype(np.float64)
        vals64 -= np.max(vals64, axis=1, keepdims=True)
        prob_rows = np.exp(vals64)
        prob_rows /= np.sum(prob_rows, axis=1, keepdims=True)

    if spill.any():
        _, exact_idx, exact_vals = _fixed_top_k_support(scaled, top_k=k)
        if top_p_active:
            exact_probs = mx.exp(
                exact_vals - mx.logsumexp(scaled, axis=-1, keepdims=True)
            )
        else:
            exact_probs = mx.softmax(exact_vals, axis=-1)
        mx.eval(exact_idx, exact_probs)
        exact_ids = ids_np[np.asarray(exact_idx, dtype=np.int64)]
        exact_prob_rows = np.asarray(exact_probs, dtype=np.float64)
        if top_p_active:
            ex_order = np.lexsort((exact_ids, -exact_prob_rows), axis=1)
            exact_ids = np.take_along_axis(exact_ids, ex_order, axis=1)
            exact_prob_rows = np.take_along_axis(exact_prob_rows, ex_order, axis=1)
            ex_before = np.concatenate(
                (
                    np.zeros((exact_prob_rows.shape[0], 1), dtype=np.float64),
                    np.cumsum(exact_prob_rows[:, :-1], axis=1),
                ),
                axis=1,
            )
            exact_prob_rows = np.where(
                ex_before < float(config.top_p), exact_prob_rows, 0.0
            )
        token_rows = np.where(spill[:, None], exact_ids, token_rows)
        prob_rows = np.where(spill[:, None], exact_prob_rows, prob_rows)

    return token_rows, prob_rows, vocab_size


def prescatter_sparse_distribution(
    plan: DraftK20PrescatterPlan,
    compact_row: Any,
    config: SamplerConfig,
) -> SparseDistribution:
    """``sparse_distribution_from_mlx_logits`` on the compact row."""

    import mlx.core as mx

    from .fast_sampling import _host_sparse_distribution, _serial_row_distribution

    row = compact_row.reshape(-1).astype(mx.float32)
    token_rows, prob_rows, vocab_size = prescatter_serial_support_arrays(
        plan, row, config
    )
    dist = _serial_row_distribution(token_rows[0], prob_rows[0], vocab_size)
    if dist is not None:
        return dist
    # Non-finite mass (NaN/inf logits): the shipped one-hot/dense host
    # reference, run on the compact row and mapped back.  ``apply_top_p_top_k``
    # ranks by probability then id, so the strictly ascending table preserves
    # its answer exactly.
    mx.eval(row)
    local = _host_sparse_distribution(np.asarray(row, dtype=np.float32), config)
    return SparseDistribution(
        plan.ids_np[np.asarray(local.token_ids, dtype=np.int64)],
        local.probs,
        vocab_size,
    )


def read_draft(
    plan: DraftK20PrescatterPlan,
    draft_logits: Any,
    config: SamplerConfig,
    rng: Any,
    *,
    need_distribution: bool,
) -> tuple[int, SparseDistribution | None]:
    """The stock draft read (``_sample_draft_from_logits``), pre-scatter.

    Draws exactly the same one ``rng`` value per draft step, in the same order,
    from the same stream as the stock lane: ``sample_from_distribution`` ->
    ``rng.choice(ids, p=probs)``.
    """

    import mlx.core as mx

    row = take_compact_row(plan, draft_logits)
    if float(config.temperature) <= 0:
        local = int(mx.argmax(row, axis=-1).item())
        token = int(plan.ids_np[local])
        if not need_distribution:
            return token, None
        return token, SparseDistribution.one_hot(token, int(plan.vocab_rows))
    # ``need_distribution`` is deliberately ignored here: the shipped
    # ``_sample_draft_from_logits`` also ignores it on the sampled branch (it
    # tail-calls ``_sample_from_logits``, which always returns the row it
    # sampled from), and the accept loop is fed that same row.
    del need_distribution
    distribution = prescatter_sparse_distribution(plan, row, config)
    return sample_from_distribution(distribution, rng), distribution
