#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# NO GPU.  Pure NumPy.  This script never imports mlx and never touches the
# GPU exclusive lock -- it replays logged K20 rows on the CPU.
# ---------------------------------------------------------------------------
"""Score the block-verification law (H §Option B) against the shipped law,
offline, on the K20 rows captured by ``MTPLX_FABLE_K20_LOG``.

Why offline, and which number to read
-------------------------------------
``H-tokens-per-window-design.md`` §1.4: ``Var(l) ~ 1.50`` at ``n ~ 385`` gives
``SE(E[l]) = 0.062``, i.e. **+-4.2% (1 sigma)**.  Both candidate laws move
``E[l]`` by a few percent, so a live A/B cannot resolve them.  A replay on
logged rows fixes the rows, which removes most of that; two estimators are
reported, and they are not interchangeable:

**``E[tok/win]`` -- the one to quote.**  ``1 + sum_d w_d``, where ``w_d`` is the
probability the window accepts through depth ``d`` given the rows and the
drafted tokens.  No uniform is consulted, so the accept-coin variance is
integrated out analytically; and the block law's ladder reports the closed-form
``E[w_d | x_{1:d}] = min(A_d, w_{d-1})`` rather than the realisation, which
integrates out the look-ahead draw as well.  What is left is the drafted-row
sampling, which is *identical* for both arms -- and at depth 1 the two ladders
are then provably equal entry for entry, which is the theorem (H §3.1: depth-1
acceptance is already saturated at ``min(1, rho_1)`` and no law can raise it).
The reported ``delta`` is the paired per-cycle difference with its own standard
error, and that interval is several times tighter than either arm's own.

**``replay`` -- the exactness proof.**  The full emission simulation, driven by
the four **logged PCG64 uniforms**, so it reproduces the token the device
would have emitted.  This is what compares each law's decision against the
device kernel's own logged output and what verifies the ``c = 1`` identity.
Its arm-to-arm difference is noisy (the two laws' coins land differently on
the same uniform), so read it for correctness, not for magnitude.

Two lanes, one loader
---------------------
``MTPLX_FABLE_K20_LOG`` writes one normalised schema from either of the two
lanes that make an accept decision, and this script loads both.  The only
place they differ is how far the rows have already been shaped, which
:func:`build_window` branches on and nothing else does:

``pr391_raw``
    Rows are the softfloat64 kernel's **raw** input -- top-20 ids, raw logits,
    full-vocabulary softmax.  Top-p 0.95 and the double renormalisation are the
    kernel's, so this script re-runs them (:func:`prepare_row`,
    :func:`prepare_batched_row`), mirroring
    ``pr391_softfloat64_verifier_decision.py`` line by line.
``stock_prepared``
    Rows are the stock native-MTP lane's host-side ``SparseDistribution`` /
    ``BatchedSparseDistributions`` objects, which are **already** shaped and
    renormalised (``sampling.py:122-176``).  Re-running the kernel's
    preparation would apply top-p a second time, so :func:`prepared_row` takes
    them as they are.  Two further consequences:

    * the correction id is sampled with ``rng.choice`` straight off the live
      generator, so it is not reproducible offline -- the replay self-check
      compares everything *except* the selected token;
    * an accept coin exists only for the depths the lane actually reached, so
      :func:`window_uniforms` fills the rest from a deterministic stream seeded
      by the window's own logged PCG64 state.  Both laws get the same stream.

What is replayed, and what is not
---------------------------------
Each logged cycle carries one verify window's prepared K20 rows and its
decision uniforms.  This script re-decides **that window**
under each law.  It does **not** re-run the model: under the block law a
window can accept a different number of tokens, which in production would
change every subsequent window's rows.  So the number reported here is a
**per-window counterfactual on the realised row distribution** -- the standard
and correct estimator for ``E[l]``, and the same framing H §3.4 uses for its
surrogate.  It is not a full-trajectory simulation and must not be sold as one.

The two laws
------------
**Current** -- ``mtplx/kernels/pr391_softfloat64_verifier_decision.py:220-345``,
mirrored here in float64::

    for d in 1..3:
        rho_d = p_d(x_d) / q_d(x_d)
        alpha_d = min(1, rho_d)                       # kernel line 280-287
        if u_d <= alpha_d: accept                     # kernel line 289, `<=`
        else: emit sample(normalise((p_d - q_d)+))    # kernel lines 296-306
    emit bonus ~ p_4                                  # kernel lines 335-342

**Block verification** -- H §3.2.  One clip moves: the running product is
clipped at 1 instead of each factor being clipped and then multiplied, and
depth ``d``'s accept coin is allowed to look at the depth ``d+1`` rows::

    c_0 = 1 ; w_0 = 1
    for d in 1..3:
        rho_d = p_d(x_d) / q_d(x_d)
        A_d   = min(1, c_{d-1} * rho_d)               # reach BUDGET
        if d < 3:
            base(y) = min(1, A_d * rho_{d+1}(y))      # y over the d+1 draft support
            lam_d   = solve_lambda:  sum_y q_{d+1}(y) * min(cap, base(y) + lam) = A_d
            w_d     = min(cap, base(x_{d+1}) + lam_d) # realised reach probability
        else:
            w_d     = A_d
        a_d = w_d / w_{d-1}                           # CONDITIONAL accept coin
        if u_d <= a_d: c_d = A_d ; continue
        emit sample(normalise((c_{d-1} * p_d - q_d)+))    # SCALED residual
        stop
    emit bonus ~ p_4

``lam_d`` is the water-filling level that keeps ``E_{x_{d+1}~q_{d+1}}[w_d]``
exactly at the budget ``A_d`` -- which is what preserves exactness: the
position-``d`` accept probability, averaged over the *next* drafted token,
still equals ``min(1, rho_d)``, so ``q_d(y) * P(accept | x_d = y) <= p_d(y)``
holds pointwise and the residual stays non-negative.  Within that budget the
mass is shifted toward the realisations whose next drafted token the target
likes, which is the entire gain.

**One correction to H's pseudo-code, made explicit.**  H writes the water-fill
cap as a literal ``min(1, ...)``.  With that cap ``w_d`` can exceed ``w_{d-1}``
(a good ``x_{d+1}`` can ask to reach depth ``d+1`` more often than depth ``d``
was itself reached), and then ``a_d = w_d / w_{d-1} > 1`` is not a probability.
Two caps are implemented:

``--cap reach`` (default)
    Cap the water-fill at ``w_{d-1}``.  ``w_d <= w_{d-1}`` by construction,
    ``E[w_d] = A_d`` still holds exactly (feasible because
    ``w_{d-1} >= A_d``), and the coin is always a probability.  This is the
    smallest change that makes H's law well defined and budget-exact.
``--cap one``
    H's literal cap, with ``a_d`` clipped at 1.  The clip wastes budget, so
    this arm is a lower bound; the report counts how often the clip binds.

Both caps coincide whenever ``w_{d-1} = 1``, so both reproduce the current law
exactly on the ``c = 1`` windows (H §3.2: "when c = 1 the law is bit-identical
to today's"), which the report verifies rather than assumes -- and fails on if
it ever stops holding.  Holding it needs one piece of float64 care: when the
budget reaches the cap the water-fill has exactly one solution (everything
saturates), and computing it as ``lam = cap - min(base)`` and adding it back
returns ``0.9999999999999999`` often enough to break the identity on a couple
of windows per four hundred.  :func:`_block_realised_reach` short-circuits that
case to ``cap`` instead.

Arithmetic and tie ownership (mirrored, with citations)
-------------------------------------------------------
All of it is float64, from ``pr391_softfloat64_verifier_decision.py``:

* row preparation ``_prepare_batched_candidate_row`` (lines 33-77):
  rank by ``lexsort((ids.astype(uint64), -values.astype(float64)))`` -- score
  descending, **id ascending on a score tie**; keep ``probs > 0``; top-p by
  ``cumulative_before < top_p`` (**strict**, and computed on the raw
  full-vocabulary probabilities, i.e. the mass *before* the entry); normalise
  once by ``first_total``; return **sorted by token id ascending**;
* ``_prepare_candidate_row`` (lines 88-105) adds the second normalisation
  ``_renormalize_sparse_probabilities`` (lines 80-86).  **Draft rows get the
  double normalisation, target rows only the single one** (lines 246-265);
* ``alpha`` uses the *single*-normalised target row (line 281) while the
  residual uses the *double*-normalised one (line 297).  Both are mirrored;
* accept on ``u <= alpha`` -- line 289, ``<=``, so a draw exactly equal to the
  accept probability **accepts**;
* ``_sample_prepared`` (lines 138-149): ``cdf = cumsum(p); cdf /= sum(p);``
  ``index = min(searchsorted(cdf, u, side="right"), n - 1)`` -- ``side="right"``
  means a draw exactly equal to a CDF breakpoint lands in the **next** bucket,
  and the final clip owns the ``u`` above the last breakpoint;
* ``_prepare_residual`` (lines 165-197): ``union1d`` of both supports,
  ``max(p - q, 0)``, drop non-positive, normalise twice; when nothing survives,
  fall back to the double-normalised target row (lines 303-306);
* a stop token accepted at depth ``d`` ends the window with
  ``draws_used = d + 1`` and no selected token (lines 290-303);
* ``q <= 0`` for a drafted token yields ``alpha = 1 if p > 0 else 0``
  (lines 282-284) -- unreachable in practice, mirrored anyway.

Self-check
----------
Before scoring anything, the replay is compared against the decision the run
**actually returned** for every logged cycle (``accepted`` / ``first_reject`` /
``selected_token`` / ``selected_kind`` / ``draws_used``).  A single mismatch is
a hard failure: it means this file no longer mirrors the lane, and every number
below it would be worthless.  ``--allow-replay-mismatch`` downgrades it to a
warning for debugging.

Which law that replay uses is a property of the log.  A ``pr391_raw`` or
``stock_prepared`` log ran the shipped law, so :func:`decide_current` is the
oracle.  A ``stock_prepared_bv`` log was produced with
``MTPLX_FABLE_BLOCK_VERIFY=1``, so :func:`decide_block` is -- per window, since
a window that could not arm (a row the lane never materialised) fell back to
the shipped law and records ``block_valid = 0``.  An armed log carries the
in-loop ladder as well, and :func:`block_ladder_columns` recomputes it here and
demands entry-for-entry equality: that is the exactness proof that
``mtplx/fable_block_verify.py`` implements the law in this file, and it is what
a BV-armed run must show.

Usage::

    python scripts/fable/offline_block_verification.py rows.npz
    python scripts/fable/offline_block_verification.py rows.npz --ms-per-window 37.47
    python scripts/fable/offline_block_verification.py rows.npz --cap one --json out.json

    # exactness check of the in-loop implementation, on a BV-armed run
    MTPLX_FABLE_BLOCK_VERIFY=1 MTPLX_FABLE_K20_LOG=/path/bv.npz <benchmark>
    python scripts/fable/offline_block_verification.py /path/bv.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Iterable, Sequence

import numpy as np

#: Defaults for the PR391 D3/M4 lane.  A stock-lane log carries its own
#: depth and target-row count, read off the arrays by :func:`log_spec`.
DEPTH = 3
TARGET_ROWS = DEPTH + 1
TOP_P = 0.95

LAYOUT_PR391 = "pr391_raw"
LAYOUT_STOCK = "stock_prepared"
LAYOUT_STOCK_BV = "stock_prepared_bv"

SELECTED_NONE = 0
SELECTED_CORRECTION = 1
SELECTED_BONUS = 2

ZERO = np.float64(0.0)
ONE = np.float64(1.0)


# ---------------------------------------------------------------------------
# Row preparation -- exact mirror of the kernel's NumPy reference.
# ---------------------------------------------------------------------------


def prepare_batched_row(
    ids: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    *,
    top_p: float = TOP_P,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror ``_prepare_batched_candidate_row`` (kernel lines 33-77)."""

    ids = np.asarray(ids, dtype=np.uint32)
    values = np.asarray(values, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    probabilities = probs.astype(np.float64)
    rank = np.lexsort((ids.astype(np.uint64), -values.astype(np.float64)))
    ranked_ids = ids[rank]
    ranked_probs = probabilities[rank]
    cumulative_before = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(ranked_probs[:-1], dtype=np.float64))
    )
    retained = ranked_probs > ZERO
    bounded = np.float64(top_p)
    if bounded < ONE:
        retained &= cumulative_before < bounded
    retained_ids = ranked_ids[retained]
    retained_probs = ranked_probs[retained]
    first_total = np.sum(retained_probs, dtype=np.float64)
    if not np.isfinite(first_total) or first_total <= ZERO:
        raise ValueError("candidate row must retain positive finite mass")
    token_order = np.argsort(retained_ids)
    return (
        retained_ids[token_order].astype(np.uint32, copy=False),
        retained_probs[token_order] / first_total,
    )


def renormalize_sparse(probabilities: np.ndarray) -> np.ndarray:
    """Mirror ``_renormalize_sparse_probabilities`` (kernel lines 80-86)."""

    sanitized = np.where(
        np.isfinite(probabilities) & (probabilities > ZERO), probabilities, ZERO
    )
    return sanitized / np.sum(sanitized, dtype=np.float64)


def prepare_row(
    ids: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    *,
    top_p: float = TOP_P,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror ``_prepare_candidate_row`` (kernel lines 88-105)."""

    prepared_ids, once = prepare_batched_row(ids, values, probs, top_p=top_p)
    return prepared_ids, renormalize_sparse(once)


def prepared_row(
    ids: np.ndarray, values: np.ndarray, probs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Take an ALREADY-shaped host row as-is (``stock_prepared`` layout).

    The stock native-MTP lane hands its decision ``SparseDistribution`` /
    ``BatchedSparseDistributions`` rows that have already had temperature,
    top-p and top-k applied and been renormalised (``sampling.py:122-176``,
    ``sampling.py:48-59``).  Re-running the kernel's ``_prepare_candidate_row``
    on those would apply top-p 0.95 a *second* time to an already-truncated,
    already-renormalised row and cut mass the lane never cut.  So this is the
    right preparation for that layout: drop the zero-probability padding, sort
    by token id (the order every lookup here assumes), and renormalise once --
    which is a no-op on a row that already sums to 1, and the kernel's own
    second normalisation for one that does not.

    ``values`` is unused: for this layout it carries ``log(probs)``, which
    ranks identically, and the ordering is by id regardless.
    """

    del values
    ids = np.asarray(ids, dtype=np.uint32)
    probs = np.asarray(probs, dtype=np.float64)
    keep = probs > ZERO
    kept_ids = ids[keep]
    kept_probs = probs[keep]
    if kept_ids.size == 0:
        raise ValueError("prepared row must retain positive finite mass")
    order = np.argsort(kept_ids)
    return kept_ids[order], renormalize_sparse(kept_probs[order])


def lookup(token_ids: np.ndarray, probabilities: np.ndarray, token: int) -> np.float64:
    """Mirror ``_lookup_prepared`` (kernel lines 152-162)."""

    hits = np.nonzero(token_ids == np.uint32(token))[0]
    return ZERO if hits.size == 0 else np.float64(probabilities[int(hits[0])])


def lookup_many(
    token_ids: np.ndarray, probabilities: np.ndarray, wanted: np.ndarray
) -> np.ndarray:
    """Vectorised :func:`lookup` for a prepared (id-ascending, unique) row.

    Prepared rows come out of :func:`prepare_batched_row` sorted by token id
    with no duplicates (kernel lines 73-77), so a binary search is exact.
    """

    wanted = np.asarray(wanted, dtype=np.uint32)
    out = np.zeros(wanted.size, dtype=np.float64)
    if token_ids.size == 0 or wanted.size == 0:
        return out
    position = np.searchsorted(token_ids, wanted)
    clipped = np.minimum(position, token_ids.size - 1)
    hit = token_ids[clipped] == wanted
    out[hit] = np.asarray(probabilities, dtype=np.float64)[clipped[hit]]
    return out


def sample_prepared(
    token_ids: np.ndarray, probabilities: np.ndarray, uniform: float
) -> int:
    """Mirror ``_sample_prepared`` (kernel lines 138-149)."""

    cdf = np.cumsum(probabilities, dtype=np.float64)
    cdf /= np.sum(probabilities, dtype=np.float64)
    index = min(
        int(np.searchsorted(cdf, np.float64(uniform), side="right")),
        int(token_ids.size) - 1,
    )
    return int(token_ids[index])


def prepare_residual(
    target_ids: np.ndarray,
    target_probs: np.ndarray,
    draft_ids: np.ndarray,
    draft_probs: np.ndarray,
    *,
    scale: np.float64 = ONE,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Mirror ``_prepare_residual`` (kernel lines 165-197).

    ``scale`` is the block law's reach credit ``c_{d-1}`` multiplying the target
    term (H §3.2, "one scalar multiply on the target term").  ``scale = 1``
    reproduces the kernel byte-for-byte.
    """

    union_ids = np.union1d(target_ids, draft_ids).astype(np.uint32, copy=False)
    residual = np.array(
        [
            max(
                np.float64(scale) * lookup(target_ids, target_probs, int(token))
                - lookup(draft_ids, draft_probs, int(token)),
                ZERO,
            )
            for token in union_ids
        ],
        dtype=np.float64,
    )
    residual = np.where(np.isfinite(residual) & (residual > ZERO), residual, ZERO)
    keep = residual > ZERO
    first_total = np.sum(residual[keep], dtype=np.float64)
    if not np.isfinite(first_total) or first_total <= ZERO:
        return None
    once = residual[keep] / first_total
    sanitized = np.where(np.isfinite(once) & (once > ZERO), once, ZERO)
    return union_ids[keep], sanitized / np.sum(sanitized, dtype=np.float64)


# ---------------------------------------------------------------------------
# One window's prepared rows.
# ---------------------------------------------------------------------------


class Window:
    """The seven prepared K20 rows plus the draws of one verify cycle."""

    __slots__ = (
        "draft_tokens",
        "draft",
        "target_single",
        "target_double",
        "uniforms",
        "bonus_allowed",
        "stops",
        "depth",
    )

    def __init__(
        self,
        *,
        draft_tokens: Sequence[int],
        draft_rows: Sequence[tuple[np.ndarray, np.ndarray]],
        target_rows: Sequence[tuple[np.ndarray, np.ndarray]],
        uniforms: np.ndarray,
        bonus_allowed: bool,
        stops: frozenset[int],
    ) -> None:
        self.draft_tokens = [int(token) for token in draft_tokens]
        self.draft = list(draft_rows)
        self.depth = len(self.draft)
        self.target_single = list(target_rows)
        self.target_double = [
            (ids, renormalize_sparse(probs)) for ids, probs in target_rows
        ]
        self.uniforms = np.asarray(uniforms, dtype=np.float64)
        self.bonus_allowed = bool(bonus_allowed)
        self.stops = stops
        if len(self.target_single) <= self.depth:
            raise ValueError("a window needs one target row per depth plus a bonus row")
        if self.uniforms.size <= self.depth or not np.all(np.isfinite(self.uniforms)):
            raise ValueError("a window needs one finite uniform per depth plus one")

    def rho(self, depth: int, token: int) -> np.float64:
        """``p_d(token) / q_d(token)`` on the prepared rows, kernel-style.

        Kernel lines 280-287: a drafted token with ``q <= 0`` scores 1 when the
        target keeps it and 0 otherwise; ``rho`` is otherwise the raw ratio and
        is clipped by the *caller*, which is the only difference between the
        two laws at depth 1.
        """

        target_ids, target_probs = self.target_single[depth]
        draft_ids, draft_probs = self.draft[depth]
        p_value = lookup(target_ids, target_probs, token)
        q_value = lookup(draft_ids, draft_probs, token)
        if q_value <= ZERO:
            return ONE if p_value > ZERO else ZERO
        return np.float64(p_value / q_value)


def log_spec(log: dict[str, np.ndarray]) -> dict[str, Any]:
    """Read the layout and the window shape off one loaded log."""

    layout = str(log["layout"]) if "layout" in log else LAYOUT_PR391
    if layout not in {LAYOUT_PR391, LAYOUT_STOCK, LAYOUT_STOCK_BV}:
        raise ValueError(f"unknown K20 log layout {layout!r}")
    return {
        "layout": layout,
        "block_armed": layout == LAYOUT_STOCK_BV,
        "block_cap": (str(log["block_cap"]) if "block_cap" in log else None),
        "depth": int(log["draft_tokens"].shape[1]),
        "target_rows": int(log["target_ids"].shape[1]),
        "cycles": int(log["draft_tokens"].shape[0]),
        "has_raw_logits": bool(log.get("has_raw_logits", np.uint8(1))),
        "temperature": float(log["temperature"]) if "temperature" in log else 1.0,
        "draft_temperature": (
            float(log["draft_temperature"]) if "draft_temperature" in log else 1.0
        ),
        "top_p": float(log["top_p"]) if "top_p" in log else TOP_P,
        "top_k": int(log["top_k"]) if "top_k" in log else 0,
    }


def window_uniforms(
    log: dict[str, np.ndarray], index: int, depth: int
) -> np.ndarray:
    """The window's accept coins, with the undrawn ones filled deterministically.

    The PR391 lane reserves all ``depth + 1`` draws up front, so every entry is
    real.  The **stock** lane draws an accept coin only for the depths it
    actually reaches -- once a depth rejects the loop breaks -- so a
    counterfactual law that accepts deeper has no logged draw to use.  Those
    slots arrive as NaN and are filled from a deterministic stream seeded by
    the window's own logged PCG64 state (``rng_state``) and its index.

    Every law scored on this window gets the *same* filled array, so the
    comparison stays paired; and because the fill is a pure function of the
    log, re-running the scorer reproduces it exactly.
    """

    uniforms = np.asarray(log["decision_uniforms"][index], dtype=np.float64).copy()
    missing = ~np.isfinite(uniforms)
    if not missing.any():
        return uniforms
    state = (
        [int(word) for word in np.asarray(log["rng_state"][index]).reshape(-1)]
        if "rng_state" in log
        else [0, 0, 0, 0]
    )
    stream = np.random.default_rng(
        np.random.SeedSequence(entropy=[*state, int(index), int(depth)])
    )
    uniforms[missing] = stream.random(int(missing.sum()))
    return uniforms


def build_window(
    log: dict[str, np.ndarray],
    index: int,
    stops: frozenset[int],
    *,
    spec: dict[str, Any] | None = None,
) -> Window | None:
    """One scoreable window, or ``None`` when the log did not capture it whole.

    A window is skipped when any draft row or any of the ``depth + 1`` target
    rows is absent -- the greedy stock lane builds no distributions at all, and
    the lazy per-row target path builds only the rows it reaches.  Skipping is
    reported by :func:`score` rather than papered over, because scoring a
    partial window would bias ``E[l]`` downward for both laws.
    """

    spec = spec or log_spec(log)
    depth = int(spec["depth"])
    prepared = spec["layout"] in {LAYOUT_STOCK, LAYOUT_STOCK_BV}
    draft_valid = log.get("draft_valid")
    target_valid = log.get("target_valid")
    if draft_valid is not None and not np.all(draft_valid[index, :depth]):
        return None
    if target_valid is not None and not np.all(target_valid[index, : depth + 1]):
        return None

    make_draft = prepared_row if prepared else prepare_row
    make_target = prepared_row if prepared else prepare_batched_row
    try:
        draft_rows = [
            make_draft(
                log["draft_ids"][index, position],
                log["draft_values"][index, position],
                log["draft_probs"][index, position],
            )
            for position in range(depth)
        ]
        target_rows = [
            make_target(
                log["target_ids"][index, row],
                log["target_values"][index, row],
                log["target_probs"][index, row],
            )
            for row in range(depth + 1)
        ]
    except ValueError:
        return None
    return Window(
        draft_tokens=log["draft_tokens"][index][:depth],
        draft_rows=draft_rows,
        target_rows=target_rows,
        uniforms=window_uniforms(log, index, depth),
        bonus_allowed=bool(log["bonus_allowed"][index]),
        stops=stops,
    )


# ---------------------------------------------------------------------------
# Decision outcome.
# ---------------------------------------------------------------------------


class Outcome:
    __slots__ = (
        "accepted",
        "first_reject",
        "selected_token",
        "selected_kind",
        "selected_present",
        "draws_used",
        "accept_probability",
        "ladder_all_one",
        "clipped_depths",
    )

    def __init__(self, depth: int = DEPTH) -> None:
        self.accepted = 0
        self.first_reject = -1
        self.selected_token = 0
        self.selected_kind = SELECTED_NONE
        self.selected_present = False
        self.draws_used = 0
        self.accept_probability = [0.0] * int(depth)
        self.ladder_all_one = True
        self.clipped_depths = 0

    @property
    def tokens(self) -> int:
        """Tokens this window emits: the accepted prefix plus at most one."""

        return self.accepted + (1 if self.selected_present else 0)

    def key(self, *, with_token: bool = True) -> tuple[int, ...]:
        """The decision, for comparing two laws or a law against the log.

        ``with_token=False`` drops the correction/bonus id.  That is required
        for the ``stock_prepared`` layout: the stock lane samples its
        correction with ``rng.choice`` straight off the live generator
        (``sampling.py:298-306``) rather than from a logged uniform, so the id
        it emitted is not reproducible offline -- but the accept decisions,
        which are what both laws are being scored on, are.
        """

        head = (self.accepted, self.first_reject)
        tail = (self.selected_kind, int(self.selected_present), self.draws_used)
        return (*head, self.selected_token, *tail) if with_token else (*head, *tail)


def _finish_bonus(window: Window, out: Outcome) -> Outcome:
    """Kernel lines 333-343: full accept, then the optional bonus."""

    depth = window.depth
    out.accepted = depth
    out.draws_used = depth
    if window.bonus_allowed:
        bonus_ids, bonus_probs = window.target_single[depth]
        out.selected_token = sample_prepared(
            bonus_ids, bonus_probs, window.uniforms[depth]
        )
        out.selected_kind = SELECTED_BONUS
        out.selected_present = True
        out.draws_used = depth + 1
    return out


def _emit_correction(
    window: Window, out: Outcome, depth: int, credit: np.float64
) -> Outcome:
    """Kernel lines 292-320, with H's ``c`` scaling on the target term."""

    out.first_reject = depth
    target_ids, target_double = window.target_double[depth]
    draft_ids, draft_probs = window.draft[depth]
    residual = prepare_residual(
        target_ids, target_double, draft_ids, draft_probs, scale=credit
    )
    correction_ids, correction_probs = (
        (target_ids, target_double) if residual is None else residual
    )
    out.selected_token = sample_prepared(
        correction_ids, correction_probs, window.uniforms[depth + 1]
    )
    out.selected_kind = SELECTED_CORRECTION
    out.selected_present = True
    out.draws_used = depth + 2
    return out


def decide_current(window: Window) -> Outcome:
    """The shipped law -- kernel lines 265-345, in float64."""

    out = Outcome(window.depth)
    for depth in range(window.depth):
        token = window.draft_tokens[depth]
        alpha = min(ONE, window.rho(depth, token))
        out.accept_probability[depth] = float(alpha)
        if window.uniforms[depth] <= alpha:
            out.accepted = depth + 1
            if token in window.stops:
                out.draws_used = depth + 1
                return out
            continue
        return _emit_correction(window, out, depth, ONE)
    return _finish_bonus(window, out)


def water_fill_lambda(
    q: np.ndarray, base: np.ndarray, cap: np.float64, target: np.float64
) -> np.float64:
    """Smallest ``lam >= 0`` with ``sum q * min(cap, base + lam) == target``.

    ``f(lam)`` is continuous, non-decreasing and piecewise linear with
    breakpoints at ``cap - base``.  ``f(0) <= target`` always holds here
    (``sum_y min(q(y), A * p(y)) <= A``), and ``f(inf) = cap * sum q >= target``
    whenever ``cap >= target``, so a root exists.  The walk is over the
    breakpoints sorted ascending with a stable sort, so the result is a
    deterministic function of the (id-ordered) input rows.
    """

    q = np.asarray(q, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    cap = np.float64(cap)
    target = np.float64(target)
    value = np.sum(q * np.minimum(cap, base), dtype=np.float64)
    if value >= target:
        return ZERO
    threshold = cap - base
    order = np.argsort(threshold, kind="stable")
    threshold = threshold[order]
    weights = q[order]
    active = np.sum(weights[threshold > ZERO], dtype=np.float64)
    level = ZERO
    index = int(np.searchsorted(threshold, ZERO, side="right"))
    size = int(threshold.size)
    while index < size:
        edge = np.float64(threshold[index])
        if active <= ZERO:
            return level
        gain = active * (edge - level)
        if value + gain >= target:
            return level + (target - value) / active
        value += gain
        level = edge
        while index < size and np.float64(threshold[index]) == edge:
            active -= np.float64(weights[index])
            index += 1
    if active > ZERO:
        return level + (target - value) / active
    return level


def alpha_by_depth(window: Window) -> np.ndarray:
    """``alpha_d = min(1, rho_d)`` at **every** depth, past the first rejection.

    The kernel returns as soon as a depth rejects (kernel lines 289-320), so
    the receipts' ``drafts[].accept_probability`` is censored exactly where
    block verification pays -- H §3.2.  The logged rows are not: the drafter
    always runs the full three-deep chain and the M4 window always carries all
    four target rows, so the uncensored ladder is recoverable here.  This is
    H §1.2's table without its truncation.
    """

    return np.array(
        [
            min(ONE, window.rho(depth, window.draft_tokens[depth]))
            for depth in range(window.depth)
        ],
        dtype=np.float64,
    )


def reach_ladder_current(window: Window) -> np.ndarray:
    """``w_d`` for the shipped law: ``prod_{j<=d} min(1, rho_j)``.

    ``w_d`` is the probability, given the rows and the drafted tokens, that the
    window accepts through depth ``d``.  It consults **no uniform**, so
    ``E[l] = sum_d w_d`` is the same quantity the coin-driven replay estimates
    but with the coin noise integrated out.  Both laws are evaluated on the
    same drafted tokens, so the paired difference is a common-random-numbers
    estimator with a far tighter interval than the replay's.

    The shipped law's ladder is already a deterministic function of
    ``x_{1:d}``; :func:`reach_ladder_block` matches that conditioning, so the
    two are directly comparable entry for entry.
    """

    ladder = np.zeros(window.depth, dtype=np.float64)
    credit = ONE
    for depth in range(window.depth):
        credit = credit * min(ONE, window.rho(depth, window.draft_tokens[depth]))
        ladder[depth] = credit
    return ladder


def reach_ladder_block(window: Window, *, cap_mode: str = "reach") -> np.ndarray:
    """``w_d`` for block verification, on the same drafted tokens."""

    ladder = np.zeros(window.depth, dtype=np.float64)
    credit = ONE
    reach = ONE
    for depth in range(window.depth):
        rho = window.rho(depth, window.draft_tokens[depth])
        budget = min(ONE, credit * rho)
        # E[w_d | x_{1:d}] in closed form.  The water-fill sets lambda so that
        # E_{x_{d+1} ~ q_{d+1}}[w_d] is exactly the budget A_d whenever that is
        # feasible, and the cap w_{d-1} otherwise -- i.e. min(A_d, w_{d-1}),
        # verified to 2e-16 on real rows.  Reporting the conditional
        # expectation instead of the realisation Rao-Blackwellises the
        # look-ahead draw out of the estimator: at depth 1 the two laws are
        # then *provably* equal entry-for-entry (both are min(1, rho_1)), so
        # the paired delta contains only the effect and none of the x_{d+1}
        # sampling noise.  The recursion still advances on the REALISED reach,
        # because that is what actually caps the next depth.
        ladder[depth] = min(budget, reach)
        realised = _block_realised_reach(
            window, depth, budget=budget, reach=reach, cap_mode=cap_mode
        )
        credit = budget
        reach = min(realised, reach)
    return ladder


def block_ladder_columns(
    window: Window, *, cap_mode: str = "reach"
) -> dict[str, np.ndarray]:
    """The block law's per-depth ladder, unconditional on the accept coins.

    This is what ``mtplx/fable_block_verify.BlockVerifier`` computes up front
    and what an armed ``stock_prepared_bv`` log records, so comparing the two
    is the exactness check on the in-loop implementation.  Computing every
    depth (rather than stopping at the first rejection like :func:`decide_block`)
    is not a different law: the recursion advances only on an accept, so depth
    ``d``'s entry is by construction the one that is consulted when the window
    reaches ``d`` -- and no other entry is ever read.
    """

    depth_count = window.depth
    columns = {
        "coin": np.zeros(depth_count, dtype=np.float64),
        "scale": np.ones(depth_count, dtype=np.float64),
        "budget": np.zeros(depth_count, dtype=np.float64),
        "realised": np.zeros(depth_count, dtype=np.float64),
        "clipped": np.zeros(depth_count, dtype=np.uint8),
    }
    credit = ONE
    reach = ONE
    for depth in range(depth_count):
        rho = window.rho(depth, window.draft_tokens[depth])
        budget = min(ONE, credit * rho)
        realised = _block_realised_reach(
            window, depth, budget=budget, reach=reach, cap_mode=cap_mode
        )
        coin = ONE if reach <= ZERO else np.float64(realised / reach)
        if coin > ONE:
            columns["clipped"][depth] = 1
            coin = ONE
        columns["scale"][depth] = credit
        columns["coin"][depth] = coin
        columns["budget"][depth] = budget
        columns["realised"][depth] = realised
        credit = budget
        reach = realised if realised < reach else reach
    return columns


def _block_realised_reach(
    window: Window,
    depth: int,
    *,
    budget: np.float64,
    reach: np.float64,
    cap_mode: str,
) -> np.float64:
    """``w_d`` at one depth: the water-filled look-ahead, or the raw budget."""

    if depth + 1 >= window.depth:
        return budget
    cap = ONE if cap_mode == "one" else reach
    if budget >= cap:
        # The row's mass sums to 1, so a budget at or above the cap has exactly
        # one solution: every realisation saturates.  Short-circuiting it is not
        # an optimisation -- solving for `lam = cap - min(base)` and adding it
        # back reintroduces float64 rounding, and at `cap = budget = 1` that
        # rounding is precisely what would break the c = 1 identity with the
        # shipped law (H §3.2).
        return cap
    draft_ids, draft_probs = window.draft[depth + 1]
    target_ids, target_probs = window.target_single[depth + 1]
    next_p = lookup_many(target_ids, target_probs, draft_ids)
    # Prepared rows keep only probs > 0 (kernel line 64), so the mask is
    # belt-and-braces; it also keeps NumPy from warning.
    next_rho = np.divide(
        next_p, draft_probs, out=np.zeros_like(next_p), where=draft_probs > ZERO
    )
    base = np.minimum(ONE, budget * next_rho)
    level = water_fill_lambda(draft_probs, base, cap, budget)
    next_token = window.draft_tokens[depth + 1]
    hits = np.nonzero(draft_ids == np.uint32(next_token))[0]
    return min(cap, base[int(hits[0])] + level) if hits.size else min(cap, level)


def decide_block(window: Window, *, cap_mode: str = "reach") -> Outcome:
    """Block verification -- H §Option B, §3.2.

    ``cap_mode='reach'`` caps the water-fill at ``w_{d-1}`` (budget-exact and
    always a probability); ``cap_mode='one'`` is H's literal ``min(1, .)`` with
    the coin clipped at 1, which wastes budget and is a lower bound.
    """

    if cap_mode not in {"reach", "one"}:
        raise ValueError("cap_mode must be 'reach' or 'one'")
    out = Outcome(window.depth)
    credit = ONE  # c_{d-1}: the reach budget entering this depth
    reach = ONE  # w_{d-1}: the probability this depth was reached at all
    for depth in range(window.depth):
        token = window.draft_tokens[depth]
        rho = window.rho(depth, token)
        budget = min(ONE, credit * rho)  # A_d
        # The c = 1 identity (H §3.2).  With credit = reach = 1 the final
        # depth's coin is exactly min(1, rho) and the residual scale is 1, so
        # it always agrees; an earlier depth agrees only when its budget is
        # also 1, because otherwise the water-fill redistributes it.
        if credit != ONE or reach != ONE:
            out.ladder_all_one = False
        elif depth + 1 < window.depth and budget != ONE:
            out.ladder_all_one = False
        realised = _block_realised_reach(
            window, depth, budget=budget, reach=reach, cap_mode=cap_mode
        )
        # reach == 0 is a measure-zero branch (it needs an earlier coin of 0
        # AND a uniform of exactly 0.0); the conditional is arbitrary there.
        coin = ONE if reach <= ZERO else np.float64(realised / reach)
        if coin > ONE:
            out.clipped_depths += 1
            coin = ONE
        out.accept_probability[depth] = float(coin)
        if window.uniforms[depth] <= coin:
            out.accepted = depth + 1
            if token in window.stops:
                out.draws_used = depth + 1
                return out
            credit = budget
            reach = realised if realised < reach else reach
            continue
        return _emit_correction(window, out, depth, credit)
    return _finish_bonus(window, out)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def load_log(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        log = {key: handle[key] for key in handle.files}
    required = (
        "draft_ids",
        "draft_values",
        "draft_probs",
        "target_ids",
        "target_values",
        "target_probs",
        "draft_tokens",
        "decision_uniforms",
        "accepted",
        "first_reject",
        "selected_token",
        "selected_kind",
        "selected_present",
        "draws_used",
        "bonus_allowed",
    )
    missing = [key for key in required if key not in log]
    if missing:
        raise KeyError(f"K20 log is missing {missing}; was it written by fable_k20_log?")
    log.setdefault("layout", np.asarray(LAYOUT_PR391))
    return log


def logged_key(
    log: dict[str, np.ndarray], index: int, *, with_token: bool = True
) -> tuple[int, ...]:
    head = (int(log["accepted"][index]), int(log["first_reject"][index]))
    tail = (
        int(log["selected_kind"][index]),
        int(log["selected_present"][index]),
        int(log["draws_used"][index]),
    )
    token = (int(log["selected_token"][index]),)
    return (*head, *token, *tail) if with_token else (*head, *tail)


def score(
    log: dict[str, np.ndarray],
    *,
    cap_mode: str = "reach",
    limit: int | None = None,
    block_ladder_tol: float = 0.0,
) -> dict[str, Any]:
    """Replay both laws over every logged window.

    The self-check replays **the law the device actually ran**: the shipped one
    for a ``pr391_raw`` / ``stock_prepared`` log, and the block law for the
    windows of a ``stock_prepared_bv`` log whose ``block_valid`` is 1 (a window
    that could not arm fell back to the shipped law and is checked against it).
    On an armed log the recomputed ladder is compared entry for entry against
    the one ``mtplx/fable_block_verify.py`` wrote, which is the exactness proof
    of the in-loop implementation rather than a smoke test of it.
    """

    spec = log_spec(log)
    depth = int(spec["depth"])
    # The stock lane's correction id comes off the live generator, not a
    # logged uniform, so it is not reproducible offline; every other field is.
    with_token = spec["layout"] == LAYOUT_PR391
    armed_log = bool(spec["block_armed"])
    block_valid = log.get("block_valid")
    logged_coin = log.get("block_coin")
    logged_scale = log.get("block_scale")
    stops = frozenset(int(token) for token in log.get("stop_ids", ()))
    cycles = int(log["draft_tokens"].shape[0])
    if limit is not None:
        cycles = min(cycles, int(limit))

    current_tokens: list[int] = []
    block_tokens: list[int] = []
    current_accepted: list[int] = []
    block_accepted: list[int] = []
    current_rows: list[np.ndarray] = []
    block_rows: list[np.ndarray] = []
    alpha_rows: list[np.ndarray] = []
    agree = 0
    ladder_one = 0
    ladder_one_agree = 0
    replay_mismatch: list[int] = []
    skipped: list[int] = []
    clipped = 0
    block_armed_cycles = 0
    ladder_mismatch: list[int] = []
    max_coin_error = 0.0
    max_scale_error = 0.0

    for index in range(cycles):
        window = build_window(log, index, stops, spec=spec)
        if window is None:
            skipped.append(index)
            continue
        current = decide_current(window)
        block = decide_block(window, cap_mode=cap_mode)
        armed = armed_log and (block_valid is None or bool(block_valid[index]))
        if armed:
            block_armed_cycles += 1
            columns = block_ladder_columns(window, cap_mode=cap_mode)
            if logged_coin is not None and logged_scale is not None:
                coin_error = float(
                    np.max(np.abs(columns["coin"] - logged_coin[index, :depth]))
                )
                scale_error = float(
                    np.max(np.abs(columns["scale"] - logged_scale[index, :depth]))
                )
                max_coin_error = max(max_coin_error, coin_error)
                max_scale_error = max(max_scale_error, scale_error)
                if max(coin_error, scale_error) > block_ladder_tol:
                    ladder_mismatch.append(index)
        device = block if armed else current
        if device.key(with_token=with_token) != logged_key(
            log, index, with_token=with_token
        ):
            replay_mismatch.append(index)
        current_tokens.append(current.tokens)
        block_tokens.append(block.tokens)
        current_accepted.append(current.accepted)
        block_accepted.append(block.accepted)
        current_rows.append(reach_ladder_current(window))
        block_rows.append(reach_ladder_block(window, cap_mode=cap_mode))
        alpha_rows.append(alpha_by_depth(window))
        clipped += block.clipped_depths
        same = current.key(with_token=with_token) == block.key(with_token=with_token)
        agree += int(same)
        if block.ladder_all_one:
            ladder_one += 1
            ladder_one_agree += int(same)

    scored = len(current_rows)
    empty = np.zeros((0, depth), dtype=np.float64)
    current_ladder = np.stack(current_rows) if scored else empty
    block_ladder = np.stack(block_rows) if scored else empty
    alpha = np.stack(alpha_rows) if scored else empty
    current_length = current_ladder.sum(axis=1)
    block_length = block_ladder.sum(axis=1)
    paired = block_length - current_length
    return {
        "cycles": cycles,
        "cycles_scored": scored,
        "cycles_skipped_incomplete": len(skipped),
        "layout": spec["layout"],
        "depth": depth,
        "has_raw_logits": bool(spec["has_raw_logits"]),
        "cap_mode": cap_mode,
        "log_block_cap": spec["block_cap"],
        "replay_law": "block" if armed_log else "current",
        "block_armed_cycles": block_armed_cycles,
        "block_ladder_mismatch_cycles": ladder_mismatch,
        "block_ladder_max_coin_error": max_coin_error,
        "block_ladder_max_scale_error": max_scale_error,
        "compares_selected_token": with_token,
        "replay_mismatch_cycles": replay_mismatch,
        "current": _summarise(
            np.asarray(current_tokens, dtype=np.float64),
            current_accepted,
            current_ladder,
        ),
        "block": _summarise(
            np.asarray(block_tokens, dtype=np.float64), block_accepted, block_ladder
        ),
        "paired_delta_tokens_per_window": (
            float(np.mean(paired)) if scored else float("nan")
        ),
        "paired_delta_sem": (
            float(np.std(paired, ddof=1) / np.sqrt(scored))
            if scored > 1
            else float("nan")
        ),
        "alpha_uncensored": {
            "mean": [
                float(np.mean(alpha[:, d])) if scored else float("nan")
                for d in range(depth)
            ],
            "p_zero": [
                float(np.mean(alpha[:, d] <= 0.0)) if scored else float("nan")
                for d in range(depth)
            ],
            "p_one": [
                float(np.mean(alpha[:, d] >= 1.0)) if scored else float("nan")
                for d in range(depth)
            ],
        },
        "agree_fraction": (agree / scored) if scored else float("nan"),
        "ladder_all_one_cycles": ladder_one,
        "ladder_all_one_agree_fraction": (
            (ladder_one_agree / ladder_one) if ladder_one else float("nan")
        ),
        "clipped_coin_depths": clipped,
    }


def _summarise(
    tokens: np.ndarray, accepted: Iterable[int], ladder: np.ndarray
) -> dict[str, Any]:
    """Both estimators of tokens/window, plus the per-depth reach ladder.

    ``tokens_per_window`` is the coin-driven replay -- the emission simulation,
    which is what verifies exactness and agreement.  ``tokens_per_window_e``
    integrates the coins out (``1 + sum_d w_d``) and is the number to quote:
    it removes the accept-coin variance entirely, leaving only the drafted-row
    sampling noise, which is *shared* between the two laws.
    """

    accepted_arr = np.asarray(list(accepted), dtype=np.float64)
    size = int(tokens.size)
    mean = float(np.mean(tokens)) if size else float("nan")
    std = float(np.std(tokens, ddof=1)) if size > 1 else float("nan")
    length = ladder.sum(axis=1)
    return {
        "tokens_per_window": mean,
        "tokens_per_window_sem": (std / np.sqrt(size)) if size > 1 else float("nan"),
        "tokens_per_window_e": (float(np.mean(length)) + 1.0) if size else float("nan"),
        "tokens_per_window_e_sem": (
            float(np.std(length, ddof=1) / np.sqrt(size)) if size > 1 else float("nan")
        ),
        "reach_by_depth": [
            float(np.mean(ladder[:, position])) if size else float("nan")
            for position in range(int(ladder.shape[1]))
        ],
        "accepted_mean": float(np.mean(accepted_arr)) if size else float("nan"),
        "full_accept_fraction": (
            float(np.mean(accepted_arr == int(ladder.shape[1])))
            if size
            else float("nan")
        ),
    }


def report(result: dict[str, Any], *, ms_per_window: float | None) -> str:
    depth = int(result["depth"])
    lines: list[str] = []
    lines.append(
        f"layout                     {result['layout']}   depth={depth}   "
        f"raw logits={'yes' if result['has_raw_logits'] else 'no'}"
    )
    lines.append(
        f"cycles                     {result['cycles']} "
        f"({result['cycles_scored']} scored, "
        f"{result['cycles_skipped_incomplete']} skipped as incomplete)"
    )
    lines.append(f"water-fill cap             {result['cap_mode']}")
    mismatches = result["replay_mismatch_cycles"]
    lines.append(
        f"{result['replay_law']}-law replay".ljust(27)
        + (
            f"EXACT on all {result['cycles_scored']} scored decisions"
            if not mismatches
            else f"MISMATCH on {len(mismatches)} cycles {mismatches[:8]}"
        )
    )
    if result["replay_law"] == "block":
        ladder = result["block_ladder_mismatch_cycles"]
        lines.append(
            "block ladder vs in-loop    "
            + (
                f"EXACT on {result['block_armed_cycles']} armed windows "
                f"(max |da_d| {result['block_ladder_max_coin_error']:.3e}, "
                f"max |dc_d| {result['block_ladder_max_scale_error']:.3e})"
                if not ladder
                else f"MISMATCH on {len(ladder)} windows {ladder[:8]}"
            )
        )
        if result["log_block_cap"] and result["log_block_cap"] != result["cap_mode"]:
            lines.append(
                f"                           WARNING: the run used cap "
                f"{result['log_block_cap']!r}, this replay used "
                f"{result['cap_mode']!r}"
            )
    if result["layout"] in {LAYOUT_STOCK, LAYOUT_STOCK_BV}:
        lines.append(
            "                           stock lane: accept coins past the "
            "first rejection are"
        )
        lines.append(
            "                           filled from the window's logged PCG64 "
            "state -- the same"
        )
        lines.append(
            "                           stream for both laws, so the pairing "
            "holds."
        )
    lines.append("")
    lines.append(
        "E[tokens/window] = 1 + sum_d w_d, with the accept coins integrated "
        "out.\nThis is the number to quote; the coin-driven replay below is "
        "the same\nquantity with the coin noise left in, and is there to prove "
        "exactness."
    )
    lines.append("")
    reach_header = "".join(f"{f'w{position + 1}':>9}" for position in range(depth))
    lines.append(
        f"{'':12}{'E[tok/win]':>12}{'+-sem':>9}{reach_header}"
        f"{'replay':>9}{'+-sem':>8}"
    )
    for name in ("current", "block"):
        row = result[name]
        reach = "".join(f"{value:>9.4f}" for value in row["reach_by_depth"])
        lines.append(
            f"{name:12}{row['tokens_per_window_e']:>12.4f}"
            f"{row['tokens_per_window_e_sem']:>9.4f}{reach}"
            f"{row['tokens_per_window']:>9.4f}{row['tokens_per_window_sem']:>8.4f}"
        )
    delta = result["paired_delta_tokens_per_window"]
    sem = result["paired_delta_sem"]
    base = result["current"]["tokens_per_window_e"]
    percent = (delta / base * 100.0) if base else float("nan")
    lines.append(
        f"{'delta':12}{delta:>12.4f}{sem:>9.4f}"
        f"   paired (common random rows), {percent:+.2f}%"
    )
    lines.append("")
    alpha = result["alpha_uncensored"]
    lines.append(
        "alpha = min(1, rho) at EVERY depth (H §1.2, uncensored -- the "
        "receipts stop\nat the first rejection):"
    )
    lines.append(
        f"{'':12}" + "".join(f"{f'd{i + 1}':>9}" for i in range(depth))
    )
    for label, key in (
        ("E[alpha]", "mean"),
        ("P(alpha=0)", "p_zero"),
        ("P(alpha=1)", "p_one"),
    ):
        row = alpha[key]
        lines.append(f"{label:12}" + "".join(f"{value:>9.4f}" for value in row))
    lines.append("")
    lines.append(
        f"laws agree on              {result['agree_fraction'] * 100:.2f}% of windows"
    )
    lines.append(
        f"  of which c = 1 windows   {result['ladder_all_one_cycles']} "
        f"(agreement {result['ladder_all_one_agree_fraction'] * 100:.2f}%, "
        "must be 100.00%)"
    )
    lines.append(f"coin clipped at depths     {result['clipped_coin_depths']}")
    if ms_per_window is not None and ms_per_window > 0:
        lines.append("")
        lines.append(f"at {ms_per_window:.3f} ms/window:")
        for name in ("current", "block"):
            rate = result[name]["tokens_per_window_e"] / (ms_per_window / 1000.0)
            lines.append(f"  {name:24}{rate:>10.2f} tok/s")
        rate_base = base / (ms_per_window / 1000.0)
        rate_gain = (base + delta) / (ms_per_window / 1000.0)
        lines.append(
            f"  {'delta':24}{rate_gain - rate_base:>10.2f} tok/s "
            f"(+-{sem / (ms_per_window / 1000.0):.2f}, {percent:+.2f}%)"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("npz", help="path written by MTPLX_FABLE_K20_LOG")
    parser.add_argument(
        "--cap",
        choices=("reach", "one"),
        default="reach",
        help="water-fill cap; 'reach' is budget-exact, 'one' is H's literal form",
    )
    parser.add_argument(
        "--ms-per-window",
        type=float,
        default=None,
        help="verify-window wall time, to convert tokens/window into tok/s "
        "(H/cost.py calibrates the M4 baseline at 37.47)",
    )
    parser.add_argument("--limit", type=int, default=None, help="score only the first N cycles")
    parser.add_argument(
        "--block-ladder-tol",
        type=float,
        default=0.0,
        help="tolerance for the armed-log ladder check; 0 demands bit equality "
        "with mtplx/fable_block_verify.py, which is what the two float64 "
        "mirrors are built to deliver",
    )
    parser.add_argument("--json", default=None, help="also write the result as JSON")
    parser.add_argument(
        "--allow-replay-mismatch",
        action="store_true",
        help="warn instead of failing when the current-law replay disagrees "
        "with the logged device decision",
    )
    args = parser.parse_args(argv)

    log = load_log(args.npz)
    result = score(
        log,
        cap_mode=args.cap,
        limit=args.limit,
        block_ladder_tol=args.block_ladder_tol,
    )
    print(report(result, ms_per_window=args.ms_per_window))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    if not result["cycles_scored"]:
        print(
            "\nFAIL: no window in this log carries a complete set of rows. "
            "A greedy run (temperature <= 0) builds no distributions at all, "
            "and the lazy per-row target path builds only the rows it reaches.",
            file=sys.stderr,
        )
        return 1
    if result["replay_mismatch_cycles"] and not args.allow_replay_mismatch:
        print(
            f"\nFAIL: the {result['replay_law']}-law replay no longer mirrors "
            "the decision the run actually made, so every number above is "
            "unreliable.",
            file=sys.stderr,
        )
        return 1
    if result["block_ladder_mismatch_cycles"] and not args.allow_replay_mismatch:
        print(
            "\nFAIL: the in-loop block ladder does not match this reference. "
            "mtplx/fable_block_verify.py and this file are mirrors of one law; "
            "one of them has drifted.",
            file=sys.stderr,
        )
        return 1
    if result["ladder_all_one_cycles"] and result["ladder_all_one_agree_fraction"] < 1.0:
        print(
            "\nFAIL: block verification diverged on a c = 1 window, where it "
            "is provably identical to the shipped law.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
