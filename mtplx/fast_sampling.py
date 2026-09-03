"""Runtime sampler helpers for sparse top-k speculative sampling."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Callable

import mlx.core as mx
import numpy as np

from .sampling import (
    PENALTY_MAX,
    PENALTY_MIN,
    SamplerConfig,
    SparseDistribution,
    apply_top_p_top_k,
    softmax,
)

MAX_DEVICE_TOP_K_ORDER = 32


def _host_sparse_distribution(
    logits: np.ndarray,
    config: SamplerConfig,
) -> SparseDistribution:
    logits = np.asarray(logits, dtype=np.float32).astype(np.float64).reshape(-1)
    vocab_size = int(logits.shape[0])
    try:
        probs = apply_top_p_top_k(
            softmax(logits, temperature=config.temperature),
            top_p=config.top_p,
            top_k=config.top_k,
        )
    except ValueError:
        return SparseDistribution.one_hot(0, vocab_size)
    token_ids = np.flatnonzero(probs > 0).astype(np.int64, copy=False)
    return SparseDistribution(token_ids, probs[token_ids], vocab_size)


def apply_penalties_mlx(
    logits: mx.array,
    token_counts: Mapping[int, int] | None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    penalty_overlay: Mapping[int, float] | None = None,
) -> mx.array:
    """On-device MLX twin of ``sampling.apply_penalties`` for one logit row.

    Subtracts the additive OpenAI/vLLM penalty on the RAW logits — before any
    temperature/top-k/top-p — via a sparse scatter over only the seen tokens
    (``logits.at[ids].add(-deltas)``): O(unique_seen), no dense vocab-sized
    allocation and no host round-trip beyond the small (ids, deltas) transfer.
    Penalties clamp to [-2, 2]. ``penalty_overlay`` is the Loop Guard's sparse
    token->subtraction map (not clamped; the guard caps it). Returns the same
    array unchanged (no-op) when nothing is active, preserving exactness.

    Expects a 1-D logit row (``logits.shape == (vocab,)``); batched callers apply
    it per row (the MTP draft block is a handful of positions, not a hot loop).
    """
    presence = min(max(float(presence_penalty), PENALTY_MIN), PENALTY_MAX)
    frequency = min(max(float(frequency_penalty), PENALTY_MIN), PENALTY_MAX)
    counts_active = bool(token_counts) and (presence != 0.0 or frequency != 0.0)
    overlay_active = bool(penalty_overlay)
    if not counts_active and not overlay_active:
        return logits
    if counts_active:
        ids = np.fromiter(token_counts.keys(), dtype=np.int64, count=len(token_counts))
        counts = np.fromiter(token_counts.values(), dtype=np.float64, count=len(token_counts))
        deltas = (frequency * counts + presence * (counts > 0.0)).astype(np.float32)
        logits = logits.at[mx.array(ids)].add(mx.array(-deltas))
    if overlay_active:
        overlay_ids = np.fromiter(
            penalty_overlay.keys(), dtype=np.int64, count=len(penalty_overlay)
        )
        overlay_vals = np.fromiter(
            penalty_overlay.values(), dtype=np.float64, count=len(penalty_overlay)
        ).astype(np.float32)
        logits = logits.at[mx.array(overlay_ids)].add(mx.array(-overlay_vals))
    return logits


class BatchedSparseDistributions:
    def __init__(
        self,
        token_ids: np.ndarray,
        probs: np.ndarray,
        *,
        vocab_size: int,
    ) -> None:
        token_ids = np.asarray(token_ids, dtype=np.int64)
        probs = np.asarray(probs, dtype=np.float64)
        if token_ids.ndim != 2 or probs.ndim != 2:
            raise ValueError("BatchedSparseDistributions expects 2D arrays")
        if token_ids.shape != probs.shape:
            raise ValueError("token_ids/probs shape mismatch")
        row_sums = probs.sum(axis=1)
        if np.any(row_sums <= 0) or not np.all(np.isfinite(row_sums)):
            raise ValueError("each sparse distribution row needs positive mass")
        self.token_ids = token_ids
        self.probs = probs / row_sums[:, None]
        self.vocab_size = int(vocab_size)

    def probability(self, row: int, token_id: int) -> float:
        hits = np.nonzero(self.token_ids[int(row)] == int(token_id))[0]
        if hits.size == 0:
            return 0.0
        return float(self.probs[int(row), int(hits[0])])

    def to_distribution(self, row: int) -> SparseDistribution:
        row = int(row)
        keep = self.probs[row] > 0
        return SparseDistribution(
            self.token_ids[row, keep],
            self.probs[row, keep],
            self.vocab_size,
        )

    def sample(self, row: int, rng: np.random.Generator) -> int:
        row = int(row)
        keep = self.probs[row] > 0
        return int(rng.choice(self.token_ids[row, keep], p=self.probs[row, keep]))

    @classmethod
    def _from_execution_arrays(
        cls,
        token_ids: np.ndarray,
        probs: np.ndarray,
        *,
        vocab_size: int,
    ) -> "BatchedSparseDistributions":
        token_ids = np.asarray(token_ids, dtype=np.int64)
        probs = np.asarray(probs, dtype=np.float64)
        row_sums = probs.sum(axis=1)
        if not np.all(np.isfinite(row_sums) & (row_sums > 0)):
            raise FloatingPointError(
                "fixed batched top-k sampling requires finite positive mass"
            )
        vocab_order = np.argsort(token_ids, axis=1)
        instance = cls.__new__(cls)
        instance.token_ids = np.take_along_axis(token_ids, vocab_order, axis=1)
        ordered_probs = np.take_along_axis(probs, vocab_order, axis=1)
        instance.probs = ordered_probs / row_sums[:, None]
        instance.vocab_size = int(vocab_size)
        return instance


def _deterministic_mlx_top_k_support(
    scaled: mx.array,
    top_k: int,
) -> tuple[mx.array, mx.array]:
    prefix = scaled.shape[:-1]
    rows = scaled.reshape(-1, scaled.shape[-1])
    provisional = mx.argpartition(-rows, kth=top_k - 1, axis=-1)[:, :top_k]
    provisional_values = mx.take_along_axis(rows, provisional, axis=-1)
    cutoff = mx.min(provisional_values, axis=-1, keepdims=True)
    higher = rows > cutoff
    tied = rows == cutoff
    higher_count = mx.sum(higher.astype(mx.int32), axis=-1, keepdims=True)
    tied_rank = mx.cumsum(tied.astype(mx.int32), axis=-1)
    chosen = higher | (tied & (tied_rank <= (top_k - higher_count)))
    selected = mx.where(chosen, rows, -float("inf"))
    top_idx = mx.argpartition(-selected, kth=top_k - 1, axis=-1)[:, :top_k]
    top_vals = mx.take_along_axis(rows, top_idx, axis=-1)

    return top_idx.reshape(*prefix, top_k), top_vals.reshape(*prefix, top_k)


def _order_bounded_mlx_top_k_support(
    top_idx: mx.array,
    top_vals: mx.array,
) -> tuple[mx.array, mx.array]:
    """Order a request-admission-bounded device support by score then id."""
    candidate_values = top_vals[..., :, None]
    other_values = top_vals[..., None, :]
    candidate_ids = top_idx[..., :, None]
    other_ids = top_idx[..., None, :]
    rank = mx.sum(
        (other_values > candidate_values)
        | ((other_values == candidate_values) & (other_ids < candidate_ids)),
        axis=-1,
    )
    order = mx.argsort(rank, axis=-1)
    return (
        mx.take_along_axis(top_idx, order, axis=-1),
        mx.take_along_axis(top_vals, order, axis=-1),
    )


def _fixed_top_k_support(
    logits: mx.array,
    *,
    top_k: int,
) -> tuple[mx.array, mx.array, mx.array]:
    rows = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
    top_idx, top_vals = _deterministic_mlx_top_k_support(rows, top_k)
    return rows, top_idx, top_vals


def _fixed_batched_top_k_distributions(
    logits: mx.array,
    *,
    temperature: float,
    top_k: int,
    vocab_size: int,
) -> BatchedSparseDistributions:
    rows, top_idx, _ = _fixed_top_k_support(logits, top_k=top_k)
    mx.eval(rows, top_idx)
    token_rows = np.asarray(top_idx, dtype=np.int64)
    scaled_rows = np.asarray(rows, dtype=np.float32).astype(np.float64)
    scaled_rows /= temperature
    scaled_rows -= np.max(scaled_rows, axis=1, keepdims=True)
    full_probs = np.exp(scaled_rows)
    full_probs /= np.sum(full_probs, axis=1, keepdims=True)
    probs = np.take_along_axis(full_probs, token_rows, axis=1)
    return BatchedSparseDistributions._from_execution_arrays(
        token_rows,
        probs,
        vocab_size=vocab_size,
    )


def _fixed_batched_top_p_top_k_distributions(
    logits: mx.array,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    vocab_size: int,
) -> BatchedSparseDistributions:
    rows, top_idx, _ = _fixed_top_k_support(logits, top_k=top_k)
    mx.eval(rows, top_idx)
    token_rows = np.asarray(top_idx, dtype=np.int64)
    scaled_rows = np.asarray(rows, dtype=np.float32).astype(np.float64)
    scaled_rows /= temperature
    scaled_rows -= np.max(scaled_rows, axis=1, keepdims=True)
    full_probs = np.exp(scaled_rows)
    full_probs /= np.sum(full_probs, axis=1, keepdims=True)
    prob_rows = np.take_along_axis(full_probs, token_rows, axis=1)
    probability_order = np.lexsort((token_rows, -prob_rows), axis=1)
    token_rows = np.take_along_axis(token_rows, probability_order, axis=1)
    prob_rows = np.take_along_axis(prob_rows, probability_order, axis=1)
    cumulative_before = np.concatenate(
        (
            np.zeros((prob_rows.shape[0], 1), dtype=np.float64),
            np.cumsum(prob_rows[:, :-1], axis=1),
        ),
        axis=1,
    )
    prob_rows = np.where(cumulative_before < top_p, prob_rows, 0.0)
    return BatchedSparseDistributions._from_execution_arrays(
        token_rows, prob_rows, vocab_size=vocab_size
    )


def bind_batched_top_k_distributions(
    config: SamplerConfig,
    *,
    vocab_size: int,
) -> Callable[[mx.array], BatchedSparseDistributions]:
    """Bind one non-null top-k execution route before the decode loop."""
    if config.temperature <= 0 or int(config.top_k) <= 0:
        raise ValueError("fixed batched top-k sampling requires temperature and top_k")
    if int(vocab_size) <= 0:
        raise ValueError("fixed batched top-k sampling requires a positive vocabulary")
    common = {
        "temperature": float(config.temperature),
        "top_k": min(int(config.top_k), int(vocab_size)),
        "vocab_size": int(vocab_size),
    }
    if 0 < float(config.top_p) < 1.0:
        return partial(
            _fixed_batched_top_p_top_k_distributions,
            top_p=float(config.top_p),
            **common,
        )
    return partial(_fixed_batched_top_k_distributions, **common)


def _device_serial_support_arrays(
    logits: mx.array,
    config: SamplerConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Deterministic device top-k support with device float32 mass.

    Serial-lane numerics. Support selection shares the bound route's
    deterministic device selector (cutoff ties keep lower vocabulary ids,
    matching the dense float64 host reference exactly); probability mass for
    top-p decisions uses the device float32 full-vocab logsumexp normalizer
    (the 2.5.4 serial lineage), so only the k-token support ever crosses the
    host boundary. Materializing full vocab rows on the host per sampled
    token was a measured 15-19% serve-lane decode regression (2026-08-11
    four-arm sweep). The cohort bound route keeps the float64 host
    reference; b1-exact binds these serial runners themselves, so no
    contract requires the two lanes to be bitwise-identical to each other.

    Returns (token_rows [N,k] int64, prob_rows [N,k] float64 with top-p
    dropped entries exactly zero, vocab_size). Non-finite logits surface as
    non-finite prob rows for the caller's fallback.
    """
    rows = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
    vocab_size = int(rows.shape[-1])
    k = min(int(config.top_k), vocab_size)
    scaled = rows * (1.0 / float(config.temperature))

    # Hot path: ONE device argpartition to an M=4k candidate superset, then
    # exact deterministic selection on the host over M values. In-loop this
    # is ~5 kernel launches against the deterministic device selector's ~12
    # (measured +0.25 ms/token inside the busy decode stream, 2026-08-11).
    # A cutoff tie can only be truncated if a token OUTSIDE the candidate
    # set ties the k-th selected value, and argpartition guarantees every
    # outside token is <= the candidate minimum — so min(candidates) ==
    # cutoff is the exact spillover condition, and those rows fall back to
    # the deterministic device selector.
    m = min(max(4 * k, k), vocab_size)
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
    cand_ids = np.asarray(cand_idx, dtype=np.int64)
    cand_val_rows = np.asarray(cand_vals, dtype=np.float32)

    # Deterministic selection: value desc, then id asc — the same contract
    # as _deterministic_mlx_top_k_support and the dense host reference.
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
        # Support order for top_p >= 1 stays value-desc (already sorted).

    if spill.any():
        # Exact path for rows whose cutoff tie group may extend beyond the
        # candidate superset.
        _, exact_idx, exact_vals = _fixed_top_k_support(scaled, top_k=k)
        if top_p_active:
            exact_probs = mx.exp(
                exact_vals - mx.logsumexp(scaled, axis=-1, keepdims=True)
            )
        else:
            exact_probs = mx.softmax(exact_vals, axis=-1)
        mx.eval(exact_idx, exact_probs)
        exact_ids = np.asarray(exact_idx, dtype=np.int64)
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


def _device_serial_support_arrays_relaxed_ties(
    logits: mx.array,
    config: SamplerConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Serial support without the 4k cutoff-tie proof superset.

    The selected top-k set may differ only when multiple vocabulary rows tie
    exactly at the cutoff. Probability arithmetic, support ordering, nucleus
    filtering, and the host RNG remain unchanged.
    """
    rows = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
    vocab_size = int(rows.shape[-1])
    k = min(int(config.top_k), vocab_size)
    scaled = rows * (1.0 / float(config.temperature))

    cand_idx = mx.argpartition(-scaled, kth=k - 1, axis=-1)[:, :k]
    cand_vals = mx.take_along_axis(scaled, cand_idx, axis=-1)
    top_p_active = 0.0 < float(config.top_p) < 1.0
    if top_p_active:
        log_total = mx.logsumexp(scaled, axis=-1, keepdims=True)
        cand_probs = mx.exp(cand_vals - log_total)
        mx.eval(cand_idx, cand_vals, cand_probs)
        prob_rows = np.asarray(cand_probs, dtype=np.float64)
    else:
        mx.eval(cand_idx, cand_vals)
        prob_rows = None
    token_rows = np.asarray(cand_idx, dtype=np.int64)
    value_rows = np.asarray(cand_vals, dtype=np.float32)

    order = np.lexsort((token_rows, -value_rows), axis=1)
    token_rows = np.take_along_axis(token_rows, order, axis=1)
    value_rows = np.take_along_axis(value_rows, order, axis=1)
    if prob_rows is not None:
        prob_rows = np.take_along_axis(prob_rows, order, axis=1)
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
        vals64 = value_rows.astype(np.float64)
        vals64 -= np.max(vals64, axis=1, keepdims=True)
        prob_rows = np.exp(vals64)
        prob_rows /= np.sum(prob_rows, axis=1, keepdims=True)

    return token_rows, prob_rows, vocab_size


def _serial_row_distribution(
    token_ids: np.ndarray,
    probs: np.ndarray,
    vocab_size: int,
) -> SparseDistribution | None:
    """One SparseDistribution from a serial support row; None on bad mass."""
    keep = probs > 0
    kept_ids = token_ids[keep]
    kept_probs = probs[keep]
    total = kept_probs.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    order = np.argsort(kept_ids)
    kept_ids = kept_ids[order]
    kept_probs = kept_probs[order] / total
    return SparseDistribution(kept_ids, kept_probs, vocab_size)


def sparse_distribution_from_mlx_logits(
    logits: mx.array,
    config: SamplerConfig,
    *,
    token_counts: Mapping[int, int] | None = None,
    penalty_overlay: Mapping[int, float] | None = None,
) -> SparseDistribution | None:
    """Return an exact sparse distribution for top-p then top-k sampling.

    The Qwen coding sampler uses `top_k=20`, so the final support can never be
    larger than 20 tokens. Selection and RNG ordering match the dense host
    reference; mass is the serial lane's device float32 numerics
    (see ``_device_serial_support_arrays``).

    ``token_counts`` (completion tokens seen so far, scoped by the caller) applies
    the additive presence/frequency penalty to the raw logits BEFORE the
    temperature divide — a no-op (same array) when penalties are 0.
    ``penalty_overlay`` is the Loop Guard's sparse steering map, applied at the
    same raw-logit stage.
    """

    if config.temperature <= 0 or config.top_k <= 0:
        return None

    row = apply_penalties_mlx(
        logits.reshape(-1),
        token_counts,
        config.presence_penalty,
        config.frequency_penalty,
        penalty_overlay=penalty_overlay,
    )
    row = row.astype(mx.float32)
    token_rows, prob_rows, vocab_size = _device_serial_support_arrays(row, config)
    dist = _serial_row_distribution(token_rows[0], prob_rows[0], vocab_size)
    if dist is not None:
        return dist
    # Non-finite mass (NaN/inf logits): keep the host reference's one-hot
    # fallback semantics.
    mx.eval(row)
    return _host_sparse_distribution(np.asarray(row, dtype=np.float32), config)


def sparse_distribution_from_mlx_logits_relaxed_ties(
    logits: mx.array,
    config: SamplerConfig,
) -> SparseDistribution | None:
    """Return the serial distribution while permitting cutoff tie flips."""
    if config.temperature <= 0 or config.top_k <= 0:
        return None
    row = logits.reshape(-1).astype(mx.float32)
    token_rows, prob_rows, vocab_size = (
        _device_serial_support_arrays_relaxed_ties(row, config)
    )
    dist = _serial_row_distribution(token_rows[0], prob_rows[0], vocab_size)
    if dist is not None:
        return dist
    mx.eval(row)
    return _host_sparse_distribution(np.asarray(row, dtype=np.float32), config)


def sparse_distributions_from_mlx_logits(
    logits: mx.array,
    config: SamplerConfig,
) -> list[SparseDistribution] | None:
    """Return exact sparse distributions for a batch of logit rows.

    This is the batched equivalent of ``sparse_distribution_from_mlx_logits``.
    It shares one MLX materialization boundary across rows and then applies the
    exact host reference arithmetic to each row.
    """

    if config.temperature <= 0 or config.top_k <= 0:
        return None

    rows = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
    token_rows, prob_rows, vocab_size = _device_serial_support_arrays(rows, config)
    host_rows: np.ndarray | None = None
    distributions: list[SparseDistribution] = []
    for index in range(token_rows.shape[0]):
        dist = _serial_row_distribution(
            token_rows[index], prob_rows[index], vocab_size
        )
        if dist is None:
            if host_rows is None:
                mx.eval(rows)
                host_rows = np.asarray(rows, dtype=np.float32)
            dist = _host_sparse_distribution(host_rows[index], config)
        distributions.append(dist)
    return distributions


def batched_sparse_distributions_from_mlx_logits(
    logits: mx.array,
    config: SamplerConfig,
) -> BatchedSparseDistributions | None:
    """Return batched sparse distributions without per-row Python objects."""

    if config.temperature <= 0 or config.top_k <= 0:
        return None

    rows = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
    vocab_size = int(rows.shape[-1])
    k = min(int(config.top_k), vocab_size)
    if k <= 0:
        return None
    token_rows, prob_rows, _ = _device_serial_support_arrays(rows, config)
    try:
        return BatchedSparseDistributions._from_execution_arrays(
            token_rows, prob_rows, vocab_size=vocab_size
        )
    except FloatingPointError:
        pass
    mx.eval(rows)
    distributions = [
        _host_sparse_distribution(row, config)
        for row in np.asarray(rows, dtype=np.float32)
    ]
    token_rows = np.full((len(distributions), k), -1, dtype=np.int64)
    prob_rows = np.zeros((len(distributions), k), dtype=np.float64)
    for row_index, distribution in enumerate(distributions):
        width = int(distribution.token_ids.shape[0])
        token_rows[row_index, :width] = distribution.token_ids
        prob_rows[row_index, :width] = distribution.probs
    return BatchedSparseDistributions(token_rows, prob_rows, vocab_size=vocab_size)


def sample_token_ids_from_mlx_logits(
    logits: mx.array,
    config: SamplerConfig,
) -> mx.array | None:
    """Sample token ids on-device with the same top-k/top-p semantics.

    This is for exact target-prefix verification paths that need sampled target
    ids, not p/q residual distributions. It keeps the small-support sampling on
    MLX and returns one token id per input row.
    """

    if config.temperature <= 0:
        return mx.argmax(logits, axis=-1)

    rows = logits.astype(mx.float32) / float(config.temperature)
    vocab_size = int(rows.shape[-1])
    if config.top_k <= 0:
        if 0 < config.top_p < 1.0:
            return None
        return mx.random.categorical(rows)

    k = min(int(config.top_k), vocab_size)
    if k <= 0:
        return None
    if 0 < config.top_p < 1.0 and k > MAX_DEVICE_TOP_K_ORDER:
        raise ValueError(
            "device top-p sampling requires top_k <= "
            f"{MAX_DEVICE_TOP_K_ORDER}"
        )

    top_idx, top_vals = _deterministic_mlx_top_k_support(rows, k)

    if 0 < config.top_p < 1.0:
        top_idx, top_vals = _order_bounded_mlx_top_k_support(top_idx, top_vals)
        log_total = mx.logsumexp(rows, axis=-1, keepdims=True)
        top_probs = mx.exp(top_vals - log_total)
        higher_mass = mx.cumsum(top_probs, axis=-1) - top_probs
        first = mx.arange(k) == 0
        keep = (higher_mass < float(config.top_p)) | first
        top_vals = mx.where(keep, top_vals, -float("inf"))
    else:
        vocab_order = mx.argsort(top_idx, axis=-1)
        top_idx = mx.take_along_axis(top_idx, vocab_order, axis=-1)
        top_vals = mx.take_along_axis(top_vals, vocab_order, axis=-1)

    sampled_offsets = mx.random.categorical(top_vals)
    return mx.take_along_axis(top_idx, sampled_offsets[..., None], axis=-1)[..., 0]
