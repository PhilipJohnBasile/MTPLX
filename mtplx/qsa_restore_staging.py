"""Host-side capacity staging for restored QSA caches.

Session-bank snapshots intentionally persist only the logical cache prefix.
That is the right storage representation, but it means the first restored
suffix forward would otherwise grow three independent QSA backings:

* the attention K/V cache, normally once per prefill chunk;
* the indexer's raw projected-key stream; and
* the indexer's pooled block stream.

This module plans and performs those promotions once, before suffix prefill.
It deliberately has no MLX import.  The caller supplies the two array-runtime
operations that are impossible to express as host metadata:

``allocate_zeros(shape, dtype)``
    Allocate an array with the requested shape and dtype.

``materialize_cache(caches)``
    Evaluate all staged cache roots in one batch.  This callback is required;
    without it MLX could defer the copies into the first suffix forward and
    defeat the staging boundary.

The QSA cache itself is discovered through its existing
``reserve_indexer_capacity`` hook.  No model-class import or type-name check is
needed, and non-QSA entries in the heterogeneous trunk cache are ignored.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .qsa_mtp_precompute import precompute_qsa_replay_capacity


class QSARestoreStagingError(ValueError):
    """The restored cache cannot be promoted without risking stale state."""


class QSAZeroFactory(Protocol):
    """Array allocation injected by the MLX-owning integration layer."""

    def __call__(self, shape: tuple[int, ...], dtype: Any, /) -> Any: ...


@dataclass(frozen=True, slots=True)
class RestoredQSACapacityPlan:
    """One QSA layer's immutable restore-to-suffix capacity contract."""

    cache_index: int
    start_offset: int
    suffix_tokens: int
    end_offset: int
    compress_ratio: int
    indexer_step: int
    kv_step: int
    raw_capacity_before: int
    pooled_capacity_before: int
    kv_capacity_before: int
    raw_capacity: int
    pooled_capacity: int
    kv_capacity: int

    @property
    def raw_needs_promotion(self) -> bool:
        return self.raw_capacity > self.raw_capacity_before

    @property
    def pooled_needs_promotion(self) -> bool:
        return self.pooled_capacity > self.pooled_capacity_before

    @property
    def kv_needs_promotion(self) -> bool:
        return self.kv_capacity > self.kv_capacity_before


@dataclass(frozen=True, slots=True)
class RestoredQSAStageReport:
    """Host-visible receipt for one grouped restore staging operation."""

    plans: tuple[RestoredQSACapacityPlan, ...]
    kv_promotions: int
    raw_promotions: int
    pooled_promotions: int
    materialized: bool

    @property
    def qsa_entries(self) -> int:
        return len(self.plans)


def _nonnegative(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise QSARestoreStagingError(f"{name} must be >= 0; got {value}")
    return value


def _positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise QSARestoreStagingError(f"{name} must be > 0; got {value}")
    return value


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _shape(array: Any, *, name: str, minimum_rank: int) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in array.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise QSARestoreStagingError(f"{name} has no static shape metadata") from exc
    if len(shape) < minimum_rank:
        raise QSARestoreStagingError(
            f"{name} must have rank >= {minimum_rank}; got shape {shape}"
        )
    if any(dim < 0 for dim in shape):
        raise QSARestoreStagingError(f"{name} has a negative dimension: {shape}")
    return shape


def _optional_capacity(array: Any, *, axis: int, name: str) -> int:
    if array is None:
        return 0
    shape = _shape(array, name=name, minimum_rank=axis + 1)
    return shape[axis]


def _is_qsa_cache(cache: Any) -> bool:
    return callable(getattr(cache, "reserve_indexer_capacity", None))


def _inspect_qsa_cache(
    cache: Any,
    *,
    cache_index: int,
    suffix_tokens: int,
) -> RestoredQSACapacityPlan:
    """Read and validate one restored QSA cache using host metadata only."""

    offset = _nonnegative(f"cache[{cache_index}].offset", cache.offset)
    ratio = _positive(f"cache[{cache_index}].ratio", cache.ratio)
    indexer_step = _positive(
        f"cache[{cache_index}].step", getattr(cache, "step", 256)
    )
    if indexer_step & (indexer_step - 1):
        raise QSARestoreStagingError(
            f"cache[{cache_index}].step must be a power of two; got {indexer_step}"
        )

    raw = getattr(cache, "raw_keys", None)
    pooled = getattr(cache, "pooled", None)
    raw_physical = _optional_capacity(
        raw,
        axis=1,
        name=f"cache[{cache_index}].raw_keys",
    )
    pooled_physical = _optional_capacity(
        pooled,
        axis=1,
        name=f"cache[{cache_index}].pooled",
    )
    raw_reserved = _nonnegative(
        f"cache[{cache_index}]._reserved_raw_capacity",
        getattr(cache, "_reserved_raw_capacity", 0),
    )
    pooled_reserved = _nonnegative(
        f"cache[{cache_index}]._reserved_pooled_capacity",
        getattr(cache, "_reserved_pooled_capacity", 0),
    )
    if raw_physical < offset:
        raise QSARestoreStagingError(
            f"cache[{cache_index}].raw_keys capacity {raw_physical} "
            f"cannot cover restored offset {offset}"
        )
    pooled_len = _nonnegative(
        f"cache[{cache_index}].pooled_len", getattr(cache, "pooled_len", 0)
    )
    if pooled_physical < pooled_len:
        raise QSARestoreStagingError(
            f"cache[{cache_index}].pooled capacity {pooled_physical} "
            f"cannot cover logical pooled_len {pooled_len}"
        )

    kv = getattr(cache, "kv", None)
    if kv is None:
        raise QSARestoreStagingError(
            f"cache[{cache_index}] exposes QSA reservation but has no kv backing"
        )
    kv_offset = _nonnegative(f"cache[{cache_index}].kv.offset", kv.offset)
    if kv_offset != offset:
        raise QSARestoreStagingError(
            f"cache[{cache_index}] QSA/KV frontier mismatch: {offset} != {kv_offset}"
        )
    keys = getattr(kv, "keys", None)
    values = getattr(kv, "values", None)
    if keys is None or values is None:
        raise QSARestoreStagingError(
            f"cache[{cache_index}] has no restored K/V geometry; "
            "capacity staging requires a non-empty restored prefix"
        )
    key_shape = _shape(
        keys,
        name=f"cache[{cache_index}].kv.keys",
        minimum_rank=4,
    )
    value_shape = _shape(
        values,
        name=f"cache[{cache_index}].kv.values",
        minimum_rank=4,
    )
    if key_shape[:3] != value_shape[:3]:
        raise QSARestoreStagingError(
            f"cache[{cache_index}] K/V leading shapes differ: "
            f"{key_shape} != {value_shape}"
        )
    kv_physical = key_shape[2]
    if kv_physical < offset:
        raise QSARestoreStagingError(
            f"cache[{cache_index}] KV capacity {kv_physical} "
            f"cannot cover restored offset {offset}"
        )
    kv_step = _positive(
        f"cache[{cache_index}].kv.step", getattr(kv, "step", indexer_step)
    )

    replay = precompute_qsa_replay_capacity(
        start_offset=offset,
        window_tokens=suffix_tokens,
        compress_ratio=ratio,
        allocation_step=indexer_step,
        current_raw_capacity=max(raw_physical, raw_reserved),
        current_pooled_capacity=max(pooled_physical, pooled_reserved),
    )
    # KV is not part of the compiled indexer signature.  Reserve only the
    # step-rounded final frontier rather than paying power-of-two headroom for
    # the much larger attention K/V tensors.
    kv_capacity = (
        kv_physical
        if kv_physical >= replay.end_offset
        else _round_up(replay.end_offset, kv_step)
    )
    return RestoredQSACapacityPlan(
        cache_index=int(cache_index),
        start_offset=offset,
        suffix_tokens=suffix_tokens,
        end_offset=replay.end_offset,
        compress_ratio=ratio,
        indexer_step=indexer_step,
        kv_step=kv_step,
        raw_capacity_before=raw_physical,
        pooled_capacity_before=pooled_physical,
        kv_capacity_before=kv_physical,
        raw_capacity=replay.raw_capacity,
        pooled_capacity=replay.pooled_capacity,
        kv_capacity=kv_capacity,
    )


def plan_restored_qsa_suffix(
    caches: Sequence[Any],
    *,
    suffix_tokens: int,
) -> tuple[RestoredQSACapacityPlan, ...]:
    """Preflight every QSA layer for the *entire* restored suffix.

    ``suffix_tokens`` must be the full uncached suffix length, not the prefill
    chunk size.  Planning the full suffix is what guarantees that subsequent
    chunks do not grow or copy these backings again.

    An empty suffix returns no plans and performs no cache validation because
    an exact cache hit has no QSA forward to stage.
    """

    suffix = _nonnegative("suffix_tokens", suffix_tokens)
    if suffix == 0:
        return ()

    plans = tuple(
        _inspect_qsa_cache(
            cache,
            cache_index=index,
            suffix_tokens=suffix,
        )
        for index, cache in enumerate(caches)
        if _is_qsa_cache(cache)
    )
    if plans:
        frontiers = {plan.start_offset for plan in plans}
        if len(frontiers) != 1:
            raise QSARestoreStagingError(
                "restored QSA layers do not share one token frontier: "
                f"{sorted(frontiers)}"
            )
    return plans


def _copy_kv_prefix_to_capacity(
    array: Any,
    *,
    target_capacity: int,
    live_rows: int,
    allocate_zeros: QSAZeroFactory,
    name: str,
) -> Any:
    old_shape = _shape(array, name=name, minimum_rank=4)
    new_shape = list(old_shape)
    new_shape[2] = target_capacity
    promoted = allocate_zeros(tuple(new_shape), array.dtype)
    got_shape = _shape(promoted, name=f"promoted {name}", minimum_rank=4)
    if got_shape != tuple(new_shape):
        raise QSARestoreStagingError(
            f"promoted {name} has shape {got_shape}; expected {tuple(new_shape)}"
        )
    if getattr(promoted, "dtype", None) != getattr(array, "dtype", None):
        raise QSARestoreStagingError(
            f"promoted {name} changed dtype from {array.dtype} to {promoted.dtype}"
        )
    prefix = [slice(None)] * len(old_shape)
    prefix[2] = slice(0, live_rows)
    prefix_tuple = tuple(prefix)
    promoted[prefix_tuple] = array[prefix_tuple]
    return promoted


def apply_restored_qsa_stage(
    caches: Sequence[Any],
    plans: Sequence[RestoredQSACapacityPlan],
    *,
    allocate_zeros: QSAZeroFactory,
    materialize_cache: Callable[[Sequence[Any]], None],
) -> RestoredQSAStageReport:
    """Apply a preflighted plan and materialize all promoted roots once.

    Plans are revalidated against current cache metadata before any mutation.
    A stale plan therefore fails before one layer is promoted.  Allocation or
    materialization failures are deliberately propagated; continuing into a
    suffix forward with a half-staged cache would hide the actual fault.
    """

    frozen_plans = tuple(plans)
    if not frozen_plans:
        return RestoredQSAStageReport((), 0, 0, 0, False)
    if not callable(allocate_zeros):
        raise TypeError("allocate_zeros must be callable")
    if not callable(materialize_cache):
        raise TypeError("materialize_cache must be callable")

    suffixes = {plan.suffix_tokens for plan in frozen_plans}
    if len(suffixes) != 1:
        raise QSARestoreStagingError(
            f"restore plans disagree on suffix length: {sorted(suffixes)}"
        )
    try:
        current = plan_restored_qsa_suffix(
            caches,
            suffix_tokens=next(iter(suffixes)),
        )
    except QSARestoreStagingError as exc:
        raise QSARestoreStagingError(
            "restored QSA staging plan is stale; re-plan before promotion"
        ) from exc
    if current != frozen_plans:
        raise QSARestoreStagingError(
            "restored QSA staging plan is stale; re-plan before promotion"
        )

    kv_promotions = 0
    for plan in frozen_plans:
        cache = caches[plan.cache_index]
        kv = cache.kv
        if plan.kv_needs_promotion:
            # Build and validate both leaves before rebinding either one.  A
            # bad allocator result cannot split the live K/V pair.
            new_keys = _copy_kv_prefix_to_capacity(
                kv.keys,
                target_capacity=plan.kv_capacity,
                live_rows=plan.start_offset,
                allocate_zeros=allocate_zeros,
                name=f"cache[{plan.cache_index}].kv.keys",
            )
            new_values = _copy_kv_prefix_to_capacity(
                kv.values,
                target_capacity=plan.kv_capacity,
                live_rows=plan.start_offset,
                allocate_zeros=allocate_zeros,
                name=f"cache[{plan.cache_index}].kv.values",
            )
            kv.keys, kv.values = new_keys, new_values
            kv_promotions += 1

        cache.reserve_indexer_capacity(
            raw_capacity=plan.raw_capacity,
            pooled_capacity=plan.pooled_capacity,
        )

    # One grouped barrier is part of the contract.  Merely constructing these
    # MLX expressions would move the copy cost back into the first suffix
    # chunk and make the staging receipt misleading.
    materialize_cache(caches)
    return RestoredQSAStageReport(
        plans=frozen_plans,
        kv_promotions=kv_promotions,
        raw_promotions=sum(plan.raw_needs_promotion for plan in frozen_plans),
        pooled_promotions=sum(
            plan.pooled_needs_promotion for plan in frozen_plans
        ),
        materialized=True,
    )


def stage_restored_qsa_suffix(
    caches: Sequence[Any],
    *,
    suffix_tokens: int,
    allocate_zeros: QSAZeroFactory,
    materialize_cache: Callable[[Sequence[Any]], None],
) -> RestoredQSAStageReport:
    """Plan, promote, and materialize every restored QSA suffix backing."""

    plans = plan_restored_qsa_suffix(caches, suffix_tokens=suffix_tokens)
    return apply_restored_qsa_stage(
        caches,
        plans,
        allocate_zeros=allocate_zeros,
        materialize_cache=materialize_cache,
    )


__all__ = [
    "QSARestoreStagingError",
    "QSAZeroFactory",
    "RestoredQSACapacityPlan",
    "RestoredQSAStageReport",
    "apply_restored_qsa_stage",
    "plan_restored_qsa_suffix",
    "stage_restored_qsa_suffix",
]
