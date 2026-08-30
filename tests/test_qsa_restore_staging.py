"""Pure host tests for restored-QSA capacity staging.

No MLX/model import is allowed here.  NumPy stands in for a lazy array runtime
so the tests can pin capacity, prefix preservation, ordering, and fail-closed
behavior without touching Metal or model weights.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from mtplx.qsa_restore_staging import (
    QSARestoreStagingError,
    apply_restored_qsa_stage,
    plan_restored_qsa_suffix,
    stage_restored_qsa_suffix,
)


MODULE = Path(__file__).parents[1] / "mtplx" / "qsa_restore_staging.py"


class _KV:
    step = 256

    def __init__(self, *, offset: int, capacity: int, marker: float):
        self.offset = offset
        self.keys = np.zeros((1, 2, capacity, 3), dtype=np.float32)
        self.values = np.zeros((1, 2, capacity, 5), dtype=np.float32)
        self.keys[:, :, :offset, :] = marker
        self.values[:, :, :offset, :] = -marker


class _QSA:
    step = 256

    def __init__(
        self,
        *,
        offset: int,
        ratio: int = 4,
        kv_capacity: int | None = None,
        raw_capacity: int | None = None,
        pooled_capacity: int | None = None,
        marker: float = 1.0,
        events: list[str] | None = None,
    ):
        kv_capacity = offset if kv_capacity is None else kv_capacity
        raw_capacity = offset if raw_capacity is None else raw_capacity
        complete = offset // ratio
        pooled_capacity = complete if pooled_capacity is None else pooled_capacity
        self.offset = offset
        self.ratio = ratio
        self.kv = _KV(offset=offset, capacity=kv_capacity, marker=marker)
        self.raw_keys = np.full((1, raw_capacity, 7), marker, dtype=np.float32)
        self.pooled = np.full(
            (1, pooled_capacity, 7), marker,
            dtype=np.float32,
        )
        self.pooled_len = complete
        self._reserved_raw_capacity = raw_capacity
        self._reserved_pooled_capacity = pooled_capacity
        self.reserve_calls: list[tuple[int, int]] = []
        self.events = [] if events is None else events

    def reserve_indexer_capacity(self, *, raw_capacity: int, pooled_capacity: int):
        self.events.append("reserve-indexer")
        self.reserve_calls.append((raw_capacity, pooled_capacity))
        if raw_capacity > self.raw_keys.shape[1]:
            grown = np.zeros((1, raw_capacity, self.raw_keys.shape[2]), dtype=self.raw_keys.dtype)
            grown[:, : self.raw_keys.shape[1], :] = self.raw_keys
            self.raw_keys = grown
        if pooled_capacity > self.pooled.shape[1]:
            grown = np.zeros((1, pooled_capacity, self.pooled.shape[2]), dtype=self.pooled.dtype)
            grown[:, : self.pooled.shape[1], :] = self.pooled
            self.pooled = grown
        self._reserved_raw_capacity = max(
            self._reserved_raw_capacity,
            raw_capacity,
        )
        self._reserved_pooled_capacity = max(
            self._reserved_pooled_capacity,
            pooled_capacity,
        )


def _zeros(shape, dtype):
    return np.zeros(shape, dtype=dtype)


def test_module_has_no_mlx_or_model_import():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any(name == "mlx" or name.startswith("mlx.") for name in imported)
    assert not any(name.startswith("mlx_lm") for name in imported)
    assert not any(name.startswith("mtplx.models") for name in imported)


def test_plan_covers_entire_suffix_but_kv_uses_minimal_step_rounding():
    cache = _QSA(offset=4_097)

    (plan,) = plan_restored_qsa_suffix([cache], suffix_tokens=5_000)

    assert plan.start_offset == 4_097
    assert plan.end_offset == 9_097
    assert plan.raw_capacity == 16_384
    assert plan.pooled_capacity == 4_096
    assert plan.kv_capacity == 9_216
    assert plan.raw_needs_promotion
    assert plan.pooled_needs_promotion
    assert plan.kv_needs_promotion


def test_stage_promotes_all_qsa_backings_and_preserves_live_prefix():
    events: list[str] = []
    cache = _QSA(offset=513, marker=7.0, events=events)
    old_keys = cache.kv.keys.copy()
    old_values = cache.kv.values.copy()

    def materialize(caches):
        assert caches == [cache]
        events.append("materialize")

    report = stage_restored_qsa_suffix(
        [cache],
        suffix_tokens=2_048,
        allocate_zeros=_zeros,
        materialize_cache=materialize,
    )

    assert cache.kv.keys.shape[2] == 2_816
    assert cache.kv.values.shape[2] == 2_816
    np.testing.assert_array_equal(cache.kv.keys[:, :, :513, :], old_keys)
    np.testing.assert_array_equal(cache.kv.values[:, :, :513, :], old_values)
    assert cache.raw_keys.shape[1] == 4_096
    assert cache.pooled.shape[1] == 1_024
    assert cache.offset == cache.kv.offset == 513
    assert cache.pooled_len == 128
    assert events == ["reserve-indexer", "materialize"]
    assert report.qsa_entries == 1
    assert report.kv_promotions == 1
    assert report.raw_promotions == 1
    assert report.pooled_promotions == 1
    assert report.materialized is True


def test_all_layers_are_reserved_before_one_grouped_materialization():
    events: list[str] = []
    first = _QSA(offset=512, marker=1.0, events=events)
    second = _QSA(offset=512, marker=2.0, events=events)

    def materialize(caches):
        assert caches == [first, object_entry, second]
        events.append("materialize")

    object_entry = object()
    report = stage_restored_qsa_suffix(
        [first, object_entry, second],
        suffix_tokens=2_048,
        allocate_zeros=_zeros,
        materialize_cache=materialize,
    )

    assert events == ["reserve-indexer", "reserve-indexer", "materialize"]
    assert report.qsa_entries == 2
    assert report.kv_promotions == 2


def test_stage_is_physically_idempotent_for_the_same_suffix_frontier():
    cache = _QSA(offset=513)
    allocations = 0
    materializations = 0

    def allocate(shape, dtype):
        nonlocal allocations
        allocations += 1
        return np.zeros(shape, dtype=dtype)

    def materialize(_caches):
        nonlocal materializations
        materializations += 1

    first = stage_restored_qsa_suffix(
        [cache],
        suffix_tokens=2_048,
        allocate_zeros=allocate,
        materialize_cache=materialize,
    )
    second = stage_restored_qsa_suffix(
        [cache],
        suffix_tokens=2_048,
        allocate_zeros=allocate,
        materialize_cache=materialize,
    )

    assert first.kv_promotions == 1
    assert second.kv_promotions == 0
    assert second.raw_promotions == 0
    assert second.pooled_promotions == 0
    assert allocations == 2  # K and V only on the first call.
    assert materializations == 2


def test_non_qsa_entries_are_ignored_and_qsa_frontiers_must_match():
    first = _QSA(offset=256)
    second = _QSA(offset=257)

    with pytest.raises(QSARestoreStagingError, match="one token frontier"):
        plan_restored_qsa_suffix(
            [object(), first, object(), second],
            suffix_tokens=1,
        )


def test_preflight_failure_happens_before_any_layer_is_mutated():
    first = _QSA(offset=256)
    broken = _QSA(offset=256)
    broken.kv.offset = 255

    with pytest.raises(QSARestoreStagingError, match="frontier mismatch"):
        stage_restored_qsa_suffix(
            [first, broken],
            suffix_tokens=1,
            allocate_zeros=_zeros,
            materialize_cache=lambda _caches: None,
        )

    assert first.reserve_calls == []
    assert first.kv.keys.shape[2] == 256


def test_empty_suffix_is_an_exact_hit_noop():
    report = stage_restored_qsa_suffix(
        [object()],
        suffix_tokens=0,
        allocate_zeros=_zeros,
        materialize_cache=lambda _caches: pytest.fail("must not materialize"),
    )

    assert report == report.__class__((), 0, 0, 0, False)


def test_restored_qsa_without_kv_geometry_fails_closed():
    cache = _QSA(offset=256)
    cache.kv.keys = None

    with pytest.raises(QSARestoreStagingError, match="no restored K/V geometry"):
        plan_restored_qsa_suffix([cache], suffix_tokens=1)


def test_stale_plan_is_rejected_before_promotion():
    cache = _QSA(offset=256)
    plans = plan_restored_qsa_suffix([cache], suffix_tokens=1)
    cache.offset = cache.kv.offset = 257

    with pytest.raises(QSARestoreStagingError, match="stale"):
        apply_restored_qsa_stage(
            [cache],
            plans,
            allocate_zeros=_zeros,
            materialize_cache=lambda _caches: None,
        )

    assert cache.reserve_calls == []


def test_bad_allocator_cannot_rebind_only_one_side_of_kv():
    cache = _QSA(offset=256)
    old_keys = cache.kv.keys
    old_values = cache.kv.values
    calls = 0

    def broken_allocator(shape, dtype):
        nonlocal calls
        calls += 1
        if calls == 2:
            shape = (*shape[:2], shape[2] - 1, *shape[3:])
        return np.zeros(shape, dtype=dtype)

    with pytest.raises(QSARestoreStagingError, match="expected"):
        stage_restored_qsa_suffix(
            [cache],
            suffix_tokens=1,
            allocate_zeros=broken_allocator,
            materialize_cache=lambda _caches: None,
        )

    assert cache.kv.keys is old_keys
    assert cache.kv.values is old_values
    assert cache.reserve_calls == []


def test_materialization_failure_is_not_hidden():
    cache = _QSA(offset=256)

    def fail(_caches):
        raise RuntimeError("evaluation failed")

    with pytest.raises(RuntimeError, match="evaluation failed"):
        stage_restored_qsa_suffix(
            [cache],
            suffix_tokens=1,
            allocate_zeros=_zeros,
            materialize_cache=fail,
        )
