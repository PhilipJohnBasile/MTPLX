from __future__ import annotations

import mlx.core as mx
import numpy as np
from types import SimpleNamespace

from mtplx.kernels.qwen4_qsa_m4_fused_kv_gather import (
    bind_qwen4_qsa_m4_fused_kv_gather,
)
from mtplx.models.qwen4_exp import (
    _qsa_rows_gather_kv_route,
    _qsa_stock_rows_gather_kv,
)


def test_fused_kv_gather_matches_two_stock_takes_at_production_shape():
    capacity = 16_384
    rng = np.random.default_rng(20260830)
    keys = mx.array(
        rng.normal(size=(1, 2, capacity, 256)).astype(np.float32),
        dtype=mx.bfloat16,
    )
    values = mx.array(
        rng.normal(size=(1, 2, capacity, 256)).astype(np.float32),
        dtype=mx.bfloat16,
    )
    token_idx = mx.array(rng.integers(0, capacity, size=(4, 2052), dtype=np.int32))

    gather = bind_qwen4_qsa_m4_fused_kv_gather(capacity=capacity)
    selected_keys, selected_values = gather(keys, values, token_idx)
    flat = token_idx.reshape(-1)
    stock_keys = mx.take(keys, flat, axis=2).reshape(1, 2, 4, 2052, 256)
    stock_values = mx.take(values, flat, axis=2).reshape(1, 2, 4, 2052, 256)
    mx.eval(selected_keys, selected_values, stock_keys, stock_values)

    assert selected_keys.shape == stock_keys.shape
    assert selected_values.shape == stock_values.shape
    assert bool(mx.all(selected_keys == stock_keys).item())
    assert bool(mx.all(selected_values == stock_values).item())


def test_fused_kv_gather_rejects_non_production_capacity():
    try:
        bind_qwen4_qsa_m4_fused_kv_gather(capacity=4096)
    except ValueError as exc:
        assert "16K context" in str(exc)
    else:
        raise AssertionError("short-context cache must not install the production lane")


def test_fused_kv_gather_routes_only_physical_m4():
    candidate = object()
    cache = SimpleNamespace(rows_gather_kv_m4=candidate)

    assert _qsa_rows_gather_kv_route(cache, 4) is candidate
    assert _qsa_rows_gather_kv_route(cache, 9) is _qsa_stock_rows_gather_kv
