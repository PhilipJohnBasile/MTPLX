"""fused_add_rmsnorm must match the unfused reference bit-for-bit (#319).

Probed 2026-08-24: the 512-lane looped dispatch can diverge from
x+r / mx.fast.rms_norm at fp16 once the grid crosses 2^15 threads
(rows > 64 at 512 lanes; data-dependent — seed 3 diverges at 65/256/1024
rows, other seeds round clean). bf16 is bit-exact at 512 and the default
1024-lane loop is bit-exact for both dtypes in every probe, so the
gdn_capture call site keeps 512 only for bf16. These tests pin the
contract the shipped dispatch relies on.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.kernels.fused_norm import fused_add_rmsnorm

AXIS = 5120
EPS = 1e-6


def _max_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item())


def _case(dtype, rows: int):
    mx.random.seed(11)
    weight = (mx.random.normal((AXIS,)) * 0.1 + 1.0).astype(dtype)
    x = (mx.random.normal((rows, AXIS)) * 0.5).astype(dtype)
    r = (mx.random.normal((rows, AXIS)) * 0.5).astype(dtype)
    h_ref = x + r
    n_ref = mx.fast.rms_norm(h_ref, weight, EPS).astype(dtype)
    return x, r, weight, h_ref, n_ref


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("rows", [1, 64, 65, 256])
def test_default_dispatch_is_exact(dtype, rows):
    x, r, weight, h_ref, n_ref = _case(dtype, rows)
    h, normed = fused_add_rmsnorm(x, r, weight, EPS, threadgroup_size=None)
    assert _max_diff(h, h_ref) == 0.0
    assert _max_diff(normed, n_ref) == 0.0


@pytest.mark.parametrize("rows", [1, 64, 65, 256])
def test_bf16_keeps_the_tuned_512_lane_exact(rows):
    x, r, weight, h_ref, n_ref = _case(mx.bfloat16, rows)
    h, normed = fused_add_rmsnorm(x, r, weight, EPS, threadgroup_size=512)
    assert _max_diff(h, h_ref) == 0.0
    assert _max_diff(normed, n_ref) == 0.0


@pytest.mark.parametrize("seed", [3, 7, 11])
@pytest.mark.parametrize("rows", [65, 256])
def test_fp16_default_dispatch_exact_across_seeds(seed, rows):
    mx.random.seed(seed)
    weight = (mx.random.normal((AXIS,)) * 0.1 + 1.0).astype(mx.float16)
    x = (mx.random.normal((rows, AXIS)) * 0.5).astype(mx.float16)
    r = (mx.random.normal((rows, AXIS)) * 0.5).astype(mx.float16)
    h_ref = x + r
    n_ref = mx.fast.rms_norm(h_ref, weight, EPS).astype(mx.float16)
    h, normed = fused_add_rmsnorm(x, r, weight, EPS, threadgroup_size=None)
    assert _max_diff(h, h_ref) == 0.0
    assert _max_diff(normed, n_ref) == 0.0
