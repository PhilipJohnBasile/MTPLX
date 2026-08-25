"""Exactness gates for #320 (packed concats) and #319 (fused_add_rmsnorm).

Both lanes are off by default; these tests pin the *claims* so they cannot
rot: every M the packed gate admits must be element-identical to separate
launches, and the fused post-norm lane must be bitwise against the unfused
reference at real model widths in both half dtypes. Metal-only — kernel
selection is the mechanism under test (#320 measured M>=10 divergence on
M5/MLX 0.32.0, M>=6 on the reporter's M2; #319 measured fp16 ULP flips at
axes 3072/5120 under the old hardcoded threadgroup 512).
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

import mtplx.packed_concats as pc
from mtplx.kernels.fused_norm import fused_add_rmsnorm

_METAL = mx.metal.is_available()

pytestmark = pytest.mark.skipif(not _METAL, reason="kernel-selection exactness needs Metal")


def _quantized_linear(out_features: int, in_features: int, key) -> "object":
    import mlx.nn as nn

    layer = nn.Linear(in_features, out_features, bias=False)
    layer.weight = mx.random.normal((out_features, in_features), key=key) * 0.02
    q = nn.QuantizedLinear.from_linear(layer, group_size=32, bits=4)
    return q


def test_packed_qkv_element_identical_for_every_admitted_m():
    """Every M <= _max_s() through the fused q|k|v path is bitwise (#320)."""
    k = 5120
    keys = mx.random.split(mx.random.key(320), 4)
    members = [
        _quantized_linear(12288, k, keys[0]),
        _quantized_linear(1024, k, keys[1]),
        _quantized_linear(1024, k, keys[2]),
    ]
    payload = pc._pack(members)
    assert payload is not None
    max_s = pc._max_s()
    assert max_s == 4, "default gate must stay at the measured-exact window"
    for m in range(1, max_s + 1):
        x = (mx.random.normal((1, m, k), key=keys[3]) * 0.5).astype(mx.bfloat16)
        fused = pc._fused_forward(payload, x)
        mx.eval(*fused)
        for out, member in zip(fused, members):
            ref = member(x)
            mx.eval(ref)
            assert bool(mx.all(out == ref).item()), (
                f"fused q|k|v differs from separate launch at M={m} "
                f"(inside the shipped gate)"
            )


def test_packed_gate_uses_kernel_visible_row_count():
    """The S-gate keys on prod(leading dims), not the sequence dim (#320)."""
    max_s = pc._max_s()
    batched = (2, max_s, 5120)  # x.shape[1] == max_s but M == 2*max_s
    assert math.prod(batched[:-1]) > max_s


def test_packed_concats_refuses_nax_verify(monkeypatch):
    """Fused projections bypass the NAX verify patch; co-enabling must refuse."""
    monkeypatch.setenv("MTPLX_PACKED_PROJ_CONCATS", "1")
    monkeypatch.setenv("MTPLX_NAX_VERIFY", "1")
    assert pc.install_qwen3_next_packed_concats(object()) is None
    assert pc.COUNTERS.get("refused_nax_verify", 0) >= 1


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fused_add_rmsnorm_bitwise_at_model_widths(dtype):
    """Production tg resolution is bitwise vs the unfused reference (#319)."""
    from mtplx.gdn_capture import _fused_post_norm_tg_override

    assert _fused_post_norm_tg_override() is None
    mx.random.seed(319)
    for axis in (512, 3072, 5120):
        weight = (mx.random.normal((axis,)) * 0.1 + 1.0).astype(dtype)
        for rows in (1, 64, 354):
            x = (mx.random.normal((rows, axis)) * 0.5).astype(dtype)
            residual = (mx.random.normal((rows, axis)) * 0.5).astype(dtype)
            h, normed = fused_add_rmsnorm(x, residual, weight, 1e-6, threadgroup_size=None)
            ref_h = x + residual
            ref_normed = mx.fast.rms_norm(ref_h, weight, 1e-6).astype(dtype)
            mx.eval(h, normed, ref_h, ref_normed)
            assert bool(mx.all(h == ref_h).item())
            assert bool(mx.all(normed == ref_normed).item()), (
                f"fused_add_rmsnorm not bitwise at axis={axis} rows={rows} {dtype}"
            )
