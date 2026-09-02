"""Correctness gate for the TensorOps flash-decoding SDPA kernel (GPU required).

Compares sdpa_nax_flash against the production packed kernel and a fp32 reference at
real Qwen3.8-27B decode shapes, across the key-split / block geometries the router can
pick. Tolerances match the packed kernel's own acceptance (bf16 output, fp32 accumulate).
"""

import math
import os

import mlx.core as mx
import pytest

from mtplx.kernels.sdpa_gqa_packed import sdpa_gqa_packed_tail
from mtplx.kernels.sdpa_nax_flash import _nax_flash_kernel, sdpa_nax_flash

HQ, HKV, D = 24, 4, 256

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available() or _nax_flash_kernel() is None,
    reason="TensorOps kernel unavailable on this toolchain",
)


def _ref_tail_causal(q, k, v, offset, scale, q_len):
    """fp32 reference: row j attends to n <= offset - q_len + j."""
    qf = q.astype(mx.float32)
    kf = k[:, :, :offset, :].astype(mx.float32)
    vf = v[:, :, :offset, :].astype(mx.float32)
    gqa = q.shape[1] // k.shape[1]
    kf = mx.repeat(kf, gqa, axis=1)
    vf = mx.repeat(vf, gqa, axis=1)
    s = (qf @ kf.transpose(0, 1, 3, 2)) * scale
    n = mx.arange(offset)[None, None, None, :]
    j = mx.arange(q_len)[None, None, :, None]
    mask = n <= (offset - q_len + j)
    s = mx.where(mask, s, mx.array(-1e30, dtype=s.dtype))
    p = mx.softmax(s, axis=-1)
    return p @ vf


def _run(ctx, q_len, ks, blocks, qtg=0, seed=7):
    mx.random.seed(seed)
    cap = ctx + 192
    q = (mx.random.normal((1, HQ, q_len, D)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(q, k, v)
    scale = 1.0 / math.sqrt(D)
    env = {"MTPLX_NAX_FLASH_KS": str(ks), "MTPLX_NAX_FLASH_BLOCKS": str(blocks),
           "MTPLX_NAX_FLASH_QTG": str(qtg)}
    old = {kk: os.environ.get(kk) for kk in env}
    os.environ.update(env)
    try:
        out = sdpa_nax_flash(queries=q, keys=k, values=v, offset=ctx, scale=scale)
    finally:
        for kk, vv in old.items():
            if vv is None:
                os.environ.pop(kk, None)
            else:
                os.environ[kk] = vv
    return q, k, v, scale, out


@pytest.mark.parametrize("ctx,q_len", [
    (512, 4), (2048, 4), (4096, 4), (2048, 2), (2048, 5),
    (1023, 4),   # non-tile-aligned context
    (1000, 3),   # non-aligned + odd width
    (2048, 6),   # M=36 (3 simdgroups per key split)
    (2048, 8),   # wide-QL M=48
    (2048, 9),   # wide-QL M=54 (4 simdgroups, padded)
])
@pytest.mark.parametrize("ks,blocks", [(4, 32), (1, 64), (2, 32), (8, 32)])
def test_nax_flash_matches_reference(ctx, q_len, ks, blocks):
    q, k, v, scale, out = _run(ctx, q_len, ks, blocks)
    if HQ // HKV * q_len > 64:
        assert out is None, "M>64 must bail to fallback"
        return
    assert out is not None, "kernel bailed on a supported shape"
    mx.eval(out)

    ref = _ref_tail_causal(q, k, v, ctx, scale, q_len)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()

    packed = sdpa_gqa_packed_tail(queries=q, keys=k, values=v, offset=ctx, scale=scale)
    if packed is not None:
        mx.eval(packed)
        perr = mx.abs(packed.astype(mx.float32) - ref).max().item()
        # Must not be materially worse than the shipped kernel's own bf16
        # deviation from the fp32 reference.
        assert err <= max(2.5 * perr, 0.02), (err, perr)
    assert err < 0.05, err


def test_nax_flash_staged_queries_match():
    q, k, v, scale, out0 = _run(3000, 4, 4, 32, qtg=0)
    _, _, _, _, out1 = _run(3000, 4, 4, 32, qtg=1)
    assert out0 is not None and out1 is not None
    mx.eval(out0, out1)
    ref = _ref_tail_causal(q, k, v, 3000, scale, 4)
    e0 = mx.abs(out0.astype(mx.float32) - ref).max().item()
    e1 = mx.abs(out1.astype(mx.float32) - ref).max().item()
    assert e0 < 0.05 and e1 < 0.05, (e0, e1)


def test_nax_flash_array_offset_matches_int_offset():
    """Compiled graphs hand the kernel an int32 offset array; same result."""
    mx.random.seed(11)
    ctx, q_len = 1500, 4
    cap = ctx + 192
    q = (mx.random.normal((1, HQ, q_len, D)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((1, HKV, cap, D)) * 0.5).astype(mx.bfloat16)
    mx.eval(q, k, v)
    scale = 1.0 / math.sqrt(D)
    a = sdpa_nax_flash(queries=q, keys=k, values=v, offset=ctx, scale=scale)
    b = sdpa_nax_flash(queries=q, keys=k, values=v,
                       offset=mx.array([ctx], dtype=mx.int32), scale=scale)
    assert a is not None and b is not None
    mx.eval(a, b)
    assert mx.array_equal(a, b).item()


def test_nax_flash_empty_and_bail_paths():
    mx.random.seed(3)
    q = (mx.random.normal((1, HQ, 4, D))).astype(mx.bfloat16)
    k = (mx.random.normal((1, HKV, 256, D))).astype(mx.bfloat16)
    v = (mx.random.normal((1, HKV, 256, D))).astype(mx.bfloat16)
    mx.eval(q, k, v)
    assert sdpa_nax_flash(queries=q, keys=k, values=v, offset=0,
                          scale=0.0625) is None      # offset_range
    assert sdpa_nax_flash(queries=q.astype(mx.float32), keys=k, values=v,
                          offset=128, scale=0.0625) is None  # dtype gate
    os.environ["MTPLX_NAX_FLASH"] = "0"
    try:
        assert sdpa_nax_flash(queries=q, keys=k, values=v, offset=128,
                              scale=0.0625) is None  # kill switch
    finally:
        os.environ.pop("MTPLX_NAX_FLASH", None)
