"""Runtime observability (F23b): packed-GQA and NAX silent bails get
reason counters on an importable surface.

All bail paths exercised here are pure-python contract gates — no Metal
kernel is dispatched by the bailing calls themselves; the in-situ
attention test routes to the stock fused SDPA on tiny fp32 tensors.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

import mtplx.attention_split as attention_split
import mtplx.nax_verify as nax_verify
from mtplx.attention_split import configure_split_full_attention
from mtplx.kernels import sdpa_gqa_packed
from mtplx.kernels.sdpa_gqa_packed import sdpa_gqa_packed_tail
from mtplx.nax_verify import install_nax_qlinear_patch, uninstall_nax_qlinear_patch


def _delta(counts: dict, before: dict, reason: str) -> int:
    return counts.get(reason, 0) - before.get(reason, 0)


# ---------------------------------------------------------------------------
# Kernel-level contract bails (sdpa_gqa_packed_tail returns None).
# ---------------------------------------------------------------------------


def test_gqa_packed_kernel_bail_reasons_count() -> None:
    if not mx.metal.is_available():
        pytest.skip("bail-reason ordering below assumes Metal is available")
    counts = sdpa_gqa_packed.gqa_packed_bail_counts
    before = dict(counts)

    def run(**kwargs):
        defaults = dict(
            queries=mx.zeros((1, 24, 3, 64), dtype=mx.bfloat16),
            keys=mx.zeros((1, 4, 16, 64), dtype=mx.bfloat16),
            values=mx.zeros((1, 4, 16, 64), dtype=mx.bfloat16),
            offset=8,
            scale=0.125,
        )
        defaults.update(kwargs)
        return sdpa_gqa_packed_tail(**defaults)

    assert run(queries=mx.zeros((24, 3, 64), dtype=mx.bfloat16)) is None
    assert _delta(counts, before, "ndim") == 1

    assert run(queries=mx.zeros((2, 24, 3, 64), dtype=mx.bfloat16)) is None
    assert _delta(counts, before, "batch_size") == 1

    assert run(queries=mx.zeros((1, 24, 1, 64), dtype=mx.bfloat16)) is None
    assert _delta(counts, before, "q_len") == 1

    assert (
        run(
            queries=mx.zeros((1, 24, 3, 32), dtype=mx.bfloat16),
            keys=mx.zeros((1, 4, 16, 32), dtype=mx.bfloat16),
            values=mx.zeros((1, 4, 16, 32), dtype=mx.bfloat16),
        )
        is None
    )
    assert _delta(counts, before, "head_dim_unsupported") == 1

    assert (
        run(
            queries=mx.zeros((1, 24, 3, 64), dtype=mx.float32),
            keys=mx.zeros((1, 4, 16, 64), dtype=mx.float32),
            values=mx.zeros((1, 4, 16, 64), dtype=mx.float32),
        )
        is None
    )
    assert _delta(counts, before, "query_dtype") == 1

    assert run(offset=0) is None
    assert _delta(counts, before, "offset_range") == 1

    assert (
        run(values=mx.zeros((1, 4, 32, 64), dtype=mx.bfloat16)) is None
    )
    assert _delta(counts, before, "kv_layout_mismatch") == 1


def test_gqa_packed_kernel_bail_metal_unavailable(monkeypatch) -> None:
    counts = sdpa_gqa_packed.gqa_packed_bail_counts
    before = dict(counts)
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    out = sdpa_gqa_packed_tail(
        queries=mx.zeros((1, 24, 3, 64), dtype=mx.bfloat16),
        keys=mx.zeros((1, 4, 16, 64), dtype=mx.bfloat16),
        values=mx.zeros((1, 4, 16, 64), dtype=mx.bfloat16),
        offset=8,
        scale=0.125,
    )
    assert out is None
    assert _delta(counts, before, "metal_unavailable") == 1


# ---------------------------------------------------------------------------
# Route-level declines (attention_split gate) — in situ through the hook.
# ---------------------------------------------------------------------------


class _TinyProj:
    def __init__(self, out_dim: int, in_dim: int) -> None:
        self.weight = mx.zeros((out_dim, in_dim), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T


class _TinyNorm:
    def __init__(self, dim: int) -> None:
        self.weight = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x


class _TinyGatedAttention:
    num_attention_heads = 2
    num_key_value_heads = 1
    scale = 0.5

    def __init__(self, in_dim: int = 8, head_dim: int = 4) -> None:
        # q_proj emits query+gate halves: 2 * heads * head_dim rows, which
        # is what _attention_has_gated_q_proj checks against q_norm.
        self.q_proj = _TinyProj(2 * self.num_attention_heads * head_dim, in_dim)
        self.k_proj = _TinyProj(self.num_key_value_heads * head_dim, in_dim)
        self.v_proj = _TinyProj(self.num_key_value_heads * head_dim, in_dim)
        self.q_norm = _TinyNorm(head_dim)
        self.k_norm = _TinyNorm(head_dim)
        self.o_proj = lambda x: x

    def rope(self, x: mx.array, offset=0) -> mx.array:
        return x

    def __call__(self, x, mask=None, cache=None):
        # Stock path for disabled configurations; content irrelevant.
        return x


class _TinyLayer:
    is_linear = False

    def __init__(self) -> None:
        self.self_attn = _TinyGatedAttention()


class _TinyModel:
    def __init__(self) -> None:
        self.model = type("Inner", (), {"layers": [_TinyLayer()]})()


def test_gqa_packed_route_decline_counts_below_threshold(monkeypatch) -> None:
    from mlx_lm.models.cache import KVCache

    monkeypatch.delenv("MTPLX_SPLIT_FULL_ATTN", raising=False)
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_ATTN", raising=False)
    monkeypatch.delenv("MTPLX_SDPA_2PASS", raising=False)
    monkeypatch.delenv("MTPLX_BLOCKWISE_ATTN", raising=False)
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA_THRESHOLD", "8192")

    model = _TinyModel()
    stats = configure_split_full_attention(model)
    assert stats["gqa_packed_sdpa_enabled"] is True
    attn = model.model.layers[0].self_attn

    counts = attention_split.gqa_packed_route_bail_counts
    before = dict(counts)
    x = mx.zeros((1, 3, 8), dtype=mx.float32)  # q_len 3: verify-shaped
    out = attn(x, mask=None, cache=KVCache())
    assert out.shape == (1, 3, 8)
    # KVCache allocates a 256-row buffer for the 3-token window: enabled
    # verify window on a dense cache, capacity below the 8192 threshold.
    assert _delta(counts, before, "capacity_below_threshold") == 1


def test_gqa_packed_route_out_of_domain_calls_do_not_count(monkeypatch) -> None:
    from mlx_lm.models.cache import KVCache

    monkeypatch.delenv("MTPLX_SPLIT_FULL_ATTN", raising=False)
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_ATTN", raising=False)
    monkeypatch.delenv("MTPLX_SDPA_2PASS", raising=False)
    monkeypatch.delenv("MTPLX_BLOCKWISE_ATTN", raising=False)
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "1")

    model = _TinyModel()
    configure_split_full_attention(model)
    attn = model.model.layers[0].self_attn

    counts = attention_split.gqa_packed_route_bail_counts
    before = dict(counts)
    # q_len 1 (plain decode) is by-design outside the packed window.
    attn(mx.zeros((1, 1, 8), dtype=mx.float32), mask=None, cache=KVCache())
    assert dict(counts) == before

    # Lane disabled entirely: zero counting, zero route work.
    monkeypatch.setenv("MTPLX_GQA_PACKED_SDPA", "0")
    model2 = _TinyModel()
    configure_split_full_attention(model2)
    attn2 = model2.model.layers[0].self_attn
    attn2(mx.zeros((1, 3, 8), dtype=mx.float32), mask=None, cache=KVCache())
    assert dict(counts) == before


# ---------------------------------------------------------------------------
# NAX per-call silent fallbacks (patched QuantizedLinear -> stock).
# ---------------------------------------------------------------------------


def test_nax_qlinear_fallback_counts_verify_shapes_only(monkeypatch) -> None:
    # Pretend this GPU is not G17-class so the m16 NAX lane declines and a
    # verify-shaped call falls through every gate to stock.
    monkeypatch.setenv("MTPLX_FORCE_GPU_FAMILY_FALLBACK", "1")
    report = install_nax_qlinear_patch()
    assert report["installed"] is True
    counts = nax_verify.nax_qlinear_fallback_counts
    before = dict(counts)
    try:
        layer = nn.QuantizedLinear(512, 256, bias=False, group_size=64, bits=4)
        x7 = (mx.random.normal((7, 512), dtype=mx.float32) * 0.5).astype(
            mx.bfloat16
        )
        y = layer(x7)
        mx.eval(y)
        assert y.shape == (7, 256)
        assert _delta(counts, before, "b4_m7") == 1

        # Decode shape (m=1) is by-design stock: never counted.
        x1 = (mx.random.normal((1, 512), dtype=mx.float32) * 0.5).astype(
            mx.bfloat16
        )
        mx.eval(layer(x1))
        assert _delta(counts, before, "b4_m1") == 0

        # Repeat bails accumulate.
        mx.eval(layer(x7))
        assert _delta(counts, before, "b4_m7") == 2
    finally:
        uninstall_nax_qlinear_patch()
