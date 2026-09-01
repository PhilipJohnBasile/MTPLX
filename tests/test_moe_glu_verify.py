"""Parity gates for the M-batched fused MoE GLU verify kernels.

Two independent references so a shared structural mistake cannot self-certify:
  1. BIT-IDENTITY vs the shipped M=1 kernel per token — moe_glu_verify runs
     the same per-(token, expert, row) dot math on a bigger grid, so each
     token's output row must equal moe_glu_decode on that row exactly.
  2. Numeric envelope vs an mx.dequantize reference chain (fp32 math with
     bf16 stage-casts mirroring the kernel), catching pack-layout or index
     errors the twin kernels could inherit together.

K is a kernel constexpr (2560, the qwen4_exp hidden size); the synthetic
pack keeps E/topk/n_inter small so the gate runs in seconds on any Mac.
"""

import mlx.core as mx
import pytest

from mtplx.kernels.moe_glu_decode import moe_glu_decode, moe_glu_verify

K = 2560
E = 8
N_INTER = 64
TOPK = 3
GS = 64


def _rand_q4_pack(rows: int, cols: int, *, seed: int):
    """Random affine-q4 pack: uint32 words + bf16 scales/biases per group."""
    kfull = mx.random.uniform(
        shape=(E, rows, cols), low=-1.0, high=1.0, key=mx.random.key(seed)
    ).astype(mx.bfloat16)
    w, s, b = mx.quantize(kfull, group_size=GS, bits=4)
    return w, s, b


@pytest.fixture(scope="module")
def pack():
    gu_w, gu_s, gu_b = _rand_q4_pack(2 * N_INTER, K, seed=11)
    dn_w, dn_s, dn_b = _rand_q4_pack(K, N_INTER, seed=13)
    mx.eval(gu_w, gu_s, gu_b, dn_w, dn_s, dn_b)
    return gu_w, gu_s, gu_b, dn_w, dn_s, dn_b


def _routing(m_rows: int, seed: int):
    experts = []
    weights = []
    for m in range(m_rows):
        key = mx.random.key(seed + m)
        perm = mx.argsort(mx.random.uniform(shape=(E,), key=key))[:TOPK]
        experts.append(perm.astype(mx.uint32))
        w = mx.random.uniform(shape=(TOPK,), low=0.05, high=1.0, key=key)
        weights.append(w / w.sum())
    return mx.concatenate(experts), mx.concatenate(weights).astype(mx.float32)


@pytest.mark.parametrize("m_rows", [2, 3, 4])
def test_bit_identity_vs_m1_kernel_per_token(pack, m_rows):
    gu_w, gu_s, gu_b, dn_w, dn_s, dn_b = pack
    x2 = mx.random.normal(shape=(m_rows, K), key=mx.random.key(7)).astype(
        mx.bfloat16
    )
    experts, route_w = _routing(m_rows, seed=23)
    y_batched = moe_glu_verify(
        x2, gu_w, gu_s, gu_b, dn_w, dn_s, dn_b, experts, route_w,
        gu_group_size=GS, dn_group_size=GS,
    )
    mx.eval(y_batched)
    for m in range(m_rows):
        y_single = moe_glu_decode(
            x2[m],
            gu_w, gu_s, gu_b, dn_w, dn_s, dn_b,
            experts[m * TOPK : (m + 1) * TOPK],
            route_w[m * TOPK : (m + 1) * TOPK],
            gu_group_size=GS, dn_group_size=GS,
        )
        mx.eval(y_single)
        assert mx.array_equal(y_batched[m], y_single).item(), (
            f"token {m} of M={m_rows} diverges from the shipped M=1 kernel"
        )


def test_structure_vs_dequantized_reference(pack):
    """Layout/indexing gate: an fp32 dequantize-reference chain built from an
    INDEPENDENT derivation of the pack layout must agree structurally.

    Per-element relative error is the wrong metric here — the down dot
    cancels toward zero on random packs, so bf16 h-storage plus fp32
    accumulation-order differences (the kernel's affine s*qacc + b*xsum
    grouping vs dequantize-then-matmul) explode the ratio at near-zero
    elements while cosine stays ~1 (measured 0.999992). A transposed,
    mis-strided, or mis-indexed pack reads as cosine ~0, which is what this
    gate exists to catch; bit-level truth is pinned by the M=1 identity test
    above."""
    gu_w, gu_s, gu_b, dn_w, dn_s, dn_b = pack
    m_rows = 4
    x2 = mx.random.normal(shape=(m_rows, K), key=mx.random.key(29)).astype(
        mx.bfloat16
    )
    experts, route_w = _routing(m_rows, seed=31)
    y = moe_glu_verify(
        x2, gu_w, gu_s, gu_b, dn_w, dn_s, dn_b, experts, route_w,
        gu_group_size=GS, dn_group_size=GS,
    )
    mx.eval(y)

    gu_full = mx.dequantize(gu_w, gu_s, gu_b, group_size=GS, bits=4).astype(
        mx.float32
    )
    dn_full = mx.dequantize(dn_w, dn_s, dn_b, group_size=GS, bits=4).astype(
        mx.float32
    )
    for m in range(m_rows):
        xm = x2[m].astype(mx.float32)
        ref = mx.zeros((K,), dtype=mx.float32)
        for slot in range(TOPK):
            e = int(experts[m * TOPK + slot].item())
            gu = gu_full[e] @ xm
            gate, up = gu[:N_INTER], gu[N_INTER:]
            h = (gate * mx.sigmoid(gate)) * up
            ref = ref + float(route_w[m * TOPK + slot].item()) * (dn_full[e] @ h)
        got = y[m].astype(mx.float32)
        mx.eval(ref, got)
        cosine = (
            mx.sum(got * ref)
            / mx.sqrt(mx.sum(got * got) * mx.sum(ref * ref))
        ).item()
        assert cosine > 0.9999, f"token {m}: cosine {cosine} — layout mismatch"
        # Top-magnitude elements must be the same elements (order-insensitive).
        top_got = set(mx.argpartition(mx.abs(got), kth=-64)[-64:].tolist())
        top_ref = set(mx.argpartition(mx.abs(ref), kth=-64)[-64:].tolist())
        overlap = len(top_got & top_ref) / 64.0
        assert overlap >= 0.9, f"token {m}: top-64 overlap {overlap}"
