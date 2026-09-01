"""Fused routed-expert GLU decode kernels for qwen4_exp (M=1).

The step's MoE cost (8.0ms of 21.9, ablation 2026-08-27) is dependency-gap
serialization: the per-layer chain (gate_up -> silu*mul -> down -> weighted
sum) queues ~5 dependent kernels whose boundaries each pay scheduler
latency, while the kernels themselves already run at 68-99%% of wire when
back-to-back (microbench receipt). These two kernels collapse the chain to
TWO dispatches per layer:

  A) moe_glu_h:   h[e, j]  = silu(gate_e . x) * (up_e . x)   (fused gu pack)
  B) moe_down_y:  y[d]     = sum_e w_e * (down_e[d, :] . h[e, :])

Weights are the sanitize-fused 4-bit affine gu pack ([E, 2N, K/8] uint32,
scales/biases [E, 2N, K/gs]) and the stock down pack. bits=4 only;
group_size is a kernel template (32 and 64 are the shipped forges — the
2026-08-27 01:35 reforge moved the family pack from g32 to g64 and silently
disarmed the g32-hardcoded first cut of these kernels, so the group size now
travels with the pack). Anything else falls back to the stock path. Built
in-house on the affine dequant contract (w = scale*q + bias, little-endian
nibbles), with mlx's quantized.h as the reference for the packing order
(Apache-2.0, upstream dependency)."""

from functools import lru_cache

import mlx.core as mx

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# One simdgroup per (expert, inter-row): dots gate row j and up row N+j over
# K, groups strided lane -> lane+32. BITS=4 hardcoded; GS is a template.
_SRC_A = """
    constexpr int K = 2560;
    constexpr int GS = GS_GU;                 // quant group size (32/64 forges)
    constexpr int NGROUPS = K / GS;
    constexpr int WPG = GS / 8;               // uint32 words per group (8 nibbles/word)

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint gsid = threadgroup_position_in_grid.x * 32 + sg;  // global simd id
    const int n_inter = n_inter_c;
    const int topk = topk_c;
    if (gsid >= (uint)(topk * n_inter)) return;
    const int e_slot = gsid / n_inter;
    const int j = gsid % n_inter;
    const uint e = experts[e_slot];

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int hh = 0; hh < 2; ++hh) {
        const int row = hh * n_inter + j;
        const device uint32_t* wrow = gw + ((size_t)e * 2 * n_inter + row) * (K / 8);
        const device T* srow = gs + ((size_t)e * 2 * n_inter + row) * NGROUPS;
        const device T* brow = gb + ((size_t)e * 2 * n_inter + row) * NGROUPS;
        float acc = 0.0f;
        for (int g = lane; g < NGROUPS; g += 32) {
            const float s = (float)srow[g];
            const float b = (float)brow[g];
            const device uint32_t* wg = wrow + g * WPG;
            const device T* xg = x + g * GS;
            float qacc = 0.0f;
            float xsum = 0.0f;
            for (int wi = 0; wi < WPG; ++wi) {
                uint32_t word = wg[wi];
                const device T* xv = xg + wi * 8;
                for (int nib = 0; nib < 8; ++nib) {
                    const float xf = (float)xv[nib];
                    qacc += (float)((word >> (4 * nib)) & 0xF) * xf;
                    xsum += xf;
                }
            }
            acc += s * qacc + b * xsum;
        }
        acc = simd_sum(acc);
        if (hh == 0) acc_g = acc; else acc_u = acc;
    }
    if (lane == 0) {
        const float gv = (float)((T)acc_g);
        const float uv = (float)((T)acc_u);
        const float sw = gv / (1.0f + metal::exp(-gv));   // silu
        h[(size_t)e_slot * n_inter + j] = (T)((float)((T)sw) * uv);
    }
"""

# One simdgroup per output dim d: for each expert, dot down_e[d, :n_inter]
# with h[e], scale by router weight, accumulate. Lane-strided over k with the
# group index derived per value, so the same body serves any GS.
_SRC_B = """
    constexpr int GS = GS_DN;                 // quant group size (32/64 forges)

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint d = threadgroup_position_in_grid.x * 32 + sg;
    const int dmodel = dmodel_c;
    const int n_inter = n_inter_c;
    const int topk = topk_c;
    if (d >= (uint)dmodel) return;

    const int ngroups = n_inter / GS;
    float acc = 0.0f;
    for (int e_slot = 0; e_slot < topk; ++e_slot) {
        const uint e = experts[e_slot];
        const device uint32_t* wrow = dw + ((size_t)e * dmodel + d) * (n_inter / 8);
        const device T* srow = ds + ((size_t)e * dmodel + d) * ngroups;
        const device T* brow = db + ((size_t)e * dmodel + d) * ngroups;
        const device T* he = h + (size_t)e_slot * n_inter;
        float dot = 0.0f;
        for (int k = (int)lane; k < n_inter; k += 32) {
            const uint32_t word = wrow[k / 8];
            const float q = (float)((word >> (4 * (k % 8))) & 0xF);
            const int g = k / GS;
            dot = metal::fma((float)srow[g] * q + (float)brow[g], (float)he[k], dot);
        }
        dot = simd_sum(dot);
        acc += (float)rw[e_slot] * (float)((T)dot);
    }
    if (lane == 0) {
        y[d] = (T)acc;
    }
"""


# ---- M-batched verify variants (M = 2..4, the MTP verify widths) ----------
#
# FALSIFIED 2026-08-31 (like the M=1 pair before it, g64 pack, -5% AR): the
# serve ABBA quad on Flash-Next Bare-Speed read fused -16.7%/-6.9%
# (prose/code, pair 1) and -8.7%/-13.3% (pair 2) vs the stock gather chain
# at greedy — the scalar nibble-unpack inner loop loses more to MLX's tuned
# gather_qmm than the removed dispatch gaps recover, even at 4x grid fill.
# Receipts: SPEEDWAR-20260831/moe-verify-ab/. The lane stays dark
# (MTPLX_FUSED_MOE_VERIFY) as the wiring platform for a future fast inner
# loop (vectorized loads / simdgroup tiles / #242-style int8 tiling); do
# not re-arm without a new inner-loop hypothesis.
#
# Original rationale: the M=1 pair above removed the dependency-gap
# serialization from AR decode (8.0 ms of the 21.9 ms step, ablation
# 2026-08-27). The M=4 verify forward pays the same gaps inside the 23 ms
# verify_hidden_eval wall (round anatomy 2026-08-31). These variants run
# the SAME per-(token, expert, row) dot math — same group striding, same
# fma order, same simd_sum — so each token's rows are bit-identical to the
# M=1 kernel applied per token; only the grid grows (M*topk*n_inter
# simdgroups vs N=640 gathers).

_SRC_A_M = """
    constexpr int K = 2560;
    constexpr int GS = GS_GU;                 // quant group size (32/64 forges)
    constexpr int NGROUPS = K / GS;
    constexpr int WPG = GS / 8;               // uint32 words per group (8 nibbles/word)

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint gsid = threadgroup_position_in_grid.x * 32 + sg;  // global simd id
    const int n_inter = n_inter_c;
    const int topk = topk_c;
    const int m_rows = m_rows_c;
    if (gsid >= (uint)(m_rows * topk * n_inter)) return;
    const int m = gsid / (topk * n_inter);
    const int rem = gsid % (topk * n_inter);
    const int e_slot = rem / n_inter;
    const int j = rem % n_inter;
    const uint e = experts[m * topk + e_slot];
    const device T* xm = x + (size_t)m * K;

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int hh = 0; hh < 2; ++hh) {
        const int row = hh * n_inter + j;
        const device uint32_t* wrow = gw + ((size_t)e * 2 * n_inter + row) * (K / 8);
        const device T* srow = gs + ((size_t)e * 2 * n_inter + row) * NGROUPS;
        const device T* brow = gb + ((size_t)e * 2 * n_inter + row) * NGROUPS;
        float acc = 0.0f;
        for (int g = lane; g < NGROUPS; g += 32) {
            const float s = (float)srow[g];
            const float b = (float)brow[g];
            const device uint32_t* wg = wrow + g * WPG;
            const device T* xg = xm + g * GS;
            float qacc = 0.0f;
            float xsum = 0.0f;
            for (int wi = 0; wi < WPG; ++wi) {
                uint32_t word = wg[wi];
                const device T* xv = xg + wi * 8;
                for (int nib = 0; nib < 8; ++nib) {
                    const float xf = (float)xv[nib];
                    qacc += (float)((word >> (4 * nib)) & 0xF) * xf;
                    xsum += xf;
                }
            }
            acc += s * qacc + b * xsum;
        }
        acc = simd_sum(acc);
        if (hh == 0) acc_g = acc; else acc_u = acc;
    }
    if (lane == 0) {
        const float gv = (float)((T)acc_g);
        const float uv = (float)((T)acc_u);
        const float sw = gv / (1.0f + metal::exp(-gv));   // silu
        h[((size_t)m * topk + e_slot) * n_inter + j] = (T)((float)((T)sw) * uv);
    }
"""

_SRC_B_M = """
    constexpr int GS = GS_DN;                 // quant group size (32/64 forges)

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint gid = threadgroup_position_in_grid.x * 32 + sg;
    const int dmodel = dmodel_c;
    const int n_inter = n_inter_c;
    const int topk = topk_c;
    const int m_rows = m_rows_c;
    if (gid >= (uint)(m_rows * dmodel)) return;
    const int m = gid / dmodel;
    const uint d = gid % dmodel;

    const int ngroups = n_inter / GS;
    float acc = 0.0f;
    for (int e_slot = 0; e_slot < topk; ++e_slot) {
        const uint e = experts[m * topk + e_slot];
        const device uint32_t* wrow = dw + ((size_t)e * dmodel + d) * (n_inter / 8);
        const device T* srow = ds + ((size_t)e * dmodel + d) * ngroups;
        const device T* brow = db + ((size_t)e * dmodel + d) * ngroups;
        const device T* he = h + ((size_t)m * topk + e_slot) * n_inter;
        float dot = 0.0f;
        for (int k = (int)lane; k < n_inter; k += 32) {
            const uint32_t word = wrow[k / 8];
            const float q = (float)((word >> (4 * (k % 8))) & 0xF);
            const int g = k / GS;
            dot = metal::fma((float)srow[g] * q + (float)brow[g], (float)he[k], dot);
        }
        dot = simd_sum(dot);
        acc += (float)rw[m * topk + e_slot] * (float)((T)dot);
    }
    if (lane == 0) {
        y[(size_t)m * dmodel + d] = (T)acc;
    }
"""


@lru_cache(maxsize=2)
def _kernel_a():
    return mx.fast.metal_kernel(
        name="mtplx_moe_glu_h",
        input_names=["x", "gw", "gs", "gb", "experts"],
        output_names=["h"],
        header=_HEADER,
        source=_SRC_A,
    )


@lru_cache(maxsize=2)
def _kernel_a_m():
    return mx.fast.metal_kernel(
        name="mtplx_moe_glu_h_m",
        input_names=["x", "gw", "gs", "gb", "experts"],
        output_names=["h"],
        header=_HEADER,
        source=_SRC_A_M,
    )


@lru_cache(maxsize=2)
def _kernel_b_m():
    return mx.fast.metal_kernel(
        name="mtplx_moe_down_y_m",
        input_names=["h", "dw", "ds", "db", "experts", "rw"],
        output_names=["y"],
        header=_HEADER,
        source=_SRC_B_M,
    )


@lru_cache(maxsize=2)
def _kernel_b():
    return mx.fast.metal_kernel(
        name="mtplx_moe_down_y",
        input_names=["h", "dw", "ds", "db", "experts", "rw"],
        output_names=["y"],
        header=_HEADER,
        source=_SRC_B,
    )


def moe_glu_decode(
    x, gu_w, gu_s, gu_b, dn_w, dn_s, dn_b, experts, route_w,
    *, gu_group_size=32, dn_group_size=32,
):
    """x [2560] bf16; gu pack [E, 2*Ni, ...]; down pack [E, 2560, ...];
    experts uint32 [topk]; route_w float32 [topk]. Returns y [2560].
    Group sizes travel with the pack (32/64 forges); bits stay 4."""
    n_inter = gu_w.shape[1] // 2
    topk = experts.shape[0]
    dmodel = dn_w.shape[1]
    if 2560 % gu_group_size or n_inter % dn_group_size:
        raise ValueError(
            f"group sizes must divide the dot lengths (gu {gu_group_size}, "
            f"dn {dn_group_size} over n_inter {n_inter})"
        )
    ka = _kernel_a()
    rows = topk * n_inter
    tgs = (rows + 31) // 32
    (h,) = ka(
        inputs=[x, gu_w, gu_s, gu_b, experts],
        template=[
            ("T", x.dtype),
            ("n_inter_c", n_inter),
            ("topk_c", topk),
            ("GS_GU", int(gu_group_size)),
        ],
        grid=(tgs * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(topk, n_inter)],
        output_dtypes=[x.dtype],
    )
    kb = _kernel_b()
    tgs_b = (dmodel + 31) // 32
    (y,) = kb(
        inputs=[h, dn_w, dn_s, dn_b, experts, route_w],
        template=[
            ("T", x.dtype),
            ("dmodel_c", dmodel),
            ("n_inter_c", n_inter),
            ("topk_c", topk),
            ("GS_DN", int(dn_group_size)),
        ],
        grid=(tgs_b * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(dmodel,)],
        output_dtypes=[x.dtype],
    )
    return y


def moe_glu_verify(
    x2, gu_w, gu_s, gu_b, dn_w, dn_s, dn_b, experts, route_w,
    *, gu_group_size=32, dn_group_size=32,
):
    """M-batched fused MoE GLU for the verify widths.

    x2 [M, 2560] bf16 (M = 2..4); experts uint32 [M*topk] (token-major);
    route_w float32 [M*topk]. Returns y [M, 2560]. Each token's rows are
    computed with the exact per-row math of moe_glu_decode, so a token's
    output is bit-identical to running the M=1 kernel on its row alone.
    """
    m_rows = int(x2.shape[0])
    n_inter = gu_w.shape[1] // 2
    topk = experts.shape[0] // m_rows
    dmodel = dn_w.shape[1]
    if 2560 % gu_group_size or n_inter % dn_group_size:
        raise ValueError(
            f"group sizes must divide the dot lengths (gu {gu_group_size}, "
            f"dn {dn_group_size} over n_inter {n_inter})"
        )
    ka = _kernel_a_m()
    rows = m_rows * topk * n_inter
    tgs = (rows + 31) // 32
    (h,) = ka(
        inputs=[x2, gu_w, gu_s, gu_b, experts],
        template=[
            ("T", x2.dtype),
            ("n_inter_c", n_inter),
            ("topk_c", topk),
            ("m_rows_c", m_rows),
            ("GS_GU", int(gu_group_size)),
        ],
        grid=(tgs * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(m_rows, topk, n_inter)],
        output_dtypes=[x2.dtype],
    )
    kb = _kernel_b_m()
    tgs_b = (m_rows * dmodel + 31) // 32
    (y,) = kb(
        inputs=[h, dn_w, dn_s, dn_b, experts, route_w],
        template=[
            ("T", x2.dtype),
            ("dmodel_c", dmodel),
            ("n_inter_c", n_inter),
            ("topk_c", topk),
            ("m_rows_c", m_rows),
            ("GS_DN", int(dn_group_size)),
        ],
        grid=(tgs_b * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(m_rows, dmodel)],
        output_dtypes=[x2.dtype],
    )
    return y
