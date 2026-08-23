"""Cross-row wide QMV with activation chunk-sum tables (verify shapes).

Ported from the mlxfast arena E120 family (Layr-Labs/qwen-3.8-mtp-challenge,
MIT) onto ``mx.fast.metal_kernel``. Lineage and ranked receipts on the same
hardware/model class (M5 Max, Qwen3.8-27B, affine-4):
  - chunk-sum table hoist: morganmcg1/senpai 6f1cd66, official +13.24%;
  - tight launch geometry: jungjipdo 8849fad, +15.49% (this port launches
    tight by construction — exactly ceil(N/8) output threadgroups, one input
    group for M<=5);
  - M=2 extension: scarletbright e8f14c4 (+1.28%, the 270.4% record).

Form: one simdgroup owns 4 output rows x M input rows. Integer nibble dot +
per-group affine correction ``acc = scale * qdot + bias * sum(x_slice)``.
The per-lane activation slice sums depend only on (activation, k-block,
lane) — never the output row — so they can be computed once per activation
into a table read by every consuming matvec instead of recomputed for every
8-row output block (N/8 recomputes at N=16480 on the stock form).

The table and inline variants accumulate the slice sum in the identical
order (four float adds of one bf16 vec4 per inner step), so table-on and
table-off are bitwise-equal by construction — asserted in tests, with a
perturbed-table positive control proving the comparison can fail.

Numerics vs stock ``mx.quantized_matmul``: fp32 accumulate, lane-strided K —
tail-ULP class differences, same contract as every custom verify kernel
(verify_kernels.py); gated by the identity/R1b corpus before any routing.

Scope v1: 4-bit affine, group_size in {32, 64}, bf16/fp16 activations,
M in {2, 3, 4, 5} (one input group), K % 512 == 0, N % 8 == 0,
row-contiguous inputs. Everything else belongs to the caller's fallback.
Default OFF: nothing routes here until MTPLX_VK_CROSSROW=1 and the gates.
"""

from __future__ import annotations

import os

import mlx.core as mx

_KERNELS: dict[tuple, object] = {}

# One float per (k-block, lane, input-row): stride 8 covers M <= 8 and keeps
# the row stride a cache line of 4-byte floats (the arena's layout).
_SUMS_STRIDE = 8
_BLOCK = 512  # values per k-block: 16 per lane x 32 lanes


def crossrow_enabled() -> bool:
    return str(os.environ.get("MTPLX_VK_CROSSROW", "")).strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def crossrow_eligible(m: int, k: int, n: int, bits: int, group_size: int, dtype) -> bool:
    return (
        2 <= m <= 5
        and bits == 4
        and group_size in (32, 64)
        and k % _BLOCK == 0
        and n % 8 == 0
        and dtype in (mx.bfloat16, mx.float16)
    )


def _dtype_name(dtype: mx.Dtype) -> str:
    return "bfloat16_t" if dtype == mx.bfloat16 else "half"


def _sums_kernel(dtype: mx.Dtype):
    """Fill xsums[(k_block*32 + lane) * stride + m] with the lane's 16-value
    slice sum of row m, in the main kernel's exact accumulation order."""

    key = ("sums", dtype)
    if key in _KERNELS:
        return _KERNELS[key]
    t = _dtype_name(dtype)
    src = f"""
        const int M = x_shape[x_ndim - 2];
        const int K = x_shape[x_ndim - 1];
        const int slot = int(thread_position_in_grid.x);
        const int lanes = (K / {_BLOCK}) * 32;
        if (slot >= lanes) return;
        const int kb = slot / 32;
        const int lid = slot % 32;
        const int base = kb * {_BLOCK} + lid * 16;
        for (int m = 0; m < M; ++m) {{
            const device {t}* xm = x + m * K + base;
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {{
                const device vec<{t}, 4>* xv =
                    reinterpret_cast<const device vec<{t}, 4>*>(xm + 4 * i);
                acc += float((*xv)[0]) + float((*xv)[1]) +
                       float((*xv)[2]) + float((*xv)[3]);
            }}
            xsums[slot * {_SUMS_STRIDE} + m] = acc;
        }}
    """
    kernel = mx.fast.metal_kernel(
        name=f"mtplx_crossrow_sums_{t}",
        input_names=["x"],
        output_names=["xsums"],
        source=src,
    )
    _KERNELS[key] = kernel
    return kernel


def make_chunk_sums(x: mx.array) -> mx.array:
    """Per-lane activation slice sums for ``crossrow_qmm(..., xsums=...)``.

    Compute once per activation tensor; every consuming matvec of the same x
    reads it instead of re-forming the sums per 8-row output block.
    """

    m, k = int(x.shape[-2]), int(x.shape[-1])
    lanes = (k // _BLOCK) * 32
    kernel = _sums_kernel(x.dtype)
    (out,) = kernel(
        inputs=[x],
        grid=(lanes, 1, 1),
        threadgroup=(min(lanes, 256), 1, 1),
        output_shapes=[(lanes * _SUMS_STRIDE,)],
        output_dtypes=[mx.float32],
    )
    return out


def _qmm_source(m: int, group_size: int, dtype: mx.Dtype, use_table: bool) -> str:
    t = _dtype_name(dtype)
    # Lane's 16 values sit inside one quant group for both sizes:
    # g64 -> 4 lanes per group (lid/4), g32 -> 2 lanes per group (lid/2).
    lanes_per_group = group_size // 16
    table_read = (
        f"""
            const device float* st =
                xsums + ((k / {_BLOCK}) * 32 + int(lid)) * {_SUMS_STRIDE};
            for (int mm = 0; mm < {m}; ++mm) sums[mm] = st[mm];
        """
        if use_table
        else ""
    )
    inline_sum = (
        ""
        if use_table
        else "sums[mm] += float(xv[0]) + float(xv[1]) + float(xv[2]) + float(xv[3]);"
    )
    src = f"""
        const int K = x_shape[x_ndim - 1];
        const int N = w_shape[0];
        const uint3 tpos = threadgroup_position_in_grid;
        const uint lid = thread_index_in_simdgroup;
        const uint sgid = simdgroup_index_in_threadgroup;
        const int out_row = int(tpos.y) * 8 + int(sgid) * 4;
        if (out_row >= N) return;
        const int Kw = K / 2;              // bytes per weight row (4-bit)
        const int Kg = K / {group_size};   // groups per row

        float acc[4][{m}];
        for (int r = 0; r < 4; ++r)
            for (int mm = 0; mm < {m}; ++mm) acc[r][mm] = 0.0f;

        for (int k = 0; k < K; k += {_BLOCK}) {{
            thread uint16_t packed[4][4];
            thread float scale_local[4];
            thread float bias_local[4];
            for (int r = 0; r < 4; ++r) {{
                const int row = out_row + r;
                const device uint16_t* ws =
                    reinterpret_cast<const device uint16_t*>(
                        reinterpret_cast<const device uint8_t*>(w) +
                        row * Kw + k / 2 + lid * 8);
                for (int i = 0; i < 4; ++i) packed[r][i] = ws[i];
                const int gi = row * Kg + k / {group_size} +
                    int(lid) / {lanes_per_group};
                scale_local[r] = float(scales[gi]);
                bias_local[r] = float(biases[gi]);
            }}

            float sums[{m}];
            for (int mm = 0; mm < {m}; ++mm) sums[mm] = 0.0f;
            {table_read}
            float partial[4][{m}];
            for (int r = 0; r < 4; ++r)
                for (int mm = 0; mm < {m}; ++mm) partial[r][mm] = 0.0f;

            for (int i = 0; i < 4; ++i) {{
                float a0[{m}], a1[{m}], a2[{m}], a3[{m}];
                for (int mm = 0; mm < {m}; ++mm) {{
                    const device vec<{t}, 4>* xv4 =
                        reinterpret_cast<const device vec<{t}, 4>*>(
                            x + mm * K + k + lid * 16 + 4 * i);
                    const vec<{t}, 4> xv = *xv4;
                    a0[mm] = float(xv[0]);
                    a1[mm] = float(xv[1]);
                    a2[mm] = float(xv[2]);
                    a3[mm] = float(xv[3]);
                    {inline_sum}
                }}
                for (int r = 0; r < 4; ++r) {{
                    const uint16_t p = packed[r][i];
                    for (int mm = 0; mm < {m}; ++mm) {{
                        partial[r][mm] +=
                            a0[mm] * float(p & 0x000f) +
                            a1[mm] * float((p >> 4) & 0x000f) +
                            a2[mm] * float((p >> 8) & 0x000f) +
                            a3[mm] * float((p >> 12) & 0x000f);
                    }}
                }}
            }}
            for (int r = 0; r < 4; ++r)
                for (int mm = 0; mm < {m}; ++mm)
                    acc[r][mm] += scale_local[r] * partial[r][mm] +
                                  sums[mm] * bias_local[r];
        }}

        for (int r = 0; r < 4; ++r) {{
            for (int mm = 0; mm < {m}; ++mm) {{
                const float reduced = simd_sum(acc[r][mm]);
                if (lid == 0) {{
                    y[mm * N + out_row + r] = static_cast<{t}>(reduced);
                }}
            }}
        }}
    """
    return src


def _qmm_kernel(m: int, group_size: int, dtype: mx.Dtype, use_table: bool):
    key = ("qmm", m, group_size, dtype, use_table)
    if key in _KERNELS:
        return _KERNELS[key]
    t = _dtype_name(dtype)
    inputs = ["x", "w", "scales", "biases"]
    if use_table:
        inputs.append("xsums")
    kernel = mx.fast.metal_kernel(
        name=f"mtplx_crossrow_qmm_m{m}_g{group_size}_{t}_{'tab' if use_table else 'notab'}",
        input_names=inputs,
        output_names=["y"],
        source=_qmm_source(m, group_size, dtype, use_table),
    )
    _KERNELS[key] = kernel
    return kernel


def crossrow_qmm(
    x: mx.array,
    w_q: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int,
    xsums: mx.array | None = None,
) -> mx.array:
    """M-row affine-4 QMV, tight launch, optional chunk-sum table."""

    m, k = int(x.shape[-2]), int(x.shape[-1])
    n = int(w_q.shape[0])
    kernel = _qmm_kernel(m, group_size, x.dtype, xsums is not None)
    inputs = [mx.contiguous(x.reshape(m, k)), w_q, scales, biases]
    if xsums is not None:
        inputs.append(xsums)
    (y,) = kernel(
        inputs=inputs,
        grid=(32, (n // 8) * 2, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )
    return y
