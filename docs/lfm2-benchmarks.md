# LFM2.5-2.6B on MTPLX — benchmarks & optimization ledger

Model: `LiquidAI/LFM2.5-2.6B-MLX` (bf16). Hardware: Apple M5 Max (40-core GPU,
128 GB). Runtime: mlx 0.31.2 / mlx-lm 0.31.3, MTPLX `moe/main` (2.5.2+opensourcewtf.moe).
Measured GPU read bandwidth (STREAM): **581 GB/s**. Model weights: **5.39 GB**
(dense: 22 short-conv layers + 8 GQA, hidden 2048, intermediate 10752, vocab 128000, tied embeddings).

LFM2 loads and runs through the existing mlx-lm path (no loader change). This
branch adds `mtplx/lfm2_fast.py` — a bit-exact, decode-only ShortConv fast-path
that fuses the sliding-window (`concat+pad+conv1d+slice` → 3-tap FIR).

## Single-stream decode (short context, bf16)

| build | decode tok/s |
|---|---|
| stock (`mlx_lm.benchmark`) | 89.2 |
| **+ ShortConv fast-path (this branch)** | **91.2** (+2%, bit-exact) |

## Context sweep — prefill AND decode (conv fast-path, bf16)

| context | prefill tok/s | decode tok/s | peak GB |
|---:|---:|---:|---:|
| 1,024 | 11,729 | 89.6 | 6.12 |
| 32,768 | 9,804 | 83.7 | 6.84 |
| 65,536 | 7,834 | 76.7 | 7.38 |
| 98,304 | 6,327 | 69.8 | 7.98 |
| 131,072 | 5,180 | 63.2 | 8.59 |

Prefill degrades ~O(N²) (attention over the 8 GQA layers); decode degrades with
KV growth. Runs to the full 128k context under 8.6 GB.

## Why bf16 decode caps ~91 (and 100 is unreachable at bf16)

Decode is **~99% GPU-compute-bound** (dispatch census: real steady-state idle
~1.2 ms across the whole decode; the earlier "51% idle" was a one-time
warmup→burst artifact). The floor is bandwidth: 5.39 GB ÷ 581 GB/s = 9.28 ms =
**108 tok/s** absolute; stock `gemv` reaches ~92% MBU, so real decode ≈ 91.

**Ablation ceiling** — removing all reducible overhead:

| ablation | decode tok/s |
|---|---|
| baseline + conv fast-path | 91.2 |
| − all RMSNorms (77/token) | 93.9 |
| − all RMSNorms **and** rope | **94.9** |

So even a *physically-impossible* perfect fusion of every norm and rope caps at
~95. 100 tok/s at strict bf16 is above the idealized ceiling.

## Optimization ledger (what was tried, measured)

| lever | result | verdict |
|---|---|---|
| ShortConv fused decode | 89.2 → **91.2** | ✅ shipped (`lfm2_fast.py`) |
| `mx.compile` sub-blocks | 91.8 → 92.2 | neutral |
| `mx.compile` full forward (shapeless) | fails on cache slices | dead |
| `mx.compile` full forward (shape-stable rotating KV) | 90.2 → 89.2 | negative |
| addmm residual fusion | 91.2 → 91.0 | neutral |
| pack gate+up / q+k+v (decode) | 91.2 → 90.0 | negative |
| pack gate+up / q+k+v (prefill 4k–12k) | 10.2k → 8.6k | negative |
| fused RMSNorm→matmul (compile proxy) | 90.9 → 88.1 | negative |
| raw-Metal rmsnorm+gemv (uncoalesced) | −44.5% vs stock | dead (bad shape) |
| raw-Metal gemv (coalesced, right-shaped, bit-exact) | −4…−11% vs stock `gemv` | stock is optimal |

`mx.fast.metal_kernel` cannot reach MLX's `steel` simdgroup-MMA path, so no hand
gemv beats stock — the matmul floor is fixed. The only way past ~95 is reading
fewer bytes/token (quantization: q8 ≈ 150–180, q4 ≈ 340), which matches
LiquidAI's own headline (220 tok/s at <2.5 GB = quantized, not bf16).

## q4 quantized decode (M5 Max, mlx 0.31.2)

The conv fast-path is architecture-level (bf16 conv), so it also applies to the
quantized variants. The dominant cost at q4 is the `affine_qmv` quantized
matmul, which is dequant-ALU-bound.

| q4 config | decode tok/s |
|---|---|
| shipped 4bit (gs=64), canonical `mlx_lm.benchmark` | 269.6 |
| requant gs=128 | 292.2 |
| **requant gs=128 + conv fast-path** | **296.4** |
| gs=128 + norms&rope ablated (idealized ceiling) | 307.4 |

Group size is the main q4 lever (gs=64→128 halves scale reads: +8%). mlx 0.32
gives no change (289 both). q3/mixed/mxfp4/packing all regress. The `affine_qmv`
is stock-optimal (~73% MBU of the 406 bandwidth ceiling), so single-stream q4
caps at ~296 practical / ~307 idealized. Exceeding that needs speculative
decoding (multiple tokens per target forward), not a faster single forward.

## q4 context sweep — prefill AND decode (gs=128, conv fast-path)

| context | prefill tok/s | decode tok/s | peak GB |
|---:|---:|---:|---:|
| 1,024 | 10,902 | 288.8 | 2.44 |
| 32,768 | 9,293 | 209.7 | 2.99 |
| 65,536 | 7,363 | 160.9 | 3.52 |
| 98,304 | 6,069 | 131.0 | 4.12 |
| 131,072 | 5,069 | 110.7 | 4.72 |

q4 runs the full 128k context under 4.8 GB (vs bf16's 6.1–8.6 GB). Decode falls
off faster with context than bf16: when the weights are only ~1.4 GB, the
growing KV cache becomes a large share of the per-token read (at 128k the KV
rivals the weights).

## Summary — bf16 vs q4 (M5 Max, conv fast-path)

| | short-ctx decode | peak GB @128k | notes |
|---|---|---|---|
| bf16 | 91 tok/s | 8.6 | bandwidth-bound, ceiling ~108 |
| q4 gs=128 | 296 tok/s | 4.7 | dequant-ALU-bound, ceiling ~307 |
