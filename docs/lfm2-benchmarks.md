# LFM2.5-2.6B on MTPLX — benchmarks

Model: `LiquidAI/LFM2.5-2.6B-MLX`. Hardware: Apple M5 Max (40-core GPU, 128 GB).
Runtime: mlx 0.31.2 / mlx-lm 0.31.3. Measured GPU read bandwidth (STREAM):
**581 GB/s**. Architecture: dense hybrid — 22 double-gated short-conv layers + 8
GQA, hidden 2048, intermediate 10752, vocab 128000, tied embeddings.

LFM2 loads/runs through the existing mlx-lm path (no loader change). This branch
adds `mtplx/lfm2_fast.py` — a bit-exact, decode-only ShortConv fast-path that
fuses the sliding window (`concat+pad+conv1d+slice` → 3-tap FIR). "Optimized" =
this fast-path; q4 optimized also requantizes gs=64→gs=128.

All decode numbers are single-stream, greedy, batch 1.

---

## bf16

**Prefill sweep 4000→12000 — reference vs optimized**

| prefill tokens | ref prefill tok/s | ref decode tok/s | opt prefill tok/s | opt decode tok/s |
|---:|---:|---:|---:|---:|
| 4,000 | 10,162.8 | 87.98 | 10,157.5 | 87.54 |
| 6,000 | 10,157.9 | 88.06 | 10,100.4 | 87.82 |
| 8,000 | 10,103.4 | 87.70 | 10,026.2 | 87.60 |
| 10,000 | 9,805.7 | 86.62 | 9,673.9 | 87.38 |
| 12,000 | 10,066.9 | 84.85 | 9,153.2 | 87.17 |

(Short-context decode: stock 89.2 → fast-path **91.2** tok/s, bit-exact. The
fast-path's ~2% shows most at very short context; conv is decode-only so prefill
is unchanged.)

**Context sweep 1024→128k (optimized)**

| context | prefill tok/s | decode tok/s | peak GB |
|---:|---:|---:|---:|
| 1,024 | 11,729.2 | 89.57 | 6.12 |
| 32,768 | 9,804.3 | 83.66 | 6.84 |
| 65,536 | 7,833.9 | 76.69 | 7.38 |
| 98,304 | 6,326.8 | 69.76 | 7.98 |
| 131,072 | 5,180.0 | 63.22 | 8.59 |

bf16 decode is bandwidth-bound: 5.39 GB ÷ 581 GB/s = 9.28 ms = **108 tok/s**
ceiling; stock `gemv` reaches ~92% MBU → ~91 real. Ablation (remove all norms +
rope) = 94.9, so even idealized bf16 is < 100.

---

## q4

Reference = shipped 4bit (gs=64). Optimized = requant gs=128 + conv fast-path.
Dominant cost is the `affine_qmv` quantized matmul (dequant-ALU-bound).

**Prefill sweep 4000→12000 — reference vs optimized**

| prefill tokens | ref prefill tok/s | ref decode tok/s | opt prefill tok/s | opt decode tok/s |
|---:|---:|---:|---:|---:|
| 4,000 | 8,778.4 | 258.61 | 8,791.4 | 279.44 |
| 6,000 | 8,438.6 | 255.29 | 8,556.6 | 273.86 |
| 8,000 | 8,158.0 | 250.60 | 8,395.6 | 268.42 |
| 10,000 | 7,869.3 | 245.95 | 8,183.9 | 262.53 |
| 12,000 | 7,803.4 | 240.56 | 8,083.7 | 255.71 |

(Short-context decode: shipped gs=64 269.6 → gs=128+conv **296.4** tok/s.
Group size is the main lever, +8%.)

**Context sweep 1024→128k (optimized)**

| context | prefill tok/s | decode tok/s | peak GB |
|---:|---:|---:|---:|
| 1,024 | 10,902.5 | 288.77 | 2.44 |
| 32,768 | 9,292.9 | 209.69 | 2.99 |
| 65,536 | 7,363.0 | 160.86 | 3.52 |
| 98,304 | 6,068.5 | 131.04 | 4.12 |
| 131,072 | 5,069.4 | 110.71 | 4.72 |

q4 runs the full 128k context under 4.8 GB. Decode drops faster with context
than bf16 — when weights are ~1.4 GB the growing KV cache becomes a large share
of the per-token read (at 128k the KV rivals the weights). q4 ceiling ~307
(dequant-ALU-bound); single-stream can't exceed it without speculative decoding,
for which no vocab-compatible LFM2.5 draft exists (350M/1.2B use vocab 65536 vs
the 2.6B's 128000).

---

## Optimization ledger (measured)

| lever | bf16 | q4 | verdict |
|---|---|---|---|
| ShortConv fused decode | 89.2→91.2 | neutral | ✅ shipped (`lfm2_fast.py`) |
| requant gs=64→gs=128 | n/a | 270→292 | ✅ +8% (q4) |
| `mx.compile` (sub-block / shapeless / rotating) | neutral–neg | neutral | dead |
| addmm residual fusion | 91.2→91.0 | — | neutral |
| pack gate+up / q+k+v | 91.2→90.0 | 296→290 | negative |
| q3 / mixed / mxfp4 / nvfp4 | — | slower | q4 kernel is the tuned one |
| mlx 0.32 | — | 289=289 | no change |
| raw-Metal rmsnorm+gemv (coalesced, bit-exact) | −4…−11% vs stock | — | stock `gemv` optimal |

`mx.fast.metal_kernel` cannot reach MLX's `steel` simdgroup-MMA path, so no hand
gemv beats stock. Net: bf16 91 / q4 296 tok/s single-stream decode are at the
practical ceilings for this model on this hardware.
