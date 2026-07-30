# Phase 0 — measured results (run 2026-07-30, M5 Max 128 GB, machine quiet)

> **Scope, per the Codex audit (2026-07-30): reference-arm build gate GREEN;
> release/performance gate PENDING.** Every number here comes from the
> reference DFlash MLX implementation or a stock `mlx_lm` forward — not from
> MTPLX's own generation path (no NAX, no engine cache machinery, no hidden
> taps, no repair/bonus accounting). These bound the physics and justify
> building the lane; they do not establish production performance. The 4-bit
> 27B body — the intended Phase-2 prize lane — was **not** measured here.

All numbers [measured] on this machine unless noted. Conditions per arm in the
JSON files beside this summary. Engine stopped; probes run stock mlx_lm
forwards in the repo venv (no NAX, no engine overhead) — they bound physics;
engine-path arms (NAX on, `mtplx tune`) are a follow-up.

## τ gate (z-lab/Qwen3.6-27B-DFlash on stock 27B-8bit, 24 coding prompts, 256 tok)

τ here = **committed tokens per cycle** (the reference records `accepted + 1`:
matched draft tokens plus the target's replacement/bonus token — the same
convention as the published DFlash numbers). Block 16 = 1 anchor + 15 masked
draft positions, so 15 draft rows, not 16.

| Arm | τ (committed/cycle, all cycles) | τ (per-prompt mean) | e2e tok/s (reference impl) |
|---|---|---|---|
| greedy draft, block 16 | **4.31** | 4.44 | 32.1 |
| sampled draft, temp 0.6 / top-p 0.95 / top-k 20, block 16 | **3.92** | 4.06 | 22.4 |
| greedy draft, block 8 | 3.90 | 3.96 | **37.6** |

- **GATE: τ ≥ 4 greedy → reference-arm build gate GREEN. Build the DFlash
  lane (one-hot first, behind a flag). Performance/release gate pending the
  production-path arms.**
- Sampled-draft acceptance costs ~9% vs greedy. Note the MLX reference
  applies the *same sampler* to draft and target logits
  (`tools/refs/dflash/dflash/model_mlx.py:506,515`), so the temp-0.6 row is
  sampled-prefix behavior — **not** "greedy draft at product temperature."
  (The Transformers backend differs; the earlier greedy-reference claim was
  read off the wrong backend.) Greedy-draft matching is the arm that maps to
  our one-hot lane.
- Block 8 beats block 16 e2e on the 8-bit body despite lower τ — the
  predicted kernel-regime effect, confirmed (see T_V curve below).
- Quality: 24/24 prompts completed with zero runtime errors; **24 of 26
  validations passed** — both JSON-tool cases failed their JSON validator in
  every arm. Not a DFlash failure; not a clean sweep either.
- Suite bounds: per-case caps clamp generation to 160–192 tokens (not the
  256 requested), one seed per prompt, no repeat-variance or confidence
  interval, single ~1k context. Supports a coding-prompt prototype, not a
  context-independent acceptance claim.

## True AR baselines + T_V(M) verify-cost curves (stock kernels, ~1k ctx, greedy)

Rows are **identical repeated tokens** in the v1 curves. Codex audit finding:
that understates MoE expert fan-out. Re-measured with varied (real-context)
tokens where possible — see the correction row.

| Body | AR tok/s (2 runs) | T_V(M)/T_V(1): M4 | M8 | M16 | Shape notes |
|---|---|---|---|---|---|
| Qwen3.6-27B-8bit | **17.99** (17.98, 17.99) | 1.04 | 1.53 | 2.29 | dense; rows 2–4 ~free; cliff to a ~128 ms plateau at M10+ |
| Qwen3.6-35B-A3B | **112.4** (112.42, 112.22) | 1.37 | 1.84 | 2.96 | identical-token curve only (see gap below) |
| Laguna-S-2.1 oQ4e | **54.97** (54.97, 54.53) | 1.77 | 2.30 | 3.84 | identical-token; matches app-measured ~56 AR ✓ |
| Laguna-S-2.1 oQ4e — **varied tokens** | 54.18 (54.18, 53.91) | **1.88** | **2.63** | **4.40** | +14% at M8/M16 vs identical rows — Codex finding confirmed |

AR baselines are best-of-two runs; run pairs shown for spread. Laguna's
identical-token curve also carried timing outliers (M8 max 230 ms vs 42.5 ms
median) — medians are used throughout.

**Measurement gap — 35B-A3B varied-token curve not obtained.** Stock
`mlx_lm`'s GatedDeltaNet path raises `[conv] Spatial dimensions ... input
(1,3,8192), weight (8192,4,1)` whenever the concatenated conv window falls
under the kernel width of 4, and `trim_prompt_cache` does not restore GDN
conv state. Rebuilding the cache per M and skipping M∈{2,3} did not clear
it. Off-engine timing of hybrid-GDN bodies at small M is therefore
unreliable; the production path (which has `gdn_capture.py` +
`native_gdn_tail` for exactly this) is the correct instrument. Treat the
35B M16 tax as ~2.96× identical-token, likely ~3.3–3.4× with realistic
routing (scaling by Laguna's measured +14%) [estimated].

**This GDN cache/conv fragility is a direct Phase-2 task-1 datapoint:**
rejection rollback on hybrid-GDN targets cannot be a naive cache trim.
Here it crashes loudly; the dangerous version is silent state divergence.

Cross-validation: 27B-8bit curve predicts MTP D3 ≈ 2.6×; tuning.json measured
2.71× — model and reality agree within ~4%. 27B-8bit AR at ~93% of bandwidth.

## Implied speedups [estimated from measured T_V + measured/published τ]

- 27B-8bit DFlash interim: ~1.8–2.1× < MTP D3 2.71× — as designed; the M10+
  128 ms plateau is the vk M8..16 port's target (largest kernel payoff).
- 35B-A3B DFlash B8: E[commit]≈4.9 / 1.95 ≈ **2.5×** at measured-class τ.
- Laguna + poolside drafter (published τ 6.42 @ k=15): block 8 ≈ **2.0–2.2×
  (~110–120 tok/s)**, block 15 ≈ 1.8× — ABOVE the pre-measurement estimate
  for a trained MTP head; ccopy blended 1.1–1.4× stands as the no-drafter
  floor.

## Deferred, with reasons

- satgeze DSpark arm: not a config shim — needs a real MLX mini-port (causal
  anchors + Markov confidence head + gated attention + interleaved mRoPE).
  GGUF datapoint available via llama.cpp if ever needed cheaply.
- u(M) per-layer decomposition: the economic gate it served is answered by
  the T_V curves (MoE verify tax 2.96×/3.84× at M16, not ~10×).
- Engine-path arms (NAX on/off dispatch logging, `--generation-mode ar` vs
  stock, MTP D1–D3 via tune on 35B): needs the app engine; cheap to add.
- Engine-wedge forensics: engine_wedge_metrics.txt — runaway generation
  (1200 completion tokens) with zero clients; server-side disconnect
  cancellation didn't fire. Reliability bug worth filing upstream.
