# Phase 0 — measured results (run 2026-07-30, M5 Max 128 GB, machine quiet)

All numbers [measured] on this machine unless noted. Conditions per arm in the
JSON files beside this summary. Engine stopped; probes run stock mlx_lm
forwards in the repo venv (no NAX, no engine overhead) — they bound physics;
engine-path arms (NAX on, `mtplx tune`) are a follow-up.

## τ gate (z-lab/Qwen3.6-27B-DFlash on stock 27B-8bit, 24 coding prompts, 256 tok)

| Arm | τ (all cycles) | τ (per-prompt) | e2e tok/s (reference impl) |
|---|---|---|---|
| greedy, block 16 | **4.31** | 4.44 | 32.1 |
| temp 0.6 / top-p 0.95 / top-k 20, block 16 | **3.92** | 4.06 | 22.4 |
| greedy, block 8 | 3.90 | 3.96 | **37.6** |

- **GATE: τ ≥ 4 greedy → GREEN. Build the DFlash lane.**
- Sampling at product settings costs only ~9% of acceptance.
- Block 8 beats block 16 e2e on the 8-bit body despite lower τ — the
  predicted kernel-regime effect, confirmed (see T_V curve below).
- Reference drafts greedily → these τ are exactly the one-hot lane's
  acceptance (the day-one integration lane). 24/24 prompts clean, all arms.

## True AR baselines + T_V(M) verify-cost curves (stock kernels, ~1k ctx, greedy)

| Body | AR tok/s | T_V(M)/T_V(1): M4 | M8 | M16 | Shape notes |
|---|---|---|---|---|---|
| Qwen3.6-27B-8bit | **17.99** | 1.04 | 1.53 | 2.29 | rows 2–4 ~free; cliff to a ~128 ms plateau at M10+ |
| Qwen3.6-35B-A3B (Optimized-Speed) | **112.4** | 1.37 | 1.84 | 2.96 | MoE fan-out tax mild vs i.i.d. ~10×-reads fear |
| Laguna-S-2.1 oQ4e | **54.97** | 1.77 | 2.30 | 3.84 | matches app-measured ~56 AR ✓; steeper MoE tax than 35B |

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
