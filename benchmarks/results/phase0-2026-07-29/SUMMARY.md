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

## 4-bit body (the Phase-2 prize lane) — added 2026-07-30

| Arm (27B **4-bit**, greedy draft, coding suite) | τ | e2e tok/s |
|---|---|---|
| block 16 | 4.29 | 36.8 |
| **block 8** | 3.98 | **48.1** |

- **τ is quant-independent**: 4.29 (4-bit) vs 4.31 (8-bit). Acceptance is a
  property of drafter/target agreement, not weight precision. The gate result
  transfers across bodies.
- **Block 8 wins on both quant widths** — +31% on 4-bit, +17% on 8-bit.
  Independently validates the fixed-block-8 choice.

### The sobering number: reference DFlash does NOT beat MTP self-speculation

Measured AR on this machine (stock `mlx_lm`, greedy, ~1k ctx): **4-bit 31.1
tok/s**, 8-bit 18.0 tok/s. Against those baselines the reference DFlash arm is:

| Body | DFlash B8 | DFlash B16 | MTP D3 [measured, tuning.json cold suite] |
|---|---|---|---|
| 27B 4-bit | **1.55×** | 1.18× | **2.23×** |
| 27B 8-bit | **2.09×** | 1.78× | **2.71×** |

Caveats both ways: the DFlash tok/s come from the coding suite (160–192-token
generations) while AR is a 512-step steady-decode probe, and the MTP figures
are a different suite/model variant — these are not matched conditions. But
same machine, same stock kernels, same model family: **the reference DFlash
implementation lands below MTP D3 on both quant widths.** Any DFlash win has
to come from what the reference does *not* have — NAX 16-row verify kernels
(4-bit only), engine cache machinery, and block-8 routing — not from the
drafter's raw acceptance. This is the strongest argument yet for Codex's
"implementation green, release/performance gate pending" framing, and for
keeping the MTP floor as the thing any DFlash arm must beat before promotion.

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

**RETRACTED (2026-07-30): the "hybrid-GDN conv blocker" was a probe bug, not
an engine defect.** The probe's prompt tokenizes to 480 tokens while the
verify rows were sliced from `[500:516]` — an empty list. Every "verify
forward" was shape `(1, 0)`; `S=0` is what raised the conv spatial-dims
error. Tell that was missed: the three runs using synthetic 1000-id prompts
completed, and only the real-tokenizer run failed — at *all sixteen* M
values. Fixed (200-rep prompt, tail slice, shape asserts); the 27B-4bit
curve below now measures cleanly at M=1..16 on a hybrid-GDN body. There is
no GDN conv problem in this range.

**The real defect, found while checking that claim: `trim_prompt_cache` is a
silent no-op on hybrid caches.** `ArraysCache` implements no `trim`, so
`can_trim_prompt_cache` returns False and `trim_prompt_cache` returns 0
**without trimming the attention layers either** — no error. Measured: after
an 8-row verify plus a 5-token trim, the attention offset was unchanged and
next-step logits differed by max |Δ| = 1.17. MTPLX production code never
calls it, and the reference DFlash port guards correctly, so **τ = 4.29 is
uncontaminated.** It remains a Phase-2 task-1 constraint: rollback on hybrid
bodies must use captured-state restore, never a trim — the failure mode is
silent divergence, not a crash.

**All curves re-measured with the fixed probe (varied real tokens, M=1..16,
zero failures on every body).** Final numbers:

| Body | AR tok/s | T_V(1) | M4/M1 | M9/M1 | M16/M1 |
|---|---|---|---|---|---|
| Qwen3.6-27B-4bit | 31.0 | 32.2 ms | 1.19 | **2.48** | 3.44 |
| Qwen3.6-27B-8bit | 18.0 | 55.5 ms | 1.05 | **2.07** | 2.35 |
| Qwen3.6-35B-A3B | 115.8 | 9.0 ms | 1.40 | **2.10** | 3.00 |
| Laguna-S-2.1 oQ4e | 54.2 | 18.5 ms | 1.88 | 2.83 | 4.40 |

The 35B varied-token curve (earlier recorded as an unobtainable gap with a
~3.3–3.4× estimate) measures **3.00× at M16** — essentially unchanged from
the identical-token 2.96×, so MoE routing diversity costs less than feared on
this body. T_V(1) tracks 1/AR on every body, which is the cross-check that
these curves are measuring what they claim.

## Verdict from the corrected curves: the Qwen DFlash lane is heading NO-GO

The verify curve explains the whole result. MTP D3 verifies **M=4** rows,
which sit in the nearly-free region; DFlash B8 verifies **M=9**, past the
cost cliff. DFlash buys ~12% more committed tokens per cycle (τ 3.98 vs
3.556) and pays roughly double the verify cost for them.

| Body | MTP D3 [measured] | DFlash B8 [measured] | DFlash B16 |
|---|---|---|---|
| 27B 4-bit | **2.32×** | 1.55× | 1.18× |
| 27B 8-bit | **2.84×** | 2.09× | 1.78× |

With NAX applied to both arms (isolated kernel savings deflated by the 0.65
in-engine factor: s4 3.6 ms, s9 18.2, s16 42.3): MTP D3 → **2.50×**,
DFlash B8 → 1.99×, B16 → 1.86× on the 4-bit body. **R = 1.99/2.50 = 0.80**,
below the 1.00 NO-GO line. NAX lifts wide blocks more in absolute ms but
cannot cover a ~41 ms verify gap bought with 0.4 extra tokens. Flipping this
needs a ~44% error in the NAX estimate — the in-engine sweep should confirm
before the decision is called final, but the direction is clear.

Note the 8-bit B8 arithmetic yields a negative implied drafter cost
(−10.9 ms), i.e. the reference's 37.6 tok/s is *faster* than this probe's
T_V(9) allows. Conditions differ (coding suite with 160–192-token
generations vs a 512-step steady-decode probe), so the two are not perfectly
matched — one more reason the in-engine sweep, not this arithmetic, is the
final instrument.

**Where the lane still makes sense: Laguna-S.** No MTP head means the floor
is 1.0×, not 2.3–2.8×. With AR 54.2 tok/s, T_V(9) = 52.4 ms and poolside's
published τ ≈ 6.4 at k=15, a block-8 lane implies roughly **1.4–1.5×**
[estimated] — a real gain with no incumbent to displace.

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

---

## FINAL: same-body, same-suite head-to-head — Qwen NO-GO confirmed at every block size

Codex's third review returned INSUFFICIENT DATA on the blanket Qwen call: the
earlier comparison was cross-suite, and blocks 4–6 (below the M=6 verify-cost
cliff) had never been tested. Both objections are now closed by measuring
every arm on **one body (Fable-711-4bit) with one prompt suite**
(`calibration_coding.jsonl`, greedy, 512 max tokens, seed 0).

| Arm | τ | tok/s | note |
|---|---|---|---|
| **MTP D5** (NAX on) | — | **57.9** | best MTP arm found |
| MTP D3 (NAX on) | — | 55.5 | today's tuned default |
| MTP D8 (NAX on) | — | 51.8 | |
| DFlash B5 | 3.05 | 52.0 | best DFlash arm |
| DFlash B4 | 2.79 | 50.8 | |
| DFlash B6 | 3.17 | 43.5 | |
| DFlash B8 | 3.37 | 40.1 | |
| MTP D15 (NAX on) | — | 29.3 | |

Codex's required-τ table (τ needed to beat MTP): B4 ~3.02, B5 ~3.84,
B6 ~4.57, B8 ~5.06. **Measured: 2.79, 3.05, 3.17, 3.37 — every block misses
its bar.** Small blocks cut verify cost but acceptance falls faster than the
savings accrue; large blocks buy acceptance at more than its worth. There is
no block size at which the z-lab drafter beats this model's own MTP heads.

**Drafter acceptance degrades on fine-tuned targets:** τ at B8 falls from
3.98 on stock `mlx-community/Qwen3.6-27B-4bit` to 3.367 on Fable-711-4bit —
the drafter was trained against stock weights. Anyone drafting for a
customized target should expect this.

## RETRACTED spin-off: "depth 5 beats depth 3" was single-run noise

The single-prompt runs showed depth 5 at 57.9 tok/s vs depth 3's 55.5 (+4.3%)
and it was flagged as needing repeats. The repeats killed it — 3 prompts per
run, interleaved ordering, same body and suite:

| depth | runs (tok/s) | mean |
|---|---|---|
| 3 | 56.29, 56.18 | 56.23 |
| 4 | 55.41 | 55.41 |
| 5 | 57.60, **55.79** | 56.70 |
| 6 | 46.53 | 46.53 |

Depth 5's own two runs disagree by 1.80 tok/s — wider than the margin it
"won" by, while depth 3's agree to 0.11. The original comparison paired
depth 5's lucky sample against depth 3's ordinary one. **Depths 3–5 are
indistinguishable at this sample size; there is no tune-range win here.**

**The depth-6 dip is real and unexplained.** 47.5 tok/s (single-prompt run)
then 46.53 (3-prompt run) — a reproducible ~17% drop at exactly M=7 verify
rows, breaking the otherwise smooth curve. Candidate causes: a kernel
dispatch boundary at M=7 (NAX m16 pads to 16; the vk path covers M 4..6), or
a graphbank bucket edge. Worth one investigation because whatever costs 17%
at M=7 may also be shaping M=8–9 costs, which is where any external drafter
has to live.

## Corrected Laguna estimate (per Codex)

The 1.4–1.5× figure stated earlier was wrong. From measured AR 54.18 tok/s
and T_V(9) = 52.4 ms with published τ 6.42: **2.26× with no drafter overhead,
≈2.14× (~116 tok/s) with ~3 ms of it** — consistent with the §10 addendum's
2.0–2.2×. Caveats: τ is published at k=15/block-16 rather than locally
measured at B8, no MLX conversion of the poolside drafter exists yet, and the
drafter's licensing is unresolved.

---

## THE REVERSAL: DFlash wins decisively on MoE (2026-07-30, protocol-matched)

The dense-body NO-GO does **not** generalize. On Qwen3.6-35B-A3B (4-bit,
NAX on), all arms greedy / 24 prompts / 256-token cap / seed 0:

| Arm | τ | tok/s | vs AR |
|---|---|---|---|
| **DFlash B8** (z-lab 386M drafter) | 3.809 | **188.22, 188.30** | **1.79×** |
| MTP D3 | — | 89.5, 106.6, 105.1 (mean 100.4) | 0.96× |
| AR (same protocol) | — | 105.1 | 1.00× |

DFlash's two runs agree to 0.04% — the tightest measurement in this
programme. Output verified non-degenerate (24/26 validations, coherent code,
per-prompt 162–234 tok/s, 20.6 GB peak).

**MTP is a net LOSS on this body** (0.96× — slower than plain autoregressive
decoding), matching the published mlx-lm finding of ~1.11× for native MTP on
this exact model: one MTP layer cannot predict expert routing.

### Why the conclusion splits by architecture

| | Dense 27B | MoE 35B-A3B |
|---|---|---|
| MTP heads | strong (2.2–2.8×) | **net loss (0.96×)** |
| T_V(9)/T_V(1) | 2.48× | **2.10×** (expert gather amortizes across rows) |
| Drafter size | 1.73B | **386M** (~4× cheaper to run) |
| Winner | **MTP** | **DFlash, ~1.9× over MTP** |

Cheap verify rows + a tiny drafter + a weak incumbent — three independent
factors all favouring DFlash, none of which hold on the dense body.

### Ecosystem bug found: z-lab's MLX backend cannot load z-lab's newer drafters

`dflash/model_mlx.py::load_draft` reads flat top-level `rope_theta` and
`block_size`, but the June-retrained 35B-A3B drafter nests them under
`rope_parameters` / `dflash_config`. Three separate `KeyError`s before it
loads. The older 27B drafter (flat config, card still says "still under
training") loads fine — which likely explains why community MLX benchmarks
all use the weaker drafter. Local clone patched to accept both formats;
worth an upstream PR.

**Revised recommendation: build the DFlash lane for MoE targets first.** The
drafter exists, needs no MLX port, has a permissive licence (Apache-2.0), and
published numbers that now reproduce locally. Laguna remains second (bigger
per-model win, but needs a trunk port and a licence answer). Dense Qwen stays
NO-GO.

### MoE block sweep — optimum is B5–B6, ~1.85× AR

| Block | τ | tok/s |
|---|---|---|
| B4 | 2.947 | 191.1 |
| **B5** | 3.271 | **193.8, 194.2** |
| **B6** | 3.519 | **193.7** |
| B8 | 3.809 | 188.2, 188.3 |
| B16 | 4.301 | 147.6 |

Same shape as the dense body — τ rises monotonically with block size while
throughput peaks early and falls — but here the peak sits **1.85× above AR**
instead of below the incumbent. B5/B6 are indistinguishable (193.7–194.2,
B5 repeat spread 0.4); B8 is ~3% behind; B16 costs 24%. Default should be
B5–B6, not the reference's 16.
