# MTP + DFlash coordination — the "Priced Union" design

Status: measured build plan, corrected 2026-07-30. Produced by a 13-agent workflow (6 code/web mappers, 3
competing designs, judge synthesis, 3 adversarial verifiers). All three verifiers
returned SOUND_WITH_FIXES; every fix is folded into this document. Number tags:
**[measured]** = observed on this M5 Max or in local records, **[verified]** = code
read in the current checkout, **[published]** = external claim, **[estimated]** =
model/extrapolation.

## Current decision

Phase 0 changed the implementation order. Under the final protocol-matched
thinking-off runs:

| Target | Best DFlash | Best MTP | Decision |
|---|---:|---:|---|
| Qwen3.6-35B-A3B MoE | **2.07x, B8, 217.1 tok/s** | 1.05x, D1, 111.0 tok/s | Build DFlash first |
| Qwen3.6-27B dense | 1.89x, B5, 58.6 tok/s | **2.23-2.32x, D3** | Park current drafter |

The reference-arm build gate has now produced an in-engine implementation,
but **not a release candidate**. Two MoE lanes were measured on the pinned
35B-A3B target and drafter:

| MTPLX lane | Weighted result | Equality result | Decision |
|---|---:|---:|---|
| Wide B8, one-hot, capture-commit | 130.6 tok/s, 1.24x AR | DFlash = AR on 9/24; 8/24 three-way exact | Keep experimental |
| Staged-K1 B8 over compiled target-prefix + whole-MoE | 170.0 tok/s, 1.30x AR | DFlash = MTP K1 on 24/24; both = AR on 13/24 | Hold |

The wide lane reproduced the single-prompt reference result in-engine
(220.6 tok/s on the 73-token pilot) and established that the target hidden
taps and commit-only companion cache work. Its first full-suite mismatch was
classified precisely: scalar AR selected token 1724 while the eight-row target
verify selected 1752 at a 0.125-logit margin. A low-margin scalar replay fixed
the immediate token but not delayed state drift; replaying enough cycles erased
the speedup. NAX is not an answer because its accumulation order is explicitly
not bit-exact.

The staged lane amortizes one seven-token DFlash proposal across single-draft
K1 verifies. It proves the DFlash integration adds no divergence relative to
the existing compiled target-prefix control (24/24 identical) and reaches
169.95 tok/s, but the control itself is only 13/24 byte-identical to
`generate_ar`. That pre-existing target-prefix/AR discrepancy is now the
release blocker. Do not redefine the gate around MTP parity and do not revive
the rejected router/cache recalibration; fix the authoritative M1-versus-M2
stream first.

The implementation remains behind `MTPLX_DFLASH_DRAFT=1`. Wide mode has no
heuristic “exactness” margin; the failed low-margin replay path was removed
rather than left as an unsound switch. The staged mode is named `staged-k1`, not
“staged-exact”: K1 describes its execution width and makes no AR-equivalence
claim. It requires the fail-closed compiled A3B stack (compiled target-prefix,
fused GDN post-conv, row-owned routing,
packed gate/up, whole-MoE, and contiguous-dense decode layout) and still exits
the benchmark gate nonzero until AR equality is restored.

Evidence receipts: the historical wide run is
`benchmarks/results/dflash-engine-suite-112-20260730.json`; the staged-K1 run
(whose manifest predates the rename and therefore says `dflash_staged_exact`)
is
`benchmarks/results/dflash-engine-suite-staged-whole-moe-exact-b8-112-20260730.json`.
Receipt names and payloads are immutable; their failed equality fields remain
visible. The protocol-hardened wide rerun is
`benchmarks/results/dflash-engine-suite-wide-protocol-v2-112-20260730.json`;
it records every runtime switch and is the source for the current table. A
post-rename one-prompt integration smoke is
`benchmarks/results/dflash-engine-smoke-staged-k1-protocol-v2-20260730.json`
(194.3 tok/s, 1.46x AR, three-way equality); it validates the live
`staged-k1` wiring only and does not supersede the 24-prompt red gate.

Every evidence-grade run records an effective-run manifest with explicit
thinking mode, immutable target/drafter revisions, prompt-suite hash, sampler,
token cap, seed, block size, generation mode, verify strategy, compiled-verify
state, all execution-affecting runtime switches, NAX state, and
`sum(tokens)/sum(decode_seconds)` aggregation. Intended command-line settings
are not accepted as proof.

The older envelope below is retained as design history. Where it conflicts
with this section or the corrected Phase-0 summary, this section wins.

## Original design TL;DR

MTP and DFlash both save the *same* target forward passes, so their speedups share
one `1/(1−α)` budget and never multiply. The honest combined value is
**max(per-regime lane) + 10–25%** from routing, soft-q acceptance, block
truncation, and (experimentally) MTP-seeded blocks — plus a hard floor: the
controller can always collapse to today's measured MTP behavior, so the design
cannot regress. The empirical anchor: on 27B-4bit, MTP+NAX (65.5 tok/s) and
bstnxbt's DFlash (3.04× ≈ 66 tok/s over a ~21.7 tok/s AR) land at the *same
place* today — the win is in unifying them under exact acceptance and picking
per-regime, not in stacking them.

Corrected expected end-state on this M5 Max: 27B-4bit **75–90 tok/s** (opt.
100–120); 27B-8bit **3.0–3.6×** after extending verify kernels to M8..16 (the
structurally best DFlash regime); 35B-A3B **2.2–2.6×** (gated on measuring the
MoE verify-row expert fan-out); Laguna-S **1.1–1.4×** near-term via context-copy,
**1.3–1.6×** (~75–90 tok/s) with a trained MTP head.

---

## 1. Why speedups don't multiply

Both drafters exist to reduce expensive target forwards. Committed tokens per
verify cycle obeys E[τ] = (1−α^{γ+1})/(1−α) (Leviathan) regardless of who
drafted; MTP and DFlash failures correlate on the same high-entropy tokens
[published, AdaFlash]. 2.24× · 3.04× ≈ 6.8× is a category error. What
coordination *does* buy:

1. **Routing** — per-regime, per-step selection between arms (MTP depth d vs
   DFlash block B) using live acceptance/cost EWMAs.
2. **Soft-q acceptance** — the z-lab reference uses sampled prefix matching
   (accept while draft token == target's own sample), whose per-position
   coupling Σp·q is unconditionally ≤ Σmin(p,q); running DFlash drafts through
   MTPLX's exact Leviathan–Chen lanes recovers that gap for free.
3. **Block truncation** — cutting a block before verify using q-statistics is
   free compute (it just shortens `verify_input`).
4. **Seeding** — MTP-drafted seed tokens anchoring the DFlash block (gated
   experiment, not a dependency).

## 2. DFlash ground truth (web-verified 2026-07-29)

- Paper: *DFlash: Block Diffusion for Flash Speculative Decoding*,
  arXiv:2602.06036, ICML 2026. Code: github.com/z-lab/dflash (MIT). Official
  MLX backend `dflash/model_mlx.py` exists ("tested on M5 Pro").
- **Mechanism**: NOT a free-standing draft LM. A 5-layer trunk (8 for Coder)
  that reuses the target's embedding + LM head, conditions EAGLE-style on
  hidden states tapped from **5 target layers** (fused, injected into every
  draft layer's K/V), and proposes a **block of 16 in ONE forward** —
  `[last_token] + [mask]*(B−1)`, single denoising step, no iterative loop.
  Drafting cost is one tiny forward, independent of block size.
- **Reference acceptance**: sampled prefix matching (token equality against the
  target's own samples until first mismatch). Lossless (see §5) but weaker
  coupling than exact rejection sampling.
- **Published numbers**: τ ≈ 6.5–7.9 out of block 16 at temp 0 (Qwen3-8B,
  H200); up to 6× speedup, 2.5× over EAGLE-3. Qwen3.6-35B-A3B drafter (386M):
  3.61× at concurrency 1 [published]. **Qwen3.6-27B drafter (1.73B): card says
  "still under training", results N/A** — yet bstnxbt/dflash-mlx measured
  **3.04× e2e on 27B-4bit on an M5 Max** (26.03 → 79.12 tok/s, 8K run)
  [published]. bstnxbt notes degradation at 8K+ contexts.
- **Trainer is NOT public** ("recipe coming soon"); training a new drafter =
  reimplementation + BF16 target forwards over ~800K samples — a cloud job,
  not an M5 job. Drafters are target-representation-coupled (finetune reuse
  degrades, doesn't break — EAGLE-family experience [published]).
- Licensing: dflash MIT, dflash-mlx Apache-2.0, 27B drafter MIT, 35B-A3B
  drafter Apache-2.0 — all fine to vendor with NOTICE entries.

## 3. What's already in-tree (the seams) [all verified]

- Four-lane exact acceptance loop: `generation.py:7778–7958` — soft-q
  Leviathan–Chen with residual (`sampling.py:228–264`), one-hot point-mass
  (context-copy lane, `generation.py:6789–6803`), `target_prefix`
  (`:7808–7812`, requires top-k or top_p=1 per the guard at `:7571–7574`).
- Draft injection points: verify build `generation.py:7467`; device-core skip
  pattern `range(0 if used_device_core else cycle_depth)` `:7127`; draft-q
  contract `_sample_draft_from_logits` `:3480`, per-depth `draft_probs`
  lookups `:7814/:7876`.
- Companion-model precedent: the gemma4 pair loads a second model on the
  model-owner thread (`runtime.py:317–351`), with `unit="block"` descriptors
  (`descriptors.py:581–588`) and `draft_head_identity` partitioning session
  bank + policy fingerprints (`openai.py:1649–1664`).
- Verify kernels: NAX m16 tile is **4-bit only** (`nax_verify.py:820` gate,
  `:1008` dispatch, 4 ≤ M ≤ 16); `verify_kernels.py` covers **M 4..6 only**,
  bits {4,8}. MoE: NAX ~neutral (dense projections only, turbo-verify.md).
- Verify is a flat `[1,N]` causal-chain forward (`generation.py:7494–7526`) —
  wide blocks need no tree masks, no per-row positions, no batched cache trim.
- Adaptive EV policy runs on **static constants that are wrong per-model**
  (`adaptive.py:82–84`: baseline 40 tok/s vs measured cold ARs 9.4–13.2).
- DFlash baseline harness: `mtplx dflash-mlx-baseline` (`cli.py:3728`),
  `competitor_baselines.py` (note at `:149`), `DFLASH_MODEL_REPO`
  (`constants.py:10`, imported nowhere else). Turbo-verify kernels were
  already ported *from* dflash-mlx — the verify path is literally shaped for
  DFlash-sized batches.

## 4. Architecture

**Spine**: cost-model-driven controller (design C) · **seams**: minimal-diff
integration points above (design A) · **math**: union-chain + block-q
exactness + seeding (design B).

- **`draft_source ∈ {auto, mtp, dflash, ccopy, none}` as a setting, not a new
  wire mode.** The `{"mtp","ar"}` validator (`openai.py:10399–10408`)
  generalizes only its capability string. Drafter identity (repo + revision +
  block config) folds into `draft_head_identity` so session-bank gates and
  tune keys partition automatically. Zero schema change.
- **Every draft source emits `(tokens, q)`** where q is the distribution
  actually sampled from (post sampler shaping, same
  `_sample_draft_from_logits` contract as MTP). All acceptance runs through
  the existing four lanes — no new acceptance math.
- **One verify forward per cycle, causal chain shape. No tree in v1.**
  Hybrid GDN recurrence can't ancestor-mask and commits are suffix-only
  (`trim_verified_window_to_prefix`/`commit_captured_prefix`); branch reuse of
  target samples/coins across siblings breaks exactness (SpecTr-style residual
  bookkeeping would be required). Re-entry condition: telemetry shows ≥10% of
  cycles dying at seed position 1 with healthy downstream DFlash acceptance.
- **Commit-only companion caches**: drafter conditioning features append only
  for committed rows (the `append_mtp_history` discipline,
  `generation.py:8039–8046`). No drafter rollback machinery exists, so none
  can be wrong. (5-layer hidden taps ride the existing prefill + verify
  forwards; capture is committed-rows-only.)
- **Controller `DraftSourcePolicy`**: arms {AR, MTP(1..3), DFLASH(B∈{4,8,12,15}),
  SEED(1..2)+DFLASH(B−d)}, scored Ê[τ]/T̂ from live per-arm per-position
  acceptance EWMAs and **live cost EWMAs** (replacing the static
  `adaptive.py:82–84` constants). Declaration {one-hot, soft-q} is itself a
  measurable arm dimension (see §5 — neither dominates pointwise). Collapse
  rule → (d=2..3, m=0) = today's mode. **Guaranteed floor: measured MTP
  behavior; the design cannot regress.**
- **Drafter execution**: on the model-owner thread (thread affinity
  non-negotiable), one ~5-layer pass per cycle, priced by the controller's
  live C_draft EWMA. `mx.compile` may wrap **logits computation only** —
  sampling stays on the host with the engine's numpy Generator (see §5.3).

## 5. Exactness contract (the load-bearing section)

MTPLX's identity is exactness; these are invariants, not suggestions.

1. **Chained Leviathan–Chen over a block is exact iff the declared q_i equals
   the true conditional law of draft i given the accepted prefix.** For
   DFlash this holds because the block is produced by ONE forward with
   per-position conditionally independent fresh sampling. **This must be an
   enforced invariant of the vendored drafter module**: one forward, every
   position sampled from its own logits with fresh randomness, an assertion
   that no refinement/remasking path is reachable, verified against the
   reference implementation at Phase 0. Until asserted, the DFlash lane
   defaults to one-hot declarations.
2. **One-hot declarations are UNCONDITIONALLY exact** — committed law = p for
   arbitrary drafter dependence structure (algebra: q̃(x)p(x) + p(x)(1−q̃(x))
   = p(x)). Robustness ordering: **one-hot > target_prefix (sampler-config
   restricted) > soft-q (conditional on q-fidelity)**. One-hot — not
   target_prefix — is the universal fallback and debug lane. Rate note:
   soft-q *typically* wins for a calibrated drafter (Σmin(p,q)→1 as q→p) but
   does not dominate pointwise (sharp target + diffuse drafter favors
   one-hot); hence declaration-as-controller-arm.
3. **No sampling inside `mx.compile`** without threading MLX random state
   through the graph — a captured PRNG key freezes the draws while the
   declared q stays full softmax = silent bias that both temp-0 identity and
   math-level marginal tests are blind to. Compile logits only; sample on
   host. Runtime canary: two drafter calls on identical input at temp>0 must
   differ.
4. **Truncation measurability**: a pre-verify block-cut rule may use only
   (a) distribution-level statistics of q (entropy, max-prob — deterministic
   given logits), or (b) a position-causal scan (keep i decided by positions
   < i only). A keep-if-token-confident rule tilts the true conditional law
   away from the declared q — either declare the restricted-renormalized q or
   run truncated rows one-hot.
5. **Sampled prefix matching is lossless** (proved in review: every committed
   token is the target's own fresh sample conditioned on the actual committed
   prefix; the match event gates only whether *further* tokens commit). The
   reference's greedy-draft variant accepts w.p. p(argmax q) — identical to
   one-hot LC. Threshold/"typical-acceptance" variants would be LOSSY; MTPLX
   routes DFlash rows through its own lanes, so this class of bug cannot
   enter.
6. **Seeding from unverified MTP frontier tokens biases nothing** — drafter
   conditioning enters only through q; a rejected seed kills the downstream
   block (wasted draft compute, zero bias). Controller arm choice from
   history EWMAs is likewise exact (history-measurable, fresh coins).
7. **Gates (launch blockers, existing machinery)**: seeded temp-0 token
   identity vs AR; `speculative_output_marginal` extended with an
   independent-product-q case; a dflash-lane `exactness_baseline` in
   `mtplx_runtime.json` (registry blocker path `registry.py:690–723` — label,
   never a load gate); **plus one end-to-end statistical gate**: run the real
   engine path at temp>0 on a small model, chi-square committed-token
   frequencies at fixed probe prefixes against AR — the only gate class that
   catches q-mis-declaration (frozen RNG, shaping-order bugs, truncation
   tilt) on the actual runtime path.

## 6. Kernel/regime map

| Body | Verify-row economics | Default arm family |
|---|---|---|
| 27B-class 4-bit dense | NAX m16: rows 4..16 nearly free [verified + measured] | DFLASH B12–15 / seeded union — the prize lane |
| 27B 8-bit dense | vk M 4..6 only [verified]; M>6 = stock qmm | MTP D3 (2.71× cold [measured]) until vk M8..16; DFlash capped B≤5 interim |
| 35B-A3B MoE | NAX ~neutral; expert fan-out u(M) **unmeasured** | MTP D2–3; block width set by measured u(M) |
| Laguna-S 118B-A8B (no MTP head) | u(M) unmeasured; official poolside DFlash drafter EXISTS (§10) | ccopy near-term; poolside drafter mid-term |

u(M) = unique experts activated across M verify rows. i.i.d. routing predicts
u(16) ≈ 82 of 128 (~10× per-token expert reads) — real routing correlation
should come in below that, but **nobody has measured it and it single-handedly
decides the MoE envelopes.** Phase 0 item.

## 7. Performance envelope (post-verification, corrected)

Provenance discipline: `~/.mtplx/tuning.json` holds exactly 3 records, all
cold-long-code-192 suite, all Fable-711-family dense 27B-class (4-bit AR 13.17
→ D3 2.2276×; 8-bit AR 9.39 → D3 2.7074×; Fusion 2.609×) [measured]. The NAX
figures (48.3→65.5 tok/s, 1k-decode, reasoning on) are a *different* model
(Qwen3.6-27B Optimized-Speed) [measured]. There is **no MoE tuning record**;
the repo headline 2.24× is 27B-class. Bandwidth sanity (~550–575 GB/s class):
27B-4bit true AR ≈ **21–22 tok/s** at 1k ctx (48.3 was already speculative;
"AR 29.4" obtained by dividing across kernel configs was a derivation error —
caught in verification and corrected below).

| Target | Today | Conservative | Expected | Optimistic |
|---|---|---|---|---|
| 27B 4-bit dense | 65.5 tok/s ≈ 3.0× vs AR≈21.7 [measured] | 65–70 (MTP floor) | **75–90 (3.4–4.1×)** | 100–120 (τ≈6–7 needed); hard cycle-time cap ≈130 |
| 27B 8-bit dense | D3 2.71× cold [measured] | parity (B≤5 interim) | **3.0–3.6× after vk M8..16** | 4.0–4.5× — structurally the best DFlash regime |
| 35B-A3B MoE | **unmeasured** (likely 1.8–2.3× at D3) | Phase-0 measured baseline | **2.2–2.6×** (B≤8 + routing + truncation) | ~2.8×, iff u(M) ≪ i.i.d. AND τ≥5 |
| Laguna-S | 1.0× (≈56 tok/s [measured]) | ccopy 1.0–1.1× blended | **ccopy 1.1–1.4× on code/edit**; trained-MTP 1.3–1.6× (75–90 tok/s) | ccopy ~1.5–1.6×; trained-MTP ~1.8× (~100) |

> **Superseded in part by Phase 0 (§10a).** Measured on this machine: stock
> AR is 31.1 tok/s (27B-4bit) and 18.0 (27B-8bit); the reference DFlash arm
> reaches 1.55× / 2.09× at block 8, i.e. **below** the measured MTP D3
> multipliers of 2.23× / 2.71×. The expected end-state rows below therefore
> hold only if NAX wide-row verify kernels favour 9–16-row DFlash blocks
> substantially more than they favour MTP's 4 rows. That is now a bounded
> decision experiment to run before the multi-day build, not an assumption.
> Note also that Laguna-S has no MTP head, so its floor is 1.0× — the lane's
> case is strongest there.

All end-state numbers [estimated], pending Phase-0 τ / u(M) / T_V(M). Key
empirical datapoint: under the corrected AR, bstnxbt's 3.04× ≈ 66 tok/s ≈
today's MTP+NAX 65.5 — the two lanes currently land at the same place, which
is the strongest support for "max + 10–25%, never multiply."

## 8. Phased plan with go/no-go gates

**Phase 0 — measure the blind spots (1–2 days, zero engine changes).**
(a) T_V(M) microbench, M=1..16, NAX on/off, on 27B-4bit / 27B-8bit / 35B-A3B —
via the `attention_phase("decode_verify")`-wrapped forwards and their
`perf_counter` instrumentation → c(M) and MoE u(M) curves. Include a direct AR
re-measure (NAX off, `--generation-mode ar`) and re-baseline the
*actually-cached* models (two of the three tuning.json subjects are no longer
on disk). Also measure Laguna u(4).
(b) Revive the baseline harness (pyproject `competitors` extra, real z-lab
checkout, download the 1.73B drafter): local τ on our prompts, drafter pass
time, checkpoint facts (tap layers; whether embed/head are shared views; real
loaded bytes; confirm the reference sampler is genuinely one-shot per §5.1).
(c) Record everything in `benchmarks/results/`.
**Gates**: τ ≥ 4 greedy on coding prompts → full plan green. 3 ≤ τ < 4 →
proceed, envelope shrinks. τ < 3 → park DFlash (wait for a finished
checkpoint, or lead with the 35B-A3B 386M drafter — healthier card); Phases 1
and L proceed regardless. **Total cost of a kill: one afternoon.**

**Phase 1 — controller hygiene on existing arms (valuable even if DFlash
dies).** `DraftSourcePolicy` skeleton over {AR, D1–D3}; live cost EWMAs
replace the wrong-per-model static constants. **Gate**: AUTO ≥ fixed-D3 on
tune suites (expect +0–8% [estimated]); any regression → ship the EWMA fix
alone.

**Phase 2 — DFlash draft source, exact, env-gated (`MTPLX_DFLASH_DRAFT=1`).**
Vendored drafter module (MIT, NOTICE entry) with the §5.1 independence
invariant asserted; 5-tap hidden capture on prefill + verify forwards,
committed-rows-only; companion load on the model-owner thread via the gemma4
pair seam; injection filling `draft_tokens`/`draft_probs` before
`generation.py:7467` with the device-core skip pattern. **One-hot declarations day one** — both adversarial reviews and the Codex
review converge here; soft-q is promoted only after the q-fidelity gate
passes on the real runtime path (§11). Fixed-B, dense 4-bit body first
(kernels exist).
**Exactness gates**: the four blockers in §5.7. **Perf gate**: fixed-B beats
measured MTP D3 on the same suite, or the lane ships disabled-by-default.

**Phase 3 — cooperation (each toggle measured independently vs Phase-2
fixed-B).** (a) Router arms extended to DFLASH(B) + SEED(d)+DFLASH; (b)
pre-verify block truncation under the §5.4 measurability constraint (compute-
free); (c) seeded-block A/B — seeds unmasked into the block, seeded-vs-
unseeded acceptance per position. Keep only measured wins; seeding is an
experiment, not a dependency.

**Phase 4 — ops promotion + coverage.** Server flag; `draft_source` setting +
capability 400; health/truth labels (never label a DFlash number "MTP"); tune
candidates {AR, D1–D3, DF8, DF12, AUTO} with drafter identity in the tune
key; memory accounting (drafter bytes → `model_weights_bytes`);
`companion_drafter` metadata via `RuntimeContract.raw`. Kernel work as the
c(M) data dictates: **8-bit vk M8..M16** (m16 tiling recipe in-tree as
template; DFlash's relative payoff is largest on 8-bit) and MoE per-layer
expert-dedup across verify rows; graphbank compiled-verify buckets quantized
to {4,8,12,16} (the bucket machinery exists; the compiled-capture gate still
refuses on Python-int cache offsets). Long-context cutover back to MTP if the
8K+ degradation reproduces locally, via `resolve_long_context_mtp_depth`.

**Phase L — Laguna-S (parallel, explicit).**
- *Now (no training)*: the `draft_source` generalization + `none|ccopy`
  replaces the 5 AR-serving patch hunks (plus `patch_mtplx_gate.py`) in
  `/Users/pjb/models/laguna-s-mtplx/` with first-class concepts; promote the
  context-copy lane (already exact, one-hot + residual) as Laguna's
  speculative mode → 1.1–1.4× expected on code/edit over the measured ~56
  tok/s AR.
- *Mid-term — SUPERSEDED by the 2026-07-29 drafter scan (§10)*: poolside
  ships an **official Laguna-S-2.1-DFlash drafter** (1.115B, published τ ≈
  6.4 / 3.78× at c1 on GPU), which replaces the trained-MTP-head plan
  entirely. It lands in the same `draft_source` seam, gated on the 27B
  DFlash lane existing first (tap capture is Phase-2 machinery) and on
  Laguna's in-tree promotion (this phase). The earlier trained-head estimate
  (1.3–1.6×) stands only as the fallback if the drafter's license question
  (§10) resolves badly.

## 9. Top residual risks

1. **Drafter τ on this checkpoint** (card: "still under training") — Phase-0
   gate; a kill costs an afternoon; Phases 1/L ship value regardless.
2. **q-fidelity → silent distribution drift** (the one unforgivable failure) —
   §5 invariants + the end-to-end chi-square gate; one-hot as unconditional
   fallback.
3. **MoE u(M) kills wide blocks** — Phase-0 curve; controller prices rows
   live and shrinks B; dense 27B unaffected.
4. **8-bit kernel gap on the primary cached model** — interim B≤5 cap can
   never regress below measured MTP D3; vk extension is bounded work with an
   in-tree template.
5. **Owner-thread drafter serialization** — one ~5-layer pass per cycle,
   priced live; compile logits-only per §5.3.

---

## 10. Addendum — drafter market scan (2026-07-29, deep-research + 2 adversarial verifiers)

Question: is there a better drafter than the z-lab pair? Answer per target
(all numbers [pub] unless tagged; nothing DFlash-related has been measured
on this machine yet — the only [measured] speculative numbers remain MTP D3
2.23×/2.71× and MTP+NAX 65.5 tok/s):

- **27B dense: z-lab/Qwen3.6-27B-DFlash stays the winner available today**
  (MIT, 1.73B, card stale since Apr 27 and still "under training" — but
  dflash-mlx measured 2.78–3.06× e2e with it on an M5 Max [pub, third-party
  same-hardware-class]). New head-to-head to test: **satgeze/Qwen3.6-27B-
  DSpark** (Jul 23, Apache-2.0, stock-27B DSpark head warm-started from the
  z-lab head; measured 2.54–2.67× greedy on GPU, 1.39× on M3 Max Metal,
  accepted length ~5.4 at block 15). DSpark's published edge over DFlash is
  +16–18% τ / 1.21× e2e (llama.cpp PR #25173, Qwen3-8B). MLX 6-bit
  conversions of both z-lab drafters exist (funnygeeker). EAGLE-3 heads for
  27B exist (Ex0bit PRISM, Apache-2.0) but are strictly worse (τ 2.2 chain,
  worse AR economics) — ablation arm only. Same-tokenizer small-AR drafting
  is measured-dead (published negatives on both llama.cpp and Metal).
- **35B-A3B MoE: z-lab/Qwen3.6-35B-A3B-DFlash, retrained Jun 19** (40k-seq
  + SWA; Apache-2.0, 386M — smallest drafter found anywhere): c1 block-16
  HumanEval 3.29×, MT-Bench 2.24× (SGLang/B200). Apple Silicon anchor
  already exists: dflash-mlx reports **1.56–2.20× on M5 Max** for this
  target — positive but far below B200; the local question is closing that
  gap and pricing expert-union verify rows (u(M)), not existence. The
  native MTP head stays the Phase-0 baseline arm MTPLX must beat (mlx-lm's
  1.11× is evidence about mlx-lm's implementation, not our lane).
- **Laguna-S: the premise flipped — poolside ships an OFFICIAL drafter.**
  `poolside/Laguna-S-2.1-DFlash` (Jul 21/27): 1.115B, 6 SWA-512 layers,
  h=3072 (matches target — embed/head shared, no vocab projection), taps
  [1,10,19,29,38,47] of 48, block 16, full 100352 vocab. Published (INT4
  target, GPU, k=15): τ 6.42 HumanEval / 3.78× at c1. Geometry verified
  compatible with our local laguna_mlx.py; 64 GB oQ4e + 2.2 GB bf16 drafter
  fits. No MLX conversion exists yet — the trunk is a 6-layer plain-SWA
  port (no MoE, no GDN), gated on Phase-2 tap-capture machinery landing
  first. **License gap: target is OpenMDW-1.1 (commercial-OK) but the
  drafter repo is `license:other` with NO LICENSE file — ask poolside
  before shipping.** Clean-licensed fallback: olka-fi/Laguna-S-2.1-MXFP4-
  dspark (OpenMDW-1.1 tag, quality unpublished). This kills the
  train-an-MTP-head plan for Laguna.
- **Trainers are now public** (z-lab's own recipe still isn't): SpecForge
  (MIT, ships qwen3.6-27b-dflash + domino configs), DeepSpec (MIT,
  DSpark/DFlash/EAGLE-3; online mode = single 96GB GPU, ~5.5h, no
  hidden-state cache — the "8-GPU node + 38 TB" figure is offline-mode
  only), vLLM speculators (DSpark in mainline since 0.25).
- **DDTree tree-drafting (+35% e2e over DFlash [pub]) is BLOCKED on the
  Qwen3.6 targets** — architectural, not kernel work: the GDN-hybrid
  recurrence cannot ancestor-mask, and the engine is chain-only/suffix-
  commit by design (§4). Laguna (pure softmax attention) is maskable in
  principle but tree verify remains an engine rework + SpecTr residual
  bookkeeping. The chain-safe free add-ons stand: §5.4-constrained block
  truncation and a suffix-automaton upgrade to the ccopy lane
  (best-single-chain per cycle — multi-branch variants hit the same
  tree wall).

Phase-0 download/τ-test order stays 27B → 35B-A3B → Laguna, now with four
drafter arms on 27B: z-lab DFlash w4-trunk, satgeze DSpark, bf16-vs-w4
trunk A/B, and (10-minute curiosity) the AEON DSpark head on stock 27B.
Run day-one DFlash on the 4-bit target body (NAX m16); the 8-bit body caps
at B≤5 until the vk M8..16 extension (§8 Phase 4).

---

## 10a. Phase 0 — EXECUTED 2026-07-30: reference-arm build gate GREEN, release gate PENDING [measured]

Scope (Codex audit): every Phase-0 number is a reference-implementation or
stock-`mlx_lm` measurement, not MTPLX's own path — no NAX, no engine cache
machinery, no hidden taps, no repair/bonus accounting. They justify building
the lane; they do not establish production performance. The 4-bit 27B body
(the Phase-2 prize lane) was not measured. τ = committed tokens per cycle
(matched drafts + the target's replacement token); block 16 = 1 anchor + 15
draft rows. The temp-0.6 arm is sampled-draft (the MLX reference samples the
draft too), so only the greedy arm maps to our one-hot lane. Quality: 24/26
validations passed (both JSON-tool cases failed their validator).
Corrections and gaps in full: `benchmarks/results/phase0-2026-07-29/SUMMARY.md`.

Full results: `benchmarks/results/phase0-2026-07-29/SUMMARY.md`. Headlines:
**τ = 4.31 greedy / 3.92 at the product sampler** (z-lab 27B drafter,
stock 27B-8bit, 24/24 coding prompts clean) — the τ ≥ 4 gate passes; the
DFlash lane proceeds to Phase 2 (one-hot first per §11). True AR baselines:
27B-8bit **18.0 tok/s**, 35B-A3B **112.4**, Laguna **55.0** (validates the
app-measured ~56). T_V(M)/T_V(1) at M16: 2.29 (27B-8bit, with a stock-qmm
plateau from M10 — the vk M8..16 port's exact target), **2.96 (35B MoE)**,
3.84 (Laguna MoE) — the i.i.d. ~10×-reads fear did not materialize; wide
blocks are economically viable on both MoEs (block 8 sweet spot). Measured
block-8 beats block-16 e2e on 8-bit (37.6 vs 32.1 tok/s), confirming the
kernel-regime map with data. Cross-check: the 27B-8bit curve predicts MTP
D3 ≈ 2.6× vs 2.71× measured in tuning.json — cost model within 4% of
reality. Laguna upgrade [estimated from measured T_V + published τ 6.42]:
poolside-drafter lane ≈ **2.0–2.2× (~110–120 tok/s)** at block 8 —
above the earlier trained-MTP-head estimate; §7's Laguna row superseded
accordingly.

## 11. Addendum — Codex external review deltas (2026-07-30)

Codex (codex-cli 0.145.0, read-only, full code access) returned **FLAWED**,
vs our verifiers' SOUND_WITH_FIXES. The split: its architectural
recommendations converge with this doc (chain-only, fixed arm first,
one-hot first, measured controller last); the FLAWED verdict targets
implementation-readiness framing — places where this doc reads as if
machinery exists that Phase 2 has to build. Accepted deltas:

- **Drafter-cache transaction (new, real).** The DFlash drafter has its own
  KV cache; the reference crops it after every block (dflash/model.py:
  108–121) and crops the target cache post-acceptance. "Commit-only
  companion caches" must therefore mean: rebuild drafter KV from the
  committed prefix each cycle (simplest, cost must be priced) or implement
  crop/rollback and prove committed-prefix logits identical vs AR. Ignoring
  the drafter cache is not an option. Build order: transactions before any
  speed measurement.
- **Cycle cost is not one target forward (new, real).** All-accept cycles
  may trigger a lazy bonus forward (generation.py:8047–8073); rejections
  can trigger correction repair or full re-forward (:8319–8360). The
  controller objective is E[committed] / E[draft + taps + verify(M) +
  snapshot + repair + commit + bonus] — full-cycle wall clock, not
  accepted-length over a static verify estimate.
- **Per-position lane dispatch is new verifier code (accepted reword).**
  `verify_strategy` is cycle-wide (:5630–5643); target_prefix ignores
  draft_probs. The union chain reuses the existing acceptance *math*, but
  mixing lane types per position inside one chain is new verifier
  semantics + RNG accounting, not routing.
- **Phase-0 measurement upgrades (accepted):** verify_ratio.py defaults
  max_k=8 — extend to M=16 and record actual kernel dispatch per M, not an
  extrapolated curve; measure BOTH `--generation-mode ar` and stock AR
  (MTP sidecar loaded vs absent are different baselines, cli.py:2527–2548);
  τ readout gets histogram + first-reject position + p50/p90 by category/
  context-length/sampler, not just the mean; u(M) probe adds per-layer
  routed IDs, assignment counts, and dispatch/MLP/scatter timings (union
  count alone under-measures MoE row cost); prototype the 5-tap hook and
  measure its target-forward slowdown BEFORE any DFlash throughput claim;
  record drafter-block-vs-verify-rows accounting explicitly (reference
  block 16 = 1 anchor + 15 masks = 15 draft tokens).
- **Harness fixes (applied):** the τ runner now seeds MLX RNG per prompt
  (it recorded `seed` but never used it). Interpretation note: the
  reference drafts GREEDILY (draft-side sample has no temperature), so
  measured τ is exactly the one-hot lane's acceptance — convenient for the
  gate, but it cannot validate the soft-q uplift; that needs the §5.7
  chi-square gate on our own lane.
- **Scope fix (accepted):** grammar-constrained decoding under top-k/top-p
  has a documented distribution gap in-engine (generation.py:7892–7907) —
  the exactness claim explicitly excludes constrained generation until the
  masked-verify-row variant exists.
- **MTP-floor caveat (accepted):** the floor is a controller property, not
  a crash property — a DFlash exception after mutating companion state
  needs an explicit transaction boundary before falling back to MTP.

Rejected as overreach: "the +10–25% and envelope are unsupported" (every
such number is tagged [estimated] and Phase-0-gated — that is the design,
not a flaw); "MTP floor does not follow" in the routing sense (collapse to
(d,0) is trivially today's mode); FLAWED-as-architecture (its own
alternative plan is this doc's §8 order with more conservative wording).

---

*Full workflow reports (3 designs, 3 verdicts, 6 phase-1 maps, the
drafter-scan sweeps and verdicts, and the Codex review log) archived in the
session transcript; verifier-corrected numbers in this doc supersede all
earlier drafts.*
