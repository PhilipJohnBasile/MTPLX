# DFlash / compiled-target-prefix release gate — pre-registration

**Status: PRE-REGISTERED. Written and committed BEFORE the Monday measurement. Supersedes the conflicting gate statements in `docs/mtp-dflash-coordination.md:96-101` and `benchmarks/dflash_engine_suite.py:677-687` (see §3, R3).**
Intended path: `docs/dflash-gate-preregistration.md`. The commit that lands this file must be an **ancestor** of the `git_head` recorded in every Monday receipt (`benchmarks/target_prefix_divergence_probe.py` `_env_snapshot`), and that ancestry must be *checked in the receipt-validation step*, not asserted in prose.

---

## 1. The two questions, and which gate answers which

**Question A — proposal-distribution correctness.** Is the `q` the acceptance math uses *the distribution the drafter actually sampled from*? This is the Leviathan–Chen guarantee. It is a property of the acceptance code and the drafter's declaration, provable against synthetic distributions with no model involved. **This is MTPLX's identity claim.**

**Question B — target-implementation identity.** Does implementation #2 of the same model (compiled target-prefix + fused GDN post-conv + shadow cache under `mx.compile`) compute the same logits as implementation #1 (`generate_ar` → `rt.forward_ar` → `model(...)`)? This is a numerics question. It has nothing to do with speculation.

Both papers quantify over arbitrary given distributions and take `p` as an input: Leviathan §3.6 — "guarantee an identical output distribution for any choice of approximation model M_q without restriction" ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)); Chen et al. p.6 — "the different computation graphs lead to different numerics, we cannot expect identical outputs", guarantee stated as "provably lossless **within numerics**" ([arXiv:2302.01318](https://arxiv.org/abs/2302.01318)). [confirmed — quoted in Sweep 1 from the primary sources; I did not re-open the PDFs this session]

### Gate map (in-tree, verified)

| Gate | A or B | Bit-exact? |
|---|---|---|
| seeded byte-equality vs `generate_ar` (the 13/24 receipt) | **B** | yes |
| `runners/mtp1_gate.py` temp-0 stream identity | **B** (degenerate) | yes |
| `runners/batch_equivalence.py` (max\|Δlogit\| ≤ 1e-3 + argmax) | **B** | no — tolerance |
| `scripts/phase0h_paged_verifier_exactness.py` (`--max-logit-diff 3e-2`, `--max-total-variation 5e-3`, `--min-topk-overlap 0.95`, `--min-sample-agreement 0.995`, defaults at `:507-510`) [confirmed, read this session] | **B** | no — **and it ships** |
| `a3b_whole_moe` M1↔M2 row parity, limit **0.0**, install-gating (`mtplx/a3b_whole_moe.py:995-1010`) [confirmed] | **B** (kernel) | yes |
| `_SELFCHECK_LIMITS` fused-M2-vs-stock (`a3b_whole_moe.py:95-102`, reference at `:929`) [confirmed] | **B** (kernel) | no — up to 0.5 |
| `summarize_external_draft_contract` (`mtplx/benchmarks/protocol.py:119-242`) | **A-adjacent**: audits the *declaration*, not the law | n/a |
| `compare_token_position_distributions` (`protocol.py:245-387`) | **conflates A and B**; `release_gate_pass` hardcoded `False` at `:273, :299, :317, :369` [confirmed] | n/a |
| `tests/test_sampling.py:72-76` (`speculative_output_marginal`) | **A** — the only empirical A evidence in-tree: one (p,q) pair, 4-token vocab, single position | n/a |

**Design rule adopted here: any gate that mixes A and B in one verdict is a bug in the gate.** The receipt prints a separate A verdict and B verdict, plus one final `A AND B` release verdict. (Amended per §1b A6: an end-to-end AR-vs-DFlash run is a permitted combined *diagnostic*, but it is not A evidence.)

---

### 1a. Independent verification of the load-bearing claims (2026-07-31)

Re-checked against the code by hand, not taken from the analysis that produced
this document:

- **C8 CONFIRMED.** `release_gate_pass` is hardcoded `False` at all four return
  sites in `mtplx/benchmarks/protocol.py` (`:273, :299, :317, :369`) —
  *including* the no-divergence path, which also reports
  `equivalence_bound_available: False`, `power_calibration_available: False`,
  `joint_sequence_law_tested: False`. `--distribution-repeats` defaults to 0
  (`dflash_engine_suite.py:76-80`), whose help text says "Zero records a
  fail-closed deferred gate." This is deliberate fail-closed design, not an
  oversight. The suite is honest; it simply cannot say PASS yet.
- **C9 CONFIRMED.** Neither named blocker exists: no hit for
  `independent.product` or `canary` anywhere in `mtplx/`, `tests/`,
  `benchmarks/`.
- **R1 CONFIRMED, and stronger than stated.** The dispatch docstring
  (`a3b_whole_moe.py:1024-1034`) does not merely permit byte-equality — it
  claims it as the design intent, in the author's own words: single decode
  rows "run the M1 route, whose per-row arithmetic **bit-matches M2** — the
  whole decode-time model function is one consistent arithmetic, **which is
  what makes the greedy K1 stream byte-comparable to greedy `generate_ar` of
  the same configuration**. Prefill keeps the stock path (identical for every
  entrypoint)."
- **R2 CONFIRMED.** `rows not in {1,2,3}` falls through to
  `route.accepted_call` (stock), so the eight-row wide lane genuinely runs an
  axis staged-K1 does not.

**What R1's strengthened form implies, and it is not comfortable.** If
byte-comparability of the greedy K1 stream to `generate_ar` is the *stated
design intent* of the installed dispatch, then a 13/24 result is not evidence
that the gate measures the wrong thing — it is evidence that **the code is not
doing what its own author documented**. That materially demotes H1 for this
lane and promotes "genuine defect", with a blast radius covering the
already-shipping compiled target-prefix stack rather than the experimental
DFlash lane.

Consequently §5's decision table should be read with its thumb on the scale
*against* relaxing the gate: the burden is now on H1 to explain why an
arithmetic that is documented as bit-matching is not.

## 2. Facts established by code-read before any measurement

- **C1** [confirmed] `_SELFCHECK_LIMITS` (`activations 0.125`, `stage3_output 0.5`, `output 0.5`) is applied to exactly one lane, `a3b_whole_moe_target_m2`, at **rows=2**, comparing the fused M2 route to `mx.compile(lambda current: binding.block(current))` — the **stock** block — on a synthetic `mx.arange` fixture (`a3b_whole_moe.py:848-955`, aggregation `:969-991`).
- **C2** [confirmed] The installed dispatch `_target_a3b_whole_moe_call` (`a3b_whole_moe.py:1024-1047`) routes on phase and row count: `phase ∈ {decode_verify, ar_decode}` and rows==1 → M1, rows==2 → M2, rows==3 → M3, **everything else → stock**; prefill keeps stock. M1↔M2 per-row parity is enforced at **limit 0.0** and gates install (`:995-1010`).
- **C3** [confirmed] The row-owned router applies the same phase/row rule and treats `decode_verify` and `ar_decode` **identically** (`mtplx/qwen_row_owned_router.py:366-382`). Phase is not itself an arithmetic axis for MoE/routing.
- **C4** [confirmed] `forward_with_a3b_gdn_postconv_capture` (`mtplx/gdn_capture.py:2527+`) is a **separately written 40-layer forward** — its own `embed_tokens`, its own `create_attention_mask`, its own layer loop — not merely a fused GDN kernel swap. The differing surface vs `model(...)` is larger than "post-conv accumulation order".
- **C5** [confirmed] The compiled M1 step (`mtplx/a3b_compiled_target_prefix.py:276-312`) assigns shadow-cache leaves from traced state, clears `rollback_state`, and runs under `attention_phase("decode_verify")`; `generate_ar` decode runs under `attention_phase("ar_decode")` (`mtplx/generation.py:5061`).
- **C6** [confirmed] Default attention phase is `"unknown"` (`mtplx/attention_context.py:17-20`).
- **C7** [confirmed] `benchmarks/target_prefix_divergence_probe.py` `step_logits`: fresh `rt.make_cache()` + **full 256-token prefill each step**, **no `attention_phase` context**, calls `rt._forward_ar_capture_a3b_postconv` **directly** — not the compiled step, no `mx.compile`, no shadow-cache round-trip. `--step length` is a stub returning `status: "not_implemented"`.
- **C8** [confirmed] `--distribution-repeats` defaults to **0** (`benchmarks/dflash_engine_suite.py:76-83`), and `release_gate_pass` is hardcoded `False`, so `gate_status == "passed"` / `exit_code == 0` is **unreachable** (`:637-659`). The suite cannot currently emit PASS by any input.
- **C9** [confirmed by grep across `mtplx/ tests/ benchmarks/ scripts/`] Neither of the project's own named launch blockers exists: no independent-product-q extension of `speculative_output_marginal`, and **no drafter canary** (`docs/mtp-dflash-coordination.md:298-330` requires both).

---

## 3. Contradictions between the four sweeps, resolved

**R1 — Sweep 4's "F1 is dispositive" is WRONG, and this materially changes the argument.** [confirmed, C1+C2] The 0.5/0.125 tolerances bound **fused-vs-stock**. At staged-K1 both arms are rows==1 in a decode phase and therefore run the **same** M1 fused route; prefill in both arms runs the **same** stock path. Fused-vs-stock is a *shared-mode* property here, not a differing axis. **It follows that byte-equality vs `generate_ar` is NOT unattainable-by-construction for the staged-K1 lane**, and the strongest published-in-repo argument for retiring the gate does not survive contact with the dispatch code. Sweep 4's conclusion "(a) fails as a gate on F1" is retracted.

**R2 — The evidence everyone is reasoning from comes from a different lane with a different axis set.** [confirmed] The recorded 0.125-margin flip and the failed scalar replay are from the **wide** lane's eight-row target verify (`docs/mtp-dflash-coordination.md:88-93`). Any verify width outside {1,2,3} falls through to the **stock** MoE (C2), while AR decode uses fused M1 — so the wide lane genuinely differs on the MoE axis, and its divergence is *expected* inside `_SELFCHECK_LIMITS`. The staged-K1/target-prefix 13/24 divergence has a **disjoint** axis set (alternative forward trace, `mx.compile`, shadow cache). **Wide-lane evidence may not be transferred to the staged-K1 question, and this pre-registration forbids doing so.** All four sweeps conflated them.

**R3 — Doc/code drift is itself the gate-shopping hazard, and is resolved here.** `docs/mtp-dflash-coordination.md:96-101` names byte-equality "the release blocker"; `benchmarks/dflash_engine_suite.py:677-687` already demotes it to `"release_gate": False` with the note that same-seed byte equality "cannot establish stochastic exactness". Both are in-tree today, so *any* Monday result has a pre-written sentence endorsing it. **Resolution, effective on commit of this file:** byte-equality vs `generate_ar` is (i) **not** the release gate, (ii) **a mandatory tracked diagnostic with a monotone floor** (§4.4), and (iii) **not retired, renamed, or removed**. Both files gain a pointer to this document in the *same commit as this document*, before any measurement.

**R4 — Codex's sentence is ambiguous and must be read the strict way.** "Fix the authoritative M1-versus-M2 stream first" can mean the *kernel lanes* M1/M2 (already at limit 0.0, C2 — nothing to fix) or the two *streams* (AR vs compiled target-prefix). [uncertain — no way to resolve from the tree]. This pre-registration reads it as **the streams**, and discharges it as **"determine the mechanism"**, not "force byte-identity". Which obligation is being discharged must be stated in the release note.

**R5 — The Monday probe as written cannot decide what the sweeps want it to decide.** [confirmed, C6+C7] With phase `"unknown"` and rows=256, **both** arms fall through to stock MoE and stock router. So `--step logits` isolates the alternative forward trace **at prefill shape only**; it does not exercise M1, `mx.compile`, the shadow cache, or incremental decode state. Therefore `max|Δlogit| == 0.0` in step 1 does **not** implicate H2 — it only exonerates the postconv trace at prefill shape. And `--step length` is a stub, so Sweep 4's monotonicity criterion is **not evaluable Monday** unless it is implemented first. Sweep 4's proposed decision rule is unrunnable as stated; §4.1 replaces it.

**R6 — The NAX precedent is weaker than it looks and may not be leaned on unmodified.** [confirmed] `docs/turbo-verify.md:24-32` cites `scripts/nax_distribution_gate_expanded` — which does not exist in this repo. NAX ships on evidence that cannot be re-run from this tree, and its mitigation ("decode/draft/prefill stay bit-identical") **inverts** here: the DFlash divergence is *in* the AR-equivalent path. Reusing the NAX disposition requires the gate script to land **in-tree**.

**R7 — Where the sweeps agree, and it holds.** [confirmed] No engine gates an optimized path on byte-equality vs another implementation: vLLM caveats hardware numerics and does not guarantee stable logprobs; llama.cpp gates per-op NMSE; TensorRT-LLM states no losslessness claim; fla allows 0.2–0.5% relative RMS error for fused vs naive GDN; mlx-lm's own GatedDeltaNet batch/cache test uses `mx.allclose(rtol=1e-4, atol=1e-4)` on logprobs. And "it's just numerics" has been false before in this exact kernel family (vLLM PR #25393, chunked GDN inter-chunk state race). Both facts are load-bearing and neither is decisive alone.

---

## 4. The pre-registered gate

Four components. **All four must pass. Any one deferred ⇒ the release is deferred, never passed.**

### 4.1 B-side mechanism: the axis ladder (replaces probe steps 1–3)

Each rung is *intended* to add one axis; inputs and state identical across arms; all logits compared in fp32. **Amended per §1b A2: the ladder as written is not one-axis** (L1 moves both the rewritten 40-layer forward and the postconv arithmetic; L2 must become production-shaped M2), and a first-nonzero rung names the **first exposed difference — a provisional hypothesis requiring follow-up state hashes, never a mechanism verdict.**

| Rung | Arms | Axis isolated |
|---|---|---|
| **L0** | `rt.forward_ar` vs `rt.forward_ar`, re-run | run-to-run determinism (**must be 0.0**) |
| **L1** | `rt.forward_ar` vs `rt._forward_ar_capture_a3b_postconv`, prefill shape, phase `unknown` | the alternative 40-layer trace + fused GDN post-conv (= today's `--step logits`) |
| **L2** | same, rows==1 with an incremental cache, inside `attention_phase("ar_decode")` and `("decode_verify")` respectively | decode shape + fused M1 MoE + phase |
| **L3** | L2's postconv arm wrapped in `mx.compile`, no shadow-cache assignment | `mx.compile` |
| **L4** | the real step from `prepare_a3b_compiled_target_prefix` (shadow cache, state in/out) | shadow-cache leaf assignment / `rollback_state` |
| **L5** | full-stream `generate_ar` vs compiled control, 24 prompts × {64,128,192} tokens | the 13/24 receipt + length dependence |
| **L6** | `inline_g` vs `headquarter` (`MTPLX_A3B_GDN_POSTCONV_IMPL`) | fused-impl cross-check |

**Pre-committed reading (fixed now, so no outcome can be re-narrated):**
- The first rung at which the delta becomes nonzero names the **first exposed difference** (provisional; interactions and cancellation can defeat the ordering). L1/L2 first ⇒ arithmetic (H1). L3 first ⇒ compiler-induced numerics — *still not H1-as-written*, and requires its own written argument. **L4 first ⇒ state (H2). That is a bug, not a numerics finding.**
- **L6 is predicted null.** `gdn_capture.py:1181-1188` asserts in a source comment "bit-exact vs inline_g: parity 0.0 on y and states at m1 and m2" [confirmed as source text, unverified by measurement]. A null at L6 is **not** evidence for H1; it is a regression check on that comment. Only a *disagreement* is informative — and a disagreement means the comment is false and an unowned kernel defect exists.
- **L5 requires implementing the stub** (`--step length`, C7). If it is not implemented, the release is **deferred**; no length-based criterion may be inferred from anything else.

### 4.2 B-side equivalence: exact teacher-forced logit TV (the pass/fail instrument)

**We do not certify B by sampling.** With a 151,936-token vocabulary, closeness testing to L1 ε costs Θ(max{m^(2/3)/ε^(4/3), m^(1/2)/ε²}) ≈ 4×10⁴ samples *per prefix* at ε=0.10 and ≈ 3.9×10⁶ at ε=0.01 ([Chan–Diakonikolas–Valiant–Valiant, arXiv:1308.3946](https://arxiv.org/abs/1308.3946)) [confirmed via Sweep 1]. We have **logit access to both implementations**, which removes the vocab dependence entirely (logit-access TV estimation is O(n/ε²), n = sequence length, and tight — [arXiv:2607.19510](https://arxiv.org/abs/2607.19510)) [uncertain — abstract-level, not read at source]. Teacher-forced on a shared prefix, per-position conditional TV is **not a statistic at all**: it is `½Σ_v |p₁(v) − p₂(v)|`, computed exactly from two logit vectors.

**Sample plan (SUPERSEDED by §1b A4 — rule-of-three requires ZERO flips, positions within a continuation are correlated, and B6 is a path diagnostic not a joint-law bound; the numbers below stand only as a first draft):** ≥ **100 prompts**, ≥ **30,000 probed positions** total, of which ≥ **10,000 must be decode-shape (L2/L4 geometry)**. Rationale, stated in the receipt: rule of three (Hanley & Lippman-Hand 1983) — zero argmax flips in 30,000 positions bounds the flip rate at **≤ 1×10⁻⁴ (95%)**. For contrast, **the current 24-prompt gate at a perfect 24/24 bounds the per-prompt divergence rate only at ≤ 12.5%**, and the probe's default `--probe-steps 8` bounds nothing below 37.5%. Both numbers go in the release note.

**Thresholds — ANCHOR CORRECTED (see §1b A3).** These were originally described as anchored to the strictest already-shipping in-tree B gate (`phase0h_paged_verifier_exactness.py:507-510`). That is **false**: `mtplx/benchmarks/runners/batch_equivalence.py:34` uses **1e-3** and additionally requires support and argmax equality. The table below is therefore **looser than existing in-tree practice** and must either be re-anchored to 1e-3 with support equality, or accompanied by a written defence of why this lane gets a weaker standard. Do not read it as a settled threshold set:

| # | Metric | PASS | ESCALATE | FAIL |
|---|---|---|---|---|
| B1 | max\|Δlogit\|, every probed position | ≤ 3e-2 | 3e-2 < d ≤ 1e-1 | > 1e-1 |
| B2 | per-position TV at the product sampler (temp 0.6 / top_p 0.95 / top_k 20), after shaping | ≤ 5e-3 at ≥ 99% of positions **and** max ≤ 2e-2 | — | otherwise |
| B3 | top-20 overlap | ≥ 0.95 all positions | — | otherwise |
| B4 | shared-u inverse-CDF sample agreement | ≥ 0.995 | — | otherwise |
| B5 | argmax agreement over ≥ 30,000 positions | ≥ 0.9999 **and** every flip has `delta_exceeds_margin == True` | — | any flip with `delta_exceeds_margin == False` |
| B6 | chain-rule bound Σ_t TV over 192 teacher-forced steps | report-only, **publication mandatory** | — | — |

**ESCALATE is not a pass.** A B1 landing in 3e-2..1e-1 means the lane is looser than a threshold this project already ships elsewhere; it may proceed only with a written, committed justification naming why the phase0h standard does not apply, signed off before the release commit. Choosing 1e-1 up front because the recorded flip was at a 0.125 margin would be gate-shopping — hence the band, not a moved line.

### 4.3 A-side: proposal exactness — the identity claim, currently at **zero** empirical evidence

None of this needs the GPU window. **It is the critical path, not Monday.**

- **A1 — model-free rejection-sampler convergence.** Synthetic `p`,`q` with analytic answers; `n ∈ {10, 10², 10³, 10⁴, 10⁵}`; distance to the analytic target vs a random reference; require improvement ratio ≥ 20. (vLLM's shape: [`tests/samplers/test_rejection_sampler.py`](https://github.com/vllm-project/vllm/blob/main/tests/samplers/test_rejection_sampler.py).) **No model involved.**
- **A2 — block oracle.** Extend `speculative_output_marginal` (`mtplx/sampling.py:299-318`) to the **independent-product-q** block case: vocab 4, block 3, ≥ 1,000 random (p,q) pairs, max |recovered marginal − p| ≤ 1e-12. This is a named launch blocker (`docs/mtp-dflash-coordination.md:322-323`) and does not exist (C9).
- **A3 — drafter canary.** Two drafter calls on identical input at temp > 0 must differ. Named blocker (`:302-303`), does not exist (C9). This is the only cheap guard against the frozen-PRNG-under-`mx.compile` failure, which marginal tests are **blind** to by construction.
- **A4 — falsifier battery (power calibration).** Five planted defects, each a real bug in a shipped engine or a named risk in this repo's own design doc: (1) sampler shaping applied to the draw but not the declared `q` (vLLM PR #10198); (2) `q` truncated-and-renormalized but declared untruncated; (3) residual `(p−q)₊` computed against a different `q` than was sampled; (4) bonus token drawn from a `p` that is not the conditional at the full prefix; (5) frozen PRNG key inside `mx.compile`. Requirement: **≥ 18/20 detections per defect at the registered N**. If any defect is missed, the end-to-end gate is under-powered and its non-detection is **uninterpretable** → DEFER.
- **A5 — end-to-end q-mis-declaration gate.** The project's own words: "the only gate class that catches q-mis-declaration" (`docs/mtp-dflash-coordination.md:326-331`). Real engine path, small model, temp 0.6 / top_p 0.95 / top_k 20, fixed probe prefixes, committed-token frequencies vs AR; block-permutation p and KL over top-200+other, reusing `scripts/r1_chisquare_verifier_correctness.py` machinery; **fail if p ≤ 0.01 or KL ≥ 1e-3**; N per prefix = the N* that A4 shows achieves ≥ 90% detection of the *smallest* planted defect. **A5 is used as a bug-catcher with calibrated power, never as proof of equality.** The equivalence claim lives entirely in B2/B6, which are computed exactly from logits and require no power analysis at all. This is the design that dissolves the "non-rejection proves nothing" problem instead of arguing around it.

### 4.4 The ratchet (what makes replacing the gate a ratchet and not a retreat)

- Byte-equality receipts are **immutable**. `seeded_stream_equality_diagnostic.{all_arms_equal_cases, dflash_vs_ar_equal_cases, mtp_vs_ar_equal_cases}` keep their names and their failing values.
- **Floors, pre-registered now: `dflash_vs_control` ≥ 24/24, `control_vs_ar` ≥ 13/24.** Any future change that lowers either is a **release-blocking regression**, independent of every distributional result.
- Scope: the gate change applies to the **compiled-target-prefix + whole-MoE stack**, which is already shipping — not to the DFlash lane alone. Narrowing it to DFlash would be gate-shopping by scoping.
- `release_gate_pass` (`protocol.py:273/299/317/369`) **may not** go from hardcoded `False` to computed in the same commit that first makes it `True`. It becomes computed in an earlier commit that lands A4 and ships **known-bad receipts showing the gate FAIL** on planted defects.

---

## 5. Pre-committed decision rule

Per §1b A2 every "mechanism verdict" below is **provisional** — a first-nonzero rung names the first *exposed* difference and requires follow-up state hashes before any mechanism is claimed.

| Monday outcome | Provisional mechanism hypothesis | Gate becomes | Ships? |
|---|---|---|---|
| **L0 ≠ 0.0** | measurement invalid | — | **No.** Everything downstream uninterpretable. Fix determinism first. |
| **Δ = 0.0 at L1 and L2, nonzero first at L3** | compiler-induced numerics — **not** H1 as documented | B-side gate (§4.2) permitted **only** with a written argument for why compile-induced divergence is acceptable where kernel-induced is | Only after that argument + all of §4.3 |
| **Δ = 0.0 through L3, nonzero first at L4** | **H2 — shadow-cache state bug** | unchanged; byte-equality stays the gate | **No.** Root-cause and fix. Escalate: the divergent stack is already in users' hands. |
| **0 < max\|Δ\| ≤ 3e-2 first at L1/L2**, argmax agreement ≥ 0.9999, all flips `delta_exceeds_margin == True`, L5 mismatch fraction strictly increasing 64→192, L6 null | **H1 confirmed** | B-side §4.2 replaces byte-equality-as-gate; byte-equality becomes the §4.4 ratchet | **Only if §4.3 A1–A5 all pass.** H1 alone is not sufficient (§9). |
| **3e-2 < max\|Δ\| ≤ 1e-1**, otherwise as above | H1 with an out-of-house-standard magnitude | **ESCALATE** — written justification vs the shipped phase0h 3e-2, committed before release | Not without that justification |
| **max\|Δ\| > 1e-1 anywhere** | the fused route is outside any tolerance this repo has ever accepted | tolerance itself is the defect; re-derive, do not accommodate | **No.** |
| **Any flip with `delta_exceeds_margin == False`** | logic/state, not numerics | unchanged | **No.** |
| **L5 mismatch fraction flat or fixed-onset** | fixed-onset signature = state | unchanged | **No.** |
| **L6 disagreement (`inline_g` ≠ `headquarter`)** | unowned kernel defect; `gdn_capture.py:1188` is false | unchanged | **No.** |
| **L5 not implemented, or N < 30,000, or A4 < 18/20 on any defect** | insufficient evidence | — | **Deferred, exit 3. Never "passed".** |

**Explicitly foreclosed rationalizations,** written down now so they cannot be invented later: a null at L6 is not evidence for H1 (§4.1); wide-lane evidence does not transfer to staged-K1 (R2); "the selfcheck already allows 0.5" does not excuse the staged-K1 arms, which share that route (R1); `Δ = 0.0` at L1 does **not** implicate H2 by itself (R5); and non-rejection of A5 is never reported as equivalence.

---

## 6. Do-not-ship conditions (any one is sufficient)

1. L0 nonzero. 2. First-nonzero rung is L4. 3. Any argmax flip with `delta_exceeds_margin == False`. 4. `max|Δlogit| > 1e-1`. 5. L5 flat / fixed-onset / unimplemented. 6. L6 disagreement. 7. Any A4 falsifier detected < 18/20. 8. A2 or A3 absent (they are the project's own named blockers, C9). 9. A5 rejects at p ≤ 0.01 or KL ≥ 1e-3. 10. Probed positions < 30,000 or decode-shape positions < 10,000. 11. Any byte-equality floor (§4.4) regressed. 12. Distribution-gate parameters changed after first sight of data.

---

## 7. Receipt requirements

Every receipt carries: `git_head`, `git_dirty`, the full `MTPLX_*` env snapshot, model + draft **revision SHAs** (`protocol.py:435-439` refuses blanks), prompt-suite SHA, sampler triple, and the **pre-registration commit hash with an ancestry check against `git_head`**. Receipts print **a separate A verdict and B verdict, plus a final `A AND B` release verdict** (§1b A6): `question_a_verdict` (A1–A5, with the A4 power table) and `question_b_verdict` (L0–L6 ladder, B1–B6, probed-position count, and the rule-of-three bound `3/N` for the achieved N). `release_gate_pass` is computed from both. Failed fields stay visible under their original names, forever.

---

## 8. Honest limits — verbatim release-note / model-card text

> This lane is **not bit-identical** to `generate_ar`. Byte-equality against `generate_ar` measures whether two implementations of the same model are numerically identical; it does not measure whether speculation preserves the target distribution. The Leviathan/Chen guarantee is with respect to the target distribution **as computed by the verify path**, and both original papers state the guarantee holds only "within hardware numerics".
> What we certify: (A) the drafter's declared `q` is the distribution it sampled from, and acceptance + residual correction recover the verify-path target — evidenced by a model-free rejection-sampler suite, a block-level marginal oracle, and an end-to-end chi-square gate whose power was calibrated against five planted defects. (B) the compiled target-prefix implementation agrees with `generate_ar` to max\|Δlogit\| ≤ [measured], per-position TV ≤ [measured] at the product sampler, and argmax agreement ≥ [measured] over [N] probed positions (rule-of-three bound: flip rate ≤ 3/[N] at 95%).
> **What we do NOT certify:** identical token streams at a fixed seed; stable logprobs across batch shapes or engine versions; equality of the **joint** sequence law (we report an exact per-step TV chain-rule upper bound, which is a bound, not an equality); behavior outside the probed model, quantization, sampler, and context lengths. **`generate_ar` remains the designated bit-exact reference.** `mtplx qa exactness` reference runs and batch-equivalence gates must use `generate_ar`, not this lane — the same exclusion `docs/turbo-verify.md:31-32` already applies to NAX. MoE targets therefore have **no bit-exact reference inside the compiled lane**, and this is a user-visible mode distinction, not a footnote.

---

## 9. Can the lane ship if H1 is confirmed? — recommendation, not a hedge

**No. Not on H1 alone — and per §1b A8, H1 does not remove the B blocker at all.**

H1 confirmation would at most justify *considering* a pre-committed compatibility exception for **B**. It creates **zero A evidence**, and A is the identity claim. Today the entire empirical A-side of this project is one 4-token, single-position unit test (`tests/test_sampling.py:72-76`) plus a declaration-bookkeeping audit that checks labels, not laws. The two A-blockers the project itself wrote down — the independent-product-q oracle and the temp>0 drafter canary — **do not exist** (C9), and the gate class its own doc calls "the only gate class that catches q-mis-declaration" has never been built or run. A lane whose selling point is exactness cannot ship with the exactness question untested and the numerics question freshly relaxed. That combination is exactly what an external reviewer will read as gate-shopping, and they would be right.

**Concretely:**
1. **Before Monday, no GPU needed:** land A1, A2, A3, and the A4 falsifier battery with committed known-bad receipts. This is the critical path and it is being blocked on a measurement it does not depend on.
2. **Fix the probe before it runs**, or Monday produces an uninterpretable number: implement rungs L0/L2/L3/L4, wrap the arms in the right `attention_phase`, implement the `--step length` stub, and raise probed positions from 8 to the §4.2 plan. As written, the probe compares two stock-MoE prefill forwards and cannot see M1, `mx.compile`, or the shadow cache (C6, C7).
3. **Then** run the ladder. If the first nonzero rung is L4 → stop, it is a state bug, and the blast radius is the **already-shipping** compiled target-prefix stack, not the experimental DFlash lane.
4. **Ship condition:** A1–A5 pass, B1–B6 pass at the registered N, the §4.4 floors hold, the caveats of §8 land in the *same commit* as the gate change, and the lane stays **off by default behind `MTPLX_DFLASH_DRAFT=1`**.

**One-line status:** the lane is blocked on evidence that does not require the GPU window, and the Monday measurement — as currently coded — cannot decide the question it was written to decide.
---

## 1b. Amendments from external audit (2026-07-31, before any measurement)

Verdict on this document: **SOUND WITH CHANGES**. The A/B distinction was
upheld as sound reasoning rather than rationalization, with the anti-gate-shopping
line stated crisply: *replace a metric only when a pre-data counterexample proves
it cannot establish the stated property, retain it as a non-regression diagnostic,
add the missing direct test, and require both component verdicts to pass.*
The following corrections are binding.

**A1. The probe is not merely inadequate — it cannot run.** It calls
`prepare_a3b_compiled_target_prefix(rt, cache=...)`; the real signature is
`(model, *, config, gdn_postconv_factory)` returning a *factory*
(`a3b_compiled_target_prefix.py:156`). `TypeError` after the model load, before
any measurement. Now guarded to fail immediately; rewrite required.

**A2. The ladder is not one-axis, and its central claim is too strong.**
- L1 moves **two** axes at once: the wholly rewritten 40-layer forward
  (`gdn_capture.py:2527+`) *and* the postconv arithmetic. Split them.
- L2 uses rows==1, but production staged-K1 verification is **rows==2 / M2**.
  L2 must be production-shaped M2, with M1 repair/rebase tested separately.
- L3/L4 are not executable as written (same API error as A1).
- **"First nonzero rung names the mechanism" is false.** It names the first
  *exposed* difference. Interactions, cancellation, and state-dependent effects
  can defeat the ordering. A first failing rung is a **hypothesis requiring
  follow-up state hashes**, never a mechanism verdict. §5's table is amended
  accordingly: every row's "mechanism verdict" is provisional.

**A3. The threshold anchor is wrong.** `phase0h` is *not* the strictest
in-tree B gate: `mtplx/benchmarks/runners/batch_equivalence.py:34` uses
**1e-3**, and phase0h additionally requires support equality and argmax
equality. As written, B1's 3e-2 and B2's "1% of positions may exceed" are
*looser* than existing practice while claiming to be anchored to the strictest.
Re-anchor to 1e-3 with support equality, or state explicitly and defend why
this lane gets a weaker standard.

**A4. The statistics overclaim.**
- Rule-of-three requires **zero** flips; B5 permits up to three at N=30,000, so
  `3/N` is not its bound.
- Positions within a continuation are **correlated**. Use a cluster-aware bound
  over pre-declared independent prompts, or require zero flips and state that
  the bound applies only to that sampling frame.
- **B6 is not a joint-law bound.** Summing per-step TV along one teacher-forced
  path is a path diagnostic. Either give it a release threshold or mark it
  explicitly non-certifying.
- **A4's 18/20 is not "≥90% detection."** Its one-sided 95% lower confidence
  bound is far below 90%. Raise repetitions and report the LCB.

**A5. The A-side test as specified is confounded.** DFlash-vs-`generate_ar` can
reject because **B** differs even when `q` is perfect. A-side A5 must compare
speculative DFlash against a non-speculative control **using the same compiled
verifier**; B separately compares that verifier with `generate_ar`.

**A6. "Never one verdict" was too absolute.** Correct form: separate A and B
verdicts **plus** one final `A AND B` release verdict. An end-to-end
AR-vs-DFlash run is a permitted combined *diagnostic* — it is not A evidence.

**A7. C9 was overstated.** More A-adjacent testing exists than claimed:
`tests/test_dflash_soft_q.py:86` compares sparse-q shaping against a NumPy
reference, and queue/commit behaviour is tested. The two *named* blockers are
still absent; the A-side is thin, not empty.

**A8. §9's recommendation was not conservative enough.** H1 confirmation would
**not** remove the B blocker — it would at most justify considering a
pre-committed compatibility exception. Given that the installed whole-MoE
dispatch presents byte-comparability as its design contract (§1a), the 13/24
result stands as a **possible defect in already-shipping compiled-target-prefix
behaviour until root-caused**.

**A9. The recommended weekend build supersedes A2-as-written.** Marginal
recovery alone is too weak. Build an **independent finite-state exact
joint-law oracle** for the real DFlash K1 state machine: enumerate vocab-4 /
block-3 outcomes including product-q draws, acceptance, residuals, bonus
tokens, staged queue reuse, rejection clearing and stop behaviour — **without
importing the production acceptance helpers** — then mutation-test the
production adapter against it. This repairs A2 and is what makes A4 meaningful.
