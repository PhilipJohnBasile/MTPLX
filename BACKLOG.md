# MTPLX backlog

Living state. Kept current so the next session (or the next person) can pick up
without reconstructing anything. Newest block first.

---

### 2026-07-29 → 07-31 — MTP + DFlash coordination: designed, measured, redirected

**Done and committed:**
- ✅ `docs/mtp-dflash-coordination.md` — the "Priced Union" design: `draft_source`
      seam, soft-q through the existing acceptance lanes, live-cost controller,
      no tree in v1, exactness invariants. Adversarially reviewed 4× (3 verifiers
      + Codex twice); every accepted delta folded in (§11).
- ✅ Phase 0 executed and recorded in `benchmarks/results/phase0-2026-07-29/`:
      τ, true AR baselines, T_V(M) curves on 27B-4bit / 27B-8bit / 35B-A3B / Laguna.
- ✅ **The result that redirected the project:** DFlash loses to native MTP on
      dense 27B (1.89× vs 2.23–2.32×, protocol-matched) but wins ~2× on the
      35B-A3B MoE, where MTP is 1.05× at best. Mechanism: cheap verify rows +
      a 386M drafter + an incumbent that can't predict expert routing.
- ✅ Block 5–8 optimal on this hardware; the reference default of 16 is 24–29%
      off peak. DeepSeek independently ships `dspark_block_size: 5`.
- ✅ MoE DFlash lane implemented in-engine behind `MTPLX_DFLASH_DRAFT=1`
      (`mtplx/dflash_source.py`, tap capture, evidence gates, protocol manifests).
- ✅ Exact sampled-q reference lane — greedy same-seed equality cannot certify
      the product sampling law, so evidence moved to temp 0.6 / top-p 0.95 with
      declared sparse q and residual-corrected acceptance.
- ✅ Registry: DeepSeek V4 entry corrected — that architecture has **no MTP
      head**; its `mtp.*` tensors are a DSpark block drafter.
- ✅ Upstream to z-lab: [PR #150](https://github.com/z-lab/dflash/pull/150)
      (the MLX `load_draft` reads flat `rope_theta`/`block_size`; the newer
      35B-A3B card nests them, raising KeyError — reproduced locally) and
      [issue #151](https://github.com/z-lab/dflash/issues/151) (block-size data).
- ✅ `docs/target-prefix-ar-divergence.md` + `benchmarks/target_prefix_divergence_probe.py`
      — release-blocker analysis, plus a guarded scaffold for the eventual
      measurement (the scaffold is non-runnable by design; see its header).

**Done (cont.):**
- ✅ `docs/dflash-gate-preregistration.md` — the release gate, pre-registered
      BEFORE any measurement so no outcome can be rationalized after the fact.
      Separates question A (proposal-distribution correctness — the identity
      claim) from question B (target-implementation identity); replaces the
      single probe with a 7-rung axis ladder whose first nonzero rung names the
      mechanism; pre-commits the reading of every outcome including the
      do-not-ship ones.

**The critical path is NOT the GPU** (from the pre-registration §9): the A-side
rests on one 4-token single-position unit test, and both A-blockers the project
itself named — the independent-product-q oracle and the temp>0 drafter canary —
do not exist. H1 confirmation would remove the B blocker and create zero A
evidence, so the lane cannot ship on it. These are CPU-only:
- 📋 **A1** model-free rejection-sampler convergence suite (no model involved).
- 📋 **A2** the real thing: a production-adapter harness driving the actual
      staged source + acceptance transitions across MULTIPLE cycles, with an
      independent finite-state oracle over a fixed emitted-token horizon
      (queue reuse, rejection clearing, bonus, max-length, stops), compared
      against the prefix-conditional target AR law.
      `tests/test_dflash_joint_law_oracle.py` is a starting point ONLY — audited
      2026-07-31 as NOT EVIDENCE: it never calls production, and its
      conditioning "fix" was a weakening that hid the modelling gap (a real
      decoder continues into the next cycle after a rejection, so the
      unconditional two-token law it originally asserted was right and the
      oracle's truncation was wrong).
- 📋 **A3** drafter canary: two calls, identical input, temp>0 must differ —
      the only cheap guard against a frozen PRNG under `mx.compile`, which
      marginal tests are blind to by construction.
- 📋 **A4** falsifier battery: planted defects the gate must catch ≥18/20.
- 📋 Fix the probe before Monday, or it produces an uninterpretable number:
      it currently compares two stock-MoE *prefill* forwards with no
      `attention_phase`, no `mx.compile`, no shadow cache — it cannot see the
      axes in question. Implement rungs L0/L2/L3/L4 and the `--step length` stub.

**Blocked on GPU availability:**
- ⏸ Run the L0–L6 axis ladder (after the probe is fixed). First nonzero rung
      names the mechanism; L4-first means a state bug in the **already-shipping**
      compiled target-prefix stack, not the experimental lane.
- ⏸ Complete the block-16 sweep with thinking disabled, to close the caveat
      in z-lab#151 (current tables are thinking-enabled; the partial
      thinking-off set flips the 5-vs-8 ordering).
- ⏸ MoE `u(M)` per-layer expert-fanout decomposition (economic question already
      answered by the T_V curves; this is the mechanism).

**Queued:**
- 📋 Laguna-S lane — no MTP head, so the floor is 1.0× and an official poolside
      drafter exists (τ 6.42 published). Needs the 60 GB target re-downloaded
      and the drafter's licensing resolved (repo ships no LICENSE file).
- 📋 Server-side request cancellation on client disconnect. The engine kept
      generating 1200-token completions for dead clients and wedged the machine;
      forensic snapshot in `benchmarks/results/phase0-2026-07-29/engine_wedge_metrics.txt`.
- 📋 `trim_prompt_cache` is a silent no-op on hybrid caches (`ArraysCache` has no
      `trim`, so it returns 0 without trimming attention either — measured max
      logit delta 1.17 after an 8-row verify + trim, no error). Production code
      never calls it and the reference port guards correctly, so no measurement
      was contaminated — but rollback must use captured-state restore, never a trim.
- 📋 Widen `mtplx tune`'s depth search. It only tests AR/D1/D2/D3; depths 4–6
      were never in the space. (Note: the depth-5 "win" did NOT survive repeats —
      3–5 are indistinguishable at this sample size. The reproducible finding is
      a ~17% dip at depth 6 / M=7, likely a kernel-dispatch boundary.)

**On hold:**
- ⏸ Dense-27B DFlash — parked, not dead. It loses by ~18% against a drafter
      whose own card says "still under training". Worth one re-test if z-lab
      ships a finished 27B checkpoint.

**Won't do:**
- ❌ DeepSeek-V4-Flash MLX port. Scoped 2026-07-31, dropped same day: the source
      is already ~4.4 bpw (167 GB), so nothing above ~2.5-bit fits 128 GB, and
      MTPLX classifies it `recognized-backend-pending` and mlx-lm has no
      `deepseek_v4` module (ml-explore/mlx-lm#1233, #1281); 15–25 days. Facts kept in
      memory so it isn't re-derived; workspace deleted.
- ❌ Tree verification in v1. Hybrid-GDN targets can't ancestor-mask, commits are
      suffix-only, and branch-shared coins break exactness.

---

### Standing gates (do not quietly relax)

- Speculation must not change the output distribution. Two *distinct* questions:
  (A) is the drafter's proposal distribution correctly declared, and (B) is the
  target implementation numerically identical to another implementation. Byte
  equality vs `generate_ar` tests (B), not (A). Conflating them is a gate bug.
- Intent is not evidence: every evidence-grade run records its full effective
  configuration, not the flags someone meant to pass. A single unpinned
  `enable_thinking` corrupted every comparison in this project on 2026-07-30.
- Never benchmark under contention. Never derive a baseline by dividing
  measurements from different configurations.
