# M5 Max speed with explicit target-quality gates

Status: primary-source research plus local directional measurements,
2026-08-10 through 2026-08-11.
This is a build-selection note, not a release claim. Local results are tagged
**[measured]**, source and code audits **[verified]**, upstream reports
**[published]**, and proposals not yet tested **[hypothesis]**.

## Decision

The best path is not blanket 1-bit or 2-bit target weights. It is an exact
session engine that reuses complete target state, verifies cheap proposals, and
chooses the lowest measured cost per committed token.

For DeepSeek V4 Flash, the recommended build order is **[hypothesis]**:

1. a source-bound mixed-precision target package that fits 128 GiB and passes
   quality, provenance, and lifecycle gates; the official 155.43 GiB weights
   cannot be the live M5 baseline;
2. persistent, source-faithful checkpoints of the complete hybrid state;
3. device-side DSpark acceptance bookkeeping, after profiling proves host
   synchronization is material;
4. verified prompt-lookup proposals for copy-heavy coding and agent work;
5. a controller choosing target-only, DSpark, or prompt lookup per request;
6. aggressive quantization of the *drafter* before lowering target precision
   further.

The public 2.44-bit DeepSeek control proves fit and useful speed on a 128-GB
M5 Max, but its own MMLU-Pro comparison is 7.4 points below the hosted BF16/API
reference. MAPS must beat that control on source-relative quality and lifecycle,
not merely reproduce its capacity result. **[published, calculated]**

For Qwen3.5/3.6 MTPLX models, the recommended sequence starts with the
head-dim-256 full-attention *numerical-change* experiment, then the packed
Gated DeltaNet prefill kernel and exact hybrid-cache fixes. The attention
branch is not an exact-invariant uplift: it needs target-quality, state, and
lifecycle evidence before any default changes.
BF16 is the source-quality baseline and fits both Qwen targets on 128 GiB.
No mixed-bit recipe is conservative merely because it uses 6 or 8 bits; the
official MLX-LM learned-quantization guide says sensitivity can be weak at
those widths and presents dynamic 4/5-bit allocation as the cheaper control.
Every compressed target therefore needs source-relative quality evidence.
Reserve 3-bit for sensitivity-selected tensors or routed experts after a fresh
holdout. **[published, hypothesis]**

This preserves the useful distinction:

- changing the target can change intelligence;
- changing a proposal source can reduce acceptance, but exact target
  verification preserves the target distribution;
- changing a mathematically equivalent kernel still needs a numerical and
  lifecycle gate on the exact deployment path.

## What the MLX organization already provides

The inference-owning projects are [MLX](https://github.com/ml-explore/mlx),
[MLX-LM](https://github.com/ml-explore/mlx-lm),
[MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm), and
[MLX-C](https://github.com/ml-explore/mlx-c). The other organization
repositories are bindings, examples, data, ONNX, or site infrastructure rather
than primary Metal decode paths.

### Suggested experimental queue

The ordering and “build” dispositions below are recommendations, not measured
speed claims. **[hypothesis]**

| Rank | Opportunity | Expected scope | Current decision |
|---:|---|---|---|
| 1 | Merged adaptive MoE NAX (#4023 plus #3960/#4056 correctness) | Qwen35/Laguna large sorted server prefill | only after B>=16 and B/E>=4 dispatch evidence; verify B is rows×top-k but remains too sparse |
| 2 | Packed GDN with reduction-tree oracle (#1559) | Qwen3.6 long prefill | exact candidate; repeat thermal envelope |
| 3 | Merged Metal barrier-table reuse (#3882) | all four targets | lifecycle/allocator A/B; no model-speed inference |
| 4 | Exact DeepSeek V4 dead-cache-chain fix (#1662) | long-generation lifecycle | implement and soak-test |
| 5 | Merged oMLX WSDPA-first sparse attention (#2568) | DeepSeek V4 long prefill | port/reference candidate; require M5 and source-quality gates |
| 6 | Sparse complete-state checkpoints | repeated-prefix TTFT and session reuse | simulate placement, then build |
| 7 | Copy-on-write prefix forks (#1547) | concurrent shared-prefix sessions | port after hybrid-state and concat-cost gates |
| 8 | DSpark whole-round profiling | DeepSeek V4 decode | profile before compression or device work |
| 9 | Exact hybrid rollback/checkpoints (#1486 + #1596) | Qwen/Laguna reuse and safe speculation | consolidate |
| 10 | Head-dim-256 NAX full attention (#3842) | Qwen3.6 long prefill | open review-required numerical-change experiment |
| 11 | Post-#3934 NVFP4 source reconversion | Qwen35 target compression | rebuild, then rerun source-quality and M5 gates |
| 12 | Rotated asymmetric KV (#1555 + #1550) | long-context Qwen/Laguna cache | local coding-quality and memory experiment |
| 13 | Small-row transposed GEMM (#3888) | BF16/FP16 verify rows 2-15 | numerically gated candidate; no target identity proof |
| 14 | Merged D128 NAX scheduling (#3843) | Laguna attention prefill | trace first; no target-range whole-prefill evidence |
| 15 | Historical wide-verify SDPA (#3838) plus QMM (#4171) | block-12-to-31 speculation at long context | archive; #3838 closed unmerged |
| 16 | Verified prompt lookup | copy-heavy code, RAG, and tool loops | build after checkpoint gate |
| 17 | Domino versus DFlash | Qwen3.6-27B speculative decode | port official checkpoint and run matched M5 gate |

Relevant merged or under-review MLX work includes:

- [adaptive many-expert NAX tiles, #4023](https://github.com/ml-explore/mlx/pull/4023);
- [ragged sorted-RHS large-row correctness, #3922](https://github.com/ml-explore/mlx/pull/3922),
  still open at the final 2026-08-11 refresh;
- [Metal barrier-table reuse, #3882](https://github.com/ml-explore/mlx/pull/3882);
- [small-row transposed GEMM, #3888](https://github.com/ml-explore/mlx/pull/3888);
- [D128 NAX attention-loop scheduling, #3843](https://github.com/ml-explore/mlx/pull/3843);
- [empty-output NAX group suppression, #3941](https://github.com/ml-explore/mlx/pull/3941),
  now merged; its published M5 Max cases remain below a material model-level
  speed envelope;
- [M5 Max NVFP4 QMV, #3961](https://github.com/ml-explore/mlx/pull/3961);
- [correct NVFP4 per-group scale reduction, #3934](https://github.com/ml-explore/mlx/pull/3934),
  which is a conversion-correctness dependency for any newly generated NVFP4
  checkpoint;
- [a wider M5 speculative QMV window, #3791](https://github.com/ml-explore/mlx/pull/3791);
- [sorted-gather stride correctness, #3960](https://github.com/ml-explore/mlx/pull/3960);
- [quantized QMV alignment, #3965](https://github.com/ml-explore/mlx/pull/3965);
- [NVFP4 matrix-tail correctness, #3912](https://github.com/ml-explore/mlx/pull/3912),
  required when a quantized dimension is 16 modulo 32;
- [ragged-K sorted-MoE correctness, #4009](https://github.com/ml-explore/mlx/pull/4009),
  required when sorted `gather_qmm` sees `K % 64 != 0`.

Open MLX-LM work supplies useful designs and tests:

- [packed GDN with an explicit reduction-tree oracle, #1559](https://github.com/ml-explore/mlx-lm/pull/1559);
- [exact hybrid rollback, #1486](https://github.com/ml-explore/mlx-lm/pull/1486),
  which is a prerequisite for speculative and prompt-lookup work on recurrent
  Qwen caches;
- [exact recurrent/hybrid prompt checkpoints, #1596](https://github.com/ml-explore/mlx-lm/pull/1596)
  and the narrower server checkpoint design [#1667](https://github.com/ml-explore/mlx-lm/pull/1667);
- [bounded `ArraysCache` metadata graphs, #1632](https://github.com/ml-explore/mlx-lm/pull/1632),
  the broader deferred-metadata/serialization design
  [#1642](https://github.com/ml-explore/mlx-lm/pull/1642), and recurrent-state
  [copy-on-extract, #1701](https://github.com/ml-explore/mlx-lm/pull/1701);
- the current [prompt-lookup decoder, #1508](https://github.com/ml-explore/mlx-lm/pull/1508)
  at head `2621d370f4f0ea003e7c36747223f88a6821d011`, whose 4,096-token
  `max_lookback` bounds the otherwise quadratic miss scan without changing
  target verification,
  the earlier independent [#1478](https://github.com/ml-explore/mlx-lm/pull/1478),
  and its measured fallback [#1557](https://github.com/ml-explore/mlx-lm/pull/1557);
- prompt-cache fail-closed guards [#1502](https://github.com/ml-explore/mlx-lm/pull/1502)
  and [#1503](https://github.com/ml-explore/mlx-lm/pull/1503), which reject
  partial trims and keyed reuse after a chunked window has slid;
- [short sequential SSM scan, #1586](https://github.com/ml-explore/mlx-lm/pull/1586),
  which is a Mamba2 design reference, not a Qwen GDN or DeepSeek V4 drop-in;
- [rotating-cache trimming, #1437](https://github.com/ml-explore/mlx-lm/pull/1437),
  the rotated-flag persistence fix [#1619](https://github.com/ml-explore/mlx-lm/pull/1619),
  and public Laguna checkpoint loading [#1704](https://github.com/ml-explore/mlx-lm/pull/1704);
- [fused quantized-KV prefill, #1510](https://github.com/ml-explore/mlx-lm/pull/1510),
  a large memory/TTFT candidate whose KV arithmetic is intentionally lossy.
- [DeepSeek V4 dead-cache-chain leak, #1662](https://github.com/ml-explore/mlx-lm/issues/1662),
  which reports a deterministic Metal buffer-count crash after about 11.5K
  generated tokens. Rebinding the unused values cache to keys is exact for this
  model because its attention uses `K == V`; the reported 13K-token soak was
  byte-identical. This is an open lifecycle fix, not a throughput result.
- Merged oMLX [WSDPA-first sparse attention, #2568](https://github.com/jundot/omlx/pull/2568),
  whose maintainer rerun on DeepSeek V4/M3 Ultra reports prompt throughput
  rising from 548.8 to 630.6 tok/s at 4K (+14.9%) and from 549.5 to 588.3
  tok/s at 32K (+7.1%), with generation throughput unchanged. The optimized
  reduction order changes temperature-zero output tokens, so this is a
  tolerance-level prefill result, not exact target identity. The merged stack
  also skips discarded intermediate chunk logits, grows the pooled cache in
  place, and folds the standard causal mask into the DSA indexer; those pieces
  have narrower exactness contracts than the WSDPA swap. Port or reuse them as
  separate ablations behind M5 speed, source-relative quality, and long-context
  lifecycle gates. All three CI jobs passed at merge commit
  `b128b23290392b4eab8df236a78308448c20642c`.
- [Qwen3.6 conversion normalization, #1623](https://github.com/ml-explore/mlx-lm/pull/1623),
  which prevents an already-converted checkpoint with `mtp.*` tensors from
  receiving a second RMSNorm shift. This is an intelligence-preservation fix,
  not a speed path. Existing artifacts need a provenance and norm-range audit.
- [GenerationBatch logprobs, #1479](https://github.com/ml-explore/mlx-lm/pull/1479),
  which is rejected below because its contract is broken and it supplies no
  MTPLX direct-argmax gain.

Recently evaluated MLX branches change the next measurement queue:

- Merged [adaptive many-expert NAX tiles, #4023](https://github.com/ml-explore/mlx/pull/4023)
  enter the enclosing sorted-RHS route only when `M == 1`, `B >= 16`,
  `right_sorted`, and `B / E >= 4`. Inside that route, BM32 applies only when
  integer `total_routed_rows / E < 64`. At pinned MLX-LM revision
  `254d153fdeb6f150edd4fc5a54f9828638481fa8`, Qwen3Next selects top-k experts
  per token and SwitchGLU sorts when `indices.size >= 64`, then flattens routed
  assignments before GatherQMM. Therefore target verification uses GatherQMM
  `M=1` and `B=verify_rows*top_k`, not `M=verify_rows` or `B=1`. For rows
  8 through 16, Qwen35 top-8 reaches sorted `B=64..128`; Laguna top-10 reaches
  sorted `B=80..160`. Both still fail only the outer `B/E >= 4` density term
  because `E=256`. For the 256-expert Qwen35 and Laguna prefill cases, BM32
  applies at 512 and 1,024 tokens, not at 2,048 or 4,096. Integrate it with the
  merged #3960/#4056 sorted-gather correctness fixes, then require exact routed
  outputs, logits, continuation, cache state, and traced tile engagement. The
  published kernel gain is 1.28-1.46x; the final heuristic has no current
  end-to-end receipt. This document does not count the sparse verify geometry
  as an interactive MTPLX acceleration. **[published, verified]**
- Open [ragged sorted-RHS large-row correctness, #3922](https://github.com/ml-explore/mlx/pull/3922)
  covers totals above 32,768 routed rows. The exact power-of-two Qwen35 cells
  do not need it, but arbitrary Laguna prompt/chunk lengths above about 3,276
  can: `10Q` is then above 32,768 and, when `10Q % 64 != 0`, ragged. Treat it
  as a production correctness dependency for the sorted path, not a speed
  result.
  **[verified]**
- Open [ragged-K sorted-MoE correctness, #4009](https://github.com/ml-explore/mlx/pull/4009)
  fixes a separate M5 corruption when `sorted_indices=True` and `K % 64 != 0`.
  The author reports 94-97% of output elements differing in representative
  unpatched shapes and no measurable performance change after the fix. It is
  review-required and has no checks. Fall back to unsorted execution for that
  shape until the fix or an equivalent guard lands. **[published, verified]**
- Merged [Metal barrier-table reuse, #3882](https://github.com/ml-explore/mlx/pull/3882)
  reuses write-after-read tracking hash tables at barriers and is exact
  lifecycle/allocator work rather than a model-speed claim. In a balanced
  four-runs-per-arm Qwen35 measurement, the prefill midpoint was 483.0903334 ms
  for parent and 486.8543285 ms for head (0.992269x); decode means were
  93.10392211452184 and 94.71917662343782 tok/s (1.017349x). Hashes were exact,
  but parent/head within-arm decode spreads were 12.48% and 7.96%, so the run
  supports no model-speed inference. **[measured, verified]**
- Merged [D128 NAX scheduling, #3843](https://github.com/ml-explore/mlx/pull/3843)
  adds `unroll_count(4)` to the existing D128 attention loop without changing
  reduction order. Only Laguna calls dispatching full-SDPA NAX with `D = Dv =
  128` are eligible. Its published 12% M5 Max kernel gain is at `Q = K =
  8,192`, outside the target 512-4,096 range; there is no target-range
  whole-prefill evidence. Require traced engagement and a numerical gate before
  treating it as an MTPLX candidate. **[published, verified]**

- [head-dim-256 NAX full attention, #3842](https://github.com/ml-explore/mlx/pull/3842)
  is an open, review-required numerical-change experiment, not an exact
  Qwen3.6 uplift. Its dispatch is causal, no array mask, head/value dimension
  256, and query length at least 1,024; both local target configs meet the
  architectural shape. Published M5 Max end-to-end results range from +3.8% at
  an 8K default-chunk prefill to +27.0% at 32K with 8K chunks, while peak
  memory falls 34.6 to 26.5 GB in the latter case. Against the exact current PR
  base/head, independent M5 Max A/B measured a 2.046x Q2048/K2048 attention-
  kernel gain. Two complete 35B prefill envelopes measured 1.031x and 1.077x;
  the second used the hardened per-run shard manifest check but had 4.5%
  same-arm spread on the PR head. It does not affect decode, and the numerical
  delta remains quality gated.
- A maintainer has separately endorsed a fail-closed
  [`force_fused=True` API](https://github.com/ml-explore/mlx/issues/3658#issuecomment-5249947631)
  that bypasses SDPA heuristics and throws when no fused kernel supports the
  shape. That is the right instrumentation seam for memory-pressure tests.
  The old implementation in [#3660](https://github.com/ml-explore/mlx/pull/3660)
  is conflicting, checkless, and 13-22% slower in its Qwen35/M4 prefill rows,
  despite lower peak memory. Build the narrow selector against current main;
  do not revive #3660 as a speed claim. **[published, verified]**
- Merged [small-row transposed GEMM, #3888](https://github.com/ml-explore/mlx/pull/3888)
  targets BF16/FP16 `x @ W.T` at rows 2-15 and publishes 1.3-6x M5 kernel
  gains. It is a numerically gated verification candidate only: no target
  identity proof exists for the relevant MTPLX paths. **[published]**
- [FusionML](https://arxiv.org/abs/2607.22785) and its public
  [benchmarks branch](https://github.com/ommo007/FusionML/tree/benchmarks) at
  `4e7121a8f9c70fd6512b233c5705374626323471` identify an exact-ish prefill
  opportunity: MLX lazy graphs serialize CPU/GPU
  split work when the CPU consumes an unmaterialized GPU result, while an
  explicit `mx.eval` boundary restores concurrency. The published unquantized
  Qwen2.5-7B results are 1.15-1.38x block-prefill and 1.18-1.25x TTFT on
  M1-M4, with the same greedy text and neutral decode. It is unmeasured on M5,
  and the reference patch intercepts only unquantized `nn.Linear`, so it does
  not directly cover 4-bit Qwen35. The local mechanism probe ran while
  unrelated host work consumed roughly 330-370% CPU, so it is a
  **contended-host adverse observation**, not an M5 mechanism verdict:
  every tested 5-35% CPU split was 2.1-5.7x slower than GPU-only execution.
  An explicit materialization boundary changed split time by only 1.7%.
  Outputs stayed within 3.1e-5 of the GPU reference. This is a synthetic
  mechanism warning, not a whole-model refutation or clean M5 NO-GO. The receipt is
  [`fusionml_stream_overlap_m5.json`](../../benchmarks/results/mlx-m5-research-20260810/fusionml_stream_overlap_m5.json).
  **[published, verified, measured]**
- [multi-row two-pass SDPA, #3838](https://github.com/ml-explore/mlx/pull/3838)
  is **closed without merge** as of 2026-08-11. Its pinned code removed the
  full-attention latency cliff at selected query rows 9-16 and long caches.
  The historical measurements below remain mechanism evidence, not an
  available upstream integration path. Today's D3/D5 paths did not engage it.

- [32-row `qmm_t_nax`, #4171](https://github.com/ml-explore/mlx/pull/4171)
  reports 1.21-1.22x kernel gains for wide 4-bit matrices at M=14 and M=32
  on a base M5. Its transposed affine kernel family supports 2/3/4/5/6/8-bit
  modes; only the published timing fixture was 4-bit. Our exact-shape result
  below confirms a fast regime but also finds a bimodal regime. Current D3/D5
  verification stays below its dispatch threshold, so the branch is conditional
  research rather than a present MTPLX upgrade.
- [wider M5 speculative QMV, #3791](https://github.com/ml-explore/mlx/pull/3791)
  uses QMV for `M < 33` when both dimensions are at most 2,048, `M < 25` when
  both are at most 4,096, and `M < 13` otherwise. The latter is the current
  wide-matrix M13 boundary: below it QMV remains selected, while M13 can enter
  the changed QMM regime. That keeps D3/D5 verification out of #4171 and makes
  any wider-row promotion numerically gated. **[verified]**
- [GQA-8 decode attention, #4077](https://github.com/ml-explore/mlx/pull/4077)
  reports 2.7-9.5% end-to-end decode gains at 8K-32K context on an M5 Pro.
  Its dispatch requires GQA factor 8 and head dimension 64 or 128. The current
  27B target is factor 6/head-dim 256 and the 35B-A3B target is factor
  8/head-dim 256, so neither engages it without upstream generalization.
- [Gated DeltaNet Metal kernels, #4020](https://github.com/ml-explore/mlx/pull/4020)
  adds sequential, simdgroup, and NAX forward primitives and reports matching
  perplexity across three Qwen3.5 models. Its M5 Max chart shows the strongest
  prompt-throughput gain on the 35B-A3B model at chunk 16. It is a
  higher-upside GDN prefill experiment than Python-level packing, but remains
  open and lacks masking and a backward pass. Its native operation returns
  only the final recurrent state, not one state per input row. That is enough
  for ordinary unpadded Qwen prefill, but not for MTPLX speculative rollback,
  which must commit the state corresponding to the accepted prefix.
- [Metal extension packaging, #4004](https://github.com/ml-explore/mlx/pull/4004)
  would let MTPLX ship C++/Metal extensions without maintaining a complete MLX
  fork. Its binaries still bind to the regular CPython and MLX C++ ABI, so they
  must be version-pinned and rebuilt with the runtime.
- [per-command-buffer resource retention, #4048](https://github.com/ml-explore/mlx/pull/4048)
  removes one completion handler per evaluated operation and retains input and
  sibling buffers once on the command encoder. The PR is one commit behind the
  current base, so the local experiment rebased only that commit before build.
  One Qwen 35B-A3B block observed 1.171x under heavy Codex host contention,
  while Qwen 27B was 0.989x with high variance and another 35B block failed
  stability. The candidate always occupied the middle positions, so patch,
  order, and host state are inseparable. This establishes no robust speed gain.
  Long-generation resource-lifetime tests and interaction checks with merged
  #4099, open #4096, and the #4174 replacement for closed-unmerged #4134
  remain mandatory.

The historical #3838 + #4171 combination mostly disappears after dispatch
analysis. Under #3838's required runtime key-length relation, its threadgroup
bound leaves Qwen 27B conditionally eligible only at M9-M10, while #4171 can
reach supported affine 8-bit projections only from the measured QMM boundary
M13. Qwen 35B-A3B is statically excluded from
#3838 across M9-M16. Laguna remains conditional on the long-key branch at
M9-M16 and can overlap #4171 at M13-M16; its measured #3838 attention benefit
fades from 1.42x at M9 to 1.11x at M13 and a regression at M16. A combined
Laguna target-width sweep remains legitimate; there is no current Qwen compound
path. **[verified, measured]**

MLX-LM also gained a new
[distribution-exact acceptance-rule proposal, #1709](https://github.com/ml-explore/mlx-lm/pull/1709).
It adds Leviathan-Chen residual acceptance and block verification for plain
temperature sampling while preserving the existing exact-match default.
The underlying formulas pass a rational joint-law oracle, but the integration
is **not merge-ready**: bfloat16/float16 temperature is reconstructed in a
different dtype order than sampling, history-dependent processors are applied
along the wrong conditional path, and a custom sampler can forge the `.temp`
capability marker. The narrow float32, temperature-1, trusted-categorical,
no-processor contract remains mathematically exact. The full independent audit
and remediation list are in
[`mlx-lm-1709-exactness-audit-20260810.md`](mlx-lm-1709-exactness-audit-20260810.md).
The linked audit now includes a CPU-only reproducer and JSON receipt for all
three counterexamples and the positive narrow-contract oracle.
**[verified]**

A CPU-verified adapter audit found one additional numerical constraint for
#4020. The core primitive promotes and casts gates to the query/key activation
dtype, while MLX-LM intentionally computes decay gates in float32. A safe first
adapter therefore routes only unmasked scalar-gate calls whose activations,
gate, beta, and carried state are all float32; every other call must retain the
current path. The normal quantized-Qwen GPU path still needs a live dtype probe.
**[verified]**

### MLX Swift LM cross-map

MLX Swift LM has two genuine generic gaps: [#516](https://github.com/ml-explore/mlx-swift-lm/pull/516)
is the rotating staged-KV protocol, and [#467](https://github.com/ml-explore/mlx-swift-lm/pull/467)
is the segmented compiled hybrid-decode fallback. [#470](https://github.com/ml-explore/mlx-swift-lm/pull/470)
is only a partial balanced-prefill result. MTPLX is stronger or already has the
relevant mechanisms in [#468](https://github.com/ml-explore/mlx-swift-lm/pull/468),
[#469](https://github.com/ml-explore/mlx-swift-lm/pull/469),
[#459](https://github.com/ml-explore/mlx-swift-lm/pull/459),
[#426](https://github.com/ml-explore/mlx-swift-lm/pull/426), and
[#510](https://github.com/ml-explore/mlx-swift-lm/pull/510); do not duplicate
them without a measured MTPLX gap. **[verified]**

Merged Swift [complete-state prompt snapshots, #475](https://github.com/ml-explore/mlx-swift-lm/pull/475)
provide the strongest current checkpoint contract reference. The format binds
KV and continuation state, version-marks state-bearing files, and rejects old
readers or incomplete restores. Both checks passed at approved head
`80515c275661f0d49a60fd9947933f2c2f496d42`. It has no large-model M5 speed
receipt, batching remains unsupported, and some Qwen2 paths still fail open.
Adopt the complete-state and fail-closed contract, not a throughput claim.
Open Swift [Qwen MTP #351](https://github.com/ml-explore/mlx-swift-lm/pull/351)
is likewise a correctness reference: emitted-token history, per-stream draft
state, and caller cache ownership were fixed at `c1a96af70a7e38dbcc86578d49a986a478bf7132`.
It falls back to target-only generation at nonzero temperature and publishes
no protocol-matched M5 result. **[published, verified]**

The #516 CPU ring oracle is complete. It covers 152 prefixes: 16 before wrap,
16 at exact wrap, and 120 after wrap. It also covers reject, partial, all-
accept, both bonus behaviors, reset, and cleanup. All six CPU tests pass.
This validates the rotating staged-commit algebra, not the Swift branch or a
Metal speedup. The receipt is
[`rotating_staged_kv_cpu_oracle.json`](../../benchmarks/results/mlx-m5-research-20260810/rotating_staged_kv_cpu_oracle.json).
The remaining cross-map experiments are a #467 compiled-AR A/B and an opt-in
#470 balanced-prefill numerical gate. **[verified, hypothesis]**

Two newer Swift results sharpen the service and compressed-cache boundaries.
Merged [single-dispatch TurboFlash, #520](https://github.com/ml-explore/mlx-swift-lm/pull/520)
fuses compressed-KV attention for contexts through 128 tokens, dimensions
through 256, and 2-4-bit K/V. Its published M2/Qwen3-4B median generation gain
is 5.39%. Qwen27, Qwen35, and Laguna are shape-eligible; DeepSeek V4's D512
attention is not. The dispatch is exact relative to the same TurboQuant cache,
but that cache is approximate relative to the source target and the result says
nothing about longer contexts. Open [issue #522](https://github.com/ml-explore/mlx-swift-lm/issues/522)
reports that tagged releases re-prefill a large `instructions:` prompt on every
turn. Current main likely avoids the GPU repeat through #472 suffix
reconciliation, but it lacks a dedicated regression receipt. Require warm turns
to prefill only the uncached suffix before claiming session-level speed.
**[published, verified]**

### Target map

| Target | Highest-value upstream path | Important exclusion |
|---|---|---|
| Qwen3.6-27B dense | #3842 numerical gate; #1559 packed GDN; #1486/#1596 exact cache mechanics | #4023 is MoE-only; #4077 excludes head-dim 256 |
| Qwen3.6-35B-A3B | #4023 with #3922/#4009 correctness, then #3842/#3888 numerical gates and #1559 | current native MTP acceptance is weak; #4171 needs much wider rows |
| Laguna-S-2.1 | #1437/#1619 cache correctness and #1704 loading, then #4023 with #3922/#4009 | MLX-LM #1531 targets Laguna-XS, not Laguna-S |
| DeepSeek-V4-Flash-0731 | exact #1662 cache-chain fix, source-faithful checkpointing, Indexer, DSpark, and device/host cost attribution in the dedicated runtime | sorted RHS is excluded: default `unsorted_exact` diverged at layer-0 routed-down row 59 |

Capacity changes the order for the two largest targets. Laguna's 8-bit weights
are about 116.33 GiB and leave unsafe runtime headroom; the official NVFP4-MLX
artifact is about 66.96 GiB and is the source-trusted 128 GiB starting point.
Its model architecture advertises 1M context, but the NVFP4 model card's
serving point is currently 262K; a 1M MLX claim still needs evidence.
DeepSeek-0731's official 155.43 GiB payload cannot fit before workspace, so a
source-bound target quantization is a prerequisite rather than an optional
speed experiment. **[calculated, published]**

#3552 reports a large sliding-prefill gain on an M3, but its bfloat16
accumulation order can flip near-tie argmaxes and it has no M5 receipt. #1437
should be tested first because correct wrapped rotating-cache reuse is a
prerequisite; #3552 is a separate numerical/quality-gated acceleration.
**[published, verified]**

Three service-side branches are exact and useful, but solve different scales:

- [request-local forward contexts, #1706](https://github.com/ml-explore/mlx-lm/pull/1706)
  add stable prefill/decode/draft/verify metadata around graph construction
  without changing the no-callback hot path. This is the clean upstream seam
  for DSpark cost attribution and request-local kernel policy, not a speedup by
  itself.
- [linear-time prompt-trie search, #1607](https://github.com/ml-explore/mlx-lm/pull/1607)
  preserves traversal and tie semantics while replacing repeated path copies.
  Its M5 report improves an 8K divergent suffix from 156.5 to 3.43 ms and a
  32K suffix from 2.822 s to 13.57 ms. This accelerates cache selection, not
  model inference.
- [shared-sampler batch vectorization, #1540](https://github.com/ml-explore/mlx-lm/pull/1540)
  reports 1.08x at B8 and 1.15x at B32 on a 0.6B model, but only noise on a
  30B-A3B target. It is a high-concurrency small-model win, not a reason to
  promise faster B1 coding.

Open [copy-on-write prefix forks, #1547](https://github.com/ml-explore/mlx-lm/pull/1547)
is the strongest newly found server-memory candidate. On M5 Max with
Qwen3-0.6B-4bit, one 8,192-token prefix and eight consumers reduced resident
growth from 7.73 GB to 0.21 GB and first continued-token latency from 15.6 to
7.2 ms; first tokens matched. The evidence is real-model but small-model, the
PR has no checks or review, only plain `KVCache` is supported, and transient
peak rose about 1 GB because every attention step concatenates prefix and
tail. Port it only after a full hybrid-state identity gate and either a bounded
concat cost or two-segment attention. **[published, verified]**

[Sparse Prefix Caching](https://arxiv.org/abs/2605.05219) improves the
checkpoint design for recurrent and hybrid targets. It stores exact recurrent
state only at selected prefix positions, restores the deepest hit, and
recomputes the unmatched suffix. Its placement optimizer uses the measured
overlap-depth distribution; fixed intervals are only a baseline. Before disk
persistence, trace real MTPLX overlap depths and simulate prompt-end, LRU,
fixed-interval, sparse-placement, and a Marconi-style expected-saved-compute-
per-byte admission policy. Append-only chat may need only the newest state,
while shared-document and agent prefixes can justify multiple checkpoints.
**[published, hypothesis]**

Any disk-backed design then needs three separate measurements: snapshot load
without compute, suffix recompute without load, and overlapped load plus
recompute. [Cake](https://proceedings.mlr.press/v267/jin25d.html) demonstrates
the overlap idea on discrete NVIDIA memory/storage paths; unified memory does
not guarantee the same benefit. **[published, hypothesis]**

For long hybrid batches, #1632 is the smallest graph-bound fix. #1642 adds
deferred host metadata, stricter serialization, and broader cache tests; it is
a larger review surface rather than a prerequisite to the smaller fix.
[Copy-on-extract, #1701](https://github.com/ml-explore/mlx-lm/pull/1701)
prevents a one-row recurrent-state view from retaining its entire former
batch. These are lifecycle gates: a benchmark that runs for 100 tokens does
not substitute for proving bounded residency across many batch insertions and
evictions. **[published, verified]**

### Why DeepSeek V4 needs its own cache

DeepSeek V4 does not expose a conventional transformer KV cache. The local
source-faithful path combines a 128-token raw ring, compressed history,
compressor pending state, indexer state, quantization metadata, DSpark stage
state, and an absolute timeline. A safe persistent checkpoint must capture or
reconstruct all of them and bind the snapshot to the exact model, tokenizer,
quantization, adapter, and cache geometry.

The runtime already has exact in-memory rollback primitives. The remaining
work is portable capture, identity, serialization, restoration, and complete
cancellation/eviction lifecycle coverage. Generic `QuantizedKVCache` is not a
substitute.

It is not automatically a capacity win either. MLX-LM
[#1587](https://github.com/ml-explore/mlx-lm/issues/1587) reports that on an
M4 Max, quantized KV increased whole-run peak memory and reduced decode speed
at 8K and 32K because float attention and quantize/dequantize scratch dominate
the stored-byte saving. [#1573](https://github.com/ml-explore/mlx-lm/issues/1573)
shows a separate lifecycle failure: a server can start healthy and then throw
on its first real request when `RotatingKVCache.to_quantized()` is reached.
Every cache experiment therefore records steady-state residency separately
from run peak and must exercise first request, ring wrap, trim, serialization,
and restore. Unsupported cache types fail closed at startup. **[published]**

For approximate long-context capacity, the strongest directly reusable Metal
reference found is [Open-TQ-Metal](https://arxiv.org/abs/2604.16957), which
keeps INT4 K/V compressed inside attention rather than materializing full-
precision K/V. Its M1 Max kernel evidence does not establish M5 speed,
representative long-context quality, or sampled-distribution identity. Treat
it as a Laguna capacity experiment after the exact checkpoint work, not as a
default target path. **[published, hypothesis]**

There is also a concrete decode-lifecycle defect before checkpoint promotion.
MLX-LM [issue #1662](https://github.com/ml-explore/mlx-lm/issues/1662) shows that
discarding `RotatingKVCache.update_and_fetch`'s values result leaves a dead lazy
`slice_update` chain. DeepSeek V4 leaks one Metal buffer per layer per generated
token and reportedly reaches the resource-count limit near 11.5K output tokens
even while byte-memory metrics remain healthy. For its `K == V` path,
`cache.values = cache.keys` drops the dead chain without changing attention;
the reporter validated 13K byte-identical tokens. The dedicated runtime needs
that model-side fix plus a beyond-threshold soak receipt. A general library fix
would periodically evaluate every cache leaf that may otherwise be excluded
from the logits graph. **[published, verified]**

### Why sorted MoE gather is not a DeepSeek production path

The specialized sorted-RHS route requires GatherQMM `M == 1`, `B >= 16`,
`right_sorted`, and `B / E >= 4`; BM32 is then selected when integer total
routed rows per expert is below 64. MLX-LM SwitchGLU supplies `M=1`, flattens
`B=token_rows*top_k` routed assignments, and sorts when `B >= 64`. The
target-shape rows/expert are:

| Target routing | Q512 | Q1024 | Q2048 | Q4096 |
|---|---:|---:|---:|---:|
| Qwen35 top-8 | 16 | 32 | 64 | 128 |
| Laguna top-10 | 20 | 40 | 80 | 160 |
| DeepSeek top-6 | 12 | 24 | 48 | 96 |

For target-verify rows 8 through 16, Qwen35 and Laguna are sorted and meet the
`B >= 16` term, but reach only `B/E=0.25..0.5` and `0.3125..0.625`; the route
therefore fails the density term. That makes it a prefill or high-concurrency
server opportunity, not a single interactive DSpark optimization.
Operationally, DeepSeek remains
excluded regardless: its source-faithful default is `unsorted_exact` after
sorted execution diverged at layer-0 routed-down row 59. **[verified]**

## Local M5 Max evidence

The structured receipt is
[`benchmarks/results/mlx-m5-research-20260810/summary.json`](../../benchmarks/results/mlx-m5-research-20260810/summary.json).
All comparisons used an Apple M5 Max with 128 GB unified memory. The receipt
contains the named fixture, checkpoint-shard, wheel, `libmlx`, core-extension,
and Metal-library hashes for the experiments that recorded them; it does not
certify any unlisted dependency.

### Current MLX versus the installed baseline

An A/B/B/A process-isolated microbenchmark compared installed MLX 0.32.0 with
current MLX main plus the unrelated large-row safety fix from
[MLX #3922](https://github.com/ml-explore/mlx/pull/3922). Identical serialized
inputs produced identical serialized outputs in both arms. **[measured]**

| Micro-path | Baseline median | Current median | Result |
|---|---:|---:|---:|
| sorted many-expert pipeline | 0.7137 ms | 0.5567 ms | 1.282x |
| independently timed pre-sorted QMM pair | 0.6328 ms | 0.6120 ms | 1.034x |
| large NVFP4 QMV | 0.2962 ms | 0.2894 ms | 1.023x |

The first two timings are independent scopes. Their values cannot be
subtracted or used for an Amdahl decomposition. The 1.282x result spans the
whole core update and must not be attributed to #4023 alone.

### Per-command-buffer resource retention on real models

MLX [#4048](https://github.com/ml-explore/mlx/pull/4048) moves input and sibling
buffer retention from one completion handler per evaluated operation to the
command encoder's per-command-buffer lifetime. The original PR head was based
on an older commit. The measurement therefore used a synthetic rebase whose
binary diff contains only the PR's three Metal files. The Python extension and
Metal library hashes were identical across arms; only `libmlx.dylib` changed.
Focused eval, memory, and threading tests passed 19/19. **[verified]**

On Qwen3.6-35B-A3B, one A/B/B/A envelope failed same-arm stability and is
retained as adverse evidence. A second block produced identical 128-token
greedy continuations and unchanged shard manifests in all four processes. Its
base decode midpoint was 87.62 tok/s and the rebased midpoint was 102.59 tok/s,
or an observed 1.171x ratio. The candidate always occupied the middle B/B
positions, the Codex renderer and GPU service remained materially active, and
the patch directly removes host bookkeeping. Arm, order, and host state are
therefore inseparable. The arithmetic is valid, but no patch speed magnitude
is inferred. **[measured]**

The dense Qwen3.6-27B control was 0.989x on decode and had 5-10% same-arm
variance. It supplies no positive magnitude. The full receipt is
[`mlx_pr_4048_real_model.json`](../../benchmarks/results/mlx-m5-research-20260810/mlx_pr_4048_real_model.json).
Before promotion, rebase onto live main and run predeclared randomized,
balanced ABBA and BAAB blocks on a quiet host. Preserve raw per-block timings,
timestamps, thermal and memory pressure, competing-process snapshots, and
loaded-image provenance. Then combine the patch with the donated dynamic-slice
fence fix [#4099](https://github.com/ml-explore/mlx/pull/4099), compiled-function
cache lifecycle repair [#4096](https://github.com/ml-explore/mlx/pull/4096),
and the replacement Metal error propagation work
[#4174](https://github.com/ml-explore/mlx/pull/4174). The #4174 contract can
still surface an OOM on a later stream operation instead of the failing
`mx.eval`. These patches touch the same resource-lifetime neighborhood, so
passing focused tests in isolation does not prove the combined lifecycle
contract. FP8 artifact loaders also need the exhaustive E4M3 decoding fix in
[#4164](https://github.com/ml-explore/mlx/pull/4164): current `mx.from_fp8`
maps NaN encodings `0x7f` and `0xff` to finite values. **[published]**

### The new 32-row QMM tile on an actual Qwen shape

MLX #4171 was built at its exact base and head and tested with one serialized
Qwen dense-MLP fixture: M up to 64, K=5120, N=17408, fp16 activations, and
affine 4-bit group-64 weights. M=16 exercises the new tile; M=48 is an
unchanged same-process control. Every output hash matched across both arms.
**[measured]**

Four base processes were stable: M16/M48 ranged 0.9987-1.0049. Four PR-head
processes were bimodal: two normalized runs were about 0.80, while two were
near parity at 0.99-1.03. The midpoint of normalized ratios suggests 1.117x,
but the 0.792-1.029 spread is the decision. It is not a defensible standalone
speed claim.

Dispatch inspection supplies the more important limit: this M5 Max uses QMV
below M=13 for the tested wide matrix. Current depth-3 and depth-5 verification
produce M=4 and M=6, so they never reach the changed QMM kernel. A real-model
test is warranted only after a block-12-or-wider proposal lane wins on
acceptance economics. That future test also needs a kernel trace and thermal
controls. The receipt keeps both fast and adverse cells rather than averaging
the instability away.

### Historical exact combined-kernel wheel

MLX #3838, #4171, and #3842 were then cherry-picked without conflict onto one
isolated base and built as a single wheel. #3838 later closed unmerged, so this
wheel is a pinned research artifact rather than a current upstream candidate.
Four focused upstream tests passed.
Every shape used one serialized fixture and process-isolated A/B/B/A ordering.
The full receipt is
[`exact_kernel_stack.json`](../../benchmarks/results/mlx-m5-research-20260810/exact_kernel_stack.json).
**[measured]**

| Target-shaped attention call | Base | Stack | Result | Arithmetic |
|---|---:|---:|---:|---|
| 27B, Q512/K8192, H24/4, D256 | 3.6508 ms | 3.6693 ms | 0.995x | bit-identical negative control |
| 27B, Q1024/K8192, H24/4, D256 | 6.9429 ms | 4.8559 ms | 1.430x | max abs delta 1.22e-4 |
| 27B, Q2048/K8192, H24/4, D256 | 13.6655 ms | 9.0993 ms | 1.502x | max abs delta 1.22e-4 |
| 35B-A3B, Q2048/K8192, H16/2, D256 | 9.5327 ms | 6.1449 ms | 1.551x | max abs delta 1.22e-4 |
| 27B, Q9/K16384 verify | 1.6189 ms | 1.0093 ms | 1.604x | max abs delta 6.10e-5 |
| 27B, Q16/K16384 control | 1.2950 ms | 1.2282 ms | 1.054x noise | bit-identical; #3838 does not dispatch |
| 35B-A3B, Q16/K16384 control | 0.8709 ms | 0.8771 ms | 0.993x | bit-identical; #3838 does not dispatch |
| Laguna-S, Q9/K16384 verify | 1.8394 ms | 1.2910 ms | 1.425x | max abs delta 1.22e-4 |
| Laguna-S, Q13/K16384 verify | 1.8371 ms | 1.6490 ms | 1.114x | max abs delta 1.22e-4 |
| Laguna-S, Q16/K16384 verify | 1.8791 ms | 1.9134 ms | 0.982x | max abs delta 1.22e-4 |

The Q512 control proves the #3842 routing boundary rather than merely showing
that two builds have different global performance. At Q1024 and Q2048, both
real Qwen head layouts gain 1.43-1.55x in the targeted attention operation.
The bfloat16 results are numerically close but not byte-identical, so this is a
GO for real-model continuation, greedy-token, quality, memory, and end-to-end
prefill gates—not a release result.

The wide-verification result is deliberately adverse as well as positive.
#3838's full predicate includes
`GQA * ceil(query_rows / max_rows_per_simdgroup) <= 32` and
`query_rows <= key_rows`. Its multi-row branch accepts mask and causal forms;
it needs `key_rows >= 1024` only when the ordinary full kernel supports the
head dimension. Thus Qwen 27B's D256 cells are conditional at Q9-Q10 only on
`query_rows <= key_rows`; Qwen 35B-A3B is statically excluded in Q9-Q16; and
Laguna's D128 cells remain conditional on the long-key branch throughout the
band. The measured Q9 calls gain 1.60x and 1.42x, but Laguna falls to 1.11x at
Q13 and regresses at Q16. Current D3/D5 verification has only four or six rows
and cannot use the route.
The CPU-only 288-cell source/config audit is summarized in
[`dispatch_truth_table.json`](../../benchmarks/results/mlx-m5-research-20260810/dispatch_truth_table.json).
That v1 file published only a digest, so its hash was not independently
reproducible. The replacement file
[`dispatch_matrix_v2.json`](../../benchmarks/results/mlx-m5-research-20260810/dispatch_matrix_v2.json)
publishes all 288 cells, its canonicalization rule, sixteen self-checks, and
the generator hash. Its corrected v5 schema records the pinned MLX-LM
SwitchGLU consumer source and derives routed-assignment geometry from each
target's recorded top-k. It also records unrepresented call-shape
and runtime conditions as `conditional`, rather than treating them as true. It
preserves the envelope but does not claim a computable cell-by-cell comparison
with the opaque v1 matrix.

#4171 was present in the wheel but no SDPA call exercises QMM. More
importantly, its measured M5 QMM boundary begins at M13. That makes the two
patches disjoint on Qwen 27B, while Qwen 35B excludes #3838 entirely. They
overlap only on Laguna M13-M16, exactly where the attention gain fades. The
commits are mechanically compatible; the original compound-speed hypothesis
is rejected for the current Qwen targets and remains unproven for Laguna.

The historical stack was then run through complete real-model prefills. A short
process-isolated A/B/B/A envelope avoided the severe monotonic drift seen in a
discarded eight-repeat pilot. The receipt is
[`real_model_prefill.json`](../../benchmarks/results/mlx-m5-research-20260810/real_model_prefill.json).
**[measured]**

| Historical target, 2,048-token prefill | Base | Stack | Speedup | Numerical result |
|---|---:|---:|---:|---|
| Qwen3.6-27B dense 6-bit | 2,929.0 ms | 2,862.8 ms | 1.023x | same next token and 16-token continuation; TV 0.0089 |
| Qwen3.6-35B-A3B 4-bit | 532.4 ms | 514.1 ms | 1.035x | same next token and 16-token continuation; TV 0.0682 |

Patch isolation attributed the historical 35B gain and logit change to #3842:
#3842 alone measured 1.039x, while #4171 alone measured 1.008x with
byte-identical logits and continuation.

The exact live #3842 head was refreshed on 2026-08-11 at
`31f1555a80d529a0ab27441733293b37ada1223b`, against its exact base
`158118bb152f59ad1aef0d8c835c49317ff9637b`. Both focused head-dim-256 tests
passed. Every loaded weight shard is individually hashed in the receipt.
**[measured, verified]**

| Current #3842, 2,048-token prefill | Base | PR head | Result |
|---|---:|---:|---|
| Qwen3.6-35B-A3B 4-bit | 488.5 ms | 473.8 ms | 1.031x; same 16-token continuation; TV 0.06815 |
| Qwen3.6-35B-A3B 4-bit, hardened repeat | 762.6 ms | 707.9 ms | 1.077x; same 32-token continuation; exact shard manifest; 4.5% head spread |
| Qwen3.6-27B dense 6-bit | 3,170.6 ms | 3,229.4 ms | no claim; head process medians differed by 27.7% |

At the real Q2048/K2048 35B attention shape, the current isolated kernel
measured 2.046x; the two complete-model envelopes measured only 1.031x and
1.077x. That is the Amdahl bound in practice. The direction repeated, but the
second envelope was not precise enough to replace the first magnitude. On
Qwen35, head `31f1555a80d529a0ab27441733293b37ada1223b` changes one-step
logits by up to 1.1328 and has total variation 0.06815. The 16- and 32-token
greedy continuations happened to match; they do not establish identity or
quality preservation. #3842 therefore remains an open review-required,
multi-prompt, long-generation, state, memory, and task-quality experiment, not
a runtime default. **[measured]**

The same core comparison was then run through MTPLX with real local models,
greedy sampling, and a warmed A/B/B/A sequence. **[measured]**

| Target | AR decode | MTP decode | Token result |
|---|---:|---:|---|
| Qwen3.6-35B-A3B 4-bit, D1 | +3.28% | +4.70% | MTP identical; AR token hashes differed |
| Qwen3.6-27B dense 6-bit, D3 | +1.95% | +1.87% | AR and MTP identical |

The MoE AR divergence is a release blocker for an *exact implementation*
claim, even though both streams were deterministic and coherent. It does not
prove a quality regression, but it does prove that a core bump is not a
drop-in byte-identity update. A multi-prompt logit, state, lifecycle, and task
quality gate is required.

### Packed GDN

The exact PR head for MLX-LM #1559 passed all eight focused upstream tests on
this M5 Max. In a real Qwen3.6 dense D3 MTPLX run, packed on/off produced the
same AR and MTP token hashes, but changed decode throughput by less than 1%.
That fails the predeclared 3% decode gate. **[measured]**

At 2,048-token prefill, the A/B/B/A run was directionally positive, with the
midpoint of run averages at about +8.2%. The sequence also showed severe
monotonic thermal drift, so this is not a standalone magnitude claim. It is
consistent with the PR's published +6.4% end-to-end prefill result. The kernel
should remain a prefill-only candidate until a thermally controlled envelope
repeats the result. **[measured, published]**

### Apple M5 runtime ceiling

Apple's [M5 TensorOps and Metal Performance Primitives guidance](https://developer.apple.com/videos/play/wwdc2026/330/)
and the [BaseRT M5 paper](https://arxiv.org/abs/2607.19438) reinforce the
highest-upside runtime path: exact prefill/NAX work that can use M5 matrix
hardware. BaseRT's engine is proprietary, so its paper is a performance-ceiling
signal rather than portable kernel evidence. Its Apache-2.0 converter and
profile schema are public, however, and were audited at commit
`bbded2c2fdb5e8fc1893fbbc5853a9e131250034`. They provide useful fail-closed
tensor-glob profiles and source/calibration provenance fields, but not a ready
AWQ implementation source: conversion applies the selected weight rotation
without writing the inverse activation scales into tensor metadata;
`AwqProfile.check_fingerprint()` is never called; `--calib`, `--calib-tokens`,
and `--awq-mode` do not feed `QuantContext`; and `clip_grid` is defined but
unused. The 14 `base-awq` tests and 16 selected `base-quant` tests pass because
they cover isolated search/packing and profile resolution, not that end-to-end
runtime contract. Reuse the profile/provenance ideas only; do not import this
AWQ path until an end-to-end source-logit oracle passes. **[verified]**

This does not invalidate BaseRT's published speed measurements: catalog
artifacts can be produced by private calibration tooling, and the proprietary
engine is a separate product. It does mean the public converter cannot yet be
used as reproducible evidence for intelligence-preserving low-bit allocation.

## Compression: what to do first

### 1. Compress the drafter

This section is a gated model-side experiment plan, not a demonstrated speed
result. **[hypothesis]** Qwen's trained MTP heads, Laguna's official DFlash
sidecar, and DeepSeek's DSpark stages can all be quantized more aggressively—or
omitted—without changing target intelligence when acceptance is correct.
Measure acceptance and total cost per committed token; nominal draft size is
not the decision.

Profile the entire DSpark round before quantizing it. The current DeepSeek path
moves a complete `[B+1, vocab]` result to the CPU, so host transfer, target-head
work, replay, or cache handoff may dominate even when drafting looks expensive
in isolation. If drafting is at least 10% of the warmed round, quantize DSpark
before changing DeepSeek V4 target weights. Sweep 4-, 3-, and 2-bit stage
projections while retaining confidence/Markov and shared target components at
their prescribed precision. A worse drafter may lower acceptance, but exact
target verification preserves target intelligence.

Promote only if the candidate:

- saves at least 4 GiB and leaves at least 6 GiB working headroom;
- retains at least 90% of baseline accepted tokens per target pass;
- improves end-to-end throughput by at least 10%;
- passes the same target-distribution and lifecycle tests as the uncompressed
  drafter.

This is model-side work. It should be handed to the model-training/conversion
agent with a frozen official-source manifest and held-out prompt set.
Target-only generation remains DeepSeek's baseline until DSpark clears both
end-to-end speed and lifecycle gates.

The public
[`DeepSeek-V4-Flash-0731-2.4bit-mixed`](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed)
artifact at revision `10001e0065f8394e03e968e652cbbe7cd2ca122c`
establishes a useful 128-GB-class control: its card reports 92.8 GB on disk,
84.4 GB short-context peak memory, 31.5-36.1 tok/s single-stream decode, and
all three DSpark blocks retained. The same card reports MMLU-Pro 0.573 versus
0.647 for a hosted BF16/API reference, with no source-relative code, agentic,
or long-context quality gate. It proves that the model can fit and run; it
does not satisfy the no-intelligence-loss objective. **[published, verified]**

### 2. Use calibrated mixed precision for conventional models

MLX-LM already documents mixed 2/6-, 3/4-, 3/6-, and 4/6-bit recipes plus
dynamic quantization, AWQ, GPTQ, and DWQ. See the official
[learned-quantization guide](https://github.com/ml-explore/mlx-lm/blob/254d153fdeb6f150edd4fc5a54f9828638481fa8/mlx_lm/LEARNED_QUANTS.md).

For Qwen/Fable coding models, mixed 8/6-bit is the low-risk compressed frontier
and calibrated 4-bit is the practical speed/memory knee. Keep embeddings, the
LM head, routers, attention projections, recurrent parameters, and
sensitivity-selected first/last layers at higher precision. Use 3-bit first on
routed experts or isolated low-sensitivity tensors, not as a blind dense-model
default.

When a target was quantization-aware trained, preserve its training lattice
instead of requantizing from min/max. MLX-LM
[#1646](https://github.com/ml-explore/mlx-lm/pull/1646) reproduces Google's
Q4_0 grid as ordinary affine 4-bit weights and reports better KL and top-1
agreement at the same size on dense Gemma 4 models. The author found no
significant MoE benefit, so this is a model-specific calibration principle,
not a generic MTPLX recipe.

### Low-bit target boundary

Low-bit speed reports are not source-relative quality proof. MLX-LM issue
[#1450](https://github.com/ml-explore/mlx-lm/issues/1450) bundles mixed-bit
recipes with several decode proposals, but its aggregate comparison changes
both precision and runtime mechanisms. It does not isolate their gains or
provide a source-relative quality gate. PR
[#1466](https://github.com/ml-explore/mlx-lm/pull/1466) also provides no such
proof and reports nonsensical 7B output. [PolarQuant, #1059](https://github.com/ml-explore/mlx-lm/pull/1059)
is primarily a capacity path: a naive 0.5x comparison is not a speed result,
and sequencing/server issues remain. Where #1555's roughly 4.5-average-bit
option is available, it is a safer low-bit experiment than the observed 3-4
bit collapse, not a promotion of approximate target weights. **[published,
verified]**

Two current proposals establish a narrower, useful quality-preservation rule.
MLX-LM [#1646](https://github.com/ml-explore/mlx-lm/pull/1646) and its Swift
counterpart [#507](https://github.com/ml-explore/mlx-swift-lm/pull/507)
reconstruct Google's signed-extremum Q4_0 lattice as ordinary affine W4/g32.
They report better BF16-relative KL and top-1 agreement at identical size on
four dense Gemma 4 targets, plus exact packed-code/scale agreement against the
shipped Q4_0 grid. The same calibration had no statistically significant
benefit on Gemma 4 26B-A4B. The transferable lesson is not “use Q4_0” for
Qwen or DeepSeek; it is to treat a source checkpoint's trained lattice as
provenance that conversion must preserve. **[published, verified]**

Three attractive shortcuts are excluded by current evidence. Closed MLX-LM
[#1676](https://github.com/ml-explore/mlx-lm/pull/1676) fused MoE gate/up
projections, but the author withdrew the +5.2% decode claim after controlled
reruns ranged from +4.7% to -2.1%; real-size independent projections already
overlapped. Swift [variance-normalized KV, #329](https://github.com/ml-explore/mlx-swift-lm/pull/329)
explicitly remains slower than the default cache. MLX [#4051](https://github.com/ml-explore/mlx/pull/4051)
is a valuable non-transposed QMM correctness and training acceleration, but
stock quantized inference uses the transposed path and therefore does not gain
from it. **[published, verified]**

Issue [#1497](https://github.com/ml-explore/mlx-lm/issues/1497) reports a 17%
n-gram speculative gain only on the experimental `quant2-all` setup. Its claim
that the relative gain should reproduce or grow on standard precision is an
untested extrapolation. It also needs full hybrid-state rollback evidence
before it can serve as an exact Qwen baseline. **[published, verified]**

### 3. Treat KV compression as model-specific

[Asymmetric K8/V4, #1550](https://github.com/ml-explore/mlx-lm/pull/1550)
reports about 24% less cache memory than K8/V8 with a small measured quality
cost on a 30B target. [Hadamard-rotated KV, #1555](https://github.com/ml-explore/mlx-lm/pull/1555)
is the strongest published smart-compression candidate found for outlier-heavy
Qwen keys. At 4K context, its independently measured K5/V4 and K6/V3 settings
use 4.5 average cache bits and reach perplexity 23.15 and 22.98, respectively,
versus 22.39 for FP16. Symmetric K4/V4 still degrades to 28.82 even after
rotation, and K3 collapses. The published decode overhead is about 2%.
These are source-relative perplexity results, not coding or agentic quality
evidence. Three MLA-like checkpoints showed no benefit, so this is a gated
Qwen/Laguna experiment rather than a DeepSeek V4 cache replacement.
**[published, verified]**

[Fused quantized-KV prefill, #1510](https://github.com/ml-explore/mlx-lm/pull/1510)
reports a 2.66x 10K-prompt TTFT improvement on an M5 for a conventional Qwen
cache. It is a compelling capacity path when full-precision KV prevents a
request from fitting, but it is not an exact-speed path: the cache values and
attention result change. It also does not replace DeepSeek V4's raw ring,
compressed history, pending compressor state, or Indexer state.

The rotating quantized-cache work in
[#1584](https://github.com/ml-explore/mlx-lm/pull/1584) remains blocked for
hybrid models: it does not recursively convert `CacheList` leaves, and current
review evidence identifies mixed cache cohorts and merge failures. Treat it as
approximate capacity work, not an exact Qwen, Laguna, or DeepSeek path.
**[published, verified]**

Two fused kernels narrow, but do not remove, the Apple penalty. Merged Swift
[TurboFlash #520](https://github.com/ml-explore/mlx-swift-lm/pull/520) uses one
dispatch for 2-4-bit K/V only while total context is at most 128 tokens and
head/value dimensions are at most 256. It reports +5.39% median generation on
M2/Qwen3-4B. Open MLX [quantized SDPA #3026](https://github.com/ml-explore/mlx/pull/3026)
fuses dequantization into attention only when query rows are at most eight and
`query_rows * GQA <= 32`; the current PR is conflicting and has no checks.
That makes Qwen27 eligible through five rows, Qwen35 through four, and Laguna
through five, while DeepSeek V4's GQA64/custom cache is excluded. Both kernels
remain approximate relative to an uncompressed source target. #3026 needs a
clean rebase plus M5 speed, quality, and peak-memory gates before it can enter
the experimental queue. **[published, verified]**

The capacity order is model-specific. At 262K context, Qwen3.6-35B-A3B needs
only about 5 GiB of full-precision cache, while dense 27B needs about 16 GiB.
Laguna reaches about 48.1 GiB at 1M and is therefore the first KV8/KV4 target.
DeepSeek's native HCA/CSA state is only about 6.7 GiB at 1M, so replacing it
with a generic approximate cache remains low priority. **[calculated]**

### SRFT + INT4 KV: published, but unavailable as an implementation source

[SRFT + INT4 KV cache](https://arxiv.org/abs/2605.05699) reports that a fused
path can beat FP16 on small Gemma/Qwen checkpoints, but it is approximate and
model-sensitive. The paper says its full cache/kernel code is released at
[`aminems/AppleSiliconFFT`](https://github.com/aminems/AppleSiliconFFT), yet
public main `5d0d51dbd983691ee99822ed74bc3f9a47136511` contains only generic
FFT/SAR/DNA code. It has no `SRFTInt4Cache`, fused KV quantizer, Python bridge,
`RESULTS.md`, or raw cache results, and no other public branch or tag supplies
them. Record it as an unavailable, unreproducible candidate—not an
implementation source. **[published, verified]**

### Compressed Qwen35 targets: four measured no-speed promotions

Four compressed artifacts were compared with the pinned stock affine 4-bit
target. The performance protocol used process-isolated 512-token prefill and
127 teacher-forced decode forwards on one synthetic prompt, with two process
receipts per arm in A/B/B/A order. Every process consumed the same frozen
continuation, bound by token and source-receipt hash. That fixes token IDs and
forward count only: quantization may change hidden states, so expert assignments
were unobserved and may differ across artifacts. The earlier natural-greedy
receipts are retained as behavioral smokes, but their cross-artifact decode
percentages are workload-confounded and are not used below. **[measured]**

The historical HumanEval NLL receipts independently tokenized every artifact
and retain neither tokenizer-asset hashes nor per-task prompt/solution token-ID
hashes. Their producing source is unavailable. Equal token counts are not proof
of equal tokenization, so the recorded NLL percentages and task signs are
**NOT COMPARABLE** across artifacts and are not decision evidence. The raw
numbers remain in the linked receipts solely for audit. **[measured, corrected]**

| Candidate | Prefill vs stock | Decode vs stock | Historical NLL status | Decision |
|---|---:|---:|---|---|
| ParoQuant | -4.50% | -39.60%; candidate repeat spread 23.21% | NOT COMPARABLE | No speed promotion; precise penalty indeterminate |
| Majentik NVFP4 | -1.40% | -3.21%; candidate spread 1.01% | NOT COMPARABLE | No speed promotion for this pre-fix artifact |
| OptiQ mixed 4/8-bit | -3.30% | -13.92%; candidate spread 1.36% | NOT COMPARABLE | No speed promotion |
| MLX-Community DWQ 4/8-bit | -0.98% | -13.67%; candidate spread 0.35% | NOT COMPARABLE | No speed promotion |

ParoQuant 0.1.16 inserts a pairwise Givens-rotation Metal kernel before each
quantized projection, then calls stock `mx.quantized_matmul`; its MoE wrapper
also rotates both gate/up input and down-projection input. The pinned checkpoint
contains 130 direct projection rotations plus 40 shared gate/up and 40 shared
down MoE rotations. Its 20,655,904,040 weight bytes are also 1.24% above the
stock artifact. That extra M=1 work
is consistent with the measured decode penalty, but it is a mechanism-level
explanation, not an isolated attribution experiment. **[verified, inferred]**

The historical NVFP4 NLL arm was not bit-repeatable across processes: 24/32
artifact-local task losses changed, with a maximum per-task change of 0.0254
nat/token. This does not establish a stock-relative regression, because the
cross-artifact tokenization contract is missing. It reinforces the NOT
COMPARABLE classification rather than supplying a quality gate. ParoQuant,
OptiQ, and stock repeated exactly within their own historical arms.

The NVFP4 artifact has an important provenance boundary. Its four weight
shards were published at Hugging Face commit
`55b0b778953193473078c4418919d399e7bce694` on 2026-04-21. The later pinned
revision only changes its card. Merged MLX
[#3934](https://github.com/ml-explore/mlx/pull/3934) corrected NVFP4 scale
reduction on 2026-07-30: SIMD groups after the first had shared a maximum
across two independently scaled 16-value groups. The PR reports Qwen35
WikiText-2 perplexity improving from 6.848 to 6.778 after the fix, versus a
6.432 BF16 reference. The artifact's own card says it was generated through
`mlx_lm.convert(... q_mode="nvfp4", q_group_size=16)`, so the local NO-GO is
evidence against this pre-fix checkpoint, not against corrected NVFP4. A fresh
conversion from the pinned source must be produced and rerun through the same
speed, repeatability, and quality gates before judging the method. Loading the
old shards under a fixed runtime cannot repair their already stored scales.
**[published, verified, measured]**

As of the 2026-08-11 Hub API and file-history sweep, no target artifact has
demonstrated weight regeneration after #3934. The Qwen27 NVFP4 weights date to
2026-07-07, Laguna NVFP4 to 2026-07-23, and Qwen35 to April; later card updates
are not replacement weights. **[verified, search-bounded]**
The structured inventory is
[`hf_artifact_refresh.json`](../../benchmarks/results/mlx-m5-research-20260810/hf_artifact_refresh.json).

Open MLX [#3912](https://github.com/ml-explore/mlx/pull/3912) is a second
generic NVFP4 prerequisite. Matrix-prefill and gather kernels can silently
corrupt legal dimensions that are 16 modulo 32, while decode appears healthy.
Its current head has 28 successful checks and remains review-required. A
post-#3934 conversion must either prove every quantized reduction dimension is
32-aligned or run on a core containing #3912; unsupported shapes should fail
closed until then. **[published, verified]**

The current Majentik artifact uses NVFP4 group size 16. MTPLX's custom verify
lanes accept only affine-like group sizes 32, 64, or 128, so this checkpoint
falls through to stock MLX today. That makes the measured path safe but slow.
The patch still keys on `bits` and group size rather than `mode`; a future
NVFP4 group-32 artifact could reach code that requires affine `biases`. Add an
explicit `mode == "affine"` guard before claiming generic QQLinear support.
**[verified]**

These comparisons are intentionally compressed-artifact versus
compressed-artifact. The fixed-token/forward-count timings show no speed
promotion for these exact artifacts. They do not establish BF16-relative or
stock-relative quality, because the historical likelihood records are not
comparable. Corrected NVFP4 remains untested. Full receipts:

- [`bundle_manifest.json`](../../benchmarks/results/mlx-m5-research-20260810/bundle_manifest.json)
- [exact v4 timing-producer source](../../benchmarks/provenance/research_paroquant_q35_fixed_work_v4_producer_1578f6f8.py) — historical only; its routing interpretation is superseded by this section
- [`compressed_artifact_sources.json`](../../benchmarks/results/mlx-m5-research-20260810/compressed_artifact_sources.json)
- [`paroquant_q35_m5_performance.json`](../../benchmarks/results/mlx-m5-research-20260810/paroquant_q35_m5_performance.json)
- [`paroquant_q35_m5_fixed_work_performance.json`](../../benchmarks/results/mlx-m5-research-20260810/paroquant_q35_m5_fixed_work_performance.json)
- [`paroquant_q35_humaneval_nll.json`](../../benchmarks/results/mlx-m5-research-20260810/paroquant_q35_humaneval_nll.json) — historical, NOT COMPARABLE
- [`nvfp4_q35_m5_performance.json`](../../benchmarks/results/mlx-m5-research-20260810/nvfp4_q35_m5_performance.json)
- [`nvfp4_q35_m5_fixed_work_performance.json`](../../benchmarks/results/mlx-m5-research-20260810/nvfp4_q35_m5_fixed_work_performance.json)
- [`nvfp4_q35_humaneval_nll.json`](../../benchmarks/results/mlx-m5-research-20260810/nvfp4_q35_humaneval_nll.json) — historical, NOT COMPARABLE
- [`optiq_q35_m5_performance.json`](../../benchmarks/results/mlx-m5-research-20260810/optiq_q35_m5_performance.json)
- [`optiq_q35_m5_fixed_work_performance.json`](../../benchmarks/results/mlx-m5-research-20260810/optiq_q35_m5_fixed_work_performance.json)
- [`optiq_q35_humaneval_nll.json`](../../benchmarks/results/mlx-m5-research-20260810/optiq_q35_humaneval_nll.json) — historical, NOT COMPARABLE
- [`dwq_q35_m5_performance.json`](../../benchmarks/results/mlx-m5-research-20260810/dwq_q35_m5_performance.json)
- [`dwq_q35_m5_fixed_work_performance.json`](../../benchmarks/results/mlx-m5-research-20260810/dwq_q35_m5_fixed_work_performance.json)
- [`dwq_q35_humaneval_nll.json`](../../benchmarks/results/mlx-m5-research-20260810/dwq_q35_humaneval_nll.json) — historical, NOT COMPARABLE

DWQ contains 240 W4 and 272 W8 affine modules. It keeps most switch-expert
matrices eligible for MTPLX's W4 kernels, but its protected attention and
shared-expert paths still make decode slower. The Hub revision publishes no
teacher, calibration fixture, conversion command, or source-relative quality
report. The local NO-GO therefore closes this artifact, not the DWQ method.

Issue [#1700](https://github.com/ml-explore/mlx-lm/issues/1700) reports that
the fused Metal KL backward can overflow and corrupt DWQ parameters above
roughly 1B parameters on an M5 Max while the reference KL path completes.
That report is not locally reproduced. Any MAPS or DWQ calibration run must
compare fused and reference losses and gradients before trusting its trained
precision map. **[published]**

The new source-pinned
[`EigenLabs/Qwen3.6-35B-A3B-MLX-4bit-g64-router8`](https://huggingface.co/EigenLabs/Qwen3.6-35B-A3B-MLX-4bit-g64-router8)
artifact at revision `a52e548cb2f96f6e1180d50ee8e61b4018a4f921`
is a reproducible MAPS calibration control: 432 W4/g64 modules, 80 W8/g64
routers, and 19,508,787,456 tensor-payload bytes. Its card's limited BF16
smoke reports 90.70% top-1 agreement, mean KL 0.2206 nat, and final DeltaNet
state relative error 0.139. It removes vision and MTP and has no task-quality
gate, so it is neither a quality-preserving winner nor a speculative target.
**[published, verified]**

Its new vision-preserving sibling at revision
`4cb5661d5273cb35e6bb8e261154f1eb156f63ea` restores all 333 source vision
tensors in BF16 and publishes shard hashes, conversion/runtime versions, and
two pixel-grounded inference fixtures. The 19.00-GiB payload is a useful
provenance template and shows that selective preservation can keep the visual
tower practical. It still omits MTP, shares the same 90.70% text top-1 smoke,
and has no population task or multimodal quality gate. It improves artifact
completeness, not the speed/intelligence verdict. **[published, verified]**

The open MLX-LM #1623 conversion bug did not explain these candidate deltas.
Three representative norm tensors matched the stock artifact byte-for-byte
after float32 expansion in stock, NVFP4, OptiQ, and DWQ. This is only a
spot-check; it does not prove source-checkpoint fidelity or validate every
norm. Receipt:
[`qwen36_norm_shift_spotcheck.json`](../../benchmarks/results/mlx-m5-research-20260810/qwen36_norm_shift_spotcheck.json).
**[measured]**

### Compression methods that remain research, not candidates

[Atlas](https://github.com/ml-explore/mlx/discussions/3798) measures a finite-
calibration quality cost for each block and allocates a mixed-bit budget. Its
current SmoothQuant adapter is Llama-shaped and does not cover Qwen MoE,
Laguna, or DeepSeek V4. Its additive loss model is an approximation.

The public Apple-native
[`DeepSeek-V4-Flash-0731-iQ-MLX-3.3bpw`](https://huggingface.co/ddalcu/DeepSeek-V4-Flash-0731-iQ-MLX-3.3bpw)
at revision `8a1ae85f8da7a816fc70dbc9e91e409026e5b28d` is the closest direct
offline-allocation baseline. Its card describes per-expert activation
statistics from 2.9M tokens, a weighted affine scale/bias search, and greedy
error-reduction-per-byte allocation across layer/projection bit widths. The
public converter implements the weighted search and per-projection override
grammar, and emits stock-affine-compatible packs. The artifact retains DSpark,
but its stated approximately 118-GB residency leaves no safe 128-GB room for
those draft stages. The public package does not contain the calibration corpus,
imatrix hash, complete allocator trace, or source-relative task-quality gate.
It therefore narrows MAPS's claim substantially but remains a reproducibility
and quality baseline, not evidence of preserved intelligence. **[published,
verified]**

The fresh Majentik RotorQuant and TurboQuant NVFP4/MXFP4 repositories do not
establish another method win. RotorQuant and TurboQuant checkpoints share the
same shard hashes within each format, and their own affine cards call the names
release labels rather than distinct algorithms. The new 200-example ARC-Easy
and HellaSwag card cells are not source-relative or agentic gates; their NVFP4
scores do not beat the plain NVFP4 sibling. Avoid another 20-GB benchmark until
a stronger card, source-relative loss result, or genuinely distinct checkpoint
appears. **[published, verified]**

OptiQ's new SSD-streaming 2-bit artifacts solve a different problem. The
[DeepSeek V4](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit)
card reports 92.5 GB on disk, approximately 6.5 GB resident, and about 2.5
tok/s on M3 Max; the
[Laguna-S](https://huggingface.co/mlx-community/Laguna-S-2.1-OptiQ-2bit)
card reports 41 GB on disk, 9.4 GB resident, and about 3 tok/s. Both stream
routed experts from SSD, publish no capability score, and explicitly describe
the 2-bit experts as lossy. They are remarkable capacity demonstrations, but
SSD-bound decode plus missing quality gates makes them controls for fit, not
answers to the speed-with-intelligence objective. **[published]**

DASHQ INT2 and Escha W2 save substantial capacity, but neither uses stock MLX
affine weights. DASHQ has no M5 path. Escha is a real custom backend, yet its
M=2-6 verifier remains on per-row GEMV and misses MTPLX's W4 NAX lane. Escha's
own card reports LiveCodeBench 62.6 versus 67.0 for FP8. Both stay opt-in
research until target-specific quality and multi-row timing pass.

A new 2026-08-11
[`dogfoodai` trellis MLX repack](https://huggingface.co/dogfoodai/Qwen3.6-35B-A3B-Escha-W2-trellis-mlx)
at revision `c968c7c72de5f328f1d4818c092fb9e29f281bc9` makes Escha's
12.1-GB custom format directly relevant to Apple Silicon. It requires draft
oMLX [#2592](https://github.com/jundot/omlx/pull/2592), whose pinned head
`e40cf4d5869d6c20d702cc25fb9a0761900e116e` currently fails all three CI jobs
during collection. The card's single-stream and concurrency figures are
author-reported, not local receipts. Its bit-exactness is relative to the Escha
W2 runtime, while the published task score is already below FP8. This is a
valuable capacity/backend experiment, not evidence that a 2-bit Qwen target
preserves source-model intelligence. **[published, verified]**

Two stock-compatible Qwen35 artifacts change the next download order.
[`WinterCharm/Qwen3.6-35B-A3B-wMix58`](https://huggingface.co/WinterCharm/Qwen3.6-35B-A3B-wMix58)
at `a746d280f0830780b0bf139346168502676f4e11` is a 24.28-GiB stock-MLX
mixed-precision candidate with matched-perplexity card evidence. It does not
pin the exact source revision or publish broad capability evaluation, so it is
**TEST**, not preserved-intelligence evidence. The current
[`AutomatosX AXQ`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP)
weights at `b057b2ecde64724b37d9130d34a000bf098bc731` have different primary
shard OIDs from the revision evaluated by their card. That head is **SKIP
pending recertification**; old quality and MTP claims do not transfer.
**[published, verified]**

The original 11.45-GiB
[`EschaLabs W2`](https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2)
at `6072a54913552616b64599c30f5e8a3c8fecad9c` remains the strongest extreme-
fit Qwen35 candidate for a controlled Apple benchmark. Its own LiveCodeBench
score falls from 67.0 for FP8 to 62.6 and the exact BF16 source revision is not
pinned. The disposition is **GO for measurement**, not quality-equivalent or
production-ready. **[published, verified]**

### Additional mechanisms: ordered by cheapest falsifier

- [ReplaySSM](https://dao-lab.ai/blog/2026/replayssm/) reassociates recurrent
  updates in finite precision. Current public reports show numerical deltas and
  little or no B=1 whole-model benefit; gains concentrate at larger batches.
  Treat it as a numerical-change concurrency experiment, not an exact
  interactive-decode priority. **[published]**
- [VeriCache](https://arxiv.org/abs/2605.17613) composes a compressed-KV
  drafter with authoritative full-KV verification. Its overlap argument relies
  on transfer and compute paths that are cleaner on discrete accelerators than
  unified memory, while retaining both caches can increase pressure. Run a
  residency and cycle-cost upper bound before implementation. **[published,
  hypothesis]**
- Early-exit proposals such as [KnapSpec](https://arxiv.org/abs/2602.20217)
  and [HiSpec](https://arxiv.org/abs/2510.01336) preserve final target
  verification, but published hybrid-Qwen ablation evidence is adverse. Reject
  a port unless a cheap probe reaches 25% depth-two acceptance while reading
  no more than 25% of target bytes. **[published, hypothesis]**
- Static allocator controls must include oMLX
  [oQ](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md),
  calibration-free [AlphaQ](https://arxiv.org/abs/2606.04980), expert-level
  [GEMQ](https://github.com/jndeng/GEMQ), hardware-aligned
  [ScaleBITS](https://arxiv.org/abs/2602.17698), and dynamic-expert
  [DynaExq](https://arxiv.org/abs/2511.15015). Router fine-tuning creates a new
  target and stays outside source-faithful comparisons. **[published]**

DeepSeek V4 requires a stronger source-fidelity contract than a generic
"4-bit" label. Its official package uses MXFP4 E2M1 routed-expert values with
E8M0 group-32 scales, while its compressed-attention indexer has separate FP4
simulation points and keeps indexer KV in BF16. A generic affine MLX repack is
therefore a new approximate target unless it proves equivalence to those
trained lattice and rounding semantics. **[published, verified]**

Every capacity receipt must also record process physical footprint, MLX peak,
compressor state, and swap state. These metrics can diverge materially on
unified memory; MLX [#3896](https://github.com/ml-explore/mlx/issues/3896)
reports an M5 case where the MLX peak counter was roughly 46 GB while process
footprint was roughly 110 GB. **[published]**

### Domino: the next Qwen27 drafter experiment **[published, hypothesis]**

[Domino](https://arxiv.org/abs/2605.29707) now has an
[official implementation](https://github.com/jianuo-huang/Domino) and a public
[Qwen3.6-27B checkpoint](https://huggingface.co/Huang2020/Qwen3.6-27B-Domino).
It preserves DFlash's block-parallel proposal while adding a small causal GRU
correction head. The official repository reports better acceptance/throughput
than DFlash on A100-class hardware, but no Apple result. This makes it a port
and measurement candidate, not a claimed M5 win.

Run Domino and z-lab DFlash against the identical Qwen27 target, prompts,
sampler, lifecycle code, and blocks 8 and 16. Kill the port unless Domino
reduces measured cost per committed token by at least 10%. No Qwen35 Domino
checkpoint is public yet, so the existing 386M z-lab DFlash remains that
target's available drafter.

The Modal Qwen35 DFlash repository is an exact weight mirror of z-lab and does
not add a second candidate. For Laguna, poolside's BF16 and INT4 DFlash
payloads are genuinely distinct, but both Hub repositories declare
`license: other` and contain no license file. They are technically testable
through oMLX, but not redistribution-cleared. DeepSeek V4's three embedded
`mtp.*` DSpark stages in the official checkpoint remain that model's
provenance anchor. **[verified]**

Before designing a progressive-bit Metal format, also test the simpler
composition supported by Apple's
[QuantSpec](https://machinelearning.apple.com/research/quantspec): use a Q4
copy as the proposal model and keep BF16 as the authoritative verifier. Proceed
only if combined residency stays at most 110 GiB, draft cost is at most 35% of
a BF16 target pass, block-4 or block-8 acceptance reaches 0.85, and whole-
request speed reaches 1.15x BF16 autoregressive decode. **[hypothesis]**

### MAPS-MLX: the research direction supported by these failures **[hypothesis]**

The strongest next project is **Measured Allocation and Proposal Scheduling
for Apple Silicon (MAPS-MLX)**. It would jointly choose a fixed, quality-gated
precision layout and an exact proposal/depth policy from the same measured M5
cost surface. The objective is cost per committed token:

```text
(restore + draft + verify + commit cost) / expected committed tokens
```

The novelty boundary is search-bounded, not universal. We do not claim novelty
for mixed-precision allocation, hardware-aware quantization, compression-aware
draft depth, proposal-family routing, recurrent rollback, or cost-aware draft
trees in isolation. [MxMoE](https://arxiv.org/abs/2505.05799) co-optimizes MoE
block/expert precision using quantization sensitivity, activation frequency,
profiled hardware cost, and generated mixed-precision GroupGEMM kernels.
[CAS-Spec](https://arxiv.org/abs/2510.26843) routes among quantized or sparse
self-drafters and assigns draft lengths from acceptance and latency estimates.
[SpecKV](https://arxiv.org/abs/2605.02888) makes draft depth compression-aware.
[QSpec](https://arxiv.org/abs/2410.11305),
[SpecMQuant](https://arxiv.org/abs/2505.22179), and related work combine
quantization with lossless target verification.
[Sequoia](https://arxiv.org/abs/2402.12374) and
[CaDDTree](https://arxiv.org/abs/2606.01813) optimize draft trees against
hardware or measured verification cost, while
[EcoSpec](https://arxiv.org/abs/2607.12696) prices marginal MoE expert
activation. **[published]**

The public Apple-specific
[`mlx-mtp` hybrid](https://github.com/junainfinity/mlx-mtp/blob/1e26698c706e4356647e03d7ca6d1cff21f4904e/mlx_mtp/hybrid.py)
already selects MTP versus DFlash per round using a tokens-per-second EWMA and
a shared verify/rollback loop. It resets DFlash state on switch-in and captures
the union of required hidden layers. Its quantization policy is rule-based and
global rather than a learned per-bank layout, and it does not include prompt
lookup or DSpark. This is the closest direct MLX counterexample and must be the
controller baseline. **[verified]**

The iQ-MLX artifact above is the corresponding Apple-native *allocator*
baseline. MAPS may not claim activation-aware affine fitting, per-projection
bit assignment, or greedy error-per-byte allocation. Its remaining systems
question is whether measured Metal dispatch cost and exact proposal-arm
interaction justify a different deployment-fixed layout and online policy.
Keeping multiple layouts resident would have to charge duplicated weights,
compilation, switching, and cache invalidation.

Two additional prior-art clusters change the implementation order. For dense
targets, [QSpec](https://arxiv.org/abs/2410.11305) uses W4A4 drafting and
W4A16 verification with shared W4 weights and KV; it is not a general
subset-of-weight-bits design. [SPEQ](https://arxiv.org/abs/2510.18525) does
construct its drafter from target-bit subsets, but its gain depends on bespoke
hardware. Progressive-bit Qwen27 is therefore a conditional Metal-microkernel
hypothesis, not the strongest ready candidate. Test the plain Q4-copy/BF16-
target composition first. If progressive-bit work follows, Metal must execute
the draft representation at least 1.25x faster and exact stochastic acceptance
must use its true proposal distribution.
MLX's merged [small-batch `qmv_wide`, #3764](https://github.com/ml-explore/mlx/pull/3764)
already supplies the M=2-8 verification primitive: the published M5 Max INT4
kernel gain is about 1.4x/1.7x/1.6x at M=2/4/8 over per-vector QMV. Those are
isolated kernel figures, not a progressive-bit model result. **[published,
hypothesis]**

For MoE targets, generic cost-aware depth control is already occupied.
[Cascade](https://arxiv.org/abs/2506.20675) disables harmful speculation and
tunes K from token gain versus verification cost; [EVICT](https://arxiv.org/abs/2605.00342)
selects a lossless cost-effective draft-tree prefix from profiled MoE verify
cost; EcoSpec prices marginal expert activation;
[MoE-SpeQ](https://arxiv.org/abs/2511.14102) combines an INT4 MoE drafter,
expert lookahead, mixed precision, scheduling, and hardware-aware depth; and
[BASTION](https://arxiv.org/abs/2605.29727) performs online hardware-cost
control. MAPS must therefore model
the *per-layer union of routed experts and bytes*, not merely draft depth. The
decisive primitive is a fused mixed-bit expert kernel: separately dispatching
protected and compressed banks may erase the theoretical traffic saving.
Approximate [AcceptMoE](https://arxiv.org/abs/2608.02989) is an upper-bound
competitor because constraining verifier expert eligibility changes the target
distribution. **[published, hypothesis]**

Our claim is narrower: among the primary-source papers and public
implementations checked through 2026-08-11, we found no Apple/MLX system that
co-calibrates (1) one deployment-fixed, per-bank target precision layout and (2)
an exact online choice among AR, prompt lookup, MTP, DFlash, DSpark, and draft
depth using the same measured Metal cost-per-committed-token surface. MAPS is
therefore an unimplemented Apple systems co-design hypothesis, not a new
quantization or speculative-decoding algorithm. Kernel eligibility, complete
hybrid state, and source-relative quality are promotion gates, not novelty
claims.

MAPS-Q35-v0 is the smallest falsifiable artifact:

- keep routed-expert banks homogeneous and eligible for profitable M=2-6
  affine-W4 verification kernels;
- measure source-teacher KL/NLL, top-k changes, router preservation, and the
  most important bank interactions on frozen calibration data;
- measure the actual M5 kernel and latency for every phase, row count, shape,
  precision mode, bit width, and group size;
- freeze one precision map for the whole request, then let the runtime choose
  only exact work-saving arms: restored checkpoints, AR, prompt lookup, MTP,
  DFlash, or DSpark;
- invalidate learned costs whenever the model, MLX, `libmlx`, metallib, or
  quantization map changes.

The expert-union model must predict complete-model verification cost as
`C_verify(M, u_hat, layout)`, where `u_hat` is a causal prediction of per-layer
routed-expert sets and multiplicities. The actual target expert union is
future information and belongs only in the offline oracle. Stop
unless held-out cost error is at most 5% median and 10% p95. A candidate
expert-aware policy must reduce expert bytes per cycle by at least 10%, keep
accepted tokens per round within 1%, and improve whole-request throughput by
at least 8% over the best fixed exact arm. A mixed-bit expert kernel must beat
split dispatch by at least 15% at M=2-6 and improve the whole model by at least
5%, without regressing M=1 by more than 2%. **[hypothesis]**

Phase 1 stops unless observed dispatch matches the registered table and a
candidate predicts at least 5% over stock and DWQ. Phase 2 stops unless the
materialized model stays within 0.5% source-relative NLL, avoids material
coding/agent regressions, improves target forwards by at least 3%, and improves
mixed-workload goodput by at least 10% without a bucket losing more than 3%.
Phase 3 tests whether the unchanged planner generalizes to dense Qwen and one
non-Qwen target. If each target requires a hand-written policy, MAPS is only a
benchmark collection and its architecture claim fails.

The first factorial experiment must measure every frozen layout against every
proposal arm from identical checkpoints. Kill MAPS unless the target-layout by
proposal-policy interaction improves whole-request goodput by at least 5% over
the best independently composed layout and scheduler, with the 95% confidence
interval above zero. A cross-arm replay oracle must then compare emitted tokens,
target logits or declared numerical tolerances, KV offsets, recurrent/conv
state, rejection rollback, and subsequent continuation. Finally, offline
counterfactual replay must compare the best fixed arm, the public EWMA hybrid,
the measured-cost controller, and a future-knowing oracle while charging
switch, reset, restore, and commit costs. If oracle uplift over the best fixed
arm is below 10%, do not build the live controller. **[hypothesis]**

Target mixed precision remains approximate relative to the source model.
Proposal selection, depth adaptation, and checkpoint restoration must remain
exact relative to that fixed target. Promotion therefore has four independent
gates: target-distribution exactness, serial numerical identity or an explicit
tolerance contract, lifecycle/state identity, and source-model fidelity.
Correct speculative acceptance does not prove source-model fidelity, and close
target logits do not prove that the drafter declared its true proposal law.

### MLXcel M5 Max ledger: capacity evidence, not compressed-KV speed

Apache-2.0 [MLXcel](https://github.com/lablup/mlxcel/tree/45dea248926c5d0a8f09bdfb2ce1d21aed8d504a) at
`45dea248926c5d0a8f09bdfb2ce1d21aed8d504a` provides a valuable M5 Max
evidence ledger in its [model tests](https://github.com/lablup/mlxcel/blob/45dea248926c5d0a8f09bdfb2ce1d21aed8d504a/docs/benchmark_results/model_tests_m5max.md)
and [benchmark report](https://github.com/lablup/mlxcel/blob/45dea248926c5d0a8f09bdfb2ce1d21aed8d504a/docs/benchmark_results/benchmark-report.md).
Its advertised 2.70-2.78x `mlx-lm` prefill median is a stale May, same-host
campaign on the tiny “Hello, how are you today?” prompt with `mlx-lm` 0.31.3
and MLX 0.31.2; the current v0.4.0 documentation explicitly says the baseline
was not rerun. Treat that as native-host-overhead/short-prompt evidence, not a
long-context kernel-speed claim. Decode is the stronger comparison and is
roughly 99-100% median parity overall, not a broad speedup. **[published,
verified]**

The M5 TurboQuant ledger is especially important adverse evidence:

| Mode | 4K decode / FP16 | 16K decode / FP16 |
|---|---:|---:|
| int8 KV | 0.719x | 0.572x |
| turbo4-asym | 0.090x | 0.061x |
| turbo4 | 0.205x | 0.106x |
| delegated | 0.269x | 0.054x |
| turbo3 | 0.063x | 0.029x |

At 8K prefill, int8 KV reaches 1.09x and delegated 1.204x, but most compressed
modes are slower. Later FP16-fast-path sidecars recover about 0.98-1.04x decode
only by retaining full FP16 K/V, so they are not compressed-only results. On
M5, aggressive KV compression remains capacity work until a genuinely fused
SDPA-inner-loop kernel proves otherwise; 3-bit is a clear NO-GO, and int8 is a
conservative capacity choice rather than speed-free. MLXcel's MTP Gemma data is
model-specific and does not supersede the Qwen measurements in this note.
**[published, verified]**

## A separate approximate tier: ASD

[Approximate Speculative Decoding](https://arxiv.org/abs/2608.03447) is the
most relevant new speed/quality tradeoff found in this sweep. It accepts a
bounded number of low-target-logit-regret draft mismatches, then reuses later
draft tokens that are target-greedy under the now-realized prefix. A
request-wide regret ledger, per-token gate, and per-block mismatch cap limit
the approximation. It needs no extra target pass or drafter training.
**[published, verified]**

The paper reports 3.05-15.26% throughput improvement over matched strict
speculation across Qwen/DSpark, EAGLE3, and Medusa experiments. Its
DeepSeek-V4-Flash DSpark transfer raises accepted tokens per proposal about
10-16% on GSM8K and MATH-500, but those eight-H20 experiments do **not**
establish end-to-end speed: FP4 DSpark ran through an FP8 compatibility path,
so the authors explicitly present acceptance and quality only. On a separate
Qwen3-14B campaign, HumanEval changed -0.61 percentage points and MT-Bench
-0.64 points, while more than 95% of GSM8K and MATH completion hashes changed.
This is useful controlled approximation, not lossless acceleration.
**[published]**

An Apple implementation would be novel, but it must be a separate opt-in
service tier:

- default and release gate remain strict, distribution/greedy exact;
- approximate-regret mode requires an explicit per-request budget;
- budget zero must reproduce strict token IDs and cache states;
- speed is measured with fixed work, while quality uses natural EOS;
- every promoted profile is model-and-workload-specific;
- report altered-output rate even when task scores are unchanged.

The smallest experiment is verifier-only and does not touch weights: apply the
public Apache-2.0 selector to recorded DSpark target logits, replay strict and
candidate prefix decisions, and estimate target-pass savings. Only if that
offline upper bound exceeds 5% should it enter the live MLX path.

### Topology-aware GDN tree verification: implemented, but not free

The earlier PCTree result established only that the local miniature prototype
lacked a packed tree verifier. It did **not** establish that Gated DeltaNet tree
verification was unavailable in the MLX ecosystem. A fresh source audit found
that [`dflash-mlx`](https://github.com/bstnxbt/dflash-mlx) already ships a
best-first DDTree planner, parent-indexed recurrent-state tape, tree-aware
depthwise convolution, attention mask, and accepted-path cache commit. This is
the same systems boundary highlighted independently by
[SpecLA](https://arxiv.org/abs/2607.16673): stateful tree speculation needs a
topology-aware verifier and accepted-state recovery, not an ordinary causal
batch. **[verified]**

The preserved public PCTree result is commit
[`ebb3d845`](https://github.com/PhilipJohnBasile/mlx-serve/commit/ebb3d845b1293308d5fc965e8c198aedbcf19d49).
Its miniature planner is useful negative evidence only: tied scores, cache
ownership, and packed real-model verification were not closed, and the local
DSV4_MINI fixture found no accepted-token gain. **[measured, verified]**

[Bole](https://arxiv.org/abs/2608.01651) invalidates any broader claim that
efficient GDN trees are impossible. It derives a tree-structured closed form,
stores token-level state factors, and reconstructs only the accepted state;
the published GPU implementation reports 3.4-7.7x faster recurrent tree
verification and 82-99x less transient speculative-state memory. That is a
new kernel/runtime design, not the materialized-state tape rejected by the
local oracle. A Bole-to-Metal feasibility study is now the correct long-term
tree lane. **[published, hypothesis]**

The live z-lab Qwen3.6-35B-A3B drafter uses a newer nested configuration layout.
The upstream compatibility fix already exists as
[`dflash-mlx` #46](https://github.com/bstnxbt/dflash-mlx/pull/46); an equivalent
two-field shim was applied only in a disposable checkout for measurement. The
focused DDTree, recurrent-kernel, and Qwen tree tests passed 32/32. A 64-token
greedy smoke produced identical chain and tree token hashes. **[verified]**

The one-prompt M5 Max measurements were adverse for this high-acceptance
drafter:

| Qwen3.6-35B-A3B | Chain/adaptive | DDTree | Tree result |
|---|---:|---:|---:|
| verify cap 6, 512 tokens, A/B/B/A midpoint | 193.60 tok/s | 166.50 tok/s | 0.860x |
| verify cap 16, 256-token directional pair | 224.36 tok/s | 156.94 tok/s | 0.700x |

At cap 16, DDTree improved acceptance by only 0.39 percentage point and added
0.228 committed token per cycle, while verify time rose from 78 ms to 1,308 ms
and drafting from 59 ms to 269 ms. The full receipt is
[`dflash_ddtree_q35_m5.json`](../../benchmarks/results/mlx-m5-research-20260810/dflash_ddtree_q35_m5.json).
This is a **NO-GO for the Qwen 35B default from this directional probe**, not a
population benchmark or a rejection of topology-aware verification. The cap-6
cell is one prompt despite A/B/B/A timing, the cap-16 cell is one pair, and the
runner did not expose every sampler control. Reconsider only with a matched
multi-prompt protocol, materially lower chain acceptance, or a tree-size cost
model that predicts enough benefit to pay the measured recurrent-state cost.
**[measured]**

That conclusion agrees with the broader hardware evidence in
[Lossless but Not Free](https://arxiv.org/abs/2607.17283): consumer-hardware
speculation wins only when verification is genuinely batch-parallel and the
draft/target latency gap is large. It also sharpens the MoE requirement: token
count and the union of activated experts are separate costs. Approximate
expert-budget methods such as [MoE-Spec](https://arxiv.org/abs/2602.16052) and
[AcceptMoE](https://arxiv.org/abs/2608.02989) may reduce that union, but they
change target routing and therefore sit outside this document's exact-quality
frontier.

## Ideas rejected by this audit

- Blanket 1-bit or 2-bit target weights. The single-row bandwidth advantage
  largely disappears in multi-row verification, while quality risk remains.
  MLX [#3161](https://github.com/ml-explore/mlx/pull/3161) is a real 1-bit
  packing/runtime implementation and reports large single-row kernel gains,
  but it explicitly assumes a model already trained or externally quantized
  for 1-bit weights. Its near-zero kernel-vs-dequantized-reference divergence
  proves kernel fidelity, not preservation of the original model. It is
  relevant to native BitNet/PRISM-class checkpoints, not a license to
  requantize Qwen3.6 or DeepSeek V4.
  MLX [#3852](https://github.com/ml-explore/mlx/issues/3852) reports an M4 Pro,
  not M5, boundary: its 2-bit advantage is about 1.7x at one row, then
  disappears by three rows. Treat that as a transfer hypothesis until a pinned
  M5 rerun; the reported economics are adverse for M=2-6 verification.
- MLX-LM [#1479](https://github.com/ml-explore/mlx-lm/pull/1479). Its
  `GenerationBatch` logprobs contract is broken, and it has no MTPLX
  direct-argmax gain. It is rejected rather than a fallback implementation
  path.
- Generic DeepSeek V4 KV quantization. It ignores the source QAT cache contract
  and attacks a bounded component rather than the dominant target weights.
- PCTree promotion from the current miniature result. The corrected CPU planner
  now implements Algorithm 1 and preserves exact `k=1` DSpark selection, but
  the DSV4_MINI experiment still found zero accepted-token improvement and that
  prototype has no packed tree verifier. `dflash-mlx` proves that a separate
  topology-aware GDN implementation exists; the real-model Qwen result above
  shows why mechanism availability alone is not a speed result. The miniature
  result remains preserved negative evidence, not a general proof against
  real-model PCTree.
- A naive materialized-state [STree](https://arxiv.org/abs/2505.14969) port to
  Qwen Gated DeltaNet. A CPU algebra oracle confirmed that parent-state
  gathering is source-order exact, but STree's efficient diagonal scan does
  not transfer: Qwen's transition `g(I - beta*k*k^T)` is dense and
  non-commuting. Materialized fp32 GDN state alone would cost about 2.25 GiB
  for 16 tree nodes on 27B and 960 MiB on 35B-A3B, before convolution tails,
  KV, or activations. This specific representation is NO-GO. Bole supplies a
  different closed-form factorization and therefore reopens a bounded
  Bole-to-Metal feasibility study.
- Fusing MoE gate/up projections for launch-count reduction. MLX-LM
  [#1676](https://github.com/ml-explore/mlx-lm/pull/1676) retracted its original
  speed claim after careful reruns showed -2.1% to +4.7% session variance and
  no reproducible gain; the independent projections already overlap.
- Expert-union fusion. The authors of MLX
  [discussion #3801](https://github.com/ml-explore/mlx/discussions/3801)
  retracted the proposal after in-graph measurement. Qwen35 already averaged
  34.9% expert overlap and 13.2 unique experts at layer 2, yet the gather path
  was within about 10% of the byte-union floor. Split-K, stacked gate/up, and
  union dedup all failed. The production split/stacked ratios were 0.984x and
  0.997x with bitwise-equal output. There is no recoverable launch-count win.
- Skipping empty NAX output groups as an M5 Max priority. MLX
  [#3941](https://github.com/ml-explore/mlx/pull/3941) is merged and has useful
  mobile-tail cases, but its M5 Max measurements were insignificant and its
  tested large shapes had only a 0.63-2.50% work-count ceiling.
- MoE layer skipping. That is a new model requiring a quality program, not a
  runtime optimization.
- Expert deletion, merging, or reduced top-k routing. Those change the target
  model, unlike exact residency management. The local DeepSeek virtual-REAP
  candidates were rejected; MoEspresso retained only 0.652 selection and 0.719
  held-out geometric-mean token probability relative to Gold.
- Disk expert offload as a speed path. MLX-LM
  [#1588](https://github.com/ml-explore/mlx-lm/pull/1588) demonstrates an exact,
  opt-in residency primitive that can run a 35 GiB Qwen MoE on a 32 GiB Mac,
  but its author reports roughly 5x slower decode and 8-12x slower prefill at a
  0.3 resident fraction on a model that already fits. It is a capacity/OOM
  fallback, not a unified-memory acceleration.
- Generic low-rank activation or KV compression before an equal-byte INT4
  oracle. It is approximate and removes learned state; the equal-byte
  comparison should happen offline before any runtime kernel is built.
- Same-shape model merging as compression. Averaging weights does not reduce
  serving bytes. Expert merging, pruning plus distillation, and dense-student
  distillation create a different model and sit outside the exact-quality
  frontier.
- Expert parallelism on one M5 Max.
- `mx.compile` as an exactness argument. It can reorder arithmetic and requires
  mutable state to be explicit.
- `MLX_METAL_FAST_SYNCH=1`. Upstream documents GPU wedges requiring reboot.
- Generic attention kernels whose shape contracts exclude DeepSeek V4's
  64-query-head, one-512-dimensional-KV-head attention with sinks.

## Proposed 30/60/90-day build plan **[hypothesis]**

| Window | MLX/runtime side | Model side |
|---|---|---|
| Days 0-30 | Isolated MLX/MLX-C uplift; DeepSeek parity/lifecycle A/B; DSpark phase profile; device argmax prototype; checkpoint-v1 spec; packed-GDN prefill envelope | freeze source/corpus manifests; map component bytes and sensitivity; build DSpark 4/3/2-bit candidates |
| Days 31-60 | in-memory then disk-backed checkpoints; verified prompt lookup; measured fallback; staged commit only if replay is material | select a drafter by acceptance, cycle cost, and memory; only then test a sensitivity-mapped target mix |
| Days 61-90 | cost controller across target-only/DSpark/prompt lookup; full server ABBA; optional continuous batching | freeze calibration and holdout results; package a passing candidate or publish the negative result |

## Predeclared kill gates

| Work | Stop when |
|---|---|
| MLX uplift | any unexplained token/state/lifecycle mismatch; B=1 slows more than 3%; representative gain under 3% |
| device DSpark acceptance | any first-max/tie mismatch; round gain under 5% or end-to-end gain under 3% |
| transactional commit | rollback/replay is under 20% of round cost; any state mismatch; cycle gain under 10% |
| persistent checkpoint | restore differs from fresh prefill; snapshots exceed 10% of headroom; repeated-prompt TTFT gain under 20% |
| prompt lookup | copy-heavy gain under 1.15x; novel-prose regression over 3% after fallback; any cache mismatch |
| packed GDN | any token/state mismatch; eligible engagement under 95%; geometric-mean gain under 3% in its claimed regime |
| DSpark quantization | less than 4 GiB saved; under 6 GiB free headroom; acceptance below 90% of baseline; speed gain under 10% |
| target quantization | held-out NLL worsens over 0.5%; top-1 agreement drops over 0.5 points; material coding/agent regression |
| continuous batching | B=1 regresses over 3%; any row-state leak; B=4 aggregate gain under 20% |

## Recommended implementation order **[hypothesis]**

1. Expand MLX #3842 to multi-prompt continuation, task-quality, state, memory,
   and long-prefill gates. The isolated kernel and one-prompt 35B speed gates
   passed, but the 27B speed gate and target-distribution identity gate did not.
2. Repeat MLX-LM #1559 under thermal controls, take the narrow #1632 lifecycle
   fix, and consolidate #1486/#1596 cache semantics. Treat #4020 as an
   alternative GDN implementation, not an additive stack.
3. Revisit #4171 only if a block-12-or-wider verifier first wins acceptance
   economics. Treat closed-unmerged #3838 as historical evidence unless a new
   upstream branch supersedes it. Skip #4077 unless head-dim 256 is supported.
4. Integrate only passing MLX, MLX-C, and local DeepSeek extensions in an
   isolated worktree; do not move a submodule pointer over dirty nested trees.
5. Add a complete DeepSeek checkpoint identity and persistence contract.
6. Profile the complete DSpark round, including target body/head, full host
   transfer, replay, and cache handoff. Build device acceptance or drafter
   quantization only if the measured cost decomposition can pay back the work.
7. Add prompt lookup on top of exact restoration, then train the runtime
   controller on measured milliseconds per committed token.
8. If drafting is material, hand DSpark-only quantization to the model-side
   agent with the frozen target-logit corpus and gates above.

That combination is both more original and more defensible than another low-bit
target checkpoint: it is an exact, source-faithful Apple Silicon session engine
that accelerates the workloads where local coding models spend their time.
