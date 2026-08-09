# Qwen 35B MTP-batch numerics profiles

Status: implemented; throughput remains the default; measured balanced candidates
remain explicit and are not promoted

## Goal

Increase the coding accuracy of the fixed Qwen3.6-35B-A3B B8/T2 MTP lane
while retaining most of its aggregate throughput gain. Operators can select a
fully constructed numerics route with one startup flag. Selection never adds a
per-token eligibility check, automatic fallback, or request-time route switch.

The production target for the balanced route is at least 300 aggregate output
tokens per second under the existing greedy eight-request benchmark while
improving both HumanEval+ and MBPP+ over the current throughput route.

## Baselines

The unchanged PR #245 route is the throughput control:

- fixed target shape `[B=8, T=2]`, flattened projection shape `M=16`;
- fixed draft shape `[B=8, T=1]`, flattened projection shape `M=8`;
- greedy median: 321.070 aggregate output tokens per second;
- default-sampler median: 161.500 aggregate output tokens per second;
- HumanEval: 151/164;
- HumanEval+: 144/164;
- MBPP: 335/378;
- MBPP+: 285/378;
- combined plus score: 429/542.

The unchanged serialized solo-MTP B1 quality control scored:

- HumanEval: 151/164;
- HumanEval+: 145/164;
- MBPP: 338/378;
- MBPP+: 289/378;
- combined plus score: 434/542.

The objective is not to relabel B8 arithmetic as the reference. B1 remains the
quality control.

The original PR workload was rerun unchanged: eight marker prompts, seeds
4200-4207, 256 maximum tokens, serial B1 followed by concurrent requests, and
three rounds. The throughput rerun reproduced the old 321 TPS result. The first
balanced candidate changed only the GDN A/B projections. It kept more than 97%
of greedy throughput and exceeded throughput under the default sampler in this
bracket:

| Profile | Greedy B1 | Greedy concurrent | Default B1 | Default concurrent |
| --- | ---: | ---: | ---: | ---: |
| throughput | 159.745 | 323.570 | 141.187 | 170.406 |
| balanced v1: stock A/B | 158.899 | 315.777 | 141.053 | 172.891 |

All values are median aggregate output tokens per second. Every marker remained
row-local. Neither B8 profile reproduced complete B1 response hashes; all three
rounds matched 0/8 responses. That is why balanced is an accuracy profile, not
an exactness claim.

The strict B8-only EvalPlus audit rejected balanced v1. The completed generation
contained 68 B8 cohorts, one B7 cohort, and one B1 request, so the audit is
directional rather than promotable. It scored HumanEval 152/164 and
HumanEval+ 145/164, but MBPP 335/378 and MBPP+ 282/378. The eight tasks from
the non-B8 cohort had the same pass/fail outcomes under B1, throughput B8, and
balanced v1. The MBPP+ regression therefore was not caused by that cohort.
Balanced v1 was removed from consideration instead of being promoted from its
throughput result.

Receipt SHA-256 values: throughput greedy
`8620050108f4016b412fd34fa933be24b2e6c3c2a504758d849a4f5835894517`,
balanced greedy
`6745ad8ac53104ffbbcfe01519495b3e85a7f50979ea5783c0cebe2243f694dc`,
throughput default
`7c4c3446a518a558133fe5f7e3bdeb84263aaba97fb55f6c8fa24a276382e298`,
and balanced default
`0255aaa3182950b6adc3508c1a55209889d7fbe855c62bd3aa89920808c580bc`.

### Corrected B1 control and device-resident greedy route

The `159.745` value in the paired table is not the PR #174 optimized-B1
decode result. It is aggregate HTTP throughput for eight separate B1 requests,
including eight prefills and request overhead. The exact PR #174 K1 harness was
rerun unchanged before comparing B8:

| Control | Repeat 1 | Repeat 2 |
| --- | ---: | ---: |
| long-code natural stop | 197.321 TPS | 198.883 TPS |
| short coding prompt, 143 tokens under the current tokenizer | 207.207 TPS | 207.044 TPS |

The current branch reported the compiled target-prefix route, device draft
input, whole-MoE, GDN post-convolution, packed projections, row-owned router,
and combine-tail as installed. The historical PR #174 checkout reproduced
193.231-196.275 TPS on the same machine. The B1 harness and served aggregate
B8 harness measure different things and remain separate controls.

Auditing that stack found one real B8 omission: greedy B8 materialized full
`[8,V]` draft logits and `[8,2,V]` verify logits on the CPU every cycle. The
new construction-selected greedy sampling route keeps draft IDs on device,
feeds them directly into the B8/T2 verify input, and transfers only small token
ID arrays for primary and verify decisions. Default stochastic sampling keeps
the existing exact batched sparse route. Requests with frequency or presence
penalties keep the dense compatibility route.

The served A/B results are:

| Workload | Previous B8 | Device-ID B8 | Change | Output parity |
| --- | ---: | ---: | ---: | --- |
| coding, 8 x 192 tokens | 414.852 TPS | 452.413 TPS | +9.05% | all 8 hashes exact |
| legacy greedy | 324.019 TPS | 349.064 TPS | +7.73% | all 8 hashes exact |
| legacy default sampler | 170.406 TPS | 166.743 TPS | -2.15% | all 8 hashes exact |

The default-sampler path does not select the new greedy route; its small timing
change is recorded as run-to-run drift, not an optimization claim. Receipt
SHA-256 values are
`b0683bab11fabab4f544f3d34292d92ceb559cf034b5adf5c2cc72fc7a40ef49`
for the coding run and
`8f64743bc745a61f198dc370103d175e6c4b713c6d0573d863323aa69f6a8e47`
for the legacy greedy run.

Two adjacent candidates were rejected:

- forcing the existing unsorted MoE gather route reduced legacy greedy B8 from
  324.019 to 301.040 TPS (-7.09%) and did not change the eight measured B8
  output hashes;
- a construction-only M16 whole-MoE probe improved one real target block from
  0.5308 to 0.5112 ms (1.038x), but changed BF16 output (`max_abs=0.0491`).
  The projected gain did not justify adding a new 40-layer arithmetic route.

## Public interface

The server accepts one construction-time option:

```text
--mtp-batch-numerics throughput|balanced|b1-exact
```

The corresponding config key is `mtp_batch_numerics`. The value is resolved
once while server arguments are constructed and passed explicitly to the lane
installer. Generation does not read an environment variable or inspect the
option again.

The option applies only to Qwen 35B `--scheduler-mode mtp_batch`. Selecting
`balanced` or `b1-exact` with another scheduler fails startup with a clear
configuration error. A singleton request continues to use unchanged solo MTP.
`throughput` and `balanced` govern physical B8 cohorts at real widths two
through eight. `b1-exact` deliberately serializes every sealed request through
that same unchanged B1 runner.

The initial default remains `throughput`. `balanced` may become the persistent
Qwen launcher default only after all promotion gates pass. `b1-exact` is always
explicit and has no throughput promise.

## Profiles

### `throughput`

This is the default PR #245 implementation. Its target verify work uses the
installed B8/T2/M16 route, and its draft work uses B8/T1/M8. Greedy sampling
uses the device-ID route described above. Default stochastic sampling and
penalty-bearing compatibility requests retain their separately bound routes.
Target/draft model arithmetic, cache ownership, and scheduling do not change.

Route identity:

```text
qwen35b_a3b_mtp_batch_b8_t2_m16_throughput
```

### `balanced`

This route keeps B8 scheduling, row-owned caches, batched attention, batched
sampling, and shared weight access. Construction attribution found the first
divergence in layer 0's GDN projections: B8 flattens `[8,2,H]` to M16 while B1
uses M2, changing BF16 accumulation order. The current candidate binds eight
unchanged B1/T2 calls for layer 0's QKV, Z, and B projections. The recurrent A
gate remains on the B8/T2 route, and every projection in layers 1 through 29
also keeps the throughput route. The 24 B1 calls are traced into one fixed
B8/T2 compiled target graph; there is no request-time loop, eligibility check,
or fallback.

The real layer-0 QKV shape is `[8,2,2048] -> [8,2,8192]`, BF16 with 4-bit
affine group-64 weights. Two ten-repeat construction probes measured:

| QKV route | B1 max abs | Median latency, probe 1 | Median latency, probe 2 |
| --- | ---: | ---: | ---: |
| throughput M16 | 0.03125 | 0.1715 ms | 0.1666 ms |
| stock M16 | 0.03125 | 0.2646 ms | 0.2558 ms |
| eight unchanged B1/T2 calls | 0 | 0.2616 ms | 0.2646 ms |
| one batched-weight logical-M2 QMV | 0 | 0.6482 ms | 0.5826 ms |

The existing flattened stock-like QMV candidate was also rejected: it was not
exact (`max_abs=0.046875`) and its median was 0.6130 ms. These measurements
select the eight-call route before any new EvalPlus run. Probe receipt SHA-256
values are
`3685ff3c4cd4d8266e79f0963ca18fb5bd813c199c15e4f92605a431ed68556d`
and
`4a39f2bb96c1c35bdcaa1bcac9741c5f8604014c71ea97b99ef0b559342ace2a`.

The second candidate applied the exact QKV call to all 30 GDN layers and kept
stock B/A. Its installation proved the QKV boundary bitwise, exact layer-0
recurrent output, exact argmax, exact offsets, and exact row isolation. It ran
true B8 in all four concurrent cohorts, but reached only 261.641 aggregate TPS,
or 1.652 times its paired B1 median. This is below the 300 TPS floor, so the
candidate was rejected without running EvalPlus. Its receipt SHA-256 is
`25493b02922894c12998517a787e66198f397151942e25eb49ba9c9a062a9bb1`.

The all-layer candidate needed a 12/128 compiled/eager full-graph BF16 bound
because its measured relative error was 0.09277. The smaller layer-zero route
restores the existing 9/128 bound. The post-convolution numerical, argmax,
ownership, and candidate-boundary checks remain unchanged.

A QKV-only layer-zero candidate was rejected before timing. Its replacement
boundary was bitwise identical to eight B1/T2 calls and its full-graph relative
errors remained within 9/128, but the real construction corpus changed the
compiled/eager, heterogeneous B1, and same-geometry B8 argmax decisions. The
gate failed closed.

The next candidate made all four layer-zero projections B1-exact. It passed the
unchanged construction argmax and ownership gates and reached 322.234 greedy
aggregate TPS, or 99.6% of the throughput control. It was still rejected:
default-sampler throughput fell to 140.654 TPS, below the 153.425 floor. The
same workload used 434 fixed B8 verify cycles versus throughput's 341, showing
that reduced target/draft acceptance—not projection latency—caused the loss.
Its greedy and default receipt SHA-256 values are
`721a44abbfa5e71b1e66dfaa15a17e6494fe3f092c5128d963047ae9bcc6affa`
and
`5506894ff2865626b9533b922b36e5be19ce7d2619b44e321d836c7fd4226b9c`.

The next QKV/B/A candidate restored Z to B8/T2. It failed construction before
timing: compiled/eager relative attention error was 0.09159 and same-B8
optimized/stock attention error was 0.10849, both above the fixed 9/128
(0.0703125) bound. Its QKV/B/A replacement boundaries and argmax decisions
were exact, so the failure specifically shows that Z must remain coherent with
the corrected QKV path. The numerical bound was not relaxed.

The QKV/Z candidate made every construction argmax equal and made layer-zero
convolution state exact, but still failed the unchanged numerical gate. Its
compiled/eager relative errors were 0.08462 for logits and 0.06958 for
attention. Its same-B8 errors were 0.07968 for hidden and 0.07390 for attention.
The limit was 0.0703125. It was rejected before timing.

The QKV/Z/A candidate disproved A as that missing correction. Its
compiled/eager attention error rose to 0.10142, and its same-B8 errors rose to
0.11155 for hidden and 0.10063 for attention. Heterogeneous numerical parity
also failed. It was rejected before timing.

The final candidate added B while leaving A batched. It passed construction but
reached only 259.650 greedy aggregate TPS, below the 300 TPS floor, and was
rejected without promotion. Its receipt SHA-256 is
`2e822ed75011ded21a583ae1495c88a5c23b35ccd156fc2e0dfcca46869f4ca8`.
This exhausted the coherent rowwise-B1 layer-zero projection subsets without
relaxing the construction bound.

The balanced operator set is fixed in source after measurement. It is not
chosen dynamically from a logit margin, row count, or runtime probe. Candidate
operators are promoted one at a time, and each candidate must improve numerical
parity and pass the end-to-end throughput and quality gates before another is
added. It is closer to B1 numerically, but it is not token-exact with B1: later
B8 BF16 reductions can still change a near-tie token and subsequent context.

Route identity:

```text
qwen35b_a3b_mtp_batch_b8_t2_l0_b1_qkv_z_b_balanced
```

### `b1-exact`

This route keeps the same request queue and model-owner thread, but never calls
the B8 driver. Every sealed request is run once through the unchanged
request-local B1 MTP implementation. Failures remain request-local unless
model-owner cleanup fails, which poisons the service before another row runs.

The label `b1-exact` is therefore contractual by construction: the adapter does
not reproduce B1 arithmetic in a new kernel; it invokes the B1 implementation
itself. This preserves token and cache behavior at the cost of serial aggregate
throughput. Health and completion metadata say `serial_b1_exact` and never
claim physical B8 execution.

Route identity:

```text
qwen35b_mtp_batch_b1_exact_serial
```

## Attribution before kernel work

The first implementation step is an offline, construction-shape attribution
receipt. It runs the same prompts, token inputs, cache contents, and committed
positions through unchanged B1, throughput B8, and same-geometry B8 references.
It records the first divergent operation and per-layer maximum absolute and ULP
errors for:

- projection outputs;
- MoE router scores and expert IDs;
- expert outputs and combine results;
- GDN convolution output and recurrent state;
- attention K/V and offsets;
- hidden states;
- target and draft logits;
- argmax and speculative accept/reject decisions.

This attribution runs only in a dedicated construction/benchmark command. It
does not add counters, comparisons, synchronization, environment reads, or
fallback accounting to production generation.

No operator is changed merely because its B1 and B8 shapes differ. The first
candidate must address the first measured material divergence on the real
4-bit affine Qwen 35B shapes.

## Construction architecture

The installer parses the profile into a closed enum and selects one immutable
route specification. Each B8 specification owns:

- its exact target, capture, draft, and MTP-update callables;
- its exact GDN, attention, router, expert, combine, and projection routes;
- its construction self-check;
- its route ID and config fingerprint;
- its session/cache compatibility identity.

The exact specification instead binds the service's multi-request executor to
the unchanged solo runner. Its B8 callable table is retained only for the
startup construction receipt and is never dispatched.

The installer validates the model, dtype, quantization, geometry, callable
table, and profile-specific receipt once. It returns an immutable
`InstalledA3BMTPBatchLane` with the selected callables already bound. The decode
loop invokes those callables directly.

There is no enabled-path `if profile == ...`, no custom-then-stock exception
handler, and no automatic downgrade. Stock or B1 arithmetic used by a profile
is an explicit member of its construction route table.

## Cache and session identity

The numerics profile and route fingerprint are part of every reusable decode
state identity. A target/MTP cache or session-bank entry created under one
profile cannot be restored under another profile. Changing the flag requires a
server restart and creates a new compatibility domain.

Prompt prefill remains the unchanged request-local B1 prefill contract. For the
two B8 profiles, the profile boundary begins where requests enter the fixed B8
decode lane. Exact mode never crosses that boundary. B8 profile tests still
prove target offsets, MTP-history offsets, recurrent ownership, inactive-row
freezing, and cancellation isolation.

## Health and receipts

Health and completion metadata report:

- requested and effective `mtp_batch_numerics` profile;
- exact profile route ID;
- profile config fingerprint;
- construction receipt verdict;
- real cohort width and physical fixed width.

For exact mode, physical width is truthfully reported as one even when eight
requests were sealed together. These are existing request/cohort-boundary
statistics. The change does not add per-token, per-layer, per-cycle, or
per-dispatch engagement counters.

## Error handling

Startup fails before serving when:

- the profile name is unknown;
- a non-throughput profile is selected outside Qwen 35B `mtp_batch`;
- a required profile callable is absent;
- a profile receipt fails its declared parity contract;
- the cache/session compatibility fingerprint is incomplete;
- the installed route ID does not match the selected profile.

An installed profile never switches route after a request has been admitted.
Existing cohort cleanup failure behavior remains fail-closed.

## Correctness gates

Both B8 profiles must retain the existing PR #245 ownership and serving gates:

- eight distinct request IDs and markers with no foreign text;
- exact unaffected-row isolation when another row changes;
- exact row-permutation parity for equivalent B8 inputs;
- exact cache offsets and commit ownership;
- inert padding, completed, and cancelled rows;
- active cancellation with one model-owner cleanup;
- the 13,239-token-per-row long-context gate;
- no AR or stock fallback.

`throughput` retains its existing bounded BF16 and exact argmax construction
contract.

`balanced` must improve B1 parity at every replaced boundary, preserve exact
token decisions in the construction corpus, and pass profile-specific numerical
bounds chosen before the end-to-end quality run.

`b1-exact` must reproduce unchanged B1 output hashes for identical prompts,
seeds, sampling parameters, and token limits. Its construction receipt must say
`unchanged_solo_runner`, and the B8 driver dispatch count must remain zero.

## Quality gates

Quality is measured with EvalPlus 0.3.1, one greedy completion per task, using
the same prompts, hashes, maximum-token budget, and scoring commands already
recorded in PR #245.

The balanced route may be called an accuracy improvement only when:

- HumanEval base is at least 151/164;
- HumanEval+ is at least 145/164;
- MBPP base is at least 335/378;
- MBPP+ is at least 286/378;
- both plus suites improve over the throughput route;
- no response contains another row's task or marker;
- every scored response has an audited physical B8 route.

Matching the full B1 combined score of 434/542 is the goal, but not the minimum
balanced promotion threshold. The PR reports the exact point estimate, paired
swap counts, and exact McNemar p-value without claiming significance that the
sample does not establish.

The `b1-exact` route must reproduce the unchanged B1 program hashes on the
fixed deterministic corpus before its flag is documented as available.

## Performance gates

All model work acquires `/tmp/mtplx-gpu-exclusive.lock`; only Qwen is loaded.
Timing uses an isolated server port with no unrelated request traffic.
Profiler and dispatch-census runs are separate from timed runs.

Each candidate is measured against unchanged `throughput` with the same eight
prompts, seeds, token budgets, stop settings, service configuration, and thermal
conditions. Three paired rounds are required.

Balanced promotion requires:

- greedy median aggregate output throughput at least 300 tokens per second;
- default-sampler median at least 153.425 tokens per second, or 95% of the
  current 161.500 control;
- no regression in request isolation, cancellation, long-context behavior, or
  peak-memory safety;
- the real route ID and B8/M16-or-declared-hybrid dispatch geometry in the
  untimed census.

The `b1-exact` route is measured by the same protocol, but its result is
reported without a throughput floor. It remains explicit and never becomes the
default automatically.

## Rollout

1. Add the startup enum, config key, validation, health fields, and immutable
   route-table plumbing without changing throughput arithmetic.
2. Add the offline first-divergence attribution receipt.
3. Implement and measure one balanced operator candidate at a time.
4. Freeze the smallest balanced operator set that clears all quality and
   throughput gates.
5. Bind `b1-exact` to the unchanged request-local B1 runner and expose it only
   after paired deterministic hashes pass.
6. Run the full changed-area and repository test suites.
7. Add code, benchmark receipts, and the profile table to the existing PR #245.
   Do not create another PR.
8. Change the persistent launcher default to `balanced` only if its gates pass;
   otherwise leave `throughput` as default and report the miss honestly.

## Failure-mode check

### The balanced route overfits the published coding tasks

Severity: critical if task results are used to choose individual arithmetic
branches. Mitigation: operator selection is based on construction-shape
attribution and parity before EvalPlus is run. EvalPlus is a final promotion
gate, not a per-task tuning loop. Every attempted candidate and rejected result
is recorded.

### A profile flag creates hidden hot-path branching or fallback

Severity: critical because it can erase the measured gain and violate the lane
contract. Mitigation: the enum selects a complete route specification during
installation. Source tests inspect the installed driver call graph, and a
dispatch census confirms the selected route outside timing.

### Reusable state crosses numerics profiles

Severity: critical because a request can begin from cache state produced by a
different arithmetic contract. Mitigation: include the profile and route
fingerprint in reusable cache/session identities and reject mismatches before
restore.

### Exact B1 arithmetic erases the B8 speedup

Severity: expected risk, not a correctness failure. Mitigation: `b1-exact`
remains explicit with no throughput promise. Balanced changes one attributed
operator at a time and cannot ship below the declared throughput floors.

### Greedy clears 300 TPS while default sampling regresses materially

Severity: critical for the default product experience. Mitigation: balanced
promotion includes the separate 95%-of-control default-sampler floor.

## Non-goals

- Making B8 the quality reference by routing all singleton requests through
  padded B8 arithmetic.
- Request-level or per-row switching between numerics profiles.
- Logit-margin fallback or replay of selected rows from an active B8 cohort.
- Changing model weights, quantization, MTP depth, sampler semantics, or
  published artifacts.
- Generalizing the profiles to other models without their own geometry,
  construction receipt, and end-to-end measurements.
