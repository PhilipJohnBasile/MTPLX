# Qwen 35B MTP-batch numerics profiles

Status: approved direction; awaiting written-spec review for PR #245

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

## Public interface

The server accepts one construction-time option:

```text
--mtp-batch-numerics throughput|balanced|b1-exact
```

The corresponding config key is `mtp_batch_numerics`. The value is resolved
once while server arguments are constructed and passed explicitly to the B8
lane installer. Generation does not read an environment variable or inspect
the option again.

The option applies only to Qwen 35B `--scheduler-mode mtp_batch`. Selecting
`balanced` or `b1-exact` with another scheduler fails startup with a clear
configuration error. A singleton request continues to use unchanged solo MTP;
the selected profile governs physical B8 cohorts at real widths two through
eight.

The initial default remains `throughput`. `balanced` may become the persistent
Qwen launcher default only after all promotion gates pass. `b1-exact` is always
explicit and has no throughput promise.

## Profiles

### `throughput`

This is the existing PR #245 implementation. Its target verify work uses the
installed B8/T2/M16 route, and its draft work uses B8/T1/M8. No arithmetic,
ownership, sampling, or scheduling behavior changes.

Route identity:

```text
qwen35b_a3b_mtp_batch_b8_t2_m16_throughput
```

### `balanced`

This route keeps B8 scheduling, row-owned caches, batched attention, batched
sampling, and shared weight access. Only operations proven by construction-time
attribution to cause material B1/B8 divergence are replaced with multi-row
callables that preserve the B1 reduction tree and BF16 cast boundaries per row.

The balanced operator set is fixed in source after measurement. It is not
chosen dynamically from a logit margin, row count, or runtime probe. Candidate
operators are promoted one at a time, and each candidate must improve numerical
parity and pass the end-to-end throughput and quality gates before another is
added.

Route identity:

```text
qwen35b_a3b_mtp_batch_b8_t2_balanced
```

### `b1-exact`

This route keeps the B8 scheduler and physical row-owned cache containers but
uses B1-equivalent per-row arithmetic for every target and draft operation that
can change committed model state or token decisions. A multi-row kernel may
share a dispatch or weight tile only when each row retains the same K-reduction
order, accumulator conversions, and BF16 rounding points as the B1 M1/M2
implementation.

The label `b1-exact` is contractual. Installation requires bitwise B1 parity for
the defined construction receipt. A merely bounded numerical result cannot be
published under this name. If the receipt fails, startup fails; the server does
not fall back to `balanced` or `throughput`.

Route identity:

```text
qwen35b_a3b_mtp_batch_b8_t2_b1_exact
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
route specification. Each specification owns:

- its exact target, capture, draft, and MTP-update callables;
- its exact GDN, attention, router, expert, combine, and projection routes;
- its construction self-check;
- its route ID and config fingerprint;
- its session/cache compatibility identity.

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

Prompt prefill remains the unchanged request-local B1 prefill contract. The
profile boundary begins where requests enter the fixed B8 decode lane. Profile
tests must still prove target offsets, MTP-history offsets, recurrent ownership,
inactive-row freezing, and cancellation isolation.

## Health and receipts

Health and completion metadata report:

- requested and effective `mtp_batch_numerics` profile;
- exact profile route ID;
- profile config fingerprint;
- construction receipt verdict;
- real cohort width and physical fixed width.

These are existing request/cohort-boundary statistics. The change does not add
per-token, per-layer, per-cycle, or per-dispatch engagement counters.

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

All profiles must retain the existing PR #245 ownership and serving gates:

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

`b1-exact` must be bitwise equal to unchanged B1 for target/draft outputs,
committed attention K/V, recurrent state, logits, cache offsets, token decisions,
and next RNG state across heterogeneous rows and mixed accept/reject commits.

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
5. Implement the complete B1-equivalent route and expose `b1-exact` only after
   its exact receipt passes.
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
- Logit-margin fallback or replay of selected rows through solo MTP.
- Changing model weights, quantization, MTP depth, sampler semantics, or
  published artifacts.
- Generalizing the profiles to other models without their own geometry,
  construction receipt, and end-to-end measurements.
