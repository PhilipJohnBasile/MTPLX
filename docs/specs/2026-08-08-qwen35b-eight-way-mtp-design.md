# Qwen 35B eight-way MTP serving

Status: approved design for PR #245

## Goal

Serve up to eight independent Qwen3.6-35B-A3B requests through one fixed-width
speculative MTP cohort. Each request keeps its own prompt, sampling state,
logical KV rows, recurrent state, stop state, and cancellation state. The target
and MTP draft work execute in lockstep across the cohort so the model weights are
amortized across requests.

MTP is the default. The server must never change a concurrent request to AR as
an automatic fallback.

## Scope

This change extends the existing PR #245 branch. It includes:

1. A fixed-width A3B MTP driver for the real 35B Speed model.
2. Exact per-request sampling and speculative acceptance at MTP depth 1.
3. Server admission, streaming, cancellation, and error propagation for up to
   eight concurrent requests.
4. A persistent Qwen launcher that starts in MTP mode and installs the width-8
   lane at construction time.
5. Unit, parity, live concurrency, kernel-engagement, and performance evidence.

The implementation is split into reviewable commits in this same pull request.
No second pull request is created.

## Non-goals

- Reusing the dense Qwen 27B width-2 K2 cohort. Its route table and kernels do
  not match the 35B MoE model.
- Forcing MTP depth 2 or 3. The served 35B Speed model uses its promoted depth-1
  contract.
- Changing model weights, quantization, expert layout, or published artifacts.
- Claiming a throughput win from configuration or microbenchmarks alone.
- Adding automatic AR fallback when a cohort cannot be formed or installed.
- Enabling the lane for other model families without their own construction
  contract and measurements.

## Public behavior

The new scheduler mode is `mtp_batch`. Its required service settings are:

- `generation_mode=mtp`
- `load_mtp=true`
- `depth=1`
- `max_active_requests=8`
- `decode_batch_max=8`
- a fixed cohort capacity of eight slots

One request uses the existing optimized solo MTP route. When two or more
compatible requests are ready within the bounded gather window, they enter one
fixed eight-slot MTP cohort. Unused slots remain inert padding rows. Requests
that arrive after a cohort starts wait for the next cohort; mid-run refill is
not part of the first production version.

AR remains available only through an explicit request or explicit service
configuration. The scheduler never selects it because request count increased.

## Construction boundary

The width-8 route is installed once during server construction. Installation
validates:

- Qwen3.6-35B-A3B model and backend identity;
- MTP availability and depth-1 draft-head topology;
- dtype, quantization, group size, expert layout, hidden width, layer count,
  vocabulary width, and target/draft tensor shapes;
- the exact fixed target verify shape `[B=8, T=2]` and its flattened `M=16`
  projection and MoE shapes;
- required optimized target, draft, attention, and MoE callables;
- a numerical self-check against the unchanged route at the real shapes.

The installer returns a typed, immutable lane containing prebound callables and
fixed geometry. An invalid contract fails startup clearly. The enabled hot path
does not re-read environment variables, revalidate model metadata, or try an
optimized route and silently fall back.

## Request ownership

Each admitted request owns one slot for the life of its cohort run. A slot owns:

- prompt tokens and true prompt length;
- row-specific KV offsets and KV contents;
- row-specific GDN/recurrent state;
- latest target logits and hidden state;
- MTP draft state for the current speculative cycle;
- target and draft sampler configurations;
- an independent seeded RNG stream;
- committed tokens, stop matcher, token budget, and finish reason;
- cancellation event and output queue.

The cohort may store these rows in shared batched allocations. Sharing an
allocation is not sharing context: all reads, writes, offsets, masks, rewinds,
and commits remain row-owned.

## Decode cycle

The production lane uses MTP depth 1. For each active row:

1. Sample the next target token `x0` from that row's current target
   distribution.
2. Produce one draft token `d` from the MTP head using that row's hidden state
   and draft sampler.
3. Run one target verify forward over the fixed `[8, 2]` tensor containing
   `[x0, d]` for each slot.
4. Compute speculative acceptance independently per row using the same target
   and draft probability contract as `generate_mtpk`.
5. Apply the same acceptance, target-minus-draft residual sampling, bonus-token
   policy, and commit ordering as `generate_mtpk`, independently per row. A
   rejecting row replays only its own correction through the existing ragged
   fold-in state machine.
6. Keep inactive, cancelled, and completed rows masked and pinned without
   letting them affect another row's decisions.

The cohort follows the installed speculative-bonus policy. When the bonus is
enabled, an accepting row samples it from the matching target distribution with
that request's RNG. When it is explicitly omitted, the cohort omits it too. The
batched path must match `generate_mtpk`; it does not invent a different MTP
algorithm.

All sampler values that may differ by request are frozen in the request state.
The cycle may batch probability materialization, but it must not replace eight
independent RNG streams with one shared RNG sequence.

## Admission and compatibility

The service groups requests only when their construction-time execution
contract is compatible: model, depth, target strategy, tool/constraint route,
and other values that change the target or draft graph. Sampling values and
seeds are row state and may differ within one cohort.

The admission window is bounded. A single request does not wait for seven peers;
it takes the existing solo MTP route. Two through eight ready requests use the
fixed width-8 lane. Waiting requests form the next cohort.

## Streaming, cancellation, and completion

Committed tokens are emitted to the owning request queue only. Every request
future reaches exactly one terminal state: completed, cancelled, or failed.

Cancellation marks the row inactive and completes its future even when cache
filtering or cleanup fails. A partial cache mutation fails the whole cohort
closed through the existing PR #245 boundary; the mutated cohort is never used
again.

Normal batched MTP does not promise a reusable generation-final cache state.
The stream uses PR #245's direct compatibility classification and releases the
client without waiting behind later foreground model work. Any safe history
rebuild remains bounded idle work.

## Telemetry

Telemetry is collected at admission and cycle boundaries without adding
per-layer or per-dispatch proof work to the model hot path. Health exposes:

- scheduler mode `mtp_batch`;
- active lane `mtp_batch_width_8` while a cohort is decoding;
- real request count and fixed cohort capacity;
- a real-width histogram, including width 8;
- target verify cycles, accepted draft tokens, rejected draft tokens, and
  acceptance rate;
- cancellations and last cohort error;
- installed optimized-kernel route identity from construction.

Kernel engagement is also verified outside the measured path with the existing
dispatch census/profiler tools.

## Failure handling

- Construction mismatch: fail server startup. Do not install the lane.
- Cohort forward, draft, sampler, or cache error: fail every unfinished request
  in that cohort, close the cohort, and build a fresh cohort for later work.
- Per-request cancellation: close that request promptly and mask its row.
- Client disconnect: signal the same cancellation event used by explicit
  cancellation.
- Postcommit incompatibility: release the stream and schedule only the existing
  bounded idle history route.
- Insufficient peers: use solo MTP for one request; never substitute AR.

## Verification

### CPU and construction tests

- Red tests first for scheduler selection, no automatic AR fallback, fixed
  width, request ownership, terminal future completion, and startup failure.
- Batched-vs-solo token parity for greedy requests at fixed width 8.
- Exact sampled parity with fixed per-request seeds and mixed sampler settings.
- Mixed prompt lengths, budgets, stops, cancellation points, and completion
  order.
- A rejecting row must not alter an accepting row's tokens, RNG state, cache
  offsets, or recurrent state.
- Construction tests pin all real 35B geometry and optimized callable routes.

### Live guarded gates

All model work acquires `/tmp/mtplx-gpu-exclusive.lock`. Only Qwen is loaded.
DeepSeek stays disabled.

1. Establish unchanged solo MTP baselines three times.
2. Run the width-8 cohort with unique per-lane markers and fixed seeds.
3. Require eight separate request/session IDs, zero foreign markers, all
   terminal events, and a real width-8 histogram.
4. Cancel two rows while six continue; require both cancellation acknowledgments
   and clean completion of the survivors.
5. Cross the prior MLX resource-limit boundary with a long width-8 run.
6. Capture profiler or dispatch-census evidence for the installed M=16 target
   verify and matching draft/MoE routes outside the timed run.
7. Compare aggregate and per-request throughput against unchanged solo MTP.

Promotion requires correctness plus a measured aggregate MTP throughput gain.
The initial target is at least 1.20x aggregate throughput over serving the same
eight requests through unchanged solo MTP, with no context leak, no sampler
drift, no resource-limit error, and no silent stock or AR fallback. If the real
shape misses that target, the lane remains experimental and the service stays
on solo MTP while the receipt is reported honestly.

## Rollout

1. Land core driver and parity tests in PR #245.
2. Land server admission, streaming, cancellation, and telemetry in PR #245.
3. Run the guarded real-model gate without changing the persistent service.
4. Only after the gate passes, change the Qwen launcher default to `mtp_batch`
   and restart it while holding the GPU lock.
5. Verify exact model identity, MTP availability, optimized route identity,
   width-8 behavior, gateway health, DeepSeek disabled state, and lock release.

Rollback points the launcher back to the last known-good solo-MTP runner. It
does not restore `ar_batch` as the default.

## Adversarial review

1. **Critical: one row can corrupt another through a shared cache or RNG.**
   The design requires row-owned offsets/state, independent RNG streams, fixed
   seed sampled parity, and rejecting-neighbour tests before any live rollout.
2. **Critical: the fixed M=16 call silently misses the optimized kernels.**
   Construction pins and self-checks the callable route; external dispatch
   evidence and end-to-end A/B are promotion gates. There is no enabled-path
   fallback.
3. **Critical: 8-way MTP is correct but slower than solo MTP.**
   The persistent launcher changes only after the guarded 1.20x aggregate gate.
   A miss leaves the implementation experimental and restores solo MTP.
4. **Minor: requests arriving during a long cohort wait for the next cohort.**
   Mid-run arbitrary-length refill is intentionally deferred. Adding it later
   requires a separate parity and latency gate; it cannot be smuggled into this
   implementation without evidence.
