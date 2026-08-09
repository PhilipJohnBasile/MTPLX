# Concurrency modes

MTPLX keeps model execution on one owner thread. A scheduler decides whether
requests run alone or share a model forward. Requests never share prompt
history, sampler state, or logical KV state.

The public scheduler values are:

| Value | Concurrent decode |
|---|---|
| `serial` | One request at a time on the normal model route; this is the default |
| `cooperative` | Cooperative request scheduling with concurrent work kept off the fixed MTP runner |
| `ar_batch` | Batched target-only autoregressive decode |
| `mtp_batch` | Fixed Qwen 35B depth-one MTP cohorts described below |
| `mtp_cohort_experimental` | Experimental cohort scheduler; it is not the fixed Qwen 35B runner |

Select a scheduler when the server starts:

```bash
mtplx serve --scheduler-mode serial
mtplx serve --scheduler-mode ar_batch
```

Scheduler and kernel routes are construction-time settings. They cannot be
changed per request or through live settings.

## Qwen 35B fixed B8 MTP runner

`--scheduler-mode mtp_batch` installs a runner made specifically for the
Qwen3.6-35B-A3B MTPLX model contract. It is native MTP, not batched AR.

One request uses the unchanged solo B1 MTP route. Two through eight compatible
requests are placed in one physical B8 cohort. Empty rows are padded and remain
inactive. The eight rows execute each model cycle in lockstep because they
share one fixed-shape forward, but every row owns its own:

- prompt and generated tokens;
- target and MTP KV offsets and contents;
- recurrent GDN state;
- target and draft samplers;
- seeded random-number stream;
- token budget, stop state, cancellation event, and result stream.

Sharing batched allocations does not mean sharing context. Row-specific masks,
offsets, commits, and rewinds keep one request from reading or changing another
request's history.

### Geometry

The runner uses depth-one MTP, also called K1:

- `B1` means one request row. `B8` means eight physical request rows.
- The MTP head drafts one token per active row with shape `B8 x T1`, flattened
  to `M8` for projections and MoE work.
- The target verifies the current target token plus that draft with shape
  `B8 x T2`, flattened to `M16`.
- Acceptance, correction sampling, and commit decisions are independent for
  every row.

This is why the runner appears synchronized while still providing eight
separate contexts.

### Request capabilities

The B8 route supports streaming and non-streaming text generation, independent
request seeds and sampler settings, greedy device-side token selection,
stochastic sampling, penalties, stopping, and per-row cancellation.

The installed route fails closed. It never changes a concurrent request to AR
or silently falls back to another kernel. The current fixed contract requires:

- Qwen3.6-35B-A3B with the expected MTPLX target and MTP weights;
- native MTP generation with depth 1;
- `max_active_requests=8` and `decode_batch_max=8`;
- a 131,072-token context window;
- the stock verify-core selection used by the installed B8 graph;
- `prompt_tokens + max_tokens <= 131072` for every request.

The route does not accept `response_format`, vision splice input, background
requests, or a request-level MTP depth other than 1. Invalid server settings
fail before model construction. Invalid request settings return an OpenAI-style
400 error before cohort admission.

## Numerics profiles

Choose one route with
`--mtp-batch-numerics throughput|balanced|b1-exact`. The performance value is
spelled `throughput`; `performance` is not an accepted value.

| Value | Execution | Numerical contract |
|---|---|---|
| `throughput` | Fixed B8/T2 target and B8/T1 draft; default | Fastest aggregate route; bounded BF16 geometry drift from B1 is allowed |
| `balanced` | B8 scheduling with selected layer-zero projections using B1 arithmetic | Closer to B1, but later B8 reductions mean it is not bit- or token-exact with B1 |
| `b1-exact` | Every request uses the unchanged B1 implementation serially | B1 token and cache behavior; it does not claim B8 execution or aggregate B8 throughput |

The device-resident greedy optimization is token-exact with the earlier B8
route. It does not make B8 bit-exact with B1. On the measured legacy greedy
workload, `throughput` reached 349.064 aggregate TPS and `balanced` reached
259.650 TPS. The balanced result missed its 300 TPS promotion floor, so
`throughput` remains the default. See the
[numerics design and receipts](specs/2026-08-09-qwen35b-mtp-batch-numerics-profiles-design.md)
for the full benchmark and EvalPlus results.

## Start the Qwen 35B runner

Use the full construction contract. Change only the final numerics value when
switching among the three routes:

```bash
mtplx serve \
  --model Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed \
  --scheduler-mode mtp_batch \
  --batching-preset throughput \
  --generation-mode mtp \
  --load-mtp \
  --depth 1 \
  --max-active-requests 8 \
  --decode-batch-max 8 \
  --context-window 131072 \
  --verify-core stock \
  --mtp-batch-numerics throughput
```

To use another numerics route, replace the last value with `balanced` or
`b1-exact` and restart the server.

### Persistent configuration

The scheduler, width, context, and numerics choice can be saved in the user
config:

```bash
mtplx config set scheduler_mode mtp_batch
mtplx config set batching_preset throughput
mtplx config set max_active_requests 8
mtplx config set decode_batch_max 8
mtplx config set context_window 131072
mtplx config set mtp_batch_numerics throughput
mtplx config show --json
```

Depth, generation mode, MTP loading, and verify core remain explicit launch
arguments:

```bash
mtplx serve \
  --model Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed \
  --generation-mode mtp \
  --load-mtp \
  --depth 1 \
  --verify-core stock
```

An explicit CLI value overrides the saved value for that launch. To change a
running service, stop it, change the config or launch arguments, and start it
again. If launchd or another service manager owns the process, update that
service's arguments and restart the same service; do not start a second model
process beside it.

## Confirm the route is active

The health payload reports the installed profile and route. After sending at
least two requests concurrently, it also provides behavioral evidence that a
real multi-row cohort ran:

```bash
curl -s http://127.0.0.1:8000/health | jq '.scheduler | {
  mode,
  active_lane,
  numerics: .mtp_batch_numerics,
  route: .mtp_batch_route_id,
  last_real_width: .telemetry.last_real_width,
  batch_histogram: .telemetry.batch_histogram
}'
```

For `throughput` or `balanced`, `last_real_width` must be between 2 and 8 to
prove the B8 runner handled concurrent requests. A configured mode or route ID
alone does not prove that a cohort ran. A single request correctly reports the
solo MTP lane. `b1-exact` reports `mtp_batch_b1_exact_serial` and never claims a
fixed-width B8 execution.
