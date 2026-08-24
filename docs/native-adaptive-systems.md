# Native adaptive systems

MTPLX implements selected ideas independently for Apple Silicon. It does not
load, embed, or require FreeToken or Future AGI. The implementations below use
MTPLX runtime contracts, MLX semantics, SessionBank ownership, and existing
OpenAI-compatible serving paths.

## Expert residency and warm-set control

`mtplx.expert_residency` converts bounded expert-locality observations into a
byte-budgeted, hysteretic warm-set plan. It never changes router outputs.

- An explicit backend capability may implement real residency, prefetch, and
  eviction.
- The generic MLX backend is reported as `materialize_only`: it evaluates lazy
  expert arrays to warm them but does not claim that macOS unified-memory pages
  were individually pinned or unloaded.
- Plans and receipts expose target bytes, prefetches, evictions, failures, and
  backend mode.

Enable planning with:

```bash
export MTPLX_EXPERT_RESIDENCY=1
export MTPLX_EXPERT_RESIDENCY_BYTES=$((8 * 1024 * 1024 * 1024))
```

## Shared unified-memory budgets

`mtplx.unified_memory` coordinates one managed budget across SessionBank,
expert warm-set state, and protected KV headroom. Changes require a non-blocking
model-lock acquisition and an independently verified safe point. If a later
consumer fails, previously changed budgets are rolled back.

```bash
export MTPLX_UNIFIED_MEMORY=1
export MTPLX_UNIFIED_MEMORY_TARGET=0.88
export MTPLX_UNIFIED_MEMORY_RESERVE_BYTES=$((4 * 1024 * 1024 * 1024))
```

KV headroom is a protected planning partition. A backend must expose an
explicit mutation capability before MTPLX changes a live KV allocation.

## Privacy-first OTLP/HTTP

`mtplx.otlp_export` is a dependency-free OTLP/HTTP JSON exporter with a bounded
queue, short timeouts, background batching, and fail-open delivery. Prompt,
message, response, reasoning, tool argument, error-message, and credential
fields are redacted or reduced to byte counts and SHA-256 digests by default.
Numeric metrics such as `prompt_tokens`, `completion_tokens`, latency, cache
bytes, and MTP acceptance remain visible.

```bash
export MTPLX_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces
export OTEL_SERVICE_NAME=mtplx-local
```

Content export requires the separate explicit opt-in
`MTPLX_OTLP_ALLOW_CONTENT=1`. An exporter outage never fails a generation
request.

## Native policy hooks

`mtplx.policy_hooks.PolicyBus` supports trusted in-process hooks for:

- request
- stream event
- response
- error

Hooks have deterministic priority, isolated inputs, bounded values and
annotations, per-hook timeouts, and explicit fail-open or fail-closed behavior.
No module is imported from an environment variable and no external policy
service is contacted. The default bus has no hooks and is inert.

## Capture-to-replay orchestration

`mtplx.replay_orchestrator` selects bounded capture sets, applies deterministic
filters and deduplication, records source digests, rejects stale plans, runs the
existing provider-neutral counterfactual replay core, and writes atomic
promotion receipts.

Capture content remains disabled by default. A count/hash-only capture is not
silently treated as replayable input; execution requires either an explicitly
content-enabled capture or a trusted request resolver. Promotion receipts can
recommend promotion, but MTPLX never applies promotion automatically.

## Runtime and dashboard truth

`GET /v1/mtplx/systems` reports separate phase-two states for:

- `expert_residency`
- `unified_memory`
- `otlp_export`
- `policy_hooks`
- `replay_orchestration`

The Systems dashboard displays these states without equating source presence or
configuration with successful work. Backend mode, samples, exports, hook
executions, and receipts provide the evidence for `active` or `observed` state.
