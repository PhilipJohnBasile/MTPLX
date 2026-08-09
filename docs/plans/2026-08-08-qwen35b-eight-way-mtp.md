# Qwen 35B eight-way MTP implementation plan

> **Execution mode:** Inline. Keep every commit on `fix/ar-batch-filter-fail-closed`
> and update existing PR #245. Do not create another branch, worktree, or PR.

**Goal:** Serve two through eight independent Qwen3.6-35B-A3B requests in one
fixed eight-row, depth-one MTP cohort while keeping single requests on the
existing solo MTP path. AR remains available only when explicitly selected.

**Architecture:** Extend the existing A3B fixed-shape speculative decoder rather
than the dense Qwen 27B cohort. Install a typed width-eight lane once during
server construction after validating the real Qwen model contract and running
an exact self-check. A dedicated server service seals requests into cohorts at
batch boundaries, owns per-request queues/futures/cancellation, and invokes the
prebound MTP driver on the existing model-owner thread. The hot decode loop has
no environment reads, eligibility checks, or automatic AR fallback.

**Technology:** Python 3.11+, MLX, the installed `mlx-lm` PR #1642 cache fix,
NumPy per-request RNGs, pytest, Ruff, FastAPI/OpenAI compatibility layer.

## Fixed constraints

- Scheduler mode is exactly `mtp_batch`.
- Construction requires `generation_mode=mtp`, loaded MTP weights, depth 1,
  `max_active_requests=8`, `decode_batch_max=8`, and fixed cohort capacity 8.
- The target verify input is `[B=8,T=2]`; its flattened projection/MoE row count
  is `M=16`.
- One request uses the unchanged solo `generate_mtpk` route. Two through eight
  compatible ready requests use the new lane. Later arrivals wait for the next
  sealed cohort.
- Each request owns its sampler, draft sampler, RNG, logical cache/recurrent
  rows, budget, stop state, callback, cancellation event, and future.
- A failed cohort closes every unfinished request in that cohort and is never
  reused. A cancelled row becomes inert without changing neighboring rows.
- DeepSeek remains disabled. Every live model command acquires
  `/tmp/mtplx-gpu-exclusive.lock` before loading or restarting Qwen.
- The persistent launcher changes only after the correctness, resource, kernel,
  and throughput gates pass. A miss leaves the server on solo MTP, never AR.

## Task 1: Add the fail-closed `mtp_batch` configuration contract

**Files:**

- Modify: `mtplx/batching/state.py`
- Modify: `mtplx/batching/scheduler.py`
- Modify: `mtplx/cli.py`
- Modify: `mtplx/commands/public.py`
- Modify: `mtplx/server/openai.py`
- Test: `tests/test_batching_foundation.py`
- Test: `tests/test_public_cli.py`
- Test: `tests/test_server_openai.py`
- Test: `tests/test_dashboard_endpoints.py`

### Step 1: Write failing configuration tests

Add tests that pin the public spelling and reject invalid construction instead
of silently changing modes:

```python
def test_mtp_batch_config_is_fixed_width_eight():
    config = BatchSchedulerConfig.from_values(
        mode="mtp_batch",
        preset="throughput",
        max_active_requests=8,
        decode_batch_max=8,
    )
    assert config.mode is SchedulerMode.MTP_BATCH
    assert config.to_dict()["decode_batch_max"] == 8


@pytest.mark.parametrize(
    ("generation_mode", "max_active", "decode_max"),
    [("ar", 8, 8), ("mtp", 4, 8), ("mtp", 8, 4)],
)
def test_mtp_batch_server_rejects_non_contract_settings(
    generation_mode, max_active, decode_max
):
    args = _serve_args(
        scheduler_mode="mtp_batch",
        generation_mode=generation_mode,
        max_active_requests=max_active,
        decode_batch_max=decode_max,
    )
    with pytest.raises(RuntimeError, match="mtp_batch requires"):
        openai._validate_mtp_batch_settings(args)
```

Also assert `SCHEDULER_MODE_CHOICES`, the public command parser, and dashboard
payload accept/report `mtp_batch` without changing the existing default.

### Step 2: Run the focused tests and confirm RED

Run:

```bash
python -m pytest -q \
  tests/test_batching_foundation.py \
  tests/test_public_cli.py \
  tests/test_server_openai.py \
  tests/test_dashboard_endpoints.py -k 'mtp_batch'
```

Expected: failures because `SchedulerMode.MTP_BATCH` and the construction
validator do not exist.

### Step 3: Implement the contract

- Add `SchedulerMode.MTP_BATCH = "mtp_batch"`.
- Add `mtp_batch` to both CLI choice tables and public command validation.
- Add `_validate_mtp_batch_settings(args)` and call it once while constructing
  `ServerState`.
- Require MTP generation, depth 1, and both request limits equal to 8.
- Make `_scheduler_policy_label` return `fixed_mtp_batch_width_8`.
- Remove `mtp_batch` from every AR/cooperative routing set. Do not route it
  through `_use_live_ar_batch` or assign `batch_size_gt_1`.

### Step 4: Run focused tests and commit

Run the command from Step 2, then:

```bash
git add mtplx/batching/state.py mtplx/batching/scheduler.py mtplx/cli.py \
  mtplx/commands/public.py mtplx/server/openai.py \
  tests/test_batching_foundation.py tests/test_public_cli.py \
  tests/test_server_openai.py tests/test_dashboard_endpoints.py
git commit -m "Add fixed eight-way MTP scheduler contract"
```

## Task 2: Implement exact per-row depth-one speculative decisions

**Files:**

- Modify: `mtplx/batched_decode.py`
- Test: `tests/test_batched_decode.py`

### Step 1: Write failing decision and isolation tests

Introduce request-local inputs and pin the exact `generate_mtpk` contract:

```python
def test_sampled_k1_decision_matches_reference_for_accept_and_reject():
    target = np.array([0.55, 0.35, 0.10])
    draft = np.array([0.20, 0.70, 0.10])
    for seed in range(32):
        expected_rng = np.random.default_rng(seed)
        actual_rng = np.random.default_rng(seed)
        expected = verify_one_token(target, draft, 1, expected_rng)
        actual = _verify_mtp_k1_row(target, draft, 1, actual_rng)
        assert actual == expected


def test_eight_rows_keep_independent_rng_streams():
    requests = _sampled_requests(seeds=range(8))
    batched = generate_greedy_batched(
        _FakeRuntime(),
        [request.prompt_ids for request in requests],
        max_new_tokens=24,
        cohort_slots=8,
        request_states=requests,
    )
    solo = [
        generate_greedy_batched(
            _FakeRuntime(),
            [request.prompt_ids],
            max_new_tokens=24,
            cohort_slots=8,
            request_states=[request],
        ).streams[0].tokens
        for request in requests
    ]
    assert [stream.tokens for stream in batched.streams] == solo
```

Add mixed temperature/top-p/top-k/penalty cases. Add a forced-reject case where
row 0 rejects while rows 1-7 accept, then assert rows 1-7 retain the same token
sequences, RNG next values, cache offsets, and recurrent-state fingerprints as
their run-alone references. Add bonus-enabled and bonus-omitted cases.

### Step 2: Run and confirm RED

Run:

```bash
python -m pytest -q tests/test_batched_decode.py \
  -k 'sampled_k1 or independent_rng or bonus_policy or rejecting_row'
```

Expected: failures because request-local sampled state and `_verify_mtp_k1_row`
are absent.

### Step 3: Implement the sampled row state

- Add a frozen public request specification and an internal mutable row state
  holding the sampler configs, `np.random.Generator`, token counter, per-request
  budget, stop IDs, cancellation event, and bonus policy.
- Precompute all invariant sampler route choices before entering the decode
  loop.
- Materialize target and draft distributions per row with the existing sampling
  helpers. Use the row RNG for target sampling, draft sampling, the acceptance
  draw, residual correction, and bonus sampling in the same order as
  `generate_mtpk`.
- Preserve the existing greedy path byte-for-byte when request state is absent.
- Keep the verify forward fixed at `[8,2]`; only host-side row decisions differ.
- Fold rejection into the existing ragged replay for the rejecting rows only.

### Step 4: Verify parity and commit

Run:

```bash
python -m pytest -q tests/test_sampling.py tests/test_batched_decode.py
git add mtplx/batched_decode.py tests/test_batched_decode.py
git commit -m "Add request-local sampled MTP batch decisions"
```

## Task 3: Add construction-time Qwen 35B lane installation

**Files:**

- Create: `mtplx/a3b_mtp_batch.py`
- Test: `tests/test_a3b_mtp_batch.py`

### Step 1: Write failing installer tests

Pin an immutable installed lane and fail startup on every invariant mismatch:

```python
def test_installer_pins_qwen35b_width8_depth1_geometry():
    runtime = _qwen35b_runtime_stub()
    lane = install_a3b_mtp_batch_lane(runtime)
    assert lane.geometry.cohort_slots == 8
    assert lane.geometry.depth == 1
    assert lane.geometry.verify_tokens == 2
    assert lane.geometry.verify_rows == 16
    assert lane.route_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "deepseek_v4"),
        ("num_hidden_layers", 39),
        ("mtp_depth", 2),
        ("quantization.group_size", 128),
    ],
)
def test_installer_rejects_wrong_runtime_contract(field, value):
    runtime = _qwen35b_runtime_stub(**{field: value})
    with pytest.raises(MTPBatchInstallError, match=field.split(".")[-1]):
        install_a3b_mtp_batch_lane(runtime)
```

Add tests that missing optimized target/draft/attention/MoE callables and a
failed numerical self-check abort installation. Assert the returned lane holds
prebound callables so invocation does not read environment variables or inspect
model metadata.

### Step 2: Run and confirm RED

Run:

```bash
python -m pytest -q tests/test_a3b_mtp_batch.py
```

Expected: import failure because `mtplx.a3b_mtp_batch` does not exist.

### Step 3: Implement the installer

- Define frozen `A3BMTPBatchGeometry` and `InstalledA3BMTPBatchLane` types.
- Read the loaded runtime/model config once and validate the exact served Qwen
  identity, 40-layer topology, depth-one MTP head, dtype, quantization, group
  size, expert layout, hidden width, vocabulary width, and tensor shapes.
- Bind the actual promoted target, draft, attention, projection, and MoE
  callables already installed on the runtime.
- Run an exact small real-shape self-check against the unchanged runtime route,
  including `[8,2]` target verification and `M=16` projection/MoE output.
- Return the installed lane only after the self-check passes; otherwise raise
  `MTPBatchInstallError` with the failing invariant.

### Step 4: Verify and commit

Run:

```bash
python -m pytest -q tests/test_a3b_mtp_batch.py tests/test_batched_decode.py
git add mtplx/a3b_mtp_batch.py tests/test_a3b_mtp_batch.py
git commit -m "Install the Qwen 35B width-eight MTP lane"
```

## Task 4: Add cohort streaming, cancellation, and terminal futures

**Files:**

- Modify: `mtplx/batched_decode.py`
- Create: `mtplx/server/mtp_batch.py`
- Create: `tests/test_mtp_batch_serving.py`
- Modify: `tests/test_batched_decode.py`

### Step 1: Write failing service tests

Test the actual service contract rather than configuration alone:

```python
def test_eight_requests_stream_only_their_own_tokens():
    service = MTPBatchGenerationService(_state_with_fake_lane())
    jobs = [_job(index=i, seed=100 + i) for i in range(8)]
    futures = [service.submit(job) for job in jobs]
    service.pump_once()
    results = [future.result(timeout=1) for future in futures]
    assert [result["request_id"] for result in results] == [job.request_id for job in jobs]
    assert all(result["foreign_markers"] == [] for result in results)
    assert service.snapshot()["batch_histogram"]["8"] == 1


def test_two_cancelled_rows_do_not_change_six_survivors():
    service = MTPBatchGenerationService(_state_with_fake_lane())
    jobs = [_job(index=i, max_tokens=32) for i in range(8)]
    futures = [service.submit(job) for job in jobs]
    jobs[1].cancel_event.set()
    jobs[6].cancel_event.set()
    service.pump_once()
    assert isinstance(futures[1].exception(timeout=1), StreamCancelled)
    assert isinstance(futures[6].exception(timeout=1), StreamCancelled)
    assert _survivor_tokens(futures) == _run_alone_survivor_tokens(jobs)
```

Add tests for: a solo job bypasses the cohort service and calls solo MTP; a
cohort seals after the bounded gather window; later requests form the next
cohort; each callback gets commits in order; each future completes exactly
once; driver errors fail all unfinished jobs; a cancelled row is masked inside
the driver; cache cleanup failure fails the cohort closed; and shutdown closes
queued requests.

### Step 2: Run and confirm RED

Run:

```bash
python -m pytest -q tests/test_mtp_batch_serving.py \
  tests/test_batched_decode.py -k 'commit_callback or cancel_event'
```

Expected: module import failure and missing decoder callback/cancellation hooks.

### Step 3: Implement the service and hooks

- Add `on_commit(request_index, tokens)` and per-row cancellation hooks to the
  decoder. Invoke callbacks only after tokens are committed.
- Implement `MTPBatchJob`, request handle/future completion, and
  `MTPBatchGenerationService` in `mtplx/server/mtp_batch.py`.
- Seal at most eight compatible requests per pump. Pad two through seven real
  rows to eight and do not admit mid-run arrivals.
- Schedule the pump through the existing foreground model-owner queue; do not
  create another model worker.
- Cancel queued requests without admission. Mask active cancelled rows and close
  their futures immediately. Never wait for subsequent foreground work before
  releasing a completed stream.
- On a decode/cache/sampler error, fail every unfinished job, discard the cohort
  state, record the error, and allow a later fresh cohort.

### Step 4: Verify and commit

Run:

```bash
python -m pytest -q tests/test_mtp_batch_serving.py tests/test_batched_decode.py
git add mtplx/batched_decode.py mtplx/server/mtp_batch.py \
  tests/test_mtp_batch_serving.py tests/test_batched_decode.py
git commit -m "Serve independent requests through MTP cohorts"
```

## Task 5: Wire `mtp_batch` into the OpenAI server without AR fallback

**Files:**

- Modify: `mtplx/server/openai.py`
- Modify: `tests/test_server_openai.py`
- Modify: `tests/test_dashboard_endpoints.py`

### Step 1: Write failing route and telemetry tests

```python
def test_mtp_batch_concurrency_never_calls_ar_service(monkeypatch):
    state = _server_state(scheduler_mode="mtp_batch", generation_mode="mtp")
    monkeypatch.setattr(
        state.ar_batch_service,
        "submit",
        lambda job: pytest.fail("mtp_batch must not route through AR"),
    )
    futures = [_submit_generation(state, index=i) for i in range(8)]
    assert all(future.result(timeout=1)["stats"]["generation_mode"] == "mtp" for future in futures)


def test_mtp_batch_health_reports_real_width_and_acceptance():
    payload = openai._mtplx_scheduler_state(_state_with_mtp_batch_stats())
    assert payload["active_lane"] == "mtp_batch_width_8"
    assert payload["telemetry"]["batch_histogram"]["8"] > 0
    assert payload["telemetry"]["target_verify_cycles"] > 0
    assert payload["telemetry"]["accepted_draft_tokens"] >= 0
    assert payload["mtp_disabled_reason"] is None
```

Also test an explicit per-request `generation_mode="ar"` continues to use the
existing AR path, while default/concurrent MTP never does. Test incompatible
tool/constraint graph routes fail clearly rather than silently falling back.

### Step 2: Run and confirm RED

Run:

```bash
python -m pytest -q tests/test_server_openai.py tests/test_dashboard_endpoints.py \
  -k 'mtp_batch'
```

Expected: route/telemetry assertions fail because the service is not installed.

### Step 3: Implement server integration

- Construct and attach `mtp_batch_lane` and `mtp_batch_service` only when the
  validated scheduler mode is `mtp_batch`.
- Route default MTP jobs through the service. Route a single sealed request to
  the unchanged solo MTP function without waiting for seven peers.
- Keep explicit request-level AR routing intact; delete no rollback mode.
- Freeze the compatibility key before admission from values that change the
  target/draft graph. Keep sampler values and seeds out of that key.
- Finalize results with MTP stats: depth 1, verify cycles/time, accepted and
  rejected draft counts, request ID, queue wait, fixed capacity, real width,
  route identity, and zero MTP-disabled reason.
- Expose cohort telemetry from service snapshots at admission/cycle boundaries,
  not through model-layer counters.

### Step 4: Verify and commit

Run:

```bash
python -m pytest -q tests/test_server_openai.py tests/test_dashboard_endpoints.py \
  tests/test_mtp_batch_serving.py
git add mtplx/server/openai.py tests/test_server_openai.py \
  tests/test_dashboard_endpoints.py
git commit -m "Route OpenAI concurrency through eight-way MTP"
```

## Task 6: Run the complete local verification gate

### Step 1: Run focused and full tests

```bash
python -m pytest -q \
  tests/test_sampling.py \
  tests/test_batched_decode.py \
  tests/test_a3b_mtp_batch.py \
  tests/test_mtp_batch_serving.py \
  tests/test_batching_foundation.py \
  tests/test_server_openai.py \
  tests/test_public_cli.py \
  tests/test_dashboard_endpoints.py
python -m pytest -q
```

Expected: all tests pass.

### Step 2: Run style and diff checks

```bash
python -m ruff check mtplx tests
git diff --check origin/main...HEAD
git status --short
```

Expected: Ruff and diff check pass; status is clean after commits.

### Step 3: Review the complete PR diff

```bash
git diff --stat origin/main...HEAD
git diff origin/main...HEAD -- \
  mtplx/batched_decode.py mtplx/a3b_mtp_batch.py \
  mtplx/server/mtp_batch.py mtplx/server/openai.py
```

Confirm there is no enabled-path environment read, metadata revalidation,
per-layer proof counter, dense-27B route reuse, or `mtp_batch` to AR fallback.

## Task 7: Run guarded Qwen-only live gates and promote the launcher conditionally

**Local deployment file (not committed to the upstream repository):**

- Modify only after promotion passes:
  `/Users/davidtai/projects/qwen36-server/scripts/start-qwen-a3b-cohort8-mtp.sh`

### Step 1: Verify ownership and acquire the GPU lock

```bash
test "$(launchctl print-disabled gui/$(id -u) | rg 'com\.tea\.deepseek-v4' | tr -d '[:space:]')" = '"com.tea.deepseek-v4"=>true'
python - <<'PY'
import fcntl
from pathlib import Path

path = Path('/tmp/mtplx-gpu-exclusive.lock')
handle = path.open('a+')
fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
print('gpu-lock-acquired')
input()
PY
```

Keep that process alive for every model stop/start and benchmark command. If
lock acquisition fails, stop without changing services.

### Step 2: Record unchanged solo MTP baseline

With only Qwen loaded, run the same eight prompts three times through the
existing serialized solo MTP route. Save raw JSON receipts under `/tmp` with
request IDs, seeds, prompt/completion tokens, wall time, and aggregate TPS.

Expected: all requests finish with MTP depth 1 and no Metal resource error.

### Step 3: Start the candidate without changing persistent defaults

Launch the current Qwen command manually with:

```text
--generation-mode mtp
--scheduler-mode mtp_batch
--max-active-requests 8
--decode-batch-max 8
--batching-preset throughput
--depth 1
```

Do not launch DeepSeek. Confirm `/health` reports the installed Qwen 35B route,
`mtp_batch`, depth 1, capacity 8, and no construction fallback.

### Step 4: Run correctness and resource gates

- Eight concurrent requests with unique markers and fixed seeds: require eight
  request/session IDs, zero foreign markers, eight terminal events, and a real
  width-8 histogram.
- Cancel two rows after their first commits: require two cancellation results,
  six unchanged survivor streams, zero pending/active requests afterward.
- Run eight requests past the previous 13,000-token resource boundary: require
  no negative counters, overflow, Metal resource exhaustion, or foreign marker.
- Capture dispatch census/profiler evidence outside timing and require the
  installed `M=16` target verify plus matching draft/MoE route identity.

Any failure stops promotion and restores the existing solo MTP service.

### Step 5: Run the performance gate

Repeat candidate measurement three times using the exact baseline prompts,
seeds, token budgets, and stop settings. Compute aggregate completed tokens per
wall second and per-request latency.

Promotion requires candidate aggregate throughput at least 1.20 times the
median unchanged serialized solo MTP baseline, with all correctness gates
passing. A lower result is recorded honestly and the persistent service stays
on solo MTP.

### Step 6: Promote and retest only if every gate passes

Use `apply_patch` to change the local runner defaults to:

```bash
SCHED="${MTPLX_SCHED:-mtp_batch}"
GENMODE="${MTPLX_GENMODE:-mtp}"
```

Restart Qwen while still holding the GPU lock. Repeat the short eight-request
isolation and two-cancel tests against the persistent service. Release the lock
only after health and cleanup are confirmed.

### Step 7: Push the existing branch and update PR #245

```bash
git status --short
git push mtplx1 fix/ar-batch-filter-fail-closed
gh pr view 245 --repo youssofal/MTPLX --json url,headRefName,checks
```

Update the existing PR body with plain-English correctness and benchmark
receipts. Do not create another PR. Report whether the local persistent launcher
was promoted or left on solo MTP and why.
