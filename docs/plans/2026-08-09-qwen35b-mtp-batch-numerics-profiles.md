# Qwen 35B MTP-Batch Numerics Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add construction-selected throughput, balanced, and b1-exact Qwen 35B B8 MTP numerics profiles, promote balanced only when it improves both coding suites while retaining at least 300 greedy aggregate TPS, and publish all receipts in existing PR #245.

**Architecture:** A no-MLX enum is resolved by CLI/config once and passed into the Qwen 35B lane installer. The installer selects an immutable profile specification containing prebound target, draft, capture, commit, attention, and cache callables; generation never branches on the profile. A construction-only attribution command identifies the first real B1/B8 divergence before any balanced kernel is selected. Profile and route fingerprints enter session-cache identity, health, and completion receipts.

**Tech Stack:** Python 3.12, MLX/Metal, pytest, Ruff, EvalPlus 0.3.1, gh, guarded macOS M5 Max GPU benchmarking.

**Assumptions:**

- Assumes the first material B1/B8 divergence can be attributed to a finite set of target/draft operators — this plan stops after attribution and is revised before kernel work if divergence is caused by an unowned MLX graph transformation that cannot be construction-bound.
- Assumes balanced can clear 300 greedy aggregate TPS after surgical B1-order operations — it will not become default if the measured median is below 300.
- Assumes b1-exact may be slower than 300 TPS — it remains explicit and will not be relabeled or silently downgraded if exact parity fails.
- Assumes all live model work can acquire /tmp/mtplx-gpu-exclusive.lock with only Qwen loaded — no service or model change occurs when another owner holds the lock.

---

## File structure

- Create mtplx/mtp_batch_numerics.py: no-MLX profile enum and normalization.
- Modify mtplx/config.py, mtplx/cli.py, and mtplx/server/openai.py: startup flag/config, validation, health, and session identity.
- Modify mtplx/a3b_mtp_batch.py: immutable profile specifications and profile-specific installation contracts.
- Create mtplx/qwen35b_mtp_batch_exact.py: B1-order multi-row callables selected after attribution.
- Create scripts/qwen35b_mtp_batch_numerics_attribution.py: guarded construction-only divergence receipt.
- Create scripts/qwen35b_mtp_batch_numerics_guarded.py: isolated throughput/correctness bracket.
- Modify tests/test_config.py, tests/test_server_openai.py, tests/test_a3b_mtp_batch.py, tests/test_mtp_batch_serving.py, and tests/test_session_bank.py.
- Create tests/test_mtp_batch_numerics.py and tests/test_qwen35b_mtp_batch_numerics_attribution.py.
- Update the approved spec and existing PR #245 plan with measured receipts.

### Task 1: Add the no-MLX public profile flag

**Files:**
- Create: mtplx/mtp_batch_numerics.py
- Modify: mtplx/config.py
- Modify: mtplx/cli.py
- Modify: mtplx/server/openai.py
- Create: tests/test_mtp_batch_numerics.py
- Modify: tests/test_config.py
- Modify: tests/test_server_openai.py

**Security flag:** none

**Does NOT cover:** Profile selection does not activate a new kernel, change singleton behavior, or allow request-level profile overrides.

- [ ] **Step 1: Write failing enum, config, CLI, and validation tests**

~~~python
def test_numerics_names_are_closed_and_normalized():
    from mtplx.mtp_batch_numerics import (
        MTPBatchNumerics,
        normalize_mtp_batch_numerics,
    )

    assert normalize_mtp_batch_numerics(None) is MTPBatchNumerics.THROUGHPUT
    assert normalize_mtp_batch_numerics("balanced") is MTPBatchNumerics.BALANCED
    assert normalize_mtp_batch_numerics("b1-exact") is MTPBatchNumerics.B1_EXACT
    with pytest.raises(ValueError, match="throughput, balanced, b1-exact"):
        normalize_mtp_batch_numerics("auto")


def test_direct_server_parser_exposes_mtp_batch_numerics():
    args = parse_args(
        ["--mtp-batch-numerics", "b1-exact", "--warmup-tokens", "0"]
    )
    assert args.mtp_batch_numerics == "b1-exact"


def test_non_default_numerics_requires_mtp_batch():
    args = parse_args(
        ["--mtp-batch-numerics", "balanced", "--warmup-tokens", "0"]
    )
    with pytest.raises(
        RuntimeError, match="balanced requires scheduler_mode=mtp_batch"
    ):
        openai._validate_mtp_batch_settings(args)
~~~

Add a config fixture containing mtp_batch_numerics = "balanced". Assert loaded UserConfig and runtime arguments contain balanced unless CLI explicitly supplies --mtp-batch-numerics throughput.

- [ ] **Step 2: Run tests and capture the red state**

Run:

~~~bash
.venv/bin/python -m pytest -q \
  tests/test_mtp_batch_numerics.py \
  tests/test_config.py \
  tests/test_server_openai.py \
  -k 'mtp_batch_numerics or non_default_numerics'
~~~

Expected: collection/import or assertion failures because the enum, flag, and config key do not exist.

- [ ] **Step 3: Implement the no-MLX enum and startup parsing**

~~~python
from enum import Enum


class MTPBatchNumerics(str, Enum):
    THROUGHPUT = "throughput"
    BALANCED = "balanced"
    B1_EXACT = "b1-exact"


MTP_BATCH_NUMERICS_CHOICES = tuple(item.value for item in MTPBatchNumerics)


def normalize_mtp_batch_numerics(value: object | None) -> MTPBatchNumerics:
    raw = str(value or MTPBatchNumerics.THROUGHPUT.value).strip().lower()
    try:
        return MTPBatchNumerics(raw)
    except ValueError as exc:
        choices = ", ".join(MTP_BATCH_NUMERICS_CHOICES)
        raise ValueError(
            f"unknown mtp_batch numerics profile {raw!r}; "
            f"expected one of: {choices}"
        ) from exc
~~~

Add mtp_batch_numerics to CONFIG_VALUE_KEYS, UserConfig, config parsing, _RUNTIME_DEFAULTS, both argument parsers, and help text. Default to throughput. Extend _validate_mtp_batch_settings so non-default profiles fail outside mtp_batch and every value is normalized before model construction.

- [ ] **Step 4: Run focused tests and no-MLX smoke**

~~~bash
.venv/bin/python -m pytest -q \
  tests/test_mtp_batch_numerics.py tests/test_config.py tests/test_server_openai.py \
  -k 'mtp_batch_numerics or non_default_numerics'
.venv/bin/python -c \
  'from mtplx.mtp_batch_numerics import MTPBatchNumerics; print(MTPBatchNumerics.BALANCED.value)'
~~~

Expected: focused tests pass and smoke prints balanced without loading MLX.

- [ ] **Step 5: Commit**

~~~bash
git add mtplx/mtp_batch_numerics.py mtplx/config.py mtplx/cli.py \
  mtplx/server/openai.py tests/test_mtp_batch_numerics.py \
  tests/test_config.py tests/test_server_openai.py
git commit -m "Add MTP batch numerics profile flag"
~~~

### Task 2: Install immutable profile specifications and isolate reusable state

**Files:**
- Modify: mtplx/a3b_mtp_batch.py
- Modify: mtplx/server/openai.py
- Modify: tests/test_a3b_mtp_batch.py
- Modify: tests/test_server_openai.py
- Modify: tests/test_session_bank.py

**Security flag:** none

**Does NOT cover:** Balanced and exact arithmetic are unavailable until their profile-specific factories and receipts pass. Missing factories fail closed.

- [ ] **Step 1: Write failing route, health, and fingerprint tests**

~~~python
@pytest.mark.parametrize(
    ("profile", "suffix"),
    [
        ("throughput", "m16_throughput"),
        ("balanced", "balanced"),
        ("b1-exact", "b1_exact"),
    ],
)
def test_installer_route_identity_includes_numerics_profile(
    tmp_path, profile, suffix
):
    lane = install_a3b_mtp_batch_lane(
        _runtime(tmp_path),
        numerics=profile,
        selfcheck=_passing_selfcheck,
        profile_factories=_fake_profile_factories(),
    )
    assert lane.numerics_profile == profile
    assert lane.route_id.endswith(suffix)
    assert profile in lane.config_fingerprint


def test_policy_fingerprint_changes_with_mtp_batch_numerics():
    throughput = _fingerprint_state(mtp_batch_numerics="throughput")
    balanced = _fingerprint_state(mtp_batch_numerics="balanced")
    assert throughput != balanced


def test_health_reports_effective_numerics_and_route():
    payload = openai._mtplx_scheduler_state(_state_with_mtp_batch_stats())
    assert payload["mtp_batch_numerics"] == "balanced"
    assert payload["mtp_batch_route_id"].endswith("balanced")
~~~

- [ ] **Step 2: Run tests and verify profile identity is absent**

~~~bash
.venv/bin/python -m pytest -q \
  tests/test_a3b_mtp_batch.py tests/test_server_openai.py tests/test_session_bank.py \
  -k 'numerics or profile_identity or policy_fingerprint'
~~~

Expected: failures because the lane has no profile field and session identity omits it.

- [ ] **Step 3: Add immutable construction specifications**

~~~python
@dataclass(frozen=True)
class A3BMTPBatchProfileSpec:
    numerics: MTPBatchNumerics
    route_id: str
    target_forward: Callable[..., Any]
    capture_forward: Callable[..., Any]
    draft_forward: Callable[..., Any]
    update_mtp_cache: Callable[..., Any]
    commit_rows: Callable[..., Any]
    selfcheck_contract: Callable[[Mapping[str, Any]], bool]
~~~

Add numerics_profile to InstalledA3BMTPBatchLane. Change install_a3b_mtp_batch_lane to construct exactly one spec, run that spec's self-check, and bind its callables directly. Throughput reuses current callables unchanged. Balanced/exact require explicit factories; absence raises A3BMTPBatchInstallError.

- [ ] **Step 4: Add identity to session policy and health**

When mtp_batch is installed, append to _policy_fingerprint:

~~~python
parts.extend(
    (
        f"mtp_batch_numerics={state.mtp_batch_lane.numerics_profile}",
        f"mtp_batch_route={state.mtp_batch_lane.route_id}",
    )
)
~~~

Pass the normalized profile from ServerState to installation. Report requested/effective profile, route ID, and config fingerprint at request/cohort boundaries only.

- [ ] **Step 5: Run focused tests**

~~~bash
.venv/bin/python -m pytest -q \
  tests/test_a3b_mtp_batch.py tests/test_server_openai.py tests/test_session_bank.py \
  -k 'numerics or profile_identity or policy_fingerprint or mtp_batch_route'
~~~

Expected: all pass and throughput behavior remains unchanged.

- [ ] **Step 6: Commit**

~~~bash
git add mtplx/a3b_mtp_batch.py mtplx/server/openai.py \
  tests/test_a3b_mtp_batch.py tests/test_server_openai.py tests/test_session_bank.py
git commit -m "Bind immutable MTP batch numerics routes"
~~~

### Task 3: Build and run construction-only divergence attribution

**Files:**
- Create: scripts/qwen35b_mtp_batch_numerics_attribution.py
- Modify: mtplx/a3b_mtp_batch.py
- Modify: tests/test_a3b_mtp_batch.py
- Create: tests/test_qwen35b_mtp_batch_numerics_attribution.py

**Security flag:** none

**Does NOT cover:** Attribution cannot run from a request, health endpoint, production decode loop, or timed throughput cell.

- [ ] **Step 1: Write failing receipt and hot-path exclusion tests**

~~~python
def test_attribution_names_first_divergence_and_real_shapes():
    report = build_report(_fake_lane_report())
    assert report["geometry"] == {"target": [8, 2], "draft": [8, 1]}
    first = report["first_material_divergence"]
    assert first["operator"] == "target.layers.0.q_proj"
    assert first["b1_shape"] == [1, 2, 2048]
    assert first["b8_shape"] == [8, 2, 2048]


def test_driver_does_not_call_attribution():
    source = inspect.getsource(generate_a3b_mtp_batch)
    assert "attribution" not in source
    assert "first_material_divergence" not in source
~~~

- [ ] **Step 2: Run and verify the absent receipt fails**

~~~bash
.venv/bin/python -m pytest -q \
  tests/test_a3b_mtp_batch.py \
  tests/test_qwen35b_mtp_batch_numerics_attribution.py \
  -k 'attribution or first_divergence'
~~~

Expected: import/collection failure for absent helpers.

- [ ] **Step 3: Add ordered construction boundary records**

Each record has this exact schema:

~~~python
{
    "operator": operator_name,
    "layer": layer_index,
    "phase": phase,
    "b1_shape": list(b1.shape),
    "b8_shape": list(b8.shape),
    "bitwise": bool(bitwise),
    "max_abs": float(max_abs),
    "max_ulp": int(max_ulp),
    "argmax_equal": bool(argmax_equal),
}
~~~

Evaluate only in the existing startup self-check or standalone command. Production retains only the frozen final report and no callbacks/counters.

- [ ] **Step 4: Implement guarded command**

The command acquires /tmp/mtplx-gpu-exclusive.lock non-blocking before construction, rejects non-Qwen models, loads only the requested Qwen model, and writes JSON to --output. It exits nonzero without changing a service when lock acquisition fails.

~~~bash
PYTHONPATH=. .venv/bin/python scripts/qwen35b_mtp_batch_numerics_attribution.py \
  --model /Users/davidtai/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed \
  --lock /tmp/mtplx-gpu-exclusive.lock \
  --output /tmp/qwen35b-mtp-b8-numerics-attribution.json
~~~

- [ ] **Step 5: Run CPU tests then guarded real-model attribution**

Expected receipt: exact model identity, target [8,2], draft [8,1], ordered errors, exact row ownership, and named first material divergence.

If the first divergence is not at an explicit construction-owned callable boundary, stop, leave throughput as the only available route, and revise the design with the receipt. Do not guess a kernel.

- [ ] **Step 6: Commit**

~~~bash
git add scripts/qwen35b_mtp_batch_numerics_attribution.py \
  mtplx/a3b_mtp_batch.py tests/test_a3b_mtp_batch.py \
  tests/test_qwen35b_mtp_batch_numerics_attribution.py
git commit -m "Attribute Qwen B8 numerical divergence"
~~~

### Task 4: Implement the smallest balanced operator set

**Files:**
- Create: mtplx/qwen35b_mtp_batch_exact.py
- Modify: mtplx/a3b_mtp_batch.py
- Modify: tests/test_mtp_batch_numerics.py
- Modify: tests/test_a3b_mtp_batch.py
- Create: scripts/qwen35b_mtp_batch_numerics_guarded.py
- Modify: tests/test_mtp_batch_serving.py

**Security flag:** none

**Does NOT cover:** Balanced cannot add an unattributed operator, switch by row/logit margin, or become default from microbenchmarks.

- [ ] **Step 1: Pin the attribution result in a failing exact-callable test**

Copy the exact operator name, shape, dtype, quantization, group size, reduction order, and BF16 cast points from the receipt into the test. Compare one B8 candidate with eight unchanged B1 calls:

~~~python
candidate = exact_callable(b8_input, **real_quantized_weights)
references = mx.concatenate(
    [
        b1_callable(
            b8_input[row : row + 1],
            **real_quantized_weights,
        )
        for row in range(8)
    ]
)
mx.eval(candidate, references)
assert np.array_equal(np.asarray(candidate), np.asarray(references))
~~~

Also change row zero and require rows one through seven to stay bitwise equal.

- [ ] **Step 2: Run and verify failure**

~~~bash
.venv/bin/python -m pytest -q \
  tests/test_mtp_batch_numerics.py tests/test_a3b_mtp_batch.py \
  -k 'balanced and exact_callable'
~~~

Expected: failure because the attributed callable is absent.

- [ ] **Step 3: Implement one B1-order multi-row callable**

Derive from the actual B1 operation named by attribution. Preserve K traversal, accumulator type, multiply/conversion sequence, and BF16 stores. Encode row only in an outer grid dimension. Share dispatch or immutable weight tile only when arithmetic stays unchanged.

~~~python
@dataclass(frozen=True)
class Qwen35BExactOperator:
    name: str
    b1_shape: tuple[int, ...]
    b8_shape: tuple[int, ...]
    call_b8: Callable[..., mx.array]
    receipt_sha256: str
~~~

Validate invariant shape/dtype/quantization at installation, never inside call_b8.

- [ ] **Step 4: Bind only into balanced and rerun construction parity**

Retain throughput callables elsewhere. Require exact parity at replaced boundaries, exact row isolation/offsets/argmax, and existing bounds elsewhere. Fail installation on a missed receipt.

- [ ] **Step 5: Run focused unit/serving tests**

~~~bash
.venv/bin/python -m pytest -q \
  tests/test_mtp_batch_numerics.py tests/test_a3b_mtp_batch.py \
  tests/test_mtp_batch_serving.py tests/test_server_openai.py \
  -k 'balanced or numerics or mtp_batch'
~~~

- [ ] **Step 6: Run guarded balanced bracket**

Run throughput/control/throughput drift brackets plus balanced with fixed prompts/seeds. Record tokens, wall time, TPS, hashes, route, physical width, peak memory, and receipt.

If greedy median is below 300, remove the losing candidate and record rejection. If it clears 300 but a later material divergence remains, repeat Steps 1-6 for only that next boundary. Stop when the next addition violates the floor or decision boundaries meet the balanced contract.

- [ ] **Step 7: Commit**

~~~bash
git add mtplx/qwen35b_mtp_batch_exact.py mtplx/a3b_mtp_batch.py \
  tests/test_mtp_batch_numerics.py tests/test_a3b_mtp_batch.py \
  scripts/qwen35b_mtp_batch_numerics_guarded.py tests/test_mtp_batch_serving.py
git commit -m "Add balanced Qwen B8 numerics route"
~~~

### Task 5: Implement fail-closed B1-exact

**Files:**
- Modify: mtplx/qwen35b_mtp_batch_exact.py
- Modify: mtplx/a3b_mtp_batch.py
- Modify: tests/test_mtp_batch_numerics.py
- Modify: tests/test_a3b_mtp_batch.py
- Modify: tests/test_mtp_batch_serving.py

**Security flag:** none

**Does NOT cover:** B1-exact has no 300-TPS guarantee and cannot install on bounded-only parity or automatically replace another profile.

- [ ] **Step 1: Write failing full-state exact test**

For heterogeneous rows and keeps [0,1,2,0,2,1,0,2], compare exact B8 with eight unchanged B1 references. Require bitwise target/draft outputs, hidden, attention K/V, recurrent state, logits, offsets, tokens, accept/reject decisions, and next RNG.

~~~python
assert report["b1_exact_bitwise"] is True
assert report["b1_exact_failed_boundaries"] == []
assert report["row_isolation_parity"] is True
assert report["mixed_commit_parity"] is True
~~~

- [ ] **Step 2: Run and verify listed failures**

~~~bash
.venv/bin/python -m pytest -q \
  tests/test_mtp_batch_numerics.py tests/test_a3b_mtp_batch.py \
  -k 'b1_exact'
~~~

Expected: failure listing remaining non-exact boundaries.

- [ ] **Step 3: Replace each listed boundary one at a time**

For each boundary: pin real arithmetic in a failing test, implement one construction-bound callable, then rerun full-state parity. If a multi-row exact kernel is unavailable, b1-exact may bind eight explicit unchanged B1 calls through a row-owned adapter. The adapter must be proven bitwise and is forbidden in balanced/throughput.

- [ ] **Step 4: Require exact install and explicit failure**

Accept only b1_exact_bitwise=True with no failed boundaries. Any mismatch raises A3BMTPBatchInstallError naming boundaries. Add a server test proving no request runner executes after failure.

- [ ] **Step 5: Run exact unit/serving tests**

~~~bash
.venv/bin/python -m pytest -q \
  tests/test_mtp_batch_numerics.py tests/test_a3b_mtp_batch.py \
  tests/test_mtp_batch_serving.py tests/test_server_openai.py \
  -k 'b1_exact or numerics or mtp_batch'
~~~

- [ ] **Step 6: Commit**

~~~bash
git add mtplx/qwen35b_mtp_batch_exact.py mtplx/a3b_mtp_batch.py \
  tests/test_mtp_batch_numerics.py tests/test_a3b_mtp_batch.py \
  tests/test_mtp_batch_serving.py tests/test_server_openai.py
git commit -m "Add fail-closed B1-exact B8 route"
~~~

### Task 6: Run promotion gates and update PR #245

**Files:**
- Modify: docs/specs/2026-08-09-qwen35b-mtp-batch-numerics-profiles-design.md
- Modify: docs/plans/2026-08-08-qwen35b-eight-way-mtp.md
- Modify: persistent Qwen launcher only if balanced passes every gate

**Security flag:** none

**Does NOT cover:** No new PR, DeepSeek load, AR default, promotion from unit/microbenchmarks, or benchmark under another lock owner.

- [ ] **Step 1: Run changed-area and full CPU verification**

~~~bash
.venv/bin/python -m pytest -q \
  tests/test_mtp_batch_numerics.py tests/test_a3b_mtp_batch.py \
  tests/test_a3b_mtp_batch_driver.py tests/test_mtp_batch_serving.py \
  tests/test_server_openai.py tests/test_config.py tests/test_session_bank.py
.venv/bin/python -m ruff check \
  mtplx/mtp_batch_numerics.py mtplx/qwen35b_mtp_batch_exact.py \
  mtplx/a3b_mtp_batch.py mtplx/config.py mtplx/cli.py \
  mtplx/server/openai.py \
  scripts/qwen35b_mtp_batch_numerics_attribution.py \
  scripts/qwen35b_mtp_batch_numerics_guarded.py
git diff --check
~~~

Then run the repository suite with the same two documented cached vllm-metal ABI deselections used by PR #245. Expected: zero new failures.

- [ ] **Step 2: Run isolated three-profile performance brackets under the lock**

Verify DeepSeek is absent. Run three paired rounds for throughput and balanced under greedy/default sampling, plus measured b1-exact rounds. Require real route IDs, markers, no foreign text, no negative counters, no cleanup errors, and isolated traffic.

Balanced floors:

~~~text
greedy median >= 300.000 aggregate output tok/s
default-sampler median >= 153.425 aggregate output tok/s
~~~

Report b1-exact without a floor.

- [ ] **Step 3: Run full isolated EvalPlus for all profiles**

Use EvalPlus 0.3.1, HumanEval+ hash fe585eb4df8c88d844eeb463ea4d0302, MBPP+ hash ee43ecabebf20deef4bb776a405ac5b1, one completion, temperature 0, top-p 0.95, and 768 maximum tokens.

Balanced minimums:

~~~text
HumanEval base >= 151/164
HumanEval+ >= 145/164
MBPP base >= 335/378
MBPP+ >= 286/378
~~~

Both plus suites must improve over throughput. B1-exact must reproduce fixed B1 deterministic hashes or remain unavailable.

- [ ] **Step 4: Apply promotion decision**

If balanced passes every gate, set the persistent Qwen launcher to pass --mtp-batch-numerics balanced. Restart only Qwen while holding the lock, rerun marker/cancellation health gates, then release. If any gate misses, leave throughput default and publish the miss.

- [ ] **Step 5: Update documentation and commit receipts**

Replace design status with measured outcome. Add a table of route IDs, parity, HumanEval+, MBPP+, greedy/default TPS, peak memory, rejected candidates, and exact commands.

~~~bash
git add docs/specs/2026-08-09-qwen35b-mtp-batch-numerics-profiles-design.md \
  docs/plans/2026-08-08-qwen35b-eight-way-mtp.md
git commit -m "Document MTP batch numerics profile receipts"
~~~

- [ ] **Step 6: Push existing branch and update only PR #245**

~~~bash
git status --short
git push mtplx1 fix/ar-batch-filter-fail-closed
gh pr view 245 --repo youssofal/MTPLX \
  --json url,headRefOid,statusCheckRollup
~~~

Update the existing PR body with flags, exact benchmark tables, quality results, rejected candidates, default decision, and verification commands. Do not create another PR.

## Plan self-review

- Spec coverage: flag, three immutable profiles, attribution, cache identity, health, failure behavior, quality/performance gates, rollout, and same-PR publication each have a task.
- Completeness scan: every step contains concrete implementation guidance and an explicit verification command.
- Type consistency: MTPBatchNumerics, A3BMTPBatchProfileSpec, InstalledA3BMTPBatchLane.numerics_profile, and Qwen35BExactOperator retain the same names.
- Scope reduction: balanced retains both floors; b1-exact is contractual and fail-closed; no profile is silently omitted.
