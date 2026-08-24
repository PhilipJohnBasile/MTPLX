from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.deterministic_replay import Evaluation, RegressionPolicy
from mtplx.expert_residency import ExpertRef
from mtplx.native_adaptive import (
    _expert_config_from_env,
    _unified_config_from_env,
    augment_systems_snapshot,
    ensure_native_adaptive_state,
    native_adaptive_tick,
)
from mtplx.otlp_export import (
    OTLPExporterConfig,
    reset_default_exporter_for_tests,
)
from mtplx.policy_hooks import (
    HookPhase,
    HookResult,
    reset_default_policy_bus_for_tests,
)

_ENV_KEYS = (
    "MTPLX_EXPERT_RESIDENCY",
    "MTPLX_EXPERT_RESIDENCY_BYTES",
    "MTPLX_EXPERT_RESIDENCY_MIN_OBSERVATIONS",
    "MTPLX_EXPERT_RESIDENCY_HALF_LIFE_S",
    "MTPLX_EXPERT_RESIDENCY_MAX_TRACKED",
    "MTPLX_EXPERT_RESIDENCY_PREFETCH_PER_TICK",
    "MTPLX_EXPERT_RESIDENCY_EVICT_PER_TICK",
    "MTPLX_UNIFIED_MEMORY",
    "MTPLX_UNIFIED_MEMORY_RESERVE_BYTES",
    "MTPLX_UNIFIED_MEMORY_TARGET",
    "MTPLX_UNIFIED_MEMORY_WARNING",
    "MTPLX_UNIFIED_MEMORY_CRITICAL",
    "MTPLX_UNIFIED_MEMORY_MIN_SESSION_BYTES",
    "MTPLX_UNIFIED_MEMORY_MIN_EXPERT_BYTES",
    "MTPLX_UNIFIED_MEMORY_MIN_KV_BYTES",
    "MTPLX_OTLP_ENDPOINT",
    "MTPLX_OTLP_DISABLED",
    "MTPLX_OTLP_ALLOW_CONTENT",
    "MTPLX_OTLP_HEADERS",
    "MTPLX_OTLP_TIMEOUT_S",
    "MTPLX_OTLP_BATCH_SIZE",
    "MTPLX_OTLP_FLUSH_INTERVAL_S",
    "MTPLX_OTLP_QUEUE_SIZE",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_SERVICE_NAME",
    "MTPLX_VERSION",
    "MTPLX_REQUEST_CAPTURE_DIR",
    "MTPLX_REPLAY_MAX_CASES",
)


@pytest.fixture(autouse=True)
def _isolated_runtime_globals(monkeypatch):
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    reset_default_exporter_for_tests()
    reset_default_policy_bus_for_tests()
    yield
    reset_default_exporter_for_tests()
    reset_default_policy_bus_for_tests()


class TrackingLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.acquire_calls = 0
        self.release_calls = 0

    def acquire(self, blocking: bool = True) -> bool:
        self.acquire_calls += 1
        return self._lock.acquire(blocking=blocking)

    def release(self) -> None:
        self.release_calls += 1
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


class FakeBank:
    def __init__(self, lock: TrackingLock) -> None:
        self.lock = lock
        self.max_bytes = 8 * 1024**3
        self.per_session_max_bytes = 4 * 1024**3
        self.total_nbytes = 3 * 1024**3
        self.calls: list[tuple[int, int, str]] = []

    def rebalance_limits(
        self,
        *,
        max_bytes: int,
        per_session_max_bytes: int,
        reason: str,
    ) -> None:
        assert self.lock.locked()
        self.calls.append((max_bytes, per_session_max_bytes, reason))
        self.max_bytes = int(max_bytes)
        self.per_session_max_bytes = int(per_session_max_bytes)


class FakeScheduler:
    def __init__(self, *, busy: bool = False, malformed: bool = False) -> None:
        self.busy = busy
        self.malformed = malformed

    def stats(self):
        if self.malformed:
            return None
        return {
            "foreground_pending": 1 if self.busy else 0,
            "idle_pending": 0,
            "persistence_pending": 0,
            "active_kind": "generation" if self.busy else None,
        }


class FakeLocality:
    @staticmethod
    def snapshot() -> dict[str, object]:
        return {"expert_counts": {"0:0": 10, "0:1": 5}}


class FakeBackend:
    mode = "test"

    def __init__(self) -> None:
        self.resident: set[ExpertRef] = set()
        self.prefetch_calls: list[tuple[ExpertRef, ...]] = []
        self.evict_calls: list[tuple[ExpertRef, ...]] = []

    def resident_experts(self):
        return tuple(self.resident)

    def expert_nbytes(self, _expert: ExpertRef) -> int:
        return 1024

    def prefetch_experts(self, experts):
        values = tuple(ExpertRef.coerce(item) for item in experts)
        self.prefetch_calls.append(values)
        self.resident.update(values)
        return values

    def evict_experts(self, experts):
        values = tuple(ExpertRef.coerce(item) for item in experts)
        self.evict_calls.append(values)
        self.resident.difference_update(values)
        return values


class FakeKVHeadroom:
    def __init__(self, lock: TrackingLock) -> None:
        self.lock = lock
        self.kv_headroom_bytes = 2 * 1024**3
        self.calls: list[tuple[int, str]] = []

    def set_kv_headroom_bytes(self, value: int, *, reason: str) -> int:
        assert self.lock.locked()
        self.kv_headroom_bytes = int(value)
        self.calls.append((int(value), reason))
        return self.kv_headroom_bytes


def _enable_expert(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_EXPERT_RESIDENCY", "1")
    monkeypatch.setenv("MTPLX_EXPERT_RESIDENCY_BYTES", "4096")
    monkeypatch.setenv("MTPLX_EXPERT_RESIDENCY_MIN_OBSERVATIONS", "1")


def _enable_unified(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY", "1")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_RESERVE_BYTES", "0")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_MIN_SESSION_BYTES", "1024")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_MIN_EXPERT_BYTES", "1024")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_MIN_KV_BYTES", "1024")


def _state(*, scheduler: FakeScheduler | None = None) -> SimpleNamespace:
    lock = TrackingLock()
    bank = FakeBank(lock)
    backend = FakeBackend()
    kv = FakeKVHeadroom(lock)
    return SimpleNamespace(
        lock=lock,
        sessions=SimpleNamespace(bank=bank),
        model_scheduler=scheduler or FakeScheduler(),
        foreground_count=lambda: 0,
        model_weights_bytes=20 * 1024**3,
        machine_info={"unified_memory_bytes": 100 * 1024**3},
        expert_locality=FakeLocality(),
        expert_residency_backend=backend,
        kv_cache_manager=kv,
        kv_cache_bytes=2 * 1024**3,
    )


def _selected_memory_profiles() -> tuple[str, ...]:
    selected = os.environ.get("MTPLX_MATRIX_PROFILE", "").strip()
    profiles = ("default-off", "expert-only", "unified-only", "combined-memory")
    return (selected,) if selected in profiles else profiles


@pytest.mark.parametrize("profile", _selected_memory_profiles())
def test_process_level_memory_configuration_matrix(profile, monkeypatch):
    if profile in {"expert-only", "combined-memory"}:
        _enable_expert(monkeypatch)
    if profile in {"unified-only", "combined-memory"}:
        _enable_unified(monkeypatch)

    state = _state()
    systems = ensure_native_adaptive_state(state)
    result = native_adaptive_tick(state)
    snapshot = augment_systems_snapshot({}, state)

    expert_enabled = profile in {"expert-only", "combined-memory"}
    unified_enabled = profile in {"unified-only", "combined-memory"}
    assert systems["expert_residency"].config.enabled is expert_enabled
    assert systems["unified_memory"].config.enabled is unified_enabled
    assert snapshot["expert_residency"]["enabled"] is expert_enabled
    assert snapshot["unified_memory"]["enabled"] is unified_enabled
    assert snapshot["expert_residency"]["router_mutation"] is False

    if profile == "default-off":
        assert result["reason"] == "disabled"
        assert state.lock.acquire_calls == 0
    else:
        assert result["safe"] is True
        assert result["reason"] == "safe"
        assert state.lock.acquire_calls == 1
        assert state.lock.release_calls == 1
        assert not state.lock.locked()
        assert result["expert_residency_apply"]["applied"] is expert_enabled
        assert result["unified_memory_apply"]["applied"] is unified_enabled

    assert bool(state.expert_residency_backend.resident) is expert_enabled
    assert bool(state.sessions.bank.calls) is unified_enabled
    assert bool(state.kv_cache_manager.calls) is unified_enabled


@pytest.mark.parametrize(
    ("blocker", "expected_reason"),
    (
        ("foreground", "foreground_active"),
        ("foreground-error", "foreground_unknown"),
        ("scheduler", "scheduler_busy"),
        ("scheduler-malformed", "scheduler_busy"),
        ("restore_active", "restore_active"),
        ("commit_active", "commit_active"),
        ("mtp_transaction_active", "mtp_transaction_active"),
        ("postcommit_active", "postcommit_active"),
        ("generation_active", "generation_active"),
    ),
)
def test_safe_point_blocker_matrix(blocker, expected_reason, monkeypatch):
    _enable_expert(monkeypatch)
    _enable_unified(monkeypatch)
    scheduler = FakeScheduler(
        busy=blocker == "scheduler",
        malformed=blocker == "scheduler-malformed",
    )
    state = _state(scheduler=scheduler)
    if blocker == "foreground":
        state.foreground_count = lambda: 1
    elif blocker == "foreground-error":
        state.foreground_count = lambda: (_ for _ in ()).throw(RuntimeError("x"))
    elif blocker not in {"scheduler", "scheduler-malformed"}:
        setattr(state, blocker, True)

    result = native_adaptive_tick(state)
    assert result["safe"] is False
    assert result["reason"] == expected_reason
    assert result["unified_memory_apply"]["applied"] is False
    assert result["expert_residency_apply"]["applied"] is False
    assert state.sessions.bank.calls == []
    assert state.kv_cache_manager.calls == []
    assert state.expert_residency_backend.resident == set()
    assert not state.lock.locked()


def test_environment_parsing_clamps_invalid_and_unsafe_values(monkeypatch):
    monkeypatch.setenv("MTPLX_EXPERT_RESIDENCY", "YES")
    monkeypatch.setenv("MTPLX_EXPERT_RESIDENCY_BYTES", "-9")
    monkeypatch.setenv("MTPLX_EXPERT_RESIDENCY_MIN_OBSERVATIONS", "0")
    monkeypatch.setenv("MTPLX_EXPERT_RESIDENCY_HALF_LIFE_S", "invalid")
    monkeypatch.setenv("MTPLX_EXPERT_RESIDENCY_MAX_TRACKED", "0")
    monkeypatch.setenv("MTPLX_EXPERT_RESIDENCY_PREFETCH_PER_TICK", "-2")
    monkeypatch.setenv("MTPLX_EXPERT_RESIDENCY_EVICT_PER_TICK", "-3")
    expert = _expert_config_from_env()
    assert expert.enabled is True
    assert expert.budget_bytes == 0
    assert expert.minimum_observations == 1
    assert expert.half_life_s == 45.0
    assert expert.maximum_tracked_experts == 1
    assert expert.maximum_prefetch_per_tick == 0
    assert expert.maximum_evict_per_tick == 0

    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY", "on")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_RESERVE_BYTES", "-1")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_TARGET", "1.5")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_WARNING", "0.1")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_CRITICAL", "0.1")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_MIN_SESSION_BYTES", "-1")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_MIN_EXPERT_BYTES", "-1")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_MIN_KV_BYTES", "-1")
    unified = _unified_config_from_env()
    assert unified.enabled is True
    assert unified.reserve_bytes == 0
    assert unified.target_utilization == 0.99
    assert unified.warning_utilization == 0.99
    assert unified.critical_utilization == 0.99
    assert unified.minimum_session_bank_bytes == 0
    assert unified.minimum_expert_bytes == 0
    assert unified.minimum_kv_headroom_bytes == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("http://127.0.0.1:4318", "http://127.0.0.1:4318/v1/traces"),
        ("http://127.0.0.1:4318/", "http://127.0.0.1:4318/v1/traces"),
        ("http://127.0.0.1:4318/v1/traces", "http://127.0.0.1:4318/v1/traces"),
    ),
)
def test_otlp_environment_endpoint_matrix(raw, expected, monkeypatch):
    monkeypatch.setenv("MTPLX_OTLP_ENDPOINT", raw)
    monkeypatch.setenv("MTPLX_OTLP_HEADERS", "authorization=Bearer%20abc,x-test=1")
    config = OTLPExporterConfig.from_env()
    assert config.enabled is True
    assert config.endpoint == expected
    assert config.headers == {"authorization": "Bearer abc", "x-test": "1"}
    assert config.allow_content is False

    monkeypatch.setenv("MTPLX_OTLP_DISABLED", "true")
    assert OTLPExporterConfig.from_env().enabled is False


class _Collector(BaseHTTPRequestHandler):
    payloads: list[dict[str, object]] = []

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        type(self).payloads.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.end_headers()

    def log_message(self, _format, *_args):
        return


def _write_capture(root: Path) -> None:
    (root / "case.json").write_text(
        json.dumps(
            {
                "request_id": "case",
                "request": {"payload": {"prompt": "private request"}},
                "outcome": {"response": {"answer": "baseline"}},
            }
        ),
        encoding="utf-8",
    )


def test_all_enabled_cross_system_matrix(monkeypatch, tmp_path):
    _enable_expert(monkeypatch)
    _enable_unified(monkeypatch)
    capture_root = tmp_path / "captures"
    capture_root.mkdir()
    _write_capture(capture_root)
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_DIR", str(capture_root))
    monkeypatch.setenv("MTPLX_REPLAY_MAX_CASES", "4")

    _Collector.payloads.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "MTPLX_OTLP_ENDPOINT",
        f"http://127.0.0.1:{server.server_port}",
    )
    monkeypatch.setenv("MTPLX_OTLP_BATCH_SIZE", "1")
    monkeypatch.setenv("MTPLX_OTLP_FLUSH_INTERVAL_S", "0.01")

    state = _state()
    systems = ensure_native_adaptive_state(state)
    systems["policy_hooks"].register(
        "matrix-rewrite",
        lambda value, _context: HookResult.rewrite({**value, "matrix": True}),
        phases=[HookPhase.REQUEST],
    )

    try:
        tick = native_adaptive_tick(state)
        policy = systems["policy_hooks"].execute(HookPhase.REQUEST, {"value": 1})
        with systems["otlp"].span(
            "mtplx.matrix",
            attributes={
                "prompt": "must not leave process",
                "authorization": "Bearer private",
                "prompt_tokens": 7,
            },
        ):
            pass
        assert systems["otlp"].flush(timeout_s=2.0)

        orchestrator = systems["replay_orchestrator"]
        assert orchestrator is not None
        plan = orchestrator.build_plan()
        receipt = orchestrator.run(
            plan,
            candidate=lambda request: request,
            evaluators={
                "ok": lambda _case, _output, _baseline: Evaluation(
                    "ok", score=1.0, passed=True, baseline_score=1.0
                )
            },
            policy=RegressionPolicy(minimum_pass_rate=1.0),
        )
        snapshot = augment_systems_snapshot({}, state)
    finally:
        systems["otlp"].shutdown(timeout_s=1.0)
        server.shutdown()
        thread.join(timeout=1.0)

    assert tick["unified_memory_apply"]["applied"] is True
    assert tick["expert_residency_apply"]["applied"] is True
    assert policy.allowed is True
    assert policy.rewritten is True
    assert policy.value == {"value": 1, "matrix": True}
    assert receipt.promotion_applied is False
    assert receipt.decision["promote"] is True
    assert snapshot["expert_residency"]["enabled"] is True
    assert snapshot["unified_memory"]["enabled"] is True
    assert snapshot["otlp_export"]["enabled"] is True
    assert snapshot["policy_hooks"]["enabled"] is True
    assert snapshot["replay_orchestration"]["enabled"] is True
    assert snapshot["replay_orchestration"]["promotion_is_automatic"] is False
    assert _Collector.payloads
    exported = json.dumps(_Collector.payloads)
    assert "must not leave process" not in exported
    assert "Bearer private" not in exported
    assert "prompt_tokens" in exported


def test_concurrent_first_use_returns_one_namespaced_system_set(monkeypatch):
    monkeypatch.setenv("MTPLX_OTLP_DISABLED", "1")
    state = SimpleNamespace()
    barrier = threading.Barrier(32)
    rows: list[tuple[int, int, int, int]] = []
    rows_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        systems = ensure_native_adaptive_state(state)
        row = (
            id(systems["expert_residency"]),
            id(systems["unified_memory"]),
            id(systems["otlp"]),
            id(systems["policy_hooks"]),
        )
        with rows_lock:
            rows.append(row)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert all(not thread.is_alive() for thread in threads)
    assert len(rows) == 32
    assert [len({row[index] for row in rows}) for index in range(4)] == [1, 1, 1, 1]
    assert hasattr(state, "_mtplx_native_adaptive_lock")
    assert hasattr(state, "_mtplx_expert_residency_controller")
    assert hasattr(state, "_mtplx_unified_memory_coordinator")
    assert hasattr(state, "_mtplx_otlp_exporter")
    assert hasattr(state, "_mtplx_policy_bus")
