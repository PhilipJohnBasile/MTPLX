from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from mtplx.expert_residency import ExpertRef
from mtplx.native_adaptive import (
    augment_systems_snapshot,
    ensure_native_adaptive_state,
    install_native_adaptive_middleware,
    native_adaptive_tick,
)
from mtplx.policy_hooks import (
    HookPhase,
    HookResult,
    default_policy_bus,
    reset_default_policy_bus_for_tests,
)


class FakeBank:
    def __init__(self, lock):
        self.lock = lock
        self.max_bytes = 8 * 1024**3
        self.per_session_max_bytes = 4 * 1024**3
        self.total_nbytes = 3 * 1024**3
        self.calls = []

    def rebalance_limits(self, *, max_bytes, per_session_max_bytes, reason):
        assert self.lock.locked()
        self.calls.append((max_bytes, per_session_max_bytes, reason))
        self.max_bytes = int(max_bytes)
        self.per_session_max_bytes = int(per_session_max_bytes)


class FakeScheduler:
    @staticmethod
    def stats():
        return {
            "foreground_pending": 0,
            "idle_pending": 0,
            "persistence_pending": 0,
            "active_kind": None,
        }


class FakeLocality:
    @staticmethod
    def snapshot():
        return {"expert_counts": {"0:0": 10, "0:1": 5}}


class FakeBackend:
    mode = "test"

    def __init__(self):
        self.resident = set()

    def resident_experts(self):
        return tuple(self.resident)

    def expert_nbytes(self, _expert):
        return 1024

    def prefetch_experts(self, experts):
        values = tuple(ExpertRef.coerce(item) for item in experts)
        self.resident.update(values)
        return values

    def evict_experts(self, experts):
        values = tuple(ExpertRef.coerce(item) for item in experts)
        self.resident.difference_update(values)
        return values


def make_state(monkeypatch):
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY", "1")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_RESERVE_BYTES", "0")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_MIN_SESSION_BYTES", "1024")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_MIN_EXPERT_BYTES", "1024")
    monkeypatch.setenv("MTPLX_UNIFIED_MEMORY_MIN_KV_BYTES", "1024")
    monkeypatch.setenv("MTPLX_EXPERT_RESIDENCY", "1")
    monkeypatch.setenv("MTPLX_EXPERT_RESIDENCY_BYTES", "4096")
    monkeypatch.setenv("MTPLX_EXPERT_RESIDENCY_MIN_OBSERVATIONS", "1")
    lock = threading.Lock()
    bank = FakeBank(lock)
    return SimpleNamespace(
        lock=lock,
        sessions=SimpleNamespace(bank=bank),
        model_scheduler=FakeScheduler(),
        foreground_count=lambda: 0,
        model_weights_bytes=20 * 1024**3,
        machine_info={"unified_memory_bytes": 100 * 1024**3},
        expert_locality=FakeLocality(),
        expert_residency_backend=FakeBackend(),
    )


def test_native_tick_holds_model_lock_for_all_budget_mutations(monkeypatch):
    state = make_state(monkeypatch)
    result = native_adaptive_tick(state)
    assert result["safe"] is True
    assert state.sessions.bank.calls
    assert state.expert_residency_backend.resident
    assert not state.lock.locked()
    assert result["unified_memory_apply"]["applied"] is True
    assert result["expert_residency_apply"]["applied"] is True


def test_native_tick_fails_closed_when_model_lock_is_busy(monkeypatch):
    state = make_state(monkeypatch)
    state.lock.acquire()
    try:
        result = native_adaptive_tick(state)
    finally:
        state.lock.release()
    assert result["safe"] is False
    assert result["reason"] == "model_lock_busy_or_unavailable"
    assert state.sessions.bank.calls == []


def test_system_snapshot_reports_each_phase_two_capability(monkeypatch, tmp_path):
    state = make_state(monkeypatch)
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_DIR", str(tmp_path))
    # Build after setting capture root so the orchestrator is installed.
    systems = ensure_native_adaptive_state(state)
    payload = augment_systems_snapshot({"existing": True}, state)
    assert payload["existing"] is True
    assert payload["expert_residency"]["available"] is True
    assert payload["unified_memory"]["available"] is True
    assert payload["otlp_export"]["available"] is True
    assert payload["policy_hooks"]["available"] is True
    assert payload["replay_orchestration"]["available"] is True
    assert payload["replay_orchestration"]["promotion_is_automatic"] is False


def test_fastapi_middleware_can_reject_request_without_external_service():
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    reset_default_policy_bus_for_tests()
    bus = default_policy_bus()
    bus.register(
        "deny",
        lambda value, _ctx: (
            HookResult.reject("model denied")
            if isinstance(value, dict) and value.get("model") == "blocked"
            else HookResult.allow()
        ),
        phases=[HookPhase.REQUEST],
    )
    app = FastAPI()

    @app.post("/v1/test")
    async def endpoint(payload: dict):
        return payload

    assert install_native_adaptive_middleware(app) is True
    assert install_native_adaptive_middleware(app) is False
    with TestClient(app) as client:
        blocked = client.post("/v1/test", json={"model": "blocked"})
        allowed = client.post("/v1/test", json={"model": "ok"})
    assert blocked.status_code == 403
    assert blocked.json()["error"]["type"] == "policy_rejection"
    assert allowed.status_code == 200
    assert allowed.json() == {"model": "ok"}
    reset_default_policy_bus_for_tests()


def test_middleware_applies_request_rewrite():
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    reset_default_policy_bus_for_tests()
    bus = default_policy_bus()
    bus.register(
        "rewrite",
        lambda value, _ctx: HookResult.rewrite({**value, "rewritten": True}),
        phases=[HookPhase.REQUEST],
    )
    app = FastAPI()

    @app.post("/v1/test")
    async def endpoint(payload: dict):
        return payload

    install_native_adaptive_middleware(app)
    with TestClient(app) as client:
        response = client.post("/v1/test", json={"value": 1})
    assert response.status_code == 200
    assert response.json() == {"value": 1, "rewritten": True}
    reset_default_policy_bus_for_tests()
