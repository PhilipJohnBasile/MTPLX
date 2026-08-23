from __future__ import annotations

from mtplx.runtime_systems import RuntimeSystemsRegistry


def test_registry_reports_both_integrated_system_families(monkeypatch, tmp_path):
    monkeypatch.delenv("MTPLX_REQUEST_CAPTURE_DIR", raising=False)
    payload = RuntimeSystemsRegistry().snapshot()
    assert payload["semantic_memory"]["available"] is True
    assert payload["expert_locality"]["available"] is True
    assert payload["memory_governor"]["available"] is True
    replay = payload["deterministic_replay"]
    assert replay["available"] is True
    assert replay["runtime_mutation"] is False
    assert replay["request_capture"]["wired"] is True
    assert replay["request_capture"]["enabled"] is False
    assert replay["request_capture"]["content_capture_default"] is False
    assert replay["replay"]["promotion_is_automatic"] is False
    assert replay["trace_parity"]["mode"] == "offline_compare"


def test_registry_tracks_only_persisted_capture_events(monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_DIR", str(tmp_path))
    registry = RuntimeSystemsRegistry()
    registry.record_request_capture(
        "dispatched", request_id="chatcmpl-a", session_id="s1", persisted=True
    )
    registry.record_request_capture(
        "completed", request_id="chatcmpl-a", session_id="s1", persisted=True
    )
    registry.record_request_capture(
        "completed", request_id="chatcmpl-b", session_id="s2", persisted=False
    )
    capture = registry.snapshot()["deterministic_replay"]["request_capture"]
    assert capture["enabled"] is True
    assert capture["dispatched"] == 1
    assert capture["completed"] == 2
    assert capture["failures"] == 1
    assert capture["latest"] == {
        "phase": "completed",
        "request_id": "chatcmpl-b",
        "session_id": "s2",
        "persisted": False,
        "updated_at_s": capture["latest"]["updated_at_s"],
    }
