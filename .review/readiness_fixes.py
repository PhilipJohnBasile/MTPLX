"""Apply scoped readiness fixes to an already merged, pinned worktree."""
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
number = int(sys.argv[2])

def replace(path, old, new):
    p = root / path
    source = p.read_text()
    assert source.count(old) == 1, (path, source.count(old))
    p.write_text(source.replace(old, new))

if number == 219:
    p = root / 'mtplx/server/openai.py'
    source = p.read_text()
    pattern = re.compile(r'^<<<<<<< .*?\n(.*?)^=======\n(.*?)^>>>>>>> .*?\n', re.M | re.S)
    matches = list(pattern.finditer(source))
    assert len(matches) == 1
    ours, theirs = matches[0].groups()
    assert ours.strip() == 'observability = {'
    assert 'client_links' in theirs and 'return {' in theirs
    p.write_text(pattern.sub(lambda match: theirs.replace('    return {', '    observability = {', 1), source))
    # Test both preserved request-link whitelisting and the adapter's caller contract.
    with (root / 'tests/test_request_observability_golden.py').open('a') as handle:
        handle.write('''

def test_current_main_request_links_survive_responses_integration():
    from mtplx.server.openai import ChatCompletionRequest, _request_observability

    request = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "hello"}])
    result = _request_observability(
        request,
        {"x-mtplx-client-turn-id": "turn-123", "x-mtplx-client-entry-id": "bad value\\nsecret"},
    )
    assert result["request_client_turn_id"] == "turn-123"
    assert "request_client_entry_id" not in result
    assert result["request_message_count"] == 1
''')
elif number == 362:
    provider = root / 'mtplx/server/runtime_status.py'
    assert not provider.exists()
    provider.write_text('''"""Read-only serving-status provider for the generic runtime systems registry.

Only bounded aggregate scalars cross this boundary. No model lock, allocator,
cache mutation, configuration changes, prompt content, paths, or client IDs.
"""
from __future__ import annotations

import threading
from typing import Any


def _counter(value: Any) -> int | None:
    # Do not stringify or invoke arbitrary object conversion for telemetry.
    return value if type(value) is int and 0 <= value <= 2**63 - 1 else None


class ServingStatusProvider:
    """Refresh from the existing server's CPU-side state, not a second owner."""

    def __init__(self, state: Any) -> None:
        self._state = state
        self._lock = threading.Lock()

    def publish(self) -> None:
        # Serialize collection/publication so an older read cannot replace a newer one.
        with self._lock:
            try:
                status = self._read()
            except Exception:
                # Never leave a prior healthy snapshot visible after a failed refresh.
                # Error text may contain paths or credentials, so only a fixed code escapes.
                status = {"available": False, "enabled": False, "wired": True,
                          "phase": "unavailable", "reason": "status_read_failed"}
            self._state.runtime_systems.update("serving", status)

    def _read(self) -> dict[str, Any]:
        state = self._state
        runtime = getattr(state, "runtime", None)
        released = getattr(state, "aime_parent_runtime_released", False) is True
        available = runtime is not None and not released
        foreground = getattr(state, "foreground_count", None)
        active = _counter(foreground() if callable(foreground)
                          else getattr(state, "foreground_active", None))
        in_flight = getattr(getattr(state, "dashboard", None), "in_flight", None)
        count = getattr(in_flight, "count", None)
        if callable(count):
            dashboard_active = _counter(count())
            if dashboard_active is not None:
                active = max(active or 0, dashboard_active)
        mode = getattr(getattr(state, "args", None), "generation_mode", None)
        mode = mode if type(mode) is str and mode in {"ar", "mtp"} else None
        return {
            "available": available,
            "enabled": available,
            "wired": True,
            "phase": ("unavailable" if not available else "unknown" if active is None
                      else "busy" if active else "idle"),
            "generation_mode": mode,
            "active_requests": active,
            "requests_completed": _counter(getattr(state, "requests_completed", None)),
            "requests_cancelled": _counter(getattr(state, "requests_cancelled", None)),
            "mtp_enabled": getattr(runtime, "mtp_enabled", False) is True if available else False,
            "sample_scope": "aggregate_scalars_not_a_transactional_snapshot",
        }
''')
    replace('mtplx/runtime_systems.py',
            'def install_runtime_systems_endpoint(app: Any, state: Any) -> None:',
            'def install_runtime_systems_endpoint(\n    app: Any, state: Any, *, refresh: Callable[[], None] | None = None\n) -> None:')
    replace('mtplx/runtime_systems.py',
            '    def mtplx_runtime_systems() -> dict[str, Any]:\n        return runtime_systems_snapshot(state)',
            '    def mtplx_runtime_systems() -> dict[str, Any]:\n        if refresh is not None:\n            refresh()\n        return runtime_systems_snapshot(state)')
    replace('mtplx/server/openai.py',
            '    install_runtime_systems_endpoint(app, state)',
            '''    from mtplx.server.runtime_status import ServingStatusProvider

    if getattr(state, "runtime_systems", None) is None:
        state.runtime_systems = RuntimeSystemsRegistry()
    serving_status = ServingStatusProvider(state)
    install_runtime_systems_endpoint(app, state, refresh=serving_status.publish)''')
    tests = root / 'tests/test_runtime_serving_status.py'
    assert not tests.exists()
    tests.write_text('''"""Live provider contract, including the actual production FastAPI wiring."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
from mtplx.runtime_systems import RuntimeSystemsRegistry
from mtplx.server.runtime_status import ServingStatusProvider


def state():
    return SimpleNamespace(runtime_systems=RuntimeSystemsRegistry(), runtime=SimpleNamespace(mtp_enabled=True),
        args=SimpleNamespace(generation_mode="mtp", model="/private/model", api_key="secret"),
        foreground_active=0, requests_completed=0, requests_cancelled=0)


def status(value):
    return value.runtime_systems.snapshot()["systems"]["serving"]["status"]


def test_provider_tracks_existing_serving_lifecycle_without_mutating_it():
    value = state()
    provider = ServingStatusProvider(value)
    provider.publish()
    assert status(value)["phase"] == "idle"
    value.foreground_active = 2
    value.requests_completed = 7
    value.requests_cancelled = 1
    provider.publish()
    assert status(value)["phase"] == "busy"
    assert status(value)["active_requests"] == 2
    assert status(value)["requests_completed"] == 7
    assert status(value)["requests_cancelled"] == 1
    assert value.foreground_active == 2
    value.runtime = None
    provider.publish()
    assert status(value)["available"] is False
    assert status(value)["mtp_enabled"] is False


def test_provider_counts_dashboard_and_foreground_without_model_lock():
    value = state()
    value.foreground_count = lambda: 1
    value.dashboard = SimpleNamespace(in_flight=SimpleNamespace(count=lambda: 3))
    class ForbiddenLock:
        def __enter__(self):
            raise AssertionError("Provider must never acquire the model lock")
    value.model_lock = ForbiddenLock()
    ServingStatusProvider(value).publish()
    assert status(value)["active_requests"] == 3


def test_failed_refresh_replaces_stale_status_and_redacts_errors():
    value = state()
    provider = ServingStatusProvider(value)
    provider.publish()
    def broken():
        raise RuntimeError("/private/path?api_key=secret")
    value.foreground_count = broken
    provider.publish()
    assert status(value) == {"available": False, "enabled": False, "wired": True,
                             "phase": "unavailable", "reason": "status_read_failed"}
    assert "secret" not in json.dumps(status(value))
    value.foreground_count = lambda: 0
    provider.publish()
    assert status(value)["phase"] == "idle"


def test_unknown_values_are_not_fabricated_as_valid_metrics():
    value = state()
    value.foreground_active = None
    value.requests_completed = True
    value.requests_cancelled = -1
    value.args.generation_mode = "secret"
    ServingStatusProvider(value).publish()
    result = status(value)
    assert result["phase"] == "unknown"
    assert result["active_requests"] is None
    assert result["requests_completed"] is None
    assert result["requests_cancelled"] is None
    assert result["generation_mode"] is None
    assert "private" not in json.dumps(result)
    assert "secret" not in json.dumps(result)


def test_released_runtime_is_not_advertised_available():
    value = state()
    value.aime_parent_runtime_released = True
    ServingStatusProvider(value).publish()
    assert status(value)["phase"] == "unavailable"
    assert status(value)["enabled"] is False


def test_concurrent_refresh_is_bounded_and_snapshots_are_detached():
    value = state()
    provider = ServingStatusProvider(value)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: provider.publish(), range(64)))
    snapshot = value.runtime_systems.snapshot()
    assert snapshot["system_count"] == 1
    assert snapshot["revision"] == 64
    snapshot["systems"]["serving"]["status"]["active_requests"] = 999
    assert status(value)["active_requests"] == 0


def test_production_app_refreshes_real_state_and_keeps_auth_boundary():
    from test_server_openai import _fake_state
    from mtplx.server.openai import create_app
    value = _fake_state(api_key="test-key")
    value.foreground_count = lambda: value.foreground_active
    value.foreground_active = 0
    value.requests_completed = 4
    client = TestClient(create_app(value))
    assert client.get("/v1/mtplx/systems").status_code == 401
    assert value.runtime_systems.snapshot()["system_count"] == 0
    response = client.get("/v1/mtplx/systems", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert response.json()["systems"]["serving"]["status"]["requests_completed"] == 4
    value.foreground_active = 3
    value.requests_completed = 5
    response = client.get("/v1/mtplx/systems", headers={"Authorization": "Bearer test-key"})
    payload = response.json()["systems"]["serving"]["status"]
    assert payload["phase"] == "busy"
    assert payload["active_requests"] >= 3
    assert payload["requests_completed"] == 5
    assert "test-key" not in response.text


def test_provider_import_does_not_import_mlx():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c",
        "import sys; from mtplx.server.runtime_status import ServingStatusProvider; "
        "assert not any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)"],
        cwd=root, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
''')
    doc = root / 'docs/runtime-systems.md'
    assert not doc.exists()
    doc.write_text('''# Read-only runtime systems

`GET /v1/mtplx/systems` uses the same authentication and rate-limit middleware as other `/v1` endpoints. The server now publishes a concrete `serving` provider from its existing CPU-side state whenever this endpoint is read. No separate polling thread or second runtime controller is introduced.

The provider exposes loaded/released availability, AR/MTP mode, foreground/dashboard request counts, completed/cancelled counters, and whether MTP is enabled. Its phase changes between unavailable, unknown, idle and busy; unknown counters are null, not invented zeroes. These aggregate scalars are sampled sequentially, not promised to be a transactional multi-counter snapshot.

Only fixed enums, booleans and bounded integers are published. Model paths, API keys, request/client IDs, prompt/tool content and exception messages are not copied. A failed refresh replaces prior healthy data with an unavailable status and fixed reason rather than leaving stale success visible. Publication is serialized per provider and the registry returns detached JSON.

The provider never loads or runs a model, acquires the model lock, resizes caches, changes configuration or touches the Metal allocator. Existing inference/runtime owners remain authoritative. The generic registry does not import this adapter; the server wires it as an optional refresh callback. Other components can publish independent bounded statuses through the same registry.

The Systems dashboard in #365 can render this actual provider without a placeholder. Production-app tests use the real `create_app` and HTTP route with an injected runtime test state, including authentication and live counter changes. That proves the wiring and contract, not a physical-model or packaged-desktop smoke.
''')
