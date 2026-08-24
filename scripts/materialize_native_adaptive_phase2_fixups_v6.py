#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(value: str, old: str, new: str, *, label: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return value.replace(old, new, 1)


def main() -> None:
    path = ROOT / "mtplx/native_adaptive.py"
    value = path.read_text(encoding="utf-8")
    value = replace_once(
        value,
        '''    controller: ExpertResidencyController = systems["expert_residency"]
    coordinator: UnifiedMemoryCoordinator = systems["unified_memory"]
    backend = systems["expert_backend"]
    model_lock = getattr(state, "lock", None)
''',
        '''    controller: ExpertResidencyController = systems["expert_residency"]
    coordinator: UnifiedMemoryCoordinator = systems["unified_memory"]
    backend = systems["expert_backend"]
    if not controller.config.enabled and not coordinator.config.enabled:
        return {
            "safe": True,
            "reason": "disabled",
            "unified_memory": coordinator.snapshot(),
            "expert_residency": controller.snapshot(backend),
        }
    model_lock = getattr(state, "lock", None)
''',
        label="disabled adaptive tick fast path",
    )
    value = replace_once(
        value,
        '''        policy_snapshot = bus.snapshot()
        hook_counts = policy_snapshot.get("hooks_by_phase", {})
        method = str(scope.get("method", ""))
''',
        '''        policy_snapshot = bus.snapshot()
        if not exporter.config.enabled and not policy_snapshot.get("enabled", False):
            await self.app(scope, receive, send)
            return
        hook_counts = policy_snapshot.get("hooks_by_phase", {})
        method = str(scope.get("method", ""))
''',
        label="disabled middleware fast path",
    )
    value = replace_once(
        value,
        '''                more = bool(message.get("more_body", False))
                body = bytes(message.get("body", b""))
                if more:
''',
        '''                more = bool(message.get("more_body", False))
                body = bytes(message.get("body", b""))
                original_body = body
                if more:
''',
        label="response original body",
    )
    value = replace_once(
        value,
        "                    if not streaming and not more:\n",
        "                    if not streaming and not more and body != original_body:\n",
        label="conditional content length rewrite",
    )
    path.write_text(value, encoding="utf-8")

    test_path = ROOT / "tests/test_native_adaptive.py"
    tests = test_path.read_text(encoding="utf-8")
    tests += '''


def test_disabled_adaptive_tick_does_not_touch_model_lock(monkeypatch):
    monkeypatch.delenv("MTPLX_UNIFIED_MEMORY", raising=False)
    monkeypatch.delenv("MTPLX_EXPERT_RESIDENCY", raising=False)

    class UntouchedLock:
        def acquire(self, **_kwargs):
            raise AssertionError("disabled tick touched the model lock")

    state = SimpleNamespace(lock=UntouchedLock())
    result = native_adaptive_tick(state)
    assert result["safe"] is True
    assert result["reason"] == "disabled"
'''
    test_path.write_text(tests, encoding="utf-8")


if __name__ == "__main__":
    main()
