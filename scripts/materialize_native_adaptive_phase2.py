#!/usr/bin/env python3
"""One-shot source integration for the audited native-adaptive phase-two branch.

The publishing workflow deletes this file before committing the final PR tree.
Every edit is marker-guarded and fails closed when the expected source shape is
not present.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, value: str) -> None:
    (ROOT / path).write_text(value, encoding="utf-8")


def replace_once(value: str, old: str, new: str, *, label: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return value.replace(old, new, 1)


def fix_expert_residency() -> None:
    path = "mtplx/expert_residency.py"
    value = read(path)
    value = replace_once(
        value,
        "from dataclasses import asdict, dataclass, field",
        "from dataclasses import dataclass, field, replace",
        label="expert dataclass imports",
    )
    pattern = re.compile(
        r"receipt = ExpertResidencyReceipt\(\*\*\{\*\*asdict\(receipt\), \"duration_ms\": duration_ms\}\)"
    )
    value, count = pattern.subn(
        "receipt = replace(receipt, duration_ms=duration_ms)", value, count=1
    )
    if count != 1:
        raise RuntimeError("expert receipt duration replacement did not match")
    write(path, value)


def fix_replay_request_detection() -> None:
    path = "mtplx/replay_orchestrator.py"
    value = read(path)
    value = replace_once(
        value,
        '        _first(record, ("messages",)),\n        _first(record, ("prompt_tokens",)),\n',
        '        _first(record, ("messages",)),\n',
        label="count-only prompt token candidate",
    )
    write(path, value)


def fix_unified_env_ordering() -> None:
    path = "mtplx/native_adaptive.py"
    value = read(path)
    start = value.index("def _unified_config_from_env() -> UnifiedMemoryConfig:\n")
    end = value.index("\n\ndef ensure_native_adaptive_state", start)
    replacement = '''def _unified_config_from_env() -> UnifiedMemoryConfig:
    gib = 1024**3
    target = min(
        0.99,
        max(0.20, _env_float("MTPLX_UNIFIED_MEMORY_TARGET", 0.88)),
    )
    warning = min(
        1.0,
        max(target, _env_float("MTPLX_UNIFIED_MEMORY_WARNING", 0.92)),
    )
    critical = min(
        1.0,
        max(warning, _env_float("MTPLX_UNIFIED_MEMORY_CRITICAL", 0.96)),
    )
    return UnifiedMemoryConfig(
        enabled=_env_bool("MTPLX_UNIFIED_MEMORY", False),
        reserve_bytes=max(0, _env_int("MTPLX_UNIFIED_MEMORY_RESERVE_BYTES", 4 * gib)),
        target_utilization=target,
        warning_utilization=warning,
        critical_utilization=critical,
        minimum_session_bank_bytes=max(
            0, _env_int("MTPLX_UNIFIED_MEMORY_MIN_SESSION_BYTES", 256 * 1024**2)
        ),
        minimum_expert_bytes=max(
            0, _env_int("MTPLX_UNIFIED_MEMORY_MIN_EXPERT_BYTES", 0)
        ),
        minimum_kv_headroom_bytes=max(
            0, _env_int("MTPLX_UNIFIED_MEMORY_MIN_KV_BYTES", 256 * 1024**2)
        ),
    )
'''
    value = value[:start] + replacement + value[end:]
    write(path, value)


def fix_response_content_length() -> None:
    path = "mtplx/native_adaptive.py"
    value = read(path)
    marker = "# MTPLX_NATIVE_ADAPTIVE_CONTENT_LENGTH"
    if marker in value:
        return
    old = '''                if response_start is not None:
                    await send(response_start)
                    response_start = None
                await send({**message, "body": body})
'''
    new = f'''                if response_start is not None:
                    {marker}
                    if not streaming and not more:
                        headers = [
                            (key, value)
                            for key, value in response_start.get("headers", [])
                            if key.lower() != b"content-length"
                        ]
                        headers.append((b"content-length", str(len(body)).encode()))
                        response_start["headers"] = headers
                    await send(response_start)
                    response_start = None
                await send({{**message, "body": body}})
'''
    value = replace_once(value, old, new, label="ASGI final response send")
    write(path, value)


def integrate_openai_server() -> None:
    path = "mtplx/server/openai.py"
    value = read(path)
    marker = "# MTPLX_NATIVE_ADAPTIVE_PHASE2"
    if marker in value:
        return
    block = r'''

# MTPLX_NATIVE_ADAPTIVE_PHASE2
# Installed at module import after the FastAPI app and phase-one helpers exist.
# Every capability remains disabled/inert unless its own explicit configuration
# or trusted in-process registration is present.
try:
    from mtplx.native_adaptive import (
        augment_systems_snapshot as _augment_native_adaptive_snapshot,
        install_native_adaptive_middleware as _install_native_adaptive_middleware,
        native_adaptive_tick as _native_adaptive_tick,
    )

    _phase_one_systems_snapshot = globals().get("_mtplx_runtime_systems_snapshot")
    if callable(_phase_one_systems_snapshot):

        def _mtplx_runtime_systems_snapshot(state: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            phase_one = _phase_one_systems_snapshot(state, *args, **kwargs)
            if not isinstance(phase_one, dict):
                phase_one = {"phase_one": phase_one}
            return _augment_native_adaptive_snapshot(phase_one, state)

    _phase_one_memory_governor_tick = globals().get("_memory_governor_tick")
    if callable(_phase_one_memory_governor_tick):

        def _memory_governor_tick(state: Any, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
            phase_one = _phase_one_memory_governor_tick(state, *args, **kwargs)
            try:
                adaptive = _native_adaptive_tick(state)
            except Exception as exc:
                adaptive = {
                    "safe": False,
                    "reason": f"adaptive_tick_failed:{type(exc).__name__}",
                }
            if isinstance(phase_one, dict):
                return {**phase_one, "native_adaptive": adaptive}
            return {"phase_one": phase_one, "native_adaptive": adaptive}

    def _native_adaptive_state_provider() -> Any | None:
        for _state_name in ("_STATE", "STATE", "state", "_state"):
            _candidate = globals().get(_state_name)
            if _candidate is not None:
                return _candidate
        return None

    for _candidate_app in list(globals().values()):
        _module_name = getattr(type(_candidate_app), "__module__", "")
        if _module_name.startswith("fastapi") and callable(
            getattr(_candidate_app, "add_middleware", None)
        ):
            _install_native_adaptive_middleware(
                _candidate_app,
                runtime_state_provider=_native_adaptive_state_provider,
            )
            break
except Exception:
    # Optional native systems must never make the OpenAI-compatible server
    # unimportable. Their unavailable state remains visible through phase-one
    # diagnostics when installation cannot complete.
    pass
'''
    write(path, value.rstrip() + block + "\n")


def integrate_dashboard() -> None:
    path = "dashboard/src/components/SystemsTab.tsx"
    value = read(path)
    marker = "MTPLX_NATIVE_ADAPTIVE_PANEL"
    if marker in value:
        return
    import_line = 'import { AdaptiveSystemsPanel } from "./AdaptiveSystemsPanel";\n'
    imports = list(re.finditer(r"^import .*?;\n", value, flags=re.MULTILINE))
    if not imports:
        raise RuntimeError("SystemsTab has no import block")
    insertion = imports[-1].end()
    value = value[:insertion] + import_line + value[insertion:]

    closing_patterns = (
        "    </div>\n  );\n}\n",
        "    </div>\n  );\n}",
    )
    closing = next((item for item in closing_patterns if item in value), None)
    if closing is None:
        raise RuntimeError("SystemsTab root closing pattern not found")
    panel = '''      {/* MTPLX_NATIVE_ADAPTIVE_PANEL */}
      <div className="col-span-12">
        <AdaptiveSystemsPanel />
      </div>
'''
    index = value.rfind(closing)
    value = value[:index] + panel + value[index:]
    write(path, value)


def update_docs() -> None:
    path = "docs/native-systems.md"
    value = read(path)
    marker = "## Phase-two adaptive systems"
    if marker not in value:
        value = value.rstrip() + f'''\n\n{marker}\n\nMTPLX also provides active expert warm-set planning, shared unified-memory\nbudgets, privacy-first OTLP/HTTP export, bounded lifecycle policy hooks, and\ncapture-to-replay planning with stale-source checks and non-automatic\npromotion receipts. See [Native adaptive systems](native-adaptive-systems.md).\n'''
        write(path, value)


def update_changelog() -> None:
    path = "CHANGELOG.md"
    value = read(path)
    marker = "Native adaptive systems: expert residency"
    if marker in value:
        return
    entry = (
        "- Native adaptive systems: expert residency/warm-set planning, atomic "
        "unified-memory budgets, dependency-free privacy-first OTLP export, "
        "bounded lifecycle policy hooks, and capture-to-replay orchestration.\n"
    )
    headings = list(re.finditer(r"^## .*?$", value, flags=re.MULTILINE))
    if headings:
        position = value.find("\n", headings[0].end()) + 1
        if position <= 0:
            position = headings[0].end() + 1
        value = value[:position] + "\n" + entry + value[position:]
    else:
        value = value.rstrip() + "\n\n## Unreleased\n\n" + entry
    write(path, value)


def main() -> None:
    fix_expert_residency()
    fix_replay_request_detection()
    fix_unified_env_ordering()
    fix_response_content_length()
    integrate_openai_server()
    integrate_dashboard()
    update_docs()
    update_changelog()


if __name__ == "__main__":
    main()
