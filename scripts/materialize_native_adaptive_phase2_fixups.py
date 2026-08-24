#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def update(path: str, replacements: list[tuple[str, str]]) -> None:
    target = ROOT / path
    value = target.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in value:
            raise RuntimeError(f"{path}: expected text not found: {old!r}")
        value = value.replace(old, new, 1)
    target.write_text(value, encoding="utf-8")


def main() -> None:
    update(
        "mtplx/expert_residency.py",
        [("from dataclasses import dataclass, field, replace", "from dataclasses import dataclass, replace")],
    )
    update(
        "mtplx/otlp_export.py",
        [("from dataclasses import dataclass, field, replace", "from dataclasses import dataclass, field")],
    )
    update(
        "mtplx/policy_hooks.py",
        [("import threading\nimport time\n", "import threading\n")],
    )
    update(
        "mtplx/replay_orchestrator.py",
        [("from typing import Any, Callable, Iterable, Mapping, Sequence", "from typing import Any, Callable, Mapping, Sequence")],
    )
    update(
        "tests/test_expert_residency.py",
        [("\nimport time\n", "\n")],
    )
    update(
        "tests/test_unified_memory.py",
        [("    BudgetConsumer,\n", "")],
    )
    update(
        "tests/test_native_adaptive.py",
        [("    systems = ensure_native_adaptive_state(state)\n", "    ensure_native_adaptive_state(state)\n")],
    )
    update(
        "dashboard/src/components/AdaptiveSystemsPanel.tsx",
        [
            (
                '  if (state.sampled === true || state.exported_spans) return "observed";',
                '  if (\n    state.sampled === true ||\n    (typeof state.exported_spans === "number" && state.exported_spans > 0)\n  )\n    return "observed";',
            ),
            ('  return JSON.stringify(value);', '  return JSON.stringify(value) ?? "—";'),
        ],
    )


if __name__ == "__main__":
    main()
