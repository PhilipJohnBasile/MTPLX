"""App/CLI coding-agent environment SYNC PAIR.

The macOS app's ``codingAgentRuntimeEnvironment`` (MTPLXCommandBuilder.swift)
and the CLI's ``_opencode_memory_env_defaults`` (commands/public.py) must
launch the same engine: every drifted key means app users and CLI users run
different runtimes for the same workload (2026-08-03 parity audit found the
GQA-SDPA route app-only and MTPLX_LAZY_TARGET_DISTRIBUTIONS CLI-only, and
the CLI Pi lane missing the entire block). These tests parse the Swift
source — the same approach test_model_catalog.py uses for the catalog pair —
so future drift fails loudly in CI.
"""

from __future__ import annotations

import re
from pathlib import Path

from mtplx.commands.public import (
    _apply_pi_history_budget_env_defaults,
    _opencode_memory_env_defaults,
)

_SWIFT = (
    Path(__file__).parents[1]
    / "apps"
    / "MTPLXApp"
    / "Sources"
    / "MTPLXAppCore"
    / "Services"
    / "MTPLXCommandBuilder.swift"
)

# RAM-tiered on both sides; value equality is platform-dependent.
_DYNAMIC_KEYS = {"MTPLX_SESSION_BANK_MAX_ENTRIES"}


def _swift_coding_agent_env() -> dict[str, str]:
    source = _SWIFT.read_text(encoding="utf-8")
    start = source.index("private static func codingAgentRuntimeEnvironment")
    end = source.index("return environment", start)
    body = source[start:end]
    env: dict[str, str] = {}
    for key, value in re.findall(r'"(MTPLX_[A-Z0-9_]+)"\s*:\s*"([^"]*)"', body):
        env[key] = value
    for key in re.findall(r'"(MTPLX_[A-Z0-9_]+)"\s*:\s*highMemory', body):
        env[key] = "<dynamic>"
    for key, value in re.findall(r'environment\["(MTPLX_[A-Z0-9_]+)"\]\s*=\s*"([^"]*)"', body):
        env[key] = value
    return env


def _swift_pi_overrides() -> dict[str, str]:
    source = _SWIFT.read_text(encoding="utf-8")
    start = source.index("case .pi:")
    end = source.index("case .openWebUI:", start)
    body = source[start:end]
    return dict(re.findall(r'piEnv\["(MTPLX_[A-Z0-9_]+)"\]\s*=\s*"([^"]*)"', body))


def test_opencode_coding_agent_env_matches_app():
    app_env = _swift_coding_agent_env()
    cli_env = _opencode_memory_env_defaults()

    assert app_env, "failed to parse codingAgentRuntimeEnvironment from Swift"
    missing_on_cli = sorted(set(app_env) - set(cli_env))
    missing_on_app = sorted(set(cli_env) - set(app_env))
    assert not missing_on_cli, f"app-only env keys (CLI drift): {missing_on_cli}"
    assert not missing_on_app, f"CLI-only env keys (app drift): {missing_on_app}"

    mismatched = {
        key: (cli_env[key], app_env[key])
        for key in cli_env
        if key not in _DYNAMIC_KEYS and app_env[key] != "<dynamic>"
        and str(cli_env[key]) != app_env[key]
    }
    assert not mismatched, f"value drift (cli, app): {mismatched}"


def test_pi_lane_env_matches_app_composition():
    """CLI Pi = the shared coding-agent block, nothing more.

    Since #282 the app's .pi case sets no env overrides (the compaction
    battery re-armed compactors past the engine's passthrough default),
    so parity here means no overrides on either side.
    """
    app_pi = _swift_pi_overrides()
    assert app_pi == {}, f"the app grew .pi env overrides again: {app_pi}"

    cli_env: dict[str, str] = {}
    _apply_pi_history_budget_env_defaults(cli_env)

    # The shared engine block must be present (the pre-audit CLI Pi lane had
    # no session bank / SDPA route / frontier flags at all).
    shared = _opencode_memory_env_defaults()
    for key in shared:
        assert key in cli_env, f"Pi lane lost shared coding-agent key {key}"

    # And nothing beyond it: the Pi lane adds no rewrite machinery (#282).
    extra = sorted(set(cli_env) - set(shared))
    assert not extra, f"Pi lane grew keys beyond the shared block: {extra}"


_TRANSCRIPT_REWRITE_KEYS = (
    "MTPLX_TOOL_RESULT_COMPACT_THRESHOLD_CHARS",
    "MTPLX_ACTIVE_READ_INSPECTION_COMPACT_MAX_LINES",
    "MTPLX_ACTIVE_READ_INSPECTION_LINE_MAX_CHARS",
    "MTPLX_ACTIVE_TOOL_RESULT_COMPACT_MAX_LINES",
    "MTPLX_ACTIVE_TOOL_RESULT_LINE_MAX_CHARS",
    "MTPLX_ACTIVE_READ_INSPECTION_TOTAL_MAX_LINES",
    "MTPLX_ACTIVE_READ_INSPECTION_MIN_LINES_PER_FILE",
    "MTPLX_ACTIVE_READ_INSPECTION_MULTI_FILE_LINE_MAX_CHARS",
    "MTPLX_READ_ONLY_INSPECTION_FORCE_ANSWER_AFTER_TOOLS",
)


def test_no_launcher_exports_transcript_rewrite_envs():
    """#282 contract: an explicit env re-arms its compactor past the engine's
    passthrough default, so no launcher may export any of them."""
    app_env = _swift_coding_agent_env()
    cli_env = _opencode_memory_env_defaults()
    pi_env: dict[str, str] = {}
    _apply_pi_history_budget_env_defaults(pi_env)
    for key in _TRANSCRIPT_REWRITE_KEYS:
        assert key not in app_env, f"app exports rewrite env {key}"
        assert key not in cli_env, f"CLI exports rewrite env {key}"
        assert key not in pi_env, f"CLI Pi lane exports rewrite env {key}"
