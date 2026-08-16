"""Golden-file matrix for per-request observability (Phase 6.1 safety net).

The ~900-line request prologue exists three times (chat_completions,
completions, count_tokens) and writes ~130 scattered observability keys that
end up merged into ``mtplx_stats``. Before the RequestPolicy extraction may
touch any of it, this matrix pins the CURRENT envelope for the real client
mix (plain / OpenCode / Pi / Claude Code x tools x thinking) so the refactor
must reproduce it byte-for-byte (after volatile-field normalization).

Regenerate intentionally with:

    MTPLX_UPDATE_GOLDENS=1 .venv/bin/python -m pytest \
        tests/test_request_observability_golden.py

A diff in a golden file is a behavior change and must be explained in the
commit that carries it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import create_app

from test_server_openai import (  # noqa: E402 - shared fixtures
    ForegroundState,
    _fake_state,
)

GOLDEN_DIR = Path(__file__).parent / "golden" / "request_observability"
UPDATE = os.environ.get("MTPLX_UPDATE_GOLDENS", "").strip() in {"1", "true", "yes"}

# Values under these keys change run-to-run (clocks, ids, host facts) and are
# replaced by type placeholders. Keep this list tight: every key matched here
# is a key the goldens can no longer regress.
_VOLATILE_KEY_RE = re.compile(
    r"(_time_s$|_time$|_s$|_at$|^created$|^id$|elapsed|tok_s|_bytes$|"
    r"memory|_rpm$|uuid|request_id|response_id|_hash$|timestamp|seed)",
    re.IGNORECASE,
)


def _normalize(value, key: str = ""):
    if isinstance(value, dict):
        return {k: _normalize(v, k) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize(v, key) for v in value]
    if key and _VOLATILE_KEY_RE.search(key) and isinstance(value, (int, float, str)):
        return f"<{type(value).__name__}>"
    if isinstance(value, float):
        # Floats that survive normalization must be policy constants
        # (temperatures, fractions); round defensively against repr drift.
        return round(value, 6)
    return value


TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get local time for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]

TOOLS_ANTHROPIC = [
    {
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
]

BASE_HEADERS = {"x-mtplx-cache-mode": "bypass"}

# (arm name, route, headers, body)
MATRIX: list[tuple[str, str, dict[str, str], dict]] = [
    (
        "plain_chat",
        "/v1/chat/completions",
        {},
        {
            "model": "default",
            "messages": [{"role": "user", "content": "Reply OK only."}],
            "max_tokens": 8,
        },
    ),
    (
        "plain_chat_tools",
        "/v1/chat/completions",
        {},
        {
            "model": "default",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "max_tokens": 32,
            "tools": TOOLS_OPENAI,
            "tool_choice": "auto",
        },
    ),
    (
        "plain_chat_sampler_override",
        "/v1/chat/completions",
        {},
        {
            "model": "default",
            "messages": [{"role": "user", "content": "Reply OK only."}],
            "max_tokens": 8,
            "temperature": 0.2,
            "top_p": 0.9,
        },
    ),
    (
        "plain_chat_greedy",
        "/v1/chat/completions",
        {},
        {
            "model": "default",
            "messages": [{"role": "user", "content": "Reply OK only."}],
            "max_tokens": 8,
            "temperature": 0.0,
        },
    ),
    (
        "opencode_chat",
        "/v1/chat/completions",
        {"x-mtplx-client": "opencode", "user-agent": "opencode/1.4.2"},
        {
            "model": "default",
            "messages": [{"role": "user", "content": "Reply OK only."}],
            "max_tokens": 8,
        },
    ),
    (
        "opencode_chat_tools",
        "/v1/chat/completions",
        {"x-mtplx-client": "opencode", "user-agent": "opencode/1.4.2"},
        {
            "model": "default",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "max_tokens": 64,
            "tools": TOOLS_OPENAI,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        },
    ),
    (
        "opencode_chat_ua_sniffed",
        "/v1/chat/completions",
        {"user-agent": "opencode/1.4.2"},
        {
            "model": "default",
            "messages": [{"role": "user", "content": "Reply OK only."}],
            "max_tokens": 8,
        },
    ),
    (
        "pi_chat",
        "/v1/chat/completions",
        {"x-mtplx-client": "pi", "user-agent": "pi/0.9.0"},
        {
            "model": "default",
            "messages": [{"role": "user", "content": "Reply OK only."}],
            "max_tokens": 8,
        },
    ),
    (
        "pi_chat_tools",
        "/v1/chat/completions",
        {"x-mtplx-client": "pi", "user-agent": "pi/0.9.0"},
        {
            "model": "default",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "max_tokens": 64,
            "tools": TOOLS_OPENAI,
            "tool_choice": "auto",
        },
    ),
    (
        "claude_code_messages",
        "/v1/messages",
        {"user-agent": "claude-cli/1.0.44", "anthropic-version": "2023-06-01"},
        {
            "model": "default",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Reply OK only."}],
        },
    ),
    (
        "claude_code_messages_tools_noparallel",
        "/v1/messages",
        {"user-agent": "claude-cli/1.0.44", "anthropic-version": "2023-06-01"},
        {
            "model": "default",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": TOOLS_ANTHROPIC,
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        },
    ),
    (
        "claude_code_messages_thinking",
        "/v1/messages",
        {"user-agent": "claude-cli/1.0.44", "anthropic-version": "2023-06-01"},
        {
            "model": "default",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Reply OK only."}],
            "thinking": {"type": "enabled", "budget_tokens": 512},
        },
    ),
    (
        "plain_completions",
        "/v1/completions",
        {},
        {"model": "default", "prompt": "Say OK.", "max_tokens": 8},
    ),
]


def _fake_generation_output(*_args, **_kwargs):
    """Stand-in for generate_mtpk/generate_ar built on the REAL stats
    dataclass, so _run_generation's envelope assembly (and its
    request_observability merge — the whole point of these goldens) runs
    exactly as in production."""
    from mtplx.generation import GenerationStats

    stats = GenerationStats(
        mode="mtpk",
        generated_tokens=2,
        elapsed_s=0.01,
        tok_s=200.0,
        decode_elapsed_s=0.005,
        decode_tok_s=400.0,
        prompt_eval_time_s=0.005,
        prompt_tps=600.0,
        verify_calls=1,
        accepted_by_depth=[1],
    )
    return SimpleNamespace(
        tokens=[79, 75],
        text="OK",
        stats=stats,
        final_state=None,
        finish_reason="stop",
    )


def _client(monkeypatch) -> TestClient:
    monkeypatch.delenv("MTPLX_CLIENT", raising=False)
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.begin_foreground = foreground.begin_foreground
    state.end_foreground = foreground.end_foreground
    state.has_foreground = foreground.has_foreground
    state.foreground_count = foreground.foreground_count
    state.requests_completed = 0
    state.requests_cancelled = 0
    state.last_request_at = 0.0
    state.last_request_started_at = 0.0
    state.active_requests = 0
    # /v1/completions tokenizes the raw prompt itself.
    state.runtime.tokenizer.encode = lambda _text, **_kwargs: [1, 2, 3]
    monkeypatch.setattr(
        openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3]
    )
    # Fake the generators, not the serial worker: _run_generation must run
    # for real because it performs the request_observability merge.
    monkeypatch.setattr(openai, "generate_mtpk", _fake_generation_output)
    monkeypatch.setattr(openai, "generate_ar", _fake_generation_output)
    monkeypatch.setattr(openai, "generate_mtp1", _fake_generation_output, raising=False)
    return TestClient(create_app(state))


def _observability_from_response(route: str, payload: dict) -> dict:
    if route == "/v1/messages":
        stats = payload.get("mtplx_stats") or {}
    else:
        stats = payload.get("mtplx_stats") or {}
    if not stats:
        raise AssertionError(f"no mtplx_stats in response for {route}: {list(payload)}")
    return stats


@pytest.mark.parametrize(
    ("name", "route", "headers", "body"),
    MATRIX,
    ids=[row[0] for row in MATRIX],
)
def test_request_observability_matches_golden(monkeypatch, name, route, headers, body):
    client = _client(monkeypatch)

    response = client.post(route, headers={**BASE_HEADERS, **headers}, json=body)
    assert response.status_code == 200, f"{name}: {response.status_code} {response.text[:300]}"
    stats = _observability_from_response(route, response.json())
    normalized = _normalize(stats)

    golden_path = GOLDEN_DIR / f"{name}.json"
    if UPDATE:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(normalized, indent=1, sort_keys=True) + "\n")
        return

    assert golden_path.exists(), (
        f"missing golden {golden_path.name}; run MTPLX_UPDATE_GOLDENS=1 pytest "
        f"tests/test_request_observability_golden.py and review the diff"
    )
    golden = json.loads(golden_path.read_text())
    assert normalized == golden, (
        f"{name}: observability envelope drifted from golden "
        f"{golden_path.name} — if intentional, regenerate goldens and explain "
        f"the diff in the commit"
    )
