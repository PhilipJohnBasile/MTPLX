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


# ---------------------------------------------------------------------------
# Streaming arms (2026-08-16 tail sweep): the matrix above never consumed
# SSE, so the stream lane — reassembly, final-chunk envelope, terminal —
# had zero golden coverage. Same harness contract: fake ONLY the
# generators (callback-aware here, so tokens flow through the real
# record_tokens -> SSE assembly), let _run_generation and the endpoint
# run for real, pin the normalized final-chunk mtplx_stats.
# ---------------------------------------------------------------------------


def _fake_streaming_generation_output(text: str):
    """Callback-aware generator stand-in for the STREAM arms only.

    The non-stream arms keep ``_fake_generation_output`` (which never
    invokes ``token_callback``) so their pinned envelopes stay
    byte-identical; the stream arms need the tokens on the wire, exactly
    like the real generators deliver them.
    """
    from mtplx.generation import GenerationStats

    tokens = [ord(char) for char in text]

    def fake(*_args, **kwargs):
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        stats = GenerationStats(
            mode="mtpk",
            generated_tokens=len(tokens),
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
            tokens=tokens,
            text=text,
            stats=stats,
            final_state=None,
            finish_reason="stop",
        )

    return fake


def _stream_client(monkeypatch, text: str) -> TestClient:
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
    # Deterministic frames: flush every token, no footer prose on the wire
    # (same conventions as test_stream_transcript_golden).
    state.args.stream_interval = 1
    state.args.stats_footer = False
    state.runtime.tokenizer.encode = lambda _text, **_kwargs: [1, 2, 3]
    monkeypatch.setattr(
        openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3]
    )
    fake = _fake_streaming_generation_output(text)
    monkeypatch.setattr(openai, "generate_mtpk", fake)
    monkeypatch.setattr(openai, "generate_ar", fake)
    monkeypatch.setattr(openai, "generate_mtp1", fake, raising=False)
    return TestClient(create_app(state))


def _consume_sse(client: TestClient, route: str, headers: dict, body: dict) -> dict:
    """Minimal SSE consumption: data: frames, reassembly, final chunk, [DONE]."""
    with client.stream(
        "POST", route, headers={**BASE_HEADERS, **headers}, json=body
    ) as response:
        assert response.status_code == 200, response.status_code
        raw = "".join(response.iter_text())

    frames = [
        json.loads(line.removeprefix("data: "))
        for line in raw.splitlines()
        if line.startswith("data: {")
    ]
    done_terminal = any(
        line.strip() == "data: [DONE]" for line in raw.splitlines()
    )
    content = ""
    reasoning = ""
    tool_calls: dict[int, dict[str, str]] = {}
    finish_reason = None
    final_frame = None
    for frame in frames:
        for choice in frame.get("choices") or []:
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            for item in delta.get("tool_calls") or []:
                if not isinstance(item, dict):
                    continue
                slot = tool_calls.setdefault(
                    int(item.get("index") or 0), {"name": "", "arguments": ""}
                )
                function = item.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
                final_frame = frame
    return {
        "raw": raw,
        "frames": frames,
        "done_terminal": done_terminal,
        "content": content,
        "reasoning": reasoning,
        "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
        "finish_reason": finish_reason,
        "final_frame": final_frame,
    }


def _assert_stream_golden(name: str, stats: dict) -> None:
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
        f"{name}: streaming observability envelope drifted from golden "
        f"{golden_path.name} — if intentional, regenerate goldens and explain "
        f"the diff in the commit"
    )


def test_plain_chat_stream_matches_golden_and_nonstream(monkeypatch):
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply OK only."}],
        "max_tokens": 8,
    }

    stream = _consume_sse(
        _stream_client(monkeypatch, "OK"),
        "/v1/chat/completions",
        {},
        {**body, "stream": True},
    )
    nonstream = (
        _stream_client(monkeypatch, "OK")
        .post("/v1/chat/completions", headers=BASE_HEADERS, json=body)
        .json()
    )

    # Terminal contract: exactly one [DONE] sentinel closes the stream.
    assert stream["done_terminal"], "stream must end with data: [DONE]"
    assert stream["raw"].rstrip().endswith("data: [DONE]")

    # Delta reassembly equals the non-stream content.
    assert stream["content"] == nonstream["choices"][0]["message"]["content"]
    assert stream["finish_reason"] == nonstream["choices"][0]["finish_reason"]

    # Final chunk carries the envelope: usage + mtplx_stats + finish_reason.
    final = stream["final_frame"]
    assert final is not None, "no finish_reason frame seen"
    assert final.get("usage"), "final chunk must carry usage"
    assert final.get("mtplx_stats"), "final chunk must carry mtplx_stats"
    assert final["usage"]["completion_tokens"] == (
        nonstream["usage"]["completion_tokens"]
    )

    _assert_stream_golden("plain_chat_stream", final["mtplx_stats"])


def test_tool_call_stream_matches_golden_and_nonstream(monkeypatch):
    from test_server_openai import _tool_history_messages, _write_tool_schema

    tool_text = (
        "Let me write the file.\n<tool_call>\n<function=write>\n"
        "<parameter=filePath>src/app.py</parameter>\n"
        "<parameter=content>print('hello')\nprint('world')</parameter>\n"
        "</function>\n</tool_call>"
    )
    body = {
        "model": "default",
        "messages": _tool_history_messages(),
        "tools": [_write_tool_schema()],
        "tool_choice": "auto",
        "max_tokens": 4096,
    }

    stream = _consume_sse(
        _stream_client(monkeypatch, tool_text),
        "/v1/chat/completions",
        {},
        {**body, "stream": True},
    )
    nonstream = (
        _stream_client(monkeypatch, tool_text)
        .post("/v1/chat/completions", headers=BASE_HEADERS, json=body)
        .json()
    )
    nonstream_message = nonstream["choices"][0]["message"]

    assert stream["done_terminal"], "stream must end with data: [DONE]"
    assert stream["raw"].rstrip().endswith("data: [DONE]")

    # Tool-call reassembly: streamed fragments rebuild the exact non-stream
    # tool calls (names and full argument JSON, byte for byte).
    assert stream["tool_calls"] == [
        {
            "name": call["function"]["name"],
            "arguments": call["function"]["arguments"],
        }
        for call in nonstream_message["tool_calls"]
    ]
    assert stream["finish_reason"] == "tool_calls"
    assert nonstream["choices"][0]["finish_reason"] == "tool_calls"

    # KNOWN DIVERGENCE (pinned, not endorsed — 2026-08-16 tail sweep): the
    # pre-tool-call preamble reaches the non-stream client as `content`
    # but streams out on the `reasoning_content` channel and is never
    # reconciled into a content delta. Stream and non-stream therefore
    # disagree about which channel carries the preamble. The fix belongs
    # in mtplx/server/openai.py's stream reconcile (owned by the
    # request-policy lane); when it lands, flip these two assertions to
    # plain content equality and regenerate this arm's golden.
    assert nonstream_message["content"] == "Let me write the file."
    assert stream["content"] == ""
    assert stream["reasoning"] == "Let me write the file.\n"

    final = stream["final_frame"]
    assert final is not None, "no finish_reason frame seen"
    assert final.get("usage"), "final chunk must carry usage"
    assert final.get("mtplx_stats"), "final chunk must carry mtplx_stats"

    _assert_stream_golden("plain_chat_tool_call_stream", final["mtplx_stats"])
