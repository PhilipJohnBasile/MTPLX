"""Deterministic SSE transcript capture for the stream hot path.

Serves two roles:

1. A normal regression test: deterministic fake token streams through the
   real endpoint must produce a parseable SSE body with the expected content
   reassembly.
2. A byte-identity gate for hot-path refactors: with MTPLX_SSE_DUMP set, the
   raw SSE bodies are written to that path. Running the same capture on two
   code states and diffing (after normalizing the per-request id and created
   timestamp) proves the serialization layer emits identical bytes for
   identical token streams — the live serve path cannot gate this because
   temp-0 output is not run-to-run deterministic across daemons.
"""

from __future__ import annotations

import json
import os

from fastapi.testclient import TestClient

import test_server_openai as tso
from mtplx.server.openai import create_app


def _capture_stream(state, body: dict) -> str:
    client = TestClient(create_app(state))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json=body,
    ) as response:
        assert response.status_code == 200
        return "".join(response.iter_text())


def _plain_content_scenario(monkeypatch) -> str:
    state = tso._fake_state()
    state.runtime.tokenizer = tso.CaptureTokenizer()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    text = (
        "Voil\u00e0 — a \"quoted\" backslash \\ and some \u4e2d\u6587.\n"
        "A second line with trailing words and a long single-line segment: "
        + " ".join(f"word{i}" for i in range(160))
        + "\nDone."
    )
    monkeypatch.setattr(
        tso.openai, "_run_generation", tso._fake_streaming_generation(text)
    )
    return _capture_stream(
        state,
        {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "max_tokens": 4096,
            "enable_thinking": False,
        },
    )


def _long_json_line_scenario(monkeypatch) -> str:
    state = tso._fake_state()
    state.runtime.tokenizer = tso.CaptureTokenizer()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    payload = json.dumps(
        {"items": [{"index": i, "value": f"v {i}"} for i in range(120)]}
    )
    text = "Result follows:\n" + payload + "\nEnd."
    monkeypatch.setattr(
        tso.openai, "_run_generation", tso._fake_streaming_generation(text)
    )
    return _capture_stream(
        state,
        {
            "messages": [{"role": "user", "content": "emit json"}],
            "stream": True,
            "max_tokens": 8192,
            "enable_thinking": False,
        },
    )


def _tool_call_scenario(monkeypatch) -> str:
    state = tso._fake_state()
    state.runtime.tokenizer = tso.CaptureTokenizer()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    text = (
        "Let me write the file.\n<tool_call>\n<function=write>\n"
        "<parameter=filePath>src/app.py</parameter>\n"
        "<parameter=content>print('hello')\nprint('world')</parameter>\n"
        "</function>\n</tool_call>"
    )
    monkeypatch.setattr(
        tso.openai,
        "_run_generation",
        tso._fake_streaming_generation(text, finish_reason="stop"),
    )
    return _capture_stream(
        state,
        {
            "messages": tso._tool_history_messages(),
            "tools": [tso._write_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 4096,
            "enable_thinking": False,
        },
    )


def test_stream_transcript_capture(monkeypatch):
    sections = {
        "plain": _plain_content_scenario(monkeypatch),
        "long_json_line": _long_json_line_scenario(monkeypatch),
        "tool_call": _tool_call_scenario(monkeypatch),
    }

    plain_payloads = tso._stream_payloads(sections["plain"])
    plain_content = "".join(
        choice.get("delta", {}).get("content", "") or ""
        for payload in plain_payloads
        for choice in payload.get("choices", [])
    )
    assert "Voil\u00e0" in plain_content
    assert "word159" in plain_content
    assert plain_content.endswith("Done.")

    json_payloads = tso._stream_payloads(sections["long_json_line"])
    json_content = "".join(
        choice.get("delta", {}).get("content", "") or ""
        for payload in json_payloads
        for choice in payload.get("choices", [])
    )
    assert '"index": 119' in json_content
    assert json_content.endswith("End.")

    tool_payloads = tso._stream_payloads(sections["tool_call"])
    tool_names = [
        item.get("function", {}).get("name")
        for payload in tool_payloads
        for choice in payload.get("choices", [])
        for item in choice.get("delta", {}).get("tool_calls", []) or []
        if isinstance(item, dict)
    ]
    assert "write" in tool_names

    dump_path = os.environ.get("MTPLX_SSE_DUMP")
    if dump_path:
        with open(dump_path, "w", encoding="utf-8") as fh:
            for name, body in sections.items():
                fh.write(f"===== scenario: {name} =====\n")
                fh.write(body)
                fh.write("\n")
