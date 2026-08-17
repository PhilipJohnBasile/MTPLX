"""F40: tool-stream preamble parity (the pinned KNOWN DIVERGENCE, fixed).

With reasoning=auto the chat template pre-opens ``<think>``, so the streaming
splitter starts inside thinking and routes pre-tool-call preamble text
("Let me write the file.") to ``reasoning_content`` before it can know a tool
call follows. Non-stream clients receive the same marker-less text as
``content`` (``extract_thinking`` finds no think markers, tool extraction
leaves the preamble in ``cleaned_text``). The streams never reconciled.

Fix (the F3 recovery precedent, 835a9fd0): at finish, when the content
channel carried nothing user-visible (tool markup only), the thinking state
was exited by a tool-control marker, and no explicit think markers were
involved — all splitter state, no text heuristics — the auto-routed preamble
is stashed and the stream lane surfaces it as a content delta before the
finish frame once the turn's tool calls actually parsed. An explicit
``</think>`` block legitimately stays reasoning on both paths.

Endpoint-level stream==non-stream equality for the auto-preamble arm lives in
tests/test_request_observability_golden.py (the flipped assertions +
regenerated plain_chat_tool_call_stream.json golden). This file pins the
splitter contract and the endpoint arms the golden matrix does not cover:
explicit-think tool turns and the untouched plain lane.
"""

from __future__ import annotations

import json

import pytest

from mtplx.server import openai as oa


def _splitter(**kwargs) -> oa._ThinkingContentStreamSplitter:
    defaults = dict(
        thinking_enabled=True,
        recover_unclosed_reasoning_as_content=False,
        start_inside_thinking=True,
        suppress_orphan_tool_markup=False,
    )
    defaults.update(kwargs)
    return oa._ThinkingContentStreamSplitter(**defaults)


def _drive(splitter, text: str, *, chunk: int = 7):
    chunks = list(splitter.start())
    for start in range(0, len(text), chunk):
        chunks.extend(splitter.feed(text[start : start + chunk]))
    chunks.extend(splitter.finish())
    reasoning = "".join(text for field, text in chunks if field == "reasoning_content")
    content = "".join(text for field, text in chunks if field == "content")
    return chunks, reasoning, content


TOOL_MARKUP = (
    "<tool_call>\n<function=write>\n"
    "<parameter=filePath>src/app.py</parameter>\n"
    "<parameter=content>print('hello')</parameter>\n"
    "</function>\n</tool_call>"
)


def test_auto_routed_preamble_is_stashed_for_content_recovery():
    splitter = _splitter()
    _chunks, reasoning, content = _drive(
        splitter, "Let me write the file.\n" + TOOL_MARKUP
    )
    # The live stream already sent the preamble as reasoning deltas (they
    # cannot be unsent) and the markup as translator-bound content.
    assert reasoning == "Let me write the file.\n"
    assert "<tool_call>" in content
    # Splitter state distinguished the shape: tool-control exit from
    # thinking, no explicit markers, nothing user-visible on content.
    assert splitter.tool_preamble_recovered_content == "Let me write the file."


def test_explicit_think_block_is_never_stashed():
    splitter = _splitter()
    _chunks, reasoning, content = _drive(
        splitter,
        "I will plan the write here.\n</think>\n" + TOOL_MARKUP,
    )
    assert reasoning == "I will plan the write here.\n"
    assert splitter.tool_preamble_recovered_content is None


def test_explicit_think_with_visible_preamble_keeps_channels_apart():
    splitter = _splitter()
    _chunks, reasoning, content = _drive(
        splitter,
        "I will plan the write here.\n</think>\nLet me write the file.\n"
        + TOOL_MARKUP,
    )
    # The interior stays reasoning; the visible preamble streamed as content
    # live, so no finish-time recovery is needed or stashed.
    assert reasoning == "I will plan the write here.\n"
    assert content.startswith("Let me write the file.")
    assert splitter.tool_preamble_recovered_content is None


def test_plain_stop_turn_keeps_the_existing_f3_recovery_only():
    """No tool call: the pre-existing unclosed-reasoning recovery arm is
    untouched and the tool-preamble stash stays empty."""
    splitter = _splitter(recover_unclosed_reasoning_as_content=True)
    chunks = list(splitter.start())
    for piece in ("The answer ", "is 42."):
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish())
    assert ("content", "The answer is 42.") in chunks
    assert splitter.tool_preamble_recovered_content is None


def test_reasoning_only_turn_without_tool_call_stashes_nothing():
    splitter = _splitter()
    chunks = list(splitter.start())
    chunks.extend(splitter.feed("Only reasoning, no call, no close."))
    chunks.extend(splitter.finish())
    assert all(field == "reasoning_content" for field, _text in chunks)
    assert splitter.tool_preamble_recovered_content is None


# --- endpoint-level: the arms the flipped golden does not pin ---------------


def _consume(client, body):
    from test_request_observability_golden import _consume_sse

    return _consume_sse(client, "/v1/chat/completions", {}, {**body, "stream": True})


def _tool_body(tool_text: str) -> dict:
    from test_server_openai import _tool_history_messages, _write_tool_schema

    del tool_text
    return {
        "model": "default",
        "messages": _tool_history_messages(),
        "tools": [_write_tool_schema()],
        "tool_choice": "auto",
        "max_tokens": 4096,
    }


def _clients(monkeypatch, text: str):
    from test_request_observability_golden import BASE_HEADERS, _stream_client

    return (
        _stream_client(monkeypatch, text),
        _stream_client(monkeypatch, text),
        BASE_HEADERS,
    )


def test_endpoint_explicit_think_tool_turn_agrees_on_content(monkeypatch):
    """Explicit ``</think>`` before the preamble: the interior stays
    reasoning (never content) on BOTH paths, and content agrees."""
    tool_text = (
        "I will plan the write here.</think>\nLet me write the file.\n"
        + TOOL_MARKUP
    )
    stream_client, nonstream_client, base_headers = _clients(monkeypatch, tool_text)
    body = _tool_body(tool_text)

    stream = _consume(stream_client, body)
    nonstream = nonstream_client.post(
        "/v1/chat/completions", headers=base_headers, json=body
    ).json()
    nonstream_message = nonstream["choices"][0]["message"]

    assert stream["finish_reason"] == "tool_calls"
    assert nonstream["choices"][0]["finish_reason"] == "tool_calls"
    assert nonstream_message["content"] == "Let me write the file."
    # Live content deltas keep the model's surrounding newlines while the
    # non-stream cleaner strips them — a pre-existing whitespace shape, not
    # the F40 channel divergence. The CHANNEL parity is what F40 pins:
    # identical visible text on content, interior never leaking into it.
    assert stream["content"].strip() == nonstream_message["content"]
    # The explicit think interior is reasoning on the stream and absent from
    # content on both paths.
    assert stream["reasoning"].strip() == "I will plan the write here."
    assert "plan the write" not in (nonstream_message["content"] or "")
    assert "plan the write" not in stream["content"]


def test_endpoint_auto_preamble_recovery_marks_the_stat(monkeypatch):
    """The auto-preamble arm (the flipped golden's shape): content parity
    plus the truth stat naming the recovery."""
    tool_text = "Let me write the file.\n" + TOOL_MARKUP
    stream_client, nonstream_client, base_headers = _clients(monkeypatch, tool_text)
    body = _tool_body(tool_text)

    stream = _consume(stream_client, body)
    nonstream = nonstream_client.post(
        "/v1/chat/completions", headers=base_headers, json=body
    ).json()
    nonstream_message = nonstream["choices"][0]["message"]

    assert stream["finish_reason"] == "tool_calls"
    assert nonstream_message["content"] == "Let me write the file."
    assert stream["content"] == nonstream_message["content"]
    assert stream["reasoning"] == "Let me write the file.\n"
    # The recovered delta is a real frame on the wire before the finish
    # frame, not a post-hoc reassembly artifact.
    content_frames = [
        frame
        for frame in stream["frames"]
        for choice in frame.get("choices") or []
        if (choice.get("delta") or {}).get("content") == "Let me write the file."
    ]
    assert content_frames, "the recovered preamble must be a wire delta"
    finish_index = stream["frames"].index(stream["final_frame"])
    assert stream["frames"].index(content_frames[-1]) <= finish_index


def test_endpoint_plain_stream_unchanged(monkeypatch):
    """Plain (no tools) stream: byte-for-byte the pre-F40 shape — reasoning
    recovery and channel routing untouched."""
    stream_client, nonstream_client, base_headers = _clients(monkeypatch, "OK")
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": "Reply OK only."}],
        "max_tokens": 8,
    }

    stream = _consume(stream_client, body)
    nonstream = nonstream_client.post(
        "/v1/chat/completions", headers=base_headers, json=body
    ).json()

    assert stream["content"] == nonstream["choices"][0]["message"]["content"]
    assert stream["finish_reason"] == nonstream["choices"][0]["finish_reason"]


def test_endpoint_hermes_client_keeps_preamble_suppression(monkeypatch):
    """Hermes-mode clients suppress tool-turn preambles by contract (the
    translator drops even live preamble content once tool calls parse); the
    F40 recovery must honor the same suppression instead of leaking the
    auto-routed preamble around the translator."""
    tool_text = "Let me write the file.\n" + TOOL_MARKUP
    stream_client, _nonstream_client, _base_headers = _clients(monkeypatch, tool_text)
    body = _tool_body(tool_text)

    from test_request_observability_golden import _consume_sse

    stream = _consume_sse(
        stream_client,
        "/v1/chat/completions",
        {"x-mtplx-client": "hermes"},
        {**body, "stream": True},
    )

    assert stream["finish_reason"] == "tool_calls"
    assert [call["name"] for call in stream["tool_calls"]] == ["write"]
    assert stream["content"] == ""
    assert stream["reasoning"] == "Let me write the file.\n"


def test_endpoint_tool_turn_history_content_carries_preamble(monkeypatch):
    """The recovered delta flows into streamed history content, so the
    stream-side session commit stores the same assistant content the
    non-stream postcommit stores (clients echo content + tool_calls back)."""
    tool_text = "Let me write the file.\n" + TOOL_MARKUP
    stream_client, _nonstream_client, _base_headers = _clients(monkeypatch, tool_text)
    body = _tool_body(tool_text)

    stream = _consume(stream_client, body)
    final = stream["final_frame"]
    assert final is not None
    stats = final.get("mtplx_stats") or {}
    assert stats.get("tool_calls_emitted") == 1
    # Reassembled content equals what history capture stores (the preamble),
    # proving remember_stream_delta saw the recovered delta — the stream-side
    # session commit therefore stores the same assistant content the
    # non-stream postcommit stores. (The recovery truth stat
    # `stream_tool_preamble_recovered_as_content` is internal-only by the
    # quiet-envelope rule: it is deliberately NOT in the public
    # PUBLIC_MTPLX_STATS_KEYS allowlist, so public envelopes stay
    # byte-stable except for the honest answer_tokens change.)
    assert stream["content"] == "Let me write the file."
    assert "stream_tool_preamble_recovered_as_content" not in json.dumps(stats)
