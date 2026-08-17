"""Empty-content-on-capped-thinking-row truth (F3, 2.8 wave).

Ivan's exact settings — a thinking model with a small max_tokens — produced
rows with empty content and no explanation:

  (1) the stream path recovers unclosed reasoning as content on finish=stop,
      the non-stream path did not (and /v1/messages inherits non-stream);
  (2) finish=length rows that ran out of budget inside the reasoning block
      now stamp ``content_empty_reason: "truncated_inside_reasoning"``
      (quiet-envelope: stamped only when it applies);
  (3) with thinking DISABLED, the stream splitter dropped the interior of an
      unclosed <think> block while the non-stream cleaner keeps the prose —
      reconciled to keep-prose.

Engine always monkeypatched — no model loads, CPU only.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.reasoning_codecs import strip_qwen_style_reasoning_from_content
from mtplx.server import openai
from mtplx.server.openai import (
    _nonstream_chat_message_parts,
    _ThinkingContentStreamSplitter,
    create_app,
)

from test_server_openai import (  # noqa: E402 - shared fixtures
    _fake_state,
    _fake_streaming_generation,
    _stream_payloads,
)


def _thinking_row_state():
    state = _fake_state()
    state.args.stats_footer = False
    return state


def _generated(text: str, finish_reason: str) -> dict:
    return {"text": text, "stats": {}, "finish_reason": finish_reason}


# --- (1) non-stream parity with the stream recovery -------------------------


def test_nonstream_recovers_unclosed_reasoning_on_stop():
    state = _thinking_row_state()
    generated = _generated("<think>the whole answer lives here", "stop")

    content, reasoning = _nonstream_chat_message_parts(
        state,
        generated,
        thinking_enabled=True,
        recover_unclosed_reasoning=True,
    )

    assert content == "the whole answer lives here"
    # Parity with the stream, where the reasoning deltas were already sent.
    assert reasoning == "the whole answer lives here"
    assert "content_empty_reason" not in generated["stats"]


def test_nonstream_recovery_needs_the_callers_gate():
    state = _thinking_row_state()
    generated = _generated("<think>private plan", "stop")

    content, reasoning = _nonstream_chat_message_parts(
        state,
        generated,
        thinking_enabled=True,
    )

    assert content == ""
    assert reasoning == "private plan"


def test_nonstream_recovery_refuses_runaway_sentinel_turns():
    state = _thinking_row_state()
    generated = _generated(
        "<think>looping forever<|im_start|>assistant", "stop"
    )

    content, _reasoning = _nonstream_chat_message_parts(
        state,
        generated,
        thinking_enabled=True,
        recover_unclosed_reasoning=True,
    )

    assert content == ""


# --- (2) content_empty_reason quiet-envelope stat ---------------------------


def test_nonstream_stamps_truncated_inside_reasoning_on_length():
    state = _thinking_row_state()
    generated = _generated("<think>thinking that never closes", "length")

    content, reasoning = _nonstream_chat_message_parts(
        state,
        generated,
        thinking_enabled=True,
    )

    assert content == ""
    assert reasoning == "thinking that never closes"
    assert (
        generated["stats"]["content_empty_reason"] == "truncated_inside_reasoning"
    )


def test_nonstream_quiet_envelope_when_content_present():
    state = _thinking_row_state()
    generated = _generated("<think>notes</think>real answer", "length")

    content, _reasoning = _nonstream_chat_message_parts(
        state,
        generated,
        thinking_enabled=True,
    )

    assert content == "real answer"
    assert "content_empty_reason" not in generated["stats"]


def test_nonstream_chat_endpoint_ivan_shaped_row(monkeypatch):
    """Thinking model + max_tokens=128 + finish=length: content stays empty
    (mirroring reasoning into content is a founder decision, NOT done here)
    but the envelope now says WHY."""

    state = _thinking_row_state()
    monkeypatch.setattr(
        openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3]
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: {
            "text": "<think>step 1... step 2... (budget ran out)",
            "tokens": [4],
            "stats": {
                "generation_mode": "ar",
                "mtp_depth": 0,
                "completion_tokens": 128,
            },
            "prompt_tokens": 3,
            "completion_tokens": 128,
            "finish_reason": "length",
        },
    )
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "hard question"}],
            "max_tokens": 128,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    message = payload["choices"][0]["message"]
    assert not (message.get("content") or "")
    assert message["reasoning_content"]
    assert payload["choices"][0]["finish_reason"] == "length"
    assert (
        payload["mtplx_stats"]["content_empty_reason"]
        == "truncated_inside_reasoning"
    )


def test_nonstream_chat_endpoint_recovers_on_stop(monkeypatch):
    state = _thinking_row_state()
    monkeypatch.setattr(
        openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3]
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: {
            "text": "<think>capped but finished naturally",
            "tokens": [4],
            "stats": {
                "generation_mode": "ar",
                "mtp_depth": 0,
                "completion_tokens": 8,
            },
            "prompt_tokens": 3,
            "completion_tokens": 8,
            "finish_reason": "stop",
        },
    )
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "hard question"}],
            "max_tokens": 128,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    message = payload["choices"][0]["message"]
    assert message["content"] == "capped but finished naturally"
    assert "content_empty_reason" not in payload["mtplx_stats"]


def test_v1_messages_inherits_nonstream_recovery(monkeypatch):
    state = _thinking_row_state()
    monkeypatch.setattr(
        openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3]
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: {
            "text": "<think>anthropic dialect sees the recovery too",
            "tokens": [4],
            "stats": {
                "generation_mode": "ar",
                "mtp_depth": 0,
                "completion_tokens": 8,
            },
            "prompt_tokens": 3,
            "completion_tokens": 8,
            "finish_reason": "stop",
        },
    )
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/messages",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "model": "mtplx-test-model",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "hard question"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    text_blocks = [
        block.get("text", "")
        for block in payload.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    assert any(
        "anthropic dialect sees the recovery too" in text for text in text_blocks
    )


def test_stream_stamps_truncated_inside_reasoning_on_length(monkeypatch):
    state = _thinking_row_state()
    monkeypatch.setattr(
        openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3]
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation("only thoughts", finish_reason="length"),
    )
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "hard question"}],
            "stream": True,
            "max_tokens": 128,
        },
    )

    assert response.status_code == 200
    frames = _stream_payloads(response.text)
    # The prompt pre-opened <think>; every streamed char is reasoning.
    content_deltas = [
        frame["choices"][0]["delta"].get("content")
        for frame in frames
        if frame.get("choices") and frame["choices"][0].get("delta")
    ]
    assert not any(content_deltas)
    final = [
        frame
        for frame in frames
        if frame.get("choices") and frame["choices"][0].get("finish_reason")
    ][-1]
    assert final["choices"][0]["finish_reason"] == "length"
    assert (
        final["mtplx_stats"]["content_empty_reason"]
        == "truncated_inside_reasoning"
    )


def test_stream_quiet_envelope_and_recovery_on_stop(monkeypatch):
    state = _thinking_row_state()
    monkeypatch.setattr(
        openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3]
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation("only thoughts", finish_reason="stop"),
    )
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "hard question"}],
            "stream": True,
            "max_tokens": 128,
        },
    )

    assert response.status_code == 200
    frames = _stream_payloads(response.text)
    content_text = "".join(
        frame["choices"][0]["delta"].get("content") or ""
        for frame in frames
        if frame.get("choices") and frame["choices"][0].get("delta")
    )
    # finish=stop recovery: the unclosed reasoning surfaced as content.
    assert content_text == "only thoughts"
    final = [
        frame
        for frame in frames
        if frame.get("choices") and frame["choices"][0].get("finish_reason")
    ][-1]
    assert "content_empty_reason" not in final["mtplx_stats"]


# --- (3) thinking-disabled keep-prose parity --------------------------------


def test_disabled_stream_splitter_keeps_unclosed_think_prose():
    splitter = _ThinkingContentStreamSplitter(thinking_enabled=False)

    chunks = []
    chunks.extend(splitter.feed("<think>pro"))
    chunks.extend(splitter.feed("se that must survive"))
    chunks.extend(splitter.finish())

    content = "".join(text for field, text in chunks if field == "content")
    assert content == "prose that must survive"
    # Byte-parity with the non-stream cleaner the mismatch was measured
    # against.
    assert content == strip_qwen_style_reasoning_from_content(
        "<think>prose that must survive"
    )


def test_disabled_stream_splitter_still_drops_closed_think_blocks():
    splitter = _ThinkingContentStreamSplitter(thinking_enabled=False)

    chunks = []
    chunks.extend(splitter.feed("<think>hidden reasoning</think>visible"))
    chunks.extend(splitter.finish())

    content = "".join(text for field, text in chunks if field == "content")
    assert content == "visible"
    assert content == strip_qwen_style_reasoning_from_content(
        "<think>hidden reasoning</think>visible"
    )


def test_disabled_stream_splitter_keeps_prose_after_visible_prefix():
    splitter = _ThinkingContentStreamSplitter(thinking_enabled=False)

    chunks = []
    chunks.extend(splitter.feed("lead text <think>tail that never closes"))
    chunks.extend(splitter.finish())

    content = "".join(text for field, text in chunks if field == "content")
    assert "lead text" in content
    assert "tail that never closes" in content
