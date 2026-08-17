"""Stream/non-stream parity for the stop-path generation-result envelope.

For the same stop-sequence request, stream=true and stream=false must emit
the same ``mtplx_stats`` KEY SET — the shape a client is told about a
request must not depend on the transport it chose. The stream side may
additionally carry ``dashboard_progress_*`` telemetry: those keys are
attached only on the SSE paths by design (there is no non-stream
``_attach_dashboard_progress_stats`` call site) and are not part of the
stop-envelope contract, so they are stripped before comparing.

Would-have-caught receipt: the non-stream chat stop literal historically
nested ``stats.finish_reason`` while the stream literal relied on a
post-final patch — removing that patch makes ``test_chat_stop_stats_key_set_parity``
fail with ``finish_reason`` missing on the stream side.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import test_server_openai as tso
from mtplx.server import openai
from mtplx.server.openai import create_app
from mtplx.server.response_envelope import build_generation_result

STOP_TEXT_CHUNKS = ("Hello ", "STOP\n")


def _fake_stream_generation(_state, _prompt_ids, **kwargs):
    token_callback = kwargs["token_callback"]
    cancel_event = kwargs["cancel_event"]
    for chunk in STOP_TEXT_CHUNKS:
        token_callback([ord(char) for char in chunk])
    assert cancel_event.wait(timeout=10), "stop match must cancel generation"
    token_callback([ord(char) for char in "after"])
    raise AssertionError("cancelled token callback must raise")


def _fake_nonstream_generation(_state, _prompt_ids, **kwargs):
    token_callback = kwargs["token_callback"]
    for chunk in STOP_TEXT_CHUNKS:
        token_callback([ord(char) for char in chunk])
    raise AssertionError("stop match must abort generation via callback")


def _final_stream_stats(response_text: str) -> dict:
    payloads = tso._stream_payloads(response_text)
    final = [
        payload for payload in payloads if payload["choices"][0].get("finish_reason")
    ]
    assert final, "stream must emit a finish_reason chunk"
    return final[-1]["mtplx_stats"]


def _stop_envelope_keys(stats: dict) -> set[str]:
    return {key for key in stats if not key.startswith("dashboard_progress_")}


def _chat_stats(monkeypatch, *, stream: bool) -> dict:
    state = tso._fake_streaming_session_state()
    state.args.stream_interval = 1
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_stream_generation if stream else _fake_nonstream_generation,
    )
    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "messages": [{"role": "user", "content": "Say hello"}],
            "enable_thinking": False,
            "stream": stream,
            "max_tokens": 32,
            "stop": ["STOP"],
        },
    )
    assert response.status_code == 200
    if stream:
        return _final_stream_stats(response.text)
    return response.json()["mtplx_stats"]


def _completions_stats(monkeypatch, *, stream: bool) -> dict:
    state = tso._fake_streaming_session_state()
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_stream_generation if stream else _fake_nonstream_generation,
    )
    response = client.post(
        "/v1/completions",
        json={
            "prompt": "say hello",
            "max_tokens": 32,
            "stream": stream,
            "stop": ["STOP"],
        },
    )
    assert response.status_code == 200
    if stream:
        return _final_stream_stats(response.text)
    return response.json()["mtplx_stats"]


def test_chat_stop_stats_key_set_parity(monkeypatch):
    stream_stats = _chat_stats(monkeypatch, stream=True)
    nonstream_stats = _chat_stats(monkeypatch, stream=False)
    assert _stop_envelope_keys(stream_stats) == _stop_envelope_keys(nonstream_stats)
    assert stream_stats["finish_reason"] == "stop"
    assert nonstream_stats["finish_reason"] == "stop"


def test_completions_stop_stats_key_set_parity(monkeypatch):
    stream_stats = _completions_stats(monkeypatch, stream=True)
    nonstream_stats = _completions_stats(monkeypatch, stream=False)
    assert _stop_envelope_keys(stream_stats) == _stop_envelope_keys(nonstream_stats)
    assert stream_stats["finish_reason"] == "stop"
    assert nonstream_stats["finish_reason"] == "stop"


def test_build_generation_result_canonical_shape():
    generated = build_generation_result(
        text="Hello ",
        tokens=[72, 105],
        prompt_tokens=3,
        completion_tokens=2,
        finish_reason="stop",
        generation_mode="mtp",
        mtp_depth=3,
        stats_extra={"stop_sequence_hit": True, "stop_sequence_matched": "STOP"},
    )
    assert set(generated) == {
        "text",
        "tokens",
        "prompt_tokens",
        "completion_tokens",
        "finish_reason",
        "stats",
    }
    # finish_reason must exist at BOTH levels at construction time; the
    # stream paths may not depend on a post-final patch to add it.
    assert generated["finish_reason"] == "stop"
    assert generated["stats"]["finish_reason"] == "stop"
    assert set(generated["stats"]) == {
        "generation_mode",
        "mtp_depth",
        "prompt_tokens",
        "completion_tokens",
        "finish_reason",
        "stop_sequence_hit",
        "stop_sequence_matched",
    }
    assert generated["stats"]["prompt_tokens"] == 3
    assert generated["stats"]["completion_tokens"] == 2
