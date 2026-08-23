"""Tests for the privacy-first request-capture ring."""

from __future__ import annotations

import json
import os

from mtplx import request_capture
from mtplx.request_capture import CaptureOptions


def _enable(monkeypatch, tmp_path, keep="200"):
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("MTPLX_REQUEST_CAPTURE_KEEP", keep)
    for name in (
        "MTPLX_REQUEST_CAPTURE_INCLUDE_PROMPT_TOKENS",
        "MTPLX_REQUEST_CAPTURE_INCLUDE_PROMPT_TEXT",
        "MTPLX_REQUEST_CAPTURE_INCLUDE_RESPONSE_TEXT",
        "MTPLX_REQUEST_CAPTURE_INCLUDE_MESSAGES",
        "MTPLX_REQUEST_CAPTURE_INCLUDE_EXCEPTION_TEXT",
    ):
        monkeypatch.delenv(name, raising=False)
    request_capture._PATHS_BY_ID.clear()


def _record(tmp_path):
    files = [path for path in tmp_path.iterdir() if path.suffix == ".json"]
    assert len(files) == 1
    return json.loads(files[0].read_text())


def test_capture_defaults_to_content_off(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request(
        "chatcmpl-abc",
        {
            "prompt_token_ids": [1, 2, 3],
            "prompt_text": "private prompt",
            "messages": [{"role": "user", "content": "secret"}],
            "max_tokens": 64,
        },
    )
    record = _record(tmp_path)
    assert record["phase"] == "dispatched"
    assert record["prompt_token_count"] == 3
    assert "prompt_token_ids" not in record
    assert "prompt_text" not in record
    assert "prompt_text_excerpt" not in record
    assert "messages" not in record
    assert record["capture_content_policy"]["prompt_tokens"] is False
    assert record["max_tokens"] == 64


def test_explicit_content_options_are_bounded(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request(
        "chatcmpl-optin",
        {"prompt_token_ids": [1, 2, 3, 4], "prompt_text": "abcdefghij"},
        options=CaptureOptions(
            include_prompt_tokens=True,
            include_prompt_text=True,
            prompt_token_limit=2,
            text_head_chars=3,
            text_tail_chars=2,
        ),
    )
    record = _record(tmp_path)
    assert record["prompt_token_ids"] == [1, 2]
    assert record["prompt_tokens_clipped"] is True
    assert record["prompt_text_excerpt"] == {
        "text_head": "abc",
        "text_tail": "ij",
        "text_chars": 10,
        "text_clipped": True,
    }


def test_capture_then_outcome_merge_redacts_response_and_exception(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request("chatcmpl-abc", {"max_tokens": 64})
    request_capture.capture_outcome(
        "chatcmpl-abc",
        {
            "finish_reason": "stop",
            "completion_tokens": 5,
            "text": "private response",
            "exception_message": "token=secret",
        },
    )
    record = _record(tmp_path)
    assert record["phase"] == "completed"
    outcome = record["outcome"]
    assert outcome["finish_reason"] == "stop"
    assert "text" not in outcome
    assert "exception_message" not in outcome
    assert "text_sha256" in outcome
    assert "exception_message_sha256" in outcome


def test_dispatch_record_survives_missing_outcome(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request("chatcmpl-hang", {"prompt_token_ids": [7]})
    record = _record(tmp_path)
    assert record["phase"] == "dispatched"
    assert "outcome" not in record


def test_ring_prunes_to_keep_without_deleting(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path, keep="3")
    for index in range(6):
        request_capture.capture_request(f"r{index}", {"i": index})
    live = [name for name in os.listdir(tmp_path) if name.endswith(".json")]
    assert len(live) == 3
    assert len(os.listdir(tmp_path / "pruned")) == 3


def test_disabled_is_a_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("MTPLX_REQUEST_CAPTURE_DIR", raising=False)
    request_capture.capture_request("x", {"a": 1})
    request_capture.capture_outcome("x", {"b": 2})
    assert list(tmp_path.iterdir()) == []


def test_stable_json_digest_ignores_mapping_order():
    assert request_capture.stable_json_digest({"b": 2, "a": 1}) == (
        request_capture.stable_json_digest({"a": 1, "b": 2})
    )


def test_clip_text_head_tail():
    small = request_capture.clip_text_head_tail("hello", head=10, tail=10)
    assert small == {"text": "hello", "text_clipped": False}
    big = request_capture.clip_text_head_tail("a" * 100, head=10, tail=10)
    assert big["text_clipped"] and big["text_chars"] == 100
    assert len(big["text_head"]) == 10 and len(big["text_tail"]) == 10


def test_unsafe_request_ids_are_sanitized(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    request_capture.capture_request("../../etc/passwd", {"a": 1})
    files = [name for name in os.listdir(tmp_path) if name.endswith(".json")]
    assert len(files) == 1
    assert ".." not in files[0] and "/" not in files[0]


def test_large_clipped_outcome_cannot_bypass_content_off(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    assert request_capture.capture_request("chatcmpl-large", {"max_tokens": 64})
    clipped = request_capture.clip_text_head_tail("private-response-" * 500)
    assert "text_head" in clipped and "text_tail" in clipped
    assert request_capture.capture_outcome("chatcmpl-large", clipped)
    record = _record(tmp_path)
    outcome = record["outcome"]
    assert "text_head" not in outcome
    assert "text_tail" not in outcome
    assert "text_head_sha256" in outcome
    assert "text_tail_sha256" in outcome


def test_nested_credentials_are_redacted_even_with_message_opt_in(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    assert request_capture.capture_request(
        "chatcmpl-redact",
        {
            "observability": {"authorization": "Bearer secret", "mode": "mtp"},
            "messages": [
                {"role": "user", "content": "hello", "api_key": "message-secret"}
            ],
        },
        options=CaptureOptions(include_messages=True),
    )
    record = _record(tmp_path)
    assert record["observability"]["authorization"] == "<redacted>"
    assert record["messages"][0]["api_key"] == "<redacted>"
    assert "Bearer secret" not in str(record)
    assert "message-secret" not in str(record)
