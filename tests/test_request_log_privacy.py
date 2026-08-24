"""The default-on request JSONL must keep its "no prompt content" promise (#326).

Literal user text stays on the in-RAM surfaces (dashboard ring, trace labels);
the durable line carries a non-reversible digest under the same key unless
MTPLX_REQUEST_LOG_CONTENT=1 explicitly opts back in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtplx.server.openai import (  # noqa: E402
    _record_request_metrics,
    _redact_request_log_record,
    parse_args,
)

SECRET = "the launch codes are in the blue folder"


def _state_with_log(tmp_path: Path) -> SimpleNamespace:
    log_path = tmp_path / "request-log.jsonl"
    args = parse_args(["--warmup-tokens", "0", "--request-log-jsonl", str(log_path)])
    return SimpleNamespace(args=args, last_metrics=[], log_path=log_path)


def _logged_record(state: SimpleNamespace) -> dict:
    lines = state.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_durable_log_redacts_user_preview(tmp_path, monkeypatch):
    monkeypatch.delenv("MTPLX_REQUEST_LOG_CONTENT", raising=False)
    state = _state_with_log(tmp_path)
    _record_request_metrics(
        state,
        {"request_id": "r1", "request_last_user_preview": SECRET, "completion_tokens": 3},
    )
    record = _logged_record(state)
    assert SECRET not in state.log_path.read_text(encoding="utf-8")
    assert record["request_last_user_preview"].startswith("sha256:")
    # The digest is stable so forensics can still correlate repeated turns.
    assert record["request_last_user_preview"] == _redact_request_log_record(
        {"request_last_user_preview": SECRET}
    )["request_last_user_preview"]
    # The in-RAM ring (dashboard "recent") keeps the literal preview.
    assert state.last_metrics[-1]["request_last_user_preview"] == SECRET


def test_opt_in_keeps_literal_preview(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_REQUEST_LOG_CONTENT", "1")
    state = _state_with_log(tmp_path)
    _record_request_metrics(
        state,
        {"request_id": "r2", "request_last_user_preview": SECRET},
    )
    assert _logged_record(state)["request_last_user_preview"] == SECRET


def test_redaction_leaves_other_fields_alone(monkeypatch):
    monkeypatch.delenv("MTPLX_REQUEST_LOG_CONTENT", raising=False)
    record = {"completion_tokens": 7, "request_last_user_preview": None}
    assert _redact_request_log_record(record) == record
