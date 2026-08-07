"""B' arrival-side bounded finish window (2026-08-06 design note).

The 2026-07-17 policy aborted ANY pending postcommit immediately on the
next same-session request; the causal probes disproved its premise — the
pending snapshot is the arriving request's own exact-prefix restore
anchor. B': a pending job that has NOT started still aborts immediately;
a RUNNING job (record.mark_started, exactly what async_postcommit calls
at job start) gets MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S (default 0.6s; <= 0
restores the exact 2026-07-17 behavior; invalid falls back to default),
then aborts on timeout exactly as before. Legacy
MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S precedence, identity-safe concurrent
replacement, observability vocabulary, and the worker's separate 2.0s
self-yield are all unchanged.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from threading import Thread

from mtplx.engine_session import EngineSession, _postcommit_arrival_wait_s


def _session(sid: str = "sess-arrival-wait") -> EngineSession:
    return EngineSession(sid)


def _running_record(session: EngineSession, future: Future):
    future.set_running_or_notify_cancel()
    record = session.set_pending_postcommit(
        future, reason="tool_call_history_rewrite", token_count=16_000
    )
    record.mark_started()
    return record


def test_env_parser_default_invalid_and_zero(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", raising=False)
    assert _postcommit_arrival_wait_s() == 0.6
    monkeypatch.setenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", "garbage")
    assert _postcommit_arrival_wait_s() == 0.6
    monkeypatch.setenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", "")
    assert _postcommit_arrival_wait_s() == 0.6
    monkeypatch.setenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", "0")
    assert _postcommit_arrival_wait_s() == 0.0
    monkeypatch.setenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", "-3")
    assert _postcommit_arrival_wait_s() == 0.0
    monkeypatch.setenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", "1.25")
    assert _postcommit_arrival_wait_s() == 1.25
    # Non-finite values are configuration mistakes: NaN would read as
    # "disabled" and +inf would defeat the bounded policy entirely.
    for bad in ("nan", "inf", "-inf", "Infinity", "-Infinity", "NaN"):
        monkeypatch.setenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", bad)
        assert _postcommit_arrival_wait_s() == 0.6, bad


def test_queued_not_started_aborts_immediately(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", raising=False)
    monkeypatch.delenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", raising=False)
    session = _session()
    future: Future = Future()  # queued: never started, never marked
    record = session.set_pending_postcommit(future, reason="postcommit")
    assert record.started_at_s is None
    t0 = time.monotonic()
    outcome = session.resolve_pending_postcommit_for_request()
    assert time.monotonic() - t0 < 0.2, "queued abort must not wait"
    assert outcome["outcome"] == "aborted_for_foreground"
    assert outcome["waited"] is False
    assert record.abort_event.is_set()
    assert session._pending_postcommit is None


def test_running_completes_within_window_and_is_not_aborted(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", raising=False)
    monkeypatch.delenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", raising=False)
    session = _session()
    future: Future = Future()
    record = _running_record(session, future)
    finisher = Thread(target=lambda: (time.sleep(0.1), future.set_result(None)))
    finisher.start()
    try:
        t0 = time.monotonic()
        outcome = session.resolve_pending_postcommit_for_request()
        elapsed = time.monotonic() - t0
        assert outcome["outcome"] == "completed"
        assert outcome["waited"] is True
        assert 0.05 <= elapsed < 0.5
        assert outcome["timeout_s"] == 0.6
        assert not record.abort_event.is_set(), "completed job must not be aborted"
        assert session._pending_postcommit is None
    finally:
        finisher.join(timeout=2)


def test_running_timeout_is_bounded_then_aborts(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", raising=False)
    monkeypatch.setenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", "0.15")
    session = _session()
    future: Future = Future()  # running forever
    record = _running_record(session, future)
    t0 = time.monotonic()
    outcome = session.resolve_pending_postcommit_for_request()
    elapsed = time.monotonic() - t0
    assert 0.15 <= elapsed < 0.6, "wait must be bounded by the window"
    assert outcome["outcome"] == "aborted_for_foreground"
    assert outcome["waited"] is True
    assert outcome["abort_requested"] is True
    assert outcome["abort_reason"] == "foreground_preempted_postcommit"
    assert record.abort_event.is_set()
    assert record.last_abort_reason == "foreground_preempted_postcommit"
    assert session._pending_postcommit is None


def test_zero_window_restores_exact_immediate_abort(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", raising=False)
    monkeypatch.setenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", "0")
    session = _session()
    future: Future = Future()
    record = _running_record(session, future)
    t0 = time.monotonic()
    outcome = session.resolve_pending_postcommit_for_request()
    assert time.monotonic() - t0 < 0.1
    assert outcome["outcome"] == "aborted_for_foreground"
    assert outcome["waited"] is False
    assert record.abort_event.is_set()


def test_invalid_env_falls_back_to_default_window(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", raising=False)
    monkeypatch.setenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", "not-a-number")
    session = _session()
    future: Future = Future()
    _running_record(session, future)
    finisher = Thread(target=lambda: (time.sleep(0.05), future.set_result(None)))
    finisher.start()
    try:
        outcome = session.resolve_pending_postcommit_for_request()
        assert outcome["outcome"] == "completed"
        assert outcome["timeout_s"] == 0.6, "invalid env must use the default"
    finally:
        finisher.join(timeout=2)


def test_already_complete_and_no_pending_paths_unchanged(monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", raising=False)
    monkeypatch.delenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", raising=False)
    session = _session()
    outcome = session.resolve_pending_postcommit_for_request()
    assert outcome["outcome"] == "no_pending" and outcome["waited"] is False
    done: Future = Future()
    done.set_result(None)
    session.set_pending_postcommit(done)
    outcome = session.resolve_pending_postcommit_for_request()
    assert outcome["outcome"] == "completed"
    assert outcome["waited"] is False
    assert session._pending_postcommit is None


def test_concurrent_newer_record_is_not_cleared(monkeypatch) -> None:
    """Identity-safe replacement: a newer same-session record set while the
    arrival wait runs must survive the resolver's clear."""
    monkeypatch.delenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", raising=False)
    monkeypatch.setenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", "0.25")
    session = _session()
    stale_future: Future = Future()  # will time out
    stale = _running_record(session, stale_future)
    newer_future: Future = Future()
    newer_holder: dict = {}

    def replace_mid_wait() -> None:
        time.sleep(0.05)
        newer_holder["record"] = session.set_pending_postcommit(
            newer_future, reason="tool_call_history_rewrite"
        )

    replacer = Thread(target=replace_mid_wait)
    replacer.start()
    try:
        outcome = session.resolve_pending_postcommit_for_request()
        assert outcome["outcome"] == "aborted_for_foreground"
        assert stale.abort_event.is_set()
        assert session._pending_postcommit is newer_holder["record"], (
            "the concurrent newer record must not be cleared"
        )
        assert not newer_holder["record"].abort_event.is_set()
    finally:
        replacer.join(timeout=2)


def test_legacy_wait_timeout_env_precedence_still_works(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", "0.2")
    monkeypatch.setenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", "5.0")
    session = _session()
    future: Future = Future()  # running forever
    _running_record(session, future)
    t0 = time.monotonic()
    outcome = session.resolve_pending_postcommit_for_request()
    elapsed = time.monotonic() - t0
    # Legacy path: the OLD bounded wait governs (0.2s), not the 5s window.
    assert outcome["timeout_s"] == 0.2
    assert outcome["outcome"] == "timeout"
    assert elapsed < 1.0
