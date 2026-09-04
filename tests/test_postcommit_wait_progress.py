"""The bounded postcommit wait keeps waiting for a job that is still prefilling.

2026-09-03 founder session (Flash-Next, OpenCode, 43k context): the 25,644
token xhigh turn's postcommit was ~75% through re-prefilling the assistant
turn when the 30 s bound expired; the waiter aborted it and the request then
re-prefilled the same 25,647 tokens from scratch -- 30 s waited + 38 s
prefill for a state the engine held in memory 120 ms earlier. The
postcommit's remaining work is the request's own alternative, so the wait
now extends while the job heartbeats (every chunk boundary stamps the
record) and abandons only a job that never started or has gone silent for
the stall window.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from threading import Thread

import pytest

from mtplx.engine_session import (
    EngineSession,
    PendingPostcommit,
    _wait_while_progressing,
)


def _session() -> EngineSession:
    return EngineSession("sess-progress")


def test_a_heartbeating_job_is_waited_for_past_the_bound(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_POSTCOMMIT_WAIT_STALL_S", "1.0")
    session = _session()
    future: Future = Future()
    record = session.set_pending_postcommit(future)
    record.mark_started()

    def job() -> None:
        # Three "chunks" of 0.15 s, each stamping progress, landing at ~0.45 s:
        # the first boundary lands inside the 0.25 s bound, the rest past it
        # but inside the stall window.
        for _ in range(3):
            time.sleep(0.15)
            record.note_progress()
        future.set_result("committed")

    Thread(target=job, daemon=True).start()
    t0 = time.monotonic()
    outcome = session.wait_for_pending_postcommit(timeout_s=0.25)
    elapsed = time.monotonic() - t0

    assert outcome["outcome"] == "completed", outcome
    assert outcome["waited"] is True
    assert 0.4 < elapsed < 2.0
    assert outcome["progress_extended_s"] > 0.1
    assert record.progress_ticks == 3
    assert session.pending_postcommit is None


def test_a_silent_job_is_still_abandoned_at_the_bound(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_POSTCOMMIT_WAIT_STALL_S", "0.05")
    session = _session()
    future: Future = Future()  # never resolves, never heartbeats
    record = session.set_pending_postcommit(future)
    record.mark_started()
    time.sleep(0.08)  # the start stamp is older than the stall window

    t0 = time.monotonic()
    outcome = session.wait_for_pending_postcommit(timeout_s=0.1)
    elapsed = time.monotonic() - t0

    assert outcome["outcome"] == "timeout"
    assert outcome["abort_requested"] is True
    assert elapsed < 0.6
    assert "progress_extended_s" not in outcome


def test_a_job_that_never_started_is_not_extended() -> None:
    session = _session()
    future: Future = Future()
    record = session.set_pending_postcommit(future)
    assert record.started_at_s is None

    t0 = time.monotonic()
    outcome = session.wait_for_pending_postcommit(timeout_s=0.1)
    assert outcome["outcome"] == "timeout"
    assert time.monotonic() - t0 < 0.6


def test_stall_window_zero_restores_the_plain_bounded_wait(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_POSTCOMMIT_WAIT_STALL_S", "0")
    future: Future = Future()
    record = PendingPostcommit(future=future)
    record.mark_started()
    record.note_progress()
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        _wait_while_progressing(future, record, timeout_s=0.05)
    assert time.monotonic() - t0 < 0.5


def test_the_ceiling_caps_a_job_that_heartbeats_forever(monkeypatch) -> None:
    monkeypatch.setenv("MTPLX_POSTCOMMIT_WAIT_STALL_S", "0.2")
    monkeypatch.setenv("MTPLX_POSTCOMMIT_WAIT_CEILING_S", "0.4")
    future: Future = Future()
    record = PendingPostcommit(future=future)
    record.mark_started()
    stop = False

    def heartbeat() -> None:
        while not stop:
            record.note_progress()
            time.sleep(0.02)

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match="ceiling"):
            _wait_while_progressing(future, record, timeout_s=0.05)
        assert 0.3 < time.monotonic() - t0 < 1.5
    finally:
        stop = True
        thread.join(timeout=1.0)


def test_progressing_reads_the_heartbeat() -> None:
    record = PendingPostcommit(future=Future())
    assert not record.progressing(stall_s=1.0)
    record.mark_started()
    # Starting is not progress: the first chunk boundary must land inside
    # the caller's bound for the extension to apply at all.
    assert not record.progressing(stall_s=1.0)
    record.note_progress()
    assert record.progressing(stall_s=1.0)
    record.last_progress_mono_s = time.monotonic() - 5.0
    assert not record.progressing(stall_s=1.0)
    record.note_progress()
    assert record.progressing(stall_s=1.0)


def test_tool_call_finish_commits_the_generation_boundary() -> None:
    """2026-09-03: a complete tool-call turn is a natural stop; refusing it
    left agent sessions without a committed stream, so the committed-think /
    committed-body canonicalization never applied to tool rounds."""
    session = EngineSession("sess-toolcalls")
    result = session.commit(
        prompt_ids=[1, 2, 3], generated_ids=[4, 5], finish_reason="tool_calls"
    )
    assert result.committed is True, result
    assert tuple(session.committed_token_ids) == (1, 2, 3, 4, 5)
    refused = session.commit(
        prompt_ids=[1, 2, 3], generated_ids=[9], finish_reason="cancelled"
    )
    assert refused.committed is False
    assert refused.reason.startswith("unsafe_finish")
    assert tuple(session.committed_token_ids) == (1, 2, 3, 4, 5)
