"""The PLE lookahead's inertness verdicts are receipts when serving.

2026-09-03, live app daemon: a 31k-token cold prompt whose first chunk the
sidecar declined paid its whole 30 s prefill and then reached the client as
``finish_reason: "error"`` -- ``verify_full_engagement`` raised a measurement
verdict inside a user request.  The same day, every warm session-bank turn of
a 43k-token OpenCode session sat 5-7.5 s in ``EarlyFirstGather.close()``
waiting for a first-chunk gather that no prefill was going to consume.

These tests pin the serving contract: verdicts are counted and readable, never
raised, unless the strict flag a benchmark arm exports is set; an unadopted
early gather never blocks the owner thread; and a request whose prompt a RAM
bank entry can already serve past the first chunk declines the early gather
before a single row is read.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from mtplx import generation
from mtplx import ple_prefill_lookahead as lookahead_mod
from mtplx import ple_row_gather as row_gather
from mtplx.ple_prefill_lookahead import (
    EarlyFirstGather,
    PrefillLookahead,
    prefill_lookahead_scope,
)
from mtplx.session_bank import SessionBank


@pytest.fixture(autouse=True)
def _serving_mode(monkeypatch):
    monkeypatch.delenv(lookahead_mod.STRICT_ENV_FLAG, raising=False)
    lookahead_mod.strict_enabled.cache_clear()
    lookahead_mod.reset_counters()
    lookahead_mod._WARNED.clear()
    yield
    lookahead_mod.reset_counters()
    lookahead_mod.strict_enabled.cache_clear()
    lookahead_mod._WARNED.clear()


class _Immediate:
    def __init__(self, fn, *args):
        self._value = fn(*args)

    def result(self, timeout=None):
        return self._value

    def cancel(self):
        return False

    def add_done_callback(self, fn):
        fn(self)


def _lookahead(total=64, chunk=16, prepare=None):
    ids = np.arange(total, dtype=np.int64)
    spans = [(s, min(total, s + chunk)) for s in range(0, total, chunk)]
    prepare = prepare or (lambda s, e: (ids[s:e].copy(), {}))
    return PrefillLookahead(
        ids, spans, prepare, submit=lambda fn, *a: _Immediate(fn, *a)
    ), ids


# --------------------------------------------------------------------------
# engagement verdicts
# --------------------------------------------------------------------------


def test_serving_records_an_unengaged_lane_instead_of_failing_the_request(caplog):
    look, _ids = _lookahead()
    with prefill_lookahead_scope(look):
        pass  # a prefill that never consumed a chunk: the 2026-09-01 shape
    counters = lookahead_mod.snapshot_counters()
    assert counters["verdict_engagement_incomplete"] == 1
    status = lookahead_mod.last_scope_status()
    assert status["armed"] is True
    unserved = status["engagement_incomplete"]["unserved"]
    assert (0, "never_taken") in unserved
    assert status["engagement_incomplete"]["engagement"]["hits"] == 0
    assert any("serving continues" in rec.getMessage() for rec in caplog.records)


def test_serving_names_the_declined_first_span_and_keeps_going():
    """The live 2026-09-03 failure: span 0 declined by the sidecar."""

    def prepare(start, end):
        if start == 0:
            return None
        return (np.arange(1000 + start, 1000 + end, dtype=np.int64), {})

    look, ids = _lookahead(total=64, chunk=16, prepare=prepare)
    with prefill_lookahead_scope(look):
        for index in range(4):
            look.take(index)
            look.submit(index + 1)
    assert lookahead_mod.snapshot_counters()["verdict_engagement_incomplete"] == 1
    assert (0, "ineligible") in lookahead_mod.last_scope_status()[
        "engagement_incomplete"
    ]["unserved"]


def test_the_verdict_warns_once_per_process_not_per_request(caplog):
    for _ in range(3):
        look, _ids = _lookahead()
        with prefill_lookahead_scope(look):
            pass
    assert lookahead_mod.snapshot_counters()["verdict_engagement_incomplete"] == 3
    warnings = [r for r in caplog.records if "did not engage" in r.getMessage()]
    assert len(warnings) == 1


def test_strict_mode_still_raises_the_measurement_verdict(monkeypatch):
    monkeypatch.setenv(lookahead_mod.STRICT_ENV_FLAG, "1")
    lookahead_mod.strict_enabled.cache_clear()
    look, _ids = _lookahead()
    with pytest.raises(RuntimeError, match="did not engage"):
        with prefill_lookahead_scope(look):
            pass


def test_unwired_loop_is_a_verdict_when_serving(monkeypatch):
    monkeypatch.setenv(lookahead_mod.ENV_FLAG, "1")
    lookahead_mod.enabled.cache_clear()
    try:
        assert lookahead_mod.reject_unwired_prefill_loop("_prefill") is None
        assert lookahead_mod.snapshot_counters()["verdict_unwired_prefill_loop"] == 1
        monkeypatch.setenv(lookahead_mod.STRICT_ENV_FLAG, "1")
        lookahead_mod.strict_enabled.cache_clear()
        with pytest.raises(RuntimeError, match="not wired"):
            lookahead_mod.reject_unwired_prefill_loop("_prefill")
    finally:
        lookahead_mod.enabled.cache_clear()


def test_strict_flag_rejects_garbage(monkeypatch):
    monkeypatch.setenv(lookahead_mod.STRICT_ENV_FLAG, "sometimes")
    lookahead_mod.strict_enabled.cache_clear()
    with pytest.raises(ValueError):
        lookahead_mod.strict_enabled()


# --------------------------------------------------------------------------
# the early first-chunk gather never blocks the owner thread
# --------------------------------------------------------------------------


def test_closing_an_unadopted_early_gather_does_not_wait_for_it():
    """A running, unconsumed gather is orphaned, not awaited."""

    release = threading.Event()
    started = threading.Event()

    def slow_prepare(ids, a, b, record):
        started.set()
        release.wait(timeout=10)
        return ("rows", {})

    pool = __import__("concurrent.futures").futures.ThreadPoolExecutor(1)
    try:
        early = EarlyFirstGather(
            np.arange(600, dtype=np.int64), (0, 256), slow_prepare, submit=pool.submit
        )
        assert started.wait(timeout=5)
        t0 = __import__("time").perf_counter()
        early.close()
        assert __import__("time").perf_counter() - t0 < 1.0
        assert early.outcome == "never_needed"
        release.set()
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_a_closed_early_gather_never_queues_the_rest_of_prompt_warm():
    release = threading.Event()
    calls: list[str] = []

    def slow_prepare(ids, a, b, record):
        release.wait(timeout=10)
        calls.append("span0")
        return ("rows", {})

    def prefetch_rest(ids, a, record):
        calls.append("rest")
        return 0

    pool = __import__("concurrent.futures").futures.ThreadPoolExecutor(1)
    try:
        early = EarlyFirstGather(
            np.arange(600, dtype=np.int64),
            (0, 256),
            slow_prepare,
            submit=pool.submit,
            prefetch_rest=prefetch_rest,
        )
        early.close()
        release.set()
        pool.shutdown(wait=True)
    finally:
        release.set()
    assert calls == ["span0"], calls


# --------------------------------------------------------------------------
# bank-aware arming: a warm turn declines the gather before reading a row
# --------------------------------------------------------------------------


def test_shares_ram_prefix_is_a_slice_compare_over_ram_entries():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    entry = SimpleNamespace()
    bank._entries[tuple(range(1000))] = entry
    assert bank.shares_ram_prefix(list(range(1200)), min_tokens=512)
    assert not bank.shares_ram_prefix([7] + list(range(1, 1200)), min_tokens=512)
    assert not bank.shares_ram_prefix(list(range(300)), min_tokens=512)
    assert not SessionBank(
        max_entries=4, max_bytes=1024, per_session_max_bytes=512
    ).shares_ram_prefix(list(range(1200)), min_tokens=512)


def test_bank_preemption_probe_uses_the_block_restore_minimum(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", raising=False)
    seen: list[int] = []

    class _Bank:
        def shares_ram_prefix(self, ids, *, min_tokens):
            seen.append(min_tokens)
            return True

    assert generation._bank_may_preempt_first_span(_Bank(), list(range(4000)), (0, 2048))
    assert seen == [512]
    assert not generation._bank_may_preempt_first_span(None, list(range(4000)), (0, 2048))
    assert not generation._bank_may_preempt_first_span(object(), list(range(4000)), (0, 2048))


def test_warm_turn_declines_the_early_gather_before_any_row_is_read(monkeypatch):
    monkeypatch.setenv(lookahead_mod.EARLY_ENV_FLAG, "1")
    row_gather.enabled.cache_clear()
    lookahead_mod.early_enabled.cache_clear()
    monkeypatch.setattr(generation, "_sustained_prefill_enabled", lambda: True)
    hook_calls: list[tuple] = []

    class _Model:
        def ple_first_gather_early(self, token_ids, span):
            hook_calls.append((len(token_ids), span))
            return None

    rt = SimpleNamespace(model=_Model())
    prompt = list(range(6000))
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    bank._entries[tuple(range(5000))] = SimpleNamespace()
    try:
        with generation._ple_first_gather_early_scope(
            rt, prompt, session_bank=bank
        ) as early:
            assert early is None
        assert hook_calls == []
        assert lookahead_mod.last_early_status()["reason"] == (
            "bank_prefix_may_serve_first_span"
        )
        # A genuinely cold prompt still arms the gather.
        cold = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
        with generation._ple_first_gather_early_scope(rt, prompt, session_bank=cold):
            pass
        assert hook_calls and hook_calls[0][1] == (0, 2048)
    finally:
        row_gather.enabled.cache_clear()
        lookahead_mod.early_enabled.cache_clear()
