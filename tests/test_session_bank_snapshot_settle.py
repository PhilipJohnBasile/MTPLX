"""Idle-lane snapshot settling (MTPLX_SESSION_SNAPSHOT_SETTLE, opt-in).

The lazy put's snapshot views hold references to live cache buffers; until
owner copies replace them, buffer donation is blocked and the next turn's
first writes pay full COW divergence copies. The settle dispatches an
owner-copy job to the model-owner idle lane at put time. DEFAULT OFF:
the 2026-08-30 phase-3 A/B falsified it as a default (stalls are
dominated by idle-lane scheduling, and the extra copy job worsened the
worst case) — these tests pin the opt-in mechanics.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SessionBank, SessionBankEntry


@pytest.fixture(autouse=True)
def _settle_opt_in(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_SNAPSHOT_SETTLE", "1")


def _entry_with_lazy_view() -> SessionBankEntry:
    base = mx.ones((8, 8))
    mx.eval(base)
    view = base[...]  # lazy view: holds a reference to base's buffer
    return SessionBankEntry(
        token_ids=(1, 2),
        token_hash="settle-hash",
        model_path="models/example",
        mtp_enabled=True,
        hidden_variant=None,
        cache_snapshot=CacheSnapshot(states=((view,),), meta_states=()),
        logits=None,
        hidden=None,
    )


def test_settle_job_dispatches_with_session_key_and_stamps_entry():
    bank = SessionBank(max_entries=4, max_bytes=1 << 20, per_session_max_bytes=1 << 20)
    dispatched: list = []
    bank.cold_enqueue_dispatch = dispatched.append
    entry = _entry_with_lazy_view()
    view_before = entry.cache_snapshot.states[0][0]

    bank._schedule_snapshot_settle(entry)

    assert len(dispatched) == 1
    assert dispatched[0].coalesce_key == "snapshot_settle:hash:settle-hash"
    assert entry.snapshot_settled_at is None
    dispatched[0]()
    assert entry.snapshot_settled_at is not None
    # The settle must REBIND the snapshot leaves onto owner copies —
    # evaluating a full-range lazy view merely aliases the live buffer
    # and leaves the donation-blocking reference alive.
    leaf_after = entry.cache_snapshot.states[0][0]
    assert leaf_after is not view_before
    assert mx.array_equal(leaf_after, view_before).item()


def test_put_dispatches_settle_before_cold_encode():
    bank = SessionBank(max_entries=4, max_bytes=1 << 20, per_session_max_bytes=1 << 20)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    dispatched: list = []
    bank.cold_enqueue_dispatch = dispatched.append
    timing: dict = {}

    entry = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-settle",
        timing_out=timing,
    )

    assert entry is not None
    assert timing["snapshot_settle"] == {"dispatched": True}
    keys = [getattr(job, "coalesce_key", "") for job in dispatched]
    assert keys and keys[0] == "snapshot_settle:session-settle"
    dispatched[0]()
    assert entry.snapshot_settled_at is not None


def test_settle_env_off_switch(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_SNAPSHOT_SETTLE", "0")
    bank = SessionBank(max_entries=4, max_bytes=1 << 20, per_session_max_bytes=1 << 20)
    dispatched: list = []
    bank.cold_enqueue_dispatch = dispatched.append

    bank._schedule_snapshot_settle(_entry_with_lazy_view())

    assert dispatched == []


def test_live_ref_entries_never_settle():
    bank = SessionBank(max_entries=4, max_bytes=1 << 20, per_session_max_bytes=1 << 20)
    dispatched: list = []
    bank.cold_enqueue_dispatch = dispatched.append
    entry = _entry_with_lazy_view()
    entry.live_ref_only = True

    bank._schedule_snapshot_settle(entry)

    assert dispatched == []


def test_no_dispatch_lane_keeps_lazy_contract():
    bank = SessionBank(max_entries=4, max_bytes=1 << 20, per_session_max_bytes=1 << 20)
    bank.cold_enqueue_dispatch = None
    entry = _entry_with_lazy_view()
    timing: dict = {}

    bank._schedule_snapshot_settle(entry, timing_out=timing)

    assert timing["snapshot_settle"] == {"dispatched": False}
    assert entry.snapshot_settled_at is None
