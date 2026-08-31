"""#415: pre-prefill admission shed.

The shipped failure: a Pi agent auto-compacted at 248k, the compaction
rewrote the whole prefix, and the 38k replacement prefill — a guaranteed
cache miss — started while the superseded 6.09 GiB SessionBank snapshot
of the pre-compaction transcript stayed resident on a memory plan that
admits 262K with zero headroom. The footprint crossed the Metal cap
mid-prefill and the sustained-pressure guard killed the request with a
structured 507 ~30 s later.

These tests pin the admission guard that sheds BEFORE the prefill:
superseded same-session entries first, then LRU idle entries with active
sessions protected, then the allocator cache — and stays perfectly inert
when memory is healthy.
"""

from __future__ import annotations

from types import SimpleNamespace

import mtplx.server.openai as srv

GIB = 1024**3
LIMIT = 96 * GIB
KV_PER_TOKEN = 24576
AUX_PER_TOKEN = 7872


class _Entry:
    def __init__(self, token_ids, session_id, nbytes):
        self.token_ids = tuple(int(t) for t in token_ids)
        self.session_id = session_id
        self.nbytes = int(nbytes)


class _Bank:
    def __init__(self, entries):
        self.entries = list(entries)
        self.cleared_sessions: list[str | None] = []
        self.shrink_calls: list[tuple[int, str, bool]] = []
        self.touched: list[str] = []
        self.probe_calls = 0

    @property
    def total_nbytes(self):
        return sum(entry.nbytes for entry in self.entries)

    def longest_prefix(self, token_ids):
        self.probe_calls += 1
        tokens = tuple(int(t) for t in token_ids)
        best = None
        for entry in self.entries:
            prefix = entry.token_ids
            if len(prefix) > len(tokens) or tokens[: len(prefix)] != prefix:
                continue
            if best is None or len(prefix) > len(best.token_ids):
                best = entry
        return best

    def clear(self, *, session_id=None):
        victims = [e for e in self.entries if e.session_id == session_id]
        self.entries = [e for e in self.entries if e.session_id != session_id]
        self.cleared_sessions.append(session_id)
        return len(victims)

    def touch_sessions(self, session_ids):
        self.touched.extend(session_ids)

    def shrink_to_bytes(self, target_bytes, *, reason="", protect_active=False):
        self.shrink_calls.append((int(target_bytes), reason, protect_active))
        evicted = 0
        while self.entries and self.total_nbytes > int(target_bytes):
            self.entries.pop(0)
            evicted += 1
        return evicted


def _state():
    return SimpleNamespace(
        metal_memory_caps={"memory_limit_bytes": LIMIT},
        memory_plan=SimpleNamespace(
            kv_bytes_per_token_effective=KV_PER_TOKEN,
            aux_bytes_per_token=AUX_PER_TOKEN,
            prefill_transient_bytes_per_token=0,
        ),
        dashboard=SimpleNamespace(),
    )


def _pin_live_stats(monkeypatch, *, active, cache):
    monkeypatch.setattr(
        srv,
        "_mlx_memory_stats_live",
        lambda: {
            "ok": True,
            "active_memory_bytes": int(active),
            "cache_memory_bytes": int(cache),
        },
    )


def _shed(state, prompt_ids, bank, session_id):
    return srv._prefill_admission_shed(
        state,
        prompt_ids=prompt_ids,
        session_bank=bank,
        session_id=session_id,
    )


class TestInertWhenHealthy:
    def test_healthy_memory_is_a_no_op_without_probing(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=60 * GIB, cache=2 * GIB)
        bank = _Bank([_Entry(range(100), "pi", 6 * GIB)])
        assert _shed(_state(), list(range(40_000)), bank, "pi") is None
        assert bank.probe_calls == 0
        assert bank.cleared_sessions == []
        assert bank.shrink_calls == []

    def test_short_prompt_never_triggers(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        bank = _Bank([_Entry(range(100), "pi", 6 * GIB)])
        assert _shed(_state(), list(range(1024)), bank, "pi") is None

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv("MTPLX_PREFILL_ADMISSION_SHED", "0")
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        bank = _Bank([_Entry(range(100), "pi", 6 * GIB)])
        assert _shed(_state(), list(range(40_000)), bank, "pi") is None

    def test_no_metal_caps_is_inert(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        state = _state()
        state.metal_memory_caps = None
        assert _shed(state, list(range(40_000)), _Bank([]), "pi") is None


class TestIncidentShape:
    """The #415 timeline: cache-miss prefill + superseded resident snapshot."""

    def test_superseded_session_snapshot_released_first(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=93 * GIB, cache=2 * GIB)
        # The compaction rewrote the prefix: the session's banked snapshot
        # (prefix 900..999) can never match the new prompt (0..39999).
        superseded = _Entry(range(900, 1000), "pi", 6 * GIB)
        stranger = _Entry(range(500, 600), "other", 2 * GIB)
        bank = _Bank([stranger, superseded])
        receipt = _shed(_state(), list(range(40_000)), bank, "pi")
        assert receipt is not None
        assert receipt["action"] == "prefill_admission_shed"
        assert receipt["reusable_prefix_tokens"] == 0
        assert receipt["miss_tokens"] == 40_000
        assert receipt["superseded_session_entries_evicted"] == 1
        assert bank.cleared_sessions == ["pi"]
        assert receipt["cache_cleared"] is True

    def test_guard_event_recorded(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        state = _state()
        bank = _Bank([_Entry(range(900, 1000), "pi", 6 * GIB)])
        receipt = _shed(state, list(range(40_000)), bank, "pi")
        assert receipt is not None
        events = list(getattr(state.dashboard, "memory_guard_events", []))
        assert any(
            event.get("action") == "prefill_admission_shed" for event in events
        )

    def test_lru_shrink_protects_active_and_runs_after_superseded(
        self, monkeypatch
    ):
        # Deficit larger than the superseded snapshot alone: the LRU pass
        # must run with protect_active=True.
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=3 * GIB)
        superseded = _Entry(range(900, 1000), "pi", 1 * GIB)
        strangers = [
            _Entry(range(50_000 + i * 100, 50_000 + i * 100 + 50), f"idle-{i}", GIB)
            for i in range(4)
        ]
        bank = _Bank([*strangers, superseded])
        receipt = _shed(_state(), list(range(40_000)), bank, "pi")
        assert receipt is not None
        assert receipt["superseded_session_entries_evicted"] == 1
        assert len(bank.shrink_calls) == 1
        _target, reason, protect_active = bank.shrink_calls[0]
        assert reason == "prefill_admission"
        assert protect_active is True

    def test_reusable_prefix_is_pinned_not_cleared(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        prompt = list(range(40_000))
        # The session HAS a restorable prefix: entry matches prompt[:8192].
        restorable = _Entry(prompt[:8192], "pi", 3 * GIB)
        bank = _Bank([restorable])
        receipt = _shed(_state(), prompt, bank, "pi")
        assert receipt is not None
        assert receipt["reusable_prefix_tokens"] == 8192
        assert bank.cleared_sessions == []
        assert bank.touched == ["pi"]
        assert "superseded_session_entries_evicted" not in receipt

    def test_no_bank_still_sheds_allocator_cache(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)
        receipt = _shed(_state(), list(range(40_000)), None, None)
        assert receipt is not None
        assert receipt["cache_cleared"] is True
        assert "bank_bytes_before" not in receipt

    def test_never_raises(self, monkeypatch):
        _pin_live_stats(monkeypatch, active=95 * GIB, cache=1 * GIB)

        class _ExplodingBank(_Bank):
            def longest_prefix(self, token_ids):  # noqa: ARG002
                raise RuntimeError("bank exploded")

            @property
            def total_nbytes(self):
                raise RuntimeError("bank exploded")

        receipt = _shed(
            _state(), list(range(40_000)), _ExplodingBank([]), "pi"
        )
        # Probe failure degrades to miss==prompt; bank steps report the
        # error but the shed still completes with the cache clear.
        assert receipt is not None
        assert receipt["cache_cleared"] is True
        assert "bank_error" in receipt
