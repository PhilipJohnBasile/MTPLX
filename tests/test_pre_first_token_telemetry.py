"""Request-local timing telemetry for the pre-first-token attribution lane.

Covers the SessionBank.put timing paths (cold disabled / deferred /
synchronous / dispatch-error fallback / oversized early return), the
caller-owned timing_out contract (never shared bank state), the Site A
stored-must-reflect-put's-return rule, and envelope durability — the
2026-08-06 audit found session_prompt_prefix_bank_commit was measured in
GenerationStats but never reached the request-log JSONL, which is the only
trail for footerless agent clients like pi.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mtplx.generation import _prefill_store_result
from mtplx.server.openai import _json_safe, _metrics_envelope
from mtplx.session_bank import SessionBank


def _bank(**kwargs) -> SessionBank:
    return SessionBank(
        max_entries=4,
        max_bytes=1 << 20,
        per_session_max_bytes=1 << 20,
        **kwargs,
    )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)


def _put(bank: SessionBank, timing: dict | None):
    return bank.put(
        runtime=_runtime(),
        token_ids=[1, 2, 3],
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        timing_out=timing,
    )


def test_put_timing_cold_disabled_records_phases_and_skip_reason():
    timing: dict = {}
    entry = _put(_bank(), timing)
    assert entry is not None
    assert timing["trunk_snapshot_s"] >= 0.0
    assert timing["entry_build_s"] >= 0.0
    assert timing["cold_enqueue"] == {
        "enabled": False,
        "skip_reason": "no_cold_tier",
    }


def test_put_timing_deferred_dispatch_never_charges_serialization():
    stored: list = []
    dispatched: list = []
    cold = SimpleNamespace(
        put_entry=lambda entry, capabilities=None: stored.append(entry)
    )
    bank = _bank(cold_tier=cold)
    bank.cold_enqueue_dispatch = dispatched.append
    timing: dict = {}
    entry = _put(bank, timing)
    assert entry is not None
    cold_t = timing["cold_enqueue"]
    assert cold_t["enabled"] is True
    assert cold_t["deferred"] is True
    assert cold_t["dispatch_elapsed_s"] >= 0.0
    # The request path must never be charged for deferred serialization.
    assert "synchronous_serialize_elapsed_s" not in cold_t
    assert stored == []
    # The idle lane runs the job later; the entry still reaches the tier.
    dispatched[0]()
    assert len(stored) == 1


def test_put_timing_synchronous_fallback_records_serialize():
    stored: list = []
    cold = SimpleNamespace(
        put_entry=lambda entry, capabilities=None: stored.append(entry)
    )
    bank = _bank(cold_tier=cold)  # no dispatch wired -> synchronous path
    timing: dict = {}
    entry = _put(bank, timing)
    assert entry is not None
    assert len(stored) == 1
    cold_t = timing["cold_enqueue"]
    assert cold_t["enabled"] is True
    assert cold_t["deferred"] is False
    assert cold_t["synchronous_serialize_elapsed_s"] >= 0.0
    assert "dispatch_elapsed_s" not in cold_t


def test_put_timing_dispatch_error_falls_back_synchronously():
    stored: list = []
    cold = SimpleNamespace(
        put_entry=lambda entry, capabilities=None: stored.append(entry)
    )
    bank = _bank(cold_tier=cold)

    def bad_dispatch(job):
        raise RuntimeError("scheduler gone")

    bank.cold_enqueue_dispatch = bad_dispatch
    timing: dict = {}
    entry = _put(bank, timing)
    assert entry is not None
    assert len(stored) == 1
    cold_t = timing["cold_enqueue"]
    assert cold_t["enabled"] is True
    assert cold_t["dispatch_error"] is True
    assert cold_t["dispatch_elapsed_s"] >= 0.0
    assert cold_t["deferred"] is False
    assert cold_t["synchronous_serialize_elapsed_s"] >= 0.0
    assert bank.eviction_log[-1]["reason"] == "ssd_enqueue_dispatch_error"


def test_put_timing_oversized_early_return_leaves_keys_absent():
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    timing: dict = {}
    entry = bank.put(
        runtime=_runtime(),
        token_ids=[1, 2, 3],
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
        nbytes_override=2048,
        timing_out=timing,
    )
    assert entry is None
    # The oversized-override guard returns before the snapshot/build/enqueue
    # phases, so no timing keys may pretend those phases ran.
    assert timing == {}


def test_put_timing_not_requested_changes_nothing():
    entry = _put(_bank(), None)
    assert entry is not None


def test_prefill_store_result_cannot_report_a_none_put_as_stored():
    record = _prefill_store_result(
        None,
        suffix_tokens=2048,
        elapsed_s=0.5,
        mtp_snapshot_elapsed_s=0.1,
        put_elapsed_s=0.4,
        put_timing={},
    )
    assert record["stored"] is False
    assert record["reason"] == "sessionbank_snapshot_skipped"
    stored = _prefill_store_result(
        object(),
        suffix_tokens=2048,
        elapsed_s=0.5,
        mtp_snapshot_elapsed_s=0.1,
        put_elapsed_s=0.4,
        put_timing={"trunk_snapshot_s": 0.2},
    )
    assert stored["stored"] is True
    assert stored["reason"] == "committed_prefill_prefix"
    assert stored["put_timing"] == {"trunk_snapshot_s": 0.2}


def _envelope(stats: dict) -> dict:
    return _metrics_envelope(
        stats=stats,
        prompt_tokens=10,
        completion_tokens=5,
        request_elapsed_s=4.0,
        token_times=[],
        request_started_s=0.0,
        lock_wait_time_s=0.0,
        session_id="session-1",
        session_cache_hit=True,
        cache_miss_reason=None,
        session_restore_mode="near_prefix_clone",
        mtp_depth=3,
        generation_limits={},
    )


def test_metrics_envelope_carries_pre_first_token_fields():
    envelope = _envelope(
        {
            "session_prompt_prefix_bank_commit": {
                "stored": True,
                "elapsed_s": 1.23,
            },
            "session_prefill_store": {"stored": False, "skip_reason": "min_suffix"},
            "pre_first_token_setup_s": 1.5,
            "prompt_eval_time_s": 2.0,
        }
    )
    assert envelope["session_prompt_prefix_bank_commit"]["elapsed_s"] == 1.23
    assert envelope["session_prefill_store"] == {
        "stored": False,
        "skip_reason": "min_suffix",
    }
    assert envelope["pre_first_token_setup_s"] == 1.5
    # Durability: the request-log writer json.dumps(_json_safe(record)); the
    # fields must survive that round trip intact.
    parsed = json.loads(json.dumps(_json_safe(envelope), default=str))
    assert parsed["session_prompt_prefix_bank_commit"]["stored"] is True
    assert parsed["session_prefill_store"]["skip_reason"] == "min_suffix"
    assert parsed["pre_first_token_setup_s"] == 1.5


def test_metrics_envelope_defaults_empty_without_new_stats():
    envelope = _envelope({"prompt_eval_time_s": 2.0})
    assert envelope["session_prompt_prefix_bank_commit"] == {}
    assert envelope["session_prefill_store"] == {}
    assert envelope["pre_first_token_setup_s"] == 0.0
