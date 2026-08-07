"""Passive-probe telemetry: per-entry cold-encode completion, served-entry
truth from restore_entry_prefix_cache, feature-detected served_out, and
request-envelope carriage.

The probe observes the lazy graph without perturbing it: no new mx.eval
sites, no metric redefinition. Completion state lives on the exact
SessionBankEntry object — Site A and Site B can create distinct entries
with the SAME token hash, and an old entry finishing its encode must never
report a newer lazy replacement as settled. The served_out kwarg is
feature-detected once, never TypeError-retried: a partially executed
restore (e.g. a consumed live lease) must not run twice.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mtplx.generation import _accepts_served_out
from mtplx.server.openai import _json_safe, _metrics_envelope
from mtplx.session_bank import SessionBank


def _bank(**kwargs) -> SessionBank:
    return SessionBank(
        max_entries=8,
        max_bytes=1 << 20,
        per_session_max_bytes=1 << 20,
        **kwargs,
    )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)


def _put(bank: SessionBank, token_ids: list[int]):
    return bank.put(
        runtime=_runtime(),
        token_ids=token_ids,
        cache=[],
        logits=None,
        hidden=None,
        session_id="session-1",
    )


def test_cold_encode_completion_set_on_entry_after_synchronous_store():
    stored: list = []
    cold = SimpleNamespace(
        put_entry=lambda entry, capabilities=None: stored.append(entry) or True
    )
    bank = _bank(cold_tier=cold)
    entry = _put(bank, [1, 2, 3])
    assert entry is not None and len(stored) == 1
    assert entry.cold_encode_completed_at is not None
    assert entry.cold_encode_completed_at > 0.0


def test_cold_encode_completion_deferred_until_job_runs():
    stored: list = []
    dispatched: list = []
    cold = SimpleNamespace(
        put_entry=lambda entry, capabilities=None: stored.append(entry) or True
    )
    bank = _bank(cold_tier=cold)
    bank.cold_enqueue_dispatch = dispatched.append
    entry = _put(bank, [1, 2, 3])
    assert entry is not None
    assert entry.cold_encode_completed_at is None
    dispatched[0]()
    assert entry.cold_encode_completed_at is not None


def test_cold_encode_completion_not_set_when_tier_declines():
    cold = SimpleNamespace(put_entry=lambda entry, capabilities=None: False)
    bank = _bank(cold_tier=cold)
    entry = _put(bank, [1, 2, 3])
    assert entry is not None
    assert entry.cold_encode_completed_at is None


def test_same_token_hash_replacement_never_reports_new_entry_settled():
    """The blocker scenario: entry A enqueues, is replaced by entry B with
    the SAME token prefix/hash, then A's encode completes late. B must stay
    unsettled; only A's object carries the completion."""
    stored: list = []
    dispatched: list = []
    cold = SimpleNamespace(
        put_entry=lambda entry, capabilities=None: stored.append(entry) or True
    )
    bank = _bank(cold_tier=cold)
    bank.cold_enqueue_dispatch = dispatched.append
    entry_a = _put(bank, [1, 2, 3])
    entry_b = _put(bank, [1, 2, 3])  # replaces A under the same key
    assert entry_a is not None and entry_b is not None
    assert entry_a is not entry_b
    assert entry_a.token_hash == entry_b.token_hash
    # A's encode job (enqueued first) completes AFTER B was installed.
    dispatched[0]()
    assert entry_a.cold_encode_completed_at is not None
    assert entry_b.cold_encode_completed_at is None
    # B settles only when B's own job runs.
    dispatched[1]()
    assert entry_b.cold_encode_completed_at is not None


def test_accepts_served_out_detection_truth_table():
    def with_kwarg(a, *, served_out=None):
        return a

    def without_kwarg(a):
        return a

    assert _accepts_served_out(with_kwarg) is True
    assert _accepts_served_out(without_kwarg) is False
    # Signature-less callables (builtins) read as unsupported, never retried.
    assert _accepts_served_out(len) is False


def test_internal_typeerror_executes_once_and_propagates():
    """Contract pinned by the call pattern: feature-detect, then ONE call.
    An internal TypeError from a served_out-accepting bank must propagate
    after exactly one execution — a retry could re-consume a live lease."""
    calls: list = []

    def restore_like(rt, entry, matched, *, mode, cache_factory, served_out=None):
        calls.append(mode)
        raise TypeError("internal failure after side effect")

    supports = _accepts_served_out(restore_like)
    assert supports is True
    kwargs = {"served_out": {}} if supports else {}
    try:
        restore_like(None, None, 4, mode="clone", cache_factory=list, **kwargs)
        raise AssertionError("expected TypeError")
    except TypeError:
        pass
    assert calls == ["clone"]


def test_restore_entry_prefix_cache_fills_served_out():
    bank = _bank()
    entry = _put(bank, [1, 2, 3, 4, 5, 6])
    assert entry is not None
    served: dict = {}
    result = bank.restore_entry_prefix_cache(
        _runtime(),
        entry,
        4,
        mode="clone",
        cache_factory=list,
        served_out=served,
    )
    assert result is not None
    cache, mtp_cache, mode, restore_point, boundary_hidden = result
    assert mode == "clone" and boundary_hidden is None
    assert served["restore_point"] == restore_point == 4
    assert served["boundary_used"] is False
    assert served["mode"] == "clone"
    maintenance = served["maintenance"]
    assert maintenance["factory_s"] >= 0.0
    assert maintenance["install_s"] >= 0.0
    assert maintenance["trim_s"] >= 0.0


def test_restore_entry_prefix_cache_backward_compatible_without_served_out():
    bank = _bank()
    entry = _put(bank, [1, 2, 3, 4, 5, 6])
    result = bank.restore_entry_prefix_cache(
        _runtime(),
        entry,
        4,
        mode="clone",
        cache_factory=list,
    )
    assert result is not None
    assert len(result) == 5


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
        session_restore_mode="block_prefix_boundary_clone",
        mtp_depth=3,
        generation_limits={},
    )


def test_metrics_envelope_carries_probe_fields():
    served = {
        "entry_prefix_len": 15578,
        "entry_token_hash": "a548ccf7503078fc",
        "requested_matched": 15576,
        "actual_restore_point": 15360,
        "boundary_restore": True,
        "candidate_index": 2,
        "encode_completed": False,
        "bank": {"maintenance": {"install_s": 0.001}},
    }
    envelope = _envelope(
        {
            "session_restore_served": served,
            "prompt_state_total_time_s": 2.5,
            "prompt_state_unattributed_time_s": 0.8,
            "first_primary_sample_time_s": 0.9,
            "first_round": {"wall_s": 0.95, "verify_calls": 1, "single_cycle": True},
            "prompt_eval_time_s": 1.3,
        }
    )
    assert envelope["session_restore_served"]["actual_restore_point"] == 15360
    assert envelope["session_restore_served"]["candidate_index"] == 2
    assert envelope["prompt_state_total_time_s"] == 2.5
    assert envelope["prompt_state_unattributed_time_s"] == 0.8
    assert envelope["first_primary_sample_time_s"] == 0.9
    assert envelope["first_round"]["verify_calls"] == 1
    assert envelope["first_round"]["single_cycle"] is True
    # Durability through the request-log writer's round trip.
    parsed = json.loads(json.dumps(_json_safe(envelope), default=str))
    assert parsed["session_restore_served"]["bank"]["maintenance"]["install_s"] == 0.001
    assert parsed["first_round"]["wall_s"] == 0.95


def test_metrics_envelope_probe_defaults_empty():
    envelope = _envelope({"prompt_eval_time_s": 1.0})
    assert envelope["session_restore_served"] == {}
    assert envelope["prompt_state_total_time_s"] == 0.0
    assert envelope["prompt_state_unattributed_time_s"] == 0.0
    assert envelope["first_primary_sample_time_s"] == 0.0
    assert envelope["first_round"] == {}
