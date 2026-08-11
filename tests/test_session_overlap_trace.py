from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SessionBank, SessionBankEntry


def _entry(token_ids: list[int], **overrides: object) -> SessionBankEntry:
    values: dict[str, object] = {
        "token_ids": tuple(token_ids),
        "token_hash": "secret-token-hash",
        "model_path": "/secret/model/path",
        "mtp_enabled": True,
        "hidden_variant": "post_norm",
        "cache_snapshot": CacheSnapshot(states=(), meta_states=()),
        "logits": None,
        "hidden": None,
        "session_id": "secret-session-id",
        "template_hash": "template",
        "mtp_history_policy": "committed",
        "draft_head_identity": "draft-head",
        "policy_fingerprint": "policy",
        "mtp_history_snapshot": CacheSnapshot(states=(), meta_states=()),
        "snapshot_epoch": 7,
        "mtp_snapshot_epoch": 7,
    }
    values.update(overrides)
    return SessionBankEntry(**values)  # type: ignore[arg-type]


def _install(bank: SessionBank, entry: SessionBankEntry) -> None:
    bank._set_entry_for_overlap_trace(entry.token_ids, entry)


def _begin(
    bank: SessionBank, token_ids: list[int], *, lane: str = "solo_mtp"
) -> dict[str, object] | None:
    return bank.begin_overlap_probe(
        token_ids,
        lane=lane,
        model_path="/secret/model/path",
        mtp_enabled=True,
        hidden_variant="post_norm",
        template_hash="template",
        mtp_history_policy="committed",
        draft_head_identity="draft-head",
        policy_fingerprint="policy",
    )


def test_overlap_trace_is_default_off_and_clear_does_not_clear_cache() -> None:
    bank = SessionBank()
    _install(bank, _entry([1, 2, 3]))

    assert _begin(bank, [1, 2, 3, 4]) is None
    assert bank.overlap_trace_snapshot()["schema_version"] == 2
    assert bank.overlap_trace_snapshot()["enabled"] is False
    assert bank.overlap_trace_snapshot()["events"] == []
    assert bank.overlap_trace_snapshot()["pending_probe_count"] == 0
    assert bank.overlap_trace_snapshot()["sequence_high_watermark"] == 0
    assert bank.clear_overlap_trace() == 0
    assert len(bank) == 1
    assert "events" not in bank.to_dict()["overlap_trace"]


def test_overlap_trace_is_content_free_and_never_consults_cold_tier() -> None:
    class NoColdTier:
        def __getattr__(self, _name: str) -> object:
            raise AssertionError("begin_overlap_probe must not use the cold tier")

    secret = "do-not-export-this-secret"
    bank = SessionBank(overlap_trace_max_events=2, cold_tier=NoColdTier())
    _install(
        bank,
        _entry(
            [1, 2, 3, 4, 5],
            token_hash=secret,
            session_id=secret,
            extra_state={"capture_candidate_tokens": 4, "diagnostic": secret},
            has_recurrent=True,
            gdn_boundaries=[(2, None, None), (4, None, None)],
        ),
    )

    probe = _begin(bank, [1, 2, 3, 9])
    assert probe is not None
    candidate = probe["candidates"][0]  # type: ignore[index]
    assert candidate["capture_candidate_tokens"] == [4]  # type: ignore[index]
    assert candidate["retained_checkpoint_tokens"] == [2, 4, 5]  # type: ignore[index]
    assert bank.finalize_overlap_probe(
        probe,
        terminal_status="completed",
        bank_consulted=True,
        outcome={
            "selected_entry_ordinal": candidate["entry_ordinal"],  # type: ignore[index]
            "cache_hit": True,
            "cached_tokens": 2,
            "cache_source": "ram",
            "restore_kind": "boundary",
            "new_prefill_tokens": 2,
            "target_prefill_time_s": 0.2,
            "cache_restore_time_s": 0.1,
        },
    )

    snapshot = bank.overlap_trace_snapshot()
    serialized = json.dumps(snapshot, sort_keys=True)
    assert secret not in serialized
    assert set(snapshot["events"][0]["candidates"][0]) == {
        "entry_ordinal",
        "stored_prefix_tokens",
        "common_prefix_tokens",
        "retained_checkpoint_tokens",
        "capture_candidate_tokens",
        "incumbent_interior_budget",
        "has_recurrent",
        "structurally_restorable",
    }


def test_probe_filters_compatibility_before_ranking_and_limits_candidates() -> None:
    bank = SessionBank(overlap_trace_max_events=2, max_entries=32)
    _install(bank, _entry([1, 2, 3, 4, 5, 6], model_path="wrong-model"))
    for value in range(17):
        _install(bank, _entry([1, 2, value + 10]))

    probe = _begin(bank, [1, 2, 99, 100])
    assert probe is not None
    assert probe["compatible_entry_count"] == 17
    candidates = probe["candidates"]  # type: ignore[assignment]
    assert len(candidates) == 16
    assert all(candidate["stored_prefix_tokens"] == 3 for candidate in candidates)
    assert all(candidate["common_prefix_tokens"] == 2 for candidate in candidates)


def test_probe_excludes_expired_and_nonconsumable_ram_entries_without_purging() -> None:
    bank = SessionBank(overlap_trace_max_events=4, idle_ttl_s=1.0)
    expired = _entry(
        [1, 2, 3, 4, 5],
        last_access_s=time.time() - 2.0,
    )
    live = _entry([1, 2, 3, 4], last_access_s=time.time())
    consumed_lease = _entry(
        [1, 2, 3, 4, 6],
        live_ref_only=True,
        cache_ref=None,
    )
    _install(bank, expired)
    _install(bank, live)
    _install(bank, consumed_lease)

    probe = _begin(bank, [1, 2, 3, 4, 9])
    assert probe is not None
    candidates = probe["candidates"]  # type: ignore[assignment]
    assert [candidate["stored_prefix_tokens"] for candidate in candidates] == [4]
    # Probe collection itself is observational: the expired entry remains
    # resident until the ordinary lookup path performs its standard purge.
    assert expired.token_ids in bank._entries
    assert consumed_lease.token_ids in bank._entries

    restored = bank.restore(
        SimpleNamespace(model_path="/secret/model/path"),
        [1, 2, 3, 4, 9],
        mode="clone",
        cache_factory=list,
        mtp_cache_factory=list,
    )
    assert restored is not None
    assert restored.entry is live
    assert expired.token_ids not in bank._entries
    assert live.token_ids in bank._entries


def test_probe_accepts_a_live_reference_lease_with_required_mtp_history() -> None:
    bank = SessionBank(overlap_trace_max_events=2)
    lease = _entry(
        [1, 2, 3, 4],
        live_ref_only=True,
        cache_ref=[],
        mtp_history_cache_ref=[],
    )
    _install(bank, lease)

    probe = _begin(bank, [1, 2, 3, 4, 9])
    assert probe is not None
    assert probe["compatible_entry_count"] == 1
    assert probe["candidates"][0]["structurally_restorable"] is True  # type: ignore[index]


def test_probe_excludes_committed_history_lanes_without_a_history_payload() -> None:
    """Telemetry must not advertise entries the restore path would reject."""
    for policy in ("committed", "last_window"):
        bank = SessionBank(overlap_trace_max_events=2)
        missing_history = _entry(
            [1, 2, 3, 4],
            mtp_history_policy=policy,
            mtp_history_snapshot=None,
            mtp_history_cache_ref=None,
        )
        durable_history = _entry(
            [1, 2, 3],
            mtp_history_policy=policy,
            mtp_history_snapshot=CacheSnapshot(states=(), meta_states=()),
            mtp_history_cache_ref=None,
        )
        _install(bank, missing_history)
        _install(bank, durable_history)

        probe = bank.begin_overlap_probe(
            [1, 2, 3, 4, 9],
            lane="solo_mtp",
            model_path="/secret/model/path",
            mtp_enabled=True,
            hidden_variant="post_norm",
            template_hash="template",
            mtp_history_policy=policy,
            draft_head_identity="draft-head",
            policy_fingerprint="policy",
        )

        assert probe is not None
        assert probe["compatible_entry_count"] == 1
        candidates = probe["candidates"]  # type: ignore[assignment]
        assert [candidate["stored_prefix_tokens"] for candidate in candidates] == [3]


def test_trace_keeps_only_allowlisted_server_lanes_and_statuses() -> None:
    bank = SessionBank(overlap_trace_max_events=4)
    _install(bank, _entry([1, 2, 3]))

    for lane, status in (
        ("solo_mtp", "completed"),
        ("solo_ar", "cancelled"),
        ("ar_batch", "error"),
    ):
        probe = _begin(bank, [1, 2, 3], lane=lane)
        assert probe is not None
        assert bank.finalize_overlap_probe(
            probe,
            terminal_status=status,
            bank_consulted=True,
            outcome={},
        )

    assert [event["lane"] for event in bank.overlap_trace_snapshot()["events"]] == [
        "solo_mtp",
        "solo_ar",
        "ar_batch",
    ]


def test_finalize_accepts_server_outcome_mapping_exactly_once_and_copies_snapshot() -> (
    None
):
    bank = SessionBank(overlap_trace_max_events=2)
    _install(bank, _entry([1, 2, 3, 4]))

    probe = _begin(bank, [1, 2, 3, 9])
    assert probe is not None
    selected = probe["candidates"][0]["entry_ordinal"]  # type: ignore[index]
    assert bank.finalize_overlap_probe(
        probe,
        terminal_status="completed",
        bank_consulted=True,
        outcome={
            "selected_entry_ordinal": selected,
            "cache_hit": True,
            "cached_tokens": 3,
            "cache_source": "ram",
            "restore_kind": "clone",
            "new_prefill_tokens": 1,
            "target_prefill_time_s": 0.125,
            "cache_restore_time_s": 0.01,
            "mtp_history_time_s": 0.02,
            "checkpoint_capture_time_s": 0.03,
        },
    )
    assert not bank.finalize_overlap_probe(
        probe,
        terminal_status="completed",
        bank_consulted=True,
    )

    snapshot = bank.overlap_trace_snapshot()
    assert snapshot["events_collected"] == 1
    event = snapshot["events"][0]
    assert event["outcome"] == {
        "selected_entry_ordinal": selected,
        "cache_hit": True,
        "cached_tokens": 3,
        "cache_source": "ram",
        "restore_kind": "clone",
        "new_prefill_tokens": 1,
    }
    assert event["prompt_tokens"] == 4
    assert event["timings"] == {
        "cache_restore_time_s": 0.01,
        "target_prefill_time_s": 0.125,
        "mtp_history_time_s": 0.02,
        "checkpoint_capture_time_s": 0.03,
    }
    event["candidates"][0]["retained_checkpoint_tokens"].append(99)
    assert (
        99
        not in bank.overlap_trace_snapshot()["events"][0]["candidates"][0][
            "retained_checkpoint_tokens"
        ]
    )


def test_entry_ordinals_are_stable_until_replacement_and_cache_clear_advances_epoch() -> (
    None
):
    bank = SessionBank(overlap_trace_max_events=2)
    _install(bank, _entry([1, 2, 3]))

    first = _begin(bank, [1, 2, 3, 4])
    assert first is not None
    first_ordinal = first["candidates"][0]["entry_ordinal"]  # type: ignore[index]
    assert _begin(bank, [1, 2, 3, 5])["candidates"][0]["entry_ordinal"] == first_ordinal  # type: ignore[index]

    _install(bank, _entry([1, 2, 3], token_hash="replacement"))
    replacement = _begin(bank, [1, 2, 3, 6])
    assert replacement is not None
    assert replacement["candidates"][0]["entry_ordinal"] != first_ordinal  # type: ignore[index]

    epoch = replacement["bank_epoch"]
    assert bank.clear() == 1
    assert bank._overlap_entry_ordinals == {}
    _install(bank, _entry([1, 2, 3], token_hash="after-clear"))
    after_clear = _begin(bank, [1, 2, 3, 7])
    assert after_clear is not None
    assert after_clear["bank_epoch"] == epoch + 1
    assert after_clear["candidates"][0]["entry_ordinal"] != first_ordinal  # type: ignore[index]


def test_lazy_boundary_hydration_cannot_mutate_trace_placement_for_an_ordinal() -> None:
    bank = SessionBank(overlap_trace_max_events=4)
    boundary_snapshot = CacheSnapshot(states=(), meta_states=())
    loader_calls: list[None] = []

    def load_boundaries() -> list[tuple[int, CacheSnapshot, None]]:
        loader_calls.append(None)
        return [(3, boundary_snapshot, None)]

    entry = _entry(
        [1, 2, 3, 4, 5, 6],
        has_recurrent=True,
        gdn_boundary_loader=load_boundaries,
    )
    _install(bank, entry)

    before = _begin(bank, [1, 2, 3, 4])
    assert before is not None
    before_candidate = before["candidates"][0]  # type: ignore[index]
    assert before_candidate["retained_checkpoint_tokens"] == [6]
    assert before_candidate["incumbent_interior_budget"] == 0
    assert before_candidate["structurally_restorable"] is False
    assert loader_calls == []

    assert entry.recurrent_boundary_at_or_below(3) == (3, boundary_snapshot, None)
    assert loader_calls == [None]

    after = _begin(bank, [1, 2, 3, 4])
    assert after is not None
    after_candidate = after["candidates"][0]  # type: ignore[index]
    assert after_candidate["entry_ordinal"] != before_candidate["entry_ordinal"]
    assert after_candidate["retained_checkpoint_tokens"] == [3, 6]
    assert after_candidate["incumbent_interior_budget"] == 1
    assert after_candidate["structurally_restorable"] is True

    # Probing stable, already-hydrated metadata must preserve the revision.
    stable = _begin(bank, [1, 2, 3, 4])
    assert stable is not None
    stable_candidate = stable["candidates"][0]  # type: ignore[index]
    assert stable_candidate == after_candidate
    assert bank.finalize_overlap_probe(
        before,
        terminal_status="completed",
        bank_consulted=True,
        outcome={
            "cache_hit": False,
            "cache_source": "none",
            "restore_kind": "cold",
        },
    )
    assert bank.finalize_overlap_probe(
        after,
        terminal_status="completed",
        bank_consulted=True,
        outcome={
            "selected_entry_ordinal": after_candidate["entry_ordinal"],
            "cache_hit": True,
            "cached_tokens": 3,
            "cache_source": "ram",
            "restore_kind": "boundary",
        },
    )
    retained = bank.overlap_trace_snapshot()["events"]
    assert [event["candidates"][0]["entry_ordinal"] for event in retained] == [
        before_candidate["entry_ordinal"],
        after_candidate["entry_ordinal"],
    ]


def test_placement_metadata_changes_allocate_a_new_ordinal() -> None:
    bank = SessionBank(overlap_trace_max_events=8)
    entry = _entry([1, 2, 3, 4], live_ref_only=True)
    _install(bank, entry)

    # Placement identities are revised even for a lease that telemetry must
    # exclude until it has the full real restore state.
    first_placement = bank._overlap_trace_placement_for_entry(entry)
    assert first_placement.has_restorable_cache is False

    # A restorable-cache state change is placement-relevant.
    entry.cache_ref = []
    cache_ready_placement = bank._overlap_trace_placement_for_entry(entry)
    assert cache_ready_placement.entry_ordinal != first_placement.entry_ordinal
    assert cache_ready_placement.has_restorable_cache is True

    # The MTP lease is separately required by restore(); once it is present,
    # the current revision becomes a probe candidate.
    entry.mtp_history_cache_ref = []
    cache_ready = _begin(bank, [1, 2, 3, 4])
    assert cache_ready is not None
    cache_ready_candidate = cache_ready["candidates"][0]  # type: ignore[index]
    assert cache_ready_candidate["entry_ordinal"] == cache_ready_placement.entry_ordinal
    assert cache_ready_candidate["structurally_restorable"] is True

    # Capture opportunities are likewise part of the placement identity.
    entry.capture_candidate_tokens = [2]
    capture_added = _begin(bank, [1, 2, 3, 4])
    assert capture_added is not None
    capture_added_candidate = capture_added["candidates"][0]  # type: ignore[index]
    assert (
        capture_added_candidate["entry_ordinal"]
        != cache_ready_candidate["entry_ordinal"]
    )
    assert capture_added_candidate["capture_candidate_tokens"] == [2]

    # No further metadata change means the ordinal remains stable.
    stable = _begin(bank, [1, 2, 3, 4])
    assert stable is not None
    assert stable["candidates"][0] == capture_added_candidate  # type: ignore[index]


def test_recurrent_provenance_is_part_of_trace_entry_identity() -> None:
    bank = SessionBank(overlap_trace_max_events=4)
    entry = _entry([1, 2, 3], has_recurrent=False)
    _install(bank, entry)

    attention_probe = _begin(bank, [1, 2, 3, 4])
    assert attention_probe is not None
    attention_candidate = attention_probe["candidates"][0]  # type: ignore[index]
    assert attention_candidate["has_recurrent"] is False

    # A recurrence classification correction changes partial-restore rights,
    # so it must never reuse the same public placement identity.
    entry.has_recurrent = True
    recurrent_probe = _begin(bank, [1, 2, 3, 4])
    assert recurrent_probe is not None
    recurrent_candidate = recurrent_probe["candidates"][0]  # type: ignore[index]
    assert recurrent_candidate["has_recurrent"] is True
    assert recurrent_candidate["entry_ordinal"] != attention_candidate["entry_ordinal"]


def test_recurrent_exact_prefix_is_structurally_restorable_without_interior_capture() -> (
    None
):
    bank = SessionBank(overlap_trace_max_events=2)
    _install(bank, _entry([1, 2, 3], has_recurrent=True))

    probe = _begin(bank, [1, 2, 3])
    assert probe is not None
    candidate = probe["candidates"][0]  # type: ignore[index]
    assert candidate["retained_checkpoint_tokens"] == [3]  # type: ignore[index]
    assert candidate["structurally_restorable"] is True  # type: ignore[index]


def test_overlap_trace_sequence_is_request_start_order_not_completion_order() -> None:
    bank = SessionBank(overlap_trace_max_events=2)
    _install(bank, _entry([1, 2, 3, 4, 5]))

    started_a = _begin(bank, [1, 2, 3, 10])
    started_b = _begin(bank, [1, 2, 3, 4, 20])
    assert started_a is not None
    assert started_b is not None

    # B finishes first, so retained payload order is B then A. The public
    # sequence nevertheless follows the successful begin order A then B.
    assert bank.finalize_overlap_probe(
        started_b, terminal_status="completed", bank_consulted=True
    )
    assert bank.finalize_overlap_probe(
        started_a, terminal_status="completed", bank_consulted=True
    )

    events = bank.overlap_trace_snapshot()["events"]
    assert [event["prompt_tokens"] for event in events] == [5, 4]
    assert [event["sequence"] for event in events] == [2, 1]
    assert events[1]["sequence"] < events[0]["sequence"]
    assert len({event["sequence"] for event in events}) == 2


def test_trace_snapshot_exposes_quiescence_contract_and_clear_resets_segment() -> None:
    bank = SessionBank(overlap_trace_max_events=4)
    _install(bank, _entry([1, 2, 3, 4]))

    first = _begin(bank, [1, 2, 3, 9])
    second = _begin(bank, [1, 2, 3, 8])
    assert first is not None
    assert second is not None
    in_flight = bank.overlap_trace_snapshot()
    assert in_flight["pending_probe_count"] == 2
    assert in_flight["sequence_high_watermark"] == 2
    assert in_flight["events_collected"] == 0
    assert in_flight["events_dropped"] == 0

    assert bank.clear_overlap_trace() == 0
    cleared = bank.overlap_trace_snapshot()
    assert cleared["pending_probe_count"] == 0
    assert cleared["sequence_high_watermark"] == 0
    assert cleared["events_collected"] == 0
    assert cleared["events_dropped"] == 0
    assert cleared["events"] == []
    # The pending ids are never reused, so an old request cannot terminalize
    # a new trace segment even though public event numbering resets.
    assert not bank.finalize_overlap_probe(
        first, terminal_status="cancelled", bank_consulted=False
    )

    fresh = _begin(bank, [1, 2, 3, 7])
    assert fresh is not None
    assert fresh["probe_id"] > second["probe_id"]
    assert bank.finalize_overlap_probe(
        fresh, terminal_status="completed", bank_consulted=True
    )
    quiescent = bank.overlap_trace_snapshot()
    assert quiescent["pending_probe_count"] == 0
    assert quiescent["sequence_high_watermark"] == 1
    assert quiescent["events_collected"] == 1
    assert quiescent["events_dropped"] == 0
    assert quiescent["events"][0]["sequence"] == 1


def test_snapshot_discloses_pending_requests_before_and_after_terminalization() -> None:
    """A replay consumer can reject a cohort that still has begun requests."""
    bank = SessionBank(overlap_trace_max_events=512)
    _install(bank, _entry([1, 2, 3, 4]))

    for index in range(340):
        probe = _begin(bank, [1, 2, 3, 4, index])
        assert probe is not None
        assert bank.finalize_overlap_probe(
            probe, terminal_status="completed", bank_consulted=True
        )
    pending = []
    for index in range(100):
        probe = _begin(bank, [1, 2, 3, 4, 1_000 + index])
        assert probe is not None
        pending.append(probe)

    in_flight = bank.overlap_trace_snapshot()
    assert len(in_flight["events"]) == 340
    assert in_flight["events_collected"] == 340
    assert in_flight["pending_probe_count"] == 100
    assert in_flight["sequence_high_watermark"] == 440
    assert in_flight["events_dropped"] == 0

    for probe in pending:
        assert bank.finalize_overlap_probe(
            probe, terminal_status="cancelled", bank_consulted=False
        )
    terminal = bank.overlap_trace_snapshot()
    assert terminal["pending_probe_count"] == 0
    assert terminal["events_collected"] == 440
    assert terminal["sequence_high_watermark"] == 440
    assert terminal["events_dropped"] == 0
    assert [event["sequence"] for event in terminal["events"]] == list(range(1, 441))
    assert all(
        event["terminal_status"] == "cancelled" for event in terminal["events"][340:]
    )


def test_overlap_trace_is_lock_safe_and_bounded() -> None:
    bank = SessionBank(overlap_trace_max_events=1)
    _install(bank, _entry([1, 2, 3, 4]))

    def record_once(_index: int) -> bool:
        probe = _begin(bank, [1, 2, 3, 9])
        assert probe is not None
        return bank.finalize_overlap_probe(
            probe,
            terminal_status="cancelled",
            bank_consulted=False,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert all(executor.map(record_once, range(32)))

    snapshot = bank.overlap_trace_snapshot()
    assert snapshot["events_collected"] == 32
    assert snapshot["events_dropped"] == 31
    assert len(snapshot["events"]) == 1
    assert snapshot["events"][0]["terminal_status"] == "cancelled"
