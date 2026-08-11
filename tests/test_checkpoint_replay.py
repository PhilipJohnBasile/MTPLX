"""CPU-only contract tests for SessionBank schema-v2 checkpoint replay."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import copy
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mtplx import cli
import mtplx.checkpoint_replay as checkpoint_replay
from mtplx.checkpoint_replay import replay_checkpoint_trace


def _candidate(
    entry_ordinal: int,
    *,
    stored: int = 4096,
    common: int = 2048,
    retained: tuple[int, ...] = (256, 512, 4096),
    capture: tuple[int, ...] | None = None,
    recurrent: bool = True,
    restorable: bool = True,
) -> dict[str, object]:
    return {
        "entry_ordinal": entry_ordinal,
        "stored_prefix_tokens": stored,
        "common_prefix_tokens": common,
        "retained_checkpoint_tokens": list(retained),
        "capture_candidate_tokens": list(retained if capture is None else capture),
        "incumbent_interior_budget": sum(position < stored for position in retained),
        "has_recurrent": recurrent,
        "structurally_restorable": restorable,
    }


def _event(
    sequence: int,
    *,
    prompt: int = 4096,
    candidates: list[dict[str, object]] | None = None,
    bank_consulted: bool = True,
    cache_hit: bool = True,
    selected_entry_ordinal: int | None = 1,
    cached_tokens: int | None = None,
) -> dict[str, object]:
    candidates = candidates if candidates is not None else [_candidate(1)]
    selected = next(
        (
            item
            for item in candidates
            if item["entry_ordinal"] == selected_entry_ordinal
        ),
        None,
    )
    if cached_tokens is None:
        if selected is None:
            cached_tokens = 0
        else:
            overlap = int(selected["common_prefix_tokens"])
            retained = [int(item) for item in selected["retained_checkpoint_tokens"]]
            cached_tokens = max(
                (item for item in retained if item <= overlap), default=0
            )
    return {
        "schema_version": 2,
        "sequence": sequence,
        "bank_epoch": 1,
        "lane": "solo_mtp",
        "prompt_tokens": prompt,
        "terminal_status": "completed",
        "bank_consulted": bank_consulted,
        "compatible_entry_count": len(candidates),
        "candidates": candidates,
        "outcome": {
            "selected_entry_ordinal": selected_entry_ordinal,
            "cache_hit": cache_hit,
            "cached_tokens": cached_tokens,
            "new_prefill_tokens": prompt - cached_tokens,
            "cache_source": "ram" if cache_hit else "none",
            "restore_kind": "boundary" if cache_hit else "cold",
        },
        "timings": {
            "cache_restore_time_s": 0.001,
            "target_prefill_time_s": 0.010,
            "mtp_history_time_s": 0.0,
            "checkpoint_capture_time_s": 0.0,
        },
    }


def _payload(events: list[dict[str, object]], *, dropped: int = 0) -> dict[str, object]:
    return {
        "schema_version": 2,
        "enabled": True,
        "max_events": 4096,
        "events_collected": len(events),
        "events_dropped": dropped,
        "pending_probe_count": 0,
        "sequence_high_watermark": len(events),
        "bank_epoch": 1,
        "events": events,
    }


def _history_then_eval(
    *,
    stored: int = 4096,
    retained: tuple[int, ...] = (256, 512, 4096),
    capture: tuple[int, ...] | None = None,
    history: tuple[int, ...] = (3840, 2000, 3840, 2000, 3840, 2000, 3840),
    evaluation: tuple[int, ...] = (3840, 2000, 3840),
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for sequence, common in enumerate((*history, *evaluation), start=1):
        events.append(
            _event(
                sequence,
                prompt=stored,
                candidates=[
                    _candidate(
                        1,
                        stored=stored,
                        common=common,
                        retained=retained,
                        capture=capture,
                    )
                ],
            )
        )
    return events


def _bruteforce_transformed(
    depths: tuple[int, ...], *, stored: int, budget: int, legal: tuple[int, ...]
) -> tuple[int, ...]:
    legal = tuple(position for position in legal if position < stored)
    last = legal[-1]
    count = min(budget, len(legal))
    transformed = [min(depth, last) for depth in depths if 0 < depth < stored]

    def loss(positions: tuple[int, ...]) -> int:
        return sum(
            depth
            - max((position for position in positions if position <= depth), default=0)
            for depth in transformed
        )

    return min(
        combinations(legal, count), key=lambda positions: (loss(positions), positions)
    )


def test_consumes_exact_producer_snapshot_without_synthetic_candidate_flags() -> None:
    """A SessionBank-produced snapshot is the direct replay contract."""
    pytest.importorskip("mlx.core")
    from mtplx.cache_state import CacheSnapshot
    from mtplx.session_bank import SessionBank, SessionBankEntry

    entry = SessionBankEntry(
        token_ids=(1, 2, 3, 4, 5),
        token_hash="not-in-output",
        model_path="/model",
        mtp_enabled=True,
        hidden_variant="post_norm",
        cache_snapshot=CacheSnapshot(states=(), meta_states=()),
        logits=None,
        hidden=None,
        session_id="session",
        template_hash="template",
        mtp_history_policy="committed",
        draft_head_identity="draft",
        policy_fingerprint="policy",
        mtp_history_snapshot=CacheSnapshot(states=(), meta_states=()),
        snapshot_epoch=1,
        mtp_snapshot_epoch=1,
        has_recurrent=True,
        gdn_boundaries=[(2, None, None), (4, None, None)],
    )
    bank = SessionBank(overlap_trace_max_events=4096)
    bank._set_entry_for_overlap_trace(entry.token_ids, entry)
    probe = bank.begin_overlap_probe(
        [1, 2, 3, 9],
        lane="solo_mtp",
        model_path="/model",
        mtp_enabled=True,
        hidden_variant="post_norm",
        template_hash="template",
        mtp_history_policy="committed",
        draft_head_identity="draft",
        policy_fingerprint="policy",
    )
    assert probe is not None
    ordinal = probe["candidates"][0]["entry_ordinal"]
    assert bank.finalize_overlap_probe(
        probe,
        terminal_status="completed",
        bank_consulted=True,
        outcome={
            "selected_entry_ordinal": ordinal,
            "cache_hit": True,
            "cached_tokens": 2,
            "new_prefill_tokens": 2,
            "cache_source": "ram",
            "restore_kind": "boundary",
        },
    )
    snapshot = bank.overlap_trace_snapshot()
    assert snapshot["pending_probe_count"] == 0
    assert snapshot["sequence_high_watermark"] == 1
    assert snapshot["events_collected"] == len(snapshot["events"]) == 1
    candidate = snapshot["events"][0]["candidates"][0]
    assert "cache_hit" not in candidate
    assert "identity_compatible" not in candidate
    result = replay_checkpoint_trace(snapshot)
    assert result["status"] == "insufficient_data"
    assert result["trace"]["events_total"] == 1
    assert result["trace"]["pending_probe_count"] == 0
    assert result["trace"]["sequence_high_watermark"] == 1

    for _ in range(339):
        finalized = bank.begin_overlap_probe(
            [1, 2, 3, 9],
            lane="solo_mtp",
            model_path="/model",
            mtp_enabled=True,
            hidden_variant="post_norm",
            template_hash="template",
            mtp_history_policy="committed",
            draft_head_identity="draft",
            policy_fingerprint="policy",
        )
        assert finalized is not None
        assert bank.finalize_overlap_probe(
            finalized,
            terminal_status="completed",
            bank_consulted=True,
            outcome={
                "selected_entry_ordinal": ordinal,
                "cache_hit": True,
                "cached_tokens": 2,
                "new_prefill_tokens": 2,
                "cache_source": "ram",
                "restore_kind": "boundary",
            },
        )
    pending = []
    for _ in range(100):
        in_flight = bank.begin_overlap_probe(
            [1, 2, 3, 9],
            lane="solo_mtp",
            model_path="/model",
            mtp_enabled=True,
            hidden_variant="post_norm",
            template_hash="template",
            mtp_history_policy="committed",
            draft_head_identity="draft",
            policy_fingerprint="policy",
        )
        assert in_flight is not None
        pending.append(in_flight)

    incomplete = bank.overlap_trace_snapshot()
    assert len(incomplete["events"]) == 340
    assert incomplete["events_collected"] == 340
    assert incomplete["pending_probe_count"] == 100
    assert incomplete["sequence_high_watermark"] == 440
    assert incomplete["events_dropped"] == 0
    with pytest.raises(ValueError, match="pending_probe_count must equal zero"):
        replay_checkpoint_trace(incomplete)

    last_probe_id = pending[-1]["probe_id"]
    assert bank.clear_overlap_trace() == 340
    cleared = bank.overlap_trace_snapshot()
    assert cleared["pending_probe_count"] == 0
    assert cleared["sequence_high_watermark"] == 0
    assert cleared["events_collected"] == len(cleared["events"]) == 0
    assert replay_checkpoint_trace(cleared)["status"] == "insufficient_data"

    fresh = bank.begin_overlap_probe(
        [1, 2, 3, 9],
        lane="solo_mtp",
        model_path="/model",
        mtp_enabled=True,
        hidden_variant="post_norm",
        template_hash="template",
        mtp_history_policy="committed",
        draft_head_identity="draft",
        policy_fingerprint="policy",
    )
    assert fresh is not None
    assert fresh["probe_id"] > last_probe_id
    assert bank.finalize_overlap_probe(
        fresh,
        terminal_status="completed",
        bank_consulted=True,
        outcome={
            "selected_entry_ordinal": ordinal,
            "cache_hit": True,
            "cached_tokens": 2,
            "new_prefill_tokens": 2,
            "cache_source": "ram",
            "restore_kind": "boundary",
        },
    )
    fresh_snapshot = bank.overlap_trace_snapshot()
    assert fresh_snapshot["pending_probe_count"] == 0
    assert fresh_snapshot["events_collected"] == 1
    assert fresh_snapshot["sequence_high_watermark"] == 1
    assert fresh_snapshot["events"][0]["sequence"] == 1
    assert replay_checkpoint_trace(fresh_snapshot)["status"] == "insufficient_data"


def test_sessionbank_excludes_missing_committed_history_from_both_replay_policies() -> (
    None
):
    """An impossible committed-MTP entry cannot improve either replay arm."""
    pytest.importorskip("mlx.core")
    from mtplx.cache_state import CacheSnapshot
    from mtplx.session_bank import SessionBank, SessionBankEntry

    def entry(
        token_ids: tuple[int, ...], *, mtp_history_snapshot: object | None
    ) -> SessionBankEntry:
        return SessionBankEntry(
            token_ids=token_ids,
            token_hash="not-in-output",
            model_path="/model",
            mtp_enabled=True,
            hidden_variant="post_norm",
            cache_snapshot=CacheSnapshot(states=(), meta_states=()),
            logits=None,
            hidden=None,
            session_id="session",
            template_hash="template",
            mtp_history_policy="committed",
            draft_head_identity="draft",
            policy_fingerprint="policy",
            mtp_history_snapshot=mtp_history_snapshot,
            snapshot_epoch=1,
            mtp_snapshot_epoch=1,
            has_recurrent=True,
            capture_candidate_tokens=[2, 4, 6, 8, 10],
            gdn_boundaries=[
                (2, None, None),
                (4, None, None),
                (6, None, None),
                (8, None, None),
            ],
        )

    bank = SessionBank(overlap_trace_max_events=16)
    # This entry has the better common prefix, so either replay policy would
    # choose it if the producer emitted its impossible restore contract.
    missing_history = entry((1, 2, 3, 4, 5, 6, 7, 8, 90, 91), mtp_history_snapshot=None)
    durable_history = entry(
        (1, 2, 3, 4, 5, 6, 7, 80, 81, 82),
        mtp_history_snapshot=CacheSnapshot(states=(), meta_states=()),
    )
    bank._set_entry_for_overlap_trace(missing_history.token_ids, missing_history)
    bank._set_entry_for_overlap_trace(durable_history.token_ids, durable_history)
    missing_ordinal = bank._overlap_trace_placement_for_entry(
        missing_history
    ).entry_ordinal
    durable_ordinal = bank._overlap_trace_placement_for_entry(
        durable_history
    ).entry_ordinal

    for _ in range(10):
        probe = bank.begin_overlap_probe(
            [1, 2, 3, 4, 5, 6, 7, 8, 99, 100],
            lane="solo_mtp",
            model_path="/model",
            mtp_enabled=True,
            hidden_variant="post_norm",
            template_hash="template",
            mtp_history_policy="committed",
            draft_head_identity="draft",
            policy_fingerprint="policy",
        )
        assert probe is not None
        assert [candidate["entry_ordinal"] for candidate in probe["candidates"]] == [
            durable_ordinal
        ]
        assert bank.finalize_overlap_probe(
            probe,
            terminal_status="completed",
            bank_consulted=True,
            outcome={
                "selected_entry_ordinal": durable_ordinal,
                "cache_hit": True,
                "cached_tokens": 6,
                "new_prefill_tokens": 4,
                "cache_source": "ram",
                "restore_kind": "boundary",
            },
        )

    result = replay_checkpoint_trace(
        bank.overlap_trace_snapshot(), min_events=1, min_bucket_events=1
    )
    for event in result["events"]:
        for policy in ("observed_fitted", "future_known_upper"):
            selected = event["policies"][policy]["candidate"]
            assert selected is not None
            assert selected["entry_ordinal"] == durable_ordinal
            assert selected["entry_ordinal"] != missing_ordinal


def test_fitted_placement_matches_independent_tail_folded_bruteforce() -> None:
    history = (2, 5, 7, 8, 9, 2, 7)
    events = _history_then_eval(
        stored=9,
        retained=(2, 6, 9),
        capture=(2, 4, 6, 9),
        history=history,
        evaluation=(8, 8, 8),
    )
    result = replay_checkpoint_trace(
        _payload(events), block_size=3, min_events=1, min_bucket_events=1
    )
    expected = _bruteforce_transformed(history, stored=9, budget=2, legal=(2, 4, 6, 9))
    selection = result["events"][0]["policies"]["observed_fitted"]
    assert selection["checkpoint_tokens"] == [*expected, 9]
    assert selection["requested_budget"] == 2
    assert selection["effective_budget"] == 2


def test_future_known_upper_dominates_observed_selection_and_uses_best_candidate() -> (
    None
):
    events: list[dict[str, object]] = []
    for sequence in range(1, 8):
        events.append(
            _event(
                sequence,
                candidates=[
                    _candidate(1, common=256),
                    _candidate(2, common=3840),
                ],
                selected_entry_ordinal=1,
            )
        )
    for sequence in range(8, 11):
        events.append(
            _event(
                sequence,
                candidates=[
                    _candidate(1, common=256),
                    _candidate(2, common=3840),
                ],
                selected_entry_ordinal=1,
            )
        )
    result = replay_checkpoint_trace(
        _payload(events), min_events=1, min_bucket_events=1
    )
    row = result["events"][0]["policies"]
    assert row["incumbent"]["candidate"]["entry_ordinal"] == 1
    assert row["observed_fitted"]["candidate"]["entry_ordinal"] == 2
    assert row["future_known_upper"]["candidate"]["entry_ordinal"] == 2
    assert (
        row["future_known_upper"]["requested_budget"]
        == row["observed_fitted"]["requested_budget"]
    )
    assert (
        row["future_known_upper"]["work_tokens"]
        <= row["observed_fitted"]["work_tokens"]
    )
    assert "optimistic" in result["labels"]["future_known_upper"]


def test_future_known_upper_places_at_the_actual_prefix_and_handles_endpoints() -> None:
    events = _history_then_eval(
        stored=10,
        retained=(2, 10),
        capture=(2, 10),
        history=(2, 2, 2, 2, 2, 2, 2),
        evaluation=(2, 10),
    )
    result = replay_checkpoint_trace(
        _payload(events), block_size=3, min_events=1, min_bucket_events=1
    )
    rows = {row["sequence"]: row for row in result["events"]}

    useful = rows[8]["policies"]["future_known_upper"]
    assert useful["checkpoint_tokens"] == [2, 10]
    assert useful["requested_budget"] == useful["effective_budget"] == 1
    assert useful["work_tokens"] == 8

    endpoint = rows[9]["policies"]["future_known_upper"]
    assert endpoint["checkpoint_tokens"] == [10]
    assert endpoint["effective_budget"] == 0
    assert endpoint["work_tokens"] == 0

    zero_budget_candidate = checkpoint_replay._Candidate(
        index=0,
        entry_ordinal=1,
        stored_prefix_tokens=10,
        common_prefix_tokens=5,
        retained_checkpoint_tokens=(10,),
        capture_candidate_tokens=(10,),
        incumbent_interior_budget=0,
        has_recurrent=True,
        structurally_restorable=False,
    )
    assert (
        checkpoint_replay._placement_future_known_upper(zero_budget_candidate)
        == "no_captured_restore_at_common_prefix"
    )


def test_future_known_upper_repositions_a_captured_nonrestorable_recurrent_entry() -> (
    None
):
    """The oracle may retain a recorded boundary that the incumbent did not."""

    events: list[dict[str, object]] = []
    for sequence, common in enumerate((3840, 2000) * 5, start=1):
        events.append(
            _event(
                sequence,
                candidates=[
                    _candidate(
                        1,
                        common=common,
                        retained=(256, 4096),
                        capture=(256, 4096),
                        restorable=True,
                    ),
                    _candidate(
                        2,
                        common=common,
                        retained=(256, 4096),
                        capture=(3840, 4096),
                        restorable=False,
                    ),
                ],
                selected_entry_ordinal=1,
            )
        )

    result = replay_checkpoint_trace(
        _payload(events), min_events=1, min_bucket_events=1
    )
    rows = {row["sequence"]: row["policies"] for row in result["events"]}
    first = rows[9]
    future_bootstrap = result["bootstrap"]["future_known_upper"]

    assert first["incumbent"]["work_tokens"] == 3840
    assert first["observed_fitted"]["work_tokens"] == 3840
    assert first["future_known_upper"] == {
        "covered": True,
        "candidate": {
            "entry_ordinal": 2,
            "stored_prefix_tokens": 4096,
            "common_prefix_tokens": 3840,
            "has_recurrent": True,
            "structurally_restorable": False,
            "incumbent_interior_budget": 1,
        },
        "checkpoint_tokens": [3840, 4096],
        "checkpoint_loss_tokens": 0,
        "target_suffix_tokens": 256,
        "work_tokens": 256,
        "requested_budget": 1,
        "effective_budget": 1,
        "planned_budget": 1,
        "rejection_counts": {},
    }
    assert future_bootstrap is not None
    assert future_bootstrap["point_gain_vs_incumbent_percent"] > 10.0
    assert "future-known upper bound" not in " ".join(
        result["exploratory_assessment"]["numeric_reasons"]
    )


def test_future_known_upper_never_exceeds_observed_capture_counterfactual() -> None:
    rng = random.Random(17)
    for _ in range(24):
        stored = rng.choice((6, 8, 10))
        selected_boundary = rng.randint(1, stored - 1)
        selected_capture = tuple(range(1, stored + 1))
        alternate_recurrent = rng.choice((True, False))
        alternate_restorable = (
            True if not alternate_recurrent else rng.choice((True, False))
        )
        alternate_capture = tuple(
            sorted({rng.randint(1, stored - 1) for _ in range(3)} | {stored})
        )
        alternate_retained = (
            (rng.randint(1, stored - 1), stored) if alternate_recurrent else (stored,)
        )
        events = []
        for sequence in range(1, 11):
            selected_common = rng.randint(selected_boundary, stored)
            alternate_common = rng.randint(1, stored)
            events.append(
                _event(
                    sequence,
                    prompt=stored,
                    candidates=[
                        _candidate(
                            1,
                            stored=stored,
                            common=selected_common,
                            retained=(selected_boundary, stored),
                            capture=selected_capture,
                            recurrent=True,
                            restorable=True,
                        ),
                        _candidate(
                            2,
                            stored=stored,
                            common=alternate_common,
                            retained=alternate_retained,
                            capture=alternate_capture,
                            recurrent=alternate_recurrent,
                            restorable=alternate_restorable,
                        ),
                    ],
                    selected_entry_ordinal=1,
                )
            )
        result = replay_checkpoint_trace(
            _payload(events), block_size=1, min_events=1, min_bucket_events=1
        )
        for row in result["events"]:
            policies = row["policies"]
            assert (
                policies["future_known_upper"]["work_tokens"]
                <= policies["observed_fitted"]["work_tokens"]
            )


def test_misses_and_bypasses_remain_zero_benefit_workload_rows() -> None:
    events = _history_then_eval(evaluation=())
    events.extend(
        [
            _event(
                8,
                candidates=[],
                bank_consulted=True,
                cache_hit=False,
                selected_entry_ordinal=None,
            ),
            _event(
                9,
                candidates=[],
                bank_consulted=False,
                cache_hit=False,
                selected_entry_ordinal=None,
            ),
            _event(10, candidates=[_candidate(1, common=3840)]),
        ]
    )
    result = replay_checkpoint_trace(
        _payload(events), min_events=1, min_bucket_events=1
    )
    rows = {row["sequence"]: row for row in result["events"]}
    assert rows[8]["cache_case"] == "miss"
    assert rows[9]["cache_case"] == "bypass"
    assert rows[8]["policies"]["incumbent"]["work_tokens"] == 4096
    assert rows[9]["policies"]["observed_fitted"]["work_tokens"] == 4096
    assert result["trace"]["cache_case_counts"] == {"hit": 1, "miss": 1, "bypass": 1}


def test_ssd_hits_are_workload_rows_but_never_ram_checkpoint_eligible() -> None:
    events = _history_then_eval(evaluation=())
    ssd_event = _event(
        8,
        candidates=[],
        cache_hit=True,
        selected_entry_ordinal=None,
        cached_tokens=1024,
    )
    ssd_event["outcome"]["cache_source"] = "ssd"
    events.extend(
        [
            ssd_event,
            _event(9, candidates=[_candidate(1, common=3840)]),
            _event(10, candidates=[_candidate(1, common=2000)]),
        ]
    )

    result = replay_checkpoint_trace(
        _payload(events), min_events=1, min_bucket_events=1
    )
    rows = {row["sequence"]: row for row in result["events"]}

    assert rows[8]["cache_source"] == "ssd"
    assert rows[8]["policies"]["incumbent"]["work_tokens"] == 4096
    assert rows[8]["policies"]["observed_fitted"]["work_tokens"] == 4096
    assert rows[8]["policies"]["observed_fitted"]["rejection_counts"] == {}
    assert result["trace"]["cache_source_counts"] == {
        "ram": 2,
        "ssd": 1,
        "none": 0,
        "other": 0,
    }
    assert result["trace"]["eligible_evaluation_event_count"] == 2


def test_workload_bootstrap_dilutes_conditionally_positive_hits_with_misses() -> None:
    capture = tuple(range(256, 4097, 256))
    events: list[dict[str, object]] = []
    # 167 total rows split chronologically into 116 fit and 51 evaluation rows.
    for sequence in range(1, 117):
        common = 3840 if sequence % 2 == 1 else 2000
        events.append(
            _event(
                sequence,
                candidates=[_candidate(1, common=common, capture=capture)],
            )
        )
    for sequence in range(117, 125):
        common = 3840 if sequence % 2 == 1 else 2000
        events.append(
            _event(
                sequence,
                candidates=[_candidate(1, common=common, capture=capture)],
            )
        )
    for sequence in range(125, 168):
        bypass = sequence % 2 == 1
        events.append(
            _event(
                sequence,
                candidates=[],
                bank_consulted=not bypass,
                cache_hit=False,
                selected_entry_ordinal=None,
            )
        )

    result = replay_checkpoint_trace(
        _payload(events), min_events=8, min_bucket_events=4
    )

    observed = result["policy_summaries"]["observed_fitted"]
    assert observed["conditional_nonappend_gain"]["gain_vs_incumbent_percent"] > 10.0
    assert observed["workload_wide_gain_vs_incumbent_percent"] < 10.0
    assert result["status"] == "exploratory_configuration"
    assert result["canonical_gate"] is False
    assert result["exploratory_assessment"]["numeric_outcome"] == "no_go"
    assert result["trace"]["eligible_evaluation_event_count"] == 8
    assert result["trace"]["bootstrap_event_count"] == 51
    assert result["bootstrap"]["bootstrap_event_count"] == 51
    assert result["bootstrap"]["observed_fitted"]["block_lengths"] == [4, 8]
    assert (
        result["bootstrap"]["observed_fitted"]["point_gain_vs_incumbent_percent"] < 10.0
    )


def test_insufficient_repeated_history_is_not_eligible() -> None:
    events = [
        _event(
            sequence,
            candidates=[_candidate(sequence)],
            selected_entry_ordinal=sequence,
        )
        for sequence in range(1, 11)
    ]
    result = replay_checkpoint_trace(
        _payload(events), min_events=1, min_bucket_events=1
    )
    assert result["status"] == "exploratory_configuration"
    assert result["exploratory_assessment"]["numeric_outcome"] == "insufficient_data"
    assert result["trace"]["eligible_evaluation_event_count"] == 0
    assert result["events"][0]["policies"]["observed_fitted"]["rejection_counts"] == {
        "insufficient_fit_history": 1
    }


def test_missing_capture_candidates_make_observed_and_future_arms_ineligible() -> None:
    events = _history_then_eval(capture=(4096,))
    result = replay_checkpoint_trace(
        _payload(events), min_events=1, min_bucket_events=1
    )
    observed = result["events"][0]["policies"]["observed_fitted"]
    future = result["events"][0]["policies"]["future_known_upper"]
    assert observed["covered"] is False
    assert observed["rejection_counts"] == {"missing_capture_candidate_tokens": 1}
    assert future["covered"] is False
    assert future["rejection_counts"] == {"no_captured_restore_at_common_prefix": 1}


def test_candidate_truncation_is_reported_and_forces_insufficient_data() -> None:
    events = _history_then_eval()
    events[0]["compatible_entry_count"] = 2
    result = replay_checkpoint_trace(
        _payload(events), min_events=1, min_bucket_events=1
    )
    assert result["status"] == "exploratory_configuration"
    assert result["exploratory_assessment"]["numeric_outcome"] == "insufficient_data"
    assert result["trace"]["truncated_candidate_event_count"] == 1
    assert "truncated" in " ".join(result["status_reasons"])


def test_terminal_rows_remain_workload_wide_zero_benefit_rows() -> None:
    events = _history_then_eval()
    events[6]["terminal_status"] = "cancelled"
    events[7]["terminal_status"] = "error"
    result = replay_checkpoint_trace(
        _payload(events), min_events=1, min_bucket_events=1
    )
    rows = {row["sequence"]: row for row in result["events"]}
    assert result["trace"]["fit_window_event_count"] == 7
    assert result["trace"]["fit_completed_event_count"] == 6
    assert result["trace"]["evaluation_window_event_count"] == 3
    assert result["trace"]["evaluation_completed_event_count"] == 2
    assert rows[8]["token_work_included"] is True
    assert rows[8]["zero_benefit_reason"] == "terminal request"
    assert rows[8]["policies"]["incumbent"]["work_tokens"] == 4096
    assert rows[8]["policies"]["observed_fitted"]["work_tokens"] == 4096
    assert result["policy_summaries"]["incumbent"]["coverage"]["total_events"] == 3
    assert result["trace"]["terminal_zero_benefit_rows"] == [
        {
            "sequence": 7,
            "bank_epoch": 1,
            "partition": "fit",
            "terminal_status": "cancelled",
            "cache_case": "hit",
            "reason": "terminal request is a workload-wide zero-benefit row",
        },
        {
            "sequence": 8,
            "bank_epoch": 1,
            "partition": "evaluation",
            "terminal_status": "error",
            "cache_case": "hit",
            "reason": "terminal request is a workload-wide zero-benefit row",
        },
    ]


@pytest.mark.parametrize(
    "replacement",
    [
        _candidate(1, stored=8192, retained=(256, 512, 8192)),
        _candidate(1, retained=(128, 512, 4096)),
        _candidate(1, capture=(512, 4096)),
        _candidate(1, retained=(4096,), capture=(4096,), recurrent=False),
    ],
    ids=(
        "stored-prefix-tokens",
        "retained-checkpoint-tokens",
        "capture-candidate-tokens",
        "has-recurrent",
    ),
)
def test_entry_revision_identity_drift_fails_closed_before_replay(
    replacement: dict[str, object],
) -> None:
    events = _history_then_eval()
    for event in events[7:]:
        event["candidates"] = [copy.deepcopy(replacement)]
        event["outcome"] = _event(
            int(event["sequence"]), candidates=[copy.deepcopy(replacement)]
        )["outcome"]
    with pytest.raises(
        ValueError,
        match="complete placement and restore contract across the trace",
    ):
        replay_checkpoint_trace(_payload(events), min_events=1, min_bucket_events=1)


def test_recurrent_fit_then_attention_only_evaluation_fails_closed() -> None:
    events = _history_then_eval()
    attention_only = _candidate(
        1,
        retained=(4096,),
        capture=(4096,),
        recurrent=False,
    )
    for event in events[7:]:
        event["candidates"] = [copy.deepcopy(attention_only)]
        event["outcome"] = _event(
            int(event["sequence"]), candidates=[copy.deepcopy(attention_only)]
        )["outcome"]
    with pytest.raises(
        ValueError,
        match="complete placement and restore contract across the trace",
    ):
        replay_checkpoint_trace(_payload(events), min_events=1, min_bucket_events=1)


def test_bank_epoch_regression_fails_closed() -> None:
    events = _history_then_eval()
    for event in events[:1]:
        event["bank_epoch"] = 1
    for event in events[1:7]:
        event["bank_epoch"] = 2
    for event in events[7:]:
        event["bank_epoch"] = 1
    payload = _payload(events)
    payload["bank_epoch"] = 2
    with pytest.raises(ValueError, match="bank_epoch must be nondecreasing"):
        replay_checkpoint_trace(payload, min_events=1, min_bucket_events=1)


def test_monotonic_bank_epoch_transition_remains_valid() -> None:
    events = _history_then_eval()
    for event in events[:7]:
        event["bank_epoch"] = 1
    for event in events[7:]:
        event["bank_epoch"] = 2
    payload = _payload(events)
    payload["bank_epoch"] = 2
    result = replay_checkpoint_trace(payload, min_events=1, min_bucket_events=1)
    observed = result["events"][0]["policies"]["observed_fitted"]
    assert observed["covered"] is False
    assert observed["rejection_counts"] == {"insufficient_fit_history": 1}
    assert set(result["trace"]["history_observations_by_epoch_and_entry_ordinal"]) == {
        "1:1"
    }


def test_endpoint_is_free_and_observed_budget_is_not_expanded() -> None:
    events = _history_then_eval(
        stored=10,
        retained=(6, 10),
        history=(2, 4, 8, 4, 8, 2, 4),
        evaluation=(8, 8, 8),
    )
    result = replay_checkpoint_trace(
        _payload(events), block_size=3, min_events=1, min_bucket_events=1
    )
    observed = result["events"][0]["policies"]["observed_fitted"]
    future_known = result["events"][0]["policies"]["future_known_upper"]
    assert observed["checkpoint_tokens"][-1] == 10
    assert observed["requested_budget"] == observed["planned_budget"] == 1
    assert future_known["checkpoint_tokens"][-1] == 10
    assert future_known["requested_budget"] == 1
    assert future_known["planned_budget"] == 1


def test_metrics_buckets_and_selected_entry_incumbent_sanity_are_reported() -> None:
    events = _history_then_eval(evaluation=(3840, 2000, 300))
    result = replay_checkpoint_trace(
        _payload(events), min_events=1, min_bucket_events=1
    )
    summary = result["policy_summaries"]["observed_fitted"]
    assert set(summary["checkpoint_loss_tokens"]) == {"total", "mean", "p50", "p95"}
    assert set(summary["target_suffix_tokens"]) == {"total", "mean", "p50", "p95"}
    assert set(result["nonappend_divergence_buckets"]) == {
        "<=256",
        "257-1024",
        "1025-8192",
        ">8192",
    }
    assert result["incumbent_sanity"] == {
        "eligible_ram_hits": 10,
        "unambiguous": 10,
        "verified": 10,
        "matches": 10,
        "mismatches": 0,
        "not_reported": 0,
        "impossible": 0,
    }


def test_pure_attention_provenance_uses_the_exact_common_prefix() -> None:
    events = _history_then_eval()
    for event in events:
        candidate = event["candidates"][0]
        candidate["has_recurrent"] = False
        common = candidate["common_prefix_tokens"]
        event["outcome"]["cached_tokens"] = common
        event["outcome"]["new_prefill_tokens"] = event["prompt_tokens"] - common

    result = replay_checkpoint_trace(
        _payload(events), min_events=1, min_bucket_events=1
    )
    incumbent = result["events"][0]["policies"]["incumbent"]
    observed = result["events"][0]["policies"]["observed_fitted"]

    assert result["incumbent_sanity"]["matches"] == 10
    assert incumbent["checkpoint_loss_tokens"] == 0
    assert incumbent["work_tokens"] == observed["work_tokens"] == 256
    assert incumbent["requested_budget"] == observed["requested_budget"] == 0


def test_gate_no_go_and_counterfactual_only_and_bootstrap_are_deterministic() -> None:
    no_go_events = _history_then_eval(
        retained=tuple(range(256, 4097, 256)),
        capture=tuple(range(256, 4097, 256)),
        history=(3840, 2000, 3840, 2000, 3840, 2000, 3840) * 34,
        evaluation=(3840, 2000) * 51,
    )
    no_go = replay_checkpoint_trace(_payload(no_go_events))
    assert no_go["status"] == "no_go"

    passing_events = _history_then_eval(
        capture=tuple(range(256, 4097, 256)),
        history=(3840, 2000, 3840, 2000, 3840, 2000, 3840) * 34,
        evaluation=(3840, 2000) * 51,
    )
    first = replay_checkpoint_trace(_payload(passing_events))
    second = replay_checkpoint_trace(_payload(copy.deepcopy(passing_events)))
    assert first["status"] == "counterfactual_only"
    assert first["canonical_gate"] is True
    assert first["exploratory_assessment"] is None
    assert first["runtime_go"] is False
    assert first["deployable_go_available"] is False
    assert "capture-candidate counterfactual" in first["labels"]["observed_fitted"]
    assert first["capture_candidate_counterfactual"] == {
        "policy": "observed_fitted",
        "payload_availability": (
            "unavailable by contract: selected capture positions need not remain "
            "among retained checkpoint payloads"
        ),
        "may_falsify": True,
        "may_authorize_go": False,
        "future_go_prerequisite": (
            "implement policy-before-capture, bounded retention, or measured "
            "rematerialization before a deployable GO gate can exist"
        ),
    }
    assert first["bootstrap"] == second["bootstrap"]
    assert first["bootstrap"]["observed_fitted"]["dependence_runs"] == 2
    assert first["bootstrap"]["observed_fitted"]["block_lengths"] == [5, 10]
    assert first["events"][0]["policies"]["incumbent"]["checkpoint_tokens"] == [
        256,
        512,
        4096,
    ]
    assert first["events"][0]["policies"]["observed_fitted"]["checkpoint_tokens"] == [
        1792,
        3840,
        4096,
    ]
    assert first["gate_config"] == {
        "effective_inputs": {
            "block_size": 256,
            "decay": 1.0,
            "min_events": 100,
            "min_bucket_events": 20,
            "bootstrap_iterations": 10_000,
            "bootstrap_block_length": None,
            "seed": 0,
            "chronological_fit_fraction": 0.70,
            "decision_threshold_percent": 10.0,
        },
        "derived": {
            "minimum_history_observations": 2,
            "minimum_populated_nonappend_buckets": 2,
            "bootstrap_lower_percentile": 0.05,
            "bootstrap_upper_percentile": 0.95,
            "fit_window_event_count": 238,
            "evaluation_window_event_count": 102,
            "completed_evaluation_event_count": 102,
            "workload_evaluation_event_count": 102,
            "bootstrap_event_count": 102,
            "bootstrap_base_block_length": 5,
            "bootstrap_block_lengths": [5, 10],
            "bootstrap_block_length_rule": (
                "ceil(workload_evaluation_event_count ** (1 / 3))"
            ),
            "bootstrap_dependence_runs": 2,
        },
        "canonical_defaults": {
            "block_size": 256,
            "decay": 1.0,
            "min_events": 100,
            "min_bucket_events": 20,
            "bootstrap_iterations": 10_000,
            "bootstrap_block_length": None,
            "seed": 0,
            "chronological_fit_fraction": 0.70,
            "decision_threshold_percent": 10.0,
        },
    }


def test_discarded_capture_counterfactual_cannot_emit_go_or_cli_success(
    monkeypatch, tmp_path, capsys
) -> None:
    events = _history_then_eval(
        retained=(256, 512, 4096),
        capture=tuple(range(256, 4097, 256)),
        history=(3840, 2000, 3840, 2000, 3840, 2000, 3840) * 34,
        evaluation=(3840, 2000) * 51,
    )
    result = replay_checkpoint_trace(_payload(events))
    observed = result["events"][0]["policies"]["observed_fitted"]

    assert observed["checkpoint_tokens"] == [1792, 3840, 4096]
    assert result["status"] == "counterfactual_only"
    assert result["status"] != "passes_cpu_falsifier"
    assert "unavailable" in result["labels"]["observed_fitted"]

    trace_path = tmp_path / "trace.json"
    trace_path.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        trace=str(trace_path),
        output=None,
        block_size=256,
        decay=1.0,
        min_events=100,
        min_bucket_events=20,
        bootstrap_iterations=10_000,
        bootstrap_block_length=None,
        seed=0,
    )
    monkeypatch.setattr(
        checkpoint_replay, "replay_checkpoint_trace", lambda *_args, **_kwargs: result
    )

    assert cli._cmd_checkpoint_replay(args) == 2
    assert '"status": "counterfactual_only"' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("overrides", "effective_key", "effective_value"),
    [
        ({"block_size": 128}, "block_size", 128),
        ({"decay": 0.9}, "decay", 0.9),
        ({"min_events": 1}, "min_events", 1),
        ({"min_bucket_events": 1}, "min_bucket_events", 1),
        ({"bootstrap_iterations": 10_001}, "bootstrap_iterations", 10_001),
        ({"bootstrap_block_length": 7}, "bootstrap_block_length", 7),
        ({"seed": 1}, "seed", 1),
    ],
)
def test_every_gate_override_is_exploratory_even_if_numeric_thresholds_clear(
    monkeypatch, overrides, effective_key, effective_value
) -> None:
    def fake_bootstrap(_rows, **_kwargs):
        return {
            "point_gain_vs_incumbent_percent": 15.0,
            "lower95_percent": 12.0,
            "upper95_percent": 18.0,
        }

    monkeypatch.setattr(checkpoint_replay, "_bootstrap", fake_bootstrap)
    events = _history_then_eval(
        capture=tuple(range(256, 4097, 256)),
        history=(3840, 2000, 3840, 2000, 3840, 2000, 3840) * 34,
        evaluation=(3840, 2000) * 51,
    )
    result = replay_checkpoint_trace(_payload(events), **overrides)

    assert result["canonical_gate"] is False
    assert result["status"] == "exploratory_configuration"
    assert result["exploratory_assessment"] == {
        "numeric_outcome": "counterfactual_only",
        "numeric_reasons": [
            "capture-candidate counterfactual lower95 gain clears 10%, but selected "
            "snapshot payloads may be unavailable; no deployable GO gate exists"
        ],
    }
    assert result["gate_config"]["effective_inputs"][effective_key] == effective_value


def test_weakened_ten_event_reproduction_is_exploratory_not_a_cpu_falsifier(
    monkeypatch,
) -> None:
    def fake_bootstrap(_rows, **_kwargs):
        return {
            "point_gain_vs_incumbent_percent": 15.0,
            "lower95_percent": 12.0,
            "upper95_percent": 18.0,
        }

    monkeypatch.setattr(checkpoint_replay, "_bootstrap", fake_bootstrap)
    result = replay_checkpoint_trace(
        _payload(_history_then_eval(capture=tuple(range(256, 4097, 256)))),
        min_events=1,
        min_bucket_events=1,
    )

    assert result["trace"]["events_total"] == 10
    assert result["canonical_gate"] is False
    assert result["status"] == "exploratory_configuration"
    assert result["exploratory_assessment"]["numeric_outcome"] == "counterfactual_only"


def test_incumbent_mismatches_fail_closed_even_when_the_token_work_would_pass() -> None:
    events = _history_then_eval(
        capture=tuple(range(256, 4097, 256)),
        history=(3840, 2000, 3840, 2000, 3840, 2000, 3840) * 34,
        evaluation=(3840, 2000) * 51,
    )
    assert replay_checkpoint_trace(_payload(events))["status"] == "counterfactual_only"

    for event in events:
        event["outcome"]["cached_tokens"] = 0
        event["outcome"]["new_prefill_tokens"] = event["prompt_tokens"]
    result = replay_checkpoint_trace(_payload(events))

    assert result["status"] == "insufficient_data"
    assert result["incumbent_sanity"] == {
        "eligible_ram_hits": 340,
        "unambiguous": 340,
        "verified": 0,
        "matches": 0,
        "mismatches": 340,
        "not_reported": 0,
        "impossible": 0,
    }
    assert result["bootstrap"]["observed_fitted"] is None
    assert result["bootstrap"]["future_known_upper"] is None
    assert "incumbent provenance" in " ".join(result["status_reasons"])


def test_bad_fit_partition_incumbent_provenance_blocks_counterfactual_only() -> None:
    events = _history_then_eval(
        capture=tuple(range(256, 4097, 256)),
        history=(3840, 2000, 3840, 2000, 3840, 2000, 3840) * 34,
        evaluation=(3840, 2000) * 51,
    )
    assert replay_checkpoint_trace(_payload(events))["status"] == "counterfactual_only"

    bad_fit_row = events[0]
    bad_fit_row["outcome"]["selected_entry_ordinal"] = None
    result = replay_checkpoint_trace(_payload(events))

    assert result["status"] == "insufficient_data"
    assert result["status"] != "counterfactual_only"
    assert result["incumbent_sanity"]["not_reported"] == 1
    assert result["trace"]["evaluation_ram_hit_provenance"]["verified"] == 102
    assert result["incumbent_provenance"]["gate_eligible"] is False
    assert result["bootstrap"]["observed_fitted"] is None
    assert "full trace" in " ".join(result["status_reasons"])


def test_unverified_eligible_ram_hits_cannot_mint_a_canonical_pass() -> None:
    events = _history_then_eval(
        capture=tuple(range(256, 4097, 256)),
        history=(3840, 2000, 3840, 2000, 3840, 2000, 3840) * 34,
        evaluation=(3840, 2000) * 51,
    )
    for event in events[-102:]:
        event["outcome"]["selected_entry_ordinal"] = None

    result = replay_checkpoint_trace(_payload(events))

    assert result["canonical_gate"] is True
    assert result["status"] == "insufficient_data"
    assert result["incumbent_provenance"] == {
        "completed_ram_hit_rows": 102,
        "verified_rows": 0,
        "minimum_verified_rows": 100,
        "unverified_rows": 102,
        "gate_eligible": False,
    }
    assert result["bootstrap"]["observed_fitted"] is None
    assert "without a selected incumbent" in " ".join(result["status_reasons"])


def test_unreported_ram_hit_blocks_even_when_it_has_no_fitted_placement() -> None:
    """A fallback row still participates in workload comparison, so it needs proof."""

    events = _history_then_eval(
        capture=tuple(range(256, 4097, 256)),
        history=(3840, 2000, 3840, 2000, 3840, 2000, 3840) * 34,
        evaluation=(3840, 2000) * 51,
    )
    unreported = events[-1]
    unreported["outcome"]["selected_entry_ordinal"] = None
    unreported["candidates"][0]["entry_ordinal"] = 2
    unreported["candidates"][0]["capture_candidate_tokens"] = [4096]

    result = replay_checkpoint_trace(_payload(events))

    assert result["canonical_gate"] is True
    assert result["status"] == "insufficient_data"
    assert result["trace"]["eligible_evaluation_event_count"] == 101
    assert result["incumbent_provenance"] == {
        "completed_ram_hit_rows": 102,
        "verified_rows": 101,
        "minimum_verified_rows": 100,
        "unverified_rows": 1,
        "gate_eligible": False,
    }
    assert result["bootstrap"]["observed_fitted"] is None


def test_impossible_ram_restore_cannot_mint_a_canonical_pass() -> None:
    events = _history_then_eval(
        capture=tuple(range(256, 4097, 256)),
        history=(3840, 2000, 3840, 2000, 3840, 2000, 3840) * 34,
        evaluation=(3840, 2000) * 51,
    )
    events[-1]["candidates"][0]["structurally_restorable"] = False

    result = replay_checkpoint_trace(_payload(events))

    assert result["status"] == "insufficient_data"
    assert result["incumbent_provenance"]["unverified_rows"] == 1
    assert result["trace"]["evaluation_ram_hit_provenance"]["impossible"] == 1
    assert result["bootstrap"]["observed_fitted"] is None


def test_ssd_only_workload_cannot_satisfy_the_ram_checkpoint_gate() -> None:
    events = _history_then_eval(
        capture=tuple(range(256, 4097, 256)),
        history=(3840, 2000, 3840, 2000, 3840, 2000, 3840) * 34,
        evaluation=(3840, 2000) * 51,
    )
    for event in events:
        event["outcome"]["selected_entry_ordinal"] = None
        event["outcome"]["cache_source"] = "ssd"

    result = replay_checkpoint_trace(_payload(events))

    assert result["status"] == "insufficient_data"
    assert result["trace"]["eligible_evaluation_event_count"] == 0
    assert result["incumbent_provenance"]["completed_ram_hit_rows"] == 0
    assert result["bootstrap"]["observed_fitted"] is None


def test_request_start_sequence_sorts_out_of_order_finalization_before_split() -> None:
    events = [
        _event(
            sequence,
            candidates=[_candidate(1 if sequence <= 7 else 2, common=2048)],
            selected_entry_ordinal=1 if sequence <= 7 else 2,
        )
        for sequence in range(1, 11)
    ]
    # A concurrent producer may finalize in this order, but `sequence` identifies
    # request start.  Candidate 2 must remain entirely in evaluation.
    result = replay_checkpoint_trace(
        _payload(list(reversed(events))), min_events=1, min_bucket_events=1
    )

    assert [row["sequence"] for row in result["events"]] == [8, 9, 10]
    assert result["trace"]["fit_window_event_count"] == 7
    assert result["trace"]["evaluation_window_event_count"] == 3
    assert result["events"][0]["policies"]["observed_fitted"]["rejection_counts"] == {
        "insufficient_fit_history": 1
    }


def test_duplicate_request_start_sequence_fails_closed() -> None:
    events = _history_then_eval()
    events[1]["sequence"] = events[0]["sequence"]
    with pytest.raises(ValueError, match="must exactly cover"):
        replay_checkpoint_trace(_payload(events))


def test_gapped_request_start_sequence_fails_closed() -> None:
    events = _history_then_eval()
    events[-1]["sequence"] += 1
    with pytest.raises(ValueError, match="must exactly cover"):
        replay_checkpoint_trace(_payload(events))


def test_cli_supplied_bootstrap_block_is_the_base_length() -> None:
    result = replay_checkpoint_trace(
        _payload(
            _history_then_eval(
                capture=tuple(range(256, 4097, 256)),
                evaluation=(3840, 2000, 3840),
            )
        ),
        min_events=1,
        min_bucket_events=1,
        bootstrap_block_length=7,
    )
    assert result["bootstrap"]["observed_fitted"]["block_lengths"] == [7, 14]


def test_amber_uses_capture_counterfactual_lower_bound_not_ideal_upper(
    monkeypatch,
) -> None:
    def fake_bootstrap(_rows, *, policy, **_kwargs):
        if policy == "observed_fitted":
            return {
                "point_gain_vs_incumbent_percent": 12.0,
                "lower95_percent": 9.0,
                "upper95_percent": 18.0,
            }
        return {
            "point_gain_vs_incumbent_percent": 20.0,
            "lower95_percent": 15.0,
            "upper95_percent": 25.0,
        }

    monkeypatch.setattr(checkpoint_replay, "_bootstrap", fake_bootstrap)
    result = replay_checkpoint_trace(
        _payload(
            _history_then_eval(
                capture=tuple(range(256, 4097, 256)),
                evaluation=(3840, 2000, 3840),
            )
        ),
        min_events=1,
        min_bucket_events=1,
    )
    assert result["status"] == "exploratory_configuration"
    assert result["exploratory_assessment"]["numeric_outcome"] == "amber"
    assert result["runtime_go"] is False


def test_dropped_events_reject_incomplete_evidence_before_status() -> None:
    with pytest.raises(ValueError, match="events_dropped must equal zero"):
        replay_checkpoint_trace(_payload(_history_then_eval(), dropped=1))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("schema_version", 1),
        lambda payload: payload.pop("events_dropped"),
        lambda payload: payload.pop("pending_probe_count"),
        lambda payload: payload.pop("sequence_high_watermark"),
        lambda payload: payload["events"][0].pop("candidates"),
        lambda payload: payload["events"][0]["candidates"][0].pop(
            "common_prefix_tokens"
        ),
        lambda payload: payload["events"][0]["candidates"][0].__setitem__(
            "incumbent_interior_budget", 99
        ),
        lambda payload: payload["events"][0]["candidates"][0].__setitem__(
            "retained_checkpoint_tokens", [256, 512]
        ),
        lambda payload: payload["events"][0]["outcome"].__setitem__(
            "selected_entry_ordinal", 999
        ),
        lambda payload: payload["events"][0]["outcome"].__setitem__(
            "cached_tokens", 9999
        ),
        lambda payload: payload["events"][0].__setitem__("bank_consulted", "yes"),
        lambda payload: payload["events"][0].__setitem__("lane", "cohort_local"),
        lambda payload: payload["events"][0]["outcome"].__setitem__(
            "restore_kind", "unrecognized"
        ),
    ],
)
def test_malformed_producer_contract_fails_closed(mutate) -> None:
    payload = _payload(_history_then_eval())
    mutate(payload)
    with pytest.raises((TypeError, ValueError)):
        replay_checkpoint_trace(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("enabled", False),
        lambda payload: payload.__setitem__("max_events", 1),
        lambda payload: payload.__setitem__("pending_probe_count", 1),
        lambda payload: payload.__setitem__(
            "sequence_high_watermark", len(payload["events"]) + 1
        ),
        lambda payload: payload.__setitem__(
            "events_collected", len(payload["events"]) - 1
        ),
        lambda payload: payload.__setitem__(
            "events_collected", len(payload["events"]) + 1
        ),
        lambda payload: payload.__setitem__("bank_epoch", 0),
        lambda payload: payload["events"][0].__setitem__(
            "candidates", [_candidate(index + 1) for index in range(17)]
        ),
    ],
)
def test_complete_producer_root_contract_fails_closed(mutate) -> None:
    payload = _payload(_history_then_eval())
    mutate(payload)
    with pytest.raises((TypeError, ValueError)):
        replay_checkpoint_trace(payload)


def test_root_contract_rejects_impossible_max_events_snapshot() -> None:
    events = [copy.deepcopy(event) for _ in range(34) for event in _history_then_eval()]
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
    payload = _payload(events)
    payload["max_events"] = 1

    with pytest.raises(ValueError, match="fit within payload.max_events"):
        replay_checkpoint_trace(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: event["outcome"].__setitem__(
            "new_prefill_tokens", event["prompt_tokens"] - 1
        ),
        lambda event: event.__setitem__("bank_consulted", False),
        lambda event: event["outcome"].__setitem__("cache_source", "other"),
        lambda event: event["outcome"].__setitem__("cache_source", "ssd"),
    ],
)
def test_completed_outcome_combinations_fail_closed(mutate) -> None:
    payload = _payload(_history_then_eval())
    event = payload["events"][0]
    mutate(event)
    if event["outcome"]["cache_source"] == "ssd":
        # A cold-tier hit cannot name the pre-probe RAM candidate.
        event["outcome"]["selected_entry_ordinal"] = 1
    with pytest.raises((TypeError, ValueError)):
        replay_checkpoint_trace(payload)


def test_cancelled_outcome_is_conservatively_zero_benefit_without_exact_sum() -> None:
    events = _history_then_eval()
    events[7]["terminal_status"] = "cancelled"
    events[7]["outcome"]["cached_tokens"] = 100
    events[7]["outcome"]["new_prefill_tokens"] = 100
    result = replay_checkpoint_trace(
        _payload(events), min_events=1, min_bucket_events=1
    )

    row = {item["sequence"]: item for item in result["events"]}[8]
    assert row["zero_benefit_reason"] == "terminal request"
    assert row["policies"]["incumbent"]["work_tokens"] == 4096


def test_schema_privacy_irrelevant_values_are_neither_required_nor_returned() -> None:
    payload = _payload(_history_then_eval())
    payload["events"][0]["unknown_secret"] = "do-not-export"
    payload["events"][0]["timings"]["secret"] = "ignored"
    result = replay_checkpoint_trace(payload)
    assert "do-not-export" not in repr(result)
    assert "ignored" not in repr(result)


def test_no_mlx_import() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mtplx.checkpoint_replay; "
            "assert not any(name == 'mlx' or name.startswith('mlx.') for name in sys.modules)",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_random_small_fitted_placement_matches_bruteforce() -> None:
    rng = random.Random(5)
    for _ in range(12):
        stored = rng.choice((6, 8, 10))
        block = rng.choice((1, 2, 3))
        legal = tuple(range(block, ((stored - 1) // block) * block + 1, block))
        if not legal:
            continue
        budget = rng.randint(1, len(legal))
        retained = (*legal[:budget], stored)
        history = tuple(rng.randint(legal[0], stored) for _ in range(7))
        events = _history_then_eval(
            stored=stored,
            retained=retained,
            capture=(*legal, stored),
            history=history,
            evaluation=(history[-1],) * 3,
        )
        result = replay_checkpoint_trace(
            _payload(events), block_size=block, min_events=1, min_bucket_events=1
        )
        expected = _bruteforce_transformed(
            history, stored=stored, budget=budget, legal=(*legal, stored)
        )
        assert result["events"][0]["policies"]["observed_fitted"][
            "checkpoint_tokens"
        ] == [
            *expected,
            stored,
        ]
