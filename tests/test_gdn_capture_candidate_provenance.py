from __future__ import annotations

from types import SimpleNamespace

import mtplx.generation as generation
from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SessionBank, SessionBankEntry


def _entry(
    token_ids: list[int],
    *,
    has_recurrent: bool = False,
    capture_candidate_tokens: list[int] | None = None,
    gdn_boundaries: list[tuple[int, object, object]] | None = None,
) -> SessionBankEntry:
    return SessionBankEntry(
        token_ids=tuple(token_ids),
        token_hash="test",
        model_path="/unit/model",
        mtp_enabled=False,
        hidden_variant=None,
        cache_snapshot=CacheSnapshot(states=(), meta_states=()),
        logits=None,
        hidden=None,
        has_recurrent=has_recurrent,
        capture_candidate_tokens=list(capture_candidate_tokens or []),
        gdn_boundaries=list(gdn_boundaries or []),
    )


def _begin(bank: SessionBank, token_ids: list[int]) -> dict[str, object] | None:
    return bank.begin_overlap_probe(
        token_ids,
        lane="solo_ar",
        model_path="/unit/model",
        mtp_enabled=False,
        hidden_variant=None,
        template_hash=None,
        mtp_history_policy=None,
        draft_head_identity=None,
        policy_fingerprint=None,
    )


def test_capture_candidates_keep_successful_pre_thinning_positions(
    monkeypatch,
) -> None:
    boundaries: list[tuple[int, object, object]] = []
    capture_candidates: list[int] = []
    monkeypatch.setattr(
        generation,
        "snapshot_untrimmable_cache",
        lambda _cache: CacheSnapshot(states=(), meta_states=()),
    )
    monkeypatch.setattr(generation, "_gdn_boundary_max_count", lambda: 2)

    for position in (128, 256, 384):
        generation._capture_gdn_boundary(
            boundaries,
            position,
            [],
            capture_candidate_sink=capture_candidates,
        )

    assert [record[0] for record in boundaries] != [128, 256, 384]
    assert capture_candidates == [128, 256, 384]


def test_failed_capture_does_not_report_candidate(monkeypatch) -> None:
    boundaries: list[tuple[int, object, object]] = []
    capture_candidates: list[int] = []

    def fail_snapshot(_cache: object) -> CacheSnapshot:
        raise RuntimeError("capture failure")

    monkeypatch.setattr(generation, "snapshot_untrimmable_cache", fail_snapshot)

    generation._capture_gdn_boundary(
        boundaries,
        256,
        [],
        capture_candidate_sink=capture_candidates,
    )

    assert boundaries == []
    assert capture_candidates == []


def test_capture_candidate_metadata_is_default_off_and_enabled_trace_keeps_it(
    monkeypatch,
) -> None:
    """Replay-only provenance is absent unless the bounded trace is enabled."""

    snapshots: list[object] = []
    monkeypatch.setattr(
        generation,
        "snapshot_untrimmable_cache",
        lambda cache: (
            snapshots.append(cache) or CacheSnapshot(states=(), meta_states=())
        ),
    )

    disabled = SessionBank(overlap_trace_max_events=0)
    disabled_sink = generation._new_gdn_capture_candidate_sink(
        disabled,
        vision_splice=None,
    )
    generation._capture_gdn_boundary(
        disabled_sink,
        128,
        [],
        capture_candidate_sink=disabled_sink,
    )

    assert disabled_sink is None
    assert snapshots == []

    enabled = SessionBank(overlap_trace_max_events=2)
    enabled_sink = generation._new_gdn_capture_candidate_sink(
        enabled,
        vision_splice=None,
    )
    assert enabled_sink == []
    generation._capture_gdn_boundary(
        [],
        128,
        [],
        capture_candidate_sink=enabled_sink,
    )

    assert snapshots == [[]]
    assert enabled_sink == [128]


def test_warm_capture_candidate_inheritance_clips_to_restore_point() -> None:
    entry = _entry([1, 2, 3, 4, 5, 6], capture_candidate_tokens=[2, 4, 6])

    assert generation._inherited_gdn_capture_candidates(entry, 4) == [2, 4]


def test_put_inherits_capture_candidates_from_same_key_and_prefix_donor() -> None:
    bank = SessionBank()
    runtime = SimpleNamespace(model_path="/unit/model", mtp_enabled=False)

    first = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3, 4],
        cache=[],
        logits=None,
        hidden=None,
        capture_candidate_tokens=[2, 4],
    )
    assert first is not None
    same_key = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3, 4],
        cache=[],
        logits=None,
        hidden=None,
    )
    assert same_key is not None
    assert same_key.capture_candidate_tokens == [2, 4]

    extended = bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3, 4, 5, 6],
        cache=[],
        logits=None,
        hidden=None,
    )
    assert extended is not None
    assert extended.capture_candidate_tokens == [2, 4]


def test_telemetry_reads_capture_candidates_and_leaves_ambiguous_ram_hit_unknown() -> (
    None
):
    bank = SessionBank(overlap_trace_max_events=2)
    bank._set_entry_for_overlap_trace(
        (1, 2, 3, 4),
        _entry([1, 2, 3, 4], capture_candidate_tokens=[2, 3]),
    )
    bank._set_entry_for_overlap_trace(
        (1, 2, 3, 5),
        _entry([1, 2, 3, 5], capture_candidate_tokens=[2, 3]),
    )

    probe = _begin(bank, [1, 2, 3, 9])
    assert probe is not None
    assert probe["candidates"][0]["capture_candidate_tokens"] == [2, 3]  # type: ignore[index]
    assert bank.finalize_overlap_probe(
        probe,
        terminal_status="completed",
        bank_consulted=True,
        outcome={
            "cache_hit": True,
            "cache_source": "ram",
            "cached_tokens": 3,
        },
    )

    event = bank.overlap_trace_snapshot()["events"][0]
    assert event["outcome"]["selected_entry_ordinal"] is None


def test_telemetry_infers_only_a_unique_compatible_ram_entry() -> None:
    bank = SessionBank(overlap_trace_max_events=2)
    bank._set_entry_for_overlap_trace(
        (1, 2, 3, 4),
        _entry([1, 2, 3, 4], capture_candidate_tokens=[2, 3]),
    )

    probe = _begin(bank, [1, 2, 3, 9])
    assert probe is not None
    ordinal = probe["candidates"][0]["entry_ordinal"]  # type: ignore[index]
    assert bank.finalize_overlap_probe(
        probe,
        terminal_status="completed",
        bank_consulted=True,
        outcome={
            "cache_hit": True,
            "cache_source": "ram",
            "cached_tokens": 3,
        },
    )

    event = bank.overlap_trace_snapshot()["events"][0]
    assert event["outcome"]["selected_entry_ordinal"] == ordinal
