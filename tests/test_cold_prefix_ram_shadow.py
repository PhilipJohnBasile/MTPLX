"""Resident-duplicate shadowing of cold prefix lookups.

The 2026-08-06 causal probe pair measured 0.66-1.17s of unattributed
prompt-state wall per warm turn: near_prefix_candidates() called the cold
tier's lookup_prefix_boundary() unconditionally, which fully hydrated
(_restore_row) a candidate that then LOST the combined sort to its own
RAM-resident twin. The fix shadows ONLY a serve-equivalent resident
duplicate of the cold metadata candidate — same token hash and stored
length, identity-compatible with the request, snapshot-capable, boundary-
covered, and matching the row's committed-MTP coverage. Raw RAM matches
are never a floor: generation may later reject them, and an ineligible
RAM match must never suppress a valid cold candidate or cold-only
recovery.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SessionBank
from mtplx.cache_bank import SessionBankColdTier


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)


# 41-token stored prefix; the prompt shares its first 40 tokens then
# diverges -> matched=40, gap=1: the tiny-gap near-prefix case the live
# receipts showed (matched == prefix_len routes to the EXACT path instead
# and the near-prefix loop rejects it, so a gap-0 fixture would test
# nothing).
PREFIX = list(range(1, 42))
PROMPT = tuple(PREFIX[:40] + [99, 98])
MODEL = str(Path("models/example"))


def _tier(tmp_path) -> SessionBankColdTier:
    return SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        min_prefix_tokens=2,
    )


def _bank(cold) -> SessionBank:
    return SessionBank(
        max_entries=8, max_bytes=4096, per_session_max_bytes=4096, cold_tier=cold
    )


def _put(bank: SessionBank, token_ids, *, template_hash=None, mtp_snapshot=None):
    return bank.put_snapshot(
        runtime=_runtime(),
        token_ids=token_ids,
        cache_snapshot=CacheSnapshot(states=(), meta_states=()),
        logits=None,
        hidden=None,
        template_hash=template_hash,
        mtp_history_snapshot=mtp_snapshot,
        snapshot_epoch=len(token_ids),
        nbytes_override=128,
    )


def _spy_restore_row(cold: SessionBankColdTier, monkeypatch) -> list:
    calls: list = []
    original = cold._restore_row

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cold, "_restore_row", spy)
    return calls


def _candidates(bank: SessionBank, **overrides):
    kwargs = dict(
        min_matched_tokens=4,
        block_size=8,
        block_min_matched_tokens=8,
        model_path=MODEL,
        mtp_enabled=True,
    )
    kwargs.update(overrides)
    return bank.near_prefix_candidates(PROMPT, **kwargs)


def test_resident_duplicate_shadows_cold_without_hydration(tmp_path, monkeypatch):
    """The measured causal case: the same put resident in RAM and on SSD."""
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        _put(bank, PREFIX)
        assert cold.flush(timeout_s=5.0) is True
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank)
        assert matches and matches[0][1] == 40
        assert str(getattr(matches[0][0], "cache_source", "ram") or "ram") == "ram"
        assert calls == [], "resident-duplicate lookup must not hydrate from SSD"
        assert cold.stats().get("prefix_lookups_shadowed_by_ram") == 1
    finally:
        cold.close()


def test_identity_incompatible_higher_ram_match_does_not_shadow_valid_cold(
    tmp_path, monkeypatch
):
    """Regression guard: a raw-higher but request-incompatible RAM entry
    must never floor out the valid SSD candidate."""
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        _put(bank, PREFIX, template_hash="template-a")
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()
        # Higher raw match (48 tokens of the prompt region... longer stored
        # prefix), but stored under a DIFFERENT template identity.
        longer = list(range(1, 41)) + [99, 98, 97, 96, 95, 94, 93, 92]
        _put(bank, longer, template_hash="template-other")
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank, template_hash="template-a")
        # The valid SSD candidate must still hydrate and be present.
        assert calls == [1], "valid cold candidate must hydrate despite raw RAM match"
        ssd = [m for m in matches if getattr(m[0], "cache_source", None) == "ssd"]
        assert ssd and ssd[0][1] == 40
        assert not cold.stats().get("prefix_lookups_shadowed_by_ram")
    finally:
        cold.close()


def test_coverage_missing_ram_twin_does_not_shadow_mtp_bearing_cold_row(
    tmp_path, monkeypatch
):
    """Same tokens resident in RAM but WITHOUT committed-MTP history must not
    shadow a cold row that carries it."""
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        _put(
            bank,
            PREFIX,
            mtp_snapshot=CacheSnapshot(states=(), meta_states=()),
        )
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()
        # Re-put the SAME tokens RAM-only (tier detached) without MTP history.
        bank.cold_tier = None
        _put(bank, PREFIX)
        bank.cold_tier = cold
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank)
        assert calls == [1], "mtp-bearing cold row must hydrate past an uncovered twin"
        assert any(getattr(m[0], "cache_source", None) == "ssd" for m in matches)
    finally:
        cold.close()


def test_epoch_desynced_ram_twin_does_not_shadow_valid_cold(tmp_path, monkeypatch):
    """Regression guard: restore() rejects entries whose MTP snapshot epoch
    desynced from the trunk epoch; such a twin must not shadow the valid
    SSD copy. (put_snapshot forbids creating desync, so mutate the resident
    entry the way a production defect would present.)"""
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        _put(bank, PREFIX, mtp_snapshot=CacheSnapshot(states=(), meta_states=()))
        assert cold.flush(timeout_s=5.0) is True
        entry = next(iter(bank._entries.values()))
        entry.mtp_snapshot_epoch = int(entry.snapshot_epoch) + 1
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank)
        assert calls == [1], "desynced twin must not suppress the valid cold row"
        assert any(getattr(m[0], "cache_source", None) == "ssd" for m in matches)
        assert not cold.stats().get("prefix_lookups_shadowed_by_ram")
    finally:
        cold.close()


def test_recurrent_boundary_below_min_restore_does_not_shadow_valid_cold(
    tmp_path, monkeypatch
):
    """Regression guard: generation rejects a recurrent candidate whose
    achievable boundary is <= min_restore_tokens; such a RAM match must not
    shadow a valid cold candidate."""
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        _put(bank, PREFIX)  # valid 40-token SSD row
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()
        # RAM-only recurrent entry: 50 stored tokens sharing 40 with the
        # prompt (gap 10 > tiny limit 8 -> boundary path), only boundary at 8.
        bank.cold_tier = None
        _put(bank, list(range(1, 51)))
        bank.cold_tier = cold
        entry = next(iter(bank._entries.values()))
        entry.has_recurrent = True
        entry.gdn_boundaries = [(8, CacheSnapshot(states=(), meta_states=()), None)]
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank, min_restore_tokens=10)
        assert calls == [1], (
            "achievable-boundary-below-floor RAM match must not suppress cold"
        )
        assert any(getattr(m[0], "cache_source", None) == "ssd" for m in matches)
        assert not cold.stats().get("prefix_lookups_shadowed_by_ram")
    finally:
        cold.close()


def test_cold_only_recovery_still_hydrates(tmp_path, monkeypatch):
    cold = _tier(tmp_path)
    try:
        bank = _bank(cold)
        _put(bank, PREFIX)
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()
        calls = _spy_restore_row(cold, monkeypatch)
        matches = _candidates(bank)
        assert matches and matches[0][1] == 40
        assert getattr(matches[0][0], "cache_source", None) == "ssd"
        assert calls == [1]
    finally:
        cold.close()


def test_duck_tier_without_capability_marker_gets_pre_shadow_call(tmp_path):
    seen: list = []

    def legacy_lookup(
        tokens,
        *,
        model_path,
        mtp_enabled,
        hidden_variant=None,
        template_hash=None,
        mtp_history_policy=None,
        draft_head_identity=None,
        policy_fingerprint=None,
        max_token_gap=8,
        min_matched_tokens=64,
        block_size=256,
        block_min_matched_tokens=512,
        allow_block_prefix=True,
    ):
        seen.append(tokens)
        return None

    bank = SessionBank(
        max_entries=8,
        max_bytes=4096,
        per_session_max_bytes=4096,
        cold_tier=SimpleNamespace(lookup_prefix_boundary=legacy_lookup),
    )
    _put(bank, PREFIX)
    matches = _candidates(bank)
    # No capability marker -> the shadow kwarg is never passed (no TypeError,
    # no retry) and RAM still serves.
    assert len(seen) == 1
    assert matches and matches[0][1] == 40
