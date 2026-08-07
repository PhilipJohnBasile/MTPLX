"""min_useful_matched_tokens gate on the cold-tier prefix lookup (2026-08-07).

A cold candidate whose matched length is strictly below the caller's best
RAM match can never win the caller's (matched, prefix_len) sort — hydrating
it is a multi-GB request-path disk read discarded unread. The gate skips the
hydration before _restore_row; ties keep the resident-duplicate-shadow
semantics; ram_best=0 (cold-only recovery after restart) is unaffected.
"""

from __future__ import annotations

import mlx.core as mx

from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot


def _entry(tokens, session="s1"):
    class Entry:
        token_ids = tuple(tokens)
        nbytes = 2048
        cache_snapshot = CacheSnapshot(
            states=[mx.zeros((1, 2, 8, 4), dtype=mx.float16)],
            meta_states=[{"offset": len(tokens)}],
        )
        logits = mx.zeros((1, 8), dtype=mx.float16)
        hidden = mx.zeros((1, 8), dtype=mx.float16)
        mtp_history_snapshot = None
        gdn_boundaries = ()
        has_recurrent = False
        session_id = session
        token_hash = f"hash-{len(tokens):04d}" * 2
        prefix_len = len(tokens)
        model_path = "model"
        mtp_enabled = False
        hidden_variant = None
        template_hash = None
        mtp_history_policy = None
        policy_fingerprint = None

    return Entry()


def _tier(tmp_path):
    return SessionBankColdTier(
        base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=1
    )


def _store(tier, tokens):
    assert tier.put_entry(_entry(tokens), capabilities=["ar_insert"]) is True
    import time

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if tier.stats()["writes_completed"] >= 1:
            return
        time.sleep(0.05)
    raise AssertionError("writer did not complete")


def _lookup(tier, tokens, **kw):
    return tier.lookup_prefix_boundary(
        tokens,
        model_path="model",
        mtp_enabled=False,
        max_token_gap=8,
        min_matched_tokens=8,
        block_size=16,
        block_min_matched_tokens=16,
        **kw,
    )


def test_gate_skips_hydration_when_ram_is_better(tmp_path):
    tier = _tier(tmp_path)
    stored = list(range(600))
    _store(tier, stored)
    query = tuple(stored[:596] + [9999, 9998, 9997, 9996])
    # Without the gate: hydrates (matched 596 via near/block prefix).
    assert _lookup(tier, query) is not None
    # RAM already matches more: skip entirely, distinct miss reason.
    assert _lookup(tier, query, min_useful_matched_tokens=597) is None
    stats = tier.stats()
    assert stats["last_miss_reason"] == "ssd_prefix_not_better_than_ram"
    assert stats["prefix_lookups_not_better_than_ram"] >= 1


def test_gate_allows_equal_and_better(tmp_path):
    tier = _tier(tmp_path)
    stored = list(range(600))
    _store(tier, stored)
    query = tuple(stored[:596] + [9999, 9998, 9997, 9996])
    hit = _lookup(tier, query, min_useful_matched_tokens=100)
    assert hit is not None
    matched = int(getattr(hit, "matched_tokens", 0) or 0)
    assert matched > 100
    # Equality falls through the gate (twin-shadow semantics own ties).
    assert _lookup(tier, query, min_useful_matched_tokens=matched) is not None


def test_gate_zero_is_todays_behavior(tmp_path):
    tier = _tier(tmp_path)
    stored = list(range(600))
    _store(tier, stored)
    query = tuple(stored[:596] + [9999, 9998, 9997, 9996])
    assert _lookup(tier, query, min_useful_matched_tokens=0) is not None
    assert tier.SUPPORTS_MIN_USEFUL_MATCHED_TOKENS is True
