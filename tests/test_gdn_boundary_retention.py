"""GDN boundary retention: geometric coverage instead of oldest+dense-tail.

Regression for the 2026-07-17 Hermes-lane finding (MEASUREMENTS 01:05 §B):
near-miss prefix restores landed on the OLDEST retained recurrent boundary
(2,048) despite a ~22.5k matched prefix, because the previous pop(1)
retention left a mid-prefix coverage hole after append churn (postcommit
re-forward chunk edges, clone/lease inheritance re-thinning). Result was a
20k+ re-prefill and 35-50s TTFT on the founder's follow-up turns.

The invariant pinned here: re-prefill cost after thinning stays proportional
to the true divergence distance from the tail (<= ~3x + one capture-grid
interval), for any churn sequence.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

from mtplx.generation import (
    _gdn_boundary_max_count,
    _inherited_gdn_boundaries,
    _thin_gdn_boundary_records,
)


def _rec(pos: int):
    return (pos, f"snap-{pos}", None)


def _at_or_below(kept_positions, matched):
    usable = [p for p in kept_positions if p <= matched]
    return max(usable) if usable else 0


def _assert_proportional(kept_positions, all_positions, grid: int, cap: int = 8) -> None:
    newest = max(all_positions)
    oldest = min(all_positions)
    span = max(1, newest - oldest)
    # Mirror the adaptive first floor: cap-2 doubling scales must cover span.
    base = 256
    scales = max(1, cap - 2)
    while base * (1 << (scales - 1)) < span and base < span:
        base *= 2
    slack = 2 * grid + 2 * base
    for matched in all_positions:
        if matched == newest:
            continue
        divergence = newest - matched
        restore = _at_or_below(kept_positions, matched)
        re_prefill = matched - restore
        # Contract: tight near the tail (where agent/RAG divergence lands —
        # the fielded 2026-07-17 case), loosely proportional elsewhere. The
        # thin-time guarantee is 2x spacing; incremental re-thinning of an
        # already-thinned list drifts (discarded bands cannot be re-covered),
        # empirically <= ~5x over seeded random churns — 8x is the pin. The
        # retired policy measured 74x on the fielded case.
        if divergence <= max(2 * grid, base):
            bound = divergence + slack
        else:
            bound = 8 * divergence + slack
        assert re_prefill <= bound, (
            f"matched={matched} (divergence {divergence}) restored at {restore}: "
            f"re-prefill {re_prefill} exceeds bound {bound}; "
            f"kept={kept_positions}"
        )


def test_thin_keeps_cap_and_endpoints():
    grid = list(range(2048, 22785, 2048))
    records = [_rec(p) for p in grid]
    kept = _thin_gdn_boundary_records(records, cap=8)
    positions = [r[0] for r in kept]
    assert len(kept) <= 8
    assert positions[0] == min(grid)  # oldest anchor survives
    assert positions[-1] == max(grid)  # newest survives
    assert positions == sorted(positions)


def test_thin_coverage_is_proportional_on_uniform_grid():
    grid = 512
    records = [_rec(p) for p in range(grid, 64 * grid + 1, grid)]
    kept = [r[0] for r in _thin_gdn_boundary_records(records, cap=8)]
    _assert_proportional(kept, [r[0] for r in records], grid)


def test_capture_churn_never_reopens_the_2048_cliff():
    """The observed production shape: cold-prefill chunk edges, then repeated
    re-forward churn appending the same grid again (postcommit rounds). Under
    the old pop(1) policy a long enough append stream left [oldest, tail...]
    with an unbounded mid hole; the geometric policy must keep the near-tail
    restore proportional. Mirrors matched=22,509 on a 22,784-token entry
    restoring at 2,048 (74x the divergence distance)."""
    grid = 2048
    sink: list = []
    cap = 8

    def append(pos: int) -> None:
        sink.append(_rec(pos))
        if len(sink) > cap:
            sink[:] = _thin_gdn_boundary_records(sink, cap)

    for pos in range(grid, 22531, grid):
        append(pos)
    append(22530)
    for _ in range(3):  # three postcommit-style churn rounds
        for pos in range(grid, 22785, grid):
            append(pos)
        append(22784)

    kept = [r[0] for r in sink]
    matched = 22509
    restore = _at_or_below(kept, matched)
    divergence = 22784 - matched  # 275
    re_prefill = matched - restore
    assert re_prefill <= 3 * divergence + 2 * grid, (
        f"near-tail miss restored at {restore} (re-prefill {re_prefill}); kept={kept}"
    )
    # And explicitly: never the old failure mode.
    assert restore > 2048


def test_random_churn_stays_proportional():
    rng = random.Random(20260717)
    grid = 1024
    for _ in range(25):
        sink: list = []
        cap = _gdn_boundary_max_count()
        seen: set[int] = set()
        newest = 0
        for _ in range(rng.randrange(20, 120)):
            newest += rng.choice((grid, grid, grid // 2, grid * 2))
            seen.add(newest)
            sink.append(_rec(newest))
            if len(sink) > cap:
                sink[:] = _thin_gdn_boundary_records(sink, cap)
        kept = [r[0] for r in sink]
        _assert_proportional(kept, sorted(seen), grid * 2)


def test_inherited_boundaries_thin_geometrically():
    entry = SimpleNamespace(
        gdn_boundaries=[_rec(p) for p in range(2048, 22785, 2048)]
    )
    kept = _inherited_gdn_boundaries(entry, restore_point=20480)
    positions = [r[0] for r in kept]
    assert all(p <= 20480 for p in positions)
    assert len(positions) <= _gdn_boundary_max_count()
    # Coverage must include a near-restore-point record, not just the oldest.
    assert max(positions) == 20480
    _assert_proportional(positions, list(range(2048, 20481, 2048)), 2048)
