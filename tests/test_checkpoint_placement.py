"""Exactness and boundary tests for the pure-CPU sparse checkpoint planner."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
import random
import subprocess
import sys

import pytest

from mtplx.checkpoint_placement import (
    ExponentialOverlapHistogram,
    balanced_checkpoint_positions,
    expected_recompute,
    geometric_tail_checkpoint_positions,
    normalize_overlap_weights,
    optimal_checkpoint_positions,
)


def _legal_positions(length: int, block_size: int) -> tuple[int, ...]:
    positions = list(range(block_size, length + 1, block_size))
    if not positions or positions[-1] != length:
        positions.append(length)
    return tuple(positions)


def _brute_force(
    weights: tuple[int, ...], budget: int, block_size: int = 1
) -> tuple[int, tuple[int, ...]]:
    """Return the exact integer-cost winner under production tie semantics."""
    legal = _legal_positions(len(weights), block_size)
    count = min(max(budget, 0), len(legal))
    if count == 0:
        return _exact_recompute_numerator(weights, ()), ()
    candidates = tuple(combinations(legal, count))
    winner = min(
        candidates,
        key=lambda candidate: (
            _exact_recompute_numerator(weights, candidate),
            tuple(reversed(candidate)),
        ),
    )
    return _exact_recompute_numerator(weights, winner), winner


def _exact_recompute_numerator(
    weights: tuple[int, ...], checkpoints: tuple[int, ...]
) -> int:
    checkpoint_index = 0
    restored_depth = 0
    total = 0
    for depth, weight in enumerate(weights, start=1):
        while (
            checkpoint_index < len(checkpoints)
            and checkpoints[checkpoint_index] <= depth
        ):
            restored_depth = checkpoints[checkpoint_index]
            checkpoint_index += 1
        total += weight * (depth - restored_depth)
    return total


@pytest.mark.parametrize("length", range(1, 7))
def test_monotone_hull_dp_matches_brute_force_for_every_small_ternary_distribution(
    length: int,
) -> None:
    for weights in product(range(3), repeat=length):
        if not any(weights):
            continue
        for budget in range(length + 2):
            for block_size in (1, 2, 3, 4):
                numerator, expected = _brute_force(weights, budget, block_size)
                actual = optimal_checkpoint_positions(weights, budget, block_size)
                assert actual == expected
                assert expected_recompute(weights, actual) == pytest.approx(
                    numerator / sum(weights)
                )


def test_monotone_hull_dp_matches_random_small_blocked_cases() -> None:
    rng = random.Random(0)
    for _ in range(80):
        length = rng.randint(1, 7)
        weights = tuple(rng.randint(0, 7) for _ in range(length))
        if not any(weights):
            weights = (1,) + weights[1:]
        budget = rng.randint(0, length + 2)
        block_size = rng.randint(1, 4)
        numerator, expected = _brute_force(weights, budget, block_size)
        actual = optimal_checkpoint_positions(weights, budget, block_size)
        assert actual == expected
        assert expected_recompute(weights, actual) == pytest.approx(
            numerator / sum(weights)
        )


def test_exact_dynamic_range_regression_uses_the_better_checkpoint() -> None:
    weights = (10**16, 10**16, 1)
    assert optimal_checkpoint_positions(weights, budget=1) == (2,)
    assert expected_recompute(weights, (2,)) < expected_recompute(weights, (1,))


def test_integral_weights_do_not_round_trip_through_binary64() -> None:
    weights = (10**400, 10**400, 1)
    assert optimal_checkpoint_positions(weights, budget=1) == (2,)


def test_ties_are_reverse_lexicographic_final_checkpoint_first() -> None:
    weights = (2, 3, 2)
    # (1, 2) and (2, 3) have the same exact recompute numerator.  The final
    # checkpoint 2 is earlier than 3, so the documented winner is (1, 2).
    assert _exact_recompute_numerator(weights, (1, 2)) == _exact_recompute_numerator(
        weights, (2, 3)
    )
    assert optimal_checkpoint_positions(weights, budget=2) == (1, 2)


def test_nonintegral_reals_follow_their_binary64_values() -> None:
    weights = (Fraction(1, 10), Fraction(2, 10), Fraction(3, 10), Fraction(4, 10))
    # The public conversion rule is intentionally binary64, so rerunning with
    # explicit float values cannot alter the chosen exact placement.
    assert optimal_checkpoint_positions(
        weights, budget=2
    ) == optimal_checkpoint_positions(
        tuple(float(weight) for weight in weights), budget=2
    )


@pytest.mark.parametrize(
    ("prefix_length", "budget", "expected"),
    [
        (5, 1, (3,)),
        (5, 2, (2, 4)),
        (4, 1, (2,)),
        (10, 3, (2, 5, 8)),
    ],
)
def test_uniform_distribution_uses_the_balanced_exact_optimum(
    prefix_length: int, budget: int, expected: tuple[int, ...]
) -> None:
    weights = (1.0,) * prefix_length
    assert balanced_checkpoint_positions(prefix_length, budget) == expected
    assert optimal_checkpoint_positions(weights, budget) == expected


def test_nonuniform_spike_draws_a_checkpoint_to_the_spike() -> None:
    weights = (0.01, 0.01, 0.01, 0.01, 50.0, 0.01, 0.01, 0.01)
    assert optimal_checkpoint_positions(weights, budget=1) == (5,)
    # Positions 2 and 3 tie here; the solver retains the earlier checkpoint.
    assert optimal_checkpoint_positions(weights, budget=2) == (2, 5)


def test_block_positions_are_clipped_to_full_blocks_with_a_legal_partial_tail() -> None:
    weights = (1.0,) * 5
    assert optimal_checkpoint_positions(weights, budget=99, block_size=2) == (2, 4, 5)
    assert optimal_checkpoint_positions(weights, budget=2, block_size=2) == (2, 4)
    # Exact depth semantics remain in force: position 2 cannot restore depth 1.
    assert expected_recompute(weights, (2, 4, 5)) == pytest.approx(2.0 / 5.0)
    assert balanced_checkpoint_positions(5, 99, block_size=2) == (2, 4, 5)
    assert geometric_tail_checkpoint_positions(5, 99, block_size=2) == (2, 4, 5)


def test_geometric_tail_policy_is_legal_unique_sorted_and_tail_dense() -> None:
    positions = geometric_tail_checkpoint_positions(100, budget=4, block_size=5)
    assert positions == (50, 75, 85, 90)
    assert positions == tuple(sorted(set(positions)))
    assert all(position in _legal_positions(100, 5) for position in positions)


def test_zero_and_invalid_input_are_rejected_or_empty_as_contract_requires() -> None:
    assert normalize_overlap_weights((1.0, 3.0)) == pytest.approx((0.25, 0.75))
    with pytest.raises(ValueError, match="positive finite total"):
        optimal_checkpoint_positions((0.0, 0.0), 1)
    with pytest.raises(ValueError, match="non-negative and finite"):
        optimal_checkpoint_positions((1.0, -1.0), 1)
    with pytest.raises(ValueError, match="non-negative and finite"):
        optimal_checkpoint_positions((1.0, float("nan")), 1)
    with pytest.raises(ValueError, match="non-negative and finite"):
        optimal_checkpoint_positions((1.0, float("inf")), 1)
    with pytest.raises(TypeError, match="weights must be a sequence"):
        optimal_checkpoint_positions((weight for weight in (1.0, 2.0)), 1)
    with pytest.raises(ValueError, match="block_size"):
        optimal_checkpoint_positions((1.0, 2.0), 1, 0)
    with pytest.raises(TypeError, match="budget"):
        optimal_checkpoint_positions((1.0, 2.0), 1.5)

    assert optimal_checkpoint_positions((1.0, 2.0), 0) == ()
    assert optimal_checkpoint_positions((1.0, 2.0), -3) == ()
    assert balanced_checkpoint_positions(0, 3) == ()
    assert geometric_tail_checkpoint_positions(0, 3) == ()


def test_expected_recompute_requires_sorted_unique_in_range_positions() -> None:
    weights = (1.0, 2.0, 3.0)
    assert expected_recompute(weights, (2,)) == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError, match="sorted and unique"):
        expected_recompute(weights, (2, 2))
    with pytest.raises(ValueError, match="sorted and unique"):
        expected_recompute(weights, (3, 2))
    with pytest.raises(ValueError, match="must not exceed"):
        expected_recompute(weights, (4,))
    with pytest.raises(ValueError, match="at least 1"):
        expected_recompute(weights, (0,))


def test_exponential_histogram_decay_clipping_and_reset() -> None:
    histogram = ExponentialOverlapHistogram(max_depth=4, decay=0.5)
    histogram.observe(-10, weight=2.0)
    histogram.observe(99, weight=4.0)
    assert histogram.weights() == pytest.approx((1.0, 0.0, 0.0, 4.0))
    assert histogram.total_weight == pytest.approx(5.0)
    assert histogram.sample_count == 2

    histogram.observe(2, weight=0.0)
    assert histogram.weights() == pytest.approx((0.5, 0.0, 0.0, 2.0))
    assert histogram.total_weight == pytest.approx(2.5)
    assert histogram.sample_count == 3

    histogram.reset()
    assert histogram.weights() == (0.0, 0.0, 0.0, 0.0)
    assert histogram.total_weight == 0.0
    assert histogram.sample_count == 0


@pytest.mark.parametrize("decay", (0.0, -0.1, 1.1, float("nan"), float("inf")))
def test_histogram_rejects_invalid_decay(decay: float) -> None:
    with pytest.raises(ValueError):
        ExponentialOverlapHistogram(max_depth=2, decay=decay)


def test_histogram_rejects_invalid_weights_and_nonintegral_depth() -> None:
    histogram = ExponentialOverlapHistogram(max_depth=2)
    with pytest.raises(ValueError, match="non-negative and finite"):
        histogram.observe(1, weight=-1.0)
    with pytest.raises(ValueError, match="non-negative and finite"):
        histogram.observe(1, weight=float("nan"))
    with pytest.raises(TypeError, match="depth"):
        histogram.observe(1.0)
    with pytest.raises(ValueError, match="max_depth"):
        ExponentialOverlapHistogram(max_depth=0)


def test_module_import_does_not_load_mlx() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mtplx.checkpoint_placement; "
            "assert not any(name == 'mlx' or name.startswith('mlx.') for name in sys.modules)",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
