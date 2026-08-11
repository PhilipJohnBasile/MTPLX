"""Pure-CPU placement policies for bounded sparse prefix checkpoints.

The exact objective follows Sparse Prefix Caching (arXiv:2605.05219v1).  A
weight at index ``t - 1`` describes the frequency of overlap depth ``t``.  If
the saved checkpoint positions are ``C``, its recompute work is
``t - max({0} union {c in C | c <= t})``.

``block_size`` restricts *where a state may be saved*, not what an overlap
depth means.  For a prefix of length ``N``, legal positions are
``block_size, 2 * block_size, ...`` plus ``N`` when the final block is partial.
The DP still aggregates the original depth weights exactly between those legal
positions.  In particular, a checkpoint at ``2`` does not cover an overlap of
depth ``1``: there was no state at depth ``1`` to restore.  This keeps the
reported expected recompute value faithful to the positional definition.

No MLX objects or imports belong in this module.  It is intentionally useful
to offline placement planners and CPU-only telemetry collectors.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real


__all__ = [
    "ExponentialOverlapHistogram",
    "balanced_checkpoint_positions",
    "expected_recompute",
    "geometric_tail_checkpoint_positions",
    "normalize_overlap_weights",
    "optimal_checkpoint_positions",
    "optimal_checkpoint_positions_from_candidates",
]


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    """Return an integer argument without accepting booleans or float aliases."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _nonnegative_finite(value: object, name: str) -> float:
    """Validate one real, non-negative, finite value represented as a float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _exact_scaled_weights(weights: Sequence[Real]) -> tuple[int, ...]:
    """Return weights as gcd-reduced, common-scale non-negative integers.

    Integral inputs are kept as Python integers without a float round trip.
    Every other accepted ``Real`` is deliberately interpreted as the binary64
    value returned by :class:`float`, then converted with
    :meth:`float.as_integer_ratio`.  Binary64 denominators are powers of two,
    so one common power-of-two scale makes every input exact.  Removing the
    common gcd preserves all objective comparisons while keeping intermediates
    smaller.
    """
    if isinstance(weights, (str, bytes, bytearray)) or not isinstance(
        weights, Sequence
    ):
        raise TypeError("weights must be a sequence indexed by depths 1..N")

    numerators: list[int] = []
    denominator_shifts: list[int] = []
    for index, weight in enumerate(weights):
        name = f"weights[{index}]"
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise TypeError(f"{name} must be a real number")
        if isinstance(weight, Integral):
            numerator = int(weight)
            if numerator < 0:
                raise ValueError(f"{name} must be non-negative and finite")
            numerators.append(numerator)
            denominator_shifts.append(0)
            continue

        converted = float(weight)
        if not math.isfinite(converted) or converted < 0.0:
            raise ValueError(f"{name} must be non-negative and finite")
        numerator, denominator = converted.as_integer_ratio()
        # Finite binary64 denominators are powers of two, including subnormals.
        shift = denominator.bit_length() - 1
        if denominator != 1 << shift:
            raise AssertionError("binary64 denominator must be a power of two")
        numerators.append(numerator)
        denominator_shifts.append(shift)

    if not numerators:
        raise ValueError("weights must contain at least one depth")

    common_shift = max(denominator_shifts)
    scaled = tuple(
        numerator << (common_shift - shift)
        for numerator, shift in zip(numerators, denominator_shifts, strict=True)
    )
    divisor = 0
    for weight in scaled:
        divisor = math.gcd(divisor, weight)
    if divisor == 0:
        raise ValueError("weights must have a positive finite total")
    return tuple(weight // divisor for weight in scaled)


def _normalized_weights(weights: Sequence[Real]) -> tuple[float, ...]:
    """Validate depth weights and return a public, normalized float view."""
    scaled = _exact_scaled_weights(weights)
    total = sum(scaled)
    return tuple(weight / total for weight in scaled)


def normalize_overlap_weights(weights: Sequence[Real]) -> tuple[float, ...]:
    """Validate and normalize non-negative finite overlap weights for depths 1..N."""
    return _normalized_weights(weights)


def _legal_positions(prefix_length: int, block_size: int) -> tuple[int, ...]:
    """Return legal checkpoint positions, retaining a final partial-block tail."""
    if prefix_length == 0:
        return ()
    positions = list(range(block_size, prefix_length + 1, block_size))
    if not positions or positions[-1] != prefix_length:
        positions.append(prefix_length)
    return tuple(positions)


def _validated_positions(
    positions: Sequence[Integral], prefix_length: int, name: str
) -> tuple[int, ...]:
    if isinstance(positions, (str, bytes, bytearray)) or not isinstance(
        positions, Sequence
    ):
        raise TypeError(f"{name} must be a sorted sequence of positions")

    result: list[int] = []
    previous = 0
    for index, position_value in enumerate(positions):
        position = _integer(position_value, f"{name}[{index}]", minimum=1)
        if position > prefix_length:
            raise ValueError(f"{name} must not exceed the prefix length")
        if position <= previous:
            raise ValueError(f"{name} must be sorted and unique")
        result.append(position)
        previous = position
    return tuple(result)


def _validated_checkpoints(
    checkpoints: Sequence[Integral], prefix_length: int
) -> tuple[int, ...]:
    return _validated_positions(checkpoints, prefix_length, "checkpoints")


def _prefix_sums(weights: tuple[int, ...]) -> tuple[list[int], list[int]]:
    """Return exact P_j=sum(w_t) and T_j=sum(t*w_t), with zero entries."""
    probability = [0]
    weighted_depth = [0]
    running_probability = 0
    running_weighted_depth = 0
    for depth, weight in enumerate(weights, start=1):
        term = depth * weight
        running_probability += weight
        running_weighted_depth += term
        probability.append(running_probability)
        weighted_depth.append(running_weighted_depth)
    return probability, weighted_depth


@dataclass(frozen=True, slots=True)
class _Line:
    """One DP predecessor represented as a line in P_j space."""

    slope: int
    intercept: int
    checkpoint: int

    def value_at(self, x: int) -> int:
        return self.slope * x + self.intercept


def _redundant(first: _Line, middle: _Line, last: _Line) -> bool:
    """Whether ``middle`` is never strictly better for monotone queries.

    Slopes are inserted in strictly decreasing order.  Keeping the earlier
    line at an exact intersection gives deterministic, leftmost tie handling.
    """
    # m_first > m_middle > m_last.  Compare the two intersection abscissas
    # without division.  On equality, ``middle`` is removable: at the sole
    # three-way tie, ``first`` has the earliest current checkpoint and wins.
    return (middle.intercept - first.intercept) * (middle.slope - last.slope) >= (
        last.intercept - middle.intercept
    ) * (first.slope - middle.slope)


def _add_line(hull: deque[_Line], line: _Line) -> None:
    while len(hull) >= 2 and _redundant(hull[-2], hull[-1], line):
        hull.pop()
    hull.append(line)


def _query(hull: deque[_Line], x: int) -> _Line:
    """Query a decreasing-slope lower hull at non-decreasing ``x``."""
    while len(hull) >= 2 and hull[1].value_at(x) < hull[0].value_at(x):
        hull.popleft()
    return hull[0]


def expected_recompute(
    weights: Sequence[Real], checkpoints: Sequence[Integral]
) -> float:
    """Return exact expected recompute depth for sorted checkpoint positions.

    ``weights`` is normalized internally, so callers may pass counts or any
    other positive-total non-negative weighting.  Checkpoints are validated
    rather than silently sorted: positional mistakes should be visible at the
    telemetry boundary.
    """
    scaled = _exact_scaled_weights(weights)
    positions = _validated_checkpoints(checkpoints, len(scaled))

    numerator = 0
    checkpoint_index = 0
    restored_depth = 0
    for depth, weight in enumerate(scaled, start=1):
        while (
            checkpoint_index < len(positions) and positions[checkpoint_index] <= depth
        ):
            restored_depth = positions[checkpoint_index]
            checkpoint_index += 1
        numerator += weight * (depth - restored_depth)
    return numerator / sum(scaled)


def optimal_checkpoint_positions(
    weights: Sequence[Real], budget: Integral, block_size: Integral = 1
) -> tuple[int, ...]:
    """Return an exact, deterministic O(N*M) sparse checkpoint placement.

    ``budget`` is the maximum number of checkpoints and ``M`` is its effective
    value after restricting it to legal positions.  With non-negative weights,
    using every available checkpoint cannot worsen the objective, so the
    returned placement uses that effective budget.  Ties compare placements
    from their final checkpoint backward: the earlier final checkpoint wins;
    if those match, the earlier predecessor wins recursively.  This is
    reverse-lexicographic order on the ascending position tuple.

    The dynamic program stores parents for reconstruction.  Its monotone
    convex hull evaluates the recurrence with exact integer arithmetic in
    O(N*M) arithmetic operations, where N is the number of depth weights and
    M is the effective checkpoint budget.
    """
    scaled = _exact_scaled_weights(weights)
    requested_budget = _integer(budget, "budget")
    size = _integer(block_size, "block_size", minimum=1)
    if requested_budget <= 0:
        return ()

    prefix_length = len(scaled)
    legal_positions = _legal_positions(prefix_length, size)
    effective_budget = min(requested_budget, len(legal_positions))
    if effective_budget == len(legal_positions):
        return legal_positions

    legal = set(legal_positions)
    probability, weighted_depth = _prefix_sums(scaled)
    previous = [weighted_depth[depth] for depth in range(prefix_length + 1)]
    parents: list[list[int]] = [
        [0] * (prefix_length + 1) for _ in range(effective_budget + 1)
    ]

    for count in range(1, effective_budget + 1):
        current: list[int | None] = [None] * (prefix_length + 1)
        hull: deque[_Line] = deque()
        for depth in range(1, prefix_length + 1):
            predecessor = previous[depth - 1]
            if depth in legal and predecessor is not None:
                intercept = (
                    predecessor
                    - weighted_depth[depth - 1]
                    + depth * probability[depth - 1]
                )
                _add_line(
                    hull, _Line(slope=-depth, intercept=intercept, checkpoint=depth)
                )
            if hull:
                line = _query(hull, probability[depth])
                value = weighted_depth[depth] + line.value_at(probability[depth])
                current[depth] = value
                parents[count][depth] = line.checkpoint
        previous = current

    result: list[int] = []
    depth = prefix_length
    for count in range(effective_budget, 0, -1):
        checkpoint = parents[count][depth]
        if checkpoint == 0:
            raise RuntimeError("checkpoint placement reconstruction failed")
        result.append(checkpoint)
        depth = checkpoint - 1
    result.reverse()
    return tuple(result)


def optimal_checkpoint_positions_from_candidates(
    weights: Sequence[Real],
    budget: Integral,
    candidate_positions: Sequence[Integral],
    fixed_positions: Sequence[Integral] = (),
) -> tuple[int, ...]:
    """Return the exact placement constrained to irregular capture positions.

    ``candidate_positions`` identifies sorted, unique, legal checkpoint
    positions that may consume ``budget``.  ``fixed_positions`` has the same
    validation contract, but every fixed position is mandatory and free.  A
    position appearing in both inputs is therefore fixed and does not consume
    a candidate slot.  The result contains the sorted union of both sets.

    The objective is the same positional recompute objective used by
    :func:`optimal_checkpoint_positions`: at overlap depth ``t``, restore the
    latest saved position not greater than ``t``.  This matters for actual
    capture grids because an absent position cannot be synthesized by a nearby
    checkpoint.  Non-negative weights mean every distinct non-fixed candidate
    available within the effective budget is selected.

    The dynamic program scans only candidate/fixed boundaries (plus a terminal
    sentinel), while exact prefix sums account for every depth weight.  Its
    monotone lower hull is O(K * M) after O(N) prefix sums, where ``K`` is the
    number of supplied positions and ``M`` is the effective candidate budget.
    At equal objective values it retains the earlier predecessor, matching the
    existing solver's reverse-lexicographic tie rule for ascending placements.
    """
    scaled = _exact_scaled_weights(weights)
    prefix_length = len(scaled)
    requested_budget = _integer(budget, "budget")
    candidates = _validated_positions(
        candidate_positions, prefix_length, "candidate_positions"
    )
    fixed = _validated_positions(fixed_positions, prefix_length, "fixed_positions")

    fixed_set = set(fixed)
    optional_candidates = tuple(
        position for position in candidates if position not in fixed_set
    )
    effective_budget = min(max(requested_budget, 0), len(optional_candidates))
    if not optional_candidates or effective_budget == 0:
        return fixed

    probability, weighted_depth = _prefix_sums(scaled)
    candidate_set = set(optional_candidates)
    fixed_set = set(fixed)
    terminal = prefix_length + 1
    nodes = tuple(sorted(candidate_set | fixed_set | {terminal}))

    # ``hulls[count]`` contains exact predecessor states with ``count`` paid
    # candidates.  A line for position ``p`` evaluates the recompute cost up
    # to (but excluding) the next checkpoint boundary.  A fixed boundary
    # resets every hull, which prevents a transition from skipping it.
    hulls: list[deque[_Line]] = [deque() for _ in range(effective_budget + 1)]
    hulls[0].append(_Line(slope=0, intercept=0, checkpoint=0))
    parents: dict[tuple[int, int], tuple[int, int]] = {}

    for position in nodes:
        is_candidate = position in candidate_set
        is_fixed = position in fixed_set
        state_costs: list[int | None] = [None] * (effective_budget + 1)
        state_parents: list[tuple[int, int] | None] = [None] * (effective_budget + 1)
        query_mass = probability[position - 1]

        for count in range(effective_budget + 1):
            predecessor_count = count - 1 if is_candidate else count
            if predecessor_count < 0:
                continue
            hull = hulls[predecessor_count]
            if not hull:
                continue
            line = _query(hull, query_mass)
            state_costs[count] = weighted_depth[position - 1] + line.value_at(
                query_mass
            )
            state_parents[count] = (line.checkpoint, predecessor_count)

        if position == terminal:
            break

        if is_fixed:
            for count, cost in enumerate(state_costs):
                hulls[count].clear()
                parent = state_parents[count]
                if cost is None or parent is None:
                    continue
                parents[(position, count)] = parent
                hulls[count].append(
                    _Line(
                        slope=-position,
                        intercept=(
                            cost
                            - weighted_depth[position]
                            + position * probability[position]
                        ),
                        checkpoint=position,
                    )
                )
            continue

        # Candidate lines are appended only after every state at this boundary
        # has been queried, so one capture position cannot consume two slots.
        for count, cost in enumerate(state_costs):
            parent = state_parents[count]
            if cost is None or parent is None:
                continue
            parents[(position, count)] = parent
            _add_line(
                hulls[count],
                _Line(
                    slope=-position,
                    intercept=(
                        cost
                        - weighted_depth[position]
                        + position * probability[position]
                    ),
                    checkpoint=position,
                ),
            )

    terminal_parent = state_parents[effective_budget]
    if terminal_parent is None:
        raise RuntimeError("candidate checkpoint placement reconstruction failed")
    parents[(terminal, effective_budget)] = terminal_parent

    selected: list[int] = []
    position = terminal
    count = effective_budget
    while position != 0:
        parent = parents.get((position, count))
        if parent is None:
            raise RuntimeError("candidate checkpoint placement reconstruction failed")
        if position in candidate_set:
            selected.append(position)
        position, count = parent

    return tuple(sorted(fixed_set | set(selected)))


def balanced_checkpoint_positions(
    prefix_length: Integral, budget: Integral, block_size: Integral = 1
) -> tuple[int, ...]:
    """Return the exact uniform-overlap placement under the same block contract.

    The uniform objective's optimum partitions ``prefix_length + 1`` into
    nearly equal segments.  Calling the exact shared solver retains that
    property under block restrictions and makes tie handling identical to the
    weighted placement API.
    """
    length = _integer(prefix_length, "prefix_length", minimum=0)
    if length == 0:
        _integer(block_size, "block_size", minimum=1)
        _integer(budget, "budget")
        return ()
    return optimal_checkpoint_positions((1.0,) * length, budget, block_size)


def geometric_tail_checkpoint_positions(
    prefix_length: Integral, budget: Integral, block_size: Integral = 1
) -> tuple[int, ...]:
    """Return a deterministic tail-dense, legal checkpoint heuristic.

    Ideal positions approach the tail geometrically: N/2, 3N/4, 7N/8, ... .
    Each is clipped down to a legal saved-state position.  If clipping makes
    positions collide, the next later unused legal position is selected; this
    preserves a tail-dense ordering while honoring the requested budget when
    possible.  This is a heuristic for comparisons, not a replacement for the
    exact weighted solver.
    """
    length = _integer(prefix_length, "prefix_length", minimum=0)
    requested_budget = _integer(budget, "budget")
    size = _integer(block_size, "block_size", minimum=1)
    if length == 0 or requested_budget <= 0:
        return ()

    legal_positions = _legal_positions(length, size)
    count = min(requested_budget, len(legal_positions))
    if count == len(legal_positions):
        return legal_positions

    selected: list[int] = []
    selected_set: set[int] = set()
    for index in range(count):
        denominator = 1 << (index + 1)
        target = length - (length // denominator)
        candidate_index = 0
        for position_index, position in enumerate(legal_positions):
            if position <= target:
                candidate_index = position_index
            else:
                break

        while (
            candidate_index < len(legal_positions)
            and legal_positions[candidate_index] in selected_set
        ):
            candidate_index += 1
        if candidate_index == len(legal_positions):
            candidate_index = len(legal_positions) - 1
            while legal_positions[candidate_index] in selected_set:
                candidate_index -= 1
        candidate = legal_positions[candidate_index]
        selected.append(candidate)
        selected_set.add(candidate)
    return tuple(sorted(selected))


class ExponentialOverlapHistogram:
    """Bounded, exponentially-forgetting overlap-depth telemetry.

    Each observation first multiplies every bin by ``decay``, then adds its
    non-negative finite weight to ``depth`` clipped into ``[1, max_depth]``.
    ``sample_count`` counts calls to :meth:`observe`; ``total_weight`` is the
    current, decayed mass.  :meth:`weights` returns unnormalized bin weights
    for the placement functions, which normalize their inputs themselves.
    """

    __slots__ = ("_bins", "_decay", "_max_depth", "_sample_count", "_total_weight")

    def __init__(self, max_depth: Integral, decay: Real = 0.99) -> None:
        self._max_depth = _integer(max_depth, "max_depth", minimum=1)
        if isinstance(decay, bool) or not isinstance(decay, Real):
            raise TypeError("decay must be a real number")
        self._decay = float(decay)
        if not math.isfinite(self._decay) or not 0.0 < self._decay <= 1.0:
            raise ValueError("decay must be finite and in (0, 1]")
        self._bins = [0.0] * self._max_depth
        self._sample_count = 0
        self._total_weight = 0.0

    @property
    def max_depth(self) -> int:
        """The fixed number of depth bins retained by this histogram."""
        return self._max_depth

    @property
    def total_weight(self) -> float:
        """Current decayed total weight."""
        return self._total_weight

    @property
    def sample_count(self) -> int:
        """Number of observations since construction or the last reset."""
        return self._sample_count

    def observe(self, depth: Integral, weight: Real = 1.0) -> None:
        """Record one clipped depth observation without growing storage."""
        raw_depth = _integer(depth, "depth")
        observed_weight = _nonnegative_finite(weight, "weight")
        clipped_depth = min(max(raw_depth, 1), self._max_depth)

        try:
            updated = [value * self._decay for value in self._bins]
            updated[clipped_depth - 1] += observed_weight
            total = math.fsum(updated)
        except OverflowError as exc:
            raise ValueError(
                "observation would produce a non-finite histogram"
            ) from exc
        if not all(math.isfinite(value) for value in updated) or not math.isfinite(
            total
        ):
            raise ValueError("observation would produce a non-finite histogram")

        self._bins = updated
        self._total_weight = total
        self._sample_count += 1

    def weights(self) -> tuple[float, ...]:
        """Return current unnormalized weight by exact depth, 1 through max_depth."""
        return tuple(self._bins)

    def reset(self) -> None:
        """Discard all observations while retaining max depth and decay."""
        self._bins = [0.0] * self._max_depth
        self._sample_count = 0
        self._total_weight = 0.0
