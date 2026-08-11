"""CPU-only reference oracle for staged rotating-KV cache transactions.

This is deliberately not an MLX cache implementation, production integration,
or performance benchmark.  It is a small standard-library state model for the
transactional contract that a rotating cache must satisfy when speculative
verification accepts only a prefix of staged K/V rows.

The model follows the staged-round shape documented by mlx-swift-lm #516:
staging is read-only with respect to the live ring, commit writes only retained
rows through normal ring arithmetic, and discarded rows never reach the live
cache.  MTPLX's current Gemma exact round has a deferred ``bonus_token_id``;
the oracle records and checks that behavior separately from a commit-time
bonus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT_PATH = (
    REPOSITORY_ROOT
    / "benchmarks/results/mlx-m5-research-20260810/rotating_staged_kv_cpu_oracle.json"
)
MAX_CAPACITY = 16
_PAIR_GROUP_CODES = {
    "prefill": 1,
    "proposal": 2,
    "live": 3,
    "cleanup-rejected": 4,
    "bonus": 5,
}


@dataclass(frozen=True)
class KVPair:
    """One position-encoded K/V row in the reference model."""

    tag: str
    key: int
    value: int


@dataclass(frozen=True)
class StoredKV:
    """A K/V row after it has been committed to an absolute position."""

    position: int
    pair: KVPair


@dataclass(frozen=True)
class SlotWrite:
    """One normal-path ring write performed by a commit."""

    position: int
    slot: int
    pair: KVPair


@dataclass(frozen=True)
class RoundCommit:
    """The retained and discarded portions of one staged round."""

    accepted_prefix: int
    discarded_suffix: int
    bonus_committed: bool
    writes: tuple[SlotWrite, ...]


class RotatingStagedKVReference:
    """Pure-Python staged transaction model for a keep=0 rotating KV cache.

    ``offset`` is the absolute count of committed rows.  A physical ring slot is
    therefore always ``position % capacity``.  The staged proposal is never
    merged into ``_slots`` before commit, making rejected-suffix invisibility a
    state-machine property rather than a best-effort rollback.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least one")
        self.capacity = int(capacity)
        self.offset = 0
        self._slots: list[StoredKV | None] = [None] * self.capacity
        self._staged: tuple[KVPair, ...] | None = None

    @property
    def staged_count(self) -> int:
        return 0 if self._staged is None else len(self._staged)

    @property
    def next_write_slot(self) -> int:
        return self.offset % self.capacity

    def stage(self, proposal: Sequence[KVPair]) -> None:
        """Open one round without changing the live cache or timeline."""

        if self._staged is not None:
            raise RuntimeError("a staged round is already open")
        self._staged = tuple(proposal)

    def commit(self, accepted_prefix: int, *, bonus: KVPair | None = None) -> RoundCommit:
        """Commit a retained proposal prefix and, optionally, a round bonus.

        A commit-time bonus models the Swift staged-round contract.  MTPLX's
        current Gemma exact round instead reports the bonus as its next primary
        token, so callers model that behavior by leaving ``bonus`` as ``None``.
        """

        if self._staged is None:
            raise RuntimeError("no staged round is open")
        if not 0 <= int(accepted_prefix) <= len(self._staged):
            raise ValueError("accepted prefix is outside the staged proposal")

        retained = self._staged[: int(accepted_prefix)]
        discarded = self._staged[int(accepted_prefix) :]
        committed = retained + (() if bonus is None else (bonus,))
        writes = tuple(self._append(pair) for pair in committed)
        self._staged = None
        return RoundCommit(
            accepted_prefix=len(retained),
            discarded_suffix=len(discarded),
            bonus_committed=bonus is not None,
            writes=writes,
        )

    def cleanup(self) -> int:
        """Discard an open proposal without changing committed state."""

        if self._staged is None:
            return 0
        discarded = len(self._staged)
        self._staged = None
        return discarded

    def reset(self) -> None:
        """Reset both the ring and the absolute timeline."""

        self.offset = 0
        self._slots = [None] * self.capacity
        self._staged = None

    def visible(self) -> tuple[StoredKV, ...]:
        """Return live K/V rows in chronological rather than physical order."""

        return tuple(sorted((item for item in self._slots if item is not None), key=lambda item: item.position))

    def next_token_plan(self) -> dict[str, Any]:
        """Describe the query position and physical mask after its K/V write.

        The next token is at the current absolute offset.  Its K/V row lands at
        ``next_write_slot`` before it attends, so the returned physical mask and
        position layout include that row.  This mirrors the common
        update-and-fetch attention contract while keeping the model read-only.
        """

        positions_by_slot = [
            None if item is None else int(item.position) for item in self._slots
        ]
        positions_by_slot[self.next_write_slot] = int(self.offset)
        visible_positions = tuple(
            sorted(position for position in positions_by_slot if position is not None)
        )
        return {
            "query_position": int(self.offset),
            "write_slot": int(self.next_write_slot),
            "mask_by_slot": tuple(position is not None for position in positions_by_slot),
            "positions_by_slot": tuple(positions_by_slot),
            "visible_positions": visible_positions,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return only primitive data so results can be made deterministic."""

        def as_row(item: StoredKV | None) -> dict[str, Any] | None:
            if item is None:
                return None
            return {
                "key": int(item.pair.key),
                "position": int(item.position),
                "tag": item.pair.tag,
                "value": int(item.pair.value),
            }

        return {
            "capacity": int(self.capacity),
            "next_token": self.next_token_plan(),
            "next_write_slot": int(self.next_write_slot),
            "offset": int(self.offset),
            "slots": [as_row(item) for item in self._slots],
            "staged_count": int(self.staged_count),
            "visible": [as_row(item) for item in self.visible()],
        }

    def _append(self, pair: KVPair) -> SlotWrite:
        position = int(self.offset)
        slot = position % self.capacity
        self._slots[slot] = StoredKV(position=position, pair=pair)
        self.offset += 1
        return SlotWrite(position=position, slot=slot, pair=pair)


def _pair(group: str, capacity: int, index: int) -> KVPair:
    """Create a deterministic, position-distinguishable K/V pair."""

    try:
        group_code = _PAIR_GROUP_CODES[group]
    except KeyError as error:
        raise ValueError(f"unknown deterministic K/V group: {group}") from error
    seed = capacity * 100_000 + group_code * 1_000 + index
    return KVPair(tag=f"{group}/k{capacity}/i{index}", key=seed, value=-seed - 17)


def _serialize_commit(commit: RoundCommit) -> dict[str, Any]:
    return {
        "accepted_prefix": int(commit.accepted_prefix),
        "bonus_committed": bool(commit.bonus_committed),
        "discarded_suffix": int(commit.discarded_suffix),
        "writes": [
            {
                "key": int(write.pair.key),
                "position": int(write.position),
                "slot": int(write.slot),
                "tag": write.pair.tag,
                "value": int(write.pair.value),
            }
            for write in commit.writes
        ],
    }


def assert_contract(cache: RotatingStagedKVReference, history: Sequence[KVPair]) -> None:
    """Check chronological visibility, ring placement, and next-token masking."""

    if cache.offset != len(history):
        raise AssertionError(f"offset {cache.offset} did not equal history {len(history)}")

    expected = tuple(history[-cache.capacity :])
    visible = cache.visible()
    if tuple(item.pair for item in visible) != expected:
        raise AssertionError("chronological visible K/V did not match the retained history")
    expected_positions = tuple(range(len(history) - len(expected), len(history)))
    if tuple(item.position for item in visible) != expected_positions:
        raise AssertionError("visible K/V positions were not chronological")

    for item in visible:
        expected_slot = item.position % cache.capacity
        if cache._slots[expected_slot] != item:
            raise AssertionError("a visible K/V row occupied the wrong physical slot")

    plan = cache.next_token_plan()
    expected_next_positions = tuple(
        range(max(0, len(history) - cache.capacity + 1), len(history) + 1)
    )
    if tuple(plan["visible_positions"]) != expected_next_positions:
        raise AssertionError("next-token mask did not expose the correct chronological positions")
    if int(plan["query_position"]) != len(history):
        raise AssertionError("next-token query position did not equal the cache offset")
    if int(plan["write_slot"]) != len(history) % cache.capacity:
        raise AssertionError("next-token write slot did not follow ring arithmetic")

    expected_by_slot = [None] * cache.capacity
    for position in expected_next_positions:
        expected_by_slot[position % cache.capacity] = position
    if tuple(plan["positions_by_slot"]) != tuple(expected_by_slot):
        raise AssertionError("next-token physical positions did not match the ring view")
    if tuple(plan["mask_by_slot"]) != tuple(position is not None for position in expected_by_slot):
        raise AssertionError("next-token mask did not hide empty ring slots")


def _seed_before_wrap(capacity: int) -> tuple[RotatingStagedKVReference, list[KVPair]]:
    cache = RotatingStagedKVReference(capacity)
    history = [_pair("prefill", capacity, index) for index in range(capacity - 1)]
    cache.stage(history)
    cache.commit(len(history))
    assert_contract(cache, history)
    return cache, history


def run_prefix_matrix(max_capacity: int = MAX_CAPACITY) -> dict[str, Any]:
    """Exercise accepted prefixes 0..K at K=1..``max_capacity``."""

    if not 1 <= int(max_capacity) <= MAX_CAPACITY:
        raise ValueError(f"max_capacity must be between 1 and {MAX_CAPACITY}")

    scenario_counts = {
        "before_first_wrap": 0,
        "exactly_at_first_wrap": 0,
        "post_first_wrap": 0,
    }
    case_count = 0
    traces: list[dict[str, Any]] = []
    for capacity in range(1, int(max_capacity) + 1):
        proposal = [_pair("proposal", capacity, index) for index in range(capacity)]
        for accepted_prefix in range(capacity + 1):
            cache, history = _seed_before_wrap(capacity)
            cache.stage(proposal)
            commit = cache.commit(accepted_prefix)
            history.extend(proposal[:accepted_prefix])
            assert_contract(cache, history)

            visible_keys = {item.pair.key for item in cache.visible()}
            rejected_keys = {pair.key for pair in proposal[accepted_prefix:]}
            if visible_keys & rejected_keys:
                raise AssertionError("a rejected proposal suffix became visible")
            if commit.discarded_suffix != capacity - accepted_prefix:
                raise AssertionError("discarded suffix count did not match the proposal")

            if accepted_prefix == 0:
                scenario = "before_first_wrap"
                if cache.offset != capacity - 1:
                    raise AssertionError("zero acceptance changed the pre-wrap offset")
            elif accepted_prefix == 1:
                scenario = "exactly_at_first_wrap"
                if cache.offset != capacity or cache.next_write_slot != 0:
                    raise AssertionError("the first full ring did not wrap to slot zero")
            else:
                scenario = "post_first_wrap"
                if cache.offset <= capacity:
                    raise AssertionError("post-wrap acceptance did not advance beyond the window")
            scenario_counts[scenario] += 1
            case_count += 1

            if capacity == 4 and accepted_prefix in {0, 1, 2, 4}:
                traces.append(
                    {
                        "accepted_prefix": accepted_prefix,
                        "commit": _serialize_commit(commit),
                        "scenario": scenario,
                        "snapshot": cache.snapshot(),
                    }
                )

    return {
        "accepted_prefixes": "0..K for every capacity K",
        "capacities": list(range(1, int(max_capacity) + 1)),
        "case_count": case_count,
        "scenario_counts": scenario_counts,
        "selected_capacity_4_traces": traces,
    }


def run_cleanup_and_reset_checks(max_capacity: int = MAX_CAPACITY) -> dict[str, Any]:
    """Prove cleanup and reset stay correct after a ring has wrapped."""

    cleanup_cases = 0
    reset_cases = 0
    for capacity in range(1, int(max_capacity) + 1):
        cache = RotatingStagedKVReference(capacity)
        history = [_pair("live", capacity, index) for index in range(capacity + 1)]
        cache.stage(history)
        cache.commit(len(history))
        assert_contract(cache, history)
        before_cleanup = cache.snapshot()

        rejected = [_pair("cleanup-rejected", capacity, index) for index in range(capacity)]
        cache.stage(rejected)
        if cache.snapshot() != {**before_cleanup, "staged_count": len(rejected)}:
            raise AssertionError("staging changed live cache state before cleanup")
        if cache.cleanup() != len(rejected):
            raise AssertionError("cleanup did not report the staged suffix length")
        if cache.snapshot() != before_cleanup:
            raise AssertionError("cleanup changed committed rotating-KV state")
        cleanup_cases += 1

        cache.reset()
        assert_contract(cache, ())
        if cache.snapshot()["staged_count"] != 0:
            raise AssertionError("reset left a staged proposal open")
        reset_cases += 1

    return {"cleanup_cases": cleanup_cases, "reset_cases": reset_cases}


def run_bonus_checks(max_capacity: int = MAX_CAPACITY) -> dict[str, Any]:
    """Check both Swift commit-time and current MTPLX deferred bonus semantics."""

    committed_bonus_cases = 0
    deferred_bonus_cases = 0
    for capacity in range(1, int(max_capacity) + 1):
        proposal = [_pair("proposal", capacity, index) for index in range(capacity)]
        bonus = _pair("bonus", capacity, 0)

        committed_cache, committed_history = _seed_before_wrap(capacity)
        committed_cache.stage(proposal)
        commit = committed_cache.commit(len(proposal), bonus=bonus)
        committed_history.extend(proposal)
        committed_history.append(bonus)
        assert_contract(committed_cache, committed_history)
        if not commit.bonus_committed or bonus.key not in {
            item.pair.key for item in committed_cache.visible()
        }:
            raise AssertionError("commit-time bonus was not present in the live ring")
        committed_bonus_cases += 1

        deferred_cache, deferred_history = _seed_before_wrap(capacity)
        deferred_cache.stage(proposal)
        deferred_commit = deferred_cache.commit(len(proposal))
        deferred_history.extend(proposal)
        assert_contract(deferred_cache, deferred_history)
        if deferred_commit.bonus_committed or bonus.key in {
            item.pair.key for item in deferred_cache.visible()
        }:
            raise AssertionError("deferred bonus became visible before its next primary write")
        deferred_bonus_cases += 1

    return {
        "mtplx_gemma4_exact_round": {
            "behavior": "deferred_next_primary_not_committed_in_this_round",
            "cases": deferred_bonus_cases,
        },
        "swift_516_staged_round": {
            "behavior": "commit_bonus_with_accepted_prefix",
            "cases": committed_bonus_cases,
        },
    }


def run_all_contract_checks(max_capacity: int = MAX_CAPACITY) -> dict[str, Any]:
    """Run the complete deterministic CPU reference suite."""

    return {
        "bonus": run_bonus_checks(max_capacity),
        "cleanup_and_reset": run_cleanup_and_reset_checks(max_capacity),
        "prefix_matrix": run_prefix_matrix(max_capacity),
    }


def build_receipt(max_capacity: int = MAX_CAPACITY) -> dict[str, Any]:
    """Build a stable, evidence-oriented receipt without timing or hardware data."""

    payload = {
        "contract": {
            "accepted_prefix": "Every K in 1..16 accepts exactly 0..K staged rows.",
            "chronology": "Visible K/V is sorted by absolute position, not physical ring slot.",
            "next_token": "The mask is evaluated after the next token writes at offset % K.",
            "rejection": "Rejected staged suffixes never enter the live ring or its mask plan.",
            "timeline": "Commit advances the absolute offset by exactly its retained writes.",
        },
        "limits": {
            "capacity_range": [1, int(max_capacity)],
            "keep": 0,
            "model_loading": "not performed",
            "performance_measurement": "not performed",
        },
        "oracle_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "result_kind": "cpu_reference_oracle",
        "results": run_all_contract_checks(max_capacity),
        "scope": {
            "imports": "Python standard library only",
            "metal": "not touched",
            "mlx": "not imported",
            "production_cache_integration": "not performed",
            "production_performance_evidence": "not provided",
        },
        "semantics_read": {
            "graphbank": "Rotating or indexed caches are excluded from tensor-offset promotion.",
            "mtplx_gemma4": "Verified suffix is rolled back to primary plus accepted drafts; a bonus is the next primary token.",
            "mtplx_laguna": "Ring index is normalized to the next legal write slot while absolute offset keeps advancing.",
            "swift_516": "Staged rotating rows are read without live mutation; commit keeps a prefix and advances the timeline atomically.",
        },
        "version": 1,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def render_receipt(max_capacity: int = MAX_CAPACITY) -> str:
    """Serialize the receipt in a byte-stable form."""

    return json.dumps(build_receipt(max_capacity), indent=2, sort_keys=True) + "\n"


def write_receipt(path: Path = DEFAULT_RECEIPT_PATH) -> Path:
    """Generate the deterministic receipt at ``path`` and return that path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_receipt(), encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT_PATH)
    args = parser.parse_args()
    print(f"Wrote CPU reference-oracle receipt: {write_receipt(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
