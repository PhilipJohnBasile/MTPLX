"""Opt-in expert-routing locality instrumentation for Apple Silicon.

This module is diagnostic only.  It never changes router outputs or expert
placement.  When ``MTPLX_EXPERT_LOCALITY=1`` is unset, the public record helper
returns before touching an MLX array, preserving the normal lazy execution
path.  Enabled runs intentionally materialize sampled router indices so the
result is measurement evidence, not a production cache policy.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter, OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence


_LOCALITY_LANE: ContextVar[str] = ContextVar(
    "mtplx_expert_locality_lane",
    default="model_forward",
)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"", "0", "false", "off", "no"}


def expert_locality_enabled() -> bool:
    return _env_truthy("MTPLX_EXPERT_LOCALITY", False)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, int(default))
    try:
        return max(minimum, int(str(raw).strip()))
    except (TypeError, ValueError):
        return max(minimum, int(default))


def _cache_capacities() -> tuple[int, ...]:
    raw = str(os.environ.get("MTPLX_EXPERT_LOCALITY_CACHE_SIZES") or "16,32,64,96,128")
    values: set[int] = set()
    for item in raw.split(","):
        try:
            values.add(max(1, int(item.strip())))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(values or {16, 32, 64, 96, 128}))


def _flatten_python(value: Any) -> Iterator[int]:
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield int(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_python(item)
        return
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            yield from _flatten_python(tolist())
        except Exception:
            return


def _rows_python(value: Any) -> list[tuple[int, ...]]:
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            value = tolist()
        except Exception:
            return []
    if isinstance(value, int):
        return [(int(value),)]
    if not isinstance(value, (list, tuple)):
        return []
    if not value:
        return []
    if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return [tuple(int(item) for item in value)]
    rows: list[tuple[int, ...]] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            flattened = tuple(_flatten_python(item))
            if flattened:
                rows.append(flattened)
        elif isinstance(item, int) and not isinstance(item, bool):
            rows.append((int(item),))
    return rows


def _reuse_bucket(distance: int) -> str:
    value = max(0, int(distance))
    if value <= 1:
        return str(value)
    lower = 2
    upper = 3
    while value > upper:
        lower = upper + 1
        upper = upper * 2 + 1
    return f"{lower}-{upper}"


@dataclass
class _LRUSimulation:
    capacity: int
    entries: OrderedDict[int, None] = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def observe(self, expert: int) -> None:
        key = int(expert)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return
        self.misses += 1
        self.entries[key] = None
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)

    def to_dict(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "capacity": int(self.capacity),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "hit_rate": (self.hits / total) if total else 0.0,
            "resident": len(self.entries),
        }


@dataclass
class _LayerLaneStats:
    layer_id: str
    lane: str
    cache_capacities: tuple[int, ...]
    events: int = 0
    rows: int = 0
    assignments: int = 0
    invalid_assignments: int = 0
    expert_counts: Counter[int] = field(default_factory=Counter)
    last_seen_event: dict[int, int] = field(default_factory=dict)
    reuse_distance: Counter[str] = field(default_factory=Counter)
    consecutive_overlap_sum: float = 0.0
    consecutive_overlap_samples: int = 0
    previous_experts: set[int] = field(default_factory=set)
    lru: dict[int, _LRUSimulation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.lru:
            self.lru = {
                capacity: _LRUSimulation(capacity)
                for capacity in self.cache_capacities
            }

    def observe_rows(self, rows: Sequence[Sequence[int]], num_experts: int | None) -> None:
        self.events += 1
        event_index = self.events
        for row in rows:
            row_set: set[int] = set()
            self.rows += 1
            for raw in row:
                expert = int(raw)
                if expert < 0 or (
                    num_experts is not None and expert >= int(num_experts)
                ):
                    self.invalid_assignments += 1
                    continue
                self.assignments += 1
                self.expert_counts[expert] += 1
                row_set.add(expert)
                previous = self.last_seen_event.get(expert)
                if previous is not None:
                    self.reuse_distance[_reuse_bucket(event_index - previous)] += 1
                self.last_seen_event[expert] = event_index
                for simulation in self.lru.values():
                    simulation.observe(expert)
            if self.previous_experts or row_set:
                union = self.previous_experts | row_set
                overlap = (
                    len(self.previous_experts & row_set) / len(union)
                    if union
                    else 1.0
                )
                self.consecutive_overlap_sum += overlap
                self.consecutive_overlap_samples += 1
            self.previous_experts = row_set

    def _coverage_count(self, fraction: float) -> int:
        if self.assignments <= 0:
            return 0
        target = self.assignments * max(0.0, min(1.0, fraction))
        cumulative = 0
        for index, count in enumerate(
            sorted(self.expert_counts.values(), reverse=True),
            start=1,
        ):
            cumulative += count
            if cumulative >= target:
                return index
        return len(self.expert_counts)

    def to_dict(self) -> dict[str, Any]:
        top = self.expert_counts.most_common(16)
        return {
            "layer_id": self.layer_id,
            "lane": self.lane,
            "events": int(self.events),
            "rows": int(self.rows),
            "assignments": int(self.assignments),
            "invalid_assignments": int(self.invalid_assignments),
            "unique_experts": len(self.expert_counts),
            "consecutive_jaccard": (
                self.consecutive_overlap_sum / self.consecutive_overlap_samples
                if self.consecutive_overlap_samples
                else 0.0
            ),
            "working_set_50": self._coverage_count(0.50),
            "working_set_90": self._coverage_count(0.90),
            "working_set_99": self._coverage_count(0.99),
            "top_experts": [[int(expert), int(count)] for expert, count in top],
            "reuse_distance_events": dict(sorted(self.reuse_distance.items())),
            "lru_simulation": {
                str(capacity): simulation.to_dict()
                for capacity, simulation in sorted(self.lru.items())
            },
        }


class ExpertLocalityTracker:
    """Bounded per-layer/lane routing statistics and LRU simulations."""

    def __init__(
        self,
        *,
        max_events: int = 4096,
        sample_every: int = 1,
        cache_capacities: Sequence[int] | None = None,
    ) -> None:
        self.max_events = max(1, int(max_events))
        self.sample_every = max(1, int(sample_every))
        self.cache_capacities = tuple(
            sorted({max(1, int(value)) for value in (cache_capacities or _cache_capacities())})
        )
        self._lock = threading.Lock()
        self._stats: dict[tuple[str, str], _LayerLaneStats] = {}
        self._calls = 0
        self._accepted_calls = 0
        self._dropped_calls = 0
        self._started_s = time.monotonic()

    def record(
        self,
        indices: Any,
        *,
        layer_id: int | str,
        lane: str | None = None,
        num_experts: int | None = None,
    ) -> bool:
        with self._lock:
            self._calls += 1
            call_index = self._calls
            if call_index % self.sample_every != 0:
                return False
            if self._accepted_calls >= self.max_events:
                self._dropped_calls += 1
                return False
        rows = _rows_python(indices)
        if not rows:
            return False
        key = (str(layer_id), str(lane or _LOCALITY_LANE.get()))
        with self._lock:
            stats = self._stats.get(key)
            if stats is None:
                stats = _LayerLaneStats(
                    layer_id=key[0],
                    lane=key[1],
                    cache_capacities=self.cache_capacities,
                )
                self._stats[key] = stats
            stats.observe_rows(rows, num_experts)
            self._accepted_calls += 1
        return True

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()
            self._calls = 0
            self._accepted_calls = 0
            self._dropped_calls = 0
            self._started_s = time.monotonic()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = [
                stats.to_dict()
                for _, stats in sorted(self._stats.items(), key=lambda item: item[0])
            ]
            return {
                "enabled": True,
                "calls": int(self._calls),
                "accepted_calls": int(self._accepted_calls),
                "dropped_calls": int(self._dropped_calls),
                "max_events": int(self.max_events),
                "sample_every": int(self.sample_every),
                "cache_capacities": list(self.cache_capacities),
                "elapsed_s": max(0.0, time.monotonic() - self._started_s),
                "layers": rows,
            }

    def recommended_capacity(
        self,
        *,
        minimum_hit_rate: float = 0.60,
        lane: str | None = None,
    ) -> int | None:
        target = max(0.0, min(1.0, float(minimum_hit_rate)))
        with self._lock:
            candidates = [
                stats
                for stats in self._stats.values()
                if lane is None or stats.lane == lane
            ]
            if not candidates:
                return None
            for capacity in self.cache_capacities:
                hits = sum(stats.lru[capacity].hits for stats in candidates)
                misses = sum(stats.lru[capacity].misses for stats in candidates)
                if hits + misses and hits / (hits + misses) >= target:
                    return capacity
        return None


_GLOBAL_TRACKER: ExpertLocalityTracker | None = None
_GLOBAL_LOCK = threading.Lock()


def get_expert_locality_tracker() -> ExpertLocalityTracker:
    global _GLOBAL_TRACKER
    with _GLOBAL_LOCK:
        if _GLOBAL_TRACKER is None:
            _GLOBAL_TRACKER = ExpertLocalityTracker(
                max_events=_env_int("MTPLX_EXPERT_LOCALITY_MAX_EVENTS", 4096),
                sample_every=_env_int("MTPLX_EXPERT_LOCALITY_SAMPLE_EVERY", 1),
            )
        return _GLOBAL_TRACKER


def reset_expert_locality_tracker() -> None:
    global _GLOBAL_TRACKER
    with _GLOBAL_LOCK:
        _GLOBAL_TRACKER = None


def record_expert_routes(
    indices: Any,
    *,
    layer_id: int | str,
    lane: str | None = None,
    num_experts: int | None = None,
) -> bool:
    """Record one router output; return before array materialization when off."""

    if not expert_locality_enabled():
        return False
    return get_expert_locality_tracker().record(
        indices,
        layer_id=layer_id,
        lane=lane,
        num_experts=num_experts,
    )


@contextmanager
def expert_locality_lane(lane: str) -> Iterator[None]:
    token = _LOCALITY_LANE.set(str(lane))
    try:
        yield
    finally:
        _LOCALITY_LANE.reset(token)


def expert_locality_metrics() -> dict[str, Any]:
    if not expert_locality_enabled():
        return {"enabled": False}
    return get_expert_locality_tracker().snapshot()


__all__ = [
    "ExpertLocalityTracker",
    "expert_locality_enabled",
    "expert_locality_lane",
    "expert_locality_metrics",
    "get_expert_locality_tracker",
    "record_expert_routes",
    "reset_expert_locality_tracker",
]
