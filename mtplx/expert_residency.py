"""Apple-native expert warm-set and residency control for MTPLX.

The controller is intentionally backend-neutral.  It converts bounded expert
locality observations into a byte-budgeted warm-set plan, then applies that
plan only through an explicit backend capability.  It never changes router
outputs and never assumes CUDA-style device/host placement on Apple unified
memory.

Backends may provide true residency, rematerialization, or page-warming.  The
built-in :class:`MLXMaterializationBackend` is deliberately labelled
``materialize_only``: it evaluates selected expert arrays to warm lazy MLX
state but does not claim that individual unified-memory pages can be pinned or
unloaded.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, order=True)
class ExpertRef:
    """Stable expert identity within one loaded model."""

    layer: int
    expert: int

    def __post_init__(self) -> None:
        if self.layer < 0 or self.expert < 0:
            raise ValueError("expert coordinates must be non-negative")

    @classmethod
    def coerce(cls, value: Any) -> "ExpertRef":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(layer=int(value["layer"]), expert=int(value["expert"]))
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return cls(layer=int(value[0]), expert=int(value[1]))
        raise TypeError(f"unsupported expert reference: {value!r}")

    def to_dict(self) -> dict[str, int]:
        return {"layer": self.layer, "expert": self.expert}


@dataclass(frozen=True)
class ExpertResidencyConfig:
    """Bounded planning policy.

    ``half_life_s`` decays old routing evidence.  ``hysteresis_ratio`` prevents
    churn near the budget boundary.  Per-tick limits cap page-fault and cache
    disruption even when a locality distribution changes abruptly.
    """

    enabled: bool = False
    budget_bytes: int = 0
    minimum_observations: int = 8
    half_life_s: float = 45.0
    hysteresis_ratio: float = 0.12
    maximum_tracked_experts: int = 4096
    maximum_prefetch_per_tick: int = 8
    maximum_evict_per_tick: int = 8
    minimum_tick_interval_s: float = 0.25
    preserve_resident_bonus: float = 0.08

    def __post_init__(self) -> None:
        if self.budget_bytes < 0:
            raise ValueError("budget_bytes must be non-negative")
        if self.minimum_observations < 1:
            raise ValueError("minimum_observations must be at least 1")
        if self.half_life_s <= 0:
            raise ValueError("half_life_s must be positive")
        if not 0 <= self.hysteresis_ratio < 1:
            raise ValueError("hysteresis_ratio must be in [0, 1)")
        if self.maximum_tracked_experts < 1:
            raise ValueError("maximum_tracked_experts must be at least 1")
        if self.maximum_prefetch_per_tick < 0 or self.maximum_evict_per_tick < 0:
            raise ValueError("per-tick limits must be non-negative")
        if self.minimum_tick_interval_s < 0:
            raise ValueError("minimum_tick_interval_s must be non-negative")
        if self.preserve_resident_bonus < 0:
            raise ValueError("preserve_resident_bonus must be non-negative")


@dataclass
class _Score:
    value: float = 0.0
    observations: int = 0
    last_update_s: float = 0.0
    last_seen_s: float = 0.0


@dataclass(frozen=True)
class ExpertResidencyPlan:
    plan_id: str
    generated_at_s: float
    budget_bytes: int
    target: tuple[ExpertRef, ...]
    keep: tuple[ExpertRef, ...]
    prefetch: tuple[ExpertRef, ...]
    evict: tuple[ExpertRef, ...]
    target_bytes: int
    observed_experts: int
    eligible: bool
    reason: str
    backend_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "generated_at_s": self.generated_at_s,
            "budget_bytes": self.budget_bytes,
            "target": [item.to_dict() for item in self.target],
            "keep": [item.to_dict() for item in self.keep],
            "prefetch": [item.to_dict() for item in self.prefetch],
            "evict": [item.to_dict() for item in self.evict],
            "target_bytes": self.target_bytes,
            "observed_experts": self.observed_experts,
            "eligible": self.eligible,
            "reason": self.reason,
            "backend_mode": self.backend_mode,
        }


@dataclass(frozen=True)
class ExpertResidencyReceipt:
    plan_id: str
    applied: bool
    prefetched: tuple[ExpertRef, ...] = ()
    evicted: tuple[ExpertRef, ...] = ()
    failed: tuple[ExpertRef, ...] = ()
    reason: str = ""
    backend_mode: str = "unavailable"
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "applied": self.applied,
            "prefetched": [item.to_dict() for item in self.prefetched],
            "evicted": [item.to_dict() for item in self.evicted],
            "failed": [item.to_dict() for item in self.failed],
            "reason": self.reason,
            "backend_mode": self.backend_mode,
            "duration_ms": self.duration_ms,
        }


@runtime_checkable
class ExpertResidencyBackend(Protocol):
    """Explicit capability consumed by the planner.

    Implementations must be idempotent.  ``resident_experts`` should describe
    the backend-managed warm/resident set, not every expert tensor owned by the
    model object.
    """

    mode: str

    def resident_experts(self) -> Iterable[ExpertRef]: ...

    def expert_nbytes(self, expert: ExpertRef) -> int: ...

    def prefetch_experts(self, experts: Sequence[ExpertRef]) -> Iterable[ExpertRef]: ...

    def evict_experts(self, experts: Sequence[ExpertRef]) -> Iterable[ExpertRef]: ...


class MLXMaterializationBackend:
    """Best-effort lazy-array warmer for an MLX model.

    This backend does not advertise hard eviction.  It discovers common expert
    containers, evaluates selected parameter arrays, and maintains a bounded
    logical warm set.  The distinction is surfaced as ``materialize_only`` so
    operators are not told that macOS unified-memory pages were pinned.
    """

    mode = "materialize_only"

    def __init__(self, model: Any, *, maximum_arrays_per_expert: int = 64) -> None:
        self._model = model
        self._maximum_arrays_per_expert = max(1, int(maximum_arrays_per_expert))
        self._index: dict[ExpertRef, tuple[Any, ...]] | None = None
        self._sizes: dict[ExpertRef, int] = {}
        self._warm: OrderedDict[ExpertRef, None] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _iter_arrays(value: Any, *, limit: int) -> tuple[Any, ...]:
        arrays: list[Any] = []
        seen: set[int] = set()

        def visit(item: Any, depth: int) -> None:
            if len(arrays) >= limit or depth > 5 or item is None:
                return
            identity = id(item)
            if identity in seen:
                return
            seen.add(identity)
            if hasattr(item, "shape") and hasattr(item, "dtype"):
                arrays.append(item)
                return
            if isinstance(item, Mapping):
                for child in item.values():
                    visit(child, depth + 1)
                return
            if isinstance(item, (list, tuple)):
                for child in item:
                    visit(child, depth + 1)
                return
            parameters = getattr(item, "parameters", None)
            if callable(parameters):
                try:
                    visit(parameters(), depth + 1)
                except Exception:
                    pass
            namespace = getattr(item, "__dict__", None)
            if isinstance(namespace, dict):
                for name, child in namespace.items():
                    if name.startswith("_"):
                        continue
                    visit(child, depth + 1)

        visit(value, 0)
        return tuple(arrays)

    @staticmethod
    def _candidate_expert_container(layer: Any) -> Any | None:
        paths = (
            ("mlp", "experts"),
            ("block_sparse_moe", "experts"),
            ("moe", "experts"),
            ("feed_forward", "experts"),
            ("experts",),
        )
        for path in paths:
            value = layer
            try:
                for part in path:
                    value = getattr(value, part)
            except Exception:
                continue
            if value is not None:
                return value
        return None

    def _build_index(self) -> dict[ExpertRef, tuple[Any, ...]]:
        with self._lock:
            if self._index is not None:
                return self._index
            result: dict[ExpertRef, tuple[Any, ...]] = {}
            layers = getattr(self._model, "layers", None)
            if layers is None:
                nested = getattr(self._model, "model", None)
                layers = getattr(nested, "layers", None)
            if layers is None:
                self._index = result
                return result
            try:
                layer_values = list(layers)
            except Exception:
                self._index = result
                return result
            for layer_id, layer in enumerate(layer_values):
                container = self._candidate_expert_container(layer)
                if container is None:
                    continue
                try:
                    expert_values = list(container)
                except Exception:
                    continue
                for expert_id, expert in enumerate(expert_values):
                    arrays = self._iter_arrays(
                        expert,
                        limit=self._maximum_arrays_per_expert,
                    )
                    if arrays:
                        result[ExpertRef(layer_id, expert_id)] = arrays
            self._index = result
            return result

    @staticmethod
    def _array_nbytes(array: Any) -> int:
        nbytes = getattr(array, "nbytes", None)
        if isinstance(nbytes, int):
            return max(0, nbytes)
        shape = getattr(array, "shape", ())
        size = 1
        try:
            for dim in shape:
                size *= int(dim)
        except Exception:
            return 0
        dtype = getattr(array, "dtype", None)
        itemsize = getattr(dtype, "itemsize", None)
        if callable(itemsize):
            try:
                itemsize = itemsize()
            except Exception:
                itemsize = None
        try:
            return max(0, size * int(itemsize or 2))
        except Exception:
            return 0

    def resident_experts(self) -> Iterable[ExpertRef]:
        with self._lock:
            return tuple(self._warm)

    def expert_nbytes(self, expert: ExpertRef) -> int:
        expert = ExpertRef.coerce(expert)
        with self._lock:
            if expert in self._sizes:
                return self._sizes[expert]
            arrays = self._build_index().get(expert, ())
            total = sum(self._array_nbytes(array) for array in arrays)
            self._sizes[expert] = total
            return total

    def prefetch_experts(self, experts: Sequence[ExpertRef]) -> Iterable[ExpertRef]:
        try:
            import mlx.core as mx
        except Exception:
            return ()
        completed: list[ExpertRef] = []
        with self._lock:
            index = self._build_index()
            for raw in experts:
                expert = ExpertRef.coerce(raw)
                arrays = index.get(expert)
                if not arrays:
                    continue
                try:
                    mx.eval(*arrays)
                except Exception:
                    continue
                self._warm.pop(expert, None)
                self._warm[expert] = None
                completed.append(expert)
        return tuple(completed)

    def evict_experts(self, experts: Sequence[ExpertRef]) -> Iterable[ExpertRef]:
        # Logical eviction only.  Individual model-owned arrays cannot be
        # unloaded safely by generic MLX code.
        completed: list[ExpertRef] = []
        with self._lock:
            for raw in experts:
                expert = ExpertRef.coerce(raw)
                if expert in self._warm:
                    self._warm.pop(expert, None)
                    completed.append(expert)
        return tuple(completed)


class ExpertResidencyController:
    """Convert locality evidence into a bounded, hysteretic warm set."""

    def __init__(self, config: ExpertResidencyConfig | None = None) -> None:
        self.config = config or ExpertResidencyConfig()
        self._scores: dict[ExpertRef, _Score] = {}
        self._total_observations = 0
        self._last_tick_s = 0.0
        self._last_plan: ExpertResidencyPlan | None = None
        self._last_receipt: ExpertResidencyReceipt | None = None
        self._lock = threading.RLock()

    @staticmethod
    def _now(value: float | None = None) -> float:
        return time.monotonic() if value is None else float(value)

    def _decayed(self, score: _Score, now_s: float) -> float:
        if score.last_update_s <= 0:
            return 0.0
        elapsed = max(0.0, now_s - score.last_update_s)
        return score.value * math.exp2(-elapsed / self.config.half_life_s)

    def observe(
        self,
        layer: int,
        experts: Iterable[int],
        *,
        weights: Iterable[float] | None = None,
        now_s: float | None = None,
    ) -> None:
        now = self._now(now_s)
        expert_values = tuple(int(item) for item in experts)
        if weights is None:
            weight_values = (1.0,) * len(expert_values)
        else:
            weight_values = tuple(float(item) for item in weights)
            if len(weight_values) != len(expert_values):
                raise ValueError("weights must match experts")
        with self._lock:
            for expert_id, weight in zip(expert_values, weight_values, strict=True):
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError("expert weights must be finite and non-negative")
                ref = ExpertRef(int(layer), expert_id)
                score = self._scores.setdefault(ref, _Score(last_update_s=now))
                score.value = self._decayed(score, now) + weight
                score.observations += 1
                score.last_update_s = now
                score.last_seen_s = now
                self._total_observations += 1
            self._prune_locked(now)

    def observe_refs(
        self,
        experts: Iterable[ExpertRef | tuple[int, int] | Mapping[str, int]],
        *,
        now_s: float | None = None,
    ) -> None:
        grouped: dict[int, list[int]] = {}
        for raw in experts:
            ref = ExpertRef.coerce(raw)
            grouped.setdefault(ref.layer, []).append(ref.expert)
        for layer, values in grouped.items():
            self.observe(layer, values, now_s=now_s)

    def ingest_locality_snapshot(self, snapshot: Mapping[str, Any]) -> int:
        """Accept common telemetry shapes without coupling to one collector.

        Supported rows include ``{layer, expert, count}``, a mapping keyed by
        ``"layer:expert"``, and matrices under ``experts``/``frequency``.
        Returns the number of weighted observations accepted.
        """

        accepted = 0
        candidates = snapshot.get("experts") or snapshot.get("frequency") or snapshot.get(
            "expert_counts"
        )
        rows: list[tuple[int, int, float]] = []
        if isinstance(candidates, Mapping):
            for key, value in candidates.items():
                if isinstance(key, str) and ":" in key:
                    left, right = key.split(":", 1)
                    try:
                        rows.append((int(left), int(right), float(value)))
                    except (TypeError, ValueError):
                        continue
                elif isinstance(value, Mapping):
                    try:
                        rows.append(
                            (
                                int(value.get("layer", key)),
                                int(value["expert"]),
                                float(value.get("count", value.get("weight", 1.0))),
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
        elif isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            for row in candidates:
                if not isinstance(row, Mapping):
                    continue
                try:
                    rows.append(
                        (
                            int(row["layer"]),
                            int(row["expert"]),
                            float(row.get("count", row.get("weight", 1.0))),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        for layer, expert, weight in rows:
            if weight <= 0 or not math.isfinite(weight):
                continue
            self.observe(layer, [expert], weights=[weight])
            accepted += 1
        return accepted

    def _prune_locked(self, now_s: float) -> None:
        excess = len(self._scores) - self.config.maximum_tracked_experts
        if excess <= 0:
            return
        ranked = sorted(
            self._scores.items(),
            key=lambda item: (self._decayed(item[1], now_s), item[1].last_seen_s),
        )
        for ref, _ in ranked[:excess]:
            self._scores.pop(ref, None)

    @staticmethod
    def _plan_id(
        *, budget_bytes: int, target: Sequence[ExpertRef], generated_at_s: float
    ) -> str:
        import hashlib

        payload = ";".join(f"{item.layer}:{item.expert}" for item in target)
        raw = f"{budget_bytes}|{generated_at_s:.6f}|{payload}".encode()
        return hashlib.sha256(raw).hexdigest()[:24]

    def plan(
        self,
        backend: ExpertResidencyBackend | None,
        *,
        budget_bytes: int | None = None,
        now_s: float | None = None,
    ) -> ExpertResidencyPlan:
        now = self._now(now_s)
        budget = self.config.budget_bytes if budget_bytes is None else max(0, int(budget_bytes))
        mode = getattr(backend, "mode", "unavailable") if backend is not None else "unavailable"
        with self._lock:
            if not self.config.enabled:
                reason = "disabled"
                eligible = False
            elif backend is None:
                reason = "backend_unavailable"
                eligible = False
            elif self._total_observations < self.config.minimum_observations:
                reason = "insufficient_observations"
                eligible = False
            elif now - self._last_tick_s < self.config.minimum_tick_interval_s:
                reason = "tick_interval"
                eligible = False
            elif budget <= 0:
                reason = "zero_budget"
                eligible = True
            else:
                reason = "ready"
                eligible = True

            resident = set(ExpertRef.coerce(item) for item in backend.resident_experts()) if backend else set()
            ranked: list[tuple[float, ExpertRef, int]] = []
            for ref, score in self._scores.items():
                nbytes = max(1, int(backend.expert_nbytes(ref))) if backend else 1
                value = self._decayed(score, now)
                if ref in resident:
                    value *= 1.0 + self.config.preserve_resident_bonus
                ranked.append((value / nbytes, ref, nbytes))
            ranked.sort(key=lambda item: (-item[0], item[1].layer, item[1].expert))

            target: list[ExpertRef] = []
            target_bytes = 0
            effective_budget = int(budget * (1.0 + self.config.hysteresis_ratio))
            for _, ref, nbytes in ranked:
                cap = effective_budget if ref in resident else budget
                if target_bytes + nbytes > cap:
                    continue
                target.append(ref)
                target_bytes += nbytes

            target_set = set(target)
            keep = tuple(sorted(target_set & resident))
            prefetch = tuple(
                item
                for item in target
                if item not in resident
            )[: self.config.maximum_prefetch_per_tick]
            evict_candidates = sorted(
                resident - target_set,
                key=lambda ref: (
                    self._decayed(self._scores.get(ref, _Score()), now),
                    ref.layer,
                    ref.expert,
                ),
            )
            evict = tuple(evict_candidates[: self.config.maximum_evict_per_tick])
            plan = ExpertResidencyPlan(
                plan_id=self._plan_id(
                    budget_bytes=budget,
                    target=target,
                    generated_at_s=now,
                ),
                generated_at_s=now,
                budget_bytes=budget,
                target=tuple(target),
                keep=keep,
                prefetch=prefetch,
                evict=evict,
                target_bytes=target_bytes,
                observed_experts=len(self._scores),
                eligible=eligible,
                reason=reason,
                backend_mode=mode,
            )
            self._last_plan = plan
            if eligible:
                self._last_tick_s = now
            return plan

    def apply(
        self,
        plan: ExpertResidencyPlan,
        backend: ExpertResidencyBackend | None,
        *,
        safe: bool,
    ) -> ExpertResidencyReceipt:
        started = time.perf_counter()
        if not plan.eligible:
            receipt = ExpertResidencyReceipt(
                plan_id=plan.plan_id,
                applied=False,
                reason=plan.reason,
                backend_mode=plan.backend_mode,
            )
        elif not safe:
            receipt = ExpertResidencyReceipt(
                plan_id=plan.plan_id,
                applied=False,
                reason="unsafe_point",
                backend_mode=plan.backend_mode,
            )
        elif backend is None:
            receipt = ExpertResidencyReceipt(
                plan_id=plan.plan_id,
                applied=False,
                reason="backend_unavailable",
                backend_mode="unavailable",
            )
        else:
            prefetched = tuple(
                ExpertRef.coerce(item) for item in backend.prefetch_experts(plan.prefetch)
            )
            # Only evict after prefetch attempts complete.  This avoids throwing
            # away a useful warm expert when the replacement cannot materialize.
            evicted = tuple(ExpertRef.coerce(item) for item in backend.evict_experts(plan.evict))
            requested = set(plan.prefetch)
            failed = tuple(sorted(requested - set(prefetched)))
            receipt = ExpertResidencyReceipt(
                plan_id=plan.plan_id,
                applied=bool(prefetched or evicted or (not plan.prefetch and not plan.evict)),
                prefetched=prefetched,
                evicted=evicted,
                failed=failed,
                reason="applied" if not failed else "partial",
                backend_mode=getattr(backend, "mode", "custom"),
            )
        duration_ms = (time.perf_counter() - started) * 1000.0
        receipt = ExpertResidencyReceipt(**{**asdict(receipt), "duration_ms": duration_ms})
        with self._lock:
            self._last_receipt = receipt
        return receipt

    def snapshot(self, backend: ExpertResidencyBackend | None = None) -> dict[str, Any]:
        with self._lock:
            resident = ()
            if backend is not None:
                try:
                    resident = tuple(ExpertRef.coerce(item) for item in backend.resident_experts())
                except Exception:
                    resident = ()
            return {
                "available": True,
                "enabled": self.config.enabled,
                "backend_mode": getattr(backend, "mode", "unavailable")
                if backend is not None
                else "unavailable",
                "tracked_experts": len(self._scores),
                "total_observations": self._total_observations,
                "resident_experts": len(resident),
                "budget_bytes": self.config.budget_bytes,
                "router_mutation": False,
                "last_plan": self._last_plan.to_dict() if self._last_plan else None,
                "last_receipt": self._last_receipt.to_dict() if self._last_receipt else None,
            }
