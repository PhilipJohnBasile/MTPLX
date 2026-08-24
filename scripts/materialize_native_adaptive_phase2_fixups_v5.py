#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(value: str, old: str, new: str, *, label: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return value.replace(old, new, 1)


def otlp_inflight_accounting() -> None:
    path = ROOT / "mtplx/otlp_export.py"
    value = path.read_text(encoding="utf-8")
    value = replace_once(
        value,
        "        self._failed_exports = 0\n        self._last_error_type: str | None = None\n",
        "        self._failed_exports = 0\n        self._active_exports = 0\n        self._last_error_type: str | None = None\n",
        label="OTLP active export counter",
    )
    insertion_point = "    def _drain(self, limit: int) -> list[OTLPSpan]:\n"
    tracked = '''    def _send_tracked(self, spans: Sequence[OTLPSpan]) -> None:
        with self._lock:
            self._active_exports += 1
        try:
            self._send(spans)
        finally:
            with self._lock:
                self._active_exports = max(0, self._active_exports - 1)

'''
    value = replace_once(
        value,
        insertion_point,
        tracked + insertion_point,
        label="OTLP tracked sender insertion",
    )

    worker_start = value.index("    def _worker(self) -> None:\n")
    worker_end = value.index("\n    def flush(self, *, timeout_s: float = 5.0) -> bool:\n", worker_start)
    worker = value[worker_start:worker_end]
    send_count = worker.count("self._send(rows)")
    if send_count != 2:
        raise RuntimeError(f"OTLP worker expected two send sites, found {send_count}")
    worker = worker.replace("self._send(rows)", "self._send_tracked(rows)")
    value = value[:worker_start] + worker + value[worker_end:]

    old_flush = '''    def flush(self, *, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while not self._queue.empty() and time.monotonic() < deadline:
            rows = self._drain(self.config.batch_size)
            if rows:
                self._send(rows)
        return self._queue.empty()
'''
    new_flush = '''    def flush(self, *, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        self._wake.set()
        while time.monotonic() < deadline:
            rows = self._drain(self.config.batch_size)
            if rows:
                self._send_tracked(rows)
                continue
            with self._lock:
                active = self._active_exports
            if self._queue.empty() and active == 0:
                return True
            time.sleep(0.005)
        with self._lock:
            active = self._active_exports
        return self._queue.empty() and active == 0
'''
    value = replace_once(value, old_flush, new_flush, label="OTLP flush")
    value = replace_once(
        value,
        '                "failed_exports": self._failed_exports,\n',
        '                "failed_exports": self._failed_exports,\n                "active_exports": self._active_exports,\n',
        label="OTLP snapshot active exports",
    )
    path.write_text(value, encoding="utf-8")


def expert_locality_ingestion() -> None:
    path = ROOT / "mtplx/expert_residency.py"
    value = path.read_text(encoding="utf-8")
    start = value.index("    def ingest_locality_snapshot(self, snapshot: Mapping[str, Any]) -> int:\n")
    end = value.index("\n    def _prune_locked", start)
    replacement = '''    def ingest_locality_snapshot(self, snapshot: Mapping[str, Any]) -> int:
        """Ingest flat, nested, keyed, and per-layer locality snapshots."""

        aggregated: dict[ExpertRef, float] = {}
        seen_objects: set[int] = set()
        count_keys = (
            "count",
            "frequency",
            "hits",
            "uses",
            "observations",
            "weight",
        )
        vector_keys = (
            "expert_counts",
            "expert_frequency",
            "expert_frequencies",
            "frequencies",
            "counts",
            "histogram",
        )

        def integer(value: Any) -> int | None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def number(value: Any) -> float | None:
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            return result if math.isfinite(result) and result > 0 else None

        def add(layer: Any, expert: Any, weight: Any) -> None:
            layer_id = integer(layer)
            expert_id = integer(expert)
            amount = number(weight)
            if layer_id is None or expert_id is None or amount is None:
                return
            if layer_id < 0 or expert_id < 0:
                return
            ref = ExpertRef(layer_id, expert_id)
            aggregated[ref] = aggregated.get(ref, 0.0) + amount

        def consume_vector(value: Any, layer_hint: int | None) -> bool:
            if layer_hint is None:
                return False
            consumed = False
            if isinstance(value, Mapping):
                for expert, weight in value.items():
                    add(layer_hint, expert, weight)
                    consumed = True
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for expert, weight in enumerate(value):
                    add(layer_hint, expert, weight)
                    consumed = True
            return consumed

        def walk(value: Any, layer_hint: int | None = None) -> None:
            if isinstance(value, (Mapping, list, tuple)):
                identity = id(value)
                if identity in seen_objects:
                    return
                seen_objects.add(identity)
            if isinstance(value, Mapping):
                current_layer = integer(
                    value.get("layer", value.get("layer_id", layer_hint))
                )
                expert = value.get("expert", value.get("expert_id"))
                if expert is not None and current_layer is not None:
                    weight = next(
                        (value[key] for key in count_keys if key in value),
                        1.0,
                    )
                    add(current_layer, expert, weight)

                consumed_keys: set[str] = set()
                for key in vector_keys:
                    if key in value and consume_vector(value[key], current_layer):
                        consumed_keys.add(key)

                for key, child in value.items():
                    key_text = str(key)
                    if key_text in consumed_keys or key_text in count_keys:
                        continue
                    if current_layer is None and ":" in key_text:
                        left, right = key_text.split(":", 1)
                        amount = number(child)
                        if amount is not None:
                            add(left, right, amount)
                            continue
                    next_layer = current_layer
                    if key_text.isdigit() and isinstance(child, Mapping):
                        next_layer = integer(key_text)
                    walk(child, next_layer)
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for child in value:
                    walk(child, layer_hint)

        walk(snapshot)
        for ref, weight in sorted(aggregated.items()):
            self.observe(ref.layer, [ref.expert], weights=[weight])
        return len(aggregated)
'''
    value = value[:start] + replacement + value[end:]

    old_index = '''            try:
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
'''
    new_index = '''            try:
                if isinstance(layers, Mapping):
                    layer_items = [
                        (int(key), item) for key, item in layers.items()
                    ]
                else:
                    layer_items = list(enumerate(layers))
            except Exception:
                self._index = result
                return result
            for layer_id, layer in layer_items:
                container = self._candidate_expert_container(layer)
                if container is None:
                    continue
                try:
                    if isinstance(container, Mapping):
                        expert_items = [
                            (int(key), item) for key, item in container.items()
                        ]
                    else:
                        expert_items = list(enumerate(container))
                except Exception:
                    continue
                for expert_id, expert in expert_items:
'''
    value = replace_once(value, old_index, new_index, label="mapping expert index")
    path.write_text(value, encoding="utf-8")


def unified_kv_capability() -> None:
    path = ROOT / "mtplx/unified_memory.py"
    value = path.read_text(encoding="utf-8")
    value = replace_once(
        value,
        '    _ORDER = ("session_bank", "expert_residency")',
        '    _ORDER = ("session_bank", "expert_residency", "kv_headroom")',
        label="unified order",
    )
    value = replace_once(
        value,
        '''        targets = {
            "session_bank": plan.session_bank_budget_bytes,
            "expert_residency": plan.expert_budget_bytes,
        }
''',
        '''        targets = {
            "session_bank": plan.session_bank_budget_bytes,
            "expert_residency": plan.expert_budget_bytes,
            "kv_headroom": plan.kv_headroom_bytes,
        }
''',
        label="unified targets",
    )
    path.write_text(value, encoding="utf-8")


def native_kv_consumer() -> None:
    path = ROOT / "mtplx/native_adaptive.py"
    value = path.read_text(encoding="utf-8")
    anchor = "\ndef _locality_snapshot(state: Any) -> Mapping[str, Any] | None:\n"
    block = '''

class _KVHeadroomBudgetConsumer:
    name = "kv_headroom"

    def __init__(self, getter: Callable[[], int], setter: Callable[..., Any]) -> None:
        self.getter = getter
        self.setter = setter

    def current_budget_bytes(self) -> int:
        return max(0, int(self.getter() or 0))

    def apply_budget_bytes(self, value: int, *, reason: str) -> int:
        requested = max(0, int(value))
        try:
            result = self.setter(requested, reason=reason)
        except TypeError:
            result = self.setter(requested)
        return requested if result is None else max(0, int(result))


def _kv_headroom_consumer(state: Any) -> _KVHeadroomBudgetConsumer | None:
    candidates = (
        state,
        getattr(state, "kv_cache_manager", None),
        getattr(state, "cache_manager", None),
        getattr(getattr(state, "runtime", None), "kv_cache_manager", None),
    )
    setter_names = (
        "set_kv_headroom_bytes",
        "apply_kv_headroom_bytes",
        "rebalance_kv_headroom",
    )
    for target in candidates:
        if target is None:
            continue
        setter = next(
            (
                getattr(target, name, None)
                for name in setter_names
                if callable(getattr(target, name, None))
            ),
            None,
        )
        if setter is None:
            continue
        getter_method = getattr(target, "current_kv_headroom_bytes", None)
        if callable(getter_method):
            getter = getter_method
        elif hasattr(target, "kv_headroom_bytes"):
            getter = lambda target=target: int(
                getattr(target, "kv_headroom_bytes", 0) or 0
            )
        else:
            continue
        return _KVHeadroomBudgetConsumer(getter, setter)
    return None
'''
    value = replace_once(value, anchor, block + anchor, label="KV consumer insertion")
    value = replace_once(
        value,
        '''        if bank is not None:
            consumers.append(SessionBankBudgetConsumer(bank))
        consumers.append(
''',
        '''        if bank is not None:
            consumers.append(SessionBankBudgetConsumer(bank))
        kv_consumer = _kv_headroom_consumer(state)
        if kv_consumer is not None:
            consumers.append(kv_consumer)
        consumers.append(
''',
        label="KV consumer wiring",
    )
    path.write_text(value, encoding="utf-8")


def append_tests() -> None:
    expert_path = ROOT / "tests/test_expert_residency.py"
    expert = expert_path.read_text(encoding="utf-8")
    expert += '''


def test_ingest_phase_one_nested_layer_vectors():
    controller = configured()
    accepted = controller.ingest_locality_snapshot(
        {
            "layers": [
                {"layer_id": 2, "expert_counts": [0, 4, 1]},
                {
                    "layer": 3,
                    "experts": [
                        {"expert_id": 5, "observations": 7},
                        {"expert": 6, "frequency": 2},
                    ],
                },
            ]
        }
    )
    assert accepted == 4
    assert controller.snapshot()["tracked_experts"] == 4
'''
    expert_path.write_text(expert, encoding="utf-8")

    unified_path = ROOT / "tests/test_unified_memory.py"
    unified = unified_path.read_text(encoding="utf-8")
    unified += '''


def test_apply_mutates_explicit_kv_headroom_consumer():
    coordinator = UnifiedMemoryCoordinator(config())
    plan = coordinator.plan(sample(), safe=True, now_s=10.0)
    session = FakeConsumer("session_bank", 10 * GIB)
    expert = FakeConsumer("expert_residency", 4 * GIB)
    kv = FakeConsumer("kv_headroom", 2 * GIB)
    receipt = coordinator.apply(plan, [session, expert, kv], safe=True)
    assert receipt.applied is True
    assert kv.budget == plan.kv_headroom_bytes
    assert any(item.consumer == "kv_headroom" for item in receipt.mutations)
'''
    unified_path.write_text(unified, encoding="utf-8")


def main() -> None:
    otlp_inflight_accounting()
    expert_locality_ingestion()
    unified_kv_capability()
    native_kv_consumer()
    append_tests()


if __name__ == "__main__":
    main()
