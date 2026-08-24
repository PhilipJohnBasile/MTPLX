#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(value: str, old: str, new: str, *, label: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return value.replace(old, new, 1)


def expert_partial_failure() -> None:
    path = ROOT / "mtplx/expert_residency.py"
    value = path.read_text(encoding="utf-8")
    old = '''            prefetched = tuple(
                ExpertRef.coerce(item) for item in backend.prefetch_experts(plan.prefetch)
            )
            # Only evict after prefetch attempts complete.  This avoids throwing
            # away a useful warm expert when the replacement cannot materialize.
            evicted = tuple(ExpertRef.coerce(item) for item in backend.evict_experts(plan.evict))
            requested = set(plan.prefetch)
            failed = tuple(sorted(requested - set(prefetched)))
'''
    new = '''            prefetched = tuple(
                ExpertRef.coerce(item) for item in backend.prefetch_experts(plan.prefetch)
            )
            requested = set(plan.prefetch)
            failed = tuple(sorted(requested - set(prefetched)))
            # A pure budget shrink may evict every planned cold expert.  During
            # replacement, however, evict at most one cold expert per successful
            # prefetch so a failed materialization cannot reduce the useful warm set.
            evict_request = (
                plan.evict
                if not plan.prefetch
                else plan.evict[: len(prefetched)]
            )
            evicted = tuple(
                ExpertRef.coerce(item)
                for item in backend.evict_experts(evict_request)
            )
'''
    value = replace_once(value, old, new, label="expert partial prefetch apply")
    path.write_text(value, encoding="utf-8")

    test_path = ROOT / "tests/test_expert_residency.py"
    tests = test_path.read_text(encoding="utf-8")
    tests += '''


class PartialBackend(FakeBackend):
    def __init__(self, sizes, resident=(), failed=()):
        super().__init__(sizes, resident=resident)
        self.failed = set(failed)

    def prefetch_experts(self, experts):
        values = tuple(ExpertRef.coerce(item) for item in experts)
        self.prefetch_calls.append(values)
        completed = tuple(item for item in values if item not in self.failed)
        self.resident.update(completed)
        return completed


def test_failed_prefetch_does_not_evict_every_replacement_candidate():
    hot_a = ExpertRef(0, 0)
    hot_b = ExpertRef(0, 1)
    cold_a = ExpertRef(0, 2)
    cold_b = ExpertRef(0, 3)
    backend = PartialBackend(
        {item: 100 for item in (hot_a, hot_b, cold_a, cold_b)},
        resident=(cold_a, cold_b),
        failed=(hot_b,),
    )
    controller = configured(budget_bytes=200)
    controller.observe_refs([hot_a] * 10 + [hot_b] * 9)
    plan = controller.plan(backend)
    assert len(plan.prefetch) == 2
    assert len(plan.evict) == 2

    receipt = controller.apply(plan, backend, safe=True)

    assert receipt.prefetched == (hot_a,)
    assert receipt.failed == (hot_b,)
    assert len(receipt.evicted) == 1
    assert len(backend.resident) == 2
'''
    test_path.write_text(tests, encoding="utf-8")


def bounded_policy_executor_and_response_rejection() -> None:
    path = ROOT / "mtplx/policy_hooks.py"
    value = path.read_text(encoding="utf-8")
    value = replace_once(
        value,
        '''import math
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
''',
        '''import math
import queue
import threading
from concurrent.futures import Future, TimeoutError
''',
        label="policy imports",
    )
    value = replace_once(
        value,
        '''    maximum_annotation_bytes: int = 4096
    default_failure_mode: FailureMode = FailureMode.OPEN
''',
        '''    maximum_annotation_bytes: int = 4096
    maximum_workers: int = 4
    maximum_pending_tasks: int = 16
    default_failure_mode: FailureMode = FailureMode.OPEN
''',
        label="policy executor config fields",
    )
    value = replace_once(
        value,
        '''        if self.maximum_annotations < 1 or self.maximum_annotation_bytes < 1:
            raise ValueError("annotation limits must be positive")
''',
        '''        if self.maximum_annotations < 1 or self.maximum_annotation_bytes < 1:
            raise ValueError("annotation limits must be positive")
        if self.maximum_workers < 1 or self.maximum_pending_tasks < 1:
            raise ValueError("policy executor limits must be positive")
''',
        label="policy executor config validation",
    )
    anchor = "\n\nclass PolicyBus:\n"
    executor = '''

class PolicyExecutorSaturated(RuntimeError):
    pass


class _BoundedHookExecutor:
    """Fixed daemon-worker executor; timed-out hooks cannot create new threads."""

    _STOP = object()

    def __init__(self, *, workers: int, pending: int) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(workers, pending))
        self._closed = False
        self._lock = threading.Lock()
        self._threads = tuple(
            threading.Thread(
                target=self._worker,
                name=f"mtplx-policy-{index}",
                daemon=True,
            )
            for index in range(workers)
        )
        for thread in self._threads:
            thread.start()

    def submit(self, callback: PolicyCallable, *args: Any) -> Future[Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("policy executor is closed")
        future: Future[Any] = Future()
        try:
            self._queue.put_nowait((future, callback, args))
        except queue.Full as exc:
            raise PolicyExecutorSaturated("policy executor queue is full") from exc
        return future

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._STOP:
                return
            future, callback, args = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = callback(*args)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is self._STOP:
                continue
            future, _callback, _args = item
            future.cancel()
        for _ in self._threads:
            try:
                self._queue.put_nowait(self._STOP)
            except queue.Full:
                break
'''
    value = replace_once(value, anchor, executor + anchor, label="bounded policy executor")
    value = replace_once(
        value,
        '''        self._hooks: dict[str, _Registration] = {}
        self._lock = threading.RLock()
''',
        '''        self._hooks: dict[str, _Registration] = {}
        self._lock = threading.RLock()
        self._executor = _BoundedHookExecutor(
            workers=min(self.config.maximum_workers, self.config.maximum_hooks),
            pending=self.config.maximum_pending_tasks,
        )
''',
        label="policy executor initialization",
    )
    old_run = '''    @staticmethod
    def _run_one(registration: _Registration, value: Any, context: PolicyContext) -> HookResult:
        isolated_value = deepcopy(value)
        isolated_context = deepcopy(context)
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mtplx-policy")
        future = pool.submit(registration.callback, isolated_value, isolated_context)
        try:
            raw = future.result(timeout=registration.timeout_s)
            return _coerce_result(raw)
        except TimeoutError:
            future.cancel()
            raise
        finally:
            # Python cannot safely kill a running callback.  Detach it so a
            # timed-out policy cannot hold the serving caller open.
            pool.shutdown(wait=False, cancel_futures=True)
'''
    new_run = '''    def _run_one(
        self,
        registration: _Registration,
        value: Any,
        context: PolicyContext,
    ) -> HookResult:
        isolated_value = deepcopy(value)
        isolated_context = deepcopy(context)
        future = self._executor.submit(
            registration.callback,
            isolated_value,
            isolated_context,
        )
        try:
            raw = future.result(timeout=registration.timeout_s)
            return _coerce_result(raw)
        except TimeoutError:
            future.cancel()
            raise
'''
    value = replace_once(value, old_run, new_run, label="policy hook execution")
    value = replace_once(
        value,
        '''    def snapshot(self) -> dict[str, Any]:
''',
        '''    def shutdown(self) -> None:
        self._executor.shutdown()

    def snapshot(self) -> dict[str, Any]:
''',
        label="policy shutdown method",
    )
    value = replace_once(
        value,
        '''        if _DEFAULT_BUS is not None:
            _DEFAULT_BUS = None
''',
        '''        if _DEFAULT_BUS is not None:
            _DEFAULT_BUS.shutdown()
        _DEFAULT_BUS = None
''',
        label="default policy reset shutdown",
    )
    path.write_text(value, encoding="utf-8")

    native_path = ROOT / "mtplx/native_adaptive.py"
    native = native_path.read_text(encoding="utf-8")
    old_response = '''                    if outcome.rewritten and isinstance(outcome.value, Mapping):
                        body_value = outcome.value.get("body", body)
                        body = (
                            body_value.encode()
                            if isinstance(body_value, str)
                            else bytes(body_value)
                        )
                        status_code = int(outcome.value.get("status_code", status_code))
                        if response_start is not None:
                            response_start["status"] = status_code
'''
    new_response = '''                    if not outcome.allowed:
                        status_code = outcome.status_code
                        body = json.dumps(
                            {
                                "error": {
                                    "type": "policy_rejection",
                                    "message": outcome.reason,
                                }
                            },
                            separators=(",", ":"),
                        ).encode("utf-8")
                        if response_start is not None:
                            response_start["status"] = status_code
                            headers = [
                                (key, value)
                                for key, value in response_start.get("headers", [])
                                if key.lower() not in {b"content-type", b"content-length"}
                            ]
                            headers.append((b"content-type", b"application/json"))
                            response_start["headers"] = headers
                    elif outcome.rewritten and isinstance(outcome.value, Mapping):
                        body_value = outcome.value.get("body", body)
                        body = (
                            body_value.encode()
                            if isinstance(body_value, str)
                            else bytes(body_value)
                        )
                        status_code = int(outcome.value.get("status_code", status_code))
                        if response_start is not None:
                            response_start["status"] = status_code
'''
    native = replace_once(native, old_response, new_response, label="response rejection")
    native_path.write_text(native, encoding="utf-8")

    test_path = ROOT / "tests/test_policy_hooks.py"
    tests = test_path.read_text(encoding="utf-8")
    tests += '''


def test_repeated_timeouts_use_a_fixed_worker_pool():
    import threading

    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def slow(_value, _context):
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.12)
        finally:
            with state_lock:
                active -= 1
        return HookResult.allow()

    bus = PolicyBus(
        PolicyHookConfig(
            default_timeout_s=0.005,
            maximum_workers=2,
            maximum_pending_tasks=2,
        )
    )
    bus.register("slow", slow, phases=[HookPhase.REQUEST])
    started = time.monotonic()
    for _ in range(12):
        bus.execute(HookPhase.REQUEST, {})
    elapsed = time.monotonic() - started
    bus.shutdown()
    assert elapsed < 0.30
    assert maximum_active <= 2
'''
    test_path.write_text(tests, encoding="utf-8")

    native_test_path = ROOT / "tests/test_native_adaptive.py"
    native_tests = native_test_path.read_text(encoding="utf-8")
    native_tests += '''


def test_middleware_enforces_response_rejection():
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    reset_default_policy_bus_for_tests()
    bus = default_policy_bus()
    bus.register(
        "reject-response",
        lambda _value, _ctx: HookResult.reject("response blocked", status_code=451),
        phases=[HookPhase.RESPONSE],
    )
    app = FastAPI()

    @app.get("/v1/test")
    async def endpoint():
        return {"private": "value"}

    install_native_adaptive_middleware(app)
    with TestClient(app) as client:
        response = client.get("/v1/test")
    assert response.status_code == 451
    assert response.json() == {
        "error": {"type": "policy_rejection", "message": "response blocked"}
    }
    reset_default_policy_bus_for_tests()
'''
    native_test_path.write_text(native_tests, encoding="utf-8")


def active_memory_partitions() -> None:
    path = ROOT / "mtplx/unified_memory.py"
    value = path.read_text(encoding="utf-8")
    value = replace_once(
        value,
        '''    reason: str

    def to_dict(self) -> dict[str, Any]:
''',
        '''    reason: str
    active_partitions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
''',
        label="plan active partitions field",
    )
    value = replace_once(
        value,
        '''            "eligible": self.eligible,
            "reason": self.reason,
''',
        '''            "eligible": self.eligible,
            "reason": self.reason,
            "active_partitions": list(self.active_partitions),
''',
        label="plan active partitions serialization",
    )
    value = replace_once(
        value,
        '''        safe: bool,
        now_s: float | None = None,
    ) -> UnifiedMemoryPlan:
        now = time.monotonic() if now_s is None else float(now_s)
''',
        '''        safe: bool,
        now_s: float | None = None,
        active_partitions: Sequence[str] | None = None,
    ) -> UnifiedMemoryPlan:
        now = time.monotonic() if now_s is None else float(now_s)
        active = set(
            active_partitions
            if active_partitions is not None
            else ("session_bank", "expert_residency", "kv_headroom")
        )
        allowed_partitions = {"session_bank", "expert_residency", "kv_headroom"}
        unknown = active - allowed_partitions
        if unknown:
            raise ValueError(f"unknown memory partitions: {sorted(unknown)}")
''',
        label="plan active partitions parameter",
    )
    old_partition = '''        minima = {
            "session_bank": self.config.minimum_session_bank_bytes,
            "expert": self.config.minimum_expert_bytes,
            "kv": self.config.minimum_kv_headroom_bytes,
        }
        minimum_total = sum(minima.values())
        if available < minimum_total:
            # Preserve KV headroom first, then SessionBank; the expert warm set
            # is the first elastic partition to collapse.
            kv = min(available, minima["kv"])
            remaining = max(0, available - kv)
            session = min(remaining, minima["session_bank"])
            expert = max(0, available - kv - session)
        else:
            remainder = available - minimum_total
            weights = {
                "session_bank": self.config.session_bank_weight,
                "expert": self.config.expert_weight,
                "kv": self.config.kv_headroom_weight,
            }
            total_weight = sum(weights.values())
            session = minima["session_bank"] + int(
                remainder * weights["session_bank"] / total_weight
            )
            expert = minima["expert"] + int(remainder * weights["expert"] / total_weight)
            kv = available - session - expert
'''
    new_partition = '''        minima = {
            "session_bank": (
                self.config.minimum_session_bank_bytes
                if "session_bank" in active
                else 0
            ),
            "expert_residency": (
                self.config.minimum_expert_bytes
                if "expert_residency" in active
                else 0
            ),
            "kv_headroom": (
                self.config.minimum_kv_headroom_bytes
                if "kv_headroom" in active
                else 0
            ),
        }
        weights = {
            "session_bank": (
                self.config.session_bank_weight if "session_bank" in active else 0.0
            ),
            "expert_residency": (
                self.config.expert_weight if "expert_residency" in active else 0.0
            ),
            "kv_headroom": (
                self.config.kv_headroom_weight if "kv_headroom" in active else 0.0
            ),
        }
        minimum_total = sum(minima.values())
        if available < minimum_total:
            # Preserve KV headroom first, then SessionBank; the expert warm set
            # is the first elastic partition to collapse.
            kv = min(available, minima["kv_headroom"])
            remaining = max(0, available - kv)
            session = min(remaining, minima["session_bank"])
            expert = max(0, available - kv - session)
        else:
            remainder = available - minimum_total
            total_weight = sum(weights.values())
            allocations = dict(minima)
            if total_weight > 0:
                distributed = 0
                ordered = ("session_bank", "expert_residency", "kv_headroom")
                active_ordered = [name for name in ordered if weights[name] > 0]
                for name in active_ordered[:-1]:
                    addition = int(remainder * weights[name] / total_weight)
                    allocations[name] += addition
                    distributed += addition
                if active_ordered:
                    allocations[active_ordered[-1]] += remainder - distributed
            session = allocations["session_bank"]
            expert = allocations["expert_residency"]
            kv = allocations["kv_headroom"]
'''
    value = replace_once(value, old_partition, new_partition, label="active partition allocation")
    value = replace_once(
        value,
        '''            eligible=eligible,
            reason=reason,
        )
''',
        '''            eligible=eligible,
            reason=reason,
            active_partitions=tuple(sorted(active)),
        )
''',
        label="plan active partition receipt",
    )
    path.write_text(value, encoding="utf-8")

    native_path = ROOT / "mtplx/native_adaptive.py"
    native = native_path.read_text(encoding="utf-8")
    native = replace_once(
        native,
        '''        plan = coordinator.plan(sample, safe=safe)
        consumers = []
''',
        '''        active_partitions = {"kv_headroom"}
        if bank is not None:
            active_partitions.add("session_bank")
        if controller.config.enabled:
            active_partitions.add("expert_residency")
        plan = coordinator.plan(
            sample,
            safe=safe,
            active_partitions=active_partitions,
        )
        consumers = []
''',
        label="runtime active partition plan",
    )
    native_path.write_text(native, encoding="utf-8")

    test_path = ROOT / "tests/test_unified_memory.py"
    tests = test_path.read_text(encoding="utf-8")
    tests += '''


def test_inactive_expert_partition_is_redistributed():
    coordinator = UnifiedMemoryCoordinator(config())
    all_partitions = coordinator.plan(sample(), safe=True, now_s=10.0)
    without_expert = coordinator.plan(
        sample(),
        safe=True,
        now_s=11.0,
        active_partitions={"session_bank", "kv_headroom"},
    )
    assert without_expert.expert_budget_bytes == 0
    assert "expert_residency" not in without_expert.active_partitions
    assert (
        without_expert.session_bank_budget_bytes
        + without_expert.kv_headroom_bytes
        == without_expert.managed_budget_bytes
    )
    assert without_expert.session_bank_budget_bytes > all_partitions.session_bank_budget_bytes
'''
    test_path.write_text(tests, encoding="utf-8")


def namespaced_runtime_state_and_current_rss() -> None:
    path = ROOT / "mtplx/native_adaptive.py"
    value = path.read_text(encoding="utf-8")
    value = replace_once(
        value,
        '''import resource
import threading
import time
''',
        '''import resource
import subprocess
import threading
import time
''',
        label="current RSS subprocess import",
    )
    old_rss = '''def _process_rss_bytes() -> int:
    try:
        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Darwin reports bytes, Linux KiB.
        if os.uname().sysname == "Darwin":
            return maximum
        return maximum * 1024
    except Exception:
        return 0
'''
    new_rss = '''def _process_rss_bytes() -> int:
    # Prefer current RSS over ru_maxrss, which is a lifetime high-water mark.
    try:
        statm = Path("/proc/self/statm")
        if statm.exists():
            resident_pages = int(statm.read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        pass
    try:
        if os.uname().sysname == "Darwin":
            value = subprocess.check_output(
                ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
                text=True,
                timeout=0.5,
            ).strip()
            if value:
                return int(value) * 1024
    except Exception:
        pass
    try:
        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return maximum if os.uname().sysname == "Darwin" else maximum * 1024
    except Exception:
        return 0
'''
    value = replace_once(value, old_rss, new_rss, label="current RSS sampling")
    value = replace_once(
        value,
        "import time\nfrom dataclasses import replace\n",
        "import time\nfrom dataclasses import replace\nfrom pathlib import Path\n",
        label="Path import",
    )
    replacements = (
        (
            'controller = getattr(state, "expert_residency", None)',
            'controller = getattr(state, "_mtplx_expert_residency_controller", None)',
        ),
        (
            'setattr(state, "expert_residency", controller)',
            'setattr(state, "_mtplx_expert_residency_controller", controller)',
        ),
        (
            'backend = getattr(state, "expert_residency_backend", None)',
            'backend = getattr(state, "_mtplx_expert_residency_backend", None)\n        if backend is None:\n            backend = getattr(state, "expert_residency_backend", None)',
        ),
        (
            'setattr(state, "expert_residency_backend", backend)',
            'setattr(state, "_mtplx_expert_residency_backend", backend)',
        ),
        (
            'coordinator = getattr(state, "unified_memory_coordinator", None)',
            'coordinator = getattr(state, "_mtplx_unified_memory_coordinator", None)',
        ),
        (
            'setattr(state, "unified_memory_coordinator", coordinator)',
            'setattr(state, "_mtplx_unified_memory_coordinator", coordinator)',
        ),
        (
            'exporter = getattr(state, "otlp_exporter", None)',
            'exporter = getattr(state, "_mtplx_otlp_exporter", None)\n        if exporter is None and isinstance(getattr(state, "otlp_exporter", None), OTLPExporter):\n            exporter = state.otlp_exporter',
        ),
        (
            'setattr(state, "otlp_exporter", exporter)',
            'setattr(state, "_mtplx_otlp_exporter", exporter)',
        ),
        (
            'policy_bus = getattr(state, "policy_bus", None)',
            'policy_bus = getattr(state, "_mtplx_policy_bus", None)\n        if policy_bus is None and isinstance(getattr(state, "policy_bus", None), PolicyBus):\n            policy_bus = state.policy_bus',
        ),
        (
            'setattr(state, "policy_bus", policy_bus)',
            'setattr(state, "_mtplx_policy_bus", policy_bus)',
        ),
        (
            'orchestrator = getattr(state, "replay_orchestrator", None)',
            'orchestrator = getattr(state, "_mtplx_replay_orchestrator", None)\n        if orchestrator is None and isinstance(\n            getattr(state, "replay_orchestrator", None), ReplayOrchestrator\n        ):\n            orchestrator = state.replay_orchestrator',
        ),
        (
            'setattr(state, "replay_orchestrator", orchestrator)',
            'setattr(state, "_mtplx_replay_orchestrator", orchestrator)',
        ),
    )
    for old, new in replacements:
        value = replace_once(value, old, new, label=f"namespaced state: {old}")
    path.write_text(value, encoding="utf-8")

    test_path = ROOT / "tests/test_native_adaptive.py"
    tests = test_path.read_text(encoding="utf-8")
    tests += '''


def test_runtime_wiring_does_not_overwrite_generic_application_attributes(monkeypatch):
    monkeypatch.delenv("MTPLX_REQUEST_CAPTURE_DIR", raising=False)
    sentinels = {
        "policy_bus": object(),
        "otlp_exporter": object(),
        "replay_orchestrator": object(),
        "unified_memory_coordinator": object(),
        "expert_residency": object(),
    }
    state = SimpleNamespace(**sentinels)
    systems = ensure_native_adaptive_state(state)
    for name, sentinel in sentinels.items():
        assert getattr(state, name) is sentinel
    assert systems["policy_hooks"] is not sentinels["policy_bus"]
    assert hasattr(state, "_mtplx_policy_bus")
    assert hasattr(state, "_mtplx_unified_memory_coordinator")
'''
    test_path.write_text(tests, encoding="utf-8")


def replay_path_and_timestamp_hardening() -> None:
    path = ROOT / "mtplx/replay_orchestrator.py"
    value = path.read_text(encoding="utf-8")
    value = replace_once(
        value,
        "import time\nfrom dataclasses import asdict, dataclass, field\n",
        "import time\nfrom dataclasses import asdict, dataclass, field\nfrom datetime import datetime\n",
        label="replay datetime import",
    )
    old_float = '''def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
'''
    new_float = '''def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return None
'''
    value = replace_once(value, old_float, new_float, label="ISO timestamp parsing")
    old_files = '''        rows = [
            path
            for path in self.capture_root.rglob("*.json")
            if self.receipt_directory not in path.parents
            and "pruned" not in path.parts
            and path.is_file()
        ]
        rows.sort(key=lambda path: path.as_posix())
        return tuple(rows[: self.config.maximum_scan_files])
'''
    new_files = '''        rows: list[Path] = []
        for path in self.capture_root.rglob("*.json"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(self.capture_root)
            except (OSError, ValueError):
                continue
            if self.receipt_directory in resolved.parents or "pruned" in resolved.parts:
                continue
            rows.append(resolved)
        rows.sort(key=lambda path: path.as_posix())
        return tuple(rows[: self.config.maximum_scan_files])
'''
    value = replace_once(value, old_files, new_files, label="capture path hardening")
    path.write_text(value, encoding="utf-8")

    test_path = ROOT / "tests/test_replay_orchestrator.py"
    tests = test_path.read_text(encoding="utf-8")
    tests += '''


def test_iso_timestamp_filter_and_external_symlink_are_handled_safely(tmp_path):
    write_capture(
        tmp_path,
        "iso",
        {
            "request_id": "iso",
            "created_at": "2026-08-24T00:00:00Z",
            "request": {"payload": {"prompt": "safe"}},
        },
    )
    outside = tmp_path.parent / "outside-capture.json"
    outside.write_text(
        json.dumps(
            {"request_id": "outside", "request": {"payload": {"prompt": "outside"}}}
        ),
        encoding="utf-8",
    )
    link = tmp_path / "outside.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pass
    orchestrator = ReplayOrchestrator(tmp_path)
    plan = orchestrator.build_plan(
        CaptureFilter(created_after_s=1_700_000_000)
    )
    assert [item.capture_id for item in plan.cases] == ["iso"]
'''
    test_path.write_text(tests, encoding="utf-8")


def remove_vendor_branding_from_ui() -> None:
    path = ROOT / "dashboard/src/components/AdaptiveSystemsPanel.tsx"
    value = path.read_text(encoding="utf-8")
    value = replace_once(
        value,
        'subtitle="Independent MTPLX implementations; no FreeToken or Future AGI runtime dependency"',
        'subtitle="Independent MTPLX implementations with no external runtime dependency"',
        label="dashboard vendor branding",
    )
    path.write_text(value, encoding="utf-8")


def main() -> None:
    expert_partial_failure()
    bounded_policy_executor_and_response_rejection()
    active_memory_partitions()
    namespaced_runtime_state_and_current_rss()
    replay_path_and_timestamp_hardening()
    remove_vendor_branding_from_ui()


if __name__ == "__main__":
    main()
