#!/usr/bin/env python3
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(value: str, old: str, new: str, *, label: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return value.replace(old, new, 1)


def load_v8():
    path = ROOT / "scripts/materialize_native_adaptive_phase2_fixups_v8.py"
    spec = spec_from_file_location("mtplx_phase2_v8", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load v8 hardening module")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
            self._queue.put_nowait(self._STOP)
'''
    value = replace_once(value, anchor, executor + anchor, label="bounded policy executor")
    value = replace_once(
        value,
        '''        self._hooks: dict[str, _Registration] = {}
        self._lock = threading.RLock()
''',
        '''        self._hooks: dict[str, _Registration] = {}
        self._lock = threading.RLock()
        self._executor: _BoundedHookExecutor | None = None
''',
        label="lazy policy executor initialization",
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
    new_run = '''    def _executor_for_use(self) -> _BoundedHookExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = _BoundedHookExecutor(
                    workers=min(
                        self.config.maximum_workers,
                        self.config.maximum_hooks,
                    ),
                    pending=self.config.maximum_pending_tasks,
                )
            return self._executor

    def _run_one(
        self,
        registration: _Registration,
        value: Any,
        context: PolicyContext,
    ) -> HookResult:
        isolated_value = deepcopy(value)
        isolated_context = deepcopy(context)
        future = self._executor_for_use().submit(
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
        with self._lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown()

    def snapshot(self) -> dict[str, Any]:
''',
        label="policy shutdown method",
    )
    value = replace_once(
        value,
        '''                "external_policy_dependency": False,
''',
        '''                "external_policy_dependency": False,
                "executor_started": self._executor is not None,
''',
        label="policy executor snapshot",
    )
    value = replace_once(
        value,
        '''    with _DEFAULT_LOCK:
        _DEFAULT_BUS = None
''',
        '''    with _DEFAULT_LOCK:
        if _DEFAULT_BUS is not None:
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


def test_policy_executor_is_lazy_and_worker_count_is_bounded():
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
    assert bus.snapshot()["executor_started"] is False
    bus.register("slow", slow, phases=[HookPhase.REQUEST])
    assert bus.snapshot()["executor_started"] is False
    started = time.monotonic()
    for _ in range(12):
        bus.execute(HookPhase.REQUEST, {})
    elapsed = time.monotonic() - started
    assert bus.snapshot()["executor_started"] is True
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


def main() -> None:
    v8 = load_v8()
    v8.expert_partial_failure()
    bounded_policy_executor_and_response_rejection()
    v8.active_memory_partitions()
    v8.namespaced_runtime_state_and_current_rss()
    v8.replay_path_and_timestamp_hardening()
    v8.remove_vendor_branding_from_ui()


if __name__ == "__main__":
    main()
