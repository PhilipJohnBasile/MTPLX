"""Runtime wiring for MTPLX's phase-two native adaptive systems.

All capabilities are disabled or inert by default.  Installation is idempotent,
policy/telemetry failures are fail-open unless a trusted hook explicitly asks
for fail-closed behavior, and memory mutations require a non-blocking model
lock plus an independently verified safe point.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from .expert_residency import (
    ExpertResidencyConfig,
    ExpertResidencyController,
    MLXMaterializationBackend,
)
from .otlp_export import OTLPExporter, default_exporter
from .policy_hooks import HookPhase, PolicyBus, PolicyContext, default_policy_bus
from .replay_orchestrator import ReplayOrchestrator, ReplayPlanConfig
from .unified_memory import (
    AttributeBudgetConsumer,
    SessionBankBudgetConsumer,
    UnifiedMemoryConfig,
    UnifiedMemoryCoordinator,
    UnifiedMemorySample,
)

_TRUE = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _total_memory_bytes() -> int:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, ValueError):
        pass
    # Apple Silicon deployments normally provide machine metadata on state;
    # this conservative fallback is used only for standalone tests.
    return 16 * 1024**3


def _process_rss_bytes() -> int:
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


def _scheduler_busy(state: Any) -> bool:
    scheduler = getattr(state, "model_scheduler", None)
    if scheduler is None:
        return False
    try:
        stats = scheduler.stats()
    except Exception:
        return True
    if not isinstance(stats, Mapping):
        return True
    for key in (
        "foreground_pending",
        "idle_pending",
        "persistence_pending",
        "pending",
        "active",
    ):
        value = stats.get(key)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, (int, float)) and value > 0:
            return True
    return bool(stats.get("active_kind"))


def _safe_point(state: Any, *, model_lock_held: bool) -> tuple[bool, str]:
    if not model_lock_held:
        return False, "model_lock_not_held"
    try:
        if int(state.foreground_count()) > 0:
            return False, "foreground_active"
    except Exception:
        return False, "foreground_unknown"
    if _scheduler_busy(state):
        return False, "scheduler_busy"
    for name in (
        "restore_active",
        "commit_active",
        "mtp_transaction_active",
        "postcommit_active",
        "generation_active",
    ):
        value = getattr(state, name, False)
        try:
            if callable(value):
                value = value()
            if bool(value):
                return False, name
        except Exception:
            return False, f"{name}_unknown"
    return True, "safe"


def _machine_total_from_state(state: Any) -> int:
    for source in (
        getattr(state, "machine_info", None),
        getattr(state, "system_info", None),
    ):
        if callable(source):
            try:
                source = source()
            except Exception:
                source = None
        if isinstance(source, Mapping):
            for key in ("unified_memory_bytes", "total_memory_bytes", "memory_bytes"):
                try:
                    value = int(source.get(key) or 0)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    return value
    return _total_memory_bytes()


def _model_from_state(state: Any) -> Any | None:
    candidates = (
        getattr(state, "model", None),
        getattr(getattr(state, "runtime", None), "model", None),
        getattr(getattr(state, "engine", None), "model", None),
    )
    return next((item for item in candidates if item is not None), None)


def _bank_from_state(state: Any) -> Any | None:
    return getattr(getattr(state, "sessions", None), "bank", None)


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


def _locality_snapshot(state: Any) -> Mapping[str, Any] | None:
    candidates = (
        getattr(state, "expert_locality", None),
        getattr(state, "expert_locality_collector", None),
        getattr(getattr(state, "runtime", None), "expert_locality", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        snapshot = getattr(candidate, "snapshot", None)
        try:
            value = snapshot() if callable(snapshot) else candidate
        except Exception:
            continue
        if hasattr(value, "to_dict"):
            try:
                value = value.to_dict()
            except Exception:
                continue
        if isinstance(value, Mapping):
            return value
    return None


def _expert_config_from_env() -> ExpertResidencyConfig:
    return ExpertResidencyConfig(
        enabled=_env_bool("MTPLX_EXPERT_RESIDENCY", False),
        budget_bytes=max(0, _env_int("MTPLX_EXPERT_RESIDENCY_BYTES", 0)),
        minimum_observations=max(
            1, _env_int("MTPLX_EXPERT_RESIDENCY_MIN_OBSERVATIONS", 8)
        ),
        half_life_s=max(0.001, _env_float("MTPLX_EXPERT_RESIDENCY_HALF_LIFE_S", 45.0)),
        maximum_tracked_experts=max(
            1, _env_int("MTPLX_EXPERT_RESIDENCY_MAX_TRACKED", 4096)
        ),
        maximum_prefetch_per_tick=max(
            0, _env_int("MTPLX_EXPERT_RESIDENCY_PREFETCH_PER_TICK", 8)
        ),
        maximum_evict_per_tick=max(
            0, _env_int("MTPLX_EXPERT_RESIDENCY_EVICT_PER_TICK", 8)
        ),
    )


def _unified_config_from_env() -> UnifiedMemoryConfig:
    gib = 1024**3
    target = min(
        0.99,
        max(0.20, _env_float("MTPLX_UNIFIED_MEMORY_TARGET", 0.88)),
    )
    warning = min(
        1.0,
        max(target, _env_float("MTPLX_UNIFIED_MEMORY_WARNING", 0.92)),
    )
    critical = min(
        1.0,
        max(warning, _env_float("MTPLX_UNIFIED_MEMORY_CRITICAL", 0.96)),
    )
    return UnifiedMemoryConfig(
        enabled=_env_bool("MTPLX_UNIFIED_MEMORY", False),
        reserve_bytes=max(0, _env_int("MTPLX_UNIFIED_MEMORY_RESERVE_BYTES", 4 * gib)),
        target_utilization=target,
        warning_utilization=warning,
        critical_utilization=critical,
        minimum_session_bank_bytes=max(
            0, _env_int("MTPLX_UNIFIED_MEMORY_MIN_SESSION_BYTES", 256 * 1024**2)
        ),
        minimum_expert_bytes=max(
            0, _env_int("MTPLX_UNIFIED_MEMORY_MIN_EXPERT_BYTES", 0)
        ),
        minimum_kv_headroom_bytes=max(
            0, _env_int("MTPLX_UNIFIED_MEMORY_MIN_KV_BYTES", 256 * 1024**2)
        ),
    )


def ensure_native_adaptive_state(state: Any) -> dict[str, Any]:
    """Lazily attach native phase-two systems to one server state object."""

    lock = getattr(state, "_mtplx_native_adaptive_lock", None)
    if lock is None:
        lock = threading.RLock()
        try:
            setattr(state, "_mtplx_native_adaptive_lock", lock)
        except Exception:
            pass
    with lock:
        controller = getattr(state, "_mtplx_expert_residency_controller", None)
        if not isinstance(controller, ExpertResidencyController):
            controller = ExpertResidencyController(_expert_config_from_env())
            try:
                setattr(state, "_mtplx_expert_residency_controller", controller)
            except Exception:
                pass
        backend = getattr(state, "_mtplx_expert_residency_backend", None)
        if backend is None:
            backend = getattr(state, "expert_residency_backend", None)
        if backend is None and controller.config.enabled:
            model = _model_from_state(state)
            if model is not None:
                backend = MLXMaterializationBackend(model)
                try:
                    setattr(state, "_mtplx_expert_residency_backend", backend)
                except Exception:
                    pass
        coordinator = getattr(state, "_mtplx_unified_memory_coordinator", None)
        if not isinstance(coordinator, UnifiedMemoryCoordinator):
            coordinator = UnifiedMemoryCoordinator(_unified_config_from_env())
            try:
                setattr(state, "_mtplx_unified_memory_coordinator", coordinator)
            except Exception:
                pass
        exporter = getattr(state, "_mtplx_otlp_exporter", None)
        if exporter is None and isinstance(
            getattr(state, "otlp_exporter", None), OTLPExporter
        ):
            exporter = state.otlp_exporter
        if not isinstance(exporter, OTLPExporter):
            exporter = default_exporter()
            try:
                setattr(state, "_mtplx_otlp_exporter", exporter)
            except Exception:
                pass
        policy_bus = getattr(state, "_mtplx_policy_bus", None)
        if policy_bus is None and isinstance(
            getattr(state, "policy_bus", None), PolicyBus
        ):
            policy_bus = state.policy_bus
        if not isinstance(policy_bus, PolicyBus):
            policy_bus = default_policy_bus()
            try:
                setattr(state, "_mtplx_policy_bus", policy_bus)
            except Exception:
                pass
        orchestrator = getattr(state, "_mtplx_replay_orchestrator", None)
        if orchestrator is None and isinstance(
            getattr(state, "replay_orchestrator", None), ReplayOrchestrator
        ):
            orchestrator = state.replay_orchestrator
        capture_root = os.environ.get("MTPLX_REQUEST_CAPTURE_DIR")
        if orchestrator is None and capture_root:
            orchestrator = ReplayOrchestrator(
                capture_root,
                config=ReplayPlanConfig(
                    maximum_cases=max(1, _env_int("MTPLX_REPLAY_MAX_CASES", 128))
                ),
            )
            try:
                setattr(state, "_mtplx_replay_orchestrator", orchestrator)
            except Exception:
                pass
        return {
            "expert_residency": controller,
            "expert_backend": backend,
            "unified_memory": coordinator,
            "otlp": exporter,
            "policy_hooks": policy_bus,
            "replay_orchestrator": orchestrator,
        }


def native_adaptive_tick(state: Any) -> dict[str, Any]:
    """Run one non-blocking, lock-atomic adaptive-memory tick."""

    systems = ensure_native_adaptive_state(state)
    controller: ExpertResidencyController = systems["expert_residency"]
    coordinator: UnifiedMemoryCoordinator = systems["unified_memory"]
    backend = systems["expert_backend"]
    if not controller.config.enabled and not coordinator.config.enabled:
        return {
            "safe": True,
            "reason": "disabled",
            "unified_memory": coordinator.snapshot(),
            "expert_residency": controller.snapshot(backend),
        }
    model_lock = getattr(state, "lock", None)
    acquired = False
    if model_lock is not None:
        try:
            acquired = bool(model_lock.acquire(blocking=False))
        except Exception:
            acquired = False
    if not acquired:
        return {
            "safe": False,
            "reason": "model_lock_busy_or_unavailable",
            "unified_memory": coordinator.snapshot(),
            "expert_residency": controller.snapshot(backend),
        }
    try:
        safe, reason = _safe_point(state, model_lock_held=True)
        locality = _locality_snapshot(state)
        if locality is not None:
            try:
                controller.ingest_locality_snapshot(locality)
            except Exception:
                pass
        bank = _bank_from_state(state)
        total = _machine_total_from_state(state)
        rss = _process_rss_bytes()
        session_bytes = int(getattr(bank, "total_nbytes", 0) or 0) if bank else 0
        model_bytes = int(getattr(state, "model_weights_bytes", 0) or 0)
        resident_bytes = 0
        if backend is not None:
            try:
                resident_bytes = sum(
                    max(0, int(backend.expert_nbytes(item)))
                    for item in backend.resident_experts()
                )
            except Exception:
                resident_bytes = 0
        sample = UnifiedMemorySample(
            total_bytes=max(1, total),
            process_bytes=max(0, rss),
            model_bytes=max(0, model_bytes),
            session_bank_bytes=max(0, session_bytes),
            expert_bytes=max(0, resident_bytes),
            kv_bytes=max(0, int(getattr(state, "kv_cache_bytes", 0) or 0)),
            timestamp_s=time.time(),
        )
        active_partitions = {"kv_headroom"}
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
        if bank is not None:
            consumers.append(SessionBankBudgetConsumer(bank))
        kv_consumer = _kv_headroom_consumer(state)
        if kv_consumer is not None:
            consumers.append(kv_consumer)
        consumers.append(
            AttributeBudgetConsumer(
                controller,
                name="expert_residency",
                attribute="budget_bytes",
            )
        )
        memory_receipt = coordinator.apply(plan, consumers, safe=safe)
        expert_budget = (
            plan.expert_budget_bytes
            if plan.eligible
            else controller.config.budget_bytes
        )
        expert_plan = controller.plan(backend, budget_bytes=expert_budget)
        expert_receipt = controller.apply(expert_plan, backend, safe=safe)
        return {
            "safe": safe,
            "reason": reason,
            "sample": {
                "total_bytes": sample.total_bytes,
                "process_bytes": sample.process_bytes,
                "session_bank_bytes": sample.session_bank_bytes,
                "expert_bytes": sample.expert_bytes,
                "kv_bytes": sample.kv_bytes,
            },
            "unified_memory_plan": plan.to_dict(),
            "unified_memory_apply": memory_receipt.to_dict(),
            "expert_residency_plan": expert_plan.to_dict(),
            "expert_residency_apply": expert_receipt.to_dict(),
        }
    finally:
        try:
            model_lock.release()
        except Exception:
            pass


def augment_systems_snapshot(
    payload: Mapping[str, Any] | None, state: Any
) -> dict[str, Any]:
    result = dict(payload or {})
    try:
        systems = ensure_native_adaptive_state(state)
        result["expert_residency"] = systems["expert_residency"].snapshot(
            systems["expert_backend"]
        )
        result["unified_memory"] = systems["unified_memory"].snapshot()
        result["otlp_export"] = systems["otlp"].snapshot()
        result["policy_hooks"] = systems["policy_hooks"].snapshot()
        orchestrator = systems["replay_orchestrator"]
        result["replay_orchestration"] = (
            orchestrator.snapshot()
            if orchestrator is not None
            else {
                "available": True,
                "enabled": False,
                "capture_root_configured": False,
                "promotion_is_automatic": False,
            }
        )
    except Exception as exc:
        result["native_adaptive_error"] = type(exc).__name__
    return result


def _runtime_state_from_app(app: Any) -> Any | None:
    app_state = getattr(app, "state", None)
    if app_state is None:
        return None
    for name in (
        "mtplx_state",
        "server_state",
        "runtime_state",
        "model_state",
        "runtime",
    ):
        value = getattr(app_state, name, None)
        if value is not None:
            return value
    # Some MTPLX versions attach the model/runtime fields directly to app.state.
    if any(hasattr(app_state, name) for name in ("sessions", "model", "lock")):
        return app_state
    return None


class NativeAdaptiveMiddleware:
    """ASGI policy and OTLP integration with no external framework dependency."""

    def __init__(
        self,
        app: Callable[..., Awaitable[Any]],
        *,
        runtime_state_provider: Callable[[], Any | None] | None = None,
    ) -> None:
        self.app = app
        self.runtime_state_provider = runtime_state_provider

    def _state(self, scope: Mapping[str, Any]) -> Any | None:
        if self.runtime_state_provider is not None:
            try:
                value = self.runtime_state_provider()
                if value is not None:
                    return value
            except Exception:
                pass
        return _runtime_state_from_app(scope.get("app"))

    @staticmethod
    async def _read_body(
        receive: Callable[[], Awaitable[dict[str, Any]]], limit: int
    ) -> tuple[bytes, list[dict[str, Any]]]:
        chunks: list[bytes] = []
        messages: list[dict[str, Any]] = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") != "http.request":
                break
            body = bytes(message.get("body", b""))
            total += len(body)
            if total > limit:
                raise ValueError("request body exceeds policy limit")
            chunks.append(body)
            if not message.get("more_body", False):
                break
        return b"".join(chunks), messages

    @staticmethod
    def _receive_replay(
        messages: list[dict[str, Any]],
    ) -> Callable[[], Awaitable[dict[str, Any]]]:
        pending = list(messages)

        async def receive() -> dict[str, Any]:
            if pending:
                return pending.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        return receive

    @staticmethod
    async def _reject(
        send: Callable[[dict[str, Any]], Awaitable[None]], status: int, reason: str
    ) -> None:
        body = json.dumps(
            {"error": {"type": "policy_rejection", "message": reason}},
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": int(status),
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        state = self._state(scope)
        systems = (
            ensure_native_adaptive_state(state)
            if state is not None
            else {
                "policy_hooks": default_policy_bus(),
                "otlp": default_exporter(),
            }
        )
        bus: PolicyBus = systems["policy_hooks"]
        exporter: OTLPExporter = systems["otlp"]
        policy_snapshot = bus.snapshot()
        if not exporter.config.enabled and not policy_snapshot.get("enabled", False):
            await self.app(scope, receive, send)
            return
        hook_counts = policy_snapshot.get("hooks_by_phase", {})
        method = str(scope.get("method", ""))
        route = str(scope.get("path", ""))
        request_id = None
        for key, value in scope.get("headers", []):
            if key.lower() in {b"x-request-id", b"x-mtplx-request-id"}:
                request_id = value.decode("utf-8", errors="replace")[:128]
                break
        context = PolicyContext(
            phase=HookPhase.REQUEST,
            request_id=request_id,
            route=route,
            metadata={"method": method},
        )
        span_cm = exporter.span(
            "mtplx.http.request",
            attributes={"http_method": method, "http_route": route},
        )
        with span_cm as span_attrs:
            replay_receive = receive
            if int(hook_counts.get(HookPhase.REQUEST.value, 0) or 0) > 0:
                try:
                    body, messages = await self._read_body(
                        receive,
                        bus.config.maximum_value_bytes,
                    )
                except Exception as exc:
                    await self._reject(send, 413, type(exc).__name__)
                    span_attrs["http_status_code"] = 413
                    return
                content_type = b""
                for key, value in scope.get("headers", []):
                    if key.lower() == b"content-type":
                        content_type = value.lower()
                        break
                parsed: Any = body
                if b"json" in content_type and body:
                    try:
                        parsed = json.loads(body)
                    except json.JSONDecodeError:
                        parsed = body
                outcome = bus.execute(HookPhase.REQUEST, parsed, context=context)
                if not outcome.allowed:
                    await self._reject(send, outcome.status_code, outcome.reason)
                    span_attrs["http_status_code"] = outcome.status_code
                    span_attrs["policy_rejected"] = True
                    return
                if outcome.rewritten:
                    if b"json" in content_type:
                        rewritten = json.dumps(
                            outcome.value, separators=(",", ":")
                        ).encode()
                    elif isinstance(outcome.value, str):
                        rewritten = outcome.value.encode()
                    else:
                        rewritten = bytes(outcome.value)
                    messages = [
                        {"type": "http.request", "body": rewritten, "more_body": False}
                    ]
                replay_receive = self._receive_replay(messages)
                span_attrs["policy_request_hooks"] = len(outcome.executed_hooks)

            response_start: dict[str, Any] | None = None
            streaming = False
            status_code = 500

            async def wrapped_send(message: dict[str, Any]) -> None:
                nonlocal response_start, streaming, status_code
                if message.get("type") == "http.response.start":
                    response_start = dict(message)
                    status_code = int(message.get("status", 200))
                    return
                if message.get("type") != "http.response.body":
                    if response_start is not None:
                        await send(response_start)
                        response_start = None
                    await send(message)
                    return
                more = bool(message.get("more_body", False))
                body = bytes(message.get("body", b""))
                original_body = body
                if more:
                    streaming = True
                if int(hook_counts.get(HookPhase.STREAM_EVENT.value, 0) or 0) > 0:
                    stream_context = replace(context, phase=HookPhase.STREAM_EVENT)
                    outcome = bus.execute(
                        HookPhase.STREAM_EVENT,
                        body,
                        context=stream_context,
                    )
                    if not outcome.allowed:
                        body = b""
                    elif outcome.rewritten:
                        value = outcome.value
                        body = (
                            value.encode() if isinstance(value, str) else bytes(value)
                        )
                if (
                    not streaming
                    and not more
                    and int(hook_counts.get(HookPhase.RESPONSE.value, 0) or 0) > 0
                ):
                    response_context = replace(context, phase=HookPhase.RESPONSE)
                    outcome = bus.execute(
                        HookPhase.RESPONSE,
                        {"status_code": status_code, "body": body},
                        context=response_context,
                    )
                    if not outcome.allowed:
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
                                if key.lower()
                                not in {b"content-type", b"content-length"}
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
                if response_start is not None:
                    # MTPLX_NATIVE_ADAPTIVE_CONTENT_LENGTH
                    if not streaming and not more and body != original_body:
                        headers = [
                            (key, value)
                            for key, value in response_start.get("headers", [])
                            if key.lower() != b"content-length"
                        ]
                        headers.append((b"content-length", str(len(body)).encode()))
                        response_start["headers"] = headers
                    await send(response_start)
                    response_start = None
                await send({**message, "body": body})

            try:
                await self.app(scope, replay_receive, wrapped_send)
            except Exception as exc:
                if int(hook_counts.get(HookPhase.ERROR.value, 0) or 0) > 0:
                    error_context = replace(context, phase=HookPhase.ERROR)
                    try:
                        bus.execute(
                            HookPhase.ERROR,
                            {"error_type": type(exc).__name__},
                            context=error_context,
                        )
                    except Exception:
                        pass
                span_attrs["error_type"] = type(exc).__name__
                raise
            finally:
                span_attrs["http_status_code"] = status_code


def install_native_adaptive_middleware(
    app: Any,
    *,
    runtime_state_provider: Callable[[], Any | None] | None = None,
) -> bool:
    """Install middleware once; return whether installation occurred."""

    marker = "_mtplx_native_adaptive_middleware_installed"
    if getattr(app, marker, False):
        return False
    add_middleware = getattr(app, "add_middleware", None)
    if not callable(add_middleware):
        return False
    add_middleware(
        NativeAdaptiveMiddleware,
        runtime_state_provider=runtime_state_provider,
    )
    setattr(app, marker, True)
    return True
