"""Runtime status registry for semantic memory, locality, memory, and replay.

The registry is deliberately observational. It holds the latest request-local
semantic plan, startup instrumentation report, and memory-governor state so the
HTTP/dashboard surface can tell the truth about what is available, enabled,
wired, and actually sampled.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Mapping

from .expert_locality import expert_locality_enabled, expert_locality_metrics
from .request_capture import capture_dir
from .semantic_anchors import SemanticAnchorPlan


def env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"", "0", "false", "off", "no"}


def env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError, OverflowError):
        value = int(default)
    return max(int(minimum), value)


def semantic_anchors_enabled() -> bool:
    return env_truthy("MTPLX_SEMANTIC_ANCHORS", False)


def memory_governor_enabled() -> bool:
    return env_truthy("MTPLX_MEMORY_GOVERNOR", False)


class RuntimeSystemsRegistry:
    """Thread-safe state shared by serving and the dashboard endpoint."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._semantic_latest: dict[str, Any] | None = None
        self._semantic_requests = 0
        self._semantic_skips = 0
        self._expert_install_report: dict[str, Any] = {
            "enabled": expert_locality_enabled(),
            "installed": False,
            "instrumented_modules": 0,
            "reason": "startup_not_run",
        }
        self._memory_governor: Any | None = None
        self._capture_dispatched = 0
        self._capture_completed = 0
        self._capture_failures = 0
        self._capture_latest: dict[str, Any] | None = None
        self._updated_at_s = time.time()

    def record_semantic_plan(
        self,
        plan: SemanticAnchorPlan,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
        candidate_count: int | None = None,
    ) -> None:
        payload = {
            **plan.to_metrics(),
            "anchors": [anchor.to_dict() for anchor in plan.anchors],
            "rejected": [item.to_dict() for item in plan.rejected],
            "request_id": request_id,
            "session_id": session_id,
            "candidate_count": (
                int(candidate_count) if candidate_count is not None else None
            ),
            "status": "planned",
            "updated_at_s": time.time(),
        }
        with self._lock:
            self._semantic_latest = payload
            self._semantic_requests += 1
            self._updated_at_s = payload["updated_at_s"]

    def record_semantic_skip(
        self,
        reason: str,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        payload = {
            "status": "skipped",
            "reason": str(reason),
            "request_id": request_id,
            "session_id": session_id,
            "updated_at_s": time.time(),
        }
        with self._lock:
            self._semantic_latest = payload
            self._semantic_skips += 1
            self._updated_at_s = payload["updated_at_s"]

    def set_expert_install_report(self, report: Mapping[str, Any] | None) -> None:
        with self._lock:
            self._expert_install_report = dict(report or {})
            self._updated_at_s = time.time()

    def set_memory_governor(self, governor: Any | None) -> None:
        with self._lock:
            self._memory_governor = governor
            self._updated_at_s = time.time()

    def record_request_capture(
        self,
        phase: str,
        *,
        request_id: str | None,
        session_id: str | None = None,
        persisted: bool,
    ) -> None:
        now = time.time()
        payload = {
            "phase": str(phase),
            "request_id": request_id,
            "session_id": session_id,
            "persisted": bool(persisted),
            "updated_at_s": now,
        }
        with self._lock:
            if phase == "dispatched":
                self._capture_dispatched += 1
            elif phase == "completed":
                self._capture_completed += 1
            if not persisted:
                self._capture_failures += 1
            self._capture_latest = payload
            self._updated_at_s = now

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            semantic_latest = (
                dict(self._semantic_latest) if self._semantic_latest else None
            )
            semantic_requests = int(self._semantic_requests)
            semantic_skips = int(self._semantic_skips)
            expert_install = dict(self._expert_install_report)
            governor = self._memory_governor
            capture_dispatched = int(self._capture_dispatched)
            capture_completed = int(self._capture_completed)
            capture_failures = int(self._capture_failures)
            capture_latest = (
                dict(self._capture_latest) if self._capture_latest else None
            )
            updated_at_s = float(self._updated_at_s)

        try:
            expert_metrics = expert_locality_metrics()
        except Exception as exc:
            expert_metrics = {"enabled": expert_locality_enabled(), "error": str(exc)}

        memory_metrics: dict[str, Any] = {}
        memory_config: dict[str, Any] = {}
        if governor is not None:
            try:
                memory_metrics = dict(governor.to_metrics())
                config = getattr(governor, "config", None)
                if config is not None:
                    from dataclasses import asdict

                    memory_config = asdict(config)
            except Exception as exc:
                memory_metrics = {"error": str(exc)}

        semantic_enabled = semantic_anchors_enabled()
        expert_enabled = expert_locality_enabled()
        governor_enabled = memory_governor_enabled()
        return {
            "ts": time.time(),
            "updated_at_s": updated_at_s,
            "semantic_memory": {
                "available": True,
                "enabled": semantic_enabled,
                "wired": True,
                "mode": "exact_message_prefix_checkpoints",
                "config": {
                    "max_anchors": env_int(
                        "MTPLX_SEMANTIC_ANCHOR_MAX", 8, minimum=1
                    ),
                    "candidate_limit": env_int(
                        "MTPLX_SEMANTIC_ANCHOR_CANDIDATES", 16, minimum=1
                    ),
                    "max_checkpoint_bytes": env_int(
                        "MTPLX_SEMANTIC_ANCHOR_MAX_BYTES", 0, minimum=0
                    ),
                    "estimated_checkpoint_bytes": env_int(
                        "MTPLX_SEMANTIC_ANCHOR_ESTIMATED_BYTES", 0, minimum=0
                    ),
                },
                "planned_requests": semantic_requests,
                "skipped_requests": semantic_skips,
                "latest": semantic_latest,
            },
            "expert_locality": {
                "available": True,
                "enabled": expert_enabled,
                "wired": bool(expert_install.get("installed")),
                "install": expert_install,
                "metrics": expert_metrics,
            },
            "memory_governor": {
                "available": True,
                "enabled": governor_enabled,
                "wired": governor is not None,
                "config": memory_config,
                "metrics": memory_metrics,
            },
            "deterministic_replay": {
                "available": True,
                "mode": "offline_verifier",
                "runtime_mutation": False,
                "request_capture": {
                    "available": True,
                    "enabled": bool(capture_dir()),
                    "wired": True,
                    "content_capture_default": False,
                    "dispatched": capture_dispatched,
                    "completed": capture_completed,
                    "failures": capture_failures,
                    "latest": capture_latest,
                },
                "replay": {
                    "available": True,
                    "wired": False,
                    "mode": "explicit_offline_call",
                    "promotion_is_automatic": False,
                },
                "trace_parity": {
                    "available": True,
                    "wired": False,
                    "mode": "offline_compare",
                },
            },
        }


__all__ = [
    "RuntimeSystemsRegistry",
    "env_int",
    "env_truthy",
    "memory_governor_enabled",
    "semantic_anchors_enabled",
]
