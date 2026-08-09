"""Fixed-width Qwen 35B MTP cohort service.

Request threads enqueue independent jobs.  One existing model-owner thread seals
an immutable cohort, runs the preinstalled B8/T2 lane, and closes each future.
No request is admitted into an active cohort and no AR route exists here.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Condition, Event
from typing import Any

from mtplx.a3b_mtp_batch import (
    A3BMTPBatchRequest,
    A3BMTPBatchResult,
    generate_a3b_mtp_batch,
)
from mtplx.sampling import SamplerConfig


@dataclass
class MTPBatchJob:
    request_id: str
    prompt_ids: list[int]
    max_tokens: int
    sampler: SamplerConfig
    draft_sampler: SamplerConfig
    seed: int
    stop_token_ids: set[int]
    token_callback: Callable[[list[int]], None] | None
    compatibility_key: Hashable
    generation_limits: dict[str, Any]
    solo_runner: Callable[["MTPBatchJob"], dict[str, Any]] | None
    cancel_error: Callable[["MTPBatchJob"], BaseException]
    cancel_event: Event = field(default_factory=Event)
    prefill_callback: Callable[[dict[str, Any]], None] | None = None
    request_observability: dict[str, Any] = field(default_factory=dict)
    omit_speculative_bonus: bool = False
    future: Future = field(default_factory=Future, init=False)
    tokens: list[int] = field(default_factory=list, init=False)
    token_times: list[float] = field(default_factory=list, init=False)
    callback_error: BaseException | None = field(default=None, init=False)
    decode_started_s: float | None = field(default=None, init=False)
    created_s: float = field(default_factory=time.perf_counter, init=False)
    admitted_s: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.prompt_ids = [int(token) for token in self.prompt_ids]
        self.max_tokens = max(1, int(self.max_tokens))
        self.stop_token_ids = {int(token) for token in self.stop_token_ids}
        self.request_observability = dict(self.request_observability)
        self.generation_limits = dict(self.generation_limits)

    def cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    def emit_token(self, token: int) -> None:
        if self.cancel_requested():
            return
        value = int(token)
        self.tokens.append(value)
        if value not in self.stop_token_ids and self.token_callback is not None:
            try:
                self.token_callback([value])
            except Exception as exc:
                self.callback_error = exc
                self.cancel_event.set()
                if not self.future.done():
                    self.future.set_exception(exc)
        self.token_times.append(time.perf_counter())

    def emit_prefill(self, payload: dict[str, Any]) -> None:
        if self.prefill_callback is None:
            return
        try:
            self.prefill_callback(dict(payload))
        except Exception:
            pass

    def mark_decode_started(self) -> None:
        if self.decode_started_s is None:
            self.decode_started_s = time.perf_counter()

    def close_cancelled(self) -> None:
        self.cancel_event.set()
        if not self.future.done():
            self.future.set_exception(self.cancel_error(self))


class MTPBatchGenerationService:
    """Seal and execute independent fixed-width MTP cohorts."""

    def __init__(
        self,
        state: Any,
        *,
        lane: Any,
        driver: Callable[[Any, list[A3BMTPBatchRequest]], A3BMTPBatchResult] = (
            generate_a3b_mtp_batch
        ),
        batch_wait_s: float = 0.02,
        auto_schedule: bool = True,
    ) -> None:
        self.state = state
        self.lane = lane
        self.driver = driver
        self.batch_wait_s = max(0.0, float(batch_wait_s))
        self.auto_schedule = bool(auto_schedule)
        self._condition = Condition()
        self._pending: list[MTPBatchJob] = []
        self._active: list[MTPBatchJob] = []
        self._pump_scheduled = False
        self._shutdown = False
        self._last_error: str | None = None
        self._last_real_width = 0
        self._last_route_id: str | None = None
        self._batch_histogram: Counter[int] = Counter()
        self._fixed_width_histogram: Counter[int] = Counter()
        self._target_verify_cycles = 0
        self._accepted_drafts = 0
        self._rejected_drafts = 0
        self._solo_runs = 0

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "pending": len(self._pending),
                "active": len(self._active),
                "pump_scheduled": self._pump_scheduled,
                "last_real_width": self._last_real_width,
                "last_route_id": self._last_route_id,
                "last_error": self._last_error,
                "batch_histogram": {
                    str(width): count
                    for width, count in sorted(self._batch_histogram.items())
                },
                "fixed_width_histogram": {
                    str(width): count
                    for width, count in sorted(self._fixed_width_histogram.items())
                },
                "target_verify_cycles": self._target_verify_cycles,
                "accepted_draft_tokens": self._accepted_drafts,
                "rejected_draft_tokens": self._rejected_drafts,
                "solo_runs": self._solo_runs,
            }

    def submit(self, job: MTPBatchJob) -> Future:
        schedule = False
        with self._condition:
            if self._shutdown:
                job.future.set_exception(RuntimeError("MTP batch service is shut down"))
                return job.future
            self._pending.append(job)
            if self.auto_schedule and not self._pump_scheduled:
                self._pump_scheduled = True
                schedule = True
            self._condition.notify_all()
        if schedule:
            self._schedule_pump()
        return job.future

    def _schedule_pump(self) -> None:
        scheduler = getattr(self.state, "model_scheduler", None)
        if scheduler is None or not hasattr(scheduler, "submit_foreground"):
            exc = RuntimeError("MTP batch service requires the model-owner scheduler")
            self._fail_pending(exc)
            return
        scheduler.submit_foreground(self._pump, batch_key="mtp_batch.pump")

    def _cancelled_exception(self, job: MTPBatchJob) -> BaseException:
        return job.cancel_error(job)

    def _drain_cancelled_locked(self) -> None:
        keep: list[MTPBatchJob] = []
        for job in self._pending:
            if job.cancel_requested() or job.future.cancelled():
                if not job.future.done():
                    job.future.set_exception(self._cancelled_exception(job))
            else:
                keep.append(job)
        self._pending = keep

    def _compatible_pending_locked(self, key: Hashable) -> list[MTPBatchJob]:
        return [
            job
            for job in self._pending
            if job.compatibility_key == key and not job.cancel_requested()
        ][:8]

    def _seal(self, *, wait: bool) -> list[MTPBatchJob]:
        with self._condition:
            self._drain_cancelled_locked()
            if not self._pending:
                return []
            key = self._pending[0].compatibility_key
            deadline = time.perf_counter() + (self.batch_wait_s if wait else 0.0)
            selected = self._compatible_pending_locked(key)
            while wait and len(selected) < 8:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
                self._drain_cancelled_locked()
                if not self._pending:
                    return []
                selected = self._compatible_pending_locked(key)
            selected_ids = {id(job) for job in selected}
            self._pending = [
                job for job in self._pending if id(job) not in selected_ids
            ]
            now = time.perf_counter()
            for job in selected:
                job.admitted_s = now
            self._active = list(selected)
            self._last_real_width = len(selected)
            self._batch_histogram[len(selected)] += 1
            return selected

    def pump_once(self) -> bool:
        jobs = self._seal(wait=False)
        if not jobs:
            return False
        self._run_sealed(jobs)
        return True

    def _pump(self) -> None:
        try:
            while True:
                jobs = self._seal(wait=True)
                if not jobs:
                    return
                self._run_sealed(jobs)
        finally:
            schedule = False
            with self._condition:
                self._pump_scheduled = False
                if self._pending and not self._shutdown:
                    self._pump_scheduled = True
                    schedule = True
                self._condition.notify_all()
            if schedule:
                self._schedule_pump()

    def _run_sealed(self, jobs: list[MTPBatchJob]) -> None:
        try:
            if len(jobs) == 1:
                self._run_solo(jobs[0])
            else:
                self._run_cohort(jobs)
        except BaseException as exc:
            with self._condition:
                self._last_error = f"{type(exc).__name__}: {exc}"
            for job in jobs:
                if not job.future.done():
                    job.future.set_exception(exc)
        finally:
            with self._condition:
                self._active = []
                self._condition.notify_all()

    def _run_solo(self, job: MTPBatchJob) -> None:
        if job.cancel_requested():
            raise self._cancelled_exception(job)
        if job.solo_runner is None:
            raise RuntimeError("MTP batch solo request has no solo MTP runner")
        with self._condition:
            self._solo_runs += 1
        result = dict(job.solo_runner(job))
        result["_mtp_batch_solo"] = True
        if not job.future.done():
            job.future.set_result(result)

    def _run_cohort(self, jobs: list[MTPBatchJob]) -> None:
        started = time.perf_counter()
        real_width = len(jobs)
        for job in jobs:
            job.emit_prefill(
                {
                    "phase": "started",
                    "tokens_total": len(job.prompt_ids),
                    "scheduler_lane": "mtp_batch",
                    "request_id": job.request_id,
                }
            )
        requests = [
            A3BMTPBatchRequest(
                request_id=str(row),
                prompt_ids=tuple(job.prompt_ids),
                sampler=job.sampler,
                draft_sampler=job.draft_sampler,
                seed=job.seed,
                max_tokens=job.max_tokens,
                stop_token_ids=frozenset(job.stop_token_ids),
                omit_speculative_bonus=job.omit_speculative_bonus,
                on_token=job.emit_token,
                on_decode_start=job.mark_decode_started,
                on_terminal=(
                    lambda finish_reason, cycles, job=job: self._close_terminal_job(
                        job,
                        finish_reason=finish_reason,
                        target_cycles=cycles,
                        route_id=str(getattr(self.lane, "route_id", "")),
                        real_width=real_width,
                        cohort_started_s=started,
                    )
                ),
                cancelled=job.cancel_requested,
            )
            for row, job in enumerate(jobs)
        ]
        result = self.driver(self.lane, requests)
        streams = list(result.streams)
        with self._condition:
            self._last_route_id = result.route_id
            self._target_verify_cycles += int(result.cycles)
            self._accepted_drafts += int(result.accepted_drafts)
            self._rejected_drafts += int(result.rejected_drafts)
            self._fixed_width_histogram.update(
                {int(width): int(count) for width, count in result.width_histogram.items()}
            )
        for job, stream in zip(jobs, streams, strict=True):
            if job.callback_error is not None:
                if not job.future.done():
                    job.future.set_exception(job.callback_error)
                continue
            if stream.finish_reason == "cancelled" or job.cancel_requested():
                if not job.future.done():
                    job.future.set_exception(self._cancelled_exception(job))
                continue
            self._complete_cohort_job(
                job,
                finish_reason=stream.finish_reason,
                route_id=result.route_id,
                real_width=real_width,
                target_cycles=result.cycles,
                cohort_started_s=started,
            )

    def _close_terminal_job(
        self,
        job: MTPBatchJob,
        *,
        finish_reason: str,
        target_cycles: int,
        route_id: str,
        real_width: int,
        cohort_started_s: float,
    ) -> None:
        if finish_reason == "cancelled" or job.cancel_requested():
            job.close_cancelled()
            return
        self._complete_cohort_job(
            job,
            finish_reason=finish_reason,
            route_id=route_id,
            real_width=real_width,
            target_cycles=target_cycles,
            cohort_started_s=cohort_started_s,
        )

    def _decode(self, tokens: list[int], stop_token_ids: set[int]) -> str:
        tokenizer = getattr(getattr(self.state, "runtime", None), "tokenizer", None)
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            return ""
        return str(
            decode([token for token in tokens if token not in stop_token_ids])
        )

    def _complete_cohort_job(
        self,
        job: MTPBatchJob,
        *,
        finish_reason: str,
        route_id: str,
        real_width: int,
        target_cycles: int,
        cohort_started_s: float,
    ) -> None:
        if job.future.done():
            return
        completed_s = time.perf_counter()
        request_elapsed_s = max(0.0, completed_s - job.created_s)
        decode_started_s = job.decode_started_s or cohort_started_s
        decode_elapsed_s = max(0.0, completed_s - decode_started_s)
        prefill_elapsed_s = max(0.0, decode_started_s - cohort_started_s)
        completion_tokens = len(job.tokens)
        decode_tok_s = (
            completion_tokens / decode_elapsed_s if decode_elapsed_s > 0 else 0.0
        )
        end_to_end_tok_s = (
            completion_tokens / request_elapsed_s if request_elapsed_s > 0 else 0.0
        )
        stats = {
            "mode": "mtp",
            "generation_mode": "mtp",
            "generated_tokens": completion_tokens,
            "elapsed_s": decode_elapsed_s,
            "decode_elapsed_s": decode_elapsed_s,
            "request_elapsed_s": request_elapsed_s,
            "prompt_eval_time_s": prefill_elapsed_s,
            "prefill_wall_time_s": prefill_elapsed_s,
            "decode_tok_s": decode_tok_s,
            "tok_s": decode_tok_s,
            "end_to_end_tok_s": end_to_end_tok_s,
            "mtp_depth": 1,
            "requested_mtp_depth": 1,
            "speculative_depth": 1,
            "requested_speculative_depth": 1,
            "verify_calls": int(target_cycles),
            "target_verify_cycles": int(target_cycles),
            "scheduler_lane": "mtp_batch",
            "scheduler_mode": "mtp_batch",
            "scheduler_policy": "fixed_mtp_batch_width_8",
            "request_id": job.request_id,
            "active_batch_size": real_width,
            "mtp_batch_real_width": real_width,
            "mtp_batch_fixed_width": 8,
            "mtp_batch_route_id": route_id,
            "mtp_disabled_reason": None,
            "queue_wait_s": max(
                0.0, (job.admitted_s or job.created_s) - job.created_s
            ),
            "request_started_s": job.created_s,
            "server_seed": job.seed,
        }
        stats.update(job.request_observability)
        job.emit_prefill(
            {
                "phase": "completed",
                "tokens_total": len(job.prompt_ids),
                "elapsed_s": request_elapsed_s,
                "scheduler_lane": "mtp_batch",
                "request_id": job.request_id,
            }
        )
        job.future.set_result(
            {
                "request_id": job.request_id,
                "text": self._decode(job.tokens, job.stop_token_ids),
                "tokens": list(job.tokens),
                "stats": stats,
                "prompt_tokens": len(job.prompt_ids),
                "completion_tokens": completion_tokens,
                "elapsed_s": decode_elapsed_s,
                "request_elapsed_s": request_elapsed_s,
                "tok_s": decode_tok_s,
                "end_to_end_tok_s": end_to_end_tok_s,
                "_final_state": None,
                "_token_times": list(job.token_times),
                "_generation_limits": dict(job.generation_limits),
                "finish_reason": finish_reason,
            }
        )

    def _fail_pending(self, exc: BaseException) -> None:
        with self._condition:
            pending = list(self._pending)
            self._pending.clear()
            self._pump_scheduled = False
            self._last_error = f"{type(exc).__name__}: {exc}"
        for job in pending:
            if not job.future.done():
                job.future.set_exception(exc)

    def shutdown(self) -> None:
        with self._condition:
            self._shutdown = True
            jobs = [*self._pending, *self._active]
            self._pending.clear()
            for job in jobs:
                job.cancel_event.set()
            self._condition.notify_all()
        for job in jobs:
            if not job.future.done():
                job.future.set_exception(RuntimeError("MTP batch service is shut down"))
