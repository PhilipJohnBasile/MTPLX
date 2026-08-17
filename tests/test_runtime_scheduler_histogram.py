"""Model-owner batch histogram must record batches, not bookkeeping (F13a).

``batch_histogram`` is public scheduler telemetry: microbatch sizes executed
on the model owner. The completion path used to stamp ``[1] += 1`` for EVERY
finished work item — idle postcommits, SSD-persistence encodes, and the
long-lived decode pump itself (whose real per-step sizes already arrive via
``record_batch_step``). Under batch-8 load the histogram grew a fat spurious
"1" bar, which reads as "batching is broken" to anyone auditing /health
during a benchmark.
"""

from __future__ import annotations

from mtplx.model_scheduler import ModelWorkScheduler


def _histogram(scheduler) -> dict[str, int]:
    return scheduler.stats()["batch_histogram"]


def test_plain_foreground_item_counts_as_one():
    scheduler = ModelWorkScheduler(name="test-hist-fg", idle_grace_s=0.0)
    try:
        scheduler.submit_foreground(lambda: "ok").result(timeout=2)
        assert _histogram(scheduler) == {"1": 1}
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_idle_kinds_do_not_pollute_the_histogram():
    scheduler = ModelWorkScheduler(name="test-hist-idle", idle_grace_s=0.0)
    try:
        scheduler.submit_idle_postcommit(lambda: "commit").result(timeout=2)
        scheduler.submit_idle_persistence(lambda: "encode").result(timeout=2)
        assert _histogram(scheduler) == {}
        assert scheduler.stats()["completed"] == 2
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_self_reporting_pump_records_only_true_sizes():
    scheduler = ModelWorkScheduler(name="test-hist-pump", idle_grace_s=0.0)
    try:

        def pump() -> str:
            # A long-lived decode pump: three microbatch steps of size 8.
            for _ in range(3):
                scheduler.record_batch_step(size=8, batch_key="ar_batch.decode")
            return "drained"

        scheduler.submit_foreground(pump, batch_key="ar_batch.pump").result(
            timeout=2
        )
        # The pump's completion must not append a phantom size-1 batch.
        assert _histogram(scheduler) == {"8": 3}
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_self_report_flag_resets_between_items():
    scheduler = ModelWorkScheduler(name="test-hist-reset", idle_grace_s=0.0)
    try:

        def pump() -> str:
            scheduler.record_batch_step(size=4, batch_key="ar_batch.decode")
            return "drained"

        scheduler.submit_foreground(pump, batch_key="ar_batch.pump").result(
            timeout=2
        )
        # The next plain foreground item is not exempted by the pump's report.
        scheduler.submit_foreground(lambda: "ok").result(timeout=2)
        assert _histogram(scheduler) == {"1": 1, "4": 1}
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)
