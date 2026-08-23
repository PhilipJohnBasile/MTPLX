from __future__ import annotations

from dataclasses import replace

import pytest

from mtplx.memory_governor import (
    GIB,
    MemoryGovernorAction,
    MemoryGovernorConfig,
    MemoryPressureLevel,
    MemorySafePoint,
    MemorySample,
    RuntimeMemoryGovernor,
    sample_process_memory,
)


class FakeBank:
    def __init__(self, max_bytes=20 * GIB, per_session=8 * GIB):
        self.max_bytes = max_bytes
        self.per_session_max_bytes = per_session
        self._entries = {"a": object(), "b": object()}
        self.total_nbytes = 12 * GIB
        self.calls = []

    def rebalance_limits(self, *, max_bytes, per_session_max_bytes, reason):
        self.calls.append((max_bytes, per_session_max_bytes, reason))
        self.max_bytes = int(max_bytes)
        self.per_session_max_bytes = int(per_session_max_bytes)
        if self.total_nbytes > self.max_bytes:
            self._entries.pop("a", None)
            self.total_nbytes = self.max_bytes


def _sample(
    rss_gib: float,
    *,
    bank_gib: float = 12,
    total_gib: float = 100,
    safe: bool = True,
    timestamp: float = 10.0,
):
    return MemorySample(
        total_bytes=int(total_gib * GIB),
        rss_bytes=int(rss_gib * GIB),
        session_bank_bytes=int(bank_gib * GIB),
        model_bytes=60 * GIB,
        timestamp_s=timestamp,
        safe_point=(
            MemorySafePoint()
            if safe
            else MemorySafePoint(foreground_active=1)
        ),
    )


def _governor(**config_overrides):
    config = MemoryGovernorConfig(
        minimum_apply_interval_s=0.0,
        **config_overrides,
    )
    return RuntimeMemoryGovernor(
        initial_bank_max_bytes=20 * GIB,
        initial_per_session_max_bytes=8 * GIB,
        config=config,
    )


def test_critical_pressure_shrinks_immediately_and_disables_speculation():
    governor = _governor()
    decision = governor.observe(_sample(95))
    assert decision.pressure == MemoryPressureLevel.CRITICAL
    assert decision.action == MemoryGovernorAction.SHRINK
    assert decision.target_bank_max_bytes < 20 * GIB
    assert decision.prefill_chunk_tokens == 512
    assert decision.max_concurrency == 1
    assert decision.speculative_allowed is False


def test_high_pressure_requires_hysteresis_observations():
    governor = _governor(high_observations=2)
    first = governor.observe(_sample(87, timestamp=1))
    second = governor.observe(_sample(87, timestamp=2))
    assert first.action == MemoryGovernorAction.HOLD
    assert second.action == MemoryGovernorAction.SHRINK
    assert second.reason == "sustained_high_pressure"


def test_normal_observation_resets_pressure_streak():
    governor = _governor(high_observations=2)
    governor.observe(_sample(87, timestamp=1))
    governor.observe(_sample(80, timestamp=2))
    decision = governor.observe(_sample(87, timestamp=3))
    assert decision.action == MemoryGovernorAction.HOLD
    assert governor.high_streak == 1


def test_recovery_grows_only_after_hysteresis_and_never_past_startup_budget():
    governor = _governor(recovery_observations=3)
    governor.current_bank_max_bytes = 8 * GIB
    governor.current_per_session_max_bytes = 5 * GIB
    assert governor.observe(_sample(55, bank_gib=6, timestamp=1)).action == MemoryGovernorAction.HOLD
    assert governor.observe(_sample(55, bank_gib=6, timestamp=2)).action == MemoryGovernorAction.HOLD
    decision = governor.observe(_sample(55, bank_gib=6, timestamp=3))
    assert decision.action == MemoryGovernorAction.GROW
    assert 8 * GIB < decision.target_bank_max_bytes <= 20 * GIB
    assert decision.target_per_session_max_bytes <= decision.target_bank_max_bytes


def test_unknown_total_memory_holds_current_budget():
    governor = _governor()
    decision = governor.observe(
        MemorySample(
            total_bytes=None,
            rss_bytes=10 * GIB,
            session_bank_bytes=2 * GIB,
            timestamp_s=1,
        )
    )
    assert decision.pressure == MemoryPressureLevel.UNKNOWN
    assert decision.action == MemoryGovernorAction.HOLD


def test_unsafe_point_rejects_budget_mutation():
    governor = _governor()
    bank = FakeBank()
    decision = governor.observe(_sample(95, safe=False))
    receipt = governor.apply(decision, bank=bank)
    assert receipt.applied is False
    assert receipt.reason == "unsafe_point:foreground"
    assert bank.calls == []
    assert bank.max_bytes == 20 * GIB


def test_safe_point_applies_limits_and_policy_callbacks():
    governor = _governor()
    bank = FakeBank()
    values = {}
    decision = governor.observe(_sample(95, safe=True))
    receipt = governor.apply(
        decision,
        bank=bank,
        policy_callbacks={
            "prefill_chunk_tokens": lambda value: values.__setitem__("chunk", value),
            "max_concurrency": lambda value: values.__setitem__("concurrency", value),
            "speculative_allowed": lambda value: values.__setitem__("speculative", value),
        },
    )
    assert receipt.applied is True
    assert bank.max_bytes == decision.target_bank_max_bytes
    assert bank.per_session_max_bytes == decision.target_per_session_max_bytes
    assert bank.calls[-1][2] == "runtime_memory_governor"
    assert values == {"chunk": 512, "concurrency": 1, "speculative": False}
    assert receipt.evicted_entries == 1


@pytest.mark.parametrize(
    ("safe_point", "blocker"),
    [
        (MemorySafePoint(foreground_active=1), "foreground"),
        (MemorySafePoint(scheduler_pending_or_active=True), "scheduler"),
        (MemorySafePoint(session_restore_active=True), "session_restore"),
        (MemorySafePoint(session_commit_active=True), "session_commit"),
        (MemorySafePoint(mtp_transaction_active=True), "mtp_transaction"),
        (MemorySafePoint(postcommit_active=True), "postcommit"),
    ],
)
def test_every_live_state_blocks_application(safe_point, blocker):
    governor = _governor()
    bank = FakeBank()
    decision = governor.observe(replace(_sample(95), safe_point=safe_point))
    receipt = governor.apply(decision, bank=bank)
    assert receipt.applied is False
    assert blocker in receipt.reason


@pytest.mark.parametrize(
    ("rss", "expected"),
    [
        (65, MemoryPressureLevel.LOW),
        (78, MemoryPressureLevel.NORMAL),
        (86, MemoryPressureLevel.HIGH),
        (93, MemoryPressureLevel.CRITICAL),
    ],
)
def test_pressure_matrix(rss, expected):
    governor = _governor(high_observations=1)
    assert governor.observe(_sample(rss)).pressure == expected


def test_sample_process_memory_respects_supplied_values_without_platform_probe():
    sample = sample_process_memory(
        session_bank_bytes=3 * GIB,
        model_bytes=50 * GIB,
        total_bytes=128 * GIB,
        rss_bytes=90 * GIB,
        safe_point=MemorySafePoint(),
    )
    assert sample.total_bytes == 128 * GIB
    assert sample.rss_bytes == 90 * GIB
    assert sample.session_bank_bytes == 3 * GIB
    assert sample.safe_point.is_safe
    assert sample.to_dict()["utilization"] == pytest.approx(90 / 128)


def test_metrics_include_last_decision_and_receipt():
    governor = _governor()
    bank = FakeBank()
    decision = governor.observe(_sample(95))
    governor.apply(decision, bank=bank)
    metrics = governor.to_metrics()
    assert metrics["memory_governor_last_decision"]["pressure"] == "critical"
    assert metrics["memory_governor_last_apply"]["applied"] is True
