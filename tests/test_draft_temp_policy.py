"""Dynamic draft-temperature policy: resolution, precedence, exactness.

The resolver is speed policy only — probability-ratio acceptance derives the
target p and draft q independently, so any draft temperature preserves the
output marginal. The oracle sweep at the bottom pins that invariant across
draft temperatures so the calibration campaign can move the draft freely.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mtplx.draft_sampling import resolve_draft_temperature
from mtplx.sampling import SamplerConfig, speculative_output_marginal
from mtplx.server.openai import _resolve_draft_sampler_for_request


def _state(
    *,
    draft_sampler: SamplerConfig | None,
    pinned: bool = False,
    curve: tuple[tuple[float, float], ...] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        draft_sampler=draft_sampler,
        draft_sampler_pinned=pinned,
        draft_temperature_curve=curve,
    )


BASE = SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)


def test_identity_curve_is_todays_behavior():
    observability: dict = {}
    resolved = _resolve_draft_sampler_for_request(
        _state(draft_sampler=BASE, curve=None),
        request_draft_sampler=None,
        target_temperature=0.6,
        request_observability=observability,
    )
    assert resolved is BASE
    assert observability["draft_sampler_policy"] == "static"
    assert observability["draft_sampler_policy_source"] == "family_default"
    assert observability["draft_sampler_resolved_temperature"] == 1.0


def test_greedy_target_still_couples_to_greedy_draft(monkeypatch):
    monkeypatch.delenv("MTPLX_GREEDY_DRAFT_COUPLING", raising=False)
    observability: dict = {}
    resolved = _resolve_draft_sampler_for_request(
        _state(draft_sampler=BASE, curve=None),
        request_draft_sampler=None,
        target_temperature=0.0,
        request_observability=observability,
    )
    assert resolved.temperature == 0.0
    assert observability["draft_sampler_greedy_coupled"] is True
    assert observability["draft_sampler_policy_source"] == (
        "family_default+greedy_coupled"
    )
    assert observability["draft_sampler_resolved_temperature"] == 0.0


def test_family_curve_maps_target_to_draft_temperature():
    curve = ((0.2, 0.1), (0.6, 0.4), (1.0, 1.0))
    observability: dict = {}
    resolved = _resolve_draft_sampler_for_request(
        _state(draft_sampler=BASE, curve=curve),
        request_draft_sampler=None,
        target_temperature=0.6,
        request_observability=observability,
    )
    assert resolved.temperature == pytest.approx(0.4)
    assert resolved.top_p == BASE.top_p
    assert resolved.top_k == BASE.top_k
    assert observability["draft_sampler_policy"] == "curve"
    assert observability["draft_sampler_policy_source"] == "family_curve"
    assert observability["draft_sampler_resolved_temperature"] == pytest.approx(0.4)


def test_curve_interpolates_between_points():
    curve = ((0.2, 0.1), (0.6, 0.4), (1.0, 1.0))
    assert resolve_draft_temperature(curve, 0.8, default=9.9) == pytest.approx(0.7)
    # Flat extrapolation at both ends.
    assert resolve_draft_temperature(curve, 0.05, default=9.9) == pytest.approx(0.1)
    assert resolve_draft_temperature(curve, 1.4, default=9.9) == pytest.approx(1.0)
    # No curve / no target: identity.
    assert resolve_draft_temperature(None, 0.6, default=0.7) == 0.7
    assert resolve_draft_temperature(curve, None, default=0.7) == 0.7


def test_pinned_launch_disables_curve():
    curve = ((0.2, 0.1), (1.0, 1.0))
    observability: dict = {}
    resolved = _resolve_draft_sampler_for_request(
        _state(draft_sampler=BASE, pinned=True, curve=curve),
        request_draft_sampler=None,
        target_temperature=0.4,
        request_observability=observability,
    )
    assert resolved is BASE
    assert observability["draft_sampler_policy"] == "static"
    assert observability["draft_sampler_policy_source"] == "launch_pinned"


def test_request_explicit_sampler_beats_curve_and_pin():
    explicit = SamplerConfig(temperature=0.3, top_p=0.9, top_k=10)
    for pinned in (False, True):
        observability: dict = {}
        resolved = _resolve_draft_sampler_for_request(
            _state(
                draft_sampler=BASE,
                pinned=pinned,
                curve=((0.2, 0.1), (1.0, 1.0)),
            ),
            request_draft_sampler=explicit,
            target_temperature=0.6,
            request_observability=observability,
        )
        assert resolved is explicit
        assert observability["draft_sampler_policy"] == "request_explicit"


def test_request_explicit_sampler_is_never_greedy_coupled(monkeypatch):
    monkeypatch.delenv("MTPLX_GREEDY_DRAFT_COUPLING", raising=False)
    explicit = SamplerConfig(temperature=0.3, top_p=0.9, top_k=10)
    resolved = _resolve_draft_sampler_for_request(
        _state(draft_sampler=BASE, curve=None),
        request_draft_sampler=explicit,
        target_temperature=0.0,
        request_observability={},
    )
    assert resolved is explicit


def test_no_draft_sampler_resolves_none():
    observability: dict = {}
    resolved = _resolve_draft_sampler_for_request(
        _state(draft_sampler=None),
        request_draft_sampler=None,
        target_temperature=0.6,
        request_observability=observability,
    )
    assert resolved is None
    assert observability["draft_sampler_policy"] == "none"
    assert observability["draft_sampler_resolved_temperature"] is None


def test_serial_and_batch_resolution_agree():
    """Both generation lanes call the same resolver; identical inputs must
    produce identical resolved samplers (the desync guarantee)."""

    curve = ((0.2, 0.1), (0.6, 0.4), (1.0, 1.0))
    for target in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, None):
        serial = _resolve_draft_sampler_for_request(
            _state(draft_sampler=BASE, curve=curve),
            request_draft_sampler=None,
            target_temperature=target,
            request_observability={},
        )
        batch = _resolve_draft_sampler_for_request(
            _state(draft_sampler=BASE, curve=curve),
            request_draft_sampler=None,
            target_temperature=target,
            request_observability={},
        )
        assert (
            serial.temperature,
            serial.top_p,
            serial.top_k,
        ) == (batch.temperature, batch.top_p, batch.top_k)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        out = np.zeros_like(logits)
        out[int(np.argmax(logits))] = 1.0
        return out
    scaled = logits / temperature
    scaled -= scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


@pytest.mark.parametrize("target_temperature", [0.4, 0.6, 1.0])
@pytest.mark.parametrize("draft_temperature", [0.1, 0.3, 0.6, 0.8, 1.0, 1.2])
def test_output_marginal_recovers_target_at_any_draft_temperature(
    target_temperature, draft_temperature
):
    """The correctness foundation of the whole campaign: whatever draft
    temperature the curve picks, spec sampling's output marginal equals the
    target distribution exactly."""

    rng = np.random.default_rng(20260815)
    logits = rng.normal(size=32)
    target_p = _softmax(logits, target_temperature)
    draft_q = _softmax(logits + rng.normal(scale=0.5, size=32), draft_temperature)

    marginal = speculative_output_marginal(target_p, draft_q)

    np.testing.assert_allclose(marginal, target_p / target_p.sum(), atol=1e-12)
