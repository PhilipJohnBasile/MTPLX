"""Draft proposal sampler metadata helpers."""

from __future__ import annotations

from typing import Any


def resolve_draft_temperature(
    curve: Any,
    target_temperature: float | None,
    *,
    default: float,
) -> float:
    """Map an effective target temperature to a draft temperature.

    ``curve`` is a sequence of (target_temperature, draft_temperature)
    points; interpolation is piecewise linear with flat extrapolation at the
    ends. An empty/None curve or unknown target returns ``default`` — the
    identity policy (today's static draft temperature). Correctness never
    depends on this value: probability-ratio acceptance derives p and q
    independently, so any draft temperature preserves the output marginal.
    """

    if not curve or target_temperature is None:
        return float(default)
    try:
        points = sorted(
            (float(target), float(draft)) for target, draft in curve
        )
    except (TypeError, ValueError):
        return float(default)
    if not points:
        return float(default)
    t = float(target_temperature)
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= t <= x1:
            if x1 == x0:
                return y1
            frac = (t - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return float(default)


def normalize_draft_sampler_spec(
    value: Any,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a validated draft sampler override from profile/contract data."""
    if value is None:
        return fallback
    if not isinstance(value, dict):
        raise ValueError("draft sampler spec must be an object")
    temperature = float(value.get("temperature", 0.6))
    top_p = float(value.get("top_p", 0.95))
    top_k = int(value.get("top_k", 20))
    if temperature < 0:
        raise ValueError("draft sampler temperature must be non-negative")
    if top_p <= 0:
        raise ValueError("draft sampler top_p must be positive")
    if top_k < 0:
        raise ValueError("draft sampler top_k must be non-negative")
    return {"temperature": temperature, "top_p": top_p, "top_k": top_k}


def draft_sampler_spec_from_runtime_contract(
    contract_data: Any,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve a model-specific draft sampler recommendation."""
    if not isinstance(contract_data, dict):
        return fallback
    return normalize_draft_sampler_spec(
        contract_data.get("recommended_draft_sampler"),
        fallback=fallback,
    )
