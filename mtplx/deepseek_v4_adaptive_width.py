"""Construction contract for the preregistered DeepSeek-V4 max-K3 policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .sampling import SamplerConfig


D1_MARGIN_THRESHOLD = 0.25
D2_MARGIN_THRESHOLD = 1.0
MAX_SPECULATIVE_DEPTH = 3


@dataclass(frozen=True, slots=True)
class DeepSeekV4TargetWidthRoute:
    """One prebound target-forward surface for an exact verify width."""

    target_rows: int
    forward: Callable[..., Any]

    def __call__(self, input_ids: Any, **kwargs: Any) -> Any:
        return self.forward(input_ids, **kwargs)


@dataclass(frozen=True, slots=True)
class DeepSeekV4AdaptiveWidthPolicy:
    """The single preregistered policy and its construction-validated surfaces."""

    runtime_object_id: int
    target_routes: tuple[
        DeepSeekV4TargetWidthRoute,
        DeepSeekV4TargetWidthRoute,
        DeepSeekV4TargetWidthRoute,
    ]
    d1_margin_threshold: float = field(default=D1_MARGIN_THRESHOLD, init=False)
    d2_margin_threshold: float = field(default=D2_MARGIN_THRESHOLD, init=False)
    max_speculative_depth: int = field(default=MAX_SPECULATIVE_DEPTH, init=False)
    verify_strategy: str = field(default="capture_commit", init=False)
    verify_core: str = field(default="stock", init=False)
    mtp_history_policy: str = field(default="committed", init=False)

    def stop_after_d1(self, margin: float) -> bool:
        return float(margin) < self.d1_margin_threshold

    def stop_after_d2(self, margin: float) -> bool:
        return float(margin) < self.d2_margin_threshold

    def validate_request(
        self,
        rt: Any,
        *,
        sampler: SamplerConfig,
        draft_sampler: SamplerConfig,
        speculative_depth: int,
        verify_strategy: str,
        verify_core: str,
        mtp_history_policy: str,
    ) -> None:
        """Reject a launch that differs from the installed policy contract."""

        if id(rt) != self.runtime_object_id:
            raise ValueError("adaptive width policy belongs to a different runtime")
        _validate_launch(
            sampler=sampler,
            draft_sampler=draft_sampler,
            speculative_depth=speculative_depth,
            verify_strategy=verify_strategy,
            verify_core=verify_core,
            mtp_history_policy=mtp_history_policy,
        )


def _validate_launch(
    *,
    sampler: SamplerConfig,
    draft_sampler: SamplerConfig,
    speculative_depth: int,
    verify_strategy: str,
    verify_core: str,
    mtp_history_policy: str,
) -> None:
    if float(sampler.temperature) > 0.0:
        raise ValueError("adaptive width policy requires a greedy target sampler")
    if float(draft_sampler.temperature) > 0.0:
        raise ValueError("adaptive width policy requires a greedy draft sampler")
    if int(speculative_depth) != MAX_SPECULATIVE_DEPTH:
        raise ValueError("adaptive width policy requires fixed planned max-K3")
    if verify_strategy != "capture_commit":
        raise ValueError("adaptive width policy requires capture_commit verification")
    if verify_core != "stock":
        raise ValueError("adaptive width policy requires the stock verify core")
    if mtp_history_policy != "committed":
        raise ValueError("adaptive width policy requires committed MTP history")


def _validate_runtime(rt: Any) -> Callable[..., Any]:
    if not bool(getattr(rt, "mtp_enabled", False)):
        raise ValueError("adaptive width policy requires an MTP-enabled runtime")
    model = getattr(rt, "model", None)
    model_type = str(getattr(model, "model_type", "") or "").lower()
    if model_type != "deepseek_v4":
        raise ValueError("adaptive width policy is only valid for DeepSeek-V4")

    report = getattr(rt, "deepseek_v4_o_lora_report", None)
    census = report.get("callable_census", {}) if isinstance(report, dict) else {}
    route_ok = bool(
        isinstance(report, dict)
        and report.get("mode") == "gather_qmm"
        and report.get("module_count") == 44
        and report.get("trunk_module_count") == 43
        and report.get("mtp_module_count") == 1
        and report.get("body_direct") == 43
        and report.get("mtp_stock") == 1
        and report.get("body_all_mode_matches") is True
        and report.get("route_plan_matches") is True
        and census.get("body_route_objects") == 43
        and census.get("body_route_kind") == "gather_qmm_direct"
        and census.get("body_callable_class") == "_DirectGatherOLora"
        and census.get("mtp_route_objects") == 1
        and census.get("mtp_route_kind") == "dense_bf16_stock_direct"
        and census.get("mtp_callable_class") == "_DirectDenseMTPOLora"
        and census.get("total_route_objects") == 44
        and census.get("unique_route_objects") == 44
        and census.get("mtp_distinct_type") is True
    )
    if not route_ok:
        raise ValueError("adaptive width policy requires the canonical o-LoRA route")

    forward = getattr(rt, "forward_ar_capture", None)
    if not callable(forward):
        raise ValueError("adaptive width policy requires a callable capture target forward")
    return forward


def install_deepseek_v4_adaptive_width_policy(
    rt: Any,
    *,
    sampler: SamplerConfig,
    draft_sampler: SamplerConfig | None,
    speculative_depth: int,
    verify_strategy: str,
    verify_core: str,
    mtp_history_policy: str,
) -> DeepSeekV4AdaptiveWidthPolicy:
    """Validate and bind the only supported adaptive-width configuration."""

    resolved_draft_sampler = sampler if draft_sampler is None else draft_sampler
    _validate_launch(
        sampler=sampler,
        draft_sampler=resolved_draft_sampler,
        speculative_depth=speculative_depth,
        verify_strategy=verify_strategy,
        verify_core=verify_core,
        mtp_history_policy=mtp_history_policy,
    )
    target_forward = _validate_runtime(rt)
    target_routes = tuple(
        DeepSeekV4TargetWidthRoute(target_rows=rows, forward=target_forward)
        for rows in (2, 3, 4)
    )
    return DeepSeekV4AdaptiveWidthPolicy(
        runtime_object_id=id(rt),
        target_routes=target_routes,  # type: ignore[arg-type]
    )
