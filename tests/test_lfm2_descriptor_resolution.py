"""LFM2 runtime resolves the LFM-grammar descriptor on the mlx-lm AR lane.

MLX-free: descriptors are deliberately import-light, so these run everywhere.
"""

from __future__ import annotations

from types import SimpleNamespace

from mtplx.backends.descriptors import (
    MLX_LM_AR_DESCRIPTOR,
    MLX_LM_AR_LFM2_DESCRIPTOR,
    descriptor_from_runtime,
    model_family_from_inspection,
    reasoning_policy_for_model,
)


def _runtime(model_type: str, backend_id: str = "mlx_lm_ar", path: str = "/m/x"):
    return SimpleNamespace(
        backend_id=backend_id,
        model=SimpleNamespace(args=SimpleNamespace(model_type=model_type)),
        model_path=path,
    )


def test_lfm2_runtime_gets_lfm_grammar_descriptor() -> None:
    descriptor = descriptor_from_runtime(_runtime("lfm2_moe"))
    assert descriptor is MLX_LM_AR_LFM2_DESCRIPTOR
    assert descriptor.backend_id == MLX_LM_AR_DESCRIPTOR.backend_id
    assert descriptor.reasoning_codec.parser == "lfm2"


def test_lfm2_dense_model_type_also_matches() -> None:
    assert descriptor_from_runtime(_runtime("lfm2")).reasoning_codec.parser == "lfm2"


def test_non_lfm_runtime_keeps_generic_ar_descriptor() -> None:
    descriptor = descriptor_from_runtime(_runtime("iquestcoder"))
    assert descriptor is MLX_LM_AR_DESCRIPTOR
    assert descriptor.reasoning_codec.parser == "none"


def test_lfm2_model_path_is_a_fallback_marker() -> None:
    runtime = SimpleNamespace(
        backend_id="mlx_lm_ar",
        model=SimpleNamespace(args=SimpleNamespace(model_type="")),
        model_path="/Users/x/.mtplx/models/LiquidAI--LFM2.5-8B-A1B-MLX-8bit",
    )
    assert descriptor_from_runtime(runtime) is MLX_LM_AR_LFM2_DESCRIPTOR


def test_lfm2_sniff_never_rebrands_other_backends() -> None:
    runtime = _runtime("lfm2_moe", backend_id="laguna_ar")
    assert descriptor_from_runtime(runtime).backend_id == "laguna_ar"


def test_lfm2_family_marker_and_reasoning_policy() -> None:
    family = model_family_from_inspection(
        model_ref="LiquidAI--LFM2.5-8B-A1B-MLX-8bit",
    )
    assert family == "lfm2"
    codec = reasoning_policy_for_model(
        model_ref="LiquidAI--LFM2.5-8B-A1B-MLX-8bit",
    )
    assert codec.parser == "lfm2"
