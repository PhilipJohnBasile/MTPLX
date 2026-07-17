"""Regression tests for the qwen3_5_mtp backend.

Hermetic: covers config detection, the trunk-load shim, mtp.* key remapping,
and arch registration. The full-checkpoint draft-acceptance contract is
validated during hardware bring-up (see the module docstring), not here.
"""
import sys

from mtplx.qwen3_5_mtp_patch import (
    is_qwen3_5_mtp_config,
    install_qwen3_5_mtp_trunk_shim,
    _strip_mtp_prefix,
)


def test_config_detection_positive():
    assert is_qwen3_5_mtp_config({"model_type": "qwen3_5_mtp", "num_nextn_predict_layers": 1})
    # num_nextn nested under text_config is also honored
    assert is_qwen3_5_mtp_config(
        {"model_type": "qwen3_5_mtp", "text_config": {"num_nextn_predict_layers": 1}}
    )


def test_config_detection_negative():
    # AR export (same trunk, different model_type) must NOT trigger the MTP path
    assert not is_qwen3_5_mtp_config({"model_type": "qwen3_5_moe", "num_nextn_predict_layers": 1})
    # MTP model_type but no predictor declared
    assert not is_qwen3_5_mtp_config({"model_type": "qwen3_5_mtp", "num_nextn_predict_layers": 0})


def test_trunk_shim_makes_model_type_importable():
    # The shim is process-global by design in production; tests must undo the
    # sys.modules mutation or it leaks into other suites (it made
    # test_mtp_alias_load_path's known-alias case see a "native" module and
    # skip building the #147 wrapper).
    import importlib

    try:
        install_qwen3_5_mtp_trunk_shim()

        import mlx_lm.models.qwen3_5_moe as base

        mod = importlib.import_module("mlx_lm.models.qwen3_5_mtp")
        # shim exposes the trunk classes; Model subclasses the vanilla MoE trunk
        # but strips mtp.* in sanitize to avoid the double norm-shift
        assert hasattr(mod, "Model") and hasattr(mod, "ModelArgs")
        assert issubclass(mod.Model, base.Model)
        assert mod.ModelArgs is base.ModelArgs
    finally:
        sys.modules.pop("mlx_lm.models.qwen3_5_mtp", None)


def test_strip_mtp_prefix():
    assert _strip_mtp_prefix("mtp.fc.weight") == "fc.weight"
    assert _strip_mtp_prefix("language_model.mtp.norm.weight") == "norm.weight"
    assert _strip_mtp_prefix("model.mtp.layers.0.self_attn.q_proj.weight") == "layers.0.self_attn.q_proj.weight"
    # trunk weights are not MTP keys
    assert _strip_mtp_prefix("language_model.model.layers.0.self_attn.q_proj.weight") is None
    assert _strip_mtp_prefix("lm_head.weight") is None


def test_arch_registered():
    from mtplx.backends.registry import SUPPORTED_ARCH_IDS

    # qwen3_5_mtp routes through the existing qwen3-next-mtp arch/backend
    assert "qwen3-next-mtp" in SUPPORTED_ARCH_IDS
