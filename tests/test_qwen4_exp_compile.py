import mlx.core as mx
import pytest
from mtplx.models.qwen4_exp import ModelArgs, Model


def _make_dummy_model():
    args = ModelArgs(
        model_type="qwen4_exp",
        text_config={
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 1000,
            "layer_types": ["linear_attention", "linear_attention"],
        },
    )
    return Model(args)


def test_qwen4_exp_compile_flag_support(monkeypatch):
    """Verify that MTPLX_QWEN4EXP_COMPILE flag enables the compiled GDN decode path."""
    monkeypatch.setenv("MTPLX_QWEN4EXP_COMPILE", "1")
    monkeypatch.delenv("MTPLX_COMPILED_GDN", raising=False)
    model = _make_dummy_model()
    text_model = model.language_model.model
    assert text_model._gdn_compiled_env is True


def test_qwen4_exp_compile_kill_switch_precedence(monkeypatch):
    """Verify that an explicit 0 on either flag takes precedence as a kill switch."""
    # When MTPLX_COMPILED_GDN=0, compilation must be disabled even if MTPLX_QWEN4EXP_COMPILE=1
    monkeypatch.setenv("MTPLX_COMPILED_GDN", "0")
    monkeypatch.setenv("MTPLX_QWEN4EXP_COMPILE", "1")
    model = _make_dummy_model()
    text_model = model.language_model.model
    assert text_model._gdn_compiled_env is False
    assert text_model._gdn_compile_explicit_off is True

    # When MTPLX_QWEN4EXP_COMPILE=0, compilation must be disabled even if MTPLX_COMPILED_GDN=1
    monkeypatch.setenv("MTPLX_COMPILED_GDN", "1")
    monkeypatch.setenv("MTPLX_QWEN4EXP_COMPILE", "0")
    model2 = _make_dummy_model()
    text_model2 = model2.language_model.model
    assert text_model2._gdn_compiled_env is False
    assert text_model2._gdn_compile_explicit_off is True


def test_qwen4_exp_compile_kill_switch_overrides_pipeline_lane(monkeypatch):
    """Verify that explicit compile-off overrides set_ar_pipeline_mode."""
    monkeypatch.setenv("MTPLX_COMPILED_GDN", "0")
    model = _make_dummy_model()
    model.set_ar_pipeline_mode(True)
    text_model = model.language_model.model
    assert text_model._gdn_compiled_lane is False
