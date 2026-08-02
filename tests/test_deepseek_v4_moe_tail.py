"""Source and construction contracts for the DeepSeek-V4 MoE tail lane.

The candidate deliberately leaves ``SwitchGLU`` alone.  The only replacement is
the stock tail::

    (routed * weights[..., None].astype(routed.dtype)).sum(axis=-2) + shared

for the shipped body geometry: BF16, top-k six, hidden 4096.  The direct Metal
test belongs in a guarded GPU window; these CPU-safe tests pin the invariants
that decide whether such a route may be installed at all.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_HERE, "..", "mtplx", "models", "deepseek_v4.py")
_spec = importlib.util.spec_from_file_location("dsv4_moe_tail_undertest", _MODEL)
D = importlib.util.module_from_spec(_spec)
sys.modules["dsv4_moe_tail_undertest"] = D
_spec.loader.exec_module(D)


def _args(**over):
    kwargs = dict(
        hidden_size=4096,
        moe_intermediate_size=1536,
        n_routed_experts=256,
        num_experts_per_tok=6,
        num_hash_layers=3,
        compress_ratios=[0] * 44,
    )
    kwargs.update(over)
    return D.ModelArgs(**kwargs)


def test_tail_default_is_off_and_stock_expression_remains_visible():
    """The opt-in cannot affect the ordinary model construction path."""
    assert D._MOE_TAIL is False
    source = open(_MODEL, encoding="utf-8").read()
    assert "(routed * weights[..., None].astype(routed.dtype)).sum(axis=-2)" in source


def test_tail_geometry_validation_accepts_only_shipped_body_contract():
    """Top-k, hidden width, and BF16-store arm are installation invariants."""
    D._validate_moe_tail_config(_args())
    with pytest.raises(ValueError, match="top-k=6"):
        D._validate_moe_tail_config(_args(num_experts_per_tok=4))
    with pytest.raises(ValueError, match="hidden_size=4096"):
        D._validate_moe_tail_config(_args(hidden_size=2048))


def test_tail_rejects_fp32_activation_arm_at_installation():
    """The kernel has BF16 arithmetic by contract, never a hot-path dtype test."""
    saved = D._FP32_ACTIVATIONS
    try:
        D._FP32_ACTIVATIONS = True
        with pytest.raises(ValueError, match="BF16 activation storage"):
            D._validate_moe_tail_config(_args())
    finally:
        D._FP32_ACTIVATIONS = saved


def test_tail_kernel_uses_one_output_owner_and_real_metal_exact_selfcheck():
    """Association is not assumed: the constructed GPU route has to prove it."""
    source = D._MOE_TAIL_METAL_SOURCE
    assert "uint i = thread_position_in_grid.x" in source
    assert "for (uint route = 0; route < TOPK; ++route)" in source
    assert "T product" in source
    assert "T(mixed + shared" in source
    implementation = open(_MODEL, encoding="utf-8").read()
    assert "_verify_moe_tail_exact(kernel)" in implementation
    assert "for rows in (1, 4):" in implementation
    assert "routes = {" in implementation
    assert "_stock_moe_tail_combine" in implementation


def test_tail_is_not_a_cpu_silent_fallback_when_explicitly_enabled():
    """An enabled Metal lane must fail before generation on an unsupported device."""
    with pytest.raises(RuntimeError, match="GPU"):
        D._install_moe_tail_combine(_args())


def test_guarded_tail_gate_is_one_load_and_synchronizes_each_sample():
    """Its timings are diagnostics only; the later full TPS bracket is the verdict."""
    source = (
        Path(_HERE).parent / "scripts" / "deepseek_v4_moe_tail_gate.py"
    ).read_text(encoding="utf-8")
    assert "_load_base_model" in source
    assert "--prompt-tokens" in source
    assert "mx.eval(out)" in source
    assert "mx.synchronize()" in source
    assert "exact_parity" in source
    assert "promotion" not in source.lower()
