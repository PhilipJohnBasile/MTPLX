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
from mtplx.attention_context import attention_phase  # noqa: E402

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
    assert "current_attention_phase()" in implementation
    assert "_stock_moe_tail_combine" in implementation


@pytest.mark.parametrize("rows", [1, 4])
def test_prefill_tiny_shapes_remain_stock(monkeypatch, rows):
    """Flattened M alone cannot turn a tiny prefill into decode/verify."""
    monkeypatch.setattr(D, "_moe_tail_apply", lambda *_: mx.array([-99.0]))
    route = D._InstalledMoETailRoute(kernel=object())
    routed = mx.zeros((rows, 6, 8), dtype=mx.bfloat16)
    weights = mx.ones((rows, 6), dtype=mx.bfloat16)
    shared = mx.ones((rows, 8), dtype=mx.bfloat16)
    with attention_phase("prefill"):
        got = route(routed, weights, shared)
    assert tuple(got.shape) == (rows, 8)
    assert bool(mx.all(got == 1))


def test_decode_verify_m4_uses_custom(monkeypatch):
    sentinel = mx.array([-99.0])
    monkeypatch.setattr(D, "_moe_tail_apply", lambda *_: sentinel)
    route = D._InstalledMoETailRoute(kernel=object())
    with attention_phase("decode_verify"):
        got = route(
            mx.zeros((4, 6, 8)), mx.zeros((4, 6)), mx.zeros((4, 8))
        )
    assert got is sentinel


@pytest.mark.parametrize("phase", ["ar_decode", "unknown"])
def test_m1_stays_stock_outside_verify_route(monkeypatch, phase):
    sentinel = mx.array([-99.0])
    monkeypatch.setattr(D, "_moe_tail_apply", lambda *_: sentinel)
    route = D._InstalledMoETailRoute(kernel=object())
    routed = mx.zeros((1, 6, 8), dtype=mx.bfloat16)
    weights = mx.zeros((1, 6), dtype=mx.bfloat16)
    shared = mx.ones((1, 8), dtype=mx.bfloat16)
    with attention_phase(phase):
        got = route(routed, weights, shared)
    assert tuple(got.shape) == (1, 8)
    assert bool(mx.all(got == 1))


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
    assert '_REQUIRED_MLX_VERSION = "0.31.2"' in source
    assert "d7bd29fc20b4a08318d21161c3dfb340889cc9454c5e554ad749eb0127cfa2d6" in source
    assert "ee94397faa812c91d5f1a0ee17c5bb6ca6032883653591dd33d4cfddb737ac33" in source
    assert "smoke-2bitdq-20260731-prompt2.txt" in source
    assert "_REQUIRED_PROMPT_TOKENS = 328" in source
    assert "hashlib.sha256" in source
    assert "mx.eval(out)" in source
    assert "mx.synchronize()" in source
    assert "exact_parity" in source
    assert "promotion" not in source.lower()
