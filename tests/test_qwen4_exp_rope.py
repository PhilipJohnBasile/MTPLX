"""Static-YaRN contract tests for Qwen4-Exp's million-token extension."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from mtplx.models.qwen4_exp import (
    TextArgs,
    _apply_partial_rope,
    _rope_cos_sin,
    _rope_inv_freq_and_scaling,
)


def _args(factor: float) -> TextArgs:
    return TextArgs(
        max_position_embeddings=int(262_144 * factor),
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
            "factor": factor,
            "original_max_position_embeddings": 262_144,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
    )


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_static_yarn_matches_qwen_frequency_ramp_and_mscale(factor: float):
    args = _args(factor)
    inv_freq, attention_scaling = _rope_inv_freq_and_scaling(args)
    mx.eval(inv_freq)

    assert tuple(inv_freq.shape) == (32,)
    assert attention_scaling == pytest.approx(1.0 + 0.1 * math.log(factor))

    base = 10_000_000.0
    default = [1.0 / (base ** ((2 * index) / 64)) for index in range(32)]
    values = inv_freq.tolist()
    # For this exact Qwen geometry, Transformers' truncated correction range
    # is [14,22]. Frequencies before it extrapolate unchanged; frequencies
    # after it interpolate by the full static factor.
    assert values[0] == pytest.approx(default[0], rel=1e-6)
    assert values[14] == pytest.approx(default[14], rel=1e-6)
    assert values[18] == pytest.approx(
        0.5 * default[18] + 0.5 * default[18] / factor,
        rel=1e-6,
    )
    assert values[22] == pytest.approx(default[22] / factor, rel=1e-6)
    assert values[31] == pytest.approx(default[31] / factor, rel=1e-6)


@pytest.mark.parametrize("factor,position", [(2.0, 524_287), (4.0, 1_048_575)])
def test_static_yarn_scales_only_the_rotary_prefix(
    factor: float,
    position: int,
):
    args = _args(factor)
    inv_freq, attention_scaling = _rope_inv_freq_and_scaling(args)
    positions = mx.array([position], dtype=mx.int32)
    cosine, sine = _rope_cos_sin(positions, inv_freq, attention_scaling)
    values = mx.arange(256, dtype=mx.float32).reshape(1, 1, 1, 256) / 257.0
    actual = _apply_partial_rope(values, cosine, sine)
    unscaled_cosine, unscaled_sine = _rope_cos_sin(positions, inv_freq)
    unscaled = _apply_partial_rope(values, unscaled_cosine, unscaled_sine)
    mx.eval(actual, unscaled)

    rotary_dim = args.rotary_dim
    assert bool(
        mx.allclose(
            actual[..., :rotary_dim],
            unscaled[..., :rotary_dim] * attention_scaling,
            rtol=1e-5,
            atol=1e-6,
        ).item()
    )
    assert bool(
        mx.array_equal(
            actual[..., rotary_dim:],
            values[..., rotary_dim:],
        ).item()
    )


def test_unknown_qwen_rope_type_fails_closed():
    args = TextArgs(
        rope_parameters={
            "rope_type": "not-a-real-rope",
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
        }
    )
    with pytest.raises(ValueError, match="supports rope_type"):
        _rope_inv_freq_and_scaling(args)

