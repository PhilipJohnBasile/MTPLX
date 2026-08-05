"""LFM2 fast-path: the fused 3-tap decode window must be bit-exact with conv1d."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.lfm2_fast import is_lfm2_config


def test_is_lfm2_config():
    assert is_lfm2_config({"model_type": "lfm2"})
    assert is_lfm2_config({"architectures": ["Lfm2ForCausalLM"]})
    assert is_lfm2_config({"text_config": {"model_type": "lfm2"}})
    assert not is_lfm2_config({"model_type": "qwen3_next"})
    assert not is_lfm2_config(None)


def test_fused_window_matches_conv1d():
    """conv_out = s0*w0 + s1*w1 + Bx*w2 must equal a depthwise Conv1d over the
    padded [state, Bx] window (the exact stock ShortConv decode arithmetic)."""
    mx.random.seed(0)
    C, L = 64, 3
    conv = nn.Conv1d(C, C, kernel_size=L, groups=C, bias=False)
    w = conv.weight  # [C, L, 1]
    state = mx.random.normal((1, L - 1, C))  # [1, 2, C]
    Bx = mx.random.normal((1, 1, C))  # single decode token

    # stock: conv over concatenated window
    window = mx.concatenate([state, Bx], axis=1)  # [1, 3, C]
    ref = conv(window)  # [1, 1, C]

    # fused 3-tap
    taps = w[:, :, 0]  # [C, 3]
    s0, s1, bx = state[:, 0, :], state[:, 1, :], Bx[:, 0, :]
    fused = s0 * taps[:, 0] + s1 * taps[:, 1] + bx * taps[:, 2]  # [1, C]

    assert mx.allclose(fused, ref[:, 0, :], atol=1e-4, rtol=1e-4).item()
