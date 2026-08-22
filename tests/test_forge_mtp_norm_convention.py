"""Set-level MTP norm convention decision (#301).

The forge extraction must decide delta-vs-absolute ONCE per tensor set and
apply it uniformly: the old per-tensor always-shift tier double-shifted
three norms of absolute sources (q/k means ~1.79 -> ~2.79), producing 0-2%
acceptance. Pure tensors, no model load.
"""

from __future__ import annotations

import mlx.core as mx

from mtplx.compressed_tensors import (
    mtp_norms_are_delta_encoded,
    sanitize_plain_weight,
)

_NORM_KEYS = (
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.k_norm.weight",
    "norm.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.post_attention_layernorm.weight",
    "pre_fc_norm_hidden.weight",
    "pre_fc_norm_embedding.weight",
)


def _norm_set(level: float) -> dict[str, mx.array]:
    tensors = {key: mx.full((64,), level) for key in _NORM_KEYS}
    # q/k in absolute packs sit higher than the low set; model the fleet shape.
    tensors["layers.0.self_attn.q_norm.weight"] = mx.full((64,), level + 0.76)
    tensors["layers.0.self_attn.k_norm.weight"] = mx.full((64,), level + 0.76)
    tensors["some.weight"] = mx.zeros((8, 8))  # 2-D bystander
    return tensors


def test_absolute_set_detected_and_left_bit_identical():
    absolute = _norm_set(1.03)  # q/k ~1.79, low set ~1.03 (healthy shipped shape)
    assert mtp_norms_are_delta_encoded(absolute) is False
    for key, value in absolute.items():
        out = sanitize_plain_weight(f"mtp.{key}", value, mtp_norm_shift=False)
        assert bool(mx.all(out == value).item()), f"absolute tensor shifted: {key}"


def test_delta_set_detected_and_all_norms_shifted():
    delta = _norm_set(0.03)  # q/k ~0.79, low set ~0.03 (raw export shape)
    assert mtp_norms_are_delta_encoded(delta) is True
    for key in _NORM_KEYS:
        out = sanitize_plain_weight(f"mtp.{key}", delta[key], mtp_norm_shift=True)
        expected = delta[key] + 1.0
        assert bool(mx.all(out == expected).item()), f"delta norm not shifted: {key}"


def test_heuristic_none_keeps_historical_behavior():
    delta = _norm_set(0.03)
    q = delta["layers.0.self_attn.q_norm.weight"]
    assert bool(
        mx.all(
            sanitize_plain_weight("mtp.layers.0.self_attn.q_norm.weight", q)
            == q + 1.0
        ).item()
    )


def test_empty_or_partial_sets_never_shift():
    assert mtp_norms_are_delta_encoded({}) is False
    assert mtp_norms_are_delta_encoded({"some.weight": mx.zeros((8, 8))}) is False
