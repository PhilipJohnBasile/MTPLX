"""MTP norm +1.0 convention is a whole-sidecar decision, never per-tensor (#301).

Synthetic gains mirror the measured fleet (2026-08-24): delta exports carry q/k
means near 0.75 and low-set means at 0.30-0.39; absolute sidecars sit at 1.73+
and 0.87+. The final norm overlaps across conventions (raw-delta 4B mean 2.58
vs absolute 3.8 mean 2.25) so it must follow the ensemble verdict.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mtplx.compressed_tensors import (
    mtp_sidecar_norms_are_delta,
    sanitize_plain_weight,
    shift_delta_mtp_norms,
)
from mtplx.mtp_patch import _heal_raw_delta_mtp_norms


def _gain(mean: float, size: int = 8) -> mx.array:
    return mx.array(np.full(size, mean, dtype=np.float32))


def _absolute_sidecar(prefix: str = "mtp.") -> dict[str, mx.array]:
    return {
        f"{prefix}layers.0.self_attn.q_norm.weight": _gain(1.79),
        f"{prefix}layers.0.self_attn.k_norm.weight": _gain(1.78),
        f"{prefix}layers.0.input_layernorm.weight": _gain(1.04),
        f"{prefix}layers.0.post_attention_layernorm.weight": _gain(1.21),
        f"{prefix}norm.weight" if prefix else "norm.weight": _gain(2.25),
        f"{prefix}layers.0.self_attn.q_proj.weight": mx.zeros((8, 8)),
    }


def _delta_sidecar(prefix: str = "mtp.") -> dict[str, mx.array]:
    return {
        f"{prefix}layers.0.self_attn.q_norm.weight": _gain(0.746),
        f"{prefix}layers.0.self_attn.k_norm.weight": _gain(0.734),
        f"{prefix}layers.0.input_layernorm.weight": _gain(0.303),
        f"{prefix}layers.0.post_attention_layernorm.weight": _gain(0.387),
        f"{prefix}norm.weight" if prefix else "norm.weight": _gain(2.58),
        f"{prefix}layers.0.self_attn.q_proj.weight": mx.zeros((8, 8)),
    }


def test_absolute_sidecar_passes_through_untouched():
    weights = _absolute_sidecar()
    assert mtp_sidecar_norms_are_delta(weights) is False
    shifted = shift_delta_mtp_norms(weights)
    for key, value in weights.items():
        assert bool(mx.array_equal(shifted[key], value)), key


def test_delta_sidecar_shifts_all_seven_norm_gains():
    weights = _delta_sidecar()
    assert mtp_sidecar_norms_are_delta(weights) is True
    shifted = shift_delta_mtp_norms(weights)
    for key, value in weights.items():
        if key.endswith("proj.weight"):
            assert bool(mx.array_equal(shifted[key], value))
        else:
            assert bool(mx.array_equal(shifted[key], value + 1.0)), key
    # The final norm follows the ensemble verdict despite its high mean.
    assert float(shifted["mtp.norm.weight"].mean().item()) > 3.5


def test_stripped_key_namespace_matches_runtime_heal_shape():
    # The runtime loader strips the "mtp." namespace before healing.
    healed = _heal_raw_delta_mtp_norms(_delta_sidecar(prefix=""))
    assert float(healed["norm.weight"].mean().item()) > 3.5
    assert float(healed["layers.0.input_layernorm.weight"].mean().item()) > 1.2
    untouched = _heal_raw_delta_mtp_norms(_absolute_sidecar(prefix=""))
    assert float(untouched["norm.weight"].mean().item()) < 2.5


def test_missing_family_never_shifts():
    weights = _delta_sidecar()
    weights.pop("mtp.layers.0.self_attn.q_norm.weight")
    weights.pop("mtp.layers.0.self_attn.k_norm.weight")
    assert mtp_sidecar_norms_are_delta(weights) is False
    shifted = shift_delta_mtp_norms(weights)
    assert bool(
        mx.array_equal(shifted["mtp.norm.weight"], weights["mtp.norm.weight"])
    )


def test_sanitize_plain_weight_no_longer_shifts_mtp_norms():
    value = _gain(1.79)
    out = sanitize_plain_weight("mtp.layers.0.self_attn.q_norm.weight", value)
    assert bool(mx.array_equal(out, value))
    # Trunk suffixes keep their per-tensor shift.
    trunk = sanitize_plain_weight("model.layers.0.input_layernorm.weight", _gain(0.3))
    assert float(trunk.mean().item()) > 1.2
