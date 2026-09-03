"""Guarded Metal parity for the fixed-M4 paired routed-GU producer."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mtplx.kernels.qwen4_m4_routed_down import bind_residual_tail
from mtplx.kernels.qwen4_m4_routed_glu import bind as bind_routed_glu


def _u32_pack(shape):
    size = 1
    for extent in shape:
        size *= extent
    values = mx.arange(size, dtype=mx.uint32)
    return (values * mx.array(2654435761, dtype=mx.uint32)).reshape(shape)


def _metadata(shape, *, offset: int):
    size = 1
    for extent in shape:
        size *= extent
    values = mx.arange(size, dtype=mx.float32)
    return (
        mx.sin((values + float(offset)) * 0.03125) * 0.00390625
    ).reshape(shape).astype(mx.bfloat16)


def test_paired_routed_glu_is_bit_exact_at_physical_m4_shape() -> None:
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal parity requires the guarded GPU lane")

    experts = 2
    value = mx.sin(mx.arange(4 * 2560, dtype=mx.float32) * 0.001953125)
    value = value.reshape(1, 4, 2560).astype(mx.bfloat16)
    weights = _u32_pack((experts, 1280, 320))
    scales = _metadata((experts, 1280, 80), offset=3)
    biases = _metadata((experts, 1280, 80), offset=11)
    expert_ids = (mx.arange(4 * 10, dtype=mx.uint32) % experts).reshape(1, 4, 10)

    routed_input = mx.expand_dims(value, (-2, -3))
    gu = mx.gather_qmm(
        routed_input,
        weights,
        scales,
        biases,
        rhs_indices=expert_ids,
        transpose=True,
        group_size=32,
        bits=4,
        mode="affine",
        sorted_indices=False,
    )
    gate, up = mx.split(gu, 2, axis=-1)
    expected_h = (nn.silu(gate) * up).reshape(4, 10, 640)
    actual_h = bind_routed_glu()(
        value.reshape(4, 2560),
        weights,
        scales,
        biases,
        expert_ids.reshape(4, 10),
    )
    mx.eval(expected_h, actual_h)
    assert bool(mx.array_equal(expected_h, actual_h).item())

    down_weights = _u32_pack((experts, 2560, 80))
    down_scales = _metadata((experts, 2560, 20), offset=17)
    down_biases = _metadata((experts, 2560, 20), offset=23)
    route_logits = mx.cos(mx.arange(40, dtype=mx.float32) * 0.125).reshape(4, 10)
    route_scores = mx.softmax(route_logits, axis=-1, precise=True).astype(mx.bfloat16)
    shared_down = mx.sin(mx.arange(4 * 2560, dtype=mx.float32) * 0.00390625)
    shared_down = shared_down.reshape(4, 2560).astype(mx.bfloat16)
    shared_factor = mx.sigmoid(mx.arange(4, dtype=mx.float32)).astype(mx.bfloat16)
    hyper = mx.cos(mx.arange(4 * 4 * 2560, dtype=mx.float32) * 0.0009765625)
    hyper = hyper.reshape(4, 4 * 2560).astype(mx.bfloat16)
    inject = mx.sigmoid(mx.arange(16, dtype=mx.float32) * 0.0625)
    inject = inject.reshape(4, 4).astype(mx.bfloat16)
    tail = bind_residual_tail()
    expected = tail(
        expected_h,
        down_weights,
        down_scales,
        down_biases,
        expert_ids.reshape(4, 10),
        route_scores,
        shared_down,
        shared_factor,
        hyper,
        inject,
    )
    actual = tail(
        actual_h,
        down_weights,
        down_scales,
        down_biases,
        expert_ids.reshape(4, 10),
        route_scores,
        shared_down,
        shared_factor,
        hyper,
        inject,
    )
    mx.eval(expected, actual)
    assert bool(mx.array_equal(expected, actual).item())
