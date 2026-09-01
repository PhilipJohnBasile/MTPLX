"""Construction-bound physical-M4 stock-QMM combine route for Flash-Next."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .kernels.qwen4_m4_stage3 import bind, bind_residual_tail
from .kernels.qwen4_m4_routed_down import bind as bind_routed_down_reduce
from .models.qwen4_exp import (
    SparseMoeBlock,
    _FusedGateUpMLP,
    _FusedGateUpSwitchGLU,
)
from .runtime_options import env_bool


def qwen4_m4_stage3_enabled() -> bool:
    return env_bool("MTPLX_QWEN4_M4_STAGE3", default=False)


def qwen4_m4_routed_down_reduce_enabled() -> bool:
    return env_bool("MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE", default=False)


def _text_model(runtime: Any):
    return getattr(runtime.model, "language_model", runtime.model)


def _m4_forward(block: SparseMoeBlock, x: mx.array, stage3) -> mx.array:
    gates = mx.softmax(block.gate(x), axis=-1, precise=True)
    expert_ids = mx.argpartition(gates, kth=-block.top_k, axis=-1)[
        ..., -block.top_k :
    ]
    route_scores = mx.take_along_axis(gates, expert_ids, axis=-1)
    if block.norm_topk_prob:
        route_scores = route_scores / route_scores.sum(axis=-1, keepdims=True)

    routed = block.switch_mlp
    routed_input = mx.expand_dims(x, (-2, -3))
    routed_gate, routed_up = routed._gu(
        routed_input,
        expert_ids,
        sorted_indices=False,
    )
    routed_h = nn.silu(routed_gate) * routed_up
    routed_down = routed.down_proj(
        routed_h,
        expert_ids,
        sorted_indices=False,
    ).squeeze(-2).reshape(4, 10, 2560)

    shared = block.shared_expert
    shared_gu = mx.quantized_matmul(
        x,
        shared.gu_weight,
        shared.gu_scales,
        shared.gu_biases,
        transpose=True,
        group_size=shared.group_size,
        bits=shared.bits,
        mode=shared.mode,
    )
    shared_gate, shared_up = mx.split(shared_gu, 2, axis=-1)
    shared_h = nn.silu(shared_gate) * shared_up
    shared_down = shared.down_proj(shared_h).reshape(4, 2560)
    shared_factor = mx.sigmoid(block.shared_expert_gate(x)).reshape(4)

    output = stage3(
        routed_down,
        shared_down,
        route_scores.reshape(4, 10).astype(mx.bfloat16),
        shared_factor.astype(mx.bfloat16),
    )
    return output.reshape(x.shape).astype(x.dtype)


def _m4_routed_down_reduce_forward(
    block: SparseMoeBlock,
    x: mx.array,
    routed_down_reduce,
) -> mx.array:
    gates = mx.softmax(block.gate(x), axis=-1, precise=True)
    expert_ids = mx.argpartition(gates, kth=-block.top_k, axis=-1)[
        ..., -block.top_k :
    ]
    route_scores = mx.take_along_axis(gates, expert_ids, axis=-1)
    if block.norm_topk_prob:
        route_scores = route_scores / route_scores.sum(axis=-1, keepdims=True)

    routed = block.switch_mlp
    routed_input = mx.expand_dims(x, (-2, -3))
    routed_gate, routed_up = routed._gu(
        routed_input,
        expert_ids,
        sorted_indices=False,
    )
    routed_h = (nn.silu(routed_gate) * routed_up).reshape(4, 10, 640)

    shared = block.shared_expert
    shared_gu = mx.quantized_matmul(
        x,
        shared.gu_weight,
        shared.gu_scales,
        shared.gu_biases,
        transpose=True,
        group_size=shared.group_size,
        bits=shared.bits,
        mode=shared.mode,
    )
    shared_gate, shared_up = mx.split(shared_gu, 2, axis=-1)
    shared_h = nn.silu(shared_gate) * shared_up
    shared_down = shared.down_proj(shared_h).reshape(4, 2560)
    shared_factor = mx.sigmoid(block.shared_expert_gate(x)).reshape(4)

    output = routed_down_reduce(
        routed_h,
        routed.down_proj.weight,
        routed.down_proj.scales,
        routed.down_proj.biases,
        expert_ids.reshape(4, 10),
        route_scores.reshape(4, 10).astype(mx.bfloat16),
        shared_down,
        shared_factor.astype(mx.bfloat16),
    )
    return output.reshape(x.shape).astype(x.dtype)


class _M4Stage3SparseMoeBlock(SparseMoeBlock):
    def __call__(self, x: mx.array) -> mx.array:
        rows = int(x.size // x.shape[-1])
        if rows == 4:
            return _m4_forward(self, x, self._mtplx_m4_stage3)
        return super().__call__(x)


class _M4RoutedDownReduceSparseMoeBlock(SparseMoeBlock):
    def __call__(self, x: mx.array) -> mx.array:
        rows = int(x.size // x.shape[-1])
        if rows == 4:
            return _m4_routed_down_reduce_forward(
                self,
                x,
                self._mtplx_m4_routed_down_reduce,
            )
        return super().__call__(x)


def _projection_contract(projection: Any) -> tuple[Any, ...]:
    return (
        int(projection.bits),
        int(projection.group_size),
        str(projection.mode),
        projection.weight.dtype,
        projection.scales.dtype,
        projection.biases.dtype,
        tuple(projection.weight.shape),
        tuple(projection.scales.shape),
        tuple(projection.biases.shape),
    )


def _fused_gu_contract(owner: Any) -> tuple[Any, ...]:
    return (
        int(owner.bits),
        int(owner.group_size),
        str(owner.mode),
        owner.gu_weight.dtype,
        owner.gu_scales.dtype,
        owner.gu_biases.dtype,
        tuple(owner.gu_weight.shape),
        tuple(owner.gu_scales.shape),
        tuple(owner.gu_biases.shape),
    )


_ROUTER_CONTRACT = (
    8,
    64,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (512, 640),
    (512, 40),
    (512, 40),
)
_SHARED_GATE_CONTRACT = (
    8,
    64,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (1, 640),
    (1, 40),
    (1, 40),
)
_ROUTED_GU_CONTRACT = (
    4,
    32,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (512, 1280, 320),
    (512, 1280, 80),
    (512, 1280, 80),
)
_SHARED_GU_CONTRACT = (
    8,
    64,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (1280, 640),
    (1280, 40),
    (1280, 40),
)
_ROUTED_DOWN_CONTRACT = (
    4,
    32,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (512, 2560, 80),
    (512, 2560, 20),
    (512, 2560, 20),
)
_SHARED_DOWN_CONTRACT = (
    8,
    64,
    "affine",
    mx.uint32,
    mx.bfloat16,
    mx.bfloat16,
    (2560, 160),
    (2560, 10),
    (2560, 10),
)


def _validate_block_contract(block: Any, *, index: int) -> None:
    label = f"qwen4 M4 stage3 layer {index}"
    if type(block) is not SparseMoeBlock:
        raise ValueError(f"{label} has wrong MoE owner")
    if int(block.num_experts) != 512:
        raise ValueError(f"{label} requires 512 experts")
    if int(block.top_k) != 10:
        raise ValueError(f"{label} requires exact top-k 10")
    if not bool(block.norm_topk_prob):
        raise ValueError(f"{label} requires top-k probability normalization")
    if block.sharding_group is not None:
        raise ValueError(f"{label} does not support sharding")
    if type(block.switch_mlp) is not _FusedGateUpSwitchGLU:
        raise ValueError(f"{label} lacks exact routed fused GU owner")
    if type(block.shared_expert) is not _FusedGateUpMLP:
        raise ValueError(f"{label} lacks exact shared fused GU owner")
    if _projection_contract(block.gate) != _ROUTER_CONTRACT:
        raise ValueError(f"{label} router mismatch")
    if _projection_contract(block.shared_expert_gate) != _SHARED_GATE_CONTRACT:
        raise ValueError(f"{label} shared gate mismatch")
    if _fused_gu_contract(block.switch_mlp) != _ROUTED_GU_CONTRACT:
        raise ValueError(f"{label} routed fused GU mismatch")
    if _fused_gu_contract(block.shared_expert) != _SHARED_GU_CONTRACT:
        raise ValueError(f"{label} shared fused GU mismatch")
    if _projection_contract(block.switch_mlp.down_proj) != _ROUTED_DOWN_CONTRACT:
        raise ValueError(f"{label} routed down mismatch")
    if _projection_contract(block.shared_expert.down_proj) != _SHARED_DOWN_CONTRACT:
        raise ValueError(f"{label} shared down mismatch")


def _validate_input_contract(layer: Any, *, index: int) -> None:
    norm = getattr(getattr(layer, "mlp_hyper_connection", None), "hc_norm", None)
    weight = getattr(norm, "weight", None)
    if (
        weight is None
        or tuple(weight.shape) != (4 * 2560,)
        or weight.dtype != mx.bfloat16
    ):
        raise ValueError(
            f"qwen4 M4 stage3 layer {index} requires BF16 hidden input ownership"
        )


def install_qwen4_m4_stage3(runtime: Any) -> dict[str, Any]:
    """Validate every owner, self-check the kernel, then install M4 directly."""

    text = _text_model(runtime)
    layers = tuple(text.model.layers)
    if len(layers) != 48:
        raise ValueError(f"qwen4 M4 stage3 requires 48 layers, got {len(layers)}")

    plans = []
    routed_down_reduce_enabled = qwen4_m4_routed_down_reduce_enabled()
    stage3 = bind()
    routed_down_reduce = (
        bind_routed_down_reduce() if routed_down_reduce_enabled else None
    )
    for index, layer in enumerate(layers):
        block = layer.mlp
        _validate_input_contract(layer, index=index)
        _validate_block_contract(block, index=index)
        plans.append((block, stage3, routed_down_reduce))

    sample = mx.sin(mx.arange(4 * 2560, dtype=mx.float32) * 0.0009765625)
    sample = sample.reshape(1, 4, 2560).astype(mx.bfloat16)
    max_delta = 0.0
    for index, (block, stage3, routed_down_reduce) in enumerate(plans):
        stock = block(sample)
        candidate = (
            _m4_routed_down_reduce_forward(block, sample, routed_down_reduce)
            if routed_down_reduce_enabled
            else _m4_forward(block, sample, stage3)
        )
        if routed_down_reduce_enabled:
            reference = _m4_forward(block, sample, stage3)
            exact = mx.array_equal(reference, candidate)
            mx.eval(exact)
            if not bool(exact.item()):
                raise ValueError(
                    f"qwen4 M4 stage3 layer {index} routed-down "
                    "reduction self-check failed"
                )
        delta = mx.max(mx.abs(stock.astype(mx.float32) - candidate.astype(mx.float32)))
        finite = mx.all(mx.isfinite(candidate))
        mx.eval(delta, finite)
        observed = float(delta.item())
        if not bool(finite.item()) or observed > 0.001953125:
            raise ValueError(
                f"qwen4 M4 stage3 layer {index} self-check failed: dmax={observed}"
            )
        max_delta = max(max_delta, observed)

    for block, stage3, routed_down_reduce in plans:
        block._mtplx_m4_stage3 = stage3
        if routed_down_reduce_enabled:
            block._mtplx_m4_routed_down_reduce = routed_down_reduce
            block.__class__ = _M4RoutedDownReduceSparseMoeBlock
        else:
            block.__class__ = _M4Stage3SparseMoeBlock
    report = {
        "installed": True,
        "layers": len(plans),
        "rows": 4,
        "max_abs_diff": max_delta,
        "boundary": "stock_qmm_combine_tail",
        "routed": "stock_q4/g32",
        "shared": "stock_q8/g64",
        "routed_down_reduce": routed_down_reduce_enabled,
    }
    runtime.qwen4_m4_stage3_report = report
    return report


__all__ = [
    "install_qwen4_m4_stage3",
    "qwen4_m4_routed_down_reduce_enabled",
    "qwen4_m4_stage3_enabled",
]
