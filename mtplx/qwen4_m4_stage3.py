"""Construction-bound physical-M4 stock-QMM combine route for Flash-Next."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .kernels.qwen4_m4_stage3 import bind
from .kernels.qwen4_m4_routed_down import (
    bind as bind_routed_down_reduce,
    bind_residual_tail,
)
from .models.qwen4_exp import (
    DecoderLayer,
    SparseMoeBlock,
    _FusedGateUpMLP,
    _FusedGateUpSwitchGLU,
)
from .runtime_options import env_bool


def qwen4_m4_stage3_enabled() -> bool:
    return env_bool("MTPLX_QWEN4_M4_STAGE3", default=False)


def qwen4_m4_routed_down_reduce_enabled() -> bool:
    return env_bool("MTPLX_QWEN4_M4_ROUTED_DOWN_REDUCE", default=False)


def qwen4_m4_routed_down_residual_tail_enabled() -> bool:
    return env_bool("MTPLX_QWEN4_M4_ROUTED_DOWN_RESIDUAL_TAIL", default=False)


def qwen4_m4_stage3_flags() -> tuple[bool, bool, bool]:
    """Capture and validate the complete construction-time feature route."""

    stage3_enabled = qwen4_m4_stage3_enabled()
    routed_down_reduce_enabled = qwen4_m4_routed_down_reduce_enabled()
    routed_down_residual_tail_enabled = (
        qwen4_m4_routed_down_residual_tail_enabled()
    )
    if not stage3_enabled and (
        routed_down_reduce_enabled or routed_down_residual_tail_enabled
    ):
        raise ValueError("qwen4 M4 child routes require M4 stage3")
    _validate_feature_combination(
        routed_down_reduce_enabled=routed_down_reduce_enabled,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
    )
    return (
        stage3_enabled,
        routed_down_reduce_enabled,
        routed_down_residual_tail_enabled,
    )


def _validate_feature_combination(
    *,
    routed_down_reduce_enabled: bool,
    routed_down_residual_tail_enabled: bool,
) -> None:
    if routed_down_residual_tail_enabled and not routed_down_reduce_enabled:
        raise ValueError(
            "qwen4 M4 routed-down residual tail requires routed-down reduction"
        )


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


def _m4_routed_down_residual_tail_forward(
    block: SparseMoeBlock,
    x: mx.array,
    routed_down_residual_tail,
    hyper: mx.array,
    inject: mx.array,
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

    return routed_down_residual_tail(
        routed_h,
        routed.down_proj.weight,
        routed.down_proj.scales,
        routed.down_proj.biases,
        expert_ids.reshape(4, 10),
        route_scores.reshape(4, 10).astype(mx.bfloat16),
        shared_down,
        shared_factor.astype(mx.bfloat16),
        hyper,
        inject,
    )


def _m4_routed_down_residual_tail_layer_forward(
    layer: DecoderLayer,
    hidden: mx.array,
    *,
    input_ids: mx.array,
    ssm_mask: mx.array,
    cache: Any,
) -> mx.array:
    if "ple" in layer:
        hidden = hidden + layer.ple(hidden, input_ids, cache)

    mixed, hyper, inject = layer.attn_hyper_connection(hidden)
    if layer.is_linear:
        block_out = layer.linear_attn(mixed, ssm_mask, cache)
    else:
        block_out = layer.self_attn(mixed, cache)
    hidden = hyper + (block_out[..., None, :] * inject[..., :, None]).reshape(
        *hyper.shape
    )

    mixed, hyper, inject = layer.mlp_hyper_connection(hidden)
    return _m4_routed_down_residual_tail_forward(
        layer.mlp,
        mixed,
        layer._mtplx_m4_routed_down_residual_tail,
        hyper,
        inject,
    )


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


class _M4RoutedDownResidualTailDecoderLayer(DecoderLayer):
    def __call__(self, hidden, *, input_ids, ssm_mask, cache):
        rows = int(hidden.size // hidden.shape[-1])
        if rows == 4:
            return _m4_routed_down_residual_tail_layer_forward(
                self,
                hidden,
                input_ids=input_ids,
                ssm_mask=ssm_mask,
                cache=cache,
            )
        return super().__call__(
            hidden,
            input_ids=input_ids,
            ssm_mask=ssm_mask,
            cache=cache,
        )


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


def _validate_layer_owner(layer: Any, *, index: int) -> None:
    if type(layer) is not DecoderLayer:
        raise ValueError(f"qwen4 M4 stage3 layer {index} has wrong decoder owner")


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


def _validate_residual_tail_contract(layer: Any, *, index: int) -> None:
    """Admit only the exact BF16 hyper geometry consumed by the fused store."""

    label = f"qwen4 M4 residual tail layer {index}"
    connection = getattr(layer, "mlp_hyper_connection", None)
    if (
        connection is None
        or int(getattr(connection, "hc_count", -1)) != 4
        or int(getattr(connection, "hidden_size", -1)) != 2560
    ):
        raise ValueError(f"{label} requires exactly four hyper streams")
    inject_projection = getattr(connection, "block_inject_weight", None)
    inject_weight = getattr(inject_projection, "weight", None)
    if (
        inject_weight is None
        or tuple(inject_weight.shape) != (4, 4 * 2560)
        or inject_weight.dtype != mx.bfloat16
    ):
        raise ValueError(f"{label} requires BF16 inject ownership")


def bind_qwen4_m4_residual_tail(layer: Any, *, index: int):
    """Validate and bind the optional M4 residual tail at construction."""

    _validate_input_contract(layer, index=index)
    _validate_residual_tail_contract(layer, index=index)
    _validate_block_contract(layer.mlp, index=index)
    return bind_residual_tail()


def _build_install_plans(
    layers,
    *,
    stage3,
    routed_down_reduce,
    routed_down_residual_tail_enabled: bool,
):
    plans = []
    for index, layer in enumerate(layers):
        block = layer.mlp
        if routed_down_residual_tail_enabled:
            _validate_layer_owner(layer, index=index)
        _validate_input_contract(layer, index=index)
        _validate_block_contract(block, index=index)
        if routed_down_residual_tail_enabled:
            _validate_residual_tail_contract(layer, index=index)
        plans.append((layer, block, stage3, routed_down_reduce))
    return tuple(plans)


def _installation_report(
    *,
    layer_count: int,
    max_delta: float,
    routed_down_reduce_enabled: bool,
    routed_down_residual_tail_enabled: bool,
) -> dict[str, Any]:
    return {
        "installed": True,
        "layers": layer_count,
        "rows": 4,
        "max_abs_diff": max_delta,
        "boundary": (
            "routed_q4g32_reduce_shared_add_mlp_residual"
            if routed_down_residual_tail_enabled
            else "stock_qmm_combine_tail"
        ),
        "reference_boundary": (
            "retained_m4_routed_down_then_stock_residual"
            if routed_down_residual_tail_enabled
            else "stock_sparse_moe_block"
        ),
        "routed": "stock_q4/g32",
        "shared": "stock_q8/g64",
        "routed_down_reduce": routed_down_reduce_enabled,
        "routed_down_residual_tail": routed_down_residual_tail_enabled,
        "exact_layers": layer_count,
        "combined_residual_tail_layers": (
            layer_count if routed_down_residual_tail_enabled else 0
        ),
    }


def install_qwen4_m4_stage3(
    runtime: Any,
    *,
    routed_down_reduce_enabled: bool,
    routed_down_residual_tail_enabled: bool,
) -> dict[str, Any]:
    """Validate every owner, self-check the kernel, then install M4 directly."""

    text = _text_model(runtime)
    layers = tuple(text.model.layers)
    if len(layers) != 48:
        raise ValueError(f"qwen4 M4 stage3 requires 48 layers, got {len(layers)}")

    _validate_feature_combination(
        routed_down_reduce_enabled=routed_down_reduce_enabled,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
    )
    validated_plans = _build_install_plans(
        layers,
        stage3=None,
        routed_down_reduce=None,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
    )
    stage3 = bind()
    routed_down_reduce = (
        bind_residual_tail()
        if routed_down_residual_tail_enabled
        else bind_routed_down_reduce()
        if routed_down_reduce_enabled
        else None
    )
    plans = tuple(
        (layer, block, stage3, routed_down_reduce)
        for layer, block, _, _ in validated_plans
    )

    sample = mx.sin(mx.arange(4 * 2560, dtype=mx.float32) * 0.0009765625)
    sample = sample.reshape(1, 4, 2560).astype(mx.bfloat16)
    max_delta = 0.0
    for index, (layer, block, stage3, routed_down_reduce) in enumerate(plans):
        stock = None if routed_down_residual_tail_enabled else block(sample)
        reference = _m4_forward(block, sample, stage3)
        if routed_down_residual_tail_enabled:
            hyper = mx.sin(
                mx.arange(4 * 4 * 2560, dtype=mx.float32) * 0.000244140625
            ).reshape(1, 4, 4 * 2560).astype(mx.bfloat16)
            inject = mx.cos(
                mx.arange(4 * 4, dtype=mx.float32) * 0.0625
            ).reshape(1, 4, 4).astype(mx.bfloat16)
            candidate = _m4_routed_down_residual_tail_forward(
                block,
                sample,
                routed_down_reduce,
                hyper,
                inject,
            )
            reference = hyper + (
                reference[..., None, :] * inject[..., :, None]
            ).reshape(*hyper.shape)
        elif routed_down_reduce_enabled:
            candidate = _m4_routed_down_reduce_forward(
                block, sample, routed_down_reduce
            )
        else:
            candidate = reference
        if routed_down_reduce_enabled:
            exact = mx.array_equal(reference, candidate)
            mx.eval(exact)
            if not bool(exact.item()):
                raise ValueError(
                    f"qwen4 M4 stage3 layer {index} routed-down "
                    "reduction self-check failed"
                )
        comparison = reference if routed_down_residual_tail_enabled else stock
        delta = mx.max(
            mx.abs(comparison.astype(mx.float32) - candidate.astype(mx.float32))
        )
        finite = mx.all(mx.isfinite(candidate))
        mx.eval(delta, finite)
        observed = float(delta.item())
        if not bool(finite.item()) or observed > 0.001953125:
            raise ValueError(
                f"qwen4 M4 stage3 layer {index} self-check failed: dmax={observed}"
            )
        max_delta = max(max_delta, observed)

    for layer, block, stage3, routed_down_reduce in plans:
        block._mtplx_m4_stage3 = stage3
        if routed_down_residual_tail_enabled:
            layer._mtplx_m4_routed_down_residual_tail = routed_down_reduce
            layer.__class__ = _M4RoutedDownResidualTailDecoderLayer
        elif routed_down_reduce_enabled:
            block._mtplx_m4_routed_down_reduce = routed_down_reduce
            block.__class__ = _M4RoutedDownReduceSparseMoeBlock
        else:
            block.__class__ = _M4Stage3SparseMoeBlock
    report = _installation_report(
        layer_count=len(plans),
        max_delta=max_delta,
        routed_down_reduce_enabled=routed_down_reduce_enabled,
        routed_down_residual_tail_enabled=routed_down_residual_tail_enabled,
    )
    runtime.qwen4_m4_stage3_report = report
    return report


__all__ = [
    "install_qwen4_m4_stage3",
    "qwen4_m4_routed_down_reduce_enabled",
    "qwen4_m4_routed_down_residual_tail_enabled",
    "qwen4_m4_stage3_enabled",
    "qwen4_m4_stage3_flags",
]
