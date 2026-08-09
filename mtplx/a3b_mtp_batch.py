"""Construction-time contract for Qwen3.6-35B-A3B eight-row MTP decode."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any

import numpy as np
import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import ArraysCache

from mtplx.artifacts import load_config
from mtplx.ragged_kv_cache import RaggedBatchKVCache
from mtplx.sampling import SamplerConfig


_LAYER_TYPES = tuple(
    "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
    for index in range(40)
)
A3B_MTP_BATCH_MAX_CONTEXT_TOKENS = 131072
# B8 and B1 use different BF16 reduction geometries.  Nine BF16 rounding
# units is the construction-time semantic-parity bound; token decisions and
# cross-row isolation are still required to match exactly.
_BF16_GEOMETRY_RELATIVE_LIMIT = 9.0 / 128.0
_MTP_BATCH_ATTENTION_ACTIVE: ContextVar[bool] = ContextVar(
    "mtplx_qwen35b_mtp_batch_attention_active",
    default=False,
)


class A3BMTPBatchInstallError(RuntimeError):
    """The fixed Qwen 35B MTP batch lane cannot be installed safely."""


class A3BMTPBatchCapacityError(RuntimeError):
    """A cohort exceeds the installed fixed-width KV capacity contract."""


@dataclass(frozen=True)
class A3BMTPBatchGeometry:
    cohort_slots: int = 8
    speculative_depth: int = 1
    verify_tokens: int = 2
    projection_rows: int = 16
    hidden_size: int = 2048
    vocab_size: int = 248320
    num_kv_heads: int = 2
    head_dim: int = 256
    hidden_layers: int = 40
    experts: int = 256
    experts_per_token: int = 8
    body_quant_bits: int = 4
    body_quant_group_size: int = 64
    mtp_quant_bits: int = 4
    mtp_quant_group_size: int = 32
    max_context_tokens: int = A3B_MTP_BATCH_MAX_CONTEXT_TOKENS
    prefill_chunk_tokens: int = 2048
    prefill_cleanup_every: int = 4


@dataclass(frozen=True)
class InstalledA3BMTPBatchLane:
    """Prevalidated, prebound fixed-shape lane used directly by serving."""

    geometry: A3BMTPBatchGeometry
    route_id: str
    attention_route_id: str
    config_fingerprint: str
    target_forward: Callable[..., Any]
    capture_forward: Callable[..., Any]
    draft_forward: Callable[..., Any]
    update_mtp_cache: Callable[..., Any]
    commit_rows: Callable[..., Any]
    prefill_request: Callable[..., Any]
    merge_target_caches: Callable[..., Any]
    merge_mtp_caches: Callable[..., Any]
    make_cache: Callable[..., Any]
    make_mtp_cache: Callable[..., Any]
    selfcheck: Mapping[str, Any]


def _not_cancelled() -> bool:
    return False


@dataclass(frozen=True)
class A3BMTPBatchRequest:
    request_id: str
    prompt_ids: tuple[int, ...]
    sampler: SamplerConfig
    draft_sampler: SamplerConfig
    seed: int
    max_tokens: int
    stop_token_ids: frozenset[int] = frozenset()
    omit_speculative_bonus: bool = False
    on_token: Callable[[int], None] | None = None
    on_decode_start: Callable[[], None] | None = None
    on_terminal: Callable[[str, int], None] | None = None
    cancelled: Callable[[], bool] = _not_cancelled


@dataclass(frozen=True)
class A3BMTPBatchStreamResult:
    request_id: str
    tokens: tuple[int, ...]
    finish_reason: str
    cycles: int = 0
    accepted_drafts: int = 0
    rejected_drafts: int = 0


@dataclass(frozen=True)
class A3BMTPBatchResult:
    streams: tuple[A3BMTPBatchStreamResult, ...]
    cycles: int
    accepted_drafts: int
    rejected_drafts: int
    route_id: str
    width_histogram: Mapping[int, int]


def _fail(name: str, actual: Any, expected: Any) -> None:
    raise A3BMTPBatchInstallError(
        f"Qwen 35B mtp_batch {name} mismatch: expected {expected!r}, got {actual!r}"
    )


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        _fail(name, actual, expected)


def _require_callable(runtime: Any, name: str) -> Callable[..., Any]:
    value = getattr(runtime, name, None)
    if not callable(value):
        raise A3BMTPBatchInstallError(
            f"Qwen 35B mtp_batch requires callable runtime.{name}"
        )
    return value


def _model_layers(runtime: Any) -> tuple[list[Any], list[Any]]:
    model = getattr(runtime, "model", None)
    language_model = getattr(model, "language_model", None)
    trunk = getattr(language_model, "model", None)
    trunk_layers = getattr(trunk, "layers", None)
    mtp = getattr(model, "mtp", None)
    mtp_layers = getattr(mtp, "layers", None)
    if not isinstance(trunk_layers, (list, tuple)):
        raise A3BMTPBatchInstallError(
            "Qwen 35B mtp_batch requires constructed trunk layers"
        )
    if not isinstance(mtp_layers, (list, tuple)):
        raise A3BMTPBatchInstallError(
            "Qwen 35B mtp_batch requires constructed MTP layers"
        )
    return list(trunk_layers), list(mtp_layers)


def _validate_config(runtime: Any) -> tuple[dict[str, Any], str]:
    config = load_config(runtime.model_path)
    text = config.get("text_config")
    body_quant = config.get("quantization")
    mtp_quant = config.get("mtplx_mtp_quantization")
    if not isinstance(text, dict):
        raise A3BMTPBatchInstallError("Qwen 35B mtp_batch requires text_config")
    if not isinstance(body_quant, dict):
        raise A3BMTPBatchInstallError("Qwen 35B mtp_batch requires body quantization")
    if not isinstance(mtp_quant, dict):
        raise A3BMTPBatchInstallError("Qwen 35B mtp_batch requires MTP quantization")

    expected = {
        "model_type": "qwen3_5_moe",
        "architecture": ["Qwen3_5MoeForConditionalGeneration"],
        "text model_type": "qwen3_5_moe_text",
        "dtype": "bfloat16",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "layer_types": list(_LAYER_TYPES),
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
        "body bits": 4,
        "body group_size": 64,
        "body mode": "affine",
        "MTP bits": 4,
        "MTP group_size": 32,
        "MTP mode": "affine",
        "MTP policy": "prequantized-int4",
        "MTP prequantized": True,
    }
    actual = {
        "model_type": config.get("model_type"),
        "architecture": config.get("architectures"),
        "text model_type": text.get("model_type"),
        "dtype": text.get("dtype"),
        "hidden_size": text.get("hidden_size"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "layer_types": text.get("layer_types"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "head_dim": text.get("head_dim"),
        "num_experts": text.get("num_experts"),
        "num_experts_per_tok": text.get("num_experts_per_tok"),
        "moe_intermediate_size": text.get("moe_intermediate_size"),
        "shared_expert_intermediate_size": text.get(
            "shared_expert_intermediate_size"
        ),
        "vocab_size": text.get("vocab_size"),
        "mtp_num_hidden_layers": text.get("mtp_num_hidden_layers"),
        "body bits": body_quant.get("bits"),
        "body group_size": body_quant.get("group_size"),
        "body mode": body_quant.get("mode"),
        "MTP bits": mtp_quant.get("bits"),
        "MTP group_size": mtp_quant.get("group_size"),
        "MTP mode": mtp_quant.get("mode"),
        "MTP policy": mtp_quant.get("policy"),
        "MTP prequantized": mtp_quant.get("prequantized"),
    }
    for name, expected_value in expected.items():
        _require_equal(name, actual[name], expected_value)

    encoded = json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()
    return config, hashlib.sha256(encoded).hexdigest()[:16]


def _validate_runtime(runtime: Any) -> None:
    _require_equal("runtime mtp_enabled", bool(runtime.mtp_enabled), True)
    router_report = getattr(runtime, "qwen_row_owned_router_report", None)
    if not isinstance(router_report, Mapping):
        raise A3BMTPBatchInstallError(
            "Qwen 35B mtp_batch requires a row-owned router install receipt"
        )
    _require_equal(
        "runtime row-owned router installed",
        bool(router_report.get("installed")),
        True,
    )
    _require_equal(
        "runtime row-owned target routers", router_report.get("target_routers"), 40
    )
    _require_equal(
        "runtime row-owned MTP routers", router_report.get("mtp_routers"), 1
    )
    router_contract = router_report.get("validated_contract")
    if not isinstance(router_contract, Mapping):
        raise A3BMTPBatchInstallError(
            "Qwen 35B mtp_batch requires the row-owned router contract"
        )
    routes = router_contract.get("routes")
    combine_tail = router_contract.get("combine_tail")
    _require_equal(
        "runtime row-owned M1-M16 decode route",
        routes.get("decode_verify") if isinstance(routes, Mapping) else None,
        list(range(1, 17)),
    )
    _require_equal(
        "runtime combine-tail M1-M2-M8-M16 route",
        combine_tail.get("decode_verify")
        if isinstance(combine_tail, Mapping)
        else None,
        [1, 2, 8, 16],
    )
    contract = getattr(runtime, "contract", None)
    if contract is None:
        raise A3BMTPBatchInstallError("Qwen 35B mtp_batch requires MTP contract")
    _require_equal(
        "runtime hidden_variant", getattr(contract, "hidden_variant", None), "post_norm"
    )
    _require_equal(
        "runtime concat_order",
        getattr(contract, "concat_order", None),
        "embedding_hidden",
    )
    _require_equal(
        "runtime MTP bits", getattr(contract, "mtp_quant_bits", None), 4
    )
    _require_equal(
        "runtime MTP group_size",
        getattr(contract, "mtp_quant_group_size", None),
        32,
    )
    _require_equal(
        "runtime MTP mode", getattr(contract, "mtp_quant_mode", None), "affine"
    )
    if (
        getattr(runtime, "mtp_adapter_path", None) is not None
        or getattr(runtime, "mtp_adapter_metadata", None) is not None
    ):
        raise A3BMTPBatchInstallError(
            "Qwen 35B mtp_batch does not install over an MTP adapter"
        )
    trunk_layers, mtp_layers = _model_layers(runtime)
    _require_equal("constructed num_hidden_layers", len(trunk_layers), 40)
    _require_equal("constructed mtp_num_hidden_layers", len(mtp_layers), 1)


def _bind_postconv_capture_forward(
    runtime: Any,
    *,
    implementation_field: str,
    contract_label: str,
) -> Callable[..., Any]:
    factory = getattr(runtime, "a3b_compiled_target_prefix_factory", None)
    if factory is None:
        raise A3BMTPBatchInstallError(
            "Qwen 35B mtp_batch requires the compiled target-prefix capture factory"
        )
    _require_equal("compiled GDN layers", getattr(factory, "gdn_layers", None), 30)
    _require_equal(
        "compiled full-attention layers",
        getattr(factory, "full_attention_layers", None),
        10,
    )
    _require_equal("compiled hidden_size", getattr(factory, "hidden_size", None), 2048)
    _require_equal(
        "compiled quantization",
        getattr(factory, "quantization", None),
        "affine_q4_group64",
    )
    _require_equal(
        "compiled layer_types", getattr(factory, "layer_types", None), _LAYER_TYPES
    )
    postconv = getattr(factory, "gdn_postconv", None)
    implementations = tuple(getattr(postconv, implementation_field, ()) or ())
    if len(implementations) != 30 or not all(callable(item) for item in implementations):
        raise A3BMTPBatchInstallError(
            f"Qwen 35B mtp_batch requires 30 {contract_label} post-conv implementations"
        )
    capture = _require_callable(runtime, "_forward_ar_capture_a3b_postconv")
    return partial(
        capture,
        hidden_variant="post_norm",
        postconv_implementations=implementations,
    )


def _bind_capture_forward(runtime: Any) -> Callable[..., Any]:
    eager_capture = _bind_postconv_capture_forward(
        runtime,
        implementation_field="b8_t2_implementations",
        contract_label="B8/T2",
    )
    return _compile_qwen35b_b8_t2_capture(eager_capture)


def _compile_qwen35b_b8_t2_capture(
    eager_capture: Callable[..., Any],
) -> Callable[..., Any]:
    """Compile the fixed B8/T2 target graph with explicit row-owned state."""
    shadow: list[Any] = []
    for layer_type in _LAYER_TYPES:
        if layer_type == "full_attention":
            shadow.append(RaggedBatchKVCache(batch_size=8, step=256))
        else:
            shadow.append(ArraysCache(2))

    def step(input_ids: Any, *state_in: Any) -> tuple[Any, ...]:
        position = 0
        for entry, layer_type in zip(shadow, _LAYER_TYPES, strict=True):
            if layer_type == "full_attention":
                entry.keys = state_in[position]
                entry.values = state_in[position + 1]
                entry.offsets = state_in[position + 2]
                entry._frozen_capacity = int(entry.keys.shape[2])
                position += 3
            else:
                entry[0] = state_in[position]
                entry[1] = state_in[position + 1]
                position += 2
        logits, hidden, captures = eager_capture(input_ids, cache=shadow)
        captured_state: list[Any] = []
        attention_state: list[Any] = []
        for layer_idx, (entry, layer_type) in enumerate(
            zip(shadow, _LAYER_TYPES, strict=True)
        ):
            if layer_type == "full_attention":
                attention_state.extend((entry.keys, entry.values, entry.offsets))
            else:
                capture = captures[layer_idx]
                captured_state.extend(
                    (capture["conv_states"], capture["states"])
                )
        return (logits, hidden, *captured_state, *attention_state)

    compiled = mx.compile(step)

    def capture_forward(input_ids: Any, *, cache: list[Any]) -> tuple[Any, ...]:
        state_in: list[Any] = []
        for entry, layer_type in zip(cache, _LAYER_TYPES, strict=True):
            if layer_type == "full_attention":
                state_in.extend((entry.keys, entry.values, entry.offsets))
            else:
                state_in.extend((entry[0], entry[1]))
        outputs = compiled(input_ids, *state_in)
        captures: dict[int, dict[str, Any]] = {}
        position = 2
        for layer_idx, (entry, layer_type) in enumerate(
            zip(cache, _LAYER_TYPES, strict=True)
        ):
            if layer_type == "full_attention":
                continue
            conv_states = outputs[position]
            states = outputs[position + 1]
            position += 2
            captures[layer_idx] = {
                "conv_states": conv_states,
                "states": states,
            }
            entry[0] = conv_states[:, -1]
            entry[1] = states[:, -1]
        for entry, layer_type in zip(cache, _LAYER_TYPES, strict=True):
            if layer_type != "full_attention":
                continue
            entry.keys = outputs[position]
            entry.values = outputs[position + 1]
            entry.offsets = outputs[position + 2]
            position += 3
        mx.async_eval(*outputs)
        return outputs[0], outputs[1], captures

    capture_forward._mtplx_compiled_qwen35b_b8_t2 = True
    return capture_forward


def _bind_solo_capture_forward(runtime: Any) -> Callable[..., Any]:
    return _bind_postconv_capture_forward(
        runtime,
        implementation_field="m2_implementations",
        contract_label="B1/T2",
    )


def _qwen35b_b8_stock_attention(
    self: Any,
    x: Any,
    mask: Any | None = None,
    cache: Any | None = None,
) -> Any:
    """Exact fused SDPA route for installed B1 prefill and B8/T2 verify."""
    if not _MTP_BATCH_ATTENTION_ACTIVE.get():
        return self._mtplx_mtp_batch_original_call(x, mask=mask, cache=cache)
    batch, length, _hidden = x.shape
    projected = self.q_proj(x)
    queries, gate = mx.split(
        projected.reshape(batch, length, self.num_attention_heads, -1),
        2,
        axis=-1,
    )
    gate = gate.reshape(batch, length, -1)
    keys = self.k_proj(x)
    values = self.v_proj(x)
    queries = self.q_norm(queries).transpose(0, 2, 1, 3)
    keys = self.k_norm(
        keys.reshape(batch, length, self.num_key_value_heads, -1)
    ).transpose(0, 2, 1, 3)
    values = values.reshape(
        batch, length, self.num_key_value_heads, -1
    ).transpose(0, 2, 1, 3)
    queries = self.rope(queries, offset=cache.offset)
    keys = self.rope(keys, offset=cache.offset)
    keys, values = cache.update_and_fetch(keys, values)
    output = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=cache,
        scale=self.scale,
        mask=mask,
    )
    output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
    return self.o_proj(output * mx.sigmoid(gate))


def _call_with_qwen35b_mtp_batch_attention(
    *args: Any,
    call: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    token = _MTP_BATCH_ATTENTION_ACTIVE.set(True)
    try:
        return call(*args, **kwargs)
    finally:
        _MTP_BATCH_ATTENTION_ACTIVE.reset(token)


def _install_qwen35b_b8_attention_route(runtime: Any) -> str:
    trunk_layers, mtp_layers = _model_layers(runtime)
    target_attention = [
        layer.self_attn
        for layer, layer_type in zip(trunk_layers, _LAYER_TYPES, strict=True)
        if layer_type == "full_attention"
    ]
    _require_equal("constructed full-attention layers", len(target_attention), 10)
    mtp_attention = [mtp_layers[0].self_attn]
    full_attention = [*target_attention, *mtp_attention]
    exact_classes: dict[type, type] = {}
    for attention in full_attention:
        base = type(attention)
        exact = exact_classes.get(base)
        if exact is None:
            exact = type(
                f"MTPLXQwen35B8Stock{base.__name__}",
                (base,),
                {
                    "__call__": _qwen35b_b8_stock_attention,
                    "_mtplx_mtp_batch_original_call": base.__call__,
                },
            )
            exact_classes[base] = exact
        attention.__class__ = exact
    runtime.qwen35b_mtp_batch_attention_report = {
        "installed": True,
        "target_layers": 10,
        "mtp_layers": 1,
        "route_id": "qwen35b_b8_t2_stock_fused_sdpa",
    }
    return "qwen35b_b8_t2_stock_fused_sdpa"


def _commit_qwen35b_b8_t2_rows(
    cache: list[Any],
    captures: dict[int, dict[str, Any]],
    keep_tokens_by_row: list[int],
    base_recurrent: dict[int, tuple[Any, Any]],
) -> None:
    """Commit the prevalidated B8/T2 cache layout without hot-path proof work."""
    import mlx.core as mx

    keeps = mx.array(keep_tokens_by_row, dtype=mx.int32)
    positions = [max(0, int(value) - 1) for value in keep_tokens_by_row]
    active_rows = mx.array(
        [int(value) > 0 for value in keep_tokens_by_row], dtype=mx.bool_
    )
    for layer_idx, layer_type in enumerate(_LAYER_TYPES):
        entry = cache[layer_idx]
        if layer_type == "full_attention":
            entry.offsets = (entry.offsets - 2 + keeps).astype(mx.int32)
            continue
        capture = captures[layer_idx]
        conv_states = capture["conv_states"]
        states = capture["states"]
        conv_position_selector = mx.array(positions, dtype=mx.int32).reshape(
            (8, 1) + (1,) * (int(conv_states.ndim) - 2)
        )
        state_position_selector = mx.array(positions, dtype=mx.int32).reshape(
            (8, 1) + (1,) * (int(states.ndim) - 2)
        )
        conv_selector = mx.broadcast_to(
            conv_position_selector, (8, 1) + tuple(conv_states.shape[2:])
        )
        state_selector = mx.broadcast_to(
            state_position_selector, (8, 1) + tuple(states.shape[2:])
        )
        selected_conv = mx.contiguous(
            mx.take_along_axis(conv_states, conv_selector, axis=1)[:, 0]
        )
        selected_state = mx.contiguous(
            mx.take_along_axis(states, state_selector, axis=1)[:, 0]
        )
        conv_mask = active_rows.reshape(
            (8,) + (1,) * (int(selected_conv.ndim) - 1)
        )
        state_mask = active_rows.reshape(
            (8,) + (1,) * (int(selected_state.ndim) - 1)
        )
        base_conv, base_state = base_recurrent[layer_idx]
        entry[0] = mx.where(conv_mask, selected_conv, base_conv)
        entry[1] = mx.where(state_mask, selected_state, base_state)


def _default_selfcheck(lane: InstalledA3BMTPBatchLane, runtime: Any) -> dict[str, Any]:
    """Run one real B8/T2 route and compare row zero with unchanged B1."""

    import mlx.core as mx
    import numpy as np

    from .attention_context import attention_phase
    from .qwen_row_owned_router import (
        call_with_stock_qwen_row_owned_routers,
    )

    token = int(getattr(getattr(runtime, "tokenizer", None), "eos_token_id", 1) or 1)

    solo_capture_forward = _bind_solo_capture_forward(runtime)
    eager_b8_capture_forward = partial(
        _call_with_qwen35b_mtp_batch_attention,
        call=_bind_postconv_capture_forward(
            runtime,
            implementation_field="b8_t2_implementations",
            contract_label="B8/T2",
        ),
    )
    stock_b8_capture_forward = partial(
        _call_with_qwen35b_mtp_batch_attention,
        call=partial(
            call_with_stock_qwen_row_owned_routers,
            call=partial(
                runtime.forward_ar_capture,
                return_hidden=True,
                hidden_variant="post_norm",
                capture_backend="stock",
            ),
        ),
    )
    solo_target_forward = runtime.model
    solo_draft_forward = partial(
        runtime.model.mtp_forward,
        concat_order=getattr(runtime.contract, "concat_order", None),
        return_hidden=False,
        mtp_hidden_variant="post_norm",
    )
    (
        prefill_cache,
        prefill_logits,
        prefill_hidden,
        prefill_mtp_cache,
        *_prefill_metadata,
    ) = lane.prefill_request([token, token], abort_check=None)
    from .cache_state import _is_trimmable

    prefill_contract = bool(
        tuple(prefill_logits.shape) == (1, lane.geometry.vocab_size)
        and tuple(prefill_hidden.shape) == (1, 1, lane.geometry.hidden_size)
        and len(prefill_cache) == lane.geometry.hidden_layers
        and all(
            int(getattr(entry, "offset", -1)) == 2
            for entry, layer_type in zip(
                prefill_cache, _LAYER_TYPES, strict=True
            )
            if layer_type == "full_attention" and _is_trimmable(entry)
        )
        and len(prefill_mtp_cache) == 1
        and _is_trimmable(prefill_mtp_cache[0])
        and int(getattr(prefill_mtp_cache[0], "offset", -1)) == 1
    )
    del prefill_cache, prefill_logits, prefill_hidden, prefill_mtp_cache

    parity_prompt = [
        int((token + index) % lane.geometry.vocab_size) for index in range(5)
    ]
    prefill_keywords = dict(getattr(lane.prefill_request, "keywords", {}) or {})
    dedicated_prefill = _prefill_qwen35b_batch_request(
        parity_prompt,
        target_forward=prefill_keywords["target_forward"],
        target_cache_factory=prefill_keywords["target_cache_factory"],
        mtp_cache_factory=prefill_keywords["mtp_cache_factory"],
        update_mtp_cache=prefill_keywords["update_mtp_cache"],
        chunk_size=2,
        cleanup_every=0,
        abort_check=None,
    )
    from .generation import _prefill_committed_mtp_history_streaming

    import os

    context_key = "MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS"
    saved_context = os.environ.get(context_key)
    os.environ[context_key] = str(len(parity_prompt))
    try:
        reference_prefill = _prefill_committed_mtp_history_streaming(
            runtime,
            parity_prompt,
            base_hidden_variant="post_norm",
            mtp_hidden_variant="post_norm",
            mtp_position_mode="cache",
            prefill_chunk_size=2,
        )
    finally:
        if saved_context is None:
            os.environ.pop(context_key, None)
        else:
            os.environ[context_key] = saved_context
    dedicated_target, dedicated_logits, dedicated_hidden, dedicated_mtp = (
        dedicated_prefill[:4]
    )
    reference_target, reference_logits, reference_hidden, reference_mtp = (
        reference_prefill[:4]
    )
    prefill_comparisons = [
        mx.all(dedicated_logits == reference_logits),
        mx.all(dedicated_hidden == reference_hidden),
    ]
    prefill_offsets_match = True
    for layer_idx, layer_type in enumerate(_LAYER_TYPES):
        dedicated_entry = dedicated_target[layer_idx]
        reference_entry = reference_target[layer_idx]
        if layer_type == "full_attention":
            prefill_offsets_match = bool(
                prefill_offsets_match
                and int(dedicated_entry.offset) == int(reference_entry.offset)
            )
            prefill_comparisons.extend(
                (
                    mx.all(dedicated_entry.keys == reference_entry.keys),
                    mx.all(dedicated_entry.values == reference_entry.values),
                )
            )
        else:
            prefill_comparisons.extend(
                mx.all(left == right)
                for left, right in zip(
                    dedicated_entry, reference_entry, strict=True
                )
            )
    prefill_offsets_match = bool(
        prefill_offsets_match
        and int(dedicated_mtp[0].offset) == int(reference_mtp[0].offset)
    )
    prefill_comparisons.extend(
        (
            mx.all(dedicated_mtp[0].keys == reference_mtp[0].keys),
            mx.all(dedicated_mtp[0].values == reference_mtp[0].values),
        )
    )
    mx.eval(*prefill_comparisons)
    prefill_numerical_parity = bool(
        prefill_offsets_match
        and all(bool(np.asarray(value).item()) for value in prefill_comparisons)
    )
    del dedicated_prefill, reference_prefill

    one_token_prefills = [
        lane.prefill_request(
            [int((token + row) % lane.geometry.vocab_size)], abort_check=None
        )
        for row in range(lane.geometry.cohort_slots)
    ]
    one_token_hidden = mx.concatenate([item[2] for item in one_token_prefills])
    one_token_primary = mx.argmax(
        mx.concatenate([item[1] for item in one_token_prefills]), axis=-1
    )
    solo_empty_drafts = []
    for row in range(lane.geometry.cohort_slots):
        with attention_phase("ar_decode"):
            solo_empty_drafts.append(
                lane.draft_forward(
                    one_token_hidden[row : row + 1],
                    one_token_primary[row : row + 1, None],
                    mtp_cache=prefill_keywords["mtp_cache_factory"](),
                )
            )
    empty_merged_mtp = lane.merge_mtp_caches(
        [item[3] for item in one_token_prefills]
    )
    empty_merged_mtp[0]._capacity_bound = 0
    empty_merged_mtp[0].reserve(1)
    with attention_phase("ar_decode"):
        batch_empty_draft = lane.draft_forward(
            one_token_hidden,
            one_token_primary[:, None],
            mtp_cache=empty_merged_mtp,
        )
    empty_draft_errors = [
        mx.max(
            mx.abs(
                batch_empty_draft[row : row + 1] - solo_empty_drafts[row]
            ).astype(mx.float32)
        )
        for row in range(lane.geometry.cohort_slots)
    ]
    empty_draft_reference_max = [
        mx.max(mx.abs(value).astype(mx.float32))
        for value in solo_empty_drafts
    ]
    empty_draft_argmax_comparisons = [
        mx.all(
            mx.argmax(batch_empty_draft[row : row + 1], axis=-1)
            == mx.argmax(solo_empty_drafts[row], axis=-1)
        )
        for row in range(lane.geometry.cohort_slots)
    ]
    mx.eval(
        *empty_draft_errors,
        *empty_draft_reference_max,
        *empty_draft_argmax_comparisons,
    )
    empty_mtp_draft_max_abs = max(
        float(np.asarray(value).item()) for value in empty_draft_errors
    )
    empty_mtp_draft_reference_max_abs = max(
        float(np.asarray(value).item()) for value in empty_draft_reference_max
    )
    empty_mtp_draft_relative_error = (
        empty_mtp_draft_max_abs
        / max(1.0, empty_mtp_draft_reference_max_abs)
    )
    empty_mtp_draft_argmax_parity = all(
        bool(np.asarray(value).item())
        for value in empty_draft_argmax_comparisons
    )

    isolated_empty_mtp = lane.merge_mtp_caches(
        [lane.make_mtp_cache() for _ in range(lane.geometry.cohort_slots)]
    )
    isolated_empty_mtp[0]._capacity_bound = 0
    isolated_empty_mtp[0].reserve(1)
    isolated_hidden = mx.concatenate(
        (one_token_hidden[:1] + 1, one_token_hidden[1:]), axis=0
    )
    isolated_primary = mx.concatenate(
        (
            ((one_token_primary[:1] + 1) % lane.geometry.vocab_size),
            one_token_primary[1:],
        ),
        axis=0,
    )
    with attention_phase("ar_decode"):
        isolated_empty_draft = lane.draft_forward(
            isolated_hidden,
            isolated_primary[:, None],
            mtp_cache=isolated_empty_mtp,
        )
    empty_isolation_check = mx.all(
        batch_empty_draft[1:] == isolated_empty_draft[1:]
    )
    mx.eval(empty_isolation_check)
    empty_mtp_row_isolation_parity = bool(
        np.asarray(empty_isolation_check).item()
    )
    del one_token_prefills, empty_merged_mtp, batch_empty_draft

    def run(
        token_ids: list[int],
        *,
        capture_forward: Callable[..., Any],
        target_forward: Callable[..., Any] | None = None,
        draft_forward: Callable[..., Any] | None = None,
        verify_input_override: Any | None = None,
        keeps: list[int] | None = None,
        installed_commit: bool = False,
    ):
        batch = len(token_ids)
        selected_target_forward = target_forward or lane.target_forward
        selected_draft_forward = draft_forward or lane.draft_forward
        if batch == lane.geometry.cohort_slots:
            row_prefills = [
                lane.prefill_request([row_token], abort_check=None)
                for row_token in token_ids
            ]
            cache = lane.merge_target_caches(
                [item[0] for item in row_prefills]
            )
            mtp_cache = lane.merge_mtp_caches(
                [item[3] for item in row_prefills]
            )
            mtp_cache[0].reserve(1)
            logits = mx.concatenate([item[1] for item in row_prefills], axis=0)[
                :, None, :
            ]
            hidden = mx.concatenate([item[2] for item in row_prefills], axis=0)
        else:
            cache = lane.make_cache()
            prompt = mx.array(token_ids, dtype=mx.int32).reshape(batch, 1)
            with attention_phase("prefill"):
                logits, hidden = selected_target_forward(
                    prompt,
                    cache=cache,
                    return_hidden=True,
                    hidden_variant="post_norm",
                )
            mtp_cache = lane.make_mtp_cache()
        if verify_input_override is None:
            primary = mx.argmax(logits[:, -1, :], axis=-1)
            with attention_phase("ar_decode"):
                draft_logits = selected_draft_forward(
                    hidden[:, -1:, :],
                    primary[:, None],
                    mtp_cache=mtp_cache,
                )
            draft = mx.argmax(draft_logits[:, -1, :], axis=-1)
            verify_input = mx.stack((primary, draft), axis=1)
        else:
            verify_input = verify_input_override
        pre_verify_recurrent = {
            layer_idx: (cache[layer_idx][0], cache[layer_idx][1])
            for layer_idx, layer_type in enumerate(_LAYER_TYPES)
            if layer_type == "linear_attention"
        }
        with attention_phase("decode_verify"):
            verify_logits, verify_hidden, captures = capture_forward(
                verify_input,
                cache=cache,
            )
        from .gdn_capture import commit_captured_rows

        row_keeps = keeps or ([2] * batch)
        if installed_commit:
            lane.commit_rows(
                cache, captures, row_keeps, pre_verify_recurrent
            )
            row_commit = True
        else:
            reference_keeps = [max(1, int(value)) for value in row_keeps]
            row_commit = commit_captured_rows(
                cache,
                captures,
                keep_tokens_by_row=reference_keeps,
                verified_tokens=2,
            )
            if 0 in row_keeps:
                inactive_rows = mx.array(
                    [int(value) == 0 for value in row_keeps], dtype=mx.bool_
                )
                for layer_idx, layer_type in enumerate(_LAYER_TYPES):
                    entry = cache[layer_idx]
                    if layer_type == "full_attention":
                        entry.offsets = (
                            entry.offsets
                            - mx.array(
                                [int(value) == 0 for value in row_keeps],
                                dtype=mx.int32,
                            )
                        ).astype(mx.int32)
                        continue
                    before_conv, before_state = pre_verify_recurrent[layer_idx]
                    conv_mask = inactive_rows.reshape(
                        (batch,) + (1,) * (int(entry[0].ndim) - 1)
                    )
                    state_mask = inactive_rows.reshape(
                        (batch,) + (1,) * (int(entry[1].ndim) - 1)
                    )
                    entry[0] = mx.where(conv_mask, before_conv, entry[0])
                    entry[1] = mx.where(state_mask, before_state, entry[1])
        mx.eval(verify_logits, verify_hidden)
        return verify_input, verify_logits, verify_hidden, captures, row_commit, cache

    batch_tokens = [
        int((token + row) % lane.geometry.vocab_size)
        for row in range(lane.geometry.cohort_slots)
    ]
    mixed_keeps = [0, 1, 2, 0, 1, 2, 0, 1]
    (
        batch_input,
        batch_logits,
        batch_hidden,
        batch_captures,
        batch_commit,
        reference_cache,
    ) = run(
        batch_tokens,
        capture_forward=lane.capture_forward,
        keeps=mixed_keeps,
    )
    (
        eager_input,
        eager_logits,
        eager_hidden,
        eager_captures,
        eager_commit,
        eager_cache,
    ) = run(
        batch_tokens,
        capture_forward=eager_b8_capture_forward,
        verify_input_override=batch_input,
        keeps=mixed_keeps,
    )
    (
        stock_input,
        stock_logits,
        stock_hidden,
        stock_captures,
        stock_commit,
        stock_cache,
    ) = run(
        batch_tokens,
        capture_forward=stock_b8_capture_forward,
        verify_input_override=batch_input,
        keeps=mixed_keeps,
    )
    compiled_eager_output_checks = [
        mx.all(batch_input == eager_input),
        mx.all(batch_logits == eager_logits),
        mx.all(batch_hidden == eager_hidden),
    ]
    compiled_eager_capture_checks = []
    compiled_eager_cache_checks = []
    compiled_eager_attention_offset_checks = []
    compiled_eager_attention_errors = []
    compiled_eager_attention_reference_max = []
    compiled_eager_errors = [
        mx.max(mx.abs(batch_logits - eager_logits).astype(mx.float32)),
        mx.max(mx.abs(batch_hidden - eager_hidden).astype(mx.float32)),
    ]
    compiled_eager_reference_max = [
        mx.max(mx.abs(eager_logits).astype(mx.float32)),
        mx.max(mx.abs(eager_hidden).astype(mx.float32)),
    ]
    same_geometry_errors = [
        mx.max(mx.abs(batch_logits - stock_logits).astype(mx.float32)),
        mx.max(mx.abs(batch_hidden - stock_hidden).astype(mx.float32)),
    ]
    same_geometry_reference_max = [
        mx.max(mx.abs(stock_logits).astype(mx.float32)),
        mx.max(mx.abs(stock_hidden).astype(mx.float32)),
    ]
    same_geometry_shapes = bool(
        tuple(batch_input.shape) == tuple(stock_input.shape) == (8, 2)
        and tuple(batch_logits.shape) == tuple(stock_logits.shape)
        and tuple(batch_hidden.shape) == tuple(stock_hidden.shape)
        and len(batch_captures) == len(stock_captures) == 30
    )
    for layer_idx in batch_captures:
        same_geometry_shapes = bool(
            same_geometry_shapes
            and tuple(batch_captures[layer_idx]["conv_states"].shape)
            == tuple(stock_captures[layer_idx]["conv_states"].shape)
            and tuple(batch_captures[layer_idx]["states"].shape)
            == tuple(stock_captures[layer_idx]["states"].shape)
            and tuple(batch_captures[layer_idx]["conv_states"].shape[:2])
            == (8, 2)
            and tuple(batch_captures[layer_idx]["states"].shape[:2])
            == (8, 2)
        )
        compiled_eager_capture_checks.extend(
            (
                mx.all(
                    batch_captures[layer_idx]["conv_states"]
                    == eager_captures[layer_idx]["conv_states"]
                ),
                mx.all(
                    batch_captures[layer_idx]["states"]
                    == eager_captures[layer_idx]["states"]
                ),
            )
        )
        compiled_eager_errors.extend(
            (
                mx.max(
                    mx.abs(
                        batch_captures[layer_idx]["conv_states"]
                        - eager_captures[layer_idx]["conv_states"]
                    ).astype(mx.float32)
                ),
                mx.max(
                    mx.abs(
                        batch_captures[layer_idx]["states"]
                        - eager_captures[layer_idx]["states"]
                    ).astype(mx.float32)
                ),
            )
        )
        compiled_eager_reference_max.extend(
            (
                mx.max(
                    mx.abs(eager_captures[layer_idx]["conv_states"]).astype(
                        mx.float32
                    )
                ),
                mx.max(
                    mx.abs(eager_captures[layer_idx]["states"]).astype(
                        mx.float32
                    )
                ),
            )
        )
        same_geometry_errors.extend(
            (
                mx.max(
                    mx.abs(
                        batch_captures[layer_idx]["conv_states"]
                        - stock_captures[layer_idx]["conv_states"]
                    ).astype(mx.float32)
                ),
                mx.max(
                    mx.abs(
                        batch_captures[layer_idx]["states"]
                        - stock_captures[layer_idx]["states"]
                    ).astype(mx.float32)
                ),
            )
        )
        same_geometry_reference_max.extend(
            (
                mx.max(
                    mx.abs(stock_captures[layer_idx]["conv_states"]).astype(
                        mx.float32
                    )
                ),
                mx.max(
                    mx.abs(stock_captures[layer_idx]["states"]).astype(
                        mx.float32
                    )
                ),
            )
        )
    same_geometry_attention_offset_checks = []
    same_geometry_attention_errors = []
    same_geometry_attention_reference_max = []
    for layer_idx, layer_type in enumerate(_LAYER_TYPES):
        compiled_entry = reference_cache[layer_idx]
        eager_entry = eager_cache[layer_idx]
        if layer_type != "full_attention":
            compiled_eager_cache_checks.extend(
                mx.all(compiled_value == eager_value)
                for compiled_value, eager_value in zip(
                    compiled_entry, eager_entry, strict=True
                )
            )
            continue
        stock_entry = stock_cache[layer_idx]
        compiled_eager_cache_checks.extend(
            (
                mx.all(compiled_entry.offsets == eager_entry.offsets),
                mx.all(compiled_entry.keys == eager_entry.keys),
                mx.all(compiled_entry.values == eager_entry.values),
            )
        )
        compiled_eager_attention_offset_checks.append(
            mx.all(compiled_entry.offsets == eager_entry.offsets)
        )
        compiled_eager_attention_errors.extend(
            (
                mx.max(
                    mx.abs(compiled_entry.keys - eager_entry.keys).astype(
                        mx.float32
                    )
                ),
                mx.max(
                    mx.abs(compiled_entry.values - eager_entry.values).astype(
                        mx.float32
                    )
                ),
            )
        )
        compiled_eager_attention_reference_max.extend(
            (
                mx.max(mx.abs(eager_entry.keys).astype(mx.float32)),
                mx.max(mx.abs(eager_entry.values).astype(mx.float32)),
            )
        )
        same_geometry_attention_offset_checks.append(
            mx.all(compiled_entry.offsets == stock_entry.offsets)
        )
        same_geometry_attention_errors.extend(
            (
                mx.max(
                    mx.abs(compiled_entry.keys - stock_entry.keys).astype(
                        mx.float32
                    )
                ),
                mx.max(
                    mx.abs(compiled_entry.values - stock_entry.values).astype(
                        mx.float32
                    )
                ),
            )
        )
        same_geometry_attention_reference_max.extend(
            (
                mx.max(mx.abs(stock_entry.keys).astype(mx.float32)),
                mx.max(mx.abs(stock_entry.values).astype(mx.float32)),
            )
        )
    compiled_eager_checks = [
        *compiled_eager_output_checks,
        *compiled_eager_capture_checks,
        *compiled_eager_cache_checks,
    ]
    same_geometry_argmax_check = mx.all(
        mx.argmax(batch_logits, axis=-1) == mx.argmax(stock_logits, axis=-1)
    )
    compiled_eager_argmax_check = mx.all(
        mx.argmax(batch_logits, axis=-1) == mx.argmax(eager_logits, axis=-1)
    )
    mx.eval(
        *compiled_eager_checks,
        *compiled_eager_attention_offset_checks,
        *compiled_eager_errors,
        *compiled_eager_reference_max,
        *compiled_eager_attention_errors,
        *compiled_eager_attention_reference_max,
        *same_geometry_errors,
        *same_geometry_reference_max,
        *same_geometry_attention_offset_checks,
        *same_geometry_attention_errors,
        *same_geometry_attention_reference_max,
        compiled_eager_argmax_check,
        same_geometry_argmax_check,
    )
    compiled_eager_bitwise_parity = bool(
        eager_commit
        and all(bool(np.asarray(value).item()) for value in compiled_eager_checks)
    )
    compiled_eager_check_values = [
        bool(np.asarray(value).item()) for value in compiled_eager_checks
    ]
    compiled_eager_error_values = [
        float(np.asarray(value).item()) for value in compiled_eager_errors
    ]
    compiled_eager_reference_values = [
        float(np.asarray(value).item())
        for value in compiled_eager_reference_max
    ]
    compiled_eager_relative_errors = {
        "logits": compiled_eager_error_values[0]
        / max(1.0, compiled_eager_reference_values[0]),
        "hidden": compiled_eager_error_values[1]
        / max(1.0, compiled_eager_reference_values[1]),
        "conv": max(compiled_eager_error_values[2::2])
        / max(1.0, max(compiled_eager_reference_values[2::2])),
        "state": max(compiled_eager_error_values[3::2])
        / max(1.0, max(compiled_eager_reference_values[3::2])),
        "attention": max(
            float(np.asarray(value).item())
            for value in compiled_eager_attention_errors
        )
        / max(
            1.0,
            max(
                float(np.asarray(value).item())
                for value in compiled_eager_attention_reference_max
            ),
        ),
    }
    compiled_eager_argmax_parity = bool(
        np.asarray(compiled_eager_argmax_check).item()
    )
    compiled_eager_offset_parity = all(
        bool(np.asarray(value).item())
        for value in compiled_eager_attention_offset_checks
    )
    compiled_eager_numerical_parity = bool(
        compiled_eager_output_checks
        and bool(np.asarray(compiled_eager_output_checks[0]).item())
        and same_geometry_shapes
        and eager_commit
        and compiled_eager_argmax_parity
        and compiled_eager_offset_parity
        and all(
            value <= _BF16_GEOMETRY_RELATIVE_LIMIT
            for value in compiled_eager_relative_errors.values()
        )
    )
    same_geometry_error_values = [
        float(np.asarray(value).item()) for value in same_geometry_errors
    ]
    same_geometry_reference_values = [
        float(np.asarray(value).item())
        for value in same_geometry_reference_max
    ]
    same_geometry_relative_errors = {
        "logits": same_geometry_error_values[0]
        / max(1.0, same_geometry_reference_values[0]),
        "hidden": same_geometry_error_values[1]
        / max(1.0, same_geometry_reference_values[1]),
        "conv": max(same_geometry_error_values[2::2])
        / max(1.0, max(same_geometry_reference_values[2::2])),
        "state": max(same_geometry_error_values[3::2])
        / max(1.0, max(same_geometry_reference_values[3::2])),
        "attention": max(
            float(np.asarray(value).item())
            for value in same_geometry_attention_errors
        )
        / max(
            1.0,
            max(
                float(np.asarray(value).item())
                for value in same_geometry_attention_reference_max
            ),
        ),
    }
    same_geometry_argmax_parity = bool(
        np.asarray(same_geometry_argmax_check).item()
    )
    same_geometry_attention_parity = all(
        bool(np.asarray(value).item())
        for value in same_geometry_attention_offset_checks
    )
    same_geometry_numerical_parity = bool(
        same_geometry_shapes
        and stock_commit
        and same_geometry_attention_parity
        and same_geometry_argmax_parity
        and all(
            value <= _BF16_GEOMETRY_RELATIVE_LIMIT
            for value in same_geometry_relative_errors.values()
        )
    )
    *_, installed_cache = run(
        batch_tokens,
        capture_forward=lane.capture_forward,
        keeps=mixed_keeps,
        installed_commit=True,
    )
    commit_comparisons = []
    for layer_idx, layer_type in enumerate(_LAYER_TYPES):
        reference_entry = reference_cache[layer_idx]
        installed_entry = installed_cache[layer_idx]
        if layer_type == "full_attention":
            commit_comparisons.append(
                mx.all(reference_entry.offsets == installed_entry.offsets)
            )
        else:
            commit_comparisons.extend(
                mx.all(reference_value == installed_value)
                for reference_value, installed_value in zip(
                    reference_entry, installed_entry, strict=True
                )
            )
    next_tokens = mx.array(
        [(value + 17) % lane.geometry.vocab_size for value in batch_tokens],
        dtype=mx.int32,
    ).reshape(8, 1)
    with attention_phase("ar_decode"):
        reference_next_logits, reference_next_hidden = lane.target_forward(
            next_tokens,
            cache=reference_cache,
            return_hidden=True,
        )
        installed_next_logits, installed_next_hidden = lane.target_forward(
            next_tokens,
            cache=installed_cache,
            return_hidden=True,
        )
    commit_comparisons.extend(
        (
            mx.all(reference_next_logits == installed_next_logits),
            mx.all(reference_next_hidden == installed_next_hidden),
        )
    )
    mx.eval(*commit_comparisons)
    mixed_commit_parity = all(
        bool(np.asarray(value).item()) for value in commit_comparisons
    )
    target_shape = [int(value) for value in batch_input.shape]
    logits_shape = [int(value) for value in batch_logits.shape]
    hidden_shape = [int(value) for value in batch_hidden.shape]
    heterogeneous_row_parity = True
    heterogeneous_row_max_abs = 0.0
    heterogeneous_logits_max_abs = 0.0
    heterogeneous_hidden_max_abs = 0.0
    heterogeneous_conv_max_abs = 0.0
    heterogeneous_state_max_abs = 0.0
    heterogeneous_logits_reference_max_abs = 0.0
    heterogeneous_hidden_reference_max_abs = 0.0
    heterogeneous_conv_reference_max_abs = 0.0
    heterogeneous_state_reference_max_abs = 0.0
    heterogeneous_argmax_parity = True
    heterogeneous_layer_max_abs: dict[int, list[float]] = {
        layer_idx: [0.0, 0.0] for layer_idx in batch_captures
    }
    solo_commit = True
    solo_capture_layers = 30
    for row, row_token in enumerate(batch_tokens):
        (
            solo_input,
            solo_logits,
            solo_hidden,
            solo_captures,
            row_commit,
            _solo_cache,
        ) = run(
            [row_token],
            capture_forward=solo_capture_forward,
            target_forward=solo_target_forward,
            draft_forward=solo_draft_forward,
            verify_input_override=batch_input[row : row + 1],
        )
        comparisons = [
            mx.all(batch_input[row : row + 1] == solo_input),
            mx.all(batch_logits[row : row + 1] == solo_logits),
            mx.all(batch_hidden[row : row + 1] == solo_hidden),
        ]
        row_errors = [
            mx.max(
                mx.abs(batch_logits[row : row + 1] - solo_logits).astype(
                    mx.float32
                )
            ),
            mx.max(
                mx.abs(batch_hidden[row : row + 1] - solo_hidden).astype(
                    mx.float32
                )
            ),
        ]
        row_reference_max = [
            mx.max(mx.abs(solo_logits).astype(mx.float32)),
            mx.max(mx.abs(solo_hidden).astype(mx.float32)),
        ]
        row_argmax_parity = mx.all(
            mx.argmax(batch_logits[row : row + 1], axis=-1)
            == mx.argmax(solo_logits, axis=-1)
        )
        for layer_idx in batch_captures:
            comparisons.extend(
                (
                    mx.all(
                        batch_captures[layer_idx]["conv_states"][row : row + 1]
                        == solo_captures[layer_idx]["conv_states"]
                    ),
                    mx.all(
                        batch_captures[layer_idx]["states"][row : row + 1]
                        == solo_captures[layer_idx]["states"]
                    ),
                )
            )
            row_errors.extend(
                (
                    mx.max(
                        mx.abs(
                            batch_captures[layer_idx]["conv_states"][
                                row : row + 1
                            ]
                            - solo_captures[layer_idx]["conv_states"]
                        ).astype(mx.float32)
                    ),
                    mx.max(
                        mx.abs(
                            batch_captures[layer_idx]["states"][row : row + 1]
                            - solo_captures[layer_idx]["states"]
                        ).astype(mx.float32)
                    ),
                )
            )
            row_reference_max.extend(
                (
                    mx.max(
                        mx.abs(
                            solo_captures[layer_idx]["conv_states"]
                        ).astype(mx.float32)
                    ),
                    mx.max(
                        mx.abs(solo_captures[layer_idx]["states"]).astype(
                            mx.float32
                        )
                    ),
                )
            )
        mx.eval(*comparisons, *row_errors, *row_reference_max, row_argmax_parity)
        error_values = [float(np.asarray(value).item()) for value in row_errors]
        reference_values = [
            float(np.asarray(value).item()) for value in row_reference_max
        ]
        heterogeneous_argmax_parity = bool(
            heterogeneous_argmax_parity
            and bool(np.asarray(row_argmax_parity).item())
        )
        heterogeneous_row_parity = bool(
            heterogeneous_row_parity
            and all(bool(np.asarray(value).item()) for value in comparisons)
        )
        heterogeneous_row_max_abs = max(
            heterogeneous_row_max_abs,
            *error_values,
        )
        heterogeneous_logits_max_abs = max(
            heterogeneous_logits_max_abs, error_values[0]
        )
        heterogeneous_hidden_max_abs = max(
            heterogeneous_hidden_max_abs, error_values[1]
        )
        heterogeneous_conv_max_abs = max(
            heterogeneous_conv_max_abs,
            *error_values[2::2],
        )
        heterogeneous_state_max_abs = max(
            heterogeneous_state_max_abs,
            *error_values[3::2],
        )
        heterogeneous_logits_reference_max_abs = max(
            heterogeneous_logits_reference_max_abs, reference_values[0]
        )
        heterogeneous_hidden_reference_max_abs = max(
            heterogeneous_hidden_reference_max_abs, reference_values[1]
        )
        heterogeneous_conv_reference_max_abs = max(
            heterogeneous_conv_reference_max_abs,
            *reference_values[2::2],
        )
        heterogeneous_state_reference_max_abs = max(
            heterogeneous_state_reference_max_abs,
            *reference_values[3::2],
        )
        for capture_position, layer_idx in enumerate(batch_captures):
            conv_error = error_values[2 + 2 * capture_position]
            state_error = error_values[3 + 2 * capture_position]
            layer_errors = heterogeneous_layer_max_abs[layer_idx]
            layer_errors[0] = max(layer_errors[0], conv_error)
            layer_errors[1] = max(layer_errors[1], state_error)
        solo_commit = bool(solo_commit and row_commit)
        solo_capture_layers = min(solo_capture_layers, len(solo_captures))
    heterogeneous_relative_errors = {
        "logits": heterogeneous_logits_max_abs
        / max(1.0, heterogeneous_logits_reference_max_abs),
        "hidden": heterogeneous_hidden_max_abs
        / max(1.0, heterogeneous_hidden_reference_max_abs),
        "conv": heterogeneous_conv_max_abs
        / max(1.0, heterogeneous_conv_reference_max_abs),
        "state": heterogeneous_state_max_abs
        / max(1.0, heterogeneous_state_reference_max_abs),
    }
    b8_t2_gdn_numerical_parity = all(
        same_geometry_relative_errors[name] <= _BF16_GEOMETRY_RELATIVE_LIMIT
        for name in ("conv", "state")
    )
    heterogeneous_numerical_parity = bool(
        heterogeneous_argmax_parity
        and all(
            value <= _BF16_GEOMETRY_RELATIVE_LIMIT
            for value in heterogeneous_relative_errors.values()
        )
    )
    empty_mtp_draft_numerical_parity = bool(
        empty_mtp_draft_argmax_parity
        and empty_mtp_draft_relative_error
        <= _BF16_GEOMETRY_RELATIVE_LIMIT
    )
    heterogeneous_row_parity = heterogeneous_numerical_parity
    empty_mtp_draft_parity = empty_mtp_draft_numerical_parity

    isolation_tokens = list(batch_tokens)
    isolation_tokens[0] = int(
        (isolation_tokens[0] + 97) % lane.geometry.vocab_size
    )
    isolation_verify_input = mx.concatenate(
        (
            mx.array(
                [
                    [
                        (batch_tokens[0] + 193) % lane.geometry.vocab_size,
                        (batch_tokens[0] + 389) % lane.geometry.vocab_size,
                    ]
                ],
                dtype=mx.int32,
            ),
            batch_input[1:],
        ),
        axis=0,
    )
    (
        _isolation_input,
        isolation_logits,
        isolation_hidden,
        isolation_captures,
        isolation_commit,
        _isolation_cache,
    ) = run(
        isolation_tokens,
        capture_forward=lane.capture_forward,
        verify_input_override=isolation_verify_input,
    )
    row_isolation_checks = [
        mx.all(batch_logits[1:] == isolation_logits[1:]),
        mx.all(batch_hidden[1:] == isolation_hidden[1:]),
    ]
    for layer_idx in batch_captures:
        row_isolation_checks.extend(
            (
                mx.all(
                    batch_captures[layer_idx]["conv_states"][1:]
                    == isolation_captures[layer_idx]["conv_states"][1:]
                ),
                mx.all(
                    batch_captures[layer_idx]["states"][1:]
                    == isolation_captures[layer_idx]["states"][1:]
                ),
            )
        )
    mx.eval(*row_isolation_checks)
    row_isolation_parity = bool(
        isolation_commit
        and all(bool(np.asarray(value).item()) for value in row_isolation_checks)
    )
    solo_parity = heterogeneous_numerical_parity
    fixed_row_commit = all(
        layer_idx in batch_captures
        and "tape" not in batch_captures[layer_idx]
        and int(batch_captures[layer_idx].get("capture_start", 0)) == 0
        and tuple(batch_captures[layer_idx]["conv_states"].shape[:2]) == (8, 2)
        and tuple(batch_captures[layer_idx]["states"].shape[:2]) == (8, 2)
        for layer_idx, layer_type in enumerate(_LAYER_TYPES)
        if layer_type == "linear_attention"
    )
    return {
        "ok": bool(
            target_shape == [8, 2]
            and logits_shape[:2] == [8, 2]
            and hidden_shape[:2] == [8, 2]
            and solo_parity
            and batch_commit
            and solo_commit
            and len(batch_captures) == 30
            and len(solo_captures) == 30
            and fixed_row_commit
            and heterogeneous_numerical_parity
            and heterogeneous_argmax_parity
            and b8_t2_gdn_numerical_parity
            and compiled_eager_numerical_parity
            and compiled_eager_argmax_parity
            and compiled_eager_offset_parity
            and same_geometry_numerical_parity
            and same_geometry_argmax_parity
            and same_geometry_attention_parity
            and mixed_commit_parity
            and prefill_contract
            and prefill_numerical_parity
            and empty_mtp_draft_numerical_parity
            and empty_mtp_draft_argmax_parity
            and empty_mtp_row_isolation_parity
            and row_isolation_parity
        ),
        "target_shape": target_shape,
        "logits_shape": logits_shape,
        "hidden_shape": hidden_shape,
        "projection_rows": 16,
        "solo_parity": solo_parity,
        "heterogeneous_row_parity": heterogeneous_row_parity,
        "heterogeneous_numerical_parity": heterogeneous_numerical_parity,
        "b8_t2_gdn_numerical_parity": b8_t2_gdn_numerical_parity,
        "compiled_eager_bitwise_parity": compiled_eager_bitwise_parity,
        "compiled_eager_numerical_parity": compiled_eager_numerical_parity,
        "compiled_eager_argmax_parity": compiled_eager_argmax_parity,
        "compiled_eager_offset_parity": compiled_eager_offset_parity,
        "compiled_eager_output_bitwise_parity": all(
            compiled_eager_check_values[: len(compiled_eager_output_checks)]
        ),
        "compiled_eager_capture_bitwise_parity": all(
            compiled_eager_check_values[
                len(compiled_eager_output_checks) :
                len(compiled_eager_output_checks)
                + len(compiled_eager_capture_checks)
            ]
        ),
        "compiled_eager_cache_bitwise_parity": all(
            compiled_eager_check_values[-len(compiled_eager_cache_checks) :]
        ),
        "compiled_eager_relative_errors": compiled_eager_relative_errors,
        "compiled_eager_failed_checks": [
            index
            for index, passed in enumerate(compiled_eager_check_values)
            if not passed
        ],
        "same_geometry_numerical_parity": same_geometry_numerical_parity,
        "same_geometry_argmax_parity": same_geometry_argmax_parity,
        "same_geometry_attention_parity": same_geometry_attention_parity,
        "stock_b8_unchanged_moe_reference": True,
        "same_geometry_relative_errors": same_geometry_relative_errors,
        "captured_gdn_layers": len(batch_captures),
        "solo_captured_gdn_layers": solo_capture_layers,
        "row_commit": bool(batch_commit and solo_commit),
        "fixed_row_commit": fixed_row_commit,
        "mixed_commit_parity": mixed_commit_parity,
        "prefill_contract": prefill_contract,
        "prefill_numerical_parity": prefill_numerical_parity,
        "empty_mtp_draft_parity": empty_mtp_draft_parity,
        "empty_mtp_draft_numerical_parity": (
            empty_mtp_draft_numerical_parity
        ),
        "empty_mtp_draft_max_abs": empty_mtp_draft_max_abs,
        "empty_mtp_draft_reference_max_abs": (
            empty_mtp_draft_reference_max_abs
        ),
        "empty_mtp_draft_relative_error": empty_mtp_draft_relative_error,
        "empty_mtp_draft_argmax_parity": empty_mtp_draft_argmax_parity,
        "empty_mtp_row_isolation_parity": empty_mtp_row_isolation_parity,
        "heterogeneous_row_max_abs": heterogeneous_row_max_abs,
        "heterogeneous_logits_max_abs": heterogeneous_logits_max_abs,
        "heterogeneous_hidden_max_abs": heterogeneous_hidden_max_abs,
        "heterogeneous_conv_max_abs": heterogeneous_conv_max_abs,
        "heterogeneous_state_max_abs": heterogeneous_state_max_abs,
        "heterogeneous_logits_reference_max_abs": (
            heterogeneous_logits_reference_max_abs
        ),
        "heterogeneous_hidden_reference_max_abs": (
            heterogeneous_hidden_reference_max_abs
        ),
        "heterogeneous_conv_reference_max_abs": (
            heterogeneous_conv_reference_max_abs
        ),
        "heterogeneous_state_reference_max_abs": (
            heterogeneous_state_reference_max_abs
        ),
        "heterogeneous_argmax_parity": heterogeneous_argmax_parity,
        "heterogeneous_relative_errors": heterogeneous_relative_errors,
        "row_isolation_parity": row_isolation_parity,
        "heterogeneous_layer_max_abs": {
            str(layer_idx): values
            for layer_idx, values in heterogeneous_layer_max_abs.items()
        },
    }


def _prefill_qwen35b_batch_request(
    prompt_ids: list[int],
    *,
    target_forward: Callable[..., Any],
    target_cache_factory: Callable[[], list[Any]],
    mtp_cache_factory: Callable[[], list[Any]],
    update_mtp_cache: Callable[..., Any],
    chunk_size: int,
    cleanup_every: int,
    abort_check: Callable[[], bool] | None = None,
) -> tuple[Any, Any, Any, Any, float, float, int]:
    """Prefill one fixed-lane row without generic runtime policy dispatch."""
    import mlx.core as mx

    from .attention_context import attention_phase
    from .generation import _check_postcommit_abort

    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    _check_postcommit_abort(abort_check)
    cache = target_cache_factory()
    mtp_cache = mtp_cache_factory()
    body = prompt_ids[:-1]
    if body:
        body_array = mx.array([body], dtype=mx.int32)
        chunk_index = 0
        for start in range(0, len(body), chunk_size):
            _check_postcommit_abort(abort_check)
            end = min(len(body), start + chunk_size)
            with attention_phase("prefill"):
                _logits, hidden = target_forward(
                    body_array[:, start:end],
                    cache=cache,
                    return_hidden=True,
                    hidden_variant="post_norm",
                )
            mx.eval(hidden)
            _check_postcommit_abort(abort_check)
            history_ids = mx.array([prompt_ids[start + 1 : end + 1]], dtype=mx.int32)
            with attention_phase("prefill"):
                history_hidden = update_mtp_cache(
                    hidden,
                    history_ids,
                    mtp_cache=mtp_cache,
                    position_offset=None,
                )
            mx.eval(history_hidden)
            del _logits, hidden, history_hidden
            chunk_index += 1
            if cleanup_every > 0 and chunk_index % cleanup_every == 0:
                mx.synchronize()
                mx.clear_cache()
            _check_postcommit_abort(abort_check)

    with attention_phase("prefill"):
        logits, hidden = target_forward(
            mx.array([[prompt_ids[-1]]], dtype=mx.int32),
            cache=cache,
            return_hidden=True,
            hidden_variant="post_norm",
        )
    mx.eval(logits, hidden)
    _check_postcommit_abort(abort_check)
    return (
        cache,
        logits[:, -1, :],
        hidden[:, -1:, :],
        mtp_cache,
        0.0,
        0.0,
        0,
    )


def _bind_qwen35b_batch_prefill(
    runtime: Any,
    *,
    update_mtp_cache: Callable[..., Any],
    chunk_size: int,
) -> Callable[..., Any]:
    model = getattr(runtime, "model", None)
    if not callable(model):
        raise A3BMTPBatchInstallError(
            "Qwen 35B mtp_batch requires a callable target model"
        )
    language_model = getattr(model, "language_model", None)
    target_cache_factory = getattr(language_model, "make_cache", None)
    mtp_cache_factory = getattr(model, "make_mtp_cache", None)
    if not callable(target_cache_factory):
        raise A3BMTPBatchInstallError(
            "Qwen 35B mtp_batch requires a direct target cache factory"
        )
    if not callable(mtp_cache_factory):
        raise A3BMTPBatchInstallError(
            "Qwen 35B mtp_batch requires a direct MTP cache factory"
        )
    from .generation import _prefill_chunk_cache_cleanup_enabled

    cleanup_every = (
        A3BMTPBatchGeometry.prefill_cleanup_every
        if _prefill_chunk_cache_cleanup_enabled()
        else 0
    )
    return partial(
        _prefill_qwen35b_batch_request,
        target_forward=model,
        target_cache_factory=target_cache_factory,
        mtp_cache_factory=mtp_cache_factory,
        update_mtp_cache=update_mtp_cache,
        chunk_size=int(chunk_size),
        cleanup_every=cleanup_every,
    )


def install_a3b_mtp_batch_lane(
    runtime: Any,
    *,
    selfcheck: Callable[[InstalledA3BMTPBatchLane], Mapping[str, Any]] | None = None,
) -> InstalledA3BMTPBatchLane:
    """Validate and freeze the exact Qwen 35B B8/T2 route once at startup."""

    _config, fingerprint = _validate_config(runtime)
    _validate_runtime(runtime)
    model_target_forward = _require_callable(runtime, "model")
    model_capture_forward = _bind_capture_forward(runtime)
    model_draft_forward = _require_callable(runtime.model, "mtp_forward")
    model_update_mtp_cache = _require_callable(runtime.model, "mtp_update_cache")
    language_model = getattr(runtime.model, "language_model", None)
    make_cache = _require_callable(language_model, "make_cache")
    make_mtp_cache = _require_callable(runtime.model, "make_mtp_cache")
    attention_route_id = _install_qwen35b_b8_attention_route(runtime)
    from .generation import _prefill_chunk_size

    prefill_chunk_tokens = int(_prefill_chunk_size())
    if prefill_chunk_tokens > A3BMTPBatchGeometry.prefill_chunk_tokens:
        raise A3BMTPBatchInstallError(
            "Qwen 35B mtp_batch prefill chunk must be <= "
            f"{A3BMTPBatchGeometry.prefill_chunk_tokens} tokens; "
            f"got {prefill_chunk_tokens}"
        )

    model_draft_forward = partial(
        model_draft_forward,
        concat_order=getattr(runtime.contract, "concat_order", None),
        return_hidden=False,
        mtp_hidden_variant="post_norm",
    )
    model_update_mtp_cache = partial(
        model_update_mtp_cache,
        concat_order=getattr(runtime.contract, "concat_order", None),
    )
    target_forward = partial(
        _call_with_qwen35b_mtp_batch_attention,
        call=model_target_forward,
    )
    capture_forward = partial(
        _call_with_qwen35b_mtp_batch_attention,
        call=model_capture_forward,
    )
    draft_forward = partial(
        _call_with_qwen35b_mtp_batch_attention,
        call=model_draft_forward,
    )
    update_mtp_cache = partial(
        _call_with_qwen35b_mtp_batch_attention,
        call=model_update_mtp_cache,
    )
    prefill_request = _bind_qwen35b_batch_prefill(
        runtime,
        update_mtp_cache=update_mtp_cache,
        chunk_size=prefill_chunk_tokens,
    )
    geometry = A3BMTPBatchGeometry()
    lane = InstalledA3BMTPBatchLane(
        geometry=geometry,
        route_id="qwen35b_a3b_mtp_batch_b8_t2_m16",
        attention_route_id=attention_route_id,
        config_fingerprint=fingerprint,
        target_forward=target_forward,
        capture_forward=capture_forward,
        draft_forward=draft_forward,
        update_mtp_cache=update_mtp_cache,
        commit_rows=_commit_qwen35b_b8_t2_rows,
        prefill_request=prefill_request,
        merge_target_caches=_merge_qwen35b_target_caches,
        merge_mtp_caches=_merge_qwen35b_mtp_caches,
        make_cache=make_cache,
        make_mtp_cache=make_mtp_cache,
        selfcheck=MappingProxyType({}),
    )
    report = dict(
        selfcheck(lane) if selfcheck is not None else _default_selfcheck(lane, runtime)
    )
    if (
        not bool(report.get("ok"))
        or report.get("target_shape") != [8, 2]
        or int(report.get("projection_rows", 0) or 0) != 16
        or not bool(report.get("solo_parity"))
        or int(report.get("captured_gdn_layers", 0) or 0) != 30
        or not bool(report.get("row_commit"))
        or not bool(report.get("fixed_row_commit"))
        or not bool(report.get("heterogeneous_row_parity"))
        or not bool(report.get("heterogeneous_numerical_parity"))
        or not bool(report.get("heterogeneous_argmax_parity"))
        or not bool(report.get("b8_t2_gdn_numerical_parity"))
        or not bool(report.get("compiled_eager_numerical_parity"))
        or not bool(report.get("compiled_eager_argmax_parity"))
        or not bool(report.get("compiled_eager_offset_parity"))
        or not bool(report.get("same_geometry_numerical_parity"))
        or not bool(report.get("same_geometry_argmax_parity"))
        or not bool(report.get("same_geometry_attention_parity"))
        or not bool(report.get("stock_b8_unchanged_moe_reference"))
        or not bool(report.get("mixed_commit_parity"))
        or not bool(report.get("prefill_contract"))
        or not bool(report.get("prefill_numerical_parity"))
        or not bool(report.get("empty_mtp_draft_parity"))
        or not bool(report.get("empty_mtp_draft_numerical_parity"))
        or not bool(report.get("empty_mtp_draft_argmax_parity"))
        or not bool(report.get("empty_mtp_row_isolation_parity"))
        or not bool(report.get("row_isolation_parity"))
    ):
        raise A3BMTPBatchInstallError(
            "Qwen 35B mtp_batch numerical self-check failed: "
            + json.dumps(report, sort_keys=True, default=str)
        )
    return InstalledA3BMTPBatchLane(
        geometry=geometry,
        route_id=lane.route_id,
        attention_route_id=attention_route_id,
        config_fingerprint=fingerprint,
        target_forward=target_forward,
        capture_forward=capture_forward,
        draft_forward=draft_forward,
        update_mtp_cache=update_mtp_cache,
        commit_rows=_commit_qwen35b_b8_t2_rows,
        prefill_request=prefill_request,
        merge_target_caches=_merge_qwen35b_target_caches,
        merge_mtp_caches=_merge_qwen35b_mtp_caches,
        make_cache=make_cache,
        make_mtp_cache=make_mtp_cache,
        selfcheck=MappingProxyType(report),
    )


def _merge_qwen35b_kv_rows(
    caches: list[list[Any]],
    layer_idx: int,
    *,
    allow_empty: bool,
) -> Any:
    entries = [cache[layer_idx] for cache in caches]
    offsets = [int(entry.offset) for entry in entries]
    populated = [entry for entry in entries if entry.keys is not None]
    if not populated:
        if not allow_empty:
            raise RuntimeError("Qwen 35B target prefill produced an empty KV layer")
        keys = mx.zeros((len(entries), 2, 0, 256), dtype=mx.bfloat16)
        values = mx.zeros((len(entries), 2, 0, 256), dtype=mx.bfloat16)
    else:
        template = populated[0]
        capacity = max(int(entry.keys.shape[2]) for entry in populated)
        key_rows = []
        value_rows = []
        for entry in entries:
            keys = entry.keys
            values = entry.values
            if keys is None:
                keys = mx.zeros((1, 2, capacity, 256), dtype=template.keys.dtype)
                values = mx.zeros(
                    (1, 2, capacity, 256), dtype=template.values.dtype
                )
            elif int(keys.shape[2]) < capacity:
                pad = capacity - int(keys.shape[2])
                keys = mx.concatenate(
                    (keys, mx.zeros((1, 2, pad, 256), dtype=keys.dtype)), axis=2
                )
                values = mx.concatenate(
                    (values, mx.zeros((1, 2, pad, 256), dtype=values.dtype)),
                    axis=2,
                )
            key_rows.append(keys)
            value_rows.append(values)
        keys = mx.concatenate(key_rows, axis=0)
        values = mx.concatenate(value_rows, axis=0)
    merged = RaggedBatchKVCache(
        batch_size=8,
        step=256,
        keys=keys,
        values=values,
        offsets=mx.array(offsets, dtype=mx.int32),
    )
    merged._capacity_bound = max(offsets)
    mx.eval(merged.keys, merged.values, merged.offsets)
    for source in caches:
        source[layer_idx] = None
    return merged


def _merge_qwen35b_target_caches(caches: list[list[Any]]) -> list[Any]:
    """Execute the installed 30 ArraysCache + 10 KVCache merge table."""
    merged_cache: list[Any] = []
    for layer_idx, layer_type in enumerate(_LAYER_TYPES):
        if layer_type == "full_attention":
            merged_cache.append(
                _merge_qwen35b_kv_rows(caches, layer_idx, allow_empty=False)
            )
            continue
        merged = ArraysCache(2)
        merged[0] = mx.concatenate(
            [source[layer_idx][0] for source in caches], axis=0
        )
        merged[1] = mx.concatenate(
            [source[layer_idx][1] for source in caches], axis=0
        )
        mx.eval(merged[0], merged[1])
        merged_cache.append(merged)
        for source in caches:
            source[layer_idx] = None
    return merged_cache


def _merge_qwen35b_mtp_caches(caches: list[list[Any]]) -> list[Any]:
    """Execute the installed one-layer Qwen MTP KV merge table."""
    return [_merge_qwen35b_kv_rows(caches, 0, allow_empty=True)]


def generate_a3b_mtp_batch(
    lane: InstalledA3BMTPBatchLane,
    requests: list[A3BMTPBatchRequest] | tuple[A3BMTPBatchRequest, ...],
) -> A3BMTPBatchResult:
    """Generate one immutable 2-8 request cohort through the fixed B8/T2 lane."""
    import mlx.core as mx

    from .attention_context import attention_phase
    from .batched_decode import (
        _finish_mtp_k1_row_cycle,
        _sample_mtp_k1_draft,
        _sample_mtp_k1_primary,
    )
    from .ragged_kv_cache import RaggedBatchKVCache

    real = list(requests)
    width = lane.geometry.cohort_slots
    if not 2 <= len(real) <= width:
        raise ValueError("Qwen 35B mtp_batch requires 2-8 requests per cohort")
    for request in real:
        if not request.prompt_ids:
            raise ValueError("Qwen 35B mtp_batch prompts must not be empty")
        if int(request.max_tokens) < 1:
            raise ValueError("Qwen 35B mtp_batch max_tokens must be >= 1")
        if len(request.prompt_ids) + int(request.max_tokens) > int(
            lane.geometry.max_context_tokens
        ):
            raise A3BMTPBatchCapacityError(
                "Qwen 35B mtp_batch requires prompt_tokens + max_tokens <= "
                f"{lane.geometry.max_context_tokens}"
            )

    slots: list[A3BMTPBatchRequest | None] = [*real, *([None] * (width - len(real)))]
    finish: list[str | None] = [
        "cancelled" if request.cancelled() else None for request in real
    ]
    terminal_notified = [False for _ in real]
    replacement_rows: set[int] = set()

    def notify_terminal(row: int, cycle_count: int) -> None:
        if terminal_notified[row] or finish[row] is None:
            return
        terminal_notified[row] = True
        callback = real[row].on_terminal
        if callback is not None:
            callback(str(finish[row]), int(cycle_count))

    def poll_prefill_cancellations(current_row: int) -> bool:
        for peer_row, peer in enumerate(real):
            if finish[peer_row] is None and peer.cancelled():
                finish[peer_row] = "cancelled"
                notify_terminal(peer_row, 0)
                if peer_row < current_row:
                    replacement_rows.add(peer_row)
        return current_row < len(real) and finish[current_row] == "cancelled"

    prefills: list[tuple[Any, Any, Any, Any]] = []
    prefill_prompt_lengths: list[int] = []
    for row, request in enumerate(slots):
        if request is not None and row < len(real) and finish[row] is not None:
            notify_terminal(row, 0)
        prompt = [0] if request is None or request.cancelled() else list(request.prompt_ids)
        try:
            cache, logits, hidden, mtp_cache, *_timing = lane.prefill_request(
                prompt,
                abort_check=lambda row=row: poll_prefill_cancellations(row),
            )
        except Exception as exc:
            from .generation import PostcommitAbort

            if (
                request is None
                or row >= len(real)
                or not isinstance(exc, PostcommitAbort)
                or not request.cancelled()
            ):
                raise
            finish[row] = "cancelled"
            notify_terminal(row, 0)
            prompt = [0]
            cache, logits, hidden, mtp_cache, *_timing = lane.prefill_request(
                [0], abort_check=None
            )
        for peer_row in sorted(replacement_rows):
            replacement = lane.prefill_request([0], abort_check=None)
            prefills[peer_row] = tuple(replacement[:4])
            prefill_prompt_lengths[peer_row] = 1
        replacement_rows.clear()
        prefill_prompt_lengths.append(len(prompt))
        prefills.append((cache, logits, hidden, mtp_cache))

    # Close the cancellation race after the final prefill callback and before
    # any row is admitted to the merged B8 cache.
    poll_prefill_cancellations(len(real))
    for peer_row in sorted(replacement_rows):
        replacement = lane.prefill_request([0], abort_check=None)
        prefills[peer_row] = tuple(replacement[:4])
        prefill_prompt_lengths[peer_row] = 1

    cache = lane.merge_target_caches([item[0] for item in prefills])
    mtp_cache = lane.merge_mtp_caches([item[3] for item in prefills])
    logits_last = mx.concatenate([item[1] for item in prefills], axis=0)
    hidden_last = mx.concatenate([item[2] for item in prefills], axis=0)
    mx.eval(logits_last, hidden_last)
    del prefills
    for request in real:
        if request.on_decode_start is not None:
            request.on_decode_start()

    rngs = [np.random.default_rng(request.seed) for request in real]
    tokens: list[list[int]] = [[] for _ in real]
    pending: list[int | None] = [None for _ in real]
    accepted_drafts = 0
    rejected_drafts = 0
    cycles = 0
    max_cycles = max(int(request.max_tokens) for request in real) + 2

    def active(row: int) -> bool:
        return row < len(real) and finish[row] is None

    target_ragged_entries = [
        entry for entry in cache if isinstance(entry, RaggedBatchKVCache)
    ]
    target_recurrent_entries = [
        (layer_idx, entry)
        for layer_idx, entry in enumerate(cache)
        if not isinstance(entry, RaggedBatchKVCache)
    ]
    mtp_ragged_entries = [
        entry for entry in mtp_cache if isinstance(entry, RaggedBatchKVCache)
    ]
    target_row_bounds = list(prefill_prompt_lengths)
    mtp_row_bounds = [max(0, value - 1) for value in prefill_prompt_lengths]

    def install_host_bounds(entries: list[Any], row_bounds: list[int]) -> None:
        capacity_bound = max(row_bounds)
        for entry in entries:
            entry._capacity_bound = capacity_bound

    install_host_bounds(target_ragged_entries, target_row_bounds)
    install_host_bounds(mtp_ragged_entries, mtp_row_bounds)

    while any(reason is None for reason in finish):
        if cycles >= max_cycles:
            raise RuntimeError("Qwen 35B mtp_batch exceeded its bounded cycle count")
        for row, request in enumerate(real):
            if finish[row] is None and request.cancelled():
                finish[row] = "cancelled"
                notify_terminal(row, cycles)
        if not any(reason is None for reason in finish):
            break
        primary_rows = np.asarray(logits_last.astype(mx.float32))
        primary_ids = [0] * width
        primary_was_pending = [False] * width
        may_finish_cycle = [False] * width
        cycle_tokens: list[list[int]] = [[] for _ in range(width)]
        for row in range(width):
            if not active(row):
                continue
            request = real[row]
            was_pending = pending[row] is not None
            primary = _sample_mtp_k1_primary(
                primary_rows[row],
                sampler=request.sampler,
                rng=rngs[row],
                history_tokens=tokens[row],
                pending_primary=pending[row],
            )
            primary_ids[row] = primary
            primary_was_pending[row] = was_pending
            history_after_primary = list(tokens[row])
            if not was_pending:
                cycle_tokens[row].append(primary)
                history_after_primary.append(primary)
            may_finish_cycle[row] = (
                len(history_after_primary) < int(request.max_tokens)
                and primary not in request.stop_token_ids
            )

        primary_array = mx.array(primary_ids, dtype=mx.int32)
        mtp_offsets_before_primary = [
            entry.offsets for entry in mtp_ragged_entries
        ]
        for entry in mtp_ragged_entries:
            entry.reserve(1)
        with attention_phase("ar_decode"):
            draft_logits = lane.draft_forward(
                hidden_last,
                primary_array[:, None],
                mtp_cache=mtp_cache,
            )
        primary_append_mask_values = [active(row) for row in range(width)]
        primary_append_mask = mx.array(
            primary_append_mask_values, dtype=mx.bool_
        )
        for entry, before_offsets in zip(
            mtp_ragged_entries, mtp_offsets_before_primary, strict=True
        ):
            entry.offsets = mx.where(
                primary_append_mask, entry.offsets, before_offsets
            ).astype(mx.int32)
        mtp_row_bounds = [
            bound + int(append)
            for bound, append in zip(
                mtp_row_bounds, primary_append_mask_values, strict=True
            )
        ]
        install_host_bounds(mtp_ragged_entries, mtp_row_bounds)
        mx.eval(draft_logits)
        draft_rows = np.asarray(draft_logits[:, -1, :].astype(mx.float32))
        proposals: list[Any | None] = [None] * width
        draft_ids = [0] * width
        for row in range(width):
            if active(row) and may_finish_cycle[row]:
                request = real[row]
                proposal = _sample_mtp_k1_draft(
                    primary_ids[row],
                    draft_rows[row],
                    draft_sampler=request.draft_sampler,
                    rng=rngs[row],
                )
                proposals[row] = proposal
                draft_ids[row] = proposal.draft_token
            else:
                draft_ids[row] = int(np.argmax(draft_rows[row]))

        verify_input = mx.stack(
            (primary_array, mx.array(draft_ids, dtype=mx.int32)), axis=1
        )
        base_recurrent = {
            layer_idx: (entry[0], entry[1])
            for layer_idx, entry in target_recurrent_entries
        }
        for entry in target_ragged_entries:
            entry.reserve(lane.geometry.verify_tokens)
        with attention_phase("decode_verify"):
            verify_logits, verify_hidden, captures = lane.capture_forward(
                verify_input, cache=cache
            )
        mx.eval(verify_logits)
        verify_rows = np.asarray(verify_logits.astype(mx.float32))
        keeps = [0] * width
        accepted_mask = [False] * width
        next_pending: list[int | None] = [None] * len(real)

        for row, proposal in enumerate(proposals):
            if proposal is None or not active(row):
                if active(row):
                    keeps[row] = 1
                    accepted_mask[row] = False
                continue
            request = real[row]
            history_after_primary = list(tokens[row])
            if not primary_was_pending[row]:
                history_after_primary.append(primary_ids[row])
            bonus_allowed = (
                not request.omit_speculative_bonus
                and len(history_after_primary) + 1 < int(request.max_tokens)
                and proposal.draft_token not in request.stop_token_ids
            )
            decision = _finish_mtp_k1_row_cycle(
                proposal,
                verify_rows[row, 0],
                verify_rows[row, 1] if bonus_allowed else None,
                sampler=request.sampler,
                rng=rngs[row],
                history_tokens=history_after_primary,
                omit_speculative_bonus=not bonus_allowed,
            )
            accepted_mask[row] = decision.accepted
            keeps[row] = 2 if decision.accepted else 1
            accepted_drafts += int(decision.accepted)
            rejected_drafts += int(not decision.accepted)
            cycle_tokens[row].append(decision.second_token)
            if decision.bonus_token is not None:
                cycle_tokens[row].append(decision.bonus_token)
            next_pending[row] = decision.next_primary

        lane.commit_rows(cache, captures, keeps, base_recurrent)
        target_row_bounds = [
            bound + int(keep)
            for bound, keep in zip(target_row_bounds, keeps, strict=True)
        ]
        install_host_bounds(target_ragged_entries, target_row_bounds)

        append_mask = mx.array(
            [
                bool(
                    row < len(real)
                    and proposals[row] is not None
                    and accepted_mask[row]
                )
                for row in range(width)
            ],
            dtype=mx.bool_,
        )
        mtp_offsets_before_append = [
            entry.offsets for entry in mtp_ragged_entries
        ]
        for entry in mtp_ragged_entries:
            entry.reserve(1)
        with attention_phase("ar_decode"):
            lane.update_mtp_cache(
                verify_hidden[:, 0:1, :],
                mx.array(draft_ids, dtype=mx.int32)[:, None],
                mtp_cache=mtp_cache,
            )
        for entry, before_offsets in zip(
            mtp_ragged_entries, mtp_offsets_before_append, strict=True
        ):
            entry.offsets = mx.where(
                append_mask, entry.offsets, before_offsets
            ).astype(mx.int32)
        append_mask_values = [
            bool(
                row < len(real)
                and proposals[row] is not None
                and accepted_mask[row]
            )
            for row in range(width)
        ]
        mtp_row_bounds = [
            bound + int(append)
            for bound, append in zip(
                mtp_row_bounds, append_mask_values, strict=True
            )
        ]
        install_host_bounds(mtp_ragged_entries, mtp_row_bounds)

        accept_array = mx.array(accepted_mask).reshape(width, 1)
        logits_last = mx.where(
            accept_array, verify_logits[:, 1, :], verify_logits[:, 0, :]
        )
        hidden_last = mx.where(
            accept_array[:, :, None],
            verify_hidden[:, 1:2, :],
            verify_hidden[:, 0:1, :],
        )

        for row, request in enumerate(real):
            if finish[row] is not None:
                continue
            if request.cancelled():
                finish[row] = "cancelled"
                pending[row] = None
                continue
            for token in cycle_tokens[row]:
                if request.cancelled():
                    finish[row] = "cancelled"
                    break
                tokens[row].append(int(token))
                if request.on_token is not None:
                    request.on_token(int(token))
                if request.cancelled():
                    finish[row] = "cancelled"
                    break
                if int(token) in request.stop_token_ids:
                    finish[row] = "stop"
                    break
                if len(tokens[row]) >= int(request.max_tokens):
                    finish[row] = "length"
                    break
            pending[row] = next_pending[row] if finish[row] is None else None
            notify_terminal(row, cycles + 1)
        cycles += 1

    for row in range(len(real)):
        notify_terminal(row, cycles)

    return A3BMTPBatchResult(
        streams=tuple(
            A3BMTPBatchStreamResult(
                request_id=request.request_id,
                tokens=tuple(tokens[row]),
                finish_reason=str(finish[row]),
            )
            for row, request in enumerate(real)
        ),
        cycles=cycles,
        accepted_drafts=accepted_drafts,
        rejected_drafts=rejected_drafts,
        route_id=lane.route_id,
        width_histogram=MappingProxyType({width: cycles}),
    )
