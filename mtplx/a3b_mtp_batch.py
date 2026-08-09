"""Construction-time contract for Qwen3.6-35B-A3B eight-row MTP decode."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Any

import numpy as np

from mtplx.artifacts import load_config
from mtplx.sampling import SamplerConfig


_LAYER_TYPES = tuple(
    "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
    for index in range(40)
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
    hidden_layers: int = 40
    experts: int = 256
    experts_per_token: int = 8
    body_quant_bits: int = 4
    body_quant_group_size: int = 64
    mtp_quant_bits: int = 4
    mtp_quant_group_size: int = 32
    max_context_tokens: int = 131072


@dataclass(frozen=True)
class InstalledA3BMTPBatchLane:
    """Prevalidated, prebound fixed-shape lane used directly by serving."""

    geometry: A3BMTPBatchGeometry
    route_id: str
    config_fingerprint: str
    target_forward: Callable[..., Any]
    capture_forward: Callable[..., Any]
    draft_forward: Callable[..., Any]
    update_mtp_cache: Callable[..., Any]
    commit_rows: Callable[..., Any]
    prefill_request: Callable[..., Any]
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
        "runtime combine-tail M1-M2 route",
        combine_tail.get("decode_verify")
        if isinstance(combine_tail, Mapping)
        else None,
        [1, 2],
    )
    contract = getattr(runtime, "contract", None)
    if contract is None:
        raise A3BMTPBatchInstallError("Qwen 35B mtp_batch requires MTP contract")
    _require_equal(
        "runtime hidden_variant", getattr(contract, "hidden_variant", None), "post_norm"
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
    trunk_layers, mtp_layers = _model_layers(runtime)
    _require_equal("constructed num_hidden_layers", len(trunk_layers), 40)
    _require_equal("constructed mtp_num_hidden_layers", len(mtp_layers), 1)


def _bind_capture_forward(runtime: Any) -> Callable[..., Any]:
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
    implementations = tuple(getattr(postconv, "m2_implementations", ()) or ())
    if len(implementations) != 30 or not all(callable(item) for item in implementations):
        raise A3BMTPBatchInstallError(
            "Qwen 35B mtp_batch requires 30 M2 post-conv implementations"
        )
    capture = _require_callable(runtime, "_forward_ar_capture_a3b_postconv")
    return partial(
        capture,
        hidden_variant="post_norm",
        postconv_implementations=implementations,
    )


def _commit_qwen35b_b8_t2_rows(
    cache: list[Any],
    captures: dict[int, dict[str, Any]],
    keep_tokens_by_row: list[int],
) -> None:
    """Commit the prevalidated B8/T2 cache layout without hot-path proof work."""
    import mlx.core as mx

    keeps = mx.array(keep_tokens_by_row, dtype=mx.int32)
    positions = [int(value) - 1 for value in keep_tokens_by_row]
    for layer_idx, layer_type in enumerate(_LAYER_TYPES):
        entry = cache[layer_idx]
        if layer_type == "full_attention":
            entry.offsets = (entry.offsets - 2 + keeps).astype(mx.int32)
            continue
        capture = captures[layer_idx]
        conv_states = capture["conv_states"]
        states = capture["states"]
        selector = mx.array(positions, dtype=mx.int32).reshape(
            (8, 1) + (1,) * (int(conv_states.ndim) - 2)
        )
        conv_selector = mx.broadcast_to(
            selector, (8, 1) + tuple(conv_states.shape[2:])
        )
        state_selector = mx.broadcast_to(
            selector, (8, 1) + tuple(states.shape[2:])
        )
        entry[0] = mx.contiguous(
            mx.take_along_axis(conv_states, conv_selector, axis=1)[:, 0]
        )
        entry[1] = mx.contiguous(
            mx.take_along_axis(states, state_selector, axis=1)[:, 0]
        )


def _default_selfcheck(lane: InstalledA3BMTPBatchLane, runtime: Any) -> dict[str, Any]:
    """Run one real B8/T2 route and compare row zero with unchanged B1."""

    import mlx.core as mx
    import numpy as np

    from .attention_context import attention_phase

    token = int(getattr(getattr(runtime, "tokenizer", None), "eos_token_id", 1) or 1)

    def run(batch: int):
        cache = lane.make_cache()
        prompt = mx.full((batch, 1), token, dtype=mx.int32)
        with attention_phase("prefill"):
            logits, hidden = lane.target_forward(
                prompt,
                cache=cache,
                return_hidden=True,
            )
        primary = mx.argmax(logits[:, -1, :], axis=-1)
        with attention_phase("ar_decode"):
            draft_logits = lane.draft_forward(
                hidden[:, -1:, :],
                primary[:, None],
                mtp_cache=lane.make_mtp_cache(),
                mtp_depth=1,
            )
        draft = mx.argmax(draft_logits[:, -1, :], axis=-1)
        verify_input = mx.stack((primary, draft), axis=1)
        with attention_phase("decode_verify"):
            verify_logits, verify_hidden, captures = lane.capture_forward(
                verify_input,
                cache=cache,
            )
        from .gdn_capture import commit_captured_rows

        row_commit = commit_captured_rows(
            cache,
            captures,
            keep_tokens_by_row=[2] * batch,
            verified_tokens=2,
        )
        mx.eval(verify_logits, verify_hidden)
        return verify_input, verify_logits, verify_hidden, captures, row_commit

    batch_input, batch_logits, batch_hidden, batch_captures, batch_commit = run(8)
    solo_input, solo_logits, solo_hidden, solo_captures, solo_commit = run(1)
    target_shape = [int(value) for value in batch_input.shape]
    logits_shape = [int(value) for value in batch_logits.shape]
    hidden_shape = [int(value) for value in batch_hidden.shape]
    batch_logits_row = np.asarray(batch_logits[0], dtype=np.float32)
    solo_logits_row = np.asarray(solo_logits[0], dtype=np.float32)
    batch_hidden_row = np.asarray(batch_hidden[0], dtype=np.float32)
    solo_hidden_row = np.asarray(solo_hidden[0], dtype=np.float32)
    solo_parity = bool(
        np.array_equal(np.asarray(batch_input[0]), np.asarray(solo_input[0]))
        and np.array_equal(batch_logits_row, solo_logits_row)
        and np.array_equal(batch_hidden_row, solo_hidden_row)
    )
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
        ),
        "target_shape": target_shape,
        "logits_shape": logits_shape,
        "hidden_shape": hidden_shape,
        "projection_rows": 16,
        "solo_parity": solo_parity,
        "captured_gdn_layers": len(batch_captures),
        "row_commit": bool(batch_commit and solo_commit),
        "fixed_row_commit": fixed_row_commit,
    }


def install_a3b_mtp_batch_lane(
    runtime: Any,
    *,
    selfcheck: Callable[[InstalledA3BMTPBatchLane], Mapping[str, Any]] | None = None,
) -> InstalledA3BMTPBatchLane:
    """Validate and freeze the exact Qwen 35B B8/T2 route once at startup."""

    _config, fingerprint = _validate_config(runtime)
    _validate_runtime(runtime)
    target_forward = _require_callable(runtime, "forward_ar")
    capture_forward = _bind_capture_forward(runtime)
    model_draft_forward = _require_callable(runtime.model, "mtp_forward")
    model_update_mtp_cache = _require_callable(runtime.model, "mtp_update_cache")
    make_cache = _require_callable(runtime, "make_cache")
    make_mtp_cache = _require_callable(runtime, "make_mtp_cache")
    from .generation import _prefill_committed_mtp_history_streaming

    prefill_request = partial(
        _prefill_committed_mtp_history_streaming,
        runtime,
        base_hidden_variant="post_norm",
        mtp_hidden_variant="post_norm",
        mtp_position_mode="cache",
    )
    draft_forward = partial(
        model_draft_forward,
        concat_order=getattr(runtime.contract, "concat_order", None),
        return_hidden=False,
        mtp_hidden_variant="post_norm",
        mtp_depth=1,
    )
    update_mtp_cache = partial(
        model_update_mtp_cache,
        concat_order=getattr(runtime.contract, "concat_order", None),
    )
    geometry = A3BMTPBatchGeometry()
    lane = InstalledA3BMTPBatchLane(
        geometry=geometry,
        route_id="qwen35b_a3b_mtp_batch_b8_t2_m16",
        config_fingerprint=fingerprint,
        target_forward=target_forward,
        capture_forward=capture_forward,
        draft_forward=draft_forward,
        update_mtp_cache=update_mtp_cache,
        commit_rows=_commit_qwen35b_b8_t2_rows,
        prefill_request=prefill_request,
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
    ):
        raise A3BMTPBatchInstallError(
            "Qwen 35B mtp_batch numerical self-check failed: "
            + json.dumps(report, sort_keys=True, default=str)
        )
    return InstalledA3BMTPBatchLane(
        geometry=geometry,
        route_id=lane.route_id,
        config_fingerprint=fingerprint,
        target_forward=target_forward,
        capture_forward=capture_forward,
        draft_forward=draft_forward,
        update_mtp_cache=update_mtp_cache,
        commit_rows=_commit_qwen35b_b8_t2_rows,
        prefill_request=prefill_request,
        make_cache=make_cache,
        make_mtp_cache=make_mtp_cache,
        selfcheck=MappingProxyType(report),
    )


def _merge_prefilled_caches(caches: list[list[Any]]) -> list[Any]:
    """Merge eight exact solo prefills into the fixed ragged decode cache."""
    if not caches or len({len(cache) for cache in caches}) != 1:
        raise RuntimeError("Qwen 35B mtp_batch prefill caches do not share a layout")

    from .cache_state import OwnedRecurrentStateCache, _is_trimmable
    from .ragged_kv_cache import RaggedBatchKVCache

    merged_cache: list[Any] = []
    for layer_idx in range(len(caches[0])):
        entries = [cache[layer_idx] for cache in caches]
        first = entries[0]
        if _is_trimmable(first):
            offsets = [int(getattr(entry, "offset", 0)) for entry in entries]
            populated = [entry for entry in entries if getattr(entry, "keys", None) is not None]
            if not populated:
                merged = RaggedBatchKVCache(
                    batch_size=len(entries),
                    step=int(getattr(first, "step", 256)),
                )
            else:
                import mlx.core as mx

                template = populated[0]
                capacity = max(int(entry.keys.shape[2]) for entry in populated)
                key_rows = []
                value_rows = []
                for entry in entries:
                    keys = getattr(entry, "keys", None)
                    values = getattr(entry, "values", None)
                    if keys is None:
                        keys = mx.zeros(
                            (
                                1,
                                int(template.keys.shape[1]),
                                capacity,
                                int(template.keys.shape[3]),
                            ),
                            dtype=template.keys.dtype,
                        )
                        values = mx.zeros(
                            (
                                1,
                                int(template.values.shape[1]),
                                capacity,
                                int(template.values.shape[3]),
                            ),
                            dtype=template.values.dtype,
                        )
                    elif int(keys.shape[2]) < capacity:
                        key_pad = mx.zeros(
                            (
                                1,
                                int(keys.shape[1]),
                                capacity - int(keys.shape[2]),
                                int(keys.shape[3]),
                            ),
                            dtype=keys.dtype,
                        )
                        value_pad = mx.zeros(
                            (
                                1,
                                int(values.shape[1]),
                                capacity - int(values.shape[2]),
                                int(values.shape[3]),
                            ),
                            dtype=values.dtype,
                        )
                        keys = mx.concatenate((keys, key_pad), axis=2)
                        values = mx.concatenate((values, value_pad), axis=2)
                    key_rows.append(keys)
                    value_rows.append(values)
                merged = RaggedBatchKVCache(
                    batch_size=len(entries),
                    step=int(getattr(first, "step", 256)),
                    keys=mx.concatenate(key_rows, axis=0),
                    values=mx.concatenate(value_rows, axis=0),
                    offsets=mx.array(offsets, dtype=mx.int32),
                )
            merged._capacity_bound = max(offsets)
            merged_cache.append(merged)
            continue

        merge = getattr(type(first), "merge", None)
        if callable(merge):
            merged = merge(entries)
        else:
            extract = getattr(first, "extract", None)
            extend = getattr(first, "extend", None)
            if not callable(extract) or not callable(extend):
                raise RuntimeError(
                    "Qwen 35B mtp_batch recurrent cache cannot merge layer "
                    f"{layer_idx} ({type(first).__name__})"
                )
            merged = extract(0)
            for entry in entries[1:]:
                merged.extend(entry)
        state = getattr(merged, "state", None)
        if isinstance(state, list) and state:
            merged = OwnedRecurrentStateCache.from_cache(merged)
        merged_cache.append(merged)
    return merged_cache


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

    def notify_terminal(row: int, cycle_count: int) -> None:
        if terminal_notified[row] or finish[row] is None:
            return
        terminal_notified[row] = True
        callback = real[row].on_terminal
        if callback is not None:
            callback(str(finish[row]), int(cycle_count))

    prefills: list[tuple[Any, Any, Any, Any]] = []
    for row, request in enumerate(slots):
        if request is not None and row < len(real) and finish[row] is not None:
            notify_terminal(row, 0)
        prompt = [0] if request is None or request.cancelled() else list(request.prompt_ids)
        try:
            cache, logits, hidden, mtp_cache, *_timing = lane.prefill_request(
                prompt,
                abort_check=request.cancelled if request is not None else None,
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
            cache, logits, hidden, mtp_cache, *_timing = lane.prefill_request(
                [0], abort_check=None
            )
        if (
            int(logits.shape[0]) != 1
            or int(hidden.shape[0]) != 1
            or int(hidden.shape[1]) != 1
        ):
            raise RuntimeError(
                "Qwen 35B mtp_batch solo prefill did not preserve [1,1] ownership"
            )
        prefills.append((cache, logits, hidden, mtp_cache))

    cache = _merge_prefilled_caches([item[0] for item in prefills])
    mtp_cache = _merge_prefilled_caches([item[3] for item in prefills])
    logits_last = mx.concatenate([item[1] for item in prefills], axis=0)
    hidden_last = mx.concatenate([item[2] for item in prefills], axis=0)
    mx.eval(logits_last, hidden_last)
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

    while any(reason is None for reason in finish):
        if cycles >= max_cycles:
            raise RuntimeError("Qwen 35B mtp_batch exceeded its bounded cycle count")
        for row, request in enumerate(real):
            if finish[row] is None and request.cancelled():
                finish[row] = "cancelled"
                notify_terminal(row, cycles)
        if not any(reason is None for reason in finish):
            break
        primary_rows = np.asarray(logits_last, dtype=np.float32)
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
        with attention_phase("ar_decode"):
            draft_logits = lane.draft_forward(
                hidden_last,
                primary_array[:, None],
                mtp_cache=mtp_cache,
                mtp_depth=1,
            )
        mx.eval(draft_logits)
        draft_rows = np.asarray(draft_logits[:, -1, :], dtype=np.float32)
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
        for entry in cache:
            if isinstance(entry, RaggedBatchKVCache):
                entry.reserve(lane.geometry.verify_tokens)
        with attention_phase("decode_verify"):
            verify_logits, verify_hidden, captures = lane.capture_forward(
                verify_input, cache=cache
            )
        mx.eval(verify_logits)
        verify_rows = np.asarray(verify_logits, dtype=np.float32)
        keeps = [2] * width
        accepted_mask = [True] * width
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

        lane.commit_rows(cache, captures, keeps)

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
        mtp_offsets_before_append = [entry.offsets for entry in mtp_cache]
        with attention_phase("ar_decode"):
            lane.update_mtp_cache(
                verify_hidden[:, 0:1, :],
                mx.array(draft_ids, dtype=mx.int32)[:, None],
                mtp_cache=mtp_cache,
            )
        for entry, before_offsets in zip(mtp_cache, mtp_offsets_before_append):
            entry.offsets = mx.where(
                append_mask, entry.offsets, before_offsets
            ).astype(mx.int32)

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
