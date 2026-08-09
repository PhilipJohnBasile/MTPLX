"""Construction-time contract for Qwen3.6-35B-A3B eight-row MTP decode."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from mtplx.artifacts import load_config


_LAYER_TYPES = tuple(
    "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
    for index in range(40)
)


class A3BMTPBatchInstallError(RuntimeError):
    """The fixed Qwen 35B MTP batch lane cannot be installed safely."""


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


@dataclass(frozen=True)
class InstalledA3BMTPBatchLane:
    """Prevalidated, prebound fixed-shape lane used directly by serving."""

    geometry: A3BMTPBatchGeometry
    route_id: str
    config_fingerprint: str
    target_forward: Callable[..., Any]
    draft_forward: Callable[..., Any]
    make_cache: Callable[..., Any]
    make_mtp_cache: Callable[..., Any]
    selfcheck: Mapping[str, Any]


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


def _default_selfcheck(lane: InstalledA3BMTPBatchLane, runtime: Any) -> dict[str, Any]:
    """Run one real B8/T2 route and compare row zero with unchanged B1."""

    import mlx.core as mx
    import numpy as np

    token = int(getattr(getattr(runtime, "tokenizer", None), "eos_token_id", 1) or 1)

    def run(batch: int):
        cache = lane.make_cache()
        prompt = mx.full((batch, 1), token, dtype=mx.int32)
        logits, hidden = lane.target_forward(
            prompt,
            cache=cache,
            return_hidden=True,
        )
        primary = mx.argmax(logits[:, -1, :], axis=-1)
        draft_logits = lane.draft_forward(
            hidden[:, -1:, :],
            primary[:, None],
            mtp_cache=lane.make_mtp_cache(),
            mtp_depth=1,
        )
        draft = mx.argmax(draft_logits[:, -1, :], axis=-1)
        verify_input = mx.stack((primary, draft), axis=1)
        verify_logits, verify_hidden = lane.target_forward(
            verify_input,
            cache=cache,
            return_hidden=True,
        )
        mx.eval(verify_logits, verify_hidden)
        return verify_input, verify_logits, verify_hidden

    batch_input, batch_logits, batch_hidden = run(8)
    solo_input, solo_logits, solo_hidden = run(1)
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
    return {
        "ok": bool(
            target_shape == [8, 2]
            and logits_shape[:2] == [8, 2]
            and hidden_shape[:2] == [8, 2]
            and solo_parity
        ),
        "target_shape": target_shape,
        "logits_shape": logits_shape,
        "hidden_shape": hidden_shape,
        "projection_rows": 16,
        "solo_parity": solo_parity,
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
    draft_forward = _require_callable(runtime, "draft_mtp")
    make_cache = _require_callable(runtime, "make_cache")
    make_mtp_cache = _require_callable(runtime, "make_mtp_cache")
    geometry = A3BMTPBatchGeometry()
    lane = InstalledA3BMTPBatchLane(
        geometry=geometry,
        route_id="qwen35b_a3b_mtp_batch_b8_t2_m16",
        config_fingerprint=fingerprint,
        target_forward=target_forward,
        draft_forward=draft_forward,
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
        draft_forward=draft_forward,
        make_cache=make_cache,
        make_mtp_cache=make_mtp_cache,
        selfcheck=MappingProxyType(report),
    )
