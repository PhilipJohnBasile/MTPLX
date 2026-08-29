"""Construction-bound fixed-M4 capture surface for Qwen4 Flash-Next."""

from __future__ import annotations

import os
from types import MethodType
from typing import Any

import mlx.core as mx

from .models.qwen4_exp import (
    compiled_verify_ple_scope,
    verify_capture_scope,
)

_GDN_ROW_NAMES = ("qkv", "q", "k", "v", "a", "b")
_PLE_ROW_NAMES = ("ple_hidden", "ple_ids")


def is_qwen4_fixed_verify_config(config: dict[str, Any]) -> bool:
    """Identify the one measured Flash-Next fixed-M4 model geometry."""

    text = config.get("text_config")
    if not isinstance(text, dict):
        return False
    observed = (
        config.get("model_type"),
        text.get("model_type"),
        text.get("hidden_size"),
        text.get("num_hidden_layers"),
        text.get("hc_count"),
        text.get("hc_lowrank"),
        text.get("indexer_compress_ratio"),
        text.get("linear_num_key_heads"),
        text.get("linear_num_value_heads"),
        text.get("linear_key_head_dim"),
        text.get("linear_value_head_dim"),
        text.get("ple_layer_ids"),
        text.get("ngram_size"),
        text.get("ngram_vocab_size_base"),
        text.get("heads_per_ngram"),
        text.get("ple_embed_dim"),
        text.get("ngram_sidecar"),
        text.get("num_experts"),
        text.get("num_experts_per_tok"),
        text.get("moe_intermediate_size"),
        text.get("vocab_size"),
    )
    return observed == (
        "qwen4_exp",
        "qwen4_exp_text",
        2560,
        48,
        4,
        320,
        4,
        16,
        48,
        128,
        128,
        [2],
        3,
        20_000_000,
        8,
        2560,
        True,
        512,
        10,
        640,
        248_320,
    )


def qwen4_fixed_verify_enabled() -> bool:
    raw = os.environ.get("MTPLX_QWEN4_FIXED_M4_VERIFY", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _text_model(runtime: Any):
    return getattr(runtime.model, "language_model", runtime.model)


def _inner(runtime: Any):
    return _text_model(runtime).model


def _prepare_compiled_verify_aux(self: Any, input_ids, cache) -> mx.array:
    """Materialize the one host-backed PLE embedding before graph dispatch."""

    inner = _inner(self)
    layer_index = int(inner._ple_stage_idx)
    ple = inner.layers[layer_index].ple
    entry = cache[layer_index]
    previous = entry[ple.NGRAM_IDX]
    embedding = ple.ple_embedding._graph_path(
        input_ids,
        None,
        ple.NGRAM_IDX,
        prev=previous,
    )
    # The SSD sidecar returns a lazy dequantization graph whose inputs are
    # host-acquired row payloads.  It must cross the materialization boundary
    # before becoming an explicit input to mx.compile; otherwise MLX sees the
    # row graph's cache leaves as uncaptured inputs to the verifier graph.
    mx.eval(embedding)
    return embedding


def _forward_ar_capture(
    self: Any,
    input_ids,
    *,
    cache=None,
    return_hidden: bool = True,
    hidden_variant: str | None = None,
    capture_backend: str | None = None,
    compiled_aux: mx.array | None = None,
):
    del hidden_variant, capture_backend
    text = _text_model(self)
    if cache is None:
        cache = text.make_cache()
    with verify_capture_scope(), compiled_verify_ple_scope(compiled_aux):
        logits, hidden = text(
            input_ids,
            cache=cache,
            return_hidden=True,
        )

    captures: dict[int, dict[str, mx.array]] = {}
    for index, layer in enumerate(text.model.layers):
        if not layer.is_linear:
            continue
        entry = cache[index]
        rows = getattr(entry, "_mtplx_verify_rows", None)
        if rows is None or len(rows) != len(_GDN_ROW_NAMES):
            raise RuntimeError(f"qwen4 fixed-M4 capture missing GDN rows at layer {index}")
        layer_capture = dict(zip(_GDN_ROW_NAMES, rows))
        if getattr(layer, "ple", None) is not None:
            ple_rows = getattr(entry, "_mtplx_verify_ple", None)
            if ple_rows is None or len(ple_rows) != len(_PLE_ROW_NAMES):
                raise RuntimeError(
                    f"qwen4 fixed-M4 capture missing PLE rows at layer {index}"
                )
            layer_capture.update(zip(_PLE_ROW_NAMES, ple_rows))
        captures[index] = layer_capture
    if return_hidden:
        return logits, hidden, captures
    return logits, captures


def _commit_compiled_verify_captures(
    self: Any, cache, captures: dict[int, dict[str, mx.array]]
) -> None:
    """Install materialized capture leaves on the live family cache."""

    for index, layer in enumerate(_inner(self).layers):
        if not layer.is_linear:
            continue
        layer_capture = captures[index]
        cache[index]._mtplx_verify_rows = tuple(
            layer_capture[name] for name in _GDN_ROW_NAMES
        )
        if getattr(layer, "ple", None) is not None:
            cache[index]._mtplx_verify_ple = tuple(
                layer_capture[name] for name in _PLE_ROW_NAMES
            )


def install_qwen4_fixed_verify_route(runtime: Any) -> dict[str, Any]:
    """Validate the Qwen4 state topology once and bind direct fixed-M4 hooks."""

    inner = _inner(runtime)
    layers = tuple(inner.layers)
    linear = tuple(index for index, layer in enumerate(layers) if layer.is_linear)
    qsa = tuple(index for index, layer in enumerate(layers) if not layer.is_linear)
    ple = tuple(
        index for index, layer in enumerate(layers) if getattr(layer, "ple", None)
    )
    if not linear or not qsa or len(ple) != 1 or not layers[ple[0]].is_linear:
        raise ValueError("qwen4 fixed-M4 verifier requires mixed QSA/GDN and one PLE")

    args = inner.args
    if int(args.hidden_size) == 2560:
        observed = (
            len(layers),
            len(linear),
            len(qsa),
            int(args.hc_count),
            int(args.indexer_compress_ratio),
        )
        if observed != (48, 36, 12, 4, 4):
            raise ValueError(
                f"qwen4 fixed-M4 production geometry mismatch: {observed}"
            )

    extra = []
    for index in linear:
        names = _GDN_ROW_NAMES + (_PLE_ROW_NAMES if index == ple[0] else ())
        extra.append((index, names))

    runtime.forward_ar_capture = MethodType(_forward_ar_capture, runtime)
    runtime.prepare_compiled_verify_aux = MethodType(
        _prepare_compiled_verify_aux, runtime
    )
    runtime.commit_compiled_verify_captures = MethodType(
        _commit_compiled_verify_captures, runtime
    )
    runtime._mtplx_capture_layout = ()
    runtime._mtplx_capture_extra_layout = tuple(extra)
    runtime.qwen4_fixed_m4_compiled_verify = True
    return {"installed": True, "linear_layers": len(linear), "rows": 4}


__all__ = [
    "is_qwen4_fixed_verify_config",
    "install_qwen4_fixed_verify_route",
    "qwen4_fixed_verify_enabled",
]
