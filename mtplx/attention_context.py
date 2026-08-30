"""Lightweight attention-phase telemetry context."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

VALID_ATTENTION_PHASES = {
    "prefill",
    "decode_verify",
    "ar_decode",
    "postcommit",
    "unknown",
}
VALID_MODEL_FORWARD_KINDS = {
    "target_verify",
    "repair",
    "other",
}

_ATTENTION_PHASE: ContextVar[str] = ContextVar(
    "mtplx_attention_phase",
    default="unknown",
)
_MODEL_FORWARD_KIND: ContextVar[str] = ContextVar(
    "mtplx_model_forward_kind",
    default="other",
)


def normalize_attention_phase(phase: str | None) -> str:
    value = (phase or "unknown").strip().lower()
    return value if value in VALID_ATTENTION_PHASES else "unknown"


def current_attention_phase() -> str:
    return normalize_attention_phase(_ATTENTION_PHASE.get())


def normalize_model_forward_kind(kind: str | None) -> str:
    value = (kind or "other").strip().lower()
    return value if value in VALID_MODEL_FORWARD_KINDS else "other"


def current_model_forward_kind() -> str:
    return normalize_model_forward_kind(_MODEL_FORWARD_KIND.get())


_EXACT_VERIFY_REQUIRED: ContextVar[bool] = ContextVar(
    "mtplx_exact_verify_required",
    default=False,
)

# Multi-axis rope state for vision requests: (positions [3, prompt_len] mx
# array or None, rope_delta int). Families that implement M-RoPE (qwen4_exp)
# read it inside their attention layers and self-slice by cache offset; every
# other family ignores it. Set per request around generation entry points —
# never stored in cache state, so bank restores stay format-stable (the
# request re-derives it from its own content).
_VISION_ROPE: ContextVar["tuple[object, int] | None"] = ContextVar(
    "mtplx_vision_rope",
    default=None,
)


def vision_rope_state() -> "tuple[object, int] | None":
    return _VISION_ROPE.get()


@contextmanager
def vision_rope(positions: object, delta: int) -> Iterator[None]:
    token = _VISION_ROPE.set((positions, int(delta)))
    try:
        yield
    finally:
        _VISION_ROPE.reset(token)


def exact_verify_required() -> bool:
    """True while the current forward must use stock (bit-exact) matmuls.

    The greedy exactness contract: at temperature <= 0 the product promise is
    MTP output == AR output token-for-token. The vk/nax verify kernels are
    argmax- and distribution-validated but NOT bit-exact vs stock (~6e-3
    dmax, lane-strided fp32 accumulation), and AR decode runs M=1 stock — so
    a near-tie logit row can flip argmax between the two paths. While this
    flag is set, the QuantizedLinear verify patch falls through to stock so
    both paths share one numeric frame.
    """
    return bool(_EXACT_VERIFY_REQUIRED.get())


@contextmanager
def exact_verify(required: bool) -> Iterator[None]:
    token = _EXACT_VERIFY_REQUIRED.set(bool(required))
    try:
        yield
    finally:
        _EXACT_VERIFY_REQUIRED.reset(token)


@contextmanager
def attention_phase(phase: str | None) -> Iterator[None]:
    token = _ATTENTION_PHASE.set(normalize_attention_phase(phase))
    try:
        yield
    finally:
        _ATTENTION_PHASE.reset(token)


@contextmanager
def model_forward_kind(kind: str | None) -> Iterator[None]:
    """Identify whether one decode-verify-phase target call verifies or repairs."""

    token = _MODEL_FORWARD_KIND.set(normalize_model_forward_kind(kind))
    try:
        yield
    finally:
        _MODEL_FORWARD_KIND.reset(token)
