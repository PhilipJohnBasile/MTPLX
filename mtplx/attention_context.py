"""Lightweight attention-phase telemetry context."""

from __future__ import annotations

import os
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

# The t<=0 stock-matmul guard is OPT-IN as of 2026-08-31 (founder order:
# restore turbo at greedy). Receipts (SPEEDWAR-20260831/turbo-guard-27b):
# on the 27B turbo serve path the guard cost 5-21% greedy decode across a
# position-matched ABBA quad (guard-off 54.3/52.4 vs guard-on 51.0/49.7
# tok/s in the clean pair), while the identity it promised held on NEITHER
# route — 5/6 default-suite prompts diverge from greedy AR at identical
# token indexes with the guard on or off, because the divergence lives in
# the cross-M numeric frame (M=1 AR vs M=4..16 verify shapes), which stock
# kernels share. MTPLX_EXACT_T0_GUARD=1 re-arms the stock frame for
# operators who want it. Read once at import (hot-path flag pattern).
_EXACT_T0_GUARD_ARMED = (
    (os.environ.get("MTPLX_EXACT_T0_GUARD") or "").strip().lower()
    in {"1", "true", "yes", "on"}
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

    Only meaningful when the operator arms MTPLX_EXACT_T0_GUARD=1: the
    vk/nax verify kernels are argmax- and distribution-validated but not
    bit-exact vs stock (~6e-3 dmax, lane-strided fp32 accumulation), and an
    armed guard makes t<=0 verify forwards fall through to stock so both
    paths share one numeric frame. The shipping default leaves the guard
    dark — turbo kernels run at every temperature — because the guard never
    delivered MTP==AR greedy identity (cross-M frame flips survive stock
    kernels) and cost 5-21% greedy decode on the 27B turbo profile.
    """
    if not _EXACT_T0_GUARD_ARMED:
        return False
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
