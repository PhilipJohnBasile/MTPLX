"""Canonical generation-result envelope for the serving endpoints.

The stop-path terminations of /v1/chat/completions and /v1/completions
(stream and non-stream) build their ``generated`` result dict through
:func:`build_generation_result` instead of per-site literals.

Contract notes:

- The ``stats`` KEY SET of a stop-path result must not depend on whether
  the client streamed. tests/test_response_envelope_parity.py pins this:
  the same fixture request run with stream=true and stream=false must
  yield identical ``mtplx_stats`` key sets.
- ``finish_reason`` is part of the canonical ``stats`` shape at
  construction time. This is the one deliberate behavior change of the
  Phase 6.2 extraction (2026-08-16): previously only the non-stream chat
  stop literal nested ``stats.finish_reason``; the stream literals relied
  on the endpoint's post-final patch (``stats["finish_reason"] = ...``
  just before the final chunk), so anything reading ``stats`` between
  construction and that patch — the first last-metrics merge — saw a
  different key set depending on stream mode. Direction of the fix: the
  richer non-stream shape wins, so no consumer loses a field.
- ``stats_extra`` keys are site-owned (stop-sequence markers, chat bridge
  markers). They may override base keys; call sites must not rely on that.
"""

from __future__ import annotations

from typing import Any


def build_generation_result(
    *,
    text: str,
    tokens: list[int],
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str,
    generation_mode: str,
    mtp_depth: int,
    stats_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical generated/stats result dict for a finished request."""
    prompt_tokens = int(prompt_tokens)
    completion_tokens = int(completion_tokens)
    stats: dict[str, Any] = {
        "generation_mode": generation_mode,
        "mtp_depth": mtp_depth,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
    }
    if stats_extra:
        stats.update(stats_extra)
    return {
        "text": text,
        "tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "stats": stats,
    }
