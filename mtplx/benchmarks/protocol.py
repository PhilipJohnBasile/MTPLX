"""Canonical effective-run records for comparable benchmark arms.

Benchmark labels describe intent.  This module records what the runner
actually executed and provides a strict comparison gate so a result cannot be
silently compared across thinking modes, prompt suites, sampler settings, or
aggregation methods.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_SCHEMA = 1
WEIGHTED_TOK_S_AGGREGATION = "sum_generated_tokens/sum_decode_seconds"

_COMPARISON_FIELDS = (
    "prompt_suite_sha256",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "seed",
    "seed_policy",
    "enable_thinking",
    "runtime_contract",
    "aggregation",
)


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_effective_run_record(
    *,
    backend: str,
    model_ref: Path | str,
    model_revision: str | None,
    draft_ref: Path | str | None,
    draft_revision: str | None,
    prompt_suite: Path | str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    seed: int,
    seed_policy: str = "fixed",
    enable_thinking: bool,
    block_size: int | None,
    generation_mode: str,
    nax_enabled: bool,
    verify_strategy: str,
    compiled_verify: str,
    runtime_switches: Mapping[str, str],
    runtime_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    suite = Path(prompt_suite).expanduser().resolve()
    return {
        "schema": PROTOCOL_SCHEMA,
        "backend": str(backend),
        "model": {
            "ref": str(model_ref),
            "revision": _required_revision(model_revision, "model_revision"),
        },
        "draft": (
            None
            if draft_ref is None
            else {
                "ref": str(draft_ref),
                "revision": _required_revision(draft_revision, "draft_revision"),
            }
        ),
        "prompt_suite": str(suite),
        "prompt_suite_sha256": file_sha256(suite),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "max_tokens": int(max_tokens),
        "seed": int(seed),
        "seed_policy": str(seed_policy),
        "enable_thinking": bool(enable_thinking),
        "block_size": None if block_size is None else int(block_size),
        "generation_mode": str(generation_mode),
        "nax_enabled": bool(nax_enabled),
        "verify_strategy": str(verify_strategy),
        "compiled_verify": str(compiled_verify),
        "runtime_contract": {
            str(key): value
            for key, value in sorted((runtime_contract or {}).items())
        },
        "runtime_switches": {
            str(key): str(value)
            for key, value in sorted(runtime_switches.items())
        },
        "aggregation": WEIGHTED_TOK_S_AGGREGATION,
    }


def weighted_tok_s(rows: list[Mapping[str, Any]]) -> float:
    generated_tokens = 0
    decode_seconds = 0.0
    for row in rows:
        tokens = int(row.get("generated_tokens") or 0)
        tok_s = float(row.get("tok_s") or 0.0)
        if tokens <= 0 or tok_s <= 0:
            continue
        generated_tokens += tokens
        decode_seconds += tokens / tok_s
    return generated_tokens / decode_seconds if decode_seconds > 0 else 0.0


def protocol_mismatches(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Return fields that make two arms protocol-incomparable.

    Backend, model/draft identity, block size, generation mode, and NAX are arm
    dimensions.  The fields below are the shared protocol and must match.
    """

    mismatches: dict[str, tuple[Any, Any]] = {}
    for field in _COMPARISON_FIELDS:
        if left.get(field) != right.get(field):
            mismatches[field] = (left.get(field), right.get(field))
    return mismatches


def assert_protocol_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> None:
    mismatches = protocol_mismatches(left, right)
    if not mismatches:
        return
    details = ", ".join(
        f"{field}={before!r} != {after!r}"
        for field, (before, after) in mismatches.items()
    )
    raise ValueError(f"benchmark protocol mismatch: {details}")


def _required_revision(value: str | None, field: str) -> str:
    revision = str(value or "").strip()
    if not revision:
        raise ValueError(f"{field} must be pinned for an evidence-grade run")
    return revision
