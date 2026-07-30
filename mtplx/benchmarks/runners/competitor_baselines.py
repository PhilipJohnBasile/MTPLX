"""Same-harness runners for external speculative baselines."""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mtplx.benchmarks.schema import load_prompt_suite
from mtplx.benchmarks.validators.basic import (
    validate_json_text,
    validate_no_degenerate_loop,
)


def run_dflash_mlx_baseline(
    model_path: Path | str,
    draft_model: str,
    prompt_suite: Path | str,
    *,
    dflash_source: Path | str = "REFERENCES:TOOLS/dflash",
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 96,
    block_size: int | None = None,
    seed: int = 0,
    limit: int | None = None,
    enable_thinking: bool | None = None,
    draft_sliding_window_size: int | None = None,
) -> dict[str, Any]:
    _add_source_path(dflash_source)

    try:
        from dflash.model_mlx import load, load_draft, stream_generate
        from mlx_lm.sample_utils import make_sampler
    except Exception as exc:  # pragma: no cover - environment/reporting path
        return _error_result(
            "dflash_mlx_official",
            model_path,
            draft_model,
            prompt_suite,
            "import_failed",
            exc,
        )

    try:
        import inspect

        # Upstream dflash removed the sliding_window_size kwarg (window config now
        # comes from the draft checkpoint's layer_types/sliding_window fields).
        if "sliding_window_size" in inspect.signature(load_draft).parameters:
            draft = load_draft(str(draft_model), sliding_window_size=draft_sliding_window_size)
        else:
            draft = load_draft(str(draft_model))
        target_model, tokenizer = load(str(model_path))
    except Exception as exc:
        return _error_result(
            "dflash_mlx_official",
            model_path,
            draft_model,
            prompt_suite,
            "load_failed",
            exc,
        )

    sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
    rows = []
    prompts = load_prompt_suite(prompt_suite)
    if limit is not None:
        prompts = prompts[:limit]

    for index, case in enumerate(prompts):
        # Seed MLX RNG per prompt so runs are reproducible; the seed was
        # previously recorded in the result but never applied.
        import mlx.core as _mx

        _mx.random.seed(seed + index)
        messages = case.messages or [{"role": "user", "content": case.prompt}]
        kwargs: dict[str, Any] = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )

        chunks = []
        acceptance_lengths = []
        final = None
        try:
            for response in stream_generate(
                target_model,
                draft,
                tokenizer,
                prompt,
                block_size=block_size,
                max_tokens=min(max_tokens, case.max_tokens),
                temperature=temperature,
                sampler=sampler,
            ):
                final = response
                if response.text:
                    chunks.append(response.text)
                if response.accepted:
                    acceptance_lengths.append(int(response.accepted))
        except Exception as exc:
            rows.append(
                {
                    "prompt_id": case.id,
                    "category": case.category,
                    "prompt_sha256": case.prompt_sha256,
                    "error": repr(exc),
                    "validations": [],
                }
            )
            continue

        text = "".join(chunks)
        validations = [asdict(validate_no_degenerate_loop(text))]
        if case.category == "json_tool":
            validations.append(asdict(validate_json_text(text.strip())))

        generated_tokens = int(getattr(final, "generation_tokens", 0) or 0)
        tok_s = float(getattr(final, "generation_tps", 0.0) or 0.0)
        rows.append(
            {
                "prompt_id": case.id,
                "category": case.category,
                "prompt_sha256": case.prompt_sha256,
                "generated_tokens": generated_tokens,
                "tok_s": tok_s,
                "acceptance_lengths": acceptance_lengths,
                "mean_acceptance_length": (
                    statistics.mean(acceptance_lengths) if acceptance_lengths else None
                ),
                "prompt_tps": float(getattr(final, "prompt_tps", 0.0) or 0.0),
                "peak_memory_gb": float(getattr(final, "peak_memory", 0.0) or 0.0),
                "finish_reason": getattr(final, "finish_reason", None),
                "validations": validations,
                "text": text,
            }
        )

    validations = [v for row in rows for v in row.get("validations", [])]
    successful = [row for row in rows if row.get("tok_s") is not None]
    return {
        "backend": "dflash_mlx_official",
        "model_path": str(model_path),
        "draft_model": str(draft_model),
        "prompt_suite": str(prompt_suite),
        "sampler": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "note": "DFlash MLX uses sampled prefix matching at temperature, not residual-corrected speculative sampling.",
        },
        "max_tokens": max_tokens,
        "block_size": block_size,
        "seed": seed,
        "enable_thinking": enable_thinking,
        "rows": rows,
        "summary": {
            "prompts": len(rows),
            "successful_prompts": len(successful),
            "generated_tokens": sum(int(row.get("generated_tokens") or 0) for row in rows),
            "mean_tok_s": (
                statistics.mean([float(row["tok_s"]) for row in successful])
                if successful
                else 0.0
            ),
            "mean_acceptance_length": _mean_present(
                row.get("mean_acceptance_length") for row in successful
            ),
            "validations_passed": sum(1 for v in validations if v["passed"]),
            "validations_total": len(validations),
            "peak_memory_gb": max([float(row.get("peak_memory_gb") or 0.0) for row in rows] or [0.0]),
        },
    }


def run_ddtree_mlx_baseline(
    model_path: Path | str,
    draft_model: str,
    prompt_suite: Path | str,
    *,
    ddtree_source: Path | str = "REFERENCES:TOOLS/ddtree-mlx",
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 96,
    tree_budget: int = 4,
    limit: int | None = None,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    _add_source_path(ddtree_source)

    try:
        from dflash_mlx.generate import get_stop_token_ids, load_runtime_components
        from ddtree_mlx.runtime import generate_ddtree_once
    except Exception as exc:  # pragma: no cover - environment/reporting path
        return _error_result(
            "ddtree_mlx",
            model_path,
            draft_model,
            prompt_suite,
            "import_failed",
            exc,
        )

    try:
        target_model, tokenizer, draft, draft_ref = load_runtime_components(
            model_ref=str(model_path),
            draft_ref=str(draft_model),
        )
        if draft is None:
            raise RuntimeError(
                "DDTree load_runtime_components returned no draft model; "
                "the DFlash drafter is likely unavailable or gated."
            )
        stop_ids = get_stop_token_ids(tokenizer)
    except Exception as exc:
        return _error_result(
            "ddtree_mlx",
            model_path,
            draft_model,
            prompt_suite,
            "load_failed",
            exc,
        )

    rows = []
    prompts = load_prompt_suite(prompt_suite)
    if limit is not None:
        prompts = prompts[:limit]

    for case in prompts:
        messages = case.messages or [{"role": "user", "content": case.prompt}]
        kwargs: dict[str, Any] = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        prompt_tokens = list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **kwargs,
            )
        )
        try:
            result = generate_ddtree_once(
                target_model=target_model,
                draft_model=draft,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_new_tokens=min(max_tokens, case.max_tokens),
                tree_budget=tree_budget,
                stop_token_ids=stop_ids,
            )
        except Exception as exc:
            rows.append(
                {
                    "prompt_id": case.id,
                    "category": case.category,
                    "prompt_sha256": case.prompt_sha256,
                    "error": repr(exc),
                    "validations": [],
                }
            )
            continue

        text = tokenizer.decode(result.get("generated_token_ids", []))
        validations = [asdict(validate_no_degenerate_loop(text))]
        if case.category == "json_tool":
            validations.append(asdict(validate_json_text(text.strip())))
        rows.append(
            {
                "prompt_id": case.id,
                "category": case.category,
                "prompt_sha256": case.prompt_sha256,
                "generated_tokens": result.get("generation_tokens", 0),
                "tok_s": result.get("tokens_per_second", 0.0),
                "avg_acceptance": result.get("avg_acceptance"),
                "fast_path_ratio": result.get("fast_path_ratio"),
                "tree_aware_linear": result.get("tree_aware_linear", False),
                "validations": validations,
                "text": text,
                "raw": result,
            }
        )

    validations = [v for row in rows for v in row.get("validations", [])]
    successful = [row for row in rows if row.get("tok_s") is not None]
    return {
        "backend": "ddtree_mlx",
        "model_path": str(model_path),
        "draft_model": str(draft_ref),
        "prompt_suite": str(prompt_suite),
        "sampler": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "note": "DDTree baseline depends on dflash_mlx runtime semantics.",
        },
        "max_tokens": max_tokens,
        "tree_budget": tree_budget,
        "enable_thinking": enable_thinking,
        "rows": rows,
        "summary": {
            "prompts": len(rows),
            "successful_prompts": len(successful),
            "generated_tokens": sum(int(row.get("generated_tokens") or 0) for row in rows),
            "mean_tok_s": (
                statistics.mean([float(row["tok_s"]) for row in successful])
                if successful
                else 0.0
            ),
            "mean_acceptance": _mean_present(row.get("avg_acceptance") for row in successful),
            "validations_passed": sum(1 for v in validations if v["passed"]),
            "validations_total": len(validations),
        },
    }


def write_competitor_result(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))


def _add_source_path(source: Path | str) -> None:
    path = str(Path(source).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def _mean_present(values) -> float | None:
    present = [float(v) for v in values if v is not None]
    return statistics.mean(present) if present else None


def _error_result(
    backend: str,
    model_path: Path | str,
    draft_model: str,
    prompt_suite: Path | str,
    stage: str,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "model_path": str(model_path),
        "draft_model": str(draft_model),
        "prompt_suite": str(prompt_suite),
        "error": {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "repr": repr(exc),
        },
        "rows": [],
        "summary": {
            "prompts": 0,
            "successful_prompts": 0,
            "generated_tokens": 0,
            "mean_tok_s": 0.0,
            "validations_passed": 0,
            "validations_total": 0,
        },
    }
