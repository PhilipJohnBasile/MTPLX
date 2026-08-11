#!/usr/bin/env python3
"""Teacher-forced HumanEval loss for compressed Qwen3.6-35B artifacts.

Each process records one artifact-local likelihood measurement. Cross-artifact
comparisons are valid only when tokenizer assets and every prompt/solution
token-ID sequence are hashed and identical. This script now records that
contract for future runs; historical v1/v2 receipts lack it and are retained
as explicitly non-comparable observations, not quality evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_ids_sha256(token_ids: list[int]) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(token_ids, dtype=np.int32).tobytes()).hexdigest()


def _tokenizer_identity(model_path: Path) -> dict[str, Any]:
    """Hash the tokenizer assets needed to make a cross-artifact claim."""

    names = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    )
    assets = []
    for name in names:
        path = model_path / name
        if path.is_file():
            assets.append({"filename": name, "sha256": _sha256(path)})
    if not assets:
        raise ValueError("model directory has no recognized tokenizer assets")
    encoded = json.dumps(assets, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "asset_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "assets": assets,
    }


def _load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("task_id", "prompt", "canonical_solution"):
                if field not in row:
                    raise ValueError(f"{path} row lacks {field!r}")
            rows.append(row)
            if len(rows) == limit:
                break
    if not rows:
        raise ValueError(f"no HumanEval rows parsed from {path}")
    return rows


def _summary(losses: list[float], token_counts: list[int]) -> dict[str, Any]:
    if not losses or len(losses) != len(token_counts):
        raise ValueError("loss and token-count vectors must be non-empty and aligned")
    total_tokens = sum(token_counts)
    weighted_nll = sum(loss * count for loss, count in zip(losses, token_counts))
    mean_nll = weighted_nll / total_tokens
    return {
        "task_count": len(losses),
        "scored_token_count": total_tokens,
        "mean_nll_nats_per_token": mean_nll,
        "perplexity": math.exp(mean_nll),
        "median_task_nll_nats_per_token": statistics.median(losses),
        "min_task_nll_nats_per_token": min(losses),
        "max_task_nll_nats_per_token": max(losses),
    }


def _read_receipt(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    supported = {
        "mtplx-paroquant-q35-humaneval-nll-v1",
        "mtplx-q35-compressed-artifact-humaneval-nll-v2",
        "mtplx-q35-compressed-artifact-humaneval-nll-v3",
    }
    if report.get("schema") not in supported:
        raise ValueError(f"{path} is not a supported HumanEval NLL receipt")
    return report


def _arm_from_receipts(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("each comparison arm requires at least two receipts")
    reports = [_read_receipt(path) for path in paths]
    first = reports[0]
    dataset = first["dataset"]
    manifest = first["model_identity"]["weight_manifest_after_sha256"]
    task_ids = [row["task_id"] for row in first["tasks"]]
    token_counts = [row["canonical_solution_tokens"] for row in first["tasks"]]
    tokenizer_identity = first.get("tokenizer_identity")
    prompt_hashes = [row.get("prompt_token_ids_sha256") for row in first["tasks"]]
    solution_hashes = [
        row.get("canonical_solution_token_ids_sha256") for row in first["tasks"]
    ]
    if not isinstance(tokenizer_identity, dict):
        raise ValueError(f"{paths[0]} lacks tokenizer identity required for comparison")
    if not all(isinstance(value, str) for value in prompt_hashes + solution_hashes):
        raise ValueError(f"{paths[0]} lacks token-id hashes required for comparison")
    nll_vectors: list[list[float]] = []
    for path, report in zip(paths, reports):
        if report["dataset"] != dataset:
            raise ValueError(f"dataset mismatch in {path}")
        if report["model_identity"]["weight_manifest_after_sha256"] != manifest:
            raise ValueError(f"weight-manifest mismatch in {path}")
        rows = report["tasks"]
        if [row["task_id"] for row in rows] != task_ids:
            raise ValueError(f"task-order mismatch in {path}")
        if [row["canonical_solution_tokens"] for row in rows] != token_counts:
            raise ValueError(f"token-count mismatch in {path}")
        if report.get("tokenizer_identity") != tokenizer_identity:
            raise ValueError(f"tokenizer identity mismatch in {path}")
        if [row.get("prompt_token_ids_sha256") for row in rows] != prompt_hashes:
            raise ValueError(f"prompt token-id mismatch in {path}")
        if [
            row.get("canonical_solution_token_ids_sha256") for row in rows
        ] != solution_hashes:
            raise ValueError(f"solution token-id mismatch in {path}")
        nll_vectors.append([float(row["nll_nats_per_token"]) for row in rows])

    reference = nll_vectors[0]
    exact_repeat = all(vector == reference for vector in nll_vectors[1:])
    mean_task_nlls = [
        statistics.mean(values) for values in zip(*nll_vectors, strict=True)
    ]
    aggregate_nlls = [
        float(report["aggregate"]["mean_nll_nats_per_token"]) for report in reports
    ]
    return {
        "labels": [str(report["label"]) for report in reports],
        "receipt_sha256": [_sha256(path) for path in paths],
        "weight_manifest_sha256": manifest,
        "dataset": dataset,
        "tokenizer_identity": tokenizer_identity,
        "task_ids": task_ids,
        "token_counts": token_counts,
        "prompt_token_ids_sha256": prompt_hashes,
        "canonical_solution_token_ids_sha256": solution_hashes,
        "nll_vectors": nll_vectors,
        "mean_task_nlls": mean_task_nlls,
        "aggregate_nlls": aggregate_nlls,
        "exact_task_nll_repeat": exact_repeat,
        "mean_nll_nats_per_token": statistics.mean(aggregate_nlls),
        "perplexity": math.exp(statistics.mean(aggregate_nlls)),
    }


def _directional_status(
    baseline_nll: float, candidate_nll: float, repeats_match: bool
) -> str:
    if candidate_nll > baseline_nll:
        status = "directional_quality_regression"
    elif candidate_nll < baseline_nll:
        status = "directional_quality_improvement"
    else:
        status = "no_directional_quality_change"
    if not repeats_match:
        status += "_with_repeat_mismatch"
    return status


def compare_receipts(
    stock_paths: list[Path], paro_paths: list[Path], json_out: Path
) -> dict[str, Any]:
    stock = _arm_from_receipts(stock_paths)
    paro = _arm_from_receipts(paro_paths)
    if stock["dataset"] != paro["dataset"]:
        raise ValueError("stock and PARO datasets differ")
    if stock["task_ids"] != paro["task_ids"]:
        raise ValueError("stock and PARO task orders differ")
    if stock["token_counts"] != paro["token_counts"]:
        raise ValueError("stock and PARO token counts differ")
    if stock["tokenizer_identity"] != paro["tokenizer_identity"]:
        raise ValueError("stock and PARO tokenizer identities differ")
    if stock["prompt_token_ids_sha256"] != paro["prompt_token_ids_sha256"]:
        raise ValueError("stock and PARO prompt token IDs differ")
    if (
        stock["canonical_solution_token_ids_sha256"]
        != paro["canonical_solution_token_ids_sha256"]
    ):
        raise ValueError("stock and PARO solution token IDs differ")

    stock_task_nll = stock["mean_task_nlls"]
    paro_task_nll = paro["mean_task_nlls"]
    deltas = [
        candidate - baseline
        for baseline, candidate in zip(stock_task_nll, paro_task_nll)
    ]
    ranked = sorted(
        (
            {
                "task_id": task_id,
                "stock_nll_nats_per_token": baseline,
                "paro_nll_nats_per_token": candidate,
                "delta_nats_per_token": delta,
            }
            for task_id, baseline, candidate, delta in zip(
                stock["task_ids"], stock_task_nll, paro_task_nll, deltas
            )
        ),
        key=lambda row: row["delta_nats_per_token"],
        reverse=True,
    )
    baseline_nll = float(stock["mean_nll_nats_per_token"])
    candidate_nll = float(paro["mean_nll_nats_per_token"])
    repeats_match = bool(
        stock["exact_task_nll_repeat"] and paro["exact_task_nll_repeat"]
    )
    report = {
        "schema": "mtplx-paroquant-q35-humaneval-nll-comparison-v2",
        "comparison_script_sha256": _sha256(Path(__file__).resolve()),
        "status": _directional_status(baseline_nll, candidate_nll, repeats_match),
        "repeat_gate_pass": repeats_match,
        "promotion_decision": "not_decided_by_teacher_forced_nll_alone",
        "dataset": stock["dataset"],
        "stock": {
            key: value
            for key, value in stock.items()
            if key
            not in {
                "dataset",
                "task_ids",
                "token_counts",
                "nll_vectors",
                "mean_task_nlls",
            }
        },
        "paro": {
            key: value
            for key, value in paro.items()
            if key
            not in {
                "dataset",
                "task_ids",
                "token_counts",
                "nll_vectors",
                "mean_task_nlls",
            }
        },
        "comparison": {
            "mean_nll_delta_nats_per_token": candidate_nll - baseline_nll,
            "mean_nll_ratio": candidate_nll / baseline_nll,
            "mean_nll_percent_change": 100.0 * (candidate_nll / baseline_nll - 1.0),
            "mean_task_delta_nats_per_token": statistics.mean(deltas),
            "median_task_delta_nats_per_token": statistics.median(deltas),
            "tasks_worse": sum(delta > 0 for delta in deltas),
            "tasks_improved": sum(delta < 0 for delta in deltas),
            "tasks_equal": sum(delta == 0 for delta in deltas),
        },
        "task_deltas_descending": ranked,
        "interpretation_limit": (
            "This compares two compressed artifacts, not PARO with source BF16. "
            "Teacher-forced HumanEval loss is directional likelihood evidence, not "
            "pass@1 or a complete intelligence evaluation. Lower NLL is better."
        ),
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def compare_artifacts(
    baseline_paths: list[Path],
    candidate_paths: list[Path],
    json_out: Path,
    baseline_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    """Compare repeated artifact receipts without implying source-BF16 parity."""
    baseline = _arm_from_receipts(baseline_paths)
    candidate = _arm_from_receipts(candidate_paths)
    if baseline["dataset"] != candidate["dataset"]:
        raise ValueError("baseline and candidate datasets differ")
    if baseline["task_ids"] != candidate["task_ids"]:
        raise ValueError("baseline and candidate task orders differ")
    if baseline["token_counts"] != candidate["token_counts"]:
        raise ValueError("baseline and candidate token counts differ")
    if baseline["tokenizer_identity"] != candidate["tokenizer_identity"]:
        raise ValueError("baseline and candidate tokenizer identities differ")
    if baseline["prompt_token_ids_sha256"] != candidate["prompt_token_ids_sha256"]:
        raise ValueError("baseline and candidate prompt token IDs differ")
    if (
        baseline["canonical_solution_token_ids_sha256"]
        != candidate["canonical_solution_token_ids_sha256"]
    ):
        raise ValueError("baseline and candidate solution token IDs differ")

    baseline_task_nll = baseline["mean_task_nlls"]
    candidate_task_nll = candidate["mean_task_nlls"]
    deltas = [
        proposed - reference
        for reference, proposed in zip(baseline_task_nll, candidate_task_nll)
    ]
    ranked = sorted(
        (
            {
                "task_id": task_id,
                "baseline_nll_nats_per_token": reference,
                "candidate_nll_nats_per_token": proposed,
                "delta_nats_per_token": delta,
            }
            for task_id, reference, proposed, delta in zip(
                baseline["task_ids"], baseline_task_nll, candidate_task_nll, deltas
            )
        ),
        key=lambda row: row["delta_nats_per_token"],
        reverse=True,
    )
    baseline_nll = float(baseline["mean_nll_nats_per_token"])
    candidate_nll = float(candidate["mean_nll_nats_per_token"])
    repeats_match = bool(
        baseline["exact_task_nll_repeat"] and candidate["exact_task_nll_repeat"]
    )
    status = _directional_status(baseline_nll, candidate_nll, repeats_match)

    hidden = {
        "dataset",
        "task_ids",
        "token_counts",
        "tokenizer_identity",
        "prompt_token_ids_sha256",
        "canonical_solution_token_ids_sha256",
        "nll_vectors",
        "mean_task_nlls",
    }
    report = {
        "schema": "mtplx-q35-compressed-artifact-nll-comparison-v2",
        "comparison_script_sha256": _sha256(Path(__file__).resolve()),
        "status": status,
        "repeat_gate_pass": repeats_match,
        "promotion_decision": "not_decided_by_teacher_forced_nll_alone",
        "dataset": baseline["dataset"],
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "baseline": {
            key: value for key, value in baseline.items() if key not in hidden
        },
        "candidate": {
            key: value for key, value in candidate.items() if key not in hidden
        },
        "comparison": {
            "mean_nll_delta_nats_per_token": candidate_nll - baseline_nll,
            "mean_nll_ratio": candidate_nll / baseline_nll,
            "mean_nll_percent_change": 100.0 * (candidate_nll / baseline_nll - 1.0),
            "mean_task_delta_nats_per_token": statistics.mean(deltas),
            "median_task_delta_nats_per_token": statistics.median(deltas),
            "tasks_worse": sum(delta > 0 for delta in deltas),
            "tasks_improved": sum(delta < 0 for delta in deltas),
            "tasks_equal": sum(delta == 0 for delta in deltas),
        },
        "task_deltas_descending": ranked,
        "task_delta_basis": "per-task mean across all receipts in each arm",
        "interpretation_limit": (
            "This compares two compressed artifacts, not either artifact with source "
            "BF16. Teacher-forced HumanEval loss is directional likelihood evidence, "
            "not pass@1 or a complete intelligence evaluation. Lower NLL is better."
        ),
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _load_model_and_tokenizer(loader: str, model_path: Path):
    import mlx.core as mx

    started = time.perf_counter_ns()
    if loader == "stock":
        from mlx_lm.utils import load

        model, tokenizer = load(str(model_path))
        source = "mlx_lm.utils.load"
    else:
        from paroquant.inference.backends.mlx.load import load

        model, tokenizer, is_vlm = load(str(model_path), force_text=True)
        if is_vlm:
            raise RuntimeError("force_text=True unexpectedly loaded a VLM")
        source = "paroquant.inference.backends.mlx.load(force_text=True)"
    mx.eval(model.parameters())
    mx.synchronize()
    return model, tokenizer, source, (time.perf_counter_ns() - started) / 1e9


def _score_task(
    model, tokenizer, row: dict[str, Any]
) -> tuple[float, int, float, str, str]:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    prompt_tokens = tokenizer.encode(str(row["prompt"]), add_special_tokens=True)
    solution_tokens = tokenizer.encode(
        str(row["canonical_solution"]), add_special_tokens=False
    )
    if not prompt_tokens or not solution_tokens:
        raise ValueError(f"{row['task_id']} produced an empty token sequence")

    cache = make_prompt_cache(model)
    started = time.perf_counter_ns()
    prompt = mx.array([prompt_tokens], dtype=mx.int32)
    first_logits = model(prompt, cache=cache)[:, -1:, :].astype(mx.float32)

    if len(solution_tokens) == 1:
        prediction_logits = first_logits
    else:
        teacher = mx.array([solution_tokens[:-1]], dtype=mx.int32)
        teacher_logits = model(teacher, cache=cache).astype(mx.float32)
        prediction_logits = mx.concatenate([first_logits, teacher_logits], axis=1)

    targets = mx.array([solution_tokens], dtype=mx.int32)
    selected = mx.take_along_axis(
        prediction_logits, targets[..., None], axis=-1
    ).squeeze(-1)
    token_nll = mx.logsumexp(prediction_logits, axis=-1) - selected
    total_nll = mx.sum(token_nll)
    mx.eval(total_nll)
    mx.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return (
        float(total_nll.item()) / len(solution_tokens),
        len(solution_tokens),
        elapsed_ms,
        _token_ids_sha256(prompt_tokens),
        _token_ids_sha256(solution_tokens),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    from research_mlx_m5_exact_paths import provenance
    from research_paroquant_q35 import weight_manifest

    rows = _load_rows(args.dataset, args.limit)
    manifest_before = weight_manifest(args.model, args.manifest_mode)
    if (
        args.expected_weight_manifest is not None
        and manifest_before["aggregate_sha256"] != args.expected_weight_manifest
    ):
        raise ValueError(
            "loaded weight manifest mismatch: "
            f"expected {args.expected_weight_manifest}, "
            f"got {manifest_before['aggregate_sha256']}"
        )

    model, tokenizer, loader_source, load_seconds = _load_model_and_tokenizer(
        args.loader, args.model
    )
    tokenizer_identity = _tokenizer_identity(args.model)
    mx.reset_peak_memory()

    task_results: list[dict[str, Any]] = []
    losses: list[float] = []
    token_counts: list[int] = []
    for row in rows:
        nll, token_count, elapsed_ms, prompt_hash, solution_hash = _score_task(
            model, tokenizer, row
        )
        losses.append(nll)
        token_counts.append(token_count)
        task_results.append(
            {
                "task_id": str(row["task_id"]),
                "canonical_solution_tokens": token_count,
                "prompt_token_ids_sha256": prompt_hash,
                "canonical_solution_token_ids_sha256": solution_hash,
                "nll_nats_per_token": nll,
                "elapsed_ms": elapsed_ms,
            }
        )

    manifest_after = weight_manifest(args.model, args.manifest_mode)
    if manifest_after != manifest_before:
        raise RuntimeError("checkpoint weights changed during the benchmark")

    report = {
        "schema": "mtplx-q35-compressed-artifact-humaneval-nll-v3",
        "status": "artifact_local_likelihood_not_source_relative_quality",
        "label": args.label,
        "loader": args.loader,
        "loader_source": loader_source,
        "model_identifier": args.model.name,
        "model_identity": {
            "weight_manifest_before": manifest_before,
            "weight_manifest_after_sha256": manifest_after["aggregate_sha256"],
            "weights_unchanged_during_run": True,
        },
        "dataset": {
            "path_name": args.dataset.name,
            "sha256": _sha256(args.dataset),
            "selection": f"first_{len(rows)}_file_order",
        },
        "tokenizer_identity": tokenizer_identity,
        "protocol": {
            "teacher_forcing": True,
            "candidate_code_executed": False,
            "prompt_add_special_tokens": True,
            "solution_add_special_tokens": False,
            "scored_region": "canonical_solution_tokens_only",
        },
        "aggregate": _summary(losses, token_counts),
        "tasks": task_results,
        "load_seconds": load_seconds,
        "peak_memory_bytes_after_weight_residency": int(mx.get_peak_memory()),
        "provenance": provenance(Path(__file__).resolve()),
        "interpretation_limit": (
            "This measures one compressed artifact, not source-BF16 parity. It can "
            "be compared across artifacts only when the comparator verifies identical "
            "tokenizer assets and every prompt/solution token-ID hash. It is not pass@1 "
            "or a complete intelligence evaluation."
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def downgrade_historical_comparison(
    historical_path: Path, json_out: Path
) -> dict[str, Any]:
    """Retain an old aggregate receipt without presenting it as comparable.

    The archived v1/v2 raw receipts independently tokenized each model and do
    not retain tokenizer-asset or token-ID identities. Their original producer
    source is also unavailable. The numeric outputs remain inspectable below,
    but this function deliberately removes them from the comparison decision
    surface without pretending to rerun a model.
    """

    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    if historical.get("schema") != "mtplx-q35-compressed-artifact-nll-comparison-v2":
        raise ValueError(
            "historical input must be a v2 compressed-artifact NLL comparison"
        )
    preserved = {
        key: historical.get(key)
        for key in (
            "baseline",
            "candidate",
            "comparison",
            "comparison_script_sha256",
            "repeat_gate_pass",
            "status",
            "task_delta_basis",
            "task_deltas_descending",
        )
    }
    report = {
        "schema": "mtplx-q35-compressed-artifact-nll-comparison-v3",
        "status": "historical_not_comparable_missing_tokenizer_and_token_id_provenance",
        "promotion_decision": "not_decided_by_historical_nll",
        "quality_evidence": "not_usable_for_cross_artifact_quality_direction",
        "baseline_name": historical.get("baseline_name"),
        "candidate_name": historical.get("candidate_name"),
        "historical_input_sha256": _sha256(historical_path),
        "historical_raw_producer_source": {
            "status": "unavailable",
            "reason": (
                "The underlying artifact-local v1/v2 receipts do not preserve the "
                "exact producing script alongside tokenizer asset hashes or per-task "
                "prompt/solution token-ID hashes."
            ),
        },
        "post_hoc_interpretation_correction": (
            "No model was rerun or numeric result changed. The legacy aggregate is "
            "retained under historical_unverified_numeric_output, but equal token "
            "counts do not prove equal tokenization, so its percentage and per-task "
            "signs are not cross-artifact decision evidence."
        ),
        "historical_unverified_numeric_output": preserved,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "downgrade-historical-comparison":
        parser = argparse.ArgumentParser()
        parser.add_argument("downgrade_historical_comparison", nargs="?")
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--json-out", type=Path, required=True)
        args = parser.parse_args()
        downgrade_historical_comparison(args.input, args.json_out)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "compare-artifacts":
        parser = argparse.ArgumentParser()
        parser.add_argument("compare_artifacts", nargs="?")
        parser.add_argument("--baseline", type=Path, nargs="+", required=True)
        parser.add_argument("--candidate", type=Path, nargs="+", required=True)
        parser.add_argument("--baseline-name", required=True)
        parser.add_argument("--candidate-name", required=True)
        parser.add_argument("--json-out", type=Path, required=True)
        args = parser.parse_args()
        compare_artifacts(
            args.baseline,
            args.candidate,
            args.json_out,
            args.baseline_name,
            args.candidate_name,
        )
        return
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        parser = argparse.ArgumentParser()
        parser.add_argument("compare", nargs="?")
        parser.add_argument("--stock", type=Path, nargs="+", required=True)
        parser.add_argument("--paro", type=Path, nargs="+", required=True)
        parser.add_argument("--json-out", type=Path, required=True)
        args = parser.parse_args()
        compare_receipts(args.stock, args.paro, args.json_out)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--loader", choices=("stock", "paro"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest-mode", choices=("indexed", "all"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--label", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--expected-weight-manifest")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
