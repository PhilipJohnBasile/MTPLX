#!/usr/bin/env python3
"""Process-isolated Qwen3.6-35B-A3B compressed-artifact receipt.

This measures load, real-model prefill, and fixed-work teacher-forced decode.
Every artifact must consume the same continuation-token sequence so MoE expert
routing cannot be selected by the artifact under test. It does not claim that
any compressed artifact preserves the source model's quality; that requires a
separate source-relative evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

from research_mlx_m5_exact_paths import provenance, sha256


def deterministic_tokens(vocab_size: int, length: int) -> list[int]:
    return [int((1000 + index * 104729) % vocab_size) for index in range(length)]


def weight_manifest(model_dir: Path, mode: str) -> dict:
    if mode == "indexed":
        index_path = model_dir / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        names = sorted(set(index.get("weight_map", {}).values()))
        if not names:
            raise ValueError("checkpoint index contains no weight_map entries")
    else:
        names = sorted(path.name for path in model_dir.glob("*.safetensors"))
        if not names:
            raise ValueError("checkpoint contains no safetensors")

    aggregate = hashlib.sha256()
    shards = []
    for name in names:
        path = model_dir / name
        size = path.stat().st_size
        digest = sha256(path)
        aggregate.update(f"{name}\0{size}\0{digest}\n".encode())
        shards.append({"filename": name, "size_bytes": size, "sha256": digest})
    return {
        "algorithm": "sha256(sorted(filename\\0size_bytes\\0file_sha256\\n))",
        "aggregate_sha256": aggregate.hexdigest(),
        "shard_count": len(shards),
        "total_size_bytes": sum(item["size_bytes"] for item in shards),
        "shards": shards,
    }


def _read_process_receipt(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    supported = {
        "mtplx-paroquant-q35-process-v1",
        "mtplx-q35-compressed-artifact-process-v2",
        "mtplx-q35-compressed-artifact-process-v3",
        "mtplx-q35-compressed-artifact-process-v4",
    }
    if report.get("schema") not in supported:
        raise ValueError(f"{path} is not a supported process receipt")
    return report


def _performance_arm(paths: list[Path]) -> dict:
    if len(paths) < 2:
        raise ValueError("each performance arm requires at least two receipts")
    reports = [_read_process_receipt(path) for path in paths]
    first = reports[0]
    manifest = first["model_identity"]["weight_manifest_after_sha256"]
    prompt_hash = first["prompt"]["token_ids_sha256"]
    continuation_tokens = first["continuation"]["tokens"]
    decode_workload = first["continuation"].get("decode_workload")
    fixed_work_hash = first["continuation"].get("fixed_work_token_ids_sha256")
    fixed_work_source_hash = first["continuation"].get(
        "fixed_work_source_receipt_sha256"
    )
    campaign_id = first.get("campaign", {}).get("id")
    if not campaign_id:
        raise ValueError("fixed-work receipt is missing a campaign id")
    if decode_workload != "teacher_forced_fixed_continuation":
        raise ValueError("performance comparison requires fixed-work decode receipts")
    if not fixed_work_hash or not fixed_work_source_hash:
        raise ValueError("fixed-work decode receipt is missing provenance hashes")
    for path, report in zip(paths, reports):
        if report["model_identity"]["weight_manifest_after_sha256"] != manifest:
            raise ValueError(f"weight-manifest mismatch in {path}")
        if report["prompt"]["token_ids_sha256"] != prompt_hash:
            raise ValueError(f"prompt mismatch in {path}")
        if report["continuation"]["tokens"] != continuation_tokens:
            raise ValueError(f"continuation-length mismatch in {path}")
        if report["continuation"].get("decode_workload") != decode_workload:
            raise ValueError(f"decode-workload mismatch in {path}")
        if (
            report["continuation"].get("fixed_work_token_ids_sha256")
            != fixed_work_hash
        ):
            raise ValueError(f"fixed-work token mismatch in {path}")
        if (
            report["continuation"].get("fixed_work_source_receipt_sha256")
            != fixed_work_source_hash
        ):
            raise ValueError(f"fixed-work source mismatch in {path}")
        if report.get("campaign", {}).get("id") != campaign_id:
            raise ValueError(f"campaign-id mismatch in {path}")

    prefill_ms = [float(report["prefill"]["median_ms"]) for report in reports]
    decode_tps = [
        float(report["continuation"]["decode_tokens_per_second"])
        for report in reports
    ]
    continuation_hashes = [
        str(report["continuation"]["token_ids_sha256"]) for report in reports
    ]
    return {
        "labels": [str(report["label"]) for report in reports],
        "receipt_sha256": [sha256(path) for path in paths],
        "weight_manifest_sha256": manifest,
        "prompt_token_ids_sha256": prompt_hash,
        "continuation_tokens": continuation_tokens,
        "decode_workload": decode_workload,
        "fixed_work_token_ids_sha256": fixed_work_hash,
        "fixed_work_source_receipt_sha256": fixed_work_source_hash,
        "campaign_id": campaign_id,
        "campaign_arms": [str(report["campaign"]["arm"]) for report in reports],
        "campaign_ordinals": [int(report["campaign"]["ordinal"]) for report in reports],
        "started_at_utc": [str(report["campaign"]["started_at_utc"]) for report in reports],
        "finished_at_utc": [str(report["campaign"]["finished_at_utc"]) for report in reports],
        "continuation_token_ids_sha256": continuation_hashes,
        "exact_continuation_repeat": len(set(continuation_hashes)) == 1,
        "prefill_median_ms_by_run": prefill_ms,
        "prefill_mean_ms": statistics.mean(prefill_ms),
        "prefill_relative_range": (
            (max(prefill_ms) - min(prefill_ms)) / statistics.mean(prefill_ms)
        ),
        "decode_tokens_per_second_by_run": decode_tps,
        "decode_mean_tokens_per_second": statistics.mean(decode_tps),
        "decode_relative_range": (
            (max(decode_tps) - min(decode_tps)) / statistics.mean(decode_tps)
        ),
        "load_seconds_by_run": [float(report["load_seconds"]) for report in reports],
        "peak_memory_bytes_by_run": [
            int(report["peak_memory_bytes_after_weight_residency"])
            for report in reports
        ],
    }


def compare_performance_artifacts(
    baseline_paths: list[Path],
    candidate_paths: list[Path],
    json_out: Path,
    baseline_name: str,
    candidate_name: str,
    run_order: list[str],
) -> dict:
    """Compare process-isolated runs without turning speed into quality evidence."""
    if len(run_order) != len(baseline_paths) + len(candidate_paths):
        raise ValueError("run order length does not match the supplied receipts")
    allowed = {baseline_name, candidate_name}
    if set(run_order) != allowed:
        raise ValueError("run order must contain exactly the baseline and candidate names")
    if run_order.count(baseline_name) != len(baseline_paths):
        raise ValueError("run order baseline count does not match baseline receipts")
    if run_order.count(candidate_name) != len(candidate_paths):
        raise ValueError("run order candidate count does not match candidate receipts")

    baseline = _performance_arm(baseline_paths)
    candidate = _performance_arm(candidate_paths)
    if baseline["prompt_token_ids_sha256"] != candidate["prompt_token_ids_sha256"]:
        raise ValueError("baseline and candidate prompts differ")
    if baseline["continuation_tokens"] != candidate["continuation_tokens"]:
        raise ValueError("baseline and candidate continuation lengths differ")
    if (
        baseline["fixed_work_token_ids_sha256"]
        != candidate["fixed_work_token_ids_sha256"]
    ):
        raise ValueError("baseline and candidate fixed-work continuations differ")
    if (
        baseline["fixed_work_source_receipt_sha256"]
        != candidate["fixed_work_source_receipt_sha256"]
    ):
        raise ValueError("baseline and candidate fixed-work sources differ")
    if baseline["campaign_id"] != candidate["campaign_id"]:
        raise ValueError("baseline and candidate campaign ids differ")
    observed = sorted(
        zip(
            baseline["campaign_ordinals"] + candidate["campaign_ordinals"],
            baseline["campaign_arms"] + candidate["campaign_arms"],
        )
    )
    if [ordinal for ordinal, _ in observed] != list(range(1, len(run_order) + 1)):
        raise ValueError("campaign ordinals are not exactly contiguous from one")
    if [arm for _, arm in observed] != run_order:
        raise ValueError("recorded campaign arms do not match declared run order")

    prefill_speedup = baseline["prefill_mean_ms"] / candidate["prefill_mean_ms"]
    decode_speedup = (
        candidate["decode_mean_tokens_per_second"]
        / baseline["decode_mean_tokens_per_second"]
    )
    report = {
        "schema": "mtplx-q35-compressed-artifact-performance-comparison-v2",
        "comparison_script_sha256": sha256(Path(__file__).resolve()),
        "status": "performance_only_not_quality_evidence",
        "promotion_decision": "not_decided_by_performance_alone",
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "run_order": run_order,
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "prefill_speedup_candidate_over_baseline": prefill_speedup,
            "prefill_percent_change": 100.0 * (prefill_speedup - 1.0),
            "decode_speedup_candidate_over_baseline": decode_speedup,
            "decode_percent_change": 100.0 * (decode_speedup - 1.0),
            "same_fixed_work_across_artifacts": True,
        },
        "interpretation_limit": (
            "This is a process-isolated, fixed-work compressed-artifact performance "
            "comparison on one synthetic prompt. Teacher forcing controls token-driven "
            "MoE routing, but this is not source-relative quality or intelligence "
            "evidence."
        ),
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def load(args):
    started = time.perf_counter_ns()
    if args.loader == "stock":
        model, _ = load_model(args.model)
        loader_source = "mlx_lm.utils.load_model"
    else:
        from paroquant.inference.backends.mlx.load import load as load_paro

        model, _, is_vlm = load_paro(str(args.model), force_text=True)
        if is_vlm:
            raise RuntimeError("force_text=True unexpectedly loaded a VLM")
        loader_source = "paroquant.inference.backends.mlx.load(force_text=True)"
    mx.eval(model.parameters())
    mx.synchronize()
    return model, loader_source, (time.perf_counter_ns() - started) / 1e9


def one_prefill(model, prompt: mx.array):
    cache = make_prompt_cache(model)
    started = time.perf_counter_ns()
    logits = model(prompt, cache=cache)[:, -1, :]
    mx.eval(logits)
    mx.synchronize()
    return cache, logits, (time.perf_counter_ns() - started) / 1e6


def array_sha256(value: mx.array) -> str:
    host = np.asarray(value.astype(mx.float32))
    return hashlib.sha256(host.tobytes()).hexdigest()


def fixed_continuation(path: Path, expected_tokens: int) -> tuple[list[int], dict]:
    """Load one frozen continuation and bind it to its source receipt."""
    source = _read_process_receipt(path)
    token_ids = source.get("continuation", {}).get("token_ids")
    if not isinstance(token_ids, list) or not all(isinstance(item, int) for item in token_ids):
        raise ValueError("fixed-continuation receipt has no integer token_ids list")
    if len(token_ids) != expected_tokens:
        raise ValueError(
            "fixed-continuation length mismatch: "
            f"expected {expected_tokens}, got {len(token_ids)}"
        )
    token_bytes = np.asarray(token_ids, dtype=np.int32).tobytes()
    token_hash = hashlib.sha256(token_bytes).hexdigest()
    recorded_hash = source["continuation"].get("token_ids_sha256")
    if recorded_hash != token_hash:
        raise ValueError("fixed-continuation receipt token hash does not match token_ids")
    return token_ids, {
        "fixed_work_source_receipt": path.name,
        "fixed_work_source_receipt_sha256": sha256(path),
        "fixed_work_token_ids_sha256": token_hash,
    }


def run(args):
    started_at_utc = datetime.now(timezone.utc).isoformat()
    config_path = args.model / "config.json"
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    vocab_size = int(text_config["vocab_size"])
    token_ids = deterministic_tokens(vocab_size, args.prompt_length)
    prompt = mx.array([token_ids], dtype=mx.int32)

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

    model, loader_source, load_s = load(args)
    mx.reset_peak_memory()

    warm_cache, warm_logits, first_touch_ms = one_prefill(model, prompt)
    del warm_cache, warm_logits

    prefill_samples_ms = []
    cache = None
    logits = None
    for _ in range(args.prefill_repetitions):
        cache, logits, elapsed_ms = one_prefill(model, prompt)
        prefill_samples_ms.append(elapsed_ms)
    assert cache is not None and logits is not None

    fixed_tokens, fixed_provenance = fixed_continuation(
        args.fixed_continuation_receipt, args.continuation_tokens
    )
    decode_started = time.perf_counter_ns()
    for current in fixed_tokens[:-1]:
        next_logits = model(mx.array([[current]], dtype=mx.int32), cache=cache)[
            :, -1, :
        ]
        mx.eval(next_logits)
    mx.synchronize()
    decode_elapsed_ms = (time.perf_counter_ns() - decode_started) / 1e6

    manifest_after = weight_manifest(args.model, args.manifest_mode)
    if manifest_after != manifest_before:
        raise RuntimeError("checkpoint weights changed during the benchmark")

    token_bytes = np.asarray(token_ids, dtype=np.int32).tobytes()
    continuation_bytes = np.asarray(fixed_tokens, dtype=np.int32).tobytes()
    decode_steps = args.continuation_tokens - 1
    report = {
        "schema": "mtplx-q35-compressed-artifact-process-v4",
        "status": "fixed_work_performance_not_quality_evidence",
        "label": args.label,
        "loader": args.loader,
        "loader_source": loader_source,
        "model_identifier": args.model.name,
        "campaign": {
            "id": args.campaign_id,
            "arm": args.campaign_arm,
            "ordinal": args.run_ordinal,
            "started_at_utc": started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "command_contract": {
                "loader": args.loader,
                "manifest_mode": args.manifest_mode,
                "prompt_length": args.prompt_length,
                "continuation_tokens": args.continuation_tokens,
                "prefill_repetitions": args.prefill_repetitions,
                "fixed_continuation_receipt": args.fixed_continuation_receipt.name,
            },
        },
        "model_identity": {
            "config_sha256": sha256(config_path),
            "weight_manifest_before": manifest_before,
            "weight_manifest_after_sha256": manifest_after["aggregate_sha256"],
            "weights_unchanged_during_run": True,
        },
        "provenance": provenance(Path(__file__).resolve()),
        "prompt": {
            "length": args.prompt_length,
            "token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
            "vocab_size": vocab_size,
        },
        "load_seconds": load_s,
        "prefill": {
            "first_touch_ms": first_touch_ms,
            "samples_ms": prefill_samples_ms,
            "median_ms": statistics.median(prefill_samples_ms),
            "prompt_tokens_per_second": args.prompt_length
            / (statistics.median(prefill_samples_ms) / 1000),
            "last_logits_sha256": array_sha256(logits),
        },
        "continuation": {
            "tokens": args.continuation_tokens,
            "decode_forward_steps": decode_steps,
            "decode_workload": "teacher_forced_fixed_continuation",
            "decode_elapsed_ms": decode_elapsed_ms,
            "decode_tokens_per_second": decode_steps
            / (decode_elapsed_ms / 1000),
            "token_ids_sha256": hashlib.sha256(continuation_bytes).hexdigest(),
            "token_ids": fixed_tokens,
            **fixed_provenance,
        },
        "peak_memory_bytes_after_weight_residency": int(mx.get_peak_memory()),
        "quality_limit": (
            "This compares two compressed artifacts and is not a source-relative "
            "quality or intelligence evaluation."
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "compare-artifacts":
        parser = argparse.ArgumentParser()
        parser.add_argument("compare_artifacts", nargs="?")
        parser.add_argument("--baseline", type=Path, nargs="+", required=True)
        parser.add_argument("--candidate", type=Path, nargs="+", required=True)
        parser.add_argument("--baseline-name", required=True)
        parser.add_argument("--candidate-name", required=True)
        parser.add_argument("--run-order", nargs="+", required=True)
        parser.add_argument("--json-out", type=Path, required=True)
        args = parser.parse_args()
        compare_performance_artifacts(
            args.baseline,
            args.candidate,
            args.json_out,
            args.baseline_name,
            args.candidate_name,
            args.run_order,
        )
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--loader", choices=("stock", "paro"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest-mode", choices=("indexed", "all"), required=True)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--continuation-tokens", type=int, default=128)
    parser.add_argument(
        "--fixed-continuation-receipt", type=Path, required=True
    )
    parser.add_argument("--prefill-repetitions", type=int, default=2)
    parser.add_argument("--label", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--campaign-arm", required=True)
    parser.add_argument("--run-ordinal", type=int, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--expected-weight-manifest")
    args = parser.parse_args()
    if args.prompt_length < 1 or args.continuation_tokens < 2:
        parser.error("prompt length must be positive and continuation at least 2")
    if args.prefill_repetitions < 1:
        parser.error("prefill repetitions must be positive")
    if args.run_ordinal < 1:
        parser.error("run ordinal must be positive")
    run(args)


if __name__ == "__main__":
    main()
