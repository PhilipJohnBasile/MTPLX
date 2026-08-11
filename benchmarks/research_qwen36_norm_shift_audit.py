#!/usr/bin/env python3
"""Spot-check converted Qwen3.6 norm tensors for cross-artifact drift.

This does not prove source-checkpoint fidelity. It detects candidate-specific
normalization changes and records whether representative tensors match the
pinned local baseline byte-for-byte after float32 expansion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from research_mlx_m5_exact_paths import provenance
from research_paroquant_q35 import weight_manifest


DEFAULT_KEYS = (
    "language_model.model.layers.0.input_layernorm.weight",
    "language_model.model.layers.0.post_attention_layernorm.weight",
    "language_model.model.norm.weight",
)


def _parse_artifact(spec: str) -> tuple[str, Path, str, str]:
    parts = spec.split("::")
    if len(parts) != 4:
        raise ValueError(
            "artifact must be LABEL::PATH::MANIFEST_MODE::EXPECTED_MANIFEST"
        )
    label, raw_path, mode, expected = parts
    if not label or not raw_path or not expected or mode not in {"indexed", "all"}:
        raise ValueError(f"invalid artifact specification: {spec!r}")
    return label, Path(raw_path), mode, expected


def _tensor_record(value: mx.array) -> dict[str, Any]:
    host = np.asarray(value.astype(mx.float32))
    return {
        "shape": list(host.shape),
        "float32_sha256": hashlib.sha256(host.tobytes()).hexdigest(),
        "min": float(host.min()),
        "max": float(host.max()),
        "mean": float(host.mean()),
    }


def _comparison(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    if len(artifacts) < 2:
        raise ValueError("at least two artifacts are required")
    baseline = artifacts[0]
    baseline_tensors = baseline["tensors"]
    matches: dict[str, dict[str, bool]] = {}
    for artifact in artifacts[1:]:
        matches[artifact["label"]] = {
            key: artifact["tensors"][key]["float32_sha256"]
            == baseline_tensors[key]["float32_sha256"]
            for key in baseline_tensors
        }
    return {
        "baseline_label": baseline["label"],
        "candidate_tensor_matches": matches,
        "all_representative_tensors_match": all(
            matched for per_artifact in matches.values() for matched in per_artifact.values()
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = []
    for spec in args.artifact:
        label, model_path, manifest_mode, expected_manifest = _parse_artifact(spec)
        manifest = weight_manifest(model_path, manifest_mode)
        if manifest["aggregate_sha256"] != expected_manifest:
            raise ValueError(
                f"{label} manifest mismatch: expected {expected_manifest}, "
                f"got {manifest['aggregate_sha256']}"
            )
        index = json.loads(
            (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
        )["weight_map"]
        tensors: dict[str, dict[str, Any]] = {}
        loaded_shards: dict[str, dict[str, mx.array]] = {}
        for key in args.tensor_key:
            if key not in index:
                raise ValueError(f"{label} index lacks {key!r}")
            shard_name = index[key]
            if shard_name not in loaded_shards:
                loaded_shards[shard_name] = mx.load(str(model_path / shard_name))
            tensors[key] = _tensor_record(loaded_shards[shard_name][key])
        artifacts.append(
            {
                "label": label,
                "model_identifier": model_path.name,
                "weight_manifest_sha256": expected_manifest,
                "tensors": tensors,
            }
        )

    report = {
        "schema": "mtplx-qwen36-norm-shift-spotcheck-v1",
        "status": "cross_artifact_spotcheck_not_source_fidelity_proof",
        "artifacts": artifacts,
        "comparison": _comparison(artifacts),
        "provenance": provenance(Path(__file__).resolve()),
        "interpretation_limit": (
            "Matching representative norm tensors rules out candidate-specific drift "
            "at those keys. It does not prove that the shared values match the original "
            "source checkpoint or that every norm tensor is correct."
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--tensor-key", action="append", default=list(DEFAULT_KEYS))
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.artifact) < 2:
        parser.error("at least two --artifact values are required")
    run(args)


if __name__ == "__main__":
    main()
