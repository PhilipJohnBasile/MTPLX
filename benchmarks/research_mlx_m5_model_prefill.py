#!/usr/bin/env python3
"""Process-isolated real-model prefill and continuation receipt.

Run the same command under exact base and candidate MLX wheels in A/B/B/A
order. Synthetic token IDs avoid tokenizer or network dependencies; the model,
cache implementation, and complete target forward are real.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

from research_mlx_m5_exact_paths import provenance, sha256

SEED = 20260810


def array_sha256(value: mx.array) -> str:
    host = np.asarray(value.astype(mx.float32))
    return hashlib.sha256(host.tobytes()).hexdigest()


def deterministic_tokens(vocab_size: int, length: int) -> list[int]:
    # A large odd stride avoids an identical-token synthetic path while keeping
    # every ID inside the checkpoint vocabulary.
    return [int((1000 + index * 104729) % vocab_size) for index in range(length)]


def loaded_weight_manifest(model_dir: Path, index_path: Path) -> dict:
    """Hash every shard selected by the checkpoint index.

    The aggregate is SHA256 over sorted UTF-8 records of the form
    ``filename\\0size_bytes\\0file_sha256\\n``. This deliberately excludes
    unrelated safetensors that happen to share the directory.
    """
    if not index_path.exists():
        raise FileNotFoundError(
            "model.safetensors.index.json is required for an auditable run"
        )
    index = json.loads(index_path.read_text())
    filenames = sorted(set(index.get("weight_map", {}).values()))
    if not filenames:
        raise ValueError("checkpoint index contains no weight_map shard entries")

    shards = []
    aggregate = hashlib.sha256()
    for filename in filenames:
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"indexed shard is missing: {filename}")
        size = path.stat().st_size
        digest = sha256(path)
        aggregate.update(f"{filename}\0{size}\0{digest}\n".encode())
        shards.append({"filename": filename, "size_bytes": size, "sha256": digest})
    return {
        "algorithm": "sha256(sorted(filename\\0size_bytes\\0file_sha256\\n))",
        "aggregate_sha256": aggregate.hexdigest(),
        "shard_count": len(shards),
        "total_size_bytes": sum(item["size_bytes"] for item in shards),
        "shards": shards,
    }


def one_prefill(model, prompt: mx.array):
    cache = make_prompt_cache(model)
    started = time.perf_counter_ns()
    last_logits = model(prompt, cache=cache)[:, -1, :]
    mx.eval(last_logits)
    mx.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return cache, last_logits, elapsed_ms


def run(args):
    config_path = args.model / "config.json"
    index_path = args.model / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    vocab_size = int(text_config["vocab_size"])
    tokens = deterministic_tokens(vocab_size, args.prompt_length)
    prompt = mx.array([tokens], dtype=mx.int32)

    weights_before = loaded_weight_manifest(args.model, index_path)
    if (
        args.expected_weight_manifest is not None
        and weights_before["aggregate_sha256"] != args.expected_weight_manifest
    ):
        raise ValueError(
            "loaded weight manifest mismatch: "
            f"expected {args.expected_weight_manifest}, "
            f"got {weights_before['aggregate_sha256']}"
        )

    model, _ = load_model(args.model)
    mx.eval(model.parameters())
    mx.synchronize()
    mx.reset_peak_memory()

    # First touch is reported separately and excluded from the steady result.
    warm_cache, warm_logits, first_touch_ms = one_prefill(model, prompt)
    del warm_cache, warm_logits

    samples_ms = []
    final_cache = None
    final_logits = None
    for _ in range(args.repetitions):
        final_cache, final_logits, elapsed_ms = one_prefill(model, prompt)
        samples_ms.append(elapsed_ms)

    assert final_cache is not None and final_logits is not None
    prefill_logits_sha256 = array_sha256(final_logits)
    prefill_argmax = int(mx.argmax(final_logits, axis=-1).item())
    if args.logits_out is not None:
        args.logits_out.parent.mkdir(parents=True, exist_ok=True)
        mx.save(str(args.logits_out), final_logits.astype(mx.float32))
        mx.synchronize()

    generated = [prefill_argmax]
    current = prefill_argmax
    decode_started = time.perf_counter_ns()
    for _ in range(args.continuation_tokens - 1):
        next_logits = model(mx.array([[current]], dtype=mx.int32), cache=final_cache)[
            :, -1, :
        ]
        mx.eval(next_logits)
        current = int(mx.argmax(next_logits, axis=-1).item())
        generated.append(current)
    mx.synchronize()
    decode_elapsed_ms = (time.perf_counter_ns() - decode_started) / 1e6
    decode_steps = args.continuation_tokens - 1

    weights_after = loaded_weight_manifest(args.model, index_path)
    if weights_after != weights_before:
        raise RuntimeError("indexed model weights changed during the benchmark envelope")

    token_bytes = np.asarray(tokens, dtype=np.int32).tobytes()
    continuation_bytes = np.asarray(generated, dtype=np.int32).tobytes()
    report = {
        "schema": "mtplx-mlx-m5-real-model-prefill-v1",
        "label": args.label,
        "model_identifier": args.model.name,
        "model_identity": {
            "config_sha256": sha256(config_path),
            "index_sha256": sha256(index_path),
            "loaded_weight_manifest_before": weights_before,
            "loaded_weight_manifest_after_sha256": weights_after[
                "aggregate_sha256"
            ],
            "weights_unchanged_during_run": True,
        },
        "provenance": provenance(Path(__file__).resolve()),
        "prompt": {
            "length": args.prompt_length,
            "token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
            "vocab_size": vocab_size,
        },
        "prefill": {
            "first_touch_ms": first_touch_ms,
            "samples_ms": samples_ms,
            "median_ms": statistics.median(samples_ms),
            "mean_ms": statistics.fmean(samples_ms),
            "prompt_tokens_per_second": args.prompt_length
            / (statistics.median(samples_ms) / 1000),
            "last_logits_sha256": prefill_logits_sha256,
            "greedy_next_id": prefill_argmax,
        },
        "continuation": {
            "tokens": args.continuation_tokens,
            "decode_forward_steps": decode_steps,
            "decode_elapsed_ms": decode_elapsed_ms,
            "decode_tokens_per_second": (
                decode_steps / (decode_elapsed_ms / 1000) if decode_steps else None
            ),
            "token_ids_sha256": hashlib.sha256(continuation_bytes).hexdigest(),
            "token_ids": generated,
        },
        "peak_memory_bytes_after_weight_residency": int(mx.get_peak_memory()),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--label", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--logits-out", type=Path)
    parser.add_argument(
        "--expected-weight-manifest",
        help="fail unless the indexed shard manifest has this SHA256",
    )
    args = parser.parse_args()
    if args.prompt_length < 1 or args.continuation_tokens < 2 or args.repetitions < 1:
        parser.error(
            "prompt length and repetitions must be positive; continuation tokens "
            "must be at least 2"
        )
    run(args)


if __name__ == "__main__":
    main()
