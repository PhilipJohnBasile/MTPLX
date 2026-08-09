#!/usr/bin/env python3
"""Run an isolated served B8 numerics throughput bracket under the GPU guard."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any
from urllib import request

if __package__:
    from scripts.qwen35b_mtp_batch_numerics_attribution import (
        _QWEN_ROUTE_ENV,
        _assert_no_other_model_runner,
        _validate_qwen_model,
        _verify_parent_guard_attestation,
    )
else:
    from qwen35b_mtp_batch_numerics_attribution import (
        _QWEN_ROUTE_ENV,
        _assert_no_other_model_runner,
        _validate_qwen_model,
        _verify_parent_guard_attestation,
    )


_PROFILE_ROUTE_IDS = {
    "throughput": "qwen35b_a3b_mtp_batch_b8_t2_m16_throughput",
    "balanced": "qwen35b_a3b_mtp_batch_b8_t2_l0_b1_qkv_z_b_balanced",
    "b1-exact": "qwen35b_mtp_batch_b1_exact_serial",
}


def summarize_round(
    responses: list[dict[str, Any]], *, wall_s: float
) -> dict[str, Any]:
    response_ids = [str(item.get("id") or "") for item in responses]
    if any(not value for value in response_ids) or len(set(response_ids)) != len(
        response_ids
    ):
        raise RuntimeError("benchmark responses require unique non-empty IDs")
    completion_tokens = sum(
        int((item.get("usage") or {}).get("completion_tokens") or 0)
        for item in responses
    )
    if completion_tokens <= 0 or wall_s <= 0:
        raise RuntimeError("benchmark round produced no measurable output")
    return {
        "requests": len(responses),
        "completion_tokens": completion_tokens,
        "wall_s": float(wall_s),
        "aggregate_output_tps": completion_tokens / float(wall_s),
        "unique_response_ids": len(set(response_ids)),
    }


def summarize_paired_round(
    *, serial: dict[str, Any], b8: dict[str, Any]
) -> dict[str, float | int]:
    serial_tokens = int(serial["completion_tokens"])
    serial_wall = float(serial["wall_s"])
    b8_tokens = int(b8["completion_tokens"])
    b8_wall = float(b8["wall_s"])
    serial_tps = serial_tokens / serial_wall
    b8_tps = b8_tokens / b8_wall
    return {
        "serial_completion_tokens": serial_tokens,
        "serial_wall_s": serial_wall,
        "serial_aggregate_output_tps": serial_tps,
        "b8_completion_tokens": b8_tokens,
        "b8_wall_s": b8_wall,
        "b8_aggregate_output_tps": b8_tps,
        "speedup": b8_tps / serial_tps,
    }


def validate_health_profile(health: dict[str, Any], *, expected: str) -> dict[str, Any]:
    scheduler = health.get("scheduler") or {}
    installed = str(scheduler.get("mtp_batch_numerics") or "")
    if installed != expected:
        raise RuntimeError(
            f"benchmark requested {expected} but server installed {installed or 'none'}"
        )
    route_id = str(scheduler.get("mtp_batch_route_id") or "")
    expected_route = _PROFILE_ROUTE_IDS[expected]
    if route_id != expected_route:
        raise RuntimeError(
            f"benchmark requested route {expected_route} but server installed "
            f"{route_id or 'none'}"
        )
    return scheduler


def validate_b8_benchmark_health(
    health: dict[str, Any], *, expected: str
) -> dict[str, Any]:
    """Require the private benchmark traffic to have formed only real B8 cohorts."""

    scheduler = validate_health_profile(health, expected=expected)
    if expected == "b1-exact":
        return scheduler
    mtp_batch = scheduler.get("mtp_batch") or {}
    batch_histogram = mtp_batch.get("batch_histogram") or {}
    real_widths = {
        int(width) for width, count in batch_histogram.items() if int(count) > 0
    }
    unexpected_widths = sorted(real_widths - {1, 8})
    if 8 not in real_widths or unexpected_widths:
        raise RuntimeError(
            "benchmark real cohort widths must contain only serial B1 and B8; "
            f"unexpected widths: {unexpected_widths or sorted(real_widths)}"
        )
    fixed_width_histogram = mtp_batch.get("fixed_width_histogram") or {}
    fixed_widths = {
        int(width) for width, count in fixed_width_histogram.items() if int(count) > 0
    }
    if fixed_widths != {8}:
        raise RuntimeError(
            f"benchmark physical widths must be exactly B8; got {sorted(fixed_widths)}"
        )
    route_id = str(scheduler["mtp_batch_route_id"])
    if str(mtp_batch.get("last_route_id") or "") != route_id:
        raise RuntimeError("benchmark last cohort did not execute the selected route")
    return scheduler


def _json_get(url: str, *, timeout: float = 3.0) -> dict[str, Any]:
    with request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _wait_health(base_url: str, process: subprocess.Popen[Any], timeout: float):
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"benchmark server exited with {process.returncode}")
        try:
            payload = _json_get(f"{base_url}/health")
            if payload.get("ok"):
                return payload
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"benchmark server was not healthy: {last_error}")


def completion_payload(
    *, row: int, mode: str, max_tokens: int, workload: str
) -> tuple[dict[str, Any], str]:
    if workload == "legacy":
        marker = f"ROW_{row}_ONLY"
        content = (
            f"Begin with the exact marker {marker}. Explain why deterministic "
            "concurrent request ownership matters in a model server. Do not "
            "mention any other marker."
        )
        seed = 4200 + row
        top_p = 1.0 if mode == "greedy" else 0.95
    else:
        marker = f"NUMERICS_ROW_{row}_ONLY"
        content = (
            "Return only Python code. Define a function named "
            f"solve_{row}(values) that sorts integers, removes duplicates, "
            "and returns the running sums. Include the exact comment "
            f"# {marker}. Add a short doctest."
        )
        seed = 4100 + row
        top_p = 0.95
    payload: dict[str, Any] = {
        "model": "qwen35b-mtp-b8-numerics",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(max_tokens),
        "temperature": 0.0 if mode == "greedy" else 0.6,
        "top_p": top_p,
        "top_k": 20,
        "seed": seed,
        "stream": False,
    }
    return payload, marker


def _completion(
    base_url: str,
    *,
    row: int,
    mode: str,
    max_tokens: int,
    workload: str,
    barrier: threading.Barrier | None,
) -> dict[str, Any]:
    payload, marker = completion_payload(
        row=row,
        mode=mode,
        max_tokens=max_tokens,
        workload=workload,
    )
    body = json.dumps(payload).encode()
    call = request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-MTPLX-Request-ID": f"numerics-{mode}-{row}-{time.monotonic_ns()}",
        },
        method="POST",
    )
    if barrier is not None:
        barrier.wait(timeout=30)
    started = time.perf_counter()
    with request.urlopen(call, timeout=300) as response:
        result = json.load(response)
    result["_client_elapsed_s"] = time.perf_counter() - started
    text = str(
        ((result.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    )
    result["_output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    result["_marker_isolated"] = marker in text and all(
        f"ROW_{other}_ONLY" not in text for other in range(8) if other != row
    )
    return result


def _run_round(
    base_url: str,
    *,
    mode: str,
    max_tokens: int,
    workload: str,
    cohort: bool,
    synchronize_cohort: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    if cohort:
        barrier = (
            threading.Barrier(8) if workload != "legacy" or synchronize_cohort else None
        )
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(
                    _completion,
                    base_url,
                    row=row,
                    mode=mode,
                    max_tokens=max_tokens,
                    workload=workload,
                    barrier=barrier,
                )
                for row in range(8)
            ]
            responses = [future.result() for future in futures]
    else:
        responses = [
            _completion(
                base_url,
                row=row,
                mode=mode,
                max_tokens=max_tokens,
                workload=workload,
                barrier=None,
            )
            for row in range(8)
        ]
    summary = summarize_round(responses, wall_s=time.perf_counter() - started)
    summary["output_sha256"] = [item["_output_sha256"] for item in responses]
    summary["request_elapsed_s"] = [
        float(item["_client_elapsed_s"]) for item in responses
    ]
    summary["marker_isolation"] = all(item["_marker_isolated"] for item in responses)
    return responses, summary


def _server_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.mtplx),
        "serve",
        "--model",
        str(args.model),
        "--model-id",
        "qwen35b-mtp-b8-numerics",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--context-window",
        "131072",
        "--generation-mode",
        "mtp",
        "--scheduler-mode",
        "mtp_batch",
        "--max-active-requests",
        "8",
        "--decode-batch-max",
        "8",
        "--batching-preset",
        "throughput",
        "--mtp-batch-numerics",
        args.numerics,
        "--depth",
        "1",
        "--verify-strategy",
        "target_prefix",
        "--verify-core",
        "stock",
        "--profile",
        "turbo",
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--draft-temperature",
        "0.6",
        "--draft-top-p",
        "0.95",
        "--draft-top-k",
        "20",
        "--ssd-session-cache",
        "off",
        "--reasoning",
        "off",
        "--reasoning-parser",
        "qwen3",
        "--preserve-thinking",
        "off",
        "--chat-template-path",
        str(args.chat_template),
        "--tool-prompt-mode",
        "native",
        "--warmup-tokens",
        "16",
        "--rate-limit",
        "0",
        "--no-stats-footer",
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--mtplx", required=True, type=Path)
    parser.add_argument("--chat-template", required=True, type=Path)
    parser.add_argument("--lock", default="/tmp/mtplx-gpu-exclusive.lock", type=Path)
    parser.add_argument(
        "--numerics",
        choices=("throughput", "balanced", "b1-exact"),
        required=True,
    )
    parser.add_argument("--mode", choices=("greedy", "default"), required=True)
    parser.add_argument("--workload", choices=("coding", "legacy"), default="coding")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.model = args.model.expanduser().resolve()
    _validate_qwen_model(args.model)
    if not _verify_parent_guard_attestation(args.lock):
        raise RuntimeError("benchmark must run under the attested GPU guard")
    _assert_no_other_model_runner()
    env = os.environ.copy()
    env.update(_QWEN_ROUTE_ENV)
    log_path = Path(
        f"/tmp/qwen35b-mtp-b8-{args.numerics}-{args.workload}-{args.mode}.log"
    )
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        _server_command(args),
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        startup_health = _wait_health(base_url, process, 240)
        validate_health_profile(startup_health, expected=args.numerics)
        _run_round(
            base_url,
            mode=args.mode,
            max_tokens=32,
            workload=args.workload,
            cohort=True,
            synchronize_cohort=True,
        )
        rounds = []
        for index in range(args.rounds):
            if args.workload == "legacy":
                _, serial = _run_round(
                    base_url,
                    mode=args.mode,
                    max_tokens=args.max_tokens,
                    workload=args.workload,
                    cohort=False,
                )
                responses, b8 = _run_round(
                    base_url,
                    mode=args.mode,
                    max_tokens=args.max_tokens,
                    workload=args.workload,
                    cohort=True,
                )
                summary = summarize_paired_round(serial=serial, b8=b8)
                summary["serial_marker_isolation"] = serial["marker_isolation"]
                summary["b8_marker_isolation"] = b8["marker_isolation"]
                summary["serial_output_sha256"] = serial["output_sha256"]
                summary["b8_output_sha256"] = b8["output_sha256"]
                summary["hash_parity"] = serial["output_sha256"] == b8["output_sha256"]
            else:
                responses, summary = _run_round(
                    base_url,
                    mode=args.mode,
                    max_tokens=args.max_tokens,
                    workload=args.workload,
                    cohort=True,
                )
            summary["round"] = index + 1
            summary["finish_reasons"] = [
                (item.get("choices") or [{}])[0].get("finish_reason")
                for item in responses
            ]
            rounds.append(summary)
        final_health = _json_get(f"{base_url}/health", timeout=5)
        validate_b8_benchmark_health(final_health, expected=args.numerics)
        tps_field = (
            "b8_aggregate_output_tps"
            if args.workload == "legacy"
            else "aggregate_output_tps"
        )
        tps_values = [float(item[tps_field]) for item in rounds]
        scheduler = final_health.get("scheduler") or {}
        mtp_batch = scheduler.get("mtp_batch") or {}
        receipt = {
            "schema_version": 1,
            "profile": args.numerics,
            "mode": args.mode,
            "workload": args.workload,
            "rounds": rounds,
            "median_aggregate_output_tps": statistics.median(tps_values),
            "route_id": scheduler.get("mtp_batch_route_id"),
            "config_fingerprint": scheduler.get("mtp_batch_config_fingerprint"),
            "construction_receipt": scheduler.get("mtp_batch_construction_receipt"),
            "batch_histogram": mtp_batch.get("batch_histogram"),
            "fixed_width_histogram": mtp_batch.get("fixed_width_histogram"),
            "last_route_id": mtp_batch.get("last_route_id"),
            "startup_model": startup_health.get("model"),
            "log_path": str(log_path),
        }
        if args.workload == "legacy":
            serial_tps = [float(item["serial_aggregate_output_tps"]) for item in rounds]
            receipt["median_serial_aggregate_output_tps"] = statistics.median(
                serial_tps
            )
            receipt["speedup_of_medians"] = (
                receipt["median_aggregate_output_tps"]
                / receipt["median_serial_aggregate_output_tps"]
            )
        canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(receipt, sort_keys=True))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        log_handle.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"guarded benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
