#!/usr/bin/env python3
"""Counterbalanced FusionML stream-overlap mechanism probe.

This is a *synthetic mechanism probe*, not a model benchmark.  It deliberately
uses two dense matrices to ask one narrow question: when a CPU-stream matmul
consumes an MLX value made by a GPU operation in the same lazy graph, does an
explicit materialization boundary change the wall-clock envelope?  It cannot
establish a model speedup, generation quality, or a safe MTPLX integration.

The probe is derived from FusionML's ``stream_parallelism_probe.py`` at
``4e7121a8f9c8``.  Compared with that upstream script it counterbalances case
order, retains raw samples, records runtime provenance, and checks numerical
agreement within each algebraically equivalent case family.

Nothing is imported from MLX at module import time.  Running ``--help``,
``py_compile``, and the focused tests does not create a Metal context.  A
hardware run must opt in explicitly with ``--execute``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


SCHEMA = "mtplx-fusionml-stream-overlap-probe-v1"
SOURCE_PROVENANCE = {
    "fusionml_repository": "https://github.com/ommo007/FusionML",
    "fusionml_main_commit": "2e36b64da1fc8e00ea90afa2ec7cc1b9744f0464",
    "fusionml_benchmark_commit": "4e7121a8f9c70fd6512b233c5705374626323471",
    "upstream_script": "benchmarks/python/stream_parallelism_probe.py",
}
RESULT_KIND = "synthetic_stream_overlap_mechanism_probe"
PUBLIC_SAFE_ENVIRONMENT_KEYS = (
    "METAL_DEBUG_ERROR_MODE",
    "METAL_DEVICE_WRAPPER_TYPE",
    "MLX_ENABLE_TF32",
    "MLX_METAL_CAPTURE",
    "MLX_METAL_DEBUG",
    "MLX_METAL_JIT",
    "MLX_METAL_PREWARM",
)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without recording the path."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _safe_command(command: Sequence[str]) -> str | None:
    """Read a small machine status value; failure is evidence, not an error."""

    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _thermal_snapshot() -> dict[str, object]:
    """Collect public-safe, best-effort thermal state without privilege."""

    pmset = _safe_command(("pmset", "-g", "therm"))
    return {
        "available": pmset is not None,
        "source": "pmset -g therm" if pmset is not None else None,
        "raw": pmset,
        "limitation": (
            "This best-effort snapshot is not a calibrated die-temperature "
            "measurement and cannot prove a thermally quiet host."
        ),
    }


def _process_snapshot() -> dict[str, object]:
    """Record aggregate competing-compute state without paths or usernames."""

    try:
        completed = subprocess.run(
            ["ps", "-Ao", "pcpu=,pmem=,comm="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "available": False,
            "limitation": "process snapshot unavailable",
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "limitation": "process snapshot command failed",
        }

    compute_tokens = (
        "python",
        "mlx",
        "metal",
        "ollama",
        "llama",
        "serve",
        "vllm",
        "mps",
    )
    candidates: list[dict[str, object]] = []
    total_cpu = 0.0
    process_count = 0
    for line in completed.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            continue
        process_count += 1
        try:
            cpu = float(fields[0])
            memory = float(fields[1])
        except ValueError:
            continue
        total_cpu += cpu
        executable = Path(fields[2]).name.lower()
        if any(token in executable for token in compute_tokens):
            if "python" in executable:
                category = "python_process"
            elif "mlx" in executable or "metal" in executable:
                category = "mlx_or_metal_process"
            elif any(token in executable for token in ("ollama", "llama", "vllm")):
                category = "llm_runtime_process"
            elif "serve" in executable:
                category = "serving_process"
            else:
                category = "mps_related_process"
            candidates.append(
                {
                    "cpu_percent": round(cpu, 3),
                    "memory_percent": round(memory, 3),
                    "executable_category": category,
                }
            )
    candidates.sort(
        key=lambda item: (float(item["cpu_percent"]), float(item["memory_percent"])),
        reverse=True,
    )
    return {
        "available": True,
        "process_count": process_count,
        "aggregate_cpu_percent": round(total_cpu, 3),
        "candidate_compute_processes": candidates[:16],
        "limitation": (
            "This is a point-in-time host snapshot. It does not establish GPU "
            "ownership, process causality, or a quiet host."
        ),
    }


def _host_snapshot() -> dict[str, object]:
    """Return public-safe device information available before MLX import."""

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "mac_model": _safe_command(("sysctl", "-n", "hw.model")),
        "cpu_brand": _safe_command(("sysctl", "-n", "machdep.cpu.brand_string")),
        "memory_bytes": _safe_command(("sysctl", "-n", "hw.memsize")),
    }


def _find_first(root: Path, patterns: Iterable[str]) -> Path | None:
    for pattern in patterns:
        candidates = sorted(root.rglob(pattern))
        if candidates:
            return candidates[0]
    return None


def _runtime_provenance(mx_module: Any, mlx_module: Any) -> dict[str, object]:
    """Hash the active MLX Python/core/library/metallib components."""

    package_location = getattr(mlx_module, "__file__", None) or getattr(
        mx_module, "__file__", None
    )
    if package_location is None:  # pragma: no cover - an MLX module must have a file.
        raise RuntimeError("could not locate the active MLX package file")
    package_file = Path(package_location).resolve()
    package_root = package_file.parent
    core_file = Path(mx_module.__file__).resolve()
    libmlx = _find_first(package_root, ("libmlx.dylib",))
    metallib = _find_first(package_root, ("mlx.metallib", "*.metallib"))
    device_info: object
    try:
        device_info = (
            mx_module.device_info()
            if hasattr(mx_module, "device_info")
            else mx_module.metal.device_info()
        )
    except Exception as error:  # pragma: no cover - depends on MLX release.
        device_info = {"unavailable": type(error).__name__}

    try:
        package_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - source tree only.
        package_version = None
    result: dict[str, object] = {
        "mlx_version": getattr(mlx_module, "__version__", None) or package_version,
        "mlx_python_sha256": sha256(package_file),
        "mlx_core_sha256": sha256(core_file),
        "mlx_core_filename": core_file.name,
        "device": device_info,
    }
    if libmlx is not None:
        result.update(
            {
                "libmlx_filename": libmlx.name,
                "libmlx_sha256": sha256(libmlx),
            }
        )
    else:
        result["libmlx_sha256"] = None
    if metallib is not None:
        result.update(
            {
                "metallib_filename": metallib.name,
                "metallib_sha256": sha256(metallib),
            }
        )
    else:
        result["metallib_sha256"] = None
    return result


def _balanced_latin_square(size: int) -> list[list[int]]:
    """Construct a Williams-style balanced Latin square for an even case count."""

    if size < 2 or size % 2:
        raise ValueError("balanced order requires an even number of at least two cases")
    first = [0]
    for offset in range(1, size):
        first.append(offset // 2 + 1 if offset % 2 else size - offset // 2)
    return [[(value + row) % size for value in first] for row in range(size)]


def counterbalanced_schedule(case_names: Sequence[str], rounds: int, seed: int) -> list[list[str]]:
    """Predeclare an order with every case in every ordinal position per cycle."""

    if rounds < 1:
        raise ValueError("rounds must be positive")
    if len(set(case_names)) != len(case_names):
        raise ValueError("case names must be unique")
    square = _balanced_latin_square(len(case_names))
    schedule: list[list[str]] = []
    cycle = 0
    while len(schedule) < rounds:
        names = list(case_names)
        random.Random(seed + cycle).shuffle(names)
        for row in square:
            schedule.append([names[index] for index in row])
            if len(schedule) == rounds:
                break
        cycle += 1
    return schedule


def _summary(samples_ms: Sequence[float]) -> dict[str, object]:
    if not samples_ms:
        raise ValueError("cannot summarize no samples")
    ordered = sorted(float(sample) for sample in samples_ms)
    median = statistics.median(ordered)
    return {
        "count": len(ordered),
        "samples_ms": list(samples_ms),
        "min_ms": ordered[0],
        "median_ms": median,
        "mean_ms": statistics.fmean(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "max_ms": ordered[-1],
        "spread_percent_of_median": (
            100.0 * (ordered[-1] - ordered[0]) / median if median else None
        ),
    }


def _parse_ratio(value: str) -> float:
    ratio = float(value)
    if not 0.0 < ratio < 1.0:
        raise argparse.ArgumentTypeError("cpu ratio must be strictly between zero and one")
    return ratio


def _array_sha256(mx_module: Any, np_module: Any, value: Any) -> str:
    host = np_module.asarray(value.astype(mx_module.float32))
    return hashlib.sha256(host.tobytes()).hexdigest()


def _numeric_delta(mx_module: Any, np_module: Any, reference: Any, candidate: Any) -> dict[str, object]:
    reference_host = np_module.asarray(reference.astype(mx_module.float32))
    candidate_host = np_module.asarray(candidate.astype(mx_module.float32))
    difference = candidate_host - reference_host
    return {
        "reference_output_sha256": hashlib.sha256(reference_host.tobytes()).hexdigest(),
        "candidate_output_sha256": hashlib.sha256(candidate_host.tobytes()).hexdigest(),
        "bit_identical_float32_expansion": bool(np_module.array_equal(reference_host, candidate_host)),
        "max_abs_delta": float(np_module.max(np_module.abs(difference))),
        "mean_abs_delta": float(np_module.mean(np_module.abs(difference))),
        "rms_delta": float(np_module.sqrt(np_module.mean(np_module.square(difference)))),
    }


def _build_cases(mx_module: Any, x: Any, weight: Any, cpu_rows: int) -> tuple[dict[str, Callable[[], Any]], dict[str, dict[str, str]]]:
    """Return the upstream cases plus CPU and same-stream controls."""

    def split(h: Any, *, cpu_stream: bool) -> Any:
        cpu_or_gpu = mx_module.cpu if cpu_stream else mx_module.gpu
        left = mx_module.matmul(h[:cpu_rows], weight, stream=cpu_or_gpu)
        right = mx_module.matmul(h[cpu_rows:], weight, stream=mx_module.gpu)
        return mx_module.concatenate([left, right], axis=0)

    def gpu_ready() -> Any:
        return mx_module.matmul(x, weight, stream=mx_module.gpu)

    def same_stream_ready() -> Any:
        return split(x, cpu_stream=False)

    def split_ready() -> Any:
        return split(x, cpu_stream=True)

    def gpu_dependency() -> Any:
        return mx_module.matmul(mx_module.maximum(x, 0), weight, stream=mx_module.gpu)

    def same_stream_dependency() -> Any:
        return split(mx_module.maximum(x, 0), cpu_stream=False)

    def split_lazy_dependency() -> Any:
        return split(mx_module.maximum(x, 0), cpu_stream=True)

    def split_materialized_boundary() -> Any:
        hidden = mx_module.maximum(x, 0)
        mx_module.eval(hidden)
        mx_module.synchronize()
        return split(hidden, cpu_stream=True)

    def gpu_materialized_boundary() -> Any:
        hidden = mx_module.maximum(x, 0)
        mx_module.eval(hidden)
        mx_module.synchronize()
        return mx_module.matmul(hidden, weight, stream=mx_module.gpu)

    cases = {
        "gpu_only_ready": gpu_ready,
        "same_stream_ready": same_stream_ready,
        "split_ready": split_ready,
        "gpu_only_dependency": gpu_dependency,
        "same_stream_lazy_dependency": same_stream_dependency,
        "split_lazy_dependency": split_lazy_dependency,
        "split_materialized_boundary": split_materialized_boundary,
        "gpu_materialized_boundary": gpu_materialized_boundary,
    }
    descriptions = {
        "gpu_only_ready": {"family": "ready", "stream_layout": "gpu_only", "boundary": "none"},
        "same_stream_ready": {"family": "ready", "stream_layout": "two_gpu_stream_ops", "boundary": "none"},
        "split_ready": {"family": "ready", "stream_layout": "cpu_plus_gpu", "boundary": "none"},
        "gpu_only_dependency": {"family": "dependency", "stream_layout": "gpu_only", "boundary": "lazy"},
        "same_stream_lazy_dependency": {"family": "dependency", "stream_layout": "two_gpu_stream_ops", "boundary": "lazy"},
        "split_lazy_dependency": {"family": "dependency", "stream_layout": "cpu_plus_gpu", "boundary": "lazy"},
        "split_materialized_boundary": {"family": "dependency", "stream_layout": "cpu_plus_gpu", "boundary": "explicit"},
        "gpu_materialized_boundary": {"family": "dependency", "stream_layout": "gpu_only", "boundary": "explicit"},
    }
    return cases, descriptions


def _timed(mx_module: Any, operation: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    value = operation()
    mx_module.eval(value)
    mx_module.synchronize()
    return value, (time.perf_counter_ns() - started) / 1e6


def _run_hardware_probe(args: argparse.Namespace) -> dict[str, object]:
    """Import MLX only after the caller explicitly opts into a hardware run."""

    mlx_module = importlib.import_module("mlx")
    mx_module = importlib.import_module("mlx.core")
    np_module = importlib.import_module("numpy")

    rng = np_module.random.default_rng(args.seed)
    x_host = (rng.standard_normal((args.rows, args.inner)) * 0.02).astype(np_module.float16)
    weight_host = (rng.standard_normal((args.inner, args.columns)) * 0.02).astype(np_module.float16)
    x = mx_module.array(x_host)
    weight = mx_module.array(weight_host)
    mx_module.eval(x, weight)
    mx_module.synchronize()
    cpu_rows = int(args.rows * args.cpu_ratio)
    if not 0 < cpu_rows < args.rows:
        raise ValueError("selected dimensions and cpu ratio leave an empty stream partition")

    cases, case_descriptions = _build_cases(mx_module, x, weight, cpu_rows)
    case_names = list(cases)
    if len(case_names) % 2:
        raise AssertionError("counterbalance case count must remain even")
    warmup_schedule = counterbalanced_schedule(case_names, args.warmup_rounds, args.seed - 1)
    measurement_schedule = counterbalanced_schedule(case_names, args.rounds, args.seed)
    first_touch_order = list(case_names)
    random.Random(args.seed - 2).shuffle(first_touch_order)
    snapshots_before = {
        "host": _host_snapshot(),
        "thermal": _thermal_snapshot(),
        "processes": _process_snapshot(),
    }

    first_touch_ms: dict[str, float] = {}
    for name in first_touch_order:
        _, elapsed_ms = _timed(mx_module, cases[name])
        first_touch_ms[name] = elapsed_ms
    for round_order in warmup_schedule:
        for name in round_order:
            _timed(mx_module, cases[name])
    mx_module.reset_peak_memory()

    raw_trials: list[dict[str, object]] = []
    samples: dict[str, list[float]] = {name: [] for name in case_names}
    outputs: dict[str, Any] = {}
    for round_index, round_order in enumerate(measurement_schedule):
        for ordinal, name in enumerate(round_order):
            output, elapsed_ms = _timed(mx_module, cases[name])
            samples[name].append(elapsed_ms)
            outputs[name] = output
            raw_trials.append(
                {
                    "round": round_index,
                    "ordinal": ordinal,
                    "case": name,
                    "elapsed_ms": elapsed_ms,
                }
            )
    mx_module.synchronize()
    snapshots_after = {
        "thermal": _thermal_snapshot(),
        "processes": _process_snapshot(),
    }

    references = {"ready": "gpu_only_ready", "dependency": "gpu_only_dependency"}
    case_reports: dict[str, object] = {}
    for name in case_names:
        family = case_descriptions[name]["family"]
        reference = references[family]
        case_reports[name] = {
            **case_descriptions[name],
            "first_touch_ms": first_touch_ms[name],
            "timing": _summary(samples[name]),
            "output_sha256": _array_sha256(mx_module, np_module, outputs[name]),
            "numeric_delta_vs_family_gpu_reference": _numeric_delta(
                mx_module, np_module, outputs[reference], outputs[name]
            ),
        }

    split_ready = statistics.median(samples["split_ready"])
    split_lazy = statistics.median(samples["split_lazy_dependency"])
    split_boundary = statistics.median(samples["split_materialized_boundary"])
    gpu_dependency = statistics.median(samples["gpu_only_dependency"])
    report = {
        "schema": SCHEMA,
        "result_kind": RESULT_KIND,
        "status": "synthetic_measurement_not_model_or_quality_evidence",
        "scope": {
            "model_loaded": False,
            "generation_run": False,
            "quality_evaluated": False,
            "speed_claim": (
                "No model or MTPLX speed claim follows from this result. It tests "
                "only a synthetic MLX CPU/GPU dependency mechanism."
            ),
        },
        "source_provenance": SOURCE_PROVENANCE,
        "benchmark_script_sha256": sha256(Path(__file__).resolve()),
        "runtime_provenance": _runtime_provenance(mx_module, mlx_module),
        "environment": {
            key: os.environ[key]
            for key in PUBLIC_SAFE_ENVIRONMENT_KEYS
            if key in os.environ
        },
        "host_snapshots": {"before": snapshots_before, "after": snapshots_after},
        "fixture": {
            "seed": args.seed,
            "rows": args.rows,
            "inner": args.inner,
            "columns": args.columns,
            "dtype": "float16",
            "cpu_ratio": args.cpu_ratio,
            "cpu_rows": cpu_rows,
            "gpu_rows": args.rows - cpu_rows,
            "input_sha256": hashlib.sha256(x_host.tobytes()).hexdigest(),
            "weight_sha256": hashlib.sha256(weight_host.tobytes()).hexdigest(),
        },
        "schedule": {
            "kind": "seeded_williams_balanced_latin_square",
            "seed": args.seed,
            "warmup_rounds": args.warmup_rounds,
            "measurement_rounds": args.rounds,
            "first_touch_order": first_touch_order,
            "warmup_schedule": warmup_schedule,
            "measurement_schedule": measurement_schedule,
            "measurement_schedule_sha256": _canonical_sha256(measurement_schedule),
        },
        "raw_trials": raw_trials,
        "cases": case_reports,
        "mechanism_observations": {
            "ready_split_vs_gpu_only_ratio": statistics.median(samples["gpu_only_ready"])
            / split_ready,
            "lazy_split_vs_dependency_gpu_only_ratio": gpu_dependency / split_lazy,
            "materialized_split_vs_dependency_gpu_only_ratio": gpu_dependency / split_boundary,
            "materialized_split_vs_lazy_split_ratio": split_lazy / split_boundary,
            "interpretation_limit": (
                "Ratios describe this synthetic fixture only. They are not a model "
                "speedup, a proof of overlap, or a recommendation to introduce an "
                "eager boundary into a production graph."
            ),
        },
        "peak_memory_bytes_during_measurement": int(mx_module.get_peak_memory()),
    }
    report["report_payload_sha256"] = _canonical_sha256(report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="explicitly import MLX and run the Metal/CPU mechanism probe",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--inner", type=int, default=1600)
    parser.add_argument("--columns", type=int, default=6400)
    parser.add_argument("--cpu-ratio", type=_parse_ratio, default=0.35)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args(argv)
    if min(args.rows, args.inner, args.columns) < 1:
        parser.error("matrix dimensions must be positive")
    if args.warmup_rounds < 0 or args.rounds < 1:
        parser.error("warmup rounds must be nonnegative and rounds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing to touch MLX/Metal without --execute")
    report = _run_hardware_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
