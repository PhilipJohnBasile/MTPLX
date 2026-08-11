#!/usr/bin/env python3
"""Reproduce the exact-path MLX M5 microbenchmarks.

Generate each fixture once with ``--make-fixture``, then run the same fixture
under two process-isolated MLX installations in A/B/B/A order. The two gather
cases are independent timing scopes; do not subtract one from the other.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path

import mlx
import mlx.core as mx
import numpy as np

SEED = 20260810
SAFE_ENVIRONMENT_KEYS = (
    "METAL_DEBUG_ERROR_MODE",
    "METAL_DEVICE_WRAPPER_TYPE",
    "MLX_ENABLE_TF32",
    "MLX_METAL_CAPTURE",
    "MLX_METAL_DEBUG",
    "MLX_METAL_JIT",
    "MLX_METAL_PREWARM",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(fn):
    value = fn()
    mx.eval(value)
    mx.synchronize()
    return value


def timed(fn):
    start = time.perf_counter_ns()
    value = evaluate(fn)
    return value, (time.perf_counter_ns() - start) / 1e6


def summarize(samples):
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_ms": ordered[-1],
        "samples_ms": samples,
    }


def provenance(benchmark_script: Path | None = None):
    package_file = getattr(mlx, "__file__", None) or mx.__file__
    package_dir = Path(package_file).resolve().parent
    dylibs = sorted(package_dir.rglob("libmlx.dylib"))
    extensions = sorted(package_dir.rglob("core*.so"))
    metallibs = sorted(package_dir.rglob("mlx.metallib"))
    result = {
        "benchmark_script_sha256": sha256(
            benchmark_script or Path(__file__).resolve()
        ),
        "mlx_version": getattr(mlx, "__version__", None)
        or importlib.metadata.version("mlx"),
        "package": "mlx",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": (
            mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        ),
    }
    if dylibs:
        result["libmlx"] = dylibs[0].name
        result["libmlx_sha256"] = sha256(dylibs[0].resolve())
    if extensions:
        result["core_extension"] = extensions[0].name
        result["core_extension_sha256"] = sha256(extensions[0].resolve())
    if metallibs:
        result["metallib"] = metallibs[0].name
        result["metallib_sha256"] = sha256(metallibs[0].resolve())
    return result


def make_gather_fixture(path: Path):
    # The same 16 rows/expert regime used by MLX PR #4023.
    n, experts, routes, dim, hidden = 512, 256, 8, 512, 512
    kx, kw1, kw2 = mx.random.split(mx.random.key(SEED), 3)
    x = (mx.random.normal((n, 1, 1, dim), key=kx) / dim**0.5).astype(mx.float16)
    w1 = (mx.random.normal((experts, hidden, dim), key=kw1) / dim**0.5).astype(
        mx.float16
    )
    w1q, w1s, w1b = mx.quantize(w1, group_size=64, bits=4, mode="affine")
    del w1
    w2 = (mx.random.normal((experts, dim, hidden), key=kw2) / hidden**0.5).astype(
        mx.float16
    )
    w2q, w2s, w2b = mx.quantize(w2, group_size=64, bits=4, mode="affine")
    del w2

    rng = np.random.default_rng(SEED)
    flat = np.tile(np.arange(experts, dtype=np.uint32), n * routes // experts)
    rng.shuffle(flat)
    indices = mx.array(flat.reshape(n, routes))
    mx.eval(x, w1q, w1s, w1b, w2q, w2s, w2b, indices)
    mx.savez(
        str(path),
        x=x,
        w1q=w1q,
        w1s=w1s,
        w1b=w1b,
        w2q=w2q,
        w2s=w2s,
        w2b=w2b,
        indices=indices,
    )
    mx.synchronize()


def make_nvfp4_fixture(path: Path):
    kx, kw = mx.random.split(mx.random.key(SEED + 1))
    x = (mx.random.normal((1, 4096), key=kx) / 4096**0.5).astype(mx.float16)
    weight = mx.random.normal((8192, 4096), key=kw).astype(mx.float16)
    packed, scales = mx.quantize(weight, mode="nvfp4")
    mx.eval(x, packed, scales)
    mx.savez(str(path), x=x, packed=packed, scales=scales)
    mx.synchronize()


def make_qmm_fixture(path: Path, args):
    kx, kw = mx.random.split(mx.random.key(SEED + 2))
    x = (mx.random.normal((args.m, args.k), key=kx) / args.k**0.5).astype(mx.float16)
    weight = (mx.random.normal((args.n, args.k), key=kw) / args.k**0.5).astype(
        mx.float16
    )
    packed, scales, biases = mx.quantize(
        weight, group_size=args.group_size, bits=args.bits, mode="affine"
    )
    del weight
    mx.eval(x, packed, scales, biases)
    mx.savez(
        str(path),
        x=x,
        packed=packed,
        scales=scales,
        biases=biases,
        group_size=mx.array(args.group_size),
        bits=mx.array(args.bits),
    )
    mx.synchronize()


def make_sdpa_fixture(path: Path, args):
    kq, kk, kv = mx.random.split(mx.random.key(SEED + 3), 3)
    scale = args.head_dim**-0.5
    q = (
        5e-1
        * mx.random.normal(
            (1, args.query_heads, args.query_length, args.head_dim), key=kq
        )
    ).astype(mx.bfloat16)
    k = (
        5e-1
        * mx.random.normal((1, args.kv_heads, args.key_length, args.head_dim), key=kk)
    ).astype(mx.bfloat16)
    v = (
        5e-1
        * mx.random.normal((1, args.kv_heads, args.key_length, args.head_dim), key=kv)
    ).astype(mx.bfloat16)
    mx.eval(q, k, v)
    mx.savez(str(path), q=q, k=k, v=v, scale=mx.array(scale))
    mx.synchronize()


def gather_cases(fixture):
    x = fixture["x"]
    w1 = (fixture["w1q"], fixture["w1s"], fixture["w1b"])
    w2 = (fixture["w2q"], fixture["w2s"], fixture["w2b"])
    indices = fixture["indices"]
    n, routes = indices.shape
    order = mx.argsort(indices.flatten())
    inverse = mx.argsort(order)
    sorted_x = x.flatten(0, -3)[order // routes]
    sorted_indices = indices.flatten()[order]
    mx.eval(sorted_x, sorted_indices, inverse)

    def presorted_qmm_pair():
        value = mx.gather_qmm(
            sorted_x,
            *w1,
            transpose=True,
            rhs_indices=sorted_indices,
            sorted_indices=True,
        )
        return mx.gather_qmm(
            value,
            *w2,
            transpose=True,
            rhs_indices=sorted_indices,
            sorted_indices=True,
        )

    def sort_qmm_unsort_pipeline():
        current_order = mx.argsort(indices.flatten())
        current_inverse = mx.argsort(current_order)
        current_indices = indices.flatten()[current_order]
        value = x.flatten(0, -3)[current_order // routes]
        value = mx.gather_qmm(
            value,
            *w1,
            transpose=True,
            rhs_indices=current_indices,
            sorted_indices=True,
        )
        value = mx.gather_qmm(
            value,
            *w2,
            transpose=True,
            rhs_indices=current_indices,
            sorted_indices=True,
        )
        return value[current_inverse].reshape(n, routes, -1)

    return {
        "presorted_qmm_pair": presorted_qmm_pair,
        "sort_qmm_unsort_pipeline": sort_qmm_unsort_pipeline,
    }


def nvfp4_cases(fixture):
    packed = fixture["packed"] if "packed" in fixture else fixture["wq"]
    x, scales = fixture["x"], fixture["scales"]

    def qmv():
        return mx.quantized_matmul(x, packed, scales, transpose=True, mode="nvfp4")

    return {"nvfp4_qmv": qmv}


def qmm_cases(fixture, rows):
    packed = fixture["packed"]
    scales = fixture["scales"]
    biases = fixture["biases"]
    group_size = int(fixture["group_size"].item())
    bits = int(fixture["bits"].item())
    cases = {}
    for row_count in rows:
        if row_count < 1 or row_count > fixture["x"].shape[0]:
            raise ValueError(
                f"QMM rows must be between 1 and the fixture row count "
                f"({fixture['x'].shape[0]})"
            )
        x = fixture["x"][:row_count]

        def qmm(current=x):
            return mx.quantized_matmul(
                current,
                packed,
                scales,
                biases,
                transpose=True,
                group_size=group_size,
                bits=bits,
                mode="affine",
            )

        cases[f"qmm_t_m{row_count}"] = qmm
    return cases


def sdpa_cases(fixture):
    q, k, v = fixture["q"], fixture["k"], fixture["v"]
    scale = float(fixture["scale"].item())

    def causal_sdpa():
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")

    return {"causal_sdpa": causal_sdpa}


def run(args):
    fixture = mx.load(str(args.fixture))
    if args.case == "gather":
        cases = gather_cases(fixture)
    elif args.case == "nvfp4":
        cases = nvfp4_cases(fixture)
    elif args.case == "sdpa":
        cases = sdpa_cases(fixture)
    else:
        rows = (
            [int(value) for value in args.qmm_rows.split(",")]
            if args.qmm_rows
            else [args.m]
        )
        cases = qmm_cases(fixture, rows)
    report = {
        "schema": "mlx-m5-exact-path-benchmark-v1",
        "case": args.case,
        "label": args.label,
        "fixture_sha256": sha256(args.fixture),
        "provenance": provenance(),
        "environment": {
            key: os.environ[key]
            for key in SAFE_ENVIRONMENT_KEYS
            if key in os.environ
        },
        "benchmarks": {},
    }
    final = None
    for name, fn in cases.items():
        final, first_touch = timed(fn)
        convergence = [timed(fn)[1] for _ in range(args.warmup)]
        steady = [timed(fn)[1] for _ in range(args.iterations)]
        # NumPy does not expose MLX bfloat16 through the buffer protocol.
        # Hash its exactly representable float32 expansion instead.
        host_value = final.astype(mx.float32) if final.dtype == mx.bfloat16 else final
        final_host = np.asarray(host_value)
        report["benchmarks"][name] = {
            "first_touch_ms": first_touch,
            "convergence_ms": convergence,
            "steady_state": summarize(steady),
            "output_sha256": hashlib.sha256(final_host.tobytes()).hexdigest(),
            "output_shape": list(final.shape),
            "output_dtype": str(final.dtype),
        }
    if args.output_array is not None:
        args.output_array.parent.mkdir(parents=True, exist_ok=True)
        mx.save(str(args.output_array), final)
        mx.synchronize()
        report["output_sha256"] = sha256(args.output_array)
        report["output_shape"] = list(final.shape)
        report["output_dtype"] = str(final.dtype)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("gather", "nvfp4", "qmm", "sdpa"), required=True
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--make-fixture", action="store_true")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument(
        "--qmm-rows",
        help="comma-separated QMM row counts to benchmark in one process",
    )
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--n", type=int, default=17408)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--query-length", type=int, default=16)
    parser.add_argument("--key-length", type=int, default=16384)
    parser.add_argument("--query-heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--label", default="fixture-build")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--output-array", type=Path)
    args = parser.parse_args()

    if args.make_fixture:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        if args.case == "gather":
            make_gather_fixture(args.fixture)
        elif args.case == "nvfp4":
            make_nvfp4_fixture(args.fixture)
        elif args.case == "sdpa":
            make_sdpa_fixture(args.fixture, args)
        else:
            make_qmm_fixture(args.fixture, args)
        print(f"created {args.fixture} sha256={sha256(args.fixture)}")
        return
    if args.json_out is None:
        parser.error("--json-out is required unless --make-fixture is used")
    run(args)


if __name__ == "__main__":
    main()
