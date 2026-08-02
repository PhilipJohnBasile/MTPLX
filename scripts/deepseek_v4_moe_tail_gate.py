"""One-load, guarded-GPU parity/compile gate for the DeepSeek-V4 MoE tail.

This is deliberately an operator safety receipt, not a throughput benchmark.
It loads the real 2-bit checkpoint once, captures the stock score-layer MoE
tail after a real 328-token coding prefill, then compares the exact stock tail
with the fused BF16 tail over the authentic [4, 6, 4096] verify-shaped tensors.
Every diagnostic sample is explicitly evaluated and synchronized; it never
queues hundreds of independent outputs and mistakes host enqueue time for GPU
work.  The subsequent full 328-token / 256-generated C0->candidate->C1 run is
the only performance decision.

Run only through ``bench/laguna/run_guarded.py``.  The gate accepts exactly the
official MLX 0.31.2 serving runtime and the immutable official 328-token prompt
identity below.  A profiler/dev MLX build or altered/copied prompt is rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import mlx.core as mx


_REQUIRED_MLX_VERSION = "0.31.2"
_REQUIRED_MLX_CORE_SHA256 = (
    "d7bd29fc20b4a08318d21161c3dfb340889cc9454c5e554ad749eb0127cfa2d6"
)
_REQUIRED_MLX_LIB_SHA256 = (
    "2ee6fbd32ff22e22e1301ebe3c3bece95584104ff9cbc900513d41a095211bbd"
)
_REQUIRED_PROMPT_PATH = Path(
    "/Users/davidtai/projects/OpenSourceWTF/bench/deepseek-v4/"
    "smoke-2bitdq-20260731-prompt2.txt"
)
_REQUIRED_PROMPT_SHA256 = (
    "ee94397faa812c91d5f1a0ee17c5bb6ca6032883653591dd33d4cfddb737ac33"
)
_REQUIRED_PROMPT_TOKENS = 328


def _default_model() -> str | None:
    root = Path.home() / ".cache/huggingface/hub"
    hits = sorted(root.glob("models--mlx-community--DeepSeek-V4-Flash-2bit-DQ/snapshots/*"))
    return str(hits[0]) if hits else None


def _median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2]


def _timed(fn, *, cycles: int) -> list[float]:
    for _ in range(2):
        out = fn()
        mx.eval(out)
        mx.synchronize()
    values = []
    for _ in range(cycles):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        values.append(time.perf_counter() - t0)
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_default_model())
    ap.add_argument("--prompt-file", required=True,
                    help="the exact official coding prompt used by the TPS bracket")
    ap.add_argument("--layer", type=int, default=3,
                    help="score-routed body layer captured (hash layers stay stock)")
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--out", required=True, help="JSON receipt path")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("no 2-bit DeepSeek-V4 model found; pass --model")
    if args.cycles < 3:
        raise SystemExit("--cycles must be >= 3 for a useful diagnostic median")
    if not mx.metal.is_available():
        raise SystemExit("this gate requires Metal")
    if mx.__version__ != _REQUIRED_MLX_VERSION:
        raise SystemExit(
            f"requires official MLX {_REQUIRED_MLX_VERSION}, got {mx.__version__} "
            f"from {getattr(mx, '__file__', None)}"
        )
    mlx_core_path = Path(mx.__file__).resolve()
    mlx_lib_path = mlx_core_path.parent / "lib" / "libmlx.dylib"
    mlx_core_sha256 = hashlib.sha256(mlx_core_path.read_bytes()).hexdigest()
    mlx_lib_sha256 = hashlib.sha256(mlx_lib_path.read_bytes()).hexdigest()
    if (
        mlx_core_sha256 != _REQUIRED_MLX_CORE_SHA256
        or mlx_lib_sha256 != _REQUIRED_MLX_LIB_SHA256
    ):
        raise SystemExit(
            "MLX 0.31.2 binary identity mismatch: "
            f"core={mlx_core_sha256} lib={mlx_lib_sha256}"
        )
    mx.set_default_device(mx.gpu)

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from mlx_lm.utils import load_config
    from mtplx.attention_context import attention_phase
    from mtplx.models import deepseek_v4 as D
    from mtplx.runtime import _load_base_model

    model_path = Path(args.model).expanduser().resolve()
    prompt_path = Path(args.prompt_file).expanduser().resolve()
    if prompt_path != _REQUIRED_PROMPT_PATH:
        raise SystemExit(
            f"requires official prompt path {_REQUIRED_PROMPT_PATH}, got {prompt_path}"
        )
    prompt_bytes = prompt_path.read_bytes()
    prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    if prompt_sha256 != _REQUIRED_PROMPT_SHA256:
        raise SystemExit(
            f"official prompt SHA mismatch: expected {_REQUIRED_PROMPT_SHA256}, "
            f"got {prompt_sha256}"
        )
    prompt = prompt_bytes.decode("utf-8")
    config = load_config(model_path)
    t0 = time.perf_counter()
    model, tokenizer = _load_base_model(model_path, config)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - t0
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) != _REQUIRED_PROMPT_TOKENS:
        raise SystemExit(
            f"prompt has {len(prompt_ids)} tokens, expected {_REQUIRED_PROMPT_TOKENS}; "
            "pass the exact official 328-token coding prompt"
        )
    if not (0 <= args.layer < len(model.layers)):
        raise SystemExit(f"--layer={args.layer} outside body [0,{len(model.layers)})")
    if args.layer < int(model.args.num_hash_layers):
        raise SystemExit("capture a score layer; hash layers are intentionally stock")

    # A real 328-token prefill establishes authentic cache/routing state.  Then
    # four deterministic next ids exercise the K3 verifier's M=4 body shape.
    cache = model.make_cache()
    logits = model(mx.array(prompt_ids)[None], cache=cache)
    next_id = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(next_id)
    verify_ids = mx.broadcast_to(next_id[:, None], (1, 4))

    target = model.layers[args.layer].ffn
    captured: dict[str, mx.array] = {}
    original = target._tail_combine

    def capture(routed, weights, shared):
        captured["routed"] = routed
        captured["weights"] = weights
        captured["shared"] = shared
        return original(routed, weights, shared)

    target._tail_combine = capture
    try:
        logits = model(verify_ids, cache=cache)
        mx.eval(logits)
    finally:
        target._tail_combine = original
    if set(captured) != {"routed", "weights", "shared"}:
        raise SystemExit("score-layer tail capture did not engage")
    routed, weights, shared = (captured[k] for k in ("routed", "weights", "shared"))
    mx.eval(routed, weights, shared)
    expected = (4, 6, 4096)
    if tuple(routed.shape) != expected or tuple(weights.shape) != (4, 6):
        raise SystemExit(
            f"capture geometry {tuple(routed.shape)}/{tuple(weights.shape)} != "
            f"{expected}/(4, 6)"
        )
    if routed.dtype != mx.bfloat16 or shared.dtype != mx.bfloat16:
        raise SystemExit(
            f"capture dtype must be BF16/BF16, got {routed.dtype}/{shared.dtype}"
        )

    candidate = D._install_moe_tail_combine(model.args)
    stock = D._stock_moe_tail_combine(routed, weights, shared)
    with attention_phase("decode_verify"):
        fused = candidate(routed, weights, shared)
    mx.eval(stock, fused)
    exact = bool(mx.array_equal(stock, fused))
    max_abs = float(mx.max(mx.abs(stock.astype(mx.float32) - fused.astype(mx.float32))).item())
    if not exact:
        raise SystemExit(f"FAIL exact parity: max_abs={max_abs:g}")

    stock_seconds = _timed(lambda: D._stock_moe_tail_combine(routed, weights, shared), cycles=args.cycles)
    def fused_call():
        with attention_phase("decode_verify"):
            return candidate(routed, weights, shared)

    fused_seconds = _timed(fused_call, cycles=args.cycles)
    receipt = {
        "harness": "scripts/deepseek_v4_moe_tail_gate.py",
        "purpose": "one-load real-capture exact-parity and compile safety gate; TPS verdict is external",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "identity": {
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "required_mlx_version": _REQUIRED_MLX_VERSION,
            "mlx_core_sha256": mlx_core_sha256,
            "mlx_lib_sha256": mlx_lib_sha256,
            "prompt_path": str(prompt_path),
            "prompt_sha256": prompt_sha256,
            "prompt_tokens": len(prompt_ids),
            "model_snapshot": str(model_path),
        },
        "host": {"platform": platform.platform(),
                 "mlx_required": _REQUIRED_MLX_VERSION,
                 "mlx": mx.__version__, "mlx_file": str(mlx_core_path),
                 "mlx_lib_file": str(mlx_lib_path)},
        "model_path": str(model_path),
        "load_seconds": load_seconds,
        "prompt": {"path": str(prompt_path), "sha256": prompt_sha256,
                   "tokens": len(prompt_ids)},
        "capture": {"body_layer": args.layer, "routed_shape": list(routed.shape),
                    "weights_shape": list(weights.shape), "routed_dtype": str(routed.dtype),
                    "shared_dtype": str(shared.dtype)},
        "exact_parity": exact,
        "max_abs": max_abs,
        "diagnostic_seconds": {"stock": stock_seconds, "fused": fused_seconds,
                               "stock_median": _median(stock_seconds),
                               "fused_median": _median(fused_seconds)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
