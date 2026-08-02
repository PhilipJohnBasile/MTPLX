"""One-load, guarded-GPU parity/compile gate for the DeepSeek-V4 MoE tail.

This is deliberately an operator safety receipt, not a throughput benchmark.
It loads the real 2-bit checkpoint once, captures the stock score-layer MoE
tail after a real 328-token coding prefill, then compares the exact stock tail
with the fused BF16 tail over the authentic [4, 6, 4096] verify-shaped tensors.
Every diagnostic sample is explicitly evaluated and synchronized; it never
queues hundreds of independent outputs and mistakes host enqueue time for GPU
work.  The subsequent full 328-token / 256-generated C0->candidate->C1 run is
the only performance decision.

Run only through ``bench/laguna/run_guarded.py``.  It records whether the MLX
runtime came from the profiler tree; ``--require-profiler`` makes that mandatory
for a profiling capture, while the same exact-parity gate also runs on the
official 0.31 serving build before the full TPS bracket.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import mlx.core as mx


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
    ap.add_argument("--prompt-tokens", type=int, default=328)
    ap.add_argument("--layer", type=int, default=3,
                    help="score-routed body layer captured (hash layers stay stock)")
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--require-profiler", action="store_true")
    ap.add_argument("--out", required=True, help="JSON receipt path")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("no 2-bit DeepSeek-V4 model found; pass --model")
    if args.cycles < 3:
        raise SystemExit("--cycles must be >= 3 for a useful diagnostic median")
    if not mx.metal.is_available():
        raise SystemExit("this gate requires Metal")
    instrumented_mlx = "mlx-profiler" in str(getattr(mx, "__file__", ""))
    if args.require_profiler and not instrumented_mlx:
        raise SystemExit("--require-profiler needs mlx-profiler first on PYTHONPATH")
    mx.set_default_device(mx.gpu)

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from mlx_lm.utils import load_config
    from mtplx.models import deepseek_v4 as D
    from mtplx.runtime import _load_base_model

    model_path = Path(args.model).expanduser().resolve()
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    config = load_config(model_path)
    t0 = time.perf_counter()
    model, tokenizer = _load_base_model(model_path, config)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - t0
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) != args.prompt_tokens:
        raise SystemExit(
            f"prompt has {len(prompt_ids)} tokens, expected {args.prompt_tokens}; "
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
    fused = candidate(routed, weights, shared)
    mx.eval(stock, fused)
    exact = bool(mx.array_equal(stock, fused))
    max_abs = float(mx.max(mx.abs(stock.astype(mx.float32) - fused.astype(mx.float32))).item())
    if not exact:
        raise SystemExit(f"FAIL exact parity: max_abs={max_abs:g}")

    stock_seconds = _timed(lambda: D._stock_moe_tail_combine(routed, weights, shared), cycles=args.cycles)
    fused_seconds = _timed(lambda: candidate(routed, weights, shared), cycles=args.cycles)
    receipt = {
        "harness": "scripts/deepseek_v4_moe_tail_gate.py",
        "purpose": "one-load real-capture exact-parity and compile safety gate; TPS verdict is external",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {"platform": platform.platform(), "mlx": mx.__version__,
                 "mlx_file": str(mx.__file__), "instrumented_mlx": instrumented_mlx},
        "model_path": str(model_path),
        "load_seconds": load_seconds,
        "prompt_tokens": len(prompt_ids),
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
