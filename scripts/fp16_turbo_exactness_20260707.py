#!/usr/bin/env python3
"""FP16 27B turbo-path exactness gate (2.0.1 Phase 2.2, 2026-07-07).

Two components, both on the real Speed-FP16 artifact (INT4/g64 weights,
fp16 activations — the M1/M2 routing target):

1. Hot-shape kernel logit-diff on REAL weights: for each hot trunk
   projection class (q/gate/down/lm_head), run the exact verify kernels the
   turbo patch routes (m4 -> env impl, m5/6 -> m6 ksplit, m7..16 -> NAX
   tile when available) against stock ``mx.quantized_matmul`` on fp16
   activations. Gate: dmax <= 0.02 * max(1, max|ref|) — the q8 lane's
   accepted ULP-class band, scale-aware because lm_head logits are O(30).

2. Greedy 50-token continuation A/B in one process: turbo arm (qlinear
   patch installed + MTPLX_COMPILED_VERIFY=1) vs eager arm (patch
   uninstalled + MTPLX_COMPILED_VERIFY=0), same prompt/seed/route
   (D3 capture_commit). Reports token-level agreement and first
   divergence, plus decode tok/s per arm for context.

House rules honored: same-route logit diffs (never text equality as the
primary gate), launch from a neutral cwd, max-fans verified by the caller
before the model load.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = Path.home() / ".mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed-FP16"

DEFAULT_PROMPT = (
    "Create a single-file HTML5 Canvas flappy bird game. All visuals drawn "
    "procedurally. Animated bird with distinct up-stroke and down-stroke wing "
    "shapes, body tilt, squash-and-stretch on flap, feather particles from "
    "wing tips. Pipes with gradient shading, cap/lip, cylindrical highlight. "
    "Start screen, death screen with best score in localStorage. Delta-time "
    "physics. Make it gorgeous."
)


@contextmanager
def patched_env(updates: dict[str, str | None]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _collect_hot_projections(model: Any) -> list[tuple[str, Any]]:
    """One representative QuantizedLinear per hot projection class."""
    import mlx.nn as nn

    text_model = getattr(model, "language_model", model)
    inner = getattr(text_model, "model", text_model)
    picks: dict[str, Any] = {}
    for layer in getattr(inner, "layers", []) or []:
        for label, path in (
            ("attn_q_proj", ("self_attn", "q_proj")),
            ("attn_o_proj", ("self_attn", "o_proj")),
            ("mlp_gate_proj", ("mlp", "gate_proj")),
            ("mlp_down_proj", ("mlp", "down_proj")),
            ("gdn_in_proj_qkvz", ("linear_attn", "in_proj_qkvz")),
            ("gdn_out_proj", ("linear_attn", "out_proj")),
        ):
            if label in picks:
                continue
            node = layer
            for name in path:
                node = getattr(node, name, None)
                if node is None:
                    break
            if isinstance(node, nn.QuantizedLinear):
                picks[label] = node
        if len(picks) >= 6:
            break
    lm_head = getattr(text_model, "lm_head", None)
    if isinstance(lm_head, nn.QuantizedLinear):
        picks["lm_head"] = lm_head
    return sorted(picks.items())


def run_kernel_dmax(model: Any, *, rel_band: float) -> list[dict[str, Any]]:
    import mlx.core as mx

    from mtplx import nax_verify

    rows: list[dict[str, Any]] = []
    for label, module in _collect_hot_projections(model):
        bits = int(getattr(module, "bits", 0) or 0)
        group_size = int(getattr(module, "group_size", 0) or 0)
        w_q = module["weight"]
        scales = module["scales"]
        biases = module["biases"]
        n, k_packed = int(w_q.shape[0]), int(w_q.shape[1])
        k = k_packed * (32 // bits)
        dtype = scales.dtype
        mx.random.seed(23)
        cases: list[tuple[str, int, Any]] = []
        if bits == 4:
            if nax_verify.m4_ksplit_eligible(4, k, n, bits, group_size, dtype):
                cases.append(
                    (
                        "m4",
                        4,
                        lambda x, w=w_q, s=scales, b=biases, g=group_size: nax_verify.nax_qmm_m4(
                            x, w, s, b, group_size=g
                        ),
                    )
                )
            for m in (5, 6):
                if nax_verify.m6_ksplit_eligible(m, k, n, bits, group_size, dtype):
                    cases.append(
                        (
                            f"m6_pad{m}",
                            m,
                            lambda x, w=w_q, s=scales, b=biases, g=group_size: nax_verify.nax_qmm_m6(
                                x, w, s, b, group_size=g
                            ),
                        )
                    )
            for m in (8, 16):
                if nax_verify.m16_nax_eligible(m, k, n, bits, group_size, dtype):
                    cases.append(
                        (
                            f"m16_pad{m}",
                            m,
                            lambda x, w=w_q, s=scales, b=biases, g=group_size: nax_verify.nax_qmm_m16(
                                x, w, s, b, group_size=g
                            ),
                        )
                    )
        for case_label, m, fn in cases:
            x = (mx.random.normal((m, k), dtype=mx.float32) * 0.5).astype(dtype)
            y = fn(x)
            ref = mx.quantized_matmul(
                x,
                w_q,
                scales=scales,
                biases=biases,
                transpose=True,
                group_size=group_size,
                bits=bits,
            )
            diff = mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))
            dmax = float(diff.max())
            ref_max = float(mx.abs(ref.astype(mx.float32)).max())
            threshold = rel_band * max(1.0, ref_max)
            rows.append(
                {
                    "projection": label,
                    "case": case_label,
                    "M": m,
                    "K": k,
                    "N": n,
                    "bits": bits,
                    "dtype": str(dtype),
                    "dmax": dmax,
                    "ref_max": ref_max,
                    "threshold": threshold,
                    "passed": bool(dmax <= threshold),
                }
            )
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
            mx.clear_cache()
    return rows


def run_greedy_ab(rt: Any, *, prompt: str, max_tokens: int, depth: int) -> dict[str, Any]:
    from mtplx import nax_verify
    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    prompt_ids = list(rt.tokenizer.encode(prompt))
    greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)

    def _arm(name: str, *, compiled: str, patch: bool) -> dict[str, Any]:
        if patch:
            nax_verify.install_nax_qlinear_patch()
        else:
            nax_verify.uninstall_nax_qlinear_patch()
        started = time.perf_counter()
        with patched_env({"MTPLX_COMPILED_VERIFY": compiled}):
            out = generate_mtpk(
                rt,
                prompt_ids,
                max_tokens=max_tokens,
                sampler=greedy,
                speculative_depth=depth,
                verify_strategy="capture_commit",
                stop_token_ids=set(),
                seed=0,
            )
        elapsed = time.perf_counter() - started
        tokens = list(out.tokens)
        return {
            "arm": name,
            "tokens": tokens,
            "generated": int(out.stats.generated_tokens),
            "elapsed_s": elapsed,
            "decode_tok_s": getattr(out.stats, "decode_tok_s", None),
        }

    turbo = _arm("turbo", compiled="1", patch=True)
    eager = _arm("eager_stock", compiled="0", patch=False)
    # Restore the launch state (patch installed) for any later use.
    nax_verify.install_nax_qlinear_patch()

    t_tokens, e_tokens = turbo["tokens"], eager["tokens"]
    first_divergence = None
    for i, (a, b) in enumerate(zip(t_tokens, e_tokens)):
        if a != b:
            first_divergence = i
            break
    agree = first_divergence is None and len(t_tokens) == len(e_tokens)
    return {
        "prompt_tokens": len(prompt_ids),
        "depth": depth,
        "max_tokens": max_tokens,
        "turbo": {k: v for k, v in turbo.items() if k != "tokens"},
        "eager": {k: v for k, v in eager.items() if k != "tokens"},
        "greedy_match": agree,
        "first_divergence_index": first_divergence,
        "matched_prefix": first_divergence if first_divergence is not None else len(t_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--rel-band", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from mtplx.runtime import load

    rt = load(args.model, mtp=True)

    from mtplx.kernel_selfcheck import report_for_health

    selfcheck = report_for_health()
    print(json.dumps({"kernel_selfcheck": selfcheck}, sort_keys=True), flush=True)

    kernel_rows = run_kernel_dmax(rt.model, rel_band=float(args.rel_band))
    greedy = run_greedy_ab(
        rt, prompt=args.prompt, max_tokens=int(args.max_tokens), depth=int(args.depth)
    )
    print(json.dumps({"greedy_ab": greedy}, sort_keys=True), flush=True)

    kernel_pass = all(row["passed"] for row in kernel_rows) and bool(kernel_rows)
    result = {
        "run_id": f"fp16-turbo-exactness-{time.strftime('%Y%m%d-%H%M%S')}",
        "model": str(args.model),
        "m4_impl": os.environ.get("MTPLX_NAX_M4_IMPL", "legacy"),
        "kernel_selfcheck": selfcheck,
        "kernel_rows": kernel_rows,
        "kernel_pass": kernel_pass,
        "greedy_ab": greedy,
        "passed": bool(kernel_pass and greedy["greedy_match"]),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "kernel_pass": kernel_pass,
                "greedy_match": greedy["greedy_match"],
                "output": str(args.output) if args.output else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
