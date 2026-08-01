#!/usr/bin/env python3
"""Discriminate the hypotheses in docs/target-prefix-ar-divergence.md.

.. warning::

   **BROKEN — DO NOT RUN. Needs a rewrite, not a patch.** Confirmed by
   external review 2026-07-31:

   1. ``step_logits`` calls ``prepare_a3b_compiled_target_prefix(rt,
      cache=...)``. The real signature is ``(model, *, config,
      gdn_postconv_factory)`` and it returns a *factory*, not a route
      (``mtplx/a3b_compiled_target_prefix.py:156``). This raises ``TypeError``
      before any measurement — after loading the model.
   2. It compares two **prefill-shaped** forwards with no ``attention_phase``,
      so both arms take the stock MoE path. It cannot see M1, ``mx.compile``,
      or the shadow cache — i.e. none of the axes in dispute.
   3. Production staged-K1 verification is **rows==2 / M2**, not rows==1.
   4. ``--step length`` is a stub.
   5. ``--prefill-tokens`` is a cap, not a guarantee.

   The replacement design is the L0–L6 ladder in
   ``docs/dflash-gate-preregistration.md`` §4.1, with the amendments in §1b —
   in particular, the ladder as first written was **not** one-axis (L1 moves
   both the rewritten 40-layer forward and the postconv arithmetic), and a
   first-nonzero rung names the first *exposed* difference, not the mechanism.

   Kept in-tree rather than deleted so the failure is visible and the
   replacement has something to diff against.

Original description follows.

The compiled target-prefix control is only 13/24 byte-identical to
``generate_ar``, which blocks the DFlash release. That divergence predates
DFlash. This probe answers *why* in a couple of minutes rather than a session,
by running the two forward paths against the SAME cache state and reporting the
logit delta directly.

Steps, ordered so the earliest result discriminates the most:

  1  ``--step logits``  one forward through both paths on identical state.
                        H1 (accumulation-order numerics) predicts a small but
                        nonzero delta with argmax agreeing on most positions.
  2  ``--step impl-ab`` the two fused post-conv implementations against each
                        other. If they disagree, the fused route is the source
                        and the AR path is exonerated without touching it.
  3  ``--step length``  mismatch fraction vs generation length. H1 predicts it
                        grows; a state bug predicts a fixed onset step.

Run all three with ``--step all``. Writes one receipt per step.

Usage (Monday, machine quiet):

    .venv/bin/python benchmarks/target_prefix_divergence_probe.py \\
        --model ~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed \\
        --step all --output benchmarks/results/divergence-<date>.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--step", choices=["logits", "impl-ab", "length", "all"], default="all")
    p.add_argument("--prompt", default="Write a Python function that merges two sorted lists.")
    p.add_argument("--prefill-tokens", type=int, default=256)
    p.add_argument("--probe-steps", type=int, default=8, help="decode steps compared in --step logits")
    p.add_argument("--lengths", default="64,128,192", help="--step length sweep")
    p.add_argument("--prompts", type=Path, default=Path("mtplx/benchmarks/prompts/calibration_coding.jsonl"))
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path)
    p.add_argument("--allow-busy", action="store_true", help="skip the quiet-machine guard")
    return p.parse_args()


def _machine_is_quiet() -> tuple[bool, str]:
    """Refuse to measure under contention — the project has been burned by this."""
    try:
        out = subprocess.run(["ps", "-Ao", "pcpu,comm"], capture_output=True, text=True, timeout=10).stdout
    except Exception as exc:  # pragma: no cover
        return False, f"could not inspect processes: {exc}"
    busy = []
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            cpu = float(parts[0])
        except ValueError:
            continue
        name = parts[1]
        if cpu < 40.0:
            continue
        if any(tag in name for tag in ("python", "mtplx", "train", "wisp")):
            busy.append(f"{name} ({cpu:.0f}%)")
    if busy:
        return False, "busy: " + ", ".join(busy[:4])
    return True, "quiet"


def _env_snapshot() -> dict:
    """Every execution-affecting switch, recorded. Intent is not evidence."""
    keys = [k for k in os.environ if k.startswith("MTPLX_")]
    return {
        "mtplx_env": {k: os.environ[k] for k in sorted(keys)},
        "platform": platform.platform(),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        ),
    }


def _encode(tokenizer, prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(t) for t in encoded]


def step_logits(rt, tokenizer, args) -> dict:
    """H1's discriminator: same state, both forwards, report the delta.

    Deliberately does NOT go through the accept loop — the point is to isolate
    the forward implementations from every other moving part.
    """
    import mlx.core as mx

    from mtplx.a3b_compiled_target_prefix import prepare_a3b_compiled_target_prefix

    ids = _encode(tokenizer, args.prompt)[: args.prefill_tokens]
    rows = []

    for i in range(args.probe_steps):
        cache_ar = rt.make_cache()
        out_ar = rt.forward_ar(mx.array([ids]), cache=cache_ar, return_hidden=False)
        logits_ar = out_ar[0] if isinstance(out_ar, tuple) else out_ar
        mx.eval(logits_ar)

        cache_pc = rt.make_cache()
        route = prepare_a3b_compiled_target_prefix(rt, cache=cache_pc)
        out_pc = rt._forward_ar_capture_a3b_postconv(
            mx.array([ids]),
            cache=cache_pc,
            hidden_variant=None,
            postconv_implementations=route.postconv_implementations,
        )
        logits_pc = out_pc[0]
        mx.eval(logits_pc)

        a = logits_ar[0, -1].astype(mx.float32)
        b = logits_pc[0, -1].astype(mx.float32)
        delta = float(mx.max(mx.abs(a - b)).item())
        top_a, top_b = int(mx.argmax(a).item()), int(mx.argmax(b).item())
        sorted_a = mx.sort(a)
        margin = float((sorted_a[-1] - sorted_a[-2]).item())
        rows.append({
            "step": i,
            "max_abs_logit_delta": delta,
            "argmax_ar": top_a,
            "argmax_postconv": top_b,
            "argmax_agrees": top_a == top_b,
            "ar_top1_margin": margin,
            "delta_exceeds_margin": delta > margin,
        })
        ids.append(top_a)

    agree = sum(1 for r in rows if r["argmax_agrees"])
    deltas = [r["max_abs_logit_delta"] for r in rows]
    at_risk = sum(1 for r in rows if r["delta_exceeds_margin"])
    return {
        "rows": rows,
        "argmax_agreement": f"{agree}/{len(rows)}",
        "max_delta": max(deltas) if deltas else None,
        "median_delta": sorted(deltas)[len(deltas) // 2] if deltas else None,
        "steps_where_delta_exceeds_top1_margin": at_risk,
        "reading": (
            "H1 (numerics) supported if delta is small-but-nonzero and argmax "
            "mostly agrees, with the disagreements concentrated where "
            "delta_exceeds_margin is true. A delta of exactly 0.0 refutes H1 "
            "and points at H2 (shadow-cache state) instead."
        ),
    }


def step_impl_ab(args) -> dict:
    """Run the two fused post-conv implementations against each other.

    Exonerates or implicates the fused route without involving the AR path.
    Spawns subprocesses because the implementation is selected at import time.
    """
    results = {}
    for impl in ("inline_g", "headquarter"):
        env = dict(os.environ, MTPLX_A3B_GDN_POSTCONV_IMPL=impl)
        cmd = [
            os.sys.executable, __file__,
            "--model", str(args.model), "--step", "logits",
            "--prompt", args.prompt, "--probe-steps", str(args.probe_steps),
            "--allow-busy",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        try:
            payload = json.loads(proc.stdout[proc.stdout.index("{"):])
            results[impl] = payload.get("logits", {})
        except Exception:
            results[impl] = {"error": proc.stderr[-400:]}
    return {
        "per_impl": results,
        "reading": (
            "If the two implementations disagree with each other, the fused "
            "post-conv route is the divergence source and the AR path is not "
            "implicated."
        ),
    }


def main() -> int:
    raise SystemExit(
        "This probe is BROKEN (see module docstring): step_logits calls "
        "prepare_a3b_compiled_target_prefix with the wrong signature and "
        "compares two stock-MoE prefill forwards, so it cannot measure the "
        "axes in dispute. Use the L0-L6 ladder from "
        "docs/dflash-gate-preregistration.md instead. Remove this guard only "
        "together with the rewrite."
    )
    args = _args()
    quiet, why = _machine_is_quiet()
    if not quiet and not args.allow_busy:
        raise SystemExit(
            f"machine is not quiet ({why}); timings and near-tie flips are both "
            "contention-sensitive. Re-run when idle or pass --allow-busy."
        )

    from mtplx.runtime import load

    t0 = time.time()
    rt, tokenizer = load(str(args.model))
    payload: dict = {
        "model": str(args.model),
        "quiet_check": why,
        "load_seconds": round(time.time() - t0, 1),
        **_env_snapshot(),
    }

    if args.step in ("logits", "all"):
        payload["logits"] = step_logits(rt, tokenizer, args)
    if args.step in ("impl-ab", "all"):
        payload["impl_ab"] = step_impl_ab(args)
    if args.step in ("length", "all"):
        payload["length"] = {
            "status": "not_implemented",
            "note": (
                "Only run if steps 1-2 come back clean. Compares byte equality "
                "of generate_ar vs the compiled control across --lengths; H1 "
                "predicts the mismatch fraction grows with length, a state bug "
                "predicts a fixed onset step."
            ),
        }

    text = json.dumps(payload, indent=1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
