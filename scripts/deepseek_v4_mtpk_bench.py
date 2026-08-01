"""Four-arm MTP speculative benchmark for the deepseek_v4 backend.

ONE model load, FOUR arms in-window: an AR control plus ``speculative_depth``
1/2/3.  In-window pairing is the box rule -- cross-window thermal drift on this
machine is 15-20%, which is larger than the effect being measured, so an arm
compared against a number from another window measures the fan, not the depth.

Both arms run through the real ``mtplx.generation`` machine over a real
:class:`~mtplx.runtime.MTPLXRuntime` (``runtime.load(..., mtp=True)`` ->
``inject_deepseek_v4_mtp_support``), not a hand-rolled loop: the point is to
measure the lane that would actually serve, including prefill, draft chain,
batched verify, accept/reject and the rollback repair.

The gate the arms carry, beyond speed: greedy speculative decode is a pure
latency optimisation, so every K arm's committed token sequence must be
*identical* to the AR arm's.  A divergence means the rollback is lossy and the
tok/s number is meaningless -- so it is reported as a failure, not a footnote.
This is the shop's standard spec==AR gate (tests/test_deepseek_v4_spec.py) run
at real dims on real weights instead of a shrunk seeded model.

``--tiny`` builds the shrunk seeded model the spec gates use and runs the whole
four-arm shape on CPU in seconds.  That is a harness self-test -- it validates
the arm loop, the stats extraction and the receipt writing without spending a
~90 GiB load -- not a performance measurement.

MUST run inside the box's serialized MLX window (bench/laguna/run_guarded.py):
the 2-bit checkpoint plus its MTP bank is ~93 GiB and does not fit beside the
served model.

Usage:
  python scripts/deepseek_v4_mtpk_bench.py \
      --model ~/models/DeepSeek-V4-Flash-2bit-DQ-mtp \
      --prompt-file bench/deepseek-v4/smoke-2bitdq-20260731-prompt2.txt \
      --max-tokens 256 --out bench/deepseek-v4/mtpk-2bitdq-YYYYMMDD
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx


# Peak memory is read per arm, so the ceiling is a per-arm claim.  Kept as a
# guard rather than an assertion: the wired knob is 112 GiB and is never raised
# (that is a box rule), so an arm that would cross it should stop the window
# rather than let the allocator fall off the wired cliff -- over-limit collapses
# throughput ~4x and the number would be garbage anyway.
_PEAK_ABORT_GIB = 108.0


def _gib(n: int) -> float:
    return n / (1024**3)


def _peak_bytes() -> int:
    fn = getattr(mx, "get_peak_memory", None)
    if callable(fn):
        return int(fn())
    fn = getattr(getattr(mx, "metal", None), "get_peak_memory", None)
    return int(fn()) if callable(fn) else -1


def _active_bytes() -> int:
    fn = getattr(mx, "get_active_memory", None)
    if callable(fn):
        return int(fn())
    fn = getattr(getattr(mx, "metal", None), "get_active_memory", None)
    return int(fn()) if callable(fn) else -1


def _reset_peak() -> None:
    fn = getattr(mx, "reset_peak_memory", None)
    if callable(fn):
        fn()


def _clear_cache() -> None:
    fn = getattr(mx, "clear_cache", None)
    if callable(fn):
        fn()


# The stats surface is huge (every counter every backend ever needed).  Pull the
# ones this measurement is actually about, so the receipt stays readable; the
# full dict is kept alongside under "stats_full".
_STAT_KEYS = (
    "mode",
    "generated_tokens",
    "elapsed_s",
    "tok_s",
    "decode_elapsed_s",
    "decode_tok_s",
    "end_to_end_tok_s",
    "runtime_mtp_enabled",
    "draft_head_installed",
    "speculative_depth",
    "requested_speculative_depth",
    "accepted_by_depth",
    "drafted_by_depth",
    "accepted_drafts",
    "rejected_drafts",
    "drafted_tokens",
    "skipped_drafts",
    "bonus_tokens",
    "correction_tokens",
    "verify_calls",
    "mtp_forward_calls",
    "make_mtp_cache_calls",
    "update_mtp_cache_calls",
    "mtp_history_append_calls",
    "forward_ar_hidden_calls",
    "forward_ar_plain_calls",
    "draft_time_s",
    "verify_time_s",
    "verify_forward_time_s",
    "verify_eval_time_s",
    "target_forward_time_s",
    "snapshot_time_s",
    "prompt_eval_time_s",
    "prompt_tps",
    "mtp_history_policy",
    "reject_path_counts",
    "peak_memory_bytes",
)


def _accept_rates(stats: dict) -> list[dict]:
    """Per-depth accept rate.  ``drafted_by_depth[i]`` is how often depth ``i``
    was even proposed (it is not proposed when a shallower depth was rejected),
    so the rate is conditional on reaching that depth -- which is the number that
    predicts the speedup, unlike accepted/total-drafted."""
    accepted = list(stats.get("accepted_by_depth") or [])
    drafted = list(stats.get("drafted_by_depth") or [])
    rows = []
    for i in range(max(len(accepted), len(drafted))):
        a = int(accepted[i]) if i < len(accepted) else 0
        d = int(drafted[i]) if i < len(drafted) else 0
        rows.append(
            {
                "depth": i + 1,
                "drafted": d,
                "accepted": a,
                "accept_rate": (a / d) if d else None,
            }
        )
    return rows


def _mean_accepted_per_cycle(stats: dict) -> float | None:
    """Committed tokens per verify call: the cycle-anatomy number the projection
    leans on.  1.0 means speculation bought nothing."""
    calls = int(stats.get("verify_calls") or 0)
    if not calls:
        return None
    return float(stats.get("generated_tokens") or 0) / calls


def _run_arm(
    *,
    rt,
    label: str,
    depth: int | None,
    prompt_ids: list[int],
    max_tokens: int,
    verify_strategy: str,
    verify_core: str,
    mtp_history_policy: str,
    baseline_tokens: list[int] | None,
) -> dict:
    from mtplx.generation import generate_ar, generate_mtpk
    from mtplx.sampling import SamplerConfig

    print(f"\n{'#' * 72}\n# ARM {label}\n{'#' * 72}")
    sys.stdout.flush()

    _clear_cache()
    _reset_peak()
    sampler = SamplerConfig(temperature=0.0)
    started = time.perf_counter()
    error = None
    out = None
    try:
        if depth is None:
            out = generate_ar(
                rt,
                prompt_ids,
                max_tokens=max_tokens,
                sampler=sampler,
                # Forced full length in every arm: the arms are compared token
                # for token, so an early stop in one of them would compare
                # different amounts of work as well as different sequences.
                stop_token_ids=set(),
            )
        else:
            out = generate_mtpk(
                rt,
                prompt_ids,
                max_tokens=max_tokens,
                sampler=sampler,
                speculative_depth=depth,
                mtp_history_policy=mtp_history_policy,
                verify_strategy=verify_strategy,
                verify_core=verify_core,
                stop_token_ids=set(),
            )
    except Exception:
        error = traceback.format_exc()
        print(error)
        sys.stdout.flush()
    wall = time.perf_counter() - started
    peak = _peak_bytes()

    arm: dict = {
        "label": label,
        "speculative_depth": depth,
        "verify_strategy": None if depth is None else verify_strategy,
        "verify_core": None if depth is None else verify_core,
        "mtp_history_policy": None if depth is None else mtp_history_policy,
        "wall_seconds": wall,
        "peak_bytes": peak,
        "peak_gib": _gib(peak),
        "active_end_gib": _gib(_active_bytes()),
        "error": error,
    }
    if out is None:
        return arm

    stats = out.stats.to_dict()
    decode_s = float(stats.get("decode_elapsed_s") or 0.0)
    n_new = int(stats.get("generated_tokens") or len(out.tokens))
    arm.update(
        {
            "tokens": list(out.tokens),
            "text": out.text,
            "finish_reason": out.finish_reason,
            "generated_tokens": n_new,
            "decode_seconds": decode_s,
            "decode_tokens_per_second": (n_new / decode_s) if decode_s else 0.0,
            "ms_per_token": (1000.0 * decode_s / n_new) if n_new else 0.0,
            "prefill_seconds": float(stats.get("prompt_eval_time_s") or 0.0),
            "prefill_tokens_per_second": float(stats.get("prompt_tps") or 0.0),
            "accept_rates": _accept_rates(stats),
            # Only meaningful on a speculative arm: AR's "verify calls" are just
            # its forwards, so the ratio there is 1 by construction and would
            # read as if the control were speculating.
            "mean_accepted_per_verify_call": (
                None if depth is None else _mean_accepted_per_cycle(stats)
            ),
            "stats": {k: stats.get(k) for k in _STAT_KEYS},
            "stats_full": stats,
        }
    )
    if baseline_tokens is not None:
        same = list(out.tokens) == list(baseline_tokens)
        first_div = None
        if not same:
            for i, (a, b) in enumerate(zip(out.tokens, baseline_tokens)):
                if a != b:
                    first_div = i
                    break
            if first_div is None:
                first_div = min(len(out.tokens), len(baseline_tokens))
        arm["spec_equals_ar"] = {
            "pass": same,
            "baseline_tokens": len(baseline_tokens),
            "arm_tokens": len(out.tokens),
            "first_divergence_index": first_div,
            "baseline_at_divergence": (
                None if first_div is None or first_div >= len(baseline_tokens)
                else baseline_tokens[first_div]
            ),
            "arm_at_divergence": (
                None if first_div is None or first_div >= len(out.tokens)
                else out.tokens[first_div]
            ),
        }

    print(f"[arm {label}] generated {n_new} tok  "
          f"decode {decode_s:.2f}s = {arm['decode_tokens_per_second']:.3f} tok/s "
          f"({arm['ms_per_token']:.1f} ms/tok)")
    print(f"[arm {label}] prefill {arm['prefill_seconds']:.2f}s = "
          f"{arm['prefill_tokens_per_second']:.1f} tok/s   "
          f"peak={arm['peak_gib']:.2f} GiB")
    if depth is not None:
        st = arm["stats"]
        print(f"[arm {label}] accepted={st['accepted_drafts']} "
              f"rejected={st['rejected_drafts']} drafted_tokens={st['drafted_tokens']} "
              f"verify_calls={st['verify_calls']} mtp_forward_calls={st['mtp_forward_calls']}")
        print(f"[arm {label}] accepted_by_depth={st['accepted_by_depth']} "
              f"drafted_by_depth={st['drafted_by_depth']}")
        for row in arm["accept_rates"]:
            rate = row["accept_rate"]
            print(f"[arm {label}]   depth {row['depth']}: "
                  f"{row['accepted']}/{row['drafted']} = "
                  f"{'n/a' if rate is None else f'{rate:.3f}'}")
        mac = arm["mean_accepted_per_verify_call"]
        print(f"[arm {label}] committed tokens per verify call: "
              f"{'n/a' if mac is None else f'{mac:.3f}'}")
        print(f"[arm {label}] draft {st['draft_time_s']:.2f}s  "
              f"verify {st['verify_time_s']:.2f}s  "
              f"target_forward {st['target_forward_time_s']:.2f}s  "
              f"snapshot {st['snapshot_time_s']:.2f}s")
        gate = arm.get("spec_equals_ar")
        if gate is not None:
            print(f"[arm {label}] spec==AR: "
                  f"{'PASS' if gate['pass'] else 'FAIL'}"
                  + ("" if gate["pass"]
                     else f" (first divergence at index {gate['first_divergence_index']}: "
                          f"AR={gate['baseline_at_divergence']} "
                          f"spec={gate['arm_at_divergence']})"))
    sys.stdout.flush()
    return arm


def _tiny_runtime_and_prompt(n_prompt: int):
    """Reuse the spec gate's shrunk seeded model so --tiny exercises exactly the
    wiring the gates cover.  CPU device, no download, no checkpoint."""
    here = Path(__file__).resolve().parents[1]
    path = here / "tests" / "test_deepseek_v4_spec.py"
    spec = importlib.util.spec_from_file_location("_dsv4_spec_for_bench", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dsv4_spec_for_bench"] = module
    spec.loader.exec_module(module)
    return module._runtime(vocab=8), module._prompt(n_prompt, vocab=8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--prompt-file")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="speculative_depth values; one arm each, after the AR control",
    )
    ap.add_argument("--verify-strategy", default="capture_commit")
    ap.add_argument("--verify-core", default="stock")
    ap.add_argument("--mtp-history-policy", default="committed")
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument(
        "--warmup-tokens",
        type=int,
        default=8,
        help="unrecorded AR warmup before the measured arms (0 to skip)",
    )
    ap.add_argument("--out", help="receipt path stem; writes <stem>.json and <stem>.txt")
    ap.add_argument(
        "--tiny",
        action="store_true",
        help="harness self-test on the spec gate's shrunk seeded model (CPU, "
             "seconds); not a performance measurement",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    load_seconds = 0.0
    config: dict = {}
    quant: dict = {}
    model_path = Path(args.tiny and "." or (args.model or "."))

    if args.tiny:
        rt, prompt_ids = _tiny_runtime_and_prompt(17)
        print(f"[bench] TINY harness self-test: {len(prompt_ids)} prompt tokens")
    else:
        if not args.model:
            sys.exit("no model path; pass --model (or --tiny)")
        model_path = Path(os.path.expanduser(args.model)).resolve()
        from mlx_lm.utils import load_config

        from mtplx import runtime as mtplx_runtime

        config = load_config(model_path)
        quant = config.get("quantization") or {}
        overrides = [k for k in quant if k not in ("group_size", "bits", "mode")]
        print(f"[bench] model      : {model_path}")
        print(f"[bench] model_type : {config.get('model_type')}  "
              f"layers={config.get('num_hidden_layers')}  "
              f"nextn={config.get('num_nextn_predict_layers')}")
        print(f"[bench] quantization: default bits={quant.get('bits')} "
              f"group_size={quant.get('group_size')} mode={quant.get('mode')} "
              f"per-path overrides={len(overrides)}")
        sys.stdout.flush()

        t0 = time.perf_counter()
        rt = mtplx_runtime.load(model_path, mtp=True)
        mx.eval(rt.model.parameters())
        load_seconds = time.perf_counter() - t0
        print(f"[bench] loaded in {load_seconds:.1f}s  "
              f"active={_gib(_active_bytes()):.2f} GiB  "
              f"peak={_gib(_peak_bytes()):.2f} GiB  "
              f"mtp_enabled={rt.mtp_enabled}")
        sys.stdout.flush()
        if not rt.mtp_enabled:
            sys.exit(
                "runtime loaded with mtp_enabled=False: the draft head did not "
                "bind, so there is no speculative lane to benchmark"
            )

        prompt_text = Path(args.prompt_file).read_text() if args.prompt_file else None
        if prompt_text is None:
            sys.exit("no prompt; pass --prompt-file")
        prompt_ids = list(rt.tokenizer.encode(prompt_text))
        total_context = len(prompt_ids) + args.max_tokens
        print(f"[bench] prompt tokens: {len(prompt_ids)}  new: {args.max_tokens}  "
              f"total context: {total_context}")
        if total_context > args.max_context:
            sys.exit(
                f"total context {total_context} exceeds --max-context "
                f"({args.max_context}); raise it deliberately, after checking the "
                f"quadratic score-tensor cost against the wired-memory budget"
            )
        sys.stdout.flush()

    after_load_active = _active_bytes()

    # Unrecorded warmup so the AR control is not the arm that pays first-call
    # allocator and kernel-compile cost.  Decode tok/s on this backend is stable
    # across loads (4.513 vs 4.514 in the 20260731 smoke receipts) but prefill is
    # not, and prefill is recorded per arm.
    if args.warmup_tokens > 0:
        print(f"[bench] warmup: AR, {args.warmup_tokens} tokens (not recorded)")
        sys.stdout.flush()
        _run_arm(
            rt=rt,
            label="warmup",
            depth=None,
            prompt_ids=prompt_ids,
            max_tokens=args.warmup_tokens,
            verify_strategy=args.verify_strategy,
            verify_core=args.verify_core,
            mtp_history_policy=args.mtp_history_policy,
            baseline_tokens=None,
        )

    arms: list[dict] = []
    ar = _run_arm(
        rt=rt,
        label="AR",
        depth=None,
        prompt_ids=prompt_ids,
        max_tokens=args.max_tokens,
        verify_strategy=args.verify_strategy,
        verify_core=args.verify_core,
        mtp_history_policy=args.mtp_history_policy,
        baseline_tokens=None,
    )
    arms.append(ar)
    baseline_tokens = ar.get("tokens")
    status = 0
    if baseline_tokens is None:
        print("[bench] AR control failed; the K arms have nothing to be gated against")
        status = 1

    for depth in args.depths:
        if not args.tiny and _gib(_peak_bytes()) > _PEAK_ABORT_GIB:
            print(f"[bench] ABORT: peak {_gib(_peak_bytes()):.2f} GiB is over the "
                  f"{_PEAK_ABORT_GIB} GiB per-arm guard; the wired knob is never "
                  f"raised, so the remaining arms are not run")
            status = 1
            break
        arm = _run_arm(
            rt=rt,
            label=f"K={depth}",
            depth=depth,
            prompt_ids=prompt_ids,
            max_tokens=args.max_tokens,
            verify_strategy=args.verify_strategy,
            verify_core=args.verify_core,
            mtp_history_policy=args.mtp_history_policy,
            baseline_tokens=baseline_tokens,
        )
        arms.append(arm)
        if arm.get("error"):
            status = 1
        gate = arm.get("spec_equals_ar")
        if gate is not None and not gate["pass"]:
            status = 1

    # ---- summary table ----------------------------------------------------
    ar_tps = float(ar.get("decode_tokens_per_second") or 0.0)
    print(f"\n{'=' * 78}\n=== FOUR-ARM SUMMARY ===\n{'=' * 78}")
    header = (f"{'arm':>6}  {'tok/s':>8}  {'ms/tok':>7}  {'x AR':>6}  "
              f"{'tok/cycle':>9}  {'peak GiB':>8}  {'spec==AR':>9}")
    print(header)
    print("-" * len(header))
    for arm in arms:
        if arm.get("error"):
            print(f"{arm['label']:>6}  {'ERROR':>8}")
            continue
        tps = float(arm.get("decode_tokens_per_second") or 0.0)
        mac = arm.get("mean_accepted_per_verify_call")
        gate = arm.get("spec_equals_ar")
        print(f"{arm['label']:>6}  {tps:8.3f}  {arm['ms_per_token']:7.1f}  "
              f"{(tps / ar_tps if ar_tps else 0.0):6.3f}  "
              f"{('n/a' if mac is None else f'{mac:.3f}'):>9}  "
              f"{arm['peak_gib']:8.2f}  "
              f"{('-' if gate is None else ('PASS' if gate['pass'] else 'FAIL')):>9}")
    for arm in arms:
        if arm.get("speculative_depth") is None or arm.get("error"):
            continue
        parts = []
        for r in arm["accept_rates"]:
            rate = r["accept_rate"]
            shown = "n/a" if rate is None else f"{rate:.3f}"
            parts.append(f"d{r['depth']}={shown} ({r['accepted']}/{r['drafted']})")
        print(f"  {arm['label']} accept rates: {', '.join(parts)}")
    sys.stdout.flush()

    receipt = {
        "harness": "scripts/deepseek_v4_mtpk_bench.py",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": ["python", *sys.argv],
        "host": {
            "platform": platform.platform(),
            "mlx_version": mx.__version__,
            "python": sys.version.split()[0],
        },
        "env": {
            k: v for k, v in sorted(os.environ.items())
            if k.startswith("MTPLX_") or k in ("HF_HUB_OFFLINE", "PYTHONPATH")
        },
        "tiny": bool(args.tiny),
        "model_path": str(model_path),
        "model_type": config.get("model_type"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
        "quantization": {
            "default_bits": quant.get("bits"),
            "default_group_size": quant.get("group_size"),
            "default_mode": quant.get("mode"),
        },
        "sampling": {"greedy": True, "temperature": 0.0, "stop_token_ids": []},
        "prompt_file": args.prompt_file,
        "prompt_tokens": len(prompt_ids),
        "max_tokens": args.max_tokens,
        "verify_strategy": args.verify_strategy,
        "verify_core": args.verify_core,
        "mtp_history_policy": args.mtp_history_policy,
        "load_seconds": load_seconds,
        "active_after_load_gib": _gib(after_load_active),
        "arms": arms,
        "status": status,
    }

    if args.out:
        stem = Path(args.out)
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".json").write_text(json.dumps(receipt, indent=2))
        blocks = []
        for arm in arms:
            if arm.get("error"):
                blocks.append(f"{'=' * 72}\nARM {arm['label']}: ERROR\n"
                              f"{'=' * 72}\n{arm['error']}\n")
                continue
            blocks.append(
                f"{'=' * 72}\nARM {arm['label']}  "
                f"({arm['generated_tokens']} tokens, greedy, "
                f"{arm['decode_tokens_per_second']:.3f} tok/s)\n"
                f"{'=' * 72}\n{arm['text']}\n"
            )
        stem.with_suffix(".txt").write_text(
            f"PROMPT ({len(prompt_ids)} tokens) from {args.prompt_file}\n"
            + "\n".join(blocks)
        )
        print(f"receipts         : {stem.with_suffix('.json')}")
        print(f"                   {stem.with_suffix('.txt')}")
        sys.stdout.flush()

    return status


if __name__ == "__main__":
    raise SystemExit(main())
