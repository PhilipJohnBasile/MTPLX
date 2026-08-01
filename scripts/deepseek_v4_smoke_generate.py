"""Real-weights generation smoke test for the MTPLX deepseek_v4 backend.

Loads a real ``mlx-community/DeepSeek-V4-Flash-*`` checkpoint through the *mtplx*
load path (``mtplx.runtime._load_base_model`` -> ``mlx_lm.utils.load_model`` with
``get_model_classes`` resolving to ``mtplx.models.deepseek_v4``), runs one greedy
completion off ``Model.make_cache()``, and reports load time, prefill tok/s,
decode tok/s and peak memory.

This is a smoke harness, not a benchmark suite: one prompt, one run, temp 0.

Context budget: the ratio-4 dense attention path is exact only while every
compressed window is selected (``n_comp <= index_topk``, i.e. ~2048 tokens of
context). The harness refuses to run past that so a "coherent output" verdict
is never taken from a regime the backend does not yet cover.

MUST run inside the box's serialized MLX window (bench/laguna/run_guarded.py) —
the 2-bit checkpoint is ~90 GiB and does not fit beside the served model.

Usage:
  python scripts/deepseek_v4_smoke_generate.py \
      --model ~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-2bit-DQ/snapshots/<rev> \
      --max-tokens 128 --out bench/deepseek-v4/smoke-2bitdq-YYYYMMDD
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
import time
from pathlib import Path

import mlx.core as mx


DEFAULT_PROMPT = '''"""Rolling-window rate limiter used by the ingest workers."""

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    """Allow at most ``max_events`` in any ``window_seconds`` sliding window.

    The limiter is intentionally not thread-safe; each ingest worker owns one.
    Timestamps are monotonic so a clock adjustment cannot open the gate early.
    """

    max_events: int
    window_seconds: float
    _events: deque = field(default_factory=deque)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()

    def allow(self) -> bool:
        """Record and admit one event, or return False if the window is full."""
        now = time.monotonic()
        self._evict(now)
        if len(self._events) >= self.max_events:
            return False
        self._events.append(now)
        return True

    def retry_after(self) -> float:
        """Seconds until the next event would be admitted (0.0 if admissible)."""
        now = time.monotonic()
        self._evict(now)
        if len(self._events) < self.max_events:
            return 0.0
        return self._events[0] + self.window_seconds - now

    def reset(self) -> None:
'''

# The dense-over-compressed attention path is exact while n_comp <= index_topk.
_CONTEXT_GUARD_TOKENS = 2048


def _default_model() -> str | None:
    hits = sorted(
        glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/"
                "models--mlx-community--DeepSeek-V4-Flash-2bit-DQ/snapshots/*/"
            )
        )
    )
    return hits[0] if hits else None


def _peak_bytes() -> int:
    for getter in ("get_peak_memory",):
        fn = getattr(mx, getter, None)
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


def _gib(n: int) -> float:
    return n / (1024**3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_default_model())
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument(
        "--out",
        default=None,
        help="receipt path stem; writes <stem>.json and <stem>.txt",
    )
    args = ap.parse_args()
    if not args.model:
        sys.exit("no model path; pass --model")
    model_path = Path(os.path.expanduser(args.model)).resolve()

    prompt = (
        Path(args.prompt_file).read_text()
        if args.prompt_file
        else DEFAULT_PROMPT
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mlx_lm.utils import load_config

    from mtplx.runtime import _load_base_model

    config = load_config(model_path)
    print(f"[smoke] model      : {model_path}")
    print(f"[smoke] model_type : {config.get('model_type')}  "
          f"layers={config.get('num_hidden_layers')}")
    quant = config.get("quantization") or {}
    overrides = [k for k in quant if k not in ("group_size", "bits", "mode")]
    print(f"[smoke] quantization: default bits={quant.get('bits')} "
          f"group_size={quant.get('group_size')} mode={quant.get('mode')} "
          f"per-path overrides={len(overrides)}")
    sys.stdout.flush()

    t0 = time.perf_counter()
    model, tokenizer = _load_base_model(model_path, config)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - t0
    after_load_active = _active_bytes()
    print(f"[smoke] loaded in {load_seconds:.1f}s  "
          f"active={_gib(after_load_active):.2f} GiB  "
          f"peak={_gib(_peak_bytes()):.2f} GiB")
    sys.stdout.flush()

    prompt_ids = tokenizer.encode(prompt)
    n_prompt = len(prompt_ids)
    total_context = n_prompt + args.max_tokens
    print(f"[smoke] prompt tokens: {n_prompt}  "
          f"new tokens: {args.max_tokens}  total context: {total_context}")
    if total_context > _CONTEXT_GUARD_TOKENS:
        sys.exit(
            f"total context {total_context} exceeds the exact-path budget "
            f"({_CONTEXT_GUARD_TOKENS}); the ratio-4 indexer top-k filter is "
            f"deferred, so a longer run would not be a valid verdict"
        )
    sys.stdout.flush()

    cache = model.make_cache()
    ids = mx.array(prompt_ids)[None]

    # ---- prefill ----------------------------------------------------------
    t0 = time.perf_counter()
    logits = model(ids, cache=cache)
    last = logits[:, -1]
    token = mx.argmax(last, axis=-1)
    mx.eval(token)
    prefill_seconds = time.perf_counter() - t0
    first_id = int(token.item())
    print(f"[smoke] prefill {n_prompt} tok in {prefill_seconds:.2f}s = "
          f"{n_prompt / prefill_seconds:.1f} tok/s   first token id={first_id}")
    sys.stdout.flush()

    ln = logits[0, -1].astype(mx.float32)
    mx.eval(ln)
    finite = bool(mx.all(mx.isfinite(ln)).item())
    spread = float(mx.std(ln).item())
    print(f"[smoke] first-token logits finite={finite} std={spread:.4f}")
    del logits, last, ln

    eos_ids = set()
    for attribute in ("eos_token_ids", "eos_token_id"):
        value = getattr(tokenizer, attribute, None)
        if isinstance(value, int):
            eos_ids.add(value)
        elif isinstance(value, (list, tuple, set)):
            eos_ids |= {int(v) for v in value}

    # ---- decode -----------------------------------------------------------
    generated = [first_id]
    step_seconds: list[float] = []
    printed = 0
    text = ""
    stopped_on_eos = first_id in eos_ids
    token = token[:, None]
    if not stopped_on_eos:
        for _ in range(args.max_tokens - 1):
            t0 = time.perf_counter()
            logits = model(token, cache=cache)
            token = mx.argmax(logits[:, -1], axis=-1)
            mx.eval(token)
            step_seconds.append(time.perf_counter() - t0)
            next_id = int(token.item())
            token = token[:, None]
            generated.append(next_id)
            if next_id in eos_ids:
                stopped_on_eos = True
                break
            # streamed print, deliberately outside the timed region
            text = tokenizer.decode(generated)
            if len(text) > printed:
                sys.stdout.write(text[printed:])
                sys.stdout.flush()
                printed = len(text)

    text = tokenizer.decode(generated)
    if len(text) > printed:
        sys.stdout.write(text[printed:])
    sys.stdout.write("\n")
    sys.stdout.flush()

    decode_seconds = sum(step_seconds)
    decode_tps = (len(step_seconds) / decode_seconds) if decode_seconds else 0.0
    peak = _peak_bytes()

    print("\n=== SMOKE SUMMARY ===")
    print(f"load             : {load_seconds:.1f} s")
    print(f"prefill          : {n_prompt} tok / {prefill_seconds:.2f} s = "
          f"{n_prompt / prefill_seconds:.2f} tok/s")
    print(f"decode           : {len(step_seconds)} tok / {decode_seconds:.2f} s = "
          f"{decode_tps:.3f} tok/s  ({decode_seconds / max(len(step_seconds), 1):.3f} s/tok)")
    print(f"tokens generated : {len(generated)} (eos hit: {stopped_on_eos})")
    print(f"peak memory      : {_gib(peak):.2f} GiB")
    print(f"active memory    : {_gib(_active_bytes()):.2f} GiB")
    print(f"logits finite    : {finite}   std={spread:.4f}")

    receipt = {
        "harness": "scripts/deepseek_v4_smoke_generate.py",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": ["python", *sys.argv],
        "host": {
            "platform": platform.platform(),
            "mlx_version": mx.__version__,
            "python": sys.version.split()[0],
        },
        "model_path": str(model_path),
        "model_type": config.get("model_type"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "quantization": {
            "default_bits": quant.get("bits"),
            "default_group_size": quant.get("group_size"),
            "default_mode": quant.get("mode"),
            "per_path_overrides": len(overrides),
        },
        "sampling": {"greedy": True, "temperature": 0.0},
        "prompt": prompt,
        "prompt_tokens": n_prompt,
        "max_tokens": args.max_tokens,
        "generated_token_ids": generated,
        "generated_text": text,
        "stopped_on_eos": stopped_on_eos,
        "first_token_logits": {"finite": finite, "std": spread},
        "timings": {
            "load_seconds": load_seconds,
            "prefill_seconds": prefill_seconds,
            "prefill_tokens_per_second": n_prompt / prefill_seconds,
            "decode_seconds": decode_seconds,
            "decode_tokens": len(step_seconds),
            "decode_tokens_per_second": decode_tps,
            "decode_step_seconds": step_seconds,
        },
        "memory": {
            "peak_bytes": peak,
            "peak_gib": _gib(peak),
            "active_after_load_bytes": after_load_active,
            "active_after_load_gib": _gib(after_load_active),
            "active_end_gib": _gib(_active_bytes()),
        },
    }

    if args.out:
        stem = Path(args.out)
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".json").write_text(json.dumps(receipt, indent=2))
        stem.with_suffix(".txt").write_text(
            f"PROMPT ({n_prompt} tokens)\n{'=' * 72}\n{prompt}\n"
            f"{'=' * 72}\nGENERATED ({len(generated)} tokens, greedy)\n"
            f"{'=' * 72}\n{text}\n"
        )
        print(f"receipts         : {stem.with_suffix('.json')}")
        print(f"                   {stem.with_suffix('.txt')}")

    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
