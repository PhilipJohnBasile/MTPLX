#!/usr/bin/env python3
"""Bisect the verify-forward depth wall in-process.

Loads the runtime exactly like serve (same env), prefills a synthetic context
to several depths, then times forward_ar at q_len in {1, 4} per depth:
  1. stock          - as-served
  2. attn_window    - full-attention layers truncate KV reads to the last 512
                      tokens (timing-only hack; output wrong, never committed)
This splits the depth slope into "attention reads" vs "everything else".
"""
import os
import time

os.environ.setdefault("MTPLX_NAX_VERIFY", "1")
os.environ.setdefault("MTPLX_NAX_M4_IMPL", "vk_k")

import mlx.core as mx

from mtplx.profiles import SUSTAINED_PREFILL_ENV

for key, value in SUSTAINED_PREFILL_ENV.items():
    os.environ.setdefault(key, value)

from mtplx import runtime as mtplx_runtime  # noqa: E402

MODEL = os.path.expanduser(
    "~/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed"
)
DEPTHS = [32, 2048, 4096, 8192, 12288]
QLENS = [1, 4]
ITERS = 12


def load_runtime():
    return mtplx_runtime.load(MODEL, mtp=True)


def prefill_to(rt, cache, depth, chunk=2048):
    import random
    random.seed(7)
    vocab_hi = 140000
    remaining = depth
    while remaining > 0:
        n = min(chunk, remaining)
        ids = [random.randrange(1000, vocab_hi) for _ in range(n)]
        rt.forward_ar(mx.array([ids]), cache=cache, return_hidden=False,
                      emit_logits=False)
        from mtplx.generation import _eval_cache_roots
        _eval_cache_roots(cache)
        remaining -= n


def time_forward(rt, cache, q_len):
    ids = mx.array([[11 + i for i in range(q_len)]])
    # warmup
    for _ in range(3):
        logits, hidden = rt.forward_ar(ids, cache=cache, return_hidden=True)
        mx.eval(hidden)
        cache_trim(cache, q_len)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        logits, hidden = rt.forward_ar(ids, cache=cache, return_hidden=True)
        mx.eval(hidden)
        cache_trim(cache, q_len)
    mx.synchronize()
    return (time.perf_counter() - t0) / ITERS


def cache_trim(cache, n):
    for layer in cache:
        if hasattr(layer, "trim"):
            layer.trim(n)
        elif hasattr(layer, "offset"):
            layer.offset = max(0, int(layer.offset) - n)


def install_attention_window(window=512):
    """Monkeypatch the split hook's SDPA to read only the last `window` KV."""
    import mlx_lm.models.base as base

    orig = base.scaled_dot_product_attention

    def windowed(queries, keys, values, cache, scale, mask, sinks=None):
        n = keys.shape[2]
        if n > window:
            keys = keys[:, :, n - window:, :]
            values = values[:, :, n - window:, :]
            mask = None
        return orig(queries, keys, values, cache=cache, scale=scale,
                    mask=mask, sinks=sinks)

    base.scaled_dot_product_attention = windowed
    # attention_split imports it inside the function body, so patching the
    # module attribute is enough.
    return orig


def main():
    print("loading runtime...", flush=True)
    rt = load_runtime()
    from mtplx.generation import _make_target_prefill_cache

    results = {}
    for mode in ("stock", "attn_window"):
        restore = None
        if mode == "attn_window":
            restore = install_attention_window(512)
        rows = []
        for depth in DEPTHS:
            cache = _make_target_prefill_cache(rt)
            prefill_to(rt, cache, depth)
            for q in QLENS:
                t = time_forward(rt, cache, q)
                rows.append((depth, q, t))
                print(f"{mode:12} depth={depth:>6} q={q} {1000*t:8.2f} ms",
                      flush=True)
            del cache
            mx.clear_cache()
        results[mode] = rows
        if restore is not None:
            import mlx_lm.models.base as base
            base.scaled_dot_product_attention = restore

    print("\nDELTA (stock - attn_window) = attention read cost:")
    for (d, q, ts), (_, _, tw) in zip(results["stock"], results["attn_window"]):
        print(f"depth={d:>6} q={q}  stock={1000*ts:7.2f}  window={1000*tw:7.2f}"
              f"  attn_share={1000*(ts-tw):6.2f} ms")


if __name__ == "__main__":
    main()
