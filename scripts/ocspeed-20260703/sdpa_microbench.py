#!/usr/bin/env python3
"""Microbench mx.fast.scaled_dot_product_attention at the exact verify shape.

Qwen3.6-27B full-attention geometry: B=1, Hq=24, Hk=4 (GQA 6), D=256.
Verify q_len = depth+1 = 4. Sweep KV length; compare against the
memory-bound floor and alternate implementations.
"""
import time

import mlx.core as mx

B, HQ, HK, D = 1, 24, 4, 256
BYTES_PER_TOKEN_PER_LAYER = HK * D * 2 * 2  # K+V bf16


def bench(fn, *args, warmup=5, iters=30, **kw):
    for _ in range(warmup):
        mx.eval(fn(*args, **kw))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(*args, **kw))
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def fused(q, k, v, scale, mask):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def manual(q, k, v, scale, mask):
    # unfused reference: expand GQA, matmul + softmax + matmul
    kx = mx.repeat(k, HQ // HK, axis=1)
    vx = mx.repeat(v, HQ // HK, axis=1)
    scores = (q * scale) @ kx.transpose(0, 1, 3, 2)
    if mask is not None and not isinstance(mask, str):
        scores = scores + mask
    probs = mx.softmax(scores, axis=-1, precise=True)
    return probs @ vx


def main():
    print(f"geometry B={B} Hq={HQ} Hk={HK} D={D}")
    print(f"{'kv_len':>8} {'q_len':>5} {'fused_ms':>9} {'membound_ms':>11} {'x_over':>7}")
    for q_len in (1, 4):
        for kv in (512, 2048, 4096, 8192, 16384, 32768):
            q = mx.random.normal((B, HQ, q_len, D)).astype(mx.bfloat16)
            k = mx.random.normal((B, HK, kv + q_len, D)).astype(mx.bfloat16)
            v = mx.random.normal((B, HK, kv + q_len, D)).astype(mx.bfloat16)
            mask = "causal" if q_len > 1 else None
            t = bench(fused, q, k, v, 1.0 / (D ** 0.5), mask)
            mem_ms = 1000 * (kv + q_len) * BYTES_PER_TOKEN_PER_LAYER / 614e9
            print(f"{kv:>8} {q_len:>5} {1000*t:>9.3f} {mem_ms:>11.4f} {t*1000/mem_ms:>7.1f}",
                  flush=True)
    # 16-layer verify estimate at 8k
    q = mx.random.normal((B, HQ, 4, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, HK, 8196, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, HK, 8196, D)).astype(mx.bfloat16)
    t = bench(fused, q, k, v, 1.0 / (D ** 0.5), "causal")
    print(f"\n16-layer projection at 8k, q=4: {16*1000*t:.2f} ms per verify forward")
    tm = bench(manual, q, k, v, 1.0 / (D ** 0.5), None)
    print(f"manual unfused single layer at 8k, q=4: {1000*tm:.3f} ms")


if __name__ == "__main__":
    main()
