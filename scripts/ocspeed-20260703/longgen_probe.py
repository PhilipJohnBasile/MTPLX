#!/usr/bin/env python3
"""Long-generation probe: 1600-token generations at bare and 7k contexts.

Watches for compile-cliff/stall behavior on the W2 dense bucket ladder:
reports decode tok/s plus sliding first/last-32 and max inter-chunk gap.
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
TAG = sys.argv[2]
OUT = sys.argv[3]
MODEL = "mtplx-qwen36-27b-optimized-speed"

LOREM = None


def lorem7k():
    global LOREM
    if LOREM is None:
        sys.path.insert(0, "scripts/ocspeed-20260703")
        from accept_depth_probe import make_padding
        LOREM = make_padding("lorem", 7000)
    return LOREM


def fire(label, messages, max_tokens, session):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.6, "top_p": 0.95, "stream": True,
            "enable_thinking": False}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "X-MTPLX-Session-ID": session})
    gaps = []
    last = None
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        for _ in r:
            now = time.perf_counter()
            if last is not None:
                gaps.append(now - last)
            last = now
    wall = time.perf_counter() - t0
    with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/v1/mtplx/snapshot", timeout=10) as r:
        lat = json.loads(r.read().decode()).get("latest") or {}
    gaps.sort()
    row = {
        "cell": label, "tag": TAG, "wall_s": round(wall, 2),
        "decode_tok_s": round(lat.get("decode_tok_s") or 0, 2),
        "completion": lat.get("completion_tokens"),
        "verify_calls": lat.get("verify_calls"),
        "verify_ms": round(1000 * (lat.get("verify_time_s") or 0)
                           / max(1, lat.get("verify_calls") or 1), 2),
        "tokens_per_verify": round((lat.get("completion_tokens") or 0)
                                   / max(1, lat.get("verify_calls") or 1), 3),
        "first32": round(lat.get("sliding_decode_tok_s_first_32") or 0, 1),
        "last32": round(lat.get("sliding_decode_tok_s_last_32") or 0, 1),
        "gap_p99_ms": round(1000 * gaps[int(len(gaps) * 0.99)], 1) if gaps else None,
        "gap_max_ms": round(1000 * gaps[-1], 1) if gaps else None,
        "prompt_tokens": lat.get("prompt_tokens"),
    }
    print(json.dumps(row), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    time.sleep(2)


PROMPT_LONG = ("Write a detailed design document for a 2D physics game engine: "
               "architecture, collision system, integration loop, memory layout, "
               "and a full example. Be thorough and keep going.")

for rep in range(2):
    fire(f"bare_long#r{rep}", [{"role": "user", "content": PROMPT_LONG}],
         1600, f"lg-{TAG}-bare")
for rep in range(2):
    fire(f"lorem7k_long#r{rep}",
         [{"role": "system", "content": lorem7k()},
          {"role": "user", "content": PROMPT_LONG}],
         1600, f"lg-{TAG}-7k")
print("LONGGEN DONE", flush=True)
