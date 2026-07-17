#!/usr/bin/env python3
"""Reproduce within-session verify degradation: simulated agent loop.

Same session id, transcript grows ~600 tokens per round (assistant tool_call +
tool result), 300-token generations. Tracks verify_ms round over round at
near-constant context depth (starts ~12k).
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
TAG = sys.argv[2] if len(sys.argv) > 2 else "loop"
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 16

sys.path.insert(0, "scripts/ocspeed-20260703")
from accept_depth_probe import make_padding  # noqa: E402

MODEL = "mtplx-qwen36-27b-optimized-speed"
TOOLS = [{"type": "function", "function": {
    "name": n, "description": f"{n} tool for the workspace",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}}, "required": []}}}
    for n in ("read", "grep", "edit", "bash")]

messages = [
    {"role": "system", "content": make_padding("rules", 3000)},
    {"role": "user", "content": (
        "Work through the archive module by module. For each module, request "
        "its file with the read tool, then record two observations. Keep "
        "going until told to stop.")},
]
# seed depth ~12k with a big first tool round
messages.append({"role": "assistant", "content": "",
                 "tool_calls": [{"id": "call_seed", "type": "function",
                                 "function": {"name": "read",
                                              "arguments": '{"path":"archive/seed.txt"}'}}]})
messages.append({"role": "tool", "tool_call_id": "call_seed",
                 "content": make_padding("lorem", 8000)})

out = open(f"outputs/ocspeed-20260703/loopdeg_{TAG}.jsonl", "a")
for rnd in range(ROUNDS):
    body = {"model": MODEL, "messages": messages, "max_tokens": 300,
            "temperature": 0.6, "top_p": 0.95, "stream": True,
            "enable_thinking": False, "tools": TOOLS, "tool_choice": "auto"}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "X-MTPLX-Session-ID": f"loopdeg-{TAG}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        for _ in r:
            pass
    wall = time.perf_counter() - t0
    with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/v1/mtplx/snapshot", timeout=10) as r:
        lat = json.loads(r.read().decode()).get("latest") or {}
    vc = lat.get("verify_calls") or 1
    row = {"round": rnd, "tag": TAG,
           "ptok": lat.get("prompt_tokens"),
           "cached": lat.get("cached_tokens"),
           "newpf": lat.get("new_prefill_tokens"),
           "ttft": round(lat.get("ttft_s") or 0, 2),
           "decode": round(lat.get("decode_tok_s") or 0, 1),
           "vms": round(1000 * (lat.get("verify_time_s") or 0) / vc, 1),
           "tokv": round((lat.get("completion_tokens") or 0) / vc, 2),
           "ctok": lat.get("completion_tokens"), "wall": round(wall, 1)}
    print(json.dumps(row), flush=True)
    out.write(json.dumps(row) + "\n")
    out.flush()
    # extend the transcript like a real agent loop: assistant tool_call,
    # tool result (~500 tokens), no think-time gap.
    messages.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"call_{rnd}", "type": "function",
                                     "function": {"name": "read",
                                                  "arguments": json.dumps({"path": f"archive/mod{rnd}.txt"})}}]})
    messages.append({"role": "tool", "tool_call_id": f"call_{rnd}",
                     "content": make_padding("lorem", 450) + f" module {rnd} end."})
print("LOOPDEG DONE", flush=True)
