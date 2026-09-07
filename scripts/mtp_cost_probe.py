#!/usr/bin/env python3
"""Replay an uncapped request at AR and fixed MTP depths, retaining evidence.

Use an isolated, already-running daemon with request logging and verified
max fans. The request JSON supplies the real task and native sampler; no
output/reasoning limit or stop string is inserted by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from mtplx.commands.trace_metrics import mtp_economics


def read_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def fans(command):
    state = json.loads(subprocess.check_output([command, "status"], text=True))
    rows = state.get("fans", [])
    if not rows or not all(r["mode"] != "auto" and
                           r["actual_rpm"] >= .95 * r["max_rpm"] for r in rows):
        raise RuntimeError("Max-fan ramp is not verified")
    return state


def request(base, body, path):
    req = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
        "x-mtplx-allow-client-controls": "true"})
    started = time.time()
    first = last = None
    gap = 0.
    rid = finish = None
    usage = {}
    try:
        response = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as exc:
        error_path = path.with_suffix(".http-error.json")
        error_path.write_bytes(exc.read())
        raise RuntimeError(f"HTTP {exc.code}; response saved to {error_path}") from exc
    with response, path.open("w") as sink:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            now = time.time()
            sink.write(json.dumps({"ts": now, "event": event}, ensure_ascii=False) + "\n")
            sink.flush()
            if "error" in event:
                raise RuntimeError(event["error"])
            rid = event.get("id", rid)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if any(delta.get(k) for k in ("content", "reasoning_content", "reasoning", "tool_calls")):
                    first = first or now
                    if last is not None:
                        gap = max(gap, now-last)
                    last = now
                finish = choice.get("finish_reason") or finish
    if finish in (None, "error", "length"):
        raise RuntimeError(f"Request did not complete naturally: {finish}")
    return {"request_id": rid, "started_s": started, "wall_s": time.time()-started,
            "ttft_s": first-started if first else None, "max_emit_gap_s": gap,
            "finish_reason": finish, "usage": usage}


def receipt(path, rid):
    deadline = time.monotonic()+15
    while True:
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get("request_id") == rid:
                    return row
        if time.monotonic() > deadline:
            raise RuntimeError(f"No durable receipt for {rid}")
        time.sleep(.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8211")
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depths", default="0,1,2,3")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed-map", type=Path,
                        help="JSON mapping run labels (r1-d0, ...) to baseline resolved seeds")
    parser.add_argument("--fan-command", default=shutil.which("thermalforge"))
    args = parser.parse_args()
    depths = [int(d) for d in args.depths.split(",")]
    if not depths or min(depths) < 0 or args.repeats < 1 or not args.fan_command:
        parser.error("Need nonnegative depths, positive repeats, and thermalforge")
    raw = args.request.read_bytes()
    body = json.loads(raw)
    seeds = json.loads(args.seed_map.read_text()) if args.seed_map else {}
    args.out.mkdir(parents=True, exist_ok=False)
    health = read_json(args.base_url + "/health")
    model = read_json(args.base_url + "/v1/models")["data"][0]["id"]
    if health.get("active_requests"):
        raise RuntimeError("Daemon is busy; run the probe after the active client finishes")
    identity = {"request_sha256": hashlib.sha256(raw).hexdigest(), "health": health, "model": model,
                "fans": fans(args.fan_command), "depths": depths, "repeats": args.repeats,
                "seed_map": seeds}
    (args.out / "identity.json").write_text(json.dumps(identity, indent=2))
    rows = []
    for repeat in range(args.repeats):
        for depth in (depths if repeat % 2 == 0 else depths[::-1]):
            label = f"r{repeat+1}-d{depth}"
            sent = {**body, "model": model, "stream": True,
                    "stream_options": {"include_usage": True},
                    "generation_mode": "ar" if depth == 0 else "mtp", "depth": depth}
            if args.seed_map:
                sent["seed"] = int(seeds[label])
            fan = fans(args.fan_command)
            (args.out / f"{label}-request.json").write_text(json.dumps(sent, indent=2))
            row = request(args.base_url, sent, args.out / f"{label}-stream.jsonl")
            rec = receipt(args.request_log, row["request_id"])
            if rec.get("generation_mode") != sent["generation_mode"] or (depth and rec.get("mtp_depth") != depth):
                raise RuntimeError("The daemon did not honor the requested generation mode/depth")
            row.update(label=label, depth=depth, repeat=repeat+1, receipt=rec, fans=fan,
                       guard_passed=not bool(rec.get("repetition_stop_triggered")))
            rows.append(row)
            (args.out / f"{label}-result.json").write_text(json.dumps(row, indent=2))
            print(json.dumps({"label": label, "wall_s": row["wall_s"],
                              "decode_tok_s": rec.get("decode_tok_s"),
                              "tokens": rec.get("completion_tokens")}), flush=True)
    ar = [r["receipt"]["decode_tok_s"] for r in rows if r["depth"] == 0 and r["guard_passed"]]
    ar_median = statistics.median(ar) if ar else None
    for row in rows:
        row["economics"] = mtp_economics(row["receipt"], ar_median)
    (args.out / "summary.json").write_text(json.dumps({"ar_median_tok_s": ar_median,
        "guard_passed": all(r["guard_passed"] for r in rows),
        "caveat": "Sampled outputs vary. Guard-stopped samples are invalid; passing this guard does not validate the generated code. Compare repeated matched workloads and full task completion.",
        "runs": rows}, indent=2))
    if not all(r["guard_passed"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
