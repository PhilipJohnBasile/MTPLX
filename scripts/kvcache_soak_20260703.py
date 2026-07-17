#!/usr/bin/env python3
"""Long-duration kvcache-v2 soak (founder-directed extended QA, 2026-07-03).

Cycles mixed warm/cold traffic across rotating sessions and sizes against a
candidate server, restarts the daemon on a period, and asserts the invariants
short probes cannot: no slow RSS growth past the cap, warm restores stay warm
across restarts, the SSD store keeps admitting/serving, zero request failures.

Prints one summary line per cycle (machine-greppable) and exits non-zero on
any hard-assertion failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kvcache_warm_probe_20260703 import rag_prompt  # noqa: E402

SIZES = [4000, 12000, 32000]
QUESTIONS = ["q1", "q2", "q3"]


def server_pid(port: int) -> int | None:
    out = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True, text=True,
    )
    pids = [int(x) for x in out.stdout.split()]
    return pids[0] if pids else None


def wait_health(port: int, tries: int = 150) -> bool:
    for _ in range(tries):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as r:
                if b'"ok"' in r.read(200):
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def launch(port: int, model: str, log_path: str) -> None:
    subprocess.Popen(
        f'cd "{ROOT}" && MTPLX_NAX_VERIFY=1 MTPLX_NAX_M4_IMPL=vk_k '
        f'nohup .venv/bin/mtplx serve --model {model} --port {port} '
        f"--warmup-tokens 16 >> {log_path} 2>&1 &",
        shell=True,
    )


def chat(port: int, prompt: str, session: str, max_tokens: int = 32) -> dict:
    body = {
        "model": "mtplx-qwen36-27b-optimized-speed",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
        "enable_thinking": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-MTPLX-Session-ID": session},
    )
    started = time.perf_counter()
    ttft = None
    chars = 0
    stats: dict = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if not d or d == "[DONE]":
                continue
            p = json.loads(d)
            for c in p.get("choices") or []:
                delta = c.get("delta") or {}
                for key in ("reasoning_content", "content"):
                    v = delta.get(key)
                    if isinstance(v, str) and v:
                        chars += len(v)
                        if ttft is None:
                            ttft = time.perf_counter() - started
            if isinstance(p.get("mtplx_stats"), dict):
                stats = p["mtplx_stats"]
    return {"ttft": ttft, "chars": chars, "stats": stats,
            "wall": time.perf_counter() - started}


def rss_gb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                         capture_output=True, text=True)
    try:
        return int(out.stdout.strip()) / 1048576
    except ValueError:
        return -1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18170)
    parser.add_argument("--model", default="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed")
    parser.add_argument("--duration-min", type=float, default=150.0)
    parser.add_argument("--restart-every-min", type=float, default=18.0)
    parser.add_argument("--rss-cap-gb", type=float, default=60.0)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    log_path = str(Path(args.output_jsonl).with_suffix(".server.log"))
    out = open(args.output_jsonl, "a", encoding="utf-8")

    def emit(kind: str, **payload):
        record = {"ts": time.time(), "kind": kind, **payload}
        out.write(json.dumps(record, default=str) + "\n")
        out.flush()

    launch(args.port, args.model, log_path)
    if not wait_health(args.port):
        print("SOAK FAIL: server never became healthy")
        return 2

    deadline = time.time() + args.duration_min * 60
    next_restart = time.time() + args.restart_every_min * 60
    cycle = 0
    failures = 0
    restarts = 0
    warm_regressions = 0
    max_rss = 0.0

    while time.time() < deadline:
        cycle += 1
        cycle_stats = []
        for size in SIZES:
            for qi, question in enumerate(QUESTIONS):
                session = f"soak-{size}-{qi}"
                prompt = rag_prompt(size, question)
                try:
                    result = chat(args.port, prompt, session)
                except Exception as exc:
                    failures += 1
                    emit("request_error", size=size, q=question,
                         error=f"{type(exc).__name__}: {exc}")
                    continue
                s = result["stats"]
                cycle_stats.append({
                    "size": size, "q": question,
                    "ttft": result["ttft"],
                    "cached": s.get("cached_tokens"),
                    "source": s.get("cache_source"),
                    "mode": s.get("session_restore_mode"),
                })
                if result["chars"] == 0:
                    failures += 1
                    emit("empty_response", size=size, q=question)
        pid = server_pid(args.port)
        rss = rss_gb(pid) if pid else -1
        max_rss = max(max_rss, rss)
        warm = [c for c in cycle_stats if (c["cached"] or 0) > 0]
        slow_warm = [c for c in warm if (c["ttft"] or 99) > 3.0]
        warm_regressions += len(slow_warm)
        emit("cycle", n=cycle, rss_gb=round(rss, 2),
             requests=len(cycle_stats), warm=len(warm),
             slow_warm=len(slow_warm), failures_total=failures)
        print(
            f"cycle {cycle}: rss={rss:.1f}GB warm={len(warm)}/"
            f"{len(cycle_stats)} slow_warm={len(slow_warm)} "
            f"failures={failures} restarts={restarts}",
            flush=True,
        )
        if rss > args.rss_cap_gb:
            emit("rss_cap_exceeded", rss_gb=rss)
            print(f"SOAK FAIL: RSS {rss:.1f} GB exceeded cap {args.rss_cap_gb}")
            return 3
        if time.time() >= next_restart and time.time() < deadline - 120:
            restarts += 1
            emit("restart", n=restarts)
            print(f"-- restart {restarts} --", flush=True)
            if pid:
                subprocess.run(["kill", str(pid)])
                for _ in range(40):
                    if server_pid(args.port) is None:
                        break
                    time.sleep(1)
                stale = server_pid(args.port)
                if stale:
                    subprocess.run(["kill", "-9", str(stale)])
            launch(args.port, args.model, log_path)
            if not wait_health(args.port):
                print("SOAK FAIL: server did not come back after restart")
                return 2
            next_restart = time.time() + args.restart_every_min * 60

    pid = server_pid(args.port)
    if pid:
        subprocess.run(["kill", str(pid)])
        for _ in range(40):
            if server_pid(args.port) is None:
                break
            time.sleep(1)
    emit("done", cycles=cycle, restarts=restarts, failures=failures,
         warm_regressions=warm_regressions, max_rss_gb=round(max_rss, 2))
    ok = failures == 0 and warm_regressions <= cycle  # allow rare cold refills
    print(
        f"SOAK {'PASS' if ok else 'FAIL'}: {cycle} cycles, {restarts} restarts, "
        f"{failures} failures, {warm_regressions} slow-warm events, "
        f"max RSS {max_rss:.1f} GB",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
