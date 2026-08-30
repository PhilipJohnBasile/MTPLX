"""One instrumented decode cell for Pulse X-Ray: warm 91K turn on Flash-Next.

Drives the serve half only — the operator wraps it with pulsectl capture
start/stop (see the pulse-xray SKILL). Serve on :8399, salted 91K code
prompt, prefill+store turn, then ONE warm 384-token follow-up whose start
and end the operator marks. Fan pinning, die-temp gating, and teardown
stay with the caller so pulse's own gate is the authority.

Usage:
  .venv/bin/python scripts/decodecliff_pulse_cell_20260830.py serve   # boots serve, blocks
  .venv/bin/python scripts/decodecliff_pulse_cell_20260830.py prefill # turn 1
  .venv/bin/python scripts/decodecliff_pulse_cell_20260830.py turn    # warm measured turn
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8399"
STATE = ROOT / "outputs" / "decodecliff-probe-20260830" / "pulse-cell-messages.json"
TARGET_PROMPT_CHARS = 350_000


def http_json(method: str, url: str, payload: dict | None, timeout: float) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def build_prompt() -> str:
    parts: list[str] = []
    total = 0
    for rel in ("mtplx", "mtplx/kernels", "mtplx/models"):
        for f in sorted((ROOT / rel).glob("*.py")):
            text = f.read_text(errors="ignore")
            parts.append(f"\n# ==== file: {f.relative_to(ROOT)} ====\n{text}")
            total += len(text)
            if total >= TARGET_PROMPT_CHARS:
                break
        if total >= TARGET_PROMPT_CHARS:
            break
    dump = "".join(parts)[:TARGET_PROMPT_CHARS]
    return (
        "# probe-salt pulse-cell-20260830\n"
        "Below is a dump of the MTPLX codebase. Study it carefully; in "
        "follow-up turns you will continue implementations in its exact "
        "style.\n" + dump
    )


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "turn"
    if mode == "prefill":
        messages = [{"role": "user", "content": build_prompt()}]
        t0 = time.time()
        r = http_json(
            "POST",
            f"{BASE}/v1/chat/completions",
            {
                "model": "mtplx-flash-next-optimized-speed",
                "messages": messages,
                "max_tokens": 16,
                "stream": False,
            },
            timeout=1200,
        )
        messages.append(
            {"role": "assistant", "content": r["choices"][0]["message"]["content"] or "ok"}
        )
        STATE.write_text(json.dumps(messages))
        print(f"prefill done in {time.time()-t0:.0f}s; messages staged")
        return 0
    if mode == "turn":
        messages = json.loads(STATE.read_text())
        messages.append(
            {
                "role": "user",
                "content": "Now write ~60 lines of a clean paged-KV allocator "
                "for this codebase, matching its style. Code only.",
            }
        )
        t0 = time.time()
        r = http_json(
            "POST",
            f"{BASE}/v1/chat/completions",
            {
                "model": "mtplx-flash-next-optimized-speed",
                "messages": messages,
                "max_tokens": 384,
                "stream": False,
            },
            timeout=600,
        )
        print(f"turn done in {time.time()-t0:.0f}s")
        return 0
    print(f"unknown mode {mode}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
