#!/usr/bin/env python3
"""Warm prefix-reuse probe (kvcache-v2, 2026-07-03).

Reproduces the reviewer's RAG scenario: a large repeated context block with a
short varying question tail, measured cold and warm across cache paths.

Variants (run in the order given by --variants):
  cold     fresh session, block + Q1                     (baseline prefill)
  a        identical repeat, same session                (exact-hit path)
  b_rag    same block + different question, same session (mutated-tail path)
  b_agent  transcript append (cold turn + short follow), same session
  c        same block + different question, NEW session  (cross-session path)

Restart-warm (plan variant d) is orchestrated externally: run `--variants cold`,
restart the daemon, then run `--variants a --session-id <same>` — the block is
deterministic for a given --target-tokens, so prefixes match across invocations.
Sizes (plan variant e) are `--target-tokens 12000` / `48000` runs.

Flavors:
  mtplx    OpenAI-compatible SSE + mtplx_stats + /v1/mtplx/snapshot
  ollama   native /api/chat stream (prompt_eval_count/duration telemetry)

Works against current main, the public v1.0.4 build, and Ollama so the same
shapes are comparable engine-to-engine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTIONS = {
    "q1": "Summarize the three biggest risks in this repo state.",
    "q2": "Which module should we refactor first and why? Answer briefly.",
    "q3": "List the files most likely to contain the aiming bug.",
}

APPROX_TOKENS_PER_LINE = 40


def context_block(target_tokens: int) -> str:
    lines = max(8, target_tokens // APPROX_TOKENS_PER_LINE)
    rows = []
    for index in range(lines):
        rows.append(
            "repo-file-{index:05d}: src/game/system_{bucket}/module_{feature}.ts "
            "contains camera, WASD movement, bow aiming, terrain props, destructible "
            "environment state, and TypeScript strict errors. Keep identifiers stable.".format(
                index=index, bucket=index % 113, feature=index % 47
            )
        )
    return "\n".join(rows)


def rag_prompt(target_tokens: int, question_key: str) -> str:
    return (
        "You are a coding agent reviewing a large TypeScript game project. "
        "Here is the repository context:\n\n"
        + context_block(target_tokens)
        + "\n\nQuestion: "
        + QUESTIONS[question_key]
    )


def http_json(method: str, url: str, *, timeout_s: float = 20.0) -> dict[str, Any]:
    try:
        request = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}
    except Exception as exc:  # noqa: BLE001 - probe must not die on telemetry
        return {"probe_error": f"{type(exc).__name__}: {exc}"}


def fan_state() -> dict[str, Any]:
    try:
        from mtplx.thermal import fan_summary

        summary = fan_summary()
        fans = summary.get("fans") or []
        ramped = bool(fans) and all(
            f.get("actual_rpm") and f.get("actual_rpm") >= 7000 for f in fans
        )
        return {"ok": bool(summary.get("ok")), "ramped": ramped, "fans": fans}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "ramped": False, "error": f"{type(exc).__name__}: {exc}"}


class Jsonl:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, payload: dict[str, Any]) -> None:
        record = {"ts": time.time(), "event": event, "payload": payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def stream_mtplx(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    session_id: str,
    max_tokens: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": False,
        "metadata": {"client": "kvcache_warm_probe", "session_id": session_id},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-MTPLX-Session-ID": session_id,
        },
    )
    started = time.perf_counter()
    ttft = None
    text_parts: list[str] = []
    stats: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            payload = json.loads(data)
            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}
                for key in ("reasoning_content", "content"):
                    value = delta.get(key)
                    if isinstance(value, str) and value:
                        text_parts.append(value)
                        if ttft is None:
                            ttft = time.perf_counter() - started
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
            if isinstance(payload.get("mtplx_stats"), dict):
                stats = payload["mtplx_stats"]
    wall = time.perf_counter() - started
    return {
        "ttft_s": ttft,
        "wall_s": wall,
        "text_chars": len("".join(text_parts)),
        "text_tail": "".join(text_parts)[-400:],
        "usage": usage,
        "stats": stats,
    }


def stream_ollama(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    session_id: str,  # unused; ollama has no session concept
    max_tokens: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_predict": max_tokens, "temperature": 0.0},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    ttft = None
    text_parts: list[str] = []
    final: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            payload = json.loads(line)
            content = ((payload.get("message") or {}).get("content")) or ""
            if content:
                text_parts.append(content)
                if ttft is None:
                    ttft = time.perf_counter() - started
            if payload.get("done"):
                final = payload
    wall = time.perf_counter() - started
    nanos = 1e9
    stats = {
        "load_duration_s": (final.get("load_duration") or 0) / nanos,
        "prompt_eval_count": final.get("prompt_eval_count"),
        "prompt_eval_s": (final.get("prompt_eval_duration") or 0) / nanos,
        "eval_count": final.get("eval_count"),
        "eval_s": (final.get("eval_duration") or 0) / nanos,
        "total_s": (final.get("total_duration") or 0) / nanos,
    }
    if stats["prompt_eval_s"] and stats["prompt_eval_count"]:
        stats["prompt_tok_s"] = stats["prompt_eval_count"] / stats["prompt_eval_s"]
    if stats["eval_s"] and stats["eval_count"]:
        stats["decode_tok_s"] = stats["eval_count"] / stats["eval_s"]
    return {
        "ttft_s": ttft,
        "wall_s": wall,
        "text_chars": len("".join(text_parts)),
        "text_tail": "".join(text_parts)[-400:],
        "usage": {
            "prompt_tokens": final.get("prompt_eval_count"),
            "completion_tokens": final.get("eval_count"),
        },
        "stats": stats,
    }


def build_variant_messages(
    variant: str, target_tokens: int, cold_tail: str
) -> tuple[list[dict[str, str]], str]:
    """Returns (messages, session_suffix). session_suffix '' = primary session."""
    if variant in {"cold", "a"}:
        return [{"role": "user", "content": rag_prompt(target_tokens, "q1")}], ""
    if variant == "b_rag":
        return [{"role": "user", "content": rag_prompt(target_tokens, "q2")}], ""
    if variant == "b_agent":
        return (
            [
                {"role": "user", "content": rag_prompt(target_tokens, "q1")},
                {"role": "assistant", "content": cold_tail or "Understood."},
                {"role": "user", "content": QUESTIONS["q3"]},
            ],
            "",
        )
    if variant == "c":
        return [{"role": "user", "content": rag_prompt(target_tokens, "q3")}], "-xsession"
    raise ValueError(f"unknown variant: {variant}")


CURATED_KEYS = (
    "cached_tokens",
    "new_prefill_tokens",
    "cache_source",
    "session_restore_mode",
    "cache_miss_reason",
    "session_cache_hit",
    "ssd_cache_hit",
    "ssd_cached_tokens",
    "ssd_restore_s",
    "prompt_target_prefill_time_s",
    "prompt_target_prefill_tok_s",
    "queue_wait_s",
    "prompt_tok_s",
    "decode_tok_s",
    "prompt_eval_count",
    "prompt_eval_s",
    "load_duration_s",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--flavor", choices=["mtplx", "ollama"], default="mtplx")
    parser.add_argument("--target-tokens", type=int, default=4000)
    parser.add_argument(
        "--variants", default="cold,a,b_rag,b_agent,c",
        help="comma list from: cold,a,b_rag,b_agent,c",
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--pause-s", type=float, default=1.5)
    parser.add_argument("--label", default="")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--require-max-fans", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)

    writer = Jsonl(Path(args.output_jsonl).expanduser().resolve())
    session_id = args.session_id or f"kvprobe-{args.target_tokens}-{int(time.time())}"
    fans = fan_state()
    writer.write("run_start", {"args": vars(args), "session_id": session_id, "fans": fans})
    if args.require_max_fans and not fans.get("ramped"):
        print(f"FANS NOT RAMPED — aborting (fans={fans})", file=sys.stderr)
        writer.write("run_failed", {"reason": "fans_not_ramped"})
        return 2

    stream = stream_mtplx if args.flavor == "mtplx" else stream_ollama
    if args.flavor == "mtplx":
        writer.write("snapshot_before", http_json("GET", args.base_url + "/v1/mtplx/snapshot"))

    cold_tail = ""
    rows: list[dict[str, Any]] = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        messages, suffix = build_variant_messages(variant, args.target_tokens, cold_tail)
        sid = session_id + suffix
        result = stream(args.base_url, args.model, messages, sid, args.max_tokens)
        if variant == "cold":
            cold_tail = result.get("text_tail") or ""
        curated = {k: result["stats"].get(k) for k in CURATED_KEYS if k in result["stats"]}
        row = {
            "label": args.label,
            "variant": variant,
            "target_tokens": args.target_tokens,
            "session_id": sid,
            "ttft_s": result["ttft_s"],
            "wall_s": result["wall_s"],
            "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens"),
            **curated,
        }
        rows.append(row)
        writer.write("request", {**row, "stats_full": result["stats"], "usage": result["usage"]})
        print(json.dumps(row, default=str))
        time.sleep(args.pause_s)

    if args.flavor == "mtplx":
        writer.write("snapshot_after", http_json("GET", args.base_url + "/v1/mtplx/snapshot"))
    writer.write("fan_state_after", fan_state())

    print("\n=== SUMMARY ===")
    header = f"{'variant':10} {'ttft_s':>8} {'cached':>7} {'newpf':>7} {'source':>8} {'mode':>10} {'miss':>18}"
    print(header)
    for row in rows:
        print(
            f"{row['variant']:10} "
            f"{(row.get('ttft_s') or 0):8.3f} "
            f"{str(row.get('cached_tokens', '-')):>7} "
            f"{str(row.get('new_prefill_tokens', '-')):>7} "
            f"{str(row.get('cache_source', '-')):>8} "
            f"{str(row.get('session_restore_mode', '-')):>10} "
            f"{str(row.get('cache_miss_reason', '-'))[:18]:>18}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
