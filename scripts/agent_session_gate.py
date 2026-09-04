#!/usr/bin/env python3
"""Agent-session gate: the coding-agent turn loop, judged from the engine's own receipts.

Every coding harness (OpenCode, Pi, Hermes, Claude Code, Cline) drives the
server the same way: one long prompt, then many short same-session turns that
append the assistant's last turn (often a tool call) plus a tool result. The
2026-09-03 founder session lost 146 s of a 14-minute OpenCode task to three
engine defects that no unit test and no single-request benchmark could see --
5-7.5 s of dead time before the first token of every warm turn, hidden
postcommit waits, and whole-turn re-prefills after tool calls -- and
"decode 21 tok/s" was the visible symptom. This gate drives that loop
directly against a serving daemon, harness-agnostically (the request shape
is the OpenAI-compatible contract every harness ends up sending), and fails
on the receipts that predicted every one of those symptoms:

  warm_dead_time      prompt_state_unattributed_time_s on a warm turn (time
                      before the first token that is neither prefill nor
                      restore) must stay under --max-dead-time-s (1.0 s)
  warm_ttft           TTFT on a warm turn with a tiny suffix must stay under
                      --max-warm-ttft-s (1.5 s)
  bank_hit            every warm turn must restore >= --min-cached-fraction
                      (0.9) of its prompt from the bank (no whole-turn
                      re-prefill after a tool call)
  postcommit_wait     no turn may wait more than --max-postcommit-wait-s
                      (2.0 s) on the previous turn's postcommit
  final_snapshot      the auto tool-call turn's generation-final snapshot must
                      be banked in O(1) (mode generation_final_*), never routed
                      to the retokenizing GPU re-prefill; a forced-choice
                      round must stay warm across the tool_choice flip
  decode_floor        decode tok/s on warm turns with >= 64 output tokens
                      must stay >= --min-decode-fraction (0.8) of the cold
                      first turn's decode, and >= --min-decode-tok-s if given
  stream_errors       no request may end with finish_reason "error"

Corpus: real source files (the installed mtplx package by default), never
word salad -- a synthetic corpus measures the context-copy lane, not decode.

Usage:
  agent_session_gate.py --base-url http://127.0.0.1:8000 [--context-tokens 40000]
  Exit 0 = pass, 1 = a gate failed, 2 = the run could not complete.
  JSON report on stdout. Run under verified max fans (the RPM is recorded).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

TOOL_NAMES = [
    "bash", "edit", "glob", "grep", "question", "read", "skill", "task",
    "todowrite", "webfetch", "write",
]

SYSTEM_PROMPT = (
    "You are opencode, an interactive CLI tool that helps users with software "
    "engineering tasks. Use the tools available to you to assist the user. Be "
    "concise. When asked to write a file, call the write tool exactly once.\n"
)


def _tools() -> list[dict[str, Any]]:
    tools = []
    for name in TOOL_NAMES:
        props: dict[str, Any] = {"arg": {"type": "string", "description": "argument"}}
        required = ["arg"]
        if name == "write":
            props = {
                "filePath": {"type": "string", "description": "absolute path"},
                "content": {"type": "string", "description": "file content"},
            }
            required = ["filePath", "content"]
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        })
    return tools


def _corpus(target_chars: int, root: str | None) -> str:
    """Real code, deterministic order, sized to the requested character count."""
    if root is None:
        try:
            import mtplx  # noqa: WPS433 -- resolve the installed package

            root = os.path.dirname(os.path.abspath(mtplx.__file__))
        except Exception:
            root = os.path.dirname(os.path.abspath(__file__))
    parts: list[str] = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
            except OSError:
                continue
            parts.append(f"\n\n# ===== {os.path.relpath(path, root)} =====\n{text}")
            total += len(text)
            if total >= target_chars:
                return "".join(parts)[:target_chars]
    return "".join(parts)[:target_chars]


class Client:
    def __init__(self, base_url: str, model: str | None, session_id: str, timeout_s: float):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.timeout_s = timeout_s
        self.model = model or self._served_model()

    def _served_model(self) -> str:
        with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=30) as response:
            data = json.load(response)
        models = data.get("data") or []
        if not models:
            raise RuntimeError("no served model")
        return str(models[0]["id"])

    def turn(self, messages: list[dict[str, Any]], *, reasoning_effort: str,
             tool_choice: Any = "auto") -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "tools": _tools(),
            "tool_choice": tool_choice,
            "reasoning_effort": reasoning_effort,
        }
        request_id = f"gate-{uuid.uuid4().hex[:10]}"
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "content-type": "application/json",
                "x-mtplx-client": "opencode",
                "x-mtplx-session-id": self.session_id,
                "x-mtplx-request-id": request_id,
            },
        )
        started = time.perf_counter()
        first_token_at: float | None = None
        content: list[str] = []
        reasoning: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        stats: dict[str, Any] = {}
        finish_reason: str | None = None
        error: Any = None
        with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except ValueError:
                    continue
                if event.get("error"):
                    error = event["error"]
                if event.get("mtplx_stats"):
                    stats = event["mtplx_stats"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    text = delta.get("content")
                    think = delta.get("reasoning_content") or delta.get("reasoning")
                    if (text or think) and first_token_at is None:
                        first_token_at = time.perf_counter() - started
                    if text:
                        content.append(text)
                    if think:
                        reasoning.append(think)
                    for call in delta.get("tool_calls") or []:
                        index = int(call.get("index") or 0)
                        slot = tool_calls.setdefault(
                            index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                        )
                        if call.get("id"):
                            slot["id"] = call["id"]
                        function = call.get("function") or {}
                        if function.get("name"):
                            slot["function"]["name"] = function["name"]
                        if function.get("arguments"):
                            slot["function"]["arguments"] += function["arguments"]
        return {
            "request_id": request_id,
            "wall_s": time.perf_counter() - started,
            "wall_ttft_s": first_token_at,
            "content": "".join(content),
            "reasoning": "".join(reasoning),
            "tool_calls": [tool_calls[k] for k in sorted(tool_calls)],
            "finish_reason": finish_reason,
            "error": error,
            "stats": stats,
        }


def _pick(stats: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "prompt_tokens", "cached_tokens", "new_prefill_tokens", "completion_tokens",
        "ttft_s", "prompt_eval_time_s", "cache_restore_time_s",
        "prompt_state_unattributed_time_s", "decode_tok_s", "sliding_decode_tok_s_last_32",
        "session_restore_mode", "cache_source", "finish_reason", "accepted_by_depth",
        "drafted_by_depth", "verify_time_s", "context_copy_accepted_tokens",
        "active_memory_bytes", "peak_memory_bytes", "producer_gaps_over_200ms",
        "stable_prefix_len", "forced_tool_choice_sentinel_injected",
    )
    out = {key: stats.get(key) for key in keys}
    wait = stats.get("postcommit_wait") or {}
    out["postcommit_wait_s"] = float(wait.get("elapsed_s") or 0.0) if isinstance(wait, dict) else 0.0
    snapshot = stats.get("session_postcommit_snapshot") or {}
    if isinstance(snapshot, dict):
        out["final_snapshot_mode"] = snapshot.get("mode")
        out["final_snapshot_stored"] = snapshot.get("stored")
        out["final_snapshot_reason"] = snapshot.get("reason")
    return out


def _probe_fan_rpm() -> int:
    import shutil
    import subprocess

    tool = shutil.which("thermalforge") or os.path.expanduser("~/.mtplx/bin/thermalforge")
    if not os.path.exists(tool):
        return 0
    try:
        out = subprocess.run([tool, "status"], capture_output=True, text=True, timeout=10, check=False).stdout
        data = json.loads(out)
        return max((int(fan.get("actual_rpm") or 0) for fan in data.get("fans") or []), default=0)
    except Exception:
        return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    session_id = f"ses_gate{uuid.uuid4().hex[:20]}"
    client = Client(args.base_url, args.model, session_id, args.timeout_s)
    corpus = _corpus(int(args.context_tokens * args.chars_per_token), args.corpus_root)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Read this source dump and reply in one sentence: which module owns "
            "session eviction?\n\n" + corpus
        )},
    ]
    turns: list[dict[str, Any]] = []

    def record(kind: str, result: dict[str, Any]) -> dict[str, Any]:
        row = {"kind": kind, "wall_s": round(result["wall_s"], 3),
               "wall_ttft_s": result["wall_ttft_s"], "error": result["error"],
               "stream_finish_reason": result["finish_reason"], **_pick(result["stats"])}
        turns.append(row)
        return row

    def append_assistant(result: dict[str, Any]) -> None:
        assistant: dict[str, Any] = {"role": "assistant", "content": result["content"]}
        if result["reasoning"]:
            assistant["reasoning_content"] = result["reasoning"]
        if result["tool_calls"]:
            assistant["tool_calls"] = result["tool_calls"]
        messages.append(assistant)

    # Turn 0: cold long prompt.
    result = client.turn(messages, reasoning_effort=args.reasoning_effort)
    record("cold", result)
    append_assistant(result)

    # Short warm turns: the tool-result cadence.
    follow_ups = [
        "One sentence: name one public method of that module.",
        "One sentence: what does longest_prefix do?",
        "One word: which module has _run_loop?",
    ]
    for text in follow_ups[: args.warm_turns]:
        messages.append({"role": "user", "content": text})
        result = client.turn(messages, reasoning_effort=args.reasoning_effort)
        record("warm", result)
        append_assistant(result)

    # Two tool rounds, each followed by its tool result.
    #  - auto: the model chooses the call (every harness's normal round). Its
    #    generation-final snapshot must bank in O(1): parse -> re-render is
    #    not byte-stable for tool arguments, so this is where the committed
    #    body substitution is proven.
    #  - forced: tool_choice pins the function. The instruction travels as a
    #    transient trailing sentinel, so the session prefix (and its bank
    #    identity) must survive the flip; the snapshot check is waived (the
    #    generated KV sits after bytes the next turn will not resend).
    rounds = [
        ("auto", "auto", (
            "Use the write tool to create /tmp/mtplx-gate/hello.html containing a "
            "minimal HTML page with a <title>, one <h1>, and a trailing newline. "
            "Call the tool now; do not answer in prose."
        )),
        ("forced", {"type": "function", "function": {"name": "write"}}, (
            "Now use the write tool to create /tmp/mtplx-gate/second.html with a "
            "one-line <p> paragraph and a trailing newline."
        )),
    ]
    for label, tool_choice, text in rounds:
        messages.append({"role": "user", "content": text})
        result = client.turn(messages, reasoning_effort=args.reasoning_effort, tool_choice=tool_choice)
        tool_row = record(f"tool_call_{label}", result)
        append_assistant(result)
        if result["tool_calls"]:
            call = result["tool_calls"][0]
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id") or "call_gate",
                "content": "File written (1 file, 12 lines).",
            })
            tool_row["tool_call_emitted"] = True
            result = client.turn(messages, reasoning_effort=args.reasoning_effort)
            record(f"tool_result_{label}", result)
            append_assistant(result)
        else:
            tool_row["tool_call_emitted"] = False

    return {"session_id": session_id, "model": client.model, "turns": turns}


def judge(report: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    turns = report["turns"]
    failures: list[dict[str, Any]] = []
    cold = next((t for t in turns if t["kind"] == "cold"), None)
    warm = [t for t in turns if t["kind"] != "cold"]

    for turn in turns:
        if turn.get("error") or turn.get("stream_finish_reason") == "error":
            failures.append({"gate": "stream_errors", "turn": turn["kind"], "error": turn.get("error")})
        wait_s = float(turn.get("postcommit_wait_s") or 0.0)
        if wait_s > args.max_postcommit_wait_s:
            failures.append({"gate": "postcommit_wait", "turn": turn["kind"], "wait_s": wait_s})
    for turn in warm:
        dead = float(turn.get("prompt_state_unattributed_time_s") or 0.0)
        if dead > args.max_dead_time_s:
            failures.append({"gate": "warm_dead_time", "turn": turn["kind"], "dead_time_s": dead})
        prompt = int(turn.get("prompt_tokens") or 0)
        cached = int(turn.get("cached_tokens") or 0)
        if prompt and cached < args.min_cached_fraction * prompt:
            failures.append({"gate": "bank_hit", "turn": turn["kind"], "prompt_tokens": prompt, "cached_tokens": cached})
        new = int(turn.get("new_prefill_tokens") or 0)
        ttft = float(turn.get("ttft_s") or 0.0)
        if new <= args.small_suffix_tokens and ttft > args.max_warm_ttft_s:
            failures.append({"gate": "warm_ttft", "turn": turn["kind"], "ttft_s": ttft, "new_prefill_tokens": new})
    tool_turn = next((t for t in turns if t["kind"] == "tool_call_auto"), None)
    if tool_turn is not None and tool_turn.get("tool_call_emitted"):
        mode = str(tool_turn.get("final_snapshot_mode") or "")
        if not mode.startswith("generation_final"):
            failures.append({"gate": "final_snapshot", "turn": "tool_call_auto", "mode": mode,
                             "reason": tool_turn.get("final_snapshot_reason")})
    if cold is not None:
        cold_decode = float(cold.get("decode_tok_s") or 0.0)
        for turn in warm:
            out = int(turn.get("completion_tokens") or 0)
            decode = float(turn.get("decode_tok_s") or 0.0)
            if out < 64:
                continue
            floor = max(args.min_decode_tok_s, args.min_decode_fraction * cold_decode)
            if decode < floor:
                failures.append({"gate": "decode_floor", "turn": turn["kind"], "decode_tok_s": decode,
                                 "floor_tok_s": round(floor, 1), "completion_tokens": out})
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--context-tokens", type=int, default=40_000)
    parser.add_argument("--chars-per-token", type=float, default=3.9)
    parser.add_argument("--corpus-root", default=None)
    parser.add_argument("--warm-turns", type=int, default=3)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--max-dead-time-s", type=float, default=1.0)
    parser.add_argument("--max-warm-ttft-s", type=float, default=1.5)
    parser.add_argument("--small-suffix-tokens", type=int, default=512)
    parser.add_argument("--min-cached-fraction", type=float, default=0.9)
    parser.add_argument("--max-postcommit-wait-s", type=float, default=2.0)
    parser.add_argument("--min-decode-fraction", type=float, default=0.8)
    parser.add_argument("--min-decode-tok-s", type=float, default=0.0)
    parser.add_argument("--fan-rpm-verified", type=int, default=0)
    args = parser.parse_args(argv)

    fan_rpm = args.fan_rpm_verified or _probe_fan_rpm()
    try:
        report = run(args)
    except (urllib.error.URLError, OSError, RuntimeError) as exc:
        print(json.dumps({"pass": False, "error": repr(exc)}, indent=1))
        return 2
    failures = judge(report, args)
    report.update({
        "fan_rpm_verified": fan_rpm,
        "thresholds": {
            "max_dead_time_s": args.max_dead_time_s,
            "max_warm_ttft_s": args.max_warm_ttft_s,
            "min_cached_fraction": args.min_cached_fraction,
            "max_postcommit_wait_s": args.max_postcommit_wait_s,
            "min_decode_fraction": args.min_decode_fraction,
            "min_decode_tok_s": args.min_decode_tok_s,
        },
        "failures": failures,
        "pass": not failures,
    })
    print(json.dumps(report, indent=1, default=str))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
