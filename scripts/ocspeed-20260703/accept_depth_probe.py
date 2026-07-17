#!/usr/bin/env python3
"""Decompose the OpenCode decode gap: acceptance-at-depth vs content vs fixed cost.

Fires controlled shapes at ONE daemon (caller launches it) and dumps the full
verify decomposition from /v1/mtplx/snapshot latest as JSONL rows.

Cells (selected by --set):
  depth    same haiku task at 0/2k/7k/12k lorem padding, + rules/code padding at 7k,
           + code-gen task at 0/7k, + OC-round shape (think, short) at 7k
  fixed    completion-length sweep (48/160/512) at 7k lorem, warm session
  quick    bare + lorem7k + ocrules7k haiku only (for A/B daemon relaunches)
"""
import argparse
import json
import time
import urllib.request

MODEL = "mtplx-qwen36-27b-optimized-speed"

# ---------------------------------------------------------------- padding ---

_LOREM_SENTENCES = [
    "The harbour master logged {i} arrivals before the fog lifted over the quay.",
    "Every ledger entry from voyage {i} was copied twice into the archive book.",
    "A crate of navigational charts numbered {i} waited beside the customs shed.",
    "The lighthouse keeper recorded wind speed {i} knots at the third watch.",
    "Merchants from the northern route traded {i} bolts of dyed cloth that week.",
    "Repairs to pier {i} continued despite the spring tide warnings.",
    "The clerk stamped manifest {i} and filed it under the winter season.",
    "Sailors recalled storm {i} as the roughest passage of the decade.",
]

_RULE_TEMPLATE = (
    "Rule {i}: Be precise, minimal, and verify every edit against the file state "
    "before and after. Never fabricate file contents; always read before writing. "
    "Prefer small diffs. Preserve style. Use ripgrep for search. "
)

_CODE_TEMPLATE = '''export function computeStage{i}(input: StageInput): StageResult {{
  const normalized = input.values.map((v) => v * {i} + OFFSET_TABLE[{i} % 7]);
  const total = normalized.reduce((acc, v) => acc + v, 0);
  if (total > THRESHOLDS.stage{i}) {{
    logger.warn("stage {i} exceeded threshold", {{ total }});
    return {{ ok: false, stage: {i}, total }};
  }}
  return {{ ok: true, stage: {i}, total: Math.round(total * 100) / 100 }};
}}

'''


def make_padding(kind: str, approx_tokens: int) -> str:
    if approx_tokens <= 0:
        return ""
    parts: list[str] = []
    total_chars = 0
    target_chars = approx_tokens * 4  # rough calibration, recorded via usage
    i = 0
    while total_chars < target_chars:
        i += 1
        if kind == "lorem":
            chunk = _LOREM_SENTENCES[i % len(_LOREM_SENTENCES)].format(i=i) + " "
        elif kind == "rules":
            chunk = _RULE_TEMPLATE.format(i=i)
        elif kind == "code":
            chunk = _CODE_TEMPLATE.format(i=i)
        else:
            raise ValueError(kind)
        parts.append(chunk)
        total_chars += len(chunk)
    header = {
        "lorem": "Background archive material for reference:\n\n",
        "rules": "You are a coding assistant working inside a repository. Follow the rules exactly.\n\n",
        "code": "Repository source files currently loaded for reference:\n\n```ts\n",
    }[kind]
    tail = "\n```\n" if kind == "code" else ""
    return header + "".join(parts) + tail


HAIKU = "Write six haikus about the sea, numbered."
CODEGEN = (
    "Write a Python function `column_sums(path)` that parses a CSV file and "
    "returns a dict of column name to numeric sum. Include type hints and a "
    "docstring. Then show one usage example."
)
GREET = "how are you"

SNAP_KEYS = [
    "decode_tok_s", "completion_tokens", "decode_elapsed_s", "verify_calls",
    "accepted_by_depth", "drafted_by_depth", "bonus_tokens", "correction_tokens",
    "verify_time_s", "draft_time_s", "accept_time_s", "repair_time_s",
    "verify_forward_time_s", "verify_eval_time_s", "verify_logits_eval_time_s",
    "verify_hidden_eval_time_s", "verify_target_distribution_time_s",
    "snapshot_time_s", "commit_time_s", "capture_commit_time_s", "rollback_time_s",
    "prompt_tokens", "cached_tokens", "new_prefill_tokens", "ttft_s",
    "prompt_eval_time_s", "prompt_mtp_history_time_s", "cache_restore_time_s",
    "mtp_history_policy", "mtp_history_window_tokens",
    "mean_accept_probability_by_depth", "generation_mode",
    "sliding_decode_tok_s_first_32", "sliding_decode_tok_s_last_32",
]


def fire(port: int, label: str, messages, *, max_tokens: int, thinking: bool,
         session: str, out):
    body = {
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.6, "top_p": 0.95, "stream": True,
        "enable_thinking": thinking,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "X-MTPLX-Session-ID": session})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        for _ in r:
            pass
    wall = time.perf_counter() - t0
    with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/mtplx/snapshot", timeout=10) as r:
        lat = json.loads(r.read().decode()).get("latest") or {}
    row = {"cell": label, "wall_s": round(wall, 2), "session": session}
    for key in SNAP_KEYS:
        row[key] = lat.get(key)
    vc = row.get("verify_calls") or 0
    comp = row.get("completion_tokens") or 0
    if vc:
        row["tokens_per_verify"] = round(comp / vc, 3)
        row["verify_ms_per_call"] = round(1000 * (row.get("verify_time_s") or 0) / vc, 2)
        row["draft_ms_per_call"] = round(1000 * (row.get("draft_time_s") or 0) / vc, 2)
    abd = row.get("accepted_by_depth") or []
    if vc and abd:
        row["accept_rate_by_pos"] = [round(a / vc, 3) for a in abd]
    print(json.dumps(row), flush=True)
    out.write(json.dumps(row) + "\n")
    out.flush()
    time.sleep(1.5)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--set", default="depth",
                    choices=["depth", "fixed", "quick", "deep"])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pads = {}

    def cell_messages(pad_kind: str, pad_tokens: int, task: str):
        if pad_tokens:
            key = (pad_kind, pad_tokens)
            if key not in pads:
                pads[key] = make_padding(pad_kind, pad_tokens)
            return [{"role": "system", "content": pads[key]},
                    {"role": "user", "content": task}]
        return [{"role": "user", "content": task}]

    out = open(args.output, "a")
    tag = args.tag

    if args.set == "depth":
        plan = [
            ("bare_haiku", "lorem", 0, HAIKU, 160, False, 3),
            ("lorem2k_haiku", "lorem", 2000, HAIKU, 160, False, 2),
            ("lorem7k_haiku", "lorem", 7000, HAIKU, 160, False, 3),
            ("lorem12k_haiku", "lorem", 12000, HAIKU, 160, False, 2),
            ("rules7k_haiku", "rules", 7000, HAIKU, 160, False, 3),
            ("code7k_haiku", "code", 7000, HAIKU, 160, False, 2),
            ("bare_codegen", "lorem", 0, CODEGEN, 200, False, 2),
            ("lorem7k_codegen", "lorem", 7000, CODEGEN, 200, False, 2),
            ("rules7k_greet_think", "rules", 7000, GREET, 256, True, 2),
        ]
    elif args.set == "quick":
        plan = [
            ("bare_haiku", "lorem", 0, HAIKU, 160, False, 3),
            ("lorem7k_haiku", "lorem", 7000, HAIKU, 160, False, 3),
            ("rules7k_haiku", "rules", 7000, HAIKU, 160, False, 3),
        ]
    elif args.set == "deep":
        plan = [
            ("lorem16k_haiku", "lorem", 16000, HAIKU, 160, False, 2),
            ("lorem24k_codegen", "lorem", 24000, CODEGEN, 200, False, 2),
            ("lorem30k_haiku", "lorem", 30000, HAIKU, 160, False, 2),
        ]
    else:  # fixed
        plan = [
            ("lorem7k_len48", "lorem", 7000, HAIKU, 48, False, 3),
            ("lorem7k_len160", "lorem", 7000, HAIKU, 160, False, 3),
            ("lorem7k_len512", "lorem", 7000, HAIKU, 512, False, 3),
            ("bare_len48", "lorem", 0, HAIKU, 48, False, 3),
            ("bare_len512", "lorem", 0, HAIKU, 512, False, 3),
        ]

    for label, kind, tokens, task, max_tokens, thinking, repeats in plan:
        for rep in range(repeats):
            session = f"ocspeed-{tag}-{label}"
            fire(args.port, f"{label}#r{rep}", cell_messages(kind, tokens, task),
                 max_tokens=max_tokens, thinking=thinking,
                 session=session, out=out)
    print("PROBE DONE", flush=True)
    out.close()


if __name__ == "__main__":
    main()
