#!/usr/bin/env python3
"""Poll /v1/mtplx/snapshot and append every NEW request row to a JSONL file.

Keys off (prompt_tokens, completion_tokens, decode_elapsed_s) tuples in the
`recent` ring to avoid duplicates. Run alongside a real client session.
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
OUT = sys.argv[2]

KEEP = [
    "decode_tok_s", "completion_tokens", "prompt_tokens", "cached_tokens",
    "new_prefill_tokens", "verify_calls", "accepted_by_depth", "drafted_by_depth",
    "bonus_tokens", "correction_tokens", "verify_time_s", "draft_time_s",
    "verify_forward_time_s", "verify_hidden_eval_time_s",
    "verify_logits_eval_time_s", "ttft_s", "decode_elapsed_s",
    "prompt_eval_time_s", "request_elapsed_s", "finish_reason",
    "mtp_history_policy", "session_restore_mode", "cache_source",
    "generation_mode", "request_enable_thinking",
    "sliding_decode_tok_s_first_32", "sliding_decode_tok_s_last_32",
    "mean_accept_probability_by_depth",
]

seen = set()
out = open(OUT, "a")
print(f"tailing snapshot :{PORT} -> {OUT}", flush=True)
while True:
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/v1/mtplx/snapshot", timeout=5) as r:
            recent = json.loads(r.read().decode()).get("recent") or []
    except Exception:
        time.sleep(1)
        continue
    for row in recent:
        key = (
            row.get("prompt_tokens"), row.get("completion_tokens"),
            round(row.get("decode_elapsed_s") or 0, 4),
            round(row.get("ttft_s") or 0, 5),
        )
        if key in seen:
            continue
        seen.add(key)
        slim = {k: row.get(k) for k in KEEP}
        vc = slim.get("verify_calls") or 0
        comp = slim.get("completion_tokens") or 0
        if vc:
            slim["tokens_per_verify"] = round(comp / vc, 3)
            slim["verify_ms_per_call"] = round(
                1000 * (slim.get("verify_time_s") or 0) / vc, 2)
        out.write(json.dumps(slim) + "\n")
        out.flush()
        print(json.dumps({k: slim.get(k) for k in (
            "prompt_tokens", "completion_tokens", "decode_tok_s",
            "tokens_per_verify", "verify_ms_per_call", "ttft_s")}), flush=True)
    time.sleep(0.7)
