#!/bin/zsh
# Position-matched alternating longgen pairs: B(candidate) A(eager) B A.
set -u
ROOT="/Users/youssof/Projects/MTPLX-release/mtplx-ocspeed-20260703"
PORT=18170
cd "$ROOT"

teardown() {
  local pid
  pid=$(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t 2>/dev/null | head -1) || true
  if [ -n "$pid" ]; then
    kill "$pid" || true
    for i in {1..40}; do
      if ! lsof -nP -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then break; fi
      sleep 1
    done
  fi
}

one_long() {
  local arm="$1"; shift
  teardown
  env MTPLX_NAX_VERIFY=1 MTPLX_NAX_M4_IMPL=vk_k "$@" \
    .venv/bin/mtplx serve --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed \
    --port $PORT --warmup-tokens 16 >> "outputs/ocspeed-20260703/paired-$arm.log" 2>&1 &
  for i in {1..90}; do
    curl -s -m 2 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q '"id"' && break
    sleep 3
  done
  .venv/bin/python - "$PORT" "$arm" <<'PYEOF'
import json, sys, time, urllib.request
sys.path.insert(0, "scripts/ocspeed-20260703")
from accept_depth_probe import make_padding
port, arm = int(sys.argv[1]), sys.argv[2]
body = {"model": "mtplx-qwen36-27b-optimized-speed",
        "messages": [{"role": "system", "content": make_padding("lorem", 7000)},
                     {"role": "user", "content": "Write a detailed design document for a 2D physics game engine: architecture, collision system, integration loop, memory layout, and a full example. Be thorough and keep going."}],
        "max_tokens": 1600, "temperature": 0.6, "top_p": 0.95, "stream": True,
        "enable_thinking": False}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(body).encode(), method="POST",
    headers={"Content-Type": "application/json", "X-MTPLX-Session-ID": f"paired-{arm}-{time.time()}"})
with urllib.request.urlopen(req, timeout=900) as r:
    for _ in r: pass
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/mtplx/snapshot", timeout=10) as r:
    lat = json.loads(r.read().decode()).get("latest") or {}
row = {"arm": arm, "decode": round(lat.get("decode_tok_s") or 0, 2),
       "vms": round(1000*(lat.get("verify_time_s") or 0)/max(1, lat.get("verify_calls") or 1), 2)}
print(json.dumps(row), flush=True)
with open("outputs/ocspeed-20260703/paired_longgen.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
PYEOF
  teardown
  sleep 60
}

one_long B MTPLX_COMPILED_VERIFY=1 MTPLX_COMPILED_VERIFY_MAX_CONTEXT=12288
one_long A
one_long B MTPLX_COMPILED_VERIFY=1 MTPLX_COMPILED_VERIFY_MAX_CONTEXT=12288
one_long A
echo PAIRED DONE
