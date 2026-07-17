#!/bin/zsh
# Alternated A/B for the W2 agent-band candidate vs eager baseline.
# Each pass: launch daemon -> quick probe (agent shapes) + 7k longgen cell -> teardown.
set -e
ROOT="/Users/youssof/Projects/MTPLX-release/mtplx-ocspeed-20260703"
PORT=18170
PASSES=${1:-3}
cd "$ROOT"

fan_gate() {
  RPM=$(sudo -n ~/.mtplx/bin/thermalforge status | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(min(f['actual_rpm'] for f in d['fans']))")
  if [ "$RPM" -lt 7000 ]; then
    sudo -n ~/.mtplx/bin/thermalforge max; sleep 8
    RPM=$(sudo -n ~/.mtplx/bin/thermalforge status | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(min(f['actual_rpm'] for f in d['fans']))")
    [ "$RPM" -lt 7000 ] && { echo "fan ramp failed"; exit 1; }
  fi
  echo "fans ok ($RPM)"
}

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
  if lsof -nP -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "port stuck"; exit 1
  fi
}

run_arm() {
  local arm="$1"; shift
  local pass="$1"; shift
  teardown
  fan_gate
  env MTPLX_NAX_VERIFY=1 MTPLX_NAX_M4_IMPL=vk_k "$@" \
    .venv/bin/mtplx serve --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed \
    --port $PORT --warmup-tokens 16 >> "outputs/ocspeed-20260703/abba-serve-${arm}-p${pass}.log" 2>&1 &
  for i in {1..90}; do
    curl -s -m 2 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q '"id"' && break
    sleep 3
  done
  curl -s -m 2 http://127.0.0.1:$PORT/v1/models | grep -q '"id"' || { echo "never ready"; exit 1; }
  .venv/bin/python scripts/ocspeed-20260703/accept_depth_probe.py $PORT --set quick \
    --tag "abba-${arm}-p${pass}" --output "outputs/ocspeed-20260703/abba_${arm}.jsonl" > /dev/null
  .venv/bin/python - "$PORT" "$arm" "$pass" <<'PYEOF'
import json, sys, time, urllib.request
sys.path.insert(0, "scripts/ocspeed-20260703")
from accept_depth_probe import make_padding
port, arm, p = int(sys.argv[1]), sys.argv[2], sys.argv[3]
body = {"model": "mtplx-qwen36-27b-optimized-speed",
        "messages": [{"role": "system", "content": make_padding("lorem", 7000)},
                     {"role": "user", "content": "Write a detailed design document for a 2D physics game engine: architecture, collision system, integration loop, memory layout, and a full example. Be thorough and keep going."}],
        "max_tokens": 1600, "temperature": 0.6, "top_p": 0.95, "stream": True,
        "enable_thinking": False}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(body).encode(), method="POST",
    headers={"Content-Type": "application/json", "X-MTPLX-Session-ID": f"abba-{arm}-{p}-long"})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=900) as r:
    for _ in r: pass
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/mtplx/snapshot", timeout=10) as r:
    lat = json.loads(r.read().decode()).get("latest") or {}
row = {"cell": "longgen7k", "arm": arm, "pass": p,
       "decode_tok_s": round(lat.get("decode_tok_s") or 0, 2),
       "verify_ms": round(1000*(lat.get("verify_time_s") or 0)/max(1, lat.get("verify_calls") or 1), 2),
       "completion": lat.get("completion_tokens")}
print(json.dumps(row))
with open(f"outputs/ocspeed-20260703/abba_{arm}.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
PYEOF
  teardown
}

for p in $(seq 1 $PASSES); do
  echo "=== pass $p arm A (w2off) ==="
  run_arm w2off "$p"
  echo "=== pass $p arm B (w2agent) ==="
  run_arm w2agent "$p" MTPLX_COMPILED_VERIFY=1 MTPLX_COMPILED_VERIFY_MAX_CONTEXT=12288
done
echo "ABBA DONE"
