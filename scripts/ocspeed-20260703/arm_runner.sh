#!/bin/zsh
# Relaunch the 18170 daemon with an env arm and run the quick depth probe.
# Usage: arm_runner.sh <arm_name> [EXTRA_ENV...]
set -e
ROOT="/Users/youssof/Projects/MTPLX-release/mtplx-ocspeed-20260703"
PORT=18170
ARM="$1"; shift
OUT="$ROOT/outputs/ocspeed-20260703/arm_${ARM}.jsonl"
LOG="$ROOT/outputs/ocspeed-20260703/serve-arm-${ARM}.log"

# teardown by port
PID=$(lsof -nP -iTCP:$PORT -sTCP:LISTEN -t 2>/dev/null | head -1)
if [ -n "$PID" ]; then kill "$PID"; for i in {1..30}; do lsof -nP -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1 || break; sleep 1; done; fi
lsof -nP -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1 && { echo "port still held"; exit 1; }

# fan gate
RPM=$(sudo -n ~/.mtplx/bin/thermalforge status | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(min(f['actual_rpm'] for f in d['fans']))")
if [ "$RPM" -lt 7000 ]; then
  sudo -n ~/.mtplx/bin/thermalforge max; sleep 8
  RPM=$(sudo -n ~/.mtplx/bin/thermalforge status | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(min(f['actual_rpm'] for f in d['fans']))")
  [ "$RPM" -lt 7000 ] && { echo "fan ramp failed"; exit 1; }
fi
echo "fans ok ($RPM rpm), launching arm=$ARM env: $*"

cd "$ROOT"
env MTPLX_NAX_VERIFY=1 MTPLX_NAX_M4_IMPL=vk_k "$@" \
  .venv/bin/mtplx serve --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed \
  --port $PORT --warmup-tokens 16 >> "$LOG" 2>&1 &
SERVER_PID=$!
for i in {1..90}; do
  if curl -s -m 2 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q '"id"'; then break; fi
  sleep 3
done
curl -s -m 2 http://127.0.0.1:$PORT/v1/models | grep -q '"id"' || { echo "server never ready"; exit 1; }
echo "ready (pid $SERVER_PID)"

.venv/bin/python scripts/ocspeed-20260703/accept_depth_probe.py $PORT --set quick --tag "$ARM" --output "$OUT" | tail -3
echo "ARM $ARM DONE -> $OUT"
