#!/bin/zsh
# Tool-calling reliability gauntlet vs the candidate server on :18170.
# Runs the three client-contract lanes of agent_user_path_qa with tools on.
# Zero tolerance: any failed/malformed tool call fails the gate.
set -uo pipefail
ROOT=/Users/youssof/Projects/MTPLX-release/mtplx-kvcache-v2-20260703
SCRATCH=/private/tmp/claude-501/-Users-youssof-Projects-MTPLX/19ebc628-6580-4b3e-8f15-f3f9a34c06c4/scratchpad
BASE=http://127.0.0.1:18170
MODEL=mtplx-qwen36-27b-optimized-speed
PROJECT="/Users/youssof/Documents/bow masters 3d"

# own the candidate server lifecycle
sudo -n ~/.mtplx/bin/thermalforge max >/dev/null 2>&1; sleep 4
RPM=$(sudo -n ~/.mtplx/bin/thermalforge status 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(min(f['actual_rpm'] for f in d['fans']))")
echo "fans min rpm: $RPM"
[ "$RPM" -ge 7000 ] || { echo "FANS NOT MAX - abort"; exit 9; }
(cd "$ROOT" && MTPLX_NAX_VERIFY=1 MTPLX_NAX_M4_IMPL=vk_k nohup .venv/bin/mtplx serve \
  --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed --port 18170 \
  --warmup-tokens 16 > /tmp/gauntlet_server.log 2>&1 &)
for i in {1..90}; do curl -s -m 2 $BASE/v1/models 2>/dev/null | grep -q '"id"' && break; sleep 3; done
echo "gauntlet server ready $(date '+%H:%M:%S')"

for mode in openai anthropic opencode; do
  echo "=== mode: $mode $(date '+%H:%M:%S') ==="
  "$ROOT/.venv/bin/python" "$ROOT/scripts/agent_user_path_qa.py" \
    --base-url "$BASE" --model "$MODEL" --mode "$mode" \
    --prompt-kind tool --tools --concurrency 2 \
    --project-root "$PROJECT" \
    --output-jsonl "$SCRATCH/gauntlet_${mode}.jsonl" 2>&1 | tail -4
done
echo "=== concurrency lane $(date '+%H:%M:%S') ==="
"$ROOT/.venv/bin/python" "$ROOT/scripts/opencode_concurrency_qa.py" \
  --base-url "$BASE" --model "$MODEL" --mode http --concurrency 3 \
  --prompt-kind mixed \
  --output-jsonl "$SCRATCH/gauntlet_concurrency.jsonl" 2>&1 | tail -4
GPID=$(lsof -nP -iTCP:18170 -sTCP:LISTEN -t 2>/dev/null | head -1)
[ -n "$GPID" ] && kill "$GPID"
for i in {1..40}; do [ -z "$(lsof -nP -iTCP:18170 -sTCP:LISTEN -t 2>/dev/null)" ] && break; sleep 1; done
echo "teardown 18170: $(lsof -nP -iTCP:18170 -sTCP:LISTEN -t 2>/dev/null || echo free)"
echo "GAUNTLET DONE $(date '+%H:%M:%S')"
