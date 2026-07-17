#!/bin/zsh
# Competitor head-to-head (2026-07-03 PM wave): Ollama (both runners) + oMLX.
# Warm/cold probe at 4k/12k, plus an oMLX restart-warm leg (their SSD cache).
set -uo pipefail
ROOT=/Users/youssof/Projects/MTPLX-release/mtplx-kvcache-v2-20260703
SCRATCH=/private/tmp/claude-501/-Users-youssof-Projects-MTPLX/19ebc628-6580-4b3e-8f15-f3f9a34c06c4/scratchpad
PROBE="$ROOT/.venv/bin/python $ROOT/scripts/kvcache_warm_probe_20260703.py"

fans_max() {
  sudo -n ~/.mtplx/bin/thermalforge max >/dev/null 2>&1; sleep 4
  RPM=$(sudo -n ~/.mtplx/bin/thermalforge status 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(min(f['actual_rpm'] for f in d['fans']))")
  echo "fans min rpm: $RPM"
  [ "$RPM" -ge 7000 ] || { echo "FANS NOT MAX - abort"; exit 9; }
}

echo "===== OLLAMA (server on 11434) ====="
fans_max
for tag in "qwen3.6:35b-a3b-mtp-q4_K_M" "qwen3.6:35b-a3b"; do
  for size in 4000 12000; do
    echo "--- ollama $tag @ $size $(date '+%H:%M:%S') ---"
    eval $PROBE --flavor ollama --base-url http://127.0.0.1:11434 \
      --model "$tag" --target-tokens $size --variants cold,a,b_rag \
      --label "ollama-$tag-$size" \
      --output-jsonl "$SCRATCH/competitor_ollama.jsonl" --no-require-max-fans 2>&1 | tail -4
  done
  # unload between runners so models don't stack in memory
  curl -s http://127.0.0.1:11434/api/generate -d "{\"model\": \"$tag\", \"keep_alive\": 0}" >/dev/null 2>&1
  sleep 8
done

echo "===== OMLX (our Speed artifact, SSD cache on, port 18310) ====="
fans_max
mkdir -p "$SCRATCH/omlx-ssd"
( cd "$SCRATCH" && nohup ./omlx-venv/bin/omlx serve --model-dir "$SCRATCH/omlx-models" \
    --port 18310 --paged-ssd-cache-dir "$SCRATCH/omlx-ssd" \
    --paged-ssd-cache-max-size 32GB > /tmp/omlx_serve.log 2>&1 & )
for i in {1..60}; do curl -s -m 2 http://127.0.0.1:18310/v1/models 2>/dev/null | grep -q "qwen36-27b-speed" && break; sleep 3; done
echo "omlx up $(date '+%H:%M:%S')"
for size in 4000 12000; do
  echo "--- omlx @ $size $(date '+%H:%M:%S') ---"
  eval $PROBE --flavor mtplx --base-url http://127.0.0.1:18310 \
    --model qwen36-27b-speed --target-tokens $size --variants cold,a,b_rag \
    --session-id "omlx-restartwarm-$size" --label "omlx-$size" \
    --output-jsonl "$SCRATCH/competitor_omlx.jsonl" --no-require-max-fans 2>&1 | tail -4
done
echo "--- omlx restart-warm leg $(date '+%H:%M:%S') ---"
OPID=$(lsof -nP -iTCP:18310 -sTCP:LISTEN -t 2>/dev/null | head -1)
[ -n "$OPID" ] && kill "$OPID"
for i in {1..40}; do [ -z "$(lsof -nP -iTCP:18310 -sTCP:LISTEN -t 2>/dev/null)" ] && break; sleep 1; done
( cd "$SCRATCH" && nohup ./omlx-venv/bin/omlx serve --model-dir "$SCRATCH/omlx-models" \
    --port 18310 --paged-ssd-cache-dir "$SCRATCH/omlx-ssd" \
    --paged-ssd-cache-max-size 32GB > /tmp/omlx_serve2.log 2>&1 & )
for i in {1..60}; do curl -s -m 2 http://127.0.0.1:18310/v1/models 2>/dev/null | grep -q "qwen36-27b-speed" && break; sleep 3; done
eval $PROBE --flavor mtplx --base-url http://127.0.0.1:18310 \
  --model qwen36-27b-speed --target-tokens 12000 --variants a \
  --session-id "omlx-restartwarm-12000" --label "omlx-restartwarm" \
  --output-jsonl "$SCRATCH/competitor_omlx.jsonl" --no-require-max-fans 2>&1 | tail -4

OPID=$(lsof -nP -iTCP:18310 -sTCP:LISTEN -t 2>/dev/null | head -1)
[ -n "$OPID" ] && kill "$OPID"
for i in {1..40}; do [ -z "$(lsof -nP -iTCP:18310 -sTCP:LISTEN -t 2>/dev/null)" ] && break; sleep 1; done
echo "teardown 18310: $(lsof -nP -iTCP:18310 -sTCP:LISTEN -t 2>/dev/null || echo free)"
echo "COMPETITORS DONE $(date '+%H:%M:%S')"
