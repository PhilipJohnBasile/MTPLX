#!/bin/zsh
# Discarded full K3 control primer -> C0 -> MoE-tail candidate -> C1 in one
# attested GPU window. Invoke only through bench/laguna/run_guarded.py.
set -euo pipefail

VENV=/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python
WORKTREE=/private/tmp/claude-501/-Users-davidtai-projects-OpenSourceWTF/8e9b6abf-6a38-4e6e-ade0-6b0f191bb256/scratchpad/moe-tail
BENCH=/Users/davidtai/projects/OpenSourceWTF/bench/deepseek-v4
MODEL=/Users/davidtai/models/DeepSeek-V4-Flash-2bit-DQ-mtp
PROMPT="$BENCH/smoke-2bitdq-20260731-prompt2.txt"
VALIDATOR="$WORKTREE/scripts/deepseek_v4_validate_moe_tail_k3_bracket.py"
PROMPT_SHA256=ee94397faa812c91d5f1a0ee17c5bb6ca6032883653591dd33d4cfddb737ac33
CONFIG_SHA256=c8ff87fd5ee5c9587d0c937e9bfd3193e1a1621141aa367848a9610b3291fa6f
INDEX_SHA256=c84d2b369f5d5023d0f2d183fc36a935a3981751414996243b65f069983e43d8
MLX_CORE_SHA256=d7bd29fc20b4a08318d21161c3dfb340889cc9454c5e554ad749eb0127cfa2d6
MLX_LIB_SHA256=2ee6fbd32ff22e22e1301ebe3c3bece95584104ff9cbc900513d41a095211bbd
TAG="${1:-moe-tail-k3-$(date -u +%Y%m%dT%H%M%SZ)}"

# Consume run_guarded's one-shot pipe before any MLX import. The issued private
# receipt is reusable by all four descendants and remains bound to this process
# ancestry and the still-held canonical lock.
GUARD_PIPE_FD=${MTPLX_GUARD_ATTEST_FD:-}
GUARD_ISSUED=$("$VENV" -u "$WORKTREE/scripts/deepseek_v4_guard_window.py" issue)
GUARD_RECEIPT=${GUARD_ISSUED%%$'\t'*}
GUARD_DIGEST=${GUARD_ISSUED#*$'\t'}
[[ -n "$GUARD_PIPE_FD" && "$GUARD_RECEIPT" != "$GUARD_ISSUED" && ${#GUARD_DIGEST} == 64 ]] || {
  print -u2 "[moe-tail-arms] malformed guard-window metadata"
  exit 1
}
exec {GUARD_PIPE_FD}<&-
unset MTPLX_GUARD_ATTEST_FD MTPLX_GUARD_ATTEST_NONCE GUARD_ISSUED
GUARD_DIR=${GUARD_RECEIPT:h}
cleanup_guard_receipt() {
  /bin/rm -f -- "$GUARD_RECEIPT"
  /bin/rmdir -- "$GUARD_DIR" 2>/dev/null || true
}
trap cleanup_guard_receipt EXIT

[[ -x "$VENV" && -f "$WORKTREE/scripts/deepseek_v4_mtpk_bench.py" \
  && -f "$VALIDATOR" && -f "$PROMPT" && -d "$MODEL" ]] || {
  print -u2 "[moe-tail-arms] interpreter, scripts, prompt, or model missing"
  exit 1
}
[[ -z "$(git -C "$WORKTREE" status --porcelain)" ]] || {
  print -u2 "[moe-tail-arms] worktree is dirty; refusing an unrepeatable bracket"
  exit 1
}
actual_prompt_sha=$(shasum -a 256 "$PROMPT" | awk '{print $1}')
actual_config_sha=$(shasum -a 256 "$MODEL/config.json" | awk '{print $1}')
actual_index_sha=$(shasum -a 256 "$MODEL/model.safetensors.index.json" | awk '{print $1}')
[[ "$actual_prompt_sha" == "$PROMPT_SHA256" \
  && "$actual_config_sha" == "$CONFIG_SHA256" \
  && "$actual_index_sha" == "$INDEX_SHA256" ]] || {
  print -u2 "[moe-tail-arms] canonical prompt/config/index identity mismatch"
  exit 1
}
mlx_identity=$(PYTHONPATH="$WORKTREE" "$VENV" -u - <<'PY'
import hashlib
from pathlib import Path
import mlx.core as mx
core = Path(mx.__file__).resolve()
library = core.parent / "lib" / "libmlx.dylib"
print(mx.__version__)
print(hashlib.sha256(core.read_bytes()).hexdigest())
print(hashlib.sha256(library.read_bytes()).hexdigest())
PY
)
actual_mlx=${${(f)mlx_identity}[1]}
actual_mlx_core_sha=${${(f)mlx_identity}[2]}
actual_mlx_lib_sha=${${(f)mlx_identity}[3]}
[[ "$actual_mlx" == 0.31.2 && "$actual_mlx_core_sha" == "$MLX_CORE_SHA256" \
  && "$actual_mlx_lib_sha" == "$MLX_LIB_SHA256" ]] || {
  print -u2 "[moe-tail-arms] official MLX 0.31.2 binary identity mismatch"
  exit 1
}
MODEL_PATH="$MODEL" PYTHONPATH="$WORKTREE/scripts:$WORKTREE" "$VENV" -u - <<'PY'
import os
from pathlib import Path
from deepseek_v4_moe_tail_gate import _validate_model_artifact
_validate_model_artifact(Path(os.environ["MODEL_PATH"]))
PY

# Remove every inherited experiment selector, including future MTPLX knobs.
# Re-export only the fixed Stage-4 arm below; the wired-memory knob is untouched.
for entry in ${(f)"$(env)"}; do
  name=${entry%%=*}
  if [[ "$name" == MTPLX_* ]]; then
    unset "$name"
  fi
done
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKTREE/scripts:$WORKTREE"
export HF_HUB_OFFLINE=1
export MTPLX_COMPILED_VERIFY=off
export MTPLX_DSV4_ATTN=fused
export MTPLX_DSV4_FP32_ACTIVATIONS=0
export MTPLX_DSV4_HC_COMPILE=1
export MTPLX_DSV4_O_LORA=cached
export MTPLX_DSV4_SINKHORN_KERNEL=1
export MTPLX_DSV4_GUARD_WINDOW_PATH="$GUARD_RECEIPT"
export MTPLX_DSV4_GUARD_WINDOW_SHA256="$GUARD_DIGEST"

run_arm() {
  local label="$1" enabled="$2" stem="$3" role="$4"
  print "\n################################################################"
  print "### ARM $label: MTPLX_DSV4_MOE_TAIL=$enabled"
  print "### canonical 328 prompt; 256 output; K3; started $(date +%H:%M:%S)"
  print "################################################################"
  env MTPLX_DSV4_MOE_TAIL="$enabled" "$VENV" -u \
    "$WORKTREE/scripts/deepseek_v4_mtpk_bench.py" \
    --model "$MODEL" --prompt-file "$PROMPT" --max-tokens 256 --depths 3 \
    --verify-strategy capture_commit --verify-core stock \
    --mtp-history-policy committed --warmup-tokens 8 \
    --receipt-role "$role" --out "$BENCH/$TAG-$stem"
}

# The first complete K3 process pays the observed first-arm Metal/library cold
# bias (23.773 -> 32.269 tok/s in the gate-cache bracket). Persist it as evidence
# but mark it mechanically ineligible for verdict.
run_arm "DISCARDED full K3 control primer" 0 primer discarded_control_primer
run_arm "C0 Stage-4 control" 0 before measurement
run_arm "MoE-tail M4 candidate" 1 candidate measurement
run_arm "C1 Stage-4 control" 0 after measurement

VALIDATION="$BENCH/$TAG-validation.json"
if "$VENV" -u "$VALIDATOR" \
  --primer "$BENCH/$TAG-primer.json" \
  --before "$BENCH/$TAG-before.json" \
  --candidate "$BENCH/$TAG-candidate.json" \
  --after "$BENCH/$TAG-after.json" \
  --peak-ceiling-gib 108 --out "$VALIDATION"; then
  print "[moe-tail-arms] PASS: $VALIDATION"
else
  validation_rc=$?
  print -u2 "[moe-tail-arms] non-promotable (exit=$validation_rc); receipts preserved at $BENCH/$TAG-{primer,before,candidate,after,validation}.json"
  exit "$validation_rc"
fi
