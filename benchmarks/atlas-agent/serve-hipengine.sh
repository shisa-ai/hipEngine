#!/usr/bin/env bash
# Launch hipEngine for the atlas 1:1 comparison.
#
# One engine at a time: atlas and hipEngine must never hold the GPU together,
# or neither rate is attributable. run.sh enforces the ordering.
#
# Configuration is the qualified production cell, not a tuning sweep:
#   Qwen3.8-27B Q4_K_M GGUF, hip_gfx1151, production execution profile,
#   BF16 KV, prefix cache on, MTP enabled at candidate budget 3.
#
# MTP AND LONG CONTEXT: hipEngine's dense MTP adapter admits only inside a
# 1,023-token window (hipengine/generation/qwen35_gguf_mtp2.py
# _MTP2_QUALIFIED_CONTEXT_WINDOW). Above it a row is set to candidate_budget 0
# with target_context_k0 and decodes autoregressively. That is the qualified
# behaviour, so at 256K this server decodes AR unless FORCE_LONG_MTP=1 is set,
# which raises HIPENGINE_MTP2_MAX_CONTEXT_TOKENS and is an UNQUALIFIED
# diagnostic: raising the window alone measures 0.57x, because the target and
# draft graphs decline into their eager paths per cycle (docs/REFACTOR.md,
# "Long-context MTP window override").
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:-/home/lhl/models/gguf/Qwen3.8-27B-Q4_K_M.gguf}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b-q4km}"
PORT="${PORT:-8097}"
MAX_CONTEXT="${MAX_CONTEXT:-262144}"
CANDIDATE_BUDGET="${CANDIDATE_BUDGET:-3}"
LOG_DIR="${LOG_DIR:-/tmp/atlas-agent/hipengine}"
PY="${PY:-/home/lhl/miniforge3/envs/vllm/bin/python}"
ENVWRAP="${ENVWRAP:-/tmp/t17-env.sh}"

mkdir -p "$LOG_DIR"
echo "hipengine: model=$MODEL max_context=$MAX_CONTEXT budget=$CANDIDATE_BUDGET port=$PORT" | tee "$LOG_DIR/launch.txt"
git rev-parse HEAD | tee -a "$LOG_DIR/launch.txt"

EXTRA=()
if [ "${FORCE_LONG_MTP:-0}" = "1" ]; then
  EXTRA+=(--max-context-tokens "$MAX_CONTEXT")
  export HIPENGINE_MTP2_MAX_CONTEXT_TOKENS="$MAX_CONTEXT"
  echo "WARNING: FORCE_LONG_MTP=1 raises the MTP window past its qualified 1,023 tokens." \
       "This arm is an unqualified diagnostic, not a promotion." | tee -a "$LOG_DIR/launch.txt"
fi

exec bash "$ENVWRAP" "$PY" -m hipengine.server \
  --model "$MODEL" --backend hip_gfx1151 --served-model-name "$SERVED_NAME" \
  --max-context-tokens "$MAX_CONTEXT" --kv-storage bf16 \
  --execution-profile production \
  --speculative-mtp-serving enabled \
  --speculative-candidate-budget "$CANDIDATE_BUDGET" \
  --prefix-cache radix \
  --host 127.0.0.1 --port "$PORT" --log-level info \
  "${EXTRA[@]}" > "$LOG_DIR/server.log" 2>&1
