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
# MTP AND LONG CONTEXT: the dense MTP adapter is bounded only by the target's
# own max_sequence_length, which is allocated capacity rather than an evidence
# window. A row whose prompt leaves no room for even one generated token is set
# to candidate_budget 0 with target_context_k0 and decodes autoregressively;
# prompt length is otherwise not an admission axis.
#
# Measured on this host (gfx1151, Qwen3.8-27B Q4_K_M, BF16 KV, budget 3),
# greedy requests are served through speculative_mtp at 128, 512, 600, 1,024,
# 1,025, 2,048, 4,096 and 8,192 prompt tokens via /v1/completions and at 917,
# 1,738, 5,674 and 11,291 via /v1/chat/completions, so long agent turns are
# served speculatively. The axis that still falls back to AR is sampling, not
# context: temperature > 0 is refused with automatic_mtp_scope_not_promoted and
# ignore_eos with sampling_mode_not_qualified.
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

exec bash "$ENVWRAP" "$PY" -m hipengine.server \
  --model "$MODEL" --backend hip_gfx1151 --served-model-name "$SERVED_NAME" \
  --max-context-tokens "$MAX_CONTEXT" --kv-storage bf16 \
  --execution-profile production \
  --speculative-mtp-serving enabled \
  --speculative-candidate-budget "$CANDIDATE_BUDGET" \
  --prefix-cache radix \
  --host 127.0.0.1 --port "$PORT" --log-level info \
  > "$LOG_DIR/server.log" 2>&1
