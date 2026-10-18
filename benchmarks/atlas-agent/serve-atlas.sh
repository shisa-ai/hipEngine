#!/usr/bin/env bash
# Launch atlas for the hipEngine 1:1 comparison.
#
# Atlas is an external read-only peer at $ATLAS_ROOT; this script invokes it and
# never edits it. All atlas source and build output stay in its own tree.
#
# Why serve-amd.sh and not target/release/spark: atlas's HIP backend links the
# CUDA symbol names and supplies them from three HIP shims that build.rs writes
# into target/release/build/atlas-kernels-*/out. serve-amd.sh locates that
# directory and prepends it to LD_LIBRARY_PATH; invoking the binary directly
# fails with "libcuda.so: cannot open shared object file". The same applies to
# the ROCm runtime, which on this host is a conda ROCm SDK rather than
# /opt/rocm, so ATLAS_ROCM_HOME and PATH are set explicitly.
#
# Configuration is atlas's own validated serving config at its best setting:
#   nvidia/Qwen3.8-27B-NVFP4 (NVFP4, W4A8 DP4A decode arm), strix-hip backend,
#   K=4 MTP speculative decode (NUM_DRAFTS=4), BF16 KV, MAX_SEQ_LEN=262144.
#
# CROSS-QUANT: atlas runs NVFP4 and hipEngine runs Q4_K_M GGUF. Atlas carries no
# k-quant kernels and hipEngine has no NVFP4 execution path, so an identical
# quant is not available on either side. Every reported number must say so.
set -uo pipefail

ATLAS_ROOT="${ATLAS_ROOT:-/home/lhl/atlas}"
ROCM_HOME="${ATLAS_ROCM_HOME:-/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel}"
SITE="/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages"
PORT="${PORT:-8081}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-262144}"
NUM_DRAFTS="${NUM_DRAFTS:-4}"
LOG_DIR="${LOG_DIR:-/tmp/atlas-agent/atlas}"

# The local snapshot, so a run never depends on the network or on which
# revision the Hub would serve today.
SNAPSHOT="${ATLAS_MODEL:-$(ls -d /home/lhl/.cache/huggingface/hub/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/*/ 2>/dev/null | head -1)}"
if [ -z "$SNAPSHOT" ] || [ ! -f "${SNAPSHOT}config.json" ]; then
  echo "atlas: no local Qwen3.8-27B-NVFP4 snapshot; set ATLAS_MODEL to a snapshot path" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
{
  echo "atlas: root=$ATLAS_ROOT snapshot=$SNAPSHOT"
  echo "       max_seq_len=$MAX_SEQ_LEN num_drafts=$NUM_DRAFTS port=$PORT"
  echo "       model_name=${MODEL_NAME:-qwen3.8-27b-nvfp4} max_batch=${MAX_BATCH:-3}"
  echo "       rocm=$ROCM_HOME"
  ( cd "$ATLAS_ROOT" && git rev-parse HEAD ) 2>/dev/null
} | tee "$LOG_DIR/launch.txt"

export ATLAS_ROCM_HOME="$ROCM_HOME"
export PATH="$ROCM_HOME/bin:$ROCM_HOME/lib/llvm/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$ROCM_HOME/lib:$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_libraries/lib:${LD_LIBRARY_PATH:-}"
export PORT MAX_SEQ_LEN NUM_DRAFTS
# A clean served name: atlas only honours --model-name when the model argument is
# a local snapshot path, which is what this script passes.
export MODEL_NAME="${MODEL_NAME:-qwen3.8-27b-nvfp4}"
# atlas defaults --max-batch-size to 1, which would serialise the concurrency arm
# and make its aggregate rate meaningless. It cannot simply be raised, though:
# atlas sizes its KV pool from --gpu-memory-utilization and then warns how many
# sequences fit at full --max-seq-len. At 256K on this host the pool fits 3, and
# asking for more does not degrade gracefully -- sequence allocation fails with
# "cuMemsetD32Async failed: status 901" on every prefill, the SSM MTP
# intermediate buffers never allocate, and the server stays up while refusing
# all work. 3 is therefore the largest batch this context length supports here;
# lower MAX_SEQ_LEN if a wider ladder is needed.
export MAX_BATCH="${MAX_BATCH:-3}"
export MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-4096}"
export HOST="${HOST:-127.0.0.1}"

cd "$ATLAS_ROOT"
exec ./serve-amd.sh "$SNAPSHOT" >> "$LOG_DIR/server.log" 2>&1
