#!/usr/bin/env bash
# Same-GGUF four-way comparison on gfx1151 (Framework Desktop, Radeon 8060S):
#   hipEngine production AR vs llama.cpp HIP/Vulkan vs halo-box strix-llama.cpp HIP/Vulkan.
# Model: /models/gguf/Qwen3.8-27B-Q4_K_M.gguf, BF16 K/V, no speculation.
#
# Run from the repository root:  bash benchmarks/results/20260913-qwen38-27b-comparison/run_comparison.sh
#
# Outputs land in this directory (raw/*.json) plus raw/run_comparison.log.
# Set BLOCKING_ARM=1 to add the strix-llama.cpp HIP diagnostic arm that runs the
# binary under its documented gfx1151 async workaround (HIP_LAUNCH_BLOCKING=1).
# The 2026-09-13 recorded run used this script with OUT=/tmp/hip1151-4way and
# BLOCKING_ARM=1; the artifacts were copied into raw/ unmodified.
set -uo pipefail
ROOT=/home/lhl/hipEngine
PY="$ROOT/.venv/bin/python"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/raw}"
MODEL=/models/gguf/Qwen3.8-27B-Q4_K_M.gguf
HB="${HB:-/tmp/hipengine-halobox-head-20260913}"   # halo-box/strix-llama.cpp @ 654803517
UP="${UP:-/home/lhl/llama.cpp}"
HIPENGINE_WORKTREE="${HIPENGINE_WORKTREE:-/tmp/hipengine-qwen38-release-20260913}"
mkdir -p "$OUT"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate therock
ROCM_ROOT=$(python -m rocm_sdk path --root)
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib64:$ROCM_ROOT/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT"

echo "=== host $(hostname) | $(date -Is)"
echo "=== dpm=$(cat /sys/class/drm/card1/device/power_dpm_force_performance_level) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"

# --- 1. hipEngine production AR (published-protocol resident sweep) -----------
echo "=== [hipengine] start $(date -Is)"
(
  cd "$HIPENGINE_WORKTREE"
  env GPU_MAX_HW_QUEUES=2 \
      HIPENGINE_HIP_ARCH=gfx1151 \
      HIPENGINE_COMPILER_VERSION_FILE=/tmp/hip1151-t6/hipcc-version-gfx1151.txt \
      HIPENGINE_GGUF_FP16_RECURRENT_STATE=0 \
      HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1 \
      HIPENGINE_GGUF_VERIFY_PRODUCTION_Q4_ROWTILE=1 \
      HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256=c4a4a342e2243c2dcc430174606dde682393a2bd2e30acc83129027fcf572acc \
      PYTHONPATH=. "$PY" scripts/qwen38_gfx1151_readme_sweep.py \
        --prompt-lengths 512 1024 4096 --decode-tokens 128 \
        --warmups 1 --repetitions 3 \
        --output "$OUT/hipengine-production-ar.json"
) 2>&1 | tail -25
echo "=== [hipengine] done $(date -Is)"

# --- 2. llama.cpp-family backends (llama-bench split timing, -r 5) ------------
bench() { # name binary backend dev
  local name="$1" bin="$2" backend="$3" dev="$4"
  echo "=== [$name] start $(date -Is) :: $bin"
  "$PY" "$ROOT/scripts/llamacpp_bench_with_peak.py" \
    --llama-bench "$bin" --model "$MODEL" --quant gguf_q4_k_m \
    --backend "$backend" --workloads 512/128 1K/128 4K/128 --repetitions 5 \
    --ngl 99 --flash-attn 1 --cache-type-k bf16 --cache-type-v bf16 \
    --poll 10 --card-name card1 --memory-domain gtt \
    --extra-args "-dev $dev" --status complete_measured \
    --note "same-GGUF four-way comparison on gfx1151, $name" \
    --output "$OUT/$name.json" 2>&1 | tail -20
  echo "=== [$name] done $(date -Is)"
}

bench upstream-hip    "$UP/llama.cpp-hip/build/bin/llama-bench"    hip    ROCm0
bench upstream-vulkan "$UP/llama.cpp-vulkan/build/bin/llama-bench" vulkan Vulkan0
bench halobox-hip     "$HB/build-hip/bin/llama-bench"              hip    ROCm0
bench halobox-vulkan  "$HB/build-vulkan/bin/llama-bench"           vulkan Vulkan0

# --- 3. diagnostic arm: strix-llama.cpp HIP with the gfx1151 async workaround --
if [ "${BLOCKING_ARM:-0}" = "1" ]; then
  echo "=== [halobox-hip-blocking] start $(date -Is)"
  HIP_LAUNCH_BLOCKING=1 "$PY" "$ROOT/scripts/llamacpp_bench_with_peak.py" \
    --llama-bench "$HB/build-hip/bin/llama-bench" --model "$MODEL" --quant gguf_q4_k_m \
    --backend hip --workloads 512/128 4K/128 --repetitions 5 \
    --ngl 99 --flash-attn 1 --cache-type-k bf16 --cache-type-v bf16 \
    --poll 10 --card-name card1 --memory-domain gtt \
    --extra-args "-dev ROCm0" --status diagnostic \
    --note "strix-llama.cpp HIP with its documented gfx1151 async correctness workaround" \
    --output "$OUT/halobox-hip-launch-blocking.json" 2>&1 | tail -20
  echo "=== [halobox-hip-blocking] done $(date -Is)"
fi

echo "=== ALL DONE $(date -Is)"
