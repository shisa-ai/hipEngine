#!/usr/bin/env bash
# DP-8 value measurement: the end-to-end prefill value of staged 13-15.
#
# DP-8's composition (staged 13-15 + WMMA 16-47) passes every hard gate, but its
# value has only ever been ESTIMATED, as `3 layers x 8.985 ms per layer-step x 4
# chunks of 1024 rows = ~0.66% of prefill`. That is exactly the extrapolation the
# source artifact's own `extrapolation_warning` forbids, so this run measures the
# share instead of deriving it.
#
# Both arms run the shipping WMMA prefill (use_wmma_prefill=True). The only
# difference between them is HIPENGINE_QWEN4_EXP_Q8_STAGED_F32_LAYERS, so the
# delta is the staged tile replacing the exact chain on layers 13-15.
set -uo pipefail
cd /home/lhl/hipEngine

ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# hipcc lives in the therock env's bin, which is not on the default PATH; without
# it the first JIT build fails with FileNotFoundError: 'hipcc'.
export PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH"

OUT=benchmarks/results/2026-09-30-dp8-staged-scope-prefill-value
mkdir -p "$OUT"

python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')" || exit 2

# Prompt 4096 at the 1024-row chunk cap is the four chunks the estimate assumed.
# Later arguments override these defaults, so a shorter diagnostic run only has
# to name what it changes.
exec uv run --no-sync python scripts/qwen4exp_dp8_staged_scope_prefill_ab.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --prompt-length 4096 \
  --candidate-layers 13,14,15 \
  --warmups 2 \
  --repetitions 9 \
  --json "$OUT/artifact-staged-13-15-prefill-ab.json" \
  "$@"
