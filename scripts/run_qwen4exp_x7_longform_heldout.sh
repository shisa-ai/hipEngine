#!/usr/bin/env bash
# X-7 long-form heldout certificate run: dense_wide at layers 16-47.
#
# The promoted 16-47 default is `dense_wide`, whose 257-row floor the 18-prompt
# admission suite cannot reach (longest prompt 71 tokens). This suite's four
# natural long tasks are 424-512 tokens, one per category, so the route can
# actually engage and the run is qualification rather than a control.
set -uo pipefail
cd /home/lhl/hipEngine

ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# hipcc lives in the therock env's bin, which is not on the default PATH; without
# it the first JIT build fails with FileNotFoundError: 'hipcc'.
export PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH"

OUT=benchmarks/results/2026-09-30-qwen4exp-x7-longform-heldout
mkdir -p "$OUT"

python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')" || exit 2

exec uv run --no-sync python scripts/execution_profile_q8_wmma_prefill_layers_gate.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --fixture benchmarks/fixtures/qwen4exp_longform_heldout_4prompt.json \
  --route dense_wide \
  --layers 16-47 \
  --decode-steps 24 \
  --repeat-runs 3 \
  --prefill-chunk-size 1024 \
  --output "$OUT/artifact-dense-wide-16-47-longform-heldout.json"
