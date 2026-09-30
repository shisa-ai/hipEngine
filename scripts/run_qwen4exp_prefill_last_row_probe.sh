#!/usr/bin/env bash
# What is the gate's `prefill_last` row a measurement of?
#
# The X-7 long-form heldout run fails `shape=prefill_last` / `transition=
# prefill_to_c1` at mean KL 1.439e-3 on BOTH dense_wide and wmma. Every
# canonical 12-case gate artifact reports that same row at 1e-9 to 1e-11.
# Five orders of magnitude apart, so before the transition is localized as a
# defect, establish what the row measures: the route, or prefill noise.
set -uo pipefail
cd /home/lhl/hipEngine

ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# hipcc lives in the therock env's bin, which is not on the default PATH; without
# it the first JIT build fails with FileNotFoundError: 'hipcc'.
export PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH"

OUT=benchmarks/results/2026-09-30-prefill-last-row-probe
mkdir -p "$OUT"

python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')" || exit 2

exec uv run --no-sync python scripts/qwen4exp_prefill_last_row_probe.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --repeats 3 \
  --json "$OUT/artifact-prefill-last-row-probe.json" \
  "$@"
