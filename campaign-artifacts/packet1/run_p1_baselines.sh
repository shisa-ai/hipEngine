#!/bin/bash
# Packet 1 stage 1: unprofiled staged-cycle baselines via the specdec2 bridge
# (final-leaf mode semantics without the profiler), one prompt, D12.
# Also serves as the profiler-cache warmup for stage 2.
set -u
cd /home/lhl/hipEngine-gfx1100-concurrency2-better-mtp
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt
OUT=/tmp/he-bettermtp-raw/packet1
mkdir -p "$OUT"
MODEL=/models/gguf/Qwen3.8-27B-Q4_K_M.gguf
PY=.venv/bin/python

bridge () {
  local tag="$1"; shift
  if [ -s "$OUT/base-$tag.json" ]; then echo "=== skip base-$tag"; return; fi
  echo "=== START base-$tag $(date -u +%H:%M:%S)"
  "$PY" -u scripts/specdec2_perf_bridge.py \
    --model "$MODEL" --backend hip_gfx1100 --target-arch gfx1100 \
    --quant-label Q4_K_M --execution-profile production \
    --scope train --limit 1 --max-tokens 12 --runs 1 --max-sequence-length 1024 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build --roctx-markers \
    --output "$OUT/base-$tag.json" "$@" 2>&1 | tail -4
  echo "=== DONE base-$tag rc=$? $(date -u +%H:%M:%S)"
}

# Product-route cells (no screening env).
bridge c8k3  --concurrency 8 --service-capacity 8 --budgets 3
bridge c2k2  --concurrency 2 --service-capacity 2 --budgets 2

# Screening cells (explicitly unqualified; diagnostic plan + screening env).
export HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS=1
bridge c5k3 --concurrency 5 --service-capacity 8 --budgets 3 --diagnostic-plan
bridge c3k3 --concurrency 3 --service-capacity 8 --budgets 3 --diagnostic-plan
unset HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS
echo "=== STAGE P1-1 COMPLETE $(date -u +%H:%M:%S)"
