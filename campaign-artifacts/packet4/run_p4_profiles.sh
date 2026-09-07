#!/bin/bash
# Packet 4 stage 1: post-band staged-cycle attribution via rocprofv3 on the
# final-leaf bridge child (kernel + marker + HIP runtime + copy + allocation
# traces). Builds are cache-only inside the profiler (prebuild outside).
# Ranks the accept/commit/provider, prompt-prime, and AR-tail shares AFTER the
# packet3 row64 band so the draft/head/commit lever is chosen on fresh data.
set -u
cd /home/lhl/hipEngine-gfx1100-concurrency2-better-mtp
ROCM_SDK_LIB=/home/lhl/mambaforge/envs/therock/lib/python3.12/site-packages/_rocm_sdk_core/lib
OUT=/tmp/he-bettermtp-raw/packet4
mkdir -p "$OUT"
MODEL=/models/gguf/Qwen3.8-27B-Q4_K_M.gguf
PY=.venv/bin/python

# Cache-only builds require a warm JIT cache: warm the bridge child once
# WITHOUT the profiler before profiling.
warm () {
  local tag="$1"; shift
  if [ -s "$OUT/warm-$tag.json" ]; then echo "=== skip warm-$tag"; return; fi
  echo "=== WARM $tag $(date -u +%H:%M:%S)"
  env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100 \
    HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
    PYTHONPATH=. \
    "$PY" -u scripts/specdec2_perf_bridge.py \
    --model "$MODEL" --backend hip_gfx1100 --target-arch gfx1100 \
    --quant-label Q4_K_M --execution-profile production \
    --scope train --limit 1 --max-tokens 12 --runs 1 --max-sequence-length 1024 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build --roctx-markers --profile-child \
    --output "$OUT/warm-$tag.json" "$@" > "$OUT/warm-$tag.log" 2>&1
  echo "=== WARM $tag rc=$? $(date -u +%H:%M:%S)"
}

profile () {
  local tag="$1"; shift
  if [ -s "$OUT/profile-$tag-child.json" ]; then echo "=== skip profile-$tag"; return; fi
  echo "=== START profile-$tag $(date -u +%H:%M:%S)"
  env LD_LIBRARY_PATH="$ROCM_SDK_LIB:$ROCM_SDK_LIB/rocm_sysdeps/lib" \
    HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100 \
    HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
    HIPENGINE_REQUIRE_CACHED_BUILD=1 PYTHONPATH=. \
    rocprofv3 --kernel-trace --marker-trace --hip-runtime-trace \
    --memory-copy-trace --memory-allocation-trace --output-format csv \
    -d "$OUT/trace-$tag" -- \
    "$PY" -u scripts/specdec2_perf_bridge.py \
    --model "$MODEL" --backend hip_gfx1100 --target-arch gfx1100 \
    --quant-label Q4_K_M --execution-profile production \
    --scope train --limit 1 --max-tokens 12 --runs 1 --max-sequence-length 1024 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build --roctx-markers --profile-child \
    --output "$OUT/profile-$tag-child.json" "$@" > "$OUT/profile-$tag.log" 2>&1
  local rc=$?
  echo "=== DONE profile-$tag rc=$rc $(date -u +%H:%M:%S)"
  if [ $rc -ne 0 ]; then tail -5 "$OUT/profile-$tag.log"; fi
}

hipcc --version > /tmp/hipengine-hipcc-version.txt
warm c8k3 --concurrency 8 --service-capacity 8 --budgets 3
profile c8k3 --concurrency 8 --service-capacity 8 --budgets 3
profile c2k2 --concurrency 2 --service-capacity 2 --budgets 2
echo "=== STAGE P4-1 COMPLETE $(date -u +%H:%M:%S)"
