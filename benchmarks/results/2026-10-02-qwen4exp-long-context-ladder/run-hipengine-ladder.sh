#!/usr/bin/env bash
# X-8 long-context ladder: hipEngine canonical AR (prefill + 128 decode
# transitions) at 8K/16K/32K/64K/128K/256K on the same host, same file, same
# exact-token fixture.  Hermetic TheRock gfx1151 environment per
# docs/BENCHMARK.md ("gfx1151 README rows").
set -euo pipefail

REPO_ROOT=/home/lhl/hipEngine
MODEL_ROOT=/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
FIXTURE=${FIXTURE:-/tmp/q4x-long/qwen4exp_long_context_p8192_p262144.json}
OUTDIR=/tmp/q4x-long/hipengine
LOGDIR=/tmp/q4x-long/logs
WARMUPS=${WARMUPS:-1}
REPS=${REPS:-1}
CHUNK=${CHUNK:-1024}
PROFILE=${PROFILE:-production}
RUNGS=${RUNGS:-"8192:4 16384:4 32768:4 65536:1 131072:1 262144:1"}

THEROCK_PY=/home/lhl/miniforge3/envs/therock/bin/python3.12
THEROCK_ROOT=$("$THEROCK_PY" -m rocm_sdk path --root)
THEROCK_SITE=$("$THEROCK_PY" -c 'import site; print(site.getsitepackages()[0])')
THEROCK_CORE_LIB="$THEROCK_SITE/_rocm_sdk_core/lib"
THEROCK_GFX_LIB="$THEROCK_SITE/_rocm_sdk_libraries_gfx1151/lib"
HIPCC_VERSION_FILE=/tmp/q4x-long/hipcc-version-gfx1151.txt

mkdir -p "$OUTDIR" "$LOGDIR"
if [[ ! -s "$HIPCC_VERSION_FILE" ]]; then
  "$THEROCK_ROOT/bin/hipcc" --version > "$HIPCC_VERSION_FILE"
fi

run_rung() {
  local shape="$1" ncateg="$2"
  local cases=""
  if [[ "$ncateg" == "4" ]]; then
    cases="code-p${shape} general_en-p${shape} general_ja-p${shape} mixed_ja_en-p${shape}"
  else
    cases="code-p${shape}"
  fi
  local out="$OUTDIR/hipengine-p${shape}.json"
  local log="$LOGDIR/hipengine-p${shape}.log"
  echo "[$(date -u +%H:%M:%S)] START p${shape} cases='$cases' warmups=$WARMUPS reps=$REPS"
  local start=$SECONDS
  set +e
  env -i \
    HOME="$HOME" USER="${USER:-lhl}" LOGNAME="${LOGNAME:-lhl}" SHELL=/bin/bash TERM=xterm \
    PATH="$THEROCK_ROOT/bin:$(dirname "$THEROCK_PY"):/usr/local/bin:/usr/bin:/bin" \
    LD_LIBRARY_PATH="$THEROCK_ROOT/lib:$THEROCK_CORE_LIB:$THEROCK_GFX_LIB" \
    HIP_PATH="$THEROCK_ROOT" ROCM_PATH="$THEROCK_ROOT" \
    HIP_LIB_PATH="$THEROCK_ROOT/lib" HIP_INCLUDE_PATH="$THEROCK_ROOT/include" \
    HIPENGINE_HIP_ARCH=gfx1151 \
    HIPENGINE_COMPILER_VERSION_FILE="$HIPCC_VERSION_FILE" \
    PYTHONPATH="$REPO_ROOT" \
    GPU_MAX_HW_QUEUES=2 \
    "$THEROCK_PY" "$REPO_ROOT/scripts/qwen4exp_canonical_ar_bench.py" hipengine \
      --model-root "$MODEL_ROOT" \
      --fixture "$FIXTURE" \
      --case-id $cases \
      --prefill-chunk-size "$CHUNK" \
      --execution-profile "$PROFILE" \
      --ple-cache-mode warm \
      --ple-random-access off \
      --warmups "$WARMUPS" \
      --repetitions "$REPS" \
      --compiler-version-file "$HIPCC_VERSION_FILE" \
      --output "$out" > "$log" 2>&1
  local rc=$?
  set -e
  echo "[$(date -u +%H:%M:%S)] END p${shape} rc=$rc elapsed=$((SECONDS-start))s log=$log"
  if [[ $rc -ne 0 ]]; then
    tail -5 "$log" || true
  fi
  return $rc
}

for spec in $RUNGS; do
  shape="${spec%%:*}"
  ncateg="${spec##*:}"
  run_rung "$shape" "$ncateg" || true
done
echo "[$(date -u +%H:%M:%S)] ladder done"
