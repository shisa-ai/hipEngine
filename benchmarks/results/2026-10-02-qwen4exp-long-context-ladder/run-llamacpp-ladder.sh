#!/usr/bin/env bash
# X-8 long-context ladder for llama.cpp-family Strix Halo builds: same host,
# same file, same exact-token fixture, same rungs as the hipEngine ladder.
set -euo pipefail

REPO_ROOT=/home/lhl/hipEngine
MODEL=/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
FIXTURE_DIR=/tmp/q4x-long/fixtures
OUTDIR=/tmp/q4x-long/llamacpp
LOGDIR=/tmp/q4x-long/logs
WARMUPS=${WARMUPS:-0}
REPS=${REPS:-1}
RUNGS=${RUNGS:-"8192 16384 32768 65536 131072 262008"}
# engine-label:build-dir:source-root
ENGINES=${ENGINES:-"pwilkin-40a9f4d01:/home/lhl/comparators-20260915/build-pwilkin:/home/lhl/comparators-20260915/pwilkin-llamacpp upstream-6011c34ce:/home/lhl/comparators-20260915/build-upstream:/home/lhl/comparators-20260915/upstream-llamacpp"}
KVARGS=${KVARGS:-"--server-arg=-ctk --server-arg=bf16 --server-arg=-ctv --server-arg=bf16"}
PORT_BASE=${PORT_BASE:-18080}

THEROCK_PY=/home/lhl/miniforge3/envs/therock/bin/python3.12
THEROCK_ROOT=$("$THEROCK_PY" -m rocm_sdk path --root)
THEROCK_SITE=$("$THEROCK_PY" -c 'import site; print(site.getsitepackages()[0])')
THEROCK_CORE_LIB="$THEROCK_SITE/_rocm_sdk_core/lib"
THEROCK_GFX_LIB="$THEROCK_SITE/_rocm_sdk_libraries_gfx1151/lib"
export LD_LIBRARY_PATH="$THEROCK_ROOT/lib:$THEROCK_CORE_LIB:$THEROCK_GFX_LIB:${LD_LIBRARY_PATH:-}"
export PATH="$THEROCK_ROOT/bin:$(dirname "$THEROCK_PY"):$PATH"

mkdir -p "$OUTDIR" "$LOGDIR"
port=$PORT_BASE

for engine in $ENGINES; do
  label="${engine%%:*}"
  rest="${engine#*:}"
  build="${rest%%:*}"
  src="${rest##*:}"
  for shape in $RUNGS; do
    port=$((port+1))
    ctx=$((shape+512)); if [ "$ctx" -gt 262144 ]; then ctx=262144; fi
    out="$OUTDIR/${label}-p${shape}.json"
    log="$LOGDIR/llamacpp-${label}-p${shape}.log"
    echo "[$(date -u +%H:%M:%S)] START $label p${shape} ctx=$ctx port=$port"
    start=$SECONDS
    set +e
    "$THEROCK_PY" "$REPO_ROOT/scripts/qwen4exp_canonical_ar_bench.py" llamacpp \
      --server-bin "$build/bin/llama-server" \
      --model "$MODEL" \
      --engine-label "$label" \
      --source-root "$src" \
      --fixture "$FIXTURE_DIR/p${shape}-4cat.json" \
      --output "$out" \
      --server-log "$LOGDIR/llamacpp-server-${label}-p${shape}.log" \
      --host 127.0.0.1 --port "$port" \
      --warmups "$WARMUPS" --repetitions "$REPS" \
      --startup-timeout 900 --request-timeout 7200 \
      --server-arg=-ngl --server-arg=999 --server-arg=-fa --server-arg=on \
      --server-arg=-t --server-arg=4 --server-arg=-b --server-arg=8192 \
      --server-arg=-ub --server-arg=2048 --server-arg=-c --server-arg="$ctx" \
      $KVARGS > "$log" 2>&1
    rc=$?
    set -e
    echo "[$(date -u +%H:%M:%S)] END $label p${shape} rc=$rc elapsed=$((SECONDS-start))s"
    if [[ $rc -ne 0 ]]; then
      tail -6 "$log" || true
      tail -6 "$LOGDIR/llamacpp-server-${label}-p${shape}.log" 2>/dev/null || true
    fi
    pkill -f "llama-server.*--port $port" 2>/dev/null || true
    sleep 3
    if [[ $rc -ne 0 && "${NO_FALLBACK:-0}" != "1" ]]; then
      echo "[$(date -u +%H:%M:%S)] RETRY $label p${shape} with q8_0 KV (f16/bf16 KV did not load or run)"
      port=$((port+1))
      set +e
      "$THEROCK_PY" "$REPO_ROOT/scripts/qwen4exp_canonical_ar_bench.py" llamacpp \
        --server-bin "$build/bin/llama-server" --model "$MODEL" \
        --engine-label "${label}-kvq8" --source-root "$src" \
        --fixture "$FIXTURE_DIR/p${shape}-code.json" --output "$out" \
        --server-log "$LOGDIR/llamacpp-server-${label}-p${shape}-kvq8.log" \
        --host 127.0.0.1 --port "$port" \
        --warmups "$WARMUPS" --repetitions "$REPS" \
        --startup-timeout 900 --request-timeout 7200 \
        --server-arg=-ngl --server-arg=999 --server-arg=-fa --server-arg=on \
        --server-arg=-t --server-arg=4 --server-arg=-b --server-arg=8192 \
        --server-arg=-ub --server-arg=2048 --server-arg=-c --server-arg="$ctx" \
        --server-arg=-ctk --server-arg=q8_0 --server-arg=-ctv --server-arg=q8_0 >> "$log" 2>&1
      echo "[$(date -u +%H:%M:%S)] RETRY END $label p${shape} rc=$? "
      set -e
      pkill -f "llama-server.*--port $port" 2>/dev/null || true
      sleep 3
    fi
  done
done
echo "[$(date -u +%H:%M:%S)] llama.cpp ladder done"
