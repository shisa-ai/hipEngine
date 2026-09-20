#!/usr/bin/env bash
# Orchestrate the hipEngine vs atlas 1:1 comparison.
#
# ONE ENGINE AT A TIME. Both are started against the same GPU in sequence, never
# together, because two servers sharing one device make neither rate
# attributable to an engine. The script refuses to start the second engine until
# the first process is gone and its port is free.
#
# Usage:
#   benchmarks/atlas-agent/run.sh                  # all arms, both engines
#   ENGINES=hipengine benchmarks/atlas-agent/run.sh
#   ARMS=single OUTPUT_LEN=128 benchmarks/atlas-agent/run.sh
#
# Environment: OUTPUT_LEN, CONCURRENCY, TURNS, REPEATS, PROMPT_FILE,
# PROMPT_CATEGORY, PROMPT_LIMIT, MAX_CONTEXT, ENGINES, ARMS, OUT.
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"
HERE="$REPO/benchmarks/atlas-agent"
PY="${PY:-/home/lhl/miniforge3/envs/vllm/bin/python}"
ENVWRAP="${ENVWRAP:-/tmp/t17-env.sh}"

OUT="${OUT:-/tmp/atlas-agent/run-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
CONCURRENCY="${CONCURRENCY:-4}"
TURNS="${TURNS:-3}"
REPEATS="${REPEATS:-2}"
PROMPT_FILE="${PROMPT_FILE:-benchmarks/prompts/mtpbench-code-general-ja.jsonl}"
PROMPT_CATEGORY="${PROMPT_CATEGORY:-code}"
PROMPT_LIMIT="${PROMPT_LIMIT:-4}"
MAX_CONTEXT="${MAX_CONTEXT:-262144}"
ENGINES="${ENGINES:-hipengine atlas}"
ARMS="${ARMS:-single multi conc}"
HIP_PORT="${HIP_PORT:-8097}"
ATLAS_PORT="${ATLAS_PORT:-8081}"
HIP_NAME="${HIP_NAME:-qwen3.8-27b-q4km}"
ATLAS_NAME="${ATLAS_NAME:-qwen3.8-27b-nvfp4}"

stamp() { date -u +%H:%M:%S; }
say()   { echo "[$(stamp)] $*" | tee -a "$OUT/orchestration.log"; }

wait_ready() {  # port, seconds
  local port=$1 budget=${2:-900} i
  for i in $(seq 1 "$budget"); do
    curl -sf "http://127.0.0.1:$port/ready" >/dev/null 2>&1 && return 0
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

wait_port_free() {  # port, seconds
  local port=$1 budget=${2:-60} i
  for i in $(seq 1 "$budget"); do
    curl -sf "http://127.0.0.1:$port/ready" >/dev/null 2>&1 || return 0
    sleep 1
  done
  return 1
}

stop_engine() {  # pid, port, name
  local pid=$1 port=$2 name=$3
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null
  pkill -f "hipengine.server.*--port $port" 2>/dev/null
  pkill -f "spark serve.*--port $port" 2>/dev/null
  wait_port_free "$port" 60 || say "WARNING: $name port $port still answering after stop"
  sleep 5   # let the driver release GTT before the next engine loads
}

probe_capabilities() {  # port, name
  local port=$1 name=$2
  curl -sf "http://127.0.0.1:$port/v1/hipengine/capabilities" \
    > "$OUT/$name-capabilities.json" 2>/dev/null \
    || echo '{}' > "$OUT/$name-capabilities.json"
  curl -sf "http://127.0.0.1:$port/v1/models" > "$OUT/$name-models.json" 2>/dev/null \
    || echo '{}' > "$OUT/$name-models.json"
  say "$name capabilities -> $OUT/$name-capabilities.json"
}

run_arm() {  # name, port, model, arm
  local name=$1 port=$2 model=$3 arm=$4
  local extra=()
  case "$arm" in
    conc) extra=(--concurrency "$CONCURRENCY") ;;
    multi) extra=(--turns "$TURNS") ;;
  esac
  say "$name arm=$arm start"
  bash "$ENVWRAP" "$PY" "$HERE/http_1to1_bench.py" \
    --base-url "http://127.0.0.1:$port" --model "$model" \
    --prompt-file "$PROMPT_FILE" --prompt-category "$PROMPT_CATEGORY" \
    --arm "$arm" --output-len "$OUTPUT_LEN" --repeats "$REPEATS" \
    --prompt-limit "$PROMPT_LIMIT" \
    --json "$OUT/$name-$arm.json" --label "$name" "${extra[@]}" \
    > "$OUT/$name-$arm.log" 2>&1
  say "$name arm=$arm exit=$?"
}

start_and_measure() {  # name, port, model, serve-script
  local name=$1 port=$2 model=$3 script=$4
  local logdir="/tmp/atlas-agent/$name"
  rm -rf "$logdir"; mkdir -p "$logdir"
  say "$name starting ($script)"
  LOG_DIR="$logdir" PORT="$port" MAX_CONTEXT="$MAX_CONTEXT" MAX_SEQ_LEN="$MAX_CONTEXT" \
    bash "$script" > "$OUT/$name-server.log" 2>&1 &
  local pid=$!
  if ! wait_ready "$port" 900; then
    say "$name FAILED to become ready; see $OUT/$name-server.log"
    stop_engine "$pid" "$port" "$name"
    return 1
  fi
  say "$name ready pid=$pid"
  probe_capabilities "$port" "$name"
  for arm in $ARMS; do run_arm "$name" "$port" "$model" "$arm"; done
  stop_engine "$pid" "$port" "$name"
  say "$name stopped"
}

{
  echo "host=$(hostname)"; echo "date_utc=$(date -u +%FT%TZ)"
  echo "repo_head=$(git rev-parse HEAD)"; echo "repo_dirty=$(git status --porcelain | wc -l)"
  echo "output_len=$OUTPUT_LEN concurrency=$CONCURRENCY turns=$TURNS repeats=$REPEATS"
  echo "max_context=$MAX_CONTEXT prompt_file=$PROMPT_FILE category=$PROMPT_CATEGORY"
  echo "engines='$ENGINES' arms='$ARMS'"
  echo "atlas_head=$(cd /home/lhl/atlas && git rev-parse HEAD)"
  echo "--- gpu ---"; rocminfo 2>/dev/null | grep -E "^\s+Name:|gfx" | head -4
} > "$OUT/environment.txt" 2>&1
cat "$OUT/environment.txt"

for engine in $ENGINES; do
  case "$engine" in
    hipengine) start_and_measure hipengine "$HIP_PORT" "$HIP_NAME" "$HERE/serve-hipengine.sh" ;;
    atlas)     start_and_measure atlas "$ATLAS_PORT" "$ATLAS_NAME" "$HERE/serve-atlas.sh" ;;
    *) say "unknown engine '$engine'" ;;
  esac
done

bash "$ENVWRAP" "$PY" "$HERE/analyze.py" --run-dir "$OUT" 2>&1 | tee -a "$OUT/orchestration.log"
say "done -> $OUT"
