#!/usr/bin/env bash
# Reproduce the W7900/GPU0 README performance refresh.
#
# This script intentionally pins local paths, GPU device selectors, model paths,
# benchmark flags, and the hermetic TheRock ROCm runtime environment used by the
# README comparison tables. Use this wrapper for retained W7900 hipEngine rows:
# direct shell runs that only point at the TheRock Python can silently inherit the
# ambient ROCm stack and under-report GGUF prefill while decode stays normal. Run
# it from a clean worktree when producing retained/diagnostic rows. When running
# from a detached worktree, set OUTDIR=/home/lhl/hipEngine/benchmarks/results if
# artifacts should be written back to the main checkout.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_w7900_readme_refresh.sh <phase>

Phases:
  hipengine       hipEngine PARO + GGUF Q4_K_M resident README sweeps
  llamacpp        llama.cpp HIP + Vulkan Q4_K_M split prefill/decode sweeps
  concurrency     hipEngine + llama.cpp Vulkan concurrency sweeps
  vllm-server     start the pinned local vLLM W7900 server and wait for readiness
  vllm-client     run the vLLM OpenAI concurrency client against an existing server
  vllm            start vLLM server, run client, then stop the server
  all             hipengine + llamacpp + concurrency + vllm

Useful overrides:
  RUN_TAG=20260614-141414
  OUTDIR=/home/lhl/hipEngine/benchmarks/results
  LOGDIR=/tmp/hipengine-readme-runs/$RUN_TAG
  REPO_ROOT=/tmp/clean-hipengine-worktree
  HIP_VISIBLE_DEVICES_W7900=0
  AMDGPU_CARD_NAME_W7900=card1
  PARO_MODEL=/path/to/Qwen3.6-35B-A3B-PARO-packed/snapshot
  LLAMACPP_Q4KM_MODEL=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  VLLM_PORT=8008
EOF
}

phase="${1:-all}"
if [[ "$phase" == "-h" || "$phase" == "--help" || "$phase" == "help" ]]; then
  usage
  exit 0
fi

case "$phase" in
  hipengine|llamacpp|concurrency|vllm-server|vllm-client|vllm|all) ;;
  *)
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUTDIR="${OUTDIR:-$REPO_ROOT/benchmarks/results}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
LOGDIR="${LOGDIR:-/tmp/hipengine-readme-runs/$RUN_TAG}"
DATE_PREFIX="${DATE_PREFIX:-$(date -u +%Y-%m-%d)-w7900-gpu0-readme-refresh-$RUN_TAG}"
mkdir -p "$OUTDIR" "$LOGDIR"

HIP_VISIBLE_DEVICES_W7900="${HIP_VISIBLE_DEVICES_W7900:-0}"
AMDGPU_CARD_NAME_W7900="${AMDGPU_CARD_NAME_W7900:-card1}"
THEROCK_PY="${THEROCK_PY:-/home/lhl/mambaforge/envs/therock/bin/python3.12}"
THEROCK_ROOT="${THEROCK_ROOT:-$("$THEROCK_PY" -m rocm_sdk path --root)}"
HIPCC_VERSION_FILE="${HIPCC_VERSION_FILE:-$LOGDIR/hipcc-version-713.txt}"

PARO_MODEL="${PARO_MODEL:-/home/lhl/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1}"
GGUF_Q4KM_MODEL="${GGUF_Q4KM_MODEL:-/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
DEFAULT_LLAMACPP_Q4KM_MODEL="$REPO_ROOT/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
if [[ ! -e "$DEFAULT_LLAMACPP_Q4KM_MODEL" && -e /home/lhl/hipEngine/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf ]]; then
  # Clean detached worktrees usually do not contain the large untracked GGUF;
  # keep REPO_ROOT as the portable default, with a host-local fallback for the
  # exact W7900 README refresh setup.
  DEFAULT_LLAMACPP_Q4KM_MODEL=/home/lhl/hipEngine/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
fi
LLAMACPP_Q4KM_MODEL="${LLAMACPP_Q4KM_MODEL:-$DEFAULT_LLAMACPP_Q4KM_MODEL}"
FIXTURE="${FIXTURE:-/tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json}"

LLAMACPP_HIP_BENCH="${LLAMACPP_HIP_BENCH:-/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-bench}"
LLAMACPP_VULKAN_BENCH="${LLAMACPP_VULKAN_BENCH:-/home/lhl/llama.cpp/llama.cpp-vulkan/build/bin/llama-bench}"
LLAMACPP_VULKAN_REPO="${LLAMACPP_VULKAN_REPO:-/home/lhl/llama.cpp/llama.cpp-vulkan}"
LLAMACPP_VULKAN_SERVER="${LLAMACPP_VULKAN_SERVER:-$LLAMACPP_VULKAN_REPO/build/bin/llama-server}"

VLLM_BIN="${VLLM_BIN:-/home/lhl/mambaforge/envs/vllm/bin/vllm}"
VLLM_PY="${VLLM_PY:-/home/lhl/mambaforge/envs/vllm/bin/python}"
VLLM_MODEL="${VLLM_MODEL:-palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4}"
VLLM_SERVED_MODEL="${VLLM_SERVED_MODEL:-qwen36-gptq-int4}"
VLLM_PORT="${VLLM_PORT:-8008}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:$VLLM_PORT}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-128000}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"

TIMEOUT_LONG="${TIMEOUT_LONG:-21600}"
TIMEOUT_SHORT="${TIMEOUT_SHORT:-7200}"

"$THEROCK_ROOT/bin/hipcc" --version > "$HIPCC_VERSION_FILE"

cat > "$LOGDIR/env-capture.txt" <<EOF
run_tag=$RUN_TAG
repo_root=$REPO_ROOT
outdir=$OUTDIR
date_prefix=$DATE_PREFIX
git_head=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || true)
git_status=$(git -C "$REPO_ROOT" status -sb 2>/dev/null || true)
therock_py=$THEROCK_PY
therock_root=$THEROCK_ROOT
hip_visible_devices=$HIP_VISIBLE_DEVICES_W7900
amdgpu_card=$AMDGPU_CARD_NAME_W7900
paro_model=$PARO_MODEL
gguf_q4km_model=$GGUF_Q4KM_MODEL
llamacpp_q4km_model=$LLAMACPP_Q4KM_MODEL
llamacpp_hip_bench=$LLAMACPP_HIP_BENCH
llamacpp_vulkan_bench=$LLAMACPP_VULKAN_BENCH
vllm_bin=$VLLM_BIN
vllm_model=$VLLM_MODEL
EOF

THEROCK_ENV=(
  env -i
  HOME="$HOME"
  USER="${USER:-$(id -un)}"
  LOGNAME="${LOGNAME:-${USER:-$(id -un)}}"
  SHELL="${SHELL:-/bin/bash}"
  TERM="${TERM:-xterm}"
  PATH="$THEROCK_ROOT/bin:/home/lhl/mambaforge/envs/therock/bin:/usr/local/bin:/usr/bin:/bin"
  LD_LIBRARY_PATH="$THEROCK_ROOT/lib:/home/lhl/mambaforge/envs/therock/lib/python3.12/site-packages/_rocm_sdk_core/lib:/home/lhl/mambaforge/envs/therock/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx110X_all/lib"
  HIP_PATH="$THEROCK_ROOT"
  ROCM_PATH="$THEROCK_ROOT"
  HIP_LIB_PATH="$THEROCK_ROOT/lib"
  HIP_INCLUDE_PATH="$THEROCK_ROOT/include"
  HSA_OVERRIDE_GFX_VERSION=11.0.0
  HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES_W7900"
  HIPENGINE_HIP_ARCH=gfx1100
  HIPENGINE_COMPILER_VERSION_FILE="$HIPCC_VERSION_FILE"
  PYTHONPATH="$REPO_ROOT"
)

compact_readme_sweep_json() {
  local path="$1"
  python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())

NUMERIC_MEMORY_KEYS = (
    "tracked_peak_allocated_bytes",
    "tracked_peak_allocated_gib",
    "tracked_current_allocated_bytes_before_close",
    "tracked_current_allocated_gib_before_close",
    "tracked_current_allocated_bytes_after_close",
    "tracked_current_allocated_gib_after_close",
    "owned_session_peak_bytes",
    "owned_session_peak_gib",
    "hip_used_peak_sampled_bytes",
    "hip_used_peak_sampled_gib",
)


def compact_memory(memory):
    if not isinstance(memory, dict):
        return memory
    compact = {key: memory[key] for key in NUMERIC_MEMORY_KEYS if key in memory}
    audit = memory.get("kv_memory_audit")
    if isinstance(audit, dict):
        compact_audit = {key: audit[key] for key in ("passed", "latest_label") if key in audit}
        latest = audit.get("latest")
        if isinstance(latest, dict):
            compact_audit["latest"] = {
                key: latest[key]
                for key in (
                    "required",
                    "passed",
                    "kv_storage_dtype",
                    "payload_bytes",
                    "scale_bytes",
                    "total_bytes",
                )
                if key in latest
            }
        compact["kv_memory_audit"] = compact_audit
    if memory.get("notes"):
        compact["notes"] = memory["notes"]
    return compact

if isinstance(data.get("persistent_session_memory"), dict):
    data["persistent_session_memory"] = compact_memory(data["persistent_session_memory"])

runs_by_workload = data.get("runs_by_workload")
if isinstance(runs_by_workload, dict):
    for runs in runs_by_workload.values():
        if not isinstance(runs, list):
            continue
        for run in runs:
            if not isinstance(run, dict):
                continue
            run.pop("memory_snapshots", None)
            if isinstance(run.get("memory"), dict):
                run["memory"] = compact_memory(run["memory"])

note = (
    "Artifact compacted by scripts/run_w7900_readme_refresh.sh: verbose per-run "
    "memory_snapshots and retained-KV buffer inventories omitted; summary_by_workload, "
    "run timings/throughput, correctness sanity fields, and numeric memory peaks retained."
)
notes = data.setdefault("notes", [])
if isinstance(notes, list) and note not in notes:
    notes.append(note)

path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
PY
}

run_prebuild_hipengine() {
  if [[ "${SKIP_HIPENGINE_PREBUILD:-0}" == "1" ]]; then
    return
  fi
  echo "[prebuild] hipEngine PARO cache" | tee -a "$LOGDIR/run.log"
  "${THEROCK_ENV[@]}" timeout "$TIMEOUT_SHORT" "$THEROCK_PY" "$REPO_ROOT/scripts/qwen35_readme_sweep.py" \
    --engine paro --model "$PARO_MODEL" --backend hip_gfx1100 \
    --shared-expert-format packed_paro_w4 --token-id 9707 \
    --workloads 512/1 --warmup-runs 0 --measured-runs 1 --warmup-decode-tokens 1 \
    --compiler-version-file "$HIPCC_VERSION_FILE" --attn-aotriton-min-tokens 512 --graph-replay-decode \
    --json "$LOGDIR/prebuild-paro.json" > "$LOGDIR/prebuild-paro.log" 2>&1

  echo "[prebuild] hipEngine GGUF cache" | tee -a "$LOGDIR/run.log"
  "${THEROCK_ENV[@]}" HIPENGINE_GGUF_DECODE_REPACK=1 timeout "$TIMEOUT_SHORT" "$THEROCK_PY" "$REPO_ROOT/scripts/qwen35_readme_sweep.py" \
    --engine gguf --model "$GGUF_Q4KM_MODEL" --quant gguf_q4_k_m \
    --workloads 512/1 --warmup-runs 0 --measured-runs 1 --warmup-decode-tokens 1 \
    --force-bulk-prefill --bulk-prefill-attention-mode bulk \
    --use-wmma-prefill --use-gemv-decode --compiler-version-file "$HIPCC_VERSION_FILE" \
    --json "$LOGDIR/prebuild-gguf.json" > "$LOGDIR/prebuild-gguf.log" 2>&1
}

run_hipengine() {
  run_prebuild_hipengine
  local paro_json="$OUTDIR/${DATE_PREFIX}-hipengine-paro-packed-5run.json"
  local gguf_json="$OUTDIR/${DATE_PREFIX}-hipengine-gguf-q4km-5run.json"
  echo "[run] hipEngine PARO -> $paro_json" | tee -a "$LOGDIR/run.log"
  "${THEROCK_ENV[@]}" timeout "$TIMEOUT_LONG" "$THEROCK_PY" "$REPO_ROOT/scripts/qwen35_readme_sweep.py" \
    --engine paro --model "$PARO_MODEL" --backend hip_gfx1100 \
    --shared-expert-format packed_paro_w4 --token-id 9707 \
    --workloads 512/128 1K/128 4K/128 32K/128 64K/128 128K/128 \
    --warmup-runs 2 --measured-runs 5 --warmup-decode-tokens 4 \
    --compiler-version-file "$HIPCC_VERSION_FILE" --require-cached-build \
    --attn-aotriton-min-tokens 512 --graph-replay-decode \
    --json "$paro_json" > "$LOGDIR/hipengine-paro.log" 2>&1
  compact_readme_sweep_json "$paro_json"

  echo "[run] hipEngine GGUF Q4_K_M -> $gguf_json" | tee -a "$LOGDIR/run.log"
  "${THEROCK_ENV[@]}" HIPENGINE_GGUF_DECODE_REPACK=1 timeout "$TIMEOUT_LONG" "$THEROCK_PY" "$REPO_ROOT/scripts/qwen35_readme_sweep.py" \
    --engine gguf --model "$GGUF_Q4KM_MODEL" --quant gguf_q4_k_m \
    --workloads 512/128 1K/128 4K/128 32K/128 64K/128 128K/128 \
    --warmup-runs 2 --measured-runs 5 --warmup-decode-tokens 1 \
    --force-bulk-prefill --bulk-prefill-attention-mode bulk \
    --use-wmma-prefill --use-gemv-decode \
    --compiler-version-file "$HIPCC_VERSION_FILE" --require-cached-build \
    --json "$gguf_json" > "$LOGDIR/hipengine-gguf.log" 2>&1
  compact_readme_sweep_json "$gguf_json"

  printf 'PARO_JSON=%s\nGGUF_JSON=%s\n' "$paro_json" "$gguf_json" > "$LOGDIR/hipengine-results.env"
}

run_llamacpp() {
  local hip_json="$OUTDIR/${DATE_PREFIX}-llamacpp-hip-q4km-f16kv.json"
  local vulkan_json="$OUTDIR/${DATE_PREFIX}-llamacpp-vulkan-q4km-f16kv.json"
  echo "[run] llama.cpp HIP -> $hip_json" | tee -a "$LOGDIR/run.log"
  HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES_W7900" PYTHONPATH="$REPO_ROOT" timeout "$TIMEOUT_LONG" python3 "$REPO_ROOT/scripts/llamacpp_bench_with_peak.py" \
    --llama-bench "$LLAMACPP_HIP_BENCH" \
    --model "$LLAMACPP_Q4KM_MODEL" --backend hip \
    --workloads 512/128 1K/128 4K/128 32K/128 64K/128 128K/128 \
    --repetitions 1 --ngl 99 --flash-attn 1 \
    --cache-type-k f16 --cache-type-v f16 --poll 10 --card-name "$AMDGPU_CARD_NAME_W7900" \
    --extra-args "-dev ROCm0" \
    --output "$hip_json" > "$LOGDIR/llamacpp-hip.log" 2>&1

  echo "[run] llama.cpp Vulkan -> $vulkan_json" | tee -a "$LOGDIR/run.log"
  PYTHONPATH="$REPO_ROOT" timeout "$TIMEOUT_LONG" python3 "$REPO_ROOT/scripts/llamacpp_bench_with_peak.py" \
    --llama-bench "$LLAMACPP_VULKAN_BENCH" \
    --model "$LLAMACPP_Q4KM_MODEL" --backend vulkan \
    --workloads 512/128 1K/128 4K/128 32K/128 64K/128 128K/128 \
    --repetitions 1 --ngl 99 --flash-attn 1 \
    --cache-type-k f16 --cache-type-v f16 --poll 10 --card-name "$AMDGPU_CARD_NAME_W7900" \
    --extra-args "-dev Vulkan0" \
    --output "$vulkan_json" > "$LOGDIR/llamacpp-vulkan.log" 2>&1

  printf 'LLAMACPP_HIP_JSON=%s\nLLAMACPP_VULKAN_JSON=%s\n' "$hip_json" "$vulkan_json" > "$LOGDIR/llamacpp-results.env"
}

run_concurrency() {
  local hipengine_json="$OUTDIR/${DATE_PREFIX}-hipengine-concurrency-w7900/summary.json"
  local llamacpp_json="$OUTDIR/${DATE_PREFIX}-llamacpp-vulkan-concurrency-w7900/summary.json"
  local require_cached=()
  if [[ "${CONCURRENCY_REQUIRE_CACHED:-0}" == "1" ]]; then
    require_cached=(--require-cached-build)
  fi
  echo "[run] hipEngine concurrency -> $hipengine_json" | tee -a "$LOGDIR/run.log"
  "${THEROCK_ENV[@]}" timeout "$TIMEOUT_LONG" "$THEROCK_PY" "$REPO_ROOT/scripts/qwen35_concurrency_decode_sweep.py" \
    --model "$PARO_MODEL" \
    --fixture "$FIXTURE" \
    --compiler-version-file "$HIPCC_VERSION_FILE" \
    --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 8 \
    --concurrencies 1,2,4,8 --reps 3 \
    "${require_cached[@]}" \
    --work-dir "$LOGDIR/hipengine-concurrency-work" \
    --json "$hipengine_json" > "$LOGDIR/hipengine-concurrency.log" 2>&1

  echo "[run] llama.cpp Vulkan concurrency -> $llamacpp_json" | tee -a "$LOGDIR/run.log"
  PYTHONPATH="$REPO_ROOT" timeout "$TIMEOUT_LONG" python3 "$REPO_ROOT/scripts/llamacpp_vulkan_concurrency_sweep.py" \
    --repo "$LLAMACPP_VULKAN_REPO" \
    --server-bin "$LLAMACPP_VULKAN_SERVER" \
    --model "$GGUF_Q4KM_MODEL" \
    --fixture "$FIXTURE" \
    --gpu 0 --prompt-length 512 --decode-tokens 128 --ctx-per-seq 1024 \
    --concurrencies 1,2,4,8 --reps 3 \
    --work-dir "$LOGDIR/llamacpp-vulkan-concurrency-work" \
    --json "$llamacpp_json" > "$LOGDIR/llamacpp-vulkan-concurrency.log" 2>&1

  printf 'HIPENGINE_CONCURRENCY_JSON=%s\nLLAMACPP_VULKAN_CONCURRENCY_JSON=%s\n' "$hipengine_json" "$llamacpp_json" > "$LOGDIR/concurrency-results.env"
}

wait_for_vllm() {
  local deadline=$((SECONDS + ${VLLM_READY_TIMEOUT:-1800}))
  until "$VLLM_PY" - <<PY >/dev/null 2>&1
import json, urllib.request
with urllib.request.urlopen('$VLLM_URL/v1/models', timeout=5) as r:
    data=json.loads(r.read())
assert isinstance(data, dict)
PY
  do
    if (( SECONDS >= deadline )); then
      echo "vLLM server did not become ready; tail follows" >&2
      tail -120 "$LOGDIR/vllm-server.log" >&2 || true
      return 1
    fi
    sleep 5
  done
}

start_vllm_server() {
  if "$VLLM_PY" - <<PY >/dev/null 2>&1
import json, urllib.request
with urllib.request.urlopen('$VLLM_URL/v1/models', timeout=2) as r:
    json.loads(r.read())
PY
  then
    echo "[vLLM] existing server is already ready at $VLLM_URL" | tee -a "$LOGDIR/run.log"
    return
  fi
  echo "[vLLM] starting server at $VLLM_URL" | tee -a "$LOGDIR/run.log"
  HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES_W7900" \
  TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
  HSA_NO_SCRATCH_RECLAIM=1 \
  "$VLLM_BIN" serve "$VLLM_MODEL" \
    --host 127.0.0.1 --port "$VLLM_PORT" \
    --served-model-name "$VLLM_SERVED_MODEL" \
    --dtype "$VLLM_DTYPE" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --enforce-eager \
    > "$LOGDIR/vllm-server.log" 2>&1 &
  echo $! > "$LOGDIR/vllm-server.pid"
  wait_for_vllm
}

stop_vllm_server() {
  if [[ -f "$LOGDIR/vllm-server.pid" ]]; then
    local pid
    pid=$(cat "$LOGDIR/vllm-server.pid")
    if kill -0 "$pid" 2>/dev/null; then
      echo "[vLLM] stopping server pid=$pid" | tee -a "$LOGDIR/run.log"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  fi
}

run_vllm_client() {
  local vllm_json="$OUTDIR/${DATE_PREFIX}-vllm-localbuild-gptq-int4-concurrency-c1-c8-w7900.json"
  echo "[run] vLLM OpenAI concurrency -> $vllm_json" | tee -a "$LOGDIR/run.log"
  "$VLLM_PY" "$REPO_ROOT/scripts/vllm_openai_concurrency_sweep.py" \
    --url "$VLLM_URL" \
    --model "$VLLM_SERVED_MODEL" \
    --fixture "$FIXTURE" \
    --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 8 \
    --concurrencies 1,2,4,8 --reps 3 \
    --json "$vllm_json" > "$LOGDIR/vllm-client.log" 2>&1
  printf 'VLLM_JSON=%s\n' "$vllm_json" > "$LOGDIR/vllm-results.env"
}

case "$phase" in
  hipengine)
    run_hipengine
    ;;
  llamacpp)
    run_llamacpp
    ;;
  concurrency)
    run_concurrency
    ;;
  vllm-server)
    start_vllm_server
    ;;
  vllm-client)
    run_vllm_client
    ;;
  vllm)
    trap stop_vllm_server EXIT
    start_vllm_server
    run_vllm_client
    ;;
  all)
    trap stop_vllm_server EXIT
    run_hipengine
    run_llamacpp
    run_concurrency
    start_vllm_server
    run_vllm_client
    ;;
esac

echo "[done] phase=$phase run_tag=$RUN_TAG logdir=$LOGDIR outdir=$OUTDIR" | tee -a "$LOGDIR/run.log"
