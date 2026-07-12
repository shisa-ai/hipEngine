#!/usr/bin/env bash
# Reproduce the gfx1151/Radeon 8060S README model-throughput refresh.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_gfx1151_readme_refresh.sh <phase>

Phases:
  hipengine   hipEngine PARO + GGUF Q4_K_M resident sweeps
  llamacpp    llama.cpp HIP + Vulkan Q4_K_M split sweeps
  summary     validate and assemble the four-column README topline
  all         hipengine + llamacpp + summary

Useful overrides:
  RUN_TAG=20260711-120000
  OUTDIR=/home/lhl/hipEngine-main/benchmarks/results
  LOGDIR=/tmp/hipengine-readme-gfx1151/$RUN_TAG
  REPO_ROOT=/tmp/clean-hipengine-worktree
  THEROCK_PY=/home/lhl/miniforge3/envs/therock/bin/python3.12
  PARO_MODEL=/path/to/Qwen3.6-35B-A3B-PARO-packed/snapshot
  GGUF_Q4KM_MODEL=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
EOF
}

phase="${1:-all}"
if [[ "$phase" == "-h" || "$phase" == "--help" || "$phase" == "help" ]]; then
  usage
  exit 0
fi
case "$phase" in
  hipengine|llamacpp|summary|all) ;;
  *) usage >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUTDIR="${OUTDIR:-$REPO_ROOT/benchmarks/results}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
LOGDIR="${LOGDIR:-/tmp/hipengine-readme-gfx1151/$RUN_TAG}"
DATE_PREFIX="${DATE_PREFIX:-$(date -u +%Y-%m-%d)-gfx1151-readme-refresh-$RUN_TAG}"
mkdir -p "$OUTDIR" "$LOGDIR"

if ! git -C "$REPO_ROOT" diff --quiet --no-ext-diff ||
   ! git -C "$REPO_ROOT" diff --cached --quiet --no-ext-diff; then
  echo "ERROR: retained gfx1151 refresh requires a tracked-clean worktree" >&2
  exit 2
fi
if [[ "${ALLOW_UNTRACKED:-0}" != "1" ]] &&
   [[ -n "$(git -C "$REPO_ROOT" ls-files --others --exclude-standard)" ]]; then
  echo "ERROR: retained gfx1151 refresh requires no untracked files" >&2
  exit 2
fi

THEROCK_PY="${THEROCK_PY:-/home/lhl/miniforge3/envs/therock/bin/python3.12}"
THEROCK_ROOT="${THEROCK_ROOT:-$("$THEROCK_PY" -m rocm_sdk path --root)}"
THEROCK_SITE=$("$THEROCK_PY" -c 'import site; print(site.getsitepackages()[0])')
THEROCK_CORE_LIB="$THEROCK_SITE/_rocm_sdk_core/lib"
THEROCK_GFX_LIB="$THEROCK_SITE/_rocm_sdk_libraries_gfx1151/lib"
HIPCC_VERSION_FILE="${HIPCC_VERSION_FILE:-$LOGDIR/hipcc-version-gfx1151.txt}"
TIMEOUT_LONG="${TIMEOUT_LONG:-21600}"
TIMEOUT_SHORT="${TIMEOUT_SHORT:-7200}"

PARO_MODEL="${PARO_MODEL:-/home/lhl/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1}"
GGUF_Q4KM_MODEL="${GGUF_Q4KM_MODEL:-/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
LLAMACPP_HIP_BENCH="${LLAMACPP_HIP_BENCH:-/home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-bench}"
LLAMACPP_VULKAN_BENCH="${LLAMACPP_VULKAN_BENCH:-/home/lhl/llama.cpp/llama.cpp-vulkan/build/bin/llama-bench}"
AMDGPU_CARD_NAME="${AMDGPU_CARD_NAME:-card1}"

"$THEROCK_ROOT/bin/hipcc" --version > "$HIPCC_VERSION_FILE"

git -C "$REPO_ROOT" status -sb > "$LOGDIR/git-status.txt"
{
  echo "run_tag=$RUN_TAG"
  echo "date_prefix=$DATE_PREFIX"
  echo "repo_root=$REPO_ROOT"
  echo "git_head=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "therock_py=$THEROCK_PY"
  echo "therock_root=$THEROCK_ROOT"
  echo "paro_model=$PARO_MODEL"
  echo "gguf_q4km_model=$GGUF_Q4KM_MODEL"
  echo "llamacpp_hip_bench=$LLAMACPP_HIP_BENCH"
  echo "llamacpp_vulkan_bench=$LLAMACPP_VULKAN_BENCH"
  echo "amdgpu_card=$AMDGPU_CARD_NAME"
  echo "memory_domain=gtt"
  rocminfo | grep -E 'Name:|gfx' | head -12
} > "$LOGDIR/environment.txt"

THEROCK_ENV=(
  env -i
  HOME="$HOME"
  USER="${USER:-$(id -un)}"
  LOGNAME="${LOGNAME:-${USER:-$(id -un)}}"
  SHELL="${SHELL:-/bin/bash}"
  TERM="${TERM:-xterm}"
  PATH="$THEROCK_ROOT/bin:$(dirname "$THEROCK_PY"):/usr/local/bin:/usr/bin:/bin"
  LD_LIBRARY_PATH="$THEROCK_ROOT/lib:$THEROCK_CORE_LIB:$THEROCK_GFX_LIB"
  HIP_PATH="$THEROCK_ROOT"
  ROCM_PATH="$THEROCK_ROOT"
  HIP_LIB_PATH="$THEROCK_ROOT/lib"
  HIP_INCLUDE_PATH="$THEROCK_ROOT/include"
  HIPENGINE_HIP_ARCH=gfx1151
  HIPENGINE_COMPILER_VERSION_FILE="$HIPCC_VERSION_FILE"
  PYTHONPATH="$REPO_ROOT"
)

compact_readme_sweep_json() {
  local path="$1"
  "$THEROCK_PY" - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
for runs in data.get("runs_by_workload", {}).values():
    for run in runs:
        run.pop("memory_snapshots", None)
        memory = run.get("memory")
        if isinstance(memory, dict):
            memory.pop("kv_memory_audit", None)
for key in ("snapshots",):
    if isinstance(data.get("persistent_session_memory"), dict):
        data["persistent_session_memory"].pop(key, None)
data.setdefault("notes", []).append(
    "Compacted by scripts/run_gfx1151_readme_refresh.sh: verbose memory "
    "snapshots omitted; numeric summaries, timings, correctness, and provenance retained."
)
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

run_prebuild() {
  if [[ "${SKIP_HIPENGINE_PREBUILD:-0}" == "1" ]]; then
    return
  fi
  echo "[prebuild] PARO gfx1151" | tee -a "$LOGDIR/run.log"
  "${THEROCK_ENV[@]}" timeout "$TIMEOUT_SHORT" "$THEROCK_PY" \
    "$REPO_ROOT/scripts/qwen35_readme_sweep.py" \
    --engine paro --model "$PARO_MODEL" --backend hip_gfx1151 \
    --shared-expert-format packed_paro_w4 --token-id 9707 \
    --workloads 512/1 --warmup-runs 0 --measured-runs 1 \
    --warmup-decode-tokens 1 --attn-aotriton-min-tokens 512 \
    --graph-replay-decode --compiler-version-file "$HIPCC_VERSION_FILE" \
    --json "$LOGDIR/prebuild-paro.json" > "$LOGDIR/prebuild-paro.log" 2>&1

  echo "[prebuild] GGUF gfx1151" | tee -a "$LOGDIR/run.log"
  "${THEROCK_ENV[@]}" HIPENGINE_GGUF_DECODE_REPACK=1 \
    timeout "$TIMEOUT_SHORT" "$THEROCK_PY" \
    "$REPO_ROOT/scripts/qwen35_readme_sweep.py" \
    --engine gguf --model "$GGUF_Q4KM_MODEL" --quant gguf_q4_k_m \
    --backend hip_gfx1151 --workloads 512/1 --warmup-runs 0 \
    --measured-runs 1 --warmup-decode-tokens 1 --force-bulk-prefill \
    --bulk-prefill-attention-mode bulk --use-wmma-prefill --use-gemv-decode \
    --no-graph-replay-decode --compiler-version-file "$HIPCC_VERSION_FILE" \
    --json "$LOGDIR/prebuild-gguf.json" > "$LOGDIR/prebuild-gguf.log" 2>&1
}

run_hipengine() {
  run_prebuild
  local paro_json="$OUTDIR/${DATE_PREFIX}-hipengine-paro-packed-5run.json"
  local gguf_json="$OUTDIR/${DATE_PREFIX}-hipengine-gguf-q4km-5run.json"
  local workloads=(512/128 1K/128 4K/128 32K/128 64K/128 128K/128)
  local paro_components=()
  local gguf_components=()
  local workload safe component

  echo "[run] hipEngine PARO -> $paro_json" | tee -a "$LOGDIR/run.log"
  for workload in "${workloads[@]}"; do
    safe=${workload//\//-}
    component="$LOGDIR/hipengine-paro-$safe.json"
    paro_components+=("$component")
    echo "[run] hipEngine PARO $workload" | tee -a "$LOGDIR/run.log"
    "${THEROCK_ENV[@]}" timeout "$TIMEOUT_LONG" "$THEROCK_PY" \
      "$REPO_ROOT/scripts/qwen35_readme_sweep.py" \
      --engine paro --model "$PARO_MODEL" --backend hip_gfx1151 \
      --shared-expert-format packed_paro_w4 --token-id 9707 \
      --workloads "$workload" \
      --warmup-runs 2 --measured-runs 5 --warmup-decode-tokens 4 \
      --compiler-version-file "$HIPCC_VERSION_FILE" --require-cached-build \
      --attn-aotriton-min-tokens 512 --graph-replay-decode \
      --json "$component" > "$LOGDIR/hipengine-paro-$safe.log" 2>&1
    compact_readme_sweep_json "$component"
  done
  "${THEROCK_ENV[@]}" "$THEROCK_PY" \
    "$REPO_ROOT/scripts/merge_readme_sweep_components.py" \
    --engine paro --components "${paro_components[@]}" --json "$paro_json" \
    > "$LOGDIR/hipengine-paro-merge.log" 2>&1

  echo "[run] hipEngine GGUF -> $gguf_json" | tee -a "$LOGDIR/run.log"
  for workload in "${workloads[@]}"; do
    safe=${workload//\//-}
    component="$LOGDIR/hipengine-gguf-$safe.json"
    gguf_components+=("$component")
    echo "[run] hipEngine GGUF $workload" | tee -a "$LOGDIR/run.log"
    "${THEROCK_ENV[@]}" HIPENGINE_GGUF_DECODE_REPACK=1 \
      timeout "$TIMEOUT_LONG" "$THEROCK_PY" \
      "$REPO_ROOT/scripts/qwen35_readme_sweep.py" \
      --engine gguf --model "$GGUF_Q4KM_MODEL" --quant gguf_q4_k_m \
      --backend hip_gfx1151 --workloads "$workload" \
      --warmup-runs 2 --measured-runs 5 --warmup-decode-tokens 1 \
      --force-bulk-prefill --bulk-prefill-attention-mode bulk \
      --use-wmma-prefill --use-gemv-decode --graph-replay-decode \
      --compiler-version-file "$HIPCC_VERSION_FILE" --require-cached-build \
      --json "$component" > "$LOGDIR/hipengine-gguf-$safe.log" 2>&1
    compact_readme_sweep_json "$component"
  done
  "${THEROCK_ENV[@]}" "$THEROCK_PY" \
    "$REPO_ROOT/scripts/merge_readme_sweep_components.py" \
    --engine gguf --components "${gguf_components[@]}" --json "$gguf_json" \
    > "$LOGDIR/hipengine-gguf-merge.log" 2>&1
}

run_llamacpp() {
  local hip_json="$OUTDIR/${DATE_PREFIX}-llamacpp-hip-q4km-f16kv.json"
  local vulkan_json="$OUTDIR/${DATE_PREFIX}-llamacpp-vulkan-q4km-f16kv.json"

  echo "[run] llama.cpp HIP -> $hip_json" | tee -a "$LOGDIR/run.log"
  PYTHONPATH="$REPO_ROOT" timeout "$TIMEOUT_LONG" "$THEROCK_PY" \
    "$REPO_ROOT/scripts/llamacpp_bench_with_peak.py" \
    --llama-bench "$LLAMACPP_HIP_BENCH" --model "$GGUF_Q4KM_MODEL" \
    --backend hip \
    --workloads 512/128 1K/128 4K/128 32K/128 64K/128 128K/128 \
    --repetitions 5 --ngl 99 --flash-attn 1 --cache-type-k f16 \
    --cache-type-v f16 --poll 10 --card-name "$AMDGPU_CARD_NAME" \
    --memory-domain gtt --extra-args "-dev ROCm0" --output "$hip_json" \
    > "$LOGDIR/llamacpp-hip.log" 2>&1

  echo "[run] llama.cpp Vulkan -> $vulkan_json" | tee -a "$LOGDIR/run.log"
  PYTHONPATH="$REPO_ROOT" timeout "$TIMEOUT_LONG" "$THEROCK_PY" \
    "$REPO_ROOT/scripts/llamacpp_bench_with_peak.py" \
    --llama-bench "$LLAMACPP_VULKAN_BENCH" --model "$GGUF_Q4KM_MODEL" \
    --backend vulkan \
    --workloads 512/128 1K/128 4K/128 32K/128 64K/128 128K/128 \
    --repetitions 5 --ngl 99 --flash-attn 1 --cache-type-k f16 \
    --cache-type-v f16 --poll 10 --card-name "$AMDGPU_CARD_NAME" \
    --memory-domain gtt --extra-args "-dev Vulkan0" --output "$vulkan_json" \
    > "$LOGDIR/llamacpp-vulkan.log" 2>&1
}

run_summary() {
  local paro_json="$OUTDIR/${DATE_PREFIX}-hipengine-paro-packed-5run.json"
  local gguf_json="$OUTDIR/${DATE_PREFIX}-hipengine-gguf-q4km-5run.json"
  local hip_json="$OUTDIR/${DATE_PREFIX}-llamacpp-hip-q4km-f16kv.json"
  local vulkan_json="$OUTDIR/${DATE_PREFIX}-llamacpp-vulkan-q4km-f16kv.json"
  local summary_json="$OUTDIR/${DATE_PREFIX}-summary.json"

  echo "[run] four-column promotion gate -> $summary_json" | tee -a "$LOGDIR/run.log"
  "${THEROCK_ENV[@]}" "$THEROCK_PY" \
    "$REPO_ROOT/scripts/assemble_gfx1151_readme_topline.py" \
    --hipengine-paro "$paro_json" --hipengine-gguf "$gguf_json" \
    --llamacpp-hip "$hip_json" --llamacpp-vulkan "$vulkan_json" \
    --json "$summary_json" --markdown "$LOGDIR/topline-tables.md" \
    > "$LOGDIR/topline-summary.log" 2>&1
}

case "$phase" in
  hipengine) run_hipengine ;;
  llamacpp) run_llamacpp ;;
  summary) run_summary ;;
  all) run_hipengine; run_llamacpp; run_summary ;;
esac

echo "[done] phase=$phase run_tag=$RUN_TAG logdir=$LOGDIR outdir=$OUTDIR" | tee -a "$LOGDIR/run.log"
