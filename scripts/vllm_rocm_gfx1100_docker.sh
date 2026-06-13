#!/usr/bin/env bash
# Run vLLM ROCm containers on RDNA3/gfx1100.
#
# Defaults target W7900 GPU 0 and a Qwen3.6 35B-A3B GPTQ Int4 checkpoint that
# includes MTP weights.  Override via environment variables; pass extra vLLM
# args after the subcommand.

set -euo pipefail

cmd="${1:-serve}"
if [[ $# -gt 0 ]]; then
  shift
fi

DOCKER_BIN="${DOCKER_BIN:-sudo docker}"
IMAGE="${VLLM_ROCM_IMAGE:-vllm/vllm-openai-rocm:latest}"
PINNED_IMAGE="${VLLM_ROCM_PINNED_IMAGE:-vllm/vllm-openai-rocm:v0.19.1}"
AMD_GFX110X_IMAGE="${VLLM_ROCM_AMD_IMAGE:-rocm/vllm:rocm7.13.0_gfx110X-all_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1}"
MODEL="${VLLM_MODEL:-palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen36-35b-a3b-gptq-mtp}"
GPU="${HIP_VISIBLE_DEVICES:-${VLLM_GPU:-0}}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
DTYPE="${VLLM_DTYPE:-float16}"
KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}"
DEFAULT_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":2}'
if [[ -v VLLM_SPECULATIVE_CONFIG ]]; then
  SPECULATIVE_CONFIG="${VLLM_SPECULATIVE_CONFIG}"
else
  SPECULATIVE_CONFIG="${DEFAULT_SPECULATIVE_CONFIG}"
fi
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
MODELS_DIR="${MODELS_DIR:-/models}"

docker_tty_args=()
case "${VLLM_DOCKER_TTY:-auto}" in
  1|true|TRUE|yes|YES|on|ON)
    docker_tty_args=(-it)
    ;;
  0|false|FALSE|no|NO|off|OFF)
    ;;
  auto|AUTO|"")
    if [[ -t 0 && -t 1 ]]; then
      docker_tty_args=(-it)
    fi
    ;;
  *)
    echo "Invalid VLLM_DOCKER_TTY=${VLLM_DOCKER_TTY}; use auto, on, or off" >&2
    exit 2
    ;;
esac

common_docker_args=(
  --rm "${docker_tty_args[@]}"
  --network=host
  --group-add=video
  --ipc=host
  --cap-add=SYS_PTRACE
  --security-opt seccomp=unconfined
  --device /dev/kfd
  --device /dev/dri
  -v "${HF_CACHE}:/root/.cache/huggingface"
  -v "${MODELS_DIR}:${MODELS_DIR}"
  -e "HIP_VISIBLE_DEVICES=${GPU}"
  -e "ROCR_VISIBLE_DEVICES=${GPU}"
  -e "CUDA_VISIBLE_DEVICES=${GPU}"
  -e "HF_HOME=/root/.cache/huggingface"
  -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}"
  -e "HF_TOKEN=${HF_TOKEN:-}"
  -e "VLLM_USE_TRITON_AWQ=1"
  -e "HSA_NO_SCRATCH_RECLAIM=1"
)

run_image() {
  local image="$1"
  shift
  # shellcheck disable=SC2086
  ${DOCKER_BIN} run "${common_docker_args[@]}" --entrypoint vllm "${image}" "$@"
}

run_shell() {
  local image="$1"
  shift
  # shellcheck disable=SC2086
  ${DOCKER_BIN} run "${common_docker_args[@]}" --entrypoint bash "${image}" "$@"
}

serve_args=(
  serve "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0
  --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
  --trust-remote-code
  --reasoning-parser qwen3
)

if [[ "${KV_CACHE_DTYPE}" != "auto" ]]; then
  serve_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
if [[ -n "${VLLM_TENSOR_PARALLEL_SIZE:-}" ]]; then
  serve_args+=(--tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}")
fi
if [[ -n "${VLLM_ENABLE_EXPERT_PARALLEL:-}" ]]; then
  serve_args+=(--enable-expert-parallel)
fi
if [[ -n "${VLLM_ENABLE_TOOL_CALLING:-1}" ]]; then
  serve_args+=(--enable-auto-tool-choice --tool-call-parser qwen3_coder)
fi
case "${SPECULATIVE_CONFIG}" in
  ""|0|false|False|off|OFF|none|None|null|nullptr)
    ;;
  *)
    serve_args+=(--speculative-config "${SPECULATIVE_CONFIG}")
    ;;
esac
serve_args+=("$@")

case "${cmd}" in
  serve)
    run_image "${IMAGE}" "${serve_args[@]}"
    ;;
  serve-amd)
    run_image "${AMD_GFX110X_IMAGE}" "${serve_args[@]}"
    ;;
  serve-pinned)
    run_image "${PINNED_IMAGE}" "${serve_args[@]}"
    ;;
  shell)
    run_shell "${IMAGE}" "$@"
    ;;
  shell-amd)
    run_shell "${AMD_GFX110X_IMAGE}" "$@"
    ;;
  shell-pinned)
    run_shell "${PINNED_IMAGE}" "$@"
    ;;
  pull)
    # shellcheck disable=SC2086
    ${DOCKER_BIN} pull "${IMAGE}"
    ;;
  pull-amd)
    # shellcheck disable=SC2086
    ${DOCKER_BIN} pull "${AMD_GFX110X_IMAGE}"
    ;;
  pull-pinned)
    # shellcheck disable=SC2086
    ${DOCKER_BIN} pull "${PINNED_IMAGE}"
    ;;
  print)
    printf 'official image: %s\n' "${IMAGE}"
    printf 'pinned official image: %s\n' "${PINNED_IMAGE}"
    printf 'amd gfx110X image: %s\n' "${AMD_GFX110X_IMAGE}"
    printf 'model: %s\n' "${MODEL}"
    printf 'gpu: %s\n' "${GPU}"
    printf 'port: %s\n' "${PORT}"
    printf 'max num batched tokens: %s\n' "${MAX_NUM_BATCHED_TOKENS}"
    printf 'speculative config: %s\n' "${SPECULATIVE_CONFIG:-disabled}"
    printf '\nServe command args:\n  vllm'
    printf ' %q' "${serve_args[@]}"
    printf '\n'
    ;;
  *)
    cat >&2 <<'EOF'
usage: scripts/vllm_rocm_gfx1100_docker.sh [serve|serve-pinned|serve-amd|shell|shell-pinned|shell-amd|pull|pull-pinned|pull-amd|print] [extra vLLM args]

Environment overrides:
  DOCKER_BIN                    default: sudo docker
  VLLM_ROCM_IMAGE               default: vllm/vllm-openai-rocm:latest
  VLLM_ROCM_PINNED_IMAGE        default: vllm/vllm-openai-rocm:v0.19.1
  VLLM_ROCM_AMD_IMAGE           default: rocm/vllm:rocm7.13.0_gfx110X-all_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1
  VLLM_MODEL                    default: palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4
  HIP_VISIBLE_DEVICES/VLLM_GPU  default: 0
  VLLM_MAX_MODEL_LEN            default: 8192
  VLLM_MAX_NUM_SEQS             default: 8
  VLLM_MAX_NUM_BATCHED_TOKENS   default: 8192
  VLLM_GPU_MEMORY_UTILIZATION   default: 0.88
  VLLM_SPECULATIVE_CONFIG       default: {"method":"mtp","num_speculative_tokens":2}; set empty/off/none to disable

Examples:
  scripts/vllm_rocm_gfx1100_docker.sh pull
  scripts/vllm_rocm_gfx1100_docker.sh serve
  scripts/vllm_rocm_gfx1100_docker.sh serve-pinned
  scripts/vllm_rocm_gfx1100_docker.sh serve-amd
EOF
    exit 2
    ;;
esac
