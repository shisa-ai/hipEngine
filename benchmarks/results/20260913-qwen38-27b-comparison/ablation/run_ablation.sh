#!/usr/bin/env bash
# Reproduce the Qwen3.8-27B Q4_K_M prefill-lead attribution on gfx1151.
#
#   1. build upstream llama.cpp + ONLY the fork's ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh
#   2. measure prefill with the same harness used for the engine comparison
#   3. kernel-trace all three binaries at 4K prefill and diff the kernel families
#
# Prereqs: /home/lhl/llama.cpp/llama.cpp-hip (upstream checkout at UPSTREAM_PIN with
#          build/bin/llama-bench built), /tmp/hipengine-halobox-head-20260913
#          (halo-box/strix-llama.cpp checkout at FORK_PIN with build-hip/bin),
#          conda env "therock" with rocprofv3, model at /models/gguf/Qwen3.8-27B-Q4_K_M.gguf.
#
# The recorded pins are asserted, not assumed: both source trees are archived at the
# pinned revision, and the fork's tile table is read from that revision rather than from
# the working tree. Set ALLOW_PIN_MISMATCH=1 only for a deliberately different rerun.
set -euo pipefail

WORK=${WORK:-/tmp/qwen38-ablation}
UPSTREAM=${UPSTREAM:-/home/lhl/llama.cpp/llama.cpp-hip}
FORK=${FORK:-/tmp/hipengine-halobox-head-20260913}
MODEL=${MODEL:-/models/gguf/Qwen3.8-27B-Q4_K_M.gguf}
HIPENGINE=${HIPENGINE:-/home/lhl/hipEngine}
UPSTREAM_PIN=${UPSTREAM_PIN:-002a12ad25503a93501b2e188c360029830a241a}
FORK_PIN=${FORK_PIN:-654803517b06da47f5210553a661bf6c80deb97f}

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate therock
ROCM_ROOT=$(python -m rocm_sdk path --root)
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib64:$ROCM_ROOT/lib/llvm/lib:${LD_LIBRARY_PATH:-}"

# --- 0. assert the recorded pins ---------------------------------------------------------------
for spec in "upstream:$UPSTREAM:$UPSTREAM_PIN" "fork:$FORK:$FORK_PIN"; do
    name=${spec%%:*}; rest=${spec#*:}; tree=${rest%%:*}; pin=${rest#*:}
    head=$(git -C "$tree" rev-parse HEAD)
    if [ "$head" != "$pin" ]; then
        echo "$name checkout is at $head, not the recorded pin $pin" >&2
        if [ "${ALLOW_PIN_MISMATCH:-0}" != "1" ]; then
            echo "set ALLOW_PIN_MISMATCH=1 to override (results then describe a different revision)" >&2
            exit 1
        fi
    fi
done

# --- 1. upstream source + the fork's MMQ config table only ------------------------------------
mkdir -p "$WORK/src"
rm -rf "${WORK:?}/src"
mkdir -p "$WORK/src"git -C "$UPSTREAM" archive --format=tar "$UPSTREAM_PIN" | tar -xf - -C "$WORK/src"
git -C "$FORK" show "$FORK_PIN:ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh" \
    > "$WORK/src/ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh"
REPO_DIR="$WORK/src" BUILD_DIR="$WORK/build-hip" JOBS=16 bash /home/lhl/llama.cpp/build-hip.sh

# --- 2. prefill measurements ------------------------------------------------------------------
for spec in "upstream-hip:$UPSTREAM/build/bin/llama-bench" \
            "upstream-hip-mmqconfig:$WORK/build-hip/bin/llama-bench" \
            "halobox-hip:$FORK/build-hip/bin/llama-bench"; do
    name=${spec%%:*}; bin=${spec#*:}
    (cd "$HIPENGINE" && .venv/bin/python scripts/llamacpp_bench_with_peak.py --llama-bench "$bin" \
        --model "$MODEL" --quant gguf_q4_k_m --backend hip \
        --workloads 512/128 1K/128 4K/128 --phases prefill --repetitions 5 \
        --ngl 99 --flash-attn 1 --cache-type-k bf16 --cache-type-v bf16 \
        --poll 10 --card-name card1 --memory-domain gtt --extra-args "-dev ROCm0" \
        --note "MMQ config ablation, $name" --output "$WORK/$name-prefill.json")
done

# --- 3. kernel traces -------------------------------------------------------------------------
for spec in "upstream:$UPSTREAM/build/bin/llama-bench" \
            "upstream-mmqconfig:$WORK/build-hip/bin/llama-bench" \
            "halobox:$FORK/build-hip/bin/llama-bench"; do
    name=${spec%%:*}; bin=${spec#*:}
    rocprofv3 --kernel-trace --output-format csv -d "$WORK/trace/$name" -- \
        "$bin" -m "$MODEL" -ngl 99 -fa 1 -ctk bf16 -ctv bf16 -r 1 -p 4096 -n 0 -d 0 -dev ROCm0
done

# --- 4. attribution report --------------------------------------------------------------------
cd "$(dirname "$0")"
python3 compare_kernel_traces.py "$WORK"/trace/upstream/gfx1151/*_kernel_trace.csv \
                                 "$WORK"/trace/halobox/gfx1151/*_kernel_trace.csv --top 8
python3 compare_kernel_traces.py "$WORK"/trace/upstream/gfx1151/*_kernel_trace.csv \
                                 "$WORK"/trace/upstream-mmqconfig/gfx1151/*_kernel_trace.csv --top 8
python3 launch_geometry.py "upstream=$WORK"/trace/upstream/gfx1151/*_kernel_trace.csv \
                           "upstream+config=$WORK"/trace/upstream-mmqconfig/gfx1151/*_kernel_trace.csv \
                           "strix-llama.cpp=$WORK"/trace/halobox/gfx1151/*_kernel_trace.csv
