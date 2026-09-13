#!/usr/bin/env bash
# Compare greedy outputs of the three builds on one prompt via llama-server.
set -uo pipefail
MODEL=/models/gguf/Qwen3.8-27B-Q4_K_M.gguf
PORT=${PORT:-18299}
OUT=${OUT:-/tmp/qwen38-ablation/output-compare}
mkdir -p "$OUT"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate therock
ROCM_ROOT=$(python -m rocm_sdk path --root)
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib64:$ROCM_ROOT/lib/llvm/lib:${LD_LIBRARY_PATH:-}"

run() { # name binary [extra env]
  local name="$1" bin="$2"
  echo "=== $name $(date -Is)"
  HIP_LAUNCH_BLOCKING=1 "$bin" -m "$MODEL" -ngl 99 -fa on -ctk bf16 -ctv bf16 \
    --host 127.0.0.1 --port "$PORT" --no-webui -c 512 -b 512 -ub 512 -t 4 \
    --temp 0 --seed 0 > "$OUT/$name.server.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || { echo "server exited early"; break; }
    sleep 1
  done
  curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"The capital of France is","n_predict":48,"temperature":0,"top_k":1,"cache_prompt":false,"seed":0,"return_tokens":true}' \
    > "$OUT/$name.completion.json"
  python3 - "$OUT/$name.completion.json" "$name" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as exc:
    print(f"  {sys.argv[2]}: FAILED to parse ({exc})"); raise SystemExit(0)
print(f"  {sys.argv[2]}: {json.dumps(d.get('content'))}")
print(f"  tokens: {d.get('tokens')}")
PY
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 2
}

run upstream          /home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-server
run upstream-mmqconfig /tmp/qwen38-ablation/build-hip/bin/llama-server
run halobox           /tmp/hipengine-halobox-head-20260913/build-hip/bin/llama-server
echo "=== done $(date -Is)"
