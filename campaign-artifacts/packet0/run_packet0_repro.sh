#!/bin/bash
# Packet 0 reproduction: reachable cells on current source (da407f89e).
# Each run pairs MTP against AR in-process with per-prompt arm alternation.
# Sequential on GPU0 (W7900). Model load ~1 min + ~3-5 min per run.
set -u
cd /home/lhl/hipEngine-gfx1100-concurrency2-better-mtp
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1
OUT=/tmp/he-bettermtp-raw/packet0
mkdir -p "$OUT"
MODEL=/models/gguf/Qwen3.8-27B-Q4_K_M.gguf
PROMPTS=benchmarks/prompts/mtpbench-code-general-ja.jsonl
PY=.venv/bin/python

run () {
  local name="$1"; shift
  if [ -s "/tmp/he-bettermtp-raw/packet0/$name.json" ]; then echo "=== skip $name (exists)"; return; fi
  echo "=== START $name $(date -u +%H:%M:%S)"
  "$PY" scripts/gguf_mtp_c1c8_server_bench.py \
    --model "$MODEL" \
    --backend hip_gfx1100 --quant gguf_q4_k_m --execution-profile production \
    --prompts "$PROMPTS" \
    --max-tokens 24 --batch-window-ms 20 --correctness-contract ar_exact \
    --output "/tmp/he-bettermtp-raw/packet0/$name.json" "$@" 2>&1
  local rc=$?
  echo "=== DONE $name rc=$rc $(date -u +%H:%M:%S)"
}

# 1. Explicit C8/K3 at its own capacity-8 evidence row (product route, no diagnostic plan)
run p0-c8k3-explicit-rep1 --mtp-request-mode explicit --widths 8 --resident-capacity 8 \
  --expected-mtp-widths 8 --candidate-budget 3
# 2. Explicit C2/K2 at its own capacity-2 evidence row
run p0-c2k2-explicit-rep1 --mtp-request-mode explicit --widths 2 --resident-capacity 2 \
  --expected-mtp-widths 2 --candidate-budget 2
# 3. Automatic K0 control at C8 (expected MTP widths: none)
run p0-c8-automatic-k0 --mtp-request-mode automatic --widths 8 --resident-capacity 8 \
  --expected-mtp-widths none --candidate-budget 3
# 4. Automatic K0 control at C2
run p0-c2-automatic-k0 --mtp-request-mode automatic --widths 2 --resident-capacity 2 \
  --expected-mtp-widths none --candidate-budget 2
echo "=== ALL RUNS COMPLETE $(date -u +%H:%M:%S)"
