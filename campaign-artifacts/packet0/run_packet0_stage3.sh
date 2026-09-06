#!/bin/bash
# Packet 0 stage 3: decline-reason diagnostics (outside timed measurements).
# - explicit C5/K3 WITHOUT the screening env: adapter capability declines the
#   unlisted cell; HIPENGINE_MTP2_TRACE_DECLINE=1 captures the reason.
# - automatic C5: runtime K0 point at a middle width.
set -u
cd /home/lhl/hipEngine-gfx1100-concurrency2-better-mtp
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1
OUT=/tmp/he-bettermtp-raw/packet0
MODEL=/models/gguf/Qwen3.8-27B-Q4_K_M.gguf
PROMPTS=benchmarks/prompts/mtpbench-code-general-ja.jsonl
PY=.venv/bin/python

if [ ! -s "/tmp/he-bettermtp-raw/packet0/p0-c5k3-decline-trace.log" ]; then
  echo "=== START decline-trace C5/K3 $(date -u +%H:%M:%S)"
  HIPENGINE_MTP2_TRACE_DECLINE=1 "$PY" scripts/gguf_mtp_c1c8_server_bench.py \
    --model "$MODEL" \
    --backend hip_gfx1100 --quant gguf_q4_k_m --execution-profile production \
    --prompts "$PROMPTS" \
    --mtp-request-mode explicit --widths 5 --resident-capacity 8 \
    --expected-mtp-widths none --candidate-budget 3 \
    --max-tokens 24 --batch-window-ms 20 --correctness-contract ar_exact \
    --output "/tmp/he-bettermtp-raw/packet0/p0-c5k3-decline-trace.json" \
    > "/tmp/he-bettermtp-raw/packet0/p0-c5k3-decline-trace.log" 2>&1
  echo "rc=$? decline lines: $(grep -c 'mtp2-decline' "/tmp/he-bettermtp-raw/packet0/p0-c5k3-decline-trace.log")"
fi

if [ ! -s "/tmp/he-bettermtp-raw/packet0/p0-c5-automatic-k0.json" ]; then
  echo "=== START automatic C5 K0 $(date -u +%H:%M:%S)"
  HIPENGINE_MTP2_TRACE_DECLINE=1 "$PY" scripts/gguf_mtp_c1c8_server_bench.py \
    --model "$MODEL" \
    --backend hip_gfx1100 --quant gguf_q4_k_m --execution-profile production \
    --prompts "$PROMPTS" \
    --mtp-request-mode automatic --widths 5 --resident-capacity 8 \
    --expected-mtp-widths none --candidate-budget 3 \
    --max-tokens 24 --batch-window-ms 20 --correctness-contract ar_exact \
    --output "/tmp/he-bettermtp-raw/packet0/p0-c5-automatic-k0.json" \
    > "/tmp/he-bettermtp-raw/packet0/p0-c5-automatic-k0.log" 2>&1
  echo "rc=$?"
fi
echo "=== STAGE 3 COMPLETE $(date -u +%H:%M:%S)"
