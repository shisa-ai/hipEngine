#!/bin/bash
# Packet 0 stage 4: adapter-level decline trace (explicit C5/K3 without the
# screening env but with the diagnostic plan, so the adapter capability is
# consulted and declines) + C2/K1 screening retry after the proposal-width fix.
set -u
cd /home/lhl/hipEngine-gfx1100-concurrency2-better-mtp
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1
OUT=/tmp/he-bettermtp-raw/packet0
MODEL=/models/gguf/Qwen3.8-27B-Q4_K_M.gguf
PROMPTS=benchmarks/prompts/mtpbench-code-general-ja.jsonl
PY=.venv/bin/python

if [ ! -s "/tmp/he-bettermtp-raw/packet0/p0-c5k3-decline-trace.json" ]; then
  echo "=== START decline-trace C5/K3 (diagnostic plan, no screening env) $(date -u +%H:%M:%S)"
  HIPENGINE_MTP2_TRACE_DECLINE=1 "$PY" scripts/gguf_mtp_c1c8_server_bench.py \
    --model "$MODEL" \
    --backend hip_gfx1100 --quant gguf_q4_k_m --execution-profile production \
    --prompts "$PROMPTS" \
    --generation2-diagnostic \
    --mtp-request-mode explicit --widths 5 --resident-capacity 8 \
    --expected-mtp-widths none --candidate-budget 3 \
    --max-tokens 24 --batch-window-ms 20 --correctness-contract ar_exact \
    --output "/tmp/he-bettermtp-raw/packet0/p0-c5k3-decline-trace.json" \
    > "/tmp/he-bettermtp-raw/packet0/p0-c5k3-decline-trace.log" 2>&1
  echo "rc=$? decline lines: $(grep -c 'mtp2-decline' "/tmp/he-bettermtp-raw/packet0/p0-c5k3-decline-trace.log")"
fi

if [ ! -s "/tmp/he-bettermtp-raw/packet0/p0-c2k1-screen-rep1.json" ]; then
  echo "=== START C2/K1 screening retry $(date -u +%H:%M:%S)"
  HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS=1 "$PY" scripts/gguf_mtp_c1c8_server_bench.py \
    --model "$MODEL" \
    --backend hip_gfx1100 --quant gguf_q4_k_m --execution-profile production \
    --prompts "$PROMPTS" \
    --generation2-diagnostic \
    --mtp-request-mode explicit --widths 2 --resident-capacity 2 \
    --expected-mtp-widths 2 --candidate-budget 1 \
    --max-tokens 24 --batch-window-ms 20 --correctness-contract ar_exact \
    --output "/tmp/he-bettermtp-raw/packet0/p0-c2k1-screen-rep1.json" \
    > "/tmp/he-bettermtp-raw/packet0/p0-c2k1-screen-rep1.log" 2>&1
  echo "rc=$?"
fi
echo "=== STAGE 4 COMPLETE $(date -u +%H:%M:%S)"
