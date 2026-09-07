#!/bin/bash
# Packet 0 stage 2: screening cells, repeat pairs, and the C2/K3 rejection proof.
# Runs after run_packet0_repro.sh frees GPU0.
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

# Repeat pairs for spread (no screening env; qualified cells).
run p0-c8k3-explicit-rep2 --mtp-request-mode explicit --widths 8 --resident-capacity 8 \
  --expected-mtp-widths 8 --candidate-budget 3
run p0-c2k2-explicit-rep2 --mtp-request-mode explicit --widths 2 --resident-capacity 2 \
  --expected-mtp-widths 2 --candidate-budget 2

# Screening cells (explicitly unqualified; env must be set). The diagnostic
# plan resolver is the fail-closed explicitly-unqualified candidate path the
# 2026-09-06 sweep used for sub-capacity widths: it admits only
# greedy_fast/D24/context<=95/horizon=24 groups at width <= 8 and records
# reason diagnostic_physical_gguf_mtp; the adapter's width/depth policy plus
# the screening env remain the operative gate.
export HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS=1
run p0-c5k3-screen-rep1 --generation2-diagnostic --mtp-request-mode explicit --widths 5 --resident-capacity 8 \
  --expected-mtp-widths 5 --candidate-budget 3
run p0-c5k3-screen-rep2 --generation2-diagnostic --mtp-request-mode explicit --widths 5 --resident-capacity 8 \
  --expected-mtp-widths 5 --candidate-budget 3
run p0-c2k1-screen-rep1 --generation2-diagnostic --mtp-request-mode explicit --widths 2 --resident-capacity 2 \
  --expected-mtp-widths 2 --candidate-budget 1
run p0-c7k3-screen-rep1 --generation2-diagnostic --mtp-request-mode explicit --widths 7 --resident-capacity 8 \
  --expected-mtp-widths 7 --candidate-budget 3
unset HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS

# C2/K3 rejection proof: no evidence row covers K3 at capacity 2 -> engaged 0/10.
run p0-c2k3-rejection-proof --mtp-request-mode explicit --widths 2 --resident-capacity 2 \
  --expected-mtp-widths none --candidate-budget 3

echo "=== ALL STAGE-2 RUNS COMPLETE $(date -u +%H:%M:%S)"
