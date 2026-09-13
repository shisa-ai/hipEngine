#!/bin/bash
# Packet 6 stage 2: retained economics reproduction at the grid-selected
# cells, mirroring the packet3 retained protocol.
#
# Product-route cells through REGISTERED evidence rows (no diagnostic
# resolver); the C2-K3 qualification cell through the bench's diagnostic
# resolver (no registered C2-K3 row yet — this run qualifies it); automatic
# K0 controls re-prove the unchanged AR baselines. Sequential on GPU0
# (W7900). ~1 min model load + ~3-5 min per run.
set -u
cd /home/lhl/hipEngine-gfx1100-concurrency2-better-mtp
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1 HIPENGINE_HIP_ARCH=gfx1100
# The C2-K3 cell has no listed policy cell; explicit opt-in per 96fa31006.
export HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS=1
OUT=/tmp/he-bettermtp-raw/packet6
mkdir -p "$OUT"
PY=.venv/bin/python
MODEL=/models/gguf/Qwen3.8-27B-Q4_K_M.gguf
PROMPTS=benchmarks/prompts/mtpbench-code-general-ja.jsonl

run () {
  local name="$1"; shift
  if [ -s "$OUT/$name.json" ]; then echo "=== skip $name"; return; fi
  echo "=== START $name $(date -u +%H:%M:%S)"
  "$PY" scripts/gguf_mtp_c1c8_server_bench.py \
    --model "$MODEL" \
    --backend hip_gfx1100 --quant gguf_q4_k_m --execution-profile production \
    --prompts "$PROMPTS" \
    --max-tokens 24 --batch-window-ms 20 --correctness-contract ar_exact \
    --output "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "=== DONE $name rc=$rc $(date -u +%H:%M:%S)"
  [ $rc -ne 0 ] && tail -4 "$OUT/$name.log"
}

# Product-route retained cells (registered evidence rows)
run p6-c1k3-retained --mtp-request-mode explicit --widths 1 --resident-capacity 8 \
  --expected-mtp-widths 1 --candidate-budget 3
run p6-c1k2-retained --mtp-request-mode explicit --widths 1 --resident-capacity 8 \
  --expected-mtp-widths 1 --candidate-budget 2
run p6-c2k2-retained --mtp-request-mode explicit --widths 2 --resident-capacity 2 \
  --expected-mtp-widths 2 --candidate-budget 2
run p6-c8k3-retained --mtp-request-mode explicit --widths 8 --resident-capacity 8 \
  --expected-mtp-widths 8 --candidate-budget 3
# C2-K3 qualification cell (diagnostic resolver)
run p6-c2k3-screen --generation2-diagnostic --mtp-request-mode explicit --widths 2 --resident-capacity 2 \
  --expected-mtp-widths 2 --candidate-budget 3
# Automatic K0 controls (unchanged AR baseline re-proof)
run p6-c1-automatic-k0 --mtp-request-mode automatic --widths 1 --resident-capacity 8 \
  --expected-mtp-widths none --candidate-budget 3
run p6-c2-automatic-k0 --mtp-request-mode automatic --widths 2 --resident-capacity 2 \
  --expected-mtp-widths none --candidate-budget 2
run p6-c8-automatic-k0 --mtp-request-mode automatic --widths 8 --resident-capacity 8 \
  --expected-mtp-widths none --candidate-budget 3
echo "=== PACKET6 RETAINED COMPLETE $(date -u +%H:%M:%S)"
