#!/bin/bash
# Packet 3: retained-cell re-measurement after extending the planar-Q6
# FFN-down prefill sibling band to rows 4-128 (bit-exact row64 owner).
# The down projection runs in every staged cycle, so every explicit MTP cell
# is affected; automatic K0 arms re-prove the unchanged AR baseline.
# Sequential on GPU0 (W7900). ~1 min model load + ~3-5 min per run.
set -u
cd /home/lhl/hipEngine-gfx1100-concurrency2-better-mtp
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1 HIPENGINE_HIP_ARCH=gfx1100
OUT=/tmp/he-bettermtp-raw/packet3
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
run p3-c1k3-retained --mtp-request-mode explicit --widths 1 --resident-capacity 8 \
  --expected-mtp-widths 1 --candidate-budget 3
run p3-c1k2-retained --mtp-request-mode explicit --widths 1 --resident-capacity 8 \
  --expected-mtp-widths 1 --candidate-budget 2
run p3-c2k2-retained --mtp-request-mode explicit --widths 2 --resident-capacity 2 \
  --expected-mtp-widths 2 --candidate-budget 2
run p3-c8k3-retained --mtp-request-mode explicit --widths 8 --resident-capacity 8 \
  --expected-mtp-widths 8 --candidate-budget 3
# Sweep-era cells (diagnostic resolver, sub-capacity widths)
run p3-c5k3-screen --generation2-diagnostic --mtp-request-mode explicit --widths 5 --resident-capacity 8 \
  --expected-mtp-widths 5 --candidate-budget 3
run p3-c2k1-screen --generation2-diagnostic --mtp-request-mode explicit --widths 2 --resident-capacity 2 \
  --expected-mtp-widths 2 --candidate-budget 1
run p3-c7k3-screen --generation2-diagnostic --mtp-request-mode explicit --widths 7 --resident-capacity 8 \
  --expected-mtp-widths 7 --candidate-budget 3
# Automatic K0 controls (unchanged AR baseline re-proof)
run p3-c2-automatic-k0 --mtp-request-mode automatic --widths 2 --resident-capacity 2 \
  --expected-mtp-widths none --candidate-budget 2
run p3-c8-automatic-k0 --mtp-request-mode automatic --widths 8 --resident-capacity 8 \
  --expected-mtp-widths none --candidate-budget 3
echo "=== PACKET3 RETAINED COMPLETE $(date -u +%H:%M:%S)"
