#!/bin/bash
# Packet 2: retained explicit C1 measurement through the REGISTERED evidence
# rows (no diagnostic resolver).
set -u
cd /home/lhl/hipEngine-gfx1100-concurrency2-better-mtp
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100
OUT=/tmp/he-bettermtp-raw/packet2
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

run c1k3-suite-retained --mtp-request-mode explicit --widths 1 --resident-capacity 8 \
  --expected-mtp-widths 1 --candidate-budget 3
run c1k2-suite-retained --mtp-request-mode explicit --widths 1 --resident-capacity 8 \
  --expected-mtp-widths 1 --candidate-budget 2
echo "=== PACKET2 RETAINED COMPLETE $(date -u +%H:%M:%S)"
