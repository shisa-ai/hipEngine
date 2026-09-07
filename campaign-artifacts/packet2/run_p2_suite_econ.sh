#!/bin/bash
# Packet 2: full-suite C1 economics through the server bench (diagnostic).
# Width-1 explicit requests at resident capacity 8 with AR control in-run.
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

run c1k3-suite-econ --mtp-request-mode explicit --widths 1 --resident-capacity 8 \
  --expected-mtp-widths 1 --candidate-budget 3 --generation2-diagnostic
run c1k2-suite-econ --mtp-request-mode explicit --widths 1 --resident-capacity 8 \
  --expected-mtp-widths 1 --candidate-budget 2 --generation2-diagnostic
echo "=== PACKET2 SUITE ECON COMPLETE $(date -u +%H:%M:%S)"
