#!/bin/bash
# Packet 6 stage 1: the K1-K7 x C1-C8 diagnostic grid sweep.
#
# Runs the packet5 watchdog probe per cell (full 10-prompt canonical suite,
# AR + MTP arms, engaged/exact/budget-conformed per prompt) across all 56
# positive-depth cells. Every cell carries its own same-process AR baseline,
# so the K0 points ride along in the K1 rows. Diagnostic evidence only —
# the retained economics reproduction at the selected cell goes through the
# canonical retained harness afterwards.
#
# Resume support: a cell is skipped when its probe-summary.json already
# records status=complete.
set -u
cd /home/lhl/hipEngine-gfx1100-concurrency2-better-mtp
MODEL=/models/gguf/Qwen3.8-27B-Q4_K_M.gguf
PROMPTS=benchmarks/prompts/mtpbench-code-general-ja.jsonl
OUT=/tmp/he-bettermtp-raw/packet6
mkdir -p "$OUT"
PY=.venv/bin/python

for width in 1 2 3 4 5 6 7 8; do
  for budget in 1 2 3 4 5 6 7; do
    tag="k4-w${width}-b${budget}"
    summary="$OUT/${tag}-probe-summary.json"
    if [ -s "$summary" ] && grep -q '"status": "complete"' "$summary"; then
      echo "=== skip $tag (complete)"
      continue
    fi
    echo "=== START $tag $(date -u +%H:%M:%S)"
    "$PY" -u scripts/qwen38_packet5_k4_watchdog_probe.py \
      --model "$MODEL" \
      --prompts "$PROMPTS" \
      --output-dir "$OUT" \
      --width "$width" \
      --budget "$budget" \
      > "$OUT/${tag}.log" 2>&1
    rc=$?
    echo "=== DONE $tag rc=$rc $(date -u +%H:%M:%S)"
    if [ $rc -ne 0 ]; then
      tail -3 "$OUT/${tag}.log"
    fi
  done
done
echo "=== GRID SWEEP COMPLETE $(date -u +%H:%M:%S)"
