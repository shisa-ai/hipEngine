#!/bin/bash
# Packed-route C1 K0-K7 economics sweep on GPU0 (W7900).
# Every run goes through run_packed_bench.py: the fail-closed diagnostic
# harness injects the one-request packed_c1_target evidence row (explicit-only,
# unqualified test candidate path per Packet 0) and forbids the legacy
# singleton target verifier, so every engaged cycle is the repaired packed
# target by construction. Protocol otherwise mirrors the retained cells:
#   scripts/gguf_mtp_c1c8_server_bench.py, width 1, resident-capacity 8,
#   explicit mode, D24 greedy, 20 ms batch window, ar_exact contract,
#   full canonical 10-prompt suite, GPU_MAX_HW_QUEUES=1.
# One run = 10 balanced AR/MTP pairs (per-prompt arm order alternates);
# three independent runs per depth = the three-balanced-pairs protocol.
# K0 arms: --mtp-request-mode automatic (expected engaged 0/10 = automatic
# stays K0 control); its "ar" arm is the true no-MTP AR baseline.
# Resumable: existing complete outputs are skipped. Any run failure, watchdog
# timeout, or verdict-gate failure aborts the sweep for diagnosis.
set -u
cd /home/lhl/hipEngine-main
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1
export HIPENGINE_HIP_ARCH=gfx1100
OUT=/tmp/he-bettermtp-raw/packed-sweep
mkdir -p "$OUT"
PY=.venv/bin/python
MODEL=/models/gguf/Qwen3.8-27B-Q4_K_M.gguf
PROMPTS=benchmarks/prompts/mtpbench-code-general-ja.jsonl

echo "=== SWEEP START $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) head=$(git rev-parse HEAD)"

run () {
  local name="$1" mode="$2" expw="$3" budget="$4"
  if [ -s "$OUT/$name.json" ]; then echo "=== skip $name (exists)"; return 0; fi
  echo "=== START $name mode=$mode budget=$budget $(date -u +%H:%M:%S)"
  timeout 900 "$PY" campaign-artifacts/c1-k-sweep/run_packed_bench.py \
    --model "$MODEL" \
    --backend hip_gfx1100 --quant gguf_q4_k_m --execution-profile production \
    --prompts "$PROMPTS" \
    --mtp-request-mode "$mode" --widths 1 --resident-capacity 8 \
    --expected-mtp-widths "$expw" --candidate-budget "$budget" \
    --max-tokens 24 --batch-window-ms 20 --correctness-contract ar_exact \
    --output "$OUT/$name.json" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "=== DONE $name rc=$rc $(date -u +%H:%M:%S)"
  if [ $rc -ne 0 ]; then
    echo "=== FAIL $name rc=$rc — last log lines:"
    tail -6 "$OUT/$name.log"
    exit $rc
  fi
  if ! grep -q "packed_target_calls=" "$OUT/$name.log"; then
    echo "=== FAIL $name: no packed-call accounting"
    exit 4
  fi
  if ! "$PY" - "$OUT/$name.json" "$name" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
if not d.get("passed") or d.get("status") != "complete":
    print(f"VERDICT-FAIL {sys.argv[2]}: reasons={d.get('failure_reasons')}")
    sys.exit(1)
s = d["summary"]["1"]
print(f"OK ratio={s['mtp_vs_ar_ratio']:.4f} ar={s['ar']['tok_s']:.2f} "
      f"mtp={s['mtp']['tok_s']:.2f} exact={s['exact_cells']}/10 "
      f"engaged={s['engaged_cells']}/10 budget={s['budget_conformed_cells']}/10")
EOF
  then
    echo "=== FAIL $name verdict gate"
    exit 3
  fi
}

# Three balanced rounds; within a round each depth runs once so an
# interruption still leaves one complete pair per finished depth.
for r in 1 2 3; do
  run "r${r}-k0" automatic none 3
  for k in 1 2 3 4 5 6 7; do
    run "r${r}-k${k}" explicit 1 "$k"
  done
done
echo "=== SWEEP COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
