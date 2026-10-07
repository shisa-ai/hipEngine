#!/usr/bin/env bash
# DE-11 paired decode wall + kernel sum, one interleaved session.
#
# The ledger's decode remainder subtracts a wall captured in one session from a
# kernel sum captured in another; a sequential code-p4096 comparison on this
# host drifts up to 8%. This driver runs both producers for both depths in one
# session, counterbalanced A/B/A/B -> B/A/B/A, with the thermal protocol's
# cooldown to <=50 C and one case per process.
#
# Wall   : scripts/qwen4exp_canonical_ar_bench.py hipengine, warmup 1 + 3 reps.
# Kernel : the same command under rocprofv3 --kernel-trace, warmup 1 + 1 rep.
#          The bench's phase_windows_ns["decode"] slices the trace.
set -uo pipefail

REPO=/home/lhl/hipEngine
MODEL=/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
FIXTURE=$REPO/benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json
OUT=${DE11_OUT:-$REPO/benchmarks/results/2026-10-07-qwen4exp-de11-paired-decode-wall-kernel}
COMPILER_VERSION_FILE=${DE11_COMPILER_VERSION_FILE:-$OUT/compiler-version.txt}
TEMP=${DE11_TEMP:-/sys/class/hwmon/hwmon6/temp1_input}
COOL_TARGET=${DE11_COOL_TARGET:-50000}
SCRATCH=${DE11_SCRATCH:-/tmp/de11/traces}

export PYTHONPATH=$REPO
export HIPENGINE_HIP_ARCH=gfx1151

mkdir -p "$OUT/wall" "$OUT/kernel" "$OUT/wall-raw" "$OUT/kernel-raw" "$SCRATCH"

cooldown() {
  local t
  while :; do
    t=$(cat "$TEMP")
    if [ "$t" -le "$COOL_TARGET" ]; then break; fi
    sleep 10
  done
}

bench() {  # case chunk reps output
  local case_id=$1 chunk=$2 reps=$3 output=$4
  python "$REPO/scripts/qwen4exp_canonical_ar_bench.py" hipengine \
    --model-root "$MODEL" --fixture "$FIXTURE" --case-id "$case_id" \
    --prefill-chunk-size "$chunk" --warmups 1 --repetitions "$reps" \
    --compiler-version-file "$COMPILER_VERSION_FILE" --require-cached-build \
    --output "$output"
}

run_wall() {  # case tag
  local case_id=$1 tag=$2
  cooldown
  local t0; t0=$(cat "$TEMP")
  echo "WALL $case_id $tag temp_start=$t0 $(date -Is)"
  bench "$case_id" 1024 3 "$SCRATCH/wall-${case_id}-${tag}.json"
  local rc=$?
  if [ $rc -eq 0 ]; then
    cp "$SCRATCH/wall-${case_id}-${tag}.json" "$OUT/wall-raw/${case_id}-${tag}.json"
    python "$REPO/scripts/qwen4exp_de11_paired_decode_wall_kernel.py" wall \
      "$SCRATCH/wall-${case_id}-${tag}.json" \
      --output "$OUT/wall/${case_id}-${tag}.json" >/dev/null
  fi
  echo "WALL $case_id $tag rc=$rc temp_end=$(cat "$TEMP") $(date -Is)"
}

run_kernel() {  # case tag
  local case_id=$1 tag=$2
  local dir="$SCRATCH/${case_id}-${tag}"
  mkdir -p "$dir"
  cooldown
  local t0; t0=$(cat "$TEMP")
  echo "KERNEL $case_id $tag temp_start=$t0 $(date -Is)"
  ( cd "$dir" && rocprofv3 --kernel-trace -f csv -o trace -- \
      python "$REPO/scripts/qwen4exp_canonical_ar_bench.py" hipengine \
        --model-root "$MODEL" --fixture "$FIXTURE" --case-id "$case_id" \
        --prefill-chunk-size 1024 --warmups 1 --repetitions 1 \
        --compiler-version-file "$COMPILER_VERSION_FILE" --require-cached-build \
        --output "$dir/bench.json" )
  local rc=$?
  if [ $rc -eq 0 ]; then
    cp "$dir/bench.json" "$OUT/kernel-raw/${case_id}-${tag}.json"
    python "$REPO/scripts/qwen4exp_de11_paired_decode_wall_kernel.py" kernel \
      "$dir/bench.json" --trace "$dir/trace_kernel_trace.csv" \
      --output "$OUT/kernel/${case_id}-${tag}.json" >/dev/null
  fi
  echo "KERNEL $case_id $tag rc=$rc temp_end=$(cat "$TEMP") $(date -Is)"
}

# Counterbalanced interleaved order: A/B/A/B for depth, then reversed.
run_wall   code-p512  A
run_kernel code-p512  A
run_wall   code-p4096 A
run_kernel code-p4096 A
run_kernel code-p4096 B
run_wall   code-p4096 B
run_kernel code-p512  B
run_wall   code-p512  B

python "$REPO/scripts/qwen4exp_de11_paired_decode_wall_kernel.py" session "$OUT" \
  --output "$OUT/paired-summary.json" >/dev/null

echo "DE11_DONE"

