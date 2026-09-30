#!/bin/bash
# P8 sgemm tier: same-binary route A/B bench, then the two teacher-forced
# gate pairs (capture with the tile tier pinned = baseline; gate with the
# tree as-is = candidate).  Sequential on one GPU; every stage tees a log.
set -o pipefail
cd /home/lhl/hipEngine-gemma4 || exit 1
unset HIP_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=0 PYTHONPATH=.
PY=.venv/bin/python
BENCH="--prompt 1024 --output 128 --samples 3 --warmup 1 --context 8192"
BASE="$HOME/.cache/hipengine/tmp"
RES=benchmarks/results

echo "=== stage 0: router screen on this GPU (row evidence, dynamic gpu label)"
$PY scratch/p8_router_sgemm_screen.py 2>&1 | tee /tmp/p8_screen.log
s0=${PIPESTATUS[0]}; echo "stage0 rc=$s0"; [ $s0 -ne 0 ] && exit $s0

echo "=== stage 1: head bench, sgemm tier, route counter"
$PY scratch/p8_route_counter_wrapper.py $BENCH --out $RES/2026-09-30-gemma4-p8-router-sgemm-head.json --label p8-router-sgemm-head --expect-gpu "W7900" 2>&1 | tee /tmp/p8_bench_new.log
s1=${PIPESTATUS[0]}; echo "stage1 rc=$s1"; [ $s1 -ne 0 ] && exit $s1

echo "=== stage 2: baseline bench, tile tier (pre-change route)"
$PY scratch/p8_baseline_route_wrapper.py $BENCH --out $RES/2026-09-30-gemma4-p8-router-tile-base.json --label p8-router-tile-base --expect-gpu "W7900" 2>&1 | tee /tmp/p8_bench_base.log
s2=${PIPESTATUS[0]}; echo "stage2 rc=$s2"; [ $s2 -ne 0 ] && exit $s2

echo "=== stage 3a: capture baseline arm at prompt 2048 / prefill 1024"
$PY scratch/p8_gate_baseline_wrapper.py capture --prompt 2048 --prefill 1024 --out $BASE/p8_router_base_1024.npz --manifest $BASE/p8_router_base_1024.json 2>&1 | tee /tmp/p8_cap1024.log
s3=${PIPESTATUS[0]}; echo "stage3a rc=$s3"; [ $s3 -ne 0 ] && exit $s3

echo "=== stage 3b: gate sgemm tier at prompt 2048 / prefill 1024"
$PY -m scripts.gemma4_teacher_forced_gate gate --prompt 2048 --prefill 1024 --baseline $BASE/p8_router_base_1024.npz --out $RES/2026-09-30-gemma4-26b-a4b-p8-router-sgemm-gate.json 2>&1 | tee /tmp/p8_gate1024.log
s4=${PIPESTATUS[0]}; echo "stage3b rc=$s4"; [ $s4 -ne 0 ] && exit $s4

echo "=== stage 4a: capture baseline arm at prompt 5120 / prefill 4096"
$PY scratch/p8_gate_baseline_wrapper.py capture --prompt 5120 --prefill 4096 --context 8192 --out $BASE/p8_router_base_4096.npz --manifest $BASE/p8_router_base_4096.json 2>&1 | tee /tmp/p8_cap4096.log
s5=${PIPESTATUS[0]}; echo "stage4a rc=$s5"; [ $s5 -ne 0 ] && exit $s5

echo "=== stage 4b: gate sgemm tier at prompt 5120 / prefill 4096"
$PY -m scripts.gemma4_teacher_forced_gate gate --prompt 5120 --prefill 4096 --context 8192 --baseline $BASE/p8_router_base_4096.npz --out $RES/2026-09-30-gemma4-26b-a4b-p8-router-sgemm-gate-4096.json 2>&1 | tee /tmp/p8_gate4096.log
s6=${PIPESTATUS[0]}; echo "stage4b rc=$s6"; [ $s6 -ne 0 ] && exit $s6

echo "ALL_DONE"
