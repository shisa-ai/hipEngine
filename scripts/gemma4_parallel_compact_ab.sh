#!/usr/bin/env bash
# Same-session A/B for the parallel MoE compactor.
#
# The end-to-end effect is ~11 ms of a ~656 ms prefill, about 1.7 percent, which
# is inside this host's session noise. A single before/after pair therefore
# cannot resolve it, so this follows the protocol the accepted
# 2026-09-28-gemma4-q5-1-down-mmq-load-width candidate used: restore the parent
# revision of the one changed file for the baseline arm and the committed
# revision for the candidate arm, nothing else differing, and interleave the
# arms so drift hits both.
set -u
cd /home/lhl/hipEngine-gemma4
export HIPENGINE_HIP_ARCH=gfx1151 ROCR_VISIBLE_DEVICES=0 PYTHONPATH=.

FILE=hipengine/kernels/hip_gfx1100/gemma4/gemma4_experts.py
BENCH=".venv/bin/python scripts/gemma4_campaign_bench.py --expect-gpu 8060S --prompt 512 --output 128 --samples 3 --warmup 1"

arm() {  # arm <label> <rev>
  git show "$2:$FILE" > "$FILE"
  $BENCH >/dev/null 2>&1
  .venv/bin/python -c "
import json
d = json.load(open('/tmp/gemma4_campaign_bench.json'))
s = d['stats']; p = d['public']
print(f\"$1  prefill_s={s['prefill_s']:.4f}  tps={s['prefill_tps']:.1f}  decode={s['decode_tps']:.2f}  parity={p.get('public_generated_equals_expected')}\")
"
}

for round in 1 2; do
  echo "--- round $round ---"
  arm "baseline(serial)  " HEAD~1
  arm "candidate(parallel)" HEAD
done

# Leave the worktree on the committed revision whatever happens.
git checkout -- "$FILE"
echo "--- restored committed revision; diff should be empty ---"
git diff --stat -- "$FILE"
