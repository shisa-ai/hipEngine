#!/usr/bin/env bash
# Same-session interleaved A/B for the router token-tiling change.
#
# The loop's stored baseline for this iteration is a SINGLE reading (0.6496 s)
# taken from the dense-route A/B's default arm, and it is being compared against
# a seven-run median. A single sample and a median are not comparable, so this
# settles it the way the repo does: restore the parent revisions of the two
# changed files for the baseline arm, keep the committed ones for the candidate,
# and interleave so drift hits both.
set -u
cd /home/lhl/hipEngine-gemma4
export HIPENGINE_HIP_ARCH=gfx1151 ROCR_VISIBLE_DEVICES=0 PYTHONPATH=.

FILES=(
  hipengine/kernels/hip_gfx1100/gemma4/gemma4_router.py
  hipengine/kernels/hip_gfx1100/gemma4/gemma4_layer.py
)
BENCH=".venv/bin/python scripts/gemma4_campaign_bench.py --expect-gpu 8060S --prompt 512 --output 128 --samples 3 --warmup 1"

arm() {  # arm <label> <rev>
  for f in "${FILES[@]}"; do git show "$2:$f" > "$f"; done
  $BENCH >/dev/null 2>&1
  .venv/bin/python -c "
import json
d = json.load(open('/tmp/gemma4_campaign_bench.json'))
s = d['stats']; p = d['public']
print(f\"$1  prefill_s={s['prefill_s']:.4f}  tps={s['prefill_tps']:.1f}  parity={p.get('public_generated_equals_expected')}\")
"
}

for round in 1 2; do
  echo "--- round $round ---"
  arm "baseline (untiled router) " HEAD~1
  arm "candidate (token_tile_8)  " HEAD
done

for f in "${FILES[@]}"; do git checkout -- "$f"; done
echo "--- restored committed revisions; diff should be empty ---"
git diff --stat -- "${FILES[@]}"
