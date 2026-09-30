#!/usr/bin/env bash
# DE-1 remaining action: localize the warp256 gate/up max-KL flip to a layer.
#
# The warp256 gate/up is only reachable where use_dp4a is true, so
# HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS is its effective layer scope. Sweeping that
# scope over the single case that carries the outlier (heldout_mixed_review, flip
# at step 63 of 83) localizes the flip without any kernel change.
#
# Production scope is PRODUCTION_Q4_DP4A_DECODE_LAYERS = (0,2,5,6,8,9,10,11) +
# range(13,48); the excluded layers are 1,3,4,7,12.
set -u
cd /home/lhl/hipEngine

# RETIRED 2026-09-30 - this harness has no live subject.
#
# It bisected the warp256 gate/up max-KL flip by sweeping
# HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS. The warp256 MoE decode route was removed
# from production (see REFACTOR.md "Unread warp-route env vars"), so the flip it
# localized no longer exists, and the probe it drove
# (scripts/qwen4exp_moe_decode_warp_logits_probe.py) was deleted with that route
# because its warp arms selected the route through env vars that are now unread
# - they would silently measure the incumbent and report a clean bisect.
#
# The measurement this harness produced is recorded and immutable: worklog
# 20260930T023729 (de1-layer-bisect). The kernels stay registered, so revisiting
# them needs a NEW harness that selects them directly rather than through the
# removed env vars - restoring this file unedited would restore the silent
# mismeasurement, which is why it fails here instead.
echo "qwen4exp_de1_layer_bisect.sh: RETIRED - its route is removed and its probe is deleted." >&2
echo "See the header of this file and docs/REFACTOR.md 'Unread warp-route env vars'." >&2
exit 2

# --- retained below as documentation of the retired experiment ---

# The ROCm SDK lives in the therock staging env; libamdhip64.so is not on the
# default loader path, so the probe exits "HIP runtime unavailable" without this.
ROCm_ENV=/home/lhl/miniforge3/envs/therock10-staging-20260828/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$ROCm_ENV/_rocm_sdk_core/lib:$ROCm_ENV/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

PROBE=scripts/qwen4exp_moe_decode_warp_logits_probe.py
COMMON=(
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
  --compiler-version-file /tmp/hipengine-hipcc-version.txt
  --fixture benchmarks/fixtures/qwen4exp_admission_suite_18prompt.json
  --case-id heldout_mixed_review
  --decode-steps 82
  --packet dual
  --teacher /tmp/r11-18case-strict-teacher.npz
)
OUT=/tmp/r11-layer-bisect
mkdir -p "$OUT"

run() {
  local tag="$1"; shift
  local layers="$1"; shift
  local out="$OUT/$tag.json"
  if [ -f "$out" ]; then echo "SKIP $tag (exists)"; return 0; fi
  echo "=== $tag : layers=$layers ==="
  timeout 1800 uv run --no-sync python "$PROBE" "${COMMON[@]}" \
      --dp4a-layers "$layers" --output "$out" > "$OUT/$tag.log" 2>&1
  echo "   exit=$? -> $out"
}

# Tier 0a: dp4a on ALL 48 layers (the 5 normally-excluded layers included).
# Already measured: incumbent max KL 0.19363 vs strict - 3.2x the 43-layer scope,
# which is direct evidence the exclusion set is load-bearing.
run t_all48          all
# Tier 0b: dp4a on NO layer, so the warp path is unreachable - the true control.
run t0_none          none
# Tier 1: halves of the 43-layer production scope.
run t1_lo            "$(python3 -c 'print(",".join(map(str,(0,2,5,6,8,9,10,11)+tuple(range(13,31)))))')"
run t1_hi            "$(python3 -c 'print(",".join(map(str,tuple(range(31,48)))))')"
# Tier 2: quarters of the half that carries the flip (filled in after tier 1).
run t2_a             "$(python3 -c 'print(",".join(map(str,(0,2,5,6,8,9,10,11)+tuple(range(13,22)))))')"
run t2_b             "$(python3 -c 'print(",".join(map(str,tuple(range(22,31)))))')"
# Tier 3: eighths of the quarter that carries the flip (filled in after tier 2).
run t3_a             "$(python3 -c 'print(",".join(map(str,(0,2,5,6,8,9,10,11)+tuple(range(13,18)))))')"
run t3_b             "$(python3 -c 'print(",".join(map(str,tuple(range(18,22)))))')"

echo
echo "================ SUMMARY ================"
python3 - <<'PY'
import json, pathlib, glob
FLIP_STEP = 63
print(f"{'tag':10} {'layers':>4} {'max_kl':>9} {'step63_top1':>11} {'flips':>5}  verdict")
for f in sorted(glob.glob('/tmp/r11-layer-bisect/*.json')):
    tag = pathlib.Path(f).stem
    try:
        d = json.loads(pathlib.Path(f).read_text())
    except Exception as e:
        print(f"{tag:10} unreadable: {e}"); continue
    for c in d.get('cases', []):
        st = c.get('step_top1') or []
        flips = [i for i, x in enumerate(st) if not x]
        s63 = st[FLIP_STEP] if len(st) > FLIP_STEP else None
        mx = c.get('kl_vs_incumbent_max')
        n = d.get('dp4a_layers_override')
        n = 'none' if n == 'none' else (len(str(n).split(',')) if n else 'default')
        v = 'FLIP PRESENT' if (s63 is False or (flips and FLIP_STEP in flips)) else 'clean'
        print(f"{tag:10} {str(n):>4} {mx:9.5f} {str(s63):>11} {len(flips):5}  {v}")
PY
