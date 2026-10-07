# MX-1 F16 routed-expert owner: recheck of the gate/up and down main kernels

This artifact records a fresh, independent run of the routed-expert owner
comparison for `MX-1` (the "F16 routed experts with no repair" route) on the
current tree. It is a main-kernel timing screen, not an end-to-end rate, and it
makes no correctness claim.

## What was already known, and what is new

`MX-1` was already refuted on both halves of its own acceptance target before
this run:

- **Numerics.** The full route fails the production envelope *worse* than its
  own gate/up half (`benchmarks/results/2026-09-30-qwen4exp-mx1-full-route/gate.json`),
  and the gate/up-only variant fails the 18-row `prefill_last` scope by a
  whisker (`benchmarks/results/2026-09-30-qwen4exp-f16-compact-gate-up/gate.json`).
- **Performance (gate/up half).** The F16 gate/up owner is slower than the
  production iu8-risk owner in 8 of 8 cases, median +138.5%
  (`benchmarks/results/2026-10-04-qwen4exp-moe-f16-compact-owner-ab/gate-up-3arm.json`).

The down half's main-kernel cost had never been measured against the
production down owner. This run measures it and finds the opposite sign: the
F16 down owner is the **fastest** of the registered down owners.

One naming caution: the owner A/B's `down` primary is
`selected_grouped_prefill_pair2_row_publish_bf16_bf16_out`, but the **actual**
production down owner is the iu8-risk kernel selected via
`HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT=1` (see the 2026-09-29 CHANGELOG correction),
so the pair2 primary is not the production comparator. Against the production
iu8-risk owner the F16 down is **-42.29% median** (range -49.54% to -32.71%).

## Candidate paths and fallback

| Half | Candidate (F16, no repair) | Production owner it would replace |
| --- | --- | --- |
| gate/up | `selected_dual_wmma_prefill_compact_bf16_bf16_out` | `selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out` (+ sparse repair) |
| down | `selected_grouped_wmma_prefill_compact_bf16_bf16_out` | `selected_wmma_iu8_risk_prefill_bf16_bf16_out` (+ sparse repair) |

Both F16 owners are registered against `(hip_gfx1151, moe_linear, gguf_q4_k)`
and `(hip_gfx1151, moe_linear, gguf_q5_1)`. Their strict/registered fallbacks
are the production owners in the right-hand column, so the fallback registration
is preserved. The gate/up F16 owner is reached by clearing
`HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4`; the down F16 owner by clearing
`HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN` and setting
`HIPENGINE_QWEN4_EXP_Q5_1_WMMA=1`.

## Command and host

```
PY=/home/lhl/miniforge3/envs/therock/bin/python
ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib" \
PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH" \
HIPENGINE_HIP_ARCH=gfx1151 \
$PY scripts/qwen4exp_moe_owner_ab.py --route gate-up --output gate-up.json
$PY scripts/qwen4exp_moe_owner_ab.py --route down    --output down.json
```

- Tree: `eacf833d0` plus the harness fix described below.
- Host: `gfx1151`, machine id `55ea6c509d0b49eea8de7094a1023668`.
- Hardware: AMD RYZEN AI MAX+ 395 / Radeon 8060S (Strix Halo).
- ROCm: TheRock `10.0.0`, Python 3.12.13.
- Geometry: 512 experts; gate/up `in=2560, out=640, dual`; down
  `in=640, out=2560`. Weights are synthesized at the real shapes and byte
  layout; the access pattern and byte counts are representative, the values are
  not. 15 HIP-event repetitions per case, 2 primed launches, medians.

## Result

`vs_primary_pct` is positive when that owner is slower than the primary.

**gate/up** (primary `selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out`):

| owner | slower cases | median | min | max |
| --- | ---: | ---: | ---: | ---: |
| `selected_dual_wmma_prefill_compact_bf16_bf16_out` (F16) | 8/8 | **+138.35%** | +53.16% | +141.48% |
| `selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out` (fallback) | 8/8 | +151.49% | +20.28% | +392.71% |
| `selected_dual_wmma_iu8_risk_j32_prefill_bf16_bf16_out` | 8/8 | +115.17% | +14.45% | +124.28% |

This reproduces the committed gate/up result (F16 median +138.5%, range
+53.9% to +141.6%) within the run-to-run spread. The primary arm's own repeat
drift is 0.293%.

**down** (primary `selected_grouped_prefill_pair2_row_publish_bf16_bf16_out`,
which is the A/B primary but not the production owner):

| owner | slower cases | median | min | max |
| --- | ---: | ---: | ---: | ---: |
| `selected_grouped_wmma_prefill_compact_bf16_bf16_out` (F16) | **0/8** | **-59.45%** | -75.51% | -49.15% |
| `selected_wmma_iu8_risk_prefill_bf16_bf16_out` (production) | 0/8 | -36.22% | -52.92% | -21.74% |
| `selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out` (fallback) | 8/8 | +156.54% | +82.52% | +346.04% |

The F16 down owner is the fastest registered down owner in all 8 cases. The
primary arm's repeat drift is 0.310%. Absolute medians (ms), balanced rows
512/1024/4096/10240: F16 down 3.07 / 3.21 / 3.59 / 4.46 against pair2
6.90 / 7.22 / 9.80 / 17.12. Recomputed against the production iu8-risk owner,
the F16 down is **-42.29% median**, range -49.54% to -32.71%.

## What this means for MX-1

- The full F16 route stays refuted: the gate/up half alone is +138% slower,
  which already exceeds the 1,219.978 ms repair the route exists to remove, and
  the full route fails the numerical envelope worse than the gate/up half.
- The down half is **not** the reason the full route loses. It is ~1.5-2.0x
  faster than the production iu8-risk down main kernel (-42.29% median), so a
  **down-only** F16 variant (F16 down, gate/up left exact) is the only MX-1
  configuration with a favorable main-kernel measurement whose numerical status
  is unmeasured.
- That variant has not been gated. This artifact does not claim it is
  numerically admissible: the full-route gate showed the F16 down roughly
  doubles the accumulated drift when added to an already-drifted gate/up, and
  the down-only variant has never been measured against the strict teacher.

## Remaining gate (not run here)

The decisive next experiment is the canonical layer2 numerical/task gate for a
down-only candidate:

```
HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN=0 HIPENGINE_QWEN4_EXP_Q5_1_WMMA=1
```

with `HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4` left at production, run through
`scripts/qwen4exp_layer2_profile_gate.py`. Until that gate passes, the down F16
owner is diagnostic only and no default changes.

## Harness fix required to run this

`scripts/qwen4exp_moe_owner_ab.py` could not run at `eacf833d0`:
`measure()` referenced `padded_rows`, which a 2026-10-04 refactor
(`8544227b7`) moved into a nested `build()` and left unbound, raising
`NameError` on every case. The down route also lacked an arm for the F16 down
owner. Both are fixed in this unit; the gate/up arm reproduces its committed
values, which is the control that the fix did not perturb the comparison.

## Limits

- Main-kernel only; no repair pass, no risk-queue writes, no packing, no engine
  work, no end-to-end rate, no correctness claim.
- Synthesized weights at real shapes and byte layout; the values are not real.
- One host, one quant family, one tile configuration. The `down` F16 arm uses
  the harness's default 16-row map; the kernel's internal output tile is 32.
- `performance_claim: false` in both JSON files.
