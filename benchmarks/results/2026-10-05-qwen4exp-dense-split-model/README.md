# Range-safe split-operand dense Q8 diagnostic

The compensated high/low split WMMA candidate measures **1.270–2.031x**
complete-projection speedup against the live pipelined F32 DPP coltile parent
across 15 real-weight cases. This is an operation-level diagnostic, not model
throughput or production qualification. Both the uncompensated and repaired
compositions fail the full production numerical envelope. The failed runtime
selector is removed; the diagnostic leaf remains available for direct operation
and private-buffer attribution tests. Historical uncompensated measurements
below describe the earlier source, not the repaired kernel.

## Historical uncompensated comparison basis

- Physical host `gfx1151`, machine-id `55ea6c509d0b49eea8de7094a1023668`,
  AMD Radeon 8060S / Strix Halo, HIP 7.15.26333 / clang 23.
- Qwen3.8-Flash-Next UD-Q4_K_XL, real Q8_0 banks from layers 0 and 3;
  seeded synthetic F32 activations. Weight hashes and exact shapes are in
  [the range-safe screen](screen-range.json).
- Parent `gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out`;
  candidate `gguf_q8_0_dense_split64x128_f32_f32_out`.
- Rows 512/1024/4096, K/N pairs 2560/10240, 2560/6144, 6144/2560,
  640/2560 and 2560/12288. Three warmups and twenty alternating-order pairs
  per case; prebound launch plus device synchronization. No cast/workspace pass.
- Source `e433a8472` plus the recorded candidate, `tracked_clean: false`.
  Both order strata improve in all 15 cases (minimum 1.651x).

| Rows | Sum of five parent operations (ms) | Sum of candidates (ms) | Ratio |
| ---: | ---: | ---: | ---: |
| 512 | 28.694 | 13.629 | 2.105x |
| 1024 | 57.418 | 27.829 | 2.063x |
| 4096 | 227.970 | 109.976 | 2.073x |

These sums describe independent projections, not one model request. The
[earlier unguarded leaf](../2026-10-05-qwen4exp-dense-split-wmma/README.md)
measured 2.047–3.691x; its faster timing does not include the range guard and
must not be used as the range-safe runtime candidate's cost.

## Historical uncompensated safety and integration

The kernel splits each F32 operand into F16 high plus a 2048-scaled F16
residual, computes high-high and both cross products, and omits low-low. Any
block with an operand outside the F16-high range recomputes through an
in-kernel strict F32 fallback. No host readback, shared external workspace or
additional kernel launch is required.

Eight GPU fixtures pass, including oversized activation and dequantized-weight
cases that match the strict wave-scale parent bitwise. Ordinary sampled CPU
F64 relative L2 error is 1.55e-7–3.87e-7, top-1 100%, max KL 6.58e-15 over
eight evenly spaced rows and every output column per case. Candidate repeats
are deterministic. These checks are not a model-level production certificate.

[Cached-only trace](trace-range.json) identifies the split kernel at full and
tail shapes: 27,852–95,218 ns for ordinary inputs, 1,073,932 ns for the
range-fallback fixture, allocated LDS 55,808 B, scratch 0, VGPR 248. gfx1100
execution is unverified.

`HIPENGINE_QWEN4_EXP_Q8_DENSE_SPLIT_LAYERS` is an explicit comma-separated
layer set. Empty means off; `0` selects layer 0. Named strict/production binders
clear it. Post-binder gate overrides replace only raw-F32 parents at rows>256
in layers 0–12, preserving late dense-wide and other quant/ABI owners. Layer
scope participates in the dispatch cache key. The variant cannot inherit the
raw-Q8 family's different shared library.

[One-case plumbing smoke](smoke.json) has only five compared logit rows and
four decode steps. It reports deterministic repeats but fails its steady-state
mean-KL scope and has incomplete/dirty provenance. It cannot qualify or reject
the route under the complete protocol. Full canonical, engaging heldout,
strict-teacher composition, task, control/lifecycle and engine-wall gates are
still required before a default change.

## Historical uncompensated reproduction

The following commands reproduce these earlier measurements only when run
from `ad9c2c5d3`; HEAD now runs the repaired split arithmetic.

```bash
ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH"
export HIPENGINE_HIP_ARCH=gfx1151
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-gr-dpp-20261005/hipcc-version.txt
# The version file is precomputed with hipcc --version; build outside rocprof.
uv run --no-sync pytest -q tests/test_gpu_qwen4exp_dense_split_wmma.py
export HIPENGINE_REQUIRE_CACHED_BUILD=1
uv run --no-sync python scripts/qwen4exp_dense_split_screen.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --rows 512 1024 4096 --pairs 20 --output /tmp/qwen4exp-dense-split-screen-range.json
rocprofv3 --kernel-trace --output-format csv \
  --output-directory /tmp/hipengine-dense-split-20261005/range-trace -- \
  uv run --no-sync pytest -q \
  'tests/test_gpu_qwen4exp_dense_split_wmma.py::test_split_wmma_recovers_f32_operand_precision[128-256-64-False]' \
  'tests/test_gpu_qwen4exp_dense_split_wmma.py::test_split_wmma_recovers_f32_operand_precision[257-1024-129-False]' \
  'tests/test_gpu_qwen4exp_dense_split_wmma.py::test_split_wmma_range_miss_runs_the_strict_f32_fallback[False]'
# Tracked-clean full canonical incremental numerical gate (not profile admission):
uv run --no-sync python scripts/execution_profile_q8_wmma_prefill_layers_gate.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --route dense_split --layers 0-12 --decode-steps 128 --repeat-runs 3 \
  --prefill-chunk-size 4096 --require-cached-build \
  --output /tmp/qwen4exp-dense-split-canonical-increment.json
```

The [complete canonical incremental run](canonical-increment.json) on clean
`ad9c2c5d3` compares early split layers 0–12 against current production at
chunk 4096, preserving late dense-wide layers 13–47 in both arms. All 12 cases
supply 128 decode transitions, 1,548 full-vocabulary rows and three candidate
repeats.

| Metric | Measured | Limit | Result |
| --- | ---: | ---: | --- |
| Mean KL | 1.3716e-3 | ≤1e-3 | Fail |
| p95 KL | 7.0943e-3 | ≤5e-3 | Fail |
| p99 KL | 1.7567e-2 | ≤2e-2 | Pass |
| Maximum KL | 0.051950 | ≤0.05 | Fail |
| Overall top-1 | 1538/1548 (99.354%) | ≥99% | Pass |

All four category scopes, the c1 shape and the steady transition fail one or
more continuous limits. Twelve rows exceed the 0.02 review boundary. Logits
are finite, repeats are bit-identical, and actual split dispatch counts are
**0 teacher / 3,924 candidate**. Provenance and complete coverage pass; the
collector exits 1 because quality fails, not because collection crashed.

## Repaired composition qualification

The [repaired candidate's named-strict production gate](canonical-strict-repaired.json)
compares
strict chunk1024 with production plus `dense_split_early` at chunk4096 over all
12 canonical cases, 1,548 full-vocabulary rows, 128 teacher-forced transitions
and three candidate repeats. Candidate split dispatch is engaged for 3,924
calls; the strict teacher has zero split calls. The candidate is finite,
repeat-deterministic, state/layout-valid and leak-free, but fails the binding
quality envelope:

| Metric | Measured | Limit | Result |
| --- | ---: | ---: | --- |
| Mean KL | 1.4386e-3 | ≤1e-3 | Fail |
| p95 KL | 7.5336e-3 | ≤5e-3 | Fail |
| p99 KL | 1.7430e-2 | ≤2e-2 | Pass |
| Maximum KL | 5.9744e-2 | ≤5e-2 | Fail |
| Overall top-1 | 1526/1548 (98.579%) | ≥99% | Fail |

All four categories and the steady transition fail at least one scope gate;
prefill-last and prefill-to-c1 remain near-zero. Since the binding numerical
gate fails, heldout, task, BF16-relative, isolation and engine-wall promotion
gates were not run. The strict fallback remains the admitted early-layer route.
The historical uncompensated strict result is preserved separately in
[canonical-strict.json](canonical-strict.json).

The [repaired natural-input attribution summary](natural-localize-repaired-summary.json)
covers 436 projections across four p512 categories, layers 0–12 and 14 role kinds.
Parent full logits remain bit-identical with observation enabled, all candidate
repeats are deterministic and finite, and teardown leaves zero allocations.
Sampled candidate-vs-F64 relative L2 is at most 2.13e-7; full-projection
candidate-vs-parent relative L2 is at most 3.03e-7. This collector evaluates
candidate outputs privately and does not establish model-level causation.

The [fresh allocation probe](allocation-repaired.json) prepares 89,322,462,984
tracked bytes with an 83,886,080-byte device-scratch margin. The repaired gate's
candidate peak is 89,404,925,864 bytes; both packets report zero allocations
after close. Allocation, projection error and repeat checks are independent of
the failed whole-model quality gate.

## Historical uncompensated scope screens

Three disjoint early-layer ranges were screened against current production at
chunk1024 using all 12 canonical cases, 24 transitions and three repeats. The
[scope-screen packet](scope-screen.json) is diagnostic only: it uses neither the
full depth nor the named-strict teacher.

| Layers | Mean KL | p95 KL | Max KL | Top-1 |
| --- | ---: | ---: | ---: | ---: |
| 0–3 | 3.6121e-3 | 1.7708e-2 | 5.9101e-2 | 289/300 (96.33%) |
| 4–7 | 2.1360e-3 | 8.1354e-3 | 2.5518e-2 | 293/300 (97.67%) |
| 8–12 | 1.8151e-3 | 9.2457e-3 | 1.9402e-2 | 292/300 (97.33%) |

All ranges fail the calibrated envelope. Outputs are finite and deterministic;
the teacher dispatches zero split calls while the candidates dispatch 2,520,
2,376 and 2,952 calls respectively. Clean source is `ad9c2c5d3`. These results
show that layer-range narrowing alone has not found a qualifying route; they
do not establish independent per-role arithmetic error or justify promotion.

## Natural-input attribution before repair

The [natural-input attribution packet](natural-localize.json) observes 436
projections across four canonical p512 categories and 14 early-layer role kinds.
It runs the parent normally, evaluates split outputs in private buffers, and
compares eight evenly spaced rows/all columns against F64 CPU matmul. All four
parent full-logit arrays are bit-identical with observation enabled. Candidate
repeats are deterministic and teardown leaves zero allocations.

Candidate sampled relative L2 is `1.225e-7–7.195e-7`, projection top-1 is 100%,
and maximum projection KL is `7.228e-10`. Long-K hyper-connection projections
have the largest errors; no large operand error appears. Full-projection
parent/candidate BF16 cast crossings total 118,679. Those counts are diagnostic,
not an admission gate or proof of model-level causation. The collector was
untracked during execution, so the packet explicitly disclaims complete
qualification provenance. Natural-input NPZ captures remain local and are not
part of the repository.

## Compensated candidate

The current candidate uses compensated F32 high-high summation,
positive/negative high-high WMMA averaging, and ordinary cross-plane
summation. TwoSum's high-high correction shares the 2048-scaled cross
accumulator; the low-low term is still omitted. All non-split variants and
the range-miss strict F32 fallback are unchanged. The complete projection
screen remains operation-level only. Across 15 real-weight cases, candidate
speedups are **1.270–2.031x** (all counterbalanced order strata improve); aggregate parent →
candidate walls are 28.777 → 18.828 ms at 512, 57.486 → 37.960 ms at 1024,
and 228.139 → 149.753 ms at 4096. The [compensated screen](compensated-screen.json)
records the exact cases and provenance. It is not a model-rate claim.

The final TwoSum kernel passes the `<1.2e-7` F64-relative long-K RED at
K2560 and K10240. The earlier `1.019e-7` values printed during exploration
belong to an intermediate signed-dot variant with a separate Kahan accumulator
plane, not the final TwoSum implementation. Final real-weight sampled error
is ≤`9.93e-8`, with top-1 100% and deterministic repeats across all 15 cases.
Five layer0 natural-capture diagnostics improve sampled relative L2 from
`1.53e-7–3.61e-7` to `4.93e-8–9.48e-8`; these are not model gates.

The [cached-only final trace](trace-compensated.json) identifies the split
kernel at rows129/K2560/N72 and rows129/K10240/N72. Durations are
311,900 ns and 1,233,229 ns, VGPR256, LDS55,808 B, scratch16 B. The larger
separate Kahan-plane exploration used scratch96 B and was not retained.
The final source is `fdb026c95`; source SHA256 and exact operation command,
build, environment, and raw trace hash are in the compact packets.

A [fresh chunk4096 allocation probe](allocation-compensated.json) at
`fdb026c95` passes at 89,322,462,984 tracked bytes, 83,886,080
device-scratch margin, and zero after-close
allocations. Its production manifest is still
`850829d509ebf741403e71e30450b8b7b681994f16e0f02038ba672c1f493d5b`:
the manifest hashes registry selections, not kernel bodies, and the opt-in
split candidate is not part of that named production plan. Source/build
identity must distinguish the repair; manifest identity alone cannot.

For the current operation screen, use the environment in
[compensated-screen.json](compensated-screen.json), prebuild outside profiling,
and run:

```bash
uv run --no-sync python scripts/qwen4exp_dense_split_screen.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --rows 512 1024 4096 --pairs 20 \
  --output /tmp/qwen4exp-dense-split-screen-compensated.json
```

The candidate's runtime selector and full-model gate modes are removed after
this failed qualification. Direct operation/F64 and private-buffer natural-input
diagnostics retain the low-level leaf and its numerical RED fixtures.
