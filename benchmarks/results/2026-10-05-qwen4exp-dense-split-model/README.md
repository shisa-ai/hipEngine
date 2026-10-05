# Range-safe split-operand dense Q8 candidate

The range-safe high/low split WMMA kernel reduces complete projection wall
39.7–64.1% (1.659–2.786x) against the live pipelined F32 DPP coltile parent
across 15 real-weight cases. This is an operation-level diagnostic, not model
throughput or production qualification. The default remains unchanged.

## Comparison basis

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

## Safety and integration

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

## Reproduce

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

The full canonical command is the next qualification step, not completed
evidence in this admission artifact.
