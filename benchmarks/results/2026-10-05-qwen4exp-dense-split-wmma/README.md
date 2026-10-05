# Q8 dense split-operand WMMA: leaf admission

The measurement-only high/low split kernel reduces complete projection wall
**51.2–72.9% (2.047–3.691x)** against the live pipelined F32 DPP coltile leaf
across 15 real-weight cases. Both order strata improve in every case. No
production binding changes and no full-engine speedup or numerical
qualification is claimed.

## Basis

- Physical host `gfx1151`, machine-id `55ea6c509d0b49eea8de7094a1023668`,
  AMD Radeon 8060S / Strix Halo; HIP 7.15.26333, clang 23.
- Qwen3.8-Flash-Next UD-Q4_K_XL, real Q8_0 banks from layers 0 and 3:
  attention QKV/gate/Q, SSM output, shared-expert down. Raw weight hashes are
  recorded for each case. Activations are seeded synthetic F32 inputs.
- Rows 512/1024/4096, K/N pairs 2560/10240, 2560/6144, 6144/2560,
  640/2560 and 2560/12288.
- Parent `gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out`;
  candidate `gguf_q8_0_dense_split64x128_f32_f32_out`.
- Three warmups, twenty alternating-order pairs, prebound library, host wall
  around launch plus device synchronization. No hidden cast/workspace pass.
- Source `31a757cc5` plus the recorded candidate; `tracked_clean: false`.
  These are diagnostic primitive timings, not clean publication/model rates.

| Rows | Sum of five parent operations (ms) | Sum of candidates (ms) | Ratio |
| ---: | ---: | ---: | ---: |
| 512 | 28.674 | 10.228 | 2.803x |
| 1024 | 57.392 | 21.214 | 2.705x |
| 4096 | 227.917 | 84.155 | 2.708x |

These sums describe five independent projection cases, not one model request.
[Scratch-free live-DPP screen](screen-noscratch.json). The initial
[pre-DPP-parent screen](screen.json) and [spilling full-tile screen](screen-dpp.json)
are kept separately, not quoted as the current candidate comparison.

## Arithmetic and correctness

Both F32 operands are represented as F16 high plus a 2048-scaled F16 residual.
The kernel computes high-high and two cross products, omitting only low-low.
High-high 16-term dots are independently rounded and added in ordinary F32;
feeding the growing sum back through WMMA failed the K1024 numerical fixture.
Cross products have a separate F32 accumulator and are unscaled once at output.
The 64-column x 128-row tile uses 55,296 bytes of LDS.

- RED: missing callable/key before implementation; F16-only substitution fails
  at relative L2 `2.87e-4` against the `1e-6` ordinary-input bound.
- GREEN: six independent GPU fixtures cover non-F16-exact weights, signed
  extrema, zero scales, true cancellation, tails, K up to 6144 and repeat bits.
  Ordinary-input error is about `1.2e-7–3.8e-7` versus independent F64 matmul.
  Cancellation uses an unsigned-dot forward-error bound for both CPU/GPU;
  relative recovery is `4.17e-4` against F16-only `1.171`, with 100% top-1.
- Eight unit surface/export/backend/fallback/shape tests and twenty existing
  parent GPU tests pass. The CPU-reference KL <= 0.05/top-1 >= 90% outer floor
  passes; it is not a production certificate.
- Each screen checks eight evenly spaced rows across every output column
  against CPU F64 dequant/matmul: top-1 100%, max KL `6.58e-15`, relative L2
  `1.55e-7–3.88e-7`. Full-output repeated candidate hashes match.
- Cached-only full/tail trace: expected split symbol, 19,517–75,422 ns,
  LDS 55,296 B, scratch 0, allocated VGPR 248. Static count is 241 VGPR and
  770 instructions. Register-only occupancy estimates exclude LDS.
  The unpredicated full-tile specialization had 256 VGPR and 204 bytes/lane
  scratch; it is no longer emitted. Both full and tail shapes use the
  predicated body, with all fifteen output hashes unchanged. Removing those
  spills increased the primitive win from 1.261–1.984x to 2.047–3.691x.
  [Trace](trace-summary.json), [static census](register-census-noscratch.json).

The exact raw-F32 coltile remains registered. gfx1100 execution is unverified.
The F16-high formulation requires representable operands; range-safe model
integration and the unchanged full-logit production/control/task/heldout gate
are still required before a public binding.

## Reproduce

```bash
ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH"
export HIPENGINE_HIP_ARCH=gfx1151
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-gr-dpp-20261005/hipcc-version.txt
# This precomputed file contains hipcc --version. Build outside the profiler.
uv run --no-sync pytest -q tests/test_gpu_qwen4exp_dense_split_wmma.py \
  tests/test_unit_qwen4exp_dense_wide_f16_activation.py \
  tests/test_gpu_gguf_q8_0_dense_wide.py
export HIPENGINE_REQUIRE_CACHED_BUILD=1
uv run --no-sync python scripts/qwen4exp_dense_split_screen.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --rows 512 1024 4096 --pairs 20 --output /tmp/dense-split-screen.json
rocprofv3 --kernel-trace --output-format csv \
  --output-directory /tmp/hipengine-dense-split-20261005/trace -- \
  uv run --no-sync pytest -q \
  'tests/test_gpu_qwen4exp_dense_split_wmma.py::test_split_wmma_recovers_f32_operand_precision[128-256-64-False]' \
  'tests/test_gpu_qwen4exp_dense_split_wmma.py::test_split_wmma_recovers_f32_operand_precision[257-1024-129-False]'
uv run --no-sync python scripts/qwen4exp_repair_register_census.py \
  --source hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_dense_wide.hip \
  --threads-per-block 256 --output /tmp/dense-split-census.json
```
