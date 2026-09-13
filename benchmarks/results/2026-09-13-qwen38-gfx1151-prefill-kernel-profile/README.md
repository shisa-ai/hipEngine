# gfx1151 Qwen3.8-27B Q4_K_M prefill kernel profile

Which kernels own prompt processing on the Radeon 8060S, how much of the
prompt they cover, and what limits them.

## Result

Dense K-quant WMMA prefill owns **55.0-57.8%** of prefill kernel time at 512,
1024 and 4096 prompt tokens. Two owners dominate:

| Owner | 512 tok | 1024 tok | 4096 tok | Tile | Threads | VGPR |
| --- | --- | --- | --- | --- | --- | --- |
| `dense_dual_wmma_prefill_bf16_bf16_out` (fused gate+up) | 36.46% | 34.22% | 33.01% | 32 cols x 256 rows | 128 | 248 |
| `t16_wmma_prefill_shared_b_bf16_bf16_out` (48 columns) | 21.35% | 21.08% | 22.02% | 48 cols x 256 rows | 128 | 256 |
| `qmicro_planar_wmma_prefill_shared4r4_bf16_bf16_out` | 10.86% | 11.17% | 8.09% | 32 cols x 256 rows | 128 | 176 |
| `t16_wmma_prefill_shared8r3_bf16_bf16_out` | 7.08% | 6.33% | 6.11% | 32 cols x 384 rows | 256 | 96 |

Q6_K is the next family at 17.5-18.7%. The planar `shared4r4` owner covers
8.1-11.2% and the standard `shared8r3` owner 6.1-7.1%; at 4096 tokens the
48-column planar band takes over 2.8% of the wide down shape from `shared4r4`,
which is why that owner drops from 11.17% to 8.09%. Gated DeltaNet and its
preparation kernels take 12.8-14.5%, the rocBLAS F16 route 3.1-4.1%, and flash
attention 1.0-3.0%.

Traced device time is 98.1-99.4% of the measured prefill wall time, so prefill
cost is kernel execution rather than launch gaps.

## Structure

The model has 64 transformer layers: 48 Gated DeltaNet layers and 16
full-attention layers. Prefill processes the 48 linear-attention layers in
1024-row chunks and the 16 full-attention layers in a single pass over the
prompt, so at 4096 tokens the same owner appears twice with different grid
heights, for example 192 launches at 1024 rows plus 16 launches at 4096 rows
for the fused gate+up kernel. Both pass modes reach the same throughput
(27.74 and 27.97 TFLOP/s), so chunking costs nothing by itself.

Every owner row is explained by the GGUF tensor table in `artifact.json`: the
launch count divided by the chunk count gives the number of tensor slots a row
covers, and that matches the 12 tensor groups exactly at all three prompt
lengths. The only unattributed tensors are the MTP `nextn` layer's copies,
which prefill does not run.

## What limits prefill

- **Not memory.** The dense prefill re-reads 12.4 GB of weights at 512 tokens
  and 47.8 GB at 4096 tokens (the 48 linear-attention layers are re-read once
  per chunk) at an aggregate 4.7-10.4 GB/s, with the busiest single owner at
  25 GB/s.
- **Occupancy.** Every dense owner runs at 2 waves per SIMD because its
  248-256 VGPR footprint fits only twice into the 512-VGPR per-thread budget.
  Dropping below 171 VGPR would admit a third wave. The 96-VGPR `shared8r3`
  owner is the only dense owner above 2 waves.
- **Owner choice on the wide down shape.** On the identical `(17408, 5120)`
  shape the Q4_K `shared_b` owner reaches 27.50 TFLOP/s while the Q6_K planar
  owners reach 21.28 and 20.34 TFLOP/s, a 25-29% gap that quant size does not
  explain. That shape is 10.9% of 4096-token prefill kernel time.
- **Narrow outputs.** Owners with 1024 output columns (attention K and V) run
  at 8.75-19.12 TFLOP/s against 30-33 TFLOP/s for the same kernel families on
  6144- to 12288-wide outputs. This profile does not establish the mechanism.

## Reproduce

Set up the ROCm environment, then run the driver. The driver itself opens the
three process boundaries described below, so one invocation covers the whole
protocol:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate therock
ROCM_ROOT=$(python -m rocm_sdk path --root)
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib64:$ROCM_ROOT/lib/llvm/lib:${LD_LIBRARY_PATH:-}"

PYTHONPATH=. .venv/bin/python scripts/qwen38_gfx1151_prefill_kernel_profile.py \
    --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
    --backend hip_gfx1151 --quant gguf_q4_k_m \
    --prompt-lengths 512 1024 4096 \
    --compiler-version-file /tmp/hip1151-t6/hipcc-version-gfx1151.txt \
    --cache-root /tmp/qwen38-prefill-profile/cache \
    --run-root /tmp/qwen38-prefill-profile/run \
    --run-tag qwen38-gfx1151-prefill-kernel-profile \
    --compiler-version-file /tmp/hip1151-t6/hipcc-version-gfx1151.txt \
    --out /tmp/qwen38-prefill-profile/profile.json
```

The exact command of the recorded run, including its `--python`, `--rocprofv3`
and `--roctx-sdk` values, is stored in `artifact.json` under
`provenance.command`. To rebuild `artifact.json` from an existing run without
touching the GPU:

```bash
PYTHONPATH=. python3 benchmarks/results/2026-09-13-qwen38-gfx1151-prefill-kernel-profile/assemble.py \
    --driver-json /tmp/qwen38-prefill-profile/profile.json \
    --run-root /tmp/qwen38-prefill-profile/run \
    --out benchmarks/results/2026-09-13-qwen38-gfx1151-prefill-kernel-profile/artifact.json
```

`scripts/qwen38_gfx1151_prefill_kernel_profile.py` runs three process
boundaries: an unprofiled build child populates a scoped JIT cache, an
unprofiled warm child proves the same workload starts with
`HIPENGINE_REQUIRE_CACHED_BUILD=1` and an empty compiler guard, and rocprofv3
then wraps one child per prompt length with `--selected-regions`, so the trace
holds the measured prefill and not model load, the discarded warmup or session
teardown.

## Protocol and limits

Measured on the physical `gfx1151` host (Radeon 8060S, 40 CUs, wave32, 64 KiB
LDS per CU) with `Qwen3.8-27B-Q4_K_M.gguf` (SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`), BF16 K/V,
bulk prefill, one discarded warmup and one measured pass per length, under the
production execution profile with the Q4 rowtile and GDN capture verifiers
enabled. All three profiled children returned the same first token (9707) and
finite logits.

This artifact reports attribution, not a performance claim: the recorded
prefill rates (431.2, 412.7 and 399.8 tok/s) are single-pass context for the
published resident-sweep row, and no per-kernel confidence interval is
claimed. The prompt is one repeated token id, so attention is not
representative of natural text; both attention owners stay under 3% at every
length. VGPR, SGPR and LDS are the dispatch-recorded resource counts, and
waves per SIMD is derived from the VGPR budget rather than measured from
hardware counters. Owner-to-tensor attribution is a launch-count and
column-count argument over the GGUF tensor table, not a per-dispatch tensor
probe. The bulk path also records per-stage host timers; their sum exceeds the
measured wall time because a fallback alias is recorded next to its parent, so
they are not used here.
