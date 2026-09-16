# Wide-row Q8_0 prefill candidate: `attn_qkv` replay packet

On the identical-operand replay packet where hipEngine measured 11.70x the
pinned comparator, a new wide-row Q8_0 kernel runs **2.570 ms** against the
production dispatch's **19.663 ms** — **7.65x** — and 1.72x the comparator's
complete-operation 1.492 ms. The remaining difference is the activation
representation, and that is measured rather than inferred: the same kernel
reading a pre-converted f16 activation runs **1.303 ms**, at the comparator's
kernel time of 1.339 ms.

The candidate is registered but is not a default. Its f16 operands change
prefill arithmetic, and the calibrated production gates in
`docs/EXECUTION-PROFILES.md` have not been run.

## Result

Operation: `layers.8.attn_qkv` of Qwen3.8-Flash-Next `UD-Q4_K_XL`, rows = 1024,
K = 2560, M = 10240, 53.69 GFLOP. Host: Radeon 8060S / `gfx1151`, 120 GB.
Complete-operation times are GPU events on the stream the kernel launched on.
Every engine was handed the packet's exact weight bytes and activation matrix and
read back through the same float32 output.

| Engine | Kernel | ms | TFLOP/s | max abs err vs exact f64 | Bit-exact vs capture |
| --- | --- | ---: | ---: | ---: | --- |
| hipEngine production (strict) | `coltile8_rowbatch4_f32_f32_out` | 19.663 | 2.73 | 2.06e-06 | yes |
| hipEngine production (wave-scale) | `coltile8_rowbatch4_wave_scale_f32_f32_out` | 17.319 | 3.10 | 2.06e-06 | yes |
| hipEngine best registered int8 WMMA | `iu8_wmma_prefill_f32_f32_out` | 7.129 | 7.53 | 5.30e-06 | no |
| hipEngine best registered f16 WMMA | `wmma_prefill_f32_f32_out@tile16x32` | 4.401 | 12.20 | 2.83e-03 | no |
| **candidate** | `dense_wide256_f32_f32_out` | **2.573 / 2.566** | **20.87 / 20.92** | 2.83e-03 | no |
| comparator, kernel only | `mmb_dense_kernel<128, 256, 64, 64, 1>` | 1.339 | — | — | — |
| comparator, complete operation | plus `mmb_cvt_f32_bf16` | 1.492 | 35.99 | — | — |
| candidate, pre-converted f16 activation | same kernel | **1.303** | — | — | — |

The candidate row is two independent 50-repetition runs; the other rows are one
30-repetition run. The comparator's complete-operation time and its 35.99 TFLOP/s
come from the cross-engine replay unit on this same packet
([`../2026-09-16-cross-engine-replay/`](../2026-09-16-cross-engine-replay/README.md));
its kernel-only 1.339 ms is from this unit's own shared trace.

The tile family is a measured crossover rather than a constant, so all four
layouts are registered and were run twice:

| Layout | Run 1 ms | Run 2 ms | TFLOP/s |
| --- | ---: | ---: | ---: |
| `dense_wide256` (128 columns x 256 rows) | 2.573 | 2.566 | 20.87 / 20.92 |
| `dense_wide128x128` | 2.679 | 2.705 | 20.04 / 19.85 |
| `dense_wide64x256` | 2.974 | 2.987 | 18.06 / 17.98 |
| `dense_wide64x128` | 3.289 | 3.300 | 16.32 / 16.27 |

## The kernel

`hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_dense_wide.{hip,py}`, ported from
`ggml/src/ggml-cuda/mmb.cu` `mmb_dense_kernel<128, 256, 64, 64, 1>` at
`c4aa302294fcd5121af2039fd4d3dee0d472ec03`.

A 256-thread block covers 128 output columns by 256 rows and stages a
dequantized weight tile plus a converted activation tile in 55,296 bytes of LDS.
The K loop is double-buffered through registers, so the next K-tile loads from
global memory while the current one is consumed from LDS. Each weight byte is
read from global memory once per 256 rows instead of once per 4-32 rows, which
takes this projection's derived weight traffic from 7.13 GB to 0.111 GB.

Three deliberate differences from the ported source, all recorded in the `.hip`
header:

- The activation arrives as f32 and is converted to f16 during LDS staging, so
  the variant keeps an f32 ABI with no hidden scratch state. This is the one
  difference that costs real time.
- Operands are f16 rather than bf16. RDNA3 runs both at the same WMMA rate, and
  f16 matches the in-tree f16 WMMA lineage in `gguf_q8_0_prefill.hip`.
- The epilogue stages each wave's 16x16 result through a per-wave LDS region
  with a wave barrier, instead of the source's block-wide barrier pair per output
  tile.

Tail handling is a template parameter, so the exactly-divisible geometry has no
per-fragment predication. Two development steps accounted for most of the final
rate: removing a K-tile staging spill, worth 11% on the layout that was then
fastest (2.911 -> 2.599 ms), and making tail handling a template parameter, worth
13% on the spilling layout (2.911 -> 2.536 ms).

## Kernel-level comparison

One `rocprofv3 --kernel-trace` capture of the replay, both engines in one
process, 83 launches each, identical geometry. `Grid_Size_X` as reported by
`rocprofv3` is `gridDim.x * blockDim.x`.

| Kernel | avg us | VGPR | LDS | Scratch | grid | block |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `q8_0_dense_wide_kernel<128, 256, 64, 64, false>` | 2580.1 | 256 | 55,296 | 0 | (20480, 4) | 256 |
| `mmb_dense_kernel<128, 256, 64, 64, 1>` | 1339.0 | 256 | 55,296 | 0 | (20480, 4) | 256 |
| `mmb_cvt_f32_bf16` | 68.4 | 16 | 0 | 0 | (327680, 1) | 256 |

The two matmul kernels now use identical resources and identical launch
geometry, and differ by 1.93x. What they read is not identical: hipEngine
re-reads a 10.5 MB f32 activation once per column block (80 times, 838 MB) and
converts it in-kernel, while the comparator reads a bf16 activation that ggml
produced once in a 68 us pass.

## The remaining difference is the activation path

A timing-only probe isolates it. The probe is the same kernel with the
activation load changed from two `float4` plus eight `f32 -> f16` conversions to
a single `half8_t` load, reading f32 bytes as f16 — its values are garbage, so it
is an instrument for the load path only. It was built outside the tree and is not
registered.

| Variant | ms |
| --- | ---: |
| f32 activation, converted in kernel | 2.541 |
| pre-converted f16 activation | **1.303** |
| comparator kernel | 1.339 |

Converting the activation once instead of 80 times accounts for the whole
remaining gap. Reaching this in the engine needs a pre-converted input variant, a
conversion kernel, and a dispatch change to run the conversion first — the
pattern `gguf_q8_0_mmq_prefill` already uses with its caller-supplied
`x_d4_ptr`. That is a separate unit.

## Tiling invariance

Ten registered f16 WMMA variants with tilings from 16x16 to 128x256 all reported
the same maximum error against every reference. That is either tiling invariance
or a harness that cannot resolve differences, so the elementwise comparison was
run over a pair matrix with negative controls that must differ:

| Left | Right | Tiling | Differing elements |
| --- | --- | --- | ---: |
| `dense_wide256_f32_f32_out` | `wmma_prefill_f32_f32_out` | 16x32 | 0 / 10,485,760 |
| `dense_wide256_f32_f32_out` | `wmma_prefill_f32_f32_out` | 16x16 | 0 / 10,485,760 |
| `dense_wide256_f32_f32_out` | `wmma_prefill_f32_f32_out` | 64x32 | 0 / 10,485,760 |
| `dense_wide64x128_f32_f32_out` | `wmma_prefill_f32_f32_out` | 64x32 | 0 / 10,485,760 |
| `dense_wide128x128_f32_f32_out` | `wmma_prefill_f32_f32_out` | 32x16 | 0 / 10,485,760 |
| `dense_wide256_f32_f32_out` | `coltile8_rowbatch4_f32_f32_out` | — | 10,484,319 / 10,485,760 |
| `dense_wide256_f32_f32_out` | `iu8_wmma_prefill_f32_f32_out` | — | 10,484,403 / 10,485,760 |

The candidate is bit-identical to the pre-existing f16 WMMA family despite a
completely different tiling, and both negative controls differ on 99.99% of
elements. The candidate therefore introduces no new numerical class: the
2.83e-03 error against the exact reference is the arithmetic of the registered
f16 WMMA family, and it is already present in the engine on shapes where
`wmma_prefill` owns the dispatch.

## Correctness status

Not gate-passed. What is established:

- The output tracks an f16-simulated reference (both operands rounded to f16,
  product accumulated in float64) to max absolute 1.111e-04. That is
  accumulation-order error for K = 2560; a structural bug would miss by the f16
  rounding scale of 2.8e-03 instead.
- `tests/test_gpu_gguf_q8_0_dense_wide.py` covers registry binding for the tile
  family, the build plan, the `in_features % 64` contract, both oracles over
  shapes with and without tails in rows and output columns, and gfx1151 alias
  reachability. 20 tests pass.
- The binary targets `gfx1151` natively (verified from the ELF), so this is not
  gfx1100 code running in a compatibility mode.

Not established: any model-level KL, top-1, determinism, isolation or task gate.
No end-to-end A/B was run; 7.65x is one leaf on one packet, not a model rate.
One geometry, one layer, one chunk and one host were measured, and `ssm_out`,
`attn_gate` and the Q5_1 expert-down shapes were not.

## Reproduction

```bash
export ROCM=<rocm sdk>; export LD_LIBRARY_PATH=$ROCM/lib:$LD_LIBRARY_PATH

# every registered variant on the packet, 30 repetitions each
.venv/bin/python tools/replay_bridge/sweep_variants.py \
  --packet /tmp/replay-bridge/packets/q8-attnqkv-L8-c0 \
  --sweep-tiles --warmup 5 --reps 30 \
  --output benchmarks/results/2026-09-16-dense-wide-q8-prefill-candidate/sweep-registered-variants.json

# the candidate tile family, 50 repetitions, twice
.venv/bin/python tools/replay_bridge/sweep_variants.py \
  --packet /tmp/replay-bridge/packets/q8-attnqkv-L8-c0 \
  --filter dense_wide --warmup 5 --reps 50 \
  --output benchmarks/results/2026-09-16-dense-wide-q8-prefill-candidate/sweep-dense-wide-family-run1.json

# the elementwise tiling-invariance matrix
.venv/bin/python benchmarks/results/2026-09-16-dense-wide-q8-prefill-candidate/verify_tiling_invariance.py \
  --packet /tmp/replay-bridge/packets/q8-attnqkv-L8-c0

# both engines' kernels under one trace
rocprofv3 --kernel-trace --output-format csv -d /tmp/prof-final -- \
  .venv/bin/python tools/replay_bridge/profile_pair.py \
    --packet /tmp/replay-bridge/packets/q8-attnqkv-L8-c0 \
    --shim /tmp/replay-bridge/libmmb_replay.so

# rebuild this directory's artifact.json from the JSON files above
.venv/bin/python benchmarks/results/2026-09-16-dense-wide-q8-prefill-candidate/assemble.py
```

`assemble.py` reads the sweep JSON files and the trace rather than transcribing
numbers, so the artifact cannot drift from the runs it describes.
