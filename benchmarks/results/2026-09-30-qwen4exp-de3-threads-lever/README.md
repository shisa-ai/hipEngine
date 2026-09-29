# DE-3: the block-size lever on the decode linear family

**Date:** 2026-09-30
**Hardware:** AMD Radeon Pro W7900 (gfx1100), ROCm 7.15 / clang 23, gfx1151 code object
**Script:** `scripts/qwen4exp_de3_starved_shape_screen.py`
**Artifact:** `de3_threads_screen.json`

## Question

The decode linear owner `gguf_k_pack8_prefill_out_kernel<float, float, 8>` accounts for
24.005 ms/step over 1491 dispatches. Its eight shapes are byte-exact, and the measured
rate depends on grid size rather than arithmetic intensity: every shape at 320 or more
workgroups reaches 90.8-102.9% of the 221 GB/s roof, while every shape below 100
workgroups reaches only 42.4-58.8%.

The launcher already takes a `threads` argument that defaults to 128. The question was
whether raising it lifts the per-workgroup parallelism at the starved shapes without
touching the grid.

## Method

Each shape is swept as the engine sweeps it: one distinct device weight per real tensor
in the step (97 / 96 / 24 for the starved shapes, up to 134 for the largest), so the
working set matches the trace and nothing is L3-resident. The sweep is timed whole and
divided by the sweep's byte total.

**This discipline was forced by a first attempt that got it wrong.** That version timed a
single 3.48 MB weight across 35 re-reads and reported 326.9 GB/s - above the 221 GB/s
roof, therefore invalid, because the entire weight was L3-resident. The real step streams
4477 MB. Every row below is marked with whether its sweep exceeds the 64 MB L3.

`out=512` is the one shape whose 33.4 MB sweep still fits in L3, so its absolute rate is
not comparable to the trace; its row is retained only for the sign of the change.

## Validation

At `threads=128` the screen reproduces the trace rate at every shape it can be compared
at, which is what makes the `threads=256` column citable:

| shape | screen 128 | trace | 
|---|---:|---:|
| 248320x2560 | 225.1 GB/s | 227.5 GB/s |
| 12288x2560 | 215.0 | 220.4 |
| 2560x6144 | 202.6 | 206.0 |
| 6144x2560 | 198.2 | 209.8 |
| 320x10240 | 92.4 | 93.8 |
| 640x2560 | 121.5 | 130.0 |

## Result

`pack8_gemv_f32_f32_out`, best-of-12 sweeps, second of two runs (runs agree within 0.5%):

| shape | thr=128 µs | thr=256 µs | Δ time | GB/s 128 | GB/s 256 | save ms/step |
|---|---:|---:|---:|---:|---:|---:|
| 320x10240 | 3654.8 | 2469.2 | **−32.4%** | 92.4 | **136.8** | **+1.168** |
| 640x2560 | 1375.0 | 1108.5 | **−19.4%** | 121.5 | **150.8** | **+0.249** |
| 512x2560 | 270.5 | 206.1 | −23.8% | 123.6 | 162.2 | +0.073 |
| 10240x2560 | 17488.5 | 17681.4 | +1.1% | 213.4 | 211.1 | −0.075 |
| 2560x6144 | 8000.3 | 7858.4 | −1.8% | 202.6 | 206.3 | +0.077 |
| 248320x2560 | 3000.5 | 3046.2 | +1.5% | 225.1 | 221.7 | −0.045 |
| 6144x2560 | 3035.1 | 2977.1 | −1.9% | 198.2 | 202.1 | +0.055 |
| 12288x2560 | 1865.5 | 1889.5 | +1.3% | 215.0 | 212.3 | −0.023 |

**The sign split is exact and follows `out_features`, not size:** every shape with
`out_features <= 640` gains 19.4-32.4%, and every shape with `out_features >= 2560` loses
1.1-1.9%. Gating the change at 640 nets **+1.490 ms/step = 6.2% of the 24.005 ms family =
2.71% of the 54.603 ms/step decode kernel sum**.

The competing variant `gemv_f32_f32_out` was measured at every shape and is decisively
worse - 41-53% of roof everywhere, and insensitive to `threads` - so `pack8` is the right
family and this is not a variant-selection miss.

## Caveats

- **`threads` changes arithmetic.** 256 threads per 8-output workgroup means 32 threads
  per output rather than 16, so the K-reduction tree differs. Measured divergence from
  the 128-thread result is max relative 2.6e-7 (float32 rounding level) at both 320x10240
  and 640x2560. This is a T1/T2 arithmetic change and requires the production numerical
  envelope before promotion; it is not exact.
- The 1.490 ms/step figure is a per-kernel projection from a microbenchmark, not an
  end-to-end measurement.
- `threads` is validated to be one of {64, 128, 256}; 512 was rejected by the launcher.
