# The LDS-staged f32 tile beats the dense-projection leaf by 23.6%

**Date:** 2026-09-29
**Host:** Radeon 8060S (gfx1151), ROCm platform 10.0.0, hostname `gfx1151`,
machine-id `55ea6c509d0b49eea8de7094a1023668`, kernel `7.1.6-1-cachyos`,
`power_dpm_force_performance_level=high`.
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL
**Tensor:** `attn_qkv.weight`, layers 0/9/22/34/46, Q8_0, 2560 → 10240, rows=1024
**Command:** `scripts/qwen4exp_dense_projection_ab.py --model-root
/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL --tensor-suffix
attn_qkv.weight --in-features 2560 --out-features 10240 --rows 1024 --repetitions 5
--output /tmp/dp8-staged.json`
**Artifact:** `ab.json` (this directory)

## Result

The LDS-staged f32 prototype written for the GR projection screen — a general
tiled GEMM over Q8_0 weights, BM=128 × BN=64 × BK=32, 256-thread block, f32 FMA
on f32 operands — is **faster than the promoted dense-projection leaf on every
layer**, and it is the best variant in the sweep at all five.

| layer | promoted leaf ms | staged_f32 ms | speedup |
|---|---:|---:|---:|
| 0 | 16.1363 | 13.1740 | **1.2248** (+22.5%) |
| 9 | 16.1803 | 13.1611 | **1.2294** (+22.9%) |
| 22 | 16.1757 | 13.0907 | **1.2357** (+23.6%) |
| 34 | 16.1929 | 13.0773 | **1.2383** (+23.8%) |
| 46 | 16.1915 | 13.0574 | **1.2400** (+24.0%) |

Median **+23.6%**. Achieved throughput goes 3315–3327 GFLOP/s → **4075–4112
GFLOP/s** (4.1 TFLOP/s), 6.70% of the F32 FMA peak.

## Baseline correction

The harness's `PRODUCTION_VARIANT` named
`coltile8_rowbatch4_wave_scale_f32_f32_out`, the **pre-DP-8** parent, not the
software-pipelined leaf that
`hipengine.runtime.gguf_linear._raw_k_coltile_pipelined_dispatch` actually
selects. That made the sweep report 17.39 ms where the real default is
16.19 ms, understating every candidate. The constant is corrected in this unit.
The two leaves are bit-identical to each other (the DP-8 promotion measured
`max_abs_diff = 0.0`), so the reference change does not alter any
`bit_identical_to_production` verdict — it only stops the baseline being wrong.

The numbers above are computed against the corrected, real default.

## Numerical character

`max_abs_diff_vs_production` is **6.9e-06 to 1.05e-05** across the five layers,
`bit_identical = false`.

That magnitude is consistent with **f32 summation-order drift** for K=2560
(sqrt(K)·eps·|Σ| ≈ 3e-6·|Σ|, so |Σ| ≈ 3) and is **two to three orders of
magnitude too small** to be f16 operand rounding, which is what the
`dense_wide256` WMMA tile produces (relative L2 4.7e-05, max abs 7.3e-05 on the
GR shape). The prototype does the multiply in f32 FMA on f32 operands, so no
operand is rounded.

This was **not** measured as a relative figure; it is an inference from the
absolute magnitude against the known f16 error of the sibling route. Treat the
"summation-order, not operand precision" characterization as strong but
inferred, not as a measured relative-error bound.

## Why this matters

`dense_projection` is the largest family in the `code-p4096` attribution:
**4553.007 ms of a 15741.963 ms kernel union, 28.9% of prefill**. A 23.6%
reduction on the whole family would be ~1075 ms, about **6.8% of prefill**.

**That family-wide figure is an upper bound, not a measurement.** This run
measured one tensor suffix (`attn_qkv.weight`) at one row count. `dense_projection`
also contains `attn_output` and the dense-layer MLP projections, whose shapes
differ and whose staged-f32 performance is unmeasured. Do not quote 6.8% as a
result.

## Status: candidate, not promoted

Nothing changed in the engine. The prototype is still unregistered and
unreachable from the engine. Promoting it into the dense path is **not** a
strict change: the output is not bit-identical to the current leaf, so it needs
the `docs/EXECUTION-PROFILES.md` production gate (mean/p95/p99/max KL, top-1 by
category, determinism, isolation, task quality) plus a registered strict
fallback. That gate has not been run.

This is the same arithmetic question as HC-1, and the answer is now measured on
two shapes: the traffic win is available at f32 precision, worth 2.39× on the GR
projection (K=320) and 1.24× here (K=2560). The GR shape gains more because it
is the more traffic-starved of the two.

## Provenance caveat

Weights are real (layers 0/9/22/34/46 `attn_qkv.weight` Q8_0 from the named
GGUF). Activations are **synthetic**: per-layer deterministic standard normal
scaled by 0.5. The harness's own note records this, and it is the right choice
for separating kernel variants on identical operands — but it is not a
natural-activation measurement, and the kernel's performance on real activation
ranges is a follow-up.

The timing boundary is a hip-event window around `repetitions=5` launches after
a warm launch, per (layer, rows, variant). This is a per-kernel microbenchmark,
not an end-to-end engine rate.
