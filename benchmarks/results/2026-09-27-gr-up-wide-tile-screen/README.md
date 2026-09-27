# GR up wide-tile route: 4.38x projection, cliff screen

**Date:** 2026-09-27
**Hardware:** AMD Radeon 8060S (gfx1151), Framework Desktop, ROCm therock10-staging-20260828
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL / BF16 KV, chunk 1024
**Source:** `479be908f` + the default-off route in this unit
**Artifacts:** `projection-screen.json`, `prefill-boundary-probe.json`,
`qwen4exp_gr_projection_screen.py`

## Why

The fused GR up+sigmoid+gated-mean kernel is the largest single kernel in the
code-p4096 prefill: 2589.5 ms of 17114 ms (15.1%), 6.5 ms per launch at
`rows=1024, K=320, N=10240`. Two equal-accumulator retiles (row-reuse RB8/HT1
0.80x, col-reuse RB2/HT4 1.007x) were already measured and rejected on
2026-09-06, so this screen tests a different lever: the tile geometry the dense
route already ships.

## Measured floor and the limiter

Real layer weights, five GR layers, order-rotated, `rows=1024`. The wide arms
are the promoted dense tile (`dense_wide256`); the exact arms are the
production fused kernel and its registered strict unfused chain.

| arm | up K=320 N=10240 | down K=10240 N=320 |
| --- | ---: | ---: |
| production fused exact | **6.538 ms** (1.03 TFLOP/s) | **2.807 ms** |
| strict unfused chain | 7.899 ms (0.83x) | — |
| wide tile, f32 input | 1.493 ms (4.38x) | 1.839 ms (1.53x) |
| wide tile, f16 input | 1.419 ms (4.61x) | 1.299 ms (2.16x) |

The limiter is **load issue on the K=320 weight stream, not the epilogue and not
the reduction.** Two independent readings agree:

- The production fused kernel runs at **1.03 TFLOP/s = 3.9% of the gfx1151 FP32
  peak**, which no reduction- or epilogue-bound kernel would do.
- The same weights through a 256-column tile reach 4.50 TFLOP/s, and the entire
  difference is tile geometry: the fused kernel's `N`-column walk re-reads the
  320-element weight row per output column, while the wide tile stages it once
  per block.

Fusion is already earning its keep, but only **1.21x** over the strict chain
(6.538 against 7.899 ms). The wide tile is **4.38x** over the fused kernel.

At the code-p4096 shape this is 384 launches per leg:

| leg | saving per launch | prefill saving |
| --- | ---: | ---: |
| up | 5.045 ms | 1937 ms |
| down | 1.508 ms | 579 ms |
| **total** | | **2516 ms of 17114 ms (14.7%)** |

## The numerical constraint

The projection screen's wide arms are **not** bit-identical to the parent:
relative L2 4.709e-05 (up) and 2.943e-04 (down), against the 1.12e-07 - 3.69e-07
that the rejected GR-up iu8 route measured. The GR family's gate is a cliff at a
prefill-boundary logits KL of ~5e-9, so operand-class drift is the binding
question, not the speedup.

A default-off `HIPENGINE_QWEN4_EXP_GR_WIDE` route was added (mirroring the
existing `GR_IU8` branch: wide projection plus the retained exact sigmoid and
gated-mean kernels) and screened with
`scripts/qwen4exp_gr_iu8_logits_probe.py --route wide`:

| case | prefill-boundary KL vs incumbent | top-1 | argmax gap / drift |
| --- | ---: | ---: | --- |
| code-p512 | **5.17e-11** | 1.0 | 23.23 / 0.48 |
| general_en-p512 | **3.40e-10** | 1.0 | 21.92 / 1.76 |

Both sit **8.5-85x below** the passing bracket (dense wide 4.41e-9, GR-down
compensated 8.48e-9) and the run was deterministic in both arms.

**This is a screen, not a qualification, and the margin is not as large as it
looks.** The publication row has a ~22-logit argmax gap, so the softmax is
effectively one-hot and KL is insensitive to the logits that actually moved:
median absolute logit drift is 0.095 and max is 1.20, i.e. the logits change
substantially while the *distribution* at that position barely moves. The
screen therefore cannot rule out the flip mechanism the localization identified
(margin-limited near-ties at decode rows, teacher top-2 margins 0.002-0.24).
The 780-row production gate is the binding test and is required before any
promotion.

## Status

Route is **default-off**. `benchmarks/README.md` and `benchmarks/CHANGELOG.md`
are untouched because there is no retained benchmark row yet: the projection
screen is diagnostic (no end-to-end prefill rate) and the gate has not run.

## Next

1. Run `qwen4exp_layer2_profile_gate.py --candidate production_gr_up_wide`
   (12 prompts, `--decode-steps 64` = 780 rows) and record mean/p95/max KL,
   top-1 by category, determinism, isolation, and the task gate.
2. If it passes, measure the end-to-end prefill rate and promote; if it fails,
   record the measured mean KL against the cliff and close the GR-up wide
   direction the way the iu8 direction was closed.
