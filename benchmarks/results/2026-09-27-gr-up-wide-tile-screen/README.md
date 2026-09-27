# GR up wide-tile route: 4.38x projection, cliff screen

**Date:** 2026-09-27
**Hardware:** AMD Radeon 8060S (gfx1151), Framework Desktop, ROCm therock10-staging-20260828
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL / BF16 KV, chunk 1024
**Source:** `479be908f` + the default-off route in this unit
**Artifacts:** `projection-screen.json`, `prefill-boundary-probe.json`,
`production-gate-780.json`, `short-prompt-gate-invalid.json`,
`qwen4exp_gr_projection_screen.py`

## Verdict

The wide tile is **4.38x faster on the GR-up projection and rejected by the
780-row production gate**. It fails on the numerical envelope only: mean KL
1.554e-3, p95 7.081e-3, max 7.349e-2, top-1 768/780. Determinism, state
repeatability, and lifecycle all pass. The route stays default-off.

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

## The screen did not predict the outcome

The 780-row production gate
(`qwen4exp_q8_repair_depth_gate.py --candidate production_gr_up_wide
--decode-steps 64 --repeats 3`, 12 canonical 512/1K/4K cases) ran the route for
real - 6912 direct calls, shapes identical to the GR-up iu8 gate's - and
rejected it:

| metric | measured | threshold | verdict |
| --- | ---: | ---: | --- |
| mean KL | 1.554e-03 | <= 1e-03 | FAIL |
| p95 KL | 7.081e-03 | <= 5e-03 | FAIL |
| p99 KL | 1.797e-02 | <= 2e-02 | pass |
| max KL | 7.349e-02 | <= 5e-02 | FAIL |
| top-1 | 768/780 (98.46%) | >= 99% | FAIL |
| per-category top-1 | 97.95% - 99.49% | >= 97% | pass |

All 12 top-1 mismatches are flip-eligible and none is outside it; every
category, shape, and the steady transition fail. Determinism, state
repeatability, and lifecycle pass, so the rejection is purely numerical.

This lands on the same plateau as every other GR arithmetic route:

| route | projection drift (rel L2) | prefill-KL screen | gate mean KL | top-1 |
| --- | ---: | ---: | ---: | ---: |
| GR-up iu8 (rejected) | 1.12e-07 - 3.69e-07 | 1.48e-08 | 1.402e-03 | 772/780 |
| GR-up wide (this) | **4.709e-05** | **5.17e-11** | 1.554e-03 | 768/780 |

**The screen is not a predictor.** A route with 300x more projection drift
screened 285x *better* on prefill-boundary KL and failed at the same place. The
cause is visible in the probe rows themselves: the publication row has a
~22-logit argmax gap, so the softmax is effectively one-hot and KL is blind to
the logits that actually moved (median absolute drift 0.095, max 1.20).

Across the routes measured so far the gate outcome is insensitive to the
projection's accuracy, including corrections that were *more* accurate than the
incumbent. That is consistent with the unconfirmed hypothesis that the
downstream discrete step - the MoE router's top-k, which consumes the GR
output - saturates the metric, but no router-selection capture exists yet and
this artifact does not establish it.

## A screen that reports a pass without running the route

The first gate attempt used the harness default prompt suite. That suite is
39-71 tokens, below the GR routes' `rows > 256` admission, so the route never
ran and the candidate was **bit-identical** to strict: 780 rows of
`max_abs_logit_delta: 0.0`, `kl_mean: 0.0`, `top1_agreement: 1.0`.

That reads as a perfect pass. The harness caught it only because it
independently counts dispatches: `candidate_dispatch_calls: 0`,
`measurement_valid: false`. The capture is kept as
`short-prompt-gate-invalid.json` because it is a clean negative control for the
valid run - an unchanged candidate scores exactly zero on this harness, so the
1.554e-3 above is the route's own contribution - and because it is a trap any
future GR or dense candidate can fall into.

## Status

Route is **default-off**. `benchmarks/README.md` and `benchmarks/CHANGELOG.md`
are untouched because there is no retained benchmark row: the projection screen
is diagnostic (no end-to-end prefill rate) and the gate rejected the route.

## Next

The GR family is closed to arithmetic change: every measured route, including
more-accurate ones, saturates the gate at 1.3e-3 - 1.6e-3. Only a bit-exact
change can land, and the two equal-accumulator retiles were already measured
(0.80x, 1.007x). Reopening needs either a confirmed router-flip mechanism with
a way to make the discrete selection robust, or a bit-exact route that beats
the fused kernel without touching operand or accumulation order.
