# Hyper-connection drift localization

**Date:** 2026-09-22
**Hardware:** AMD Radeon 8060S (gfx1151), Framework Desktop, ROCm therock10-staging-20260828
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL / BF16 KV, chunk 1024
**Artifact:** `artifact.json`
**New engine runs:** none. Derived from four existing captures: the actual-input
projection replay, the two isolated GR production gates, the two corrected
candidates' gates, and the passing dense wide gate as the control.

## Question

The GR (hyper-connection) family's iu8 projection routes are rejected by the
780-row production numerical gate, and two independent corrections that made the
projection *more accurate against FP64* did not help. Localize the drift by
stage and magnitude, then decide between a corrected arithmetic and a
tiling/fusion alternative.

## Stage: the projection boundary, and nothing else

On actual model-produced inputs, strict-order projection followed by the
standalone sigmoid and gated-mean kernels equals the fused production output
**exactly** in all six GR-up captures. The GR flags replace only the projection
kernel; the epilogue is the same code in both arms. So the projection is the
only stage that can drift, and it does, by

| quantity | measured |
| --- | --- |
| strict vs iu8 projection, relative L2 | 1.12e-07 - 3.69e-07 |
| in fp32 ulps (eps = 1.19e-7) | **1-3 ulp** |
| max absolute difference, full arrays | < 5e-4 |

## Propagation: the gate is a cliff at ~5e-9, not a gradient

The gate's 780 rows are 12 prefill-boundary rows plus 64 teacher-forced decode
rows per case. During decode `rows = 1`, so *both* arms take the strict
sigmoid + gated-mean chain: the decode rows differ only through the state
inherited from the prefill. Splitting each capture's rows by that transition:

| route | prefill-boundary KL | decode KL | amplification | gate mean KL | top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense wide 16-47 (PASSES) | 4.41e-09 | 3.83e-04 | 8.7e+04 | 3.796e-04 | 1538/1548 (99.35%) |
| GR-down + compensated | 8.48e-09 | 1.34e-03 | 1.6e+05 | 1.316e-03 | 769/780 (98.59%) |
| GR-up iu8 (baseline) | 1.48e-08 | 1.42e-03 | 9.6e+04 | 1.402e-03 | 772/780 (98.97%) |
| GR-up + 4th plane | 1.70e-08 | 1.64e-03 | 9.7e+04 | 1.614e-03 | 770/780 (98.72%) |
| GR-down iu8 (baseline) | 4.32e-08 | 1.33e-03 | 3.1e+04 | 1.313e-03 | 769/780 (98.59%) |

Across these five routes the prefill-boundary KL spans **9.8x** while the gate
mean KL spans only **4.25x**. Everything above the cliff lands at
1.31e-3 - 1.61e-3 regardless of how far above it the candidate sits. The
sharpest evidence is the pair that differs only by the projection's arithmetic:

- **GR-down + compensated** improved the prefill-boundary KL **5.1x**
  (4.32e-8 -> 8.48e-9) and the gate metric did not move at all: mean KL
  1.310e-3 -> 1.316e-3, top-1 identical at **769/780**.
- **GR-up + 4th plane** improved FP64 MSE 1.76-4.60x while making the
  prefill-boundary KL *worse* (1.48e-8 -> 1.70e-8), and the gate followed it
  down (1.40e-3 -> 1.614e-3).

So the pass condition is a **prefill-boundary logits KL below ~5e-9**,
bracketed by the passing dense wide route at 4.41e-9 and the failing
compensated candidate at 8.48e-9. The decode horizon amplifies by
3.0e4 - 1.6e5, but above the cliff the amplified result is nearly independent
of the perturbation size - arithmetic accuracy alone does not cross it.

## Model-visible effect: margin-limited near-ties

GR-up flips 8/780 top-1 rows and GR-down 11/780. **Every flip is a decode row;
none is a prefill row.** At those rows:

- the teacher's top-1 is the candidate's rank-2 token in 18 of 19 cases (one
  rank-3), with top-5 overlap 0.80-1.00 - the preference *set* is unchanged and
  only its order moves;
- teacher top-2 margins are 0.002-0.24, while the per-row vocabulary-max logit
  delta is 0.38-0.83, so the swaps are forced by near-ties rather than by a
  systematic shift.

## Decision

**The arithmetic route is not closed - it is re-targeted to a cliff-crossing
screen.** The screen is the prefill-boundary logits KL against the strict
parent, measured with one prefill per arm and no decode
(`scripts/qwen4exp_gr_iu8_logits_probe.py` already measures exactly this). The
best GR candidate measured sits at 8.48e-9, **1.9x above** the passing route's
4.41e-9, so a further ~2x reduction is the concrete next step and it costs one
prefill per arm to test. FP64 MSE is an unreliable proxy for that screen: it
improved 16.9-67.4x and 1.76-4.60x in the two corrections while the screen moved
5.1x better and 1.15x worse respectively. **Stop rule: do not spend another
780-row production gate run on a GR candidate until its prefill-boundary KL is
below ~5e-9.**

**Fusion/tiling is retained as the path that needs no numerical gate at all.**
The production default is already bit-exact against the strict chain, so a
restructuring that preserves the projection's summation order and the
epilogue's arithmetic sits below the cliff by construction. The family's cost is
structural - 2523.8 ms of reads against 1098 ms of matmuls consuming them, with
the 10240-wide branch tensors traversed repeatedly. Concrete levers, in order of
how little arithmetic risk they carry:

1. The GR gate buffer (rows x 10240 f32, 168 MB per layer per chunk) is written
   by the fused kernel and **never read** by the model path -
   `attention_read.gate` and `ffn_read.gate` have no consumer.
2. Fuse the grouped-rmsnorm producer of `normalized` so the 10240-wide tensor is
   never materialized, replicating the norm's reduction order to stay bit-exact.
3. Recover ILP in the Q8_0 projection without changing any output's summation
   order.

## Blocked confirmation

`mixed` is the MoE router's input and `qwen35_router_select` is a discrete
top-k on the GR output, so the router is the only discrete amplifier in the
dataflow; counting differing selections per decode step would confirm that step
directly. `scripts/qwen4exp_gr_router_localize.py` is written and fails on host
memory: an idle `hipengine serve` (pid 4175913, started 2026-09-21 17:48, 2m08s
CPU) holds 98.6 GB of GTT in `/home/lhl/hipEngine-main`, the production generator
needs 95.7 GB including reserve, and 25.5 GB is free at context 584 as well as
4168 (so the requirement is weights/scratch, not KV). This does not affect the
localization above, which rests on five measured routes.

## Limits

Stage and projection magnitudes are measured on actual model inputs. The
amplification factor is measured per route (3.0e4 to 1.6e5) and is not a
universal constant; the cliff location is bracketed by one passing and four
failing routes on this model, quant, chunk and host. The implied prefill budget
uses the empirical bracket, not the tightest amplification factor. No
performance claim and no default change results from this packet.
