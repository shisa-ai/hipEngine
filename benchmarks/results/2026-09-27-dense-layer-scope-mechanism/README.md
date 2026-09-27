# Dense layer scope: the boundary at 16 is a drift budget, not a tile or shape limit

**Date:** 2026-09-27
**Hardware:** AMD Radeon 8060S (gfx1151), Framework Desktop, ROCm therock10-staging-20260828
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL / BF16 KV
**Artifacts:** `mechanism.json`, `layer-trace.py`
**New engine runs:** one instrumented production-default prefill (1024 tokens).
Everything else is derived from the seven published scope gates.

## Verdict

The split is measured, the mechanism is a **numerical drift budget**, and it
cannot be moved by retiling. Layers 0-15 run the exact coltile chain and 16-47
run the wide tile because the wide tile's arithmetic - the f16 WMMA arithmetic -
exhausts the calibrated envelope when carried deep enough. `dense_wide256` is a
retiling of that arithmetic, not a different one, so no tile geometry change can
move the boundary. The extension path is to reduce per-layer operand rounding.

## What actually runs where (measured)

Instrumenting the three Q8_0 dense selectors on a production-default 1024-token
prefill:

| route | transition | count |
| --- | --- | ---: |
| wide | `coltile8_rowbatch4_wave_scale_f32_f32_out` -> **`dense_wide256_f32_f32_out`** | **264** |
| wide | `coltile8_rowbatch4_wave_scale_f32_f32_out` -> unchanged (**exact chain**) | **134** |
| wmma | `prefill_f32_f32_out` -> unchanged | 398 |
| iu8 | `coltile8_rowbatch4_wave_scale_f32_f32_out` -> unchanged | 398 |

264 / 398 = **0.663**, against the scope's 32/48 = 0.667. The wide route is the
only route claiming anything; the f16 WMMA rewrite is inactive in this lane and
`HIPENGINE_QWEN4_EXP_Q8_IU8_WMM` reads `0` in this binding. The 134 declines are
the layer scope and nothing else - the role, quant, variant, and shape are
identical on both sides of the split, which is what makes the boundary purely
numerical rather than a shape or coverage gap.

## Why the boundary sits there: a monotone drift budget

Seven scopes have now been gated on the same fixture and protocol (12 cases, 4
categories at 512/1024/4096, 128 decode transitions, production profile, chunk
1024). Mean KL against the exact coltile chain, by how many layers carry the f16
arithmetic:

| scope | wide layers | mean KL | p95 | max | top-1 | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 32-47 | 16 | 5.81e-05 | 1.75e-04 | 1.36e-02 | 0.99677 | pass |
| 28-47 | 20 | 9.34e-05 | 4.06e-04 | 9.67e-03 | 0.99742 | pass |
| 20-47 | 28 | 2.53e-04 | 1.19e-03 | 1.99e-02 | 0.99419 | pass |
| **16-47** | **32** | **3.80e-04** | **1.76e-03** | 1.53e-02 | 0.99354 | **pass, promoted** |
| 12-47 | 36 | 6.47e-04 | 3.61e-03 | 2.25e-02 | 0.99031 | fail (`general_en`) |
| 8-47 | 40 | 8.44e-04 | 4.07e-03 | 7.30e-02 | 0.98966 | fail (max KL, top-1) |
| 0-47 | 48 | **1.099e-03** | 5.017e-03 | 3.913e-02 | 0.9922 | fail |
| limit | | 1e-3 | 5e-3 | 5e-2 | 0.99 | |

The growth is **smooth and monotone - there is no cliff**. The mean-KL limit is
crossed at roughly **layer 11**; 16-47 was the deepest certified scope and sits
at 0.38x of the mean limit and 0.35x of p95.

The per-layer cost of the arithmetic rises as the scope deepens, which is
amplification rather than a per-layer constant: 3.6e-6 per wide layer at 32-47
against 2.3e-5 per wide layer at 0-47. A geometric amplification model with
g ~ 1.10 per downstream layer fits the 16/20/28-layer points to within ~10%, and
over-predicts the deepest extensions - so the shape is amplification, but the
model is not tight enough to place a boundary without gating the scope.

## Why retiling cannot move it

The gate packet for 12-47 states it directly, and it is the load-bearing fact
for this sprint:

> "The route's arithmetic is the f16 WMMA arithmetic (`dense_wide256` is a
> retiling of it, not a reassociation - both emit `candidate_logits_sha256
> 65370710c0b9c40f` at 16-47), and that arithmetic is what the WMMA `0-47` arm
> already failed at (mean KL 1.099e-3)."

So the wide route did not buy accuracy - it bought the *same* arithmetic at
4.65x lower per-layer cost. The 0-47 failure predates it and belongs to the f16
operand rounding, not to the tile. Sprint H's GR finding is the mirror image:
there the fastest tile was numerically rejected; here the fast tile is
numerically identical to the already-certified arithmetic and the boundary is set
by something the tile does not control.

## The prize, and what would actually collect it

| quantity | value |
| --- | ---: |
| layers 0-15 on the exact chain | 3262.5 ms (203.9 ms/layer) |
| layers 16-47 on the wide tile | 1404.2 ms (43.9 ms/layer) |
| ratio | **4.65x** |
| layers 0-15 at the wide tile's per-layer cost | 702.1 ms |
| **extension prize** | **2560.4 ms** |

That is 19.7% of the 12.97 s gap, and it is the single largest dense block left.
It is not reachable by scope tuning: the reachable extension is at most a few
layers (~0.4 s by the per-layer costs) and it would sit at the numerical limit
with a real chance of failing its own gate.

The mechanism says what would work instead: **reduce the per-layer operand
rounding while keeping the tile.** The wide kernel stages weights as f16 and
converts activations to f16, so the drift is operand rounding, not accumulation
order. A multi-term f16 split - carry each operand as hi + lo and run 2-3 WMMA
passes against the same LDS tile - recovers most of the f32 accuracy at the same
memory traffic, which is the budget the kernel is actually bound by. The
comparator does exactly this: the PR #63 env capture carries
`LLAMA_MMB_F32SPLIT=2`.

**This is untested here.** No split-precision variant was written or measured.
The claim is that the boundary is operand precision, which follows from the
arithmetic identity above; that a split actually crosses the gate is a
prediction, not a result.

## What this does not say

- Not a rejection of the wide route. 16-47 stays promoted.
- Not a statement that 13-47, 14-47, or 15-47 would fail. They were not run.
  The measured facts are 8-47 and 12-47 fail and 16-47 passes with 2.6x headroom.
- No timing was taken for this artifact. The 2560.4 ms is an interpolation from
  the published per-layer costs of the two routes, not a measurement of an
  extended run.
