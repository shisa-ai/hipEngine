# GR up-projection: the LDS staging win is separable from f16 operand rounding

**Date:** 2026-09-29
**Host:** Radeon 8060S (gfx1151), ROCm platform 10.0.0, hostname `gfx1151`,
machine-id `55ea6c509d0b49eea8de7094a1023668`, kernel `7.1.6-1-cachyos`,
`power_dpm_force_performance_level=high`.
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL, `hc_ffn_up.weight` / `hc_ffn_down.weight`
**Shape:** rows=1024, low_rank K=320, 4 branches x hidden 2560 = 10240 outputs
**Command:** `scripts/qwen4exp_gr_projection_screen.py --model-root
/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL --rows 1024
--repetitions 5 --output /tmp/gr-staged-f32-final.json`
**Artifact:** `screen.json` (this directory)

## The question

The GR up projection is the worst-ratio family in the Qwen4 ledger. The
production fused kernel takes 6.54 ms; the promoted `dense_wide256` WMMA tile
takes 1.43 ms, a 4.59x gap. The recorded explanation is weight traffic: the
fused kernel re-reads a 3.48 MB Q8_0 weight tensor once per 4-row batch
(3.565 GB per launch) because a block holds only 8 weight rows, while the wide
tile stages a dequantized weight tile in LDS once per K-tile and reuses it
across every row in the block.

The wide tile is also rejected: RDNA3 matrix cores take only f16/bf16 operands,
so its multiply rounds both operands to 2^-11 and it misses the production
numerical gate.

Those two facts were recorded together, which left the important question open:
**is the traffic mechanism itself unusable at production precision, or is only
the matrix-core multiply?** If the mechanism needs f16, every route that wants
it (HC-1 hyper_connection, DP-1 dense_projection layers 8-15) has to change
arithmetic to collect it. If it does not, the mechanism is free and the
precision cost is a separate, optional 1.9x.

## What was measured

A prototype (`gguf_q8_0_gr_staged_f32.hip`) keeps the wide tile's staging and
tiling — BM=128 rows x BN=64 outputs x BK=32 K (one Q8_0 block), 256 threads,
LDS-staged f32 weight and activation tiles — and replaces the WMMA multiply with
f32 FMA on f32 operands.

### Up projection, rows=1024, K=320, N=10240, Q8_0

| arm | median ms | speedup vs production | error vs production (relative L2) | max abs |
|---|---:|---:|---:|---:|
| `fused_exact` (production) | 6.5387 | 1.000 | 0 (reference) | 0 |
| `unfused_exact` (strict chain) | 7.8743 | 0.830 | 0 | 0 |
| **`staged_f32`** | **2.7380** | **2.388** | **7.442e-08** | 7.451e-08 |
| `wide_f32in` | 1.5094 | 4.332 | 4.709e-05 | 7.327e-05 |
| `wide_f16in` | 1.4252 | 4.588 | 4.709e-05 | 7.327e-05 |

### Down projection, rows=1024, K=10240, N=320, Q8_0

| arm | median ms | speedup vs production | error vs production (relative L2) |
|---|---:|---:|---:|
| `exact` (production) | 2.8098 | 1.000 | 0 (reference) |
| `staged_f32` | 3.2722 | 0.859 | 1.721e-06 |
| `wide_f32in` | 1.8398 | 1.527 | 2.943e-04 |
| `wide_f16in` | 1.2938 | 2.172 | 2.943e-04 |

## Findings

**1. The traffic mechanism does not require f16 operands.** `staged_f32`
collects 2.388x on the up projection with a relative-L2 deviation of 7.44e-08
from the exact production kernel — about 1.25 f32 ulp on sigmoid outputs in
[0,1], i.e. the summation-order floor for K=320. The wide tile's deviation is
633x larger at 4.71e-05.

**2. The wide arms' error is entirely operand rounding, and the f32-input
variant does not remove it.** `wide_f32in` and `wide_f16in` report *identical*
error (relative L2 4.709e-05, max abs 7.327e-05, both to every printed digit).
Both stage the dequantized *weight* tile as f16 halves; the f32-input variant
only changes how the activation enters the multiply. So converting the
activation to f32 buys nothing — the weight operand is what rounds.

**3. The residual 1.92x is the arithmetic, not the memory.** `staged_f32`
(2.738 ms) versus `wide_f16in` (1.425 ms) is the scalar-FMA-versus-matrix-core
throughput difference at the same tiling. That, and only that, is what the f16
rounding buys.

**4. The mechanism loses on the down projection.** `staged_f32` is 0.859x there
(3.2722 ms vs 2.8098 ms). At rows=1024, K=10240, N=320 the grid is
(320/64) x (1024/128) = 40 blocks for 40 CUs — a single wave with no latency
hiding, which is why the loss appears here and not on the up projection. This is
a tiling/occupancy defect of the prototype at this shape, not evidence against
the mechanism; the down projection is also not where the family's time is
(896.3 ms down vs 2589.5 ms up in the ledger window).

## What this changes

The ledger recorded the wide tile's speed and its gate failure as one fact and
concluded that a route wanting the traffic win needs "a changed arithmetic".
That is now half wrong: **a route can have the traffic win at production
precision.** The corrected statement is that the traffic win is free and the
remaining 1.92x costs f16 operand rounding.

This does not by itself promote anything. `staged_f32` is a prototype: it is not
registered as a production variant, is not reachable from the engine, and its
deviation from the exact kernel — while at the f32 summation-order floor — is
still a non-bit-exact arithmetic change, so a promotion would have to clear the
`docs/EXECUTION-PROFILES.md` production gate (mean/p95/p99/max KL, top-1 by
category, determinism, isolation, task quality) rather than inherit strict
parity. That gate has not been run.

## Reproduction

The prototype is in-tree at
`hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_gr_staged_f32.hip` with the
launcher in the sibling `.py`. The screen arm is `staged_f32` in
`scripts/qwen4exp_gr_projection_screen.py`. The numerical contract is
`tests/test_gpu_qwen4exp_gr_staged_f32.py`, which pins the claim this packet
rests on: the staged f32 arm stays within the f32 summation-order floor
(relative L2 <= 1e-6) of the exact fused kernel at five shapes including the
full production shape (rows=257, K=320, h=2560), and is deterministic across
repeats.

Build environment: TheRock staging `therock10-staging-20260828`,
`HIPENGINE_HIP_ARCH=gfx1151`, `HIPENGINE_COMPILER_VERSION_FILE` pointing at the
recorded `hipcc-version.txt`.

## Provenance caveat

The timing boundary is an operation-complete host call over three launches for
the epilogue-bearing arms, after two warmups, with the layer and the arm order
rotated per repetition. It excludes any end-to-end engine cost. The weights are
real (layer 0/9/22/34/46 `hc_ffn_up`/`hc_ffn_down` from the named GGUF); the
activation and the `normalized` epilogue input are synthetic (`np.random.default_rng(20260926)`,
sigma 0.2), matching the existing screen protocol. This is not an end-to-end
benchmark and the rows are not natural-token activations.
