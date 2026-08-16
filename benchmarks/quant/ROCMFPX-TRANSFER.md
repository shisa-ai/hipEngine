# ROCmFPX transferable-mechanism campaign

This is the canonical comparison index for the five mechanisms screened from
the read-only ROCmFPX audit against hipEngine's Qwen3.6-35B-A3B paths on the
ZBook/Radeon 8060S (`gfx1151`). It complements the [quantization-quality
protocol](README.md): that protocol answers *how faithfully an exact artifact
runs*, while this report answers *which implementation ideas transfer to
hipEngine and survive complete correctness/performance gates*.

Aggregate machine-readable index:
[`2026-08-16-qwen36-35b-gfx1151-rocmfpx-transfer-campaign.json`](../results/2026-08-16-qwen36-35b-gfx1151-rocmfpx-transfer-campaign.json).

## Binding scope

- **Opportunities 1–4:** repaired packed-PARO revision
  `437eba06df05aad71a4dacdcaf3fff70ae1ee8a1`, 40 layers, BF16 KV. This
  artifact is implementation-correct but remains **quality-traded** versus the
  admitted Q4_K_M baseline.
- **Opportunity 5:** admitted MTP-bearing
  `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`. The packed-PARO snapshot has no
  NextN sidecar, so it cannot honestly host the adaptive-MTP test.
- **Hardware:** HP ZBook Ultra G1a, Ryzen AI MAX+ PRO 395 / Radeon 8060S,
  `hip_gfx1151`, ROCm 7.15.
- **External code:** none was imported. ROCmFPX remained a read-only source of
  hypotheses; implementation and measurement stayed in this repository.

The repaired campaign baseline is
[`2026-08-16-qwen36-35b-gfx1151-rocmfpx-transfer-baseline.json`](../results/2026-08-16-qwen36-35b-gfx1151-rocmfpx-transfer-baseline.json).

## Decision matrix

| # | Transferable opportunity | Result | Decision |
|---:|---|---|---|
| 1 | True small-M T16/pack8 weight reuse | Existing output-column-tiled route improves c4 aggregate/median-step **+3.41%/-2.50%** and c8 **+4.00%/-3.96%**; all c4/c8 IDs match independent c1 | **Keep existing default**; do not build a duplicate |
| 2 | One-plane Q8_1 activation + `sudot4` | Projection screens reach **1.02x–1.83x**, but operation-complete SiLU max KL is **0.693–54.209**, above the `0.05` gate | **Reject; do not wire** |
| 3 | Remaining arithmetic/store-launch boundary | Exact shared-prefill SiLU+down-rotation improves the actual leaf **29.643 → 21.617 us (1.371x)** and removes **69** c8/L4 trace dispatches | **Retain default-on** with registered strict fallback |
| 4 | Selective unsafe-math code object | Top-1 **100%**, mean KL **4.74e-13**, and NaN/Inf classes match, but the leaf regresses **20.032 → 21.569 us (-7.67%)** | **Reject and remove** |
| 5 | Adaptive MTP admission/depth | Prompt-agnostic adaptive B3 reaches **27.054 tok/s / 37.140 ms-output**, **1.134x** current true AR and **1.116x** best fixed depth; candidate/fixed are each exact **30/30** | **Keep existing adaptive structure**; no new budget/topline promotion |

Net result: one already-default mechanism was revalidated, one exact fused
boundary became default-on, two candidates were rejected and removed, and one
existing T3 policy was revalidated without changing its production cap.

## Opportunity details

### 1. Small-M weight reuse: already solved

hipEngine already had the mechanism the ROCmFPX audit suggested. Its pack8
output-column-tiled single, dual, split-output, and residual-combine kernels
stream one weight tile across 2/4/8 activation rows rather than launching a
row-parallel GEMV. Fresh repaired-runtime c4/c8 complete runs versus
`HIPENGINE_DISABLE_PACK8_OUTPUT_TILED=1` preserve every generated token and
improve complete decode throughput.

Evidence:
[`opp1`](../results/2026-08-16-qwen36-35b-gfx1151-rocmfpx-opp1-small-m-revalidated.json).
No code change was needed.

### 2. One-plane Q8_1: fast projection, failed operation

One-plane Q8_1 plus integer dot product looks strong if gate/up projections are
scored alone: projection top-1 is 100% and mean KL is at most `1.28e-4` in the
screened geometries. That is not an admission gate. After `SiLU(gate) * up`,
three of four cases fail mean KL and every case fails maximum KL; c4 unique is
also only 1.023x. The nonlinear amplification is the reason projection-only
quality must never promote activation quantization.

Evidence:
[`opp2`](../results/2026-08-16-qwen36-35b-gfx1151-rocmfpx-opp2-q8-oneplane-rejected.json).
Strict BF16/FP16 activations remain. Revisit only with residual/two-plane
Q8_1x2 and the same operation-complete oracle.

### 3. Exact SiLU+rotation boundary: retained

Packed-PARO shared-expert prefill previously wrote FP16 SiLU output and launched
registered down-rotation separately. The retained route calls the already
registered pair-rotate primitive directly, preserving the FP16 activation
rounding point. A strict registered two-primitive fallback remains; the
rollback env is `HIPENGINE_SHARED_PREFILL_SILU_ROTATE_FUSED=0`.

The actual rows512/features512/group128/krot8 leaf improves 1.371x. A kernel
trace proves the expected `silu_mul_separate_out` launch disappears once per
layer, reducing the c8/L4 batch+oracle trace from 4,644 to 4,575 dispatches.
Primitive output is bit-exact, profiled hidden state is exact, and c4/c8 token
trajectories match. Bracketed aggregate wall is neutral, so this is a retained
sub-window/launch-count win, not a new PARO topline.

Evidence:
[`opp3`](../results/2026-08-16-qwen36-35b-gfx1151-rocmfpx-opp3-silu-rotate-retained.json).

### 4. Selective unsafe math: operation-level loser

The candidate was a distinct hashed code object using only
`-funsafe-math-optimizations`; blanket `-ffast-math` and
`-ffinite-math-only` were prohibited. Explicit NaN/Inf handling preceded the
finite SiLU path. It passed the numeric gate but was 7.67% slower at the actual
shared-prefill leaf. The transient build route was removed, leaving no unsafe
runtime flag or code object.

Evidence:
[`opp4`](../results/2026-08-16-qwen36-35b-gfx1151-rocmfpx-opp4-unsafe-math-rejected.json).

### 5. Adaptive MTP: structure positive, cap unchanged

The tested policy is not a prompt map and does not branch on token IDs. It
admits cycles at `p_min=0.5`, begins at B1, promotes only after full acceptance,
demotes after partial/zero acceptance, and never exceeds a global B1/B2/B3
cap. Matched fixed-depth controls use the same p-min, 32K draft-vocab cap,
strict top-1 verifier, target block geometry, and direct commit.

Across all ten category+heldout prompts at D24:

| Route | tok/s | Complete ms/output | Accepted/output | Target rows/output |
|---|---:|---:|---:|---:|
| True AR | 23.852 | 41.925 | — | — |
| Adaptive B1 cap | 26.339 | 38.154 | 0.3625 | 1.133 |
| Adaptive B2 cap | 26.795 | 37.496 | 0.4375 | 1.221 |
| **Adaptive B3 cap** | **27.054** | **37.140** | **0.4625** | **1.288** |
| Best fixed (B1) | 24.250 | 41.328 | 0.3625 | 1.242 |
| Fixed B3 | 22.115 | 45.297 | 0.4792 | 1.596 |

Adaptive B3 is +8.15% versus the best fixed train control and +12.26% versus
the best fixed heldout control. It deliberately gives up four accepted tokens
versus fixed B3 while drafting 27 fewer and evaluating 19.32% fewer target
rows/output. Every adaptive and fixed prompt/budget token stream matches the
same-run AR stream (30/30 each).

This does **not** promote B3: its +0.97% over adaptive B2 is one-run noise-scale,
and unrelated dirty docs/tests plus a changed execution-profile denominator
block a new topline. The existing adaptive structure stays; a clean repeated
B2/B3 bracket under one resolved profile is the only valid next gate.

Evidence:
[`opp5`](../results/2026-08-16-qwen36-35b-gfx1151-rocmfpx-opp5-adaptive-mtp-diagnostic.json).

## Comparison and sampler layout

The quant/comparison work now has explicit homes:

- `benchmarks/quant/README.md` — binding exact-artifact quality protocol and
  current compact tables.
- This file — transferable implementation-mechanism decisions.
- `scripts/quant_quality/qwen36_teacher.py` — the reusable fixture/sampler,
  full-logit capture adapters, comparison, and paired bootstrap entry point.
- `scripts/quant_quality/llama_teacher_logits.cpp` — same-runtime llama/ROCmFPX
  full-logit sampler.
- `benchmarks/results/` — compact committed evidence; large logits remain local.

Keep quality and speed separate. A kernel win cannot relabel a quality-traded
quant, and a good projection-only metric cannot override an operation-complete
failure.
