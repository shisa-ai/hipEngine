# DMS quality analysis

Status: **plan**. This document defines the quality bar and test ladder for
the gfx1100 DMS route; it records no results. The end-to-end DMS campaign
record (training, sidecar, gfx1151 finals, production punchlist) lives in
[`DMS.md`](DMS.md). Capacity and speed results live in
[`QWEN38-27B-GFX1100-24GB-CAPACITY.md`](QWEN38-27B-GFX1100-24GB-CAPACITY.md)
and the artifacts it links.

## Background

[Dynamic Memory Sparsification (DMS)](https://arxiv.org/abs/2506.05345)
(Łańcucki et al., NeurIPS 2025) trains a small per-head eviction policy by
logit distillation so a model can discard most of its KV cache while
preserving next-token quality. [FastDMS](https://github.com/shisa-ai/FastDMS)
is the production-speed reference implementation. hipEngine's DMS route
implements its compact layout for the Qwen3.8-27B `Q4_K_M`
[GGUF](GGUF.md) on `gfx1100`, with a sidecar-trained eviction policy bound
to that exact model file. On a single 24 GB RX 7900 XTX this route holds
232,448 prompt tokens; dense INT8 tops out near 129K (capacity evidence:
[`2026-09-08-rx7900xtx-dms-int8-merged-lane-capacity.json`](../benchmarks/results/2026-09-08-rx7900xtx-dms-int8-merged-lane-capacity.json)).

## Why the quality instrument matters

DMS's risk is long-range recall: at the advertised lengths it evicts roughly
48% of history beyond the 8,192-token full-attention window. Short-context
task benchmarks (HumanEval/EvalPlus and similar) never exercise eviction and
cannot qualify this. The paper and the reference implementation qualify
quality with two instruments:

1. Distribution agreement against an eviction-disabled reference
   (token-by-token KL, token-match, comparable-prefix length).
2. Downstream task accuracy at the advertised lengths.

## Gate 1 — the quality bar (paper-matched)

Reference arm: same engine, same quantization, same code, eviction disabled
(retention window effectively unbounded). The comparison isolates the
eviction policy, as in
[FastDMS's quality tables](https://github.com/shisa-ai/FastDMS#compression-quality).

Metrics — the paper's KLD/token-match plus the house KL/top-1 used in the
gfx1151 finals ([`DMS.md`](DMS.md)):

| Metric | Definition | Bar |
| --- | --- | --- |
| Max KL (nats) | worst single-step KL, candidate vs reference logits | ≤ 0.01 target, ≤ 0.05 hard floor |
| Mean KL (nats/token) | KLD averaged over scored steps | ≤ 0.005 |
| Top-1 agreement | argmax equality per scored step | 100% |
| Token match | greedy-decoded tokens identical to reference | ≥ 95% |
| Tokens scored | comparable prefix before first divergence | reported, not gated |
| PPL | perplexity of the candidate on the scored corpus | reported vs reference, not gated |

A run fails closed on nonfinite logits, tracked-allocation leaks, or failure
to return to baseline (the capacity lane's wrapper assertions).

Calibration targets from the published record: FastDMS FP8 compact-DMS at 1K
context — KLD 0.0030 nats/token, 96.9% token match, 64/64 tokens scored; the
paper reports task-accuracy gains under hyper-scaling (Qwen-R1 32B: +12.0
AIME 24, +8.6 GPQA, +9.7 LiveCodeBench). Our gfx1151 finals at 32K/128K
recorded max/mean KL 0.003784/0.000485 and 0.002899/0.000321 with 100%
top-1.

### Test 1 — the 1K paper-matched point

Match the published FastDMS setup so the numbers compare directly to their
table: context 1,024, 16 decode steps, four held-out prompts, INT8 candidate
vs no-evict reference. Regime: mild compression (~1.3x), minimal eviction.
This anchors comparability; it is not the claim.

### Test 2 — the long-context ladder

The Gate 1 protocol at 8K / 32K / 128K / 232K on `gfx1100`:

| Length | Eviction regime | Reference arm runs on | Notes |
| --- | --- | --- | --- |
| 8K | window mostly covers context | XTX 24 GB | baseline point |
| 32K | substantial eviction | XTX 24 GB | gfx1151 32K final's length |
| 128K | heavy eviction (~48% discarded) | XTX 24 GB (tight) | gfx1151 128K final's length |
| 232K | heaviest; no dense route fits | W7900 48 GB (same `gfx1100` arch) | no-evict payload (~2x compact) does not fit in 24 GB |

Prompts come from the DMS data manifest's disjoint validation split
(per-category calibration/validation), so the ladder is held out with
respect to the sidecar's training data. At 232K the reference arm requires
the 48 GB card: same architecture, same kernels. Cross-card logit comparison
is exact only if kernel determinism holds; verify once at a short length
before trusting the 232K point.

## Gate 2 — long-context task suite (after Gate 1)

RULER-style synthetic tasks (needle-in-a-haystack variants, variable tracking,
multi-hop aggregation, common-word extraction) at 32K/128K/232K, DMS vs
no-evict reference. At 232K, where nothing else fits on 24 GB, report the
result as capability (absolute task scores), not as a delta. Gate 2 is what
turns the capacity result into a quality claim, and it gates any default-on
decision (see [`DMS.md`](DMS.md), P6).

## Gate 3 — short-context task smoke (optional)

A codegen suite (HumanEval/EvalPlus class) with the DMS route enabled, to
confirm no regression where compression is mild. Not a DMS quality
instrument; the field's instruments are Gates 1 and 2.

## Prior evidence

- Measured compression at 139,264 tokens: 1.889x (kept fraction 0.5185,
  monotone in context; store ~16.8 KiB/token marginal).
- Eviction decisions and greedy trajectories are invariant across the speed
  kernel merge and the hidden-plane alias adoption (byte-identical logits
  alias on/off; identical store state across the merge).
- Decode at the advertised lengths: ~38 ms/step at 16K, ~53 ms/step at 232K
  (single-session diagnostics; `performance_claim: false`).

## Effort estimate

- Gate 1 harness (KL/token-match/PPL vs the no-evict arm): CPU-testable
  probe extension, about a day including tests.
- Test 1 plus the ladder: ~1 GPU-day across both cards, dominated by the
  232K prefills (~41 min per arm).
- Gate 2 harness: a few days; GPU cost depends on task count.
