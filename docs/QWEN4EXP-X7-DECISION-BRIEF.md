# The X-7 Decision Brief: the `prefill_last` scope and the compensated-T16 candidates

Status: awaiting the human owner's decision (ledger row X-7). This brief is
self-contained: it states the question, the evidence, both branches' exact
consequences, and what does not change either way. Every number carries its
artifact pointer.

## The question

`docs/EXECUTION-PROFILES.md` gates arithmetic-changing candidates per scope.
One scope — **`prefill_last`**, 18 rows measuring the logits at the final
prefill-chunk boundary — carries a **mean-KL floor of 1e-3** that two
completed candidates fail while passing everything else. The question: **does
the `prefill_last` scope's mean-KL floor (or the scope itself) admit the
compensated-T16 class, or not?**

This is a policy decision about the numerical envelope, not an experiment:
more engineering cannot change these numbers, only the scope definition or
the threshold can.

## The two candidates, complete

Both are bit-lossless repack routes: the GGUF weights are rearranged into
tile layouts (identical bytes), decoded to F16 in registers, and multiplied
with **compensated F32 accumulation** — no repair pass by construction. Both
are built, gated, committed, and **default-off** behind
`HIPENGINE_QWEN4_EXP_T16_GATE_UP` / `HIPENGINE_QWEN4_EXP_T16_DOWN`, each
with a registered strict fallback.

### Routed Q4 gate/up (ledger row T16-GU)

- **Same-residency engine A/B** (12 canonical cases, 3 reps, balanced arms,
  30-of-46-layer window): **+6.53% aggregate prefill, all 12 cases
  positive**; client wall 0.9715x.
- **Owner A/B**: the fastest registered gate/up owner, -32.90% median vs
  the production iu8 owner (8/8 cases, 126x the repeat noise).
- **Layer2 gate** (18 prompts, 24-layer window, 2,160 engaged dispatch
  calls, state gate passed): every global aggregate passes (mean KL 6.254e-4
  vs 1e-3, p95 3.220e-3, p99 7.637e-3, max 1.657e-2, top-1 0.99556); every
  category passes. **Rejected only at `prefill_last`: mean KL 1.686e-3 =
  1.69x the floor** — with p99, max, and **top-1 18/18 all passing on that
  same scope**.
- Evidence: `benchmarks/results/2026-10-08-qwen4exp-t16-owner-ab/`.

### Routed Q5_1 down (ledger row T16-DOWN)

- **Same-residency engine A/B** (12 cases, all 43 Q5_1 layers resident):
  **+9.46% aggregate prefill (343.35 -> 375.82 tok/s), all 12 positive**
  (per-case ratios 1.0896-1.1113); client wall -3.68%; decode unchanged.
- **Owner A/B** (real geometry, 15 reps, 8 cases): -49.4% to -69.5% vs the
  production pair2 down, median -57.7%.
- **Layer2 gate** (18 prompts, all 43 layers): every global aggregate passes
  (mean KL 6.034e-4, p95 3.286e-3, p99 6.811e-3, max 1.962e-2, top-1
  0.99111); every category passes (code 99.333%, general_en 99.000%,
  general_ja 98.000%, mixed 100%). **Rejected only at `prefill_last`: mean
  KL 1.529e-3 = 1.53x the floor** — with p99 and max passing and **top-1
  18/18 (100%) on that same scope**, all 4 global top-1 mismatches on
  flip-eligible rows.
- **The comparison that frames the decision**: the F16 compact down sibling
  — same F16-operand class, no compensated accumulate — was **rejected at
  `prefill_last` top-1 98.667%**. The compensated T16 holds **100% top-1**
  there. The compensation cleared the exact axis the F16 sibling failed;
  only the mean-KL floor remains.
- Evidence: `benchmarks/results/2026-10-09-qwen4exp-t16-down-engine-ab/`,
  `benchmarks/results/2026-10-09-qwen4exp-t16-down-gate/`.

## What the numbers mean

The `prefill_last` scope measures distribution shape at the chunk boundary
where arithmetic changes concentrate. Both candidates sit at 1.53-1.69x the
mean-KL floor **with zero top-1 disagreement** on those same rows — the drift
is sub-argmax ordering noise, not different outputs. For calibration on the
same scope: plain-iu8 sat at 2.16x, F16-compact at 1.13x (and failed top-1);
the compensated T16s sit between them, carrying none of the repair and the
campaign's largest measured in-envelope gains.

## Branch A — admit

1. The full-window engine A/B and layer2 gate re-run (the gate/up evidence
   is a 30-of-46-layer window; the full window's figure gets measured, not
   extrapolated — the residency mechanism below makes it fit).
2. The net-zero residency phase turns on: tiles materialize at load, the raw
   bytes they replace free (gate/up 42.38 GiB, Q5_1 down ~26.3 GiB — the
   tile sets equal the raw sets exactly, so this is net-zero memory; the
   mechanism is implemented and default-off:
   `HIPENGINE_QWEN4_EXP_T16_RESIDENT`).
3. The routes promote to the production defaults; the env flags delete per
   their recorded removal conditions (`docs/REFACTOR.md`).
4. Expected: **~+16% combined prefill** on the canonical protocol
   (~340 -> ~394 tok/s settled-state), the campaign's largest single
   lever, with the strict profile remaining the exact oracle.

## Branch B — reject

1. The env flags, runner branches, and sidecar caches delete per their
   recorded removal conditions (`docs/REFACTOR.md`); the kernels, layouts,
   and their round-trip/quality-gate tests stay as reference lineage.
2. The residency populate and the `t16` stage's generator call site delete;
   the loading-layer primitives stay only if another consumer names them.
3. The `prefill_last` scope and threshold stand unchanged; the X-7 row
   records the rejection as final for this class.
4. The ~+16% closes as measured-but-not-admitted; the remaining prefill
   levers are the multi-session builds (the whole-step decode graph, the
   deep QSA restructure).

## What does not change either way

- The three promoted production defaults (dense Q8 DPP, K320 GR-up,
  tiled16_multi GDN), the validated MoE decode graph, and the production
  default path's measured behavior (~339.8 tok/s aggregate settled-state,
  12/12 deterministic).
- The strict profile and the exact chains remain the correctness oracle for
  every future candidate.
- The public scoreboard carries no candidate figures; the STATUS doc names
  both candidates as candidates with their flags.
- The thermal caveat: this host measures ~7% lower aggregate under ~4h of
  sustained GPU load (recorded in `docs/QWEN4EXP-STATUS.md`); settled-state
  figures remain the comparison basis.
