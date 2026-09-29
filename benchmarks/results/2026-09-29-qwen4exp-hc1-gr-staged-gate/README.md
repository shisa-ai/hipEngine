# HC-1: the GR staged-f32 tile engages at 2.388x and fails the production gate

**Date:** 2026-09-29
**Host:** Radeon 8060S (gfx1151), hostname `gfx1151`, machine-id
`55ea6c509d0b49eea8de7094a1023668`
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL
**Candidate:** `production_gr_up_staged` - `coltile2_branch4_rowbatch4_staged_f32`,
selected by `HIPENGINE_QWEN4_EXP_GR_STAGED`, T1
**Command:** `scripts/qwen4exp_layer2_profile_gate.py --model-root
/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
--candidate production_gr_up_staged --prompts
benchmarks/prompts/mtpbench-code-general-ja.jsonl --prompts
benchmarks/prompts/gdn-prefill-category-heldouts.jsonl --prompts
benchmarks/prompts/qwen4exp-natural-long-tasks-v1.jsonl --output
benchmarks/results/2026-09-29-qwen4exp-hc1-gr-staged-gate/gate-final.json`
**Artifact:** `gate-final.json` (this directory)
**Tree:** `ead0d67dd`, `tracked_clean: true`, `measurement_valid: true`

## Verdict

**Rejected.** `status: failed`, `numerical_passed: false`, `task_passed: false`,
`requires_outlier_review: true`, `eligible_for_automatic_admission: false`. Next
action per the gate: reject the candidate and keep the strict fallback.

## The route engaged - this is a measurement, not a control

`candidate_dispatch_calls: 1920`, at exactly the four long-form shapes:

| rows | low_rank | width | calls |
|---:|---:|---:|---:|
| 424 | 320 | 10240 | 480 |
| 438 | 320 | 10240 | 480 |
| 508 | 320 | 10240 | 480 |
| 512 | 320 | 10240 | 480 |

`rows` 424-512 clears the selector's `rows > 256` floor and `low_rank=320`
satisfies its `% 64 == 0` requirement. The gate reached these by counting the
runner's **direct `gr_staged_f32_up` call**, which is the only proof available:
the route bypasses variant selection entirely.

## The numbers

550 full-vocabulary rows, three candidate repeats, determinism clean.

| scope | rows | mean KL | p95 | p99 | max | top-1 |
|---|---:|---:|---:|---:|---:|---:|
| **overall** | 550 | 5.194e-4 | 1.519e-3 | 7.495e-3 | **9.066e-2** | 0.99818 |
| code | 175 | 3.448e-5 | 5.067e-5 | 6.908e-4 | 3.081e-3 | 1.0 |
| general_en | 125 | 2.520e-4 | 1.631e-3 | 5.022e-3 | 6.294e-3 | 1.0 |
| general_ja | 125 | 4.861e-4 | 1.750e-3 | 1.468e-2 | 1.933e-2 | 1.0 |
| **mixed_ja_en** | 125 | **1.499e-3** | 3.318e-3 | 4.348e-2 | **9.066e-2** | 0.992 |

**The failure is one category.** Overall mean KL (5.19e-4), p95 (1.52e-3) and
top-1 (0.9982) are all inside limits, and three of four categories are clean -
`code` is three orders of magnitude inside. `mixed_ja_en` alone reads **mean KL
1.499e-3 against the 1e-3 limit**, and it carries both rows over the 0.02 review
boundary and the single top-1 mismatch.

## What the tail looks like, and why it is not a ranking hazard

- `flip_eligible_rows: 19` of 550 (3.45%), `top1_mismatches_flip_eligible: 1`,
  `top1_mismatches_outside_flip_eligible: 0` - the one mismatch is a row the
  strict teacher was itself close on (`strict_margin_min: 0.0248`), not a
  spurious ranking inversion.
- `topk_overlap_mean: 0.99345`, so the ranking is largely preserved.
- `max_abs_logit_delta: 1.886`, `teacher_nll_delta: 9.669e-4` - the candidate is
  very slightly worse on teacher-forced NLL.

So this is **broad-based drift concentrated in one category**, not a
tail-redistribution artifact. That distinction matters for what it rules out: a
tail-only failure would invite a per-shape or per-transition guard, and this one
does not.

## Why this was predicted

The runner's own comment on the selector said it:

> Default-off: the GR numerical gate is a cliff at ~5e-9 prefill-boundary KL and
> this route's screened drift is ~1e-6 relative L2, so it has to cross that
> screen before it can be selected.

The screened drift is ~1e-6 relative L2 on the projection; the gate's own
prefill-boundary baseline for this route is ~5e-9. The mechanism was measured
with the arithmetic intact, but "arithmetic intact" and "inside the calibrated
envelope" are different claims, and this is the second one failing.

**Consistent with the dense sibling.** `DP-8`'s LDS-staged f32 tile is the same
mechanism on the Q8_0 dense suffixes (f32 FMA on f32 operands, summation order
the only difference) and failed its layer-scoped gate the same way -
**broad-based mean KL**, not a tail: 0-47 mean 1.269e-3, 0-15 mean 1.177e-3,
0-7 mean 1.354e-3, all failing top-1 as well. Two independent routes, same
mechanism, same failure shape.

## The screen win is real but unusable

The screen (`../2026-09-29-gr-staged-f32-screen/`) measured **2.388x** on the
largest single kernel in the family (2507.0 ms, 384 launches at 6.529 ms,
987 MB/launch, 90% of it a 256x re-read of one 3.48 MB tensor). That timing
result stands and is not retracted. What this packet establishes is that the
mechanism cannot be selected at production precision, so the win is not
reachable by this route.

## A gate trap found on the way, which changed how this had to be run

The first run of this candidate was a **control, not a measurement**, and looked
green: `candidate_dispatch_calls: 0`, all 18 trajectories bit-identical, every
individual quality gate `true`. Cause: all four GR selectors require
`rows > 256`, while the layer2 gate's **default** suites produce 18 prompts of
**39-71 tokens**, so prefill is a single short call and every GR route declines.

**The gate is fail-closed and refused it** - `measurement_valid` requires
`counter["calls"] > 0` whenever the candidate declares a dispatch target - so the
false pass was never certified. This is the same defect class as the layer-scoped
gate's `wmma` false pass, but caught by construction rather than after the fact.

The fix is to pass the long-form suite as well:
`benchmarks/prompts/qwen4exp-natural-long-tasks-v1.jsonl`, 438/508/424/512
chat-formatted tokens across all four required categories, written for exactly
this row floor. Recorded in `docs/REFACTOR.md`.

**Consequence for the record:** the two pre-existing GR candidates
(`production_gr_up_restore`, `production_gr_up_wide`) have **no gate artifact in
`benchmarks/results/`**, and a default-suite run of either would be a control for
the same reason. Neither route has measured numerical qualification from this
gate.

## Limits

One model, one quant, one host, one profile, one route. **The pass/fail verdict
is specific to this candidate at these shapes**; the long-form suite is four
cases, and the `mixed_ja_en` mean rests on 125 rows from two of them. The gate
records `performance_claim: false` - **no timing was measured here**, and the
2.388x figure is carried over from the screen, not re-measured. `task_passed:
false` means the free-output task gate did not pass either; this packet does not
separate that from the numerical failure, and both point the same way.

## Reproduce

The tree must be clean and committed: the gate raises
`GateError("profile qualification requires a clean committed tracked tree")` on a
dirty tree, and `tracked_clean` is evaluated near the **end** of `run()`, so an
edit made while a run is in flight invalidates that run's artifact rather than
stopping it. Two earlier attempts were discarded for exactly that reason
(`gate.json` - default suites, control; `gate-longform.json` - correct suites,
tree dirtied mid-run).
