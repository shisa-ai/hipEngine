---
status: current
owns: The Gemma 4 router-class numerical calibration: stability rule, bar
  derivation, promotion criteria, and the record of the lead's 2026-10-07
  decision that WMMA-class prefill attention is admissible for this model
  class.
---

# Gemma 4 WMMA calibration protocol (pre-registered 2026-10-07)

This protocol is frozen before the calibration packet runs. Nothing in it may
change after candidate results are read; a change requires a new dated section
naming what moved and why.

## The decision being calibrated

On 2026-10-07 the lead decided WMMA-class prefill attention is admissible for
Gemma 4 26B-A4B on gfx1151: the 2026-10-03 localization showed the KL tail
failures come from rounding-midpoint differences (strict is not the
mathematical oracle) amplified by the top-8-of-128 sparse router, on chains
where the previously accepted full-attention route fails identically. The
frozen 2026-08-16 bars were calibrated on a dense Qwen3.5 GDN route with no
router bifurcation. This protocol re-derives the envelope for router-MoE
models without widening any numeric bar.

## Frozen stability rule (computed from the strict arm only)

A teacher-forced row is **unstable** when the strict arm's own logits satisfy
either:

- top-2 logit gap `< 0.25`, or
- maximum post-softmax probability `< 2^-8`.

Derivation, recorded before the run: `final_logit_softcapping = 30.0` bounds
logit magnitudes, so one bf16 ulp at the cap is `30 * 2^-8 = 0.117`; the gap
threshold `0.25` is slightly over two ulp, the scale at which two valid
roundings of the same float64 value can reorder adjacent logits. The
probability threshold `2^-8` screens near-uniform rows whose KL is unbounded
under any perturbation. Both are computed from strict logits only; candidate
behavior never influences classification.

## Frozen bars (unchanged numerically; the row set is redefined)

| Metric over stable rows only | Requirement |
| --- | ---: |
| Mean KL, candidate versus strict | <= `1e-3` |
| p95 row KL | <= `5e-3` |
| p99 row KL | <= `2e-2` |
| Maximum row KL | <= `5e-2` |
| Top-1 agreement (stable rows), global | >= `99%` |
| Top-1 agreement (stable rows), per category | >= `97%` |

Unstable rows are excluded from every bar above and reported separately: their
count, their KL distribution, and the candidate's top-1 flip rate on them are
recorded as diagnostics. Task non-inferiority (exact JSON retrieval, both arms
on the same case), three-run repeatability, single-slot isolation, and the
unrelated-request replay check remain binding exactly as in
`scripts/gemma4_production_quality.py`. The September-10 row-count standard
binds: promotion evidence requires the full 18-case, 1152-row suite; smaller
screens can only fail, never pass.

## Sanity anchors (recorded, not bars)

- strict versus strict is bitwise zero on every row, stable or not.
- The INT8-per-token/head KV arm must fail the stable-row bars by at least 3x
  in mean; if it does not, the stability rule is not doing its job and the
  calibration is invalid regardless of the candidate.
- The 2026-10-03 artifacts stand as recorded evidence: the parent full-attention
  route (accepted 2026-09-29) fails the old bars on the sensitive chains, and
  the identical-inputs capture shows strict and WMMA rounding the same float64
  value to opposite bf16 neighbors at rounding midpoints.

## Arms

- `strict`: `gemma4_plain` (baseline and teacher).
- `wmma`: `gemma4_wmma_flash` + `gemma4_wmma_flash_full` (candidate).
- `int8_kv`: the INT8-per-token/head KV route (rejected-class anchor).
- BF16-relative arm: unavailable; no BF16 artifact of this model exists on this
  host. Recorded as in the 2026-10-03 packet. The external llama.cpp comparator
  (itself tensor-core MMA flash attention) is cited as the production-quality
  reference for this arithmetic class, not as a bar.

## Promotion criteria

Promote the WMMA prefill candidates to the production default only when:

1. the full 18-case packet passes every stable-row bar and every binding
   control above with the candidate in place;
2. the focused edge-case verification (FP16-range masked-V contamination of
   the sliding kernel) is recorded clean or repaired and re-verified;
3. the router-margin measurement confirms the unstable-row set contains the
   rows that produced the 2026-10-03 tail failures;
4. strict remains the registered fallback and the execution profile remains
   the rollback lever;
5. `docs/EXECUTION-PROFILES.md` gains the decision entry and
   `docs/REFACTOR.md`'s candidate entry is updated to the promoted state.

No numeric bar was widened to admit the candidate. If the candidate fails the
stable-row bars, this protocol does not admit it and the result is recorded as
a rejection.
