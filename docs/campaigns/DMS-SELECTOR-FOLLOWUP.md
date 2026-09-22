---
status: current
owns: Follow-up evidence and reopening conditions for the closed DMS selector campaign.
---
# DMS selector follow-up: G0 repair and multilingual sampling

State: **blocked before balanced controls**. This follow-up preserves the original
campaign closure in [DMS-SELECTOR-IMPROVEMENT.md](DMS-SELECTOR-IMPROVEMENT.md)
and does not revise its negative or blocked conclusions.

## Current evidence

The same-host compact-no-evict control failed deterministically before any
multilingual selector comparison:

- Max KL: `0.004982890284225346` versus the G0 limit `0.001`.
- Top-1: `11/12` rather than 100%.
- Reproduced mismatch: mixed Japanese/English, decode step 1.
- Teardown: tracked allocations returned to baseline and the dense prefill pool
  was released.
- Pilot artifacts:
  - `~/dms-artifacts/selector-improvement-run-20260922-062438/pilots/g0-no-evict-1k-d2.json`
  - `~/dms-artifacts/selector-improvement-run-20260922-062438/pilots/g0-no-evict-1k-d2-mixed-repeat.json`

The follow-up diagnosis narrowed this to compact-versus-dense decode numerical
parity. Current-token inclusion and live-count metadata are consistent in the
pilot. Compact and dense use different split-attention producer/reduction paths
at this context. A possible compact reducer shared-memory synchronization hazard
was identified but not proven; no speculative repair was applied.

## Controls and sampling ablation

Balanced code/English/Japanese/mixed controls were **not run** because G0 did not
pass. The planned source-disjoint sampling ablation was therefore also **not
run**:

1. balanced baseline;
2. Japanese oversampling with matched optimizer updates;
3. additional source-disjoint Japanese data;
4. additional source-disjoint mixed Japanese/English data.

No multilingual sampling result, selector improvement, or Japanese-specific
production recommendation is supported by this follow-up. The only related
positive signal is the noncausal continuation-mass CPU diagnostic, which had
mean discarded mass `0.1002371595` versus recency `0.6766432918` and random
seed 0 `0.5141876483`. That result is hindsight-only label evidence, not runtime
quality or causal query evidence.

## Reopen sequence

1. Compare identical Q/K/V inputs through compact and generic dense attention at
   1,025/1,026 live rows, including intermediate attention outputs.
2. Check the compact reducer synchronization hypothesis and any producer arithmetic
   difference with a focused parity fixture.
3. Rerun the same four-category 1K/two-step G0 pilot. Require max KL `<= 0.001`
   and 100% top-1 before proceeding.
4. Run balanced multilingual controls with equal category counts and report
   prefill/decode metrics separately.
5. Only then run the four-arm source-disjoint multilingual sampling ablation with
   fixed optimizer updates and unchanged production defaults.

This follow-up makes no product-default change and does not reopen the original
campaign automatically.
