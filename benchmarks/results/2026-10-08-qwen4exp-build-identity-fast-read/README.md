# Build-identity fast read: measured and reverted

Iteration-4 candidate for the decode host path: `_resolution_environment_identity`
rewritten to read the eight build-identity variables from the live raw
environment mapping instead of the str view. Isolated: **173 ns vs 2,292 ns per
call** (13x), projecting ~1.95 ms/step at the ~980 resolutions/step decode
rate, with live-change observation preserved (monkeypatched tests and the
`hip_target_arch_environment` scope still observed; unit-tested).

## Same-residency decode A/B (one process, balanced arms, 6 reps/arm)

| Case | slow (str view) | fast (raw mapping) | delta |
| --- | ---: | ---: | ---: |
| code-p512 | 6091.2 ms / 128 steps | 6083.7 ms | +7.5 ms (+0.12%) |
| code-p4096 | 7068.6 ms | 7261.4 ms | **-192.9 ms (-2.7%)** |

Cross-arm outputs exact in both cases; within-arm deterministic; zero
`plan_hip_build` calls in either arm (no memo misses). The p4096 inversion has
no identified mechanism - identical libraries, launch families and outputs -
and the host was heat-soaked at the time (p4096 decode walls moved 55 -> 61 ->
97 ms/step across the evening on unchanged code).

## Verdict

**Reverted.** The loop's acceptance policy requires a measured improvement;
the isolated 13x is not an engine-rate claim, and the only same-residency
depth sample contradicts it without explanation. Re-measure with a cooled
paired protocol (>=30 min idle, single-case fresh-process A/B) or fold into
the full library-threading change from the REFACTOR.md removal condition.

[decode-ab.json](decode-ab.json) carries the full per-rep samples; the
worklog entry records the isolated benchmarks and probe counts
(20261008T041526.949462Z).
