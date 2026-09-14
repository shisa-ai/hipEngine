# Guarded Q8 Performance Restoration

Target: Flash-Next UD-Q4_K_XL, unchanged GGUF files and BF16 KV,
Framework Desktop `gfx1151` / Radeon 8060S, machine
`55ea6c509d0b49eea8de7094a1023668`. Run dates are September14 UTC and
September14-15 JST. Exact commands, source pins and hashes are in `artifact.json`.

## Correction

The old selected-down WMMA path rounds dequantized Q8 weights to FP16.
The correction multiplies the original BF16 activations by raw integer Q8
codes in BF16 WMMA, then applies each32-value block's scale in FP32.
It does not change the model's quantization or expand its weights.

Unguarded block scaling cuts the real-weight owner MSE by roughly1500x,
but rare BF16 output-boundary differences still produce substantial
model-level drift. The guarded variant therefore queues risky outputs and
recomputes only those outputs in the strict128-lane GEMV reduction order.
The queue has full worst-case capacity and is owned by the request scratch.

The rounding-risk coefficient is empirical, not a universal mathematical
error bound. Model numerical gates remain binding. Non-finite estimates force
repair; an extreme-cancellation RED test caught and fixed that corner case.
Signed Q8 endpoints, empty experts, tails, repeated execution and full-queue
ownership are covered by GPU tests.

## Owner Evidence

Real `blk.2.ffn_down_exps.weight`, K640/N2560,512 experts, synthetic BF16
activations and seeded skewed expert populations. GPU event timings include
counter reset, WMMA and sparse repair, but not routing/combine/model work.

| Prompt rows | Recovery GEMV | Corrected owner | Speedup |
| --- | ---: | ---: | ---: |
| 64 | 6.522 ms | 3.034 ms | 2.15x |
| 512 | 32.129 ms | 9.174 ms | 3.50x |
| 1024 | 61.084 ms | 14.225 ms | 4.29x |

Corrected outputs match the GPU parent on these samples. About3.9% of
outputs are repaired. The first screen repaired34% and lost the speedup;
that version is not selected. These are owner results, not model tok/s.

## Model Gates

Guarded Q8 alone, clean `1cc04b320`:594/594 strict-exact logits and top-1,
three numerical repeats,450 counted dispatches, state/lifecycle pass and
18/18 short free trajectories match strict.

Guarded Q8 plus unchanged dense MMQ passes the short594-row gate
(mean KL0.0000809,max0.01050,594/594 top-1;18/18 free64-token trajectories),
but fails canonical depth: mean0.001465,p950.006841,771/780 top-1.
**Dense MMQ is not restored.** Its passing short result does not qualify
the longer prefill shapes.

Guarded Q8 plus exact four-head QSA-prefill grouping passes all780 canonical
rows exactly, with360 guarded calls. A broader dense/GR/ordered-QSA restore
fails at depth (meanKL0.001410,p950.007991,767/780 top-1), including shorter
contexts where ordered sparse decode is inactive.

Ordered QSA decode was therefore tested separately: all four4K categories,
128 teacher-forced decode transitions, three repeats,516 scored rows.
It matches strict logits/top-1 exactly, with state/repeat gates passing and
240 guarded Q8 calls. The shorter contexts retain the already-qualified
base because sparse decode is not engaged there.

The initial independent-graph-cache ordered-QSA A/B is numerically exact,
but4K decode improves37% for code and regresses5.4/5.8/6.4% for
English/Japanese/mixed categories. A prefill-only A/B also changes decode
rates despite unchanged decode algorithms. These runs do not isolate the
algorithm from graph-instance placement and capture-stream differences.

## Final Timing Protocol

The final three-arm run compares recovery, guarded Q8 with exact prefill
grouping, and that combination with ordered QSA decode. All share the same
captured GDN/MoE decode graph instances; the changed switches affect prefill
or uncaptured QSA attention. An explicit safe-flag check rejects candidates
that can alter captured decode units.

There are108 unique measured samples and36 warmups across all12 canonical
cases. Each arm occupies each round position once; case parity reverses
neighbor direction. Each candidate comparison contains72 samples, sharing
the36 baseline samples rather than measuring or counting them twice.
Runtime source and other test workloads are held quiet during timing.

The initial ordered run is retained in full. The duplicate-cache prefill-only
run was stopped and is retained as incomplete. Neither is silently relabeled
as the final shared-graph protocol.

MMQ, dense/GR iu8, flash prefill, tiled GDN and DP4A remain outside the
qualified restoration.

No MTP or multimodal throughput claim is made. Remaining arithmetic
families require independent correction and qualification.

## Default And Measured Result

UD-Q4_K_XL production selects guarded Q8 grouped down, page256/quad QSA
prefill and ordered/v2 QSA decode. The mapped token-major Q8 scope keeps
its row4-register owner. Q4_K_M is unchanged. The fresh no-override default
smoke matches strict on 9/9 rows at code/4K, with three repeats, state and
zero-allocation teardown. That worktree smoke supplements, not replaces,
the clean source-pinned gates above.

Final same-residency weighted tok/s, BF16 KV, chunk 1024, 128 decode steps:

| Configuration | 512 PP / TG | 1K PP / TG | 4K PP / TG |
| --- | ---: | ---: | ---: |
| Conservative recovery | 170.108 / 17.505 | 177.803 / 16.748 | 128.991 / 10.710 |
| Guarded Q8 + QSA | 177.488 / 17.497 | 186.219 / 16.731 | 178.630 / 10.861 |
| PP gain | +4.34% | +4.73% | +38.48% |

All 108 measured samples have repeat/cross-arm identical output IDs, final
logits and state. Complete-request time improves in all 12 cases, by
1.01-1.37x. Short decode is effectively flat. Weighted 4K decode improves
1.41%, but English/Japanese/mixed decode rates regress versus recovery;
do not describe this as an across-category decode win. Ordered decode is
16.09% faster at 4K than the same optimized prefill without ordered decode.

Shared graphs control instance/capture differences, not CPU frequency.
Earlier CPU handoff evidence is in
`docs/QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md`; the current measurement
does not independently establish the cause of the category differences.
No clock or affinity workaround was introduced.

## Final Review Limits

- The empirical repair screen is not a universal bit-exact guarantee.
  Its finite/nonfinite, cancellation and capacity tests plus model gates
  support only the recorded qualification scope.
- The 780-row base and 516-row ordered gate overlap. They are not 1296
  unique rows, and neither certifies arbitrary context depth.
- Short free trajectories and teacher-forced depth parity do not replace
  symmetric complete-EOS factual review. The earlier candidate's recorded
  task failure stands; one observed error is not proof of lower expected
  quality, and this restoration does not claim long-form certification.
- Dense MMQ fails the longer numerical gate. Dense/GR restore fails as a
  combination; individual members and disabled GDN/MoE paths are not
  thereby proven defective. Their requalification remains separate work.
- Original recovery-cost timing had CPU-test overlap. Only the uncontended
  final shared-graph run supplies this restoration's old-to-new claim.

Validation: 40 focused GPU tests; reviewed CPU tier completed with one stale
registration-list expectation, repaired and followed by 57 passing backend
tests; focused profile tests and the fresh default smoke pass. No broad
suite rerun or new GPU experiment was needed for this final review.
