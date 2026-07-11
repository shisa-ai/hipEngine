# HIP vs Vulkan Current Dashboard

Last reviewed: 2026-07-11. Last retained measurement: 2026-07-11.

This file contains only the current cross-backend conclusions and open gates.
The verbatim attribution notebook, including every pre-v2 hypothesis and local
completion checklist, is preserved in
[`HIP-vs-VULKAN-HISTORY.md`](HIP-vs-VULKAN-HISTORY.md). Do not cite a numeric
row from that history as current evidence unless it is also present here or in
the canonical scoreboard.

## Timing Contract

Current comparisons use timing-contract v2:

- `serial_latency` carries a real dependency between logical iterations;
- `independent_throughput` gives each concurrently eligible iteration disjoint
  writable storage;
- both backends report GPU elapsed time and host wall for the same sequence;
- the exact timed sequence, not only one dispatch, passes correctness;
- comparators reject incomplete matrices, duplicates, mismatched source or
  device provenance, and incompatible submission classes.

The executable contract is in
[`benchmarks/micro/timing_contract.py`](../benchmarks/micro/timing_contract.py)
and [`benchmarks/micro/schemas/result.schema.json`](../benchmarks/micro/schemas/result.schema.json).

## Retained gfx1100 Matrix

The clean bounded run used hipEngine `c57f21b5d5d` on Radeon Pro
W7900/gfx1100, TheRock ROCm `7.15.0a20260711`, and RADV/Mesa `26.1.4`. All 22
comparisons and 232 burst GPU rows pass exact-matrix, timed-command correctness,
clean same-commit provenance, matching device/arch, and GPU-clock gates with
`performance_claim=true`.

Ratios are Vulkan/HIP speedup (`HIP GPU time / Vulkan GPU time`); above `1.0x`
favors Vulkan.

| Family | Serial V/H | Independent V/H | Current read |
| --- | ---: | ---: | --- |
| Dispatch/grid | `2.437x-10.122x` | `1.980x-65.325x` | Vulkan replay has the lower tiny-dispatch floor; independent is throughput, not call latency. |
| Geometry | `0.360x-0.790x` | `1.100x-3.925x` | HIP wins required ordering; Vulkan wins independent overlap. |
| Reduction | `0.304x-0.729x` | `1.110x-4.035x` | Same mode split as geometry. |
| Memory/waitcnt | `0.517x-0.936x` | `0.544x-2.139x` | HIP wins serialized rows; independent rows are mixed. |
| Packed dot | `1.052x-1.133x` | `1.872x-2.106x` | Vulkan leads, but the gfx1151 `3x-4x` magnitude does not transfer. |
| VOPD | `0.391x-0.561x` | `0.516x-0.616x` | Every configured row favors HIP. |
| Sampler top-1/top-k8 | `0.259x-0.501x` | `0.782x-2.563x` | Serialized sampling favors HIP; independent rows are mixed. |
| Two-stage reduction | `0.324x-0.925x` | `0.394x-0.813x` | Every configured row favors HIP in both modes. |

Production-shaped combined rows favor HIP in every serialized case: Q4
selected-dual is `0.501x-0.562x`, Q6 selected-down X8 is `0.675x`, and dense
Q8_0 is `0.393x-0.966x`. Independent combined Q4 is `0.432x-0.477x`, Q6 is
`0.673x`, and dense Q8_0 is `0.388x-1.030x`; only three small `768x2048`
dense rows barely favor Vulkan. A higher-sample Q6 serial follow-up also favored
HIP, while its independent Vulkan run failed timed-sequence correctness and is
not used as a comparison.

Artifact:
[`2026-07-11-hip-vulkan-timing-v2-bounded.json`](../benchmarks/micro/results/gfx1100/w7900/2026-07-11-hip-vulkan-timing-v2-bounded.json).
The local full report is `~/ROCm-report-gfx1100.md`.

The W7900 and gfx1151 systems use different ROCm, kernel, firmware, and Mesa
versions. Cross-architecture differences are descriptive only; they are not a
controlled gfx1100-versus-gfx1151 attribution.

## Retained gfx1151 Matrix

The clean bounded run used hipEngine `ca241dae795d` on Radeon 8060S/gfx1151,
TheRock ROCm `7.13.0a20260411`, and RADV/Mesa `26.1.2`. All 22 comparisons (11
families in both modes) pass exact-matrix, timed-command correctness, and clean
same-commit provenance gates with `performance_claim=true`.

Ratios are Vulkan/HIP speedup (`HIP GPU time / Vulkan GPU time`); above `1.0x`
favors Vulkan.

| Family | Serial V/H | Independent V/H | Current read |
| --- | ---: | ---: | --- |
| Dispatch/grid | `1.162x-16.789x` | `1.116x-150.459x` | Vulkan command replay has a real runtime advantage; this is not compiler evidence. |
| Geometry | `0.677x-0.988x` | `4.133x-7.708x` | HIP wins required ordering; Vulkan wins independent overlap. |
| Reduction | `0.689x-0.981x` | `4.246x-7.502x` | Same mode split as geometry. |
| Memory/waitcnt | `0.869x-1.071x` | `1.077x-1.370x` | Serialized work is mixed near parity; independent work modestly favors Vulkan. |
| Packed dot | `3.052x-3.243x` | `3.840x-4.272x` | Strongest synthetic Vulkan compiler/layout diagnostic. |
| VOPD | `1.031x-1.200x` | `1.040x-1.110x` | Small Vulkan lead even though static evidence shows HIP, not RADV, emitting VOPD. |
| Sampler top-1/top-k8 | `0.507x-1.134x` | `2.461x-5.646x` | Serialized sampling is HIP-favored or mixed; Vulkan's lead is independent throughput. |
| Two-stage reduction | `0.682x-0.934x` | `1.087x-1.466x` | HIP wins serialized chains; Vulkan has a limited independent lead. |

Production-shaped combined rows favor HIP in every serialized case: Q4
selected-dual is `0.922x-0.973x`, Q6 selected-down X8 is `0.549x`, and dense
Q8_0 is `0.540x-0.903x`. Independent combined Q4 is `0.911x-0.978x`, Q6 is
`0.587x`, and dense Q8_0 is `0.558x-1.144x`; only three small `768x2048`
dense rows favor Vulkan. Q6 lm-head remains unratioed because the HIP T16 BF16
and Vulkan X8 q8_1 paths use different math/layouts.

Artifact:
[`2026-07-10-hip-vulkan-timing-v2-bounded.json`](../benchmarks/micro/results/gfx1151/strix-halo/2026-07-10-hip-vulkan-timing-v2-bounded.json).

### Latest matched-stack diagnostic

A 2026-07-11 rerun at clean hipEngine `18255d264425` used the active TheRock
HIP `7.15.0a20260711` runtime/compiler, kernel `7.1.3-2-cachyos`, and
RADV/Mesa `26.1.4` on the same Radeon 8060S/gfx1151. It produced 20/22 valid
comparisons and 224 valid burst GPU rows. Vulkan failed timed-sequence
correctness for independent-throughput Q4 selected-dual twice and Q6
selected-down once, so those rows have no ratio and the run does **not**
supersede the retained 22/22 matrix above.

The valid serial/independent ranges are dispatch
`1.131x-10.937x`/`1.129x-148.100x`, geometry
`0.690x-0.987x`/`4.234x-19.557x`, packed dot
`3.054x-3.203x`/`3.599x-3.795x`, and two-stage reduction
`0.694x-0.947x`/`0.660x-0.969x`. Valid combined production rows still favor
HIP: serial Q4 is `0.926x-0.989x`, serial Q6 is `0.589x`, and serial/independent
dense Q8_0 is `0.568x-0.901x`/`0.391x-0.906x`.

Observed end-to-end wall was 5m49s including failure handling and a 5.18s Q4
confirmation; active benchmark work was approximately 5m10s, so reserve about
six minutes per bounded update. The environment still contains the older
`rocm-sdk-libraries-gfx1151 7.13` BLAS package, but the measured microbenchmark
binaries resolve HIP/HSA/LLVM from TheRock 7.15 and do not use those BLAS
libraries. Full old-versus-updated tables are in `~/gfx1151-scratch.md`.
Compact artifact:
[`2026-07-11-gfx1151-hip-vulkan-matched-stack-diagnostic.json`](../benchmarks/results/2026-07-11-gfx1151-hip-vulkan-matched-stack-diagnostic.json).

## gfx1100 versus gfx1151

The two retained matrices use the same benchmark shapes, timing modes,
dependency contracts, correctness gates, and ratio definition. They do **not**
use the same software stack: W7900 ran TheRock ROCm 7.15 plus Mesa 26.1.4,
while Strix Halo ran ROCm 7.13 plus Mesa 26.1.2. The table therefore answers
"does the observed pattern transfer?", not "what does architecture alone
cause?"

| Family | gfx1100 serial / independent | gfx1151 serial / independent | Transfer read |
| --- | ---: | ---: | --- |
| Dispatch/grid | `2.437x-10.122x` / `1.980x-65.325x` | `1.162x-16.789x` / `1.116x-150.459x` | Vulkan leads on both; the absolute floor and range differ. |
| Geometry | `0.360x-0.790x` / `1.100x-3.925x` | `0.677x-0.988x` / `4.133x-7.708x` | Same HIP-serial/Vulkan-independent split, much narrower independent gap on gfx1100. |
| Reduction | `0.304x-0.729x` / `1.110x-4.035x` | `0.689x-0.981x` / `4.246x-7.502x` | Same mode split, stronger serialized HIP and narrower independent Vulkan lead on gfx1100. |
| Memory/waitcnt | `0.517x-0.936x` / `0.544x-2.139x` | `0.869x-1.071x` / `1.077x-1.370x` | Does not broadly transfer: gfx1100 serial favors HIP and independent crosses parity. |
| Packed dot | `1.052x-1.133x` / `1.872x-2.106x` | `3.052x-3.243x` / `3.840x-4.272x` | Vulkan leads on both, but the gfx1151 `3x-4x` magnitude collapses on gfx1100. |
| VOPD | `0.391x-0.561x` / `0.516x-0.616x` | `1.031x-1.200x` / `1.040x-1.110x` | Direction flips: HIP wins every gfx1100 row; Vulkan modestly wins gfx1151. |
| Sampler | `0.259x-0.501x` / `0.782x-2.563x` | `0.507x-1.134x` / `2.461x-5.646x` | Serialized HIP advantage strengthens; gfx1100 independent rows become mixed. |
| Two-stage reduction | `0.324x-0.925x` / `0.394x-0.813x` | `0.682x-0.934x` / `1.087x-1.466x` | Serialized HIP transfers; independent direction flips to HIP on gfx1100. |

Production-shaped combined operations are more consistent than the synthetic
families:

| Combined operation | gfx1100 serial / independent | gfx1151 serial / independent | Transfer read |
| --- | ---: | ---: | --- |
| Q4 selected-dual | `0.501x-0.562x` / `0.432x-0.477x` | `0.922x-0.973x` / `0.911x-0.978x` | HIP wins both architectures; the W7900 margin is much larger. |
| Q6 selected-down X8 | `0.675x` / `0.673x` | `0.549x` / `0.587x` | HIP wins both bounded matrices. |
| Dense Q8_0 | `0.393x-0.966x` / `0.388x-1.030x` | `0.540x-0.903x` / `0.558x-1.144x` | Mostly HIP on both; only small independent rows approach or cross parity. |

The architecture-specific signal is therefore real enough to block ratio
transfer: only dispatch and the geometry/reduction mode split reproduce
qualitatively across the full synthetic families. Packed-dot magnitude, VOPD,
sampler independent throughput, and two-stage independent throughput differ
materially. Before assigning those differences to gfx1100 versus gfx1151,
rerun both devices with matched ROCm/compiler, Mesa/RADV, kernel/firmware, fixed
clock policy, and interleaved backend order.

## Current Decision

- Keep HIP as the production backend. The data does not justify a broad Vulkan
  backend, a broad LLVM/ACO claim, or a hand-ISA program.
- Treat timing mode as part of the workload. Independent rows are not proxies
  for one request's dependent decode chain.
- Use the dispatch result to prioritize fewer launches, graph replay, and fused
  boundaries. Compare host wall only when submission classes match.
- Keep packed-dot as a diagnostic until a matched production slice transfers
  the win. Its gfx1151 magnitude collapses on gfx1100, and current Q4, Q6, and
  dense-Q8 combined production slices mostly favor HIP.
- Add a new cross-backend slice only when a current production profile exposes
  a hot bucket whose answer would change routing or implementation priority.

## Open Gates

| Priority | Work | Status | Exit gate |
| ---: | --- | --- | --- |
| 0 | Fix-clock W7900 dispatch/stream attribution | Open | Interleaved one/four-stream and graph controls plus queue/AQL traces separate runtime submission from clock residency. |
| 0 | Resolve Q6 50-repetition independent correctness | Open | The Vulkan timed-sequence oracle passes the higher-repetition disjoint-output run before its timing is compared. |
| 1 | Profile current PARO and GGUF server paths | Open | A shipped hot slice is identified by layer family and submission behavior before another Vulkan experiment is added. |
| 1 | Match Q6 lm-head math/layout | Blocked on comparable implementation | HIP and Vulkan use identical quantization, activation layout, output coverage, and rowtile algorithm before any ratio is reported. |
| 2 | Production Vulkan or hand ISA | Decision-gated | A clean matched production slice wins in the relevant timing mode and final combined operation, then improves end-to-end wall without a memory/correctness regression. |

## References

- [Dated attribution notebook](HIP-vs-VULKAN-HISTORY.md)
- [Microbenchmark runner guide](../benchmarks/micro/README.md)
- [Canonical benchmark scoreboard](../benchmarks/README.md)
- [RDNA3 roofline](ROOFLINE.md)
- [Optimization punchlist](SOL-OPTIMIZATION.md)
