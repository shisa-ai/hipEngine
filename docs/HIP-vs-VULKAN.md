# HIP vs Vulkan Current Dashboard

Last reviewed: 2026-07-12. Last retained measurement: 2026-07-11.

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

The retained W7900 and gfx1151 runs use the same TheRock ROCm/HIP build, AMD
clang/LLVM build, CachyOS kernel, firmware packages, Mesa/RADV release, Vulkan
loader, sampling counts, and executable benchmark sources. The measured
variables are the host/GPU platforms and their automatic clock behavior; this
is the intended cross-device comparison rather than a software-version
mismatch.

## Retained gfx1151 Matrix

The clean matched-protocol refresh used hipEngine `0e566a4559b5` on Radeon
8060S/gfx1151, TheRock ROCm `7.15.0a20260711`, kernel
`7.1.3-2-cachyos`, and RADV/Mesa `26.1.4`. The corrected environment contains
only gfx1151 target-specific device wheels at the 20260711 nightly. All 22
comparisons and 232 burst GPU rows pass exact-matrix, timed-command correctness,
clean same-commit provenance, and environment gates with
`performance_claim=true`.

This run deliberately matches the prior retained sampling: paired families use
10 repetitions, 3 warmups, and 5 samples; dispatch uses 20 repetitions and 5
warmups. It is therefore directly comparable to both the historical gfx1151
snapshot and the retained gfx1100 matrix. The historical gfx1151 version delta
does not isolate a single software component because several components moved
together; that does not affect the current matched cross-device matrix.

Ratios are Vulkan/HIP speedup (`HIP GPU time / Vulkan GPU time`); above `1.0x`
favors Vulkan.

| Family | Serial V/H | Independent V/H | Current read |
| --- | ---: | ---: | --- |
| Dispatch/grid | `1.128x-10.751x` | `1.115x-142.384x` | Vulkan command replay has a real runtime advantage; this is not compiler evidence. |
| Geometry | `0.707x-0.992x` | `2.619x-20.832x` | HIP wins required ordering; Vulkan wins independent overlap. |
| Reduction | `0.659x-0.984x` | `2.525x-21.024x` | Same mode split as geometry. |
| Memory/waitcnt | `0.891x-1.109x` | `1.006x-1.170x` | Serialized work is mixed near parity; independent work modestly favors Vulkan. |
| Packed dot | `3.054x-3.204x` | `3.833x-4.197x` | Strongest stable synthetic Vulkan compiler/layout diagnostic. |
| VOPD | `1.061x-1.181x` | `1.010x-1.103x` | Small Vulkan lead even though static evidence shows HIP, not RADV, emitting VOPD. |
| Sampler top-1/top-k8 | `0.517x-1.142x` | `1.526x-10.015x` | Serialized sampling is HIP-favored or mixed; Vulkan's lead is independent throughput. |
| Two-stage reduction | `0.681x-0.958x` | `0.825x-1.826x` | HIP wins serialized chains; independent rows now cross parity. |

Production-shaped combined rows favor HIP in every serialized case: Q4
selected-dual is `0.916x-0.980x`, Q6 selected-down X8 is `0.553x`, and dense
Q8_0 is `0.552x-0.879x`. Independent combined Q4 is `0.854x-0.973x`, Q6 is
`0.480x`, and dense Q8_0 is `0.448x-1.152x`; only the smallest dense shapes
favor Vulkan. Q6 lm-head remains unratioed because the HIP T16 BF16 and Vulkan
X8 q8_1 paths use different math/layouts.

Artifact:
[`2026-07-11-gfx1151-hip-vulkan-matched-protocol.json`](../benchmarks/results/2026-07-11-gfx1151-hip-vulkan-matched-protocol.json).
The exact matrix took 249.323 seconds (`4m09.323s`). The prior ROCm 7.13,
kernel 7.0.12, Mesa 26.1.2 snapshot remains preserved in
[`2026-07-10-hip-vulkan-timing-v2-bounded.json`](../benchmarks/micro/results/gfx1151/strix-halo/2026-07-10-hip-vulkan-timing-v2-bounded.json)
and the side-by-side notebook is `~/gfx1151-scratch.md`.

### Stricter 20-repetition diagnostic

A separate run used the newer README template: 20 repetitions/5 warmups/7
samples for paired families and 50 repetitions/10 warmups for dispatch. It
produced 20/22 valid comparisons and 224 valid burst GPU rows. Vulkan
independent Q4 failed KL (`0.079520 > 0.05`) and Q6 failed top-1 agreement
(`0.875 < 0.9`). Repeats and one/two/four-queue controls are bit-identical.

The same strict probes fail identically under Mesa/RADV 26.1.2, while both
Mesa versions pass at the retained 10-repetition coverage. Cumulative replay
pinpoints the fixture boundaries (zero-based): Q4 first fails when slice `10`
(the 11th repetition) is included, where aggregate KL moves from `0.004753` to
`0.079520`; Q6 KL changes from `0.006046` to `0.007756` at slice `14` but still
passes, then the gate first fails at slice `17` (the 18th repetition), where
top-1 moves from `1.0` to `0.875`. Repetition does not degrade a fixed input;
the longer protocol simply includes these deterministic fixture slices.
Benchmark sources are unchanged from the old retained revision. This is not a
Mesa `.2` to `.4` regression, gfx1151 synchronization failure, or TheRock
package mismatch. The current gfx1100 bounded 10-repetition rows also pass; its
separate Q6 50-repetition diagnostic exposes the same coverage-sensitive class
of numerical miss.

The strict matrix took approximately 5m10s of benchmark work and 5m49s
end-to-end with failure handling and confirmation. Compact artifact:
[`2026-07-11-gfx1151-hip-vulkan-matched-stack-diagnostic.json`](../benchmarks/results/2026-07-11-gfx1151-hip-vulkan-matched-stack-diagnostic.json).

## gfx1100 versus gfx1151

The two retained matrices use the same executable benchmark sources, shapes,
sampling counts, timing modes, dependency contracts, correctness gates, ratio
definition, TheRock/AMD-clang build, kernel, firmware packages, Mesa/RADV, and
Vulkan loader. The host/GPU platforms and their automatic clock behavior are
the material differences.

| Family | gfx1100 serial / independent | gfx1151 serial / independent | Transfer read |
| --- | ---: | ---: | --- |
| Dispatch/grid | `2.437x-10.122x` / `1.980x-65.325x` | `1.128x-10.751x` / `1.115x-142.384x` | Vulkan leads on both; the absolute floor and range differ. |
| Geometry | `0.360x-0.790x` / `1.100x-3.925x` | `0.707x-0.992x` / `2.619x-20.832x` | Same HIP-serial/Vulkan-independent split, much wider independent range on gfx1151. |
| Reduction | `0.304x-0.729x` / `1.110x-4.035x` | `0.659x-0.984x` / `2.525x-21.024x` | Same mode split, stronger serialized HIP and narrower independent Vulkan lead on gfx1100. |
| Memory/waitcnt | `0.517x-0.936x` / `0.544x-2.139x` | `0.891x-1.109x` / `1.006x-1.170x` | Does not broadly transfer: gfx1100 serial favors HIP and independent crosses parity. |
| Packed dot | `1.052x-1.133x` / `1.872x-2.106x` | `3.054x-3.204x` / `3.833x-4.197x` | Vulkan leads on both, but the gfx1151 `3x-4x` magnitude collapses on gfx1100. |
| VOPD | `0.391x-0.561x` / `0.516x-0.616x` | `1.061x-1.181x` / `1.010x-1.103x` | Direction flips: HIP wins every gfx1100 row; Vulkan modestly wins gfx1151. |
| Sampler | `0.259x-0.501x` / `0.782x-2.563x` | `0.517x-1.142x` / `1.526x-10.015x` | Serialized HIP advantage strengthens; gfx1100 independent rows become mixed. |
| Two-stage reduction | `0.324x-0.925x` / `0.394x-0.813x` | `0.681x-0.958x` / `0.825x-1.826x` | Serialized HIP transfers; gfx1151 independent rows cross parity. |

Production-shaped combined operations are more consistent than the synthetic
families:

| Combined operation | gfx1100 serial / independent | gfx1151 serial / independent | Transfer read |
| --- | ---: | ---: | --- |
| Q4 selected-dual | `0.501x-0.562x` / `0.432x-0.477x` | `0.916x-0.980x` / `0.854x-0.973x` | HIP wins both architectures; the W7900 margin is much larger. |
| Q6 selected-down X8 | `0.675x` / `0.673x` | `0.553x` / `0.480x` | HIP wins both bounded matrices. |
| Dense Q8_0 | `0.393x-0.966x` / `0.388x-1.030x` | `0.552x-0.879x` / `0.448x-1.152x` | Mostly HIP on both; only small independent rows approach or cross parity. |

The architecture-specific signal is therefore real enough to block ratio
transfer: only dispatch and the geometry/reduction mode split reproduce
qualitatively across the full synthetic families. Packed-dot magnitude, VOPD,
sampler independent throughput, and two-stage independent throughput differ
materially. Software versions and executable sources are already matched.
Separating GPU architecture from automatic clock residency and runtime
scheduling now requires fixed or continuously recorded clocks, interleaved
backend order, and queue/kernel counters—not another version-matching pass.

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
| 1 | Profile current PARO and GGUF server paths | Open | A shipped hot slice is identified by layer family and submission behavior before another Vulkan experiment is added. |
| 1 | Match Q6 lm-head math/layout | Blocked on comparable implementation | HIP and Vulkan use identical quantization, activation layout, output coverage, and rowtile algorithm before any ratio is reported. |
| 2 | Production Vulkan or hand ISA | Decision-gated | A clean matched production slice wins in the relevant timing mode and final combined operation, then improves end-to-end wall without a memory/correctness regression. |

## References

- [Dated attribution notebook](HIP-vs-VULKAN-HISTORY.md)
- [Microbenchmark runner guide](../benchmarks/micro/README.md)
- [Canonical benchmark scoreboard](../benchmarks/README.md)
- [RDNA3 roofline](ROOFLINE.md)
- [Optimization punchlist](SOL-OPTIMIZATION.md)
