# HIP vs Vulkan Current Dashboard

Last reviewed: 2026-07-11. Last retained measurement: 2026-07-10.

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

## Current Decision

- Keep HIP as the production backend. The data does not justify a broad Vulkan
  backend, a broad LLVM/ACO claim, or a hand-ISA program.
- Treat timing mode as part of the workload. Independent rows are not proxies
  for one request's dependent decode chain.
- Use the dispatch result to prioritize fewer launches, graph replay, and fused
  boundaries. Compare host wall only when submission classes match.
- Keep packed-dot as a diagnostic until a matched production slice transfers
  the win. Current Q4, Q6, and dense-Q8 production slices mostly do not.
- Add a new cross-backend slice only when a current production profile exposes
  a hot bucket whose answer would change routing or implementation priority.

## Open Gates

| Priority | Work | Status | Exit gate |
| ---: | --- | --- | --- |
| 0 | Run the exact bounded v2 matrix on W7900/gfx1100 | Blocked on hardware | Same matrices, timing modes, correctness, provenance, and clocks pass on gfx1100; results remain separate from gfx1151. |
| 1 | Profile current PARO and GGUF server paths | Open | A shipped hot slice is identified by layer family and submission behavior before another Vulkan experiment is added. |
| 1 | Match Q6 lm-head math/layout | Blocked on comparable implementation | HIP and Vulkan use identical quantization, activation layout, output coverage, and rowtile algorithm before any ratio is reported. |
| 2 | Production Vulkan or hand ISA | Decision-gated | A clean matched production slice wins in the relevant timing mode and final combined operation, then improves end-to-end wall without a memory/correctness regression. |

## References

- [Dated attribution notebook](HIP-vs-VULKAN-HISTORY.md)
- [Microbenchmark runner guide](../benchmarks/micro/README.md)
- [Canonical benchmark scoreboard](../benchmarks/README.md)
- [RDNA3 roofline](ROOFLINE.md)
- [Optimization punchlist](SOL-OPTIMIZATION.md)
