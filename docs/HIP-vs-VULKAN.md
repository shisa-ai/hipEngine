# HIP vs Vulkan Current Dashboard

Last reviewed: 2026-07-12. Last retained measurement: 2026-07-12.

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
loader, and sampling counts. The eight synthetic headline families retain
identical executable sources. The refreshed gfx1151 Q4/Q6/dense rows alone use
the portable q8 shader from `50bea8f3`; that shader has not yet run on the
unavailable W7900 host. This is a deliberate one-kernel source delta, not a
software-stack version mismatch.

## Retained gfx1151 Matrix

The portable-q8 refresh used clean hipEngine `50bea8f330fe` on Radeon
8060S/gfx1151, TheRock ROCm `7.15.0a20260711`, kernel
`7.1.3-2-cachyos`, and RADV/Mesa `26.1.4`. The corrected environment contains
only gfx1151 target-specific device wheels at the 20260711 nightly. Both the
gfx1100-matched protocol and the current stricter protocol pass all 22
comparisons and 232 burst GPU rows with clean same-commit provenance and
`performance_claim=true`.

The table below uses the run that exactly matches the retained gfx1100
sampling: paired families use 10 repetitions, 3 warmups, and 5 samples;
dispatch uses 20 repetitions and 5 warmups. The separate strict run uses the
current README template: paired `20/5/7` and dispatch `50/10`.

Ratios are Vulkan/HIP speedup (`HIP GPU time / Vulkan GPU time`); above `1.0x`
favors Vulkan.

| Family | Serial V/H | Independent V/H | Current read |
| --- | ---: | ---: | --- |
| Dispatch/grid | `1.128x-11.125x` | `1.100x-149.207x` | Vulkan command replay has a real runtime advantage; this is not compiler evidence. |
| Geometry | `0.708x-0.990x` | `2.616x-21.328x` | HIP wins required ordering; Vulkan wins independent overlap. |
| Reduction | `0.662x-0.987x` | `2.545x-20.794x` | Same mode split as geometry. |
| Memory/waitcnt | `0.877x-1.087x` | `0.970x-1.149x` | Both modes are mixed near parity. |
| Packed dot | `3.049x-3.210x` | `3.764x-4.151x` | Strongest stable synthetic Vulkan compiler/layout diagnostic. |
| VOPD | `1.085x-1.181x` | `1.001x-1.108x` | Small Vulkan lead even though static evidence shows HIP, not RADV, emitting VOPD. |
| Sampler top-1/top-k8 | `0.508x-1.167x` | `1.522x-3.973x` | Serialized sampling is HIP-favored or mixed; Vulkan's lead is independent throughput. |
| Two-stage reduction | `0.685x-0.941x` | `0.679x-4.095x` | HIP wins serialized chains; independent rows remain variable and cross parity. |

Production-shaped combined Q4 is `0.932x-1.007x` serialized and
`0.874x-0.921x` independent: one serialized row crosses parity by only 0.7%,
while the strict run's maximum is `0.9997x`. Q6 selected-down X8 is `0.550x`
serialized and `0.477x` independent. Dense Q8_0 is `0.545x-0.902x` serialized
and `0.457x-1.194x` independent; only small independent shapes favor Vulkan.
Q6 lm-head remains unratioed because the HIP T16 BF16 and Vulkan X8 q8_1 paths
use different math/layouts.

Artifact:
[`2026-07-12-gfx1151-hip-vulkan-portable-q8.json`](../benchmarks/results/2026-07-12-gfx1151-hip-vulkan-portable-q8.json).
The matched matrix took 246.705 seconds (`4m06.705s`); the strict matrix took
298.786 seconds (`4m58.786s`). The prior matched stock-q8 result remains in
[`2026-07-11-gfx1151-hip-vulkan-matched-protocol.json`](../benchmarks/results/2026-07-11-gfx1151-hip-vulkan-matched-protocol.json),
and the ROCm 7.13, kernel 7.0.12, Mesa 26.1.2 snapshot remains in
[`2026-07-10-hip-vulkan-timing-v2-bounded.json`](../benchmarks/micro/results/gfx1151/strix-halo/2026-07-10-hip-vulkan-timing-v2-bounded.json)
and the side-by-side notebook is `~/gfx1151-scratch.md`.

### Portable q8 rounding and strict coverage

The stock shader's deterministic strict misses were isolated to its shared
q8_1 activation quantizer. Every mismatched stored `d` scale was one FP16 code
below the CPU/HIP oracle (`1,374/2,816` Q4 blocks and `1,686/2,304` Q6 blocks),
while CPU-prequantized Vulkan dot runs passed. This ruled out the Q4/Q6 dot
kernels, synchronization, TheRock packaging, and the Mesa 26.1.2-to-26.1.4
update. The isolation artifact preserves the original failure boundaries and
field-level readback:
[`2026-07-12-gfx1151-vulkan-q8-isolation-diagnostic.json`](../benchmarks/results/2026-07-12-gfx1151-vulkan-q8-isolation-diagnostic.json).

Commit `50bea8f3` makes the shader's rounding contract explicit: software
FP32-to-FP16 round-to-nearest-even stores `d`/`s`, and integer quantization uses
explicit ties-away-from-zero rounding. The clean strict `20/5/7` run now passes
22/22 comparisons and all 232 burst rows. Independent Q4 reaches KL
`0.004180817473`, top-1 `1.0`; independent Q6 reaches KL
`0.0002289689978`, top-1 `1.0`. Serial Q4 has the same KL and top-1; serial Q6
reaches KL `1.574600576e-05`, top-1 `1.0`. Dense Q8_0 passes all 52 rows in
each timing mode. Across all 20 strict slices, stored `d` mismatches fall to
zero. Sparse one-integer `q` differences remain (`85` Q4, `7` Q6), as do
packed `s` differences that these dot kernels do not consume; downstream
quality remains inside the required gate.

An interleaved stock/fixed diagnostic measured the explicit conversion at
`+0.020 us` Q4 and `+0.005 us` Q6. Combined quantize-plus-dot moved `-2.107%`
Q4 and `-0.063%` Q6; unchanged-dot run variance was larger, so this is retained
as a correctness fix with no observed combined-path regression, not a speedup.

The local host exposes only gfx1151. The documented W7900 alias `epyc` timed
out, so the fix has not yet received a fresh gfx1100 runtime check. No gfx1100
pass is inferred; the prior W7900 matrix remains valid for its measured source.

## gfx1100 versus gfx1151

The two retained matrices use the same shapes, sampling counts, timing modes,
dependency contracts, correctness gates, ratio definition, TheRock/AMD-clang
build, kernel, firmware packages, Mesa/RADV, and Vulkan loader. Executable
sources are identical for every synthetic row below. The refreshed gfx1151
production rows contain only the documented portable-q8 shader delta; their
W7900 counterparts still use the stock shader pending hardware access.

| Family | gfx1100 serial / independent | gfx1151 serial / independent | Transfer read |
| --- | ---: | ---: | --- |
| Dispatch/grid | `2.437x-10.122x` / `1.980x-65.325x` | `1.128x-11.125x` / `1.100x-149.207x` | Vulkan leads on both; the absolute floor and range differ. |
| Geometry | `0.360x-0.790x` / `1.100x-3.925x` | `0.708x-0.990x` / `2.616x-21.328x` | Same HIP-serial/Vulkan-independent split, much wider independent range on gfx1151. |
| Reduction | `0.304x-0.729x` / `1.110x-4.035x` | `0.662x-0.987x` / `2.545x-20.794x` | Same mode split, stronger serialized HIP and narrower independent Vulkan lead on gfx1100. |
| Memory/waitcnt | `0.517x-0.936x` / `0.544x-2.139x` | `0.877x-1.087x` / `0.970x-1.149x` | Does not broadly transfer: gfx1100 serial favors HIP while gfx1151 stays near parity. |
| Packed dot | `1.052x-1.133x` / `1.872x-2.106x` | `3.049x-3.210x` / `3.764x-4.151x` | Vulkan leads on both, but the gfx1151 `3x-4x` magnitude collapses on gfx1100. |
| VOPD | `0.391x-0.561x` / `0.516x-0.616x` | `1.085x-1.181x` / `1.001x-1.108x` | Direction flips: HIP wins every gfx1100 row; Vulkan modestly wins gfx1151. |
| Sampler | `0.259x-0.501x` / `0.782x-2.563x` | `0.508x-1.167x` / `1.522x-3.973x` | Serialized HIP advantage strengthens; gfx1100 independent rows become mixed. |
| Two-stage reduction | `0.324x-0.925x` / `0.394x-0.813x` | `0.685x-0.941x` / `0.679x-4.095x` | Serialized HIP transfers; gfx1151 independent rows cross parity and remain variable. |

Production-shaped combined operations are more consistent than the synthetic
families:

| Combined operation | gfx1100 serial / independent | gfx1151 serial / independent | Transfer read |
| --- | ---: | ---: | --- |
| Q4 selected-dual | `0.501x-0.562x` / `0.432x-0.477x` | `0.932x-1.007x` / `0.874x-0.921x` | HIP wins gfx1100 and gfx1151 independent rows; gfx1151 serialized is within 0.7% of parity. |
| Q6 selected-down X8 | `0.675x` / `0.673x` | `0.550x` / `0.477x` | HIP wins both bounded matrices. |
| Dense Q8_0 | `0.393x-0.966x` / `0.388x-1.030x` | `0.545x-0.902x` / `0.457x-1.194x` | Mostly HIP on both; only small independent rows approach or cross parity. |

The architecture-specific signal is therefore real enough to block ratio
transfer: only dispatch and the geometry/reduction mode split reproduce
qualitatively across the full synthetic families. Packed-dot magnitude, VOPD,
sampler independent throughput, and two-stage independent throughput differ
materially. Software versions and all synthetic executable sources are already
matched. Separating GPU architecture from automatic clock residency and
runtime scheduling now requires fixed or continuously recorded clocks,
interleaved backend order, and queue/kernel counters—not another
version-matching pass.

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

## Optimization and Escalation Plan

### Start here: already tested, ruled out, or inapplicable

The generic optimization checklist is useful, but it must not reset completed
controls or turn hypotheses into conclusions. “Ruled out” below means ruled
out as the broad explanation for the retained gaps; a new production kernel
may still present a different measured bottleneck.

| Suggested lever or explanation | Classification | Current evidence | Policy |
| --- | --- | --- | --- |
| Native target and optimization level | Already satisfied | Retained HIP micros compile explicitly for `gfx1100` or `gfx1151` at `-O3`; the ISA tools already generate saved intermediates and final code objects during their runs. | Keep these fixed in every A/B. They are prerequisites, not new optimizations. |
| `__launch_bounds__` / fixed workgroup | Already tested; not a general cause | Relevant dot, memory, sampler, geometry, and two-stage runners already have fixed-shape launch-bound controls. Fixed-block/workgroup experiments did not broadly close the gaps. | Retest the second occupancy hint only for a profiled kernel whose resource report predicts a useful occupancy boundary. Never assume it is automatically high impact. |
| Register pressure, missing `__restrict__`, or compiler spills | Ruled out as the broad explanation | Relevant retained HIP micros already use restricted pointers. Retained ISA rows report no HIP scratch or spills as the general explanation. | Shorten live ranges or tune unrolling only against measured VGPR/occupancy evidence. Do not manually spill to LDS without a demonstrated register bottleneck. |
| Explicit wave/block address indexing | Tested and retained on one production leaf | The dominant BF16 Q8T16 dual-split GEMV was already spill-free/full-occupancy, but expressing K traversal as `(wave, block_idx, lane)` reduced the production-shaped micro **3.108%**, the marked model leaf **1.349%**, and clean p512/d128 eager wall **0.308%**, all bit-exact. | Keep the production BF16 change. Apply the same rewrite to another Q8T16 body only after its own paired micro and model-family profile; do not infer that every vector-derived address loop will improve. |
| Missing packed-dot lowering | Ruled out | HIP already uses `__builtin_amdgcn_sudot4`, and retained Q4/Q6 ISA joins show dot4 instructions on both backends. | Investigate surrounding loads, address arithmetic, reductions, waits, and scheduling; do not repeat basic dot-lowering probes. |
| Wave64 | Ruled out as a recovery | Wave64 did not close the packed-dot or memory gaps; fixed-wave64 geometry regressed. | Keep wave32 for these paths unless a different production kernel supplies contrary evidence. |
| Generic reduction, accumulator, two-stage, or VOPD variants | Already tested; no broad recovery | Generic LDS/subgroup/accumulator/two-stage and VOPD controls are retained. HIP, not RADV, emits VOPD in the relevant retained gfx1151 ISA rows. | Do not repeat broad sweeps. Add a variant only when a production profile identifies a different hot shape or dependency contract. |
| `-ffast-math` | Inapplicable as a blanket fix | It changes the floating-point contract and is not needed to explain integer dot lowering. | Test only on a named floating-point kernel with an explicit accuracy budget and the normal KL/top-1 gate; never enable globally from this comparison. |
| `-munsafe-fp-atomics` | Inapplicable to retained targets | The retained target kernels are not blocked on floating-point atomic lowering. | Reconsider only if profiling identifies an FP-atomic hot path. |
| Manual LDS spilling | Inapplicable without a register bottleneck | The retained HIP ISA rows do not spill, and adding LDS traffic can reduce rather than improve performance. | Do not try it without resource/counter evidence that occupancy is limited by a live range we cannot otherwise remove. |
| Undocumented `-mllvm` if-conversion or scheduler switches | Unproven and unstable | No retained result establishes that these switches help, and backend option names/semantics can change between LLVM builds. | Disposable diagnosis only: first confirm the exact compiler accepts the option, record it in the artifact, and never make it a product default without a same-suite win. |
| “gfx1151 LLVM regression” | Not established | Cross-architecture ratios differ, but gfx1100 and gfx1151 are different devices. The matched ROCm 7.13-to-7.15 gfx1151 snapshots show no uniform performance change. | Do not call this a compiler regression without a same-device compiler-version A/B or a minimized target-specific ISA/codegen defect. |

The first production-backed source result is now the Q8T16 dual-split
wave/block rewrite. It is deliberately modest but real: clean `8184355c ->
e20cdc13` p512/d128 eager moves **20.5342 -> 20.4709 ms/token**, while marked
dual-split GPU time moves **4245.4 -> 4188.2 us/token**. Static occupancy and
spill state are unchanged, so this is evidence that making wave-uniform address
structure explicit can help current LLVM scheduling/address generation even
when the usual register-pressure diagnosis is negative. The compact evidence is
[`2026-07-12-gfx1151-q8-t16-waveblock-production.json`](../benchmarks/results/2026-07-12-gfx1151-q8-t16-waveblock-production.json).

The most promising remaining compiler diagnostic remains packed dot on gfx1151,
because its Vulkan lead is stable and much larger than on gfx1100. It is not
yet a production optimization target: current combined Q4, Q6, and dense-Q8
operations mostly favor HIP or sit near parity. Extend its retained ISA work
with dynamic counters only if production profiling exposes the same packed-dot
body. The dispatch floor is separately actionable through production fusion
and graph replay, followed by HSA/AQL/queue tracing if a residual floor remains.

### What we can inspect despite having no PTX equivalent

The absence of a stable PTX-like virtual ISA is not an attribution dead end.
HIP/Clang can preserve LLVM IR, AMDGPU assembly, and the final HSA ELF code
object; that object records kernel resource metadata and contains the final
machine code. Vulkan supplies SPIR-V, while RADV can dump its lowered shader
statistics and final ACO ISA. The two front-end pipelines do not share an IR,
but their final ISA executes on the same GPU and can be compared alongside
dynamic counters.

For this repository, use the existing `benchmarks/micro/runners/isa_stats.py`,
`q4_selected_dual_isa_stats.py`, and `q6_x8_isa_stats.py` extractors rather than
an approximate one-off disassembly command. Their artifacts retain exact build
commands, source hashes, HIP code-object metadata, and parsed
`llvm-objdump`/RADV `shaderstats` evidence. On the retained TheRock 7.15 compiler,
`-Rpass-analysis=kernel-resource-usage` also reports SGPR/VGPR use, scratch,
spills, LDS, and estimated waves/SIMD. The similarly named
`hipcc --resource-usage` option shown in current ROCm documentation is rejected
by this exact compiler build, so it is not a reproducibility command here.

### Work we control

Optimization should proceed from a shipped hot bucket, not from the largest
synthetic ratio. Each stage has a stop gate:

| Order | Lever in this tree | Required evidence before retaining it |
| ---: | --- | --- |
| 1 | Profile the current PARO and GGUF server paths, including graph replay and eager fallbacks. Name the kernel, layer, quant, shape, launch count, and serialized/overlapped behavior. | The bucket is exposed in production wall time and the proposed change can affect it. |
| 2 | Make the A/B controlled: exact source and math, native `--offload-arch`, fixed or continuously sampled clocks, interleaved backend/variant order, identical buffers, and the applicable correctness oracle. | Timing distributions reproduce; a clock, submission-class, or layout mismatch cannot explain the result. |
| 3 | Join static and dynamic evidence: LLVM IR/AMDGPU ISA, RADV final ISA, workgroup/wave mode, SGPR/VGPR/LDS/scratch/spills, instruction classes and waitcnts, plus filtered `rocprofv3` counters or PC/thread trace where supported. | One concrete bottleneck is identified; static instruction counts alone are not treated as timing proof. |
| 4 | Try the lowest-risk matching HIP change: fewer launches/fusion or graph reuse; then layout and work distribution; then lifetime, aliasing, and unroll changes; then a documented AMDGPU builtin or scheduler intrinsic. | The same production-shaped slice improves, passes its full correctness gate, and does not regress the surrounding suite. |
| 5 | Use narrow inline AMDGCN assembly or a standalone HSACO only when the final ISA shows a specific sequence LLVM cannot express or schedule adequately. | The sequence is small, architecture-gated, has an unfused/reference fallback, and improves end-to-end production wall—not only a synthetic kernel. |
| 6 | Bisect compiler builds on the same hardware and unchanged source when codegen remains suspect. | A good/bad compiler boundary or a minimal stable codegen reproducer exists before escalation. |

Dispatch and kernel-body work remain separate. For the former, our controllable
levers are graph instantiation/reuse, graph-node parameter updates, fewer graph
launches, fusion, persistent buffers, and removal of host synchronizations. For
the latter, the controllable levers are algorithm/layout, native workgroup and
wave geometry, source-level live ranges, alias information, bounded unrolling,
and target-specific builtins. A faster Vulkan command-buffer floor is not
evidence that changing HIP kernel math will reduce dispatch overhead.

### Work that may require upstream ownership

We can work around or even patch open-source components, but these durable
fixes are not under normal hipEngine release control:

| Isolated symptom | Likely owner after isolation | What hipEngine must supply |
| --- | --- | --- |
| Inferior HIP ISA for matched math after source/layout controls | Clang/LLVM AMDGPU backend, normally triaged through ROCm and then LLVM when minimized | Standalone HIP reproducer, exact compiler commits and commands, saved IR/code object, final ISA/resource diff, same-device timings/counters, and correctness. |
| Residual HIP graph/direct-launch submission floor with a trivial kernel body | HIP runtime and possibly HSA queue handling | Direct-versus-instantiated-graph trace, graph topology/reuse details, host API and kernel timestamps, node counts, stream/dependency contract, and Vulkan command-buffer control. |
| Queue scheduling or clock-residency behavior that survives runtime controls | KFD/MES, amdgpu kernel driver, firmware, or power-management stack | Fixed/sampled clocks, kernel/firmware versions, queue/AQL evidence, system trace, and a minimal non-hipEngine reproducer where practical. |
| Incorrect or inferior Vulkan final shader after matched SPIR-V/source controls | Mesa RADV/NIR/ACO | Minimal SPIR-V/GLSL, `RADV_DEBUG=shaders,shaderstats` output, exact Mesa commit, ISA/counters, and correctness. |

Do not file a generic “LLVM is slower than ACO” bug from ratio tables. A
compiler-quality issue is ready when one minimal matched kernel reproduces on
the named GPU, backend submission overhead is excluded, correctness passes,
the final ISA or counters identify a concrete loss, and a compiler-version
boundary or stable current-main result is recorded. The parent ROCm issue can
carry the user-visible cross-backend report; a linked LLVM issue should contain
the reduced backend/codegen case. Runtime or driver evidence should be routed
to that component rather than folded into the compiler claim.

## Open Gates

| Priority | Work | Status | Exit gate |
| ---: | --- | --- | --- |
| 0 | Fix-clock W7900 dispatch/stream attribution | Open | Interleaved one/four-stream and graph controls plus queue/AQL traces separate runtime submission from clock residency. |
| 1 | Profile current PARO and GGUF server paths | GGUF eager slice complete; server/PARO remain open | SOL-G4 and the Q8T16 follow-up identify dense Q8_0 as 44% of marked GPU time and retain one exact source-level win. Repeat the same evidence chain before selecting a different shipped bucket. |
| 1 | Select one production-backed HIP optimization target | Q8T16 dual-split complete | Wave/block K indexing is retained at **-1.349% leaf GPU time** and **-0.308% p512/d128 eager wall**. The next target must come from a fresh profile; do not keep tuning this leaf without a new bottleneck. |
| 1 | Build an issue-quality compiler/runtime packet | Decision-gated | A minimal matched reproducer satisfies the component-specific evidence gate above; broad ratio tables alone do not open an upstream bug. |
| 1 | Validate portable Vulkan q8_1 rounding on gfx1100 | gfx1151 complete; W7900 access pending | The retained shader receives the same strict Q4/Q6 and full paired runtime validation on gfx1100; no result is inferred while the W7900 host is unavailable. |
| 1 | Match Q6 lm-head math/layout | Blocked on comparable implementation | HIP and Vulkan use identical quantization, activation layout, output coverage, and rowtile algorithm before any ratio is reported. |
| 2 | Production Vulkan or hand ISA | Decision-gated | A clean matched production slice wins in the relevant timing mode and final combined operation, then improves end-to-end wall without a memory/correctness regression. |

## References

- [Dated attribution notebook](HIP-vs-VULKAN-HISTORY.md)
- [Microbenchmark runner guide](../benchmarks/micro/README.md)
- [Canonical benchmark scoreboard](../benchmarks/README.md)
- [RDNA3 roofline](ROOFLINE.md)
- [Optimization punchlist](SOL-OPTIMIZATION.md)
- [LLVM AMDGPU backend and code-object reference](https://llvm.org/docs/AMDGPUUsage.html)
- [Clang AMDGPU builtin reference](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html)
- [HIP performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)
- [HIP graph guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)
- [rocprofv3 tracing and counter guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
- [Mesa RADV shader compilation pipeline](https://docs.mesa3d.org/drivers/radv.html)
