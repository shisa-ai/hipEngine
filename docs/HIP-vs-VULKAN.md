# HIP vs Vulkan Attribution Results And Plan

This document records the HIP vs Vulkan/RADV attribution suite and the retained
conclusions from it. It started as a plan for turning "Vulkan/RADV/ACO is better
than HIP/LLVM here" into actionable compiler or backend work; it is now also the
project ledger for what the microbenchmarks have actually proven. Retained
results must follow the evidence policy in `docs/BENCHMARK.md`.

The motivating observation from `docs/ROOFLINE.md` and
`docs/MTP-LLAMACPP-PARITY.md` is narrow:

- hipEngine HIP can beat llama.cpp HIP on exact AR decode.
- llama.cpp Vulkan is still the higher absolute backend ceiling on several
  GGUF decode/prefill rows.
- The suspected causes are a mix of compiler quality, workgroup/subgroup
  shape, launch/runtime behavior, fusion topology, and quant precision/layout.

The goal of this suite is to split those causes cleanly enough to decide
whether the next high-leverage path is an LLVM issue, a HIP kernel rewrite, a
Vulkan backend, or a tiny hand-ISA path.

## Current Conclusion

As of the retained gfx1151/STRIX_HALO runs on 2026-07-08, the retained
conclusion is split. The short answer is **no**: the evidence does not support
"HIP is slower simply because Mesa RADV/ACO is better optimized than
LLVM-AMDGPU" as a single explanation.

- Vulkan/RADV command-buffer replay has a real runtime-dispatch advantage for
  launch-heavy tiny-kernel bursts. This is retained as `runtime_dispatch`, not
  `compiler_aco`.
- Vulkan remains much faster on the matched repeat-shifted f32 GEMV/reduction
  harness, and both backends prefer wg256 on all retained best-native rows.
  This rules out the simple "HIP picked the wrong workgroup size" explanation
  for that harness.
- The first retained ISA/stat extraction does **not** support a simple "ACO
  wins through VOPD pairing" story for this f32 geometry harness. HIP emits two
  static VOPD instructions while RADV emits none in the final shader disassembly
  for the tested wg64/wg256 shapes. HIP also reports no scratch or spills.
- The targeted VOPD scheduling sweep also does **not** support the idea that
  RADV/ACO's ceiling comes from better VOPD pairing on gfx1151. HIP emits VOPD
  in all retained VOPD rows; RADV emits zero VOPD in all final shader
  disassemblies. Vulkan is only modestly faster in the heavier independent and
  mixed/dequant rows, while HIP is faster in independent-2, independent-4, and
  dependent-4.
- The retained memory/waitcnt sweep is the first direct evidence that Vulkan has
  a real advantage on memory-side scheduling/access shapes. On matched
  device-memory load+accumulate rows, Vulkan is `1.30x-2.35x` faster for most
  coalesced, strided, and interleave variants; gather is essentially tied
  (`1.02x`). HIP reports no scratch and no spills, while RADV still exposes only
  estimated register spans. Simple rows show slightly fewer RADV waitcnt-family
  instructions; fixed-shape controls and better RADV allocation stats are still
  needed before turning this into an LLVM-AMDGPU waitcnt/scheduling claim.
- The retained packed dot-path sweep rules out a missing-HIP-dot-instruction
  story for the current q8/q4/q6 idiom. HIP and RADV both emit final dot4
  instructions for q8 signed, q4 unsigned-byte by signed-q8, and q6 zero-point
  correction rows; HIP reports no scratch/spills. Vulkan is still `3.29x-3.43x`
  faster, including the scalar-dequant row (`3.29x`). After the HIP wave64
  controls, the remaining dot-path gap is more likely fixed-shape/surrounding
  scheduling or layout/activation quantization economics than basic dot
  lowering or wave mode.
- The first retained HIP wave64 controls do not close the gap. On packed-dot
  rows, forcing HIP wave64 makes HIP `1.007x-1.061x` slower than the retained
  wave32 HIP rows. On memory/waitcnt rows, HIP wave64 is mixed but still leaves
  Vulkan faster on most shapes; gather regresses `6.35x` versus HIP wave32.
- We still cannot claim `compiler_aco` for the f32 geometry gap. RADV official
  VGPR/SGPR allocation counts were not exposed by `RADV_DEBUG=shaders`, and the
  current evidence mixes HIP runtime `blockDim` code with Vulkan specialization
  constants and different wave/subgroup modes.

What we have ruled out:

- Do not attribute the current Vulkan ceiling to RADV/ACO finding VOPD/dual
  issue opportunities that LLVM/HIP misses. The retained gfx1151 VOPD evidence
  points the other way: HIP emits VOPD, RADV does not.
- Do not attribute the f32 geometry gap to HIP spills or scratch use. HIP
  reports `0` scratch and `0` spills in the retained ISA/stat rows.
- Do not attribute the f32 geometry gap only to a bad HIP workgroup-size choice.
  HIP and Vulkan both prefer wg256 in the retained best-native rows.
- Do not attribute current q8/q4/q6 dot-path gaps to HIP failing to emit dot4.
  HIP emits the expected dot4 instructions in the retained packed-dot rows.
- Do not treat HIP wave64 as the missing switch for the retained dot/memory
  gaps. It does not close dot, and it severely hurts the retained gather row.

What remains plausible:

- Vulkan has a proven dispatch/runtime advantage on gfx1151.
- Vulkan has a large matched-math advantage in one f32 diagnostic, but that row
  is still `diagnostic_unclassified`.
- Memory/access scheduling is still the strongest compiler-facing lead, but HIP
  wave64 alone is not the fix. It may become an LLVM-AMDGPU waitcnt/scheduling
  issue only after fixed-shape controls and real-slice checks remove the
  remaining specialization/layout confounds.
- Dot-instruction availability is no longer untested for the packed q8/q4/q6
  idiom: HIP emits dot4. HIP wave64 also does not close the dot gap. The
  remaining dot-path work is to remove fixed-shape confounds, check q8_1
  materialization/layout costs in a real slice, and only then decide whether any
  narrow hand-ISA sequence is worth carrying.
- Real inference slices still need to confirm that the microbench deltas predict
  shipped hot buckets.

Operationally, keep the Vulkan work as an attribution/probe path until real
inference slices prove production value. The HIP roadmap should first try to
reproduce the useful pieces inside HIP: fixed-shape kernels, dot intrinsics or
small hand-ISA sequences where proven, better launch fusion, and memory/waitcnt
controls. A production Vulkan backend is not justified by the dispatch row or
generic geometry row alone.

HIP also does not give us a PTX-equivalent escape hatch in the normal runtime
path. We can inspect LLVM IR, AMDGPU assembly, and code-object metadata, and we
can use AMDGCN builtins, inline AMDGCN assembly, or standalone HSACO/module
kernels for narrow cases. But normal HIP source is ultimately relying on
LLVM-AMDGPU codegen, so confirmed compiler misses become either LLVM roadmap
items or carefully scoped hand-ISA candidates.

## What To Test Next

Yes, there is more worth testing, but the useful list is now narrow. Do **not**
spend more gfx1151 time on dispatch-only, broad geometry-only, generic VOPD, or
generic memory sweeps. Those have already answered the coarse questions. The
next useful tranche is decision-grade controls:

1. HIP fixed-shape/specialization controls for the f32 geometry, memory, and
   dot-path kernels: compile fixed workgroup-size variants and remove runtime
   `blockDim` address/control overhead before attributing the remaining gap to
   LLVM scheduling. HIP wave64 has now been tested for dot and memory and did
   not close the gap.
2. One representative inference slice after the microbench deltas are
   classified. The best first slices are selected-MoE small-K or q6 lm-head
   rowtile, because they map directly to exposed hipEngine buckets.
3. q8_1 materialization/layout accounting for the dot path, preferably inside
   the same real slice, because the retained packed-dot result says the basic
   instruction is present but not whether activation quantization and layout
   economics are production-positive.
4. Cross-GPU reruns only after the harnesses above stabilize. gfx1100/W7900 and
   7900 XTX reruns should check portability of a classified diagnosis, not
   replace the missing controls.

Priority summary:

| Priority | Test | Decision It Enables |
| --- | --- | --- |
| Done | Dot-path q8/q4/q6 kernels with dot ISA counts | HIP emits dot4; remaining gap is not basic dot lowering |
| Done | HIP wave64 dot/memory controls | Wave64 does not close dot/memory gaps and regresses gather |
| P0 | HIP fixed-shape memory/dot/geometry controls | Decide whether retained gaps are specialization/runtime-shape or LLVM scheduling |
| P1 | Fixed-workgroup HIP geometry variants | Remove the Vulkan specialization-constant vs HIP runtime-`blockDim` confound |
| P1 | One real selected-MoE or q6 lm-head slice | Check whether the microbench diagnosis predicts a shipped hot bucket |
| P2 | gfx1100/W7900 and 7900 XTX reruns | Check portability after fixed-shape and real-slice harnesses are classified on gfx1151 |

Tooling that would improve attribution quality, but should not displace the P0
tests:

- Better RADV allocation/stat extraction, via whatever Mesa/RADV or RGP path
  exposes official VGPR/SGPR allocation counts for final shaders.
- A small comparison utility that rolls HIP/Vulkan artifacts into a one-page
  retained-result diff with timing, correctness, wave mode, instruction counts,
  waits, dot/VOPD counts, and classification.

Cross-GPU reruns on gfx1100/W7900 and 7900 XTX are important after the harnesses
are stable. They should confirm portability of the gfx1151 conclusions, not
replace the remaining HIP fixed-shape/specialization and real-slice tests.

## Current Retained Evidence

### gfx1151 Dispatch/Grid Floor

Retained artifact:
`benchmarks/micro/results/gfx1151/strix-halo/dispatch-floor-comparison.json`.
The artifact records exact HIP/Vulkan commands and references the shared
environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- Benchmark family: dispatch/grid floor with trivial shader bodies.
- Classification: `runtime_dispatch`.

Retained result:

| Shape | Vulkan command-buffer replay | HIP direct | HIP graph |
| --- | ---: | ---: | ---: |
| 941 dispatches, 1 block, tiny args | `0.043621 us/dispatch` | `2.0087 us/dispatch` | `1.8069 us/dispatch` |

Grid sweep at 941 dispatches:

| Blocks | Vulkan | HIP graph | Vulkan vs HIP graph |
| ---: | ---: | ---: | ---: |
| 1 | `0.042902 us/dispatch` | `1.85857 us/dispatch` | `43.3x` |
| 128 | `0.230992 us/dispatch` | `1.86143 us/dispatch` | `8.1x` |
| 1024 | `1.69237 us/dispatch` | `3.07608 us/dispatch` | `1.8x` |
| 8192 | `11.977879 us/dispatch` | `13.02259 us/dispatch` | `1.09x` |

Conclusion: Vulkan/RADV command-buffer replay is dramatically cheaper than HIP
direct launches and HIP graph replay for launch-heavy one-block bursts on this
gfx1151 system. The gap collapses as grid work grows, so this result supports
kernel fusion, fewer launches, and a narrow Vulkan probe for launch-heavy paths.
It does **not** prove that RADV/ACO emits better math code than LLVM-AMDGPU,
because the shader body is intentionally trivial and no ISA/stat evidence was
collected for compiler scheduling.

Immediate reads:

- HIP graph replay trims the direct-launch floor only modestly in the retained
  N=941 row (`2.0087` to `1.8069 us/dispatch` for tiny args).
- HIP argument count is not the dominant cost in this harness: wide args add
  about `0.05 us/dispatch` at N=941.
- The remaining compiler questions need matched math kernels with disassembly,
  VGPR/SGPR/scratch, waitcnt, VOPD, and dot-instruction evidence.

### gfx1151 F32 GEMV Geometry Sweep

Retained artifact:
`benchmarks/micro/results/gfx1151/strix-halo/geometry-sweep-comparison.json`.
Backend artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-geometry-sweep.json` and
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-geometry-sweep.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-geometry-sweep.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- Benchmark family: repeat-shifted f32 GEMV/reduction, one workgroup per row,
  CPU oracle, K=`512/2048/8192`, rows=`1/4/8`,
  workgroup=`32/64/128/256`, body repeats=`128`.
- Classification: `diagnostic_unclassified`.

Retained best-native shape summary:

| K | Rows | HIP best | Vulkan best | Vulkan vs HIP |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | wg256, `23.5944 us` | wg256, `3.0412 us` | `7.76x` |
| 512 | 4 | wg256, `23.6302 us` | wg256, `3.2942 us` | `7.17x` |
| 512 | 8 | wg256, `23.6439 us` | wg256, `4.0837 us` | `5.79x` |
| 2048 | 1 | wg256, `85.6870 us` | wg256, `7.3849 us` | `11.60x` |
| 2048 | 4 | wg256, `86.2845 us` | wg256, `7.7645 us` | `11.11x` |
| 2048 | 8 | wg256, `82.9754 us` | wg256, `8.7624 us` | `9.47x` |
| 8192 | 1 | wg256, `392.8771 us` | wg256, `28.0126 us` | `14.03x` |
| 8192 | 4 | wg256, `396.4413 us` | wg256, `31.5848 us` | `12.55x` |
| 8192 | 8 | wg256, `408.1349 us` | wg256, `33.9988 us` | `12.00x` |

Conclusion: in this matched f32 GEMV/reduction harness, HIP and Vulkan both
prefer the 256-thread workgroup. Moving HIP from 64 to 256 threads improves
substantially, but it does **not** close the Vulkan gap. Vulkan remains
`5.79x-14.03x` faster on best-native rows, and the largest identical-shape
speedups are `12.88x-15.34x` at smaller workgroups.

This rules out the simple "HIP used the wrong workgroup size" explanation for
this microbench. It still does **not** prove `compiler_aco`; the ISA/stat
extraction below rules out some simple compiler stories but does not yet
identify a primary cause.

Immediate reads:

- Workgroup shape matters for HIP, but all retained best rows are wg256 on both
  backends.
- The matched-shape Vulkan gap is large enough that wave/subgroup mode, fixed
  workgroup specialization, memory scheduling, or a hidden harness/runtime
  difference must be tested directly.
- This result makes VOPD-specific, memory/waitcnt, dot-path, and HIP
  fixed-workgroup-specialization controls higher priority than more geometry
  sweeps on gfx1151.

### gfx1151 F32 Geometry ISA/Stat Extraction

Retained artifact:
`benchmarks/micro/results/gfx1151/strix-halo/geometry-isa-stats-comparison.json`.
Backend artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-geometry-isa-stats.json` and
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-geometry-isa-stats.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-isa-stats.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- Benchmark family: ISA/stat extraction for the retained f32 geometry kernel,
  K=`2048`, rows=`1`, workgroup=`64/256`, body repeats=`128`.
- HIP evidence: `hipcc --save-temps`, `llvm-readobj --notes`, and
  `llvm-objdump -d --no-show-raw-insn`.
- Vulkan evidence: `RADV_DEBUG=shaders` final disassembly plus ACO after-RA
  presence. RADV did not print official VGPR/SGPR allocation counts in this
  environment, so Vulkan register rows are estimated physical register spans,
  not allocation-count claims.
- Classification: `diagnostic_unclassified`.

Retained ISA/stat summary:

| Workgroup | HIP ISA | Vulkan/RADV ISA |
| ---: | --- | --- |
| 64 | actual `18` SGPR, `11` VGPR, scratch `0`, spills `0`, wave32, `118` static instructions, `6` waitcnt-family instructions, `2` VOPD instructions / `4` VOPD ops | estimated span `16` SGPR / `9` VGPR, wave64, `100` static instructions, `9` waitcnt-family instructions, `0` VOPD |
| 256 | actual `18` SGPR, `11` VGPR, scratch `0`, spills `0`, wave32, `118` static instructions, `6` waitcnt-family instructions, `2` VOPD instructions / `4` VOPD ops | estimated span `16` SGPR / `9` VGPR, wave64, `142` static instructions, `20` waitcnt-family instructions including `9` depctr waits, `0` VOPD |

Matched timing context from the geometry sweep:

| Shape | HIP median | Vulkan median | Vulkan vs HIP |
| --- | ---: | ---: | ---: |
| K=2048 rows=1 wg64 | `292.1010 us` | `21.6447 us` | `13.50x` |
| K=2048 rows=1 wg256 | `85.6870 us` | `7.3849 us` | `11.60x` |

Conclusion: the retained f32 geometry gap is **not** explained by LLVM missing
VOPD pairing. In this kernel, HIP emits VOPD and RADV does not. It is also not
explained by HIP spills or scratch use; HIP reports no spills and no scratch.
The remaining plausible causes are narrower: Vulkan specialization constants
versus HIP runtime `blockDim` code, wave64/subgroup reduction behavior,
memory/address scheduling, pipeline/runtime effects not captured by static ISA,
or a source/harness detail still not controlled.

Immediate reads:

- Do not file an LLVM VOPD issue from this geometry kernel.
- Do not claim RADV has better register allocation from this artifact; Vulkan
  official allocation counts are missing.
- Add fixed-workgroup HIP variants to remove dynamic `blockDim` overhead before
  using this f32 geometry row as a compiler-codegen proxy.
- The next compiler-facing tests should be memory/waitcnt microbenches where
  load scheduling is isolated by design.

### gfx1151 VOPD/VALU Scheduling Sweep

Retained artifact:
`benchmarks/micro/results/gfx1151/strix-halo/vopd-sweep-comparison.json`.
Backend artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-vopd-sweep.json` and
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-vopd-sweep.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-vopd-sweep.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- Benchmark family: pure VALU VOPD diagnostics with sampled CPU oracle,
  N=`65536`, body iters=`2048`, block size=`256`.
- Variants: independent f32 FMA with `2/4/8` accumulators, dependent f32 FMA
  with `4` chained operations, mixed int+float with `4` accumulators, and
  dequant-like shift/mask/cvt/FMA with `4` accumulators.
- Classification: `diagnostic_unclassified`, with a negative result for the
  "ACO wins through VOPD" hypothesis.

Retained timing and VOPD summary:

| Variant | HIP median | Vulkan median | Vulkan vs HIP | HIP VOPD | RADV VOPD |
| --- | ---: | ---: | ---: | ---: | ---: |
| independent f32 FMA, 2 accum | `250.7432 us` | `285.6322 us` | `0.88x` | `4` | `0` |
| independent f32 FMA, 4 accum | `447.9288 us` | `462.4270 us` | `0.97x` | `9` | `0` |
| independent f32 FMA, 8 accum | `877.3634 us` | `833.0101 us` | `1.05x` | `12` | `0` |
| dependent f32 FMA, 4 ops | `426.6930 us` | `453.9037 us` | `0.94x` | `1` | `0` |
| mixed int+float, 4 accum | `1020.2676 us` | `946.9001 us` | `1.08x` | `6` | `0` |
| dequant-like, 4 accum | `1019.7762 us` | `978.8506 us` | `1.04x` | `5` | `0` |

Register/stat summary:

- HIP reports no scratch in every retained VOPD row. VGPRs rise with
  independent accumulators: `8/11/21` VGPR for `2/4/8` independent accumulators.
- RADV official VGPR/SGPR allocation counts are still unavailable from
  `RADV_DEBUG=shaders`; the artifact records estimated physical register spans.
- RADV emits wave64 final shaders in these rows; HIP emits wave32 code objects.
- Both backends pass the sampled CPU oracle with max abs `2.384185791e-07`.

Conclusion: do **not** attribute the current Vulkan ceiling to RADV/ACO finding
better VOPD pairing than LLVM/HIP. In this targeted family, LLVM/HIP is the
backend emitting VOPD. Vulkan's modest wins on independent-8, mixed int+float,
and dequant-like rows must come from something else: wave64 execution shape,
non-VOPD scheduling, instruction selection, runtime/pipeline effects, or
measurement noise. The next relevant compiler tests are fixed-shape controls
and real-slice confirmation, not more generic VOPD speculation.

### gfx1151 Memory / Waitcnt Sweep

Retained artifact:
`benchmarks/micro/results/gfx1151/strix-halo/memory-waitcnt-comparison.json`.
Backend artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-memory-waitcnt.json` and
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-memory-waitcnt.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-memory-waitcnt.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- Benchmark family: device-memory load+accumulate diagnostics with sampled CPU
  oracle, N=`32768`, body iters=`128`, block size=`256`.
- Variants: coalesced vector-width loads `1/2/4/8`, strided loads
  `2/4/8/16`, gather-ID loads, and load/compute interleave
  `1/2/4/8/16`.
- Classification: `diagnostic_unclassified`, with strong memory/waitcnt
  evidence but not yet a pure `compiler_aco` proof because wave32 vs wave64 and
  RADV allocation-count visibility remain confounds.

Retained timing and ISA summary:

| Variant | HIP median | Vulkan median | Vulkan vs HIP | HIP GB/s | Vulkan GB/s | HIP wait/load | RADV wait/load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coalesced width 1 | `26.6127 us` | `16.8441 us` | `1.58x` | `630.42` | `996.03` | `4/1` | `3/1` |
| coalesced width 2 | `44.6948 us` | `32.8837 us` | `1.36x` | `750.75` | `1020.40` | `5/2` | `4/2` |
| coalesced width 4 | `282.1829 us` | `124.3344 us` | `2.27x` | `237.82` | `539.74` | `7/4` | `6/4` |
| coalesced width 8 | `639.5150 us` | `289.0281 us` | `2.21x` | `209.87` | `464.38` | `11/8` | `10/8` |
| strided stride 2 | `44.5751 us` | `31.4315 us` | `1.42x` | `376.38` | `533.77` | `4/1` | `3/1` |
| strided stride 4 | `281.3058 us` | `119.6366 us` | `2.35x` | `59.64` | `140.23` | `4/1` | `3/1` |
| strided stride 8 | `557.0604 us` | `269.2068 us` | `2.07x` | `30.12` | `62.32` | `4/1` | `3/1` |
| strided stride 16 | `1144.5523 us` | `645.8128 us` | `1.77x` | `14.66` | `25.98` | `4/1` | `3/1` |
| gather IDs | `493.0500 us` | `484.9378 us` | `1.02x` | `68.05` | `69.19` | `5/2` | `4/2` |
| interleave unroll 1 | `25.8722 us` | `15.4284 us` | `1.68x` | `648.47` | `1087.42` | `4/1` | `3/1` |
| interleave unroll 2 | `49.8455 us` | `38.2383 us` | `1.30x` | `673.17` | `877.51` | `5/2` | `4/2` |
| interleave unroll 4 | `281.1671 us` | `128.7873 us` | `2.18x` | `238.68` | `521.08` | `6/4` | `6/4` |
| interleave unroll 8 | `580.6280 us` | `288.4835 us` | `2.01x` | `231.16` | `465.25` | `10/8` | `10/8` |
| interleave unroll 16 | `1611.1334 us` | `1428.9660 us` | `1.13x` | `166.61` | `187.85` | `13/16` | `18/16` |

Register/stat summary:

- All HIP and Vulkan rows pass the sampled CPU oracle with max abs `0.0`.
- HIP reports wave32, no scratch, and no spills in all retained rows. HIP VGPR
  rises with interleave width from `8` at unroll 1 to `36` at unroll 16.
- RADV final shaders are wave64. Official RADV VGPR/SGPR allocation counts are
  still unavailable; the artifact records estimated physical register spans.
- Simple coalesced, strided, gather, and low-unroll interleave rows show one
  fewer RADV waitcnt-family instruction than HIP at the same static load count.
  Wider interleave rows have equal or higher RADV waitcnt counts, so waitcnt
  count alone is not the whole story.

Conclusion: memory/access scheduling is now a serious candidate for the Vulkan
ceiling. Vulkan is consistently faster on coalesced, strided, and most
interleave rows, while gather is essentially tied. This does **not** yet justify
a clean LLVM `compiler_aco` issue because the retained rows still compare HIP
wave32 against RADV wave64 and RADV allocation counts are estimated. The next
control is fixed-shape memory and geometry variants; the dot-path result below
separately shows basic q8/q4/q6 dot lowering is present in HIP.

### gfx1151 Packed Dot Path

Retained artifact:
`benchmarks/micro/results/gfx1151/strix-halo/dot-path-comparison.json`.
Backend artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-dot-path.json` and
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-dot-path.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-dot-path.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- Benchmark family: packed int8 dot diagnostics with exact sampled CPU oracle,
  N=`32768`, body iters=`128`, groups=`16`, block size=`256`.
- Variants: q8 signed dot, q4 unsigned-byte by signed-q8 dot, q6 zero-point
  correction (`dot_u - 32 * q8_sum`), and scalar q4 dequant.
- Classification: `diagnostic_unclassified`; useful dot-lowering evidence, but
  not a clean `compiler_aco` proof because HIP wave32 vs RADV wave64 and
  specialization/lowering differences remain.

Retained timing and ISA summary:

| Variant | HIP median | Vulkan median | Vulkan vs HIP | HIP dot4 | RADV dot4 | SPIR-V dot op | HIP wait/load | RADV wait/load |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| q8 signed | `7114.77 us` | `2071.66 us` | `3.43x` | `16` | `16` | `OpSDot=1` | `19/32` | `18/32` |
| q4 unsigned x q8 | `7109.06 us` | `2085.04 us` | `3.41x` | `16` | `16` | `OpSUDot=1` | `19/32` | `18/32` |
| q6 zero-corrected | `6831.89 us` | `2076.40 us` | `3.29x` | `32` | `32` | `OpSUDot=2` | `20/32` | `18/32` |
| scalar q4 dequant | `7342.70 us` | `2228.58 us` | `3.29x` | `0` | `0` | none | `35/32` | `20/32` |

Register/stat summary:

- All HIP and Vulkan rows pass the exact sampled CPU oracle with max abs `0.0`.
- HIP reports wave32, no scratch, and no spills. HIP dot rows use
  `41-42` VGPR and `14` SGPR; scalar dequant uses `50` VGPR.
- RADV final shaders are wave64. Official RADV VGPR/SGPR allocation counts are
  still unavailable; estimated spans are `34` VGPR for q8/q4 and `48` VGPR for
  q6/scalar.
- HIP and RADV emit the same final dot4 counts in q8/q4/q6 rows. Vulkan SPIR-V
  also contains the expected `OpSDot`/`OpSUDot` operations before RADV lowering.
- The scalar row is also `3.29x` faster on Vulkan despite using no dot4, so the
  retained gap cannot be explained only by dot-instruction selection.

Conclusion: do **not** spend more gfx1151 time proving whether HIP can emit the
basic q8/q4/q6 dot instruction; it can. The retained dot-path gap is still
large, but the useful next controls are fixed-shape dot variants and
a representative real slice that includes q8_1 activation materialization and
layout costs. A hand-ISA path is not justified by this artifact alone; it would
need to beat the same HIP dot body after wave/fixed-shape controls and then
move a shipped selected-MoE or q6 lm-head slice.

### gfx1151 HIP Wave64 Controls

Retained artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/dot-path-wave64-comparison.json`
and
`benchmarks/micro/results/gfx1151/strix-halo/memory-waitcnt-wave64-comparison.json`.
Backend artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-dot-path-wave64.json`,
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-dot-path-wave64-control.json`,
`benchmarks/micro/results/gfx1151/strix-halo/hip-memory-waitcnt-wave64.json`,
and
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-memory-waitcnt-wave64-control.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-wave64-controls.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- HIP flag: `-mwavefrontsize64` via `--hip-wavefront-size 64`.
- Classification: `diagnostic_unclassified`; this is a wave-mode control, not a
  same-geometry compiler proof.

Dot-path wave64 result:

| Variant | HIP wave32 median | HIP wave64 median | Wave64 / wave32 | Same-commit Vulkan vs HIP wave64 |
| --- | ---: | ---: | ---: | ---: |
| q8 signed | `7114.77 us` | `7548.48 us` | `1.061x` slower | `3.13x` |
| q4 unsigned x q8 | `7109.06 us` | `7350.55 us` | `1.034x` slower | `3.55x` |
| q6 zero-corrected | `6831.89 us` | `7106.02 us` | `1.040x` slower | `2.63x` |
| scalar q4 dequant | `7342.70 us` | `7397.10 us` | `1.007x` slower | `3.16x` |

Memory/waitcnt wave64 result:

| Group | HIP wave64 vs wave32 | Same-commit Vulkan vs HIP wave64 |
| --- | --- | --- |
| coalesced | width 1/2/4 slower by `1.017x-1.098x`; width 8 faster by `0.921x` | Vulkan still `1.70x-2.08x` faster |
| strided | mixed `0.977x-1.022x` vs wave32 | Vulkan still `1.68x-2.29x` faster |
| gather | `6.349x` slower than wave32 | Vulkan `6.59x` faster than HIP wave64 |
| interleave | mixed; unroll 16 faster by `0.922x`, unroll 1 slower by `1.146x` | Vulkan faster except unroll 16, where HIP wave64 is `1.15x` faster |

Conclusion: HIP wave64 is not the missing switch. It does not close the packed
dot gap, and the memory result is mixed with a severe gather regression. Keep
fixed-shape/specialization and real-slice tests as the next controls; do not
promote broad HIP wave64 routing from this evidence.

## Questions To Answer

1. **Compiler scheduling:** When the algorithm, data layout, wave/subgroup size,
   and workgroup geometry are matched, does RADV/ACO still beat
   LLVM-AMDGPU? If yes, is the delta visible as fewer VGPRs, less scratch, fewer
   `s_waitcnt`, better unroll, or more VOPD pairing?
2. **Geometry:** How much of the Vulkan win comes from 64-thread subgroup
   shapes versus the common HIP 128/256-thread block shapes?
3. **Wave mode:** HIP wave64 did not close retained dot/memory gaps; does the
   same answer hold for fixed-shape geometry and cross-GPU reruns?
4. **Dispatch/runtime:** Is Vulkan faster because individual shaders are faster,
   or because command-buffer/pipeline execution reduces per-dispatch cost?
5. **Memory scheduling:** Retained gfx1151 rows show higher Vulkan bandwidth on
   coalesced, strided, and interleave loops. Does that survive fixed-shape
   controls, and does it predict quantized GEMV inner loops?
6. **VOPD portability:** gfx1151 retained evidence is negative for "ACO finds
   VOPD that LLVM misses." Do gfx1100/W7900 and 7900 XTX reproduce that answer,
   or is this driver/GPU-specific?
7. **dp4a/sudot4:** Does the compiler matter once the code uses the intended
   RDNA3 dot instruction? Retained gfx1151 packed-dot rows say HIP and RADV
   both emit dot4, and HIP wave64 does not close the gap, so the remaining gap
   is fixed-shape codegen, surrounding scheduling, or layout/activation
   quantization economics.
8. **LLVM roadmap:** For each confirmed RADV/ACO win, what exactly would LLVM
   need to improve?

## Ground Rules

- Run HIP and Vulkan on the same hardware, power/clock state, model-shape
  constants, and input data.
- Prebuild HIP code and Vulkan pipelines outside the timed region. Do not time
  shader compilation, `hipcc`, pipeline creation, or descriptor setup unless the
  row is explicitly a startup/build benchmark.
- Every numerical kernel must compare against a CPU oracle or bitwise
  cross-backend reference before timing rows are retained.
- Each benchmark must report both "same algorithm/shape" and "best native shape"
  when possible. The first isolates compiler quality; the second measures the
  backend ceiling users actually care about.
- Record ISA/stat evidence, not only wall time: VGPR, SGPR, scratch/private
  memory, LDS, wave size, workgroup size, occupancy estimate, instruction mix,
  `v_dot4_i32_iu8` presence, VOPD presence, and waitcnt density.
- Treat all Vulkan/RADV diagnostics as platform-specific unless rerun on both
  W7900/gfx1100 and gfx1151.

## Measurement Harness Shape

Build one harness that can generate paired HIP and Vulkan kernels from the same
benchmark descriptor:

```json
{
  "bench": "q4k_gemv_decode",
  "backend": "hip|vulkan",
  "hardware": "gfx1100|gfx1151",
  "compiler": "llvm-amdgpu|radv-aco",
  "algorithm": "dequant_f32|q8_1_sudot4|lds_reduce|subgroup_reduce",
  "workgroup_size": 64,
  "wave_or_subgroup": 32,
  "rows": 4,
  "k": 2048,
  "n": 8192,
  "quant": "q4_k",
  "iters": 1000,
  "warmup": 100,
  "correctness": {
    "max_abs": 0.0,
    "kl": 0.0,
    "top1": 1.0
  },
  "timing": {
    "median_ns": 0,
    "p05_ns": 0,
    "p95_ns": 0
  },
  "isa": {
    "vgpr": 0,
    "sgpr": 0,
    "scratch_bytes": 0,
    "lds_bytes": 0,
    "wave_size": 0,
    "vopd_count": 0,
    "dot4_count": 0,
    "waitcnt_count": 0
  }
}
```

The first implementation can use separate HIP and Vulkan source templates. A
later version can auto-generate both from a small DSL, but that is not required
for the first attribution pass.

## Microbenchmark Matrix

### 1. Dispatch And Grid Floor

Purpose: separate command/runtime overhead from shader body speed.

| Bench | Variants | Attribution |
| --- | --- | --- |
| no-op kernel | 1, 16, 64, 256, 1024, 8192, 65536 blocks | Per-dispatch and grid-size scheduling cost |
| tiny ALU kernel | 2 args vs 16 args, same grid | Arg marshaling vs GPU dispatch |
| command burst | N kernels in one host call / command buffer | HIP launch loop vs Vulkan command-buffer replay |
| dependent chain | one block, fixed cycles | Compiler body scheduling without grid effects |

Retain this first. If Vulkan only wins here, then a Vulkan backend may help
launch-heavy paths, but LLVM kernel codegen is not the target.

Status: retained on gfx1151/STRIX_HALO. Repeat on gfx1100/W7900 before treating
the magnitude as cross-GPU, but do not spend more iteration time here until the
matched math kernels exist.

### 2. Geometry Sweep

Purpose: quantify dead lanes and subgroup shape effects independently of quant.

| Bench | Shapes | Variants |
| --- | --- | --- |
| f32 GEMV row | K=512, 2048, 8192 | workgroup 32/64/128/256; subgroup/wave 32/64 |
| reduction only | K=512, 2048, 8192 | LDS tree, shuffle/subgroup, one-wave |
| selected-MoE index gather | 8 selected experts, K=512 and 2048 | compact IDs vs scattered IDs |
| rows>1 verifier GEMV | rows=1/2/4/8 | `grid.y=rows` vs rowtile vs subgroup batch |

Attribution rule: if matched 64-thread HIP closes the Vulkan gap, this is a
kernel geometry issue, not ACO superiority. If Vulkan still wins at identical
geometry, inspect compiler statistics.

### 3. Memory Coalescing And Waitcnt

Purpose: isolate ACO/LLVM memory scheduling and wait-state differences.

| Bench | Variants | What To Inspect |
| --- | --- | --- |
| coalesced load+accumulate | vector widths 1/2/4/8 | GB/s, waitcnt density, VGPR |
| strided load+accumulate | stride 2/4/8/16 | cache behavior, waitcnt placement |
| gather IDs | random selected expert rows | scalar/vector address math |
| load-compute interleave | unroll 1/2/4/8/16 | whether compiler overlaps memory with ALU |

Attribution rule: same memory traffic but lower waitcnt density and higher
effective GB/s on Vulkan is an ACO scheduling win. Same waitcnt but better
coalescing is a source/layout issue.

Status: retained on gfx1151. Vulkan is `1.30x-2.35x` faster on most coalesced,
strided, and interleave rows, while gather is essentially tied at `1.02x`.
HIP wave64 controls do not close the gap and severely regress gather. This is
strong memory-side evidence but remains `diagnostic_unclassified` because
fixed-shape controls and official RADV register allocation counts are still
missing.

### 4. VOPD And VALU Scheduling

Purpose: determine whether RADV/ACO materially beats LLVM on RDNA3 dual-issue.

| Bench | Variants | Expected Read |
| --- | --- | --- |
| independent f32 FMA pairs | 2, 4, 8 accumulators | VOPD pairing opportunity |
| dependent f32 chain | no independent accumulators | VOPD should not help |
| dequant-like chain | shift/mask/sub/cvt/mul/fma | limited VOPD opportunity |
| mixed integer+float chain | address math plus fma | register-bank and scheduler stress |

Attribution rule: count VOPD instructions and compare elapsed cycles. ACO only
"wins on dual issue" if the Vulkan shader emits more useful VOPD or equivalent
dual-issue scheduling at matched occupancy and instruction count.

Status: retained on gfx1151. HIP emitted VOPD in all retained rows and RADV
emitted none, so the current retained evidence is a negative result for
"Vulkan wins because ACO finds VOPD that LLVM misses." Cross-GPU reruns can
check portability, but the next gfx1151 compiler tests should move to HIP
wave/specialization controls and real-slice confirmation.

### 5. Dot4 / q8_1 / sudot4

Purpose: determine whether the dot path is compiler-bound or layout-bound.

| Bench | Shapes | Variants |
| --- | --- | --- |
| q8_1 x q4 dot | K=512/2048/8192 | scalar dequant vs `sudot4` |
| q8_1 x q5/q6 selected-down | K=512/2048 | raw sidecar vs T16 decode-repack |
| q8_0 dense GEMV | rows=1/4/8 | exact f32 path vs q8_1 dot diagnostic |
| quantize activation | rows=1/4/8, K=2048/8192 | cost of q8_1 materialization |

Attribution rule: if both compilers emit `v_dot4_i32_iu8` and timing is close,
the next work is layout/quant economics, not LLVM. If HIP fails to emit the
expected dot instruction or surrounds it with worse waitcnt/spills, that is an
LLVM codegen target or a hand-ISA candidate.

Status: retained on gfx1151 for packed q8 signed, q4 unsigned-byte by signed-q8,
q6 zero-point correction, and scalar q4 dequant rows. HIP and RADV both emit
final dot4 instructions in q8/q4/q6 rows, and HIP reports no scratch/spills.
Vulkan remains `3.29x-3.43x` faster, including the scalar row, and HIP wave64
does not close the gap, so basic dot-instruction availability and wave mode are
no longer the main questions. The next dot work is fixed-shape control plus real
q8_1 materialization and layout economics.

### 6. LDS, Barriers, And Subgroup Reductions

Purpose: test the old HIP LDS/barrier reduction shape against Vulkan subgroup
reduction and HIP shuffle variants.

| Bench | Variants | Retain If |
| --- | --- | --- |
| one-row reduction | LDS tree, wave shuffle, subgroup reduce | Shows crossover by K |
| two-stage reduction | block partials + final reduce | Useful for large K |
| no-LDS accumulators | 4/8/16 accumulators per lane | Beats LDS without spills |
| barrier stress | same math with inserted barriers | Quantifies barrier tax |

Attribution rule: subgroup reduction wins are not automatically compiler wins.
They are usually algorithm/geometry wins unless matched HIP shuffle code still
lags with similar ISA quality.

### 7. Representative Inference Slices

Purpose: keep microbenchmarks tied to real hipEngine work.

| Slice | Current Importance | What To Compare |
| --- | --- | --- |
| small-K expert-down | known dead-lane risk | 64-thread subgroup/Vulkan vs HIP 64/128/256 |
| selected gate+up dual | hot MoE bucket | T16 exact, raw q8_1 dot, subgroup variants |
| dense q8_0 attention proj | hot AR/verify bucket | row=1 and rows=4/8 |
| GDN/recurrent chain | verifier bucket | memory scheduling and register pressure |
| q6 lm-head rowtile | shipped rowtile win | matched Vulkan large GEMV bandwidth |
| sampler/top-k/argmax | exposed server bucket | subgroup reductions and command fusion |

Attribution rule: no real-kernel lesson is retained unless a microbench result
predicts the direction and the actual slice confirms it.

## Result Classification

Each retained result should classify the win into exactly one primary bucket:

| Bucket | Definition | Likely Next Action |
| --- | --- | --- |
| `compiler_aco` | Same algorithm/layout/geometry, Vulkan faster with better ISA stats | File LLVM issue or try hand-ISA on that kernel |
| `geometry` | Matched HIP geometry closes most of the gap | Rewrite HIP kernel shape |
| `wave_mode` | HIP wave64 or subgroup-size control changes result materially | Add guarded wave experiment, then full correctness gate |
| `runtime_dispatch` | No-op/grid/command rows explain the gap | Consider Vulkan backend or fewer/larger kernels |
| `layout_quant` | Dot/layout changes dominate compiler choice | Port layout, not backend |
| `fusion_topology` | Per-op kernels match, fused Vulkan command wins | Fuse kernels or add backend-level composite |
| `not_reproducible` | Difference disappears under controlled harness | Do not roadmap work from the old observation |
| `diagnostic_unclassified` | Gap remains but evidence is insufficient for one primary cause | Add the missing control or ISA/stat evidence |

## LLVM Improvement Map

If a row lands in `compiler_aco`, map it to an LLVM-facing request:

| Symptom | Evidence Required | LLVM/HIP Improvement Target |
| --- | --- | --- |
| Under-unrolled loops | Same source, Vulkan fewer loop overhead instructions | gfx11 loop unroll heuristic for small GEMV/reduction loops |
| Excess waitcnt | Similar memory traffic, HIP has more waitcnt/nops and lower GB/s | waitcnt placement and memory scheduling |
| Register spill | HIP `Scratch_Size > 0`, Vulkan no spill at same live values | VGPR allocator / scheduling pressure reduction |
| Missed VOPD | Independent VALU pairs present, ACO emits VOPD, LLVM does not | VOPD pairing and register-bank aware scheduling |
| Bad wave64 codegen | HIP wave64 compiles but regresses from spills/incorrect reductions | wave64 lowering and shuffle/subgroup correctness |
| Missed dot pattern | HIP source expresses dot, no `v_dot4_i32_iu8` appears | intrinsic use or pattern matching for q8_1/q4 block dots |
| Poor immediate/address code | Same loads, HIP has more scalar/vector address ops | address-combine and scalarization passes |

Rows without ISA/stat evidence should not be used as LLVM asks. They can still
justify hipEngine kernel rewrites or a Vulkan backend.

## Vulkan Backend Effort Scale

There are two distinct scopes.

### Narrow Vulkan Probe Backend

Purpose: enough Vulkan compute infrastructure to run paired microbenchmarks and
one or two hot inference slices.

Effort class: **bounded probe**. This is a contained runtime/tooling project,
not a production backend. The value is high because it can falsify or confirm
the Vulkan ceiling without touching `hipengine.LLM.generate()`.

Work:

- Create a tiny Vulkan compute runtime: instance/device/queue selection,
  command pool, command buffer, fence/timeline synchronization, buffer
  allocation, descriptor sets, pipeline cache.
- Add shader build flow for GLSL or SPIR-V templates with specialization
  constants for K, rows, quant, and workgroup size.
- Add CPU-oracle validation and JSON result output compatible with benchmark
  artifacts.
- Add shader dump/disassembly collection where RADV exposes it; record exact
  Mesa/RADV/ACO version and environment.
- Keep it outside `hipengine.LLM.generate()` until correctness and packaging are
  understood.

This scope is the right first step because it answers whether Vulkan is worth
productizing without disturbing the current HIP backend.

The retained gfx1151 dispatch result strengthens the case for this probe, but
only for launch-heavy or command-fusion-sensitive paths. It is not sufficient
evidence for a production backend by itself.

### Production Vulkan Backend

Purpose: a real hipEngine backend registered as a peer of `hip_gfx1100`, likely
`vulkan_radv_gfx11`, with enough kernels to run GGUF/PARO decode.

Effort classes:

- **Large, narrow path** for a single-model/single-quant decode backend after
  the probe exists. This means real registry integration, real device memory
  ownership, and a small set of hot shader ports. It is still narrow enough to
  stay below "second full backend" scope.
- **Very large, systemic path** for a maintainable Vulkan backend with server
  integration, profiling, startup caching, multiple quant paths, and parity
  tests comparable to HIP. This should be treated as a second kernel stack
  unless the probe proves only a few shaders need Vulkan.

Major work items:

- Backend runtime abstraction: Vulkan device buffers, command submission,
  descriptor/pipeline caches, error reporting, and lifecycle management.
- Kernel registry integration using the existing `(backend, layer, quant,
  variant)` model. Do not add `if backend == "vulkan"` branches in engine code.
- Shader ports for the hot decode kernels: selected MoE gate/up/down, dense
  q8/q4 GEMVs, attention decode, GDN/recurrent pieces, lm-head/sample buckets,
  and KV writers.
- Persistent pipeline and specialization-cache policy keyed by GPU, Mesa/RADV,
  shader source hash, quant, and shape.
- Correctness gates against `kernels/cpu_reference/` or existing CPU oracles for
  each shader family.
- Benchmark integration for prefill/decode/server rows and RGP/RADV profiling
  capture.
- Packaging story for systems without RADV or without required subgroup/dot
  extensions.

Risks:

- Shader duplication can become a second kernel stack as large as HIP if not
  limited to proven hot kernels.
- Vulkan memory/descriptors are a different ABI from HIP pointers; zero-copy
  sharing is not the default design.
- Some wins may depend on RADV/ACO behavior and not general Vulkan.
- Debug/profiling loops are slower than HIP until the harness is mature.

Decision gate for starting production Vulkan: at least two real inference slices
must show retained Vulkan wins that are not reproduced by HIP geometry, compiler
flags, or small hand-ISA changes.

## Hand-ISA / Inline Assembly Candidates

Hand-ISA is narrower than a Vulkan backend. It is justified when a hot HIP
kernel is stable, isolated, and blocked by LLVM codegen rather than algorithm.

Good candidates:

- Inner q8_1/q4/q5/q6 dot loops where the desired `v_dot4_i32_iu8` sequence is
  known and LLVM emits extra work.
- Small-K selected-MoE kernels where ACO proves better waitcnt/register
  scheduling at identical geometry.
- F32/BF16 dequant chains only if a future microbench proves missed pairing or
  a specific instruction sequence matters after occupancy and memory traffic are
  controlled. Current gfx1151 VOPD rows do not provide that proof.
- Tiny sampler/top-k reductions if subgroup/VOPD scheduling is the only
  remaining exposed bucket.

Bad candidates:

- Launch overhead or command scheduling problems.
- Kernels whose performance changes mostly with workgroup geometry.
- Kernels that still have unresolved correctness drift or volatile ABIs.
- Whole-model assembly rewrites.
- Attention/GDN kernels where math/control complexity would make maintenance
  worse than the expected gain.

Implementation options, in increasing maintenance cost:

1. Use AMDGCN builtins such as `__builtin_amdgcn_sudot4` in normal HIP source.
2. Add small inline-asm blocks inside HIP kernels for a proven instruction
   sequence.
3. Build a standalone amdgcn assembly/HSACO kernel and load it through the HIP
   module path.

Promotion requirements:

- CPU-reference correctness gate.
- Disassembly proving the intended instruction sequence.
- `rocprofv3 --kernel-trace` proving the kernel ran.
- Same-suite non-regression in the real inference slice.
- A `docs/REFACTOR.md` entry for any diagnostic env flag.

## Initial Execution Order

1. Implement dispatch/grid floor rows for HIP and Vulkan. Status: retained on
   gfx1151; rerun on gfx1100/W7900 when that machine is available.
2. Implement geometry sweep for f32 GEMV/reduction at K=512/2048/8192. Status:
   retained on gfx1151; workgroup shape alone does not explain the gap in the
   repeat-shifted f32 GEMV/reduction harness.
3. Extract ISA/stat evidence for the retained f32 GEMV/reduction geometry
   kernel. Status: retained on gfx1151 for K=2048 rows=1 wg64/wg256; HIP emits
   VOPD, RADV does not, and the row remains `diagnostic_unclassified`.
4. Add dependent-chain and independent-accumulator VOPD microbenches with
   ISA/stat extraction. Status: retained on gfx1151; HIP emits VOPD and RADV
   does not in all retained rows, so VOPD pairing is not the current ACO
   explanation.
5. Add memory waitcnt microbenches: coalesced load+accumulate, strided
   load+accumulate, gather IDs, and load-compute interleave. Status: retained
   on gfx1151; Vulkan has a broad memory-side advantage except gather, but
   wave64 controls do not close it and RADV allocation-count/fixed-shape
   confounds keep it `diagnostic_unclassified`.
6. Add q8_1/sudot4 and scalar-dequant GEMV pairs. Status: retained on gfx1151
   for packed dot-path diagnostics; HIP and RADV both emit dot4 in q8/q4/q6
   rows, but Vulkan remains `3.29x-3.43x` faster. HIP wave64 does not close the
   gap, and the row is `diagnostic_unclassified`.
7. Port one real slice: selected-MoE small-K or q6 lm-head rowtile.
8. Classify each retained row using the result buckets above.
9. Only then decide between LLVM issue, HIP rewrite, hand-ISA, or production
   Vulkan backend.

The next most useful tests are now HIP fixed-shape controls for the retained
memory, geometry, and dot rows, plus one representative real slice.
The gfx1151 geometry, VOPD, memory/waitcnt, and dot-path extractions already
found that the current gap is not a missed-HIP-VOPD, HIP-spill, or missed-dot4
story. The next stop is isolating the remaining hypotheses rather than rerunning
broader geometry, generic VOPD, generic memory, or basic dot-lowering sweeps.

The expected useful output is not a single "Vulkan is faster" number. It is a
ranked list of deltas like: "Vulkan wins small-K expert-down by X%; Y% is
geometry, Z% is ACO waitcnt/VGPR quality, remaining is dispatch." That is the
level of evidence needed to guide LLVM work or justify a backend investment.
