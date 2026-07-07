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

The practical conclusion is:

- Stop broad gfx1151 attribution sweeps for now. Dispatch, f32 geometry,
  VOPD, memory/waitcnt, dot-path, HIP wave64, HIP fixed-shape, HIP q8_1
  real-slice layout, RADV shaderstats, and one matched Vulkan real slice are
  already retained.
- The only useful near-term benchmark still missing is a Vulkan Q4_K
  selected-dual gate/up real-slice probe, and only if we want a second
  production hot-bucket transfer check after the negative Q6_K X8 result.
- Do not start a production Vulkan backend from the current evidence. The
  dispatch win is real, but it is a runtime result; the first production-shaped
  Vulkan q6 slice is slower than HIP.
- Do not start a hand-ISA path from the current generic VOPD or packed-dot
  evidence. HIP already emits VOPD in the retained VOPD rows and dot4 in the
  retained q8/q4/q6 rows. Hand-ISA only becomes reasonable for a specific hot
  HIP slice with a proven LLVM codegen miss after matched real-slice controls.

The detailed retained reads are:

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
  device-memory load+accumulate rows, Vulkan is `1.02x-2.25x` faster for most
  coalesced, strided, and interleave variants; gather is essentially tied
  (`1.02x`). HIP and RADV both report no scratch and no spills in the refreshed
  shaderstats artifacts. Simple rows show slightly fewer RADV waitcnt-family
  instructions; fixed-shape controls do not close the gap, but the row still
  needs wave-shape and memory-bound real-slice confirmation before becoming an
  LLVM-AMDGPU waitcnt/scheduling claim.
- The retained packed dot-path sweep rules out a missing-HIP-dot-instruction
  story for the current q8/q4/q6 idiom. HIP and RADV both emit final dot4
  instructions for q8 signed, q4 unsigned-byte by signed-q8, and q6 zero-point
  correction rows; HIP reports no scratch/spills. Vulkan is still `3.28x-3.42x`
  faster, including the scalar-dequant row (`3.30x`). After the HIP wave64 and
  fixed-block controls, the remaining dot-path gap is more likely surrounding
  scheduling or layout/activation quantization economics than basic dot
  lowering, wave mode, or runtime block indexing.
- The first retained HIP wave64 controls do not close the gap. On packed-dot
  rows, forcing HIP wave64 makes HIP `1.007x-1.061x` slower than the retained
  wave32 HIP rows. On memory/waitcnt rows, HIP wave64 is mixed but still leaves
  Vulkan faster on most shapes; gather regresses `6.35x` versus HIP wave32.
- The retained HIP fixed-shape controls also do not close the gap. Dot-path
  fixed block indexing is flat versus same-commit runtime HIP
  (`0.993x-1.000x` fixed/runtime) and Vulkan remains `3.31x-3.43x` faster.
  Memory fixed block indexing is mixed (`0.906x-1.290x` fixed/runtime) and
  Vulkan remains faster on every retained row (`1.04x-2.36x`). Fixed-workgroup
  geometry improves some HIP wg256 rows by up to `6.3%`, but Vulkan still leads
  best-native geometry by `5.56x-14.03x`.
- The first retained HIP real-slice q8_1 layout controls are positive for HIP:
  Q4_K selected-dual gate/up q8_1 quantize+dp4a is `2.77x` faster than the raw
  selected-dual path, and Q6_K selected-down X8 q8_1 quantize+dp4a is `1.68x`
  faster than the production T16 float path. q8_1 materialization itself is
  small (`0.0025-0.0027 ms`) in these retained slices. This is **not** a Vulkan
  real-slice comparison; it only says q8_1 layout cost is not the blocker on
  the HIP side.
- The first matched Vulkan production-slice probe does **not** transfer the
  synthetic Vulkan dot-path win to the Q6_K selected-down X8 hot shape. On the
  retained rows=8, experts=256, in=512, out=2048 slice, the best Vulkan
  local-size row is local_size=64 at `0.03076 ms` prequantized dot and
  `0.03217 ms` quantize+dot. The retained HIP path is `0.01665 ms` dot and
  `0.01925 ms` quantize+dot, so Vulkan is `1.85x` slower on dot and `1.67x`
  slower on the combined path. This is retained as `real_slice_probe`, not
  `compiler_aco`.
- We still cannot claim `compiler_aco` for the f32 geometry gap. The refreshed
  RADV shaderstats extraction reports no Vulkan scratch/spills and official
  `12` VGPR / `108` SGPR allocation for the retained wg64/wg256 shaders, while
  HIP reports `11` VGPR / `18` SGPR and no scratch/spills. That removes the
  missing-allocation-stat blocker, but the f32 geometry row is still
  `diagnostic_unclassified` because wave/subgroup shape and source/runtime
  structure remain confounds.

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
- Do not treat HIP runtime `blockDim`/block-indexing overhead as the missing
  switch for retained dot, memory, or f32 geometry gaps. Fixed-shape HIP
  controls are flat or small relative to the remaining Vulkan lead.
- Do not reject q8_1/dp4a purely because of activation materialization cost in
  the tested HIP selected-MoE slices. The measured quantization cost is small
  and the full quantize+dot path is faster than the retained HIP float controls.
- Do not pursue a Vulkan port for the Q6_K selected-down X8 q8_1+dp4a slice as
  currently implemented. The matched Vulkan real-slice probe is slower than HIP
  despite passing correctness and using the intended SPIR-V/RADV dot path.

What remains plausible:

- Vulkan has a proven dispatch/runtime advantage on gfx1151.
- Vulkan has a large matched-math advantage in one f32 diagnostic, but that row
  is still `diagnostic_unclassified`.
- Memory/access scheduling is still the strongest compiler-facing lead, but HIP
  wave64 and fixed block indexing are not the fix. Refreshed RADV shaderstats
  remove the allocation-count visibility blocker and show no Vulkan
  scratch/spills, so the remaining blockers are wave/subgroup shape and
  memory-bound real-slice confirmation.
- Dot-instruction availability is no longer untested for the packed q8/q4/q6
  idiom: HIP emits dot4. HIP wave64 also does not close the dot gap. The
  fixed-block control also does not close it. HIP real-slice q8_1 materialization
  is positive, and the first Vulkan real-slice probe is negative for Q6_K X8
  selected-down. A narrow hand-ISA sequence is only worth considering for a
  specific hot slice after the remaining layout/stat confounds are gone.
- Real Vulkan inference slices still need caution: the first production-shaped
  q6 selected-down probe did **not** predict a Vulkan win, so synthetic Vulkan
  dot-path wins should not be promoted without a matching real slice.
- The most actionable HIP-side work from this suite is not "switch to Vulkan";
  it is to preserve the q8_1/dp4a real-slice wins, keep reducing launches and
  fusion boundaries, and isolate memory/waitcnt behavior inside shipped hot
  kernels before filing LLVM work.

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

There is still one useful near-term test, but the broad attribution phase is
done for gfx1151. Do **not** spend more gfx1151 time on dispatch-only, broad
geometry-only, generic VOPD, generic memory, basic dot-lowering, HIP wave64, or
HIP fixed-block sweeps. Those have already answered the coarse questions. The
remaining useful work is decision-grade only:

Immediate triage:

| Decision | Area | Reason |
| --- | --- | --- |
| Stop for now | Generic VOPD/dual-issue sweeps | Retained gfx1151 rows show HIP emits VOPD and RADV does not; this is not the current ACO win. |
| Stop for now | More HIP wave64 dot/memory rows | Wave64 did not close dot or memory gaps and badly regressed gather. |
| Stop for now | More HIP fixed-block dot/memory controls | Fixed block indexing was flat or mixed and did not close the Vulkan lead. |
| Stop for now | Dispatch-only Vulkan probes | The runtime-dispatch win is already classified and does not prove compiler quality. |
| Stop for now | Vulkan Q6_K selected-down X8 port | The matched real-slice probe is slower than retained HIP by `1.67x` on quantize+dot. |
| Done | Better RADV allocation/stat extraction | `RADV_DEBUG=shaders,shaderstats` now gives official RADV VGPR/SGPR/spill/scratch data for retained Vulkan rows. |
| Test next, if one more real-slice check is needed | Vulkan Q4_K selected-dual gate/up slice | Useful as the only remaining production hot-bucket transfer check after the negative q6 result. |
| Optional | HIP wave64 plus fixed-workgroup geometry | Only needed if we want to remove the remaining f32 geometry wave/subgroup confound. |

Recommended next tests, in order:

1. Vulkan Q4_K selected-dual gate/up real-slice probe, only if we want a second
   production hot-bucket transfer check. The Q6_K selected-down Vulkan probe is
   already negative, so a Q4_K win would need to be large and clean before
   changing the backend roadmap.
2. HIP wave64 plus fixed-workgroup geometry control, only if we want to remove
   the remaining wave-mode confound before making any compiler-facing f32
   geometry claim.
3. Cross-GPU reruns after the harnesses above stabilize. gfx1100/W7900 and
   7900 XTX reruns should check portability of a classified diagnosis, not
   replace the Q4_K real-slice check.

Priority summary:

| Priority | Test | Decision It Enables |
| --- | --- | --- |
| Done | Dot-path q8/q4/q6 kernels with dot ISA counts | HIP emits dot4; remaining gap is not basic dot lowering |
| Done | HIP wave64 dot/memory controls | Wave64 does not close dot/memory gaps and regresses gather |
| Done | HIP fixed-shape memory/dot/geometry controls | Runtime `blockDim`/fixed-shape overhead does not close retained gaps |
| Done | HIP q8_1 real-slice layout controls | q8_1 materialization is small and positive on tested HIP selected-MoE slices |
| Done | Vulkan Q6_K selected-down X8 real-slice probe | Synthetic Vulkan dot-path win does not transfer to this shipped hot bucket |
| Done | RADV shaderstats allocation extraction | Official RADV allocation counts show no Vulkan scratch/spills in retained rows |
| P1 | Vulkan Q4_K selected-dual gate/up slice | Only remaining near-term production hot-bucket transfer check |
| P2 | HIP wave64 plus fixed-workgroup geometry | Remove the remaining f32 geometry wave/subgroup confound if needed |
| P2 | gfx1100/W7900 and 7900 XTX reruns | Check portability after fixed-shape and real-slice harnesses are classified on gfx1151 |

Tooling that would improve attribution quality, but should not displace the
remaining real-slice work:

- A small comparison utility that rolls HIP/Vulkan artifacts into a one-page
  retained-result diff with timing, correctness, wave mode, instruction counts,
  waits, dot/VOPD counts, and classification.

Cross-GPU reruns on gfx1100/W7900 and 7900 XTX are important after the harnesses
are stable. They should confirm portability of the gfx1151 conclusions, not
replace the remaining Q4_K real-slice transfer check.

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
- Vulkan evidence: `RADV_DEBUG=shaders,shaderstats` final disassembly, ACO
  after-RA presence, and RADV shaderstats allocation counts. Estimated physical
  register spans are retained as a cross-check, but the primary RADV VGPR/SGPR
  rows below are official shaderstats allocation counts.
- Classification: `diagnostic_unclassified`.

Retained ISA/stat summary:

| Workgroup | HIP ISA | Vulkan/RADV ISA |
| ---: | --- | --- |
| 64 | actual `18` SGPR, `11` VGPR, scratch `0`, spills `0`, wave32, `118` static instructions, `6` waitcnt-family instructions, `2` VOPD instructions / `4` VOPD ops | official `108` SGPR, `12` VGPR, scratch `0`, spills `0`, estimated span `16` SGPR / `9` VGPR, wave64, `100` static instructions, `9` waitcnt-family instructions, `0` VOPD |
| 256 | actual `18` SGPR, `11` VGPR, scratch `0`, spills `0`, wave32, `118` static instructions, `6` waitcnt-family instructions, `2` VOPD instructions / `4` VOPD ops | official `108` SGPR, `12` VGPR, scratch `0`, spills `0`, estimated span `16` SGPR / `9` VGPR, wave64, `142` static instructions, `20` waitcnt-family instructions including `9` depctr waits, `0` VOPD |

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
- Do not claim RADV has better register allocation from this artifact. Official
  RADV shaderstats now show no Vulkan scratch/spills, but they do not explain
  the geometry timing gap or prove better allocation than HIP.
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
- RADV shaderstats reports official `12/12/24` VGPR for independent `2/4/8`,
  official `12/24/24` VGPR for dependent/mixed/dequant rows, `108` SGPR in all
  rows, and `0` scratch/spills in all rows.
- RADV emits wave64 final shaders in these rows; HIP emits wave32 code objects.
- Both backends pass the sampled CPU oracle with max abs `2.384185791e-07`.

Conclusion: do **not** attribute the current Vulkan ceiling to RADV/ACO finding
better VOPD pairing than LLVM/HIP. In this targeted family, LLVM/HIP is the
backend emitting VOPD. Vulkan's modest wins on independent-8, mixed int+float,
and dequant-like rows must come from something else: wave64 execution shape,
non-VOPD scheduling, instruction selection, runtime/pipeline effects, or
measurement noise. The next relevant compiler tests are real-slice confirmation
and memory-bound production transfer checks, not more generic VOPD speculation.

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
  real-slice transfer remain confounds.

Retained timing and ISA summary:

| Variant | HIP median | Vulkan median | Vulkan vs HIP | HIP GB/s | Vulkan GB/s | HIP wait/load | RADV wait/load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coalesced width 1 | `26.6127 us` | `18.9185 us` | `1.41x` | `630.42` | `887.02` | `4/1` | `3/1` |
| coalesced width 2 | `44.6948 us` | `32.0182 us` | `1.40x` | `750.75` | `1048.01` | `5/2` | `4/2` |
| coalesced width 4 | `282.1829 us` | `125.2440 us` | `2.25x` | `237.82` | `535.82` | `7/4` | `6/4` |
| coalesced width 8 | `639.5150 us` | `303.7756 us` | `2.11x` | `209.87` | `441.94` | `11/8` | `10/8` |
| strided stride 2 | `44.5751 us` | `34.3159 us` | `1.30x` | `376.38` | `488.92` | `4/1` | `3/1` |
| strided stride 4 | `281.3058 us` | `146.2196 us` | `1.92x` | `59.64` | `114.75` | `4/1` | `3/1` |
| strided stride 8 | `557.0604 us` | `249.6609 us` | `2.23x` | `30.12` | `67.19` | `4/1` | `3/1` |
| strided stride 16 | `1144.5523 us` | `576.7278 us` | `1.98x` | `14.66` | `29.09` | `4/1` | `3/1` |
| gather IDs | `493.0500 us` | `484.7561 us` | `1.02x` | `68.05` | `69.22` | `5/2` | `4/2` |
| interleave unroll 1 | `25.8722 us` | `15.8026 us` | `1.64x` | `648.47` | `1061.67` | `4/1` | `3/1` |
| interleave unroll 2 | `49.8455 us` | `40.9710 us` | `1.22x` | `673.17` | `818.98` | `5/2` | `4/2` |
| interleave unroll 4 | `281.1671 us` | `135.2524 us` | `2.08x` | `238.68` | `496.19` | `6/4` | `6/4` |
| interleave unroll 8 | `580.6280 us` | `289.8444 us` | `2.00x` | `231.16` | `463.06` | `10/8` | `10/8` |
| interleave unroll 16 | `1611.1334 us` | `1452.4936 us` | `1.11x` | `166.61` | `184.80` | `13/16` | `18/16` |

Register/stat summary:

- All HIP and Vulkan rows pass the sampled CPU oracle with max abs `0.0`.
- HIP reports wave32, no scratch, and no spills in all retained rows. HIP VGPR
  rises with interleave width from `8` at unroll 1 to `36` at unroll 16.
- RADV final shaders are wave64. RADV shaderstats reports official VGPR
  allocation of `12` for simple coalesced/strided/gather rows, `12/24/24/48/48`
  for interleave `1/2/4/8/16`, `108` SGPR in all rows, and `0`
  scratch/spills in all rows. Estimated physical register spans remain in the
  artifacts as cross-checks.
- Simple coalesced, strided, gather, and low-unroll interleave rows show one
  fewer RADV waitcnt-family instruction than HIP at the same static load count.
  Wider interleave rows have equal or higher RADV waitcnt counts, so waitcnt
  count alone is not the whole story.

Conclusion: memory/access scheduling is now a serious candidate for the Vulkan
ceiling. Vulkan is consistently faster on coalesced, strided, and most
interleave rows, while gather is essentially tied. This does **not** yet justify
a clean LLVM `compiler_aco` issue because the retained rows still compare HIP
wave32 against RADV wave64 and do not yet show the same effect inside a
production memory-bound real slice. The fixed-block control below does not
close the memory gap; the dot-path result below separately shows basic
q8/q4/q6 dot lowering is present in HIP.

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
| q8 signed | `7114.77 us` | `2079.47 us` | `3.42x` | `16` | `16` | `OpSDot=1` | `19/32` | `18/32` |
| q4 unsigned x q8 | `7109.06 us` | `2088.64 us` | `3.40x` | `16` | `16` | `OpSUDot=1` | `19/32` | `18/32` |
| q6 zero-corrected | `6831.89 us` | `2082.13 us` | `3.28x` | `32` | `32` | `OpSUDot=2` | `20/32` | `18/32` |
| scalar q4 dequant | `7342.70 us` | `2223.05 us` | `3.30x` | `0` | `0` | none | `35/32` | `20/32` |

Register/stat summary:

- All HIP and Vulkan rows pass the exact sampled CPU oracle with max abs `0.0`.
- HIP reports wave32, no scratch, and no spills. HIP dot rows use
  `41-42` VGPR and `14` SGPR; scalar dequant uses `50` VGPR.
- RADV final shaders are wave64. RADV shaderstats reports official `36` VGPR
  for q8/q4, `48` VGPR for q6/scalar, `108` SGPR in all rows, and `0`
  scratch/spills in all rows. Estimated physical register spans remain in the
  artifact as a cross-check.
- HIP and RADV emit the same final dot4 counts in q8/q4/q6 rows. Vulkan SPIR-V
  also contains the expected `OpSDot`/`OpSUDot` operations before RADV lowering.
- The scalar row is also `3.30x` faster on Vulkan despite using no dot4, so the
  retained gap cannot be explained only by dot-instruction selection.

Conclusion: do **not** spend more gfx1151 time proving whether HIP can emit the
basic q8/q4/q6 dot instruction; it can. The retained dot-path gap is still
large, but the fixed-block control below shows runtime block indexing is not
the missing switch. HIP q8_1 real-slice layout controls are positive, and the
first matched Vulkan Q6_K X8 real slice is negative. The only remaining
dot-related production transfer check worth running on gfx1151 is Q4_K
selected-dual gate/up. A hand-ISA path is not justified by this artifact alone;
it would need to beat the same HIP dot body after wave/fixed-shape controls and
then move a shipped selected-MoE or q6 lm-head slice.

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
dot gap, and the memory result is mixed with a severe gather regression. The
later fixed-shape controls also fail to close these gaps; do not promote broad
HIP wave64 routing from this evidence.

### gfx1151 HIP Fixed-Shape Controls

Retained artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/dot-path-fixed-block-comparison.json`,
`benchmarks/micro/results/gfx1151/strix-halo/memory-waitcnt-fixed-block-comparison.json`,
and
`benchmarks/micro/results/gfx1151/strix-halo/geometry-sweep-fixed-workgroup-comparison.json`.
Backend artifacts include same-commit runtime HIP controls, fixed HIP controls,
and same-commit Vulkan controls:
`hip-dot-path-runtime-control.json`, `hip-dot-path-fixed-block.json`,
`vulkan-dot-path-fixed-control.json`,
`hip-memory-waitcnt-runtime-control.json`,
`hip-memory-waitcnt-fixed-block.json`,
`vulkan-memory-waitcnt-fixed-control.json`,
`hip-geometry-sweep-runtime-control.json`,
`hip-geometry-sweep-fixed-workgroup.json`, and
`vulkan-geometry-sweep-fixed-control.json`. The run uses
`environment-fixed-shape-controls.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- HIP dot/memory control: `--hip-fixed-block-index`, which compiles
  `__launch_bounds__(256)` and `kBlockSize` global indexing instead of
  `blockDim.x`.
- HIP geometry control: `--hip-workgroup-specialization fixed`, which compiles
  one HIP binary per requested workgroup size and replaces runtime
  `blockDim.x` in the reduction/indexing path.
- Classification: `diagnostic_unclassified`; this is a runtime-shape control,
  not final compiler attribution.

Dot fixed-block result:

| Variant | Fixed / runtime HIP | Same-commit Vulkan vs fixed HIP |
| --- | ---: | ---: |
| q8 signed | `1.000x` | `3.43x` |
| q4 unsigned x q8 | `1.000x` | `3.42x` |
| q6 zero-corrected | `0.997x` | `3.31x` |
| scalar q4 dequant | `0.993x` | `3.31x` |

Memory fixed-block result:

| Group | Fixed / runtime HIP | Same-commit Vulkan vs fixed HIP |
| --- | --- | --- |
| coalesced | width 1 slower `1.063x`; widths 2/4 flat; width 8 faster `0.906x` | Vulkan `1.47x-2.28x` faster |
| strided | `1.004x-1.026x` slower | Vulkan `1.35x-2.36x` faster |
| gather | `1.290x` slower | Vulkan `1.36x` faster |
| interleave | mixed: unroll 1/16 faster `0.913x/0.930x`, unroll 2 slower `1.120x`, others flat | Vulkan `1.04x-2.29x` faster |

Geometry fixed-workgroup result:

| Shape group | Fixed / runtime HIP at best HIP wg256 | Same-commit Vulkan vs fixed HIP best-native |
| --- | ---: | ---: |
| K=512 rows=1/4/8 | `0.985x-0.987x` | `5.56x-7.60x` |
| K=2048 rows=1/4/8 | `0.937x-0.999x` | `9.79x-11.56x` |
| K=8192 rows=1/4/8 | `0.976x-0.991x` | `12.07x-14.03x` |

Conclusion: HIP runtime `blockDim`/shape specialization is not the missing
switch for the retained gfx1151 gaps. Fixed geometry gives a useful small HIP
improvement, especially K=2048 rows=1 wg256, but the Vulkan geometry lead
remains large. Dot fixed-block is flat, and memory fixed-block is mixed with a
gather regression. Treat memory/access scheduling and geometry as still
unclassified until real-slice and wave/subgroup confounds are resolved.

### gfx1151 HIP q8_1 Real-Slice Layout Controls

Retained artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-real-q4-selected-dual-q8_1-dp4a.json`
and
`benchmarks/micro/results/gfx1151/strix-halo/hip-real-q6-selected-down-x8-q8_1-dp4a.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-real-slice-q8_1.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics`, `gfx1151`.
- Backend: HIP only. These controls do **not** include a Vulkan shader for the
  same production slice.
- Slice 1: Q4_K selected-dual gate/up, `x_rows=4`, selected rows=`32`,
  experts=`256`, in=`2048`, out=`512`, threads=`256`.
- Slice 2: Q6_K selected-down, selected rows=`8`, experts=`256`, in=`512`,
  out=`2048`, production T16 float versus X8 q8_1 dp4a.
- Classification: `layout_quant` for HIP-side economics; not a Vulkan compiler
  attribution.

Q4_K selected-dual gate/up result:

| Metric | Result |
| --- | ---: |
| Raw selected-dual | `0.9584 ms` |
| q8_1 quantize | `0.00247 ms` |
| q8_1 dp4a dot, prequantized | `0.3464 ms` |
| q8_1 quantize+dp4a dot | `0.3458 ms` |
| Raw / quantize+dot | `2.77x` |
| Correctness vs raw | KL mean `0.00311`, max abs `2.0`, top-1 `1.0` for both gate and up |

Q6_K selected-down result:

| Metric | Result |
| --- | ---: |
| Production T16 float | `0.03228 ms` |
| q8_1 quantize | `0.00275 ms` |
| X8 q8_1 dp4a dot, prequantized | `0.01665 ms` |
| X8 q8_1 quantize+dp4a dot | `0.01925 ms` |
| Production T16 / quantize+dot | `1.68x` |
| Raw float / quantize+dot | `2.38x` |
| Correctness vs production T16 | KL mean `0.00137`, max abs `0.5`, top-1 `1.0` |
| X8 vs raw dp4a | max abs `1.91e-6`, top-1 `1.0` |

Conclusion: q8_1 materialization cost is not the reason to avoid q8_1/dp4a on
these HIP selected-MoE slices. The quantization stage is only about
`0.0025-0.0027 ms` and the full quantize+dot path is faster than the tested HIP
float controls. The matched Vulkan Q6_K X8 selected-down probe below shows that
this positive HIP layout result does **not** automatically transfer to a Vulkan
backend.

### gfx1151 Vulkan Q6_K X8 Real-Slice Probe

Retained artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q6-selected-down-x8-q8_1-dp4a.json`,
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q6-selected-down-x8-q8_1-dp4a-ls128.json`,
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q6-selected-down-x8-q8_1-dp4a-ls256.json`,
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q6-selected-down-x8-q8_1-dp4a-isa-stats.json`,
and
`benchmarks/micro/results/gfx1151/strix-halo/q6-x8-real-slice-hip-vulkan-comparison.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Shape: Q6_K selected-down X8, rows=`8`, experts=`256`, in=`512`,
  out=`2048`, q8_1 activation format, bf16 output.
- Vulkan implementation: standalone microbench probe with a q8_1 quantize
  shader plus Q6_K X8 selected-down dp4a shader. Timings use pre-recorded
  command buffers and exclude shader compilation, pipeline creation, and
  host-device transfers.
- Correctness: full CPU reference for q8_1 quantize plus Q6_K X8 selected-down
  dp4a. All retained local-size rows pass with top-1 `1.0`; best local_size=64
  max abs is `0.25`, mean abs `0.00308`.
- Classification: `real_slice_probe`; this is production-shaped backend
  evidence, not a generic compiler proof.

Timing result:

| Backend / local size | q8_1 quantize | X8 dot, prequantized | q8_1 quantize+X8 dot | Correctness |
| --- | ---: | ---: | ---: | --- |
| HIP retained, threads=64 | `0.00275 ms` | `0.01665 ms` | `0.01925 ms` | top-1 `1.0` vs production T16 |
| Vulkan local_size=64 | `0.000357 ms` | `0.03076 ms` | `0.03217 ms` | top-1 `1.0` vs CPU |
| Vulkan local_size=128 | `0.000356 ms` | `0.03258 ms` | `0.03326 ms` | top-1 `1.0` vs CPU |
| Vulkan local_size=256 | `0.000356 ms` | `0.03500 ms` | `0.03707 ms` | top-1 `1.0` vs CPU |

Best retained Vulkan row is local_size=64. Relative to retained HIP, Vulkan is:

| Metric | Vulkan / HIP |
| --- | ---: |
| X8 dot, prequantized | `1.85x` slower |
| q8_1 quantize+X8 dot | `1.67x` slower |

ISA/stat extraction for local_size=64:

| Shader | SPIR-V dot ops | RADV final dot4 | RADV subgroup | RADV VOPD | RADV registers | Wait/load notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| q8_1 quantize | `0` | `0` | `64` | `0` | official `96` VGPR / `108` SGPR, no scratch/spills; span `96/16` | `6` waitcnt, `1` buffer load |
| Q6_K X8 dot | `9` `OpSUDot` | `9` | `64` | `0` | official `48` VGPR / `108` SGPR, no scratch/spills; span `48/24` | `89` waitcnt, `82` buffer loads |

Conclusion: the first matched Vulkan production slice is negative. The
synthetic packed dot-path sweep showed Vulkan can be much faster on a simplified
dot loop, but this Q6_K selected-down X8 production-shaped shader does not beat
HIP once the real layout, q8_1 materialization, selected rows, output shape, and
subgroup reduction are present. Do not promote a Vulkan backend or hand-ISA
path from this q6 selected-down evidence. The remaining compiler-facing work is
now optional transfer checks for other hot slices such as Q4_K selected-dual
gate/up, plus a HIP wave64/fixed-workgroup geometry control only if we need to
remove the remaining f32 geometry wave/subgroup confound.

## Questions To Answer

1. **Compiler scheduling:** When the algorithm, data layout, wave/subgroup size,
   and workgroup geometry are matched, does RADV/ACO still beat
   LLVM-AMDGPU? If yes, is the delta visible as fewer VGPRs, less scratch, fewer
   `s_waitcnt`, better unroll, or more VOPD pairing?
2. **Geometry:** How much of the Vulkan win comes from 64-thread subgroup
   shapes versus the common HIP 128/256-thread block shapes?
3. **Wave mode:** HIP wave64 did not close retained dot/memory gaps; a
   fixed-workgroup wave64 geometry row remains optional if we need to remove the
   last wave-mode confound before filing compiler work.
4. **Dispatch/runtime:** Is Vulkan faster because individual shaders are faster,
   or because command-buffer/pipeline execution reduces per-dispatch cost?
5. **Memory scheduling:** Retained gfx1151 rows show higher Vulkan bandwidth on
   coalesced, strided, and interleave loops. This survives fixed-block controls;
   the remaining question is whether it predicts quantized GEMV inner loops.
6. **VOPD portability:** gfx1151 retained evidence is negative for "ACO finds
   VOPD that LLVM misses." Do gfx1100/W7900 and 7900 XTX reproduce that answer,
   or is this driver/GPU-specific?
7. **dp4a/sudot4:** Does the compiler matter once the code uses the intended
   RDNA3 dot instruction? Retained gfx1151 packed-dot rows say HIP and RADV
   both emit dot4, while HIP wave64 and fixed-block indexing do not close the
   gap. The first Q6_K X8 real-slice Vulkan probe also emits dot4 but is slower
   than HIP, so synthetic dot-path wins do not transfer automatically.
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

Status: retained on gfx1151. Vulkan is `1.02x-2.25x` faster on most coalesced,
strided, and interleave rows, while gather is essentially tied at `1.02x`.
HIP wave64 and fixed-block controls do not close the gap; both severely regress
gather in their retained runs. This is strong memory-side evidence but remains
`diagnostic_unclassified` because wave/subgroup shape and memory-bound
production-slice confirmation are still unresolved. RADV shaderstats now gives
official allocation counts and shows no Vulkan scratch/spills.

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
check portability, but the next gfx1151 compiler tests should move to real-slice
confirmation, not more generic VOPD sweeps.

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
Vulkan remains `3.28x-3.42x` faster, including the scalar row, and HIP wave64
and fixed-block controls do not close the gap, so basic dot-instruction
availability, wave mode, and runtime block indexing are no longer the main
questions. HIP real-slice q8_1 materialization/layout controls are retained and
positive; the Q6_K X8 Vulkan real-slice check is retained and negative. The
only remaining dot-layout transfer check is Q4_K selected-dual gate/up.

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

Status: retained on gfx1151 for HIP q8_1 layout economics on Q4_K selected-dual
gate/up and Q6_K selected-down X8. The retained HIP slices confirm q8_1
materialization cost is small and the full q8_1+dp4a path is faster than the
tested HIP float controls. The matched Vulkan Q6_K X8 selected-down production
slice is retained and negative; the Q4_K selected-dual gate/up Vulkan slice is
the only remaining near-term production-slice transfer check.

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

There are two distinct scopes. The retained evidence supports the first scope
as an attribution tool. It does **not** yet support the second scope as product
work.

### Narrow Vulkan Probe Backend

Purpose: enough Vulkan compute infrastructure to run paired microbenchmarks and
one or two hot inference slices.

Effort class: **bounded probe**. This is a contained runtime/tooling project,
not a production backend. The value is high because it can falsify or confirm
the Vulkan ceiling without touching `hipengine.LLM.generate()`.

Current status: **partially implemented and already useful**. The retained
dispatch, geometry, memory/waitcnt, VOPD, dot-path, shaderstats, and Q6_K X8
real-slice rows all came from this style of standalone Vulkan probe. The only
near-term extension still worth doing is Q4_K selected-dual gate/up if we want
one more production hot-bucket transfer check.

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

Decision gate for continuing the probe: add only tests that can change an
engineering decision. More generic dispatch/VOPD/dot/memory rows should stop;
Q4_K selected-dual is still useful because it tests whether the negative Q6_K
real-slice result is slice-specific or general.

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

Current decision: **do not start production Vulkan yet**. The retained
dispatch-row win is a runtime result, and the first matched production-shaped
Q6_K X8 Vulkan slice is slower than HIP. Production Vulkan should wait for at
least two retained real inference slices showing Vulkan wins that are not
reproduced by HIP geometry, compiler flags, launch fusion, or small hand-ISA
changes.

## Hand-ISA / Inline Assembly Candidates

Hand-ISA is narrower than a Vulkan backend. It is justified when a hot HIP
kernel is stable, isolated, and blocked by LLVM codegen rather than algorithm.

Current decision: **no broad hand-ISA path yet**. The retained gfx1151 evidence
does not show LLVM missing VOPD or dot4 in the generic diagnostics. HIP emits
VOPD in the retained VOPD rows, HIP emits dot4 in the retained q8/q4/q6
dot-path rows, and the first matched Vulkan Q6_K X8 real slice is slower than
HIP. A hand-ISA candidate must therefore come from a specific hot HIP slice with
a measured codegen problem, not from the generic Vulkan ceiling observation.

Good candidates:

- Inner q8_1/q4/q5/q6 dot loops only where the desired `v_dot4_i32_iu8`
  sequence is known, HIP's final ISA contains avoidable surrounding work, and a
  real slice proves that fixing it moves wall time.
- Small-K selected-MoE kernels only where ACO proves better waitcnt/register
  scheduling at identical geometry and the same production slice is faster on
  Vulkan.
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

Do not use hand-ISA to address launch overhead, backend command replay, or
generic geometry gaps. Those need launch fusion, backend scheduling, or HIP
kernel-shape work, not inline assembly.

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
   wave64 and fixed-block controls do not close it. RADV shaderstats now shows
   no Vulkan scratch/spills, so wave/subgroup shape and real-slice transfer
   keep it `diagnostic_unclassified`.
6. Add q8_1/sudot4 and scalar-dequant GEMV pairs. Status: retained on gfx1151
   for packed dot-path diagnostics; HIP and RADV both emit dot4 in q8/q4/q6
   rows, but Vulkan remains `3.28x-3.42x` faster. HIP wave64 and fixed-block
   controls do not close the gap, and the row is `diagnostic_unclassified`.
7. Add HIP fixed-shape controls for dot, memory, and geometry. Status:
   retained on gfx1151; runtime block indexing/workgroup specialization is not
   the missing switch.
8. Port HIP real-slice q8_1 layout controls: selected-MoE small-K and q6
   selected-down. Status: retained on gfx1151; q8_1 materialization is small
   and production-layout HIP q8_1+dp4a is positive.
9. Port one matched Vulkan real slice: selected-MoE small-K or q6 lm-head
   rowtile. Status: retained for Q6_K selected-down X8 on gfx1151; Vulkan is
   slower than HIP by `1.67x` on quantize+dot, so this slice is not a Vulkan
   backend candidate as implemented.
10. Classify each retained row using the result buckets above.
11. Only then decide between LLVM issue, HIP rewrite, hand-ISA, or production
   Vulkan backend.

The next most useful test is now a Vulkan Q4_K selected-dual gate/up real-slice
probe if we need a second production hot-bucket transfer check; the Q6_K
selected-down X8 slice is already negative. A HIP wave64 plus fixed-workgroup
geometry control is optional if we want to remove the remaining f32 geometry
wave/subgroup confound.
The gfx1151 geometry, VOPD, memory/waitcnt, and dot-path extractions already
found that the current gap is not a missed-HIP-VOPD, HIP-spill, or missed-dot4
story. The next stop is isolating the remaining hypotheses rather than rerunning
broader geometry, generic VOPD, generic memory, or basic dot-lowering sweeps.

The expected useful output is not a single "Vulkan is faster" number. It is a
ranked list of deltas like: "Vulkan wins small-K expert-down by X%; Y% is
geometry, Z% is ACO waitcnt/VGPR quality, remaining is dispatch." That is the
level of evidence needed to guide LLVM work or justify a backend investment.
