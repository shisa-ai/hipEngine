# HIP vs Vulkan Attribution Plan

This document defines the microbenchmark suite we need before turning the
current "Vulkan/RADV/ACO is better than HIP/LLVM here" read into actionable
compiler or backend work. It is a plan, not a performance claim. Retained
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
conclusion is split:

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
- We still cannot claim `compiler_aco` for the f32 geometry gap. RADV official
  VGPR/SGPR allocation counts were not exposed by `RADV_DEBUG=shaders`, and the
  current evidence mixes HIP runtime `blockDim` code with Vulkan specialization
  constants and different wave/subgroup modes.

So the current project-level answer is **not** "HIP is simply slower because
ACO is better." The retained answer is: Vulkan has a proven dispatch/runtime
advantage on gfx1151, Vulkan has a large matched-math advantage in one f32
diagnostic, and the next tests must determine whether that second advantage is
ACO/LLVM memory scheduling, wave/subgroup behavior, specialization, hidden
runtime/pipeline behavior, or an algorithmic detail we have not controlled yet.

HIP also does not give us a PTX-equivalent escape hatch in the normal runtime
path. We can inspect LLVM IR, AMDGPU assembly, and code-object metadata, and we
can use AMDGCN builtins, inline AMDGCN assembly, or standalone HSACO/module
kernels for narrow cases. But normal HIP source is ultimately relying on
LLVM-AMDGPU codegen, so confirmed compiler misses become either LLVM roadmap
items or carefully scoped hand-ISA candidates.

## What To Test Next

Do **not** spend more gfx1151 time on dispatch-only or geometry-only sweeps. The
next useful tranche is:

1. VOPD-specific paired microbenches: independent f32 FMA chains, dependent f32
   chains, dequant-like integer/float chains, and mixed address-math plus FMA.
   Only call this an ACO dual-issue win if Vulkan emits more useful VOPD or an
   equivalent dual-issue schedule at matched occupancy and instruction count.
2. Memory/waitcnt microbenches: coalesced load+accumulate, strided
   load+accumulate, gather IDs, and load-compute interleave. These decide
   whether the f32 geometry gap is memory scheduling/waitcnt quality rather than
   generic ALU scheduling.
3. Dot-path microbenches: q8_1/q4 or q8_0 shapes that prove whether HIP and
   Vulkan both emit the intended RDNA3 dot instructions. If both do and timing
   is close, the remaining work is layout/quant economics; if HIP misses the
   instruction or surrounds it with worse scheduling, that becomes an LLVM or
   hand-ISA target.
4. HIP specialization controls for the f32 geometry kernel: compile fixed
   workgroup-size variants or otherwise remove runtime `blockDim` address/control
   overhead, then compare against Vulkan specialization constants.
5. One representative inference slice after the microbench deltas are
   classified. The best first slices are selected-MoE small-K or q6 lm-head
   rowtile, because they map directly to exposed hipEngine buckets.

Cross-GPU reruns on gfx1100/W7900 and 7900 XTX are important after the harnesses
are stable. They should confirm portability of a conclusion, not replace the
missing gfx1151 ISA evidence.

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
- The next compiler-facing tests should be targeted VOPD and memory/waitcnt
  microbenches where the expected instruction-level win is isolated by design.

## Questions To Answer

1. **Compiler scheduling:** When the algorithm, data layout, wave/subgroup size,
   and workgroup geometry are matched, does RADV/ACO still beat
   LLVM-AMDGPU? If yes, is the delta visible as fewer VGPRs, less scratch, fewer
   `s_waitcnt`, better unroll, or more VOPD pairing?
2. **Geometry:** How much of the Vulkan win comes from 64-thread subgroup
   shapes versus the common HIP 128/256-thread block shapes?
3. **Wave mode:** Does HIP wave64 close any Vulkan gap on reduction/GEMV shapes,
   or does it regress because of occupancy/register pressure?
4. **Dispatch/runtime:** Is Vulkan faster because individual shaders are faster,
   or because command-buffer/pipeline execution reduces per-dispatch cost?
5. **Memory scheduling:** Does ACO sustain higher effective bandwidth on
   coalesced, strided, and quantized GEMV inner loops?
6. **VOPD:** Does ACO find dual-issue opportunities that LLVM misses on
   independent VALU chains? Are those opportunities material for decode kernels?
7. **dp4a/sudot4:** Does the compiler matter once the code uses the intended
   RDNA3 dot instruction, or is the remaining gap layout and quant economics?
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
- F32/BF16 dequant chains where a microbench proves missed VOPD pairing matters
  after occupancy and memory traffic are controlled.
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
   ISA/stat extraction.
5. Add memory waitcnt microbenches: coalesced load+accumulate, strided
   load+accumulate, gather IDs, and load-compute interleave.
6. Add q8_1/sudot4 and scalar-dequant GEMV pairs.
7. Port one real slice: selected-MoE small-K or q6 lm-head rowtile.
8. Classify each retained row using the result buckets above.
9. Only then decide between LLVM issue, HIP rewrite, hand-ISA, or production
   Vulkan backend.

The next most useful tests are now targeted VOPD and memory/waitcnt
microbenches, plus a fixed-workgroup HIP geometry control. The gfx1151 geometry
ISA extraction already found that the f32 geometry gap is not a missed-HIP-VOPD
or HIP-spill story. The next stop is isolating the remaining hypotheses rather
than rerunning broader geometry sweeps.

The expected useful output is not a single "Vulkan is faster" number. It is a
ranked list of deltas like: "Vulkan wins small-K expert-down by X%; Y% is
geometry, Z% is ACO waitcnt/VGPR quality, remaining is dispatch." That is the
level of evidence needed to guide LLVM work or justify a backend investment.
