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

1. Implement dispatch/grid floor rows for HIP and Vulkan.
2. Implement geometry sweep for f32 GEMV/reduction at K=512/2048/8192.
3. Add VOPD and waitcnt microbenches with ISA/stat extraction.
4. Add q8_1/sudot4 and scalar-dequant GEMV pairs.
5. Port one real slice: selected-MoE small-K or q6 lm-head rowtile.
6. Classify each retained row using the result buckets above.
7. Only then decide between LLVM issue, HIP rewrite, hand-ISA, or production
   Vulkan backend.

The expected useful output is not a single "Vulkan is faster" number. It is a
ranked list of deltas like: "Vulkan wins small-K expert-down by X%; Y% is
geometry, Z% is ACO waitcnt/VGPR quality, remaining is dispatch." That is the
level of evidence needed to guide LLVM work or justify a backend investment.
