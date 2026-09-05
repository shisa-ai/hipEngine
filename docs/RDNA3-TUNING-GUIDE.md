# RDNA3 Tuning Guide

This guide describes how to tune inference kernels and their runtime on AMD
RDNA3-class GPUs in hipEngine. It covers the rules that transfer across the
architecture, then separates the decisions that need independent `gfx1100` and
`gfx1151` evidence.

The guidance is based on the current source, design documents, benchmark
artifacts, immutable worklog entries, frozen legacy worklog, and Git history in
hipEngine and its `amd-gpu-tuning` lineage through 2026-09-03. It is a method and
design guide, not a benchmark scoreboard. Use [BENCHMARK.md](BENCHMARK.md) and
[benchmarks/README.md](../benchmarks/README.md) for current results.

## Contents

1. [The short version](#1-the-short-version)
2. [Hardware model](#2-hardware-model)
3. [Classify the workload before changing code](#3-classify-the-workload-before-changing-code)
4. [Build trustworthy evidence](#4-build-trustworthy-evidence)
5. [Core kernel-tuning rules](#5-core-kernel-tuning-rules)
6. [Workload-specific patterns](#6-workload-specific-patterns)
7. [Host, memory, and dispatch tuning](#7-host-memory-and-dispatch-tuning)
8. [`gfx1100`: discrete Navi 31](#8-gfx1100-discrete-navi-31)
9. [`gfx1151`: Strix Halo RDNA 35](#9-gfx1151-strix-halo-rdna-35)
10. [What transfers and what does not](#10-what-transfers-and-what-does-not)
11. [Rejected shortcuts](#11-rejected-shortcuts)
12. [A repeatable tuning procedure](#12-a-repeatable-tuning-procedure)
13. [hipEngine implementation map](#13-hipengine-implementation-map)
14. [Further reading](#14-further-reading)

## 1. The short version

Use these rules until measurements prove an exception:

1. **Compile for the real architecture.** A `gfx1100` code object is not a
   `gfx1151` code object. Include the target, compiler version, source, and all
   flags in the build-cache key.
2. **Assume wave32.** CU mode and wavefront width are separate choices.
   hipEngine's native HIP paths use wave32; do not infer wave64 from `-mcumode`.
3. **Start from bytes and occupancy, not peak arithmetic.** Token-at-a-time
   quantized decode usually streams weights. Prefill and sufficiently wide
   verification can become compute-bound. Long-context attention can become a
   separate KV-bandwidth problem.
4. **Treat layout as part of the algorithm.** Coalesced packed loads, reusable
   resident formats, and row amortization usually matter more than selecting a
   fashionable instruction.
5. **Specialize by measured shape bands.** Token-at-a-time decode, 2-8 packed
   rows, 17-48 verifier rows, large prefill tiles, and long attention are
   different workloads. Keep a strict generic fallback.
6. **Watch VGPRs, LDS, barriers, and scratch.** More workgroups do not improve
   throughput if register pressure permits too few resident waves. Hot kernels
   should normally have zero private scratch.
7. **Use LDS only to create reuse or repair a layout.** Copying already
   coalesced traffic into LDS usually adds barriers without saving bytes.
8. **Prefetch narrowly.** Prefetch the next payload or operand window when it
   overlaps useful work. Stop when VGPR growth, waits, or spills erase the gain.
9. **Do not target instruction counts in isolation.** WMMA, dot-product
   instructions, data-parallel primitives, and VOPD are useful only when the
   production layout and shape can feed them efficiently.
10. **Measure the production owner.** A faster leaf kernel, graph, synthetic
    shader, or cache-hot microbenchmark is not a retained result until the real
    route and its complete wall improve under the applicable correctness gate.
11. **Count overlapping time correctly.** Inclusive kernel durations can exceed
    wall time. Merge timestamp intervals to calculate GPU coverage and overlap.
12. **Promote only operation-complete paths.** A new resident layout or staging
    buffer must support every required owner before the old copy is removed.

## 2. Hardware model

### 2.1 Names and execution geometry

RDNA groups two compute units (CUs) into one workgroup processor (WGP). Each CU
contains two physical SIMD32 units. Tools do not always report the same unit:
`rocminfo` reports CUs on the hosts used here, while some framework properties
report WGP-like counts. State the unit whenever using a device count.

A wave32 has 32 work-items and occupies one SIMD32. A wave64 is executed across
two 32-lane halves. RDNA3 supports both ISA modes, but support does not make
them interchangeable:

- hipEngine's checked HIP paths use wave32;
- `-mcumode` does **not** imply wave64;
- a wave64-oriented build still needs a tensor-level shuffle and reduction
  correctness test;
- cross-wave behavior cannot be inferred from compilation success or a nominal
  `warpSize` alone.

The maximum workgroup size on the validated hosts is 1,024 work-items, or 32
wave32 waves. This is a hard boundary for in-block row or split reduction.
Crossing it requires multiple workgroups and therefore a partial-output or
atomic strategy.

### 2.2 Validated host comparison

The values below describe the project's two primary machines, not every product
that shares an architecture name.

| Property | Radeon Pro W7900 (`gfx1100`) | Radeon 8060S (`gfx1151`) |
| --- | ---: | ---: |
| Design | Navi 31 discrete chiplet GPU | Strix Halo integrated GPU, RDNA 3.5 |
| Compute units | 96 | 40 |
| WGPs | 48 | 20 |
| SIMD32 units | 192 | 80 |
| Default wave | 32 | 32 |
| Maximum waves per CU | 32 | 32 |
| Maximum workgroup | 1,024 work-items | 1,024 work-items |
| Exposed LDS | 64 KiB per CU | 64 KiB group segment |
| L2 | 6 MiB | 2 MiB |
| Last-level cache | 96 MiB Infinity Cache | 32 MiB L3/MALL |
| External-memory roof | 864 GB/s GDDR6 theoretical | 256 GB/s LPDDR5X theoretical |
| Practical large-stream read reference | workload-dependent | about 221-234 GB/s locally |
| Capacity model | 48 GiB dedicated VRAM | unified memory with configured GTT limits |
| FP16/BF16 WMMA peak class | about 123 TFLOP/s | about 59.4 TFLOP/s at 2.9 GHz |

These ceilings explain why the same kernel can occupy a different regime on the
two systems. `gfx1151` has roughly half the matrix-compute roof but less than a
third of the W7900's theoretical external bandwidth. It also has one third of
the L2 and last-level cache. Never copy a split count, row threshold, cache
policy, or chunk size solely because both devices execute RDNA3-family ISA.

### 2.3 A practical memory hierarchy

Think about traffic in this order:

1. **VGPRs** hold per-thread operands and accumulators. They are fastest, but
   allocation is quantized and high usage reduces resident waves.
2. **LDS** is explicitly managed memory shared by a workgroup. It can repair
   scattered layouts and enable reuse, but every phase boundary may need a
   barrier.
3. **L1 and L2** absorb local and repeated reads. A small tensor may behave very
   differently from a model-scale stream.
4. **MALL/Infinity Cache** is the last-level cache. It can make a reused
   microbenchmark appear faster than physical memory.
5. **External memory** supplies model weights and long KV streams once their
   working sets exceed cache.

The chiplet W7900 places its 96 MiB Infinity Cache on memory-cache dies, whereas
Strix Halo uses a smaller shared last-level cache in an integrated-memory
system. Both details affect latency and cache policy, but neither changes the
first question: *how many useful bytes does the production operation move?*

### 2.4 Rooflines are a classifier, not a prediction

Arithmetic intensity is useful operations divided by bytes transferred from the
limiting memory level. Compare it with:

```text
ridge point = peak arithmetic rate / sustainable memory bandwidth
```

Examples:

- Token-at-a-time quantized matrix-vector multiplication reads each active
  weight for very little reuse. It is normally memory- or instruction-overhead
  bound.
- A multi-row verifier can reuse one weight tile across many rows. Arithmetic
  intensity rises with the row count until other costs dominate.
- Large prefill matrix multiplication can fill WMMA tiles and become
  compute-bound.
- Long-context attention repeatedly streams K and V. It can become
  KV-bandwidth-bound even when short-context decode is weight-bound.

A byte roof is not an expected throughput number. Dequantization, address
arithmetic, cache-line waste, low occupancy, barriers, synchronization, launch
cost, and unrelated model traffic all reduce achieved performance. Conversely,
a cache-hot or multi-row test can legitimately exceed a one-stream external
bandwidth calculation because it reuses data.

## 3. Classify the workload before changing code

Name the complete workload tuple before selecting a technique:

```text
(physical host, architecture, model, quantization, execution profile,
 operation, M/rows, K, N, context, batch/packed width, state shape)
```

For a linear operation, `M` is the number of activation rows, `K` is the
reduction width, and `N` is the output width. For attention, include the query
and KV head counts, head dimension, live-token geometry, and whether the cache
is a dense prefix, a dense ring, or has explicit eviction.

### 3.1 Token-at-a-time decode

Typical properties:

- `M=1` or a small packed width;
- weights dominate bytes;
- limited reuse inside one operation;
- many small launches can expose dispatch cost;
- matrix instructions waste most of an `M=16` tile at `M=1`;
- simple one-wave or few-wave kernels often beat tiled prefill designs.

Optimize coalescing, dequant instruction count, row packing, reduction cost,
VGPR occupancy, launch count, and residency before trying larger matrix tiles.

### 3.2 Multi-row verification and batching

Typical properties:

- the same weight tile serves several activation rows;
- row amortization can reduce weight traffic per output row;
- WMMA becomes useful when enough real work fills the tile;
- accumulator count can push the kernel over a VGPR occupancy cliff;
- optimal policies can repeat by tile-capacity bands rather than increase
  monotonically with row count.

Treat `17-32`, `33-48`, and `49-64`, for example, as distinct candidates if the
kernel processes rows in 16- or 32-row units. Do not label all of them
"small batch" and force one owner.

### 3.3 Prefill

Typical properties:

- enough rows to fill matrix tiles;
- compute throughput and K-loop scheduling matter more;
- launch overhead is amortized over more work;
- chunk size trades launch count and reuse against scratch, state traffic, and
  tail behavior;
- long prefills can expose allocator and queue-policy problems absent in
  decode.

Use a separate build/profile and dispatch policy from decode. A compiler
unrolling flag or WMMA shape that helps prefill can be neutral or harmful for
weight-streaming decode.

### 3.4 Long-context attention

Short and long attention are not one regime. As live KV grows:

- K/V bytes exceed L2 and then last-level cache;
- address and page metadata repeat across many query heads;
- grouped-query attention creates query reuse opportunities;
- token-loop width, QK reduction, softmax chronology, PV prefetch, split count,
  and cache policy acquire different crossover points;
- non-temporal loads can become useful after reuse distance exceeds cache
  value, even though they lose at short context.

Keep short-context and eviction-capable fallbacks when adding a dense-prefix or
long-stream specialization.

### 3.5 Mixture-of-experts models

Mixture-of-experts (MoE) execution adds routing, selected-expert gathers,
variable active shapes, and sometimes an independent shared branch. Important
costs include:

- dispatch and temporary-buffer storms around grouped gathers;
- sparse weight access and poor coalescing;
- gate/up and down-projection launch count;
- selected-expert layout and row grouping;
- routing state and exact accumulation order;
- overlap potential between genuinely independent branches.

An expert-sequential schedule is not automatically cache-friendly enough to
win. Measure selected-expert locality and full wall rather than assuming L2
reuse.

## 4. Build trustworthy evidence

### 4.1 Define the owner and the baseline

Before editing a kernel:

1. Trace the runtime selector to the exact registry key and variant.
2. Confirm the model, quantization, row/shape band, execution profile, and
   architecture that own the call.
3. Record the current full-operation and full-request baseline.
4. Capture a kernel trace proving the expected symbol ran.
5. Record code-object resources and final ISA for the hot kernel.

A production result is invalid if the benchmark silently exercised a fallback,
another checkout, the wrong model file, stale JIT output, or a child kernel that
the real route does not own.

### 4.2 Use layered timing

Use each timing layer for its proper question:

| Layer | Answers | Does not prove |
| --- | --- | --- |
| Device-event microbenchmark | Did this exact launch get faster? | The application owner or full wall improved |
| Kernel trace | Which kernels ran, for how long, with what geometry? | Inclusive sums equal elapsed time |
| Launch-attributed phase window | What work was enqueued by this phase? | End-to-end request economics |
| Full operation/request wall | Did the user-visible path improve? | Which mechanism caused it |
| Prompt/task suite | Did the optimization retain useful output quality? | Bitwise arithmetic identity |

Warm up the code path and clocks before timed samples. Use repeated,
counterbalanced baseline/candidate runs. Report median and tails when variance
matters; do not promote a result smaller than uncontrolled run-to-run movement
without stronger paired evidence.

### 4.3 Calculate overlap with interval unions

On multiple streams, summing `DurationNs` double-counts simultaneous kernels.
Calculate GPU coverage by sorting all `[start,end)` intervals and merging
intersections. Also report:

- inclusive duration by kernel family;
- unioned device-active time;
- overlap between named streams;
- unaccounted complete wall;
- host API behavior around synchronization.

A large gap in a kernel timeline is not automatically active Python dispatch.
The host can be blocked in a stream synchronize while device work or an
unobserved transport phase drains. Use an API trace and a controlled transport
A/B before assigning the gap to launch overhead.

### 4.4 Separate cache-hot and cache-cold tests

For a cold-stream bandwidth test:

- calculate the true encoded bytes for the quant format;
- rotate through a buffer pool larger than twice the relevant last-level cache;
- include metadata, scales, zero points, activations, and outputs as applicable;
- distinguish read bytes from useful logical values;
- do not infer external-memory bandwidth from a 4-13 MiB buffer reused inside a
  32 or 96 MiB cache;
- explain row reuse when a multi-row kernel exceeds a one-row byte roof.

Cache-hot tests remain useful for instruction, synchronization, and dispatch
comparisons. Label them accurately.

### 4.5 Triangulate profiler counters

Counters on current ROCm stacks can be unavailable, zero, or collected in units
that differ from the tool label. Cross-check every important conclusion with:

- grid and workgroup dimensions;
- static VGPR, SGPR, LDS, and scratch metadata;
- code-object disassembly;
- event timing;
- complete wall;
- a known-control kernel when using hardware counters.

Do not build an occupancy theory from a zero `SQ_WAVES` value or a bandwidth
theory from an implausible `MemUnitBusy` result.

### 4.6 Correctness is part of performance

Follow [EXECUTION-PROFILES.md](EXECUTION-PROFILES.md) and
[TESTING.md](TESTING.md):

- **Strict paths** preserve their declared exact or parent-parity arithmetic
  contract.
- **Production paths** may reassociate only after exact state/control ownership,
  deterministic/isolation checks, calibrated mean/tail/max KL and top-1 gates,
  BF16-relative checks where applicable, and the full task/category suite.
- Every fused or production variant keeps a registered strict fallback.
- The broad smoke floor—KL at most 0.05 and at least 90% top-1 agreement against
  the CPU reference—is necessary but not enough to promote a production
  default.

Reduction order, conversion points, and softmax chronology are semantics, not
mere implementation detail. A faster kernel that changes one mandatory token or
fails a category gate is not an optimization for that profile.

## 5. Core kernel-tuning rules

### 5.1 Fix layout before instruction selection

For quantized inference, layout determines whether lanes can load useful packed
bytes together. Start by mapping:

- which contiguous bytes each lane reads;
- how scales and metadata are shared;
- whether packed values require scattered extraction;
- whether activation values are reloaded for every output group;
- whether a format conversion occurs in every launch;
- whether the output layout causes scalar LDS or global transactions.

Preferred order:

1. coalesce the stored bytes;
2. vectorize aligned loads;
3. reuse activation and metadata values;
4. reduce unpack/dequant instructions;
5. only then evaluate dot-product or matrix instructions.

A dot-product intrinsic cannot recover bandwidth lost to a scattered format.
Any repack should occur once at load time or into a persistent operation-complete
layout, not on each token.

### 5.2 Amortize weights across rows

When several activation rows use the same weights, assign one workgroup or tile
to load a weight record once and update multiple row accumulators. This
**row-amortized** pattern has transferred across output heads, selected experts,
Q4/Q5/Q6 kernels, and packed autoregressive paths.

The tradeoff is accumulator state:

- more rows increase reuse;
- more row/column accumulators increase VGPR allocation;
- lower occupancy can reduce outstanding memory requests;
- collapsing the M grid can also remove useful N-direction parallelism.

Measure a ladder of row capacities and column tiles. The best selector often
uses periodic capacity bands rather than one formula.

### 5.3 Optimize resident waves, not block count

Grid coverage is necessary, not sufficient. Check:

- total workgroups relative to 96 CUs on W7900 or 40 CUs on Strix Halo;
- waves per workgroup;
- allocated VGPRs and SGPRs;
- LDS per workgroup;
- private scratch;
- resident wave limit after all resources are combined.

A low-row WMMA kernel can launch hundreds of blocks and remain latency-bound if
roughly 200-250 VGPRs per thread permit only a few waves per issue slot. In that
case, reducing the accumulator tile can outperform adding more blocks. Treat
any sharp VGPR rise as a possible occupancy-class transition and verify it with
resources plus timing.

Keep private scratch at zero on hot paths unless a measured exception justifies
it. A tiny spill inside a deep K loop can dominate the kernel.

### 5.4 Unroll for independent work, not source size

Deep K-loop unrolling can expose independent loads and arithmetic to the
scheduler. It is especially useful when the compiler otherwise emits a serial
load-dequant-accumulate chain. But unrolling also increases live values and
code size.

For each candidate:

1. inspect VGPR and scratch changes;
2. inspect `s_waitcnt` placement and instruction grouping;
3. measure representative K and row bands;
4. test tails and exact reduction order;
5. re-tune after any layout, cache, or producer change.

A compiler profile flag is not a universal architecture feature. The
`-amdgpu-unroll-threshold-local=600` setting produced a large external prefill
win in one lineage but was neutral in a direct current hipEngine PARO ablation.
Keep such flags in an architecture/profile matrix.

### 5.5 Prefetch only what overlaps

Useful prefetch candidates have a clear consumer distance:

- the next packed weight payload;
- the next activation record;
- the next QK or PV operand window;
- a double-buffered activation tile that removes a barrier.

Avoid prefetching every decoded field or metadata value speculatively. That can
lengthen live ranges enough to lower occupancy. Compare depth 1, 2, 4, and only
then larger windows; stop at the first robust plateau. Inspect whether the
compiler actually moved loads ahead of arithmetic and whether waits still
serialize the loop.

The best prefetch depth is not permanent. Long-attention PV prefetch became
more valuable after later layout and cache-policy changes altered latency. Reopen
old depth sweeps after a structural producer change.

### 5.6 Use the cheapest correct reduction

For wave32-local reductions, prefer register shuffles or architecture-supported
lane operations over an LDS round trip. Preserve the required accumulation
chronology.

Data-parallel primitive (DPP) and `permlane` operations can replace repeated
cross-lane permutations in the right family. They are not blanket switches:

- a planar Q6 reduction on `gfx1100` reduced permutation count and VGPRs and
  won;
- analogous Q4/Q5 forms were mixed or slower;
- narrow QK reductions on `gfx1151` have their own exact DPP wins.

Use them when the lane map matches the instruction and final resources improve.
Retain only production-slice gains.

### 5.7 Make LDS earn every barrier

LDS is appropriate when it does at least one of these:

- converts a scattered access into a coalesced/vector access;
- shares a value many times within a workgroup;
- supports a reduction that cannot remain wave-local;
- double-buffers a tile while removing, rather than adding, a synchronization
  phase.

It is usually a loss when it merely copies already coalesced data or stores a
large FP32 accumulator plane. Successful examples in this project used LDS to
make an output-major layout vectorizable or to share staged activations with
fewer barriers. Failed examples copied aligned weights, staged KV already served
by cache, or held complete FP32 partials while reducing output parallelism.

As an initial filter, require meaningful reuse—often more than four consumers—or
a demonstrated layout repair. Then measure. Count LDS instructions and barriers,
not just allocated bytes.

### 5.8 Choose WMMA by useful tile occupancy

Wave matrix multiply-accumulate (WMMA) is the right default for sufficiently
large prefill and can win for multi-row verification. It is normally the wrong
default for `M=1`: 15 of 16 rows in the minimum matrix tile are unused.

For low and medium rows:

- fill tiles with real request/verifier rows;
- vary row capacity and output-column tile together;
- reduce accumulator tiles if VGPRs limit resident waves;
- include quant unpack and activation-pack cost;
- compare against row-amortized vector/dot owners, not only a generic GEMV;
- inspect tail handling at non-multiple row counts.

A single-sweep WMMA kernel can still lose if it replaces parallel workgroups
with a large LDS accumulator, reaches the 1,024-thread limit, or requires
cross-workgroup partials.

### 5.9 Treat integer dot/MMQ as a layout decision

RDNA3 exposes mixed signed/unsigned dot operations such as `sudot4`; portable
signed `sdot4` assumptions have failed in this lineage. Integer matrix paths can
reduce weight and activation bytes, but their end-to-end value depends on:

- native instruction availability in the selected compiler path;
- resident packed layout;
- activation quantization and sum preparation;
- scale/zero-point traffic;
- row count and tile fill;
- whether a second weight copy is affordable;
- exact or production-profile arithmetic requirements.

Current evidence illustrates the boundary: a planar Q6 integer MMQ path is
useful for a measured `gfx1151` verifier row band, while Q4 and Q5 integer bodies
lost and standard Q6 could not justify changing its sole c=1 resident layout.
"Fewer-bit math" is not enough.

### 5.10 Let VOPD emerge from independent work

VOPD encodes compatible pairs of independent vector arithmetic operations. It
can raise FP32 throughput, but its presence is not a useful optimization metric
by itself. The compiler may emit VOPD for a slower schedule and omit it for a
faster one.

Create independent accumulator chains, avoid unnecessary dependencies, inspect
the final ISA, and measure the real kernel. Do not preserve a slower path to
increase a VOPD count.

### 5.11 Make cache policy follow reuse distance

Cache modifiers are workload decisions:

- repeated or row-amortized weights should normally remain cacheable;
- broad non-temporal weight loads improved one cold microbenchmark but were
  flat or worse end to end because they destroyed useful reuse;
- dense-prefix attention at very long context has a different working set, and
  non-temporal K/V loads became beneficial only around a measured 64K-token
  crossover;
- thresholds can differ by architecture and context geometry.

Add cache policy only behind an explicit reuse/working-set hypothesis and test
both sides of the crossover. Never generalize one non-temporal win to every
large tensor.

### 5.12 Remove invariant address work

Once a state invariant is proven, compile it into a specialization:

- power-of-two block sizes permit shift/mask instead of divide/modulo;
- dense-prefix or saturated dense-ring states permit direct physical addressing;
- shared GQA page metadata can be hoisted across query heads;
- repeated activation-only quant sums can be precomputed once;
- fixed shapes can replace general indexing with constant strides.

Keep the generic path for eviction, partial rings, non-standard block sizes, and
unlisted shapes. State specialization must never weaken ownership semantics.

## 6. Workload-specific patterns

### 6.1 Quantized decode GEMV

A strong starting design is:

- architecture-native wave32;
- coalesced vector loads of packed weights;
- scalar or packed arithmetic with multiple independent accumulators;
- one wave or a small number of waves per output group;
- shuffle/DPP reduction when wave-local;
- no LDS unless it repairs a measured layout problem;
- zero private scratch;
- shape-specific workgroup sizing for small K or N;
- persistent converted/repacked weights only when the full operation set can use
  the layout.

For `M=1`, reduce instructions per encoded byte and preserve enough waves to
cover memory latency. For packed widths, reuse each weight record across rows
without eliminating too much output parallelism.

### 6.2 Multi-row quantized linear and verifier paths

Test these owner classes independently:

1. per-row GEMV for the smallest or strict shapes;
2. row-amortized vector/dot kernels;
3. low-VGPR WMMA siblings for tile-filling rows;
4. integer MMQ where quant layout and profile permit it;
5. large-row prefill owners.

Dispatch transfer can be the largest win. In current `gfx1151` evidence,
routing Q5/Q6 verifier rows away from hundreds of per-row GEMV launches to
existing row-amortized WMMA owners removed most of the pass time. Before writing
a new kernel, ask whether a qualified owner already exists for the real shape.

### 6.3 Attention decode

Preserve the `KVLiveSpans` ABI:

```text
(base_offsets, live_counts, token_positions, evict_mask)
```

Dense policies fill it uniformly; sparse/evicting policies vary it. A fast path
may exploit a proven dense prefix or dense ring, but must retain the generic
owner.

Tune attention in layers:

1. hoist repeated address, page, and position work;
2. reuse one KV load across grouped query heads where the mapping permits it;
3. tune token-loop width;
4. use a correct wave-local QK reduction;
5. preserve or explicitly qualify online-softmax order;
6. prefetch PV operands at measured depths;
7. tune V-stage/output sharding;
8. consider split context only when one workgroup cannot expose enough work;
9. apply long-stream cache policy after the working set crosses cache.

Grouping more queries is not automatically better. It can reduce wave
parallelism or inflate registers. Likewise, split-K/context is useful only when
added workgroups cover an undersubscribed kernel enough to pay for partial
reduction and scratch traffic.

### 6.4 Prefill attention and linear layers

Prefill prefers larger tiles, WMMA, and enough row work to amortize loads.
Important controls include:

- architecture-specific chunk size;
- complete vs tail tile owners;
- stable scratch ownership;
- numerically stable softmax and explicit row-validity masks;
- K-loop unroll and staged operands;
- fused operations only when they reduce traffic or launches without harming
  tile occupancy.

A chunk-size win can be model- and architecture-specific. Test short, medium,
and long prompts, including tails, and keep memory-capacity effects in the same
comparison.

### 6.5 MoE execution

Prioritize structural traffic and launch reductions:

- keep routing decisions and selected indices device-resident;
- group real selected rows so weights are reused;
- avoid host readback of one scalar to choose a kernel;
- fuse gate/up or combine/residual operations where the strict fallback exists;
- replace per-expert gather/temporary/launch sequences with grouped owners when
  the layout supports it;
- pack selected layouts at load time or into bounded persistent storage;
- overlap a shared branch only when dependencies prove independence.

Report branch overlap with interval unions. A secondary stream can overlap
almost completely while inclusive family sums rise from resource contention;
only complete wall decides whether the overlap is useful.

### 6.6 Pointwise and normalization kernels

Small pointwise kernels are often launch-bound. Fusion is valuable when it:

- removes an intermediate global write/read;
- removes one launch;
- shares a reduction or already-loaded value;
- keeps exact ownership and output semantics.

Do not fuse merely to reduce a kernel count if the result raises VGPR/LDS enough
to slow the dominant operation. Every fused composite needs a registered strict
unfused chain.

## 7. Host, memory, and dispatch tuning

### 7.1 Keep JIT work out of launch wrappers

A hot launch path must not:

- invoke `hipcc` or query `hipcc --version`;
- hash source repeatedly;
- load the same shared library repeatedly;
- reconstruct `ctypes` prototypes;
- allocate and free reusable workspaces.

This is not minor bookkeeping. Historical fixes that cached compiler-version
queries, loaded `ctypes.CDLL` handles, and family bindings removed milliseconds
to tens of milliseconds from affected routes. Audit the production step until
it performs zero JIT builds.

hipEngine's build plan includes source bytes, normalized flags, compiler
version, target architecture, includes, and extra flags in a deterministic
cache key. Profiled children should prebuild outside `rocprofv3` and run with
`HIPENGINE_REQUIRE_CACHED_BUILD=1`.

### 7.2 Own scratch by phase and lifetime

Preallocate repeated scratch and pack non-overlapping ranges by a real lifetime
graph. Useful techniques include:

- one resident session owner;
- phase-liveness aliasing;
- bounded verifier and attention workspaces;
- stable graph-visible slabs;
- deliberate reserve for runtime private scratch where required by the host.

Do not alias buffers because their names suggest different phases. Prove the
last consumer. An early FFN alias caused nondeterministic logits because the
down projection still consumed an intermediate that had been overwritten.

### 7.3 Count physical allocations

Allocation metadata and ownership have costs beyond requested bytes. Packing
hundreds of immutable allocations into a few physical arenas can reduce whole-
card peak and startup overhead. Choose arena boundaries from operation/family
lifetimes and fallback needs, not an arbitrary size tuned to one inventory.

For integrated memory, distinguish:

- firmware-reserved GART/VRAM aperture;
- GTT/TTM allocation limits;
- process-visible capacity;
- physical system memory pressure.

For discrete memory, distinguish requested device bytes from runtime reserves,
allocator overhead, and mapped host pages.

### 7.4 Keep one operation-complete resident layout

Alternate packed layouts can improve one kernel and still make the product
worse by doubling model storage. Before changing the canonical resident layout,
list every consumer:

- scalar decode;
- packed decode;
- verifier rows;
- prefill;
- selected expert paths;
- output head;
- strict fallback;
- CPU/reference or transfer boundaries.

Promote the new layout only after all required owners are qualified. Otherwise,
keep a bounded derived workspace or reject the conversion. The sole-layout
migration in this project removed more than 10 GiB only after the complete
owner set existed.

### 7.5 Distinguish conversion savings from byte savings

BF16 and FP16 occupy the same bytes, but a bounded FP16 activation staging
buffer can still win by removing repeated conversion instructions and enabling
packed half loads. Treat this as an instruction/load-shape optimization, not
memory compression. Keep the strict BF16 owner when required, and make the
workspace part of session lifecycle rather than module-global mutable state.

### 7.6 Use host mapping only for sparse access

Mapped host memory can avoid a large device copy for data touched sparsely, such
as one token-embedding row per step. It is usually a bad trade for a tensor
streamed in full every token, such as a large output head. Validate graph
compatibility, alignment, lifetime, and file-backed mapping behavior; one direct
file mapping corrupted trajectories while an immutable copied mapping was safe.

### 7.7 Treat graphs as stateful executables

HIP graph replay reduces repeated submission overhead, but capture and
instantiation must amortize over enough transitions. A graph key must include
all state that affects pointer identity or kernel arguments, including:

- batch/packed width and physical rows;
- sequence/KV state generation;
- workspace and allocation generation;
- model, quant, profile, and architecture policy;
- prefix clone, compaction, or slot reuse;
- transport-specific constraints.

Invalidate or flush on identity changes. Output buffers alone rarely help when
allocation work is already captured. Multi-step graphs can lose when capture or
state-management cost exceeds replay savings.

### 7.8 Use PM4 only where qualified

PM4 is the low-level packet stream submitted to AMD's command processor. In
hipEngine, retained native PM4 is a narrow `gfx1100` transport optimization for
qualified immutable graph shapes. It has strict code-object, wave32, scratch,
state, and replay-horizon requirements and keeps HIP graph/eager fallbacks.

Do not infer a large PM4 opportunity from a visual trace gap. The measured
`gfx1100` gains are modest and shape-dependent. Direct AQL removed visible gaps
in one experiment but regressed complete wall. `gfx1151` currently uses HIP
graphs rather than copying `gfx1100` packet evidence.

See [PM4.md](PM4.md) for the transport contract.

### 7.9 Overlap only independent work

Multiple streams help when branches are independent and use complementary
resources. They can hurt when they contend for the same bandwidth, cache, or
compute units. Requirements:

- prove dependency and output ownership;
- use explicit events rather than broad synchronizations;
- measure overlap and complete wall;
- remeasure after queue-count or runtime changes;
- preserve a sequential rollback.

Queue count is host- and workload-specific. One queue fixed an older long-
prefill stall, while a later full `gfx1151` matrix selected two queues for an
independent MoE branch. Neither is a universal RDNA3 setting.

## 8. `gfx1100`: discrete Navi 31

### 8.1 Architecture-specific assumptions

For the validated W7900:

- use 96 CUs, not 48 WGPs, for grid-sufficiency reasoning;
- compile native `gfx1100` code objects;
- use wave32; the decode build profile adds `-mcumode` without requesting
  wave64;
- account for 6 MiB L2, 96 MiB Infinity Cache, and a 48 GiB dedicated-memory
  budget;
- use 864 GB/s only as a theoretical GDDR6 roof, not an expected kernel rate;
- treat RX 7900 XTX or another `gfx1100` card as a separate physical-host lane.

The W7900's larger CU count favors enough output/context splits to cover 96
CUs, but its larger external bandwidth can expose instruction, reduction, and
launch overhead sooner than on `gfx1151`.

### 8.2 Proven `gfx1100` patterns

The most transferable retained patterns are:

- one-wave/no-LDS token-at-a-time quantized decode where shape permits;
- vectorized packed loads and load-time layout conversion;
- deep but resource-checked K-loop unrolling;
- row-amortized packed-width owners;
- DPP/permlane only in measured reduction families, including planar Q6;
- grouped-query KV reuse and architecture-specific attention split thresholds;
- dense-prefix long-attention address simplification;
- non-temporal K/V only at long-context crossover;
- stable scratch and graph-visible allocations;
- native PM4 only for qualified graph shapes and replay horizons.

### 8.3 `gfx1100` cautions

- Do not assume the 96 MiB cache makes model-scale weights resident.
- Do not derive external bandwidth from a reused buffer smaller than the
  Infinity Cache.
- Do not use a 48-WGP tool count as the CU launch target.
- Do not infer wave64 from CU mode or from Vulkan subgroup size.
- Do not transfer a `gfx1151` low-row WMMA band without a W7900 gate; different
  bandwidth and occupancy economics can change the winner.
- PM4 is a transport optimization, not a reason to defer dominant kernel work.
- Keep runtime private-scratch reserve and graph-memory ownership in capacity
  measurements; requested model bytes alone are incomplete.

### 8.4 Suggested first checks

For a slow `gfx1100` kernel:

1. Is the grid capable of covering 96 CUs?
2. Is it a model-scale stream or a cache-resident shape?
3. Are VGPRs, LDS, or scratch limiting resident waves?
4. Are packed loads coalesced and aligned?
5. Can one weight read serve more packed rows or GQA queries?
6. Is the reduction wave-local, and can DPP replace a costly permutation in
   this exact lane map?
7. Does the full owner remain slow after launch/graph overhead is separated?
8. At long context, has the K/V working set crossed the measured cache-policy
   threshold?

## 9. `gfx1151`: Strix Halo RDNA 3.5

### 9.1 Architecture-specific assumptions

For the validated Radeon 8060S:

- use 40 CUs, not the 20 WGP-like count reported by some APIs;
- compile native `gfx1151` code objects;
- use wave32;
- account for 2 MiB L2 and 32 MiB L3/MALL;
- use 256 GB/s as the theoretical LPDDR5X roof and about 221 GB/s as an
  optimistic local read reference until the exact kernel proves otherwise;
- model capacity through GTT/TTM and system memory, not a discrete-VRAM number;
- select row bands, chunking, graph thresholds, and queue policy separately from
  `gfx1100`.

The lower bandwidth per unit of compute makes byte reduction and reuse
especially valuable. At the same time, high-VGPR low-row kernels can still be
latency-bound because too few waves remain resident.

### 9.2 Proven `gfx1151` patterns

Retained patterns include:

- architecture-specific dispatch rather than inherited `gfx1100` defaults;
- low-VGPR WMMA siblings for measured small/mid-row bands;
- row-capacity-periodic selector tables;
- row-amortized Q4/Q5/Q6 and verifier owners;
- scoped activation staging and bounded production FP16 conversion;
- planar Q6 integer MMQ for its qualified production row band;
- dense-prefix and dense-ring attention fast paths with generic fallbacks;
- exact DPP QK reductions in qualified attention shapes;
- a local-1024 saturated-ring owner where all 40 workgroups remain available;
- architecture-specific prefill chunking;
- two-stream overlap for a proven independent MoE shared branch;
- HIP graph replay after a measured amortization floor.

### 9.3 Unified-memory setup

Large-model capacity depends on the complete system configuration. Record:

- firmware GART/VRAM aperture;
- `/sys/module/ttm/parameters/pages_limit` and GTT limit;
- process-visible allocation limit;
- system RAM pressure and swap behavior;
- IOMMU mode;
- power and clock policy.

A small firmware-reserved aperture plus a sufficiently large GTT/TTM limit is
the normal large-model setup on the validated host. `amd_iommu=off` can affect
performance and capacity behavior, but it also disables IOMMU-dependent devices
such as XDNA and is not a universal recommendation. TuneD, host governor, and
GPU `high` performance level were neutral or noisy in retained application
tests. Treat all system settings as same-host experimental variables.

### 9.4 `gfx1151` cautions

- A 20-"multiprocessor" report is WGP-like; do not stop at 20 workgroups when
  reasoning about 40 CUs.
- Cache-hot buffers below 32 MiB do not measure LPDDR5X streaming.
- More workgroups do not fix a 200+ VGPR occupancy cliff.
- Wave64, fixed blocks, generic VOPD targeting, and broad non-temporal weights
  repeatedly failed as blanket switches.
- Integer MMQ needs the right quant layout and row band; Q6 evidence does not
  admit Q4/Q5 or all Q6 consumers.
- A global prefill chunk size outside the measured model/context band can lose.
- Synthetic Vulkan instruction or dot-product gaps have not justified replacing
  production HIP owners.
- `gfx1100` PM4 packets and replay thresholds are not portable evidence;
  current `gfx1151` policy uses HIP graphs.

### 9.5 Suggested first checks

For a slow `gfx1151` kernel:

1. Is the launch target expressed in 40 CUs or 20 WGPs?
2. Is the benchmark cold beyond the 32 MiB last-level cache?
3. Does the kernel have enough resident waves after VGPR/LDS allocation?
4. Can a smaller accumulator tile improve occupancy without excessive launches?
5. Can weights be shared across a row-capacity band?
6. Is repeated BF16 conversion or scalar operand loading visible in ISA?
7. Does an existing prefill/WMMA owner already fit the verifier shape?
8. Is a second layout affordable, or must the operation use the sole resident
   form?
9. Is unified-memory configuration limiting capacity or causing migration?
10. Does stream overlap reduce complete wall rather than only move inclusive
    kernel time?

## 10. What transfers and what does not

| Decision | Transfer across both | Must be qualified per architecture/host |
| --- | --- | --- |
| Native compilation | Yes | Target code object and flags |
| Default wave32 | Yes | Any wave64 experiment |
| Registry/fallback structure | Yes | Variant and selector table |
| Layout-before-intrinsics rule | Yes | Winning resident layout |
| Zero hot-path scratch goal | Yes | Runtime reserve and exceptions |
| Row amortization | Yes | Row capacity and column tile |
| Low-VGPR occupancy tuning | Yes | VGPR cliffs and winning tile |
| WMMA for broad prefill | Usually | Small/mid-row crossover |
| DPP/permlane | Mechanism only | Exact family and lane map |
| Integer dot/MMQ | Mechanism only | Quant, layout, profile, row band |
| Non-temporal loads | Reuse-distance rule | Tensor and context crossover |
| Attention dense-state fast path | Structure | Split, local size, prefetch depth |
| Prefill chunking | Method | Chunk size and tails |
| Multiple streams | Method | Queue count and branch economics |
| Graph replay | State contract | Capture floor and shape policy |
| PM4 | No | Qualified `gfx1100` paths only |
| Capacity policy | No | Dedicated VRAM vs unified GTT/TTM |
| Absolute throughput | No | Every physical host and workload |

A source-compatible kernel is only a starting candidate. Architecture promotion
requires same-host correctness, resource, leaf, slice, and complete-wall
evidence.

## 11. Rejected shortcuts

These ideas are not forbidden. They are failed defaults that require a new,
specific mechanism before another sweep.

| Shortcut | Why it failed | Better question |
| --- | --- | --- |
| "Use wave64 on RDNA3" | Physical SIMD32 halves, shuffle ambiguity, and repeated neutral/negative application results | Which exact reduction or occupancy problem would wave64 solve, and can a tensor test prove it? |
| "More workgroups means more occupancy" | VGPR/LDS limits can keep resident waves unchanged | Which resource sets residency, and can a smaller tile cross a resource cliff? |
| "Use WMMA for every linear" | `M=1` wastes tile rows; low-M accumulators can consume too many VGPRs | Are enough real rows present, and what is the WMMA-versus-row-amortized crossover? |
| "Dot4 is four times faster" | Pack/unpack, activation quantization, scales, and scattered layouts dominate | Can the resident bytes feed the instruction with less total work? |
| "More VOPD is faster" | VOPD count did not correlate reliably with slice wall | Does independent work improve final scheduling and complete time? |
| "Stage it in LDS" | Barriers and duplicate traffic outweighed cache savings | Does LDS create reuse, vectorize a scatter, or remove a barrier? |
| "One sweep avoids all rereads" | Huge accumulator/LDS state and less N parallelism lost | What traffic is actually removed, and what occupancy/parallelism is surrendered? |
| "Split-K always fills the GPU" | Partial reduction and scratch traffic can exceed the benefit | Is the original grid undersubscribed after within-block parallelism is fixed? |
| "Mark large loads non-temporal" | Short/reused weights lost cache value | Is reuse distance beyond cache, and where is the measured crossover? |
| "A graph timeline gap is host dispatch" | Inclusive sums and synchronization hid the real owner | What do merged intervals, API trace, and transport A/B show? |
| "Graph/PM4 will remove most decode time" | Kernel work dominated; retained transport gains were narrow | Is submission a measured fraction after correct interval accounting? |
| "A cache-hot bandwidth result is DRAM bandwidth" | Working set stayed inside MALL/L3; one byte formula was also wrong by 8x | Does a rotating pool exceed twice last-level cache, with encoded bytes counted correctly? |
| "A faster synthetic Vulkan shader should replace HIP" | Compiler/backend microbench gaps did not transfer to production slices | Which source-level schedule/layout mechanism survives the real owner? |
| "One chunk size fits all prefill" | Model, context, scratch, and architecture changed the optimum | Which measured shape band owns this chunk? |
| "Persistent alternate layouts are free" | Duplicate weights consumed many GiB and narrowed model capacity | Is the layout operation-complete, or can a bounded workspace capture the benefit? |
| "Alias scratch with an apparently later buffer" | A hidden consumer caused nondeterministic corruption | What is the proven last-use graph for every range? |
| "Tune clocks/governor first" | `high` performance mode and TuneD were neutral/noisy in retained tests | Is there measured throttling under a same-commit control? |

## 12. A repeatable tuning procedure

### Step 1: Establish the contract

- Name the model, quant, profile, operation, exact shape band, context, and host.
- Identify strict versus production arithmetic requirements.
- Trace the registered owner and fallback.
- Add a RED fixture before changing math or state when practical.

### Step 2: Record a trustworthy baseline

- Warm the exact production route.
- Measure complete wall with repeated counterbalanced runs.
- Capture leaf and phase timings.
- Capture a kernel trace with names, timestamps, grid, workgroup, VGPR, SGPR,
  LDS, and scratch.
- Inspect current ISA for loads, waits, reductions, barriers, spills, and useful
  packed instructions.

### Step 3: Classify the limiter

Ask in order:

1. Is the production owner actually running?
2. Is the operation launch/host-bound, memory-bound, instruction-bound,
   occupancy-bound, synchronization-bound, or undersubscribed?
3. Does the working set reside in L2/MALL or stream from external memory?
4. Is the grid large enough in **CUs**?
5. Are resident waves limited by VGPR, LDS, or scratch?
6. Does the layout coalesce encoded bytes?
7. Is there cross-row, cross-query, or cross-expert reuse?

Do not choose a kernel technique until this classification has evidence.

### Step 4: Make one structural change

Prefer changes in this order:

1. correct dispatch to an existing better owner;
2. eliminate redundant build/load/allocation/readback work;
3. reduce bytes or make the resident layout operation-complete;
4. create real row/query/expert reuse;
5. remove invariant address/conversion work;
6. lower VGPR/LDS/scratch residency pressure;
7. remove barriers or launches;
8. tune unroll, prefetch, DPP, WMMA, dot, and cache policy.

This order avoids polishing an owner that should not exist.

### Step 5: Validate from leaf to product

- Run the strict or production-profile numerical gate.
- Confirm deterministic repeatability and state isolation.
- Confirm the expected symbol and resources in `rocprofv3`.
- Measure the leaf with hot/cold labeling.
- Measure the production phase with launch-attributed intervals.
- Measure complete request wall and peak memory.
- For sampling or speculative paths, run the full multi-prompt category suite
  and held-outs; never promote from one fixed prompt.

A leaf win may be retained as useful evidence without claiming an end-to-end win,
but it must not silently become the default if the complete route regresses.

### Step 6: Promote and collapse the seam

After a positive same-suite result:

- make the qualified path the architecture/shape default;
- preserve the registered strict fallback;
- remove dead experiment variants and arbitrary fallback chains;
- record any temporary flag and its removal condition in [REFACTOR.md](REFACTOR.md);
- update the immutable worklog entry, compact benchmark artifact, scoreboard,
  and benchmark changelog;
- include host, hardware, model, quant, workload, command, result, and
  correctness gate in every performance claim.

### Step 7: Re-audit after structural changes

A new layout, producer, cache policy, or owner invalidates old local optima.
Rerun focused sweeps for:

- prefetch depth;
- row/column tile;
- local size;
- split count;
- chunk size;
- queue count;
- graph amortization floor.

Do not rerun every historical idea. Reopen only parameters whose mechanism the
structural change affected.

## 13. hipEngine implementation map

Use these paths when applying the guide:

| Area | Source of truth |
| --- | --- |
| Architecture and plugin design | [PLAN.md](PLAN.md) |
| Kernel inventory, lineage, and path map | [KERNELS.md](KERNELS.md) |
| Strict and production numerical contracts | [EXECUTION-PROFILES.md](EXECUTION-PROFILES.md) |
| Fixtures, oracles, and validation tiers | [TESTING.md](TESTING.md) |
| Benchmark protocols and evidence policy | [BENCHMARK.md](BENCHMARK.md) |
| W7900 hardware/roofline detail | [ROOFLINE.md](ROOFLINE.md) |
| Strix Halo hardware/roofline detail | [ROOFLINE-gfx1151.md](ROOFLINE-gfx1151.md) |
| General accumulated lessons | [LESSONS-LEARNED.md](LESSONS-LEARNED.md) |
| Prefill design | [PREFILL.md](PREFILL.md) |
| PM4 transport | [PM4.md](PM4.md) |
| Backend policy and selectors | `hipengine/kernels/hip_gfx1100/__init__.py`, `hipengine/kernels/hip_gfx1151/__init__.py` |
| HIP kernel bodies | `hipengine/kernels/hip_gfx1100/` and architecture-specific siblings under `hipengine/kernels/hip_gfx1151/` |
| Torch-free JIT and cache keys | `hipengine/core/build.py` |
| Runtime ownership and GGUF dispatch | `hipengine/runtime/` |
| CPU correctness oracle | `kernels/cpu_reference/` |
| Accepted and rejected benchmark evidence | `benchmarks/results/`, `benchmarks/CHANGELOG.md`, and immutable `worklog/entries/` |

The kernel registry is keyed by `(backend, layer, quant, variant)`. Add
architecture and shape policy through registration and backend capability data;
do not add backend or quantization branches to engine/model dispatch code.

Before porting a kernel, run the lineage check described in
[KERNELS.md](KERNELS.md). External repositories are read-only lineage and idea
sources. New kernels, tests, profiling, and promotion evidence belong in this
tree.

## 14. Further reading

Start with these documents for details deliberately omitted here:

- [ROOFLINE.md](ROOFLINE.md): W7900 cache, compute, memory, dispatch, and
  per-family analysis.
- [ROOFLINE-gfx1151.md](ROOFLINE-gfx1151.md): Strix Halo geometry, unified
  memory, local roofs, and architecture bring-up.
- [KERNELS.md](KERNELS.md): active variants, source lineage, build profiles, and
  the current optimal path map.
- [EXECUTION-PROFILES.md](EXECUTION-PROFILES.md): exact ownership and numerical
  promotion rules.
- [LESSONS-LEARNED.md](LESSONS-LEARNED.md): detailed case studies, including
  negative results.
- [SOL-OPTIMIZATION.md](SOL-OPTIMIZATION.md): speed-of-light campaign method.
- [GGUF-PREFILL-OPTIMIZATION.md](GGUF-PREFILL-OPTIMIZATION.md): staged prefill
  optimization.
- [STRIX-HALO-LLAMACPP-REVIEW.md](STRIX-HALO-LLAMACPP-REVIEW.md): comparative
  `gfx1151` source and profiler review.
- [VLLM_RDNA3.md](VLLM_RDNA3.md): external implementation patterns that were
  evaluated for transfer.

The durable lesson across all of them is simple: tune the bytes, ownership,
layout, and resident execution of the real operation first. Architecture
intrinsics and submission mechanisms become valuable only after that foundation
is measured and correct.
