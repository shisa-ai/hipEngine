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

It is also the entry point for the accumulated tuning lessons of both trees. The
rules are consolidated here; the case studies, derivations, and campaign
evidence behind each one remain in the documents listed in section 14, starting
with [LESSONS-LEARNED.md](LESSONS-LEARNED.md).

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
13. **Bound the prize before building.** A component's share of complete wall
    multiplied by its plausible speedup is the ceiling on the end-to-end gain.
    Compute it, and run the cheapest probe that could disprove the premise,
    before implementing.
14. **Confirm the work volume held constant.** Non-finite outputs, collapsed
    routing, repeated tokens, and silent fallback all produce speedups by
    removing real work. Gate the comparison on work volume, not only on time.

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

A power envelope is part of the host identity. The same `gfx1151` silicon in a
60 W laptop chassis and in a 140 W desktop chassis are separate measurement
lanes with separate noise floors: roughly 10% clock-driven run spread in the
60 W lane against roughly 1% in the 140 W lane. A 1.2% prefill-chunk win
measured on the 60 W lane did not reproduce on the 140 W lane and was below the
60 W lane's own noise floor. Establish a lane's noise floor before promoting any
percent-scale result, and never carry an absolute rate between lanes.

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

Useful first-pass classification thresholds:

- arithmetic intensity below about 10 usually indicates a memory-bound
  operation; above about 50 usually indicates a compute-bound one;
- more than about four ALU operations per encoded weight element indicates a
  dequantization-overhead-bound kernel rather than a bandwidth-bound one;
- for a packed rowtile that reuses one weight read across `M` rows, useful
  arithmetic intensity rises approximately as `2M / bpw`, where `bpw` is bytes
  per encoded weight value.

Two symmetric errors are common. Measuring below a bandwidth roof does not prove
the operation is not bandwidth-limited: low occupancy, cache-line waste, poor
channel utilization, and dequantization issue pressure all reduce attainable
bandwidth. And sub-linear wall growth across batch widths does not identify the
limiter. Packing eight rows measured 4.20x aggregate throughput at 1.92x round
wall, which proves weight and metadata amortization but says nothing about the
mechanism; a fitted "fixed" term contains the weight stream, dequantization
setup, device kernels, dispatch, and synchronization, and cannot be relabeled
host overhead from scaling alone.

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

Distinguish rows that arrive from independent concurrent requests from rows that
arrive from speculative verification of one sequence. Both present as a wider
`M` to the kernel, but only the first amortizes weight reads into useful
throughput unconditionally. Section 6.7 covers when the second is worth doing at
all.

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
win, and the reason is structural rather than tunable: grid dimensions do not
control block scheduling order on RDNA3. Reshaping a grid from
`(packs, total_rows)` to `(packs, num_experts)` to hold one expert's weights in
L2 measured 59% slower, because blocks sharing a `blockIdx.y` are not
co-scheduled, the added per-expert token loop serialized work that had been
parallel across blocks, and the smaller grid removed the thread-level
parallelism that was hiding memory latency. Guaranteeing expert order requires a
cooperative launch with device-wide synchronization, a persistent kernel with an
atomic work queue, or separate launches. Measure selected-expert locality and
full wall rather than assuming L2 reuse.

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

Warm up the code path and clocks before timed samples. The gap is large enough
to invert conclusions: identical 512-token prefill measured about 458 tok/s cold
and about 1,930 tok/s on the next pass, and 4K prefill moved from about 2,142 to
about 3,255 tok/s, almost all of it JIT and allocator warmup. Keep an explicit
cold row when first-request latency is the subject, but never mix cold and warm
rows in one speed table. Use repeated, counterbalanced baseline/candidate runs.
Report median and tails when variance matters; do not promote a result smaller
than uncontrolled run-to-run movement without stronger paired evidence.

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

On the current stack, `MemUnitBusy` reads zero for every sampled dispatch in a
production c8 profile and is unusable. `SQ_WAVES` is populated but is exactly
grid-derived, so it confirms launch geometry and is not independent evidence
about residency. Do not build an occupancy theory from either.

When counters cannot supply bandwidth, compute achieved rates from exact
resident tensor ledgers instead: divide each kernel family's encoded weight
bytes by its summed duration. One c8 production profile ranked Q4-pair, Q4
singleton, Q6, and Q5 projections at 508.0, 387.4, 322.2, and 245.5 GB/s this
way. Label such numbers as lower-bound traffic rates. They exclude activations,
metadata, overfetch, and cache reuse, and they are not hardware-counter
bandwidth.

Reported utilization is not a work signal either. A `gfx1151` long-prefill stall
holds the GPU at approximately 100% reported activity and 2.9 GHz while drawing
only 41-59 W against a roughly 120 W working regime, with device memory stable
and no fault, timeout, or reset logged. High utilization at low power means the
device is not doing useful work. The complementary signature, a hang at 0%
activity with no error, is usually a stale JIT cache rather than a kernel
defect.

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

### 4.7 Confirm that work volume held constant

The two largest false results in this lineage were both speedups produced by
doing less real work, and both were visible in signals the harness already
recorded but did not gate on.

A prefill baseline measured 4,509 tok/s while emitting entirely non-finite
logits; NaN propagation had collapsed expert routing, so the model skipped most
of its matrix work. That baseline was the comparator for a WMMA kernel, which
therefore looked like a 6% to 44% regression across shapes. Re-gating the
comparison on finite outputs reversed the verdict: WMMA was ahead by 78% at 512
tokens and behind by 9% at 4K. Separately, an incorrect RoPE configuration made
4K prefill measure 3,231 tok/s while routing to roughly 24 active experts per
layer instead of roughly 213, and generating a repeated token.

Before comparing two speed rows, confirm the candidate performed the same
quantity of work:

- outputs are finite, with NaN and Inf counted rather than only detected;
- route diversity or active experts per layer, or the equivalent fan-out
  measure for the model;
- unique generated tokens, and agreement with a known seed row;
- graph-versus-eager token agreement wherever replay is involved;
- for speculative paths, the share of work that ran as ordinary autoregressive
  fallback;
- peak memory, since a row that fits differently is not the same row.

Put these in the summary table, not only in the raw record. Both failures above
sat in saved JSON for weeks while the summary column a reviewer actually reads
showed throughput alone.

Top-1 agreement does not certify calibration. One smoke found native and
reference tokenization matching exactly across a long document while perplexity
was about 1.97e6 against 9.054. Use top-1 as a cheap tripwire and KL,
perplexity, or the profile gate for the numerical claim itself.

### 4.8 Keep the evaluator separate from the candidate

An optimization loop is most dangerous when it can modify both the candidate and
the scorer. Freeze the evaluation surface before the first candidate, and record
or hash the benchmark and profiler scripts, the objective extractor, prompt and
shape fixtures with their train and heldout split, oracles and profile
thresholds, baseline and comparator commands, timing boundaries, and the
required route manifest. The loop may read those files; it must not edit them
while optimizing.

If the evaluator proves wrong or is missing a field, stop the exploration, fix
and commit the evaluator as its own logical unit, refresh the baseline, and
reopen against the new evaluator identity. Never repair the scorer and the
candidate inside one keep-or-revert iteration, because the result cannot then be
attributed to either.

Long adaptive searches overfit their own measurements without any deliberate
gaming. Where the workload permits, separate three surfaces:

| Surface | Used for | Rule |
| --- | --- | --- |
| Discovery | fast iteration on representative shapes | chooses what to investigate; never supports publication |
| Qualification | all production roles, boundary and tail shapes, full suites | a candidate may displace an incumbent only after this |
| Confirmation | one frozen finalist, fresh process, counterbalanced order | no candidate edits after the run begins |

If a confirmation failure leads to further tuning, that surface has become
development evidence and the modified candidate needs a new clean confirmation.
A set repeatedly tuned against is no longer a heldout set, whatever it is
called. Not every exact leaf kernel needs three separate files; the principle is
separation of selection from final evidence. For sampling, speculative, routing,
and adaptive policies the committed train and heldout suites are mandatory.

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

As a working ladder for a bandwidth-bound decode kernel, at most 96 VGPRs allows
16 waves per SIMD and ample latency hiding; 192 allows 8 and remains adequate;
above 256 allows only 4-5 and starts to starve the memory controller; 1-2 waves
per SIMD is critically undersubscribed and can drop effective bandwidth to
30-40%. Treat any allocation above about 128 VGPRs as worth inspecting. A
low-row WMMA kernel can launch hundreds of blocks and remain latency-bound if
roughly 200-250 VGPRs per thread permit only a few waves per issue slot. In that
case, reducing the accumulator tile can outperform adding more blocks. Treat
any sharp VGPR rise as a possible occupancy-class transition and verify it with
resources plus timing.

Grid size also has an upper limit past which extra blocks buy nothing and cost
dispatch. A W7900 GEMV at 104 VGPRs reaches about half occupancy, roughly 14
workgroups per CU, so the machine fills at about 1,350 workgroups; the same
kernel's verifier grids ran 49x to 97x that depth. Before shrinking such a grid,
decide which kind of excess it is. Blocks that repeat identical work collapse
for free: a GDN recurrence launching one block per `(v_head, dv_idx)` repeated
the same q/k loads, reductions, and transcendentals across 128 blocks per head.
Blocks that each compute a unique output do not; folding them into an output
tile trades grid size for per-block serial work and must be measured per shape.

Any runtime knob that changes thread or block size must be validated against the
kernel's `__launch_bounds__`, its statically allocated shared memory, and its
reduction scratch size. Wrappers that accepted a 256-thread request for kernels
compiled with `__launch_bounds__(128, 4)` produced HIP unspecified launch
failures during otherwise ordinary sweeps. Fall back to the compiled default for
out-of-contract values and keep one smoke test covering rejected values; failing
safely is better than treating an invalid knob as a benchmark candidate.

Keep private scratch at zero on hot paths unless a measured exception justifies
it. A tiny spill inside a deep K loop can dominate the kernel.

### 5.4 Unroll for independent work, not source size

Deep K-loop unrolling can expose independent loads and arithmetic to the
scheduler. It is especially useful when the compiler otherwise emits a serial
load-dequant-accumulate chain. But unrolling also increases live values and
code size.

The default hypothesis for a simple dot or FMA loop of the form
`for (k = tid; k < N; k += blockDim.x)` is manual vec8 unrolling with a tail
loop, applied when `N / blockDim.x` falls below roughly 64 iterations per
thread. HIP and ROCm did not expose enough instruction-level parallelism at
those trip counts on their own. In the parent lineage this was the largest
single decode-era kernel family win: vec4 across the W8A16 kernels gave roughly
+42% decode and vec8 added a further +8%. Checks at vec16 showed diminishing
returns, so treat vec8 as the plateau rather than the first rung of a ladder.

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

Check instruction availability before designing an integer path. On `gfx1100`,
`__builtin_amdgcn_sdot4` and `__builtin_amdgcn_udot4` require the unavailable
`dot1-insts` feature and do not compile; mixed signed/unsigned
`__builtin_amdgcn_sudot4` uses `dot8-insts` and is the available route, which is
what llama.cpp HIP uses for RDNA3 Q4/Q8 matrix-vector work. `v_dot8_i32_iu4`
exists for INT4-packed layouts and `v_dot2_f32_f16` for FP16 pairs. Signedness
and the bias or minimum fold must match the quantization math exactly, and a
builtin that compiles is not evidence of speed: inspect the ISA for real
`v_dot4*` instructions, zero scratch, and healthy VGPR counts, then measure the
exact shape.

Two arithmetic assumptions fail on this architecture. INT8 and FP16 WMMA both
run at 512 operations per cycle per CU, so INT8 is not a 2x compute path; a
measured `4096^3` comparison on W7900 gave about 84.8 TFLOP/s BF16 against about
75.3 TOP/s INT8. Integer wins therefore come from lower memory traffic and
better residency, not from the multiply. And `gfx11` has no native FP8 hardware,
so FP8 decode and dequantization are software bit manipulation while INT8 has
native dot and conversion paths; prefer INT8 for 8-bit storage work unless
targeting RDNA4 or later.

Integer matrix paths can reduce weight and activation bytes, but their
end-to-end value depends on:

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

Two encoding facts bound what VOPD can do. Published FP32 peak assumes full VOPD
pairing, so the unpaired rate is roughly half of it; a roofline built on the
headline number overstates the compute roof for any dependent chain. And
`v_dot4*` instructions are VOP3-encoded and do not participate in VOPD at all,
so a dot-based path is not competing for those issue slots. The
shift-mask-subtract-convert-multiply dequantization chain has little VOPD
opportunity because its steps depend on each other; replacing that chain with a
dot instruction removes work rather than pairing it better.

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

### 5.13 Treat a vendor library as one banded candidate

A vendor or third-party library is a candidate owner with a shape band and an
architecture gate. It is neither a default nor a floor, and two retained results
point in opposite directions.

Default rocBLAS beat `TORCH_BLAS_PREFER_HIPBLASLT=1` on tested BF16 GEMM shapes
on this W7900 stack: 84.8 against 71.2 TFLOP/s at `4096^3`, and 71.0 against
51.7 TFLOP/s at `8192^3`. Guidance carried from another architecture or an older
ROCm release does not survive here, and this comparison should be repeated when
the stack changes. When a library route is retained, pin the exact solution
index so the result depends on recorded evidence rather than on the library's
own heuristics.

AOTriton's tiled flash attention wins on `gfx1100` above 512 prefill tokens and
loses below it, which is where the retained 512-token threshold comes from. On
`gfx1151` there is no crossover: the native scan is 2-5% faster at every prefill
length from 64 to 2048, because the library's tiling targets larger GPUs and its
wrapper adds a query conversion, a head-major KV copy, and a stream bridge that
a 40-CU part cannot amortize. The same library and the same model produced
opposite verdicts on two RDNA3 architectures.

Measure the library against the native owner on the real shape band and the real
architecture, keep whichever wins behind an explicit threshold, and record that
threshold as measured evidence rather than as a policy preference.

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

Size the workgroup from the smallest `K` in the kernel's dispatch set. Threads
whose loop never executes are pure overhead, and the effect is severe at small
`K`: llama.cpp HIP's Q4_K matrix-vector path maps thread groups to quant blocks,
so at `ncols=512` only 32 of 256 threads enter the useful inner loop and 224
idle. That structural difference, not an instruction gap, is most of why a
64-thread single-wave shape beat a 256-thread block on the same expert-down
shape. As a starting point, `K=512` suits 64 threads and `K=2048` suits 128; a
global 256 is usually wrong at one end of the set. An independent RDNA3 engine
reached the same conclusion from a different direction: hipfire's `gfx1100` Q4
matrix-vector path uses 32-thread workgroups with `__launch_bounds__(32, 16)`,
packed `uint32_t` nibble loads, and four independent FP32 accumulators, and does
not default to dot instructions either.

The rule inverts for a latency-bound kernel with no dead lanes. A `gfx1151`
short-context attention leaf carrying an inherited `gfx1100` 256-thread geometry
was 6-26% faster at 1,024 threads across contexts 256-1024. Block geometry is
the first inherited constant to re-derive when moving between architectures, and
the answer comes from the shape rather than from the previous host.

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

Two attention-specific traps recur. First, a populated grid can still hide
serial work: a split-K path with enough workgroups assigned each token's
256-wide QK dot to a single thread, and making that dot wave-cooperative was
worth 1.12x at 4K and 1.62x at 128K before any layout change. Audit work
distribution inside the block after fixing grid coverage. Second, a one-pass
streaming rewrite in the FlashAttention style changes reduction order and is
easy to mistake for an exact speedup. One grouped online-softmax prototype
showed the right direction at 128K and then failed exact graph-versus-eager
token agreement at 32K and at `decode_len=1`. Build a fixture comparing
producer outputs, split partials, top logits, and greedy tokens against the
retained exact path before wiring such a rewrite end to end; without it the
candidate is an approximate-attention path, not an exact default.

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

Price the fusion before writing it. Removing a small-grid launch recovers only
the per-launch dispatch floor, on the order of 5.6 microseconds, which a staging
kernel can easily lose to barrier spin and a staged round trip through memory:
two bit-exact staged-rotate fusions that removed roughly 68 and 146 launches per
pass both regressed the verify cycle. A launch that is already overlapped
recovers nothing at all; fusing a gate multiply that occupied about 1
microsecond of a 95 microsecond attention call was flat to +0.7% at the leaf and
within noise end to end.

Some pairs cannot be fused at all under the current grid. A router logits kernel
using one block per expert to keep occupancy high cannot absorb a top-k that
needs every expert logit for a token: without inter-block synchronization the
fused form is either racy or collapses to one block per token, recreating the
undersubscription the split was designed to fix.

Do not fuse merely to reduce a kernel count if the result raises VGPR/LDS enough
to slow the dominant operation. Every fused composite needs a registered strict
unfused chain.

### 6.7 Speculative decode and verification economics

Speculative decoding, multi-token prediction, and tree drafting are throughput
trades, not kernel optimizations. Their outcome is decided by one measurable
property of the target model, and that property should be measured before any
drafter or verifier kernel work begins.

Define verification efficiency:

```text
eta = Verify(B) / (B * AR_cost)
```

`Verify(B)` is the cost of verifying `B` candidate positions in one pass, and
`AR_cost` is one ordinary autoregressive decode step on the same host, model,
quantization, and context. A target that streams the same weights once for all
`B` positions approaches `eta ~ 1/B`. A target whose verification cost grows
with `B` approaches `eta ~ 1`, at which point verifying is no cheaper than
decoding. Break-even follows directly:

```text
speedup = committed_per_step * AR_cost / (draft_cost + B * eta * AR_cost)
speedup > 1  requires  committed_per_step > draft_cost / AR_cost + B * eta
```

Measured bands and what they imply at `B=8`:

| `eta` | Decision | Typical class |
| --- | --- | --- |
| below 0.20 | speculate; expect 2.5-4x | dense 7B-70B, strongly bandwidth-bound |
| 0.20-0.40 | probably speculate; 1.5-2.5x | small MoE, dense hybrids |
| 0.40-0.60 | marginal; requires high acceptance | large MoE, 64-128 experts |
| 0.60-0.80 | probably not; requires above 85% acceptance at every position | high-cardinality MoE with sequential state |
| above 0.80 | do not speculate; verification costs more than decoding | effectively sequential verification |

Four properties drive `eta`: expert count and routing diversity, top-k,
sequential state layers such as linear attention or Mamba that force
per-position processing, and how bandwidth-bound the target already is. Larger
and more compressed dense targets have better `eta` because their autoregressive
step is already dominated by a single weight stream.

This produces a result that is easy to get backwards. A 35B-A3B target with 256
experts, top-8 routing, and 30 of 40 layers recurrent measured `eta = 0.736`,
and its speculative path ran at 0.7-0.85x its own autoregressive baseline, while
a dense 27-32B target on the same card projects 2.5-4x. Sparsity had already
made the autoregressive step cheap and simultaneously made verification
expensive. A sparse model has less to gain from speculation and pays more to get
it. Tree drafting amplifies both directions, because every tree node still
requires its own sequential expert evaluation.

Break-even is often unreachable rather than merely distant. At `eta = 0.72` with
`B = 8`, even a free drafter needs roughly 6.9 accepted tokens out of 8, about
86% acceptance at every position, while acceptance decays with depth by
construction. Compute that requirement before tuning acceptance.

An equivalent formulation helps when the drafter is fixed: express one verify
cycle in autoregressive-token-equivalents, `C_B = cycle wall / AR_cost`, and
require `C_B` to fall below the visible tokens the cycle emits. One B=3 verifier
at `C_B = 4.67` emitting 2.38 visible tokens per cycle needed its 45.1 ms cycle
to reach about 17.9 ms, which turned an open-ended tuning problem into a
launch-budget problem.

Report three ledgers, never one:

- **acceptance economics:** proposed, accepted draft, correction, bonus or root,
  and committed tokens per iteration;
- **verified throughput:** same-session autoregressive tok/s, speculative tok/s,
  verifier time, drafter time, proposal or tree time, commit and restore time,
  and synchronization counts;
- **fallback coverage:** how much work silently ran as ordinary autoregressive
  decode.

The third ledger exists because of a specific failure mode: a speculative path
can appear faster by degrading into mostly autoregressive fallback. Acceptance
alone never proves speed, and a verifier-derived `off` or `B0` row is
diagnostic. A speedup claim requires a true no-speculation autoregressive
baseline measured under the same protocol.

One further diagnostic separates kernel work from cycle economics. Measure the
wall reduction required on a short fixed cell and on the full prompt suite
separately. When the short cell needs 25% and the full suite needs 57% for the
same target speedup, the binding cost is prompt activation and repeated
partial-accept cycle behavior, and no amount of verifier kernel tuning will
close it.

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

Runtime reserves can be large and are tunable. ROCr 7.2.4 reserves 140 MiB of
single-dispatch scratch per process and GPU by default, with dispatches above
that limit using a use-once scheme. Lowering the homogeneous `gfx1100` default
to 8 MiB through `HSA_SCRATCH_SINGLE_LIMIT` recovered 132 MiB of unused reserve
while retaining full-engine behavior. The variable must be applied before
`libamdhip64` loads, and an explicit user value should always win over an engine
default.

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

HIP graph replay reduces repeated submission overhead, but the effect on this
stack is small and capture and instantiation must amortize over enough
transitions. Section 7.10 gives the measured per-launch numbers. In production
terms, graph replay improved a matched `gfx1151` Q4_K_M wall by 1.00% at 512
tokens, 0.86% at 4K, and 0.36% in a bounded 128K confirmation, and was neutral
to worse on a `gfx1100` verifier at over 900 nodes. Treat replay as a modest,
measured saving that carries a state contract, not as the remedy for a
launch-bound path.

A graph key must include all state that affects pointer identity or kernel
arguments, including:

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

The compute frontend bounds what overlap can buy. The chip has 8 ACEs and the
queue manager connects one queue per pipe at a time, so about 8 kernels can be
in flight simultaneously; a single HIP stream uses one. Two bandwidth-bound
branches will not exceed the single-stream bandwidth ceiling between them, so
overlap pays only when the branches use complementary resources. `rocprofv3`
kernel traces carry a queue ID, so pipe spread is directly verifiable rather
than inferred.

Queue count is host- and workload-specific, and no single setting is a fix.
`GPU_MAX_HW_QUEUES=1` was an initial mitigation for a `gfx1151` long-prefill
stall and is documented as risk reduction only: the stall still reproduces with
one queue, with SDMA disabled, and under both tested HIP 7.13 and 7.15
user-space stacks. A later full `gfx1151` matrix selected two queues for an
independent MoE branch. Neither is a universal RDNA3 setting.

### 7.10 Model the dispatch floor

Launch cost on this platform is large, measurable, and only partly removable.
Measured on W7900 with a graph-node microbenchmark:

| Grid blocks | Direct (us/launch) | Graph (us/launch) | Graph speedup |
| ---: | ---: | ---: | ---: |
| 1-64 | 5.61 | 5.61 | 1.00x |
| 1024 | 7.25 | 6.32 | 1.15x |
| 2048 | 7.95 | 7.09 | 1.12x |
| 4096 | 9.36 | 8.50 | 1.10x |
| 8192 | 12.34 | 11.31 | 1.09x |

Three facts follow. Per-launch cost is roughly 5.6 microseconds plus a term that
grows with grid size, because the cost is command-processor and workgroup
scheduling rather than submission alone. It is independent of argument count:
moving from 2 to 16 kernel arguments added 0.0 microseconds, so it is not
argument marshaling. And HIP graph replay is close to neutral in steady state,
because graphs amortize PM4, doorbell, and MES work but not the per-dispatch MEC
and SPI cost. That last point is a platform difference rather than a tuning gap.
CUDA graph node replay costs roughly 1-2 microseconds per node; ROCm 7.x replay
costs about what a direct launch costs. Do not carry a CUDA graph expectation
onto this stack.

Dispatch cost does not appear in a kernel trace. `rocprofv3 --kernel-trace` sums
GPU-active `DurationNs` only, so the interval between kernels is invisible in
the CSV while being real on the wall clock. Measure it as a replay delta,
complete wall minus summed kernel time over a marker-scoped window, or with a
dedicated dispatch microbenchmark. Never infer it by attaching an assumed
per-launch model to a `DurationNs` sum.

Turn the floor into a budget. For a target step wall `W` and a measured
per-launch cost `c`, `W / c` is the launch ceiling even at zero kernel time. One
verifier at roughly 20 microseconds per launch against a 17.9 ms target could
afford about 500 launches while issuing 971, which made launch count rather than
kernel time the binding constraint. A budget computed this way tells you whether
to tune kernels or restructure the path.

Measure launch reductions in one batch. Removing 30 to 80 launches at a time
sits inside run-to-run noise, so a sequence of individually unmeasurable commits
produces no attributable evidence. Consolidate the change and land it against
one measurement.

Finally, different regimes usually run different code. An autoregressive decode
path consolidated into fused decode-shaped kernels does not confer that
consolidation on a verifier path running the `tokens > 1` shapes; one measured
971 launches per pass while the other could not have afforded more than a few
hundred. Take a launch and kernel census per path, not per model.

### 7.11 Evaluate persistent kernels by working set

A persistent kernel that replaces `N` launches with `N` in-kernel stages is an
attractive answer to the dispatch floor, and it is the right answer for exactly
one class of stage.

A cooperative-launch grid barrier is cheap. Measured on W7900 with a
cache-resident stage, `grid.sync()` costs about 1 microsecond, well below a
dispatch boundary. The decision therefore turns entirely on whether the stage is
dispatch-bound or memory-bound, which the last-level cache boundary decides:

| Stage working set | Regime | Persistent versus N launches |
| --- | --- | ---: |
| 0.5-2 MB | sub-cache, dispatch-bound | 6-13x |
| 16 MB | sub-cache | 1.48x |
| 64 MB | cache edge | 1.08x |
| 128-256 MB | beyond last-level cache, external-memory-bound | 0.93-0.96x |

A microbenchmark that re-reads one buffer overstates the win. Repeating the test
so each stage streams a distinct fresh slice, which is what weight-streaming
decode actually does, gives 1.27x at a 3 MB slice, 1.15x at 6 MB, and 1.08x at
12 MB. That is the recoverable dispatch gap, not a transformative speedup.

The consequence is specific. Fuse or eliminate the small-grid glue, the
rotations, norms, router steps, and format casts whose working sets sit inside
cache. Do not consolidate the large memory-bound projections into a megakernel:
they already run near effective streaming bandwidth, and a persistent form would
pay barrier and lost-launch-overlap cost to consolidate work that is already
efficient. Ordinary fusion reaches the same glue with far less ABI and lifecycle
risk.

This also corrects a common reading of a low utilization number. When a decode
path shows a small fraction of peak bandwidth, the big-grid kernels are usually
not the underutilized part. The token wall is, because of the dispatch intervals
between kernels. Attribute the shortfall before designing around it.

One structural constraint bounds the design space: HIP does not support dynamic
parallelism for this use, so a persistent kernel cannot device-launch existing
`__global__` kernels. Every inner loop it needs must be extracted or rewritten
as a device-callable helper, which is the real cost of the approach.

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
- A long-prefill no-progress hazard is open on this architecture
  (`ROCm/ROCm#6437`). It is intermittent, reproduces with a single queue and
  with SDMA disabled, and constrains both published long-context rows and any
  proposal to add another low-level queue or transport path.

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
| "A microbenchmark with pre-filled inputs predicts end-to-end" | An 8x `grouped_mm` microbenchmark became a 16% end-to-end loss and +1.6 GiB peak once dequantization, stacking, and allocator pressure were included | What does the complete pipeline cost, staged one layer at a time? |
| "Reorder `blockIdx` to get cache locality" | Grid dimensions do not control block scheduling order on RDNA3; the reshape measured 59% slower | Does the schedule need a cooperative launch, a persistent work queue, or separate launches? |
| "A fast row is a fast row" | Non-finite logits and collapsed expert routing each produced large apparent speedups by removing real work | Did work volume hold constant: finite outputs, route diversity, unique tokens, fallback share? |

One cross-backend comparison did survive matched controls and should be kept
separate from the rest. On `gfx1100`, Vulkan command-buffer replay holds a
measured 2.44x-10.12x advantage over HIP graph replay for serialized tiny
dispatches. That is a submission result, not a shader-compiler result, and it is
consistent with the dispatch model in section 7.10: the gap is in how work is
submitted, not in the code the two backends generate. Production-shaped Q4 and
Q6 quantize-and-dot controls on the same hardware favor HIP.

## 12. A repeatable tuning procedure

### Step 1: Establish the contract

- Name the model, quant, profile, operation, exact shape band, context, and host.
- Identify strict versus production arithmetic requirements.
- Trace the registered owner and fallback.
- Add a RED fixture before changing math or state when practical.
- State the maximum prize: the component's measured share of complete wall, the
  roofline or Amdahl ceiling that share implies, and the expected end-to-end
  range. Mark every value as measured, derived, or estimated.
- Fix the GO and STOP thresholds, the required matrix cells, and the variant or
  time budget before seeing any candidate result.

The prize check is `bucket_fraction * expected_speedup`. If that product is not
worth the work, the bucket is the wrong target however improvable the kernel is.
Rank competing candidates by milliseconds recoverable from complete wall rather
than by leaf speedup: a call-weighted family saving that does not survive the
complete step is not a priority. Before the audit was enforced in the parent
lineage, roughly a hundred iterations went into micro-optimizing a paged
attention kernel that launched on 16 of 96 CUs while a different family owned
the majority of decode time.

Where a cheap probe can disprove the premise, run it first. If removing an
entire cost class from the path cannot move the target by a meaningful margin,
that cost class is not the bottleneck, and the probe costs far less than the
implementation it avoids. If a threshold must change later, log a dated
correction naming the invalid premise and rerun the control; do not reinterpret
an old result under a newly convenient bar.

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
8. Is the block actually parallel inside? A reduction, scan, or top-k guarded by
   `if (threadIdx.x == 0)` in a hot kernel is a defect, not a style choice, and
   it leaves the grid looking populated while the work is serial.
9. Are the per-thread loops long enough for the compiler to schedule? Below
   roughly 64 iterations per thread, expect to unroll manually.
10. Is synchronization density excessive? A `__syncthreads()` inside a token or
    tile loop can degrade scheduling well beyond the kernel's own share of time.

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

### Step 8: Know when to stop

An optimization loop needs a termination rule as much as a starting point. Pivot
to a different lane when retained wins fall into the 1% range while larger
structural lanes remain open, or after roughly three consecutive non-improving
attempts on the same mechanism. A run of correct, memory-neutral, sub-1% keeps
is acceptable polish, but it does not substitute for the structural lane that
would change the bottleneck class.

Maintain three to five conceptually distinct hypothesis families rather than a
single-incumbent hill climb: one low-risk improvement close to the incumbent,
one that changes dataflow, layout, ownership, or algorithm, and at least one
that would invalidate the current framing if it succeeded. Preserve the concept
behind a rejected candidate and record the new fact that would reopen it; do not
preserve candidate debris.

Stopping a design does not discard its parts. An exact, measured, same-suite
non-regressive component win is retained and promoted on its own merits even
when the larger design or the headline target is abandoned.

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
| Speculative-decode economics and gates | [SPECULATIVE-DECODE.md](SPECULATIVE-DECODE.md), [MTP.md](MTP.md), [DFLASH.md](DFLASH.md) |
| Dispatch floor, launch census, persistent kernels | [MEGAKERNEL.md](MEGAKERNEL.md) |
| Sprint contracts, prize framing, and stop rules | [PROCESS-IMPROVEMENT.md](PROCESS-IMPROVEMENT.md) |
| Exploration firewall and evaluation-set discipline | [PROCESS-EXPLORATION.md](PROCESS-EXPLORATION.md) |
| Concurrent serving, admission, and c=N economics | [CONCURRENCY2.md](CONCURRENCY2.md) |
| Tuning knobs and their measured defaults | [ENVS.md](ENVS.md) |
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

For the topics this guide compresses to a page or two:

- [SPECULATIVE-DECODE.md](SPECULATIVE-DECODE.md): the full `eta` decomposition,
  break-even derivation, per-architecture projections, and the procedure for
  measuring `eta` on a new model.
- [MTP.md](MTP.md) and [DFLASH.md](DFLASH.md): the multi-token-prediction and
  draft-verify campaigns, their launch budgets, and their do-not-chase lists.
- [MEGAKERNEL.md](MEGAKERNEL.md): the launch census, the measured dispatch
  model, the grid-reduction analysis, and the persistent-barrier microbenchmark
  that closed the megakernel program.
- [PROCESS-IMPROVEMENT.md](PROCESS-IMPROVEMENT.md): the sprint brief and
  measure-first experiment contract behind step 1.
- [PROCESS-EXPLORATION.md](PROCESS-EXPLORATION.md): the evaluation firewall,
  discovery/qualification/confirmation sets, hypothesis beam, and stagnation
  triggers behind sections 4.8 and step 8.
- [CONCURRENCY2.md](CONCURRENCY2.md) and [CONCURRENCY.md](CONCURRENCY.md):
  batched serving, scheduler ownership, admission policy, and the c=N scaling
  interpretation used in section 2.4.
- [GFX1151-TUNING-LANDSCAPE.md](GFX1151-TUNING-LANDSCAPE.md) and
  [TUNING-gfx1151.md](TUNING-gfx1151.md): the inherited-constant audit, the
  AOTriton crossover measurements, and the ranked `gfx1151` candidate ledger.
- [TUNING-gguf.md](TUNING-gguf.md) and
  [GGUF_DECODE_REPACK.md](GGUF_DECODE_REPACK.md): the GGUF tuning lanes and the
  tile-major decode slab layouts behind the row-amortized owners.
- [LLAMACPP-HIP-PARITY.md](LLAMACPP-HIP-PARITY.md) and
  [HIP-vs-VULKAN.md](HIP-vs-VULKAN.md): matched cross-implementation evidence,
  including which structural deltas transferred and which did not.
- [DEBUG-GFX1151-STALL.md](DEBUG-GFX1151-STALL.md): the open long-prefill
  no-progress hazard, its controls, and its containment path.
- [QUANTS.md](QUANTS.md) and [KVCACHE.md](KVCACHE.md): format coverage, quality
  cliffs, capacity math, and KV precision policy.
- [OOM.md](OOM.md): startup accounting, runtime reserves, and capacity failure
  modes.
- [PM4.md](PM4.md): the transport contract and the qualification bar an
  alternative submission path must clear.

The durable lesson across all of them is simple: tune the bytes, ownership,
layout, and resident execution of the real operation first. Architecture
intrinsics and submission mechanisms become valuable only after that foundation
is measured and correct, and a speedup is only real once the work it performed
has been shown to be unchanged.
