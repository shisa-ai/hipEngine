# ROOFLINE.md — RDNA3 W7900 Performance Model for LLM Inference

_Ported from `~/amd-gpu-tuning/docs/ROOFLINE.md` (upstream last updated 2026-05-10). Kernel R&D evidence referenced below lives in that parent workspace and is not vendored into hipEngine; path-qualified pointers keep the references navigable._

A standalone technical reference for understanding where performance comes
from and where it is lost on AMD Radeon Pro W7900 (gfx1100 / RDNA3) during
local LLM inference. Covers hardware capabilities, memory hierarchy,
instruction throughput by type, per-operation roofline analysis, and measured
gaps. Written so that someone with basic understanding of memory bandwidth and
compute throughput can follow the full reasoning.

Companion docs (all under `~/amd-gpu-tuning/`):
`PLAN-PAROQUANT.md` (optimization roadmap), `LESSONS-LEARNED.md`
(hard-won rules), `WORKLOG.md` (chronological evidence), and
`docs/LLAMACPP-VULKAN.md` (llama.cpp HIP vs Vulkan source analysis).

## How to read this document

- **If you only read one section:** §5 (Amdahl per-bucket decode analysis) —
  the single most important framing for any optimization decision.
- **If you are debugging a specific slow kernel:** §10 (decision tree) starts
  from profiler output and walks to an intervention class.
- **If you are tempted to try dp4a, LDS staging, or wave32 reductions:** §11
  (what not to chase) lists experimentally rejected approaches with evidence.
- **If you are new to the hardware:** read §1 (hardware) and §2 (roofline
  fundamentals) first. Then skim §3 to understand why c=1 decode is
  memory-bound and how it differs from prefill and c>1.
- **If you are writing a new benchmark or making roofline claims:** §8
  (measured vs assumed) and §12 (profiling reference) show which intuitions
  have been falsified and how to collect real counter data.

---

## Table of Contents

1. [Hardware: W7900 / gfx1100 / RDNA3](#1-hardware-w7900--gfx1100--rdna3)
2. [Roofline Fundamentals](#2-roofline-fundamentals)
3. [Regimes: c=1 Decode, c>1 Decode, Prefill](#3-regimes-c1-decode-c1-decode-prefill)
4. [Our Model: Qwen3.5-35B-A3B PARO W4A16](#4-our-model-qwen35-35b-a3b-paro-w4a16)
5. [The Amdahl Reality: Per-Bucket Decode Analysis](#5-the-amdahl-reality-per-bucket-decode-analysis)
6. [Why W4 Doesn't Mean 2× W8](#6-why-w4-doesnt-mean-2-w8)
7. [Context-Length Scaling](#7-context-length-scaling)
8. [Measured vs. Assumed](#8-measured-vs-assumed)
9. [RDNA3 Gotchas and Architecture Rules](#9-rdna3-gotchas-and-architecture-rules)
10. [Optimization Decision Tree](#10-optimization-decision-tree)
11. [What Not To Chase](#11-what-not-to-chase)
12. [Profiling Reference](#12-profiling-reference)
13. [References and Further Reading](#references-and-further-reading)

---

## 1. Hardware: W7900 / gfx1100 / RDNA3

### 1.1 Die Configuration

Navi 31 is a **chiplet GPU**: 1× Graphics Compute Die (GCD, 5nm) + 6× Memory
Cache Dies (MCDs, 6nm) connected via Infinity Fabric. The GCD contains all
compute (CUs, shader engines, command processor, L2). The six MCDs each hold
a 16 MB slice of Infinity Cache plus a 64-bit GDDR6 memory controller, for a
total of 96 MB L3 and a 384-bit GDDR6 bus. All VRAM traffic crosses the
GCD↔MCD boundary, which adds a small latency on top of raw GDDR6 latency.

From `rocminfo` on this system:

| Property | Value | Source |
|---|---|---|
| GPU | AMD Radeon Pro W7900 | rocminfo |
| Architecture | RDNA3, gfx1100, Navi 31 (chiplet: 1×GCD + 6×MCD) | ISA name + AMD docs |
| Compute Units (CUs) | 96 | rocminfo |
| SIMDs per CU | 2 (SIMD32) | rocminfo |
| Total SIMD32 units | 192 | 96 × 2 |
| Shader Engines | 6 | rocminfo |
| Shader Arrays per Engine | 2 | rocminfo |
| Work Group Processors (WGPs) | 48 | PyTorch reports 48 "multiprocessors" |
| CUs per WGP | 2 | standard RDNA3 |
| Default Wavefront Size | 32 | rocminfo |
| Max Waves per CU | 32 | rocminfo (16 per SIMD) |
| Max Workgroup Size | 1024 threads | rocminfo |
| Max Work-items per CU | 1024 | rocminfo |
| Boost Clock | 2499 MHz | AMD product spec |
| HSA-reported max clock | 1760 MHz | `rocminfo`; not the product boost clock |

**SIMD organization.** Each CU contains 2 SIMD32 units. Each SIMD32 executes
one instruction per cycle across 32 lanes (threads) simultaneously. A wave32
wavefront occupies one SIMD; a wave64 wavefront spans both SIMDs in a CU
(CU mode). Max 16 waves per SIMD32 means 16 independent instruction streams
sharing one SIMD's resources.

**WGP mode vs CU mode.** RDNA3 can run wave32 and wave64 code paths, but the
details are build-profile and compiler dependent. hipEngine's gfx1100 decode
profile uses `-mcumode` without `-mwavefrontsize64`, so device code should
treat `warpSize == 32` as the default. CU mode and wavefront size are
orthogonal. llama.cpp HIP on gfx1100 also defaults to wave32 in the checked
path. The Vulkan driver on the same W7900 reports subgroup size 64. Do not
infer shuffle or reduction correctness from the nominal wave size alone; we
have observed gfx1100 shuffle/reduction candidates that behaved like 32-lane
halves even under wave64-oriented builds.

### 1.2 Memory Hierarchy

Sizes are from `rocminfo`. Bandwidth and latency values are from the Chips
and Cheese microbenchmarks of the 7900 XTX (same Navi 31 GCD, faster GDDR6
memory than W7900), adjusted where relevant. Use them as a useful ceiling,
not as measurements from this project's kernels.

| Level | Size | Scope | Measured BW (7900 XTX) | Measured latency (cycles) |
|---|---|---|---|---|
| VGPR file | 1536 × 32-bit per SIMD32 (192 KB/SIMD, 384 KB/CU) | Per-SIMD | Operand speed (register file) | 0 (operand) |
| LDS (GL0 share) | 64 KB per CU (128 KB per WGP) | Per-CU/WGP | ~10–20 TB/s aggregate (order-of-mag) | ~1–2 |
| L0 scalar | ~16 KB | Per-CU | — | small |
| L1 vector (GL1) | 32 KB | Per-CU / per-Shader-Array | **6.14 TB/s** | **24** |
| L2 (GL2) | 6 MB | Shared across GCD | **2.88 TB/s** | **131** |
| L3 / Infinity Cache (MALL) | 96 MB (16 MB × 6 MCDs) | Distributed across MCDs | **2.30 TB/s** | **612** (includes chiplet cross) |
| VRAM (GDDR6) | 48 GB | Global, behind MCDs | 864 GB/s theoretical peak (W7900: 18 Gbps × 384-bit); 7900 XTX measured ~960 GB/s at 24 Gbps | **~1045** (includes GCD↔MCD↔GDDR6) |

**Chiplet implication.** L2 (6 MB) lives on the GCD. L3 (96 MB Infinity Cache)
is distributed across six MCD chiplets. Every L3 miss, every L2 miss, and
every VRAM access crosses the GCD↔MCD Infinity Fabric boundary, which adds
roughly ~90 cycles over an equivalent monolithic design (Chips and Cheese
measured RDNA 2 Navi 21 L3 at 521 cycles vs. RDNA 3 Navi 31 at 612). This is
small relative to the ~1045-cycle VRAM round-trip, but it's why scattered
access patterns are punished more severely on chiplet RDNA3 than on monolithic
RDNA2.

Key implications for inference:

- **L3 (Infinity Cache) is 96 MB.** For a W4 model with 3B active params ×
  0.5 bytes = 1.5 GB/tok of weight traffic, the entire active weight set for
  one decode token vastly exceeds L3. Weights stream from VRAM on every token.
  No cache level saves us from external VRAM bandwidth for weights.
- **L2 is 6 MB.** Small projections (`2048→256` = 256 KB W4 weights, or
  `512→2048` = 512 KB) can potentially hit L2 on repeated access within one
  token. But the standard large projections (`2048→8192` = 8 MB) will not.
- **LDS is 64 KB per CU.** Shared memory for intra-workgroup communication.
  On RDNA3, LDS is not free: using LDS requires synchronization barriers that
  stall the wavefront, and on bandwidth-bound kernels this coordination cost
  can exceed the savings from data reuse (see Section 9).
- **VGPR pressure determines occupancy.** 1536 VGPRs per SIMD shared across
  all active waves on that SIMD. A kernel using 96 VGPRs/wave allows 16 waves
  (max occupancy); one using 192 VGPRs/wave drops to 8 waves; one using 256+
  drops to ~5–6 waves. Lower occupancy means fewer in-flight memory requests,
  less latency hiding, and lower effective bandwidth — critical for
  memory-bound decode kernels.
- **~1045-cycle VRAM latency.** At 2.5 GHz that's ~418 ns per miss. For a
  decode kernel to stay bandwidth-bound, it must keep hundreds of memory
  requests in flight across all CUs to hide that latency. This is the
  mechanistic reason occupancy matters: a kernel with 4 waves/SIMD has ~1/4
  the outstanding-request capacity of one with 16 waves/SIMD.

### 1.3 Compute Throughput by Instruction Class

Based on the architectural layout (96 CUs × 2 SIMD32 × 32 lanes, 2.499 GHz
boost) and AMD product specifications:

| Instruction class | Throughput | Spec value | Notes |
|---|---|---|---|
| **FP32 FMA (with VOPD)** | 1 FMA/lane/cycle × VOPD 2-issue | **61.3 TFLOP/s** | Requires VOPD dual-issue (compatible op pairs). Without VOPD: ~30.7 TFLOP/s. |
| **FP16 packed vector** | Packed FP16 FMA in 32-bit lanes | **~123 TFLOP/s peak class** | Product FP16 peak is 2x FP32. Actual kernel speed still depends on packing, issue, and data layout. |
| **FP16 / BF16 WMMA** | Tile 16×16×16 via SIMD | **123 TFLOP/s** | Matrix multiply-accumulate. Only useful when M ≥ 16. |
| **INT8 dot product (`v_dot4`)** | 4 int8 MACs/lane/cycle | **123 TOPS** | `v_dot4_i32_iu8` (sudot4, mixed signed/unsigned). Single-issue VALU, no VOPD. |
| **INT4 dot product (`v_dot8`)** | 8 int4 MACs/lane/cycle | **245 TOPS** | `v_dot8_i32_iu4` (dot5-insts). Requires INT4-packed operands. |
| **BF16 WMMA (measured)** | `torch.matmul` 4096³ | **84.8 TFLOP/s** | 69% of 123 spec (occupancy, overhead) |
| **INT8 WMMA (measured)** | `torch._int_mm` 4096³ | **75.3 TOPS** | 61% of 123 spec |

**Critical insight for c=1 decode:** WMMA (the "matrix" instruction) is
useless for M=1. A single output token produces a `[1, out_features]` vector.
WMMA tiles are 16×16×16 — using WMMA for M=1 would waste 15/16 of the output
tile. All c=1 decode math must use scalar/vector instructions (FP32 FMA, INT8
`v_dot4`, or packed FP16 `v_pk_fma`), not matrix cores.

**INT8 dp4a vs FP32 FMA for decode.** Both are VALU instructions, but dp4a
does 4 multiply-accumulates per lane per cycle versus 1 FMA per lane per
cycle for FP32 (or 2 if VOPD pairs). In ops/cycle, INT8 dp4a is 2–4× FP32
FMA depending on VOPD utilization. However: dp4a consumes 4 bytes of weight
per instruction, FP32 FMA consumes one dequantized element from a nibble. The
comparison is not "4× throughput" but "4× MACs/instruction × how many
instructions are needed per weight element × dequant cost × layout cost."

### 1.4 Memory Bandwidth

| Property | Value |
|---|---|
| VRAM type | GDDR6 with ECC |
| Bus width | 384-bit |
| Data rate | 18 Gbps effective |
| Theoretical peak | **864 GB/s** |
| Back-calculated active-weight effective BW | ~150 GB/s for whole-token PARO W4 at 100 tok/s and 1.5 GB/tok |
| Back-calculated llama.cpp HIP Q8 active-weight BW | ~232–258 GB/s (77.4 tok/s `llama-bench tg4096` × 3.0–3.33 GB/tok) |
| PCIe link | Not a decode roofline limiter once weights are resident |

(An earlier back-calc from the historical `/completion` server row at
71.49 tok/s gave ~214–238 GB/s. The current `llama-bench` reference is faster;
the current back-calc is the one to compare against.)

**Why measured bandwidth is below theoretical.** Real memory access patterns
deviate from the perfectly sequential, maximally coalesced reads that achieve
864 GB/s. Factors include:

- GDDR6 page/bank conflicts on non-sequential patterns
- L2 miss overhead and cache line waste (128-byte cache lines vs. actual used bytes)
- Partial memory channel utilization on small transfers
- Reduced occupancy from high VGPR usage → fewer outstanding memory requests
- GDDR6 controller scheduling inefficiency under mixed read/write workloads

A well-written bandwidth-bound GPU kernel on this hardware typically achieves
**75–85%** of theoretical peak for large sequential streams. Smaller working
sets, scattered access patterns, or low-occupancy kernels drop to 40–60%.

### 1.5 Clock and Thermal Behavior Under Sustained Load

The 864 GB/s peak assumes the memory clock (`mclk`) stays at its boost state.
The compute ceiling (61.3 / 123 TFLOP/s etc.) assumes the shader clock (`sclk`)
stays at its boost state. Real sustained inference loads can cause either to
drop:

- `sclk` (GPU core clock) may throttle under sustained >90% utilization when
  the GPU hits thermal or power limits (W7900 TBP 295 W).
- `mclk` (memory clock) generally holds at boost for decode but can step down
  on long idle gaps between dispatches; short-burst benchmarks may see a
  different effective bandwidth than a 5+ minute sustained run.
- `rocm-smi --showclocks` during a long-running benchmark verifies whether
  boost is actually sustained. Idle values (`sclk=36 MHz`, `mclk=96 MHz`) mean
  the driver has deep-clocked; running values should show the boost level.
- For 4K/4K or longer sustained decode, assume effective peak bandwidth is
  closer to **~750–820 GB/s** rather than the full 864, until you confirm
  otherwise with live counters.

This is a second-order effect relative to dequant/layout/occupancy losses,
but it is worth checking when a benchmark row that used to reproduce now
reports lower numbers without a code change.

### 1.6 Command Processor and Dispatch Path

Every HIP/ROCm kernel launch goes through a chain of microcontrollers and
hardware queues before a single wave hits the shader complex. Understanding
this path is the prerequisite for reasoning about per-dispatch overhead and
why graph replay helps.

**The micro-engines (all RS64 = RISC-V RV64I with AMD custom instructions):**

| Engine | Role | Relevant to compute? |
|---|---|---|
| PFP (Pre-Fetch Parser) | Graphics command parsing | No; idle during compute |
| ME (Micro Engine) | Graphics draw frontend | No; idle during compute |
| MEC (Micro Engine Compute) | Parses PM4/AQL packets, drives compute pipes | **Yes**; 2 MECs, 4 pipes each = 8 ACEs |
| MES (Micro Engine Scheduler) | Maps user queues onto HW queues; manages priority/quantum | **Yes** (indirect) |
| RLC (RunList Controller) | Broader execution context mgmt | Indirect |

**The dispatch chain, simplified:**

```
1. Application calls hipLaunchKernel (or graph replay submits captured packets)
2. Runtime writes a PM4 DISPATCH_DIRECT or AQL dispatch packet into a ring
   buffer in VRAM
3. Runtime writes a doorbell (MMIO poke) corresponding to that queue
4. MES firmware (or the pipe's queue manager directly) notices the doorbell:
     • If the user queue is already mapped to a HW queue on a pipe,
       QueueManager HW picks it up on its next arbitration cycle.
     • If unmapped, MES firmware processes the work and maps the user queue
       onto an available HW queue (more overhead; measured in microseconds).
5. The pipe's QueueManager selects one "connected queue" to feed the shader
   complex. Only one connected queue per pipe runs at a time.
6. The MEC parses the next PM4/AQL packet, sets up COMPUTE_* registers, and
   issues the dispatch to SPI (Shader Processor Interface).
7. SPI allocates waves across SIMDs on available CUs per the kernel's
   launch bounds, VGPR/SGPR/LDS requirements.
```

**Per-dispatch cost structure:**

| Cost component | Typical magnitude | Amortizable by graph replay? |
|---|---|---|
| PM4 packet build (host CPU) | ~200 ns–1 µs | Yes (captured once) |
| Ring-buffer write + doorbell MMIO | ~0.5–1 µs | Yes |
| MES scheduling decision (unmapped queue) | several µs | Yes if queue stays mapped |
| QueueManager arbitration | fraction of a µs | Not directly, but reduced per-token by fewer launches |
| MEC packet parse + COMPUTE_* register setup | ~0.5–1 µs | Partially (registers may stay set) |
| SPI wave allocation on shader complex | fraction of a µs | No (actual launch cost) |

This is why **graph replay is a per-dispatch overhead mitigation, not a
per-kernel math speedup**. Capturing a graph bakes the PM4 packet sequence
into a pre-built ring buffer, so one doorbell ring can drive a long stream
of dispatches. The MES scheduling decision is amortized across the whole
graph (the queue stays mapped for the life of the capture). The MEC and SPI
still do per-dispatch work, which is why graph replay in our PARO work was
`+2.6–3.9%` on 4K decode rather than the larger win it would be if we were
dispatch-bound end-to-end.

**Budget sanity check.** At 100 tok/s decode with ~1600 HIP dispatches per
token (llama.cpp rocprof observation for similar workloads), the budget is
~6 µs per dispatch. A realistic per-dispatch overhead of 1–3 µs consumes
17–50% of the budget. This puts a hard floor on how fast c=1 decode can get
without fusing kernels to reduce launch count. Reducing ~1600 dispatches/tok
to ~800 would roughly halve the dispatch cost, offering a larger E2E win
than making any individual kernel 2× faster.

**Pipe capacity is largely unused in single-stream inference.** The chip has
8 compute ACEs (pipes), each capable of running a different "connected
queue" in parallel. Single-HIP-stream inference puts all dispatches on one
queue ↔ one pipe, so 7/8 of the compute frontend is idle. Multi-stream HIP
can in principle overlap independent kernels (e.g., attention on stream A,
MoE expert prep on stream B), but they still share the CU pool and L2, so
the win requires the two streams to actually have independent data paths.
See §9.8 for the gotcha list.

---

## 2. Roofline Fundamentals

The roofline model classifies every computation by its **arithmetic intensity**
(AI): the ratio of compute work to memory traffic.

```
AI = FLOPs (or OPs) / Bytes transferred
```

A kernel hits one of two ceilings:

- **Compute-bound:** AI is high enough that the processor runs out of ALU
  cycles before it runs out of data. Performance is limited by FLOP/s.
- **Memory-bound:** AI is low enough that the processor runs out of data
  before it runs out of ALU cycles. Performance is limited by bytes/s.

The **ridge point** is the AI at which both ceilings bind equally:

```
Ridge point = Peak FLOP/s / Peak Bytes/s
```

For W7900 with INT8 dp4a (our target compute path for W4 decode):

```
Ridge point = 123 TOPS / 864 GB/s = 142.4 OPs/byte
```

For FP32 FMA (our *current* compute path):

```
Ridge point = 61.3 TFLOP/s / 864 GB/s = 70.9 FLOPs/byte
            (or 30.7 / 864 = 35.5 FLOPs/byte without VOPD)
```

### How to interpret for a W4 GEMV

A c=1 W4 matrix-vector multiply for shape `[in_features, out_features]`:

- **Bytes read:** `in_features × out_features / 2` (W4 weights) + `in_features × 2`
  (BF16 activation) + scales/zeros (~1/group overhead, typically 1/128)
- **Compute:** `2 × in_features × out_features` FLOPs (multiply-accumulate)
- **AI ≈** `2 × in × out / (in × out / 2)` ≈ **4 FLOPs/byte** (for W4, ignoring activation and scale overhead)

At AI = 4, we are deep in the **memory-bound** regime (well below the 35–142
ridge point). This means:

> For c=1 decode GEMVs at W4 precision, compute throughput is essentially
> irrelevant. Performance is determined by how efficiently we can stream weight
> bytes from VRAM through the caches into the ALU. The ALU will always be
> waiting for data.

This is why "W4 should be 2× faster than W8" has theoretical merit: at the
same effective bandwidth, W4 moves half the bytes per token. But it also means
any overhead that reduces effective bandwidth (dequant instructions, scattered
layout, LDS synchronization, low occupancy) directly caps performance, and
no amount of "faster math" compensates.

---

## 3. Regimes: c=1 Decode, c>1 Decode, Prefill

The same model running on the same GPU exhibits three very different
performance regimes depending on how many tokens are processed per forward
pass. The roofline for each is set by different ceilings.

| Scenario | Weight reuse | Dominant ceiling | W7900 ideal class |
|---|---|---|---|
| c=1 decode | 1× (each weight read → one output element) | Memory bandwidth | ~576 tok/s W4 |
| c=4–8 decode | 4–8× per weight load | Transitional | ~432 tok/s ×c output cols |
| Prefill (4096 tokens) | 4096× | Compute (matrix cores / WMMA) | ~14,000 tok/s |

### 3.1 c=1 Decode (Our Primary Target)

At batch size 1 (one token being generated), the computation graph for each
decode step is dominated by matrix-vector multiplies (GEMVs) that read the
full active weight set once:

```
One decode token:
  For each of ~40 layers:
    Read active MoE expert weights (8 of 256 experts × gate/up/down)
    Read shared expert weights
    Read dense attention/linear-attention projections
    Read KV cache (grows with context)
    Small elementwise ops (RMSNorm, gating, rotation, routing)
  Read lm_head weight (final projection to vocabulary)
```

The key property: **every weight byte is used exactly once per token** in c=1.
There is no reuse across batch dimension, so the GPU cannot amortize a
weight-matrix load across multiple output columns.

For c=1, the **ideal decode rate** is purely a bandwidth calculation:

```
Ideal tok/s = Effective_Bandwidth / Active_Weight_Bytes_Per_Token
```

Any other overhead (dequant, attention, routing, barriers, dispatch, Python) is
additive time that reduces the effective rate below this ceiling.

### 3.2 c>1 Decode (Batched Serving)

With `c` concurrent decode columns (`c` independent sequences generating
tokens simultaneously), each weight load feeds `c` output dot products. This
shifts the roofline:

```
Bytes per weight load: same as c=1
Ops per weight load: c × (2 × K)   (multiply-accumulate per output column)
AI: grows linearly with c
```

**W7900 ceiling estimates for PARO W4A16 at different c values:**

| c | Active weight bytes | Activation bytes | Total BW/tok | 864 GB/s aggregate tok/s | Per-column tok/s |
|---|---:|---:|---:|---:|---:|
| 1 | 1.5 GB | small | ~1.5 GB | 576 | 576 |
| 4 | 1.5 GB | ~4× activation | ~1.55 GB | ~2230 | ~558 |
| 8 | 1.5 GB | ~8× activation | ~1.6 GB | ~4320 | ~540 |
| 16 | 1.5 GB | ~16× activation | ~1.7 GB | ~8130 | ~508 |

In the ideal bandwidth regime, per-column throughput decays slowly while
aggregate throughput scales near-linearly. In practice, the crossover
between memory-bound and compute-bound happens around c=8–16 for current
RDNA3 INT8 dp4a paths:

- At c ≤ 4, the regime is still memory-bound; c=1 kernels extended naively
  to multiple columns typically reload weights and waste most of the
  theoretical gain.
- At c = 4–8, memory reuse becomes meaningful; dp4a / block-aligned layouts
  and shared activation quantization become strictly more attractive than at
  c=1.
- At c ≥ 16, compute may start to bind for some kernel shapes. MoE expert
  kernels with small intermediate (512) still stay memory-bound longer than
  dense models would.
- llama.cpp MMVQ has a hard `MMVQ_MAX_BATCH_SIZE = 8` cutover to matrix-matrix
  (MMQ) shaders above 8 columns. Treat c > 8 as a different kernel family
  entirely (matrix tiling, WMMA-friendly), not an extension of the c=1 GEMV.

**Implications for PARO:** the kernel-level design choices that win at c=1
(one-wave, no-LDS, single-row) do not automatically extend to c>1. Any c>1
variant must tile output columns deliberately, cache the quantized activation
once and reuse across columns, and be aware of growing VGPR pressure (`c ×
rows_per_block` accumulators). PARO's c=1 promotion gate does not currently
require c>1 proof; when c>1 becomes an active target, it is treated as a
separate kernel family.

### 3.3 Prefill (Compute-Bound)

Prefill processes a full prompt in one forward pass. Every weight matrix is
read once, but it participates in a dot product with all `N` prompt tokens
simultaneously. Weight reuse is `N` (often 512–4096×), which drives AI far
above the roofline ridge point and puts the ceiling on compute throughput, not
bandwidth.

For Qwen3.5-35B-A3B at 4K prompt:

| Metric | Value | Notes |
|---|---|---|
| Active params | ~3.0B | A3B, 8 of 256 experts |
| FLOPs per token | ~6.0 GFLOP | `2 × active_params`, rough main-linear work |
| FLOPs per 4K prompt | ~24 TFLOP | `4096 × 6.0 GFLOP` |
| BF16 WMMA peak (spec) | 123 TFLOP/s | |
| BF16 WMMA measured (4096³ `torch.matmul`) | 84.8 TFLOP/s | 69% of spec |
| Prefill ceiling @ 123 TFLOP/s | ~20,500 tok/s | `123e12 / 6e9` |
| Prefill ceiling @ 84.8 TFLOP/s (measured) | ~14,133 tok/s | Realistic upper bound |
| PARO prefill @ 4K | 3,207 tok/s | 22.7% of measured ceiling |
| llama.cpp HIP prefill @ 4K | 2,159 tok/s | 15.3% of measured ceiling |
| llama.cpp Vulkan prefill @ 4K | 1,577 tok/s | 11.2% of measured ceiling |

**Prefill losses are compute-utilization losses**, not bandwidth losses:

- Kernel overheads: launch cost, reduction cost, barrier stalls
- WMMA tile coverage: small shapes (e.g., MoE expert intermediate 512) don't
  fill 16×16×16 tiles efficiently
- Compiler scheduling (the `-amdgpu-unroll-threshold-local=600` llama.cpp
  finding was a compute-bound win, +166% prefill, that did nothing for
  bandwidth-bound decode)
- Dequant work at W4 that would be amortized over batched dots but still
  costs instruction slots
- MoE dispatch overhead per expert (only 8 of 256 are active, but routing
  + expert dispatch adds per-token overhead)

**Why prefill has much more headroom than decode:**

Prefill is at ~23% of its realistic compute ceiling. Decode is at ~17% of
its theoretical bandwidth ceiling. The gap framing matters: decode is closer
to a hard wall (bandwidth doesn't increase with better kernels, just effective
utilization), while prefill could plausibly gain 2–4× from better WMMA
utilization, fewer launches, and compiler tuning before hitting the compute
wall.

The practical implication is that **prefill and decode optimizations do not
transfer directly**:

- Prefill wants: large WMMA tiles, high arithmetic intensity, maximally
  packed matrix operations, aggressive unrolling, fewer launches per tile.
- Decode wants: minimal VALU instructions per weight byte, single-wave
  reductions, coalesced loads, low VGPR pressure, no LDS synchronization.

A kernel tuned for prefill is typically bad at decode (too many threads
per output, wrong reduction shape, WMMA tile waste). A kernel tuned for
decode is typically bad at prefill (undersubscribed grid, no matrix-core
usage, high relative launch overhead).

---

## 4. Our Model: Qwen3.5-35B-A3B PARO W4A16

### 4.1 Architecture Parameters

From the model config and our bench scripts:

| Parameter | Value |
|---|---|
| Total parameters | 34.66B |
| Active parameters per token | ~3.0B (A3B: 8 of 256 experts active) |
| Hidden dimension | 2048 |
| Num attention heads (Q) | 16 |
| Num KV heads | 2 |
| Full-attention head dimension | 256 |
| Linear-attention key/value head dimension | 128 |
| Num layers | 40 (30 linear-attention + 10 full-attention) |
| Full-attention pattern | Every 4th layer (layers 3, 7, 11, ...) |
| Expert count | 256 |
| Active experts per token | 8 |
| Expert intermediate size | 512 |
| Shared expert intermediate | 512 |
| MoE routing | Top-8 softmax |

### 4.2 Weight Traffic Per Token

For PARO W4A16 quantization, weight traffic must be read as a whole-model
estimate. The first three rows below are per-layer shapes; the all-layer
contribution is shown explicitly because the model has 40 layers.

| Component | Bytes per token | Notes |
|---|---|---|
| MoE selected experts (gate+up+down) | **12.6 MB/layer**, **~503 MB across 40 layers** | 8 experts × 3 projections, W4 packed |
| Shared expert (gate+up+down) | **~3.1 MB/layer**, **~126 MB across 40 layers** | Current decode uses W8A16 shared expert; original BF16 residency would be ~2x |
| Dense projections (q/k/v/o, per layer) | varies by layer type | Linear-attention + full-attention |
| Linear-attention projections | ~30 layers × assorted small projs | Many small `2048→*` shapes |
| Full-attention q/k/v/o | ~10 layers × mixed GQA shapes | q/gate is much wider than hidden; k/v use 2 KV heads × 256 dim; o maps Q heads back to hidden |
| lm_head | 2048×248320 ≈ **509 MB** | W8A16 int8 weight; old BF16 residency would be ~1.02 GB |
| Router weights | 40 × 2048×257 × 2 ≈ **40 MB** | BF16, cached |
| **Estimated total active weight traffic** | **~1.2–1.5 GB** | Order of magnitude |

### 4.3 Theoretical Decode Ceilings

These rows estimate the whole-model active bytes per token under each
quantization scheme. "W4A16 (PARO)" is a mixed-format entry: W4 on experts
and most dense projections, W8A16 on `lm_head` and the shared expert, BF16
on the router. Roughly 45% of the 1.5 GB total is not pure W4. The label
keeps the scheme name (PARO) because the serving path is mixed by design.

| Quantization | Est. active bytes/tok | 864 GB/s ceiling | 700 GB/s effective |
|---|---:|---:|---:|
| W4A16 (PARO, mixed) | ~1.5 GB | **576 tok/s** | 467 tok/s |
| W8A8 (Quark) | ~3.0 GB | **288 tok/s** | 233 tok/s |
| Q8_K (GGUF) | ~3.3 GB | **262 tok/s** | 212 tok/s |
| Q4_K (GGUF) | ~1.7 GB | **508 tok/s** | 412 tok/s |

**Current actual performance:**

| Path | Measured tok/s | Fraction of 864 GB/s ceiling | Optimistic active-weight BW |
|---|---:|---:|---:|
| PARO W4A16, 4K/4K | 100.0 | 17.4% | ~150 GB/s |
| W8A8, 4K/4K | 97.2 | 33.8% | ~292 GB/s |
| llama.cpp HIP Q8, 4K/4K | 77.4 | 29.5% | ~255 GB/s if 3.3 GB/tok |
| llama.cpp Vulkan Q4, 4K/4K | 122.2 | 24.0% | ~208 GB/s if 1.7 GB/tok |

**The core problem is visible here:** PARO W4A16 achieves only 17.4% of its
theoretical bandwidth ceiling, while W8A8 achieves 33.8% of its ceiling.
W4 should be much faster than W8 in absolute tok/s, but in practice both
paths are roughly equal because PARO is losing more of its theoretical
bandwidth advantage to overhead.

---

## 5. The Amdahl Reality: Per-Bucket Decode Analysis

### 5.1 Amdahl's Law Applied

If a bucket is X% of total decode time, the maximum E2E speedup from
optimizing only that bucket is bounded by:

```
Max_speedup = 1 / (1 - X/100)
```

And if you speed up that bucket by S×:

```
E2E_speedup = 1 / ((1 - X/100) + (X/100) / S)
```

### 5.2 Current Decode Profile (PARO W4A16, 4K context, graph replay)

From rocprofv3 kernel trace, post-fusion (PLAN-PAROQUANT.md §Profile). These
shares come from subtracting a matched 4K/0 prefill trace from a 4K/128 trace.
Use the shares for ranking; profiler overhead makes the traced tok/s lower than
the headline benchmark row.

| Bucket | Time share | Amdahl ceiling (∞ speedup) | Amdahl at 2× | Amdahl at 4× |
|---|---:|---:|---:|---:|
| W4 GEMV family | 33.8% | 1.51× (151 tok/s) | 1.20× (120) | 1.34× (134) |
| Paged full-attention | 18.6% | 1.23× (123 tok/s) | 1.10× (110) | 1.16× (116) |
| W8A16 GEMV (lm_head + shared) | 14.9% | 1.18× (118 tok/s) | 1.08× (108) | 1.13× (113) |
| PARO rotations | 8.4% | 1.09× (109 tok/s) | 1.04× (104) | 1.07× (107) |
| Router logits/select | 6.2% | 1.07× (107 tok/s) | 1.03× (103) | 1.05× (105) |
| GDN decode | 4.8% | 1.05× (105 tok/s) | 1.02× (102) | 1.04× (104) |
| Other (RMSNorm, gating, residual) | ~13.3% | — | — | — |

**Key takeaway:** Even making W4 GEMV infinitely fast only reaches ~151 tok/s.
To reach significantly higher (200+ tok/s), multiple buckets must improve
simultaneously. The remaining 66% of decode time is non-W4 work that sets a
hard floor.

### 5.3 Dispatch Overhead: The Hidden Bucket

The kernel-trace profile in §5.2 misses a significant component: the time
between kernel launches (dispatch overhead) is invisible in per-kernel
time accounting but real on the wall clock.

**Replay-only dispatch profile (4K/128, step graph replay, rocprofv3):**

This is the cleanest current dispatch measurement. It uses
`--decode-use-step-graph-replay --decode-step-graph-skip-validation`, then
subtracts a matched 4K/0 prefill-only trace. A 16-token
version is useful for quick iteration, but graph-capture setup is visible
enough there to overstate steady replay dispatch. The 128-token version below
amortizes capture setup and is the current source of truth after both the
linear-attention QKV/Z and full-attention Q/K pack8 fusions.

| Metric | Value |
|---|---:|
| 4K/128 trace dispatches | 150,033 |
| 4K/0 prefill-only dispatches | 35,630 |
| Decode-delta dispatches | 114,403 |
| Replay-path dispatches per token | **893.8** |
| Decode-delta kernel time | 1024.448 ms |
| Kernel time per token | 8.003 ms |
| Profiled decode wall time | 1534.543 ms |
| Profiled decode wall per token | 11.989 ms |

Top replay-visible dispatch counts. Times are decode-delta totals over the
128-token trace:

| Kernel family | Dispatches/tok | Decode-delta time |
|---|---:|---:|
| W8A16 lowp/final/shared GEMV | 86.1 | 159.295 ms |
| W4 generic pack8 GEMV | 53.1 | 76.493 ms |
| Fused dual W4 pack8 GEMV | 42.5 | 122.397 ms |
| Selected gate/up W4 GEMV | 42.5 | 93.192 ms |
| Selected down W4 GEMV | 42.5 | 66.661 ms |
| PARO single rotation | 170.0 | 96.385 ms |
| RMSNorm / add-RMSNorm | 128.6 | 89.366 ms |
| Router logits + select | 85.0 | 59.464 ms |
| Paged/full-attention context/reduce | 21.3 | 199.658 ms |
| Other glue / Torch elementwise / copies | 222.3 | 60.704 ms |

**Older mixed-trace profile (4K/128, step graph replay, rocprofv3):**

| Metric | Value |
|---|---|
| Total 4K/128 trace dispatches | 381,652 |
| 4K/0 prefill-only dispatches | 35,630 |
| Estimated decode dispatches | ~346,000 |
| Mixed-trace dispatches per timed decode token | **~2,703** |
| Measured replay-path dispatches per token | **893.8** current, **909.4** after QKV/Z pack8 fusion, **939.4** before projection fusion |
| Median inter-kernel gap | 3.76 μs |
| Mean inter-kernel gap | 32.93 μs |
| Dispatches < 10 μs (overhead class) | 70.6% |
| Dispatches < 5 μs (pure overhead) | 52.6% |

Important caveat: this trace was taken through
`scripts/bench_paro_native_engine.py --decode-use-step-graph-replay`. That
harness does more than timed graph replay: after prefill it runs a full normal
eager decode for correctness, a full graph-compatible eager decode for
correctness, then the timed replay loop. The default warmup also includes a
short eager decode. Subtracting a 4K/0 prefill-only trace therefore removes
prefill work but not the two validation decodes. The `~2,703` number is a
mixed-trace normalized count, not a clean serving-loop replay count. Dividing
by the roughly three decode passes in the harness gives an inferred
replay-only order of `~880-1,000` dispatches/token. The first dedicated
replay-only trace gave `939.4` replay-path dispatches/token; after QKV/Z pack8
fusion the same 16-token subtraction gave `909.4`; after full-attention Q/K
fusion the longer 128-token subtraction gives `893.8`.

Per-token dispatch breakdown (mixed trace, normalized by timed decode tokens):

| Category | Dispatches/tok |
|---|---:|
| W4 generic GEMV | 398 |
| RMSNorm | 279 |
| Rotation (single) | 245 |
| W8A16 (lowp + shared) | 245 |
| Router (logits + select) | 245 |
| Selected dual gate/up | 123 |
| SiLU/mul (unfused) | 123 |
| Selected down | 123 |
| Shared expert combine | 123 |
| SiLU+rotate (fused) | 123 |
| Torch elementwise | 98 |
| GDN decode | 92 |
| Linear-attn conv | 92 |
| Rotation (double) | 92 |
| Dense GEMV | 92 |
| fp16→fp32 cast | 61 |
| Write paged KV | 31 |
| Rotation (triple) | 31 |
| Paged attention | 20 |
| **Total** | **~2,703** |

At the measured `893.8` replay-path dispatches/token, the conservative
`1 μs/dispatch` floor is `~0.89 ms/token`, roughly `9%` of the 4K/128 token
budget. This is large enough to treat fusion as a first-order lever, but it is
not proof that dispatch dominates W4 GEMV arithmetic. The clean trace also
shows that many dispatch-heavy families are individually cheap; fuse them only
when the fusion preserves grid parallelism and removes real memory traffic.

**Implication for optimization priority:** Dispatch overhead is a
first-order lever comparable to individual kernel buckets, but the current
measurement should be treated as an upper-bound steering signal. If replay-only
dispatch is `~894/tok`, reducing it by 30-50% through fusion saves
`~0.3-0.5 ms/token` at `1 μs/dispatch` before overlap, with larger upside if
the effective serial gap is higher. This makes fusion competitive with W4 GEMV
arithmetic improvements, especially when a fusion also removes memory traffic
or a small Torch op.

The largest fusion targets in the clean trace, ranked by dispatch count and
plausible safety:

1. **RMSNorm**: 279/tok across ~7 separate norm kernels per layer.
   Fuse only single-use add+norm or producer/consumer boundaries. Do not
   naively fold a shared normalized vector into every projection row and
   recompute the norm per output block.
2. **MoE per-layer**: ~12 dispatches for router+gate/up/SiLU/down/combine.
   Fusing router into a single topk+select+gather would eliminate ~3.
3. **Torch elementwise + fp16→fp32 cast**: 160/tok of PyTorch autograd ops.
   Absorbing `.float()` and elementwise ops into custom HIP kernels.
4. **Rotations**: 368/tok of rotation launches. Fusing into adjacent GEMVs
   would eliminate the separate launch (the SiLU/down fusion is already
   an example of this pattern working).

### 5.4 Per-Bucket Roofline Status

For each major bucket, we assess whether it's limited by bandwidth, compute,
occupancy, or overhead/dispatch:

#### W4 GEMV (33.8%)

| Property | Value |
|---|---|
| Dominant shapes | `2048→8192`, `2048→4096`, `2048→512`, `512→2048`, `2048→256` |
| Arithmetic intensity | ~4 FLOP/byte (W4 weight dominated) |
| Regime | **Deeply memory-bound** (AI ≪ ridge point) |
| Current inner loop | FP32 FMA over per-element dequantized W4 nibbles |
| Current reduction | LDS + `__syncthreads()` across wave32 halves (CU mode, 128 threads) |
| Occupancy | **Max occupancy** (VGPR=48, 32 waves/SIMD, Scratch=0) |
| Primary limiter | **Instruction overhead from dequant, not raw bandwidth** |

The W4 GEMV bucket is nominally bandwidth-bound (AI=4), but does not achieve
bandwidth-bound performance because per-element dequantization (bit-extract +
widen + float-convert + scale + FP32 FMA) consumes more instruction slots than
the memory subsystem needs to fill them. The kernel is effectively
**dequant/instruction-throughput-limited within a bandwidth-bound envelope.**

This is the key pathology: if the kernel were a clean bandwidth streamer, it
would reach ~400+ tok/s for the W4 fraction alone. Instead it reaches
proportionally much less because each weight byte requires ~4–6 ALU instructions
before it contributes to the output dot product.

#### Paged Full-Attention (18.6%)

| Property | Value |
|---|---|
| Dominant operation | Split-K dot product over KV cache (4K context) |
| Arithmetic intensity | O(context × head_dim) / O(context × head_dim × 2) ≈ 1 (BF16 KV) |
| Regime | **Memory-bound** (reads grow linearly with context) |
| Current implementation | Split-K with 16 chunks × 16 heads = 256 blocks, warp-per-token cooperative dot |
| Primary limiter | **KV read volume grows with context length** |

At 4K context with 10 full-attention layers, 2 KV heads, 256 head_dim, BF16, the
KV read lower bound is:

```
4096 × 2 KV heads × 256 dim × 2 (K+V) × 2 bytes × 10 layers = ~84 MB/token
```

A naive per-query-head implementation can reread K/V for each Q group, pushing
the traffic toward:

```
4096 × 16 Q heads × 256 dim × 2 (K+V) × 2 bytes × 10 layers = ~671 MB/token
```

The true path sits between those bounds depending on kernel reuse and cache
behavior. At 4K this is still not the sole raw-bandwidth bottleneck, but it
grows linearly with context: at 32K the same bounds are ~0.67–5.37 GB/token.

The real attention limiter at 4K is occupancy, reduction overhead, and the
split-K reduce step, not raw KV bandwidth.

#### W8A16 GEMV — lm_head + shared expert (14.9%)

| Property | Value |
|---|---|
| Dominant shapes | `2048→248320` (lm_head), `2048→512` and `512→2048` (shared expert) |
| Arithmetic intensity | ~2 FLOP/byte (INT8 weight + FP16 activation + FP16 output) |
| Regime | **Deeply memory-bound** |
| Current inner loop | W8A16 vec8 unrolled FP16 FMA after int8 dequant |
| Primary limiter | **lm_head is huge (248320 output features = 509 MB INT8)** |

The `lm_head` alone reads 248320 × 2048 ≈ 509 MB of INT8 weights per token. At
864 GB/s that takes 0.589 ms minimum. At ~100 tok/s (10 ms/tok), the `lm_head`
minimum alone is ~5.9% of the token budget. The profiled W8A16 final/shared
bucket is 14.9%, so there is still meaningful headroom, but the right
comparison is against a roughly half-gigabyte final projection, not the smaller
figure earlier drafts assumed.

#### PARO Rotations (8.4%)

| Property | Value |
|---|---|
| Operation | Element-wise complex rotation (RoPE-style, PARO-specific rotate multiply) |
| Arithmetic intensity | ~2–4 FLOP/byte (read activation, rotate, write) |
| Regime | **Memory-bound / launch-bound** |
| Current implementation | Separate HIP kernels per rotation group, some fused (SiLU/down-rotate) |
| Primary limiter | **Launch count + intermediate tensor traffic** |

Rotations are fundamentally cheap per-element math, but they sit between other
kernels and create read-write-read chains. The 8.4% share is mostly intermediate
tensor traffic and kernel launch overhead, not compute. Fusing rotation into
adjacent GEMV work eliminates the intermediate write/read round-trip.

#### Router/Select (6.2%)

| Property | Value |
|---|---|
| Operation | Dense GEMV (router logits), top-k, softmax, expert selection |
| Arithmetic intensity | Router GEMV: ~4 FLOP/byte; top-k: negligible compute |
| Primary limiter | **Dispatch count + serial selection logic** |

Prior history: the original router had a serial thread-0 top-k loop with
scratch spills (5.7× kernel speedup from parallel warp-shuffle fix). Current
router is clean but still launches separate kernels for logits, top-k, softmax,
and expert gather.

#### GDN Decode (4.8%)

| Property | Value |
|---|---|
| Operation | Linear-attention recurrent state update (GDN) |
| Arithmetic intensity | Moderate (small matmuls + state update) |
| Current implementation | Native HIP kernel with lowp inputs, FP32 recurrent state |
| Primary limiter | **Sequential recurrence + small per-layer work** |

GDN is inherently sequential (state at layer N depends on state at layer N−1
within the token's recurrence). The 4.8% share is already well-optimized from
the earlier barrier removal and lowp input work. Further gains require either
algebraic restructuring of the recurrence or fusing GDN with adjacent linear-
attention work.

---

## 6. Why W4 Doesn't Mean 2× W8

### 6.1 The Theoretical Argument

If decode is purely bandwidth-bound and W4 reads half the bytes of W8:
- W8: 3.0 GB/tok @ 864 GB/s → 288 tok/s ceiling
- W4: 1.5 GB/tok @ 864 GB/s → 576 tok/s ceiling

So W4 *should* be 2× in the ideal case.

### 6.2 Why It Isn't (Current PARO vs W8A8: 100 vs 97 tok/s)

The full picture of why the 2× advantage evaporates:

**Factor 1: W4 GEMV is only 33.8% of decode time.**

Both PARO and W8A8 share most of the non-GEMV overhead: attention, routing,
recurrence, normalization, gating, and graph replay machinery. Even if the
GEMV bucket were perfectly bandwidth-optimal:
- W4 at 2× the GEMV bandwidth → 33.8% of time cuts in half → E2E 1.20×
- Not 2× E2E because 66.2% of time doesn't benefit from W4 at all.

**Factor 2: The W4 inner loop pays dequantization overhead that W8 avoids.**

Current PARO pack8 W4 GEMV (`gemv_awq_v8`) per 8 weight elements:

```
// Per element: extract nibble from packed int32, widen to FP32, apply scale/zero
w = float(((packed_word >> shift) & 0xF) - zero_point) * scale
acc += activation * w
```

Instructions per weight element:
1. `v_bfe_u32` or shift+mask (bit-field extract)
2. `v_sub_i32` (subtract zero point)
3. `v_cvt_f32_i32` (integer → float)
4. `v_mul_f32` (apply scale)
5. `v_fma_f32` (accumulate with activation)

That's **5+ VALU instructions per weight element**. On RDNA3 at peak, one
SIMD32 does 32 VALU ops/cycle. For a 2048×8192 projection: 2048×8192 = 16.7M
weight elements × 5 instructions = 83.7M instructions ÷ 32 lanes = 2.6M
cycles per SIMD, or ~1.05 ms at 2.5 GHz for a single SIMD. With 192 SIMDs
and perfect parallelism across output rows: 1.05 ms × 1 SIMD / 192 = 5.5 μs.
That's faster than the bandwidth limit of 8 MB / 864 GB/s = 9.3 μs. So for
this large shape, the kernel *should* still be bandwidth-bound.

But the effective bandwidth drops because:
- The 5 instructions per element fill instruction issue slots that could have
  been memory-request instructions
- The compiler may not overlap memory loads with dequant chains perfectly
- LDS reduction adds barrier stalls between productive work

For contrast, a dp4a-style inner loop:

```
// Per 4 weight elements (packed as one int32):
acc_int += __builtin_amdgcn_sudot4(activation_q8_packed, weight_q4_packed)
// Apply scale once per block (16-128 elements)
```

Instructions per 4 weight elements: **1 VALU instruction** (the dp4a).
Plus amortized scale/bias application every 128 elements. This is why dp4a
has theoretical 4× advantage: 4 MACs per instruction vs. 1 FMA per element
with 4 instructions of overhead.

**Factor 3: The W8A8 path has better effective instruction efficiency.**

W8A8 GEMVs (vec8 unrolled W8A16 kernels) dequant INT8 weights to BF16/FP16
with fewer instructions per element:
1. Load int8 word
2. `v_cvt_f16_i16` or equivalent (int8 → FP16, sometimes fused)
3. `v_fma_f16` (FP16 FMA with activation)

That's 2–3 instructions per element vs. 5+ for W4. And the W8A8 path got
vec8 unrolling early, which was +54% E2E. The W4 path has not yet received
the equivalent dp4a treatment.

**Factor 4: Non-GEMV buckets are roughly equal between W4 and W8.**

Attention reads the same KV cache (BF16) regardless of weight quantization.
Rotations operate on the same-sized activations. Router operates on the same
hidden states. GDN recurrence is the same math. So the 66.2% of decode time
that isn't GEMV is nearly identical between PARO W4 and W8A8. The only
advantage W4 has outside GEMV is lower memory pressure (21.7 vs 37.1 GiB),
which helps avoid fragmentation and enables larger KV caches at long context.

**Factor 5: PARO adds model-specific overhead that W8A8 does not have.**

PARO-specific rotations (the learned rotation matrices that are part of the
ParoQuant quantization scheme) add 8.4% of decode time that a straightforward
W8A8 model does not have. This is a pure quantization-scheme tax that partially
offsets the bandwidth advantage of smaller weights.

### 6.3 What Would Make W4 Actually 2× Faster?

For W4 to realize its theoretical advantage, the following would all need to
be true simultaneously:

1. The W4 inner loop achieves the same *effective bandwidth* as W8 (i.e., the
   dequant overhead becomes negligible relative to memory latency)
2. Non-GEMV overhead is reduced to a small fraction (attention, routing,
   recurrence are all fast)
3. PARO rotation overhead is fused into GEMV (eliminating the extra 8.4%)

In practice, (1) requires dp4a with block-aligned layout, (2) requires
long-term multi-bucket work, and (3) requires rotation fusion. The realistic
near-term target is not "2× W8A8" but "close the 17% → 34% efficiency gap
within the W4 GEMV bucket" — which would mean doubling W4 GEMV throughput
and getting E2E from 100 to ~120 tok/s.

---

## 7. Context-Length Scaling

The decode profile changes dramatically with context length because attention
work grows linearly while weight-read work stays constant:

### 7.1 Bucket Fraction vs. Context Length (Projected)

| Bucket | 512 context | 4K context (measured) | 32K context | 128K context |
|---|---:|---:|---:|---:|
| W4 GEMV | ~38% | 33.8% | ~25% | ~15% |
| Paged attention | ~8% | 18.6% | ~40% | ~55% |
| W8A16 (lm_head + shared) | ~17% | 14.9% | ~11% | ~7% |
| Rotations | ~10% | 8.4% | ~6% | ~4% |
| Router/select | ~7% | 6.2% | ~5% | ~3% |
| GDN | ~5% | 4.8% | ~4% | ~3% |
| Other | ~15% | ~13.3% | ~9% | ~13% |

(Projections based on linear attention scaling; actual ratios depend on
split-K tuning and KV cache layout efficiency at each context length.)

### 7.2 Implications

- **At 512 context:** GEMV dominates even more. Optimization priority is
  overwhelmingly W4 GEMV + W8A16 + rotation fusion. Attention is cheap.
- **At 4K context (current primary target):** Balanced profile. Both GEMV
  and attention matter. This is where we benchmark.
- **At 32K+ context:** Attention becomes the majority. KV cache bandwidth,
  split-K efficiency, and KV quantization (INT8 or lower) become the priority.
  W4 GEMV improvements have diminishing E2E impact.
- **At 128K context:** Attention dominates. Exact BF16 work has now lifted the
  short-tail row to the low 60s tok/s, but the remaining gap is still inside
  attention producer geometry, KV traffic, and split efficiency rather than W4
  GEMV.

### 7.3 Measured Context Scaling

From PARO compact bench rows at 24GB. The 2026-05-11 rows use the retained
warp-cooperative paged split-K context-tensor kernel, wave-partial max/sum
sync cleanup, page/address hoist, and grouped-GQA default where the split count
is high enough (`num_splits >= 64`). The older sustained 4K-generation rows
remain useful capacity data but are no longer the current short-tail decode
kernel efficiency.

| Context | Decode tok/s | Relative to 512 | Notes |
|---|---:|---:|---|
| 512 | 116.5 | 1.00× | default |
| 4K/128 | 122.0 | 1.05× | default |
| 4K (sustained 4K gen) | 113.6 | 0.98× | sustained row before later short-tail attention keeps |
| 32K/128 | 102.4 | 0.88× | default grouped GQA |
| 32K (sustained 4K gen) | 35.2 (long-decode mode) | 0.30× | older capacity row |
| 64K (sustained 4K gen) | 25.1 (long-decode mode) | 0.22× | older capacity row |
| 128K/128 | 63.7 | 0.55× | grouped GQA + split cap 512 |
| 128K (sustained 4K gen) | 15.9 (long-decode mode) | 0.14× | older capacity row |

The 512→4K drop is now small after the warp-cooperative split-K fix. The
remaining long-context cliff starts at 32K+ and is dominated by KV attention
traffic and split-K efficiency rather than W4 GEMV. The important postmortem:
the previous split-K path had enough CTAs but computed each token's QK dot
serially in one thread. Grid occupancy alone was not sufficient; within-block
work distribution was the roofline limiter. After that fix, the next exact
win was not another split-count sweep: hoisting page/block address work out of
the V loop raised 32K/128 `84.3 -> 92.1` tok/s and 128K/128 `42.6 -> 51.1`
tok/s with unchanged peak memory. The next retained win was exploiting
GQA reuse: scanning each KV stream once for the eight Q heads sharing it raised
32K/128 `92.1 -> 102.4` and 128K/128 `51.1 -> 56.7`, and it is now the
guarded default for long-context split paths. Retuning split geometry after
grouped-GQA then raised 128K/128 `56.7 -> 63.7` by making cap 512 the default,
with exactness fixture and canonical proof still green. A one-pass
online-softmax prototype reached `62.6` at 128K with validation skipped before
the split-cap retune, but failed exact graph/eager validation at 32K, so the
roofline path now needs a correctness fixture before another streaming rewrite
is promoted.

A post-retune 128K/128 rocprof trace filtered to the final decode replay window
still has the grouped-GQA context producer first at `869 ms / 1280 calls`,
about `49.5%` of decode-window kernel time. The fused split-K reduce+gate is
only `54 ms / 1280 calls`, about `3.1%`. That shifts roofline work toward the
producer's tile/value accumulation/reuse pattern rather than split-reduce
polish.

---

## 8. Measured vs. Assumed

Every roofline assumption should be checked against measurement. Here are the
key places where intuition failed:

| Assumption | Prediction | Measured Reality | Lesson |
|---|---|---|---|
| "W4 = half bytes = 2× W8" | 200 tok/s | 100 tok/s (parity) | Dequant overhead + Amdahl + PARO rotation tax |
| "dp4a = 4× FP32 per lane" | W4 dp4a should be much faster | Naive dp4a *regressed* on small shapes | Layout must be fixed first; dp4a without coalescing is slower |
| "Wave32/no-LDS = Vulkan-like" | Should match Vulkan's subgroup reduction style | +3–10% microbench, 0% E2E | The win is structural (less overhead), but not enough alone |
| "Graph replay = big decode win" | Should remove Python overhead | +2.6–3.9% | Decode was already GPU-kernel-dominated |
| "LDS staging = faster KV reads" | Cache hot data in LDS | Regressed or neutral in all tested cases | L1/L2 on RDNA3 are good enough; LDS adds barrier cost |
| "WMMA for decode" | Matrix cores accelerate everything | Useless at M=1 | 15/16 of WMMA tile wasted; scalar/vector only |
| "864 GB/s available" | Should achieve ~500+ tok/s for W4 | Whole-token active-weight BW is only ~150 GB/s; W4-bucket BW still needs counter validation | Real kernels lose bandwidth to layout, dequant, occupancy, reductions, and non-weight traffic |
| "INT8 = 2× FP16 compute" | INT8 should be faster than FP16 | Tested PyTorch path was 75.3 TOPS INT8 vs 84.8 TFLOP/s BF16 | RDNA3 peak class is similar for both; INT8 wins on bandwidth/residency, not guaranteed compute throughput |
| "-mllvm unroll-threshold-local=600" | Compiler fix = free perf | +166% prefill, ~0% decode | Prefill is compute-bound (helped); decode is BW-bound (didn't) |
| "Removing __syncthreads = faster" | Less sync = more throughput | Can corrupt recurrent state | Correctness first; barrier-free is only valid when proven safe |

---

## 9. RDNA3 Gotchas and Architecture Rules

### 9.1 Available Instructions

| Instruction | ISA feature | Status on gfx1100 | Notes |
|---|---|---|---|
| `v_dot4_i32_iu8` (sudot4) | `dot8-insts` | ✅ Available | Mixed signed/unsigned. This is the target for W4 dp4a. |
| `v_dot4_i32_i8` (sdot4) | `dot1-insts` | ❌ Not available | Signed-signed. Does NOT compile on gfx1100. |
| `v_dot4_u32_u8` (udot4) | `dot1-insts` | ❌ Not available | Unsigned-unsigned. Same constraint. |
| `v_dot8_i32_iu4` | `dot5-insts` | ✅ Available | 8 int4 MACs/lane/cycle. For INT4-packed layout. |
| `v_dot2_f32_f16` | `dot3-insts` | ✅ Available | 2 FP16 MACs → FP32 accumulator. |
| `v_fma_f32` | base VALU | ✅ Available | Standard FP32 FMA. Current PARO path. |
| `v_pk_fma_f16` | packed FP16 | ✅ Available | 2 FP16 FMAs per lane per cycle. |
| `v_wmma_f16_16x16x16_f16` | WMMA | ✅ Available | Only for M≥16 (prefill/batch). |

**Rule:** Any dp4a candidate for W4 decode MUST use `sudot4` (mixed
signed/unsigned) — not `sdot4` or `udot4`. The W4 weight nibble should be
treated as unsigned (0–15, subtract zero point separately), and the activation
operand can be signed int8. Failing to match signedness either won't compile
or silently produces wrong arithmetic.

### 9.2 VOPD (Dual-Issue)

RDNA3 introduced VOPD: two compatible VALU instructions can be issued to the
same SIMD in one cycle if they form a valid X+Y pair. Compatible pairs include
combinations of `v_fmac`, `v_add`, `v_mul`, `v_mov`, `v_cndmask`, etc.

**Not all ops VOPD-pair.** `v_dot4*` instructions are VOP3-encoded and do
NOT participate in VOPD dual-issue. This means:
- The "61.3 TFLOP/s FP32" spec assumes full VOPD utilization (~30.7 TFLOP/s
  without VOPD)
- INT8 dp4a at 123 TOPS does NOT rely on VOPD — it's a single-issue
  instruction that does 4 MACs per cycle per lane intrinsically
- When comparing dp4a to FP32 FMA: dp4a is 4 MACs/lane/cycle (single-issue);
  FP32 FMA is 1 FMA/lane/cycle (or 2 with VOPD). So dp4a is 2–4× FP32.

**Practical implication for W4 GEMV:** The dequant chain (shift, mask, sub,
cvt, mul) has limited VOPD opportunity because these ops depend on each
other sequentially. The compiler cannot easily dual-issue them. Switching to
dp4a eliminates most of this chain, giving both fewer instructions AND those
instructions don't need VOPD to be fast.

### 9.3 Wave32 vs Wave64

| Mode | Threads/wave | Scheduling shape | Execution | Notes |
|---|---|---|---|---|
| Wave32 | 32 | one 32-lane half | native/fine-grained | Default for hipEngine and llama.cpp HIP on RDNA3 in the checked paths |
| Wave64 / subgroup64 | 64 | two 32-lane halves | explicit/experimental | Vulkan reports subgroup 64; HIPCC can emit wave64 with `-mwavefrontsize64` |

Scheduling facts and trade-offs:
- Wave64 remains a first-class RDNA3 ISA mode. It is not deprecated, and LLVM can
  emit it for gfx1100 with `-mwavefrontsize64` / `+wavefrontsize64`.
- The hardware SIMD is still physically 32 lanes wide. A wave64 instruction is
  decomposed into two 32-lane halves.
- RDNA3 adds dual-issue VALU. For wave32, VOPD packs two independent compatible
  VALU ops into one instruction word. For wave64, eligible VALU halves can be
  co-issued in one cycle on the dual-issue path instead of always taking two
  cycles as on RDNA1/2. The same eligibility limits apply: no cross-lane ops,
  restricted operand classes, and source-cache/bank constraints.
- Wave32 has friendlier register-pressure and occupancy accounting, works with
  gfx11 WMMA `_w32` intrinsics, and gives LLVM direct VOPD pairing choices.
- For memory-bound decode kernels, the key question is total outstanding memory
  requests and occupancy, not nominal wave size.
- Parent W7900 experiments have no retained wave64 default win so far. Treat
  wave64 as an isolated experiment, not a default optimization lane.

Additional guardrail: do not assume `warpSize == 64` implies every shuffle or
lane-crossing pattern is correct across all 64 lanes. We have hit gfx1100
reduction candidates where `__shfl`/single-wave shortcuts behaved incorrectly
or effectively reduced within 32-lane halves. Any wave64 reduction rewrite
needs a tensor-level correctness test, not only a proof that it compiles.

### 9.4 LDS Is Not Free on RDNA3

LDS usage requires `__syncthreads()` barriers between producer and consumer
phases. On RDNA3, where the L1/L2 cache hierarchy is relatively capable:

- A barrier stalls ALL threads in the workgroup until the last thread arrives
- This idle time wastes memory-request slots that could be issuing loads
- For purely bandwidth-bound kernels, the data you'd put in LDS is often
  already in L1/L2 after the first access
- Every tested LDS-staging approach in this project (prefill scratch, hidden
  staging, paged-attention K tiling, dynamic LDS pack8, split-K workspace)
  either regressed or was neutral

**Rule:** Only add LDS when data is reused >4× within the kernel AND the
staged layout eliminates a real scatter/uncoalesced pattern. Default to
register-based per-thread accumulators with shuffle-based reduction.

### 9.5 Occupancy vs. Latency Hiding

For a bandwidth-bound kernel reading from VRAM (~500 cycle latency round-trip),
you need enough in-flight memory requests to keep the memory controller busy.
Each active wave can have independent memory requests outstanding.

- At 16 waves/SIMD (max, ≤96 VGPRs): plenty of latency hiding
- At 8 waves/SIMD (≤192 VGPRs): adequate for most patterns
- At 4–5 waves/SIMD (>256 VGPRs): starting to starve the memory controller
- At 1–2 waves/SIMD: critically undersubscribed, effective BW drops to 30–40%

**Rule:** For bandwidth-bound decode kernels, VGPR usage above ~128 should
trigger inspection. Any kernel with `Scratch_Size > 0` (register spills to
VRAM) on a hot decode path is a bug — spills add memory traffic and destroy
latency hiding.

### 9.6 Workgroup Geometry and Dead Lanes

For a simple PARO-style c=1 GEMV with `in_features = K`, each thread processes
roughly `K / threads_per_workgroup` elements. If that value is small, loop
setup, address arithmetic, and reduction overhead dominate useful work. For
MoE expert-down (`K = 512`), a 256-thread workgroup gives only two scalar
iterations per thread. That is not literally zero useful work, but it is a bad
overhead/useful-work ratio.

**llama.cpp HIP-specific example:** Q4_K MMVQ groups threads differently. Its
`kbx` loop maps thread groups to quant blocks, and for Qwen3.6-35B-A3B
expert-down (`ncols_x=512`) only 32 of the 256 HIP threads enter the useful
inner loop; 224/256 threads are idle. This is why the small-K dead-lane issue
is especially severe in llama.cpp HIP MMVQ, and why Vulkan's 64-thread subgroup
shape is structurally better for that path.

**Rule:** Match workgroup size to the smallest K dimension in the kernel's
dispatch set. For PARO MoE with `in_features=512`, 64 threads is appropriate.
For `in_features=2048`, 128 threads is reasonable. Do not use 256 threads
globally.

### 9.7 Compiler Quality: LLVM-AMDGPU vs ACO

The same RDNA3 ISA can have very different scheduling depending on the
compiler:

- **ROCm LLVM-AMDGPU** (used by HIP/hipcc): general-purpose LLVM backend.
  Tends to under-unroll tight loops, over-spill registers, and produce
  suboptimal waitcnt placement on gfx1100.
- **RADV/ACO** (used by Vulkan on open-source AMD drivers): purpose-built
  compiler for GPU compute shaders. Better register allocation, waitcnt
  scheduling, and VOPD pairing for this class of workload.

**Measured evidence:**
- `-mllvm -amdgpu-unroll-threshold-local=600` gave +166% on llama.cpp HIP
  prefill for the same kernel (same ISA instructions, just better scheduling)
- Vulkan's decode advantage over HIP is partly compiler-quality (same dp4a
  instruction, better surrounding scheduling)
- Our PARO v8 gets this flag automatically via the native extension build
  profile, but it was neutral/negative for decode (decode is BW-bound, so
  better scheduling doesn't help as much as for compute-bound prefill)

**Implication:** When a kernel appears to be at a fundamental hardware limit,
check if the ISA output (`--save-temps` or `amdgcn-dis`) shows unnecessary
spills, unrolling failures, or excessive waitcnt/nop insertion. The ceiling may
be compiler-imposed, not hardware-imposed.

### 9.8 Pipe Parallelism (Largely Unused)

The compute frontend has 8 ACEs (pipes) × ~8 HW queues each = up to 64
concurrent HW queues. But the QueueManager selects **one "connected" queue
per pipe** at a time to feed the shader complex, so the practical ceiling
for simultaneous in-flight kernels is 8 — one per pipe. Single-stream HIP
inference uses exactly 1.

This is not a primary c=1 decode lever for PARO right now (we're bandwidth-
bound on weight reads, so more parallel kernels wouldn't help if they all
contend for the same VRAM→L2 path). But it's worth noting because:

- Multi-stream overlap can help when one stream is compute-bound and another
  is bandwidth-bound (e.g., a small elementwise kernel running during a
  large GEMV).
- Prefill batched serving or c>1 decode with independent expert shards may
  benefit from spreading kernels across pipes.
- The MES scheduler's `compute_hqd_mask` and CU-reservation knobs exist to
  partition the GPU between queues, but they're Real-Time priority features
  and not exercised by a Normal-priority single-process workload.

Known gotchas for anyone experimenting with multi-stream HIP:

- Two streams on the same process share one VMID by default unless the
  driver assigns a different one; this affects page-table setup but usually
  not user-level behavior.
- Streams must use independent output tensors or explicit synchronization;
  implicit torch memory reuse can create cross-stream RAW hazards.
- For a pure bandwidth-bound decode, expect the sum of two streams' BW to
  not exceed the single-stream BW ceiling; you only win if compute and
  bandwidth overlap.
- `rocprofv3` kernel-trace CSVs separate kernels by queue ID, so you can
  verify pipe spread directly.

---

## 10. Optimization Decision Tree

When facing a decode performance gap, follow this diagnostic flow:

```
1. Profile: Which kernel buckets dominate total time?
   └─ Use: rocprofv3 --kernel-trace, sum by kernel name
   
2. For the largest bucket, determine the limiting factor:
   ├─ Check arithmetic intensity (AI)
   │   ├─ AI < 10: Probably memory-bound → check effective bandwidth
   │   └─ AI > 50: Probably compute-bound → check instruction throughput
   │
   ├─ Check occupancy (VGPR count, Scratch_Size)
   │   ├─ VGPR > 128: Occupancy concern
   │   ├─ Scratch > 0: Bug on hot path (spilling to VRAM)
   │   └─ Max waves/SIMD < 8: May be starving memory controller
   │
   ├─ Check grid coverage
   │   ├─ grid_size / workgroup_size < 96 CUs: Undersubscribed
   │   └─ Many idle threads per workgroup: Dead lanes (small-K problem)
   │
   └─ Check instruction mix
       ├─ >4 ALU ops per weight element: Dequant-overhead-bound
       ├─ Many __syncthreads/LDS ops: Barrier-overhead-bound
       └─ Many small kernel launches: Dispatch-overhead-bound

3. Based on limiter, choose intervention:
   ├─ Bandwidth-bound + good occupancy: Better layout/coalescing, wider loads
   ├─ Dequant-overhead-bound: dp4a (if layout supports it), block-quantize
   ├─ Barrier/LDS-bound: Remove barriers, use shuffle reductions
   ├─ Undersubscribed: Flash-decoding / split-K / change grid shape
   ├─ Dead lanes: Smaller workgroups, multi-row dispatch, shape-specific kernel
   └─ Dispatch-bound: Fusion, graph replay, fewer kernels per token

4. Validate with Amdahl:
   └─ bucket_fraction × expected_speedup = plausible E2E improvement?
      If not: you're optimizing the wrong bucket.
```

### Priority Map for Current PARO State (100 tok/s → target ~150+)

| Priority | Bucket | Intervention | Expected E2E gain |
|---|---|---|---|
| 1 | W4 GEMV (33.8%) | Block-aligned layout + dp4a + activation reuse | Up to 1.20× at 2× GEMV speed |
| 2 | Dispatch/fusion cleanup | First get a replay-only trace; then remove high-count RMSNorm/router/cast/rotation launches where grid shape is preserved | Compound ~5–10% if replay count is ~900/tok |
| 3 | W4 GEMV + Rotations (33.8% + 8.4%) | Fuse only boundaries where rotated/normalized input is not recomputed per output pack | Additional ~4–8% from eliminating launch + traffic |
| 4 | Paged attention (18.6%) | KV read coalescing, split-K tuning, warp geometry | ~5–10% (30% bucket improvement) |
| 5 | W8A16 (14.9%) | Fused lm_head+top-k, shared expert fusion | ~5–7% |
| 6 | Router/select (6.2%) | Fold into MoE boundary while preserving per-expert dot-product grid | ~2–3% |

**Target: 100 → 120–135 tok/s** is achievable by landing priorities 1+2+3.
Going beyond 135 requires simultaneous gains across all major buckets.

---

## 11. What Not To Chase

These are experimentally rejected approaches. Each was tested, measured, and
found to not produce retainable E2E improvement. Do not repeat them without
a fundamentally different framing:

| Approach | Why it was tried | Why it failed | Source |
|---|---|---|---|
| Naive dp4a on AWQ/pack8 layout | dp4a should be 2–4× FP32 | Wrong layout → scattered loads, activation quant overhead dominated | PLAN-PAROQUANT §dp4a-retry |
| Wave32/no-LDS W4 GEMV | Match Vulkan's subgroup-only style | +3–10% microbench, 0% E2E (wave shape isn't the primary limiter) | docs/PARO.md |
| `-amdgpu-unroll-threshold-local=600` for W4 | Worked for llama.cpp prefill | Decode is BW-bound; better scheduling doesn't help | docs/PARO.md |
| LDS staging in any form | Reuse data closer to ALU | L1/L2 already adequate; barrier cost exceeded savings | LESSONS-LEARNED.md |
| Multi-step graph replay | Capture 2–4 steps per graph | Regressed (graph capture overhead > replay savings) | docs/PARO.md |
| Generic W4 `_out` scratch buffers | Avoid allocation per kernel | Neutral under graph replay (allocations already captured) | WORKLOG.md |
| WMMA for c=1 decode | Matrix cores are fast | M=1 wastes 15/16 of tile | Arithmetic impossibility |
| GDN barrier removal | Sync is overhead | Corrupted recurrent state | LESSONS-LEARNED.md |
| Chunked lm-head top-1 | Skip most logit computation | 1.062 ms vs 0.766 ms for full (chunked overhead > savings) | PLAN-PAROQUANT |
| Global 64-thread pack8 override | Fewer dead lanes everywhere | Regressed gate/up (wrong size for large shapes) | docs/PARO.md |
| Paged attention forced at 512 | Consistent code path | Regressed 512/128 decode badly | docs/PARO.md |
| Dynamic LDS for pack8 | Flexible thread scaling | Regressed microbench AND E2E | WORKLOG.md |

---

## 12. Profiling Reference

### 12.1 Kernel Trace (Total Time by Kernel)

```bash
mamba run -n therock rocprofv3 --kernel-trace -f csv -o /tmp/trace -- \
  python3 your_benchmark.py
```

Output: `/tmp/trace_kernel_trace.csv` with columns including `KernelName`,
`DurationNs`, `GridSize`, `WorkgroupSize`, `ScratchSize`, `LDSSize`,
`VGPRCount`, `SGPRCount`.

Sum `DurationNs` by kernel name, rank by total. This is the entry point for
all optimization: identify which kernel to touch before touching anything.

### 12.2 Hardware Counters (Bandwidth, Compute, Occupancy)

```bash
cat > /tmp/pmc.txt <<'EOF'
pmc: SQ_WAVES
pmc: SQ_INSTS_VALU SQ_INSTS_SMEM SQ_INSTS_LDS
pmc: GL2C_HIT_sum GL2C_MISS_sum
pmc: GL2C_EA_RDREQ_32B_sum GL2C_EA_RDREQ_64B_sum GL2C_EA_RDREQ_96B_sum GL2C_EA_RDREQ_128B_sum
EOF

mamba run -n therock rocprofv3 -i /tmp/pmc.txt -f csv -o /tmp/counters -- \
  python3 your_benchmark.py
```

On the current ROCm 7.13 stack, `rocprofv3 -L` exposes these L2/EA counters
under `GL2C_*` names rather than the older `TCC_*` names. Do not pack all GL2C
counters into one `--pmc` group; rocprof aborts with "Request exceeds the
capabilities of the hardware to collect." Use multiple `--pmc` groups or
separate runs. A 2026-05-11 filtered counter attempt on the grouped-GQA context
kernel completed but returned all-zero GL2C values for both raw and derived
counters, so kernel traces remain the source of truth until counter collection
is revalidated on this stack.

Key derived metrics:
- **Effective bandwidth:** `(GL2C_EA_RDREQ_32B_sum × 32 + GL2C_EA_RDREQ_64B_sum × 64 + GL2C_EA_RDREQ_96B_sum × 96 + GL2C_EA_RDREQ_128B_sum × 128) / kernel_duration_s`
- **L2 hit rate:** `GL2C_HIT_sum / (GL2C_HIT_sum + GL2C_MISS_sum)`
- **VALU utilization:** `SQ_INSTS_VALU / (kernel_duration_cycles × waves × 1)`
- **Waves launched:** `SQ_WAVES` (compare to grid_size / workgroup_size)

### 12.3 ISA Inspection

For JIT-compiled HIP extensions:

```bash
# Find the cached .so
ls ~/.cache/torch_extensions/py*/nanovllm_amd_native_gfx1100_*/
ls ~/.cache/nanovllm_amd/torch_extensions/paroquant_kernels_v8*/**/*.so

# For load_inline extensions, add --save-temps to build flags
# Or disassemble the .so:
llvm-objdump -d --mcpu=gfx1100 path/to/extension.so | less
```

If a native HIP kernel stalls with no error, no completion, and near-zero GPU
activity, clear both cache roots before interpreting the result:

```bash
rm -rf ~/.cache/torch_extensions/py*/nanovllm_amd_native_gfx1100_*
rm -rf ~/.cache/nanovllm_amd/torch_extensions/paroquant_kernels_v8*
```

For Triton kernels:

```python
# Add to kernel call:
grid = (grid_size,)
kernel[grid](args..., num_warps=N)
# Then check ~/.triton/cache/ for .amdgcn files
```

Look for:
- Presence of `v_dot4_i32_iu8` (confirms dp4a emission)
- `scratch_*` instructions (register spills — should be 0 on hot paths)
- `s_waitcnt` density (excessive waits = poor scheduling)
- `ds_read`/`ds_write` (LDS accesses — count them)
- `s_barrier` (synchronization points)

### 12.4 Back-Calculating Effective Bandwidth

Given a kernel's measured duration and known data transfer:

```
Effective_BW = Total_Bytes_Transferred / Kernel_Duration_Seconds
```

For the W4 GEMV bucket at the current decode rate, the exact answer requires
either hardware counters or an exact per-bucket byte ledger. A rough
back-calculation is still useful, but it should not be called a direct
measurement:

```
Total decode time per token: ~10 ms (at 100 tok/s)
W4 GEMV share: 33.8% → 3.38 ms for all W4 GEMVs
Assumed W4-family bytes per token: ~0.8–1.2 GB, depending on which dense
projection bytes are counted in the W4 bucket versus W8A16/shared buckets
Illustrative W4-family bandwidth: 0.8–1.2 GB / 3.38 ms ≈ 237–355 GB/s
Illustrative fraction of 864 GB/s peak: ~27–41%
```

The exact percentage is assumption-sensitive. The reliable conclusion is that
the W4 bucket is far below the ideal streaming roof and needs real counter
validation. The likely losses are dequantization overhead, reduction stalls,
workgroup geometry, access pattern inefficiency, and activation/scale/zero
traffic.

---

## Summary

The W7900's 864 GB/s bandwidth sets a hard ceiling of ~576 tok/s for a W4
model with ~1.5 GB active weights. We achieve 100 tok/s (17.4% of ceiling).
The gap breaks down as:

1. **Amdahl (33.8% GEMV share):** Even perfect W4 GEMV caps at 151 tok/s
   given the current profile
2. **W4 GEMV efficiency below the streaming roof:** Dequant overhead, scale/
   zero traffic, reduction cost, and imperfect layout prevent the bucket from
   behaving like a clean W4 weight streamer
3. **Non-GEMV overhead (66.2% of decode time):** Attention, routing,
   recurrence, rotations, normalization, and dispatch overhead

To move from 100 to 150+ tok/s requires:
- Fixing (2): dp4a + block-aligned layout + activation reuse plus rotation
  fusion → materially faster W4 GEMV → E2E ~120–135 tok/s if the W4 bucket
  improves by roughly 2–4×
- Partially fixing (3): attention kernel improvements + MoE boundary fusion
  → additional ~10–15%

To reach 200+ tok/s would require additionally:
- Compressing (1): fusing enough non-GEMV work to shift the Amdahl breakdown
  so GEMV becomes 50%+ of (reduced) total time
- OR: multi-token decode (c>1) which enables weight reuse and moves the
  problem from bandwidth-bound toward compute-bound. Two distinct ways to
  realize c>1:
  - **Batched concurrent requests** (true c>1 across independent sequences):
    weight loads are amortized across sequences regardless of routing
    diversity; this is a real lever on W7900 and is not constrained by the
    analysis below.
  - **Speculative decoding / MTP** (apparent c>1 within a single sequence
    via B drafted tokens verified per step): only wins when
    `verify(B) ≈ verify(1)` — i.e. the verifier itself compresses to faster
    than a c=1 decode. This holds on dense 27–32B targets (projected
    2–3.5×) but **not** on Qwen3.6-35B-A3B at c=1, where 256 experts /
    top-8 / sequential MoE dispatch plus 30/40 linear-attention layers
    force verification cost to scale ~linearly with B. Measured MTP on
    this model is 0.68–0.70× AR. See `docs/SPECULATIVE-DECODE.md` for the
    η decomposition, break-even math, and decision gate.

The roofline is a diagnostic tool: it identifies *where* performance is being
lost and *what class* of fix is needed. It is not a guarantee of any specific
throughput. Every claimed improvement must be validated by measurement against
the profiled bucket, not just the theoretical ceiling.

---

## References and Further Reading

### Official AMD documentation

- **RDNA3 Shader ISA** — the authoritative reference for gfx11 instructions,
  encoding, wave execution model, and VOPD pairing rules:
  [rdna3-shader-instruction-set-architecture-feb-2023_0.pdf](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
- **MES (Micro Engine Scheduler) Specification, April 2024** — published via
  gpuopen. Covers queue mapping, priority bands, gang scheduling, preemption
  granularity, and the KMD↔MES API used throughout §1.6:
  [micro_engine_scheduler.pdf](https://gpuopen.com/download/micro_engine_scheduler.pdf)
  (see also the index at
  [AMD GPU architecture programming documentation](https://gpuopen.com/amd-gpu-architecture-programming-documentation/))
- **AMD W7900 product page** — authoritative peak spec values used in
  §1.3/§1.4 (FP32, FP16, INT8, VRAM bandwidth):
  [Radeon PRO W7900](https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7900.html)
- **"RDNA 3: Beyond the Current Gen"** — AMD's GPUOpen architecture deep
  dive covering WGP/CU organization, cache hierarchy, and chiplet design:
  [RDNA3_Beyond-the-current-gen-v4.pdf](https://gpuopen.com/presentations/2023/RDNA3_Beyond-the-current-gen-v4.pdf)
- **"How to accelerate AI applications on RDNA 3 using WMMA"** — AMD
  GPUOpen guide to `v_wmma_*` instructions, tile shapes, and when they win:
  [AMD GPUOpen WMMA guide](https://gpuopen.com/learn/wmma_on_rdna3/)
- **AMDGPU usage / kernel dispatch (LLVM docs)** — how HIP kernels become
  AQL dispatch packets and then COMPUTE_* register writes:
  [llvm.org/docs/AMDGPUUsage.html#kernel-dispatch](https://llvm.org/docs/AMDGPUUsage.html#kernel-dispatch)

### Independent analyses and measurements

- **Chips and Cheese, "Microbenchmarking AMD's RDNA 3 Graphics Architecture"
  (Jan 2023)** — source for the measured cache latencies (L1 24 cyc / L2 131
  cyc / L3 612 cyc / VRAM 1045 cyc) and bandwidths (L1 6.14 / L2 2.88 / L3
  2.30 TB/s, VRAM ~960 GB/s on 7900 XTX) used in §1.2:
  [chipsandcheese.com RDNA 3 microbenchmark article](https://chipsandcheese.com/2023/01/07/microbenchmarking-amds-rdna-3-graphics-architecture/)
- **geohot / tinygrad `7900xtx` documentation** — reverse-engineered details
  of the command processor, MEC/MES firmware, PM4 packet layout, and
  register-level queue state. Mirrored in this repo under `7900xtx/`:
  [github.com/geohot/7900xtx](https://github.com/geohot/7900xtx)
- **woct0rdho, RDNA3.5 ISA markdown** — searchable/markdown rendering of
  the RDNA3/3.5 ISA PDF, handy for grepping instruction forms and encodings:
  [github.com/woct0rdho/rdna35-isa-markdown](https://github.com/woct0rdho/rdna35-isa-markdown)
- **NaviSim (PACT 2022)** — academic timing simulator for RDNA-class GPUs,
  useful for understanding queue arbitration and wave dispatch at cycle
  granularity:
  [bu-icsg.github.io NaviSim paper](https://bu-icsg.github.io/publications/2022/navisim_pact_2022.pdf)
- **Otternes et al., "AMD's FreeSync for Real-Time Compute" (RTSJ 2022)**
  — analysis of AMD's compute queue scheduling behavior:
  [cs.unc.edu/~otternes/papers/rtsj2022.pdf](https://www.cs.unc.edu/~otternes/papers/rtsj2022.pdf)

### Related docs

In hipEngine:

- `docs/PLAN.md` — hipEngine architecture, phase roadmap, extensibility design
- `docs/BENCHMARK.md` — benchmark protocols, baselines, correctness gate
- `docs/KERNELS.md` — kernel port playbook, JIT cache, build profiles

In `~/amd-gpu-tuning/` (parent workspace, kernel R&D):

- `PLAN-PAROQUANT.md` — optimization roadmap and per-bucket priority for
  PARO decode
- `LESSONS-LEARNED.md` — hard-won rules with evidence per topic
- `WORKLOG.md` — chronological evidence ledger
- `docs/PARO.md` — PARO/W4A16 progress log
- `docs/LLAMACPP-VULKAN.md` — source-level analysis of llama.cpp HIP vs
  Vulkan backends on W7900
- `docs/SPECULATIVE-DECODE.md` — speculative decoding / MTP analysis;
  when `verify(B) ≈ verify(1)` holds (dense 27–32B) vs when it doesn't
  (Qwen3.6-35B-A3B 256-expert MoE at c=1)
- `7900xtx/` — local mirror of the geohot reverse-engineering notes
  referenced above, including `docs/CP.md`, `docs/MEC.md`, `docs/MES.md`,
  `docs/CU.md`, and `docs/launching.md`
