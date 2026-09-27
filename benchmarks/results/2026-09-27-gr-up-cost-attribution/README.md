# GR-up cost attribution: 2507 ms, weight-bandwidth-bound on a 256x re-read

**Date:** 2026-09-27
**Subject:** `q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32<true>`
(`gguf_k_gemv.hip:952`), the fused Q8_0 hyper-connection up projection plus
sigmoid plus gated-mean.
**Verdict:** the largest single kernel in the prefill at **15.5%**, and it is
**weight-bandwidth-bound at 59% of DRAM peak** with **90% of its traffic being a
256x re-read of one 3.48 MB tensor**. Three bit-exact levers follow from that; the
arithmetic-change routes are closed for a reason recorded in `docs/REFACTOR.md`.

## 1. Measured

| | |
| --- | --- |
| 1x time | **2507.0 ms** |
| launches | **384** per prefill (2 per layer per chunk) |
| per launch | 6.529 ms |
| share of prefill kernel time | **15.5%** of 16189.5 ms - **rank 1** |
| shape | rows=1024, K=320 (`low_rank`), N=10240 (4 branches x 2560) |
| grid | (1280, 256) x 128 threads = 327680 blocks |
| tile | ROW_BATCH 4, HIDDEN_TILE 2, BRANCHES 4, 4 waves |

Source: `benchmarks/results/2026-09-27-router-warp-reduce/trace-a_kernel_trace.csv`,
a current-composition trace taken after the `dense_wide256` layer-scope promotion
moved layers 16-47 off the dense Q8_0 prefill route. The raw CSV holds two prefill
passes; every figure here is 1x.

**The 2431.8 ms figure this task started from is stale.** The kernel is unchanged;
the composition around it moved. Current is 2507.0 ms.

Two column-semantics traps worth stating, both of which this analysis had to clear
first:

- `Grid_Size_X/Y` in a rocprofv3 CSV are **total work-items**, not grid
  dimensions: 163840 = 1280 x 128 and 256 = 256 x 1. Reading them as grid dims
  gives 4.2e7 blocks.
- `--repetitions 1` still traces **two** passes, so the raw launch count is 768,
  not 384.

## 2. Roofline position

| | |
| --- | ---: |
| FLOP per launch | 6.71e9 |
| measured | **1.03 TFLOP/s** = **3.4%** of FP32 peak |
| traffic per launch | 987 MB |
| effective bandwidth | **151 GB/s** = **59%** of DRAM peak |
| operational intensity | **6.8 FLOP/byte** against a machine balance of ~117 |
| | = **5.8% of balance** |

The kernel is nowhere near compute-bound and is using most of the memory system it
can reach. Traffic breakdown per launch:

| component | bytes | share |
| --- | ---: | ---: |
| **weights, re-read 256x** | 891.3 MB | **90.3%** |
| `gate` store | 41.9 MB | 4.2% |
| `normalized` read | 41.9 MB | 4.2% |
| `mixed` store | 10.5 MB | 1.1% |
| `x` read | 1.3 MB | 0.1% |

## 3. The re-read, and why the access pattern is not the problem

`row_base = blockIdx.y * ROW_BATCH` and `blockIdx.y` spans `rows / ROW_BATCH` =
256. Every row-batch sweeps all 1280 hidden tiles, so the entire 3.48 MB weight
tensor is swept once per row-batch and the reuse distance is the full 3.48 MB -
served from DRAM, not cache. That is a **256x** amplification, and it exists
because the dequantized weight is amortized over only **four** rows.

This is worth separating from a common misdiagnosis. `q8_0_weight_wave_scale`
(`gguf_k_gemv.hip:105`) uses `__builtin_amdgcn_readfirstlane` for the block index,
so all 32 lanes of a warp read one 34-byte q8_0 block: the 2-byte scale is
broadcast and the 32 quant bytes are lane-indexed (`k & 31 == lane`). **The read is
coalesced.** The problem is how many times it happens, not how it is issued.

### A traffic-basis correction to the recorded GR-up numbers

`docs/REFACTOR.md` "GR-Up Wide Tile" records the fused kernel's weight traffic as
**3.565 GB and 1.88 FLOP/byte**. That is the **f32-dequantized** size
(`10240 x 320 x 4 = 13.11 MB`, times 256 = 3.355 GB), not the **packed q8_0**
tensor the kernel actually reads (`3.48 MB`, times 256 = **891 MB**).

| basis | weight size | x256 | FLOP/byte |
| --- | ---: | ---: | ---: |
| packed q8_0 (**what the kernel reads**) | 3.48 MB | **891 MB** | **7.53** |
| f32 dequantized (recorded) | 13.11 MB | 3.355 GB | 1.88 |

The conclusion survives - 90% of traffic is still the re-read, and the kernel is
still weight-bandwidth-bound - but the magnitude is 4x smaller than recorded. The
wide tile's recorded "13.9 MB, 483 FLOP/byte" has the same conflation: 13.9 MB is
its **f16 LDS staging** size, while its DRAM weight read is the same 3.48 MB packed
tensor. That also weakens the 483 FLOP/byte framing of why the wide tile wins:
its advantage is amortizing the **dequantization work** over 256 rows, not a 256x
reduction in DRAM traffic - it reads 3.48 MB either way.

## 4. The epilogue is the one component the trace cannot separate

The epilogue runs on `tid < ROW_BATCH * HIDDEN_TILE` = **8 of 128 threads**,
against a main loop of roughly 220 instructions per thread. From source that is
about **13% of the critical path of one warp** while three warps idle, plus four
`expf` and a set of scattered stores.

The stores are the risk: `gate[index]` has `index = row*10240 + branch*2560 +
hidden_base + col`, so the 8 active threads write 2 consecutive floats per
`(row, branch)` at 10 KB and 40 KB strides - up to 32 cache lines for 128 bytes of
data. The 64-byte line at `hidden_base` is shared with the seven neighbouring
blocks at consecutive `blockIdx.x`, so whether the writes merge depends on their
co-residency, which **this analysis does not establish**.

**This is marked as an estimate, not a measurement.** Splitting the projection from
the epilogue needs a variant microbenchmark at the same geometry; no counter
reading in the trace separates them.

## 5. Ranked levers

### 1. ROW_BATCH 4 -> 8 at HIDDEN_TILE=2 - bit-exact

Halves the weight re-read (891 -> 445 MB/launch), taking total traffic to 541 MB
and operational intensity from 6.8 to 12.4 FLOP/byte. **The arithmetic does not
change:** the per-output k-order, the 16/8/4/2/1 shuffle tree, and the 4-wave LDS
sum are all untouched. `ROW_BATCH` only adds independent row accumulators, and
`weights[branch][col]` is dequantized once and consumed by every row in the same
order.

Cost: `acc[8][4][2]` is 64 floats against 32, plus the `weight_rows` pointers -
estimated 66 -> 98 VGPRs, so roughly 7 -> 5 blocks/CU. Occupancy loss is the
risk, and it is the reason the estimated prize below is an upper bound.

**The existing 0.80x result does not test this.** The 2026-09-06 `RB8/HT1` retile
halved the re-read and measured 0.80x, and `docs/REFACTOR.md` records that it "does
not fit a pure-bandwidth story and is open". It changed **two** things at once:
`HT1` halves the FMAs per `x` load and doubles the block count, so it is a
confounded experiment. It is not evidence about row batching. Only a `rowbatch4`
kernel exists in-tree; **a clean RB8/HT2 variant has never been built here.**

Estimated prize: **~900 ms**, an upper bound from 350 MB/launch less traffic at the
measured 151 GB/s, assuming the kernel stays bandwidth-bound and occupancy loss is
free.

### 2. Drop the dead `gate` store - bit-exact, free

In the `fused_up` path the kernel writes `gate[index]` and then uses its own local
`sigmoid_gate` to build `mixed`. **Nothing reads `scratch.gate` afterwards**: the
follow-up `qwen4_exp_sigmoid_f32` is guarded by
`not fused_up and not fused_sigmoid_mean` (`qwen4_exp_runner.py:3127`) and is
skipped. The 41.9 MB/launch store is dead - 16.1 GB per prefill.

Estimated prize: **~107 ms** (4.3%), excluding the store-coalescing amplification
in section 4, which would only increase it. Needs a separate entry point or a
template parameter - **not** outright deletion: the `gr_wide` and `gr_iu8` paths do
read `scratch.gate` back through their separate sigmoid and gated-mean kernels.

### 3. HIDDEN_TILE 2 -> 4 at ROW_BATCH=4 - bit-exact

Doubles the FMAs per `x` load (32 -> 64 per `xv`) and halves the block count and
epilogue invocations. It does **not** change weight traffic: the re-read factor is
`rows / ROW_BATCH` and is independent of the hidden tiling. Same register class as
lever 1, so the two cannot be combined for free. The kernel is at 3.4% of FP32
peak, so this addresses a non-binding resource; no prize is estimated.

### 4. Fuse the grouped-rmsnorm producer of `normalized`

`normalized` costs 41.9 MB/launch of reads. If a separate kernel produces it,
fusing removes that launch plus its write and this read. Arithmetic-preserving if
the rmsnorm math is unchanged. The producer is not identified in this pass.

### 5. "Recover projection ILP" - not a lever

Recorded in the task's no-arithmetic-risk trio, but the source does not support it
as an independent lever: at `ROW_BATCH=4 x BRANCHES=4 x HIDDEN_TILE=2` there are
already **32 independent FMA chains** per thread. ILP is not the limiter; traffic
is, and lever 1 is the traffic lever.

## 6. Why the arithmetic-change routes are closed

Independent of this attribution, the GR family is closed to arithmetic change
(`docs/REFACTOR.md`): every measured arithmetic route saturates the production gate
at 1.3e-3 - 1.6e-3 mean KL, including corrections more accurate than the
incumbent.

The shared mechanism is the MoE router's top-k expert selection, which is a
**discrete amplification node**: a ~1 ulp router-logit difference flips which of
512 experts runs, moving the model logits by O(1). Measured on the router
warp-reduce candidate, where all six top-1 mismatches were flip-eligible and none
were outside (`benchmarks/results/2026-09-27-router-warp-reduce/`). Every
downstream GR or dense arithmetic route inherits that saturation.

**Levers 1-3 are bit-exact, so they do not need the numerical gate at all** - only
the strict parity contract, which they satisfy by construction. That is why they
are the only ones ranked as actionable.

## Files

| file | content |
| --- | --- |
| `attribution.json` | machine-readable attribution, roofline model, and lever list |

The kernel trace this derives from is a gitignored raw dump under
`2026-09-27-router-warp-reduce/`. The launch count and geometry are reproducible
from the same `rocprofv3 --kernel-trace` command; the `ms` figures are 1x.
