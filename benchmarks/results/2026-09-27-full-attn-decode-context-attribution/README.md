# Full-attention decode-context kernel: 801 ms, 81% of it in one launch

**Date:** 2026-09-27
**Subject:** `qwen35_paged_full_attn_decode_context_tensor_batch_kernel`, the
prefill attention path for the full-attention (non-QSA) layers.
**Verdict:** **56.4% of the `qsa_attention` family** and **5.0% of the whole
prefill**. It runs at **3.2% of FP32 peak** while demanding **968 GB/s of
L1-level K/V load traffic against 100.7 MB of unique data - a 7,707x re-read**,
because every `(head, row)` block rescans the entire context. In the launch that
carries 81% of its time it also runs at **25% occupancy**, LDS-limited.

## 1. Measured

| | |
| --- | ---: |
| 1x time | **801.4 ms** |
| launches | **36** per prefill = 12 layers x 3 |
| share of `qsa_attention` family | **56.4%** of 1419.9 ms |
| share of prefill kernel total | 5.0% of 16189.5 ms |
| layers | 3, 7, 11, ..., 47 - twelve, `full_attention_interval=4` |
| grid | (24 query heads, rows) x 256 threads |
| instantiation | `<true, 2>` - `PRECOMPUTE_TOKEN_OFFSETS=true`, `VALUE_VECTOR=2` |

### The task description's launch count was 2x and its layer count was 2x

It stated "72 launches over 24 full-attention layers". Measured: **36 launches
over 12 layers**. The authority is `role-analysis.json` `exact_role_kernels`,
which carries exactly 12 entries - `qsa_prefill:layers.{3,7,11,15,19,23,27,31,35,39,43,47}.attn_q`,
`rows=3` each, 66.70-66.84 ms each. 48 layers / `full_attention_interval=4` = 12,
and `compress_ratios` has a nonzero every fourth position. 72 was the raw
2-pass count.

### The per-layer split

Three launches per layer, and one of them is 81% of the kernel:

| grid | rows | context | causal | ms/launch | n | 1x total | share | ns/pair |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| (24, 1024) | 1024 | 1024 | yes | 12.140 | 12 | 145.7 ms | 18.2% | 23.16 |
| **(24, 1024)** | **1024** | **2048** | no | **54.300** | 12 | **651.6 ms** | **81.3%** | 25.89 |
| (24, 3) | 3 | 3072 | no | 0.346 | 12 | 4.2 ms | 0.5% | 37.54 |

Per layer: 66.79 ms, matching the authority's 66.70-66.84.

**The context lengths are inferred, and the inference is cross-checked.** They
are not readable from the trace - `context_len` arrives as a device pointer. But
the three launches agree at **23.2 / 25.9 / 37.5 ns per (row, context-token)
pair**, a 1.6x spread across a 3-row launch and two 1024-row launches. That
uniformity is what supports the assignment; a wrong context would break it.

## 2. Roofline position

| | |
| --- | ---: |
| `(row, head, token)` triples per layer | 6.31e7 |
| FLOP total | 7.76e11 |
| measured | **0.968 TFLOP/s** = **3.2%** of ~30 TFLOP/s peak |
| K/V load bytes | **776 GB** |
| demanded load rate | **968 GB/s** |
| unique K/V | 100.7 MB (8.4 MB/layer) |
| **re-read factor** | **7,707x** |
| after 12-way GQA sharing | 64.7 GB, 81 GB/s |

Not compute-bound, and not DRAM-bound either - the 968 GB/s is L1/L2-level
demand, and DRAM could serve the 100.7 MB of unique data almost for free. The
kernel is bound by a **redundant load stream**.

## 3. Structure

**One block per `(query head, query row)`** - grid (24, 1024), 24576 blocks, 256
threads, 8 warps.

- **No cross-row reuse.** `row = blockIdx.y`, so every query row rescans the
  context independently.
- **No GQA blocking.** `num_q_heads / num_kv_heads` = 24/2 = **12**, so twelve
  head-blocks share each kv_head's K/V - and they are separate blocks that each
  read it again.
- **QK phase**: a warp-per-token loop. Per token, 256 dims / 32 lanes = **8 FMA
  per lane to produce one score**, then a 5-step shuffle + 5 adds to reduce it.
  That is **18 lane-instructions per 8 useful FMAs - 125% reduction overhead.**
- **PV phase**: `value_groups == 1` because `threads=256 < 512`. Each of the 256
  threads holds `VALUE_VECTOR=2` output dims and **serially scans all
  `context_len` tokens**: one LDS read of `scores[token]`, one coalesced global
  bf16 read, two FMA.
- **Three passes over the context**: QK + running max, then exp + sum, then PV -
  with the whole `scores` vector parked in LDS between them.

## 4. Occupancy: LDS-limited, and it tightens with context

`shared_bytes = (max_context_len + num_warps + head_dim + value_groups*head_dim)*4 + 7 + max_context_len*8`

The trailing term is the **int64 `token_offsets` table**.

| `max_context_len` | `scores` | `token_offsets` | total/block | blocks/CU | occupancy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 4.00 KB | 8.00 KB | 14.04 KB | 4 | 50% |
| **2048** | **8.00 KB** | **16.00 KB** | **26.04 KB** | **2** | **25%** |
| 4097 | 16.00 KB | 32.01 KB | 50.05 KB | 1 | 12% |

At the context of the 81% launch, **`token_offsets` is 61% of the LDS** and the
kernel gets 2 blocks/CU. The largest offset is
`(physical_block*block_size + block_offset) * num_kv_heads * head_dim`, i.e.
`4096 * 512 = 2.1e6` at a 4096-token context - **int32 holds any context up to
about 4M tokens.**

## 5. Ranked levers

### 1. Narrow `token_offsets` from int64 to int32 - bit-exact

Changes only an address table's width. The accumulation order, the shuffle tree,
the wave reduction, and the softmax are all untouched.

| context | before | after |
| ---: | --- | --- |
| 1024 | 14.04 KB, 4 blocks/CU, 50% | 10.04 KB, 6 blocks/CU, **75%** |
| 2048 | 26.04 KB, 2 blocks/CU, 25% | 18.04 KB, 3 blocks/CU, **38%** |

Low risk, with one caveat: the `<false, N>` leaves do **not** declare the table,
so their allocation must keep its current formula.

No prize is estimated. Occupancy is the most likely limiter and this is the only
occupancy lever with no arithmetic risk, but the occupancy-to-time mapping is not
measured here.

### 2. Stop storing the whole scores vector in LDS - changing

The `scores` array is `max_context_len` floats and forces three passes over the
context with two `__syncthreads` each. Tiling the context so scores stay in
registers removes the array entirely and collapses three passes into one - the
standard fix for exactly this shape.

This is the lever that **removes** the occupancy ceiling rather than moving it.
It is also the one that changes the arithmetic: online-softmax rescaling changes
the accumulation order, so it needs the production numerical gate and a
registered strict fallback.

### 3. GQA blocking: the 12 heads of a kv_head in one block - bit-exact in principle

Cuts K/V traffic by up to 12x (776 GB -> 64.7 GB). The heads are independent, so
a per-head reduction keeping its own warp set and token striding is bit-exact.
**But it competes for the same LDS budget**: scores for 12 heads x 2048 context
is 96 KB, which does not fit. A group of 2-4 heads is 16-32 KB on top of the
current 26 KB, so this probably needs lever 2 first.

### 4. Row tiling - changing unless each row keeps its own warp set

Reuses the K/V scan and the offset table across rows. Bit-exact only if each row
keeps 8 warps and the same token striding, which multiplies LDS by the row count.
Same budget conflict as lever 3.

### 5. Fold `inv_denom` into the PV accumulation - bit-exact

The `value_groups==1` path runs a full pass over `context_len` in which only
lane 0 of each warp is active, doing `scores[token] *= inv_denom`. The
`value_groups>1` path already folds it as `(scores[token] * inv_denom)`, which
produces identical values in an identical order. Removes that pass and one
`__syncthreads`. Small - at most 0.5% of the kernel.

## 6. What this attribution does not establish

**The precise hardware limiter is not isolated.** Three candidates are each
consistent with the measurements:

- LDS-limited occupancy at 25% in the dominant launch,
- the 968 GB/s redundant K/V load stream,
- the 125% shuffle-reduction overhead per score.

**No counter in the trace separates them**, and the levers are ranked on
structural argument, not on a measured counter attribution. Separating them needs
a standalone microbenchmark of the 54.30 ms launch with variants that change one
factor at a time - which this pass did not run.

Also not established: the context lengths are inferred (section 1), and no
per-lever prize is estimated for the same reason.

## Files

| file | content |
| --- | --- |
| `attribution.json` | machine-readable launch split, roofline model, occupancy table, lever list |

The trace is a gitignored raw dump under `2026-09-27-router-warp-reduce/`.
`Grid_Size_X/Y` are total work-items (6144/256 = 24 heads, 1024/1 = 1024 rows),
and the raw CSV holds two prefill passes; all figures here are 1x.
