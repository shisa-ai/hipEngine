# moe_reduce is 81% maps, and the router is reduction-bound, not traffic-bound

**Date:** 2026-09-27
**Hardware:** AMD Radeon 8060S (gfx1151), Framework Desktop
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL, code-p4096 prefill, production default
**Artifacts:** `attribution.json`, `router-variants.hip`, `router-variants-bench.py`, `router-tile2d-bench.py`
**New runs:** two kernel screens at the real shape (1024 x 2560 x 512).

## Verdict

`moe_reduce` is **81% maps, not reduction.** The router logits kernel is 617.7 of
784.9 ms; the actual reduction is 67.7 ms. Fusing the reduction caps at 8.6% of
the bucket.

And the router's limiter is **the reduction structure, not traffic**. A variant
that replaces the 8-barrier 256-thread LDS tree with a 32-lane shuffle is
**2.03x faster**. A 2D tile that cuts traffic 10x is **3.9x slower**.

On packing: our `quantize_pack` bucket is **0.0 ms by construction**, and our
complete explicit conversion is **16.2 ms**. The comparator's 284.8 ms bucket is
not recoverable here - claiming it would double-count with `dense_projection`.

## 1. moe_reduce attribution

From the shipped-default role trace (halved from 2 repetitions; every `ms`
and `n` in this artifact is a 1x per-prefill value):

| component | class | kernel | ms | share |
| --- | --- | --- | ---: | ---: |
| **router logits** | **maps** | `qwen35_router_logits_f32_token_tile_dense_exact_kernel<4>` | **617.73** | **78.7%** |
| weighted lanes sum | reduction | `weighted_lanes_sum_out_kernel` | 67.66 | 8.6% |
| group scatter/gather | scatter/gather | `qwen35_moe_group_scatter_gather_kernel` | 52.06 | 6.6% |
| silu mul dual | activation | `silu_mul_dual_out_kernel` | 29.22 | 3.7% |
| router select | maps | `qwen35_router_select_kernel` | 18.25 | 2.3% |
| | | **total** | **784.93** | |

| class | ms | share |
| --- | ---: | ---: |
| **maps** | **635.98** | **81.0%** |
| reduction | 67.66 | 8.6% |
| scatter/gather | 52.06 | 6.6% |
| activation | 29.22 | 3.7% |

The measured total is 784.93 ms against the taxonomy's 758.4 ms - the difference
is the `silu_mul_dual` and `router_select` assignments, which sit near the
`elementwise_norm` / `moe_reduce` boundary.

**So the target is the router, and the reduction is a rounding error on it.**
Fusing the reduction and the scatter/gather together is worth 119.7 ms.

## 2. What limits the router

The shipped kernel launches `grid(512, 256)` at block 256 with a 4-token tile: one
block per (expert, 4-token group). It ends with a 256-thread LDS tree reduction
that costs **8 `__syncthreads()` per block**, issued for only 4 output values.

Screen at the real shape. The faithful T=4 copy is **bit-identical** to the
shipped library (`max_abs_diff` 0.0), so the comparison is attributable:

| variant | ms | vs shipped | max abs diff |
| --- | ---: | ---: | ---: |
| shipped (library) | 3.0126 | 1.000 | - |
| `router_token_tile4` (faithful copy) | 3.0877 | 0.976 | **0.0** |
| `router_token_tile8` | 2.7062 | 1.113 | 0.0 |
| `router_token_tile16` | 2.9069 | 1.036 | 0.0 |
| `router_token_tile32` | 4.0032 | 0.753 | 0.0 |
| 2D tile 16 tokens x 16 experts | 11.7314 | **0.257** | 5.96e-06 |
| **warp-per-token (shuffle reduction)** | **1.4834** | **2.031** | 4.92e-07 |

Three things are ruled out and one is confirmed:

- **Tile width is not the lever.** T=8 gives 1.11x, T=16 gives 1.04x, T=32 is
  worse. Flat within noise.
- **Traffic is not the lever.** The shipped kernel moves 6.71 GB of re-reads
  (each block re-reads its activation rows; the weight matrix is read 256 times)
  in 3.01 ms - **2222 GB/s effective**, which is L2 bandwidth, not DRAM. The
  2D tile cuts that to 0.671 GB and is **3.9x slower**, because it adds an LDS
  round-trip per operand element and the LDS path becomes the bottleneck. The
  traffic model is refuted by measurement.
- **The reduction structure is the lever.** Warp-per-token makes each warp own
  one token, so the 8-barrier block tree becomes a 5-step shuffle, and the same
  arithmetic is **2.03x faster**.

**Prize: 313.6 ms** - the router at 617.73 ms, 2.03x.

The warp variant is **not bit-exact** (`max_abs_diff` 4.9e-7): the reduction order
changes from a 256-thread tree to a 32-lane shuffle. It therefore needs the
production profile gate and a registered strict fallback, not a parity argument.

## 3. Activation packing: the 284.8 ms is not ours to collect

The shared taxonomy defines `quantize_pack` as *activation quantization feeding a
quantized matmul*, and it is a **comparator-side** family:

| | hipEngine | halobox-base | halobox-pr63 |
| --- | ---: | ---: | ---: |
| `quantize_pack` | **0.0** | 284.8 | **0.0** |

Our bucket is 0.0 ms because our activation conversion happens **inside** the
matmul kernels - there is no standalone kernel to eliminate. The complete
explicit conversion cost on our side is:

| kernel | ms | n |
| --- | ---: | ---: |
| `f32_to_bf16_kernel` | 11.16 | 384 |
| `bf16_to_f32_kernel` | 5.06 | 192 |
| **total** | **16.22** | |

**So the honest complete-operation cost of activation packing here is 16.22 ms of
explicit kernels plus whatever the matmuls spend converting inline - not 284.8 ms.**
Quoting the comparator's number as recoverable would double-count with
`dense_projection`, where our conversion already lives.

It is also not a clean 284.8 ms win on the comparator side: PR #63 changed the
matmuls and the packing together. In the same PR `dense_projection` fell only
162.4 ms (1536.7 -> 1374.3) and `expert_gate_up` only 57.6 ms (819.4 -> 761.8)
while `expert_down` fell 737.0 ms (1149.8 -> 412.8). The bucket's disappearance
and the matmuls' speedup are entangled.

## Double-counting guard

The same saving is not counted twice because each is reported once, in the family
that owns it:

- `moe_reduce` is split into maps / reduction / scatter / activation with no
  overlap; the class shares sum to 100%.
- The comparator's `quantize_pack` elimination is **not** claimed as recoverable
  on our side, because our conversion is inside `dense_projection` and the expert
  families.
- The iu8 repair passes are not counted here; they are folded into
  `expert_gate_up` and `expert_down`, as the taxonomy requires.

## What this does not establish

- **No production kernel was written or gated.** Both screens are standalone
  microbenches against a faithful copy of the shipped kernel; the warp variant
  has not been registered, has no RED test, and has not been through the profile
  gate.
- **Synthetic operands.** Values are random normal, so this is a geometry and
  access-pattern measurement, not a prefill rate. Byte counts and the reduction
  structure are representative.
- **One shape.** 1024 tokens, hidden 2560, 512 experts, code-p4096.
- **The router screen used 30 repetitions after 5 warmups**, single run each; the
  1.11x T=8 result is within the spread of the flat variants and should not be
  read as a win.
