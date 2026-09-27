# MoE expert_down: the LDS weight-staging path is prescribed, bit-exact, and never tried

**Date:** 2026-09-27
**Hardware:** AMD Radeon 8060S (gfx1151), Framework Desktop, ROCm therock10-staging-20260828
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL / BF16 KV, code-p4096
**Artifact:** `feasibility.json`
**New engine runs:** none. Derived from the shipped kernels' source, the
2026-09-22 MoE cost and j32 packets, and the comparator's PR #63 tree.

## Verdict

The routed expert_down gap is **2.23 s of the 12.97 s code-p4096 gap**, and the
family is weight-traffic-bound, not compute-bound. The one lever that reduces
that traffic is staging the quantized weight chunk in LDS so the block's row
groups share it. That lever is **prescribed by our own 2026-09-22 packet, is
bit-exact, and has never been tried** - the 32-row rejection tested a tile-height
parameter, which the packet itself says is not the same thing. It is not
implemented here; this artifact establishes the mechanism, the arithmetic, and
the feasibility, and hands the kernel off as a bounded unit.

## The comparator's move is llama.cpp MMB, not a bespoke kernel

PR #63 adds `ggml/src/ggml-cuda/mmb.cu` (938 lines) and `mmb-quant.cuh` (336);
neither exists in the base tree at `/home/lhl/comparators-20260915/halobox-strix`.
It is the upstream-style MoE multiblock kernel, selected by `LLAMA_MMB*`
environment variables captured in
[`2026-09-13-framework-qwen4exp-strix-journey-baselines.json`](../2026-09-13-framework-qwen4exp-strix-journey-baselines.json)
(`LLAMA_MMB=1`, `MIN_T=512`, `BF16W=1`, `GLU=1`, `TALL=2`, `CACHE=4`,
`F32SPLIT=2`, `HC16=2`, `SHADOW=2`, `DOWN16=1`).

Its tile geometry is **the geometry we already ship**:

| | MMB (`mmb.cu:16`) | our `dense_wide256` (`gguf_q8_0_dense_wide.hip`) |
| --- | --- | --- |
| K tile | `MMB_BK = 64` | `WIDE_BK = 64` |
| N tile | `MMB_NT = 256` | `BN = 256` |
| LDS stride | `MMB_BK + 8` | `WIDE_LDS_STRIDE = WIDE_BK + 8` |
| LDS contents | dequantized **bf16** weight tile | dequantized **f16** weight tile |
| MMA | `v_wmma_bf16` | `v_wmma_f32_16x16x16_f16` |

Two consequences:

1. **We are not missing a tile geometry.** We are missing the routing wrapper
   that applies that geometry to expert weights. The tile itself is proven
   in-tree at 4.50 TFLOP/s on the GR-up shape.
2. **Part of the comparator's -64.1% is an ungated numerics trade.** `BF16W=1`
   means the weights are dequantized to BF16 - 8 significand bits - before the
   MMA. Our expert_down runs `selected_wmma_iu8_risk` with raw Q5_1 codes and
   the comment "the weight path has no requantization error at all", plus Kahan
   accumulation and at-risk repair. That is a stricter contract, and it is 721 ms
   of our 2389 ms. The comparison table's ratio column already reads 5.77x here
   and 2.07x against the base, and this is why: the base column flatters us on
   exactly the family where the comparator moved furthest.

## Why the family is traffic-bound

The shipped down kernel gathers weights **per lane**:

```c
const int64_t global_col = col_base + warp_col * 32 + lane;
const uint8_t* block_base = qweight + expert_id * (out_features * row_bytes)
                          + global_col * row_bytes;      // row_bytes = 480 for down
const uint8_t* block_ptr = block_base + (col0 / 32) * 24;
```

Each block owns one 16-row tile and 128 output columns, so it reads that
expert's whole `128 x K` slice. The two row groups of an 8-warp block read the
same bytes, but not through any shared path. Cost is therefore

```
bytes = sum_experts(ceil(rows_in_expert / 16)) * out_features_total * row_bytes
```

and the 2026-09-22 packet measured the model to 2-3% of the engine on the same
kernel and geometry. At the engine's real routing (`grid_y = 1120` tiles for
10240 compact rows, padding 1.75):

| quantity | value |
| --- | ---: |
| traffic per launch | **1.261 GB** |
| launches per code-p4096 prefill (48 layers x 4 chunks) | 192 |
| prefill traffic | **242 GB** |
| distinct expert bytes per launch (512 experts x 1.126 MB) | 0.576 GB |
| **over-read ratio** | **2.19x** |

242 GB at the measured 104-126 GB/s reproduces the 2229.8 ms. The counter work
confirms the read is not wasteful per request - `real/modeled = 1.08` - so the
kernel is already moving close to exactly the bytes its tile count implies. The
only way down is fewer tiles.

## What the prior rejection actually rejected

The 2026-09-22 j32 packet widened the tile to 32 rows and measured it 1.3-2.2x
slower with only a 6.3% real DRAM reduction, and its own conclusion says why:

> "The tile-count traffic model is **necessary but not sufficient**: it counts
> *distinct* weight reads, and a taller tile makes the reads shared rather than
> distinct. Sharing only pays if it is explicit. A 32-row tile that actually
> banked the traffic reduction would have to stage the chunk's quantized weights
> in LDS once per block and have all 8 warps read them from there, instead of
> each lane gathering its own column's blocks from global memory. That is a
> different weight path, not a tile-height parameter."

So the rejected experiment changed `TILE_ROWS` and kept the per-lane gather. The
prescribed fix - a shared LDS weight path - was never built. This artifact does
not claim it will win; it claims the rejection does not cover it.

## Feasibility: the staging fits

The kernel already stages the *activation* in LDS (`lds_q8a`, `lds_da`,
`lds_sumq8a`, `lds_row_atrisk`), which is what "our at-risk flags are already in
LDS" means. Adding a weight chunk alongside it, at `TILE_ROWS=32`:

| LDS item | bytes |
| --- | ---: |
| `lds_q8a` (3 planes x 32 rows x 272) | 26,112 |
| `lds_da` | 3,072 |
| `lds_sumq8a` | 3,072 |
| `lds_row_atrisk` | 128 |
| **activation subtotal** | **32,384** |
| weight chunk, 128 col x 256 K, Q5_1 quantized | 24,576 |
| **total** | **56,960 of 65,536** |

It fits with 8.4 KB to spare. A 128-K chunk instead halves the weight tile to
12,288 B, for 44,672 B total and 20.4 KB of headroom, which leaves room to
double-buffer the staging against the MMA loop.

## Why this one is worth building when the GR tile was not

- **It is bit-exact.** The staged bytes are the same Q5_1 codes, read in the
  same order, feeding the same iu8 WMMA and the same Kahan/at-risk repair. Only
  the address the lane reads from changes. Sprint H closed the GR family to
  arithmetic change; this path does not need an arithmetic change, so that
  closure does not apply to it.
- **The prize is arithmetic, not speculative**: tiles 1120 -> ~600 at 32 rows,
  traffic 242 -> ~130 GB, so ~1.0-1.1 s of the 12.97 s gap at the current
  bandwidth. The 2.19x over-read ratio bounds the total at ~1.2 s without any
  bandwidth recovery.
- **The reference exists and its geometry is already ours**, so the port is a
  routing wrapper plus a staging pass, not a new tile.

## What is not established

- No kernel was written or measured. The ~1.0-1.1 s is a traffic-model
  projection at constant bandwidth, not a timing.
- The 32-row 8-warp block ran at half the occupancy of the 16-row owner ("one
  block per CU instead of two"), and that cost is what turned a 6.3% traffic win
  into a 1.3-2.2x loss. Whether explicit LDS sharing plus the removal of the
  per-lane gather recovers more than it pays is **unmeasured**. The gather
  removal is a second, independent effect that the j32 variant did not have.
- The added `__syncthreads()` per K chunk (2-3 per block at K=640) is assumed
  cheap; that is an assumption, not a measurement.

## Next

Build the variant as one unit: `TILE_ROWS=32` with a cooperative LDS weight
staging pass, `lds_w` sized as above, the per-lane gather replaced by an LDS
read, and a bit-exact parity test against
`selected_wmma_iu8_risk_*_j16` on real routing. Measure at real expert
populations with `--pmc GCEA_RDRAM_SIZE_REQ_sum` before any gate, because the
whole question is whether the real traffic follows the model once the sharing is
explicit. If the traffic halves and the time does not, the loss is occupancy and
the answer is a smaller tile with the same staging, not a rejection of the path.
