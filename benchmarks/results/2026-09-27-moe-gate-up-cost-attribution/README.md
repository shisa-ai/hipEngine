# MoE gate/up: the cost is the matmul and the repair, and the tile-count lever is dead

**Date:** 2026-09-27
**Hardware:** AMD Radeon 8060S (gfx1151), Framework Desktop
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL, code-p4096 prefill, production default
**Artifacts:** `expert-population.json`, `expert-population.py`
**New engine runs:** one 1024-token prefill with all 48 layers instrumented.

## Verdict

`expert_gate_up` is **82.5% main matmul and 17.5% iu8 exact repair**. Packing,
maps, and reduction are **0%** of it - they are a separate family
(`moe_reduce`, 766.9 ms) and fusing them would not touch this cost.

Expert-population-aware tiling is **measured dead as a direction**. The real
routing skew is far worse than any balanced model (181x max/median, only 39.6% of
experts used), and it does make the balanced model wrong - but wrong by 6.6%, and
the tile count it predicts is not the binding constraint anyway.

## The attribution

The `moe:layers.*.expert_gate` roctx role decomposes exactly into the three
published families (826 launches, 22 symbols, 6596.2 ms):

| component | kernels | ms | share |
| --- | --- | ---: | ---: |
| **gate/up main matmul** | `gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_kernel<bf16,3>` 2708.8 + `gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_kernel` 66.3 | **2775.1** | **81.6%** |
| **gate/up repair** | `gguf_q4_k_selected_dual_sparse_exact_repair_bf16_kernel` 568.2 + `gguf_q5_k_selected_dual_sparse_exact_repair_bf16_kernel` 59.1 | **627.3** | **18.4%** |
| | | **3402.4** | = `expert_gate_up` |
| down main matmul | `q5_1_selected_wmma_iu8_risk_prefill_kernel` 1483.8 + `q8_0_selected_grouped_wmma_prefill_bf16_bf16_kernel` 104.9 | 1588.7 | |
| down repair | `q5_1_selected_sparse_exact_repair_row_publish_kernel` 659.4 + `q8_0_selected_sparse_repair_kernel` 134.2 | 793.6 | |
| | | 2382.3 | = `expert_down` |
| router logits (maps) | `qwen35_router_logits_f32_token_tile_dense_exact_kernel<4>` | 618.0 | |
| reduction | `weighted_lanes_sum_out_kernel` | 67.6 | |
| packing / scatter | `qwen35_moe_group_scatter_gather_kernel` | 52.1 | |
| activation | `silu_mul_dual_out_kernel` | 29.2 | |
| | | 766.9 | = `moe_reduce` |

The two family sums reproduce the published numbers to the decimal, which is what
makes this an attribution rather than an estimate.

**Post-unroll, with the 1.240 s repair included:**

| component | ms | share of gate/up |
| --- | ---: | ---: |
| main matmul | 2786.1 | **82.5%** |
| iu8 exact repair | 591.0 | **17.5%** |
| packing + maps + reduction | 0.0 | **0%** |

The gate/up repair share is derived from the pre-unroll 627.3 ms scaled by the
**measured** 5.8% gain from the unroll pragma on that inner loop
([repair cost structure](../2026-09-17-qwen4exp-iu8-repair-cost-structure/README.md)).
That same packet already establishes what sets the repair cost, and it is not
bytes: cutting the repair's traffic 3.3x is 22% slower, four grid sizes from
17920 to 262144 are within 4%, and only 13.8% is recovered by making the
re-read activations L2-resident. It is bound by **memory-level parallelism**.

So the answer to "attribute before fusing" is: **there is nothing to fuse.** The
epilogue, the maps and the reduction are not in this family. The cost is a
weight-streaming matmul plus a sparse repair whose floor is already measured.

## Real expert population, all 48 layers

The shipped tile map pads every expert to a multiple of `tile_rows = 16`
(`qwen35_moe_wmma_tile_map_kernel`, `group_scatter.hip:328`), so the tile count is
`sum(ceil(count_e / 16))`. Instrumented on a 1024-token prefill:

| quantity | value |
| --- | ---: |
| assignments per layer | 10240 (1024 tokens x top-10) |
| experts used | **203.0 / 512 = 39.6%** |
| median expert count | 5.9 |
| p95 | 339.9 |
| max | 937.6 |
| **max / median** | **181.5x** |
| tiles, real skew | **772.9** |
| tiles, balanced model | 725.2 |
| balanced model error | **-6.6%** |
| tiles, ideal (no padding) | 640.0 |
| padding overhead | 1.21x |

The engine's `wmma_total_rows` matched the model exactly on every one of the 48
layers, so the model is verified, not assumed.

**The balanced model is genuinely wrong.** One expert takes 9.2% of a layer's
assignments while the median expert takes 6 rows; 61% of experts are never
touched at all. Any tile-count argument built on a balanced assumption is
quantitatively off.

**But it is wrong by 6.6%, not by a multiple.** And the tile sweep shows where the
lever would be if tile count were the cost:

| tile_rows | tiles | vs shipped |
| ---: | ---: | ---: |
| 8 | 1394.1 | 1.80x |
| **16 (shipped)** | **772.9** | **1.00x** |
| 32 | 468.8 | 0.61x |
| 64 | 322.0 | 0.42x |

## Why the tile lever is dead

The 2026-09-22 j32 packet widened the tile to 32 rows and measured **1.3-2.2x
slower with only a 6.3% real DRAM reduction** - against a model that predicts a
39% reduction. Its own conclusion is the falsification:

> "The tile-count traffic model is **necessary but not sufficient**: it counts
> *distinct* weight reads, and a taller tile makes the reads shared rather than
> distinct. Sharing only pays if it is explicit."

L2 already deduplicates repeated reads of one expert's weights across that
expert's tiles, so tile count over-predicts DRAM traffic badly. This is why the
measured 6.3% and the modeled 39% are both correct statements about different
things.

**Consequence:** an expert-population-aware tile - whether it adapts height per
expert or reorders to pack the tail - moves a quantity that is not the
constraint. The padding overhead is 1.21x of ideal tile rows, so even eliminating
every padded row caps the tile-count win at 21%, and the real DRAM win would be a
fraction of that, bought with the occupancy loss that already killed the 32-row
tile.

The lever the evidence does support is the one Sprint H scoped: make the weight
sharing **explicit** by staging the quantized weight chunk in LDS once per block
instead of each lane gathering its own column's blocks from global memory. That
is a different weight path, not a tile-height parameter.

## What this does not establish

- **No kernel was written or measured.** No tile variant was run; the sweep is a
  count over the measured histogram, and the verdict against it comes from the
  prior 32-row measurement, not from a new one.
- **One shape.** code-p4096, 1024-token chunk, one prefill. Routing skew at other
  token counts and other prompts is not measured here.
- **Not a rate.** The expert-population numbers are counts, not timings.
- **The repair share is derived, not captured.** 591.0 ms is the pre-unroll
  627.3 ms scaled by the measured 5.8% unroll gain; the post-unroll capture
  reports only the 1240 ms combined figure.
