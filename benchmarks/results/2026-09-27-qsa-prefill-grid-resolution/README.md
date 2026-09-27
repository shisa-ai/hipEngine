# QSA prefill: the grid/head_count anomaly is a profiling artifact, and K/V reuse is measured dead

**Date:** 2026-09-27
**Hardware:** AMD Radeon 8060S (gfx1151), Framework Desktop
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL, code-p4096 prefill, production default
**Artifacts:** `resolution.json`, `head-tile.json`, `qsa-launch-probe.py`
**New runs:** one 1024-token prefill probe; one head-tile microbench at the real geometry.

## Verdict

**There is no grid/head_count anomaly.** `rocprofv3` reports total *work-items* in
`Grid_Size_X/Y`, not grid dimensions. The traced `(192, 1021, 1)` is a true grid of
`(6, 1021, 1)`, and `6 = query_heads / HEADS = 24 / 4` - exactly the metadata's
`head_count` 24. The "768" came from reading `Grid_Size_X` as a grid dimension.

**And the K/V-reuse lever is measured dead.** Halving K/V traffic from HEADS=2 to
HEADS=4 buys **1.08x**. The shipped HEADS=4 already captures essentially all the
available amortization, so the kernel is not K/V-traffic limited - which kills
both arithmetic-preserving candidates at once and leaves tensor cores as the only
lever on the real bottleneck.

## 1. Resolving the observation

The kernel launches with `dim3(query_heads / HEADS, rows)` at `dim3(32)`
(`qwen4_exp_qsa.hip:1778`), unchanged since 2026-09-08. With the production
`HEADS=4` that is `dim3(6, rows)`.

`rocprofv3` multiplies by the workgroup size. Verified exactly on three kernels
whose launches are known from source:

| kernel | launch (source) | reported | check |
| --- | --- | --- | --- |
| `qwen35_moe_group_scatter_gather_kernel` | `dim3(10240), dim3(256)` | `(2621440, 1)` | 10240 x 256 = 2621440 |
| `weighted_lanes_sum_out_kernel` | `dim3(20, 1024), dim3(128)` | `(2560, 1024)` | 20 x 128 = 2560 |
| `qsa_sparse_attention_h256_wave_rows_f32_kernel<true,4>` | `dim3(6, rows), dim3(32)` | `(192, 1021)` | 6 x 32 = 192 |

So the real grid is `(6, 1021, 1)`.

**Where 192 came from:** `query_heads/HEADS x workgroup_size = 6 x 32`. Dividing by
nothing and reading it as heads gives `192 x 4 = 768`. The correct reading gives
`6 x 4 = 24`.

The model metadata, read directly from the GGUF, agrees: `head_count` 24,
`head_count_kv` 2, `key_length` 256. The production route is
`strict_h256_head_quad_rows_spans` (HEADS=4), selected by the profile, which sets
`HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL=page256` and
`HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR=quad` itself - not by anything in the benchmark
command, which is why an env-only reading misses it.

**The y-value is `sparse_rows`, not the chunk length.** `dense_rows =
max(0, min(count, dense_equivalent_limit - start))` and `sparse_rows = count -
dense_rows`, with `dense_equivalent_limit = block_budget x compression_ratio +
compression_ratio - 1 = 512 x 4 + 3 = 2051`. For code-p4096's four 1024-token
chunks that gives sparse rows 0, 0, **1021**, **1024** - matching the two observed
grids, 12 launches each over the 12 QSA layers (3, 7, ..., 47).

## 2. Where the family's time is

The `qsa_attention` family, at 1x, from the role trace. Both `ms` and `n` are
1x per-prefill values; the raw CSV holds two repetitions.

| kernel | ms | n | share |
| --- | ---: | ---: | ---: |
| **`qwen35_paged_full_attn_decode_context_tensor_batch_kernel`** | **801.2** | 36 | **56.4%** |
| **`qsa_sparse_attention_h256_wave_rows_f32_kernel`** | **455.4** | 24 | **32.1%** |
| `qsa_score_f32_kernel` | 106.1 | 24 | 7.5% |
| `qsa_split_norm_rope_rows_f32_kernel` | 23.3 | 48 | 1.6% |
| `qsa_topk_expand_f32_i64_kernel` | 17.2 | 24 | 1.2% |
| `qsa_gate_context_f32_kernel` | 14.2 | 48 | 1.0% |
| `qsa_norm_rope_rows_f32_kernel` | 1.9 | 48 | 0.1% |
| `qsa_pool_norm_rope_f32_kernel` | 0.3 | 24 | 0.0% |
| `qsa_scatter_index_keys_f32_kernel` | 0.3 | 48 | 0.0% |
| **total** | **1419.9** | | |

The total reconciles with the shared taxonomy's 1402.5 ms, which is what makes
this a complete family listing.

**The sparse attention kernel is 32.1% of the family, not the majority.** The
largest single piece is the **full-attention** decode-context kernel at 801.2 ms
(56.4%), which the ranking below does not address at all - it was not examined in
this sprint. The sparse kernel runs at 9.49 ms per launch.

## 3. The ranking

The kernel amortizes each selected K/V load over `HEADS` query heads. The GQA
ratio is `24/2 = 12`, so the reuse ceiling is 12x and HEADS=4 captures a third of
it. `scripts/qwen4exp_qsa_head_tile_microbench.py` prices exactly this at the real
geometry (`--rows 1024 --stride 2051`), and the three variants are
**bit-identical** - so this measures the arithmetic-preserving ceiling directly:

| HEADS | blocks/row | K/V GB | median ms | effective GB/s | vs HEADS=4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 24 | 51.59 | 54.938 | 939.1 | 0.352 |
| 2 | 12 | 25.80 | 20.823 | 1238.8 | 0.929 |
| **4 (shipped)** | **6** | **12.90** | **19.335** | **667.1** | **1.000** |

**The 2 -> 4 step halves K/V traffic and buys 1.08x.** The curve has already
saturated: effective bandwidth *falls* from 1238.8 to 667.1 GB/s as the tile
widens, which is the signature of a kernel that has stopped being traffic-limited
and started being issue-limited. Extrapolating the last step, HEADS=12 would cut
traffic to 0.33x for roughly 8% - on a 455.4 ms kernel, about 36 ms.

That gives the ranking the task asked for, **scoped to the sparse kernel (32.1% of
the family)**:

| rank | lever | arithmetic | verdict |
| ---: | --- | --- | --- |
| **1** | **tensor cores** (WMMA `Q*K^T`, `P*V`) | **changing** | the only lever on the real bottleneck (instruction issue, not K/V bytes). Needs the production profile gate plus a registered strict fallback. |
| 2 | K/V reuse, HEADS 4 -> 6 or 12 | preserving (bit-identical) | **measured dead.** Ceiling ~8% on 455.4 ms (~36 ms). |
| 3 | LDS staging of the K/V tile | preserving | **dead for the same reason**, plus the block is a single 32-lane wave, so there is no intra-block cross-warp sharing to capture without a multi-wave restructure. |

Ranks 2 and 3 are the same lever - reduce K/V traffic - and the measurement says
K/V traffic is not what costs. The microbench docstring anticipated this: it
answers "the same question a row-tiling or LDS-staging rewrite would answer,
without writing one."

## What this does not establish

- **No kernel was written or measured.** Ranks 1 and 3 are analysis, not results.
- **The microbench uses synthetic K/V and the full stride as the selected count**
  (2051), which is a worst case; real per-row counts are lower. The *scaling*
  across HEADS is the measurement; the absolute GB/s is not a prefill rate.
- **One shape.** code-p4096, 1024-token chunks, one capture.
- **HEADS=6 and HEADS=12 were not run** - only 1, 2, 4 are instantiated. The ~8%
  figure is extrapolation from the 2 -> 4 step, not a measurement.
- **The ranking covers only the sparse kernel.** The 801.2 ms full-attention
  decode-context kernel is 56.4% of the family and was not examined; no statement
  here applies to it.
- **Correction (2026-09-27).** An earlier revision of this artifact read the
  `rocprofv3` CSV raw and filtered kernel names on `qsa`, which both doubled
  every figure (the trace is two repetitions) and dropped
  `qwen35_paged_full_attn_decode_context_tensor_batch_kernel`, whose name
  contains no `qsa`. The 910.8 ms / 73.6% / ~73 ms figures in that revision were
  wrong and are superseded by the table above. The HEADS-saturation conclusion is
  unaffected: it is a kernel-level microbenchmark at fixed geometry.
