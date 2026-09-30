# p512 prefill family breakdown - and a correction to GD-3

**This packet measures where prefill time actually goes, and in doing so corrects
a 4.5x error in the GD-3 result committed just before it.**

## The measurement

`rocprofv3 --kernel-trace` over `qwen4exp_profile_gap.py --mode prefill
--case-id code-p512`, one repetition, production profile, cached build, at the
newly promoted chunk 1024. Single process (the harness has no `subprocess` use),
so profiling it is safe.

**5699 kernel dispatches, 3914.468 ms total kernel time** - about two passes
(3914 / 2014 ms ≈ 1.94), which is the warmup plus the measured repetition.

| family | calls | total ms | share |
| --- | ---: | ---: | ---: |
| **moe** | 576 | 1855.127 | **47.39%** |
| **gr** | 654 | 1338.149 | **34.18%** |
| other | 3231 | 314.040 | 8.02% |
| **gdn** | 72 | 194.487 | **4.97%** |
| linear | 662 | 117.020 | 2.99% |
| attention | 48 | 49.856 | 1.27% |
| norm | 200 | 16.178 | 0.41% |
| conv | 146 | 13.625 | 0.35% |
| **qsa** | 96 | 8.418 | **0.22%** |
| cast | 6 | 6.252 | 0.16% |
| ple | 6 | 1.262 | 0.03% |
| embed | 2 | 0.054 | 0.00% |

Top individual kernels:

| calls | total ms | share | kernel |
| ---: | ---: | ---: | --- |
| 94 | 852.157 | 21.77% | `gguf_q4_k_selected_dual_wmma_iu8_risk_prefill` |
| 268 | 706.377 | 18.05% | `gguf_k_prefill_out_coltile_rowbatch_pipelined` |
| 192 | 613.970 | 15.68% | `q8_0_gr_up_sigmoid_mean_coltile2_branch4` |
| 86 | 478.917 | 12.23% | `q5_1_selected_wmma_iu8_risk_prefill` |
| 72 | 194.487 | 4.97% | `qwen4_exp_gdn_prefill_f32_kernel<true, true>` |

**Prefill is 81.6% MoE + GR.** Those two families, and four individual kernels
within them, are where the time is. Everything else is under 5% each.

**A classifier correction, recorded because it changes a conclusion.** The first
revision of `scripts/qwen4exp_family_share.py` matched the `qsa` pattern against
any name containing `sparse`, which swept in
`*_selected_sparse_exact_repair_*` - **MoE expert-repair kernels**, where
"sparse" means sparse *expert selection*, not sparse attention. That overstated
the QSA slice as 7.90%. Corrected, the pattern is `qsa_|block_pool|selection`
and the MoE patterns explicitly claim `selected_sparse|sparse_exact_repair|sparse_repair`:
**the real QSA kernels total 8.418 ms = 0.22%** (`qsa_split_norm_rope_rows`
0.14%, `qsa_gate_context` 0.06%, `qsa_norm_rope_rows` 0.01%,
`qsa_scatter_index_keys` 0.00%), and MoE rises from 39.70% to **47.39%**.

Classification is otherwise by kernel-name substring, which is deliberately
crude and therefore auditable: every bucket is printed with its member kernels
so a mis-filing is visible rather than hidden inside a total. That is exactly
how this error was caught, and it is why the two largest families have their top
kernels listed above.

## Correction to GD-3

The GD-3 entry committed immediately before this packet states the recurrence
lever is "bounded at **1.1%** of prefill", derived as:

> 9 certified GDN layers x 4.961 ms/1K = 22.7 ms at p508 against today's 2014 ms
> p512 prefill = 1.1%

**The layer count is wrong.** 9 is the *numerically certified* layer subset
(global layers 35-47), not the number of GDN layers in the model. The model has
**48 layers = 36 GDN + 12 full attention**, established three independent ways
from this same trace:

- GDN recurrence dispatches: **72** = 36 layers x 2 passes.
- Router dispatches: **96** = 48 layers x 2 passes.
- Conv dispatches: **144** = 2 convs x 36 GDN layers x 2 passes.

**Corrected ceiling: 36 x 2.54 ms at p512 = 91.4 ms of 2014 ms = ~4.5%**, which
matches the measured 4.97% (194.487 / 3914.468) directly. So the recurrence
lever is bounded at **~5% of prefill, not 1.1%** - 4.5x larger than the GD-3
entry says.

**The direction of GD-3's conclusion is unchanged** - the row's cited baseline
still matches no registered route, and gufo's quoted 65 ms is still ~13x slower
than our production route - but its cost gate was overstated by 4.5x, and
GD-3's second paragraph ("the GDN family's cost is overwhelmingly outside the
recurrence") is **wrong**: the family is 194.487 ms and the single recurrence
kernel accounts for all of it. There is no hidden 97% to redirect to. The
GDN-adjacent stages are small and separately classified: conv 13.625 ms (0.35%)
and norm 16.178 ms (0.41%).

## Why this also confirms GD-3's mechanism with hardware evidence

The trace records dispatch resources. The recurrence runs:

| field | value |
| --- | --- |
| `VGPR_Count` | **256** |
| `SGPR_Count` | 128 |
| `LDS_Block_Size` | 0 |
| `Scratch_Size` | 36 |
| `Workgroup_Size_X` | 128 |
| `Grid_Size_X` | 6144 |

**256 VGPRs per thread is the hardware maximum.** At 256 x 32 lanes x 4 B =
32 KB against a 64 KB VGPR file per SIMD, that permits **2 waves per SIMD** out
of a 64-wave-per-CU maximum - roughly 12.5% occupancy. So GD-3's
"latency-bound" reading was right, and the cause is specifically **register-file
capacity**, not issue width. That is the same conclusion QA-2 reached
independently for the QSA kernel from a static VGPR census.

This replaces GD-3's inference-from-roof-distance with a recorded dispatch
resource, which is strictly better evidence.

## What this changes

- **GD-3's ceiling is ~5%, not 1.1%.** Still not the largest lever, but no
  longer negligible.
- **GD-4's ceiling is the same ~5%** plus the fused conv/SiLU and norms
  (0.35% + 0.41%), so ~5.5% - it is the more complete version of the same idea
  rather than a separate prize.
- **The actual prefill prize is MoE (39.70%) and GR (34.18%).** Four kernels
  account for 67.7% of all kernel time. That is where optimization effort
  belongs, and it is a different row.

## Limits

One host (gfx1100), one model, one quant, one case (`code-p512`), one repetition
plus warmup, production profile, chunk 1024. Shares are of **kernel time only** -
host time, gaps between kernels and launch overhead are excluded, so these are
shares of the kernel total (3914 ms), not of the 2014 ms wall. Family
classification is by name substring and is heuristic; the member kernels for the
two largest families are listed so the assignment can be checked. The 2-pass
inference is consistent with the wall-time ratio but is an inference.
