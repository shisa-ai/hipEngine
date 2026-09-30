# Qwen4Exp optimization ledger: completion audit

The ledger's objective was to work through every remaining optimization candidate, keep
what improves performance, and update the results. This packet is the audit that closes
that loop: what the ledger's open rows target, what each target actually costs, and which
of them are worth building.

The audit's method is a **cost gate**, not a re-litigation of each row's argument. Both
phases now have a measured per-family breakdown, so every open row can be charged against
the slice it can move. A row whose ceiling is 0.2% of prefill is not wrong; it is
mis-ranked, and the ranking is what decides what gets built.

## Both phases are dominated by the same two families

Prefill: `rocprofv3 --kernel-trace` of a p512 prefill, production profile, cached build,
chunk 1024, one repetition plus warmup; 5699 dispatches, 3914.468 ms kernel time, ~2 passes.
Decode: the X-4 capture's steady-state repetition (`rep2`), 1771 dispatches, 54.173 ms.

| family | prefill ms | prefill share | decode ms/step | decode share |
| --- | ---: | ---: | ---: | ---: |
| MoE (routed + shared + router) | 1855.127 | **47.39%** | 24.733 | **45.66%** |
| GR / dense linear | 1338.149 | **34.18%** | 23.121 | **42.68%** |
| GDN | 194.487 | 4.97% | 2.374 | 4.38% |
| QSA + full attention | 58.274 | 1.49% | 2.589 | 4.78% |
| everything else | 468.431 | 11.97% | 1.356 | 2.50% |

**MoE + dense linear is 81.6% of prefill and 88.3% of decode.** The two phases have the same
shape, which is the single most useful fact in this audit: a lever that helps one family
helps both phases, and a lever aimed anywhere else is competing for a small remainder.

## Four prefill kernels are 67.7% of all prefill time

| calls | ms | share | kernel | status |
| ---: | ---: | ---: | --- | --- |
| 94 | 852.157 | 21.77% | `gguf_q4_k_selected_dual_wmma_iu8_risk_prefill` | fastest registered owner (`ED-4`) |
| 268 | 706.377 | 18.05% | `gguf_k_prefill_out_coltile_rowbatch_pipelined` | already the two-stage K-pipelined form |
| 192 | 613.970 | 15.68% | `q8_0_gr_up_sigmoid_mean_coltile2_branch4` | `HC-1`: 2.388x screen, **gate-rejected** |
| 86 | 478.917 | 12.23% | `q5_1_selected_wmma_iu8_risk_prefill` | production owner |

Each was checked against its own ledger row rather than assumed optimal:

- **`ED-4`** records that `selected_wmma_iu8_risk_prefill` is not named by any `_selection()`
  entry but *is* what production runs, and that it is the fastest registered owner. The
  ledger already corrected its own earlier claim to the contrary.
- The 18.05% kernel's name contains `pipelined`, and its source comment states the
  difference from the unpipelined form is only *where the loads are issued*: "the next
  iteration's weight and activation loads are issued before the current iteration's FMAs
  retire... Per-output arithmetic is untouched." **The prefetch pattern `GD-3` proposes is
  already production here.** This is why the pattern's novelty had to be assessed from the
  trace, not from the row.
- **`HC-1`** measured 2.388x with the arithmetic intact and was then **rejected by its own
  production gate**: `mixed_ja_en` mean KL 1.499e-3 against the 1e-3 limit, with broad-based
  drift rather than a tail redistribution. The 2.388x is a real timing result that the
  route cannot reach at production precision. It stays default-off.

## The audit table

Every open or blocked row, charged against its own measured target. "Ceiling" is what a
perfect fix could recover, not what the row claims it will recover.

| row | target | measured ceiling | verdict |
| --- | --- | ---: | --- |
| `RT-1` | serving prefill chunk | — | **PROMOTED**: -29.0% to -35.7% wall, 12/12 cases |
| `DE-1` | decode routed experts | **0.591 ms/step** | **PROMOTED**; 93.7% of headroom consumed, unit (2) re-ranked |
| `GD-4` | chunked GDN on fp16 WMMA | ~5.5% prefill | real, 5x bigger than `GD-3` claimed, below MoE/GR |
| `GD-1` | GDN column warps + peer | inside noise | low |
| `DP-9` | dense shapes outside the wide class | **~35% of decode** | **highest-value open row**; owner now named |
| `ED-3` | Q5_1 / Q8_0 split populations | inside MoE 47.39% | real but inside the biggest family |
| `MR-4` | MoE epilogue folded into HC combine | inside MoE 47.39% | standalone fused combine already lost 1.69% |
| `MR-3` | reduction fusion | caps at 8.6% of its bucket | low |
| `QA-2` | QSA sparse attention on tensor cores | **0.22% prefill** | **cost-gated out of its Tier-B build** |
| `QA-4` | indexer scoring / selection epilogues | attention 1.27% | low |
| `QA-6` | raw index-key ring ownership | attention | memory, not time |
| `QA-7` | multi-block parallel threshold | attention | low |
| `DP-2` | one cast shared by two consumers | 7.478 ms packet estimate | tiny |
| `DP-4` | F16 vs BF16 operands | evidence-gated | not a presumed speed win |
| `RT-5` | prefix reuse with state snapshots | serving lane | no local gain recorded |
| `RT-6` | private runtime controls, graph updates | serving lane | prerequisite only |
| `DE-6` | PM4 / retained command lists | runtime | decode gaps, not kernel time |
| `DE-8` | head | decode | inside the 42.68% linear family |
| `DE-9` | constrained serving | serving lane | separate lane |
| `DE-10` | concurrent serving | serving lane | separate lane |
| `MT-2` | capacity beyond 1K | **blocked by the 1,024 cap** | provider + draft + request all enforce it |
| `MT-3` | adaptive draft depth | blocked on `MT-2` | — |
| `MT-4` | cheaper draft head | **measured: not the blocker** | deleting the whole head still gave 0.964x AR |
| `MT-6` | compact recurrent rollback | memory, not time | — |
| `MT-8` | duplicate replay forwards | diagnostic | — |
| `RT-2`, `MR-4` | — | **blocked on `MX-1`** | — |

## The one row the audit promotes in ranking

`DP-9` was filed "OPEN - measured-owner prerequisite" and reads as a prerequisite, not a
prize. The measurement inverts that. Splitting the decode `linear` family by kernel shape
shows **five of its six largest shapes share one launch signature** - `wg=128`, `VGPR=48`,
`LDS=512` - at different grid sizes, and together they are **81.9% of the family = 34.9% of
decode**:

| grid | calls | ms/step | % of decode | owner slots |
| ---: | ---: | ---: | ---: | --- |
| 163840 | 177 | 12.266 | 22.6% | `attn_qkv`, `expert_gate` |
| 40960 | 144 | 18.511 | 34.2% | `expert_gate` |
| 5120 | 97 | 3.586 | 6.6% | `hc_attn_down`, `hc_ffn_down` |
| 98304 | 36 | 2.901 | 5.4% | `attn_gate` |
| 196608 | 12 | 1.817 | 3.4% | `attn_q` |

Two facts make this actionable rather than merely interesting. First, the *same* signature
serves both `linear:*` and `moe:*` roles, so a single kernel improvement crosses both of the
two dominant families. Second, the comparator is in the same capture: `linear:root.lm_head`
is a single `wg=128` GEMV streaming 675.43 MB in 2.965 ms = **227.8 GB/s, 103% of the
221 GB/s roof**, while the routed-expert owner of the same era measured **63.43 GB/s = 28.7%
of roof**. Same host, same step, same file, same block width - a 3.6x spread.

### The identity was then resolved, and it tempers the claim above

The `x4full` stem carries a rocpd symbol table (`rocpd_info_kernel_symbol_*`) keyed by the
same `Kernel_Id` the renamed kernel trace uses, so the role **and** the real kernel name can
be recovered together from one stem. Doing that gives the decode's true top three:

| calls | ms/step | share | wg | grid | kernel |
| ---: | ---: | ---: | ---: | ---: | --- |
| **497** | **23.846** | **44.55%** | 128 | 5120 | `gguf_k_pack8_prefill_out_kernel<float,float,8>` |
| 47 | 13.968 | 26.10% | 64 | 40960 | `gguf_q4_k_selected_dual_silu_logical128_t64` |
| 43 | 5.420 | 10.13% | 64 | 163840 | `q5_1_selected_weighted_sum_logical256_t64_bf16` |

**One kernel is 44.55% of decode**, and it is the `pack8` GEMV family - which is the family
`DE-3` already measured and optimized. It serves *every* dense slot, `root.lm_head` among
them, and its name says `prefill` while it runs 497 times per decode step.

**That tempers rather than confirms the ranking above.** The largest owner is not an
unexamined latency-bound shape: `DE-3` measured this exact family at `out=320` running
40 workgroups at **92.4 GB/s (42% of roof)** and every shape at 2560+ at **90.8-102.9%**,
then widened the block 128 -> 256 at the two starved shapes for -32.4% and -19.4% of kernel
time. Its family figure was **24.005 ms/step**, matching the 23.846 measured here. So the
dominant decode kernel is *already near roof at wide shapes and already fixed at narrow
ones*, and its remaining gap is the small narrow shapes `DE-3` did not cover rather than a
structural one.

The honest reading: `DP-9`'s target is real and large, but the first thing its experiment
should do is **subtract what `DE-3` already captured**, because the largest owner is the one
already optimized. The `grid=163840` and `grid=40960` rows in the table above are the
`expert_gate` shapes of a *different* kernel pair and remain the less-examined part.

**Caveat on the shape table.** The shape split is by launch signature and a signature is not
unique - in the real-name prefill trace both `q5_1_selected_sparse_exact_repair_row_publish`
and `gguf_k_pack8_prefill_out_kernel` carry `wg=128, VGPR=48, LDS=512`. The symbol-table join
above removes that ambiguity for the named kernels; the shape table is retained because it is
what showed the `linear`/`moe` sharing in the first place.

## What this audit does not claim

- No kernel, default, flag, or dispatch changed. This is a measurement and ranking packet.
- The per-family figures are **sums of kernel durations**, so inter-kernel gaps are excluded
  by construction. That is deliberate: the X-4 packet established the decode kernel sum is
  stable to 1.0% across repetitions while the step span varies 2.8x, so the kernel sum is the
  reproducible quantity and the wall is not.
- The decode figures come from a profiled capture whose per-kernel times are diagnostic.
  They are used for **relative ranking within one capture**, which is what a cost gate needs.
- The prefill classification is by kernel-name substring and is deliberately crude; every
  bucket is printed with its member kernels so a mis-filing is visible. That is how the QSA
  mis-filing below was caught.

## A correction this audit had to make to itself

The first revision of the prefill classifier matched its `qsa` pattern against any name
containing `sparse`, which swept in `*_selected_sparse_exact_repair_*` - **MoE
expert-repair** kernels, where "sparse" means sparse *expert selection*, not sparse
attention. That overstated QSA as 7.90%. Corrected, the real QSA kernels total **8.418 ms =
0.22%**, MoE rises 39.70% -> **47.39%**, and every attention-arithmetic row in the ledger
(`QA-2`, `QA-4`, `QA-6`, `QA-7`) shares a combined ceiling of ~1.5% of prefill.

## Files

- `family-share-both-phases.json` - both family breakdowns, machine-readable.
- Prefill source trace: `benchmarks/results/2026-09-30-qwen4exp-gd4-family-share/`
- Decode source trace: `benchmarks/results/2026-09-29-qwen4exp-x4-decode-split/`
- Classifier: `scripts/qwen4exp_family_share.py`
