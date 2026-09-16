# Qwen3.8-Flash-Next (Qwen4Exp) Optimization Status

- **Status date:** 2026-09-17
- **Model / quant:** unsloth `Qwen3.8-Flash-Next` `UD-Q4_K_XL`, fingerprint
  `fb1f2fbf73d588c9…`, 111.3 GB, four shards
- **Host:** Framework `gfx1151`, Radeon 8060S / Strix Halo, machine
  `55ea6c509d0b49eea8de7094a1023668`
- **Current default:** named production profile, chunk1024, BF16 KV, warm PLE,
  and since `04f1dde42` the certified Q8 WMMA dense prefill scope at layers
  16-47 (`PRODUCTION_Q8_WMMA_PREFILL_LAYERS`)

**This document is rewritten in place. It is the current-state tracker for this
model; the `QWEN4EXP-*` and `QWEN3.8-FLASH-NEXT-*` campaign documents are dated
records and are not edited to reflect new state.** Every number below carries a
pointer to the artifact that owns it. Where a number is derived rather than
measured, or is an interpolation rather than a measurement, it says so.

The campaign's shape is simple: the comparators are faster because they ship a
list of mechanisms we have not adopted, and each item on that list has a
numerical price under our envelope. Sections 3 and 5 are that list.

## 1. Our own progress

Rates are tokens/s. The two blocks use **different protocols** and their rows
are not comparable to each other; each block declares its own.

### 1.1 Journey screen — PP/TG at 512/1024/4096

Protocol: canonical exact-token fixture, 4 categories, one warmup and three
repetitions, 128 AR transitions, token/time-weighted, matched cache/clock/
thermal policy. Source:
[`benchmarks/results/2026-09-13-framework-qwen4exp-strix-journey-baselines.json`](../benchmarks/results/2026-09-13-framework-qwen4exp-strix-journey-baselines.json).

| Arm | p512 PP / TG | p1024 PP / TG | p4096 PP / TG |
| --- | ---: | ---: | ---: |
| hipEngine production, **journey start** (2026-09-13) | 297.1 / 20.40 | 316.9 / 19.71 | 294.1 / 19.17 |
| hipEngine production, current | not re-measured | not re-measured | not re-measured |

This is the campaign's starting baseline and it is also, as of this date, the
newest measurement of our own PP **and** TG on one declared protocol. The
September 16-17 work certified routes and fixed dispatch and provenance, and on
September 17 it **did** change the default path: `04f1dde42` promoted the
certified Q8 WMMA dense prefill scope to the production default at layers 16-47,
measured on `code-p4096` at 19.054 s against the old default's 23.596 s
([worklog](../worklog/entries/20260916T192306.422564Z-lhl-q8-wmma-1647-promotion-b024aa.md)).
The 2026-09-13 row above is therefore **not** HEAD's default path, and the
byte-identity argument below only covers the days before that promotion: the
September 17 attribution run at `725794c3f` reproduced the production profile's
`logits_sha256` and `token_id` exactly (§1.3), and it predates the promotion.
Re-running this screen at HEAD is still the first measurement owed by this
document, because a reproduction is not a rate, and it now has two default
changes to absorb rather than none.

### 1.2 Cross-engine prefill — canonical fixture, equal-weight mean

Protocol: prefill only (`n_predict=1`), exact token ids, 12 cases (4 categories
at 512/1024/4096), one warmup, three repetitions, median per case then
equal-weight mean across cases. Source:
[`benchmarks/results/2026-09-15-flashnext-engine-comparison/README.md`](../benchmarks/results/2026-09-15-flashnext-engine-comparison/README.md).

| Engine | 512 | 1K | 4K |
| --- | ---: | ---: | ---: |
| **hipEngine production** (`ddfc2a746`) | **185.2** | **190.6** | **183.6** |
| pwilkin `strix-halo` `40a9f4d01`, f16 | 339.2 | 858.9 | **1061.9** |
| halo-box `strix-llama.cpp` `69946438a`, f16 | 451.6 | 620.7 | 660.6 |
| upstream llama.cpp `6011c34ce`, f16 | 321.3 | 415.1 | 459.0 |

Read the two blocks against each other with care: 294.1 tok/s at p4096 in 1.1
and 183.6 in 1.2 are the same engine on different protocols, not a regression.

### 1.3 Kernel time per 4096-token prefill

Protocol: role-marked `rocprofv3` capture with `--launch-census` in the same
process, one `code-p4096` prefill. Source:
[`benchmarks/results/2026-09-17-qwen4exp-per-role-cost/README.md`](../benchmarks/results/2026-09-17-qwen4exp-per-role-cost/README.md),
which re-runs the 2026-09-16 protocol at `725794c3f` and reproduces it (same
9652 kernels, same `logits_sha256`, window −1.0%).

| Measure | Value |
| --- | ---: |
| Prefill wall | 22699 ms |
| Attributed kernel time | 22083.6 ms over a 22698.6 ms window, 9652 kernels, **0 ms unattributed** |
| Matmul / non-matmul / risk-or-repair | 14390 / 6309 / 1373 ms |
| Dense projections | 10562 ms (47.8%) through **one** kernel family, `gguf_k_prefill_out_coltile_rowbatch_kernel` |
| Dense projections, achieved rate | 2650 GFLOP/s = **9.3%** of the 28521 GFLOP/s register-resident FP32 measurement |
| Largest single role | `moe:expert_gate` 6619 ms (30.0%) |

The dense projection row is the tuning target, and §2.3 shows it is also half
the gap to the fastest comparator: 47.8% of our prefill runs through one kernel
family at under a tenth of the machine's measured FMA rate.

## 2. Versus the competition

Competitor numbers are **not accuracy-comparable to ours** and the table says so
per row. Their arithmetic is not gated by anything equivalent to
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) §6.1.

### 2.1 Kernel time, one 4096-token prefill

Us: 22083.6 ms attributed (see 1.3). Competitors, same host, same file:

| Engine | Prompt ms | Kernel sum ms | Source |
| --- | ---: | ---: | --- |
| **hipEngine production** | **22699** | **22083.6** | [`2026-09-17-qwen4exp-per-role-cost`](../benchmarks/results/2026-09-17-qwen4exp-per-role-cost/README.md) |
| halo-box `69946438a`, bf16 | 5569.2 | 5411.0 | [`2026-09-16-flashnext-delimited-components`](../benchmarks/results/2026-09-16-flashnext-delimited-components/component-gap.json) |
| halo-box PR #63 `c4aa30229`, bf16 | 3989.3 | 3760.4 | same |
| pwilkin `40a9f4d01` | — | 3972.8 | [`2026-09-15-flashnext-engine-comparison`](../benchmarks/results/2026-09-15-flashnext-engine-comparison/README.md) |
| upstream llama.cpp `6011c34ce` | — | 9633.5 | same |

The pwilkin and upstream kernel sums are whole-window totals divided by the
number of prefills the window held; that record's per-family bucketing is
withdrawn (its correction notes mis-assigned `quantize_mmq_q8_1`,
`qsa3_attn_kernel` and the rocBLAS `Cijk_*` GEMM), so their **totals** are
usable and their **categories** are not. The halo-box rows are request-delimited
and are the trustworthy per-family comparison.

### 2.2 Family-by-family, in one taxonomy

The 2026-09-16 comparison of our role table against the comparator's family
table compared two taxonomies: we carry the tensor role in a ROCTX range, so one
symbol serves several roles, and the comparator names the family in the symbol.
`scripts/qwen4exp_comparator_role_map.py` already expressed both in one
vocabulary and already classified the comparators; it gained a `hipengine`
mapper, so all three engines are now classified from kernel symbols by one
module with one family list. Source:
[`shared-family-comparison.json`](../benchmarks/results/2026-09-17-qwen4exp-per-role-cost/shared-family-comparison.json).

Same case, same host, kernel milliseconds. Not the same arithmetic.

| Family | ours ms | ours % | base ms | ratio | PR #63 ms | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense_projection` | 10562.2 | 47.8 | 1536.7 | 6.87x | 1374.3 | **7.69x** |
| `expert_gate_up` | 3414.2 | 15.5 | 819.4 | 4.17x | 761.8 | 4.48x |
| `hyper_connection` | 2523.8 | 11.4 | 428.9 | 5.88x | 335.7 | 7.52x |
| `expert_down` | 2389.0 | 10.8 | 1149.8 | 2.08x | 412.8 | 5.79x |
| `qsa_attention` | 1402.7 | 6.4 | 156.6 | 8.96x | 155.7 | **9.01x** |
| `gdn` | 821.2 | 3.7 | 287.6 | 2.86x | 263.8 | 3.11x |
| `moe_reduce` | 762.5 | 3.5 | 204.0 | 3.74x | 103.9 | 7.34x |
| `elementwise_norm` | 180.2 | 0.8 | 531.4 | **0.34x** | 331.7 | **0.54x** |
| `indexer` | 17.5 | 0.1 | 5.6 | 3.12x | 15.0 | 1.17x |
| `ple` | 9.9 | 0.0 | 0.0 | — | 0.0 | — |
| `other` | 0.2 | 0.0 | 6.2 | 0.03x | 5.7 | 0.04x |
| `quantize_pack` | 0.0 | 0.0 | 284.8 | — | 0.0 | — |
| **total** | **22083.6** | **100.0** | **5411.0** | **4.08x** | **3760.4** | **5.87x** |

Our 1434.4 ms of iu8 exact-repair is folded into the matmul family it corrects
and also totalled separately; the comparator has no equivalent pass, so that
1.43 s sits inside our `expert_gate_up` and `expert_down` rows.

### 2.3 Where the gap actually is

`22083.6 − 3760.4 = 18323.2 ms` against PR #63, decomposed:

| Family | Gap | Share |
| --- | ---: | ---: |
| `dense_projection` | 9187.9 | **50.1%** |
| `expert_gate_up` | 2652.4 | 14.5% |
| `hyper_connection` | 2188.1 | 11.9% |
| `expert_down` | 1976.2 | 10.8% |
| `qsa_attention` | 1247.0 | 6.8% |
| `moe_reduce` | 658.6 | 3.6% |
| `gdn` | 557.4 | 3.0% |
| `elementwise_norm` | −151.5 | −0.8% |
| remainder | 7.1 | 0.0% |

**Dense projection alone is half the gap; four families are 87% of it.** The
ratio column and the gap column disagree about priority, and the gap column is
the one that matters for tuning: `qsa_attention` has the worst ratio (9.01x) but
only 6.8% of the gap, because its absolute size on both sides is small.

Two readings the ratio column alone gets wrong:

- **`elementwise_norm` is not a gap; we are ahead.** 180.2 ms against PR #63's
  331.7 ms, 0.54x, with the comparator spending 8.8% of its kernel time there
  against our 0.8%.
- **`expert_down` is not one of our better families.** Against the base it reads
  2.08x, our second-best ratio; against PR #63 it is 5.79x, because PR #63 cut
  that family by 64.1% and we have no equivalent kernel. The base column flatters
  us on exactly the family where the comparator moved furthest.

### 2.4 How much of the gap we can close

§2.2 says where the gap is; this says how much of it we have evidence to take
back, and whether that evidence is admissible. Two sources, and their bases
differ:

- **The disabled-family sweep**
  ([`2026-09-16-disabled-family-recoverable-time`](../benchmarks/results/2026-09-16-disabled-family-recoverable-time/README.md))
  measured each disabled family as a post-binder override on this case, two
  repetitions, median, against a 23.648 s fallback. Those savings are **wall**
  seconds and **measured**.
- **The layer gate** ([§4](#4-certified-but-not-promoted)) certifies a *scope* and
  records no timing at all (`timing_protocol: none_full_logits_only_v1`). The
  saving attached to a scope comes from the recoverable-time record, and every
  figure below 32-47 there is a **byte-share prediction**; only 32-47 (+2.436 s)
  and 0-47 (+7.041 s) are measured.

Kernel time is 97% of wall on this case, so the two bases are close but they are
not mixed in one column.

| Family | ours ms | gap ms | recoverable | basis | Admissible today | residual gap | family gap closed |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `dense_projection` | 10562.2 | 9187.9 | **+4420** certified `Q8_WMMA_LAYERS` 16-47<br>+6614 `Q8_IU8_WMM` family-wide | prediction<br>measured wall | **yes at 16-47**<br>no verdict recorded | 4767.9 | **48.1%** |
| `expert_gate_up` | 3414.2 | 2652.4 | 0 | both overrides are *slower*: `Q4_IU8_PREFILL` −77 ms, `PRODUCTION_MOE_PREFILL` −595 ms | — | 2652.4 | 0% |
| `hyper_connection` | 2523.8 | 2188.1 | +2494 | measured wall | **no — rejected** | 2188.1 | 0% |
| `expert_down` | 2389.0 | 1976.2 | none measured | — | — | 1976.2 | 0% |
| `qsa_attention` | 1402.7 | 1247.0 | none measured | — | — | 1247.0 | 0% |
| `gdn` | 821.2 | 557.4 | +354 | measured wall, **inside the run-to-run spread** | n/a | 557.4 | 0% |
| `moe_reduce` | 762.5 | 658.6 | none measured | — | — | 658.6 | 0% |
| `elementwise_norm` | 180.2 | −151.5 | — | already ahead of the comparator | — | −151.5 | — |

**Residual and closed count only what is admissible.** The recoverable column
shows what is *measured*; a family whose recovery is rejected keeps its full
gap in the residual column, which is why `hyper_connection` shows +2494
recoverable next to an unchanged 2188.1 residual. That +2494 is also larger than
the family gap, so a corrected arithmetic there would close the family and
contribute to the total rather than merely removing that row.

**Admissible today: +4.42 s, which is 24.1% of the 18323.2 ms gap.** That takes
the total ratio from **5.87x to 4.70x** — and it is a prediction, not a
measurement, because the gate records no timing.

The `hyper_connection` row is the one worth dwelling on. `GR_IU8` and
`GR_IU8_DOWN` are **+2.494 s measured**, which would close that family's entire
gap and 13.6% of the total, and both are numerically **rejected**: mean KL
`1.4024e-3` and `1.3128e-3` against the `1e-3` limit, top-1 772/780 and 769/780
([`REFACTOR.md`](REFACTOR.md)). A corrected arithmetic for that family is worth
more than anything else on this list except dense, and it is blocked on
numerics rather than on engineering.

#### What the measured ceiling is

The sweep also measured its four largest families **together**, which is the
honest upper bound because the individual rows overlap:

| | median prefill s | saved | speedup |
| --- | ---: | ---: | ---: |
| fallback | 23.648 | — | — |
| sum of the four individually | — | +13.094 | (not reachable) |
| `Q8_IU8_WMM` + `Q8_MMQ_PREFILL` + `GR_IU8` + `GR_IU8_DOWN` | **15.840** | **+7.808** | **1.493x** |

**+7.808 s measured would close 42.6% of the gap and take the ratio to about
3.80x.** None of it is promotable as measured, and the reasons differ per row:

- `GR_IU8` and `GR_IU8_DOWN` are numerically **rejected** at this scope.
- `Q8_MMQ_PREFILL` is a diagnostic override, not a profile state.
- `Q8_IU8_WMM` family-wide has **no numerical verdict recorded at that scope**.
  The rejection we do hold is for a *different* selector on the same tensors:
  the f16 `Q8_WMMA_LAYERS` path, which fails at 0-47 (mean KL 1.099e-3) and
  passes at 16-47. The two are siblings, not the same route, so one route's
  verdict does not transfer to the other — the family-wide `Q8_IU8_WMM` number
  is unmeasured for quality, not measured-and-failed.

The promotable part of the ceiling is the certified 16-47 slice, which is the
+4.42 s above.

So the summary, in shares of the 18323.2 ms gap:

| | Share of gap |
| --- | ---: |
| `dense_projection` + `hyper_connection` — the only families where recovery is measured | **62.1%** |
| ...of which **admissible today** (certified dense 16-47, predicted) | **24.1%** |
| ...of which measured but not shippable (the +7.808 s combination) | 42.6% total |
| No candidate at all: `expert_gate_up`, `expert_down`, `qsa_attention`, `moe_reduce` | 35.7% |
| `gdn` — candidates measured, inside the run-to-run spread | 3.0% |
| `elementwise_norm` — we are ahead of the comparator | −0.8% |

The first two rows overlap by construction: the 24.1% admissible slice is inside
the 42.6% measured combination. Read it as **24.1 points of the gap are
shippable, another 18.5 points are measured but blocked, and 35.7 points have no
candidate at all.**

### 2.5 Where the comparator's time went, and what PR #63 changed

Same case, base versus PR #63, milliseconds. Source:
[`component-gap.json`](../benchmarks/results/2026-09-16-flashnext-delimited-components/component-gap.json).

| Family | base | PR #63 | delta |
| --- | ---: | ---: | ---: |
| dense_projection | 1536.7 | 1374.3 | −10.6% |
| expert_down | 1149.8 | 412.8 | **−64.1%** |
| expert_gate_up | 819.4 | 761.8 | −7.0% |
| elementwise_norm | 531.4 | 331.7 | −37.6% |
| hyper_connection | 428.9 | 335.7 | −21.7% |
| gdn | 287.6 | 263.8 | −8.3% |
| quantize_pack | 284.8 | **0.0** | −100% |
| moe_reduce | 204.0 | 103.9 | −49.1% |
| qsa_attention | 156.6 | 155.7 | −0.6% |
| indexer | 5.6 | 15.0 | +168% |
| **total (kernel sum)** | **5411.0** | **3760.4** | **−30.5%** |

PR #63 is 13.9% faster end to end on the same case with identical 48/48 output
([worklog](../worklog/entries/20260916T042440.370650Z-lhl-flashnext-halobox-pr63-verified-e29646.md)).

### 2.6 Accuracy basis

Max error against the exact F64 result on the identical-operand replay packet
(`layers.8.attn_qkv`, rows 1024, K 2560, M 10240). Source:
[`QWEN4EXP-EXTERNAL-FORKS-REVIEW.md`](QWEN4EXP-EXTERNAL-FORKS-REVIEW.md).

| Arithmetic class | Max relative vs F64 | Gated? |
| --- | ---: | --- |
| F32 coltile — our strict and production default | 1.51e-7 | inside the envelope everywhere |
| IU8 WMMA — `iu8_wmma_prefill` | 3.90e-7 | gated per scope |
| **F16 operands — our `dense_wide256`, `wmma_prefill`** | **2.08e-4** | gated: passes at layers 20-47, fails at 0-47 |
| **BF16 operands — the comparator's class** | **6.8e-3 – 7.2e-3** | **ungated** |

The comparator's class is about **33x coarser** than the path this campaign
gated. That row is *derived* — from our F16 kernel's recorded deviation from a
BF16-both reference — not measured on the comparator's own output; experiment
E10 closes it with the replay bridge that already captures that output.

Our bar is **production correctness**, not bit-exactness: a candidate that is
not bit-identical to the strict parent is admissible if it passes the calibrated
envelope. The consequence for this campaign is that we are not choosing between
their speed and our accuracy — both paths are inexact and theirs is more so.

## 3. What makes the comparators fast, and our state

Ordered by share of the 18323.2 ms gap to PR #63 (see §2.3), not by ratio.

**Read the "our state" column with care.** For `gguf_ud_q4_k_xl` the production
binder zeroes fifteen `PRODUCTION_ARITHMETIC_RECOVERY_FLAGS` and then restores
six `PRODUCTION_Q8_QSA_RESTORED_FLAGS`, so this quant's arithmetic comes from
the production **selection table** (keyed on rows and quant) rather than from
those flags. Reading the flag map therefore gives the wrong layer scope for
several routes; the scopes below are read from the kernel trace instead
([`role-analysis.json`](../benchmarks/results/2026-09-17-qwen4exp-per-role-cost/role-analysis.json)).

| Mechanism | Their evidence | Gap share | Our state |
| --- | --- | ---: | --- |
| Dense projection: quantized weights dequantized to BF16 in LDS, activations converted to BF16 **once per graph and cached**, F32 accumulate on BF16 WMMA (`mmb.cu`, PR #63 `08de004`) | PR #63 dense_projection 1536.7 → 1374.3 ms | **50.1%** | **Partly built.** `dense_wide256` ports the tile (`mmb_dense_kernel<128,256,64,64,1>`) at 2.573 ms against the production dispatch's 17.319 ms on the packet (6.74x). Two gaps: f16 rather than bf16 operands (deliberate), and per-launch activation conversion — with a pre-converted f16 activation the same kernel runs **1.303 ms** against the comparator's 1.339 ms, so the activation path is the remaining 1.93x |
| Routed MoE gate/up WMMA-iu8 selection | PR #63 expert_gate_up −7.0% | 14.5% | **On for every layer, and already the best ratio in the table.** `gguf_q4_k_selected_dual_wmma_iu8_risk_prefill` ran on layers 0-1 and 3-47; the `selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out` selection is a production manifest entry keyed on `prefill_rows_ge64_exact_grouped_q4_gate_up`. This is **not** the `PRODUCTION_MOE_PREFILL` route (that flag is `0` for this quant) and **not** a 27-47 scope, which is what the profile's own comment claims |
| Hyper-connection combine and mix (`hc_combine_norm_f32`, `hc_mix_reduce_f32`) | PR #63 hyper_connection −21.7% | 11.9% | **Open and diagnosed.** `q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32` ran on all 48 layers for 2437.4 ms, plus `gr_write` at 86 ms: 2523.8 ms against their 335.7 ms for the fused combine-plus-norm. The same tensor is traversed twice, 2523.8 ms of reads against 1098 ms of matmuls consuming them. The `GR_IU8` variants that would fix it are numerically **rejected**; see §2.4 |
| Routed MoE down MMB kernel (`mmb_routed_kernel`) | PR #63 expert_down **−64.1%** (1149.8 → 412.8 ms) | 10.8% | **Not built.** The largest per-family move the comparator made and we have no equivalent kernel. Our `expert_down` is 2389.0 ms, of which 721 ms is iu8 exact-repair. The down projection runs Q5_1 on 43 layers and Q8_0 on five (2, 4, 30, 46, 47) |
| QSA attention | PR #63 −0.6% (156.6 → 155.7 ms) | 6.8% | **On, but not as "QSA flash".** The binder zeroes `QSA_FLASH_PREFILL` for this quant and restores `QSA_H256_WAVE_PREFILL=page256` and `QSA_HEAD_PAIR=quad`; `qsa_sparse_attention_h256_wave_rows_f32` ran on the 12 attention layers (3, 7, …, 47). Our 1402.7 ms is 9.01x theirs, the worst ratio in the table, but only 6.8% of the gap |
| MoE reduction fusion | PR #63 moe_reduce −49.1% | 3.6% | **Open.** Our 762.5 ms covers router logits and select, group scatter/gather, the tile map and the weighted-lane reduction |
| GDN prefill | PR #63 −8.3% | 3.0% | **The base kernel, not the column-warp variant.** `GDN_COLWARPS_PREFILL` is `0` for this quant, so `qwen4_exp_gdn_prefill_f32` ran on the 36 GDN layers. Enabling column warps is worth **+0.228 s measured**, inside the run-to-run spread; our 821.2 ms is 3.11x |
| Elementwise and norm fusion | PR #63 −37.6% (531.4 → 331.7 ms) | **−0.8%** | **Not a gap: we are ahead.** 180.2 ms against their 331.7 ms, 0.54x, with the comparator spending 8.8% of its kernel time there against our 0.8%. An earlier revision of this document listed this as open; §2.2 retired that |
| Activation packing elimination | PR #63 quantize_pack 284.8 → **0.0 ms** | 0.0% | **Done.** We have no packing row at all |
| Dense Q8 prefill tiling | pwilkin dense variants 1387.5 ms per prefill against our 10562 ms | — | **Certified, not promoted.** F16 WMMA dense Q8 at layers 16-47 passes every calibrated gate; see §4 |
| Indexer | PR #63 +168% | 0.0% | **Not a target** — their regression, 15.0 ms absolute |

## 4. Certified but not promoted

Numerical verdicts are from the calibrated envelope in
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) §6.1. Sources:
[`2026-09-16-q8-wmma-dense-prefill-layers-gate`](../benchmarks/results/2026-09-16-q8-wmma-dense-prefill-layers-gate/README.md),
[`2026-09-16-q8-wmma-layers-recoverable-time`](../benchmarks/results/2026-09-16-q8-wmma-layers-recoverable-time/README.md).

| Scope | Numerical verdict | Recoverable time | Basis |
| --- | --- | ---: | --- |
| layers 0-47 | **fail** — mean KL 1.099e-3 against 1e-3, code only, 387 rows | +7.041 s | measured |
| layers 32-47 | pass, 17x mean headroom | +2.436 s | measured |
| layers 28-47 | pass, `measurement_valid: true`, no blockers | ~+3.6 s | byte-share prediction |
| layers 20-47 | pass, `measurement_valid: true` | ~+4.15 s | byte-share prediction |
| **layers 16-47** | **pass, deepest certified** — mean KL 3.80e-4 (2.6x), p95 1.76e-3, p99 5.42e-3, max 1.53e-2, top-1 1538/1548 = 0.99354, 3/3 deterministic, no scope failures | ~+4.42 s | byte-share prediction |
| layers 12-47, 8-47 | screens inconclusive, not excluded | ~+4.69 / +4.96 s | byte-share prediction |

Only the 0-47 and 32-47 rows are measured time. Every deeper figure is
`7.0409 s x owned byte share`, an interpolation from those two points that the
source README explicitly labels as a prediction. **We have therefore measured
+2.436 s at 32-47, and the "captured" share of the +7.04 s maximum is a
prediction until a sweep arm measures it.** The 16-47 arm confirms a scope, not
a time: this gate records no timing (`timing_protocol: none_full_logits_only_v1`).

16-47 supersedes 20-47 as the deepest certified scope, and its screen
overestimated the full arm by 1.8x, the second deep-scope control point to do so.
12-47 and 8-47 remain screen-inconclusive, and the contiguous upside left below
16-47 is about +0.5 s of *predicted* gain across two full-model arms.

`dense_wide256` is the larger prize and is not in this table because it has no
quality verdict yet. The route now reaches the kernel (see §5 item 2) and, on
the one case where both were traced, it is arithmetic-identical to the route
that is already certified here: on `code-p512` the named default runs
`hipengine_gguf_q8_0_wmma_prefill_f32_f32_out` on 264 roles at layers 16-47 and
the wide arm runs `hipengine_gguf_q8_0_dense_wide256_f32_f32_out` on the same 264
roles, both producing `logits_sha256` `e15dce79…` and `token_id` 248068. The
wide kernel is a retiling of the certified f16 arithmetic rather than a new
arithmetic class, so the certified 16-47 envelope figures above are the number
its own arm has to reproduce — which is a prediction from one case, not a
verdict. The arm has not been run on the 12-case protocol.

## 5. What is left

1. **Re-measure our own PP/TG at HEAD** on the §1.1 protocol. The current column
   is empty because nothing has re-measured it since 2026-09-13.
2. **Gate the `dense_wide` arm at 16-47.** The route has a layer scope
   (`be2d3fa73`) and it now reaches the kernel: the promoted WMMA default owned
   the same window and the wide selector only accepted pre-rewrite parents, so
   it declined before its own gates ran. Fixed on 2026-09-17 (§5 item 2 below);
   `scripts/execution_profile_q8_wmma_prefill_layers_gate.py --route dense_wide
   --layers 16-47` is the run that is owed.
3. **Cache the activation conversion** (convert once per graph, not per launch).
   Measured at 2.573 → 1.303 ms on the packet, against the comparator's 1.339 ms
   kernel. This is the whole remaining dense gap.
4. **Measure a sweep arm** for the certified scopes so §4 stops mixing measured
   and predicted time.
5. **Decide the accuracy policy.** The recalibration proposal is pending user
   decision ([`PRODUCTION-ACCURACY-RECALIBRATION-PROPOSAL-2026-09-16.md`](PRODUCTION-ACCURACY-RECALIBRATION-PROPOSAL-2026-09-16.md));
   it does not change the current envelope, which remains the admission lane.
6. **Close E10** — measure the comparator's own output against F64 with the
   existing replay bridge, replacing the derived BF16 row in §2.6.
7. **The largest blocked item is numerical, not mechanical.** `GR_IU8` and
   `GR_IU8_DOWN` are **+2.494 s measured** — 13.6% of the total gap, more than
   the entire `hyper_connection` family gap — and both are rejected at
   mean KL `1.4e-3` against a `1e-3` limit. A corrected arithmetic for the
   hyper-connection read is the **largest blocked item in the campaign** —
   larger than the unbuilt routed MoE down projection, whose whole family gap is
   1976 ms — and the only place where a numerics result buys seconds directly.
   See §2.4.
8. **Unbuilt mechanisms, in gap order**: the routed MoE down kernel (10.8%,
   unbuilt, and the family where PR #63 moved furthest), MoE reduction (3.6%).
   Elementwise/norm fusion is **not** on this list — we are 0.54x the comparator
   there.
9. **Reach layers 0-7**, which a contiguous scope cannot. Needs a per-layer cost
   profile rather than a boundary search.
9. **Decode versus the comparators is not measured for this model.** The
   September 15 record is prefill only and says so: "Not a decode result.
   Decode is a separate campaign." The only matched decode comparison is the
   §1.1 journey screen against halo-box HIP (15.441 vs our 20.396 tok/s at
   p512) and halo-box Vulkan (27.477).

## 6. Evidence index

| Topic | Artifact |
| --- | --- |
| PP/TG journey baseline, three engines | [`2026-09-13-framework-qwen4exp-strix-journey-baselines.json`](../benchmarks/results/2026-09-13-framework-qwen4exp-strix-journey-baselines.json) |
| Cross-engine prefill rates and kernel totals | [`2026-09-15-flashnext-engine-comparison`](../benchmarks/results/2026-09-15-flashnext-engine-comparison/README.md) |
| Per-role cost, one prefill, 100% attribution | [`2026-09-17-qwen4exp-per-role-cost`](../benchmarks/results/2026-09-17-qwen4exp-per-role-cost/README.md) |
| Same measurement one day earlier, with the reproduction check | [`2026-09-16-flashnext-per-role-cost`](../benchmarks/results/2026-09-16-flashnext-per-role-cost/README.md) |
| Shared-taxonomy family comparison and gap decomposition | [`shared-family-comparison.json`](../benchmarks/results/2026-09-17-qwen4exp-per-role-cost/shared-family-comparison.json) |
| The taxonomy both engines are classified by | [`scripts/qwen4exp_comparator_role_map.py`](../scripts/qwen4exp_comparator_role_map.py) |
| Request-delimited comparator components | [`2026-09-16-flashnext-delimited-components`](../benchmarks/results/2026-09-16-flashnext-delimited-components/component-gap.json) |
| Q8 WMMA dense prefill layer gate | [`2026-09-16-q8-wmma-dense-prefill-layers-gate`](../benchmarks/results/2026-09-16-q8-wmma-dense-prefill-layers-gate/README.md) |
| Wide-row dense Q8 candidate | [`2026-09-16-dense-wide-q8-prefill-candidate`](../benchmarks/results/2026-09-16-dense-wide-q8-prefill-candidate/README.md) |
| Recoverable time by disabled family | [`2026-09-16-disabled-family-recoverable-time`](../benchmarks/results/2026-09-16-disabled-family-recoverable-time/README.md) |
| Comparator arithmetic and ranked external work | [`QWEN4EXP-EXTERNAL-FORKS-REVIEW.md`](QWEN4EXP-EXTERNAL-FORKS-REVIEW.md) |
| Normative numerical envelope | [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) |
| Kernel catalog and dispatch map | [`KERNELS.md`](KERNELS.md) |
| Default-off flags and removal conditions | [`REFACTOR.md`](REFACTOR.md) |
