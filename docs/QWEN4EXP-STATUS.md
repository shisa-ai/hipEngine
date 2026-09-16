# Qwen3.8-Flash-Next (Qwen4Exp) Optimization Status

- **Status date:** 2026-09-17
- **Model / quant:** unsloth `Qwen3.8-Flash-Next` `UD-Q4_K_XL`, fingerprint
  `fb1f2fbf73d588c9…`, 111.3 GB, four shards
- **Host:** Framework `gfx1151`, Radeon 8060S / Strix Halo, machine
  `55ea6c509d0b49eea8de7094a1023668`
- **Current default:** named production profile, chunk1024, BF16 KV, warm PLE

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
September 16-17 work certified routes and fixed dispatch and provenance; it did
not change the default path, so there is no later same-protocol row to report.
Re-running this screen at HEAD is the first measurement owed by this document.

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
process, request-delimited, one `code-p4096` prefill. Source:
[`benchmarks/results/2026-09-16-flashnext-per-role-cost/README.md`](../benchmarks/results/2026-09-16-flashnext-per-role-cost/README.md).

| Measure | Value |
| --- | ---: |
| Prefill wall | 22046 ms |
| Attributed kernel time | 22368.1 ms over a 22932.5 ms window, 9652 kernels, **0 ms unattributed** |
| Matmul / non-matmul / risk-or-repair | 14368 / 6308 / 1370 ms |
| Dense projections (`linear:` matmul) | 10070 ms through **one** kernel family, `gguf_k_prefill_out_coltile_rowbatch_kernel` |
| Dense projections, achieved rate | 2650 GFLOP/s = **9.3%** of the 28521 GFLOP/s register-resident FP32 measurement |

The dense projection row is the whole tuning target: 46% of the prefill runs
through one kernel family at under a tenth of the machine's measured FMA rate.

## 2. Versus the competition

Competitor numbers are **not accuracy-comparable to ours** and the table says so
per row. Their arithmetic is not gated by anything equivalent to
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) §6.1.

### 2.1 Kernel time, one 4096-token prefill, request-delimited

Us: 22368.1 ms attributed (see 1.3). Competitors, same host, same file:

| Engine | Prompt ms | Kernel sum ms | Source |
| --- | ---: | ---: | --- |
| **hipEngine production** | **22046** | **22368.1** | [`2026-09-16-flashnext-per-role-cost`](../benchmarks/results/2026-09-16-flashnext-per-role-cost/README.md) |
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

### 2.2 Where the comparator's time went, and what PR #63 changed

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

### 2.3 Accuracy basis

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

| Mechanism | Their evidence | Our state |
| --- | --- | --- |
| Quantized weights dequantized to BF16 in LDS; activations converted to BF16 **once per graph and cached**; F32 accumulate on BF16 WMMA (`mmb.cu`, PR #63 `08de004`) | PR #63 dense_projection 1536.7 → 1374.3 ms | **Partly built.** `dense_wide256` ports the tile (`mmb_dense_kernel<128,256,64,64,1>`) at 2.573 ms vs the production dispatch's 17.319 ms on the packet (6.74x). Two gaps: f16 rather than bf16 operands (deliberate), and per-launch activation conversion — with a pre-converted f16 activation the same kernel runs **1.303 ms** against the comparator's 1.339 ms, so the activation path is the remaining 1.93x |
| Routed MoE MMB kernels (`mmb_routed_kernel`, `mmb_routed_glu_kernel`) | PR #63 expert_down −64.1%, expert_gate_up −7.0% | **Default-on at a certified scope.** WMMA-MoE layers 27-47, with iu8 gate/up 35-47 — both in the named production profile |
| Activation packing elimination | PR #63 quantize_pack 284.8 → **0.0 ms** | **Done.** Our per-role capture shows no packing row; the September 15 bucket table also read 0.0 ms for us |
| Elementwise/norm fusion | PR #63 −37.6% | **Open.** Our non-matmul class is 6308 ms (28.6%) |
| MoE reduction fusion | PR #63 moe_reduce −49.1% | **Open.** Part of our 6599 ms `moe` bucket; routing/scatter/reduction is 931 ms of it |
| Hyper-connection cost | PR #63 −21.7% | **Open and diagnosed.** Our `gr_read` is 2539 ms (11.5%), of which the two `hc_*_down` reads are 2539 ms against 1098 ms for the matmuls that consume them — the same tensor is read twice |
| GDN prefill | PR #63 −8.3% | **Default-on at a certified scope.** Column-warp GDN layers 27-47 (supersedes peer-GDN) |
| QSA attention | PR #63 −0.6% | **Default-on at a certified scope.** QSA flash layers 35-47 |
| Dense Q8 prefill tiling | pwilkin dense variants 1387.5 ms per prefill against our 10070 ms | **Certified, not promoted.** F16 WMMA dense Q8 at layers 20-47 passes every calibrated gate; see §4 |
| Indexer | PR #63 +168% | **Not a target** — their regression, 15.0 ms absolute |

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

`dense_wide256` is the larger prize and is not in this table because it cannot
be gated yet: the route committed on 2026-09-17 is a plain boolean with no layer
scope, so enabling it covers all 48 layers — the 0-47 scope the sibling f16
route already fails. It needs a layer filter before an arm is worth running.

## 5. What is left

1. **Re-measure our own PP/TG at HEAD** on the §1.1 protocol. The current column
   is empty because nothing has re-measured it since 2026-09-13.
2. **Give the `dense_wide256` selector a layer scope**, then gate it at 20-47.
   Without this the route cannot express the scope its own removal condition
   requires.
3. **Cache the activation conversion** (convert once per graph, not per launch).
   Measured at 2.573 → 1.303 ms on the packet, against the comparator's 1.339 ms
   kernel. This is the whole remaining dense gap.
4. **Measure a sweep arm** for the certified scopes so §4 stops mixing measured
   and predicted time.
5. **Decide the accuracy policy.** The recalibration proposal is pending user
   decision ([`PRODUCTION-ACCURACY-RECALIBRATION-PROPOSAL-2026-09-16.md`](PRODUCTION-ACCURACY-RECALIBRATION-PROPOSAL-2026-09-16.md));
   it does not change the current envelope, which remains the admission lane.
6. **Close E10** — measure the comparator's own output against F64 with the
   existing replay bridge, replacing the derived BF16 row in §2.3.
7. **Open mechanisms from §3**: elementwise/norm fusion, MoE reduction, and the
   duplicated hyper-connection read (2539 ms against 1098 ms of consuming
   matmuls).
8. **Reach layers 0-7**, which a contiguous scope cannot. Needs a per-layer cost
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
| Per-role cost, one prefill, 100% attribution | [`2026-09-16-flashnext-per-role-cost`](../benchmarks/results/2026-09-16-flashnext-per-role-cost/README.md) |
| Request-delimited comparator components | [`2026-09-16-flashnext-delimited-components`](../benchmarks/results/2026-09-16-flashnext-delimited-components/component-gap.json) |
| Q8 WMMA dense prefill layer gate | [`2026-09-16-q8-wmma-dense-prefill-layers-gate`](../benchmarks/results/2026-09-16-q8-wmma-dense-prefill-layers-gate/README.md) |
| Wide-row dense Q8 candidate | [`2026-09-16-dense-wide-q8-prefill-candidate`](../benchmarks/results/2026-09-16-dense-wide-q8-prefill-candidate/README.md) |
| Recoverable time by disabled family | [`2026-09-16-disabled-family-recoverable-time`](../benchmarks/results/2026-09-16-disabled-family-recoverable-time/README.md) |
| Comparator arithmetic and ranked external work | [`QWEN4EXP-EXTERNAL-FORKS-REVIEW.md`](QWEN4EXP-EXTERNAL-FORKS-REVIEW.md) |
| Normative numerical envelope | [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) |
| Kernel catalog and dispatch map | [`KERNELS.md`](KERNELS.md) |
| Default-off flags and removal conditions | [`REFACTOR.md`](REFACTOR.md) |
