# Qwen3.8-Flash-Next (Qwen4Exp) Optimization Status

- **Status date:** 2026-09-17
- **Model / quant:** unsloth `Qwen3.8-Flash-Next` `UD-Q4_K_XL`, fingerprint
  `fb1f2fbf73d588c9…`, 111.3 GB, four shards
- **Host:** Framework `gfx1151`, Radeon 8060S / Strix Halo, machine
  `55ea6c509d0b49eea8de7094a1023668`
- **Current default:** the shipped default is the `production` profile when the
  caller names none (a named profile still overrides), chunk1024, BF16 KV, warm
  PLE, and since `15b056111` the certified wide-row Q8 dense prefill route at
  layers 16-47 (`PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS`), which replaced the
  f16 WMMA route bound there since `04f1dde42`; that route's layer env is now
  bound empty for this quant and its selector stays registered for re-gating

**This document is rewritten in place. It is the current-state tracker for this
model; the `QWEN4EXP-*` and `QWEN3.8-FLASH-NEXT-*` campaign documents are dated
records and are not edited to reflect new state.** Every number below carries a
pointer to the artifact that owns it. Where a number is derived rather than
measured, or is an interpolation rather than a measurement, it says so.

The largest demonstrated improvement is the layer-16-47 dense-route change
(§4). The next engineering priority is a correct, operation-complete activation
conversion path, followed by routed MoE and hyper-connection work (§5).
The remaining family ranking is provisional: the detailed trace predates the
shipped route, comparator arithmetic differs, and family boundaries do not
account for fused work identically. No new GPU measurements were taken for this
review.

## 1. Our own progress

Rates are tokens/s. Each block declares its own protocol. The two hipEngine rows
are **not** comparable to each other even though they share a harness and a
fixture: 1.1's row measures a different arithmetic composition, and 1.2 sets out
how that was established.

### 1.1 Journey screen — PP/TG at 512/1024/4096

Protocol: canonical exact-token fixture, 4 categories, one warmup and three
repetitions, 128 AR transitions, token/time-weighted, matched cache/clock/
thermal policy. Source:
[`benchmarks/results/2026-09-13-framework-qwen4exp-strix-journey-baselines.json`](../benchmarks/results/2026-09-13-framework-qwen4exp-strix-journey-baselines.json).

| Arm | p512 PP / TG | p1024 PP / TG | p4096 PP / TG |
| --- | ---: | ---: | ---: |
| hipEngine production, **journey start** (2026-09-13, pre-`49ffa3cb5`: fifteen recovery flags bound on) | 297.1 / 20.40 | 316.9 / 19.71 | 294.1 / 19.17 |
| hipEngine production, HEAD (wide route at 16-47) | not re-measured | not re-measured | not re-measured |

This row is the campaign's starting baseline and it is **not** a rate for the
path we ship. Three things changed under it. The arithmetic: at `00602e556`, the
commit it was measured at, the production binder still bound fifteen
arithmetic-recovery flags **on** for this quant (`Q8_MMQ_PREFILL`, `GR_IU8`,
`GR_IU8_DOWN`, `Q8_IU8_WMM`, `Q4_IU8_PREFILL`, `GDN_COLWARPS_PREFILL`,
`QSA_FLASH_PREFILL`, …), and
[`2026-09-14-q8-prefill-numerics`](../benchmarks/results/2026-09-14-q8-prefill-numerics/README.md)
measured that composition's prefill at max KL `0.0546`, a failed envelope.
`49ffa3cb5` then zeroed all fifteen for `gguf_ud_q4_k_xl`
(`PRODUCTION_ARITHMETIC_RECOVERY_FLAGS`), providing the conservative base for
§1.2 and the later route experiments. The routes: `04f1dde42` promoted the
certified f16 WMMA dense prefill scope at 16-47, measured on `code-p4096` at
19.054 s against the old default's 23.596 s
([worklog](../worklog/entries/20260916T192306.422564Z-lhl-q8-wmma-1647-promotion-b024aa.md)),
and `15b056111` replaced it there with `dense_wide256` (§4). And the attribution
anchor: the September 17 attribution run at `725794c3f` reproduced the production
profile's `logits_sha256` and `token_id` exactly (§1.3), but it predates both
promotions, so its composition is the exact chain rather than HEAD's.

Re-running this screen at HEAD is therefore still the first measurement owed by
this document — a reproduction is not a rate — and it now has one arithmetic
change and two default-path changes to absorb.

### 1.2 Cross-engine prefill — canonical fixture, equal-weight mean

Protocol: prefill only (`n_predict=1`), exact token ids, 12 cases (4 categories
at 512/1024/4096), one warmup, three repetitions, median per case then
equal-weight mean across cases. The hipEngine row is that engine's own canonical
bench instead — the same harness as 1.1, with 128 decode transitions *after* the
measured prefill, which do not enter the prefill wall. Source:
[`benchmarks/results/2026-09-15-flashnext-engine-comparison/README.md`](../benchmarks/results/2026-09-15-flashnext-engine-comparison/README.md).

| Engine | 512 | 1K | 4K |
| --- | ---: | ---: | ---: |
| **hipEngine production** (`ddfc2a746`, pre-promotion) | **185.2** | **190.6** | **183.6** |
| pwilkin `strix-halo` `40a9f4d01`, f16 | 339.2 | 858.9 | **1061.9** |
| halo-box `strix-llama.cpp` `69946438a`, f16 | 451.6 | 620.7 | 660.6 |
| upstream llama.cpp `6011c34ce`, f16 | 321.3 | 415.1 | 459.0 |

The two runs share the recorded collection protocol:
294.1 tok/s at p4096 in 1.1 and 183.6 in 1.2 use the same engine and
harness (`qwen4exp_canonical_ar_bench`), the same recorded protocol (12 cases,
128 decode transitions, chunk1024, warm PLE, one warmup, three measured
repetitions, the same timing-boundary string), the same fixture
(`sha256 42b562bd8e9644be…`) and hash-identical prompts (`4ea99919…` at
`code-p512`). The reported aggregate rates differ by **1.60x**; the earlier
review also records 1723 ms against 2760 ms for `code-p512`. The arithmetic
composition changed (§1.1): 1.1 ran with the fifteen
recovery flags on, and §2.4 measures an overlapping subset at
**7.808 s less wall** on `code-p4096` (23.648 → 15.840 s). This supports a
composition explanation, but it is not a controlled attribution of the entire
historical difference. The aggregation also differs between these summaries:
§1.1 is token/time-weighted and §1.2 averages per-case rates. Recompute a common
statistic before quoting an aggregate regression or recovery. The 1.2 row's own raw run
(`/tmp/comparators-final/hipengine-current.json`) is not committed, so its
per-case walls are no longer recoverable from the artifact — only its medians
are.

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

The dense projection row identified the pre-promotion tuning target. Its 9.3%
ratio is not a realizable speedup estimate: a register-resident FP32 microbenchmark
does not account for this quantized kernel's loads, dequantization, occupancy,
or the different WMMA arithmetic of its replacement.

The capture is the **exact-chain composition**, and that is provable rather than
assumed: its `logits_sha256` `e717076fe080c887…` is the digest the route A/B
records for its `exact` arm on this case. At HEAD the default is the wide route,
and on this case it measures **17.217 s** wall in the route A/B (§4).
That A/B's own exact arm is
22.648 s, giving the measured 5.430 s saving. The 22.699 s profiled capture is
a separate run. Subtracting an A/B wall delta from its kernel-family time would
only be a provisional estimate, not a successor attribution or an error-bounded
measurement. Re-capturing this table on the shipped route is owed (§5).

## 2. Versus the competition

Competitor numbers are **not accuracy-comparable to ours**. The cited records
do not establish a matched qualification under
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) §6.1.

### 2.1 Kernel time, one 4096-token prefill

Us: 22083.6 ms attributed (see 1.3) — and that capture is the **exact-chain
composition** (`logits_sha256` `e717076f…`, the route A/B's `exact` arm). At HEAD
the default is the wide route and this case's wall is **17.217 s**, measured in
the route A/B; the kernel-sum attribution at HEAD is not re-captured, so the
first row below is the pre-promotion composition and is labelled as such.
Competitors, same host, same file:

| Engine | Prompt ms | Kernel sum ms | Source |
| --- | ---: | ---: | --- |
| **hipEngine, exact chain at `725794c3f`** (pre-promotion) | **22699** | **22083.6** | [`2026-09-17-qwen4exp-per-role-cost`](../benchmarks/results/2026-09-17-qwen4exp-per-role-cost/README.md) |
| **hipEngine production at HEAD** (wide route at 16-47), same case | **17217** | not re-attributed | [`2026-09-17-q8-dense-route-ab`](../benchmarks/results/2026-09-17-q8-dense-route-ab/README.md) |
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

#### This capture is the exact composition, not HEAD

Everything in §2.2 and §2.3 comes from the pre-promotion 22083.6 ms kernel
sum. There is no measured kernel-family decomposition for the shipped wide
route. Do not subtract its 17.217 s unprofiled wall from the comparator's
3760.4 ms kernel sum, or use the 5.430 s A/B wall saving to claim a measured
family-gap closure. Earlier estimates of a 13.5 s residual gap and 38% dense
share mixed these bases and are withdrawn.

The taxonomy is useful for locating work, not a like-for-like operation boundary:
`map_mmb()` assigns `mmb_cvt_*` to `elementwise_norm`, while hipEngine's
activation conversion is inside the dense kernel. Routed GLU fusion similarly
moves work across boundaries. Thus the smaller `elementwise_norm` bucket does
not prove that activation preparation or fusion is finished. Before refreshing
the trace, extend and test `map_hipengine()` for `q8_0_dense_wide_kernel` and
WMMA symbols; its current dense rules cover the old coltile/pack8 families only.

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

**In this pre-promotion trace, dense projection is half the gap and four
families are 87% of it.** Absolute costs are more useful than ratios alone for
locating opportunities, subject to the arithmetic and taxonomy caveats above:
`qsa_attention` has the worst ratio (9.01x) but 6.8% of this historical gap.

Two readings the ratio column alone gets wrong:

- **`elementwise_norm` is a smaller standalone bucket**, not proof of lower
  complete-operation cost: 180.2 ms against PR #63's 331.7 ms, 0.54x. Conversion
  and fused work cross the family boundary (§2.2).
- **`expert_down` is not one of our better families.** Against the base it reads
  2.08x, our second-best ratio; against PR #63 it is 5.79x, because PR #63 cut
  that family by 64.1% and we have no equivalent kernel. The base column flatters
  us on exactly the family where the comparator moved furthest.

### 2.4 What candidate timing establishes

These are **separate wall-time experiments**, not a decomposition of the
18323.2 ms historical kernel gap. All concern `code-p4096` on the declared
Framework host/model; sources own the exact commands and configurations.

| Experiment | Baseline → candidate | Saving | What is established |
| --- | ---: | ---: | --- |
| Dense exact → WMMA at 16-47, route A/B | 22.648 → 18.208 s | 4.439 s | Measured in the interleaved A/B; numerical gate passes at this scope |
| Dense exact → wide at 16-47, same A/B | 22.648 → 17.217 s | 5.430 s | Shipped route; 0.991 s incremental saving over WMMA |
| `Q8_IU8_WMM`, disabled-family sweep | 23.648 → 17.034 s | 6.614 s | Historical diagnostic override; no admission established for this arm |
| `Q8_MMQ_PREFILL`, same sweep | 23.648 → 19.662 s | 3.986 s | Historical diagnostic override, not a supported profile state |
| `GR_IU8`, same sweep | 23.648 → 21.800 s | 1.849 s | Independent numerical rejection |
| `GR_IU8_DOWN`, same sweep | 23.648 → 23.004 s | 0.645 s | Independent numerical rejection |
| `GDN_COLWARPS_PREFILL` / `GDN_PEER_PREFILL`, separate arms | 23.648 → 23.420 / 23.522 s | 0.228 / 0.126 s | Inside the reported run-to-run spread; not additive |
| Four flags together: `Q8_IU8_WMM`, `Q8_MMQ_PREFILL`, `GR_IU8`, `GR_IU8_DOWN` | 23.648 → 15.840 s | 7.808 s | One measured, unqualified combination; not an upper bound |

Sources: [route A/B](../benchmarks/results/2026-09-17-q8-dense-route-ab/README.md)
and [disabled-family sweep](../benchmarks/results/2026-09-16-disabled-family-recoverable-time/README.md).
The latter uses two repetitions and `--risk-diagnostics`; the route A/B uses
three interleaved repetitions and a different baseline composition/protocol.
Small differences in baseline wall do not establish uncertainty bounds.

**Do not add individual savings or subtract these configurations from each
other to infer unmeasured combinations.** In particular:

- 2.494 s is **1.849 + 0.645**, not a measured GR-up-plus-down arm. Neither
  the pair nor either flag on top of the shipped wide route has been timed here.
  GR-down is a projection, so the whole sum also cannot be assigned to the
  `hyper_connection` bucket.
- The four-flag combination does not contain the promoted `dense_wide256`
  route. Its 7.808 s saving is not a ceiling containing the wide route's
  5.430 s saving. The former “29.6 points shipped, 13.0 points blocked” claim,
  and the per-family residual/closed columns derived from it, are withdrawn.
- The 0.228 and 0.126 s GDN arms do not establish a 0.354 s combined saving.
- The slower `Q4_IU8_PREFILL` and `PRODUCTION_MOE_PREFILL` overrides do not
  exhaust expert gate/up opportunities. They reject those switches as speed
  candidates on this case, not new tiling, packing, fusion or repair work.

The independent GR rejections are real: `GR_IU8` mean KL `1.4024e-3`, p95
`7.4015e-3`; `GR_IU8_DOWN` mean KL `1.3128e-3`, p95 `6.8352e-3`. Both exceed
mean `1e-3` **and p95 `5e-3`**, so changing only the mean threshold would not
admit them. Earlier P4-up and compensated-down corrections also failed; see
[`REFACTOR.md`](REFACTOR.md). A new mechanism needs boundary-localized evidence,
not another run of the unchanged flags.

The layer-scope timing predictions in §4 apply to the older f16 WMMA route.
They are neither measured wide-route timings nor incremental savings against
the shipped default.

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

A separate earlier unprofiled, single-request pair measured **+13.9% prompt
throughput** (5552.5 → 4873.5 ms) with identical 48/48 output
([worklog](../worklog/entries/20260916T042440.370650Z-lhl-flashnext-halobox-pr63-verified-e29646.md)).
It is not the wall-time delta for the profiled table above, whose prompt walls
are 5569.2 → 3989.3 ms. Keep the two experiments separate.

### 2.6 Accuracy basis

Max absolute error divided by the maximum absolute exact-F64 output on the
identical-operand replay packet
(`layers.8.attn_qkv`, rows 1024, K 2560, M 10240). Source:
[`QWEN4EXP-EXTERNAL-FORKS-REVIEW.md`](QWEN4EXP-EXTERNAL-FORKS-REVIEW.md).

| Arithmetic class | Max-abs error / max-abs F64 output | Evidence scope |
| --- | ---: | --- |
| F32 coltile — the strict dense route and production dense fallback | 1.51e-7 | This packet only; not an everywhere/model-level certificate |
| IU8 WMMA — `iu8_wmma_prefill` | 3.90e-7 | This packet only; route/scope admission is separate |
| **F16 operands — `dense_wide256`, `wmma_prefill`** | **2.08e-4** | Candidate-local model gate passes at 16-47; the sibling WMMA route fails at 0-47 |
| **BF16-both simulated reference** | **6.8e-3 – 7.2e-3** | Derived bound, not measured comparator output |

The simulated BF16-both reference has roughly 33x the F16 kernel's normalized
maximum error on this packet. This is a triangle-inequality bound derived from
our F16 output's distance to that reference, not the comparator kernel's measured
error, elementwise relative error, or end-to-end model quality. E10 should compare
the captured comparator output directly against F64. Neither this packet nor
operand precision establishes which engine has better task quality. BF16 also
has greater dynamic range than F16; a precision-only ranking is not universal.

Production correctness remains the admission criterion, not bit-exactness.
Pre-converting to the **same F16 values** is worth testing before changing operand
format or relaxing the numerical policy; no BF16 speed advantage over F16 is
established by this evidence.

## 3. What makes the comparators fast, and our state

Ordered by share of the **pre-promotion** 18323.2 ms gap (see §2.3), not by
current recoverable time.

**Read the "our state" column with care.** For `gguf_ud_q4_k_xl` the production
binder zeroes fifteen `PRODUCTION_ARITHMETIC_RECOVERY_FLAGS` and then restores
six `PRODUCTION_Q8_QSA_RESTORED_FLAGS`, so this quant's arithmetic comes from
the production **selection table** (keyed on rows and quant) rather than from
those flags. Reading the flag map therefore gives the wrong layer scope for
several routes; the scopes below are read from the kernel trace instead
([`role-analysis.json`](../benchmarks/results/2026-09-17-qwen4exp-per-role-cost/role-analysis.json)).

| Mechanism | Their evidence | Gap share | Our state |
| --- | --- | ---: | --- |
| Dense projection: quantized weights dequantized to BF16 in LDS, activations converted to BF16 **once per graph and cached**, F32 accumulate on BF16 WMMA (`mmb.cu`, PR #63 `08de004`) | PR #63 dense_projection 1536.7 → 1374.3 ms | **50.1%** (pre-promotion) | **Partly built, and the ported half is now the default.** `dense_wide256` ports the tile (`mmb_dense_kernel<128,256,64,64,1>`) at 2.573 ms against the production dispatch's 17.319 ms on the packet (6.74x) and is the production default at layers 16-47 since 2026-09-17: 5.4% below the f16 WMMA route it replaced at 1K/4K, and **5.43 s (23.98%) below the exact chain at `code-p4096`**, both measured in the route A/B. The shipped default was verified to run it on 264 roles at layers 16-47 with no coltile role at or above 16 and a census byte-identical to the explicit production profile ([`2026-09-17-q8-dense-default-path-census`](../benchmarks/results/2026-09-17-q8-dense-default-path-census/README.md)). Two gaps remain: f16 rather than bf16 operands (deliberate), and repeated in-kernel activation conversion. The **1.303 ms** load-path probe used reinterpreted, invalid values, not a correctly pre-converted activation; it excludes conversion cost and is only a hypothesis for this packet, which is at layer 8 outside the promoted scope |
| Routed MoE gate/up WMMA-iu8 selection | PR #63 expert_gate_up −7.0% | 14.5% | **On across the model, but still a major opportunity (3414.2 ms; 4.48x PR #63).** `gguf_q4_k_selected_dual_wmma_iu8_risk_prefill` ran on layers 0-1 and 3-47; the `selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out` selection is a production manifest entry keyed on `prefill_rows_ge64_exact_grouped_q4_gate_up`. This is **not** the `PRODUCTION_MOE_PREFILL` route (that flag is `0` for this quant) and **not** a 27-47 scope, which is what the profile's own comment claims |
| Hyper-connection combine and mix (`hc_combine_norm_f32`, `hc_mix_reduce_f32`) | PR #63 hyper_connection −21.7% | 11.9% | **Open and diagnosed.** `q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32` ran on all 48 layers for 2437.4 ms, plus `gr_write` at 86 ms: 2523.8 ms against their 335.7 ms for the fused combine-plus-norm. The same tensor is traversed twice, 2523.8 ms of reads against 1098 ms of matmuls consuming them. The `GR_IU8` variants are independently numerically **rejected**; their separate wall savings do not establish a combined recovery or a pure hyper-connection-family saving (§2.4) |
| Routed MoE down MMB kernel (`mmb_routed_kernel`) | PR #63 expert_down **−64.1%** (1149.8 → 412.8 ms) | 10.8% | **Not built.** The largest per-family move the comparator made and we have no equivalent kernel. Our `expert_down` is 2389.0 ms, of which 721 ms is iu8 exact-repair. The down projection runs Q5_1 on 43 layers and Q8_0 on five (2, 4, 30, 46, 47) |
| QSA attention | PR #63 −0.6% (156.6 → 155.7 ms) | 6.8% | **On, but not as "QSA flash".** The binder zeroes `QSA_FLASH_PREFILL` for this quant and restores `QSA_H256_WAVE_PREFILL=page256` and `QSA_HEAD_PAIR=quad`; `qsa_sparse_attention_h256_wave_rows_f32` ran on the 12 attention layers (3, 7, …, 47). Our 1402.7 ms is 9.01x theirs, the worst ratio in the table, but only 6.8% of the gap |
| MoE reduction fusion | PR #63 moe_reduce −49.1% | 3.6% | **Open.** Our 762.5 ms covers router logits and select, group scatter/gather, the tile map and the weighted-lane reduction |
| GDN prefill | PR #63 −8.3% | 3.0% | **The base kernel, not the column-warp variant.** `GDN_COLWARPS_PREFILL` is `0` for this quant, so `qwen4_exp_gdn_prefill_f32` ran on the 36 GDN layers. Enabling column warps is worth **+0.228 s measured**, inside the run-to-run spread; our 821.2 ms is 3.11x |
| Elementwise and norm fusion | PR #63 −37.6% (531.4 → 331.7 ms) | **−0.8%** | **Smaller standalone bucket, not an operation-complete advantage.** 180.2 ms against 331.7 ms. The comparator's conversion kernels land here while ours are fused into dense projection; reassess producer/consumer fusion with complete-operation costs |
| Activation packing elimination | PR #63 quantize_pack 284.8 → **0.0 ms** | 0.0% | **No separate packing bucket.** This does not mean activation preparation is free or eliminated: conversion is inside our dense kernel and in the comparator's `elementwise_norm` bucket |
| Dense Q8 prefill tiling | Use the request-delimited halo-box dense rows above; pwilkin's old family breakdown is withdrawn (§2.1) | — | **Promoted 2026-09-17.** The wide-row route (`dense_wide256`) is the production default at layers 16-47, with the exact coltile chain on 0-15 and as the registered strict fallback. It holds the f16 WMMA route's certified envelope byte for byte, it uses 5.4% less prefill wall than that route at 1K/4K, and is 5.43 s below the exact chain at `code-p4096`; see §4 |
| Indexer | PR #63 +168% | 0.0% | **Low absolute priority.** The mapper moves some old hipCUB work into `elementwise_norm`, so +168% is not a clean operation-level regression |

## 4. Certified and promoted

**Layers 16-47 is the promoted scope.** The wide-row route (`dense_wide256`) is
the production default there since 2026-09-17: it holds the certified f16 WMMA
route's envelope byte for byte (see below) and uses 5.4% less prefill wall at
1K/4K, so it
replaced that route rather than sitting beside it. Layers 0-15 stay on the exact
coltile chain, which is also the registered strict fallback for the whole scope.
The end-to-end check after promotion: the default runs
`hipengine_gguf_q8_0_dense_wide256_f32_f32_out` on 264 roles at layers 16-47 and
`..._gemv_coltile8_rowbatch4_wave_scale_f32_f32_out` on the other 134, with zero
`wmma_prefill` launches and the same `logits_sha256` (`e15dce79…`) and
`token_id` 248068 as before the promotion. That check was first run with the
profile named explicitly; it has since been repeated through the shipped default
— no profile named, no env overrides — where the resolved profile is
`production`, `fell_back_to_strict` is false, and the launch census is
**byte-identical** to the explicit run (`sha256 2b5dbcaf32b2e72b…`,
[`2026-09-17-q8-dense-default-path-census`](../benchmarks/results/2026-09-17-q8-dense-default-path-census/README.md)).

The table includes the promoted scope and unpromoted passing, failing, or
inconclusive scopes. “Pass” below means the **candidate-local numerical gate**,
not a complete production-profile certificate. Its harness constructs the
production profile and clears the dense selectors for the teacher; the field
`strict_logits_sha256` does not make that teacher the named strict profile.
The gate covers 12 canonical cases and repeat determinism, but supplies no
separate category-heldout/task-quality, BF16-relative, or serving-isolation
certificate. The gate README explicitly disclaims task-quality certification.
Link applicable full-stack evidence before claiming all requirements of
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) are closed; runtime promotion
and proof of complete qualification are separate facts.

Numerical verdicts use the thresholds in
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) §6.1. Sources:
[`2026-09-16-q8-wmma-dense-prefill-layers-gate`](../benchmarks/results/2026-09-16-q8-wmma-dense-prefill-layers-gate/README.md),
[`2026-09-16-q8-wmma-layers-recoverable-time`](../benchmarks/results/2026-09-16-q8-wmma-layers-recoverable-time/README.md).

| Scope | Numerical verdict | Recoverable time | Basis |
| --- | --- | ---: | --- |
| layers 0-47 | **fail** — mean KL 1.099e-3 against 1e-3, code only, 387 rows | +7.041 s | measured |
| layers 32-47 | pass, 17x mean headroom | +2.436 s | measured |
| layers 28-47 | pass, `measurement_valid: true`, no blockers | ~+3.6 s | byte-share prediction |
| layers 20-47 | pass, `measurement_valid: true` | ~+4.15 s | byte-share prediction |
| **layers 16-47** | **pass, deepest certified** — mean KL 3.80e-4 (2.6x), p95 1.76e-3, p99 5.42e-3, max 1.53e-2, top-1 1538/1548 = 0.99354, 3/3 deterministic, no scope failures | +4.44 s WMMA / +5.43 s wide | measured (route A/B, `code-p4096`); **promoted** |
| layers 12-47, 8-47 | screens inconclusive, not excluded | ~+4.69 / +4.96 s | byte-share prediction |

Only the 0-47 and 32-47 rows are measured by the sweep; 16-47 is measured by the
2026-09-17 route A/B
([`2026-09-17-q8-dense-route-ab`](../benchmarks/results/2026-09-17-q8-dense-route-ab/README.md),
`code-p4096`: 22.648 -> 18.208 s on the WMMA arm and -> **17.217 s** on the wide
arm that replaced it). The WMMA arm's measurement lands within 0.4% of the
+4.42 s byte-share prediction for the same scope, but the two protocols' exact
arms differ by 5% (23.840 s with the sweep's `--risk-diagnostics` instrument
against 22.648 s interleaved), so it confirms the interpolation's shape rather
than reproducing its number. Every figure deeper than 16-47 is still
`7.0409 s x owned byte share`, an interpolation from two measured points that
the source README explicitly labels as a prediction. **We have therefore measured
+2.436 s at 32-47 and +4.439 s at 16-47; the "captured" share of the +7.04 s
maximum is still a prediction below 16-47.** The 16-47 quality gate itself
records no timing (`timing_protocol: none_full_logits_only_v1`).

16-47 supersedes 20-47 as the deepest certified scope, and its screen
overestimated the full arm by 1.8x, the second deep-scope control point to do so.
12-47 and 8-47 remain screen-inconclusive, and the contiguous upside left below
16-47 is about +0.5 s of *predicted* gain across two full-model arms.

`dense_wide256` is the larger prize and **passed its own gate at 16-47 on
2026-09-17**
([`2026-09-17-q8-dense-wide-16-47-gate`](../benchmarks/results/2026-09-17-q8-dense-wide-16-47-gate/README.md)):
12 cases, 1548 rows, mean KL 3.80e-4, p95 1.76e-3, p99 5.42e-3, max 1.53e-2,
top-1 1538/1548 = 0.99354, `measurement_valid: true`, no blockers, no scope
failures, three identical trajectory hashes, all ten top-1 misses inside the
flip-eligible set. The envelope is **bit-identical to the certified WMMA row
above**: `strict_logits_sha256` `550bb9b832bc7ba2…` and `candidate_logits_sha256`
`65370710c0b9c40f…` match that gate's artifact byte for byte, and every numerical
field of the quality summary is identical. Route, provenance, manifests,
timestamps and other metadata differ; the artifacts as a whole are not
byte-identical. The wide kernel's measured trajectories match the certified
f16 route on these cases, supporting the local route substitution, not
unmeasured scopes or full-stack task equivalence.

The route's performance half is measured too. On the `code` category, three
interleaved arms in one process, one model load, selectors flipped after the
production binder, three repetitions each with the arm order rotated, and a
second process reproducing every ratio within 0.1 percentage points
([`2026-09-17-q8-dense-route-ab`](../benchmarks/results/2026-09-17-q8-dense-route-ab/README.md)):

| Route at layers 16-47 | 512 | 1K | 4K |
| --- | ---: | ---: | ---: |
| exact coltile chain | 2.822 s | 5.487 s | 22.648 s |
| WMMA (the default this route replaced) | 2.225 s | 4.389 s | 18.208 s |
| `dense_wide256` (default at 16-47 since 2026-09-17) | 2.201 s | 4.151 s | 17.217 s |

The wide route is 5.4% below the WMMA default at 1K and 4K and indistinguishable
from it at 512, and 5.43 s below the exact chain at 4K against the default's
4.44 s. The wide arm's logits digest and sampled token match the certified WMMA arm on all
three cases in both runs. The **7191 launches per arm are census-covered linear
launches aggregated across the benchmark**, not all GPU kernels for one prefill.
The numerical and timing records support this scoped route swap; the production
binder now binds
`HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE=1` with `..._DENSE_WIDE_LAYERS=16..47` for
`gguf_ud_q4_k_xl` and binds `HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` empty. Its
selector and env var remain registered for explicit opt-in and re-gating.

## 5. What is left

### Measurement and qualification prerequisites

1. **Refresh the shipped default's baseline and attribution.** Run the canonical
   four-category PP/TG protocol with raw per-case walls, then a cache-only
   role-marked capture on `code-p4096`. The family mapper now covers the wide
   symbols (§2.2): `DENSE_PROJECTION_STEMS` in
   `scripts/qwen4exp_comparator_role_map.py` names every dense Q8_0 prefill
   route, pinned from the kernel sources by
   `tests/test_unit_qwen4exp_role_map_dense_routes.py`. Before that the promoted
   route's kernel symbol was `other`, which the comparison's `--strict`
   `--unmapped-floor-ms 1` gate turns into a failed run rather than a
   mis-attributed table. Verify selector/manifest identity and launch coverage,
   and report unprofiled wall separately from profiled kernel time. The current
   dense-family cost is unknown; the old 50.1% gap share is not today's ranking.
   Use one aggregation rule for historical comparisons. Do not turn this review
   into a new topline rate: the closest route timing covers `code` only.
2. **Close the qualification accounting.** Keep the passing local gate and
   promotion facts, but identify the applicable full-stack strict-teacher,
   heldout/task, BF16-relative and ownership/isolation evidence (§4). Missing
   evidence is not evidence of failure, nor is a local pass a complete
   certificate. The separate
   [recalibration proposal](PRODUCTION-ACCURACY-RECALIBRATION-PROPOSAL-2026-09-16.md)
   does not change the current admission policy.

### Engineering priorities

This is a recommended experiment order, not a prediction of additive gains.
The costs below are the **old** `code-p4096` trace on the model/host declared
above; comparator differences are opportunity indicators, not achievable
speedups or quality-equivalent targets.

| Priority | Opportunity | Evidence and limit | First bounded experiment |
| --- | --- | --- | --- |
| 1 | **Materialize correct F16 activations for the wide route** | Strong load-path hypothesis; 2.54–2.57 ms versus a 1.303 ms invalid-value probe on one layer-8 packet. No working pre-conversion speedup or current-family total is measured | Add a real conversion plus F16-input variant; compare conversion + GEMM + scratch/launch overhead on the admitted layers' actual shapes. Reuse only across consumers of the same live activation; invalidate on producer writes and request/graph reuse. Preserve the existing rounding semantics and strict fallback |
| 2 | **Routed MoE gate/up and down, including repair** | Gate/up 3.414 s, down 2.389 s; together **5.803 s**, with 1.434 s of repair already included. Historical comparator gap totals 4.629 s. Gate/up being enabled does not make it optimized | Profile useful rows per expert, tile padding, map/scatter cost and repair cost. Test expert-row-aware tiling and a down-projection tiled WMMA implementation on Q5_1/Q8_0. Charge the complete operation; <1% repair incidence does not mean negligible repair time |
| 3 | **Hyper-connection read/combine and its projections** | 2.524 s old family cost; GR-up alone saved 1.849 s and GR-down alone 0.645 s in a different diagnostic composition. Both fail mean and p95 KL; the pair is unmeasured | Localize drift at projection/sigmoid/mean/publication boundaries. Test new tiling or producer/consumer fusion as well as numerically corrected arithmetic; do not rerun the rejected P4/compensated variants unchanged |
| 4 | **QSA prefill** | 1.403 s old cost versus 0.156 s comparator bucket. Smaller than MoE but not negligible, and much larger than PLE/indexer | Match selected-position work and complete-operation boundaries, then test D=256 tiling/occupancy on the current wave-row path with exact ownership and the applicable numerical gate |
| 5 | **MoE map/reduction fusion** | 0.763 s old bucket, shared with the MoE work above | Attribute router/map/scatter/reduction separately and fuse exposed producer/consumer boundaries; do not add this bucket again to a complete-MoE result |
| 6 | **Dense scope extension** | Layers 12-47 and 8-47 remain inconclusive. The roughly 0.5 s extension estimate belongs to old WMMA byte-share predictions, not today's wide route | Obtain current per-layer costs, screen useful scopes, then fully gate the chosen scope and time it against the shipped route. Neither a failed 0-47 arm nor a boundary search proves layers 0-7 individually impossible |

Do not prioritize unchanged GDN switches on the 0.126–0.228 s noisy screens,
standalone elementwise ratios, or warm-PLE/indexer work ahead of these measured
large owners. Larger chunks and long-context scaling remain separate hypotheses:
first establish current context-dependent costs and memory/admission limits,
rather than transferring short-context rankings to 16K–256K.

### Separate follow-ups

- **Comparator arithmetic (E10):** compare its captured output directly with
  F64. This resolves the simulated-BF16 bound, not end-to-end task quality.
- **Decode:** historical matched results exist in §1.1 (p512: hipEngine 20.396,
  halo-box HIP 15.441, Vulkan 27.477 tok/s), but the hipEngine composition is
  obsolete. Current-default matched decode is missing. Measure and attribute
  it separately; prefill gains do not establish decode or MTP gains.
- **Cleanup and other models:** retire obsolete selectors only when their
  rollback/re-gating role is finished (`REFACTOR.md`). Other quants and models,
  including Qwen3.6-35B-A3B, need their own scope and quality evidence before
  default promotion. Do not repeat timing of every old certified scope merely
  to replace historical predictions; measure the next decision against today's
  default.

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
| Wide-row dense Q8 numerical gate at 16-47 | [`2026-09-17-q8-dense-wide-16-47-gate`](../benchmarks/results/2026-09-17-q8-dense-wide-16-47-gate/README.md) |
| Route A/B: exact vs WMMA vs wide at 16-47 | [`2026-09-17-q8-dense-route-ab`](../benchmarks/results/2026-09-17-q8-dense-route-ab/README.md) |
| Shipped-default launch census for the wide route | [`2026-09-17-q8-dense-default-path-census`](../benchmarks/results/2026-09-17-q8-dense-default-path-census/README.md) |
| The composition change that invalidates the journey-start rate | [`2026-09-14-q8-prefill-numerics`](../benchmarks/results/2026-09-14-q8-prefill-numerics/README.md) |
| Recoverable time by disabled family | [`2026-09-16-disabled-family-recoverable-time`](../benchmarks/results/2026-09-16-disabled-family-recoverable-time/README.md) |
| Comparator arithmetic and ranked external work | [`QWEN4EXP-EXTERNAL-FORKS-REVIEW.md`](QWEN4EXP-EXTERNAL-FORKS-REVIEW.md) |
| Normative numerical envelope | [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) |
| Kernel catalog and dispatch map | [`KERNELS.md`](KERNELS.md) |
| Default-off flags and removal conditions | [`REFACTOR.md`](REFACTOR.md) |
