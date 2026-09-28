# Qwen3.8-Flash-Next (Qwen4Exp) optimization ledger

- **Ledger date:** 2026-09-29 (rewritten in place; this revision reorganizes the
  ledger by phase and functional block, adds the competitor mechanism review
  and a complete tried-and-not-retained registry)
- **Host:** Framework `gfx1151`, Radeon 8060S / Strix Halo, machine
  `55ea6c509d0b49eea8de7094a1023668`
- **Model / quant:** unsloth `Qwen3.8-Flash-Next` `UD-Q4_K_XL`, BF16 KV, warm
  PLE, chunk1024, shipped `production` profile

**This is the single open-work list for closing the prefill, decode and MTP gap
on this model, and the single registry of what has already been tried.** Every
open row names a measured cost, the evidence that bounds the opportunity, the
first bounded experiment, and the gate that decides it. Rows leave the open
list only by promotion, by recorded rejection, or by a stated blocker, and every
rejection lands in §7 with the condition that would reopen it.

## Contents

- §0 How this document relates to the others
- §1 Where we stand against every engine
- §2 Why the gap exists: arithmetic policy first, kernel technology second
- §3 Cross-cutting decisions and measurement prerequisites
- §4 Prefill open work, by functional block
- §5 Decode open work, by functional block
- §6 MTP open work
- §7 Tried and not retained, by functional block (the registry)
- §8 Measurement rules for this model
- §9 Cadence

## 0. How this document relates to the others

| Document | Role |
| --- | --- |
| [`QWEN4EXP-STATUS.md`](QWEN4EXP-STATUS.md) | **State.** Rates, the competition as measured, the mechanism ledger, certified and promoted scopes, qualification gaps. Not an experiment list. |
| **This ledger** | **Open work and the tried registry.** Ranked open rows per phase and block, plus every tried-and-not-retained mechanism with its reopen condition. |
| [`QWEN4EXP-JOURNEY-REMAINING.md`](QWEN4EXP-JOURNEY-REMAINING.md) | Frozen dated inventory of the September 15 journey (`R01`-`R30`). Rows still live here carry their `R`-ID. |
| [`QWEN4EXP-EXTERNAL-FORKS-REVIEW.md`](QWEN4EXP-EXTERNAL-FORKS-REVIEW.md) | Dated September 8-9 source review (Halogen, Myhacsint, halo-box, nasone32, mlx-serve) and the protocol-confound checklist C1-C6. Its `E0`-`E10` experiments are folded into §3-§6 here. |
| [`QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md`](QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md), [`QWEN3.8-FLASH-NEXT.md`](QWEN3.8-FLASH-NEXT.md), [`QWEN4EXP-STRIX-JOURNEY-CAMPAIGN.md`](QWEN4EXP-STRIX-JOURNEY-CAMPAIGN.md), [`QWEN4EXP-WILKIN-RUNTIME-REVIEW.md`](QWEN4EXP-WILKIN-RUNTIME-REVIEW.md), [`QWEN4EXP-CONTEXT-CAPACITY.md`](QWEN4EXP-CONTEXT-CAPACITY.md), [`QWEN4EXP-TOOLS-GRAMMAR.md`](QWEN4EXP-TOOLS-GRAMMAR.md) | Dated records. Evidence, not work order. Their numbers are on superseded compositions unless re-quoted here. |
| [`REFACTOR.md`](REFACTOR.md) | Every retained default-off flag and its removal trigger. |

Worklog entries are cited by their timestamp prefix (for example
`20260927T165141`); `ls worklog/entries/<prefix>*` finds the file.

## 1. Where we stand against every engine

### 1.1 Rates

Two kinds of rows, never to be mixed into one ratio: **measured here** (this
host, our canonical exact-token fixture, the protocol in `QWEN4EXP-STATUS.md`
§1.2) and **author-reported** (another host, the author's own protocol).
Author rows are opportunity indicators only; the protocol confounds C1-C6 in
the external-forks review apply to each of them.

| Engine and build | Prefill tok/s | AR decode tok/s | Basis |
| --- | --- | --- | --- |
| **hipEngine** default at `42ce71156` | 239.5 / 253.4 / 241.9 at p512 / 1K / 4K | 17.70 / 17.18 / 16.24 | Measured here, canonical fixture ([source](../benchmarks/results/2026-09-17-qwen4exp-shipped-default-baseline/README.md)) |
| **hipEngine** default after the F16 activation staging and the full-attention context promotions | p4096 category mean **255.066** | not re-measured | Measured here (`20260928T155352`); p512/1K not refreshed |
| pwilkin `strix-halo` `40a9f4d01`, f16 KV | 339.2 / 858.9 / **1061.9** | — | Measured here, same fixture ([source](../benchmarks/results/2026-09-15-flashnext-engine-comparison/README.md)) |
| halo-box `69946438a`, f16 KV | 451.6 / 620.7 / 660.6 | — | Measured here, same fixture |
| upstream llama.cpp `6011c34ce`, f16 KV | 321.3 / 415.1 / 459.0 | — | Measured here, same fixture |
| halo-box PR #63 `c4aa30229` | kernel sum 3760.4 ms for one `code-p4096` prefill (≈1089 tok/s) | — | Measured here, request-delimited ([source](../benchmarks/results/2026-09-16-flashnext-delimited-components/component-gap.json)) |
| Vulkan (historical composition) | — | 27.477 at p512 | Measured here in September; the hipEngine composition it was paired with is obsolete |
| **gufo** (dedicated Strix Halo engine), Sept 22-27 builds | **1628.52** at pp2048 d0; 1523.39 at d4096; 1292.02 at d131072 | **25.87** d0; 25.84 at d4096; 22.89 at d131072 | Author-reported: same UD-Q4_K_XL file, HTTP, greedy, one warmed sample per point ([BENCHMARKS.md](https://github.com/gufo-org/gufo/blob/fee9d2a/docs/models/qwen3.8-flash-next/BENCHMARKS.md)) |
| llama.cpp `b11069` as measured by gufo | 489.59 at pp2048 | 22.20 | Author-reported (gufo's table) |
| halo-box master `52d7e100b` | 1312.51 at pp4096 (UD-Q4_K_XL) | — | Author-reported: `llama-bench -ub 4096`, f16 KV, synthetic tokens ([PR #91](https://github.com/halo-box/strix-llama.cpp/pull/91)) |
| halo-box PR #91 head `e652a3c59` (open) | 1098.6 / 1339.6 / 1636.6 at pp512 / 1K / 4K | — | Author-reported, same protocol |
| halo-box PR #87 (merged 2026-09-25) | pp2048 938 → 1025 (hot) | 18.7 → 22.1 | Author-reported, the "gufo comparison, day 1" PR |

MTP rows, author-reported by gufo at d0: mixed text **32.20**, repetitive text
**59.30** tok/s (llama.cpp `6fcaa16f`: 31.73 / 48.12). Our MTP is explicit,
default-off, capped at 1K context, and the retained serial provider measured
**0.7407x** true AR (§6).

**What the table says.** The fastest prefill measured on this host is
pwilkin's at 1061.9 tok/s, 4.2x ours at 4K. The fastest reported anywhere on
this model file is gufo and halo-box PR #91 at ~1630 tok/s, ~6.4x. Decode is a
smaller gap: 25.9 against ~17 tok/s, 1.5x. The target this ledger works toward
is gufo's, because gufo is the only competitor that, like us, is a
purpose-built engine on this exact payload with its own quality checks.

### 1.2 Kernel time per family, per 1K prompt tokens

Ours is the `42ce71156` attribution (§1 of STATUS) divided by four; gufo's is
its own 2026-09-21 d0 pp2048 profile (1362.8 ms kernel time in a 1380 ms span,
[EXPERIMENTS.md](https://github.com/gufo-org/gufo/blob/fee9d2a/docs/models/qwen3.8-flash-next/EXPERIMENTS.md))
divided by two. **Approximate by construction:** the prompt lengths differ
(4096 against 2048, so our attention includes sparse rows theirs does not),
the taxonomies differ, and gufo's figure is author-reported.

| Family | Ours ms / 1K | gufo ms / 1K | Ratio | Notes |
| --- | ---: | ---: | ---: | --- |
| Routed experts (gate/up + down) | 1401.7 | ~238 (35%) | ~5.9x | Ours post-unroll 3377.1 + 2229.8 ms, including 1240 ms exact repair |
| Dense projections | 1308.7 | ~198 (29%) | ~6.6x | Gufo runs its dense F16 route at 30-34 TFLOPS against a measured 59 TFLOPS F16 matrix ceiling |
| Hyper-connections | 631.4 | ~72 combine (10.5%) + part of ~48 (7%, HC down/inject/shared) | ~5-9x | |
| GDN | 205.4 | ~65 (9.5%) | ~3.2x | |
| Attention and indexer | 350.6 | ~37 (5.5%) | ~9x | Ours predates the 1.912x full-attention promotion; theirs has no sparse rows at 2048 |
| MoE router/reduce | 189.6 | not separated | — | |
| **Total** | **4168** (post-unroll 16673.2 / 4) | **681** | **~6.1x** | |

Every family is 3-9x behind and none is close. That breadth is the finding:
the gap is not one missing kernel.

Our rows mix the pre-unroll anchor (dense, HC, GDN, attention, reduce) with
the post-unroll MoE capture; `elementwise_norm` and the remainder are omitted,
so the rows do not sum exactly to the post-unroll total.

## 2. Why the gap exists: arithmetic policy first, kernel technology second

### 2.1 What the fast engines compute

Read from source at gufo `fee9d2a`
(`src/models/qwen38_flash_next/kernels/rocm/executor.cpp`, `kernels.hip.cpp`),
halo-box PR #91 / #87 / #104 descriptions and pwilkin `40a9f4d01`:

| Block | gufo (prefill) | halo-box / pwilkin | hipEngine production default |
| --- | --- | --- | --- |
| Dense Q8_0 projections | `DenseF16GEMMKernel`: Q8_0 codes decoded to F16 in LDS (exact magic-number construction), **F16 activation rows**, F16 WMMA, F32 accumulate, at **every layer** for shapes with ≥2048 outputs and K ≤ 2560; the other Q8_0 shapes run `W8A8BlockedWmmaGEMMKernel` (int8 WMMA over activations quantized per 32-block) | `mmb.cu`: BF16 weights and BF16 activations, BF16 WMMA, all layers | F16 operands (`dense_wide256`) at layers 16-47 only; **exact F32 coltile chain at 0-15** |
| Routed experts | `RoutedF16GEMMKernel`: Q4_K / Q5_1 / Q8_0 decoded to F16 in LDS, F16 activations, F16 WMMA; gate/up paired with a SwiGLU epilogue; 48/64/128-row expert tiles picked from the real bucket counts; **no repair pass** | MMB routed GLU (BF16), Q4_K two-lane decode | iu8 WMMA with at-risk flags plus a **sparse exact repair** (1240 ms per `code-p4096`) |
| Hyper-connections | `HcDownF16Gemm` (int8 WMMA from a tiled-Q8 copy of the norm) plus `HcMixF16Gemm` (Q8 → F16 up projection with the mix and inject in the epilogue); combine fused with the MoE epilogue and the next norm | fused gate GEMM + mix, bf16 streams end to end, combine+norm computes the next inject | exact F32 `q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4` (2507 ms, 256x weight re-read) |
| Router and small F32 weights | hipBLAS SGEMM (router and shared-expert gate in one GEMM) | activations rounded to F16 on WMMA ("every F32 weight of these GGUFs is F16-exact") | exact F32 tile4 producer |
| Attention | `WmmaCausalAttentionKernel`: **F16 K/V cache**, WMMA QK and PV, four queries x twelve heads packed over the union of their selections | HS256 WMMA flash attention; WMMA indexer scoring | BF16 KV; scalar exact FMA sparse attention (`__fmul_rn` with a contraction barrier) |
| GDN | F32 row-split recurrence, four tokens staged in LDS, three windows of loads in flight | chunked fp16 WMMA chunk scan (NMSE ~2e-7) with fused conv/SiLU, q/k norm and gated RMS norm | exact F32 register-state / tile-16 recurrence |
| Activation conversion | producers write the F16 (and tiled Q8) copy their consumer reads; no separate conversion pass on the wide route | producers write the BF16 copy themselves | one bounded staging cast per consumer set (`HIPENGINE_GGUF_Q8_DENSE_WIDE_F16_ACT`) |
| Prefill chunk | 2048 | `-ub 4096` in the published benchmarks | 1024 |

Their quality evidence is not our gate. gufo's is consistency (batched against
isolated, chunk boundaries, snapshot replay, 63 tg128 requests matching fresh
AR) plus an FP64 attention-operator check; its own scalar-versus-bulk prefill
comparison reports "same top-1; logit RMSE 0.18, not bit-identical". halo-box
uses PPL and KLD against its own reference build. Neither is measured against a
strict F32 teacher at our calibrated envelope.

### 2.2 What that means for us

1. **Our arithmetic choices already cost roughly half the prefill.** The
   September 14 recovery (`20260914T090826`, `49ffa3cb5`) zeroed fifteen
   unqualified arithmetic selectors after the old composition measured max KL
   `0.0546`. Same host, same residency, counterbalanced: PP
   **292.7 / 309.3 / 285.8 → 172.9 / 179.7 / 130.0** (−41 / −42 / −55%) at
   p512 / 1K / 4K and TG **20.2 / 19.4 / 18.1 → 17.6 / 17.0 / 11.4**. Q8 and QSA
   were then partly restored under qualified variants, and the September 16-28
   promotions have brought p4096 from 130.0 to ~255 tok/s — still below the
   285.8 of the unqualified composition.
2. **Every competitor runs reduced-precision operands (F16/BF16, or int8
   activations) on every large GEMM at every layer.** The single
   F16-operand route we ship (`dense_wide256`) passes our gate at 16-47 and
   fails at 0-47 by 10% of the mean-KL budget (1.099e-3 against 1e-3). The
   router is the model's discrete amplification node (§8 rule 6): a 1-ulp
   logit change flips a near-tie and changes which experts run, which is how an
   otherwise small drift saturates the top-1 gate.
3. **So there are two tracks, and they must be ranked separately.**
   - **Track A, kernel technology at our arithmetic.** Arithmetic-preserving
     changes (LDS-staged weights, epilogue fusion, larger chunks, producer
     publication) are cheaper to admit and are ranked in §4-§5.
   - **Track B, arithmetic class.** Moving more of the model to the
     competitors' operand classes is the larger prize and the harder gate. It
     is ranked by `X-1` first: measure how far the competitors' own outputs sit
     from our strict teacher, so the envelope question is answered with data
     before another route is built to find out.

## 3. Cross-cutting decisions and measurement prerequisites

| ID | Item | Evidence | First bounded experiment | State |
| --- | --- | --- | --- | --- |
| X-1 | **Where the competitors sit on our gate.** Extends `E10` from one replay packet to the whole model | No competitor output has been scored against our strict teacher. The only related record is the August 27 heldout run *against* a llama.cpp oracle at broad-floor thresholds, on another host | Teacher-force gufo (its `qwen38_flash_next_gpu_probe` / session prefill-only tools) and halo-box (`llama-debug --save-logits`) on the canonical 12-case fixture; compute mean/p95/p99/max KL and top-1 by category against our strict teacher. If they pass our envelope, Track B is admissible in principle and the work is kernels; if they fail by a wide margin, matching their speed needs either a different envelope (a policy decision for the maintainers, not this ledger) or Track A alone | **OPEN — highest-information experiment in the ledger** |
| X-2 | **Same-host reproduction of the new comparators** (`E0` refresh) | gufo and halo-box master / PR #91 rows are author-reported; halo-box's use `llama-bench` synthetic tokens (confound C3) | Build gufo `fee9d2a` (MIT) and halo-box `ec01a7dfc` plus PR #91 head `85d8480c6` in a private prefix; run our canonical fixture with exact token ids, BF16 and F16 KV arms, cache disabled, prefill and decode reported separately; record per-family kernel time with the shared mapper in `scripts/qwen4exp_comparator_role_map.py` (add `map_gufo()`) | OPEN |
| X-3 | **Fresh attribution on the current default** | The family table in §1.2 predates the repair unroll, the F16 activation staging and both full-attention promotions | One role-marked `rocprofv3` capture of `code-p4096` at HEAD, 1x authority from `role-analysis.json` (§8 rule 1). Re-rank §4 from it | OPEN — cheapest re-ranking step |
| X-4 | **Current-default decode attribution** (was `CC-2`) | Last split is September 9 (`R6`, 54.6 ms wall / 49.3 ms device) on a composition the September 14 recovery changed | Re-run the `R6` split at HEAD before ranking any §5 row | **PREREQ for §5** |
| X-5 | **Scratch-throttle audit** | halo-box PR #87: after a kernel with a private segment runs (`flash_attn_ext_f16<256>`, 252 B/lane), the command processor throttled every later scratch-enabled dispatch ~3.3x; their Q5_1 expert-down GEMV went 220 → 67 µs by replacing `volatile` float rounding barriers with a register-only `asm` barrier, bit-identical | List `private_segment_fixed_size` for every Qwen4Exp kernel in the prefill and decode census. Any non-zero scratch on a hot kernel is a candidate for the same register-only barrier, bit-exact by construction | OPEN — cheap, exact |
| X-6 | **Qualification accounting** (was `CC-7`, `R30`) | `scripts/execution_profile_q8_wmma_prefill_layers_gate.py:426` claims qualification with no coverage term; `--case-id`, `--limit`, `--decode-steps`, `--repeat-runs` can shrink a run; `--prefill-chunk-size` defaults to 1024 | Add the coverage term; re-run any gate at a new chunk size | OPEN, prevention |
| X-7 | **Missing qualification evidence for promoted 16-47** (was `CC-8`, `R29`) | No heldout, candidate-local teacher, no isolation/graph/lifecycle, task or BF16-relative verdicts | Close before extending the admitted scope | OPEN, prerequisite |
| X-8 | **Long context 16K-256K** (was `CC-6`, `R28`) | gufo holds 1292 tok/s at d131072; halo-box PR #82 speeds 9-127-token prompts at 131K-250K depth by 28-152% by extending the per-query QSA kernel to 1-512 queries; PR #91's QSA scorer changes are +50.5% at d250000 and unchanged at d0 | Measure our own context-scaling curve first; do not transfer short-context rankings | OPEN, separate lane |
| X-9 | **Other models, quants, selector cleanup** (was `CC-9`) | Per [`REFACTOR.md`](REFACTOR.md) | Retire selectors when their rollback role ends | OPEN, maintenance |

## 4. Prefill open work, by functional block

Ranked inside each block by recoverable complete-request milliseconds. Across
blocks the order is: **B routed experts, A dense, D hyper-connections,
H chunk size, E attention, F GDN, C router/reduce**, pending `X-3`.

### 4A. Dense projections — 5234.8 ms at the anchor (31.3%)

Wide route 1404.2 ms at layers 16-47; exact coltile chain 3262.5 ms at 0-15;
565 ms of other dense roles. `dense_wide256` runs the packet shape (rows 1024,
K 2560, N 10240) in 2.573 ms, **20.9 TFLOP/s**; gufo reports 30-34 TFLOP/s for
its F16 dense kernels at 2048 rows and rejected hipBLASLt at 19-26.

| ID | Mechanism / measured cost | Opportunity and limit | First bounded experiment | Track | State |
| --- | --- | --- | --- | --- | --- |
| DP-1 | **Layers 0-15 on the exact chain: 3262.5 ms** | Drift is a monotone budget over seven measured scopes (5.81e-5 at 32-47 to 1.099e-3 at 0-47); 12-47 and 8-47 fail | Cost-weighted layer/role sensitivity over 0-15, then gate the cheapest unmeasured scope (13-47, 14-47, 15-47 or a selective role set). Re-rank after `X-1` | B | OPEN (deferred). `R04`, `R07` |
| DP-5 | **Producers publish the F16 copy.** gufo's HC mix epilogue writes the mixed rows as F32 + F16 + tiled Q8; its combine writes the F16 norm; SwiGLU writes F16 for the shared-expert down; halo-box PR #87 does the same for gated norm, SwiGLU, gated attention and HC low-rank scale | Removes the staging cast and its read of the F32 rows. The measured sharing term of the current staging is only 7.478 ms, but that term measures sharing, not the elimination of the cast pass itself | Price the current staging-cast kernels in the X-3 capture; if material, move the F16 write into the HC mix epilogue first (it feeds the most dense consumers). Exact: the F16 values are identical to the staged ones | A | OPEN |
| DP-6 | **Projection epilogue fusion.** gufo fuses the SSM causal conv into the `ssm_in` GEMM's LDS tile (`DenseF16SsmGemm`, ≥1024 rows) and q/k norm + RoPE + KV store into `attn_qkv` (`AttentionF16Gemm`) | Removes whole passes over 10-13K-wide activations; applies only where the projection already runs the wide route | Attribute the conv, q/k-norm, RoPE and KV-store passes at 16-47 in X-3; build the `attn_qkv` epilogue first if it is the larger | A (for layers already on F16) | OPEN |
| DP-7 | **Wide-tile efficiency at a larger M.** 20.9 TFLOP/s at 1024 rows against gufo's 30-34 at 2048 | Part of this is the chunk (see `RT-1`); the rest is tile quality | Time `dense_wide256` at rows 2048 and 4096 on the real shapes before any tile work; if it reaches 30 TFLOP/s at 2048 the lever is `RT-1`, not the kernel | A | OPEN |
| DP-8 | **Exact coltile chain latency and scheduling.** The projection runs at 2650 GFLOP/s, **16.1% of its own instruction-mix ceiling** (a probe with the same 40 FMA + 105 integer mix sustains 16467 GFLOP/s, 57.7% of the measured FP32 peak) (`20260915T182229`) | About 6x headroom in memory latency and scheduling without changing arithmetic. Applies to the 3262.5 ms at layers 0-15 and to the exact fallbacks; tile shape is already exhausted, so this is a load-pipeline problem | Hardware-counter capture of the layer-0-15 coltile launches (wait cycles, VMEM latency, occupancy); then a software-pipelined K loop (next-K weight/activation loads issued before the current FMAs retire, as in halo-box `90ad6cd267`) with the per-output FMA order unchanged | A | OPEN — Track A alternative to `DP-1` |
| DP-2 | One cast shared by two consumers | 7.478 ms of the packet estimate | One-cast/two-consumer timing on real inputs; do not build a general activation cache for it | A | OPEN, low value |
| DP-3 | Split-F16-operand variant (three WMMA products per tile) | Unmeasured | Microbenchmark against `dense_wide256` before any scope work | B-lite | OPEN, unmeasured |

### 4B. Routed MoE experts — 5606.9 ms post-unroll (gate/up 3377.1, down 2229.8)

Gate/up is 82.5% main matmul and 17.5% repair, 0% packing/maps/reduction
(`20260927T081954`). Both legs are weight-traffic-bound at 104-126 GB/s; gufo's
routed kernels stream 0.9-1.1 GB of expert weights per layer at 160-190 GB/s.
Down reads 1.261 GB per launch against 0.576 GB of distinct expert bytes, a
2.19x over-read and 242 GB per prefill.

| ID | Mechanism / measured cost | Opportunity and limit | First bounded experiment | Track | State |
| --- | --- | --- | --- | --- | --- |
| ED-1 | **LDS-staged expert weights for `expert_down`** (`20260927T080534`). Replace the per-lane Q5_1 block gather with a cooperative per-block LDS stage, then take a 32-row tile | Bit-exact (same bytes, same order, same iu8 WMMA and repair). Traffic 242 → ~130 GB, ~1.0-1.1 s, bounded ~1.2 s. The rejected 32-row tile kept the gather; this does not. LDS budget checked: 56,960 B of 64 KB at 128 columns x 256 K | One unit: `TILE_ROWS=32`, cooperative staging, parity test against the 16-row owner on real routing, then `--pmc GCEA_RDRAM_SIZE_REQ_sum` at real populations before any gate. If traffic halves and time does not, answer it with a smaller tile carrying the same staging | A | **OPEN — top Track A row** |
| EG-1 | **The same staging for gate/up** | 2775.1 ms main matmul; the family is weight-bound the same way | Build after ED-1 answers the traffic question; the at-risk flags are already in LDS (`qwen4_exp_q5_1.hip:1069`), so staging is scoped to the weight operand | A | OPEN, after ED-1 |
| MX-1 | **The competitor route: F16 routed experts with no repair.** Q4_K / Q5_1 / Q8_0 decoded to F16 in LDS, F16 activations cast once, F16 WMMA, F32 accumulate; gate/up paired with SwiGLU; down writes F16; expert tiles of 48/64/128 rows chosen from host-read bucket counts (gufo `RouteHints`, `RoutedGatedF16Gemm`; halo-box PR #104 `WTYPE=3` is the same design for Qwen3.6) | Removes the 1240 ms repair outright and replaces the kernel family whose ratio to gufo is ~5.9x. Same operand class as `dense_wide256`, which passes at 16-47 | After `X-1`: build the routed wrapper on our existing wide tile, scope it like the dense route (layers 16-47 first), and run the full production gate. Keep the iu8+repair chain as the strict fallback | B | OPEN — largest single prize; gated on `X-1` |
| EG-3 | **Repair: 627.3 ms of gate/up, 1240 ms family-wide** | Unroll landed (−12.7%); incidence reduction and row-bitmap measured slower; the repair is MLP-bound, not bytes-bound (3.3x less traffic was 22% slower) | None queued. Either `MX-1` removes it or a measured mechanism (registers, instruction count, occupancy) reopens it | A | OPEN, no experiment |
| ED-3 | **Q5_1 / Q8_0 split and real populations** | 43 Q5_1 layers, five Q8_0 (2, 4, 30, 46, 47); 203 of 512 experts used per layer at 1024 rows, max/median 181x | Use the `20260927T081954` population probe as the input to ED-1 and MX-1; do not tune tiles on a balanced model | — | OPEN. `R12` |

`EG-4` (expert-population-aware tiles) and `EG-2` (32-row tile without staging)
are measured dead and moved to §7.

### 4C. Router, top-k and MoE reduce — 758.4 ms (4.5%)

Router logits 617.73 ms (78.7%), `weighted_lanes_sum` 67.66, scatter/gather
52.06, `silu_mul_dual` 29.22, `router_select` 18.25.

| ID | Mechanism / measured cost | Opportunity and limit | First bounded experiment | Track | State |
| --- | --- | --- | --- | --- | --- |
| MR-1 | **Router logits producer, 617.73 ms**, reduction-structure-bound | The warp-per-token reduction wins 2.16x and +1.65% end to end and fails top-1 at the amplification node (§7C). Any reduction-order change flips the same near-ties | Only structure changes that keep each logit's reduction order: fold the shared-expert gate row into the router GEMM (gufo computes both in one GEMM), or remove a pass. Price the reorder-free share first | A | OPEN, re-targeted |
| MR-4 | **MoE epilogue folded into the next HC combine** (gufo `HcCombineMoeF16`; halo-box PR #91 fused weighted reduction with sigmoid-gated shared expert) | Our standalone fused combine lost 1.69% (§7C). What differs in the competitors is that the routed output is already F16 and the combine is the one pass that must read it anyway | Reopen only together with `MX-1`, where the down output is F16 and the combine already reads it | B | BLOCKED on `MX-1` |
| MR-3 | Reduction fusion | Caps at 8.6% of the bucket | Do not start here | A | OPEN, low value |

### 4D. Hyper-connections — 2525.4 ms (15.1%)

| ID | Mechanism / measured cost | Opportunity and limit | First bounded experiment | Track | State |
| --- | --- | --- | --- | --- | --- |
| HC-1 | **`q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32`: 2507.0 ms, 384 launches at 6.529 ms**, the largest single kernel; 987 MB/launch, 90% of it a 256x re-read of one 3.48 MB tensor at `ROW_BATCH=4` | **Row-batch retiles at a constant accumulator budget are measured dead** (RB8/H1 lost, RB2/H4 not repeatably positive, GR-specific MMQ64x128 rejected; §7D). What has not been tried is sharing the weight read explicitly: stage the dequantized 320-wide K chunk in LDS once per block and let more row groups consume it, the ED-1 mechanism on this shape | Build an LDS-staged-weight variant of the exact kernel with the same per-output reduction order; measure the complete fused kernel (sigmoid and gated-mean epilogue included) at the production shape | A | OPEN — largest single-kernel target |
| HC-3 | **Unconsumed 168 MB/layer GR gate write** | No arithmetic risk | Remove or fuse the write; measure the complete owner | A | OPEN |
| HC-6 | **The competitor HC route**: down projection as int8 WMMA from a tiled-Q8 norm produced by the previous combine, up projection as Q8 → F16 WMMA with the mix, inject and F16/Q8 publication in the epilogue, combine fused with the MoE epilogue and the next norm (gufo `HcDownF16Gemm`, `HcMixF16Gemm`, `HcCombineMoeF16`; pwilkin `6ec4a5f0d`, `90aba037c`, `816667e9d`) | This is the arithmetic the GR wide tile tried and failed at the 780-row gate (mean KL 1.554e-3). The prefill-boundary KL screen does not predict the gate here | Only after `X-1`, and only with the route scoped by layer like the dense route | B | BLOCKED on `X-1` |
| HC-4 / HC-5 | Wilkin's two mechanisms (gate GEMM + mix through memory 1.19x; bf16 streams 1.08x); GR down leg 579 ms | Folded into HC-1 and HC-6 | — | — | OPEN, folded |

### 4E. Attention, indexer and selection — 1402.5 ms at the anchor (8.4%)

Its largest member is the full-attention decode-context kernel (801.4 ms at the
anchor), now promoted twice (1.912x on its largest launch, +2.65% p4096). QSA
sparse attention proper is 910.8 ms at 18.98 ms/launch, `qsa_score_f32` 212.2 ms.

| ID | Mechanism / measured cost | Opportunity and limit | First bounded experiment | Track | State |
| --- | --- | --- | --- | --- | --- |
| QA-2 | **QSA sparse attention on tensor cores.** The kernel is issue-limited, not traffic-limited: HEADS 2 → 4 halves K/V traffic and buys only 1.08x, and effective bandwidth falls 1238.8 → 667.1 GB/s as the tile widens (`20260927T090500`). gufo runs WMMA QK/PV over the union of four queries' selections; PR #63's bucket runs at an effective 11.93 TFLOP/s. (An earlier ledger revision called that "above this host's ~9.5 TFLOP/s FP32 peak"; the measured FP32 ceiling in `20260915T182229` is ~28.5 TFLOP/s with dual issue, so that figure alone does not prove tensor-core use. The issue-limited measurement stands on its own) | Tensor cores are the only lever on the measured bottleneck; K/V reuse and LDS staging are measured dead (ceiling ~8%, §7E) | F16 WMMA `Q·Kᵀ` / `P·V` candidate over the existing selection, strict kernel as the registered fallback, full production gate. Our KV is BF16; the operand cast is part of the candidate | B | OPEN — the only live QSA row |
| QA-4 | **Indexer scoring and selection epilogues** (halo-box PR #91: relu, four-head sum and visibility in the block-score GEMM epilogue; keys-stationary scorer in LDS; one-pass top-k with 1024 threads, same bits) | `qsa_score_f32` is 212.2 ms; PR #91 reports the gain at depth (+50.5% at 250K) and none at d0, so this is a long-context row first | Measure the scorer's share on a ≥16K prompt under `X-8` before building | A | OPEN, long-context lane |
| QA-5 | **Full-attention context kernel, further** | 25% occupancy, LDS-limited in the launch carrying 81% of its time; two bit-exact levers already taken | Re-read its share in X-3 before another lever | A | OPEN, re-rank |

### 4F. GDN — 821.6 ms (4.9%)

| ID | Mechanism / measured cost | Opportunity and limit | First bounded experiment | Track | State |
| --- | --- | --- | --- | --- | --- |
| GD-3 | **Latency-pipelined token windows.** gufo's F32 row-split recurrence stages four tokens at a time through LDS and keeps three windows of loads in flight because each step otherwise pays a DRAM latency; theirs is ~65 ms per 1K tokens against our 205 | F32 arithmetic; a different reduction layout, so exactness against our parent must be checked, not assumed | Measure stall cycles in our register-state kernel first; if latency-bound, add the windowed prefetch while keeping our per-row reduction order | A | OPEN |
| GD-4 | **Chunked GDN on fp16 WMMA** (halo-box PR #91: chunk scan with fused conv/SiLU, q/k norm and gated RMS norm; NMSE ~2e-7 against the sequential kernel) | Changes arithmetic in the recurrence, which carries state across the whole prompt | Only after `X-1`; screen state drift over 4K prompts before any model gate | B | OPEN, after `X-1` |
| GD-1 | Column warps (+0.228 s) and peer (+0.126 s) | Inside noise, not additive | Do not re-screen unchanged | A | OPEN, no experiment. `R03` |

### 4G. PLE — 9.9 ms

Resident weights; not a prefill owner. halo-box's parallel host gather and
gufo's lazy `pread` serve the offload case, which is not ours. No open row.

### 4H. Chunking and runtime

| ID | Mechanism / measured cost | Opportunity and limit | First bounded experiment | Track | State |
| --- | --- | --- | --- | --- | --- |
| RT-1 | **Prefill chunk 2048 / 4096** (was `CC-1`). Competitors run 2048 (gufo) and 4096 (halo-box benchmarks) | Dense half measured −23.5% (325 ms) at 4096; MoE half modeled ~1.0 + 0.6 s; revised total ~1.3 s. The September 15 engine A/B measured +3.94% on 4K on a composition ~32% slower. 2048 is allocation-qualified within the existing scratch allowance; 4096 exposes a memory-accounting gap | Re-measure 2048 and 4096 against today's default with real per-expert populations; measure together with `DP-7` (wide-tile TFLOP/s at larger M). Re-run the gate at the new chunk (`X-6`) | A | OPEN — largest Track A lever outside the MoE rows. `R11` |
| RT-2 | **Host-read expert counts to size routed grids** (gufo downloads counts behind the shared expert and sizes the grid to the largest bucket) | Our device-owned maps avoid the sync; the maximum-grid D2H removal was neutral (§7B) | Only as part of `MX-1` | B | BLOCKED on `MX-1` |

## 5. Decode open work, by functional block

**Prerequisite: `X-4`.** The last split (`R6`, September 9, p4096
fixed-live4097): device 49.3 ms/token, of which MoE 18.09 (gate/up 5.68,
down/combine 5.60, shared expert 6.19, router 0.33, quantize/cast 0.29),
linear 24.73 (attn_qkv 5.05, lm_head 3.25, attn_gate 3.12, ssm_out 3.10,
attn_q 1.98, GR down 3.63, GR up 2.08), GDN 2.44, QSA 2.69, GR epilogue 0.86.
The September 14 recovery then disabled `Q4_DP4A64` and the other decode
selectors (TG 18.1 → 11.4 at 4K before QSA ordered decode was restored), so the
split no longer describes the default. Target: gufo's 25.87 tok/s is 38.7 ms per
token against our ~56-61.

| ID | Block | Mechanism / measured cost | Opportunity and limit | First bounded experiment | Track | State |
| --- | --- | --- | --- | --- | --- | --- |
| DE-1 | Routed experts | **MoE expert GEMV pair, 11.3 ms/token** (September 9) | The R11 warp256 kernels run 2.0x (gate/up) and 5.8x (down) at bandwidth parity and gave +6.5% TG, then failed the 18-prompt admission suite with max KL 0.121-0.431 (§7I). They use dp4a with Q8_1 activations, the same arithmetic class as gufo's `qfn_mmq_moe_gated_vec` | An exact-activation variant of the warp256 structure (the structure, not the dp4a arithmetic), measured for bandwidth first. Otherwise the row joins Track B behind `X-1` | A, then B | OPEN, after `X-4` |
| DE-2 | Shared expert | **Shared-expert block, 6.19 ms/token** | gufo runs it as one fused gated Q8_0 GEMV over activations quantized once (`qfn_mmq_q8_0_dense_vec_preq` with two weight pointers) | Attribute gate/up/down separately in `X-4`; screen a fused gate+up GEMV that keeps our per-output arithmetic | A | OPEN, after `X-4` |
| DE-3 | Dense / linear | **Linear family, 24.73 ms/token**, +3.2 ms against Vulkan with GR | halo-box PR #56: two output rows per block in MMVQ, bit-exact, +12% TG on the 27B; gufo shares one activation quantization across projections | Re-attribute in `X-4`; screen row-pairing on the largest exact GEMV (attn_qkv) | A | OPEN, after `X-4` |
| DE-4 | All | **Scratch throttle** — see `X-5` | A 3.3x per-dispatch penalty would dominate a GEMV | Run `X-5` on the decode census first | A | OPEN |
| DE-5 | Runtime | **Serving-wrapper host time, ~3.6 ms/token** above the runner loop (logits D2H, sampling, bookkeeping) after the R7 one-H2D promotion. Layer graphs are measured negative: the launch host time is already overlapped | gufo captures decode in thread-local graphs; our MoE block is graphed and wider graphs lost | Trace the wrapper in `X-4`; target the remaining synchronous copies and sampling path, not graph coverage | A | OPEN, re-derive. `R10`, `R22` |
| DE-6 | Runtime | **PM4 / retained command lists** (Wilkin review) | Decode has 48 kernel-only MoE graphs; intra-graph gaps 1.35-1.43 ms per ~68.5 ms window | Only if `X-4` shows an exposed submission bucket above those gaps | A | OPEN, low |
| DE-7 | Attention, GDN | QSA 2.69 and GDN 2.44 ms/token, both faster than Vulkan in September | No gap to chase | — | — | CLOSED as targets |
| DE-8 | Head | lm_head 3.25 ms/token | gufo keeps the full Q8 head; the vocab-subset trick belongs to the draft (§6) | — | — | CLOSED as a target |

## 6. MTP open work

**State.** Explicit, default-off, capped at 1K context. Serial exact target
verification is the only admitted verifier. The retained ten-prompt serial
provider is exact and reaches **0.7407x** true AR, regressing every category
(`20260902T092733`). Every verifier that batches target rows has either changed
a token or been slower (§7J).

**What the fast engines do differently.** gufo's decode arithmetic is designed
to be **width-invariant**: verification "keeps the same quantization and
reduction as single-token decoding", its integer-WMMA Q8 verifier "preserves
K8 partials/FMA order and reduces each result once" through 48 rows, and its
batched decode matches isolated execution exactly at C2-C8. That is what lets it
verify several drafts in one pass. Ours differs: the multirow prefill path and
the c1 decode path use different kernels and arithmetic, so a batched verifier
built from prefill primitives fails the state RED (`20260902T082732`).

| ID | Mechanism | Evidence | First bounded experiment | State |
| --- | --- | --- | --- | --- |
| MT-1 | **Width-invariant rows≤8 verifier.** One target pass over the draft rows that uses the c1 decode arithmetic per row, so every row equals what serial decode would have produced | The pooled deferred-head verifier is exact but its 0.172 ms head saving cannot pay for the transaction (−3.52% complete, −12.7% public); the missing piece is a body pass whose cost grows sub-linearly with rows while keeping c1 arithmetic | Pick the two largest decode owners (`DE-1`, `DE-3`) and build rows≤8 variants that reproduce the c1 kernels' per-output reduction bit for bit; RED against serial `target.step` at rows 2-8 before any timing | OPEN — prerequisite for every MTP row below |
| MT-2 | **Capacity beyond 1K** | Provider, draft runner and request checks all enforce 1,024 tokens; the 4K MTP suite cannot run | Lift the cap under the context-capacity policy once `MT-1` exists | BLOCKED on `MT-1` |
| MT-3 | **Adaptive / cost-aware draft depth** (gufo default adaptive 1-7; halo-box PR #102 learns verify cost per width online) | Depth budget fitting was blocked by the 1K cap and the absence of a paying verifier | After `MT-1`/`MT-2`, fit on a disjoint partition, never on the evaluation prompts | BLOCKED |
| MT-4 | **Cheaper draft head** (halo-box PR #101 vocabulary subset: draft step 3.13 → 1.42 ms on Qwen3.6-35B-A3B; our compact-head rung) | Even deleting our 3.153 ms full head left serial MTP at 0.964x AR, so the head is not the blocker today | Re-open after `MT-1` changes the economics | BLOCKED on `MT-1` |
| MT-5 | **Prefill-side MTP cost** (gufo: one-row MTP prefill lag keeps 320 KiB of hidden state instead of replaying a final chunk; final-tile catch-up restricts the predictor to the last aligned tile) | Unmeasured for us | Measure our MTP prefill catch-up cost at 1K first | OPEN |
| MT-6 | **Compact recurrent rollback** (gufo: one full state plus exact FP32 update operands, ~147 MiB/session at seven drafts) | Our rejection-depth RED grid passes; rollback memory cost at depth unmeasured | Charge rollback bytes in the context-capacity ledger once `MT-2` opens | OPEN, after `MT-2` |
| MT-7 | **Benchmark protocol** | Speedup claims need a true no-MTP AR baseline from the same protocol on the full mtp-bench category suite (AGENTS.md anti-gaming rules) | Keep; no single-prompt or repetitive-only rows | Standing rule |

## 7. Tried and not retained, by functional block (the registry)

Rows here are **not** to be re-run unchanged. Outcome classes:

- **accuracy** — failed the numerical or task gate;
- **perf** — exact or admissible but slower or neutral on the complete owner or whole model;
- **state** — failed determinism, state or parity;
- **withdrawn** — promoted, then disabled by the September 14 recovery (`49ffa3cb5`, `PRODUCTION_ARITHMETIC_RECOVERY_FLAGS`) because the composition failed max KL 0.0546;
- **dead** — measured not to address the bottleneck;
- **blocked** — stopped by a structural or evidence prerequisite.

Promotions that are live are listed per block for context; their evidence is in
STATUS and the cited entries.

### 7A. Dense projections

Live: `dense_wide256` at 16-47 (`20260916T205304`), F16 activation staging
(`20260926T143853`), exact Q8 wave-scale prefill (`20260905T234311`).

| Tried | Class | Evidence | Entry | Reopen only with |
| --- | --- | --- | --- | --- |
| Dense Q8 MMQ stack (`Q8_MMQ_PREFILL`, raw-Q token64, vector staging) | withdrawn | Passed short prompts, failed canonical depth after the recovery | `20260829T113335`, `20260914T154413` | Operand localization showing which role carries the depth failure |
| iu8 WMMA dense Q8_0 (`Q8_IU8_WMM`), generic and non-GR restorations | withdrawn, then accuracy | Both restorations fail the canonical envelope | `20260909T024000`, `20260915T015835`, `20260915T022059` | Role-isolated evidence |
| F16 WMMA / wide at 0-47, 12-47, 8-47 | accuracy | 0-47 mean KL 1.099e-3; 12-47 `general_en` 1.089e-3; 8-47 max 7.30e-2, top-1 1532/1548 | `20260917T053701` | A cheaper unmeasured scope (`DP-1`) or changed arithmetic |
| Single-plane d4 MMQ (the 2.40x lever) | accuracy | Fails unit numerics | `20260903T000259` | — |
| Two-plane d4x2 MMQ (1.38-1.4x) | accuracy | mean KL 1.598e-3, p95 8.70e-3, max 7.18e-2, top-1 0.983 | `20260910T113000` | — |
| MMQ tile retiles (128x64, 64x64, 64x128) | perf | 1.00-1.05x, 0.93-0.97x, 0.77-0.82x; 128x128 optimal in family | `20260902T235458`, `20260910T103000` | — |
| Raw-vector square64; prepacked token64; bank-first loop order; output-scale hoist | perf | Slower, mixed, or spills | `20260907T120910`, `20260907T115636`, `20260907T101740`, `20260907T101037` | — |
| Q8 MMQ attention gate (K2560 / N6144) | state, then perf | +3.52% p508 with numerics passing, first state capture differed; after state fixes, five-pair screen noisy | `20260831T085446`, `20260902T003351` | A stable five-pair win |
| Q8 paired-load coltile; early wave publication; expanded F32 sidecar | perf | Full A/B regress; loses at all four shapes; 6.2x slower | `20260907T050824`, `20260907T180850`, `20260907T181554` | — |
| Exact-chain tile-shape tuning | dead | Tile-shape family exhausted, no spills, occupancy at the register-only maximum; attention-gate geometry exhausted. The same entry's "cannot be made faster" ceiling was **overturned** by `20260915T182229` (see `DP-8`) | `20260915T182013`, `20260831T124221` | — |
| Two-Q8 fallback | accuracy (task) | Adds a false model-specific GQA example absent from strict | `20260914T080323` | — |
| Per-block base-offset table (dense-context sibling) | perf | 1.41x slower than per-token | `20260928T155352` | — |

### 7B. Routed MoE experts

Live: exact iu8-risk + repair gate/up (`20260907T190613`), weight-exact iu8
Q5_1 down (`20260908T194500`), guarded block-scale Q8_0 selected down
(restored 2026-09-14/15), T0 grouped selected Q8_0 down (`20260904T060311`),
repair unroll (`7804517a2`), Q5_1 fold128 / folded pair / register reuse /
per-row publication, Q8 down register reuse and bundled reduction.

| Tried | Class | Evidence | Entry | Reopen only with |
| --- | --- | --- | --- | --- |
| Grouped / WMMA / peer fast prefill profile | accuracy | 8.67 → 211.76 tok/s structural; fails the 687-row gate | `20260827T103921` | — |
| Q4 T16 and pack8 replacement layouts | accuracy, perf | Neutral prefill, slower decode, fail mean/p95 KL; pack8 repack impractical | `20260827T113708`, `20260827T133503` | — |
| Layer-2 grouped Q5_K route | perf, accuracy | ~1% regression over five pairs; exceeds prefill-last KL | `20260830T223941`, `20260831T065604` | — |
| `PRODUCTION_MOE_PREFILL`, `Q4_IU8_PREFILL` overrides | withdrawn, perf | Slower on `code-p4096` in the disabled-family sweep | `2026-09-16-disabled-family-recoverable-time` | New tiling, packing, fusion or repair work |
| Exact scheduling family: expert grid 128, wider output tiles, Q4 team2 | perf | All bit-exact, neutral or negative | `20260831T073529`, `20260831T074736`, `20260831T075708`, `20260901T223441` | — |
| Q4 register caches (decoded floats 0.666x, packed nibbles 0.404x, volatile metadata 0.136x), packed-nibble LDS cache, fixed-geometry specialization, row4/output4 retile | perf | All exact, all slower or distribution-dependent | `20260906T083411`, `20260906T084224`, `20260906T084909`, `20260906T091305`, `20260906T142927` | — |
| Q5_1 simultaneous pair reduction; M2 hierarchical reduction; first-wave tail; row16 | perf | Leaf wins (M2 +6.99%) that regress the complete owner | `20260906T103958`, `20260905T005246`, `20260907T051623`, `20260907T095223` | — |
| Q4_K M1 lever; Q4 paired-block weight pipelining; Q4 dual-SiLU 16-column tile | perf | Measured loss; 29-59% slower | `20260903T212252`, `20260907T063824`, `20260913T125803` | — |
| Q5_K bundled row4 | perf | Five prefill and six request-wall cases regress | `20260906T153612` | — |
| Two-plane MoE gate/up iu8 (R13) | perf, accuracy | +2.1% only; T1 drift and repair explosion | `20260910T123000` | — |
| Hot-expert weight reuse; next-K pipelining | dead | Routing diffuse (~335 of 512 experts per layer-chunk; top-16 cover 25.3%) | `20260910T111500` | — |
| IU4 grouped prefill | dead | 1.892x instruction nets 0.945x precision-matched; residual ceiling 1.06x | R8 | A different precision split with a proven budget |
| 32-row M tile, 128-thread and 8-warp (`EG-2`) | perf | Spills 492 B/thread (4.8-8.1x slower); spill-free 8-warp still 1.3-2.2x slower; real DRAM −6.3% against a modeled 45% | `20260921T160629`, `20260921T161611` | The LDS-staged weight path (`ED-1`), which this did not test |
| Expert-population-aware tiles (`EG-4`) | dead | Real skew moves the tile count 6.6% from the balanced model; the tile count is not the constraint | `20260927T081954` | — |
| Grouped Q4 SiLU epilogue | state | GPU fault at production width | `20260901T222445` | — |
| Q5 down + route epilogue; Q8 down route scaling | blocked | Incompatible ownership; incumbent FMA boundary binding | `20260901T222659`, `20260901T223337` | — |
| Maximum-grid MoE D2H removal | perf | Exact, complete p512 wall neutral | `20260901T212612` | — |
| P1 device-driven grouped Q8_0 down | perf | Slower than strict selected GEMV at every shape | `20260830T202256` | — (superseded by the promoted T0 grouped down) |

### 7C. Router, top-k and MoE reduce

Live: exact F32 router tile4 (`20260831T103038`, +1.6% p508).

| Tried | Class | Evidence | Entry | Reopen only with |
| --- | --- | --- | --- | --- |
| Warp-per-token router reduction | accuracy | 2.16x kernel, +1.65% e2e; top-1 444/450 against ≥99%; `prefill_last` 3.5x over budget | `20260927T165141` | Making the selection itself exact |
| Router fused stable top-10 (two variants) | perf | Exact; global coordination costs more than the saved launch | `20260831T104943` | — |
| Router shuffle tail | perf | One prefill and four request-wall cases regress | `20260907T015737` | — |
| Routing compaction `top10_parallel` | perf | Bit-exact, measured loss | `20260903T213943` | — |
| Fused MoE combine (PF-4 lever 2) | perf | −1.69% prefill, negative in 12/12 cases | `20260903T234527` | `MR-4` with `MX-1` |
| Shared down + combine; shared Q8 activation pair; shared epilogue contraction | perf | Kernel windows shrink, whole-model wall does not | `20260831T113724`, `20260831T112025`, `20260901T223722` | — |

### 7D. Hyper-connections

Live: exact fused GR sigmoid mean and GR up sigmoid mean (`20260831T094724`,
`20260831T120921`), exact GR wave-scale prefill (`20260906T012226`).

| Tried | Class | Evidence | Entry | Reopen only with |
| --- | --- | --- | --- | --- |
| iu8 WMMA GR up / down (`GR_IU8`, `GR_IU8_DOWN`) | withdrawn, accuracy | +11.3% prefill combined when promoted; independently mean KL 1.40e-3 / 1.31e-3, p95 7.40e-3 / 6.84e-3 | `20260908T224000`, `20260908T234000`, `20260914T235202`, `20260915T001253` | A structurally different mechanism |
| Compensated GR down; P4 up correction | accuracy | Pass operand screens, fail the full gate; code removed | `20260915T010915`, `20260915T013552` | — |
| GR wide tile (`HIPENGINE_QWEN4_EXP_GR_WIDE`) | accuracy | 4.38x on the projection; mean KL 1.554e-3, max 7.349e-2, top-1 768/780 | `20260927T075248` | `X-1`, then `HC-6` |
| Constant-accumulator row reuse RB8/H1; equal-accumulator RB2/H4; GR-specific MMQ64x128 | perf | RB8/H1 loses on both real weight banks; RB2/H4 not repeatably positive; 64x128 rejected | `20260906T075517`, `20260906T080204`, `20260906T021059` | The LDS-staged weight path (`HC-1`) |
| GR down + scaled SiLU; qkv + gate composite; Q8-down / F32-inject composite | perf, blocked | Regresses the rows508 window; no paying shared-arithmetic mechanism | `20260831T123724`, `20260902T005602` | — |

### 7E. Attention, indexer and selection

Live: wave32 sparse attention, persistent compressed QSA key blocks
(`20260827T193728`, `20260827T182330`), dense fixed256 (`20260831T133248`), four-head (quad) sparse
prefill (`20260907T173548`), ordered-v2 QSA decode (restored), full-attention
int32 token-offset table and four-token QK unroll (`20260927T175027`,
`20260928T155352`).

| Tried | Class | Evidence | Entry | Reopen only with |
| --- | --- | --- | --- | --- |
| H256 wave8 sparse attention; contiguous-chunk H256 | accuracy | Large deterministic speedup; fails p4096 numerics, top-1 and tasks | `20260831T181221`, `20260831T182919` | Tensor-core route under `QA-2` |
| QSA flash prefill (`QSA_FLASH_PREFILL`, 33% family win) | withdrawn | Part of the failed recovery composition | `20260829T203508`, `20260914T090826` | — |
| Grid-exploded multirow QSA prefill; row-tiled persistent QSA prefill | perf, blocked | Untenable CTA grid; static geometry cannot reach the owner | `20260901T221408`, `20260901T221630` | — |
| Grouped-column ordered QSA value pass | perf | Exact, loses the leaf screen | `20260901T211624` | — |
| QSA long-context prefill reuse (PF-2) | perf | Slower at every shape | `20260903T142550` | — |
| Paired-head QSA (HEADS=2) | perf | Prefill gain with a decode-transition loss traced to on-CPU time; superseded by quad | `20260907T131004` | — |
| CPU-active transition workaround | perf | Loses once its added wall is charged | `20260907T153915` | — |
| K/V reuse by wider HEADS; LDS K/V staging | dead | HEADS 2 → 4 halves traffic, buys 1.08x; ceiling ~8% (~73 ms) | `20260927T090500` | — |
| D256 tile/occupancy sweep (nasone32 `ed11a0d2f`) | blocked, not run | Gate closed; design pre-declared | `20260910T114500` | The issue-limited finding makes it a tensor-core question now |
| Masked-slice skipping | dead | No dense selected-token mask remains to skip | `20260901T224112` | — |

### 7F. GDN

Live: tile-16 prefill (`20260905T050430`), register-state prefill
(`20260905T182225`), wave-tail normalization (`20260907T094528`).

| Tried | Class | Evidence | Entry | Reopen only with |
| --- | --- | --- | --- | --- |
| Column warps, peer (`GDN_COLWARPS_PREFILL`, `GDN_PEER_PREFILL`) | withdrawn, perf | +0.228 / +0.126 s, inside run-to-run spread, not additive | `20260829T185915`, `20260829T195152`, `20260914T090826` | A complete-owner mechanism |
| Journey multi-column (`R01`) / tiled (`R02`) restorations | accuracy (task) | Numerics pass; Japanese complete-EOS task fails | `20260914T201148` | A passing predeclared task review |
| Early column blocks | state | Fails production-width parity; a one-block fixture missed it | `20260831T135620` | — |
| Prepared strict GDN; persistent transposed state | perf | Exact, regress the operation-complete boundary | `20260831T130106`, `20260831T214146` | — |
| t128 normalization; w32 (1024-thread) prefill blocks | perf | 0.982x; +1.6-6.8% slower at ≥64 rows | `20260901T224609`, `20260903T164443` | — |
| DPP within the admitted suffix | promoted, inactive | Exact, but the suffix it applies to is off after the recovery | `20260913T222708` | Suffix re-admission |

### 7G. PLE

Live: exact cold PLE mapping (`20260913T214314`).

| Tried | Class | Evidence | Entry | Reopen only with |
| --- | --- | --- | --- | --- |
| Direct-to-ring IQ4_NL staging | perf | Breaks vectorized multi-row dequantization | `20260902T024158` | — |
| One-chunk-ahead prefill overlap | perf | Ownership-safe, neutral | `20260902T032749` | — |
| Page-cache warm sweep | perf | Promoted then demoted: unqualified resource cost | `20260910T103000` | — |

### 7H. Chunking and runtime

Live: chunk64 → 512 → 1024 (`20260907T002907`), MoE backend registration cache
(+3.29 / 1.59 / 1.88% PP), wrapper-host single H2D (`20260910T153000`, −0.77
ms/token), device argmax (`20260831T153159`).

| Tried | Class | Evidence | Entry | Reopen only with |
| --- | --- | --- | --- | --- |
| P8 production graphs | blocked | Short-context submission savings are real; cannot cross into p4096 sparse QSA | `20260902T014656` | — |
| Submission batching without dispatch removal | dead | External sweep −1.3 to −2.4% | `20260901T224939` | — |
| Decode layer graphs (`HIPENGINE_QWEN4_EXP_LAYER_GRAPHS`) | perf | Exact, 49.8 → 50.5 ms/step; launch host time already overlapped | `20260909T172500` | — |
| Persistent streamed workers | perf | Rejected on gfx1151 | `20260905T033643` | — |

### 7I. Decode kernels

| Tried | Class | Evidence | Entry | Reopen only with |
| --- | --- | --- | --- | --- |
| Q5_1 wave64 selected decode | accuracy | +6-8% warm decode; fails mean/p95 KL | `20260827T105018` | — |
| R11 warp256 dp4a MoE GEMV pair | accuracy (task) | +6.5% TG; admission suite max KL 0.121-0.431, top-10 overlap 6-7/10 | `20260910T083000` | An exact-activation variant (`DE-1`) or `X-1` |
| DP4A safe-43 decode (`Q4_DP4A64`) | withdrawn | Part of the failed composition | `20260829T000503`, `20260914T090826` | — |
| GDN t128 decode | perf | Memory-dominated recurrence, no gain | `20260831T205428` | — |

### 7J. MTP

| Tried | Class | Evidence | Entry | Reopen only with |
| --- | --- | --- | --- | --- |
| Transactional multirow target verification | accuracy | 1.088x on one p512 row; changed a `general_en` target token | `20260828T052108` | `MT-1` |
| Prompt `_prefill_chunk` as a rows2 verifier | state | Top-1 matches; hidden, logits and recurrent state fail | `20260902T082732` | `MT-1` |
| Deferred-head verifier (serial body, batched head) | perf | Exact at rows 2-8; −3.52% complete | `20260902T084202` | — |
| Public pooled deferred verifier | perf | 10/10 exact; 12.7% slower than serial MTP in every category | `20260902T091827` | — |
| Device acceptance / commit kernels | blocked | Host acceptance is 0.020% of wall | `20260902T092117` | A paying verifier |
| Compact draft head at 8K/16K/32K | blocked | Deleting the 3.153 ms full head still leaves 0.964x AR | `20260902T080349` | `MT-1` |
| Confidence thresholds; depth budget; EXP-016 falsification | blocked | No calibrated statistic or disjoint data; 1K cap; external patch unavailable | `20260902T080024`, `20260902T092420`, `20260902T075334` | `MT-1`, `MT-2` |

## 8. Measurement rules for this model

1. **The 1x authority is `role-analysis.json` `exact_role_kernels`**
   ([`2026-09-17-qwen4exp-per-role-cost`](../benchmarks/results/2026-09-17-qwen4exp-per-role-cost/)).
   The raw `rocprofv3` CSV there is **two repetitions**; three artifacts
   published 2x counts before this was corrected on 2026-09-27.
2. **`rocprofv3` `Grid_Size_X/Y` are work-items, not grid dimensions.** The
   traced `(192, 1021, 1)` QSA launch is a `(6, 1021, 1)` grid.
3. **Do not classify families by kernel-name substring.** The QSA family's
   largest member is a full-attention kernel; our activation conversion is
   inside the dense kernel rather than an `mmb_cvt_*` symbol.
4. **Do not add the A/B savings** in STATUS §2.4; they are separate
   experiments on different compositions.
5. **Rank by recoverable complete-request milliseconds.** A leaf ratio finds a
   row; it never ranks one. Many §7 rows are leaf wins that lost the whole
   model.
6. **A passing KL envelope beside a failing top-1 is the router's discrete
   amplification node**, not a sloppy kernel. Check `flip_eligible_rows`
   before blaming arithmetic.
7. **Production candidates do not need bit-exactness**; the binding contract
   is [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md). A byte-identical change
   is cheaper to justify, not more admissible.
8. **Competitor rows carry their basis.** Same-host fixture rows and
   author-reported rows never share a ratio. `llama-bench` prompts are
   synthetic tokens, so their MoE routing and PLE locality are not our
   fixture's (confound C3).

## 9. Cadence

- Re-rank after every retained change and after `X-3`. A promotion
  invalidates the family table for its own family and any family whose work it
  moved.
- Update [`QWEN4EXP-STATUS.md`](QWEN4EXP-STATUS.md) in the same commit as a
  promotion that changes it, and this ledger when a row changes state.
- A rejected row moves to §7 with its class, evidence, entry and reopen
  condition in the same commit that records the rejection.
- New candidates enter through this ledger, not through a new campaign
  document. External source reviews add rows here and cite the pinned commit.
