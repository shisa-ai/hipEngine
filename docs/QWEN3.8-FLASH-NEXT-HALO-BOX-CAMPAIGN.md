# Qwen3.8-Flash-Next halo-box Follow-up Campaign

Status: **HB-0 complete 2026-09-02; local Q4_K_XL smoke evidence exists, but
no binding comparison exists until HB-1.** Values attributed to the PR author
remain author-reported until reproduced on `zbook`. The binding comparator set and the section-6 closure rules remain
owned by
[`QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md`](QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md);
this document is a subordinate lane that (a) adds the halo-box fork as a
measured comparator, (b) profiles it role-by-role against hipEngine, and (c)
converts every confirmed per-mechanism advantage into a `W/C/O/s` candidate for
the main campaign's impact queue.

The premise, stated explicitly: matching or beating each matched
microbenchmark family one at a time is how the aggregate gap closes. This
campaign exists to enumerate those families from the halo-box diff, measure
each one on both engines, and feed the winners into the main campaign. It does
not relax any correctness, evidence, or anti-gaming rule.

## 1. Source identity and claim summary

Lane source: [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp)
[PR #11](https://github.com/halo-box/strix-llama.cpp/pull/11) by
gaetan-puleo, published 2026-08-30, branch `qwen-3.8-flash-pp-improvement`,
AGENT-AUTHORED (OpenCode) per the PR disclosure.

| Identity | Value |
| --- | --- |
| Base ("halo-box stock") | `6c84c7d5d8833c6e0df69628f75a0f599797934e` (the upstream #27742 Qwen4Exp merge) |
| Change ("halo-box PR11") | `a7ad7b7f508746870e03d0da60cca420295a04b9` |
| PR commit diff size | 2,374 insertions, 248 deletions, 35 files for `a7ad7b7f^..a7ad7b7f`; all compute changes gated on `GGML_CUDA_CC_IS_RDNA3_5` or AMD Vulkan paths |
| Their host | Ryzen AI Max+ 395 / Radeon 8060S, 128 GB, kernel 7.1.8-200.fc44, ROCm 7.14, RADV |
| Their protocol | UD-**IQ4_XS** (4.25 bpw), PP2048, `-b 2048 -ub 2048`, F16 K/V, FA on, `--load-mode none`, 4 repeats |
| Author-reported ROCm gain | +44.78% / +25.55% / +15.28% / +13.46% PP2048 at depth 0/12K/32K/64K (441.37→639.03 tok/s at depth 0) |
| Author-reported Vulkan gain | +7.86% / +5.85% / +4.71% / +2.10% at the same depths |
| Author-reported correctness | Byte-identical prompt tokens and complete final logits vs a separately built reference (ROCm logits SHA-256 `4631827e…`, Vulkan `aaa573fb…`), deterministic decode, 5798/5798 HIP backend ops, 17335/17338 Vulkan (3 pre-existing stock failures) |

Important baseline note: `6c84c7d5` **predates** our pinned upstream
comparator `f1793c1c4` (which includes #27880 graph-split reduction). The
halo-box "stock" column is therefore not interchangeable with our existing
"patched upstream" lane; keep all four lanes separate.

### 1.1 Mechanism inventory from the reviewed diff

Every mechanism below is gated to RDNA 3.5 (`GGML_CUDA_CC_IS_RDNA3_5`) unless
noted. The third column records whether the mechanism is expected to activate
on our binding `UD-Q4_K_XL` payload (44.35 GB Q4_K, 27.05 GB Q5_1, 9.61 GB
Q8_0, 1.15 GB Q5_K, 313 MB F32, 39 MB BF16, 28.80 GB IQ4_NL PLE); HB-2 confirms
or denies each by kernel-name census, never by assumption.

| # | Mechanism (source) | Expected on UD-Q4_K_XL? |
| --- | --- | --- |
| M1 | gfx1151 MMQ tile retune, `mmq-config-rdna3-5.cuh`: Q4_K/Q5_K/Q6_K/Q8_0 tile configs shrunk from 256-wide to 128-wide across most `ny`; partial IQ3_XXS/IQ3_S/IQ4_XS retune | **Yes** for Q4_K/Q5_K/Q8_0 MMQ shapes; Q5_1 coverage unverified |
| M2 | Parallel top-10 expert-id compaction, `mmid.cu` `mm_ids_helper_top10_parallel`: 8 warps per expert block, `n_tokens >= 64`; new `case 10` dispatch | **Yes** — type-agnostic routing prepass; `f1793c1c4` has a simpler top-10 specialization |
| M3 | Device-built routed-compact MMQ, `mmq.cu` `build_mmq_routed_descriptors` + `mul_mat_q_routed_compact`, gated `(Q6_K && J==32) \|\| (Q8_0 && J==48)` with `nchannels_y == 256` | **Expected no** — our routed experts are Q4_K/Q5_1; verify at runtime |
| M4 | J48 dispatch specialization for MoE-id MMQ: Q8_0 (`mmq_use_rdna3_5_q8_id_j48`), IQ2_XS/IQ3_XXS (12288-col), J64 for IQ3_S/IQ4_NL/IQ4_XS at `nchannels_y == 512` | **Expected no** on Q4_K_XL; **yes** on the IQ4_XS diagnostic arm |
| M5 | Fused weighted top-10 expert-output sum, `args.cu` `ggml_cuda_op_weighted_expert_sum` with `<10, 2560>` specialization, plus fused shared-expert `shared_mul_add_f32` (mul+add+residual) | **Yes** — n_embd 2560 matches Qwen4Exp |
| M6 | GDN tuning, `gated_delta_net.cu`: 32 warps/block (vs 4) for S_v=128, H∈{32,48,64}; token-tile-16 state-in-register prefill kernel for H=32, `n_tokens >= 16`, single sequence | **Yes** — matches Qwen4Exp GDN geometry |
| M7 | Decode FA GQA opt, `fattn*`: gated to `Q->ne[1]==1`, `gqa_ratio==6`, **Q8_0 K/V only**; faster FA tile variants were rejected by the authors for changing logits | **No** under the BF16-KV baseline; Q8_0-KV diagnostic arm only |
| M8 | Elementwise/recurrent specializations: transposed concat tile-16, direct-index contiguous mul, quantize/norm/sumrows gfx1151 paths | **Yes** where shapes match; map per kernel in HB-3 |
| M9 | Vulkan: cooperative-matrix dequant selection for wide prompt matmuls and large MoE batches, tiled recurrent concat, direct-index mul | Backend-disjoint; design evidence only, per main-campaign scope rules |

## 2. Objective and boundaries

### In scope

- Host `zbook` (gfx1151), same ROCm toolchain and build flags as the existing
  comparator: Release, `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx1151`,
  `GGML_HIP_GRAPHS=ON`, `GGML_HIP_MMQ_MFMA=ON`. Vulkan builds are optional
  design evidence, never a closure target here.
- Two halo-box binaries, built once and frozen: base `6c84c7d5` and PR head
  `a7ad7b7f`. If startup is blocked on the 111-GB artifact, apply the same two
  documented loader patches (`aca70db1…`, `971d428d…`) and label the binaries
  **patched**, exactly as the existing patched-upstream lane.
- Binding arm: pinned `UD-Q4_K_XL` + BF16 K/V through the canonical exact-token
  fixture and protocols of the main campaign.
- Diagnostic arms (separate denominators, never mixed into binding rows):
  UD-**IQ4_XS** matching the PR's tested quant, and Q8_0 K/V to exercise M7.
- Concurrency extension c=1..8 (section 3) as the topline matrix.

### Out of scope

- Promoting any halo-box number, reproduced or not, into the main campaign's
  closure targets without a main-campaign baseline event.
- Changing hipEngine's binding quant, KV type, fixture, or profile manifests.
- Treating the author-reported +44.78% (IQ4_XS, PP2048, `--load-mode none`,
  older baseline) as transferable to `UD-Q4_K_XL` exact-token workloads.
- Vulkan shader work.

## 3. Topline matrix (c=1..8)

These tables are the campaign scoreboard. Every cell starts as `TBD` and is
filled only from a measured, identity-pinned run under section 4's protocol.
`c` is the number of simultaneous request slots: `llama-server -np c` for the
llama.cpp lanes, hipEngine's serving/batch path for its lane. A lane with no
working concurrent mode records `unsupported`, never zero. c=1 binds first;
c=2..8 are extension rungs that must not average away a c=1 result.

Lanes: **hipEngine** (named production, pinned UD-Q4_K_XL, BF16 K/V),
**upstream HIP** (patched `f1793c1c4`, existing comparator),
**HB-base** (halo-box `6c84c7d5`), **HB-PR11** (halo-box `a7ad7b7f`).

### 3.1 Prefill tok/s, UD-Q4_K_XL, BF16 K/V (canonical p512/p1024/p4096)

| c | hipEngine | upstream HIP | HB-base | HB-PR11 |
| ---: | --- | --- | --- | --- |
| 1 | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD | TBD |
| 6 | TBD | TBD | TBD | TBD |
| 7 | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD |

### 3.2 Decode tok/s, UD-Q4_K_XL, BF16 K/V (tg128 after p512/p1024/p4096)

| c | hipEngine | upstream HIP | HB-base | HB-PR11 |
| ---: | --- | --- | --- | --- |
| 1 | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD | TBD |
| 6 | TBD | TBD | TBD | TBD |
| 7 | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD |

### 3.3 MTP vs same-lane AR (complete request wall ratio, full category suite)

Upstream qwen4exp MTP (#27836) is unmerged; halo-box PR11 ships no MTP changes.
Record `n/a` where the lane has no merged qwen4exp MTP path.

| c | hipEngine | upstream HIP | HB-base | HB-PR11 |
| ---: | --- | --- | --- | --- |
| 1 | TBD | n/a | n/a | n/a |
| 2 | TBD | n/a | n/a | n/a |
| 3 | TBD | n/a | n/a | n/a |
| 4 | TBD | n/a | n/a | n/a |
| 5 | TBD | n/a | n/a | n/a |
| 6 | TBD | n/a | n/a | n/a |
| 7 | TBD | n/a | n/a | n/a |
| 8 | TBD | n/a | n/a | n/a |

### 3.4 PR-shape reproduction diagnostic, UD-IQ4_XS (their protocol, not binding)

Reproduce the PR's own protocol exactly: `llama-bench -p 2048 -d
0,12000,32000,64000 -b 2048 -ub 2048 -n 0 -r 4 -ngl 99 -fa on -ctk f16 -ctv f16
--load-mode none`. The published column is author-reported context, never a
pass/fail gate.

| Depth | HB-base published | HB-base local | HB-PR11 published | HB-PR11 local | Local gain |
| ---: | ---: | --- | ---: | --- | --- |
| 0 | 441.37 ± 1.85 | TBD | 639.03 ± 4.84 | TBD | TBD |
| 12000 | 323.14 ± 1.14 | TBD | 405.70 ± 1.71 | TBD | TBD |
| 32000 | 235.37 ± 1.01 | TBD | 271.34 ± 0.96 | TBD | TBD |
| 64000 | 167.44 ± 1.49 | TBD | 189.98 ± 0.88 | TBD | TBD |

## 4. Measurement protocol

1. **Freeze identity first.** Record hipEngine commit, halo-box source and
   binary SHA-256 for both lanes, toolchain/ROCm/Mesa versions, model part
   hashes, power profile, and host state before any number is collected.
2. **Build once, label forever.** Both halo-box binaries in one session with
   identical flags; note patch state in the lane label (`hb-base`,
   `hb-pr11`, plus `+loader-patches` when applied).
3. **Binding runs** use `scripts/qwen4exp_canonical_ar_bench.py llamacpp` with
   the canonical fixture and the main campaign's server arguments
   (`-ngl 999 -fa on -ctk bf16 -ctv bf16 -c 4352 -b 8192 -ub 2048`), one
   discarded warmup, three measured requests per case, and the repeatability
   classification from the main campaign (repeat arm per build before any
   cross-build comparison; tie-class vs state-class split).
4. **PR-shape runs** (section 3.4) keep `--load-mode none` and F16 K/V exactly
   as published; they are a reproduction anchor, not a campaign target.
5. **Profiling** uses `scripts/qwen4exp_llamacpp_exact_profile.py` under
   `rocprofv3` for both halo-box lanes at p512/p1024/p4096 prefill and
   live-513/1025/4097 decode, then `scripts/qwen4exp_trace_analyze.py` with
   explicit windows. hipEngine's canonical role ledger already exists; do not
   re-profile hipEngine unless a retained unit landed since the last ledger.
6. Every artifact carries a `lifecycle` block and follows the main campaign's
   section-3.6 rules (first-arm discard, counterbalancing, per-step census,
   no nested-process profiling).
7. Rollup: each measured unit emits a compact JSON under
   `benchmarks/results/`, a dated `benchmarks/CHANGELOG.md` line, and a
   worklog entry; comparator refresh rules from the main campaign apply.

## 5. Mechanism match ledger

One row per mechanism from section 1.1. HB-2's kernel-name census fills
"Active on Q4_K_XL?" from the HB-PR11 trace (expected kernel names:
`mm_ids_helper_top10_parallel`, `mul_mat_q_routed_compact`,
`build_mmq_routed_descriptors`, `weighted_expert_sum_rows_f32`,
`shared_mul_add_f32`, the 32-warp GDN kernel, retuned `mul_mat_q` configs).
HB-3 then pairs each active family against the hipEngine owner from the main
campaign's aligned-delta ledger and records a per-family ratio with the exact
matched shape. A family where hipEngine already wins is recorded as such —
that is a match-ledger result, not a failure.

| # | halo-box mechanism | Active on Q4_K_XL? (measured) | hipEngine owner (current role) | Matched shape / microbench | HB-base vs HB-PR11 | hipEngine vs HB-PR11 | Disposition |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | MMQ tile retune Q4_K/Q5_K/Q6_K/Q8_0 256→128 | TBD | Selected Q4 gate/up, selected Q5_1 down, dense Q8, dense projection compute | TBD per shape | TBD | TBD | TBD |
| M2 | Parallel top-10 `mm_ids` compaction | TBD | MoE routing (99/180/634 ms deltas) | top-10/512-expert id build at chunk 512/1024/2048 | TBD | TBD | TBD |
| M3 | Routed-compact descriptor MMQ (Q6_K J32 / Q8_0 J48) | TBD (expected no) | Routed MoE operation owner | TBD | TBD | TBD | TBD |
| M4 | J48/J64 MoE-id dispatch (Q8_0/IQ2_XS/IQ3_XXS/IQ4_XS) | TBD (expected no; yes on IQ4_XS arm) | Routed MoE operation owner | TBD | TBD | TBD | TBD |
| M5 | Fused weighted top-10 sum + shared mul-add-residual | TBD | Elementwise/materialization; GR read/tail | 10×2560 weighted combine per token-batch | TBD | TBD | TBD |
| M6 | 32-warp GDN + tile-16 state-in-register prefill | TBD | GDN mixer (655 ms/1.233 s/4.983 s; 2.670 ms live-513) | S_v=128, H=32 prefill; H∈{32,48,64} decode | TBD | TBD | TBD |
| M7 | Q8_0-KV decode FA GQA opt | TBD (BF16 baseline: no) | QSA/FA decode owners | Q8_0-KV diagnostic arm only | TBD | TBD | TBD |
| M8 | Elementwise/recurrent specializations (concat/mul/quantize/norm/sumrows) | TBD | Elementwise/materialization; GR read/tail | TBD per kernel | TBD | TBD | TBD |

Every confirmed hipEngine-side deficit becomes a main-campaign candidate with
a full `W/C/O/s` row, an exclusive owner, a RED oracle, and a registered
strict fallback before implementation, per the main campaign's sections 5.1
and 5.2. Source ports cite halo-box path + commit `a7ad7b7f` and run
`scripts/check_lineage.py`.

## 6. Punchlist

| Phase | Unit | Exit condition |
| --- | --- | --- |
| HB-0 | **Done** — checked out `6c84c7d5` and `a7ad7b7f`, preserved pristine HIP Release binaries, and froze separately labeled loader-patched binaries. Pristine HB-base produced zero samples at the 1,800-second startup timeout; the two documented patches reduced startup to 24.09/21.49 seconds. Both patched lanes completed all four exact p512 categories with matching output hashes. | Identity/smoke artifact: `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb0.json`. |
| HB-1 | Canonical screen + PR-shape reproduction. Fill section 3.1/3.2 c=1 for all four lanes and section 3.4 in full; classify repeatability per lane (repeat arm on each build). | Sections 3.1/3.2 c=1 and 3.4 complete; artifact + worklog committed. |
| HB-2 | Role-resolved profiling of HB-base and HB-PR11 on UD-Q4_K_XL; kernel-name census confirms/denies each section 1.1 activation prediction; aligned family ledger regenerated against HB-PR11. | "Active on Q4_K_XL?" column fully measured; family ledger artifact committed. |
| HB-3 | Microbenchmark match ledger. For every active family, pair the HB-PR11 kernel/config against the hipEngine owner at identical shapes with counterbalanced pairs; record per-family ratios and rank by absolute delta against the main campaign's Amdahl owners. | Section 5 table complete; candidate list handed to the main campaign impact queue. |
| HB-4 | IQ4_XS diagnostic arm. Download/pin UD-IQ4_XS as a separate artifact (hashes recorded); exercise M4/J48/J64 and, under Q8_0 K/V, M7; document which mechanisms only exist on that payload. | Diagnostic tables filled; explicit "different quant, does not bind" label on every row. |
| HB-5 | Concurrency extension. Fill c=2..8 in sections 3.1–3.3 for every lane that supports it; record `unsupported` honestly; keep thermal windows shared across lanes. | Topline matrix complete or explicitly partial; artifact committed. |
| HB-6 | Port decisions. For each confirmed deficit: admit (with `W/C/O/s`), defer (named blocker), or reject (measured loss). Update the main campaign's section 4 row and mechanism transfer audit to point at measured rows instead of this doc's hypotheses. | Main campaign cross-references updated; this doc's status line advanced. |

## 7. Standing risks

- **Baseline drift:** halo-box base `6c84c7d5` is older than pinned upstream
  `f1793c1c4`. If HB-1 shows HB-base slower than patched upstream on
  UD-Q4_K_XL, the PR's local gain may partially reproduce already-merged
  upstream work; attribute gains against HB-base only, never against upstream.
- **Quant-dependent mechanisms:** M3/M4 likely do not activate on UD-Q4_K_XL.
  If HB-2 confirms that, the reproducible PR gain on our payload isolates to
  M1/M2/M5/M6/M8 — record that split explicitly rather than chasing the
  author's headline.
- **Concurrency is a new axis** for the llama lanes (`-np`, continuous
  batching, KV pressure at 111 GB): record per-slot memory and any
  prompt-cache interference instead of assuming c=1 behavior scales.
- **Author evidence gaps:** power not recorded, no repeat-arm across
  separately built binaries beyond the logit comparison, one prompt shape.
  Our repeatability classification applies to halo-box lanes exactly as it did
  to Nathan and EngramHalo.
