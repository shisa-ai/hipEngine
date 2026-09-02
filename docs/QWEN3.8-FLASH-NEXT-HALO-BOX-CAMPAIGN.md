# Qwen3.8-Flash-Next halo-box Follow-up Campaign

Status: **HB-0 through HB-2 complete and HB-3 blocked 2026-09-02 on the
approved UD-Q4_K_XL/BF16 c=1 scope; stop for review.** Frozen traces confirm
M1, M2, M5 weighted-sum, M6, and M8 activation; M3, M4, M5 shared-mul-add, and
M7 are inactive on this payload/graph. HB-3's pinned test harnesses do not expose
identical operation-complete boundaries against shipped hipEngine owners, so no
cross-engine microbenchmark ratio or candidate is valid yet. HB-PR11 improves
retained-arm prefill over HB-base by 10.12%/11.42%/17.44% at
p512/p1024/p4096; short-shape magnitude remains
provisional because HB-base drifted between arms. The IQ4_XS PR-shape diagnostic
was not run because this campaign scope forbids that download. Values attributed
to the PR author remain author-reported.
**HB-3 review outcome (2026-09-02): prefill is the binding deficit.** The
review admitted the section 5.2 hipEngine prefill gap ledger and approved the
PF-1…PF-5 prefill closure queue in section 6 as the next executable work;
HB-4 (IQ4_XS diagnostic) and HB-5 (concurrency) are deferred behind it, and
HB-6 records each PF decision as it lands. The binding comparator set and the
main campaign's section-6 closure rules remain
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
- Post-review in-tree PF units (section 6) are hipEngine kernel work admitted
  under the main campaign's sections 5.1/5.2 rules; the halo-box trees remain
  read-only references (`a7ad7b7f` citations only, `scripts/check_lineage.py`
  on every port).

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

Cells list p512 / p1024 / p4096 weighted tok/s from retained arm B. Arm A is a
stabilization diagnostic and remains in the HB-1 artifact.

| c | hipEngine | upstream HIP | HB-base | HB-PR11 |
| ---: | --- | --- | --- | --- |
| 1 | 83.366 / 82.908 / 69.200 | 235.890 / 306.510 / 283.734 | 223.893 / 308.278 / 301.621 | **246.546 / 343.485 / 354.209** |
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD | TBD |
| 6 | TBD | TBD | TBD | TBD |
| 7 | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD |

### 3.2 Decode tok/s, UD-Q4_K_XL, BF16 K/V (tg128 after p512/p1024/p4096)

Cells list p512 / p1024 / p4096 weighted tok/s from retained arm B.

| c | hipEngine | upstream HIP | HB-base | HB-PR11 |
| ---: | --- | --- | --- | --- |
| 1 | 14.315 / 14.268 / 12.177 | 17.745 / 16.987 / 14.893 | 17.663 / 16.882 / 14.897 | **18.125 / 17.284 / 15.232** |
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

Diagnostic artifact identity (acquired 2026-09-03 ahead of scope approval; the
arm remains deferred per HB-3 review):
`/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-IQ4_XS/UD-IQ4_XS/`, repo
`unsloth/Qwen3.8-Flash-Next-GGUF` revision
`38bb39ee97821de2c9009abb7e93950eec396e66`, three shards totaling
93,682,584,224 bytes, SHA-256-verified against the HuggingFace LFS oids
(`5ce89370…`, `577a38a2…`, `d4634e6d…`; manifest pair beside the shards). The
llama.cpp model argument is `-00001-of-00003`.

| Depth | HB-base published | HB-base local | HB-PR11 published | HB-PR11 local | Local gain |
| ---: | ---: | --- | ---: | --- | --- |
| 0 | 441.37 ± 1.85 | not run — IQ4_XS excluded | 639.03 ± 4.84 | not run — IQ4_XS excluded | n/a |
| 12000 | 323.14 ± 1.14 | not run — IQ4_XS excluded | 405.70 ± 1.71 | not run — IQ4_XS excluded | n/a |
| 32000 | 235.37 ± 1.01 | not run — IQ4_XS excluded | 271.34 ± 0.96 | not run — IQ4_XS excluded | n/a |
| 64000 | 167.44 ± 1.49 | not run — IQ4_XS excluded | 189.98 ± 0.88 | not run — IQ4_XS excluded | n/a |

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
| M1 | MMQ tile retune Q4_K/Q5_K/Q6_K/Q8_0 256→128 | **Yes, partial** — Q4_K/Q8_0/Q5_K launch geometry changed; Q5_1 did not | Selected Q4 gate/up, selected Q5_1 down, dense Q8, dense projection compute | Qwen4Exp tensor dimensions, rows 512/1024/2048; complete quantize+matmul+epilogue | blocked — no shared packed-weight fixture | blocked — F32 llama vs owner-specific hipEngine boundaries | **Blocked before timing** |
| M2 | Parallel top-10 `mm_ids` compaction | **Yes, prefill only** — 94/94/188 launches at p512/p1024/p4096; decode stays below the 64-token gate | MoE routing (99/180/634 ms deltas) | 512 experts × top-10 × rows 512/1024/2048; counts+prefix+forward/inverse maps | blocked — helper is internal to `MUL_MAT_ID` | blocked — no common four-output map ABI | **Blocked before timing** |
| M3 | Routed-compact descriptor MMQ (Q6_K J32 / Q8_0 J48) | **No** — zero builder or routed-compact MMQ symbols in all six PR11 windows | Routed MoE operation owner | n/a on binding payload | n/a | n/a | Inactive; no HB-3 row |
| M4 | J48/J64 MoE-id dispatch (Q8_0/IQ2_XS/IQ3_XXS/IQ4_XS) | **No** — zero J48/J64 specialization symbols; expert payload is Q4_K/Q5_1 | Routed MoE operation owner | n/a on binding payload | n/a | n/a | Inactive; IQ4_XS remains excluded |
| M5 | Fused weighted top-10 sum + shared mul-add-residual | **Partial** — weighted `10×2560` kernel active; `shared_mul_add_f32` absent | Elementwise/materialization; GR read/tail | F32 `[T,10,2560]` + weights → F32 `[T,2560]`, T=1/512/1024/2048 | blocked — base catalog lacks the PR case | blocked — shipped hipEngine owner consumes BF16 or fuses extra work | **Blocked before timing** |
| M6 | 32-warp GDN + tile-16 state-in-register prefill | **Yes** — workgroup Y changes 4→32; core p4096 prefill kernel sum 2,139.377→647.976 ms | GDN mixer (655 ms/1.233 s/4.983 s; 2.670 ms live-513) | Hk=16, Hv=32, Dk=Dv=128, T=1/512/1024/2048; normalized Q/K through core+state | blocked — catalog geometry/boundary differs | blocked — state layout and prepare/post boundary differ | **Blocked before timing** |
| M7 | Q8_0-KV decode FA GQA opt | **No by contract** — BF16 K/V; zero Q8-KV/GQA-opt symbols | QSA/FA decode owners | n/a under BF16 K/V | n/a | n/a | Inactive; Q8_0 K/V excluded |
| M8 | Elementwise/recurrent specializations (concat/mul/quantize/norm/sumrows) | **Yes, partial** — transposed concat, contiguous binary, dual L2 norm, four-column Q8 matvec, RMS128, and sum-4 symbols present | Elementwise/materialization; GR read/tail | Six separate operation-complete rows; four-column Q8 deduplicated from M1 | blocked — PR symbols are internal graph selections | blocked — no six shared stride/dtype/output fixtures | **Blocked before timing** |

### 5.1 HB-3 readiness outcome

The pinned `test-backend-ops` binaries build and pass the PR11 cases they expose,
but they do not provide a valid common denominator. HB-base contains no
`WEIGHTED_EXPERT_SUM(2560,10,32)` case; neither catalog contains the actual Q4_K
512-expert/top-10 model case; and the available GDN graph uses a different
head/repeat, normalization, state-layout, and post-gate boundary from the shipped
hipEngine owner. M2 is an internal `MUL_MAT_ID` helper rather than a standalone
operation, while M8 expands into six distinct operations.

For every blocked active row, `C` is therefore unmeasured, `O=null`, and
`s=null`. Arithmetic class is `blocked/unclassified` until a common fixture fixes
each engine's dtype/layout representation and permits a T0–T3 assignment. The
unchanged registered strict/unfused hipEngine owner is the fallback. Exclusions are HB-2 overlapping kernel sums,
HB-1 whole-engine rates, different operation boundaries, IQ4_XS, and Q8_0 K/V.
The exact per-row `W/C/O/s`, required fixture/ABI, harness hashes, commands, and
correctness probes are in
`benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb3-blocked.json`.
No row enters the main-campaign candidate queue.

Every confirmed hipEngine-side deficit becomes a main-campaign candidate with
a full `W/C/O/s` row, an exclusive owner, a RED oracle, and a registered
strict fallback before implementation, per the main campaign's sections 5.1
and 5.2. Source ports cite halo-box path + commit `a7ad7b7f` and run
`scripts/check_lineage.py`.

### 5.2 hipEngine prefill gap ledger (HB-3 review input)

The whole-engine prefill gap that motivates this campaign is already fully
attributed on the binding payload by the
[`2026-09-01 canonical impact profile`](../benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-canonical-impact-profile.json):
ROCTX-scoped hipEngine prefill device time with 100% role coverage at
p512/p1024/p4096, aligned family-by-family against rocprofv3 device traces of
pinned upstream llama.cpp `f1793c1c4` running the same exact canonical token
fixtures on the same host. hipEngine prefill commits `3ddb748d`/`6e32efcf`;
HB-1's later production lane (manifest `37d59564…`) measured 83.366/82.908/69.200
pp tok/s versus the profile packet's unprofiled 85.711 pp at p1024, so the
ledger still ranks current production within ~3% and stays the ranking basis
until a retained prefill unit lands (protocol item 5).

Device-time totals per profiled request (hipEngine vs llama.cpp, ms):

| Shape | hipEngine | llama.cpp | Gap | Ratio |
| --- | ---: | ---: | ---: | ---: |
| p512 | 5,965.1 | 1,911.1 | 4,054.0 | 3.12x |
| p1024 | 11,183.7 | 2,971.9 | 8,211.8 | 3.76x |
| p4096 | 54,558.9 | 10,802.0 | 43,756.8 | 5.05x |

Per-family rows (device ms; llama.cpp advantage in parentheses; rows grouped
by workstream, largest deficits first):

| Family | p512 | p1024 | p4096 | PR11 mechanism (HB-2) | Workstream |
| --- | ---: | ---: | ---: | --- | --- |
| Dense projection compute | 1,181.3 vs 50.4 (23.4x) | 2,173.3 vs 94.6 (23.0x) | 8,767.0 vs 337.7 (26.0x) | M1 Q8_0 MMQ tile retune (active) | **PF-1** |
| QSA attention | 51.0 vs 12.9 (3.9x) | 180.1 vs 48.8 (3.7x) | 10,229.2 vs 526.0 (19.5x) | none under BF16 K/V (M7 inactive) | **PF-2** (native) |
| MoE gate/up Q4_K | 1,368.5 vs 700.3 (2.0x) | 2,528.6 vs 851.7 (3.0x) | 10,233.3 vs 1,906.5 (5.4x) | M1 Q4_K tile retune (active) | **PF-3** |
| MoE down Q5_1 | 1,173.9 vs 501.4 (2.3x) | 2,192.8 vs 625.0 (3.5x) | 8,782.4 vs 1,454.1 (6.0x) | none measured (M1 Q5_1 did not retune) | **PF-3** (native) |
| Dense Q8 quantize | 957.4 vs 269.0 (3.6x) | 1,802.3 vs 451.2 (4.0x) | 7,241.7 vs 1,592.1 (4.6x) | M8 quantize/elementwise paths (active, partial) | **PF-1** |
| GDN | 655.2 vs 83.3 (7.9x) | 1,233.0 vs 222.5 (5.5x) | 4,982.8 vs 1,845.3 (2.7x) | M6 32-warp + tile-16 (active; their p4096 kernel sum 2,139.4→648.0 ms) | **PF-5** |
| MoE routing | 120.2 vs 21.6 (5.6x) | 225.0 vs 45.3 (5.0x) | 903.0 vs 269.5 (3.4x) | M2 parallel top-10 compaction (active, prefill) | **PF-4** |
| Elementwise/materialization | 457.6 vs 272.2 (1.7x) | 848.5 vs 632.9 (1.3x) | 3,419.5 vs 2,870.8 (1.2x) | M5 weighted top-10 sum (active) + M8 set (partial) | **PF-4** |

Reading rules and caveats, binding on any use of these rows:

- This is a device-time diagnostic alignment on identical fixtures, not a
  single merged wall ratio; family boundaries are per-engine role
  attributions (`performance_claim=false`, status
  `accepted_diagnostic_profile_with_external_profiler_teardown_caveat`).
- llama.cpp family sums come from the pinned `f1793c1c4` patched-HIP lane,
  which HB-1 shows is slower than HB-PR11; closing to these rows therefore
  understates, not overstates, the remaining external gap.
- The QSA row grows superlinearly with context (3.7x → 19.5x): chunked-512
  prefill re-attends history without the reuse the retained ordered-QSA
  decode route introduced for decode. PF-2 is the only workstream with no
  external reference mechanism under BF16 K/V; it resumes the main
  campaign's P4 QSA prefill subowner evidence
  ([`p4-qsa-prefill-subowner`](../benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p4-qsa-prefill-subowner.json),
  [`dense-other subowners`](../benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-dense-other-subowners.json)).
- HB-3's matching blocker stands: these are role-aligned whole-request rows,
  not valid cross-engine microbenchmark ratios. PF units promote on in-tree
  whole-model same-session A/B plus their RED gates, using HB-2 only as
  mechanism-existence evidence.

## 6. Punchlist

| Phase | Unit | Exit condition |
| --- | --- | --- |
| HB-0 | **Done** — checked out `6c84c7d5` and `a7ad7b7f`, preserved pristine HIP Release binaries, and froze separately labeled loader-patched binaries. Pristine HB-base produced zero samples at the 1,800-second startup timeout; the two documented patches reduced startup to 24.09/21.49 seconds. Both patched lanes completed all four exact p512 categories with matching output hashes. | Identity/smoke artifact: `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb0.json`. |
| HB-1 | **Done** — completed two exact 36-sample arms for each of HB-base, HB-PR11, current hipEngine, and freshly rebuilt patched upstream. Retained arm B shows HB-PR11/base prefill gains of 1.1012x/1.1142x/1.1744x and decode gains of 1.0261x/1.0238x/1.0225x at p512/p1024/p4096. Every lane is cross-arm output-exact. HB-base p512/p1024 prefill drift makes those magnitudes provisional; p4096 direction reproduces. Section 3.4 is explicitly not run because IQ4_XS is excluded by the approved scope. | Artifact: `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb1.json`; proceed to frozen-binary HB-2 profiling. |
| HB-2 | **Done** — profiled frozen HB-base and HB-PR11 over exact p512/p1024/p4096 prefill and live-513/1025/4097 decode. All six pairs are output-exact across lanes and cached decode evaluates one appended token. Kernel-name and launch-geometry census confirms M1, M2, M5 weighted-sum, M6, and M8; denies M3, M4, M5 shared-mul-add, and M7 on this binding graph/payload. The aligned broad-family ledger is diagnostic because kernel sums can overlap. | Artifact: `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb2.json`; HB-3 admits only measured-active subfamilies. |
| HB-3 | **Blocked** — completed the shape/ABI readiness ledger and built pinned base/PR11 `test-backend-ops` binaries (`7af38994…` / `bc640014…`). Available cases pass their own correctness probes, but no active family has an identical operation-complete cross-engine fixture: M1 lacks shared packed weights/dtypes, M2 is internal to `MUL_MAT_ID`, M5 lacks a common F32/BF16 boundary, M6 differs in heads/state/prepare-post ownership, and M8 must split into six stride-aware operations. Timing current surfaces would violate the matching rule. | Block artifact: `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb3-blocked.json`. Stop for review; no candidate handed off. |
| HB-4 | **Deferred behind PF-1…PF-5 by the HB-3 review** — IQ4_XS diagnostic arm. Artifact already downloaded and hash-verified (section 3.4); when admitted, exercise M4/J48/J64 and, under Q8_0 K/V, M7; document which mechanisms only exist on that payload. | Diagnostic tables filled; explicit "different quant, does not bind" label on every row. |
| HB-5 | **Deferred behind PF-1…PF-5 by the HB-3 review** — concurrency extension. Fill c=2..8 in sections 3.1–3.3 for every lane that supports it; record `unsupported` honestly; keep thermal windows shared across lanes. | Topline matrix complete or explicitly partial; artifact committed. |
| HB-6 | Port decisions. For each PF unit (and each confirmed deficit): admit (with `W/C/O/s`), defer (named blocker), or reject (measured loss). Update the main campaign's section 4 row and mechanism transfer audit to point at measured rows instead of this doc's hypotheses. | Main campaign cross-references updated; this doc's status line advanced. |
| PF-1 | **Approved at HB-3 review** — Dense projection + dense Q8 MMQ schedule (largest short/mid-shape delta; flat 23–26x ratio says schedule, not tuning; M1 evidence). Port the 128-wide RDNA3.5 tile geometry for Q8_0 dense matmuls and the M8 quantize paths behind a registry variant; RED exact-parity vs the registered strict unfused fallback. | `W/C/O/s` admitted in main campaign; exact-parity RED green; rocprofv3 expected-kernel trace; same-session whole-model A/B prefill improvement at p512/p1024 (p4096 recorded); artifact + rollup + worklog committed. |
| PF-2 | **Approved at HB-3 review** — QSA long-context prefill reuse (largest single p4096 delta, 19.5x; no external mechanism under BF16 K/V). Resume the P4 ordered-QSA prefill subowner line: extend the retained ordered three-pass decode reuse to chunked prefill without partial-softmax reassociation. | Same admission gates as PF-1 plus the exact ordered-QSA arithmetic contract; p4096 QSA role reduction measured in a refreshed role ledger; whole-model p4096 prefill A/B. |
| PF-3 | **Approved at HB-3 review** — MoE expert GEMM schedules: Q4_K gate/up via the M1 tile retune; Q5_1 down via native tile work (no PR11 mechanism; HB-2 confirmed Q5_1 did not retune). Median 325–333/512 active experts at ~7 rows each is the binding shape. | Same admission gates as PF-1 plus expert-count invariance on the canonical fixture; whole-model A/B at all three shapes. |
| PF-4 | **Approved at HB-3 review** — MoE routing + materialization: port M2 parallel top-10 compaction (prefill arm) and the M5 fused weighted top-10 expert sum; fold the active M8 elementwise specializations where the boundaries match hipEngine owners. | Same admission gates as PF-1; routing+materialization combined role reduction measured; whole-model A/B. |
| PF-5 | **Approved at HB-3 review** — GDN port of M6: 32 warps/block for S_v=128 and the token-tile-16 state-in-register prefill kernel, adapted to the hipEngine GDN owner's state layout and prepare/post boundary (HB-3 documented the mismatch). | Same admission gates as PF-1; exact-parity RED vs strict fallback; p4096 GDN role reduction against the 3.3x PR11 kernel-level evidence; whole-model A/B. |

### 6.1 PF-1 execution checklist

Sub-units of the PF-1 row. Status markers: **Done**, **Blocked** (with named
prerequisite), or open. The dense-family re-rank from the
[`dense-other subowner audit`](../benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-dense-other-subowners.json)
orders the work; layer-2 Q5_K WMMA stays rejected on the production-numerics
gate and is out of PF-1.

| Sub-unit | Scope | Exit condition | Status |
| --- | --- | --- | --- |
| PF-1a | Scope and lineage: map the `dense_projection_compute` and `dense_quant_q8` role families to exact in-tree kernels and registry keys; run `scripts/check_lineage.py`; record the halo-box `a7ad7b7f` source files and tile geometry to port. | Mapping table committed to this doc's PF-1 worklog entry; lineage check green; no code change. | **Done** — mapping in [PF-1a entry](../worklog/entries/20260902T233138.374147Z-lhl-qwen4exp-pf1a-dense-scope-199383.md): `dense_quant_q8` = `q8_0_raw_mmq128_q8_1_d4_kernel` (6,926 ms/req, one block per output column) + its quantize kernel in `gguf_q8_0_mmq_prefill.hip`; `dense_projection_compute` = `gguf_k_prefill_out_coltile_rowbatch_kernel<float,float,8,8,4>` + selected quant8 down in `gguf_k_gemv.hip`; halo-box pinned at `a7ad7b7f` with I=64/128-row SRAM tile geometry; layer-2 Q5_K stays out. |
| PF-1b | RED oracle: exact-token fixtures plus strict-parity tests for the dense F32 projection (`gguf_k_prefill_out_coltile_rowbatch<float,float,8,8,4>` family) and the dense Q8 activation-quantize + selected Q8_0 down path, green on the current path before any edit. | New tests pass on current kernels; HIP-availability guarded; oracle identity recorded. | **Done** — `tests/test_qwen4exp_pf1_dense_parity.py` (19 tests, green on unmodified kernels): policy-shape MMQ chain determinism + bounded-vs-exact-owner envelope + top-1 at all 7 production shapes; coltile strict bit-parity across 3 variants at 4 attention shapes; selected Q8_0 down vs CPU GEMV oracle at top-10 gather ABI. PF-1c/d variants enter via `MMQ_CHAIN_VARIANTS`/`COLTILE_VARIANTS`/`SELECTED_VARIANTS`. |
| PF-1c | Dense F32 projection schedule: rebalanced tile/row-batch geometry for the 48 attention-gate roles behind a registry variant; exact same-output contract (no reassociation) unless a production-profile gate is later declared and passed. | Exact-parity RED green on the variant; rocprofv3 expected-kernel trace; kernel-family device-time reduction measured in isolation. | Open |
| PF-1d | Dense Q8 quantize + Q8_0 down MMQ geometry: port the RDNA3.5 128-wide tile shape from `a7ad7b7f` (M1) and the M8 quantize specializations behind registry variants with strict fallbacks. | Same gates as PF-1c per variant. | Open |
| PF-1e | Closure: same-session whole-model A/B (p512/p1024 primary, p4096 recorded) against the frozen production denominator; artifact under `benchmarks/results/`, rollup + changelog, HB-6 admit/defer/reject decision, campaign doc rows advanced. | All gates recorded; retained-or-blocked decision committed. | Open |

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
