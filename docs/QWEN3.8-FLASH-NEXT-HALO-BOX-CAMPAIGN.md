# Qwen3.8-Flash-Next halo-box Follow-up Campaign

Status: **HB-0 through HB-3 complete/blocked; PF-1 dense-Q8 arithmetic levers
closed 2026-09-03 with no retained change; PF-0 natural-text route-covering
fixture landed 2026-09-03 — the next executable unit is the incumbent-vs-strict
capture on it to resolve section 6.4 D1.**
Frozen traces confirm M1, M2, M5 weighted-sum, M6, and M8 activation; M3, M4,
M5 shared-mul-add, and M7 are inactive on this payload/graph. HB-3's pinned
test harnesses do not expose identical operation-complete boundaries against
shipped hipEngine owners, so no cross-engine microbenchmark ratio is valid.
HB-PR11 improves retained-arm prefill over HB-base by 10.12%/11.42%/17.44% at
p512/p1024/p4096; short-shape magnitude remains provisional because HB-base
drifted between arms. The IQ4_XS PR-shape diagnostic was not run because this
campaign scope forbids that download. Values attributed to the PR author remain
author-reported.

**HB-3 review outcome (2026-09-02): prefill is the binding deficit.** The
review admitted the section 5.2 hipEngine prefill gap ledger and approved the
PF-1…PF-5 prefill closure queue in section 6 as the next executable work;
HB-4 (IQ4_XS diagnostic) and HB-5 (concurrency) are deferred behind it, and
HB-6 records each PF decision as it lands.

**PF-1 outcome (2026-09-03): every arithmetic-changing dense-Q8 lever is
closed, and the production-numerics gate has a fixture problem that blocks the
next one.** Tile retiles were bit-exact but slower; the plane and policy-shape
levers changed arithmetic and failed the `EXECUTION-PROFILES.md`
section 6.1 envelope. Investigating
that failure surfaced a defect in the *evidence*, not the standard: the
2026-08-29 admission packet dispatched the route it admitted in 50 of 450 rows
(11.1%), and the only route-covering fixture available is synthetic repeated
material (104 unique tokens in 512). Section 6.2 freezes the resulting
admissibility rules, section 6.3 routes future levers by arithmetic class, and
section 6.4 lists the two owner decisions that are still open. **PF-0 (build a
natural-text route-covering fixture) is the prerequisite for every remaining
arithmetic-changing candidate in this campaign.**

The binding comparator set and the main campaign's section-6 closure rules
remain owned by
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

### Start here (2026-09-03)

The goal is unchanged: close the section 5.2 prefill gap against the external
engines. What changed is that the cheap arithmetic dials on the dense-Q8 family
are spent, so the next wins come from **exact restructuring** and from
**workstreams that have never been attempted**, not from more precision
trading.

Ordered queue for the next coder. Do not skip PF-0 if your lever changes
arithmetic; you may start any `T0` lever immediately in parallel.

| # | Do this | Why now | Gate cost |
| ---: | --- | --- | --- |
| 1 | **PF-0** — build a natural-text route-covering fixture. | Every remaining arithmetic-changing candidate in this campaign is unadjudicable without it (section 6.2 Rule B + section 6.4 D1). Blocks PF-1c lever 2, PF-3, PF-4, and any PF-5 reassociation. | One fixture unit, no GPU measurement claim. |
| 2 | **PF-2** — QSA long-context prefill reuse. | Largest single p4096 deficit (10,229 vs 526 ms, 19.5x) and the only workstream with no external mechanism to port, so it is pure hipEngine headroom. Ordered three-pass reuse is expected `T0`. | Exact-parity RED + trace. No numerics packet if it stays `T0`. |
| 3 | **PF-5** — GDN M6 port (32 warps/block, tile-16 state-in-register). | 4,982.8 vs 1,845.3 ms at p4096 with direct external evidence that the mechanism works (their kernel sum 2,139.4 → 648.0 ms). Launch-geometry and state-residency changes are `T0` if the reduction order is preserved. | Exact-parity RED + trace. |
| 4 | **PF-3** — MoE Q5_1 down + Q4_K gate/up schedules. | Second-largest combined deficit (Q5_1 down 6.0x, Q4_K gate/up 5.4x at p4096); Q5_1 has no PR11 mechanism, so it is native tile work. | `T0` if exact; otherwise PF-0 first. |
| 5 | **PF-4** — MoE routing + materialization. | Smaller absolute deficit but M2/M5 are confirmed-active external mechanisms with clear shapes. | `T0` for the routing compaction (integer maps); the fused weighted sum needs PF-0 if it reassociates. |
| 6 | **PF-1 remainder** — see section 6.3, fork (b). | Still the largest p512/p1024 deficit (23–26x), but the remaining lever is either a coverage-complete numerics packet or a new bit-exact dense kernel. Both are expensive; the four units above are cheaper per unit of gap closed. | High. |

**The one thing not to do:** do not reach for another activation-precision dial
on the dense-Q8 chain. Three were measured and closed (section 6.1 PF-1d); the
family is exhausted and the 23x deficit it targets is a schedule problem, not a
precision problem.

| Phase | Unit | Exit condition |
| --- | --- | --- |
| HB-0 | **Done** — checked out `6c84c7d5` and `a7ad7b7f`, preserved pristine HIP Release binaries, and froze separately labeled loader-patched binaries. Pristine HB-base produced zero samples at the 1,800-second startup timeout; the two documented patches reduced startup to 24.09/21.49 seconds. Both patched lanes completed all four exact p512 categories with matching output hashes. | Identity/smoke artifact: `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb0.json`. |
| HB-1 | **Done** — completed two exact 36-sample arms for each of HB-base, HB-PR11, current hipEngine, and freshly rebuilt patched upstream. Retained arm B shows HB-PR11/base prefill gains of 1.1012x/1.1142x/1.1744x and decode gains of 1.0261x/1.0238x/1.0225x at p512/p1024/p4096. Every lane is cross-arm output-exact. HB-base p512/p1024 prefill drift makes those magnitudes provisional; p4096 direction reproduces. Section 3.4 is explicitly not run because IQ4_XS is excluded by the approved scope. | Artifact: `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb1.json`; proceed to frozen-binary HB-2 profiling. |
| HB-2 | **Done** — profiled frozen HB-base and HB-PR11 over exact p512/p1024/p4096 prefill and live-513/1025/4097 decode. All six pairs are output-exact across lanes and cached decode evaluates one appended token. Kernel-name and launch-geometry census confirms M1, M2, M5 weighted-sum, M6, and M8; denies M3, M4, M5 shared-mul-add, and M7 on this binding graph/payload. The aligned broad-family ledger is diagnostic because kernel sums can overlap. | Artifact: `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb2.json`; HB-3 admits only measured-active subfamilies. |
| HB-3 | **Blocked** — completed the shape/ABI readiness ledger and built pinned base/PR11 `test-backend-ops` binaries (`7af38994…` / `bc640014…`). Available cases pass their own correctness probes, but no active family has an identical operation-complete cross-engine fixture: M1 lacks shared packed weights/dtypes, M2 is internal to `MUL_MAT_ID`, M5 lacks a common F32/BF16 boundary, M6 differs in heads/state/prepare-post ownership, and M8 must split into six stride-aware operations. Timing current surfaces would violate the matching rule. | Block artifact: `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb3-blocked.json`. Stop for review; no candidate handed off. |
| HB-4 | **Deferred behind PF-1…PF-5 by the HB-3 review** — IQ4_XS diagnostic arm. Artifact already downloaded and hash-verified (section 3.4); when admitted, exercise M4/J48/J64 and, under Q8_0 K/V, M7; document which mechanisms only exist on that payload. | Diagnostic tables filled; explicit "different quant, does not bind" label on every row. |
| HB-5 | **Deferred behind PF-1…PF-5 by the HB-3 review** — concurrency extension. Fill c=2..8 in sections 3.1–3.3 for every lane that supports it; record `unsupported` honestly; keep thermal windows shared across lanes. | Topline matrix complete or explicitly partial; artifact committed. |
| HB-6 | Port decisions. For each PF unit (and each confirmed deficit): admit (with `W/C/O/s`), defer (named blocker), or reject (measured loss). Update the main campaign's section 4 row and mechanism transfer audit to point at measured rows instead of this doc's hypotheses. | Main campaign cross-references updated; this doc's status line advanced. |
| PF-0 | **Done 2026-09-03** — built a natural-text route-covering numerics fixture `benchmarks/fixtures/qwen4exp_natural_ar_pf0.json` (12 cases, three per canonical category, 535–877 prompt tokens, per-case token-ID and source SHA-256 hashes, provenance in `benchmarks/fixtures/natural_sources/`). Every case clears the `rows >= 64` Q8 MMQ policy, giving 100% route-engagement coverage (1548 compared rows) recorded by `scripts/qwen4exp_route_coverage_finding.py --single-fixture`; unique-token ratio recorded per case; no measurement claim. | Neither existing fixture qualifies: the 18-prompt admission suite is natural but 39-71 tokens (11.1% route coverage against the `rows >= 64` Q8 MMQ policy), and `qwen4exp_admission_suite_18prompt.json` records that; the canonical `p512/p1024/p4096` fixture is 100% route-covering but synthetic (`code-p512` holds 104 unique tokens in 512, built by repeating four short prompts). Natural long-form material is being sourced per category at >=512 tokens, keeping the four canonical categories, with per-case token-ID hashes. | Fixture committed with construction provenance and suite hashes; route-engagement coverage >= 50% recorded by `scripts/qwen4exp_route_coverage_finding.py`; unique-token ratio recorded per case; no measurement claim. Then re-run the incumbent-vs-strict capture on it to resolve section 6.4 D1. |
| PF-1 | **Arithmetic levers closed 2026-09-03; remainder is section 6.3 fork (b).** Dense projection + dense Q8 MMQ schedule is still the largest short/mid-shape delta (23–26x), but all three arithmetic-changing levers were measured and rejected (6.1 PF-1c/PF-1d) and the `T0` tile retiles lost on speed. The remaining lever is a **bit-exact** faster dense kernel for the coltile-served shapes, or a coverage-complete numerics packet after PF-0. No retained change; d4x3 chain and current policy shapes remain production. | Either a `T0` dense kernel meeting the section 6.5 definition of done, or a PF-0-based section 6.2 packet that admits a `T1/T2` candidate. Do not propose another activation-precision dial. |
| PF-2 | **Rejected 2026-09-03 — measured loss, lever closed.** The ordered three-pass prefill kernels (`strict_ordered_three_pass_rows_spans`, bit-exact vs the strict rows owner at kernel level, traced via rocprofv3) were wired behind the default-off `HIPENGINE_QWEN4_EXP_QSA_ORDERED_PREFILL` flag and evaluated with a same-session counterbalanced whole-model A/B (1 warmup + 3 measured reps per case, canonical p512/p1024/p4096 fixture, 12 cases x 4 categories via `scripts/qwen4exp_candidate_ar.py`). Result: the candidate is **slower at every shape** (p512 86.19→83.38 tok/s, 0.967x; p1024 85.56→83.22, 0.973x; p4096 70.93→68.67, 0.968x) and p4096 outputs are **not bit-identical** to the incumbent production route, which dispatches the wave32 variant (the ordered variant is bit-exact only vs strict rows, not vs wave32). The extra launches of the three-pass split outweigh any reuse benefit. Per the `docs/REFACTOR.md` removal trigger, the env flag and runner branch were deleted; the ordered rows kernels remain registered as a strict variant with their exact-parity RED tests. No promotion, incumbent (wave32/dense) route stays production. | Artifact: [`pf2-qsa-ordered-prefill-ab`](../benchmarks/results/2026-09-03-gfx1151-qwen38-flash-next-pf2-qsa-ordered-prefill-ab.json); the refreshed p4096 QSA role ledger is moot given the whole-model measured loss. |
| PF-3 | **Done** 2026-09-04 — MoE expert GEMM schedules. Production widths corrected first (ffn=640, hidden=2560, 512 experts, top-k 10 from GGUF metadata + tensor shapes; the 7680 ffn above was wrong). Q4_K gate/up M1 (PAIR=false instantiation): **lever closed as measured loss** (+57% to +123% slower at rows 16/64/512, bit-exact; per-column global dequant loses the PAIR=true shared-metadata amortization). Q5_1 down M1 (fused single-loop logical256): **promoted as the production default** in the `exact_grouped_down` expertgrid64 arm — kernel-level binding-shape A/B −10.6% (9,423.0→8,425.6 µs, bit-exact), whole-model same-session counterbalanced A/B (4 arms, 12 cases) prefill **p512 +1.89%, p1024 +1.58%, p4096 +0.95%**, decode unchanged within noise, all 144 output samples bit-identical across arms; previous owner stays registered as the strict fallback; temporary A/B flag deleted. Artifacts: [`pf3-q51-m1-promotion-ab`](../benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-pf3-q51-m1-promotion-ab.json), [`pf3-moe-schedules-kernel-ab`](../benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-pf3-moe-schedules-kernel-ab.json). | Q4_K lever needs no further work; Q5_1 promoted with the full section 6.5 gate (parity RED 7/7, rocprofv3 trace, whole-model A/B at all three shapes, expert-count invariance via uneven-count/empty-expert parity coverage at 64- and 512-expert fixtures). |
| PF-4 | **Approved at HB-3 review** — MoE routing + materialization: port M2 parallel top-10 compaction (prefill arm) and the M5 fused weighted top-10 expert sum; fold the active M8 elementwise specializations where the boundaries match hipEngine owners. | Section 6.5 definition of done (`T0` for the integer routing compaction; the fused weighted sum needs PF-0 if it reassociates); routing+materialization combined role reduction measured; whole-model A/B. |
| PF-5 | **Rejected 2026-09-04 — measured loss, lever closed.** The 32-warp/block (1024-thread) prefill kernel (`hipengine_qwen4_exp_gdn_prefill_w32_f32`, a `WARPS_PER_BLOCK=32` instantiation of the templated columnwarps kernel — identical per-warp arithmetic, block composition only) was evaluated with an interleaved counterbalanced kernel-level A/B vs the production columnwarps owner (24 pairs, median-of-pairs, rows 16/64/256/1024/4096, H∈{32,48,64} covered by the parity RED suite). Result: **w32 slower at every shape ≥64** (p64 +1.6%, p256 +6.4%, p1024 +4.0%, p4096 +3.4%) and only a noise-level −1.6% at rows=16. Parity held bit-exact (0.0 diff vs colwarps) at all shapes. On this hardware (gfx1151, 32-wide wavefronts), 1024-thread blocks reduce occupancy to 1 block/CU and the 8× block-count reduction does not recover the loss; the M6 mechanism does not transfer to this host. No promotion; incumbent columnwarps <4,4> route stays production. The tile-16 state-in-register lever (lever 2) is deferred behind PF-0 per the binding T1 escalation rule in declaration entry 55a7d9 (state-residency restructure). w32 remains registered as a strict non-default variant with its exact-parity RED tests. | Artifact: [`pf5-gdn-w32-prefill-ab`](../benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-pf5-gdn-w32-prefill-ab.json); rocprofv3 trace confirms the `<32, 4>` instantiation at block=1024 with the norm-gate tail. |

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
| PF-1c | **Blocked** — lever 1 (extend the admitted MMQ128 chain to coltile-served shapes via policy rows) rejected on the canonical route-covering fixture: the P3 shape row `(2560,6144)` retest is deterministic 3/3 (original blocker confirmed stale/harness-scoped) but fails the same bars as d4x2 (incumbent-relative mean 7.27e-4 passes, median 0, p95 5.26e-3 > 5e-3, top-1 98.67% < 0.99, per-scope code/t1/t10/t14) — while keeping the incumbent's exact arithmetic; the attn_output row `(3072,2560)` is closed by measured-analogy inference (same chain/delta class, not separately measured). Lever 2 (Q8_0 selected WMMA) remains open. | **Blocked on PF-0**, not on a comparison basis. The prior "trajectory vs single-transition basis" blocker is **withdrawn as factually wrong**: both bases are the same 24-step teacher-forced trajectory, and the August harness is durably in-tree (`scripts/qwen4exp_layer2_profile_gate.py`). The real defect is route coverage — the 18-prompt admission fixture dispatches the Q8 MMQ route in only 50 of 450 rows (11.1%), so the 2026-08-29 packet is `route_vacuous_for_scope`. Evidence: [`basis correction`](../worklog/entries/20260903T034205.467427Z-lhl-qwen4exp-pf1-basis-correction-6336a4.md), [`coverage finding`](../benchmarks/results/2026-09-03-gfx1151-qwen38-flash-next-pf1-q8-mmq-route-coverage-finding.json), [`admission-suite reconciliation`](../benchmarks/results/2026-09-03-gfx1151-qwen38-flash-next-pf1-admission-suite-reconciliation.json), [`p3shape retest`](../benchmarks/results/2026-09-03-gfx1151-qwen38-flash-next-pf1c-p3shape-retest.json), [`plane2 dual-basis`](../benchmarks/results/2026-09-03-gfx1151-qwen38-flash-next-pf1d-mmq-plane2-dualbasis.json). |
| PF-1d | Dense Q8 quantize + Q8_0 down MMQ geometry: port the RDNA3.5 128-wide tile shape from `a7ad7b7f` (M1) and the M8 quantize specializations behind registry variants with strict fallbacks. | Same gates as PF-1c per variant. | **Done** (rejected with complete evidence) — tile retiles measured slower (m64x64 0.89x, m32n128 0.35x; bit-exact but weight-traffic-bound); single-plane d4 fails unit numerics (top-1 99.0-99.6%, max_abs 1.5-3.4); two-plane d4x2 (1.38x isolated) fails dual-basis admission (incumbent-relative mean 7.45e-4 passes, median 0, but p95 5.065e-3 > 5e-3 and per-scope failures on code/t1/t10; deterministic 3/3 everywhere). The d4x3 guarded chain is final for this family. Artifacts: plane2-gate, plane2-dualbasis, pf1d worklog entries. |
| PF-1e | **Blocked** — no candidate is currently admissible (PF-1c/PF-1d levers rejected on the canonical route-covering fixture; d4x3 chain and current policy shapes remain production), so there is no retained change to whole-model A/B. | Unblocks when PF-0 lands and a coverage-complete packet admits a candidate; then run the same-session counterbalanced whole-model A/B (p512/p1024 primary, p4096 recorded) and the rollup per the unit rules. |

**What PF-0 can and cannot reopen.** The PF-1c/PF-1d rejections were measured
on the canonical fixture, which is route-covering but synthetic (section 7
standing risk). PF-0 therefore *may* legitimately change the verdict on the
`d4x2` plane candidate and the P3 shape row, and re-running them is a
reasonable use of the new fixture — but only under a rule predeclared before
the run, per the 2026-08-31 accuracy review and the precedent in the
[basis predeclaration](../worklog/entries/20260903T033202.234656Z-lhl-qwen4exp-pf1-basis-predeclaration-6f6810.md).
Their numbers are now known, so choosing a rule afterwards is not available.
PF-0 does **not** reopen the `T0` tile retiles (m64x64 0.89x, m32n128 0.35x):
those were bit-exact and lost on speed, which no fixture change affects.
Single-plane `d4` also stays closed as a drift candidate for the section 6.3
`T3` reason, independent of any fixture.

### 6.2 Production-numerics packet admissibility (route coverage)

Frozen 2026-09-03 in
[`basis predeclaration`](../worklog/entries/20260903T033202.234656Z-lhl-qwen4exp-pf1-basis-predeclaration-6f6810.md)
before the confirming measurement, and binding on every PF unit:

- Every production-numerics packet records **route-engagement coverage**: the
  fraction of compared rows whose trajectory dispatched the candidate route at
  least once. Below 50% the packet is `route_vacuous_for_scope` and is binding
  evidence for **neither** admission nor rejection of that route.
- The binding comparison basis is the existing `EXECUTION-PROFILES.md`
  section 6/6.3 strict-teacher trajectory. No new bar was introduced and no
  threshold moved.
- Per-scope verdicts over fewer than 25 rows stay binding as implemented and
  are additionally annotated `underpowered_scope` for the recalibration
  campaign; section 10 forbids moving a budget without calibrated evidence.
- Single-plane `d4` is a **representation** question, not an
  implementation-drift question. `gguf_q8_0_mmq_prefill.hip:6,35,342` records
  that our `d4` block is a port of llama.cpp's `block_q8_1_mmq` D4 layout
  (one FP32 scale per 32 values), which llama.cpp selects for every
  `GGML_TYPE_Q8_0` matmul on RDNA3/3.5 (`mmq.cuh:73`, `mmq.cu:348-368`);
  `d4x2`/`d4x3` add residual passes llama.cpp does not have. If single-plane
  is revisited it goes through the `EXECUTION-PROFILES.md` section 6.2
  BF16-relative and section 6.4 task gates as a declared representation
  configuration, never through the
  strict-teacher drift limits.

### 6.3 Arithmetic-class routing for PF levers

Decide this **before** writing the kernel, and record the class in the unit's
worklog entry. Classes are `EXECUTION-PROFILES.md` section 5.

| Class | What it covers | What it costs you | Examples in this campaign |
| --- | --- | --- | --- |
| **T0** | Bit-exact: same reduction order, same intermediates, same output bytes. Tiling, launch geometry, occupancy, LDS/SRAM residency, fusion that preserves order, launch-count reduction, layout changes, graph capture, chunk-size policy. | Exact-parity RED vs the registered owner + `rocprofv3 --kernel-trace` + whole-model A/B. **No production-numerics packet.** | PF-1d tile retiles (measured, rejected on speed); M6 warps-per-block and state-in-register; M2 integer routing compaction. |
| **T1/T2** | Local drift or reassociation: different accumulation order, split-K, WMMA accumulate, online softmax merge, precision of intermediates. | Everything in T0 **plus** a coverage-complete section 6.2 packet on a PF-0 fixture: mean/p95/p99/max KL, top-1 by category/shape/transition, three-repeat determinism, BF16-relative where available, task gates. Expect a full gate run per candidate. | PF-1c lever 1 (policy shape rows, rejected); PF-5 if partial-softmax is reassociated; PF-4 fused weighted expert sum. |
| **T3** | Representation/algorithm/decision-policy: activation or weight quantization change, approximate routing, changed acceptance or sampling. | **Not admissible through the drift gate at all.** Needs a declared product-configuration decision, the `EXECUTION-PROFILES.md` section 6.2 BF16-relative and section 6.4 task gates, and its own strict fallback. | Single-plane `d4` (2.40x, rejected as drift; see section 6.2 last bullet). |

Practical consequence for ordering: a `T0` lever can be landed by one coder in
one unit. A `T1/T2` lever costs a fixture prerequisite plus a multi-hour gate
per candidate. **Exhaust `T0` on a family before proposing `T1/T2` for it.**

The PF-1 fork, stated explicitly, because it is the most likely place to lose
time: the 23-26x dense-projection deficit sits on shapes served by the exact
F32 coltile owner (`gguf_k_prefill_out_coltile_rowbatch_kernel<float,float,8,8,4>`,
PF-1a). Extending the existing MMQ route to those shapes is inherently
`T1/T2` — that is exactly what PF-1c lever 1 measured and what failed. The two
honest paths are **(a)** pay the PF-0 + coverage-complete packet cost and
re-judge, or **(b)** build a *bit-exact* faster dense kernel for those shapes
(matching coltile's reduction order rather than replacing it), which is `T0`
and needs no packet. Path (b) is unexplored and is the better first attempt.

### 6.4 Open owner decisions (do not resolve these as a coder)

**D1 — does the shipped Q8 MMQ route re-qualify?** On the canonical fixture the
incumbent `d4x3` chain measures mean KL 2.73e-3, p95 1.14e-2, max 3.25e-2,
top-1 96.67% against strict, with all four categories over the `EXECUTION-PROFILES.md` section 6.1 limits
([`plane2 dual-basis`](../benchmarks/results/2026-09-03-gfx1151-qwen38-flash-next-pf1d-mmq-plane2-dualbasis.json)
`context_quality`). That is a single-layer isolation, so the full production
manifest would stack more. It is **not** yet a defect finding, because that
fixture is synthetic repeated material (PF-0 exists to fix this) and because
the route's unit-tier evidence at 512/128 rows is sound. Resolution path:
land PF-0, re-capture incumbent-vs-strict on it, then the owner decides among
re-qualified / scope-narrowed / strict-fallback for the affected shapes. Until
then the shipped default is unchanged and this row stays open.

**D2 — scope power.** Section 6.2 keeps sub-25-row scope verdicts binding as
implemented, which means a 12-row transition scope effectively requires 12/12
top-1. The 2026-08-31 accuracy review already recommended a recalibration
campaign for this. Any threshold or scope-size change goes through that
campaign with predeclared rules, never inside a PF unit.

Both decisions are inputs to the owner, not blockers on `T0` work. Start
PF-2/PF-5 while they are open.

### 6.5 Per-unit definition of done

Every PF unit, in addition to its row's exit condition:

1. Arithmetic class declared (section 6.3) in the worklog entry **before**
   implementation, with the registered strict fallback named.
2. `scripts/check_lineage.py` green; halo-box ports cite path + `a7ad7b7f`.
3. RED test green on the **unmodified** path first, then with the candidate.
4. `rocprofv3 --kernel-trace` showing the expected kernel name and a plausible
   duration; prebuild the `.so` and use `require_cached` (see `AGENTS.md`).
5. Same-session counterbalanced whole-model A/B: p512/p1024 primary, p4096
   recorded. A cycle-wall or sub-window win is retainable even when the
   aggregate ratio is flat within noise — say which it is.
6. If `T1/T2`: a section 6.2 packet on a PF-0 fixture with route-engagement
   coverage recorded. A packet below the 50% floor is `route_vacuous_for_scope`
   and settles nothing in either direction.
7. Artifact under `benchmarks/results/`, dated `benchmarks/CHANGELOG.md` line,
   immutable worklog entry, committed together with the code.
8. A measured loss is a result. Record it with its numbers and close the lever
   rather than iterating blind (`AGENTS.md` blocker table).

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
- **Numerics-fixture representativeness:** the two committed fixtures trade
  off against each other — the 18-prompt admission suite is natural text but
  reaches the `rows >= 64` Q8 MMQ route in 11.1% of rows, and the canonical
  `p512/p1024/p4096` fixture reaches it in 100% of rows but is repeated
  synthetic material (104 unique tokens in `code-p512`). A quality verdict from
  either one carries that caveat explicitly until PF-0 lands. Do not quote a
  KL or top-1 number from either fixture without naming which one it came from.
- **Author evidence gaps:** power not recorded, no repeat-arm across
  separately built binaries beyond the logit comparison, one prompt shape.
  Our repeatability classification applies to halo-box lanes exactly as it did
  to Nathan and EngramHalo.
