# Qwen3.8-Flash-Next halo-box Follow-up Campaign

Status: **Current Framework Desktop comparator screen completed 2026-09-05 at
hipEngine `c0cfdc3ef`, upstream llama.cpp `4d9176092`, and halo-box master
`b212548e0`.** Section 0 is the current c=1 overview. The original HB/PF
campaign evidence below remains tied to the power- and heat-limited `zbook` and
its pinned historical revisions; do not rewrite it as a cross-host delta. The
retained PF-1/PF-3/PF-5 T0 package is production. PF-0 is complete; D1 remains
an owner decision. Individual rejected levers do not close their parent
bottleneck.

**Correct the geometry before the next kernel unit:** binding GDN is
Hk=16/Hv=48/D=128, and QSA is 24 query heads/2 KV heads/D=256. The
PF-5 tile-16 candidate initially accepted **Hv=32 only**. Commit `617038db9`
repaired Hv48 coverage, and the subsequent engagement-verified whole-model A/B
promoted it at weighted prefill **89.435→89.873 (+0.49%)**,
**88.553→88.966 (+0.47%)**, and **72.661→72.929 tok/s (+0.37%)** for
p512/p1024/p4096, with all 72 trajectories exact. HB-2 demonstrates the
non-KDA 32-warp GDN path, not PR11's H=32 KDA tile-16 path. The prior PF-2 rejection's wave32 explanation is also incompatible
with D=256. Section 5.3 records these corrections and the actual Q8 tile
geometry; section 5.4 explains the remaining costs.

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

**PF-1 arithmetic-changing outcome (2026-09-03, before PF-0): every tested
dense-Q8 arithmetic lever closed, and the production-numerics gate exposed a
fixture problem.** Tile retiles were bit-exact but slower; the plane and policy-shape
levers changed arithmetic and failed the `EXECUTION-PROFILES.md`
section 6.1 envelope. Investigating
that failure surfaced a defect in the *evidence*, not the standard: the
2026-08-29 admission packet dispatched the route it admitted in 50 of 450 rows
(11.1%), and the only route-covering fixture available is synthetic repeated
material (104 unique tokens in 512). Section 6.2 freezes the resulting
admissibility rules, section 6.3 routes future levers by arithmetic class, and
section 6.4 lists the two owner decisions. PF-0 subsequently landed and the
natural-text incumbent-vs-strict capture failed the calibrated gate; D1 now
awaits the named owner decision.

The valid remediation packet measured weighted before→after prefill at
**86.62→89.34 (+3.13%)**, **85.88→88.54 (+3.09%)**, and
**70.80→72.58 tok/s (+2.51%)** for p512/p1024/p4096. All 12 category/shape
cases improved, all 72 measured trajectories were exact across modes, and
shape-level decode changed by −0.15%/+0.13%/−0.07%. A fresh matched-BF16
halo-box PR11 screen measured **240.11/18.04**, **324.64/17.24**, and
**349.49/15.10 pp/tg128**. Its p512/p1024 prefill rows are screening-only
because maximum per-case CV reached 10.0%/9.7%; p4096 remained stable at 1.26%.
Evidence:
[`halo PF-1/PF-3 production refresh`](../benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-halo-pf13-production-refresh.json).

The binding comparator set and the main campaign's section-6 closure rules
remain owned by
[`QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md`](QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md);
this document is a subordinate lane that (a) adds the halo-box fork as a
measured comparator, (b) profiles it role-by-role against hipEngine, and (c)
converts every confirmed per-mechanism advantage into a `W/C/O/s` candidate for
the main campaign's impact queue.

The execution unit is an exclusive, operation-complete hipEngine owner,
ranked by recoverable request time. The halo-box diff supplies mechanisms;
its pre-existing MMQ and attention architecture also matters. Matching a
microbenchmark alone does not close the aggregate gap. This campaign does
not relax any correctness, evidence, or anti-gaming rule.

## 0. Current Framework Desktop c=1 overview

**MMQ prepack promotion (September6 UTC):** same-residency full12-case
prefill156.707->157.748 /154.396->155.375 /143.903->144.736 tok/s
at512/1K/4K (+0.664%/+0.634%/+0.579%),72/72 trajectories exact.
All prefill rows improve;11/12 request-wall rows improve, mixed512 loses0.14%.
Decode is slightly lower; no intrinsic decode claim. Additional memory1.67GiB,
cold preparation0.107s. [Evidence](../benchmarks/results/2026-09-06-framework-qwen4exp-mmq-prepack-production.json).
The generated family tables now use post-MMQ production `ef63870f9`;
the earlier pinned Vulkan profile is explicitly reused, not newly measured.

**GR wave-scale promotion (September6 UTC):** the next exact full12-case A/B
improves prefill156.017->156.711 /153.748->154.285 /143.419->143.908 tok/s
at512/1K/4K (+0.445%/+0.350%/+0.341%). All72 trajectories are exact,
every case improves request wall, and elapsed is33m22s. No decode-kernel win
or new external parity claim. [Evidence](../benchmarks/results/2026-09-06-framework-qwen4exp-gr-wave-production.json).
The generated family costs below now use post-MMQ production `ef63870f9`,
with the earlier pinned Vulkan profile explicitly reused.

**Later Q8 promotion:** normal-model full A/B improves prefill
154.29->155.45 /151.67->153.43 /141.58->143.41 tok/s at512/1K/4K
(+0.75%/+1.17%/+1.29%), with all72 trajectories exact and every complete
request faster. This incremental result is not a new external baseline;
decode drifts in both arms. [Q8 promotion](../benchmarks/results/2026-09-05-framework-qwen4exp-q8-wave-scale-production.json).
The following frozen throughput baseline is the **pre-Q8-wave-scale snapshot**.
The generated family table has now been refreshed for hipEngine `ef63870f9`,
while explicitly reusing the pinned prior Vulkan profile; it is not a new
simultaneous throughput comparison.

**Frozen three-engine refresh (2026-09-05 UTC):** controller/runtime
`bd451a417`, halo-box `b212548e0`, Framework machine
`55ea6c509d0b49eea8de7094a1023668`, UD-Q4_K_XL/BF16 KV, identical12-case
canonical fixture, one warmup plus three measured tg128 requests per case.
All108 measured trajectories repeat within their engine, hipEngine closes
to zero allocations, and both external servers exit0. Logger/profiler off.

| Engine | p512 PP / TG | p1024 PP / TG | p4096 PP / TG | Max per-case PP / TG CV |
| --- | ---: | ---: | ---: | ---: |
| hipEngine combined production | 153.96 / 19.38 | 152.30 / 18.75 | 142.03 / 14.42 | 2.45% / 9.02% |
| halo-box Vulkan target | 316.28 / 25.27 | 391.68 / 25.30 | 425.72 / 24.51 | 12.52% / 3.91% |
| halo-box HIP diagnostic | 282.76 / 21.08 | 368.33 / 20.56 | 351.08 / 18.83 | 11.26% / 4.10% |

Rates are weighted tok/s. The corresponding Vulkan/hipEngine target factors
are **2.054/2.572/2.997x prefill** and **1.304/1.349/1.700x decode**.
These are sequential same-host screening ratios, not inter-engine
counterbalanced confidence bounds. Every lane exceeds the2% stability
criterion on at least one metric; no statistical match/beat claim follows.
Full execution36m18s (hipEngine17m21s, Vulkan8m39s, HIP10m17s).
[Frozen baseline packet](../benchmarks/results/2026-09-05-framework-qwen4exp-refreshed-baselines.json).
This closes the missing standalone combined-default decode measurement.
The shared-family matrix is now generated in section5.2.1 from matching
Framework captures; the overall optimization campaign remains open.

The following promotion notes and earlier comparator screen are retained
as revision-specific history, not the current baseline table above.

Latest Q4 pair admission measures combined-stack prefill
**153.78/151.21/141.30 tok/s** at p512/p1024/p4096, with all72 trajectories
exact. Decode measures19.383/18.512/12.899 tok/s in that A/B, with explicit
drift and incremental losses of0.01%/0.23%/1.63%; do not treat these as a
fresh same-session comparison against the earlier Vulkan table.
[Q4 pair production](../benchmarks/results/2026-09-05-framework-qwen4exp-q4-pair-production.json).
The owner has requested a frozen combined-default Framework baseline and
shared-taxonomy family refresh against halo-box Vulkan. This is the next
campaign priority; historical `zbook`/upstream-HIP buckets are not substitutes.

**Later retained hipEngine update (2026-09-05 UTC):** exact serial-prefix
register GDN on top of the retained Q51/Q4/QSA stack measures prefill
**145.49/143.10/134.32 tok/s** at p512/p1024/p4096 in the full-category
same-residency A/B, a9.50-10.55% incremental gain. All72 trajectories are exact
and all12 complete requests improve3.28-6.79%. Decode p1024/p4096 loses
0.45%/3.63% amid drift; retained under the owner's prefill-first direction.
The comparator table below remains its earlier pinned screen, not a new
cross-engine run. [Latest production evidence](../benchmarks/results/2026-09-05-framework-qwen4exp-gdn-register-production.json).

The later 2026-09-05 owner decision retains exact QSA page256 and bundled-Q4
prefill in production, accepting the recorded small hot-decode tradeoffs
as follow-up work. The table below remains the pinned earlier comparator
screen, not a measurement of the combined new defaults.
[Promotion evidence](../benchmarks/results/2026-09-05-framework-qwen4exp-prefill-promotion.json).

Execution follows the main campaign's **active execution contract**: exclusive
Framework host `gfx1151`, halo-box Vulkan as the working performance target
despite screening variance, native FP8 remote quality reference, and same-Q4
llama.cpp implementation reference without production cross-engine text
equality. The historical section-6 tile-16 pending step is complete in
`22dc56268`; serial-prefix GDN is now retained as described above. Remaining
routed MoE/dense and decode work is still open.

This is the active same-host snapshot. It uses the verified four-part Unsloth
`UD-Q4_K_XL` artifact, BF16 K/V, the canonical code/English/Japanese/mixed
fixture at p512/p1024/p4096, 128 decode transitions, one warmup, and three
measured requests per case. Cells are weighted prompt-processing / decode
tokens per second.

| Engine and backend | p512 | p1024 | p4096 | Repeated output |
| --- | ---: | ---: | ---: | --- |
| hipEngine production `c0cfdc3ef`, HIP | 118.44 / 19.92 | 117.79 / 19.22 | 95.14 / 15.21 | **12/12** |
| Upstream llama.cpp `4d9176092`, HIP | 283.85 / 21.06 | 367.97 / 20.79 | 395.02 / 19.63 | **11/12**; `mixed_ja_en-p4096` varies |
| Upstream llama.cpp `4d9176092`, Vulkan | 230.35 / 24.94 | 305.47 / 24.59 | 357.44 / 23.53 | **11/12**; `mixed_ja_en-p4096` varies |
| halo-box master `b212548e0`, HIP | 265.69 / 21.02 | 368.90 / 20.50 | 356.62 / 18.63 | **12/12** |
| halo-box master `b212548e0`, Vulkan | **298.97 / 24.92** | **369.72 / 24.52** | **402.46 / 23.47** | **12/12** |

Physical host `gfx1151`, machine ID `55ea6c509d0b49eea8de7094a1023668`,
Framework Desktop / Ryzen AI Max+ 395 / Radeon 8060S, kernel
`7.1.6-1-cachyos`, accelerator-performance profile, performance CPU governors,
and high GPU clock policy. The external servers used `-ngl 999 -fa on -ctk
bf16 -ctv bf16 -c 4352 -b 8192 -ub 2048 -t 4`; all exited cleanly. hipEngine
used production manifest `86e5c619…`, required cached JIT artifacts, did not
fall back to strict, and repeated all 36 measured outputs.

The screen does **not** freeze a new closure target. Every external lane exceeds
the 2% maximum per-case coefficient-of-variation rule on at least one metric;
halo-box Vulkan is the fastest repeatable raw row but reaches 19.8%/8.0%/4.3%
maximum per-case prefill variation at p512/p1024/p4096. The two upstream lanes
also fail output repeatability on one case. Relative to halo-box Vulkan,
hipEngine reaches 39.6%/31.9%/23.6% of prefill throughput and
79.9%/78.4%/64.8% of decode throughput. These are same-host screening ratios,
not `zbook` old-to-new deltas. See the
[compact packet](../benchmarks/results/2026-09-05-framework-gfx1151-qwen38-flash-next-current-comparators.json)
and the active
[Strix Halo survey](QWEN3.8-FLASH-NEXT-STRIX-HALO-SURVEY.md).

## 1. Historical source identity and claim summary

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
| M6 | GDN tuning, `gated_delta_net.cu`: 32 warps/block (vs 4) for S_v=128, H∈{48,64}, non-KDA; token-tile-16 state-in-register KDA prefill kernel for H=32, `n_tokens >= 16`, single sequence | **Partial** — binding Hv=48 non-KDA uses 32 warps; H=32 KDA tile-16 is inactive (section 5.3) |
| M7 | Decode FA GQA opt, `fattn*`: gated to `Q->ne[1]==1`, `gqa_ratio==6`, **Q8_0 K/V only**; faster FA tile variants were rejected by the authors for changing logits | **No** — binding BF16 KV and GQA=12 both fail the gate; changing KV alone does not activate it |
| M8 | Elementwise/recurrent specializations: transposed concat tile-16, direct-index contiguous mul, quantize/norm/sumrows gfx1151 paths | **Yes** where shapes match; map per kernel in HB-3 |
| M9 | Vulkan: cooperative-matrix dequant selection for wide prompt matmuls and large MoE batches, tiled recurrent concat, direct-index mul | Backend-disjoint; design evidence only, per main-campaign scope rules |

## 2. Historical `zbook` objective and boundaries

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
  UD-**IQ4_XS** matching the PR's tested quant, and Q8_0 K/V as a separate cache diagnostic. M7 also requires GQA=6; the binding model has GQA=12.
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

## 3. Historical `zbook` topline matrix (c=1..8)

These tables are the campaign scoreboard. Every cell starts as `TBD` and is
filled only from a measured, identity-pinned run under section 4's protocol.
`c` is the number of simultaneous request slots: `llama-server -np c` for the
llama.cpp lanes, hipEngine's serving/batch path for its lane. A lane with no
working concurrent mode records `unsupported`, never zero. c=1 binds first;
c=2..8 are extension rungs that must not average away a c=1 result.

Lanes: **hipEngine** (named production, pinned UD-Q4_K_XL, BF16 K/V),
**upstream HIP** (patched `f1793c1c4`, existing comparator),
**HB-base** (halo-box `6c84c7d5`), **HB-PR11** (halo-box `a7ad7b7f`).

### 3.0 Latest production refresh (2026-09-04; screening comparator)

The [PF-1/PF-3 refresh artifact](../benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-halo-pf13-production-refresh.json)
contains exact commands, source/binary/profile hashes, correctness, and physical
host identity: `zbook`, machine ID `87c566d30a5645cf8d12ed7ef6b6e1e8`, Ryzen AI
Max+ Pro 395 / Radeon 8060S, gfx1151, UD-Q4_K_XL, BF16 K/V, c=1, four canonical
categories, tg128. These are existing measurements, not a new benchmark.

| Shape | hipEngine pp / tg128 tok/s | HB-PR11 pp / tg128 tok/s | HB/hipEngine prefill rate | hipEngine decode rate deficit vs HB |
| --- | ---: | ---: | ---: | ---: |
| p512 | 89.34 / 14.84 | 240.11 / 18.04 | 2.69x | 17.8% |
| p1024 | 88.54 / 14.79 | 324.64 / 17.24 | 3.67x | 14.2% |
| p4096 | 72.58 / 12.40 | 349.49 / 15.10 | 4.82x | 17.9% |

Thus the approximate 15% decode deficit is consistent with this screen;
prefill approaches 5x at p4096, with a smaller short-prompt gap. Rate deficit
is `1 - hipEngine/HB`; HB's rate advantage uses the other denominator. Matching
these prefill rates would require approximately 63%/73%/79% less prefill wall.
HB p512/p1024 prefill CV reaches 10.0%/9.7%, so those magnitudes remain
screening-only; p4096 CV is 1.26%. Five matched thermal pairs for campaign
closure remain outstanding. Sections 3.1/3.2 preserve **historical HB-1 arm B**,
not the current production row.

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
| M6 | 32-warp non-KDA GDN; separate KDA tile-16 | **Partial** — workgroup Y changes 4→32; core p4096 prefill kernel sum 2,139.377→647.976 ms | GDN mixer (655 ms/1.233 s/4.983 s; 2.670 ms live-513) | Hk=16, Hv=48, Dk=Dv=128, T=1/512/1024/2048; normalized Q/K through core+state | blocked — catalog geometry/boundary differs | blocked — state layout and prepare/post boundary differ | **Blocked before timing** |
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
pp tok/s versus the profile packet's unprofiled 85.711 pp at p1024, so it was
a useful initial ranking. PF-1/PF-3 and other units have since
landed. Keep this ledger as **historical scale evidence**, not a fresh route
census or a claim that every current owner remains within 3%. Source inspection
supports the priorities below; remeasure changed owners when the GPU is free.

Device-time totals per profiled request (hipEngine vs llama.cpp, ms):

| Shape | hipEngine | llama.cpp | Gap | Ratio |
| --- | ---: | ---: | ---: | ---: |
| p512 | 5,965.1 | 1,911.1 | 4,054.0 | 3.12x |
| p1024 | 11,183.7 | 2,971.9 | 8,211.8 | 3.76x |
| p4096 | 54,558.9 | 10,802.0 | 43,756.8 | 5.05x |

Historical classifier buckets (device ms; bucket ratios in parentheses).
The labels are not exclusive semantic operations: `dense_other` includes
selected experts, and `dense_quant_q8` includes matrix compute. Do not interpret
the 23–26x bucket ratio as a matched dense-kernel ratio (section 5.3).

| Family | p512 | p1024 | p4096 | PR11 mechanism (HB-2) | Workstream |
| --- | ---: | ---: | ---: | --- | --- |
| Mixed projection bucket (`dense_other`) | 1,181.3 vs 50.4 (23.4x) | 2,173.3 vs 94.6 (23.0x) | 8,767.0 vs 337.7 (26.0x) | M1 Q8_0 MMQ tile retune (active) | **PF-1** |
| QSA attention | 51.0 vs 12.9 (3.9x) | 180.1 vs 48.8 (3.7x) | 10,229.2 vs 526.0 (19.5x) | none under BF16 K/V (M7 inactive) | **PF-2** (native) |
| MoE gate/up Q4_K | 1,368.5 vs 700.3 (2.0x) | 2,528.6 vs 851.7 (3.0x) | 10,233.3 vs 1,906.5 (5.4x) | M1 Q4_K tile retune (active) | **PF-3** |
| MoE down Q5_1 | 1,173.9 vs 501.4 (2.3x) | 2,192.8 vs 625.0 (3.5x) | 8,782.4 vs 1,454.1 (6.0x) | none measured (M1 Q5_1 did not retune) | **PF-3** (native) |
| Q8 MMQ compute + packing (`dense_quant_q8`) | 957.4 vs 269.0 (3.6x) | 1,802.3 vs 451.2 (4.0x) | 7,241.7 vs 1,592.1 (4.6x) | M8 quantize/elementwise paths (active, partial) | **PF-1** |
| GDN | 655.2 vs 83.3 (7.9x) | 1,233.0 vs 222.5 (5.5x) | 4,982.8 vs 1,845.3 (2.7x) | M6 non-KDA 32-warp (active; p4096 core 2,139.4→648.0 ms); KDA tile-16 inactive | **PF-5** |
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
- The QSA ratio grows from 3.7x to 19.5x as p4096 crosses the dense-to-sparse
  transition. Each query must attend its own history; chunking alone does not
  prove redundant computation. The sparse implementation's per-key reduction
  and synchronization are the actionable costs (section 5.4). M7 is inactive,
  but halo-box's existing masked Flash Attention is relevant BF16 design
  evidence. PF-2 resumes the main campaign's P4 QSA prefill subowner evidence
  ([`p4-qsa-prefill-subowner`](../benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p4-qsa-prefill-subowner.json),
  [`dense-other subowners`](../benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-dense-other-subowners.json)).
- HB-3's matching blocker stands: these are role-aligned whole-request rows,
  not valid cross-engine microbenchmark ratios. PF units promote on in-tree
  whole-model same-session A/B plus their RED gates, using HB-2 only as
  mechanism-existence evidence.

### 5.2.1 Framework starting and current owner snapshots

The current snapshot is post-MMQ production `ef63870f9`, captured on
September6 UTC. The post-GR snapshot remains linked historical evidence.

The generated tables below supersede the subsequent post-GDN checkpoint
for the current family comparison. Captures use the post-MMQ production
runtime, halo-box Vulkan `b212548e0`, identical fixtures and this Framework
machine. Fine per-node/per-kernel provenance and the regeneration command
are in the [post-MMQ family packet](../benchmarks/results/2026-09-06-framework-qwen4exp-post-mmq-family.json).
The fresh hipEngine capture is from `ef63870f9` on September6 UTC; Vulkan is the unchanged
earlier capture named in that packet, not a newly measured target. The
[pre-Q8 packet](../benchmarks/results/2026-09-05-framework-qwen4exp-family-alignment.json)
remains immutable historical evidence, as does the
[post-Q8 packet](../benchmarks/results/2026-09-05-framework-qwen4exp-post-q8-family.json).
The [post-GR packet](../benchmarks/results/2026-09-06-framework-qwen4exp-post-gr-family.json)
is also immutable history. Family gaps do not establish that Vulkan rates
are achievable under parent-bit-exact arithmetic; instruments and numerical
contracts differ.

<!-- BEGIN FRAMEWORK FAMILY REFRESH -->
#### Generated Framework Family Tables

Taxonomy: `qwen4exp-semantic-owners-v2-complete-gr`. Same Framework host and UD-Q4_K_XL/BF16 KV.
Device timings are diagnostic: HIP kernel sums versus serial Vulkan query intervals.
Use the logger-off baseline table for throughput and parity factors.

**Starting versus current, exact code-p4096 fixture:**

| Owner | Framework arrival (ms) | Current (ms) | Device share | Device zero-cost ceiling | Wall zero-cost ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| MoE/FFN (routed + shared) | 19,773.694 | 14,634.927 | 54.02% | 2.175x | 2.149x |
| Non-FFN, non-GR linear | 5,828.219 | 5,374.271 | 19.84% | 1.247x | 1.244x |
| GR projections + read/mix | 4,351.404 | 4,279.129 | 15.80% | 1.188x | 1.185x |
| QSA | 8,982.811 | 1,924.080 | 7.10% | 1.076x | 1.076x |
| GDN | 3,906.453 | 783.717 | 2.89% | 1.030x | 1.029x |
| Boundary / residual combine | 76.563 | 80.051 | 0.30% | 1.003x | 1.003x |
| PLE | 14.224 | 14.102 | 0.05% | 1.001x | 1.001x |

Current kernel sum 27,090.277 ms; profiled wall 27,370.913 ms.
Ceilings are sensitivity bounds, not expected realizable speedups. Snapshot deltas
are historical attribution, not replacements for each retained A/B.

**Prefill, four-category p4096 mean (ms):**

| Owner | hipEngine | halo-box Vulkan | HE / Vulkan | Difference |
| --- | ---: | ---: | ---: | ---: |
| MoE/FFN (routed + shared) | 14,432.901 | 4,372.034 | 3.301x | +10,060.867 |
| Non-FFN, non-GR linear | 5,366.975 | 1,019.237 | 5.266x | +4,347.738 |
| GR projections + read/mix | 4,281.765 | 1,707.311 | 2.508x | +2,574.454 |
| QSA | 1,921.995 | 645.424 | 2.978x | +1,276.571 |
| PLE | 14.062 | 75.456 | 0.186x | -61.394 |
| Boundary / residual combine | 76.954 | 506.861 | 0.152x | -429.907 |
| GDN | 786.544 | 1,390.400 | 0.566x | -603.856 |
| **Total device time** | **26,881.195** | **9,716.723** | | |

**Decode, four-category p4096 mean (ms):**

| Owner | hipEngine | halo-box Vulkan | HE / Vulkan | Difference |
| --- | ---: | ---: | ---: | ---: |
| QSA | 16.949 | 3.161 | 5.363x | +13.788 |
| MoE/FFN (routed + shared) | 17.330 | 12.469 | 1.390x | +4.861 |
| Non-FFN, non-GR linear | 17.078 | 16.394 | 1.042x | +0.683 |
| GR projections + read/mix | 6.610 | 6.115 | 1.081x | +0.496 |
| PLE | 0.032 | 0.115 | 0.274x | -0.084 |
| GDN | 2.351 | 2.777 | 0.847x | -0.426 |
| Boundary / residual combine | 0.458 | 1.386 | 0.330x | -0.928 |
| **Total device time** | **60.806** | **42.416** | | |

Decode is a fixed-live4097 diagnostic, averaged over three restored HIP repetitions
and one Vulkan appended-root query per category, not a tg128 trajectory average.
Negative differences are not automatically transferable savings; intervals, fusion
and dispatch instrumentation differ. Every timestamp has one semantic owner.

<!-- END FRAMEWORK FAMILY REFRESH -->

**Historical post-GDN checkpoint (before Q4 pair promotion):**

These are **same-physical-host diagnostic snapshots**, not joined to the
historical `zbook` classifier above. Both use UD-Q4_K_XL, BF16 KV and the
exact `code-p4096` fixture. The Framework arrival snapshot is `cf9c55920`
from the [owner refresh](../benchmarks/results/2026-09-05-framework-gfx1151-qwen38-flash-next-owner-refresh.json);
the post-GDN snapshot is `511dd977a` from
[Q4 pair screen -> owner_refresh](../benchmarks/results/2026-09-05-framework-qwen4exp-q4-pair-reuse.json).
The latter precedes Q4 pair production. The subsequent measured post-Q4
and post-Q8/post-GR/post-MMQ packets supersede its current-cost role; no later values are
inferred from microbenchmark gains.

| Exclusive owner | Framework arrival (ms) | Post-GDN (ms) | Post-GDN device share | Device zero-cost ceiling | Wall zero-cost ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| Routed MoE | 19,023.675 | 15,511.808 | 53.32% | 2.142x | 2.096x |
| Non-routed linear | 8,290.897 | 8,165.727 | 28.07% | 1.390x | 1.380x |
| GR read | 2,638.745 | 2,623.620 | 9.02% | 1.099x | 1.097x |
| QSA prefill | 8,982.811 | 1,921.000 | 6.60% | 1.071x | 1.069x |
| GDN | 3,906.453 | 779.854 | 2.68% | 1.028x | 1.027x |
| PLE | 14.224 | 13.568 | 0.05% | 1.000x | 1.000x |
| Prefill boundary | 76.563 | 74.891 | 0.26% | 1.003x | 1.003x |

Post-GDN owner sums are **29,090.468 ms**; the profiled wall window is
**29,661.252 ms**, with 100% launch attribution. Device share uses the first
denominator. For owner cost `C`, the ceilings are `D/(D-C)` and `W/(W-C)`.
The 570.784-ms difference is **not automatically Python overhead**. These
are optimistic sensitivity bounds, not predicted realizable speedups.
The arrival profiled window is 43,171.036 ms; individual phase gains must
still be established by their retained same-session A/B packets.

### 5.2.2 Shared-taxonomy refresh protocol

**Owner-requested refresh completed; optimization resumes from its ranking.**
The [machine-readable queue](../benchmarks/results/2026-09-05-framework-qwen4exp-family-refresh-queue.json)
has **frozen baselines and six-case family alignment captured**. It freezes the combined Q4-pair production default,
then measures halo-box **Vulkan `b212548e0`** as the target and HIP as a
separate diagnostic. GPU stages run serially.

The generated section5.2.1 tables replace the former pending alignment.
The [family packet](../benchmarks/results/2026-09-05-framework-qwen4exp-family-alignment.json)
contains the original six exact prompt cases and12 phase comparisons: code512/1024/4096
and all four p4096 categories, with fixed-live513/1025/4097 decode anchors.
All profiles have100% semantic coverage; Vulkan outputs match the original
engine's prefill/next-token reference, and all HIP decode repetitions pass
state and lifecycle checks. Fine role sources, node IDs, binary/library hashes,
commands and regeneration inputs are retained. This is diagnostic alignment,
not identical instrumentation or statistical throughput parity. The generated
table is now updated by the separately linked post-MMQ current-HE refresh.

**Collector implementation update:** `scripts/qwen4exp_framework_family_refresh.py`
now captures request-bounded Vulkan logs, runs serial baseline commands,
captures HIP roles, rejects identity/coverage mismatches and renders joined
tables. `scripts/qwen4exp_vulkan_owner_build.py` builds owned host-library
copies against pinned halo-box; the original source and shader objects stay
unchanged. Metadata-only graph scopes and fused-node IDs give the p512/p4096
smokes 100% semantic coverage for prefill and one-token decode. Both return
the same greedy prefix as the original uninstrumented binary. That initial
instrumentation validation is now followed by the full baseline and six-case
family packet; full-logit/task-quality qualification remains separate.
[Instrumentation evidence](../benchmarks/results/2026-09-05-framework-qwen4exp-vulkan-owner-instrumentation.json).

The versioned comparison taxonomy uses **complete MoE/FFN, including shared
projections**, because HIP decode's MoE graph includes those projections
while HIP prefill exposes them as nested linear roles. The collector
normalizes shared projection slots/weight names into MoE on both sides,
preserving their original tags. Consequently these older routed-only
snapshot numbers are not directly interchangeable with the new complete-FFN
table. Taxonomy v2 also groups all GR down/up/inject projections and read/mixing
together. This is necessary because HIP small-row up is a nested linear role
while fused large-row up is inside GR read; residual combine stays boundary.
Original role/weight tags remain available. Unknown/mixed fusions block
matched-gap publication, and stored v1 captures are regenerated from raw
role/log sources before joining rather than relabeled without recomputation.

The quoted llama.cpp 3.63/1.93/0.53/1.85-second values belong to the older
**zbook patched-upstream HIP** classifier. They are not current halo-box
Vulkan measurements, and the dense/MoE boundaries do not match these rows.
Do not use their ratios as matched gaps, call GDN 2.4x ahead of the target,
or turn that hypothetical substitution into proof that there is no hardware
or implementation limit. The retained QSA/GDN reductions are real; their
remaining matched Vulkan gaps need the new capture.

Execution order and acceptance:

1. Commit the repeatable capture/join/table generator using
   `scripts/llamacpp_vulkan_perf_summary.py` for log parsing and
   `scripts/qwen4exp_role_analyze.py` for HIP attribution. Selected matmul
   must be classified before quant type: Q5_1 and selected Q8 are MoE, not
   elementwise/dense. Stock Vulkan unary/layout names can lack ownership;
   collect node/source/fusion metadata or expose them as unclassified.
2. Freeze controller/runtime revisions, original and instrumented binary
   hashes, model/fixture identity, driver/compiler, clocks and production
   manifest. Keep reference repos read-only. Check instrumented output
   parity against its own uninstrumented engine.
3. Run all 12 canonical AR cases at one warmup and three repetitions,
   including tg128, on the combined hipEngine default, halo-box Vulkan,
   and halo-box HIP. No profiler/logger in throughput rows.
4. Separately profile prefill and aligned-context decode. Vulkan uses
   `GGML_VK_PERF_LOGGER=1`, not rocprofv3; select request-bounded graph
   sections, including every ubatch. Collect all p4096 categories and code
   p512/p1024 anchors. Distinguish fixed-live diagnostics from tg128 averages.
5. Emit both sides under one versioned, exclusive taxonomy. Reconcile sums,
   disclose unknown/mixed fused cost, reject host/model/fixture/phase/context
   mismatches, and withhold matched-gap claims until coverage is resolved.
   Regenerate these tables after every retained promotion.

There **are** combined-stack decode measurements in the GDN and Q4 A/B
packets, plus post-binder state checks. The frozen-default cross-engine
baseline and aligned current decode-family costs are now recorded above.
The 20.913/80.061-ms ordered-QSA figures and its historical comparator
are `zbook` evidence, not a Framework 35x precedent. Compute the decode
parity factor from the new matching rows rather than carrying forward 1.54x.

### 5.2.3 Next measured lever after refresh

Queue a joint **prefill chunk {512,1024,2048} x Q4 ROW_BATCH {8,16,32}**
screen. The current gate/up family uses row batches of eight; heavy experts
can therefore reread packed weight values over multiple row groups. Larger
row batches may change the earlier chunk-size tradeoff, so the old
chunk-only sweep does not close this joint experiment.

The suggested ~800->444 weight passes and ~1.8x traffic ratio are **inferred,
not measured speedups**. Measure actual expert-count distributions, register
pressure (pair2 currently uses 88 VGPR), LDS/scratch, cache/weight traffic,
the complete MoE operation and whole-request wall. Preserve the exact parent
gate and smaller-row fallback; do not promote on traffic arithmetic alone.

**First joint kernel screen (2026-09-05 UTC): rejected.** Instantiate the
current pair2 kernel at ROW_BATCH16/32, keeping the same K/reduction sequence
and testing token counts512/1024/2048 with real layer0/layer4 weights.
Routing and activations in this screen are synthetic uniform/skewed inputs,
not captured model routing. After a resource audit, an exact variant also
skips absent-row reduction/publication work. Its speed ratios versus RB8 are:

| Routing | Row batch | tokens512 | tokens1024 | tokens2048 |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 16 | 0.981x | 0.947x | 0.916x |
| Uniform | 32 | 0.606x | 0.623x | 0.590x |
| Skewed | 16 | 0.921x | 0.924x | 0.880x |
| Skewed | 32 | 0.610x | 0.598x | 0.586x |

Every output is exact but all12 final cells regress. On the skewed map,
inferred weight passes at tokens2048 fall2801->1557->946 for RB8/16/32;
this does not translate into a speedup. RB8 uses88 VGPR, masked RB16 uses120,
and original RB32 uses176, all with zero scratch. Register pressure is a
measured change, not proof of the sole cause; DRAM/MALL traffic was not
measured. Both larger-RB variants and selectors are removed. The harness
retains weight-pass telemetry, and production keeps RB8.
[Rejected screen and source recipe](../benchmarks/results/2026-09-05-framework-qwen4exp-q4-rowbatch-rejected.json).
No expensive full-model chunk sweep was run for these losing candidates.
This does not close every adaptive/heavy-expert or changed-layout reuse
design; a future attempt must name a new mechanism rather than simply
instantiate the same larger row batches again.

### 5.3 Source audit corrections (2026-09-05)

Audit basis: hipEngine `3574a1bd2` and read-only halo-box
`/home/lhl/halo-box-strix-llama/hb-pr11` at `a7ad7b7f`. These are code findings,
not new timings. They supersede conflicting explanations in earlier PF rows
and worklogs; stored measurements and immutable entries are preserved.

1. **Use binding geometry.** The validated model contract in
   [`qwen4_exp_gguf.py`](../hipengine/loading/qwen4_exp_gguf.py)
   (`_geometry_errors`) requires QSA 24 query heads, 2 KV heads,
   key/value dimension **256**; indexer dimension 128 is a different tensor.
   GDN is **16 K heads, 48 V heads, D=128**, inner width 6144.
   [`Qwen4ExpRunner._prefill_chunk`](../hipengine/runtime/qwen4_exp_runner.py)
   passes these config fields directly to the mixers. PR11
   `ggml/src/ggml-cuda/gated_delta_net.cu:310–322` selects 32 warps for
   H=48/64 non-KDA, whereas tile-16 requires **KDA and H=32**. HB-2's M6
   symbols are `gated_delta_net_cuda<128,false,false>`: its retained census
   proves the former, not the latter. GDN tile-16 remains an adaptation idea,
   not a measured-active mechanism on this payload.
2. **PF-5's original candidate had a binding-shape gap (now repaired).** At
   audited commit `3574a1bd2`, both
   [`qwen4_exp_gdn.py`](../hipengine/kernels/hip_gfx1100/linear_attn/qwen4_exp_gdn.py)
   and the raw launcher in its `.hip` reject `num_v_heads != 32`;
   [`test_qwen4_exp_gdn_tiled16_prefill.py`](../tests/test_qwen4_exp_gdn_tiled16_prefill.py)
   covers Hv=32 only. The
   [implementation checkpoint](../worklog/entries/20260905T011450.253353Z-lhl-qwen4exp-pf5-gdn-tiled16-implementation-447dc4.md)
   reports a 32.1% rows-512 operation-chain improvement at that other shape.
   It cannot support wiring or promotion at Hv=48. Extend the contract and
   measure the actual shape before claiming a candidate saving here.
   **Concurrent update:** `617038db9` now accepts Hv in {32,48}, and its
   [correction checkpoint](../worklog/entries/20260905T022638.146026Z-lhl-qwen4exp-pf5-gdn-tiled16-hv48-correction-c178a9.md)
   records binding-shape parity, leaf timing and trace. No default changed.
   It also invalidates a first whole-model A/B: changing only a registry key
   did not replace the runner's directly imported columnwarp function, so both
   arms ran the parent. Next, require candidate invocation counts on the live
   path and a corrected one-residency A/B. That no-op timing is no rejection.
3. **PF-2's failure is observed; its stated cause is not established.** The
   runner selects the wave32 sparse rows function only for `head_dim == 128`.
   D=256 uses `qsa_sparse_attention_paged_bf16_rows_f32_kernel`. The old RED
   fixture uses 4 query heads/2 KV heads/D=8 and three selected keys, so it
   does not establish exactness at the binding geometry and selection budget.
   The [PF-2 artifact](../benchmarks/results/2026-09-03-gfx1151-qwen38-flash-next-pf2-qsa-ordered-prefill-ab.json)
   records p4096 output mismatch and process-separated timing; keep the
   candidate unpromoted, but withdraw “production wave32 caused it” and
   “extra launches proved the loss.” Reproduce with actual-shape output/state
   capture, route census and one-residency A/B before attributing a cause.
4. **Q8 MMQ already tiles both matrix axes and uses matrix instructions.**
   [`gguf_q8_0_mmq_prefill.hip`](../hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_mmq_prefill.hip)
   defines M=N=128; `launch_mmq_tile` launches
   `ceil(out_features/128) × ceil(rows/128)` blocks, each 32×8 threads.
   It cooperatively stages weights/activations and calls
   `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`. The same M/N and grid formulas
   exist at profile commit `3ddb748d`. The
   [PF-1a entry](../worklog/entries/20260902T233138.374147Z-lhl-qwen4exp-pf1a-dense-scope-199383.md)
   incorrectly inferred one output column per block from trace grid sizes.
   Reconcile profiler work-item dimensions with launch block counts before
   any occupancy argument. “Add MMQ tiling” is not a missing feature.
5. **The family names hide different operations.**
   [`qwen4exp_trace_analyze.py`](../scripts/qwen4exp_trace_analyze.py)
   classifies Q8 MMQ **matmul** and packing together as `dense_quant_q8`.
   PF-1a's historical p4096 split is about 6.926 s matmul versus 0.229 s
   activation packing, within the 7.242 s bucket. A pack-only optimization
   cannot remove the matmul deficit. `dense_other` includes selected Q8 down
   and layer-2 Q5_K gate/up alongside Q8 weights with F32 activations; it
   does **not** mean F32 weights. The
   [subowner artifact](../benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-dense-other-subowners.json)
   splits its historical 8.767 s into approximately 3.086 s selected Q8 down,
   2.775 s F32-activation coltile, 2.337 s selected Q5_K, plus smaller work.
   The selected-Q8 part changed with PF-1. Do not assign the whole 23–26x
   bucket ratio or all 8.767 s to a dense coltile kernel.
6. **A failed transplant does not disprove its external mechanism.** PF-4's
   O(E×T) per-expert scan loses to hipEngine's existing map chain, but the
   earlier “designed for E≤32” explanation is unsupported: halo-box runs the
   same 512-expert payload and HB-2 observes this helper. Different incumbent
   maps, output ABIs, gather work and batching matter. Similarly, the PF-5
   32-warp loss closes that in-tree instantiation, not transfer to gfx1151
   in general; the external measured host is also gfx1151.

### 5.4 Why prefill is much farther behind than decode

**Inference from source and the historical profile:** too much prefill work
still follows small-row exact schedules, and the faster arithmetic routes
cover only admitted shapes/layers. Decode has one new row and cannot amortize
weight reads across many tokens. Prefill can amortize dequantization, weight
loads and synchronization across rows and use matrix instructions. A runtime
can consequently be close on decode yet far behind on prefill. This is not
proof that host overhead is zero, but the attributed device owners are large
enough to establish the engineering priority without another GPU run.

Use the historical **exclusive operation roles**, not the mismatched buckets,
for sizing. Source: the section 5.2 canonical profile, exact category fixture,
zbook/gfx1151, UD-Q4_K_XL/BF16; seconds per profiled request. These are
historical owner costs, not predicted savings or a refreshed HB comparison.

| Impact rank | Exclusive owner | p512 | p1024 | p4096 | Main explanation / workstream |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Routed MoE, all quant types and its local routing/combine | 3.408 | 6.307 | 25.398 | Reuse across routed rows; early exact paths versus matrix suffix. PF-3 plus selected PF-1 tails. |
| 2 | Dense linear projections + GR read/tail | 1.839 | 3.442 | 13.864 | Exact Q8 coltile coverage and three-plane MMQ cost. PF-1. |
| 3 | QSA mixer/index/attention | 0.056 | 0.190 | 10.423 | D=256 sparse rows beyond the dense-equivalent budget. PF-2; rank 2 for a p4096-focused unit. |
| 4 | GDN mixer | 0.655 | 1.233 | 4.983 | Serial early layers versus columnwarp suffix. PF-5. |

Routing/materialization are already included in these owners; adding their
family rows again double-counts them. At p4096, even making the entire
historical MoE owner free would yield only about 1.79x complete-wall speedup
in the artifact's Amdahl model. Closing a roughly 4.8x current screen requires
several large owners, not one small fusion or a percent-level tile change.

**MoE: fix reuse and coverage before another tile-number sweep.**
[`run_qwen4_exp_moe`](../hipengine/runtime/qwen4_exp_runner.py) and
[`qwen4_exp_profiles.py`](../hipengine/generation/qwen4_exp_profiles.py) admit
WMMA MoE at layers 27–47; earlier layers retain exact grouped Q4_K/Q5_1
owners. Those exact kernels reuse weights across small row batches, but still
perform scalar dequantized accumulation and ordered reductions. Q5_1 M1 in
[`qwen4_exp_q5_1.hip`](../hipengine/kernels/hip_gfx1100/quant/qwen4_exp_q5_1.hip)
keeps the logical256 tree; its output batch still walks output columns.
Halo-box `mmq.cu`/`mmq.cuh` instead use expert row maps, activation
quantization and matrix tiles; `mmq.cu` also quantizes broadcast gate/up input
once before scattering to expert rows. Compare the full boundaries, including
hipEngine's fused gate/up and BF16 intermediate roundings.

The current runner uses **512-token chunks** versus the comparator's
`-ub 2048` upper bound. At p4096 this means eight hipEngine chunks; HB's actual
ubatches must be read from the trace rather than assumed to equal the bound.
Each hipEngine chunk has 5120 routed token/expert pairs. Historical telemetry
shows a median 325–333 active experts and a median seven rows per active
expert: padding every expert to a large tile wastes work. In a uniform
512-expert thought experiment, `T*10/512` grows from 10 rows/expert at T=512
to 40 at T=2048. That is a reuse opportunity, not the measured distribution.
Explore compact row scheduling, dequantization/metadata reuse and input staging
on broad/uneven expert counts. Budget the larger scratch against 111-GB-class
model residency. Chunk growth cannot explain or fix the p512 gap by itself.

**Dense/GR: two distinct problems, both larger than routing cleanup.**
The exact `gguf_k_prefill_out_coltile_rowbatch_kernel<float,float,8,8,4>` in
[`gguf_k_gemv.hip`](../hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip)
uses 128 logical K lanes, four token rows and eight output columns, repeated
Q8 block decoding and FP32 FMA/reduction. It is a small reuse tile, not the
same arithmetic as the admitted Q8 MMQ route. Its attention-gate shapes remain
outside the production MMQ policy. A bit-exact improvement must preserve the
K subsequences and reduction tree while reducing repeated dequantization and
staging costs. Extending MMQ coverage changes arithmetic and is gated.

The already tiled MMQ owner uses **three residual activation planes**. Its
loop order is `(k_iter, k_half, activation_pass, subblock)`; each plane repeats
activation staging and integer matrix products followed by scaling. Halo-box's
Q8 D4 path uses one plane. This is a real difference in work, not evidence for
an automatic 3x speedup or permission to discard residuals. Preserve planes
and ordered accumulation for T0 reuse work; a different plane count goes
through the declared arithmetic/representation contract in sections 6.2–6.4.
A small number of failed retiles does not exhaust exact dense-kernel designs.

**QSA: a sparse selection can still feed an inefficient attention kernel.**
The binding D=256 rows kernel in
[`qwen4_exp_qsa.hip`](../hipengine/kernels/hip_gfx1100/attention/qwen4_exp_qsa.hip)
assigns one block per query/head, walks selected keys serially, performs a
block reduction plus barriers for each key, then updates online softmax and
weighted V. Queries and the 12 query heads per KV head do not cooperatively
reuse a KV tile. Beyond the approximately 2048-token dense-equivalent budget,
this becomes expensive: the historical p4096 subowner is 9.418 s sparse
attention versus 0.129 s index scoring and 0.023 s top-k expansion.

Halo-box `src/models/qwen4exp.cpp:build_attn_qsa` creates an exact selected-key
mask and calls `build_attn_mha`, using its existing tiled Flash Attention
backend with BF16 KV. That design evidence is available even though PR11 M7
is inactive. A hipEngine T0 design can precompute ordered QK work and stage
KV while preserving each row's arithmetic; a tiled softmax/matrix alternative
is usually T1/T2 and must preserve exact selection/causality plus pass the
profile gate. The prior failed three-pass candidate is not permission to
reintroduce it unchanged or proof that the whole workstream is closed.

**GDN: prioritize the serial prefix, and size the suffix honestly.**
Before layer 27, `qwen4_exp_gdn_prefill_f32_kernel` keeps each value column's
K reduction serial and reads/writes recurrent state through a global pointer
inside the token loop. The later columnwarp owner keeps four state values per
lane in registers and distributes K work across a wave, changing the reduction
relative to the serial owner; only its admitted layer suffix is production.
Tile-16 can amortize repeated Q/K normalization and loads across those column
warps, but neither the Hv=32 screen nor an eventual Hv=48 suffix win applies
to the serial prefix. An exact prefix state-residency design must account for
register pressure/spills and preserve its serial summation order. Expanding
columnwarps to early layers requires a new numerical gate.

### 5.5 Decode: separate residual portfolio

The section 3.0 screen puts short decode within roughly 14–18% of HB's rate.
The historical live-513 aligned deltas versus pinned upstream rank selected
Q5_1 down (3.379 ms/token), selected Q4 gate/up (3.238), Q8 (2.866), GDN
(2.139), then other projections. These are comparator bucket diagnostics,
not additive recoverable wall. Prioritize selected-weight access/dequantization,
reuse of shared inputs, and already-ordered epilogues; do not import large
prefill tiles into a one-row decode owner.

At long context, first revisit the **residual ordered QSA owner**, not the
pre-optimization 35-ms deficit. The main campaign already retained exact
ordered decode at 20.913 ms/token QSA role and 80.061 ms/token complete wall
in its own counterbalanced packet. Reuse KV across GQA heads / reduce repeated
loads only with an actual-owner exact oracle. M7's Q8-KV/GQA=6 route is inactive
under the binding BF16 KV and GQA=12 geometry. Cold PLE, Python submission,
MTP, and concurrency are separate denominators; none presently outranks the
large prefill owners. A wall-minus-kernel subtraction across separate profiler
runs is not evidence that Python or graph capture owns the remaining gap.

## 6. Punchlist

### Start here (2026-09-05): ordered coder work

This order supersedes the 2026-09-03 queue. PF numbers remain historical unit
identifiers, not priority numbers. This review authorizes planning; use the
existing per-unit gates before changing any runtime default. No GPU work was
performed for this review.

| Order | Work through this | Concrete deliverable / stop rule |
| ---: | --- | --- |
| 0 | **Repair the evidence assumptions before choosing a kernel.** Pin actual model dimensions, selected registry owner, layer scope, and profiler grid units. PF-0 already exists. | GDN 16/48/128 RED coverage is now recorded in `617038db9`; verify live candidate invocation rather than registry resolution alone. Add QSA 24/2/256 coverage including the 2048-budget transition; preserve nonzero-state/tile-tail coverage for recurrence changes. Correct the PF-2 causal diagnosis. Keep its failed candidate off. D1 goes to the owner now; it does not block T0 restructuring. |
| 1 | **PF-3 next: operation-complete routed MoE, including selected Q8/Q5_K tails.** Highest prefill owner at every shape. Start with exact early-layer Q4_K/Q5_1 dequantization/input reuse, then layer-2 Q5_K and the remaining selected Q8 down cost. | Separate layers 0–26 from the admitted WMMA suffix and report gate/up → SiLU → down → ordered combine. Preserve M1/grouped-Q8 defaults. For each new T0 mechanism, declare which repeated load/dequantization/barrier it removes; stop after a negative complete-owner result. An early-layer WMMA/MMQ expansion is a separately declared T1/T2 or T3 candidate, never a flag-only promotion. |
| 2 | **PF-1 next: dense/GR compute, restored to high priority.** Split exact F32-activation Q8 coltile projections from the existing three-plane MMQ chain. | First target the actual coltile attention-gate shapes with exact block dequantization and reuse while preserving K ownership/reduction. For MMQ, reduce repeated staging/packing across consumers of the same input, preserving all three planes and accumulation order; report matmul separately from packing and repair. Do not repeat the rejected tile/plane settings. D1 owns any numerical route widening. |
| 3 | **PF-2 next: D=256 sparse prefill attention.** Co-leading owner at p4096; move ahead of order 2 only for an explicitly p4096-focused unit. | Reproduce/localize the old failure on the real shape. Then screen a new tiled QK/KV reuse mechanism preserving selected order, per-row score tree, and online value recurrence. A masked/tiled Flash Attention production alternative requires the full numerical gate. Include selection/mask construction and scratch in the complete owner; score/top-k alone is too small. |
| 4 | **PF-5 next: validate live-route tile-16 A/B, then address the serial prefix.** Finish the existing candidate as a bounded unit if already in progress; `617038db9` completed Hv48 kernel coverage. | Use the Hv48 candidate, instrument invocation counts and fail closed if an after arm calls it zero times. Its first registry-only A/B was a no-op. Compare the suffix against columnwarps in one residency before promotion. Separately target pre-layer-27 serial recurrence with an exact state-residency design, or declare reassociation and pay its profile gate. No extrapolation of the Hv=32 leaf gain to all GDN. |
| 5 | **Batch-size policy within PF-3/PF-1.** Evaluate 512 → 1024 → 2048 prefill chunks after the principal owners support them; develop its static memory/route map alongside orders 1–2. | Record expert-count distributions, actual GGML ubatches, packed scratch/live memory, PLE/KV/state continuity and route changes. One-process A/B must include p512 unchanged and p1024/p4096. Four times more rows is not a promised 4x win, and a chunk-induced route change is not automatically T0. |
| 6 | **PF-4 routing/combine cleanup and decode portfolio (section 5.5).** | PF-4 fused combine still needs one-residency adjudication; reuse the committed A/B harness. Routing must improve total map + gather + consumers. Decode: selected Q4/Q5_1 and Q8 first at short context; residual ordered QSA first at long context. Keep graph/launch work behind a measured exclusive owner. |

Prioritize saved request milliseconds, not percentage improvement in a tiny
leaf. Use `saving = O * (1 - 1/s)` and `new_wall = W - saving` only as a
labeled projection from one exclusive owner. Do not add cross-engine bucket
deltas or apply suffix-only speedups to a whole family. Finite rejected T0
experiments close those mechanisms; they do not establish that all exact
scheduling is exhausted or that parity is achievable without a separately
qualified production arithmetic path.

The following table is the **historical lever ledger**. “Done/rejected” closes
that experiment; the next mechanisms above remain open.

| Phase | Unit | Exit condition |
| --- | --- | --- |
| HB-0 | **Done** — checked out `6c84c7d5` and `a7ad7b7f`, preserved pristine HIP Release binaries, and froze separately labeled loader-patched binaries. Pristine HB-base produced zero samples at the 1,800-second startup timeout; the two documented patches reduced startup to 24.09/21.49 seconds. Both patched lanes completed all four exact p512 categories with matching output hashes. | Identity/smoke artifact: `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb0.json`. |
| HB-1 | **Done** — completed two exact 36-sample arms for each of HB-base, HB-PR11, current hipEngine, and freshly rebuilt patched upstream. Retained arm B shows HB-PR11/base prefill gains of 1.1012x/1.1142x/1.1744x and decode gains of 1.0261x/1.0238x/1.0225x at p512/p1024/p4096. Every lane is cross-arm output-exact. HB-base p512/p1024 prefill drift makes those magnitudes provisional; p4096 direction reproduces. Section 3.4 is explicitly not run because IQ4_XS is excluded by the approved scope. | Artifact: `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb1.json`; proceed to frozen-binary HB-2 profiling. |
| HB-2 | **Done** — profiled frozen HB-base and HB-PR11 over exact p512/p1024/p4096 prefill and live-513/1025/4097 decode. All six pairs are output-exact across lanes and cached decode evaluates one appended token. Kernel-name and launch-geometry census confirms M1, M2, M5 weighted-sum, M6, and M8; denies M3, M4, M5 shared-mul-add, and M7 on this binding graph/payload. The aligned broad-family ledger is diagnostic because kernel sums can overlap. | Artifact: `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb2.json`; HB-3 admits only measured-active subfamilies. |
| HB-3 | **Blocked** — completed the shape/ABI readiness ledger and built pinned base/PR11 `test-backend-ops` binaries (`7af38994…` / `bc640014…`). Available cases pass their own correctness probes, but no active family has an identical operation-complete cross-engine fixture: M1 lacks shared packed weights/dtypes, M2 is internal to `MUL_MAT_ID`, M5 lacks a common F32/BF16 boundary, M6 differs in heads/state/prepare-post ownership, and M8 must split into six stride-aware operations. Timing current surfaces would violate the matching rule. | Block artifact: `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-halo-box-hb3-blocked.json`. Stop for review; no candidate handed off. |
| HB-4 | **Deferred behind PF-1…PF-5 by the HB-3 review** — IQ4_XS diagnostic arm. Artifact already downloaded and hash-verified (section 3.4); when admitted, census M4/J48/J64; a Q8_0 KV diagnostic still cannot activate M7 without its GQA=6 geometry. Document actual activation on each payload. | Diagnostic tables filled; explicit "different quant, does not bind" label on every row. |
| HB-5 | **Deferred behind PF-1…PF-5 by the HB-3 review** — concurrency extension. Fill c=2..8 in sections 3.1–3.3 for every lane that supports it; record `unsupported` honestly; keep thermal windows shared across lanes. | Topline matrix complete or explicitly partial; artifact committed. |
| HB-6 | Port decisions. For each PF unit (and each confirmed deficit): admit (with `W/C/O/s`), defer (named blocker), or reject (measured loss). Update the main campaign's section 4 row and mechanism transfer audit to point at measured rows instead of this doc's hypotheses. | Main campaign cross-references updated; this doc's status line advanced. |
| PF-0 | **Done** 2026-09-03 — natural-text fixture `benchmarks/fixtures/qwen4exp_natural_ar_pf0.json`: 12 cases across all four canonical categories, 535–877 prompt tokens, source/token hashes and construction provenance. The route-coverage finding records 100% engagement and 1548 compared rows; no performance claim. | Fixture is available. The incumbent-vs-strict capture failed calibrated gates; section 6.4 D1 remains an owner decision. No new fixture-construction prerequisite on T0 work. |
| PF-1 | **Done** 2026-09-04 — retained the bit-exact fork-(b) grouped selected Q8_0 down as a production owner, together with PF-3 Q5_1 M1. The kernel remains **47.42→32.42 ms median (−31.6%)** at the p4096 compact shape and the real MoE layer remains **−6.9%/−8.2%/−8.7%** at rows 512/1024/4096. The correcting whole-model packet uses one Python process, one resident generator, one warmup per mode/case, and an ABBA order reversed by adjacent cases: combined PF-1/PF-3 prefill improves **+3.13%/+3.09%/+2.51%** at p512/p1024/p4096, all 12 cases win, and all 72 measured outputs are cross-mode exact. Production selects `selected_grouped_gemv_bf16_bf16_out`; strict keeps `selected_gemv_bf16_bf16_out`. The earlier process-separated and pre-wiring-fix runs remain invalid diagnostics. The dense-projection retile/precision axes remain closed. | Artifact: [`halo PF-1/PF-3 production refresh`](../benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-halo-pf13-production-refresh.json). |
| PF-2 | **Prior candidate unpromoted; cause reopened by source audit.** The 2026-09-03 ordered three-pass route failed p4096 cross-mode output equality; its process-separated timing diagnostic was negative at p512/p1024/p4096. The env flag/runner branch were removed; registered strict variants and their small fixtures remain. The stored wave32 explanation is wrong for binding D=256, and the timing cannot establish a launch-overhead cause. | [Historical artifact](../benchmarks/results/2026-09-03-gfx1151-qwen38-flash-next-pf2-qsa-ordered-prefill-ab.json). Section 5.3 corrects the interpretation; actual-shape RED/localization precedes the new section-6 mechanism. No restoration of the failed route. |
| PF-3 | **Done** 2026-09-04 — Q4_K gate/up M1 remains closed as a bit-exact measured loss (+57% to +123% at rows 16/64/512). The fused single-loop logical256 Q5_1 down remains a **−10.6%** binding-shape kernel win (9,423.0→8,425.6 µs), and the valid combined one-residency PF-1/PF-3 packet restores it as the production exact-grouped-down owner. Production selects `selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out`; strict keeps the preceding expertgrid64 owner. Uneven counts, empty experts, 64/512-expert fixtures, kernel trace, and bit parity remain covered; the superseded process-separated packet is diagnostic only. A later T0 M2 hierarchical-reduction candidate improved the 512-row leaf by 6.99% but failed the binding one-process/one-residency whole-model gate: weighted prefill changed +0.083%/-0.152%/-0.095% at p512/p1024/p4096, with 8/12 cases negative. M2 was removed; M1 remains production. | Artifacts: [`halo PF-1/PF-3 production refresh`](../benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-halo-pf13-production-refresh.json), [`PF-3 schedule A/B`](../benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-pf3-moe-schedules-kernel-ab.json), and [`Q5_1 M2 whole-model rejection`](../benchmarks/results/2026-09-05-gfx1151-qwen38-flash-next-q51-m2-whole-model-ab.json). |
| PF-4 | **Done 2026-09-04 — both levers closed as measured losses.** Lever 1 (`top10_parallel_i64`) is bit-exact but slower at the production gate: +30.8%/+83.1%/+114.6% at rows 64/512/4096; the incumbent 5-op route stays production. Lever 2 (`weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w`) is bit-exact and wins its isolated chain by 26.1%/24.3%/6.1%/2.3% at rows 1/16/64/512, but the binding one-residency counterbalanced A/B loses 1.69% mean prefill with all 12/12 cases negative; decode changes −0.19%. The unfused chain stays production and the fused variant remains an explicit opt-in. | [`routing artifact`](../benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-pf4-m2-group-map-kernel-ab.json), [`fusion kernel artifact`](../benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-pf4-lever2-fused-combine-kernel-ab.json), and [`one-residency whole-model rejection`](../benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-pf4-lever2-fused-combine-whole-model-ab.json). |
| PF-5 | **Done 2026-09-05 — tile-16 promoted.** The w32 variant remains rejected. Tile-16 at `3574a1bd2` covered Hv=32 only; `617038db9` extended it to binding Hv=48 with exact output/state parity. After invalidating a no-op registry-only A/B, a fail-closed engagement-verified one-residency A/B improved weighted prefill 89.435→89.873 (+0.49%), 88.553→88.966 (+0.47%), and 72.661→72.929 tok/s (+0.37%) at p512/p1024/p4096; all 12 cases were non-negative and all 72 trajectories exact. | [w32 rejection](../benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-pf5-gdn-w32-prefill-ab.json), [Hv48 repair](../worklog/entries/20260905T022638.146026Z-lhl-qwen4exp-pf5-gdn-tiled16-hv48-correction-c178a9.md), and [whole-model promotion](../benchmarks/results/2026-09-05-gfx1151-qwen38-flash-next-pf5-gdn-tiled16-whole-model-ab.json). |

### 6.1 PF-1 execution checklist

Historical sub-units of PF-1; the current work order is section 6 above.
The original dense-family re-rank came from the
[`dense-other subowner audit`](../benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-dense-other-subowners.json)
and is corrected by section 5.3. Layer-2 Q5_K WMMA stays rejected on the
production-numerics gate; a new exact Q5_K owner belongs to PF-3.

| Sub-unit | Scope | Exit condition | Status |
| --- | --- | --- | --- |
| PF-1a | Scope and lineage: map the `dense_projection_compute` and `dense_quant_q8` role families to exact in-tree kernels and registry keys; run `scripts/check_lineage.py`; record the halo-box `a7ad7b7f` source files and tile geometry to port. | Mapping table committed to this doc's PF-1 worklog entry; lineage check green; no code change. | **Done** — mapping in [PF-1a entry](../worklog/entries/20260902T233138.374147Z-lhl-qwen4exp-pf1a-dense-scope-199383.md): `dense_quant_q8` = `q8_0_raw_mmq128_q8_1_d4_kernel` (historical 6,926 ms/req; 128×128 output/token tiles, not one block per output column — section 5.3 corrects the entry) + its quantize kernel in `gguf_q8_0_mmq_prefill.hip`; `dense_projection_compute` = `gguf_k_prefill_out_coltile_rowbatch_kernel<float,float,8,8,4>` + selected quant8 down in `gguf_k_gemv.hip`; halo-box pinned at `a7ad7b7f` with I=64/128-row SRAM tile geometry; layer-2 Q5_K stays out. |
| PF-1b | RED oracle: exact-token fixtures plus strict-parity tests for the dense F32 projection (`gguf_k_prefill_out_coltile_rowbatch<float,float,8,8,4>` family) and the dense Q8 activation-quantize + selected Q8_0 down path, green on the current path before any edit. | New tests pass on current kernels; HIP-availability guarded; oracle identity recorded. | **Done** — `tests/test_qwen4exp_pf1_dense_parity.py` (19 tests, green on unmodified kernels): policy-shape MMQ chain determinism + bounded-vs-exact-owner envelope + top-1 at all 7 production shapes; coltile strict bit-parity across 3 variants at 4 attention shapes; selected Q8_0 down vs CPU GEMV oracle at top-10 gather ABI. PF-1c/d variants enter via `MMQ_CHAIN_VARIANTS`/`COLTILE_VARIANTS`/`SELECTED_VARIANTS`. |
| PF-1c | **Blocked** — lever 1 (extend the admitted MMQ128 chain to coltile-served shapes via policy rows) rejected on the canonical route-covering fixture: the P3 shape row `(2560,6144)` retest is deterministic 3/3 (original blocker confirmed stale/harness-scoped) but fails the same bars as d4x2 (incumbent-relative mean 7.27e-4 passes, median 0, p95 5.26e-3 > 5e-3, top-1 98.67% < 0.99, per-scope code/t1/t10/t14) — while keeping the incumbent's exact arithmetic; the attn_output row `(3072,2560)` is closed by measured-analogy inference (same chain/delta class, not separately measured). Lever 2 (Q8_0 selected WMMA) remains open. | **Historical fixture block resolved by PF-0; D1 remains open.** The prior "trajectory vs single-transition basis" blocker is **withdrawn as factually wrong**: both bases are the same 24-step teacher-forced trajectory, and the August harness is durably in-tree (`scripts/qwen4exp_layer2_profile_gate.py`). The real defect is route coverage — the 18-prompt admission fixture dispatches the Q8 MMQ route in only 50 of 450 rows (11.1%), so the 2026-08-29 packet is `route_vacuous_for_scope`. Evidence: [`basis correction`](../worklog/entries/20260903T034205.467427Z-lhl-qwen4exp-pf1-basis-correction-6336a4.md), [`coverage finding`](../benchmarks/results/2026-09-03-gfx1151-qwen38-flash-next-pf1-q8-mmq-route-coverage-finding.json), [`admission-suite reconciliation`](../benchmarks/results/2026-09-03-gfx1151-qwen38-flash-next-pf1-admission-suite-reconciliation.json), [`p3shape retest`](../benchmarks/results/2026-09-03-gfx1151-qwen38-flash-next-pf1c-p3shape-retest.json), [`plane2 dual-basis`](../benchmarks/results/2026-09-03-gfx1151-qwen38-flash-next-pf1d-mmq-plane2-dualbasis.json). | Historical candidate unpromoted; PF-0 done, D1 open. |
| PF-1d | Dense Q8 quantize + Q8_0 down MMQ geometry: port the RDNA3.5 128-wide tile shape from `a7ad7b7f` (M1) and the M8 quantize specializations behind registry variants with strict fallbacks. | Same gates as PF-1c per variant. | **Done** (rejected with complete evidence) — tile retiles measured slower (m64x64 0.89x, m32n128 0.35x; bit-exact but weight-traffic-bound); single-plane d4 fails unit numerics (top-1 99.0-99.6%, max_abs 1.5-3.4); two-plane d4x2 (1.38x isolated) fails dual-basis admission (incumbent-relative mean 7.45e-4 passes, median 0, but p95 5.065e-3 > 5e-3 and per-scope failures on code/t1/t10; deterministic 3/3 everywhere). The d4x3 guarded chain remains incumbent pending D1; these rejected settings do not exhaust structural reuse work. Artifacts: plane2-gate, plane2-dualbasis, pf1d worklog entries. |
| PF-1e | **Blocked** — no arithmetic-changing candidate is currently admissible (PF-1c/PF-1d levers rejected; d4x3 chain and current policy shapes remain production), so there is no retained arithmetic change to whole-model A/B. | Unblocks only if D1's owner decision admits or scope-narrows a candidate; PF-0 already landed and its natural-text incumbent capture failed the calibrated strict gate. | No default change; exact reuse work remains open. |

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
| **T0** | Bit-exact: same reduction order, same intermediates, same output bytes. Tiling, launch geometry, occupancy, LDS/SRAM residency, fusion that preserves order, launch-count reduction, layout changes, graph capture, chunk-size policy only when route/intermediate arithmetic stays identical. | Exact-parity RED vs the registered owner + `rocprofv3 --kernel-trace` + whole-model A/B. **No production-numerics packet.** | PF-1d tile retiles (measured, rejected on speed); M6 warps-per-block and state-in-register; M2 integer routing compaction. |
| **T1/T2** | Local drift or reassociation: different accumulation order, split-K, WMMA accumulate, online softmax merge, precision of intermediates. | Everything in T0 **plus** a coverage-complete section 6.2 packet on a PF-0 fixture: mean/p95/p99/max KL, top-1 by category/shape/transition, three-repeat determinism, BF16-relative where available, task gates. Expect a full gate run per candidate. | PF-1c lever 1 (policy shape rows, rejected); PF-5 if the recurrent K reduction is reassociated; PF-2 if softmax is reassociated; PF-4 fused weighted expert sum. |
| **T3** | Representation/algorithm/decision-policy: activation or weight quantization change, approximate routing, changed acceptance or sampling. | **Not admissible through the drift gate at all.** Needs a declared product-configuration decision, the `EXECUTION-PROFILES.md` section 6.2 BF16-relative and section 6.4 task gates, and its own strict fallback. | Single-plane `d4` (2.40x, rejected as drift; see section 6.2 last bullet). |

Prefer a concrete T0 reuse mechanism when it targets a large owner. Bound the
experiment and stop on a measured loss; do not use “exhaust all T0” as an
indefinite prerequisite to designing a needed production kernel. T1/T2 still
requires the complete unchanged gate, now using the existing PF-0 fixture.

The dense-coltile fork is specific to its historical approximately 2.775 s
p4096 subowner, not the entire mixed bucket's 23–26x ratio. Either build a
faster exact kernel preserving its K/reduction order, or declare an MMQ
coverage expansion with a coverage-complete numerical packet. Existing shape
and plane rejections remain evidence; this review changes no threshold and
admits no rejected arithmetic. Reuse/staging improvements to the existing
three-plane MMQ chain are a separate T0 mechanism.

### 6.4 Open owner decisions (do not resolve these as a coder)

**D1 — does the shipped Q8 MMQ route re-qualify?** PF-0 is available, and the
[natural-text capture](../benchmarks/results/2026-09-03-gfx1151-qwen38-flash-next-pf0-natural-fixture-incumbent-vs-strict.json)
reports incumbent-vs-strict `context_quality`: mean KL 2.01e-3, p95 8.12e-3,
p99 3.07e-2, max 6.17e-2, top-1 99.67%. Its mean/tail/max KL exceeds the
calibrated limits. Code's top-1 is **100%** in this incumbent comparison;
the earlier code-top-1 failure description mixed in the two-plane candidate's
`quality` block. General-Japanese max KL and code mean/p95 KL fail in
`context_quality`. The artifact is explicitly `invalid_or_screen_only`, with
clean-worktree provenance and calibrated quality listed as blockers: this is
an adverse diagnostic, not a valid qualification packet.

The earlier [canonical-fixture diagnostic](../benchmarks/results/2026-09-03-gfx1151-qwen38-flash-next-pf1d-mmq-plane2-dualbasis.json)
is additionally limited by repeated synthetic material. Neither a local
comparison nor its scalar KL predicts the full stacked-manifest drift. The
owner must settle evidence admissibility and choose re-qualification, scope
narrowing, or strict fallback; a clean route-covering packet is required for
any qualification claim. This review changes neither defaults nor thresholds.
T0 restructuring may proceed while D1 is open.

**D2 — scope power.** Section 6.2 keeps sub-25-row scope verdicts binding as
implemented, which means a 12-row transition scope effectively requires 12/12
top-1. The 2026-08-31 accuracy review already recommended a recalibration
campaign for this. Any threshold or scope-size change goes through that
campaign with predeclared rules, never inside a PF unit.

Both decisions are inputs to the owner, not blockers on `T0` work. The full
section-6 queue remains coder-owned; PF-5 and PF-4 are not its only open work.
D1 is urgent because it controls which fast dense routes can legitimately
remain or expand, but no default or numerical threshold changes in this review.

### 6.5 Per-unit definition of done

Every PF unit, in addition to its row's exit condition:

1. Arithmetic class declared (section 6.3) in the worklog entry **before**
   implementation, with the registered strict fallback named.
2. `scripts/check_lineage.py` green; halo-box ports cite path + `a7ad7b7f`.
3. RED test green on the **unmodified** path first, then with the candidate.
4. `rocprofv3 --kernel-trace` showing the expected kernel name and a plausible
   duration; prebuild the `.so` and use `require_cached` (see `AGENTS.md`).
5. Same-session counterbalanced whole-model A/B: one Python process and one
   model residency, with p512/p1024 primary and p4096 recorded. A cycle-wall or sub-window win is retainable even when the
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
  either one carries that caveat explicitly now that PF-0 has landed. Do not quote a
  KL or top-1 number from either fixture without naming which one it came from.
- **Author evidence gaps:** power not recorded, no repeat-arm across
  separately built binaries beyond the logit comparison, one prompt shape.
  Our repeatability classification applies to halo-box lanes exactly as it did
  to Nathan and EngramHalo.
