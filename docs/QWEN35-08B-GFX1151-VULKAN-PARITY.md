# Qwen3.5 0.8B gfx1151 Vulkan-Parity Campaign

Status: D08-C0-C2, D08-M1-M12, accepted D08-P1/P2/P4/P6/D3/D4/D3B/D5, and D08-scoped G1 completed 2026-08-14; fresh G2 parity failed, so G3 and D08-T1 remain blocked. The human-approved D08-X extension then retained X2a pack8 WMMA, X2-K2 Q8 cluster8 GDN, X2-K5 dense-BF16 WMMA, and X3 operation-complete Q4 pack8 gate+up+SiLU on 2026-08-15; X2-K1/K3/K4 closed without production routing. Threshold and cumulative semantic gates pass. A fresh post-X3 synchronized packet confirms Q4 core pp512 at **1.010x same-source llama HIP / 0.889x Vulkan** and cuts the Vulkan wall gap **21.458 -> 11.657 ms**; Q8 core pp/tg is **0.876x/0.888x Vulkan**. The post-X3 owner rerank selected one Q5T16-QKV + pack8-Q4-gate heterogeneous prefill screen; it was byte-exact but only 1.0059x and was removed. The subsequent GDN cluster8 wave-broadcast screen was exact but 0.6483x and was also removed. D08-X6 then retained exact dense-BF16 WMMA down+residual fusion: the 12-owner leaf saves 0.155 ms and five paired Q4 blocks improve core/public pp512 3.09%/1.68%. D08-X7 tested the same rounded boundary in the remaining 12 pack8-Q4 down owners, but two paired blocks lost 4.22%/4.80% core/public pp512; all candidate code was removed. D08-X8 then retained a byte-exact two-wave Q8T16 alpha/beta owner: its 18-pair leaf is 5.010x and paired Q4/Q8 core pp512 improves 3.64%/2.45%. A separate human-approved narrow gfx1100 27B transfer audit retained only the existing Q4T16 c1 dual+SiLU owner. Core Vulkan parity and G3 remain open.

Scope: Qwen3.5-0.8B dense GGUF on Radeon 8060S / `gfx1151`, batch 1,
512-token prompt processing (`pp512`) and 128-step autoregressive decode
(`tg128`). `Q4_K_M` is the primary target and `Q8_0` is the quant-coverage
guard. The external comparator is llama.cpp Vulkan build `1d2869c6e` (build
10415) on RADV STRIX_HALO with flash attention enabled.

This campaign is the 0.8B prerequisite for the later Qwen3.x 27B dense
optimization campaign. Do not transfer a candidate to 27B merely because it
wins a microbenchmark here. The 0.8B route must first complete the semantic
module census, correctness gate, and same-session parity gate defined below.

Related documents:

- [`HIP-vs-VULKAN.md`](HIP-vs-VULKAN.md) — timing-contract and cross-backend
  attribution rules.
- [`STRIX-HALO-LLAMACPP-REVIEW.md`](STRIX-HALO-LLAMACPP-REVIEW.md) — prior
  gfx1151 llama.cpp source review and the rule to select production owners from
  profiles rather than porting every upstream patch.
- [`GGUF-PREFILL-OPTIMIZATION.md`](GGUF-PREFILL-OPTIMIZATION.md) — retained and
  rejected GGUF GDN/prefill schedules. This campaign must not reopen a closed
  schedule without a new 0.8B profile signal.
- [`TUNING-gguf.md`](TUNING-gguf.md) — generic GGUF measurement and tuning
  lanes.
- [`OPTIMIZE-DENSE.md`](OPTIMIZE-DENSE.md) — dense-campaign lane format and
  audit-first precedent.
- [`ROOFLINE-gfx1151.md`](ROOFLINE-gfx1151.md) — 40-CU, ~221 GB/s practical
  read roof, WMMA, cache, and occupancy model.
- [`KERNELS.md`](KERNELS.md), [`TESTING.md`](TESTING.md), and
  [`BENCHMARK.md`](BENCHMARK.md) — kernel catalog, correctness contract, and
  evidence policy.

## 1. Executive objective

Close the current Qwen3.5-0.8B gap to llama.cpp Vulkan in this order:

1. **Certify the actual routes.** The first hipEngine rows used auto bulk
   prefill, eager decode, and explicitly recorded both WMMA prefill and GEMV
   decode as disabled. They are fallback diagnostics, not the fastest
   hipEngine baseline. Measure fallback, forced bulk+WMMA+GEMV, and production
   graph routes before changing a kernel.
2. **Account for every module.** Produce prefill and decode GPU-time ledgers for
   both engines. Assign every kernel/node to a semantic model role and account
   separately for host submission, synchronization, copies, and sampling.
3. **Fix the largest shipped owner first.** A 7-10x prefill gap cannot be
   approached as a tile-width sweep until route selection and the complete
   module ledger rule out a scalar/row-serial fallback. Decode work follows its
   measured Amdahl order, not a generic GEMV checklist.
4. **Match or beat llama.cpp on 0.8B.** Close Q4_K_M `pp512` and `tg128` with
   Q8_0 non-regression and the normal correctness gates.
5. **Only then transfer to 27B.** Re-profile 27B from zero; retain only ideas
   whose 27B owner, shape, and bottleneck reproduce.

The target is not “make one kernel faster than a Vulkan shader.” It is matched
or better end-to-end prompt processing and text generation with a complete
explanation of the remaining wall time.

### 1.1 Impact-ranked active board

Only one implementation owner may be active at a time. After every accepted or
rejected package, recompute the semantic ledger and select the remaining package
with the largest projected whole-request saving.

Potential bands refer to projected end-to-end wall, not isolated leaf speed:

- **critical:** structural route correction or >25% projected request saving;
- **high:** 10-25%;
- **medium:** 3-10%;
- **low:** <3%.

For a leaf speedup `S` on a role owning `role_ms`, calculate the candidate's
upper bound as `role_ms * (1 - 1/S) - added_boundary_ms`. Divide by current
request wall for the impact band. Route changes use measured complete wall,
not a synthetic leaf projection.

| Rank | Package | Current potential | Why it is ordered here | Completion decision |
| ---: | --- | --- | --- | --- |
| 0 | **D08-C0 route matrix** | **completed** | Both opening hipEngine rows disabled WMMA prefill, GEMV decode, and graph replay; changing route invalidated the opening gap magnitudes. | Forced bulk+WMMA+GEMV and graph decode are certified; Q4 remains 4.55x/2.31x and Q8 1.48x/1.45x behind fresh Vulkan. |
| 1 | **D08-M1-M5 full module ledger** | **completed** | Both backends and quants now map every operation to a semantic role; submission residual is explicit and `other=0`. | Q4 linear projections and Q8 GDN are the measured prefill leaders; eager decode is projection-heavy but remains graph-scope-caveated. |
| 2 | **D08-P1 route/default correction** | **accepted; +33.68% pp / +42.19% eager tg** | The existing Q5T16 family replaces 18 expanded BF16 QKV residents for the exact 0.8B role/shape. | Promoted by default after one route repair; no kernel variants were tested. |
| 3 | **Mandatory post-P1 re-profile** | **completed** | P1 invalidated every prior Q4 Amdahl percentage. | M6 reconciles 99.60% of post-P1 prefill wall and supersedes the old ranking. |
| 4 | **D08-P3 dense FFN projections** | **closed/rejected** | All three sole-resident layouts won at pp512, but raw Q4 regressed c1-c8, Q4T16 regressed c8, and Q6T16 regressed c1. | Preserve the evidence; do not duplicate resident weights or trade decode for prefill. |
| 5 | **D08-P2 GDN recurrence** | **accepted: +4.33% paired Q4 pp512** | The Q4/16K/16V shape-scoped cluster8 route cuts marker GDN **67.60 -> 42.83 ms (-36.64%)** and passes the complete semantic/graph-decode gate. | P2 promoted Q4 only after Q8's strict guard missed by 0.0108%; D08-X2-K2 later superseded that Q8 disposition with a fresh retained cluster8 gate. |
| 6 | **Mandatory post-P2 re-profile** | **completed** | The structural GDN route invalidates the P2-era ranking. | Reconciles 99.61% of wall; dense FFN is exhausted, so remaining linear projections are the largest non-exhausted owner. |
| 7 | **D08-P6 remaining linear-attention projections** | **accepted: +14.18% graph-scope Q4 pp512 / +0.69% graph tg128; -46.69 MiB** | The audit selected 35.93-ms Q5 SSM-out; sole Q5T16 direct/rowtile/WMMA wins the complete production route and correctness gate. | Closed after exactly three shipped leaves and one full-model A/B; Q8 and 27B remain unchanged. |
| 8 | **Mandatory post-P6 re-profile** | **completed** | P6 removes 48.96 MB of weights and changes 18 bulk projection owners, invalidating the post-P2 ranking. | M7 reconciles 99.58% of wall, confirms SSM-out at 9.68 ms, and corrects one 1.20-ms Vulkan Q4 role assignment without changing backend totals. |
| 9 | **D08-P7 residual linear-attention projections** | **closed/rejected; Q4 gate bound unrealized** | Native Q4T16 wins pp512 2.006x and exact split-c4x2 wins c8 1.390x, but c1 is 0.883x; raw Q4 regresses every c1-c8 width. | Preserve sole pack8; source-F16 is ineligible after the exact-T16 c1 failure, so no full-model A/B or production change. |
| 10 | **D08-P4 full attention and RoPE/KV** | **accepted: +4.79% graph pp512 / +1.41% graph tg128; -4.13 MiB** | Sole Q4T16 for six source-Q4_K `[N4096,K1024]` Q projections passes all leaf widths, 447/450 top-1, and exact graph/eager trajectories. | Closed with direct c1, rowtile c2-c4, split-c4x2 c8, and WMMA bulk; all other Q4 roles, 27B, and peer backends retain prior owners. |
| 11 | **Mandatory post-P4 re-profile** | **completed** | Six physical weights and their complete Q projection stages changed owner; pre-P4 ranking was no longer authoritative. | M8 reconciles 99.46% of wall, confirms Q at 2.71 ms and T16 WMMA bulk dispatch, and closes every >=1% prefill package as accepted/exhausted or rejected. |
| 12 | **D08-M2 graph/direct census** | **completed: device-critical, not host submission bound** | Q4/Q8 graph launch+Python residual is 0.20%, replay copies are zero, and all 334/288 graph kernels are stage-assigned with exact trajectories. | Close D1; Q4/Q8 remain 1.94x/1.47x behind Vulkan, so admit arithmetic by current graph-stage ownership. |
| 13 | **D08-D3 dense decode projections** | **accepted: +8.29% graph tg128 / +2.28% eager; 24 nodes removed** | Model/shape-qualified fused-SiLU t128 passes 446/450 top-1, max KL 0.002843, exact trajectories, 5/5 decode wins, and identical physical weight bytes. | Closed for gate/up; retain generic c1 unfused, rows>1, Q8, other models/shapes, and peer backends as fallbacks. |
| 14 | **Mandatory post-D3 graph rerank / M9** | **completed: current graph 111.93 tok/s / 310 nodes** | Exact trajectory/zero KL and 97.60% stage coverage show dense matched gap falls 1.676 -> 1.101 ms. | Current full-attention core/KV leads at 1.431 ms / 16.02%; admit D4 ahead of residual projections. |
| 15 | **D08-D4 full-attention core/KV** | **accepted: Q4 +5.97% / Q8 +5.95% graph tg128; zero bytes/nodes** | Exact-shape generic split-K3+fused-gate wins 5/5 graph pairs for both quants, with 448/450 Q4 and 449/450 Q8 top-1, max KL 0.001944, exact trajectories, and neutral pp512. | Closed for the production graph cap514-641 window; retain fixed256, rows>1, 16Q, unsupported shapes/backends, and threshold-zero rollback as fallbacks. |
| 16 | **Mandatory post-D4 graph rerank / M10** | **completed: Q4/Q8 120.62/119.14 tok/s; 310/288 nodes** | Exact trajectories/zero KL, complete node assignment, and 97.27%/97.34% marker coverage reconcile D4's owner movement at -0.512/-0.492 ms. | Q4 dense FFN projections now lead at 1.063 ms / 12.82%; separately marked unworked down+residual is 1.126 ms, so admit only its owner audit. |
| 17 | **D08-D3B dense FFN down projections** | **accepted: +4.13% eager / +1.31% graph tg128; 311 -> 287 recording nodes** | Exact same-resident Q4/dense residual siblings pass 900/900 Q4+Q8 transitions with zero KL; Q4 wins 5/5 crossed-session blocks in both decode scopes, while Q8 selects no fused leaves and stays within 1%. | Closed for exact gfx1151 0.8B Q4_K_M c1 down owners; mandatory post-D3B graph rerank next. |
| 18 | **Mandatory post-D3B graph rerank / M11** | **completed: current graph 120.21/117.80 tok/s Q4/Q8; 286/288 nodes** | Exact trajectory/zero KL and 97.35%/97.32% coverage assign D3B's Q4 dense movement at -0.102 ms; Q8 graph ownership is unchanged. | Larger arithmetic packages are exhausted; admit only D5's exact 24 RMSNorm + 24 add-RMSNorm owner audit at a 0.402-ms / 4.84% joined bound. |
| 19 | **D08-D5 RMSNorm/residual boundary** | **accepted: Q4 graph +2.884% / eager +0.207%; zero bytes/nodes** | Fixed-1024 C wins 5/5 crossed-session graph blocks, passes 900/900 Q4+Q8 transitions at max KL 0.001745, replaces exact 24+24 graph owners, and leaves Q8/prefill neutral. | Closed for exact gfx1151 0.8B Q4_K_M c1 attention/post-attention norm owners. |
| 20 | **Mandatory post-D5 graph rerank / M12** | **completed: current graph 119.88/117.18 tok/s Q4/Q8; 286/288 nodes** | Exact trajectories/zero KL, complete assignment, and 97.23%/97.31% coverage reconcile Q4 norm owners 0.67405 -> 0.56181 ms (-0.11224 ms); Q8 norm movement is +0.00394 ms noise. | All bounded packages are accepted/exhausted or rejected; run G1-G2. Current decode is only 0.596x/0.709x frozen Vulkan, so G2 cannot close without fresh parity. |
| 21 | **Medium/low prefill tail** | **parked: P5 current bound 0.82%** | Every named >=1% prefill package is exhausted under its frozen budget. | Reopen only after a fresh profile raises a complete package above 1% or an exact measured small win is already ready to retain. |
| 22 | **D08-G1-G3 closure** | **blocked: G2 exact Q4 is 0.424x/0.734x core and 0.463x/0.942x public pp/tg** | C1/C2 pass; exact cross-engine top-1 is 645/645 per quant, and Q8 public tg is 1.028x, but Q4 loses every required scope and all 5 blocks. The milestone suite also has unrelated open failures. | Do not close G3 or open D08-T1. Any architectural extension beyond the exhausted bounded packages requires human approval and a new complete-package contract. |
| 23 | **D08-X1 cross-engine rerank** | **completed** | A fresh HIP-stage/Vulkan-op join found per-kernel structure rather than launch count as the gap and opened one bounded extension. | Reconciled X2 ladder: pack8 bulk, GDN, decode GEMV, full-attention c1, and dense-BF16 owners. |
| 24 | **D08-X2a pack8 WMMA bulk route** | **accepted: +35.31% Q4 exact-core pp512** | Existing registered small-tile WMMA replaced tile8x8 for the five qualified p512 pack8 shapes; 447/450 top-1 and max KL 0.003848. | Retained/default on the measured gfx1151 row/shape matrix; sole residency and decode owners unchanged. |
| 25 | **D08-X2-K1 large-tile pack8 WMMA** | **closed; diagnostic not routed** | LDS-staged large tiles are bit-exact but do not beat the qualified small-tile owner on gfx1151 wave32. | Keep the registered 128x64 diagnostic; durable artifact replaces the former `/tmp`-only reference. |
| 26 | **D08-X2-K2 Q8 GDN cluster8** | **accepted: +16.70% Q8 exact-core pp512** | Fresh five-block and 18-prompt gates supersede P2's 0.0108%-miss rejection: 448/450 top-1, max KL 0.003260. | Q4 and Q8 now both use the quant/geometry-qualified cluster8 route. |
| 27 | **D08-X2-K3/K4 decode screens** | **closed; no production change** | K3 found a distributed 1.2-1.4x GEMV grind; K4 corrected marker-inflated attention from 153 to 57 us/layer and only ~0.2 ms/token ROI. | Post-review replay profiling closes the apparent ~2.3-ms gap as isolated-microbench undercount: API/Python residual is only 0.114/0.127 ms Q4/Q8. |
| 28 | **D08-X2-K5 dense-BF16 WMMA bulk** | **accepted: +26.86% Q4 exact-core pp512** | LDS-staged dense WMMA passes 446/450 top-1/max KL 0.004215 and all complete-model guards. | Retained/default for the two measured p512 dense-BF16 shapes; scalar fallback remains for every miss. |
| 29 | **Post-review current-HEAD baseline** | **completed: 4314/4976 tok/s Q4/Q8 exact-core pp512** | Six counter-rotated clean-tree blocks are finite and deterministic with one shared top-1 trajectory across current and X2 controls. | Current Q4 is 1.754x its pre-X2 control and Q8 is 1.175x strict pre-X2; core decode is bimodal, so no fresh decode-speed claim is made. |
| 30 | **Post-review semantic/graph rerank** | **completed: 99.1%/99.0% prefill coverage; 286/288 graph nodes assigned** | Q4 dense FFN is the raw p512 leader; Q8 linear projections lead narrowly over GDN. Production public graph wall is 8.365/8.742 ms Q4/Q8. | No material graph API residual and no K4 reopen. Run threshold and cumulative semantic gates before selecting a new prefill mechanism. |
| 31 | **Post-review p16-p4096 threshold sweep** | **completed: 187/187 fresh processes finite/exact-ID** | Q4 current/pre-X2 is 1.764x only at p512 and 0.997x-1.032x elsewhere; automatic GDN beats strict at every measured Q4/Q8 length. | Keep exact p512 WMMA scope and current GDN policies; no expansion. |
| 32 | **Final cumulative semantic packet** | **completed: 1794/1800 current top-1; max KL 0.005930** | Natural and category-derived p512 profiles pass for Q4/Q8; all repeats and states are deterministic/finite, and 72/72 recording-graph trajectories match eager. | Post-review validation complete; Vulkan G2/G3 parity remains blocked. |
| 33 | **Synchronized exact HIP/Vulkan three-way** | **completed: Q4/Q8 core pp 0.818x/0.880x Vulkan; public tg 0.976x/1.047x** | Six serial blocks use one llama.cpp source revision and explicit backend synchronization; all 36 child rows are finite/deterministic with exact cross-engine core/public top-1 trajectories. | Keep G3 blocked on core scopes. Prioritize Q4 prefill; hipEngine already beats llama HIP decode and Vulkan Q8 public decode. |
| 34 | **D08-X3 operation-complete Q4 pack8 prefill** | **accepted: +13.81% core / +13.85% public pp512** | One 128-thread block owns same-resident gate+up+SiLU, reuses each A fragment, and preserves both BF16 projection boundaries; the leaf is byte-exact and 2.089x. | Five fresh-process Q4 blocks win 5/5; decode and Q8 guards stay within 1.1%. Retain for exact 0.8B/Q4/rows512/K1024/N3584 only. |
| 35 | **Post-X3 synchronized exact three-way** | **completed: Q4 core pp 1.010x HIP / 0.889x Vulkan** | Six clean-HEAD serial blocks confirm the retained route with unchanged same-source llama helpers; all 36 Q4/Q8 children and cross-engine trajectories pass. | The hipEngine-specific core-prefill gap to llama HIP is closed at current variance, but Vulkan retains 11.657 ms. Re-profile Q4 semantic owners before another package. |
| 36 | **D08-X4 post-X3 Q4 owner rerank** | **completed: 100.08% marker-wall coverage** | Dense FFN falls to 34.007 ms; normalized historical gaps now rank linear-attention projections 13.460 ms, GDN 10.641 ms, and dense FFN 8.948 ms. | Admit exactly one heterogeneous Q5T16-QKV + pack8-Q4-gate operation-complete p512 screen; its current explicit fallback stage owns 17.628 ms. |
| 37 | **D08-X4 heterogeneous QKV/gate screen** | **rejected: 1.0059x leaf; no production change** | One 128-thread schedule is byte-exact on actual resident weights but saves only 0.114 ms across all 18 pairs, projecting 0.11% of exact wall. | Stop before profile/full-model gates and remove all transient code. Reopen only for a materially different >=1.10x mechanism; GDN remains next. |
| 38 | **D08-X5 GDN cluster8 wave broadcast** | **rejected: 0.6483x leaf; no production change** | Wave-sharing Q/K/value/beta/decay preserves every output/state bit but regresses the exact rows512 recurrence 0.77453 -> 1.19467 ms. | Stop before profile/full-model gates and remove all transient code. Do not revisit this load-sharing mechanism; dense FFN is next. |
| 39 | **D08-X6 dense-BF16 WMMA down+residual** | **accepted: 1.0158x 12-owner leaf; +3.09%/+1.68% core/public Q4 pp512** | Exact intermediate/output BF16 boundaries remove 12 standalone residual-add launches and save 0.155 ms in the causal leaf. | Retain for gfx1151 rows512/K3584/N1024 dense-BF16 only; primitive chain and temporary env rollback remain. Q8/decode/memory guards pass. |
| 40 | **D08-X7 pack8-Q4 WMMA down+residual** | **rejected: 0.9578x/0.9520x core/public Q4 pp512** | The exact rounded-residual store removes 12 add launches but lengthens the pack8 WMMA critical output path; both paired blocks lose in both prefill windows. | Remove the kernel/registry/capability/selector candidate. Keep pack8 WMMA projection + `gguf_bf16_add`; do not retry without a materially different producer schedule. |
| 41 | **D08-X8 Q8T16 alpha/beta dual WMMA** | **accepted: 5.010x 18-pair leaf; +3.64%/+2.45% Q4/Q8 core pp512** | Two waves share one exact BF16-to-F16 activation tile while independently preserving each N16 Q8T16 projection's WMMA order and BF16 store. | Retain for gfx1151 rows512/K1024/N16+N16 only; singleton WMMAs and temporary env rollback remain. Public pp512 improves 5.05%/2.13%. |

### 1.2 Bounded task contract

Every task records before work starts: semantic owner, baseline time/share,
maximum plausible whole-request saving, exact experiment budget, correctness
gate, accept threshold, reject condition, and revisit trigger. A task cannot
remain indefinitely `in-progress`.

| Task class | Hard experiment bound | Accept rule | Reject / park rule |
| --- | --- | --- | --- |
| Route certification (`C0`) | At most 3 hipEngine routes x 2 quants, 2 supported embedding-placement controls, and 2 fresh llama rows. Each topline row is 1 warmup + 5 measures. No source edit. | Effective route matches the request, correctness passes, and the fastest intended route becomes the certified baseline. | One failed route receives one focused diagnosis. If unresolved, open a named blocker; do not start kernel tuning on an unknown route. |
| Profile (`M1-M10`) | One clean capture per backend/quant/phase; one replacement capture only for incomplete/corrupt output. | 100% node assignment and <=1% timing residual, with API/launch gap separate. | If the tool cannot expose a complete ledger after one repair, record the missing surface and add the smallest instrumentation needed; do not infer owners from names alone. |
| Kernel/algorithm leaf | Audit current lineage first; test at most 3 predeclared variants and one tuning dimension on the actual hot shape. | Any exact, reproducible, non-regressive production win is retained per project policy. Continue to full-model routing only with >=1.10x leaf speed or >=1% projected request saving (or >=0.5 ms/token decode). | Stop after the budget misses continuation, correctness fails, or measured Amdahl falls below 1%. Preserve the result and revisit trigger; remove rejected transient code. |
| Full-model A/B | Only the best admitted leaf; one counterbalanced control/candidate sequence with 1 warmup + 5 measured samples, then the named correctness gate. | Correctness and all guards pass; request wall improves reproducibly. Promote the exact route by default unless a concrete blocker is recorded. | Reject on correctness, route mismatch, or a reproducible guard regression. Do not rescue it with an unplanned compound. |
| Small exact win | No further variant ladder in the same package after the win is measured and retained. | Keep and publish the exact non-regressive improvement even when below the continuation threshold. | Close the package; only a fresh profile may reopen the semantic owner. |
| Expensive follow-up | Obey the repository approval rule before any repeated run expected to exceed five minutes. | User-approved run answers a named unresolved gate. | Park with projected impact and revisit trigger rather than consuming an open-ended benchmark budget. |

The continuation threshold limits exploration; it does not override the project
rule that a measured exact non-regressive win is retained.

### 1.3 Decision states

| State | Meaning |
| --- | --- |
| `accepted` | Correctness and guards pass; a reproducible production win is retained and promoted or has a concrete recorded promotion blocker. |
| `rejected` | The bounded candidate failed correctness/performance/guard gates; transient implementation is removed and evidence remains durable. |
| `parked` | The measured upper bound is too small or a precondition is absent. The ledger names the evidence and exact revisit trigger. |
| `blocked` | External/tool/hardware dependency prevents the declared gate; no unrelated tuning proceeds under that task ID. |
| `superseded` | A later structural route invalidated the old Amdahl premise; old evidence remains historical and is not reused as current projection. |

## 2. Workload and provisional baselines

### 2.1 Model shape

Both GGUFs contain 320 tensors and the same dense architecture:

| Field | Value |
| --- | ---: |
| Layers | 24 |
| Linear-attention / GDN layers | 18 |
| Full-attention layers | 6 (`full_attention_interval=4`) |
| Hidden size | 1024 |
| Dense FFN size | 3584 |
| Query heads / KV heads | 8 / 2 |
| Key length / value length | 256 / 256 |
| Linear-attention inner size | 2048 |
| GDN state size / groups | 128 / 16 |
| Vocabulary | 248,320 |

Tensor inventory:

| File | Tensor types | Encoded tensor bytes |
| --- | --- | ---: |
| `Qwen3.5-0.8B-Q4_K_M.gguf` | F32 133, Q4_K 98, Q5_K 36, Q6_K 17, Q8_0 36 | 521,555,200 |
| `Qwen3.5-0.8B-Q8_0.gguf` | F32 133, Q8_0 187 | 800,881,920 |

The Q4_K_M token embedding is Q6_K. The Q8_0 token embedding is Q8_0.
Embedding placement is therefore part of the route and must be recorded; it
must not be hidden in an environment variable.

### 2.2 External llama.cpp Vulkan reference supplied at campaign opening

Hardware: AMD Radeon 8060S Graphics, RADV STRIX_HALO, UMA, Vulkan flash
attention enabled. Command family:

```bash
cd ~/llama.cpp/llama.cpp-vulkan
build/bin/llama-bench -fa 1 -m <model.gguf>
```

| Quant | llama.cpp pp512 | llama.cpp tg128 |
| --- | ---: | ---: |
| Q4_K_M | **6565.11 ± 540.27 tok/s** | **202.41 ± 2.01 tok/s** |
| Q8_0 | **6586.65 ± 182.93 tok/s** | **165.73 ± 0.48 tok/s** |

These are opening targets, not the closing comparator. The final gate uses a
fresh same-session, interleaved comparison and records clocks, kernel, Mesa,
ROCm, source revisions, and model hashes.

### 2.3 Initial hipEngine diagnostics

| Quant/file | hipEngine pp512 | hipEngine tg128 | Fraction of llama pp / tg | Recorded route |
| --- | ---: | ---: | ---: | --- |
| Q4_K_M | **906.1 tok/s** | **69.8 tok/s** | 13.8% / 34.5% | auto bulk, WMMA off, GEMV off, eager decode; device embedding was reported externally |
| Q8_0 | **660.0 tok/s** | **73.9 tok/s** | 10.0% / 44.6% | auto bulk, WMMA off, GEMV off, eager decode; saved row reports host embedding disabled |

The apparent speedup required from these fallback diagnostics is 7.25x/2.90x
for Q4_K_M prefill/decode and 9.98x/2.24x for Q8_0. Do **not** use those ratios
as an Amdahl plan yet.

The initial rows are explicitly non-canonical:

- `effective_use_wmma_prefill=false`;
- `effective_use_gemv_decode=false`;
- `effective_graph_replay_decode=false`;
- the Q8_0 command omitted `--quant gguf_q8_0`, so its JSON labels the route
  `gguf_q4_k_m` even though the actual file contains only F32/Q8_0 tensors;
- the opening Q4 embedding override and the claimed Q8 host-placement path are
  not consistently represented by the saved temporary JSON.

Campaign step `D08-C0` below reruns both files with exact quant keys and route
provenance; the opening rows remain historical diagnostics only.

### 2.4 D08-C0 route certification (2026-08-14)

C0 ran the bounded fallback / forced-fast-eager / forced-fast-graph matrix with
one warmup and five measurements per hipEngine row. Fresh llama.cpp rows were
run serially on the same GPU; an accidentally concurrent Q4/Q8 pair was
explicitly discarded as contaminated and is not used below.

| Quant | hipEngine fallback pp/tg | Fast eager pp/tg | Fast graph pp/tg | Fresh llama.cpp pp/tg | Remaining llama/hip gap | hip tracked peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | 914.79 / 49.64 | **1427.45** / 49.05 | 1370.39 / **87.12** | 6492.02 / 201.17 | **4.55x / 2.31x** | **1.180 GiB** |
| Q8_0 | 631.85 / 59.65 | **4144.52** / 60.06 | 4137.91 / **114.31** | 6123.47 / 165.32 | **1.48x / 1.45x** | **1.210 GiB graph** |

Rates are tok/s. Q4 uses explicit device embedding because its tied table is
Q6_K. Q8 host/device eager is throughput-neutral within run variance and has
the same 0.959-GiB tracked high-water. Graph capture requires Q8 device
materialization and raises tracked peak by about 0.252 GiB.

Decisions:

- certify forced bulk+WMMA+GEMV for prefill and production graph replay for
  decode;
- admit P1 as a route/default package, but do not implement it before the
  semantic ledger identifies all remaining owners;
- proceed to Q4/Q8 module attribution because every certified row still misses
  the matching Vulkan row materially;
- treat C0 token/finite-logit checks as route sanity, not the final D08-G1
  correctness packet.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-vulkan-parity-c0.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-vulkan-parity-c0.json).

The ROCm 7.15 selected-region control functions live in
`librocprofiler-sdk-roctx.so`, not the legacy `libroctx64.so`; the bench harness
now supports both. Dispatch/resource traces are valid (491 Q4 prefill
dispatches and 334 graph-decode dispatches/token), but rocprofv3 1.3.5 emits
zero kernel durations on this gfx1151 stack even without selected regions.
HIP events are not a substitute on this stack: `hipEventElapsedTime` returned
near-zero intervals and large negative clock-wrap values. M1/M2 therefore keep
rocprof for names/resources and use a same-stream `wall_clock64()` marker kernel
for semantic ownership. A 20-ms CPU sleep calibrated to 20.207 ms, adjacent
markers measured 0.013 ms, and rocprof captured all three marker dispatches
under the expected kernel name (while retaining the global zero-duration
blocker).

### 2.5 HIP semantic attribution checkpoint (2026-08-14)

The repaired profiling-only route records device steady-clock boundaries around
every semantic stage. It is intentionally eager and marker-perturbed, so C0
remains the only topline throughput source. Route-specific prefill keys replace
their generic aliases in the reconciliation sum.

| Quant/scope | Stage sum / instrumented wall | Coverage | Largest roles by stage share |
| --- | ---: | ---: | --- |
| Q4_K_M prefill | 360.34 / 362.13 ms | **99.51%** | linear-attention projections **43.79%**; dense FFN projections **28.00%**; GDN **18.73%** |
| Q8_0 prefill | 130.48 / 132.05 ms | **98.81%** | GDN **38.70%**; dense FFN projections **21.25%**; linear-attention projections **18.79%** |
| Q4_K_M eager decode | 19.36 / 20.05 ms/token | **96.59%** | linear-attention projections **25.75%**; dense FFN projections **24.50%**; full-attention projections/core **19.29%** |
| Q8_0 eager decode | 17.44 / 17.87 ms/token | **97.58%** | dense FFN projections **26.62%**; linear-attention projections **18.86%**; full-attention projections/core **18.02%** |

The Q4 linear-attention QKV/gate and alpha/beta rows explicitly report the
`fallback` route. QKV/gate alone consumes 118.59 ms versus 14.04 ms in Q8,
although Q4 carries fewer encoded bytes. The joined M5 ledger below confirms
P1 as the first prefill package. Its bound stays one route repair followed by
one C0 remeasurement; do not tune arithmetic in P1. Decode remains projection-dominated across both files (linear+dense+full:
**60.1% Q4, 53.2% Q8**), making D3 the leading arithmetic candidate after the
required graph/direct submission census.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-stage-attribution.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-stage-attribution.json).

### 2.6 Vulkan attribution and joined M5 ledger (2026-08-14)

One serial logger capture per quant emitted 131 graphs: one prefill warmup, one
measured prefill, one decode warmup, and 128 measured decode-token graphs. Every
printed operation is assigned by exact matrix shape and architecture call
inventory. Aggregated rows use fixed splits (RMSNorm 24/24/1, GDN/full gate
18/6, and Q8 linear/full output 18/6); logger decimal rounding is below 0.0003
ms. Submission/queue/host wall is a named residual rather than `other`.

| Quant/scope | Vulkan logger / benchmark wall | Coverage | Largest Vulkan roles |
| --- | ---: | ---: | --- |
| Q4_K_M prefill | 77.091 / 80.966 ms | 95.21% | dense FFN projections 33.82%; linear projections 25.10%; GDN 17.37% |
| Q8_0 prefill | 87.872 / 91.807 ms | 95.71% | dense FFN projections 34.32%; linear projections 23.47%; GDN 17.72% |
| Q4_K_M decode | 4.885 / 6.014 ms/token | 81.22% | dense projections 22.44%; linear projections 21.51%; LM head 19.00% |
| Q8_0 decode | 5.992 / 7.139 ms/token | 83.94% | dense projections 26.12%; linear projections 22.17%; LM head 19.78% |

Logger-on rates are diagnostic and regress fresh logger-off C0 by 2.59%/17.35%
(Q4 pp/tg) and 8.93%/15.27% (Q8 pp/tg). Do not publish them as topline.

| Rank | Package | Measured upper bound versus Vulkan | Disposition / hard bound |
| ---: | --- | --- | --- |
| 1 | **D08-P1** | Q4 linear-attention projections: **8.16x**, **38.42%** projected stage saving | **accepted:** sole-resident Q5T16 QKV raises canonical Q4 pp512/tg128 by **33.68%/42.19%**; full re-profile is now mandatory before another owner |
| 2 | **D08-P3** | Q4 dense FFN projections: **3.87x**, **20.77%**; Q8 is already faster | **pre-P1 bound superseded:** M6 admits P3 at a 29.42% projected saving |
| 3 | **D08-P2** | GDN: Q4 **5.04x / 15.02%**, Q8 **3.24x / 26.76%** | **pre-P1 bound superseded:** M6 ranks P2 second at 19.39% |
| 4 | **D08-D3** | eager projection deltas: **47.70% Q4 / 34.75% Q8** | blocked by M2 graph/direct census; eager marker gaps are not production-graph GPU time |
| 5 | **D08-P4/D4** | full-attention roles are material but below ranks 1-3 | future; one route/layout or semantic owner after structural re-profile |
| 6 | **D08-D2** | LM head: only **1.86% Q4 / 1.31% Q8** eager saving | parked; reopen only if graph census changes ownership |

M3-M5 are complete with `other=0`; explicit submission residuals are 3.88/3.93
ms for Q4/Q8 prefill and 1.13/1.15 ms/token for Vulkan decode. This ledger
selected P1 rather than a generic kernel sweep; P1 is now accepted and the
pre-P1 percentages remain historical until the replacement capture.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-vulkan-semantic-ledger.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-vulkan-semantic-ledger.json).

### 2.7 D08-P1 accepted Q5T16 QKV route (2026-08-14)

The route audit found that all 18 Q5_K linear-attention QKV tensors were
expanded to dense BF16 even though the existing direct, rowtile, and WMMA
Q5T16 leaves cover their exact `[6144,1024]` shape. An actual-weight screen
admitted the shipped family without kernel arithmetic changes: c1/c2/c4/c8/
p512 speedups were **2.67x/5.93x/4.23x/1.62x/7.46x**, with BF16-output top-1
agreement of **100%/100%/100%/100%/99.22%** and maximum absolute error no
larger than 0.015625. The gfx1151 materializer now selects one Q5T16 resident
for this exact semantic role and shape; Qwen3.6 `ssm_out` remains excluded.
Native c2-c4 uses rowtile and c5-c8 uses the same-ABI WMMA fallback rather than
calling the c1-only direct leaf with an invalid row count.

The one admitted full-model A/B kept control and candidate sessions resident
simultaneously and alternated five 512/128 eager samples per role. It measured
**1482.31 -> 1982.06 tok/s prefill (+33.71%)** and **48.31 -> 55.43 tok/s
decode (+14.74%)** with identical finite token trajectories. The separate
canonical single-session publication command measured **1427.45 -> 1908.17
tok/s prefill (+33.68%)** and **49.05 -> 69.75 tok/s eager decode (+42.19%)**;
tracked peak fell from **1.180 to 1.043 GiB (-11.59%)**. The larger canonical
decode gain includes the cache/residency benefit that the simultaneous-session
A/B intentionally suppresses.

Control/candidate full logits on the natural fixture preserve top-1 token 220
with **KL 0.000173**. Public generation is deterministic and identical between
roles. A graph smoke remains active and finite at 103.55 tok/s; it is a guard,
not a repeated topline row. P1 is accepted and promoted by default. Its
structural route change invalidates the old Amdahl ranking, so no P2/P3 work
starts before one replacement semantic capture.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-q5t16-qkv-route.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-q5t16-qkv-route.json).

### 2.8 Post-P1 semantic rerank (2026-08-14)

The first post-P1 marker capture was rejected under the one-repair rule because
a 47.83-ms first-use/power-ramp interval was charged to the normally ~1-ms
embedding owner. Its single replacement uses one untimed warmup followed by
one measured capture. The accepted stage sum is **274.75 / 275.84 ms (99.60%)**
for prefill and **17.83 / 18.52 ms/token (96.24%)** for eager decode.

| Post-P1 prefill rank | hipEngine time/share | Matched Vulkan time | Ratio | Projected request saving | Disposition |
| ---: | ---: | ---: | ---: | ---: | --- |
| Dense FFN projections / **P3** | **107.22 ms / 39.03%** | 26.07 ms | **4.11x** | **29.42%** | **admitted-next** |
| GDN / **P2** | **66.87 ms / 24.34%** | 13.39 ms | **4.99x** | **19.39%** | future after P3 |
| Remaining linear projections | **68.98 ms / 25.11%** | 19.35 ms | **3.56x** | **17.99%** | future role-specific audit |

P1 reduced the QKV/gate group from **118.59 to 28.68 ms (-75.82%)** and the
full linear-projection role from **157.79 to 68.98 ms (-56.29%)**. Dense FFN
projection time is now the largest owner and has the largest matched-Vulkan
request bound, so P3 is the sole active implementation package. P2 remains
second; do not compound the two. Decode projections still account for 57.21%
of the measured eager stage sum, but D3 remains blocked until M2 resolves the
production graph/direct scope.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-post-q5t16-rerank.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-post-q5t16-rerank.json).

### 2.9 D08-P3 frozen experiment contract

The route audit freezes one tuning dimension—resident layout plus its already
registered consumer—and no kernel arithmetic sweep. Current Q4_K_M FFN
ownership is:

| Role | Count | Source quant/layout | Actual shape |
| --- | ---: | --- | --- |
| Gate | 24 | Q4_K resident pack8 | `K1024 x N3584` |
| Up | 24 | Q4_K resident pack8 | `K1024 x N3584` |
| Down | 12 | Q4_K resident pack8 | `K3584 x N1024` |
| Down | 12 | Q6_K expanded dense BF16 | `K3584 x N1024` |

At most these three existing in-tree route candidates may be screened:

1. sole Q6T16 for the twelve Q6 down weights;
2. sole Q4T16 for the sixty Q4 gate/up/down weights;
3. sole raw-Q4 WMMA for those sixty Q4 weights.

First compare actual-weight pp512 leaves. A candidate continues only at
>=1.10x leaf speed with finite output and >=90% top-1 agreement. Only the best
admitted layout receives c1/c2/c4/c8 guards and one full-model A/B. Stop P3 if
none projects >=1% complete-wall saving; do not combine layouts in the first
A/B. External lineage checks are currently blocked because the manifest's
`~/amd-gpu-tuning`, nano-vllm, Atlas, vLLM, and llama.cpp-HIP reference trees
are absent on this machine. Therefore P3 may reuse only cataloged in-tree
families; any external port remains blocked until those references are restored.

### 2.10 Origin merge reprofile and D08-P3 closure (2026-08-14)

Merge `41c29b30b` joins local campaign parent `fa46c9d56` with upstream parent
`841f639c6`. After rebuilding changed JIT hashes outside measurement, the exact
1+5 canonical row measured **1938.00 tok/s pp512**, **69.02 tok/s eager tg128**,
and **1.043 GiB** tracked peak. Relative to the pre-merge P1 publication row,
that is **+1.56% prefill**, **-1.03% decode** (inside the five-run spread), and
unchanged memory, with finite logits and identical final IDs.

The warmed merged marker capture reconciles **277.50 / 278.70 ms (99.57%)** of
prefill. Dense FFN remains first at **108.56 ms (39.12%)**, followed by linear
projections at **70.01 ms (25.23%)** and GDN at **67.60 ms (24.36%)**. The
upstream 27B-oriented source-F16/compact routes therefore do not structurally
change this gfx1151 0.8B path.

P3 then consumed its frozen existing-layout budget. All pp512 leaves passed
correctness and continuation: Q6T16 down was **2.87x**, Q4T16 gate/down were
**2.23x/1.23x**, and raw-Q4 WMMA gate/down were **2.57x/1.45x** versus their
production controls. Mandatory sole-resident operational guards rejected each
candidate before a full-model A/B:

- raw Q4 was only **0.22-0.49x** current pack8 at c1-c8;
- Q4T16 won c1-c4 but fell to **0.34x gate / 0.10x down** at c8;
- Q6T16 won c2-c8 but was **0.90x** dense BF16 at c1.

All operational outputs were finite with 100% top-1 agreement and maximum
absolute error <=0.00390625. Duplicate resident layouts would violate the P3
memory contract, while accepting Q6T16 would explicitly sacrifice decode.
P3 is therefore closed without a production change. The same merged ledger
admits P2 next: GDN's matched-Vulkan bound is **54.21 ms / 19.45%** of current
request wall. No P2 implementation may begin before its retained-schedule route
and resource audit.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-origin-merge-p3-reject.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-origin-merge-p3-reject.json).

### 2.11 D08-P2 accepted Q4-scoped cluster8 GDN route (2026-08-14)

The 0.8B GDN geometry has 16 K heads, 16 V heads, and 128x128 state/value
fragments. Production exact LDS32 therefore exposes only **64 one-wave,
32-thread blocks** across 40 CUs, consumes 16 KiB LDS/block, and compiled with
the observed waves/EU target falling from four to two. The newly merged compact
peer path is not a candidate here: one V head per K head means compact Q/K
materialization saves exactly zero bytes.

The bounded actual-shape complete-chain screen reused three cataloged in-tree
schedules. Peer wave32, peer cluster8, and wave32 tree were respectively
**1.53x/1.62x/1.32x** the exact route; all were finite with 100% row top-1,
output NMSE <=1.09e-9, and state NMSE <=1.80e-13. The selected Vulkan-shaped
cluster8 route launches 64 spill-free 256-thread blocks, assigns eight lanes to
each value column, and removes LDS. rocprof records all 18 expected recurrent
dispatches with 96 VGPR and zero scratch/LDS; its gfx1151 timestamps retain the
known zero-duration tool blocker.

One superset-scratch resident A/B measured **2050.24 -> 2138.95 tok/s pp512
(+4.33%, 5/5 pairs)**. All repeated-prompt 128-step trajectories match. The
complete 18-prompt category+heldout gate then records **448/450 top-1 (99.56%)**,
max KL **0.003455**, and non-regressive production graph decode
**20536.58 -> 20526.27 ms (+0.05%)**. The independent default snapshot is
**2050.96 tok/s pp512 (+5.83% versus the merged exact snapshot)** at the same
**1.043 GiB** tracked peak. Its absolute eager-decode row is lower under
independent-run drift; route causality is assigned from the same-session and
complete production-graph gates instead.

The policy is keyed by `(quant, K heads, V heads, K dim, V dim)`, not backend
branches in model code. It promotes cluster8 only for
`(MOSTLY_Q4_K_M,16,16,128,128)` on gfx1151, using the actual GGUF file type
rather than a caller-selected benchmark label. A Q8 candidate diagnostic reaches
**4890.57 tok/s pp512 (+18.00% versus C0)** and passes numerical quality, but
strict graph decode regresses **0.0108%**; Q8 therefore remains on the exact
route and the diagnostic row is rejected rather than published as a win.

The post-route marker capture reduces GDN **67.60 -> 42.83 ms (-36.64%)** and
reconciles **269.40 / 270.45 ms (99.61%)**. Dense FFN is again largest but P3
is exhausted. Remaining linear projections are the next non-exhausted owner at
**74.77 ms versus 19.35 ms Vulkan**, a **20.49%** request bound; D08-P6 is
admitted next.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-q4-cluster8-gdn-route.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-q4-cluster8-gdn-route.json).

### 2.12 D08-P6 accepted Q5T16 SSM-out route (2026-08-14)

The required post-P2 split measures SSM-out **35.93 ms**, residual QKV/gate
**30.43 ms**, alpha/beta **6.33 ms**, and QKV conversion **2.07 ms**. SSM-out
is the largest non-exhausted sub-role: all 18 `[1024,2048]` source-Q5_K weights
were expanded to dense BF16 while the existing Q5T16 family covers the exact
K2,048/N1,024 runtime geometry. P6 froze only the three shipped direct,
rowtile, and WMMA leaves; no new kernel arithmetic or duplicate layout was
allowed.

The actual-weight screen selects direct at c1 and c5-c8, exact rowtile at c2-c4,
and WMMA only for bulk rows. Speedups versus dense BF16 at c1/c2/c4/c8/pp512
are **0.945x/5.848x/4.649x/1.238x/4.097x**. The generic QKV-derived c8 WMMA
fallback is explicitly rejected at **0.419x**; the shape policy keeps QKV c8 on
WMMA but uses direct for exact 0.8B SSM-out. The c1 leaf loss is carried into the
complete gate rather than hidden.

One combined two-resident A/B covers all 18 category+heldout prompts and five
counterbalanced eager and production-graph 512/128 pairs. It records **449/450
top-1 (99.78%)**, max KL **0.003273**, and exact trajectories. Eager pp512
improves **2098.97 -> 2410.75 tok/s (+14.85%, 5/5)**; graph-scope pp512 improves
**2086.23 -> 2382.12 (+14.18%, 5/5)**. Binding graph tg128 improves
**99.29 -> 99.98 tok/s (+0.69%, 5/5)**. The eager tg128 diagnostic moves
**67.02 -> 66.38 (-0.96%, 1/5)** and remains disclosed; it does not override the
non-regressive shipped graph route. SSM-out residency falls **72.00 -> 25.31
MiB (-64.84%)**, reducing all physical weights by **46.69 MiB / 5.49%**.

The policy is exact-role/shape and gfx1151 scoped. Q8 contains Q8_0 SSM-out and
cannot enter the Q5 selector; the Qwen3.6-27B capability and shape remain
separate and disabled on gfx1151. P6 is closed. D08-M7 completes its mandatory
replacement capture, confirms the retained route, and selects the residual
linear-attention group rather than reopening SSM-out.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-q5t16-ssm-out-route.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-q5t16-ssm-out-route.json).

### 2.13 D08-M7 post-P6 semantic rerank (2026-08-14)

One clean post-P6 device-clock capture on `832af97ba` reconciles **232.628 /
233.605 ms (99.58%)** of pp512 wall. Relative to the post-P2 marker snapshot,
complete wall falls **270.454 -> 233.605 ms (-13.62%)**, stage sum falls
**269.398 -> 232.628 ms (-13.65%)**, and SSM-out falls **35.931 -> 9.677 ms
(-73.07%)**. Tracked peak falls **1.0434 -> 0.9978 GiB (-4.37%)** and physical
weight bytes fall by **48,955,392**. Eager decode reconciles **18.351 / 18.998
ms/token (96.59%)**, but remains diagnostic until D08-M2 assigns production
graph ownership.

M7 also corrects a narrow M5 classifier edge case. The Q8-only 18/6 merged-row
rule had been applied separately to Q4's quant-disambiguated K2,048/N1,024
rows. Model inventory proves the Q5_K x18 row is SSM-out and the Q4_K x6 row is
full-attention output. M7 therefore moves **1.197 ms prefill / 0.082 ms/token
decode** from linear attention to full attention; Vulkan total time and complete
backend accounting are unchanged.

Dense FFN remains the largest theoretical gap at **38.78%**, but P3 is closed
because every bounded sole-resident family failed an operational width. The
largest non-exhausted owner is therefore D08-P7: residual QKV/gate, alpha/beta,
and conversion measure **37.940 ms versus corrected 14.439 ms Vulkan**, a
**10.06%** request bound. Accepted P2 GDN remains a 9.77% residual comparison,
and pending P4 full attention is 6.06%. Admit only P7 and split its combined
marker group before selecting a leaf.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-post-p6-rerank.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-post-p6-rerank.json).

### 2.14 D08-P7 residual linear-attention audit (2026-08-14)

Immediate same-stream markers around every shipped target leaf split M7's
**37.940-ms** residual group into QKV **15.440 ms**, gate **11.741 ms**,
alpha+beta **5.965 ms**, conversion **1.801 ms**, and **2.992 ms** of group
boundary/submission gap. The leaf sum covers **92.11%** of the enclosing stage;
the explicit residual is not assigned to arithmetic.

Exact Vulkan quant/shape rows assign QKV **9.585 ms**, gate **3.379 ms**, and
alpha+beta **1.476 ms**. Gate therefore has the largest leaf gap at **8.363 ms /
3.58% of current request wall**, ahead of accepted-P1 QKV at 2.51% and
alpha/beta at 1.92%. Its 18 weights are source Q4_K `[N2048,K1024]`, currently
sole pack8 with `pack8_exact_prefill_tile8x8_bf16_bf16_out`; no prior
exact-role route repair covers them.

Freeze one tuning dimension: sole resident layout plus its existing consumer
chain. The bounded candidates are native Q4T16, raw-Q4 native, and—only if
exact T16 operational guards pass but native bulk misses continuation—the
existing Q4T16 source-F16/rocBLAS lineage. Native T16 screens first with direct
c1, rowtile c2-c4, independently measured
direct/WMMA c8, and WMMA bulk. A route continues only with >=1.10x leaf speed,
>=1% request projection, and non-regressive c1/c2/c4/c8 from one resident
payload. Only the best qualifier receives the one full-model A/B.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-p7-residual-linear-audit.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-p7-residual-linear-audit.json).

### 2.15 D08-P7 Q4 gate route closure (2026-08-14)

Native sole Q4T16 wins the actual Q4 gate at pp512 **2.006x**, c4 **1.205x**,
and c8 **1.390x** when c8 uses two exact c4 rowtile launches. This corrects the
audit plan: dense Q4T16 direct enforces `rows == 1`, so the valid c8 alternatives
are split-rowtile or generic WMMA, not a multirow direct launch. Generic c8 WMMA
is only **0.262x** pack8. Most importantly, exact Q4T16 c1 is **0.883x** and
therefore fails the sole-resident operational guard despite its 2.52% projected
prefill saving and 6.19-MiB residency reduction.

Sole raw Q4 wins pp512 **1.483x** but is only **0.251x/0.287x/0.505x/0.511x**
pack8 at c1/c2/c4/c8. All native outputs are finite with 100% row top-1 and max
absolute difference 0.00390625. The conditional source-F16/rocBLAS route is not
run: it was eligible only if exact T16 passed c1-c8 but native bulk missed, and
it cannot repair the binding c1 owner.

P7 is closed without a full-model A/B or production change. Keep sole pack8 and
do not duplicate layouts. The next non-exhausted package is corrected-M7 P4
full attention at a 6.06% request bound; audit its projection, RoPE/KV, and core
sub-roles before selecting a route.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-q4-gate-routes-rejected.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-q4-gate-routes-rejected.json).

### 2.16 D08-P4 full-attention sub-role audit (2026-08-14)

Three independent same-stream direct-leaf runs split all six full-attention
layers after a complete warmup. The Q projection is the clear owner: six
source-Q4_K `[N4096,K1024]` pack8 weights take a median **7.57 ms HIP versus
2.20 ms Vulkan**, a **5.37-ms / 2.30%** current request gap. The output
projection is **3.73 versus 1.20 ms / 1.09%**, and the mixed K/V projection
family is **3.67 versus 1.24 ms / 1.04%**. Q therefore carries over twice the
matched gap of either remaining eligible projection owner.

The complete KV-write+core/gate package is **3.68 versus 1.82 ms / 0.79%** and
the M7 residual for split/cast/head-normalization/partial-RoPE is **2.43 versus
1.01 ms / 0.61%**. Direct markers further expose split-qgate at median 0.438 ms
and head-normalization/partial-RoPE at median 0.831 ms. Both packages are below
the one-percent continuation threshold, and any attention-core work must in all
cases preserve `KVLiveSpans`.

P4 selects exactly one route/layout owner: sole native Q4T16 for the six Q
projections. Screen actual `blk.3.attn_q.weight` at pp512 and c1/c2/c4/c8,
using direct c1, rowtile c2/c4, exact split-c4x2 or WMMA c8, and WMMA pp512.
Continue only at >=1.10x pp512, >=1% projected request saving, and no operational
regression from the one resident payload; only then spend the one full-model
A/B. No source changes are part of this audit.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-p4-full-attention-audit.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-p4-full-attention-audit.json).

### 2.17 D08-P4 sole-Q4T16 full-attention Q route (2026-08-14)

The actual `blk.3.attn_q.weight` screen qualifies sole Q4T16 at every binding
width: **2.567x pp512**, then **1.411x/1.263x/1.353x/1.360x** at
c1/c2/c4/c8. All outputs are finite with 100% row top-1 and max absolute
difference 0.0078125. Generic c8 WMMA is rejected at 0.352x; gfx1151 backend
policy instead splits physical c8 into two exact c4 rowtile launches. The six
weights use direct c1, rowtile c2-c4, split-c4x2 c8, and WMMA bulk from one
resident payload.

The sole combined full-model A/B passes all binding gates. Paired pp512 improves
**2383.55 -> 2481.63 tok/s (+4.11%, 5/5)** eager and **2380.52 -> 2494.52
tok/s (+4.79%, 5/5)** in production graph scope. Production graph tg128
improves **100.58 -> 102.00 tok/s (+1.41%, 4/5)**. Eager tg128 is a disclosed
diagnostic **67.98 -> 67.33 tok/s (-0.95%, 2/5)**; both eager and graph
trajectories remain exact. Across all 18 category/heldout prompts and 450
teacher-forced transitions, correctness is **447/450 top-1 (99.33%)** with max
KL **0.003574**.

Retain Q4T16 only for the six exact 0.8B Q4_K `[N4096,K1024]` full-attention Q
weights on gfx1151. Their residency falls **18.00 -> 13.88 MiB (-4.13 MiB)**.
All other Q4 roles, Qwen3.6-27B, Q8, and peer backends remain unchanged, and
`KVLiveSpans` is untouched. D08-M8 must now re-capture the canonical semantic
ledger and rerank before another owner is selected.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-q4t16-attn-q-route.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-q4t16-attn-q-route.json).

### 2.18 D08-M8 post-P4 semantic rerank (2026-08-14)

One clean post-P4 device-clock capture on `a34e7b922` reconciles **222.077 /
223.288 ms (99.46%)** of pp512 wall. Relative to M7 immediately before P4,
instrumented wall improves **233.605 -> 223.288 ms (-4.42%)** and assigned
stage wall improves **232.628 -> 222.077 ms (-4.54%)**. This diagnostic agrees
with, but does not replace, P4's binding five-pair full-model result.

The expected semantic owner moves: full-attention QKV/head-normalization/RoPE
falls **13.631 -> 8.681 ms (-36.31%)** and the complete projection+core package
falls **21.623 -> 16.096 ms (-25.56%)**. A direct same-stream split confirms all
six Q projections as sole `gguf_q4_k_t16_v1`, resolves pp512 through
`t16_wmma_prefill_bf16_bf16_out`, and measures Q **7.538 -> 2.710 ms (-64.05%)**.
Weight residency remains **838,835,456 bytes**, exactly 4.125 MiB below M7.

The one allowed rerank closes the named prefill ladder rather than reopening an
exhausted package. P3/P7 remain rejected; P2/P4/P6 are accepted/exhausted. The
only unworked aggregate, P5 glue/norm/activation/input, is **5.521 versus 3.692
ms**, a current **0.82%** matched request bound, and remains parked. Eager decode
markers are diagnostic only. D08-M2 production graph/direct census is therefore
next before campaign closure.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-post-p4-rerank.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-post-p4-rerank.json).

### 2.19 D08-M2 production graph/direct decode census (2026-08-14)

M2 replaces the old zero-duration rocprof graph trace with three independent
surfaces per quant: 128 production graph token timings with graph-launch and
stream-sync API wrappers, a separate exact recording graph, and a graph-captured
same-stream wall-clock marker around every semantic stage. The instrumented
graph assigns every non-marker node between boundaries with no prefix/suffix
residual: **334 Q4** and **288 Q8** kernel nodes. Marker-subtracted device-stage
coverage is **97.64% Q4 / 97.52% Q8** of instrumented host wall. Absolute role
times below are share-normalized to the directly measured production sync span;
production total/API walls and node counts remain direct measurements.

Production graph Q4 is **9.646 ms/token / 103.67 tok/s** versus Vulkan **4.971
ms / 201.17 tok/s**, a **1.94x** remaining gap. Q8 is **8.905 ms / 112.30
tok/s** versus **6.049 ms / 165.32 tok/s**, a **1.47x** gap. Graph launch plus
Python residual is only **0.20%** of wall for both quants, each token has exactly
one launch and one synchronization, and replay performs zero copies. The sync
span contains the device-critical graph; D1 redundant submission/sync/copy work
therefore closes without a candidate. Graph allocation adds zero tracked bytes
and only 217,088/204,800 sampled HIP bytes for Q4/Q8. All 128-token eager/graph
trajectories are exact, all final KL values are zero, and top-1 is 100%.

The graph-stage rerank admits D3. Q4 dense FFN projections are **2.772 versus
1.096 ms Vulkan**, a **1.676-ms / 17.37%** request bound; linear-attention and
full-attention projections add 9.70% and 3.35% bounds. For Q8, full-attention
core/KV leads at **1.584 versus 0.163 ms / 15.96%**, while dense FFN projections
remain material at 7.57%. D3 goes first because Q4 is primary and dense FFN is
its largest matched role; D4 is second. D2 remains parked because the normalized
LM-head/sampler role is already no slower than Vulkan for both quants.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-graph-direct-census.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-graph-direct-census.json).

### 2.20 D08-D3 dense decode owner audit (2026-08-14)

The primary-Q4 dense route has one exact gate/up shape: 24 physical
`[N3584,K1024]` Q4_K pairs, all solely resident as pack8. Their combined
**132,120,576 bytes** form a **3.9375x MALL** cycling pool. Current production
issues 24 dual-t32 projection kernels and then 24 separate BF16 SiLU kernels.
The M2 production-sync join assigns **1.626 ms** to gate/up and **0.220 ms** to
SiLU, an operation-complete **1.846-ms / 19.13%** wall component. Down is split
between 12 Q4 pack8 and 12 source-Q6 expanded-BF16 residents; its complete
projection+residual span is smaller at 1.146 ms.

The screen is now frozen before qualification: the control is existing dual-t32
plus separate SiLU, and the only three candidates are the existing same-resident
fused-SiLU leaf at t32, t64, and t128. No repack, sidecar, duplicate resident,
workspace, hot-path scratch, or new kernel body is allowed; the unfused chain
remains the mandatory fallback. Only a finite/correct candidate projecting at
least 1% production-graph saving may enter one full-model gate. The kernel
lineage command is currently blocked by an absent optional external Atlas
checkout, but no port or new body is in scope.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-dense-decode-audit.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-dense-decode-audit.json).

### 2.21 D08-D3 dense gate/up leaf screen (2026-08-14)

Twenty rotated and counter-reversed same-stream samples each cycle all 24
physical Q4 gate/up pairs (**132.12 MB / 3.9375x MALL**) per observation. The
current operation-complete dual-t32 plus separate BF16 SiLU control is **1.7891
ms**. Existing same-resident fused-SiLU t32 is **1.7148 ms / 1.043x** but
projects only **0.80%** graph saving and fails the 1% gate. T64 reaches **1.2654
ms / 1.414x / 5.60% projected**, but is dominated under the same contract.

Fused-SiLU t128 wins at **0.9955 ms / 1.797x**, projecting **0.819 ms / 8.49%**
of current production graph wall. All 24 actual-weight outputs are finite with
**24/24 top-1**, max absolute delta **6.10e-5**, and mean absolute delta
1.65e-9 versus the current BF16-boundary control. It uses the same sole pack8
residents and no workspace. Q8 remains unchanged: all 24 pairs use a separate
Q8T16 registry key and cycle 187.17 MB, so the Q4 candidate cannot match.

Only t128 enters RED/GREEN and one combined full-model gate. No production code
or runtime behavior changes in this decision unit.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-dense-decode-screen.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-dense-decode-screen.json).

### 2.22 D08-D3 retained fused dense decode route (2026-08-14)

The qualified schedule is now the default only for
`(hip_gfx1151, linear_pair_silu, gguf_q4_k,
pack8_dual_decode_t128_bf16_bf16_out)` at Qwen3.5-0.8B Q4 c1
K=1,024/N=3,584. It reuses the existing sole pack8 residents and fused-SiLU
body, adds zero persistent/hot-scratch bytes, and retains generic c1 unfused,
rows>1, Q8, other models/shapes, and peer backends as fallbacks.

The sole combined full-model A/B passes. Production graph tg128 improves
**101.86 -> 110.31 tok/s (+8.29%, 5/5)**; eager tg128 improves **65.26 ->
66.75 (+2.28%, 5/5)**. Graph replay falls **335 -> 311 nodes**, exactly the
expected 24-launch reduction in every pair. Prefill is a same-route neutrality
control: eager is **+0.48%** and graph is **-0.29%**, inside the frozen 1% guard.
Physical weight bytes are identical at **838,835,456** and gate/up allocations
remain 132,120,576 bytes.

All eager and graph trajectory pairs are exact. The full 18-prompt
category+heldout gate records **446/450 top-1 (99.11%)** and max KL **0.002843**;
repeated-prompt final graph top-1 is exact with KL 1.03e-6. D3 is retained. Its
24-node/8.29% topology change makes M2 stale, so mandatory current graph rerank
M9 precedes D4 or any residual projection work.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-dense-fused-decode-retained.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-dense-fused-decode-retained.json).

### 2.23 D08-M9 post-D3 graph/direct rerank (2026-08-14)

The current Q4 production graph has **310 kernels**, exactly 24 fewer than M2,
and runs at **8.934 ms/token / 111.93 tok/s** in the single-session census
versus pre-D3 **9.646 ms / 103.67 tok/s**. Graph launch plus Python residual
remains only **0.25%**, replay copies remain zero, and stream synchronization
contains the device-critical span. The separately instrumented graph covers
**97.60%** of host wall after marker subtraction, with no nodes outside semantic
boundaries. Its recording/eager trajectory is exact and all final KL values are
zero. Graph capture adds zero tracked bytes.

The current Vulkan join changes the owner order. Full-attention core/KV remains
**1.590 ms HIP versus 0.158 ms Vulkan**, a **1.431-ms / 16.02%** current request
bound, and is now first. Dense FFN falls from a 1.676-ms matched gap before D3
to **1.101 ms / 12.32%**; linear-attention projections follow at **0.947 ms /
10.60%**. D4 therefore becomes the sole next audit, bounded to the six complete
paged-attention+gate owners and at most two variants. Q8 is not rerun: D3's exact
Q4 model/file/quant/shape key cannot match its Q8T16 route, so its M2 evidence
remains unchanged.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-post-d3-graph-rerank.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-post-d3-graph-rerank.json).

### 2.24 D08-D4 full-attention core/KV owner audit (2026-08-14)

The production graph fixes its attention cap at **641** while complete
`KVLiveSpans` advance from live context **514 through 641**. Each of
full-attention layers **3/7/11/15/19/23** writes one 2KV-head/D256 BF16 K/V row,
launches the gfx1151 short-context registry owner
`qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans` at
rows1/8Q/2KV/D256, then launches a separate 2,048-element BF16 sigmoid gate.
There is no current split-K: per layer the write is 2x256 threads, attention is
8x256 with 9,779 bytes dynamic LDS at cap641, and gate is 8x256. The fixed256
leaf is the already-retained precomputed-offset/value-vector2 body, not a new
D4 candidate.

Device-clock boundaries around all 18 leaves, inside the actual 128-transition
production graph, measure a marker-adjusted package mean of **1.659 ms**:
attention **1.553 ms (93.60%)**, KV write **0.060 ms**, and gate **0.047 ms**.
The standalone write and gate are only **0.67%/0.52%** of current 8.934-ms
request wall, so neither can independently clear the one-percent continuation
gate. The route's existing split scratch is only **24,768 bytes** and is already
resident; D4 adds no bytes.

Freeze exactly two existing 8Q-compatible candidates. Candidate A is the
generic context-batch 1,024-thread leaf plus the same gate. Candidate B is
generic split-K with chunk256/three splits (24 producer blocks) plus its fused
8-block BF16 gated reducer. Warp/grouped-GQA leaves are ineligible because they
require the parent 16Q/2KV/D256 shape. A candidate must save at least **0.0893
ms (1% request wall)**, win at live contexts 514/576/641, and pass finite,
KL/top-1, `KVLiveSpans`, state, full-logit, and trajectory gates. Otherwise D4
closes without implementation. Prior broad wave/reduction and direct/split-
threshold sweeps remain closed.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-full-attention-core-audit.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-full-attention-core-audit.json).

### 2.25 D08-D4 bounded existing-leaf screen (2026-08-14)

Three counterbalanced 128-transition production-graph runs, five rotated
marker-graph runs, and one recording/state graph per route consume the complete
two-candidate budget. The generic 1,024-thread context leaf regresses graph wall
**8.999 -> 9.101 ms/token (+1.14%)**, is **1.262x slower** across the complete
six-layer package, and fails the 576/641 context guards. It is rejected.

Generic split-K3 plus its existing fused BF16 gated reducer wins all gates. The
complete package falls **1.777 -> 1.207 ms (1.472x; -0.570 ms)**, while
production graph wall falls **8.999 -> 8.483 ms/token (-0.516 ms / -5.73%,
3/3 wins)** at the same **310 nodes** and zero new persistent or hot-scratch
bytes. Package savings at live contexts 514/576/641 are respectively
**7.33%/34.46%/41.81%**. The candidate uses the already-resident 24,768-byte
split scratch and changes no KV format.

The repeated-prompt 128-step trajectory is exact; final top-1 is exact with KL
**4.33e-7**. `KVLiveSpans` base/live metadata is byte-exact and every captured
KV/linear-state component is finite. Alternate reduction order propagates
expected state hashes after the first full-attention layer, so a separate
8Q/2KV/D256 BF16 CPU-reference fixture checks the primitive at contexts
514/576/641: all outputs are finite, head top-1 is 100%, maximum absolute error
is **7.61e-6**, and split versus current differs by at most **9.54e-7**.

Only generic split-K3 enters RED/GREEN and one combined Q4+Q8 full-model gate.
The production policy must be exact gfx1151 short-context 8Q/2KV/D256 shape,
retain fixed256 below/when unsupported, preserve rows>1 and 16Q routes, and add
no backend/quant branch to runtime dispatch.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-full-attention-core-screen.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-full-attention-core-screen.json).

### 2.26 D08-D4 retained short-context split-K3 route (2026-08-14)

RED/GREEN adds one gfx1151 package capability keyed to the exact 0.8B attention
shape `(hidden=1024, layers=24, 8Q, 2KV, D256, block256)` and live-context cap
**514-641**. The runtime remains quant-neutral and asks the backend capability
whether to use generic split-K; no backend or quant branch enters model
dispatch. The candidate uses the existing chunk256/three-split producer and
fused 8-block BF16 gated reducer. It adds zero persistent or hot-scratch bytes,
reuses the existing 24,768-byte split scratch, and leaves graph node count
unchanged. Cap513 and cap642-1023 retain fixed256 plus the separate gate; the
established generic long-context split remains at >=1024. Rows>1, 16Q,
unsupported shapes, gfx1100/peer backends, BF16-mirror diagnostics, and the
threshold-zero rollback retain their prior routes.

The combined gate validates Q4_K_M and Q8_0 independently. All 18
category+heldout chat prompts per quant are deterministically self-tiled on the
left to the target 512-token context while preserving each complete natural
prompt as the final suffix; 24 teacher-forced decode steps then exercise the
new context window. Q4 records **448/450 top-1 (99.56%)**, max KL **0.001494**;
Q8 records **449/450 (99.78%)**, max KL **0.001944**. Every repeated-prompt
128-step eager/graph trajectory is exact, and graph inventory changes exactly
six fixed256+gate pairs to six split-K3+fused-gate pairs at the same **311 Q4 /
289 Q8 recording nodes** (**310/288** without the recorder).

Production graph Q4 tg128 improves **109.48 -> 116.01 tok/s (+5.97%, 5/5)**,
saving **0.514 ms/token**; eager improves **66.08 -> 67.44 (+2.06%, 5/5)**.
Graph pp512 is neutral at **2493.80 -> 2493.18 tok/s (-0.025%)**. The first Q8
fixed-session run already improved graph tg128 **110.02 -> 116.42 (+5.82%,
5/5)**, but the physical session permanently labeled candidate was also 1.36%
slower in untouched prefill and therefore failed the frozen neutrality guard.
Per the focused-repair rule, the completed 18-prompt correctness evidence was
preserved and only Q8 performance was rerun with both physical sessions
executing both roles inside every pair. The repaired isolation records graph
**110.43 -> 117.00 tok/s (+5.95%, 5/5)**, eager **73.81 -> 74.03 (+0.30%,
4/5)**, and graph prefill **-0.47%**, inside the 1% guard. D4 is retained for
both quants. Its arithmetic-owner change makes the M9/M2 stage shares stale, so
D08-M10 is mandatory before another package opens.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-full-attention-splitk3-retained.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-full-attention-splitk3-retained.json).

### 2.27 D08-M10 post-D4 graph/direct rerank (2026-08-14)

One clean current-production capture per quant repeats M9/M2's three independent
surfaces: 128 direct graph token/API timings, a separately recorded trajectory,
and a graph-captured same-stream stage ledger. Both eager/recorded/production/
instrumented final logits are exact with zero KL and 100% top-1. Production
inventory remains **310 Q4 / 288 Q8 nodes**, contains exactly six generic
split-K3 producers plus six fused gated reducers per quant, and performs zero
replay copies. Every non-marker node lies inside a semantic boundary. Raw
marker-stage coverage is **97.27% Q4 / 97.34% Q8** of the instrumented host
wall; marker-subtracted shares are normalized to the directly measured stream
sync, with the **0.268% / 0.260%** launch+Python residual explicit. Graph capture
adds zero tracked bytes.

The direct single-session census measures Q4 at **8.291 ms/token / 120.62
tok/s** and Q8 at **8.394 ms / 119.14 tok/s**. These current snapshots do not
replace D4's counterbalanced binding improvements of **+5.97% / +5.95%**. They
leave hipEngine **1.668x Q4 / 1.388x Q8** behind the frozen Vulkan rows. D4's
intended semantic movement reconciles directly: full-attention core/KV falls
**1.590 -> 1.078 ms (-32.19%)** for Q4 and **1.584 -> 1.092 ms (-31.06%)** for
Q8, matching about 0.51/0.49 ms of removed device-critical wall at unchanged
node count.

Current primary-Q4 ranking is dense FFN projections **2.159 vs 1.096 ms Vulkan
(1.063-ms / 12.82% bound)**, full-attention core/KV **0.920 ms / 11.09%**, and
linear-attention projections **0.918 ms / 11.07%**. Q8 full-attention core/KV
remains largest at **0.929 ms / 11.07%**, but D4 has just closed its exact
two-existing-route budget. Within the Q4 dense owner, accepted fused gate/up is
**1.033 ms** while the separately marked, unworked down+residual stage is
**1.126 ms**. M10 therefore admits only D08-D3B's owner audit: split Vulkan's
aggregate dense role by exact gate/up versus down shapes and split HIP down
between its 12 Q4-pack8 and 12 Q6-dense owners before freezing any candidate.
No down-only Vulkan gap is inferred from the aggregate. D5 remains behind the
three leading arithmetic packages.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-post-d4-graph-rerank.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-post-d4-graph-rerank.json).

### 2.28 D08-D3B dense-down owner audit (2026-08-14)

The exact current owner is 24 c1 dense down projections followed by 24 explicit
BF16 residual adds. Twelve source-Q4_K layers use sole pack8 residents at
**33,030,144 bytes** total and dispatch 12 grids of `128x1` blocks at t32.
Twelve source-Q6_K layers use dense-BF16 residents at **88,080,384 bytes** and
dispatch 12 grids of `1024x1` blocks at t256. The residual boundary is 24
`4x1` grids at t256. The complete physical down pool is **121,110,528 bytes /
3.61x MALL**; no sidecar or duplicate layout is present.

One focused replacement capture corrects the first name-wide inventory by
placing device-clock boundaries around each of those exact 48 production graph
intervals. All 12/12/24 target intervals map to the expected kernel and launch
geometry, the 128-step marked/control trajectories and logits are bit-exact,
and the marker-adjusted split is normalized to M10's directly measured
**1.126-ms** down+residual stage: Q4 pack8 projection **0.346 ms**, Q6 dense
projection **0.589 ms**, and residual adds **0.191 ms**.

The frozen 128-block Vulkan log supplies an exact operation-complete join rather
than an aggregate estimate. Its 48-call Q4 gate/up row is **0.668 ms**. Its
12-call Q4 and 12-call Q6 down rows are already `MUL_MAT_ADD` and measure
**0.191 / 0.238 ms**, or **0.429 ms** combined. The matched down+residual gap is
therefore **0.698 ms / 8.42%** of current Q4 graph wall. Equal-count allocation
of HIP's residual cost gives diagnostic owner bounds of **3.03% Q4 / 5.39%
Q6**; only the aggregate package gap is binding.

Prior P3 repacks remain closed: Q6T16 down failed c1 at **0.900x**, and D3B adds
no resident representation. Freeze exactly three candidates: (A) current Q4
pack8 t32 fused with rounded-BF16 residual, (B) current Q6 dense-BF16 t256 fused
with the same boundary, and (C) their combined package. Each fused leaf must
round the projection to BF16 before adding the BF16 residual and round the sum
to BF16, bit-exact with projection plus `gguf_bf16_add`; the unfused chain
remains registered. No repack, sidecar, thread sweep, persistent bytes, or hot
scratch is allowed. Only exact combined C with >=1% graph saving proceeds to
one full-model gate.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-dense-down-audit.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-dense-down-audit.json).

### 2.29 D08-D3B bounded fused-residual screen (2026-08-14)

RED first requires two registered four-axis siblings and bit-exact GPU results
against projection plus `gguf_bf16_add` and a CPU rounded-BF16 boundary oracle.
GREEN adds no new layout: Q4 reuses the current pack8 t32 body, Q6 reuses the
current dense-BF16 t256 body, and each fused store rounds projection to BF16,
adds BF16 residual in FP32, then rounds the sum to BF16. All 24 actual-weight
outputs are bit-exact for Q4-only A, Q6-only B, and combined C; every output is
finite. The unfused primitive chain remains registered.

The sole screen runs all 24 current down weights in model order per route,
cycling **121,110,528 bytes / 3.61x MALL**. Five blocks of four counter-rotated
orders provide 20 same-stream device-clock observations per route after one
warmup and marker-baseline subtraction. Control is **1.1033 ms** median.
Q4-only A is **1.0048 ms / 1.098x**, projects **0.101 ms / 1.21%** graph saving,
and wins 18/20 samples plus 5/5 balanced blocks. Q6-only B is **0.9855 ms /
1.119x**, projects **0.120 ms / 1.45%**, and wins 19/20 plus 5/5.

Combined C is selected at **0.9420 ms / 1.171x**, with **20/20 sample and 5/5
balanced-block wins**. Normalized to M10, it projects the down stage **1.126 ->
0.962 ms**, saving **0.165 ms / 1.99%** of current graph wall and reducing the
production graph projection from **310 -> 286 nodes**. It adds zero persistent
or hot-scratch bytes. No thread/layout variant was tested.

A cached-build `rocprofv3 --kernel-trace` smoke confirms both expected `true`
fused kernel names and exact t32/t256 geometries. This gfx1151/TheRock profiler
still emits equal start/end timestamps, the known M2 zero-duration tool blocker;
the screen's device-clock observations provide plausible timing. Only combined
C enters RED/GREEN production binding and one Q4 full-model plus Q8 guard gate.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-dense-down-screen.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-dense-down-screen.json).

### 2.30 D08-D3B retained full-model gate (2026-08-14)

Production RED/GREEN binds combined C through
`GGUF_DENSE_DOWN_RESIDUAL_DECODE_POLICIES` only for gfx1151, the immutable
24-layer/H1024 Qwen3.5 geometry, `MOSTLY_Q4_K_M`, and c1 `[K3584,N1024]`.
The Q4 pack8 normal owner derives the `pack8_bf16_residual_bf16_out` sibling
through its pack8 ABI;
the dense-BF16 normal owner derives `out_bf16_residual_bf16_out` through the
dense ABI. Policy resolution is hoisted to resident-runner initialization so
Q8 and other unsupported owners pay no repeated per-layer package lookup.
Every policy, ABI, allocation, or registry miss fails closed to the existing
projection plus `gguf_bf16_add`; native rows 2-4 keep their prior registered
residual routes.

The completed semantic gate covers all 18 category/heldout prompts and 24
teacher-forced transitions per prompt for both quantizations. Q4_K_M and Q8_0
are each **450/450 top-1, KL 0, and trajectory exact**. The first fixed-label
performance pass exposed the same physical-session bias already documented by
D4, so its correctness and graph mechanics are retained while its performance
labels are superseded. The binding repair counter-rotates both roles across
both simultaneously resident physical sessions in every one of five blocks,
providing ten observations per role without repeating correctness.

On pp512/tg128, Q4 eager decode improves **68.38 -> 71.20 tok/s (+4.13%, 5/5
balanced blocks)** and production graph decode improves **116.21 -> 117.73
tok/s (+1.31%, 5/5)**. Eager/graph prefill improve **+0.66%/+0.40%**. The
recording graph changes **311 -> 287 nodes**: all 24 standalone down adds vanish
and exactly 12 Q4 plus 12 dense fused leaves appear. Persistent and hot-scratch
byte deltas remain zero.

Q8 is an explicit no-route guard: its recording graph remains **289 nodes**
with zero fused leaves. Eager/graph decode are **0.9945x/0.9985x**, while
prefill is **1.0050x/1.0015x**; all stay within the 1% guard and trajectories
are exact. D3B is retained as the default exact owner. As with every accepted
decode package, its arithmetic and node movement require one mandatory current-
graph rerank before D5 or another package opens.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-dense-down-residual-retained.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-dense-down-residual-retained.json).

### 2.31 D08-M11 mandatory post-D3B graph rerank (2026-08-14)

M11 repeats M10's exact current-production protocol: one Q4_K_M and one Q8_0
resident session each run eager, recording, production, and separately
instrumented pp512/tg128 paths for 128 transitions. Both trajectories are exact;
recording, production, and instrumented logits have **KL 0 / top-1 100%**.
Marker-subtracted device stages cover **97.35% Q4 / 97.32% Q8** of instrumented
host wall, all non-marker nodes are assigned, replay contains zero copies, and
tracked graph bytes remain zero. Launch plus Python residual is only
**0.255%/0.272%**.

Current direct production snapshots are **120.21 tok/s Q4 / 117.80 tok/s Q8**
at **286/288 nodes**. Against M10, Q4 nodes move **310 -> 286** while Q8 stays
**288 -> 288**. Q4's dense-FFN normalized stage falls **2.159 -> 2.057 ms
(-0.102 ms / -4.71%)**, with the fused down+residual sub-window now **1.011 ms**
versus **1.126 ms** at M10. Q8 dense ownership moves only **+0.010 ms**, inside
marker noise. Q4 aggregate graph wall is **+0.34%** slower than the earlier M10
snapshot despite the crossed-session binding gate's +1.31%; this is disclosed
cross-run variance. The exact sub-window reduction and 24-node removal are the
binding D3B reconciliation.

The current Q4 joined gaps are dense FFN **0.961 ms / 11.55%**, full-attention
core/KV **0.946 ms / 11.37%**, and linear-attention projections **0.941 ms /
11.31%**. Those larger roles are exhausted respectively by accepted D3+D3B,
accepted D4's two-route budget, and accepted P2/P6 plus rejected P7 operational-
width routes. They do not reopen from an aggregate rerank.

M11 therefore admits only the promised D5 boundary audit. The still-open Q4
family is 18 linear-attention plus 6 full-attention standalone RMSNorm nodes
(**0.332 ms current; 0.224-ms / 2.69% joined gap**) and 24 already-fused post-
attention add+RMSNorm nodes (**0.342 ms; 0.179-ms / 2.15% gap**). Their joined
bound is **0.402 ms / 4.84%**, but no combined candidate is inferred. Audit the
exact widths, thread schedules, per-family node/device cost, and Vulkan boundary
semantics first; only same-resident schedules or operation-complete fusions may
enter one bounded screen. Q8 remains a required guard.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-post-d3b-graph-rerank.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-post-d3b-graph-rerank.json).

### 2.32 D08-D5 RMSNorm/residual owner audit (2026-08-14)

The exact Q4 owner is 48 c1/hidden-1,024/F32-weight nodes with no extra
resident layout. Eighteen linear-attention plus six full-attention layers call
standalone BF16-input RMSNorm at generic t256 (**24 nodes, 0.332 ms M11**).
The same 24 layers then call the already-fused BF16+BF16 add+RMSNorm at t256
(**24 nodes, 0.342 ms**), which normalizes the unrounded FP32 sum while emitting
a rounded BF16 residual and normalized BF16 output. Each family owns 98,304
bytes of F32 norm weights; all 196,608 bytes fit MALL and are already resident.

Vulkan's operation boundary is not one-to-one with HIP's fused second family.
Its 49 `RMS_NORM(1024,c1)` calls split into 24 attention, 24 post-attention, and
one final norm. The residual role also includes 18 standalone `ADD` calls while
six residual adds are fused into projection boundaries. Therefore the M11
**0.224-ms attention-norm** and **0.179-ms residual-glue** gaps are valid
operation-complete role bounds, not per-kernel speed ratios.

The actual-weight audit runs every supported t64/t128/t256/t512 schedule across
all 24 weights, plus the existing staged-F32 local256 diagnostic for add+norm.
All norm and residual outputs are bit-exact for all 24 rows, finite, and vector
top-1 exact. Five blocks of nine counter-rotated orders provide 45 device-clock
observations per route. Standalone t64/t128 regress; t512 wins 5/5 blocks but
projects only **0.00305 ms / 0.0366%** wall saving. Add+norm t512 projects
**0.00855 ms / 0.103%**, while staged local256 wins 5/5 and projects **0.01359
ms / 0.163%**. Their best possible combination saves only **0.01664 ms /
0.200%**, below D5's 1% stop gate. None enters a model gate.

The staged body remains excluded from gfx1151 and its prior gfx1100/W7900
Laguna hidden-3,072/48-call runtime rejection remains closed. The distinct
architecture/model/shape audit does not reopen that owner. The lineage command
was also blocked before reporting because manifest reference
`/home/lhl/amd-gpu-tuning/reference/atlas` is absent; no external kernel is
ported.

One new in-tree same-resident family is frozen for the bounded screen because
it has a concrete complete-package mechanism unavailable to the old schedules:
exact hidden-1,024 t256 caches four values per thread in registers, uses wave32
HIP shuffles plus a shared wave-leader final reduction, and covers standalone and
unrounded-add forms. Combined C removes **336 dynamic barriers/token** and
**147,456 bytes/token** of second-pass global reads with zero layout, persistent
bytes, hot scratch, or node change. It must reduce the current **0.67405-ms**
package below **0.59086 ms (12.34%)**, win every balanced block, remain spill/
scratch-free, and pass actual-weight plus independent CPU correctness. Freeze
only A) 24 standalone nodes, B) 24 add+norm nodes, and C) A+B. Generic t256 is
the unfused fallback; Q8, rows>1, other shapes/models/backends remain unchanged.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-norm-residual-audit.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-norm-residual-audit.json).

### 2.33 D08-D5 fixed-1024 wave-shuffle screen (2026-08-14)

RED first requires separate four-axis standalone/add+norm keys, exact c1 and
hidden-1,024 pre-load validation, all non-null pointers, fixed local256, current-
t256 GPU comparisons on random and finite BF16-edge inputs, and an independent
CPU oracle. GREEN adds one templated in-tree body. Each thread caches four
values; eight wave32 shuffle reductions publish wave sums, wave 0 computes the
final sum, and two block barriers publish `inv_rms`. The add form caches the
unrounded FP32 sums and emits the unchanged rounded BF16 residual. Generic t256
primitives remain registered fallbacks.

The complete actual-weight screen executes all 24 standalone and 24 add+norm
owners per route. All **24 standalone norms, 24 add norms, and 24 residuals are
bit-exact**, finite, and vector top-1 exact. The independent 48-row CPU gate has
max KL **2.84e-5** and top-1 **100%**. Five blocks of four counter-rotated orders
provide 20 marker-subtracted device-clock observations per route.

Standalone A improves the package **1.118x** and projects **0.07115 ms / 0.855%
wall**, but wins only 4/5 blocks and cannot proceed. Add+norm B improves
**1.133x**, projects **0.07923 ms / 0.952%**, and wins 5/5, but remains below the
1% package gate. Combined C alone qualifies at **1.229x**, normalized **0.67405
-> 0.54834 ms**, saving **0.12570 ms / 1.511%** with **17/20 samples and 5/5
balanced blocks**. It adds zero persistent bytes, hot scratch, and graph nodes.

A cached-build `rocprofv3 --kernel-trace` smoke records the expected `<false>`
and `<true>` names at grid/local **256/256**, **LDS512**, **scratch0**,
VGPR **16/24**, and SGPR128. The profiler again emits zero durations on this
stack; the device-clock screen supplies plausible timing. The audit's provisional
“DPP” label is concretized as portable HIP `__shfl_down` wave reductions; the
measured two-barrier/read-saving mechanism is unchanged. Only combined C enters
exact capability/runtime RED/GREEN and one Q4+Q8 full-model gate.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-norm-residual-screen.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-norm-residual-screen.json).

### 2.34 D08-D5 retained norm/residual route (2026-08-14)

Production RED binds the candidate through the four-axis registry, not a backend
branch. `GGUF_NORM_RESIDUAL_DECODE_POLICIES` admits only gfx1151, the immutable
24-layer/H1024 Qwen3.5 geometry, `MOSTLY_Q4_K_M`, rows1, and hidden1,024. Both
the actual attention-norm helper and post-attention add+norm site resolve their
layer key.
Q8, rows>1, other hidden/model/file/backend, output norm, verifier-F32 paths,
and generic t256 primitives remain explicit fallbacks. Public wrappers and HIP
entry points preserve pre-load validation. The resident capability caches a
prevalidated registry partial so eager execution does not repeat fixed-owner
Python checks.

The full gate uses two simultaneous physical sessions; each executes control
and candidate roles in every one of five counter-rotated blocks. On Q4,
production graph decode improves **118.18 -> 121.59 tok/s (+2.884%)** with
**5/5 block wins**. Nodes remain **287 -> 287** while graph inventory changes
exactly from generic RMSNorm **25 -> 1** and generic add+RMSNorm **24 -> 0** to
**24 fixed standalone + 24 fixed add** owners; the one generic standalone node
is final output norm. Persistent and hot-scratch deltas are zero.

The first eager pass exposed host validation overhead at -1.71%; a focused
host-only repair caches `_prevalidated=True` after capability/registry
resolution while leaving device code and public validation unchanged. The
repaired crossed-session eager gate is **69.59 -> 69.73 tok/s (+0.207%)**, 3/5
blocks, with exact trajectories. This focused repair does not invalidate the
completed correctness or graph replay evidence. Q4 prefill guards are
**1.0029x eager / 0.9976x graph**. Q8 selects no fixed leaves, preserves the
25+24 generic inventory and 289 nodes, and remains **1.0042x eager / 1.0049x
graph** with prefill **0.9992x / 1.0025x**.

The full category/heldout semantic gate passes **450/450 transitions per quant
(900/900 combined)**. Q4 max KL is **0.001745**, Q8 KL is zero, and all eager
and graph trajectories/final top-1 pairs are exact. The raw gate's initial
`passed=false` is a transparent evaluator defect: it searched demangled
`<false>/<true>` strings while graph introspection emits mangled
`ILb0E/ILb1E`; recomputation from the retained raw inventories gives the exact
24+24 Q4 route and unchanged Q8 route. Combined C is retained as the default
for its exact scope. Mandatory post-D5 rerank M12 precedes campaign closure.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-norm-residual-retained.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-norm-residual-retained.json).

### 2.35 Mandatory post-D5 graph rerank / M12 (2026-08-14)

M12 repeats M11's current-production protocol on retained commit `2cad5f2c2`:
128 eager samples, 128 one-token production graph replays, a separate 128-step
recording graph, and a separately instrumented graph with same-stream device
markers. Q4/Q8 trajectories are exact; recorded, production, and instrumented
final logits have KL zero and exact top-1. All **286/288** graph nodes are
assigned, marker-adjusted coverage is **97.23%/97.31%**, replay copies are zero,
and tracked graph bytes remain zero.

Current mean graph throughput is **119.88 tok/s Q4 / 117.18 Q8**. Relative to
M11's independent snapshot this is **-0.28%/-0.53%**, within the same-stack
snapshot variation and not a route A/B result. D5's intended owner movement is
clear: Q4 attention norm falls **0.33243 -> 0.27925 ms (-0.05318)** and residual
norm/glue falls **0.34162 -> 0.28256 ms (-0.05906)**. The joined package moves
**0.67405 -> 0.56181 ms (-0.11224 ms / -16.65%)**. Graph inventory is exactly
24 fixed standalone plus 24 fixed add norms and one generic final output norm.
Q8 retains 25+24 generic leaves and its joined norm movement is only **+0.00394
ms**.

The direct API census remains device-critical: launch plus Python residual is
well below 1%, with the synchronization span carrying graph wall. The largest
current Q4 Vulkan-joined gaps are linear-attention projections **0.985 ms**,
dense FFN projections **0.977 ms**, and full-attention core/KV **0.962 ms**;
each named package is already accepted/exhausted or rejected under its bounded
contract. No sub-1% experiment is reopened. However, current decode is only
**0.596x Q4 / 0.709x Q8** of the frozen Vulkan rows. D08-G1-G2 must therefore
run the final full correctness and fresh serial same-session parity gate; G3
cannot close if fresh Q4 still misses Vulkan.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-post-d5-graph-rerank.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-post-d5-graph-rerank.json).

### 2.36 D08-C1/C2 and G1-G3 closure gate (2026-08-14)

C1 commits one shared fixture: text `.Q` repeated 512 times tokenizes to exactly
**9707 x512** in hipEngine and the pinned llama.cpp Vulkan tokenizer for both
Q4_K_M and Q8_0. The teacher-forced continuation is **9707 x128**; its compact
RLE and little-endian int64 hashes are retained. C2 adds two reproducible core
harnesses. The hipEngine harness suppresses LM head/sampler only while capturing
its normal one-step graph; the pinned llama helper calls `llama_decode`
directly. Both also report a public greedy scope with LM head, top-1, and token
feedback included.

The final gate uses **five counter-rotated serial engine blocks per quant** on an
otherwise-idle GPU. Each child performs one internal warmup and one measured
exact-fixture row. All **645/645 forced-context and 645/645 public-equivalent
top-1 rows per quant** agree across engines; repeated traces are exact and all
logits are finite. D5's full category/heldout packet remains **900/900 top-1** at
max KL 0.001745, its CPU reference passes, and M12's recording/state checks are
exact with KL zero.

G2 nevertheless fails:

| Quant/scope | hipEngine pp512 | Vulkan pp512 | Ratio | hipEngine tg128 | Vulkan tg128 | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4 exact core | 2,527.69 | 5,957.10 | **0.424x** | 140.47 | 191.39 | **0.734x** |
| Q4 exact public | 2,530.58 | 5,466.92 | **0.463x** | 118.40 | 125.74 | **0.942x** |
| Q8 exact core | 4,152.27 | 6,257.62 | **0.664x** | 137.82 | 158.11 | **0.872x** |
| Q8 exact public | 4,074.10 | 5,591.25 | **0.729x** | **113.95** | 110.89 | **1.028x** |

Q4 loses all four measures in **0/5 blocks**. Q8 public decode wins 5/5, but its
prefill and core scopes fail. Fresh canonical shape-only rows independently give
Q4 **0.377x pp / 0.595x tg** and Q8 **0.690x / 0.697x**; they are supplemental
because llama-bench controls its own token inventory. An accidentally requested
nonexistent ICD path produced a CPU-labeled row; it is explicitly invalid and
was replaced by `radeon_icd.json` Vulkan rows.

Memory remains controlled: hipEngine Q4 owns **1,017.6 MiB** versus llama's
**1,015.4 MiB** declared device buffers (near equal). hipEngine Q8 owns **981.8
MiB** versus llama's **1,281.8 MiB** (23.4% lower). These are tracked-owned versus
declared-buffer scopes, not identical whole-card peaks.

D08-scoped G1 correctness/focused tests pass. The required milestone-wide test
attempt was stopped at 16% after concurrent closure timing invalidated it; before
stop it exposed existing scoreboard compactness/link failures and a local 27B
MTP-fixture assumption. Focused build, scheduler, D5, and closure-fixture bundles
pass, but no all-green milestone suite is claimed or automatically repeated.

G3 is **blocked**, not complete. Q4 does not match Vulkan in any required scope,
and all bounded D08 packages are already accepted/exhausted or rejected. Do not
open D08-T1. A production Vulkan backend, hand ISA, or reopened arithmetic family
would be an architectural campaign extension and requires human approval plus a
new non-gamed complete-package contract.

Artifact:
[`2026-08-14-gfx1151-qwen35-08b-vulkan-parity-closure.json`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-vulkan-parity-closure.json).

### 2.37 D08-X2 source-level cross-engine review (2026-08-15)

Human approval reopened the campaign for fine-grained profiling plus direct
source comparison against the pinned llama.cpp Vulkan checkout. The X1 gap map
(`2026-08-15-gfx1151-qwen35-08b-parity-gap-map.json`) ranks owners; this
section records what the reference implementation actually does per submodule
and the structural deltas it exposes. Files cited are from llama.cpp
`1d2869c6e54d5003f3927a79efbca0fefa034a6d` build 10415.

**Dispatch/fusion is not the gap.** A fresh logger capture counts **693 Vulkan
ops per pp512** and **663 ops per decode token** versus hipEngine's 491 prefill
dispatches and 286 production-graph nodes. llama.cpp dispatches 2.3x more ops
per token and still runs 1.68x faster. Its boundary fusions (MUL_MAT_ADD
residual-in-matvec, GLU, RMS_NORM_MUL, SSM_CONV_SILU) have hipEngine
equivalents (D3/D3B/D5, fused conv). Per-kernel memory structure, not launch
count, is the differentiator.

**Dense/linear prefill GEMM.** llama's `mul_mm.comp` + `mul_mm_funcs.glsl`
with the AMD/RADV coopmat tuning (`ggml-vulkan.cpp` l_warptile_mmq at the
`VK_VENDOR_ID_AMD && coopmat_support` branch; device subgroup is 64) runs
**256-thread workgroups, BM=128 x BN=128 block tiles, BK=32, WM=WN=64 warp
tiles, WMITER=2**, dequantizing Q4_K activations-side loads with LOAD_VEC 4
into **LDS-staged f16 tiles** (bank-conflict offset 8), then issuing coopmat
16x16x16 f32-accumulate fragments loaded from LDS, with workgroup k-split plus
a reduce shader for large K. hipEngine's `gguf_q4_k_pack8_prefill_wmma_kernel`
is a **32-thread single-wave, 16x32-tile, no-LDS, no-k-split** kernel that
re-reads activations from global once per 16-column band: for `[3584,1024]`
at rows=512 that is 224 band reads x 1.05 MB = **235 MB activation traffic per
mat** versus llama's 28 tiles x 1.05 MB = **29 MB**. The registered-but-disabled
pack8 WMMA leaf already measures 1.97x/2.33x the shipped tile8x8 leaf on the
0.8B dense shapes (within one BF16 ULP); X2a routes it. That remaining
1.5-1.7x estimate motivated X2-K1, whose actual-weight screen later found the
LDS/large-tile structure no faster than the qualified small-tile owner.

**Decode GEMV.** `mul_mat_vec_q4_k.comp` assigns a 16-thread team per Q4_K
superblock: packed 32-bit loads, unpack8 bit-trick scale decode, fully unrolled
vec4 FMA trees in registers, then one subgroup reduction per output row.
Measured Vulkan dense-FFN matvec traffic is ~198 GB/s effective
(132.1 MB gate/up in 0.668 ms) versus hipEngine pack8 t32/t128 leaves at
~123 GB/s on the same pool. The mechanism to port is wider per-thread ILP with
vectorized packed loads, not more launches (X2-K3).

**GDN.** `gated_delta_net.comp` is a **sequential register-state scan**: grid =
heads x sequences x value-columns, S_V=128 state rows sharded as
`ROWS_PER_LANE = S_V / LANES_PER_COLUMN` registers per lane, per token two
subgroup-clustered reductions, exp once per token, state written back only at
the end (plus optional snapshot slots for chunked queries). There is no chunk
decomposition, no inter-chunk kernel, and no intermediate global state traffic.
hipEngine's chunked scan pays chunk metadata, intra/inter boundaries, and
state finalize; its exact route exposes only 64 one-wave blocks. X2-K2 ports
the column-parallel register-state structure as an additional prefill schedule
for both quants.

**Full-attention decode core.** The opening marker attribution estimated the
retained split-K3 producer plus fused reducer at ~153 us/layer versus Vulkan's
~26 us/layer. X2-K4 later measured the exact production pair standalone at
56.5-57.1 us/layer, reducing the rewrite ROI to ~0.15-0.2 ms/token; no
single-kernel rewrite was started.

**Bounded extension outcomes (human-approved):** X2a retained the measured
pack8 WMMA bulk leaf; X2-K1's LDS-staged large-tile pack8 screen was bit-exact
but not faster; X2-K2 retained Q8 cluster8 GDN; X2-K3 found only a distributed
GEMV grind and landed no kernel; X2-K4 corrected the attention ROI and stopped;
X2-K5 retained dense-BF16 WMMA. Every retained route keeps its legacy fallback
and passed its named correctness/full-model gates.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-parity-gap-map.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-parity-gap-map.json).

### 2.38 D08-X2a retained pack8 WMMA bulk prefill route (2026-08-15)

The registered-but-disabled pack8 WMMA leaf was measured on actual 0.8B
dense-FFN weights at rows=512 (5 rotated marker repeats): **16x32 tile gate
`[3584,1024]` 0.585 ms = 1.97x** the shipped tile8x8 control (1.151 ms) and
**64x16 tile down `[1024,3584]` 0.480 ms = 2.33x** control (1.119 ms); t16
WMMA and the base pack8 prefill measure 0.552/0.688 and 3.19/2.44 ms. Every
WMMA output is within one BF16 ULP of the control.

The route repair is registry-clean: gfx1151 package capability
`GGUF_Q4_PACK8_WMMA_BULK_PREFILL`, a measured 0.8B tile policy in
`_default_q4_pack8_tiles`, and `_q4_pack8_wmma_dispatch` now overriding the
exact tile8x8 variant for bulk rows. `HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK=0`
rolls back (see `docs/REFACTOR.md`). Sole pack8 residency is unchanged; c1-c8
leaves are untouched; gfx1100 keeps tile8x8.

The binding gate is five counter-rotated env-pair exact-core blocks per quant.
Q4 prefill improves **2543.4 -> 3427.9 tok/s (+35.31%, 5/5)** and public
prefill **+34.97% (5/5)**; decode guards are **0.9956x core / 0.9942x public**
(inside 1%). Q8 is a no-route guard (**1.009/0.992 prefill, 1.007/1.000
decode**). Correctness over the 18 category/heldout prompts is **447/450
top-1 (99.33%) with max KL 0.003848**, matching the accepted P4 route class.
A cached-build rocprofv3 kernel trace confirms
`gguf_q4_k_pack8_prefill_wmma_kernel<16,32>` (288 dispatches) and `<64,16>`
(264) with the tile8x8 kernel absent; the stack's zero-duration blocker still
applies, so names and geometry are the smoke evidence.

Retained as the gfx1151 default. X2-K1 subsequently screened the proposed
LDS/large-tile structure and closed it unrouted on gfx1151 wave32.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-pack8-wmma-prefill-route.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-pack8-wmma-prefill-route.json).

### 2.39 D08-X2-K1 LDS large-tile screen: parity, not routed (2026-08-15)

X2-K1 ported the remaining structural delta from llama's mul_mm - LDS-staged
f16 A/B tiles with large block tiles and per-step shared scale planes - onto
pack8 in-kernel dequant (`gguf_q4_k_pack8_prefill_wmma64_kernel`, 128x64,
256 threads, plus parametric 64x64/64x128/128x128 screen variants). After a
staging-advance repair, every LDS config is **bit-exact** versus the routed
small-tile pack8 WMMA leaf (max abs diff 0.0 on actual weights).

Performance conclusion: on gfx1151 wave32 the LDS structure reaches **parity
at best**. In one marker screen the routed 16x32/64x16 small-tile leaf measures
**0.375 ms gate / 0.351 ms down** per mat (~10.0/10.7 TFLOPS effective) while
the best LDS configs tie (64x64: 0.437/0.423; 64x128: 0.410/0.465) and the
in-tree 128x64 is ~1.4x slower; a 64x64 warp tile (32 WMMA/step) also risks
the accumulator register budget. Vulkan's same-session reference is
0.328/0.317 ms, so the routed leaf is within **~14%/10% per mat** after X2a's
routing repair; the big dense-FFN gap was the tile8x8 route, already fixed.

The kernel stays registered as `pack8_wmma64_prefill_bf16_bf16_out` with its
correctness fixtures (`tests/test_gguf_q4_k_pack8_wmma64_prefill.py`), but no
production dispatch selects it (see `docs/REFACTOR.md`). The untested residual
lever is a wave64 WMMA variant (llama runs subgroup 64); X2 priority moves to
GDN (X2-K2), the largest remaining matched gap.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-pack8-wmma64-diagnostic.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-pack8-wmma64-diagnostic.json).

### 2.40 D08-X2-K2 retained Q8_0 GDN cluster8 route (2026-08-15)

X2-K2 first re-measured the GDN owner family at the exact production geometry
(16 K heads / 16 V heads / 128x128, tokens=512). The registered recurrent
variants already include llama's two schedules: `wave32_xor` (their 2048-wave
d32 form, 0.954 ms/layer) and `cluster8` (their eight-lane clustered form,
0.511 ms/layer); both agree to 1.1e-8 on L2-normalized inputs. The exact
`lds32_direct` route used by Q8_0 launches only **64 one-wave blocks**, and
the isolated full production composition (prepare + segments-cluster8 +
gate, two 256-token chunks) is **0.836 ms/layer** versus the instrumented
stage's 2.2 ms/layer - the stage markers are heavily perturbed, so binding
evidence comes from the exact-core A/B, not markers.

Routing Q8_0 to `chain_peer_cluster8` (the candidate P2 rejected when a
strict graph-decode guard missed by 0.0108% under its bounded contract):

- Exact-core pp512 **4230.6 -> 4949.4 tok/s (+16.70%, 5/5 blocks)**; public
  **+19.38% (5/5)**; core tg128 **1.0092x (3/5)** and public tg128
  **0.9909x**, both inside the 1% guard; all rows finite and deterministic.
- Correctness over the 18 category/heldout prompts: **448/450 top-1
  (99.56%), max KL 0.003260**.
- Q4_K_M keeps its existing cluster8 route; rollback is
  `HIPENGINE_GGUF_GDN_PREFILL_MODE=chain_lds32_direct_nonvolatile`.

Q8_0 exact-core prefill is now **0.79x** Vulkan. The remaining GDN gap for
both quants is the cluster8 kernel's 1.8x per-layer distance to Vulkan's
0.284 ms; the stage-marker inflation documented above also stands.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-q8-gdn-cluster8-route.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-q8-gdn-cluster8-route.json).

### 2.41 D08-X2-K5 retained dense-BF16 WMMA bulk prefill (2026-08-15)

The twelve expanded-dense-BF16 Q6_K down owners (`[1024,3584]`) ran the naive
32x8 scalar-tile dense GEMM at ~2.1 ms per mat (~25 ms of Q4_K_M pp512). A new
`dense_prefill_wmma_out_kernel` applies the same LDS-staged wave32 structure
proven in X2-K1 - 128x64 block tile, 256 threads, BK=32 shared f16 tiles,
F32-accumulating 16x16x16 WMMA - without dequant. On actual-weight-class data
it measures **0.6 ms vs 3.1 ms (5.2x)** per `[1024,3584] x 512` mat with
100% top-1 versus the naive reference.

Registered as `dense_gemv/prefill_wmma_out` behind the gfx1151 capability
`GGUF_DENSE_BF16_WMMA_BULK_PREFILL` and the
`HIPENGINE_GGUF_DENSE_WMMA_BULK=0` rollback. The backend default is limited
to the two complete-model-gated p512 dense-BF16 shapes; K not divisible by 32,
every row/shape policy miss, and peer backends keep the exact fallback. Fixtures
live in `tests/test_dense_prefill_wmma.py`.

Binding gates: five counter-rotated env-pair exact-core blocks on Q4_K_M give
prefill **3377.5 -> 4313.2 tok/s (+26.86%, 5/5)** and public **+24.63%
(5/5)** with decode guards **1.0018x / 1.0100x**; the three-block Q8_0 guard
is neutral (**1.0062x pp**, no dense-BF16 FFN owners). Correctness over the
18 category/heldout prompts is **446/450 top-1 (99.11%) with max KL
0.004215**, the accepted D3 route class.

Q4_K_M exact-core prefill reaches **4313 tok/s = 0.72x Vulkan** (from 0.424x
at campaign open).

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-dense-bf16-wmma-prefill-route.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-dense-bf16-wmma-prefill-route.json).

### 2.42A Human-approved narrow Qwen3.8-27B transfer audit (2026-08-15)

The audit compared every retained 0.8B package with Qwen3.8's actual
H5120/64-layer sole-T16 path on gfx1100. It did not copy 0.8B speed ratios or
shape-specific policies. Q5T16 SSM-out and sole-Q4T16 ownership are already
subsumed by the 27B route. Cluster8 GDN (16K/16V), short split-K3 attention
(8Q/2KV at contexts 514-641), fixed-hidden-1024 norms, pack8 WMMA, and
dense-BF16 WMMA fail hard geometry/layout checks. The 0.8B down-residual bodies
also cannot consume 27B's Q4T16/qmicro-planar c1 owners without new kernels.

One operation does transfer: D3's gate/up+SiLU fusion. The exact 0.8B pack8
t128 body is not reused, but the already registered gfx1100
`gguf_q4_k_t16_v1/dense_dual_local32_bf16_bf16_out` sibling matches the 27B
sole layout and `[1,5120,17408]` shape. Geometry-keyed admission removes 128
PM4 nodes (950 -> 822) without adding bytes. On RX 7900 XTX Qwen3.8-27B,
512/128 graph decode improves **33.542 -> 33.964 tok/s (+1.259%)**, prefill is
**+0.169%**, and tracked peak is byte-identical. The complete natural25 suite
improves true AR **19.123 -> 21.679 tok/s (+13.364%)** and native B3
**60.503 -> 60.925 (+0.697%)** with unchanged 63.095% draft acceptance, exact
greedy outputs, GPU/CPU acceptance agreement, and zero teardown ownership.

Artifact:
[`2026-08-15-qwen38-27b-dense-c1-fused-silu-retained.json`](../benchmarks/results/2026-08-15-qwen38-27b-dense-c1-fused-silu-retained.json).

### 2.42 Post-review current-HEAD baseline (2026-08-15)

After merging the correctness hardening and concurrent verifier-scratch work,
six clean-tree, cyclically counter-rotated shared-token blocks establish the new
current baseline. Exact-core pp512 is **4314.0/4975.9 tok/s Q4/Q8**; public
prefill is **4342.4/4959.5 tok/s**. Against X2-disabled controls, current Q4 is
**1.754x** the pre-X2 route and Q8 is **1.175x** its exact pre-X2 route, winning
all six prefill blocks. Every child is finite and deterministic, and current,
pre-X2, strict-X2, Q4, and Q8 all emit the same repeated-fixture top-1 trajectory.

The current Q4/Q8 core decode medians are **140.97/137.74 tok/s**, but the Q4
samples are visibly bimodal (roughly 140-141 and 147-148 tok/s), so this row
makes no new decode-speed claim. Public decode is **121.97/114.79 tok/s**. The
historical same-day Vulkan pp512 comparator remains **6015.0/6035.8 tok/s**;
those rows are not same-session denominators.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-post-review-exact-baseline.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-post-review-exact-baseline.json).

### 2.43 Post-review semantic and graph rerank (2026-08-15)

Three measured p512 device-marker runs assign **99.12%/99.05%** of Q4/Q8
instrumented prefill wall. Q4's current raw leader is dense FFN at **44.54 ms**
with a **20.05-ms** historical Vulkan-matched gap, followed by GDN at **29.27
ms / 16.68-ms gap** and linear-attention projections at **33.88 ms / 16.05-ms
gap**. Q8 is tighter: linear projections lead at **28.18 ms / 8.35-ms gap**,
GDN follows at **24.92 ms / 8.17-ms gap**, and dense FFN is **34.30 ms /
6.50-ms gap**. The Vulkan rows are the same-day logger diagnostic, not a
same-session denominator. Marker wall is 1.079x/1.053x exact-core wall, so these
shares rank roles but do not project topline gains directly.

The public production graph is **286/288 nodes** and **8.365/8.742 ms/token**
Q4/Q8. Eager, recording, production, and instrumented logits have zero KL and
100% top-1 agreement; the recorded trajectory is exact. The 240-marker graph
assigns every non-marker production node and measures **9.846/10.134 ms** of
device stages inside **9.959/10.261 ms** host wall: only **0.114/0.127 ms**
remains for API/Python, while asynchronous `graph_launch` itself is
**0.012/0.014 ms**. Therefore X2-K4's prior ~2.3-ms wall-minus-isolated estimate
is not graph API overhead; repeated isolated microbenches undercounted the
production chain. `rocprofv3` dispatch durations remain zero on this stack, and
the marker graph is itself 1.59/1.52 ms slower, so marker-normalized attention
must not reopen K4 over its exact standalone ~0.15-0.2-ms/token ROI.

The measured next-owner labels are Q4 dense FFN and Q8 linear-attention
projections, with shared GDN and linear-projection aggregate gaps effectively
tied. Q4 dense's known large-tile route is already exhausted; no candidate opens
until the prompt-length and cumulative semantic gates complete.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-post-review-semantic-rerank.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-post-review-semantic-rerank.json).

### 2.44 Post-review prompt-length threshold sweep (2026-08-15)

The approved p16/32/64/128/256/511/512/513/768/1024/4096 packet ran three
cyclic Q4 blocks and four counter-rotated Q8 blocks, one fresh resident process
per length and role. All **187** children used clean commit `6eb41219c`, cached
builds, device token embedding, production AOTriton at p512+, WMMA prefill, and
GEMV decode. Every result is finite, every final ID is 9707, and every scratch
capacity matches its prompt/decode geometry.

| Length | Q4 current / pre-X2 | Q4 current / strict-X2 | Q8 current / strict-X2 |
| ---: | ---: | ---: | ---: |
| 16 | 1.002x | 1.056x | 1.045x |
| 32 | 0.999x | 1.059x | 1.079x |
| 64 | 1.032x | 1.125x | 1.100x |
| 128 | 1.004x | 1.019x | 1.188x |
| 256 | 1.012x | 1.039x | 1.196x |
| 511 | 1.015x | 1.048x | 1.124x |
| 512 | **1.764x** | **1.820x** | **1.156x** |
| 513 | 1.011x | 1.046x | 1.140x |
| 768 | 0.997x | 1.031x | 1.131x |
| 1024 | 1.012x | 1.066x | 1.178x |
| 4096 | 1.007x | 1.087x | 1.138x |

This confirms that the pack8+dense WMMA package is isolated to exact p512:
Q4 current and pre-X2 are within 3.3% at every other measured length. Automatic
GDN is non-regressive versus strict X2 throughout the matrix, so its current
quant/geometry policies remain unchanged. Do not broaden WMMA beyond p512.

The original mixed-shape resident protocol exposed an intermittent AOTriton v3
C++ exception. Five coredumps assign that separate lifecycle defect to
`flash::attn_fwd`; the fresh-process protocol preserves all production routes
rather than hiding it with an attention fallback. The exception-boundary repair
is tracked separately and the final 18-prompt cumulative semantic packet remains
required.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-prompt-threshold-sweep.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-prompt-threshold-sweep.json).

### 2.45 Final cumulative semantic packet (2026-08-15)

The versioned clean-tree harness runs all **18 category+heldout prompts** in two
profiles per quant: unmodified natural chat-token sequences and category-derived
sequences repeated and truncated to exactly p512. Each candidate consumes the
strict-X2 free-running token prefix for **24 teacher-forced transitions** after
prefill, preventing greedy divergence from changing the comparison context.

| Quant / role | Natural top-1 / max KL | Category p512 top-1 / max KL |
| --- | ---: | ---: |
| Q4 strict-X2 | 450/450 / 0 | 450/450 / 0 |
| Q4 pre-X2 | 449/450 / 0.004529 | 449/450 / 0.004846 |
| Q4 current | 449/450 / 0.004529 | 449/450 / **0.005930** |
| Q8 strict-X2 | 450/450 / 0 | 450/450 / 0 |
| Q8 current | 448/450 / **0.003259** | 448/450 / 0.002205 |

Current defaults total **1794/1800 top-1 (99.667%)** with maximum KL
**0.005930**, comfortably inside the 90%/0.05 gate. Every strict/pre/current
repeat is trajectory+logit deterministic. All Conv/GDN and live full-attention
KV fingerprints are repeat-deterministic and finite. Candidate bytes differ
from strict on the accepted reassociated math routes (0 exact candidate profile
states), which is diagnostic; repeat determinism, finiteness, KL, and top-1 are
the binding state/math checks.

All **72** category-p512 strict/current recorded-graph prompt-role trajectories
exactly match same-role eager. Final graph top-1 is 100%, maximum graph KL is
**0.000475**, and the recording graphs contain **287 Q4 / 289 Q8 nodes**.
Current free-running trajectories match strict on **66/72** prompt profiles;
the six expected greedy divergences remain outside the same-context correctness
criterion.

This closes the post-review baseline/profile/threshold/semantic sequence and
retains every current route unchanged. It does not claim Vulkan parity: D08-G2
still fails the published same-session cross-engine target, so G3/T1 remain
blocked absent a separately approved architectural extension.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-cumulative-semantic.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-cumulative-semantic.json).

### 2.46 Synchronized exact hipEngine / llama HIP / Vulkan comparison (2026-08-15)

The final comparison uses six serial cyclic blocks per quant. Every engine
occupies every order position twice, every child performs one warmup plus one
measured core/public pair, and both llama backends are clean builds from
`1d2869c6e`. The committed p512/tg128 fixture supplies identical prompt and
forced-continuation IDs. Explicit `llama_synchronize()` now closes each core
timing boundary; the first incomplete llama HIP pilot exposed and discarded
impossible 70-71K pp512 rows before this packet began.

| Engine | Q4 core pp / tg | Q4 public pp / tg | Q8 core pp / tg | Q8 public pp / tg |
| --- | ---: | ---: | ---: | ---: |
| hipEngine | **4354.16 / 144.18** | **4376.18 / 123.30** | **5002.83 / 140.59** | **5038.64 / 117.04** |
| llama.cpp HIP | 4876.15 / 119.47 | 5016.51 / 89.22 | 4667.43 / 109.48 | 5073.18 / 83.96 |
| llama.cpp Vulkan | 5326.06 / 193.88 | 5595.94 / 126.31 | 5684.88 / 159.54 | 5790.96 / 111.81 |

All values are tok/s medians over six fresh processes. Standard production
cache defaults remain explicit: hipEngine stores BF16 KV, while these pinned
llama builds store F16 KV; both are 16-bit, and RADV reports no native BF16
capability. The exact comparison is therefore engine-default parity, not a
single-kernel same-storage microbenchmark.

| Quant/scope | hipEngine / llama HIP pp / tg | hipEngine / Vulkan pp / tg | Paired result |
| --- | ---: | ---: | --- |
| Q4 core | **0.893x / 1.207x** | **0.818x / 0.744x** | hipEngine loses both pp peers 0/6, beats HIP tg 6/6, loses Vulkan tg 0/6 |
| Q4 public | **0.872x / 1.382x** | **0.782x / 0.976x** | hipEngine beats HIP tg 6/6; Vulkan narrowly wins tg 6/6 |
| Q8 core | **1.072x / 1.284x** | **0.880x / 0.881x** | hipEngine beats HIP pp/tg 6/6, loses Vulkan pp/tg 0/6 |
| Q8 public | **0.993x / 1.394x** | **0.870x / 1.047x** | HIP pp is effectively parity; hipEngine beats both decode peers 6/6 |

The synchronized wall decomposition separates the residual to llama HIP from
the same-source HIP-to-Vulkan delta. Q4 core prefill is **117.589 / 105.001 /
96.131 ms** for hipEngine/HIP/Vulkan: the 21.458-ms total gap consists of
**12.588 ms hipEngine-to-HIP plus 8.870 ms HIP-to-Vulkan**. Q8 core prefill is
**102.342 / 109.696 / 90.063 ms**: hipEngine recovers 7.354 ms over llama HIP,
but Vulkan's 19.633-ms backend advantage leaves a 12.279-ms gap. Q4/Q8 core
decode similarly recover **1.435/2.021 ms/token** over llama HIP while remaining
**1.778/0.845 ms/token** behind Vulkan. On the public path, Q4 is only **0.193
ms/token** behind Vulkan and Q8 is **0.399 ms/token ahead**.

The Q4 llama HIP rocprof census records **712,685 dispatches / 44 names**, but
all 712,685 start/end timestamps are equal under the known gfx1151 profiler
blocker. Names and resources still expose the structure: **166,920 separate
Q8_1 activation quantizers pair one-for-one with 166,920 wave32 GEMVs** (46.84%
of all dispatches), Q4 prefill MMQ bodies use local256 with VGPR240-248, and
flash-attention uses a VGPR240/LDS19,456-B tile plus a separate reducer. No
per-op llama HIP duration is claimed.

Current hipEngine marker attribution therefore remains the operation-ranking
authority: Q4 prefill is led by dense FFN, GDN, and linear projections; Q8 by
linear projections, GDN, and dense FFN. Those historical Vulkan-joined gaps are
attribution-only and not additive. API/Python residual remains only
0.114/0.127 ms/token, so the next architecture campaign should prioritize a new
Q4 prefill dataflow rather than graph submission or the exhausted small
full-attention rewrite. Q8 prefill and core decode are now predominantly a
HIP-versus-Vulkan backend/codegen gap, while hipEngine's graph and device sampler
already outperform llama HIP decode and close public Vulkan decode.

All 36 children are finite and deterministic. Core and public 129-row top-1
trajectories are exact across all three engines and both quants. Variance is
below 5% for every metric. hipEngine owned memory remains **1017.63/981.78 MiB
Q4/Q8** versus llama-declared device buffers near **1015/1281 MiB**, preserving
Q8's ~23% memory advantage.

This result supersedes the old shape-only three-way diagnostic and the old G2
**core-prefill gap magnitude**, whose llama helper lacked an explicit asynchronous
backend drain. It does not change the historical G2/G3 disposition: current Q4
still loses every required core scope, so Vulkan parity remains open.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-current-exact-three-way.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-current-exact-three-way.json).

### 2.47 D08-X3 operation-complete Q4 pack8 prefill (2026-08-15)

The bounded continuation changes one exact dense-FFN owner rather than adding a
new resident format. One 128-thread block computes 32 output columns x 256
rows for both existing pack8 gate/up matrices, stages one gate and one up panel
in a 32-KiB LDS union, reuses each activation WMMA fragment, rounds both
projection boundaries through BF16 LDS, and emits BF16 SiLU(gate)*up directly.
Two registered singleton pack8 WMMAs plus standalone BF16 SiLU remain the exact
fallback. Model/quant/request and backend-shape policy fail closed outside
Qwen3.5-0.8B `MOSTLY_Q4_K_M`, rows512/K1024/N3584.

The actual-shape five-block leaf is byte-exact and measures **0.99043 ->
0.47409 ms (2.089x)**. Five cyclic fresh-process complete-model blocks then
measure:

| Scope | Rollback | Candidate | Ratio / paired wins |
| --- | ---: | ---: | ---: |
| Q4 exact-core pp512 | 4343.99 tok/s | **4943.76 tok/s** | **1.1381x / 5 of 5** |
| Q4 public pp512 | 4324.92 tok/s | **4923.72 tok/s** | **1.1385x / 5 of 5** |
| Q4 exact-core tg128 | 143.91 tok/s | 143.52 tok/s | 0.9973x |
| Q4 public tg128 | 122.37 tok/s | 121.76 tok/s | 0.9950x |

The candidate and rollback own identical **1,067,067,204-B** sessions. Three
Q8 no-route blocks stay inside the 2% guard: core pp/tg **0.9899x/0.9968x**
and public pp/tg **1.0059x/1.0019x**. All A/B trajectories are finite,
deterministic, and exact.

The full 18-prompt natural + 18-prompt category-derived-p512 packet passes for
both quants. Q4 candidate and rollback have identical teacher-forced logits and
persistent-state digests on **36/36** prompt profiles. Current Q4/Q8 remain
**1794/1800** top-1 versus strict, max KL **0.005930**; every repeat/state is
deterministic/finite and every current category-p512 recording trajectory
matches eager. A cached rocprof trace records the expected candidate at
local128, grid112x2, VGPR248, LDS32,768 B, and scratch0; gfx1151 timestamps are
again unavailable, so the independent HIP-event leaf supplies duration.

Reusing the synchronized same-day llama denominators only as a diagnostic puts
candidate Q4 core prefill at **1.014x llama HIP and 0.928x Vulkan**, or
**103.565 / 105.001 / 96.131 ms**. This is not a new interleaved external
packet, but it shows that the retained 14.299-ms hipEngine saving removes the
entire prior 12.588-ms hipEngine-to-HIP component and leaves about **7.434 ms**
to the Vulkan result. The next useful evidence is therefore a fresh clean-tree
three-way or a current marker rerank, not another variant in this closed
package.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-pack8-dual-wmma-silu-prefill.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-pack8-dual-wmma-silu-prefill.json).

### 2.48 Post-X3 synchronized exact three-way (2026-08-15)

The clean committed X3 default reruns the same six-block cyclic protocol with
unchanged same-source llama.cpp `1d2869c6e` helpers. Every engine occupies every
order position twice; each fresh child performs one warmup and one shared-token
exact-core/public p512/tg128 measurement.

| Engine | Q4 core pp / tg | Q4 public pp / tg | Q8 core pp / tg | Q8 public pp / tg |
| --- | ---: | ---: | ---: | ---: |
| hipEngine `5c33d9fdf` | **4896.02 / 145.80** | **4948.99 / 121.18** | **4997.25 / 141.19** | **4955.87 / 116.90** |
| llama.cpp HIP | 4848.25 / 122.58 | 5037.20 / 88.96 | 4640.50 / 110.18 | 5059.94 / 84.16 |
| llama.cpp Vulkan | 5510.21 / 194.15 | 5708.03 / 126.41 | 5703.85 / 159.05 | 5794.81 / 111.70 |

All values are tok/s medians over six fresh processes. Q4 exact-core prefill is
now **1.0099x llama HIP with 5/6 paired wins**, closing the measured
hipEngine-specific HIP component. Vulkan remains **1.1255x** faster; core wall
is **104.575 / 105.605 / 92.918 ms** for hipEngine/HIP/Vulkan, leaving **11.657
ms** to Vulkan. Across the independent pre/post snapshots, hipEngine Q4 rises
**4354.16 -> 4896.02 tok/s (+12.44%)** and the Vulkan wall gap falls **21.458 ->
11.657 ms (-45.68%)**. The separately paired X3 artifact remains the
optimization-delta authority.

Q4 public prefill is **0.9825x HIP / 0.8670x Vulkan**. X3 does not route decode:
hipEngine remains **1.189x/1.362x** llama HIP on core/public Q4 decode and
**0.751x/0.959x** Vulkan. Q8 remains the no-route guard, at **1.077x HIP /
0.876x Vulkan** core prefill and **1.047x Vulkan** public decode.

All 36 children are finite and deterministic. Core and public top-1 digests are
exact across engines and quants. Every metric CV is below 5%; the maximum is
4.764% for Q8 hipEngine public prefill. The old marker owner ranking is now
invalid because X3 materially moved dense FFN work; one current Q4 marker
rerank is required before selecting any next architecture package.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-post-x3-current-exact-three-way.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-post-x3-current-exact-three-way.json).

### 2.49 D08-X4 post-X3 Q4 prefill rerank (2026-08-15)

One clean current marker packet uses three persistent-session p512 runs after
one warmup. Marker throughput is **4663.61 / 4818.79 / 4814.85 tok/s** and wall
is **109.786 / 106.251 / 106.338 ms**. The median is only 1.017x the fresh
104.575-ms exact-core wall; semantic role medians sum to 106.426 ms for 100.08%
coverage.

| Role | Current marker | Exact-wall normalized | Historical Vulkan attribution | Normalized gap |
| --- | ---: | ---: | ---: | ---: |
| Dense FFN projections | **34.007 ms** | 33.443 | 24.495 | 8.948 |
| Linear-attention projections | **31.815 ms** | 31.288 | 17.828 | **13.460** |
| GDN recurrence | **23.617 ms** | 23.226 | 12.585 | **10.641** |
| Full-attention projections/RoPE | 7.809 ms | 7.679 | 4.385 | 3.294 |

X3 moves dense projections plus activation **45.981 -> 34.129 ms (-11.852
ms)** in the independent marker packets. The paired X3 A/B remains the
14.299-ms optimization authority because GDN and other unchanged stages also
moved with system variance. Historical Vulkan role rows remain attribution-only
and are not additive to the fresh 11.657-ms synchronized total gap.

Linear-attention projections are the new gap leader. Their current subowners are
Q5 QKV + Q4 gate explicit fallback **17.628 ms**, retained Q5 SSM-out **8.060
ms**, alpha/beta fallback **4.665 ms**, and BF16-to-F32 conversion **1.555 ms**.
SSM-out is already accepted/exhausted; the heterogeneous QKV/gate pair is the
largest unworked route.

Admit exactly one schedule: a 128-thread, 32-column x 256-row kernel over the
resident Q5T16 QKV and pack8-Q4 gate weights. It stages decoded K256 panels,
reuses activation fragments for both projections in the overlapping 2,048
columns, handles the remaining QKV columns under the same owner, preserves
separate BF16 boundaries and registered singleton fallbacks, and adds no
resident bytes. Continue only for byte-exact, no-scratch output and >=1.10x
combined-leaf speed; retain only for >=1% paired core/public pp512 with decode
and Q8 >=0.98x. No tile ladder is admitted.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-post-x3-prefill-rerank.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-post-x3-prefill-rerank.json).

### 2.50 D08-X4 heterogeneous QKV/gate screen (2026-08-15)

The one admitted implementation used a 128-thread, 32-column x 256-row K256
schedule over resident Q5T16 QKV and pack8-Q4 gate weights. It reused activation
fragments in the overlapping 2,048 columns, retained QKV-only handling through
N6144, preserved separate BF16 outputs and singleton fallbacks, and added no
resident bytes.

The actual-resident screen cycles all 18 linear-attention pairs ten times per
sample. Five rotated blocks measure:

| Block | Singleton control | Heterogeneous candidate |
| ---: | ---: | ---: |
| 0 | 21.61973 ms | 20.00583 ms |
| 1 | 19.57751 | 19.72724 |
| 2 | 19.48013 | 19.19493 |
| 3 | 19.84262 | 19.06383 |
| 4 | 19.59634 | 19.48194 |
| **Median** | **19.59634** | **19.48194** |

Layer-0 QKV and gate outputs are both byte-exact, but the candidate is only
**1.00587x / 0.11440 ms** faster. That misses the predeclared 1.10x leaf gate
and projects just **0.109%** of the fresh 104.575-ms exact wall. Per protocol,
no scratch profile, complete-model A/B, or cumulative gate was consumed. The
entire transient 699-line kernel/wrapper/route/test diff was removed; no default
or registered diagnostic remains.

This mechanism is closed. Revisit only for a materially different schedule with
>=1.10x combined-leaf evidence, not another tile ladder. The rerank is otherwise
unchanged and GDN recurrence becomes the next material package.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-q5t16-q4pack8-qkv-gate-rejected.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-q5t16-q4pack8-qkv-gate-rejected.json).

### 2.51 D08-X5 GDN cluster8 wave-broadcast screen (2026-08-15)

The next package preserved the retained 256-thread/eight-lane cluster8 geometry
and recurrent arithmetic. Its sole candidate replaced redundant global loads
with wave shuffles: Q/K shared across four columns, beta/decay across each wave,
and value across each eight-lane column. It added no resident or scratch bytes
and kept the original cluster8 owner as fallback.

Focused single-sequence and segmented tests are bit-exact for both recurrent
output and final FP32 state. The exact rows512/16V/128x128 screen nevertheless
measures retained/candidate medians of **0.77453/1.19467 ms** over five rotated
20-call blocks: a **0.64832x** result, or 54.24% candidate regression. The
predeclared 1.15x continuation gate therefore stops the package before rocprof,
complete-model A/B, or cumulative semantics. All 356 transient kernel, wrapper,
registration, and test lines were removed.

Do not revisit wave-shuffle load sharing in this cluster8 schedule. A future GDN
attempt requires a different algorithm or operation boundary. The unchanged
owner ranking now advances to dense FFN.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-gdn-cluster8-broadcast-rejected.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-gdn-cluster8-broadcast-rejected.json).

### 2.52 D08-X6 dense-BF16 WMMA down+residual (2026-08-15)

Twelve Q4_K_M FFN-down tensors are resident as dense BF16 at
rows512/K3584/N1024; the other twelve remain pack8-Q4. The retained WMMA owner
previously rounded each projection to BF16, wrote a temporary, then launched
`gguf_bf16_add`. X6 adds one registered sibling that performs the same first
round, adds the BF16 residual, and rounds again in the WMMA store. It adds no
resident/scratch bytes and keeps the two primitives as fallback.

Five rotated all-resident leaf blocks measure **9.95928 -> 9.80398 ms
(-1.56%, 1.01584x)** with byte-exact outputs, four wins, and a causal 0.15530-ms
saving. Cached tracing names the `<128,64,32,true>` body at WG256, 15,360-byte
LDS, 96 VGPR, SGPR128, and scratch0.

Five paired fresh-process Q4 blocks then measure exact-core pp512
**4938.948 -> 5091.718 tok/s (+3.09%, 5/5 wins)** and public pp512
**5020.345 -> 5104.790 (+1.68%, 3/5)**. Decode medians are positive but retain
the known core bimodality. The five-block Q8 control guard is
**0.9815x/0.9879x** core/public prefill and **0.9925x/1.0080x** decode; Q8 has
no dense-BF16 owner, and final non-dense misses return before registry work.
All children are finite/deterministic with exact core/public trajectories and
identical owned-session bytes. The full natural/category-p512 cumulative gate
also passes: current remains 1794/1800 top-1 with max KL 0.005930, all 72
recorded-graph prompt/role pairs are exact, and X6 candidate/rollback teacher
and state digests match on all 36 Q4 prompt/profile pairs.

Promotion is capability-scoped to gfx1151 and the already-qualified exact
shape. `HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL=0` temporarily restores WMMA plus
standalone add for bisection; registry/shape misses use the same fallback. The
causal performance claim is the exact 0.155-ms subwindow; the larger observed
full-wall ratio is retained as non-regression evidence rather than wholly
assigned to the removed add.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-dense-bf16-wmma-residual-prefill.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-dense-bf16-wmma-residual-prefill.json).

### 2.53 D08-X7 pack8-Q4 WMMA down+residual rejection (2026-08-15)

The other twelve FFN-down owners use the qualified small-tile resident-pack8
Q4 WMMA projection. X7 tested an architecture/shape-scoped sibling that kept
the projection's BF16 store boundary, loaded the BF16 residual, added in FP32,
and rounded the sum to BF16. A focused GPU oracle was byte-exact to the
ordinary WMMA projection plus explicit add, with no resident or scratch bytes.

The extra output-store work is not free for this producer. Two alternating
fresh-process Q4 pairs both lose in both prefill windows: exact-core pp512
**5124.452 -> 4908.151 tok/s (-4.22%, 0/2)** and public pp512
**5124.881 -> 4878.998 (-4.80%, 0/2)**. All completed children remain finite,
deterministic, top-1 exact, and byte-neutral. The screen stopped before Q8,
cumulative-semantic, and trace gates because the target performance failure is
material and repeated.

The kernel, wrapper, registry/capability route, focused tests, cumulative role,
and temporary selector were removed. Keep resident-pack8 WMMA plus
`gguf_bf16_add`; unlike the dense-BF16 X6 owner, this WMMA producer cannot
absorb the rounded residual cheaply. Reopen only for a materially different
producer schedule that first beats the complete-model control.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-q4-pack8-wmma-residual-rejected.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-q4-pack8-wmma-residual-rejected.json).

### 2.54 D08-X8 Q8T16 alpha/beta dual WMMA (2026-08-15)

All eighteen linear-attention blocks project one shared BF16 normalized input
through same-shape resident Q8T16 alpha and beta weights at
rows512/K1024/N16. The prior path launches two 16x32 singleton WMMAs. The
retained 64-thread owner assigns one matrix to each wave and stages each exact
BF16-to-F16 activation K16 tile once in 1 KiB LDS. Weight decode, WMMA K-order,
FP32 accumulation, and BF16 output stores are unchanged per matrix.

Five alternating all-actual-weight blocks are byte-exact and measure the 18
pairs **2.11263 -> 0.42168 ms (5.010x)**, saving 1.69095 ms with no resident or
hot-scratch bytes. Five fresh-process Q4 pairs improve exact-core/public pp512
**4912.861 -> 5091.743 (+3.64%, 5/5)** / **4899.499 -> 5147.088 (+5.05%,
5/5)**. Three Q8 pairs improve **5010.260 -> 5132.988 (+2.45%, 3/3)** /
**4955.990 -> 5061.490 (+2.13%, 3/3)**. Q8 decode medians are positive; Q4
public decode is 0.994x and the inactive core window retains its known
bimodality, so no decode-speed claim is made. Every completed child is finite,
deterministic, cross-role top-1 exact, and byte-neutral.

Production is scoped by the gfx1151 capability to exactly
rows512/K1024/N16+N16. `HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL=0` restores
two registered singleton WMMAs for one release window; every shape/backend/key
miss uses the same fallback. The full natural/category-p512 cumulative gate
passes, with all 72 current/rollback teacher-forced and state digests exact and
all recorded graph trajectories exact.

Artifact:
[`2026-08-15-gfx1151-qwen35-08b-q8t16-alpha-beta-dual-wmma-prefill.json`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-q8t16-alpha-beta-dual-wmma-prefill.json).

## 3. Comparison contracts

### 3.1 Two timing scopes, not one misleading ratio

llama-bench `tg128` measures model evaluation on a generated-token shape. The
hipEngine resident benchmark also performs its native sampler/token transport.
Keep two explicit scopes:

1. **Core model timing:** teacher-forced token input, no sampler ownership in
   either total. This is the strict module-to-module comparison.
2. **Public greedy generation:** embedding through sampled token and required
   device/host transport. This is the user-visible engine result.

Never subtract sampler or host costs from one engine but not the other. The
campaign closes only when both scopes are reported; the primary llama-bench
parity number is the core scope, while public generation is a non-regression
and usability gate.

### 3.2 Shared inputs

Opening throughput used shape-equivalent but not proven identical token
inventories: hipEngine repeated token 9707, while llama-bench controls its own
synthetic tokens. `D08-C0` creates a shared 512-token fixture and a deterministic
128-token teacher-forced continuation accepted by both engines. Record token
IDs and hashes in both artifacts.

For changes to hipEngine math, the repository CPU-reference gate remains
binding even if llama.cpp emits the same token:

- KL <= 0.05;
- top-1 agreement >= 90%;
- deterministic repeats;
- full state/trajectory checks required by the touched module;
- exact unfused fallback for a fused composite.

### 3.3 Same hardware and configuration

Every retained comparison records:

- Radeon 8060S / `gfx1151` identity and CU/cache snapshot;
- kernel, firmware, power profile, IOMMU state, and sampled clocks;
- TheRock ROCm/HIP and compiler revision;
- Mesa/RADV and Vulkan loader revision;
- exact engine commits and dirty-tree state;
- exact GGUF hash, tensor inventory hash, quant key, embedding placement, KV
  type, flash-attention mode, graph/submission class, and prompt/decode shape.

Profiler results are attribution evidence, not topline throughput. Both
`rocprofv3` and Vulkan timestamp logging may serialize or perturb execution.

## 4. Complete semantic module ledger

Kernel names and fusion boundaries differ across backends. Join profiles by
semantic role, then retain raw per-kernel/per-node rows underneath each role.

| Semantic role | hipEngine evidence | llama.cpp Vulkan perf-logger evidence |
| --- | --- | --- |
| Token embedding | GGUF Q6_K/Q8_0 embedding kernels, placement/copy metadata | `GET_ROWS`, transfer nodes |
| Attention RMSNorm | RMSNorm and fused norm/projection kernels | `RMS_NORM` / `RMS_NORM_MUL` |
| Linear-attention projections | Q4/Q5/Q8 prefill or c1 GEMV kernels for QKV/gate/output/decay/beta | `MUL_MAT*` grouped by shape and tensor role |
| Linear-attention conv | Conv, SiLU, state preparation kernels | `SSM_CONV_SILU`, `SILU`, copies |
| GDN recurrence | Exact/reassociated GDN prefill or decode kernels | `GATED_DELTA_NET`, `L2_NORM`, `SOFTPLUS`, `SIGMOID`, related nodes |
| Full-attention projections | QKV/gate/output projection kernels | `MUL_MAT*` joined by full-attention layer/shape |
| RoPE + KV write | RoPE, append/scatter, `KVLiveSpans` consumers | `ROPE`, `SET_ROWS`, `CPY` |
| Full-attention core | AOTriton/native prefill; grouped-GQA decode producer/reducer | `FLASH_ATTN_EXT` |
| Post-attention RMSNorm | RMSNorm or fused residual+norm boundary | `RMS_NORM_MUL` |
| Dense FFN gate/up | dual/single Q4/Q8 prefill or GEMV kernels | gate/up `MUL_MAT*` |
| Dense FFN activation | SiLU/multiply/fused activation kernels | `GLU`, `SILU`, `MUL` |
| Dense FFN down | Q5/Q6/Q8 prefill or GEMV kernels | down `MUL_MAT*` |
| Residual/common glue | add/combine/concat/copy/cast kernels | `ADD`, `MUL`, `CONCAT`, `CONT`, `CPY` |
| Final RMSNorm | final norm kernel | final `RMS_NORM_MUL` |
| LM head | Q6_K or Q8_0 vocab projection/top-1 kernels | `MUL_MAT_VEC` with `m=248320` |
| Sampler/token transport | top-1/sampler, required H2D/D2H, sync/API wall | excluded from core logger total; separately measured for public generation |
| Submission/unattributed | graph replay/eager API gaps, queue idle, profiler residual | Vulkan graph wall minus timestamped node total |

Completeness gates for each prefill and decode profile:

- 100% of timed kernels/nodes assigned to a role;
- semantic-role GPU totals sum within 1% of the backend-reported GPU total, or
  the exact timestamp-boundary difference is documented;
- all copies, synchronizations, and device-wide drains appear in a separate
  HIP/Vulkan API ledger;
- kernel/node dispatch count is recorded per token and per request;
- the top 95% of GPU time includes launch geometry and, where available,
  VGPR/SGPR/LDS/scratch data;
- no `other` bucket above 1% without an explicit owner and follow-up.

## 5. Profiling protocol

### 5.1 hipEngine

Use the existing selected-region support in
`scripts/qwen35_gguf_bench.py`. Build and warm every required library outside
rocprofv3, save `hipcc --version`, and require cached builds in the profiled
child.

Canonical fast-route command shape (C0 must confirm rather than assume it is
accepted for both files):

```bash
python3 scripts/qwen35_gguf_bench.py \
  --model /models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf \
  --quant gguf_q4_k_m --token-id 9707 \
  --prompt-length 512 --decode-tokens 128 \
  --warmup-decode-tokens 1 --warmup-runs 1 --measured-runs 5 \
  --persistent-session --force-bulk-prefill \
  --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode --graph-replay-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --json /tmp/d08-q4-fast.json
```

For profiling, use one short cached child per selected region:

```bash
rocprofv3 --kernel-trace --hip-trace --selected-regions \
  --output-format csv --output-directory /tmp/d08-hip-prefill -- \
  python3 scripts/qwen35_gguf_bench.py <same route> \
    --warmup-runs 0 --measured-runs 1 --decode-tokens 0 \
    --rocprof-selected-region prefill --require-cached-build

rocprofv3 --kernel-trace --hip-trace --selected-regions \
  --output-format csv --output-directory /tmp/d08-hip-decode -- \
  python3 scripts/qwen35_gguf_bench.py <same route> \
    --warmup-runs 0 --measured-runs 1 \
    --rocprof-selected-region measured_decode_graph --require-cached-build
```

If graph tracing is unstable, use `measured_decode` eager attribution and a
separate graph/direct API trace. Never profile a child that can spawn `hipcc`.
Extend `scripts/qwen35_gguf_rocprof_summary.py` only as needed to map the 0.8B
dense kernel families; do not discard raw names to make the buckets look clean.

### 5.2 llama.cpp Vulkan

The current Vulkan backend contains a timestamp-query logger:

```bash
cd ~/llama.cpp/llama.cpp-vulkan
GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 \
  build/bin/llama-bench -fa 1 \
  -m /models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf \
  -p 512 -n 128 -r 1
```

It emits per-graph `Vulkan Timings` with operation, quant, matrix shape,
dispatch count, mean duration, total duration, and GFLOP/s where applicable.
Capture Q4_K_M and Q8_0 separately and parse each pp/decode graph into the
semantic ledger. Use normal logger-off `-r 5` runs for topline; logger-on rows
are diagnostic only.

### 5.3 Cross-engine join

For each semantic role report:

- calls/request and calls/token;
- GPU ms/request for prefill or GPU ms/token for decode;
- share of each backend's profiled GPU total;
- matched projection shapes and encoded bytes where meaningful;
- hipEngine/Vulkan ratio only when math, shape, and timing scope are actually
  comparable;
- launch/API wall outside timestamped GPU work.

Do not infer a compiler problem from a semantic role that differs in fusion,
layout, activation reuse, or submission class.

## 6. Campaign lanes

### C lane — certification and controls

| ID | Work | Exit gate | Status |
| --- | --- | --- | --- |
| **D08-C0** | Rerun Q4_K_M and Q8_0 route matrix: fallback; forced bulk+WMMA+GEMV eager; forced fast route + production graph. Test host/device embedding only where supported. | **Complete:** exact quant/file hashes, effective routes, finite logits, 1+5 samples, serial fresh llama rows, and memory captured in the C0 artifact. | completed |
| **D08-C1** | Build shared 512-token and 128 teacher-forced token fixtures for both engines. Separate core model and public greedy timing. | **Complete:** `.Q` x512 is exactly token 9707 x512 in both engines/quants; continuation is 9707 x128 with committed RLE/int64 hashes. | completed |
| **D08-C2** | Freeze hardware/software snapshot and interleaved comparison script. | **Complete:** pinned HIP/llama harnesses, five counter-rotated blocks, hardware/software/clocks, exact commands, and raw hashes are retained. | completed |

### M lane — full module attribution

| ID | Work | Exit gate | Status |
| --- | --- | --- | --- |
| **D08-M1** | hipEngine Q4 prefill selected-region kernel/API profile. | **Complete:** names/resources captured and steady-clock semantic stages reconcile 99.51% of instrumented wall; Q4 fallback projection route identified. | completed |
| **D08-M2** | hipEngine Q4/Q8 eager and production graph decode profiles. | **Complete:** 334/288 graph kernels assigned, 97.64%/97.52% instrumented stage coverage, 0.20% host submission share, zero replay copies, exact trajectories; D3 admitted. | completed |
| **D08-M3** | llama.cpp Vulkan Q4 pp512/tg128 perf-logger profiles. | **Complete:** all measured prefill and 128 decode graphs assigned; operation totals reconcile within logger rounding and submission residual is explicit. | completed |
| **D08-M4** | Repeat M1-M3 for Q8_0. | **Complete:** explicit host-embedding HIP route plus complete Q8 Vulkan operation map; no mislabeled quant row. | completed |
| **D08-M5** | Produce joined semantic-role Amdahl table. | **Complete:** every module was joined or represented by named submission residual; `other=0`; the resulting P1 admission is now accepted and this pre-P1 ranking is superseded. | completed |
| **D08-M6** | Mandatory post-P1 Q4 semantic replacement capture and rerank. | **Complete:** 99.60% prefill and 96.24% eager-decode reconciliation; P3 is first at 29.42% projected request saving, P2 second at 19.39%. | completed |
| **D08-M7** | Mandatory post-P6 Q4 semantic replacement capture and rerank. | **Complete:** 99.58% prefill and 96.59% eager-decode reconciliation; residual linear-attention projections are first non-exhausted at a corrected 10.06% request bound. | completed |
| **D08-M8** | Mandatory post-P4 Q4 semantic replacement capture and rerank. | **Complete:** 99.46% prefill reconciliation; Q direct markers fall 64.05%, every >=1% prefill package is exhausted, and P5 remains parked at 0.82%. | completed |
| **D08-M9** | Mandatory post-D3 production graph/direct rerank. | **Complete:** current Q4 graph is 111.93 tok/s / 310 nodes at 97.60% stage coverage; full-attention core/KV leads at 1.431 ms / 16.02%, ahead of dense 12.32% and linear projections 10.60%. | completed |
| **D08-M10** | Mandatory post-D4 production graph/direct rerank. | **Complete:** current Q4/Q8 graphs are 120.62/119.14 tok/s at 310/288 nodes; D4 core/KV falls 0.512/0.492 ms, and Q4 dense projections lead at 1.063 ms / 12.82%. | completed |
| **D08-M11** | Mandatory post-D3B production graph/direct rerank. | **Complete:** current graph is 120.21/117.80 tok/s at 286/288 nodes; D3B Q4 dense movement is -0.102 ms and D5 is admitted. | completed |
| **D08-M12** | Mandatory post-D5 production graph/direct rerank. | **Complete:** current graph is 119.88/117.18 tok/s at 286/288 nodes; Q4 joined norm moves -0.11224 ms with exact trajectories and >97% coverage. | completed |

No implementation lane starts before `D08-C0` and the relevant M lane identify
a shipped owner. A trivial route correction from C0 may be retained immediately
if it passes the same correctness and benchmark gates; it is not “kernel work.”

### P lane — prefill, ordered by likely leverage but profile-gated

| ID | Candidate class | Potential if admitted | Admission signal | Hard bound / stop rule | Status |
| --- | --- | --- | --- | --- | --- |
| **D08-P1** | Fast-route/default/path selection: bulk rows, WMMA/MMQ projection coverage, correct AOTriton/native full-attention route. | **realized: +33.68% Q4 pp512 / +42.19% eager tg128** | Existing Q5T16 direct/rowtile/WMMA leaves beat dense BF16 on the actual QKV shape and pass full-model correctness. | **Closed:** one exact-role materialization/dispatch repair, no kernel variants, and M6 re-profile complete. | accepted |
| **D08-P3** | Dense Q4/Q5/Q6/Q8 gate/up/down projection kernels: tile, layout, activation reuse, and fusion. | **29.60% merged Q4 bound; unrealized** | All three frozen sole-resident candidates won pp512 but failed a required c1 or c8 operational guard. | **Closed:** no duplicate layouts, no full-model A/B, and no production change. Reopen only for a new sole-resident family that is non-regressive at every width. | rejected |
| **D08-P2** | GDN recurrence and convolution. Reuse retained GPF/LCP schedules before inventing a new one. | **realized: +4.33% paired / +5.83% independent Q4 pp512** | The 16K/16V shape exposes too few exact-LDS32 blocks; cluster8 wins the bounded screen and complete gate. | **Closed at P2:** Q4-only policy; D08-X2-K2 later retained Q8 cluster8 under a fresh gate without adding arithmetic variants. | accepted; Q8 disposition superseded by X2-K2 |
| **D08-P6** | Remaining linear-attention projections after accepted Q5 QKV routing: residual QKV/gate, alpha/beta, and SSM-out. | **realized: +14.18% graph-scope pp512 / +0.69% graph tg128; -46.69 MiB weights** | The split selected 35.93-ms Q5 SSM-out; exact-role sole Q5T16 passes 449/450 top-1, max KL 0.003273, and all graph pairs. | **Closed:** exactly three existing leaves and one combined full-model A/B; M7 confirms SSM-out at 9.68 ms. | accepted |
| **D08-P7** | Residual linear-attention QKV/gate, alpha/beta, and conversion after accepted Q5T16 SSM-out. | **3.58% selected gate bound; unrealized** | Q4T16 pp512/split-c8 win but c1 is 0.883x; raw Q4 regresses all operational widths. | **Closed:** preserve sole pack8, skip conditional source-F16 after exact-T16 c1 failure, and run no full-model A/B. | rejected |
| **D08-P4** | Full attention and RoPE/KV boundaries. | **realized: +4.79% graph pp / +1.41% graph tg; -4.13 MiB** | Exact-role sole Q4T16 passes every operational leaf, 447/450 top-1, max KL 0.003574, and exact trajectories. | **Closed:** gfx1151 0.8B Q-only plugin policy; M8 confirms Q at 2.71 ms and all other scope remains unchanged. | accepted/exhausted |
| **D08-P5** | Residual/norm/activation/copy launch coalescing. | **low: 0.82% combined M8 bound** | The combined measured package remains below the 1% continuation threshold. | Park until a fresh profile makes the package material; retain any independently measured exact non-regressive win. | parked |

### D lane — decode, ordered by the measured per-token ledger

| ID | Candidate class | Potential if admitted | Admission signal | Hard bound / stop rule | Status |
| --- | --- | --- | --- | --- | --- |
| **D08-D1** | Production graph replay, persistent buffers, and redundant sync/copy removal. | **measured host share 0.20% Q4/Q8** | M2 finds one launch+sync per token, zero replay copies, and a device-critical sync span. | **Closed:** no removable >=1% submission package; graph capture remains lifecycle-only and excluded from throughput. | rejected/no candidate |
| **D08-D2** | LM-head/top-1. Q4_K_M uses the tied Q6_K table; Q8_0 uses Q8_0. | **no positive matched graph gap** | Share-normalized graph LM-head/sampler is already no slower than Vulkan for both quants. | Preserve full vocabulary and top-1; reopen only after a fresh graph census changes ownership. | parked |
| **D08-D3** | Dense projection GEMVs, including Q4/Q5/Q6/Q8 replacement/raw layout and wave geometry. | **realized: +8.29% graph tg128 / +2.28% eager; zero bytes** | Fused-SiLU t128 removes 24 graph nodes, wins all decode pairs, passes 446/450 top-1/max KL 0.002843, and keeps exact trajectories. | **Accepted for Q4 gate/up c1.** M10 now admits the distinct dense-down owner audit as D3B. | accepted |
| **D08-D3B** | Dense FFN down projection plus residual, split across 12 Q4-pack8 and 12 Q6-dense owners. | **realized: +4.13% eager / +1.31% graph tg128; zero bytes** | Exact same-resident C wins 5/5 crossed-session blocks, passes 900/900 transitions at KL 0, and removes 24 nodes. M11 confirms dense FFN 2.159 -> 2.057 ms and down+residual at 1.011 ms. | **Accepted/exhausted for exact gfx1151 0.8B Q4_K_M c1 down owners.** | accepted/exhausted |
| **D08-D4** | GDN decode/conv and short-context full attention. | **realized: Q4 +5.97% / Q8 +5.95% graph tg128; zero bytes/nodes** | Exact-shape existing generic split-K3+fused-gate wins 5/5 graph pairs per quant, passes 897/900 combined top-1 with max KL 0.001944, and preserves exact trajectories and pp512. | **Accepted for gfx1151 0.8B rows1/8Q/2KV/D256 at cap514-641.** M10 confirms -0.512/-0.492 ms; fixed256 and unsupported routes remain fallbacks. | accepted |
| **D08-D5** | RMSNorm, SiLU/GLU, residual, embedding, sampler, and token transport. | **realized: Q4 graph +2.884% / eager +0.207%; zero bytes/nodes** | Exact fixed-1024 C wins 5/5 graph blocks, replaces 24+24 owners, passes 900/900 transitions at max KL 0.001745, and preserves Q8/prefill guards. M12 confirms joined norm **0.67405 -> 0.56181 ms**. | **Accepted/exhausted for exact gfx1151 0.8B Q4_K_M c1 attention/post-attention norm owners.** | accepted/exhausted |

### G lane — promotion and closure

| ID | Work | Exit gate | Status |
| --- | --- | --- | --- |
| **D08-G1** | Full correctness and regression packet. | **D08-scoped pass:** CPU/D5 category/state/determinism/Q8/focused gates pass. **Milestone-wide block:** broad suite attempt exposed unrelated existing failures and was not completed. | scoped pass / broad blocked |
| **D08-G2** | Same-session interleaved Q4/Q8 final comparison. | **Failed:** exact Q4 core is 0.424x pp / 0.734x tg and public is 0.463x / 0.942x; every required Q4 measure loses 0/5 blocks. | blocked-parity |
| **D08-G3** | Publish retained artifact/scoreboard/changelog and close campaign. | Blocked by G2 and the missing all-green milestone suite; exact commands/module ledger are published without claiming closure. | blocked |
| **D08-T1** | Open broad 27B transfer campaign and re-profile from zero. | D08-G3 complete; no 0.8B ratio is copied as 27B evidence. A human-approved narrow audit retained one independently measured existing T16 fusion in section 2.42. | broad campaign blocked by D08-G3; narrow audit complete |

## 7. First-pass decision tree

After the current M7 ledger, choose exactly one implementation owner:

1. **Fast flags disabled or fallback kernels present?** Fix route selection and
   defaults first. Re-profile; the Amdahl table is invalid after a structural
   route change.
2. **Prefill dominated by GDN recurrence?** Verify which retained GPF schedule
   runs on the 0.8B shape and why; port or retune only if the current route is
   absent or resource-mismatched.
3. **Prefill dominated by quant projections?** Compare same semantic shapes to
   Vulkan MMQ/coopmat. Check row count, WMMA admission, repack/layout, weight
   rereads, and grid coverage before source-level instruction tuning.
4. **Decode dominated by LM head?** Treat it as its own vocab-scale bandwidth
   and reduction problem; do not hide it inside a generic “GEMV” bucket.
5. **Decode dominated by many short kernels/API gaps?** Reduce launches,
   synchronization, and graph overhead before rewriting arithmetic.
6. **Decode dominated by weight streaming?** Compare effective bytes and
   sustained bandwidth with a >64 MiB cycling pool; inspect occupancy and
   coalescing. Do not repeat the rejected blanket non-temporal-load experiment.

## 8. Anti-rabbit-hole rules

- Do not optimize the opening fallback route unless C0 proves it is the intended
  production route.
- Do not use `llama-bench -v` metadata output as module timing; use
  `GGML_VK_PERF_LOGGER` or an external GPU trace.
- Do not compare profiler-perturbed totals as topline throughput.
- Do not call a Vulkan/HIP module ratio a compiler result when layouts, fusion,
  math, or submission differ.
- Do not repeat broad wave64, non-temporal-load, generic reduction, or tile
  sweeps already closed in `HIP-vs-VULKAN.md` and
  `GGUF-PREFILL-OPTIMIZATION.md` without new production evidence.
- Do not tune to token 9707, a fixed prompt, or candidate IDs. All retained
  math changes pass category/heldout correctness and deterministic-state gates.
- Do not sacrifice prefill to win decode or vice versa without an explicitly
  accepted tradeoff. The declared objective is to match or beat both pp512 and
  tg128.
- Do not begin the 27B campaign before D08-G3.

## 9. Parked, rejected, and future-impact ledger

A rejected idea is not silently retried. A parked idea retains its maximum
plausible impact and the evidence required to reopen it. Sort new entries by
potential band, then measured upper bound.

| Candidate / family | Current disposition | Potential | Why not active now | Exact revisit trigger |
| --- | --- | --- | --- | --- |
| Micro-tune the opening fallback kernels | parked | critical only if fallback is production | Opening rows disabled all named fast paths; tuning them first could optimize a route we should not ship. | C0 proves the fallback remains the intended route for a material semantic owner. |
| P1 Q5T16 QKV route | **accepted** | **realized: +33.68% pp / +42.19% eager tg; -11.59% tracked peak** | One exact-role sole-resident policy reused the shipped direct/rowtile/WMMA family; no arithmetic variants or duplicate weights. | Closed; reopen only if a future correctness regression identifies this exact role. |
| Dense FFN projection package / P3 | **rejected** | **29.60% merged Q4 bound; unrealized** | Raw Q4 regresses every operational width, Q4T16 regresses c8, and Q6T16 regresses c1; duplicate residents are disallowed. | A new sole-resident family passes pp512 plus c1/c2/c4/c8 without sacrificing memory or either topline scope. |
| Q4/Q8 16K/16V cluster8 GDN / P2 + X2-K2 | **accepted; X5 follow-up rejected** | **Q4 +4.33% paired / +5.83% independent; Q8 +16.70% exact-core pp512** | Complete Q4 gates pass; X2-K2's fresh Q8 gate supersedes P2's strict-guard miss. X5 wave-broadcast sharing is exact but only 0.6483x. | Keep cluster8; reopen only for a different algorithm/operation boundary or a regression in either exact quant/shape key. |
| Q5T16 SSM-out / P6 | **accepted** | **realized: +14.18% graph pp / +0.69% graph tg; -46.69 MiB** | Exact-role sole Q5T16 replaces 18 dense-BF16 expansions; correctness and all production-graph pairs pass. | Closed; M7 confirms the route at 9.68 ms and selects the residual group instead. |
| Residual linear-attention projections / P7 | **rejected** | **3.58% selected gate bound; unrealized** | Native Q4T16 c1 regresses 11.73%; raw Q4 regresses every c1-c8 width; source-F16 cannot repair c1. | A new sole-resident family passes pp512 and every operational width, or an operation-complete fusion removes the c1 regression without sidecars. |
| Full-attention Q projection / P4 | **accepted/exhausted** | **realized: +4.79% graph pp / +1.41% graph tg; -4.13 MiB** | Six exact-role sole-Q4T16 residents pass 447/450 top-1 and every graph/eager trajectory; M8 confirms 2.71-ms Q and T16 WMMA bulk ownership. | Closed; reopen only for a regression in this exact role/shape key. |
| Graph/submission work / D1 | **closed; no candidate** | **0.20% launch+Python share; zero replay copies** | M2 assigns all 334/288 Q4/Q8 graph kernels and proves stream sync contains the device-critical span. | Reopen only if a future graph transport introduces a measured >=1% host/API/copy residual. |
| Dense decode projections / D3 | **accepted/exhausted gate/up + down/residual** | **D3 +8.29% graph; D3B +1.31% graph / +4.13% eager** | Default fused-SiLU and rounded-residual owners remain exact. M11 assigns 286 Q4 nodes and confirms dense FFN -0.102 ms with zero tracked graph bytes. | Closed; M11 admits only D5's separate boundary audit. |
| Full-attention core/KV / D4 | **accepted; M10 complete** | **realized: Q4 +5.97% / Q8 +5.95% graph tg128; zero bytes/nodes** | Exact-shape split-K3+fused-gate passes 897/900 combined top-1, max KL 0.001944, exact trajectories, neutral prefill, and 5/5 graph wins per quant. M10 confirms -0.512/-0.492 ms core/KV. | Closed for cap514-641; fixed256, rows>1, 16Q, unsupported shapes/backends, and rollback control remain fallbacks. |
| LM-head specialization | parked D2 | **low: 1.86% Q4 / 1.31% Q8 eager upper bound** | Joined M5 shows the vocab node is not the leading owner. | M2 graph census materially changes ownership and projects >=1% request saving. |
| Blanket non-temporal weight loads | rejected prior family | low | Prior gfx1151 cold-leaf improvement regressed/flattened complete decode by defeating useful MALL reuse. | New profile proves the exact production owner is cold-streaming, cache-polluting, and has a >=1% whole-request bound. |
| Generic wave64/reduction sweep | rejected/parked prior family | low | Cross-backend and GGUF campaigns already found no broad recovery; wave32 is the production contract. | A minimized hot kernel shows a specific wave32 occupancy/reduction bottleneck and a wave64-correct oracle. |
| Hand ISA or production Vulkan backend | parked | unknown/high cost | Current production-shaped combined HIP kernels often match or beat Vulkan micros; engine gap is not yet attributed. | A matched production semantic slice wins after route/layout/submission controls and projects >=10% request saving. |
| 27B dense transfer | broad campaign blocked by D08-G3; narrow audit complete | **critical future** | The approved audit retained only an independently gated existing T16 fusion; all shape-specific 0.8B routes were rejected or already subsumed. | D08-G3 closes for a broad campaign, or a new human-approved bounded audit names a matching 27B owner and full gate. |

## 10. Update protocol

Update this file when a lane moves from `pending` to `in-progress` and when it
closes as accepted, rejected, blocked, or superseded. Each retained performance
unit also updates:

- a unique immutable entry under `worklog/entries/`;
- a compact JSON artifact under `benchmarks/results/`;
- `benchmarks/README.md` and its `Last updated` date;
- `benchmarks/CHANGELOG.md` with old -> new metric, percentage delta, reason,
  and artifact/source;
- `docs/REFACTOR.md` for any retained temporary flag or duplicate route.

Every logical unit is validated and committed before the next lane begins.
