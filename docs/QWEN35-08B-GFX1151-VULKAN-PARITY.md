# Qwen3.5 0.8B gfx1151 Vulkan-Parity Campaign

Status: D08-C0, D08-M1/M3-M8, and accepted D08-P1/P2/P4/P6 completed 2026-08-14; D08-P3/P7 are closed/rejected, the named prefill ladder is exhausted, and D08-M2 graph/direct census is next.

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
| 5 | **D08-P2 GDN recurrence** | **accepted: +4.33% paired Q4 pp512** | The Q4/16K/16V shape-scoped cluster8 route cuts marker GDN **67.60 -> 42.83 ms (-36.64%)** and passes the complete semantic/graph-decode gate. | Promoted for Q4 only; Q8 remains exact after its strict graph-decode guard missed by 0.0108%. |
| 6 | **Mandatory post-P2 re-profile** | **completed** | The structural GDN route invalidates the P2-era ranking. | Reconciles 99.61% of wall; dense FFN is exhausted, so remaining linear projections are the largest non-exhausted owner. |
| 7 | **D08-P6 remaining linear-attention projections** | **accepted: +14.18% graph-scope Q4 pp512 / +0.69% graph tg128; -46.69 MiB** | The audit selected 35.93-ms Q5 SSM-out; sole Q5T16 direct/rowtile/WMMA wins the complete production route and correctness gate. | Closed after exactly three shipped leaves and one full-model A/B; Q8 and 27B remain unchanged. |
| 8 | **Mandatory post-P6 re-profile** | **completed** | P6 removes 48.96 MB of weights and changes 18 bulk projection owners, invalidating the post-P2 ranking. | M7 reconciles 99.58% of wall, confirms SSM-out at 9.68 ms, and corrects one 1.20-ms Vulkan Q4 role assignment without changing backend totals. |
| 9 | **D08-P7 residual linear-attention projections** | **closed/rejected; Q4 gate bound unrealized** | Native Q4T16 wins pp512 2.006x and exact split-c4x2 wins c8 1.390x, but c1 is 0.883x; raw Q4 regresses every c1-c8 width. | Preserve sole pack8; source-F16 is ineligible after the exact-T16 c1 failure, so no full-model A/B or production change. |
| 10 | **D08-P4 full attention and RoPE/KV** | **accepted: +4.79% graph pp512 / +1.41% graph tg128; -4.13 MiB** | Sole Q4T16 for six source-Q4_K `[N4096,K1024]` Q projections passes all leaf widths, 447/450 top-1, and exact graph/eager trajectories. | Closed with direct c1, rowtile c2-c4, split-c4x2 c8, and WMMA bulk; all other Q4 roles, 27B, and peer backends retain prior owners. |
| 11 | **Mandatory post-P4 re-profile** | **completed** | Six physical weights and their complete Q projection stages changed owner; pre-P4 ranking was no longer authoritative. | M8 reconciles 99.46% of wall, confirms Q at 2.71 ms and T16 WMMA bulk dispatch, and closes every >=1% prefill package as accepted/exhausted or rejected. |
| 12 | **D08-M2 graph/direct census** | **next; required for decode ownership** | Eager markers cannot assign production graph API, synchronization, copy, or launch residual. | Census graph capture/replay/direct boundaries once; do not select decode arithmetic from eager rows. |
| 13 | **Medium/low prefill tail** | **parked: P5 current bound 0.82%** | Every named >=1% prefill package is exhausted under its frozen budget. | Reopen only after a fresh profile raises a complete package above 1% or an exact measured small win is already ready to retain. |
| 14 | **D08-G1-G3 closure** | campaign gate | Correctness, same-session parity, artifacts, and scoreboards turn diagnostics into a retained result. | Close 0.8B before D08-T1 opens 27B. |

### 1.2 Bounded task contract

Every task records before work starts: semantic owner, baseline time/share,
maximum plausible whole-request saving, exact experiment budget, correctness
gate, accept threshold, reject condition, and revisit trigger. A task cannot
remain indefinitely `in-progress`.

| Task class | Hard experiment bound | Accept rule | Reject / park rule |
| --- | --- | --- | --- |
| Route certification (`C0`) | At most 3 hipEngine routes x 2 quants, 2 supported embedding-placement controls, and 2 fresh llama rows. Each topline row is 1 warmup + 5 measures. No source edit. | Effective route matches the request, correctness passes, and the fastest intended route becomes the certified baseline. | One failed route receives one focused diagnosis. If unresolved, open a named blocker; do not start kernel tuning on an unknown route. |
| Profile (`M1-M8`) | One clean capture per backend/quant/phase; one replacement capture only for incomplete/corrupt output. | 100% node assignment and <=1% timing residual, with API/launch gap separate. | If the tool cannot expose a complete ledger after one repair, record the missing surface and add the smallest instrumentation needed; do not infer owners from names alone. |
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
| **D08-C1** | Build shared 512-token and 128 teacher-forced token fixtures for both engines. Separate core model and public greedy timing. | Exact token inventory hashes match; sampler ownership is explicit. | pending |
| **D08-C2** | Freeze hardware/software snapshot and interleaved comparison script. | Reproducible command bundle with clocks and clean provenance. | pending |

### M lane — full module attribution

| ID | Work | Exit gate | Status |
| --- | --- | --- | --- |
| **D08-M1** | hipEngine Q4 prefill selected-region kernel/API profile. | **Complete:** names/resources captured and steady-clock semantic stages reconcile 99.51% of instrumented wall; Q4 fallback projection route identified. | completed |
| **D08-M2** | hipEngine Q4 eager and graph decode profiles. | Eager role table reconciles 96.59%; graph dispatch count is captured. Graph/direct submission gap and core-vs-sampler/transport split remain. | in-progress |
| **D08-M3** | llama.cpp Vulkan Q4 pp512/tg128 perf-logger profiles. | **Complete:** all measured prefill and 128 decode graphs assigned; operation totals reconcile within logger rounding and submission residual is explicit. | completed |
| **D08-M4** | Repeat M1-M3 for Q8_0. | **Complete:** explicit host-embedding HIP route plus complete Q8 Vulkan operation map; no mislabeled quant row. | completed |
| **D08-M5** | Produce joined semantic-role Amdahl table. | **Complete:** every module was joined or represented by named submission residual; `other=0`; the resulting P1 admission is now accepted and this pre-P1 ranking is superseded. | completed |
| **D08-M6** | Mandatory post-P1 Q4 semantic replacement capture and rerank. | **Complete:** 99.60% prefill and 96.24% eager-decode reconciliation; P3 is first at 29.42% projected request saving, P2 second at 19.39%. | completed |
| **D08-M7** | Mandatory post-P6 Q4 semantic replacement capture and rerank. | **Complete:** 99.58% prefill and 96.59% eager-decode reconciliation; residual linear-attention projections are first non-exhausted at a corrected 10.06% request bound. | completed |
| **D08-M8** | Mandatory post-P4 Q4 semantic replacement capture and rerank. | **Complete:** 99.46% prefill reconciliation; Q direct markers fall 64.05%, every >=1% prefill package is exhausted, and P5 remains parked at 0.82%. | completed |

No implementation lane starts before `D08-C0` and the relevant M lane identify
a shipped owner. A trivial route correction from C0 may be retained immediately
if it passes the same correctness and benchmark gates; it is not “kernel work.”

### P lane — prefill, ordered by likely leverage but profile-gated

| ID | Candidate class | Potential if admitted | Admission signal | Hard bound / stop rule | Status |
| --- | --- | --- | --- | --- | --- |
| **D08-P1** | Fast-route/default/path selection: bulk rows, WMMA/MMQ projection coverage, correct AOTriton/native full-attention route. | **realized: +33.68% Q4 pp512 / +42.19% eager tg128** | Existing Q5T16 direct/rowtile/WMMA leaves beat dense BF16 on the actual QKV shape and pass full-model correctness. | **Closed:** one exact-role materialization/dispatch repair, no kernel variants, and M6 re-profile complete. | accepted |
| **D08-P3** | Dense Q4/Q5/Q6/Q8 gate/up/down projection kernels: tile, layout, activation reuse, and fusion. | **29.60% merged Q4 bound; unrealized** | All three frozen sole-resident candidates won pp512 but failed a required c1 or c8 operational guard. | **Closed:** no duplicate layouts, no full-model A/B, and no production change. Reopen only for a new sole-resident family that is non-regressive at every width. | rejected |
| **D08-P2** | GDN recurrence and convolution. Reuse retained GPF/LCP schedules before inventing a new one. | **realized: +4.33% paired / +5.83% independent Q4 pp512** | The 16K/16V shape exposes too few exact-LDS32 blocks; cluster8 wins the bounded screen and complete gate. | **Closed:** Q4-only quant/shape plugin policy; Q8 exact fallback retained; no new arithmetic variants. | accepted |
| **D08-P6** | Remaining linear-attention projections after accepted Q5 QKV routing: residual QKV/gate, alpha/beta, and SSM-out. | **realized: +14.18% graph-scope pp512 / +0.69% graph tg128; -46.69 MiB weights** | The split selected 35.93-ms Q5 SSM-out; exact-role sole Q5T16 passes 449/450 top-1, max KL 0.003273, and all graph pairs. | **Closed:** exactly three existing leaves and one combined full-model A/B; M7 confirms SSM-out at 9.68 ms. | accepted |
| **D08-P7** | Residual linear-attention QKV/gate, alpha/beta, and conversion after accepted Q5T16 SSM-out. | **3.58% selected gate bound; unrealized** | Q4T16 pp512/split-c8 win but c1 is 0.883x; raw Q4 regresses all operational widths. | **Closed:** preserve sole pack8, skip conditional source-F16 after exact-T16 c1 failure, and run no full-model A/B. | rejected |
| **D08-P4** | Full attention and RoPE/KV boundaries. | **realized: +4.79% graph pp / +1.41% graph tg; -4.13 MiB** | Exact-role sole Q4T16 passes every operational leaf, 447/450 top-1, max KL 0.003574, and exact trajectories. | **Closed:** gfx1151 0.8B Q-only plugin policy; M8 confirms Q at 2.71 ms and all other scope remains unchanged. | accepted/exhausted |
| **D08-P5** | Residual/norm/activation/copy launch coalescing. | **low: 0.82% combined M8 bound** | The combined measured package remains below the 1% continuation threshold. | Park until a fresh profile makes the package material; retain any independently measured exact non-regressive win. | parked |

### D lane — decode, ordered by the measured per-token ledger

| ID | Candidate class | Potential if admitted | Admission signal | Hard bound / stop rule | Status |
| --- | --- | --- | --- | --- | --- |
| **D08-D1** | Production graph replay, persistent buffers, and redundant sync/copy removal. | high if host/API gap >=10% | Host/API gap or launch count is material after C0. | One graph-vs-eager control and one sync/copy census. Graph/eager state must match; charge capture/instantiate/lifecycle honestly. | pending |
| **D08-D2** | LM-head/top-1. Q4_K_M uses the tied Q6_K table; Q8_0 uses Q8_0. | **low: 1.86% Q4 / 1.31% Q8 eager upper bound** | Reopen only if the graph census makes vocab projection the largest owner. | At most 3 leaf variants. Preserve full vocabulary and top-1/KL; no candidate-ID reranking or prompt-conditioned shortcuts. | parked |
| **D08-D3** | Dense projection GEMVs, including Q4/Q5/Q6/Q8 replacement/raw layout and wave geometry. | **high: 47.70% Q4 / 34.75% Q8 eager upper bound** | Projection bytes dominate after graph/direct timing scope is reconciled. | First complete M2; then at most 3 variants on a >2x-MALL cycling pool. No duplicate resident weights; hot kernel scratch-free. | blocked by M2 graph/direct census |
| **D08-D4** | GDN decode/conv and short-context full attention. | medium unless M5 says high | Either role is >=5% or has a clear Vulkan ratio. | One semantic owner at a time, at most 2 variants. Preserve recurrence/KV state, not only sampled token. | pending |
| **D08-D5** | RMSNorm, SiLU/GLU, residual, embedding, sampler, and token transport. | medium/low | Combined tail is material after D1-D4. | One boundary/census package at a time. Keep exact measured wins, but stop if complete-wall projection is <1%. | pending |

### G lane — promotion and closure

| ID | Work | Exit gate | Status |
| --- | --- | --- | --- |
| **D08-G1** | Full correctness and regression packet. | CPU-reference KL/top-1, deterministic repeats, touched-state checks, focused tests, and Q8 guard all pass. | pending |
| **D08-G2** | Same-session interleaved Q4/Q8 final comparison. | Q4_K_M hipEngine median pp512 and tg128 match or exceed fresh llama.cpp medians; Q8_0 does not regress from its accepted route; both timing scopes reported. | pending |
| **D08-G3** | Publish retained artifact/scoreboard/changelog and close campaign. | Exact commands and module ledger committed; no open required 0.8B work. | pending |
| **D08-T1** | Open 27B transfer campaign and re-profile from zero. | D08-G3 complete; no 0.8B ratio is copied as 27B evidence. | blocked by D08-G3 |

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
| Q4/16K/16V cluster8 GDN / P2 | **accepted** | **realized: +4.33% paired / +5.83% independent pp512; GDN -36.64%** | Complete Q4 semantic and production-graph decode gates pass; Q8 remains exact because its strict decode guard missed. | Closed; reopen only for a regression in this exact quant/shape key. |
| Q5T16 SSM-out / P6 | **accepted** | **realized: +14.18% graph pp / +0.69% graph tg; -46.69 MiB** | Exact-role sole Q5T16 replaces 18 dense-BF16 expansions; correctness and all production-graph pairs pass. | Closed; M7 confirms the route at 9.68 ms and selects the residual group instead. |
| Residual linear-attention projections / P7 | **rejected** | **3.58% selected gate bound; unrealized** | Native Q4T16 c1 regresses 11.73%; raw Q4 regresses every c1-c8 width; source-F16 cannot repair c1. | A new sole-resident family passes pp512 and every operational width, or an operation-complete fusion removes the c1 regression without sidecars. |
| Full-attention Q projection / P4 | **accepted/exhausted** | **realized: +4.79% graph pp / +1.41% graph tg; -4.13 MiB** | Six exact-role sole-Q4T16 residents pass 447/450 top-1 and every graph/eager trajectory; M8 confirms 2.71-ms Q and T16 WMMA bulk ownership. | Closed; reopen only for a regression in this exact role/shape key. |
| Graph/submission work | **M2 census next** | high if production graph residual >=10% | C0 certifies graph throughput, but eager stage gaps cannot be assigned directly to graph GPU work. | One graph/direct API and sync/copy census reports the production residual and preserves exact graph/eager state. |
| LM-head specialization | parked D2 | **low: 1.86% Q4 / 1.31% Q8 eager upper bound** | Joined M5 shows the vocab node is not the leading owner. | M2 graph census materially changes ownership and projects >=1% request saving. |
| Blanket non-temporal weight loads | rejected prior family | low | Prior gfx1151 cold-leaf improvement regressed/flattened complete decode by defeating useful MALL reuse. | New profile proves the exact production owner is cold-streaming, cache-polluting, and has a >=1% whole-request bound. |
| Generic wave64/reduction sweep | rejected/parked prior family | low | Cross-backend and GGUF campaigns already found no broad recovery; wave32 is the production contract. | A minimized hot kernel shows a specific wave32 occupancy/reduction bottleneck and a wave64-correct oracle. |
| Hand ISA or production Vulkan backend | parked | unknown/high cost | Current production-shaped combined HIP kernels often match or beat Vulkan micros; engine gap is not yet attributed. | A matched production semantic slice wins after route/layout/submission controls and projects >=10% request saving. |
| 27B dense transfer | blocked by D08-G3 | **critical future** | 0.8B must first establish route, module tools, and parity without copying shape-specific conclusions. | D08-G3 closes; begin with a fresh 27B C0/M ledger. |

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
