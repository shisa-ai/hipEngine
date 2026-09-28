# Qwen3.8-Flash-Next (Qwen4Exp) optimization ledger

- **Ledger date:** 2026-09-29
- **Host:** Framework `gfx1151`, Radeon 8060S / Strix Halo, machine
  `55ea6c509d0b49eea8de7094a1023668`
- **Model / quant:** unsloth `Qwen3.8-Flash-Next` `UD-Q4_K_XL`, BF16 KV, warm
  PLE, chunk1024, shipped `production` profile
- **Comparison basis:** same host, same model file, **different arithmetic** —
  halo-box PR #63 `c4aa30229` at 3760.4 ms of kernel time for one `code-p4096`
  prefill

**This is the single open-work list for closing the prefill and decode gap on
this model.** It is rewritten in place. Every row names a measured cost, the
evidence that bounds the opportunity, the first bounded experiment, and the
gate that decides it. Rows leave this list only by promotion, by recorded
rejection, or by a stated blocker — never by being forgotten.

## How this document relates to the others

| Document | Role |
| --- | --- |
| [`QWEN4EXP-STATUS.md`](QWEN4EXP-STATUS.md) | **State.** Rates, the competition, the mechanism ledger, certified and promoted scopes, qualification gaps. Not an experiment list. |
| **This ledger** | **Open work.** The ranked list of optimizations still worth trying, plus the rejection registry that stops us repeating failed ones unchanged. |
| [`QWEN4EXP-JOURNEY-REMAINING.md`](QWEN4EXP-JOURNEY-REMAINING.md) | Frozen dated inventory of the September 15 journey (`R01`-`R30`). Rows still live here carry their `R`-ID as a cross-reference. |
| [`QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md`](QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md) | Dated record. Its punchlist is closed (`R1`-`R6`, `R8` done, `R7` a standing bucket) and its numbers are on superseded compositions. |
| [`QWEN3.8-FLASH-NEXT.md`](QWEN3.8-FLASH-NEXT.md), [`QWEN4EXP-STRIX-JOURNEY-CAMPAIGN.md`](QWEN4EXP-STRIX-JOURNEY-CAMPAIGN.md), [`QWEN4EXP-WILKIN-RUNTIME-REVIEW.md`](QWEN4EXP-WILKIN-RUNTIME-REVIEW.md), [`QWEN4EXP-EXTERNAL-FORKS-REVIEW.md`](QWEN4EXP-EXTERNAL-FORKS-REVIEW.md), [`QWEN4EXP-CONTEXT-CAPACITY.md`](QWEN4EXP-CONTEXT-CAPACITY.md), [`QWEN4EXP-TOOLS-GRAMMAR.md`](QWEN4EXP-TOOLS-GRAMMAR.md) | Dated records and reviews. Evidence, not work order. |

## 1. Baseline anchor

The anchor is the shipped-default attribution at `42ce71156`: one `code-p4096`
prefill, 17196.5 ms profiled window, **16733.4 ms** attributed kernel time over
9652 kernels, 0 ms unattributed
([source](../benchmarks/results/2026-09-17-qwen4exp-shipped-default-attribution/README.md)).

Three retained changes postdate the anchor and are **not** in the family table
below: the repair-loop unroll (kernel total 16733.4 → 16673.2 ms), the
full-attention decode-context promotion, and its QK4 successor (largest launch
54.611 → 28.563 ms, +2.65% on the p4096 category mean). Re-read §2.5 before
spending QSA effort on the anchor's numbers.

`16733.4 − 3760.4 = 12973.0 ms` of gap against PR #63:

| Rank | Family | Ours ms | Ours % | PR #63 ms | Ratio | Gap ms | Gap share |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `dense_projection` | 5234.8 | 31.3 | 1374.3 | 3.81x | 3860.5 | 29.8% |
| 2 | `expert_gate_up` | 3402.4 | 20.3 | 761.8 | 4.47x | 2640.6 | 20.4% |
| 3 | `hyper_connection` | 2525.4 | 15.1 | 335.7 | 7.52x | 2189.7 | 16.9% |
| 4 | `expert_down` | 2382.3 | 14.2 | 412.8 | 5.77x | 1969.5 | 15.2% |
| 5 | `qsa_attention` | 1402.5 | 8.4 | 155.7 | 9.01x | 1246.8 | 9.6% |
| 6 | `moe_reduce` | 758.4 | 4.5 | 103.9 | 7.30x | 654.5 | 5.0% |
| 7 | `gdn` | 821.6 | 4.9 | 263.8 | 3.11x | 557.8 | 4.3% |
| — | `elementwise_norm` | 178.5 | 1.1 | 331.7 | 0.54x | −153.2 | −1.2% |
| — | `indexer` / `ple` / `other` / `quantize_pack` | 27.7 | 0.2 | 20.7 | — | 7.0 | 0.1% |
| | **total** | **16733.4** | | **3760.4** | **4.45x** | **12973.0** | |

Our 1420.9 ms of iu8 exact-repair sits inside the two MoE rows; the comparator
has no equivalent pass. Taxonomy boundaries are not operation boundaries
(`map_mmb()` puts `mmb_cvt_*` in `elementwise_norm`, while our activation
conversion is inside the dense kernel), so the smaller `elementwise_norm` row
is not evidence that conversion work is finished.

**Comparator numbers are opportunity indicators, not achievable speedups and
not quality-equivalent targets.** The cited records do not establish a matched
qualification under [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) §6.1.

## 2. Open rows

Ranked by recoverable complete-request milliseconds, not by leaf ratio. A leaf
ratio is how a row gets found; it is never how a row gets ranked.

### 2.1 `dense_projection` — 3860.5 ms of gap (29.8%)

Ours decomposes into the wide route (1404.2 ms over 1056 launches at layers
16-47), the exact coltile chain on layers 0-15 (3262.5 ms, 4-5x higher
per-launch), and 565 ms of other dense roles.

| ID | Mechanism / current measured cost | Opportunity and limit | First bounded experiment | State |
| --- | --- | --- | --- | --- |
| DP-1 | **Layers 0-15 still run the exact chain: 3262.5 ms.** The wide route claims 264 of 398 dispatches (0.663 against the scope's 32/48 = 0.667); the 134 declines are the layer scope and nothing else — role, quant, variant and shape are identical on both sides | Drift is a **monotone budget, not a cliff**: 5.81e-5 (32-47), 9.34e-5 (28-47), 2.53e-4 (20-47), 3.80e-4 (16-47, promoted), 6.47e-4 (12-47), 8.44e-4 (8-47), 1.099e-3 (0-47) — a measured ordering across seven scopes, not a guarantee for unmeasured ones or per category. 12-47 and 8-47 fail. The ~2.5 s full-extension figure is an inadmissible estimate | Cost-weighted layer/role sensitivity over the 0-15 role set, then gate the cheapest unmeasured scope (13-47 / 14-47 / 15-47, or a selective layer/role set). No blind full-gate sweep | OPEN (deferred, not closed). `R04`, `R07` |
| DP-2 | **Repeated in-kernel activation conversion.** The comparator converts activations to BF16 once per graph and caches them; our wide kernel converts inside the K loop and re-reads f32 | The F16 hoist is **already the default** (12-case interleaved A/B: every case +3.28-3.70%; 60 trajectories matched through 64 decode steps). What remains is sharing one cast between two consumers: **7.478 ms** of the packet's aggregate estimate. Operand type is f16, not bf16 (deliberate) | A true one-cast/two-consumer timing on real packet inputs, per-shape paired samples, finite checks. Do **not** make a general activation cache and its invalidation machinery a prerequisite for anything | OPEN, low value |
| DP-3 | **Split-F16-operand variant** (three WMMA products per tile) | Unmeasured option; not a measured 3x kernel cost or a 2x engine gain | Microbenchmark the split-operand tile against `dense_wide256` at the promoted shape before any scope work | OPEN, unmeasured |
| DP-4 | **Repair and packing tails** | From the September 10 ledger: the repair/packing tails are noise next to MMQ compute | Do not chase these on the dense side; spend repair effort in the MoE rows where the share is 17.5% | CLOSED as a dense target |

### 2.2 `expert_gate_up` — 2640.6 ms of gap (20.4%)

Post-unroll family cost is **3377.1 ms**. Attribution: main matmul 2775.1 ms
(82.5%), exact repair 627.3 ms (17.5%), packing/maps/reduction **0%**. The
family is weight-traffic-bound at a constant 104-126 GB/s, and cost is
`Σ over experts ceil(rows_in_expert / 16) × bytes_per_tile` with a 1.843 MB
gate/up tile — row count barely moves it (5120 rows cost 9.11 ms against 1280
rows at 8.65 ms, both exactly 512 tiles).

| ID | Mechanism / current measured cost | Opportunity and limit | First bounded experiment | State |
| --- | --- | --- | --- | --- |
| EG-1 | **Bytes read per output, at a fixed 16-row tile.** The tile count is set by expert occupancy, and the weight read is not shared across row groups | The obvious lever is measured dead: the 32-row M tile is rejected (below). What is left is reducing the bytes each tile reads, or sharing one weight read across more rows through a cache-resident path | Build the routed MMB-style wrapper (§2.3 `ED-1`) for gate/up as well as down: the comparator applies one tile geometry to both expert projections. Attribute the weight read per tile with `scripts/qwen4exp_moe_main_kernel_cost.py` before and after | OPEN — highest-value open row in the ledger |
| EG-2 | **32-row M tile** | Rejected twice: the 128-thread variant spills 492 B/thread past the 256-VGPR ceiling (4.8-8.1x slower); the 8-warp variant is spill-free, bit-identical, and still 1.3-2.2x slower. Real DRAM read falls **6.3%** where the tile-count model predicts 45%, because a taller tile makes the weight read shared between its two row groups and the sharing misses cache | Reopen only with compiler register-allocation evidence and a different risk-state layout, not by re-running the same tile | REJECTED |
| EG-3 | **Repair: 627.3 ms of the family** | The unroll already landed (family-wide repair 1420.9 → 1239.8 ms, −12.7%; Q4_K and Q5_1 sub-windows agree). Incidence-reduction and row-bitmap designs measured slower; instruction/load-scheduling remains a hypothesis | No experiment queued. Open only with a measured mechanism (registers, instruction count, occupancy, hardware-counter traffic), not a logical byte estimate | OPEN, no queued experiment |
| EG-4 | **Expert-population-aware tiles** | Uniform routing (exactly 20 rows/expert) is the *worst* case at 1024 tiles; real skew differs from any balanced model | Capture real per-expert populations at the production chunk, then re-run the tile screen on that distribution rather than a uniform one | OPEN. `R10` |

### 2.3 `expert_down` — 1969.5 ms of gap (15.2%)

Post-unroll family cost is **2229.8 ms** — the second-largest single owner and
2.23 s of the 12.97 s gap. The shipped kernel gathers Q5_1 blocks per lane, so
cost is `Σ over experts ceil(rows/16) × expert_tensor_bytes`: at the engine's
real routing (1120 tiles for 10240 compact rows) that is 1.261 GB per launch and
242 GB per prefill, a **2.19x over-read** against 0.576 GB of distinct expert
bytes. 242 GB at the measured 104-126 GB/s reproduces the 2229.8 ms.

| ID | Mechanism / current measured cost | Opportunity and limit | First bounded experiment | State |
| --- | --- | --- | --- | --- |
| ED-1 | **The missing routing wrapper, not a missing tile.** The comparator's move is llama.cpp's MMB (`mmb.cu`, PR #63); its tile is `MMB_BK=64`, `MMB_NT=256`, `MMB_LDS_STRIDE=BK+8` — **identical to our promoted `dense_wide256`** | We already own the geometry; we lack the wrapper that applies it to expert weights with an LDS-staged weight path. The family moved 64.1% for the comparator, the largest single move either engine made | Implement the routed wrapper on the existing wide tile for `expert_down` (43 layers Q5_1, five Q8_0 at 2/4/30/46/47), measure the complete chain including maps, packing, repair and reduction, then gate | OPEN — largest kernel-technology gap |
| ED-2 | **`LLAMA_MMB_BF16W=1`.** Part of the comparator's −64.1% is an ungated numerics trade | Our production envelope must be re-derived for bf16 weights. The router selection is the model's discrete amplification node: a near-tie flip changes which 512 experts run and turns a 4.9e-7 logit difference into an O(1) model-logit change | Measure bf16-weight drift with the router held fixed (teacher-forced), against the production envelope, before any engine A/B | OPEN, prerequisite for `ED-1` |
| ED-3 | **Role split and skew** | The Q5_1/Q8_0 split and the real expert populations set the cost of any wrapper | Reuse the `EG-4` population capture; test the wrapper on actual tails, not a balanced model | OPEN. `R12` |

### 2.4 `hyper_connection` — 2189.7 ms of gap (16.9%)

| ID | Mechanism / current measured cost | Opportunity and limit | First bounded experiment | State |
| --- | --- | --- | --- | --- |
| HC-1 | **`q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32` is the largest single kernel in the prefill: 2507.0 ms (15.5%), 384 launches at 6.529 ms.** Weight-bandwidth-bound at 59% of DRAM (987 MB/launch, 151 GB/s), 1.03 TFLOP/s = 3.4% of FP32 peak, operational intensity 6.8 FLOP/byte against a machine balance of ~117. **90% of its traffic is a 256x re-read of one 3.48 MB tensor**, because the dequantized weight is amortized over only `ROW_BATCH=4` rows | The limiter is load issue on the K=320 weight stream. The same weights through a 256-column tile reach 4.50 TFLOP/s | Raise the rows amortized per weight read (row batch or staged weight path) at the production shape, measuring the complete fused kernel including sigmoid and gated-mean epilogue | OPEN — largest single-kernel target |
| HC-2 | **GR wide tile** | 4.38x on the projection (1.4925 against 6.5379 ms at rows=1024, K=320, N=10240), worth ~1937 ms plus 579 ms on the down leg — and **rejected** by the 780-row canonical gate: mean KL 1.554e-3 against 1e-3, p95 7.081e-3, max 7.349e-2, top-1 768/780. Determinism, state repeatability and lifecycle passed, so the failure is purely numerical | Reopen only with changed arithmetic (compensated accumulation, split-F16 operands), and only after the router amplification node is addressed. The prefill-boundary KL screen does **not** predict the gate outcome — this route screened 285x better than the rejected iu8 route while carrying 300x more projection drift and failing at the same plateau | REJECTED |
| HC-3 | **The unconsumed 168 MB/layer GR gate write** | No arithmetic risk: the tensor is written and never consumed | Remove or fuse the write; measure the complete owner | OPEN |
| HC-4 | **The comparator's two matching mechanisms** | Its finished-stack ablation prices them independently: gate GEMM + mix round-tripping through memory at 1.19x, and bf16 streams end to end at 1.08x | Screen against the `HC-1` experiment rather than as a separate port; the two overlap | OPEN, folded into `HC-1` |
| HC-5 | **GR down leg** | 579 ms measured in the GR wide screen | Fold into `HC-1`; do not count separately from the up leg | OPEN, folded |

### 2.5 `qsa_attention` — 1246.8 ms of gap (9.6%)

**Read this family before trusting the anchor.** Its largest member is
`qwen35_paged_full_attn_decode_context_tensor_batch_kernel`, a full-attention
kernel whose name contains neither `qsa` nor `attention`-family markers that a
name filter would catch. At 1x it is 801.4 ms — 56.4% of the family and 5.0% of
the whole prefill — and it is now promoted twice.

| ID | Mechanism / current measured cost | Opportunity and limit | First bounded experiment | State |
| --- | --- | --- | --- | --- |
| QA-1 | **Full-attention decode-context leaf: 801.4 ms at 1x**, 3.2% of FP32 peak, 968 GB/s of L1-level K/V load traffic against 100.7 MB of unique data (a 7707x re-read), 25% occupancy and LDS-limited in the launch carrying 81% of its time | Two changes that touch no floating-point operation take the largest launch 54.611 → 28.563 ms (**1.912x**) and the p4096 category mean 248.487 → 255.066 tok/s (**+2.65%**). Byte-identical: `maxdiff = 0.000e+00` over all 1024 x 24 x 256 elements | **PROMOTED** (int32 token-offset table, QK unroll 4). A per-block base-offset table variant was implemented, measured 1.41x slower, and removed. Re-read the family total before ranking the rest | PROMOTED |
| QA-2 | **QSA sparse attention itself.** Both engines select the same positions by the same formula with the same budget (2051 = `indexer_top_k + compress_ratio - 1` on both sides), so the 9.01x — 6.58x under the most generous re-bucketing — is kernel technology. The comparator runs at an effective **11.93 TFLOP/s, above this host's ~9.5 TFLOP/s fp32 peak**, so it uses BF16 tensor cores; ours is scalar FMA | Ranked: (1) K/V reuse and LDS staging toward the 24 query heads sharing each KV head, intended to preserve row arithmetic; (2) tensor cores, which changes arithmetic and needs the production gate plus a strict fallback; (3) an occupancy sweep as a microbenchmark | Start with (1) and verify exact ownership, fixture outputs and integrated correctness. The indexer itself is already comparable (17.6 ms against 15.0 ms, 1.17x) — do not re-bucket it into a saving | OPEN |
| QA-3 | **The traced grid `(192, 1021, 1)` at block 32** that implied `query_heads` 768 against metadata `head_count` 24 | **There is no anomaly.** `rocprofv3` reports total *work-items* in `Grid_Size_X/Y`, not grid dimensions: the true grid is `(6, 1021, 1)` and `6 = query_heads/HEADS = 24/4`. Verified on three kernels whose launches are known from source. Production route is `strict_h256_head_quad_rows_spans` (`HEADS=4`), chosen by the profile | None | CLOSED (resolved) |

### 2.6 `moe_reduce` — 654.5 ms of gap (5.0%)

Bucket decomposition at 1x: router logits **617.73 ms (78.7%)**,
`weighted_lanes_sum` 67.66 (8.6%), `group_scatter_gather` 52.06 (6.6%),
`silu_mul_dual` 29.22 (3.7%), `router_select` 18.25 (2.3%). By class: **maps
81.0%, reduction 8.6%, scatter/gather 6.6%, activation 3.7%.**

| ID | Mechanism / current measured cost | Opportunity and limit | First bounded experiment | State |
| --- | --- | --- | --- | --- |
| MR-1 | **The router logits producer, 617.73 ms**, is 78.7% of the bucket and reduction-structure-bound, not traffic-bound | A warp-per-token reduction wins 2.16x on the kernel and **+1.65% end to end** (247.739 → 251.823 tok/s, non-overlapping arms) — and still fails admission (below) | See `MR-2`: the win is real and unreachable by this mechanism. Target a structure change that does **not** alter the reduction order — for example fusing the logits producer's weight read, or removing a pass | OPEN, mechanism re-targeted |
| MR-2 | **Warp-per-token router reduction: gate rejection** | Numerical **FAILED**, task **FAILED** (`requires_review`, 12/18 exact), state/determinism and lifecycle passed. Mean KL 5.714e-4 (budget 1e-3), p95 2.322e-3, p99 6.228e-3, max 3.344e-2 all pass; **top-1 agreement 98.67% (444/450) against >= 99%** fails. All six mismatches sit inside the 48 flip-eligible rows. Scope failures: `category/mixed_ja_en` (mean KL 1.083e-3) and `shape/prefill_last` / `transition/prefill_to_c1` (3.515e-3, 3.5x over budget) | **The router selection is the model's discrete amplification node.** Any reduction-order change moves the logits by ~1 ulp and flips the same near-ties, so a better kernel of the same shape cannot clear this gate. This is also why the GR family is closed to arithmetic change: a flip changes which experts run, so every downstream route inherits the saturation regardless of its own accuracy. Reopen only by making the selection itself exact | REJECTED (win retained as evidence) |
| MR-3 | **Reduction fusion** | Caps at 8.6% of the bucket | Do not start here; attribute maps first | OPEN, low value |

### 2.7 `gdn` — 557.8 ms of gap (4.3%)

| ID | Mechanism / current measured cost | Opportunity and limit | First bounded experiment | State |
| --- | --- | --- | --- | --- |
| GD-1 | `GDN_COLWARPS_PREFILL` +0.228 s, `GDN_PEER_PREFILL` +0.126 s, measured separately | Both inside the reported run-to-run spread and **not additive**. Our 821.6 ms is 3.11x the comparator's 263.8 ms | Do not re-screen unchanged. Only a complete-owner mechanism justifies a new arm | OPEN, no queued experiment. `R03` |
| GD-2 | Decode-side normalized/transposed GDN | Historical `P7`; needs a current decode anchor first | Depends on `CC-2` | BLOCKED on `CC-2` |

## 3. Cross-cutting levers

| ID | Lever | Evidence | First bounded experiment | State |
| --- | --- | --- | --- | --- |
| CC-1 | **Prefill chunk size (2048/4096)** | Dense half **measured**: 1382.2 → 1057.2 ms (−23.5%, 325 ms), and 2048 already captures 22.6% of it. MoE half **modeled from measured per-launch times**: 57.7 → 36.1 ms gate/up and 34.5 → 21.6 ms down per layer, i.e. 1.0 s + 0.6 s, because four chunks pay 4480 tiles per layer where one chunk pays 2801. Revised estimate ~**1.3 s**, not the recorded 1.7-1.9 s. A 2026-09-15 engine A/B did measure +3.94% on 4K PP (0.88 s of 22.37 s, 780/780 exact) — on a composition ~32% slower than today | Re-measure 2048 and 4096 against today's default on this host with **real per-expert populations** at each setting, reconcile against the 0.88 s result, and measure the **combination** with the F16 hoist rather than summing them. A chunk >= 2048 needs a correspondingly larger activation owner | OPEN — largest cross-cutting lever. `R11` |
| CC-2 | **Decode has no current-default matched measurement** | The last matched rows are historical (p512: hipEngine 20.396, halo-box HIP 15.441, Vulkan 27.477 tok/s) and the hipEngine composition is obsolete | Measure and attribute current-default decode separately before ranking any decode row. Prefill gains do not establish decode or MTP gains | PREREQ |
| CC-3 | **Decode host/launch overhead** | Exclusive bucket **5.3 ms/token (9.7% of wall)** at 54.6 ms wall / 49.3 ms device; ~1700 individually `hipLaunchKernel`'d kernels carry 31.2 ms of device work while the 18.1 ms MoE block is already `hipGraphLaunch`'d. The September 9 "decode graph expansion beyond the MoE block" estimate (3-5 ms/token) was followed by a negative result on wrapper tracing | Re-derive the recoverable share on the current tree before building. Do not port the September 9 estimate | OPEN, needs re-derivation. `R10`, `R22` |
| CC-4 | **Decode MoE expert GEMV pair** | 11.3 ms/token, ~5.6 ms at Vulkan parity (September 9 numbers). MoE subowners: gate/up 5.68, down/combine 5.60, shared-expert block 6.19, router 0.33, quantize/cast 0.29 ms | Refresh on the current default after `CC-2`; the shared-expert block is the largest sub-owner | OPEN, blocked on `CC-2` |
| CC-5 | **MTP economics** | Explicit and default-off, capped at 1K; rows <= 8 verifier storage, pooled state transactions and a serial oracle exist; final-head-only verification did not improve public economics | Rejection-depth, accepted-prefix and target/draft rollback before economics; a true no-MTP autoregressive baseline from the same protocol | OPEN, separate lane. `R27`, `R28` |
| CC-6 | **Long context (16K-256K)** | Native 262144 context is admitted and the 2051 dense-equivalent boundary is verified, but context-dependent costs and memory/admission limits at depth are unmeasured | Establish current context-dependent costs first. Do not transfer short-context rankings to long context | OPEN, separate hypothesis. `R28` |
| CC-7 | **Qualification accounting** | `scripts/execution_profile_q8_wmma_prefill_layers_gate.py:426` sets `profile_qualification_claim = quality_passed and deterministic` with no coverage term, while `:481/:482/:497/:498` expose `--case-id`, `--limit`, `--decode-steps` (default 24) and `--repeat-runs`, so a reduced run can claim full-profile qualification. The promoted 16-47 artifact is full coverage, so this is prevention rather than repair. `--prefill-chunk-size` also defaults to 1024 | Add the coverage term; re-run the gate at any new chunk | OPEN, prevention. `R30` |
| CC-8 | **Missing qualification evidence for the promoted 16-47 scope** | No category-heldout run, candidate-local teacher (not §6.1's named strict comparison), no isolation/graph/lifecycle verdicts, no task-quality verdicts, no BF16-relative non-inferiority. Missing evidence is not a failed check | Close the applicable fields before claiming full qualification or extending admission | OPEN, prerequisite. `R29`, `R30` |
| CC-9 | **Other models, quants and selector cleanup** | Obsolete selectors keep their rollback/re-gating role until the replacement is proven | Retire selectors per [`REFACTOR.md`](REFACTOR.md); other quants and models need their own scope and quality evidence | OPEN, maintenance |

## 4. Closed and rejected registry

Rows here are **not** to be re-run unchanged. Each names the condition that
would make it a new candidate rather than a repeat.

| Row | Verdict | Evidence | Reopen condition |
| --- | --- | --- | --- |
| MoE 32-row M tile (128-thread and 8-warp) | Rejected | Real DRAM −6.3% against a modeled 45%; 1.3-2.2x slower; 492 B/thread spill | Compiler allocation evidence plus a different risk-state layout |
| Warp-per-token router reduction | Rejected at gate | Kernel 2.16x, e2e +1.65%, top-1 98.67% | Making the router selection itself exact |
| GR wide tile | Rejected at gate | mean KL 1.554e-3, max 7.349e-2, top-1 768/780 | Changed arithmetic (compensated accumulation, split operands) |
| GR iu8 up / down (`GR_IU8`, `GR_IU8_DOWN`) | Numerically rejected | Model gates fail; P4 removed despite better operand MSE | A structurally different optimization, not the rejected variants |
| Dense Q8 iu8 (generic `Q8_IU8_WMM`) | Gates fail | Generic and counted non-GR-only 780-row gates fail | Operand localization first |
| Dense wide scopes 12-47, 8-47, 0-47 | Failed | `general_en` mean 1.089e-3 (12-47); max KL 7.30e-2, top-1 1532/1548 (8-47); mean 1.099e-3 (0-47) | A cheaper unmeasured scope, or changed arithmetic |
| IU4 grouped prefill | Bounded out | `v_wmma_i32_16x16x16_iu4` at 1.892x nets 0.945x precision-matched; residual ceiling 1.06x | A different precision split with a proven budget |
| Q5_1 iu8 exact down chain | **Promoted** 2026-09-08 | 20/20 exactness tests, 1.62-1.70x operation-complete, zero flip escapes at multiplier 4.0 | — |
| MMQ ds4 chain | Not a promotion vehicle | Single-plane pack paired with a three-plane consumer; 0.75-0.78x of pair2 | A consistent three-plane residual pack with its own exactness case |
| `Q4_IU8_PREFILL`, `PRODUCTION_MOE_PREFILL` overrides | Slower on this case | Disabled-family sweep | New tiling, packing, fusion or repair work — not these switches |
| GDN column warps / peer | Inside noise | +0.228 s / +0.126 s, not additive | A complete-owner mechanism |
| Per-block base-offset table (full-attn) | Removed | 1.41x slower than the promoted per-token table | — |
| Journey restorations `R01`, `R02` (GDN tiled / multi-column) | Not promoted | Numerics pass; EOS/task criteria fail (17/18 pairs, one truncated) | A passing predeclared task review |
| Per-token LDS `lds_row_atrisk` at 32 rows | Fixed | Latent uninitialized-shared-memory bug found while templating | — |

## 5. Measurement rules for this model

1. **The 1x authority is `role-analysis.json` `exact_role_kernels`**
   ([`2026-09-17-qwen4exp-per-role-cost`](../benchmarks/results/2026-09-17-qwen4exp-per-role-cost/)).
   The raw `rocprofv3` CSV in that directory is **two repetitions**. Three
   artifacts published 2x launch counts before this was found and corrected on
   2026-09-27; check the authority before quoting any count.
2. **Do not classify families by kernel-name substring.** The QSA family's
   largest member is a full-attention kernel, and the dense conversion work sits
   inside the dense kernel rather than in an `mmb_cvt_*` symbol.
3. **Do not add the A/B savings** in STATUS §2.4. They are separate wall-time
   experiments on different compositions, not a decomposition of the gap.
4. **Rank by recoverable complete-request milliseconds.** A leaf ratio finds a
   row; it never ranks one.
5. **Production candidates do not need bit-exactness.** The binding contract is
   the calibrated production envelope in
   [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md): exact control/ownership in
   every profile, plus the numerical, determinism, isolation and task gates. A
   byte-identical change (`QA-1`) is cheaper to justify, not more admissible.
6. **A passing KL envelope beside a failing top-1 is a discrete amplification
   node**, not a sloppy kernel. Check `flip_eligible_rows` before blaming
   arithmetic.

## 6. Cadence

- Re-rank after every retained change. A promotion invalidates the anchor's
  family table for its own family and for any family whose work it moved.
- Update [`QWEN4EXP-STATUS.md`](QWEN4EXP-STATUS.md) in the same commit as the
  promotion that changes it, and this ledger when a row changes state.
- A row is open, measured-negative, promoted, or blocked with a concrete
  prerequisite. An untested row is open, never complete.
- New candidates enter through this ledger, not through a new campaign document.
