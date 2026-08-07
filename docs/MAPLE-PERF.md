# MAPLE — gfx1151 Prefill & Decode Performance Plan

Last updated: 2026-08-07 (branch `maple`)

This is the authoritative performance punchlist for
`deepgrove/maple-preview-2bit-mlx` on Radeon 8060S / `hip_gfx1151`. It reuses
the established hipEngine methodology — **profile first (Lesson 0), then attack
launch overhead, then batched compute** — from `docs/TUNING-gfx1151.md`,
`docs/PREFILL.md`, `docs/GGUF-PREFILL-OPTIMIZATION.md`, and the gfx1151
roofline. Architecture and the current bring-up baseline live in
`docs/MAPLE.md`; kernels are catalogued in `docs/KERNELS.md`.

## 1. Hardware roof and current position

Device (from `docs/ROOFLINE-gfx1151.md`): 40 CU / 80 SIMD32, 2 MiB L2, 32 MiB
L3, ~256 GB/s LPDDR5X, 59.4 TFLOP/s BF16-WMMA, 118.8 TOP/s INT4-WMMA @ 2.9 GHz.

| Quantity | Value | Source |
| --- | ---: | --- |
| Public native prefill 128/320/512 | 700.643 / 649.280 / 614.874 tok/s | retained P0/M5 recertification |
| Current c1 decode profile | 163.459 tok/s (6.118 ms/token) | corrected cached trace |
| Fixed-helper c8 decode64 | 299.181 aggregate tok/s median | M6 recertification |
| HIP kernels per c1 decode token | 295 | corrected cached trace |
| Exact affine4 lm-head payload | 166.922 MiB | packed weight + BF16 scale/bias |
| Public tracked residency (max context 512) | 4.988 GiB | retained P0/M5 recertification |
| Native-prefill limit | 512 tokens; serial fallback above | public generator contract |

**Current conclusion (P0 retained).** Final-row-only sampling nearly doubles
qualified native prefill and makes true expert-major MoE the next owner. A
clean cached-only `rocprofv3` trace at retained P0 now supplies the active
prefill attribution. The c1/c8 rows remain the corrected decode baselines
because P0 does not alter those paths:

| Current phase | Wall / unit | Kernel / unit | Host gap | Launches / unit | Useful throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| public native prefill320, post-P0 | 498.442 ms/request | 492.866 ms | 5.576 ms (1.12%) | 586 | 642.000 tok/s |
| autoregressive c1 decode | 6.118 ms/token | 5.035 ms | 1.082 ms (17.69%) | 295 | 163.459 tok/s |
| fixed-helper c8 decode | 27.256 ms/batch | 25.337 ms | 1.919 ms (7.04%) | 293 | 293.514 aggregate tok/s |

After P0, the exact affine4 head owns only **0.30%** of prefill320 kernel time,
while gate/up + down owns **56.03%**. The unchanged corrected decode baseline
still attributes **28.75%/46.52%** of c1/c8 to the head and
**22.55%/29.67%** to selected experts. The paths therefore have different
actions:

- **Prefill:** P0 now samples only the final prompt row. Qualified
  128/320/512 throughput is **700.643/649.280/614.874 tok/s**, versus
  339.890/326.573/317.488 before P0. Max-context-512 residency falls exactly
  **148.813 MiB** to **4.988 GiB**.
- **c8:** all request rows require logits. A rows>1 affine4 tile that streams
  each 166.922-MiB payload once across request rows remains the correct owner.
- **c1:** preserve the proven head kernel until a new layout wins. A controlled
  3+3-process review measures graph at only **1.0047x** (+0.47%, 0.033 ms), so
  graph stays opt-in. The 5.035-ms kernel-only roof is 198.591 tok/s; exact
  200+ requires router/head kernel work.

The remaining prefill correction is that
`maple_selected_ternary_*_batched` is a row/route gather grid, **not grouped
MoE**: expert weights are reread for every assignment. True device
count/prefix/scatter plus expert-major ternary kernels is now the active owner.
The post-P0 profile measures this family at **276.150 ms**, or **56.03%** of
kernel time. With every other measured bucket fixed, 1000 tok/s requires
**<=97.708 ms (2.826x)**; that is the frozen P1 implementation gate.

Current exact priority:

1. P1 true expert-major compact ternary MoE.
2. P2 exact qrow/GQA-reuse attention; P3 dense ternary tile sweep.
3. D0 one-dispatch c1 router then affine4-head bandwidth work.
4. D1 c2/c4/c8 affine4 row reuse.

DeepGrove's published M4 table supports this split: **169 tok/s exact**, **218
tok/s with approximate FlashHead**, and **1075 tok/s prefill**. Its generation
path avoids discarded prefill heads and sorts >=64 routed assignments by
expert. The table omits workload/repeat/software details, so it is directional,
not a cross-device benchmark.

Evidence:
`benchmarks/results/2026-08-07-gfx1151-maple-p0-final-row-prefill-retained.json`,
`benchmarks/results/2026-08-07-gfx1151-maple-p0-phase-profile.json`,
`benchmarks/results/2026-08-07-gfx1151-maple-corrected-phase-profile.json`, and
`benchmarks/results/2026-08-07-gfx1151-maple-c1-graph-review.json`.

## 2. Phase 0 — Profile before optimizing (gate: no math changes) — DONE

`scripts/maple_profile.py` (prebuild then cached-only `rocprofv3 --kernel-trace`
of a warm decode step) produced `benchmarks/results/2026-08-07-gfx1151-maple-decode-profile.json`.

### Post-M3a c1 profile (historical; superseded above, 2026-08-07)

This earlier fixed-token profile followed the parallel-router change but
predates the corrected M5/M6 reprofile above. It remains useful as M3a
historical attribution. One token = 9707, 4 warmup + 32 measured steps.

| Family | µs/step | Share | Note |
| --- | ---: | ---: | --- |
| affine4 lm_head | 1,236 | 22.5% | grid 19.4M/wg 128; weight-bandwidth bound (M3b tiled = dead end) |
| selected dual gate/up | 848 | 15.4% | grid 524K/wg 128 |
| selected down | 745 | 13.6% | grid 524K/wg 128 |
| router softmax_topk | 722 | 13.2% | parallel router kernel B |
| router logits | 382 | 7.0% | parallel router kernel A |
| attention decode | 563 | 10.2% | — |
| QKV ternary | 407 | 7.4% | — |
| o_proj ternary | 284 | 5.2% | — |
| qknorm+RoPE+KVwrite | 89 | 1.6% | — |
| norms / swiglu / residual / embed / argmax | ~211 | ~3.8% | — |
| **host gap** | 545 | 9.0% of wall | 6,038 µs wall, 5,492 µs kernel (90.1% kernel) |

M3a (parallel router) cut the step from 12,758 → 6,038 µs (2.11×) and the
router from 7,807 → 1,104 µs (7.1×). **Remaining top targets: affine4 lm_head
(22.5%), selected-expert gate/up + down (29%), router kernel B softmax/topk
(13.2%).** lm_head and selected-expert are weight-bandwidth bound: their tiled /
grouped variants measured bit-exact but NOT faster (M3b, M3c), so the lever is
weight reuse across a batch (M4/M5 prefill, M6 batch decode) rather than
per-c1 GEMV shaping. The historical ~9% host gap was only a graph ceiling; the
current M1 review measures 0.47% actual wall recovery.

### Original M0 profile (pre-M3a, superseded)

Per-step kernel breakdown (24 layers/token, one token = 9707):

| Family | µs/step | Share | Grid/WG (per call) |
| --- | ---: | ---: | --- |
| router_topk | 7,807 | 61.0% | **grid 256 / wg 256 (one block, serial)** |
| lm_head affine4 gemv | 1,234 | 9.7% | grid 19.4M / wg 128 |
| expert gate/up (dual) | 822 | 6.4% | grid 524K / wg 128 |
| expert down | 749 | 5.9% | grid 524K / wg 128 |
| attention decode | 562 | 4.4% | — |
| QKV ternary | 434 | 3.4% | — |
| o_proj ternary | 296 | 2.3% | — |
| qknorm+RoPE+KVwrite | 89 | 0.7% | — |
| add_rmsnorm / rmsnorm | 121 | 0.9% | — |
| weighted_residual / swiglu / embed / span / argmax | ~94 | 0.7% | — |
| **host gap** | 549 | 4.3% | — |

**Highest-leverage target = the router.** Each `router_topk` call is a single
256-thread block (grid 256/wg 256) that serially computes logits and the
softmax/top-8 over all 256 experts, ~362 µs per call × 24 layers. Parallelizing
it (grid over experts / vectorized 2048-wide dot) is the single largest win
(see D3/M3). lm_head and selected-expert GEMVs follow. hipGraph (D1) could at
most recover this historical ~4.3% host gap and is deprioritized below kernel
work.

### M0 methodology (reuse)

Follow `docs/TUNING-gfx1151.md` Lesson 0: capture a `rocprofv3 --kernel-trace`
of one warm decode token and of a short serial prefill before changing anything.
Produce a kernel-time vs host-gap breakdown:

- Per-kernel `DurationNs` by family (ternary GEMV, selected-expert, router,
  attention, qknorm, affine4 embed, affine4 lm_head, argmax, norms).
- **Host gap** = step wall − Σ kernel time. This is the launch/dispatch budget
  that hipGraph capture attacks (see D1).
- Which single kernel dominates (confirmed: the serial 256-expert router).

Deliverable: a `docs/KERNELS.md` Maple trace block and a
`benchmarks/results/2026-08-07-gfx1151-maple-decode-profile.json` artifact.
Prebuild the `.so` and use `require_cached` so the profiled process never spawns
`hipcc` (per `docs/KERNELS.md` JIT gotcha).

## 2b. Historical milestone disposition

| Milestone | Result | Current interpretation |
| --- | --- | --- |
| **M3a parallel router** | Done; router 7,807 -> 1,104 us/step and decode 12,758 -> 6,038 us in the post-M3a trace | Retained. The two-dispatch router is still the first exact c1 kernel target. |
| **M3b c1 affine4 tile** | Rejected at 0.96x | Do not revive the same tile; use a materially different layout/bandwidth schedule. |
| **M3c c1 eight-route grouping** | Rejected at 0.69x | This tested activation reuse within one token, not expert-major reuse across prompt rows. |
| **M4/M5 batched prefill** | P0 retained at 615-701 tok/s | Exact through 512; P1 grouped MoE remains open. |
| **M1 c1 hipGraph** | Exact but only 1.0047x in the current review | Keep opt-in; do not use the historical host-gap estimate as a speed claim. |
| **M6 batch decode** | c2/c4/c8 helper at 218.818/261.099/299.181 aggregate tok/s | Retained helper; public scheduler/server integration remains open. |
| **M2 fusion** | Exact composites regressed kernel efficiency | Keep opt-in/fallbacks; no default promotion. |

M3a evidence:
`benchmarks/results/2026-08-07-gfx1151-maple-router-parallel.json`.

## 3. Decode plan (kernel-bound → optimize the hot kernels)

### D1 — hipGraph-capture the whole token step (implemented, opt-in)

M0 shows the host gap is only ~4.3% (549 µs/step), so capture is a secondary win
versus kernel-level work. The current review confirms that this c1 graph does
not materially improve wall time and is not reused by batch or prefill. Pattern
and caveats below.

**M1 status: DONE (opt-in).** `MapleGraphCache` captures the whole stateless
decode step (24-layer loop + final norm + affine4 lm_head + argmax) as one graph
and self-validates bit-exact on first capture (argmax parity vs a fresh eager
reference, with the output cleared so a no-op graph is rejected). The capture is
bit-exact across 120 tokens over a growing KVLiveSpans cache
(`tests/test_maple_graph_capture.py`). A post-correction counterbalanced review
(three independent eager and three graph processes, 4 warmup + 128 measured)
measures eager **7.0661 ms** versus graph **7.0329 ms** median-of-medians:
**1.0047x**, only 0.033 ms. IDs and top-logit hashes match, every graph reports
capture=1/replay=131/reject=0, and close returns tracked buffers to zero.

Keep `HIPENGINE_MAPLE_GRAPH=1` opt-in. The M6 helper has its own eager batched
kernel chain and does **not** reuse this c1 graph. Evidence:
`benchmarks/results/2026-08-07-gfx1151-maple-c1-graph-review.json` (current) and
`benchmarks/results/2026-08-07-gfx1151-maple-hipgraph-c1.json` (historical).

The repo already proved this pattern in `hipengine/runtime/moe_graph.py`: a
**stateless** per-layer unit is safe to capture and replay across arbitrarily
many relaunches (bit-exact in `tests/test_hip_graph_capture_replay.py`). Maple
decode is stateless per token: every layer recomputes from fresh
`hidden`/`qkv` buffers through fixed session-resident scratch pointers. The only
stateful memory is the KV cache, but the KV-write/attention kernels read and
write by **absolute `token_positions`** with `KVLiveSpans`, and all span
metadata (`live`/`token_positions`/`evict`/`row`) is passed as **device
pointers** read at graph-execution time — so a graph captured against fixed
buffer pointers stays valid across tokens (the eager span update runs before
each replay).

Steps:
1. Capture the whole 24-layer + tail body into a `MapleGraphCache` (M1, done).
   The per-token eager span update (`_publish_span_position`) and embed remain
   host launches before the graph; the final argmax host copies stay after.
2. Validate bit-exact on first capture, replay thereafter, eager fallback on
   any mismatch (done, with output-clearing to catch no-op graphs).
3. Because attention reads a growing cache, the captured graph must stay
   correct across positions — the device-pointer span ABI guarantees this, and
   the 120-token parity test pins it.
4. Replay only — Python runs once per token for sampling, not 271 times.

Guard: capture must not change arithmetic; the eager path is the reference and
the graph is only kept when bit-exact (self-validation enforces this).

### D2 — Fuse kernels to cut launch count

Fewer, bigger kernels reduce both launch overhead and intermediate memory
traffic (hidden buffers are bf16 2048-wide and round-trip device memory each
step).

- **Layer boundary norm + embed/output**: fuse `paro_rmsnorm` → next layer or
  `paro_add_rmsnorm` → selected-expert start into single kernels.
- **Router logits → softmax/top-k/renormalize**: M3a made both stages
  parallel, but still launches two kernels per layer. Evaluate DeepGrove's
  last-threadgroup/atomic-counter pattern as a one-dispatch HIP composite
  before attempting router-to-expert fusion.
- **QK-norm+RoPE+KV-write → attention**: already one fused qknorm write; fold
  the online-softmax attention into the same kernel or a tight two-kernel
  chain to keep Q/K/V in flight.
- **Expert down-projection + weighted residual + layer output**: fold
  `maple_selected_ternary_gemv` (down) + `clamped_swiglu` scaling + weighted
  residual into one selected-expert consumer.

Each fused composite must keep a numerically-equivalent unfused fallback
(architecture invariant).

**M2 status: investigated, not promoted.** Built two bit-exact fused composites, both
opt-in (`=1`) with the unfused chain as the fast path:
- `maple_moe_dual_swiglu_bf16` (`HIPENGINE_MAPLE_FUSE_MOE`): fuses the MoE dual gate/up
  gemv + clamped SiLU. Bit-exact (identical packed-oracle max/mean KL + top-1), cuts decode
  launches 295→271, but the fused kernel is ~9% slower per MoE layer (interleaved micro
  229.7 vs 210.7 us), so it would regress the optional M1 c1 hipGraph path where launches
  are amortized. The M6 helper does not use that graph.
- `maple_attention_fused_qknorm_decode_bf16` (`HIPENGINE_MAPLE_FUSE_QKATTN`): fuses the
  per-layer qknorm_rope_kv_write + attention_decode into one kernel (self-contained per
  q-head block with redundant in-group K/V write). Bit-exact (attention output + K/V cache
  match bit-for-bit), but ~1% slower per decode step (interleaved eager 6442 vs 6377 us);
  the in-group K/V write redundancy and folded 88 us qknorm work offset any launch saving.
- Rejected/reverted: fused down gemv + weighted residual (~3.5% slower, serial-expert grid).
Blockers for all three are recorded in `docs/REFACTOR.md`. Conclusion: on this small-batch
hardware, kernel fusion regresses efficiency (norm/qknorm kernels are small and attention
is history-bound), while the current c1 graph recovers only 0.033 ms. No fusion is
promoted; all unfused fallbacks (the invariant) are preserved.

### D3 — MoE routing and grouped selected-expert

The historical M0 router was the 7.8-ms / 61% bottleneck; M3a already replaced
it with parallel logits plus parallel stable top-k. In the corrected c1 profile,
router logits + top-k still consume **1.108 ms / 22.0% of kernel time**. The
next c1 candidate is DeepGrove's one-dispatch last-threadgroup composite, not a
return to the serial dot-product kernel.

The rejected M3c experiment grouped the eight routes of one c1 token and lost
because the shared activation was already L2-cached. It says nothing about
prefill row reuse. True **grouped MoE** scatters hundreds or thousands of routed
rows by expert and runs grouped bulk kernels; that is current P1 and is required
by `docs/PREFILL.md`.

### D4 — lm_head fast paths

`maple_affine4_gemv_f32` spans 151,936 output rows (affine4, 2048 in) plus a
two-stage FP32 argmax and now measures 1.448 ms / 28.75% of c1 kernels.

Exact work, in order:

1. Keep the rejected c1 tiled shape closed; test a materially different
   affine4 layout/launch or weight-bandwidth schedule.
2. For c2/c4/c8 only, tile request rows so each packed weight row is streamed
   once across requests.
3. Head+argmax fusion is secondary: argmax is only 0.008 ms/token.

Separately, add **FlashHead** only as an opt-in approximate path. The official
configuration probes 512 of 4,748 clusters (16,384 candidate tokens plus forced
controls) and is the mechanism behind DeepGrove's 218 tok/s M4 headline. It
needs full category/heldout exact-head agreement and quality reporting; it is
not part of the exact 200+ target.

### D5 — Batch decode (c>1) and continuous batching

Multi-request decode is independent of the c1 graph and should reuse weight
traffic across requests:
- Multi-column / MMQ-style ternary decode kernels for c=2/4/8 (the current
  row-GEMV head still rereads weights per request).
- Batched GQA attention decode and KV append with a batch grid dimension.
- A continuous-batching owner loop for admission, prompt prefill, decode,
  sampling, and reclaim, reusing the GGUF/PARO server patterns.

**M6 fixed-capacity runtime helper: DONE; public server integration: OPEN.**
Batch decode is implemented with separate request-local SWA/global rings,
sparse active masks, offset-correct reclaim, and c-specific real-checkpoint
trajectory gates. `MapleBatchRunner.batch_step` runs c requests through the full
batched ternary chain; `MapleContinuousBatcher` owns fixed slots and reclaim,
but is not wired into the public generation scheduler or a server endpoint.

The corrected Radeon 8060S/gfx1151 c=2/4/8 medians are
**218.818/261.099/299.181 aggregate tok/s** at 64 tokens/request. Every repeated
measured trajectory matches c1, the 18-prompt natural-derived seed gate passes
(including a sparse final c=8 group), tracked residency is
**4.951/4.958/4.973 GiB**, and close returns ownership to zero. The former
223.2/275.6/321.1 rows are invalid because their artifact mislabeled the device
as W7900/gfx1151 and gated c=1 rather than the measured widths. Evidence:
`benchmarks/results/2026-08-07-gfx1151-maple-m6-batch-decode-recertified.json`.

## 4. Prefill plan (current 615-701 tok/s -> exact 1,000+)

The public path is already a correct `[T, hidden]` bring-up through 512 tokens,
with 256-row chunks and `KVLiveSpans` append-position `start + r`. The remaining
work is ordered by the corrected profile rather than by the historical landing
sequence.

### P0 — Final-row-only sampling tail — DONE, RETAINED

After every chunk, the runtime preserves the complete hidden/KV update and runs
final RMSNorm, exact LM head, and argmax only for the final row of the final
chunk. Public prefill reuses the proven c1 logits/argmax buffers; the all-row
path and buffers remain batch-only.

The clean qualified result is **700.643/649.280/614.874 tok/s** at
128/320/512, improvements of **106.14%/98.82%/93.67%** over the corrected
bring-up baseline. All **18/18** natural+heldout prompt states match serial
byte-for-byte across final hidden/normalized values, all live K/V bytes, and
both `KVLiveSpans` metadata sets; all **90/90** seed/continuation logits and
tokens match with KL 0. Tracked max-context-512 residency falls
**5.133 -> 4.988 GiB (-148.813 MiB)** and close returns zero ownership.

Evidence:
`benchmarks/results/2026-08-07-gfx1151-maple-p0-final-row-prefill-retained.json`.

### P1 — True expert-major compact ternary MoE — OPEN, NEXT

- Router over all rows -> registered device count/prefix/stable-scatter by
  expert -> grouped gate/up/down -> inverse lane mapping before exact weighted
  combine.
- Reuse the generic GGUF/PARO group-scatter producer ABI, but register Maple
  ternary consumers under their own `(backend, layer, quant, variant)` keys;
  do not add runtime quant/backend branches.
- Preserve every per-row BF16 projection, clamp/SwiGLU, and weighted-sum
  boundary. Keep the current row/route gather chain as the oracle/fallback.
- Gate: reduce the expert family from **276.150 ms to <=97.708 ms (2.826x)**
  in the clean post-P0 prefill320 profile. Also pass the P0
  18-prompt/continuation/state/lifecycle gate.

### P2 — GQA/query-row prefill attention — BRING-UP LANDED; TUNING OPEN

The batched Q/K/V projection, head RMSNorm+RoPE, KV append, and causal ring
attention are correct through 512. The 62.995-ms attention family currently
uses one block per `(query head,row)`, rereads each KV stream for all four query
heads in a GQA group, and barriers once per key. Transfer the exact qrow/GQA
reuse pattern or evaluate AOTriton only after P0/P1 reprofile. Extending native
prefill beyond 512 separately requires append/attend orchestration that cannot
overwrite a still-visible SWA prefix.

### P3 — Dense ternary row-tile sweep — BRING-UP LANDED; TUNING OPEN

Dense QKV/O kernels already reuse one packed weight row across a fixed
eight-row tile and consume 122.555 ms at prefill320. Sweep exact 8/16/32 tiles
while preserving each output's reduction order. Consider INT4/BF16 WMMA only
after that sweep; a different contraction order cannot replace the byte-exact
state contract merely to hit a throughput target.

**Landed prefill primitives (2026-08-07, bit-exact vs CPU oracle):**

| Primitive | Commit | Purpose |
| --- | --- | --- |
| `maple_ternary_gemm_bf16` | `7c1624080` | `[rows,out]=[rows,in]x ternary W`; eight-row weight reuse |
| `maple_ternary_qkv_gemm_bf16` | `a4b9808d8` | full `[T, q+2kv]` projection |
| `maple_qknorm_rope_kv_write_batched_kernel` | `d7185caaa` | row-batched QK norm, RoPE, and ring write |
| `maple_attention_prefill_ring_kernel` | `ffe822eae` | causal attention through `KVLiveSpans` |
| `maple_router_topk_parallel_batched_kernel` | `3820527ed` | row-batched logits and stable softmax/top-k |
| `maple_selected_ternary_*_batched_kernel` | `2bdd79497`, `bb3ac7569` | correct row/route-gather MoE oracle/fallback, not grouped MoE |
| `maple_affine4_gemv_batched_kernel` | `8e7b400db` | all-row debug/batch head, not the public-prefill target |

RMSNorm/add-RMSNorm already accept rows, and
`maple_affine4_embed_batched_kernel` (`66c5a7a11`) provides batched embedding.

### P4 — Chunked admission and long prompts — PARTIAL

`prefill_native(...)` is public and correct through 512 tokens. Open work is
safe SWA orchestration beyond 512 and public packed prompt admission for the M6
scheduler helper. Serial remains the correctness fallback above 512.

### Prefill target

P0 delivers 649.280 tok/s at 320, close to its 645.811 tok/s projection.
Crossing 1,000 now requires P1 plus later layer work. The clean post-P0 profile
freezes the P1 target at expert-family **<=97.708 ms (2.826x)** versus the
current **276.150 ms**. DeepGrove's 1075 tok/s M4 row avoids discarded heads
and sorts routed rows by expert, independently
showing that 1,000+ is credible. The near-term exact target remains
**>=1,000 tok/s at qualified 128/320/512 shapes**; 2,000 is a later
dense/attention target.

## 5. Milestones and evidence policy

| # | Milestone | Gate | Evidence |
| --- | --- | --- | --- |
| M0 | Decode profile (host-gap + per-kernel) | profile artifact, no math change | `...-maple-decode-profile.json` (done) |
| M3a | Parallelize router topk | exact one-ULP; decode wall ↓ 61% | `WORKLOG.md` + artifact |
| M3b | lm_head affine4 | exact; dead-end (tiled 0.96×, weight-bandwidth bound) | `WORKLOG.md` |
| M3c | c1 eight-route grouping | exact; dead-end (0.69x, shared activation already L2-cached); does not close row-bulk grouped MoE | `WORKLOG.md` |
| M4 | batched ternary prefill primitives | exact vs packed oracle; prefill tok/s up | GEMM + QKV GEMM primitives in (done) |
| M5/P0 | final-row native prefill | exact through native 512-token limit; serial fallback above it; row/route-gather MoE remains to replace | 700.643/649.280/614.874 tok/s; 18/18 byte-exact states and 90/90 positions |
| M1 | D1 graph-captured c1 decode | bit-exact vs eager; no default promotion | current review 1.0047x / 0.033 ms, opt-in |
| M6 | D5 batch decode helper | exact c2/c4/c8; helper rows only | 218.818/261.099/299.181 aggregate tok/s; public scheduler/server integration remains open |
| M2 | D2 fusion | exact; not promoted (kernel-efficiency regression) | dual+swiglu + qknorm+attention opt-in; down+weighted reverted (see `WORKLOG.md`) |

Rules (per `AGENTS.md`):

- Every retained perf row: model + quant + workload shape + hardware + exact
  command + result + correctness gate. No single-prompt overfit; validate on
  the multi-prompt suite.
- Each fused composite keeps an unfused fallback.
- New kernels pass the CPU-reference gate + `rocprofv3 --kernel-trace`.
- Profile with prebuilt `.so` + `require_cached`; never let the profiled
  process spawn `hipcc`.
- Log measurements and decisions in `WORKLOG.md` as they happen; update
  `benchmarks/README.md` / `CHANGELOG.md` for every retained result, including
  accepted diagnostics.

## 6. Non-goals / deferred

- FlashHead as the default (approximate; opt-in only).
- MTP speculative decode (no drafter for Maple yet).
- CUDA-peer backend, tensor parallelism, FP16-dequant prefill — separate tracks.
- Do not land throwaway prefill paths; build the complete bulk path directly
  per `docs/PREFILL.md` policy.
