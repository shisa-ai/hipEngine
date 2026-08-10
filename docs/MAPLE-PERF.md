# MAPLE — gfx1151 Prefill & Decode Performance Plan

Last updated: 2026-08-09

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
| Public native prefill 128/320/512 | 750.854 / 741.890 / 754.458 tok/s | retained P4 recertification; <=0.224% from P2 |
| CUDA sm_120a native c1 prefill 128/320/512 | 1953.820 / 1852.124 / 1917.492 tok/s | RTX PRO 6000 GPU0; 5.412x / 9.042x / 13.419x matched serial |
| CUDA sm_120a natural+heldout exact c1 decode | 402.361 tok/s | exact split global + direct SWA; same-session wave32 397.214 tok/s, +1.30%, 1,152/1,152 wins |
| Current c1 fixed-token A/B candidate | 202.580 tok/s (4.936-ms process mean) | four-process alternating selector-snapshot gate |
| Current c1 trace companion | 199.293 tok/s (5.018 ms/token) | separate clean cached trace; 4.550-ms kernels |
| Current c1 natural-context | 153.201 tok/s | repeated complete category/state gate; -0.14% vs prior 153.409 |
| Qualified wave32-head A/B | 143.679 -> 153.409 tok/s (+6.77%) | retained D0 head qualification |
| Prior pre-head c1 short profile | 180.935 tok/s (5.527 ms/token) | post-router cached trace |
| Fixed-helper c8 decode64 | 428.063 aggregate tok/s median | retained D1 exact row reuse; excludes prompt/public scheduler |
| Public c1/c2/c4/c8 generation64 | 123.131 / 165.697 / 202.038 / 214.788 aggregate tok/s | retained P4, prompt admission + reclaim included |
| HIP kernels per c1 decode token | 271 | post-D0 cached trace |
| Exact affine4 lm-head payload | 166.922 MiB | packed weight + BF16 scale/bias |
| Public tracked residency (max context 512) | 4.988 GiB (5,355,881,852 bytes) | retained P4 recertification |
| Native-prefill qualification | performance: 128/320/512; exact state: 520/770 | public path continues to configured context |

**CUDA peer conclusion.** The complete grouped native c1 prefill path is
retained on `cuda_sm120a`: 18/18 natural+heldout states and 90/90 positions are
exact at KL 0, 520/770 physical state matches serial, and cache-only Nsight names
all batched families. The fixed matched-grid row is
**1953.820/1852.124/1917.492 tok/s** at 128/320/512. Exact wave32 direct decode
first improved local128 **341.012 -> 396.328 tok/s (+16.22%)**. The retained
split-K follow-up parallelizes only local128-exact QK score production in the six
growing global layers and preserves an ordered softmax/PV reducer; all 18 SWA
layers remain direct wave32. The source-clean complete suite improves
same-session wave32 **397.214 -> 402.361 tok/s (+1.30%, 1,152/1,152 wins)** at
natural context and **102.688 -> 106.893 (+4.10%, 1,152/1,152 wins)** at p512.
Both keep all 1,296 positions, 36 state pairs, every category, and lifecycle
exact. CUDA resident batching and serving remain separate from this gfx1151
punchlist. Evidence:
`benchmarks/results/2026-08-08-cuda-sm120a-maple-native-prefill-retained.json`,
`benchmarks/results/2026-08-09-cuda-sm120a-maple-wave32-decode-retained.json`,
and `benchmarks/results/2026-08-09-cuda-sm120a-maple-splitk-global-decode-retained.json`.

The cooperative GQA4 decode schedule between those two retained stages is
rejected and removed. Four warps per KV-head block preserve exact query-head
concurrency and load each K/V row once, but shared staging/barriers deliver only
**0.897x** wave32 at live512 and **0.896x** at 4096/8192. An eight-token shared
tile worsens to **0.863x/0.861x/0.860x**, so no product-suite rerun was
warranted. Wave32 remains the direct/SWA owner; split-K owns qualifying global
layers. Evidence:
`benchmarks/results/2026-08-09-cuda-sm120a-maple-gqa4-decode-rejected.json`.

**Current gfx1151 conclusion (P0+P1+P2+P4, D0, and D1 retained; P3 rejected).**
Final-row-only sampling nearly doubles qualified native prefill, expert-major
MoE adds 3.68-5.83%, and exact GQA4 adds another 3.13-15.87%. The clean
retained-P0/P1/P2 traces remain immutable phase baselines. Exact dense tile 16
and 32 regress tile 8 on every paired sample. D0's exact one-dispatch router and
one-wave affine4 head both pass the complete clean category gate and are now
default. A final behavior-neutral selector snapshot improves fixed-token A/B
**200.279 -> 202.580 tok/s (+1.15%)**; D0 is closed. D1's exact affine4
row-reuse head is also retained and supersedes the corrected c8 helper baseline.
P4 now carries native prefill safely across SWA wrap and exposes fixed-slot
c1/c2/c4/c8 through the public submit/poll scheduler:

| Current phase | Wall / unit | Kernel / unit | Host gap | Launches / unit | Useful throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| public native prefill320, post-P2 | **439.479 ms/request** | **431.666 ms** | **7.813 ms (1.78%)** | **730** | **728.135 tok/s** |
| autoregressive c1 decode, post-D0 selector snapshot (trace process) | **5.018 ms/token** | **4.550 ms** | **0.468 ms (9.32%)** | **271** | **199.293 tok/s** |
| fixed-helper c8 decode, post-D1 | **19.296 ms/batch** | **17.305 ms** | **1.991 ms (10.32%)** | **293** | **414.602 aggregate tok/s** |
| public c8 generation64, post-P4 | end-to-end scheduler wall | included in wall | included in wall | c-aware | **214.788 aggregate tok/s (1.744x public c1)** |

After P2, the exact affine4 head owns only **0.31%** of prefill320 kernel time,
while grouped gate/up + down owns **59.08%**. Attention is down to **21.916 ms /
5.08%**; dense QKV+O is the next bounded owner at **124.770 ms / 28.90%**. The
post-D0 c1 trace attributes **21.28%** to the head, **24.97%** to selected
experts, and **24.04%** to router compute; the post-D1 c8 trace attributes
**21.58%/43.52%** to head/selected experts. The paths therefore have different
actions:

- **Prefill:** P0 samples only the final prompt row, P1 groups routed rows by
  expert, P2 shares each K/V stream across four GQA heads, and P4 safely carries
  the same arithmetic across SWA wrap. Current qualified 128/320/512 throughput
  is **750.854/741.890/754.458 tok/s**, within 0.224% of P2 and versus
  **726.421/679.632/650.745** at P1. Max-context-512 residency remains
  **4.988 GiB**.
- **c8:** D1 now streams each 166.922-MiB affine4 payload once across request
  rows while preserving every result. Qualified c2/c4/c8 reaches
  **250.481/346.365/428.063 aggregate tok/s**.
- **c1:** the exact one-dispatch router and one-wave affine4 head are
  retained/default. The head improves its group64 rollback **143.679 ->
  153.409 tok/s (+6.77%)** on the clean complete suite. A controlled 3+3-process
  review measures graph at only **1.0047x** (+0.47%, 0.033 ms), so graph stays
  opt-in. Snapshotting the two invariant default-off fusion selectors once per
  step removes 46 environment reads/token and improves a four-process
  alternating fixed-token gate **200.279 -> 202.580 tok/s (+1.15%)**, with all
  four candidate processes above 201 tok/s. A separate cached trace is
  **5.018-ms wall / 4.550-ms kernels / 0.468-ms host gap / 271 launches =
  199.293 tok/s**; therefore 200 is protocol/noise-sensitive, not a universal
  floor.

P1 now uses registered stable int32 count/prefix/scatter metadata and
expert-major ternary consumers with direct original-lane output. The qualified
row is byte-exact and improves P0 by **3.68%/4.67%/5.83%**. The final diagnostic
expert family reaches only **254.179 ms** from **276.150 ms (1.086x)**, missing
the frozen **<=97.708-ms (2.826x)** ceiling. Metadata is only 0.444 ms and exact
2-/4-lane schedules regress, so scalar ternary unpack/dot/reduction—not route
compaction—is the blocker.

Current exact priority:

1. Future P1b exact non-WMMA SIMD ternary consumers; do not repeat grouped-lane,
   dense token-tile, or direct native-BF16-WMMA schedules.
2. Optional ragged multi-request prefill compute and HTTP serving may build on
   P4's fixed slots, but neither is required for the retained public in-process
   path.

DeepGrove's published M4 table supports this split: **169 tok/s exact**, **218
tok/s with approximate FlashHead**, and **1075 tok/s prefill**. Its generation
path avoids discarded prefill heads and sorts >=64 routed assignments by
expert. The table omits workload/repeat/software details, so it is directional,
not a cross-device benchmark.

Evidence:
`benchmarks/results/2026-08-09-cuda-sm120a-maple-splitk-global-decode-retained.json`,
`benchmarks/results/2026-08-09-cuda-sm120a-maple-wave32-decode-retained.json`,
`benchmarks/results/2026-08-08-cuda-sm120a-maple-native-prefill-retained.json`,
`benchmarks/results/2026-08-08-gfx1151-maple-p4-long-prefill-public-batch-retained.json`,
`benchmarks/results/2026-08-08-gfx1151-maple-d1-batched-affine4-rowreuse-retained.json`,
`benchmarks/results/2026-08-08-gfx1151-maple-d0-c1-router-retained.json`,
`benchmarks/results/2026-08-08-gfx1151-maple-d0-affine4-wave32-retained.json`,
`benchmarks/results/2026-08-08-gfx1151-maple-d0-selector-snapshot-retained.json`,
`benchmarks/results/2026-08-08-gfx1151-maple-d0-wave32-decode-profile.json`,
`benchmarks/results/2026-08-08-gfx1151-maple-d0-decode-profile.json`,
`benchmarks/results/2026-08-08-gfx1151-maple-p3-dense-token-tile-rejected.json`,
`benchmarks/results/2026-08-08-gfx1151-maple-p2-gqa4-prefill-retained.json`,
`benchmarks/results/2026-08-08-gfx1151-maple-p2-phase-profile.json`,
`benchmarks/results/2026-08-07-gfx1151-maple-p1-phase-profile.json`,
`benchmarks/results/2026-08-07-gfx1151-maple-p1-expert-major-prefill-retained.json`,
`benchmarks/results/2026-08-07-gfx1151-maple-p0-final-row-prefill-retained.json`,
`benchmarks/results/2026-08-07-gfx1151-maple-p0-phase-profile.json`,
`benchmarks/results/2026-08-07-gfx1151-maple-corrected-phase-profile.json`, and
`benchmarks/results/2026-08-07-gfx1151-maple-c1-graph-review.json`.

## 2. Phase 0 — Profile before optimizing (gate: no math changes) — DONE

`scripts/maple_profile.py` (prebuild then cached-only `rocprofv3 --kernel-trace`
of a warm decode step) produced the historical M0 artifact and the current
`benchmarks/results/2026-08-08-gfx1151-maple-d0-decode-profile.json` refresh.

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
(see D3/M3). lm_head and selected-expert GEMVs follow. hipGraph (M1) could at
most recover this historical ~4.3% host gap and is deprioritized below kernel
work.

### M0 methodology (reuse)

Follow `docs/TUNING-gfx1151.md` Lesson 0: capture a `rocprofv3 --kernel-trace`
of one warm decode token and of a short serial prefill before changing anything.
Produce a kernel-time vs host-gap breakdown:

- Per-kernel `DurationNs` by family (ternary GEMV, selected-expert, router,
  attention, qknorm, affine4 embed, affine4 lm_head, argmax, norms).
- **Host gap** = step wall − Σ kernel time. This is the launch/dispatch budget
  that hipGraph capture attacks (see M1).
- Which single kernel dominates (confirmed: the serial 256-expert router).

Deliverable: a `docs/KERNELS.md` Maple trace block and a
`benchmarks/results/2026-08-07-gfx1151-maple-decode-profile.json` artifact.
Prebuild the `.so` and use `require_cached` so the profiled process never spawns
`hipcc` (per `docs/KERNELS.md` JIT gotcha).

## 2b. Historical milestone disposition

| Milestone | Result | Current interpretation |
| --- | --- | --- |
| **M3a parallel router** | Done; router 7,807 -> 1,104 us/step and decode 12,758 -> 6,038 us in the post-M3a trace | Retained as D0's exact two-dispatch rollback; the one-dispatch composite is now default. |
| **M3b c1 affine4 tile** | Rejected at 0.96x | Do not revive the same tile; use a materially different layout/bandwidth schedule. |
| **M3c c1 eight-route grouping** | Rejected at 0.69x | This tested activation reuse within one token, not expert-major reuse across prompt rows. |
| **M4/M5 batched prefill** | P2 retained at 741-754 tok/s | Exact through 512; dense ternary QKV/O is next while scalar expert compute blocks the original P1 ceiling. |
| **M1 c1 hipGraph** | Exact but only 1.0047x in the current review | Keep opt-in; do not use the historical host-gap estimate as a speed claim. |
| **M6+D1 batch decode** | c2/c4/c8 exact row-reuse helper at 250.481/346.365/428.063 aggregate tok/s | Retained helper; P4 now exposes it through public fixed-slot submit/poll generation at 165.697/202.038/214.788 tok/s including prompt admission/reclaim. |
| **M2 fusion** | Exact composites regressed kernel efficiency | Keep opt-in/fallbacks; no default promotion. |

M3a evidence:
`benchmarks/results/2026-08-07-gfx1151-maple-router-parallel.json`.

## 3. Decode plan (kernel-bound → optimize the hot kernels)

### M1 — hipGraph-capture the whole token step (implemented, opt-in)

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
- **Router logits → softmax/top-k/renormalize**: D0 completes this item with
  the retained exact last-threadgroup/atomic-counter composite; see D3. Do not
  attempt router-to-expert fusion without a new measured owner.
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

The historical M0 router was the 7.8-ms / 61% bottleneck; M3a replaced it with
parallel logits plus parallel stable top-k. D0 now makes the exact
last-threadgroup composite the retained default: it keeps the same logit tree
and FP32 softmax/stable top-8, uses one four-byte owned counter, and runs
finalization in the last expert block.

The clean two-resident qualification covers all 18 natural+heldout prompts,
two repetitions, 4 warmup and 32 measured continuation steps. Selector-unset
production improves its exact two-dispatch rollback **139.538 -> 145.321 tok/s
(+4.14%)**, saves **0.301 ms** at the paired median, and wins
**1,127/1,152** pairs. All **1,296/1,296** tokens/top logits, **36/36**
native-prefill and final state pairs, and **2,592/2,592** counter checks are
exact; lifecycle returns to zero.

Cached tracing cuts router calls **48 -> 24** and total token launches **295 ->
271**. Router compute remains **1.094 ms / 22.51%** of the pre-head kernel sum,
so deleted dispatch boundaries—not changed math—supply the wall win. That prior
short fixed-token diagnostic is **5.527 ms / 180.935 tok/s**, with **4.859 ms**
of kernels. D4 records the now-retained exact head; do not return to either
older router.

The rejected M3c experiment grouped the eight routes of one c1 token and lost
because the shared activation was already L2-cached. It says nothing about
prefill row reuse. True **grouped MoE** scatters hundreds or thousands of routed
rows by expert and runs grouped bulk kernels; retained P1 now does exactly that,
as required by `docs/PREFILL.md`.

### D4 — lm_head fast paths

`maple_affine4_gemv_f32` spans 151,936 output rows (affine4, 2048 in) plus a
two-stage FP32 argmax. The pre-head trace measured **1.278 ms / 26.30%** of c1
kernels; the final selector-snapshot trace measures wave32 at **0.968 ms /
21.28%**. Argmax remains only 0.008 ms/token.

The retained/default `group64_wave32_exact` schedule computes the same four
virtual 128-thread partial groups inside one wave and reconstructs the exact
reduction tree with shuffles; it neither tiles output rows nor merely caches the
hidden vector. The real head is bit-identical and improves **1.527 -> 1.020 ms
(1.496x, 48/48 wins)**. The clean two-resident 18-prompt qualification improves
the exact group64 rollback **143.679 -> 153.409 tok/s (+6.77%)**, saves **0.442
ms** at the paired median, and wins **1,146/1,152** pairs. All **1,296/1,296**
token/top-logit positions, **36/36** native-start/final states, **2,592/2,592**
counter checks, and teardown are exact. Cached tracing names the wave32 kernel
at local32/VGPR16/LDS0/scratch0 and **0.968 ms/step** in the final trace. The
retained selector snapshot changes no kernel or launch and improves fresh-
process fixed-token A/B **200.279 -> 202.580 tok/s (+1.15%)**, paired median
saving **0.076 ms**, with **3/4** wins. The separate trace process measures
**5.018-ms wall / 4.550-ms kernels / 0.468-ms host gap / 271 launches = 199.293
tok/s**. The final roadmap audit removes the temporary head environment seam;
wave32 is the sole production route, while group64 remains separately registered
and directly tested as the exact numerical fallback.

D1 completes the rows>1 exact work: `group64_batched_rowreuse_exact` loads each
packed weight row once across c2/c4/c8 while replaying the original per-request
FP32 tree. Clean helper medians improve **218.818/261.099/299.181 ->
250.481/346.365/428.063 aggregate tok/s**. Cached c8 tracing cuts the head
**10.490 -> 3.734 ms (2.809x)**. Head+argmax fusion remains secondary because
argmax is only 0.008 ms/token.

Separately, add **FlashHead** only as an opt-in approximate path. The official
configuration probes 512 of 4,748 clusters (16,384 candidate tokens plus forced
controls) and is the mechanism behind DeepGrove's 218 tok/s M4 headline. It
needs full category/heldout exact-head agreement and quality reporting; it is
not part of the exact 200+ target.

### D5 — Batch decode (c>1) and continuous batching

Multi-request decode is independent of the c1 graph. M6 provides batched GQA
attention/KV append and fixed-slot reclaim; D1 reuses affine4 head weights
across c2/c4/c8; P4 completes public fixed-slot admission, prompt prefill,
decode, sampling, and reclaim following the GGUF/PARO scheduler patterns.

**M6+D1 helper and P4 public in-process integration: DONE.** Batch decode keeps
separate request-local SWA/global rings, sparse active masks, offset-correct
reclaim, and c-specific real-checkpoint gates. P4's
`MapleResidentModelRunner` now shares one c1 checkpoint owner, prefills prompts
directly into fixed physical slots, uses D0 for one active row, and uses D1 for
c2/c4/c8 through the public `SubmitPollTextGenerator` scheduler.

The low-level D1 helper remains **250.481/346.365/428.063 aggregate tok/s** and
excludes prompt/public scheduling. The stricter public protocol includes prompt
admission, native prefill, exact decode/sampling, output collection, and reclaim:
c1/c2/c4/c8 is **123.131/165.697/202.038/214.788 aggregate tok/s**, making
c2/c4/c8 **1.346x/1.641x/1.744x** public c1. All 15 repeated 18-prompt
trajectory sets match serial, sparse and staggered reuse are exact, and all
owners close to zero. One row inside a physical-c8 owner retains **99.714%** of
public c1 throughput. HTTP serving and ragged multi-request prompt compute remain
separate work. Evidence:
`benchmarks/results/2026-08-08-gfx1151-maple-p4-long-prefill-public-batch-retained.json`.

## 4. Prefill plan (current 741-754 tok/s -> exact 1,000+)

The public path is a correct `[T, hidden]` implementation with 256-row chunks
and `KVLiveSpans` append-position `start + r`. P4 now carries it beyond SWA-512;
the remaining compute work is ordered by the corrected profile rather than by
the historical landing sequence.

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

### P1 — True expert-major compact ternary MoE — DONE, PARTIAL WIN

Registered stable int32 count/prefix/scatter now feeds Maple-specific grouped
gate/up/down consumers. They stage one expert/output weight row and restore
original row/route order directly before the unchanged exact weighted combine.
Every per-row BF16 projection, clamp/SwiGLU, and weighted-sum boundary matches;
the gather chain remains a rollback.

The qualified result is **726.421/679.632/650.745 tok/s**, up
**3.68%/4.67%/5.83%** over P0, with **18/18** state hashes, **90/90** tokens and
top-1, KL 0, and exact close. Fixed metadata adds **45,072 bytes**. The expert
family improves only **1.086x** versus the required 2.826x; exact wider-lane
schedules regress. P1 therefore closes with a measured scalar-compute blocker,
and future work must use a different exact non-WMMA SIMD ternary contraction.

### P2 — GQA/query-row prefill attention — DONE

The batched Q/K/V projection, head RMSNorm+RoPE, KV append, and causal ring
attention are correct through 512. The clean post-P1 profile measures attention
at **63.993 ms/request (13.55% of kernel time)** across 48 chunk/layer calls.
The prior local128 body assigns one block per `(query head,row)`, scans each KV
stream four times for the four query heads in its GQA group, and executes a full
workgroup reduction with barriers for every key.

The implemented GQA4 body maps all 128 virtual threads onto one wave32 and
materializes the original 64/32/16/8/4/2/1 LDS stages exactly. It also spells
out the local128 weighted-value rounding/FMA sequence, preserving every FP32
association and online-softmax/PV operation while loading each K/V row once for
four query heads. It consumes complete `KVLiveSpans`; local128 is the explicit
rollback.

The clean retained trace reaches **21.916 ms (2.920x)**, beating the
**<=31.996-ms (2.0x)** stretch target, at local32/VGPR64/dynamic-LDS512/scratch0
and unchanged launch count. Qualified 128/320/512 throughput is
**749.175/741.368/754.000 tok/s**, up **3.13%/9.08%/15.87%** over P1. The full
M5 gate passes **18/18** byte-state hashes, **90/90** positions, KL 0, exact
lifecycle, and unchanged **5,355,881,848-byte** residency. P4 subsequently adds
the required append/attend orchestration beyond 512 without changing P2's
kernel arithmetic.

### P3 — Dense ternary row-tile + native-WMMA sweep — DONE / REJECTED

Dense QKV/O kernels reuse one 512-byte packed weight row across a fixed
eight-row tile and consume **124.770 ms** at post-P2 prefill320. Explicit tile
16/32 candidates preserved each output's packed-word/lane accumulation and
128-thread LDS tree; the production-inner-width primitive was BF16-bit exact
for all three schedules.

A same-resident, counterbalanced 8-prompt natural+heldout screen measured tile
8/16/32 at **744.116/731.182/571.923 tok/s**. Tile 16 and 32 regressed
**1.738%/23.141%**, lost all **16/16** paired samples, and retained identical
next-token/top-logit pairs. The regression is consistent with their 8/16-KiB
dynamic-LDS footprints and longer block residency outweighing fewer 512-byte
weight-row reloads. The expensive full state gate was therefore not run, every
temporary selector/export was removed, and tile 8 remains the sole production
path. Do not repeat this schedule.

The required native-BF16-WMMA follow-up is also rejected mechanically. A direct
16x16x16 wave32 probe changes **106/256 FP32** K16 partials relative to Maple's
sequential 16-term accumulation and then changes **43/655,360 BF16** outputs at
the full 320x2048x2048 production shape. Cached tracing names
`maple_ternary_gemm_wmma_kernel` at local32/VGPR48/SGPR128/LDS0/scratch0 and
23.604 us for a one-block 16x2048x16 smoke; extracted ISA contains
`v_wmma_f32_16x16x16_bf16`. The probe therefore executed as intended but fails
the byte-state contract before a model/state or timing gate. All probe surfaces
were removed. A future exact SIMD consumer must preserve both the sequential
K16 partial and the outer 128-partial tree; direct or outer-compensated WMMA
cannot substitute a different internal association.

**Landed prefill primitives (through 2026-08-08, bit-exact vs CPU oracle):**

| Primitive | Commit | Purpose |
| --- | --- | --- |
| `maple_ternary_gemm_bf16` | `7c1624080` | `[rows,out]=[rows,in]x ternary W`; eight-row weight reuse |
| `maple_ternary_qkv_gemm_bf16` | `a4b9808d8` | full `[T, q+2kv]` projection |
| `maple_qknorm_rope_kv_write_batched_kernel` | `d7185caaa` | row-batched QK norm, RoPE, and ring write |
| `maple_attention_prefill_ring_kernel` | `ffe822eae` | causal attention through `KVLiveSpans`; local128 rollback |
| `maple_attention_prefill_ring_gqa4_wave32_kernel` | `9ccc60541` | exact four-query GQA K/V reuse; retained P2 default |
| `maple_router_topk_parallel_batched_kernel` | `3820527ed` | row-batched logits and stable softmax/top-k |
| `maple_selected_ternary_*_batched_kernel` | `2bdd79497`, `bb3ac7569` | correct row/route-gather MoE oracle/fallback, not grouped MoE |
| `maple_affine4_gemv_batched_kernel` | `8e7b400db` | all-row debug/batch head, not the public-prefill target |

RMSNorm/add-RMSNorm already accept rows, and
`maple_affine4_embed_batched_kernel` (`66c5a7a11`) provides batched embedding.

### P4 — Chunked admission and long prompts — DONE, RETAINED

`prefill_native(...)` now remains exact across SWA wrap. Global layers stay
chunk-batched; each sliding layer restores the pre-chunk logical ring, batches
the safe pre-wrap segment, and append/attends post-wrap rows serially. Retained
520/770 gates match serial at FP32 top-logit bits, final hidden/norm, every live
physical K/V byte, both complete span owners, and three continuations.

Public fixed-slot admission shares the c1 weight owner and writes request-local
native prefill directly into the batch K/V arena. The clean 128/320/512 protocol
is **750.854/741.890/754.458 tok/s**, only
**+0.224%/+0.070%/+0.061%** from P2 and exact at all 18 state hashes / 90
positions. The symmetric public c1/c2/c4/c8 gate reaches
**123.131/165.697/202.038/214.788 aggregate tok/s**, with sparse/staggered
reclaim, singleton-c8 preservation, and lifecycle exact. Prompt storage is
packed; prompt compute remains per request rather than a ragged multi-request
GEMM.

### Prefill target

P0+P1+P2+P4 deliver **741.890 tok/s** at 320. P2 exceeds its attention gate at
**21.916 ms / 2.920x**, but P1's expert family still reaches only **255.050 ms /
1.083x** versus its original **<=97.708-ms / 2.826x** ceiling. P3 proves that
larger scalar dense token tiles do not close the gap, while direct BF16 WMMA
fails the byte-exact reduction gate. Crossing 1,000 now requires a materially
different exact non-WMMA SIMD ternary consumer. DeepGrove's
1075 tok/s M4 row avoids discarded heads and groups routed rows, independently
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
| M5/P0+P1+P2 | final-row + expert-major + exact GQA4 native prefill | fixed performance gate through 512; scalar ternary ceiling recorded | 749.175/741.368/754.000 tok/s; 18/18 byte-exact states and 90/90 positions |
| P3 | dense ternary tile + native-BF16-WMMA sweep | tile 16/32 regress; WMMA fails byte exactness | 744.116/731.182/571.923 tok/s; WMMA 106/256 FP32 partial and 43/655,360 BF16 output mismatches; production unchanged |
| D0 router | one-dispatch exact c1 routing | complete category/state/counter gate; default retained | 139.538 -> 145.321 tok/s (+4.14%); 1,127/1,152 wins; pre-head short profile 180.935 tok/s / 271 launches |
| D0 head+host | one-wave exact affine4 plus once-per-step selector snapshot | complete category/state/counter gate; default retained | Head: 143.679 -> 153.409 tok/s (+6.77%); host A/B: 200.279 -> 202.580 tok/s (+1.15%); trace companion 199.293 tok/s / 271 launches |
| D1 head | exact c2/c4/c8 affine4 row reuse | full-logit bits, all widths, category/sparse/reclaim/lifecycle, and trace; default retained | 218.818/261.099/299.181 -> 250.481/346.365/428.063 aggregate tok/s; c8 head 10.490 -> 3.734 ms |
| M1 | graph-captured c1 decode | bit-exact vs eager; no default promotion | current review 1.0047x / 0.033 ms, opt-in |
| M6+D1 | D5 batch decode helper plus exact row-reuse head | exact c2/c4/c8 helper | 250.481/346.365/428.063 aggregate tok/s; P4 consumes it publicly |
| P4 | safe SWA-wrap prefill + public fixed-slot admission | 520/770 physical state; same-protocol public c1/c2/c4/c8; sparse/staggered/singleton/lifecycle | prefill 750.854/741.890/754.458 tok/s; public 123.131/165.697/202.038/214.788 tok/s |
| M2 | D2 fusion | exact; not promoted (kernel-efficiency regression) | dual+swiglu + qknorm+attention opt-in; down+weighted reverted (see `WORKLOG.md`) |

Rules (per `AGENTS.md`):

- Every retained perf row: model + quant + workload shape + hardware + exact
  command + result + correctness gate. No single-prompt overfit; validate on
  the multi-prompt suite.
- Each fused composite keeps an unfused fallback.
- New kernels pass the CPU-reference gate plus the backend profiler
  (`rocprofv3` for HIP, Nsight Systems for CUDA).
- Profile with prebuilt `.so` + cached-only metadata; never let the profiled
  process spawn `hipcc` or `nvcc`.
- Log measurements and decisions in the unit's immutable worklog entry as they
  happen; update `benchmarks/README.md` / `CHANGELOG.md` for every retained
  result, including accepted diagnostics.

## 6. Non-goals / deferred

- FlashHead as the default (approximate; opt-in only).
- MTP speculative decode (no drafter for Maple yet).
- CUDA sm_120a c1 generation, native c1 prefill, exact wave32 direct decode, and
  exact split-K global decode are retained separately; CUDA resident batching/
  serving, tensor parallelism, and FP16-dequant prefill remain separate tracks.
- Do not land throwaway prefill paths; build the complete bulk path directly
  per `docs/PREFILL.md` policy.
