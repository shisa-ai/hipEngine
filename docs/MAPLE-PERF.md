# MAPLE — gfx1151 Prefill & Decode Performance Plan

Last updated: 2026-08-05 (branch `maple`)

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
| Resident repeat wall (18-prefill + 36-decode) | 0.703 s | smoke artifact |
| Inferred decode | ~76.8 tok/s (13.0 ms/token) | smoke artifact |
| HIP kernels per decode token | ~271 (24 × 11 + top-level) | `runtime/maple.py` |
| Active weight traffic per token | ~9.5 MB | ternary 0.25 B/elem |
| Bandwidth floor for decode | ~37 µs/token | 9.5 MB @ 256 GB/s |
| Serial prefill of 4K prompt | ~52 s (token-serial) | 18 tokens ≈ 234 ms → 4K ≈ 52 s |

**Conclusion (corrected by M0 profile).** Decode is **kernel-time-bound**, not
launch-bound: a `rocprofv3 --kernel-trace` of one warm decode step shows
**12,209 µs kernel / 12,758 µs wall (95.7% kernel, only 4.3% host gap)**. The
serial 256-expert router dominates at **7,807 µs (61%)**; the affine4 lm_head
is **1,234 µs**, selected-expert gate/up + down **~1,571 µs**, attention
**562 µs**, QKV **434 µs**, o_proj **296 µs**. The host gap (~549 µs) is small, so
hipGraph capture is a minor win; the real wins are kernel-level, headed by the
router. Prefill is **serial**, paying one full forward per prompt token with no
weight reuse. Every optimization must pass the repo's correctness gate (KL ≤
0.05, top-1 ≥ 90%, device argmax exact) and be measured on gfx1151.

## 2. Phase 0 — Profile before optimizing (gate: no math changes) — DONE

`scripts/maple_profile.py` (prebuild then cached-only `rocprofv3 --kernel-trace`
of a warm decode step) produced `benchmarks/results/2026-08-07-gfx1151-maple-decode-profile.json`.

### Post-M3a profile (current default path, 2026-08-07)

Re-profiled after the parallel router (M3a) so the roadmap reflects the current
kernel-time split. One token = 9707, 4 warmup + 32 measured steps.

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
per-c1 GEMV shaping. hipGraph (M1, opt-in) recovers only the ~9% host gap.

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
(see D3/M3). lm_head and selected-expert GEMVs follow. hipGraph (D1) only
recovers the ~4.3% host gap and is deprioritized below kernel work.

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

## 2b. Reprioritized milestone order (from M0 evidence)

Follow `docs/TUNING-gfx1151.md` Lesson 0: capture a `rocprofv3 --kernel-trace`
of one warm decode token and of a short serial prefill before changing anything.
Produce a kernel-time vs host-gap breakdown:

- Per-kernel `DurationNs` by family (ternary GEMV, selected-expert, router,
  attention, qknorm, affine4 embed, affine4 lm_head, argmax, norms).
- **Host gap** = step wall − Σ kernel time. This is the launch/dispatch budget
  that hipGraph capture attacks (see D1).
- Which single kernel dominates (expected: affine4 lm_head at 151,936 output
  rows, and the serial 256-expert router).

Deliverable: a `docs/KERNELS.md` Maple trace block and a
`benchmarks/results/2026-08-05-gfx1151-maple-decode-profile.json` artifact.
Prebuild the `.so` and use `require_cached` so the profiled process never spawns
`hipcc` (per `docs/KERNELS.md` JIT gotcha).

## 2b. Reprioritized milestone order (from M0 evidence)

| Order | Milestone | Expected win | Rationale (M0) |
| --- | --- | --- | --- |
| 1 | **M3a router topk** parallel | 7.8 ms → ~1.4 ms (done) | 61% of step; single-block serial expert loop |

**M3a status: DONE.** `maple_router_topk_parallel_bf16` (grid-over-experts
coalesced dot + parallel softmax/top-k) cuts the router from 277 → 48 µs/call
(5.75×) and the decode step from 12,758 → 6,132 µs (**2.08× decode**), with the
packed correctness gate passing (max_kl 0.0139, top-1 18/18) and router IDs
exact. Evidence: `benchmarks/results/2026-08-07-gfx1151-maple-router-parallel.json`.
Next: lm_head affine4 (M3b), then selected-expert (M3c).
| 2 | M3b lm_head affine4 | dead end | 9.7%→22.5% now; tiled bit-exact but 0.96× (weight-bandwidth bound) |
| 3 | M3c selected-expert grouping | dead end | gate/up+down 29%; grouped bit-exact but 0.69× (x is L2-cached) |
| 4 | M4/M5 batched prefill | serial → ≥1k tok/s | prefill is the other big axis |
| 5 | M1 hipGraph decode | ~0.55 ms | host gap is only 4.3% |
| 6 | M6 batch decode / server | width reuse | after c1 is fast |
| 7 | M2 fusion | moderate | secondary to router/lm_head |

## 3. Decode plan (kernel-bound → optimize the hot kernels)

### D1 — hipGraph-capture the whole token step (implemented, opt-in)

M0 shows the host gap is only ~4.3% (549 µs/step), so capture is a secondary win
versus kernel-level work. It remains valuable once kernels are fast and for
batch/prefill. Pattern and caveats below.

**M1 status: DONE (opt-in).** `MapleGraphCache` captures the whole stateless
decode step (24-layer loop + final norm + affine4 lm_head + argmax) as one graph
and self-validates bit-exact on first capture (argmax parity vs a fresh eager
reference, with the output cleared so a no-op graph is rejected). The capture is
bit-exact across 120 tokens over a growing KVLiveSpans cache
(`tests/test_maple_graph_capture.py`). Kept opt-in (`HIPENGINE_MAPLE_GRAPH=1`,)
because on c1 decode is kernel-bound: repeated interleaved benches ranged
0.99–1.10× (noise-dominated; only the ~4% host gap is recoverable), so the eager
path stays the default. The captured graph is the infrastructure M6 batch
decode reuses, where the per-token launch win compounds. Evidence:
`benchmarks/results/2026-08-07-gfx1151-maple-hipgraph-c1.json`.

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
- **Router → topk → softmax → renormalize**: currently one 256-thread block
  does the full serial softmax/argmax over all 256 experts; make it parallel
  and fuse the gate/up dispatch.
- **QK-norm+RoPE+KV-write → attention**: already one fused qknorm write; fold
  the online-softmax attention into the same kernel or a tight two-kernel
  chain to keep Q/K/V in flight.
- **Expert down-projection + weighted residual + layer output**: fold
  `maple_selected_ternary_gemv` (down) + `clamped_swiglu` scaling + weighted
  residual into one selected-expert consumer.

Each fused composite must keep a numerically-equivalent unfused fallback
(architecture invariant).

### D3 — MoE routing and grouped selected-expert (highest leverage — M0)

- **Router is the #1 bottleneck (7.8 ms/step, 61%).** It is a single 256-thread
  block (grid 256 / wg 256) serially computing logits + softmax + top-8 over all
  256 experts (~362 µs/call × 24 layers). Parallelize the logits as a
  [256, 2048] GEMV (grid over experts, vectorized 2048-wide dot), then a
  parallel stable top-8 (block-reduce), matching the existing one-ULP FP32 gate.
  Target < 1 ms/step.
- Selected-expert kernels already operate without expert unpacking; verify grid
  = top-8 blocks is not leaving the 40-CU machine under-occupied, and consider
  grouping the 8 selected experts into one launch that reuses the shared
  activation buffer.
- Prefer true **grouped MoE** (scatter routed rows by expert, run grouped bulk
  kernels) for batch decode (D5) and prefill (P3) over repeated c1 selected
  GEMV, per `docs/PLAN.md` grouped-MoE direction.

### D4 — lm_head fast path

`maple_affine4_gemv_f32` spans 151,936 output rows (affine4, 2048 in) plus a
two-stage FP32 argmax — expected to be a large fraction of per-token wall.
Options, in order:
1. Profile it first (Phase 0) — if it dominates, target it.
2. Add the **FlashHead** approximate head (`lm_head_flash.*` present in the
   checkpoint) as an opt-in fast path, gated behind an exactness decision
   (approximate, so not the default).
3. Optimize the affine4 lm_head GEMV shape (threads/rows per block, LDS reuse
   of the 2048-wide hidden input) and the argmax reduction.

### D5 — Batch decode (c>1) and continuous batching

After c1 graph capture is solid, add multi-request decode so weight traffic is
reused across requests:
- Multi-column / MMQ-style ternary decode kernels for c=2/4/8 (row-GEMV today;
  reuse weights streamed once per group, per `docs/PLAN.md`).
- Batched GQA attention decode and KV append with a batch grid dimension.
- Then a continuous-batching owner loop (admission, chunked prefill, decode
  steps, sampler routing, reclaim) reusing the GGUF/PARO server patterns.

## 4. Prefill plan (serial → batched, compute-bound)

The serial `prefill()` must become a true `[T, hidden]` bulk path. Follow
`docs/PREFILL.md` and `docs/GGUF-PREFILL-OPTIMIZATION.md` shape conventions:
`T` = prompt rows, layer I/O = `[T, hidden]`, KV spans use
append-position = `start + r` / context = `start + r + 1`.

### P1 — Batched ternary GEMM (the core win)

Today the ternary kernels are c1 GEMV (`grid = out_features`, one row per
block). Add tiled/GEMM variants that **reuse weights across the T prompt rows**
and exploit the 2-bit packing:

- Embedding + all projections (`qkv`, `o_proj`, expert `gate/up/down`) become
  `[T, out]` = `x[T,in] · W[in,out]`.
- Use RDNA3.5 **INT4-WMMA** (`118.8 TOP/s`) to do the 2-bit ternary matmul as
  packed codes, or BF16-WMMA on dequantized tiles. Validate against the
  bit-exact CPU/NumPy oracle and the packed-formula gate.
- Chunk `T` into 256-row chunks per `docs/TUNING-gfx1151.md` Lesson 0 (the
  original gfx1151 win was shape/chunking; 4K → 1K+ prefill tok/s for PARO).

### P2 — Batched attention prefill (append-then-attend)

- Batched Q/K/V projection, batched head RMSNorm+RoPE, batched KV append.
- SWA layers: the 512-token window means prompt rows beyond 512 only attend to
  the recent window — exploit block/band sparsity rather than computing a full
  causal `T×T` mask.
- Global layers: causal attention over `[T, T]` (or prefix + chunk).

### P3 — Grouped MoE over prompt rows

- Router over all rows → scatter routed rows by expert → grouped bulk
  gate/up/down over the selected rows per expert. Avoids a per-row top-8 loop.
- Reuse the grouped-MoE machinery and `KVLiveSpans` ABI conventions.

### P4 — Chunked prefill + admission

- Wire the bulk prefill into the generator as `prefill_native(...)` (per
  `docs/PREFILL.md`), then into the continuous-batching loop (D5). Serial
  remains an oracle/fallback for correctness comparisons.

### Prefill target

Compute floor for active MoE ≈ 1.6B active params/token. At 59.4 TFLOP/s, a 4K
prompt (≈13 TFLOP) is ~0.22 s theoretical; with 256-row chunking and MoE
gather overhead, a realistic near-term target is **≥ 1,000–2,000 prefill
tok/s** (vs ~77 tok/s serial), matching the PARO/Laguna gfx1151 trajectory.

## 5. Milestones and evidence policy

| # | Milestone | Gate | Evidence |
| --- | --- | --- | --- |
| M0 | Decode profile (host-gap + per-kernel) | profile artifact, no math change | `...-maple-decode-profile.json` (done) |
| M3a | Parallelize router topk | exact one-ULP; decode wall ↓ 61% | `WORKLOG.md` + artifact |
| M3b | lm_head affine4 | exact; dead-end (tiled 0.96×, weight-bandwidth bound) | `WORKLOG.md` |
| M3c | selected-expert grouping | exact; dead-end (grouped 0.69×, x is L2-cached) | `WORKLOG.md` |
| M4 | P1 batched ternary prefill | exact vs packed oracle; prefill tok/s ↑ | `...-maple-prefill.json` |
| M5 | P2/P3/P4 full bulk prefill | exact; retained prefill row | `benchmarks/README.md` |
| M1 | D1 graph-captured c1 decode | bit-exact vs eager; decode wall ↓ | `WORKLOG.md` + artifact (opt-in; c1 is kernel-bound, ~1.0× within noise) |
| M6 | D5 batch decode / server | exact c2/c4/c8; retained rows | `benchmarks/README.md` |
| M2 | D2 fusion | exact, non-regressive | `WORKLOG.md` |

Rules (per `AGENTS.md`):

- Every retained perf row: model + quant + workload shape + hardware + exact
  command + result + correctness gate. No single-prompt overfit; validate on
  the multi-prompt suite.
- Each fused composite keeps an unfused fallback.
- New kernels pass the CPU-reference gate + `rocprofv3 --kernel-trace`.
- Profile with prebuilt `.so` + `require_cached`; never let the profiled
  process spawn `hipcc`.
- Log measurements and decisions in `WORKLOG.md` as they happen; update
  `benchmarks/README.md` / `CHANGELOG.md` only for retained rows.

## 6. Non-goals / deferred

- FlashHead as the default (approximate; opt-in only).
- MTP speculative decode (no drafter for Maple yet).
- CUDA-peer backend, tensor parallelism, FP16-dequant prefill — separate tracks.
- Do not land throwaway prefill paths; build the complete bulk path directly
  per `docs/PREFILL.md` policy.
