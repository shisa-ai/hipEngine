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

**Conclusion.** Decode is **launch/dispatch-bound**, not compute- or
bandwidth-bound. Prefill is **serial**, paying one full forward per prompt
token with no weight reuse. Both have large, measurable headroom. Every
optimization below must pass the repo's correctness gate (KL ≤ 0.05, top-1
≥ 90%, device argmax exact) and be measured on gfx1151.

## 2. Phase 0 — Profile before optimizing (gate: no math changes)

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

## 3. Decode plan (launch-bound → graph-captured)

### D1 — hipGraph-capture the whole token step (highest leverage)

The repo already proved this pattern in `hipengine/runtime/moe_graph.py`: a
**stateless** per-layer unit is safe to capture and replay across arbitrarily
many relaunches (bit-exact in `tests/test_hip_graph_capture_replay.py`). Maple
decode is stateless per token: every layer recomputes from fresh
`hidden`/`qkv` buffers through fixed session-resident scratch pointers. The only
stateful memory is the KV cache, but the KV-write/attention kernels read and
write by **absolute `token_positions`** with `KVLiveSpans`, so a graph captured
against fixed buffer pointers is valid across tokens (same shape each step).

Steps:
1. Capture the 11-kernel per-layer body into a `MoeGraphCache`-style
   `MapleLayerGraphCache` keyed by `(layer_id, hidden_in_ptr, hidden_out_ptr,
   position)`. Validate bit-exact on first capture, replay thereafter.
2. Capture the top-level tail (embed, final norm, affine4 lm_head, argmax) as
   one graph too.
3. Because attention reads a growing cache, verify the captured attention graph
   is correct across positions (live-count metadata is a host-side arg that can
   be updated between replays without recapture if the ABI allows; otherwise
   capture per position-bucket).
4. Replay only — Python runs once per token for sampling, not 271 times.

Expected: eliminate the ~48 µs/launch host cost → several × decode speedup,
moving decode from ~77 tok/s toward the memory floor (hundreds of tok/s) and
then toward the compute/bandwidth balance for MoE.

Guard: if capture changes arithmetic, keep eager as the reference and only
retain the graph path that is bit-exact.

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

### D3 — MoE routing and grouped selected-expert

- Router is a single block doing serial work over 256 experts. Parallelize the
  logits (GEMV over [256, 2048]) and the stable top-8 (block-reduce), matching
  the existing one-ULP FP32 gate.
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
| M0 | Decode profile (host-gap + per-kernel) | profile artifact, no math change | `...-maple-decode-profile.json` |
| M1 | D1 graph-captured c1 decode | bit-exact vs eager; decode wall ↓ | `WORKLOG.md` + artifact |
| M2 | D2 fusion (launch count ↓) | exact, non-regressive | `WORKLOG.md` |
| M3 | D3/D4 MoE + lm_head | exact, decode wall ↓ | `WORKLOG.md` |
| M4 | P1 batched ternary prefill | exact vs packed oracle; prefill tok/s ↑ | `...-maple-prefill.json` |
| M5 | P2/P3/P4 full bulk prefill | exact; retained prefill row | `benchmarks/README.md` |
| M6 | D5 batch decode / server | exact c2/c4/c8; retained rows | `benchmarks/README.md` |

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
