# MEGAKERNEL.md — the M16.3 fused-kernel program (lower `C_B` to ≤ 2)

Status: **design + measured groundwork** (2026-06-09; sprint pointer refreshed
2026-06-12). This doc is the working spec for the M16.3 megakernel campaign. It
supersedes the scattered M16.3 notes in `docs/MTP.md` for *implementation*
purposes; `docs/MTP.md` remains the economics/`C_B` source of truth. Update both
when the plan moves.

Current baseline note (2026-06-12): the active break-even sprint is now locked
in `docs/MTP.md` at the retained exact 9-prompt D32 **B=1** row, **1.018x AR**
with **14.173 ms/cycle** (`12.426 ms` verify + `1.733 ms` proposal/update in the
suite rollup). The first `>1.0x` row is retained; this document is now a margin
and future higher-density-kernel reference, not the sprint scoreboard. A fresh
B=1 quicksort profile shows the verifier at `833` launches/pass,
`9.407 ms/pass` kernel, and `12.877 ms/pass` host marker; proposer-all is
`41.5` launches/cycle, `1.533 ms/cycle` kernel, and `1.790 ms/cycle` host
marker. The older 45 ms / `C_B=4.67` tables below are retained as M16.3
groundwork, not the current operating point. Any new reduced-DAG or megakernel
work must beat the retained B=1 exact suite row while preserving or improving
the B=3 density path.

Companion evidence:
- `benchmarks/results/2026-06-09-hipengine-m16.3-launch-census-batched-b3.json`
- `benchmarks/results/2026-06-09-hipengine-m16.3-staged-rotate-recheck.json`
- `benchmarks/results/2026-06-09-hipengine-economics-rerun-mtp-dflash-35b-27b.json`
- `benchmarks/results/2026-06-09-hipengine-m16.3-b3-paro-ffn-megakernel-microbench.json`
- `benchmarks/results/2026-06-11-hipengine-mtp-paro-ffn-megakernel-exact-blocked.json`
- `benchmarks/results/2026-06-12-hipengine-mtp-b1-budget-retained.json`
- `benchmarks/results/2026-06-12-hipengine-mtp-b1-current-verify-rocprof.json`
- `benchmarks/results/2026-06-12-hipengine-mtp-b1-current-proposer-all-rocprof.json`

**Progress (2026-06-09):** B0/B1 (GGUF) and **B3** (PARO fused FFN megakernel)
are built + validated; a pi-multiloop kernel-time optimize loop took the PARO
megakernel from the correctness-first ~1.38 ms to **0.163 ms** at the c=4 verify
shape (~7.8x), **4.1x past the *naive* unfused PARO chain** in a synchronized
per-call microbench, single-launch, Scratch=0, KL gate held.

**B4 measured + closed (2026-06-09; current-stack recheck 2026-06-11) —
negative result, campaign redirect.** The fp16 megakernel is wired into
`run_moe_c1_fp16` (gated `HIPENGINE_PARO_FFN_MEGAKERNEL`, default off). It
previously passed one on-model exact smoke but **REGRESSED** verify-cycle `C_B`
~4.4→~5.3. On the current P1/proposer default stack it is worse: the opt-in
fires (`tokens=4`, `rows=32`, 160 calls) but fails exact AR at token index 9
(`156973` vs `149315`) on the locked B=3 smoke, so it is not eligible for
promotion regardless of launch-count savings.

Ground truth (rocprof, one batched B=3 verify window): megakernel **216.9 us/call**
(32 blocks) vs the production selected FFN **81.6 us/call** (rotate1 4.8 +
gate_up dual 41.7 + silu+rotate+down staged 35.2 — two WIDE GPU-filling kernels)
— **2.66x slower on the GPU**, raising kernel time 83→115 ms/6-pass.

The microbench was fixed: it had been measuring **Python ctypes launch overhead**
(4 launches vs 1) against an **8x strawman** (naive non-staged chain), not GPU
time. With HIP-graph replay timing it now matches rocprof (fused ~210 us vs
production ~83 us) and shows the fused time is **flat ~210 us across c=1..8**
(occupancy/latency-bound; only 32 blocks on 48 CUs; gate_up uses 64/256 threads).

The occupancy redesign (split-K gate_up to use all 256 threads) **regressed** to
~520 us in both LDS-reduction and warp-shuffle forms — split-K scatters the
per-thread weight loads and loses the contiguous coalescing thread-owns-pack
relies on. The down GEMV needs the full intermediate ← full gate_up, so the only
intra-row parallelism is split-K (lost). **Filling the GPU requires parallelizing
each GEMV over rows × output-columns across many blocks — i.e. the production
two-kernel staged design.** Single-launch on-chip fusion is the wrong design at
the 32-row verify shape; the megakernel stays default OFF. Artifacts:
`benchmarks/results/2026-06-09-hipengine-m16.3-b4-paro-ffn-megakernel-cb.json`,
`...-b4-rocprof-megakernel-vs-production.json`,
`...-b4-megakernel-occupancy-redesign.json`. **C_B redirect:** the biggest
verify-cycle families are GDN linear attention (14.1 ms/pass), gate_up dual
(10.0), down (8.4), w4_dual (8.0) — lower `C_B` by making those wide kernels fill
the 32-row shape better, not by collapsing launches (see §9, grid-reduction).

---

## 1. Why this exists — the `C_B` wall

MTP/DFlash speculative decode only beats AR when a verify cycle costs fewer
AR-token-equivalents (`C_B`) than the tokens it emits. Current sprint state is
tracked in `docs/MTP.md`; the retained 2026-06-12 row is B=1 at `1.018x AR`,
`14.173 ms/cycle`, and `1.617` visible tokens/cycle. B=3 remains the higher
density target (`2.175` visible tokens/cycle) but is below break-even as a fixed
operating point. The table below is the historical M16.3 kickoff state
(W7900/gfx1100, 35B-A3B PARO MTP, B=3, batched, exact), retained to explain why
this document exists:

| metric | value |
|---|---|
| AR decode | 103.5 tok/s (9.66 ms/token) |
| verify cycle wall | 45.1 ms (`C_B` = **4.67** AR-tokens) |
| visible tokens/cycle | 2.38 (accept 0.46) |
| **MTP/AR** | **0.52×** (≈2× slower than AR) |
| `C_B` needed for break-even at current accept | **≤ 2.385** |

The verify cycle decomposes (task #29 / M16.1) into ~**18.5 ms kernel** +
~**19.4 ms host/dispatch residual**, the residual being **931 kernel launches/
pass × ~20 µs**. To reach `C_B ≤ 2` the verify window must fall 37.9 → ~17 ms:
roughly halve both kernel time and launch count.

**The lever is fewer, larger kernels (M16.3).** M16.1/M16.2 closed the
alternatives: HIP graph replay is neutral (1.00× at 941 nodes — the ~5.6 µs/node
floor is GPU command-processor dispatch graphs can't remove), and a native C
verify loop is parity (the per-launch cost is grid-size-bound GPU workgroup
scheduling, not host/Python/arg-marshaling). Only removing launches and
shrinking grids helps.

---

## 2. The launch census (the map) — 931 launches/pass

rocprof `--kernel-trace` of the batched B=3 verifier (the economics path),
decode-tokens=8, **931 launches/pass, 15.97 ms kernel/pass**. No single family
dominates; launches are spread ~1/layer (40 layers: 30 linear + 10 full attn)
across ~9 families.

| family | /pass | µs/pass | note |
|---|---:|---:|---|
| paro_rotate (1+2+3) | **145.7** | 803.7 | PARO input rotation before each W4 GEMV |
| gemv_awq_dual_pack8 (shared gate_up) | 76.8 | 1046 | biggest kernel-time |
| silu_mul_dual_rotate_out | 76.8 | 415 | **down-rotate already fused** (default-on) |
| router (logits+select) | 77.1 | 468 | 2 launches/layer |
| rmsnorm (norm+add_norm) | 77.6 | 275 | 2/layer |
| copyBuffer (D2D) | 55.1 | 155 | pure plumbing |
| gemv_paro_marlin_k_fma | 48.0 | 513 | |
| selected gate_up / down / combine GEMVs | ~38 ea | 756/467/116 | 1/layer each |
| GDN linear-attn conv/recurrent | ~29 ea | 89/407 | 30 linear layers |
| f32↔fp16 staging conv | 38.8 | 70 | dtype plumbing |

The old 120/pass `runtime_memset` is **gone** (`fillBufferAligned` 0.6/pass —
M7.C already eliminated it). Do not re-chase it.

---

## 3. What does NOT work — measured, do not re-litigate

**Op-pair *staging* fusion regresses `C_B`.** The existing bit-exact
staged-rotate kernels (HBM-staged, keyed-barrier — the "good" rotate-once design)
were re-measured on the current tree:

| config | `C_B` (B=3) | exact | launches removed |
|---|---:|---|---|
| baseline | **4.67** | ✓ | — |
| `SHARED_EXPERT_FUSED_ROTATE=1` | 5.13 | ✓ | ~68/pass |
| `+ SELECTED_MOE_STAGED_ROTATE=1` | 5.06 | ✓ | ~146/pass |

Removing **small-grid** launches (rotate/rmsnorm/router) saves only the ~5.6 µs
dispatch floor per launch, which is **less** than the barrier-spin + staged-HBM
round-trip a staging kernel adds. Consistent with M13.B.1 (+12.4 ms/pass
redundant LDS rotation) and M15.4 (occupancy trap).

**Consequence:** the first true megakernel must consolidate **real big-grid GEMV
work + HBM intermediate traffic**, not shuffle small-grid plumbing behind a
barrier. **§10 (#105) makes this decisive in isolation:** a persistent-barrier
microbench shows the grid barrier is ~free but persistent only beats N-launch for
*dispatch-bound (sub-cache)* stages; for HBM-bound stages (the big GEMVs) it ties
or loses (0.93–1.27×). The 3-5× persistent whole-pass is **not supported** — the
lever is glue fusion + dispatch-floor reduction, not a megakernel.

---

## 4. The target — the selected-expert FFN megakernel

Fuse the per-layer selected-expert pipeline into **one kernel**:

```
current (per layer, ~7 launches):
  paro_rotate1(hidden) → gemv_awq_selected_dual_pack8 (gate+up, W4)
  → silu_mul_dual_rotate_out (silu·, +down PARO-rotate)
  → gemv_awq_selected_pack8 (down, W4)
  → weighted_sum_shared_gate_combine_residual (routing-weighted sum + shared + residual)

target (per layer, 1 launch):
  one block per (token, expert):  rotate → gate_up GEMV → silu·mul → rotate
    → down GEMV → ×routing_weight → atomic/serial accumulate into moe_out
```

Each block carries the 512-d intermediate **on-chip** (registers/LDS), so the
gate_up-output HBM write + down-input HBM read **vanish**, ~3 big-grid GEMV
launches/layer collapse to 1, and the rotates fold in for free (already in the
block, no separate launch). Grid = `(tokens × top_k)` = 32 blocks at B=3.

Reach: ~114 launches/pass + the intermediate HBM round-trip. A step toward
`C_B ≤ 2`, not a one-shot fix — the GDN/full-attn blocks and rmsnorm are
separate later units.

**Why this is hard under the legacy constraint:** reproducing the existing
4-kernel chain bit-for-bit means matching AWQ pack8 dequant order, the PARO
butterfly (per-channel θ/scales), dual-GEMV accumulation order, silu in fp, and
the routing-weighted combine — exactly. That is the campaign's risk surface, and
the next section is how we cut it down.

---

## 5. Accuracy strategy — the biggest lever (relax the *legacy-match*, keep *self-consistency*)

The current verifier work chases **bit-exactness vs the legacy per-row chain** so
that `exact_ar_match` (spec tokens == same-session AR tokens) holds. But:

- `exact_ar_match` is just `spec_tokens == ar_tokens` (`mtp_chain_e2e_smoke.py`
  line 585) — a **self-consistency** check between the AR path and the verify
  path *in the same run*. It is **not** a model-quality bar.
- The project's actual kernel correctness gate is the **relaxed**
  KL ≤ 0.05 AND top-1 ≥ 90% vs `kernels/cpu_reference/` (`AGENTS.md`,
  `docs/TESTING.md`). Bit-exact-vs-legacy is a *self-imposed* verifier add-on.

Three accuracy tiers for the megakernel:

| tier | rule | kernel difficulty | exact_ar_match | model quality |
|---|---|---|---|---|
| **T0 bit-exact-legacy** (today) | fused == legacy 4-kernel chain, bit-for-bit | **very hard** (match slow scalar rounding/order) | preserved vs legacy AR | identical |
| **T1 self-consistent + KL** ⭐ | one **row-invariant** megakernel for **both** AR (rows=1) and verify (rows=B+1); gate KL≤0.05/top-1≥90% vs cpu_reference | **much easier** (pick the fastest row-deterministic kernel) | **preserved by construction** | within KL gate |
| **T2 fully relaxed** | verify need not equal AR; gate sequence-KL + acceptance within tolerance | easiest | dropped | within KL gate |

**Recommendation: T1 (self-consistent + KL-gated).** — **ADOPTED 2026-06-09**
(human lead sign-off) **for the verify path**; see §8.1 and §9.4. The verify GDN /
megakernel correctness gate is now KL ≤ 0.05 / top-1 ≥ 90% vs `cpu_reference`,
**not** bit-exact `exact_ar_match`.

Why T1 preserves `exact_ar_match` without bit-exact-vs-legacy: if the *same*
megakernel computes the FFN for both the AR rows=1 path and the verify rows=B+1
path, and the kernel is **row-invariant** (a row's logits are identical whether
processed alone or in a batch — trivially true for an FFN, which has no
cross-row reduction), then AR and verify produce identical per-position logits →
identical argmax → `spec_tokens == ar_tokens`. This holds *regardless* of whether
the megakernel matches the legacy chain.

What T1 buys (simpler **and** faster):
- Free to use WMMA/MFMA dual-GEMV for the rows=4 verifier shape instead of the
  scalar per-row path chosen to match AR's rows==1 numerics.
- Free to fp32-accumulate / reorder for speed; no matching legacy rounding.
- Free to fuse aggressively (rotate+GEMV+silu+GEMV+combine) without per-stage
  bit-reproduction — only the *fused* output must clear the KL gate.

T1 costs: the legacy per-row path's exact token stream shifts slightly (within
KL≤0.05), so any goldens/fixtures pinned to legacy outputs regenerate, and AR is
re-baselined through the new kernel (must be ≥ current AR tok/s).

**Hard requirement for T1: prove row-invariance.** RED test:
`megakernel(x_row alone) == megakernel(x_row inside a B+1 batch)` bit-for-bit
per row. An FFN that processes each (row, expert) with a fixed accumulation
order satisfies this; a cross-row-tiled WMMA layout might not — that constraint
shapes the kernel.

---

## 6. The GGUF simplification — develop on the easy substrate first

The MTP economics target is the **PARO** model (`w4_paro` quant + PARO rotation).
But we also ship a **GGUF** path (`Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` is present —
same architecture), and it is structurally simpler for a fused FFN:

| axis | PARO path | GGUF path |
|---|---|---|
| rotation | **PARO butterfly** (paro_rotate, 146/pass, hardest bit-exact stage) | **none** |
| quant | AWQ W4 pack8 (custom) | Q4_K / Q8_0 (llama.cpp-standard, reference exists) |
| MoE dispatch | per-expert selected GEMV + combine | grouped scatter/gather + WMMA tile map (already mmid-shaped) |

So a GGUF fused-FFN megakernel is **dequant Q4_K → gate_up GEMV → silu → down
GEMV → combine** — the *standard llama.cpp fused MoE* (`mul_mat_id` / mmvq), with
**no rotation to reproduce** and a known-good numeric reference. It also removes
the entire 146/pass paro_rotate family for free on that path.

Caveat: the base GGUF file has **no MTP head**, so it does not directly run the
MTP/DFlash economics (those need the PARO+MTP target or a matching DFlash
drafter). Two ways to use GGUF anyway:

1. **Prototype substrate (recommended).** Build + validate the fused-FFN
   *architecture* (T1 row-invariance, KL gate, on-chip intermediate, launch-count
   and kernel-time mechanics) on the **GGUF AR decode** path first, where it is
   simplest and independently valuable (faster GGUF decode is a real product
   win). Then port the proven structure to PARO, adding PARO rotation as the
   fused first stage.
2. **GGUF speculative later.** If a Q4_K-compatible drafter (or a GGUF MTP head)
   becomes available, the GGUF megakernel feeds GGUF DFlash directly.

This sequencing de-risks the hardest part (the fused-FFN control flow + T1
proof) on the substrate without the PARO butterfly, then treats PARO rotation as
an additive, separately-tested stage.

---

## 7. Build plan (RED-first, staged, each stage a commit)

Order chosen so the riskiest math is gated by a golden oracle before any
performance work, and so GGUF (no rotation) front-loads the architecture.

| # | stage | gate (must pass before commit) |
|---|---|---|
| **B0** | Golden oracle + fixtures: cpu_reference FFN (gate_up→silu→down→combine) for GGUF Q4_K and PARO W4, fixed inputs/weights | fixture committed; legacy chain reproduces it within KL≤0.05 |
| **B1** | GGUF fused FFN megakernel (one block per (token,expert), intermediate on-chip), rows∈{1,4} | KL≤0.05/top-1≥90% vs B0; **row-invariance RED** (rows=1 vs in-batch per-row identical); `rocprofv3 --kernel-trace` shows 1 launch/layer |
| **B2** | Wire GGUF AR decode + (if applicable) verify to B1; re-baseline AR | GGUF E2E KL≤0.05; AR tok/s ≥ prior; launches/layer down |
| **B3** ✅ | PARO fused FFN: B1 + fused PARO rotate as first in-block stage | **DONE** (kernel `9d2d31c`): KL≤0.05 vs PARO B0, row-invariance RED bit-exact, f32 1.8e-7, 1 launch/layer. Micro-opt loop: 4.1x past the unfused chain at c=4 (B5 crossover), ~7.8x off the correctness-first baseline. |
| **B4** ✗ closed | Wire PARO verifier to B3; measure; fix microbench; parallelize | **DONE, negative + closed.** Wired (gated, default off); exact_ar_match=True on-model, but `C_B` **regressed** ~4.4→~5.3. rocprof: megakernel 216.9 us/call vs production selected FFN 81.6 us (2.66x slower). Microbench fixed (HIP-graph GPU timing now matches rocprof; the old per-call loop measured Python launch overhead vs an 8x strawman). Occupancy redesign (split-K gate_up, LDS and warp-shuffle) **regressed** to ~520 us (lost coalescing + occupancy). **Conclusion:** single-launch on-chip fusion is the wrong design at 32 rows; the production two-kernel staged path already fills the GPU. Megakernel stays default OFF. |
| **B5+** | Next megakernels: GDN/full-attn block, rmsnorm fold, router fuse — only those that remove **big-grid** launches or real work | same gates |

Discipline: every stage keeps an unfused fallback registered (architectural
invariant), raw device pointers in kernel signatures, four-axis registry keys
(no `if quant ==` branches), and `KVLiveSpans` for any attention kernel. A stage
that regresses `C_B` is reverted with the measurement recorded (like §3), not
kept.

---

## 8. Open decisions (for the human lead)

1. ✅ **RESOLVED (2026-06-09): T1 adopted for the verify path** (human lead
   sign-off). The verify GDN / megakernel correctness gate is now KL ≤ 0.05 /
   top-1 ≥ 90% vs `cpu_reference` (the project gate), **not** bit-exact
   `exact_ar_match`. First application: the GDN dv-tiling win (§9.4) — KL-correct
   (out/leaf ~1e-6 vs cpu_reference) but flips `exact_ar_match` true→false via a
   ~1 ULP FP-reorder; landed under T1. **T1 cost CLOSED (2026-06-09):**
   real-prompt economics A/B (quicksort, B=3, 3 runs) shows acceptance
   byte-identical and `exact_ar_match=true` on the real prompt (the flip is
   degenerate-1-token-only), and **C_B unchanged within noise (4.81±0.14 →
   4.80±0.28)** — the kernel saving is below the dispatch-floored cycle's noise.
   See §9.4 and `docs/RELAXED.md` §0.1.
2. **GGUF-first or PARO-first?** GGUF-first de-risks the architecture without the
   butterfly and ships a standalone GGUF-decode win; PARO-first goes straight at
   the MTP economics but pays the rotation complexity up front.
3. **Scope expectation:** even a perfect selected-FFN megakernel removes ~114
   launches/pass — `C_B` improves but does not cross break-even alone. This is a
   multi-unit campaign (FFN → attention → rmsnorm/router), not a single kernel.
   Meanwhile the **27B-dense DFlash gate already ships 1.16× AR today**; weigh
   campaign investment against hardening that deployable path.

---

## 9. Grid-reduction — the measured dispatch model and the GDN over-launch

**Status: dv-tiling landed under T1 (2026-06-09).** Pivot after the B4 redirect: instead of
collapsing launches behind a barrier (regresses, §3) or one big megakernel
(occupancy trap, §4/B4), attack the two ways a launch costs `C_B` — *kernel
time* (M16.4) and *dispatch* (this section) — on the kernels that over-launch
tiny-work blocks.

### 9.1 The profiler blind spot (why our earlier dispatch numbers were wrong)

`rocprofv3 --kernel-trace` sums `DurationNs` per kernel — **GPU-active time
only**. ROOFLINE §5.3: *"the kernel-trace profile misses a significant
component: the time between kernel launches (dispatch overhead) is invisible in
per-kernel time accounting but real on the wall clock."* An earlier ad-hoc pass
summed `DurationNs` across the **whole** trace (prefill+AR+verify, not the
marker-windowed verify passes) and bolted on a fabricated per-launch model — both
wrong. The dispatch floor is not in the kernel CSV at all; it must be measured
by the **replay-delta** method (wall − kernel, ROOFLINE §5.3) or the dedicated
dispatch microbench, not inferred from `DurationNs`.

### 9.2 The dispatch model — re-measured on the current tree

`scripts/graph_node_microbench.py` (M16.1/M16.2), re-run W7900/gfx1100, current
tree (artifact `benchmarks/results/2026-06-09-hipengine-m16-dispatch-grid-sweep-retest.json`):

| grid blocks | direct µs/launch | graph µs/launch | graph speedup |
|---:|---:|---:|---:|
| 1 – 64 | 5.61 | 5.61 | 1.00× |
| 1024 | 7.25 | 6.32 | 1.15× |
| 2048 | 7.95 | 7.09 | 1.12× |
| 4096 | 9.36 | 8.50 | 1.10× |
| 8192 | 12.34 | 11.31 | 1.09× |

- **Per-launch dispatch cost scales with grid size** (GPU command-processor /
  workgroup scheduling), ~5.6 µs base + grid term. This is the residual a native
  loop and graphs cannot remove (re-confirms M16.2).
- **Graph-neutral in steady state** (1.00× at N≥200; ≤1.15× even at large grids
  — ROOFLINE §1.6: graphs amortize PM4/doorbell/MES, not MEC/SPI per-dispatch).
- **Arg-count independent**: 2→16 args adds 0.0 µs — not marshaling.

Verify-cycle split (B=3, batched): ≈ **13.6 ms kernel + ~19.4 ms dispatch floor**
(931 launches × ~20 µs). **C_B ≤ 2 is dispatch-floor-bound**: zeroing all kernel
time still leaves the ~19.4 ms floor. So a kernel-time win (M16.4) touches only
the 13.6 ms third; cutting the floor needs **fewer launches, smaller grids, or
multi-stream overlap**.

### 9.3 Grid-reduction targets (over-launch vs occupancy-needed)

Two verify kernels launch the largest grids; each block is then in the most
expensive dispatch class. Grids below are **total workgroups** from on-model
rocprof (`selected_*` use `dim3(out_packed, rows)`; the Grid_Y is the
token×selected-expert row count, not the expert count):

| kernel | grid (total WGs) | WG / VGPR | over-launch? |
|---|---:|---:|---|
| GDN chain recurrence | `(num_v_heads, head_v_dim)` = **4096** | 256 / 64 | **YES** — 1 dv column/block with massive redundant per-(v_head,t) recompute |
| selected gate_up dual GEMV (W4) | `(8192, 8)` = **65,536** | 64 / 104 | **NO** (slack) — each block = unique (out-pack, row) dot-product; no split-K, no redundant work |
| selected down GEMV (W4) | `(16384, 8)` = **131,072** | 64 / 104 | **NO** (slack) — same `dim3(out_packed, rows)` structure, unique work/block |

**Resolved (2026-06-09, #96 — launcher source `launch_selected_dual_pack8` /
`launch_selected_pack8` + on-model trace).** The GEMV grids are **genuine
output×row parallelism**: every block computes a unique `(output_pack, row)`
dot-product, the kernel loops the full `in_features` internally, and there is
**no split-K and no redundant recompute**. This is categorically different from
GDN, where 128 dv-blocks per v_head redid identical q/k loads, reductions, and 3
transcendentals/t (free to collapse). The GEMV blocks are *not* over-launch in
that sense.

**But they are not occupancy-bound either.** W7900 = **96 CUs** (48 WGPs; the
earlier "48 CUs" in this doc and ROOFLINE conflates WGP with CU), 32 max
waves/CU. At VGPR=104 the GEMV reaches ~half occupancy (~14 WGs/CU), so the
machine fills at **~1,350 WGs**. The verify grids run **49× (gate_up) to 97×
(down)** that depth — far past the ~4–8 waves needed for memory-latency hiding.
The excess parallelism is real **occupancy slack**.

**Consequence.** The lever for these GEMVs is **output-tiling** (each block
computes several output packs / rows, shrinking the grid into a cheaper dispatch
class and cutting x re-loads) — *not* a free collapse like GDN's dv-tiling.
Because output-tiling trades grid size for per-block serial work and must
respect the **B4 coalescing lesson**, it belongs with the gate_up/down
kernel-time work (§9.x / task #97), measured per-shape, not landed blind.
Grid-reduction is a free win only where the extra blocks carry *redundant* work
(GDN); here it is a kernel restructure with a real dispatch-vs-kernel tradeoff.

### 9.4 GDN chain recurrence — the dv-tiling lever (combined dispatch + kernel)

`qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_kernel` launches one block
per `(v_head, dv_idx)` = 4096 blocks. Per block, per t-step it recomputes work
that is **identical across all 128 dv-blocks of the same v_head**:

- q/k load (`conv_out[q_base+dk]`, `conv_out[k_base+dk]`) — depends only on
  `(v_head, t)`, not `dv_idx`.
- `q_sum`/`k_sum` block reductions → `q_scale`/`k_scale`.
- `beta` (sigmoid), `decay` (`expf(-expf(a_log)·softplus(...))`) — **3
  transcendentals/t recomputed 128×**.

Only `kv_mem`, `delta`, the state update, and the out accumulation are
dv-specific. Additionally the state write
`chain_recurrent_state[t·state_stride + state_head_base + dk·head_v_dim + dv_idx]`
is strided by `head_v_dim` per block (uncoalesced — the "8 MB strided state
write" flagged in WORKLOG), but **consecutive `dv_idx` are adjacent in memory**.

**Approach — dv-tiling.** Have each block process `VTILE` consecutive dv columns
(grid `(32, 128/VTILE)`; VTILE=4 → 1024 blocks, VTILE=8 → 512). This is the only
GDN lever that hits all three costs at once and stays compatible with the
runtime's state layout (the layout is unchanged; only the per-block write span
widens):

1. **Dispatch**: 4096 (9.36 µs) → 1024 (7.25 µs) blocks; ~−2 µs × GDN launches/pass.
2. **Redundant compute**: q/k load + scales + 3 transcendentals/t computed
   128× → 32× (VTILE=4). This is the WORKLOG "next GDN lever."
3. **State-write coalescing**: VTILE consecutive `dv_idx` writes per dk become a
   contiguous span instead of single strided floats — addresses the dominant
   remaining GDN cost without the invasive global-layout change WORKLOG warned off.

Discipline: RED-first against `scripts/gdn_chain_microbench.py` (out_max_abs /
state_max_abs oracle) before+after; correctness gate vs `cpu_reference`;
`rocprofv3 --kernel-trace` confirming the kernel name + duration; on-model
`exact_ar_match=true`; benchmarks rollup updated. State layout must stay
byte-identical (runtime reads it elsewhere — not the M-class invasive change).

**Already landed (M16.4, this campaign):** GDN chain recurrence warp-shuffle
reductions (BLOCK 64→32, `partial[]` LDS → `__shfl`): rocprof 78.5 → 72.0
µs/call (**−8.3%**), GDN family 14.14 → 12.96 ms/pass, total verifier kernel
13.84 → 13.61 ms/pass, `exact_ar_match=true`. That win is pure kernel-time (grid
unchanged at 4096); dv-tiling is the grid-reduction follow-on.

**LANDED — dv-tiling (VTILE=4, this campaign, under T1).** The chain recurrence
kernel is templated `<scalar_t, VTILE>`, each block owns 4 consecutive dv columns
(grid `(32, head_v_dim/4)` = **4096 → 1024 blocks**; VTILE=1 is the bit-identical
unfused fallback for non-divisible `head_v_dim`). Per-(v_head,t) q/k load, the
`q_sum`/`k_sum` reductions, and the `q_scale`/`k_scale`/`beta`/`decay`
transcendentals are computed once per tile; the 4 consecutive dv state writes
coalesce. Measured:
- microbench oracle vs `cpu_reference` (numpy f32): T=4 **86.99 → 67.76 µs**
  (−22%), T=8 155.02 → 118.42 µs (−24%); out_max_abs 1.07e-6, leaf 5.96e-8
  (**KL ≪ 0.05** — the T1 gate, met by 4+ orders of magnitude).
- on-model rocprof (`mtp_verifier_rocprof.py --backend hip_gfx1100 --chain-attn-mode
  batched --decode-tokens 8 --candidate-budget 3`, gate off): chain recurrence
  **72.0 → 53.39 µs/call (−25.8%)**, 2.16 → 1.602 ms/pass (−0.56 ms/pass); grid
  1024 blocks confirmed, VGPR=64, **Scratch=0 (no spill)**, kernel name confirmed.
- **Behavior under T1:** `exact_ar_match` flips true→false (~1 ULP FP-reorder
  from the restructured loops tips one verify token vs the *different* AR-path
  decode kernel, at the degenerate 1-token-prompt boundary). This is **not** a
  correctness regression under the project gate (KL/top-1 vs cpu_reference); it is
  exactly the T0→T1 trade §5 describes, and is accepted per §8.1.
- **T1 "owed" item CLOSED (real-prompt A/B).** Same-prompt economics A/B on the
  quicksort prompt (decode 32, B=3, **3 runs each**), strict shuffle vs relaxed
  dv-tiled: on the real prompt `exact_ar_match` stays **true** for all runs and
  the accept pattern is **byte-identical** (acceptance 0.4615, std 0) — the
  relaxation flips a token *only* on the degenerate 1-token smoke. **C_B is
  unchanged within noise: 4.81 ± 0.14 → 4.80 ± 0.28** (Δ ~30× below the std); the
  −0.56 ms/pass kernel saving is below the economics noise floor because the
  cycle is dispatch/host-bound (§9.2), not kernel-bound. dv-tiling is banked
  kernel-time headroom, not a standalone C_B mover. Full characterization:
  `docs/RELAXED.md` §0.1; artifact
  `benchmarks/results/2026-06-09-hipengine-m16-gdn-dvtiling-economics-cb.json`.

### 9.5 The unexploited lever — multi-stream overlap

ROOFLINE §1.6: the chip has **8 compute ACEs (pipes); single-HIP-stream
inference uses 1**, so 7/8 of the compute frontend is idle. Independent verify
kernels on separate streams could overlap dispatch+execution across pipes and
hide the floor. Limited by the layer-sequential dependency chain (layer N+1
needs N), but intra-layer independent work (e.g. across the B+1 tokens, or
attention vs MoE branches where they exist) is a candidate. Not yet scoped;
recorded here so it is not forgotten as the next floor lever after grid-reduction.

### 9.6 gate_up/down GEMV kernel-time assessment (#97) — no free win, deferred

The verify gate_up is `gemv_awq_selected_dual_pack8_strided_kernel<_,true>` at
grid **(8192, 32)** (rows = tokens×top_k = 4×8), WG=64, VGPR=104, **~45.6 µs/call**
(840 calls/smoke). The kernel is memory-bound W4 and already well-formed:
coalesced transposed weight loads, 8-way vectorized FMA, shuffle-reduce,
`__restrict__`, `__launch_bounds__(128,4)`.

**Thread-count A/B (the canonical "fill the 32-row shape" lever).** Thread count
is already per-shape-tuned (`threads = 64 if tokens > 1 else 128`, since May
`ca4796d8`). An env-gated 64→128 probe on the path where it applies (tokens=1
gate_up) was **neutral-to-worse: median 18.80 → 19.08 µs** (and 128 splits the grid
8192→16384). The tokens=4 verify gate_up already runs at the tuned WG=64. So
thread count is **not** a kernel-time lever here.

**Remaining lever = output-tiling** (§9.3: multiple out-packs/rows per block).
That shrinks the grid into a cheaper dispatch class and cuts x re-loads, but it
is a **dispatch** play with a per-block-serial-work tradeoff and B4 coalescing
risk — not a kernel-time win. **Deferred:** the §0.1/§9.4 economics A/B proved
verify kernel-time wins do not move C_B (dispatch-floored), so a speculative
output-tiling rewrite of an already-tuned, memory-bound GEMV is not warranted.
The C_B levers are §9.2 (dispatch floor) and §9.5 (multi-stream). Artifact:
`benchmarks/results/2026-06-09-hipengine-m16-gateup-threadcount-ab.json`.

---

## 10. Persistent-barrier microbench (#105) — the persistent whole-pass is NOT a 3-5× lever (measured)

Reviewer 2's prototype ORDER for the "persistent (3-5×)" track starts with step
1: a persistent-barrier microbench — *N logical stages with in-kernel global
barriers vs N HIP launches, same grid sizes as GDN/selected GEMV*. Built it
standalone (`scripts/persistent_barrier_microbench.{hip,py}`, cooperative
`cg::this_grid().sync()` launched via `hipLaunchCooperativeKernel`; the C wrapper
does the cooperative launch so no Python-side bindings are needed). gfx1100
W7900, occupancy ceiling = **384 resident blocks** (8/WGP × 48 WGP), `hipEvent`
timing, median of 12-20 reps. Artifact:
`benchmarks/results/2026-06-09-hipengine-persistent-barrier-microbench.json`.

**The grid barrier is nearly free.** Barrier-isolation (0.5 MB ×1000 stages,
cache-resident): persistent = **1.45 µs/stage**, i.e. `grid.sync()` ≈ **~1 µs** —
far below a dispatch boundary.

**But the win depends entirely on whether the stage is dispatch-bound or
HBM-bound** — set by the L3 (64 MB Infinity Cache) boundary:

| stage MB | regime | N-launch µs/st | persistent µs/st | speedup |
|---:|---|---:|---:|---:|
| 0.5–2 | sub-cache (dispatch-bound) | ~19.5 | 1.5–3 | **6–13×** |
| 16 | sub-cache | 19.6 | 13.3 | 1.48× |
| 64 | cache edge | 55 | 51 | 1.08× |
| 128–256 | **>L3 → HBM-bound** | 457–999 | 477–1076 | **0.93–0.96×** |

The first table re-reads one buffer (cache reuse) and **overstates** the win.
The **AR-faithful** test streams a *distinct* fresh HBM slice per stage (1 GB
buffer, 160 stages, no reuse — exactly weight-streaming decode):

| slice MB (≈ AR kernel working set) | N-launch µs/st | persistent µs/st | speedup | GB/s |
|---:|---:|---:|---:|---:|
| 3 | 13.9 | 10.9 | **1.27×** | 453→576 |
| 6 | 23.9 | 20.7 | **1.15×** | 526→608 |
| 12 | 44.0 | 40.7 | **1.08×** | 573→618 |

**Verdict (decisive, NO-GO on the persistent megakernel for 3-5×):** persistent
beats N-launch **only when the stage is dispatch-bound** (sub-cache, tiny working
set). For HBM-bandwidth-bound stages — which is what AR decode is: each
GEMV/expert streams *fresh* weights from HBM at ~600 GB/s effective — the only
recoverable slack is the constant ~3 µs/launch dispatch gap, worth **~1.08–1.27×**
at AR's 3–12 MB per-kernel working set, *not* 3-5×. The "25% → 70% BW util" premise
is refuted: the big-grid kernels already run near HBM-effective BW; they are not
25%-utilised, the *token wall* is (because of the dispatch gap between kernels).

**Consequence (redirects the program):**
- The 3-5× persistent whole-pass / FFN-megakernel (§4) is **not supported** — it
  would consolidate HBM-bound GEMV work that is already efficient, while paying
  barrier + lost-launch-overlap cost. Consistent with §3, M12.1/M13.D (graph
  replay ≈ direct dispatch), and #101 (verify is dispatch-bound, fix = dispatch).
- The real lever is the **dispatch-bound GLUE** (rotations/norms/router/casts,
  ~640/tok, tiny sub-cache working sets — the 6–13× column). **Fuse or eliminate
  the glue** (Phase 1) and lower the per-launch dispatch cost (M14.dispatch.1
  C-side dispatcher). Realistic AR ceiling ≈ **1.3–1.5×** (close the ~3.6 ms/tok
  gap toward the 7.06 ms busy floor), not 3-5×.
- A persistent kernel *would* help the glue (sub-cache), but so does plain fusion
  with far less ABI/runtime risk — and §3 already showed naive small-grid staging
  regresses `C_B`. So: **glue fusion + dispatch-floor reduction, not a
  persistent megakernel.**

---

## 11. The next attack (#107) — graph-replay exactness on the batched verify path

Status: **LANDED 2026-06-10/11 — graph replay is exact; verify wall −34% at B=3.**
The divergence was never graph dispatch; it was two capture-frozen host-state
channels, fixed in `qwen35_paro_runner.py` / `qwen35_paro.py`:

1. **Keyed staged-rotate barriers** (`SELECTED_MOE_DOWN_STAGED`, then
   default-on; opt-in after the 2026-06-11 exact-suite refresh):
   the host passes a cumulative `(count, epoch)` *by value*; replays reuse the
   capture-cycle epoch, the consumer never waits (silent race, 1.4–3.7 logit
   drift, run-to-run nondeterminism), and any direct pass after replays spins
   forever (the GPU hangs we hit while instrumenting). Fix: capture-safe
   memset-per-launch barrier mode whenever the verify graph path is active.
   The later graph-auto prompt-suite gate showed that memset/fill cost now
   outweighs the staged-down launch saving, so the staged path is retained only
   behind `HIPENGINE_SELECTED_MOE_DOWN_STAGED=1`.
2. **Scratch realloc churn:** `_canonicalize_decode_scratch` re-reserves the
   rows=B+1 workspace names at rows=1 every cycle, freeing buffers the graph
   holds raw pointers to. Fix: capture-time scratch snapshot in the graph
   entry (restored before commit) + keepalive while a graph is cached.

Measured (quicksort, decode 32, exact_ar=true at B=1/B=3):

| config | verify ms/cyc | cycle wall | C_B | MTP/AR |
|---|---:|---:|---:|---:|
| B=3 graph-off | 33.3 | 43.3 | 4.83 | 0.49× |
| **B=3 graph-auto fixed** | **22.1** | **32.2** | **3.57** | **0.67×** |
| B=1 graph-auto fixed | 17.4 | — | — | — |

Biggest single C_B move in the program. Break-even (C_B ≤ 2.38) still needs
~10 ms/cycle: the gap is now **proposer drafting + host loop (~10 ms)** and
verify busy. Next stacks: graph the MTP proposer steps, p_min=0.5 (#100),
gated k=2 tree (#99). Artifact:
`benchmarks/results/2026-06-10-hipengine-mtp-graph-replay-keyed-barrier-fix.json`.

Original plan (kept for record): every structural lever was measured-closed
(megakernel §4/B4, persistent §10, native loop M16.2, staged glue §3, kernel
time §9.4/§9.6). What remains open is the strongest single datum in the program:

**#101 measured batched `graph_mode=auto` verify at 20.38 ms vs 40.88 graph-off
(B=1, persistent_device, clean) — `C_B` 2.27, below break-even — rejected only
because `exact_ar_match` flipped false on the final token.**

Why this contradicts the M13.D / M16.1 "graph-neutral" verdicts, and why both
are right: M16.1 proved a graph is 1.00× vs a *native C loop* (GPU dispatch is
the floor), and M13.D measured graphs neutral-to-worse on the *old bucket-churn,
per-cycle re-capture* path. But the production batched path is now
**row-invariant with a stable bucket key** (#101: 931 launches at B=1 and B=5),
so one capture replays for the whole decode — and the thing replay removes is
the **Python+ctypes per-launch issue cost** (~20-30 µs × 931 ≈ 20-28 ms/pass),
which M14.dispatch.1 only removed for the c1 MoE path, not the batched verify.
The 20.38 ms replay wall ≈ busy (10.75) + graph-node floor (931 × 5.6 µs ≈ 5.2)
+ accept tail — i.e. the wall finally matches the M16.2 dispatch model.

Economics if exactness is fixed (projection, B=3 batched): busy ~14 ms + ~5 ms
node floor + proposer ~4 ms → cycle ~23 ms → `C_B` ≈ 2.1-2.3 vs visible 2.556
(#103) → **~1.1-1.2× AR**; stacking p_min=0.5 (#100, free) and the gated k=2
tree (#99, visible 2.82) → **~1.25-1.35× AR**. First MTP-positive lane.

The bug surface (one capture per `(rows, capture_width, base_slot,
chain_attn_mode, linear_attn_mode)` replayed every cycle): anything per-cycle
that is **host-baked by value at capture** (context/kv lengths, span counts,
positions, draft tokens passed as scalars instead of device buffers) replays
stale. "Diverges on final token" smells boundary-shaped — candidates: lm-head /
accept-payload reading a stale row count, last-cycle shorter draft, or a
position counter not advanced on-stream.

Plan (RED-first):
1. Reproduce: `mtp_chain_e2e_smoke.py --backend hip_gfx1100 --proposal-impl
   persistent_device --chain-attn-mode batched --graph-mode auto` B=1/B=3,
   decode 32; per-cycle accepted-token dump vs graph-off.
2. Localize: `graph_mode=validate` (compares replay vs direct per cycle) to find
   the first divergent cycle + row.
3. Audit `_launch_verify_chain_forward_accept` for by-value per-cycle scalars;
   fix = route through device buffers advanced on-stream (decode-graph piece C
   pattern already exists).
4. Gate: exact_ar 9-prompt suite + clean B=1/B=3/B=5 `C_B` artifact; retain only
   if exact AND `C_B` ≤ 2.6 at B=3.
