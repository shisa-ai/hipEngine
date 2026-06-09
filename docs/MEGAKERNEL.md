# MEGAKERNEL.md — the M16.3 fused-kernel program (lower `C_B` to ≤ 2)

Status: **design + measured groundwork** (2026-06-09). This doc is the working
spec for the M16.3 megakernel campaign. It supersedes the scattered M16.3 notes
in `docs/MTP.md` for *implementation* purposes; `docs/MTP.md` remains the
economics/`C_B` source of truth. Update both when the plan moves.

Companion evidence:
- `benchmarks/results/2026-06-09-hipengine-m16.3-launch-census-batched-b3.json`
- `benchmarks/results/2026-06-09-hipengine-m16.3-staged-rotate-recheck.json`
- `benchmarks/results/2026-06-09-hipengine-economics-rerun-mtp-dflash-35b-27b.json`
- `benchmarks/results/2026-06-09-hipengine-m16.3-b3-paro-ffn-megakernel-microbench.json`

**Progress (2026-06-09):** B0/B1 (GGUF) and **B3** (PARO fused FFN megakernel)
are built + validated; a pi-multiloop kernel-time optimize loop took the PARO
megakernel from the correctness-first ~1.38 ms to **0.163 ms** at the c=4 verify
shape (~7.8x), **4.1x past the *naive* unfused PARO chain** in a synchronized
per-call microbench, single-launch, Scratch=0, KL gate held.

**B4 measured + closed (2026-06-09) — negative result, campaign redirect.** The
fp16 megakernel was wired into `run_moe_c1_fp16` (gated
`HIPENGINE_PARO_FFN_MEGAKERNEL`, default off): exact_ar_match=true on-model
(fires in all 40 layers) but it **REGRESSES** verify-cycle `C_B` ~4.4→~5.3.

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
the 32-row shape better, not by collapsing launches (see §8.3).

---

## 1. Why this exists — the `C_B` wall

MTP/DFlash speculative decode only beats AR when a verify cycle costs fewer
AR-token-equivalents (`C_B`) than the tokens it emits. Current measured state
(W7900/gfx1100, 35B-A3B PARO MTP, B=3, batched, exact):

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
barrier.

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

**Recommendation: T1 (self-consistent + KL-gated).**

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

1. **Adopt T1 (self-consistent + KL) as the megakernel correctness policy?**
   This is the unlock — it drops bit-exact-vs-legacy. Needs sign-off because it
   re-baselines AR and shifts the exact token stream within the KL gate.
2. **GGUF-first or PARO-first?** GGUF-first de-risks the architecture without the
   butterfly and ships a standalone GGUF-decode win; PARO-first goes straight at
   the MTP economics but pays the rotation complexity up front.
3. **Scope expectation:** even a perfect selected-FFN megakernel removes ~114
   launches/pass — `C_B` improves but does not cross break-even alone. This is a
   multi-unit campaign (FFN → attention → rmsnorm/router), not a single kernel.
   Meanwhile the **27B-dense DFlash gate already ships 1.16× AR today**; weigh
   campaign investment against hardening that deployable path.
