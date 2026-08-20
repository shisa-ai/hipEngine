# gfx1151 Non-Overlapping Tuning Landscape

Last updated: 2026-08-20
Host: HP ZBook Ultra G1a / Radeon 8060S / `gfx1151` (60 W power-limited lane)
Model: `Qwen/Qwen3.6-35B-A3B` GGUF Q4_K_M (MTP-bearing UD file) — c1 decode.

This doc records the **current gfx1151 performance-tuning surface split across
active agents** so new work lands in the open slots and does not collide with
concurrent ownership. It is a coordination + decision record, not a protocol
(see `TUNING-gfx1151.md` / `ROOFLINE-gfx1151.md` for the playbook and
`QWEN36-35B-ZBOOK-PRODUCTION-NUMERICS.md` for the active candidate plan).

## Ownership map

| Agent / owner | Scope | Covered stages (c1, ms/token GPU-exclusive*) |
| --- | --- | --- |
| **Agent 1 — recurrent state** | GDN / linear-attention state: state cache, SSM output, decay projections | `gdn_attention_core` 5.19, `gdn_decay_projections` 3.85, `gdn_input_projections` 2.29, `gdn_output_projection` 2.14 ≈ **13.5 ms/token** |
| **Agent 2 — concurrency / KV** | KV cache layout, paged/continuous batching scaling (gfx1100 first, global effects) | scheduler / KV-pool axis; not in the c1 stage ranking |
| **OPEN — this lane** | MoE dispatch (selected + shared expert GEMV, router, combine), full-attention math, LM-head | see table below |

\* Fresh GPU-exclusive ranking from `scripts/pn3_stage_ranking_from_trace.py`
(2026-08-17, pre-PN5/PN6), ROCTX nested-exclusive GPU-visible wall.

## The open (non-overlapping) c1 surface

| Stage | ms/token* | Notes |
| --- | ---: | --- |
| `moe_router_combine` | **10.657** | largest open cost; includes router + group scatter/gather + weighted-sum combine + residual. Much is host-dispatch idle (see PN5/PN6) |
| `shared_expert_gate_up` | 3.054 | shared-expert GEMV |
| `shared_expert_down` | 2.936 | shared-expert GEMV |
| `selected_expert_down` | 2.895 | selected-expert GEMV (the per-expert W4 path) |
| `selected_expert_gate_up` | 2.678 | selected-expert GEMV |
| `full_attention_core` | 2.386 | attention math (not KV layout — agent 2 boundary) |
| `full_attention_qkv` | 0.937 | QKV projections |
| `full_attention_output` | 0.738 | attention output proj |
| `selected_expert_other` | 0.796 | scatter/gather/elementwise |
| `lm_head` | 0.121 | small |

MoE total (router/combine + selected + shared) ≈ **23.1 ms/token** of the c1
wall — the dominant non-recurrent, non-concurrency surface.

## The dominant mechanism: host dispatch overhead, not kernel math

PN5 + PN6 (2026-08-18, both retained on the default path) established that a
large fraction of the c1 decode wall is **per-call host-side library-build /
dispatch overhead**, not GPU kernel execution:

- **PN5** `router-lib-hoist`: hoisted the `qwen35_router` CDLL into a module
  cache → 30.78 → 29.83 ms/token (~3.2%). Exact (byte-identical tokens).
- **PN6** `gemv-lib-hoist`: hoisted the `q8_0_t16` / `q6_k_t16` / `dense-gemv`
  CDLLs → 29.42 → 26.83 ms/token (~9.6%); per-call host fell 43.2 → 23.2 µs.
  **"Removing per-call build_hip recovers ~96% of removed host CPU as wall"** —
  i.e. the wall is host-side, not GPU-bound, for that slice. Overturns the
  earlier PN4 "GPU-bound" reading for the GEMV slice.
- PN6 handoff: **"MoE active-expert per-call build sites (group_scatter /
  laguna_router / maple_moe) remain a separate follow-up unit."**

Confirmed current state (2026-08-20): `hipengine/kernels/hip_gfx1100/moe/`
`group_scatter.py` (14 launch fns), `laguna_router.py` (4), `maple_moe.py` (8) all
still do `library = library or build_*(load=True)` per call, and the
selected-expert MoE launch wrappers (`_launch_selected_raw_gguf_moe_*` in
`qwen35_gguf_runner.py`) still do per-call env-flag reads + allocation lookups.

**Measured outcome (2026-08-20): the remaining dispatch host is HIDDEN — the
c1 MoE slice is GPU-bound, so the host-dispatch lever is a measured no-win.**

- `pn6-t16-selected-dense-hoist` (2026-08-19): hoisting the `t16_selected`
  library (25 sites, host 48.7 → 23.9 µs/call) recovered **~0 wall (-88 µs/tok,
  noise)** for the dense `launch_gguf_linear` slice — GPU-bound.
- Selected-expert MoE dispatch fast-path A/B (counter-rotated, 35B-A3B c1,
  scripts/pn3_moeselect_dispatch_ab.py): memoizing the exact default-branch
  launches for the 3 selected wrappers (skipping env reads + allocation lookups
  + branch chain, 1.66 ms/step host) recovered **~0 wall (-120 µs/tok, -0.5%,
  noise)** — the 1.66 ms host is overlapped under GPU execution.
- `pn4-c1-no-win-close` (2026-08-17) already flagged host memoization as "not
  worth it" (model ~94% GPU-bound); PN6's win was specifically the *blocking*
  library build, which is now hoisted tree-wide.

The 3.48 ms selected-expert slice that remains is therefore GPU kernel math
(~1.8 ms) + unavoidable dispatch; the W4 GEMV is near the gfx1151 practical
bandwidth ceiling (LAQ1-B: 510-540 vs 650 GB/s, latency/occupancy-bound). See
PN3 closeout artifact `benchmarks/results/2026-08-20-gfx1151-qwen36-35b-pn3-moeselect-no-win.json`.

## Aotriton usage and attention economics (2026-08-20, gfx1151 35B-A3B)

**Aotriton is prefill-only.** The 35B-A3B has 40 layers = 30 `linear_attention`
(GDN recurrent, agent 1) + 10 `full_attention`. Decode attention is 100%
native HIP: the 10 full-attention layers run `qwen35_paged_attn_decode_int8_*`
(Q8-int8 keys + bf16 values, gqa splitk), the 30 GDN layers run the recurrent
core. Aotriton appears only in **batched prefill (rows >= 512)** of the 10
full-attention layers (v3 flash-attn via `aotriton_wrap`); below the
512-token threshold prefill uses the native sequential path.

**Measured prefill economics (this session, 8060S, 512-token prefill, 4.36 s):**

- aotriton full-attention (10 layers, serialized upper bound): ~1.06 s = 24% of
  prefill wall; native layers (GDN prefill + MoE GEMM) are 76% and dominate.
- Prior gfx1100 threshold sweep (2026-05-16 artifact): **native prefill attention
  is FASTER than aotriton below 512 tokens** (-3.5% .. -17% at 32-256), aotriton
  wins at >=512 (+6% .. +256%). The 512 threshold is that measured crossover.
  The gfx1151 crossover is unmeasured; if it sits above 512, native could serve
  more prefill at benchmark lengths. Payoff is bounded: even 2x on attention
  saves ~0.5 s of a 4.4 s prefill, and prefill is ~25% of an end-to-end
  512+512 run.

**Measured crossover on gfx1151 (2026-08-20, retained): NO aotriton crossover —
native wins at every prefill length 64-2048 (~2-5% faster on the serialized
full-attention slice, never slower).** AOTriton's tiled flash is tuned for
larger GPUs (gfx1100's 96 CU / 48 MiB MALL); on the 40-CU 8060S the native
`causal_gqa_gate_bf16` scan wins and drops the aotriton wrapper overhead (bf16
query conversion, head-major KV copy, stream bridge). Retained: gfx1151 routes
all full-attention prefill to native via `GGUF_AOTRITON_PREFILL = False`
(backend capability, env `HIPENGINE_GGUF_AOTRITON_PREFILL_ENABLE` override);
gfx1100 keeps the measured 512-crossover policy unchanged. Correctness-neutral
(KL 0.046 native-vs-aotriton vs 0.034 run-noise floor at 1024 tok, top-1
agree). Benchmark suites (56-214 tok) ran native before and after. Artifact:
`benchmarks/results/2026-08-20-gfx1151-qwen36-35b-aotriton-prefill-native-retained.json`.

So "beating aotriton" is settled for gfx1151 (native routed; bounded ~3-4%
whole-prefill at >=512, exact) and the remaining attention lever is the native
kernel itself (P3-FULLATTN, prefill scan + decode paged_attn_decode
~2.4 ms/tok), not aotriton-vs-native.

## Ranked non-overlapping candidates

1. **P3-FULLATTN — full-attention core/gate + QKV math tuning** (arithmetic /
   tiling, NOT KV layout — coordinate the agent-2 boundary before touching
   dispatch). Pre-PN6 fresh profile ranked the full-attention core at
   ~2.4 ms/token GPU; post-PN6 it is the largest remaining non-overlapping
   GPU-side slice not yet re-ranked. **Active.**
2. **P3-EXPGEMV — selected-expert W4 GEMV shape tuning** (thread/tiling/dequant
   for 40 CU + 32 MiB MALL). Kernel-side; gated on the do-not-repeat ledger
   (DP4A/Q8_1, row-compact GEMV, one-plane Q8_1 already rejected). The host
   dispatch above it is closed (no-win), leaving the kernel math as the only
   lever (~1.8 ms).
3. **P3-MOECOMBINE — MoE combine / residual kernel tuning** (GPU math visible
   once host-idle removed).
4. **P3-LMHEAD — LM-head/sample** — smallest (0.12 ms), only after the above.

## Do-not-repeat ledger (MoE family, already closed)

- selected-MoE DP4A/Q8_1 routes that failed operation-complete quality/wall;
- row-compact selected-MoE GEMV (large verifier regression);
- one-plane Q8_1 activation (operation-complete SiLU KL failure);
- forced all-width / c2 Q8T16 rowtiling (rejected);
- Q8T16 64-thread verifier pair launch (slower than 128 threads);
- selective unsafe math (7.67% slower at the actual leaf);
- c1 MoE graph (exact but ~0.84% complete-wall regression);
- prompt/token/candidate-specific routing (prohibited benchmark gaming).

## Venue caveat

The ZBook is 60 W power-limited. Its absolute rates must never be reused as a
different-power (120 W Radeon 8060S) old→new comparison. The tuning *mechanisms*
(kernels, dispatch, shapes) are the same gfx1151 surfaces and transfer; the
speed rows belong on the same-power lane they were measured on.

## 27B dense (QWEN38-27B, closed campaign)

At the practical roof: c1 decode 13.1 tok/s ≈ 83% of the ~16 tok/s bandwidth
roof (27B Q4_K_S ≈ 14 GB @ 221 GB/s); non-temporal loads measured flat e2e
(ROOFLINE 6.6, reverted); the amortization win (MTP B3 1.78x AR) is retained;
prefill is above both llama backends; G5 memory closed. No strong non-
overlapping lever without a specific reopen (e.g. long-context 32K+ attention —
coordinate with agent 2 on the KV boundary).

## Source

- Stage ranking: `benchmarks/results/2026-08-17-zbook-qwen36-pn3-laq1-declaration-red.json`
- PN5/PN6: `benchmarks/results/2026-08-18-zbook-qwen36-pn{5,6}-*-hoist.json`
- Plan: `docs/QWEN36-35B-ZBOOK-PRODUCTION-NUMERICS.md` (PN3/PN4/PN5 gates)
- Playbook/roofline: `docs/TUNING-gfx1151.md`, `docs/ROOFLINE-gfx1151.md`
