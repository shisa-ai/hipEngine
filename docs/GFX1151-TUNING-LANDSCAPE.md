# gfx1151 Non-Overlapping Tuning Landscape

Last updated: 2026-08-25
Host: HP ZBook Ultra G1a / Radeon 8060S / `gfx1151` (60 W power-limited lane)
Model: `Qwen/Qwen3.6-35B-A3B` GGUF Q4_K_M (MTP-bearing UD file) — c1 decode, BF16 KV unless explicitly noted.

**Review verdict (2026-08-25): current-main AR ownership is refreshed; no new
non-overlapping implementation is admitted.** A clean detached `822a8b00f`
ZBook/gfx1151 run records exact p512/d128 AR at **30.003 tok/s** over five
repeats and 24 marked rocprof steps with positive durations. Kernel-time owners
are GDN input Q8 **4.200 ms/token**, selected-Q4 gate/up **2.435**, Q6 LM head
**2.268**, selected-down **1.774**, and GDN output Q8 **1.487**. The order matches
the independent high-power SH closeout; absolute rates are not compared across
those physical hosts.

That stable order is a closure result, not permission to repeat the old ladder.
The completed SH campaign already rejected exact GDN DPP/raw-layout completion,
selected-Q4 tile/thread/DP4A/raw/half-sequential routes, Q6 LM-head tile8,
shared-expert composite, and cross-queue branch overlap; Q5 selected-down tile8
is already retained. The fresh marker window contains **17.831 ms/token** of
traced-kernel interval union and **16.751 ms/token** without a traced kernel, but
that residual is not labeled recoverable: prior exact MoE graph replay removed
about 64% of FFN launches and still regressed wall **0.84%**. Reopen only for a
materially new exact representation/dataflow with either >=1.15x
operation-complete leaf speedup or >=0.5 ms/token projected saving, or after
SPECDEC2/MTP2 changes the actual route/kernel identity. Evidence:
[`current-main refresh`](../benchmarks/results/2026-08-25-zbook-gfx1151-qwen36-35b-ar-moe-profile-refresh.json).

This doc records the **current gfx1151 performance-tuning surface split across
active agents** so new work lands in the open slots and does not collide with
concurrent ownership. It is a coordination + decision record, not a protocol
(see `TUNING-gfx1151.md` / `ROOFLINE-gfx1151.md` for the playbook and
`QWEN36-35B-ZBOOK-PRODUCTION-NUMERICS.md` for the closed campaign gates).

## Ownership map

| Agent / owner | Scope | Covered stages (c1, ms/token GPU-exclusive*) |
| --- | --- | --- |
| **Agent 1 — recurrent state** | GDN / linear-attention state: state cache, SSM output, decay projections | `gdn_attention_core` 5.19, `gdn_decay_projections` 3.85, `gdn_input_projections` 2.29, `gdn_output_projection` 2.14 ≈ **13.5 ms/token** |
| **Agent 2 — concurrency / KV** | KV cache layout, paged/continuous batching scaling (gfx1100 first, global effects) | scheduler / KV-pool axis; not in the c1 stage ranking |
| **Profile-only — no candidate admitted** | AR-only MoE/GDN/LM-head ownership outside SPECDEC2/MTP2 | current clean rank above; prior mechanisms below are closed unless a materially new premise appears |

\* Durable pre-PN5/PN6 ranking from `scripts/pn3_stage_ranking_from_trace.py`
(2026-08-17), ROCTX nested-exclusive GPU-visible wall. It admits candidates but
must not be added directly to the 26.83-ms post-PN6 wall.

## Historical pre-PN5/PN6 c1 surface

These rows preserve the campaign's earlier stage-wall history; they are not the
current candidate selector. The positive-duration 2026-08-25 role ranking above
supersedes them for new work.

| Stage | ms/token* | Notes |
| --- | ---: | --- |
| `moe_router_combine` | **10.657** | historical stage wall; includes router + group scatter/gather + weighted-sum combine + residual. Much is host-dispatch idle (see PN5/PN6) |
| `shared_expert_gate_up` | 3.054 | shared-expert GEMV |
| `shared_expert_down` | 2.936 | shared-expert GEMV |
| `selected_expert_down` | 2.895 | selected-expert GEMV (the per-expert W4 path) |
| `selected_expert_gate_up` | 2.678 | selected-expert GEMV |
| `full_attention_core` | 2.386 | attention math (not KV layout — agent 2 boundary) |
| `full_attention_qkv` | 0.937 | QKV projections |
| `full_attention_output` | 0.738 | attention output proj |
| `selected_expert_other` | 0.796 | scatter/gather/elementwise |
| `lm_head` | 0.121 | small |

MoE total (router/combine + selected + shared) was ≈ **23.1 ms/token** of the
pre-PN6 GPU-visible c1 stage wall — the dominant non-recurrent,
non-concurrency surface at that checkpoint. This is not a current additive
kernel-time estimate.

## PN5/PN6 host-build wins; remaining dispatch is a no-win

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
native HIP. The authoritative 2.386-ms/token c1 ranking used **BF16 KV** and,
at contexts 512-548, the gfx1151-registered fixed256 compact-row context-batch
kernel (`qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans`)
followed by the separate BF16 gate-multiply kernel. Short mirrored/direct INT8
KV lanes exist elsewhere, but they are not the source of this ranking. The 30
GDN layers run the recurrent core. Aotriton appears only in batched prefill of
the 10 full-attention layers when its backend/threshold gate admits it; the
retained gfx1151 policy now disables that route at every row count.

**Measured prefill economics (this session, 8060S, 512-token prefill, 4.36 s):**

- aotriton full-attention (10 layers, serialized upper bound): ~1.06 s = 24% of
  prefill wall; native layers (GDN prefill + MoE GEMM) are 76% and dominate.
- Prior gfx1100 threshold sweep (2026-05-16 artifact): **native prefill attention
  is FASTER than aotriton below 512 tokens** (-3.5% .. -17% at 32-256), aotriton
  wins at >=512 (+6% .. +256%). The 512 threshold is that measured gfx1100
  crossover. The gfx1151 crossover was initially unknown; the completed local
  measurement follows. Payoff is bounded: even 2x on attention saves ~0.5 s of
  a 4.4 s prefill, and prefill is ~25% of an end-to-end 512+512 run.

**Measured crossover on gfx1151 (2026-08-20, retained): NO aotriton crossover —
native wins at every prefill length 64-2048 (~2-5% faster on the serialized
full-attention slice, never slower).** AOTriton's tiled flash is tuned for
larger GPUs (gfx1100's 96 CU / 96 MiB MALL); on the 40-CU 8060S the native
`causal_gqa_gate_bf16` scan wins and drops the aotriton wrapper overhead (bf16
query conversion, head-major KV copy, stream bridge). Retained: gfx1151 routes
all full-attention prefill to native via `GGUF_AOTRITON_PREFILL = False`
(backend capability, env `HIPENGINE_GGUF_AOTRITON_PREFILL_ENABLE` override);
gfx1100 keeps the measured 512-crossover policy unchanged. Correctness-neutral
(KL 0.046 native-vs-aotriton vs 0.034 run-noise floor at 1024 tok, top-1
agree). Benchmark suites (56-214 tok) ran native before and after. Artifact:
`benchmarks/results/2026-08-20-gfx1151-qwen36-35b-aotriton-prefill-native-retained.json`.

So "beating aotriton" is settled for gfx1151: native is routed, the isolated
attention slice is consistently 2-5% faster, and the measured whole-prefill
median moved -0.8% within ~2.5% run spread. The route passes the declared
correctness gate (top-1 agrees; KL 0.046 vs a 0.034 repeat floor), but it is not
an arithmetic-exact native-vs-AOTriton comparison. P3-FULLATTN is therefore
**decode-first**: the active BF16 context-batch attention core plus separate
gate own ~2.4 ms/token.
Native prefill is secondary and receives a new tiled specialization only if a
fresh long-prompt profile justifies that larger implementation.

## gfx1100 hard-coded numbers review + gfx1151 overrides (2026-08-20)

Systematic sweep for gfx1100-tuned cutoffs/geometry inherited by gfx1151:

- **gfx1151 already overrides 47/56 GGUF-path capability knobs** (GDN prefill
  auto-modes, parallel-reduce, rowtile policies, aotriton). 5 tables inherit
  gfx1100 defaults (`GGUF_DENSE_PREFILL_SCRATCH_LIVENESS_POLICIES`,
  `GGUF_Q4_T16_UNEQUAL_PAIR_PREFILL_POLICIES`, `GGUF_T16_F16_ROCBLAS_*`) --
  mostly 27B H5120 memory/prefill-variant policies, low value, unchanged.
- **Found + retained: the GGUF prefill chunk policy.** The gfx1151
  `_ARCH_CHUNK_PROFILES` (linear/moe 256) is PARO-only; the GGUF runner never
  passes `target_arch`, so GGUF prefill used generic 1024/4096 chunks. Sweep on
  the 8060S: **512-row linear/MoE chunks are optimal for the H2048-MoE 35B-A3B**
  on the 60W ZBook lane (~1.2% @2048, ~2.3% @4096 whole-prefill). The original
  KL 0.00013 boundary smoke compared 256 vs 1024; a direct 512-vs-1024 re-check
  on the 140W desktop gives KL 0.00012652 / 0.00013192 across two
  independent-session runs and matching top-1 9707.
  Implemented as geometry-keyed `GGUF_PREFILL_CHUNK_SIZES_BY_GEOMETRY = (512,512)`
  for `(H2048-MoE, MOSTLY_Q4_K_M)`; the 27B H5120 is left on 1024 (inconclusive
  within 60W-lane variance). Artifact:
  `benchmarks/results/2026-08-20-gfx1151-qwen36-35b-prefill-chunk-512-retained.json`.
  **140W-lane re-check (2026-08-21, `scripts/pn3_27b_chunk_resolve.py`):** on the
  desktop gfx1151 the 35B ~1.2% win does **not** reproduce (512 −0.03% vs 1024
  at 2048 tok, within 0.8% clock swing) and the 27B question resolves as a null
  (512 +0.16%, within 1.1% swing). The override passes the direct 512-vs-1024
  numerical smoke (KL 0.00012652 / 0.00013192, matching top-1) and stays; the
  ~1.2% claim is marked 60W-ZBook-lane / below that lane's ~10% clock-noise
  floor and does not transfer. Note: the 60W
  `pn3_27b_chunk_pinned.py` 35B sanity leg was a 512-vs-512 no-op because the
  override forces (512,512) for H2048-MoE at session init.
- **Attention launch geometry is already partly gfx1151-specific.** The active
  BF16 prefill scan chooses 32 threads through context 1K and 64 above 1K. For
  c1 BF16 decode below context 1024, gfx1151 explicitly replaces the generic
  short-row 1024-thread route with the fixed256 compact-row leaf; the existing
  admission measured it 1.56-1.65x faster at contexts 513/576/640. Therefore
  "parameterize inherited threads=256" is not a valid next mechanism. Other
  shared helpers still contain 256-thread constants, but none is admitted
  without a current hot-kernel trace and operation-complete ceiling.
- **The registered Laguna WMMA prefill leaf is not shape-compatible.** It accepts
  only 48/72 Q heads, 8 KV heads, and D128; the 35B-A3B is 16 Q heads, 2 KV
  heads, and D256. P3-FULLATTN would need a new 16Q/2KV/D256 specialization,
  not dispatch wiring of the existing leaf.

## Ranked non-overlapping candidates

1. **P3-FULLATTN — BF16 c1 decode attention core+gate, then native prefill.**
   This is arithmetic/launch ownership, **not KV layout**; coordinate the
   agent-2 boundary before changing span dispatch. The durable pre-PN6 profile
   ranks `full_attention_core` at **2.386 ms/token**. PN6 changed blocking
   library-load host work, not these attention kernels, so this remains the
   best admission estimate; nevertheless, rerun the current same-route profile
   before device changes. First screen the active fixed256 leaf at contexts
   128/256/512/640/1023/4K and separate attention from gate wall. The first
   structural candidate is an exact fused fixed256 context-batch+gate output
   (remove 10 gate launches/token and the intermediate F32 round trip), with the
   registered unfused chain retained as strict fallback. Only then consider
   128/512-thread or split-policy variants. QKV/output projections remain the
   separately owned `launch_gguf_linear` family and are not bundled into this
   candidate. **Closed after candidates 1a-1d; the 2026-08-25 refresh does not reopen it.**

   **Candidate 1a closed 2026-08-20 (measured no-win):** the fused fixed256
   context-batch+gate leaf was implemented, RED-tested bit-exact, and measured.
   It removes 10 gate_mul launches/token + the F32 round trip but is flat to
   +0.7% at kernel level (gate_mul is ~1 µs of a ~95 µs attention call at
   context 513) and within-noise whole-decode — the decode is GPU-bound and the
   launch was already hidden. Reverted entirely; default path byte-identical to
   `698465c5a`. See `worklog/entries/20260820T084054.841923Z-lhl-pn3-fullattn-fused-gate-0ef26a.md`
   and `benchmarks/results/2026-08-20-gfx1151-qwen36-35b-pn3-fullattn-fused-gate-no-win.json`.

   **Candidate 1b resolved 2026-08-20 (promoted):** the inherited gfx1100
   `threads=256` block geometry is suboptimal on gfx1151's 40 CUs. A
   parameterized fixed256 body at **threads=1024** is 6-26% faster at the leaf
   at contexts 256-1024 (flat only at ctx 128), passes the calibrated
   execution-profile c1 threads gate (full mtp-bench category suite, KL
   envelope + per-category top-1 >= 97%, 3 repeats), and is promoted as the
   gfx1151 default via `GGUF_SHORT_C1_BATCH_ATTN_THREADS=1024` (T2 non-exact:
   wider warp reduction tree + split value reduction, last-ulp drift). The exact
   256-thread leaf stays registered as the strict fallback
   (`HIPENGINE_GGUF_SHORT_C1_ATTN_THREADS=256` / env restores it). See
   `worklog/entries/20260820T105005.889308Z-lhl-pn3-fullattn-threads-1024-f5c77a.md`
   and `benchmarks/results/2026-08-20-gfx1151-qwen36-35b-c1-short_c1_attn_1024_threads.json`.

   **Candidates 1c/1d closed 2026-08-20:** split-K is a measured no-win for
   the c1 window (20-45% slower at contexts 513-768; the apparent ~1023
   crossover is run noise) and WMMA-tile is closed on analysis (M=1 per block,
   latency-bound leaf, no in-tree 16Q/2KV WMMA decode kernel). The c1 attention
   leaf is at its practical gfx1151 optimum; remaining full-attention headroom
   is the QKV/output projection family (separate owner). See
   `worklog/entries/20260820T124504.891610Z-lhl-pn3-fullattn-splitk-wmma-close-c418f6.md`.
2. **P3-EXPGEMV — closed pending a materially new dataflow.** Current selected-
   Q4 gate/up is 2.435 ms/token, but local128 tile8 already reached only 1.1466x
   / 0.2958 ms projected and missed both frozen gates; tile4, launch-width,
   half-sequential, DP4A/raw, and graph routes are rejected. Q5 tile8 is already
   retained and Q6 variants missed.
3. **P3-MOECOMBINE — closed under current ownership.** Current router/combine is
   1.009 ms/token. The exact shared-expert composite measured 0.899x and genuine
   cross-queue selected/shared overlap regressed complete wall 2.058%.
4. **P3-LMHEAD — closed under current Q6T16 ownership.** The refresh measures
   2.268 ms/token including Q6 projection and two-stage argmax; exact tile8 was
   0.9978x and was removed.

## Do-not-repeat ledger (MoE family, already closed)

- selected-MoE DP4A/Q8_1 routes that failed operation-complete quality/wall;
- row-compact selected-MoE GEMV (large verifier regression);
- one-plane Q8_1 activation (operation-complete SiLU KL failure);
- forced all-width / c2 Q8T16 rowtiling (rejected);
- Q8T16 64-thread verifier pair launch (slower than 128 threads);
- selective unsafe math (7.67% slower at the actual leaf);
- c1 MoE graph (exact but ~0.84% complete-wall regression);
- prompt/token/candidate-specific routing (prohibited benchmark gaming).
- fused fixed256 context-batch+gate leaf (bit-exact but no-win: gate_mul is
  ~1 µs of a ~95 µs attention call; GPU-bound decode hides the launch);
- launch-count reductions in c1 decode as a wall-time lever (host is overlapped);
- split-K (online-softmax + fused gate) for the c1 short-batch window
  (<1024): 20-45% slower at contexts 513-768, no reliable crossover;
  keep split-K only at >=1024 where it already runs;
- WMMA-tile for the single-row c1 decode leaf (M=1 per block, latency-bound;
  WMMA tiles waste 15/16 lanes).

## Venue caveat and lane handoff (updated 2026-08-20)

The ZBook is 60 W power-limited and, measured this session, **cannot be clock-pinned**:
sysfs `pp_dpm_sclk` level pinning and `pp_od_clk_voltage` overrides are rejected,
and `power_dpm_force_performance_level` writes are accepted but ignored. Under
sustained load the sclk settles to a thermal-equilibrium band of ~1180-1415 MHz
(±~10% swing, `scripts/pn3_clock_probe.py`) and recovers to 2900 MHz at every
load gap. Consequences:

- ZBook absolute rates must never be reused as a different-power (120 W
  Radeon 8060S / 140 W desktop) old→new comparison.
- Wall-time A/B verdicts smaller than the clock-band swing (~10%) are
  structurally unreliable on this lane. The best-effort protocol (sustained
  thermal pre-warmup + interleaved counter-rotated legs + per-leg sclk
  sampling, `scripts/pn3_27b_chunk_pinned.py`) can only resolve effects well
  beyond that swing; everything else defers.
- **Lane policy going forward:** the ZBook remains the venue for correctness
gates, RED tests, KL/top-1 suites, kernel bring-up, and structural/launch
  verification. All wall-time retention claims, old→new comparisons, and
  benchmark rollup rows move to the non-power-limited gfx1151 system (140 W
  PL). The interleaved/warmup protocol itself transfers and stays required
  there; on the ZBook it is necessary but often not sufficient.

## 27B dense (QWEN38-27B, closed campaign)

At the practical roof: c1 decode 13.1 tok/s ≈ 83% of the ~16 tok/s bandwidth
roof (27B Q4_K_S ≈ 14 GB @ 221 GB/s); non-temporal loads measured flat e2e
(ROOFLINE 6.6, reverted); the amortization win (MTP B3 1.78x AR) is retained;
prefill is above both llama backends; G5 memory closed. No strong non-
overlapping lever without a specific reopen (e.g. long-context 32K+ attention —
coordinate with agent 2 on the KV boundary).

## Source

- Current-main ownership refresh: `benchmarks/results/2026-08-25-zbook-gfx1151-qwen36-35b-ar-moe-profile-refresh.json`
- Completed SH residual audit: `benchmarks/results/2026-08-06-gfx1151-gguf-sh17-c0-post-sh16-residual-audit.json`
- Stage ranking: `benchmarks/results/2026-08-17-zbook-qwen36-pn3-laq1-declaration-red.json`
- PN5/PN6: `benchmarks/results/2026-08-18-zbook-qwen36-pn{5,6}-*-hoist.json`
- Selected-expert closeout: `benchmarks/results/2026-08-20-gfx1151-qwen36-35b-pn3-moeselect-no-win.json`
- Native/AOTriton crossover: `benchmarks/results/2026-08-20-gfx1151-qwen36-35b-aotriton-prefill-native-retained.json`
- Active decode registration/launch: `hipengine/kernels/hip_gfx1151/__init__.py`,
  `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.{py,hip}`
- Plan: `docs/QWEN36-35B-ZBOOK-PRODUCTION-NUMERICS.md` (closed PN3-PN8 gates)
- Playbook/roofline: `docs/TUNING-gfx1151.md`, `docs/ROOFLINE-gfx1151.md`
