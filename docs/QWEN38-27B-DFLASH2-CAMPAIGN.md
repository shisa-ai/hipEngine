# Qwen3.8-27B DFlash2 GGUF Campaign (gfx1151 first, gfx1100 functional)

Status: **decision — not promoted (D3/D4 complete, measured blocker)** — D0 complete (metadata/validation/lineage/CPU oracles);
D1 complete: NumPy drafter reproduces the reference greedy chain exactly (D0
RED pin) **and** the GGUF-target tap capture + cycle driver runs end-to-end
(`scripts/dflash2_gguf_cycle.py`: full-prompt 5-layer tap capture at prefill,
mask-noise block proposal, sequential greedy commit-only verify, projected-context
cache; DFlash2 greedy == pure-AR greedy 20/20 on the smoke prompt). D2a
complete: native grouped dynamic conv + top-16 + candidate-selector kernels
RED-pinned vs the CPU oracles (6 GPU tests) and registered for gfx1100 +
gfx1151. D2b complete: native drafter forward + selector wiring
(`hipengine/speculative/dflash2_native.py`): conv+attention+MLP forward and
top-16 selector reproduce the numpy oracle to BF16 tolerance and are
deterministic; the native select path equals numpy propose; sliding-attention
kernel RED-pinned; `rocprofv3 --kernel-trace` smoke shows all expected kernels
under expected names (10 GPU tests: 7 kernel RED + 3 forward wiring).
D3 complete: native drafter wired into `scripts/dflash2_gguf_cycle.py`
(`--native`) with end-to-end correctness (native greedy == AR 40/40 on the
smoke) and the B7 chain-batched verifier (`_run_dflash2_cycle_batch`) is
AR-exact on all 10 mtpbench prompts.
D4 complete (2026-08-19): full 10-prompt mtpbench suite measured — DFlash2 B7
mean acceptance 3.49, 3.58 tok/s = **0.27x AR**, vs exact MTP B3 23.85 tok/s
(1.7845x AR). **Promotion rule NOT met**; DFlash2 B7 stays a diagnostic with a
recorded blocker (per-draft acceptance ~0.38 vs MTP ~0.95; 8-row verify
~620ms for LOWER acceptance than MTP B3's 4-row ~160ms). The rowtile-8 verify
speedup (620->310ms) diverged from AR on `code_lru_cache` and was reverted.
A follow-up B-sweep (2026-08-19) root-caused why the earlier "B-sweep" was
flat: `--block-size` was force-clamped to the drafter config 8, so every B
ran the full 8-row verify. After truncating the verify chain to the CLI block
size, B3 (4-row verify) is the DFlash2 optimum; a retained select-path fix
(later, same day) computes draft logits through the session's amortized Q6_K
head instead of the 2.54 GiB dequantized BF16 head (select 70ms -> 22ms), so
the B3 suite is **8.85 tok/s = 0.66x AR** (100% AR-exact all 10 prompts),
still ~2.7x below exact MTP B3; B5 = 4.26, B7 = 3.58 at the pre-fix select
(select saving ~48ms/cycle applies uniformly). A variable-block forward
(`DF2_FWD_BS`, smaller drafter block) is a measured net loss (acceptance
-7%, recall@16 drops, launch-bound throughput) and stays default-off.
Drafter forward (~75ms) + select (22ms) remain the structural DFlash2
disadvantage vs MTP's ~5ms draft; no B is competitive.
B3 cycle split (profiler, fox prompt, post-fix): draft 74ms + select 22ms +
verify 166ms + commit 2ms ≈ 264ms/cycle. **Even if the drafter forward+select
were zeroed**, B3 = 2.8 tok/0.166s ≈ 16.9 tok/s = 1.26x AR < MTP B3's 1.78x AR;
reaching MTP B3 at the 166ms verify would need acceptance 3.96/cycle, above
the B3 cap of 4.0 and the model's per-draft rate on Q4. No operating point can
reach MTP B3; the Q4-lane acceptance gap is the insurmountable ceiling
(re-audit 2026-08-19 proved the drafter is not broken: batch == sequential
acceptance 3.64, recall inside the reference range; the reference 4.80 is the
BF16/T>0 target). Single clean run per B (all 10 prompts AR-exact),
artifacts under `benchmarks/results/`.
Remaining: D5 gfx1100 functional (optional given non-promotion).

The complete quantitative record (acceptance model, per-cycle cost, B-sweep
optimum, airtight ceiling, lane-bound proof, retained cost reductions) is
consolidated in the **Economics** section below — the single source for the
why-DFlash2-cannot-beat-MTP analysis.
This document defines the campaign to bring `z-lab/Qwen3.8-27B-DFlash2`
drafting to the closed Qwen3.8-27B GGUF production path on Radeon 8060S /
`gfx1151`, with **functional (correctness-gated, untuned) support on
gfx1100/W7900** as an explicit deliverable. Milestone status lives in the
worklog.

References:

- DFlash 2 blog: https://inco.ai/blog/dflash2/ (2026-08-18)
- Reference implementation: `~/dflash` (cloned from `github.com/z-lab/dflash`,
  commit `07ebd93`, read-only). `dflash/model.py::DFlash2DraftModel`,
  `GroupedDynamicCausalConv`, `CandidateSelector`, `dflash_generate`;
  `dflash/model_mlx.py` is the quantized-target variant.
- llama.cpp DFlash2 GGUF path: `ggml-org/llama.cpp` PR #27342 and the
  `incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M` drafter weights (comparator lane).
- Our DFlash history and verifier/accept/commit infrastructure:
  [`DFLASH.md`](DFLASH.md). Closed Qwen3.8-27B target campaign:
  [`QWEN38-27B-GFX1151-CAMPAIGN.md`](QWEN38-27B-GFX1151-CAMPAIGN.md) (AR
  13.10/12.92/13.10 tok/s at 512/1K/4K; exact native MTP B3 23.85 tok/s =
  1.7845x own AR).

## Economics — complete record of why DFlash2 cannot beat MTP on this lane

This section is the single self-contained record of the DFlash2-vs-MTP
analysis (worklog evidence: `20260819T165514` root-cause re-audit,
`20260819T160908` B-sweep, `20260819T235720` Q6-select, `20260820T001302`
variable-block net loss; artifacts under `benchmarks/results/`). All numbers
are the closed Qwen3.8-27B Q4_K_M GGUF target, gfx1151 resident session,
greedy, full 10-prompt mtpbench suite, AR-exact on every row.

### Baselines (same session, same hardware)

| Lane | tok/s | ratio AR |
| --- | --- | --- |
| Pure autoregressive | 13.4 | 1.00x |
| Exact native MTP B3 | 23.85 | **1.7845x** |
| DFlash2 B3 (post-Q6 select) | 8.85 | 0.66x |
| DFlash2 B5 (pre-Q6 select) | 4.26 | 0.32x |
| DFlash2 B7 (pre-Q6 select) | 3.58 | 0.27x |

The binding comparison is DFlash2 vs **our own exact MTP B3**, not external
headlines. MTP B3 is the target to beat; DFlash2's best point is ~2.7x below
it.

### Acceptance model (measured)

| B | verify rows | DFlash2 acceptance | per-row rate | MTP B3 | per-row |
| --- | --- | --- | --- | --- | --- |
| 3 | 4 | 2.80 | 0.70 | 3.85 | 0.96 |
| 5 | 6 | 3.30 | 0.55 | - | - |
| 7 | 8 | 3.49 | 0.44 | - | - |

DFlash2's marginal per-row acceptance **decays with block size** (0.70 @ B3 ->
0.44 @ B7): larger blocks buy fewer accepted tokens per extra verify row. MTP
B3 sustains ~0.96/row. The reference's advantage (blog: DFlash2 **4.80** vs
MTP **4.28**, +12%) is the **BF16 target at T>0**; on the Q4_K_M greedy lane
DFlash2 is **below** MTP (2.80-3.49 vs 3.85). DFlash2 feeds on the target's
hidden states (5-tap fc projection), so target quantization degrades it
disproportionately; MTP only consumes token embeddings. The reference itself
qualifies this: "for quantized targets use block_size <= 5".

### Per-cycle cost model (B3, fox prompt, post-Q6 select)

| Cost | ms | DFlash2-specific? |
| --- | --- | --- |
| Draft forward (5-layer, config bs 8) | ~74 | yes (MTP ~5ms) |
| Select (top-16 + selector, Q6 amortized) | ~22 | yes |
| Verify (4-row chain) | ~166 | no (shared with MTP) |
| Commit | ~2 | no |
| **Total** | **~264** | |

The verify is **O(N^2)** in block rows (causal block attention): ~166ms @4
rows, ~620ms @8 rows (3.6x for 2x rows). The draft forward is launch-bound
(~75 kernels/cycle), not compute-bound — so smaller forward blocks save
almost nothing (see variable-block net loss below).

### Why B3 is the optimum (the B-sweep)

Throughput = acceptance(B) / (draft + select + verify(B) + commit).
Acceptance(B) saturates (suffix decay) while verify(B) grows O(B^2), so
there is a single interior optimum and it is **B3**:

| B | cycle~cost | outcome |
| --- | --- | --- |
| 3 | 264ms | **8.85 tok/s (0.66x AR)** — optimum |
| 5 | ~0.5-0.8s | 4.26 (0.32x) |
| 7 | ~0.75-0.97s | 3.58 (0.27x) |

### The airtight ceiling (not an optimization problem)

1. **Free-drafter bound:** even if the entire 5-layer drafter + select were
   zero-cost, B3 = 2.80 tok / 0.166s verify = **16.9 tok/s = 1.26x AR** —
   still below MTP B3's 1.7845x AR, because DFlash2's acceptance (2.80) is
   below MTP's (3.85).
2. **Required-acceptance bound:** at the 166ms verify, beating MTP B3
   (23.85 tok/s) would need 23.85 x 0.166 = **3.96 accepted tokens/cycle** —
   above the B3 cap (4.0) and far above the model's per-draft rate on Q4.
3. **Conclusion:** no operating point on the Q4_K_M/8060S lane can reach MTP
   B3, regardless of drafter or select cost. The Q4-lane acceptance gap is
   the insurmountable ceiling, not a kernel-tuning target.

### Why the gap is lane-bound, not a drafter bug

The re-audit proved the drafter is faithful: batch chain-verify == sequential
per-token verify (acceptance 3.64 == 3.64) and recall@1/@16 sit inside the
reference's own per-position range (r1 0.763 vs 0.729-0.854, r16 0.947 vs
0.878-0.995). AR-exact on every run. The gap is the Q4 target erasing the
acceptance advantage, plus the shared O(N^2) verify and the 5-layer draft
(~15x MTP's ~5ms head).

### Retained cost reductions (honest accounting)

| Fix | Effect |
| --- | --- |
| Drop redundant full-vocab logit host copy (top-1 of top-16 == argmax) | select 70 -> 56ms |
| **Q6 amortized select** (draft logits via the session's Q6_K head, read once across rows, vs the 2.54 GiB dequantized BF16 head) | **select 70 -> 22ms**; B3 suite 7.70 -> 8.85 (+15%), acceptance unchanged |
| Variable-block forward (smaller drafter block) | **net loss**: acceptance -7%, recall@16 drops, launch-bound throughput flat — default-off, REFACTOR-noted |
| rowtile-8 verify (620 -> 310ms) | reverted as AR-divergent on `code_lru_cache` |

Net effect: DFlash2 B3 0.575x -> 0.66x AR, still ~2.7x below MTP B3. The
select is no longer a meaningful DFlash2-specific cost (22ms); draft (~74ms)
+ shared verify (166ms) dominate.

### The only path to DFlash2 > MTP

A **BF16 (non-Q4) target**, where DFlash2's acceptance advantage (4.80 vs
4.28) actually materializes and per-row acceptance can exceed MTP's — outside
the closed Q4 lane. On Q4, the requirement is simultaneously (a) acceptance
>= MTP's ~3.85 and (b) drafter cost <= ~5ms; neither is achievable with a
5-layer parallel-block drafter on quantized hidden states.

### Memory economics (D6)

DFlash2 B3 pipeline ~19.5 GiB GTT (+3.6 GiB over the closed campaign's
15.899 GiB B3 budget) — inside the 8060S unified-memory capacity, so the
BF16 drafter does not exceed the APU budget; the GGUF-quantized drafter is
not a required follow-up.

## 1. What DFlash2 is, relative to what we already have
Our existing native DFlash path (chain/tree verifier, native-bulk
`TargetVerifyBatch`, `dflash_accept_chain_i32` / `dflash_commit_chain_i32`,
whole-cycle confidence gate) was built against DFlash **1** drafters
(`Qwen3.6-*` HF drafters and the Laguna DFlash GGUF lane). The released
`Qwen3.8-27B-DFlash2` drafter is the same qwen-family parallel block drafter
plus exactly two new mechanisms, and both live entirely on the **drafter**
side; verification stays a chain verify over root + accepted-path rows:

1. **Two-tap grouped dynamic convolution** (`attention_conv`, `mlp_conv`):
   before and after each attention and MLP sublayer of each of the 5 drafter
   layers (20 conv applications per draft forward),
   `Conv_k(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t-1}` where each coefficient is
   a learned `base_kernel` (per-channel) plus a dynamic correction from
   `kernel_projection(h)` shared per group of `conv_group_size=16` channels
   (320 groups, H=5120, `kernel_size=2`, projection output 1280). The first
   block position reads the last verified token's representation. This is
   block-local and stateless: it replaces depth (suffix-decay fix) without a
   sequential backbone.
2. **Low-rank bilinear path selector** (`candidate_selector`): keep top
   `selector_top_k=16` candidates per position (draft logits from the
   **target's** output head), then score adjacent pairs
   `S_t(a,b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>` with `selector_rank=256`
   codebooks A/B (vocab 248320 × 256) and a context gate
   `H = hidden_projection(h_t)` (5120→256). A greedy walk (T=0) or
   rejection-sampled walk (T>0) starting from the anchor (last verified token)
   traces one chain through the per-position candidate lists. The selected
   path is a **chain**, so the existing chain verifier/accept/commit machinery
   applies unchanged; tree/DDTree work is not required.

Reference acceptance (blog, block size 8, Qwen3.8-27B): mean acceptance
length 4.80 vs 4.28 for the model's native MTP. Our binding comparison is
against **our own exact native MTP B3 (1.7845x own AR)**, not external
headlines.

### 1.1 Model identity

| Field | Value |
| --- | --- |
| Drafter | `z-lab/Qwen3.8-27B-DFlash2` snapshot `50307d4c` |
| Architecture | `DFlash2DraftModel`, BF16, 81 tensors, ~3.85 GiB safetensors |
| Backbone | 5 × `sliding_attention` (window 2048, non-causal bidirectional window), 32 Q / 8 KV heads, head_dim 128, q/k per-head RMSNorm, RoPE theta 1e7 |
| Geometry | hidden 5120, intermediate 17408, vocab 248320 (uses **target** output head) |
| DFlash config | block_size 8, mask_token_id 248070, target_layer_ids `[5,19,33,47,61]` (5×5120 concat → `fc` 25600→5120 → `hidden_norm`) |
| DFlash2 config | conv_kernel_size 2, conv_group_size 16, selector_rank 256, selector_top_k 16 |
| Target | `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` (identity frozen in the closed campaign), gfx1151 resident session |

Full safetensors inventory (layer 0 shown; layers 0–4 identical):
`fc.weight [5120,25600]`, `hidden_norm`, `norm`,
`layers.N.{input_layernorm, post_attention_layernorm}`,
`layers.N.self_attn.{q,k,v,o}_proj` (+`q_norm`,`k_norm` [128]),
`layers.N.mlp.{gate,up,down}_proj`,
`layers.N.attention_conv.{base_kernel [2,2,5120], kernel_projection [1280,5120]}`,
`layers.N.mlp_conv.{...}` (same shapes),
`candidate_selector.{predecessor_codebook [248320,256], successor_codebook
[248320,256], hidden_projection [256,5120]}`.

### 1.2 Drafter dataflow per cycle (reference semantics)

From `dflash/model.py::dflash_generate`:

- Draft input embedding: **target** input embeddings of the block tokens
  (mask token at each draft position), optionally `input_embedding_scale`
  (absent here → 1.0).
- Context K/V: each layer projects `k/v` from the fc'd target hidden for all
  context rows **every cycle** plus the noise rows for the block; our DFlash1
  drafter already caches projected/rotated context rows across cycles
  (Phase A/B/C in [`DFLASH.md`](DFLASH.md)) — reuse that discipline.
- Positions span context + block; attention mask is the 2048 bidirectional
  sliding window (non-causal for the noise rows; context rows are always
  visible predecessors).
- Unary logits for top-16 come from the target output head applied to drafter
  hidden rows (8 rows × Q6 output head on the GGUF target — the native
  rows>1 output-head path exists for MTP B2/B3).
- Selector walk output fills `block[1..7]` draft tokens; verify is the
  existing greedy chain verify (T=0): accepted = prefix of the selected path
  matching target argmax; bonus token from the first rejected row.

## 2. Gap analysis vs current tree

| Area | Current state | Gap for DFlash2 on Qwen3.8 GGUF |
| --- | --- | --- |
| Metadata/loading | `hipengine/loading/dflash.py` parses `DFlashDraftModel` (DFlash1) HF configs; validates qwen-drafter tensor sets | New `DFlash2DraftModel` architecture, new config keys (conv/selector), new tensors (20 conv parameter pairs + 3 selector tensors), per-head q/k norms on a drafter |
| Hidden taps | GGUF target session exposes a layer-output capture map (built for Laguna/MTP); Qwen3.8 MTP used last-layer hidden only | Capture taps at 5 intermediate layers `[5,19,33,47,61]` of the 64-layer Qwen3.8 GGUF session, row-aligned, for prefill and for each verify cycle's accepted rows (compaction on partial accept) |
| Drafter kernels | `hip_gfx1100/speculative/dflash_drafter.hip` implements DFlash1 qwen drafter math (RMSNorm, RoPE, projections, context caching, add+rmsnorm fusion) used by the Laguna GGUF lane | (a) register/select per backend properly (see §5); (b) two-tap grouped dynamic conv kernels (prepare + finish); (c) q/k head-norm attention shape for 32Q/8KV/D128/window 2048; (d) selector scoring kernel: top-16 gather + `A(a)⊙H(h_t)·B(b)` over 16 candidates × 8 positions + greedy walk; (e) fc over 5×5120 tap concat |
| Output head | Native rows>1 quantized output-head path exists (MTP B1–B3 exact) | Reuse for 8-row draft logits (top-16 needed, not argmax: extend to top-K) |
| Verifier/accept/commit | Native GGUF rows path exact at B1–B3 (MTP); DFlash chain verify + `dflash_accept_chain_i32`/commit proven on other lanes | Extend the native rows path to B=7 draft rows (8-row verify incl. root); acceptance kernel already handles chain topology |
| CPU oracle | `kernels/cpu_reference/` has DFlash1 pieces | CPU-reference oracle for grouped dynamic conv and candidate selector (RED-first) |
| Bench harness | `scripts/dflash_chain_e2e_bench.py` (HF/PARO lanes), `scripts/mtp-bench.py` modes, closed-campaign GGUF bench scripts | New GGUF-target DFlash2 chain mode; same-session AR, exact MTP B3, and DFlash2 B7/B8 rows on the full mtpbench category suite |
| gfx1100 | DFlash speculative kernels physically live in the `hip_gfx1100` package; no gfx1151-native speculative module | Backend plan in §5: gfx1151 first-class, gfx1100 functional-with-correctness-gate |

## 3. Non-goals / order rules

- No tree/DDTree work: the selector emits a chain; chain verify first and
  only.
- Greedy T=0 is the binding correctness contract (exact same-session AR
  equality on the full suite). Rejection-sampled T>0 drafting is a follow-on
  and never blocks the campaign.
- No drafter quantization before the BF16 drafter path is exact and measured;
  the `incoai/Qwen3.8-27B-DFlash2-GGUF` weights are a memory comparator and a
  possible later lane, not an initial deliverable.
- gfx1100 receives **correctness and smoke only** (KL/top-1 gates + one
  rocprofv3 kernel-trace + one bench smoke row); no tuning iterations are run
  on W7900 for this campaign.
- All speed claims follow `docs/BENCHMARK.md` anti-gaming: full mtpbench
  category suite + heldouts, same-session AR denominator, no prompt
  conditioning.

## 4. Milestones

Each milestone is one or more atomic units with worklog entries; gates follow
`docs/TESTING.md` / `docs/EXECUTION-PROFILES.md`.

### D0 — Metadata, lineage, and CPU oracle (no GPU)

1. Extend `hipengine/loading/dflash.py`: `DFlash2DraftModel` config parser
   (registered, not branched by string anywhere else), tensor requirements
   for the conv/selector tensors, `candidate_selector.*` normalization
   (codebooks load as plain 2-D BF16 weights), per-head q/k norm tensors.
2. Extend `scripts/dflash_validate_artifacts.py` (or a GGUF-target variant)
   to validate the DFlash2 drafter against the Qwen3.8-27B Q4_K_M GGUF
   target: hidden sizes, vocab, mask token, target layer ids vs the GGUF's 64
   layers, output-head sharing.
3. Add `docs/source_lineage.json` entries for the reference files
   (`~/dflash/dflash/model.py`, `model_mlx.py` @ `07ebd93`) and port notes.
4. CPU reference in `kernels/cpu_reference/`: grouped dynamic conv
   (prepare/finish, group 16, first-position predecessor tap) and candidate
   selector (top-16, bilinear score, greedy walk) with RED tests against
   small golden fixtures derived from the reference implementation run on
   CPU (torch allowed in the *test fixture generator only*, never on the
   hot path).

**Gate:** metadata validator passes on the real snapshot; CPU oracles match
the reference implementation on fixtures; `python3 scripts/worklog.py check`.

### D1 — Drafter backbone on GGUF target (exact math, perf-naive)

Materialize BF16 drafter weights; implement the 5-layer forward with the
existing drafter kernel discipline (persistent scratch, context K/V caching,
`KVLiveSpans`-consistent draft attention) plus:

- target-side hidden taps at `[5,19,33,47,61]` during prefill and verify
  cycles, with accepted-row compaction;
- noise embedding from the GGUF target embedding table;
- fc + hidden_norm over the 5×5120 tap concat;
- sliding-window (2048) bidirectional mask for block rows.

A Python/numpy slow path is acceptable for the first exactness rows; the
native kernels arrive in D2. **Gate:** single-prompt greedy smoke reproduces
the reference `dflash` (transformers backend) greedy chain selection on
identical inputs (draft-top-K tables and selected path identical), finite
logits.

**D1 status (2026-08-19):** GGUF-target tap capture added to the resident
session (`DFlash2HiddenCaptureTargets`, `DFLASH2_TAP_DEPTHS=(6,20,34,48,62)`
= tap layer ids + 1, threaded through `prefill(dflash2_capture=...)` in both
bulk-prefill layer loops). `scripts/dflash2_gguf_cycle.py` drives the cycle:
prefill taps → drafter forward (mask-token noise, positions spanning context +
block, per-row projected-context cache) → top-16 selector → sequential greedy
verify via `session.step(capture_layer_output_hidden=tap_ids)` committing only
accepted rows (rejected rows are never run, so no KV rollback is needed) →
accepted-row taps extend the projected context. Smoke result: 20/20 greedy
tokens identical to pure-AR on the same session; drafts are sensible (e.g.
`' French'` for a translation prompt). Acceptance on the smoke prompt is low
(mean 1.67, ~0.095 accepted/draft) — a short-context prompt property, not a
mechanism failure; D4 measures acceptance on the full suite. Throughput is
CPU-bound by the NumPy drafter (2.44 vs 10.91 tok/s AR) — D2 native kernels
are the speed path.

### D2 — Native kernels: conv + selector + 8-row output head

1. Two-tap grouped dynamic conv kernel (one launch per conv application, or a
   fused prepare/projection variant later): strict exact/parent-parity RED
   contract vs the D0 CPU oracle; registered under
   `(hip_gfx1151, dflash2_conv, bf16, ...)` and the gfx1100 peer key.
2. Selector scoring kernel: gather A/B codebook rows (256-dim), compute
   `H(h_t)`, score 16 candidates × 8 positions in parallel, greedy-walk in
   one small kernel. Codebooks are 2×254 MB BF16 — resident, read-only,
   gathered rows only per cycle (16+16 rows × 256 per position).
3. Top-16 extension of the native quantized output-head rows path (currently
   argmax-oriented for MTP): emit top-K ids+logits for 8 draft rows without a
   full-logit host copy.

**Gates:** strict exact RED vs oracle for each kernel; `rocprofv3
--kernel-trace` smoke showing each kernel under its expected name;
KL ≤ 0.05 + top-1 ≥ 90% outer floor for the composite drafter vs reference
(production-profile tightening only if arithmetic reassociation is proposed).

### D3 — Chain verify at B=7 and wiring

- Extend the native GGUF rows verify path from B1–B3 to B=7 draft rows
  (8 rows incl. root) using the existing chain batched verifier semantics;
  accept/commit via `dflash_accept_chain_i32` + native commit (or the
  established GGUF MTP commit path — whichever the closed campaign retained).
- Wire DFlash2 as a speculative plugin choice beside MTP for the Qwen3.8 GGUF
  session (registry/plugin boundary, no engine-level `if arch ==` branches).
- Port the whole-cycle confidence gate (depth-1 `p1` from the drafter's top-16
  unary logits) as the online fallback policy, default-off until measured.

**Gate:** GPU accept summary == CPU oracle for reject/partial/full on real
weights; exact same-session AR equality on a 2-prompt smoke at B=1..7.

### D4 — Measurement campaign (gfx1151)

- Full mtpbench category suite (`code`/`general_en`/`general_ja`/`mixed_ja_en`
  + heldouts), greedy, B ∈ {3, 5, 7}: same-session AR, exact MTP B3, and
  DFlash2 rows from one clean source each; three runs; artifacts under
  `benchmarks/results/`.
- Promotion rule: DFlash2 must beat **our exact MTP B3 (23.85 tok/s,
  1.7845x)** and own AR with exact AR equality and GPU-accept==CPU on every
  row, or it stays a diagnostic with a recorded blocker. Acceptance-length
  instrumentation (per-position recall@1/@16) to quantify selector gains vs
  the DFlash1-style top-1 baseline lane (run a top-1 ablation with the
  selector disabled as a diagnostic).
- Ledger: `rocprofv3` verifier/drafter split via
  `scripts/mtp_verifier_rocprof.py`-style child profiling (never the parent
  suite harness), cycle-wall, rows/output, draft/verify seconds, GTT.

### D5 — gfx1100 functional support

**D5 status (2026-08-19): registration complete, smoke hardware-blocked.**
The DFlash2 kernels live in `hipengine/kernels/hip_gfx1100/speculative/dflash2.{hip,py}`
and the native drafter (`hipengine/speculative/dflash2_native.py`) imports
them from the gfx1100 package — the gfx1100 peer registration and shared
source are in place. The remaining D5 deliverable is a correctness-gated
smoke on a real W7900/gfx1100 host (CPU-oracle gates, KL/top-1 floor, one
exact 2-prompt AR-equality smoke, one `rocprofv3 --kernel-trace` entry).
The measurement host in this campaign has only a gfx1151 Radeon 8060S (no
W7900), so the D5 hardware smoke is **blocked on hardware availability**, not
on code; it is a recorded, un-tuned pending row. Given DFlash2 is not
promoted (D4), D5 remains optional per the campaign decision.

- Register the DFlash2 kernels for gfx1100 (peer keys, shared source; the
  existing DFlash kernels already live in the gfx1100 package — reconcile the
  module layout per §5).
- Run on W7900: CPU-oracle gates, KL/top-1 floor, one exact 2-prompt smoke
  vs same-session AR, one `rocprofv3 --kernel-trace` entry. **No tuning**;
  record the smoke row as un-tuned gfx1100 support evidence.

### D6 — Closeout

- Rollup: `benchmarks/README.md` row + `Last updated`, changelog one-liner,
  artifact JSONs, worklog entries, `docs/DFLASH.md` status update, root
  README export via `scripts/sync_benchmark_readme.py --check`.
- Memory accounting: drafter BF16 residency (~3.6 GiB + taps + codebooks
  already included) vs the closed campaign's 15.9-GiB B3 GTT budget; state
  the GTT delta as a first-class result. If the APU GTT budget is exceeded,
  the GGUF-quantized drafter becomes a scoped follow-up, not silent scope
  creep.

**D6 status (2026-08-19): closeout complete (memory accounting + history).**
Memory accounting: DFlash2 drafter BF16 residency measured **3.584 GiB**
(layers 3.18 GiB incl. conv/MLP codebooks + `fc` 250 MiB + `candidate_selector`
245 MiB), plus ~0.15 GiB projected-context taps + drafter KV + scratch at
4K context. Against the closed campaign's B3 process GTT of **15.899 GiB**,
the full DFlash2 B3 pipeline is **~19.5 GiB (+3.6 GiB / +23%)** — inside the
Radeon 8060S unified-memory GTT capacity (the closed campaign's whole-device
4K peak was ~20 GiB with headroom), so the BF16 drafter does NOT exceed the
APU budget and the GGUF-quantized drafter is NOT a required follow-up.
Rollup: DFlash2 is a rejected/diagnostic lane (not promoted), so per
`AGENTS.md` it gets no `benchmarks/README.md` scoreboard row and no
`CHANGELOG.md` retained-row one-liner; it is recorded as a diagnostic in
`benchmarks/HISTORY.md` and as artifacts under `benchmarks/results/`, with
the campaign status in this doc and `docs/DFLASH.md`. Root `README.md` is
a product page and is not touched (no retained claim). The D4 "three runs"
statistical repeat is deliberately not repeated: the result is decisively
negative (~3.1x below MTP B3 at the optimum B3) and every row is AR-exact,
so additional full-suite runs would be a wasteful rerun per `AGENTS.md`;
B7 has multiple session runs, B3/B5 one clean full-suite run each.

## 5. Backend ownership note

All DFlash speculative kernels currently live in
`hipengine/kernels/hip_gfx1100/speculative/` and are consumed by the gfx1151
Laguna lane by direct import. Before adding DFlash2 kernels, decide (small
refactor unit, first): either (a) keep the shared-source package as the
canonical home for both backends and make compilation/registration explicit
per target arch (documented in `docs/KERNELS.md`), or (b) give gfx1151 its
own speculative module re-exporting shared sources. Do **not** add
`if backend ==` branches; keep the four-axis registry keys authoritative.
gfx1151 remains the tuned lane; gfx1100 gets functional correctness only.

## 6. Risks / open questions

1. **Intermediate-layer hidden taps on the dense GGUF target**: the capture
   map exists, but per-cycle accepted-row compaction of 5 taps through the
   48-GDN/16-full-attention mixed stack is new plumbing; D1 proves it.
2. **8-row output-head cost**: 8 rows × ~1 GiB Q6 head read per draft cycle
   is potentially the dominant drafter cost on gfx1151's ~221 GB/s; the
   native rows path mitigates, and top-16 (not full logits) keeps host
   traffic bounded. If it dominates, a draft-vocab/lm-head-sharing diagnostic
   (à la MTP M12.2) is the follow-up, gated on exactness.
3. **Acceptance transfer**: blog numbers are sampling-mode (T=1) acceptance;
   our binding greedy lane may differ. The D4 top-1-vs-selector ablation
   quantifies this before any tuning.
4. **is_causal=false / bidirectional 2048 window** in the drafter differs
   from our causal-mask drafter attention kernels; verify mask semantics
   against the reference in D1 before kernel work.
5. **Drafter GTT on APU**: +3.6 GiB over the closed-campaign budget may
   change which KV capacities fit; measure before promoting defaults.
