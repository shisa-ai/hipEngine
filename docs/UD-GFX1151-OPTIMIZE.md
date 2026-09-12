# UD gfx1151 Optimization Plan

Last updated: 2026-09-12.

This is the active optimization and certification handoff for the published
Qwen3.8-27B Unsloth Dynamic `Q4_K_M` and `Q4_K_S` artifacts on the active
gfx1151 lane: the RX 7900 XTX in physical GPU1. It starts from the valid
graph-replay paired measurement and tracks absolute throughput as well as
relative parity. A ratio improvement is not sufficient if both paths get
slower; an absolute improvement is not sufficient if it makes UD MTP
economically unattractive.

## References

### Governing architecture and policy

- [Project plan](PLAN.md)
- [UD quant support and U0-U7 campaign](UD-QUANTS.md)
- [UD quant reproduction and identity notes](UD-QUANTS-REPRO.md)
- [UD optimized-route plan](UD-OPTIMIZED-ROUTE-PLAN.md)
- [Execution profiles and numerical gates](EXECUTION-PROFILES.md)
- [Testing and RED/GREEN workflow](TESTING.md)
- [Benchmark protocol and evidence policy](BENCHMARK.md)
- [Kernel catalog and lineage workflow](KERNELS.md)
- [Roofline and gfx11 performance model](ROOFLINE.md)
- [Refactor debt ledger](REFACTOR.md)

### Current baseline and evidence

- [Valid paired GPU1 artifact](../benchmarks/results/paired-ud-plain-mtp-c1-natural25-b3-graph-xtx.json)
- [Benchmark scoreboard](../benchmarks/README.md)
- [Benchmark changelog](../benchmarks/CHANGELOG.md)
- [U6 certification artifact](../benchmarks/results/ud-mtp-certification-u6.json)
- [K_S near-tie localization](../benchmarks/results/ud-mtp-ks-near-tie-localization.json)
- [Latest paired-run worklog](../worklog/entries/20260911T214114.050532Z-lhl-publish-graph-baseline-ud-mtp-parity-62f377.md)
- [U6 certification worklog](../worklog/entries/20260911T162055.348538Z-lhl-ud-mtp-certification-u6-unit-564c7d.md)
- [Published UD campaign](QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md)

### Required benchmark and profiling tools

- `scripts/ud_mtp_paired.py`
- `scripts/ud_mtp_certification.py`
- `scripts/mtp_verifier_rocprof.py`
- `scripts/mtp_chain_e2e_smoke.py`
- `scripts/gguf_true_ar_category_bench.py`
- `scripts/gguf_mtp_category_bench.py`
- `benchmarks/prompts/mtpbench-code-general-ja.jsonl`

## 1. Objective

Close the UD/plain performance gap on gfx1151 while preserving the
artifact-qualified correctness and admission contracts.

The primary objective is:

> For each promoted UD tier, `UD MTP / UD AR > 1.0` under the valid
> graph-replay protocol and the complete certification suite.

An MTP route below `1.0x` versus the same UD artifact's true no-MTP AR path
is not worth running, regardless of its acceptance rate. The secondary
objective is to improve UD/plain parity without regressing absolute UD
throughput:

- AR: increase UD tok/s and move `UD AR / plain AR` toward `1.0x`.
- MTP: increase UD tok/s and move `UD MTP / plain MTP` toward `1.0x`.
- Economics: increase `UD MTP / UD AR` above `1.0x`.
- Correctness: retain the applicable strict or production-profile gates.
- Scope: certify only measured artifact, backend, profile, width, and context
  envelopes.

No optimization is retained from a single prompt, fixed token sequence,
candidate-specific branch, or verifier-only denominator.

## 2. Current Starting Point

The following is the valid GPU1 / RX 7900 XTX baseline from the paired
graph-replay artifact. Rates are tokens per second.

### Absolute throughput

| Tier | UD AR | UD MTP B3 | Plain AR | Plain MTP B3 | UD MTP / UD AR | Plain MTP / Plain AR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | 31.993 | 35.541 | 37.186 | 63.465 | 1.1109x | 1.7067x |
| Q4_K_S | 31.311 | 33.061 | 39.565 | 64.094 | 1.0559x | 1.6199x |

### UD versus plain parity

| Tier | UD / plain AR | UD / plain MTP B3 | AR gap to plain | MTP gap to plain |
| --- | ---: | ---: | ---: | ---: |
| Q4_K_M | 0.860x | 0.560x | -5.193 tok/s | -27.924 tok/s |
| Q4_K_S | 0.791x | 0.516x | -8.254 tok/s | -31.033 tok/s |

### Interpretation

- K_M is currently the best certification and optimization target.
- K_S has the larger AR and MTP deficits and repeats two known
  `general_ja_plan` near-tie divergences.
- Prefill is already relatively close at approximately `0.918x` K_M and
  `0.944x` K_S parity; it is not the first optimization target.
- The largest deficit is MTP target verification, not the repaired AR host
  denominator.
- Both UD MTP admission pins remain empty.

Every future result must report the equivalent absolute and ratio fields. A
ratio-only table is insufficient.

## 3. Non-Negotiable Measurement Rules

1. Use physical GPU1 explicitly and record device identity.
2. Use the production graph-replay path for the true no-MTP AR denominator.
3. Keep model file, quant tier, prompt fixture, decode length, warmup,
   compiler version, KV policy, and timing protocol fixed within a comparison.
4. Use the complete category suite:
   `code`, `general_en`, `general_ja`, and `mixed_ja_en`.
5. Include the declared category-heldouts for every certification result.
6. Use at least two fresh-process deterministic repeats for candidate runs.
7. Compare MTP against true UD AR, never against verifier `off` or `B0`.
8. Profile the leaf verifier workload, not a parent harness that spawns
   profiled children.
9. Record absolute rates, UD/plain ratios, UD MTP/AR ratio, acceptance,
   control-plane accounting, and correctness gates in the same artifact.
10. Do not retain a change that improves a fixed prompt while regressing the
    full suite or its heldouts.

## 4. Work Phases

### Phase 0: Freeze and reproduce the baseline

- [ ] Verify a clean worktree and record the starting commit.
- [ ] Reproduce the paired artifact from its raw payloads with
  `--from-raw`; confirm byte-identical regeneration.
- [ ] Run a short smoke to confirm physical GPU1, model identity, graph replay,
  and expected UD/plain route selection.
- [ ] Copy the baseline rates into the new optimization result manifest.
- [ ] Create a new immutable worklog entry for the optimization unit.

Exit criteria: baseline artifact validates, device identity is correct, and
all future comparisons can be traced to this exact starting point.

### Phase 1: Attribute the MTP deficit

Run `rocprofv3 --kernel-trace` on the leaf target-verification workload for
UD K_M, plain K_M, UD K_S, and plain K_S. Keep draft generation, target
verification, acceptance/commit, host submission, and graph replay separately
visible.

- [x] Record kernel names, launch counts, duration, stream, grid, and block.
- [x] Attribute wall time to decoder, projection, norm, gate/up, down, logits,
  synchronization, and host/device transfer categories.
- [x] Compare row/tile decomposition between UD and plain.
- [x] Check whether UD uses BF16 expansion where plain uses compressed
  consumers. No dequantize/expand/repack kernel runs in any verifier window;
  UD reads raw IQ bytes in `gguf_iq_dense_strict`.
- [x] Confirm whether the deficit is steady verifier work or transition-only
  overhead. It is steady per-step verifier work.
- [x] Check Q5/Q6/IQ/Q3 decoder launch count and occupancy (grid, block, VGPR,
  SGPR). Memory traffic counters were not collected.
- [ ] Capture c1, c2, c4, and c8 where the caller supports them. The census is
  a single-request leaf; width capture needs a multi-request caller.
- [x] Capture representative verifier rows such as 6, 9, 12, 16, 28, and 32.
  Rows 4, 5, 6, and 8 were captured in Phase 5, after the four-row cap was
  lifted. Rows 9 and above are not reachable: the native target graph is
  defined for two to eight rows, so 9/12/16/28/32 would need either a different
  verifier shape or a caller that splits the block. Evidence:
  `benchmarks/results/2026-09-12-ud-gfx1100-verifier-rows-4-8-census.json`.

Decision gate:

- If verifier device work dominates, optimize the responsible kernel family.
- If host or synchronization dominates, optimize submission/graph ownership
  before changing arithmetic.
- If expansion/residency dominates, qualify compact residents and consumers.
- If only a narrow shape loses, keep the optimization shape-scoped.

No kernel change starts before this attribution is recorded.

**Result (2026-09-12, physical GPU1 / RX 7900 XTX / gfx1100).** The hypothesis
is confirmed. Device work dominates (kernel share 0.861-0.928, host residual
4.3-5.3 ms/step). At the B3 shape, UD's IQ-family quants run
`gguf_iq_dense_strict` (51.6% of UD-Q4_K_M verifier kernel time, 68.0% of
UD-Q4_K_S) while plain's Q4_K/Q5_K/Q6_K/Q8_0 run the weight-amortized rowtile
family and never launch it. UD/plain verifier kernel time is 1.97x (K_M) and
2.25x (K_S), matching the paired production ordering. The loss is
shape-scoped: the owner-map dead zone runs from 2 to 7 rows because the
dense-IQ decode owner is `rows == 1` and the dense-IQ prefill owner has
`min_rows == 8`. The selected action is to optimize the responsible kernel
family with a rows 2-4 IQ verifier sibling. Lowering `min_rows` is not the
fix: the recorded crossover sweep measures the existing prefill owner at 0.59x
(2 rows) and 0.68x (4 rows). Evidence:
`benchmarks/results/2026-09-12-ud-gfx1100-phase1-attribution.json`,
`worklog/entries/20260912T081032.217161Z-lhl-ud-phase1-attribution-a570a6.md`.

### Phase 2: Transfer prefill wins into verifier shapes

Prefill parity establishes that a resident/layout/kernel combination is
competitive at large M. It does not establish that the same route is efficient
for MTP verification, where rows are usually small and are often split into
physical verifier tiles. Treat every prefill owner as a verifier candidate,
not as an automatic verifier default.

For every retained prefill owner, classify its verifier behavior:

- [x] **Direct transfer:** the same resident, layout, kernel family, and
  dispatch are efficient at verifier rows. Q4_K/Q5_K/Q6_K/Q8_0 (rowtile),
  norm/residual, gate/up SiLU, and attention/GDN.
- [x] **Verifier sibling:** the same resident/layout is retained, but a
  small-row or row-tile kernel is required. IQ4_XS/IQ4_NL/IQ3_S/IQ3_XXS/
  IQ2_S/IQ2_XS: the rows==1 local32 owner already wins 2.0-7.4x at rows 2-4.
- [x] **Prefill-only:** the route is efficient for large M but loses at
  verifier rows and remains scoped to prefill. The W4A16 IQ prefill owner
  loses at rows 2-4 on every IQ quant and wins from 8-16 rows.
- [x] **Shared bottleneck:** none found. The dominant loss is local to one
  kernel family, not shared across residents or consumers.
- [x] **Non-transferable:** none found. Q3_K is *not* non-transferable; it
  needs a verifier sibling that does not exist yet.

Measure each classification with the same tensor and resident where possible:

| Measurement | Required values |
| --- | --- |
| Prefill M | representative retained prefill M values. Rows 1-128 were swept. |
| Verifier rows | 2, 4, 8, 12, 16, 28, and 32 where supported. The production
  verifier is capped at four rows, so 2 and 4 are the verifier shapes; 8-32
  are the prefill crossover. |
| Physical decomposition | actual row tiles and remainder launches. The strict
  GEMV does not retile by row; it decodes once per output block. |
| Kernel family | selected registry key and kernel name. Recorded per family. |
| Launch count | per logical target-verification transition. 135 IQ strict
  launches/step at rows 4. |
| Device time | total and time per logical output row. Recorded. |
| Residency | source format, resident layout, expansion status, sidecars. Raw
  IQ blocks, no expansion. |
| Correctness | strict/production gate, KL, top-1, repeatability. Per-cell max
  relative error <= 5.4e-3; the production gate is Phase 3. |

Prioritize the families already implicated by the gfx1151 differential
evidence:

- [x] Q6 lm-head and other wide-Q6 verifier sweeps. Direct transfer
  (`q6_k_t16_qmicro_planar_gemv_rowtile_col8`).
- [x] Q5 `ssm_out` and selected-expert/direct Q5 consumers. The Q5_K
  gate/up dual already runs the WMMA prefill kernel at verifier rows and is
  the cheapest gate/up arm measured. The selected/direct tail
  (`qk_t16_selected_direct_gemv`, 0.6% of kernel time) is classified from the
  census, not re-measured.
- [x] Q4/IQ gate-up and SiLU paths. Direct transfer.
- [x] IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, and IQ2_S small-row consumers.
  Verifier sibling required; headroom measured.
- [x] Norm, residual, activation, and logits work surrounding the GEMMs.
  Direct transfer.
- [x] Row packing, launch reuse, and graph synchronization between those
  consumers. No per-row-tile weight rescan, so the sibling keeps the same
  135 launches/step and only cuts per-launch cost.

For each transferred candidate:

- [x] Confirm UD and plain select comparable resident/layout routes. Both
  arms run the same t16 rowtile kernels for Q4_K/Q5_K/Q6_K/Q8_0.
- [x] Confirm the optimized route is not accidentally selected only for one
  artifact or fixed prompt. `GGUF_IQ_DENSE_PREFILL_POLICY` is keyed by quant
  and rows only.
- [x] Compare the prefill winner against the current verifier owner at every
  declared row shape. Rows 1-128 swept for all seven IQ quants.
- [x] Check whether the verifier is rescanning the same weights per row tile.
  No: the strict GEMV's time is flat from rows 1 to 4.
- [x] Check whether a batched owner reduces launches without changing exact
  ownership, masks, or rollback semantics. It does not reduce launches here;
  the win is per-launch decode cost.
- [ ] Add a focused RED test for the row-shape/dispatch contract.
- [ ] Keep the existing strict verifier fallback registered.
- [ ] Capture a kernel trace showing the intended owner actually ran.

Transfer decision:

- Reuse unchanged only when it wins at the verifier row shapes and passes the
  verifier numerical gate.
- Add a verifier sibling when the resident/layout is good but large-M tiling
  is the wrong geometry.
- Do not route MTP through a bulk-prefill kernel merely because it won
  prefill; prove the row-shape economics first.
- Do not claim a transfer win from kernel time alone. Re-measure complete
  target verification and end-to-end MTP.

The result must record both the local transfer effect and the full economics:

| Candidate | Prefill result | Verifier result | UD AR | UD MTP | UD MTP/AR | UD/plain MTP | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Prefill owner transfer | unchanged / improved / regressed | unchanged / improved / regressed | measured | measured | measured | measured | retain / hold |

This phase is complete only when every important prefill owner is classified as
direct transfer, verifier sibling, prefill-only, shared bottleneck, or
non-transferable. Unknown transfer behavior is an attribution gap, not a
negative result.

**Result (2026-09-12, physical GPU1 / RX 7900 XTX / gfx1100).** Every
important dense owner is classified. The rowtile family, norm/residual, gate/up
SiLU, and attention/GDN transfer directly. The W4A16 IQ prefill owner is
prefill-only: it loses at rows 2-4 on all seven IQ quants (0.21x-0.74x) and
wins from 8-16 rows. The IQ family needs a verifier sibling, and the rows==1
local32 decode owner already beats the strict per-row GEMV by **2.0-7.4x at
rows 2-4** on the same tensors, so the sibling is a small-row geometry of an
already-admitted owner. A rows 2-4 sibling covers **94.1% (K_M) / 85.9%
(K_S)** of the dead-zone MACs; Q3_K (2.28% / 8.57%) has no rows==1 owner and
needs separate work. The strict GEMV is weight-decode-bound, not row-bound (its
time is flat from rows 1 to 4), so there is no per-row-tile rescan and launch
reduction is not the lever. Evidence:
`benchmarks/results/2026-09-12-ud-gfx1100-phase2-owner-transfer.json`,
`worklog/entries/20260912T081726.905837Z-lhl-ud-phase2-owner-transfer-67c4cc.md`.
The candidate/result table below is filled by Phase 4 (local transfer effect)
and Phase 6 (full economics).

### Phase 3: Establish the teacher-forced numerical gate

Complete section 6.1 for the single-row AR route versus multi-row target
verification. This separates arithmetic drift from control and batching bugs.

- [x] Measure mean, p95, p99, and maximum row KL.
- [x] Measure per-category top-1 agreement.
- [x] Cover canonical prompts and category-heldouts. 10 canonical plus 8
  heldout, four categories.
- [ ] Cover c2/c4/c8 and the relevant verifier row shapes. Verifier rows 2/3/4
  (B1-B3) are covered at c1; c2/c4/c8 need a multi-request caller.
- [ ] Cover short, 512, 4096, and a separately budgeted long context such as
  32768. Short (64) and 512 are covered; 4096/32768 are open.
- [ ] Cover Q8 alpha/beta recurrent transitions. Not isolated as its own
  scope; the prompt set exercises them implicitly.
- [x] Cover full attention and mixed FFN gate/up pairs. The four categories
  exercise both attention kinds and the fused gate/up pairs.
- [x] Cover F32 logits and sampling where used by the serving path. Full F32
  logits are the gate's comparison surface; the serving path here is greedy.
- [x] Repeat under eager and graph replay. The 64-token arms use the captured
  native graph; the 512-token arms use the eager multi-row verifier.
- [ ] Check deterministic replay and batch-composition invariance.
  Deterministic replay passes; batch composition needs c2/c4/c8.

The gate must identify whether K_S near-ties are permitted production drift
or a binding batch-composition failure. Do not widen the admission pin to make
the result pass.

**Result (2026-09-12, physical GPU1 / RX 7900 XTX / gfx1100).** Both UD
presets pass the complete section-6.1 screen with two to three orders of
magnitude of margin, at a short root and at a 512-token root:

| Arm | rows | root | verifier path | mean KL | p95 | p99 | max | top-1 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| UD-Q4_K_M | 162 | 64 | native graph | 3.45e-05 | 2.03e-04 | 2.56e-04 | 3.48e-04 | 100% |
| UD-Q4_K_M | 162 | 512 | eager multi-row | 3.51e-05 | 2.46e-04 | 4.39e-04 | 7.00e-04 | 100% |
| UD-Q4_K_S | 162 | 64 | native graph | 3.06e-05 | 1.31e-04 | 2.60e-04 | 6.98e-04 | 100% |
| UD-Q4_K_S | 162 | 512 | eager multi-row | 1.03e-05 | 6.65e-05 | 8.27e-05 | 1.07e-04 | 100% |

Top-1 is 100% in every category and both repeats of every arm are
bit-identical. The K_S generated-ID near-ties are therefore **permitted
production drift**, not a batch-composition failure: at identical contexts the
verifier reproduces the AR logits to max row KL 1.07e-04. Also recorded:
gfx1100 caps the captured native target graph at position 95
(`GGUF_SPECDEC2_NATIVE_TARGET_GRAPH_MAX_CONTEXT`), so a 512-token root takes
the eager multi-row verifier rather than a serial route. The residual KL is the
AR route's rows==1 local32 owner against the verifier's strict per-row GEMV,
which is exactly what Phase 4 replaces. Evidence:
`benchmarks/results/2026-09-12-ud-gfx1100-phase3-ar-verify-numerics.json`,
`worklog/entries/20260912T092407.863946Z-lhl-ud-phase3-ar-verify-gate-ffc232.md`.

### Phase 4: Repair the dominant verifier path

Use the Phase 1 attribution to choose one scoped change at a time. Every
candidate must have a strict registered fallback and a RED test before
implementation.

Likely investigation order:

- [x] Multi-row target verifier launch and row packing. Done 2026-09-12: the
  local32 IQ decode family gained a `ROWS` template parameter and a rows 2-4
  verifier sibling. Each row is bit-identical to the rows == 1 owner's output
  for that row, so the sibling adds no arithmetic of its own. At kernel level
  it is 2.0-4.3x the strict per-row GEMV on the same real tensors and 1.4-2.9x
  the rows == 1 owner launched once per row, covering 94.1% (K_M) / 85.9%
  (K_S) of the dead-zone MACs. The block verifier also had to bind the
  dense-IQ execution-owner session: without it every raw-IQ verifier
  projection kept the strict GEMV. End to end, UD MTP B3 rose 35.54 -> 43.83
  (K_M) and 33.06 -> 44.46 (K_S) tok/s on the same host and protocol, and both
  UD arms are now generated-ID exact. See
  `benchmarks/results/paired-ud-plain-mtp-c1-natural25-b3-phase4.json` and
  `benchmarks/results/2026-09-12-ud-gfx1100-phase4-*.json`.
- [x] UD Q5 selected-expert path. Closed 2026-09-12 on the post-rows
  attribution: `qk_t16_selected_direct_gemv` is 0.367 ms/step (0.78%) on
  UD-Q4_K_M and 0.662 (1.46%) on UD-Q4_K_S, below the repair threshold at
  verifier rows.
- [ ] Q3_K strict decode. Q3_K has no rows==1 local32 owner, so it has no
  rows 2-4 sibling either; it keeps the strict per-row GEMV. Blocked, not
  merely unattempted: routing it was measured on gfx1151 at +176.5 tok/s but
  moved the calibrated mean KL 0.000827 -> 0.001061, 6% over the 1e-3 limit
  (`GGUF_IQ_DENSE_VERIFY_POLICY` comment in
  `hipengine/kernels/hip_gfx1100/__init__.py`).
- [x] Q5 gate/up dual execution. Closed 2026-09-12: the Q5_K gate/up dual
  runs the WMMA prefill owner at verifier rows and the whole pair costs 6.38
  ms/step against the plain Q4_K dual rowtile's 8.85, so UD's arm is already
  the cheaper one measured.
- [x] Q5/Q6 compact residency and raw consumer qualification. Closed
  2026-09-12: Q6_K col8 is 0.77 ms/step in UD against 6.23 in plain Q4_K_M,
  and the Q5_K col8 alternative lost the column-width measurement below.
- [ ] IQ/Q3 decoder vectorization and memory access. Re-scoped 2026-09-12 to
  memory-level parallelism at fixed occupancy. `gguf_iq4_xs_local32_gemv` is
  12.24 ms/step (26.0% of the UD-Q4_K_M verifier) at 46% (`ffn_gate`, N=17408
  K=5120) and 58% (`ffn_down`, N=5120 K=17408) of the 960 GB/s DRAM roofline,
  with 192 VGPRs. Two instruction-level candidates were measured and rejected,
  so the lever is explicit prefetch or a wider row/wave split, which needs a
  hardware-counter unit plus a new bit-exactness contract.
- [x] Norm, SiLU, residual, and logits tail overhead. Closed 2026-09-12:
  1.885 ms/step, 4.00% of the UD-Q4_K_M verifier, below the repair threshold.
- [ ] Graph capture/replay ownership and synchronization. The host residual is
  5.25 ms/step against 1032 kernel calls/step, 5.09 us per launch (the plain
  control is 824 calls/step, 4.39 ms/step and 5.33 us per launch), so it is
  launch overhead rather than a synchronization stall. Reducing it means
  fewer, wider launches or replay-side launch elision.
- [ ] The t16 decode-versus-rowtile accumulation order (Q4_K 27.4% + Q5_K
  25.7% + Q6_K 8.1% of rank-2 MACs) is the dominant remaining source of
  verification-specific drift: the rows 2-4 t16 rowtile owners are not
  bit-identical to the rows 1 t16 decode owners, so the AR route and the
  verifier still disagree on those tensors after the IQ family is aligned.
  Split-K, which is the measured next lever for the Q5_K rowtile, changes this
  order again and therefore owes the full production profile gate.

Two further scoped candidates were measured and rejected on 2026-09-12 and are
recorded so they are not retried:

- **Q5_K rows 2-8 rowtile column width 4 -> 8.** `TILE_COLS=4` launches four
  blocks per 16-column tile and each reads the whole tile, so the family
  re-reads its weights 4x; `TILE_COLS=8` halves that and is already registered
  under `gguf_q5_k_t16_v1 / t16_gemv_rowtile_col8_bf16_bf16_out`. Measured:
  the Q5_K rowtile went 9.23 -> 10.61 ms/step (85.5 -> 98.2 us/call, VGPR 56
  -> 88) and the arm went 47.09 -> 47.90. Rejected: the re-read is served by
  L2, and col4 wins because it launches twice as many blocks.
- **local32 activation load vectorization.** One 16-byte load per lane window
  instead of eight scalar BF16 loads, in all three local32 owners. Measured on
  the four real IQ4_XS tensors at rows 4: the rows 2-4 sibling moved -1.9% and
  the rows == 1 owner +1.5%, both inside run-to-run spread, with bit-exactness
  preserved. Rejected as neutral: the compiler already coalesces the loads, so
  the family is not load-issue-bound.

For each candidate that is carried as far as an implementation:

- [x] Record the exact kernel/dispatch registry key and source lineage.
- [x] Add or update the focused RED test before implementation.
- [x] Preserve raw-pointer ABI and four-axis registry dispatch.
- [x] Keep the strict unfused or strict decode fallback registered.
- [x] Run CPU-reference and applicable production numerical gates.
- [x] Run a kernel trace proving the intended kernel actually ran.
- [x] Run the full paired benchmark before calling it a win.
- [x] Report absolute rates and all three ratios.

A candidate may be closed without an implementation when the measured cost is
below the repair threshold, when the alternative owner is already the cheaper
one measured, or when a registered policy records a concrete blocker. It is
closed by recording the measurement and the registry key, as the rejected and
closed entries above do.

Promotion rule: retain only changes that are correct, reproducible, and
non-regressive on the declared suite. A ratio gain with lower UD tok/s is
diagnostic unless the absolute result is still an explicitly accepted tradeoff.

### Phase 5: Close lifecycle and serving coverage

Before automatic MTP admission, complete the U6 envelope:

- [ ] Caller ABI rows 1/2/3/4/5/7/8 and prefill tile/chunk boundaries. The
  native verifier cycle now runs the whole declared two-to-eight row envelope,
  so rows 2-8 are covered and rows 5 is numerically qualified. Rows 1 and 9+
  remain outside the graph, and the prefill tile/chunk boundaries are unstarted.
- [ ] c1/c2/c4/c8, ragged and sparse rows, permutations, delayed arrivals,
  neighbor replacement, cancellation, reclaim, and width transitions.
- [ ] Exact speculative accept/reject commit and rollback. The U6 control item
  `exact_ar_mtp_control_behavior` is qualified on both UD records; the
  production rollback path at width is unmeasured.
- [ ] Draft/verifier state disjointness, aliases, and teardown.
- [ ] Block-64 Q6 `eh_proj`, attention, and FFN operations.
- [ ] Explicit artifact-scoped strict manifest.
- [ ] Explicit backend/profile/context/width scope.

The pin remains empty until all declared certification items are complete.

**Result (2026-09-12, physical GPU1 / RX 7900 XTX / gfx1100).** Two units are
closed.

U6 item 5 is qualified on both UD records, so the unit's only open item is the
declared backend/profile/context/width scope. The evidence is the
teacher-forced section-6.1 gate: all four arms pass with top-1 1.0000 in every
category and at every budget, worst row-mean KL 4.23e-05 against the 1e-03
envelope, and a fresh-process reproduction matches all sixteen arm numbers to
the last digit. Evidence:
`benchmarks/results/2026-09-12-ud-gfx1100-phase5-ar-verify-numerics.json`,
`worklog/entries/20260912T143946.885296Z-lhl-ud-phase5-u6-item5-0988ce.md`.

The rows 5-8 verifier cap is lifted. The cap was one wrapper guard on the fused
rounded add/RMSNorm, whose device kernel is one block per row with an identical
reduction tree; the device accept/commit path was never row-capped, so the
earlier attribution that named it was wrong. Rows 5 is the only newly reachable
production shape (budget 4, since `_GGUF_MTP_CANDIDATE_BUDGETS` is 1-4) and it
passes the same gate at top-1 1.0000, but it is not economical: above four rows
the IQ family leaves the rows 2-4 local32 verifier sibling and reverts to
`gguf_iq_dense_strict`, which then owns 59.5-61.8% of verifier kernel time, so
rows 5 costs 18.26 ms/row against 12.79 at rows 4. Extending the sibling's
policy to rows 5-8 is the follow-up that makes budget 4 worth using. Evidence:
`benchmarks/results/2026-09-12-ud-gfx1100-verifier-rows-4-8-census.json`,
`benchmarks/results/2026-09-12-ud-gfx1100-rows5-budget4-gate.json`,
`worklog/entries/20260912T150309.673218Z-lhl-ud-phase5-rows-envelope-8351b3.md`.

Still open: the width scope is c1 only, and that is structural rather than a
missing run. Above resident capacity 1 the C1 singleton MTP route is refused
(`physical_singleton=False`) and the physical route needs a
`SpeculativeMTPServingEvidence` row for the artifact with
`realized_group_rows > 1`; the evidence tables cover only
`weight_quant="gguf_q4_k_m"`, so no UD artifact has one. The unblock is an
ordering change, not a kernel change: grant UD serving evidence in process for
a candidate run, measure c2/c4/c8, then write the rows. MTP serving is also
capped at 1023 context, and that cap is real rather than bookkeeping: above
`start_position + rows >= 1024` the verifier's `strict_long_rows` term
suppresses the batched exact rows chain and the layer falls through to a
per-row loop that is bit-identical to the single-row AR route. The 4096-token
point was measured and returned zero KL over 162 rows because there is no
multi-row verifier above that line, so it is evidence about the retained RF1
strict per-row fallback rather than a long-context pass. That fallback is
deliberate: the batched staging above 1024 crossed a BF16 rounding boundary
versus scalar AR (layer 46 row 3, max abs 0.015625). Long-context MTP
acceleration is therefore blocked on RF2 context-bucketed graphs, whose
candidate measured 0.9989x and is not auto-routed. Rows 9+ are outside the
native target graph, and the prefill tile/chunk boundaries are unstarted.
Evidence:
`worklog/entries/20260912T172746.747765Z-lhl-ud-phase5-width-scope-14e699.md`,
`worklog/entries/20260912T212716.391678Z-lhl-ud-mtp-long-context-verifier-126dac.md`.

### Phase 6: Re-run the complete paired economics gate

Run the final paired protocol for each tier and candidate:

- [ ] True no-MTP UD AR graph-replay baseline.
- [ ] UD MTP budgets including B3 and the declared positive budgets.
- [ ] Plain AR and plain MTP controls under the same protocol.
- [ ] Full four-category prompt suite.
- [ ] Category-heldouts.
- [ ] Required context and width points.
- [ ] Two or more deterministic fresh-process repeats.
- [ ] GPU/CPU acceptance and lifecycle accounting.

Required result fields:

| Field | Meaning |
| --- | --- |
| `ud_ar_toks` | Absolute UD no-MTP AR throughput |
| `ud_mtp_toks` | Absolute UD MTP throughput |
| `plain_ar_toks` | Absolute plain no-MTP AR throughput |
| `plain_mtp_toks` | Absolute plain MTP throughput |
| `ud_mtp_over_ud_ar` | Primary economic ratio; must exceed `1.0x` for promotion |
| `ud_ar_over_plain_ar` | AR parity |
| `ud_mtp_over_plain_mtp` | MTP parity |
| `ud_mtp_gain_abs` | `ud_mtp_toks - ud_ar_toks` |
| `ud_plain_mtp_gap_abs` | `ud_mtp_toks - plain_mtp_toks` |

### Phase 7: Promote or hold

Promote a tier only when:

- [ ] `UD MTP / UD AR > 1.0x` on the complete declared suite.
- [ ] The production teacher-forced gate passes.
- [ ] Control-plane acceptance and deterministic repeat gates pass.
- [ ] Width, context, lifecycle, and heldout coverage are complete.
- [ ] The result is not prompt-conditioned or candidate-conditioned.
- [ ] The artifact, worklog, benchmark README, and changelog are updated.
- [ ] The certification record is complete and the derived U6 pin is populated.

Hold a tier when it is correct but fails economics, has incomplete scope, or
has unresolved numerical/lifecycle gates. A held tier remains available only
within its already-qualified AR scope; MTP is not implicitly enabled.

## 5. Optimization Scorecard

Use this table for every retained candidate. Add absolute values before ratios.

| Candidate | Tier | Scope | UD AR | UD MTP | Plain AR | Plain MTP | UD MTP/AR | UD/plain AR | UD/plain MTP | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | K_M | GPU1, c1, B3 | 31.993 | 35.541 | 37.186 | 63.465 | 1.1109x | 0.860x | 0.560x | Open |
| Baseline | K_S | GPU1, c1, B3 | 31.311 | 33.061 | 39.565 | 64.094 | 1.0559x | 0.791x | 0.516x | Open |

Interpret results in this order:

1. Is the candidate correct?
2. Did absolute UD AR improve?
3. Did absolute UD MTP improve?
4. Did `UD MTP / UD AR` remain above `1.0x` and improve?
5. Did UD/plain AR and MTP parity improve?
6. Did any category, width, context, or lifecycle slice regress?

## 6. Handoff Rules

- The coder works one logical unit at a time and commits after validation.
- No broad refactor is bundled with a kernel optimization.
- Any new environment flag goes into `docs/REFACTOR.md` with its removal
  condition.
- Any blocker is recorded with the failing scope and evidence, not a generic
  “needs more testing” note.
- If a candidate changes arithmetic, use the production-profile gates; do not
  substitute generated-ID equality for the teacher-forced numerical gate.
- If a candidate changes ownership, batching, or lifecycle, test isolation and
  rollback independently of arithmetic.
- If a candidate is neutral or negative, re-audit the kernel-family trace
  before further tuning.

## 7. Completion Definition

This plan is complete for a tier when the tier has:

1. A valid full-suite paired artifact with absolute rates and ratios.
2. `UD MTP / UD AR > 1.0x` across its declared production scope.
3. Passing section 6.1 numerical, deterministic, control, and lifecycle gates.
4. Complete width/context/heldout evidence.
5. A populated artifact-scoped U6 certification and derived admission pin.
6. Synchronized benchmark rollup, changelog, result JSON, worklog entry, and
   commit.

Until then, UD MTP remains a measured diagnostic path, not an admitted
production capability.
