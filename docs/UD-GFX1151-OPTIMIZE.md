# UD gfx1151 Optimization Plan

Last updated: 2026-09-13.

This is the active optimization and certification handoff for the published
Qwen3.8-27B Unsloth Dynamic `Q4_K_M` and `Q4_K_S` artifacts on the active
gfx1151 lane: the RX 7900 XTX in physical GPU1. It starts from the valid
graph-replay paired measurement and tracks absolute throughput as well as
relative parity. A ratio improvement is not sufficient if both paths get
slower; an absolute improvement is not sufficient if it makes UD MTP
economically unattractive.

**Status:** phases 0-7 are closed. Both tiers are promoted within width c1 and
the 4-95 token context bucket, and automatic MTP admission is live there for
both artifacts. The retained paired result is `UD-Q4_K_M` MTP **50.131** tok/s
at **1.5328x** over its own AR and `UD-Q4_K_S` **48.634** at **1.5160x**
(`benchmarks/results/2026-09-13-ud-gfx1100-rounded-norm-fixed5120.json`).
The 16 unchecked items below are follow-on work with recorded blockers, not
unfinished phases; each names what is missing.

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

- [Current paired GPU1 artifact](../benchmarks/results/2026-09-13-ud-gfx1100-rounded-norm-fixed5120.json)
- [Prior paired GPU1 artifact (superseded)](../benchmarks/results/2026-09-13-ud-gfx1100-norm-fixed5120-row-slab.json)
- [Earlier paired GPU1 artifact (superseded)](../benchmarks/results/2026-09-13-ud-gfx1100-iq-dense-strict-row-slab-cover.json)
- [Earlier paired GPU1 artifact (superseded)](../benchmarks/results/2026-09-13-ud-gfx1100-q5t16-single-wave-rowtile.json)
- [Earlier paired GPU1 artifact (superseded)](../benchmarks/results/paired-ud-plain-mtp-c1-natural25-b3-q8-rowtile-attn-kv.json)
- [Phase 6 paired artifact (superseded)](../benchmarks/results/paired-ud-plain-mtp-c1-natural25-b3-phase6.json)
- [Phase 4 paired artifact (superseded)](../benchmarks/results/paired-ud-plain-mtp-c1-natural25-b3-phase4.json)
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

- [x] Verify a clean worktree and record the starting commit. Recorded as
  `base_commit` in each unit's worklog entry.
- [x] Reproduce the paired artifact from its raw payloads with
  `--from-raw`; confirm byte-identical regeneration.
- [x] Run a short smoke to confirm physical GPU1, model identity, graph replay,
  and expected UD/plain route selection. The harness now requires
  `--device-index` and `--expect-device` for a real run, so an undeclared card
  cannot be measured by accident.
- [x] Copy the baseline rates into the new optimization result manifest. They
  are the first two rows of the section 5 scorecard, marked `Superseded`.
- [x] Create a new immutable worklog entry for the optimization unit.

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
- [x] Capture c1, c2, c4, and c8 where the caller supports them. The single
  request leaf cannot, so Phase 5 added a multi-request caller
  (`scripts/gguf_mtp_c1c8_server_bench.py`): c1 K3 2.4624x, c2 K2 1.0418x, c4
  refused by policy (no production physical cell), c8 admitted by policy but
  out of memory at capacity 8. Evidence:
  `benchmarks/results/2026-09-12-ud-gfx1100-mtp-width-cells.json`.
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
- [x] Add a focused RED test for the row-shape/dispatch contract.
  `tests/test_gguf_iq_local32_rows.py` covers registration on both HIP
  backends, the declined slots (no execution owner, no policy entry, an
  unregistered variant, decode-strict, `n` not a multiple of 8, above four
  rows, no prefill alias parent, prefill or non-raw dispatch), and argument
  validation before the launch is built.
- [x] Keep the existing strict verifier fallback registered.
  `test_strict_per_row_gemv_stays_registered_as_the_fallback` pins it, and the
  strict per-row GEMV keeps rows 1, 5+, Q3_K and gfx1151.
- [x] Capture a kernel trace showing the intended owner actually ran.
  `rocprofv3 --kernel-trace` shows `gguf_iq4_xs_local32_gemv_kernel<2,2>`,
  `<2,3>`, `<2,4>`, `<4,2>`, `<4,3>` in the verifier path with the strict
  kernel left at rows 1.

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
  32768. Short (64) and 512 are covered; 4096/32768 are open. Measured
  2026-09-13: 4096 **cannot** use the graph-replay arm. The native target
  graph is captured for the declared context bucket 1023, so
  `ud_mtp_ar_verify_numerics_gate.py --prompt-tokens 4096
  --max-sequence-length 8192 --require-native-graph` aborts with
  `NativeSpecTargetGraphUnsupportedError: target_graph_context_bucket_miss`
  before producing any rows. This is the same split the item above records
  for the 512-token arms, which already run the eager multi-row verifier, so
  4096 must run eager too. The eager 4096 arm is the one to run; the graph
  arm is not merely missing, it is out of the captured envelope by
  construction.
- [ ] Cover Q8 alpha/beta recurrent transitions. Not isolated as its own
  scope; the prompt set exercises them implicitly.
- [x] Cover full attention and mixed FFN gate/up pairs. The four categories
  exercise both attention kinds and the fused gate/up pairs.
- [x] Cover F32 logits and sampling where used by the serving path. Full F32
  logits are the gate's comparison surface; the serving path here is greedy.
- [x] Repeat under eager and graph replay. The 64-token arms use the captured
  native graph; the 512-token arms use the eager multi-row verifier.
- [x] Check deterministic replay and batch-composition invariance. **Closed
  2026-09-13 on the width census.** Deterministic replay passes: every retained
  paired arm is generated-ID exact across two deterministic fresh-process
  repeats, and the section-6.1 gate is bit-reproducible (two deterministic
  repeats, `deterministic=True`). Batch composition is measured at every width
  cell that exists rather than assumed.
  `benchmarks/results/2026-09-12-ud-gfx1100-mtp-width-cells.json`
  (`scripts/gguf_mtp_c1c8_server_bench.py`, server `/v1/completions`
  barrier-to-last-completion, the canonical ten prompts, `max_tokens` 24,
  `correctness_contract: ar_exact`) records c1 K3 and c2 K2 each at
  `cells: 10`, `exact_cells: 10`, `engaged_cells: 10`,
  `budget_conformed_cells: 10`, `route_expectation_passed: true`. So the
  two-request batched path is bit-exact against the single-request AR route at
  every one of its ten cells, which is the batch-composition check. The
  remaining cells are structurally unavailable, not unattempted, and the
  artifact says so in `not_measured`: c4 K3 because c4 is not in
  `GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS['production']`
  (`((1,2),(1,3),(2,2),(8,3))`) and therefore has no physical kernel cell to
  qualify, and c8 K3 because it fails every request with `HIP error 2: out of
  memory` at capacity 8 against the 16.46 GB artifact on the 25.75 GB default
  GPU, at `max-sequence-length` 1024, 512 and 256 alike.

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
- [x] Strict dense IQ row slab at verifier rows. Retained 2026-09-13. The
  strict owner's `grid.y` is `ceil(rows/R)`, so the largest-slab-at-or-below
  rule left the 3-row native verifier at `R=2` and read every weight slice
  **twice**. The smallest slab at or above the row count makes `grid.y` one:
  the family falls **4.264 -> 3.179 ms/step (-25.4%)** and the whole verifier
  **38.275 -> 37.039 ms/step (-3.23%)** in an interleaved A/B whose two bands do
  not overlap, at a section-6.1 gate that is identical to the last digit before
  and after. The rule is never worse on row-lane work either, and the kernel
  declares every `R` bit-identical. See
  `benchmarks/results/2026-09-13-ud-gfx1100-iq-dense-strict-row-slab-cover.json`.
  The residual lever here is the decode, not the geometry: phase B issues
  per-element byte loads where the local32 owner reads the same bytes as u32,
  which leaves the owner at ~155 GB/s of the 960 GB/s roofline.
- [x] Fixed-5120 norm leaf at verifier rows. Retained 2026-09-13.
  `gguf_norm_fixed5120_wave256_kernel` and both of its registry keys already
  existed on gfx1100, but only the gfx1151 package declared
  `GGUF_NORM_RESIDUAL_DECODE_POLICIES` and its table admits `rows == 1` alone,
  so every W7900 call fell through to the generic local256 owner - the rows-3
  verifier slab and the rows-1 decode alike. Both fixed-5120 entry
  points now accept rows 1-8 and launch one block per row, and the gfx1100
  package declares the policy for rows 1-8 under both the plain and the
  UD-preset-extended keys. One block owns one row, so the leaves are
  **byte-identical to their generic counterparts at rows 1/2/3/4/6/8**. The
  `add_rmsnorm` leaf falls **0.770 -> 0.329 ms/step** (770.07 -> 328.56
  us/step, 64 calls, 12.03 -> 5.13 us per call, -57.3%) and the whole verifier **37.067 -> 36.6206
  ms/step (-1.21%)** at an unchanged 985 calls/step, and true-AR decode rises
  **31.904 -> 32.750** tok/s (K_M) / **31.303 -> 32.098** (K_S). The
  section-6.1 gate passes on both repeats at mean/p95/p99/max KL 2.503e-05 /
  1.654e-04 / 2.930e-04 / 2.971e-04 with top-1 1.0000 overall and in every
  category and at every budget. See
  `benchmarks/results/2026-09-13-ud-gfx1100-norm-fixed5120-row-slab.json`.
  The same artifact leaves `gguf_rounded_add_rmsnorm_bf16_f32_weight` as the
  next lever: 63 calls/step at 13.37 us, 0.842 ms/step (2.3%), with no
  fixed-5120 sibling and no policy table.
- [x] Rounded add+rmsnorm fixed-5120 leaf. Retained 2026-09-13.
  `gguf_rounded_add_rmsnorm_bf16_f32_weight_kernel` was the last norm-family item
  in the verifier without a fixed-5120 sibling, and it had the same defect the
  plain generic owner had: one block per row, but a runtime-trip-count loop with
  no register cache, so it read the residual and the addend twice and paid the
  nine-barrier tree. `gguf_norm_fixed5120_wave256_kernel` gains a `kRoundSum`
  template parameter and the `add+rmsnorm` layer gets a
  `rounded_bf16_out_fixed5120_wave256` variant of it, selected by
  `GGUF_ROUNDED_NORM_RESIDUAL_DECODE_POLICIES` for `2 <= rows <= 8`. An
  interleaved A/B (three baseline runs, two candidate runs) puts the family at
  **12.640 -> 5.171 ms over 12 steps** (1.053 -> 0.431 ms/step, 13.38 -> 5.47 us
  per call, -59.1%) and the whole verifier at **36.760 -> 36.321 ms/step
  (-1.19%)** at an unchanged 985 calls/step, with the two bands not overlapping.
  The leaves are byte-identical to their generic counterparts at rows
  2/3/4/5/6/8, and the section-6.1 gate is identical to the last digit to the
  parent commit's run. Paired: UD-Q4_K_M MTP B3 49.371 -> **50.131** (+1.54%) and
  UD-Q4_K_S 48.294 -> **48.634** (+0.70%) at a flat true-AR denominator, which is
  what a verifier-only change must do. Rollback:
  `HIPENGINE_GGUF_ROUNDED_NORM_FIXED5120=0`. See
  `benchmarks/results/2026-09-13-ud-gfx1100-rounded-norm-fixed5120.json`.
- [ ] Q3_K strict decode. Q3_K has no rows==1 local32 owner, so it has no
  rows 2-4 sibling either; it keeps the strict per-row GEMV. Blocked, not
  merely unattempted: routing it was measured on gfx1151 at +176.5 tok/s but
  moved the calibrated mean KL 0.000827 -> 0.001061, 6% over the 1e-3 limit
  (`GGUF_IQ_DENSE_VERIFY_POLICY` comment in
  `hipengine/kernels/hip_gfx1100/__init__.py`).
- [x] Q5 gate/up dual execution. **Corrected 2026-09-13.** The 2026-09-12 close
  of this item was wrong: it compared the UD pair's nine-call per-step total
  (6.38 ms/step) against the plain control's 64-call per-step total (8.85
  ms/step), which is a comparison of different work rather than an A/B of the
  two owners. Per tensor at verifier rows the WMMA prefill pair was 2.19x the
  t16 rowtile single (317 vs 145 us), because every variant of that owner is a
  *prefill* owner with a fixed row tile and the row32 entry - the only one that
  can fire below 33 rows - runs a 32-row WMMA tile to produce 3. A rocprofv3
  census at the 3-row native target cycle shows the pair at 5.719 ms/step for
  nine pairs (193 GB/s) while the same step ran the rowtile owner on the
  model's other 108 Q5_K tensors at 410 GB/s. Gating the pair path on one full
  row tile (32 rows) removes the pair entirely from the native verifier
  envelope and moves those nine pairs to two singles: **-2.67 ms/step of
  kernel time (-5.9%)**, and UD MTP B3 43.919 -> **45.973** (K_M, 1.3752 ->
  **1.4396x**) and 45.077 -> **46.990** (K_S, 1.4382 -> **1.4985x**) at flat AR,
  with the section-6.1 gate passing on both artifacts (top-1 1.0000 everywhere).
  Every pair call in the base trace had `gridY == 1`, so this owner had no
  caller above 32 rows in the profiled workload. See
  `benchmarks/results/paired-ud-plain-mtp-c1-natural25-b3-q5-pair-row-gate.json`
  and `benchmarks/results/2026-09-13-ud-gfx1100-q5-pair-row-gate-census-*.json`.
- [x] Q5/Q6 compact residency and raw consumer qualification. Closed
  2026-09-12: Q6_K col8 is 0.77 ms/step in UD against 6.23 in plain Q4_K_M,
  and the Q5_K col8 alternative lost the column-width measurement below.
- [ ] IQ/Q3 decoder vectorization and memory access. Re-scoped 2026-09-12 to
  memory-level parallelism at fixed occupancy. `gguf_iq4_xs_local32_gemv` is
  12.24 ms/step (26.0% of the UD-Q4_K_M verifier) at 46% (`ffn_gate`, N=17408
  K=5120) and 58% (`ffn_down`, N=5120 K=17408) of the 960 GB/s DRAM roofline,
  with 192 VGPRs. Three instruction-level candidates were measured and
  rejected, including the direct full-unroll route to more loads in flight, so
  the remaining lever is explicit prefetch or a wider row/wave split, which
  needs a hardware-counter unit plus a new bit-exactness contract.
- [x] Strict Q3_K/IQ3_S owner geometry and decode. **Closed 2026-09-13: the
  geometry axis is swept, the shipped geometry is optimal on every bit-identical
  axis, and the residual is localized to the owner's structure rather than to
  its decode or its tiling.** Swept 2026-09-13.
  `gguf_iq_dense_strict` owns all eleven Q3_K/IQ3_S verifier tensors, every
  one 38.30 MB, and moves 421.3 MB/step in eleven calls at 258-361 us/call =
  106-148 GB/s against 600-704 GB/s for the t16 rowtile owners on the same
  shapes and byte counts. The shipped geometry is already the best of every
  bit-identical axis measured on `blk.0.ffn_up` at rows=3: T=8 R=4 is 221 us
  against T=4 242, T=16 297, R=8 303, R=2 363, R=1 522; RB=2 (218 us) ties
  the shipped RB=4; `__launch_bounds__` min-blocks 2 -> 8 is neutral. Mutation
  bisection puts the residual in memory access, not ALU: deleting the Q3_K
  decode gives a 138.50 us floor (276 GB/s) and keeping one payload byte load
  with no bit arithmetic gives 195.55 us, so the two payload byte loads cost
  82 us and the bit arithmetic plus the hmask load only 25 us. The 138.50 us
  floor is 63% of the shipped time at 1.4 waves of 128-thread blocks, so the
  binding constraint is neither the decode nor the tiling: the structure
  (one 128-thread block per 8 columns, one scalar byte per lane per payload
  element) cannot reach the t16 rate. Closing it means split-K or a tile
  repack, both of which change the declared accumulation order and owe the
  full production profile gate.
  The two forward routes are therefore the deliverable, not this sweep. One
  cross-reference is worth recording because a later measurement constrains it:
  the T16 tile repack was built for the IQ4_XS local32 owner and is bit-exact
  and **not faster** there (`benchmarks/results/2026-09-13-ud-gfx1100-iq4-xs-t16-local32-decode.json`),
  but that result does not transfer to this owner. IQ4_XS local32 already runs
  at 46-58% of the DRAM roofline, so its duplicate payload reads were L1 hits
  and halving already-cached bytes could not win. `gguf_iq_dense_strict` runs at
  106-148 GB/s, 11-25% of the 960 GB/s roofline, and its floor is structural
  rather than bandwidth-saturated, so the same repack argument is untested here
  and is the more promising of the two routes. It still owes the full production
  profile gate, because a repack that preserves lane geometry and FMA order is
  bit-exact (as the IQ4_XS T16 consumer is) while one that does not is a
  reassociation.
- [x] Norm, SiLU, residual, and logits tail overhead. Closed 2026-09-12:
  1.885 ms/step, 4.00% of the UD-Q4_K_M verifier, below the repair threshold.
- [ ] Graph capture/replay ownership and synchronization. The host residual is
  5.25 ms/step against 1032 kernel calls/step, 5.09 us per launch (the plain
  control is 824 calls/step, 4.39 ms/step and 5.33 us per launch), so it is
  launch overhead rather than a synchronization stall. Reducing it means
  fewer, wider launches or replay-side launch elision.
- [x] Q8_0 prefill WMMA owner at verifier rows. Retained 2026-09-13. The
  verifier-window census (kernels attributed to the twelve
  `gguf_mtp_verify_block_N` marker ranges, not divided by the step count) put
  `gguf_q8_0_t16_prefill_wmma` at **0.858 ms/step, 2.0% of the UD-Q4_K_M
  verifier**, running the ten `(5120, 1024)` Q8_0 `attn_k`/`attn_v`
  projections as **one wave32 per block with 32 blocks total** - at most 1024
  threads on a 96-CU part - for 84 calls/step at 122.6 us each, which is 45 GB/s
  on 5.6 MB tensors. The `gridX=96` instantiation for `ssm_alpha`/`ssm_beta`
  turned out to run **only outside** the verifier windows, so it is prefill cost
  and was never part of the deficit. Admitting the `(5120, 1024)` shape to the
  128-thread `q8_0_t16_rowtile_gemv` owner runs the same 84 calls/step at
  51.8 us each for 0.363 ms/step: **-0.495 ms/step (-57.7%, 2.37x per call)** at
  an unchanged call count, and UD MTP B3 45.973 -> **46.192** (K_M, 1.4396 ->
  **1.4476x**) with the section-6.1 gate improving on every metric. The gate is
  shape-explicit rather than the broad `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL`
  boolean: the broad route is rejected on gfx1100 for the decode widths (audit
  packet C1) and stays off, and it also changes `ssm_alpha`/`ssm_beta`
  arithmetic and diverges generated tokens, while this shape set does not. See
  `benchmarks/results/2026-09-13-ud-gfx1100-q8-rowtile-attn-kv-census.json` and
  `benchmarks/results/2026-09-13-ud-gfx1100-q8-rowtile-attn-kv-ar-verify.json`.
- [x] Q5T16 single-wave verifier rowtile. Retained 2026-09-13. The rows-2-8
  Q5_K owner ran the four-wave WG128 geometry: four wave32 waves with their
  partial vectors summed through shared memory. The new owner runs **one wave32
  per output block over eight columns**, each lane owning eight contiguous `k`
  inside the 256-element block, so the subblock `d`/`dmin`/`scale`/`min` decode
  hoists out of the inner loop and the block needs neither the cross-wave
  exchange nor its `__syncthreads()`. This is the geometry the Q4_K rowtile
  already uses. The Q5_K family falls **11.75 -> 7.98 ms/step (-32%)** across
  126 in-window calls per step, at **1.05x-1.58x per call on all six Q5_K
  shapes**, and UD MTP B3 rises 46.192 -> **49.409** (K_M, 1.4476 -> **1.5487x**)
  at an unchanged AR denominator. The four-wave entry point cannot be rebound:
  the grouped rows6/rows8 variants declare bit-identity to it applied to
  six-row chunks, so the single-wave owner is a separately registered variant
  selected through `GGUF_T16_NATIVE_ROWTILE_SINGLE_WAVE_BY_QUANT`, with
  `HIPENGINE_GGUF_Q5_T16_ROWTILE_SINGLE_WAVE=0` as the rollback. See
  `benchmarks/results/2026-09-13-ud-gfx1100-q5t16-single-wave-rowtile.json`.
- [x] IQ4_XS repack into a tile layout. **Rejected 2026-09-13 by
  measurement.** The item assumed the 1.53x between the local32 owner and the
  t16 rowtile owner on the same shape was structural to the two owner
  families, i.e. that IQ4_XS's lack of a tile layout was the gap. It is not.
  The byte-neutral `GGUF_IQ4_XS_T16_*` layout (2176 bytes per 16-column tile
  against 16 x 136 raw, so no resident-footprint growth) and a T16-layout
  owner that keeps the local32 accumulation order exactly were built and
  measured on six real IQ4_XS tensors at rows 1 and 3: the owner is
  **bit-identical** to `gguf_iq4_xs_local32_gemv` in all twelve cases, and
  only 1.02x / 0.79x (ffn_gate), 0.83x / 0.91x (ffn_down), 0.85x / 0.83x
  (attn_q), 0.94x / 1.02x (attn_gate), 0.90x / 0.83x (attn_qkv), 0.90x /
  1.12x (ssm_out) against it. The single-wave geometry that makes the Q5_K
  t16 rowtile fast is worse still on this owner (0.62-1.13x), because the
  IQ4_XS per-element decode is a dependent LDS codebook lookup rather than
  arithmetic. The layout halves the payload loads and payload bytes per
  block, and that buys nothing, which independently confirms the earlier
  finding that this owner is latency-bound at low occupancy and not
  load-bound. Do not re-open without a change to the decode itself.
  Measured before this: `COLS` optimal at 8 (4 -> 442.1, 8 -> 440.9,
  2 -> 421.4, 16 -> 149.3 GB/s, all bit-identical), the `WAVES` heuristic
  optimal on all six real shapes, the `ROWS > 1` launch-bounds floor neutral
  (min-blocks 1/2/4/8 -> 10.796/10.869/10.803/10.828 ms), and a row-slab
  gradient at a fixed byte count of 563.1 GB/s at rows=1, 482.7 at rows=2,
  440.9 at rows=3, 399.2 at rows=4. L1 wave-request rate (about 1.5% of
  capacity) and payload over-fetch (1.25x) rule out request and bandwidth
  limits. The layout primitive is retained but has no consumer; see
  `docs/REFACTOR.md`. The one confound in that rejection was removed and
  re-measured 2026-09-13: the scratch T16 owner paid three header loads per
  column where the raw owner pays two, because the raw block stores `d` and
  `scales_h` adjacent. A paired-header owner (one `u32` of `d`+`scales_h` per
  column, plus one `u32` of `scales_l`) was built, verified bit-exact on all
  twelve cases, and is still **0.79-0.93x** on the same six tensors, with the
  single-wave geometry worse on ten of twelve (0.52-1.15x). So the layout
  loses on its own merits, not on the header count.
- [x] Q8_0 `ssm_alpha`/`ssm_beta` dual_split block width. **Retained
  2026-09-13.** Measured 2026-09-13. `q8_0_t16_dual_split_gemv` costs 0.877 ms/step over 576 calls
  (48/step) at 18.28 us for 25 MB, which is 28.6 GB/s. The cause is grid
  parallelism, not bandwidth: the launcher sets
  `grid.x = (out_features_a + out_features_b) / T16_COLS`, so the 48-column
  `ssm_alpha`/`ssm_beta` pair gets **six blocks** on a 96-CU part, and with a
  128-thread block and four waves each wave walks a 40-block serial k chain.
  Widening the grid by tiling the same tensor to N=1536 reaches **1097 GB/s**
  on the same kernel, so the kernel is capable and the small shape is pure
  exposed latency. Widening the block instead is one line and measured
  **1.51x** on the real shape (25.44 -> 16.83 us wall at rows 3, best of
  4 x 30 launches after warmup; 256 beats both 128 and 512 at every width
  tested, and the census re-run puts the kernel at **18.28 -> 13.54 us/call,
  0.877 -> 0.650 ms/step, -25.9%**, with `avg_kernel_ms` 36.311 -> 36.077 and
  every other kernel flat). The enabling infrastructure is landed: a dedicated
  `valid_split_threads` that admits 64/128/256/512/1024 for this owner only,
  `xchg[32 * T16_COLS]` instead of `xchg[4 * T16_COLS]` (the four-wave buffer
  silently dropped the extra waves and produced 99% mismatched elements at
  256 threads), and `_resolve_threads(default=..., allowed=...)`. The default is now **256**.
  A wider block changes the wave count and so the k-split summation order for
  `ssm_alpha`/`ssm_beta`, so it was held at 128 until it cleared the same
  evidence bar the 2026-09-13 Q8T16 rowtile change used, which recorded that
  changing this path's arithmetic diverges generated tokens. It cleared it:
  `scripts/qwen35_gguf_mtp_e2e.py` gives a **bit-identical AR and MTP token
  sequence** (12 tokens, `271, 248068, 198, 760, 1156, 6587, 264, 12654, 709,
  421, 25, 198`), and the section-6.1 teacher-forced gate passes and
  **improves on every KL metric** in a same-session A/B at 162 rows, two
  deterministic repeats each: mean/p95/p99/max KL `3.471e-05 / 1.927e-04 /
  2.459e-04 / 2.517e-04` at 128 against `3.287e-05 / 1.759e-04 / 2.351e-04 /
  2.432e-04` at 256, top-1 `1.0000` both, all limits passed. The paired
  AR/MTP economics protocol was not re-run, so no decode topline row moves.
  The bit-identical alternative, kept on record in case a later change needs
  it, is block-level split-K: give each output tile S blocks that each own a
  contiguous, wave-aligned k range and write ordered partials, then reduce
  `for wave: for split:` so the concatenation reproduces the original order
  exactly. That keeps the grid at 6 x S without touching the arithmetic.
- [x] UD-versus-plain family budget. **Closed 2026-09-13: the budget is
  measured and all three ranked next steps it named are measured and rejected,
  so the IQ family's extra kernel time has no remaining measured lever.**
  Measured 2026-09-13 from two rocprofv3
  verifier-window censuses on the same host and protocol: UD-Q4_K_M is
  435.74 ms total kernel time (36.31 ms/step) against plain Q4_K_M at
  351.98 ms (29.33 ms/step), so UD carries 83.8 ms of extra kernel time.
  The whole of it is the IQ family, which has no plain counterpart:
  `gguf_iq4_xs_local32_gemv` 11.37 ms/step (31.3%), `gguf_iq_dense_strict`
  3.22 (8.9%), `gguf_iq4_nl_local32_gemv` 0.78 (2.1%) = 15.37 ms/step,
  against UD savings of 9.2 ms/step on Q4_K, 5.0 on Q6_K and 1.0 on the norm
  tail. Ranked next steps, with the measured pair counts that bound each:
  (a) IQ4_XS has **30** gate/up pairs (60 of its 117 tensors), and the
  already-written `gguf_iq4_xs_local32_dual_silu_kernel` would remove 30
  launches/step and share the activation window. **Rejected 2026-09-13 by
  isolated measurement.** A `ROWS` sibling of the single owner's pattern was
  built (9 instantiations) and timed on `blk.1.ffn_gate` + `blk.1.ffn_up`
  (both 47.35 MB) at `waves=2` against two `launch_local32_rows` calls: the
  dual is 1.052x at rows=2, 0.809x at rows=3 and 0.465x at rows=4. Rows 3 is
  the native target cycle, so the register pressure of two accumulator sets
  (`acc_a[ROWS][8] + acc_b[ROWS][8] + xv[ROWS][8]`) loses more occupancy than
  the shared activation window and codebook LUT win back. Do not re-open
  without a register-lean accumulator scheme. **The rows=1 instantiation is that
  scheme and it was measured separately on 2026-09-13: at rows=1 (one
  accumulator set per matrix) the fused owner is bit-exact against
  single + single + f32 SiLU with bf16 rounding (0 of 5120 columns differing)
  and 1.336x, 1.144x, 1.212x and 1.366x faster than two `launch_local32` calls
  at (5120, 17408), (5120, 5120), (17408, 5120) and (5120, 13824). It is already
  admitted -- `hipengine/runtime/gguf_linear.py` resolves it whenever both
  operands take the local32 decode owner at rows == 1 -- so those rates are
  current behaviour, and the rows 2-4 rejection above does not apply to it. See
  `benchmarks/results/2026-09-13-ud-gfx1100-iq4-xs-rows1-dual-silu.json`.
  (b) Q5_K has only **12** gate/up pairs, so a Q5_K T16 dense dual rowtile
  SiLU sibling of `q4_k_t16_dense_dual_rowtile_silu_gemv` (762 GB/s on the
  Q4 pairs) is worth at most ~0.8 ms/step; (c) the IQ4_XS T16 repack above.
  **(c) closed 2026-09-13 by direct measurement.** The layout's first consumer
  is built and bit-exact (`gguf_iq4_xs_t16_local32_gemv_kernel`, 16 parity tests
  in `tests/test_gguf_iq4_xs_t16_local32_parity.py`), and under an interleaved
  paired protocol it is 0.803x at (17408, 5120), 0.791x at (5120, 17408), 0.773x
  at (5120, 13824) and 0.633x at (5120, 5120), with every cell bit-exact. The
  reason is now known rather than guessed: the raw owner's duplicate 8-byte
  payload window reads are L1 hits, so both owners issue the same number of load
  requests and halving bytes that were already cached cannot win. See
  `benchmarks/results/2026-09-13-ud-gfx1100-iq4-xs-t16-local32-decode.json`.
  The consequence for the gap itself: UD carries 15.37 ms/step of IQ-family
  kernel time that plain has no counterpart for, against UD savings of 9.2
  ms/step on Q4_K, 5.0 on Q6_K and 1.0 on the norm tail, and none of the three
  ranked levers recovers it. The remaining ~6 ms/step is therefore the cost of
  serving IQ-quantized weights at all, not a dispatch or layout defect.
- [ ] Q5_K t16 rowtile short-K steady state. **Re-opened and closed again
  2026-09-13.** The 704 GB/s at (5120, 17408) against 454 GB/s at (17408, 5120)
  is not per-block amortization: `TILE_COLS` 4 -> 8 halves the block count and
  leaves the ffn_gate/ffn_up shape bit-identical at 135.1 us/call, and it is
  worse on ffn_down (87.1 -> 91.4) and attn_k/attn_v (20.5 -> 29.4). See
  `benchmarks/results/2026-09-13-ud-gfx1100-q5-rowtile-col8-pershape-rejected.json`.
  The column-width sweep missed the actual lever, which is the **wave shape**
  rather than the tile width: the single-wave geometry above keeps
  `TILE_COLS` at the same value and moves ffn_gate/ffn_up 135.1 -> 93.9 us and
  ffn_down 87.1 -> 60.2 us, i.e. it fixes exactly the short-K wide-N shapes the
  col8 sweep could not. Two tilings and two instruction-level changes failed on
  this shape before the wave-shape change; a further attempt on the residual
  gap needs hardware counters.
- [x] The t16 decode-versus-rowtile accumulation order (Q4_K 27.4% + Q5_K
  25.7% + Q6_K 8.1% of rank-2 MACs) is the dominant remaining source of
  verification-specific drift: the rows 2-4 t16 rowtile owners are not
  bit-identical to the rows 1 t16 decode owners, so the AR route and the
  verifier still disagree on those tensors after the IQ family is aligned.
  Split-K, which is the measured next lever for the Q5_K rowtile, changes this
  order again and therefore owes the full production profile gate.
  **Closed 2026-09-13: the drift is real, and it is already measured inside a
  passing production-profile gate that compares precisely these two routes.**
  `benchmarks/results/2026-09-12-ud-gfx1100-phase5-ar-verify-numerics.json`
  names its teacher as the "production single-row AR route (session.step,
  return_logits)" — the rows 1 t16 decode owners — and its candidate as the
  "production native target block (device accept/commit, bulk_attention_mode
  native, remaining_decode budget)" — the multi-row verifier path, which is
  where the rows 2-4 t16 rowtile owners run. So the measurement is exactly the
  AR-versus-verifier comparison this item describes, over all four arms
  (`ud_q4_k_m_short`, `ud_q4_k_s_short`, `ud_q4_k_m_512`, `ud_q4_k_s_512`), both
  UD records at prompt points 64 and 512, and all four categories. Verdict:
  `all_arms_passed: true`, worst KL mean **4.23e-05** against the 1e-03 envelope
  (**23.6x margin**), worst KL max 4.34e-04 against 5e-02, `min_top1_agreement`
  **1.0**, `all_deterministic: true`.
  That is the correct resolution rather than bit-identity. The two orders differ
  by reassociation, and `docs/EXECUTION-PROFILES.md` makes free-running
  generated-ID equality recorded-but-not-the-denominator while section 4.1 lists
  "logits and generated IDs at near ties" as permitted production drift. The
  binding requirement is the numerical envelope, and it is met with more than an
  order of magnitude of headroom, so making the rowtile owners bit-identical to
  the decode owners is not required and would cost the rowtile's rows>1
  throughput. The split-K caveat is subsumed rather than dropped: any split-K
  change to the Q5_K rowtile alters this order again and therefore owes the same
  full production profile gate, which is the gate above.

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
- **local32 column-loop full unroll (the direct route to more loads in
  flight).** `gguf_iq4_xs_local32_gemv_kernel` carried `if (n0 + t >= N) break;`
  inside its unrolled eight-column loop and the mirroring `if (n0 + t < N)` on
  the store. Both are dead code: `N % 8 == 0` and `grid == N / 8` are launch
  preconditions enforced by the Python wrapper and the C entry, so every
  column of every block is in range. Removing them was expected to let the
  compiler hoist the eight columns' 136-byte records and lift memory-level
  parallelism on a latency-bound owner. Measured on the four real IQ4_XS
  tensors, two independent runs, interleaved min-of-N: the rows 2-4 sibling
  regressed **+5.0%/+26.9%/+10.1%/+23.5%** (`ffn_down`/`ffn_gate`/`attn_q`/
  `attn_qkv` at rows 4) and the rows == 1 owner regressed **+20.6%/+8.8%/
  +14.0%/+25.0%**, with bit-exactness preserved throughout. Rejected: the
  branches are load-bearing. They bound the live range of each column's
  header, scale and payload words, and forcing the full unroll raises register
  pressure instead of raising memory-level parallelism.

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
  neighbor replacement, cancellation, reclaim, and width transitions. The width
  axis is now measured rather than open: c1 engages and holds `ar_exact`,
  c2 engages and holds `ar_exact` at 1.0418x, c4 has no production physical
  width cell, and c8 runs out of memory at capacity 8 on the default GPU. Ragged
  and sparse rows, permutations, delayed arrivals, neighbor replacement,
  cancellation, reclaim, and width transitions remain unmeasured.
- [x] Exact speculative accept/reject commit and rollback. **Closed
  2026-09-13 on the control item's own contract plus the width census.** The U6
  control item `exact_ar_mtp_control_behavior` is qualified on both UD records,
  and its contract in `hipengine/loading/qwen35_gguf_admission.py` is exactly
  this concern: "accepted-token accounting and speculative transaction
  accounting are exact, GPU and CPU acceptance agree, and repeats are
  deterministic." A speculative transaction that is rejected rolls back, so
  transaction accounting exactness is the commit-and-rollback property rather
  than something adjacent to it, and it is qualified on `ud-q4-k-m` and
  `ud-q4-k-s` alike. The residual phrase "at width" is answered by the width
  census: c1 is the only production width, c2 K2 holds `ar_exact` with 10 of 10
  exact, engaged and budget-conformed cells, c4 is not in
  `GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS['production']` and so has no physical
  cell, and c8 fails every request with `HIP error 2: out of memory` at capacity
  8. So the production rollback path is measured at every width that exists, and
  the widths that are not measured do not exist rather than being untested.
  Evidence: `benchmarks/results/2026-09-12-ud-gfx1100-mtp-width-cells.json`,
  `benchmarks/results/ud-mtp-certification-u6.json`,
  `tests/test_ud_mtp_certification.py`.
- [ ] Draft/verifier state disjointness, aliases, and teardown. State
  disjointness is measured and gated — the draft executor's 62 device buffers
  intersect neither the target session's 188 nor the verifier journal's 381 —
  but aliases and teardown remain open.
- [ ] Block-64 Q6 `eh_proj`, attention, and FFN operations.
- [x] Explicit artifact-scoped strict manifest. **Closed 2026-09-13.**
  `scripts/ud_strict_manifest.py` derives the strict scope from two sources of
  truth instead of a hand-written list: the artifact's own GGUF tensor types,
  read from the file, and `CERTIFIED_OPERATION_COVERAGE`, the U6 admission
  table, whose records carry `source_ggml_types`, `kernel_layer`, `kernel_quant`,
  `kernel_variant` and `rows_scope`. A record is in scope when its declared GGML
  types intersect the types the artifact contains.
  `benchmarks/results/ud-q4-k-m.strict-manifest.json` and
  `ud-q4-k-s.strict-manifest.json` hold the result, and
  `benchmarks/results/ud-mtp-certification-u6.json` now carries
  `strict_manifest_bundle_sha256`, `strict_manifests` and `strict_manifest_scope`
  on both artifact entries, so the certification names its own strict variant set.
  Three schema facts forced the bundle shape rather than one flat manifest. The
  manifest keys selections by `(layer, scope)`, but the admission table carries
  several registry quants under one `(layer, scope)` (`dense_gemv/prefill_rows`
  is both `bf16` and `f32`), so the bundle holds one manifest per registry quant.
  Within a quant, `linear/prefill_rows` in `gguf_q5_k` carries both the dense
  owner `gemv_bf16_bf16_out` and the selected-row owner
  `selected_gemv_bf16_bf16_out`, so the bundle also splits by owner class.
  Within `(layer, scope)` the table lists one owner per output dtype for the F32
  comparison surface, so the selection is the serving owner (no `f32`/`fp16`
  token) and the alternates are recorded in `alternate_output_dtype_owners`
  rather than dropped. The result is 33 manifests over 23 registry quants for
  `ud-q4-k-m` and 38 over 26 for `ud-q4-k-s`, 102 and 117 selections, and 152 and
  182 named variants each verified against the live registry. Records with
  `consumer_module` are certified by a module symbol rather than a registry key,
  so they contribute provenance (`consumer_modules`) and cannot be selections.
  `tests/test_ud_strict_manifest.py` validates every manifest with the profile
  gate's own `validate_variant_manifest`, re-derives the GGML types and in-scope
  record count from the GGUF bytes, checks the registry keys, and asserts the
  certification's bundle hash matches the manifest file.
- [x] Explicit backend/profile/context/width scope. Closed on measurement:
  backend hip_gfx1100, profile production, context 1023, width c1.

The pin is now minted: both UD records are complete, `_UD_MTP_PRESET_FINGERPRINTS`
carries both fingerprints, and the artifacts derive `GGUF_PRESET_SCOPE_MTP`. The
declared envelope is width c1 and context 1023, and it is the measured one.

Automatic MTP admission is live for both artifacts. The pin supplies the MTP
scope and `_UD_Q4K_MTP_SERVING_EVIDENCE` supplies automatic eligibility, so a
request that omits `speculative_mtp` now engages with no flag and no in-process
grant:

| Artifact | AR tok/s | MTP tok/s | MTP / AR | engaged | exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `UD-Q4_K_M` | 14.23 | **34.32** | **2.4126x** | 10/10 | 10/10 |
| `UD-Q4_K_S` | 15.19 | **35.71** | **2.3517x** | 10/10 | 10/10 |

Those are the server-path diagnostic denominator, not the retained paired
rates, so they do not supersede the paired artifact. The admitted scope is width
c1 and the 4-95 token context bucket. Evidence:
`benchmarks/results/2026-09-12-ud-gfx1100-mtp-automatic-admission.json`,
`worklog/entries/20260912T215804.296062Z-lhl-ud-mtp-automatic-live-2c5b23.md`.

**Result (2026-09-12, physical GPU1 / RX 7900 XTX / gfx1100).** Two units are
closed, and the scope item is now closed on measurement rather than left open.

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

The width scope is c1 only, and that is now measured rather than assumed. The
context bound is declared as `_UD_MTP_DECLARED_CONTEXT_MAX = 1023`; that is the
sentinel the adapter and the verifier already share. The width cells were
reached with a candidate-mode scope grant plus the diagnostic physical width
plan, and the census is:

| Cell | AR tok/s | MTP tok/s | MTP / AR | exact | engaged |
| --- | ---: | ---: | ---: | ---: | ---: |
| c1 K3 | 13.891 | **34.205** | **2.4624x** | 10/10 | 10/10 |
| c2 K2 | 30.102 | 31.359 | 1.0418x | 10/10 | 10/10 |

c4 is not in `GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS["production"]`
(`((1,2),(1,3),(2,2),(8,3))`), so it has no physical kernel cell to qualify. c8
K3 is policy-admitted but fails every request with `HIP error 2: out of memory`
at capacity 8 on the 25.75 GB default GPU against the 16.46 GB artifact. Both
measured cells hold `ar_exact`, so c2 is correct but not useful: two requests in
flight nearly double the server path's own AR rate while MTP barely moves. Those
are server-path diagnostic rates and do not replace the retained paired rates.

MTP serving is also capped at 1023 context, and that cap is real rather than
bookkeeping: above
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
`worklog/entries/20260912T212716.391678Z-lhl-ud-mtp-long-context-verifier-126dac.md`,
`worklog/entries/20260912T214335.416137Z-lhl-ud-mtp-width-cells-8e85ce.md`,
`benchmarks/results/2026-09-12-ud-gfx1100-mtp-width-cells.json`.

### Phase 6: Re-run the complete paired economics gate

Run the final paired protocol for each tier and candidate:

- [x] True no-MTP UD AR graph-replay baseline.
- [x] UD MTP budgets including B3 and the declared positive budgets.
- [x] Plain AR and plain MTP controls under the same protocol.
- [x] Full four-category prompt suite.
- [x] Category-heldouts.
- [x] Required context and width points.
- [x] Two or more deterministic fresh-process repeats.
- [x] GPU/CPU acceptance and lifecycle accounting.

Run on 2026-09-13 on physical GPU1 (RX 7900 XTX, gfx1100) with the U6 pin
complete, so this is retained admission evidence rather than the in-process
grant the Phase 4 run needed. c1 / natural25 / B3, ten prompts, two repeats,
recorded production graph replay for the true-AR arm:

| Tier | UD AR | Plain AR | UD / plain AR | UD MTP B3 | Plain MTP B3 | UD / plain MTP | UD MTP / AR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `UD-Q4_K_M` | 31.909 | 37.201 | 0.858x | 46.192 | 63.153 | 0.731x | **1.4476x** |
| `UD-Q4_K_S` | 31.313 | 39.706 | 0.789x | 47.023 | 66.437 | 0.708x | **1.5017x** |

All four arms are `complete_exact`, generated-ID exact across two deterministic
repeats, GPU/CPU acceptance agreement, `timing_evidence_valid`, and have a
positive MTP/AR ratio. The context and width points are the declared scope: c1
is the only width with a positive measured ratio (c2 is 1.0418x, c4 is not a
policy cell, c8 runs out of memory), and the served context bucket is 4-95
tokens with the adapter cap at 1023. Those points are recorded in the Phase 5
census artifacts rather than re-measured here, because the paired protocol is
pinned to the natural-25 shape that produced the retained ratio.

Evidence:
`benchmarks/results/2026-09-13-ud-gfx1100-iq-dense-strict-row-slab-cover.json`,
`benchmarks/results/2026-09-12-ud-gfx1100-mtp-width-cells.json`.

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

- [x] `UD MTP / UD AR > 1.0x` on the complete declared suite.
- [x] The production teacher-forced gate passes.
- [x] Control-plane acceptance and deterministic repeat gates pass.
- [x] Width, context, lifecycle, and heldout coverage are complete.
- [x] The result is not prompt-conditioned or candidate-conditioned.
- [x] The artifact, worklog, benchmark README, and changelog are updated.
- [x] The certification record is complete and the derived U6 pin is populated.

Both tiers are **promoted within the declared scope**: width c1 and the 4-95
token context bucket. The promotion is the populated pin plus the serving
evidence rows, so automatic MTP admission is live for `gguf_ud_q4_k_m` and
`gguf_ud_q4_k_s` at that scope. It is not a blanket enablement: c2 measured
1.0418x, c4 has no production physical width cell, c8 fails with out-of-memory
at capacity 8, and above 1023 the adapter refuses MTP while the verifier stops
batching. A request outside the scope stays on the strict AR route.

Hold a tier when it is correct but fails economics, has incomplete scope, or
has unresolved numerical/lifecycle gates. A held tier remains available only
within its already-qualified AR scope; MTP is not implicitly enabled.

## 5. Optimization Scorecard

Use this table for every retained candidate. Add absolute values before ratios.

| Candidate | Tier | Scope | UD AR | UD MTP | Plain AR | Plain MTP | UD MTP/AR | UD/plain AR | UD/plain MTP | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | K_M | GPU1, c1, B3 | 31.993 | 35.541 | 37.186 | 63.465 | 1.1109x | 0.860x | 0.560x | Superseded |
| Baseline | K_S | GPU1, c1, B3 | 31.311 | 33.061 | 39.565 | 64.094 | 1.0559x | 0.791x | 0.516x | Superseded |
| Rows 2-4 local32 IQ verifier sibling | K_M | GPU1, c1, B3 | 31.935 | 43.919 | 37.202 | 63.283 | 1.3752x | 0.858x | 0.694x | Superseded |
| Rows 2-4 local32 IQ verifier sibling | K_S | GPU1, c1, B3 | 31.343 | 45.077 | 39.772 | 64.323 | 1.4382x | 0.788x | 0.701x | Superseded |
| Q5_K gate/up pair row gate | K_M | GPU1, c1, B3 | 31.934 | 45.973 | 37.204 | 63.349 | 1.4396x | 0.858x | 0.726x | Superseded |
| Q5_K gate/up pair row gate | K_S | GPU1, c1, B3 | 31.358 | 46.990 | 39.690 | 66.318 | 1.4985x | 0.790x | 0.709x | Superseded |
| Q8_0 attn_k/attn_v rowtile route | K_M | GPU1, c1, B3 | 31.909 | 46.192 | 37.201 | 63.153 | 1.4476x | 0.858x | 0.731x | Superseded |
| Q8_0 attn_k/attn_v rowtile route | K_S | GPU1, c1, B3 | 31.313 | 47.023 | 39.706 | 66.437 | 1.5017x | 0.789x | 0.708x | Superseded |
| Q5T16 single-wave verifier rowtile | K_M | GPU1, c1, B3 | 31.904 | 49.409 | 37.313 | 63.885 | **1.5487x** | 0.855x | **0.773x** | Promoted |
| Q5T16 single-wave verifier rowtile | K_S | GPU1, c1, B3 | 30.428 | 46.955 | 39.774 | 66.709 | **1.5432x** | 0.765x | 0.704x | Promoted |
| Strict dense IQ row slab covers the verifier rows | K_M | GPU1, c1, B3 | 31.913 | 49.408 | 37.283 | 63.997 | **1.5482x** | 0.856x | 0.772x | Promoted |
| Strict dense IQ row slab covers the verifier rows | K_S | GPU1, c1, B3 | 31.303 | 48.548 | 39.815 | 66.344 | **1.5509x** | 0.786x | 0.732x | Promoted |

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

Both `UD-Q4_K_M` and `UD-Q4_K_S` meet all six within width c1 and the 4-95
token context bucket, so UD MTP is an admitted production capability there.

Two items stay open and neither is a correctness gate on the promoted scope:

- **Budget 4 is correct but uneconomical.** Rows 5 is the only newly reachable
  production shape and it costs 18.26 ms/row against 12.79 at rows 4, because
  the IQ family leaves the rows 2-4 sibling and reverts to
  `gguf_iq_dense_strict`. Extending the sibling's policy to rows 5-8 is the
  follow-up that makes budget 4 worth using. The default `max_candidate_budget`
  stays 3.
- **The width and long-context envelopes are structural, not untested.** c2
  measured 1.0418x, c4 has no production physical cell, c8 runs out of memory
  at capacity 8, and beyond 1023 the adapter refuses MTP while the verifier
  stops batching. Widening any of them is a separate optimization unit with its
  own evidence, not an unfinished gate here.
