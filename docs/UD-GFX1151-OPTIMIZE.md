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

- [ ] Capture c1, c2, c4, and c8 where the caller supports them.
- [ ] Capture representative verifier rows such as 6, 9, 12, 16, 28, and 32.
- [ ] Record kernel names, launch counts, duration, stream, grid, and block.
- [ ] Attribute wall time to decoder, projection, norm, gate/up, down, logits,
  synchronization, and host/device transfer categories.
- [ ] Compare row/tile decomposition between UD and plain.
- [ ] Check whether UD uses BF16 expansion where plain uses compressed
  consumers.
- [ ] Check Q5/Q6/IQ/Q3 decoder occupancy, memory traffic, and launch count.
- [ ] Confirm whether the deficit is steady verifier work or transition-only
  overhead.

Decision gate:

- If verifier device work dominates, optimize the responsible kernel family.
- If host or synchronization dominates, optimize submission/graph ownership
  before changing arithmetic.
- If expansion/residency dominates, qualify compact residents and consumers.
- If only a narrow shape loses, keep the optimization shape-scoped.

No kernel change starts before this attribution is recorded.

### Phase 2: Establish the teacher-forced numerical gate

Complete section 6.1 for the single-row AR route versus multi-row target
verification. This separates arithmetic drift from control and batching bugs.

- [ ] Measure mean, p95, p99, and maximum row KL.
- [ ] Measure per-category top-1 agreement.
- [ ] Cover canonical prompts and category-heldouts.
- [ ] Cover c2/c4/c8 and the relevant verifier row shapes.
- [ ] Cover short, 512, 4096, and a separately budgeted long context such as
  32768.
- [ ] Cover Q8 alpha/beta recurrent transitions.
- [ ] Cover full attention and mixed FFN gate/up pairs.
- [ ] Cover F32 logits and sampling where used by the serving path.
- [ ] Repeat under eager and graph replay.
- [ ] Check deterministic replay and batch-composition invariance.

The gate must identify whether K_S near-ties are permitted production drift
or a binding batch-composition failure. Do not widen the admission pin to make
the result pass.

### Phase 3: Repair the dominant verifier path

Use the Phase 1 attribution to choose one scoped change at a time. Every
candidate must have a strict registered fallback and a RED test before
implementation.

Likely investigation order:

- [ ] Multi-row target verifier launch and row packing.
- [ ] UD Q5 selected-expert path.
- [ ] Q3_K strict decode.
- [ ] Q5 gate/up dual execution.
- [ ] Q5/Q6 compact residency and raw consumer qualification.
- [ ] IQ/Q3 decoder vectorization and memory access.
- [ ] Norm, SiLU, residual, and logits tail overhead.
- [ ] Graph capture/replay ownership and synchronization.

For each candidate:

- [ ] Record the exact kernel/dispatch registry key and source lineage.
- [ ] Add or update the focused RED test before implementation.
- [ ] Preserve raw-pointer ABI and four-axis registry dispatch.
- [ ] Keep the strict unfused or strict decode fallback registered.
- [ ] Run CPU-reference and applicable production numerical gates.
- [ ] Run a kernel trace proving the intended kernel actually ran.
- [ ] Run the full paired benchmark before calling it a win.
- [ ] Report absolute rates and all three ratios.

Promotion rule: retain only changes that are correct, reproducible, and
non-regressive on the declared suite. A ratio gain with lower UD tok/s is
diagnostic unless the absolute result is still an explicitly accepted tradeoff.

### Phase 4: Close lifecycle and serving coverage

Before automatic MTP admission, complete the U6 envelope:

- [ ] Caller ABI rows 1/2/3/4/5/7/8 and prefill tile/chunk boundaries.
- [ ] c1/c2/c4/c8, ragged and sparse rows, permutations, delayed arrivals,
  neighbor replacement, cancellation, reclaim, and width transitions.
- [ ] Exact speculative accept/reject commit and rollback.
- [ ] Draft/verifier state disjointness, aliases, and teardown.
- [ ] Block-64 Q6 `eh_proj`, attention, and FFN operations.
- [ ] Explicit artifact-scoped strict manifest.
- [ ] Explicit backend/profile/context/width scope.

The pin remains empty until all declared certification items are complete.

### Phase 5: Re-run the complete paired economics gate

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

### Phase 6: Promote or hold

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
