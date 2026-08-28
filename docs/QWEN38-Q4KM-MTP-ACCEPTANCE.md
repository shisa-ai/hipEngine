# Qwen3.8-27B `Q4_K_M` physical-C3 MTP decode-economics campaign

- Status: **replanned after evidence/source audit; E0-E7 scoped, no campaign implementation started**
- Created: 2026-08-28; corrected review: 2026-08-28
- Hardware lane: **AMD Ryzen AI MAX+ 395 / Radeon 8060S / `hip_gfx1151` / HIP 7.15**
- Primary product key: **Qwen3.8-27B `Q4_K_M`, BF16 KV, production profile, physical C3, raw greedy, context 1-67, D24**
- Opening product state: **C1/K3 and C2/K3 are automatic; C3/K3 is a retained 0.9589x diagnostic; C3 automatic remains K0**
- Primary promotion gate: **C3 `>=1.10x` true same-protocol AR overall, full/heldout/every category non-regressive, complete production correctness and serving gates**
- Stretch target: **`>1.30x` true AR**, consistent with [`BENCHMARK.md`](BENCHMARK.md)
- Binding predecessors (extend; do not reimplement):
  [`QWEN38-Q4KM-MTP-SERVING.md`](QWEN38-Q4KM-MTP-SERVING.md),
  [`CONCURRENCY2-GFX1151-MTP-DYNAMIC-ADMISSION.md`](CONCURRENCY2-GFX1151-MTP-DYNAMIC-ADMISSION.md),
  [`OLMX-IDEAS.md`](OLMX-IDEAS.md),
  [`SPECDEC2.md`](SPECDEC2.md), and [`SPECDEC2-PERF.md`](SPECDEC2-PERF.md)
- Opening evidence:
  [`C1 matched acceptance closeout`](../benchmarks/results/2026-08-28-gfx1151-qwen38-c1-acceptance-parity-closeout.json),
  [`C2 automatic production`](../benchmarks/results/2026-08-28-gfx1151-qwen38-c2-production-q4-rowtile-retained.json),
  [`C3 retained rowtiles`](../benchmarks/results/2026-08-28-gfx1151-qwen38-c3-production-rowtiles-retained.json),
  [`OI-2 adaptive rejection`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi2-adaptive-rejected.json),
  [`OI-4 post-norm rejection`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi4-postnorm-rejected.json)
- Normative dependencies: [`PLAN.md`](PLAN.md), [`TESTING.md`](TESTING.md),
  [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md),
  [`BENCHMARK.md`](BENCHMARK.md), [`KERNELS.md`](KERNELS.md), and
  [`CONCURRENCY2.md`](CONCURRENCY2.md)
- Sibling artifact campaign (separate model bytes and gates):
  [`QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md`](QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md)

## 1. Executive correction and objective

The original version of this plan was built around an alleged **11.59-point
hipEngine-versus-llama.cpp draft-acceptance deficit**. That premise is invalid.
The old row compared hipEngine literal/raw Qwen-marker prompts with llama.cpp
chat-templated prompts containing exactly 43 additional tokens per case and a
different generated root. Under identical rendered prompt bytes and 25 visible
outputs, hipEngine is **165/210 = 78.571%** and llama.cpp is **166/208 =
79.808%**: a non-actionable **-1.236 points**, with **10/10 prompts and 250/250
visible tokens identical**.

This is therefore not a C1 acceptance-repair campaign. The product defect that
made wide verification reread weights per row was real and is substantially
repaired; the remaining work is an optimization problem:

> **Lower physical-C3 complete wall enough to promote the already-correct C3
> route, while preserving the automatic C1 and C2 cells and every strict
> fallback.**

The current C3/K3 route is **19.934 vs 20.788 tok/s true AR (0.9589x)**. The
campaign succeeds if one of these outcomes is reached:

1. **Primary success:** a physical-C3 fixed-budget cell clears `>=1.10x` true
   same-protocol AR overall, has no full/heldout/category/task/SLO regression,
   passes complete production numerics/ownership/lifecycle, and is promoted
   through the existing D7 admission owner.
2. **Secondary success:** after C3 promotion, an independently qualified C4
   cell clears the same gate. A shared R16 kernel does not transfer C3 evidence
   to C4 request topology.
3. **Measured closure:** every candidate with a valid Amdahl premise is tested
   or falsified, exact non-regressive sub-wins are retained, and the remaining
   C3 gap is attributed to named measured walls. C3 stays K0 if any product gate
   fails.

A faster microkernel, verifier-derived `B0`, prompt-conditioned policy, changed
greedy acceptance rule, or aggregate win hiding a failed category is not a
campaign success.

## 2. Corrected evidence basis

### 2.1 Current product ladder

These rows are same model file and same physical host, but their declared
profiles/scopes differ and must remain explicit:

| Cell | Profile/scope | MTP vs true AR | Draft acceptance | Product state |
| --- | --- | ---: | ---: | --- |
| C1/K3 | strict, cap4 realized singleton, context 1-67, natural25 | **15.609 / 9.807 = 1.5916x** | 165/210 = **78.57%** | automatic |
| C2/K3 | production, context 1-128, D24 | **17.031 / 14.887 = 1.1441x** | 314/398 = **78.89%** | automatic |
| C3/K3 | production, canonical natural contexts (<=67), D24 | **19.934 / 20.788 = 0.9589x** | 471/597 = **78.89%** | retained diagnostic; automatic K0 |
| C4+ | unqualified economics | n/a | n/a | automatic K0 |

C2 and C3 have essentially identical draft acceptance. The C2-to-C3 loss is
therefore not evidence of proposal-quality collapse; AR scales faster than the
remaining proposal/target dataflow.

C3 category economics:

| Category | MTP tok/s | AR tok/s | Ratio | Immediate requirement |
| --- | ---: | ---: | ---: | --- |
| code | 20.268 | 20.728 | 0.9778x | recover 2.27% throughput to non-regression |
| general English | 21.748 | 20.489 | 1.0614x | preserve |
| general Japanese | 19.774 | 20.715 | 0.9546x | recover 4.76% |
| mixed Japanese/English | 17.987 | 21.298 | **0.8445x** | recover **18.41%** |
| full | 19.934 | 20.788 | **0.9589x** | recover **14.71%** to reach 1.10x AR |

The mixed category, not the aggregate, is the binding non-regression pressure.
No category-specific runtime branch is permitted; the table only sizes the
content-independent wall reduction required.

### 2.2 Acceptance and provider-state facts — closed unless re-triggered

Matched C1 evidence:

- hipEngine/llama.cpp overall acceptance: **78.571% / 79.808%**;
- initial prompt-owned root and full K3 chain: **10/10 exact**;
- later same-visible-history depth top-1 agreement: depth 1 **67/67**, depth 2
  **59/65**, depth 3 **53/62**;
- all final visible output tokens: **250/250 exact**.

The later depth-2/3 residual is localized to provider recurrence/repair
semantics, but aggregate acceptance and final outputs are already at parity. It
is not an economic defect by itself.

Current C3 position telemetry is still useful for fixed-depth economics:

| Position | Proposed cycles | Accepted-through cycles | Unconditional survival | Conditional acceptance |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 213 | 198 | 92.96% | 92.96% |
| 2 | 198 | 153 | 71.83% of roots | 83.61% when preceding/opportunity conditions hold |
| 3 | 186 | 120 | 56.34% of roots | 83.33% when preceding/opportunity conditions hold |

Per request-cycle, measured K3 visible work is
`1 + 471/213 = 3.211` tokens. Even a perfectly accepted fourth draft can add at
most `120/213 = 0.563` token per existing root cycle before K4-specific
opportunity changes: **17.54% maximum visible-token lift at zero added cost**.
K4 therefore requires an explicit cost/acceptance oracle; external K4 wins do
not authorize it.

Two proposal-policy experiments are already closed:

- OI-4 post-output-norm hidden is rejected: B3 **-1.62%**, with code/Japanese/
  mixed regressions; a B2 aggregate gain still failed heldout/Japanese gates.
- OI-2 content-agnostic adaptive B1/B2/B3 is rejected: **-0.58% to -1.72%**
  versus fixed B3, with category regressions. Its exact transition
  infrastructure may be reused; its controller is not rerun.

Provider-state alignment reopens only if a future **same-render, same-output,
repeat-confirmed** packet demonstrates a deficit large enough to change the
measured C3 keep/promotion decision.

### 2.3 Measured C3 operation map

The source path already performs real physical **decode-cycle** proposal
batching. At each draft depth,
`Qwen35GGUFNextNExecutor.run_batch_proposal_device()` runs one C-row NextN
backbone, retains request-major candidate IDs/hidden rows on device, and hands
them to one packed target group. Do not add another generic “batch the drafter”
task.

Prompt activation is different. The adapter currently advertises
`physical_prompt_streaming = False`: the adjacent `Q4_K_S` physical C2/C4
streaming candidate was disabled after general-English regressions of
11.86%/10.16%, so current physical groups retain host-hidden capture/replay.
That different-quant result is a negative prior, not Q4_K_M evidence. The current C3 category data
makes activation worth re-measuring rather than assuming away: general English
and mixed Japanese/English both accept 84.21% of drafts, yet their ratios are
1.061x and 0.845x, while the mixed prompt roots are around positions 60-67
versus 35-36. This is a correlation, not attribution; E0 must report prompt-
length/activation wall, and E1a either re-closes or qualifies the existing
candidate before any new batched-activation design.

Current cached profiles expose two steady-cycle walls as well:

**Proposal (final C3/K2 steady trace):**

- proposal wall **41.84-41.98 ms** after first-cycle noise;
- two full-vocabulary planar-Q6 head calls **27.67-27.71 ms** (about 66% of
  steady proposal kernels);
- Q4 NextN work **8.47-8.49 ms**;
- remaining proposal kernels **3.89-3.90 ms**.

The physical F32 proposal head currently bypasses the existing exact Q6 F32
rowtile and executes the direct planar body at rows=3, effectively rereading the
**1,042,944,000-byte / 994.629-MiB** head per row. Existing isolated actual-root
rowtile evidence is **4.60-4.66 ms** with exact logits/top-1. Transfer to the
physical proposal path is the first campaign candidate, not yet a claim.

**Target (final C3/K2 R9 steady trace):**

- target/accept/commit/provider wall **195.16-196.37 ms**;
- Q4 rowtiles **97.06-97.40 ms / 400 calls**;
- Q6 rowtiles **53.94-53.99 ms / 131 calls**;
- Q5 rowtiles **11.04-11.08 ms / 96 calls**;
- other + remaining Q4 WMMA about **22.8 ms**;
- about **10-11 ms** marker wall is outside kernel sum.

Current C3/K3 R12 uses the same bounded mechanism: R12 decomposes to **R8+R4**,
with target **206.62-208.27 ms**, Q4 about **102 ms**, Q6 **54-55 ms**, and Q5
**14.3-14.5 ms**. The old per-row cliff is gone, but two weight sweeps remain.
A genuine R9/R12/R15/R16 high-row owner that reads each weight tile once is the
second campaign lever.

Profiler wall carries tracing overhead and cannot be multiplied into a product
claim. E0 must join cached leaf attribution with unprofiled timing-owner totals.

### 2.4 Complete-wall break-even budget

The retained C3 packet generates 720 tokens per arm:

- true AR complete wall: `720 / 20.7882 = 34.635 s`;
- MTP complete wall: `720 / 19.9341 = 36.119 s`;
- `1.10x` promotion ceiling: `720 / (1.10 * 20.7882) = 31.486 s`.

C3 must remove **4.633 s / 12.83% of current complete wall** at unchanged AR.
That is a **14.71% throughput increase**. Mixed Japanese/English must remove
**1.244 s / 15.55%** of its current wall merely to reach 1.0x AR. E0 recomputes
these budgets from its own counterbalanced baseline and reconciles them to
actual physical cycles, tails, TTFT, and lifecycle costs before any candidate
is sized.

## 3. What already exists — do not reimplement or relabel

- Generation-2 request plans, physical proposal groups, target frontiers,
  transactional accept/selected commit, provider repair, output, cancellation,
  K0 transitions, and D7 admission.
- Physical C2/C3/C4 proposal at K1-K3, device-resident request-major candidates,
  packed target R4-R16 mechanics, one group accept payload, and zero routine
  candidate D2H before target execution.
- OI-3 exact C1 streaming prompt priming, provider pooling/groups, fixed cycle
  slabs, and zero hot allocation after warmup. Physical C2/C4 streaming was
  category-rejected on the adjacent `Q4_K_S` campaign and remains disabled;
  its implementation is the E1a starting candidate, while Q4_K_M/C3 requires
  independent evidence.
- P7 root snapshots and conditional provider repair for reject/K-1/full/other
  depths; final-state direct-commit cleanup.
- Production C3 K1/R6, K2/R9, K3/R12 full-logit manifests, three repeats,
  same-width isolation, and strict fallbacks.
- Position/category acceptance telemetry with explicit draft, unconditional
  position, and conditional-position denominators.
- Variable-budget B1/B2/B3 transition tests from OI-2.
- Matched raw-prompt llama.cpp acceptance/oracle tooling.

Closed/rejected non-repeats:

- the 90.16% chat-template “acceptance gap”;
- post-output-norm draft hidden (OI-4);
- adaptive B1/B2/B3 controller (OI-2);
- n-gram composition on canonical product traffic;
- c1 producer-owned Q6 top-1 (**4.591 -> 4.604 ms**, rejected);
- accuracy-traded X8/dp4a top-1 routes as production-exact substitutes;
- the historical multi-row direct-top1 shortcut that emitted an invalid second
  row sentinel — physical rows use exact rowtile logits + GPU argmax until a
  new RED-proven multi-row design exists;
- graph capture pursued only to reduce launch count despite negligible uncovered
  wall;
- accept/readback micro-tuning while proposal/target model work dominates.

## 4. Binding campaign method

### 4.1 Frozen axes and comparison arms

Every retained comparison freezes:

- physical host/device, `GPU_MAX_HW_QUEUES=2`, HIP/compiler versions, and KFD
  exclusivity;
- model full SHA-256, quant, BF16 KV, execution profile and manifest;
- raw prompt-token IDs/hash, category split, sampler, context bucket, D24
  horizon, resident capacity, batch window, and output accounting;
- one timing boundary and one physical shape interpretation.

Every product candidate reports these arms where applicable:

1. **true AR:** no speculative provider or mutation;
2. **intent K0:** provider ownership/catch-up but zero speculative cycles;
3. **current retained MTP control:** fixed C/K and current manifest;
4. **candidate MTP:** identical C/K unless the unit is explicitly the fixed-K4
   phase.

A verifier-derived `off`/`B0` row is diagnostic only. Never weaken, serialize,
or disable a qualified AR fast path to improve the ratio. If a shared kernel
makes AR and MTP faster but lowers the MTP/AR ratio, retain the valid absolute
AR/MTP win in its own scope, rebase against the faster AR denominator, and keep
C3 K0 until the rebased speculative gate passes.

### 4.2 Metrics and wall reconciliation

Primary decision metric: full-suite same-protocol **MTP / true-AR complete-wall
ratio**. Required secondary metrics:

- aggregate and per-request tok/s;
- full/train/heldout/every-category wall and ratio;
- TTFT, inter-token latency p50/p95/p99, E2E, queue delay, SLO goodput;
- proposed/accepted drafts, accepted/output, visible tokens/cycle, per-position
  unconditional/conditional acceptance;
- physical C/K/logical R, actual proposal rows by depth, target decomposition,
  active/padded masks, graph bucket and manifest hash;
- provider open/prime, proposal, target, accept/commit, provider repair,
  scheduler/readback/reclaim wall;
- kernel sum/union/top families, launch/sync/H2D/D2H/allocation counts;
- tracked/process/whole-device high-water and exact final conservation.

A profile must reconcile at least **90% of the selected complete cycle/target
window** before choosing a kernel family. Unknown wall remains an upper bound,
not “Python overhead.” Profile the final child after cache warmup; never wrap a
parent harness that launches nested processes.

### 4.3 Statistical discipline

- Actual-weight leaf screens: at least 5 warmups and 15 counterbalanced pairs.
- Operation-complete candidate screen: one clean full-suite control/candidate
  packet with the guarded comparator before retention.
- Promotion packet: at least three independent counterbalanced full-suite
  repetitions, with predeclared tolerance derived from baseline variance
  (default zero). No nonzero tolerance may hide a heldout/category regression.
- Cold provider construction/graph capture and steady pooled/replay rows are
  reported separately.
- No compiler process, foreign KFD owner, or unexplained thermal/clock shift may
  overlap a retained measurement.

## 5. Ordered campaign

### E0 — corrected baseline, economics model, and artifact freeze

No implementation changes.

- [ ] Run a common D24 current-source C1/C2/C3 K3 diagnostic under one committed
      raw-token rendering and timing contract. Separately rerun the certified
      strict-C1 natural25 and production-C2 D24 scopes as regression controls;
      do not silently equate their horizons/profiles.
- [ ] Run C3 K1/K2/K3 plus true AR and intent K0, counterbalanced, with full/
      train/heldout/category and position telemetry.
- [ ] Collect one final cached-only C3/K3 child trace: proposal depth families,
      target Q4/Q5/Q6/attention/GDN/head, accept/commit/repair, copies, syncs,
      allocations, and host residual.
- [ ] Reconcile non-profiled timing-owner totals to physical cycles and tails;
      separate provider open, prompt prime/TTFT, steady cycles, and reclaim.
      Report activation by prompt-length/root-position bin and explain the
      general-English vs mixed-category wall split without using content in
      policy.
- [ ] Recompute the exact complete-wall reduction required for aggregate
      `>=1.10x` and every category `>=1.0x`.
- [ ] Publish one baseline artifact and a candidate Amdahl table. No candidate
      starts without a named parent row and maximum complete-wall contribution.

Exit: clean baseline artifact, guarded objective, final manifest hashes, and a
ranked wall budget. If the Q6 proposal head is no longer material, E1b is
re-ranked from the fresh trace rather than executed by plan inertia.

### E1 — close proposal-side activation and head walls

#### E1a — physical-C3 prompt activation adjudication

This starts with an existing exact candidate, not fresh device code.

- [ ] Under the E0 protocol, compare current physical host replay with the
      already-implemented streaming prompt path at C3. Report target prefill,
      provider open, `nextn_prompt_prime`, TTFT, complete wall, prompt-length
      bins, and every category.
- [ ] Preserve shifted NextN semantics exactly: prompt row 0 consumes `t[0]`
      with zero hidden; row i consumes `t[i]` with target hidden `h[i-1]`;
      cursors/KV and final carried hidden agree with replay; no discarded prompt
      scoring.
- [ ] If the existing path repeats the C2/C4 category rejection, leave
      `physical_prompt_streaming=False` and record the C3 closeout. Do not tune
      the selector by category or prompt length.
- [ ] Only if E0 still attributes material complete wall to physical provider
      activation, screen a distinct **true multi-request state-only priming**
      design: at each prompt position, pack active request rows through one
      NextN state transition, with ragged masks and no logits. Reuse the existing
      batch state-only primitive/strict per-request replay; do not call serial
      streaming “batched.”
- [ ] RED the new design over equal/ragged prompt lengths, chunk boundaries,
      warm offsets, request permutation, cancellation between chunks, provider
      cursor/KV/state, first proposal, and teardown before a GPU candidate.
- [ ] Retain only a full/heldout/every-category complete-wall win. Otherwise
      close activation and proceed to E1b.

#### E1b — exact physical proposal-head row reuse (first new kernel-route candidate)

Hypothesis: physical rows2-4 proposal scoring should use the existing exact
planar-Q6 F32 rowtile (one head weight sweep) plus the existing GPU row argmax,
not one direct head sweep per request row.

- [ ] RED actual immutable K5120/N248320 rows2/3/4 fixture: every FP32 logit,
      lowest-ID tie behavior, top-1 ID/value, row order, and guard bytes match
      the current direct parent.
- [ ] Route only physical NextN proposal scoring through the existing exact
      planar-Q6 F32 rowtiles: the 16-column
      `t16_gemv_rowtile_bf16_f32_out` owner where qualified and the rows3/4
      `t16_gemv_rowtile_col8_bf16_f32_out` sibling otherwise. Preserve the
      current direct F32 producer as strict fallback.
- [ ] Keep full-vocabulary scoring and generic GPU argmax in the first unit.
      Do not combine row reuse with dp4a, vocab truncation, direct top-1, graph
      capture, hidden-policy changes, or K4.
- [ ] Prove C1 proposal, AR, target verification, prefill, peer backends, and
      shape misses retain prior owners.
- [ ] Profile physical C2/C3/C4 head rows and confirm one rowtile call/depth,
      no candidate D2H, no fallback, and plausible one-sweep duration.
- [ ] Run complete C2/C3 K3 control/candidate economics. T0 requires identical
      candidate IDs/hidden rows/acceptance and must not regress automatic C2.

Falsifier: reject the route if the actual head family does not improve, if
row-identity/top-1 differs, or if complete C3 wall regresses. A useful expected
signal is moving the current C3 per-depth head from about 13.8 ms toward the
existing 4.60-4.66 ms one-sweep leaf; that is a prediction, not a keep rule.

### E2 — genuine high-row target amortization (R9/R12 first; R15/R16 prepared)

Hypothesis: the retained R7+R2 / R8+R4 decomposition still streams Q4/Q5/Q6
weights twice. A true high-row owner can reduce the dominant target families
without changing control or model representation.

Candidate order follows measured wall, one logical unit at a time:

1. Q4 single + gate/up/SiLU actual shapes (about 102 ms at R12);
2. planar/standard Q6 actual shapes (about 54-55 ms);
3. Q5 recurrent output (about 14 ms);
4. only then profile-triggered attention/GDN/other leaves.

For each quant family:

- [ ] Freeze actual-weight R6/R9/R12 and prospective R15/R16 leaves, parent
      arithmetic, row maps, output dtype, and strict fallback before device
      edits.
- [ ] Screen a register-bounded column-split high-row design first (for example
      col4/col8 ROW_TILE9/12/15/16); if register/scratch pressure loses, screen
      a multi-wave shared-weight tile. Do not silently fall back to two sweeps
      and call it a high-row result.
- [ ] RED every output row against the declared strict/production parent,
      including guard rows, odd tails, active/padded masks, and same-width
      neighbor substitution.
- [ ] Require actual-weight microbench and cached trace engagement before
      runtime admission.
- [ ] Route only the declared Qwen3.8/gfx1151/production/physical shape through
      a manifest selection; strict, C1/C2, peer backends, narrow-Q4, D25+, and
      unlisted shapes retain prior owners.
- [ ] For T1/T2, run canonical + category-heldout full-logit mean/p95/p99/max
      KL, top-1 by category/shape/transition/row role, three repeats, isolation,
      task, lifecycle, and a registered strict fallback.
- [ ] Keep every exact or fully gated same-suite non-regressive win, even if the
      cumulative C3 cell remains below 1.10x.

After each family, rerun the guarded complete C3 gate. Stop tuning a leaf whose
measured complete-wall contribution cannot affect a keep/promotion decision.
Graph capture is considered only if the post-kernel trace leaves material
uncovered wall.

### E3 — fixed K4 oracle, infrastructure, and depth re-rank

Fixed K4 is not OI-2 adaptive depth. It proceeds only after E1/E2 establish a
new cost table.

**Oracle before integration:**

- [ ] Obtain same-model full-suite depth-4 proposal survival under a strict
      eager/direct diagnostic with the exact raw prompt rendering. This is
      proposal-quality evidence, not a product speed row.
- [ ] Measure or tightly bound K4 proposal-depth and physical R15->R16 target
      cost on the E1/E2 owners.
- [ ] Compute `visible(K) = 1 + sum_j P(accept through j)` and
      `score(K)=visible(K)/complete_cycle_wall(K)` by full/train/heldout/category.
      Reject K4 infrastructure if it cannot beat fixed K3 under the measured
      cost, even at the favorable confidence bound.

If the oracle passes:

- [ ] Extend Generation-2 adapter capability beyond `{1,2,3}` without broadening
      existing C1 graph admission accidentally. Update fixed candidate/result/
      hidden workspaces, claims, telemetry, graph keys, and teardown.
- [ ] Add exact B1/B2/B3/B4 transition RED coverage, especially B3<->B4,
      reject/every partial/full, tails with remaining room 1/B/B+1, cancellation,
      reset, and subsequent AR/MTP health.
- [ ] Lower C3/K4 as logical R15 with an explicit R16 physical bucket/padded-row
      contract. Inactive row state/KV/output must remain untouched and padding
      cannot enter acceptance denominators.
- [ ] Register independent strict/production C3/K4 manifests and fail closed on
      every scope miss.
- [ ] Run canonical + heldout D24 full-logit gates, three repeats, same-width
      isolation, lifecycle, and the complete fixed K1/K2/K3/K4 economics table.

No online/adaptive controller is implemented in this campaign. If fixed K4 does
not win every binding aggregate scope required for its role, fixed K3 remains
the policy input.

### E4 — residual proposal, provider-update, and host tail (profile-triggered)

Open only after E1-E3 refresh the Amdahl table.

Possible exact/T0 targets:

- fused/hoisted embedding-norm + hidden-norm packing for the physical NextN
  input, eliminating per-row D2D concat copies while preserving both BF16 norm
  boundaries;
- Q4 NextN projection row reuse if it remains material;
- batched state-only provider repair using already-retained proposal hidden
  rows;
- selected-commit/result publication only if it exceeds 5% of the refreshed
  cycle/complete wall;
- physical proposal graph replay only if uncovered submission/synchronization
  wall is material after kernels are fixed.

Rules:

- [ ] One boundary per RED/GREEN unit; no compound norm/head/graph experiment.
- [ ] Keep proposal candidate IDs/hidden, target decisions, provider cursor/KV,
      and acceptance exact for T0.
- [ ] Do not optimize the accept kernel or bounded readback while they remain a
      sub-percent wall.
- [ ] Do not overlap proposal/target streams until stable pointers, dependency
      events, and a trace prove real non-overlapped slack. Phase-serial work is
      not automatically overlap headroom.

Static vocabulary caps, a lower-quant head, or a new NextN artifact are T3
provider/model experiments. They require a separate manifest and full quality/
acceptance/task campaign and are not mixed into the exact E1-E4 ladder.

### E5 — combined production correctness and regression matrix

Run after the selected implementation stack is frozen; do not qualify every
exploratory combination.

- [ ] Evaluate all changed T0/T1/T2 boundaries against strict parents and record
      the final selected/fallback manifest.
- [ ] Canonical + category-heldout D24 full-logit production gate: calibrated
      mean/p95/p99/max KL, top-1 by category/shape/transition/accepted depth,
      three deterministic repeats, task review where required.
- [ ] Same-width C3 neighbor replacement, row permutation, slot movement,
      ragged lengths, sparse retirement, delayed arrival, cancellation/reclaim,
      C1<->C2<->C3 transitions, refill, output tails, the context67/68
      admission boundary, and graph/eager fallback. Context68+ remains K0 in
      the primary packet.
- [ ] Exact request/slot/token/position/mask/`KVLiveSpans`/state/KV/transaction/
      lifecycle ownership and zero final allocations.
- [ ] Regression controls: automatic strict C1 and production C2 remain within
      their certified scope and pass same-suite economics/correctness.
- [ ] For T0/provider changes, same-schedule candidate IDs and target acceptance
      are exact. For T1/T2 target arithmetic, strict-vs-production generated-ID
      equality is diagnostic; the production distribution/task/control gates
      are binding.

Any failed binding scope rejects the compound candidate. Do not average a
failed category or transition into a pass.

### E6 — product economics, automatic admission, and serving

- [ ] Run at least three counterbalanced full-suite true-AR/current-control/
      candidate repetitions under one public blocking timing contract.
- [ ] Require overall `>=1.10x`, full/heldout/every category non-regression,
      exact token accounting, and no TTFT/ITL/E2E/goodput/memory regression
      outside the predeclared SLO envelope.
- [ ] Add a C3 evidence row only through the model-plugin typed evidence table;
      D7 remains the sole policy owner. Static intent and dynamic actual C/K are
      separate.
- [ ] Prove actual physical C3 engagement and truthful K/R/manifest telemetry;
      no label-only MTP or request-serial C1 lowering.
- [ ] Every context/horizon/profile/quant/backend/K/physical-width miss selects
      true K0 before proposal mutation with a stable reason; the first C3 key
      stops at context67.
- [ ] Re-run blocking + SSE, mixed permanent-AR/spec peers, C2<->C3 survivor,
      cancellation/deadline, below/near/above load, overload/rejection, restart,
      soak, and complete drain.

If an implementation win is exact/non-regressive but the product cell misses
1.10x or a category/SLO gate, retain the owner in its proved scope but keep C3
automatic K0 and record the concrete blocker.

### E7 — independent extensions after C3 closure

These are ordered follow-ons, not assumptions in the primary claim:

1. **C4:** independently profile/qualify C4/K1-K4. Shared R16 device code may be
   reused, but C4 request topology, isolation, acceptance, complete wall, SLO,
   and policy evidence do not transfer from C3.
2. **Context 68-128:** run the predeclared padded full category/heldout packet,
   production numerics, state/isolation, and same-protocol economics before
   extending the first C3 key. C2's context128 evidence does not transfer.
3. **Longer decode horizon:** current C2 production arithmetic is authorized
   only through D24; the D120 screen has max KL **0.08574 > 0.05**. Localize and
   repair that transition or use strict fallback before any D64/D120 automatic
   claim. A short-D24 win is labeled short-D24.
4. **Context >128:** prior C1 diagnostics fall to 1.063x/1.017x/0.897x at
   contexts 256/512/1020. New context buckets need independent economics and
   numerical/state gates.
5. **gfx1100:** source ideas may transfer, absolute rates, manifests, and
   thresholds may not.

## 6. Candidate priority and reopen matrix

| Priority | Candidate | Measured premise | First falsifier / stop |
| ---: | --- | --- | --- |
| 0a | Physical C3 prompt activation | physical streaming is disabled after an adjacent Q4_K_S C2/C4 category rejection; mixed Q4_K_M prompts are longer despite acceptance matching English | E0 shows immaterial wall, or existing/new exact batching fails any category |
| 0b | Physical proposal Q6 F32 rowtile | 27.7 ms across two K2 head calls; existing actual root rowtile 4.60-4.66 ms | no actual-head/complete-wall win or any row/top-1 mismatch |
| 1 | True R9/R12/R16 target owner | R12 still R8+R4; Q4/Q6/Q5 about 102/55/14 ms | cannot reduce sweep count or actual target/complete wall |
| 2 | Fixed K4 | max zero-cost visible lift 17.54%; external K4 is diagnostic | measured p4/cost score cannot beat fixed K3 |
| 3 | NextN norm/concat/Q4 residual | about 12 ms total in C3/K2 proposal after head | < material refreshed Amdahl share or compound-only idea |
| 4 | Provider update/selected commit | unprofiled telemetry currently single-digit ms/cycle | <=5% refreshed wall or P7 already owns best path |
| 5 | Graph/submission/overlap | steady trace uncovered about 10 ms target wall | no trace-proven slack; kernel work still dominates |
| conditional | target-hidden provider repair | matched C1 acceptance already at parity | no repeat-confirmed matched economic deficit |
| separate T3 | static vocab/lower-quant or new draft artifact | 994.6-MiB full-vocab head | exact E1/E2 ladder still has material head wall and separate quality campaign approved |

## 7. RED contract inventory

| Surface | Required RED / gate |
| --- | --- |
| Prompt parity | exact raw prompt token IDs/hash and visible output budget; chat template cannot silently enter an external comparator |
| Proposal head | actual Q6 rows2/3/4 every-logit/top-1/row-order/guard-byte parity; device candidate handoff and no-D2H assertions |
| Prompt activation | replay/stream/batched shifted token-hidden timeline, ragged/chunk/warm-offset ordering, provider cursor/KV/state, first proposal, cancellation and teardown |
| Proposal chain | root token/position, parent/depth, candidate hidden, provider cursor/KV, reject/partial/full repair, request reuse |
| High-row target | actual R9/R12/R15/R16 outputs; active/padded masks; parent topology; positions/`KVLiveSpans`; strict fallback; no inactive mutation |
| K4 | B1-B4 transition matrix, all accept depths, remaining room 1/B/B+1, graph-key separation, fixed workspace/claims, teardown |
| Production arithmetic | canonical + heldout full logits, calibrated KL/top-1 by category/shape/transition/role, three repeats, isolation, task and manifest |
| Dynamic ownership | C1/C2/C3 grow/shrink, ragged peers, permutation, delayed arrival, cancellation/refill, post-fault health |
| Telemetry | one timing owner/group, exact generated counts, C/K/R/decomposition, position denominators, no duplicate batch wall |
| Economics | true no-MTP AR arm in the same protocol; full/train/heldout/category guarded comparator; complete wall primary |
| Memory/lifecycle | warm zero-allocation cycles, high-water, close/restart, zero active claims/allocations/provider groups |
| Dispatch | four-axis registered key, selected/fallback manifest, model/backend/quant/profile misses fail closed; no hot-path backend/quant branch |

New/ported kernels additionally require the [`KERNELS.md`](KERNELS.md) lineage
check, strict or production RED, CPU-reference outer floor, registered strict
fallback, expected-name `rocprofv3` launch trace, and complete same-suite gate.

## 8. Binding anti-gaming and decision rules

- No prompt-, category-, token-ID-, candidate-ID-, expected-output-, or fixture-
  conditioned proposal, verifier, or admission branch.
- Full committed mtpbench categories plus fixed category heldouts decide every
  keep/revert. Single prompts and oracle token histories are diagnostic only.
- Optimize speed first. Accepted/output is report-only; higher acceptance does
  not retain a slower route. Proposal-policy work additionally requires
  non-regressive draft acceptance.
- No provider-state “parity” task starts from the superseded 90.16% row.
- No adaptive controller before a fixed K4 cell wins. OI-2 is a negative control,
  not a starting implementation.
- No K4 promotion from an oracle-only or verifier-only row.
- No C4/horizon/backend transfer from a C3/D24/gfx1151 result.
- No graph/launch-count win without complete marker and end-to-end wall.
- Never slow the true-AR arm, add provider shadow work to it, or select a weaker
  AR implementation to manufacture an MTP ratio.
- Every exact same-suite non-regressive implementation win is retained in its
  proved scope even if automatic C3 remains K0.
- Automatic policy changes only through typed D7 evidence after all gates; env
  flags are temporary experiment selectors and require `REFACTOR.md` cleanup
  criteria.

## 9. Deliverables and definition of done

Required artifacts:

1. E0 corrected C1/C2/C3 baseline + full timing/Amdahl packet;
2. one compact artifact per retained/rejected E1-E4 unit, with exact source and
   raw trace hashes;
3. fixed K1-K4 score table if E3 proceeds;
4. final production profile evaluation and selected/fallback manifest;
5. three-repeat C3 product economics + public serving/load packet;
6. typed D7 policy artifact if and only if C3 promotes;
7. benchmark rollup/changelog updates for every retained performance result.

Campaign closure requires:

- all stale 90.16%-gap prose removed or explicitly marked superseded;
- every candidate in the priority matrix retained, rejected, or blocked by its
  named entry criterion;
- C1/C2 non-regression and strict fallback preservation;
- C3 automatic evidence either promoted or explicitly K0 with a measured reason;
- `docs/KERNELS.md`, `docs/EXECUTION-PROFILES.md`, `docs/REFACTOR.md`, and
  `docs/PLAN.md` updated when their ownership/profile/phase facts change;
- immutable worklog entries, `scripts/worklog.py check`, benchmark README sync,
  applicable tests/gates, and atomic validated commits.

## 10. Deferred speculative-method follow-ons

| Follow-on | Entry criterion | Disposition |
| --- | --- | --- |
| DFlash2 sidecar | Primary MTP E0-E7 closes or a separate lane is approved; §11 N1-N3 must pass | Separate provider campaign; do not mix economics |
| Tree/EAGLE3/SpecExec proposals | Fixed K3/K4 remain below gate after E1/E2 and verifier has measured headroom | Separate tree-parent/mask/acceptance campaign using `KVLiveSpans` |
| Static vocab/lower-quant NextN head | Exact full-vocab E1 remains material after row reuse and a separate T3 quality campaign is approved | Different provider/model manifest; full category/task gate |
| Stochastic speculative sampling | Product chooses non-greedy output distributions | Separate RNG/residual-correction contract |
| n-gram composition | New broad non-replay product cell | Currently closed; canonical D24 has zero useful hits |
| `UD-Q4_K_M` artifact | Sibling campaign | No evidence transfer from current file |

## 11. Appendix: what a DFlash2 revival looks like on this architecture

This is an architectural preview, not part of E0-E7 and not a commitment.
DFlash2 is not hypothetical for this tree: the
[`QWEN38-27B-DFLASH2-CAMPAIGN.md`](QWEN38-27B-DFLASH2-CAMPAIGN.md)
closed **diagnostic** on 2026-08-19 (B3 optimum 8.85 tok/s = 0.66x AR), and
its 2026-08-22 correction rewrote the attribution: DFlash2 was at **acceptance
parity** with exact MTP (2.80 vs 2.85 tokens/cycle; 0.70 vs 0.74 per verify
row). The deficit was cost — about 96 ms/cycle of drafter+select and verify at
2.14 sweeps/cycle at four rows and **8.01 sweeps at eight rows** (the historical
`_PACK8_ROWTILE_MAX_ROWS = 4` cliff).

### 11.1 What changed since closure

- Row-independent Q4/Q5/Q6 owners now amortize physical R6/R9/R12/R16 instead
  of the historical one-sweep-per-row behavior. The retained C3 route verifies
  12 rows in 206.62-208.27 ms (about 2.7 old Q4_K_M sweep equivalents) rather
  than DFlash2's eight rows in 8.01 sweeps.
- Same-width isolation, per-row selected-state commit, and explicit profile
  manifests now provide the N2 localization machinery missing from the reverted
  May rowtile-8 attempt.
- This does **not** automatically qualify DFlash2: its tap-capture target path
  still needs an independent N1 row curve, and E2's goal is to reduce the
  remaining two-sweep high-row target further.

The old verify cliff is no longer a sufficient reason to keep DFlash2 closed.
The remaining structural costs are its approximately 96 ms drafter+select
(<60 GB/s effective against 3.584 GiB residency in the old trace) and fixed-B
proposal policy.

### 11.2 Why external DFlash2 routes won where this tree lost

The externally successful DFlash2 routes are llama.cpp-family implementations:
Laurent's fork (FP4 target + 1.03 GB `Q4_0` sidecar, adaptive K3-7: 34.483
token-weighted common-suite tok/s, 60.43% acceptance; 56.532 valid structured
JSON fresh-process) and PieBru/Nathanw (UD Q5/Q6/Q8 targets 20.9-31.5 GB +
2.06 GB `Q8_0` sidecar: DFlash decode 30.659/26.470/23.044 vs AR
10.695/8.778/7.275 tok/s = **2.86x/3.01x/3.17x**, acceptance
53.19%/42.92%/43.94%). `q38rocm` is **not** DFlash2; it is built-in MTP K4 on
a custom FP4 format. On their shared FP4 target, Laurent DFlash2 measured
34.483 vs q38rocm 32.969 token-weighted tok/s.

| Mechanism | External winners | Ours at 2026-08-19 closure | Effect |
| --- | --- | --- | --- |
| Verify weight traffic | effectively about one batched target sweep/cycle | 2.14 sweeps at B4; 8.01 at B8 | >1 sweep lost at B4; about 7 at B8 |
| Drafter | 1.0-2.1 GB sidecar on the engine's efficient path | 3.584 GiB, <60 GB/s effective, about 96 ms draft+select | largest independent deficit |
| Acceptance | 43.9-60.4% per draft token | 0.70 per verify row / 2.80 per cycle | parity-scale; not the loss cause |
| Measurement | one engine/protocol per comparison | Q4_K_S vs Q4_K_M, 25 vs 40 outputs, different harness/tap boundaries | N3 attribution was not matched |

They won because a small efficient sidecar plus batched verify amortized target
weights per cycle. We lost because an inefficient larger drafter fed a verifier
behind a four-row admission cliff. The algorithm's acceptance was not the
problem.

### 11.3 Falsifiable DFlash2 reopen order

The old campaign N-numbering remains authoritative:

1. **N1:** `verify_target_block` rows1-8 with/without tap capture; record actual
   sweeps, kernel families, and complete wall.
2. **N2:** root-cause the old rowtile-8 state divergence using current
   row-independent owners and same-width isolation.
3. **N3:** one target file, one harness, one timing boundary, tap on/off, same
   suite/budget for AR/MTP/DFlash2.
4. **Native drafter cost:** profile grouped conv/attention/head/selector and
   pursue >120 GB/s effective only with CPU-reference REDs and retained strict
   parents.
5. **N4:** only then test a content-agnostic adaptive proposal gate from selector
   confidence; fixed B remains the control.

Old falsifiable projection (still not a claim): at parity 2.80 tokens/cycle, an
approximately 1.1-sweep verify (~85 ms on the old Q4_K_M sweep anchor) plus a
3.584-GiB drafter at >120 GB/s (~32 ms; less for sidecar-class bytes) and about
5 ms select gives roughly 23-27 tok/s versus that campaign's 13.4 tok/s AR.
N1-N3 must confirm or kill the arithmetic under one matched protocol.

### 11.4 Architecture mapping and honesty rules

| Concern | Existing owner DFlash2 reuses |
| --- | --- |
| Provider lifecycle | Generation-2 `SpecRequestPlan`, provider groups, activation/catch-up/refill/teardown, D7 admission |
| Draft execution | four-axis registry; drafter conv/attention/head/selector as registered layer/quant variants with strict parents |
| Verify | same packed target runner and `KVLiveSpans`; no DFlash-only verifier fork |
| State | target/provider transaction, FP16/FP32 rollback contracts, request-boundary reset, contamination gates |
| Memory | resident sidecar/drafter bytes, draft KV, graph/workspace and high-water resource claims |
| Numerics | tap capture as an execution-profile variant; unchanged strict untapped parent |

External rows are diagnostic upside only: their artifacts, formats, sidecars,
and protocols differ. A hipEngine revival still requires true same-protocol AR,
full mtpbench categories plus heldouts, exact greedy target control, sequential
contamination/lifecycle gates, and `>=1.10x` before any automatic cell exists.
A second cleanly attributed rejection is a valid result; “the gap is
unclosable” without N1-N3 is not.
