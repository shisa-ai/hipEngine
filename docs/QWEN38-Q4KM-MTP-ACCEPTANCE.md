# Qwen3.8-27B `Q4_K_M` physical-C3 MTP decode-economics campaign

- Status: **E1/E2 complete; C3/K3 external parity passed; E5/E6 promotion or wider-width re-freeze next**
- Created: 2026-08-28; corrected review: 2026-08-28; E1a retained: 2026-08-29
- Hardware lane: **AMD Ryzen AI MAX+ 395 / Radeon 8060S / `hip_gfx1151` / HIP 7.15**
- Primary product key: **Qwen3.8-27B `Q4_K_M`, BF16 KV, production profile, physical C3, raw greedy, context 1-67, D24**
- Current product state after C2 streaming: **strict C1/K3 natural25 remains automatic at 1.6445x; production C2/K3 is an exact explicit diagnostic at 1.4316x and remains automatic K0 pending refreshed evidence; C3/K3 is an exact explicit diagnostic at 1.2297x, 7.45% above the frozen external row, and remains automatic K0 pending the complete production/serving gate**
- Primary promotion gate: **C3 `>=1.10x` true same-protocol AR overall, full/heldout/every category non-regressive, complete production correctness and serving gates**
- Stretch target: **`>1.30x` true AR**, consistent with [`BENCHMARK.md`](BENCHMARK.md)
- Binding predecessors (extend; do not reimplement):
  [`QWEN38-Q4KM-MTP-SERVING.md`](QWEN38-Q4KM-MTP-SERVING.md),
  [`CONCURRENCY2-GFX1151-MTP-DYNAMIC-ADMISSION.md`](CONCURRENCY2-GFX1151-MTP-DYNAMIC-ADMISSION.md),
  [`OLMX-IDEAS.md`](OLMX-IDEAS.md),
  [`SPECDEC2.md`](SPECDEC2.md), and [`SPECDEC2-PERF.md`](SPECDEC2-PERF.md)
- Opening evidence:
  [`C1 matched acceptance closeout`](../benchmarks/results/2026-08-28-gfx1151-qwen38-c1-acceptance-parity-closeout.json),
  [`C2 production rowtiles`](../benchmarks/results/2026-08-28-gfx1151-qwen38-c2-production-q4-rowtile-retained.json),
  [`C3 retained rowtiles`](../benchmarks/results/2026-08-28-gfx1151-qwen38-c3-production-rowtiles-retained.json),
  [`OI-2 adaptive rejection`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi2-adaptive-rejected.json),
  [`OI-4 post-norm rejection`](../benchmarks/results/2026-08-25-gfx1151-qwen38-omlx-oi4-postnorm-rejected.json),
  [`current E0 baseline`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json),
  [`E1a prompt streaming`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e1a-prompt-streaming-retained.json),
  [`E1b proposal-head rowtile`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e1b-proposal-head-rowtile-retained.json)
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
> route, while preserving automatic C1, exact C2 diagnostics/K0 policy, and
> every strict fallback.**

The E0 current-source C3/K3 route was **21.382 vs 24.119 tok/s true AR
(0.8865x)**. E1a and E1b reduce complete MTP wall from 33.673 to 24.659 seconds
and E2 now reaches **29.564 vs 24.042 tok/s (1.2297x)** with exact E0
acceptance and every category positive. This beats the frozen external C3/K3 row by 7.45%
but does not replace the full production/serving promotion bundle, so automatic
C3 remains K0. The campaign succeeds if one of these outcomes is reached:

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
| C1/K3 | strict, cap1/cap4 realized singleton, context 1-67, natural25 | **18.191 / 11.062 = 1.6445x** | 161/220 = **73.18%** | automatic; current manifest refreshed |
| C2/K3 | production, cap4 physical C2, context 1-128, D24 | **25.749 / 17.986 = 1.4316x** | 314/398 = **78.89%** | retained exact diagnostic; automatic K0 pending refresh |
| C3/K3 | production, canonical natural contexts (<=67), D24 | **29.564 / 24.042 = 1.2297x** | 471/597 = **78.89%** | external parity passed; automatic K0 pending full gate |
| C4/K3 | production, physical C4, D24 | **29.493 / 30.291 = 0.9737x** | 628/796 = **78.89%** | external parity passed; automatic K0 on overall/category AR |
| C5-C8/K3 | production diagnostics, D24 | **18.708-29.527 / 35.704-47.640 = 0.5240-0.7051x** | **78.89%** | exact implementation wins; automatic K0 |

C2 and C3 still have identical draft acceptance. Post-output-norm prompt
streaming now removes the activation wall at both widths, while E1b removes
duplicate physical proposal-head sweeps without changing candidate IDs or either
acceptance trajectory. C2 clears the implementation economic/category gate too,
but its typed automatic evidence remains unchanged until the complete admission
bundle is refreshed. Its external parity gap is separately closed as a measured
blocker: target/accept/commit/provider is 60.38% of the profile child and 95.38%
kernel-bound; Q4/Q6/Q5 consume 740/246/64 ms. Qualified R8 shapes are already
one-sweep, while narrow-Q4 shared-B is numerically unqualified and insufficient
alone even at zero cost.
[`C2 blocker`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-c2-post-streaming-blocker.json)

Current E2 C3 category economics:

| Category | MTP tok/s | AR tok/s | Ratio | Status |
| --- | ---: | ---: | ---: | --- |
| code | 30.016 | 24.279 | 1.2363x | positive |
| general English | 31.419 | 24.141 | 1.3015x | positive |
| general Japanese | 28.311 | 24.272 | 1.1664x | positive |
| mixed Japanese/English | 28.296 | 23.271 | **1.2159x** | positive |
| full | 29.564 | 24.042 | **1.2297x** | external parity passed |

Runtime policies use model/quant/profile/physical-width or immutable H/N/row
shape keys only. No category, prompt, token, or prompt-length selector was added.

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

- OI-4 post-output-norm **draft output** hidden is rejected: B3 **-1.62%**,
  with code/Japanese/mixed regressions; a B2 aggregate gain still failed
  heldout/Japanese gates. That proposal-chain policy is distinct from E1a's
  post-output-norm target prompt seed contract.
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

Prompt activation is different. E1a now admits the existing shifted streaming
path only for the gfx1151 dense-H5120 `MOSTLY_Q4_K_M` production physical-C3
key. Streamed target rows are output-normalized on device before NextN consumes
them; the first pre-output-norm screen was rejected after changing acceptance
from 471/597 to 468/597. C2/C4, other quant/model/profile keys, and peer backends
retain host-hidden capture/replay. The adjacent `Q4_K_S` C2/C4 category
rejection therefore remains binding in its own scope.

On the 36-token profile child, `nextn_prompt_prime` falls from 746.7 to 41.5 ms
while complete wall falls from 3.285 to 2.664 seconds. Target prefill rises from
543.1 to 701.1 ms because it now includes streamed NextN work, but measured
activation-to-first-decode ownership falls from 1.290 to 0.701 seconds. The
refreshed 41.5 ms prime is only 1.6% of profile-child complete wall, so a new
multi-request state-only priming kernel no longer has a material E1a premise.

Current cached profiles expose two steady-cycle walls as well:

**Proposal (E0 current C3/K3 steady trace):**

- proposal wall **62.32 ms**, with **59.84 ms** in kernels;
- three full-vocabulary planar-Q6 head calls **41.26 ms** (66.2% of steady
  proposal kernels);
- Q4 NextN work **12.75 ms**;
- remaining proposal kernels about **5.83 ms**.

E1b now routes physical proposal-head H5120/N248320 rows2-4 through the existing
exact Q6 F32 rowtile plus generic GPU argmax. Actual row3 improves
**13.736→4.711 ms (-65.70%)**. Marker-scoped C2/C3/C4 traces each show 25
rowtile calls over nine proposal windows, one per physical proposal depth, with
zero direct fallback. C3 proposal wall falls **451.0→258.3 ms (-42.72%)**.
C1, AR, target verification, prefill, shape misses, and peer backends retain
prior owners.

**Target (E0 current C3/K3 R12 steady trace):**

- target/accept/commit/provider wall **200.48 ms**, with **192.22 ms** in
  kernels;
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

The E0 C3 packet generates 720 tokens per arm:

- true AR complete wall: **29.852 s** (`24.119 tok/s`);
- MTP complete wall: **33.673 s** (`21.382 tok/s`);
- `1.10x` promotion ceiling: **27.138 s** (`26.531 tok/s`).

E1a+E1b+E2 remove **9.319 s / 27.68%** from E0 complete MTP wall and improve
throughput by **38.27%**. C3/K3 is now **2.049 tok/s / 7.45% above** the frozen
27.515 tok/s external row. Complete production and serving admission remain
open.

### 2.5 Post-E1 target Amdahl refresh

The retained E1b C3/K3 child still spends **1.521/2.503 s (60.76%)** in eight
`target_accept_commit_provider` markers. Their kernel sum is **1.454 s / 95.62%
of marker wall**:

| Target class | Calls | Total ms | ms/cycle | Target-marker share |
| --- | ---: | ---: | ---: | ---: |
| Q4 | 3,254 | 821.68 | **102.71** | **54.02%** |
| Q6 | 983 | 401.50 | **50.19** | **26.40%** |
| Q5 | 720 | 103.86 | **12.98** | **6.83%** |
| all remaining kernels | 8,274 | 127.33 | 15.92 | 8.37% |

Q4+Q6+Q5 are **165.88 ms/cycle**, 87.25% of target marker wall and 53.01% of
profile-child complete wall. C3 R12 still decomposes to R8+R4, so duplicate
weight sweeps remain the dominant actionable wall even after external C3 parity.
E2 therefore passes its entry condition and starts with a true-R12 Q4 owner.
The C3 E5/E6 promotion bundle remains required, but it does not replace E2's
cross-width value while C2 and C4-C8 parity remain open.
[`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-post-e1-amdahl.json)

## 3. What already exists — do not reimplement or relabel

- Generation-2 request plans, physical proposal groups, target frontiers,
  transactional accept/selected commit, provider repair, output, cancellation,
  K0 transitions, and D7 admission.
- Physical C2/C3/C4 proposal at K1-K3, device-resident request-major candidates,
  packed target R4-R16 mechanics, one group accept payload, and zero routine
  candidate D2H before target execution.
- OI-3 exact C1 streaming prompt priming, provider pooling/groups, fixed cycle
  slabs, and zero hot allocation after warmup. Physical C2/C4 streaming remains
  disabled after the adjacent `Q4_K_S` category rejection; E1a independently
  qualifies post-output-norm streaming for the standard-`Q4_K_M` production C3
  key only.
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

- [x] Run a common D24 current-source C1/C2/C3 K3 diagnostic under one committed
      raw-token rendering and timing contract. Separately rerun the certified
      strict-C1 natural25 and production-C2 D24 scopes as regression controls;
      do not silently equate their horizons/profiles.
- [x] Run C3 K1/K2/K3 plus true AR and intent K0, counterbalanced, with full/
      train/heldout/category and position telemetry.
- [x] Collect one final cached-only C3/K3 child trace: proposal depth families,
      target Q4/Q5/Q6/attention/GDN/head, accept/commit/repair, copies, syncs,
      allocations, and host residual.
- [x] Reconcile non-profiled timing-owner totals to physical cycles and tails;
      separate provider open, prompt prime/TTFT, steady cycles, and reclaim.
      Report activation by prompt-length/root-position bin and explain the
      general-English vs mixed-category wall split without using content in
      policy.
- [x] Recompute the exact complete-wall reduction required for aggregate
      `>=1.10x` and every category `>=1.0x`.
- [x] Publish one baseline artifact and a candidate Amdahl table. No candidate
      starts without a named parent row and maximum complete-wall contribution.

Exit: complete 2026-08-29 in the
[`E0 artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e0-current-baseline.json).
K1 was the fastest unoptimized C3 depth (0.9344x), and the E0 Q6 head consumed
41.26 ms/cycle at K3. E1a and E1b subsequently raise K3 to 1.2041x AR and pass
the external comparator. Strict C1 automatic is unchanged; production C2 now
passes implementation economics but remains explicit pending refreshed typed
evidence and serving gates.

### E1 — close proposal-side activation and head walls

#### E1a — physical-C3 prompt activation adjudication

This starts with an existing exact candidate, not fresh device code.

- [~] Under the E0 protocol, compare current physical host replay with the
      already-implemented streaming prompt path at C3. Complete wall, target
      prefill, provider open, `nextn_prompt_prime`, three prompt-length bins,
      and every category are recorded in the E1a artifact. The profile child's
      activation-to-first-decode owner improves 1.290→0.701 seconds; direct
      scheduler/client SSE TTFT remains for the serving gate.
- [x] Preserve shifted NextN semantics exactly: prompt row 0 consumes `t[0]`
      with zero hidden; row i consumes `t[i]` with post-output-norm target hidden
      `h[i-1]`; E0 candidate IDs and 471/597 acceptance are exact. The rejected
      pre-output-norm screen changed acceptance to 468/597 and was not retained.
- [x] The existing path does not repeat the C2/C4 category rejection: all four
      categories are positive. Admission uses one model/quant/profile/C3 key,
      with no category, prompt, or length selector.
- [x] Do not build a distinct true multi-request state-only prime: refreshed
      `nextn_prompt_prime` is 41.5 ms, only 1.6% of the profile-child complete
      wall, so the material-entry condition no longer holds after E1a.
- [x] The conditional RED matrix for a new priming design is not applicable
      because that candidate did not pass its entry condition. Existing
      equal/ragged/chunk/offset/permutation/cancel/teardown coverage remains.
- [x] Retain the exact full/heldout/every-category win: **27.169 vs 24.085 tok/s
      (1.1280x AR)**, up **27.06%** from E0, with all ten cells exact/engaged/
      budget-conformant. Proceed to E1b; automatic C3 remains K0 pending full
      promotion gates. [`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e1a-prompt-streaming-retained.json)

#### E1b — exact physical proposal-head row reuse (first new kernel-route candidate)

Hypothesis: physical rows2-4 proposal scoring should use the existing exact
planar-Q6 F32 rowtile (one head weight sweep) plus the existing GPU row argmax,
not one direct head sweep per request row. E0 measures **41.26 ms across three
K3 head calls per cycle**, 66.2% of proposal kernels.

- [x] RED actual immutable K5120/N248320 rows2/3/4 fixture: every FP32 logit,
      lowest-ID tie behavior, top-1 ID/value, row order, and guard bytes match
      the direct parent.
- [x] Route only physical NextN proposal scoring through the existing exact
      planar-Q6 F32 rowtile. Its wrapper uses the 16-column body at rows2 and
      col8 body at rows3/4. The direct F32 producer remains the strict fallback.
- [x] Keep full-vocabulary scoring and generic GPU argmax. No dp4a, vocabulary
      truncation, direct top-1, graph, hidden-policy, or K4 change is included.
- [x] C1 proposal, AR, target verification, prefill, peer backends, non-Q6
      primitives, and H/N/row misses retain their prior owners by package and
      four-axis primitive resolution.
- [x] Marker-scoped physical C2/C3/C4 traces each show 25 rowtile calls over
      nine proposal windows, one call/depth, zero direct fallback, no candidate
      D2H, and 4.59-4.75 ms/call.
- [x] Complete clean C2/C3 K3 economics preserve exact 314/398 and 471/597
      acceptance. C2 reaches **21.690 vs 18.038 tok/s (1.2025x)** and C3 reaches
      **29.198 vs 24.249 (1.2041x)**; every category is positive.
      [`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e1b-proposal-head-rowtile-retained.json)

Falsifier: reject the route if the actual head family does not improve, if
row-identity/top-1 differs, or if complete C3 wall regresses. A useful expected
signal is moving the current C3 per-depth head from about 13.8 ms toward the
existing 4.60-4.66 ms one-sweep leaf; that is a prediction, not a keep rule.

### E2 — genuine high-row target amortization (R9/R12 first; R15/R16 prepared)

Hypothesis: the retained R7+R2 / R8+R4 decomposition still streams Q4/Q5/Q6
weights twice. A true high-row owner can reduce the dominant target families
without changing control or model representation.

Candidate order follows measured wall, one logical unit at a time:

1. ~~Q4 single + gate/up/SiLU actual shapes (**102.71 ms/cycle** at R12);~~
   **closed rejected:** exact col4 and shared-weight R12 owners are slower than
   R8+R4 on every actual shape;
2. ~~planar/standard Q6 actual shapes (**50.19 ms/cycle**);~~ **standard
   K5120/N10240 true-R12 retained; planar R12 rejected and keeps R8+R4**;
3. ~~Q5 recurrent output (**12.98 ms/cycle**);~~ **true-R12 retained**;
4. attention/GDN/other leaves remain below the campaign's material entry floor.

Q4 closeout: the exact register-bounded col4 candidate is 9.11-44.74% slower
than R8+R4 across all six actual single/dual shapes, with 0/15 paired wins per
shape. The required shared-weight follow-up is 72.93-169.68% slower. Both match
the representative actual-output parent and guard contract, but fail before
runtime admission; no complete-wall run is warranted. The named blocker is
that extra column workgroups or multi-wave LDS/barrier cost exceeds the second
cached weight sweep at R12 on gfx1151. Keep R8+R4 and proceed to Q6.
[`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e2-q4-true-r12-rejected.json)

Post-Q5 R16 revisit: Q5 proved host launch count can dominate a nearly neutral
GPU leaf, so Q4 received one bounded operation-complete R16 reconsideration.
The exact col4 single/dual candidates lose every one of 90 pairs by 43.1-70.3%;
weighted Q4 GPU work rises **101.03→156.90 ms (+55.31%)** despite halving 400
to 200 launches/group. One-prompt C4 falls **29.610→27.717 tok/s (-6.39%)**.
All candidate code is removed; do not reopen without a new dataflow premise.
[`artifact`](../benchmarks/results/2026-08-30-gfx1151-qwen38-mtp-q4-r16-rejected.json)

Q6 closeout: standard K5120/N10240 true-R12 is BF16-bit exact and improves
**0.6915→0.4468 ms (-35.38%)**, 15/15 leaf wins. Clean C3 improves
**29.198→29.409 tok/s (+0.72%)**; the trace replaces standard R8+R4
**98.12→75.21 ms (-23.35%)**. Planar K5120/N1024 and K17408/N5120 R12
candidates lose +6.13%/+3.73%, are removed, and keep R8+R4.
[`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e2-standard-q6-true-r12-retained.json)

Q5 closeout: K6144/N5120 true-R12 is BF16-bit exact and improves
**0.2654→0.1940 ms (-26.91%)**, 15/15 leaf wins. Clean C3 improves
**29.409→29.564 tok/s (+0.53%)**; the trace replaces Q5 R8+R4
**97.11→78.07 ms (-19.60%)**. E2 is complete; no remaining target class has a
measured material entry premise.
[`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-e2-q5-true-r12-retained.json)

The following rules governed the completed family screens and remain binding
if E2 is reopened by new evidence:

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

- [x] Obtain same-model full-suite depth-4 proposal survival under a strict
      eager/direct diagnostic with the exact raw prompt rendering. The exact-
      raw B3/B4 oracle covers all ten canonical/heldout prompts.
- [x] Measure the K4 proposal/native-R5 target cost on the current owners. B4
      raises proposal time 9.95%, cuts target verify 7.26%, and improves visible
      transitions/cycle 12.70% while reducing cycles 71→63.
- [x] Compute `visible(K) = 1 + sum_j P(accept through j)` and
      `score(K)=visible(K)/complete_cycle_wall(K)` by full/train/heldout/category.
      B4 improves full +4.53% and train +9.63%, but fails heldout -2.81% and
      general English -9.82%.

The oracle fails, so the following integration work is explicitly **not
opened**: Generation-2 B4 adapter/workspace/claim/graph extensions, B1-B4
transition RED, physical C3 logical-R15/padded-R16 ownership, C3/K4 manifests,
and the production/serving promotion bundle. Temporary oracle capacity,
native-R5, and R5 Q4 policy changes are removed.

No online/adaptive controller is implemented in this campaign. Fixed K4 does
not win every binding aggregate scope, so fixed K3 remains the policy input.
[`artifact`](../benchmarks/results/2026-08-30-gfx1151-qwen38-mtp-b4-reopen-rejected.json)

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

- [x] Evaluate all changed T0/T1/T2 boundaries against strict parents and record
      the final selected/fallback manifest.
- [x] Canonical + category-heldout D24 full-logit production gate: calibrated
      mean/p95/p99/max KL, top-1 by category/shape/transition/accepted depth,
      three deterministic repeats, task review where required.
- [x] Same-width C3 neighbor replacement, row permutation, slot movement,
      ragged lengths, sparse retirement, delayed arrival, cancellation/reclaim,
      C1<->C2<->C3 transitions, refill, output tails, the context67/68
      admission boundary, and graph/eager fallback. Context68+ remains K0 in
      the primary packet.
- [x] Exact request/slot/token/position/mask/`KVLiveSpans`/state/KV/transaction/
      lifecycle ownership and zero final allocations.
- [x] Regression controls: automatic strict C1 and production C2 remain within
      their certified scope and pass same-suite economics/correctness.
- [x] For T0/provider changes, same-schedule candidate IDs and target acceptance
      are exact. For T1/T2 target arithmetic, strict-vs-production generated-ID
      equality is diagnostic; the production distribution/task/control gates
      are binding.

E5 closes 2026-08-30 by composition on the frozen selected stack. Current
C3/K3 canonical/heldout KL max is **8.69e-4/8.45e-4**, top-1 **240/240 +
192/192**, with three deterministic repeats and exact teardown. Existing C3
K1-K3 evidence covers 1,296 rows; D5 plus the current 64-test bundle covers the
ownership/lifecycle matrix. Later Q5/Q6 owners are BF16-bit exact to those
qualified parents. E6 still owns automatic promotion.
[`artifact`](../benchmarks/results/2026-08-30-gfx1151-qwen38-mtp-e5-combined-correctness.json)

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

1. **C4:** exact Q6 R8+R8 cuts its traced family **1.847→0.478 s (-74.10%)**;
   exact Q5 true-R16 then raises clean K3 to **29.493 vs 30.291 tok/s AR**,
   9.17% above external, with exact 628/796 acceptance. Current wall trails AR
   by 0.8572 s across the suite; prior `nextn_prompt_prime` is 9.08% of its
   child and can cover the residual, but C4 streaming changes acceptance to
   624/800. External parity is closed; automatic C4 remains K0 until prompt
   streaming preserves replay-equivalent state plus overall/code/Japanese/mixed
   AR and independent production/serving gates.
   [`artifact`](../benchmarks/results/2026-08-30-gfx1151-qwen38-mtp-c4-post-q5-blocker.json)
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
6. **C5-C8:** the retained physical-R16 Q6 owner improves clean MTP to
   **17.970/26.904/28.205/27.393 tok/s (+17.89%/+29.16%/+21.34%/+39.41%)**
   through C4-sized provider groups. Target decompositions are C5 R16+R4, C6
   R16+R8, C7 R16+R12, and C8 R16+R16—not logical R20-R32 calls; unengaged wide
   keys are removed. All outputs and 78.894% acceptance remain exact, and a C8
   trace reduces whole-process BF16 Q6 **8.707→3.210 s (-63.13%)**. Exact Q5
   true-R16 then raises C5-C8 to **18.708/28.255/29.527/29.504 tok/s
   (+4.10%/+5.02%/+4.68%/+7.71%)**, with every category positive. Proposal
   telemetry remains rows4+remainder. Every width still trails AR and external
   by 22-46%; pursue R16/remainder Q4 work only with its exact or production
   numerical gates.
   [`artifact`](../benchmarks/results/2026-08-29-gfx1151-qwen38-mtp-q6-r20-r32-retained.json)

## 6. Candidate priority and reopen matrix

| Priority | Candidate | Measured premise | First falsifier / stop |
| ---: | --- | --- | --- |
| 0a | Physical C3 prompt activation | E1a retained: 746.7→41.5 ms prompt prime and 21.382→27.169 tok/s, exact acceptance, every category positive | closed retained; direct SSE TTFT remains in E6 |
| 0b | Physical proposal Q6 F32 rowtile | E1b retained: row3 13.736→4.711 ms; C3 27.169→29.198 tok/s; exact C2/C3 acceptance | closed retained; C3 external parity passed |
| 1 | True R9/R12 target owner | Post-E1 C3 target was 190.12 ms/cycle; Q4/Q6/Q5 were 102.71/50.19/12.98 ms | E2 C3 closed: Q4 rejected; standard Q6 + Q5 retained; planar Q6 rejected. |
| 1b | C4 exact Q6 R16 | C4 target owner was 72.68% of child; Q6 direct kernels were 1.847 s | closed retained: R8+R8 cuts Q6 74.10%, C4 reaches 27.450 tok/s and passes external parity |
| 1c | C5-C8 physical-R16 Q6 carryover | provider partitions targets as R16+R4/R8/R12/R16 | closed retained: +17.89% to +39.41%; unengaged logical R20-R32 keys removed |
| 1d | Physical-R16 Q5 one-sweep | Q5 R12 predecessor won; R16 parent paid two Python/ctypes launches | closed retained: C4-C8 +4.10% to +7.71% exactly; C4 reaches 0.9737x AR |
| 1e | Physical-R16 Q4 one-sweep revisit | Q5 proved launch count material, reopening the prior leaf-only stop once | rejected: weighted GPU +55.31% and one-prompt C4 -6.39%; candidate removed |
| 1f | C4 prompt-prime exactness | prompt prime is 9.08% of child versus 2.63% overall gap | blocked: streaming changes 628/796→624/800; require replay-equivalent shifted state before economics |
| 2 | Fixed K4 / adaptive reopen | native-R5 exact-raw B4: full +4.53%, visible/cycle +12.70% | rejected: heldout -2.81%, general English -9.82%; adaptive not opened |
| 3 | NextN norm/concat/Q4 residual | E0 Q4 NextN work is 12.75 ms/cycle at C3/K3 after the head | < material refreshed Amdahl share or compound-only idea |
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
