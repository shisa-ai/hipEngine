# Qwen3.8-27B gfx1151 Structural Differential Campaign

Status: **opened 2026-09-02; Z0 pending**
Successor to the closed
[`scaling campaign`](QWEN38-GFX1151-SCALING-CAMPAIGN.md) and the
[`external implementation survey`](QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md).
Owner: scaling loop.

Scope: the frozen product key from the scaling campaign — physical gfx1151
(Ryzen AI MAX+ 395 / Radeon 8060S), `Qwen3.8-27B` `standard_q4_k_m`
(SHA-256 `7e78da5d…c6fe169`), BF16 KV, production profile, the common
ten-prompt suite (SHA-256 `fac920be…1d86084a`), raw greedy, no prompt cache,
and the declared D24 serving scope unless a unit explicitly measures another
scope. External comparisons keep the survey's standardized complete-wall
boundary and the survey's declared route for each engine.

This campaign exists because the local same-dataflow ladder is closed. The
next source of improvement must be a different, measured mechanism: a changed
prefill dataflow, a multi-family packed-verifier dataflow, or a proposal
mechanism shown by teacher-forced evidence. Do not reopen isolated owner
morphology without that new premise.

## 1. Entry state

The latest published full-width hipEngine prefill collateral is the Y1
candidate/control matrix. It predates the retained Y2 Q6 shared3r1 and Y3
planar Q6 pair-decode units, so Z0 must re-measure current head before any
new claim. The MTP rows below are the reviewed K3 rows except where the
retained C6/C8 K1 successor supplies a newer value.

| Lane | Cell | hipEngine evidence | External leader | Gap |
| --- | --- | ---: | ---: | ---: |
| Prefill | C2 | 154.900 tok/s | Laurent 211.888 | **-26.9%** |
| Prefill | C8 | 239.658 tok/s | Laurent 305.847 | **-21.6%** |
| Prefill | C6 | 234.305 tok/s | Laurent 250.4 | -6.4% |
| Prefill | C7 | 236.532 tok/s | Laurent 252.3 | -6.2% |
| MTP | C7 | 33.106 tok/s K3 | Stock HIP 46.084 | **-28.2%** |
| MTP | C8 | 43.421 tok/s K1 | Stock HIP 56.222 | **-22.8%** |
| MTP | C5 | 27.980 tok/s K3 | Mainline Vulkan 32.713 | -14.5% |
| MTP | C2 | 28.441 tok/s K3 | Laurent 32.221 | -11.7% |
| MTP | C1 | 15.753 tok/s K3 | Laurent 21.126 | **-25.4%** |
| AR | C2 | 18.090 tok/s | Laurent 19.835 | -8.8% |

Controls and exclusions:

- MTP C6 is a control, not a target: hipEngine K1 reaches 37.074 tok/s versus
  Laurent 37.154 (**-0.22%**).
- MTP C3/C4 already lead the fixed-K3 external matrix.
- AR C3-C8 already lead. AR C1 is 0.45% behind Nathan and is too small to
  justify a dedicated campaign.
- The C5/C7 MTP values remain older K3 diagnostics because the one-pass
  C5/R10 and C7/R14 K1 owners regressed and were rejected.
- AR C2 is a secondary diagnostic. Treat it as potentially downstream of the
  C2 prefill deficit until attribution proves otherwise.

## 2. Why the previous campaign closed

The W/Y prefill extensions and M-track MTP work retained several exact wins,
but also produced bounds that rule out another local tuning pass:

- W1's best validated two-wave Q6 owner reduced R32 Q6 device time by
  **33.5%**, yet the full-wall combination was neutral or negative and a
  perfect two-stage pipeline still bottoms at **1.716x R32/R8**, above the
  1.25x flatness gate.
- W2 found that removing every measured target non-kernel gap would save only
  **4.31%** of the target stage, below the required **19.2%/20.1%** complete
  C6/C8 wall reductions. Distinct Q4/Q5/Q6 tensors cannot share weight bytes
  by co-scheduling alone.
- W3 measured scheduling/co-scheduling as null.
- Y5 found that perfect realization of the Y3 residual plus perfect removal
  of every remaining GDN/other non-GEMM dispatch would remove **19.16%** of
  C8 wall, still short of the **21.64%** required to match Laurent.
- The M1/M2 successor loop retained exact C6/C8 K1 owners, but closed on the
  same structural premise: a multi-family packed-verifier dataflow is needed
  before broad MTP can improve again.

These are measured closure conditions, not a request to try harder on the
same axes.

## 3. Correctness and arithmetic policy

**Bit-exactness is not the production bar.** The historical bias toward
bit-exact kernels helped localize bugs, but this campaign must evaluate
production candidates under the normative
[`execution-profile gates`](EXECUTION-PROFILES.md), not reject them solely
because their arithmetic differs from the strict parent.

The policy for this campaign is:

1. **Control and ownership remain exact in every profile.** Request identity,
   token ownership, positions, masks, `KVLiveSpans` metadata, recurrent state,
   sampler accounting, lifecycle, and scheduler ownership are never numerical
   drift.
2. **T0 remains welcome but is not mandatory for production.** T1 local
   implementation drift and T2 association/layout drift are eligible when the
   complete production gate passes. A repacked layout or width-specific
   reduction may change BF16 bytes or generated IDs at near ties.
3. **T3 changes stay explicit.** A changed weight quantization, changed KV
   storage policy, approximate router, changed speculative acceptance policy,
   or changed sampling distribution is not ordinary production implementation
   drift. It needs an explicit experiment/product configuration and separate
   authorization.
4. **Strict remains the oracle.** Every production fused/composite variant
   needs a registered strict fallback and a declared exact or parent-parity
   contract for the strict profile.
5. **Production quality is binding.** A candidate must pass the calibrated
   strict-teacher mean/p95/p99/max KL and top-1 gates by applicable
   category, shape, and transition; same-schedule deterministic repeats;
   dynamic isolation; applicable BF16-relative and task gates; and the full
   multi-prompt category plus heldout suite.
6. **Generated-ID equality is diagnostic for production.** It is binding only
   where the declared strict or batch-invariant contract makes it binding.
7. **Do not relabel defects.** State contamination, route-miss behavior,
   nondeterminism, or ownership bugs fail even if the aggregate KL happens to
   pass.

This policy expands the implementation search space without weakening model
quality. It also means a candidate may be retained as a production-profile
variant even when it cannot become the strict oracle.

## 4. Goals and non-goals

### Goals

1. Re-freeze the current-head C1-C8 baseline after the retained Y2/Y3 prefill
   work and C6/C8 K1 MTP work.
2. Attribute the remaining prefill gap to a named external mechanism with a
   measured upper bound.
3. Attribute the remaining MTP gap to a named proposal, target-verifier, or
   accept/commit mechanism with a measured upper bound.
4. Implement only mechanisms whose optimistic bound can cover the target gap
   with margin, then retain or reject them under the applicable profile gate.
5. Preserve the existing AR C3-C8 lead and all automatic-serving K0 fail-closed
   behavior unless a complete production/economics packet says otherwise.

### Non-goals

- Retrying W1-style double buffering, W3-style scheduling, Y5-style
  non-GEMM cleanup, or isolated Q4/Q5/Q6 morphology without a new measured
  premise.
- Acceptance-only MTP tuning or a return to serial per-request verification.
- Chasing AR C1's 0.45% deficit.
- Copying an external fork wholesale.
- Custom-format results such as `ROCmFP4_FAST`; they are useful mechanism
  evidence but not standard-`Q4_K_M` engine claims.
- Laurent adaptive DFlash2 as a product route; its sequential-request state
  leak remains a correctness blocker.
- Prompt-conditioned, token-conditioned, or candidate-conditioned benchmark
  branches.

## 5. Punchlist

Every item closes with a retained result or a measured, named blocker. A
blocked item must include the exact command, physical host, model/prompt hash,
execution profile, compact JSON artifact, and the mechanism bound that makes
further work unjustified.

### Z0 — current-head baseline and instrumentation refresh

- [ ] Re-run the standardized C1-C8 prefill, true-AR, and explicit MTP
  diagnostics at current head on the physical gfx1151 host.
- [ ] Record exact commands and raw-source hashes for every run. The Y1
  matrix is the latest full-width prefill collateral but predates Y2/Y3; do
  not use it as the campaign's final current-head baseline.
- [ ] Collect current-head C2/C8 prefill and C5/C7/C8 MTP operation-complete
  attribution: kernel family, launch count, host/API/copy time, proposal,
  target, accept/commit, KV, scheduler, and server overhead.
- [ ] Publish one compact Z0 artifact and update the benchmark rollup only if
  a retained public number changes.

Exit: one same-host, current-head matrix and attribution set that can serve
as the denominator for every later candidate.

### Z1 — prefill external differential

Primary cells: C2 and C8 against Laurent Vulkan. Secondary cells: C6/C7.
Carry C1/C3/C4/C5 as regression controls, not targets.

- [ ] Run hipEngine and Laurent under the same model, prompt suite, one-output
  prefill boundary, physical host, and concurrency schedule.
- [ ] Compare complete wall, kernel-family time, launch counts, row shapes,
  weight-byte movement, dequant strategy, LDS/register pressure, occupancy,
  and final-head behavior.
- [ ] Read the pinned Laurent implementation and cite source file plus commit
  for any mechanism considered for porting. Do not vendor or edit the external
  repository.
- [ ] Produce a mechanism table that separates measured facts from inferred
  causes.
- [ ] Compute an optimistic complete-wall bound for each mechanism. Continue
  only when the bound is at least **1.25x the required wall reduction** for a
  primary cell: at least 33.6% for C2 and 27.1% for C8.

Exit: either a named prefill dataflow candidate with a measured bound, or a
stronger impossibility result that closes the remaining prefill gap.

### Z2 — MTP external differential

Primary wide cells: C7/C8 against stock HIP and C5 against mainline Vulkan.
Primary low-width cells: C1/C2 against Laurent. C6 is the near-parity
control.

- [ ] For wide MTP, decompose one full cycle into proposal, packed target,
  accept/commit, KV/state, host synchronization, scheduler, and API time.
- [ ] Compare physical batch shape, token-budget use, target-row geometry,
  quant-family ownership, launch count, and accepted-token accounting against
  the external leader.
- [ ] For C1/C2, run teacher-forced proposal parity against the matched
  Laurent route before changing target kernels. Report agreement by category,
  position, and heldout split.
- [ ] Keep acceptance changes, draft-depth changes, and target-cost changes as
  separate mechanisms. Do not let one aggregate rate hide which mechanism
  moved.
- [ ] Compute an optimistic complete-cycle bound for each mechanism. Continue
  only when the bound is at least **1.25x the required wall reduction** for a
  primary cell: at least 35.2% for C7, 28.5% for C8, 18.1% for C5, 14.7% for
  C2, and 31.8% for C1.

Exit: either a named MTP mechanism with a measured bound, or a per-cell
blocker that supersedes the scaling campaign's broad multi-family blocker.

### Z3 — mechanism selection and RED plan

Open implementation only after Z0-Z2 identify a mechanism with enough bound.

- [ ] Declare the arithmetic class (T0/T1/T2/T3), affected layers/shapes,
  stateful surfaces, strict fallback, expected mechanism, and whether the
  candidate can alter downstream discrete decisions.
- [ ] Write the RED test or production-profile numerical gate before the
  implementation when practical. If RED-first is impractical, record the
  reason in the unit worklog entry.
- [ ] Register through the four-axis plugin registry. Do not add backend- or
  quant-specific dispatch branches.
- [ ] Keep kernel signatures on raw device pointers and preserve
  `KVLiveSpans` as the attention/paged-KV ABI.
- [ ] Prebuild HIP artifacts before `rocprofv3` measurements and record the
  expected kernel names.

Exit: an approved, bounded candidate plan with tests and fallback registered
before implementation begins.

### Z4 — implementation and retention

Implement one mechanism at a time. Do not combine a prefill dataflow, MTP
verifier dataflow, and proposal change in one retention decision.

- [ ] Strict candidates meet their declared exact or parent-parity contract.
- [ ] Production T1/T2 candidates meet the complete production gate from
  [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md), including strict-teacher
  numerical quality, determinism, isolation, category/heldout, task, and
  applicable BF16-relative checks.
- [ ] T3 candidates remain explicit experiments until separately authorized;
  they do not become ordinary production defaults through this campaign.
- [ ] Performance claims use the full category suite plus heldouts and a true
  same-protocol AR control where speculative speed is claimed.
- [ ] A retained candidate must be non-regressive on the complete same-suite
  wall, not only on a target subwindow.
- [ ] Automatic serving promotion requires the complete route/economics and
  lifecycle packet. Explicit diagnostic wins do not change automatic K0 policy
  by themselves.
- [ ] Every retained perf claim updates the compact artifact,
  `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and the immutable worklog
  entry with exact commands and host identity.

Exit: each candidate is promoted, retained default-off with a concrete
blocker, or rejected with a compact artifact.

## 6. Campaign close criteria

Close the campaign when one of the following holds for each primary cell:

1. hipEngine meets or beats the external leader under the standardized
   same-host protocol and passes the applicable correctness gate; or
2. a named mechanism is implemented and retained with its validated scope; or
3. a measured upper bound proves the remaining local mechanism cannot close
   the gap.

The campaign may close with blockers. It must not close with untested
recommendations or a claim based on a pre-Y2/Y3 baseline.

## 7. Evidence map

Current references:

- [`QWEN38-GFX1151-SCALING-CAMPAIGN.md`](QWEN38-GFX1151-SCALING-CAMPAIGN.md)
  — closed M/W/Y evidence, punchlist, blockers, and audit ledger.
- [`QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md`](QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md)
  — standardized external matrix, route decisions, and protocol.
- [`EXTERNAL-MTP-BATCHING.md`](EXTERNAL-MTP-BATCHING.md) — commit-pinned
  external verification-batching survey.
- [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) — normative strict,
  production, and batch-invariant contracts.
- [`BENCHMARK.md`](BENCHMARK.md) — evidence policy and anti-gaming rules.
- [`2026-09-01-gfx1151-qwen38-y1-q4-b3w8r3-partial-retained.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y1-q4-b3w8r3-partial-retained.json)
  — latest full-width prefill collateral before Y2/Y3.
- [`2026-09-01-gfx1151-qwen38-y2-q6-shared3r1-retained.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-q6-shared3r1-retained.json)
  — retained Y2 exact Q6 owner.
- [`2026-09-02-gfx1151-qwen38-y3-planar-q6-pair-decode-retained.json`](../benchmarks/results/2026-09-02-gfx1151-qwen38-y3-planar-q6-pair-decode-retained.json)
  — retained Y3 exact planar Q6 pair decode.
- [`2026-09-01-gfx1151-qwen38-c6c8-k1-ten-iteration-closeout.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-c6c8-k1-ten-iteration-closeout.json)
  — retained C6/C8 K1 MTP closeout.
