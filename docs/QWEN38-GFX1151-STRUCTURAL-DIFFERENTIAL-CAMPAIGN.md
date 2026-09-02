# Qwen3.8-27B gfx1151 Structural Differential Campaign

Status: **opened 2026-09-02; review of the W/Y closures folded in the same
day (section 7) with pre-sized mechanism candidates; Z0 baseline refresh
recorded with a C5 correctness failure, operation attribution partial**
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
| Prefill | C2 | 178.660 tok/s | Laurent 196.824 | **-9.2%** |
| Prefill | C8 | 283.540 tok/s | Laurent 297.325 | **-4.6%** |
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

Review addendum (2026-09-02, section 7): the W closure bounds were computed
against the `<= 1.25x R32/R8` flatness ideal, not against the pass budget the
frozen protocol actually imposes, and W closed hours before Y2 built the exact
single-sweep Q6/Q5 bodies it needed; those bodies were never registered on
the verify side. The premise "the local same-dataflow ladder is closed"
therefore holds for the prefill ladder, not yet for the verify path, and
section 7 pre-sizes the mechanisms Z0-Z3 should start from.

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

- [x] Re-run the standardized C1-C8 prefill, true-AR, and explicit MTP
  diagnostics at current head on the physical gfx1151 host.
- [x] Record exact commands and raw-source hashes for every run. The Y1
  matrix is the latest full-width prefill collateral but predates Y2/Y3; do
  not use it as the campaign's final current-head baseline.

  Current-head checkpoint: [`2026-09-01-gfx1151-qwen38-z0-current-head-baseline.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-z0-current-head-baseline.json).
  On physical host `gfx1151` at source `d61c5817f`, prefill was exact in
  80/80 cells with MTP disengaged. True AR was self-exact in 80/80 cells.
  Explicit MTP engaged and conformed to the configured budget in 80/80
  cells, but all ten C5 cells failed `mtp_self_exact` and `ar_mtp_equal`;
  C1-C4 and C6-C8 passed 70/70. The raw hashes, exact commands, model and
  prompt provenance, production manifest hashes, and corrected subgroup
  cycle accounting are in the checkpoint artifact. The observed C5 4+1
  subgroup split localizes the next investigation but does not establish a
  root cause. No performance candidate or public number was retained.
- [x] Collect current-head C2/C8 prefill and C5/C7/C8 MTP operation-complete
  attribution: kernel family, launch count, host/API/copy time, proposal,
  target, accept/commit, KV, scheduler, and server overhead.

  Partial checkpoint: [`2026-09-01-gfx1151-qwen38-z0-operation-attribution.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-z0-operation-attribution.json).
  The exact C2/C8 grouped-row grid is GPU-bound at every prompt shape: wall
  minus HIP-event span is below 0.04 ms, with no memory-copy engine event in
  any tick. At correctness-passing C7/C8, operation wall is 647.6/661.9 ms
  per average request cycle; named telemetry stages account for 439.3/439.6
  ms, leaving 208.3/222.4 ms outside those stages. Q4/Q5/Q6, attention/KV,
  state-commit, API, and copy-kernel time and launch counts are durable in the
  artifact. C5 remains attribution-only because it reproduced the baseline
  correctness failure. The C6/C8 K1 reconciliation below closes the remaining
  scheduler/server/inter-group question with a measured physical-group-entry
  bound; it does not claim to separate transition time from cross-run prefill
  error beyond that bound.
- [x] Attribute at launch granularity, not family granularity: one R16, R24,
  and R32 target pass each, naming owner, grid, and per-launch ms for every
  Q4/Q5/Q6 launch, with the Q6 lm-head, Q5 `ssm_out`, planar/standard direct
  GEMV, and rowtile-chunk launches split out (section 7, F3/F4). Reconcile
  against W0's family totals and W1's window numbers, which do not reconcile
  with W1's own per-launch results.

  Launch ledger: [`2026-09-01-gfx1151-qwen38-z0-r16-r24-r32-launch-ledger.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-z0-r16-r24-r32-launch-ledger.json).
  The current-head one-group diagnostic passes all R16/R24/R32 correctness,
  route, and budget checks. At R32 the binding launches are Q6 planar direct
  grid `[40960,32,1]` (32 launches, 330.7 ms/pass, 10.32 ms median/launch),
  Q6 standard direct `[81920,32,1]` (24, 142.6 ms, 5.94 ms), Q5 `ssm_out`
  selected-direct `[40960,32,1]` (48, 104.7 ms, 2.18 ms), and Q6 lm-head
  rowtile `[3973120,1,1]` (8, 37.9 ms, 4.74 ms). R24 uses the same direct
  owners at grids Y=24; R16 stays on rowtile/WMMA owners. This resolves the
  W0/W1 mismatch: family totals mixed distinct direct, rowtile, and lm-head
  launches and cannot substitute for the launch ledger. The diagnostic
  override is run-owned; production grouping is unchanged.
- [x] Reconcile decode-only cycle wall against the stage sum at C6/C8 K1,
  using the prefill tick from telemetry, and localize the residual (F6:
  ~57 ms per C8 cycle) to host, batch window, proposal sync, or untraced
  kernels.

  Reconciliation: [`2026-09-01-gfx1151-qwen38-z0-c6c8-k1-cycle-reconciliation.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-z0-c6c8-k1-cycle-reconciliation.json).
  A current-head full-suite refresh passes 20/20 exact, engaged, and K1-budget
  cells. After subtracting the matching grouped prefill-only tick and all
  named stages, the residual is **50.7 ms/cycle at C6** and **58.4 ms/cycle
  at C8**, reproducing F6. The two physical-group entry gaps exceed the one
  grouped prefill tick by 52.5/61.2 ms/cycle, explaining the residual within
  1.8/2.8 ms/cycle. Inter-cycle gaps match proposal telemetry within
  0.52/1.11 ms/cycle; traced kernels cover 95.7%/94.9% of cycle windows and
  no DMA copy occurs there. The named bound is therefore physical-group
  prompt streaming/transition plus matched-prefill cross-run error, not the
  50 ms batch window, proposal synchronization, or an untraced GPU family.
  Current telemetry cannot partition transition from prefill error further.
- [x] Record per-tick row composition for the C2 prefill protocol from
  `scheduler_token_chunks`/`prompt_lengths` and state whether the two prompts
  share a tick (F9). This is a baseline fact, not a mechanism.

  The two C2 requests share one grouped prefill tick in all ten cells. Their
  starts differ by 0.19-1.03 ms and completions by 0.24-0.79 ms. Per-request
  prompt lengths are 35-67 tokens; grouped ticks are 70, 72, 78, 86, 92, 96,
  120, and 134 rows. The exact-grid trace covers every shape; this admission
  fact does not itself establish mechanism C's speedup.
- [x] Publish the pass budgets per section 7.2 for C5/R20, C6/R24, C7/R28,
  C8/R32, and the C2 grouped-tick budget for 211.888 tok/s, next to the
  prefill share of each complete wall (F5).

  Budget artifact: [`2026-09-01-gfx1151-qwen38-z0-pass-budgets.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-z0-pass-budgets.json).
  The current-head one-group K3 diagnostic passes 40/40 exact, engaged, and
  budget-conformant cells. Its measured non-target stage already exceeds the
  external-parity cycle allowance at every wide cell: even with a zero-cost
  target pass and the non-stage residual removed, C5/C6/C7/C8 remain short by
  135/198/339/412 ms per cycle. Target-only mechanisms A/B therefore have a
  **zero feasible pass budget** and must follow an accept/non-target dataflow
  mechanism. This supersedes the provisional positive C8 budget in section
  7.2, which combined K3 cycle economics with the old K1 accept stage. C2
  needs 330.4-632.4 ms grouped ticks by prompt to reach 211.888 prompt tok/s;
  the current grouped suite is 178.660 tok/s and needs 15.68% wall reduction.
- [x] Publish one compact Z0 artifact and update the benchmark rollup only if
  a retained public number changes.

  Z0 closure index: [`2026-09-01-gfx1151-qwen38-z0-closure.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-z0-closure.json).
  It binds the five committed evidence packets and their hashes, correctness
  scopes, measured findings, route decisions, and checklist verdicts. No
  retained performance candidate or public number changed, so
  `benchmarks/README.md` and `benchmarks/CHANGELOG.md` are intentionally
  unchanged. Z0 is complete; automatic serving remains width-4 fail-closed.

Exit: one same-host, current-head matrix and attribution set that can serve
as the denominator for every later candidate, plus the published per-cell
pass budgets.

### Z1 — prefill external differential

Primary cells: C2 and C8 against Laurent Vulkan. Secondary cells: C6/C7.
Carry C1/C3/C4/C5 as regression controls, not targets.

- [x] Run hipEngine and Laurent under the same model, prompt suite, one-output
  prefill boundary, physical host, and concurrency schedule.

  Matched parity: [`2026-09-01-gfx1151-qwen38-z1-laurent-prefill-parity.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-z1-laurent-prefill-parity.json).
  On barrier-to-last-completion wall, current hipEngine reaches 178.660 versus
  Laurent 196.824 prompt tok/s at C2 (**-9.23%**) and 283.540 versus 297.325
  at C8 (**-4.64%**). Both use the same 17.1 GB model hash, ten prompts,
  one-output boundary, no prompt cache, and barrier-released C2/C8 schedule on
  physical host `gfx1151`; all correctness/repetition checks pass. Laurent's
  internal `prompt_ms` reports 212.057/308.340 tok/s, but those values are not
  compared against hipEngine complete wall. This corrects the earlier
  211.888/305.847 cross-boundary targets without attributing a mechanism.
- [x] Compare complete wall, kernel-family time, launch counts, row shapes,
  weight-byte movement, dequant strategy, LDS/register pressure, occupancy,
  and final-head behavior.

  Operation differential: [`2026-09-01-gfx1151-qwen38-z1-laurent-prefill-operation-differential.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-z1-laurent-prefill-operation-differential.json).
  On the canonical prompt, Laurent is only 1.01x faster in complete wall but
  1.18x/1.25x faster in summed C2/C8 GPU-node time. Quantized Q4/Q5/Q6 time
  explains the difference: hipEngine/Laurent is 1.28x at C2 and 1.31x at C8.
  Laurent has **more** node dispatches (3,588 versus 1,955 classified hipEngine
  dispatches), chunks rows72 as 64+8 and rows288 as 256+32, and executes the
  final Q6 head once at 4.57/5.63 ms. Its large/small Q4 F16-B pipelines use
  VGPR120/108, LDS20/5 KiB, scratch0, with no spills. `rocprofv3` emits no
  RADV kernel/copy trace here; exact occupancy and weight-fetch bytes are a
  named instrumentation bound. Do not infer them from the resource proxy.
- [x] Read the pinned Laurent implementation and cite source file plus commit
  for any mechanism considered for porting. Do not vendor or edit the external
  repository.

  At `LaurentZuijdwijk/llama.cpp@c28d538df`,
  `ggml/src/ggml-vulkan/ggml-vulkan.cpp:4002-4019,9647-9686` defaults
  quantized dense `MUL_MAT` to an F16 activation-B operand. The source states
  that this halves B-operand bytes and `buf_b` shared memory while preserving
  the existing F16 staging boundary; perf-node names confirm fused Q4/Q5/Q6
  matmul. This is activation/dequant dataflow, not quantized-weight reuse.
  The external tree remained clean and read-only.
- [x] Produce a mechanism table that separates measured facts from inferred
  causes.
- [x] Compute an optimistic complete-wall bound for each mechanism. Continue
  only when the bound is at least **1.25x the required wall reduction** for a
  primary cell: at least 33.6% for C2 and 27.1% for C8.
- [x] Treat C2 as an M = 35-96 owner problem plus an admission question, not
  a high-row problem: Z0's tick-composition fact decides whether grouped
  prefill (mechanism C in section 7.3) applies, and the M = 17-48 owners
  shared with the verify side (mechanisms A/E/F) are the kernel candidates.

  Bound artifact: [`2026-09-01-gfx1151-qwen38-z1-mechanism-bounds.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-z1-mechanism-bounds.json).
  Only F16 activation-B staging advances: transferring Laurent's measured
  quant-time delta while holding all other wall fixed gives optimistic
  19.8%/20.1% C2/C8 wall reductions, above the matched-gap 1.25x thresholds
  of 11.5%/5.8%. This is an inferred T2 portability bound, not a measured
  hipEngine win. Chunking and launch reduction are rejected as mechanisms.

Exit: either a named prefill dataflow candidate with a measured bound, or a
stronger impossibility result that closes the remaining prefill gap.

### Z2 — MTP external differential

Primary wide cells: C7/C8 against stock HIP and C5 against mainline Vulkan.
Primary low-width cells: C1/C2 against Laurent. C6 is the near-parity
control.

- [x] For wide MTP, decompose one full cycle into proposal, packed target,
  accept/commit, KV/state, host synchronization, scheduler, and API time.

  Current-head decomposition: [`2026-09-01-gfx1151-qwen38-z2-wide-cycle-decomposition.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-z2-wide-cycle-decomposition.json).
  C5-C8 K3 complete wall reconciles exactly from grouped prefill/admission,
  proposal, packed target, accept/synchronization/commit, provider update,
  selected-state commit, and the measured scheduler/API residual. The dominant
  owner is accept/synchronization/commit at 490.8-667.1 ms/cycle, versus
  34.7-76.4 ms target and 28.0-36.7 ms proposal. `accept_ms` is an aggregate
  synchronization boundary, not standalone accept-kernel time. K1 traces
  constrain host/API attribution but are not substituted into K3.
- [x] Compare physical batch shape, token-budget use, target-row geometry,
  quant-family ownership, launch count, and accepted-token accounting against
  the external leader.

  Comparison: [`2026-09-01-gfx1151-qwen38-z2-external-shape-comparison.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-z2-external-shape-comparison.json).
  Both routes use K3 and D24. hipEngine's measured target rows are
  C5 `{10,15,20}`, C6 `{12,18,24}`, C7 `{14,21,28}`, and C8 `{16,24,32}`;
  smaller values are tail passes. Its acceptance is 78.89%, versus leader
  acceptance of 81.01%, 70.67%, 77.91%, and 80.51%. External HTTP telemetry
  does not expose physical rows, quant-family kernels, or launch counts, so
  those cross-runtime fields are a named instrumentation bound and remain
  null. No geometry or ownership equality is inferred.
- [x] For C1/C2, run teacher-forced proposal parity against the matched
  Laurent route before changing target kernels. Report agreement by category,
  position, and heldout split.

  Capture protocol: [`2026-09-01-gfx1151-qwen38-z2-teacher-forced-proposal-plan.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-z2-teacher-forced-proposal-plan.json).
  Existing external HTTP artifacts are free-running aggregates and cannot close
  this item. The pinned fork exposes raw response tokens and verbose per-rank
  MTP proposals, but not `LLAMA_MTP_TOKEN_TRACE`; the next unit must segment
  verbose records by request and compare first-cycle K3 proposals on a
  Laurent-owned D24 teacher trajectory. `llamacpp_mtp_draft_trace.py` now
  records request indices, prompt-token counts, and per-request draft-call
  counts with host-only RED/GREEN coverage. `gguf_mtp_bench.py` also accepts
  `--prompt-token-ids` to preserve an exact teacher-owned context without a
  decode/re-encode or chat-template round trip. No target-kernel change is
  admitted before the full capture. A pinned raw-token smoke
  ([artifact](../benchmarks/results/2026-09-01-gfx1151-qwen38-z2-laurent-raw-proposal-smoke.json))
  shows that Laurent emits no draft call for `n_predict=1`; `n_predict=4`
  returns the target root plus a three-token proposal. The capture therefore
  retains only the first K3 proposal and discards the free-running response.
  Concurrent C2 log segmentation resolves draft calls through candidate
  `seq_id` to slot/task ownership rather than the most recent prompt marker;
  an interleaved two-slot RED/GREEN test pins this behavior. The Laurent half
  is now durable
  ([artifact](../benchmarks/results/2026-09-01-gfx1151-qwen38-z2-laurent-teacher-proposal-capture.json)):
  all 720 physical requests expose exactly nine first-proposal candidates. Of
  240 contexts per width, 239 preserve the teacher root. C1 diverges at
  `code_markdown_table` position 13 (6943→83889); C2 diverges at
  `general_en_explain` position 20 (7255→191280). Those contexts are bounded
  non-comparable rather than counted as proposal disagreement. hipEngine
  capture is still required before this item closes. A production-route raw
  token smoke
  ([artifact](../benchmarks/results/2026-09-01-gfx1151-qwen38-z2-hipengine-raw-proposal-smoke.json))
  confirms exact prompt accounting and diagnostic device-proposal
  materialization. Initial C1 and both C2 rows exactly match Laurent's first
  K3 `[12305, 198, 727]`. The diagnostic adds a synchronization/readback and
  carries no timing claim.

  Final parity: [`2026-09-01-gfx1151-qwen38-z2-teacher-forced-proposal-parity.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-z2-teacher-forced-proposal-parity.json).
  On contexts where both target roots equal the fixed Laurent teacher, exact
  K3 agreement is 147/222 (66.22%) at C1 and 304/446 physical rows (68.16%)
  at C2. Depth-1/2/3 agreement is 89.19/77.03/69.37% at C1 and
  90.13/78.48/71.75% at C2. All 240 C2 pairs are internally identical, so
  batching does not alter proposal IDs. Eighteen C1 and seventeen C2 contexts
  are root-non-comparable and are reported separately, not scored as proposal
  disagreement. The category, heldout, teacher-position, and draft-depth
  breakdown is in the artifact. Proposal mismatch is a measured acceptance
  mechanism and must remain separate from target cost.
- [x] Keep acceptance changes, draft-depth changes, and target-cost changes as
  separate mechanisms. Do not let one aggregate rate hide which mechanism
  moved.
- [x] Compute an optimistic complete-cycle bound for each mechanism. Continue
  only when the bound is at least **1.25x the required wall reduction** for a
  primary cell: at least 35.2% for C7, 28.5% for C8, 18.1% for C5, 14.7% for
  C2, and 31.8% for C1.
- [x] Use the pass budget from section 7.2 as the wide-cell entry condition
  in place of W1's flatness gate. Z0 now measures a **zero target-only pass
  budget**: current one-group K3 accept/non-target time exceeds the complete
  cycle allowance before R20-R32 target work. Start the wide-cell differential
  from the external accept/non-target dataflow; recompute A/B/E target budgets
  only after that mechanism has a measured complete-cycle bound. Stock HIP's
  56.222 tok/s at C8 K3 remains the existence proof that both accept and
  target costs can fit on this host.

  Mechanism ledger: [`2026-09-02-gfx1151-qwen38-z2-mechanism-bounds.json`](../benchmarks/results/2026-09-02-gfx1151-qwen38-z2-mechanism-bounds.json).
  Bounds are never added. Acceptance quality clears only C1 (49.39% optimistic
  complete-wall reduction); C2 already exceeds Laurent's measured acceptance.
  Draft-depth clears only C2 (17.79% versus 14.7%). Target-cost clears no
  primary cell. Removing the complete measured accept/synchronization/commit
  boundary is a 40.38-65.18% upper bound and clears every primary cell, so
  accept/non-target dataflow advances as the structural Z3 candidate. This
  upper bound does not yet identify a safe implementation. The measured zero
  target-only C5-C8 budget rejects wide target-kernel work until that dataflow
  has a measured complete-cycle result.

Exit: either a named MTP mechanism with a measured bound, or a per-cell
blocker that supersedes the scaling campaign's broad multi-family blocker.

### Z3 — mechanism selection and RED plan

Open implementation only after Z0-Z2 identify a mechanism with enough bound.
Section 7.3 is the starting ledger; Z0-Z2 confirm or reject each row's bound
before it enters here.

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

## 7. Review of the W/Y closures (2026-09-02) and pre-sized mechanisms

Reviewer pass over the scaling campaign's sections 8-9, the W0-W7/Y0-Y5
entries, the retained raw W0/W1 traces still under `/tmp` (hashes in the W0
and W1 artifacts), and the external survey. Every number is read from a
committed artifact or derived from one; derived values are labelled; none is
a performance claim. Z0 re-measures all of them at current head.

### 7.1 Findings

**F1 — the flatness gate was sufficient, not necessary.** From the frozen
complete-wall protocol
([W0 raw suite](../benchmarks/results/2026-09-01-gfx1151-qwen38-w0-sweep-economics.json),
C8 cell `code_merge_intervals`, 36-token prompt x 8): AR cell wall 3.726 s,
of which the rows288 prefill tick is ~0.99 s
([Y5](../benchmarks/results/2026-09-02-gfx1151-qwen38-y5-nongemm-tail-closure.json)
measures 990.82 ms), leaving 114 ms per AR decode tick — equal to the R8
forward (W0 row curve: 118.6 ms host / 111.8 ms kernel). A 1.15x-own-AR MTP
cell must finish in 3.240 s, i.e. 2.25 s of decode; K3 at the historical
78.894% acceptance commits 3.37 tokens/request/cycle, so 24 tokens need 7.12
cycles at `<= 316 ms`. Subtracting three proposal steps (3 x 10.8 ms, W0
stage telemetry), provider/commit (7.7 ms), and the non-stage residual (F6,
57 ms) leaves an R32 target pass of **~219 ms (residual kept) to ~276 ms
(residual removed)** against the current **790 ms** (W0 R32 kernel median).
Required R32/R8 is ~2.0-2.5x, not 1.25x. W1 rejected its candidate at 3.43x
without computing this budget; W3 and W7 closed on W1's gate by dependency.
Both dependency closures are conditionally void.

**F2 — W closed before Y built the owners W needed.** W0's `Next` said "build
one shared B-stationary M-loop tile mechanism, start with Q6". W1-W7 closed
between 03:59 and 04:33 UTC on 2026-09-01; Y2's exact single-sweep Q6 owners
for rows33-48 (`<3,1,2>` standard and planar,
[artifact](../benchmarks/results/2026-09-01-gfx1151-qwen38-y2-q6-shared3r1-retained.json))
landed at 10:12 the same day, followed by the rows49-96 and rows256+ Q5/Q6
bodies. None was registered under a verifier key or run through the R8-R32
row curve. `GGUF_T16_TARGET_VERIFIER_WIDE_Q6_SHARED4_VARIANTS` still points
R20-R32 at the pre-Y2 `shared4` body on three shapes only.

**F3 — verify R20-R32 and prefill rows33-48 are the same GEMMs on the same
T16 tensors, owned by two ladders that differ by up to ~9x.** Per full target
pass (W0 row curve versus Y0/Y2 prefill sizing; derived per-pass sums use the
model inventory of 33 Q6 `ffn_down` K17408/N5120, 24 Q6 `attn_qkv`
K5120/N10240, 9 Q6 `attn_v` K5120/N1024, 48 Q5 `ssm_out` K6144/N5120, and
the Q6 lm-head 248320x5120):

| Family | Verify R8 | Verify R32 (today) | Prefill rows33-48 owner (derived per pass) |
| --- | ---: | ---: | ---: |
| Q4 (10.5 GB) | 60.7 ms | 137.8 ms, flat R20-R32, fetch 0.93x | ~110-130 ms (Y0 multiplicity 1.30-1.34) |
| Q5 (1.04 GB) | 8.3 ms | 104.7 ms, fetch 33x | ~30 ms (rocBLAS-dequant route, multiplicity 1.0) |
| Q6 (4.45 GB) | 32.9 ms | 515.6 ms, fetch 21x | ~50 ms (33x0.96 + 24x0.70 + 9x0.10 ms Y2 leaves) + lm-head |
| other | ~10 ms | ~31 ms | — |
| **pass** | **112 ms** | **790 ms** | **~230-260 ms** |

The prefill-owner column lands inside the F1 budget with no new kernel. Above
R16 the verify path drops its Q6 planar/standard tensors to per-row direct
GEMVs (`q6_k_t16_qmicro_planar_gemv_bf16` gridY=rows at 6.5-7.8 ms/launch,
`q6_k_t16_gemv` gridY=20/24 at 3.7-4.5 ms) and re-sweeps Q5 `ssm_out` 33x;
both are bandwidth-bound on refetched bytes (~225 GB/s). That is why W1's
per-launch two-wave results (0.71/0.38/0.11 ms at R32) beat the owners they
replaced by 10-50x while the *family* did not flatten: the family still held
launches the candidate never owned.

**F4 — the Q6 lm-head is re-swept per row-tile chunk above R8.** In the
retained W1 trace (`/tmp/wy-w1-rowcurve-two-wave/`, sha256 in the
[W1 row-curve artifact](../benchmarks/results/2026-09-01-gfx1151-qwen38-w1-q6-two-wave-rowcurve.json);
whole-process one-prompt diagnostic), the
`q6_k_t16_qmicro_planar_gemv_rowtile_col8_kernel<float, N>` launches with
grid 3,973,120 = 248,320 vocab / 8 columns x 128 threads are the 1.04 GB
lm-head at rowtile N = 2..8: 4.6-6.7 ms per launch, one full sweep per 2-8
rows (~155-225 GB/s), 6.75 s across the run. At R32 that is ~4 sweeps
(~23 ms) where one WMMA sweep is ~6-8 ms. W0 counted it inside "Q6 family"
flatness; W1's pipeline bound reasons only about the two-wave body and cannot
see it.

**F5 — prefill is 27-37% of the C8 D24 complete wall on both arms.** C8 cells
run 35-67-token prompts x 8 = rows280-536 prefill ticks (0.99-1.64 s) inside
3.73-4.50 s AR cells. So (a) the "AR-step equivalent" metric amortizes
prefill into the AR step and stage sums cannot be compared to it directly;
(b) every retained prefill win raises both AR and MTP absolute C6/C8 rates
and the external-parity position while barely moving the ratio; (c)
decode-only AR at C8 is ~114 ms/tick = the R8 forward at ~188 GB/s actual
fetch, so AR C3-C8 is at its forward time and the AR non-goal stands.

**F6 — ~57 ms per K1 C8 cycle (~21% of decode wall) is outside every
telemetry stage.** MTP cell 4.182 s - 0.99 s prefill = 3.19 s over 11.5
cycles = 277 ms/cycle, versus the W0 stage sum target 33.4 + accept 167.8 +
proposal 10.8 + provider 4.7 + commit 3.0 = 219.7 ms. C6 shows a similar
~40-50 ms gap (rows216 prefill tick not separately measured). W0's
host-minus-kernel bound (15.5 ms) covers the target pass only; M2b-M2i closed
host explanations for the accept window, not for the cycle. Candidates:
proposal-side host sync/argmax, engine-loop tick overhead, the protocol's
50 ms `batch_window_ms` if re-armed per cycle, Python per-request
bookkeeping. Worth ~0.66 s per C8 cell alone (derived: K1 43.7 -> ~50 tok/s).

**F7 — P3 closed integer MMQ for all M from a rows256 screen.** The regime
where MMQ's dequant removal pays (M = 16-48) was never screened, and it is
the regime of both open targets. P3 measured only rows256 against the
large-M `selected-wmma` body; the M = 17-48 competitors are the direct
GEMV/shared-B WMMA owners in F3. Stock HIP llama.cpp's R32 verify runs its
MMQ path and reaches 56.222 tok/s at C8 K3 on this host and model
([survey](QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md)). In-tree `mmq32_q8_1_*`
producer bodies (gfx1100-only, rejected WPF-1B) and the gfx1151
`mmq128x32`/`mmq64x64` `ge512` bodies exist to screen from.

**F8 — Y3 optimized the schedule of the LDS-staged body, never the
staging.** Y3's ISA analysis shows the structure: per K256 slab, cooperative
decode -> LDS -> barrier -> WMMA -> barrier, 326 decode instructions with zero
WMMA between two barriers. Producer waves, quartet decode, and shared-byte
decode all lost inside that structure. The Marlin-style alternative — repack
weights at load into WMMA B-operand lane order so each lane decodes its own
16-K strip in VGPRs and feeds `v_wmma` directly, no LDS, no barrier — was
never tried; [`MARLIN.md`](MARLIN.md) covers only a rows==1 layout. It is T2
by construction and fits section 3.

**F9 — P1's C2 grouping verdict is not supported by its own artifact.** P1
recorded grouped rows72-96 ticks at 408-440 ms versus 2 x 278 ms serial (a
+26-36% C2 bound), then measured C2 at -3.0% and declared scheduling null
([P1](../benchmarks/results/2026-08-31-gfx1151-qwen38-prefill-c2-scaling-blocker.json)).
The artifact records no per-tick row composition for the benchmark run, so it
cannot show the two C2 prompts ever shared a tick under the frozen protocol
(barrier-released HTTP requests, an equal-length grouping gate, a server batch
window defaulting to 0 ms). The reviewed C2 = 139.8 tok/s equals 80 tokens /
(2 x 278 ms + overhead), i.e. serial. Post-Y2 grouped ticks (rows72/96 at
396/423 ms) put a grouped C2 at ~180-190 tok/s before any owner work.

**F10 — Y5's closure bound assumed the exact-only ladder.** Y2's first Q6
one-sweep body was 29x faster at the leaf and failed strict parity (max
difference 0.0078125 on 0.6% of outputs); it was reverted with "may reopen
as T2" and the exact variant recovered 1.95x. Section 3 already settles this:
Z candidates declare T2 up front instead of exhausting the exact ladder.

### 7.2 Pass budgets (current-head result)

Z0 invalidates the provisional target-only budget derived in F1. That
calculation combined a hypothetical one-pass **K3** cycle with W0's much
smaller **K1** accept stage. The current-head one-group C5-C8 K3 suite passes
40/40 correctness/route/budget cells and measures the complete non-target
stage under the intended R20/R24/R28/R32 geometry.

| Cell | External target | Allowed decode cycle | Non-target stage | Residual | Target-pass slack, kept / removed | Prefill share, AR / one-group MTP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C5/R20 | 32.713 tok/s | 391.7 ms | 526.7 ms | 67.8 ms | **-202.8 / -134.9 ms** | 26.3% / 16.6% |
| C6/R24 | 37.154 tok/s | 400.3 ms | 598.0 ms | 80.6 ms | **-278.3 / -197.7 ms** | 29.0% / 16.9% |
| C7/R28 | 46.084 tok/s | 350.7 ms | 689.4 ms | 84.9 ms | **-423.6 / -338.7 ms** | 29.9% / 16.6% |
| C8/R32 | 56.222 tok/s | 302.6 ms | 714.4 ms | 74.4 ms | **-486.2 / -411.8 ms** | 31.5% / 17.1% |

A positive slack would be the target-pass budget. Every slack is negative,
even after removing the residual and assigning zero time to the target pass;
the magnitude is the minimum non-target reduction still required. The
1.15x-own-AR budgets are also negative in every cell. Therefore A/B/E target
owner work cannot enter alone: Z2 must first identify an accept/non-target
mechanism, then recompute the pass budget from its measured complete cycle.
Automatic serving remains the production width-4 fail-closed route.

For C2, the exact grouped rows 70-134 require per-prompt tick budgets of
330.4-632.4 ms to reach Laurent's 211.888 prompt tok/s. Current grouped server
wall is 178.660 prompt tok/s and needs 15.68% aggregate wall reduction;
prefill is 19.2% of current C2 AR complete wall and 30.2% of current C2 MTP
complete wall. Exact per-prompt current server/direct ticks and budgets are in
the Z0 budget artifact.

### 7.3 Pre-sized mechanism candidates (Z3 starting ledger)

| Mech. | Description | Class | Cells | Sized bound (derived) | Prerequisite |
| --- | --- | --- | --- | --- | --- |
| A | Register retained exact prefill owners (Y2 `<3,1,2>` Q6 standard/planar, prefill Q5 one-sweep route, best Q4 rows17-48 owner) under verifier R17-R32 keys incl. mixed R20/R24/R32 packed subshapes; strict fallback = current owner | T0 registry transfer (T2 where the owner already carries it) | C5-C8 MTP | Target-only entry **blocked**: Z0 gives zero feasible pass budget until accept/non-target dataflow is reduced | Z2 accept differential |
| B | Q6 lm-head as one sweep at R > 8 (Y2 standard body or dense WMMA lm-head path) | T0/T2 | C5-C8 MTP, prefill | ~15-20 ms per R32 pass remains a measured target sub-bound, but cannot close complete wall before the Z0 non-target blocker | Z2 accept differential |
| C | Grouped C2 prefill on the benchmark path (ragged grouping or the declared batch window) | control/admission | C2 prefill, AR C2 | +26-36% C2 (F9); ~15-23% wall | Z0 tick-composition fact |
| D | Cycle residual outside GPU stages | host/scheduler | C5-C8 MTP | ~57 ms x 11.5 cycles = ~0.66 s per C8 cell (F6) | Z0 reconciliation |
| E | Integer MMQ (Q8_1 activations) for M = 17-48 on the three Q6 shapes, `ssm_out`, two binding Q4 shapes; screen against A, not `selected-wmma` | T2 | C2 prefill, C5-C8 MTP | anchor: stock HIP R32 verify at ~2.5x its R8 (F7) | A/C measured |
| F | Fragment-direct WMMA body (load-time repack to B-lane order, per-lane VGPR decode, no LDS/barriers) on the Y3 planar-down shape at rows32 and rows288 | T2 | C2/C8 prefill, C5-C8 MTP | opens only if E fails; Y3 ISA artifact is the before-picture (F8) | E |

Order: `Z0` (with the F3/F4/F6/F9 attribution items) -> A and C in
parallel (cheapest units relative to bound: a registry transfer of bodies
that already passed their gates, and a telemetry check) -> B and D as Z0
sizes them -> E, then F, conditional. One-pass K3/R32 and K2/R24 at C5-C8
with the A owners is the first Z4 measurement; width x depth admission stays
prompt-independent and automatic serving stays K0 until the complete
production/lifecycle gates pass.

## 8. Evidence map

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
- [`2026-09-01-gfx1151-qwen38-w0-sweep-economics.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-w0-sweep-economics.json)
  — W0 stage telemetry, R8-R32 family row curve, fetch multiplicities, and
  the raw-suite cell walls used in section 7.
- [`2026-09-01-gfx1151-qwen38-w1-q6-two-wave-rowcurve.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-w1-q6-two-wave-rowcurve.json)
  — W1 two-wave candidate row curve and the retained raw trace hash used for
  F3/F4.
- [`2026-08-31-gfx1151-qwen38-prefill-c2-scaling-blocker.json`](../benchmarks/results/2026-08-31-gfx1151-qwen38-prefill-c2-scaling-blocker.json)
  — P1 C2 blocker whose grouping verdict F9 reopens.
- [`2026-09-02-gfx1151-qwen38-y5-nongemm-tail-closure.json`](../benchmarks/results/2026-09-02-gfx1151-qwen38-y5-nongemm-tail-closure.json)
  — rows288 tick wall used for the prefill share in F1/F5.
