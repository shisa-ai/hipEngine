---
status: current
owns: Planned Flash-Next serving follow-up for prefill interference, batch host overhead, and shared MTP/AR execution.
---
# Flash-Next serving performance follow-up

State: **in progress**. Execution started 2026-10-19. The same-host baseline
and comparison criteria are recorded in the
[baseline checkpoint](../../worklog/entries/20261019T195056.259790Z-serving-followup-qwen38fn-serving-baseline-bf7ebc.md).
The original assessment below reviewed `3241dbb11645d49a0a442951bdd5203ef474adfd`.

Current outcome per track, from the same-session HTTP comparison in
[`2026-10-21-gfx1151-flashnext-serving-economics-ab.json`](../../benchmarks/results/2026-10-21-gfx1151-flashnext-serving-economics-ab.json)
(baseline `c8bbf0484`, 64 tokens per request, widths 1 and 8):

- **Prefill (P2): shipped on.** Completed-layer time sharing is the default
  `hipengine serve` policy. With eight concurrent requests, AR inter-token
  p95/p99 falls from 267/347 ms to 163/177 ms; aggregate width-8 AR throughput
  costs about 5% (55.6 to 52.7 tok/s). Long-context numerics of the 1024-token
  active chunk are within round-off of unchanged production (candidate versus
  incumbent mean KL 0.0030, below the 0.0047 control floor).
- **Host overhead (H1/H2): shipped, no measurable speedup.** Per-step
  host-to-device copies fall from 40 to 5; direct step time moves less than 0.5%.
  The lone-request batch penalty remains about 5% (batch1 42.5 ms versus c1
  40.4 ms per step).
- **Shared MTP/AR target (M1-M4): implemented, opt-in, slower.** The shared pass
  works through `hipengine serve` (HTTP cancellation/refill and sampled
  lifecycle checks pass), but it loses to separate execution: one explicit MTP
  request 30.3 to 21.0 tok/s, one MTP request beside seven AR peers 48.9 to
  44.0 tok/s. Explicit MTP therefore keeps separate execution by default;
  `HIPENGINE_QWEN4_EXP_SHARED_MTP_TARGET=1` selects the shared pass. The cost is
  eager per-layer dispatch with per-row depth-wave state copies, and AR peers
  leaving the graph-replayed batch step (`docs/REFACTOR.md`).

The production profile's own numerical drift is outside this campaign. Plain c1
production already misses the short-prompt top-1 floor (0.9865 against 0.99)
and the long-context mean/p95 round-off floor; batch and shared-target outputs
are bit-identical to each other and add no drift of their own. The qualitative
54-answer task review recorded in
`benchmarks/results/2026-10-20-gfx1151-flashnext-serving-task-review.json` found
defects in every arm, including strict and production AR controls.

## Objective and scope

Improve concurrent serving in three independent increments:

1. Reduce decode stalls while another request prefills a long prompt.
2. Reduce the host overhead paid by an ordinary autoregressive (AR) request in
   the shared batch, including when it is the only decoding request.
3. Let multi-token prediction (MTP) verification and ordinary AR decode share
   physical target execution without losing request-local commit/rollback.

Keep the current working serving and explicit-MTP routes throughout development.
Use the existing scheduler, request lifecycle, shared owners, plugin registry and
speculative transaction contracts. Do not add a second scheduler or a frontend
GPU owner. Runtime stays torch-free. Kernel changes keep registered strict
fallbacks and follow [OPTIMIZATION](../OPTIMIZATION.md),
[EXECUTION-PROFILES](../EXECUTION-PROFILES.md) and [TESTING](../TESTING.md).

The repository-wide cleanup audit and refactor backlog is **out of scope**.
That exclusion does not waive applicable checks or ownership rules for an
implementation unit. Paging the batch slot's index storage beyond its existing
position cap, broader model/backend certification, new weights or KV formats,
and true prefill/decode stream overlap are also outside this campaign.

## Starting point and evidence

The merged serving path uses a shared AR batch with c1 staging runners for
prefill. Automatic MTP serves an eligible lone request and can demote it to AR
batching when peers arrive; explicit MTP keeps its requested route. Logical
AR/MTP coexistence already exists, but those routes execute separate physical
target work. Candidate-token rows are not independent concurrent requests.

The source-backed findings are recorded in
[REFACTOR](../REFACTOR.md#flash-next-shared-decode-batch-rollback-lever-and-remaining-gaps)
and these immutable worklogs:

- [Batch decode host overhead](../../worklog/entries/20261019T132824.038407Z-lhl-qwen4exp-batch-decode-host-overhead-bce862.md):
  same-host c1/batch timing, graph-segment changes, residual host work and
  graph/eager correctness evidence. This is the source of the reported roughly
  7% lone-request batch penalty, not a fresh measurement on merged `main`.
- [Default shared batch and automatic MTP](../../worklog/entries/20261019T145934.689237Z-lhl-qwen4exp-default-batch-auto-mtp-026414.md):
  shared pools, staging ownership, automatic demotion and actual HTTP routes.
- [Token-budget scheduling](../../worklog/entries/20261019T152241.242390Z-lhl-qwen4exp-token-budget-scheduling-5b63f5.md):
  rotating prefill chunks and decode in each round, including the long-prompt
  interference experiment. Rotation avoids waiting for the entire prompt;
  decoding still waits behind a prefill chunk each round.

Refresh baselines on the implementation host before tuning. Historical zbook
and Framework measurements are separate physical-host lanes, even when both
use gfx1151. Start with the existing `UD-Q4_K_XL`, Q8_0 MTP sidecar, production
profile and BF16 KV configuration; this measurement scope is not an artifact
allowlist or a restriction on implemented capabilities.

## Work packets and initial effort estimates

Estimates are engineering time for one engineer familiar with the code, with
access to the gfx1151 host, including focused tests and real serving validation.
They are planning ranges, not measured budgets or delivery commitments.

| Packet | Scope | Difficulty | Initial estimate | State |
| --- | --- | --- | --- | --- |
| P1 | Decode-aware prefill slicing | Medium | 1–2 days for a simple policy | 64- and 512-token diagnostics miss the frozen latency/cost criteria; P2 is in progress |
| P2 | Adaptive prefill slicing/time share, if needed after P1 | Medium | 3–5 days total for the prefill track, including P1 | Shipped on; width-8 AR inter-token p95/p99 267/347 → 163/177 ms at about 5% throughput |
| H1 | Pack per-step batch metadata | Low–medium | 1–3 days for the first host-overhead increment and validation | Shipped in `f54555875`; copies 40 → 5 per step, step time flat |
| H2 | Capture additional device stages; profile remaining PLE host work | Medium | Re-estimate after H1 profiling | Output-head capture shipped; PLE profile (under 1 ms hashing at width 1) does not justify a rewrite |
| M1–M4 | Shared physical MTP/AR target execution | High | 2–4 weeks for the mixed-execution track; highest uncertainty | Serving integration committed opt-in; slower than separate execution at widths 1 and 8; graph-replayed shared pass open |

P1 and H1 are independent, subject to coordinating edits to shared runner and
scheduler files. Finish and measure them before ranking the larger MTP track.
The adaptive policy and later host work are not prerequisites for beginning
mixed-execution design.

## Prefill track: smaller bounded work while decode is active

### Mechanism

`ResidentEngineLoop._run_token_budget_round` in
`hipengine/generation/engine_loop.py` executes prefill before decode.
`Qwen4ExpGGUFTextGenerator.engine_loop_config_defaults` in
`hipengine/generation/qwen4_exp_gguf.py` sets the prefill budget from the runner's
chunk size. A time check after a large chunk cannot interrupt work already
executing, so a time budget alone does not bound that chunk's stall.

### P1 — decode-aware slices

- Preserve large prefill chunks when no requests are decoding.
- Submit smaller token slices while decoding requests are due, using the
  prepared workspace through `prefill_append`; do not allocate a second owner
  merely to submit a smaller slice.
- Restore the larger slices when decoding drains. Preserve round-robin prompt
  progress, admission, cancellation and explicit operator configuration.
- Record the selected slice, context length, measured chunk time and decode
  occupancy so serving evidence can explain the policy actually executed.

### P2 — adaptive policy, only if P1 leaves a material problem

Use measured chunk duration and context length to choose the next slice and,
if needed, prefill's share of rounds. A decode-latency target must still permit
prefill progress. State the minimum useful slice and the behavior when even
that slice exceeds the target; do not advertise a hard latency guarantee based
only on a completed-chunk timer.

### Acceptance

- Add deterministic fake-clock tests for idle/active transitions, prompt
  progress, operator overrides and policy bounds. The existing short-behind-long
  test in `tests/test_unit_qwen4_exp_resident_serving.py` proves ordering, not
  inter-token latency; extend coverage rather than treating it as a timing gate.
- Validate model state and numerical behavior across changed chunk boundaries,
  ragged final chunks, multiple prompt lengths and dense/sparse attention
  transitions. Preserve exact request/control ownership in every profile.
- Through `hipengine serve`, measure decode inter-token latency (median, p95,
  p99 and maximum), prompt time-to-first-token, prefill throughput and aggregate
  decode throughput beside long prefill. Include idle, one decoding peer and
  multiple decoding peers, plus cancellation and recovery.
- Choose the latency target and acceptable prefill cost before candidate
  timing. If the tradeoff is unacceptable, revise the policy rather than
  hiding prefill regressions in an aggregate score.

## Host-overhead track: avoid repeated metadata and eager launch work

### H1 — one per-step metadata upload

`Qwen4ExpGGUFBatchRunner._execute_step` in
`hipengine/runtime/qwen4_exp_batch_runner.py` prepares each attention and index
owner separately. Attention positions/contexts and index query positions still
cause repeated small uploads. The c1 runner already has packed position/context
storage in `hipengine/runtime/qwen4_exp_runner.py`; use that as an implementation
reference, not as proof that the batch layout is interchangeable.

Pack the batch's validated dynamic metadata into stable storage with owner
views and one upload. Preserve row-to-slot mapping, index cursor meaning,
all-before-mutation validation, graph pointer lifetimes and retry-safe cleanup.
Keep unchanged block-table and slot-index upload suppression.

### H2 — capture and re-profile

Profile H1 before extending capture to the output head and device-side PLE
stages. The PLE n-gram hash and host embedding gather cannot themselves be
removed by GPU graph capture. Preserve the immediate-consumption lifetime of
the pinned staging buffer; asynchronous uploads require explicit lifetime and
stream ordering. Investigate host hashing/gathering only after measuring the
remaining cost. Host-readback MoE graph limitations are a separate possible
increment, not a reason to expand H1.

### Acceptance

- Extend `tests/test_unit_qwen4exp_batch_runner.py`, the attention/index owner
  tests and GPU runner tests with metadata contents, upload count, permutations,
  cancellation/reuse and failure-before-publication cases.
- Compare graph/eager results under the declared arithmetic contract and test
  width transitions, sparse physical slots and dense/sparse attention.
- Measure same-host c1 versus batch width 1 and batch widths 2/4/8 with matched
  prompts, context, profile, controls and warmup. Record direct step time and
  served latency separately; repeat across the category suite and heldouts.
- Confirm `LLM.generate()` or `hipengine serve` selects the changed path and
  that improved width 1 does not conceal wider-batch regressions. Narrowing
  the historical penalty is a target, not a promised recovery of all of it.

## Mixed-execution track: transactional verification beside AR peers

### Why this is larger

The AR batch admits one token per unique request slot. MTP's
`verify_target_block_state_rows` and `commit_target_block_rows` in
`hipengine/runtime/qwen4_exp_runner.py` instead process consecutive tokens for
one request, preserve intermediate GDN recurrent/convolution state, and commit
only the accepted prefix. PLE history, attention and index cursors must match
that prefix. Concatenating these rows with independent-request rows does not
preserve those semantics.

`hipengine/generation/qwen4_exp_serving_mtp.py` exposes one physical speculative
request per cycle. The provider iterator in
`hipengine/generation/qwen4_exp_mtp.py` performs proposal, target verification
and acceptance internally. Removing demotion or raising that width is not a
shared physical implementation.

### Stages

- **M1 — execution contract and provider boundary.** Define request/slot,
  candidate-depth, position and row-group metadata, including ordinary AR rows.
  Separate proposal preparation, scheduler-owned target execution and acceptance
  without changing sampling semantics. Reuse existing speculative interfaces
  where they fit. Specify per-request commit and cancellation before kernels.
- **M2 — shared target work.** Integrate the layout with attention/index, GDN,
  convolution and PLE owners. Select registered kernels that implement the
  required dependencies; add variants where the current kernels cannot do so.
  Keep physical request width distinct from total verification rows.
- **M3 — accepted-prefix commit and lifecycle.** Commit each speculative
  request's accepted prefix while AR peers advance once. Preserve hidden seeds,
  draft catchup, KV/index/state destinations, RNG and grammar state. Rejection,
  EOS, stops, cancellation and a failed step must not corrupt surviving peers.
- **M4 — serving and economics.** Wire the physical path into the existing
  scheduler and telemetry. Exercise explicit MTP with AR peers first, then
  measure automatic-selection choices without assuming mixed speculation is
  profitable at every occupancy or candidate depth.

### Committed prerequisites

- [Provider stages](../../worklog/entries/20261019T201925.053647Z-lhl-qwen4exp-mtp-staged-provider-stages-fc10d4.md)
  separate proposal preparation and acceptance while preserving the existing
  iterator. A real explicit-MTP `LLM.generate_detailed()` smoke passes.
- [Host row layout](../../worklog/entries/20261019T210316.597743Z-lhl-qwen4exp-mixed-layout-af4053.md)
  distinguishes request width from total rows and maps accepted depths to
  request-owned destinations. Its 47 CPU tests pass; it does not execute a
  target or mutate device state.

The [mixed runtime executor](../../worklog/entries/20261019T220625.608622Z-serving-followup-qwen4exp-mixed-target-executor-684a43.md)
now composes GDN, QSA and PLE depth-wave state with shared stateless work.
Its synthetic GPU checks cover rejection, partial/full acceptance,
permutations, sparse continuation and abort. These runtime checks do not close
the scheduler/HTTP lifecycle, production numerical or economics requirements.
The serving integration is committed behind `HIPENGINE_QWEN4_EXP_SHARED_MTP_TARGET`; see the outcome summary above.

### Acceptance

- Test all-reject, each partial-accept depth, full acceptance/bonus, different
  chain lengths, AR peers, row permutations, delayed admission, retirement,
  cancellation, refill and same-schedule repeats. Assert actual state/RNG and
  processor rollback, not just emitted text.
- Extend existing batch-owner and MTP transaction tests; replace their current
  exclusion assumptions only when the corresponding shared route works.
- Run the applicable production strict-teacher numerical, category/heldout task,
  determinism, isolation and dynamic-width gates. Generated-ID equality across
  production widths is not a substitute for these contracts.
- Measure full-cycle latency and aggregate throughput against both true AR and
  the existing separate MTP/AR execution on the same host and workload. Include
  draft/verify/commit costs and AR peers' inter-token latency. Predeclare any
  automatic keep/kill policy and do not tune it to one prompt.
- Prove through `hipengine serve` that MTP verification and AR peers share
  physical target work. Report request widths, candidate rows, accepted tokens,
  route selection and fallback reasons truthfully. Separate passes within one
  scheduler round do not close this track.

## Resume and completion protocol

1. Read this campaign, the linked worklogs and current code; check for work
   completed since this assessment. Establish host ownership and ROCm health
   before GPU work, and refresh baseline commands and source identity.
2. Freeze the packet's workload, correctness contract and comparison metrics.
   Write a failing targeted test where practical, implement one coherent unit,
   and run the narrowest relevant tests plus a real user-surface check.
3. Record each unit in a new immutable worklog entry and commit after validation.
   Working improvements ship on the intended path; new rollback flags require
   a concrete cause and removal condition under the repository rules. Missing
   benchmark coverage is not a runtime capability failure.
4. Put measured results and exact commands in compact benchmark artifacts and
   the required benchmark rollup. Keep the root product README untouched unless
   the lead requests a specific change. No benchmark row is created by this plan.
5. Update packet states with evidence links. A conditional packet may be closed
   as unnecessary after a documented measurement; do not silently drop it.
   Run the milestone gate required by TESTING when closing the campaign, using
   focused repairs rather than automatic broad reruns after isolated failures.

Completion requires an explicit outcome for all three tracks and final-source
serving evidence for implemented changes. A passing audit gate, a serial mixed
request test, or a faster microbenchmark alone does not establish completion.
