# SPECDEC2 / MTP2 Implementation Plan

- Status: **approved; active implementation campaign**
- Approved: **2026-08-24**
- Primary hardware lane: **`hip_gfx1151` / AMD Radeon 8060S Graphics**
- Primary product target: **Qwen3.8-27B Q4_K_S, BF16 KV**
- First provider: **dense GGUF NextN/MTP2**
- Deferred portability lane: **`hip_gfx1100` after backend-neutral and gfx1151 closure**
- Normative dependencies: [`PLAN.md`](PLAN.md),
  [`CONCURRENCY2.md`](CONCURRENCY2.md),
  [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md),
  [`TESTING.md`](TESTING.md), [`BENCHMARK.md`](BENCHMARK.md), and
  [`KERNELS.md`](KERNELS.md)
- Research and alternatives: [`SPECDEC2-RESEARCH.md`](SPECDEC2-RESEARCH.md)
- Existing native-cycle evidence: [`NATIVE_SPEC_CYCLE.md`](NATIVE_SPEC_CYCLE.md)
- Existing real-world MTP rejection/control packet: [`MTP-FIX.md`](MTP-FIX.md)

This document is the resumable implementation ledger for SPECDEC2. It converts
`SPECDEC2-RESEARCH.md` from a research proposal into an approved sequence of
code, correctness, performance, and product gates. Checked items are complete
and must not be redone without evidence that a later change invalidated them.
Unchecked items are not complete merely because a nearby primitive or historical
benchmark passed.

## 1. Executive decision

hipEngine will implement one scheduler-owned continuous speculative execution
path named **SPECDEC2**. **MTP2** is its first provider. AR, MTP, DFlash, and
future chain/tree providers do not receive separate request schedulers or output
owners.

The common target abstraction is a bounded **target frontier**:

- every due request contributes one committed root row;
- AR is a root-only frontier with `K=0`;
- MTP/DFlash chains add candidate descendants;
- tree providers use the same parent-indexed rows;
- proposal providers execute in compatible groups;
- compatible request graphs lower into one or more physical target batches;
- acceptance and selected state/KV commit remain per request; and
- the policy selects `K=0` before mutation whenever speculation is unsupported,
  unqualified, memory-negative, or slower for the current load cell.

The implementation begins with backend-neutral contracts and fake/CPU execution,
then qualifies gfx1151. gfx1100 is a separate follow-up campaign. No gfx1151
kernel threshold, graph bucket, numerical profile, or performance result
transfers to gfx1100 by source sharing.

## 2. Definition of done

### 2.1 Backend-neutral closure

Backend-neutral work is complete only when:

1. one set of typed records represents AR roots, chain/tree candidates, target
   frontiers, provider/target transactions, and committed cycle results;
2. providers expose bounded prepare/propose/commit/rollback/close stages rather
   than whole-request generation ownership;
3. `ResidentEngineLoop` schedules at most one AR transition or speculative cycle
   per due request per fairness pass;
4. admission returns before a speculative request finishes;
5. AR and speculative requests can coexist without a second model loop;
6. late admission, refill, retirement, cancellation, backpressure, and terminal
   reclaim use the existing Generation-2 request/output lifecycle;
7. target and provider claims reserve atomically before mutation;
8. rollback conserves target/provider cursors, RNG, output, KV/state ownership,
   and resource ledger units under injected failure at every stage;
9. a fake provider and fake target prove mixed `K=0`, chain, and bounded tree
   behavior; and
10. no engine/model/backend/quant hot-path conditional violates the four-axis
    registry invariant.

### 2.2 gfx1151 closure

gfx1151 work is complete when all of the following have a durable verdict:

1. exact dense GGUF MTP2 c1 is driven one cycle at a time by Generation-2;
2. physical c2/c4 proposal and target work contains no per-request target
   backbone loop;
3. every admitted `(C, K, context, profile)` cell has exact/profile correctness,
   profiler ownership, complete-wall economics, and a strict fallback;
4. target row buckets through the largest retained speculative cell are measured
   against honest physical decomposition;
5. higher concurrency cells that lose or lack a qualified bucket select `K=0`
   before transaction open and report the exact reason;
6. blocking and SSE publish only committed IDs, once and in order;
7. fixed, ragged, delayed-admission, cancellation, prefix, pressure, overload,
   recovery, and soak gates pass;
8. the full category and heldout quality suite plus applicable long/task gates
   pass for every promoted scope;
9. same-host true AR and MTP2 SLO-goodput evidence is recorded; and
10. all accepted wins are defaults for their qualified cells, while rejected or
    merely functional cells remain explicit/default-off with durable evidence.

A valid closure may promote no automatic speculative cell if the measured
implementation loses or fails quality. In that case SPECDEC2 still ships the
correct scheduler/provider architecture, `auto` selects `K=0`, explicit
functional scopes remain clearly diagnostic, and the campaign records the
rejection rather than tuning indefinitely or weakening gates.

### 2.3 Not required for this campaign

The following are explicitly deferred:

- gfx1100 qualification or performance promotion;
- PARO MTP2, Laguna DFlash, Qwen DFlash2, EAGLE, or remote providers;
- arbitrary tree-shape performance beyond bounded correctness coverage;
- multi-stream proposal/target overlap;
- a multi-cycle device-resident generation loop;
- TP/PP/EP composition;
- sampled/speculative modes not declared by the first capability; and
- compact INT8 KV qualification outside its independent campaign.

## 3. Frozen starting point

### 3.1 Code and research base

- [x] `4fcbf00d6` records the seven-engine source audit and target-frontier
      recommendation in `SPECDEC2-RESEARCH.md`.
- [x] `fd9dd35df` closes the original gfx1151 C2 campaign.
- [x] `4efdfdffa` adds the retained gfx1151 C2 baseline to `MTP-FIX.md` and is
      the selected implementation base for this campaign.
- [x] The user approved backend-neutral plus gfx1151 implementation, with
      gfx1100 deferred until this campaign closes.
- [x] This document supersedes the research file's “not yet approved” status;
      research remains the alternatives/evidence appendix.

### 3.2 Frozen gfx1151 AR control

The retained product baseline is Qwen3.8-27B Q4_K_S, BF16 KV, scoped FP16
recurrent state with FP32 strict fallback, `fair:256`, queue2, p128/d8.
`MTP-FIX.md` section 14 is authoritative. Key closure rows are:

| C | Aggregate tok/s | TTFT p95 | ITL p99 | E2E p95 | SLO runs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.543 | 0.651 s | 0.119 s | 1.222 s | 6/6 |
| 2 | 7.474 | 1.294 s | 0.205 s | 2.141 s | 6/6 |
| 4 | 10.103 | 2.216 s | 0.308 s | 3.176 s | 6/6 |
| 8 | 11.241 | 4.340 s | 0.547 s | 5.696 s | 1/6 |
| 17 | 11.000 | 8.975 s | 1.231 s | 12.363 s | 0/6 |
| 32 | 10.859 | 17.804 s | 2.254 s | 23.583 s | 0/6 |

The committed-source c32 streaming closure row is 10.590 tok/s, TTFT p95
18.617 s, ITL p99 2.125 s, and zero SLO-goodput. SPECDEC2 must not claim that a
single-request speculative multiplier repairs those latency failures.

### 3.3 Existing controls to reuse, not reimplement

- `DraftBatch`, `TargetVerifyBatch`, `AcceptResult`, `TargetAcceptSummary`, and
  device verify buffers;
- `SpeculativeCycleSimulator` and Generation-2 `SimulatedResourceLedger`;
- `NativeSpecCycleLauncher` and exact GGUF N1/N2/N3 components;
- exact selected hidden/Conv/GDN/KV commit and rollback;
- gfx1151 B1/B2 NextN target graph and exact native B3 evidence;
- C2 stable request/slot separation, global KV ownership, `KVLiveSpans`, graph
  replay, prefix, pressure, cancellation, output, and reclaim;
- the RF0–RF7 containment, long-context, lifecycle, API, load, rejection, and
  rollback packets; and
- registered strict/eager/serial fallbacks.

Historical evidence is a control only when the same model, artifact, profile,
host, shape, and command still apply. Otherwise rerun the exact new campaign
cell; do not splice rates across hosts or protocols.

## 4. Normative architecture

### 4.1 Ownership

`EngineService` and `ResidentEngineLoop` are the sole request and visible-output
owners. A provider may own provider weights, provider KV/state, proposal graphs,
and provider workspaces. A target adapter may own target provisional buffers and
kernel execution. Neither owns a frontend request loop.

One bounded cycle is:

```text
plan K / reserve all claims
    -> provider prepare/propose
    -> lower target frontier
    -> target verify
    -> accept / stop / length
    -> atomic selected target+provider commit or rollback
    -> publish committed result
    -> release transaction
    -> yield to Generation-2 fairness
```

### 4.2 Core records

The implementation must provide typed host records and, where applicable,
device-buffer descriptors for:

- `SpeculativeCapability` — compatible target/provider/profile/sampling/KV
  contract and physical limits;
- `SpecRequestPlan` — pre-mutation provider, K/tree shape, transaction mode,
  claims, graph/physical bucket, and reason;
- `CandidateGraph` — provider-owned device or bounded host candidate rows,
  request/slot maps, parents/depths, and provider transaction identity;
- `TargetFrontier` — committed roots plus candidate rows, physical lowering
  metadata, `KVLiveSpans`, and target transaction identity;
- `SpecCycleTransaction` — one target+provider atomic operation and checkpoints;
- `SpecCycleResult` — committed IDs/lengths, accept counts, selected rows,
  cursor/RNG deltas, finish reasons, route, and transaction verdict; and
- `SpecCycleTelemetry` — logical C/K/R, physical decomposition, graph/eager
  route, provider/target/commit wall, reasons, and ownership counters.

Existing `DraftBatch`, `TargetVerifyBatch`, and `AcceptResult` remain compatible
CPU/test projections. Do not duplicate their semantics under unrelated names.
Device-native candidates must not require full-logit or full-candidate D2H on the
normal path.

### 4.3 Provider SPI

The first staged protocol is equivalent to:

```python
capabilities(target, request_semantics) -> SpeculativeCapability
resource_claims(request_states, plan) -> ResourceClaimSet
prepare_requests(plan, stream) -> None
propose_batch(plan, stream) -> CandidateGraph
commit_batch(result, stream) -> None
rollback_batch(transaction, stream) -> None
close_requests(request_ids) -> None
```

Rules:

- methods are bounded and return to the engine after one stage;
- policy selects provider and K before any provider/target mutation;
- capability/registry resolution occurs at construction or cold planning, not
  inside per-layer hot loops;
- a provider cannot publish tokens, await a client, or run repeated cycles;
- provider and target state updates share one transaction verdict; and
- unsupported combinations fail before launch or choose `K=0` with a stable
  typed reason.

### 4.4 AR as K=0

`K=0` is an ordinary target plan, not an error or out-of-band fallback. It is
selected when:

- no provider is installed;
- request sampling/grammar semantics are unsupported;
- context/output room exceeds the capability;
- the complete claim does not fit;
- the required graph/physical bucket is absent and eager is not qualified;
- policy evidence says AR wins for the current C/context/profile cell;
- an independent provider cannot cheaply catch up after an AR transition; or
- a circuit breaker disables the provider before mutation.

A stable reason taxonomy must preserve existing serialized RF0 values such as
`target_graph_context_bucket_miss` and `target_graph_output_room_miss`.

### 4.5 Target frontier and physical lowering

Logical target rows are:

```text
R = sum(1 + k_i for each due request i)
```

Logical rows do not imply one physical launch. A backend capability lowers the
frontier into declared physical groups. Telemetry must expose both logical R and
the exact decomposition/weight-sweep count. A route cannot claim R32 batching if
it executed four independent R8 target backbones.

The first gfx1151 chain matrix is:

| C | Candidate K | Logical R |
| ---: | ---: | ---: |
| 1 | 1/2/3 | 2/3/4 |
| 2 | 1/2/3 | 4/6/8 |
| 4 | 1/2/3 | 8/12/16 |
| 8 | 0/1/2/3 | 8/16/24/32 |
| 17 | 0/1 | 17/34 |
| 32 | 0/1 | 32/64 |

R34/R64 are not mandatory kernel targets. If complete-wall evidence selects
`K=0` at c17/c32, that is the correct finished policy. Build a larger bucket only
from a measured premise that it can beat AR and meet the latency gate.

### 4.6 State and KV transaction modes

A capability declares one target and one provider transaction mode:

1. reserved append plus live-count commit;
2. packed scratch plus selected copy; or
3. reversible journal with a complete exact restoration gate.

Full attention consumes canonical per-request `KVLiveSpans` plus provisional
ancestor visibility. Conv/GDN/SSM state is parent-indexed: every candidate node
starts from its parent, and commit selects one row independently per request.
No row may observe another request's prefix, provisional KV, recurrent state,
sampler RNG, stop state, or output.

All persistent and transient claims are composed and reserved atomically before
any target/provider owner opens. Page rounding, scratch slabs, graph pointer
slabs, provider KV/state, result buffers, and repair capacity are included.
Unknown claims fail closed; no hidden lazy allocation is permitted in a promoted
path.

### 4.7 Scheduling and fairness

The Generation-2 round order is:

1. drain submit/cancel/deadline commands;
2. finish safe commit/rollback and terminal reclaim;
3. admit fitting prompt work;
4. execute bounded prefill according to the retained policy;
5. select due decode requests under the existing fairness budget;
6. cold-plan each due request as AR or one provider/depth;
7. reserve complete cycle claims;
8. run compatible provider groups;
9. lower and execute compatible target groups;
10. accept and atomically commit/rollback each request;
11. publish bounded committed events; and
12. yield.

Every due request receives at most one transition before a peer receives a
second transition. A speculative group cannot hold the model lock for a complete
request. Late arrivals join future rounds. A slow stream consumer remains
isolated by its request-owned bounded output queue.

### 4.8 Streaming, stop, sampling, and cancellation

- Only committed IDs are visible.
- A cycle may publish several IDs as one ordered chunk or multiple ordered
  events, but each ID is published exactly once.
- EOS, stop-token, stop-string, and output-length handling may truncate within a
  committed cycle without publishing hidden tail IDs.
- RNG and penalties are request-owned. The first provider may declare greedy
  only; unsupported sampling chooses `K=0` before mutation.
- Cancellation during device work becomes pending and resolves at the next safe
  transaction boundary. Cancellation before reservation has no device owner.
- One cancelled/rejected row does not fail peers unless a shared physical owner
  is poisoned; that failure must roll back all affected open transactions and
  preserve later service health.

### 4.9 Graph and queue policy

Start with separate proposal, target, accept/commit, and provider-update bundles.
Graph keys include target/model/quant/profile, provider/policy, logical C/K/R,
physical bucket/decomposition, context/page bucket, transaction modes, hidden
taps, sampler class, and variant-manifest hash.

Keep the retained gfx1151 queue2 policy and one serial stage stream initially.
Do not add multi-stream overlap until a qualified trace shows recoverable slack
and the synchronous baseline is complete. Graph misses use a declared qualified
eager path or select `K=0`; they never run an unqualified optimistic route.

## 5. Phase dependency graph

```text
S0 plan/control freeze
  -> S1 contracts + simulator
    -> S2 Generation-2 one-cycle integration
      -> S3 exact gfx1151 c1 MTP2 adapter
        -> S4 physical gfx1151 c2/c4
          -> S5 gfx1151 target buckets + dynamic K
            -> S6 product closure
              -> S7 gfx1100 portability (separate follow-up)
```

Each phase is one or more small validated commits. Do not begin a dependent
phase while the prior exit gate is red or its logical unit is uncommitted.

## 6. S0 — plan, inventory, and controls

### Punchlist

- [x] Approve SPECDEC2 and gfx1151-first scope.
- [x] Preserve the research audit in `SPECDEC2-RESEARCH.md`.
- [x] Freeze the Generation-2 gfx1151 AR/C2 baseline at `4efdfdffa`.
- [x] Name Qwen3.8-27B Q4_K_S/BF16-KV as the primary product lane.
- [x] Preserve Qwen3.6/Qwen3.8 direct NativeSpecCycle controls where each exact
      artifact applies.
- [x] Defer gfx1100 qualification to S7.
- [x] Record the legacy whole-request speculative route as migration debt in
      `REFACTOR.md`.
- [ ] Before the first GPU/kernel edit, run the kernel lineage check and inspect
      any drift in touched parent families.
- [ ] Before the first GPU gate, confirm ROCm/device identity and acquire the
      repository GPU-exclusive lock/lease.

### Exit gate

The normative plan is committed with an immutable worklog, `PLAN.md` links here,
and the research file points back here as the approved successor. S1 may then
start without a GPU run.

## 7. S1 — backend-neutral contracts and RED simulator

### S1.1 Contract records

- [x] Add `SpeculativeCapability` with target/provider/profile/KV/sampling and
      shape limits, transaction modes, graph/eager support, and strict fallback.
- [x] Add `SpecRequestPlan` with `K=0` as a first-class valid plan.
- [x] Add host/device `CandidateGraph` descriptors without framework tensors.
- [x] Add `TargetFrontier` as the canonical root+candidate representation and
      project host-visible candidates through retained `TargetVerifyBatch`
      topology without duplicating it.
- [x] Add one atomic target+provider transaction record.
- [x] Add committed `SpecCycleResult` and bounded telemetry records.
- [x] Add stable typed reason values while preserving existing external strings.
- [x] Export new public-internal records through `hipengine.speculative` without
      making them a supported end-user API.

### S1.2 Staged provider SPI and registry

- [ ] Add the bounded staged provider protocol.
- [ ] Register provider factories/capabilities through plugin keys; no engine
      branch on provider, backend, or quant.
- [ ] Keep the existing whole-request `SpeculativeTextProvider` only as a
      compatibility/oracle protocol during migration.
- [ ] Reject a provider that attempts unbounded generation ownership.
- [ ] Define target-attached versus independent-provider catch-up semantics.

### S1.3 Planner and simulator

- [x] Rebase `SpeculativeCycleSimulator` on the production transaction, result,
      stage, and telemetry records while retaining compatibility aliases.
- [ ] Cover C={1,2,4,8}, K={0,1,2,3}, mixed K, chain, and bounded tree metadata.
- [ ] Cover row/request/slot permutation and compact/refill.
- [ ] Cover atomic provider+target+transient claims and fit rejection.
- [ ] Cover reject/partial/full accept plus correction/bonus and output tails.
- [ ] Inject cancel/failure at reserved, target-open, provider-open, drafted,
      verified, accepted, commit, and readback stages.
- [ ] Prove final target/provider cursors, RNG, outputs, transactions, leases,
      and ledger units are conserved.
- [ ] Prove one peer's reject/cancel cannot mutate another peer.
- [ ] Prove unsupported sampling/context/claims select K=0 before mutation.

### RED/GREEN gate

Create RED tests before implementation where practical. The initial focused
bundle is expected to include:

```bash
python3 -m pytest -q \
  tests/test_speculative_interfaces.py \
  tests/test_speculative_cycle_simulator.py \
  tests/test_speculative_provider_registry.py \
  tests/test_speculative_generic_providers.py \
  tests/test_kvcache_policy.py
```

Then run the applicable CPU deterministic bundle from `TESTING.md`. No GPU run
or performance claim is required. S1 exits only with zero leaked transaction or
ledger owner in every injected path.

## 8. S2 — Generation-2 one-cycle execution

### S2.1 Runner and scheduler contract

- [ ] Extend the runner protocol with bounded speculative stage/cycle methods or
      one typed cycle method whose internal stages remain measurable.
- [ ] Teach `ResidentBatchScheduler` to select due speculative and AR plans
      under one fairness budget.
- [ ] Materialize root tokens/positions and target frontier from scheduler-owned
      request state.
- [ ] Reserve complete claims before provider/target open.
- [ ] Record work duration/counts for proposal, target, commit, and rollback.
- [ ] Keep stable request IDs separate from physical provider/target rows.

### S2.2 Engine loop

- [ ] Teach `_tick_once()` to execute one bounded speculative cycle.
- [ ] Keep command draining and late admission possible between cycles.
- [ ] Support mixed AR and speculative due requests in one fairness round,
      lowering to separate physical groups when required.
- [ ] Publish multi-token committed cycle results through canonical events.
- [ ] Resolve pending cancellation at safe boundaries.
- [ ] Reclaim terminal requests independently and compact only stable owners.
- [ ] Preserve subsequent AR health after provider/target failure.

### S2.3 EngineService/API lifecycle

- [ ] Make speculative child submission O(1) admission/planning; no generation
      inside command handling.
- [ ] Stop calling `submit_speculative_many_detailed()` as a whole-request model
      execution path for a migrated provider.
- [ ] Preserve parent/child IDs, all-choice accounting, finish details, deadlines,
      and circuit-breaker behavior.
- [ ] Make blocking and SSE consume the same committed result events.
- [ ] Keep unsupported public sampling/streaming behavior fail-closed until its
      exact path is qualified.

### Fake/CPU proof

- [ ] Admission returns while a fake speculative request remains active.
- [ ] A late AR request advances before the first speculative request finishes.
- [ ] Two speculative requests refill after staggered completion.
- [ ] One request emits multiple committed IDs without duplicates.
- [ ] Cancellation at every stage preserves a survivor.
- [ ] Mixed K0/K1/K3 fairness and output order pass.
- [ ] A provider cannot run a second cycle without the engine yielding.
- [ ] Shutdown drains all child, transaction, output, and resource owners.

Expected focused tests include new SPECDEC2 scheduler/loop files plus:

```bash
python3 -m pytest -q \
  tests/test_generation_batch_scheduler.py \
  tests/test_generation_engine_loop.py \
  tests/test_generation_engine_service.py \
  tests/test_server_speculative_provider.py \
  tests/test_speculative_streaming.py
```

Use narrow test nodes during RED/GREEN; do not repeatedly run the existing large
scheduler/server files in full after an isolated repaired failure unless shared
state or infrastructure changed. S2 exits with backend-neutral continuous
lifecycle proof and no GPU implementation.

## 9. S3 — exact dense GGUF MTP2 c1 on gfx1151

### S3.1 Adapter

- [ ] Resolve a gfx1151 dense GGUF MTP2 capability at model/session construction.
- [ ] Wrap the existing exact NextN provider as prepare/propose/commit/rollback
      stages without changing arithmetic.
- [ ] Wrap N1/N2/N3 target/accept/selected-commit components as one-cycle target
      execution.
- [ ] Bind target/provider resource claims, stable slabs, graph keys, and strict
      eager/serial fallback.
- [ ] Preserve exact context/output-room and circuit-breaker reasons.
- [ ] Leave the old whole-request route as an explicit oracle/rollback only.

### S3.2 Correctness

For B1/B2/B3 and reject/partial/full accept:

- [ ] generated IDs and cycle semantics match the direct exact control;
- [ ] target logits/top-1 and selected hidden match;
- [ ] Conv/GDN state, full-attention KV, live counts, positions, and cursor match;
- [ ] provider MTP state/KV/cursor match;
- [ ] following-AR continuity matches;
- [ ] graph/eager and graph-miss fallback agree;
- [ ] output tails, stop, cancellation, and injected failure restore ownership;
- [ ] deterministic repeats and artifact/profile manifest identity pass; and
- [ ] no hidden allocation or dense KV/state shadow appears.

### S3.3 Economics and engagement

- [ ] Profiler trace proves expected proposal, target, accept, and commit kernel
      families and plausible durations.
- [ ] One-cycle service overhead is measured against the unchanged direct cycle.
- [ ] Suggested adapter overhead gate is <=5% complete wall before c>N work;
      larger overhead must be localized and fixed or explicitly approved.
- [ ] The public c1 route yields between cycles and does not hold a request-life
      model lock.

GPU work follows `KERNELS.md`: prebuild outside `rocprofv3`, use cached builds,
and record physical host/model/artifact/command/profile. S3 exits only when the
old route is a strict oracle rather than the migrated server owner.

## 10. S4 — physical gfx1151 c2/c4 MTP2

### S4.1 Physical proposal

- [ ] Batch provider proposal for C=2 and C=4 at K={1,2,3} where the provider
      capability admits the shape.
- [ ] Keep per-request RNG, state, positions, stop/output room, and cursors.
- [ ] Prove proposal launch/backbone counts do not scale as one full provider
      call per request unless a declared fallback is being measured.
- [ ] Keep candidates device-resident through target lowering.

### S4.2 Physical target frontier

- [ ] Implement/qualify target logical R={4,6,8,12,16} through declared physical
      buckets.
- [ ] Preserve root/candidate parent topology and per-request ancestor attention.
- [ ] Implement parent-indexed Conv/GDN candidate recurrence.
- [ ] Keep per-request `KVLiveSpans` and provisional KV ownership.
- [ ] Run one device accept/selected-state commit payload per physical group.
- [ ] Report honest physical decomposition and weight sweeps.

### S4.3 Dynamic lifecycle

- [ ] Mixed prompt lengths and context positions.
- [ ] Different accept counts in the same cycle.
- [ ] Rejecting row beside full-accept row.
- [ ] Staggered finish and refill into a future cycle.
- [ ] Cancel one row while peers continue.
- [ ] Slot permutation/compaction and neighbor substitution.
- [ ] Prefix restore/COW boundaries and pressure rejection.
- [ ] Eager/graph fallback and later health.

### S4.4 Gate

Every retained cell requires strict/profile correctness, same-schedule
repeatability, isolation, resource conservation, kernel engagement, and complete
same-host wall evidence. Merely packing request metadata or running singleton
cycles in a loop does not pass. MTPLX-style sealed fixed-width execution may be
used as an intermediate device proof, but S4 closure also requires Generation-2
late admission and future-round refill.

## 11. S5 — gfx1151 verifier buckets and dynamic K

### S5.1 Measurement-first bucket ladder

- [ ] Run the current strict physical decomposition for R={1,2,4,8,16,32} and
      record target-only plus complete-cycle wall.
- [ ] Add a native bucket only when its measured premise can beat decomposition
      and its correctness oracle exists.
- [ ] Compare R16/R32 complete target wall against two/four R8 sweeps.
- [ ] Attribute dense projection, attention/KV, Conv/GDN, accept/commit,
      submission, synchronization, and readback.
- [ ] Update `KERNELS.md` and lineage manifest when a kernel/dispatch path changes.

### S5.2 Kernel/graph qualification

For every new/ported family:

- [ ] RED fixture/oracle first;
- [ ] strict exact/parent-parity or declared production-profile gate;
- [ ] CPU-reference outer KL/top-1 floor;
- [ ] registered strict fallback;
- [ ] `rocprofv3 --kernel-trace` engagement;
- [ ] graph/eager agreement and miss fallback;
- [ ] stable pointer/slab ownership; and
- [ ] complete model/cycle wall, not only microbench speed.

### S5.3 Cost table and policy

Build immutable cells keyed by:

```text
backend / host / target artifact / quant / KV / execution profile
provider / provider artifact / policy fingerprint
C_ar / C_spec / context-page bucket / K or tree shape / logical R
physical proposal and target decomposition / transaction modes
graph-eager route / sampler class / variant-manifest hash
```

Record proposal, target, accept/commit, provider-update, scheduler/readback,
claims/high-water, accepted-output distribution, TTFT/ITL/E2E, and SLO-goodput.

- [ ] Start with deterministic offline LUT policy.
- [ ] Permit online acceptance EMA only inside an already-qualified cell.
- [ ] Select K before mutation and report the exact reason.
- [ ] Never use prompt text, token IDs, benchmark category, or heldout identity as
      a routing feature.
- [ ] Qualify c1/c2/c4 K choices.
- [ ] Measure c8 K={0,1,2,3} and retain only winning/SLO-safe cells.
- [ ] Measure c17/c32 K={0,1}; finish with K0 if larger frontiers lose or lack a
      qualified physical path.
- [ ] Keep AR neighbors within their declared SLO when mixed with speculation.

### S5.4 Performance acceptance

- Functional/default-off retention: correctness passes, but no speed claim.
- `auto` cell: >1.10x true same-protocol AR plus non-regressive quality,
  TTFT/ITL/SLO-goodput, memory, pressure, and lifecycle.
- Project target: >1.30x. Do not lower the target after seeing results.
- Exact, same-suite non-regressive wins become the qualified default.
- Production-arithmetic wins require every binding execution-profile gate.

## 12. S6 — gfx1151 product closure

### S6.1 API and semantics

- [ ] Blocking completion/chat, SSE completion/chat, and multi-prompt children.
- [ ] Exact committed token accounting per choice and all choices.
- [ ] Multiple accepted tokens stream once and in order.
- [ ] EOS/stop token/stop string/output length through cycle tails.
- [ ] Supported sampling semantics or deterministic K0 fallback before mutation.
- [ ] Stable direct reason, route, K, C/R, physical bucket, graph, and profile
      reporting.
- [ ] Circuit breaker, operator rollback, restart reset, and subsequent health.

### S6.2 Dynamic serving matrix

- [ ] Fixed c1/c2/c4/c8.
- [ ] Ragged prompt and decode lengths.
- [ ] Delayed admission and refill.
- [ ] Mixed AR/K0 and MTP2 requests.
- [ ] Cancellation/disconnect/deadline with survivor continuation.
- [ ] Prefix hit/miss/COW/eviction.
- [ ] KV pressure, retryable rejection, regrow, and graph-pointer invalidation.
- [ ] Poisson offered load at below/near/above saturation.
- [ ] Overload and bounded pending/output queues.
- [ ] Recovery after provider/target/graph/readback failure.
- [ ] 100+ request alternating/mixed soak and clean shutdown.
- [ ] Zero final allocations, claims, pages, transactions, collectors, and
      background owners.

### S6.3 Quality and benchmark packet

- [ ] Full `mtp-bench` code/general_en/general_ja/mixed_ja_en categories.
- [ ] Category heldouts and applicable long/task fixtures.
- [ ] True no-MTP AR baseline from the exact same protocol.
- [ ] Deterministic repeats and batch-composition isolation.
- [ ] Same-host counterbalanced benchmark with model/quant/KV/profile/command.
- [ ] Aggregate and per-request tok/s, TTFT/ITL/E2E/queue, SLO-goodput, memory,
      occupancy, acceptance, K/reason histogram, graph/fallback, and health.
- [ ] Compact schema-valid result artifact.
- [ ] `benchmarks/README.md` row/Last-updated and `benchmarks/CHANGELOG.md` entry
      for every retained result.
- [ ] Public `README.md` export only if a product scope promotes; no campaign
      diary or internal implementation detail.

### S6.4 Closure verdict

- [ ] List every promoted automatic cell.
- [ ] List every explicit/default-off functional cell.
- [ ] List every rejected cell with reason/artifact.
- [ ] Remove or demote superseded whole-request server routes per `REFACTOR.md`.
- [ ] Preserve strict oracles/fallbacks required by profile contracts.
- [ ] Run milestone validation according to `TESTING.md`; use focused repair after
      isolated broad-suite failures unless the fix can affect prior passes.
- [ ] Commit immutable closure worklog and update this punchlist/status.

## 13. S7 — gfx1100 portability follow-up

S7 starts only after S6 is committed. It is intentionally not part of the
current implementation campaign.

- [ ] Create a separate gfx1100 plan/worklog and select an exact clean base.
- [ ] Run backend-neutral suites unchanged.
- [ ] Resolve independent gfx1100 capabilities and strict fallbacks.
- [ ] Requalify c1/c2/c4, physical R buckets, graph/eager, and dynamic K.
- [ ] Run independent quality, lifecycle, profiler, memory, and performance
      packets on the exact gfx1100 host/model/artifact.
- [ ] Never transfer gfx1151 policy thresholds or absolute rates.

## 14. Expected code map

This map is directional. Keep edits scoped and prefer cohesive new modules over
adding more policy to existing monoliths.

| Area | Intended work |
| --- | --- |
| `hipengine/speculative/interfaces.py` | Preserve existing host records; only add compatibility projections that belong with them. |
| `hipengine/speculative/frontier.py` | New target-frontier/candidate/device descriptors and topology validation. |
| `hipengine/speculative/provider.py` | Staged provider capability/protocol and provider transaction semantics. |
| `hipengine/speculative/policy.py` | Typed K/K0 reasons and immutable cost-cell/LUT selection. |
| `hipengine/speculative/simulator.py` | Rebase fake transaction execution on production records. |
| `hipengine/speculative/registry.py` | Register staged provider factories/capabilities while retaining migration oracle route. |
| `hipengine/dispatch/batch.py` | Bounded proposal/frontier work metadata without backend/provider branching. |
| `hipengine/generation/batch_scheduler.py` | Due-request plan, complete claims, fairness, frontier packing, committed multi-token accounting. |
| `hipengine/generation/engine_loop.py` | Execute one bounded speculative cycle and yield. |
| `hipengine/generation/engine_service.py` | O(1) speculative admission and common child/output lifecycle. |
| `hipengine/generation/qwen35_gguf.py` | Thin MTP2 adapter construction; old whole-request route becomes oracle/rollback. |
| `hipengine/runtime/qwen35_gguf_nextn.py` | C-batched staged proposal adapter and device candidate ownership. |
| `hipengine/runtime/qwen35_gguf_mtp.py` | Target frontier/accept/selected-commit adapter. |
| `hipengine/runtime/qwen35_gguf_runner.py` | Physical gfx1151 target lowering only where existing façade requires it; avoid new policy branches. |
| `hipengine/kernels/hip_gfx1100/` shared sources | New verifier primitives developed in-tree and independently registered/qualified for gfx1151. |
| `hipengine/kernels/hip_gfx1151/__init__.py` or split manifest | Peer registration/capability only; no assumption of gfx1100 equivalence. |
| `tests/test_specdec2_*.py` | Split contracts, simulator, scheduler, lifecycle, and GPU-gated tests rather than growing one monolithic test file. |
| `benchmarks/schemas/` | Compact SPECDEC2 decision/capture schema if existing schemas cannot represent required cells. |

No normal module reached by `LLM.generate()` may import torch. Kernel bodies keep
raw pointer signatures. Engine/model code must not add backend/quant/provider
hot-path branches.

## 15. Validation and execution discipline

### 15.1 Test order

For each logical unit:

1. write or identify the RED oracle;
2. run the narrowest failing node;
3. implement the minimum scoped change;
4. run the repaired node and affected file/bundle;
5. run the CPU deterministic bundle for backend-neutral/shared code;
6. run GPU/profile gates only when GPU code or a GPU milestone requires them;
7. update worklog/artifacts/docs; and
8. commit immediately before starting the next unit.

A new kernel additionally requires lineage check, strict/profile gate,
CPU-reference outer gate, registered strict fallback, and profiler engagement.

### 15.2 Expensive gates

The assigned campaign is standing approval for necessary expensive validation.
Before a command expected to exceed five minutes, state:

- exact command/purpose;
- expected duration;
- GPU/host lock identity;
- timeout/stop budget; and
- expected artifact/output path.

Use background tasks for long runs. Do not rerun an equivalent expensive packet
when retained evidence plus focused repair is sufficient.

### 15.3 Commit boundaries

Expected atomic units include:

1. approved docs/plan;
2. contracts and RED tests;
3. simulator/transaction migration;
4. provider registry/SPI;
5. scheduler planning;
6. engine-loop fake cycle;
7. EngineService/API lifecycle;
8. gfx1151 c1 adapter;
9. each physical proposal/target bucket family;
10. dynamic policy/cost table;
11. each retained performance win or rejected candidate; and
12. product closure.

Never stage unrelated untracked benchmark artifacts. Never commit a partially
broken phase merely to checkpoint exploration.

## 16. Refactor and removal triggers

During migration both serving routes may exist:

- the new Generation-2 one-cycle SPECDEC2 route; and
- the old synchronous whole-request `submit_speculative_many_detailed()` path.

The old path is not a permanent fallback architecture. After S3 direct/public c1
parity and S4 continuous lifecycle pass:

1. remove its production selection and request-lifetime model-lock ownership;
2. retain only an explicit test/oracle helper if still required;
3. remove duplicate speculative child/result tables and synthetic post-hoc
   `VERIFY_CHAIN` work metadata;
4. collapse temporary provider/SPECDEC2 flags into one cold capability/policy
   selection; and
5. keep registered strict target/provider kernels and exact direct controls.

Every temporary flag introduced by this campaign must be added to
`REFACTOR.md` with its exact removal gate.

## 17. Stop, reject, and escalation rules

- Stop and fix ownership before performance work if a request can observe a
  peer's state/KV/RNG/output or any transaction leaks.
- Stop and add/localize an oracle if arithmetic changes without a binding gate.
- Reject a performance candidate that fails any binding execution-profile or
  task gate; do not relabel control bugs as numerical drift.
- Reject a speed claim based on post-hoc metadata, request-serial target loops,
  mixed hosts/protocols, verifier-derived AR, or prompt-conditioned policy.
- Record neutral/negative candidates after one attribution pass; do not tweak
  blindly.
- If complete c2/c4 speculative execution cannot beat AR, finish the correct
  default-off/K0 architecture and record the blocker rather than weakening the
  baseline.
- If a high-conflict file changes concurrently, stop and coordinate; do not
  force-stage or overwrite another owner.
- gfx1100 work begins only after the user-visible gfx1151 closure commit unless
  the user explicitly changes scope.

## 18. Current handoff

The next action after this plan commit is **S1 contract RED tests**. No GPU work
is needed until S3. The initial implementation should create the production
records and staged provider boundary first, then rebase the simulator so S2 uses
already-proven transaction semantics rather than inventing a second lifecycle.
