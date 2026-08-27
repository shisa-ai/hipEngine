# CONCURRENCY2 gfx1151 MTP Dynamic Admission Campaign

- Status: **design complete; execution not started or approved in this unit**
- Scope: **plan/design only; no GPU work was run**
- Hardware lane when execution is approved: **Radeon 8060S / `hip_gfx1151`**
- Primary product key: **Qwen3.8-27B `Q4_K_M`, BF16 KV, production profile**
- First functional width: **two independent resident requests (`C_due=2`)**
- Performance target: **promote only cells `>=1.10x` true same-protocol AR; project target remains `>1.30x`**
- Predecessors: [`SPECDEC2.md`](SPECDEC2.md),
  [`SPECDEC2-PERF.md`](SPECDEC2-PERF.md),
  [`QWEN38-Q4KM-MTP-SERVING.md`](QWEN38-Q4KM-MTP-SERVING.md), and
  [`CONCURRENCY2-GFX1151-MTP-TUNING.md`](CONCURRENCY2-GFX1151-MTP-TUNING.md)
- Normative contracts: [`PLAN.md`](PLAN.md), [`CONCURRENCY2.md`](CONCURRENCY2.md),
  [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md), [`TESTING.md`](TESTING.md),
  and [`BENCHMARK.md`](BENCHMARK.md)

This is a separate fundamental routing/ownership campaign. It does not reopen,
replace, or append work to the completed tuning campaign. In particular, it
must not race the active capacity>1 singleton-routing unit. Execution starts
only from that unit's clean committed descendant and after explicit approval.

## 1. Executive decision

The required order is:

1. **separate static request eligibility from dynamic realized-group selection;**
2. **run real MTP through the normal resident owner and prove switching,
   ownership, and lifecycle;**
3. **then optimize and promote only cells that beat true AR.**

Do not require a speed win to retain a correct explicit functional milestone.
Do not expose an automatic route merely because an explicit physical harness
runs. A slow but correct normal-owner C2 route is a valid intermediate result;
a fast route that cannot safely transition, cancel, refill, or reclaim is not.

## 2. What already exists — do not reimplement it

At planning snapshot `8e271b740`, the repository already contains:

- `ResidentEngineLoop._maybe_run_speculative_cycle()`, which receives the
  actual due decode `WorkItem`, constructs per-request semantics, and resolves
  an immutable `SpecRequestPlan` before opening transaction ownership;
- mixed candidate counts in `SpecRequestPlan`, including K0 rows beside
  speculative rows;
- staged target-frontier, provider, transaction, atomic-claim, cancellation,
  rollback, and committed-output contracts;
- GGUF NextN/MTP2 physical C2/C4 proposal, target frontier, GPU accept, and
  selected-state commit mechanics;
- exact/default-off strict C1/C2/C4 K1-K3 functional evidence for Q4_K_S;
- Q4_K_M production C1/B3 serving and numerical evidence;
- `ProviderCatchupMode.TARGET_OUTPUT` plus a GGUF K0 catch-up primitive;
- request-ID-owned provider groups, refill attachment, and close/reclaim
  primitives;
- backend-neutral fake-loop tests for mixed K, late AR arrival, staggered
  retirement, and refill; and
- `hipengine/speculative/packing.py`, which defines compatibility, fairness,
  verifier-cost, and prelaunch-budget records but is not yet the normal
  Generation-2 execution owner.

These are reusable architecture and controls, not proof that Q4_K_M production
serving admits independent C>1 requests. Q4_K_S profile/quant evidence does not
transfer to Q4_K_M, and fake/explicit lifecycle evidence does not prove the
normal public route.

## 3. Actual gap: two planning times are conflated

The current public artifact resolver receives `realized_group_rows=len(prompts)`
at the frontend. For independent HTTP children, that is prompt multiplicity in
one request, not the eventual due group inside `ResidentEngineLoop`. A request
can therefore be routed to plain AR before the resident scheduler ever sees it
as speculative intent. The cycle planner and physical C2/C4 implementation are
then unreachable even though they exist.

The fix is not another request scheduler. It is a two-level contract:

### 3.1 Static request eligibility — frontend/admission time

Resolve only fields knowable before scheduler mutation:

```text
artifact/hash/size / backend/arch / quant / execution profile + manifest
KV backend/layout / resident-owner capacity / sampling semantics
context + output horizon / memory and per-request persistent-claim fit
provider inventory / strict fallback / evidence identity
```

The result is either:

- **permanent AR**: no provider intent, no provider activation/catch-up, stable
  typed K0 reason; or
- **speculative-capable intent**: maximum allowed K, provider/catch-up mode,
  activation contract, strict fallback, and evidence fingerprint. It is not yet
  a decision that the next cycle will use MTP.

The static key must not contain prompt text, token IDs, benchmark category,
heldout identity, task output, or a guessed future concurrency. Resident
capacity is immutable owner identity and may remain in the key; `C_due` may not.

### 3.2 Dynamic cycle plan — resident fairness boundary

For each due `WorkItem`, use the actual request IDs, slots, contexts, output
room, cancellation/deadline state, active intents, claims, physical buckets,
and retained economics table to select `K_i` before mutation.

Use distinct dimensions in policy and telemetry:

```text
C_resident = all active resident requests
C_due      = requests due in this fairness work item
C_spec     = due requests with K_i > 0
R          = C_due + sum(K_i) logical target-frontier rows
physical proposal groups / physical target-row decomposition
```

Never label `C_due=8` as a physical C8 MTP owner when it executes two C4
frontiers or C4 plus K0 rows. Requested route, selected cycle plan, effective
backend route, and committed MTP usage remain separate fields.

## 4. Request transition contract

Every request occupies one of these ownership states:

| State | Provider ownership | Allowed next action |
| --- | --- | --- |
| `permanent_ar` | none | true AR only; no automatic MTP re-entry |
| `intent_activating` | persistent claims reserved; prompt activation open | commit `spec_ready` or roll back to `permanent_ar` |
| `spec_ready` | provider cursor exactly matches committed target root | MTP cycle or transitional K0 |
| `spec_active` | one target+provider transaction open | atomic commit, rollback, or cancel at the safe boundary |
| `transitional_k0` | provider remains request-owned and synchronized | true AR target transition plus exact provider catch-up, then `spec_ready` |
| `disabled_ar` | provider rolled back/released or circuit-disabled | true AR until an explicit re-admission boundary |
| terminal | none after reclaim | no transition |

Binding transitions:

1. **Static ineligible -> permanent AR.** No prompt sink, provider slot,
   checkpoint, candidate buffer, or speculative transaction may open.
2. **Eligible admission -> spec ready.** Reserve persistent provider/request
   claims before activation. Target prefill and shifted NextN priming either
   commit together or leave a healthy AR request with no residual owner.
3. **K0 -> MTP.** Allowed only from `spec_ready`; provider cursor/state/KV must
   match the current committed target root before proposal.
4. **MTP -> K0.** Finish or roll back the open transaction, publish only committed
   target output, then execute provider catch-up under the declared catch-up
   mode before the next AR root transition.
5. **C1 -> C2+ and C2+ -> C1.** Switch only between transactions. Request IDs,
   resident slots, provider rows, target rows, and output owners remain distinct.
6. **Cn -> Cm reshape.** Surviving requests keep request-owned provider state;
   moved slots and new peers cannot inherit stale state or workspace rows.
7. **Failure/circuit break.** Restore target and provider checkpoints before AR;
   postcommit failure uses the registered canonical-rebuild contract. A failed
   provider cannot silently re-enter MTP later.
8. **Terminal/cancel.** Close prompt sink, checkpoint, provider row/group,
   transaction claims, target provisional state, and output ownership exactly
   once while peers continue.

### 4.1 Two K0 classes are mandatory

Do not overload one K0 label:

- **pure K0** is permanent AR with no provider mutation;
- **transitional K0** is an eligible speculative request temporarily using AR
  while its provider stays synchronized.

Transitional K0 cost is part of speculative policy economics. It may never be
hidden inside the true-AR denominator or reported as MTP throughput. If shadow
catch-up is too expensive, the retained policy may be one-way MTP->AR or may
require bounded re-activation; that is an explicit measured verdict, not an
implicit fallback.

## 5. Ordered campaign

### D0 — clean handoff and no-repeat audit (no behavior change)

- [ ] Start from the first clean committed descendant after the active
  capacity>1 singleton-routing unit finishes. Record its commit, scope,
  tests/artifacts, and remaining diffs; do not duplicate its code.
- [ ] Freeze exact Q4_K_M model/hash/quant/KV/profile/manifest, prompt fixture,
  host, compiler, queue policy, and true-AR command.
- [ ] Produce a source-to-contract map for frontend resolver, request intent,
  `ResidentEngineLoop`, `SpecRequestPlan`, packing, GGUF MTP2 activation,
  proposal, frontier, accept/commit, catch-up, and reclaim.
- [ ] Reuse prior primitive/oracle evidence only when model/quant/profile/shape
  identity actually matches. List every reused and repeated gate explicitly.
- [ ] Capture a no-GPU RED showing two independent static-eligible children can
  each appear frontend-C1 while the later resident `WorkItem` is C2. The old
  frontend concurrency guess must not be accepted as `C_due`.

Exit: one clean base and a precise gap list; no runtime change yet.

### D1 — split eligibility from cycle selection (RED/GREEN, host/fake first)

- [ ] Introduce or adapt a typed static eligibility/intention result that does
  not select future C/K. Keep model-plugin evidence and strict fallback.
- [ ] Submit eligible explicit requests into the ordinary Generation-2
  speculative-intent lifecycle without executing a whole-request legacy route.
- [ ] Resolve C/K only from the actual due `WorkItem`; preserve immutable
  pre-mutation `SpecRequestPlan` and atomic claim composition.
- [ ] Keep pure-AR requests provider-free. Preserve stable reasons for artifact,
  profile, sampler, context, horizon, memory, and capability misses.
- [ ] Distinguish pure and transitional K0 in plan/telemetry and bind provider
  catch-up behavior to `ProviderCatchupMode`.
- [ ] Make mixed due groups plan safely: an AR-only neighbor must not cause the
  adapter capability for eligible speculative peers to disappear, and it must
  not acquire provider state. Either one mixed frontier or explicit disjoint
  work items is acceptable if fairness, target ownership, and physical labels
  are exact.
- [ ] Response telemetry must distinguish requested intent, static eligibility,
  cycle K histogram, effective route, actual MTP cycles, and final K0 reason.

Required host/fake transition matrix:

```text
pure AR only
C1 MTP -> C1 MTP
C1 MTP -> C2 mixed -> C1 MTP
C2 MTP -> C2 transitional K0 -> C2 MTP
mixed K=(3,0), (0,3), (3,1), (1,3)
claim miss / context miss / physical-bucket miss / circuit break
precommit failure / postcommit recovery / cancellation
```

Exit: planner and telemetry are correct without GPU execution; automatic policy
remains unchanged.

### D2 — normal-owner functional C2 (speed is not a gate)

- [ ] Use two independent `EngineService`/OpenAI children, not one multi-prompt
  request and not `legacy_prelaunch_fallback`.
- [ ] Start explicit-only with production Q4_K_M C2/K1. Then cover K2/K3 after
  K1 ownership passes.
- [ ] Prove nonzero Generation-2 speculative cycles, one physical proposal group,
  target frontier R4/R6/R8, target execution, accept, selected commit, and
  committed publication.
- [ ] Assert no hidden whole-request fallback and no per-request full target
  backbone loop. Physical decomposition and weight sweeps are explicit.
- [ ] Compare output, target state, Conv/GDN, KV/`KVLiveSpans`, provider cursor,
  following AR, and final ownership against independent controls.
- [ ] Inject failure before proposal, after proposal, before target commit, and
  after selected commit; preserve both requests or fail only the named request
  according to the transaction contract.

Exit: real C2 MTP functions through the normal resident owner. It may be slower
than AR and remains explicit/default-off.

### D3 — dynamic switching and lifecycle

- [ ] Exercise delayed second arrival `C1 -> C2`, retirement/cancel `C2 -> C1`,
  and repeated `C1 -> C2 -> C1` without reconstructing the request scheduler.
- [ ] Exercise two speculative peers with different remaining horizons and
  accepted counts, including reject beside full accept.
- [ ] Exercise mixed permanent-AR and speculative-capable peers, transitional
  K0, provider catch-up, and later MTP re-entry.
- [ ] Exercise refill into a live provider group, survivor continuation, slot
  permutation/compaction, prefix restore/COW, pressure/regrow, and neighbor
  substitution.
- [ ] Exercise blocking plus SSE disconnect/deadline/backpressure, overload,
  restart, and repeated clean reopen.
- [ ] Assert zero cross-talk, duplicate/missing output, stale workspace reads,
  delayed OOM, sticky fallback, active claims, provider rows, pages, collectors,
  and background owners.

The first binding dynamic sequence is:

```text
A admitted/primed -> A executes MTP
B arrives and primes -> A+B execute C2 MTP
B cancels between cycles -> A executes MTP again
C arrives while A is transitional K0 -> exact catch-up -> A+C execute C2 MTP
A retires -> C survives -> final reclaim
```

Exit: switching is mechanically safe and repeatable. Width expansion and
production qualification are now admitted; performance tuning still waits for
D5. Automatic serving remains unchanged.

### D4 — width/packing expansion

- [ ] Wire the existing verifier-specific packing concepts into the actual
  Generation-2 fairness owner rather than creating a second scheduler.
- [ ] Bound groups by capability max requests, frontier rows, transaction bytes,
  deadlines, and round budgets before mutation.
- [ ] Prove C3/C4 using qualified physical groups and exact request mapping.
- [ ] For C5-C8, choose and label either multiple <=C4 frontiers or a newly
  qualified wider owner. A decomposed C8 route is not a physical C8 claim.
- [ ] Serve each due speculative request at most once before a peer repeats;
  preserve AR progress and SLO guards.
- [ ] Repeat shrinking, refill, mixed-K, failure, and ownership gates across
  every retained decomposition.

Exit: every C1-C8 width has an explicit functional MTP or typed pre-mutation K0
verdict. Only exact functional widths proceed to production/performance gates.

### D5 — Q4_K_M production correctness and serving qualification

- [ ] Bind the selected production and strict-fallback variant manifests for
  every C/K/R/route/decomposition. Q4_K_S evidence is control only.
- [ ] Run strict-teacher mean/p95/p99/max KL and top-1 gates by category, shape,
  transition, and accepted depth; generated-ID equality remains diagnostic.
- [ ] Run deterministic repeat, neighbor/permutation isolation,
  batch-composition invariance, BF16-relative/task gates where applicable, and
  finite-logit checks.
- [ ] Run the full committed code/general English/general Japanese/mixed suite
  plus heldouts under blocking and SSE.
- [ ] Bind dynamic transitions, K0 catch-up, failure recovery, memory/pressure,
  soak, and final drain to the production profile.

Exit: explicit production C>N MTP is qualified independently of speed.

### D6 — economics and optimization (only after D5)

For each eligible width and K, measure three distinct arms:

1. **true AR:** no speculative intent or provider ownership;
2. **intent K0:** provider activation/catch-up ownership present but cycle K=0;
3. **engaged MTP:** nonzero proposal/target/accept/commit cycles.

- [ ] Use same-host, same-process where ownership permits, counterbalanced true
  AR and MTP with exact authoritative token counts and complete wall.
- [ ] Report activation, provider priming/open, K0 catch-up, proposal, target,
  accept/commit, provider repair, scheduler, readback, reclaim, TTFT, ITL, E2E,
  goodput, memory, and occupancy separately.
- [ ] Start with C2 K1/K2/K3, then C4 and other functional widths. Do not tune
  C5-C8 before their physical decomposition and lifecycle are qualified.
- [ ] Profile final cached target R4/R6/R8/R12/R16 families and only implement
  kernel/dataflow work with a measured operation-complete premise.
- [ ] Keep every exact non-regressive owner win. Do not use prompt/token/category
  features, fixed-suite reranking, or verifier-derived K0 as the AR baseline.
- [ ] Rebuild the offline C/K/load LUT from retained evidence. Online adaptation
  is out of scope until a fixed cell wins and passes the complete gate.

Promotion requires `>=1.10x` true AR overall, no category/heldout/task/SLO
regression, complete production correctness, and exact lifecycle/ownership.
The project target remains `>1.30x`. A losing but correct explicit route may
remain default-off; a losing automatic cell selects pure K0 before provider
mutation unless a separately qualified transitional policy proves its shadow
cost acceptable.

### D7 — automatic policy and closure

- [ ] Publish exact static eligibility and dynamic cycle-policy fingerprints.
- [ ] Prove independent HTTP children engage only retained C/K/load cells and
  use true AR for every losing/unqualified cell before speculative mutation.
- [ ] Prove route changes occur only at transaction boundaries and terminal
  telemetry reports actual MTP use rather than requested intent.
- [ ] Run below/near/above-load, mixed AR/MTP, cancellation, overload, restart,
  and soak with SLO-goodput and complete drain.
- [ ] Remove superseded frontend concurrency guesses, temporary flags, duplicate
  compatibility routes, and stale evidence rows according to `REFACTOR.md`.
- [ ] Publish artifacts/worklogs and update benchmark rollup/changelog only for
  retained product/default results.

## 6. RED contract inventory

Minimum new or extended tests:

| Layer | Required RED |
| --- | --- |
| Static serving | independent frontend C1 children retain eligible intent without claiming resident C1/C2 |
| Cycle planner | actual due C, mixed K, pure vs transitional K0, exact reasons, zero mutation before plan |
| Packer | max physical group, R/bytes/deadline budgets, fairness, C5-C8 decomposition, AR progress |
| Provider | activation atomicity, K0 catch-up order, AR->MTP re-entry, MTP->AR, refill, one-way/disabled behavior |
| Target transaction | mixed accepted counts, cancellation at every boundary, selected-state commit, postcommit rebuild |
| Identity | request/slot/provider/frontier/physical-row permutation and compaction |
| Public API | two independent blocking/SSE clients, truthful requested/selected/effective route and usage |
| Lifecycle | delayed arrival, survivor/refill, pressure, overload, restart, soak, zero final ownership |
| Profile | Q4_K_M production KL/top-1/determinism/isolation/task gates by C/K/transition |
| Economics | true AR vs intent-K0 vs engaged-MTP denominator separation |

GPU/HIP tests require explicit availability guards. Kernel work, if later
admitted, follows the in-tree RED/oracle/lineage/rocprof rules in `AGENTS.md`.

## 7. Evidence and artifact contract

Each phase artifact records:

- exact clean commit, excluded concurrent work, physical host and device;
- model full hash/size/quant/KV/profile/manifest and prompt fixture hash;
- static eligibility key/decision/fingerprint and persistent claim outcome;
- `C_resident/C_due/C_spec`, K vector, logical R, proposal groups, target
  decomposition, slots, request/provider rows, and cycle fingerprint;
- pure K0 vs transitional K0 vs engaged MTP counts and reasons;
- provider activation/catch-up/open/close, cursor/state hashes, target/commit
  ownership, allocation/high-water/final conservation;
- exact command/environment/warmup/repeats/order/timing boundaries;
- correctness/profile/category/heldout/task/determinism/isolation verdicts;
- complete wall and stage/SLO metrics against true AR;
- physical kernel/marker/API/copy evidence where applicable; and
- retain/reject/K0 decision, blocker, and mechanical reopen trigger.

Raw logs/profiler dumps stay outside Git. Fixed-prompt screens are diagnostic;
performance promotion uses the full committed category suite plus heldouts.

## 8. Stop rules and non-goals

- No GPU execution during this design unit.
- Do not edit or append to the active tuning campaign while another owner is
  changing it; this separate ledger is the coordination surface.
- Do not replace `ResidentEngineLoop`, `SpecRequestPlan`, provider SPI,
  target-frontier, transaction, or Generation-2 output ownership.
- Do not revive `legacy_prelaunch_fallback` as a product route.
- Do not begin kernel tuning before normal-owner C2 and D3 transition gates pass.
- Do not call requested-MTP K0 timing MTP throughput.
- Do not assume fake, Q4_K_S, strict, capacity-1, or multi-prompt evidence
  qualifies Q4_K_M production independent-request serving.
- Do not require C5-C8 implementation before C2 proves the normal-owner seam;
  do require a truthful functional/K0 verdict for every width before closure.
- Do not weaken production numerical/task/SLO gates to retain a speed result.
- Do not introduce content-conditioned or benchmark-conditioned policy.
- gfx1100, PARO, DFlash, sampled speculation, trees, overlap, and KV quant are
  independent campaigns.

## 9. Coordination and file ownership

Likely high-conflict implementation files are:

```text
hipengine/server/api.py
hipengine/llm.py
hipengine/generation/engine_loop.py
hipengine/generation/qwen35_gguf.py
hipengine/generation/qwen35_gguf_mtp2.py
hipengine/speculative/frontier.py
hipengine/speculative/policy.py
hipengine/speculative/packing.py
hipengine/speculative/serving.py
hipengine/models/qwen35.py
```

One merge owner serializes changes across these files. The active singleton
capacity unit lands first. Later workers take disjoint phases or tests and do
not force-stage, restore, or overwrite another owner's changes. Hardware
execution uses an exclusive lane/lock and begins only after the current tuning
agent releases the machine.

## 10. Definition of done

The campaign closes only when:

1. frontend static eligibility no longer guesses independent-request `C_due`;
2. the resident scheduler selects K/K0 from the actual due group before
   mutation;
3. two independent public requests execute real normal-owner Q4_K_M production
   MTP with nonzero proposal/target/accept/commit evidence;
4. pure K0, transitional K0, MTP, C1<->C2+, cancellation, refill, survivor,
   compaction, pressure, failure, and terminal transitions pass;
5. every C1-C8 width has a truthful explicit-functional or typed K0 verdict;
6. all functional product candidates pass the binding production correctness,
   category/heldout, lifecycle, and SLO gates;
7. every promoted automatic cell beats true AR by at least 1.10x with no
   regression, while every losing cell selects pure K0 before provider mutation
   unless its separately measured transitional policy is retained; and
8. artifacts, worklogs, refactor cleanup, benchmark rollup, and strict fallback
   identities are complete.

Closure may retain no automatic C>N cell if correct normal-owner MTP still loses
true AR. That is a valid measured result. Closing before the normal-owner route
and transition matrix work is not.
