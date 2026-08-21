# MTP Real-World Readiness Campaign

- Status: **active campaign plan; implementation not started; production MTP remains disabled**
- Created: 2026-08-21
- Primary scope: Qwen3.6/Qwen3.8 dense GGUF NextN MTP on `hip_gfx1151`, then independently on `hip_gfx1100`
- Authority: [`PLAN.md`](PLAN.md), [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md), [`TESTING.md`](TESTING.md), and [`BENCHMARK.md`](BENCHMARK.md) remain normative

This campaign turns the current benchmark-qualified MTP implementation into a
feature that is safe and useful under real server workloads. It is deliberately
broader than removing the 1024-token guard. The campaign is complete only when
an eligible request can use MTP across its declared context, lifecycle, API,
quality, and load envelope—or can fall back to AR before speculative state is
mutated—without hangs, state corruption, silent semantic changes, or misleading
telemetry.

Related design and historical evidence:

- [`MTP.md`](MTP.md) — implementation history, economics, and provider design;
- [`NATIVE_SPEC_CYCLE.md`](NATIVE_SPEC_CYCLE.md) — N0–N5 ownership milestones;
- [`MTP-LLAMACPP-PARITY.md`](MTP-LLAMACPP-PARITY.md) — external comparison protocol;
- [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) — numerical/control contracts;
- [`CONCURRENCY.md`](CONCURRENCY.md) — scheduler and dynamic-serving scenarios;
- [`API.md`](API.md) — current OpenAI-compatible routing and telemetry;
- [`REFACTOR.md`](REFACTOR.md) — temporary flags and rollback paths.

## 1. Executive decision

The present dense MTP route is **not production-ready** despite having strong
short-context benchmark evidence. The running production server should keep
`HIPENGINE_SPECULATIVE_MTP_SERVING=off` until at least **RF0–RF4** below pass.
The public default must not return to automatic MTP until the complete **RF6
qualification packet** passes for the exact model/backend/profile scope being
enabled.

The first implementation objective is **containment**, not speed:

1. Never launch a device proposal unless a compatible target graph is known to
   be launchable at the request's current context and output room.
2. Fall back before speculative mutation when a context/shape is unsupported.
3. Once proposal or target work is in flight, retire or roll it back through one
   owned transaction; never attempt an unsafe second execution path.
4. Ensure cancellation, deadline, EOS, stop, failure, and teardown all leave the
   target, draft, scheduler, KV, and graph caches reusable or explicitly dead.

Only after those properties hold do we extend the fast graph beyond 1024 and
optimize long-context wall time.

## 2. Product goal: what “works in real life” means

MTP readiness has four separate dimensions. Passing one does not imply the
others.

| Dimension | Required product behavior |
| --- | --- |
| Availability | Every request is classified before mutation as qualified MTP, safe pre-launch AR fallback, or explicit unsupported request. No request enters a known-ineligible graph path. |
| Semantic/control correctness | Request identity, positions, `KVLiveSpans`, recurrent state, target/draft KV, acceptance, commit, rollback, output limits, EOS/stop, cancellation, and teardown are exact. |
| Numerical/task quality | The declared execution profile passes its strict-teacher distribution, determinism, category, heldout, and task gates. “Rare argmax differences” are measured and scoped, not hand-waved. |
| Acceleration | MTP improves complete request economics against a true no-MTP AR path in the same protocol and does not create unacceptable TTFT/ITL/p99/memory regressions. |

A production deployment is allowed to use a mixed policy: qualified requests
run MTP and every other request runs AR. “MTP works” does **not** require forcing
MTP onto sampling modes, tools, streaming, contexts, or models that have not
been implemented and qualified. It does require transparent, deterministic,
and safe routing for all of them.

## 3. Current implementation audit

### 3.1 What is already real

- Dense GGUF NextN weights are detected and materialized.
- The target-attached draft provider, transactional verifier, GPU acceptance,
  selected target-state commit, draft repair, and bounded result accounting are
  implemented.
- EOS/stop-token overshoot is trimmed through scheduler completion/reclaim.
- Short-context reusable N1/N2 target graphs and device proposal chaining have
  retained gfx1100/gfx1151 evidence.
- The server has explicit MTP routing, greedy-sampling guards, a thinking policy,
  per-request acceptance reporting, Prometheus metrics, and an AR opt-out.
- Reject, partial-accept, full-accept, rollback, selected-state, and graph reuse
  have substantial focused coverage.

These are necessary foundations. They do not establish the real-world envelope.

### 3.2 Blocking findings

| ID | Severity | Finding | Current evidence / code |
| --- | --- | --- | --- |
| C0 | Critical | A cached below-1024 N2 graph remains “device proposal ready” after the live target cursor has crossed the graph's context limit. The draft proposal may be launched, after which the target graph rejects the context. In-flight proposal errors are deliberately not allowed to fall back, producing the observed long-context failure/wedge. | `Qwen35GGUFTransactionalVerifier.device_proposal_ready()` checks cache/config compatibility but not `target.position + rows <= graph.context_limit`; `Qwen35GGUFTransactionalVerifier.prepare()` re-raises misses after a device proposal exists. |
| C1 | Critical | The reusable target graph is structurally captured for a static `context_limit=min(1023, max_positions)`. Long contexts switch attention ownership at 1024 to split-K/workspace-backed routes. Removing only the admission guard would bind the wrong graph/workspace policy or capture an impractically large static context. | `gguf_native_spec_cycle.py` admission, graph `context_limit`, dynamic spans, and long-context split threshold in `qwen35_gguf_runner.py`. |
| C2 | High | Dense default admission is model-level (`has MTP tensors` and `not MoE`), not request/context/profile/backend/resource-level. It can advertise default-safe MTP for a request outside the graph's qualified envelope. | `Qwen35GGUFTextGenerator.supports_default_mtp`; server `enabled` policy. |
| C3 | High | Deadline/cancellation is checked around the dense generation call, but the inner one-request MTP cycle loop has no per-cycle cancellation/deadline poll. A timed-out HTTP request may stop waiting while GPU work continues. | `_generate_dense_speculative_mtp_detailed()` and `Qwen35GGUFMTPDecodeSession.generate()`. |
| C4 | High | Dense multi-prompt MTP currently iterates request rows serially through one target session. Server batching tests prove route grouping, not true concurrent dense-MTP execution, fairness, isolation under load, or aggregate benefit. | `_generate_dense_speculative_mtp_detailed()` loops over encoded prompts with `resident_slot_count=1`. |
| C5 | High | No binding long-context MTP matrix covers the 1024 transition, page boundaries, 4K–64K contexts, prompt-plus-decode crossings, graph/eager parity, state/KV ownership, teardown, and task quality together. | Historical short/category suites and separate KV long-context harnesses do not form this packet. |
| C6 | Medium | The default `thinking=hint` policy removes host-sampler thinking-budget enforcement. It may be a valid product mode, but it is not semantically identical to the original hard-controlled request and needs explicit API labeling plus task-quality evidence. | Server routing relaxes `thinking_budget`; prompt hints remain but soft-close/EOS suppression/hard-close controls do not. |
| C7 | Medium | Streaming, logprobs, tools/grammars, penalties, non-greedy sampling, token stop sequences, and several structured controls fall back or reject. This is acceptable only if capability and per-request route telemetry are exact and tested. | `SPECULATIVE_MTP_INCOMPATIBLE_FIELDS`, API routing, `streaming_compatible=false`. |
| C8 | Medium | A GPU hang/fatal HIP error has no MTP-specific circuit breaker or worker quarantine policy. One bad graph can make the entire server unhealthy. | Current route has rollback for owned Python exceptions, but no post-fatal process recovery contract. |
| C9 | Medium | Existing observability reports use/counts, but not a stable reason taxonomy for pre-launch fallback, graph miss, context-bucket transition, cancellation phase, post-launch failure, quarantine, or fallback wall cost. | Current response summary and `hipengine_mtp_*` metrics. |

### 3.3 Important distinction: eager long-context MTP versus fast graphs

The 1024 guard belongs to the reusable native target graph, not to the abstract
MTP transaction. The eager native verifier has existing long-context attention
machinery, including split-K paths, but it has not been qualified as the
production long-context MTP fallback. The campaign must prove that path rather
than assume it.

This yields two deliverables:

1. **Functional long-context MTP:** host-materialized proposal plus eager native
   verify/commit works correctly across long contexts.
2. **Fast long-context MTP:** context-bucketed reusable proposal/target graphs
   use the correct split-K workspace and improve complete wall.

The first may land before the second. Default routing can use AR where eager MTP
is correct but slower.

## 4. Non-negotiable invariants

These bind every phase and execution profile.

1. **Pre-launch fallback only.** Unsupported context, budget, output room,
   graph bucket, memory, profile, or request semantics must be decided before
   proposal/target mutation. A post-launch error may roll back the owned
   transaction, but must not silently execute AR or eager MTP a second time.
2. **One transaction owns one cycle.** Proposal identity, target rows,
   acceptance, target/draft commit, output IDs, and cursor updates use one
   request/transaction generation. Stale generations are rejected.
3. **Exact control ownership.** Request IDs, row maps, positions, causal limits,
   `KVLiveSpans`, page tables, live counts, Conv/GDN state, target KV, draft KV,
   output queues, and metrics belong to the correct request at every transition.
4. **Prefix-closed commit.** Rejected draft suffixes never become visible and
   never remain reachable in authoritative target or draft state.
5. **Bounded work.** Every cycle has an output-room check, cancellation/deadline
   observation point, and scheduler yield boundary. No native loop may hide
   unbounded work.
6. **Graph keys describe execution.** A graph key includes every property that
   changes launches, allocations, arithmetic, or ownership. A cached graph is
   never treated as compatible merely because candidate budget matches.
7. **No silent semantic relaxation.** Sampling/thinking controls that are
   dropped or changed are reported as a distinct policy decision. Otherwise the
   request falls back to AR.
8. **Strict fallback remains available.** Every native/fused production owner
   retains its declared eager/unfused strict fallback.
9. **No benchmark-specific policy.** Routing may use model/backend/profile,
   request semantics, context bucket, width, output room, memory, and validated
   online signals. It may not use prompt names, text, token IDs, or heldout
   outcomes.
10. **Failure is observable.** Route, reason, graph bucket, actual verifier,
    fallback, commit status, and failure/quarantine state are emitted without
    exposing prompt content.

## 5. Target serving architecture

### 5.1 Immutable per-request MTP plan

Before queue admission, construct a cold-path plan with at least:

```text
MTPRequestPlan
  requested_policy          off | opt_in | auto | enabled
  selected_route            ar | mtp
  decision_reason           stable enum
  model/backend/quant/KV/profile identities
  prompt_tokens / max_new_tokens / max_total_tokens
  candidate_budget
  proposal_mode
  target_verify_mode        serial_exact | eager_native | native_graph
  context_bucket
  attention_route           short_batch | split_k | other registered owner
  graph_key / variant-manifest hash
  strict fallback key
  memory reservation
  sampling/thinking policy
```

The plan is immutable for one admitted request except at explicit cycle
boundaries where a separately validated context-bucket transition occurs.
Responses and metrics report the realized plan, not merely server configuration.

### 5.2 Context-bucketed graph ownership

The target graph cache must not remain keyed only by draft budget. Its effective
identity must include, directly or through an immutable configuration key:

- backend, model/weight identity, quant, KV format, and execution-profile
  manifest hash;
- candidate budget / rows;
- context bucket and attention route;
- block/page size, split count, and split-workspace capacity;
- target/draft allocation generations and binding signatures;
- device accept/commit mode and captured state surfaces;
- stream/capture policy;
- output-room/tail class where launch shape differs.

Recommended context buckets are powers of two or measured route boundaries, but
must include a dedicated short bucket ending at 1023 and split-K buckets above
it. Bucket choice must use the **cycle end** (`position + verifier_rows`), not
only the prompt length or current root position.

Capturing a new target graph is forbidden while a device proposal is in flight.
A transition follows one of these safe sequences:

```text
cached compatible target graph -> launch device proposal -> launch target graph
```

or

```text
no compatible target graph -> host proposal -> eager verify
                         OR -> pre-capture target graph -> host/device proposal next cycle
```

Never:

```text
launch device proposal -> discover target graph miss -> try eager fallback
```

### 5.3 Capability and routing policy

The long-term production default should be `auto`, not “force MTP everywhere.”
Mode semantics:

| Mode | Contract |
| --- | --- |
| `off` | Always AR; MTP resources need not be prepared. |
| `opt_in` | Only explicit requests are considered; unsupported requests fail before queue admission with a structured reason. |
| `auto` | MTP only for certified request-plan scopes; all other requests use AR and report the fallback reason. This is the intended eventual default. |
| `enabled` | Prefer MTP for every semantically compatible request, but still fail closed to certified AR before mutation when the runtime scope is unsupported. It must never override safety gates. |

`supports_default_mtp: bool` is too coarse as the final admission API. Replace or
supplement it with a model/plugin capability that answers a concrete request
plan and returns a stable reason on rejection.

### 5.4 Failure containment

Classify failures by mutation boundary:

| Class | Required action |
| --- | --- |
| Admission/preflight miss | Safe AR fallback or explicit unsupported response; no MTP state exists. |
| Pre-launch graph/capture miss | Close partial graph/workspace; use eager MTP or AR according to the immutable plan; no in-flight proposal. |
| In-cycle Python error before target mutation | Retire/cancel proposal if launched; roll back draft transaction; request may fail clearly. |
| Target transaction error after provisional mutation | Restore journal and cursors, roll back scheduler KV transaction, invalidate affected graph generation; no replay in the same cycle. |
| Fatal HIP error/hang/watchdog | Mark worker unhealthy, stop admissions, emit failure reason, and let an external supervisor restart the process. Do not claim in-process GPU recovery. |

Add an MTP circuit breaker: repeated graph/runtime failures for one qualified
scope disable that scope for new requests and route them to AR until restart or
an explicit operator reset. The breaker key must be model/backend/profile/
context-bucket specific so one bad long bucket does not disable unrelated AR.

## 6. Campaign phases

### RF0 — Contain the current 1024 failure

**Goal:** long requests cannot launch an incompatible device proposal or wedge
the server.

Implementation:

- Add a side-effect-free graph launch-eligibility method covering current
  position, verifier rows, `context_limit`, remaining output, graph state,
  binding generation, and configuration key.
- Make `device_proposal_ready()` call that method before draft launch.
- Recheck the same eligibility immediately before target submission to detect
  impossible cursor/generation drift; drift is an error, not fallback.
- Add a stable reason such as `target_graph_context_bucket_miss`.
- Change production/server default back to `auto` or `off` until RF6; keep
  explicit opt-in for controlled validation.

RED tests before implementation:

- cached B1/B2/B3 graph at its last valid start position;
- start/end exactly below, at, and above 1024;
- cached short graph reused after several cycles cross 1024;
- assertion that no device proposal launcher is called on an ineligible cycle;
- injected cursor change between readiness and target launch fails without a
  second execution path;
- graph invalidation/close and subsequent AR request remain healthy.

Exit gate:

- all focused tests pass;
- a real prompt whose decode crosses 1024 completes or pre-launch-falls back;
- no hang, leaked transaction, stale graph, or subsequent AR corruption;
- route telemetry names the miss.

### RF1 — Prove eager long-context MTP correctness

**Goal:** establish whether functional MTP can run beyond 1024 without reusable
target graphs.

Implementation/measurement:

- Force host-materialized proposal and eager native target verification.
- Exercise the existing long-context split-K attention owner and verify its
  workspace allocation, spans, split count, and state journal.
- Compare every selected target state surface and committed draft state against
  serial AR/strict teacher rows.
- Measure wall/memory, but do not require a speed win for this phase.

Required contexts:

- prompt/decode cycle ends around `1016..1032` for B1/B2/B3;
- page boundaries around the configured block size;
- 2K, 4K, 8K, 16K, 32K, and 64K where the model/hardware gate permits;
- largest certified context and near-cap output-room tails.

If eager MTP fails correctness, stop and localize with layer/state/KV ladders.
Do not proceed to graph capture. If it passes but is slower than AR, retain it as
an oracle/fallback only and route production traffic to AR in those buckets.

### RF2 — Add fast long-context graph buckets

**Goal:** reusable MTP target graphs execute the correct long-context attention
route without static-max-context waste.

Implementation:

- Introduce context-bucketed N1/N2 graph cache entries.
- Allocate/bind row-sized split-K workspace for the captured bucket.
- Keep live position/context/page metadata dynamic within the static bucket.
- Capture each bucket before device-proposal chaining is admitted.
- Close and rebuild on allocation generation, profile manifest, KV policy,
  split policy, or binding drift.
- Bound cache count and memory; eviction closes graph exec, stream, workspace,
  and borrowed descriptors safely.

Correctness gates:

- graph/eager target top-1, selected hidden, Conv/GDN, target KV, draft KV,
  accepted prefix, correction token, and cursors;
- first capture, steady replay, bucket transition, eviction/re-capture;
- reject, every partial depth, and full accept;
- context/page boundaries and long-context tasks;
- same-schedule repeat determinism.

Performance gate:

- compare complete MTP wall to eager MTP and true AR by context/category;
- graph capture cost is reported separately and amortized only according to the
  declared request distribution;
- a bucket is auto-routable only when its complete request/SLO packet is
  non-regressive in its qualified scope.

### RF3 — Lifecycle, termination, and failure correctness

**Goal:** MTP behaves like a server request, not a benchmark loop.

Add explicit cycle-boundary checks for:

- cancellation token;
- absolute deadline/request timeout;
- server shutdown/drain;
- output-room tail;
- EOS and stop-token completion;
- scheduler yield/fairness.

Lifecycle rule: cancellation observed before cycle launch aborts without
mutation. Cancellation arriving during an in-flight cycle lets the owned cycle
retire or roll back, suppresses publication beyond the cancellation boundary,
then reclaims all state. It must not interrupt between target commit and draft
repair in a way that leaves cursors inconsistent.

Fault-injection matrix:

- before proposal;
- after proposal launch but before target launch;
- during/after target verify;
- after accept before target commit;
- after target commit before draft repair;
- after draft repair before output publication;
- during EOS trim;
- graph capture/instantiate/launch/readback failure;
- allocation failure and shutdown while queued/running.

For every injection assert request completion/error, transaction terminal state,
KV/state/cursor ownership, graph validity/invalidation, resource counts, server
readiness, and a clean subsequent AR then MTP request.

### RF4 — Real API semantics

**Goal:** routing preserves the user's declared request semantics.

Required endpoint matrix:

- `/v1/completions` and `/v1/chat/completions`;
- single and multi-prompt payloads;
- implicit `auto`, explicit true/false, and every server policy;
- lazy/eager model load and restart;
- app-local chat sessions;
- supported greedy requests and every declared fallback/rejection field.

Policy decisions:

- Keep streaming on AR until MTP has a real incremental publication contract.
- Keep penalties, non-greedy sampling, logprobs, tool grammar, structured
  forcing, and unsupported stop-sequence semantics on AR.
- Treat `thinking=hint` as an explicit product policy. Report that hard
  enforcement was relaxed; separately gate answer quality, closure rate,
  runaway reasoning, token use, and task score. `thinking=hard` must preserve
  controls by using AR until processed-logit MTP exists.
- EOS and tokenizer-default EOS must be distinguished from user sampler
  overrides. Multi-token text stop sequences remain AR unless implemented in
  the cycle publication contract.

Implicit auto fallback should succeed as AR. Explicit MTP outside its certified
scope should fail before admission with a stable structured reason unless the
public API explicitly promises fallback for explicit requests.

### RF5 — Concurrency, fairness, resources, and soak

**Goal:** MTP remains correct and beneficial under actual offered load.

First document the dense path honestly: current multi-prompt generation is
serial over one resident target slot. Do not use route-coalescing tests as proof
of physical MTP concurrency.

Required serving scenarios:

- c1/c2/c4/c8 fixed widths;
- independent requests and one multi-prompt request;
- ragged prompt/decode lengths and mixed context buckets;
- delayed admission while MTP is decoding;
- MTP and AR requests sharing the server;
- sparse retirement, cancellation, and re-admission;
- queue saturation/rejection and fairness;
- graph cache churn and bounded memory;
- shutdown/drain and repeated server restart.

Ownership gates follow `EXECUTION-PROFILES.md`: neighbor replacement, row
permutation, inactive rows, slot changes, and cancellation may not contaminate
another request. Production may use width-specific arithmetic, but same-schedule
determinism and same-width isolation bind.

Resource accounting must include:

- target weights and repacks;
- MTP/NextN weights;
- target KV and draft KV by request/context;
- recurrent/state journals and rollback snapshots;
- per-budget/per-context graph execs and workspaces;
- pools/high-water marks, fragmentation, and post-request/post-shutdown return.

Soak tiers:

- focused 100-request mixed AR/MTP lifecycle run;
- 1-hour mixed-context/load run for campaign iterations;
- final 8-hour or equivalent request-count promotion soak with no hang, leak,
  stale route, ownership failure, or unexplained readiness transition.

### RF6 — Quality and performance qualification

**Goal:** certify named model/backend/profile/context scopes for automatic use.

Quality packet:

- full `mtpbench-code-general-ja.jsonl` train+heldout/category suite;
- category-heldouts and no single-prompt keep decisions;
- strict-teacher mean/p95/p99/max KL and top-1 per
  `EXECUTION-PROFILES.md` for production arithmetic;
- same-schedule determinism (three repeats);
- code/task, multilingual, structured-output where supported, and reasoning
  quality for the selected thinking policy;
- long-context retrieval/multihop/aggregation/long-document/code tasks using a
  committed MTP-specific long-context fixture derived from the existing KV
  suites, not repeated-token filler alone;
- exact control/ownership and finite state/logits at all recorded boundaries.

Performance packet:

- true no-MTP AR and MTP in one same-host, same-model, same-quant, same-KV,
  same-prompt, same-warmth protocol;
- natural short/category, context-boundary, 4K/16K/long, and mixed-context rows;
- TTFT, decode throughput, complete HTTP wall, ITL median/p95/p99, queue delay,
  cancellation latency, and memory high-water;
- cold first capture and warm replay reported separately;
- acceptance/output, acceptance/draft, cycles, proposal/verify/commit split, and
  fallback counts/reasons;
- c1 and admitted concurrent widths.

Promotion rules:

- Functionality/opt-in may be retained when correctness/lifecycle pass even if
  MTP is slower, but that scope remains non-auto and carries no speed claim.
- Automatic MTP requires a true-AR ratio above the repository's speculative
  promotion floor (currently `>1.10x` in `BENCHMARK.md`) plus non-regressive
  SLO/task/resource gates in every admitted category/context scope.
- The project target remains `>1.3x`; do not lower a predeclared gate after
  seeing results.
- A short-context win cannot authorize long contexts. Route qualification is
  context/backend/model/profile scoped.
- llama.cpp MTP-off and MTP-on are both useful external diagnostics, but neither
  substitutes for hipEngine's true AR denominator or predicts hipEngine speed.

### RF7 — Staged rollout and removal of emergency controls

Rollout order:

1. local direct harness, MTP explicit only;
2. local server explicit only;
3. one canary worker with `auto`, short certified buckets only;
4. canary expands one context/backend scope at a time;
5. production percentage rollout with circuit-breaker and AR fallback metrics;
6. default `auto` only after the final soak and rollback drill;
7. consider removing temporary flags only after one release window and a new
   immutable worklog decision.

Rollback drill must prove an operator can switch to AR for new requests without
restarting, allow or cancel in-flight owned cycles safely, drain MTP resources,
and preserve readiness. Fatal GPU errors still require worker restart.

## 7. Validation matrix

The matrix is layered to avoid an unbounded Cartesian product while still
covering interactions that benchmarks missed.

### 7.1 Tier A — every change / CPU and fake-runtime RED tests

- graph eligibility at boundary and stale generation;
- no proposal launch on target miss;
- route reason and immutable request plan;
- transaction state machine and prefix-closed commit;
- cancellation/deadline phase injection;
- EOS/stop/output-tail accounting;
- graph-key completeness and cache eviction lifecycle;
- telemetry ownership/counting;
- no-ROCm skip guards for tests that load HIP.

### 7.2 Tier B — focused real-GPU gate

- B1/B2/B3 reject/partial/full;
- contexts around 1024 and one 4K case;
- eager/graph parity where graph is admitted;
- state/KV/cursor oracle;
- cancellation and subsequent-request health;
- expected kernel trace and graph/fallback telemetry.

### 7.3 Tier C — nightly/campaign gate

- full category+heldout suite;
- contexts 512, boundary, 4K, 16K, and largest practical long fixture;
- c1/c2/c4/c8 dynamic lifecycle subset;
- three deterministic repeats;
- quality/task and same-protocol true-AR economics;
- memory and graph-cache high-water.

### 7.4 Tier D — promotion gate

- all RF6 quality/performance rows;
- final long-context task suite;
- mixed AR/MTP load and final soak;
- rollback/circuit-breaker/restart drill;
- exact artifact provenance and benchmark rollup updates.

### 7.5 Boundary cases that must appear explicitly

| Surface | Cases |
| --- | --- |
| Context route | cycle end below/at/above 1024; every attention-route threshold; context bucket transition during decode |
| Pages | before/at/after page boundary; new page allocation; last allocated page; max-context rejection |
| Budget/output | B1/B2/B3; remaining output `1`, `B`, `B+1`; max-token completion after reject/partial/full |
| Termination | EOS as first target token, each accepted draft depth, correction token; explicit stop token; unsupported stop sequence AR fallback |
| Acceptance | reject depth 0; every partial depth; full accept; repeated zero accept; high acceptance |
| Graph | cold capture; replay; bucket miss; incompatible binding; close/evict/re-capture; capture failure; launch failure |
| Lifecycle | queued cancel; pre-cycle cancel; in-cycle cancel; deadline; shutdown; exception at every transaction phase |
| Composition | slot permutation; neighbor replacement; mixed contexts; AR+MTP; retirement/re-admission; c1↔cN |
| Resources | allocation failure; graph/workspace cap; repeated requests; post-cancel and post-shutdown return |

## 8. New campaign harness and evidence contract

Create one orchestrator, tentatively
`scripts/gguf_mtp_realworld_gate.py`, that reuses focused existing runners rather
than reimplementing model logic. It should emit one compact artifact with:

```text
schema/kind/status/verdict
repo commit + dirty state
physical host/hardware/software
model hash / quant / KV / profile manifest
prompt-suite and fixture hashes
request-plan and route-decision counts
context/budget/concurrency/lifecycle scenario results
strict-teacher quality summary
state/KV/transaction/determinism/isolation verdicts
AR and MTP timings/SLOs
acceptance and cycle economics
fallback/failure/circuit-breaker taxonomy
memory/graph-cache high-water and final ownership
commands and raw artifact hashes
```

Planned committed fixtures:

- `benchmarks/prompts/mtp-realworld-long-context.jsonl` — retrieval, multihop,
  aggregation, long-document, code, and mixed-language cases at deterministic
  target lengths;
- a small boundary fixture that places cycle ends around 1024/page transitions;
- a lifecycle schedule fixture for cancellation, delayed admission, retirement,
  and mixed AR/MTP requests.

Do not tune on the heldout rows. Repeated-token prompts remain mechanical
smokes only. Performance claims continue to use the canonical category suite and
true AR protocol from `BENCHMARK.md`.

## 9. Required telemetry and operator view

Per request:

- requested/effective route and stable decision reason;
- model/backend/profile/context bucket/candidate budget/verify mode;
- used MTP, cycles, proposed/accepted/rejected drafts, acceptance;
- graph captures/replays/misses and eager/AR fallback;
- cancellation/deadline/termination phase;
- transaction terminal status;
- circuit-breaker/quarantine state when relevant.

Prometheus additions/refinements:

- requests by effective route and decision reason;
- graph capture/replay/miss/failure by context bucket;
- eager-MTP and AR fallback totals;
- pre-launch versus post-launch failures;
- cancellation/deadline counts and latency;
- circuit-breaker state/transitions;
- MTP target/draft KV, graph-workspace, and cache high-water;
- complete request and cycle latency histograms, not only cumulative acceptance.

Startup/capabilities must distinguish:

- configured policy;
- model has MTP tensors;
- backend/provider support;
- certified default scopes;
- current circuit-breaker state;
- streaming/sampling/tool limitations;
- loaded graph/context buckets only as runtime state, not capability proof.

## 10. Deliverables and dependency order

| Deliverable | Depends on | Completion evidence |
| --- | --- | --- |
| D0 context-aware proposal/target admission | none | RF0 RED/GREEN + real crossing smoke |
| D1 request-plan/reason taxonomy and safe auto fallback | D0 | server route/API tests and telemetry |
| D2 eager long-context oracle | D0 | RF1 state/KV/task matrix |
| D3 bucketed split-K N1/N2 graphs | D2 | RF2 graph/eager/context packet + trace |
| D4 cycle cancellation/deadline/shutdown controls | D0 | RF3 fault-injection matrix |
| D5 circuit breaker and worker health contract | D4 | failure/restart drill |
| D6 dense concurrency implementation or honest serialized policy | D1–D4 | RF5 ownership/fairness/load packet |
| D7 real-world campaign harness/fixtures | D1–D6 | schema tests and reproducible compact artifact |
| D8 quality/performance qualification | D7 | RF6 retained/rejected artifacts |
| D9 canary/default rollout | D8 | RF7 soak and rollback drill |

D0/D1/D4 are safety work and precede performance optimization. D3 is the actual
long-context fast-path implementation. D6 must not claim concurrency if the
chosen first release remains serialized.

## 11. Campaign completion criteria

The campaign is complete for a named `(model, backend, quant, KV, execution
profile)` only when all statements below are true:

- [ ] No known context, budget, graph, or resource miss can occur after device
      proposal launch.
- [ ] Boundary, page, 4K, 16K, and declared maximum-context MTP requests pass
      control/state/KV/task gates or pre-launch route to AR.
- [ ] Cancellation, deadline, EOS/stop, output tail, failure, shutdown, and
      restart gates pass with clean resource ownership.
- [ ] Every supported API mode preserves its declared semantics; every
      unsupported mode has a tested AR fallback or explicit pre-admission error.
- [ ] Same-schedule determinism and same-width isolation pass; dynamic
      composition cannot cross-contaminate requests.
- [ ] Mixed AR/MTP load meets fairness, queue, memory, and SLO gates.
- [ ] Full category+heldout and long-context task quality pass the declared
      strict/production contract.
- [ ] Every auto-routed scope beats true same-protocol AR by the predeclared
      promotion floor and is non-regressive on task/SLO/resource gates.
- [ ] Capabilities, responses, logs, metrics, and artifacts report the realized
      route and reason accurately.
- [ ] Final soak and rollback/restart drills pass with no hang, leak, stale
      graph, ownership mismatch, or unexplained readiness loss.
- [ ] Benchmark artifact, rollup, changelog, immutable worklog, and any
      `REFACTOR.md` flag entries are current.

Anything less may be a useful MTP diagnostic or explicit opt-in, but it is not a
production-ready default.

## 12. First implementation slice

The first coding unit should be deliberately small and safety-only:

1. RED tests for cached short N2 graph eligibility at/crossing 1024 and proof
   that the device proposal launcher is not called on a miss.
2. Add graph `can_launch(position, rows, remaining_decode, binding_generation)`
   or equivalent side-effect-free admission.
3. Make `device_proposal_ready()` context/output aware.
4. Preserve host-proposal + eager verifier fallback only when no device proposal
   has launched.
5. Emit `target_graph_context_bucket_miss` telemetry.
6. Run focused unit tests, real 1024-crossing MTP/AR health smoke, rollback,
   subsequent AR/MTP request, and worklog/commit.

Do **not** remove `end >= 1024`, change `context_limit=min(1023, ...)`, or claim
long-context MTP in this first unit. Those belong to RF1/RF2 after the existing
long-context eager/split-K path has a direct oracle.
