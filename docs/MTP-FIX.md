# MTP Real-World Readiness Campaign

- Status: **complete on gfx1151; RF0–RF7 closed; no automatic MTP scope promoted; production `auto` routes AR and explicit diagnostic MTP remains available**
- Created: 2026-08-21
- Primary scope: Qwen3.6/Qwen3.8 dense GGUF NextN MTP on `hip_gfx1151`, then independently on `hip_gfx1100`
- Authority: [`PLAN.md`](PLAN.md), [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md), [`TESTING.md`](TESTING.md), and [`BENCHMARK.md`](BENCHMARK.md) remain normative

This campaign turns the current benchmark-qualified MTP implementation into a
feature that is safe and useful under real server workloads. It is deliberately
broader than removing the 1024-token guard. The campaign is complete only when
an eligible request can use MTP across its declared context, lifecycle, API,
quality, and load envelope—or can fall back to AR before speculative state is
mutated—without hangs, state corruption, silent semantic changes, or misleading
request/cycle records.

Related design and historical evidence:

- [`MTP.md`](MTP.md) — implementation history, economics, and provider design;
- [`NATIVE_SPEC_CYCLE.md`](NATIVE_SPEC_CYCLE.md) — N0–N5 ownership milestones;
- [`MTP-LLAMACPP-PARITY.md`](MTP-LLAMACPP-PARITY.md) — external comparison protocol;
- [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) — numerical/control contracts;
- [`CONCURRENCY.md`](CONCURRENCY.md) — scheduler and dynamic-serving scenarios;
- [`API.md`](API.md) — current OpenAI-compatible routing and response reporting;
- [`REFACTOR.md`](REFACTOR.md) — temporary flags and rollback paths.

## Campaign punchlist and coder handoff

**As of 2026-08-22.** This is the resumable implementation ledger. A checked
item is complete and must not be redone unless a later regression invalidates
its recorded evidence. An unchecked item is not qualified merely because a
partial probe passed.

**GPU lease:** this primary `MTP-FIX` coder session has exclusive ownership of
the GPU. The prior monolithic 32K/64K process (PID 3648063) is no longer present.
No other agent may start a GPU test, benchmark, profiler, model load, server, or
background GPU task unless the user explicitly reassigns or shares the device.
Other agents must restrict themselves to CPU/read-only work and must not infer
that an idle utilization sample releases the lease. Before a long gate, state
the command, reason, expected duration, and stop budget. On handoff, stop or
identify every surviving GPU process/background task.

### Completed and committed

- [x] **Campaign audit and plan:** commit `c964f8370` created this campaign and
      recorded C0–C9, RF0–RF7, validation tiers, and promotion rules.
- [x] **RF0 containment:** commit `9aa25016e` added context/output/binding-aware
      graph admission, prevented incompatible device proposals, retained safe
      pre-launch eager fallback, added stable miss reasons, and changed implicit
      server policy to fail-closed `auto`. The recorded gfx1151 27B crossing
      gate matched 12 AR IDs and a subsequent AR health request passed.
- [x] **RF1 oracle harness:** commit `d83879ff0` added
      `scripts/gguf_mtp_long_context_gate.py`, focused schema/unit coverage, and
      the eager-native versus serial-exact state/KV/commit gate. This completes
      the harness, **not** RF1 qualification.
- [x] **Page-boundary RF1 packet:** the existing RF1 run passed all 9 direct
      page-boundary cases (`/tmp/mtp-rf1-pages.json`, 101.88 s). Preserve this
      result; rerun only if the eventual fix changes relevant attention,
      state/KV, commit, or rollback behavior.
- [x] **Failure localization for the 1024 transition:** the original boundary
      packet passed 18/19 direct cases and its real generation case. The sole
      failure was `end1024-b3-a3` (B3 full accept ending exactly at 1024):
      top-1, accepted count, and rollback were exact, while full logits,
      selected state/KV/hidden, and commit were not. Live-prefix KV, every
      recurrent state, hidden/root, and cursor were independently proven exact
      before verification. The first verifier-output drift was layer 46 row 3
      (295 BF16 elements, max abs `0.015625`), in the staged multi-row
      linear-attention/FFN path before later KV propagation. Do not repeat the
      prefix or layer bisects unless new evidence contradicts them.

### Resolved RF0 planner questions

These decisions describe the retained RF0 implementation. They are not open
questions for the next coder.

1. **Base/starting point — validate forward, do not reimplement.** Start from a
   clean commit containing `9aa25016e` (and normally `d83879ff0`, which builds
   RF1 on it). RF0 is committed and validated; do not re-derive or port an older
   uncommitted scaffold. If a separate branch lacks RF0, cherry-pick the atomic
   RF0 commit rather than manually recreating its hunks. The only implementation
   state that remains dirty is the RF1 candidate identified below.
2. **Draft-side defense — keep the target verifier authoritative.** Do **not**
   tighten the draft executor's independent `position + budget` graph guard to
   `position + budget + 1`. The draft graph executes B proposal rows and its
   existing limit accurately describes that provider. The target graph must
   retire B+1 verifier rows, so `_maybe_launch_device_proposal()` first requires
   the verifier's `device_proposal_ready()` check with `rows=budget+1`. Tests
   bind the orchestration invariant that the draft launcher is never called
   when that target check fails. Duplicating the target cap in the draft
   provider would conflate two graph capabilities and would drift when RF2 adds
   distinct context buckets.
3. **C2/request-level admission — deferred to D1, not required to close RF0.**
   RF0 is containment: `auto` is fail-closed and an operator-selected `enabled`
   route may use MTP only while each live cycle is eligible, then safely
   pre-launch-fall back. That is sufficient to prevent the wedge. It is not
   certification of `enabled` for arbitrary requests. Immutable request/context
   planning, certified scopes, and admission-time rejection/reasoning remain D1
   work and must precede automatic production enablement.
4. **Reason taxonomy — stable strings now, centralized taxonomy at D1.** RF0's
   externally recorded values, especially `target_graph_context_bucket_miss`
   and `target_graph_output_room_miss`, are compatibility-stable and must not be
   renamed casually. Module constants are sufficient for RF0; D1 should
   centralize all route/admission reasons in a typed enum or equivalent registry
   while preserving these serialized string values. Do not churn RF0 solely to
   replace literals with an enum.
5. **Collection — use direct cycle/result records, not external telemetry.**
   RF0 requires the realized in-process cycle record and request result to
   identify the pre-launch miss, which the retained implementation and crossing
   smoke prove. D1/D7 should extend those direct structured records and campaign
   artifacts for failure class, graph bucket, and circuit-breaker state. This
   campaign does not require Prometheus or another external telemetry system.
6. **Tier-B crossing smoke — already authorized, completed, and retained.** Do
   not rerun it merely to answer the old authorization question. The exact
   gfx1151 27B command and host/model evidence are in the RF0 worklog; the final
   isolated node passed in 93.09 s and proved pre-crossing graph use,
   pre-launch fallback without device chaining, 12 AR-exact output IDs, a
   stable reason in the direct result, and subsequent AR health. Future
   necessary campaign GPU gates are covered by the repository's assigned-task
   validation policy; state the reason and expected duration before starting
   them.

### In-progress handoff — RF1 candidate retained and committed

- [x] **Review and continue, do not discard, the current uncommitted candidate**
      in `hipengine/runtime/qwen35_gguf_runner.py` and
      `tests/test_gguf_native_spec_cycle.py`. At long split-K contexts with
      multiple verifier rows, it routes each attention and FFN row through the
      registered c1/strict dispatch while retaining the native split-K leaf;
      retained short-context batching is unchanged. The associated checkpoint is
      `worklog/entries/20260821T052947.067917Z-gfx1151-mtp-rf1-boundary-4k-46c738.md`.
- [x] **Focused candidate proof:** the isolated repaired `end1024-b3-a3` case
      passed with exact logits/top-1, state, KV, hidden, commit, cursor, and
      rollback; split-K ownership was observed. This is encouraging but is not
      enough to retain the candidate.
- [x] **Finish candidate validation:** the combined transition packet for cycle
      ends `1024,1025,1032`, budgets B1/B2/B3, B3 full accept at 1024, and the
      real generation crossing passed on gfx1151 (13/13 direct + 1/1 generation,
      400.91 s, `--fail-on-fail`). All surfaces exact, split-K ownership present
      (16/32/48/64 calls), zero target-graph submissions; the real crossing
      matched all 8 AR IDs with GPU/CPU accept parity. The earlier passing page
      packet (cycle ends 255/256/257) is below the 1024 split threshold and is
      unaffected by the candidate — preserved. The committed RF0 e2e crossing
      test also passes with the candidate (92.81 s).
- [x] **Decide the candidate:** retained and committed as a **strict
      eager/oracle fallback** for long split-K multi-row verifier batches (row
      cost 0.4–1.6 s/direct cycle, 44.7 s for 8 generated tokens). It is not
      promoted as the fast long-context route and does not raise the 1023 graph
      context cap. Full RF1 (2K–64K and task/category evidence) remains open.

### RF1 extended-matrix status and corrected execution order

- [x] **Retain the completed 2K–16K evidence; do not rerun it.** The direct
      2K/4K/8K/16K packet passed 16/16 cases in 61.5 min
      (`/tmp/mtp-rf1-2k-16k-direct.json`): B1/B2/B3 plus B3 reject/partial/full
      at 4K, with exact logits/top-1, state, touched KV, hidden, cursor, commit,
      rollback, split-K ownership, and sufficient workspace. Real B3 generation
      at 2K/4K/16K passed 3/3 in 19.5 min
      (`/tmp/mtp-rf1-2k-16k-gen.json`), matching all eight AR IDs with GPU/CPU
      acceptance parity and no graph/device chaining. These are retained local
      checkpoint artifacts, not yet committed campaign artifacts.
- [x] **Record the stopped 32K/64K attempt as non-evidence.** The command mixed
      six direct oracle cases and two real-generation cases in one process,
      requiring roughly sixteen large native/strict/AR/MTP prefills. After more
      than 80 minutes it still had a zero-byte buffered log and had not written
      `/tmp/mtp-rf1-32k-64k.json`; the user stopped it. It produced no verdict
      and must not be described as a failed correctness gate or rerun unchanged.
- [x] **Fix harness operability before another >=32K run.** The harness now
      permits direct-only or generation-only invocation, emits immediate
      case/prefill start/end JSON progress on stderr, and atomically replaces the
      output with a running checkpoint after every event/result before writing
      the final artifact. A killed late case preserves completed rows and names
      the active stage. CPU/fake-runner coverage passed; no large GPU run was
      used to validate these controls.
- [x] **Run one isolated 32K B3 direct oracle case.** It passed in 1395.24 s
      total (`/tmp/mtp-rf1-32k-b3-direct.json`, case wall 0.994 s): exact
      logits/top-1, accept summary, state, touched KV, hidden, cursor, commit,
      and rollback; 64 split-K calls; workspace 129 versus 128 required splits;
      zero target-graph submissions. B1/B2/B3 row shapes were already covered
      through 16K.
- [x] **Certify 64K as the largest practical context on gfx1151.** The isolated
      64K B3 direct case passed in 2831.12 s, inside the 60-minute budget
      (`/tmp/mtp-rf1-64k-b3-direct.json`): every strict surface exact, 64
      split-K calls, workspace 257 versus 256 required splits, and zero target
      graph submissions. Model SHA-256 is
      `a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f`.
- [x] **Run real generation only at the selected maximum.** The isolated 64K B3
      eight-token case passed in 3392.03 s
      (`/tmp/mtp-rf1-64k-b3-generation.json`): all IDs equal true AR, two full
      B3 accepts, GPU/CPU acceptance parity, every cycle eager, 64,648 split-K
      calls, and no target graph/device chaining. No duplicate 32K generation
      was run.
- [x] **Run near-cap output tails at one moderate certified context.** Isolated
      4K cases for remaining output 1/2/3 (`max_new_tokens` 2/3/4) all passed
      AR-ID equality, GPU/CPU accept parity, eager-only ownership, split-K
      observation, and exact completion in one cycle. Artifacts:
      `/tmp/mtp-rf1-tail-r1.json`, `-r2.json`, and `-r3.json`.
- [x] **Close RF1 on gfx1151.** The durable rollup is
      `benchmarks/results/2026-08-22-gfx1151-qwen36-27b-rf1-eager-long-context.json`.
      Boundary/page/2K–64K strict surfaces, 64K true-AR generation, tails, and
      six task categories passed the RF1 binding contract. The task packet was
      6/6 MTP-vs-AR exact with clean eager ownership and 4/6 absolute answer
      correctness; the two AR task misses remain an explicit RF6 quality
      blocker, not an MTP functional regression. hipEngine-owned peak allocation
      was 26,920,424,625 bytes and returned to zero active allocations. RF1 does
      not authorize automatic routing or raise the 1023 reusable-graph cap.

### Remaining campaign work

- [x] **RF2:** context-bucketed split-K target graphs are exact and retained for
      explicit diagnostics through the certified 4K metadata scope. Steady B1–B3,
      capture/replay, acceptance, state/KV/hidden/commit/rollback, bounded
      eviction, and rocprof gates pass. Transition/kernel-family boundaries
      pre-launch-fallback. No long bucket is auto-admitted: graph/eager complete
      wall is 0.9989x and graph MTP/true AR is 0.7164x. Artifact:
      `benchmarks/results/2026-08-22-gfx1151-qwen36-27b-rf2-long-target-graphs.json`.
- [x] **RF3:** cancellation, deadline, shutdown, EOS/stop termination, six
      transaction-phase faults, graph capture/launch/readback/allocation errors,
      terminal commit ownership, and subsequent AR/MTP health are qualified.
      Forced server shutdown now waits for the actual model thread/GPU owner to
      retire before engine close. Artifact:
      `benchmarks/results/2026-08-22-gfx1151-qwen36-27b-rf3-lifecycle.json`.
- [x] **RF4:** completion/chat, single/multi-prompt, auto/explicit policies,
      eager/lazy restart, app-local transcript commit, thinking hint/hard,
      streaming/non-greedy rejection, exact usage, direct MTP extensions,
      capability scopes, and post-restart health are qualified. Artifact:
      `benchmarks/results/2026-08-22-gfx1151-qwen36-27b-rf4-api-semantics.json`.
- [x] **RF5:** retained and qualified the honest serialized dense-MTP policy.
      c1/c2/c4/c8 offered load, one c8 multi-prompt request, ragged prompts,
      mixed AR/MTP c8, deadline/readmission, queue/fairness controls, 100-request
      alternating soak, bounded resources, and shutdown pass. Route coalescing
      is explicitly not physical MTP concurrency. Artifact:
      `benchmarks/results/2026-08-22-gfx1151-qwen36-27b-rf5-serialized-load.json`.
- [x] **RF6:** complete canonical quality/performance qualification is retained
      as a rejection. Short full-suite server MTP is 1.925x true AR median with
      70.03% acceptance and deterministic repeats, but strict IDs differ on two
      heldout prompts, RF1 long-task score is 4/6, long graph MTP is 0.7164x AR,
      and streaming SLO/full-logit production gates are absent. No automatic
      scope is promoted. Artifact:
      `benchmarks/results/2026-08-22-gfx1151-qwen36-27b-rf6-qualification-rejected.json`.
- [x] **RF7:** zero-scope rollout closure passes scoped circuit-breaker,
      operator rollback without restart, in-flight retirement, shutdown drain,
      restart reset, auto-AR confirmation, and explicit diagnostic MTP health.
      No canary was started because RF6 promoted no scope. Artifact:
      `benchmarks/results/2026-08-22-gfx1151-qwen36-27b-rf7-zero-scope-rollout.json`.
- [x] Keep automatic production MTP disabled: no named scope passed RF6. The
      public `auto` policy remains available but selects AR with direct reason
      `automatic_mtp_scope_not_promoted`.

**Shared-worktree warning:** the many untracked benchmark artifacts and the
existing `benchmarks/README.md` modification belong to concurrent work and are
not part of this campaign handoff. Do not clean, stage, or rewrite them while
continuing RF1.

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
  per-request acceptance reporting, and an AR opt-out.
- Reject, partial-accept, full-accept, rollback, selected-state, and graph reuse
  have substantial focused coverage.

These are necessary foundations. They do not establish the real-world envelope.

### 3.2 Blocking findings

| ID | Severity | Finding | Current evidence / code |
| --- | --- | --- | --- |
| C0 | Resolved in RF0 | Historically, a cached below-1024 N2 graph remained “device proposal ready” after the live target cursor crossed its context limit, allowing a draft launch that the target graph could not retire. | Commit `9aa25016e` added exact live-cycle admission and recheck before launch/submission; see the punchlist and RF0 worklog. |
| C1 | Critical | The reusable target graph is structurally captured for a static `context_limit=min(1023, max_positions)`. Long contexts switch attention ownership at 1024 to split-K/workspace-backed routes. Removing only the admission guard would bind the wrong graph/workspace policy or capture an impractically large static context. | `gguf_native_spec_cycle.py` admission, graph `context_limit`, dynamic spans, and long-context split threshold in `qwen35_gguf_runner.py`. |
| C2 | High | Dense default admission is model-level (`has MTP tensors` and `not MoE`), not request/context/profile/backend/resource-level. It can advertise default-safe MTP for a request outside the graph's qualified envelope. | `Qwen35GGUFTextGenerator.supports_default_mtp`; server `enabled` policy. |
| C3 | High | Deadline/cancellation is checked around the dense generation call, but the inner one-request MTP cycle loop has no per-cycle cancellation/deadline poll. A timed-out HTTP request may stop waiting while GPU work continues. | `_generate_dense_speculative_mtp_detailed()` and `Qwen35GGUFMTPDecodeSession.generate()`. |
| C4 | High | Dense multi-prompt MTP currently iterates request rows serially through one target session. Server batching tests prove route grouping, not true concurrent dense-MTP execution, fairness, isolation under load, or aggregate benefit. | `_generate_dense_speculative_mtp_detailed()` loops over encoded prompts with `resident_slot_count=1`. |
| C5 | High | No binding long-context MTP matrix covers the 1024 transition, page boundaries, 4K–64K contexts, prompt-plus-decode crossings, graph/eager parity, state/KV ownership, teardown, and task quality together. | Historical short/category suites and separate KV long-context harnesses do not form this packet. |
| C6 | Medium | The default `thinking=hint` policy removes host-sampler thinking-budget enforcement. It may be a valid product mode, but it is not semantically identical to the original hard-controlled request and needs explicit API labeling plus task-quality evidence. | Server routing relaxes `thinking_budget`; prompt hints remain but soft-close/EOS suppression/hard-close controls do not. |
| C7 | Medium | Streaming, logprobs, tools/grammars, penalties, non-greedy sampling, token stop sequences, and several structured controls fall back or reject. This is acceptable only if capability and per-request route result fields are exact and tested. | `SPECULATIVE_MTP_INCOMPATIBLE_FIELDS`, API routing, `streaming_compatible=false`. |
| C8 | Medium | A GPU hang/fatal HIP error has no MTP-specific circuit breaker or worker quarantine policy. One bad graph can make the entire server unhealthy. | Current route has rollback for owned Python exceptions, but no post-fatal process recovery contract. |
| C9 | Medium | Existing direct result records report use/counts, but not a stable reason taxonomy for pre-launch fallback, graph miss, context-bucket transition, cancellation phase, post-launch failure, quarantine, or fallback wall cost. | Current response summary and in-process MTP cycle records. |

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
   output queues, and direct result counters belong to the correct request at
   every transition.
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
Responses and direct in-process records report the realized plan, not merely
server configuration.

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
- the direct route result names the miss.

**Implemented 2026-08-21:** cached N2 admission now checks the live cycle end,
output room, graph configuration, allocation binding, and row shape before a
device proposal can launch, then rechecks before target submission. Context and
tail misses report `target_graph_context_bucket_miss` /
`target_graph_output_room_miss`; the short graph remains cached while the host
proposal uses eager verification. The server default is now fail-closed `auto`,
so implicit traffic remains AR while explicit controlled MTP stays available.
The real gfx1151 dense-27B gate captured a short graph, crossed 1024, matched all
12 AR output IDs, reported the context miss, avoided device chaining on fallback
cycles, and completed a subsequent AR health request. This closes RF0
containment only; it does not qualify eager or graphed long-context MTP for
automatic use.

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

Required context coverage is pairwise and staged, not a full Cartesian product:

- prompt/decode cycle ends around `1016..1032` for B1/B2/B3;
- page boundaries around the configured block size;
- B1/B2/B3 direct coverage through 2K, 4K, 8K, and 16K;
- one maximum-width B3 direct case at 32K, then at 64K only when the measured
  host wall/memory budget permits;
- one real-generation crossing at the largest practically certified context;
- near-cap output-room tails at one moderate certified context.

For RF1, “largest practical” is a declared host-specific qualification bound,
not synonymous with the model's theoretical maximum. A context that cannot
finish one isolated mechanical case inside the predeclared resource budget may
remain uncertified and route to AR.

If eager MTP fails correctness, stop and localize with layer/state/KV ladders.
Do not proceed to graph capture. If it passes but is slower than AR, retain it as
an oracle/fallback only and route production traffic to AR in those buckets.

**RF1 harness implemented 2026-08-21:**
`scripts/gguf_mtp_long_context_gate.py` compares host-materialized target batches
on the eager-native verifier against an independently journaled serial-exact
teacher. It fails closed on graph submission, logits/top-1/accept-summary drift,
selected Conv/GDN
state, touched BF16 K/V rows, hidden/cursor, post-commit rollback, split-workspace
capacity, or missing split-K ownership at long cycle ends. Controlled B3 cases
cover reject, every partial depth, and full accept; optional real NextN runs
force host proposals with cycle-logit diagnostics and compare all generated IDs
to true AR. This establishes the reusable RF1 oracle machinery only. RF1 remains
open until the staged long-context/task packet above is recorded through the
largest practical context; the short reusable graph guard is unchanged.

**Long-run execution protocol:** before any >=32K gate, the harness must support
one independently selectable scenario per process and write a distinct artifact.
It must emit flushed start/end progress and atomically checkpoint each completed
case. Run stages in increasing cost, stop on the first binding failure, and use
background exit notification rather than repeated `sleep` polling. A monitor
may report only directly emitted stage progress; GPU utilization alone does not
prove which case is running. State an expected duration and stop budget before
launch. Do not start a monolithic 32K+64K direct+generation command.

**RF1 1024 boundary transition repaired and retained 2026-08-21:** the eager
long-context oracle localized the sole strict mismatch to a layer-46 row-3 BF16
rounding boundary in the staged multi-row linear-attention/FFN path at a B3 full
accept ending exactly at 1024. The retained candidate routes each attention and
FFN row through the registered c1/strict dispatch (with the native split-K
attention leaf) whenever a dense multi-row verifier batch is in long split-K
territory, so selected state/KV stays byte-exact; retained short-context batching
is unchanged. The complete transition packet (cycle ends 1024/1025/1032 x B1/B2/B3,
B3 reject/partial/full at 1024, and a real generation crossing 1020→1028) passes
on gfx1151: 13/13 direct + 1/1 generation, all surfaces exact, zero target-graph
submissions, split-K ownership present, all 8 real-generation IDs matching AR. It
is a strict eager/oracle fallback with measured cost (0.4–1.6 s/direct cycle,
44.7 s per 8 generated tokens), not a fast-graph claim; RF2 owns the speed path
and the 1023 graph cap is unchanged.

**RF1 closed on gfx1151 2026-08-22:** the staged packet now covers page and
1024-route transitions, direct B1/B2/B3 rows through 16K, isolated B3 at 32K and
64K, true-AR generation through 64K, remaining-output tails 1/2/3, six
long-context task categories, model identity, wall, and allocation high-water.
All mechanical and task-route rows pass exact AR/state/KV/control ownership.
Absolute task correctness was 4/6 because true AR and MTP identically selected
wrong answers on aggregation and code; this is retained as an RF6 quality
blocker. The durable artifact is
`benchmarks/results/2026-08-22-gfx1151-qwen36-27b-rf1-eager-long-context.json`.
Functional eager MTP is qualified only as an explicit strict oracle/fallback;
automatic use remains disabled and RF2 owns long-context acceleration.

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

**RF2 closed on gfx1151 2026-08-22 with zero auto-admitted long buckets.** The
bounded target cache is keyed by budget, N1/N2 ownership, and exact attention
schedule/context limit. It retains short 1023 and steady split-workspace graph
owners, closes evicted graph/stream/workspace/KV pins, and rejects cycles that
cross short/split, split-count, or kernel-family boundaries before capture.
B1/B2/B3, B3 reject/every partial/full, capture/replay, 2K, and 4095 pass exact
post-commit state/KV/hidden/control gates. Cached-child rocprof records one
`hipGraphLaunch` and the expected split producer/reducer names with plausible
durations and zero scratch. However, same-protocol 4079-token/eight-output wall
is graph 97.206 s versus eager 97.101 s (**0.9989x**) and true AR 69.635 s
(**0.7164x graph MTP/AR**). The implementation remains an explicit diagnostic;
automatic routing continues AR. Evidence:
`benchmarks/results/2026-08-22-gfx1151-qwen36-27b-rf2-long-target-graphs.json`.

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

**RF3 closed on gfx1151 2026-08-22.** Stable lifecycle boundaries cover before
proposal, after proposal/before target, after target prepare, after target
commit, after draft repair, and before output publication. Real-model fault
injection passes 11/11 rows including in-flight cancellation/deadline (observed
only after commit+repair), EOS at prefill, EOS and explicit stop inside a cycle,
and exact subsequent AR/MTP health. A real RED found that immutable scheduler
commit returned a new transaction while exception cleanup inspected the stale
pre-commit object; RF3 now tracks terminal commit explicitly and never reopens
committed KV. CPU graph faults fail closed for capture allocation, launch, and
readback. Server forced shutdown cancels request tokens and waits for the actual
threadpool model/GPU owner to retire before model close rather than cancelling
only the asyncio waiter. Peak tracked allocation was 18,720,029,538 bytes and
returned to zero. Evidence:
`benchmarks/results/2026-08-22-gfx1151-qwen36-27b-rf3-lifecycle.json`.

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

**RF4 closed on gfx1151 2026-08-22.** Real eager and lazy/restart server arms
both pass seven endpoint/policy groups: completion AR/MTP ID equality,
multi-prompt MTP, chat AR/MTP ID equality, thinking hint labeling and hard
rejection, explicit app-local `append_all` transcript ownership, non-greedy and
streaming rejection, and capability/readiness state. Auto fallback succeeds as
AR with stable reason `automatic_mtp_scope_not_promoted`; explicit MTP reports
vLLM-compatible accepted/rejected usage plus direct route/count/thinking fields.
Capabilities distinguish configured policy, engine/model MTP availability,
empty certified default scopes, and no automatic promotion. A real RED found
speculative length completion could beat EOS trim and publish one extra chat
token; speculative publication now retires at the first EOS/stop token before
length completion, and chat IDs are exact. Evidence:
`benchmarks/results/2026-08-22-gfx1151-qwen36-27b-rf4-api-semantics.json`.

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

**RF5 closed on gfx1151 2026-08-22 with serialized explicit-only dense MTP.**
Capabilities state `physical_concurrency=serialized_target_slot`, maximum one
physical target slot, and `route_coalescing_is_physical_concurrency=false`.
Real c1/c2/c4/c8 independent offered load and one c8 multi-prompt request match
sequential AR IDs; mixed AR/MTP c8, deadline/readmission, and six queue/fairness
CPU gates pass. A 100-request alternating soak completes 50 AR + 50 MTP with
zero failures. Tracked peak is 21,914,879,597 bytes; total allocation/free bytes
match at 103,913,324,658 and final ownership is zero. The 1-hour/final promotion
soaks are not applicable because RF2/RF6 admit no automatic MTP scope; production
continues AR. Evidence:
`benchmarks/results/2026-08-22-gfx1151-qwen36-27b-rf5-serialized-load.json`.

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

**RF6 closed as rejected on gfx1151 2026-08-22.** The canonical committed
10-prompt suite uses six train/four heldout rows across code, English, Japanese,
and mixed Japanese/English. A production true-AR direct baseline is 12.107
tok/s. Matched persistent-server AR is 11.54 tok/s; explicit B3 MTP repeats are
21.77/22.21/22.24 tok/s, median **1.925x AR**, with 70.03% acceptance and exact
MTP repeat IDs/acceptance. Train and heldout speedups are 1.969x and 1.781x and
every prompt exceeds 1.61x. Binding quality nevertheless fails: MTP differs
from AR on heldouts `general_ja_explain` (token 65) and
`mixed_ja_en_review` (token 72), RF1 absolute long-task quality is 4/6, long
RF2 graph MTP is 0.7164x true AR, and no streaming TTFT/ITL or binding
production full-logit KL/task packet exists after strict heldout failure. No
automatic scope is promoted; explicit diagnostic MTP remains available and
`auto` continues AR. Evidence:
`benchmarks/results/2026-08-22-gfx1151-qwen36-27b-rf6-qualification-rejected.json`.

### RF7 — Staged rollout and removal of emergency controls

Rollout order:

1. local direct harness, MTP explicit only;
2. local server explicit only;
3. one canary worker with `auto`, short certified buckets only;
4. canary expands one context/backend scope at a time;
5. production percentage rollout with directly collected circuit-breaker and
   AR fallback results;
6. default `auto` only after the final soak and rollback drill;
7. consider removing temporary flags only after one release window and a new
   immutable worklog decision.

Rollback drill must prove an operator can switch to AR for new requests without
restarting, allow or cancel in-flight owned cycles safely, drain MTP resources,
and preserve readiness. Fatal GPU errors still require worker restart.

**RF7 closed with zero rollout scope on gfx1151 2026-08-22.** A restart-scoped
circuit breaker counts backend/runtime failures by model/backend/profile/context
class, ignores user cancellation/deadline, and rejects further explicit MTP
before launch after threshold. An authenticated runtime rollback routes all new
MTP requests to AR without restart; a real in-flight MTP request retires under
its existing owner, subsequent traffic is AR, and shutdown returns allocation
ownership to zero. Restart resets the breaker; capabilities still advertise no
certified default scopes, `auto` selects AR with the RF6 reason, and explicit
MTP remains exact. No canary or production percentage rollout was started
because RF6 promoted no scope. Evidence:
`benchmarks/results/2026-08-22-gfx1151-qwen36-27b-rf7-zero-scope-rollout.json`.

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
- direct result-record ownership/counting;
- no-ROCm skip guards for tests that load HIP.

### 7.2 Tier B — focused real-GPU gate

- B1/B2/B3 reject/partial/full;
- contexts around 1024 and one 4K case;
- eager/graph parity where graph is admitted;
- state/KV/cursor oracle;
- cancellation and subsequent-request health;
- expected kernel trace and direct graph/fallback records.

### 7.3 Tier C — nightly/campaign gate

- full category+heldout suite;
- contexts 512, boundary, 4K, 16K, and one largest-practical long fixture;
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
than reimplementing model logic. It should emit one compact rollup artifact
assembled from independently checkpointed scenario artifacts. The orchestrator
must write each completed scenario atomically before starting the next, and must
be able to resume without
rerunning passing scenarios. It should contain:

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

## 9. Required direct collection and operator view

This campaign has no Prometheus or external-telemetry dependency. Collect every
required field directly from the owning runtime/server objects into structured
request/cycle results and the campaign artifact. Tests must assert those objects
rather than scrape a separate service. Aggregate summaries are computed by the
campaign harness from the direct records.

Per request/cycle record:

- requested/effective route and stable decision reason;
- model/backend/profile/context bucket/candidate budget/verify mode;
- used MTP, cycles, proposed/accepted/rejected drafts, acceptance;
- graph captures/replays/misses and eager/AR fallback;
- pre-launch versus post-launch failure class;
- cancellation/deadline/termination phase and latency;
- transaction terminal status;
- circuit-breaker/quarantine state when relevant;
- target/draft KV, graph-workspace, and cache high-water snapshots;
- complete request and cycle timing samples.

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
| D1 request-plan/reason taxonomy and safe auto fallback | D0 | server route/API tests and direct result records |
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
- [ ] Capabilities, responses, direct runtime records, and artifacts report the
      realized route and reason accurately.
- [ ] Final soak and rollback/restart drills pass with no hang, leak, stale
      graph, ownership mismatch, or unexplained readiness loss.
- [ ] Benchmark artifact, rollup, changelog, immutable worklog, and any
      `REFACTOR.md` flag entries are current.

Anything less may be a useful MTP diagnostic or explicit opt-in, but it is not a
production-ready default.

## 12. RF0 implementation slice — completed

Commit `9aa25016e` completed the original safety-only first slice: RED/GREEN
coverage, graph `can_launch()`/reason admission, context/output-aware
`device_proposal_ready()`, no draft launch on a target miss, safe pre-launch
eager fallback, a stable context reason in the direct result, and the real
crossing/health smoke. Do not reimplement this slice; the exact evidence and
resolved decisions are in the punchlist above and the RF0 worklog.

The retained constraint still applies: do **not** remove `end >= 1024`, change
`context_limit=min(1023, ...)`, or claim fast long-context MTP as part of RF0.
Functional eager qualification is RF1, and context-bucketed fast graphs are RF2.
