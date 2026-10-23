---
status: closed
owns: Bounded main-path cleanup campaign, Qwen GGUF ownership mapping, refactoring safety checks, and implementation order.
---
# Main-path architecture cleanup

## Goal and decision

Reduce duplicated policy and ambiguous state ownership in the public generation
path while preserving working functionality, defaults and execution contracts.
Start with one Qwen GGUF request path, not every model family at once.

The agreed order is **bounded test/dead-code stabilization, scoped ledger
triage, ownership mapping, then one implemented ownership improvement**. Do not
wait for all of `REFACTOR.md` to be cleared. The ledger mixes historical decisions,
small fixes, unrelated work and issues that only architecture changes can solve.

Line count is a diagnostic, not an acceptance target. Moving a large class into
several files, adding a protocol around the same implicit behavior, or making the
audit count green does not by itself simplify the architecture.

This campaign closed on October 23. The
[ownership and safety evidence](MAIN-PATH-OWNERSHIP-EVIDENCE.md) records the
baseline, implemented ownership boundaries, public LLM/HTTP checks, milestone
result and focused repair. Construction cleanup, oversized-request containment
and resident MTP upload lifetime are implemented. The plan below records the
bounded scope and acceptance criteria used; it is not a mandate to continue
repository-wide cleanup.

## Starting point

The October 23 baseline and completed cleanup are recorded in
[`audit/COVERAGE.md`](../../audit/COVERAGE.md). That document owns the historical
size measurements, audit coverage assessment and parity-tool archive details;
this campaign owns the next execution sequence.

Completed prerequisites, not work to repeat:

- Audit contract tests participate in ordinary pytest discovery. Product and
  audit test packages are separated, including audit-first collection coverage.
- The audit reports conservative overwritten-definition candidates. The dead
  DMS method was removed while preserving the live preparation guard, with
  behavior tests that reject the discarded implementation.
- Historical per-layer llama.cpp parity tools and their dedicated tests were
  removed. Shared dependencies, fixtures and evidence remain; the complete tree
  is on `archive/llamacpp-parity-tooling-20261023`.
- After syncing Gemma vision and repairing test collection, the default unit run
  at the implementation committed as `1846adec9` passed 15,223 tests, with 271
  skipped and 10 subtests. This is a unit baseline, not full GPU, public-serving
  or refactoring-safety certification. See the
  [validation entry](../../worklog/entries/20261023T213657.077017Z-cleanup-audit-product-test-package-eaf39e.md).

Three other overwritten-definition candidates were left in the audit removal
queue. Re-read their current rows before taking them; they are not prerequisites
unless they affect the chosen main-path work.

## Initial scope and hypotheses

Start at `LLM.generate()` and the server text/chat surface for the Qwen GGUF
family. Inspect ordinary autoregressive generation, a speculative request, and
cancellation/error cleanup. Record the exact model/backend/quant/configuration
used for runtime evidence rather than treating one family name as a single route.

The preliminary path below is a **static investigation guide**, not a verified
execution trace:

```text
server/api.py: _GenerationBatcher
  -> llm.py: generator construction and configuration
  -> generation/engine_service.py / engine_loop.py / batch_scheduler.py
  -> generation/qwen35_gguf.py: generator and resident adapter
  -> runtime/qwen35_gguf_runner.py: runner/session
  -> selected kernels and KV storage
```

Check whether construction, submission and execution follow this ordering; do
not force the implementation into the diagram. Follow speculative adapters and
transport-specific paths where the trace actually branches.

The earlier static review supplied these leads. Re-measure at the campaign base
before using counts as evidence; the supplied class/function sizes, direct-import
counts and environment-read counts were not reproduced for this document.

| Lead | What to establish before changing it |
| --- | --- |
| Very large model-family classes and methods | Which responsibilities and decisions are duplicated, not just where a file can be split |
| Sparse `layers/` implementation versus the design in `PLAN.md` | Whether a layer abstraction would remove real duplication; do not populate a directory to match a diagram |
| Many `prefill`, sampling and rollback implementations | Whether semantics, numerical contracts and state ownership actually match across families |
| Mixed direct kernel imports and registry resolution | Separate registration/metadata/host-helper imports from hard-wired implementation selection |
| HTTP queueing plus scheduler admission | Whether these are duplicate policies or distinct transport backpressure and device-scheduling responsibilities |
| Generator setup through `getattr` and ad hoc attributes | Mandatory lifecycle operations, optional capabilities, construction order and failure behavior |
| Many flags and a large cleanup ledger | Which selectors and workarounds belong to this request path and have concrete removal conditions |

Do not repeat the proposed comparison with vLLM host size without a matched
counting scope. Python file counts can include embedded kernel source, and test,
tool and generated-code inclusion can differ between projects.

## Execution phases

### Phase 1 — Establish the safety checks

Make a finite inventory of contracts for the selected path. Use existing tests
first; add coverage where the intended refactor would otherwise be unprotected.
A green default unit run does not cover every public or stateful contract.

| Contract | Evidence to name in the matrix |
| --- | --- |
| Public construction and capability selection | Real constructor/public entry point; selected generator, runner, variant and fallback/error reason |
| Admission and capacity | Who waits, rejects, reserves and releases; supported values on both sides of any threshold and an unrelated value |
| Request isolation | Neighbor substitution, slot reuse, reordered completion and no cross-request state |
| Cancellation and errors | Before admission, while queued and during execution; resources reclaimed once and later requests still work |
| Speculative state | Commit/rollback ownership, rejected-token cleanup and no accidental canonical-state mutation |
| Streaming and completion | Stop/EOS behavior, final usage, error propagation and transport disconnect cleanup |
| Shutdown and lifetime | Close after success/failure/cancellation; graph/buffer/session ownership and repeat-close behavior where supported |

For every row, record the exact test node or command, oracle, fixture/dependency,
execution tier, current result, and what it does **not** cover. Distinguish
construction tests from bare-instance method tests and mocked host-dispatch tests
from real-model execution.

For a small set of critical guards, introduce an isolated representative fault
and confirm the intended assertion fails. Do not build a repository-wide mutation
campaign. Retain source-based checks when they enforce real architectural or
anti-gaming constraints; replace spelling checks only with a stronger contract.

**Exit:** a runnable fast CPU bundle and named public/GPU/model-dependent checks
for the proposed change, with concrete missing coverage visible. Resolve coverage
needed for that change before implementing it, not before permitting evaluation.

### Phase 2 — Bounded cleanup and scoped ledger triage

Inspect the selected path's relevant audit rows and `docs/REFACTOR.md` entries.
Classify each as historical/resolved, small independent cleanup, required
prerequisite, architecture work addressed by this campaign, or unrelated work.
Cite the current code and retain durable decisions in the audit store.

Remove proven obsolete paths and campaign-only tools in small units. Check
public exports, registration side effects, dynamic imports, CLI consumers,
fixtures and replay dependencies before deletion. Preserve an archive reference
where historical tools are retired; do not rewrite history or edit immutable
worklog entries. Prefer retirement over maintaining unused campaign machinery.

**Bound:** after the initial safety/ledger pass, publish one finite prerequisite
list for the first ownership change. Additional cleanup enters that list only
when a concrete dependency is demonstrated. Unrelated debt stays in the existing
audit queue; it must not indefinitely delay the architecture work.

**Exit:** required small prerequisites are committed and tested, and the remaining
ledger items have explicit dispositions. Zero repository-wide debt is not required.

### Phase 3 — Map ownership through the public path

Trace the selected ordinary and speculative requests, including cancellation and
one failure/recovery case. Pair source inspection with public-surface execution;
a test-only route or silent fallback cannot establish the intended path.

Produce one table containing, for each decision/state transition:

- Current owner and callers, with file/symbol anchors.
- State read and mutated; resource acquired, transferred or released.
- Construction-time versus request-time responsibility.
- Error, cancellation, retry and shutdown behavior.
- Evidence of duplicate policy or unclear ownership, if any.
- Proposed owner and behavioral checks protecting a change.

Cover admission/waiting, capacity calculation/reservation, generator/session
construction, KV/request state, speculative commit/rollback, release and error
propagation. Two queues may be legitimate; distinguish transport limits from
execution scheduling before removing either.

**Exit:** a source-backed and runtime-checked ownership map, test matrix, and
short prerequisite list selecting one concrete refactor. No new framework is
required to produce this result.

### Phase 4 — Implement one ownership improvement

The likely first candidate is **generator construction and capability
negotiation**, subject to the ownership map. Separate required lifecycle
operations from optional capabilities such as DMS, speculative execution,
resident-runner construction and capacity declarations.

Do not replace `getattr` with one oversized protocol or a collection of
capability objects unless that removes implicit decisions and reduces duplicated
policy. Specify construction order, supported/unsupported semantics and failure
behavior. Check other generator families using any changed shared contract; a
Qwen-focused implementation must not break them.

Admission/capacity ownership is a subsequent candidate. Preserve distinct HTTP
backpressure where needed, but avoid independent owners calculating or enforcing
the same execution capacity with different rules.

Each implementation unit must name:

1. The responsibility transferred or duplicated decision eliminated.
2. The old path removed, rather than another permanent fallback added.
3. Behavior and defaults that remain unchanged.
4. Tests that fail under a representative incorrect implementation.
5. The public request and observable selected route used for verification.
6. Applicable numerical/resource checks and the exact commands/results.

**Exit:** one merged-in-code ownership improvement with its old duplicate or
implicit path removed, tests passing, public behavior verified, audit decisions
updated and an atomic commit. A design document or file extraction alone does
not close this phase.

### Phase 5 — Review the result before expanding

Evaluate what became simpler and whether the safety checks caught meaningful
failures. Select the next bounded unit based on that evidence, not the longest
file. Possible follow-ups are transport route separation, shared sampling or
state-management code, and registry-based selection family by family.

These are candidates, not a mandate to unify all families. Keep separate
implementations where numerical or ownership semantics differ. Splitting
`create_app`, sharing rollback, or rewriting PARO/Gemma dispatch requires its own
verified boundary and affected-path checks.

## Constraints and verification

Follow [PLAN.md](../PLAN.md), [TESTING.md](../TESTING.md) and
[EXECUTION-PROFILES.md](../EXECUTION-PROFILES.md). Preserve the torch-free hot
path, registry boundaries, `KVLiveSpans`, transactional state and explicit-request
run-or-error behavior. No model/artifact allowlists, evidence-based admission
restrictions or new default-off paths are introduced by this campaign.

Keep host ownership changes separate from arithmetic/kernel tuning. If a unit
changes arithmetic, kernels or makes measured performance claims, the applicable
[OPTIMIZATION.md](../OPTIMIZATION.md) gates also apply. Large structural changes
must not silently change selected algorithms, precision or execution profiles.

Run the narrowest affected tests and the applicable CPU bundle for each unit.
Use explicit integration/GPU/live targets where required by the safety matrix.
At campaign closure run `uv run pytest --suite all -v` under the repository's
milestone rules; a unit-only pass is not closure. Apply the focused-repair rule
for isolated failures rather than automatically repeating a successful broad run.
Unavailable hardware/model dependencies remain named blockers with clearing
commands, not skipped claims of completion.

Run `python3 audit/audit.py check` for cleanup and
`python3 scripts/docs/check_all.py` for documentation changes. Never raise the
audit budget to accept findings. Record substantial units in unique immutable
worklog entries and commit validated units immediately. Do not add benchmark
claims, edit the root README, or broaden this campaign to all model families as
side effects.

## Completion checklist

Evidence for each item is in the linked
[execution record](MAIN-PATH-OWNERSHIP-EVIDENCE.md).

- [x] Select and record the exact Qwen GGUF configurations and public entry points.
- [x] Complete the behavioral safety matrix with commands, results and limitations.
- [x] Confirm representative critical tests fail under deliberate faults.
- [x] Finish the finite prerequisite cleanup list and scoped ledger decisions.
- [x] Trace ordinary/speculative requests plus cancellation and failure recovery.
- [x] Record current/proposed ownership and select the first implementation unit.
- [x] Implement and validate one ownership improvement; remove its old path.
- [x] Check affected non-Qwen consumers of any changed shared contract.
- [x] Complete applicable public/stateful/numerical checks and milestone suite.
- [x] Update audit decisions, plan references and worklog evidence; commit all units.
- [x] Record remaining follow-ups and close this campaign without claiming the
      entire architecture or test suite has been cleaned.

The full suite completed with one outdated test fixture failure; the repaired
file and affected bundle passed under the focused-repair rule. Unavailable
GPU/model tests were skipped, not certified. Separate real-model LLM and HTTP
checks covered the selected Qwen path, including explicit speculative execution
and shutdown. No arithmetic changed and no performance claim is made.

The next bounded candidate is candidate-budget ownership, not broad class
splitting or queue unification. See the execution record for its scope.
