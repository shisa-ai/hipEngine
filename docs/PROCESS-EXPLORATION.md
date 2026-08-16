# Optimization Exploration Methodology

Status: approved methodology for exploratory optimization work as of 2026-08-15.

This document defines an optional mode for cases where hipEngine has a measured
performance gap but bounded, single-incumbent tuning is repeatedly converging on
small local changes. It permits a broader search over algorithms, dataflows,
representations, fusion boundaries, scheduling, compiler behavior, and runtime
ownership. It does **not** relax correctness, benchmark integrity, architecture,
or default-promotion rules.

The governing principle is:

> **Loosen the search topology, not the evidence standard.**

Exploration may spend more iterations, keep several hypothesis families alive,
and allow an incomplete structural candidate time to mature. It may not improve
a score by recognizing the benchmark, changing the denominator, skipping work,
or narrowing behavior around known prompts, token IDs, outputs, or fixtures.

This methodology complements:

- [`BENCHMARK.md`](BENCHMARK.md), especially **Anti-gaming**, which remains the
  authority for claim eligibility and prompt-sensitive optimization;
- [`TESTING.md`](TESTING.md), which remains the authority for independent
  oracles, RED/GREEN work, and correctness gates;
- [`KERNELS.md`](KERNELS.md), [`ROOFLINE.md`](ROOFLINE.md), and the applicable
  backend tuning guide for lineage, resource, and profiling requirements;
- [`PROCESS-IMPROVEMENT.md`](PROCESS-IMPROVEMENT.md), which separately discusses
  project planning and document organization;
- [`../AGENTS.md`](../AGENTS.md), whose architecture, evidence, worklog, Git,
  coordination, and commit rules always apply.

If this document conflicts with any of those authorities, the stricter rule
wins.

---

## 1. When to open an exploration

Open an exploration when at least one of these is true:

- a current, reconciled profile shows a material gap, but several bounded
  variants of the obvious implementation have failed;
- the incumbent is near a local optimum within its current dataflow, layout, or
  fusion boundary, while a comparator proves a larger complete-request prize;
- repeated micro-optimizations move resource counters or leaf time without
  moving the complete owner;
- the remaining gap likely requires a representation, ownership, scheduling, or
  algorithm change rather than another tile/thread/unroll sweep;
- a structural idea has a credible multi-step path but its first incomplete form
  cannot reasonably beat the fully tuned incumbent;
- profiling cannot explain a material comparator gap and several independent
  mechanisms need to be tested to identify it.

Do **not** use exploration to avoid a known prerequisite. Stay in the normal
bounded workflow when:

- the effective route is not certified;
- the profile does not reconcile or the target owner is not measured;
- the candidate lacks an independent correctness oracle;
- a source, hardware, model, or runtime blocker prevents a valid comparison;
- a known in-tree or upstream implementation can be ported mechanically;
- the likely request-level prize is already below the campaign's normal
  admission threshold;
- the proposed direction repeats a rejected experiment without a materially new
  premise.

An exploration is not permission for blind tuning. Its opening card must still
name the baseline, measured gap, likely owner, generalization contract, immutable
evaluator, and resource budget. The search can be broad even when the first
hypothesis is uncertain; the measurement contract cannot be.

---

## 2. What becomes broader, and what remains fixed

### Broader during exploration

- several distinct idea families can remain live at once;
- a structural candidate can remain `maturing` for a predeclared number of
  intermediate steps without beating the incumbent on every step;
- profile signatures, traffic reductions, launch removal, occupancy changes,
  or ownership simplification can justify continued maturation before an E2E
  win exists;
- a stronger advisor model, external papers, backend implementations, compiler
  output, and read-only reference repositories can seed new hypotheses;
- combinations of independent near-misses can be tested when their measured
  savings target different costs;
- cleanup/context-reset iterations are valid when accumulated candidate code is
  obscuring the mechanism or biasing subsequent work.

### Fixed throughout exploration

- architecture and plugin-registry invariants;
- independent numerical or transaction oracles;
- benchmark inputs, timing boundaries, objective extraction, and denominators;
- anti-gaming and train/heldout rules;
- exact model, quant, shape, hardware, software, and route identity;
- one hardware owner and normal thermal/idle discipline;
- unfused fallbacks, lifecycle, memory, and public-default gates;
- explicit staging, validation, worklog, and atomic commit requirements;
- cleanup of rejected code and invalid evidence.

A discovery result is diagnostic until it passes the normal qualification and
promotion gates. Exploration creates candidates; it does not create a second,
weaker class of retained performance claims.

---

## 3. Declare the generalization contract first

"General" does not mean that every optimization must support every model,
backend, quant, and shape. hipEngine deliberately supports backend-, model-,
quant-, role-, and shape-specialized kernels through registered variants. The
required generalization surface is the **declared product envelope** for the
candidate.

Before generating ideas, state:

- backend and architecture;
- model family and geometry, or the narrower explicitly supported model;
- quant and physical layout;
- operation/layer roles;
- dtype and numerical contract;
- row, batch, context, and alignment ranges;
- graph/eager, prefill/decode/verifier, and concurrency modes in scope;
- required fallback behavior outside the optimized region;
- whether the result is intended to transfer to another model or backend.

A specialization is legitimate when its predicate is a documented property of
that declared envelope and the mechanism remains correct for unseen inputs that
share the property. It is not legitimate when the predicate merely fingerprints
a benchmark case.

### Normally legitimate specialization inputs

- registered backend/architecture capability;
- model geometry, tensor dimensions, operation role, and layer type;
- quant format, physical layout, dtype, alignment, and vector width;
- row/batch/context ranges with tested boundaries and an explicit fallback;
- compile-time constants or static scheduling derived from those properties;
- model-wide prepacking or a frozen layout chosen independently of runtime
  prompt/token contents;
- resource limits and measured architecture-local code generation;
- a documented format invariant such as structural padding or guaranteed block
  sparsity, provided correctness does not depend on unvalidated data values.

### Conditional and high-risk specialization inputs

These require an explicit rationale, an independent evaluation set, and a
narrowly labelled claim:

- model-specific layer maps selected from measured layer behavior;
- numerical range detectors or data-distribution fast paths;
- sparsity detected from fixed model weights rather than guaranteed by format;
- context- or occupancy-adaptive routing learned from workload traces;
- approximate/mixed-precision policies selected from quality results.

A conditional specialization is acceptable only when the policy is frozen from
a declared development set, remains mathematically valid for all routed inputs,
includes detector/routing overhead in timing, preserves a safe fallback, and
passes untouched qualification and confirmation sets. If those conditions are
not practical, do not make it a production candidate.

### Forbidden specialization inputs

Candidate code and routing must never depend on:

- prompt strings, token IDs, token sequences, expected continuations, or fixture
  hashes;
- candidate-token IDs, expected accept/reject paths, known logits, or expected
  top-1 outputs;
- benchmark name, category, run index, repetition order, warmup state, or output
  path;
- known fixture tensor values or oracle outputs embedded into implementation;
- timing mode, profiler presence, or whether the current call is measured;
- a data detector reverse-engineered from the benchmark cases with no independent
  product justification;
- skipped computation, stale state, incomplete synchronization, hidden fallback,
  or a changed denominator that makes the measured path do less required work.

This prohibition applies even if outputs happen to remain correct on the known
suite. A shortcut that recognizes the evaluator is gaming, not specialization.

---

## 4. Establish an evaluation firewall

An optimization loop is most dangerous when it can modify both the candidate
and the scorer. Freeze the evaluation surface before the first candidate.

Record and, where practical, hash:

- benchmark and profiler scripts;
- objective extractor and comparison logic;
- prompt/shape fixtures and train/heldout split;
- CPU/reference oracle and tolerances;
- baseline and comparator commands;
- timing and synchronization boundaries;
- required environment and route manifest;
- candidate-edit allowlist and evaluator denylist.

The loop may read these files but must not edit them while optimizing. Candidate
work normally edits only the named kernel, wrapper/registration path, focused
candidate test, and explicitly allowed harness selector. A pre-existing guard
cannot be weakened, skipped, or replaced by a candidate-authored test.

If the evaluator is wrong or missing a required field:

1. stop the exploration;
2. fix and validate the evaluator as a separate logical unit;
3. commit that repair;
4. refresh the baseline;
5. reopen the exploration against the new evaluator identity.

Do not repair the scorer and the candidate in one keep/revert iteration.

For prompt-sensitive or quality-sensitive work, the expected outputs and
heldout details belong to the evaluator, not the candidate mechanism. The model
may be evaluated against them; production code must not receive or encode them.

---

## 5. Use discovery, qualification, and confirmation sets

Long adaptive searches overfit measurements even without explicit cheating. If
hundreds of variants see the same shapes, prompts, and thermal conditions, the
"winner" can be noise or an accidental property of those cases. Use three
surfaces where the workload permits it.

### Discovery set

Used frequently for fast iteration:

- representative leaf/operation shapes;
- a declared training prompt subset for prompt-sensitive diagnostics;
- one or more actual model weights/roles;
- cheap correctness fixtures and resource checks;
- enough repetitions to reject obvious noise and regressions.

Discovery results choose what to investigate. They do not support publication.

### Qualification set

Used when a candidate has a coherent mechanism:

- all production operation roles touched by the candidate;
- nearby, boundary, odd/tail, and fallback shapes;
- actual-weight cycling large enough to defeat misleading cache residency;
- complete state/KV/transaction checks where relevant;
- full train plus category-heldout suites where required by
  [`BENCHMARK.md`](BENCHMARK.md);
- one or more complete-model controls and a fresh profile.

A candidate can displace an incumbent only after qualification.

### Confirmation set

Reserved for a frozen finalist:

- fresh process and clean source state;
- counterbalanced baseline/candidate order;
- final repetition count and idle/thermal preflight;
- complete publication shapes and natural/category suite as applicable;
- shipping-default route parity and teardown;
- no candidate edits after the confirmation run begins.

If a confirmation failure causes further tuning, that confirmation surface has
become development evidence. The modified candidate needs a new clean
confirmation, and a prompt/quality policy that was tuned from a heldout failure
needs a genuinely untouched heldout set before it can claim generalization.
Never rename a repeatedly tuned test set "heldout."

Not every low-level exact kernel needs three separate files or prompt suites.
The principle is separation of selection from final evidence. For an exact leaf,
actual unseen values and boundary shapes can supply that separation; for MTP,
sampling, quality, routing, or adaptive policies, committed train/heldout and
full-suite rules are mandatory.

---

## 6. Maintain a small beam of hypothesis families

Do not run exploration as an endless single-incumbent hill climb. Maintain
three to five conceptually distinct hypothesis families. A useful opening beam
is:

1. **Exploit:** a low-risk improvement close to the incumbent.
2. **Structural:** a different dataflow, layout, fusion, ownership, or algorithm.
3. **Transfer:** a mechanism from a current external implementation or another
   in-tree backend/model, re-derived for the measured target.
4. **Compiler/runtime:** code generation, scheduling, launch, graph, memory
   lifetime, or host/device ownership.
5. **Wildcard:** an advisor- or research-generated premise outside the current
   local vocabulary.

Three good beams are better than five aliases for the same tile sweep.
Parameter values of one design are variants within a beam, not independent idea
families.

Each beam entry records:

| Field | Question |
| --- | --- |
| ID/family | What stable name identifies the mechanism? |
| Parent | Which incumbent or structural parent does it extend? |
| Target | Which measured owner, traffic, launch, or resource cost should move? |
| Mechanism | Why should this change move that target? |
| Predicted signature | What profile/resource/codegen evidence should appear before E2E wins? |
| Maximum prize | What measured/derived/estimated request saving is possible? |
| Generalization risk | How could it overfit shape, prompt, weights, or evaluator behavior? |
| Cheapest discriminator | What is the fastest independent test that can falsify it? |
| Maturation budget | How many steps or hardware hours may it consume before reassessment? |
| State | `seed`, `active`, `maturing`, `near-miss`, `parked`, `killed`, or `promoted` |
| Next decision | What exact evidence causes continue, combination, park, or kill? |

Only one implementation owner should modify a shared kernel/runtime surface at a
time. Advisors and research passes can run read-only in parallel; competing
agents must not edit the same files or benchmark on the same GPU concurrently.
The beam is search memory, not permission to create several conflicting
worktrees in the shared repository.

### Preserve concepts, not candidate debris

A near-miss is worth preserving when it:

- repeatably improves a material sub-owner;
- removes traffic, launches, memory, or complexity that another idea may also
  need;
- changes the algorithmic surface and has a credible next maturity step;
- improves one declared region without gaming, even though policy integration is
  incomplete;
- reveals that the profile or assumed bottleneck was wrong.

Preserve its mechanism, measurements, and combination opportunity in the beam
ledger/worklog. Do not retain hundreds of source copies, dead registry variants,
or default-off flags. Candidate code survives only when it remains an active
maturing parent, a reusable independently validated primitive, or a required
rollback/fallback. Track every temporary surviving path in
[`REFACTOR.md`](REFACTOR.md).

---

## 7. Give structural candidates bounded maturation

A fully tuned incumbent will usually beat the first incomplete version of a new
architecture. Structural exploration may therefore continue without an
immediate topline win when **all** of these are true:

- the independent correctness gate remains green;
- the candidate is not gaming or narrowing the declared envelope;
- the expected profile signature appears, such as fewer bytes, fewer launches,
  lower owner time, better occupancy, reduced synchronization, or simpler
  ownership;
- the remaining measured ceiling can still clear the campaign threshold;
- the next maturity step is concrete rather than "tune more";
- the candidate remains inside its predeclared time/hardware/complexity budget.

The opening exploration card should set a maturation budget. A reasonable
default is two or three meaningful structural steps before a full reassessment,
not two or three arbitrary parameter changes. Larger budgets require stronger
measured evidence that the new surface is developing as predicted.

Kill or park a structural beam when:

- its predicted signature does not appear;
- correctness requires input-conditioned repair or unsupported arithmetic drift;
- required fallback/operation completeness erases the expected saving;
- resource growth, compilation, memory, or integration cost consumes the prize;
- a fresh profile shows the targeted cost is no longer material;
- repeated steps change implementation details without changing the mechanism;
- its best plausible completion remains below the measured incumbent.

Exploration can tolerate an intermediate regression. Production defaults and
retained performance claims cannot.

---

## 8. Generate ideas deliberately

Local minima often arise because the next experiments all share one vocabulary.
Use explicit diversity rather than merely increasing iteration count.

### Idea sources

- current semantic profile and resource/code-object inspection;
- operation-complete dataflow and byte/launch ownership analysis;
- another hipEngine backend/model/quant with a similar geometry;
- read-only reference repositories and their retained/rejected evidence;
- compiler-generated assembly, occupancy, wait states, and memory transactions;
- papers, vendor guidance, and public kernel writeups;
- a stronger advisor model given a compressed evidence packet;
- adversarial review asking which premise in the current design may be wrong.

Before admitting an external idea, check [`KERNELS.md`](KERNELS.md), source
lineage, recent relevant worklogs, [`LESSONS-LEARNED.md`](LESSONS-LEARNED.md),
and [`REFACTOR.md`](REFACTOR.md). Novel wording is not a novel mechanism.

### Stagnation trigger

Run a deliberate reset after roughly three to five unsuccessful same-family
experiments, or sooner when the profile disproves the active premise. The reset
should:

1. re-profile the current incumbent if ownership may have changed;
2. summarize which costs remain material;
3. list tried mechanisms and why they failed;
4. refresh the beam with at least two ideas from different families;
5. ask whether a representation/algorithm boundary should move rather than a
   parameter;
6. remove dead candidate code and, when useful, start a fresh agent context.

### Advisor packet

Give an advisor a compact packet rather than the entire document corpus:

- exact target and declared generalization envelope;
- current baseline and reconciled owner table;
- resource/codegen facts;
- top rejected mechanisms and reopen conditions;
- architecture and evaluator invariants;
- maximum acceptable memory/complexity change;
- the question: "which materially different mechanisms could move this owner?"

Require advisor output to include mechanism, target bucket, predicted signature,
maximum prize, cheapest falsifier, and generalization risk. Do not ask only for
"more optimizations" and accept a list of thread-count variants.

Advisor suggestions are hypotheses, not evidence. They enter the same beam and
must pass the same firewall and gates.

---

## 9. Run a layered exploration loop

### E0 — Freeze the contract

Record:

- baseline commit and clean/dirty state;
- hardware/software/power identity;
- model, quant, layout, route, and shape envelope;
- complete metric plus measured owner and maximum prize;
- discovery, qualification, confirmation, and heldout surfaces;
- correctness class and independent oracle;
- evaluator identities/hashes and candidate edit allowlist;
- maturation, hardware-time, and complexity budgets;
- human-review triggers;
- stop, promotion, and reopen rules.

If these cannot be stated, measure first rather than opening an autonomous loop.

### E1 — Seed the beam

Create three to five mechanism-level entries. At least one should be structural
and at least one should come from outside the current local tuning family. Rank
by expected information value as well as likely speedup: a cheap experiment that
invalidates a major premise may be more valuable than another likely 0.2% tile
win.

### E2 — Iterate one focused hypothesis

For each iteration:

1. state the hypothesis and predicted observable before editing;
2. add or identify the independent RED/correctness check when behavior or math
   changes;
3. implement the smallest change that tests the mechanism;
4. run correctness and resource guards before performance timing;
5. measure the discovery set with all samples retained;
6. profile when the result is major, surprising, or structurally different;
7. classify the result as `promote-to-qualification`, `maturing`, `near-miss`,
   `parked`, or `killed`;
8. update the beam's evidence and next decision;
9. restore rejected source and remove unused candidate surfaces.

Do not let an agent silently reinterpret a failed objective. Changing the
metric, threshold, scope, or evaluator requires a recorded correction and, when
material, a fresh baseline.

### E3 — Combine only independent evidence

Try a combination when individual candidates target independent measured costs
or one supplies infrastructure required by the other. Do not combine several
losing variants merely to obtain another sample. Record an expected additive or
interaction signature before testing the compound.

A compound candidate inherits every component's correctness, fallback,
generalization, and cleanup obligations. Qualification compares it with the
current production parent, not with a deliberately weak intermediate.

### E4 — Qualify the finalist

Freeze the implementation and run the qualification set. Require:

- operation completeness and fallback coverage;
- representative and edge correctness;
- actual model/weight/state/KV checks;
- full required shape, category, and heldout matrix;
- complete-model A/B and resource accounting;
- a refreshed profile confirming the claimed mechanism;
- no evaluator, prompt, denominator, or timing-boundary change.

A prompt-sensitive change may remain `maturing` based on train diagnostics, but
it cannot be called a kept win, replace the incumbent, or enter promotion until
it satisfies [`BENCHMARK.md`](BENCHMARK.md)'s full/train/heldout/category rules.

### E5 — Confirm and promote

Run the untouched confirmation protocol from a clean source/process state. Then
apply the ordinary hipEngine promotion contract:

- exact or explicitly quality-qualified non-regression;
- shipping-default route parity;
- benchmark artifact, README rollup, and changelog for retained claims;
- kernel catalog/lineage updates when applicable;
- immutable worklog entry;
- refactor/rollback cleanup;
- immediate atomic commit after validation.

Exploration ends at promotion. The winning path becomes ordinary production
code and is no longer exempt from normal bounded maintenance.

---

## 10. Guard against statistical and systems overfitting

A long loop can select noise even when every candidate is honest.

- Never select the fastest single sample or discard inconvenient completed
  samples.
- Use balanced/counter-rotated A/B order and fresh processes where the campaign
  requires them.
- Prebuild JIT objects outside timing/profiling and verify the effective route.
- Keep one benchmark owner on the GPU; stop downloads, compilers, profilers, and
  unrelated servers before retained measurements.
- Track thermal/power/clock state appropriate to the host.
- Use cycling pools/actual weights when cache residency can make a leaf
  unrealistic.
- Treat timeouts and machine faults as inconclusive; treat completed correctness
  failures and robust regressions as evidence.
- Increase confirmation strength after a large adaptive search. Many trials
  increase the chance that a small apparent win is selection noise.
- Re-run the frozen finalist against the original incumbent, not only its latest
  structural parent.
- Keep profile counters secondary to required complete wall. A counter is a
  mechanistic check, not a substitute objective to game.

A result that appears only under the profiler, only after repeated hot-cache
calls, only in one candidate order, or only on the development shape remains a
diagnostic.

---

## 11. Autonomy and human checkpoints

A loop may autonomously edit and measure within the frozen candidate allowlist
when the verifier is mechanical, the source scope is isolated, and keep/revert
is safe. This is most suitable for exact kernel schedules, codegen variants,
operation-local fusion, and bounded ownership changes.

Pause for human review before:

- changing benchmark scripts, fixtures, objectives, heldout splits, timing
  boundaries, or correctness thresholds;
- accepting approximate math, quality tradeoffs, or changed sampling semantics;
- changing architecture, plugin boundaries, `KVLiveSpans`, public APIs, or
  persistent state ownership;
- adding prompt-, token-, candidate-, model-value-, or distribution-conditioned
  routing;
- changing power, clocks, firmware, kernel parameters, memory limits, or other
  system configuration;
- making a candidate the package/server default;
- materially expanding beyond the declared model/backend/quant/shape envelope;
- preserving a default-off experiment that adds significant memory or
  maintenance cost.

The agent must never revert, reset, clean, stage, or commit unrelated shared-tree
work. Automatic reversion is limited to the candidate's own known paths.

---

## 12. Keep the exploration record thin

The project already has extensive durable documentation. Exploration should
improve search memory without creating another experiment notebook.

Use:

- one compact active exploration card in the applicable campaign document or
  explicitly named local loop state;
- local/raw logs outside Git for per-iteration profiler and benchmark output;
- compact artifacts for meaningful accepted/rejected checkpoints;
- one immutable worklog entry per substantial candidate family, structural
  checkpoint, promotion, blocker, or closure—not per trivial command;
- the existing campaign/dashboard for the final status;
- [`REFACTOR.md`](REFACTOR.md) only for temporary code that actually survives.

The active card may use this template:

```markdown
# Exploration: <target>

Status: active | blocked | closed-promoted | closed-exhausted
Baseline: <commit/artifact/command>
Envelope: <backend/model/quant/roles/shapes/fallback>
Primary metric: <complete metric and direction>
Measured owner/prize: <owner share and maximum saving>
Correctness oracle: <command/threshold>
Evaluator firewall: <paths/hashes and candidate allowlist>
Discovery / qualification / confirmation: <sets>
Budget: <iterations or hardware hours; maturation steps>
Human checkpoints: <conditions>

| Beam | Mechanism | Target/signature | Generalization risk | State | Evidence | Next decision |
| --- | --- | --- | --- | --- | --- | --- |
| X1 | ... | ... | ... | seed | ... | ... |
| X2 | ... | ... | ... | maturing | ... | ... |
| X3 | ... | ... | ... | near-miss | ... | ... |
```

Do not duplicate full commands, raw tables, or old history in this card. Link the
artifact or worklog entry that owns them.

---

## 13. Exit states

Every exploration eventually closes as one of:

- **Promoted:** a finalist passes qualification, confirmation, and normal default
  gates.
- **Exhausted:** credible families were tested and no candidate retains a
  sufficient measured prize; record what new fact could reopen the area.
- **Parked:** the mechanism remains plausible but current compiler, hardware,
  model, memory, or implementation prerequisites make further work uneconomic.
- **Blocked:** evaluator, hardware, source, correctness, or profile evidence is
  insufficient; name the exact unblock condition.
- **Split:** exploration discovers a reusable primitive or prerequisite that
  should become its own bounded logical unit before the original question can
  continue.

A negative exploration is valuable when it leaves a trustworthy map of failed
mechanisms, updated bottleneck evidence, and a sharper reopen condition. It is
not successful merely because it ran for a long time.

---

## 14. Opening checklist

Before starting a less-bounded optimization exploration:

- [ ] Current route and baseline are certified.
- [ ] Complete metric and measured owner are distinct and both recorded.
- [ ] The remaining prize is material.
- [ ] The declared specialization/generalization envelope is explicit.
- [ ] Forbidden input-conditioned behavior is understood.
- [ ] Evaluator, fixtures, objective, timing, and oracle are frozen.
- [ ] Candidate edit allowlist is narrow.
- [ ] Discovery, qualification, confirmation, and heldout use are defined.
- [ ] At least three genuinely different hypothesis families exist.
- [ ] Structural maturation and total hardware budgets are finite.
- [ ] Stagnation/advisor trigger is defined.
- [ ] Human-review triggers are defined.
- [ ] Cleanup, documentation, and exit behavior are defined.

If the checklist is not complete, the next task is exploration setup, not kernel
editing.

---

## Methodology provenance

The beam, structural-maturation, advisor, and thin-loop ideas were informed by
Sankalp's [auto-research kernel optimization writeup](https://sankalp.bearblog.dev/autoresearch/)
and its [Hacker News discussion](https://news.ycombinator.com/item?id=49309549).
The contest demonstrates the value of tight verifier loops and idea diversity,
but also highlights the risk of fixed-shape specialization, numerical
instability, candidate sprawl, and out-of-distribution failure. hipEngine adopts
the search-diversity lessons while retaining its stronger product-oriented
correctness, anti-gaming, heldout, lifecycle, and default-promotion contracts.
