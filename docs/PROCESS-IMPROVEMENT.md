# hipEngine Process Improvement

Status: proposal for human review, except for the Worklog2 storage migration.
The immutable per-unit worklog design is separately approved and implemented by
[`PLAN-WORKLOG2-revamp.md`](PLAN-WORKLOG2-revamp.md), which supersedes this
file's worklog convention and no-retrofit rollout notes. All sprint naming,
planning registry, decision brief, experiment, and other migration conventions
below remain unapproved until the human lead explicitly selects them.

This document proposes how hipEngine could organize bounded sprints, durable
decisions, performance experiments, and their relationship to the existing
architecture, testing, benchmark, and history documents. It does not replace
`AGENTS.md`, `PLAN.md`, `TESTING.md`, or `BENCHMARK.md`.

## Recommendations for review

1. Use immutable, date-based sprint IDs: `S-YYYYMMDD-short-slug`.
2. Do not use `000-`, `001-`, and similar global sequence prefixes for sprints.
   Numeric order belongs inside a sprint (`P0`, `P1`, `P2`) rather than in its
   permanent identity.
3. Introduce one planning registry as the only source of truth for active,
   queued, blocked, parked, and closed sprints.
4. Keep durable architecture, feature, benchmark, and hardware references at
   their existing semantic paths. They may supply a sprint with context, but
   they do not independently own current priority or completion state.
5. Record enduring fork-in-the-road choices as short decision briefs under
   `docs/decisions/`, using `D-YYYYMMDD-short-slug.md`.
6. Require a measure-first Phase 0 with thresholds fixed before implementation
   for expensive or uncertain performance work.
7. Put the verdict and material metric in sprint outcomes and new `WORKLOG.md`
   headings so the journal is skimmable without weakening its evidence detail.
8. Add a shipping-default parity gate whenever an optimized route becomes a
   library or server default.

Under this proposal, the date is the day a sprint or decision is opened, not its
priority. The slug states the durable subject. Status, priority, and result live
in metadata and the planning registry; they never require renaming the file.

## Why change

hipEngine does not lack rigor. It already has RED/GREEN testing, independent CPU
oracles, exact benchmark commands, compact artifacts, profiler attribution,
anti-gaming rules, negative-result retention, source-lineage checks, default
promotion rules, and a cleanup ledger.

The current friction is findability and ownership:

- `WORKLOG.md` is an excellent append-only evidence journal but is too large to
  answer "what is active now?" or "why was this choice made?" quickly.
- `ROADMAP.md`, `SOL-OPTIMIZATION.md`, `OPTIMIZE*.md`, `TUNING-*.md`, feature
  plans, and current dashboards each contain useful material, but several also
  describe themselves as the active plan, coordinator, punchlist, or decision
  surface.
- A benchmark artifact usually says what ran and whether it was accepted, but
  it does not consistently preserve the hypothesis, maximum prize, pre-run
  keep/kill threshold, or condition for reopening a rejected design.
- Explicit benchmark profiles and automatic public defaults can drift even when
  both remain individually correct.

The objective is one quick path to four answers:

1. What bounded work is active?
2. What evidence allowed it to start?
3. What exact result closes or kills it?
4. Where is the durable rationale after it closes?

## Document roles

Each document should have one role. A document may link to another role, but it
should not silently compete with it.

| Role | Canonical surface | Owns | Does not own |
| --- | --- | --- | --- |
| Architecture | `PLAN.md` | Invariants, plugin boundaries, long-lived architecture | Current sprint priority |
| Strategy | `ROADMAP.md` | Where hipEngine competes and broad sequencing | Per-experiment status |
| Process | `AGENTS.md`, `TESTING.md`, `BENCHMARK.md`, this file | Working rules and promotion contracts | Product/kernel design |
| Planning registry | proposed `docs/planning/README.md` | The complete current sprint state and WIP order | Detailed experiment evidence |
| Sprint | `docs/planning/sprints/S-YYYYMMDD-slug.md` | One bounded outcome, phases, dependencies, exit gate, outcome | Permanent architecture or an unbounded backlog |
| Decision | `docs/decisions/D-YYYYMMDD-slug.md` | Enduring choice, alternatives, measured/estimated basis, revisit trigger | Chronological lab notes |
| Domain reference/dashboard | Existing semantic docs such as `MTP.md`, `KERNELS.md`, or `MTP-LLAMACPP-PARITY.md` | Current technical contract, path map, or evidence summary | Repository-wide priority |
| Benchmark evidence | `benchmarks/results/*.json` and the canonical rollup | Reproducible measurements and claim eligibility | Narrative project coordination |
| Journal | `WORKLOG.md` | Append-only chronological commands, results, and handoff detail | Current status or durable decision lookup |
| History/archive | `*-HISTORY.md`, later `docs/archive/` material | Superseded notebooks and historical context | Current claims or active work |

### Proposed layout

Use this layout for new planning material:

```text
docs/
  README.md
  PROCESS-IMPROVEMENT.md
  planning/
    README.md                         # only active-work registry
    sprints/
      S-20260713-paro-decode-recovery.md
  decisions/
    D-20260713-default-route-policy.md
  archive/
    README.md                         # index of superseded notebooks/history
```

Do not move stable root documents merely to make the tree look tidy. Their
semantic filenames and existing inbound links are valuable. Create the new
directories when their first real registry, sprint, or decision is committed.

Closed sprint documents remain at their original immutable path under
`docs/planning/sprints/`; the planning registry moves their row from Active to
Closed. Moving a file into an `active/` or `archive/` directory on every state
change would break links and create noisy commits. `docs/archive/` is for bulky
superseded notebooks that have a separate current canonical surface, matching
the existing dashboard plus `*-HISTORY.md` pattern.

## Sprint identity and lifecycle

### Naming

The canonical sprint name is:

```text
S-YYYYMMDD-short-slug
```

Examples:

- `S-20260713-paro-decode-recovery`
- `S-20260713-exact-mtp-natural-horizon`
- `S-20260720-gfx1100-hip-vulkan-refresh`

Use the date on which the bounded sprint is opened. Use lowercase ASCII words
separated by hyphens. Prefer a subject and outcome over an implementation name.
If two sprints open on the same day, distinct slugs make them unique.

Do not use a global `000-foo` sequence because it:

- looks like priority or execution order even after priorities change;
- requires a central counter and creates avoidable concurrent-edit conflicts;
- invites renumbering when work is inserted, split, or merged;
- carries less useful information when found outside its index.

Within a sprint, use `P0`, `P1`, `P2`, and task IDs for ordered phases. Existing
IDs such as `SOL-R0` remain valid aliases and should be recorded in the sprint
header rather than erased from history.

### What qualifies as a sprint

A sprint is a bounded execution unit with:

- one decision or measurable outcome;
- a frozen baseline or a Phase-0 task that establishes it;
- explicit in-scope and out-of-scope work;
- named dependencies and likely high-conflict files;
- a correctness class and applicable gate;
- a stop/exit condition;
- an outcome that can be accepted, rejected, parked, or blocked.

A feature area, permanent dashboard, open-ended tuning theme, or backlog is not
a sprint. Backlog rows become sprints only when prerequisites, baseline, and an
exit gate are concrete enough to start. This prevents every possible idea from
becoming another apparently active plan.

### Status vocabulary

The planning registry uses:

| Status | Meaning |
| --- | --- |
| `queued` | Bounded and ready, but intentionally not consuming WIP. |
| `active` | The current implementation or measurement unit. |
| `blocked` | Cannot proceed until a named external condition changes. |
| `parked` | Premise was tested or priority removed; retry only on its recorded trigger. |
| `closed-accepted` | Exit gate passed and the retained/default result landed. |
| `closed-rejected` | The tested premise failed its frozen gate; evidence and reopen trigger remain. |
| `closed-superseded` | Another sprint or decision replaced this scope. |

Avoid using `open` for both "ready" and "active." That ambiguity is one source
of overlapping work.

### WIP and overlap rules

Naming does not prevent overlap; ownership rules do:

1. `docs/planning/README.md` is the only repository-wide active-work list.
2. Exactly one sprint may own a given
   `(backend, model/quant path, phase or bottleneck family)` decision at a time.
3. There may be only one priority-zero release blocker. Other work that touches
   its code or evidence surface stays queued unless explicitly unblocked.
4. A sprint that depends on another sprint is queued, not active, until the
   dependency produces the fact it needs.
5. High-conflict-file ownership is listed in each active sprint. Two active
   sprints may read the same domain document but may not both own edits to the
   same high-conflict runtime surface without an explicit handoff.
6. A coordinator/dashboard may contain many queued rows, but only rows linked to
   active sprint IDs are in progress.
7. Instrumentation is a phase or enabling sprint, not evidence that the parent
   performance sprint is accepted.

The WIP cap should follow real hardware and coordination capacity rather than an
arbitrary global number. The tuple-ownership and dependency rules are the hard
limits; the planning registry should make any additional capacity limit visible.

## Sprint brief template

Keep the sprint document short enough to reread before each phase. Detailed
commands and raw chronology remain in artifacts and `WORKLOG.md`.

```markdown
# S-YYYYMMDD-short-slug — Outcome-oriented title

Status: active
Opened: YYYY-MM-DD
Coordinator: <person/agent or handoff role>
Parent: <ROADMAP section or portfolio ID>
Legacy aliases: <for example SOL-R0>
Scope tuple: <backend; model/quant; phase/bottleneck>
Dependencies: <IDs or none>
High-conflict files: <paths or none>
Baseline: <artifact/commit/command>
Correctness class: exact-by-construction | oracle-tolerant | quality-traded
Exit gate: <one sentence>

## Question and why now
## Measured baseline and maximum prize
## In scope / non-goals
## P0 measurement gate and frozen stop rules
## Ordered phases
## Validation and default-promotion gate
## Outcome, predicted versus observed, and reopen trigger
```

Update `Status` and the Outcome section in place. Preserve the original
hypothesis and thresholds so the record shows what was known before the result.
If a factual premise changes, add a dated correction rather than silently
rewriting it.

## Measure-first experiment contract

Use an explicit Phase 0 before substantial engine work when any of these holds:

- the change is a new kernel design rather than a mechanical port;
- the estimated implementation is more than roughly half a focused day;
- the production bottleneck or achievable prize is uncertain;
- the change touches round-critical state, math, a public default, or several
  plugin/dispatch surfaces;
- a cheap microtest, profiler pass, offline replay, or resource skeleton can
  invalidate the premise first.

Freeze these fields before implementation:

1. **Question/hypothesis.** Which measured bucket should move, and why?
2. **Baseline.** Exact model, quant, shape, hardware, source, command, artifact,
   and effective runtime route.
3. **Maximum prize.** Component share, roofline/Amdahl ceiling, and expected E2E
   range. Mark every value `measured`, `derived`, or `estimated`.
4. **Correctness class.** `exact-by-construction`, `oracle-tolerant`, or
   `quality-traded`; identify the oracle and thresholds.
5. **P0 probe.** The cheapest test that can disprove the premise without full
   integration.
6. **GO/STOP rules.** Quantitative thresholds, required matrix cells, maximum
   variants/timebox, and what happens on an ambiguous result.
7. **Promotion gate.** Required E2E shapes, default-path check, memory limits,
   profiler confirmation, and artifact/rollup updates.
8. **Reopen trigger.** The new fact required after rejection or parking.

Thresholds are fixed before seeing candidate results. If they must change, log a
dated correction explaining the invalid premise and rerun the relevant control;
do not reinterpret the old result under a newly convenient bar.

The stop bar governs whether to continue an invasive design or make it a
default. It does not override hipEngine's rule that an exact, measured,
same-suite non-regressive component win is retained. Such a component may land
independently even when the larger design or headline target is killed.

### Performance-attribution outcome

The sprint document should end with a compact narrative in this order:

1. verdict at the top (`LANDED`, `REJECTED`, `PARKED`, `BLOCKED`, or
   `CORRECTION`);
2. exact method and artifact links;
3. measured counter/component table;
4. interpretation and bottleneck classification;
5. observed result versus the frozen prediction and keep/kill bar;
6. default/cleanup consequence;
7. reopen trigger.

The compact JSON artifact remains the claim evidence. The sprint narrative
explains the decision; it must not become a second store of uncited benchmark
numbers.

## Decision briefs

Create a decision brief when a choice will outlive the sprint that discovered
it, especially for:

- architecture or plugin-boundary choices;
- changing a public default or supported semantic contract;
- accepting a performance/correctness/memory tradeoff;
- choosing build versus reuse/route-to-another-engine;
- ending or funding a multi-sprint line of work;
- a fork where the strongest option cannot be selected from benchmark policy
  alone.

Name it `docs/decisions/D-YYYYMMDD-short-slug.md`. Keep it close to one page:

```markdown
# D-YYYYMMDD-short-slug — Decision question

Status: proposed | accepted | superseded
Date: YYYY-MM-DD
Related sprints: <IDs>
Decision owner: <human/project role>

## Context and constraints
## Options A/B/C
## Comparison
## Strongest case for each viable option
## Decision and consequences
## Revisit trigger
```

Tag every comparison value as `measured`, `derived`, or `estimated`. A brief may
end with "not decided" if it clearly records the missing fact and who decides.
Once accepted, do not turn it into a live notebook. A later reversal creates a
new decision that marks the old one superseded.

## Shipping-default parity gate

An optimization is not fully promoted merely because an explicit benchmark
profile wins. Before a library or server default changes, compare the clean,
ordinary public invocation with the explicit retained profile:

- same resolved backend, model, quant, KV policy, graph policy, registry keys,
  and effective fast-path manifest;
- same tokens and state for exact-by-construction routes, or the declared
  correctness/quality gate for other classes;
- expected kernels present in the profiler trace;
- no diagnostic instrumentation or unsafe override active;
- performance within the sprint's frozen tolerance on the promotion shapes;
- public metadata reports the route that actually ran.

Route-resolution equality should have a portable unit/fixture test where
possible. The GPU performance/state parity is a promotion or release gate, not
necessarily a per-commit CI job.

## `WORKLOG.md` convention

`WORKLOG.md` remains append-only and keeps exact commands, measurements,
decisions, and handoff detail. Do not rewrite old headings. For new entries,
make the outcome skimmable:

```text
## YYYY-MM-DD - [SPRINT-ID] [VERDICT metric] Short result
```

Examples:

```text
## 2026-07-13 - [S-20260713-paro-decode-recovery] [LANDED +5.4%] Restore cached decode route
## 2026-07-13 - [S-20260713-verify-fusion] [REJECTED -4.0%] Kill lane-pair fusion
## 2026-07-13 - [CORRECTION] Reclassify prefill probe after effective-route audit
```

Allowed verdicts are `LANDED`, `REJECTED`, `NEUTRAL`, `BLOCKED`, `DECISION`,
`CORRECTION`, and `PROCESS`. Include a metric only when its scope is clear in
the heading; the body still carries the full evidence policy. A sprint-closing
entry links the sprint document, decision brief if any, artifact, and commit.

## Claims audit

At a point release or major benchmark refresh, perform a bounded claims audit of
the root README and `benchmarks/README.md`:

1. select each current headline claim;
2. follow it to an eligible artifact and measured revision;
3. verify its semantic class, protocol, hardware, and default-path status;
4. retract or relabel stale/diagnostic claims;
5. record the audit count and corrections in `WORKLOG.md`.

This is a release/milestone ritual, not a new continuous review bureaucracy.
Keep the existing current-dashboard plus immutable-history split; do not add
inline superseded annotations everywhere unless a file has no current/history
separation.

## Proposed current-document migration map

Do not perform a bulk move. Apply this role classification when a document is
next materially updated:

| Existing document(s) | Target role |
| --- | --- |
| `PLAN.md` | Architecture; unchanged path. |
| `ROADMAP.md` | Strategy; refresh priorities, but do not run experiments from it. |
| `SOL-OPTIMIZATION.md` | Seed for the single planning registry; its bounded `R*` work packages become sprint IDs only when activated. |
| `OPTIMIZE.md`, `OPTIMIZE-DENSE.md`, `TUNING-gguf.md`, `TUNING-gfx1151.md` | Domain backlogs and technical playbooks; stop describing them as independently active coordinators. |
| `MTP-LLAMACPP-PARITY.md`, `PARO-GGUF-MTP-TRANSFER.md`, `HIP-vs-VULKAN.md` | Current evidence dashboards; retain their compact current/status role. |
| `*-HISTORY.md` | Historical archive; never a current claim or sprint state. |
| `IMPLEMENTATION.md` | Foundation/implementation checklist; not the current optimization queue. |
| `WORKLOG.md` | Chronological evidence only; never the active-work registry. |

For example, activating legacy `SOL-R0` would create a sprint such as
`S-20260713-paro-decode-recovery`, list `SOL-R0` as an alias, and link back to
the relevant `SOL-OPTIMIZATION.md` context. Other `SOL-R*` rows remain queued;
they do not get empty sprint files in advance.

If a later migration moves an existing document, update all tracked links in
the same logical commit and leave a short pointer at a widely cited old path
when link stability is valuable. Do not mix mass document relocation with
kernel/runtime work.

## Possible rollout after approval

1. Apply only the conventions the human lead approves.
2. If the planning layout is approved, create `docs/planning/README.md`,
   `docs/planning/sprints/`, and its first sprint brief in one docs-only commit.
3. If decision briefs are approved, create `docs/decisions/` only when the first
   enduring decision brief is needed; do not manufacture briefs retroactively.
4. Pilot the experiment contract for two or three substantial sprints before
   adding mandatory JSON-schema fields or checker scripts.
5. After the pilot, consider generating the planning index from sprint metadata
   and adding optional artifact fields for hypothesis, semantic class, predicted
   prize, frozen thresholds, outcome, and reopen trigger.
6. Do not retrofit thousands of artifacts or old `WORKLOG.md` entries.

Second-wave documentation gaps, separate from this organization change, are a
compact checked semantic-spec card for each complex model/ABI and an explicit
hipEngine server threat/deployment model. Both are useful, but neither should
block fixing sprint ownership and default-path parity first.

## Methodology provenance

This process incorporates useful patterns from q27 at commit `97f6aba63047`,
especially its decision briefs, measure-only Phase 0 gates, precommitted
keep/kill thresholds, outcome-in-header convention, predicted-versus-observed
attribution, and explicit reopen triggers. hipEngine's existing CPU-reference,
multi-prompt anti-gaming, provenance, source-lineage, rollup, and refactor-debt
systems remain the stronger source of truth in their respective areas.
