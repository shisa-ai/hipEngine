# hipEngine cleanup audit

## What this is

hipEngine's debt is already written down — exhaustively, by agents, over months.
396 headings in `docs/REFACTOR.md`, 10,590 worklog entries, 747 environment
flags, 255 kernel sources, 62 campaign records, and 151 one-off `*audit*` scripts
in `scripts/` written per campaign and never consolidated.

The problem was never discovery. None of it was queryable, so nothing could be
worked through, and new debt landed faster than old debt retired.

This tool makes that surface queryable, keeps decisions about it durable, and
gates against it growing.

## The two halves

They share one decision store, so a `wontfix` recorded once is honoured by both.

**Inventory** — the standing catalogue. A live, idempotent scan of debt somebody
already wrote down: ledger entries, flags, kernels, campaign candidates, and the
worklog entries that declared work unfinished. Rows
persist across rescans and carry durable decisions. You *triage* inventory rows;
you rarely fix them directly, because the fix is usually a separate piece of work.

**Findings** — the active queue. Code checks scan the tree and queue things that
can be fixed now: violated architectural invariants, documented paths that moved,
unguarded GPU tests, branches on flags nobody governs. Every finding names the
edit that closes it. You *fix* findings and mark them resolved.

Some checks cross-reference the inventory: `ungoverned-flag-branch` reads
`inventory/flags.json` to find runtime branches on flags that are default-off
with no recorded way to retire them.

## Start here

**Run this first, every time:**

```bash
python3 audit/audit.py refresh
```

One command brings everything derived back in step with the tree. It rescans the
catalogue, re-runs the code checks, re-matches decisions onto rows whose text
changed, regenerates the `docs/` indexes, ratchets the budget **down** if cleanup
happened, writes a dated report, and finishes with a **needs a human** section
naming what it could not decide for you.

It is safe to run at any time and it never hides a problem: the budget only ever
moves down automatically, so a refresh cannot quietly absorb new debt. Commit the
regenerated `audit/` and `docs/` files with whatever change prompted the run.

Then see where things stand and pick up work:

```bash
python3 audit/audit.py status     # what exists and what state it is in
python3 audit/audit.py queue      # what can be fixed right now, grouped by cause
```

Then pick one of the three workflows below. **Do one workflow per session.**
Gathering, triaging, and fixing are different kinds of work with different
outputs, and mixing them produces a commit nobody can review.

---

## Workflow 1 — Gather (refresh the inventory)

`refresh` is the whole workflow. Run it when the tree has moved: after a batch of
commits, before starting a triage pass, or when `check` says the inventory is
stale.

```bash
python3 audit/audit.py refresh
python3 audit/audit.py check          # the gate; must exit 0
git add audit/inventory audit/findings audit/triage audit/budget.json audit/runs docs
```

The individual steps exist if you need one on its own — `inventory` for the
catalogue, `scan` for the code checks, `budget` to re-record the ceiling
deliberately, `report` for a dated run — but `refresh` runs them in the right
order and is what you should reach for.

What to look at in the output:

- **Re-matched decisions.** A rescan prints every decision it re-attached to a
  renamed row, with the similarity score. Sanity-check them — a wrong rebind
  carries a stale conclusion onto a different item. Anything re-matched has
  `rebound_from` set and should be re-read before being relied on.
- **New rows.** The budget will fail the gate if untriaged counts rose. That is
  the point: triage the new rows, do not raise the budget.
- **Orphans.** `audit.py orphans` lists decisions that now match nothing. Either
  the debt is gone (`triage ... --resolved`) or an item changed past matching.

Commit the regenerated JSON. It is generated but committed, so the diff between
two scans is reviewable.

---

## Workflow 2 — Triage (decide what a row is)

This is the main loop for inventory rows. The assignment is
[`agents/triage.md`](agents/triage.md) — hand it to an agent, or follow it
yourself.

```bash
python3 audit/audit.py open ledger -n 10                     # most signals first
python3 audit/audit.py open flag --signal "conflicting defaults"
python3 audit/audit.py open --stale                          # evidence moved since the call
python3 audit/audit.py expiring                              # past their review date
python3 audit/audit.py show ledger/<key>                     # full evidence
```

For each row:

1. **Investigate against the tree.** Read the files. Follow the call path. Check
   `git log` for why something is the way it is. The row's signals are leads, not
   conclusions — the extractor only sees what greps see.
2. **Decide**, and say why:

```bash
python3 audit/audit.py triage <row-id> \
    --tag STALE-LEDGER --do remove --severity medium \
    --note "hipengine/runtime/pool.py:88 was deleted in a1b2c3d; the entry describes
            a module that no longer exists"
```

3. `python3 audit/audit.py check` before finishing.

Rules that matter:

- **Never record a decision you did not verify.** If the evidence does not settle
  it, leave the row open and say so. An untriaged row is honest; a fabricated
  disposition is not.
- **`NOT-DEBT` is a first-class outcome.** Extractors produce false positives —
  `HIPENGINE_LITERAL` looks like a flag but is a placeholder inside the env-var
  scanner's own regex. Recording that is useful work, and the note must say what
  the extractor mistook.
- **The note carries the evidence.** Cite `file:line`, a command and its output,
  or a commit. A note that restates the tag is not a note.
- **Use `--expires` on anything conditional.** A `defer` whose reasoning depends
  on a blocker lifting should carry a date so it returns for re-confirmation.
- **Severity is consequence, not confidence.** A dead flag nobody reads is `low`
  however certain you are. A production route silently falling back is `high`.
- **Triage decides; it does not fix.** The fix is its own unit, with its own
  tests and commit. The exception is `--resolved` when you confirm the work was
  already done.

Choosing a disposition:

| Do | When |
| --- | --- |
| `promote` | A working implementation is not on the default path. Confirm it works first. |
| `remove` | Dead. Confirm no reader — including tests and harnesses — before saying so. |
| `qualify` | Plausible but unproven. **Name the exact command** that would settle it. If you cannot name one, that is itself a `GATE-CATCH22`. |
| `keep` | Correct as it stands. The note gives the justification. |
| `document` | The code is right and the docs disagree. Name both sides. |
| `defer` | Real, accepted, unscheduled. Name what unblocks it, and set `--expires`. |
| `wontfix` | A real problem we are deliberately not fixing. The note says why. |

### Working the worklog rows

A `worklog` row is an entry whose own text declared work unfinished. Nothing
re-opens it: entries are immutable, so the row is decided once and stays decided.
Work the unambiguous half first:

```bash
python3 audit/audit.py open worklog --signal "recorded as blocked"
python3 audit/audit.py open worklog --signal "recorded as handoff"
python3 audit/audit.py open worklog --signal "no later entry"   # nothing continued it
```

Three outcomes cover almost every row:

- **The work was finished.** Cite the entry, commit, or file that finished it and
  record `--resolved`. Most rows resolve this way, which is why the ceiling drops.
- **The work is still real.** Record what it needs — `qualify` with the exact
  command, `defer` with `--expires`, or `remove` if the entry describes something
  that should no longer happen.
- **`NOT-DEBT`.** The status was used loosely, or the row's "not in the tree"
  observation is a path the entry quotes rather than one it points at.

A pre-cutoff row describes the tree as of its date, so check the referent against
today's tree rather than trusting the entry. `no later entry by the same worker
shares this topic prefix` is a hint that nobody continued it — the prefix is not
an identity, so confirm before treating a row as dropped.

---

## Workflow 3 — Fix (clear the queue)

Findings name their own edit, so this is ordinary engineering work. The
assignment is [`agents/fix.md`](agents/fix.md).

```bash
python3 audit/audit.py queue -n 8 -e 3        # groups, largest cause first
python3 audit/audit.py queue --check doc-path-drift
```

1. **Take a whole group, not one row.** The queue groups findings by shared fix
   because one cause usually covers dozens. Fixing the cause is the unit of work;
   fixing one instance of it is not.
2. **Make the change**, following the normal repository rules in `AGENTS.md`:
   narrowest relevant test, a worklog entry for a substantial unit, an atomic
   commit with explicit staging.
3. **Close the rows out:**

```bash
python3 audit/audit.py triage <row-id> --tag DOC-DRIFT --do document \
    --severity low --note "fixed in <commit>: path corrected to hipengine/kernels/" --resolved
```

4. Re-run `scan` — the findings should be gone. Rows that disappear with a
   `--resolved` decision behind them are the record that the work happened.
5. `python3 audit/audit.py budget` to re-record the ceiling **downward**, and
   commit it. That is how cleanup shows up as a number.

If a finding is wrong, triage it `NOT-DEBT` with a note saying what the check
mistook, and consider tightening the check — a systematic false positive is worth
more than a single row.

---

## The model

**A row is evidence, not a verdict.** An extractor reports observations — "no
read site under `hipengine/`", "names 7 paths, 3 no longer exist" — and never a
conclusion.

**Triage is what turns a row into a decision**, and it is durable. Decisions live
in `triage/*.jsonl`, keyed by row id, in files that re-scanning never touches.
Every decision records a hash of the evidence it was made against.

**A rescan lands on the same audit item even after the text changed.** Row ids
for ledger entries and campaign candidates are content-derived, so rewording a
heading would otherwise orphan its decision. Each row therefore also carries
match *hints* — a stable anchor, what the row names, and its content words — and
a rescan re-attaches any decision whose id moved, reporting the similarity it
matched on. A row that already has a decision is never claimed by a rebind, so a
decision cannot be stolen from another item.

**A decision made against changed facts comes back.** When evidence moves, the
row is reported `stale` rather than silently standing on a conclusion that may no
longer hold. `--expires` does the same on a schedule. This is what lets a wrong
early call be corrected instead of calcifying.

**A decision whose row vanishes is reported, not dropped.** `audit.py orphans`
lists them.

**The budget is the anti-laziness gate.** `budget.json` records the untriaged
count per kind. `check` fails when a count rises, so new debt cannot land without
being triaged. **Never raise the budget to get past the gate.** Lowering it is
the point.

A kind whose population grows with ordinary work narrows its gate to the rows
that need a decision, in the budget's `select` block:

```json
"select": {"worklog": {"evidence.status": ["blocked", "handoff"]}}
```

A spec matches row fields (`evidence.<name>` reads the row's evidence); a list of
values means any of them, and a list of specs is ORed. Rows outside the gate are
still untriaged, still listed by `open`, and still counted in the state table —
they just do not set the ceiling. `worklog` needs this because its population
grows with every session: gating all of it would tie the ceiling to the rate of
work instead of to the backlog awaiting a decision, and a permanently red gate is
a gate nobody reads.

## Reference

### Commands

```bash
python3 audit/audit.py inventory              # rescan the standing catalogue
python3 audit/audit.py scan                   # run the code checks
python3 audit/audit.py status                 # where the cleanup stands
python3 audit/audit.py queue -n 8 -e 3        # fixable work, grouped by cause
python3 audit/audit.py open <kind> -n 20      # untriaged rows, most signals first
python3 audit/audit.py open --stale           # decisions whose evidence moved
python3 audit/audit.py expiring               # decisions past their review date
python3 audit/audit.py orphans                # decisions that match no row
python3 audit/audit.py show <row-id>          # one row, with its triage
python3 audit/audit.py triage <row-id> --tag TAG --do DISPOSITION \
    --severity high|medium|low --note "..." [--expires YYYY-MM-DD] [--resolved]
python3 audit/audit.py check                  # the gate
python3 audit/audit.py report                 # dated run under audit/runs/
python3 audit/audit.py budget                 # re-record the ceiling deliberately
python3 audit/audit.py budget --lower-only    # ratchet down only
python3 audit/audit.py refresh                # all of the above, in order
```

`refresh` is the one to run routinely. The rest are for when you want a single
step. `budget` without `--lower-only` can raise the ceiling, which is a
deliberate act needing a recorded reason — `refresh` never does it.

### Extractors (inventory)

| Kind | Source | What it pairs up |
| --- | --- | --- |
| `ledger` | `docs/REFACTOR.md` | Each `##` entry against the tree: do the paths and flags it names still exist, is it dated, does it state a removal condition. |
| `flag` | `HIPENGINE_*` across the tree | Read sites by root, default state, whether a removal condition is recorded, and conflicting defaults across modules. |
| `kernel` | `hipengine/kernels/` | Each source against its referrers, registry keys, `__global__` entry points, and tests. |
| `candidate` | `docs/campaigns/` | Rows recorded as rejected/deferred/parked **while citing a measurement** — where retrievable performance hides. |
| `worklog` | `worklog/entries/` | Entries that declared unfinished business — an open `status`, a `Next` naming a blocker or approval, or a pre-cutoff `### Next` marker — against the tree they describe. Only `blocked` and `handoff` markers set the gate; in-flight `checkpoint` rows stay visible and untriaged. |

An extractor reports observations. It does not resolve dispatch, run a kernel, or
measure anything.

### Checks (findings)

| Check | Finds |
| --- | --- |
| `torch-hot-path` | Module-level `import torch` in code reached by `LLM.generate()`. |
| `axis-branch` | `if backend == …` / `if quant == …` where a registry key belongs. |
| `unguarded-hip-test` | Tests driving ROCm with no availability guard, which fail rather than skip on a no-GPU runner. |
| `doc-path-drift` | A documented path that does not exist **but is findable elsewhere**, so the fix is a known edit. |
| `ungoverned-flag-branch` | Runtime branches on a flag that is default-off with no recorded removal condition. Cross-references the flag inventory. |
| `stub` | `raise NotImplementedError` reachable from the runtime. |
| `marker` | `TODO`/`FIXME`/`XXX`/`HACK` in shipped runtime code. |

### Tags

What a row turned out to be: `DEAD-FLAG`, `LOST-OPT`, `EXACTNESS-REJECT`,
`ORPHAN-KERNEL`, `STALE-LEDGER`, `UNREACHABLE`, `SKELETON`, `GATE-CATCH22`,
`DUP-DISPATCH`, `BENCH-INVALID`, `DOC-DRIFT`, `DEAD-CODE`, `TEST-GAP`,
`NOT-DEBT`.

Three of them encode rules from `AGENTS.md` and `docs/OPTIMIZATION.md` directly,
because these are the failure shapes this project actually has:

| Tag | The rule it enforces |
| --- | --- |
| `GATE-CATCH22` | A restriction with no command that could lift it is a bug in the gate. `AGENTS.md` "Product Defaults". |
| `LOST-OPT` | A measured, non-regressive win belongs on the default path unless a concrete blocker is recorded. `docs/OPTIMIZATION.md` §4.3. |
| `EXACTNESS-REJECT` | A candidate was discarded for not being bit-identical to the strict parent. `docs/OPTIMIZATION.md` §4.1 is explicit that **exactness alone cannot reject a production-correct candidate** — `strict` is a debugging oracle, not the promotion bar. |

`EXACTNESS-REJECT` is deliberately separate from `LOST-OPT`. A `LOST-OPT` was
never promoted for some reason; an `EXACTNESS-REJECT` was actively rejected for
failing a bar it was never required to meet. The remedy differs: an
`EXACTNESS-REJECT` pairs with `qualify`, and the note must name the production
gate — the calibrated mean/tail/max KL, top-1 by category, determinism,
isolation and task gates in `docs/EXECUTION-PROFILES.md` — that the candidate
should have been measured against instead. The `candidate` extractor already
flags these: `audit.py open candidate --signal "exactness bar"`.

Every decision requires a note. A disposition without a reason is not a decision.

## Working it with agents

The two assignments under [`agents/`](agents/) are tool-neutral and are the
canonical instructions:

| File | Posture | Does |
| --- | --- | --- |
| [`agents/triage.md`](agents/triage.md) | read-only | Decides what rows are. Writes only the triage store. |
| [`agents/fix.md`](agents/fix.md) | edits code | Takes a queue group, fixes the shared cause, commits, closes rows out. |

The split is deliberate: triage decides and fixing changes code, and keeping
them apart is what stops a cleanup commit from also changing behaviour. Scope a
run by group or signal — `queue --check doc-path-drift`, `open candidate
--signal "exactness bar"` — never "triage everything".

Any runner-specific registration (subagent stubs, agent definitions) is
per-user setup and is not checked in. The files under `agents/` are the
canonical copies — point whatever runner you use at them.

## Extending it

Add an extractor when a debt surface is not catalogued; add a check when code can
be scanned for something with a concrete fix.

**A new extractor** goes in `audit/hipaudit/inventory/<name>.py`:

```python
from ..core import Row
from . import corpus, doc_corpus, register, tokens

@register("mykind")
def extract() -> tuple[list[Row], dict]:
    rows = []
    for path, text in sorted(corpus().items()):
        ...
        rows.append(Row(
            kind="mything",                       # singular; the row id prefix
            key=stable_key,                       # stable across rescans if you can
            title=...,
            location=f"{path}:{line}",
            evidence={...},                       # facts, machine-readable
            signals=["what you observed"],        # observations, never verdicts
            hints={"anchor": path,                # required: identity across rewrites
                   "refs": sorted(named_things),
                   "tokens": tokens(title)},
        ))
    return rows, {"scanned": "..."}
```

Then import it in `inventory/__init__.py` so it registers.

**A new check** goes in `audit/hipaudit/checks/<name>.py` and uses the `finding`
helper, which requires a `fix` and a `why`:

```python
from . import finding, register

@register("my-check")
def my_check() -> tuple[list[Row], dict]:
    return [finding("my-check", key, title, location,
                    fix="the smallest edit that closes this",
                    why="the rule or fact that makes it wrong")], {}
```

Then import it in `checks/__init__.py`.

Requirements for either:

- **`hints` is not optional.** Without an anchor, a decision cannot be re-matched
  when the row's text changes, and triage will orphan on the next rescan.
- **Signals are observations.** "no test names it" is a signal. "dead kernel" is a
  verdict and belongs in triage.
- **Precision over recall.** A noisy check gets ignored, which is worse than an
  absent one. `doc-path-drift` fires only when it can say where the file actually
  is; everything illustrative is skipped.
- **Run `audit.py budget` after adding one**, or the gate fails on the new rows.
- **Add a contract test** in `audit/tests/`.

## Layout

```
audit/
  audit.py            CLI
  hipaudit/
    core.py           rows, triage, the durable store, rebinding, expiry
    inventory/        the extractors
    checks/           the code checks
    report.py         computed tables and the dated run
  inventory/*.json    the standing catalogue; generated; committed so diffs are reviewable
  findings/*.json     what code scanning queued to fix; generated
  triage/*.jsonl      durable decisions for both; re-scanning never touches these
  budget.json         the untriaged ceiling the gate enforces
  runs/<stamp>/       dated REPORT.md + snapshot.json
  agents/             assignments for whoever works the audit: triage.md, fix.md
  tests/              contract tests for durability and validation
```

## Tests

```bash
python3 -m unittest discover -s audit/tests -t audit
```

Hermetic; the store tests run against a temporary directory.

## What this does not establish

The inventory says what is *catalogued*, not what is *true*. A row with no
signals is not thereby healthy, and a row with several is not thereby debt. The
counts in any report are a floor. Nothing here replaces reading the code.
