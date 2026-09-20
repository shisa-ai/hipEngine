# hipEngine cleanup audit

hipEngine's debt is already written down — exhaustively, by agents, over months.
396 headings in `docs/REFACTOR.md`, 3,333 worklog entries, 747 environment
flags, 255 kernel sources, 61 campaign records. The problem was never discovery.
It is that none of it was queryable, so nothing could be worked through, and new
debt landed faster than old debt was retired.

This tool has two halves that share one decision store.

**Inventory** is the standing catalogue: a live, idempotent scan of debt somebody
already wrote down — ledger entries, flags, kernels, campaign candidates. Rows
persist across rescans and carry durable decisions.

**Findings** are what code scanning queues up to fix: architectural invariants
violated, documented paths that moved, stubs, flag-guarded branches nobody can
reach. Every finding names the edit that closes it. Some cross-reference the
inventory — `ungoverned-flag-branch` reads `inventory/flags.json` to find runtime
branches on flags that are default-off with no recorded way to retire them.

Both flow through the same triage store, so a `wontfix` recorded once is honoured
everywhere.

## The model

**A row is evidence, not a verdict.** An extractor sees what greps see. It
reports observations — "no read site under `hipengine/`", "names 7 paths, 3 no
longer exist" — and never a conclusion. Extractors are routinely wrong:
`HIPENGINE_LITERAL` looks like a flag but is a placeholder inside the env-var
scanner's own regex.

**Triage is what turns a row into a decision**, and it is durable. Decisions live
in `triage/*.jsonl`, keyed by row id, in their own files that re-scanning never
touches. Every decision records a hash of the evidence it was made against.

**A rescan lands on the same audit item even after the text changed.** Row ids
for ledger entries and campaign candidates are content-derived, so rewording a
heading would otherwise orphan its decision. Each row therefore also carries
match *hints* — a stable anchor plus what the row names and its content words —
and a rescan re-attaches any decision whose id moved, reporting the similarity
it matched on. A row that already has a decision is never claimed by a rebind,
so a decision cannot be stolen from another item.

**A decision made against changed facts comes back.** When evidence moves, the
row is reported `stale` rather than silently standing on a conclusion that may
no longer hold. A decision can also carry `--expires`, after which it returns as
`expired` for re-confirmation — use it on `defer` and on anything whose reasoning
depends on a blocker that may lift. This is the mechanism that lets a wrong early
call be corrected instead of calcifying.

**A decision whose row vanishes is reported, not dropped.** `audit.py orphans`
lists them: either the debt is gone (mark `--resolved`) or an item changed past
matching.

**The budget is the anti-laziness gate.** `budget.json` records the untriaged
count per kind. `check` fails when a count rises, so new debt cannot be added
without triaging it. Lowering the budget is how cleanup shows up as a number.

## Commands

```bash
python3 audit/audit.py inventory              # rescan the standing catalogue
python3 audit/audit.py scan                   # run the code checks
python3 audit/audit.py queue -n 8             # what can be fixed, grouped by cause
python3 audit/audit.py status                 # where the cleanup stands
python3 audit/audit.py open ledger -n 20      # untriaged rows, most signals first
python3 audit/audit.py open flag --signal "conflicting defaults"
python3 audit/audit.py open --stale           # decisions whose evidence moved
python3 audit/audit.py expiring               # decisions past their review date
python3 audit/audit.py orphans                # decisions that match no row
python3 audit/audit.py show flag/HIPENGINE_X
python3 audit/audit.py triage flag/HIPENGINE_X --tag LOST-OPT --do promote \
    --severity high --note "why this disposition is right" [--expires 2026-12-01]
python3 audit/audit.py check                  # the gate
python3 audit/audit.py report                 # dated run under audit/runs/
python3 audit/audit.py budget                 # re-record the ceiling
```

## Extractors

| Kind | Source | What it pairs up |
| --- | --- | --- |
| `ledger` | `docs/REFACTOR.md` | Each `##` entry against the tree: do the paths and flags it names still exist, is it dated, does it state a removal condition. |
| `flag` | `HIPENGINE_*` across the tree | Read sites by root, default state, whether a removal condition is recorded, and **conflicting defaults across modules**. |
| `kernel` | `hipengine/kernels/` | Each source against its referrers, registry keys, `__global__` entry points, and tests. |
| `candidate` | `docs/campaigns/` | Rows recorded as rejected/deferred/parked **while citing a measurement** — where retrievable performance hides. |

An extractor reports observations. It does not resolve dispatch, run a kernel,
or measure anything.

## Checks

| Check | Finds |
| --- | --- |
| `torch-hot-path` | Module-level `import torch` in code reached by `LLM.generate()`. |
| `axis-branch` | `if backend == …` / `if quant == …` where a registry key belongs. |
| `unguarded-hip-test` | Tests driving ROCm with no availability guard, which fail rather than skip on a no-GPU runner. |
| `doc-path-drift` | A documented path that does not exist **but is findable elsewhere**, so the fix is a known edit. |
| `ungoverned-flag-branch` | Runtime branches on a flag that is default-off with no recorded removal condition. Cross-references the flag inventory. |
| `stub` | `raise NotImplementedError` reachable from the runtime. |
| `marker` | `TODO`/`FIXME`/`XXX`/`HACK` in shipped runtime code. |

`doc-path-drift` only fires when it can say where the file actually is;
illustrative paths, MIME types, and placeholders are skipped. Precision matters
more than recall, because a noisy check gets ignored.

## Tags and dispositions

Tags name what a row turned out to be: `DEAD-FLAG`, `LOST-OPT`,
`ORPHAN-KERNEL`, `STALE-LEDGER`, `UNREACHABLE`, `SKELETON`, `GATE-CATCH22`,
`DUP-DISPATCH`, `BENCH-INVALID`, `DOC-DRIFT`, `DEAD-CODE`, `TEST-GAP`, and
`NOT-DEBT` for when the extractor was wrong.

Dispositions are hipEngine's cleanup verbs. Most debt here resolves by turning
something on or deleting it, not by fixing a bug:

| Do | Means |
| --- | --- |
| `promote` | Make it the default path. |
| `remove` | Delete the dead flag, path, or entry. |
| `qualify` | Run the gate that is missing, then decide. |
| `keep` | Justified as it stands; the note says why. |
| `document` | The code is right, the documentation is not. |
| `defer` | Real, accepted, unscheduled; the note says what unblocks it. |

`GATE-CATCH22` and `promote` implement `AGENTS.md` "Product Defaults" directly:
a restriction with no command that could lift it is a bug in the gate, and a
working implementation belongs on the default path.

Every decision requires a note. A disposition without a reason is not a
decision.

## Layout

```
audit/
  audit.py            CLI
  hipaudit/
    core.py           rows, triage, the durable store, reconciliation
    inventory/        the four extractors
    report.py         computed tables and the dated run
    checks/           the code checks
  inventory/*.json    the standing catalogue; generated; committed so diffs are reviewable
  findings/*.json     what code scanning queued to fix; generated
  triage/*.jsonl      durable decisions for both; re-scanning never touches these
  budget.json         the untriaged ceiling the gate enforces
  runs/<stamp>/       dated REPORT.md + snapshot.json
  tests/              contract tests for durability and validation
```

## Tests

```bash
python3 -m unittest discover -s audit/tests -t audit
```

Hermetic; the store tests run against a temporary directory.

## What this does not establish

The inventory says what is *catalogued*, not what is *true*. A row with no
signals is not thereby healthy. The counts in any report are a floor. Nothing
here replaces reading the code.
