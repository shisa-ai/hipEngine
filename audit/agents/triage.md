# Triage assignment

**Read-only.** You decide what rows are; you do not change code. The only
thing you write is the triage store, through `audit.py triage`. Applying fixes is
[`fix.md`](fix.md).

You triage rows in the hipEngine cleanup audit. Read
[`../README.md`](../README.md) first if you have not this session.

## Two halves

**Inventory rows** (`ledger`, `flag`, `kernel`, `candidate`, `worklog`) are debt somebody
already wrote down. A row is a **mechanically extracted candidate carrying
evidence, not a verdict**. The extractor sees what greps see and is routinely
wrong. Your job is to establish what is true and record a decision.

**Findings** (`axis-branch`, `doc-path-drift`, `unguarded-hip-test`, `stub`,
`ungoverned-flag-branch`, …) come from scanning the code and each names the edit
that closes it. `audit.py queue` groups them by shared cause — one cause often
covers dozens of findings, and fixing the cause is one unit of work.

Both share the triage store: a `wontfix` recorded once is honoured everywhere.

## Loop

0. `python3 audit/audit.py refresh` if the tree has moved since the last scan.
1. `python3 audit/audit.py open <kind> -n 10` for inventory — or `--signal
   "<text>"` to focus a class, `--stale` for decisions whose evidence changed,
   `expiring` for decisions past their review date, `orphans` for decisions whose
   row vanished. For fixable work use `python3 audit/audit.py queue`.
2. `python3 audit/audit.py show <row-id>` for the full evidence.
3. **Investigate against the tree.** Read the files. Follow the call path. Check
   git history for why something is the way it is. The row's signals are leads,
   not conclusions.
4. Record the decision:

```bash
python3 audit/audit.py triage <row-id> --tag <TAG> --do <disposition> \
    --severity high|medium|low --note "what you established, and why this is right"
```

5. `python3 audit/audit.py check` before finishing.

## Rules

- **Never record a decision you did not verify.** If the evidence does not
  settle it, leave the row open and say so in your report. An untriaged row is
  honest; a fabricated disposition is not.
- **`NOT-DEBT` is a first-class outcome.** Extractors produce false positives.
  Recording one is useful work, and the note must say what the extractor
  mistook.
- **The note carries the evidence.** Cite `file:line`, a command and its output,
  or a commit. A note that restates the tag is not a note.
- **Inventory: decide, do not fix.** Triage decides; the fix is its own unit with
  its own tests and commit. The exception is `--resolved` when you confirm the
  work was already done.
- **Queue: fix, then mark resolved.** A finding names its own edit. Fix the
  whole group where one cause covers many findings, run the narrowest relevant
  test, commit, then record `--resolved` against the rows.
- **Use `--expires` on anything conditional.** A `defer` whose reasoning depends
  on a blocker lifting should carry a date, so it returns for re-confirmation
  instead of standing forever.
- **Severity is about consequence, not confidence.** A dead flag nobody reads is
  `low` however certain you are. A production route silently falling back is
  `high`.

## Applying the dispositions

- `promote` — a working implementation that is not on the default path.
  `AGENTS.md` "Product Defaults" says it ships on. Check it actually works first.
- `remove` — the flag, path, or entry is dead. Confirm no reader, including
  tests and harnesses, before saying so.
- `qualify` — plausible but unproven. Name the **exact command** that would
  settle it. If you cannot name one, that is itself a `GATE-CATCH22`.
- `keep` — justified as it stands. The note gives the justification.
- `document` — the code is right and the docs disagree. Name both sides.
- `defer` — real, accepted, unscheduled. The note names what unblocks it.

## The worklog rows

A `worklog` row is an entry whose own text declared work unfinished: an open
`status` (`blocked`, `handoff`, `checkpoint`), a `Next` naming a blocker, gate, or
approval, or a pre-cutoff `### Next` marker. The row's `open_text` is the text
that made it a row, and `show` prints the whole evidence.

Entries are immutable, so a row never changes and never comes back as `stale`.
Decide it once, with the work that finished it named in the note:

Only the `blocked` and `handoff` markers set the budget ceiling; in-flight
`checkpoint` rows and bare `Next` markers stay in the visible queue. A red gate
therefore means the actionable backlog grew — work those first, and let
`budget --lower-only` record the drop.

```bash
python3 audit/audit.py open worklog --signal "recorded as blocked"
python3 audit/audit.py open worklog --signal "no later entry"
```

- **Finished already.** Name the entry, commit, or file that finished it, then
  `--resolved`. This is the common case and the cheapest correct answer.
- **Still open.** Record the real blocker: `qualify` with the exact command,
  `defer` with `--expires`, or `promote` if the work is a working path that was
  never turned on.
- **`NOT-DEBT`.** The status was used loosely, or a "not in the tree" path is one
  the entry quotes rather than points at.

Check the referent against today's tree: a pre-cutoff row describes the tree as
of its date. `no later entry by the same worker shares this topic prefix` means
nobody visibly continued it — the prefix is a hint, not an identity, so confirm
before calling the work dropped.

## What not to do

- Do not raise `audit/budget.json`. It is the ceiling that stops debt growing;
  lowering it is the point. If a kind's gate counts rows that need no decision,
  change that kind's `select` policy and say why — a policy change, not a way
  past a red gate.
- Do not edit `audit/inventory/*.json` or `audit/findings/*.json` by hand. Both
  are generated; re-run `inventory` and `scan`.
- Do not re-triage a row a rescan re-matched (`rebound_from` is set) without
  re-reading it. The text changed; the old conclusion may not survive it.
- Do not treat a campaign document as binding. Most of `docs/campaigns/` is
  closed history — check `status:` in the front-matter.
- Do not reject a candidate for not being bit-exact. `docs/OPTIMIZATION.md` §4.1
  is explicit that exactness alone cannot reject a production-correct candidate.
  When you find one that *was* rejected on those grounds, tag it
  `EXACTNESS-REJECT` with disposition `qualify`, and name in the note the
  production gate it should be measured against instead. Find them with
  `audit.py open candidate --signal "exactness bar"`.

## Report

State how many rows you triaged, the tag and disposition split, which rows you
left open and why, and any extractor weakness you found — a systematic false
positive is worth more than a single row.
