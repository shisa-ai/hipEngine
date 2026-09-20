# Worklog2 entries

This directory is hipEngine's durable cross-session journal. Current history
uses one immutable Markdown file per substantial logical unit so parallel
workers can commit and merge distinct paths instead of appending to one shared
file.

The approved design and migration gates are in
[`docs/archive/PLAN-WORKLOG2-revamp.md`](../docs/archive/PLAN-WORKLOG2-revamp.md).

## Source paths

| Path | Role |
| --- | --- |
| `worklog/entries/*.md` | Tracked immutable current entries. |
| `worklog/legacy-port-manifest.json` | Entry-to-commit mapping for the ported pre-cutoff entries; pins the retired journal's sha256 and Git history ref. |
| `WORKLOG.md` | Tracked GitHub/navigation page; never generated or appended per unit. |
| `.worklog/WORKLOG.md` | Ignored local generated chronological view. |
| `scripts/worklog.py` | Standard-library create/check/render/hook CLI. |

## Create an entry

For a substantial implementation, benchmark, decision, blocker, process change,
or handoff:

```bash
python3 scripts/worklog.py new \
  --worker <stable-worker-or-lane-id> \
  --status completed \
  --title "Short outcome"
```

Optional `--topic` supplies the lowercase filename/frontmatter topic. Without
it, the title is slugged. `WORKLOG_WORKER`, Git `user.name`, and the OS user are
the fallback worker sources, in that order.

The command prints a collision-resistant path such as:

```text
worklog/entries/20260810T123456.123456Z-perf-lane-retain-kernel-a1b2c3.md
```

Fill every placeholder while keeping the fixed section order:

1. `## Summary`
2. `## Changes`
3. `## Validation`
4. `## Next`

Allowed statuses are `completed`, `checkpoint`, `decision`, `blocked`, and
`handoff`. Benchmark evidence classes remain in compact artifacts and prose;
they are not substitutes for the logical-unit status.

Use one entry per logical unit and commit it with the code, tests, docs, and
compact artifacts it describes. Do not create entries for every trivial command
or typo. Exact performance claims still follow `docs/BENCHMARK.md` and update
the required rollup and changelog.

## Immutable after commit

A new entry may be edited until its first commit. After commit, do not modify,
rename, or delete it. Correct an old conclusion with a new `decision` or
`checkpoint` entry that names and links the superseded entry.

Validation fails on malformed schema/frontmatter, filename mismatch, missing or
reordered sections, placeholders, conflict markers, committed entry changes,
staged/working-tree divergence, or unexpected tracked paths under
`worklog/entries/`. It gates staged and tracked content only; see "Validate"
below.

## Validate

`check` validates the **commit tree** — what Git tracks or stages — not the
working directory. Stage the entry first:

```bash
git add worklog/entries/<entry>.md
python3 scripts/worklog.py check
```

An untracked working-tree file is not part of the commit. `check` reports it as
a note and never fails on it, so another worker's unfinished entry in a shared
worktree cannot force `git commit --no-verify`. Validate your own work in
progress before staging it with:

```bash
python3 scripts/worklog.py check --include-unstaged
```

The check also rejects any change to an already-tracked entry, and it rejects a
staged entry whose working-tree copy has moved on (re-run `git add` to commit
the final text). Staged deletion of the retired legacy journal pair
(`WORKLOG-LEGACY.md` and `worklog/legacy-manifest.json`) is the one sanctioned
removal: it is allowed once, and the exact journal bytes stay pinned by the
history ref in `worklog/legacy-port-manifest.json`.

The optional local pre-commit checker can be installed in a trusted clone with:

```bash
python3 scripts/worklog.py install-hook
```

The installer does not set `core.hooksPath`, does not touch Git LFS
post-checkout/post-commit/post-merge/pre-push hooks, and refuses to overwrite an
unrelated pre-commit hook. The installed hook runs the same commit-tree check.
Manual validation remains required by `AGENTS.md`; the local hook is defense in
depth.

## Read and render

Render current immutable entries to the ignored local view:

```bash
python3 scripts/worklog.py render
```

`--include-legacy` is inert now that the legacy journal is retired from the
tree; rendering prints a note with the `git show` ref for the original bytes.

Rendering is atomic and never overwrites tracked root `WORKLOG.md`. The rendered
view also includes unstaged entries from the working tree, so local work in
progress shows up before it is committed; an entry that is not valid yet is
skipped with a note on stderr. For routine handoff, inspect the latest relevant
files in `worklog/entries/`. Pre-cutoff history is readable as ported entries
(front-matter `worker: legacy`).

## Retired legacy journal

The pre-Worklog2 append-only journal was preserved byte-for-byte as
`WORKLOG-LEGACY.md` at cutoff parent `7c7c188750fcca6ff5ebefa969e7f2689a940172`,
ported verbatim into entries (see below), and then removed from the tree once
the port was verified byte-for-byte. The exact original bytes remain in Git
history, pinned by the sha256 and history ref recorded in
`worklog/legacy-port-manifest.json`:

```bash
git show <source_history.commit>:WORKLOG-LEGACY.md
```

Historical Git provenance remains available with:

```bash
git log --follow -- WORKLOG-LEGACY.md
```

Do not resurrect the file into the tree; ported entries are the in-tree
representation of pre-cutoff history.

## Ported pre-cutoff entries

All 7,232 pre-cutoff entries from the frozen journal are also imported as
individual Worklog2 entries under `worklog/entries/` with front-matter
`worker: legacy`, `branch`/`worktree: pre-cutoff`, and `status: completed`.
Each ported entry's filename timestamp and `base_commit` come from the first
Git commit that appended it to the journal, recovered from the full patch
history of `WORKLOG.md`/`WORKLOG-LEGACY.md`; repeated heading texts bind their
file occurrences to append events by author date.

Bodies are verbatim. Entries whose bodies contain lines that would break the
one-title schema rule (for example pasted `# ` shell-output comments) carry
the verbatim body inside a fenced block one backtick run longer than any fence
in the body; the manifest flags these as `wrapped`. Timestamps are commit
author dates in UTC; file order and commit order disagree around old union
merges, which the manifest records as timestamp inversions.

`worklog/legacy-port-manifest.json` records the full mapping: source line
ranges, attributed commits and subjects, timestamps, wrapped flags, duplicate
groups, the anomaly lists (append-replay re-adds and in-place edit commits),
and the retired journal's sha256 and Git history ref. Re-run the byte-exact
verification with:

```bash
python3 scripts/worklog_port_legacy.py verify
```

`verify` re-derives every ported entry from the original journal bytes — read
from the working tree while the journal exists there, or from the recorded Git
history ref afterwards — and fails closed if either side drifts, so a history
rewrite cannot silently degrade the port.

## Branches older than Worklog2

A branch created before activation may contain an append to the old
`WORKLOG.md`. When intentionally reviving such a branch, the designated merge
owner extracts its missing material into one or more new immutable entries and
identifies the source branch/commit. Never merge that append into a resurrected
legacy file; convert it to entries instead.

Worklog2 removes only worklog-content contention. Shared-worktree workers still
share one Git index and must serialize staging and commits under `AGENTS.md`.
