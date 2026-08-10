# Worklog2 entries

This directory is hipEngine's durable cross-session journal. Current history
uses one immutable Markdown file per substantial logical unit so parallel
workers can commit and merge distinct paths instead of appending to one shared
file.

The approved design and migration gates are in
[`docs/PLAN-WORKLOG2-revamp.md`](../docs/PLAN-WORKLOG2-revamp.md).

## Source paths

| Path | Role |
| --- | --- |
| `worklog/entries/*.md` | Tracked immutable current entries. |
| `WORKLOG-LEGACY.md` | Tracked byte-frozen journal through cutoff commit `7c7c18875`. |
| `worklog/legacy-manifest.json` | Hash, byte, line, heading, and cutoff invariant for the frozen journal. |
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
staged/working-tree divergence, unexpected entry-directory paths, or any change
to the frozen legacy journal or manifest.

## Validate

Before committing a logical unit that includes an entry:

```bash
python3 scripts/worklog.py check
```

The optional local pre-commit checker can be installed in a trusted clone with:

```bash
python3 scripts/worklog.py install-hook
```

The installer does not set `core.hooksPath`, does not touch Git LFS
post-checkout/post-commit/post-merge/pre-push hooks, and refuses to overwrite an
unrelated pre-commit hook. Manual validation remains required by `AGENTS.md`;
the local hook is defense in depth.

## Read and render

Render current immutable entries to the ignored local view:

```bash
python3 scripts/worklog.py render
```

Include the complete frozen journal only when older context is needed:

```bash
python3 scripts/worklog.py render --include-legacy
```

Rendering is atomic and never overwrites tracked root `WORKLOG.md`. For routine
handoff, inspect the latest relevant files in `worklog/entries/`; consult
`WORKLOG-LEGACY.md` only for pre-cutoff evidence.

## Historical journal

`WORKLOG-LEGACY.md` is preserved byte-for-byte from cutoff parent
`7c7c188750fcca6ff5ebefa969e7f2689a940172`. Its manifest is checked on every
worklog validation. Do not fix old typos, duplicate headings, dates, or merge
ordering.

Historical Git provenance remains available with:

```bash
git log --follow -- WORKLOG-LEGACY.md
```

## Branches older than Worklog2

A branch created before activation may contain an append to the old
`WORKLOG.md`. When intentionally reviving such a branch, the designated merge
owner extracts its missing material into one or more new immutable entries and
identifies the source branch/commit. Never merge that append into
`WORKLOG-LEGACY.md`; the manifest check fails closed on legacy mutation.

Worklog2 removes only worklog-content contention. Shared-worktree workers still
share one Git index and must serialize staging and commits under `AGENTS.md`.
