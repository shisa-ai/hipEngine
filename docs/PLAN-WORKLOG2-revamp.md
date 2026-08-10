# Worklog2 Revamp Plan

Status: **approved for implementation**

Approved: 2026-08-10

Scope: hipEngine worklog storage, validation, rendering, and agent workflow

Source design: `/home/lhl/tenstorrent-testing` at `dc365575553ae061754aea141f7a3aabd091defd`

This document is the implementation contract for replacing hipEngine's shared,
tracked, append-only `WORKLOG.md` with collision-resistant immutable entry
files. It is intentionally detailed so implementation does not redesign the
system while editing it.

Until the activation commit lands, current `AGENTS.md` rules and the tracked
append-only `WORKLOG.md` remain authoritative. After activation, this plan and
the updated `AGENTS.md` govern new worklog entries. This plan supersedes only
the worklog-storage and worklog-migration portions of
[`PROCESS-IMPROVEMENT.md`](PROCESS-IMPROVEMENT.md); that document's other
proposals remain unapproved unless separately accepted.

## 1. Decision summary

hipEngine will adopt the Tenstorrent repository's one-file-per-logical-unit
worklog model **forward from one explicit cutoff commit**.

The existing journal will not be retroactively converted into thousands of new
entries. It will be frozen byte-for-byte as `WORKLOG-LEGACY.md`, protected by a
tracked hash/size/line-count manifest, and remain available through Git history.
New durable handoffs will live in unique immutable files under
`worklog/entries/`.

The tracked root `WORKLOG.md` will become a small navigation page rather than a
generated aggregate. A local ignored aggregate will be rendered to
`.worklog/WORKLOG.md` on demand. This preserves a useful GitHub landing page and
avoids reintroducing a generated merge hotspot.

## 2. Measured current-state baseline

Measured on `main` at `a8ed426de` before this plan's own journal entry:

| Property | Current value |
| --- | ---: |
| Tracked journal path | `WORKLOG.md` |
| Lines | 217,095 |
| Bytes | 17,664,316 |
| Dated `## YYYY-MM-DD...` sections | 7,244 |
| Distinct dates | 92 |
| Largest entries on one date | 344 |
| Median section length | 25 lines / 2,184 bytes |
| Largest section | 2,105 lines / 147,967 bytes |
| Commits touching `WORKLOG.md` | 7,470 |
| Total commits | 7,707 |
| Current merge policy | `WORKLOG.md merge=union` |

The exact cutoff manifest will be generated from the parent commit of the
activation commit, after this plan and the dormant tooling commit have appended
their final legacy entries. The values above are diagnostic baseline values,
not the future manifest constants.

### Current failure mode

The union merge driver reduces ordinary conflict markers, but it does not
remove the shared path:

- simultaneous same-worktree appends still contend on one 17.7 MB file;
- every lane stages and commits the same path;
- independent branches can produce duplicated, reordered, or garbled tails;
- rebase, stash, rename, and pre-driver histories can still conflict;
- a very large file appears in almost every commit and staged review;
- the journal is too large to serve as a current-state index.

Unique immutable entry paths eliminate the shared-content merge target rather
than attempting to merge it more cleverly.

## 3. Goals

1. Prevent routine worklog conflicts across lane branches and linked worktrees.
2. Prevent simultaneous workers from writing the same journal file.
3. Preserve the existing journal exactly, including imperfect historical text.
4. Preserve a stable tracked GitHub landing page at `WORKLOG.md`.
5. Keep entries human-readable Markdown and searchable with ordinary tools.
6. Bind every new entry to UTC time, worker/lane, branch, worktree, and base
   commit.
7. Enforce immutable committed entries and correction-by-new-entry semantics.
8. Preserve hipEngine's exact benchmark, correctness, artifact, rollup, and
   commit policies; the worklog remains a breadcrumb, not the only evidence.
9. Preserve existing Git LFS hook behavior.
10. Keep the mechanism standard-library-only and usable without project
    dependency installation.

## 4. Non-goals

1. Do not reconstruct or invent historical timestamps, workers, branches,
   worktrees, base commits, or statuses.
2. Do not create approximately 7,244 imported entry files.
3. Do not reorganize historical entries by model, backend, campaign, or topic.
4. Do not turn the worklog into the active sprint/task registry.
5. Do not replace compact benchmark artifacts, benchmark rollups, changelogs,
   current domain dashboards, or architectural decision documents.
6. Do not add a database, service, package dependency, or Git LFS object.
7. Do not automatically execute expensive full-legacy rendering after every
   commit or checkout.
8. Do not modify model, runtime, kernel, benchmark, or GPU behavior.
9. Do not bulk-rewrite every historical plain-text reference to `WORKLOG.md`.

## 5. Approved repository layout

```text
WORKLOG.md                         # tracked, small navigation page
WORKLOG-LEGACY.md                  # tracked, frozen pre-cutoff journal
worklog/
  README.md                        # format, commands, correction rules
  legacy-manifest.json             # cutoff provenance and byte invariant
  entries/
    <unique immutable entry>.md
scripts/
  worklog.py                       # create/check/render/install-hook CLI
tests/
  test_worklog.py                  # retained repository tests
.worklog/
  WORKLOG.md                       # ignored local generated view
```

The implementation will use the repository's existing `tests/` directory, so
the actual test path is `tests/test_worklog.py`.

### Root paths

- `WORKLOG.md` remains tracked so existing GitHub links resolve. It links to the
  legacy file, immutable entries, format documentation, and local render
  command. It is not edited for each logical unit.
- `WORKLOG-LEGACY.md` is the exact old `WORKLOG.md` blob at the cutoff. It is
  never edited after activation.
- `worklog/entries/` is the only tracked location for new chronological
  handoffs.
- `.worklog/` is ignored local output and never staged.

## 6. New entry contract

### 6.1 Filename

Each entry filename is:

```text
YYYYMMDDTHHMMSS.ffffffZ-<worker-slug>-<topic-slug>-<random-hex6>.md
```

Rules:

- timestamp is UTC with six fractional digits;
- worker and topic are lowercase ASCII kebab-case in the filename;
- worker and topic slugs are independently bounded to 48 characters;
- the random suffix contains 24 random bits;
- creation uses exclusive `O_CREAT|O_EXCL` and retries collisions;
- different branch entries should merge as independent file additions;
- a same-path add/add collision is treated as a real conflict, never silently
  union-merged.

### 6.2 Versioned frontmatter

Schema 1 frontmatter has exactly this order:

```yaml
---
schema: 1
timestamp: 2026-08-10T12:34:56.123456Z
worker: <stable worker or lane id>
branch: <branch or detached-SHA>
worktree: <worktree directory name>
base_commit: <full 40-character lowercase SHA-1>
status: completed
topic: <lowercase-kebab-topic>
---
```

Required fields:

| Field | Meaning |
| --- | --- |
| `schema` | Entry schema version. Version 1 is the initial accepted schema. |
| `timestamp` | Entry creation time in UTC, not a guessed benchmark time. |
| `worker` | Stable worker, agent, or lane identifier supplied explicitly or derived from `WORKLOG_WORKER`, Git user name, or OS user. |
| `branch` | Current branch, or `detached-<short-sha>`. |
| `worktree` | Current repository worktree directory name. |
| `base_commit` | `HEAD` when the entry template was created. The containing final commit remains discoverable through Git history. |
| `status` | Logical-unit status from the fixed vocabulary below. |
| `topic` | Stable lowercase kebab-case subject used in the filename. |

Allowed schema-1 statuses:

- `completed`
- `checkpoint`
- `decision`
- `blocked`
- `handoff`

Benchmark evidence classes such as `accepted`, `diagnostic`,
`rejected_correctness`, and `blocked` remain in artifacts and prose. They do not
replace the logical-unit status field. Where useful, put `LANDED`, `REJECTED`,
`NEUTRAL`, `BLOCKED`, `DECISION`, `CORRECTION`, or `PROCESS` in the human title.

### 6.3 Body

Every entry has one level-one title and exactly one of each required section, in
this order:

```markdown
# Short outcome

## Summary

...

## Changes

- ...

## Validation

- ...

## Next

- ...
```

No required section may be empty. Template placeholders and Git conflict
markers are validation failures.

### 6.4 Entry granularity

Create one entry for a substantial logical unit, retained or rejected benchmark,
non-trivial decision, blocker, process change, or cross-session handoff. Do not
create one for every shell command or trivial typo.

The entry is committed with the code, tests, docs, and compact artifacts it
describes. Exact performance evidence still follows `docs/BENCHMARK.md` and
updates the required artifact/rollup/changelog surfaces.

An entry may be edited freely before its first commit. After commit it is
immutable.

## 7. Immutability and correction policy

`python3 scripts/worklog.py check` must reject:

- modification of a committed entry;
- deletion of a committed entry;
- rename of a committed entry;
- malformed or unknown schema/frontmatter;
- filename/frontmatter mismatch;
- duplicate/missing/out-of-order required sections;
- unfinished placeholders;
- conflict markers;
- a staged entry whose working-tree copy differs;
- unexpected files directly under `worklog/entries/`;
- any change to the frozen legacy file or its manifest invariant.

Only newly added entry paths are permitted under `worklog/entries/`.

A wrong committed entry is corrected by a new `decision` or `checkpoint` entry
that names the superseded conclusion and points to the original path. Historical
entries are not silently repaired.

## 8. Legacy preservation contract

### 8.1 Cutoff

The cutoff is the parent commit of the activation commit. At that parent:

1. current `WORKLOG.md` still contains the final append-only legacy history;
2. all prior logical units are committed;
3. the implementation computes its exact hash, byte count, line count, first
   heading, and last heading;
4. the activation commit renames it to `WORKLOG-LEGACY.md` without changing its
   bytes.

### 8.2 Manifest

`worklog/legacy-manifest.json` records at least:

```json
{
  "schema": 1,
  "path": "WORKLOG-LEGACY.md",
  "cutoff_commit": "<full parent SHA>",
  "sha256": "<64 lowercase hex>",
  "bytes": 0,
  "lines": 0,
  "first_heading": "## ...",
  "last_heading": "## ..."
}
```

The checker validates the file against the manifest on every worklog check.
Deleting the manifest while `WORKLOG-LEGACY.md` exists is a failure. Deleting or
changing the legacy file is a failure.

The migration validation records the pre-rename and post-rename SHA-256 and
requires exact equality. No historical typo, duplicate heading, malformed date,
or old conflict-resolution order is corrected during migration.

### 8.3 Git history

Because activation preserves the old blob at a new path while recreating root
`WORKLOG.md` in the same atomic commit, Git may report a **100% copy plus root
rewrite** rather than a rename. Review the staged diff with break/copy detection
(`git diff --staged -B50% -M50%`) and require an exact zero-line legacy copy.
Historical provenance remains available with:

```bash
git log --follow -- WORKLOG-LEGACY.md
```

A future optional legacy index may list heading, date, line/byte offset, and
section hash. It must be generated mechanically from the frozen file and is not
part of this initial migration.

## 9. Rendering and reading

### 9.1 Default render

```bash
python3 scripts/worklog.py render
```

atomically writes `.worklog/WORKLOG.md` from immutable **new** entries in
(timestamp, filename) order. The generated header links to
`WORKLOG-LEGACY.md`. The default render does not copy 17.7 MB of legacy text on
every invocation.

### 9.2 Full local render

```bash
python3 scripts/worklog.py render --include-legacy
```

may prepend the exact legacy journal before new entries for occasional complete
local review. It writes only to the ignored local output. It never overwrites the
tracked root navigation page.

### 9.3 Atomicity

Rendering uses a temporary file in the destination directory and `os.replace`.
Interrupted rendering must leave either the old complete output or the new
complete output, never a partial tracked or ignored file.

### 9.4 Normal reading workflow

At task start, agents read:

1. relevant architecture/domain/benchmark docs;
2. the latest relevant files under `worklog/entries/` or the rendered local
   view;
3. the tail of `WORKLOG-LEGACY.md` only when pre-cutoff context is relevant.

The worklog remains chronological evidence, not the current task registry or
canonical benchmark scoreboard.

## 10. Hook and Git LFS safety

hipEngine currently has Git LFS hooks in `.git/hooks/post-checkout`,
`post-commit`, `post-merge`, and `pre-push`. Worklog2 must not bypass, replace,
or duplicate them.

### Approved hook design

- Do **not** set `core.hooksPath`.
- Do **not** install post-checkout, post-commit, post-merge, or pre-push hooks.
- Do **not** render automatically after every Git operation.
- Provide `python3 scripts/worklog.py install-hook` for the pre-commit check
  only.
- Resolve the common Git hooks directory using Git, so linked worktrees share
  the installation.
- Install only when `pre-commit` is absent or byte-identical to the managed
  Worklog2 hook.
- Refuse to overwrite any unrelated existing pre-commit hook unless a future
  separately reviewed composition mechanism is approved. No `--force` in
  schema 1.
- The managed hook resolves the current worktree root and runs the checker only
  when `scripts/worklog.py` exists. Thus checking out an older branch without
  Worklog2 does not block commits.
- Warn that a hook executes code from the checked-out trusted repository.

Manual `python3 scripts/worklog.py check` remains mandatory before commits that
add worklog entries even when the hook is installed. CI or repository tests
must not rely solely on a developer-local hook.

## 11. Merge and old-branch transition

### 11.1 New branches

Branches created after activation add different files under `worklog/entries/`.
Normal Git merges should require no worklog content resolution. The shared index
still requires serialized staging/commit ownership; Worklog2 does not change
that Git constraint.

### 11.2 Old branches

A branch based before activation may still append to the old tracked
`WORKLOG.md`. It must not silently mutate `WORKLOG-LEGACY.md` during a later
merge.

For any such branch intentionally revived:

1. identify its merge base and exact old-worklog delta;
2. determine which sections are not already represented on current `main`;
3. create one or more new immutable entries that preserve the material
   decisions/evidence and identify the source branch/commit;
4. do not alter the frozen legacy file;
5. let the manifest check fail closed if a merge attempts to change legacy;
6. designate one merge owner for this conversion and staged commit.

Archived branches are not bulk-imported merely because they exist. The current
Maple and Moonshine feature tips are already ancestors of `main`; stale archival
branches need conversion only if intentionally revived.

### 11.3 Merge attributes

Activation removes `WORKLOG.md merge=union`. The frozen legacy path should be
marked non-union/fail-closed in `.gitattributes`; immutable entry files use
normal Git merge behavior.

The old `scripts/resolve_worklog_conflict.py` becomes obsolete and will be
removed in the activation commit. Its historical source remains available from
Git history.

## 12. Documentation migration

The activation unit updates:

- `AGENTS.md` / symlinked `CLAUDE.md` worklog source-of-truth, start/during/end,
  commit, coordination, and blocker rules;
- `docs/README.md` repository map;
- `docs/PROCESS-IMPROVEMENT.md` with a narrow supersession note;
- `.gitignore` for `/.worklog/`;
- `.gitattributes` to retire union merge behavior;
- root `WORKLOG.md` as the tracked navigation page;
- `worklog/README.md` as the user/agent command and format guide.

Do not bulk-rewrite historical prose that merely says `WORKLOG.md`. The tracked
root navigation page keeps those references useful. Update direct process
instructions that tell workers to append or stage the monolith.

## 13. Tooling implementation contract

`scripts/worklog.py` remains Python-standard-library-only and provides:

```text
new          create one unique entry template
check        validate schema, immutability, staged/worktree agreement, and legacy invariant
render       atomically render ignored current or full local view
install-hook install the non-destructive pre-commit-only checker
```

Required implementation properties:

- all Git commands run from the resolved repository root;
- errors are explicit and return non-zero;
- root and output paths are derived from the script location, not caller CWD;
- the parser is deterministic and does not use a YAML dependency;
- schema 1 rejects unknown fields rather than silently dropping them;
- renderer ordering is deterministic under equal timestamps;
- `new` does not stage or commit automatically;
- `check --allow-modified` may validate format in isolated tooling tests, but
  ordinary checks enforce immutability;
- the dormant tooling commit may run with zero new entries and no legacy file;
- once `WORKLOG-LEGACY.md` exists, a valid manifest is mandatory.

## 14. Retained test plan

Add `tests/test_worklog.py`. Tests use temporary initialized Git repositories so
behavior is validated mechanically without touching hipEngine history.

### Unit/format cases

- valid schema-1 entry parses;
- missing, extra, duplicate, or out-of-order frontmatter fails;
- invalid UTC timestamp, status, topic, base commit, or filename fails;
- missing, duplicate, empty, or reordered sections fail;
- placeholders and conflict markers fail;
- equal timestamps sort by filename;
- atomic render writes the expected source comments and ordering;
- default render excludes legacy body and full render includes it exactly.

### Git integration cases

- `new` creates a unique exclusively owned file;
- two rapid `new` calls never share a path;
- committed entry modification fails;
- committed entry deletion fails;
- committed entry rename fails;
- staged/working-tree divergence fails;
- newly added valid entry passes;
- independently created branch entries merge without content conflict;
- legacy hash/size/line mismatch fails;
- missing legacy manifest fails after legacy activation;
- root tracked `WORKLOG.md` is never overwritten by render;
- hook installation leaves existing Git LFS hooks byte-identical;
- hook installation refuses an unrelated existing pre-commit hook;
- installed managed pre-commit validates entries and tolerates an old branch
  where the tool is absent.

The narrow validation command is:

```bash
python3 -m pytest -q tests/test_worklog.py
```

Process validation also includes:

```bash
python3 -m py_compile scripts/worklog.py tests/test_worklog.py
python3 scripts/worklog.py check
python3 scripts/worklog.py render
python3 scripts/worklog.py render --include-legacy
```

No GPU or broad runtime suite is required because the migration changes no
runtime code. Run `git diff --check` and review the complete staged diff.

## 15. Implementation phases and commit boundaries

### P0 — Approve and freeze this plan

Outcome: one docs-only commit containing this plan and one final entry in the
current append-only journal.

- [x] Review Tenstorrent design and live tooling.
- [x] Measure hipEngine journal size, section count, and commit frequency.
- [x] Select forward-only immutable shards.
- [x] Select frozen legacy plus tracked root navigation page.
- [x] Reject full retroactive port.
- [x] Select schema versioning and fixed entry contract.
- [x] Select manual render plus pre-commit-only hook installation.
- [x] Preserve Git LFS hooks and avoid `core.hooksPath`.
- [x] Create and commit this approved plan with the current journal entry.

### P1 — Land dormant tooling and retained tests

Outcome: one validated tooling/test commit. The old append-only journal remains
authoritative during this phase.

- [x] Write RED format, immutability, merge, legacy, render, and hook tests.
- [x] Port and adapt `scripts/worklog.py` from Tenstorrent source.
- [x] Add `schema: 1` and legacy-manifest validation.
- [x] Change render output to `.worklog/WORKLOG.md`.
- [x] Add optional `--include-legacy`.
- [x] Implement non-destructive `install-hook` without `core.hooksPath`.
- [x] Confirm all existing Git LFS hook hashes are unchanged.
- [x] Run focused tests, compile checks, and diff checks.
- [x] Append one final current-format legacy journal entry describing the dormant
  tooling.
- [x] Commit the tooling and tests without activating Worklog2 policy.

### P2 — Activate Worklog2 and freeze legacy

Outcome: one atomic process migration commit.

- [x] Re-read live `WORKLOG.md` tail and verify no concurrent writer owns it.
- [x] Record cutoff parent commit.
- [x] Compute pre-rename SHA-256, bytes, lines, first heading, and last heading.
- [x] Rename `WORKLOG.md` to `WORKLOG-LEGACY.md` without byte changes.
- [x] Create and validate `worklog/legacy-manifest.json`.
- [x] Create tracked root `WORKLOG.md` navigation page.
- [x] Create `worklog/README.md`.
- [x] Create the first immutable migration entry under `worklog/entries/`.
- [x] Update `AGENTS.md`, `docs/README.md`, and the process supersession note.
- [x] Add `/.worklog/` to `.gitignore`.
- [x] Remove union merge configuration and fail-close the legacy path.
- [x] Remove obsolete `scripts/resolve_worklog_conflict.py`.
- [x] Render default and full ignored views.
- [x] Prove ignored output is untracked and root navigation is unchanged.
- [x] Run focused tooling tests and worklog checks.
- [x] Verify post-rename legacy SHA/bytes/lines equal pre-rename values.
- [x] Review status, staged names, complete staged diff, and 100% legacy copy/rewrite detection.
- [x] Commit activation as one atomic process unit.

### P3 — Local hook activation and post-commit smoke

Outcome: local enforcement active without changing tracked history.

- [ ] Run `python3 scripts/worklog.py install-hook` after the activation commit.
- [ ] Verify `core.hooksPath` remains unset.
- [ ] Verify Git LFS post-checkout/post-commit/post-merge/pre-push hooks are
  byte-identical to their pre-migration hashes.
- [ ] Create a temporary valid entry, confirm pre-commit acceptance, and remove
  the uncommitted temporary file.
- [ ] Create a temporary malformed entry, confirm pre-commit/check rejection,
  and remove the uncommitted temporary file.
- [ ] Confirm `git status -sb` is clean and `main` is synchronized as intended.

### P4 — Optional later follow-up, not part of activation

- [ ] Add a mechanically generated legacy heading/offset/hash index only if
  findability remains poor.
- [ ] Consider CI wiring after observing the local workflow; do not add a new
  release blocker without evidence.
- [ ] Revisit status/frontmatter fields only through `schema: 2`; schema 1 stays
  valid forever.
- [ ] Remove no compatibility surface merely to imitate Tenstorrent's exact
  implementation.

## 16. Activation acceptance gate

Worklog2 is active only when all conditions are true:

1. plan, tooling/tests, and activation commits are separate reviewed logical
   units;
2. legacy pre/post SHA-256, byte count, and line count match exactly;
3. legacy checker rejects one controlled mutation in a temporary repository;
4. all new entry schema and immutable-entry tests pass;
5. independent temporary branch entries merge without a worklog conflict;
6. default and full local rendering are deterministic and atomic;
7. root `WORKLOG.md` remains tracked and is not changed by rendering;
8. `.worklog/` is ignored;
9. no `WORKLOG.md merge=union` rule remains;
10. existing Git LFS hooks are unchanged;
11. `core.hooksPath` remains unset;
12. updated agent instructions no longer direct workers to append/stage the
    monolith;
13. the first immutable migration entry is committed with activation;
14. focused tests, Python compilation, worklog check, and diff checks pass;
15. no runtime/GPU files or benchmark claims change.

If any gate fails, do not partially claim activation. Repair within the scoped
migration before committing, or restore the pre-activation working state only
for files owned by this migration.

## 17. Rollback and recovery

Before activation is committed, ordinary scoped edits may be corrected while
preserving unrelated work. After activation is committed:

- do not rewrite committed immutable entries to roll back policy;
- revert the activation commit only as an explicit coordinated process decision;
- preserve any post-activation entries by importing them deliberately if an
  alternate system replaces Worklog2;
- never regenerate legacy from parsed sections when the original frozen blob is
  available;
- recover an accidentally removed local `.worklog/WORKLOG.md` by rendering;
- recover a malformed uncommitted entry by editing or deleting only that
  uncommitted entry;
- correct a malformed committed conclusion with a new immutable correction
  entry.

## 18. Evidence and provenance

The design review used:

- `/home/lhl/tenstorrent-testing/AGENTS.md`;
- `/home/lhl/tenstorrent-testing/worklog/README.md`;
- `/home/lhl/tenstorrent-testing/scripts/worklog.py`;
- `/home/lhl/tenstorrent-testing/.githooks/*`;
- setup commit `dc365575553ae061754aea141f7a3aabd091defd`;
- 305 valid live Tenstorrent entries rendered successfully during review;
- hipEngine `.gitattributes`, `AGENTS.md`,
  `scripts/resolve_worklog_conflict.py`, `docs/PROCESS-IMPROVEMENT.md`, current
  Git LFS hooks, worktrees/branches, and measured journal structure.

The Tenstorrent checkout did not have `core.hooksPath` configured and had no
local generated root `WORKLOG.md` at review time. That does not invalidate its
immutable-shard design, but it is why hipEngine keeps an always-present tracked
navigation page and treats hooks as optional local enforcement rather than the
only correctness mechanism.
