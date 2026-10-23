---
status: current
owns: One-time Git history cleanup scope, artifact preservation, rewrite validation and clone migration.
---
# One-time repository history cleanup

## Objective and authorization

Remove disposable traces, logs and generated caches from every affected branch
and tag in one coordinated history rewrite. The lead approved rewriting published
tags on October 23, 2026, then required this broader analysis before execution.
No remote rewrite has happened. The seven-file trial mirror is not the release
candidate for this campaign.

The inventory baseline is `2ec101ae91a298a4baf9aaebceac2aac50fe961a`.
The candidate manifest is `CLEANUP-HISTORY-MANIFEST.json` beside this document.
Its actions are proposals with prerequisites, not permission to delete every file
matching an extension. Compact evidence, fixtures and runtime assets stay.

## Findings

All locally reachable Git objects were inventoried by uncompressed size and
current on-disk object size. Every remote head/tag object in the captured remote
ref listing exists locally. The main-tree scan covers tracked files only;
untracked scratch files are outside this campaign.

| Group | Scope | Decision |
| --- | --- | --- |
| Seven July 2 llama.cpp tensor-trace JSONL files | 310,673,218 bytes in main | Purge; keep seven compact JSON siblings and reduced comparisons. |
| Eight September 11 profiler CSVs | Raw kernel-dispatch tables, including one duplicate | Purge after preserving the two missing compact kernel censuses. Six already have census companions. |
| Five duplicate/profiler console logs | Two JSON payloads duplicated in logs, two profiler driver logs and an empty compile log | Purge; structured companions remain. |
| 29 intermediate tensor JSON files | 108,322,216 bytes in main | Compact historical versions and the current files; retain metadata, summaries, hashes, token decisions and comparisons, remove only raw `values` fields under the named boundary captures. |
| Historical TP2 uneven-split smoke artifact | 60,138,000 bytes approximately, on another branch | Compact eight full-vocabulary teacher-logit arrays; keep configuration, output tokens, comparison metrics and pass/fail result. Exact inventory size is in the manifest. |
| Historical `audit/inventory/`, `audit/findings/`, `audit/runs/` | Generated caches/reports, already ignored and absent from main | Purge historical versions; retain all triage decisions, budgets, extractors and source records. |
| Benchmark server logs | All 121 tracked logs reviewed | Purge 79 with paired/duplicate evidence and five after summary extraction; retain 37 cited, hash-pinned or compact logs. |

The first three groups total **20 files / 401,767,386 bytes** (383.2 MiB).
The 29 compacted JSON files retain **3,179,060 bytes**, saving another
**105,143,156 bytes** (100.3 MiB). These are checkout savings, not clone savings.
The seven largest JSONL blobs currently occupy about **30.3 MB** in Git's pack
rather than their 310.7 MB uncompressed size. Delta sharing and repacking prevent
adding current object sizes to predict final clone size precisely; measure a
fresh rewritten clone against a freshly packed unmodified clone.

The first classifier report overstated the 29-file compacted size by including
six additional files. The figures above were recomputed for the exact 29-file
manifest. Those six extra trial outputs are not approved replacements.

The final removal list contains **99 files / 412,199,350 bytes**, including
84 logs and the 15 raw JSONL/CSV traces. Five logs already belonged to the initial
20-file group and are counted only once. The 29 tensor compactions save another
105,143,156 bytes: **517,342,506 bytes (493.4 MiB)** of main-tree savings before
small preservation summaries and documentation. Historical cache removal and the
branch-only TP2 compaction are additional.

The log review retains four hash-pinned server logs, 30 cited logs and three
compact logs. Five otherwise disposable logs contain unique measurements; these
are preserved in `CLEANUP-HISTORY-LOG-SUMMARIES.json`. Other removed server logs
may contain unclaimed warmup or per-request prefill diagnostics beyond the paired
JSON measurements; the offline recovery archive preserves those original bytes.
Do not describe every removed log as byte-for-byte duplicate evidence.

## Preserve explicitly

- `tests/fixtures/**`, including NPZ arrays, images and numerical oracles.
  Historical fixture blobs account for about 166 MB of stored Git objects;
  size alone is not a reason to remove a test oracle.
- All required kernel sources, bundled runtime assets and Git LFS objects.
- Compact benchmark summaries, per-request measurement rows, comparison JSON,
  benchmark prompts and canonical fixtures.
- Nine named replay inputs in the manifest. Several large `mtp-gguf-iter*`
  captures are defaults for comparison scripts; stripping their arrays would
  break those tools. The small July 2 non-tensor token trace is also retained.
- Immutable worklog entries, the retired journal bytes and audit triage history.
  Do not mistake those records for generated audit caches.
- Historical provenance fields such as artifact source commits and old file
  paths. They describe the original run. Preserve them and supply a commit map
  and removal manifest rather than rewriting them into fictitious new evidence.
- The main checkout's untracked scratch probes/log directory, local branches,
  stash and other worktrees. These are user-owned work, not cleanup inputs.

## Required preparation

1. Use the reviewed exact paths and per-file transforms in the manifest.
   Avoid broad `*.json`, `*.jsonl`, `*.csv` or `*.log` deletion rules.
2. Independently verify the two regenerated profiler censuses against their CSV
   sources and the study's published numbers. Preserve them before purging CSVs.
3. Verify each tensor compaction retains every non-payload value unchanged.
   Record original blob IDs and payload hashes. Mark compacted artifacts so a
   comparison tool cannot mistake a summary for a raw replay input. Historical
   raw-vector re-comparison then requires the offline archive or a fresh capture.
4. Update the live cleanup ledger and study links. Historical worklog commands
   remain historical facts; do not edit immutable entries.
5. Back up every original ref and object outside the repository being rewritten,
   along with remote-ref IDs, worktree locations, stash and untracked-file
   inventories. Validate the recovery bundle and keep it offline, not reachable
   from rewritten remote refs. A backup ref on the remote would retain the blobs.
6. Fetch and inventory every remote branch/tag again immediately before the
   rewrite. Refuse to overwrite a ref that moved since the recorded snapshot.

## Rewrite and migration contracts

Build a new disposable mirror from the original repository, not the seven-file
trial. Apply the final path removals and narrowly scoped JSON transformations in
one run. Save complete commit/ref maps outside the rewritten history.

Published release tags change their target commits by explicit lead approval.
PyPI artifacts and their attestations remain immutable and continue naming the
original source commits. Explain that distinction in the migration record; do not
republish the same package version or claim that the attestation names the new
commit. Annotated tag messages should remain; any existing signatures require
explicit handling because a rewrite invalidates them.

`worklog/legacy-port-manifest.json` has an operational `source_history.commit`:
`scripts/worklog_port_legacy.py` reads that commit's journal and checks its SHA-256.
Remap that lookup to the rewritten commit while preserving the exact journal
bytes and expected hash. Ordinary worklog `base_commit` fields are historical
facts, not live object-existence checks. Preserve their recorded values and use
the map for lookup. External upstream commit IDs in lineage records must not be
translated through this repository's commit map.

Do not force-reset the shared working tree or its other worktrees. Preserve the
old checkout as a recovery workspace and prepare a clean rewritten checkout;
migrate local changes by reviewed patches or cherry-picks with the commit map.
Old clones must not merge old ancestry back into main.

Publish only intended remote branches and tags, with explicit expected-old-ID
leases and an atomic push when supported. Never mirror-push local stash,
remote-tracking refs, private branches or backup refs. GitHub may retain old
objects through pull-request refs or caches after the branch/tag rewrite; remote
storage reclamation is not proven solely by a successful push.

## Completion gates

- Enumerate every targeted original blob/path and verify no disallowed payload
  remains reachable from the intended rewritten refs, including renamed paths.
- Compare old and new trees for each remote ref. Differences must be confined to
  manifest removals/transforms and the declared migration documentation/pointer
  updates. Fixtures and runtime asset identities must remain unchanged.
- Validate compacted JSON, profiler census values, retained references, lineage,
  worklog checks including legacy-port verification, documentation, audit budget
  and artifact provenance. Run tests for any tooling changes.
- Measure fresh packed Git size and checkout size before and after with the same
  packing settings. Record LFS size separately from Git objects.
- Inspect dry-run ref updates, publish once, then compare remote IDs with the
  expected rewritten map and verify a fresh clone. Preserve backups until the
  lead has migrated the remaining working copies.

## Status

The isolated rewrite is validated and ready for publication. All 64 original
branch/tag tip trees differ only by the declared removals, raw-payload
compactions and operational journal-pointer repair. The rewritten journal
regenerates all 7,232 legacy entries byte-for-byte. Documentation, audit budget,
artifact provenance and focused worklog/lineage tests pass.

The same 64 refs packed with `git repack -Ad` occupy 403.28 MiB before cleanup
and 350.42 MiB after cleanup, excluding unreachable loose backup objects. This
comparison precedes the preservation/migration commit. The main-tree gross
reduction is approximately 493 MiB; the two censuses and compressed commit map
add less than 1 MiB. LFS assets are unchanged.

## Migrating existing clones

Use a fresh clone instead of pulling or merging the rewritten history into an
old checkout. Keep the old checkout until all uncommitted work and private
branches have been transferred. Do not merge an old branch wholesale: that would
make the removed history reachable again. Review and cherry-pick the necessary
commits, or apply patches containing only the desired changes.

The original-to-rewritten map is
`worklog/history-rewrites/20261023-commit-map.tsv.gz`. Read it with
`gzip -dc worklog/history-rewrites/20261023-commit-map.tsv.gz`. A zero target means
the original commit became empty and was removed. The map records rewritten
history before the final preservation commit; that new commit has no original
counterpart. Historical evidence still records its original commit IDs.

On the cleanup host, the verified recovery bundle is
`~/hipengine-history-recovery-20261023/original-all-refs.bundle`; remote-ref
snapshots and uncompressed maps are in the same directory. The original
`~/hipEngine` checkout, scratch probes, stash and other worktrees are preserved.
The prepared rewritten checkout is `~/hipEngine-clean`. Never push backup refs
or merge the original history back into the rewritten repository.

Publication changes 52 refs, including 11 release tags from v0.3.0 onward and the
ROCm reproducer tag. Older unaffected tags keep their original IDs. PyPI files
are not republished. This document describes the validated candidate; the final
publication receipt is recorded separately after verifying the remote refs.
