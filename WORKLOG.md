# hipEngine Worklog

hipEngine uses immutable, one-file-per-logical-unit worklog entries so parallel
branches and worktrees do not contend on one append target.

- **Current entries:** [`worklog/entries/`](worklog/entries/)
- **Format and commands:** [`worklog/README.md`](worklog/README.md)
- **Pre-cutoff history:** 7,232 journal entries imported under
  [`worklog/entries/`](worklog/entries/) with `worker: legacy`. The original
  journal bytes live in Git history at the ref pinned in
  [`worklog/legacy-port-manifest.json`](worklog/legacy-port-manifest.json)
- **Approved migration plan:**
  [`docs/archive/PLAN-WORKLOG2-revamp.md`](docs/archive/PLAN-WORKLOG2-revamp.md)

Create a current entry with:

```bash
python3 scripts/worklog.py new \
  --worker <stable-worker-or-lane-id> \
  --status completed \
  --title "Short outcome"
```

Stage an entry, then validate and render the ignored local chronological view
with:

```bash
git add worklog/entries/<entry>.md
python3 scripts/worklog.py check
python3 scripts/worklog.py render
```

`check` validates staged and tracked content only, so another worker's unstaged
entry cannot block a commit.

Do not append to this navigation page or resurrect the retired legacy journal
into the tree.
