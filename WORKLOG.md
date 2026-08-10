# hipEngine Worklog

hipEngine uses immutable, one-file-per-logical-unit worklog entries so parallel
branches and worktrees do not contend on one append target.

- **Current entries:** [`worklog/entries/`](worklog/entries/)
- **Format and commands:** [`worklog/README.md`](worklog/README.md)
- **Frozen history through commit `7c7c18875`:**
  [`WORKLOG-LEGACY.md`](WORKLOG-LEGACY.md)
- **Approved migration plan:**
  [`docs/PLAN-WORKLOG2-revamp.md`](docs/PLAN-WORKLOG2-revamp.md)

Create a current entry with:

```bash
python3 scripts/worklog.py new \
  --worker <stable-worker-or-lane-id> \
  --status completed \
  --title "Short outcome"
```

Validate and render the ignored local chronological view with:

```bash
python3 scripts/worklog.py check
python3 scripts/worklog.py render
```

Do not append to this navigation page or edit the frozen legacy journal.
