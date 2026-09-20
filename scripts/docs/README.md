# Documentation tooling

Scripts that keep `docs/` navigable and its indexes honest.

| Script | Does |
| --- | --- |
| `check_all.py` | Runs every documentation gate and prints one summary. Start here. |
| `check_docs.py` | Validates the `status:`/`owns:` front-matter on every document under `docs/` and regenerates the index tables. |

Related gates that live elsewhere because other documents and tests already
reference them by path; `check_all.py` runs them too:
`scripts/check_envs_docs.py`, `scripts/sync_benchmark_readme.py`,
`scripts/check_published_command_drift.py`.

## The front-matter contract

Every Markdown file under `docs/` — except the generated `README.md` index
pages, `docs/examples/`, `docs/testing/`, and published Hugging Face model cards
— begins with:

```yaml
---
status: normative | current | closed | superseded
owns: one line naming what this document is the source of truth for
superseded_by: docs/...      # required when status is `superseded`
---
```

`status` exists so "is this binding?" is answerable with `grep`, without opening
a 5,000-line file:

- **normative** — binding rules. Follow them.
- **current** — the present contract or state of a subsystem.
- **closed** — finished work kept as evidence. Not binding.
- **superseded** — replaced; `superseded_by` names the replacement.

## No hard-coded home directories

A path like `/home/<you>/llama.cpp` is correct for exactly one reader. `check_docs.py`
fails on `/home/<user>/` and `/Users/<user>/` anywhere in `AGENTS.md`, `docs/`, or
`benchmarks/` prose:

- **Outside the repository** — use `~/`, which every shell expands and which
  `scripts/check_lineage.py` resolves with `Path.expanduser()`.
- **Inside the repository** — use a repo-relative path (`.venv/bin/python`,
  `scripts/foo.py`), not a path through someone's home directory.

Exempt, because they record what something said rather than instruct:
`docs/testing/` dated migration records, `docs/examples/` published configs, and
`benchmarks/results/` evidence artifacts.

## Adding a document

1. Put it in the right directory: `docs/` root only for something every agent
   must know exists, otherwise `reference/`, `campaigns/`, `model-cards/`, or
   `archive/`.
2. Add the front-matter block. `owns:` is **required** for root and
   `reference/` documents and the check fails without it. Elsewhere `owns: TODO`
   is accepted and reported as a warning so it can be filled in later.
3. Run `python3 scripts/docs/check_docs.py --write` to refresh the indexes and
   commit the regenerated `README.md` pages with your change.

## Enforcing it on commit

`check_all.py` is safe to call from a `pre-commit` hook:

```bash
printf '#!/bin/sh\nexec python3 scripts/docs/check_all.py\n' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

It is read-only by default, so it reports index drift rather than silently
rewriting files mid-commit. Pass `--write` manually when you want the fix
applied.
