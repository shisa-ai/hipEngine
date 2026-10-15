---
status: normative
owns: Rules for editing the root README product page: who it is for, what belongs there, and the failure patterns that have repeatedly made it unreadable.
---
# Root README Editing Rules

## The point

`README.md` is an **end-user product page**. Its reader has never seen this
repository, the worklog, or a benchmark artifact. They want to know four things:

1. What is hipEngine, and what models does it run?
2. How fast is it on my hardware?
3. How do I install it and start using it?
4. What will not work?

Every line on the page answers one of those questions for that reader. A line
that does not is noise, however true, careful, or hard-won it is.

## Why agents keep breaking it

Agents write README text the way they write evidence: defensively. Every number
gets its date, host, token counts, run count, matched baseline, and a caveat so
that nobody can call it wrong. That is correct in `benchmarks/README.md` and a
worklog entry. On the product page it buries the result under its own paperwork.

**On the README, accuracy comes from linking the report, not from inlining it.**
The table states the number; one link below it leads to everything that backs
the number up. If you feel the urge to qualify a number in place, that urge is
the failure mode. Put the qualification in the report.

The second failure mode is completeness: adding every new model, result, or
feature as its own paragraph because it exists. The page does not need to
mention everything that works. `docs/MODELS.md`, the model cards, and the
benchmark report do that.

## Rule 0 — Default action is to not touch it

The page has been reviewed by the lead line by line. Treat it as finished, not
as a draft to improve.

Edit `README.md` only when the lead asks for a specific README change. Not as a
side effect of a refactor, a doc sync, or a test fix. Do not reword prose you
find awkward. If something on the page looks wrong, say so in one line in your
report and leave the file alone.

When you are asked for a change, change only the lines that request names. Do
not tidy the paragraph around it, retitle the section, or fix a second thing you
noticed on the way. Report the second thing instead.

Two routine tasks touch the page, and each has a fixed scope:

- **Release prep** updates the version in the `Current release:` line and its
  one-sentence summary of what the release fixes or adds. Nothing else.
- **Benchmark rollups** update numbers in existing table cells of the generated
  block (Rule 7). Adding or removing rows, columns, tables, or prose in that
  block is a README change and needs the lead's request.

## Rule 1 — The stranger test

For each line you add or change, ask: *would someone who just found this project
on GitHub care about this, and understand it without context?* If not, cut it.

Fails the test:

- Dates, host names (`epyc`, "the HP ZBook host"), token counts per run, warmup
  and run counts, `hipcc` versions, artifact filenames.
- Ratios against a derived baseline: "**1.63x** its matched 24.36 tok/s AR".
- Explanations of why two of our own numbers differ: "the same GPU on two
  machines, so the gap is thermal headroom".
- Internal state words: `retained`, `qualified`, `certified plan`, `evidence
  ladder`, `migration path`, `source-pinned`, `opt-in result`, kernel or
  planner terms.
- Self-justification: "These are measured results, not estimates."

## Rule 2 — Performance is a table or nothing

Numbers go in tables, one result per cell. Never write performance as prose.

If a result cannot be stated as a compact table row, leave it off the page
until it can. This paragraph was removed for exactly that reason:

```markdown
Time-series forecasting (TimesFM 2.5 200M). hipEngine decodes batch=8, context
8192, horizon 512 forecasts in 0.082 s on the HP ZBook Strix Halo host (8.6x the
official torch reference) and 0.062 s on a Framework Desktop host — the same GPU
on two machines, so the gap is thermal headroom.
```

Columns follow the same rule. A column that tracks measurement state rather than
a result (`AR measured`, `Date`, `Host`, `Runs`) does not belong.

Never put notes under a table that restate its rows, its protocol, or its
provenance. This was four lines under a five-row table, and all of it was cut:

```markdown
BF16 rows: September 20, `epyc`, 512/128 tokens, warmup + three runs.
INT8: September 13. MTP/capacity not refreshed. Laguna: 4K prompts.
35B GGUF MTP is opt-in; 27B MTP is **1.63x** its matched 24.36 tok/s AR,
not either decode column.
```

## Rule 3 — A caveat must change what the reader does

Ask: *if the reader skips this sentence, do they make a wrong decision?* If not,
delete it.

- "The 16-snapshot setting uses more memory and is opt-in." Keep: it changes a
  configuration choice.
- "Peers use F16 KV where hipEngine uses BF16." Delete.
- "Different prefill routes can change generated tokens." Delete. True, but
  nobody acts on it from the product page.

At most one caveat per claim. A stack of limitations under a table is a defect.

## Rule 4 — Say each fact once

Before adding a sentence, grep the README for the fact. If a heading, table
header, or another section already states it, do not state it again.

- `### Radeon Pro W7900 — 48 GB (gfx1100)` already says the GPU and its memory.
  Do not follow it with "measured on a 48 GB W7900".
- Do not put a navigation line ("[Supported models](#supported-models) lists…")
  directly above the section it points to.
- Do not add a limit that restates figures already shown in a table.

## Rule 5 — Write plainly

Use the shortest wording a user would use.

| Instead of | Write |
| --- | --- |
| `; --prefix-cache off rolls it back` | `` (`--prefix-cache off` to disable) `` |
| MTP is speculative decoding in qualified scopes | MTP is speculative decoding |
| is not a default | do not use it |
| `retained` ROCm 7.13 | ROCm 7.13 |

## Rule 6 — Removed detail moves, it does not vanish

Before cutting a fact from the README, confirm it is recorded in
`benchmarks/README.md`, a `docs/model-cards/` page, or a worklog entry. If it is
not, put it there first. The rule is about where a fact lives, not whether it
survives.

## Rule 7 — The performance block is generated

The block between `<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->` and
`<!-- END TOPLINE:README_HIGHLIGHTS -->` is copied from `benchmarks/README.md`.
These rules apply to that source block too. Edit the source, then run:

```bash
python3 scripts/sync_benchmark_readme.py --write
```

Never hand-edit the block in `README.md`.

## Self-check before committing a README change

1. Did the lead ask for this specific change? If not, revert it.
2. Does each new line pass the stranger test?
3. Is every number in a table, with no provenance attached?
4. Does each caveat change a reader's decision?
5. Is the fact already on the page?
6. Is removed detail recorded somewhere else in the repository?
7. Did `python3 scripts/docs/check_all.py` and
   `uv run pytest tests/test_integration_benchmark_readme_sync.py` pass?

When in doubt about a line you are adding, leave it out. These rules judge your
change, not the rest of the page.

## Enforcement

| Gate | Catches |
| --- | --- |
| `scripts/sync_benchmark_readme.py --check` | Block out of sync; block over 150 lines, 10 prose paragraphs, or 2800 prose characters; document over 500 lines |
| `scripts/docs/check_all.py` | Documentation front-matter, indexes, export sync, published command drift |
| `tests/test_integration_benchmark_readme_sync.py` | Section order, retired phrases, internal terms, duplicated sentences, malformed tables, broken links |

The gates catch phrases. They cannot catch a well-written paragraph that should
not exist. Only the stranger test and Rule 0 catch that.
