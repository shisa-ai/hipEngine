---
status: normative
owns: Rules for editing the root README product page: what belongs there, what does not, and the duplication and caveat patterns that have damaged it before.
---
# Root README Editing Rules

`README.md` is a product page. It is the first thing a prospective user reads and
the only page most of them will ever read. It is not a report, a changelog, a
benchmark log, or a place to record what you learned.

Every rule below exists because an agent broke it. The examples are real text
from this repository's history, not invented.

## Rule 0 — Default action is to not touch it

Edit the root README **only** when the lead asks for a specific README change.

Do not edit it as a side effect of another task. Release prep, doc syncs,
benchmark rollups, refactors, and test repairs are not README work. Do not reword
a sentence you find awkward. Do not "improve" the page while passing through it.
Do not reorder sections, retitle headings, or convert tables to prose because you
prefer the result.

If you believe something on the page is wrong, say so in one line and leave the
file alone. If you are about to change prose the lead wrote, stop.

## Rule 1 — One fact, one place

Before you add a sentence, find the sentence that already says it. Run the grep.
Repeating a fact in a second section does not reinforce it; it makes the page
read as if its author never got past the first screen.

The README has exactly one copy of: the release version, the absence of a
PyTorch dependency, the CPU-inference limitation, the model-support link, and
each capacity figure.

## Rule 2 — State the result, not the measurement

A number belongs on the product page when a reader can act on it: throughput,
memory footprint, maximum context, price-relevant GPU size.

Warmup counts, tokens per run, run counts, dates, host codenames, artifact
filenames, `hipcc` versions, and per-row provenance do not. They belong in
`benchmarks/README.md` and the JSON artifact it links.

**Bad** — four lines of protocol bookkeeping under a five-row table:

```markdown
BF16 rows: September 20, `epyc`, 512/128 tokens, warmup + three runs.
INT8: September 13. MTP/capacity not refreshed. Laguna: 4K prompts.
35B GGUF MTP is opt-in; 27B MTP is **1.63x** its matched 24.36 tok/s AR,
not either decode column.
```

**Good** — nothing. The table stands, and the block links the benchmark report
once, below it.

## Rule 3 — A caveat must change what the reader does

Test every caveat with one question: *if the reader ignores this sentence, do
they make a wrong decision?* If the answer is no, delete it.

- "The 16-snapshot setting uses more memory and is opt-in" — changes a decision. Keep.
- "Peers use F16 KV where hipEngine uses BF16" — does not. Delete.
- "In these tests, DMS INT8 agreed more closely with BF16" — the table says it. Delete.
- "Different prefill routes can change generated tokens" — a real caveat, but it
  does not change what a reader does with a prefix-cache toggle. It belongs in
  the benchmark report.

Do not stack caveats. One caveat per claim, and only when it changes behavior.
Four limitations stacked under a three-row table is a defect, not thoroughness.

## Rule 4 — Never repeat what a heading or table header already says

A section named `### Radeon Pro W7900 — 48 GB (gfx1100)` has already stated the
GPU and its memory. Do not open the next paragraph with "The concurrency figures
come from a 48 GB W7900".

Likewise a definition. If the performance block says "prompt processing is
input, text generation is output", do not also write it in the section intro two
lines above.

## Rule 5 — No internal vocabulary

Internal words name states the reader cannot see. They are meaningless on a
product page and they make the writing look like it leaked out of a design doc.

| Do not write | Write |
| --- | --- |
| `retained` ROCm 7.13 | ROCm 7.13 |
| MTP is speculative decoding in qualified scopes | MTP is speculative decoding |
| keeps the previous behaviour otherwise | otherwise keeps hipEngine's default kernel selection |
| is not a default | do not use it |
| the certified plan / the migration path | (delete; not a user-visible concept) |

Also banned: `source-pinned`, `physical c8`, `packet reaches`, `variant
manifest`, `head-major`, and any kernel-internal or planner-internal term. The
existing test enforces several of these.

## Rule 6 — No navigation sentence that duplicates the section below it

**Bad** — a table-of-contents line placed directly above the section it
describes, whose own closing line says the same thing:

```markdown
[Supported models](#supported-models) lists the current model families and
hardware; the [model support reference](docs/MODELS.md) gives exact formats,
checkpoints, and limitations for each.

## Supported models
...
See the [model support reference](docs/MODELS.md) for exact checkpoints, ...
```

**Good** — delete the first paragraph. The section is right there.

## Rule 7 — Never state the same thing in two sections

This is Rule 1 applied to sections you did not write. Before adding a bullet to
a list of limits, check whether the section above already covers it.

**Bad** — a limit that is four restatements and one new fact:

```markdown
- The concurrency figures come from a 48 GB W7900. On a 24 GB card, single-request
  context is qualified to 232,448 tokens on the DMS and opt-in direct-INT8 routes;
  concurrent-request shapes at 24 GB are not qualified, so keep a conservative
  context limit there.
```

The 48 GB is in the table heading. The 232,448 figure and the direct-INT8 warning
are in `### Long context on a 24 GB GPU`. The release version is at the top of
the file. Only the last clause is new.

**Good:**

```markdown
- Concurrent long-context requests on a 24 GB card are untested; keep the
  context limit conservative there.
```

## Rule 8 — Detail you remove goes to the report, never nowhere

Deleting a sentence from the README is not licence to delete the fact from the
repository. Before removing anything, confirm it is recorded in
`benchmarks/README.md`, the relevant model card under `docs/model-cards/`, or a
worklog entry. If it is not, move it there first, then remove it from the
product page.

Never remove a measurement, a gate result, or a limitation from the repository.
The rule is about *where* it lives, not whether it survives.

## Rule 9 — The performance tables are generated

The block between `<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->` and
`<!-- END TOPLINE:README_HIGHLIGHTS -->` is copied verbatim from
`benchmarks/README.md`. Edit the source, then run:

```bash
python3 scripts/sync_benchmark_readme.py --write
```

Never hand-edit the block in `README.md`. `--check` fails when the two differ.

## What belongs on the page

| Belongs | Does not belong |
| --- | --- |
| What hipEngine is, in two sentences | Implementation diaries, optimization history |
| Supported model families, with model-card links | Kernel, planner, or scheduler internals |
| Current headline numbers as tables | Candidate ladders, rejected experiments |
| Install, serve, chat, and Python API examples | Exact benchmark commands |
| Limits a reader must act on | Evidence inventories, artifact lists |
| Links to the report for everything else | Internal blockers, migration states |

## The self-check before you commit a README change

1. Did the lead ask for this specific change? If not, revert it.
2. Grep the README for the fact. Is it already there?
3. Does a heading or table header already say it?
4. Would a reader make a wrong decision without this sentence?
5. Can the reader act on the number, or is it provenance?
6. Is every word here something a user could observe?
7. Is the removed detail recorded somewhere in the repository?
8. Did `scripts/docs/check_all.py` and the README sync tests pass?

If any answer is uncomfortable, cut the sentence. The page is stronger shorter.

## Enforcement

| Gate | Catches |
| --- | --- |
| `scripts/sync_benchmark_readme.py --check` | Block out of sync; block over 150 lines, 10 prose paragraphs, or 2800 prose characters; document over 500 lines |
| `scripts/docs/check_all.py` | Documentation front-matter, indexes, ENVS coverage, export sync, published command drift |
| `tests/test_integration_benchmark_readme_sync.py` | Section order, tables in the model section, retired filler phrases, duplicated sentences, malformed tables, broken local links and anchors |

Mechanical gates catch crude regressions. They cannot catch a well-written
paragraph that should not exist. Rule 0 is the real defense.
