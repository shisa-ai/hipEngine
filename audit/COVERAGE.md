# Architecture cleanup coverage and work order

## Objective

Simplify ownership and execution paths, remove genuinely dead code, and preserve
behavioral tests that make refactoring safe. Line count is a baseline, not a
success criterion. Splitting a large file without reducing responsibilities or
duplicated policy does not satisfy this objective.

This is the initial assessment at `cbc5682e5` (2026-10-23), not a certification of
the whole repository. The standing decisions remain in `audit/triage/`; findings
from a scan remain evidence to investigate, not deletion instructions.

## Size baseline

`scc` 4.1.0, Python only, counted 3,999 files. Matching its per-file locations
against `git ls-files '*.py'` confirmed that these are tracked files, not a
virtual environment. Total lines include comments and blanks; code lines exclude
them. Embedded kernel source inside Python strings still belongs to this Python
count, so it is not a pure host-language complexity measure.

| Area | Files | Total lines | Code lines |
| --- | ---: | ---: | ---: |
| `tests/` | 1,949 | 661,788 | 544,589 |
| `scripts/` | 1,308 | 506,693 | 439,700 |
| `hipengine/` | 579 | 509,686 | 430,902 |
| Other tracked Python | 163 | 40,200 | 34,323 |
| Total | 3,999 | 1,718,367 | 1,449,514 |

Reproduce the per-file input with:

```bash
scc --include-ext py --by-file --format json --no-cocomo -o /tmp/hipengine-python.json .
git ls-files '*.py'
```

Join `Files[].Location` (without an optional `./`) to the tracked paths before
summing `Lines` and `Code`. Do not include ignored environments or compare this
baseline to a differently filtered count.

The largest product files by Python code lines include
`hipengine/runtime/qwen35_gguf_runner.py` (36,028),
`hipengine/server/api.py` (19,925), and
`hipengine/runtime/qwen35_paro_runner.py` (12,845). Size identifies review targets;
it does not establish that their implementations are redundant.

## What the audit establishes today

The baseline refresh catalogued 4,290 rows: 520 ledger entries, 852 flags,
309 kernel sources, 250 campaign candidates, 1,747 worklog items, and 612
code findings. Zero untriaged code findings does **not** mean zero defects: the
checks cover a small set of patterns and a disposition can preserve real debt.
Refresh archived eight absent decisions without marking their work resolved.
The one stale raw-profiler ledger decision was reviewed against the completed
history cleanup and explicitly resolved.

| Concern | Existing coverage | Missing evidence or next check |
| --- | --- | --- |
| Plugin boundaries | `axis-branch` detects some literal comparisons | AST comparisons, aliases, conditional expressions, and actual ownership boundaries; registry resolution is not inferred by the regex |
| Torch-free runtime | `torch-hot-path` detects selected module-level imports | Function-local and indirect imports on the public execution path; the current check does not compute reachability |
| Temporary paths | Flag inventory, ledger, `ungoverned-flag-branch` | Duplicate owners and fallback chains without a flag; verify which path public requests execute |
| Dead code | Kernel references, stubs, markers, recorded ledger debt | Overwritten definitions; module/symbol reachability with public exports, dynamic imports, decorators and registrations accounted for |
| Duplication | No general check | Near-duplicate script families and repeated model/session policy; identical syntax alone is not a deletion proof |
| Test discovery | Tier-name gate under `tests/` | Audit contract tests are outside default pytest roots; other standalone validation commands need an explicit owner |
| Test usefulness | GPU-guard check and kernel-name references | Contract/oracle inventory, construction coverage, selected behavioral mutations, and source-only assertions that duplicate behavior tests |
| Refactor safety | Targeted tests and explicit integration tiers | A named CPU/public-boundary bundle for each ownership change, plus applicable GPU/stateful tests when those paths change |

The `torch-hot-path` and `unguarded-hip-test` checks are lexical approximations,
not proof of runtime reachability or effective skip behavior. A guard string
elsewhere in a test file does not prove every GPU call is guarded. New checks
must document such limits and test both positive cases and intentional patterns.

## Test assessment

The default unit run at the baseline reported 16,381 passed,
271 skipped and five failed. The failures involved absent Git LFS AOTriton
payloads in this fresh checkout, not an identified refactoring regression.
Restore the payload and rerun the affected files before using this checkout as a
green baseline. Do not skip required resource checks merely to turn it green.
The focused repair here required `git lfs install --local` followed by
`git lfs pull`; the four affected test files then passed (201 passed, 12 skipped).
No test assertions or runtime code were changed to clear these failures.

The audit has 138 unittest contract tests, separately runnable with:

```bash
python3 -m unittest discover -s audit/tests -t audit
```

They are outside `pyproject.toml`'s pytest roots. An explicit `pytest audit/tests`
also misses the four modules lacking tier prefixes. Integrate this safety net
without importing the product's global fixture machinery into the audit tests.
Preserve the standalone unittest command.

The sample review found substantial behavioral coverage, not a suite dominated
by source-text assertions. A heuristic source-read/string-assert census found
72 functions across 58 unit/integration files. That is a review list, not a
verdict: architectural checks, exception-boundary checks and anti-gaming checks
can legitimately inspect source. Replace spelling checks only when a behavioral
assertion protects the intended contract; verify it fails under a representative
fault.

Construction needs special attention. The review counted 327 `object.__new__`
uses across 96 unit files. Bare instances are useful for CPU-only method tests,
but do not test constructor wiring or initialized state. Before moving state
between owners, exercise construction through `LLM` or the server as well.
Public import, server conformance, cancellation and speculative-ownership tests
live partly in the integration tier; plain `pytest` does not select them.

The per-layer llama.cpp parity-tool family is a large consolidation target:
143 scripts plus 142 test files account for about 100,237 physical lines.
Its tests protect diagnostic classification, not product execution directly.
The campaign is closed, but that does not prove the tooling is unused. In
particular, `scripts/llamacpp_mtp_audit_layer0_attn_norm_formula.py` supplies
shared numerical helpers to many other scripts and tests. Trace CLI, replay,
documentation and import consumers before retiring any member.

## Ordered cleanup units and acceptance

1. **Make the audit a reliable guard.** Detect overwritten definitions with
   conservative AST analysis and stable identities; distinguish the earlier
   definition from the live replacement. Cover decorators, property setters,
   overloads, aliases, conditional definitions and definition-time uses to avoid
   claiming intentional rebindings are dead. Bring audit tests into ordinary
   test discovery. Do not increase the debt budget to accept new findings.
2. **Remove a proven dead definition.** The earlier
   `Qwen35GGUFBringupGenerator.configure_dms` is overwritten by a later method
   that adds a preparation-state guard. Preserve the later method, test both
   configuration forms and rejection without mutation after preparation, and
   confirm the duplicate disappears from the audit. This is a calibration unit,
   not a material reduction in repository size.
3. **Consolidate one diagnostic family.** Start with the four
   `llamacpp_mtp_audit_layer{3,7,11,15}_attn_output_oracle` implementations.
   Compare drift, preserve CLI/report schemas and shared helper consumers, and
   parameterize the layer-specific cases rather than deleting assertions.
   Decide on the remaining families only after this proof of consolidation.
4. **Map product ownership before moving code.** Trace one request through
   `LLM`/server admission, generation, resident runner, scheduler, KV ownership,
   kernel selection and teardown. Record who owns each decision and state
   transition. Start with the large Qwen GGUF/server path; identify duplicate
   policy rather than moving its branches into another large helper. Check
   imports/cycles and registry consumers statically, then confirm the selected
   route through the public surface.
5. **Refactor one ownership seam at a time.** Add behavioral coverage at that
   seam before extraction or removal. Preserve explicit-request errors, defaults,
   request isolation, cancellation, rollback, graph lifetime and close behavior.
   Arithmetic or kernel changes require their separate optimization gates; do
   not mix them into a host-ownership cleanup.

For each unit, record the owner/callers, proposed removal or consolidation,
behavioral oracle, negative case, focused command, and public-surface check where
applicable. Run the audit gate and commit a validated logical unit before taking
the next. A caller search, coverage percentage, passing test count or reduced
line count alone is not sufficient acceptance evidence.

## Implementation progress

- Audit test discovery is connected: both `tests/` and `audit/tests/` are pytest
  roots, audit test filenames carry the unit prefix, and repository-level tier
  selection no longer depends on loading product fixtures. The baseline gap
  described above is closed; standalone unittest discovery remains supported.
- The `shadowed-definition` check now reports a conservative pair of overwritten
  function definitions, with locations and signature/body comparisons. Four real
  pairs were reviewed and recorded for removal. Decorated, conditionally rebound
  and multi-definition groups are excluded; dynamic reachability is still a
  review responsibility, not a property this check proves.

## Completion boundary

This assessment does not claim comprehensive dead-code detection, full dynamic
reachability, useful coverage of every test, or a cleaned product architecture.
The first implementation pass covers the audit guard and a small proven removal.
The diagnostic consolidation and product ownership work remain explicitly
pending; they must not disappear behind a green audit count.
