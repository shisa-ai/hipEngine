# Surya/main integration

Date: 2026-09-14 (Asia/Tokyo).

The pre-rebase Surya tip `6978ce7fd` was pushed to `origin/surya` before
integration and is also saved as `backup/surya-before-main-rebase-20260914`.
Local main retained its five unpublished commits, including upload ownership
and test portability fixes, while merging `origin/main` at `110db96d4`.
The resulting local main is `40bc24c7d`; Surya's 64 commits were rebased onto it.
The integration preserves main history and permits a fast-forward merge.

The decisions in [UD/main integration](UD-MAIN-INTEGRATION.md) were preserved:
profile selection, graph admission, dispatch policies, journal ownership,
backend scope, and conservative test execution tiers. In particular:

- TimesFM, TimesFM3, Evie and Qwen35 runtime files match local main. Its
  owner-retaining upload helper and stream-ordered byte resets take precedence
  over the older overlapping Surya cleanup patches.
- TimesFM3's causal offset is an initialized read-only input, not per-request
  scratch. Its poison test excludes that field while continuing to poison the
  mutable buffers. TimesFM retains both main's request-reuse tests and Surya's
  poison test.
- Surya's runtime and kernel implementations match the pre-rebase branch. The
  CPU reference changed only its test-path documentation during integration.
- All main scoreboard content and refactor decisions were retained. Historical
  benchmark artifacts and immutable worklogs were not rewritten.
- Twenty-two new test modules now follow the execution-tier naming contract.
  The acceptance command discovers `test_*_surya*.py` across tiers and continues
  to fail on skipped coverage. The [rename record](testing/test-tier-migration-surya-2026-09-14.json)
  maps historical commands to current paths without controlling test selection.

Validation:

- Surya acceptance: **415 passed, zero skips**.
- Unit tier: **8,893 passed, 21 skipped** (optional dependencies and out-of-depth cases).
- Shared server/registry/multimodal/benchmark integration: **607 passed, zero skips**.
- TimesFM/Evie ownership, Qwen attention, and H2D hygiene: **52 passed, zero skips**.
- All-tier collection: **16,833 tests**, no collection errors.
- Test-tier naming, README exports, diff checks, and worklog validation pass.

These suites overlap; their pass counts are not additive unique coverage.
[Machine-readable commands and results](../benchmarks/results/2026-09-14-surya-main-integration-validation.json)
record the exact execution scope. This is an
integration correctness check, not a new speed measurement. The final benchmark
numbers in [MODEL-SURYA.md](MODEL-SURYA.md) retain their original source revision,
physical host, protocol and qualification scope.
