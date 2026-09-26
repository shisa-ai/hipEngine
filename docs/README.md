# hipEngine Documentation Index

This index is generated. Run `python3 scripts/docs/check_docs.py --write` after
adding or moving a document; the prose sections below are preserved.

Every document carries front-matter declaring its `status` and what it `owns`:

- **normative** — binding rules. Follow them.
- **current** — the present contract or state of a subsystem. Accurate as written.
- **closed** — a finished piece of work, kept as evidence. Not binding.
- **superseded** — replaced; the front-matter names the replacement.

Project-wide ground rules live in [`../AGENTS.md`](../AGENTS.md), which every
session reads. Start there, then come here for the document your task touches.

## Top level

<!-- BEGIN GENERATED: top -->
| Document | Status | Owns |
| --- | --- | --- |
| [`API.md`](API.md) | current | OpenAI-compatible server usage, endpoint support, request/response semantics, and current limitations. |
| [`BENCHMARK.md`](BENCHMARK.md) | **normative** | Benchmark protocols, baselines to beat, required evidence fields, correctness thresholds, and artifact/rollup format. |
| [`ENVS.md`](ENVS.md) | current | Complete environment-variable reference and the recommended profiles for normal use, ROCm setup, and profiling. |
| [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) | **normative** | Strict/production/batch-invariant contracts, numerical gates, exact ownership and failure-containment semantics, and kernel-variant selection policy. |
| [`KERNELS.md`](KERNELS.md) | current | Kernel source catalog, model/quant and registry mappings, arithmetic variants, and fused fallback map. |
| [`LESSONS-LEARNED.md`](LESSONS-LEARNED.md) | current | Architecture-independent porting, build, and runtime gotchas, plus historical integration case studies. |
| [`MODELS.md`](MODELS.md) | current | Models, quantizations, and backends hipEngine has implemented and measured. |
| [`OPTIMIZATION.md`](OPTIMIZATION.md) | **normative** | Rules for kernel work, performance claims, and benchmark rows: evidence fields, anti-gaming, correctness gates, promotion, lineage, profiling. |
| [`PLAN.md`](PLAN.md) | **normative** | Architecture, plugin boundaries, phase roadmap, LoC budgets, and the invariants that define hipEngine. |
| [`RDNA3-TUNING-GUIDE.md`](RDNA3-TUNING-GUIDE.md) | current | Externally-referenced RDNA3 tuning guide; the current respin of the kernel-level findings also recorded in LESSONS-LEARNED.md. |
| [`REFACTOR.md`](REFACTOR.md) | current | Cleanup ledger for dead flags, duplicate dispatch paths, and fallback code to remove after optimal paths are proven. |
| [`ROOFLINE.md`](ROOFLINE.md) | current | RDNA3 / W7900 performance model: hardware limits, regimes, decision tree, and the what-not-to-chase catalog. |
| [`TESTING.md`](TESTING.md) | **normative** | RED/GREEN workflow, correctness oracles, fixture policy, test naming/discovery, and the validation matrix. |
<!-- END GENERATED: top -->

## Sections

<!-- BEGIN GENERATED: sections -->
| Directory | Contents | Files |
| --- | --- | --- |
| [`reference/`](reference/) | Current subsystem contracts. Read the one your task touches. | 40 |
| [`campaigns/`](campaigns/) | Bounded model/hardware campaigns. Mostly closed; kept as evidence. | 65 |
| [`model-cards/`](model-cards/) | Per-model support status, quality, and integration notes. | 11 |
| [`archive/`](archive/) | Superseded documents, closed proposals, and frozen history. | 25 |
<!-- END GENERATED: sections -->

## Common reading paths

- **Before changing architecture or dispatch:** [`PLAN.md`](PLAN.md), then
  [`reference/IMPLEMENTATION.md`](reference/IMPLEMENTATION.md).
- **Before optimizing a kernel or asserting any number:**
  [`OPTIMIZATION.md`](OPTIMIZATION.md) first — it is the scoped rules file and
  names which gates actually apply to you. Then [`KERNELS.md`](KERNELS.md) and
  [`ROOFLINE.md`](ROOFLINE.md) to check the change matches the measured
  bottleneck, and [`RDNA3-TUNING-GUIDE.md`](RDNA3-TUNING-GUIDE.md) for
  RDNA3-specific technique.
- **Before making a performance claim:** [`BENCHMARK.md`](BENCHMARK.md), verify
  the host ROCm environment against [`reference/THEROCK.md`](reference/THEROCK.md),
  and do not compare absolute rates across independent hardware lanes. Then
  update [`../benchmarks/README.md`](../benchmarks/README.md),
  [`../benchmarks/CHANGELOG.md`](../benchmarks/CHANGELOG.md), and write an
  artifact under [`../benchmarks/results/`](../benchmarks/results/).
- **Before changing math or correctness-sensitive code:**
  [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md), the dated
  [production accuracy review](reference/PRODUCTION-ACCURACY-POLICY-REVIEW-2026-08-31.md),
  and [`TESTING.md`](TESTING.md). Declare the applicable contract and add a
  CPU-reference or fixture gate before relying on benchmark output.
- **Before changing server, API, or default behavior:** [`API.md`](API.md) and
  [`ENVS.md`](ENVS.md), plus `AGENTS.md` "Product Defaults". Kernel promotion
  gates do not apply to product decisions.
- **Before opening a less-bounded optimization search:**
  [`reference/PROCESS-EXPLORATION.md`](reference/PROCESS-EXPLORATION.md) — freeze
  the evaluator and generalization envelope, then seed distinct hypotheses.
- **Looking for what was already tried:** search
  [`campaigns/`](campaigns/) before starting a new campaign, and
  [`LESSONS-LEARNED.md`](LESSONS-LEARNED.md) for the do-not-chase catalog.

Current immutable handoff entries live under
[`../worklog/entries/`](../worklog/entries/);
[`../WORKLOG.md`](../WORKLOG.md) links the entry format, local renderer, and
frozen pre-Worklog2 history.
