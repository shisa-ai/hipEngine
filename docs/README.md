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
| [`CROSS-ENGINE-PROTOCOL.md`](CROSS-ENGINE-PROTOCOL.md) | **normative** | The protocol every cross-engine comparison on this host follows: fixtures, settings, and what a comparable row must record. |
| [`ENVS.md`](ENVS.md) | current | Complete environment-variable reference and the recommended profiles for normal use, ROCm setup, and profiling. |
| [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) | **normative** | Strict/production/batch-invariant contracts, numerical gates, exact ownership and failure-containment semantics, and kernel-variant selection policy. |
| [`HOWTO-README.md`](HOWTO-README.md) | **normative** | Rules for editing the root README product page: who it is for, what belongs there, and the failure patterns that have repeatedly made it unreadable. |
| [`KERNELS.md`](KERNELS.md) | current | Kernel source catalog, model/quant and registry mappings, arithmetic variants, and fused fallback map. |
| [`LESSONS-LEARNED.md`](LESSONS-LEARNED.md) | current | Architecture-independent porting, build, and runtime gotchas, plus historical integration case studies. |
| [`MODELS.md`](MODELS.md) | current | Models, quantizations, and backends hipEngine has implemented and measured. |
| [`MTP-SERVING.md`](MTP-SERVING.md) | current | The what-works-where MTP serving policy matrix across models, backends, sampling modes, concurrency widths, and automatic-versus-explicit admission, plus the per-request force semantics. |
| [`OPTIMIZATION.md`](OPTIMIZATION.md) | **normative** | Rules for kernel work, performance claims, and benchmark rows: evidence fields, anti-gaming, correctness gates, promotion, lineage, profiling. |
| [`PLAN.md`](PLAN.md) | **normative** | Architecture, plugin boundaries, phase roadmap, LoC budgets, and the invariants that define hipEngine. |
| [`PREFILL-WILKIN-INDEPENDENT-AUDIT.md`](PREFILL-WILKIN-INDEPENDENT-AUDIT.md) | closed | The September 2026 independent source and evidence audit of the prefill and Wilkin work. |
| [`PRODUCTION-ACCURACY-RECALIBRATION-PROPOSAL-2026-09-16.md`](PRODUCTION-ACCURACY-RECALIBRATION-PROPOSAL-2026-09-16.md) | closed | The predeclared 2026-09-16 accuracy recalibration proposal, frozen before its measurements. |
| [`QWEN4EXP-CROSS-ENGINE-CAMPAIGN-REPORT.md`](QWEN4EXP-CROSS-ENGINE-CAMPAIGN-REPORT.md) | closed | Evidence and direction from the 2026-10-13 Qwen4Exp cross-engine gap campaign. |
| [`QWEN4EXP-CROSS-ENGINE-GAP-MATRIX.md`](QWEN4EXP-CROSS-ENGINE-GAP-MATRIX.md) | closed | The 2026-10-13 Qwen4Exp cross-engine gap investigation plan for the UD-Q4_K_XL GGUF on gfx1151. |
| [`QWEN4EXP-HTTP-COVERAGE.md`](QWEN4EXP-HTTP-COVERAGE.md) | current | The Qwen3.8 Flash-Next HTTP surface coverage and the evidence each served feature rests on. |
| [`QWEN4EXP-JOURNEY-REMAINING.md`](QWEN4EXP-JOURNEY-REMAINING.md) | closed | Dated record of the Flash-Next journey experiments that were left open; live rows moved to the optimization ledger. |
| [`QWEN4EXP-OPTIMIZATION-LEDGER.md`](QWEN4EXP-OPTIMIZATION-LEDGER.md) | current | The live Qwen3.8-Flash-Next optimization ledger: open rows, measured gains, and rejected candidates. |
| [`QWEN4EXP-PRODUCTION-SERVING.md`](QWEN4EXP-PRODUCTION-SERVING.md) | current | The Flash-Next production-serving campaign checklist and the evidence for each requirement. |
| [`QWEN4EXP-STATUS.md`](QWEN4EXP-STATUS.md) | current | Current Qwen3.8-Flash-Next optimization status and headline measurements. |
| [`QWEN4EXP-STRIX-JOURNEY-CAMPAIGN.md`](QWEN4EXP-STRIX-JOURNEY-CAMPAIGN.md) | closed | Dated record of the September 13-15 2026 Flash-Next Strix journey campaign. |
| [`QWEN4EXP-T16-RESIDENCY.md`](QWEN4EXP-T16-RESIDENCY.md) | current | The Qwen4Exp T16 tile-residency design: which tiles materialize at load and which raw weights they replace. |
| [`QWEN4EXP-X7-DECISION-BRIEF.md`](QWEN4EXP-X7-DECISION-BRIEF.md) | closed | The resolved X-7 decision on the prefill_last scope and the compensated-T16 candidates. |
| [`RDNA3-TUNING-GUIDE.md`](RDNA3-TUNING-GUIDE.md) | current | Externally-referenced RDNA3 tuning guide; the current respin of the kernel-level findings also recorded in LESSONS-LEARNED.md. |
| [`REFACTOR.md`](REFACTOR.md) | current | Cleanup ledger for dead flags, duplicate dispatch paths, and fallback code to remove after optimal paths are proven. |
| [`ROOFLINE.md`](ROOFLINE.md) | current | RDNA3 / W7900 performance model: hardware limits, regimes, decision tree, and the what-not-to-chase catalog. |
| [`TESTING.md`](TESTING.md) | **normative** | RED/GREEN workflow, correctness oracles, fixture policy, test naming/discovery, and the validation matrix. |
| [`TOKEN-LOGPROBS.md`](TOKEN-LOGPROBS.md) | current | How Flash-Next token logprobs record tokenizer bytes and align them with text. |
<!-- END GENERATED: top -->

## Sections

<!-- BEGIN GENERATED: sections -->
| Directory | Contents | Files |
| --- | --- | --- |
| [`reference/`](reference/) | Current subsystem contracts. Read the one your task touches. | 42 |
| [`campaigns/`](campaigns/) | Bounded model/hardware campaigns. Mostly closed; kept as evidence. | 72 |
| [`model-cards/`](model-cards/) | Per-model support status, quality, and integration notes. | 12 |
| [`archive/`](archive/) | Superseded documents, closed proposals, and frozen history. | 25 |
<!-- END GENERATED: sections -->

## Campaign and reference documents

The generated tables above list the top-level contracts and the directory layout.
The dated campaign records and the reference documents carry their own
front-matter, and this table names the ones whose current state matters most.

| Document | Use it for |
| --- | --- |
| [`reference/PRODUCTION-ACCURACY-POLICY-REVIEW-2026-08-31.md`](reference/PRODUCTION-ACCURACY-POLICY-REVIEW-2026-08-31.md) | Dated evidence review of production cutoffs, calibration limits, practical quality impact, cutoff-excluded performance, and recalibration triggers. |
| [`reference/PRODUCTION-NUMERICS-CAMPAIGN.md`](reference/PRODUCTION-NUMERICS-CAMPAIGN.md) | Approved evaluator, calibration, historical-recovery, c1, and c>N/A4 campaign. |
| [`reference/IMPLEMENTATION.md`](reference/IMPLEMENTATION.md) | Current implementation status, concrete milestones, and integration notes. |
| [`archive/SOL-OPTIMIZATION.md`](archive/SOL-OPTIMIZATION.md) | Closed first-generation gfx1151/gfx1100 PARO/GGUF optimization ledger (`SOL-R0`-`R9`, `SOL-E1`/`E2`); its concurrency premise is superseded by [`archive/CONCURRENCY.md`](archive/CONCURRENCY.md) and it is retained as a dated record. |
| [`campaigns/OPTIMIZE.md`](campaigns/OPTIMIZE.md) | Optimization board for Qwen3.5-35B-A3B-PARO MoE; accepted/rejected/deferred candidates. |
| [`campaigns/OPTIMIZE-DENSE.md`](campaigns/OPTIMIZE-DENSE.md) | Optimization board for Qwen3.6-27B-PARO dense; mirror lane structure to `OPTIMIZE.md`. |
| [`campaigns/QWEN35-08B-GFX1151-VULKAN-PARITY.md`](campaigns/QWEN35-08B-GFX1151-VULKAN-PARITY.md) | Radeon 8060S campaign profiling every Qwen3.5-0.8B dense GGUF module against llama.cpp Vulkan; per-module owner decisions and parity gates. |
| [`campaigns/QWEN36-27B-GGUF-7900XTX.md`](campaigns/QWEN36-27B-GGUF-7900XTX.md) | RX 7900 XTX campaign to eliminate GGUF weight-layout duplication and beat same-card llama.cpp HIP/Vulkan in speed and memory. |
| [`campaigns/QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md`](campaigns/QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md) | gfx1100/gfx1151 campaign plan for exact Qwen3.8-27B `UD-Q4_K_M`: dense Q3/IQ codec support, operation-complete strict execution, and same-host Q4_K_M plus llama.cpp performance gates. |
| [`campaigns/GFX1100-SHAPE-AWARE-GEMV-CAMPAIGN.md`](campaigns/GFX1100-SHAPE-AWARE-GEMV-CAMPAIGN.md) | Shape-aware gfx1100 GEMV campaign plan seeded by Qingming: exact alpha/beta local128/SPLIT4 screening, cache-regime protocol, RX 7900 XTX relative comparison, and independent W7900 promotion gates. |
| [`campaigns/QWEN38-Q4KM-MTP-ACCEPTANCE.md`](campaigns/QWEN38-Q4KM-MTP-ACCEPTANCE.md) | gfx1151 physical-C3 decode-economics campaign for Qwen3.8-27B `Q4_K_M` MTP; gate state and next steps are in the document header. Appendix analyzes a DFlash2 revival. |
| [`campaigns/QWEN38-INT8-KV-CONTINUOUS.md`](campaigns/QWEN38-INT8-KV-CONTINUOUS.md) | INT8 KV continuous-batching campaign: artifact-scoped admission, compact no-mirror c>N prefill/decode, complete memory accounting, and resident lifecycle promotion. |
| [`campaigns/QWEN38-27B-GFX1100-24GB-CAPACITY.md`](campaigns/QWEN38-27B-GFX1100-24GB-CAPACITY.md) | 24 GB capacity campaign for Qwen3.8-27B `Q4_K_M` on gfx1100: measured RX 7900 XTX context ceiling, the inert INT8 KV defect, per-token footprint attribution, and the evidence rules for capacity claims. |
| [`campaigns/QWEN3.8-FLASH-NEXT.md`](campaigns/QWEN3.8-FLASH-NEXT.md) | Model bring-up campaign for the Flash-Next checkpoint: scope and boundaries, bounded functionality gates, and the closed 512/1K gates. Dated record. |
| [`campaigns/QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md`](campaigns/QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md) | gfx1151 gap-closure campaign, September 8 review: measured llama.cpp gap, role-resolved profiling recipe, phased plan, and punchlist. Dated record; superseded for current state by `QWEN4EXP-STATUS.md`. |
| [`campaigns/QWEN3.8-FLASH-NEXT-STRIX-HALO-SURVEY.md`](campaigns/QWEN3.8-FLASH-NEXT-STRIX-HALO-SURVEY.md) | Matched Strix Halo engine survey: speed, same-GGUF logits, AR/MTP exactness, test and promotion coverage, source audit, and bounded absolute-quality evidence. |
| [`campaigns/QWEN3.8-FLASH-NEXT-HALO-BOX-CAMPAIGN.md`](campaigns/QWEN3.8-FLASH-NEXT-HALO-BOX-CAMPAIGN.md) | Halo-box follow-up campaign: snapshot rows and prefill queue. Dated record; the September 8 review in the performance campaign takes precedence. |
| [`QWEN4EXP-STATUS.md`](QWEN4EXP-STATUS.md) | **Current state for Qwen3.8-Flash-Next / Qwen4Exp.** Rewritten in place: our start-versus-current rates, the competition on both the end-to-end and kernel-time axes with the accuracy basis per row, the mechanism ledger of what makes the comparators fast, certified-but-not-promoted routes, and what is left. |
| [`QWEN4EXP-OPTIMIZATION-LEDGER.md`](QWEN4EXP-OPTIMIZATION-LEDGER.md) | **The single open-work list for closing the Flash-Next gap.** Rewritten in place: every open optimization row with its measured owner cost, the evidence that bounds the opportunity, the first bounded experiment and the deciding gate, plus the rejection registry that stops failed candidates being repeated unchanged. |
| [`QWEN4EXP-STRIX-JOURNEY-CAMPAIGN.md`](QWEN4EXP-STRIX-JOURNEY-CAMPAIGN.md) | Flash-Next Strix journey campaign, September 13-15: objective and authority, frozen inputs, source-to-code map, measurement contract, punchlist, and completion gates. Dated record. |
| [`QWEN4EXP-JOURNEY-REMAINING.md`](QWEN4EXP-JOURNEY-REMAINING.md) | Execution inventory of the remaining journey candidates, with the gate-engagement rules for promoting them. |
| [`archive/QWEN4EXP-WILKIN-RUNTIME-REVIEW.md`](archive/QWEN4EXP-WILKIN-RUNTIME-REVIEW.md) | Source review of the comparator's Strix Halo runtime: what is worth porting and what the survey already ruled out. |
| [`archive/QWEN4EXP-EXTERNAL-FORKS-REVIEW.md`](archive/QWEN4EXP-EXTERNAL-FORKS-REVIEW.md) | Comparator fork arithmetic review: operand-precision error per route against our production envelope, plus the ranked external experiments. |
| [`campaigns/QWEN4EXP-CONTEXT-CAPACITY.md`](campaigns/QWEN4EXP-CONTEXT-CAPACITY.md) | Context and chunk capacity accounting: the 2051 boundary, admission footprint, and per-runner scratch. |
| [`campaigns/QWEN4EXP-TOOLS-GRAMMAR.md`](campaigns/QWEN4EXP-TOOLS-GRAMMAR.md) | Chat and tool protocol ownership: Jinja template sandbox, native XML tool calls, and assistant history handling. |
| [`archive/PLAN-WORKLOG2-revamp.md`](archive/PLAN-WORKLOG2-revamp.md) | Approved immutable worklog design, migration contract, and acceptance punchlist. |

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
