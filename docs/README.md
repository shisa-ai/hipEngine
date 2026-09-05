# hipEngine Documentation Index

Last updated: 2026-08-31

This directory contains the project architecture, validation, benchmarking, and
optimization notes for hipEngine. If you are new to the repo, start with
[`PLAN.md`](PLAN.md), then use the reading paths below for the task you are
working on.

## Start here

| Document | Use it for |
| --- | --- |
| [`PLAN.md`](PLAN.md) | Source of truth for architecture, plugin boundaries, phase roadmap, LoC budgets, and invariants. |
| [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) | Normative strict/production/batch-invariant contracts, exact ownership rules, numerical gates, and registry resolution policy. |
| [`PRODUCTION-NUMERICS-CAMPAIGN.md`](PRODUCTION-NUMERICS-CAMPAIGN.md) | Approved evaluator, calibration, historical-recovery, c1, and c>N/A4 campaign. |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | Current implementation status, concrete milestones, and integration notes. |
| [`API.md`](API.md) | OpenAI-compatible FastAPI server usage, endpoint support, and current limitations. |
| [`SOL-OPTIMIZATION.md`](SOL-OPTIMIZATION.md) | Closed first-generation gfx1151/gfx1100 PARO/GGUF optimization ledger (`SOL-R0`-`R9`, `SOL-E1`/`E2`); its concurrency premise is superseded by [`CONCURRENCY.md`](CONCURRENCY.md) and it is retained as a dated record. |
| [`OPTIMIZE.md`](OPTIMIZE.md) | Active optimization board for Qwen3.5-35B-A3B-PARO MoE; accepted/rejected/deferred candidates. |
| [`OPTIMIZE-DENSE.md`](OPTIMIZE-DENSE.md) | Active optimization board for Qwen3.6-27B-PARO dense; mirror lane structure to `OPTIMIZE.md`. |
| [`QWEN35-08B-GFX1151-VULKAN-PARITY.md`](QWEN35-08B-GFX1151-VULKAN-PARITY.md) | Active Radeon 8060S campaign to profile every Qwen3.5-0.8B dense GGUF module and match or beat llama.cpp Vulkan before 27B transfer. |
| [`QWEN36-27B-GGUF-7900XTX.md`](QWEN36-27B-GGUF-7900XTX.md) | RX 7900 XTX campaign to eliminate GGUF weight-layout duplication and beat same-card llama.cpp HIP/Vulkan in speed and memory. |
| [`QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md`](QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md) | Planned gfx1100/gfx1151 campaign for exact Qwen3.8-27B `UD-Q4_K_M`: dense Q3/IQ codec support, operation-complete strict execution, and same-host Q4_K_M plus llama.cpp performance gates. |
| [`GFX1100-SHAPE-AWARE-GEMV-CAMPAIGN.md`](GFX1100-SHAPE-AWARE-GEMV-CAMPAIGN.md) | Planned shape-aware gfx1100 GEMV campaign seeded by Qingming: exact alpha/beta local128/SPLIT4 screening, cache-regime protocol, RX 7900 XTX relative comparison, and independent W7900 promotion gates. |
| [`QWEN38-Q4KM-MTP-ACCEPTANCE.md`](QWEN38-Q4KM-MTP-ACCEPTANCE.md) | E0-complete gfx1151 physical-C3 decode-economics campaign: current K3 is 0.8865x AR; adjudicate physical activation, then exact multi-row proposal-head reuse, true R12/R16 target amortization, and oracle-gated fixed K4. Appendix analyzes a DFlash2 revival. |
| [`QWEN38-INT8-KV-CONTINUOUS.md`](QWEN38-INT8-KV-CONTINUOUS.md) | Next INT8 KV campaign: artifact-scoped admission, compact no-mirror c>N prefill/decode, complete memory accounting, and resident lifecycle promotion. |
| [`LESSONS-LEARNED.md`](LESSONS-LEARNED.md) | Local do-not-chase findings and recurring kernel/runtime pitfalls. |
| [`PLAN-WORKLOG2-revamp.md`](PLAN-WORKLOG2-revamp.md) | Approved immutable worklog design, migration contract, and acceptance punchlist. |

## Validation and benchmarking

| Document | Use it for |
| --- | --- |
| [`TESTING.md`](TESTING.md) | RED/GREEN workflow, correctness oracles, fixture policy, and gate selection. |
| [`BENCHMARK.md`](BENCHMARK.md) | Benchmark protocol, required evidence fields, correctness thresholds, and artifact format. |
| [`PROCESS-EXPLORATION.md`](PROCESS-EXPLORATION.md) | Optional methodology for broader optimization searches, hypothesis beams, structural maturation, evaluation firewalls, and anti-overfitting gates. |
| [`THEROCK.md`](THEROCK.md) | Retained TheRock ROCm setup, `gfx110X-all` package choice, verification commands, and ROCm 7.14 regression notes. |
| [`DEBUG-GFX1151-STALL.md`](DEBUG-GFX1151-STALL.md) | Open gfx1151 128K prefill no-progress signature, eliminated hypotheses, KFD/MES debug plan, and upstream-report checklist. |
| [`../benchmarks/README.md`](../benchmarks/README.md) | Canonical topline scoreboard, platform freshness, exact protocols, artifacts, and refresh commands. |
| [`../benchmarks/HISTORY.md`](../benchmarks/HISTORY.md) | Archived experiment rollup, source-lineage targets, external baselines, and superseded diagnostics. |
| [`../benchmarks/CHANGELOG.md`](../benchmarks/CHANGELOG.md) | Reverse-chronological summary of benchmark rollup updates. |

## Kernels and performance model

| Document | Use it for |
| --- | --- |
| [`KERNELS.md`](KERNELS.md) | Kernel catalog, source-lineage drift workflow, Qwen/PARO path map, JIT cache gotchas, and build profiles. |
| [`ROOFLINE.md`](ROOFLINE.md) | RDNA3 / W7900 roofline model, occupancy rules, decision tree, and rejected hardware-level approaches. |
| [`RELAXED.md`](RELAXED.md) | Historical relaxed-mode inventory and first changed-arithmetic kernel provenance; superseded as normative policy by `EXECUTION-PROFILES.md`. |
| [`MARLIN.md`](MARLIN.md) | Marlin-K / PARO W4 layout plan and porting context. |
| [`QUANTS.md`](QUANTS.md) | GGUF tensor-type coverage, Qwen3.5 quality cliffs, Laguna S 2.1 quant targets, hardware headroom, and BF16 K/V capacity math. |
| [`OPTIMIZE-KERNEL-IQ2_XS.md`](OPTIMIZE-KERNEL-IQ2_XS.md) | Active IQ2_XS decode/prefill bottleneck analysis, priority list, tuning order, precedent, and Laguna acceptance gates. |
| [`GGUF_DECODE_REPACK.md`](GGUF_DECODE_REPACK.md) | P9.H2 qwen35moe GGUF decode-side replacement layout, memory budget, and acceptance plan. |
| [`TUNING-gguf.md`](TUNING-gguf.md) | Active GGUF performance tuning playbook, baseline refresh protocol, and lane backlog. |
| [`source_lineage.json`](source_lineage.json) | Machine-readable parent-file manifest for `scripts/check_lineage.py`. |

## Feature plans

| Document | Use it for |
| --- | --- |
| [`CONCURRENCY2.md`](CONCURRENCY2.md) | Generation-2 lifecycle, scheduler, global pool/prefix, c1-c32, DMS, optional tiering, and the current executable completion/blocker audit. |
| [`CONCURRENCY.md`](CONCURRENCY.md) | Legacy retained c=N kernel/resident-runner roadmap, readiness ledger, and evidence history. |
| [`NATIVE_SPEC_CYCLE.md`](NATIVE_SPEC_CYCLE.md) | Canonical N0-N5 speculative-cycle milestone glossary, ownership boundaries, current W7900/gfx1151 scorecard, and evidence index. |
| [`SAMPLING.md`](SAMPLING.md) | Normal sampling parameter support plan, sampler-state contract, and CPU/GPU rollout tracks. |
| [`AGENTIC.md`](AGENTIC.md) | Serving features and functional contract for local agent harnesses built on top of sampling/decode-state primitives. |
| [`AGENTIC-OPT.md`](AGENTIC-OPT.md) | Active gfx1100 agent-serving status, limitations, optimization priorities, and coding-agent benchmark plan. |
| [`TENSOR_PARALLEL.md`](TENSOR_PARALLEL.md) | Tensor-parallel serving design gate, current disabled manifest contract, and multi-GPU validation plan. |
| [`PREFILL.md`](PREFILL.md) | Native prefill implementation plan and compact/prompt execution details. |
| [`KVCACHE.md`](KVCACHE.md) | KV cache ABI, policy notes, quantization path, and long-context considerations. |
| [`DMS.md`](DMS.md) | External DMS architecture, exact-Q4 training campaign, sidecar size/timing/results, reproduction commands, and production punchlist. |
| [`DFLASH.md`](DFLASH.md) | DFlash draft-model speculative decode plan. |
| [`MTP.md`](MTP.md) | Multi-token prediction implementation history, economics, and provider design. |
| [`MTP-FIX.md`](MTP-FIX.md) | Active campaign to make MTP safe and useful across real contexts, lifecycle events, APIs, load, quality, and rollout. |
| [`GGUF.md`](GGUF.md) | GGUF loading / comparison notes. |

## Common reading paths

- **Before changing architecture or dispatch:** read [`PLAN.md`](PLAN.md), then
  [`IMPLEMENTATION.md`](IMPLEMENTATION.md), and check [`OPTIMIZE.md`](OPTIMIZE.md)
  if the change affects a tracked candidate.
- **Before porting or editing a kernel:** read [`KERNELS.md`](KERNELS.md), run
  `python3 scripts/check_lineage.py --kind kernel --diff stat`, and use
  [`ROOFLINE.md`](ROOFLINE.md) to decide whether the proposed change matches the
  measured bottleneck.
- **Before making a performance claim:** read [`BENCHMARK.md`](BENCHMARK.md),
  verify the ROCm environment against [`THEROCK.md`](THEROCK.md) for W7900
  TheRock rows,
  update [`../benchmarks/README.md`](../benchmarks/README.md) and
  [`../benchmarks/CHANGELOG.md`](../benchmarks/CHANGELOG.md), and write a compact
  artifact under [`../benchmarks/results/`](../benchmarks/results/).
- **Before opening a less-bounded optimization search:** read
  [`PROCESS-EXPLORATION.md`](PROCESS-EXPLORATION.md), freeze the evaluator and
  generalization envelope, then seed genuinely distinct hypothesis families.
- **Before changing math or correctness-sensitive code:** read
  [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) and
  [`TESTING.md`](TESTING.md), declare the applicable strict/production/
  batch-invariant contract, and add or update a CPU-reference / fixture gate
  before relying on benchmark output.

Project-wide workflow rules live in [`../AGENTS.md`](../AGENTS.md). Current
immutable handoff entries live under
[`../worklog/entries/`](../worklog/entries/); [`../WORKLOG.md`](../WORKLOG.md)
links the entry format, local renderer, and frozen pre-Worklog2 history.
