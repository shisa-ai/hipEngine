# hipEngine Documentation Index

Last updated: 2026-06-15

This directory contains the project architecture, validation, benchmarking, and
optimization notes for hipEngine. If you are new to the repo, start with
[`PLAN.md`](PLAN.md), then use the reading paths below for the task you are
working on.

## Start here

| Document | Use it for |
| --- | --- |
| [`PLAN.md`](PLAN.md) | Source of truth for architecture, plugin boundaries, phase roadmap, LoC budgets, and invariants. |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | Current implementation status, concrete milestones, and integration notes. |
| [`API.md`](API.md) | OpenAI-compatible FastAPI server usage, endpoint support, and current limitations. |
| [`OPTIMIZE.md`](OPTIMIZE.md) | Active optimization board for Qwen3.5-35B-A3B-PARO MoE; accepted/rejected/deferred candidates. |
| [`OPTIMIZE-DENSE.md`](OPTIMIZE-DENSE.md) | Active optimization board for Qwen3.6-27B-PARO dense; mirror lane structure to `OPTIMIZE.md`. |
| [`LESSONS-LEARNED.md`](LESSONS-LEARNED.md) | Local do-not-chase findings and recurring kernel/runtime pitfalls. |

## Validation and benchmarking

| Document | Use it for |
| --- | --- |
| [`TESTING.md`](TESTING.md) | RED/GREEN workflow, correctness oracles, fixture policy, and gate selection. |
| [`BENCHMARK.md`](BENCHMARK.md) | Benchmark protocol, required evidence fields, correctness thresholds, and artifact format. |
| [`THEROCK.md`](THEROCK.md) | Retained TheRock ROCm setup, `gfx110X-all` package choice, verification commands, and ROCm 7.14 regression notes. |
| [`../benchmarks/README.md`](../benchmarks/README.md) | Current benchmark rollup, source-lineage targets, external baselines, and diagnostic rows. |
| [`../benchmarks/CHANGELOG.md`](../benchmarks/CHANGELOG.md) | Reverse-chronological summary of benchmark rollup updates. |

## Kernels and performance model

| Document | Use it for |
| --- | --- |
| [`KERNELS.md`](KERNELS.md) | Kernel catalog, source-lineage drift workflow, Qwen/PARO path map, JIT cache gotchas, and build profiles. |
| [`ROOFLINE.md`](ROOFLINE.md) | RDNA3 / W7900 roofline model, occupancy rules, decision tree, and rejected hardware-level approaches. |
| [`RELAXED.md`](RELAXED.md) | Strict/exact vs opt-in relaxed precision policy, per-kernel savings candidates, and relaxed-mode backlog. |
| [`MARLIN.md`](MARLIN.md) | Marlin-K / PARO W4 layout plan and porting context. |
| [`GGUF_DECODE_REPACK.md`](GGUF_DECODE_REPACK.md) | P9.H2 qwen35moe GGUF decode-side replacement layout, memory budget, and acceptance plan. |
| [`TUNING-gguf.md`](TUNING-gguf.md) | Active GGUF performance tuning playbook, baseline refresh protocol, and lane backlog. |
| [`source_lineage.json`](source_lineage.json) | Machine-readable parent-file manifest for `scripts/check_lineage.py`. |

## Feature plans

| Document | Use it for |
| --- | --- |
| [`CONCURRENCY.md`](CONCURRENCY.md) | c>N serving readiness, diagnostic evidence, and server/scheduler/kernel/KV punchlist. |
| [`SAMPLING.md`](SAMPLING.md) | Normal sampling parameter support plan, sampler-state contract, and CPU/GPU rollout tracks. |
| [`AGENTIC.md`](AGENTIC.md) | Serving features for local agent harnesses built on top of sampling/decode-state primitives. |
| [`TENSOR_PARALLEL.md`](TENSOR_PARALLEL.md) | Tensor-parallel serving design gate, current disabled manifest contract, and multi-GPU validation plan. |
| [`PREFILL.md`](PREFILL.md) | Native prefill implementation plan and compact/prompt execution details. |
| [`KVCACHE.md`](KVCACHE.md) | KV cache ABI, policy notes, quantization path, and long-context considerations. |
| [`DFLASH.md`](DFLASH.md) | DFlash draft-model speculative decode plan. |
| [`MTP.md`](MTP.md) | Multi-token prediction plan. |
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
- **Before changing math or correctness-sensitive code:** read
  [`TESTING.md`](TESTING.md) and add or update a CPU-reference / fixture gate
  before relying on benchmark output.

Project-wide workflow rules live in [`../AGENTS.md`](../AGENTS.md), and the
append-only cross-session journal is [`../WORKLOG.md`](../WORKLOG.md).
