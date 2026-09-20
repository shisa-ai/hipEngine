# hipEngine Optimization and Evidence Rules

Last updated: 2026-09-21

**Scope.** This file binds three activities: kernel work, measured performance
claims, and benchmark rows. It is the process those activities follow.

**It does not govern product behavior.** Server/API semantics, feature
enablement, default selection, error shapes, and ordinary application code are
governed by [`../AGENTS.md`](../AGENTS.md). Never import a gate from this file
into a product decision. A promotion gate here describes what it takes to
*assert a measured result*; it never describes what it takes to *ship a
feature*.

**Not a campaign doc.** [`OPTIMIZE.md`](OPTIMIZE.md),
[`OPTIMIZE-DENSE.md`](OPTIMIZE-DENSE.md),
[`SOL-OPTIMIZATION.md`](SOL-OPTIMIZATION.md), [`TUNING-gguf.md`](TUNING-gguf.md),
and the `QWEN*-CAMPAIGN.md` files track candidates and results for specific
models and hosts. This file is the rules they follow.

## 1. When this file applies

Read it when your task is one of these. Otherwise skip it.

| Trigger | What it pulls in |
| --- | --- |
| Writing, porting, or editing a kernel under `kernels/<backend>/` | §3 lineage, §4 correctness gate, §7 profiler smoke |
| Changing arithmetic on a path that can become a default | §4 execution-profile gate, strict fallback |
| Asserting a number — tok/s, % faster, GiB, acceptance rate | §2 evidence policy, §6 artifact and rollup |
| Adding or updating a row under `benchmarks/` | §2, §5 anti-gaming, §6 |
| Running a tuning or optimization campaign | §5 anti-gaming, §4 promotion |

**Not triggers.** Enabling a route. Flipping a default. Fixing a server bug.
Adding an endpoint. Making an already-implemented path reachable. Choosing a
parameter name. Removing a stale guard. Those are product work; see `AGENTS.md`
"What Evidence Is For" and "Product Defaults", and do not open a benchmark
campaign to justify them.

## 2. Evidence policy (claims only)

Every **performance claim** carries model + quant + workload shape + physical
host identity + hardware + exact command + result + correctness gate. No
exceptions. The full field list is in [`PLAN.md`](PLAN.md) "Evidence Policy";
protocols are in [`BENCHMARK.md`](BENCHMARK.md).

Two machines with the same backend/GPU architecture are **independent lanes**.
Never report their absolute rates as an old→new comparison unless one declared
same-host protocol measured both.

This policy is inherited from `LESSONS-LEARNED.md`: fast rows are invalid until
output sanity proves they are real. It binds what you *assert*. It does not bind
what runs, what ships, or what the lead decides.

**Default hardware:** AMD Radeon Pro W7900, gfx1100/RDNA3. Claims about other
backends require the corresponding hardware or are marked explicitly unverified.

## 3. Kernel work happens in this tree

hipEngine is not a thin port of `~/amd-gpu-tuning/`; it is substantively
different (torch-free runtime, four-axis registry, `KVLiveSpans` ABI,
verifier-shaped kernels). New kernels, fused variants, small-batch and
verifier-shaped kernels, micro-tuning, and `rocprofv3` iteration loops all live
here under `kernels/<backend>/` with a strict exact/parent-parity RED test or a
production-profile numerical RED test, plus the applicable correctness gate and
a registered strict fallback.

`~/amd-gpu-tuning/` and `nano-vllm-amd` are read-only *references* for kernel
lineage, prior evidence, and the device-code gotcha catalog. Cite source file +
commit when porting an idea; do the development and measurement in-tree.

**Before any kernel port:** read [`KERNELS.md`](KERNELS.md) for the current
catalog/path map and run `python3 scripts/check_lineage.py --kind kernel --diff
stat` (or a narrower `--file` filter). Inspect DRIFT commits/diffs and parent
WORKLOG/OPTIMAL evidence before copying code, and update the catalog/path map if
parent kernels or dispatch changed.

**Before any perf claim:** define the baseline (model, quant, workload shape,
hardware, command) from [`BENCHMARK.md`](BENCHMARK.md) and
`benchmarks/README.md` *before* making the change.

**Confirm ROCm is alive** for kernel/GPU work:

```bash
python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
rocminfo | grep -E 'Name:|gfx'
```

## 4. Correctness gates and promotion

### 4.1 There is no "exactness contract"

**The contract is production correctness.** When optimizing, do **not**
automatically discard a candidate because it is not bit-identical to the
strict/exact parent. No BF16-flip-free or bit-exact requirement gates promotion.

The binding contract is [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md): exact
control/ownership in every profile, plus the calibrated production numerical
envelope (mean/p95/p99/max KL, top-1 by category, determinism, isolation, task
quality). A candidate that is not bit-exact **must be re-reviewed under those
production gates before any rejection**. `strict` is a debugging oracle, not the
promotion bar.

Rejecting a production-correct candidate for failing an exactness test it was
never required to pass is a review error, not a safety win.

### 4.2 The gates

| Gate | Applies to | Threshold |
| --- | --- | --- |
| Outer smoke/safety floor | Any new or ported kernel | KL ≤ 0.05 **and** top-1 agreement ≥ 90% vs `kernels/cpu_reference/` on fixture inputs |
| Strict contract | Variants declaring strict/parent parity | Their declared exact/parent-parity RED contract |
| Production profile | Arithmetic-changing production defaults | Calibrated mean/tail/max KL, top-1, deterministic/isolation, BF16-relative, and task gates in [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) |

The broad floor alone cannot promote a production default. Conversely, the
strict contract alone cannot block a production candidate that meets its own
profile gate.

### 4.3 Performance wins are first-class

Every measured, non-regressive performance improvement is kept and promoted to
the default path unless there is a **concrete blocker** recorded in a durable
worklog entry. There is no minimum percentage threshold — see
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) §2.1.

- Production-profile T1/T2 arithmetic wins are retainable when the complete
  same-suite profile quality/task gate passes. They are **not** required to
  match strict generated IDs across widths.
- Cycle-wall, verified sub-window, launch-count, and H2D/D2H reductions count
  even when same-session AR variance hides the headline ratio. Microseconds
  compound. Document the distinction between a sub-window win and a flat
  aggregate rather than discarding the former.
- If a path is exact and same-suite non-regressive, make it the default. Keep
  the old path as an opt-out only when rollback or bisection still has value.
- If a T1/T2 production path changes arithmetic, require exact
  control/ownership, same-schedule determinism, strict-teacher mean/tail/max KL
  and top-1 by category/shape/transition, applicable BF16-relative/task gates,
  and a registered strict fallback. Cross-width generated-ID equality is
  diagnostic, not universally binding.
- If a path stays gated off, record the **concrete blocker** — an observed
  failure, an unmet precondition, a resource conflict. Never "needs more
  evidence".

### 4.4 A win that production never selects is not a win

Before a performance change is done, confirm the shipping path actually takes
it: exercise it through `hipengine.LLM.generate()` or `hipengine serve` and
check the reported selected variant, sampler mode, and fallback reason. A number
measured on a route the default path never chooses is a diagnostic, not a
result, and it must be labeled as one. See `AGENTS.md` "Product Defaults".

## 5. No benchmark gaming

Tuning a metric to the specific inputs being measured is an **INVALID
benchmark**, not a win.

- Never hardcode token IDs, candidate-id reranks, or prompt-conditioned branches
  that lift acceptance, top-1, or speed on the fixed prompt(s) under test.
  Optimize the model and kernels, not the score.
- Acceptance, speed, and quality for speculative or sampling paths must be
  validated on the **full multi-prompt mtp-bench category suite**
  (`benchmarks/prompts/mtpbench-code-general-ja.jsonl`, all of `code` /
  `general_en` / `general_ja` / `mixed_ja_en`) plus category-heldouts — never a
  single fixed prompt.
- MTP speedup claims require a true no-MTP autoregressive baseline from the same
  benchmark protocol. Verifier-derived `off`/`B0` rows are diagnostic only.
- Single-prompt-overfit numbers are not retainable; prior rows derived from them
  are marked INVALID.
- Do not validate only at configured threshold values. Exercising a
  length-gated path at exactly 512 and 1024 because those are the thresholds
  measures the thresholds, not the path. Cover both sides and at least one
  unrelated value.

See [`BENCHMARK.md`](BENCHMARK.md) "Anti-gaming". For less-bounded searches,
freeze the evaluator and generalization envelope first —
[`PROCESS-EXPLORATION.md`](PROCESS-EXPLORATION.md).

## 6. Benchmark rollup stays current

Every retained benchmark updates all three:

1. `benchmarks/README.md` — `Last updated` plus the table row.
2. `benchmarks/CHANGELOG.md` — dated one-liner with old→new metric, % delta,
   reason, artifact/source.
3. `benchmarks/results/` — a compact JSON artifact.

Record baseline + new measurements with exact commands in the unit's immutable
worklog entry. Exported benchmark prose must pass
`scripts/sync_benchmark_readme.py --check`.

## 7. Profiling

- When profiling Python/ctypes JIT-built kernels with `rocprofv3`, prebuild the
  `.so` outside the profiler and run the profiled command with a precomputed
  compiler-version file plus `require_cached`. Do not let the profiled process
  spawn `hipcc`/clang.
- For MTP profiling, do **not** wrap the prompt-suite/economics parent harness
  (`scripts/mtp-bench.py --mode hipengine-current` or
  `scripts/mtp_prompt_suite_economics.py`) in `rocprofv3`; it launches nested
  Python children and profiler/JIT state propagates into them. Use
  `scripts/mtp_verifier_rocprof.py`, or profile the final
  `mtp_chain_e2e_smoke.py` child after a non-profiled cache warmup.

## 8. Verification tiers

| Scope | What to run |
| --- | --- |
| New or ported kernel | Strict exact/parent-parity or production-profile numerical RED gate + CPU-reference outer gate + `rocprofv3 --kernel-trace` smoke showing the kernel ran under the expected name with plausible duration (`DurationNs` or `End_Timestamp - Start_Timestamp`). Production-profile evidence also names the execution profile and variant-manifest hash. |
| Perf claim | Re-run the exact benchmark command from [`BENCHMARK.md`](BENCHMARK.md) on stated hardware; record both runs in the unit's worklog entry. |
| Capacity / does-it-fit | Tier-1 allocation probe (`scripts/gguf_capacity_probe.py`, ~3 min) at the target context; escalate to a full-prompt harness point ONLY to confirm a found bound or to certify completion at depth. See `benchmarks/HARNESSES.md` "Capacity testing". **Never bracket bounds with full prompts.** |

An assigned optimization task is standing approval for its in-scope expensive
validation; state the reason and expected duration, then proceed. See
`AGENTS.md` "After Changes".

## 9. Blockers

| Situation | Action |
| --- | --- |
| ROCm env appears corrupted | Record symptoms in a new worklog entry before any restore; follow the `~/amd-gpu-tuning` `therock` restore commands if clearly required. |
| Kernel hangs with GPU at 0%, no error | Stale JIT cache. See [`KERNELS.md`](KERNELS.md) "JIT cache gotcha". |
| `rocprofv3` reports unexpected kernel | Registry / dispatch bug, not a kernel bug. Check `fusion.plan()` output before touching the kernel. |
| Math change lacks an oracle/test | Add a CPU-reference/golden fixture first, or record an explicit no-RED rationale in the unit's worklog entry. |
| KL / top-1 regression after a kernel edit | Add a fixture that captures and localizes the failure, and compare against the **declared profile gate** for that variant — not against strict parity the variant never claimed. Never land a strict perf win that breaks strict parity, or a production win that fails a binding mean/tail/category/task threshold. Do not relabel state/control bugs as numerical drift. |
| Kernel micro-opt shows neutral / negative results | Re-audit the rocprof kernel-family / launch breakdown in-tree (e.g. `scripts/mtp_verifier_rocprof.py`) before more tweaks. Consult `~/amd-gpu-tuning/` evidence for context; do not keep tweaking blindly. |
| Candidate is production-correct but not bit-exact | Re-review under §4.1 production gates before rejecting. This is the default outcome for a good optimization, not an anomaly. |

## 10. Related documents

| Document | Use it for |
| --- | --- |
| [`../AGENTS.md`](../AGENTS.md) | Project-wide ground rules, product defaults, evidence scope. |
| [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) | Normative strict/production/batch-invariant contracts and numerical gates. §1.1 is the "missing evidence is not a runtime failure" rule. |
| [`BENCHMARK.md`](BENCHMARK.md) | Benchmark protocols, baselines to beat, artifact/rollup format, anti-gaming. |
| [`KERNELS.md`](KERNELS.md) | Kernel catalog, lineage drift workflow, path map, JIT cache gotcha, build profiles. |
| [`ROOFLINE.md`](ROOFLINE.md) | RDNA3 W7900 performance model, regimes, decision tree, what not to chase. |
| [`TESTING.md`](TESTING.md) | RED/GREEN workflow, oracles, fixture policy, gate selection. |
| [`PRODUCTION-NUMERICS-CAMPAIGN.md`](PRODUCTION-NUMERICS-CAMPAIGN.md) | Approved evaluator, calibration, and candidate execution plan. |
| [`PROCESS-EXPLORATION.md`](PROCESS-EXPLORATION.md) | Hypothesis beams, evaluation firewalls, anti-overfitting for broad searches. |
| [`LESSONS-LEARNED.md`](LESSONS-LEARNED.md) | Local do-not-chase findings and recurring pitfalls. |
