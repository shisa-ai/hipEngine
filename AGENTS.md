# hipEngine - Agent Guide

hipEngine is a ROCm-native inference engine built around a clean Python host and the proven gfx1100 kernel lineage from `nano-vllm-amd`. See [docs/PLAN.md](docs/PLAN.md) for architecture, phase roadmap, and LoC budgets.

This `AGENTS.md` (`CLAUDE.md` symlinked) is read every session. It covers only ground rules that apply to every review / coding / benchmarking task. Activity-specific playbooks live in `docs/`.

Instruction precedence: if this file conflicts with platform / system / developer instructions, follow those first.

## Summary

- **Source of truth:** [docs/PLAN.md](docs/PLAN.md). Update it when architecture or phase plans move.
- **Cross-session handoff:** immutable files under `worklog/entries/`; root `WORKLOG.md` is a tracked navigation page and `WORKLOG-LEGACY.md` is frozen pre-cutoff history. Use `python3 scripts/worklog.py new/check/render`.
- **Testing discipline:** math changes are guilty until proven correct. Follow RED/GREEN where practical, and use `docs/TESTING.md` for fixture/oracle/gate details. Execution-profile contracts are normative in `docs/EXECUTION-PROFILES.md`: control/ownership is exact in every profile; arithmetic equality, deterministic repeatability, and batch-composition invariance are separate gates.
- **Evidence policy:** every performance claim carries model + quant + workload shape + physical host identity + hardware + exact command + result + correctness gate. No exceptions (see `docs/PLAN.md` "Evidence Policy" and `docs/BENCHMARK.md`). Two machines with the same backend/GPU architecture are independent lanes: never report their absolute rates as an old→new comparison unless one declared same-host protocol measured both.
- **No benchmark gaming.** Tuning a metric to the specific inputs being measured is an INVALID benchmark, not a win. Concretely: never hardcode token IDs / candidate-id reranks / prompt-conditioned branches that lift acceptance, top-1, or speed on the fixed prompt(s) under test. Optimize the model/kernels, not the score. Acceptance/speed/quality for speculative or sampling paths must be validated on the **full multi-prompt mtp-bench category suite** (`benchmarks/prompts/mtpbench-code-general-ja.jsonl`, all of `code`/`general_en`/`general_ja`/`mixed_ja_en`) plus category-heldouts, never a single fixed prompt. MTP speedup claims require a true no-MTP autoregressive baseline from the same benchmark protocol; verifier-derived `off`/`B0` rows are diagnostic only. Single-prompt-overfit numbers are not retainable and any prior rows derived from them are marked INVALID. See `docs/BENCHMARK.md` "Anti-gaming".
- **Benchmark rollup stays current.** Every retained benchmark updates `benchmarks/README.md` (`Last updated` plus table row), `benchmarks/CHANGELOG.md` (dated one-liner with old→new metric, % delta, reason, artifact/source), and a compact artifact under `benchmarks/results/`.
- **The root `README.md` is a public product page, never a worklog.** Keep it concise, model-first, and useful to prospective users. It may show current result tables and brief user-visible caveats; it must not contain implementation diaries, optimization history, kernel/planner internals, candidate ladders, exact benchmark commands, evidence inventories, or internal blockers. Put those in `worklog/entries/`, `benchmarks/results/`, `benchmarks/CHANGELOG.md`, or `benchmarks/HISTORY.md`. Exported benchmark prose must pass `scripts/sync_benchmark_readme.py --check`.
- **Public-facing prose is self-contained.** Write for a reader with no session or worklog context. State current behavior and the comparison basis directly. Do not use unexplained campaign labels, commit shorthand, or backward-looking phrases such as “retained,” “remains non-regressive,” or “supersedes.” Include history only when it is the subject and introduce it explicitly. Use emphasis only for clear leaders, recommendations, or headline results.
- **Performance wins are first-class.** Every measured, exact, non-regressive performance improvement is kept and promoted to the default path unless there is a concrete blocker recorded in a durable worklog entry. Production-profile T1/T2 arithmetic wins are also retainable when the complete same-suite profile quality/task gate passes; they are not required to match strict generated IDs across widths. Cycle-wall, verified sub-window, launch-count, and H2D/D2H reductions count even when same-session AR variance hides the headline ratio; microseconds compound.
- **Refactor debt is tracked.** Temporary flags, rejected paths, duplicate dispatch routes, and fallback chains that should disappear after the optimal path is proven go in `docs/REFACTOR.md`. Add to it while the context is fresh instead of relying on future archeology.
- **Correctness gate for any new/ported kernel:** KL ≤ 0.05 AND top-1 agreement ≥ 90% vs `kernels/cpu_reference/` on fixture inputs is the outer smoke/safety floor. Strict variants additionally satisfy their declared exact/parent-parity RED contract. Production-profile variants satisfy the tighter calibrated mean/tail/max KL, top-1, deterministic/isolation, BF16-relative, and task gates in `docs/EXECUTION-PROFILES.md`; the broad floor alone cannot promote a production default.
- **Default hardware:** AMD Radeon Pro W7900, gfx1100/RDNA3. Claims about other backends require the corresponding hardware or are marked explicitly unverified.
- **Kernel work happens in this tree.** hipEngine is not a thin port of `~/amd-gpu-tuning/`; it is substantively different (torch-free runtime, four-axis registry, `KVLiveSpans` ABI, verifier-shaped kernels). New kernels, fused variants, small-batch/verifier-shaped kernels, micro-tuning, and `rocprofv3` iteration loops all live here under `kernels/<backend>/` with a strict exact/parent-parity RED test or a production-profile numerical RED test, plus the applicable correctness gate and a registered strict fallback. `~/amd-gpu-tuning/` and `nano-vllm-amd` remain read-only *references* for kernel lineage, prior evidence, and the device-code gotcha catalog — cite source file + commit when porting an idea, but do the development and measurement in-tree.
- **Kernel catalog must stay current.** Before any kernel port, check `docs/KERNELS.md` and run `scripts/check_lineage.py`; update the catalog/path map if parent kernels or dispatch changed.

## Architectural Invariants

Do not drift these casually. They define what hipEngine is.

- **Torch-free runtime.** `import torch` is **not** allowed in any module reached by `hipengine.LLM.generate()`. Torch lives behind the optional `hipengine[torch]` extra and appears only as a dlpack bridge at the user boundary. Adding `import torch` anywhere on the hot path is an architectural change, not a refactor.
- **Four-axis plugin registry.** Kernels are keyed by `(backend, layer, quant, variant)`. Models, quant schemes, and layers are plugins. **Never** add `if backend == "hip_gfx1100"` or `if quant == "..."` branches in dispatch / engine / model code; register against a registry key instead. See `docs/PLAN.md` "Extensibility Design" for mechanics.
- **Fused kernels require a strict unfused fallback.** Every fused composite (`rmsnorm+rotate`, `gate_combine_residual`, …) must have a registered strict unfused chain under its primitives. Strict fused variants meet their declared exact/parent-parity contract; production variants may reassociate only after the full profile gate.
- **Kernel bodies take raw device pointers.** `__global__` signatures use `void*` / typed pointers, never `torch::Tensor`. Only the host-side launch wrappers convert.
- **`KVLiveSpans` is the attention kernel ABI, not a DMS-only concept.** Every paged-KV-write and attention-decode kernel reads `(base_offsets, live_counts, token_positions, evict_mask)`. Dense policies fill it uniformly; DMS/H2O/SnapKV fill it variably. Do not shortcut to `(block_table, context_len)`.
- **Backend tree is a peer structure.** `kernels/hip_gfx1100/`, `kernels/hip_gfx1151/`, `kernels/cuda_sm86/`, `kernels/cpu_reference/` are siblings. There is no "AMD directory".

## Key Files

| Path | Purpose |
| --- | --- |
| `docs/PLAN.md` | Architecture, phase roadmap, LoC budgets, extensibility design. |
| `docs/BENCHMARK.md` | Benchmark protocols, baselines to beat, correctness gate, artifact/rollup format. |
| `docs/TESTING.md` | RED/GREEN workflow, correctness oracles, fixture policy, validation matrix. |
| `docs/KERNELS.md` | Kernel catalog, source-lineage drift workflow, Qwen3.5/PARO optimal path map, port playbook, JIT cache gotcha, build profiles. |
| `docs/source_lineage.json` | External parent-file manifest used by `scripts/check_lineage.py`. |
| `docs/ROOFLINE.md` | RDNA3 W7900 performance model: hardware, regimes, decision tree, what-not-to-chase. |
| `docs/EXECUTION-PROFILES.md` | Normative strict/production/batch-invariant contracts, numerical gates, exact ownership semantics, and registry resolution policy. |
| `docs/PRODUCTION-NUMERICS-CAMPAIGN.md` | Approved evaluator, calibration, historical-candidate, c1, and c>N/A4 execution plan. |
| `docs/REFACTOR.md` | Cleanup ledger for dead flags, duplicate dispatch paths, and fallback code to remove after optimal paths are proven. |
| `AGENTS.md` / `CLAUDE.md` | Ground rules (this file). |
| `WORKLOG.md` | Tracked worklog navigation page. |
| `WORKLOG-LEGACY.md` | Byte-frozen pre-Worklog2 journal; never edit. |
| `worklog/entries/` | Immutable per-unit current decisions, results, blockers, and handoffs. |
| `worklog/README.md` / `scripts/worklog.py` | Worklog schema, commands, validator, renderer, and optional pre-commit hook. |
| `benchmarks/README.md` | Canonical topline scoreboard, platform freshness, protocols, artifacts, and root README exports. |
| `benchmarks/HISTORY.md` | Archived experiment rollup, superseded diagnostics, source-lineage targets, and external baselines. |
| `benchmarks/CHANGELOG.md` | Reverse-chronological one-line history of benchmark rollup updates. |
| `benchmarks/results/` | Compact JSON artifacts for accepted/blocked/rejected benchmark attempts. |
| `pyproject.toml` | Package metadata and extras. Do not casually add hard deps. |

## Workflow

### Before Starting

1. `git status -sb` — note unrelated changes and leave them alone.
2. Read the relevant section of [docs/PLAN.md](docs/PLAN.md) and the latest relevant files under `worklog/entries/` (or run `python3 scripts/worklog.py render`). Read the `WORKLOG-LEGACY.md` tail only when pre-cutoff context matters.
3. For kernel / GPU work, confirm ROCm is alive:
   ```bash
   python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
   rocminfo | grep -E 'Name:|gfx'
   ```
4. Before any kernel port, read `docs/KERNELS.md` for the current catalog/path map and run `python3 scripts/check_lineage.py --kind kernel --diff stat` (or a narrower `--file` filter). Inspect DRIFT commits/diffs and parent WORKLOG/OPTIMAL evidence before copying code.
5. For a perf claim, define the baseline (model, quant, workload shape, hardware, command) from `docs/BENCHMARK.md` and `benchmarks/README.md` before making the change.

### During Work

- Keep changes scoped to one logical unit (one kernel family, one plugin, one doc, one phase milestone).
- Write or update the targeted test/fixture before implementation when behavior or math changes. If RED-first is impractical, record why in the unit's worklog entry.
- If a performance path is exact and same-suite non-regressive, make it the default and keep the old path as an opt-out only when rollback/bisection still has value. If a T1/T2 production path changes arithmetic, require exact control/ownership, same-schedule determinism, strict-teacher mean/tail/max KL and top-1 by category/shape/transition, applicable BF16-relative/task gates, and a registered strict fallback; cross-width generated-ID equality is diagnostic rather than universally binding. A small cycle-wall or sub-window win can be retained even if the aggregate ratio is flat within noise; document the distinction. If a path remains gated off, record the concrete blocker, not a vague "needs more evidence".
- When adding an env flag or retaining a default-off/default-on experiment, add or update a `docs/REFACTOR.md` entry that says when the flag/path should be removed.
- When adding tests that call HIP/ROCm runtime, `hipcc`, or GPU kernels, add an explicit HIP-availability guard (for example `ctypes.CDLL("libamdhip64.so")` + `pytest.skip`) so no-ROCm CI/publish runners skip them instead of failing release validation.
- For a substantial unit, create a unique entry with `python3 scripts/worklog.py new`, update it with non-trivial decisions, measurements, and dependency additions as work proceeds, and commit it with the unit.
- When profiling Python/ctypes JIT-built kernels with `rocprofv3`, prebuild the `.so` outside the profiler and run the profiled command with a precomputed compiler-version file plus `require_cached`; do not let the profiled process spawn `hipcc`/clang.
- For MTP profiling, do **not** wrap the prompt-suite/economics parent harness (`scripts/mtp-bench.py --mode hipengine-current` or `scripts/mtp_prompt_suite_economics.py`) in `rocprofv3`; it launches nested Python children and profiler/JIT state propagates into them. Use `scripts/mtp_verifier_rocprof.py` or profile the final `mtp_chain_e2e_smoke.py` child after a non-profiled cache warmup.
- Do not silently add `import torch`, `flash_attn`, or other CUDA-only deps to hot-path modules.
- Do not add `if backend == "..."` or `if quant == "..."` branches in engine / dispatch / model code.

### After Changes (before claiming done)

- Run the narrowest relevant test, then the applicable `docs/TESTING.md` gate before claiming done.
- **Do not automatically rerun a broad suite after an isolated failure.** If a completed broad run establishes that all other tests passed and the repair is scoped, rerun only the failing node(s), the changed test file, and any genuinely affected narrow bundle. Preserve the original broad-run result plus the repaired focused result as the validation evidence. Repeat the full suite only when the fix can affect previously passing tests (for example shared test infrastructure, collection/order/global-state behavior, or broadly shared production code), multiple unrelated failures indicate wider risk, a release protocol explicitly requires a fresh all-green run, or the user explicitly approves it.
- **An assigned task is standing approval for in-scope expensive validation.** State the concrete reason and expected duration before starting a test or benchmark expected to take more than five minutes, then proceed without another approval when it is necessary to complete the user-assigned task or campaign; use a background task when appropriate. Do not spend an equivalent expensive rerun when existing evidence plus focused repair is sufficient. Ask only before work that materially expands beyond the assigned scope or adds unusual destructive/reset-risk behavior.
- For a new / ported kernel: the applicable strict or production-profile correctness gate vs `kernels/cpu_reference/` + a `rocprofv3 --kernel-trace` entry showing the kernel ran under the expected name with plausible duration (`DurationNs` or `End_Timestamp - Start_Timestamp`). Production-profile evidence also names the execution profile and variant-manifest hash. See `docs/KERNELS.md` and `docs/EXECUTION-PROFILES.md`.
- For a perf change: record baseline + new measurements in the unit's immutable worklog entry with exact commands, emit/update the JSON artifact under `benchmarks/results/`, update `benchmarks/README.md` with the retained row and `Last updated` date, and add a dated changelog one-liner with old→new metric, % delta, reason, and artifact/source.
- Update `docs/PLAN.md` if architectural plans shifted.
- **Commit immediately** when the logical unit is complete and validation passes.

### Verification tiers

Run the narrowest tier for your change; escalate at milestone boundaries.

| Scope | What to run |
| --- | --- |
| Docs / process | Re-read the changed file end-to-end; no GPU run needed. |
| Code / registry / dispatch | The narrowest relevant `pytest` + applicable CPU deterministic bundle (see `docs/TESTING.md`). |
| New or ported kernel | Strict exact/parent-parity or production-profile numerical RED gate + CPU-reference outer gate + `rocprofv3 --kernel-trace` smoke (see `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/EXECUTION-PROFILES.md`). |
| Perf claim | Re-run the exact benchmark command from `docs/BENCHMARK.md` on stated hardware; record both runs in the unit's worklog entry. |
| Milestone closure | One full `uv run pytest -v` + the phase's named perf target vs prior baseline. If that completed run has isolated failures, apply the focused-repair rule above rather than automatically repeating the full suite. |

## Git Discipline

Explicit, auto-commit-after-validation. Many small, atomic, working-state commits with clear provenance — not fewer larger ones.

### Commit Timing

- **Commit immediately** after a logical unit is complete and validation passes. Do not ask, do not wait to be asked, and do not start the next logical task until the previous validated unit is committed.
- Include related handoff docs in the same unit (a change that needed a worklog entry or a `docs/PLAN.md` update commits them together).
- Always commit the new `worklog/entries/<unique-entry>.md` path with the logical unit that required it. Run `python3 scripts/worklog.py check` before staging/committing; never edit, rename, or delete a committed entry.
- Do not commit mid-task while exploring, debugging, or in a broken state.
- Docs, plans, repo-setup, and dependency additions are first-class logical units.

### Commit Mechanics (hard rules)

- **Never** use `git add .`, `git add -A`, or `git commit -a`.
- **Never** revert, checkout, or restore files you did not modify for the current task.
- **Always** stage files explicitly: `git add <path1> <path2> …`.
- **Always** verify before committing:
  ```bash
  git status -sb
  git diff --staged --name-only
  git diff --staged
  ```
- If unrelated changes or staged files you didn't create exist, leave them alone — another agent or the human owns them.

### Commit Messages

```
type: short summary (imperative, ≤ 72 chars)

- Non-obvious context
- Source commit when porting (e.g. nano-vllm-amd@f3a1c2e)
- Correctness / perf evidence when relevant
```

Prefixes: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `perf:`, `port:` (upstream lineage), `kernel:` (kernel edits). **No bylines** — no `Co-authored-by`, no agent attribution, no generated-by footers.

### Never Committed

- Model weights, `*.safetensors` outside fixtures
- Compiled `.so` / JIT caches, `rocprofv3` dumps, raw benchmark logs
- Local env / secrets, Python caches
- Vendored upstream repos (nano-vllm-amd, FastDMS, etc. — referenced by absolute path)

### Never Discard Others' Work

Do not run `git restore`, `git checkout --`, `git reset --hard`, `git clean -fd`, `rm -rf` across tracked paths, or bulk rewrites (aggressive formatters, mass import reordering) unless the user explicitly asks.

## Coordination

Working tree is shared state. Other agents or the human may be editing concurrently.

- **High-conflict files:** `AGENTS.md`, `CLAUDE.md`, `docs/PLAN.md`, `docs/BENCHMARK.md`, `docs/TESTING.md`, `docs/KERNELS.md`, `docs/IMPLEMENTATION.md`, `pyproject.toml`, `scripts/worklog.py`, `worklog/README.md`, `hipengine/kernels/registry.py`, `hipengine/quant/registry.py`, `hipengine/models/registry.py`, `hipengine/dispatch/fusion.py`, `hipengine/core/*`.
- Same-file contention: stop and coordinate. The designated agent stages and commits their scoped hunks first to unblock others.
- Worklog entry filenames are unique and should not conflict. A committed entry is immutable; correct it with a new `decision` or `checkpoint` entry that links the superseded path.
- `WORKLOG-LEGACY.md` is byte-frozen and manifest-checked. If an old branch carries an append to pre-cutoff `WORKLOG.md`, the designated merge owner converts its missing material into a new immutable entry instead of changing legacy.
- Do not clean up another agent's benchmark outputs, staged files, or local artifacts unless the task explicitly asks for that cleanup.

## External Reference Repos

Read-only peers under `/home/lhl/`. Do not edit as part of a hipEngine task. When porting, record source file + commit in the commit message. If an external reference disagrees with `docs/PLAN.md`, `docs/PLAN.md` wins unless we explicitly decide the reference is correct and update `docs/PLAN.md`.

- `~/amd-gpu-tuning/` — parent workspace; kernel lineage reference, benchmark history, `LESSONS-LEARNED.md`. Read-only; kernel development now happens in this tree.
- `~/amd-gpu-tuning/nano-vllm-amd/` — kernel source of truth for the Phase-0 port.
- `~/FastDMS/` — DMS reference (Phase 4).
- `~/FastKMS/` — DFlash speculative decode reference.
- `~/kvcache-quantization-research/` — AQUA / HIGGS / DMS stacking research.

## Handling Blockers

| Situation | Action |
| --- | --- |
| ROCm env appears corrupted | Record symptoms in a new worklog entry before any restore; follow the `~/amd-gpu-tuning` `therock` restore commands if clearly required. |
| Kernel hangs with GPU at 0%, no error | Stale JIT cache. See `docs/KERNELS.md` "JIT cache gotcha". |
| `rocprofv3` reports unexpected kernel | Registry / dispatch bug, not a kernel bug. Check `fusion.plan()` output before touching the kernel. |
| Math change lacks an oracle/test | Stop and add a CPU-reference/golden fixture first, or record an explicit no-RED rationale in the unit's worklog entry. |
| KL / top-1 regression after a kernel edit | Stop, add a fixture that captures and localizes the failure, and compare against the declared profile gate. Never land a strict perf win that breaks strict parity or a production win that fails any binding mean/tail/category/task threshold; do not relabel state/control bugs as numerical drift. |
| Kernel micro-opt shows neutral / negative results | Re-audit the rocprof kernel-family / launch breakdown in-tree (e.g. `scripts/mtp_verifier_rocprof.py`) before more tweaks; consult `~/amd-gpu-tuning/` evidence for context, but do not keep tweaking blindly. |
| Merge conflict in a high-conflict file | Stop and coordinate. Do not force-stage or revert. |
| Unclear whether a change crosses a plugin-registry boundary | Check `docs/PLAN.md` "Extensibility Design" first; if still unclear, ask the human lead. |
| Unrelated files changed in the worktree | Leave them. Another agent or the human owns them. |

## Communication

- Lead with the substantive finding or result, not just what command was run.
- Distinguish measured from inferred: "measured X tok/s on Y hardware with Z command" vs "expected to be X based on Y".
- If work is still in progress, state the current concrete result or explicitly say there is no result yet.
