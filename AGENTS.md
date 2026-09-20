# hipEngine - Agent Guide

hipEngine is a ROCm-native inference engine built around a clean Python host and the proven gfx1100 kernel lineage from `nano-vllm-amd`. See [docs/PLAN.md](docs/PLAN.md) for architecture, phase roadmap, and LoC budgets.

This `AGENTS.md` (`CLAUDE.md` symlinked) is read every session. It covers only ground rules that apply to every review / coding / benchmarking task. Activity-specific playbooks live in `docs/`.

**If your task is kernel work, a measured performance claim, or a benchmark row, read [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) as well.** If it is not, you do not need that file, and its gates do not apply to you.

Instruction precedence: if this file conflicts with platform / system / developer instructions, follow those first.

## Decision Authority

The human lead owns product and production direction. Their decisions are directives, not hypotheses.

- **Execute, then report.** When the lead makes a product, default, or production call, make the change and report what it means. Do not open a pre-execution review of a decision they already made. Post-change validation still runs as this file requires — that verifies the implementation, not the decision.
- **Evidence obligations bind claims, not decisions.** The Evidence Policy governs what the agent asserts. It does not gate what the lead decides.
- **State disagreement once, and late.** If a decision looks wrong or a premise looks incomplete, say so once, briefly, after the work — or before it only when the action is irreversible, destructive, or materially wider in scope than the request. Then let the call stand.
- **"Let me check first" is not a neutral move.** It converts a directive into a negotiation and spends the lead's time re-deciding what they already decided. Before any pre-execution check, ask: would this change what I *do*, or only what I *say about it*? If only the latter, do the work first.
- **Reversible changes default to action.** A default flip, a flag, a doc fix, or a threshold change is reversible. Make it, and let the report carry the caveats.
- **Documentation outranks nothing here.** A normative gate in `docs/EXECUTION-PROFILES.md`, a `docs/REFACTOR.md` blocker note, or a calibration envelope describes the evidence behind a default. It does not overrule a lead decision to change that default. Record the new decision; do not litigate the old gate.

## What Evidence Is For

hipEngine's evidence discipline exists because a fast kernel row is worthless if the output is wrong. That is its job. It is **not** a general-purpose permission system, and using it as one is the most common way agents damage this project.

**Evidence governs:**

- **Claims.** Any number you assert — throughput, latency, memory, acceptance rate, quality — carries its full provenance. See [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) §2.
- **Arithmetic promotion.** Making a changed-arithmetic kernel path the default requires its execution-profile gate.
- **Automated gating.** Wherever a script, CI job, or agentic loop decides keep/kill with no human in the loop, the threshold *is* the decision — so it must be explicit, pre-registered, and measured.
- **Your own optimization choices.** Profile before you tune; prefer a measurement over a hunch.

**Evidence does not govern:**

- **Whether a feature is enabled.** Shipping a working implementation on the default path is a product decision, not a claim.
- **Whether an implemented path may run.** Absence of a benchmark row is not a finding. Untested is not failed.
- **Product, API, and server behavior.** Defaults, naming, error shapes, endpoint semantics, and UX are design decisions. They must be correct and tested; they do not need a benchmark artifact.
- **The lead's decisions.** See "Decision Authority" above.

If you are about to block, gate, revert, or default-off something *because evidence is missing*, you are in the second column. Run it instead.

## Product Defaults

Each of these names a failure that has actually happened in this repo.

- **Ship it on.** A working implementation lands **enabled on the default path**. A flag is a rollback lever for something already on, not a hiding place for something that has never been on. "Default off until qualified" requires a named concrete cause written where the flag is defined — an observed failure, an unmet precondition, a resource conflict, a missing dependency. "Not yet benchmarked" and "not yet qualified" are not causes.
- **A gate nothing can clear is a bug in the gate.** If you restrict a path, you own the route that lifts the restriction: name the command that would clear it. A guard justified only by "nobody has exercised this case" makes the case permanently unexercisable, because the guard is what prevents exercising it. If you cannot name the clearing command, do not add the guard.
- **Validate the space, not the gate point.** When behavior depends on a threshold — context length, batch width, prompt size, sequence count — exercise values on both sides of it and at least one value unrelated to it. Testing a length-gated path at exactly 512 and 1024 because those are the configured thresholds proves the thresholds, not the feature.
- **Works in the harness is not works.** A route reachable only from a benchmark script, a test fixture, or an explicit env var is not shipped. Before calling a change done, exercise it the way a user reaches it — `hipengine.LLM.generate()` or `hipengine serve` — and confirm the intended path actually ran (selected variant, sampler mode, fallback reason). A path that silently falls back in production while passing its own targeted test is a defect, not coverage.

## Ground Rules

- **Source of truth:** [docs/PLAN.md](docs/PLAN.md). Update it when architecture or phase plans move.
- **Cross-session handoff:** immutable files under `worklog/entries/`; root `WORKLOG.md` is a tracked navigation page and `WORKLOG-LEGACY.md` is frozen pre-cutoff history. Pre-cutoff journal history is also ported into `worklog/entries/` as entries with `worker: legacy` (mapping in `worklog/legacy-port-manifest.json`); the frozen file remains the canonical original. Use `python3 scripts/worklog.py new/check/render`.
- **Untested is not failed.** Presume implemented paths runnable within their input, resource, and execution contracts. Run representative workloads and targeted tests to find and fix defects, rather than blocking evaluation behind missing qualification. Do not require successful evaluation as a prerequisite for running that evaluation. `docs/EXECUTION-PROFILES.md` §1.1 is the normative statement and separates runnability, evaluation, and promotion.
- **Testing discipline:** follow RED/GREEN where practical; `docs/TESTING.md` has fixture/oracle/gate details. Claims and production arithmetic promotion still require the applicable correctness gates in [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md). Control/ownership, deterministic repeatability, arithmetic equality, and batch-composition invariance remain distinct contracts — do not collapse them.
- **Test naming and scope:** new/migrated modules use `test_<tier>_<subject>.py`; follow `docs/TESTING.md` "Test Naming and Discovery". Default pytest discovers the reviewed unit tier only. Use explicit file/node targets or `--suite <tier>` for other work; a full milestone run requires `--suite all`. Never classify cost by product-name substrings or introduce manual filename exclusion lists.
- **Flags are a cost, not a feature.** Prefer no flag. When one is genuinely warranted, it defaults to the behavior you want in production, and you add a `docs/REFACTOR.md` entry naming the concrete condition for removing it. The ledger entry is the price of the flag, not a licence to add one. Temporary flags, rejected paths, duplicate dispatch routes, and fallback chains that should disappear all go in `docs/REFACTOR.md` while the context is fresh.
- **The root `README.md` is a public product page, never a worklog.** Keep it concise, model-first, and useful to prospective users. It may show current result tables and brief user-visible caveats; it must not contain implementation diaries, optimization history, kernel/planner internals, candidate ladders, exact benchmark commands, evidence inventories, or internal blockers. Put those in `worklog/entries/`, `benchmarks/results/`, `benchmarks/CHANGELOG.md`, or `benchmarks/HISTORY.md`. Exported benchmark prose must pass `scripts/sync_benchmark_readme.py --check`.
- **Public-facing prose is self-contained.** Write for a reader with no session or worklog context. State current behavior and the comparison basis directly. Do not use unexplained campaign labels, commit shorthand, or backward-looking phrases such as “retained,” “remains non-regressive,” or “supersedes.” Include history only when it is the subject and introduce it explicitly. Use emphasis only for clear leaders, recommendations, or headline results.
- **Cleanup debt is inventoried, not remembered.** `audit/` mechanically inventories the flags, kernels, `docs/REFACTOR.md` entries, and campaign candidates that make up hipEngine's debt, and `audit/triage/` records what was decided about each. An inventory row is evidence, never a verdict — extractors are routinely wrong, and `NOT-DEBT` is a normal outcome. `python3 audit/audit.py check` is the gate: it fails when untriaged rows grow past `audit/budget.json`, so new debt cannot land without being triaged. Work rows with the `cleanup-triage` agent, and never raise the budget to get past the gate.
- **No hard-coded home directories in prose.** A path like `/home/<you>/llama.cpp` is correct for exactly one reader. Use `~/` for anything outside the repository and a repo-relative path (`.venv/bin/python`, `scripts/foo.py`) for anything inside it. `scripts/docs/check_docs.py` fails on `/home/<user>/` and `/Users/<user>/` in `AGENTS.md`, `docs/`, and `benchmarks/` prose; dated evidence under `docs/testing/`, `docs/examples/`, and `benchmarks/results/` is exempt because it records what something said rather than instructing.
- **Docs declare their own status.** Every file under `docs/` carries front-matter: `status: normative | current | closed | superseded` plus an `owns:` line. `normative` binds, `current` describes today's contract, `closed` is evidence from finished work, `superseded` names its replacement. Check the status before treating a document as binding — most of `docs/campaigns/` is closed history, not policy. Top-level `docs/` is the small set every agent should know exists; `reference/` holds live subsystem contracts, `campaigns/` closed campaign evidence, `model-cards/` per-model status, `archive/` superseded material. When you add or move a document, add the front-matter and run `python3 scripts/docs/check_docs.py --write` to refresh the generated indexes. `python3 scripts/docs/check_all.py` runs every documentation gate.
- **Kernel, performance, and benchmark rules live in [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md).** That file owns the evidence-claim fields, anti-gaming rules, correctness gates, promotion rules, lineage workflow, profiling recipes, and benchmark rollup. It applies to kernel work, measured claims, and benchmark rows — and to nothing else.

## Architectural Invariants

Do not drift these casually. They define what hipEngine is.

- **Torch-free runtime.** `import torch` is **not** allowed in any module reached by `hipengine.LLM.generate()`. Torch lives behind the optional `hipengine[torch]` extra and appears only as a dlpack bridge at the user boundary. Adding `import torch` anywhere on the hot path is an architectural change, not a refactor.
- **Four-axis plugin registry.** Kernels are keyed by `(backend, layer, quant, variant)`. Models, quant schemes, and layers are plugins. **Never** add `if backend == "hip_gfx1100"` or `if quant == "..."` branches in dispatch / engine / model code; register against a registry key instead. See `docs/PLAN.md` "Extensibility Design" for mechanics.
- **Fused kernels require a strict unfused fallback.** Every fused composite (`rmsnorm+rotate`, `gate_combine_residual`, …) must have a registered strict unfused chain under its primitives. Strict fused variants meet their declared exact/parent-parity contract. Production candidates may reassociate during evaluation; default promotion requires the full profile gate.
- **Kernel bodies take raw device pointers.** `__global__` signatures use `void*` / typed pointers, never `torch::Tensor`. Only the host-side launch wrappers convert.
- **`KVLiveSpans` is the attention kernel ABI, not a DMS-only concept.** Every paged-KV-write and attention-decode kernel reads `(base_offsets, live_counts, token_positions, evict_mask)`. Dense policies fill it uniformly; DMS/H2O/SnapKV fill it variably. Do not shortcut to `(block_table, context_len)`.
- **Backend tree is a peer structure.** `kernels/hip_gfx1100/`, `kernels/hip_gfx1151/`, `kernels/cuda_sm86/`, `kernels/cpu_reference/` are siblings. There is no "AMD directory".

## Key Files

| Path | Purpose |
| --- | --- |
| `docs/README.md` | Generated index of the docs tree, with each document's status and what it owns. Start here when you do not know which document you need. |
| `docs/PLAN.md` | Architecture, phase roadmap, LoC budgets, extensibility design. |
| `docs/OPTIMIZATION.md` | **Rules for kernel work, performance claims, and benchmark rows** — evidence fields, anti-gaming, correctness gates, promotion, lineage, profiling. Scoped: it does not govern product behavior. |
| `docs/EXECUTION-PROFILES.md` | Normative strict/production/batch-invariant contracts, numerical gates, exact ownership and failure-containment semantics, registry resolution policy. §1.1 is the "missing evidence is not a runtime failure" rule. |
| `docs/TESTING.md` | RED/GREEN workflow, correctness oracles, fixture policy, validation matrix. |
| `docs/BENCHMARK.md` | Benchmark protocols, baselines to beat, correctness gate, artifact/rollup format. |
| `docs/KERNELS.md` | Kernel catalog, source-lineage drift workflow, Qwen3.5/PARO optimal path map, port playbook, JIT cache gotcha, build profiles. |
| `docs/ROOFLINE.md` | RDNA3 W7900 performance model: hardware, regimes, decision tree, what-not-to-chase. |
| `docs/RDNA3-TUNING-GUIDE.md` | RDNA3 tuning technique. Externally referenced; the current respin of the kernel-level findings in `docs/LESSONS-LEARNED.md`. |
| `docs/API.md` | OpenAI-compatible server usage, endpoint support, current limitations. |
| `docs/ENVS.md` | Complete env-var reference and recommended profiles. |
| `docs/MODELS.md` | Supported models, quantizations, and the backends each is qualified on. |
| `docs/REFACTOR.md` | Cleanup ledger for dead flags, duplicate dispatch paths, and fallback code to remove after optimal paths are proven. |
| `docs/reference/PRODUCTION-NUMERICS-CAMPAIGN.md` | Approved evaluator, calibration, historical-candidate, c1, and c>N/A4 execution plan. |
| `docs/source_lineage.json` | External parent-file manifest used by `scripts/check_lineage.py`. |
| `scripts/docs/` | Documentation tooling: `check_all.py` runs every doc gate, `check_docs.py` validates front-matter and regenerates the indexes. |
| `audit/` | Cleanup audit: mechanical debt inventory, durable triage, and the budget gate. `audit/README.md` has the model. |
| `AGENTS.md` / `CLAUDE.md` | Ground rules (this file). |
| `WORKLOG.md` | Tracked worklog navigation page. |
| `WORKLOG-LEGACY.md` | Byte-frozen pre-Worklog2 journal; never edit. |
| `worklog/entries/` | Immutable per-unit current decisions, results, blockers, and handoffs. |
| `worklog/README.md` / `scripts/worklog.py` | Worklog schema, commands, validator, renderer, and optional pre-commit hook. |
| `benchmarks/README.md` | Canonical topline scoreboard, platform freshness, protocols, artifacts, and root README exports. |
| `benchmarks/HARNESSES.md` | Harness catalog **and the two-tier capacity-testing protocol** - never run full-prompt ladder points to answer a does-it-fit question; use `scripts/gguf_capacity_probe.py` first. |
| `benchmarks/HISTORY.md` | Archived experiment rollup, superseded diagnostics, source-lineage targets, and external baselines. |
| `benchmarks/CHANGELOG.md` | Reverse-chronological one-line history of benchmark rollup updates. |
| `benchmarks/results/` | Compact JSON artifacts for accepted/blocked/rejected benchmark attempts. |
| `pyproject.toml` | Package metadata and extras. Do not casually add hard deps. |

## Workflow

### Before Starting

1. `git status -sb` — note unrelated changes and leave them alone.
2. Read the relevant section of [docs/PLAN.md](docs/PLAN.md) and the latest relevant files under `worklog/entries/` (or run `python3 scripts/worklog.py render`). Read the `WORKLOG-LEGACY.md` tail only when pre-cutoff context matters.
3. For kernel work, a perf claim, or a benchmark row, follow [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) §3 — ROCm liveness, lineage check, and baseline definition — before changing code.

### During Work

- Keep changes scoped to one logical unit (one kernel family, one plugin, one doc, one phase milestone).
- Write or update the targeted test/fixture before implementation when behavior or math changes. If RED-first is impractical, record why in the unit's worklog entry.
- Land new working behavior on by default. Gate it only with a named concrete cause and a named clearing command (see "Product Defaults"); record the blocker in `docs/REFACTOR.md`.
- When adding tests that call HIP/ROCm runtime, `hipcc`, or GPU kernels, add an explicit HIP-availability guard (for example `ctypes.CDLL("libamdhip64.so")` + `pytest.skip`) so no-ROCm CI/publish runners skip them instead of failing release validation.
- For a substantial unit, create a unique entry with `python3 scripts/worklog.py new`, update it with non-trivial decisions, measurements, and dependency additions as work proceeds, and commit it with the unit.
- Do not silently add `import torch`, `flash_attn`, or other CUDA-only deps to hot-path modules.
- Do not add `if backend == "..."` or `if quant == "..."` branches in engine / dispatch / model code.

### After Changes (before claiming done)

- Run the narrowest relevant test, then the applicable `docs/TESTING.md` gate before claiming done.
- Exercise the change through the surface a user reaches, not only through its test or harness. Confirm the intended path ran.
- **Do not automatically rerun a broad suite after an isolated failure.** If a completed broad run establishes that all other tests passed and the repair is scoped, rerun only the failing node(s), the changed test file, and any genuinely affected narrow bundle. Preserve the original broad-run result plus the repaired focused result as the validation evidence. Repeat the full suite only when the fix can affect previously passing tests (for example shared test infrastructure, collection/order/global-state behavior, or broadly shared production code), multiple unrelated failures indicate wider risk, a release protocol explicitly requires a fresh all-green run, or the user explicitly approves it.
- **An assigned task is standing approval for in-scope expensive validation.** State the concrete reason and expected duration before starting a test or benchmark expected to take more than five minutes, then proceed without another approval when it is necessary to complete the user-assigned task or campaign; use a background task when appropriate. Do not spend an equivalent expensive rerun when existing evidence plus focused repair is sufficient. Ask only before work that materially expands beyond the assigned scope or adds unusual destructive/reset-risk behavior.
- For a new / ported kernel or a perf change, run the gates and rollup in [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) §4, §6, and §8.
- Update `docs/PLAN.md` if architectural plans shifted.
- **Commit immediately** when the logical unit is complete and validation passes.

### Verification tiers

Run the narrowest tier for your change; escalate at milestone boundaries.

| Scope | What to run |
| --- | --- |
| Docs / process | Re-read the changed file end-to-end, then `python3 scripts/docs/check_all.py`. No GPU run needed. |
| Flag, kernel, or ledger cleanup | `python3 audit/audit.py check` plus the narrowest relevant `pytest`. Record the triage decision with the change. |
| Code / registry / dispatch | The narrowest relevant `pytest` + applicable CPU deterministic bundle (see `docs/TESTING.md`). |
| Server / API / product surface | The narrowest relevant `pytest`, plus one real request through `hipengine serve` or `LLM.generate()` confirming the intended route ran. |
| Kernel, perf claim, or capacity | See [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) §8. |
| Milestone closure | One full `uv run pytest --suite all -v` + the phase's named perf target vs prior baseline. If that completed run has isolated failures, apply the focused-repair rule above rather than automatically repeating the full suite. |

## Git Discipline

Explicit, auto-commit-after-validation. Many small, atomic, working-state commits with clear provenance — not fewer larger ones.

### Commit Timing

- **Commit immediately** after a logical unit is complete and validation passes. Do not ask, do not wait to be asked, and do not start the next logical task until the previous validated unit is committed.
- Include related handoff docs in the same unit (a change that needed a worklog entry or a `docs/PLAN.md` update commits them together).
- Always commit the new `worklog/entries/<unique-entry>.md` path with the logical unit that required it. Stage it, then run `python3 scripts/worklog.py check` (it validates staged and tracked content only, so another worker's unstaged entry cannot block your commit); never edit, rename, or delete a committed entry.
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

Prefixes: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `perf:`, `port:` (upstream lineage), `kernel:` (kernel edits). **No bylines** — no `Co-authored-by`, no agent attribution, no generated-by footers. Session URLs (`claude.ai/code/session_…`) are additionally banned as leaked credentials. Local enforcement: `scripts/check_commit_msg.py`, installable as a `commit-msg` hook via `python3 scripts/install_commit_msg_hook.py`.

### Never Committed

- Model weights, `*.safetensors` outside fixtures
- Compiled `.so` / JIT caches, `rocprofv3` dumps, raw benchmark logs
- Local env / secrets, Python caches
- Vendored upstream repos (nano-vllm-amd, FastDMS, etc. — referenced by absolute path)

### Never Discard Others' Work

Do not run `git restore`, `git checkout --`, `git reset --hard`, `git clean -fd`, `rm -rf` across tracked paths, or bulk rewrites (aggressive formatters, mass import reordering) unless the user explicitly asks.

## Coordination

Working tree is shared state. Other agents or the human may be editing concurrently.

- **High-conflict files:** `AGENTS.md`, `CLAUDE.md`, `docs/PLAN.md`, `docs/OPTIMIZATION.md`, `docs/BENCHMARK.md`, `docs/TESTING.md`, `docs/KERNELS.md`, `docs/reference/IMPLEMENTATION.md`, `pyproject.toml`, `scripts/worklog.py`, `worklog/README.md`, `hipengine/kernels/registry.py`, `hipengine/quant/registry.py`, `hipengine/models/registry.py`, `hipengine/dispatch/fusion.py`, `hipengine/core/*`.
- Same-file contention: stop and coordinate. The designated agent stages and commits their scoped hunks first to unblock others.
- Worklog entry filenames are unique and should not conflict. A committed entry is immutable; correct it with a new `decision` or `checkpoint` entry that links the superseded path.
- `WORKLOG-LEGACY.md` is byte-frozen and manifest-checked. If an old branch carries an append to pre-cutoff `WORKLOG.md`, the designated merge owner converts its missing material into a new immutable entry instead of changing legacy.
- Do not clean up another agent's benchmark outputs, staged files, or local artifacts unless the task explicitly asks for that cleanup.

## External Reference Repos

Read-only peers under `~/`. Do not edit as part of a hipEngine task. When porting, record source file + commit in the commit message. If an external reference disagrees with `docs/PLAN.md`, `docs/PLAN.md` wins unless we explicitly decide the reference is correct and update `docs/PLAN.md`.

- `~/amd-gpu-tuning/` — parent workspace; kernel lineage reference, benchmark history, `LESSONS-LEARNED.md`. Read-only; kernel development now happens in this tree.
- `~/amd-gpu-tuning/nano-vllm-amd/` — kernel source of truth for the Phase-0 port.
- `~/FastDMS/` — DMS reference (Phase 4).
- `~/FastKMS/` — DFlash speculative decode reference.
- `~/kvcache-quantization-research/` — AQUA / HIGGS / DMS stacking research.

## Handling Blockers

| Situation | Action |
| --- | --- |
| A path is implemented but has never been exercised | Run it. That is the task. Do not gate it, and do not report it as unqualified. |
| Tempted to default-off something that works | Name the concrete cause and the command that would clear it. If you cannot name both, ship it on. |
| Math change lacks an oracle/test | Add a CPU-reference/golden fixture first, or record an explicit no-RED rationale in the unit's worklog entry. |
| Merge conflict in a high-conflict file | Stop and coordinate. Do not force-stage or revert. |
| Unclear whether a change crosses a plugin-registry boundary | Check `docs/PLAN.md` "Extensibility Design" first; if still unclear, ask the human lead. |
| Unrelated files changed in the worktree | Leave them. Another agent or the human owns them. |
| Kernel, profiler, ROCm-environment, or numerical-regression blocker | See [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) §9. |

## Communication

- Lead with the substantive finding or result, not just what command was run.
- Distinguish measured from inferred: "measured X tok/s on Y hardware with Z command" vs "expected to be X based on Y".
- If work is still in progress, state the current concrete result or explicitly say there is no result yet.
- Do not report a missing benchmark as a defect, or an ungated implemented path as a risk. Report what ran, what it did, and what is actually broken.
