# HIPENGINE - Agent Guide

HIPENGINE is a ROCm-native inference engine built around a clean Python host and the proven gfx1100 kernel lineage from `nano-vllm-amd`. See [docs/PLAN.md](docs/PLAN.md) for architecture, phase roadmap, and LoC budgets.
This `AGENTS.md` (`CLAUDE.md` symlinked) covers ground rules, process, and repo-specific invariants. Project details belong in `docs/`.

Instruction precedence: if this file conflicts with platform/system/developer instructions, follow those first.

## Summary

- **Source of truth:** [docs/PLAN.md](docs/PLAN.md). Update it when architecture or phase plans move.
- **Cross-session handoff:** `WORKLOG.md` in repo root (create on first entry; append-only, reverse-chronological).
- **Active punchlist:** `docs/IMPLEMENTATION.md` once Phase 0 coding starts.
- **Evidence policy:** every performance claim carries model + quant + workload shape + hardware + exact command + result + correctness gate. No exceptions (see `docs/PLAN.md` "Evidence Policy").
- **Correctness gate for any new/ported kernel:** KL ≤ 0.05 AND top-1 agreement ≥ 90% vs the `kernels/cpu_reference/` oracle on the layer's fixture inputs.
- **Default hardware:** AMD Radeon Pro W7900, gfx1100/RDNA3. Claims about other backends (`hip_gfx1151`, `cuda_sm86`) require the corresponding hardware or are marked explicitly unverified.

## Project Overview

HIPENGINE is a purpose-built local LLM inference engine that pairs a ~700-line torch-free Python host with a ~18,600-line gfx1100 HIP kernel tree (120 `__global__` kernels) ported from the `nano-vllm-amd` research lineage. Key architectural invariants that must not drift casually:

- **Torch-free runtime.** `import torch` is **not** allowed in any module reached by `hipengine.LLM.generate()`. Torch lives behind the optional `hipengine[torch]` extra and appears only as a dlpack bridge at the user boundary. Adding `import torch` anywhere on the hot path is an architectural change, not a refactor.
- **Four-axis plugin registry.** Kernels are keyed by `(backend, layer, quant, variant)`. Models, quant schemes, and layers are plugins. **Never** add `if backend == "hip_gfx1100"` style branches in dispatch/engine/model code; register a new implementation against a registry key instead.
- **Fused kernels require an unfused fallback.** Every fused composite (`rmsnorm+rotate`, `gate_combine_residual`, …) must have a numerically-equivalent unfused chain registered under its primitives, to serve as correctness oracle and as fallback for backends that haven't ported the fusion.
- **Kernel bodies take raw device pointers.** `__global__` signatures use `void*` / typed pointers, never `torch::Tensor`. Only the host-side launch wrappers convert.
- **`KVLiveSpans` is the attention kernel ABI, not a DMS-only concept.** Every paged-KV-write and attention-decode kernel reads `(base_offsets, live_counts, token_positions, evict_mask)`. Dense policies fill it uniformly; DMS/H2O/SnapKV fill it variably. Do not shortcut to `(block_table, context_len)`.
- **Backend tree is a peer structure.** `kernels/hip_gfx1100/`, `kernels/hip_gfx1151/`, `kernels/cuda_sm86/`, `kernels/cpu_reference/` are siblings. There is no "AMD directory"; there are backend-keyed directories.

## Key Files

| Path | Purpose |
| --- | --- |
| `docs/PLAN.md` | Architecture, phase roadmap, LoC budgets. Source of truth for what we're building and why. |
| `AGENTS.md` / `CLAUDE.md` | Ground rules, process, repo-specific invariants (this file). |
| `WORKLOG.md` (create when first needed) | Append-only cross-session journal: decisions, commands, measurements, next actions. |
| `docs/IMPLEMENTATION.md` (create when Phase 0 coding starts) | Active punchlist / checklist for ongoing work. |
| `docs/LESSONS-LEARNED.md` (create when Phase 0 coding starts) | Durable kernel / dispatch lessons that outlive individual tasks. |
| `hipengine/kernels/registry.py` | Plugin registry. High-conflict; coordinate before editing. |
| `hipengine/core/` | Torch-free primitives. Architectural; changes touch every other layer. |
| `pyproject.toml` | Package metadata and extras (`[server]`, `[torch]`). Do not casually add hard deps. |

## Workflow Expectations

### Before Starting

1. Run `git status -sb`. Note any unrelated changes and leave them alone.
2. Read the relevant section of [docs/PLAN.md](docs/PLAN.md) and the latest `WORKLOG.md` tail.
3. For kernel work, confirm the GPU and ROCm stack:
   ```bash
   python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
   rocminfo | grep -E 'Name:|gfx'
   ```
4. For a port from `nano-vllm-amd`, diff the source files you're pulling from and record the source commit in the commit message.
5. For a perf claim, define the baseline (model, quant, workload shape, hardware, command) before the change so the comparison is well-defined.

### During Work

- Keep changes scoped to one logical unit (one kernel family, one plugin, one doc, one phase milestone).
- Log non-trivial decisions, measurements, and dependency additions in `WORKLOG.md` as they happen.
- Run the narrowest relevant test before broader suites (see Verification).
- For kernel edits, clear JIT caches when behavior looks stale (see HIP Kernel Development → JIT cache).
- Do not silently add `import torch`, `torch.*`, `flash_attn`, or other CUDA-only deps to hot-path modules. If torch interop is needed, route it through `hipengine[torch]` dlpack boundary and say so explicitly.
- Do not add `if backend == "..."` or `if quant == "..."` branches in engine / dispatch / model code — register against the 4-axis registry.

### After Changes (before claiming done)

- Run the relevant verification tier (see Verification Matrix).
- For a new/ported kernel: correctness gate (KL ≤ 0.05, top-1 ≥ 90%) against `kernels/cpu_reference/` AND a microbenchmark or smoke showing the kernel runs and produces the expected `rocprofv3 --kernel-trace` entry.
- For a perf change: record baseline + new measurements in `WORKLOG.md` with exact commands.
- Update `docs/PLAN.md` if architectural plans shifted; update `docs/IMPLEMENTATION.md` checkboxes.
- **Commit immediately** when the logical unit is complete and validation passes. Do not batch unrelated changes.

## Verification Matrix

Use the narrowest relevant tier for your change. Escalate at milestone boundaries.

| Scope | Commands |
| --- | --- |
| Docs / process change | Re-read the changed file end-to-end; no GPU run needed unless behavior claims changed. |
| `hipengine.core.*` change | `uv run pytest tests/test_tensor.py tests/test_graph_capture.py -q` (once those exist) |
| New or ported kernel | CPU-reference correctness (KL ≤ 0.05, top-1 ≥ 90%) + microbenchmark smoke + `rocprofv3 --kernel-trace` shows expected kernel name and plausible `DurationNs` |
| Dispatch / fusion planner change | `uv run pytest tests/test_kernel_registry.py tests/test_fusion.py -q` plus a Qwen3-0.6B generate smoke end-to-end |
| Model plugin change | Qwen3-0.6B (or target model) generate smoke + KL vs `cpu_reference` on a fixed prompt set |
| Quant plugin change | Round-trip weight prepare/dequant on a fixture + target model smoke + KL gate |
| KV policy change | `admission_cap()` + `batch_spans()` unit tests + full decode smoke at ≥ 4K context |
| Perf claim | Re-run the exact benchmark command recorded with the claim, on the stated hardware, and record both runs in `WORKLOG.md` |
| Milestone closure | `uv run pytest -v` full suite + the phase's named perf target vs prior baseline |

Smoke commands will be added here as scripts land under `scripts/`. Until then, use the microbenchmark and generate scripts from the `nano-vllm-amd` lineage and record the exact invocation.

## HIP Kernel Development

Applies to anything under `hipengine/kernels/hip_gfx1100/` (and other backend trees). These rules transfer directly from `amd-gpu-tuning` and are non-negotiable for kernel work.

### Audit first, optimize second (MANDATORY)

Before any kernel-internal optimization (micro-tuning, `__launch_bounds__`, LDS staging, vectorized loads, fusion), run `rocprofv3 --kernel-trace` on the target workload and confirm:

1. **Time share.** Sum `DurationNs` per `KernelName` and rank. Optimize the kernel that dominates wall time, not the one that feels slow. In the parent workspace ~100 iterations were spent optimizing a kernel at 19% of decode time while the kernel at 76.9% sat untouched.
2. **Grid occupancy.** `Grid_Size / Workgroup_Size >= CU count` (W7900 = 96 CUs, Strix Halo = 40). If no, fix grid structure (split-K, more work/kernel) before touching the inner loop.
3. **Iters-per-thread.** For `for (k = threadIdx.x; k < N; k += blockDim.x)` loops with `N / blockDim.x < 64`, the compiler does not auto-unroll and loop overhead dominates FMA. The vec8 pattern is the fix (was the biggest single E2E win in the parent lineage: +54% combined for the w8a16 family).
4. **VGPR / scratch / LDS.** `VGPR_Count` ≥ 96 squeezes occupancy (consider `__launch_bounds__`). `Scratch_Size` > 0 on a hot path is a failed hypothesis unless E2E wins. `LDS_Block_Size` is not free on RDNA3 — only stage when data is reused > 4× AND staging eliminates a scatter.

Re-run the audit after any structural change (grid split, fusion, KV layout) — the time-share ranking shifts.

### JIT extension cache

Our build layer (`hipengine.core.build`) caches compiled `.so` files by a hash of `(source, flags, hipcc version)` under `~/.cache/hipengine/build/`. Symptoms of a stale cache after a `.hip` edit: kernel calls hang with GPU at 0% use and no error. Nuke the matching cache dir before re-importing:

```bash
rm -rf ~/.cache/hipengine/build/<family>-<hash>*
```

The three build profiles from `nano-vllm-amd` produce different `.so` dirs: `decode` (`-mcumode` + `-amdgpu-unroll-threshold-local=600`, wavefront=64), `prefill` (unroll-600, WGP/wavefront=32), `baseline` (no flags). Write device code assuming the profile's wavefront width.

### Device-code gotchas

- `min<int64_t>(a, b)` does not compile in HIP device code. Use a ternary: `(a < b) ? a : b`.
- `__shfl_down(acc, offset)` works on AMD for intra-warp reduction. Use the `warpSize` built-in (64 under `decode`/CU mode), not a hard-coded 32.
- `__syncthreads()` on a single-warp block is usually elided; prefer explicit warp-shuffle reductions.
- Add `__restrict__` on kernel pointer parameters when the KV / weight / activation pointers are known not to alias.

### Kernel porting from `nano-vllm-amd`

The initial port is **copy + partition + retype**, not rewrite. Kernel bodies are preserved byte-for-byte (modulo `#include` headers). The three things that change during port:

1. File split by family into `kernels/hip_gfx1100/<family>/*.hip` (see `docs/PLAN.md` "Kernel Port Strategy").
2. Host-side launch wrappers retyped from `torch::Tensor` to raw pointer + shape/stride/dtype signatures (mechanical, ~1 day scripted).
3. `paroquant_kernels.py`'s 3,766-line embedded HIP string is extracted into real `.hip` files.

**Correctness gate for the split:** (a) every kernel name still resolves via the registry, (b) `rocprofv3 --kernel-trace` reports the same kernel set with matching `DurationNs` distribution on the Qwen3.6-35B-A3B decode smoke, (c) KL ≤ 0.05 and top-1 ≥ 90% vs the monolithic build on correctness fixtures. Do not land a split that regresses any of these.

## Benchmark Hygiene

- Always record the exact command for a retained benchmark number.
- Include hardware + software context: W7900/gfx1100, ROCm version, `hipcc --version`, driver version from `rocminfo`.
- Include workload shape: model, quant, prompt length, generation length, batch size / concurrency, KV policy, warmup policy.
- Run the cheapest representative check first: imports → one-request generate → tiny prompt/gen → the actual comparison.
- Distinguish direct measurements from inference when reporting results.
- Keep a compact artifact (JSON / table) for any benchmark number that will be referenced in later comparisons; raw terminal output is not evidence.
- Report failures as useful evidence when they clarify unsupported paths, OOM limits, or ROCm incompatibilities.

## Plugin Registry Discipline

The `(backend, layer, quant, variant)` tuple is the central extensibility mechanism. Rules:

- New kernel = `register(KernelKey(...), kernel_fn)` at module import, not an `if` branch somewhere.
- New quant preset = new `QuantPlugin` with the six orthogonal axes filled in (weight storage / activation preprocess / compute dtype / scale granularity / calibration artifact / kernel family). Do not collapse two formats into one plugin because "they're similar."
- New model = new `ModelPlugin` with layer sequence, weight name map, chat template, RoPE config. Model code does not reach into kernels directly; it asks the fusion planner for a kernel plan.
- `layer` key values are either primitives (`"rmsnorm"`, `"qkv_proj"`) or `"+"`-joined fused composites (`"rmsnorm+rotate+qkv_proj"`). The planner does longest-match against the registry.

High-conflict files for this discipline: `hipengine/kernels/registry.py`, `hipengine/quant/registry.py`, `hipengine/models/registry.py`, `hipengine/dispatch/fusion.py`. Coordinate before editing.

## Git Discipline

This repo uses an explicit, auto-commit-after-validation pattern. The goal is many small, atomic, working-state commits with clear provenance — not fewer larger ones.

### Commit Timing

- **Commit immediately** after a logical unit is complete and validation passes. Do not ask, do not wait to be asked.
- A logical unit includes its related handoff docs. If the change required a `WORKLOG.md` entry or a `docs/PLAN.md` update, commit them with the same unit.
- Do not commit mid-task while exploring, debugging, or in a broken state.
- Docs, plans, repo-setup, and dependency additions are first-class logical units. A completed architecture doc update is a commit by itself.

### Commit Mechanics (hard rules)

- **Never** use `git add .`, `git add -A`, or `git commit -a`.
- **Never** revert, checkout, or restore files you did not modify for the current task.
- **Always** stage files explicitly: `git add <path1> <path2> …`.
- **Always** verify before committing:
  ```bash
  git status -sb
  git diff --staged --name-only   # confirm only your files
  git diff --staged               # review the actual diff
  ```
- If unrelated changes exist in the worktree, leave them unstaged. They belong to another agent or the human.
- If staged files are present that are not yours, treat that as another commit in progress — do not unstage them.

### Commit Messages

```
type: short summary (imperative mood, ≤ 72 chars)

- Bullet points for non-obvious context
- Source commit when porting from upstream (e.g. nano-vllm-amd@f3a1c2e)
- Correctness / perf evidence when relevant
```

Conventional prefixes: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `perf:`, `port:` (for upstream lineage ports), `kernel:` (for kernel edits).

- **No bylines.** No `Co-authored-by`, no agent attribution, no generated-by footers.

### What Never Gets Committed

- Model weights and checkpoints (`models/`, `*.safetensors` outside fixtures)
- Compiled `.so` / JIT caches (`.pi/`, `~/.cache/hipengine/` is outside the repo anyway)
- `rocprofv3` dumps, raw profiler CSVs, large benchmark logs
- Local env files, secrets, `.env`
- Python caches (`__pycache__/`, `*.pyc`, `.pytest_cache/`)
- Vendored upstream repos (nano-vllm-amd, FastDMS, etc. — they're external peers, referenced by absolute path)

Keep a minimal `.gitignore` that enforces the above.

### Never Discard Others' Work

Do not use destructive commands unless the user explicitly asks:

- `git restore`, `git checkout --`, `git checkout .`, `git reset --hard`, `git clean -fd`
- `rm -rf` across tracked paths, overwriting redirects (`> file`) against tracked files
- Bulk rewrites (aggressive formatters, mass import reordering) that destroy local edits

## Coordination Hygiene

- Treat the working tree as shared state. Other agents or the human may be editing concurrently.
- **High-conflict files:** `AGENTS.md`, `CLAUDE.md`, `docs/PLAN.md`, `docs/IMPLEMENTATION.md`, `WORKLOG.md`, `pyproject.toml`, `hipengine/kernels/registry.py`, `hipengine/quant/registry.py`, `hipengine/models/registry.py`, `hipengine/dispatch/fusion.py`, `hipengine/core/*`.
- For same-file contention (two agents, overlapping edits): **stop and coordinate**. The designated agent stages and commits their scoped hunks first to unblock others.
- For `WORKLOG.md` (append-only): re-read the live tail, append after it, commit with your logical unit. Same-file appends are expected and not a conflict unless there are actual conflict markers or interleaved garbled lines.
- Do not clean up another agent's benchmark outputs, staged files, or local artifacts unless the task explicitly asks for that cleanup.

## Reference Repos (External)

`docs/PLAN.md` references external research repos that are **not vendored** into this repo and live at absolute paths under `/home/lhl/`:

- `/home/lhl/amd-gpu-tuning/` — parent workspace with kernel lineage, benchmark history, `LESSONS-LEARNED.md`
- `/home/lhl/amd-gpu-tuning/nano-vllm-amd/` — kernel source of truth for the Phase-0 port
- `/home/lhl/FastDMS/` — DMS reference implementation for Phase 4
- `/home/lhl/FastKMS/` — DFlash speculative decode reference
- `/home/lhl/kvcache-quantization-research/` — AQUA / HIGGS / DMS stacking research

Rules for external references:
- Read-only for the purposes of HIPENGINE work. Do not edit them as part of a HIPENGINE task.
- When porting code, record the source file + commit in the commit message.
- Do not copy architectural patterns wholesale because "another tool does it this way." HIPENGINE's architecture is its own (torch-free, 4-axis registry, `KVLiveSpans` ABI). If the only rationale for a design choice is upstream precedent, reject it.
- If an external reference disagrees with `docs/PLAN.md`, `docs/PLAN.md` wins. Update `docs/PLAN.md` if the reference is actually correct.

## Handling Blockers

| Situation | Action |
| --- | --- |
| ROCm environment appears corrupted | Stop before reinstalling. Record symptoms in `WORKLOG.md`, then follow the amd-gpu-tuning `therock` env restore commands if restore is clearly required. |
| Kernel hangs with GPU at 0%, no error | Stale JIT cache. Clear `~/.cache/hipengine/build/<family>-*` and re-import. |
| `rocprofv3` reports unexpected kernel not in the plan | Registry / dispatch bug, not a kernel bug. Check `fusion.plan()` output before touching the kernel. |
| KL / top-1 regression after a kernel edit | Revert, add a correctness fixture that captures the failure, then re-try. Never land a perf win that regresses correctness. |
| A kernel micro-optimization shows neutral / negative results | Usually means you're optimizing the wrong kernel or missed a pathology — re-run the pre-optimization audit, do not keep tweaking. |
| Merge conflict in a high-conflict file | Stop and coordinate. Do not force-stage or revert. |
| Unclear whether a change crosses a plugin-registry boundary | Check `docs/PLAN.md` "Extensibility Design" first; if still unclear, ask the human lead. |
| Unrelated files changed in the worktree | Leave them. Another agent or the human owns them. |

## Communication

- Lead with the substantive finding or result, not just what command was run.
- Distinguish measured from inferred: say "measured X tok/s on Y hardware with Z command" vs "expected to be X based on Y".
- If work is still in progress, state the current concrete result or explicitly say there is no result yet.
- Truth-scoped wording for claims: "W7900 / Qwen3-0.6B / FP16 / ctx=4096 / bs=1 → X tok/s decode (cpu_reference KL=0.02, top-1=94%)" not "HIPENGINE is faster than vLLM."

## Meta: Evolving This File

Update this file when:
- A workflow pattern proves helpful or causes confusion
- A new invariant emerges that a future agent would trip over
- Phase plans shift in a way that changes ground rules (e.g., adding the CUDA backend changes how "hardware" is discussed)
- A recurring mistake deserves a hard rule

Keep it focused on process, invariants, and how to work safely in this repo. Architectural details go in `docs/PLAN.md`. Kernel lessons go in `docs/LESSONS-LEARNED.md`. Punchlists go in `docs/IMPLEMENTATION.md`.
