# HIPENGINE Work Log

Append-only, chronological journal of decisions, commands, measurements, and next actions. Oldest entries at the top, newest appended at the bottom. Format borrowed from `~/amd-gpu-tuning/WORKLOG.md`: `## YYYY-MM-DD — Title` entries with `###` subsections and evidence-heavy bullets (exact commands, file paths, numbers, commit hashes).

---

## 2026-05-12 — Docs scaffold: tightened AGENTS.md, seeded WORKLOG, split out BENCHMARK/KERNELS/ROOFLINE

### AGENTS.md tightening pass

- Read `~/amd-gpu-tuning/WORKLOG.md` top 270 lines to confirm a WORKLOG format: `## YYYY-MM-DD — Title` entries, `###` subsections, evidence-heavy bullets with exact commands / paths / numbers / commit hashes, append-only chronological. Adopted verbatim for this repo.
- Principle adopted: AGENTS.md is read every session (review, coding, benchmarking). It stays focused on ground rules that apply every session. Activity-specific playbooks go to `docs/`.
- Trimmed `AGENTS.md` from 249 → 164 lines (~34% cut):
  - Merged "Project Overview" prose into "Architectural Invariants" (drop wrapper sentences).
  - Replaced the 9-row Verification Matrix with a compact 5-row tier table referencing `docs/BENCHMARK.md` and `docs/KERNELS.md` for specifics. The dropped rows referenced tests / scripts that do not yet exist ("once those exist" parentheticals).
  - Split out "HIP Kernel Development" → new `docs/KERNELS.md` (port playbook, JIT cache gotcha, build profiles, per-family checklist, rocprofv3 smoke).
  - Split out "Benchmark Hygiene" → new `docs/BENCHMARK.md` (evidence policy, baselines to beat, standard workloads, correctness gate, artifact JSON schema, playbook). Kept only a one-line pointer in AGENTS.md via `docs/BENCHMARK.md` entry in Key Files.
  - Collapsed "Plugin Registry Discipline" into the Architectural Invariants section (the invariant already says "register, do not branch"); mechanics live in `docs/PLAN.md` "Extensibility Design".
  - Trimmed External Reference Repos to the 5-bullet list + rules; dropped per-repo prose.
  - Dropped the Meta "Evolving This File" section.

### Project boundary decision

- **Kernel R&D lives in `~/amd-gpu-tuning/`, not here.** Micro-tuning iteration loops (rocprofv3 time-share audit, VGPR / occupancy hunting, `__launch_bounds__` sweeps, fusion experiments, device-code gotcha catalog) all stay in the parent workspace. HIPENGINE ingests *stable* kernels via the port pipeline in `docs/PLAN.md` "Kernel Port Strategy".
- Consequence: HIPENGINE's `docs/KERNELS.md` is a port playbook (copy + partition + retype + gate), not a kernel-tuning guide. Tuning guide stays at `~/amd-gpu-tuning/AGENTS.md` and `~/amd-gpu-tuning/LESSONS-LEARNED.md`.
- AGENTS.md "Handling Blockers" redirects kernel-micro-opt and ROCm-restore situations to `~/amd-gpu-tuning/` rather than duplicating the procedures here.

### Doc inventory from `~/amd-gpu-tuning/`

Surveyed 12 `.md` files in `~/amd-gpu-tuning/docs/` plus the top-level design docs. Copied or referenced as follows:

| Upstream doc | Action | Rationale |
| --- | --- | --- |
| `docs/ROOFLINE.md` (1573 lines) | **Copied** to `docs/ROOFLINE.md` | Canonical RDNA3 / W7900 hardware landscape: hardware, roofline fundamentals, regimes, decision tree, what-not-to-chase. Read by anyone planning HIPENGINE kernels or setting perf targets. Added provenance header; path-qualified companion-doc cross-refs to `~/amd-gpu-tuning/`. |
| `docs/HIPENGINE.md` (1214 lines) | Already here as `docs/PLAN.md` | Same content; don't duplicate. |
| `LESSONS-LEARNED.md` (814 lines) | Referenced from `AGENTS.md` | Kernel tuning lessons; R&D. Stays in parent. |
| `docs/LLAMACPP-VULKAN.md` (592 lines) | Referenced | llama.cpp HIP vs Vulkan source analysis; R&D lens. |
| `docs/QUANTIZATION.md` (342 lines) | Not copied (yet) | Method ladder reference (GPTQ / AWQ / PARO / i-quants / EXL3 / QTIP). Worth copying when we start the W4/W8 quant plugin work; for now a pointer suffices. |
| `docs/QUALITY.md` (237 lines) | Not copied | Quant-quality comparison plan tied to parent workspace's specific model paths; overlaps our `docs/BENCHMARK.md` correctness gate. Ideas absorbed, file not needed. |
| `docs/REFERENCE.md` (186 lines) | Not copied | "External inputs we looked at but have not (yet) adopted" — R&D tracking, stays in parent. |
| `docs/PARO.md`, `docs/PAROQUANT-COMPRESSION.md`, `docs/PATHA-FUSE.md`, `docs/DFLASH.md`, `docs/SPECULATIVE-DECODE.md`, `QUARK.md`, `PLAN-*.md`, `MEGAKERNEL_*`, `MOE_KERNEL_*`, `EXPERT_KERNEL_*` | Not copied | R&D progress logs, operational runbooks, or Phase-2+ / Phase-5 topic specs. Referenced when we reach the relevant phase. |

### Files written this session

- `AGENTS.md` rewritten (249 → 164 lines).
- `docs/KERNELS.md` created (118 lines): kernel port playbook, port correctness gate (registry resolution + rocprofv3 parity + KL/top-1), `hipengine.core.build` three profiles (`decode` / `prefill` / `baseline`) with flags and wavefront widths, JIT cache gotcha with `rm -rf ~/.cache/hipengine/build/<family>-*`, rocprofv3 smoke command, `register(KernelKey(...))` template, per-family bring-up checklist.
- `docs/BENCHMARK.md` created (219 lines): evidence policy (model + quant + workload + hardware + command + result + correctness gate), default hardware context capture commands, baselines to beat grounded in `~/amd-gpu-tuning/WORKLOG.md` measurements — Qwen3.6-35B-A3B UD-Q8_K_XL on llama.cpp ROCm at 1139.72 tok/s prefill / 71.49 tok/s decode / 44.94 GiB (4K/4K), Qwen3-0.6B c=1 shootout at nano-vllm 30167.12 / 15.33 and mini-sglang 20195.46 / 22.58 (prefill / decode tok/s, 4K/4K) — standard workloads (c=1 short 4K/4K, c=1 long 16K/256, microbenchmark), correctness gate procedure, artifact JSON schema, running-a-benchmark playbook.
- `docs/ROOFLINE.md` copied from `~/amd-gpu-tuning/docs/ROOFLINE.md` (1573 → 1582 lines after header edits) with provenance header and path-qualified cross-references.
- `WORKLOG.md` created (this file).

### Repo state at end of session

- Commits: `f2a5166` docs: add HIPENGINE design plan; `f33b2a8` docs: add AGENTS.md ground rules, CLAUDE.md symlink, .gitignore.
- No `pyproject.toml`, no `hipengine/` package tree, no `tests/`, no `scripts/` yet. This commit is docs-only.

### Next

- **Scaffold and spike.** Stand up the minimal repo skeleton from `docs/PLAN.md` "Project Structure": `pyproject.toml` (core deps: safetensors, tokenizers, jinja2, numpy; extras `[server]`, `[torch]`), `hipengine/` package tree with empty `__init__.py`s in `core/`, `dispatch/`, `kernels/`, `models/`, `quant/`, `layers/`, `kvcache/`, `loading/`, `benchmark/`, `speculative/`, `distributed/`, `server/`; `tests/` and `scripts/` directories; initial skeletal `KernelKey` / `register` / `resolve` in `hipengine/kernels/registry.py` with no kernels registered yet.
- **Spike target.** One end-to-end dispatch path with *no* kernels: (a) a `ModelPlugin` for a toy 1-layer model, (b) the fusion planner producing a longest-match kernel plan from a `layer` chain, (c) resolve each plan step against the `KernelKey` registry and raise a clean "no impl" error for unregistered keys. This validates the 4-axis registry shape — registry, resolver, planner, plugin protocols — before any HIP code lands.
- **Kernel porting.** Separate later task (Phase 0 per `docs/PLAN.md`), gated on the scaffold being correct. Initial port target is `nano-vllm-amd/csrc/amd/qwen35_expert.hip` + `paroquant_kernels.py` → the split tree in `docs/PLAN.md` "Split Plan".

---

## 2026-05-12 — Phase-0 scaffold and no-kernel registry/fusion spike

### Scope

- Implemented the first coding scaffold promised in the prior "Next" section.
- This is intentionally **no-HIP and no-torch**: it validates package shape, plugin registry shape, longest-match fusion planning, and clean missing-kernel errors before any kernel port lands.

### Files added

- `pyproject.toml`: package metadata, core deps (`jinja2`, `numpy`, `safetensors`, `tokenizers`), extras `[server]`, `[torch]`, `[dev]`, pytest config. Torch remains optional only.
- `README.md`: short repo entry point linking `docs/PLAN.md`, `docs/BENCHMARK.md`, `docs/KERNELS.md`, `docs/ROOFLINE.md`.
- Package tree under `hipengine/` with empty package dirs for `loading/`, `kvcache/`, `distributed/`, `speculative/`, `server/`, `benchmark/`, backend dirs (`kernels/hip_gfx1100`, `hip_gfx1151`, `cuda_sm86`, `cpu_reference`) and first scaffold modules:
  - `hipengine/__init__.py`, `hipengine/llm.py`: torch-free public API placeholders (`LLM`, `SamplingParams`), `LLM.generate()` explicitly raises `NotImplementedError` until the engine loop lands.
  - `hipengine/core/{dtype.py,device.py,tensor.py}`: torch-free value objects (`DType`, `Device`, `Tensor` handle scaffold).
  - `hipengine/kernels/registry.py`: `KernelKey`, `register`, `resolve`, `can_resolve`, fallback order, duplicate registration error, clean `MissingKernelError` with attempted keys.
  - `hipengine/dispatch/fusion.py`: `FusionPlanner` longest-match planner over `+`-joined layer composites, primitive fallback, `resolve_plan()`.
  - `hipengine/models/{base.py,registry.py,toy.py}`: `ModelPlugin`, registry, `ToyOneLayerModel` registered under `HipEngineToyForCausalLM` with layer chain `embed -> rmsnorm -> rotate -> qkv_proj -> attention_decode -> o_proj -> lm_head`.
  - `hipengine/quant/{base.py,registry.py,fp16.py}`: `QuantPlugin`, registry, built-in `fp16` plugin.
  - `hipengine/layers/base.py`: minimal `LayerPlugin` protocol.
- `scripts/smoke.py`: source-tree smoke that plans the toy model and expects a clean `MissingKernelError` because no kernels are registered yet.
- `tests/test_kernel_registry.py`, `tests/test_fusion_spike.py`, `tests/test_model_quant_and_imports.py`: 9 tests covering registry exact/fallback/missing behavior, duplicate protection, toy model planning, fused longest-match behavior, fp16 quant registration, and "import hipengine does not import torch".
- `docs/IMPLEMENTATION.md`: lightweight Phase-0 punchlist; this repo now has an active implementation checklist separate from `docs/PLAN.md`.
- `benchmarks/results/.gitkeep`: keeps the retained benchmark-artifact directory in the scaffold.

### Verification

- Compile check:
  `python3 -m compileall -q hipengine tests scripts`
  - Result: pass.
- Unit tests:
  `python3 -m pytest -q`
  - Result: `......... [100%]` (9 tests passed).
- Source-tree smoke, first attempt:
  `python3 scripts/smoke.py`
  - Result: failed with `ModuleNotFoundError: No module named 'hipengine'` because Python sets `sys.path[0]` to `scripts/` when executing a script by path.
  - Fix: `scripts/smoke.py` now prepends the repo root (`Path(__file__).resolve().parents[1]`) to `sys.path` before importing `hipengine`.
- Source-tree smoke after fix:
  `python3 scripts/smoke.py`
  - Result: pass. Output:
    `plan: embed -> rmsnorm -> rotate -> qkv_proj -> attention_decode -> o_proj -> lm_head`
    `expected missing kernel: no kernel implementation for backend='hip_gfx1100', layer='embed', quant='fp16'; attempted: backend='hip_gfx1100', layer='embed', quant='fp16'; backend='cpu_reference', layer='embed', quant='fp16'`
- Torch hot-path audit:
  `rg -n "import torch|torch\." hipengine tests scripts pyproject.toml docs/IMPLEMENTATION.md || true`
  - Result: no executable torch imports/usages; only docstrings mention torch.

### Design notes

- The fusion planner chooses a fused composite only when that composite resolves in the kernel registry; otherwise it emits primitive steps and lets `resolve_plan()` raise `MissingKernelError`. This keeps "missing impl" errors at the registry boundary and validates the no-branch dispatch invariant.
- Kernel registry fallback order is generic: exact → no-variant → fp16 fallback → `cpu_reference` fallback. No dispatch/engine/model code branches on backend or quant.
- The toy model is a fixture, not a real inference path. It exists to exercise the 4-axis registry and longest-match planner before HIP kernels arrive.

### Next

- Add the first CPU-reference layer implementations and fixture format so the correctness gate has something real to run.
- Implement `hipengine.core.build` JIT cache skeleton before the first HIP port, matching `docs/KERNELS.md` hash/cache semantics.
- Port the tiny `smoke_add` HIP kernel as the first build-system smoke before touching the large Qwen3.5 families.

---

## 2026-05-12 — CPU-reference fixtures and JIT build cache skeleton

### Scope

- Completed both outstanding Phase-0 scaffold items requested after the registry/fusion spike:
  1. First CPU-reference layer implementations + fixture format.
  2. `hipengine.core.build` JIT cache skeleton matching `docs/KERNELS.md` semantics.
- Still no HIP compilation in this commit. Build tests are dry-run / plan-only so the scaffold remains runnable without ROCm in CI.

### CPU-reference backend

- Added `hipengine/kernels/cpu_reference/ops.py` with torch-free NumPy reference kernels:
  - `embed(token_ids, table)`
  - `rmsnorm(x, weight, eps=1e-6)`
  - `linear(x, weight, bias=None)`
  - `qkv_proj`, `o_proj`, `lm_head` wrappers around `linear`
  - `rotate(x, cos, sin, rotary_dim=None)` split-half rotary embedding
  - `attention_decode(query, key, value, mask=None, scale=None)` reference scaled dot-product attention
- `hipengine/kernels/cpu_reference/__init__.py` now self-registers these kernels under `KernelKey("cpu_reference", <layer>, "fp16")` on import, and exposes `register_cpu_reference_kernels()` for tests that clear the registry.
- Important behavior check: resolving `backend="hip_gfx1100", layer="rmsnorm", quant="fp16"` now falls back to the CPU-reference kernel when no gfx1100 implementation exists.

### Fixture and correctness format

- Added `hipengine/kernels/cpu_reference/fixtures.py`:
  - JSON schema version `1`.
  - `LayerFixture`, `Tolerances`, `LayerCheckResult` dataclasses.
  - `load_fixture()`, `save_fixture()`, `run_fixture()` helpers.
  - Inputs may be scalars or typed arrays represented as `{ "dtype": ..., "data": ... }`; expected output is a typed array.
- Added first committed fixture: `tests/fixtures/cpu_reference/rmsnorm_basic.json`.
  - Layer: `rmsnorm`, backend: `cpu_reference`, quant: `fp16`.
  - Checks a 2×4 float32 input and 4-vector weight with `atol=rtol=1e-6`.
- Added `hipengine/benchmark/correctness.py` with KL/top-1 logit gate helper:
  - `evaluate_logits(reference_logits, candidate_logits, kl_threshold=0.05, top1_threshold=0.90)`
  - Returns `LogitCorrectness(kl_mean, kl_max, top1_agreement, passed)`.

### JIT build cache skeleton

- Added `hipengine/core/build.py` with:
  - `BuildProfile`, `BuildArtifact` dataclasses.
  - Three profiles from `docs/KERNELS.md`:
    - `decode`: `-mcumode`, `-amdgpu-unroll-threshold-local=600`, wavefront 64.
    - `prefill`: `-amdgpu-unroll-threshold-local=600`, wavefront 32.
    - `baseline`: no extra flags, wavefront 32.
  - `plan_hip_build(...)`: deterministic artifact planner with no compiler invocation.
  - `build_hip(..., dry_run=True)`: returns the planned artifact without creating dirs or invoking `hipcc`.
  - `build_hip(..., dry_run=False)`: creates `~/.cache/hipengine/build/<family>-<hash>/`, writes `manifest.txt`, runs `hipcc -shared -fPIC -O3 ... -o <family>.so`, and returns `ctypes.CDLL` unless `load=False`.
  - Cache key includes source file names + bytes, normalized flags, compiler name, and compiler version text.
- `hipengine/core/__init__.py` exports `BuildArtifact`, `BuildProfile`, `build_hip`, and `plan_hip_build`.

### Tests added

- `tests/test_cpu_reference.py`:
  - RMSNorm matches manual NumPy formula.
  - Split-half rotary output is correct on a small vector.
  - `hip_gfx1100`→`cpu_reference` fallback resolves `rmsnorm`.
  - JSON fixture loads and runs with max abs ≤ `1e-6`.
  - `evaluate_logits()` passes/fails expected KL/top-1 cases.
- `tests/test_build.py`:
  - Build hash changes with profile flags and compiler version, and stays stable for identical inputs.
  - `build_hip(dry_run=True)` does not create cache dirs or require a real compiler.
  - Bad profile and missing source paths fail cleanly.

### Verification

- Compile check:
  `python3 -m compileall -q hipengine tests scripts`
  - Result: pass.
- Unit tests:
  `python3 -m pytest -q`
  - Result: `................. [100%]` (17 tests passed).
- Source-tree smoke:
  `python3 scripts/smoke.py`
  - Result: pass; still intentionally reports a missing `hip_gfx1100/embed/fp16` implementation because the smoke does not import/register CPU-reference fallback.
- Torch hot-path audit:
  `rg -n "import torch|torch\." hipengine tests scripts pyproject.toml docs/IMPLEMENTATION.md || true`
  - Result: no executable torch imports/usages; only docstrings mention torch.

### Implementation punchlist

- Updated `docs/IMPLEMENTATION.md`:
  - `[x] Add first CPU-reference kernels and correctness fixture format.`
  - `[x] Add hipengine.core.build JIT cache implementation.`
  - `smoke_add` HIP port remains unchecked.

### Next

- Port the tiny `smoke_add` HIP kernel and register it under `hip_gfx1100/smoke/fp16` (or a more precise layer key if we decide `smoke_add` should not pretend to be a model layer).
- Add a non-dry-run build test only when ROCm/hipcc availability is confirmed in the environment.
