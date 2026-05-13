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

---

## 2026-05-12 — License HIPENGINE as AGPL-3.0-or-later

### Decision

- Selected **AGPL-3.0-or-later** for HIPENGINE source code.
- Rationale: project is aimed at local/home users, and we explicitly prefer copyleft over permissive/business adoption. AGPL closes the hosted-service loophole that GPLv3 leaves open for an inference engine with optional server/API paths.
- User clarified that the future `nano-vllm-amd` kernel ports are not an upstream-license concern for this decision because those kernels were authored locally by the project lead; still, model weights/checkpoints and external datasets remain under their own licenses.

### Files changed

- Added `LICENSE` containing the full GNU Affero General Public License v3 text from the system SPDX license copy (`/usr/share/licenses/spdx/AGPL-3.0-or-later.txt`).
- Updated `pyproject.toml` project metadata from `Apache-2.0` to `AGPL-3.0-or-later`.
- Updated `README.md` with a License section: HIPENGINE source code is AGPL-3.0-or-later; model weights, checkpoints, and external datasets keep their own licenses.
- Updated `docs/PLAN.md` "License" section from the prior MIT placeholder to AGPL-3.0-or-later.

### Verification

- Docs/metadata-only change. Re-read touched snippets and checked license references with:
  `rg -n "Apache|AGPL|GPL|License|license" pyproject.toml README.md docs AGENTS.md WORKLOG.md LICENSE`

---

## 2026-05-12 — Non-GPU prep for HIP smoke path

### Scope

- User requested continuing through all next steps but pausing before touching the GPU because another process is benchmarking/tuning on it.
- Completed the safe pre-GPU subset only: expanded CPU-reference fixtures, added lazy HIP runtime/memory wrappers that do not load ROCm on import, and added `smoke_add` HIP source + registry + dry-run build planning.
- Explicitly did **not** run `rocminfo`, `rocm-smi`, `hipcc`, non-dry `build_hip()`, HIP runtime calls, or profiler commands.

### CPU-reference fixtures

- Added committed CPU-reference fixtures:
  - `tests/fixtures/cpu_reference/linear_basic.json`
  - `tests/fixtures/cpu_reference/rotate_split_half.json`
  - `tests/fixtures/cpu_reference/attention_decode_masked.json`
  - Existing `rmsnorm_basic.json` retained.
- Added `scripts/check_fixtures.py`, a CPU-only fixture runner for JSON fixture files/directories.
- Extended `tests/test_cpu_reference.py` to require all four committed fixtures and run each through `run_fixture(load_fixture(path))`.

### Lazy HIP runtime/memory skeleton

- Added `hipengine/core/hip.py`:
  - `HipMemcpyKind`, `HipError`, `HipRuntime`.
  - `HipRuntime.load()` lazily loads `libamdhip64.so` only when explicitly called.
  - `malloc`, `free`, `memcpy`, `device_synchronize`, `error_string`, `check` wrappers.
  - `is_default_runtime_loaded()` and `reset_default_runtime_for_tests()` to prove import-time laziness.
- Added `hipengine/core/memory.py`:
  - `DeviceBuffer`, `malloc`, `free`, host/device copy helpers, host pointer helpers.
  - No HIP library load on import; allocation/copy helpers load runtime only when called.
- Added `tests/test_hip_runtime.py` with a fake HIP library object, so tests cover ctypes arg/return setup and error behavior without ROCm or GPU access.

### smoke_add dry-run path

- Added `hipengine/kernels/hip_gfx1100/smoke/smoke_add.hip`:
  - Device kernel `hipengine_smoke_add_f32_kernel`.
  - C ABI host wrapper `hipengine_smoke_add_f32(...)` using `hipLaunchKernelGGL` and returning `hipGetLastError()`.
- Added `hipengine/kernels/hip_gfx1100/smoke/smoke_add.py`:
  - `plan_smoke_add_build()` dry-run-safe build artifact planner.
  - `build_smoke_add()` wrapper around `build_hip()`.
  - `smoke_add_f32()` lazy launch wrapper; first GPU-touching function, not called yet.
  - `register_smoke_add_kernel()` registering `KernelKey("hip_gfx1100", "smoke_add", "fp16")`.
- Updated `hipengine/kernels/hip_gfx1100/smoke/__init__.py` to expose the lazy smoke-add wrapper.
- Added `tests/test_smoke_add_plan.py` for registry and build-plan coverage without invoking `hipcc`.

### Scripts

- Updated `scripts/smoke.py` with CPU-only modes:
  - `--mode registry` (default): toy model registry/fusion smoke; expects clean missing `hip_gfx1100/embed/fp16`.
  - `--mode cpu-fixtures`: runs committed CPU-reference JSON fixtures.
  - `--mode smoke-add-plan`: prints the dry-run `hipcc` command/artifact for `smoke_add` without invoking it.
- Important fix: CPU-reference and smoke-add imports are now mode-local so `--mode registry` does not accidentally self-register CPU fallback kernels at import time.

### Verification (CPU-only)

- Command:
  ```bash
  set -e
  python3 - <<'PY'
  from pathlib import Path
  bad = False
  for root in ('hipengine', 'tests', 'scripts'):
      for p in Path(root).rglob('*.py'):
          if '__pycache__' in p.parts:
              continue
          for i, line in enumerate(p.read_text().splitlines(), 1):
              if len(line) > 100:
                  print(f'{p}:{i}:{len(line)}:{line}')
                  bad = True
  raise SystemExit(1 if bad else 0)
  PY
  python3 -m compileall -q hipengine tests scripts
  python3 -m pytest -q
  python3 scripts/check_fixtures.py
  python3 scripts/smoke.py --mode registry
  python3 scripts/smoke.py --mode cpu-fixtures
  python3 scripts/smoke.py --mode smoke-add-plan
  rg -n "import torch|torch\." hipengine tests scripts pyproject.toml docs/IMPLEMENTATION.md || true
  ```
- Results:
  - Line-length scan: pass (no >100-character Python lines).
  - Compile check: pass.
  - Unit tests: `........................ [100%]` (24 tests passed).
  - `scripts/check_fixtures.py`: all four fixtures PASS with `max_abs=0`.
  - `scripts/smoke.py --mode registry`: pass; expected missing kernel for `hip_gfx1100/embed/fp16`.
  - `scripts/smoke.py --mode cpu-fixtures`: all four fixtures PASS.
  - `scripts/smoke.py --mode smoke-add-plan`: pass; printed dry-run command:
    `hipcc -shared -fPIC -O3 /home/lhl/hipengine/hipengine/kernels/hip_gfx1100/smoke/smoke_add.hip -o /home/lhl/.cache/hipengine/build/smoke-101db2a5ad5526c3/smoke_add.so`
  - Torch audit: no executable torch imports/usages; only docstrings mention torch.

### Implementation punchlist

- Updated `docs/IMPLEMENTATION.md`:
  - `[x] Add lazy HIP runtime/memory skeleton (no HIP library load on import).`
  - `[x] Add first HIP smoke kernel source and dry-run registry/build plan (smoke_add).`
  - Added unchecked GPU gate: `[ ] Run first HIP smoke kernel (smoke_add) on GPU after explicit clearance.`

### Pause point

- Stopping here before the first GPU-touching action.
- Next after user confirms the GPU is clear:
  1. Run ROCm/hipcc environment checks.
  2. Build `smoke_add` with non-dry `build_hip()`.
  3. Allocate/copy/synchronize through the lazy HIP runtime wrappers.
  4. Record exact commands/results and commit the real GPU smoke.

---

## 2026-05-13 — First real HIP smoke_add build/run

### GPU clearance and environment

- User confirmed the GPU was open, so proceeded past the explicit GPU-touching pause.
- Repo state before changes:
  `git status -sb` → clean at `0baa95c feat: prepare smoke_add without touching GPU`.
- HIP runtime check:
  `python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"`
  - Result: `hip OK`.
- Compiler check:
  `command -v hipcc && hipcc --version`
  - Result: `/opt/rocm/bin/hipcc`, HIP version `7.2.53211-d40244d`, AMD clang `22.0.0git`.
- Hardware check:
  `rocminfo | grep -E 'Name:|gfx' | head -24`
  - Result included `Name: gfx1100`, `Marketing Name: AMD Radeon Pro W7900`.
- Pre-smoke GPU state:
  `rocm-smi --showmeminfo vram --showuse --showtemp`
  - Result: GPU use `0%`, VRAM used `27,930,624 B` / `48,301,604,864 B`; edge/junction/memory temps `39/48/48 C`.

### smoke_add real run

- First non-dry build/run command (ad-hoc Python, before adding script mode):
  ```bash
  python3 - <<'PY'
  import numpy as np
  from hipengine.core.hip import get_hip_runtime
  from hipengine.core.memory import malloc, free, copy_host_to_device, copy_device_to_host, host_array_ptr
  from hipengine.kernels.hip_gfx1100.smoke import build_smoke_add, smoke_add_f32
  n = 1024
  a_host = np.arange(n, dtype=np.float32)
  b_host = (np.arange(n, dtype=np.float32) * 2.0) + 1.0
  out_host = np.empty_like(a_host)
  runtime = get_hip_runtime()
  lib = build_smoke_add(load=True)
  a_dev = b_dev = out_dev = None
  try:
      a_dev = malloc(a_host.nbytes, runtime=runtime)
      b_dev = malloc(b_host.nbytes, runtime=runtime)
      out_dev = malloc(out_host.nbytes, runtime=runtime)
      copy_host_to_device(a_dev, host_array_ptr(a_host), runtime=runtime)
      copy_host_to_device(b_dev, host_array_ptr(b_host), runtime=runtime)
      smoke_add_f32(a_dev.ptr, b_dev.ptr, out_dev.ptr, n, library=lib, runtime=runtime)
      runtime.device_synchronize()
      copy_device_to_host(host_array_ptr(out_host), out_dev, runtime=runtime)
  finally:
      for buf in (out_dev, b_dev, a_dev):
          if buf is not None:
              free(buf, runtime=runtime)
  expected = a_host + b_host
  max_abs = float(np.max(np.abs(out_host - expected)))
  print(f'n={n} max_abs={max_abs}')
  print('first5=', out_host[:5].tolist())
  if not np.allclose(out_host, expected):
      raise SystemExit(1)
  PY
  ```
  - Result: `n=1024 max_abs=0.0`, `first5= [1.0, 4.0, 7.0, 10.0, 13.0]`.
- Added durable smoke mode:
  `python3 scripts/smoke.py --mode smoke-add-hip --n 1024`
  - Result: `n=1024 max_abs=0.0`, `first5= [1.0, 4.0, 7.0, 10.0, 13.0]`.
- Build artifact: `~/.cache/hipengine/build/smoke-101db2a5ad5526c3/smoke_add.so`.
- This validates the first non-dry `hipengine.core.build` path, lazy `libamdhip64.so` load, `hipMalloc`/`hipMemcpy`/kernel launch/`hipDeviceSynchronize`/copyback/free path without torch.

### rocprofv3 attempt and blocker

- Tried to capture the kernel trace:
  `rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-smoke-add-trace -- python3 scripts/smoke.py --mode smoke-add-hip --n 1024`
  - Result: hung until the harness timeout at 120 s; no trace CSV observed.
- Retried with shell timeout:
  `timeout 60s rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-smoke-add-trace -- python3 scripts/smoke.py --mode smoke-add-hip --n 1024`
  - Result: rocprofv3 caught signal 15 and waited for a child process; the wrapper did not exit cleanly before the outer 90 s harness timeout.
  - Cleanup: killed the lingering `timeout`/`python3 scripts/smoke.py` rocprof child processes.
- `command -v rocprofv3` resolved to `/home/lhl/mambaforge/envs/therock/bin/rocprofv3`, a Python wrapper around `rocm_sdk_core._cli`; `rocprofv3 --version` reports ROCm `7.13.0`, while `/opt/rocm/bin/hipcc --version` reports HIP `7.2.53211-d40244d`. This version split may be relevant.
- Action: added unchecked `docs/IMPLEMENTATION.md` item to resolve the `rocprofv3` trace hang before the first real kernel port. Do **not** start rmsnorm port until trace capture is reliable.

### Post-smoke GPU state / pause

- Post-smoke check:
  `rocm-smi --showmeminfo vram --showuse --showtemp`
  - Result: GPU use `0%`, VRAM used `4,376,268,800 B`; edge/junction/memory temps `39/48/52 C`.
- `rocm-smi --showpids` showed PID `1697754` using `4,343,508,992 B` VRAM:
  `/home/lhl/amd-gpu-tuning/scripts/bench_paro_native_engine.py --model-preset qwen35-a3b-paro --prompt-len 512 --decode-len 128 ...`
- That process is not owned by this HIPENGINE task. Pausing further GPU actions here; do not run rmsnorm port or more profiling until the GPU is explicitly clear again.

### Verification after adding script mode

- Command:
  `python3 -m compileall -q hipengine tests scripts && python3 -m pytest -q && python3 scripts/smoke.py --mode smoke-add-hip --n 1024`
- Result: `24 passed`; smoke-add HIP run passed with `max_abs=0.0`.

### Implementation punchlist

- Updated `docs/IMPLEMENTATION.md`:
  - `[x] Run first HIP smoke kernel (smoke_add) on GPU after explicit clearance.`
  - `[ ] Resolve rocprofv3 trace hang for Python/ctypes smoke before first real kernel port.`

---

## 2026-05-13 — Add testing discipline for math correctness

### Prompt / concern

- User noted HIPENGINE is becoming "real" software and should have a proper testing story: RED/GREEN, correctness guard/gates, and especially protection against silent math mistakes.
- Goal: adopt useful testing methodology/verbiage from `~/shisad/` and `~/shisad-dev/` without importing irrelevant process (multi-reviewer lanes, release machinery, implement-driven workflow).

### Sources reviewed

- `~/shisad/AGENTS.md`:
  - Useful: Spec → Plan → Test → Implement; write tests first even for ad-hoc work; run targeted tests first; exact command evidence; claim integrity; structural tests are not enough for runtime-facing behavior.
  - Not adopted: shisad-specific security roles, multi-reviewer process, live daemon harness details.
- `~/shisad-dev/AGENTS.md`:
  - Useful: validation cadence proportional to scope; do not default to broad suites for every small change; record validation evidence in worklog; truth-scoped claims.
  - Not adopted: private/public repo split, reviewer-lane rules, release-close process.
- `~/shisad-dev/implement/TEST-COVERAGE.md`:
  - Most relevant source. Key adapted concept: structural correctness is necessary but not sufficient. For shisad the real contract is user-visible correctness; for HIPENGINE the real contract is numerical correctness against an oracle.
  - Adapted RED/GREEN requirement: for regressions and math changes, add a failing fixture/test first where practical; if impossible, record no-RED rationale.
- `~/shisad-dev/planning/PLAN-test-optimization.md` and `~/shisad/docs/analysis/ANALYSIS-test-suite-optimization.md`:
  - Useful as cautionary examples on test cost and validation cadence. Adopted the principle "targeted first, CPU deterministic bundle for ordinary changes, GPU/perf gates only when relevant".

### Files changed

- Added `docs/TESTING.md` as the detailed testing playbook.
- Updated `AGENTS.md` only with concise every-session rules/pointers:
  - Summary bullet: "math changes are guilty until proven correct"; follow RED/GREEN where practical; details in `docs/TESTING.md`.
  - Key Files entry for `docs/TESTING.md`.
  - During Work: write/update targeted tests/fixtures before behavior/math implementation, or record why RED-first is impractical.
  - After Changes and Verification tiers: run applicable `docs/TESTING.md` gates.
  - Handling blockers: if a math change lacks oracle/test, stop and add a CPU-reference/golden fixture or record explicit no-RED rationale.
  - Coordination high-conflict list now includes `docs/TESTING.md`.

### Testing policy adopted

- **Core principle:** math changes are guilty until proven correct.
- **Structural vs numerical correctness:** registry resolution, build artifacts, launches, shapes, and traces are necessary diagnostics; they do not prove math correctness. Any math-touching test must assert numerical output against an oracle.
- **Oracle preference order:** analytic/high-precision NumPy CPU-reference; existing monolithic kernel for ports; external framework oracle only outside the hot path; small committed golden fixtures when stable.
- **Required gates by change type:** registry/fusion/plugin, CPU-reference primitive, HIP kernel port, math optimization, quant plugin, KV policy, runtime/build, public API/server, perf claim.
- **Validation matrix:** targeted RED/GREEN tests; CPU deterministic bundle; GPU smoke bundle only when GPU is explicitly clear; kernel correctness gate; milestone closure gate.
- **Definition of done for math/kernel changes:** oracle identified, RED fixture/test or no-RED rationale, targeted tests pass, CPU deterministic bundle, GPU smoke when relevant, profiler trace or blocker, WORKLOG evidence.

### Verification

- Docs/process change. Re-read `AGENTS.md` and `docs/TESTING.md` via `read`.
- Checked references with:
  `rg -n "TESTING|RED|GREEN|math changes|Correctness|Validation" AGENTS.md docs/TESTING.md docs/IMPLEMENTATION.md`

---

## 2026-05-13 — Benchmark output contract and kernel catalog/path map

### Prompt / concern

- User asked whether `docs/BENCHMARK.md` is up to date with the new testing methodology, specifically what benchmark output should carry so perf numbers are comparable and correctness-backed.
- User also asked to revisit `docs/KERNELS.md` after recent source-lineage updates: full implemented-kernel catalog, atomic vs fused sections, and a Qwen3.5 MoE/PARO prefill/decode path map with alternatives.

### Sources reviewed

- `docs/BENCHMARK.md` and `docs/TESTING.md` in this repo.
- `docs/KERNELS.md` and `docs/PLAN.md` "Kernel Port Strategy" in this repo.
- `hipengine/kernels/cpu_reference/ops.py` and `hipengine/kernels/hip_gfx1100/smoke/smoke_add.py` to list kernels/oracles actually landed in HIPENGINE.
- `~/amd-gpu-tuning/PLAN-PAROQUANT.md` and `~/amd-gpu-tuning/docs/PARO.md` for the current Qwen3.5-35B-A3B-PARO route, shape-gated prefill MoE split, graph replay caveats, 24GB compact path, and recent rejected/alternative routes.
- `~/amd-gpu-tuning/nano-vllm-amd` source inventory:
  - Committed stable Qwen/PARO set: 95 kernels in `csrc/amd/qwen35_expert.hip` + 25 kernels in `nanovllm/native/qwen35/paroquant_kernels.py` = 120 Qwen/PARO kernels, plus separate `smoke_add`.
  - Parent repo observed at `nano-vllm-amd@22405a9` with local modifications in `paroquant.py` and `paroquant_kernels.py`; six additional PARO kernels were documented as lineage-dirty/experimental, not HIPENGINE defaults.

### Files changed

- `docs/BENCHMARK.md`:
  - Added a benchmark-output contract: exact run context, correctness status/commands, repeated-run statistics, profiler/kernel summary, baseline comparison, and acceptance/rejection reason.
  - Added artifact statuses: `accepted`, `rejected_correctness`, `rejected_variance`, `blocked`.
  - Expanded microbenchmark and E2E measurement statistics requirements: samples, median/p95/min/max/stdev, warmup/measured counts, variance guard.
  - Upgraded retained benchmark JSON schema from `1` to `2` with `status`, command groups, correctness pass/fail fields, measurement samples, memory, profiler top kernels, baseline/comparison, and decision fields.
  - Clarified blocked/rejected attempts are still useful evidence but not retained performance numbers.
- `docs/KERNELS.md`:
  - Renamed to a kernel catalog + port playbook.
  - Added status legend distinguishing HIPENGINE-landed, CPU-reference-landed, lineage-green, lineage-dirty/experimental, and planned.
  - Added authoritative HIPENGINE-landed list: CPU-reference oracles (`embed`, `rmsnorm`, `linear`, `qkv_proj`, `rotate`, `attention_decode`, `o_proj`, `lm_head`) and `smoke_add` gfx1100 build/runtime smoke.
  - Added exact source-lineage kernel catalog grouped into atomic/primitive-oriented families and fused/composite families.
  - Added Qwen3.5 MoE/PARO target path map: current 24GB compact speed-best rows from parent docs, prefill route, decode route, alternative paths/caveats, and rejected standalone kernel ideas.
  - Documented six parent-worktree dirty/experimental PARO kernels separately from the committed 25-kernel PARO set.
- `docs/PLAN.md`:
  - Aligned split-plan family counts with the exact source-lineage catalog (`paged_attn_decode=13`, `group_scatter=11`, `w8a16_linear=5`, `w8a16_moe=17`, `paro_awq_gemv=7`, `fused_ops=12`).
  - Clarified total count as 120 Qwen/PARO kernels plus the separate `smoke_add` build smoke.

### Verification

CPU-only; no GPU/HIP/profiler commands.

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 scripts/check_fixtures.py
python3 - <<'PY'
from pathlib import Path
for path in [Path('docs/BENCHMARK.md'), Path('docs/KERNELS.md'), Path('docs/PLAN.md')]:
    text = path.read_text()
    assert text.endswith('\n'), f'{path} missing final newline'
print('docs sanity ok')
PY
```

Results:

- `python3 -m pytest -q`: `24 passed`.
- `scripts/check_fixtures.py`: all four CPU-reference fixtures passed with `max_abs=0`, `max_rel=0`.
- Docs sanity: `docs sanity ok`.

---

## 2026-05-13 — Add source-lineage drift checker before kernel ports

### Prompt / concern

- User asked for a way to track whether kernel or dispatch files in `~/amd-gpu-tuning/` are newer before continuing HIPENGINE ports.
- Desired workflow: see changed kernel/dispatch files, inspect child-repo commits/diffs for those files, and jump to corresponding `~/amd-gpu-tuning/WORKLOG.md` evidence entries.

### Implementation

- Added `docs/source_lineage.json`, a machine-readable manifest for external source-lineage inputs:
  - Parent repo: `/home/lhl/amd-gpu-tuning/nano-vllm-amd`.
  - Baseline ref: `22405a9` (the last manual KERNELS.md catalog audit baseline).
  - Evidence paths searched: `/home/lhl/amd-gpu-tuning/WORKLOG.md`, `docs/PARO.md`, `PLAN-PAROQUANT.md`, and `PLAN-MOE2.md`.
  - Tracked files: Qwen3.5 monolithic HIP source, extension bindings, smoke source, PARO embedded HIP, and native Qwen3.5/PARO dispatch/layout files.
- Added `scripts/check_lineage.py`:
  - Read-only; only calls `git` and reads parent docs/logs.
  - Reports current child repo branch/HEAD, per-file dirty status, last commit, commits since baseline, diffstat/patch, and evidence hits.
  - Supports filters: `--kind`, `--file`, `--diff {none,stat,patch}`, `--json`, `--fail-on-drift`.
  - Evidence search prefers exact commit SHA hits, falling back to precise file/family path hits to avoid generic/noisy matches.
- Added `tests/test_check_lineage.py` with temporary git repos to verify JSON output, evidence-hit detection, diffstat capture, and `--fail-on-drift` exit behavior.
- Updated `docs/KERNELS.md` with a "Source-lineage drift check" section and clarified that the six experimental PARO kernels were from the last manual catalog baseline rather than necessarily the current parent checkout.
- Updated `docs/IMPLEMENTATION.md` Phase 0 with the completed lineage-checker item.

### Current drift found

Command:

```bash
python3 scripts/check_lineage.py --diff stat --evidence-limit 4
```

Result summary at current parent checkout:

- Parent `nano-vllm-amd`: branch `gfx1100-qwen3.5`, HEAD `0627f8b`.
- Tracked sources: 17.
- Changed/dirty since baseline `22405a9`: 5.
- Drift files:
  - `csrc/amd/qwen35_expert.hip` — commit `6e2b19b`, +93 lines, compact WMMA buffer support.
  - `csrc/amd/extension.cpp` — commit `6e2b19b`, +8 lines, binding additions for the WMMA compact path.
  - `nanovllm/native/qwen35/paroquant_kernels.py` — commits `4864e0a`, `2cd28d5`, `6e2b19b`; diffstat `985 insertions(+), 165 deletions(-)`.
  - `nanovllm/native/qwen35/paroquant.py` — commits `2cd28d5`, `57bdb5a`, `6e2b19b`, `5f64c97`, `4751c84`, `0627f8b`; diffstat `184 insertions(+), 8 deletions(-)`.
  - `nanovllm/native/qwen35/expert.py` — commit `6e2b19b`, +19 lines.
- Relevant parent WORKLOG hits include lines around `48961`, `48968`, `49047`, `49183`, `49236`, and `49258` for the changed commits.

### Verification

CPU/read-only; no GPU/HIP/profiler commands.

```bash
python3 - <<'PY'
from pathlib import Path
bad = False
for root in ('scripts', 'tests'):
    for p in Path(root).rglob('*.py'):
        if '__pycache__' in p.parts:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if len(line) > 100:
                print(f'{p}:{i}:{len(line)}:{line}')
                bad = True
raise SystemExit(1 if bad else 0)
PY
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 scripts/check_fixtures.py
python3 scripts/check_lineage.py --file 'paroquant*' --diff stat --evidence-limit 4 >/tmp/hipengine-lineage-paroquant.txt
python3 scripts/check_lineage.py --json >/tmp/hipengine-lineage-default.json
python3 - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('/tmp/hipengine-lineage-default.json').read_text())
changed = [s for s in report['sources'] if s['changed']]
print(f'lineage json ok: tracked={len(report["sources"])} changed={len(changed)}')
assert changed
PY
git diff --check
```

Results:

- Python line-length check: pass.
- `python3 -m pytest -q`: `26 passed`.
- CPU fixtures: all four pass with `max_abs=0`, `max_rel=0`.
- Lineage JSON smoke: `tracked=17 changed=5`.
- `git diff --check`: pass.

### Next

- Before porting any real kernel family, run `python3 scripts/check_lineage.py --kind kernel --diff stat` and inspect DRIFT entries.
- For a drifted file selected for port, use `--diff patch --file '<pattern>'` and read the listed parent WORKLOG evidence before copying code.
- Do not advance `docs/source_lineage.json` baseline until HIPENGINE's catalog/port plan is intentionally refreshed and logged.

---

## 2026-05-13 — Wire OPTIMAL.md into kernel path and hygiene docs

### Prompt / concern

- User noted `~/amd-gpu-tuning/docs/OPTIMAL.md` should be up to date with the optimal PARO inference path and should likely be referenced from HIPENGINE's kernel catalog.
- User also asked to review `~/amd-gpu-tuning/AGENTS.md` for git/benchmark hygiene worth adopting in HIPENGINE.
- Follow-up explicit rule requested: before porting, check `docs/KERNELS.md` and use the lineage script to ensure the kernel catalog/path map is up to date.

### Sources reviewed

- `~/amd-gpu-tuning/docs/OPTIMAL.md`:
  - Current optimal path: compact-WMMA prefill + one-step graph-replay decode for Qwen3.5-35B-A3B-PARO.
  - Latest retained sweep: 512/128 `2557 / 115.7`, 1K/128 `2876 / 112.9`, 4K/128 `2703 / 112.0`, 32K/128 `1880 / 98.8`, 128K/128 `914 / 62.6` prefill/decode tok/s, graph/step validation true.
  - 23 base flags, long-prefill chunking overrides, graph replay caveats, and decode profiling note that AWQ/GEMV decode is the next target.
- `~/amd-gpu-tuning/AGENTS.md`:
  - Already covered by HIPENGINE: explicit staging rules, no destructive cleanup, WORKLOG with logical unit, audit-first kernel tuning, raw artifact exclusion.
  - Adopted/tightened here: do not start next logical task until previous validated unit is committed; post-run benchmark quality gates (finite logits / graph validation / sample match / memory); source-lineage check before ports.

### Files changed

- `AGENTS.md`:
  - Added summary rule: kernel catalog must stay current; before any kernel port, check `docs/KERNELS.md`, run `scripts/check_lineage.py`, and update catalog/path map if parent kernels or dispatch changed.
  - Updated Key Files entry for `docs/KERNELS.md` and added `docs/source_lineage.json`.
  - Added Before Starting step for kernel ports: run the lineage checker and inspect DRIFT commits/diffs plus parent WORKLOG/OPTIMAL evidence before copying code.
  - Tightened commit timing: do not start the next logical task until the prior validated unit is committed; commit `WORKLOG.md` with the unit that required it.
- `docs/KERNELS.md`:
  - Added `~/amd-gpu-tuning/docs/OPTIMAL.md` as the canonical current optimal route source.
  - Replaced stale 2026-05-11 compact path rows with OPTIMAL.md's 2026-05-13 compact-WMMA + graph-replay route.
  - Added base-flag summary and updated prefill route: compact WMMA from 64 tokens, grouped-stacked max tokens 4096, weighted-lane accumulation, grouped SiLU+down-rotation fusion, and long-prefill chunking overrides.
  - Clarified the 120-kernel catalog is now the baseline catalog; `scripts/check_lineage.py` reports drift after `22405a9`, so PARO/WMMA ports must refresh the exact kernel inventory before copying.
- `docs/source_lineage.json`:
  - Added `/home/lhl/amd-gpu-tuning/docs/OPTIMAL.md` to evidence paths searched by `scripts/check_lineage.py`.
- `docs/BENCHMARK.md`:
  - Added post-run quality gates: finite logits, graph replay validation, generated sample matching, prefill/decode/wall reporting, memory reporting, 24 GiB PARO usability gate, and compact comparison tables.

### Verification

CPU/read-only; no GPU/HIP/profiler commands.

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 scripts/check_fixtures.py
python3 scripts/check_lineage.py --file 'paroquant*' --diff stat --evidence-limit 4 >/tmp/hipengine-lineage-paroquant-optimal.txt
python3 scripts/check_lineage.py --json >/tmp/hipengine-lineage-default-optimal.json
python3 - <<'PY'
import json
from pathlib import Path
manifest = json.loads(Path('docs/source_lineage.json').read_text())
assert '/home/lhl/amd-gpu-tuning/docs/OPTIMAL.md' in manifest['evidence_paths']
report = json.loads(Path('/tmp/hipengine-lineage-default-optimal.json').read_text())
changed = [s for s in report['sources'] if s['changed']]
print(f'lineage json ok: tracked={len(report["sources"])} changed={len(changed)} optimal=present')
assert changed
PY
git diff --check
```

Results:

- `python3 -m pytest -q`: `26 passed`.
- CPU fixtures: all four pass with `max_abs=0`, `max_rel=0`.
- Lineage JSON smoke: `tracked=17 changed=5 optimal=present`.
- `git diff --check`: pass.

---

## 2026-05-13 — Add benchmark rollup and changelog contract

### Prompt / concern

- User requested a human-readable way to track current fastest performance, similar to `~/amd-gpu-tuning/PLAN-MOE2.md`, without relying only on JSON artifacts.
- Desired shape: `benchmarks/README.md` as the current scoreboard near `benchmarks/results/`, plus a reverse-chronological `benchmarks/CHANGELOG.md` so historical changes do not make the README unwieldy.
- User clarified changelog entries should be concise dated one-liners like: model/workload metric `old -> new`, percent gain/loss, reason, and artifact/source.

### Files changed

- Added `benchmarks/README.md`:
  - `Last updated: 2026-05-13` at the top.
  - Maintenance contract for retained benchmark rows.
  - Current fastest HIPENGINE table (empty until first accepted E2E `LLM.generate()` benchmark).
  - Source-lineage target table from `~/amd-gpu-tuning/docs/OPTIMAL.md` for Qwen3.5-35B-A3B-PARO compact-WMMA + graph-replay route.
  - External comparison baseline tables from `docs/BENCHMARK.md` / parent WORKLOG.
  - `smoke_add` listed as non-throughput build/runtime smoke.
- Added `benchmarks/CHANGELOG.md`:
  - Reverse-chronological benchmark rollup history.
  - Explicit entry format: `[scope] model / quant / workload: metric old -> new (+/-X%) due to reason/change; artifact/source.`
  - Initial entries for the scoreboard creation, source-lineage target rows, external baselines, and `smoke_add` smoke row.
- Updated `AGENTS.md`:
  - Benchmark rollup rule now requires `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and a compact artifact under `benchmarks/results/` for every retained benchmark.
  - Perf-change after-work rule now requires a changelog one-liner with old/new metric, percent delta, reason, and artifact/source.
- Updated `docs/BENCHMARK.md`:
  - Added the human-readable rollup contract.
  - Added `benchmarks/CHANGELOG.md` as the compact history layer.
  - Updated benchmark playbook to write JSON + README + CHANGELOG + WORKLOG together.

### Verification

CPU-only; no GPU/HIP/profiler commands.

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 scripts/check_fixtures.py
python3 - <<'PY'
from pathlib import Path
for path in [
    Path('AGENTS.md'),
    Path('docs/BENCHMARK.md'),
    Path('benchmarks/README.md'),
    Path('benchmarks/CHANGELOG.md'),
]:
    text = path.read_text()
    assert text.endswith('\n'), f'{path} missing final newline'
assert 'Last updated: 2026-05-13' in Path('benchmarks/README.md').read_text()
assert 'old -> new' in Path('benchmarks/CHANGELOG.md').read_text()
print('benchmark docs sanity ok')
PY
git diff --check
```

Results:

- `python3 -m pytest -q`: `26 passed`.
- CPU fixtures: all four pass with `max_abs=0`, `max_rel=0`.
- Docs sanity: `benchmark docs sanity ok`.
- `git diff --check`: pass.

---

## 2026-05-13 — Resolve rocprofv3 Python/ctypes smoke trace hang

### Prompt / concern

- User confirmed the W7900 GPU is available and asked to continue HIP debugging.
- Active blocker was `rocprofv3` hanging on the Python/ctypes `smoke_add` path before any real kernel port.

### Diagnosis

- Plain HIP smoke still passed:
  - `python3 scripts/smoke.py --mode smoke-add-hip --n 1024` → `n=1024 max_abs=0.0`.
- `rocprofv3 --kernel-trace` launched Python successfully for no-GPU scripts and for HIP malloc/copy-only snippets.
- The hang started when the profiled Python process called `build_smoke_add()`, before launching `hipengine_smoke_add_f32_kernel`.
- Reproducer: `rocprofv3 --kernel-trace -- python3 -c "import subprocess; subprocess.run(('hipcc','--version'))"` hung. `os.system('hipcc --version')` exposed nested profiler launch into `hipcc`/clang and clang aborted with:
  - `CommandLine Error: Option 'sanitizer-early-opt-ep' registered more than once!`
  - `LLVM ERROR: inconsistency in registered CommandLine options`
- Root cause: `rocprofv3` launch mode recursively preloads/profiles child processes. Our build path probed `hipcc --version` inside the profiled Python process, so the profiler entered `hipcc`/clang children.

### Fix

- Added `require_cached=True` support to `hipengine.core.build.build_hip()` so profiled smoke paths can refuse to spawn `hipcc` when the expected `.so` is absent.
- Added compiler-version environment/file support for cache-key computation without probing `hipcc`:
  - `HIPENGINE_<COMPILER>_VERSION_TEXT`
  - `HIPENGINE_COMPILER_VERSION_TEXT`
  - `HIPENGINE_<COMPILER>_VERSION_FILE`
  - `HIPENGINE_COMPILER_VERSION_FILE`
- Plumbed `compiler_version` + `require_cached` through `build_smoke_add()` and `scripts/smoke.py`:
  - `--compiler-version-file /tmp/hipengine-hipcc-version.txt`
  - `--require-cached-build`
- Updated `docs/TESTING.md`, `docs/KERNELS.md`, `docs/BENCHMARK.md`, `AGENTS.md`, and `docs/IMPLEMENTATION.md` with the profiler-safe workflow and ROCm 7.13 timestamp-field note.

### Verified profiler-safe command

```bash
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.smoke import build_smoke_add
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
artifact = build_smoke_add(load=False, compiler_version=version)
print('prebuilt', artifact.output_path)
print('exists', artifact.output_path.exists())
print('compiler', artifact.compiler_version.splitlines()[0])
PY
python3 scripts/smoke.py --mode smoke-add-hip --n 1024 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rm -rf /tmp/hipengine-smoke-add-trace-fixed
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-smoke-add-trace-fixed -- \
  python3 scripts/smoke.py --mode smoke-add-hip --n 1024 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
python3 - <<'PY'
import csv, pathlib
path = next(pathlib.Path('/tmp/hipengine-smoke-add-trace-fixed').glob('*/*_kernel_trace.csv'))
with path.open(newline='') as f:
    rows = list(csv.DictReader(f))
assert any('hipengine_smoke_add_f32_kernel' in str(row) for row in rows)
for row in rows:
    if row['Kernel_Name'] == 'hipengine_smoke_add_f32_kernel':
        print(row['Kernel_Name'], int(row['End_Timestamp']) - int(row['Start_Timestamp']), row['VGPR_Count'], row['Scratch_Size'], row['LDS_Block_Size'])
PY
```

Results:

- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/smoke-83b75faf4ae01990/smoke_add.so`.
- Cache-only smoke: `n=1024 max_abs=0.0`, `first5=[1.0, 4.0, 7.0, 10.0, 13.0]`.
- `rocprofv3` exit code: `0`.
- Raw trace (not committed): `/tmp/hipengine-smoke-add-trace-fixed/epyc/2837678_kernel_trace.csv`.
- Kernel trace summary for target kernel:
  - `Kernel_Name=hipengine_smoke_add_f32_kernel`
  - grid `1024x1x1`, workgroup `256x1x1`
  - computed `DurationNs=2480` from `End_Timestamp - Start_Timestamp`
  - `VGPR_Count=8`, `Scratch_Size=0`, `LDS_Block_Size=0`
- Trace also contained three `__amd_rocclr_copyBuffer` rows for host/device copies.

### Verification

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 scripts/check_fixtures.py
python3 scripts/check_lineage.py --kind kernel --diff stat --evidence-limit 2
python3 scripts/smoke.py --mode smoke-add-hip --n 1024 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-smoke-add-trace-fixed -- \
  python3 scripts/smoke.py --mode smoke-add-hip --n 1024 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
git diff --check
```

Results:

- `python3 -m pytest -q`: `28 passed`.
- CPU fixtures: all four pass with `max_abs=0`, `max_rel=0`.
- Lineage check still reports current parent kernel drift vs baseline `22405a9`:
  - `csrc/amd/qwen35_expert.hip` drift at `6e2b19b`.
  - `nanovllm/native/qwen35/paroquant_kernels.py` drift through `59195ed`.
- GPU smoke and profiler trace passed as above.
- `git diff --check`: pass.

---

## 2026-05-13 — Fix HIP build profile flag spelling before RMSNorm port

### Context

- While preparing the first real `norm/rmsnorm.hip` port, tested a `profile="decode"` build with the existing profile flags.
- `hipcc` failed with `clang++: error: unknown argument: '-amdgpu-unroll-threshold-local=600'` because the LLVM option must be passed after `-mllvm`, matching `nano-vllm-amd/nanovllm/native/amd/extension.py`.

### Files changed

- Updated `hipengine.core.build.PROFILES`:
  - `decode`: `-mllvm -amdgpu-unroll-threshold-local=600 -mcumode`
  - `prefill`: `-mllvm -amdgpu-unroll-threshold-local=600`
  - `baseline`: unchanged.
- Updated `docs/KERNELS.md` and `docs/PLAN.md` to record the exact profile flag spelling.
- Tightened `tests/test_build.py` to assert the `-mllvm` prefix.

### Verification

```bash
python3 -m pytest tests/test_build.py -q
python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.smoke.smoke_add import _SOURCE
from hipengine.core.build import build_hip
artifact = build_hip(
    sources=[_SOURCE],
    family='smoke_decode_flag_test',
    profile='decode',
    output_name='smoke_add.so',
    load=False,
    force=True,
)
print(artifact.output_path)
print('flags', artifact.flags)
PY
```

Results:

- `tests/test_build.py`: `5 passed`.
- Decode-profile smoke build succeeded at `/home/lhl/.cache/hipengine/build/smoke_decode_flag_test-1ea75e71405c6088/smoke_add.so`.
- Flags printed: `('-mllvm', '-amdgpu-unroll-threshold-local=600', '-mcumode')`.

---

## 2026-05-13 — Reviewed AICL-Lab/hetero-paged-infer relevance

### Scope

- Reviewed `https://github.com/AICL-Lab/hetero-paged-infer` at commit `a9765bd69aefd8a64591d930867d21ed3dd7fd90` as a potential reference for HIPENGINE's scheduler / paged-KV / tiered-memory design.
- Local read-only clone: `/tmp/pi-github-repos/AICL-Lab/hetero-paged-infer`.

### Evidence

```bash
cd /tmp/pi-github-repos/AICL-Lab/hetero-paged-infer
git rev-parse HEAD
cargo test --quiet
```

Results:

- Commit: `a9765bd69aefd8a64591d930867d21ed3dd7fd90`.
- Tests passed: `87 passed`, `13 passed`, `6 passed`, `29 passed`, `1 ignored` across cargo test binaries.
- Source size sampled with `wc -l`: `7,740` lines across `src/`, `tests/`, and `benches/`.

### Findings

- The repo is a Rust prototype around PagedAttention-style block allocation, continuous batching, memory-pressure rejection, an OpenAI-compatible server, and trait-shaped executor interfaces.
- It does **not** contain production kernels: README and architecture docs mark the GPU executor as mock and real CUDA kernels / pinned memory / async CPU-GPU overlap as planned or not implemented.
- Its KV abstraction is classic uniform fixed-page `block_table + context_len`. This is useful as a small scheduler/block-manager sanity reference, but it is less general than HIPENGINE's planned `KVLiveSpans` ABI and `KVPolicy.admission_cap()` contract for DMS / H2O / SnapKV / sliding policies.
- No architecture change adopted. If we need a future sanity check for host-only scheduler invariants, its property tests and simple `BlockPool`/`PageTable` model are a reasonable reference. For tiered/offloaded decode scheduling, APEX and Neo are more relevant research references than this repo.

### Next

- Do not port code from this repo into HIPENGINE.
- Optional future doc update: add it to `docs/PLAN.md` references only as a lightweight Rust host-shape / test-harness reference, not as a kernel or tiered offload source.

---

## 2026-05-13 — Port Qwen3.5 BF16 RMSNorm HIP family

### Scope

- Ported the first real model-layer gfx1100 kernel family into HIPENGINE: Qwen3.5 BF16 RMSNorm from `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip`.
- Source commit: `nano-vllm-amd@59195ed` (`gfx1100-qwen3.5`). The lineage checker reports drift vs baseline `22405a9`, but `git diff 22405a9..HEAD -- csrc/amd/qwen35_expert.hip` shows the RMSNorm region is not touched by the current compact-WMMA drift.

### Files changed

- Added `hipengine/kernels/hip_gfx1100/norm/rmsnorm.hip`:
  - Preserved Qwen kernel bodies for `qwen35_rmsnorm_kernel`, `qwen35_add_rmsnorm_kernel`, `qwen35_add_rmsnorm_f32_kernel`, and `qwen35_head_rmsnorm_kernel`.
  - Added HIPENGINE C ABI launch wrappers taking raw pointers, shapes, `eps`, and `hipStream_t`.
- Added `hipengine/kernels/hip_gfx1100/norm/rmsnorm.py` and exported from `norm/__init__.py`:
  - `plan_qwen35_rmsnorm_build`, `build_qwen35_rmsnorm`.
  - Raw-pointer ctypes wrappers for all four kernels.
  - Registry keys under `KernelKey("hip_gfx1100", <layer>, "bf16")` for `rmsnorm`, `add_rmsnorm`, `add_rmsnorm_f32`, and `head_rmsnorm`.
- Added `hipengine/quant/bf16.py` and registered a BF16 unquantized quant plugin.
- Added `scripts/smoke.py --mode qwen35-rmsnorm-hip` for a deterministic BF16-bit GPU smoke.
- Added `tests/test_qwen35_rmsnorm_plan.py` and updated quant/plugin tests.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md` with the landed family and smoke/profiler command.

### Correctness / profiler gate

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 scripts/check_fixtures.py
python3 scripts/check_lineage.py --file '*qwen35_expert.hip' --diff stat --evidence-limit 3
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.norm import build_qwen35_rmsnorm
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
artifact = build_qwen35_rmsnorm(load=False, compiler_version=version)
print('prebuilt', artifact.output_path)
print('exists', artifact.output_path.exists())
PY
python3 scripts/smoke.py --mode qwen35-rmsnorm-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rm -rf /tmp/hipengine-qwen35-rmsnorm-trace
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-qwen35-rmsnorm-trace -- \
  python3 scripts/smoke.py --mode qwen35-rmsnorm-hip --rows 2 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
git diff --check
```

Results:

- `python3 -m pytest -q`: `32 passed`.
- CPU fixtures: all four pass with `max_abs=0`, `max_rel=0`.
- RMSNorm GPU smoke: `rows=2 hidden_size=16 max_abs=0.0 bit_mismatch=0`.
- Kernel-body preservation check found all four source bodies verbatim in `rmsnorm.hip` (`31`, `36`, `82`, and `30` lines respectively).
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/qwen35_rmsnorm-0d9c4c5794992635/qwen35_rmsnorm.so`.
- `rocprofv3` trace (raw CSV not committed): `/tmp/hipengine-qwen35-rmsnorm-trace/epyc/2868072_kernel_trace.csv`.
- Target kernel row:
  - `(anonymous namespace)::qwen35_rmsnorm_kernel(unsigned short const*, unsigned short const*, unsigned short*, float, long)`
  - computed `DurationNs=6560`
  - `Grid_Size_X=512` (2 blocks × 256 threads), `Workgroup_Size_X=256`
  - `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`
- Lineage drift still noted for `csrc/amd/qwen35_expert.hip` at `6e2b19b`, but current diff hunks are compact-WMMA related, not RMSNorm.

---

## 2026-05-13 — Map current OPTIMAL MoE/PARO port dependencies

### Prompt / concern

- User suggested using the current `~/amd-gpu-tuning/docs/OPTIMAL.md` MoE path as the next port target so HIPENGINE can exercise the full `docs/KERNELS.md` checklist, correctness gates, and benchmark robustness against the parent performance rows.

### Source review

- Re-read `docs/KERNELS.md`, `docs/PLAN.md` kernel port strategy, latest WORKLOG entries, and `~/amd-gpu-tuning/docs/OPTIMAL.md`.
- Ran lineage check:

```bash
python3 scripts/check_lineage.py --diff stat --evidence-limit 4
```

Current parent checkout:

- `nano-vllm-amd` branch `gfx1100-qwen3.5`, HEAD `59195ed`.
- Drift vs HIPENGINE baseline `22405a9` in:
  - `csrc/amd/qwen35_expert.hip`
  - `csrc/amd/extension.cpp`
  - `nanovllm/native/qwen35/paroquant_kernels.py`
  - `nanovllm/native/qwen35/paroquant.py`
  - `nanovllm/native/qwen35/expert.py`
- Current kernel inventory from parent source:
  - `qwen35_expert.hip`: 96 `__global__` kernels (baseline had 95).
  - `paroquant_kernels.py`: 29 `__global__` kernels (baseline had 25).
  - Current Qwen/PARO total: 125 kernels excluding `smoke_add`.
- Added since baseline:
  - `qwen35_moe_wmma_tile_map_kernel`
  - `gemm_awq_selected_dual_pack8_wmma_kernel`
  - `gemm_awq_selected_pack8_wmma_kernel`
  - `gemm_awq_selected_dual_pack8_wmma_compact_kernel`
  - `gemm_awq_selected_pack8_wmma_compact_kernel`

### Docs changed

- Updated `docs/KERNELS.md`:
  - Recorded current OPTIMAL inventory as 96 + 29 = 125 kernels.
  - Added "Current OPTIMAL MoE port checklist (`nano-vllm-amd@59195ed`)".
  - Broke dependencies into source refresh deltas, MoE c=1 decode, MoE prefill compact-WMMA, full-inference dependencies outside MoE, and proposed port order.
  - Explicitly marked current HIPENGINE status: only Qwen BF16 RMSNorm subset is partial/landed; PARO RMSNorm out-kernels, router, selected GEMV, fused activation/down-rotation, W8A16 shared/lm-head, compact WMMA, attention/KV, model/plugin/loader, and eval harness remain missing.
- Updated `docs/IMPLEMENTATION.md`:
  - Added an OPTIMAL MoE/PARO reproduction exercise punchlist keyed to `docs/KERNELS.md`.

### Key conclusion

- We should not start by copying a random MoE kernel. The fastest path to a meaningful exercise is:
  1. add parent-baseline + HIPENGINE-blocked benchmark artifacts for 512/128 and 4K/128,
  2. port the MoE c=1 decode vertical slice,
  3. port the compact-WMMA prefill slice,
  4. only then close full inference with loader/model/attention/graph replay.
- Full OPTIMAL inference cannot be replicated yet because HIPENGINE still lacks `LLM.generate()`, `w4_paro` weight loading/layout, the Qwen3.5 model plugin, attention/KV/linear-attn/lm-head dependencies, and graph replay.

### Verification

```bash
python3 -m pytest -q
python3 scripts/check_fixtures.py
python3 - <<'PY'
from pathlib import Path
for path in [Path('docs/KERNELS.md'), Path('docs/IMPLEMENTATION.md')]:
    text = path.read_text()
    assert text.endswith('\n'), path
assert 'Current OPTIMAL MoE port checklist' in Path('docs/KERNELS.md').read_text()
assert 'qwen35_moe_wmma_tile_map_kernel' in Path('docs/KERNELS.md').read_text()
assert '125' in Path('docs/KERNELS.md').read_text()
print('optimal moe checklist docs sanity ok')
PY
git diff --check
```

Results:

- `python3 -m pytest -q`: `32 passed`.
- CPU fixtures: all four pass with `max_abs=0`, `max_rel=0`.
- Docs sanity: `optimal moe checklist docs sanity ok`.
- `git diff --check`: pass.

---

## 2026-05-13 — Capture OPTIMAL parent parity artifacts and blocked HIPENGINE row

### Scope

- Ran the parent `nano-vllm-amd` OPTIMAL Qwen3.5-35B-A3B-PARO command for `512/128` and `4K/128` on W7900 to validate the benchmark output shape and create concrete comparison artifacts before porting more kernels.
- Created a blocked HIPENGINE artifact for the same parity exercise so the missing dependencies are tracked in `benchmarks/results/`, not just prose.

### Parent commands

Both runs used the 23 base flags from `~/amd-gpu-tuning/docs/OPTIMAL.md`, `PYTHONPATH=nano-vllm-amd:paroquant`, `mamba run -n therock --no-capture-output`, and `--decode-use-step-graph-replay`.

```bash
cd /home/lhl/amd-gpu-tuning
# base NANOVLLM_* OPTIMAL flags from docs/OPTIMAL.md
PYTHONPATH=nano-vllm-amd:paroquant mamba run -n therock --no-capture-output \
  python3 scripts/bench_paro_native_engine.py \
    --prompt-len 512 --decode-len 128 \
    --decode-use-step-graph-replay \
    --output /tmp/hipengine-parent-optimal-512-128.json --json
PYTHONPATH=nano-vllm-amd:paroquant mamba run -n therock --no-capture-output \
  python3 scripts/bench_paro_native_engine.py \
    --prompt-len 4096 --decode-len 128 \
    --decode-use-step-graph-replay \
    --output /tmp/hipengine-parent-optimal-4k-128.json --json
```

### Results

| Engine | Shape | Prefill tok/s | Decode tok/s | Peak GiB | finite | Graph validation | Artifact |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `nano-vllm-amd@59195ed` parent | 512/128 | 2696.442 | 116.050 | 18.797 | true | graph/eager true, graph-compatible true | `benchmarks/results/2026-05-13-source-lineage-qwen35-paro-optimal-512-128.json` |
| `nano-vllm-amd@59195ed` parent | 4K/128 | 2741.489 | 113.049 | 21.644 | true | graph/eager true, graph-compatible true | `benchmarks/results/2026-05-13-source-lineage-qwen35-paro-optimal-4k-128.json` |
| HIPENGINE | OPTIMAL parity | — | — | — | not reached | blocked | `benchmarks/results/2026-05-13-hipengine-qwen35-paro-optimal-blocked.json` |

Blocked HIPENGINE reason: `LLM.generate`, `w4_paro` loader/layout, Qwen3.5 model plugin, MoE/attention/linear/lm-head dependency kernels, and graph replay are not landed yet.

### Files changed

- Added three compact benchmark artifacts under `benchmarks/results/`.
- Updated `benchmarks/README.md` source-lineage rows for 512/128 and 4K/128 to point at artifacts and use the local rerun values.
- Updated `benchmarks/CHANGELOG.md` with lineage-measured deltas and the blocked HIPENGINE row.
- Updated `docs/BENCHMARK.md` with the OPTIMAL MoE/PARO parity artifact policy.
- Updated `docs/IMPLEMENTATION.md` to mark parent/blocked artifacts complete.

### Verification

```bash
python3 - <<'PY'
import json
from pathlib import Path
paths = sorted(Path('benchmarks/results').glob('2026-05-13-*optimal*.json'))
assert len(paths) == 3, paths
for path in paths:
    data = json.loads(path.read_text())
    assert data['schema'] == 2, path
    assert data['status'] in {'accepted', 'blocked', 'rejected_correctness', 'rejected_variance'}, path
    assert data['workload']['model'] == 'Qwen3.5-35B-A3B-PARO', path
readme = Path('benchmarks/README.md').read_text()
for path in paths:
    data = json.loads(path.read_text())
    if data['status'] == 'accepted':
        assert path.name in readme, path
changelog = Path('benchmarks/CHANGELOG.md').read_text()
for path in paths:
    assert path.name in changelog, path
print('artifact sanity ok')
PY
python3 -m pytest -q
python3 scripts/check_fixtures.py
git diff --check
```

Results:

- Artifact sanity: pass.
- `python3 -m pytest -q`: `32 passed`.
- CPU fixtures: all four pass with `max_abs=0`, `max_rel=0`.
- `git diff --check`: pass.

---

## 2026-05-13 — Port PARO BF16 RMSNorm out-kernels

### Scope

- Ported the PARO-native RMSNorm caller-output kernels from `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py` at `nano-vllm-amd@59195ed`:
  - `paro_rmsnorm_out_kernel`
  - `paro_add_rmsnorm_out_kernel`
- Added HIPENGINE raw-pointer C ABI wrappers in the existing `norm/rmsnorm.hip` family:
  - `hipengine_paro_rmsnorm_out_bf16`
  - `hipengine_paro_add_rmsnorm_out_bf16`
- Added ctypes wrappers and registry keys:
  - `KernelKey("hip_gfx1100", "rmsnorm", "bf16", "paro_out")`
  - `KernelKey("hip_gfx1100", "add_rmsnorm", "bf16", "paro_out")`
  - `KernelKey("hip_gfx1100", "rmsnorm", "w4_paro", "paro_out")`
  - `KernelKey("hip_gfx1100", "add_rmsnorm", "w4_paro", "paro_out")`
- Added `scripts/smoke.py --mode paro-rmsnorm-hip`, checking both direct PARO RMSNorm and residual-add RMSNorm bit-for-bit against a NumPy BF16-bit reference.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md` to mark the PARO RMSNorm out-kernel slice landed.

### Correctness / preservation / profiler gate

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 scripts/check_fixtures.py
python3 scripts/smoke.py --mode registry
python3 scripts/smoke.py --mode cpu-fixtures
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.norm import build_qwen35_rmsnorm
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
artifact = build_qwen35_rmsnorm(load=False, compiler_version=version)
print(artifact.output_path)
PY
python3 scripts/smoke.py --mode qwen35-rmsnorm-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
python3 scripts/smoke.py --mode paro-rmsnorm-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-paro-rmsnorm-trace -- \
  python3 scripts/smoke.py --mode paro-rmsnorm-hip --rows 2 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: `32 passed`.
- CPU fixtures: all four pass with `max_abs=0`, `max_rel=0`.
- Existing Qwen RMSNorm smoke remains exact: `max_abs=0.0`, `bit_mismatch=0`.
- PARO RMSNorm smoke is bit-exact:
  - `norm_max_abs=0.0`, `norm_bit_mismatch=0`
  - `add_norm_max_abs=0.0`, `add_norm_bit_mismatch=0`
  - `residual_max_abs=0.0`, `residual_bit_mismatch=0`
- Source-body preservation check found current parent bodies verbatim in `rmsnorm.hip`:
  - `paro_rmsnorm_out_kernel`: 60 lines
  - `rounded_residual_sum`: 5 lines
  - `paro_add_rmsnorm_out_kernel`: 80 lines
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/qwen35_rmsnorm-1d3c74de02f98c59/qwen35_rmsnorm.so`.
- `rocprofv3` trace (raw CSV not committed): `/tmp/hipengine-paro-rmsnorm-trace/epyc/2903189_kernel_trace.csv`.
- Target kernel rows:
  - `paro_rmsnorm_out_kernel<unsigned short>`: computed `DurationNs=5760`, `VGPR_Count=40`, `Scratch_Size=0`, `LDS_Block_Size=1024`, `Workgroup_Size_X=256`.
  - `paro_add_rmsnorm_out_kernel<unsigned short>`: computed `DurationNs=5040`, `VGPR_Count=56`, `Scratch_Size=0`, `LDS_Block_Size=1024`, `Workgroup_Size_X=256`.

### Next

- Continue the MoE c=1 decode vertical slice: router/shared-gate, selected pack8 GEMV, fused activation/down-rotation, W8A16 shared expert, and weighted shared-gate residual combine.

---

## 2026-05-13 — Port Qwen3.5 BF16 router/shared-gate kernels

### Scope

- Ported native router top-k subset from `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip` at `nano-vllm-amd@59195ed`:
  - `qwen35_router_logits_kernel`
  - `qwen35_router_select_kernel`
- Added `hipengine/kernels/hip_gfx1100/moe/router.hip` with raw-pointer wrappers:
  - `hipengine_qwen35_router_logits_bf16`
  - `hipengine_qwen35_router_select`
  - `hipengine_qwen35_router_topk_shared_out_bf16`
- Added ctypes wrappers and registry keys:
  - `KernelKey("hip_gfx1100", "router_logits", "bf16")`
  - `KernelKey("hip_gfx1100", "router_select", "fp32")`
  - `KernelKey("hip_gfx1100", "router_topk_shared", "bf16", "out")`
  - `KernelKey("hip_gfx1100", "router_topk_shared", "w4_paro", "out")`
- Added `scripts/smoke.py --mode qwen35-router-hip`, using a deterministic BF16 hidden/combined-weight fixture and validating logits, selected top-k indices, and softmax routing weights.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md` to mark the router/shared-gate BF16 slice as partial-landed.

### Correctness / preservation / profiler gate

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 scripts/check_fixtures.py
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.moe import build_qwen35_router
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
artifact = build_qwen35_router(load=False, compiler_version=version)
print(artifact.output_path)
PY
python3 scripts/smoke.py --mode qwen35-router-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-qwen35-router-trace -- \
  python3 scripts/smoke.py --mode qwen35-router-hip --rows 2 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: `35 passed`.
- CPU fixtures: all four pass with `max_abs=0`, `max_rel=0`.
- Router GPU smoke: `logits_max_abs=0.0`, `routing_max_abs=1.4901161193847656e-08`, `selected_match=True`.
- Existing Qwen RMSNorm and PARO RMSNorm smokes still pass bit-exactly.
- Source-body preservation check found current parent bodies verbatim in `moe/router.hip`:
  - `qwen35_router_logits_kernel`: 46 lines
  - `qwen35_router_select_kernel`: 109 lines
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/qwen35_router-a65ac6ed49424f49/qwen35_router.so`.
- `rocprofv3` trace (raw CSV not committed): `/tmp/hipengine-qwen35-router-trace/epyc/2910857_kernel_trace.csv`.
- Target kernel rows:
  - `qwen35_router_logits_kernel<unsigned short>`: computed `DurationNs=3520`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=64`.
  - `qwen35_router_select_kernel`: computed `DurationNs=5920`, `VGPR_Count=40`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64`.

### Caveat / next

- This first HIPENGINE router wrapper supports BF16 hidden and BF16 combined weights. The parent accepts FP16 or BF16 hidden inputs; if the final HIPENGINE OPTIMAL route keeps FP16 router inputs, add an FP16 hidden specialization before claiming full router parity.
- Next MoE c=1 dependencies remain selected pack8 GEMV, fused activation/down-rotation, W8A16 shared expert, and weighted shared-gate residual combine.

---

## 2026-05-13 — Port PARO selected pack8 GEMV kernels

### Scope

- Ported selected-expert W4 pack8 GEMV bodies from `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py` at `nano-vllm-amd@59195ed`:
  - `gemv_awq_selected_dual_pack8_strided_kernel`
  - `gemv_awq_selected_pack8_kernel`
  - shared `awq_shift_for_pack_lane` / `PARO_PACK8_SHFL_REDUCE` helper block, including the current small-K/half-wave safety fix.
- Added `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip` with BF16 raw-pointer C ABI wrappers:
  - `hipengine_gemv_awq_selected_dual_pack8_strided_bf16`
  - `hipengine_gemv_awq_selected_dual_pack8_transposed_bf16`
  - `hipengine_gemv_awq_selected_pack8_strided_bf16`
  - `hipengine_gemv_awq_selected_pack8_transposed_bf16`
- Added ctypes wrappers and registry keys:
  - `KernelKey("hip_gfx1100", "selected_dual_pack8_gemv", "w4_paro", "strided")`
  - `KernelKey("hip_gfx1100", "selected_dual_pack8_gemv", "w4_paro", "transposed")`
  - `KernelKey("hip_gfx1100", "selected_pack8_gemv", "w4_paro", "strided")`
  - `KernelKey("hip_gfx1100", "selected_pack8_gemv", "w4_paro", "transposed")`
- Added `scripts/smoke.py --mode paro-selected-gemv-hip`, with a deterministic BF16/pack8 CPU oracle that validates dual gate/up and single/down strided/transposed layouts bit-for-bit.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md` for the selected pack8 GEMV slice.

### Correctness / preservation / profiler gate

GPU sharing note: another agent launched `scripts/bench_paro_native_engine.py` while this work was in progress. I waited until `rocm-smi --showpids --showuse --showmeminfo vram` reported `No KFD PIDs currently running` and ~28 MiB VRAM before the retained selected-GEMV smoke/profile below. Earlier overlapped smoke output was discarded as final evidence.

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 scripts/check_fixtures.py
python3 scripts/smoke.py --mode registry
python3 scripts/smoke.py --mode cpu-fixtures
python3 scripts/smoke.py --mode smoke-add-plan
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.quant import build_paro_awq_gemv
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
artifact = build_paro_awq_gemv(load=False, compiler_version=version)
print(artifact.output_path)
PY
python3 scripts/smoke.py --mode paro-selected-gemv-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-paro-selected-gemv-trace -- \
  python3 scripts/smoke.py --mode paro-selected-gemv-hip --rows 2 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: `38 passed`.
- CPU fixtures: all four pass with `max_abs=0`, `max_rel=0`.
- Source-body preservation check found current parent bodies verbatim in `quant/paro_awq_gemv.hip`:
  - `awq_shift_for_pack_lane` / `PARO_PACK8_SHFL_REDUCE`: 57 lines
  - `gemv_awq_selected_dual_pack8_strided_kernel`: 125 lines
  - `gemv_awq_selected_pack8_kernel`: 112 lines
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/paro_awq_gemv-0dc886e96bcd9cd2/paro_awq_gemv.so`.
- Selected GEMV smoke is bit-exact:
  - `dual_mismatch=0/0` for strided/transposed dual gate/up.
  - `single_mismatch=0/0` for strided/transposed single/down.
  - `dual_max_abs=0.0`, `single_max_abs=0.0`.
- `rocprofv3` trace (raw CSV not committed): `/tmp/hipengine-paro-selected-gemv-trace/epyc/2968040_kernel_trace.csv`.
- Target kernel rows:
  - `gemv_awq_selected_dual_pack8_strided_kernel<unsigned short, false>`: computed `DurationNs=20603`, `VGPR_Count=104`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64`, `Grid_Size=(128,2,1)`.
  - `gemv_awq_selected_dual_pack8_strided_kernel<unsigned short, true>`: computed `DurationNs=16722`, `VGPR_Count=112`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64`, `Grid_Size=(128,2,1)`.
  - `gemv_awq_selected_pack8_kernel<unsigned short, false>`: computed `DurationNs=12601`, `VGPR_Count=104`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64`, `Grid_Size=(64,2,1)`.
  - `gemv_awq_selected_pack8_kernel<unsigned short, true>`: computed `DurationNs=12882`, `VGPR_Count=112`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64`, `Grid_Size=(64,2,1)`.

### Caveat / next

- The selected gate/up fused rotate-out kernel is still missing; it belongs with the activation/down-rotation slice.
- Next MoE c=1 dependencies: fused SiLU/down rotation, W8A16 shared expert, and weighted shared-gate residual combine.

---

## 2026-05-13 — Port PARO SiLU/down-rotation kernels

### Scope

- Ported selected-expert activation/down-rotation kernels from `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py` at `nano-vllm-amd@59195ed`:
  - `silu_mul_dual_out_kernel`
  - `silu_mul_dual_rotate_out_kernel`
  - `silu_mul_pair_rotate_out_kernel`
- Added `hipengine/kernels/hip_gfx1100/fused/paro_silu.hip` with BF16 raw-pointer C ABI wrappers:
  - `hipengine_silu_mul_dual_out_bf16`
  - `hipengine_silu_mul_dual_rotate_out_bf16`
  - `hipengine_silu_mul_pair_rotate_out_bf16`
- Added ctypes wrappers and BF16/`w4_paro` registry keys for:
  - `KernelKey("hip_gfx1100", "silu_mul_dual", quant, "out")`
  - `KernelKey("hip_gfx1100", "silu_mul_dual_rotate", quant, "out")`
  - `KernelKey("hip_gfx1100", "silu_mul_pair_rotate", quant, "out")`
- Added `scripts/smoke.py --mode paro-silu-hip`, with a deterministic BF16 CPU oracle for packed dual SiLU, fused dual rotate, and separate gate/up pair-rotate fallback.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md` for the activation/down-rotation slice.

### Correctness / preservation / profiler gate

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 scripts/check_fixtures.py
python3 scripts/smoke.py --mode registry
python3 scripts/smoke.py --mode cpu-fixtures
python3 scripts/smoke.py --mode smoke-add-plan
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.fused import build_paro_silu
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
artifact = build_paro_silu(load=False, compiler_version=version)
print(artifact.output_path)
PY
python3 scripts/smoke.py --mode paro-silu-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-paro-silu-trace -- \
  python3 scripts/smoke.py --mode paro-silu-hip --rows 2 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: `41 passed`.
- CPU fixtures: all four pass with `max_abs=0`, `max_rel=0`.
- Source-body preservation check found current parent bodies verbatim in `fused/paro_silu.hip`:
  - `silu_mul_dual_out_kernel`: 18 lines
  - `silu_mul_dual_rotate_out_kernel`: 56 lines
  - `silu_mul_pair_rotate_out_kernel`: 56 lines
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/paro_silu-38ebcf975b9a1e88/paro_silu.so`.
- SiLU smoke is bit-exact for the deterministic fixture:
  - `dual_mismatch=0`, `dual_max_abs=0.0`
  - `dual_rotate_mismatch=0`, `dual_rotate_max_abs=0.0`
  - `pair_rotate_mismatch=0`, `pair_rotate_max_abs=0.0`
- `rocprofv3` trace (raw CSV not committed): `/tmp/hipengine-paro-silu-trace/epyc/2986071_kernel_trace.csv`.
- Target kernel rows:
  - `silu_mul_dual_out_kernel<unsigned short>`: computed `DurationNs=4200`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`, `Grid_Size=(256,1,1)`.
  - `silu_mul_dual_rotate_out_kernel<unsigned short>`: computed `DurationNs=14120`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=8`, `Grid_Size=(16,1,1)`.
  - `silu_mul_pair_rotate_out_kernel<unsigned short>`: computed `DurationNs=6000`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=8`, `Grid_Size=(16,1,1)`.

### Caveat / next

- This does not port the selected-dual GEMV fused rotate-out variant; the default c=1 path is now covered by selected GEMV followed by fused SiLU/down-rotation.
- Next MoE c=1 dependencies: W8A16 shared expert and weighted shared-gate residual combine.

---

## 2026-05-13 — Port PARO weighted/shared-gate combine kernels

### Scope

- Ported c=1 combine kernels from `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py` at `nano-vllm-amd@59195ed`:
  - `weighted_sum_out_kernel`
  - `weighted_sum_shared_gate_combine_residual_out_kernel`
  - `shared_gate_combine_out_kernel`
  - `shared_gate_combine_residual_out_kernel`
- Added `hipengine/kernels/hip_gfx1100/fused/paro_combine.hip` with BF16 value/output and FP32 weight/gate-logit C ABI wrappers:
  - `hipengine_weighted_sum_out_bf16_f32w`
  - `hipengine_weighted_sum_shared_gate_combine_residual_out_bf16_f32w`
  - `hipengine_shared_gate_combine_out_bf16`
  - `hipengine_shared_gate_combine_residual_out_bf16`
- Added ctypes wrappers and BF16/`w4_paro` registry keys for weighted sum, fused weighted shared-gate residual, and shared-gate fallback combine layers.
- Added `scripts/smoke.py --mode paro-combine-hip`, with a deterministic BF16 CPU oracle for weighted sum, fused weighted/shared/residual combine, and shared-gate fallback kernels.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md` for the weighted combine slice.

### Correctness / preservation / profiler gate

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 scripts/check_fixtures.py
python3 scripts/smoke.py --mode registry
python3 scripts/smoke.py --mode cpu-fixtures
python3 scripts/smoke.py --mode smoke-add-plan
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.fused import build_paro_combine
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
artifact = build_paro_combine(load=False, compiler_version=version)
print(artifact.output_path)
PY
python3 scripts/smoke.py --mode paro-combine-hip --rows 4 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-paro-combine-trace -- \
  python3 scripts/smoke.py --mode paro-combine-hip --rows 4 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: `44 passed`.
- CPU fixtures: all four pass with `max_abs=0`, `max_rel=0`.
- Source-body preservation check found the current parent combine block verbatim in `fused/paro_combine.hip`: 71 lines.
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/paro_combine-880f59d30e9f6d27/paro_combine.so`.
- Combine smoke is bit-exact:
  - `weighted_mismatch=0`, `weighted_max_abs=0.0`
  - `fused_mismatch=0`, `fused_max_abs=0.0`
  - `shared_mismatch=0`, `shared_max_abs=0.0`
  - `shared_residual_mismatch=0`, `shared_residual_max_abs=0.0`
- `rocprofv3` trace (raw CSV not committed): `/tmp/hipengine-paro-combine-trace/epyc/3003790_kernel_trace.csv`.
- Target kernel rows:
  - `weighted_sum_out_kernel<unsigned short, float>`: computed `DurationNs=3160`, `VGPR_Count=8`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`.
  - `weighted_sum_shared_gate_combine_residual_out_kernel<unsigned short, float>`: computed `DurationNs=3120`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`.
  - `shared_gate_combine_out_kernel<unsigned short>`: computed `DurationNs=3160`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`.
  - `shared_gate_combine_residual_out_kernel<unsigned short>`: computed `DurationNs=2400`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`.

### Caveat / next

- The scalar-weight fallback template instantiations are not wrapped yet; current OPTIMAL c=1 path uses FP32 router/routing weights and FP32 gate logits.
- Remaining MoE c=1 dependency before this vertical slice can execute end-to-end is W8A16 shared expert (gate/up/down/shared/lm-head family), plus the higher-level model/weight-loader plumbing.

---

## 2026-05-13 — Port W8A16 linear kernels

### Scope

- Ported W8A16 GEMV kernels from `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip` at `nano-vllm-amd@59195ed`:
  - `w8a16_linear_kernel`
  - `w8a16_linear_lowp_out_kernel`
  - `w8a16_linear_f32_kernel`
- Added `hipengine/kernels/hip_gfx1100/quant/w8a16_linear.hip` with raw-pointer C ABI wrappers:
  - `hipengine_w8a16_linear_bf16_f32_out`
  - `hipengine_w8a16_linear_bf16_lowp_out`
  - `hipengine_w8a16_linear_f32_f32_out`
- Used HIP `hip_bfloat16` for the lowp BF16 template instantiation so the parent kernel body's `static_cast<scalar_t>` preserves BF16 rounding semantics while the public ABI remains raw `uint16_t*` BF16 bits.
- Added ctypes wrappers and registry keys under `w8a16` and `w4_paro` quant keys for `bf16_f32_out`, `bf16_lowp_out`, and `f32_f32_out` variants.
- Added `scripts/smoke.py --mode w8a16-linear-hip`, validating BF16→FP32, BF16→BF16 lowp, and FP32→FP32 paths against deterministic NumPy oracles.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md` for the W8A16 linear slice.

### Correctness / preservation / profiler gate

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 scripts/check_fixtures.py
python3 scripts/smoke.py --mode registry
python3 scripts/smoke.py --mode cpu-fixtures
python3 scripts/smoke.py --mode smoke-add-plan
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.quant import build_w8a16_linear
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
artifact = build_w8a16_linear(load=False, compiler_version=version)
print(artifact.output_path)
PY
python3 scripts/smoke.py --mode w8a16-linear-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-w8a16-linear-trace -- \
  python3 scripts/smoke.py --mode w8a16-linear-hip --rows 2 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: `47 passed`.
- CPU fixtures: all four pass with `max_abs=0`, `max_rel=0`.
- Source-body preservation check found current parent bodies verbatim in `quant/w8a16_linear.hip`:
  - `w8a16_linear_kernel`: 47 lines
  - `w8a16_linear_lowp_out_kernel`: 48 lines
  - `w8a16_linear_f32_kernel`: 46 lines
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/w8a16_linear-617c51c3658bde8b/w8a16_linear.so`.
- W8A16 smoke results:
  - `bf16_f32_max_abs=0.0`
  - `f32_f32_max_abs=4.76837158203125e-07`
  - `lowp_mismatch=0`, `lowp_max_abs=0.0`
- `rocprofv3` trace (raw CSV not committed): `/tmp/hipengine-w8a16-linear-trace/epyc/3521718_kernel_trace.csv`.
- Target kernel rows:
  - `w8a16_linear_kernel`: computed `DurationNs=10200`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=64`, `Grid_Size=(512,2,1)`.
  - `w8a16_linear_lowp_out_kernel<hip_bfloat16>`: computed `DurationNs=8600`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=64`, `Grid_Size=(512,2,1)`.
  - `w8a16_linear_f32_kernel`: computed `DurationNs=8560`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=64`, `Grid_Size=(512,2,1)`.

### Caveat / next

- This lands the low-level W8A16 linear path used by parent shared expert and lm-head/auxiliary dense routes.
- Next step is a composite HIPENGINE shared-expert smoke chaining W8A16 gate/up → `silu_mul_dual_out` → W8A16 down, then a c=1 MoE vertical smoke that includes selected W4 experts and shared branch combine.

---

## 2026-05-13 — Add W8A16 shared-expert composite smoke

### Scope

- Added `scripts/smoke.py --mode w8a16-shared-expert-hip` to chain the current parent shared-expert lowp route with existing HIPENGINE kernels:
  1. `w8a16_linear_bf16_lowp_out`: hidden → fused gate/up BF16 scratch.
  2. `silu_mul_dual_out_bf16`: fused `SiLU(gate) * up` into BF16 intermediate.
  3. `w8a16_linear_bf16_lowp_out`: intermediate → shared expert BF16 output.
- Added deterministic NumPy oracle that stages the same BF16 rounding after gate/up and after activation before the down projection.
- Updated docs to mark the current parent lowp-linear shared-expert route as landed; specialized `w8a16_*shared*` fused kernels remain optional/future.

### GPU sharing note

- A separate sweep started between the first smoke and an attempted profile (`python3`, ~29–30 GiB VRAM). That profile was treated as contaminated and discarded.
- Re-ran after a hard no-KFD guard before smoke and profile; evidence below is the uncontended rerun.

### Validation

```bash
python3 -m compileall -q scripts/smoke.py
python3 -m pytest -q
python3 scripts/smoke.py --mode smoke-add-plan
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.quant import build_w8a16_linear
from hipengine.kernels.hip_gfx1100.fused import build_paro_silu
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_w8a16_linear(load=False, compiler_version=version).output_path)
print(build_paro_silu(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode w8a16-shared-expert-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-w8a16-shared-trace -- \
  python3 scripts/smoke.py --mode w8a16-shared-expert-hip --rows 2 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: `47 passed`.
- Prebuilt artifacts:
  - `/home/lhl/.cache/hipengine/build/w8a16_linear-617c51c3658bde8b/w8a16_linear.so`
  - `/home/lhl/.cache/hipengine/build/paro_silu-38ebcf975b9a1e88/paro_silu.so`
- Shared-expert smoke: `rows=2 hidden_size=16 intermediate_size=8 gate_up_mismatch=0 intermediate_mismatch=0 out_mismatch=0 out_max_abs=0.0`.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-w8a16-shared-trace/epyc/3643123_kernel_trace.csv`.
- Target kernel rows:
  - `w8a16_linear_lowp_out_kernel<hip_bfloat16>` gate/up launch: computed `DurationNs=12520`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=64`, `Grid_Size=(1024,2,1)`.
  - `silu_mul_dual_out_kernel<unsigned short>`: computed `DurationNs=9760`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=64`, `Grid_Size=(64,1,1)`.
  - `w8a16_linear_lowp_out_kernel<hip_bfloat16>` down launch: computed `DurationNs=6920`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=64`, `Grid_Size=(1024,2,1)`.

### Next

- Build a native c=1 MoE vertical smoke that chains: PARO RMSNorm → router/shared gate → selected W4 gate/up → SiLU/down rotation → selected W4 down → W8A16 shared branch → weighted/shared/residual combine.
