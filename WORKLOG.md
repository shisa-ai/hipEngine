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
