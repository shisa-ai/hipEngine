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

---

## 2026-05-13 — Add synthetic PARO MoE c=1 vertical smoke

### Scope

- Added `scripts/smoke.py --mode paro-moe-c1-hip --hidden-size 8`.
- The smoke chains the landed c=1 decode kernels end-to-end on a deterministic toy workload:
  1. `paro_rmsnorm_out_bf16`
  2. `qwen35_router_topk_shared_out_bf16`
  3. `gemv_awq_selected_dual_pack8_strided_bf16`
  4. `silu_mul_dual_out_bf16`
  5. `gemv_awq_selected_pack8_strided_bf16`
  6. W8A16 shared branch (`w8a16_linear_bf16_lowp_out` → `silu_mul_dual_out_bf16` → `w8a16_linear_bf16_lowp_out`)
  7. `weighted_sum_shared_gate_combine_residual_out_bf16_f32w`
- Added staged BF16 NumPy oracles for RMSNorm, router top-k/softmax, selected AWQ pack8, SiLU, W8A16 lowp, and final weighted/shared/residual combine.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md` to list the synthetic vertical smoke.

### Validation

```bash
python3 -m compileall -q scripts/smoke.py
python3 -m pytest -q
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.norm import build_qwen35_rmsnorm
from hipengine.kernels.hip_gfx1100.moe import build_qwen35_router
from hipengine.kernels.hip_gfx1100.quant import build_paro_awq_gemv, build_w8a16_linear
from hipengine.kernels.hip_gfx1100.fused import build_paro_silu, build_paro_combine
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
for name, fn in [
    ('rmsnorm', build_qwen35_rmsnorm),
    ('router', build_qwen35_router),
    ('awq', build_paro_awq_gemv),
    ('silu', build_paro_silu),
    ('w8a16', build_w8a16_linear),
    ('combine', build_paro_combine),
]:
    print(name, fn(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode paro-moe-c1-hip --hidden-size 8 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-paro-moe-c1-trace -- \
  python3 scripts/smoke.py --mode paro-moe-c1-hip --hidden-size 8 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: `47 passed`.
- Prebuilt artifacts:
  - `qwen35_rmsnorm-1d3c74de02f98c59/qwen35_rmsnorm.so`
  - `qwen35_router-a65ac6ed49424f49/qwen35_router.so`
  - `paro_awq_gemv-0dc886e96bcd9cd2/paro_awq_gemv.so`
  - `paro_silu-38ebcf975b9a1e88/paro_silu.so`
  - `w8a16_linear-617c51c3658bde8b/w8a16_linear.so`
  - `paro_combine-880f59d30e9f6d27/paro_combine.so`
- Vertical smoke output:
  - `hidden_size=8 top_k=2 norm_mismatch=0 selected_match=True logits_max_abs=0.0 routing_max_abs=0.0 selected_gate_up_mismatch=0 selected_act_mismatch=0 selected_down_mismatch=0 shared_out_mismatch=0 final_mismatch=0 final_max_abs=0.0`
  - `selected=[1, 0]`, `routing=[0.6870266199111938, 0.31297338008880615]`.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-paro-moe-c1-trace/epyc/3702994_kernel_trace.csv`.
- Target kernel rows from trace:
  - `paro_rmsnorm_out_kernel`: `DurationNs=15040`, `VGPR_Count=40`, `Scratch_Size=0`, `LDS_Block_Size=1024`, `Workgroup_Size_X=256`, `Grid_Size=(256,1,1)`.
  - `qwen35_router_logits_kernel`: `DurationNs=10600`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=512`, `Grid_Size=(2048,1,1)`.
  - `qwen35_router_select_kernel`: `DurationNs=14120`, `VGPR_Count=40`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=512`, `Grid_Size=(512,1,1)`.
  - `gemv_awq_selected_dual_pack8_strided_kernel`: `DurationNs=16440`, `VGPR_Count=104`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64`, `Grid_Size=(128,2,1)`.
  - `silu_mul_dual_out_kernel`: two launches, `DurationNs=20001` and `6720`, `VGPR_Count=16`, `Scratch_Size=0`.
  - `gemv_awq_selected_pack8_kernel` (small-K-selected strided path): `DurationNs=15121`, `VGPR_Count=104`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64`, `Grid_Size=(64,2,1)`.
  - `w8a16_linear_lowp_out_kernel<hip_bfloat16>`: two launches, `DurationNs=9561` and `8641`, `VGPR_Count=24`, `Scratch_Size=0`.
  - `weighted_sum_shared_gate_combine_residual_out_kernel`: `DurationNs=7001`, `VGPR_Count=16`, `Scratch_Size=0`, `Workgroup_Size_X=256`, `Grid_Size=(256,1,1)`.

### Next

- The synthetic c=1 kernel chain works. Remaining path to "all works" is no longer the c=1 MoE kernel dependency set; it is model/weight-loader/full-inference plumbing: w4_paro loader/layout, Qwen3.5 model plugin, non-MoE projections, attention/KV, lm-head route, and graph replay.

---

## 2026-05-13 — Register w4_paro quant plugin metadata

### Scope

- Added `hipengine.quant.w4_paro.W4ParoQuant` and registered built-in `w4_paro` quant plugin.
- Captures six quant axes for dispatch/planning:
  - weight storage: `uint4_pack8_awq`
  - activation preprocess: `bf16_pairwise_rotation`
  - compute dtype: `bf16`
  - scale granularity: `group128_per_output_channel`
  - calibration artifact: `paroquant_theta_pairs_scales`
  - kernel family: `paro_awq_pack8`
- Updated quant import surface and tests.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest -q
```

Results: `48 passed`.

### Next

- Add Qwen3.5/PARO model plugin metadata and loader-side tensor-name/layout scaffolding.

---

## 2026-05-13 — Register Qwen3.5/PARO model plugin metadata

### Scope

- Added `hipengine.models.qwen35.Qwen35ParoMoeModel` and registered it for:
  - `Qwen3_5MoeForConditionalGeneration`
  - `Qwen3_5MoeForCausalLM`
- Added metadata-only defaults for `default_quant=w4_paro` and `default_backend=hip_gfx1100`.
- Added representative full-attention and linear-attention decode layer primitive sequences using existing registry layer keys.
- Added canonical HF/PARO weight-name templates for loader scaffolding without importing torch.
- Updated model import surface and tests.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest -q
```

Results: `49 passed`.

### Next

- Start loader-side safetensors/config scaffolding for `w4_paro` tensor discovery and layout validation, still torch-free.

---

## 2026-05-13 — Add torch-free safetensors metadata loader

### Scope

- Added `hipengine.loading.safetensors` with:
  - `read_config()` for `config.json`.
  - `discover_safetensor_shards()` for single-file, unindexed multi-file, and `model.safetensors.index.json` layouts.
  - `load_weight_index()` returning tensor names, shard paths, dtypes, shapes, and byte counts without loading tensors into torch.
  - clean `MissingConfigError`, `MissingWeightsError`, and `MissingTensorError` errors.
- Added loading import surface and tests using `safetensors.numpy` fixtures.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest -q
```

Results: `53 passed`.

### Next

- Add Qwen3.5/PARO layout validator over `WeightIndex` to check required config fields and required tensor-name families before actual device loading.

---

## 2026-05-13 — Add Qwen3.5/PARO layout validator

### Scope

- Added `hipengine.loading.qwen35_paro`:
  - normalizes HF checkpoint names by stripping `model.language_model.`, `language_model.`, or `model.` prefixes.
  - parses the Qwen3.5/PARO config subset needed by loader planning.
  - enumerates required MoE c=1 tensor names for a layer, including router/shared branch, per-expert qweight/qzeros/scales triples, and shared rotation metadata.
  - validates required tensor presence and key dense tensor shapes against `WeightIndex` metadata.
- Exported validator APIs from `hipengine.loading` and added tests.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest -q
```

Results: `58 passed`.

### Next

- Add actual tensor materialization/layout planning records (pack8/W8A16 staging descriptors) that can sit between `WeightIndex` and device buffers.

---

## 2026-05-13 — Port PARO dense BF16 GEMV

### Scope

- Ported `dense_gemv_out_kernel` from `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py` at `nano-vllm-amd@59195ed`.
- Added `hipengine/kernels/hip_gfx1100/linear/dense_gemv.hip` with raw-pointer BF16 C ABI wrapper `hipengine_dense_gemv_out_bf16`.
- Added ctypes wrapper `dense_gemv_out_bf16`, registry keys for `bf16` and `w4_paro` quant variants, CPU-safe plan tests, and `scripts/smoke.py --mode dense-gemv-hip`.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.linear import build_dense_gemv
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_dense_gemv(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode dense-gemv-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-dense-gemv-trace -- \
  python3 scripts/smoke.py --mode dense-gemv-hip --rows 2 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: `61 passed`.
- Source-body preservation: `dense_gemv_out_kernel` current parent body found verbatim in the port (47 lines).
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/dense_gemv-bd2a7b8b20172459/dense_gemv.so`.
- Dense GEMV smoke: `rows=2 hidden_size=16 out_features=8 mismatch=0 max_abs=0.0`.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-dense-gemv-trace/epyc/3743505_kernel_trace.csv`.
- Target kernel row: `dense_gemv_out_kernel<unsigned short>` computed `DurationNs=7280`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=1024`, `Workgroup_Size_X=256`, `Grid_Size=(2048,2,1)`.

### Next

- Continue outside-MoE full-inference dependencies: generic PARO pack8 GEMV for non-MoE projections and attention/KV kernels.

---

## 2026-05-14 — Port generic PARO pack8 GEMV optimal kernels

### Scope

- Ported the known-good optimized generic non-selected pack8 GEMV kernels from `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py` at `nano-vllm-amd@59195ed` into the existing `paro_awq_gemv` family:
  - `gemv_awq_pack8_kernel`
  - `gemv_awq_dual_pack8_kernel`
- Kept the parent kernel bodies intact and added only raw-pointer C ABI wrappers:
  - `hipengine_gemv_awq_pack8_strided_bf16`
  - `hipengine_gemv_awq_pack8_transposed_bf16`
  - `hipengine_gemv_awq_dual_pack8_strided_bf16`
  - `hipengine_gemv_awq_dual_pack8_transposed_bf16`
- Added ctypes wrappers and registry keys:
  - `pack8_gemv` / `w4_paro` / `strided|transposed`
  - `dual_pack8_gemv` / `w4_paro` / `strided|transposed`
- Added `scripts/smoke.py --mode paro-pack8-gemv-hip` to validate generic single/dual, strided/transposed variants against the existing AWQ pack8 reference helper.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.quant import build_paro_awq_gemv
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_paro_awq_gemv(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode paro-pack8-gemv-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-paro-pack8-gemv-trace -- \
  python3 scripts/smoke.py --mode paro-pack8-gemv-hip --rows 2 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: `61 passed`.
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/paro_awq_gemv-6fed2869770219a0/paro_awq_gemv.so`.
- Generic pack8 smoke: `single_mismatch=0/0`, `dual_mismatch=0/0`, `max_abs=0.0`.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-paro-pack8-gemv-trace/epyc/3920744_kernel_trace.csv`.
- Target kernel rows:
  - `gemv_awq_pack8_kernel<uint16_t,false>`: `DurationNs=5520`, `VGPR_Count=72`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64`, `Grid_Size=(64,2,1)`.
  - `gemv_awq_pack8_kernel<uint16_t,true>`: `DurationNs=5000`, `VGPR_Count=72`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64`, `Grid_Size=(64,2,1)`.
  - `gemv_awq_dual_pack8_kernel<uint16_t,false,false>`: `DurationNs=4720`, `VGPR_Count=72`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64`, `Grid_Size=(128,2,1)`.
  - `gemv_awq_dual_pack8_kernel<uint16_t,true,true>`: `DurationNs=4880`, `VGPR_Count=72`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64`, `Grid_Size=(128,2,1)`.

### Next

- Continue porting the parent optimal full-inference stack, not redesigning it: pairwise rotation wrappers for non-MoE projections and then attention/KV kernels from `qwen35_expert.hip` / `full_attention.py` / `linear_attention.py`.

---

## 2026-05-14 — Port fused rotate→selected dual PARO pack8 GEMV

### Scope

- Ported the known-good optimized fused selected-dual pack8 rotate-out kernel from `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py` at `nano-vllm-amd@59195ed`:
  - `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel`
- Preserved the parent kernel body byte-for-byte and added only a raw-pointer C ABI wrapper:
  - `hipengine_gemv_awq_selected_dual_pack8_strided_rotate_out_bf16`
- Added ctypes wrapper and registry key:
  - `rotate+selected_dual_pack8_gemv` / `w4_paro` / `strided`
- Added `scripts/smoke.py --mode paro-selected-gemv-rotate-hip` to validate the fused path against the existing pack8 oracle after deterministic channel scaling/no-op pair rotation.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.quant import build_paro_awq_gemv
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_paro_awq_gemv(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode paro-selected-gemv-rotate-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-paro-selected-rotate-trace -- \
  python3 scripts/smoke.py --mode paro-selected-gemv-rotate-hip --rows 2 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: `61 passed`.
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/paro_awq_gemv-9d6e2c5b926292df/paro_awq_gemv.so`.
- Fused rotate-selected smoke: `mismatch=0`, `max_abs=0.0`.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-paro-selected-rotate-trace/epyc/3940715_kernel_trace.csv`.
- Target kernel row: `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel<uint16_t,false>`: `DurationNs=7361`, `VGPR_Count=96`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=64`, `Grid_Size=(128,2,1)`.

### Next

- Continue porting the parent optimal stack verbatim: pairwise `paro_rotate2/paro_rotate3` kernels and then full-attention/KV decode kernels from the committed parent path.

---

## 2026-05-14 — Port PARO rotate2/rotate3 pairwise rotation kernels

### Scope

- Ported the known-good optimized PARO pairwise rotation helpers from `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py` at `nano-vllm-amd@59195ed`:
  - `paro_rotate2_kernel`
  - `paro_rotate3_kernel`
- Preserved the parent kernel bodies byte-for-byte and added only raw-pointer C ABI wrappers:
  - `hipengine_paro_rotate2_bf16`
  - `hipengine_paro_rotate3_bf16`
- Added ctypes wrappers and registry keys:
  - `paro_rotate2` / `w4_paro` / `bf16`
  - `paro_rotate3` / `w4_paro` / `bf16`
- Added `scripts/smoke.py --mode paro-rotate-hip` to validate rotate2 and rotate3 outputs against deterministic no-op-theta channel-scaling references.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.rotary import build_paro_rotate
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_paro_rotate(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode paro-rotate-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-paro-rotate-trace -- \
  python3 scripts/smoke.py --mode paro-rotate-hip --rows 2 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: `64 passed`.
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/paro_rotate-96af3d5e223a911a/paro_rotate.so`.
- Rotate smoke: `mismatches=[0, 0, 0, 0, 0]`, `max_abs=0.0`.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-paro-rotate-trace/epyc/3959357_kernel_trace.csv`.
- Target kernel rows:
  - `paro_rotate2_kernel<uint16_t>`: `DurationNs=3400`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=4`, `Grid_Size=(8,2,2)`.
  - `paro_rotate3_kernel<uint16_t>`: `DurationNs=3160`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=4`, `Grid_Size=(8,2,3)`.

### Next

- Port the committed parent attention/KV decode kernels instead of inventing a new ABI, then adapt wrappers to HIPENGINE's `KVLiveSpans` ABI at the host boundary.

---

## 2026-05-14 — Port Qwen full-attention rotary prelude kernels

### Scope

- Ported the known-good Qwen full-attention prelude kernels from `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip` at `nano-vllm-amd@59195ed`:
  - `qwen35_partial_rotary_kernel`
  - `qwen35_head_rmsnorm_partial_rotary_kernel`
  - `qwen35_head_rmsnorm_partial_rotary_position_kernel`
- Preserved the parent kernel bodies byte-for-byte and added only raw-pointer C ABI wrappers:
  - `hipengine_qwen35_partial_rotary_f32`
  - `hipengine_qwen35_head_rmsnorm_partial_rotary_f32_bf16`
  - `hipengine_qwen35_head_rmsnorm_partial_rotary_position_f32_bf16`
- Added ctypes wrappers and registry keys for `partial_rotary` and `head_rmsnorm+partial_rotary` under `w4_paro`.
- Added `scripts/smoke.py --mode qwen35-rotary-hip` to validate partial rotary and both fused head-RMSNorm rotary variants.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.rotary import build_qwen35_rotary
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_qwen35_rotary(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode qwen35-rotary-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-qwen35-rotary-trace -- \
  python3 scripts/smoke.py --mode qwen35-rotary-hip \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: `67 passed`.
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/qwen35_rotary-6a6e995b22522d49/qwen35_rotary.so`.
- Qwen rotary smoke: `partial_max_abs=0`, `head_max_abs=2.38e-07`, `position_max_abs=2.38e-07`.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-qwen35-rotary-trace/epyc/3994302_kernel_trace.csv`.
- Target kernel rows:
  - `qwen35_partial_rotary_kernel`: `DurationNs=2880`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=0`.
  - `qwen35_head_rmsnorm_partial_rotary_kernel`: `DurationNs=3200`, `VGPR_Count=32`, `Scratch_Size=0`, `LDS_Block_Size=0`.
  - `qwen35_head_rmsnorm_partial_rotary_position_kernel`: `DurationNs=4360`, `VGPR_Count=32`, `Scratch_Size=0`, `LDS_Block_Size=0`.

### Next

- Port KV append (`qwen35_write_paged_kv_mixed_value*`) and paged full-attention decode from the committed parent kernels, adapting wrappers to HIPENGINE `KVLiveSpans` instead of changing kernel bodies.

---

## 2026-05-14 — Port Qwen linear-attention decode convolution kernels

### Scope

- Ported the known-good Qwen linear-attention decode convolution kernels from `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip` at `nano-vllm-amd@59195ed`:
  - `qwen35_linear_attn_conv_decode_kernel`
  - `qwen35_linear_attn_conv_decode_lowp_kernel`
- Preserved the parent kernel bodies byte-for-byte and added only raw-pointer C ABI wrappers:
  - `hipengine_qwen35_linear_attn_conv_decode_f32`
  - `hipengine_qwen35_linear_attn_conv_decode_bf16`
- Added ctypes wrappers and registry keys for `linear_attn_conv_decode` under `w4_paro` variants `f32` and `bf16`.
- Added `scripts/smoke.py --mode qwen35-linear-attn-conv-hip` to validate output and recurrent conv-state update for FP32 and BF16 inputs.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.linear_attn import build_qwen35_linear_attn_conv
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_qwen35_linear_attn_conv(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode qwen35-linear-attn-conv-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-qwen35-linear-attn-conv-trace -- \
  python3 scripts/smoke.py --mode qwen35-linear-attn-conv-hip \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: `70 passed`.
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/qwen35_linear_attn_conv-032cc49571a8bb4e/qwen35_linear_attn_conv.so`.
- Linear-attn conv smoke: `f32_out_max_abs=7.45e-09`, `f32_state_max_abs=0`, `bf16_out_max_abs=7.45e-09`, `bf16_state_max_abs=0`.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-qwen35-linear-attn-conv-trace/epyc/4018022_kernel_trace.csv`.
- Target kernel rows:
  - `qwen35_linear_attn_conv_decode_kernel`: `DurationNs=2960`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`.
  - `qwen35_linear_attn_conv_decode_lowp_kernel<uint16_t>`: `DurationNs=2720`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`.

### Next

- Port the parent GDN recurrent RMSNorm+gate lowp kernel (`qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel`) and then resume KV append with a `KVLiveSpans` wrapper boundary.

---

## 2026-05-14 — Port Qwen linear-attention GDN lowp recurrent RMSNorm+gate kernel

### Scope

- Ported the known-good Qwen linear-attention decode recurrent kernel from `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip` at `nano-vllm-amd@59195ed`:
  - `qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel`
- Preserved the parent kernel body byte-for-byte and added only a raw-pointer C ABI wrapper:
  - `hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16`
- Added ctypes wrapper and registry key `gdn_recurrent_rmsnorm_gate` / `w4_paro` / `bf16_lowp`.
- Added `scripts/smoke.py --mode qwen35-linear-attn-gdn-hip` to validate output and recurrent-state update against a deterministic NumPy oracle.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.linear_attn import build_qwen35_linear_attn_gdn
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_qwen35_linear_attn_gdn(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode qwen35-linear-attn-gdn-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-qwen35-linear-attn-gdn-trace -- \
  python3 scripts/smoke.py --mode qwen35-linear-attn-gdn-hip \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: all tests passed.
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/qwen35_linear_attn_gdn-c4397ddc15ac4854/qwen35_linear_attn_gdn.so`.
- GDN smoke: `out_max_abs=2.98e-08`, `state_max_abs=1.49e-08`.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-qwen35-linear-attn-gdn-trace/epyc/4043784_kernel_trace.csv`.
- Target kernel row: `qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<uint16_t>`: `DurationNs=12360`, `VGPR_Count=56`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=128`, `Grid_Size_X=256`.

### Next

- Resume KV append and paged full-attention decode with public wrappers shaped around HIPENGINE `KVLiveSpans`, while preserving parent kernel bodies internally.

---

## 2026-05-14 — Port Qwen paged KV write via KVLiveSpans wrapper

### Scope

- Added torch-free `KVLiveSpans` scaffold (`hipengine/kvcache/spans.py`) and integer/bool dtype identifiers needed for span metadata.
- Ported the known-good Qwen paged KV writer kernels from `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip` at `nano-vllm-amd@59195ed`:
  - `qwen35_write_paged_kv_kernel`
  - `qwen35_write_paged_kv_position_tensor_kernel`
  - `qwen35_write_paged_kv_mixed_value_kernel`
  - `qwen35_write_paged_kv_mixed_value_position_tensor_kernel`
- Preserved the parent kernel bodies byte-for-byte and added raw-pointer C ABI wrappers that use span-shaped inputs:
  - `hipengine_qwen35_write_paged_kv_mixed_value_bf16_spans`
  - `hipengine_qwen35_write_paged_kv_f32_spans`
- Public Python wrappers accept `KVLiveSpans`, not a raw `(block_table, context_len)` API. For this fixed-page parent bridge, `spans.base_offsets` carries the int32 physical block table and `spans.live_counts` carries the int64 device position tensor consumed by the parent position-tensor writer.
- Added registry keys `paged_kv_write` / `w4_paro` / `mixed_bf16_spans|f32_spans`.
- Added `scripts/smoke.py --mode qwen35-paged-kv-write-hip` to validate mixed BF16-value and FP32-value KV append into a paged BF16 cache.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.attention import build_qwen35_paged_kv_write
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_qwen35_paged_kv_write(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode qwen35-paged-kv-write-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-qwen35-paged-kv-write-trace -- \
  python3 scripts/smoke.py --mode qwen35-paged-kv-write-hip \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: all tests passed.
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/qwen35_paged_kv_write-3387785660a7ab69/qwen35_paged_kv_write.so`.
- KV write smoke: `mixed_mismatch=0/0`, `f32_mismatch=0/0`, `untouched_nonzero=0`.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-qwen35-paged-kv-write-trace/epyc/4123483_kernel_trace.csv`.
- Target kernel rows:
  - `qwen35_write_paged_kv_mixed_value_position_tensor_kernel<uint16_t>`: `DurationNs=3360`, `VGPR_Count=8`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`.
  - `qwen35_write_paged_kv_position_tensor_kernel`: `DurationNs=2760`, `VGPR_Count=8`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`.

### Next

- Port paged full-attention decode wrappers using the same `KVLiveSpans` public boundary, then bring over the split-K/gated-reduce variants from the parent optimal path.

---

## 2026-05-14 — Port Qwen paged full-attention context decode via KVLiveSpans wrapper

### Scope

- Ported the known-good Qwen paged full-attention context-tensor decode kernel from `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip` at `nano-vllm-amd@59195ed`:
  - `qwen35_paged_full_attn_decode_context_tensor_kernel`
- Preserved the parent kernel body byte-for-byte and added a raw-pointer C ABI wrapper that uses span-shaped inputs:
  - `hipengine_qwen35_paged_full_attn_decode_context_bf16_spans`
- Public Python wrapper accepts `KVLiveSpans`, not a raw `(block_table, context_len)` API. For this fixed-page bridge, `spans.base_offsets` carries the int32 physical block table and `spans.live_counts` carries the int64 device context-length tensor consumed by the parent context-tensor decoder.
- Added registry key `paged_attn_decode` / `w4_paro` / `bf16_context_spans`.
- Added `scripts/smoke.py --mode qwen35-paged-attn-decode-hip` to validate against a deterministic NumPy softmax oracle.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.attention import build_qwen35_paged_attn_decode
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_qwen35_paged_attn_decode(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode qwen35-paged-attn-decode-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-qwen35-paged-attn-decode-trace -- \
  python3 scripts/smoke.py --mode qwen35-paged-attn-decode-hip \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: all tests passed.
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/qwen35_paged_attn_decode-428bd101d5630017/qwen35_paged_attn_decode.so`.
- Paged attention smoke: `max_abs=2.98e-08` vs NumPy softmax oracle.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-qwen35-paged-attn-decode-trace/epyc/4144082_kernel_trace.csv`.
- Target kernel row: `qwen35_paged_full_attn_decode_context_tensor_kernel`: `DurationNs=7640`, `VGPR_Count=40`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`, `Grid_Size_X=512`.

### Next

- Port the parent split-K paged attention variants and reduce/gated-reduce kernels behind the same `KVLiveSpans` public boundary.

---

## 2026-05-14 — Port Qwen split-K paged full-attention decode/reduce via KVLiveSpans wrapper

### Scope

- Extended the Qwen paged attention family with parent long-context split-K kernels from `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip` at `nano-vllm-amd@59195ed`:
  - `paged_key_dot_vec8`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_kernel`
  - `qwen35_paged_full_attn_decode_split_k_reduce_kernel`
- Preserved the parent helper/kernel bodies byte-for-byte and added raw-pointer C ABI wrappers:
  - `hipengine_qwen35_paged_full_attn_decode_split_k_context_bf16_spans`
  - `hipengine_qwen35_paged_full_attn_decode_split_k_reduce_f32`
- Public Python wrapper `qwen35_paged_full_attn_decode_split_k_bf16_spans(...)` accepts `KVLiveSpans` and caller-provided workspaces, then runs parent split-K context + reduce in sequence.
- Added registry key `paged_attn_decode` / `w4_paro` / `bf16_split_k_spans`.
- Added `scripts/smoke.py --mode qwen35-paged-attn-split-k-hip` to validate split-K against a deterministic NumPy softmax oracle.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.attention import build_qwen35_paged_attn_decode
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_qwen35_paged_attn_decode(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode qwen35-paged-attn-split-k-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-qwen35-paged-attn-split-k-trace -- \
  python3 scripts/smoke.py --mode qwen35-paged-attn-split-k-hip \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: all tests passed.
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/qwen35_paged_attn_decode-f4fe340865e12dcb/qwen35_paged_attn_decode.so`.
- Split-K attention smoke: `max_abs=5.96e-08`, `finite_partials=True` vs NumPy softmax oracle.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-qwen35-paged-attn-split-k-trace/epyc/4168790_kernel_trace.csv`.
- Target kernel rows:
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_kernel`: `DurationNs=17320`, `VGPR_Count=32`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`, `Grid_Size_X=512`.
  - `qwen35_paged_full_attn_decode_split_k_reduce_kernel`: `DurationNs=6320`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=8`, `Grid_Size_X=16`.

### Next

- Port the parent gated split-K reduce (`qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel`) and GQA-specialized split-K context kernels.

---

## 2026-05-14 — Port Qwen split-K paged attention gated FP32 reduce

### Scope

- Extended the Qwen paged attention family with the parent gated split-K reduce path from `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip` at `nano-vllm-amd@59195ed`:
  - `sigmoid_f32`
  - `scalar_to_float_qwen35`
  - `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<float>`
- Preserved the parent helper/kernel bodies byte-for-byte for the FP32 instantiation and added a raw-pointer C ABI wrapper:
  - `hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_f32`
- Public Python wrapper `qwen35_paged_full_attn_decode_split_k_gate_f32_spans(...)` accepts `KVLiveSpans`, runs the parent split-K context kernel, then runs the parent FP32 gated reduce with caller-provided workspaces.
- Added registry key `paged_attn_decode` / `w4_paro` / `bf16_split_k_gate_f32_spans`.
- Added `scripts/smoke.py --mode qwen35-paged-attn-gate-hip` to validate against a deterministic NumPy softmax+sigmoid oracle.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.attention import build_qwen35_paged_attn_decode
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_qwen35_paged_attn_decode(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode qwen35-paged-attn-gate-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-qwen35-paged-attn-gate-trace -- \
  python3 scripts/smoke.py --mode qwen35-paged-attn-gate-hip \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: all tests passed.
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/qwen35_paged_attn_decode-741c55ba974a3b13/qwen35_paged_attn_decode.so`.
- Gated split-K smoke: `gated_max_abs=4.47e-08` vs NumPy softmax+sigmoid oracle.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-qwen35-paged-attn-gate-trace/epyc/4187263_kernel_trace.csv`.
- Target kernel rows:
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_kernel`: `DurationNs=16320`, `VGPR_Count=32`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`, `Grid_Size_X=512`.
  - `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<float>`: `DurationNs=5000`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=8`, `Grid_Size_X=16`.

### Next

- Port BF16/FP16 gated-output support without relying on `uint16_t` casts, then port the parent GQA-specialized split-K context kernels for the target long-context shape.

---

## 2026-05-14 — Add BF16 gated split-K paged attention reduce

### Scope

- Added BF16 gate/output support for the parent split-K gated reduce path in `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip`.
- The raw C ABI wrapper instantiates the preserved parent template with HIP `hip_bfloat16` and reinterprets BF16 bit buffers at the wrapper boundary, avoiding incorrect `uint16_t` numeric casts:
  - `hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_bf16`
- Added public Python wrapper and registry key:
  - `qwen35_paged_full_attn_decode_split_k_gate_bf16_spans(...)`
  - `paged_attn_decode` / `w4_paro` / `bf16_split_k_gate_bf16_spans`
- Added `scripts/smoke.py --mode qwen35-paged-attn-gate-bf16-hip` to validate BF16 output bits against a deterministic NumPy softmax+sigmoid oracle.
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.attention import build_qwen35_paged_attn_decode
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_qwen35_paged_attn_decode(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode qwen35-paged-attn-gate-bf16-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-qwen35-paged-attn-gate-bf16-trace -- \
  python3 scripts/smoke.py --mode qwen35-paged-attn-gate-bf16-hip \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: all tests passed.
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/qwen35_paged_attn_decode-1d0fa235cf1dd172/qwen35_paged_attn_decode.so`.
- BF16 gated split-K smoke: `bf16_mismatch=0`, `bf16_max_abs=0` vs NumPy softmax+sigmoid oracle rounded to BF16.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-qwen35-paged-attn-gate-bf16-trace/epyc/18310_kernel_trace.csv`.
- Target kernel rows:
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_kernel`: `DurationNs=16000`, `VGPR_Count=32`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`, `Grid_Size_X=512`.
  - `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<hip_bfloat16>`: `DurationNs=4600`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=8`, `Grid_Size_X=16`.

### Next

- Port GQA-specialized split-K context kernels (`qwen35_paged_full_attn_decode_split_k_ctx_tensor_warp_kernel`, `*_gqa_kernel<8,16,2>`) for the target long-context shape.

---

## 2026-05-14 — Port Qwen3.5 GQA-specialized split-K paged attention context kernels

### Scope

- Extended the Qwen paged attention family with the parent Qwen3.5 long-context GQA-specialized context kernels from `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip` at `nano-vllm-amd@59195ed`:
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_warp_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_kernel<8,16,2>`
- Preserved the parent kernel bodies byte-for-byte and added raw-pointer C ABI wrappers:
  - `hipengine_qwen35_paged_full_attn_decode_split_k_warp_context_bf16_spans`
  - `hipengine_qwen35_paged_full_attn_decode_split_k_gqa_context_bf16_spans`
- Added public Python wrappers and registry keys behind `KVLiveSpans`:
  - `qwen35_paged_full_attn_decode_split_k_warp_bf16_spans(...)`
  - `qwen35_paged_full_attn_decode_split_k_gqa_bf16_spans(...)`
  - `qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans(...)`
  - variants `bf16_split_k_warp_spans`, `bf16_split_k_gqa_spans`, `bf16_split_k_gqa_gate_bf16_spans`
- Added `scripts/smoke.py --mode qwen35-paged-attn-gqa-hip` to validate the warp context, grouped-GQA context, and grouped-GQA+BF16-gated reduce paths at the target Qwen3.5 full-attention shape (`num_q_heads=16`, `num_kv_heads=2`, `head_dim=256`, `block_size=256`).
- Updated `docs/KERNELS.md`, `docs/TESTING.md`, and `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.attention import build_qwen35_paged_attn_decode
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_qwen35_paged_attn_decode(load=False, compiler_version=version).output_path)
PY
python3 scripts/smoke.py --mode qwen35-paged-attn-gqa-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-qwen35-paged-attn-gqa-trace -- \
  python3 scripts/smoke.py --mode qwen35-paged-attn-gqa-hip \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- `python3 -m pytest -q`: all tests passed.
- Prebuilt artifact: `/home/lhl/.cache/hipengine/build/qwen35_paged_attn_decode-636e558f3a2069c1/qwen35_paged_attn_decode.so`.
- GQA smoke (`ctx=512`, `chunk_size=256`, `num_splits=2`): `warp_max_abs=4.1e-08`, `gqa_max_abs=4.1e-08`, `gqa_gate_bf16_mismatch=0`, `gqa_gate_bf16_max_abs=0` vs NumPy oracle.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-qwen35-paged-attn-gqa-trace/epyc/58429_kernel_trace.csv`.
- Target kernel rows:
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_warp_kernel`: `DurationNs=65401`, `VGPR_Count=40`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`, `Grid_Size_X=4096`.
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_kernel<8,16,2>`: `DurationNs=63081` and `55201`, `VGPR_Count=80`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`, `Grid_Size_X=512`.
  - `qwen35_paged_full_attn_decode_split_k_reduce_kernel`: `DurationNs=2440` and `2200`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`, `Grid_Size_X=4096`.
  - `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<hip_bfloat16>`: `DurationNs=3440`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`, `Grid_Size_X=4096`.

### Next

- Move from kernel-family parity toward runtime integration: device buffer allocation/materialization for safetensors weights and the minimal Qwen3.5/PARO decode loop that calls the landed kernel stack.

---

## 2026-05-14 — Add torch-free safetensors device materialization helpers

### Scope

- Added `hipengine/loading/materialize.py` with byte-preserving safetensors-to-device helpers:
  - `dtype_from_safetensors(...)`
  - `load_tensor_info_to_device(...)`
  - `load_tensor_to_device(...)`
  - `load_tensors_to_device(...)`
- Added owned allocation wrappers:
  - `DeviceTensorAllocation` = source `TensorInfo` + `DeviceBuffer` + raw-pointer `Tensor` view.
  - `DeviceWeightMap` = collection of owned materialized weights with deterministic reverse-order `free()`.
- The loader copies contiguous raw bytes via `hipengine.core.memory` and never imports torch. This preserves packed quantized weights and BF16 buffers without dtype conversion.
- Exported the materializer helpers from `hipengine.loading`.
- Added CPU-safe tests with a fake HIP runtime that verifies exact copied bytes and partial-allocation cleanup on failure.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_loading_materialize.py -q
python3 -m pytest -q
```

Results:

- Materializer tests: `4 passed`.
- Full test suite: all tests passed.

### Next

- Build Qwen3.5/PARO weight-plan objects on top of the safetensors index/materializer so model code can request normalized logical weights and receive device `Tensor` handles for the landed raw-pointer kernels.

---

## 2026-05-14 — Add Qwen3.5/PARO normalized device weight map for MoE c=1

### Scope

- Extended `hipengine/loading/qwen35_paro.py` with a Qwen-specific materialization layer on top of the safetensors index and generic device materializer:
  - `Qwen35ParoLayerDeviceWeights`
  - `materialize_qwen35_paro_moe_c1_layer(...)`
- The returned weight map is keyed by normalized names (for example `layers.0.mlp.experts.1.down_proj.qweight`) while preserving the original `TensorInfo` source name for diagnostics.
- Added `DType.INT16` and safetensors `I16` materialization support for PARO pair metadata tensors.
- Exported the Qwen materialization helper from `hipengine.loading`.
- Added CPU-safe tests with a fake HIP runtime that validate normalized name lookup, exact byte copies for packed weights, INT16 metadata tensors, and owned-buffer cleanup.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_loading_materialize.py tests/test_qwen35_paro_layout.py -q
python3 -m pytest -q
```

Results:

- Targeted loader tests: `10 passed`.
- Full test suite: all tests passed.

### Next

- Define a minimal Qwen3.5/PARO runtime state object that combines materialized layer weights, scratch/workspace buffers, and landed kernel wrappers into a one-token decode step.

---

## 2026-05-14 — Add torch-free runtime workspace allocator

### Scope

- Added `hipengine/runtime/workspace.py` with a named scratch allocator:
  - `RuntimeWorkspace.reserve_tensor(name, shape, dtype)` returns a raw-pointer `Tensor` backed by an owned `DeviceBuffer`.
  - Exact shape/dtype/device matches are reused; changed specs free the old buffer before replacement.
  - `RuntimeWorkspace.free()` releases owned scratch buffers in reverse allocation order.
- Added `WorkspaceAllocation` and `tensor_nbytes(...)` helpers.
- Added fixed element byte sizing via `DType.itemsize` and `dtype_itemsize(...)` for workspace-safe dtypes.
- Exported runtime workspace helpers from `hipengine.runtime` and `dtype_itemsize` from `hipengine.core`.
- Added CPU-safe fake-runtime tests for reuse, replacement/free, reverse-order cleanup, and shape/name validation.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_runtime_workspace.py tests/test_loading_materialize.py -q
python3 -m pytest -q
```

Results:

- Targeted runtime/materializer tests: `9 passed`.
- Full test suite: all tests passed.

### Next

- Use `RuntimeWorkspace` to define a minimal Qwen3.5/PARO one-token decode state: materialized layer weights + KV spans + scratch tensors + calls into the landed full-attention/MoE kernel wrappers.

---

## 2026-05-14 — Add Qwen3.5/PARO full-attention+MoE c=1 device weight map

### Scope

- Extended `Qwen35ParoConfig` with attention metadata needed for full-attention decode planning:
  - `num_attention_heads`
  - `num_key_value_heads`
  - `head_dim`
- Added full-attention required-name helpers:
  - `required_full_attention_c1_tensor_names(...)`
  - `required_full_attention_moe_c1_tensor_names(...)`
- Added validation/materialization for the combined full-attention + MoE c=1 layer slice:
  - `validate_qwen35_paro_full_attention_moe_c1_layout(...)`
  - `materialize_qwen35_paro_full_attention_moe_c1_layer(...)`
- Full-attention required names include input layernorm, q/k RMSNorm weights, rotated q/k/v PARO metadata (`qweight`, `qzeros`, `scales`, `theta`, `pairs`, `channel_scales`), and o-proj quant tensors.
- Reused the normalized-name device materialization map, so runtime code can request `layers.0.self_attn.q_proj.pairs` or the HF-prefixed equivalent and receive the same raw-pointer `Tensor` handle.
- Added CPU-safe fake-runtime tests for required-name coverage, full-attention validation, exact byte copies for o-proj weights, INT16 q/k/v rotation pairs, and cleanup.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_qwen35_paro_layout.py -q
python3 -m pytest -q
```

Results:

- Qwen layout/materialization tests: `8 passed`.
- Full test suite: all tests passed.

### Next

- Build a minimal one-token Qwen3.5/PARO decode-state scaffold that reserves the full-attention/MoE scratch tensors using `RuntimeWorkspace` and can call the already-smoked kernels with materialized device weights.

---

## 2026-05-14 — Add minimal Qwen3.5/PARO one-token decode-state scratch scaffold

### Scope

- Added `hipengine/runtime/qwen35_paro.py` with a concrete decode-state scaffold:
  - `Qwen35ParoDecodeState`
  - `Qwen35ParoAttentionScratch`
  - `Qwen35ParoMoeScratch`
- The state combines materialized normalized layer weights with `RuntimeWorkspace` scratch reservations.
- Added full-attention scratch reservations for the landed paged/split-K/gated attention kernels:
  - query/key/value/gate views
  - `partial_out`, `partial_m`, `partial_l`
  - FP32 attention output
  - BF16/FP16/FP32 gated attention output
- Added MoE c=1 scratch reservations for the landed MoE vertical path:
  - normed hidden
  - router logits
  - routing weights / selected expert IDs
  - gate/up intermediate
  - shared branch intermediate
  - final MoE output
- Exported the Qwen runtime state from `hipengine.runtime`.
- Added CPU-safe fake-runtime tests for scratch shapes, dtypes, reuse/replacement, validation errors, and cleanup.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_runtime_workspace.py -q
python3 -m pytest -q
```

Results:

- Targeted runtime tests: `10 passed`.
- Full test suite: all tests passed.

### Next

- Start wiring the one-token decode state into actual kernel-call methods: full-attention path first (`q/k/v` projection outputs + paged KV append + GQA split-K gated attention), then the MoE c=1 path using the materialized normalized weights.

---

## 2026-05-14 — Wire Qwen3.5/PARO decode-state full-attention kernel calls

### Scope

- Added first kernel-call methods to `Qwen35ParoDecodeState`:
  - `append_full_attention_kv(...)` calls `qwen35_write_paged_kv_mixed_value_bf16_spans(...)` using reserved `scratch.key` / `scratch.value` tensors and caller-provided paged KV cache tensors/spans.
  - `decode_full_attention_gqa_gate_bf16(...)` calls `qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans(...)` using reserved query/gate/partial/gated-output scratch tensors and caller-provided paged KV cache tensors/spans.
- The methods keep the public boundary span-shaped via `KVLiveSpans`; they do not expose a raw `(block_table, context_len)` API.
- Added monkeypatched CPU tests that verify exact raw pointer, shape, split, stride, scale, library, and runtime arguments without launching GPU.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_qwen35_decode_state.py -q
python3 -m pytest -q
```

Results:

- Decode-state tests: `7 passed`.
- Full test suite: all tests passed.

### Next

- Wire projection/MoE calls into the decode state: q/k/v/o PARO pack8 GEMV and the MoE c=1 kernel chain against materialized normalized weights.

---

## 2026-05-14 — Wire Qwen3.5/PARO decode-state pack8 projection calls

### Scope

- Added `Qwen35ParoDecodeState.project_pack8_bf16(...)`.
- The method normalizes HF-prefixed weight prefixes, resolves `{qweight,qzeros,scales}` from the materialized normalized weight map, and calls the landed `gemv_awq_pack8_strided_bf16(...)` wrapper.
- This covers q/k/v/o-style single PARO pack8 projections once the caller supplies the appropriate pre-rotated input/output scratch tensors.
- Added monkeypatched CPU tests that verify normalized weight lookup and exact GEMV raw pointer/shape/group-size/runtime arguments without launching GPU.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_qwen35_decode_state.py -q
python3 -m pytest -q
```

Results:

- Decode-state tests: `8 passed`.
- Full test suite: all tests passed.

### Next

- Wire the MoE c=1 calls into `Qwen35ParoDecodeState`: router, selected gate/up, selected down, W8A16 shared branch, and weighted/shared/residual combine using normalized materialized weights.

---

## 2026-05-14 — Add Qwen3.5/PARO prepared MoE c=1 device layouts

### Scope

- Added torch-free host preparation for the parent optimized MoE c=1 layouts:
  - `prepare_qwen35_paro_moe_c1_host_tensors(...)`
  - `prepared_moe_c1_tensor_names(...)`
  - `materialize_qwen35_paro_full_attention_moe_c1_prepared_layer(...)`
- Prepared tensors mirror the parent stack's load-time transformations:
  - `layers.N.mlp.router_shared_gate.weight` concatenates router expert rows with the shared-gate row.
  - `stacked_{gate,up,down}_{qweight,qzeros,scales}` stacks per-expert tensors on expert dimension 0.
  - `stacked_{gate,up,down}_qweight_pack8_decode` swaps qweight dimensions 1 and 2 for decode pack8 kernels.
- Added `load_host_array_to_device(...)` for byte-preserving materialization of prepared NumPy arrays into owned raw-pointer tensor allocations.
- Added CPU-safe tests for prepared array materialization, router/shared-gate concatenation, expert qweight stacking/transpose, prepared device tensor shapes/dtypes, and cleanup.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_loading_materialize.py tests/test_qwen35_paro_layout.py -q
python3 -m pytest -q
```

Results:

- Targeted loader tests: `15 passed`.
- Full test suite: all tests passed.

### Next

- Wire `Qwen35ParoDecodeState` MoE c=1 methods against the prepared names: combined router/shared-gate, selected dual gate/up GEMV, selected down GEMV, and weighted/shared/residual combine.

---

## 2026-05-14 — Wire Qwen3.5/PARO decode-state MoE c=1 kernel calls

### Scope

- Added MoE c=1 call methods to `Qwen35ParoDecodeState`:
  - `route_moe_topk_shared_bf16(...)` calls `qwen35_router_topk_shared_out_bf16(...)` using prepared `router_shared_gate.weight`.
  - `selected_moe_gate_up_pack8_bf16(...)` calls `gemv_awq_selected_dual_pack8_transposed_bf16(...)` using prepared `stacked_gate/up_qweight_pack8_decode`, qzeros, and scales.
  - `selected_moe_down_pack8_bf16(...)` calls `gemv_awq_selected_pack8_transposed_bf16(...)` using prepared `stacked_down_qweight_pack8_decode`, qzeros, and scales.
  - `combine_moe_c1_shared_residual_bf16(...)` calls `weighted_sum_shared_gate_combine_residual_out_bf16_f32w(...)` with the shared-gate logit addressed from the combined router logits row.
- Extended MoE scratch with `down_input` and `down_out` buffers so selected down GEMV and weighted combine have explicit owned workspaces.
- Added monkeypatched CPU tests verifying raw pointer, shape, split/top-k, packed-width, and runtime arguments for all new calls without launching GPU.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_qwen35_decode_state.py -q
python3 -m pytest -q
```

Results:

- Decode-state tests: `11 passed`.
- Full test suite: all tests passed.

### Next

- Fill the remaining in-between MoE operation gaps: fused gate/up activation + down-rotation call wiring and W8A16 shared-expert projection calls, then run an end-to-end one-token synthetic decode path over the decode-state methods.

---

## 2026-05-14 — Wire Qwen3.5/PARO MoE activation and W8A16 shared expert calls

### Scope

- Extended prepared MoE host/device layouts with parent shared-expert W8A16 tensors:
  - `shared_expert.gate_up_weight_w8a16`
  - `shared_expert.gate_up_weight_w8a16_scale`
  - `shared_expert.down_weight_w8a16`
  - `shared_expert.down_weight_w8a16_scale`
- The W8A16 preparation mirrors the parent `_quantize_w8a16_weight`: rowwise max-abs scale, round, clamp to `[-127, 127]`, and `int8` storage with FP32 scales.
- Added decode-state methods:
  - `activate_rotate_moe_down_bf16(...)` calls `silu_mul_dual_rotate_out_bf16(...)` using down-rotation metadata.
  - `shared_expert_w8a16_bf16(...)` chains W8A16 gate/up → `silu_mul_dual_out_bf16` → W8A16 down using prepared shared-expert tensors.
- Extended shared-expert scratch with explicit intermediate/output buffers.
- Added monkeypatched CPU tests for fused activation/down-rotation and W8A16 shared branch wrapper arguments.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py -q
python3 -m pytest -q
```

Results:

- Targeted Qwen tests: `23 passed`.
- Full test suite: all tests passed.

### Next

- Add a CPU-monkeypatched synthetic one-token decode-state chain that invokes the full MoE c=1 route in parent order, then move toward a tiny real-GPU integration smoke over the decode-state methods with prepared synthetic weights.

---

## 2026-05-14 — Add Qwen3.5/PARO decode-state MoE c=1 orchestrator

### Scope

- Added `Qwen35ParoDecodeState.run_moe_c1_bf16(...)`, a single parent-order one-token MoE chain over the landed call methods:
  1. router/shared-gate top-k
  2. selected gate/up pack8 GEMV
  3. fused SiLU + down rotation
  4. selected down pack8 GEMV
  5. W8A16 shared-expert branch
  6. weighted selected output + shared gate + residual combine
- The method uses prepared normalized MoE tensors and existing `RuntimeWorkspace` scratch; it remains a host wiring/orchestration layer and introduces no backend conditionals.
- Added a monkeypatched CPU test that asserts wrapper invocation order and final output handle without launching GPU.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_qwen35_decode_state.py -q
python3 -m pytest -q
```

Results:

- Decode-state tests: `14 passed`.
- Full test suite: all tests passed.

### Next

- Add a real-GPU decode-state smoke that reuses the existing synthetic `paro-moe-c1-hip` fixtures but routes through `Qwen35ParoDecodeState.run_moe_c1_bf16(...)`, then move to full-attention+MoE one-token integration.

---

## 2026-05-14 — Add decode-state GPU smoke for Qwen3.5/PARO MoE c=1

### Scope

- Fixed two runtime scratch ABI bugs exposed by the real GPU smoke:
  - `selected_experts` scratch is now `int64`, matching `qwen35_router_select_kernel` and selected GEMV kernels.
  - `router_logits` scratch is now `num_experts + 1` wide, matching the combined router/shared-gate kernel output.
- Added `scripts/smoke.py --mode paro-moe-c1-state-hip --hidden-size 8`.
- The new smoke reuses the staged synthetic c=1 MoE fixture but routes the MoE body through `Qwen35ParoDecodeState.run_moe_c1_bf16(...)` with normalized prepared device weights and `RuntimeWorkspace` scratch.
- The smoke uses identity down-rotation metadata to exercise the fused `silu_mul_dual_rotate_out` call while preserving the existing BF16 oracle.
- Updated `docs/IMPLEMENTATION.md`, `docs/TESTING.md`, and `docs/KERNELS.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest tests/test_qwen35_decode_state.py -q
python3 -m pytest -q
python3 scripts/smoke.py --mode paro-moe-c1-state-hip --hidden-size 8 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-paro-moe-c1-state-trace -- \
  python3 scripts/smoke.py --mode paro-moe-c1-state-hip --hidden-size 8 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- Decode-state tests: `14 passed`.
- Full test suite: all tests passed.
- Decode-state MoE GPU smoke: `norm_mismatch=0`, `selected_match=True`, `logits_max_abs=0.0`, `routing_max_abs=0.0`, `gate_up_mismatch=0`, `down_input_mismatch=0`, `down_out_mismatch=0`, `shared_out_mismatch=0`, `final_mismatch=0`, `final_max_abs=0.0`.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-paro-moe-c1-state-trace/epyc/220036_kernel_trace.csv`.
- Target kernel rows all had `Scratch_Size=0` except expected LDS use:
  - `paro_rmsnorm_out_kernel<unsigned short>`: `DurationNs=4880`, `VGPR_Count=40`, `Scratch_Size=0`, `LDS_Block_Size=1024`.
  - `qwen35_router_logits_kernel<unsigned short>`: `DurationNs=3240`, `VGPR_Count=24`, `Scratch_Size=0`.
  - `qwen35_router_select_kernel`: `DurationNs=4200`, `VGPR_Count=40`, `Scratch_Size=0`, `LDS_Block_Size=512`.
  - `gemv_awq_selected_dual_pack8_strided_kernel<unsigned short, true>`: `DurationNs=8760`, `VGPR_Count=112`, `Scratch_Size=0`, `LDS_Block_Size=512`.
  - `silu_mul_dual_rotate_out_kernel<unsigned short>`: `DurationNs=3520`, `VGPR_Count=24`, `Scratch_Size=0`.
  - `gemv_awq_selected_pack8_kernel<unsigned short, true>`: `DurationNs=5240`, `VGPR_Count=112`, `Scratch_Size=0`, `LDS_Block_Size=512`.
  - `w8a16_linear_lowp_out_kernel<hip_bfloat16>`: two launches, `DurationNs=3320` and `2200`, `VGPR_Count=24`, `Scratch_Size=0`.
  - `silu_mul_dual_out_kernel<unsigned short>`: `DurationNs=11240`, `VGPR_Count=16`, `Scratch_Size=0`.
  - `weighted_sum_shared_gate_combine_residual_out_kernel<unsigned short, float>`: `DurationNs=2520`, `VGPR_Count=16`, `Scratch_Size=0`.

### Next

- Wire a one-token full-attention+MoE decode-state smoke: projection outputs → KV append → GQA split-K gated attention → MoE c=1 orchestrator.

---

## 2026-05-14 — Add decode-state GPU smoke for Qwen3.5 GQA attention

### Scope

- Added `scripts/smoke.py --mode qwen35-paged-attn-gqa-state-hip`.
- The smoke validates the runtime-state full-attention path through `Qwen35ParoDecodeState`:
  1. reserve full-attention scratch for Qwen3.5 shape (`num_q_heads=16`, `num_kv_heads=2`, `head_dim=256`)
  2. append one FP32-K/BF16-V token into a paged BF16 KV cache via `append_full_attention_kv(...)`
  3. update the span count tensor from append position to decode context length
  4. run `decode_full_attention_gqa_gate_bf16(...)` over `ctx=512`, `chunk_size=256`, `num_splits=2`
- The fixture validates both the appended cache row and BF16 gated attention output against a NumPy softmax+sigmoid oracle.
- Updated `docs/IMPLEMENTATION.md`, `docs/TESTING.md`, and `docs/KERNELS.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 scripts/smoke.py --mode qwen35-paged-attn-gqa-state-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-qwen35-paged-attn-gqa-state-trace -- \
  python3 scripts/smoke.py --mode qwen35-paged-attn-gqa-state-hip \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Results:

- Full test suite: all tests passed.
- Decode-state GQA attention GPU smoke: `appended_key_mismatch=0`, `appended_value_mismatch=0`, `gqa_gate_bf16_mismatch=0`, `gqa_gate_bf16_max_abs=0`.
- Uncontended `rocprofv3` trace: `/tmp/hipengine-qwen35-paged-attn-gqa-state-trace/epyc/229594_kernel_trace.csv`.
- Target kernel rows:
  - `qwen35_write_paged_kv_mixed_value_position_tensor_kernel<unsigned short>`: `DurationNs=3080`, `VGPR_Count=8`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`, `Grid_Size_X=512`.
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_kernel<8,16,2>`: `DurationNs=67560`, `VGPR_Count=80`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`, `Grid_Size_X=512`, `Grid_Size_Y=2`.
  - `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<hip_bfloat16>`: `DurationNs=2760`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`, `Grid_Size_X=4096`.

### Next

- Combine the two state smokes into a one-token attention→MoE smoke: GQA gated attention output feeds `run_moe_c1_bf16(...)` with prepared MoE weights.

---

## 2026-05-14 — Prepare real Qwen3.5/PARO runtime loading for E2E generate

### Scope

- Discovered the local real PARO target checkpoint:
  - `/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd`
  - Single safetensors payload, ~20 GiB via HF cache symlink.
  - Config: `quant_method=paroquant`, `num_hidden_layers=40`, `hidden_size=2048`, `num_attention_heads=16`, `num_key_value_heads=2`, `head_dim=256`, `num_experts=256`, `num_experts_per_tok=8`, `moe_intermediate_size=512`, `shared_expert_intermediate_size=512`, layer pattern `linear_attention, linear_attention, linear_attention, full_attention, ...`.
- Fixed `Qwen35ParoDecodeState.project_pack8_bf16(...)` to infer generic strided PARO `out_packed` from qweight's last dimension (`[in_features, out_packed]`), matching the real checkpoint layout.
- Added explicit runtime host materialization helpers for BF16 bit buffers:
  - `float_array_to_bf16_bits(...)`
  - `load_host_array_to_device_as_dtype(...)`
- Added a runtime-focused Qwen3.5/PARO layer materializer:
  - `materialize_qwen35_paro_full_attention_moe_c1_runtime_layer(...)`
  - Converts F16 checkpoint tensors consumed by BF16 raw-pointer kernels into rounded BF16 bit buffers.
  - Omits per-expert individual tensors that are replaced by stacked/pack8 prepared layouts.
- Added runtime tensor-name helpers for the current decode-state path.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_loading_materialize.py tests/test_qwen35_paro_layout.py tests/test_qwen35_decode_state.py -q
python3 -m pytest -q
python3 - <<'PY'
from hipengine.core.hip import get_hip_runtime
from hipengine.loading.safetensors import load_weight_index
from hipengine.loading.qwen35_paro import materialize_qwen35_paro_full_attention_moe_c1_runtime_layer
model='/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd'
runtime=get_hip_runtime()
idx=load_weight_index(model)
layer=materialize_qwen35_paro_full_attention_moe_c1_runtime_layer(idx, layer_id=3, runtime=runtime)
try:
    print('layer_id=', layer.layer_id, 'tensor_count=', len(layer.weights.tensors))
    for name in [
        'layers.3.self_attn.q_proj.qweight',
        'layers.3.self_attn.q_proj.scales',
        'layers.3.mlp.router_shared_gate.weight',
        'layers.3.mlp.experts.stacked_gate_qweight_pack8_decode',
        'layers.3.mlp.experts.stacked_gate_scales',
        'layers.3.mlp.shared_expert.gate_up_weight_w8a16',
    ]:
        t=layer.tensor(name)
        print(name, t.shape, t.dtype.value)
finally:
    layer.free(runtime=runtime)
PY
```

Results:

- Targeted tests: `33 passed`.
- Full test suite: all tests passed.
- Real checkpoint layer-3 runtime materialization succeeded on idle W7900 and freed allocations:
  - `tensor_count=42`
  - `layers.3.self_attn.q_proj.qweight`: `(2048, 1024) int32`
  - `layers.3.self_attn.q_proj.scales`: `(16, 8192) bf16`
  - `layers.3.mlp.router_shared_gate.weight`: `(257, 2048) bf16`
  - `layers.3.mlp.experts.stacked_gate_qweight_pack8_decode`: `(256, 64, 2048) int32`
  - `layers.3.mlp.experts.stacked_gate_scales`: `(256, 16, 512) bf16`
  - `layers.3.mlp.shared_expert.gate_up_weight_w8a16`: `(1024, 2048) int8`

### Next

- Port the remaining decode-state host path needed for full real-model generate:
  - rotated PARO projections for q/k/v/o and linear-attention projections,
  - linear-attention decode state wiring,
  - final norm + lm-head + argmax/tokenizer loop.

---

## 2026-05-14 — Add Qwen3.5/PARO linear-attention runtime loading slice

### Scope

- Extended `Qwen35ParoConfig` with the real-model metadata needed beyond full-attention layers:
  - vocab size, RMSNorm eps, RoPE theta/rotary dim,
  - linear-attention key/value head counts and dims,
  - linear convolution kernel width.
- Added required/runtime tensor name helpers for Qwen3.5/PARO linear-attention+MoE c=1 layers.
- Added `validate_qwen35_paro_linear_attention_moe_c1_layout(...)`.
- Added `materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer(...)`.
- Runtime materialization converts:
  - PARO W4 scales/theta/channel-scales and dense a/b projection weights to BF16 bit buffers,
  - linear-attention `conv1d.weight`, `A_log`, `dt_bias`, and `linear_attn.norm.weight` to FP32 buffers for the existing conv/GDN kernels,
  - MoE expert tensors to stacked/pack8 prepared runtime layout.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_qwen35_paro_layout.py -q
python3 -m pytest -q
python3 - <<'PY'
from hipengine.core.hip import get_hip_runtime
from hipengine.loading.safetensors import load_weight_index
from hipengine.loading.qwen35_paro import materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer
model='/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd'
runtime=get_hip_runtime()
idx=load_weight_index(model)
layer=materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer(idx, layer_id=0, runtime=runtime)
try:
    print('layer_id=', layer.layer_id, 'tensor_count=', len(layer.weights.tensors))
    for name in [
        'layers.0.linear_attn.in_proj_qkv.qweight',
        'layers.0.linear_attn.in_proj_qkv.scales',
        'layers.0.linear_attn.in_proj_a.weight',
        'layers.0.linear_attn.conv1d.weight',
        'layers.0.linear_attn.A_log',
        'layers.0.linear_attn.norm.weight',
        'layers.0.mlp.experts.stacked_gate_qweight_pack8_decode',
    ]:
        t=layer.tensor(name)
        print(name, t.shape, t.dtype.value)
finally:
    layer.free(runtime=runtime)
PY
```

Results:

- Qwen3.5/PARO layout tests: all passed.
- Full test suite: all passed.
- Real checkpoint layer-0 runtime materialization succeeded on idle W7900:
  - `tensor_count=43`
  - `layers.0.linear_attn.in_proj_qkv.qweight`: `(2048, 1024) int32`
  - `layers.0.linear_attn.in_proj_qkv.scales`: `(16, 8192) bf16`
  - `layers.0.linear_attn.in_proj_a.weight`: `(32, 2048) bf16`
  - `layers.0.linear_attn.conv1d.weight`: `(8192, 1, 4) fp32`
  - `layers.0.linear_attn.A_log`: `(32,) fp32`
  - `layers.0.linear_attn.norm.weight`: `(128,) fp32`
  - `layers.0.mlp.experts.stacked_gate_qweight_pack8_decode`: `(256, 64, 2048) int32`

### Next

- Wire the linear-attention decode-state call chain over these materialized tensors:
  `paro_rotate2 -> in_proj_qkv/z pack8 GEMV -> dense a/b GEMV -> conv decode -> GDN recurrent RMSNorm+gate -> rotated out_proj`.

---

## 2026-05-14 — Wire Qwen3.5/PARO linear-attention decode-state chain through GDN

### Scope

- Added `Qwen35ParoLinearAttentionScratch` and `Qwen35ParoDecodeState.reserve_linear_attention_scratch(...)`.
- Wired the parent-order c=1 linear-attention decode-state chain through the existing raw-pointer wrappers:
  1. `paro_rotate2_bf16` for shared input rotation into qkv/z inputs
  2. PARO pack8 GEMV for `linear_attn.in_proj_qkv` and `linear_attn.in_proj_z`
  3. BF16 dense GEMV for `linear_attn.in_proj_a` and `linear_attn.in_proj_b`
  4. BF16-input linear-attention convolution decode
  5. BF16-gated GDN recurrent RMSNorm+gate
- Added `run_linear_attention_state_bf16(...)` orchestrator that returns the FP32 recurrent/GDN output. The rotated output projection is still the next slice.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_qwen35_decode_state.py -q
python3 -m pytest -q
python3 - <<'PY'
# Real layer-0 GPU smoke, zero hidden/state inputs.
# Materializes /models/.../Qwen3.5-35B-A3B-PARO layer 0 and runs:
# rotate2 -> qkv/z pack8 GEMV -> a/b dense GEMV -> conv -> GDN.
PY
```

Results:

- Decode-state tests: `16 passed`.
- Full test suite: all tests passed.
- Real checkpoint layer-0 GPU smoke completed on idle W7900:
  - `linear_attn_out (1, 4096) fp32`

### Next

- Add the missing single-output PARO rotation / F32→BF16 cast glue for linear-attention `out_proj`, then feed the result into post-attention RMSNorm + MoE for a real first-layer partial generate smoke.

---

## 2026-05-14 — Wire Qwen3.5/PARO linear-attention out_proj

### Scope

- Added torch-free gfx1100 runtime cast helpers (`f32_to_bf16`, `bf16_to_f32`) for small projection glue buffers.
- Added `paro_rotate1_bf16`, the single-output PARO pairwise rotation specialization needed by projection tails.
- Extended `Qwen35ParoLinearAttentionScratch` with recurrent BF16, rotated `out_proj` input, and projected output buffers.
- Added `project_linear_attention_out_bf16(...)` and `run_linear_attention_out_proj_bf16(...)` to cast GDN FP32 output, rotate `linear_attn.out_proj`, and run the generic pack8 PARO GEMV.
- Updated `docs/IMPLEMENTATION.md` and `docs/KERNELS.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_qwen35_decode_state.py -q
python3 -m pytest -q
(rocm-smi --showpids --showuse --showmeminfo vram || true) | sed -n '1,120p'
python3 - <<'PY'
# Real layer-0 GPU smoke on /models/.../Qwen3.5-35B-A3B-PARO:
# materialize linear-attention+MoE runtime tensors, zero BF16 hidden, zero FP32 conv/recurrent states,
# run rotate2 -> qkv/z pack8 -> a/b dense -> conv -> GDN -> f32_to_bf16 -> rotate1 -> out_proj pack8.
PY
```

Results:

- Decode-state tests: `18 passed`.
- Full test suite: all tests passed.
- GPU was idle before the smoke (`GPU use 0%`, no KFD PIDs).
- Real checkpoint layer-0 out-projection smoke completed on W7900:
  - `linear_attn_out_proj (1, 2048) bf16`, first output BF16 words all zero for zero input/state.

### Next

- Feed this projected linear-attention output into post-attention PARO add-RMSNorm + c=1 MoE to produce a full layer-0 BF16 output, then build the minimal real-model token loop.

---

## 2026-05-14 — Wire Qwen3.5/PARO linear-attention+MoE layer chain

### Scope

- Added decode-state helpers for PARO input RMSNorm and post-attention add-RMSNorm over caller-owned scratch buffers.
- Extended MoE scratch with a residual buffer and linear-attention scratch with an input-normalized buffer.
- Added `run_linear_attention_moe_c1_layer_bf16(...)`, matching the parent decode order:
  input RMSNorm → linear-attention out_proj → post-attention add-RMSNorm → c=1 MoE shared/residual combine.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_qwen35_decode_state.py -q
python3 -m pytest -q
(rocm-smi --showpids --showuse --showmeminfo vram || true) | sed -n '1,120p'
python3 - <<'PY'
# Real layer-0 GPU smoke on /models/.../Qwen3.5-35B-A3B-PARO:
# materialize runtime layer-0 tensors, zero hidden/state inputs, run full linear-attention+MoE layer chain.
PY
```

Results:

- Decode-state tests: `19 passed`.
- Full test suite: all tests passed.
- GPU was idle before the smoke (`GPU use 0%`, no KFD PIDs).
- Real checkpoint layer-0 full chain completed on W7900:
  - `linear_layer_out (1, 2048) bf16 nonzero=0`, first output BF16 words all zero for zero input/state.

### Next

- Add a real-model partial decode/generate harness around embeddings/layer sequencing/lm-head so the validated layer chain can advance an actual token.

---

## 2026-05-14 — Wire Qwen3.5/PARO full-attention+MoE layer chain

### Scope

- Extended full-attention scratch with input rotation, Q/K/V projection, BF16→FP32 head-norm inputs, and output-projection buffers.
- Added full-attention decode-state helpers for:
  input RMSNorm → PARO rotate3 → Q/K/V pack8 projections → Q/K head RMSNorm+partial RoPE → paged KV append/decode with BF16 gate → PARO `o_proj` → post-attention add-RMSNorm → c=1 MoE.
- Added runtime materialization for rotated `self_attn.o_proj` metadata (`theta`, `pairs`, `channel_scales`).
- Materialized `self_attn.q_norm/k_norm.weight` as Qwen delta weights (`weight - 1`) for the preserved Qwen head-RMSNorm kernel ABI.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py -q
python3 -m pytest -q
(rocm-smi --showpids --showuse --showmeminfo vram || true) | awk '/GPU use|No KFD|PID|python3|VRAM Total Used/ {print}'
python3 - <<'PY'
# Real layer-3 GPU smoke on /models/.../Qwen3.5-35B-A3B-PARO:
# materialize runtime full-attention layer tensors, zero hidden/KV inputs, one-token spans,
# run full-attention+MoE layer chain.
PY
```

Results:

- Decode-state + layout tests: `34 passed`.
- Full test suite: all tests passed.
- Waited for GPU to become idle before the smoke (`No KFD PIDs currently running`).
- Real checkpoint layer-3 full-attention chain completed on W7900:
  - `full_layer_out (1, 2048) bf16 nonzero=0`, first output BF16 words all zero for zero input/KV.

### Next

- Build the minimal all-layer real-model decode harness: embedding lookup, layer-state/KV allocation, final norm, lm-head, and tokenizer-visible next-token output.

---

## 2026-05-14 — Add and run real Qwen3.5/PARO one-token next-token harness

### Scope

- Added `scripts/qwen35_paro_next_token.py`, a torch-free bring-up harness that:
  - reads the real checkpoint/tokenizer metadata,
  - embeds one token from `embed_tokens.weight`,
  - runs all Qwen3.5/PARO decode layers through HIPENGINE linear-attention/full-attention c=1 layer chains,
  - applies final PARO RMSNorm on GPU,
  - computes `lm_head.weight @ hidden` argmax on CPU with NumPy chunks (temporary correctness path, not a perf path),
  - emits JSON with layer sequence and decoded next-token text.
- Updated `docs/IMPLEMENTATION.md`.

### Validation

```bash
python3 -m py_compile scripts/qwen35_paro_next_token.py
(rocm-smi --showpids --showuse --showmeminfo vram || true) | awk '/GPU use|No KFD|PID|python3|VRAM Total Used/ {print}'
python3 scripts/qwen35_paro_next_token.py --max-layers 1 --token-id 9707 --lm-head-chunk 8192
python3 scripts/qwen35_paro_next_token.py --max-layers 4 --token-id 9707 --lm-head-chunk 8192
python3 scripts/qwen35_paro_next_token.py --token-id 9707 --lm-head-chunk 8192
```

Results on W7900 / real checkpoint `/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd`:

- GPU was idle before runs (`GPU use 0%`, no KFD PIDs).
- 1-layer smoke: layer 0 linear attention completed; next token `62406` (`"ullo"`), CPU lm-head logit `6.594387054443359`.
- 4-layer smoke: layers 0-2 linear + layer 3 full attention completed; next token `23243` (`"-car"`), CPU lm-head logit `6.177801132202148`.
- All-layer smoke: all 40 layers completed (`linear_attention` x30, `full_attention` x10); next token `76323` (`"arra"`), CPU lm-head logit `7.267126083374023`.

### Caveats / Next

- This is a one-token decode smoke, not proper multi-token prefill: prompt tokenization currently selects one input token (or `--token-id`).
- `lm_head` is CPU NumPy argmax for bring-up only; port/use the GPU lm-head route before making performance claims.
- Next: wire this harness behind `LLM.generate()` or a smoke mode, then add persistent per-layer state/KV for multi-token generation.

---

## 2026-05-14 — Wire Qwen3.5/PARO E2E harness through resident GPU path

### Scope

- Refactored `scripts/qwen35_paro_next_token.py` into reusable `Qwen35ParoNextTokenRunner` plus a thin CLI.
- Added a generation registry and wired the Qwen3.5/PARO one-token path through `LLM.generate()` without engine-level backend/quant branches.
- Added a resident all-layer mode for the harness so all 40 layer states can be materialized before execution.
- Added cached safetensors shard handles plus progress-visible materialization events for direct tensors, expert stacking, prepared tensors, and layer execution.
- Added a GPU FP16 lm-head + GPU two-stage argmax kernel (`lm_head_fp16_argmax_bf16`) so the E2E harness no longer needs CPU NumPy for final-token selection.
- Updated `docs/IMPLEMENTATION.md` and `docs/KERNELS.md`.

### Validation

```bash
python3 -m py_compile hipengine/loading/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_paro_next_token.py scripts/smoke.py hipengine/llm.py hipengine/generation/qwen35_paro.py
python3 -m pytest tests/test_qwen35_paro_layout.py tests/test_lm_head_plan.py tests/test_llm_generate.py tests/test_model_quant_and_imports.py -q
python3 scripts/smoke.py --mode lm-head-hip --hidden-size 32
hipcc --version > /tmp/hipengine-hipcc-version.txt
rocprofv3 --kernel-trace -d /tmp/hipengine-lm-head-rocprof -f csv -- python3 scripts/smoke.py --mode lm-head-hip --hidden-size 32 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_paro_next_token.py --max-layers 1 --token-id 9707 --resident-layers --lm-head gpu_fp16_argmax --progress
python3 scripts/qwen35_paro_next_token.py --token-id 9707 --resident-layers --lm-head gpu_fp16_argmax --progress
```

Results on W7900 / real checkpoint `/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd`:

- Focused tests: `20 passed` for layout + lm-head + LLM dispatch subsets.
- Standalone GPU lm-head smoke: `lm_head_id=29`, `expected_id=29`, `index_match=True`, logit abs `0.0`.
- `rocprofv3` kernel trace captured `lm_head_fp16_logits_kernel`, `argmax_stage1_kernel`, and `argmax_stage2_kernel`, all with `Scratch_Size=0`.
- One-layer resident harness with GPU lm-head matched the earlier CPU sanity output: next token `62406` (`"ullo"`), logit `6.594387054443359`.
- All-layer resident harness completed all 40 layers (`linear_attention` x30, `full_attention` x10) and GPU lm-head/argmax:
  - next token `76323` (`"arra"`), GPU lm-head logit `7.267126560211182`.
  - This matches the prior CPU-argmax token; logit differs only by FP32 roundoff vs prior CPU value `7.267126083374023`.

### Caveats / Next

- This is still one-token decode bring-up, not full multi-token generation/prefill.
- Host load/prep remains the dominant wall-time because expert stack prep is still Python/NumPy-side; progress now makes that visible.
- Next performance path: persistent process-level layer state reuse across tokens, real KV/recurrent state advancement, GPU sampling, then graph/captured replay.

---

## 2026-05-14 — Add actual autoregressive Qwen3.5/PARO timing harness

### Scope

- Added `Qwen35ParoResidentSession` for actual multi-token autoregressive inference with:
  - resident all-layer weights,
  - per-linear-layer conv/recurrent state retained across tokens,
  - per-full-attention-layer BF16 paged KV cache retained across tokens,
  - device embedding table, final norm, W8A16 lm-head, and GPU argmax.
- Added `scripts/qwen35_paro_bench.py` to time resident load, actual token-by-token prompt prefill, warmup decode, and measured decode.
- Changed resident sampling to preload HIP libraries and pass them through wrapper calls, avoiding per-kernel `hipcc --version` / `ctypes.CDLL` overhead.
- Added exported GPU `argmax_f32` so W8A16 lm-head logits can reuse the existing two-stage GPU argmax.

### Validation

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro_runner.py hipengine/runtime/__init__.py scripts/qwen35_paro_bench.py
python3 -m pytest tests/test_lm_head_plan.py tests/test_llm_generate.py tests/test_qwen35_decode_state.py -q
python3 -m compileall -q hipengine scripts tests
python3 -m pytest -q
python3 scripts/qwen35_paro_bench.py --max-layers 1 --prompt-length 2 --decode-tokens 2 --warmup-decode-tokens 1 --token-id 9707
python3 scripts/qwen35_paro_bench.py --prompt-length 1 --decode-tokens 1 --warmup-decode-tokens 0 --token-id 9707 --json /tmp/hipengine-qwen35-bench-smoke2.json
```

Results on W7900 / real checkpoint:

- Full pytest suite passed.
- One-layer actual prompt/decode smoke completed with persistent state and W8A16 GPU lm-head:
  - load `14.25s`, token-by-token prefill `6.44 tok/s` for 2 prompt tokens, warmed decode `3.42 tok/s` for 2 measured tokens.
- All-layer actual prompt/decode smoke completed with persistent state and W8A16 GPU lm-head:
  - shape `prompt_length=1`, `decode_tokens=1`, load `35.08s`, token-by-token prefill `3.04 tok/s`, decode `3.20 tok/s`.
  - Generated preview starts with token `76323` (`"arra"`), matching the earlier all-layer one-token E2E path.

### Interpretation / Next

- This is now actual autoregressive inference, but prefill is still token-by-token c=1, not native batched/compact prefill; do not compare it to PLAN-MOE2 prefill tok/s.
- Warmed decode is a real c=1 resident decode measurement, but still lacks graph replay and is far below PLAN-MOE2 decode (~3.2 tok/s vs ~131 tok/s at 512/128 target class).
- Immediate next bottleneck work: capture a kernel/runtime trace for one measured decode step to separate Python/ctypes launch overhead from GPU kernel time, then add graph/library replay or lower-overhead dispatch before running large 512/128 measurements.

---

## 2026-05-14 — Diagnostic 512/128 actual c=1 benchmark vs PLAN-MOE2

### Command

```bash
python3 scripts/qwen35_paro_bench.py --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 --json /tmp/hipengine-qwen35-paro-512-128-diagnostic.json
```

### Result

- HIPENGINE actual autoregressive c=1 resident path completed on W7900.
- Shape: 512 prompt tokens, 4 warmup decode tokens, 128 measured decode tokens, repeated token id `9707`.
- Load/materialization: `35.35s`.
- Token-by-token prefill: `5.54s`, `92.39 tok/s` (actual inference, but not native batched/compact prefill).
- Warmed decode: `40.68s`, `3.146 tok/s`; median step `0.3161s`.
- Generated preview repeats token `62843` (`"estring"`).

### Comparison to PLAN-MOE2 2026-05-12 512/128 row

- PLAN-MOE2 parent baseline: prefill `1300.337 tok/s`, decode `131.128 tok/s`.
- HIPENGINE prefill ratio: `0.071x` of parent, **not comparable** because native prefill is not implemented.
- HIPENGINE warmed decode ratio: `0.024x` of parent, partially comparable but no graph replay/lower-overhead dispatch yet.

Artifact: `benchmarks/results/2026-05-14-hipengine-qwen35-paro-512-128-c1-diagnostic.json` (`status=blocked`, diagnostic/non-retained).

### Next

- Profile one measured decode step with prebuilt/cached libraries to split GPU kernel time from Python/ctypes launch overhead.
- Then attack dispatch/graph replay before treating larger warmed decode numbers as meaningful acceptance candidates.

---

## 2026-05-14 — Tokenizer cache removes false decode bottleneck

### Finding

Profiling showed the post-sampling host path was reopening `tokenizer.json` for every generated token. In the 1-layer cProfile smoke, `Qwen35ParoResidentSession.step()` took ~0.308s/token and `_decode_token()` accounted for ~0.277s/token. The one-layer rocprof smoke showed the actual GPU kernel work was millisecond-scale, so this was not a kernel or graph-replay issue.

### Change

- Cache the tokenizer once in `Qwen35ParoResidentSession` and decode generated-token text through the cached tokenizer.
- Add optional `--roctx` ranges to `scripts/qwen35_paro_bench.py` for future profiler correlation.

### Validation

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_paro_bench.py
python3 -m pytest tests/test_lm_head_plan.py tests/test_llm_generate.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_paro_bench.py --max-layers 1 --prompt-length 2 --decode-tokens 4 --warmup-decode-tokens 1 --token-id 9707 --json /tmp/hipengine-tokenizer-cache-smoke.json
python3 scripts/qwen35_paro_bench.py --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 --json /tmp/hipengine-qwen35-paro-512-128-tokenizer-cache.json
```

### Diagnostic result

- 512/128 actual c=1 resident run improved warmed decode from `3.146 tok/s` to `87.821 tok/s`.
- Median decode step improved from `0.3161s` to `0.01138s`.
- Token-by-token prefill stayed in the same class: `97.03 tok/s`; still not native batched/compact prefill.
- Current PLAN-MOE2 compact-WMMA target is `115.666 tok/s` decode at 512/128; HIPENGINE is now ~`75.9%` of that decode target, but remains blocked for accepted parity because graph replay and E2E correctness gates are not landed.

Artifact: `benchmarks/results/2026-05-14-hipengine-qwen35-paro-512-128-tokenizer-cache-diagnostic.json` (`status=blocked`, diagnostic/non-retained).

---

## 2026-05-14 — Add graph-friendly device token/position state kernels

### Scope

- Added `hipengine/kernels/hip_gfx1100/runtime/state.{hip,py}` with small raw-pointer kernels for graph-friendly decode state:
  - BF16 embedding row lookup from a device int64 token id,
  - device int64 scalar set,
  - device decode position/context set,
  - device decode position/context advance.
- Wired `Qwen35ParoResidentSession` eager path to use device token embedding and device position/context update instead of host-dependent D2D offset copies and H2D position copies. This is a prerequisite for one-step HIP graph replay because the replayed step can now keep token id and position state on device.
- Added dry-run/registry tests and updated kernel catalog / implementation checklist.

### Validation

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/runtime/state.py hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_paro_bench.py tests/test_runtime_state_plan.py
python3 -m pytest tests/test_runtime_state_plan.py tests/test_lm_head_plan.py tests/test_llm_generate.py tests/test_qwen35_decode_state.py -q
python3 - <<'PY'  # runtime_state GPU smoke; see session log for full script
# embedding token id 2 -> [8,9,10,11], set position 7 then advance -> position 8/context 9
PY
python3 scripts/qwen35_paro_bench.py --max-layers 1 --prompt-length 2 --decode-tokens 4 --warmup-decode-tokens 1 --token-id 9707 --json /tmp/hipengine-runtime-state-smoke.json
```

Results:

- Runtime helper GPU smoke returned `{'embedding': [8, 9, 10, 11], 'position': 8, 'context': 9}`.
- One-layer Qwen3.5/PARO resident smoke completed with unchanged generated token sequence (`229838`, `"وو"`) and median measured decode step `0.00125s`.

### Next

- Implement a non-default stream + HIP graph replay wrapper using these device-resident state updates.

---

## 2026-05-14 — Diagnostic one-step HIP graph replay for Qwen3.5/PARO decode

### Scope

- Added HIP stream/graph ctypes wrappers to `HipRuntime`.
- Propagated stream arguments through `Qwen35ParoDecodeState` wrapper/orchestrator calls.
- Added `Qwen35ParoResidentSession.capture_decode_graph(position=...)`, which captures one generated-token decode step on a non-default stream. The captured step consumes the current device argmax token, runs all resident layers, runs GPU W8A16 lm-head + argmax, writes the next token on device, and advances device position/context.
- Added `scripts/qwen35_paro_bench.py --graph-replay-decode` for measured decode graph replay.

### Validation

```bash
python3 -m compileall -q hipengine scripts tests
python3 -m pytest -q
python3 scripts/qwen35_paro_bench.py --max-layers 1 --prompt-length 2 --decode-tokens 1 --warmup-decode-tokens 1 --token-id 9707 --graph-replay-decode --json /tmp/hipengine-graph-1.json
python3 scripts/qwen35_paro_bench.py --max-layers 1 --prompt-length 2 --decode-tokens 4 --warmup-decode-tokens 1 --token-id 9707 --graph-replay-decode --json /tmp/hipengine-graph-4.json
python3 scripts/qwen35_paro_bench.py --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 --graph-replay-decode --json /tmp/hipengine-qwen35-paro-512-128-graph.json
```

1-layer graph-vs-eager sanity:

- `decode_tokens=1`: graph final token/logit matched eager (`229838`, `6.366115570068359`).
- `decode_tokens=4`: graph final token/logit matched eager (`229838`, `6.246890544891357`).

512/128 W7900 diagnostic:

- Load/materialization: `29.70s`.
- Token-by-token prefill: `5.25s`, `97.46 tok/s` (actual c=1, not native prefill).
- Graph replay measured decode: `1.381s`, `92.676 tok/s`, average step `10.79ms`.
- Current PLAN-MOE2 compact-WMMA 512/128 decode target: `115.666 tok/s`; HIPENGINE graph diagnostic is ~`80.1%` of target.

Artifact: `benchmarks/results/2026-05-14-hipengine-qwen35-paro-512-128-graph-diagnostic.json` (`status=blocked`, diagnostic/non-retained).

### Next

- Profile graph replay kernel mix. The remaining ~20% decode gap is no longer explained by tokenizer loading or simple Python per-kernel launch overhead.
- Native batched/compact prefill remains the largest missing implementation for PLAN-MOE2 prefill parity.

---

## 2026-05-14 — Fuse linear-attention QKV/Z pack8 decode projection

### Scope

- Prepared transposed generic pack8 qweights for linear-attention `in_proj_qkv` and `in_proj_z` during runtime layer materialization.
- Added contiguous `qkv_z` scratch with views for the existing `qkv` and `z` consumers.
- Switched `project_linear_attention_qkv_z_bf16` from two separate generic pack8 GEMVs to the existing dual-input transposed pack8 GEMV wrapper.

This ports the parent `NANOVLLM_PARO_LINEAR_ATTN_QKV_Z_PACK8_FUSED=1` decode route for the HIPENGINE c=1 path.

### Validation

```bash
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py -q
python3 scripts/qwen35_paro_bench.py --max-layers 1 --prompt-length 2 --decode-tokens 4 --warmup-decode-tokens 1 --token-id 9707 --json /tmp/hipengine-dual-linear-eager-smoke.json
python3 scripts/qwen35_paro_bench.py --max-layers 1 --prompt-length 2 --decode-tokens 4 --warmup-decode-tokens 1 --token-id 9707 --graph-replay-decode --json /tmp/hipengine-dual-linear-graph-smoke.json
python3 scripts/qwen35_paro_bench.py --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 --graph-replay-decode --json /tmp/hipengine-qwen35-paro-512-128-linear-fused-graph.json
```

Results:

- 1-layer eager and graph smokes preserved the prior generated token/logit sequence (`229838`, `"وو"`; graph final logit `6.246890544891357`).
- 512/128 graph diagnostic improved decode from `92.676 tok/s` to `104.066 tok/s` (`+12.3%`).
- Current PLAN-MOE2 compact-WMMA 512/128 decode target is `115.666 tok/s`; HIPENGINE is now ~`90.0%` of that decode target.
- Token-by-token c=1 prefill measured `109.4 tok/s`; still not native batched/compact prefill and not comparable to PLAN-MOE2 prefill.

Artifact: `benchmarks/results/2026-05-14-hipengine-qwen35-paro-512-128-linear-qkv-z-fused-graph-diagnostic.json` (`status=blocked`, diagnostic/non-retained).

### Next

- Remaining decode gap: profile after QKV/Z fusion. Likely buckets are generic W4 pack8 projections, W8A16 lm-head, selected MoE pack8, and attention GQA.
- Larger missing implementation remains native batched/compact prefill.

---

## 2026-05-14 — Fuse full-attention Q/K pack8 decode projection

### Scope

- Prepared transposed generic pack8 qweights for full-attention `q_proj` and `k_proj` runtime layers.
- Added contiguous `q_proj_key` scratch with existing `q_proj`/`key_bf16` views.
- Switched full-attention Q/K projection to the dual-input transposed pack8 GEMV, keeping V and O projections on the existing generic pack8 path.
- Added an optional benchmark `--graph-steps-per-replay` knob. A 4-step replay smoke matched eager, but it did not materially improve 512/128, so retained benchmark command remains one-step replay.

This ports the parent `NANOVLLM_PARO_FULL_ATTN_QK_PACK8_FUSED=1` decode route for HIPENGINE c=1 full-attention layers.

### Validation

```bash
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py -q
python3 scripts/qwen35_paro_bench.py --max-layers 4 --prompt-length 4 --decode-tokens 4 --warmup-decode-tokens 1 --token-id 9707 --graph-replay-decode --json /tmp/hipengine-full-qk-fused-smoke.json
python3 scripts/qwen35_paro_bench.py --max-layers 4 --prompt-length 4 --decode-tokens 4 --warmup-decode-tokens 1 --token-id 9707 --json /tmp/hipengine-full-qk-fused-eager-smoke.json
python3 scripts/qwen35_paro_bench.py --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 --graph-replay-decode --json /tmp/hipengine-qwen35-paro-512-128-full-qk-linear-fused-graph.json
```

Results:

- 4-layer graph final token/logit matched eager after Q/K fusion (`135534`, `"为重"`, final logit `7.168249607086182`).
- 512/128 graph diagnostic improved decode from `104.066 tok/s` to `108.503 tok/s` (`+4.3%`).
- Current PLAN-MOE2 compact-WMMA 512/128 decode target is `115.666 tok/s`; HIPENGINE is now ~`93.8%` of that decode target.
- Token-by-token c=1 prefill measured `114.39 tok/s`; still not native batched/compact prefill.

Artifact: `benchmarks/results/2026-05-14-hipengine-qwen35-paro-512-128-linear-qkv-z-full-qk-fused-graph-diagnostic.json` (`status=blocked`, diagnostic/non-retained).

### Next

- Remaining 512/128 decode gap is ~6.2%. Profile buckets after both projection fusions are W8A16 lm-head, selected MoE pack8, full-attention GQA, and remaining generic pack8.
- Native batched/compact prefill remains unimplemented.

---

## 2026-05-14 — Tune Qwen3.5/PARO graph decode lm-head threads

### Scope

- Made resident-session lm-head thread count configurable through `HIPENGINE_QWEN35_LM_HEAD_THREADS`.
- Changed default from 256 to 128 threads for W8A16 lm-head + argmax staging, matching the faster W7900 diagnostic setting.

### Validation

```bash
HIPENGINE_QWEN35_LM_HEAD_THREADS=128 python3 scripts/qwen35_paro_bench.py --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 --graph-replay-decode --json /tmp/hipengine-qwen35-paro-512-128-lmhead128.json
HIPENGINE_QWEN35_LM_HEAD_THREADS=512 python3 scripts/qwen35_paro_bench.py --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 --graph-replay-decode --json /tmp/hipengine-qwen35-paro-512-128-lmhead512.json
python3 scripts/qwen35_paro_bench.py --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 --graph-replay-decode --json /tmp/hipengine-qwen35-paro-512-128-lmhead128-default.json
```

Results:

- 128 threads: `110.03 tok/s` with explicit env; `109.99 tok/s` as default.
- 256-thread prior after Q/K fusion: `108.50 tok/s`.
- 512 threads regressed to `98.95 tok/s`.
- Current PLAN-MOE2 compact-WMMA 512/128 decode target is `115.666 tok/s`; HIPENGINE is now ~`95.1%` of that decode target.

Artifact: `benchmarks/results/2026-05-14-hipengine-qwen35-paro-512-128-lmhead128-qk-qkvz-fused-graph-diagnostic.json` (`status=blocked`, diagnostic/non-retained).

---

## 2026-05-14 — Fuse linear-attention A/B dense decode projection

### Scope

- Added `dense_dual_gemv_out_bf16` raw-pointer kernel/wrapper for two small BF16 dense GEMVs with shared input and contiguous output.
- Added contiguous linear-attention `ab` scratch with existing `a`/`b` views.
- Switched `project_linear_attention_ab_bf16` from two dense GEMV launches to the dual dense GEMV.

This ports the parent `NANOVLLM_PARO_LINEAR_ATTN_AB_FUSED=1` decode route for HIPENGINE c=1 linear-attention layers.

### Validation

```bash
python3 -m compileall -q hipengine scripts tests
python3 -m pytest -q
python3 scripts/qwen35_paro_bench.py --max-layers 1 --prompt-length 2 --decode-tokens 4 --warmup-decode-tokens 1 --token-id 9707 --graph-replay-decode --json /tmp/hipengine-ab-fused-smoke.json
python3 scripts/qwen35_paro_bench.py --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 --graph-replay-decode --json /tmp/hipengine-qwen35-paro-512-128-ab-fused.json
```

Results:

- 1-layer graph smoke preserved final token/logit (`229838`, `"وو"`, `6.246890544891357`).
- 512/128 graph diagnostic improved decode from `109.99 tok/s` to `111.104 tok/s` (`+1.0%`).
- Current PLAN-MOE2 compact-WMMA 512/128 decode target is `115.666 tok/s`; HIPENGINE is now ~`96.1%` of that decode target.
- Token-by-token c=1 prefill measured `115.81 tok/s`; still not native batched/compact prefill.

Artifact: `benchmarks/results/2026-05-14-hipengine-qwen35-paro-512-128-ab-fused-lmhead128-graph-diagnostic.json` (`status=blocked`, diagnostic/non-retained).

---

## 2026-05-14 — Port native linear-attention prefill conv/GDN kernels

### Scope

- Began native batched/compact prefill implementation by porting the parent Qwen3.5 linear-attention prefill state kernels from `nano-vllm-amd` current drifted lineage:
  - `qwen35_linear_attn_conv_prefill_kernel`
  - `qwen35_linear_attn_conv_prefill_state_kernel`
  - `qwen35_gdn_prefill_recurrent_kernel`
  - `qwen35_gdn_prefill_recurrent_k2_kernel`
- Added raw-pointer wrappers/registry keys:
  - `linear_attn_conv_prefill/w4_paro/f32`
  - `gdn_prefill_recurrent/w4_paro/f32`
  - `gdn_prefill_recurrent/w4_paro/f32_k2`
- Added `scripts/smoke.py --mode qwen35-linear-attn-prefill-hip` with NumPy oracles for native conv-prefill state update and GDN recurrent prefill regular/K2 kernels.
- This is a first native prefill slice. The complete PLAN-MOE2 prefill path still needs Q/K normalization, beta/decay production, RMSNorm+gate, batched output projection, batched/compact MoE, and full resident-session prefill orchestration.

### Lineage

```bash
python3 scripts/check_lineage.py --kind kernel --diff stat
```

Observed drift before porting:

- `csrc/amd/qwen35_expert.hip` drifted through `6e2b19b` (`perf: compact WMMA buffers eliminate 44.5% padding overhead at 512 prefill`).
- `nanovllm/native/qwen35/paroquant_kernels.py` drifted through `59195ed` plus compact-WMMA/GEMV fixes.

### Validation

```bash
python3 -m pytest tests/test_qwen35_linear_attn_conv_plan.py tests/test_qwen35_linear_attn_gdn_plan.py -q
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode qwen35-linear-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-linear-prefill-prof --output-file linear-prefill --output-format csv -- \
  python3 scripts/smoke.py --mode qwen35-linear-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Smoke result:

- `conv_out_max_abs=1.49e-08`, `conv_state_max_abs=0`
- `gdn_out_max_abs=5.59e-09`, `gdn_state_max_abs=7.45e-09`
- `gdn_k2_out_max_abs=3.73e-09`, `gdn_k2_state_max_abs=7.45e-09`

Profiler confirmed expected kernels launched:

- `qwen35_linear_attn_conv_prefill_kernel`
- `qwen35_linear_attn_conv_prefill_state_kernel`
- `qwen35_gdn_prefill_recurrent_kernel`
- `qwen35_gdn_prefill_recurrent_k2_kernel`

### Next

- Wire a native batched linear-attention prefill state path in `Qwen35ParoDecodeState`: Q/K L2 normalization, beta/decay, GDN RMSNorm+gate, output projection, and state update.
- Then move to the dominant missing piece: grouped/compact MoE prefill.
