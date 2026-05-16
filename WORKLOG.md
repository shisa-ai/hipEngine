# hipENGINE Work Log

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

- **Kernel R&D lives in `~/amd-gpu-tuning/`, not here.** Micro-tuning iteration loops (rocprofv3 time-share audit, VGPR / occupancy hunting, `__launch_bounds__` sweeps, fusion experiments, device-code gotcha catalog) all stay in the parent workspace. hipENGINE ingests *stable* kernels via the port pipeline in `docs/PLAN.md` "Kernel Port Strategy".
- Consequence: hipENGINE's `docs/KERNELS.md` is a port playbook (copy + partition + retype + gate), not a kernel-tuning guide. Tuning guide stays at `~/amd-gpu-tuning/AGENTS.md` and `~/amd-gpu-tuning/LESSONS-LEARNED.md`.
- AGENTS.md "Handling Blockers" redirects kernel-micro-opt and ROCm-restore situations to `~/amd-gpu-tuning/` rather than duplicating the procedures here.

### Doc inventory from `~/amd-gpu-tuning/`

Surveyed 12 `.md` files in `~/amd-gpu-tuning/docs/` plus the top-level design docs. Copied or referenced as follows:

| Upstream doc | Action | Rationale |
| --- | --- | --- |
| `docs/ROOFLINE.md` (1573 lines) | **Copied** to `docs/ROOFLINE.md` | Canonical RDNA3 / W7900 hardware landscape: hardware, roofline fundamentals, regimes, decision tree, what-not-to-chase. Read by anyone planning hipENGINE kernels or setting perf targets. Added provenance header; path-qualified companion-doc cross-refs to `~/amd-gpu-tuning/`. |
| Parent design doc (1214 lines) | Already here as `docs/PLAN.md` | Same content; don't duplicate. |
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

- Commits: `f2a5166` docs: add hipENGINE design plan; `f33b2a8` docs: add AGENTS.md ground rules, CLAUDE.md symlink, .gitignore.
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

## 2026-05-12 — License hipENGINE as AGPL-3.0-or-later

### Decision

- Selected **AGPL-3.0-or-later** for hipENGINE source code.
- Rationale: project is aimed at local/home users, and we explicitly prefer copyleft over permissive/business adoption. AGPL closes the hosted-service loophole that GPLv3 leaves open for an inference engine with optional server/API paths.
- User clarified that the future `nano-vllm-amd` kernel ports are not an upstream-license concern for this decision because those kernels were authored locally by the project lead; still, model weights/checkpoints and external datasets remain under their own licenses.

### Files changed

- Added `LICENSE` containing the full GNU Affero General Public License v3 text from the system SPDX license copy (`/usr/share/licenses/spdx/AGPL-3.0-or-later.txt`).
- Updated `pyproject.toml` project metadata from `Apache-2.0` to `AGPL-3.0-or-later`.
- Updated `README.md` with a License section: hipENGINE source code is AGPL-3.0-or-later; model weights, checkpoints, and external datasets keep their own licenses.
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
- That process is not owned by this hipENGINE task. Pausing further GPU actions here; do not run rmsnorm port or more profiling until the GPU is explicitly clear again.

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

- User noted hipENGINE is becoming "real" software and should have a proper testing story: RED/GREEN, correctness guard/gates, and especially protection against silent math mistakes.
- Goal: adopt useful testing methodology/verbiage from `~/shisad/` and `~/shisad-dev/` without importing irrelevant process (multi-reviewer lanes, release machinery, implement-driven workflow).

### Sources reviewed

- `~/shisad/AGENTS.md`:
  - Useful: Spec → Plan → Test → Implement; write tests first even for ad-hoc work; run targeted tests first; exact command evidence; claim integrity; structural tests are not enough for runtime-facing behavior.
  - Not adopted: shisad-specific security roles, multi-reviewer process, live daemon harness details.
- `~/shisad-dev/AGENTS.md`:
  - Useful: validation cadence proportional to scope; do not default to broad suites for every small change; record validation evidence in worklog; truth-scoped claims.
  - Not adopted: private/public repo split, reviewer-lane rules, release-close process.
- `~/shisad-dev/implement/TEST-COVERAGE.md`:
  - Most relevant source. Key adapted concept: structural correctness is necessary but not sufficient. For shisad the real contract is user-visible correctness; for hipENGINE the real contract is numerical correctness against an oracle.
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
- `hipengine/kernels/cpu_reference/ops.py` and `hipengine/kernels/hip_gfx1100/smoke/smoke_add.py` to list kernels/oracles actually landed in hipENGINE.
- `~/amd-gpu-tuning/PLAN-PAROQUANT.md` and `~/amd-gpu-tuning/docs/PARO.md` for the current Qwen3.5-35B-A3B-PARO route, shape-gated prefill MoE split, graph replay caveats, 24GB compact path, and recent rejected/alternative routes.
- `~/amd-gpu-tuning/nano-vllm-amd` source inventory:
  - Committed stable Qwen/PARO set: 95 kernels in `csrc/amd/qwen35_expert.hip` + 25 kernels in `nanovllm/native/qwen35/paroquant_kernels.py` = 120 Qwen/PARO kernels, plus separate `smoke_add`.
  - Parent repo observed at `nano-vllm-amd@22405a9` with local modifications in `paroquant.py` and `paroquant_kernels.py`; six additional PARO kernels were documented as lineage-dirty/experimental, not hipENGINE defaults.

### Files changed

- `docs/BENCHMARK.md`:
  - Added a benchmark-output contract: exact run context, correctness status/commands, repeated-run statistics, profiler/kernel summary, baseline comparison, and acceptance/rejection reason.
  - Added artifact statuses: `accepted`, `rejected_correctness`, `rejected_variance`, `blocked`.
  - Expanded microbenchmark and E2E measurement statistics requirements: samples, median/p95/min/max/stdev, warmup/measured counts, variance guard.
  - Upgraded retained benchmark JSON schema from `1` to `2` with `status`, command groups, correctness pass/fail fields, measurement samples, memory, profiler top kernels, baseline/comparison, and decision fields.
  - Clarified blocked/rejected attempts are still useful evidence but not retained performance numbers.
- `docs/KERNELS.md`:
  - Renamed to a kernel catalog + port playbook.
  - Added status legend distinguishing hipENGINE-landed, CPU-reference-landed, lineage-green, lineage-dirty/experimental, and planned.
  - Added authoritative hipENGINE-landed list: CPU-reference oracles (`embed`, `rmsnorm`, `linear`, `qkv_proj`, `rotate`, `attention_decode`, `o_proj`, `lm_head`) and `smoke_add` gfx1100 build/runtime smoke.
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

- User asked for a way to track whether kernel or dispatch files in `~/amd-gpu-tuning/` are newer before continuing hipENGINE ports.
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
- Do not advance `docs/source_lineage.json` baseline until hipENGINE's catalog/port plan is intentionally refreshed and logged.

---

## 2026-05-13 — Wire OPTIMAL.md into kernel path and hygiene docs

### Prompt / concern

- User noted `~/amd-gpu-tuning/docs/OPTIMAL.md` should be up to date with the optimal PARO inference path and should likely be referenced from hipENGINE's kernel catalog.
- User also asked to review `~/amd-gpu-tuning/AGENTS.md` for git/benchmark hygiene worth adopting in hipENGINE.
- Follow-up explicit rule requested: before porting, check `docs/KERNELS.md` and use the lineage script to ensure the kernel catalog/path map is up to date.

### Sources reviewed

- `~/amd-gpu-tuning/docs/OPTIMAL.md`:
  - Current optimal path: compact-WMMA prefill + one-step graph-replay decode for Qwen3.5-35B-A3B-PARO.
  - Latest retained sweep: 512/128 `2557 / 115.7`, 1K/128 `2876 / 112.9`, 4K/128 `2703 / 112.0`, 32K/128 `1880 / 98.8`, 128K/128 `914 / 62.6` prefill/decode tok/s, graph/step validation true.
  - 23 base flags, long-prefill chunking overrides, graph replay caveats, and decode profiling note that AWQ/GEMV decode is the next target.
- `~/amd-gpu-tuning/AGENTS.md`:
  - Already covered by hipENGINE: explicit staging rules, no destructive cleanup, WORKLOG with logical unit, audit-first kernel tuning, raw artifact exclusion.
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
  - Current fastest hipENGINE table (empty until first accepted E2E `LLM.generate()` benchmark).
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

- Reviewed `https://github.com/AICL-Lab/hetero-paged-infer` at commit `a9765bd69aefd8a64591d930867d21ed3dd7fd90` as a potential reference for hipENGINE's scheduler / paged-KV / tiered-memory design.
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
- Its KV abstraction is classic uniform fixed-page `block_table + context_len`. This is useful as a small scheduler/block-manager sanity reference, but it is less general than hipENGINE's planned `KVLiveSpans` ABI and `KVPolicy.admission_cap()` contract for DMS / H2O / SnapKV / sliding policies.
- No architecture change adopted. If we need a future sanity check for host-only scheduler invariants, its property tests and simple `BlockPool`/`PageTable` model are a reasonable reference. For tiered/offloaded decode scheduling, APEX and Neo are more relevant research references than this repo.

### Next

- Do not port code from this repo into hipENGINE.
- Optional future doc update: add it to `docs/PLAN.md` references only as a lightweight Rust host-shape / test-harness reference, not as a kernel or tiered offload source.

---

## 2026-05-13 — Port Qwen3.5 BF16 RMSNorm HIP family

### Scope

- Ported the first real model-layer gfx1100 kernel family into hipENGINE: Qwen3.5 BF16 RMSNorm from `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip`.
- Source commit: `nano-vllm-amd@59195ed` (`gfx1100-qwen3.5`). The lineage checker reports drift vs baseline `22405a9`, but `git diff 22405a9..HEAD -- csrc/amd/qwen35_expert.hip` shows the RMSNorm region is not touched by the current compact-WMMA drift.

### Files changed

- Added `hipengine/kernels/hip_gfx1100/norm/rmsnorm.hip`:
  - Preserved Qwen kernel bodies for `qwen35_rmsnorm_kernel`, `qwen35_add_rmsnorm_kernel`, `qwen35_add_rmsnorm_f32_kernel`, and `qwen35_head_rmsnorm_kernel`.
  - Added hipENGINE C ABI launch wrappers taking raw pointers, shapes, `eps`, and `hipStream_t`.
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

- User suggested using the current `~/amd-gpu-tuning/docs/OPTIMAL.md` MoE path as the next port target so hipENGINE can exercise the full `docs/KERNELS.md` checklist, correctness gates, and benchmark robustness against the parent performance rows.

### Source review

- Re-read `docs/KERNELS.md`, `docs/PLAN.md` kernel port strategy, latest WORKLOG entries, and `~/amd-gpu-tuning/docs/OPTIMAL.md`.
- Ran lineage check:

```bash
python3 scripts/check_lineage.py --diff stat --evidence-limit 4
```

Current parent checkout:

- `nano-vllm-amd` branch `gfx1100-qwen3.5`, HEAD `59195ed`.
- Drift vs hipENGINE baseline `22405a9` in:
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
  - Explicitly marked current hipENGINE status: only Qwen BF16 RMSNorm subset is partial/landed; PARO RMSNorm out-kernels, router, selected GEMV, fused activation/down-rotation, W8A16 shared/lm-head, compact WMMA, attention/KV, model/plugin/loader, and eval harness remain missing.
- Updated `docs/IMPLEMENTATION.md`:
  - Added an OPTIMAL MoE/PARO reproduction exercise punchlist keyed to `docs/KERNELS.md`.

### Key conclusion

- We should not start by copying a random MoE kernel. The fastest path to a meaningful exercise is:
  1. add parent-baseline + hipENGINE-blocked benchmark artifacts for 512/128 and 4K/128,
  2. port the MoE c=1 decode vertical slice,
  3. port the compact-WMMA prefill slice,
  4. only then close full inference with loader/model/attention/graph replay.
- Full OPTIMAL inference cannot be replicated yet because hipENGINE still lacks `LLM.generate()`, `w4_paro` weight loading/layout, the Qwen3.5 model plugin, attention/KV/linear-attn/lm-head dependencies, and graph replay.

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

## 2026-05-13 — Capture OPTIMAL parent parity artifacts and blocked hipENGINE row

### Scope

- Ran the parent `nano-vllm-amd` OPTIMAL Qwen3.5-35B-A3B-PARO command for `512/128` and `4K/128` on W7900 to validate the benchmark output shape and create concrete comparison artifacts before porting more kernels.
- Created a blocked hipENGINE artifact for the same parity exercise so the missing dependencies are tracked in `benchmarks/results/`, not just prose.

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
| hipENGINE | OPTIMAL parity | — | — | — | not reached | blocked | `benchmarks/results/2026-05-13-hipengine-qwen35-paro-optimal-blocked.json` |

Blocked hipENGINE reason: `LLM.generate`, `w4_paro` loader/layout, Qwen3.5 model plugin, MoE/attention/linear/lm-head dependency kernels, and graph replay are not landed yet.

### Files changed

- Added three compact benchmark artifacts under `benchmarks/results/`.
- Updated `benchmarks/README.md` source-lineage rows for 512/128 and 4K/128 to point at artifacts and use the local rerun values.
- Updated `benchmarks/CHANGELOG.md` with lineage-measured deltas and the blocked hipENGINE row.
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
- Added hipENGINE raw-pointer C ABI wrappers in the existing `norm/rmsnorm.hip` family:
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

- This first hipENGINE router wrapper supports BF16 hidden and BF16 combined weights. The parent accepts FP16 or BF16 hidden inputs; if the final hipENGINE OPTIMAL route keeps FP16 router inputs, add an FP16 hidden specialization before claiming full router parity.
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
- Next step is a composite hipENGINE shared-expert smoke chaining W8A16 gate/up → `silu_mul_dual_out` → W8A16 down, then a c=1 MoE vertical smoke that includes selected W4 experts and shared branch combine.

---

## 2026-05-13 — Add W8A16 shared-expert composite smoke

### Scope

- Added `scripts/smoke.py --mode w8a16-shared-expert-hip` to chain the current parent shared-expert lowp route with existing hipENGINE kernels:
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

- Port the committed parent attention/KV decode kernels instead of inventing a new ABI, then adapt wrappers to hipENGINE's `KVLiveSpans` ABI at the host boundary.

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

- Port KV append (`qwen35_write_paged_kv_mixed_value*`) and paged full-attention decode from the committed parent kernels, adapting wrappers to hipENGINE `KVLiveSpans` instead of changing kernel bodies.

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

- Resume KV append and paged full-attention decode with public wrappers shaped around hipENGINE `KVLiveSpans`, while preserving parent kernel bodies internally.

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
  - runs all Qwen3.5/PARO decode layers through hipENGINE linear-attention/full-attention c=1 layer chains,
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

- hipENGINE actual autoregressive c=1 resident path completed on W7900.
- Shape: 512 prompt tokens, 4 warmup decode tokens, 128 measured decode tokens, repeated token id `9707`.
- Load/materialization: `35.35s`.
- Token-by-token prefill: `5.54s`, `92.39 tok/s` (actual inference, but not native batched/compact prefill).
- Warmed decode: `40.68s`, `3.146 tok/s`; median step `0.3161s`.
- Generated preview repeats token `62843` (`"estring"`).

### Comparison to PLAN-MOE2 2026-05-12 512/128 row

- PLAN-MOE2 parent baseline: prefill `1300.337 tok/s`, decode `131.128 tok/s`.
- hipENGINE prefill ratio: `0.071x` of parent, **not comparable** because native prefill is not implemented.
- hipENGINE warmed decode ratio: `0.024x` of parent, partially comparable but no graph replay/lower-overhead dispatch yet.

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
- Current PLAN-MOE2 compact-WMMA target is `115.666 tok/s` decode at 512/128; hipENGINE is now ~`75.9%` of that decode target, but remains blocked for accepted parity because graph replay and E2E correctness gates are not landed.

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
- Current PLAN-MOE2 compact-WMMA 512/128 decode target: `115.666 tok/s`; hipENGINE graph diagnostic is ~`80.1%` of target.

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

This ports the parent `NANOVLLM_PARO_LINEAR_ATTN_QKV_Z_PACK8_FUSED=1` decode route for the hipENGINE c=1 path.

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
- Current PLAN-MOE2 compact-WMMA 512/128 decode target is `115.666 tok/s`; hipENGINE is now ~`90.0%` of that decode target.
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

This ports the parent `NANOVLLM_PARO_FULL_ATTN_QK_PACK8_FUSED=1` decode route for hipENGINE c=1 full-attention layers.

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
- Current PLAN-MOE2 compact-WMMA 512/128 decode target is `115.666 tok/s`; hipENGINE is now ~`93.8%` of that decode target.
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
- Current PLAN-MOE2 compact-WMMA 512/128 decode target is `115.666 tok/s`; hipENGINE is now ~`95.1%` of that decode target.

Artifact: `benchmarks/results/2026-05-14-hipengine-qwen35-paro-512-128-lmhead128-qk-qkvz-fused-graph-diagnostic.json` (`status=blocked`, diagnostic/non-retained).

---

## 2026-05-14 — Fuse linear-attention A/B dense decode projection

### Scope

- Added `dense_dual_gemv_out_bf16` raw-pointer kernel/wrapper for two small BF16 dense GEMVs with shared input and contiguous output.
- Added contiguous linear-attention `ab` scratch with existing `a`/`b` views.
- Switched `project_linear_attention_ab_bf16` from two dense GEMV launches to the dual dense GEMV.

This ports the parent `NANOVLLM_PARO_LINEAR_ATTN_AB_FUSED=1` decode route for hipENGINE c=1 linear-attention layers.

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
- Current PLAN-MOE2 compact-WMMA 512/128 decode target is `115.666 tok/s`; hipENGINE is now ~`96.1%` of that decode target.
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

---

## 2026-05-14 — Wire batched linear-attention prefill state path

### Scope

- Added GPU prefill prepare and finish kernels around the parent recurrent prefill body:
  - `qwen35_linear_attn_prefill_prepare_f32_bf16`: Q/K L2 normalization, Q scale, KV-head repeat, value split, BF16 A/B → beta/decay.
  - `qwen35_gdn_prefill_rmsnorm_gate_bf16`: per-value-head RMSNorm + SiLU gate → BF16 hidden output.
- Extended `Qwen35ParoLinearAttentionScratch` with native prefill scratch (`qkv_f32`, normalized Q/K/V, beta, decay).
- Added `Qwen35ParoDecodeState.run_linear_attention_prefill_state_bf16(...)` and `run_linear_attention_prefill_out_proj_bf16(...)` for a batched linear-attention prefill slice through PARO out-projection.
- This still does not make full E2E prefill native: the MoE block is still c=1/token-oriented, and the resident runner still needs a full native prefill orchestration path.

### Validation

```bash
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_linear_attn_gdn_plan.py -q
python3 scripts/smoke.py --mode qwen35-linear-attn-prefill-hip
```

Smoke result:

- `conv_out_max_abs=1.49e-08`, `conv_state_max_abs=0`
- `prepare_max_abs=5.96e-08`
- `gdn_out_max_abs=9.31e-10`, `gdn_state_max_abs=1.12e-08`
- `gdn_k2_out_max_abs=9.31e-10`, `gdn_k2_state_max_abs=1.12e-08`
- `gated_mismatch=0`

### Next

- Add grouped/compact MoE prefill kernels/state path; this is the dominant remaining prefill gap.
- Then wire resident-session native prefill and switch the benchmark from `native_batched_prefill=false` to a comparable native prefill row only after full-layer correctness gates are green.

---

## 2026-05-14 — Add batched c1-style MoE prefill support

### Scope

- Extended selected-dual PARO pack8 GEMV lane mapping so `rows = x_rows * lanes_per_token` reads the correct source token row for batched gate/up. This preserves c=1 decode behavior while allowing token-major selected lanes.
- Added `weighted_sum_shared_gate_combine_residual_batch_out_kernel` and wrapper/registry variant `weighted_sum+shared_gate+residual/*/batch_out` for batched selected weighted sum + shared gate + residual.
- Let `Qwen35ParoDecodeState.run_moe_c1_bf16(...)` run with `tokens > 1` using existing selected/down/shared kernels plus the new batched combine.
- Let `run_linear_attention_moe_c1_layer_bf16(...)` dispatch to the native linear-attention prefill path for `tokens > 1`, then the batched c1-style MoE path.

This is still a c1/GEMV-style batched prefill path, not the parent compact-WMMA/grouped MoE prefill. It is intended to unblock real multi-token layer orchestration before the compact MoE port.

### Validation

```bash
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_paro_combine_plan.py tests/test_paro_awq_gemv_plan.py -q
python3 scripts/smoke.py --mode paro-combine-hip --rows 4 --hidden-size 16
python3 scripts/smoke.py --mode paro-moe-c1-state-hip --hidden-size 8
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-combine-prof --output-file combine --output-format csv -- \
  python3 scripts/smoke.py --mode paro-combine-hip --rows 4 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Results:

- Combine smoke: `weighted_mismatch=0`, `fused_mismatch=0`, `batch_fused_mismatch=0`, `shared_mismatch=0`, `shared_residual_mismatch=0`.
- MoE c1 state smoke remained bit-exact (`final_mismatch=0`, `final_max_abs=0.0`).
- `rocprofv3` confirmed `weighted_sum_shared_gate_combine_residual_batch_out_kernel` launched.

### Next

- Wire resident-session layer prefill over linear-attention layers and measure the c1-style batched prefill baseline.
- Port/implement compact grouped MoE prefill (WMMA path) for the real PLAN-MOE2 throughput target.

---

## 2026-05-14 — Wire resident linear-prefix native prefill diagnostic

### Scope

- Added batched BF16 embedding lookup (`embedding_lookup_batch_bf16_i64`) for prompt-token slabs.
- Added `Qwen35ParoResidentSession.prefill_linear_tokens_native(...)` and `_run_linear_prefill_layers(...)` for native batched prefill over linear-attention-only layer prefixes.
- Added `scripts/qwen35_paro_bench.py --native-prefill`, guarded to the current linear-prefix diagnostic path. This path still refuses full-attention layers and is explicitly not the compact/grouped PLAN-MOE2 prefill path.

### Validation

```bash
python3 -m pytest tests/test_runtime_state_plan.py -q
python3 scripts/qwen35_paro_bench.py --max-layers 1 --prompt-length 4 --decode-tokens 1 --warmup-decode-tokens 0 --token-id 9707 --native-prefill --json /tmp/hipengine-native-prefill-linear1.json
python3 scripts/qwen35_paro_bench.py --max-layers 3 --prompt-length 16 --decode-tokens 1 --warmup-decode-tokens 0 --token-id 9707 --native-prefill --json /tmp/hipengine-native-prefill-linear3.json
```

Results:

- 1-layer native linear-prefix prefill smoke completed: `prefill_tok_s=364.18` for 4 tokens (diagnostic tiny shape).
- 3-layer native linear-prefix prefill smoke completed: `prefill_tok_s=1102.00` for 16 tokens (diagnostic tiny shape).
- Native linear-prefix output is not bit-equivalent to token-by-token c=1 on the sampled next token; this remains a diagnostic path pending fuller prefill correctness gates/parent parity. Do not promote it to benchmark rollups.

### Next

- Add an explicit correctness fixture comparing native batched linear-prefix prefill to the intended oracle (parent/NumPy staged math), not token-by-token decode where top-k can diverge.
- Port compact grouped MoE prefill and full-attention prefill before claiming PLAN-MOE2-comparable E2E prefill.

---

## 2026-05-14 — Document c>1 PARO roadmap in PLAN

### Scope

- Added `docs/PLAN.md` section `Concurrent Decode / c>1 PARO Roadmap`.
- Captured the code-review conclusion that hipENGINE is a better foundation for c>1 than `nano-vllm-amd`, but current Qwen3.5/PARO runtime remains effectively c=1.
- Documented current blockers: one-token smoke generator, scalar resident-session state, c1-only decode orchestrators, scalar-context GQA attention, and selected-MoE lane mapping.
- Added expected c=8 behavior and an implementation plan covering batch state, correctness harnesses, scheduler/graph buckets, batched attention, linear-attention state, MoE batch kernels, c-aware quant projection dispatch, and c=N benchmark protocol.

### Validation

- Re-read the inserted `docs/PLAN.md` section.
- Documentation-only change; no GPU run required.

### Next

- Treat Qwen3.5/PARO benchmark rows as c=1 until a c=N correctness harness and batch-state path land.
- Start any c=8 work with deterministic batched-vs-independent-c1 correctness fixtures before optimizing kernels.

---

## 2026-05-14 — Strengthen c>1 and SpecDec PLAN invariants

### Scope

- Reworked the c>1 PARO roadmap into day-1 invariants for batch-shaped runtime APIs, stable request IDs vs physical slots, continuous batching, transactional KV, draft/verify row metadata, graph shape buckets, and plugin-based c/specdec dispatch.
- Expanded `KVLiveSpans`/`KVPolicy` design notes so decode rows, prefill rows, and speculative verification rows share one attention/KV-write ABI.
- Added EAGLE3 to the SpecDec roadmap and documented `DraftModel`, `DraftBatch`, `Verifier`, and `AcceptResult` contracts.

### Validation

- Re-read `docs/PLAN.md` end-to-end.
- Ran a docs-only term check for the new batching/specdec invariants; no GPU run required.

### Next

- Review the live codebase against these invariants and create implementation tasks for batch-friendliness gaps before adding more c=1-only surfaces.

---

## 2026-05-14 — Add batch request/slot metadata scaffold

### Scope

- Added torch-free dispatch metadata for c>N work: `RequestState`, `ActiveBatch`, `BatchSlot`, `SlotMove`, `WorkKind`, `WorkItem`, and `BatchShapeKey`.
- `ActiveBatch` separates stable `request_id` from compactable physical slots, exposes active masks, slot/request maps, routed row maps, and graph shape keys keyed by mode/context/mask/top-k/replay/draft shape.
- Exported the batch metadata from `hipengine.dispatch` for future ResidentBatchSession, KVPolicy, scheduler, and SpecDec integration.

### Validation

```bash
python3 -m pytest tests/test_dispatch_batch.py -q
python3 -m compileall -q hipengine tests && python3 -m pytest tests/test_dispatch_batch.py tests/test_llm_generate.py -q
```

Results: all tests passed.

### Next

- Extend `KVLiveSpans`/`KVPolicy` with row/request metadata and transaction hooks, then vectorize runtime state kernels around these batch slots.

---

## 2026-05-14 — Extend KV spans and policy transaction scaffold

### Scope

- Extended `KVLiveSpans` with optional `request_ids`, `row_positions`, and `span_role` metadata while preserving the current c=1 fixed-page bridge.
- Added `KVPolicy` protocol plus `FixedPagedKVPolicy`, `KVReservation`, and `KVTransaction` host-side scaffolding.
- The fixed-page policy now exposes c=1 span reuse, c>1 packed-metadata span construction, admission-cap bookkeeping, transaction begin/commit/rollback, and reclaim.

### Validation

```bash
python3 -m pytest tests/test_kvcache_spans.py tests/test_kvcache_policy.py -q
python3 -m compileall -q hipengine tests && python3 -m pytest tests/test_kvcache_spans.py tests/test_kvcache_policy.py tests/test_qwen35_paged_kv_write_plan.py tests/test_qwen35_paged_attn_decode_plan.py -q
```

Results: all tests passed.

### Next

- Vectorize runtime token/position state kernels around the batch-slot metadata and keep scalar c=1 helpers as wrappers.

---

## 2026-05-14 — Vectorize runtime token and position state kernels

### Scope

- Added batch-slot runtime helpers: mapped batched embedding lookup, int64 vector set, masked batched decode-position set, and masked batched decode-position advance.
- Registered vector runtime helpers under existing kernel registry axes while preserving scalar c=1 wrappers.
- Updated kernel catalog/implementation notes for scalar + vector graph-state variants.

### Validation

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/runtime/state.py && python3 -m pytest tests/test_runtime_state_plan.py -q
python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.runtime import build_runtime_state
lib = build_runtime_state(load=True)
print('runtime_state built', getattr(lib, '_name', '<loaded>'))
PY
python3 -m compileall -q hipengine tests && python3 -m pytest tests/test_runtime_state_plan.py tests/test_dispatch_batch.py -q
```

GPU smoke result: `runtime_state batch smoke OK` for mapped embedding rows plus masked vector position/context set+advance.

### Next

- Refactor resident Qwen3.5/PARO runtime to allocate/use batch-shaped slots, then port batched full-attention KV append/decode.

---

## 2026-05-14 — Add resident batch-layout scaffold for Qwen3.5/PARO

### Scope

- Added `Qwen35ParoResidentBatchLayout` and `max_batch_size` plumbing to `Qwen35ParoResidentSession`.
- Resident hidden/norm/token/position/context buffers are now allocated as batch-slot-shaped storage with slot-0 c=1 tensor aliases for the current runtime path.
- Linear recurrent/conv state and full-attention KV cache allocations now reserve a leading batch dimension while preserving slot-0 aliases for existing kernels.
- Added internal helpers for batch token embedding and batched position/context setup using the vector runtime-state kernels.

### Validation

```bash
python3 -m compileall -q hipengine tests && python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_runtime_state_plan.py tests/test_dispatch_batch.py -q
python3 -m pytest tests/test_llm_generate.py tests/test_qwen35_resident_batch_layout.py -q
python3 scripts/qwen35_paro_bench.py --max-layers 1 --prompt-length 1 --decode-tokens 0 --warmup-decode-tokens 0 --token-id 9707 --json /tmp/hipengine-batch-layout-smoke.json
```

Results: unit tests passed; one-layer resident c=1 smoke completed and produced token id `62406` (`"ullo"`).

### Next

- The runtime is now batch-shaped in allocation but still executes slot 0 only. Next steps are c>1 full-attention KV append/decode kernels and a batched layer runner that consumes the batch slots.

---

## 2026-05-14 — Add c>1 paged KV append and context attention kernels

### Scope

- Added row-major c>1 paged KV append kernel/wrapper `qwen35_write_paged_kv_mixed_value_bf16_batch_spans(...)` using `KVLiveSpans` row metadata.
- Added row-major c>1 paged context attention kernel/wrapper `qwen35_paged_full_attn_decode_context_bf16_batch_spans(...)` for correctness bring-up over uneven context lengths.
- Registered both variants in the kernel registry and documented them in the kernel catalog/implementation checklist.

### Validation

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/attention/paged_kv_write.py hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.py
python3 -m pytest tests/test_qwen35_paged_kv_write_plan.py tests/test_qwen35_paged_attn_decode_plan.py -q
python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.attention import build_qwen35_paged_kv_write, build_qwen35_paged_attn_decode
kv = build_qwen35_paged_kv_write(load=True)
attn = build_qwen35_paged_attn_decode(load=True)
print('built', getattr(kv, '_name', '<kv>'), getattr(attn, '_name', '<attn>'))
PY
python3 -m compileall -q hipengine tests && python3 -m pytest tests/test_qwen35_paged_kv_write_plan.py tests/test_qwen35_paged_attn_decode_plan.py tests/test_kvcache_policy.py -q
```

GPU smoke result: `batched paged kv+attn smoke OK 4.76837158203125e-07` vs NumPy softmax oracle for c=2 with uneven context lengths.

### Next

- Wire the c>1 attention variants into a batched layer runner and add deterministic c>N-vs-independent-c1 correctness fixtures.

---

## 2026-05-14 — Add deterministic c>N vs c1 primitive correctness harness

### Scope

- Added `scripts/qwen35_batch_correctness.py` to compare c>N batched paged-KV append/context-attention against independent c=1 launches and a NumPy softmax oracle.
- The harness covers uneven per-row context lengths and reports append mismatches plus attention max-abs errors.

### Validation

```bash
python3 -m py_compile scripts/qwen35_batch_correctness.py
python3 scripts/qwen35_batch_correctness.py --rows 2 --json /tmp/hipengine-qwen35-batch-c2.json
python3 scripts/qwen35_batch_correctness.py --rows 4 --json /tmp/hipengine-qwen35-batch-c4.json
```

Results:

- c=2: `append_key_mismatch=0`, `append_value_mismatch=0`, `attn_batch_vs_c1_max_abs=0.0`, `attn_batch_vs_numpy_max_abs=2.235e-08`, passed.
- c=4: `append_key_mismatch=0`, `append_value_mismatch=0`, `attn_batch_vs_c1_max_abs=0.0`, `attn_batch_vs_numpy_max_abs=2.980e-08`, passed.

### Next

- Extend the harness upward to c=8 after the batched layer runner exists, and then compare generated token ids against independent resident c=1 sessions.

---

## 2026-05-14 — Add resident batch scheduler shell

### Scope

- Added torch-free `ResidentBatchScheduler` for request admission, pending queue management, active-slot compaction, prefill/decode work item emission, generated-token routing, completion, and reclaim.
- Exported scheduler types from `hipengine.generation` for future batch-friendly generator integration.
- Scheduler work items use stable request ids and row metadata from the dispatch batch scaffold.

### Validation

```bash
python3 -m compileall -q hipengine tests && python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_dispatch_batch.py tests/test_llm_generate.py -q
```

Results: all tests passed.

### Next

- Wire a Qwen3.5/PARO batch generator shell around `ResidentBatchScheduler`, then replace primitive-only correctness with generated-token c>N-vs-c1 checks.

---

## 2026-05-14 — Add graph shape-bucket cache to batch scheduler

### Scope

- Added `GraphBucketCache`/`GraphBucketStats` keyed by `BatchShapeKey` for decode/prefill/verify graph capture buckets.
- `ResidentBatchScheduler` now owns a graph-bucket cache alongside active request slots.
- Extended scheduler tests for cache hits/misses, clear semantics, and shape-key integration.

### Validation

```bash
python3 -m compileall -q hipengine tests && python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_llm_generate.py -q
```

Results: all tests passed.

### Next

- Build the Qwen3.5/PARO batched layer runner/generator shell on top of the scheduler and graph buckets.

---

## 2026-05-14 — Wire multi-token resident Qwen3.5/PARO generation

### Scope

- Updated the Qwen3.5/PARO registered generator to allow `max_tokens > 1` for greedy decoding.
- `LLM.generate()` now routes Qwen3.5/PARO prompts through real resident token-by-token prefill followed by multi-token autoregressive decode, still serial across prompts.
- Added unit coverage for prompt prefill sequencing, generated-token feedback positions, zero-token requests, and EOS stop handling.

### Validation

```bash
python3 -m compileall -q hipengine tests && python3 -m pytest tests/test_generation_qwen35_paro.py tests/test_llm_generate.py -q
```

Results: all tests passed.

### Next

- Add real-checkpoint E2E correctness gates against the parent nano-vllm-amd path for c=1 generated-token sequences, then extend to c>N once the batched layer runner is wired.

---

## 2026-05-14 — Add resident Qwen3.5/PARO E2E correctness gate

### Scope

- Added `scripts/qwen35_e2e_correctness.py` for real resident c=1 prefill/decode correctness checks.
- Gate records generated ids/logits, finite-logit status, repeated-run determinism, SpecDec-disabled metadata, and optional expected token ids for parent/reference comparisons.

### Validation

```bash
python3 -m py_compile scripts/qwen35_e2e_correctness.py
python3 scripts/qwen35_e2e_correctness.py --max-layers 1 --prompt-length 1 --max-new-tokens 1 --repeat 2 --expected-token-ids 62406 --json /tmp/hipengine-qwen35-e2e-c1.json
```

Result: passed; repeated c=1 one-layer resident runs both produced token id `62406` with finite logit `6.588076591491699`.

### Next

- Capture parent nano-vllm-amd expected token sequences for all-layer fixtures and extend this gate to generated-token equality vs parent; then add c>N runs once the batched layer runner lands.

---

## 2026-05-14 — Guard resident benchmark builds for profiler runs

### Scope

- Added `compiler_version`/`require_cached_build` plumbing to `Qwen35ParoResidentSession` kernel-library loading.
- Added `scripts/qwen35_paro_bench.py --compiler-version-file` and `--require-cached-build` so rocprof runs can fail fast instead of spawning `hipcc` inside the profiler.

### Validation

```bash
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 -m py_compile hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_paro_bench.py
python3 scripts/qwen35_paro_bench.py --max-layers 1 --prompt-length 1 --decode-tokens 0 --warmup-decode-tokens 0 --token-id 9707 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/hipengine-resident-cache-guard.json
python3 -m compileall -q hipengine scripts tests && python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_llm_generate.py -q
```

Result: cached-build guarded resident smoke passed and produced token id `62406`.

### Next

- Use this guard for any future `rocprofv3` resident benchmark/profile command.

---

## 2026-05-14 — Add SpecDec interfaces and KV transaction smoke

### Scope

- Added `DraftBatch`, `AcceptResult`, `DraftModel`, and `Verifier` under `hipengine.speculative`.
- Draft batches carry request ids, candidate rows, parent positions, draft depths, row-to-request maps, optional tree parents, and verify mode.
- Added smoke tests wiring draft batches through `FixedPagedKVPolicy.begin_transaction`, `commit`, and `rollback`.

### Validation

```bash
python3 -m compileall -q hipengine tests && python3 -m pytest tests/test_speculative_interfaces.py tests/test_kvcache_policy.py -q
```

Results: all tests passed.

### Next

- Attach MTP/EAGLE3/DFlash draft providers as plugins once target-model batch verification is available.

---

## 2026-05-14 — Add c=N benchmark protocol and blocked correctness artifact

### Scope

- Updated `docs/BENCHMARK.md` with c=N concurrent acceptance requirements, initial c=2/4/8 shapes, and required generated-token equality vs independent c=1 sessions.
- Added blocked artifact `benchmarks/results/2026-05-14-hipengine-qwen35-cn-correctness-blocked.json` recording the c=2/c=4 primitive batch correctness results and explicitly marking full generated-token c>N parity as pending.

### Validation

```bash
python3 -m json.tool benchmarks/results/2026-05-14-hipengine-qwen35-cn-correctness-blocked.json >/dev/null
python3 -m compileall -q hipengine scripts tests && python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Results: JSON artifact is valid; tests passed.

### Next

- Do not retain c=N throughput rows until generated-token equality vs independent c=1 sessions is implemented and green.

---

## 2026-05-14 — Capture parent c=1 fixture and fix first parity bugs

### Scope

- Captured a nano-vllm-amd parent Qwen3.5/PARO c=1 fixture for the OPTIMAL-style 512-token prompt / 32 decode-token shape using the parent synthetic torch CPU seed (`seed=1234`).
- Added `fixtures/qwen35_paro/parent_512_32_seed1234.json` with prompt IDs, expected parent decode tokens, parent prefill/decode throughput, and parent memory metrics.
- Extended `scripts/qwen35_e2e_correctness.py` to consume parent fixtures, distinguish public generate semantics from parent decode-loop semantics, and report timings plus owned device bytes.
- Fixed two parent-parity bugs found by the fixture:
  - Qwen normal RMSNorm weights now apply the checkpoint offset (`1.0 + weight`) for input/post/final norms; fused q/k head RMSNorm keeps checkpoint-direct offsets because the head kernel adds `1.0` internally.
  - Full-attention q_proj output is split as parent layout `[head0 query, head0 gate, head1 query, head1 gate, ...]` instead of assuming all queries followed by all gates.
- Added a small-context full-attention context+gate path so 512-token decode uses context attention plus a BF16 gate kernel before falling back to split-K for larger contexts.
- Recorded blocked parity artifact `benchmarks/results/2026-05-14-hipengine-qwen35-c1-parent-fixture-blocked.json`.

### Validation

```bash
# Parent fixture capture
cd /home/lhl/amd-gpu-tuning && <OPTIMAL env> \
  PYTHONPATH=nano-vllm-amd:paroquant mamba run -n therock --no-capture-output \
  python3 scripts/bench_paro_native_engine.py --prompt-len 512 --decode-len 32 \
  --decode-use-step-graph-replay --output /tmp/hipengine-parent-fixture-512-32.json --json

python3 -m compileall -q hipengine tests scripts
python3 -m pytest \
  tests/test_qwen35_paro_layout.py \
  tests/test_qwen35_decode_state.py \
  tests/test_qwen35_paged_attn_decode_plan.py \
  tests/test_qwen35_rotary_plan.py \
  tests/test_generation_qwen35_paro.py \
  tests/test_llm_generate.py -q

python3 scripts/qwen35_e2e_correctness.py \
  --max-layers 1 --prompt-length 1 --max-new-tokens 1 --repeat 1 \
  --expected-token-ids 627 \
  --json /tmp/hipengine-e2e-smoke-post-fixes.json

python3 scripts/qwen35_e2e_correctness.py \
  --fixture /tmp/parent_layer1_1_1_fixture.json --max-layers 1 --repeat 1 \
  --json /tmp/hipengine-layer1-parent-fixture-post-fixes.json

python3 scripts/qwen35_e2e_correctness.py \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 0 --repeat 1 \
  --json /tmp/hipengine-qwen35-parent-fixture-512-32-context-gate.json
```

Results:

- Unit tests passed: 48 passed.
- Layer-1 parent fixture passed: parent expected token `84`, hipENGINE produced `84`; parent prefill seed `6332` matched.
- Full 512/32 parent fixture still blocked: parent prefill seed `4403` matched, but generated-token parity missed at index 0 (`expected 1739`, hipENGINE `220`), then the remaining prefix matched (`220,16,15,...`).
- Current hipENGINE fixture timing remains sequential-prefill limited: ~113.65 tok/s prefill and ~96.24 tok/s decode vs parent fixture ~2682.66 tok/s prefill and ~116.26 tok/s decode.
- hipENGINE memory report is currently owned device buffers (~1.51 GiB), not parent-comparable allocator/VRAM peak (~18.8 GiB), so memory parity still needs a proper process/VRAM measurement path.

### Next

- Fix the remaining full c=1 parent fixture mismatch at the first decode token after a 512-token prompt; likely state/cache parity after sequential prefill vs parent needs deeper full-attention/linear-state comparison.
- Then wire native/bulk full prefill so prefill throughput and memory behavior can be compared to parent OPTIMAL rows.
- Only after c=1 fixture parity is green should generated-token c=2/c=4/c=8 be promoted beyond primitive correctness.

---

## 2026-05-14 — Add dense short-context full-attention decode and isolate c=1 parity blocker

### Scope

- Ported the parent `qwen35_full_attn_decode_context_tensor_kernel` into the gfx1100 attention build and registered a raw-pointer `full_attn_decode/w4_paro/bf16_context` wrapper.
- Routed Qwen3.5/PARO resident full-attention layers to the dense short-context decode path when `max_live_count < 1024`, matching the parent small-context branch before paged attention.
- Added `scripts/smoke.py --mode qwen35-full-attn-decode-hip` and cataloged the kernel in `docs/KERNELS.md`.
- Re-ran the 512/32 parent fixture gate. The first decode token is still blocked: dense short-context attention changes hipENGINE from `220,...` to `4096,220,16,...`, while the parent expects `1739,220,16,...`.
- Root-cause probe: the parent PARO native fixture runs FP16 activations/scales from the checkpoint (`embed`, RMSNorm, PARO scales/theta, LM head are `torch.float16`), while hipENGINE's current Qwen3.5/PARO resident path materializes those runtime tensors as BF16. Layer-0 prompt probes show BF16-vs-FP16 activation drift starts at input RMSNorm/rotation and is enough to flip close top logits after full decode (`parent top first-decode: 1739=6.4487, 220=6.3479, 4096=6.3336`; HIP BF16 dense path: 4096=6.7064, 220=6.5895, 1739=5.9954`).

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py \
  tests/test_qwen35_decode_state.py::test_qwen35_decode_state_runs_full_attention_moe_layer_chain -q
python3 scripts/smoke.py --mode qwen35-full-attn-decode-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-full-attn -- \
  python3 scripts/smoke.py --mode qwen35-full-attn-decode-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_e2e_correctness.py \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 0 --repeat 1 \
  --json /tmp/hipengine-qwen35-parent-fixture-dense-context.json
```

Results:

- Dense full-attention smoke passed: `max_abs=1.19e-07` vs NumPy BF16 softmax oracle.
- `rocprofv3` confirmed `qwen35_full_attn_decode_context_tensor_kernel` ran: `DurationNs=9440`, `VGPR_Count=32`, `Scratch_Size=0`, `LDS_Block_Size=1040`, `Workgroup_Size_X=256`.
- Targeted tests passed: 4 passed.
- Full 512/32 parent fixture remains blocked by FP16-vs-BF16 activation parity: parent expected `[1739, 220, 16, ...]`; hipENGINE BF16 dense path produced `[4096, 220, 16, ...]` with matching prefill seed `4403`.

### Next

- Decide whether Qwen3.5/PARO parent parity should port FP16 activation variants for the resident path or recapture/define a BF16 parent oracle. Exact generated-token equality against the current parent fixture is not a pure scheduler/cache bug; it crosses the activation dtype boundary.

---

## 2026-05-15 — Make gfx1100 wave32 the documented/build default

### Scope

- Updated the W7900/gfx1100 wavefront policy after the parent workspace probe: `-mcumode` is orthogonal to wavefront size and the HIP decode profile should be treated as wave32 unless `-mwavefrontsize64` is explicitly added for an isolated experiment.
- Added a `docs/PLAN.md` caveat section near the end: RDNA3 wave64 is architecturally real, but hipENGINE/nano-vllm-amd defaults to wave32 + ILP/VOPD exposure; wave64 requires separate flags, probes, correctness gates, ISA checks, and E2E benchmarks.
- Updated `docs/KERNELS.md` and `docs/ROOFLINE.md` to remove stale decode-wave64 wording and to document wave32-compatible reductions (`__shfl_down` within 32 lanes plus LDS/shared-memory exchange across waves).
- Updated `hipengine.core.build.PROFILES["decode"].wavefront` from `64` to `32`; decode flags remain `-mcumode` plus the unroll threshold and deliberately do not include `-mwavefrontsize64`.
- Updated dry-run build-plan tests to expect wave32 and added a guard that the decode profile does not carry `-mwavefrontsize64`.
- Clarified the PARO pack8 GEMV reduction comment: this path reduces 32-lane waves and explicitly sums cross-wave partials; it must not rely on a 64-thread block being one wave.

### Validation

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_build.py tests/test_dense_gemv_plan.py \
  tests/test_paro_awq_gemv_plan.py tests/test_paro_combine_plan.py \
  tests/test_paro_rotate_plan.py tests/test_paro_silu_plan.py \
  tests/test_qwen35_linear_attn_conv_plan.py \
  tests/test_qwen35_linear_attn_gdn_plan.py \
  tests/test_qwen35_paged_attn_decode_plan.py \
  tests/test_qwen35_paged_kv_write_plan.py \
  tests/test_qwen35_rmsnorm_plan.py tests/test_qwen35_rotary_plan.py \
  tests/test_qwen35_router_plan.py tests/test_w8a16_linear_plan.py -q
```

Results: 44 targeted build-plan/kernel-plan tests passed.

### Coordination note

- Left unrelated in-progress FP16 RMSNorm changes in `hipengine/kernels/hip_gfx1100/norm/{__init__.py,rmsnorm.hip,rmsnorm.py}` unstaged for their owner.

---

## 2026-05-15 — Start parent-mixed activation parity with FP16 PARO RMSNorm wrappers

### Decision

- For Qwen3.5/PARO c=1 parent fixture parity, target the parent nano-vllm-amd mixed activation ABI rather than recapturing a BF16-only oracle.
- Parent-compatible contract to implement: FP16 for checkpoint PARO/residual-stream tensors and lowp projection/MoE intermediates; BF16 for full-attention KV cache and q/k head RMSNorm offset weights consumed by the parent fused head-norm+rotary kernel; FP32 for recurrence, conv state, attention scores, and logits.
- Decomposed the implementation into tasks #38-#41: FP16 wrappers, materialization, resident-session switch, then fixture promotion.

### Scope

- Added raw-pointer FP16 variants for PARO RMSNorm out-kernels:
  - `hipengine_paro_rmsnorm_out_fp16` / `paro_rmsnorm_out_fp16(...)`
  - `hipengine_paro_add_rmsnorm_out_fp16` / `paro_add_rmsnorm_out_fp16(...)`
- Registered both under `KernelKey("hip_gfx1100", {"rmsnorm","add_rmsnorm"}, "w4_paro", "paro_out_fp16")`.
- Extended `scripts/smoke.py --mode paro-rmsnorm-hip` to validate both existing BF16 and new FP16 PARO RMSNorm/add-RMSNorm bit-exactness.
- Updated `docs/KERNELS.md` and `docs/IMPLEMENTATION.md` to record the parent-parity FP16 RMSNorm coverage and the remaining FP16 wrapper gap.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest tests/test_qwen35_rmsnorm_plan.py -q
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode paro-rmsnorm-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-paro-rmsnorm-fp16 -- \
  python3 scripts/smoke.py --mode paro-rmsnorm-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Results:

- Unit plan tests passed: `3 passed`.
- Smoke passed: BF16 `norm_bit_mismatch=0`, `add_norm_bit_mismatch=0`, `residual_bit_mismatch=0`; FP16 `fp16_norm_mismatch=0`, `fp16_add_norm_mismatch=0`, `fp16_residual_mismatch=0`.
- `rocprofv3` confirmed FP16 kernels ran on W7900:
  - `paro_rmsnorm_out_kernel<_Float16>`: `DurationNs=5800`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=1024`, `Workgroup_Size_X=256`.
  - `paro_add_rmsnorm_out_kernel<_Float16>`: `DurationNs=5320`, `VGPR_Count=32`, `Scratch_Size=0`, `LDS_Block_Size=1024`, `Workgroup_Size_X=256`.

### Next

- Continue task #38 by adding FP16 variants for PARO rotate/projection, router/MoE, W8A16 lowp, linear-attention lowp, and attention gate paths before switching resident materialization/session dtype.

---

## 2026-05-15 — Add parent-parity FP16 PARO rotate wrappers

### Scope

- Added raw-pointer FP16 ABI wrappers for `paro_rotate1`, `paro_rotate2`, and `paro_rotate3`:
  - `hipengine_paro_rotate1_fp16` / `paro_rotate1_fp16(...)`
  - `hipengine_paro_rotate2_fp16` / `paro_rotate2_fp16(...)`
  - `hipengine_paro_rotate3_fp16` / `paro_rotate3_fp16(...)`
- Registered the wrappers under `KernelKey("hip_gfx1100", "paro_rotate{1,2,3}", "w4_paro", "fp16")`.
- Extended `scripts/smoke.py --mode paro-rotate-hip` to keep the existing BF16 rotate2/3 oracle and add bit-exact FP16 rotate1/2/3 checks.
- Updated `docs/KERNELS.md` and `docs/IMPLEMENTATION.md` to record FP16 PARO rotate coverage for parent activation parity.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest tests/test_paro_rotate_plan.py -q
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode paro-rotate-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-paro-rotate-fp16 -- \
  python3 scripts/smoke.py --mode paro-rotate-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Results:

- Unit plan tests passed: `3 passed`.
- Smoke passed: BF16 rotate2/3 `mismatches=[0, 0, 0, 0, 0]`, `max_abs=0.0`; FP16 rotate1/2/3 `fp16_mismatches=[0, 0, 0, 0, 0, 0]`, `fp16_max_abs=0.0`.
- `rocprofv3` confirmed FP16 rotate kernels ran on W7900:
  - `paro_rotate1_kernel<_Float16>`: `DurationNs=11680`, `Scratch_Size=0`, `LDS_Block_Size=32`, `Workgroup_Size_X=4`.
  - `paro_rotate2_kernel<_Float16>`: `DurationNs=2680`, `Scratch_Size=0`, `LDS_Block_Size=32`, `Workgroup_Size_X=4`.
  - `paro_rotate3_kernel<_Float16>`: `DurationNs=2560`, `Scratch_Size=0`, `LDS_Block_Size=32`, `Workgroup_Size_X=4`.

### Next

- Continue task #38 with FP16 wrappers for PARO AWQ GEMV/projection paths, router/MoE, W8A16 lowp, linear-attention lowp, and attention gate paths.

---

## 2026-05-15 — Add FP16 generic PARO AWQ GEMV wrappers

### Scope

- Added parent-parity FP16 raw-pointer wrappers for generic non-MoE PARO projection GEMV:
  - `hipengine_gemv_awq_pack8_strided_fp16` / `gemv_awq_pack8_strided_fp16(...)`
  - `hipengine_gemv_awq_pack8_transposed_fp16` / `gemv_awq_pack8_transposed_fp16(...)`
  - `hipengine_gemv_awq_dual_pack8_strided_fp16` / `gemv_awq_dual_pack8_strided_fp16(...)`
  - `hipengine_gemv_awq_dual_pack8_transposed_fp16` / `gemv_awq_dual_pack8_transposed_fp16(...)`
- Registered the wrappers under `pack8_gemv`/`dual_pack8_gemv` `strided_fp16` and `transposed_fp16` variants.
- Extended `scripts/smoke.py --mode paro-pack8-gemv-hip` to validate both the existing BF16 oracle and the new FP16 single/dual generic pack8 paths bit-exactly.
- Updated `docs/KERNELS.md` and `docs/IMPLEMENTATION.md` to narrow the remaining parent-mixed projection gap to selected-MoE/shared/linear-attention paths.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest tests/test_paro_awq_gemv_plan.py -q
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode paro-pack8-gemv-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-paro-pack8-fp16 -- \
  python3 scripts/smoke.py --mode paro-pack8-gemv-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Results:

- Unit plan tests passed: `3 passed`.
- Smoke passed: BF16 `single_mismatch=0/0`, `dual_mismatch=0/0`, `max_abs=0.0`; FP16 `fp16_single_mismatch=0/0`, `fp16_dual_mismatch=0/0`, `fp16_max_abs=0.0`.
- `rocprofv3` confirmed FP16 generic pack8 kernels ran on W7900:
  - `gemv_awq_pack8_kernel<_Float16, false>`: `DurationNs=4721`, `Scratch_Size=0`, `LDS_Block_Size=256`, `Workgroup_Size_X=64`.
  - `gemv_awq_pack8_kernel<_Float16, true>`: `DurationNs=11680`, `Scratch_Size=0`, `LDS_Block_Size=256`, `Workgroup_Size_X=64`.
  - `gemv_awq_dual_pack8_kernel<_Float16, false, false>`: `DurationNs=11560`, `Scratch_Size=0`, `LDS_Block_Size=256`, `Workgroup_Size_X=64`.
  - `gemv_awq_dual_pack8_kernel<_Float16, true, true>`: `DurationNs=4800`, `Scratch_Size=0`, `LDS_Block_Size=256`, `Workgroup_Size_X=64`.

### Next

- Continue task #38 with FP16 wrappers for selected-MoE pack8/fused-rotate paths, router/MoE activations, W8A16 lowp, linear-attention lowp, and attention gate paths.

---

## 2026-05-15 — Add FP16 selected PARO AWQ GEMV wrappers

### Scope

- Added parent-parity FP16 raw-pointer wrappers for selected-MoE PARO pack8 GEMV and fused rotate→selected dual GEMV:
  - `gemv_awq_selected_dual_pack8_strided_rotate_out_fp16(...)`
  - `gemv_awq_selected_dual_pack8_strided_fp16(...)`
  - `gemv_awq_selected_dual_pack8_transposed_fp16(...)`
  - `gemv_awq_selected_pack8_strided_fp16(...)`
  - `gemv_awq_selected_pack8_transposed_fp16(...)`
- Registered the wrappers under `strided_fp16` / `transposed_fp16` variants for selected dual/single and fused rotate selected-dual.
- Extended `scripts/smoke.py --mode paro-selected-gemv-hip` and `--mode paro-selected-gemv-rotate-hip` with bit-exact FP16 oracles.
- Updated `docs/KERNELS.md` and `docs/IMPLEMENTATION.md`; remaining parent-mixed FP16 wrapper gaps are now linear-attention/shared/gated-attention paths.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest tests/test_paro_awq_gemv_plan.py -q
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode paro-selected-gemv-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode paro-selected-gemv-rotate-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-paro-selected-fp16 -- \
  python3 scripts/smoke.py --mode paro-selected-gemv-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-paro-selected-rotate-fp16 -- \
  python3 scripts/smoke.py --mode paro-selected-gemv-rotate-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Results:

- Unit plan tests passed: `3 passed`.
- Selected GEMV smoke passed: BF16 `dual_mismatch=0/0`, `single_mismatch=0/0`; FP16 `fp16_dual_mismatch=0/0`, `fp16_single_mismatch=0/0`.
- Fused rotate-selected smoke passed: BF16 `mismatch=0`, FP16 `fp16_mismatch=0`, `fp16_max_abs=0.0`.
- `rocprofv3` confirmed FP16 selected kernels ran on W7900:
  - `gemv_awq_selected_dual_pack8_strided_kernel<_Float16,false>`: `DurationNs=17240`, `Scratch_Size=0`, `LDS_Block_Size=256`, `Workgroup_Size_X=64`.
  - `gemv_awq_selected_dual_pack8_strided_kernel<_Float16,true>`: `DurationNs=14720`, `Scratch_Size=0`, `LDS_Block_Size=256`, `Workgroup_Size_X=64`.
  - `gemv_awq_selected_pack8_kernel<_Float16,false>`: `DurationNs=13680`, `Scratch_Size=0`, `LDS_Block_Size=256`, `Workgroup_Size_X=64`.
  - `gemv_awq_selected_pack8_kernel<_Float16,true>`: `DurationNs=12840`, `Scratch_Size=0`, `LDS_Block_Size=256`, `Workgroup_Size_X=64`.
  - `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel<_Float16,false>`: `DurationNs=21523`, `Scratch_Size=0`, `LDS_Block_Size=320`, `Workgroup_Size_X=64`.

### Next

- Continue task #38 with FP16 wrappers for W8A16 lowp/shared-expert, linear-attention lowp, router/shared-gate, and gated-attention paths.

---

## 2026-05-15 — Add FP16 W8A16 lowp wrapper

### Scope

- Added `hipengine_w8a16_linear_fp16_lowp_out` / `w8a16_linear_fp16_lowp_out(...)` for parent-mixed FP16 shared-expert/lowp W8A16 paths.
- Registered `w8a16_linear` variant `fp16_lowp_out` for both `w8a16` and `w4_paro` quant keys.
- Extended `scripts/smoke.py --mode w8a16-linear-hip` with a bit-exact FP16 lowp-output oracle.
- Updated `docs/KERNELS.md` and `docs/IMPLEMENTATION.md`; remaining parent-mixed FP16 wrapper gaps are linear-attention and gated-attention paths.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest tests/test_w8a16_linear_plan.py -q
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode w8a16-linear-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-w8a16-fp16 -- \
  python3 scripts/smoke.py --mode w8a16-linear-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Results:

- Unit plan tests passed: `3 passed`.
- Smoke passed: BF16/F32 `bf16_f32_max_abs=0.0`, F32/F32 `f32_f32_max_abs=4.76837158203125e-07`, BF16 lowp `lowp_mismatch=0`, FP16 lowp `fp16_lowp_mismatch=0`, `fp16_lowp_max_abs=0.0`.
- `rocprofv3` confirmed `w8a16_linear_lowp_out_kernel<_Float16>` ran on W7900: `DurationNs=6440`, `Scratch_Size=0`, `LDS_Block_Size=256`, `Workgroup_Size_X=64`.

### Next

- Continue task #38 with FP16 wrappers for linear-attention lowp and gated-attention paths.

---

## 2026-05-15 — Add FP16 linear-attention lowp wrappers

### Scope

- Added parent-mixed FP16 raw-pointer wrappers for Qwen3.5/PARO linear-attention lowp paths:
  - `qwen35_linear_attn_conv_decode_fp16(...)` / `hipengine_qwen35_linear_attn_conv_decode_fp16`
  - `qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16(...)` / `hipengine_qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16`
  - `qwen35_linear_attn_prefill_prepare_f32_fp16(...)` / `hipengine_qwen35_linear_attn_prefill_prepare_f32_fp16`
  - `qwen35_gdn_prefill_rmsnorm_gate_fp16(...)` / `hipengine_qwen35_gdn_prefill_rmsnorm_gate_fp16`
- Registered variants `linear_attn_conv_decode/fp16`, `gdn_recurrent_rmsnorm_gate/fp16_lowp`, `linear_attn_prefill_prepare/f32_fp16`, and `gdn_prefill_rmsnorm_gate/fp16` under `w4_paro`.
- Extended linear-attention conv/GDN/prefill smokes with FP16 oracles while preserving existing BF16/F32 coverage.
- Updated `docs/KERNELS.md` and `docs/IMPLEMENTATION.md`; remaining task #38 wrapper gap is gated full-attention output.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest tests/test_qwen35_linear_attn_conv_plan.py tests/test_qwen35_linear_attn_gdn_plan.py -q
python3 -m pytest tests/test_qwen35_linear_attn_conv_plan.py tests/test_qwen35_linear_attn_gdn_plan.py tests/test_qwen35_decode_state.py tests/test_generation_qwen35_paro.py -q
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode qwen35-linear-attn-conv-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode qwen35-linear-attn-gdn-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode qwen35-linear-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-linear-attn-fp16-conv -- \
  python3 scripts/smoke.py --mode qwen35-linear-attn-conv-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-linear-attn-fp16-gdn -- \
  python3 scripts/smoke.py --mode qwen35-linear-attn-gdn-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-linear-attn-fp16-prefill -- \
  python3 scripts/smoke.py --mode qwen35-linear-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Results:

- Plan/decode-state tests passed: targeted linear-attn plan tests `6 passed`; extended set `32 passed`.
- Conv smoke passed: F32/BF16/FP16 `*_out_max_abs=7.45e-09`; all state max abs `0`.
- GDN decode smoke passed: BF16 and FP16 `out_max_abs=2.98e-08`, `state_max_abs=1.49e-08`.
- Linear prefill smoke passed: BF16 `gated_mismatch=0`; FP16 `fp16_gated_mismatch=0`; `fp16_prepare_max_abs=5.96e-08`, `fp16_gdn_k2_out_max_abs=1.4e-09`, `fp16_gdn_k2_state_max_abs=1.12e-08`.
- `rocprofv3` confirmed FP16 kernels ran on W7900:
  - `qwen35_linear_attn_conv_decode_lowp_kernel<_Float16>`: `DurationNs=5680`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`.
  - `qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<_Float16>`: `DurationNs=9920`, `VGPR_Count=56`, `Scratch_Size=0`, `LDS_Block_Size=1616`, `Workgroup_Size_X=128`.
  - `qwen35_linear_attn_prefill_prepare_kernel<_Float16>`: `DurationNs=12960`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=1024`, `Workgroup_Size_X=128`.
  - `qwen35_gdn_prefill_rmsnorm_gate_fp16_kernel`: `DurationNs=3000`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=512`, `Workgroup_Size_X=128`.

### Next

- Finish task #38 by adding FP16 gated full-attention output wrappers, then move to task #39 parent activation dtype materialization.

---

## 2026-05-15 — Add FP16 full-attention gated output wrappers

### Scope

- Added parent-mixed FP16 gated full-attention raw-pointer wrappers:
  - `qwen35_full_attn_gate_mul_fp16(...)` / `hipengine_qwen35_full_attn_gate_mul_fp16`
  - `qwen35_paged_full_attn_decode_split_k_gate_fp16_spans(...)` / `hipengine_qwen35_paged_full_attn_decode_split_k_reduce_gate_fp16`
  - `qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_spans(...)` using the existing GQA context kernel plus FP16 gated reduce.
- Registered `full_attn_gate_mul/fp16`, `paged_attn_decode/bf16_split_k_gate_fp16_spans`, and `paged_attn_decode/bf16_split_k_gqa_gate_fp16_spans` under `w4_paro`.
- Extended `qwen35-full-attn-decode-hip` and `qwen35-paged-attn-gate-bf16-hip` smokes with bit-exact FP16 output oracles.
- Updated `docs/KERNELS.md` and `docs/IMPLEMENTATION.md` for the FP16 gated-output coverage.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py -q
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_decode_state.py -q
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode qwen35-full-attn-decode-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode qwen35-paged-attn-gate-bf16-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-attn-gate-fp16-full -- \
  python3 scripts/smoke.py --mode qwen35-full-attn-decode-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-attn-gate-fp16-split -- \
  python3 scripts/smoke.py --mode qwen35-paged-attn-gate-bf16-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Results:

- Plan/decode-state tests passed: paged-attn plan tests `3 passed`; paged-attn + decode-state set `26 passed`.
- Dense-context gate smoke passed: attention `max_abs=1.19e-07`, `gated_bf16_mismatch=0`, `gated_fp16_mismatch=0`.
- Split-K gate smoke passed: `bf16_mismatch=0`, `fp16_mismatch=0`, `bf16_max_abs=0`, `fp16_max_abs=0`.
- `rocprofv3` confirmed FP16 gated kernels ran on W7900:
  - `qwen35_full_attn_gate_mul_fp16_kernel`: `DurationNs=1360`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`, `Workgroup_Size_X=256`.
  - `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<_Float16>`: `DurationNs=10040`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=24`, `Workgroup_Size_X=8`.

### Next

- Continue task #38 by auditing remaining broad FP16 wrapper items from the TaskList (dense/dual GEMV, router hidden, SiLU/down-rotation, weighted combine/residual) before task #39 materialization.

---

## 2026-05-15 — Add FP16 dense GEMV wrappers

### Scope

- Audited task #38 after gated full-attention coverage: broad remaining FP16 wrapper items are dense/dual GEMV, router hidden, SiLU/down-rotation, and weighted/shared-gate combine/residual.
- Added `dense_gemv_out_fp16(...)` and `dense_dual_gemv_out_fp16(...)` C/Python wrappers around the existing parent dense GEMV templates using `_Float16` activations/weights/outputs.
- Registered FP16 dense GEMV variants under `bf16`/`w4_paro` variant `out_fp16` and native `fp16/out` registry keys.
- Extended `dense-gemv-hip` smoke with bit-exact FP16 single and dual GEMV oracles.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest tests/test_dense_gemv_plan.py -q
python3 scripts/smoke.py --mode dense-gemv-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-dense-fp16 -- \
  python3 scripts/smoke.py --mode dense-gemv-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Results:

- `tests/test_dense_gemv_plan.py`: `3 passed`.
- Dense GEMV smoke: BF16 `mismatch=0`, `max_abs=0.0`; FP16 single `fp16_mismatch=0`, `fp16_max_abs=0.0`; FP16 dual `dual_fp16_mismatch=0`.
- `rocprofv3` confirmed FP16 dense kernels ran on W7900:
  - `dense_gemv_out_kernel<_Float16>`: `DurationNs=3440`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=1024`, `Workgroup_Size_X=256`.
  - `dense_dual_gemv_out_kernel<_Float16>`: `DurationNs=4040`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=1024`, `Workgroup_Size_X=256`.

### Next

- Continue task #38 with router hidden FP16, SiLU/down-rotation FP16, and weighted/shared-gate combine FP16 wrappers as required by the parent-mixed materialization plan.

---

## 2026-05-15 — Add FP16 router hidden wrappers

### Scope

- Added Qwen3.5 router FP16-hidden raw-pointer wrappers while keeping BF16 router/shared-gate weights and FP32 logits/routing:
  - `qwen35_router_logits_fp16(...)` / `hipengine_qwen35_router_logits_fp16`
  - `qwen35_router_topk_shared_out_fp16(...)` / `hipengine_qwen35_router_topk_shared_out_fp16`
- Registered `router_logits/fp16`, `router_logits/w4_paro/fp16_hidden`, `router_topk_shared/*/out_fp16_hidden`, and `router_topk_shared/fp16/out` keys.
- Extended `qwen35-router-hip` smoke with an FP16 hidden top-k/shared route oracle.
- Updated `docs/KERNELS.md` to mark router/shared-gate FP16 hidden coverage landed.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest tests/test_qwen35_router_plan.py -q
python3 scripts/smoke.py --mode qwen35-router-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-router-fp16 -- \
  python3 scripts/smoke.py --mode qwen35-router-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Results:

- `tests/test_qwen35_router_plan.py`: `3 passed`.
- Router smoke: BF16 `selected_match=True`, `logits_max_abs=0.0`, `routing_max_abs=1.49e-08`; FP16 hidden `fp16_selected_match=True`, `fp16_logits_max_abs=4.77e-07`, `fp16_routing_max_abs=2.98e-08`.
- `rocprofv3` confirmed FP16 hidden router logits ran on W7900: `qwen35_router_logits_kernel<_Float16>` `DurationNs=3160`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=256`, `Workgroup_Size_X=64`.

### Next

- Continue task #38 with FP16 SiLU/down-rotation wrappers and FP16 weighted/shared-gate combine wrappers.

---

## 2026-05-15 — Add FP16 PARO SiLU/down-rotation wrappers

### Scope

- Added FP16 raw-pointer wrappers for selected-expert activation/down-rotation:
  - `silu_mul_dual_out_fp16(...)`
  - `silu_mul_dual_rotate_out_fp16(...)`
  - `silu_mul_pair_rotate_out_fp16(...)`
- Registered `out_fp16` variants for `silu_mul_dual`, `silu_mul_dual_rotate`, and `silu_mul_pair_rotate` under `bf16`/`w4_paro`, plus native `fp16/out` keys.
- Extended `paro-silu-hip` with bit-exact FP16 dual SiLU and dual/pair rotate oracles.
- Updated `docs/KERNELS.md` to mark activation/down-rotation FP16 wrappers landed.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest tests/test_paro_silu_plan.py -q
python3 scripts/smoke.py --mode paro-silu-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-silu-fp16 -- \
  python3 scripts/smoke.py --mode paro-silu-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Results:

- `tests/test_paro_silu_plan.py`: `3 passed`.
- PARO SiLU smoke: BF16 `dual_mismatch=0`, `dual_rotate_mismatch=0`, `pair_rotate_mismatch=0`; FP16 `dual_fp16_mismatch=0`, `dual_rotate_fp16_mismatch=0`, `pair_rotate_fp16_mismatch=0`.
- `rocprofv3` confirmed FP16 SiLU kernels ran on W7900:
  - `silu_mul_dual_out_kernel<_Float16>`: `DurationNs=1680`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`.
  - `silu_mul_dual_rotate_out_kernel<_Float16>`: `DurationNs=11960`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=64`.
  - `silu_mul_pair_rotate_out_kernel<_Float16>`: `DurationNs=8480`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=64`.

### Next

- Finish task #38's broad wrapper audit with FP16 weighted/shared-gate combine wrappers; then re-evaluate whether task #38 can close or needs runtime materialization from task #39 first.

---

## 2026-05-15 — Add DFlash/DDTree native implementation plan

### Scope

- Created `docs/DFLASH.md` as the hipENGINE-side plan for a proper native
  DFlash implementation.
- Consolidated lessons from `~/amd-gpu-tuning/PLAN-DFLASH.md`,
  `docs/SPECULATIVE-DECODE.md`, `docs/DFLASH-FRESH-EYES.md`, recent
  2026-05-15 WORKLOG entries, and local references (`reference/ddtree-mlx`,
  `reference/hipfire`, `reference/lucebox-hub/dflash`).
- Main decision recorded: the current Python/PyTorch DFlash harness has proven
  correctness and the corrected tree-kernel shape, but the remaining speed gap
  is a native-runtime verifier problem. The production path belongs in
  hipENGINE as a torch-free C++/HIP hot loop with stable buffers, persistent
  state rings, device-side accept summaries, and graph-capturable fixed shapes.
- Included DDTree-specific ABI/semantics: flat topological tree, `parent_ids`,
  positions/depths, ancestor mask, target-top1 edge following, no DFS-state
  contamination, commit by state/KV slot copy, and budget=4 as the default
  promotion target after chain DFlash beats AR.
- Added a phased port plan: source-lineage refresh, native chain verifier,
  device-side top1/accept, native DFlash drafter + draft context KV, DDTree
  compiler/tree verify, graph capture, and benchmark/promotion gates.

### Validation

```bash
git -C /home/lhl/hipengine status -sb
python3 - <<'PY'
from pathlib import Path
p = Path('/home/lhl/hipengine/docs/DFLASH.md')
text = p.read_text()
assert '# hipENGINE DFlash / DDTree Native Implementation Plan' in text
assert 'DDTree details to preserve' in text
assert 'First concrete hipENGINE tasks' in text
print(len(text.splitlines()), 'lines')
PY
git -C /home/lhl/hipengine diff --stat -- docs/DFLASH.md WORKLOG.md
git -C /home/lhl/hipengine diff --staged --name-only
git -C /home/lhl/hipengine diff --staged -- docs/DFLASH.md WORKLOG.md
```

Notes:

- No GPU run required; this is docs/process planning only.
- Left unrelated in-progress FP16 wrapper changes in the worktree untouched.

### Next

- Use `docs/DFLASH.md` as the launch checklist when starting hipENGINE DFlash:
  first source-lineage refresh for corrected tree Conv/GDN and pack8 small-row
  defaults, then a native topk=1 chain verifier before DDTree policy work.

---

## 2026-05-15 — Add FP16 PARO combine wrappers

### Scope

- Added FP16 raw-pointer wrappers for the weighted/shared-gate combine family while keeping FP32 routing weights/gate logits:
  - `weighted_sum_out_fp16_f32w(...)`
  - `weighted_sum_shared_gate_combine_residual_out_fp16_f32w(...)`
  - `weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w(...)`
  - `shared_gate_combine_out_fp16(...)`
  - `shared_gate_combine_residual_out_fp16(...)`
- Registered `out_fp16`/`batch_out_fp16` variants under `bf16`/`w4_paro` and native `fp16` combine keys.
- Extended `paro-combine-hip` with bit-exact FP16 weighted/shared/residual and batched fused oracles.
- Updated `docs/KERNELS.md` to mark weighted/shared-gate combine FP16 wrappers landed.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest tests/test_paro_combine_plan.py -q
python3 scripts/smoke.py --mode paro-combine-hip --rows 4 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-combine-fp16 -- \
  python3 scripts/smoke.py --mode paro-combine-hip --rows 4 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Results:

- `tests/test_paro_combine_plan.py`: `3 passed`.
- PARO combine smoke: BF16 mismatches all `0`; FP16 `weighted_fp16_mismatch=0`, `fused_fp16_mismatch=0`, `batch_fused_fp16_mismatch=0`, `shared_fp16_mismatch=0`, `shared_residual_fp16_mismatch=0`.
- `rocprofv3` confirmed FP16 combine kernels ran on W7900 with `Scratch_Size=0`, including:
  - `weighted_sum_out_kernel<_Float16,float>`: `DurationNs=1880`, `VGPR_Count=8`.
  - `weighted_sum_shared_gate_combine_residual_batch_out_kernel<_Float16,float>`: `DurationNs=2320`, `VGPR_Count=16`.
  - `shared_gate_combine_residual_out_kernel<_Float16>`: `DurationNs=11920`, `VGPR_Count=16`.

### Next

- Re-run the task #38 audit/guard; if no FP16 wrapper gaps remain, mark task #38 complete and continue to task #39 parent activation dtype materialization.

---

## 2026-05-15 — Materialize Qwen3.5/PARO parent mixed dtypes

### Scope

- Started task #39 and changed runtime materialization dtype policy to match the parent mixed contract:
  - normal RMSNorm weights are materialized as FP16 after applying the Qwen `+1.0` offset;
  - full-attention `q_norm`/`k_norm` checkpoint-direct weights remain BF16 for the fused head RMSNorm/rotary path;
  - PARO projection scales/theta/channel scales, router/shared-gate weights, stacked expert scales, and down-rotation theta/channel scales materialize as FP16;
  - linear-attention recurrence/conv/norm state tensors remain FP32; W8A16 scales remain FP32; qweights/qzeros/pairs keep integer dtypes.
- Updated Qwen3.5/PARO layout tests to assert FP16/BF16/FP32 materialized tensor dtypes and copied byte sizes.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py -q
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Results:

- Qwen3.5/PARO layout + materialize tests: `21 passed`.
- Loop guard suite: `64 passed`.

### Next

- Audit runtime dispatch to ensure the new FP16 materialized tensors select the FP16 wrappers from task #38, then mark task #39 complete if no BF16-only path remains.

### Dispatch audit addendum

- Runtime source audit after materialization showed `hipengine/runtime/qwen35_paro.py` still routes resident decode workspaces through the existing BF16 wrapper methods (`*_bf16`), including router, dense GEMV, PARO SiLU/rotate, and weighted shared/residual combine. This is expected to remain task #40 scope; task #39 is limited to checkpoint/host/device materialization dtype policy and tests.
- Hot-path torch audit for touched/runtime paths:
  `rg -n "^\\s*import torch|^\\s*from torch" hipengine/runtime hipengine/generation hipengine/llm.py hipengine/loading/qwen35_paro.py || true`
  - Result: no executable torch imports.

### Next

- Mark task #39 complete after guard passes; start task #40 by switching resident Qwen3.5/PARO session workspaces/dispatch to the FP16 wrappers from task #38 while preserving BF16 KV/full-attention q/k norm exceptions.

### Loop record

- After the guard passed, marked Task #39 completed. Active loop measurement recorded `open_or_partial_items=9` with guard `64 passed` and prompt verifier pass; no perf artifact was produced.

---

## 2026-05-15 — Switch resident Qwen3.5/PARO runtime to parent-mixed FP16

### Scope

- Completed Task #40 by switching `Qwen35ParoResidentSession`/`Qwen35ParoDecodeState` resident activations from the BF16-only bring-up path to the parent-mixed FP16 path while preserving the known exceptions:
  - embedding, hidden, scratch, MoE, router, dense GEMV, PARO SiLU/rotate/combine, linear-attention, full-attention output/gate, final norm, and final lowp scratch now use FP16 resident tensors/wrappers;
  - full-attention KV caches stay BF16;
  - full-attention q/k head RMSNorm inputs stay BF16 for the fused parent head-norm/rotary path;
  - the temporary lm-head path still consumes BF16, so final-norm FP16 output is cast to BF16 before `lm_head_fp16_argmax_bf16(...)`.
- Added helper variants needed by the resident parent-mixed path:
  - `qwen35_split_qgate_fp16(...)` for full-attention q/gate split;
  - FP16 value-input paged-KV write wrappers for scalar and batched `KVLiveSpans` paths;
  - FP16 runtime embedding lookup helpers;
  - FP16/BF16 cast wrappers including `fp16_to_bf16(...)`.
- Updated targeted plan tests and `scripts/smoke.py`; updated `docs/KERNELS.md` and wrote blocked correctness artifact `benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-mixed-blocked.json`.

### Validation

```bash
python3 scripts/check_lineage.py --kind kernel --diff stat
python3 -m compileall -q hipengine tests scripts && python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
python3 scripts/qwen35_e2e_correctness.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-new-tokens 1 --max-layers 0 --repeat 1 --json /tmp/hipengine-qwen35-parent-fp16-iter9.json
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-qwen35-fp16-switch-rotary -- \
  python3 scripts/smoke.py --mode qwen35-rotary-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-qwen35-fp16-switch-kv -- \
  python3 scripts/smoke.py --mode qwen35-paged-kv-write-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 -m compileall -q hipengine tests scripts && python3 -m pytest tests/test_cast_plan.py tests/test_qwen35_decode_state.py tests/test_runtime_state_plan.py tests/test_qwen35_paged_kv_write_plan.py tests/test_qwen35_rotary_plan.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Results:

- Source-lineage check reported expected parent drift from `nano-vllm-amd` after baseline `22405a9` (`qwen35_expert.hip`, `smoke.hip`, `paroquant_kernels.py`); this iteration used existing stable hipENGINE bodies and added local dtype/helper variants only.
- Active-loop guard passed: 68 targeted tests passed.
- Extended local test bundle passed (`[100%]`, 80 test dots across cast/runtime/rotary/KV/generation/layout suites).
- Parent fixture correctness remains blocked but narrower: full resident c=1 fixture run was finite/deterministic, but HIP seed token was `220` and first decode token was `58` with top logit `9.434697151184082`; parent fixture expected first generated token `1739`. No performance claim retained.
- `rocprofv3` W7900 evidence:
  - rotary smoke: `partial_max_abs=0`, `head_max_abs=2.38e-07`, `position_max_abs=2.38e-07`, `split_fp16_query_max_abs=0`, `split_fp16_gate_mismatch=0`; dispatch included `qwen35_split_qgate_fp16_kernel`, `DurationNs=3720`.
  - paged-KV smoke: `mixed_mismatch=0/0`, `mixed_fp16_mismatch=0/0`, `f32_mismatch=0/0`, `untouched_nonzero=0`; dispatch included `qwen35_write_paged_kv_mixed_value_position_tensor_kernel<_Float16>`, `DurationNs=5400`.

### Loop record

- Marked Task #40 completed. Active loop iteration 9 recorded `open_or_partial_items=8` (down from 9), guard pass, prompt-verifier pass, and explicit `parent_fixture_e2e_blocker` failure with token/logit evidence for Task #41.

### Next

- Start Task #41: promote/narrow the parent fixture by bisecting the remaining c=1 parity gap at per-layer hidden/logit checkpoints. The broad BF16-vs-FP16 activation policy is no longer the only blocker; next evidence should identify the first layer or projection where hipENGINE diverges from parent.
- Hot-path torch audit for touched runtime/generation paths:
  `rg -n "^\\s*import torch|^\\s*from torch" hipengine/runtime hipengine/generation hipengine/llm.py hipengine/loading/qwen35_paro.py || true` → no executable torch imports.

---

## 2026-05-15 — Narrow Qwen3.5/PARO c=1 fixture after parent-mixed switch

### Scope

- Started Task #36 after closing the activation-parity umbrella and reproduced the post-FP16-switch blocker.
- Parent-source audit found two materialization mismatches against `nano-vllm-amd`:
  - native router/shared-gate concatenates `router.weight` and `shared_expert_gate.weight` then casts the combined matrix to BF16 before `qwen35_router_logits_kernel`;
  - fused q/k head RMSNorm+RoPE receives BF16 *offset* weights computed as `(checkpoint + 1 -> FP16 -> BF16 -> -1)`, not checkpoint-direct BF16.
- Updated hipENGINE runtime materialization accordingly and refreshed layout tests.
- Added blocked artifact `benchmarks/results/2026-05-15-hipengine-qwen35-c1-router-qnorm-blocked.json` with parent and HIP top-k evidence.

### Validation and probes

```bash
# HIP forced-seed probe before the router fix.
python3 - <<'PY'
# one-off resident-session top-k probe; writes /tmp/hipengine-qwen35-parent-fp16-forced-seed-probe.json
PY

# Parent top-k probe using OPTIMAL flags and nano-vllm-amd parent.
cd /home/lhl/amd-gpu-tuning && <OPTIMAL env flags> \
  PYTHONPATH=nano-vllm-amd:paroquant mamba run -n therock --no-capture-output \
  python3 /tmp/hipengine_parent_paro_topk_probe.py

python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_qwen35_decode_state.py -q

python3 scripts/qwen35_e2e_correctness.py \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json \
  --max-new-tokens 1 --max-layers 0 --repeat 1 \
  --json /tmp/hipengine-qwen35-router-qnorm-fix-e2e.json

python3 - <<'PY'
# one-off resident-session top-k probe after router BF16 fix; writes /tmp/hipengine-qwen35-router-bf16-fix-topk.json
PY
```

Results:

- Before the router fix, forcing the previously observed parent prefill seed `4403` into HIP after the 512-token prompt did **not** recover parent output: HIP produced token `329` with top logit `8.800190925598145`; this proved the seed-`220` run had an invalid post-prefill state, not just a final sampler/lm-head issue.
- Parent probe with OPTIMAL flags:
  - bulk prefill seed `4403`; first decode argmax `1739`.
  - sequential prefill seed `4403`; first decode top logits: `1739=6.448723793029785`, `220=6.3479084968566895`, `4096=6.333559036254883`, `68=5.902808666229248`, `4403=5.789081573486328`.
- Targeted materialization/decode-state tests passed: 48 tests.
- After router BF16 + q/k head-norm offset fixes, HIP fixture still fails exact generated-token equality, but the blocker is narrower and back to the close first-token ordering:
  - `seed_token_ids=[4403]` (restored from the bad `220` seed);
  - first decode token `4096`, top logit `6.751620769500732`, parent expected `1739`.
  - HIP top logits after the router fix: prefill `4403=7.953364849090576`, `220=6.896111488342285`, `1739=6.646887302398682`; first decode `4096=6.730076789855957`, `220=6.6274542808532715`, `1739=6.0844526290893555`.
- No performance claim retained; all timings in the JSON artifact are correctness-smoke context only.

### Next

- Bisect per-layer hidden/logit outputs against parent sequential mode. The broad activation dtype and router materialization bugs are no longer the only blocker; the remaining miss is a close ordering drift where HIP raises `4096`/`220` relative to parent token `1739` after the full 512-token prompt.

---

## 2026-05-15 — Prefix-bisect Qwen3.5/PARO parent fixture mismatch

### Scope

- Continued Task #36 with a prefix top-k bisect against parent sequential mode.
- Wrote one-off probes outside the repo:
  - `/tmp/hipengine_prefix_probe.py` runs hipENGINE resident c=1 on the 512-token fixture for selected `max_layers` prefixes.
  - `/tmp/parent_prefix_probe.py` runs `nano-vllm-amd` parent sequential prefill/decode under OPTIMAL flags for the same prefixes.
- Added blocked diagnostic artifact `benchmarks/results/2026-05-15-hipengine-qwen35-prefix-bisect-blocked.json`.

### Validation and probes

```bash
PYTHONPATH=/home/lhl/hipengine python3 /tmp/hipengine_prefix_probe.py
cd /home/lhl/amd-gpu-tuning && <OPTIMAL env flags> \
  PYTHONPATH=nano-vllm-amd:paroquant mamba run -n therock --no-capture-output \
  python3 /tmp/parent_prefix_probe.py
```

Results:

- Prefix `max_layers=1` (layer 0 only) matches parent argmax:
  - parent prefill seed `111`, decode argmax `75`;
  - HIP prefill seed `111`, decode argmax `75`.
- First argmax divergence appears at `max_layers=2` (layers 0-1; layer 1 is linear_attention):
  - parent prefill top5: `83=8.541163444519043`, `6245=8.30211353302002`, `4144=8.193629264831543`, `932=8.128518104553223`, `37685=7.775232791900635`;
  - HIP prefill top5: `4144=8.706582069396973`, `6245=8.700551986694336`, `83=8.246453285217285`, `932=7.793816566467285`, `37685=7.784671783447266`;
  - parent first decode argmax `60822`; HIP first decode argmax `69906`.
- Later prefixes can re-match prefill argmax but still differ in decode (`max_layers=3`: prefill seed `315` for both, decode parent `467` vs HIP `441`; `max_layers=4`: prefill seed `169941` for both, decode parent `156206` vs HIP `25046`). Full 40-layer fixture remains parent `1739` vs HIP `4096` for first decode.

### Next

- Inspect layer 1 linear-attention/MoE internals against parent: hidden after input RMSNorm, QKV/Z/AB projections, Conv/GDN recurrent state after 512 prompt tokens, out-proj, post-attention add-RMSNorm, router logits/top-k, and MoE/shared outputs.

---

## 2026-05-15 — Fix Qwen3.5/PARO c=1 parent fixture parity

### Scope

- Completed Task #36 by fixing the remaining full resident c=1 parent fixture mismatch.
- Root cause from one-off parent/HIP component probes:
  - Layer-0 internals matched parent through input RMSNorm, QKV/Z/A/B projections, Conv/GDN, out-proj, post-attention add-RMSNorm, router logits, routing weights, and selected experts.
  - The first material divergence was the selected/shared MoE output: before the fix, layer-0 `mlp_output` differed by `max_abs=0.00836181640625`, `mean_abs=0.0015244119567796588`, `rmse=0.0019268736941739917`.
  - Parent rotates the MoE gate/up input via `experts.gate_up_weight_{pairs,theta,channel_scales}` before selected gate/up pack8 GEMV; hipENGINE was feeding the unrotated post-norm hidden into selected gate/up GEMV.
- Runtime fix:
  - Add runtime materialization for `gate_up_weight_pairs`, `gate_up_weight_theta`, and `gate_up_weight_channel_scales` on full-attention and linear-attention MoE paths.
  - Add `moe.gate_up_input` scratch and call `paro_rotate1_{bf16,fp16}` before selected gate/up pack8 GEMV.
  - Update decode-state ordering tests to assert the parent gate-up rotation step.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py -q

PYTHONPATH=/home/lhl/hipengine python3 /tmp/hip_layer0_components.py
python3 - <<'PY'
# compare /tmp/parent-layer0-components.npz vs /tmp/hip-layer0-components.npz
PY

python3 scripts/qwen35_e2e_correctness.py \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json \
  --max-new-tokens 32 --max-layers 40 \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json
```

Results:

- Targeted tests: `48 passed`.
- Synthetic decode-state smoke `python3 scripts/smoke.py --mode paro-moe-c1-state-hip --hidden-size 8` still passed with `final_mismatch=0` after adding the gate-up rotation input.
- Post-fix layer-0 component comparison: all stages before MoE output match exactly or within FP32 recurrent noise; final `mlp_output` reduced to `max_abs=1.52587890625e-05`, `mean_abs=2.1117739379405975e-07`, `rmse=1.241574182131444e-06`.
- Prefix probe after the fix now matches the parent full fixture first token at `max_layers=40`: seed `4403`, first decode `1739`.
- Full 512/32 parent fixture passed exact generated-token equality twice:
  - generated ids `[1739, 220, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]`;
  - seed ids `[4403, 4403]`, `finite_logits=true`, `deterministic=true`, `expected_match=true`, `passed=true`.
- Artifact: `benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json`.
- No performance claim retained; timings in the artifact are correctness context only.

### Next

- Continue the parity TaskList with the now-unblocked c>N/native compact prefill items before promoting any throughput claims.

---

## 2026-05-15 — Add Qwen3.5/PARO c>N serial slot runner bridge

### Scope

- Completed Task #33 after c=1 parent fixture parity was unblocked.
- Added `Qwen35ParoResidentSession.step_batch_serial(...)`, a correctness-first c>N bridge that runs one decode token per physical batch slot using the existing parent-parity c=1 layer path.
- The bridge is explicitly serial (not a throughput path): it consumes `ResidentBatchScheduler` work ownership, batch-shaped resident slot buffers, and per-slot state/cache views while native c-aware layer kernels remain Task #15 work.

### Implementation notes

- Added slot-offset helpers for:
  - batch hidden/next-hidden row views;
  - per-slot position/context scalar tensors and `KVLiveSpans`;
  - per-slot linear-attention conv/recurrent state views;
  - per-slot full-attention KV cache views.
- Extended `_run_layers(...)` with `slot` and `persist_aliases` parameters so normal c=1 `step(...)` keeps existing alias behavior, while `step_batch_serial(...)` runs rows without permanently replacing the c=1 aliases.
- Added `scripts/qwen35_batch_serial_correctness.py` scheduler mode and a unit test for slot pointer offsets/spans in `tests/test_qwen35_resident_batch_layout.py`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q

python3 scripts/qwen35_batch_serial_correctness.py \
  --prompt-length 16 --max-layers 40 --batch-size 2 --scheduler \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-c2-scheduler-serial-runner-accepted.json
```

Results:

- Scheduler/layout tests: `13 passed`.
- c=2 scheduler-backed serial slot-runner smoke passed for two deterministic 16-token prompts at `max_layers=40` (full model path):
  - independent c=1 expected row 0: seed `220`, decode `17`;
  - scheduler c=2 batch-serial row 0: seed `220`, decode `17`;
  - independent c=1 expected row 1: seed `96191`, decode `96523`;
  - scheduler c=2 batch-serial row 1: seed `96191`, decode `96523`;
  - logits matched exactly for the sampled seed/decode tokens, and scheduler completed request 0 with `[17]`, request 1 with `[96523]`.
- c=1 parent fixture regression remained exact after the slot refactor (`passed=true`, `expected_match=true`).
- Artifact: `benchmarks/results/2026-05-15-hipengine-qwen35-c2-scheduler-serial-runner-accepted.json`.
- No performance claim retained; this is a correctness bridge and remains serial.

### Next

- Extend the c>N correctness gate (Task #35) with finite-logit checks, graph/occupancy metadata, and c=4/c=8 shapes before rollup promotion.
- Replace serial row execution with native c-aware layer kernels under Task #15 only after scheduler correctness stays green.

---

## 2026-05-15 — Accept c>N generated-token equality gate

### Scope

- Completed Task #35 by extending `scripts/qwen35_batch_serial_correctness.py` from a c=2 smoke into a scheduler-backed c>N generated-token equality gate.
- The gate remains explicitly correctness-only: `step_batch_serial(...)` runs rows serially over batch-shaped resident slots, so no throughput claim is retained and native compact/c-aware kernels remain Task #15.

### Implementation notes

- Generalized the script to arbitrary positive `--batch-size` and scheduler-driven mode.
- Added finite-logit checks (`finite_logits`) and separate `generated_match`/`passed` fields.
- Added scheduler metadata to artifacts:
  - admitted request ids, slot maps, active counts;
  - prefill work-item count/order;
  - decode `BatchShapeKey` including active mask/top-k/experts/replay metadata;
  - `GraphBucketCache` stats;
  - completed request generated tokens and final occupancy.
- Added aggregate gate artifact `benchmarks/results/2026-05-15-hipengine-qwen35-cn-generated-equality-accepted.json`, superseding the prior blocked `2026-05-14-hipengine-qwen35-cn-correctness-blocked.json` for generated-token equality only.

### Validation

```bash
python3 scripts/qwen35_batch_serial_correctness.py \
  --prompt-length 16 --max-layers 40 --batch-size 2 --scheduler \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-c2-scheduler-serial-runner-accepted.json
python3 scripts/qwen35_batch_serial_correctness.py \
  --prompt-length 8 --max-layers 40 --batch-size 4 --scheduler \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-c4-scheduler-serial-runner-accepted.json
python3 scripts/qwen35_batch_serial_correctness.py \
  --prompt-length 8 --max-layers 40 --batch-size 8 --scheduler \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-c8-scheduler-serial-runner-accepted.json
```

Results:

- c=2/4/8 all passed with `finite_logits=true`, `generated_match=true`, `passed=true`.
- c=2 full-model prompt-16 generated ids: request 0 `[17]`, request 1 `[96523]`; both matched independent c=1 sessions exactly.
- c=4 full-model prompt-8 generated ids: `[96342]`, `[220]`, `[321]`, `[3709]`; all matched independent c=1 sessions exactly.
- c=8 full-model prompt-8 generated ids: `[96342]`, `[220]`, `[321]`, `[3709]`, `[198]`, `[13]`, `[16]`, `[248068]`; all matched independent c=1 sessions exactly.
- Decode shape metadata recorded active_c `2`, `4`, and `8` with `active_mask` all true, `top_k=8`, `experts_per_token=8`, `replay_steps=1`; graph bucket stats were `entries=1`, `hits=1`, `misses=1` for each shape.
- Standard guard remained green: `69 passed`.

### Next

- Task #18 is now unblocked to update benchmark rollups/artifact summaries after correctness parity.
- Task #15 remains the native compact/c-aware performance path; do not promote c>N throughput while the accepted c>N generated-token gate is still serial.

---

## 2026-05-15 — Update benchmark rollups for accepted parity gates

### Scope

- Completed Task #18 for the accepted correctness-parity artifacts.
- Updated `benchmarks/README.md` `Last updated` to `2026-05-15`.
- Added non-throughput rows for:
  - Qwen3.5/PARO c=1 parent fixture equality (`2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json`);
  - Qwen3.5/PARO c=N generated equality (`2026-05-15-hipengine-qwen35-cn-generated-equality-accepted.json`).
- Updated `benchmarks/CHANGELOG.md` with 2026-05-15 correctness and rollup entries.

### Notes

- No hipENGINE throughput row was promoted. The c=N accepted gate is explicitly serial (`step_batch_serial`) and remains a correctness gate only.
- The current-fastest hipENGINE throughput table still has no accepted row; Task #15 remains the native compact/c-aware path needed before retaining c>N performance claims.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: `69 passed`.

---

## 2026-05-15 — Label Qwen3.5/PARO c>N serial fallback as non-benchmarkable

### Scope

- Narrowed Task #15 by adding executable metadata for the current resident c>N path.
- Added `Qwen35ParoResidentBatchExecution` and `Qwen35ParoResidentSession.batch_execution_metadata()`.
- `step_batch_serial(...)` remains the correctness bridge over batch-shaped slots, but artifacts now explicitly report:
  - `row_execution=serial_c1_layer_path`
  - `native_compact_prefill=false`
  - `native_caware_decode=false`
  - `throughput_claim_eligible=false`
  - blockers for native compact/grouped MoE c>N prefill and c-aware full-attention graph replay.
- Extended `scripts/qwen35_batch_serial_correctness.py` payloads with `batch_execution` and `benchmark_eligible` so future c>N generated-equality artifacts cannot be mistaken for retained throughput rows.

### User question

- Clarified that `w4a16` is the broad 4-bit-weight/16-bit-activation quantization class, while `w4_paro` is hipENGINE's concrete PARO AWQ packed-layout/plugin variant under that umbrella.
- User also requested a FastAPI/OpenAI-compatible server. I proposed an optional `hipengine[server]` non-streaming first pass and asked whether SSE streaming should be included in v1 before changing scope.

### Validation

```bash
python3 -m pytest tests/test_qwen35_resident_batch_layout.py -q
```

Result: `10 passed`.

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: `70 passed`.

```bash
python3 - <<'PY'
from types import SimpleNamespace
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession
session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
session.layer_limit = 3
session.config = SimpleNamespace(layer_types=("linear_attention", "linear_attention", "full_attention"))
print(session.batch_execution_metadata(scheduler_owned=True).to_json_dict())
PY
```

Result includes `throughput_claim_eligible: False`, `native_compact_prefill: False`, `native_caware_decode: False`, and a full-attention layer blocker.

### Notes

- The loop's original TaskList verify command is now stale because completed parity tasks have been compacted out of the active TaskList file; the active TaskList contains only open #12 and #15, and a robust count over that file prints `2`.

---

## 2026-05-15 — Add native-prefill planning metadata for Qwen3.5/PARO

### Scope

- Narrowed Task #15 by making native prefill coverage an explicit runtime planning contract.
- Added `Qwen35ParoNativePrefillPlan` and `Qwen35ParoResidentSession.native_prefill_plan()`.
- `batch_execution_metadata()` now embeds the native prefill plan in its JSON payload, so future c>N correctness artifacts record:
  - `linear_prefix_layers`
  - `full_layer_limit_native`
  - `first_unsupported_layer`
  - `first_unsupported_type`
  - native-prefill blockers.
- `prefill_linear_tokens_native()` now uses the same plan for its NotImplemented boundary instead of duplicating the unsupported-layer scan.

### Evidence

For a synthetic Qwen-like layer order `linear_attention, linear_attention, full_attention, linear_attention`, the plan prints:

```text
path=linear_attention_prefix_only
linear_prefix_layers=2
full_layer_limit_native=False
first_unsupported_layer=2
first_unsupported_type=full_attention
```

This makes the remaining native compact prefill blocker narrower: full-model native prefill cannot be claimed until the first full-attention/compact grouped-MoE boundary after the linear prefix is implemented.

### Validation

```bash
python3 -m pytest tests/test_qwen35_resident_batch_layout.py -q
```

Result: `12 passed`.

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: `72 passed`.

### Notes

- The original pi-multiloop verify command still fails with `active hipENGINE parity TaskList not found` because it searches for completed task IDs that are no longer in the active compacted TaskList file. A robust count over the active #12/#15 file still prints `2`.

---

## 2026-05-15 — Capture real Qwen3.5/PARO native-prefill blocker artifact

### Scope

- Narrowed Task #15 with a reproducible config-only native-prefill planner.
- Added pure helper `qwen35_paro_native_prefill_plan(layer_types, layer_limit=...)` so planning does not require resident GPU state.
- Added `scripts/qwen35_native_prefill_plan.py` to load the Qwen3.5/PARO HF config and emit a compact blocked artifact.
- Updated Task #15 description with the exact blocker artifact and first unsupported layer.

### Artifact

```bash
python3 scripts/qwen35_native_prefill_plan.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-plan-blocked.json
```

Result: `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-plan-blocked.json`.

Key fields:

- `status=blocked`
- `performance_claim=false`
- `num_hidden_layers=40`
- `layer_type_counts_in_limit={"full_attention": 10, "linear_attention": 30}`
- `native_prefill_plan.path=linear_attention_prefix_only`
- `native_prefill_plan.linear_prefix_layers=3`
- `native_prefill_plan.first_unsupported_layer=3`
- `native_prefill_plan.first_unsupported_type=full_attention`

This is the current exact native compact prefill blocker: the port must implement compact/grouped MoE + full-attention/KV prefill at layer 3 before full-model native prefill or c>N throughput can be claimed.

### Validation

```bash
python3 -m pytest tests/test_qwen35_resident_batch_layout.py -q
```

Result: `13 passed`.

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0`.

### Notes

- The original pi-multiloop verify command remains stale and exits `active hipENGINE parity TaskList not found` because it requires completed task IDs no longer present in the compacted active TaskList file. A robust count over the active #12/#15 file prints `2`.
- No kernel port or throughput claim was made in this iteration.

---

## 2026-05-15 — Reject current native linear-prefix prefill correctness

### Scope

- Narrowed Task #15 again: before extending native prefill beyond layer 3, the existing native linear-prefix helper must first match serial c=1 semantics.
- Added `scripts/qwen35_native_prefill_correctness.py`, a correctness-only helper comparing:
  - serial token-by-token resident prefill + one decode token;
  - `prefill_linear_tokens_native()` + one decode token.
- The helper returns nonzero on mismatch and emits `rejected_correctness` JSON evidence.

### Artifact

```bash
python3 scripts/qwen35_native_prefill_correctness.py \
  --prompt-length 4 --max-layers 3 \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-prefill-rejected.json
```

Expected exit: `1` for the current mismatch.

Result artifact: `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-prefill-rejected.json`.

Key fields:

- `status=rejected_correctness`
- `performance_claim=false`
- `native_prefill_plan.full_layer_limit_native=true` for `max_layers=3`
- `finite_logits=true`
- `seed_match=false`: serial `95916` vs native `201383`
- `decode_match=false`: serial `158950` vs native `96022`
- `logit_abs_delta.seed=0.48061084747314453`
- `logit_abs_delta.decode=0.09302425384521484`

### Interpretation

The first full-model native-prefill blocker is still layer 3 `full_attention`, but the current native linear-prefix helper is not yet a correctness-preserving substitute for serial c=1 even when restricted to the first 3 linear-attention layers. Fixing/replacing that native prefix path is now the immediate Task #15 blocker before any compact/full-attention extension or throughput claim.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0`.

### Notes

- The original pi-multiloop verify command remains stale and exits `active hipENGINE parity TaskList not found`; robust active TaskList count over #12/#15 remains `2`.
- No performance claim was made; the new artifact is explicitly rejected correctness evidence.

---

## 2026-05-15 — Gate rejected native linear-prefix prefill behind diagnostic opt-in

### Scope

- Follow-up to the rejected correctness artifact for the current native linear-prefix prefill helper.
- Updated `Qwen35ParoResidentSession.prefill_linear_tokens_native()` to require `allow_rejected_correctness=True` before it can run the known-mismatching diagnostic path.
- Updated `scripts/qwen35_paro_bench.py` so `--native-prefill` now requires explicit `--allow-rejected-native-prefill`; otherwise it fails before building the resident session.
- Updated `scripts/qwen35_native_prefill_correctness.py` to pass the explicit opt-in because its purpose is to reproduce the rejected-correctness blocker artifact.
- Added unit coverage that the runtime helper refuses the rejected path by default before GPU allocation/work.
- Updated Task #15 description with the opt-in/gating status.

### Validation

```bash
python3 -m pytest tests/test_qwen35_resident_batch_layout.py -q
```

Result: `14 passed`.

```bash
python3 scripts/qwen35_paro_bench.py \
  --prompt-length 4 --decode-tokens 1 --warmup-decode-tokens 0 \
  --max-layers 3 --native-prefill --json /tmp/should_not_write.json
```

Expected result: exits `1` with `--native-prefill is currently rejected_correctness vs serial c=1; add --allow-rejected-native-prefill only for diagnostic blocker artifacts`.

```bash
python3 scripts/qwen35_native_prefill_correctness.py \
  --prompt-length 4 --max-layers 3 \
  --json /tmp/hipengine_native_prefix_rejected_current.json
```

Expected result: exits `1` and still reproduces the known rejected-correctness mismatch through the explicit diagnostic opt-in:
serial seed/decode `95916/158950`, native seed/decode `201383/96022`, finite logits true.

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0`.

### Notes

- This change preserves correctness/no-claim discipline; it does not fix native prefill yet.
- The original pi-multiloop verify command remains stale and exits `active hipENGINE parity TaskList not found`; robust active TaskList count over #12/#15 remains `2`.

---

## 2026-05-15 — Close PLAN-MOE2 profile loop on documented blocker

### Scope

- Completed Task #12 by the task's own blocker-exit criterion: continue the benchmark/profile loop until decode approaches PLAN-MOE2 parity **or** a clear blocker is documented with evidence.
- No throughput row was promoted.
- Task #15 remains open as the implementation blocker for native compact/c-aware prefill.

### Blocker evidence

- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-plan-blocked.json`
  - real Qwen3.5/PARO config: 40 layers, 30 `linear_attention`, 10 `full_attention`;
  - native-prefill plan is `linear_attention_prefix_only`;
  - first full-model native-prefill blocker is layer 3 `full_attention` after a 3-layer linear prefix.
- `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-prefill-rejected.json`
  - status `rejected_correctness`;
  - finite logits true but native linear-prefix helper mismatches serial c=1 at `max_layers=3`;
  - serial seed/decode `95916/158950`, native seed/decode `201383/96022`;
  - `performance_claim=false`.
- `c20e2d4 fix: gate qwen35 rejected native prefill`
  - the rejected helper now requires explicit diagnostic opt-in and normal benchmark use fails fast.

### Interpretation

Further profile/throughput iteration would violate the evidence policy until Task #15 fixes or replaces the native linear-prefix semantics and then extends native compact/full-attention prefill past layer 3.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0`.

### Notes

- The original pi-multiloop verify command remains stale and exits `active hipENGINE parity TaskList not found`; robust active TaskList count over the compacted active file is now `1` (#15 only).

---

## 2026-05-15 — Sweep native linear-prefix prefill mismatch to layer 1

### Scope

- Narrowed Task #15 with a layer-prefix sweep for the rejected native linear-prefix prefill helper.
- Extended `scripts/qwen35_native_prefill_correctness.py` with `--sweep-layer-prefixes N`.
- Emitted `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-sweep-rejected.json`.
- Updated Task #15 description with the first mismatching prefix.

### Artifact

```bash
python3 scripts/qwen35_native_prefill_correctness.py \
  --prompt-length 4 --sweep-layer-prefixes 3 \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-sweep-rejected.json
```

Expected exit: `1` for current rejected correctness.

Key fields:

- `status=rejected_correctness`
- `performance_claim=false`
- `first_mismatching_prefix=1`
- `cases[0].max_layers=1`, finite logits true, seed/decode mismatch:
  - serial seed/decode `627/356`
  - native seed/decode `308/1076`
- `cases[1].max_layers=2`, finite logits true, seed/decode mismatch:
  - serial seed/decode `627/84`
  - native seed/decode `36475/348`
- `cases[2].max_layers=3`, finite logits true, seed/decode mismatch:
  - serial seed/decode `95916/158950`
  - native seed/decode `201383/96022`

### Interpretation

The immediate native-prefill correctness blocker is in the first linear-attention native prefill layer path itself. It is not solely an interaction across the 3-layer linear prefix or the later layer-3 `full_attention` boundary.

### Validation

```bash
python3 scripts/qwen35_native_prefill_correctness.py \
  --prompt-length 4 --max-layers 3 \
  --json /tmp/hipengine_native_prefix_single_current.json
```

Result: exits `1` and preserves the previous single-prefix rejected-correctness payload shape.

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0`.

### Notes

- The original pi-multiloop verify command remains stale and exits `active hipENGINE parity TaskList not found`; robust active TaskList count remains `1` (#15 only).
- No performance claim was made; the new sweep artifact is explicitly rejected correctness evidence.

---

## 2026-05-15 — Isolate native prefix mismatch before MoE

### Scope

- Narrowed Task #15 beyond the layer-prefix sweep.
- Added `scripts/qwen35_native_prefill_stage_probe.py`, a correctness-only diagnostic that compares serial c=1 vs native prefill at the first layer's linear-attention out-projection for the final prompt token.
- Emitted `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-layer0-attn-rejected.json`.
- Updated Task #15 description with the stage-level evidence.

### Artifact

```bash
python3 scripts/qwen35_native_prefill_stage_probe.py \
  --prompt-length 4 \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-layer0-attn-rejected.json
```

Expected exit: `1` for current rejected correctness.

Key fields:

- `status=rejected_correctness`
- `performance_claim=false`
- `stage=layer0_linear_attention_out_proj_last_token`
- `diff.max_abs=0.30224609375`
- `diff.mean_abs=0.028543028980493546`
- `diff.rms_abs=0.03798133288876307`
- `diff.cosine=0.5423815250396729`
- `diff.serial_norm=0.9725803136825562`
- `diff.native_norm=2.0397138595581055`

### Interpretation

The native linear-prefix mismatch is already present before MoE and before later layer-prefix interactions: the first layer's native linear-attention out-projection for the last prompt token diverges significantly from the serial c=1 out-projection. The next implementation attempt should focus on the native linear-attention prefill conv/GDN/out-proj semantics, not compact MoE first.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0`.

### Notes

- The original pi-multiloop verify command remains stale and exits `active hipENGINE parity TaskList not found`; robust active TaskList count remains `1` (#15 only).
- No performance claim was made; the new stage probe artifact is explicitly rejected correctness evidence.

---

## 2026-05-15 — Bisect native linear-prefix divergence to conv prefill

### Scope

- Narrowed Task #15 inside the first layer's native linear-attention path.
- Reworked `scripts/qwen35_native_prefill_stage_probe.py` to compare serial c=1 vs native batched prefill at multiple layer0 stages:
  - `input_norm`
  - `qkv_rot`
  - `z_rot`
  - `qkv`
  - `z`
  - `ab`
  - `conv_out`
  - `recurrent_out`
  - `attention_out`
- Emitted `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-layer0-stage-bisect-rejected.json`.
- Updated Task #15 description with the first divergent stage.

### Artifact

```bash
python3 scripts/qwen35_native_prefill_stage_probe.py \
  --prompt-length 4 \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-layer0-stage-bisect-rejected.json
```

Expected exit: `1` for current rejected correctness.

Key fields:

- `status=rejected_correctness`
- `performance_claim=false`
- `first_divergent_stage=conv_out`
- Stages matching exactly (`max_abs=0.0`):
  - `input_norm`
  - `qkv_rot`
  - `z_rot`
  - `qkv`
  - `z`
  - `ab`
- First divergent stage:
  - `conv_out.max_abs=12.14988899230957`
  - `conv_out.mean_abs=0.0498395711183548`
  - `conv_out.rms_abs=0.35642884098821453`
  - `conv_out.cosine=0.9481420516967773`
- Downstream divergence:
  - `recurrent_out.max_abs=2.054323196411133`
  - `attention_out.max_abs=0.30224609375`

### Interpretation

The immediate native linear-prefix blocker is now localized to the native linear-attention prefill convolution/state semantics. Embedding, input RMSNorm, rotations, AWQ qkv/z projection, and a/b projection match serial c=1 exactly for the last prompt token.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0`.

### Notes

- User added four new TaskList items (#42-#45) during this loop. The robust active TaskList count is now `5`, but the Qwen3.5/PARO parity implementation blocker remains Task #15.
- The original pi-multiloop verify command remains stale and exits `active hipENGINE parity TaskList not found`.
- No performance claim was made; the new stage-bisect artifact is explicitly rejected correctness evidence.

---

## 2026-05-15 — Rename display branding to hipENGINE

### Scope

- Completed Task #42: user-facing project display name now uses `hipENGINE` while the import/package remains `hipengine`.
- Replaced tracked-file display text only; established all-caps environment variables such as `HIPENGINE_COMPILER_VERSION_FILE` and `HIPENGINE_QWEN35_LM_HEAD_THREADS` were intentionally left unchanged.
- Updated docs, benchmark rollups/artifacts, public docstrings/comments, and metadata text. No runtime identifiers, imports, package names, or kernel bodies changed.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0`.

Additional checks:

- Display-brand grep: no residual all-caps project-name matches outside intentionally preserved env-var identifiers.
- Robust active TaskList count after completing #42: `4` (#15, #43, #44, #45 remain open; #45 is blocked by #43).
- The original pi-multiloop verify command remains stale because it searches compacted-away completed task IDs; it still exits with the legacy all-caps-brand TaskList error.

### Notes

- No performance claim was made.
- No kernel port or plugin dispatch change was made, so no source-lineage or rocprof evidence is required for this branding-only iteration.

---

## 2026-05-15 — Fix native prefill qkv/z multi-token layout

### Scope

- Continued Task #15 after the layer0 stage bisect isolated the first native-prefix divergence at `conv_out`.
- Root cause found in the host runtime, not the conv kernel body: `gemv_awq_dual_pack8_transposed_*` writes row-major `[qkv,z]` per token, while native prefill conv/GDN consumes contiguous `[tokens,qkv]` and `[tokens,z]` streams.
- For `tokens == 1`, the existing dual projection is still used.
- For `tokens > 1`, `project_linear_attention_qkv_z_{bf16,fp16}` now runs two single transposed pack8 projections into the contiguous `scratch.qkv` and `scratch.z` regions.
- Updated the stage probe to compare the lowp gated recurrent tensor that actually feeds out_proj (`gated_recurrent`) instead of comparing serial gated output with native raw recurrent state.
- Emitted new artifact:
  - `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-layer0-gated-recurrent-rejected.json`

### Evidence

```bash
python3 scripts/qwen35_native_prefill_stage_probe.py \
  --prompt-length 4 \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-layer0-gated-recurrent-rejected.json
```

Expected exit: `1` for current rejected correctness.

Key fields:

- `status=rejected_correctness`
- `performance_claim=false`
- `first_divergent_stage=gated_recurrent`
- Stages matching exactly or near-exactly after the layout fix:
  - `input_norm.max_abs=0.0`
  - `qkv_rot.max_abs=0.0`
  - `z_rot.max_abs=0.0`
  - `qkv.max_abs=0.0`
  - `z.max_abs=0.0`
  - `ab.max_abs=0.0`
  - `conv_out.max_abs=2.9802322387695312e-08`
- Remaining layer0 pre-MoE blocker:
  - `gated_recurrent.max_abs=0.05126953125`
  - `gated_recurrent.rms_abs=0.00196707713575249`
  - `gated_recurrent.cosine=0.9983887672424316`
  - downstream `attention_out.max_abs=0.03759765625`

Additional check:

```bash
python3 scripts/qwen35_native_prefill_correctness.py \
  --prompt-length 4 \
  --max-layers 1 \
  --json /tmp/hipengine-native-prefill-correctness-layer1-iter28.json
```

Result: still rejected correctness after the conv layout fix (`serial.seed=627`, `native.seed=128440`; finite logits true). No throughput claim was made.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0`.

### Notes

- The original pi-multiloop verify command remains stale and exits the legacy TaskList-not-found selector error; robust active TaskList count remains `4` (#15, #43, #44, #45).
- This was not a kernel port and did not change kernel bodies, so no source-lineage or rocprof evidence is required.
- Next Task #15 target: native GDN/recurrent prefill numerical/state parity before extending to compact/full-attention prefill.

---

## 2026-05-15 — Fix native prefill a/b multi-token layout

### Scope

- Continued Task #15 after qkv/z and conv/GDN attention-side parity narrowed the native prefill blocker.
- Found the same layout class in `in_proj_a`/`in_proj_b`: `dense_dual_gemv_out_*` writes row-major `[a,b]` per token, while native GDN prefill consumes contiguous `[tokens,a]` and `[tokens,b]` streams.
- For `tokens == 1`, the existing dual dense projection is still used.
- For `tokens > 1`, `project_linear_attention_ab_{bf16,fp16}` now runs two single dense projections into contiguous `scratch.a` and `scratch.b`.
- Updated `scripts/qwen35_native_prefill_stage_probe.py` so the `ab` stage compares the logical concatenation of the `a` and `b` tensor views instead of the raw `scratch.ab` storage row.
- Emitted artifacts:
  - `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-layer0-attention-accepted.json`
  - `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-layer0-decode-rejected.json`

### Evidence

```bash
python3 scripts/qwen35_native_prefill_stage_probe.py \
  --prompt-length 4 \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-layer0-attention-accepted.json
```

Result: accepted stage parity, no throughput claim.

Key fields:

- `status=accepted`
- `performance_claim=false`
- `first_divergent_stage=null`
- `ab.max_abs=0.0`
- `conv_out.max_abs=2.9802322387695312e-08`
- `gated_recurrent.max_abs=3.814697265625e-06`
- `attention_out.max_abs=1.52587890625e-05`

Full layer0 native-prefix correctness is still rejected after attention parity:

```bash
python3 scripts/qwen35_native_prefill_correctness.py \
  --prompt-length 4 \
  --max-layers 1 \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-layer0-decode-rejected.json
```

Result: expected exit `1` / `status=rejected_correctness`:

- `seed_match=true` (`serial.seed=627`, `native.seed=627`)
- `decode_match=false` (`serial.decode=356`, `native.decode=627`)
- `finite_logits=true`
- `logit_abs_delta.seed=0.00020170211791992188`
- `logit_abs_delta.decode=0.2938823699951172`

Interpretation: the layer0 linear-attention native prefill path through out-proj is now within tolerance; the remaining layer0 native-prefix blocker is after attention, in the native batched post-attention/MoE path or hidden handoff used by the following decode step.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0`.

### Notes

- The original pi-multiloop verify command remains stale and exits the legacy TaskList-not-found selector error; robust active TaskList count remains `4` (#15, #43, #44, #45).
- This was not a kernel port and did not change kernel bodies, so no source-lineage or rocprof evidence is required.
- Next Task #15 target: native batched post-attention/MoE output parity for layer0, then layer-prefix >1 native prefill.

---

## 2026-05-15 — Accept native linear-prefix prefill after decode-scratch restore

### Scope

- Continued Task #15 after layer0 linear-attention through out-proj matched serial c=1.
- Found the post-prefill decode mismatch source: native prefill enlarged the named per-layer workspace scratch tensors to `tokens=4` and left them installed as the decode scratch.
- The next c=1 decode step then used dual qkv/z and a/b projection kernels with multi-token scratch views:
  - `scratch.z` pointed after `tokens * qkv_width`, not after the first row's qkv segment.
  - `scratch.b` pointed after `tokens * num_value_heads`, not after the first row's `a` segment.
  - Result before the fix: decode qkv matched, but decode `z`/`ab` read stale offsets and the next token mismatched.
- Fixed by restoring token-1 decode scratch after native prefill copies the last hidden row back to the resident slot.
- Removed the stale rejected-correctness requirement for `prefill_linear_tokens_native()` and updated the benchmark harness wording: the helper is accepted for all-linear prefixes but remains not PLAN-MOE2/full-model prefill comparable.

### Evidence

```bash
python3 scripts/qwen35_native_prefill_correctness.py \
  --prompt-length 4 \
  --sweep-layer-prefixes 3 \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-scratch-restore-sweep.json
```

Result: accepted correctness sweep, no throughput claim.

Key fields:

- `status=accepted`
- `performance_claim=false`
- `first_mismatching_prefix=null`
- Prefix 1: `seed_match=true`, `decode_match=true`, `decode logit delta=0.0`
- Prefix 2: `seed_match=true`, `decode_match=true`, `decode logit delta=0.004206657409667969`
- Prefix 3: `seed_match=true`, `decode_match=true`, `decode logit delta=0.0049800872802734375`

Smoke for the ungated accepted helper:

```bash
python3 scripts/qwen35_paro_bench.py \
  --native-prefill \
  --prompt-length 4 \
  --decode-tokens 0 \
  --warmup-decode-tokens 0 \
  --max-layers 3 \
  --json /tmp/hipengine-native-prefill-bench-smoke.json
```

Result: exits `0`, `native_batched_prefill=true`, `allow_rejected_native_prefill=false`, `prefill_comparable_to_plan_moe2=false`. The JSON includes timing fields as a smoke side effect only; no performance claim is made.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0`.

### Notes

- The original pi-multiloop verify command remains stale and exits the legacy TaskList-not-found selector error; robust active TaskList count remains `4` (#15, #43, #44, #45).
- This was a Python/runtime scratch lifetime fix, not a kernel body port; no source-lineage or rocprof evidence required.
- Next Task #15 target: extend native prefill past the all-linear prefix boundary at layer 3 (`full_attention`) and then return to compact/grouped MoE prefill and c>N correctness/performance.

---

## 2026-05-15 — Narrow native prefill blocker to full-attention boundary components

### Scope

- Continued Task #15 after native linear-prefix prefill was accepted for prefixes 1..3.
- Added `scripts/qwen35_native_prefill_boundary.py`, a blocker/planning diagnostic that inspects the real Qwen3.5/PARO layer map without collecting timings.
- Added `tests/test_qwen35_native_prefill_boundary.py` for the pure boundary payload helper.
- Updated Task #15 with an explicit component list for the next unsupported layer boundary.

### Evidence

```bash
python3 scripts/qwen35_native_prefill_boundary.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-attn-boundary-blocked.json
```

Expected result: exits `1` because this is a blocked diagnostic. Artifact key fields:

- `status=blocked`
- `performance_claim=false`
- `layer_type_counts={"linear_attention": 30, "full_attention": 10}`
- `accepted_linear_prefix_layers=3`
- `first_unsupported_layer=3`
- `first_unsupported_type="full_attention"`
- component blockers:
  - `full_attention_prefill_orchestrator`: `run_full_attention_moe_c1_layer_fp16` has a `tokens != 1` guard.
  - `full_attention_qkv_projection_layout`: `project_full_attention_qkv_fp16` has a `tokens != 1` guard and needs batched q/k/v projection layout.
  - `full_attention_rope_prepare_positions`: `prepare_full_attention_qkv_fp16` has a `tokens != 1`/single-position guard and needs per-token positions.
  - `full_attention_prefill_kv_append`: batch KV writer wrapper exists, but resident native prefill does not wire prompt row positions/live counts.
  - `full_attention_causal_prefill_attention`: decode context attention handles one query; multi-query causal prefill attention is not wired.

Interpretation: the native linear-prefix correctness blocker is closed; the remaining Task #15 implementation blocker is no longer generic "native prefill mismatch" but specifically layer-3 full-attention prefill/KV/causal attention wiring (or an explicitly labelled serial c=1 fallback), then compact/grouped MoE prefill and c>N work.

### Validation

```bash
python3 -m compileall -q scripts/qwen35_native_prefill_boundary.py scripts/qwen35_paro_bench.py tests/test_qwen35_native_prefill_boundary.py && \
  python3 -m pytest tests/test_qwen35_native_prefill_boundary.py tests/test_qwen35_resident_batch_layout.py -q
```

Result: `17 passed`.

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`75 passed`).

### Notes

- The original pi-multiloop verify command remains stale and exits the legacy TaskList-not-found selector error; robust active TaskList count remains `4` (#15, #43, #44, #45).
- This iteration added a diagnostic/planning helper only; no kernel body was ported, so no source-lineage or rocprof evidence is required.
- Next Task #15 target: add a layer-3 full-attention prefill row diagnostic and then wire either batched full-attention prefill or a clearly-labelled serial c=1 fallback before compact/grouped MoE work.

---

## 2026-05-15 — Probe serial layer-3 full-attention bridge after native prefix

### Scope

- Continued Task #15 after the full-attention boundary artifact identified layer 3 as the first unsupported native prefill layer.
- Added `scripts/qwen35_native_prefill_fullattn_stage_probe.py`.
  - Serial side: runs layers 0..2 token-by-token, then captures layer-3 full-attention stages for the last prompt token.
  - Native-prefix side: runs accepted batched native linear-prefix prefill for layers 0..2, then feeds each prefix row through serial c=1 layer-3 full attention and captures the last row.
- Added `tests/test_qwen35_native_prefill_fullattn_stage_probe.py` for the pure diff payload helper.
- Emitted `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-layer3-fullattn-stage-accepted.json`.

### Evidence

```bash
python3 scripts/qwen35_native_prefill_fullattn_stage_probe.py \
  --prompt-length 4 \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-layer3-fullattn-stage-accepted.json
```

Result: accepted diagnostic stage probe, no throughput claim.

Key fields:

- `status=accepted`
- `performance_claim=false`
- `linear_prefix_layers=3`
- `full_attention_layer=3`
- `atol=0.02`
- `first_divergent_stage=null`
- `prefix_hidden.max_abs=0.000244140625`
- largest intermediate observed: `v_rot.max_abs=0.015625`, `v_rot.rms_abs=0.0014288396602966233`, `v_rot.cosine=0.9999997615814209`
- `query.max_abs=0.006503105163574219`
- `attn_out.max_abs=0.004729747772216797`
- final `moe_out.max_abs=0.000244140625`, `moe_out.rms_abs=3.427448151303778e-05`, `moe_out.cosine=0.9999997615814209`

Interpretation: a labelled serial c=1 layer-3 full-attention bridge can consume accepted native linear-prefix rows for the repeated-token prompt without meaningful drift. This does not solve true batched full-attention prefill, but it provides a correctness-safe fallback path to wire before native KV row-position/causal-attention kernels.

### Validation

```bash
python3 -m compileall -q scripts/qwen35_native_prefill_fullattn_stage_probe.py tests/test_qwen35_native_prefill_fullattn_stage_probe.py && \
  python3 -m pytest tests/test_qwen35_native_prefill_fullattn_stage_probe.py -q
```

Result: `1 passed`.

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`75 passed`).

### Notes

- The original pi-multiloop verify command remains stale and exits the legacy TaskList-not-found selector error; robust active TaskList count remains `4` (#15, #43, #44, #45).
- This iteration added a diagnostic helper only; no kernel body was ported, so no source-lineage or rocprof evidence is required.
- Next Task #15 target: wire a clearly-labelled native-linear-prefix + serial full-attention fallback path, then replace the fallback with true batched full-attention prefill/KV row-position/causal attention work.

---

## 2026-05-15 — Wire native-prefix + serial full-attention fallback

### Scope

- Continued Task #15 after the layer-3 full-attention stage probe showed a serial c=1 full-attention bridge can consume accepted native linear-prefix rows.
- Updated `Qwen35ParoResidentSession.prefill_linear_tokens_native()`:
  - runs the accepted native batched linear-prefix layers first;
  - when the configured layer limit extends past the linear prefix, replays the remaining suffix layers token-by-token through the resident c=1 path;
  - keeps the path explicitly labelled as a fallback rather than PLAN-MOE2/native compact prefill.
- Added benchmark JSON labelling:
  - `native_prefill_execution="native_linear_prefix_serial_suffix_fallback"` for mixed prefixes;
  - `native_prefill_plan={...}`;
  - `prefill_comparable_to_plan_moe2=false` remains unchanged.
- Updated the resident batch-layout test that previously expected non-linear native-prefill prefixes to reject.

### Evidence

Correctness gate for the first mixed prefix:

```bash
python3 scripts/qwen35_native_prefill_correctness.py \
  --prompt-length 4 \
  --max-layers 4 \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-serial-fullattn-layer4-accepted.json
```

Result: accepted, no throughput claim.

- `status=accepted`
- `performance_claim=false`
- `max_layers=4`
- `native_prefill_plan.path="linear_attention_prefix_only"`
- `native_prefill_plan.linear_prefix_layers=3`
- `native_prefill_plan.first_unsupported_layer=3`
- `seed_match=true`, `decode_match=true`, `finite_logits=true`
- `serial.seed=232708`, `native.seed=232708`
- `serial.decode=169222`, `native.decode=169222`
- `logit_abs_delta.seed=0.0051021575927734375`
- `logit_abs_delta.decode=0.0009202957153320312`

Label smoke:

```bash
python3 scripts/qwen35_paro_bench.py \
  --native-prefill \
  --prompt-length 4 \
  --decode-tokens 0 \
  --warmup-decode-tokens 0 \
  --max-layers 4 \
  --json /tmp/hipengine-native-prefix-serial-fullattn-bench-smoke.json
```

Result: exits `0`; JSON includes `native_prefill_execution="native_linear_prefix_serial_suffix_fallback"`, `native_batched_prefill=true`, and `prefill_comparable_to_plan_moe2=false`. Timing fields are smoke side effects only and are not retained as a performance claim.

### Validation

```bash
python3 -m compileall -q hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_paro_bench.py scripts/qwen35_native_prefill_correctness.py tests/test_qwen35_resident_batch_layout.py && \
  python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_generation_qwen35_paro.py -q
```

Result: `18 passed`.

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`75 passed`).

### Notes

- The original pi-multiloop verify command remains stale and exits the legacy TaskList-not-found selector error; robust active TaskList count remains `4` (#15, #43, #44, #45).
- This is not a performance win and is not PLAN-MOE2-comparable; it is a correctness-safe bridge until full-attention prefill/KV row-position/causal-attention kernels are wired.
- No kernel bodies were changed, so source-lineage and rocprof evidence are not required for this iteration.

---

## 2026-05-15 — Task #43 c=1/2/4/8 scheduler-serial benchmark attempts

### Scope

- Added `scripts/qwen35_batch_serial_bench.py`, a schema-2 diagnostic benchmark harness for the current Qwen3.5/PARO resident scheduler serial c>N bridge.
- The harness records exact benchmark command, W7900/gfx1100 hardware/software context, scheduler metadata, `batch_execution` path/blockers, seed/generated tokens, finite-logit correctness smoke, timing samples, and a blocked/non-retained decision.
- Added unit coverage for prompt slicing and sample-stat helpers in `tests/test_generation_batch_scheduler.py`.
- Completed Task #43 with c=1/c=2/c=4/c=8 artifacts. These are diagnostic blocked benchmark attempts, not retained throughput rows.

### Commands and artifacts

All runs used full `max_layers=40` but a reduced diagnostic workload (`prompt_length=8`, `decode_tokens=1`, no warmup), so they do not satisfy the retained c=N 512/128 benchmark protocol.

```bash
python3 scripts/qwen35_batch_serial_bench.py --batch-size 1 --prompt-length 8 --decode-tokens 1 --warmup-decode-tokens 0 --max-layers 40 --json benchmarks/results/2026-05-15-hipengine-qwen35-c1-scheduler-serial-bench-blocked.json
python3 scripts/qwen35_batch_serial_bench.py --batch-size 2 --prompt-length 8 --decode-tokens 1 --warmup-decode-tokens 0 --max-layers 40 --json benchmarks/results/2026-05-15-hipengine-qwen35-c2-scheduler-serial-bench-blocked.json
python3 scripts/qwen35_batch_serial_bench.py --batch-size 4 --prompt-length 8 --decode-tokens 1 --warmup-decode-tokens 0 --max-layers 40 --json benchmarks/results/2026-05-15-hipengine-qwen35-c4-scheduler-serial-bench-blocked.json
python3 scripts/qwen35_batch_serial_bench.py --batch-size 8 --prompt-length 8 --decode-tokens 1 --warmup-decode-tokens 0 --max-layers 40 --json benchmarks/results/2026-05-15-hipengine-qwen35-c8-scheduler-serial-bench-blocked.json
```

### Results

All artifacts:

- `status=blocked`
- `performance_claim=false`
- `decision.accepted=false`
- `correctness.finite_logits=true`
- `workload.max_layers=40`
- `execution.batch_execution.path="scheduler_serial_slot_bridge"`
- `execution.batch_execution.row_execution="serial_c1_layer_path"`
- `execution.batch_execution.throughput_claim_eligible=false`
- blocked reason includes current serial bridge plus reduced diagnostic workload shape.

Diagnostic timing snapshot (not retained as performance claims):

| c | Prefill tok/s | Aggregate decode tok/s | Per-request decode tok/s |
| ---: | ---: | ---: | ---: |
| 1 | 90.464313 | 106.765109 | 106.765109 |
| 2 | 103.566729 | 107.149491 | 53.574746 |
| 4 | 111.225881 | 108.433765 | 27.108441 |
| 8 | 115.080077 | 108.904474 | 13.613059 |

Interpretation: aggregate decode stays ~107–109 tok/s as c increases because the path executes rows serially; per-request throughput falls roughly as expected for a serial bridge. This is useful blocker evidence for Task #15, not a c>N throughput win.

### Validation

```bash
python3 -m compileall -q scripts/qwen35_batch_serial_bench.py tests/test_generation_batch_scheduler.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py -q
```

Result: `5 passed`.

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`76 passed`).

### Notes

- Original pi-multiloop verify selector remains stale and exits with the legacy TaskList-not-found error; robust active TaskList count is now `3` (#15, #44, #45) after completing #43.
- Task #45 is now unblocked: summarize these c=1/2/4/8 results as blocked/diagnostic rows in `benchmarks/README.md`/`CHANGELOG.md`, not in current-fastest accepted rows.

---

## 2026-05-15 — Task #45 benchmark rollup update for c=N diagnostics

### Scope

- Updated `benchmarks/README.md` after Task #43 artifacts:
  - kept `Current fastest hipENGINE rows` empty;
  - added a clearly marked `Blocked / diagnostic benchmark attempts` table;
  - linked the c=1/c=2/c=4/c=8 scheduler-serial blocked artifacts;
  - included workload shape, correctness/status, diagnostic timing, memory availability, and blocker notes.
- Updated `benchmarks/CHANGELOG.md` with a 2026-05-15 blocked-diagnostic one-liner.
- Completed Task #45.

### Result

No retained throughput row was added. The rollup explicitly states these rows are not current-fastest results and are not performance claims because:

- artifact `status=blocked`;
- `performance_claim=false`;
- `batch_execution.throughput_claim_eligible=false`;
- path is `scheduler_serial_slot_bridge` / `serial_c1_layer_path`;
- workload is reduced prompt8/decode1 rather than the retained c=N 512/128 protocol.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`76 passed`).

Robust active TaskList count after completing #45: `2` (#15 and #44).

---

## 2026-05-15 — Task #44 DFlash/DDTree blocker artifact

### Scope

- Inspected the current speculative scaffolding and Qwen3.5/PARO resident batch evidence for DFlash/DDTree.
- Added `scripts/qwen35_dflash_ddtree_blocker.py` to emit a compact blocker artifact.
- Added `benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json`.
- Added unit coverage in `tests/test_speculative_interfaces.py` for the blocker payload.
- Updated `docs/DFLASH.md` with the current hipENGINE status and artifact link.
- Completed Task #44 via its documented blocker path; no speculative throughput claim was made.

### Evidence

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Result: exits `0`; artifact has:

- `status=blocked`
- `performance_claim=false`
- `specdec_enabled=false`
- `implementation_status.native_target_verify_ready=false`
- interfaces present: `DraftBatch`, `AcceptResult`, `DraftModel`, `Verifier`, `FixedPagedKVPolicy`, `KVTransaction`, verify-tree graph shape key
- resident API status:
  - `step_batch_serial=true`
  - `batch_execution_metadata=true`
  - `native_target_verify_batch=false`
  - `speculative_verify_batch=false`
  - `commit_verified_state=false`
- c=8 evidence from `benchmarks/results/2026-05-15-hipengine-qwen35-c8-scheduler-serial-bench-blocked.json`:
  - `batch_execution.path="scheduler_serial_slot_bridge"`
  - `batch_execution.row_execution="serial_c1_layer_path"`
  - `batch_execution.throughput_claim_eligible=false`
- native prefill evidence:
  - `linear_prefix_layers=3`
  - `first_unsupported_layer=3`
  - `first_unsupported_type="full_attention"`

Interpretation: DFlash/DDTree policy work cannot become a valid throughput path until Task #15 lands a native compact/c-aware target verifier with selectable per-row target state and GPU accept summaries. The host-side interface/KV transaction scaffolding is present, but target verify/commit is the blocker.

### Validation

```bash
python3 -m compileall -q scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_kvcache_policy.py tests/test_dispatch_batch.py -q
```

Result: `14 passed`.

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`76 passed`).

Robust active TaskList count after completing #44: `1` (#15).

---

## 2026-05-15 — TargetVerifyBatch metadata scaffold for Task #15

### Scope

- Added torch-free speculative target-verification metadata:
  - `hipengine.speculative.TargetVerifyBatch`
  - exported via `hipengine/speculative/__init__.py`
- `TargetVerifyBatch.from_draft(...)` materializes the native verifier row layout from a `DraftBatch` plus one committed root token/position per request:
  - root rows first, one per request;
  - candidate rows after roots;
  - `parent_rows` references roots or earlier candidate rows;
  - `row_to_request`, `draft_depths`, and `active_mask` cover every root+candidate row.
- Updated the DFlash/DDTree blocker helper and artifact so the scaffolding is now recorded as present while runtime verifier/commit APIs remain missing.
- Updated `docs/DFLASH.md` current-status text to mention `TargetVerifyBatch` as API scaffolding.
- Updated Task #15 with the narrowed remaining blocker.

### Evidence

Target-verify metadata tests:

```bash
python3 -m compileall -q hipengine/speculative scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_kvcache_policy.py tests/test_dispatch_batch.py -q
```

Result: `16 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records `implementation_status.interfaces_present.target_verify_batch="TargetVerifyBatch"`, while preserving:

- `status=blocked`
- `performance_claim=false`
- `implementation_status.native_target_verify_ready=false`
- `resident_api.speculative_verify_batch=false`
- c=8 evidence still points at `scheduler_serial_slot_bridge` / `throughput_claim_eligible=false`.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`76 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows the blocker but does not complete native compact/c-aware execution.

---

## 2026-05-15 — Full-stack native-linear-prefix bridge correctness

### Scope

- Ran the existing native prefill correctness helper against the full 40-layer Qwen3.5/PARO stack.
- This validates the current bridge only:
  - native batched linear-prefix layers (`linear_prefix_layers=3`);
  - serial c=1 suffix replay for the rest of the layer stack;
  - no PLAN-MOE2/compact/full-attention throughput claim.
- Updated Task #15 with the full-stack bridge evidence.

### Evidence

```bash
python3 scripts/qwen35_native_prefill_correctness.py \
  --prompt-length 4 \
  --max-layers 40 \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefix-serial-suffix-full40-accepted.json
```

Result: accepted, no performance claim.

- `status=accepted`
- `performance_claim=false`
- `max_layers=40`
- `native_prefill_plan.path="linear_attention_prefix_only"`
- `native_prefill_plan.linear_prefix_layers=3`
- `native_prefill_plan.first_unsupported_layer=3`
- `native_prefill_plan.first_unsupported_type="full_attention"`
- `seed_match=true`
- `decode_match=true`
- `finite_logits=true`
- serial/native seed token: `9707`
- serial/native decode token: `9707`
- `logit_abs_delta.seed=0.0036106109619140625`
- `logit_abs_delta.decode=0.009748458862304688`

Interpretation: the labelled native-linear-prefix + serial suffix bridge is correctness-safe on this prompt-length-4 full-stack smoke. Task #15 remains open because the retained target is true compact/native c>N prefill and target verification, not serial suffix replay.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`76 passed`).

Robust active TaskList count remains `1` (#15). The legacy loop verify selector is still stale and exits with the old TaskList-not-found selector error.

---

## 2026-05-15 — TargetVerifyBatch KV transaction row accounting

### Scope

- Narrowed Task #15's speculative/native verifier blocker by connecting `TargetVerifyBatch` to KV transaction bookkeeping.
- Updated `FixedPagedKVPolicy.begin_transaction(...)`:
  - when passed a `TargetVerifyBatch`, transaction `draft_rows` now counts `candidate_rows` only;
  - committed root rows are excluded from the speculative KV journal;
  - role falls back to `mode` for target-verify batches.
- Added tests that a root+candidate target verify batch with 5 total rows and 3 candidates creates a KV transaction with `draft_rows=3`, `role="verify_tree"`.
- Updated `scripts/qwen35_dflash_ddtree_blocker.py`, its artifact, `docs/DFLASH.md`, and Task #15 with this narrower status.

### Evidence

```bash
python3 -m compileall -q hipengine/kvcache hipengine/speculative scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_kvcache_policy.py tests/test_dispatch_batch.py -q
```

Result: `16 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.kv_transaction_target_verify.target_verify_rows=5`
- `implementation_status.kv_transaction_target_verify.candidate_rows=3`
- `implementation_status.kv_transaction_target_verify.transaction_draft_rows=3`
- `implementation_status.kv_transaction_target_verify.root_rows_excluded_from_journal=true`
- `implementation_status.kv_transaction_target_verify.role="verify_tree"`
- still `status=blocked`, `performance_claim=false`, and `native_target_verify_ready=false`.

Interpretation: host-side transaction metadata now matches the native verifier ABI: roots are committed context, candidates are speculative journal rows. Device-side selectable state/KV commit remains unimplemented and is still part of Task #15.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`76 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows the transaction/commit blocker but does not complete native compact/c-aware execution.

---

## 2026-05-15 — TargetVerifyBatch accepted-count budget guard

### Scope

- Tightened the host-side transaction/commit scaffold for Task #15's eventual native verifier path.
- Added `KVTransaction.candidate_counts` and populate it from `TargetVerifyBatch`/row maps in `FixedPagedKVPolicy.begin_transaction(...)`.
- `FixedPagedKVPolicy.commit(...)` now rejects accepted counts that exceed the verified candidate budget for each request.
- Updated speculative/KV tests, DFlash blocker helper/artifact, `docs/DFLASH.md`, and Task #15.

### Evidence

```bash
python3 -m compileall -q hipengine/kvcache hipengine/speculative scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py tests/test_kvcache_policy.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_kvcache_policy.py tests/test_dispatch_batch.py -q
```

Result: `16 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.kv_transaction_target_verify.target_verify_rows=5`
- `implementation_status.kv_transaction_target_verify.candidate_rows=3`
- `implementation_status.kv_transaction_target_verify.candidate_counts=[2,1]`
- `implementation_status.kv_transaction_target_verify.transaction_draft_rows=3`
- `implementation_status.kv_transaction_target_verify.root_rows_excluded_from_journal=true`
- still `status=blocked`, `performance_claim=false`, and `native_target_verify_ready=false`.

Interpretation: the host transaction layer now has enough per-request candidate budget metadata to prevent committing more speculative tokens than a native target verifier actually verified. Device-side state/KV commit and resident target verification remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`76 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows the host transaction guard only; no performance claim or kernel change was made.

---

## 2026-05-15 — TargetVerifyBatch commit-row selection scaffold

### Scope

- Added `TargetCommitSelection` and `TargetVerifyBatch.select_commit_rows(...)`.
- The selection maps per-request accepted counts to the target row whose state/KV would be committed:
  - accepted count `0` selects the request root row;
  - non-zero accepted counts select the unique active candidate row at that request/depth;
  - ambiguous tree depths require explicit `selected_candidate_rows` from the future GPU accept summary.
- Updated blocker artifact/status and Task #15 with selected row evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_kvcache_policy.py tests/test_dispatch_batch.py -q
```

Result: `18 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.kv_transaction_target_verify.target_verify_rows=5`
- `implementation_status.kv_transaction_target_verify.candidate_counts=[2,1]`
- `implementation_status.kv_transaction_target_verify.commit_selection_rows=[3,4]`
- `implementation_status.kv_transaction_target_verify.commit_selection_positions=[7,4]`
- still `status=blocked`, `performance_claim=false`, and `native_target_verify_ready=false`.

Interpretation: host metadata can now name which verified target row would feed selectable state/KV commit for each request. The actual device-side copy/select of linear-attention state and full-attention K/V rows remains unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`76 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows the host selectable-state metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — TargetVerifyBatch graph shape key scaffold

### Scope

- Added graph/replay bucket key derivation for `TargetVerifyBatch`.
- `TargetVerifyBatch.tree_shape` now encodes candidate parent topology with non-negative entries for `BatchShapeKey` compatibility:
  - `0` means parent is the committed request root row;
  - `N+1` means parent is candidate index `N`.
- `TargetVerifyBatch.shape_key(active_batch, ...)` delegates to the existing batch shape-key discipline with verify mode, active mask, context bucket, top-k/expert fields, replay steps, draft depth, and tree topology.
- Updated blocker artifact/status and Task #15 with shape-key evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_kvcache_policy.py tests/test_dispatch_batch.py -q
```

Result: `19 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records `implementation_status.kv_transaction_target_verify.shape_key`:

- `mode="verify_tree"`
- `active_c=2`
- `context_bucket=8`
- `active_mask=[true,true]`
- `top_k=8`
- `experts_per_token=8`
- `replay_steps=1`
- `draft_depth=2`
- `tree_shape=[0,1,0]`

It still records `status=blocked`, `performance_claim=false`, and `native_target_verify_ready=false`.

Interpretation: host metadata can now derive the graph/replay cache bucket a native target verifier should use for fixed verify shapes. Actual resident graph capture/replay and device kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`76 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows graph-bucket metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — TargetVerifyBatch candidate WorkItem projection

### Scope

- Added `TargetVerifyBatch.to_work_item()` for scheduler/kernel routing metadata.
- The projection keeps committed root rows out of scheduler candidate rows:
  - `WorkItem.row_to_request` contains candidate rows only;
  - `WorkItem.token_rows` contains candidate token rows only;
  - `WorkItem.draft_depth` and `WorkItem.tree_parents` carry the target verify topology.
- Updated blocker artifact/status and Task #15 with WorkItem evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_kvcache_policy.py tests/test_dispatch_batch.py -q
```

Result: `20 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records `implementation_status.kv_transaction_target_verify.work_item`:

- `kind="verify_tree"`
- `request_ids=[1,2]`
- `row_to_request=[1,1,2]`
- `token_rows=[[10],[11],[20]]`
- `draft_depth=2`
- `tree_parents=[0,1,0]`

It still records `status=blocked`, `performance_claim=false`, and `native_target_verify_ready=false`.

Interpretation: host metadata can now project a target verify batch to the scheduler work item that future native verifier kernels should consume, while roots remain committed context metadata. Resident target verification and device-side state/KV commit remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`76 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler work metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — TargetAcceptSummary host accept/commit bridge

### Scope

- Added `hipengine.speculative.TargetAcceptSummary` as a torch-free host/device ABI scaffold tying verifier acceptance to target rows selected for commit.
- `TargetAcceptSummary.from_accept_result(target, result, ...)` now validates that `AcceptResult.accepted_tokens` exactly matches the selected `TargetVerifyBatch` parent path, handles ambiguous tree depths via explicit selected rows, and records per-request commit rows/tokens/positions plus `full_accept` flags.
- Updated the DFlash blocker helper/artifact and Task #15 with accept-summary evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_kvcache_policy.py tests/test_dispatch_batch.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records `implementation_status.interfaces_present.target_accept_summary="TargetAcceptSummary"` and `implementation_status.kv_transaction_target_verify.accept_summary`:

- `accepted_counts=[2,1]`
- `accepted_tokens=[[10,11],[20]]`
- `commit_rows=[3,4]`
- `commit_tokens=[11,20]`
- `commit_positions=[7,4]`
- `full_accept=[true,true]`

It still records `status=blocked`, `performance_claim=false`, and `native_target_verify_ready=false`.

Interpretation: host metadata can now validate the accepted path that a future GPU accept summary must produce and map it to the target state/KV row that would be committed. Resident target verification, GPU accept-summary buffers, and device-side state/KV commit remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`76 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows accept/commit metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — TargetCommitPlan transaction binding

### Scope

- Added `hipengine.speculative.TargetCommitPlan` as a torch-free host ABI scaffold that binds a validated `TargetAcceptSummary` to a KV transaction id and candidate budget.
- `TargetCommitPlan.from_summary(summary, transaction)` now validates transaction request ids, role, committed/rolled-back state, and accepted counts against `candidate_counts`, then exposes `kv_accept_counts` for the existing `FixedPagedKVPolicy.commit(...)` call.
- Updated the DFlash blocker helper/artifact and Task #15 with commit-plan evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_kvcache_policy.py tests/test_dispatch_batch.py -q
```

Result: `22 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records `implementation_status.interfaces_present.target_commit_plan="TargetCommitPlan"` and `implementation_status.kv_transaction_target_verify.commit_plan`:

- `transaction_id=0`
- `accepted_counts=[2,1]`
- `commit_rows=[3,4]`
- `commit_positions=[7,4]`
- `candidate_counts=[2,1]`
- `mode="verify_tree"`

It still records `status=blocked`, `performance_claim=false`, and `native_target_verify_ready=false`.

Interpretation: host metadata can now carry a validated transaction-scoped commit contract from verifier acceptance to KV commit bookkeeping. Resident target verification, GPU accept-summary buffers, and device-side state/KV commit remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`76 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows transaction-scoped commit metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — TargetVerifyBuffers device-buffer ABI scaffold

### Scope

- Added `hipengine.speculative.TargetVerifyBuffers` as a torch-free device-buffer ABI descriptor for native target verification replay.
- `TargetVerifyBuffers.for_batch(target, ...)` validates row-buffer shapes for token ids, positions, parent rows, draft depths, row-to-request ids, active masks, and target top1 outputs, plus per-request summary output shapes for accepted counts and commit rows/tokens/positions.
- Updated the DFlash blocker helper/artifact and Task #15 with device-buffer evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_kvcache_policy.py tests/test_dispatch_batch.py -q
```

Result: `23 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records `implementation_status.interfaces_present.target_verify_buffers="TargetVerifyBuffers"` and `implementation_status.kv_transaction_target_verify.device_buffers`:

- `rows=5`
- `candidate_rows=3`
- `summary_rows=2`
- `device="hip:0"`
- `token_ids_dtype="int32"`
- `active_mask_dtype="bool"`
- `target_top1_shape=[5]`
- `accepted_counts_shape=[2]`

It still records `status=blocked`, `performance_claim=false`, and `native_target_verify_ready=false`.

Interpretation: the host/runtime boundary now has validated Tensor-handle metadata for the future native verifier's row inputs and GPU accept/commit summaries. Resident target verification, GPU accept-summary kernels, and device-side state/KV commit remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`76 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows device-buffer ABI metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — TargetStateCommitBuffers state/KV commit ABI scaffold

### Scope

- Added `hipengine.speculative.TargetStateCommitBuffers` as a torch-free device-buffer ABI descriptor for committing verified target state rows.
- `TargetStateCommitBuffers.for_plan(plan, ...)` validates per-request accepted-count/commit-row/commit-position summary tensors and optional src/dst buffer pairs for linear-state and full-attention KV row copies.
- Updated the DFlash blocker helper/artifact and Task #15 with state/KV commit-buffer evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_kvcache_policy.py tests/test_dispatch_batch.py -q
```

Result: `24 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records `implementation_status.interfaces_present.target_state_commit_buffers="TargetStateCommitBuffers"` and `implementation_status.kv_transaction_target_verify.state_commit_buffers`:

- `request_rows=2`
- `device="hip:0"`
- `has_linear_state=true`
- `linear_state_tail_shape=[40,128]`
- `has_kv_rows=true`
- `kv_dst_rows=3`

It still records `status=blocked`, `performance_claim=false`, and `native_target_verify_ready=false`.

Interpretation: the host/runtime boundary now has validated Tensor-handle metadata for future device-side copies from verified target rows into live recurrent state and KV destinations. Resident target verification, GPU accept-summary kernels, and device-side state/KV commit remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`76 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows state/KV commit-buffer ABI metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — Resident target_verify_batch metadata API

### Scope

- Added metadata-only `Qwen35ParoResidentSession.target_verify_batch(...)`.
- The helper materializes a `TargetVerifyBatch` inside the resident runtime and validates:
  - target row count fits resident `max_batch_size`;
  - target positions fit `max_sequence_length`;
  - target token ids fit `vocab_size` when available.
- It intentionally does **not** run native target verification or commit state/KV rows.
- Updated the DFlash blocker helper/artifact and Task #15 with resident API evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_dflash_ddtree_blocker.py tests/test_qwen35_resident_batch_layout.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_speculative_interfaces.py -q
```

Result: `29 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records `implementation_status.resident_api`:

- `native_target_verify_batch=true`
- `speculative_verify_batch=false`
- `commit_verified_state=false`
- `native_target_verify_ready=false`

It also updates the leading blocker to state that `target_verify_batch` is metadata-only and does not run a native root+candidate target forward.

Interpretation: the resident runtime can now construct/validate target-verifier row metadata, but native verifier execution, GPU accept summaries, and device-side state/KV commit remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`77 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows resident runtime metadata API only; no performance claim or kernel change was made.

---

## 2026-05-15 — Resident verify_speculative_batch buffer API

### Scope

- Added metadata-only `Qwen35ParoResidentSession.verify_speculative_batch(...)`.
- The helper binds a resident `TargetVerifyBatch` to `TargetVerifyBuffers` and validates:
  - target row count fits resident `max_batch_size`;
  - target positions fit `max_sequence_length`;
  - row tensors and per-request summary tensors satisfy the existing `TargetVerifyBuffers` ABI.
- It intentionally does **not** run native target verification, produce GPU accept summaries, or commit state/KV rows.
- Updated the DFlash blocker helper/artifact and Task #15 with resident API evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_dflash_ddtree_blocker.py tests/test_qwen35_resident_batch_layout.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_speculative_interfaces.py -q
```

Result: `29 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records `implementation_status.resident_api`:

- `native_target_verify_batch=true`
- `speculative_verify_batch=true`
- `commit_verified_state=false`
- `native_target_verify_ready=false`

The leading blocker now states that `target_verify_batch/verify_speculative_batch` are metadata-only and do not run a native root+candidate target forward.

Interpretation: the resident runtime can now construct/validate target-verifier row metadata and associated device buffer handles, but native verifier execution, GPU accept summaries, and device-side state/KV commit remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`77 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows resident verifier-buffer metadata API only; no performance claim or kernel change was made.

---

## 2026-05-15 — Resident commit_verified_state metadata API

### Scope

- Added metadata-only `Qwen35ParoResidentSession.commit_verified_state(...)`.
- The helper binds a `TargetCommitPlan` to `TargetStateCommitBuffers` and validates:
  - request ids and mode match;
  - at least one linear-state or KV-row copy buffer pair is present;
  - buffers live on the resident device when the session has device metadata.
- It intentionally does **not** copy recurrent state, mutate KV, or mark the transaction committed.
- Updated the DFlash blocker helper/artifact and Task #15 with resident API evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_dflash_ddtree_blocker.py tests/test_qwen35_resident_batch_layout.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_speculative_interfaces.py -q
```

Result: `29 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records `implementation_status.resident_api`:

- `native_target_verify_batch=true`
- `speculative_verify_batch=true`
- `commit_verified_state=true`
- `native_target_verify_executes_kernels=false`
- `commit_verified_state_executes_copies=false`
- `native_target_verify_ready=false`

The leading blocker now states that `target_verify_batch/verify_speculative_batch/commit_verified_state` are metadata-only and do not run a native root+candidate target forward or state/KV copy.

Interpretation: the resident runtime now has metadata-only entry points for target verify row layout, verifier buffers, and commit buffers. Native verifier execution, GPU accept summaries, and device-side state/KV copies remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`77 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows resident commit-buffer metadata API only; no performance claim or kernel change was made.

---

## 2026-05-15 — Resident speculative execution metadata status

### Scope

- Added `Qwen35ParoResidentSpeculativeExecution` and `Qwen35ParoResidentSession.speculative_execution_metadata()`.
- The runtime now owns the readiness distinction between metadata-only resident APIs and executable native target verification:
  - `target_verify_batch` metadata is present;
  - `verify_speculative_batch` metadata is present;
  - `commit_verified_state` metadata is present;
  - native root+candidate target-forward kernels are not wired;
  - state/KV copy kernels are not wired;
  - throughput claims remain ineligible.
- Updated the DFlash blocker helper/artifact and Task #15 to consume the runtime status instead of inferring readiness from `hasattr(...)` alone.

### Evidence

```bash
python3 -m compileall -q hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_dflash_ddtree_blocker.py tests/test_qwen35_resident_batch_layout.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_speculative_interfaces.py -q
```

Result: `30 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records `implementation_status.resident_api.blockers`:

- `target_verify_batch/verify_speculative_batch/commit_verified_state are metadata-only`
- `native root+candidate target forward kernels are not wired`
- `GPU accept-summary kernels are not wired`
- `verified state/KV copy kernels are not wired`

It also records `native_target_verify_ready=false` and `throughput_claim_eligible=false` from the resident metadata object.

Interpretation: the resident runtime now centrally reports why speculative target verification is blocked. API presence alone can no longer make the blocker artifact look ready; real kernel/copy execution must flip explicit status fields.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`78 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows resident readiness/reporting only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative verify work metadata

### Scope

- Added `SpeculativeVerifyWork` and `ResidentBatchScheduler.next_speculative_verify_work(...)`.
- The scheduler now validates that draft requests are active, prefill-complete, and still decode-eligible before materializing scheduler-owned `TargetVerifyBatch` plus candidate-row `WorkItem` metadata.
- Updated the DFlash blocker helper/artifact and Task #15 with scheduler speculative verifier evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `20 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records `implementation_status.interfaces_present.scheduler_speculative_verify_work=true` while resident speculative status remains blocked:

- `native_target_verify_ready=false`
- `throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative verifier metadata can now be emitted before resident runtime handoff, but native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`80 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative accept accounting

### Scope

- Added `ResidentBatchScheduler.record_speculative_accept(...)`.
- The scheduler now validates `TargetAcceptSummary.accepted_tokens` against each active request's remaining decode budget, then records accepted speculative tokens through the same completion/reclaim path as normal generation.
- Updated the DFlash blocker helper/artifact and Task #15 with scheduler accept-accounting evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.scheduler_speculative_verify_work=true`
- `implementation_status.interfaces_present.scheduler_speculative_accept=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative verifier metadata can now be emitted and accepted-token summaries can be accounted against request budgets. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler accept accounting only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative verify graph shape key

### Scope

- Added `ResidentBatchScheduler.speculative_verify_shape_key(...)`.
- Scheduler-owned speculative verify work now derives a graph bucket key from the active batch plus `TargetVerifyBatch` draft-depth/tree-shape metadata.
- Updated the DFlash blocker helper/artifact and Task #15 with scheduler graph-key evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.scheduler_speculative_verify_work=true`
- `implementation_status.interfaces_present.scheduler_speculative_accept=true`
- `implementation_status.interfaces_present.scheduler_speculative_shape_key=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative verifier metadata now carries the graph-bucket key needed for future replay/capture. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler graph-key metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative verify graph cache helper

### Scope

- Added `ResidentBatchScheduler.get_or_create_speculative_verify_graph(...)`.
- Scheduler-owned speculative verify work can now retrieve/cache graph or replay objects under the same `GraphBucketCache` using `TargetVerifyBatch` draft-depth/tree-shape metadata.
- Updated the DFlash blocker helper/artifact and Task #15 with scheduler graph-cache evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.scheduler_speculative_verify_work=true`
- `implementation_status.interfaces_present.scheduler_speculative_accept=true`
- `implementation_status.interfaces_present.scheduler_speculative_shape_key=true`
- `implementation_status.interfaces_present.scheduler_speculative_graph_cache=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative verifier metadata now has the cache entry point future graph replay needs. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler graph-cache metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative verify KV transaction helper

### Scope

- Added `ResidentBatchScheduler.begin_speculative_verify_transaction(...)`.
- Scheduler-owned speculative verify work can now open a `KVTransaction` through the configured KV policy using active request states and the `TargetVerifyBatch` candidate rows.
- Updated the DFlash blocker helper/artifact and Task #15 with scheduler KV-transaction evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.scheduler_speculative_verify_work=true`
- `implementation_status.interfaces_present.scheduler_speculative_accept=true`
- `implementation_status.interfaces_present.scheduler_speculative_shape_key=true`
- `implementation_status.interfaces_present.scheduler_speculative_graph_cache=true`
- `implementation_status.interfaces_present.scheduler_speculative_kv_transaction=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative verifier metadata now opens the KV transaction that future native verifier writes must use. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler KV-transaction metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative verify execution plan metadata

### Scope

- Added `SpeculativeVerifyPlan` and `ResidentBatchScheduler.plan_speculative_verify(...)`.
- The scheduler can now bundle one speculative target verifier replay as a host contract containing:
  - `TargetVerifyBatch` root+candidate metadata,
  - candidate-row `WorkItem`,
  - `KVTransaction`,
  - verify `BatchShapeKey`, and
  - the cached graph/replay object for that shape.
- The helper validates transaction request ids, candidate-row count, and per-request candidate counts against the target batch before returning the plan.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.scheduler_speculative_verify_plan=true`
- `implementation_status.kv_transaction_target_verify.scheduler_verify_plan.shape_key_matches_target=true`
- `implementation_status.kv_transaction_target_verify.scheduler_verify_plan.candidate_counts=[2,1]`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative verifier metadata now has a single plan object future native verifier launch code can consume. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler speculative-verify planning metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative verify buffer-plan binding

### Scope

- Added `SpeculativeVerifyBufferPlan` and `ResidentBatchScheduler.bind_speculative_verify_buffers(...)`.
- The scheduler can now bind a speculative verifier plan to resident `TargetVerifyBuffers` metadata before any native verifier launch exists.
- The binding validates request ids, target row count, candidate row count, and verifier mode against the scheduler-owned `TargetVerifyBatch`.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.scheduler_speculative_buffer_plan=true`
- `implementation_status.kv_transaction_target_verify.scheduler_buffer_plan.rows=5`
- `implementation_status.kv_transaction_target_verify.scheduler_buffer_plan.candidate_rows=3`
- `implementation_status.kv_transaction_target_verify.scheduler_buffer_plan.mode=verify_tree`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative verifier metadata now has a buffer-bound plan that future native target-verifier launch code can consume. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler speculative-verify buffer planning metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative commit plan metadata

### Scope

- Added `SpeculativeCommitPlan` and `ResidentBatchScheduler.plan_speculative_commit(...)`.
- The scheduler can now derive a transaction-scoped commit plan from a buffer-bound speculative verify plan and a `TargetAcceptSummary`.
- The helper validates request ids, mode, selected commit rows, row ownership, accepted depth, and commit token/position against the scheduler-owned `TargetVerifyBatch` before building the `TargetCommitPlan`.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.scheduler_speculative_commit_plan=true`
- `implementation_status.kv_transaction_target_verify.scheduler_commit_plan.accepted_counts=[2,1]`
- `implementation_status.kv_transaction_target_verify.scheduler_commit_plan.commit_rows=[3,4]`
- `implementation_status.kv_transaction_target_verify.scheduler_commit_plan.candidate_counts=[2,1]`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative verifier metadata now has the host commit contract future verified state/KV copy code must consume. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler speculative commit planning metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative state/KV commit buffer binding

### Scope

- Added `SpeculativeStateCommitPlan` and `ResidentBatchScheduler.bind_speculative_commit_buffers(...)`.
- The scheduler can now bind a scheduler-owned speculative commit plan to resident `TargetStateCommitBuffers` metadata before verified state/KV copy kernels exist.
- The binding validates request ids, verifier mode, and requires at least one linear-state or KV-row buffer pair.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.scheduler_speculative_state_commit_plan=true`
- `implementation_status.kv_transaction_target_verify.scheduler_state_commit_plan.request_rows=2`
- `implementation_status.kv_transaction_target_verify.scheduler_state_commit_plan.has_linear_state=true`
- `implementation_status.kv_transaction_target_verify.scheduler_state_commit_plan.has_kv_rows=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative verifier metadata now has the state/KV buffer-bound commit contract future copy kernels must consume. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler speculative state/KV commit buffer metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative KV transaction commit metadata

### Scope

- Added `ResidentBatchScheduler.commit_speculative_kv_transaction(...)`.
- The scheduler can now mark the scheduler-owned speculative `KVTransaction` committed after a state/KV buffer-bound commit plan is available.
- The helper validates transaction id and request ids against the scheduler commit plan, then calls the configured KV policy with `TargetCommitPlan.kv_accept_counts`.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.scheduler_speculative_kv_commit=true`
- `implementation_status.kv_transaction_target_verify.scheduler_kv_commit.accepted_counts=[2,1]`
- `implementation_status.kv_transaction_target_verify.scheduler_kv_commit.committed=true`
- `implementation_status.kv_transaction_target_verify.scheduler_kv_commit.rolled_back=false`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative verifier metadata now closes the host KV transaction lifecycle after verified state/KV commit buffers are bound. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler speculative KV transaction lifecycle metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative KV transaction rollback metadata

### Scope

- Added `ResidentBatchScheduler.rollback_speculative_kv_transaction(...)`.
- The scheduler can now roll back a scheduler-owned speculative `KVTransaction` from a `SpeculativeVerifyPlan` when verifier execution fails or is aborted before commit.
- The helper validates transaction request ids and candidate-row count against the target batch before calling the configured KV policy rollback path.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.scheduler_speculative_kv_rollback=true`
- `implementation_status.kv_transaction_target_verify.scheduler_kv_rollback.committed=false`
- `implementation_status.kv_transaction_target_verify.scheduler_kv_rollback.rolled_back=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative verifier metadata now covers both commit and rollback host KV transaction lifecycles. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler speculative KV transaction rollback metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative accept finalization metadata

### Scope

- Added `ResidentBatchScheduler.finalize_speculative_accept(...)`.
- The scheduler now validates that the committed `KVTransaction` matches the scheduler commit plan transaction id, request ids, and accepted counts before recording accepted tokens/completions.
- This closes the host-side speculative accept lifecycle after state/KV commit buffers and KV commit metadata are available.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.scheduler_speculative_accept_finalize=true`
- `implementation_status.kv_transaction_target_verify.scheduler_accept_finalize.active_generated_counts={"1":2,"2":1}`
- `implementation_status.kv_transaction_target_verify.scheduler_accept_finalize.completed_request_ids=[]`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative verifier metadata now records accepted tokens only after a matching committed KV transaction. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler speculative accept-finalization metadata only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative state-commit device coherency

### Scope

- Tightened `ResidentBatchScheduler.bind_speculative_commit_buffers(...)`.
- State/KV commit buffers must now live on the same device as the bound target-verifier buffers.
- Added a negative scheduler test that rejects a cross-device state-commit buffer plan.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.kv_transaction_target_verify.scheduler_state_commit_plan.device=hip:0`
- `implementation_status.kv_transaction_target_verify.scheduler_state_commit_plan.verify_device=hip:0`
- `implementation_status.kv_transaction_target_verify.scheduler_state_commit_plan.device_matches_verify=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative verifier metadata now rejects cross-device state/KV commit buffer bindings. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler speculative state/KV commit ABI validation only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative state-commit row coverage

### Scope

- Tightened `ResidentBatchScheduler.bind_speculative_commit_buffers(...)` again.
- Linear-state and KV source buffers must now cover all target-verifier rows.
- KV destination buffers must cover `sum(accepted_counts)` accepted token rows.
- Added a negative scheduler test that rejects a KV destination buffer sized only for request rows when the accept plan needs more token rows.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.kv_transaction_target_verify.scheduler_state_commit_plan.target_rows=5`
- `implementation_status.kv_transaction_target_verify.scheduler_state_commit_plan.accepted_rows=3`
- `implementation_status.kv_transaction_target_verify.scheduler_state_commit_plan.linear_src_covers_target=true`
- `implementation_status.kv_transaction_target_verify.scheduler_state_commit_plan.kv_src_covers_target=true`
- `implementation_status.kv_transaction_target_verify.scheduler_state_commit_plan.kv_dst_covers_accepts=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative verifier metadata now rejects state/KV commit buffers that cannot cover the target rows or accepted KV destination rows required by the commit plan. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler speculative state/KV commit ABI validation only; no performance claim or kernel change was made.

---

## 2026-05-15 — Resident target-verifier buffer device validation

### Scope

- Tightened `Qwen35ParoResidentSession.verify_speculative_batch(...)`.
- Metadata-only target-verifier buffers must now live on the resident session device when the resident device is available.
- Added a negative resident-layout test that rejects target-verifier buffers on `hip:1` for a resident `hip:0` session.

### Evidence

```bash
python3 -m compileall -q hipengine/runtime scripts/qwen35_dflash_ddtree_blocker.py tests/test_qwen35_resident_batch_layout.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_speculative_interfaces.py -q
```

Result: `30 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.resident_api.target_verify_buffers_resident_device_checked=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: resident speculative verifier metadata now rejects cross-device target-verifier buffers before native kernels are wired. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows resident speculative target-verifier ABI validation only; no performance claim or kernel change was made.

---

## 2026-05-15 — Resident verified-state commit row coverage

### Scope

- Tightened `Qwen35ParoResidentSession.commit_verified_state(...)`.
- Linear-state and KV source buffers must cover the selected commit rows in the `TargetCommitPlan`.
- KV destination buffers must cover `sum(accepted_counts)` accepted token rows.
- Added negative resident-layout tests for too-short linear-state source rows and too-short KV destination rows.

### Evidence

```bash
python3 -m compileall -q hipengine/runtime scripts/qwen35_dflash_ddtree_blocker.py tests/test_qwen35_resident_batch_layout.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_speculative_interfaces.py -q
```

Result: `30 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.resident_api.commit_verified_state_row_coverage_checked=true`
- `implementation_status.resident_api.target_verify_buffers_resident_device_checked=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: resident speculative commit metadata now rejects state/KV commit buffers that cannot cover selected commit rows or accepted KV destination rows. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows resident speculative state/KV commit ABI validation only; no performance claim or kernel change was made.

---

## 2026-05-15 — Speculative metadata unique request ids

### Scope

- Added shared unique-`request_ids` validation for speculative metadata dataclasses.
- `DraftBatch`, `TargetVerifyBatch`, `TargetVerifyBuffers`, `AcceptResult`, `TargetAcceptSummary`, `TargetCommitSelection`, `TargetCommitPlan`, and `TargetStateCommitBuffers` now reject duplicate request ids.
- Added negative speculative-interface tests for duplicate request ids in draft, accept, target batch, and commit plan metadata.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_speculative_interfaces.py -q
```

Result: `14 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.speculative_request_ids_unique_checked=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: speculative verifier metadata now rejects duplicate request ids before row/root/accept/commit maps can become ambiguous. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows speculative metadata correctness validation only; no performance claim or kernel change was made.

---

## 2026-05-15 — KV transaction unique request ids

### Scope

- Added unique-`request_ids` validation to `KVTransaction`.
- `FixedPagedKVPolicy.begin_transaction(...)` now rejects duplicate request ids before candidate counts are derived.
- Added negative KV policy tests for direct `KVTransaction` construction and duplicate sequence ids passed to `begin_transaction(...)`.

### Evidence

```bash
python3 -m compileall -q hipengine/kvcache hipengine/speculative scripts/qwen35_dflash_ddtree_blocker.py tests/test_kvcache_policy.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_kvcache_policy.py tests/test_speculative_interfaces.py -q
```

Result: `21 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.speculative_request_ids_unique_checked=true`
- `implementation_status.interfaces_present.kv_transaction_request_ids_unique_checked=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: host KV transaction metadata now rejects duplicate request ids before candidate/accept counts can become ambiguous. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows speculative KV transaction metadata correctness validation only; no performance claim or kernel change was made.

---

## 2026-05-15 — KV transaction terminal-state invariants

### Scope

- Tightened `KVTransaction` terminal-state validation:
  - committed transactions require `accepted_counts`;
  - `accepted_counts` require `committed=True`;
  - accepted counts must be non-negative and within candidate/draft budgets.
- Tightened `FixedPagedKVPolicy` terminal transitions:
  - already committed transactions cannot be committed again;
  - already rolled-back transactions cannot be rolled back again.
- Added negative KV policy tests for direct terminal-state construction and repeated commit/rollback transitions.

### Evidence

```bash
python3 -m compileall -q hipengine/kvcache hipengine/speculative scripts/qwen35_dflash_ddtree_blocker.py tests/test_kvcache_policy.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_kvcache_policy.py tests/test_speculative_interfaces.py -q
```

Result: `22 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.kv_transaction_request_ids_unique_checked=true`
- `implementation_status.interfaces_present.kv_transaction_terminal_state_checked=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: host KV transaction metadata now rejects inconsistent terminal states before scheduler finalization or rollback can become ambiguous. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows speculative KV transaction lifecycle validation only; no performance claim or kernel change was made.

---

## 2026-05-15 — KV transaction role invariant

### Scope

- Tightened `KVTransaction` verifier-role validation to accept only `verify_chain` and `verify_tree`.
- Added negative KV policy coverage for direct construction with an invalid role.
- Updated the DFlash/DDTree blocker artifact so role validation is explicit evidence instead of an inferred property of scheduler-created transactions.

### Evidence

```bash
python3 -m compileall -q hipengine/kvcache hipengine/speculative scripts/qwen35_dflash_ddtree_blocker.py tests/test_kvcache_policy.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_kvcache_policy.py tests/test_speculative_interfaces.py -q
```

Result: `23 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.kv_transaction_request_ids_unique_checked=true`
- `implementation_status.interfaces_present.kv_transaction_terminal_state_checked=true`
- `implementation_status.interfaces_present.kv_transaction_role_checked=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: host KV transaction metadata now rejects invalid verifier roles before scheduler/native verifier routing can become ambiguous. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows speculative KV transaction role validation only; no performance claim or kernel change was made.

---

## 2026-05-15 — Target commit-plan transaction role binding

### Scope

- Tightened `TargetCommitPlan.from_summary(...)` role validation:
  - transaction roles must be one of `verify_chain`/`verify_tree`;
  - transaction role must match the target accept-summary mode;
  - invalid roles are no longer silently treated as the summary mode.
- Added speculative-interface coverage for role mismatch and invalid-role transaction-like objects.
- Updated the DFlash/DDTree blocker artifact so commit-plan/transaction role binding is explicit evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative hipengine/kvcache scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py tests/test_kvcache_policy.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_kvcache_policy.py -q
```

Result: `23 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.kv_transaction_role_checked=true`
- `implementation_status.interfaces_present.target_commit_plan_transaction_role_checked=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: host commit-plan metadata now rejects verifier-mode mismatches before scheduler/native verifier commit routing can become ambiguous. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows speculative commit-plan role binding only; no performance claim or kernel change was made.

---

## 2026-05-15 — Target commit-plan candidate-budget binding

### Scope

- Added `TargetAcceptSummary.candidate_counts` so accept summaries carry the per-request candidate budget from their originating `TargetVerifyBatch`.
- Tightened `TargetCommitPlan.from_summary(...)` so transaction `candidate_counts` must match the target accept-summary budget when present.
- Added speculative-interface coverage for mismatched transaction candidate budgets and updated the DFlash/DDTree blocker artifact evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative hipengine/kvcache scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py tests/test_kvcache_policy.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_kvcache_policy.py -q
```

Result: `23 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.target_commit_plan_transaction_role_checked=true`
- `implementation_status.interfaces_present.target_commit_plan_candidate_budget_checked=true`
- `implementation_status.kv_transaction_target_verify.accept_summary.candidate_counts=[2,1]`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: host commit-plan metadata now rejects accept-summary / transaction candidate-budget mismatches before scheduler/native verifier commit routing can become ambiguous. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows speculative commit-plan candidate-budget binding only; no performance claim or kernel change was made.

---

## 2026-05-15 — State commit-buffer transaction binding

### Scope

- Added `TargetStateCommitBuffers.transaction_id`; `for_plan(...)` now stamps buffers with the source `TargetCommitPlan.transaction_id`.
- Tightened scheduler and resident commit-buffer binding:
  - `ResidentBatchScheduler.bind_speculative_commit_buffers(...)` rejects state/KV commit buffers whose transaction id does not match the scheduler commit plan;
  - `Qwen35ParoResidentSession.commit_verified_state(...)` rejects state/KV commit buffers whose transaction id does not match the resident commit plan.
- Updated speculative, scheduler, and resident tests plus blocker artifact evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative hipengine/generation hipengine/runtime scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py tests/test_qwen35_resident_batch_layout.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py tests/test_qwen35_resident_batch_layout.py -q
```

Result: `39 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.resident_api.commit_verified_state_transaction_id_checked=true`
- `implementation_status.kv_transaction_target_verify.state_commit_buffers.transaction_id=0`
- `implementation_status.kv_transaction_target_verify.scheduler_state_commit_plan.transaction_id_matches=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: host state/KV commit buffers now bind to the same speculative transaction as the commit plan before scheduler/resident copy routing can proceed. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows speculative state/KV commit transaction binding only; no performance claim or kernel change was made.

---

## 2026-05-15 — Target verify-buffer transaction binding

### Scope

- Added optional `TargetVerifyBuffers.transaction_id`; buffer construction rejects negative ids.
- Tightened `ResidentBatchScheduler.bind_speculative_verify_buffers(...)` so target-verifier buffers that carry a transaction id must match the scheduler-owned speculative KV transaction.
- Updated scheduler/speculative tests and DFlash/DDTree blocker artifact evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py -q
```

Result: `22 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.target_verify_buffers_transaction_id_checked=true`
- `implementation_status.kv_transaction_target_verify.scheduler_buffer_plan.buffer_transaction_id=1`
- `implementation_status.kv_transaction_target_verify.scheduler_buffer_plan.transaction_id_matches=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: host target-verifier device buffers can now bind to the same speculative transaction as the scheduler verify plan before native verifier kernels consume them. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows speculative target-verifier buffer transaction binding only; no performance claim or kernel change was made.

---

## 2026-05-15 — Target verify-buffer candidate-budget binding

### Scope

- Added `TargetVerifyBuffers.candidate_counts`; `for_batch(...)` now stamps target-verifier buffers with the originating `TargetVerifyBatch` per-request candidate budget.
- Tightened `ResidentBatchScheduler.bind_speculative_verify_buffers(...)` so target-verifier buffers carrying candidate counts must match the scheduler-owned target batch candidate budget.
- Updated scheduler/speculative tests and DFlash/DDTree blocker artifact evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py -q
```

Result: `22 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.target_verify_buffers_candidate_counts_checked=true`
- `implementation_status.kv_transaction_target_verify.scheduler_buffer_plan.candidate_counts=[2,1]`
- `implementation_status.kv_transaction_target_verify.scheduler_buffer_plan.candidate_counts_match=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: host target-verifier device buffers now bind to the same per-request candidate budget as the scheduler verify plan before native verifier kernels consume them. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows speculative target-verifier buffer candidate-budget binding only; no performance claim or kernel change was made.

---

## 2026-05-15 — Target verify-buffer topology binding

### Scope

- Added `TargetVerifyBuffers.draft_depth` and `tree_shape`; `for_batch(...)` now stamps target-verifier buffers with the originating `TargetVerifyBatch` topology.
- Tightened `ResidentBatchScheduler.bind_speculative_verify_buffers(...)` so target-verifier buffers carrying topology metadata must match the scheduler-owned target batch draft depth and tree shape.
- Updated scheduler/speculative tests and DFlash/DDTree blocker artifact evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py -q
```

Result: `22 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.target_verify_buffers_topology_checked=true`
- `implementation_status.kv_transaction_target_verify.scheduler_buffer_plan.draft_depth_matches=true`
- `implementation_status.kv_transaction_target_verify.scheduler_buffer_plan.tree_shape_matches=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: host target-verifier device buffers now bind to the same verify-tree topology as the scheduler verify plan before native verifier kernels consume them. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows speculative target-verifier buffer topology binding only; no performance claim or kernel change was made.

---

## 2026-05-15 — Resident target verify-buffer transaction id

### Scope

- Added optional `transaction_id` propagation to `Qwen35ParoResidentSession.verify_speculative_batch(...)`.
- Resident-created `TargetVerifyBuffers` now carry the supplied transaction id and reject negative transaction ids through the existing buffer validation path.
- Updated resident/speculative tests and DFlash/DDTree blocker artifact evidence.

### Evidence

```bash
python3 -m compileall -q hipengine/runtime scripts/qwen35_dflash_ddtree_blocker.py tests/test_qwen35_resident_batch_layout.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_speculative_interfaces.py -q
```

Result: `31 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.resident_api.target_verify_buffers_transaction_id_checked=true`
- `implementation_status.interfaces_present.target_verify_buffers_topology_checked=true`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: resident-created target-verifier device buffers can now carry a speculative transaction id before native verifier kernels consume them. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows resident target-verifier buffer transaction-id propagation only; no performance claim or kernel change was made.

---

## 2026-05-15 — Accept-summary and commit-plan topology binding

### Scope

- Added `draft_depth` and `tree_shape` metadata to `TargetAcceptSummary` and `TargetCommitPlan`.
- `TargetAcceptSummary.from_accept_result(...)` now stamps accept summaries with the originating target batch topology.
- `TargetCommitPlan.from_summary(...)` now preserves topology on the commit plan.
- `ResidentBatchScheduler.plan_speculative_commit(...)` now rejects accept summaries whose candidate budget, draft depth, or tree shape do not match the scheduler target batch.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py -q
```

Result: `22 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.target_accept_summary_topology_checked=true`
- `implementation_status.kv_transaction_target_verify.accept_summary.draft_depth=2`
- `implementation_status.kv_transaction_target_verify.accept_summary.tree_shape=[0,1,0]`
- `implementation_status.kv_transaction_target_verify.commit_plan.tree_shape=[0,1,0]`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: host accept summaries and commit plans now preserve and validate the target verify-tree topology before native verifier accept/commit routing consumes them. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows speculative accept-summary / commit-plan topology binding only; no performance claim or kernel change was made.

---

## 2026-05-15 — Accept-summary transaction binding

### Scope

- Added optional `transaction_id` metadata to `AcceptResult` and `TargetAcceptSummary`.
- `TargetAcceptSummary.from_accept_result(...)` now preserves GPU/host accept-result transaction ids.
- `TargetCommitPlan.from_summary(...)` rejects accept summaries whose transaction id does not match the KV transaction.
- `ResidentBatchScheduler.plan_speculative_commit(...)` rejects accept summaries whose transaction id does not match the scheduler-owned verify plan.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py -q
```

Result: `22 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.target_accept_summary_transaction_id_checked=true`
- `implementation_status.kv_transaction_target_verify.accept_summary.transaction_id=0`
- `implementation_status.kv_transaction_target_verify.scheduler_commit_plan.summary_transaction_id=1`
- `implementation_status.kv_transaction_target_verify.scheduler_commit_plan.transaction_id=1`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: host/GPU accept-summary metadata can now bind to the same speculative transaction as the scheduler commit plan before native accept/commit routing consumes it. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows speculative accept-summary transaction binding only; no performance claim or kernel change was made.

---

## 2026-05-15 — Accept-result selected row provenance

### Scope

- Added optional `selected_candidate_rows` metadata to `AcceptResult`.
- `TargetAcceptSummary.from_accept_result(...)` now consumes accept-result row hints by default and rejects conflicting explicit row hints.
- Ambiguous tree-depth accept results can now carry the exact target row selected by a GPU/host accept-summary producer before commit planning consumes the summary.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py -q
```

Result: `22 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.accept_result_selected_rows_checked=true`
- `implementation_status.kv_transaction_target_verify.accept_result.selected_candidate_rows=[3,4]`
- `implementation_status.kv_transaction_target_verify.accept_summary.commit_rows=[3,4]`
- `implementation_status.kv_transaction_target_verify.commit_plan.commit_rows=[3,4]`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: accept-result metadata can now disambiguate tree rows with identical accepted depths and preserves the selected target rows into accept summaries and commit plans. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows speculative accept-result selected-row provenance only; no performance claim or kernel change was made.

---

## 2026-05-15 — Accept-result next-token provenance

### Scope

- Added optional `next_tokens` metadata to `AcceptResult`.
- `TargetAcceptSummary.from_accept_result(...)` now propagates verifier next-token metadata.
- `TargetCommitPlan` now preserves next-token metadata so scheduler commit plans can carry correction/bonus tokens alongside accepted rows.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py -q
```

Result: `22 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.accept_result_next_tokens_checked=true`
- `implementation_status.kv_transaction_target_verify.accept_result.next_tokens=[12,21]`
- `implementation_status.kv_transaction_target_verify.accept_summary.next_tokens=[12,21]`
- `implementation_status.kv_transaction_target_verify.commit_plan.next_tokens=[12,21]`
- `implementation_status.kv_transaction_target_verify.scheduler_commit_plan.next_tokens=[12,21]`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: target verifier accept results can now carry the target-side correction/bonus token metadata needed after the accepted prefix. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows speculative accept-result next-token provenance only; no performance claim or kernel change was made.

---

## 2026-05-15 — Target-verifier next-token buffer ABI

### Scope

- Added optional `next_tokens` tensor metadata to `TargetVerifyBuffers`.
- `TargetVerifyBuffers` now validates optional next-token output buffer shape, device, and integer dtype with the other per-request verifier summary outputs.
- `Qwen35ParoResidentSession.verify_speculative_batch(...)` now accepts and forwards that optional next-token buffer handle.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative hipengine/runtime hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py tests/test_qwen35_resident_batch_layout.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py tests/test_qwen35_resident_batch_layout.py -q
```

Result: `39 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.target_verify_buffers_next_tokens_checked=true`
- `implementation_status.kv_transaction_target_verify.device_buffers.next_tokens_shape=[2]`
- `implementation_status.kv_transaction_target_verify.device_buffers.next_tokens_dtype=int32`
- `implementation_status.kv_transaction_target_verify.scheduler_buffer_plan.next_tokens_shape=[2]`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: the target-verifier device-buffer ABI can now expose per-request next-token outputs for future GPU accept-summary kernels. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows target-verifier next-token output-buffer provenance only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler speculative next-token recording

### Scope

- `ResidentBatchScheduler.record_speculative_accept(...)` now records accepted speculative tokens plus optional target next tokens from `TargetAcceptSummary.next_tokens`.
- The scheduler rejects accepted+next output paths that exceed the request's remaining decode budget.
- `finalize_speculative_accept(...)` inherits the same behavior because it records the scheduler-owned commit summary only after KV transaction commit validation.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `22 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.scheduler_speculative_next_tokens_checked=true`
- `implementation_status.kv_transaction_target_verify.scheduler_accept_finalize.active_generated_counts={"1":3,"2":2}`
- `implementation_status.kv_transaction_target_verify.scheduler_commit_plan.next_tokens=[12,21]`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler finalization can now emit the target-side correction/bonus token after the accepted draft prefix and enforces request decode budgets for that combined output path. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows scheduler speculative next-token finalization only; no performance claim or kernel change was made.

---

## 2026-05-15 — Target accept-summary CPU oracle

### Scope

- Added `TargetVerifyBatch.accept_from_top1(...)`, a CPU oracle for future GPU accept-summary kernels.
- The oracle follows target-top1 edges from each request root across chain/tree candidate rows, emits accepted draft counts/tokens, selected commit rows, transaction ids, and target next-token correction/bonus metadata.
- It honors optional remaining-decode budgets and rejects invalid target-top1 shapes/negative ids and ambiguous duplicate matching children.

### Evidence

```bash
python3 -m compileall -q hipengine/speculative hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py && \
  python3 -m pytest tests/test_speculative_interfaces.py tests/test_generation_batch_scheduler.py -q
```

Result: `23 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.target_accept_oracle_checked=true`
- `implementation_status.kv_transaction_target_verify.target_top1=[10,20,11,12,21]`
- `implementation_status.kv_transaction_target_verify.accept_result.selected_candidate_rows=[3,4]`
- `implementation_status.kv_transaction_target_verify.accept_result.next_tokens=[12,21]`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: future GPU accept-summary kernels now have a deterministic host oracle for accepted counts/tokens, selected commit rows, and correction/bonus tokens over the same `TargetVerifyBatch` layout. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Robust active TaskList count remains `1` (#15). This iteration narrows target accept-summary oracle coverage only; no performance claim or kernel change was made.

---

## 2026-05-15 — Scheduler commit planning from target top1

### Scope

- Added `ResidentBatchScheduler.plan_speculative_commit_from_top1(...)`.
- The helper derives a `TargetAcceptSummary` from `TargetVerifyBatch.accept_from_top1(...)`, binds it to the scheduler-owned verify transaction id, uses scheduler remaining-decode budgets by default, and reuses the existing `plan_speculative_commit(...)` validation path.
- The DFlash blocker now routes the scheduler commit-plan sample through target-top1 oracle outputs rather than manually replacing summary transaction ids.

### Evidence

```bash
python3 -m compileall -q hipengine/generation scripts/qwen35_dflash_ddtree_blocker.py tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py && \
  python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_speculative_interfaces.py -q
```

Result: `23 passed`.

Updated blocker artifact:

```bash
python3 scripts/qwen35_dflash_ddtree_blocker.py \
  --json benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json
```

Artifact now records:

- `implementation_status.interfaces_present.scheduler_speculative_commit_from_top1=true`
- `implementation_status.kv_transaction_target_verify.scheduler_commit_plan.from_top1=true`
- `implementation_status.kv_transaction_target_verify.scheduler_commit_plan.summary_transaction_id=1`
- `implementation_status.kv_transaction_target_verify.scheduler_commit_plan.next_tokens=[12,21]`
- `implementation_status.resident_api.native_target_verify_ready=false`
- `implementation_status.resident_api.throughput_claim_eligible=false`

Interpretation: scheduler-owned speculative commit planning can now consume target-top1 oracle outputs directly and bind them to the active speculative transaction before future GPU accept-summary kernels are wired. Native verifier execution, GPU accept summaries, and verified state/KV copy kernels remain unimplemented.

### Validation

```bash
python3 -m compileall -q hipengine tests scripts && \
  python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py tests/test_loading_materialize.py tests/test_generation_qwen35_paro.py tests/test_runtime_workspace.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
```

Result: pytest exit code `0` (`81 passed`).

Active TaskList count remains `1` (#15). This iteration narrows scheduler target-top1 commit planning only; no performance claim or kernel change was made.

---

## 2026-05-15 — Qwen3.5-0.8B-PARO hipENGINE feasibility check blocked

### Request

From `~/amd-gpu-tuning`: check whether `z-lab/Qwen3.5-0.8B-PARO` can be run
through `~/hipengine` for a 512/128 comparison. User clarified that only 512/128
is needed because prefill will be slow.

### Environment

```bash
python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
rocminfo | grep -E 'Name:|gfx' | head -8
rocm-smi --showmeminfo vram --showuse --showtemp
hipcc --version | head -20
```

Observed: HIP runtime loads, W7900/gfx1100 visible, idle VRAM 27,930,624 B,
`hipcc` reports `HIP version: 7.2.53211-d40244d`.

### Model inspection

```bash
mamba run -n therock --no-capture-output python3 - <<'PY'
from hipengine.loading.safetensors import load_weight_index
from hipengine.loading.qwen35_paro import qwen35_paro_config_from_hf
p='/models/huggingface/hub/models--z-lab--Qwen3.5-0.8B-PARO/snapshots/da941f4fd3fa72763c398db6cb14b2bef1ee961f'
idx=load_weight_index(p)
cfg=qwen35_paro_config_from_hf(idx.config)
normalized=[]
for name in idx.tensors:
    n=name
    for pref in ('model.language_model.','language_model.','model.'):
        if n.startswith(pref):
            n=n[len(pref):]
            break
    normalized.append(n)
print('architecture', cfg.architecture)
print('layers', cfg.num_hidden_layers)
print('hidden', cfg.hidden_size)
print('num_experts', cfg.num_experts)
print('num_experts_per_tok', cfg.num_experts_per_tok)
print('tie_word_embeddings', idx.config.get('tie_word_embeddings'), idx.config.get('text_config', {}).get('tie_word_embeddings'))
print('has_lm_head', 'lm_head.weight' in set(normalized))
print('has_dense_gate_proj', 'layers.0.mlp.gate_proj.qweight' in set(normalized))
print('has_moe_expert0', 'layers.0.mlp.experts.0.gate_proj.qweight' in set(normalized))
PY
```

Result: `Qwen3_5ForConditionalGeneration`, 24 layers, hidden 1024,
`num_experts=0`, `num_experts_per_tok=0`, tied embeddings true, no explicit
`lm_head.weight`, dense MLP tensors present, MoE expert tensors absent.

### Feasibility smoke

Attempted the smallest safe resident smoke before launching the requested
512/128 row:

```bash
MODEL=/models/huggingface/hub/models--z-lab--Qwen3.5-0.8B-PARO/snapshots/da941f4fd3fa72763c398db6cb14b2bef1ee961f
mamba run -n therock --no-capture-output python3 scripts/qwen35_paro_bench.py \
  --model "$MODEL" --prompt-length 1 --decode-tokens 1 --warmup-decode-tokens 0 \
  --max-layers 1 --progress --json /tmp/hipengine_qwen35_0p8b_smoke.json
```

Result: blocked during resident build, before prefill/decode:

```text
resident_build_start layers=1 max_sequence_length=3
load_kernel_libraries_start
load_kernel_libraries_done
load_embedding_start
load_embedding_done
load_final_norm_start
load_final_norm_done
load_lm_head_start
KeyError: 'lm_head.weight'
```

Artifact: `benchmarks/results/2026-05-15-hipengine-qwen35-0p8b-paro-512-128-blocked.json`.

### Interpretation

hipENGINE cannot currently run `z-lab/Qwen3.5-0.8B-PARO` for the requested
512/128 comparison. The first observed blocker is the tied-embedding checkpoint
layout: the resident runner requires `lm_head.weight`, while this model has no
explicit lm head and relies on `embed_tokens.weight`.

Static inspection shows a second blocker immediately behind that: the current
Qwen3.5/PARO resident layer path is MoE-specific (`materialize_qwen35_paro_*_moe_c1_runtime_layer`,
`run_*_moe_c1_layer_*`) and expects `mlp.experts.*` plus router/shared-expert
weights. The 0.8B checkpoint is dense (`num_experts=0`) with `mlp.gate_proj`,
`mlp.up_proj`, and `mlp.down_proj` PARO tensors.

For future comparison, the parent native PARO baseline from
`~/amd-gpu-tuning/artifacts/qwen35_0p8b_paro_20260515_120459/` is:

| Engine | Shape | Prefill tok/s | Decode tok/s | Peak GiB | Correctness |
| --- | --- | ---: | ---: | ---: | --- |
| `nano-vllm-amd` native PARO | 512/128 | 11363.34 | 251.78 | 1.171 | finite logits + graph/eager match |

Needed before rerunning in hipENGINE: tied lm-head fallback plus dense PARO MLP
materialization/runtime support, then rerun the 512/128 command with the normal
post-run correctness/perf/memory gates.

## 2026-05-15 — docs/PREFILL.md correction pass

Reviewed `docs/PREFILL.md` against `docs/PLAN.md`, the current Qwen3.5/PARO
resident runtime, `docs/KERNELS.md`, and retained prefill/batch artifacts. The
old plan was broadly directionally right but had stale/incomplete implementation
status: it treated grouped MoE as an immediate correctness blocker, implied the
scalar RoPE position kernel might already cover T rows, did not distinguish the
current token-major serial suffix from the desired layer-major fallback, and
said `KVLiveSpans` needed extension even though `request_ids`/`row_positions`
already exist.

Rewrote `docs/PREFILL.md` as an implementation spec. Key decisions recorded:
D2/layer-major serial full-attention fallback lands before the D1 native causal
prefill attention kernel; existing `run_moe_c1_fp16(tokens=T)` is the B/C
correctness fallback while grouped/compact MoE is a parent-parity perf step;
full-attention D1 needs a new vector-position RoPE path because the landed
`qwen35_head_rmsnorm_partial_rotary_position_f32_bf16` reads `position_ptr[0]`;
compact c>N prefill should populate existing `KVLiveSpans` metadata rather than
redesigning the span ABI.

Validation (docs/process only; no GPU run required):

```bash
git diff --check -- docs/PREFILL.md
python3 - <<'PY'
from pathlib import Path
text=Path('docs/PREFILL.md').read_text()
assert text.count('```') % 2 == 0, 'unbalanced fences'
print('docs/PREFILL.md ok', text.count('\n') + 1, 'lines')
PY
```

Result: `docs/PREFILL.md ok 714 lines`.

## 2026-05-15 — PREFILL plan switched to final-only implementation

Incorporated follow-up direction to skip throwaway intermediate prefill paths.
The earlier docs/PREFILL.md correction had specified a layer-major row-loop
full-attention fallback before the native causal kernel; that is now superseded.
The plan now targets the final implementation directly: native full-attention
prefill, grouped/compact MoE, generation wiring, and compact c>N slabs, with
serial/row-loop/c1-MoE paths kept only as correctness oracles and artifact
reproduction helpers.

Important spec decisions now recorded in `docs/PREFILL.md`:
- no retained perf artifacts for layer-major full-attention row loops,
  c1 selected-row MoE prefill, or per-request c>N packed fallback;
- first native full-attention kernel uses append-then-attend from BF16 paged KV
  cache, post-gate FP16 output, and the existing GQA gate-fused decode shape;
- `full_attn_prefill` needs a torch-free CPU-reference oracle before the gfx1100
  retained kernel key lands;
- compact c>N final path buckets slab rows by common block-table length unless a
  varlen block-table writer is ported;
- `PrefillConfig.require_full_native` defaults to true, with explicit per-call
  override semantics.

Validation (docs/process only; no GPU run required):

```bash
git diff --check -- docs/PREFILL.md
python3 - <<'PY'
from pathlib import Path
text=Path('docs/PREFILL.md').read_text()
assert text.count('```') % 2 == 0, 'unbalanced fences'
assert 'D1' not in text and 'D2' not in text, 'stale D1/D2 terminology'
assert '## Phased implementation plan' not in text, 'stale phased plan'
print('docs/PREFILL.md final-spec ok', text.count('\n') + 1, 'lines')
PY
```

Result: `docs/PREFILL.md final-spec ok 453 lines`.

## 2026-05-15 — PREFILL final spec reviewer tightening

Incorporated the final reviewer pass on `docs/PREFILL.md` before implementation.
The spec remains final-path-only, but now clarifies how code may land without
retaining intermediate perf artifacts: native pieces can be merged behind
`require_full_native=False` / probe-only entrypoints while oracle paths fill
missing pieces, but the first retained prefill performance row requires all
native pieces, `PrefillConfig.require_full_native=True`, and no c1 selected-row
MoE in the production prefill path.

Additional tightened points:
- current 117.24 tok/s c=1 row stays the retained benchmark baseline until the
  final single-request native artifact is accepted;
- full native prefill requires `T >= config.linear_conv_kernel_dim`, with short
  prompts raising instead of silently serial-falling-back;
- Qwen3.5/PARO embedding dispatch is FP16-hidden (`embedding_lookup_batch_fp16_i64`)
  through the backend/model path;
- bulk full-attention prepare must cast `T * kv_width` K elements, not the scalar
  `kv_width` amount;
- compact c>N attention uses `cu_seqlens_q/cu_seqlens_k` as the kernel mask ABI,
  while `row_to_request` stays scheduler/debug metadata;
- compact scheduler needs `bucketize_by_block_count`, and ordinary compact
  prefill commits canonical KV inline rather than using speculative KV
  transactions;
- final validation now uses greedy ID equality plus KL <= 0.05 / top-1 >= 90%,
  and adds a chunk-equivalence sweep for all non-zero PrefillConfig chunk knobs.

Validation (docs/process only; no GPU run required):

```bash
git diff --check -- docs/PREFILL.md
python3 - <<'PY'
from pathlib import Path
text=Path('docs/PREFILL.md').read_text()
assert text.count('```') % 2 == 0, 'unbalanced fences'
required = [
    'Implementation landing policy',
    'linear_conv_kernel_dim',
    'T * kv_width',
    'bucketize_by_block_count',
    'cu_seqlens_q`/`cu_seqlens_k` define',
    'non-speculative and commits canonical KV inline',
    'KL ≤ 0.05',
    'Chunk-equivalence sweep',
]
missing=[item for item in required if item not in text]
if missing:
    raise SystemExit(f'missing required text: {missing}')
print('docs/PREFILL.md reviewer-tightening ok', text.count('\n') + 1, 'lines')
PY
rg -n "same IDs/logits|bit-stable|D1|D2|## Phased implementation plan" docs/PREFILL.md || true
```

Result: `docs/PREFILL.md reviewer-tightening ok 498 lines`; stale-term grep
returned no matches.

## 2026-05-15 — Prefill API/config implementation start

Started implementation of the final-path prefill spec with the API/config
foundation, without adding a retained intermediate production prefill path.

Changes:
- added `hipengine.runtime.PrefillConfig` with typed chunk knobs and
  `require_full_native=True` default;
- wired `PrefillConfig` into `Qwen35ParoResidentSession`;
- added session-owned `prefill_token_ids` / `prefill_positions` device buffers;
- added `Qwen35ParoResidentSession.prefill_native(...)` with the public
  position-0 contract, `linear_conv_kernel_dim` short-prompt guard, per-call
  `require_full_native` override precedence, and default NotImplementedError
  until native full-attention prefill + grouped MoE are wired;
- preserved `prefill_linear_tokens_native(...)` as a compatibility alias for
  retained legacy/native-prefix artifacts;
- updated `scripts/qwen35_paro_bench.py --native-prefill` to call
  `prefill_native(..., require_full_native=False)` explicitly so oracle/probe
  usage is labelled at the call site.

Validation:

```bash
python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
rocminfo | grep -E 'Name:|gfx' | head -40
python3 -m pytest tests/test_qwen35_resident_batch_layout.py -q
python3 -m py_compile hipengine/runtime/prefill.py hipengine/runtime/__init__.py hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_paro_bench.py
```

Result: ROCm visible (`gfx1100`, W7900); targeted pytest passed `21 passed`;
py_compile succeeded.

## 2026-05-15 — CPU reference full-attention prefill oracle

Added the first retained correctness oracle required by `docs/PREFILL.md` for
native full-attention prefill bring-up.

Changes:
- added torch-free NumPy `full_attn_prefill(...)` in
  `hipengine/kernels/cpu_reference/ops.py`;
- oracle models append-then-attend causal GQA over dense or paged key/value
  cache arrays, supports BF16-bit `uint16` cache inputs, per-row positions and
  context counts, and applies the decode-compatible sigmoid gate before the
  FP16/default output cast;
- registered the oracle under generic `fp16` and exact
  `(cpu_reference, full_attn_prefill, w4_paro, qwen35_causal_gqa_gate_fp16)`
  keys;
- added a tiny causal-GQA fixture and direct paged/BF16-bit test coverage.

Validation:

```bash
python3 -m pytest tests/test_cpu_reference.py -q
python3 -m pytest tests/test_cpu_reference.py tests/test_kernel_registry.py -q
python3 -m py_compile hipengine/kernels/cpu_reference/ops.py hipengine/kernels/cpu_reference/__init__.py tests/test_cpu_reference.py
```

Result: CPU-reference tests passed (`7 passed`); registry bundle passed
(`11 passed`); py_compile succeeded.

## 2026-05-15 — Batched full-attention prelude bring-up

Landed the first native full-attention prefill component: bulk Q/K/V prelude
plumbing and vector-position Q/K head RMSNorm + RoPE. This is a native prelude
piece only; no retained prefill perf claim yet because the causal prefill
attention kernel and grouped MoE are still missing.

Lineage/drift check before kernel work:

```bash
python3 scripts/check_lineage.py --kind kernel --diff stat
```

Result: parent kernel lineage reports expected DRIFT in `qwen35_expert.hip`,
`smoke.hip`, and `paroquant_kernels.py` (latest parent HEAD `5d8f496`); no code
was copied from those drifted new kernels for this hipENGINE-local vector RoPE
variant.

Changes:
- added `qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16`, a grid
  `(num_q_heads + num_kv_heads, tokens)` vector-position variant that reads
  `positions[token]` instead of scalar `position_ptr[0]`;
- kept scalar-position wrapper for decode/oracle paths and registered the vector
  variant under `qwen35_positions_f32_bf16`;
- updated FP16/BF16 full-attention Q/K projection helpers to split multi-token
  Q and K projections into separate contiguous `[T, ...]` buffers instead of
  using the c=1 dual-output layout;
- updated FP16/BF16 QKV prepare helpers to cast `tokens * kv_width` K elements
  and call the vector-position RoPE wrapper for `tokens > 1`;
- extended the rotary smoke to validate mixed vector positions (`positions=[1,0]`)
  and updated `docs/KERNELS.md` / `docs/PREFILL.md` inventory status.

Validation:

```bash
python3 -m pytest tests/test_qwen35_rotary_plan.py tests/test_qwen35_decode_state.py -q
python3 -m py_compile hipengine/kernels/hip_gfx1100/rotary/qwen35_rotary.py hipengine/runtime/qwen35_paro.py scripts/smoke.py tests/test_qwen35_rotary_plan.py tests/test_qwen35_decode_state.py
python3 scripts/smoke.py --mode qwen35-rotary-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/check_fixtures.py
python3 -m pytest tests/test_qwen35_rotary_plan.py tests/test_qwen35_decode_state.py tests/test_cpu_reference.py -q
```

Result: targeted pytest passed (`32 passed`); rotary HIP smoke passed with
`vector_position_max_abs=2.38e-07`; fixture checks passed; combined pytest
passed (`39 passed`).

## 2026-05-15 — Native causal full-attention prefill kernel

Landed the first gfx1100 native append-then-attend full-attention prefill kernel.
This is still not a retained prefill performance path: grouped/compact MoE and
final full-layer orchestration remain open.

Changes:
- added `qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans(...)`, registered as
  `(hip_gfx1100, full_attn_prefill, w4_paro, qwen35_causal_gqa_gate_fp16)`;
- kernel consumes row-major query `[rows, q_heads, head_dim]`, BF16 paged KV
  cache, FP16 gate, row-shaped context spans, optional `row_positions`, and
  writes post-sigmoid-gate FP16 output `[rows, q_heads * head_dim]`;
- added `Qwen35ParoDecodeState.prefill_full_attention_gqa_gate_fp16(...)` wrapper
  for future full-attention prefill orchestration;
- added CPU-reference-backed HIP smoke mode `qwen35-paged-attn-prefill-hip`;
- updated tests and kernel catalog/PREFILL inventory.

Validation:

```bash
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_decode_state.py -q
python3 -m py_compile hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.py hipengine/kernels/hip_gfx1100/attention/__init__.py hipengine/runtime/qwen35_paro.py scripts/smoke.py tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_decode_state.py
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_decode_state.py tests/test_cpu_reference.py -q
```

Result: targeted pytest passed (`33 passed`); HIP smoke passed with
`prefill_gate_fp16_max_abs=0`, `prefill_gate_fp16_mismatch=0`; combined pytest
passed (`40 passed`).

Profiler evidence for the new kernel:

```bash
rm -rf /tmp/hipengine-prefill-prof
rocprofv3 --kernel-trace --output-format csv --output-directory /tmp/hipengine-prefill-prof --output-file prefill -- \
  python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Result: `/tmp/hipengine-prefill-prof/prefill_kernel_trace.csv` contains
`qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel`; computed duration
`End_Timestamp - Start_Timestamp = 11200 ns`, `Workgroup_Size_X=256`,
`Grid_Size_Y=3`.

## 2026-05-15 — Full-attention prefill KV append wrapper

Added the runtime wrapper needed to wire batched full-attention prefill through
the existing row-shaped paged KV writer.

Changes:
- added `Qwen35ParoDecodeState.append_full_attention_kv_fp16_batch(...)`, calling
  `qwen35_write_paged_kv_mixed_value_fp16_batch_spans(...)` over prompt rows;
- added unit coverage to ensure the prefill path uses the batch writer with
  row-shaped spans.

Validation:

```bash
python3 -m pytest tests/test_qwen35_decode_state.py -q
python3 -m py_compile hipengine/runtime/qwen35_paro.py tests/test_qwen35_decode_state.py
```

Result: `31 passed`; py_compile succeeded.

## 2025-05-15 — Weight index speedup + HF hub resolution + E2E generation

### Problem
`load_weight_index` used `safetensors.safe_open` + per-tensor `get_slice` to read metadata.
For `z-lab/Qwen3.5-35B-A3B-PARO` (single 20GB shard, 93,996 tensors, 12.4MB header), this
took 120s+. Also, `load_weight_index` only accepted filesystem paths, not HF model IDs.

### Fix
- Direct binary header parsing: read 8-byte header length, read header bytes, `json.loads`.
  0.6s for 94K tensors (200× speedup).
- `_resolve_hf_hub_path()`: falls back to `huggingface_hub.snapshot_download(local_files_only=True)`
  when the path doesn't exist on disk, so users can pass `z-lab/Qwen3.5-35B-A3B-PARO` directly.
- `LLM._load_model_metadata()` stores the resolved filesystem path in `self.model` so the
  tokenizer and runner get a real directory.

### E2E generation verified
```bash
LD_LIBRARY_PATH=/home/lhl/miniforge3/lib/python3.10/site-packages/_rocm_sdk_devel/lib \
  python3 -c "
from hipengine import LLM, SamplingParams
llm = LLM('z-lab/Qwen3.5-35B-A3B-PARO', quant='w4_paro')
out = llm.generate('The capital of France is', SamplingParams(max_tokens=32))
print(out)
"
```
Output: `[' Paris.\nThe capital of France is Paris.\nThe capital of France is...']` ✓

### Runtime note
Conda `_rocm_sdk_devel` installs hipcc 7.13 which compiles against `libamdhip64.so.7`.
The system has ROCm 6.2 (`libamdhip64.so.6`). Kernels load fine with:
`LD_LIBRARY_PATH=/home/lhl/miniforge3/lib/python3.10/site-packages/_rocm_sdk_devel/lib`

### 0.8B dense model assessment
`z-lab/Qwen3.5-0.8B-PARO` is `Qwen3_5ForConditionalGeneration` (dense, not MoE).
- 24 layers, 18 linear_attention + 6 full_attention
- Dense MLP: gate_proj/up_proj/down_proj with PARO quant (no router/experts/shared_expert)
- Same weight name prefix as 35B (`model.language_model.layers.*`)
- Same linear/full attention structure as 35B layers
- Main gap: no dense MLP execution path registered; MoE runner hardcodes expert dispatch
- Supporting it requires a dense model plugin + dense MLP kernel chain, but the attention
  and PARO dequant kernels are shared.

## 2026-05-15 — Grouped MoE prefill metadata kernels

Started task #13 by landing the grouped/compact MoE metadata and packed-hidden
gather layer from the parent route. This is not yet the retained grouped MoE
prefill route: compact WMMA/GEMM expert math, weighted-lane accumulation,
remaining c1 metadata helpers, and runtime orchestration are still open.

Source lineage:
- parent source: `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip`
- source commit observed before port: `nano-vllm-amd@5d8f496`
- port subset: `qwen35_moe_group_count_kernel`,
  `qwen35_moe_group_prefix_kernel`, `qwen35_moe_group_scatter_kernel`,
  `qwen35_moe_group_scatter_gather_kernel`,
  `qwen35_moe_gather_packed_hidden_kernel`, and
  `qwen35_moe_wmma_tile_map_kernel`.

Changes:
- added `hipengine/kernels/hip_gfx1100/moe/group_scatter.{hip,py}`;
- registered grouped metadata/gather keys:
  `moe_group_count`, `moe_group_prefix`, `moe_group_scatter`,
  `moe_group_scatter_gather`, `moe_gather_packed_hidden`, and
  `moe_wmma_tile_map`;
- added `qwen35-moe-group-scatter-hip` smoke covering count/prefix,
  scatter+gather, separate gather, and compact WMMA tile-map metadata;
- updated `docs/KERNELS.md` and `docs/PREFILL.md` status to show this subset as
  landed while keeping retained MoE prefill blocked on expert kernels.

Validation:

```bash
python3 -m pytest tests/test_qwen35_moe_group_scatter_plan.py -q
python3 -m pytest tests/test_qwen35_moe_group_scatter_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/smoke.py --mode qwen35-moe-group-scatter-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 -m py_compile hipengine/kernels/hip_gfx1100/moe/group_scatter.py hipengine/kernels/hip_gfx1100/moe/__init__.py scripts/smoke.py tests/test_qwen35_moe_group_scatter_plan.py
```

Result: plan tests passed (`3 passed`); combined targeted tests passed
(`34 passed`); HIP smoke reported `prefix_match=True`, `lane_match=True`,
`expert_match=True`, `weight_match=True`, `packed_match=True`, `tile_match=True`.

Profiler evidence:

```bash
rm -rf /tmp/hipengine-moe-group-prof
rocprofv3 --kernel-trace --output-format csv --output-directory /tmp/hipengine-moe-group-prof --output-file moe_group -- \
  python3 scripts/smoke.py --mode qwen35-moe-group-scatter-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Result: `/tmp/hipengine-moe-group-prof/moe_group_kernel_trace.csv` contains the
new kernels with computed durations: group_count `6640 ns`, group_prefix
`11601 ns`, group_scatter_gather `11241 ns`, gather_packed_hidden `5360 ns`,
wmma_tile_map `2561 ns`.

## 2026-05-15 — Grouped MoE prefill fallback route wiring

Continued task #13 by wiring a native grouped-MoE prefill fallback over packed
sorted lanes. This is a correctness/route milestone, not a retained throughput
claim: compact WMMA gate/up and down kernels remain the current-OPTIMAL gap.

Changes:
- added HIP runtime `memset` / `memset_async` helpers for device-side grouped
  metadata zeroing;
- extended PARO combine kernels with BF16/FP16 `weighted_lanes_sum_out` and
  batched `shared_gate_combine_residual` wrappers;
- registered `moe_prefill/w4_paro/qwen35_grouped_compact` as the promoted
  grouped route and `moe_prefill/w4_paro/qwen35_selected_c1_rows` as an
  oracle/fallback key;
- added `Qwen35ParoGroupedMoeScratch` plus
  `run_moe_grouped_compact_{bf16,fp16}` using router top-k → group
  count/prefix/tile-map/scatter-gather → sorted-lane selected GEMV fallback →
  fused activation/down rotation → weighted-lane accumulation → shared expert →
  batched shared-gate residual combine;
- changed multi-token linear-attention+MoE layer orchestration to use grouped
  compact prefill instead of the c1 selected-row path;
- extended combine and route tests/smokes and updated the prefill/kernel docs.

Validation:

```bash
python3 -m pytest tests/test_paro_combine_plan.py tests/test_qwen35_moe_group_scatter_plan.py tests/test_qwen35_decode_state.py -q
python3 -m py_compile hipengine/core/hip.py hipengine/kernels/hip_gfx1100/fused/paro_combine.py hipengine/kernels/hip_gfx1100/fused/__init__.py hipengine/kernels/hip_gfx1100/moe/prefill.py hipengine/kernels/hip_gfx1100/moe/__init__.py hipengine/runtime/qwen35_paro.py hipengine/runtime/__init__.py scripts/smoke.py tests/test_paro_combine_plan.py tests/test_qwen35_moe_group_scatter_plan.py tests/test_qwen35_decode_state.py
python3 scripts/smoke.py --mode paro-combine-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode qwen35-moe-group-scatter-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
```

Result: targeted tests passed (`37 passed`); py_compile succeeded;
`paro-combine-hip` reported zero mismatches for weighted lanes and batched
shared residual combine in BF16/FP16; grouped metadata smoke still passes.

Profiler evidence:

```bash
rm -rf /tmp/hipengine-moe-combine-prof
rocprofv3 --kernel-trace --output-format csv --output-directory /tmp/hipengine-moe-combine-prof --output-file moe_combine -- \
  python3 scripts/smoke.py --mode paro-combine-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Result: `/tmp/hipengine-moe-combine-prof/moe_combine_kernel_trace.csv`
contains `weighted_lanes_inverse_kernel`, `weighted_lanes_sum_out_kernel` for
BF16/FP16, and `shared_gate_combine_residual_batch_out_kernel` for BF16/FP16.
Representative durations: BF16 weighted-lanes sum `2080 ns`, FP16 weighted-lanes
sum `2000 ns`, BF16 shared batch `2200 ns`, FP16 shared batch `2120 ns`.

Remaining task #13 gaps: compact WMMA expert kernels
(`gemm_awq_selected_dual_pack8_wmma_compact_kernel` and
`gemm_awq_selected_pack8_wmma_compact_kernel`) are not ported yet, so grouped
MoE prefill has no retained performance artifact.

## 2026-05-15 — Compact AWQ WMMA MoE prefill expert kernels

Ported the parent compact-buffer AWQ WMMA expert kernels for the grouped MoE
prefill route. This closes the main task #13 kernel gap, while still avoiding a
retained throughput claim until final single-request native prefill orchestration
and benchmark artifact closure.

Source lineage:
- parent source: `~/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py`
- source commit observed before port: `nano-vllm-amd@5d8f496`
- port subset: `gemm_awq_selected_dual_pack8_wmma_compact_kernel` and
  `gemm_awq_selected_pack8_wmma_compact_kernel`.

Changes:
- added `hipengine/kernels/hip_gfx1100/wmma/paro_awq_wmma.{hip,py}` with BF16
  and FP16 raw-pointer wrappers;
- registered compact dual/single AWQ WMMA variants under `awq_wmma`;
- changed `run_moe_grouped_compact_{bf16,fp16}` to use compact WMMA gate/up and
  down kernels over grouped packed lanes;
- pre-cleared `tile_expert` to `-1` so runtime can safely launch over the
  preallocated max WMMA tile budget without a host readback of `wmma_total`;
- added `paro-awq-wmma-compact-hip` smoke and unit registration/build tests.

Validation:

```bash
python3 -m pytest tests/test_paro_awq_wmma_plan.py tests/test_qwen35_decode_state.py tests/test_paro_combine_plan.py tests/test_qwen35_moe_group_scatter_plan.py -q
python3 -m py_compile hipengine/kernels/hip_gfx1100/wmma/paro_awq_wmma.py hipengine/kernels/hip_gfx1100/wmma/__init__.py hipengine/runtime/qwen35_paro.py scripts/smoke.py tests/test_paro_awq_wmma_plan.py tests/test_qwen35_decode_state.py
python3 scripts/smoke.py --mode paro-awq-wmma-compact-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode paro-combine-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
```

Result: targeted tests passed (`40 passed`); py_compile succeeded; compact WMMA
smoke reported `dual_mismatch=0`, `single_mismatch=0`, `dual_fp16_mismatch=0`,
`single_fp16_mismatch=0`; combine smoke still reported zero mismatches.

Profiler evidence:

```bash
rm -rf /tmp/hipengine-wmma-prof
rocprofv3 --kernel-trace --output-format csv --output-directory /tmp/hipengine-wmma-prof --output-file wmma -- \
  python3 scripts/smoke.py --mode paro-awq-wmma-compact-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Result: `/tmp/hipengine-wmma-prof/wmma_kernel_trace.csv` contains compact WMMA
kernels for BF16 and FP16. Computed durations: BF16 dual `10520 ns`, BF16 single
`6361 ns`, FP16 dual `6760 ns`, FP16 single `5161 ns`.

## 2026-05-15 — Single-request native prefill wired into generation/bench

Continued task #14 by wiring the final single-request native prefill route into
resident generation and benchmark entrypoints. This is a correctness milestone,
not a retained throughput row: the native path is accepted against serial and
parent fixture gates, but current prefill timing is slower than the serial c=1
fixture baseline.

Changes:
- `Qwen35ParoResidentSession.prefill_native(...)` now runs the full native
  single-request path by default (`single_request_native_full`) and keeps
  `require_full_native=False` as the explicit legacy/oracle path only;
- added full-layer native orchestration across linear-attention and
  full-attention layers, including grouped/compact MoE tails and decode-scratch
  restoration;
- added the single-request prompt KV writer
  `qwen35_write_paged_kv_mixed_value_fp16_prompt_spans(...)` so prompt rows are
  appended into one request cache instead of the row-major c>N cache layout;
- changed `Qwen35ParoOneTokenGenerator` and `scripts/qwen35_paro_bench.py` to
  use `prefill_native(...)` by default, with serial prefill exposed only as an
  explicit diagnostic mode;
- added `scripts/qwen35_native_prefill_fixture_gate.py` to compare native vs
  serial resident full lm-head logits with KL/top-1 gates on the parent 512/32
  fixture;
- updated benchmark rollup/docs and retained the accepted correctness artifact
  at `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json`.

Validation:

```bash
python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_qwen35_decode_state.py tests/test_generation_qwen35_paro.py tests/test_qwen35_native_prefill_boundary.py tests/test_qwen35_paged_kv_write_plan.py -q
python3 -m py_compile hipengine/kernels/hip_gfx1100/attention/paged_kv_write.py hipengine/kernels/hip_gfx1100/attention/__init__.py hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py hipengine/generation/qwen35_paro.py scripts/qwen35_paro_bench.py scripts/qwen35_e2e_correctness.py scripts/qwen35_native_prefill_correctness.py scripts/qwen35_native_prefill_boundary.py scripts/qwen35_native_prefill_fixture_gate.py scripts/smoke.py
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/qwen35_native_prefill_correctness.py --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd --token-id 9707 --prompt-length 4 --max-layers 40 --json /tmp/hipengine-native-prefill-ml40-pl4.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json
python3 scripts/qwen35_paro_bench.py --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd --token-id 9707 --prompt-length 512 --decode-tokens 32 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/hipengine-qwen35-native-prefill-bench-512-32.json
```

Result: targeted tests passed (`62 passed`); py_compile succeeded; prefill
smoke reported `prefill_gate_fp16_max_abs=0` and mismatch `0`; 40-layer
prompt-length-4 serial/native seed+decode correctness passed. The 512/32 parent
fixture gate passed with generated IDs matching serial and parent, `max_kl=0.016753961542286394`,
`mean_kl=0.0029413754094110554`, top-1 agreement `1.0`, and finite logits.
Diagnostic timings: fixture native prefill `512 / 11.19840783206746 = 45.72 tok/s`,
repeated-token bench prefill `46.956 tok/s`, decode `101.607 tok/s`; no
performance row promoted because serial c=1 fixture prefill is `117.24 tok/s`
and parent prefill is `2682.66 tok/s`.

Profiler evidence:

```bash
rm -rf /tmp/hipengine-prefill-smoke-prof
rocprofv3 --kernel-trace --output-format csv --output-directory /tmp/hipengine-prefill-smoke-prof --output-file prefill_smoke -- \
  python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

Result: `/tmp/hipengine-prefill-smoke-prof/prefill_smoke_kernel_trace.csv`
contains the final prompt KV writer and native full-attention prefill kernels:
`qwen35_write_paged_kv_mixed_value_prompt_position_tensor_kernel<_Float16>`
`12078 ns` and `qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel` `26036 ns`.
Grouped MoE/compact WMMA profiler evidence remains in the task #13 entries
above.

Remaining follow-up: `PrefillConfig` chunk-size knobs still execute the
unchunked path; non-zero chunk-equivalence sweeps should be implemented before
using chunking to reduce scratch memory. Compact c>N native prompt slabs remain
task #15.

Lineage check after the prompt-writer kernel edit:

```bash
python3 scripts/check_lineage.py --kind kernel --diff stat
```

Result: reports known parent drift in `qwen35_expert.hip`, `smoke.hip`, and
`paroquant_kernels.py` from nano-vllm-amd head `5d8f496`; no additional hipENGINE
source-lineage manifest change needed for the local prompt-writer wrapper.

## 2026-05-15 — Compact c>N prompt slab metadata and blocker artifact

Started task #15 by adding the host-side compact prompt slab contract and
scheduler bucketization needed before native c>N prefill kernels can be wired.
This is a metadata/blocker milestone only; no native packed prefill kernels are
retained or benchmarked yet.

Changes:
- added `CompactPromptBucket` and `CompactPromptSlab` host descriptors with
  flattened token rows, absolute positions, append/context counts,
  `cu_seqlens_q`, `cu_seqlens_k`, row-to-request metadata, and row-shaped block
  tables;
- added `ResidentBatchScheduler.bucketize_by_block_count(...)` and
  `next_compact_prefill_slabs(...)`, which emits one compact slab per uniform
  block-table length and advances prompt cursors without falling back to
  per-request work items;
- added `Qwen35ParoResidentSession.prefill_native_packed(slab)` as a fail-closed
  runtime entrypoint that validates slab/session shape and reports the remaining
  native-kernel blockers instead of silently using the serial prompt loop;
- added `scripts/qwen35_native_compact_prefill_plan.py` and blocked artifact
  `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-compact-c8-blocked.json`;
- updated `docs/PREFILL.md`, benchmark rollup, and changelog to mark compact
  metadata as landed while keeping compact execution blocked.

Validation:

```bash
python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_qwen35_resident_batch_layout.py -q
python3 -m py_compile hipengine/generation/batch_scheduler.py hipengine/generation/__init__.py hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_native_compact_prefill_plan.py
python3 scripts/qwen35_native_compact_prefill_plan.py --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --batch-size 8 --prompt-length 8 --chunk-size 8 --block-size 256 --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-compact-c8-blocked.json
```

Result: targeted tests passed (`33 passed`); py_compile succeeded. The c=8
planning artifact contains one 64-row compact slab with `cu_seqlens_q/k =
[0,8,16,24,32,40,48,56,64]`, `block_count=1`, and prompt cursors advanced to 8
for all requests. Status remains `blocked` because segment-aware linear-attn
conv/GDN state kernels, varlen/block-diagonal full-attn prefill via
`cu_seqlens`, packed final-row sampling, and per-request state/KV commit are not
wired.

## 2026-05-15 — Segment-aware linear-attention prefill state kernels

Continued compact c>N prefill task #16 by adding the segment-aware linear
attention state kernels needed by packed prompt slabs. These are kernel-level
correctness/profiler gates only; `prefill_native_packed(slab)` remains blocked
until varlen full-attention and final-row state/KV commit are wired.

Lineage note:
- `scripts/check_lineage.py --kind kernel --diff stat` reports parent drift at
  `nano-vllm-amd@b95eaa5` adding tree/speculative conv/GDN kernels. Those tree
  kernels are parent-indexed decode/speculation kernels, not the compact prompt
  slab `cu_seqlens` ABI. This task implemented the hipENGINE slab ABI directly
  over `cu_seqlens` + state-slot indices.

Changes:
- added `qwen35_linear_attn_conv_prefill_segments_f32(...)`, which consumes
  packed `[T_total, qkv_width]` rows, `cu_seqlens`, and per-segment state slots,
  writes packed conv outputs, and commits each segment's tail conv state without
  reading neighboring request rows;
- added `qwen35_gdn_prefill_recurrent_segments_k2_f32(...)`, which consumes
  packed GDN Q/K/V/beta/decay rows plus `cu_seqlens` and commits each segment's
  recurrent state slot independently;
- added NumPy CPU references and unit tests for segment-state isolation;
- added `qwen35-linear-attn-segments-hip` smoke and profiler artifact
  `benchmarks/results/2026-05-15-hipengine-qwen35-linear-attn-segment-prefill-accepted.json`;
- updated `docs/KERNELS.md`, `docs/PREFILL.md`, compact blocked artifact text,
  benchmark rollup/changelog, and the fail-closed packed-prefill blocker list to
  show segment kernels landed while packed orchestration remains blocked.

Validation:

```bash
python3 -m pytest tests/test_qwen35_linear_attn_conv_plan.py tests/test_qwen35_linear_attn_gdn_plan.py tests/test_cpu_reference.py -q
python3 scripts/smoke.py --mode qwen35-linear-attn-segments-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode qwen35-linear-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
rocprofv3 --kernel-trace --output-format csv --output-directory /tmp/hipengine-linear-segments-prof --output-file linear_segments -- python3 scripts/smoke.py --mode qwen35-linear-attn-segments-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 -m pytest tests/test_qwen35_linear_attn_conv_plan.py tests/test_qwen35_linear_attn_gdn_plan.py tests/test_cpu_reference.py tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py -q
python3 scripts/check_lineage.py --kind kernel --diff stat
```

Result: targeted tests passed (`15 passed`, then `48 passed`). Segment HIP smoke
reported `segment_conv_out_max_abs=1.86e-09`, `segment_conv_state_max_abs=0`,
`segment_gdn_out_max_abs=1.86e-09`, and `segment_gdn_state_max_abs=9.31e-10`.
Existing single-request linear prefill smoke still passed. Profiler CSV contains
`qwen35_linear_attn_conv_prefill_segments_kernel` (`5800 ns`),
`qwen35_linear_attn_conv_prefill_segments_state_kernel` (`2200 ns`), and
`qwen35_gdn_prefill_recurrent_k2_segments_kernel` (`5480 ns`) on W7900.

## 2026-05-15 — Varlen block-diagonal full-attention prefill kernel

Continued compact c>N prefill task #17 by adding the append-then-attend
varlen/block-diagonal full-attention prefill ABI. This is a kernel-level
correctness/profiler gate; `prefill_native_packed(slab)` remains fail-closed
until final-row sampling/state commit and packed orchestration are wired.

Changes:
- added `qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans(...)`, keyed
  as `full_attn_prefill/w4_paro/qwen35_varlen_causal_gqa_gate_fp16`;
- the kernel consumes row-shaped `KVLiveSpans`, `cu_seqlens_q`, and
  `cu_seqlens_k`, clamps each query row to its request segment and causal
  position, and reads only that row's request-owned paged block table;
- added CPU `full_attn_prefill_varlen(...)` oracle and unit coverage comparing
  packed segments against per-request `full_attn_prefill(...)` outputs;
- added `qwen35-paged-attn-prefill-varlen-hip` smoke and artifact
  `benchmarks/results/2026-05-15-hipengine-qwen35-varlen-full-attn-prefill-accepted.json`;
- refreshed compact c=8 blocked artifact, docs, benchmark rollup/changelog, and
  fail-closed packed-prefill blockers to show varlen full-attn landed while
  final packed commit remains open.

Validation:

```bash
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py tests/test_cpu_reference.py -q
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-varlen-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
rocprofv3 --kernel-trace --output-format csv --output-directory /tmp/hipengine-varlen-prefill-prof --output-file varlen_prefill -- python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-varlen-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py tests/test_cpu_reference.py tests/test_qwen35_resident_batch_layout.py -q
python3 -m py_compile hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.py hipengine/kernels/hip_gfx1100/attention/__init__.py hipengine/kernels/cpu_reference/ops.py hipengine/kernels/cpu_reference/__init__.py hipengine/runtime/qwen35_paro_runner.py scripts/smoke.py scripts/qwen35_native_compact_prefill_plan.py tests/test_cpu_reference.py tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_resident_batch_layout.py
python3 scripts/check_lineage.py --kind kernel --diff stat
```

Result: targeted tests passed (`13 passed`, then `36 passed`); py_compile
succeeded; lineage check still reports parent drift at `nano-vllm-amd@b95eaa5`
(tree/speculative linear-attn kernels, not this varlen full-attn ABI). Varlen HIP smoke reported `varlen_prefill_gate_fp16_max_abs=0` and
`varlen_prefill_gate_fp16_mismatch=0`. Existing single-request prefill smoke
still passed. Profiler CSV contains prompt KV writer (`6880 ns`) and
`qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_kernel` (`21520 ns`) on
W7900.

## 2026-05-15 — Compact prefill physical slots and final-row commit helper

Started task #18. This is a partial logical unit only: final-row sampling/state
metadata helpers landed, but `prefill_native_packed(slab)` still rejects because
packed layer orchestration and generated-token equality gates are not wired.

Changes:
- added optional `CompactPromptSlab.slot_ids` and `physical_slot_ids`; the
  scheduler now records physical slots alongside stable request ids when
  emitting compact slabs;
- added `_packed_prefill_final_rows(...)` and
  `_commit_packed_prefill_final_rows(...)` on `Qwen35ParoResidentSession` to
  copy each request segment's tail hidden row into its physical batch slot,
  update position/context metadata, and sample each final row without a
  per-request prompt loop;
- refreshed compact c=8 blocked artifact/rollup/docs to show linear/full-attn
  kernels plus final-row commit helpers are landed, with packed orchestration
  still blocked.

Validation:

```bash
python3 -m py_compile hipengine/generation/batch_scheduler.py hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_native_compact_prefill_plan.py tests/test_generation_batch_scheduler.py tests/test_qwen35_resident_batch_layout.py
python3 -m pytest tests/test_generation_batch_scheduler.py tests/test_qwen35_resident_batch_layout.py -q
python3 scripts/qwen35_native_compact_prefill_plan.py --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --batch-size 8 --prompt-length 8 --chunk-size 8 --block-size 256 --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-compact-c8-blocked.json
```

Result: targeted scheduler/resident tests passed (`35 passed`). Compact c=8
planning artifact now includes `slot_ids` and reports the remaining blocker as
packed native layer orchestration/equality gates, not missing segment/full-attn
kernels or final-row commit helpers.

## 2026-05-15 — Native compact c>N prefill execution and equality gates

Completed task #18's retained compact prefill wiring: `prefill_native_packed(slab)`
now materializes compact slab device metadata, runs packed native layers, commits
one final hidden row per physical slot, and returns one seed sample per request.
Decode after the seed still uses `step_batch_serial`; this is therefore a
correctness milestone, not a throughput claim.

Changes:
- imported/wired segment-aware linear-attention conv/GDN wrappers into
  `Qwen35ParoDecodeState` packed prefill paths;
- added varlen/block-diagonal full-attention prefill layer orchestration using
  `qwen35_varlen_causal_gqa_gate_fp16` over row-shaped physical block tables;
- expanded resident prefill buffers to `max_sequence_length * max_batch_size`
  rows for compact slabs;
- `Qwen35ParoResidentSession.prefill_native_packed(slab)` now runs the packed
  native prefill path and commits final rows/samples via physical `slot_ids`;
- added `scripts/qwen35_batch_packed_prefill_correctness.py` and c=2/4/8
  accepted correctness artifacts under `benchmarks/results/`;
- updated docs/rollup/changelog to mark native compact prefill correctness as
  landed while keeping c-aware decode/throughput pending.

Validation:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py hipengine/generation/batch_scheduler.py scripts/qwen35_batch_packed_prefill_correctness.py scripts/qwen35_native_compact_prefill_plan.py
python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_generation_batch_scheduler.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 4 --max-layers 2 --batch-size 2 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json /tmp/packed-c2-l2.json
python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size 2 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json benchmarks/results/2026-05-15-hipengine-qwen35-c2-native-compact-prefill-correctness-accepted.json
python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size 4 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json benchmarks/results/2026-05-15-hipengine-qwen35-c4-native-compact-prefill-correctness-accepted.json
python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size 8 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json benchmarks/results/2026-05-15-hipengine-qwen35-c8-native-compact-prefill-correctness-accepted.json
```

Result: targeted tests passed (`67 passed`). The reduced c=2/max_layers=2 smoke
matched independent c=1 native prefill+decode. Full max_layers=40 prompt8 gates
passed for c=2, c=4, and c=8 with `generated_match=true` and
`finite_logits=true`. The c=8 compact slab ran as one 64-row native prefill slab
with `cu_seqlens_q/k=[0,8,16,24,32,40,48,56,64]` and slot ids `[0..7]`.

## 2026-05-15 — Prefill multiloop audit: load grouped MoE libraries once

Started `prefill-perf/run-20260515-141513` with audit-first policy and 300-iter
cap. Baseline before edits: native `single_request_native_full` 512/128 samples
`48.065, 47.986, 48.506` tok/s (median `48.065`); fixture correctness passed
but 4K/128 OOMed in prefill scratch allocation. `rocprofv3 --kernel-trace`
hung before emitting CSV for both full and max_layers=4 Qwen3.5/PARO bench runs,
so I used a wrapper-count audit while keeping the profiler blocker logged.

Root cause found in the audit: `_load_kernel_libraries()` did not preload the
`group_scatter` or `wmma` libraries, while the grouped compact MoE path passed
`library=self.libraries`. `_library_for(..., "group_scatter"/"wmma")` therefore
returned `None`, making each grouped-MoE metadata/WMMA wrapper run
`build_hip(...)` / compiler-version resolution on demand. The 512 prefill path
made 1117 HIP wrapper calls; the missing two libraries accounted for ~240
per-layer grouped-MoE calls and nearly all of the 10.7s prefill time.

Changes:
- added `group_scatter` and `wmma` to `Qwen35ParoResidentSession`'s preloaded
  kernel library map;
- added a `device_synchronize()` at `Qwen35ParoResidentSession.close()` before
  freeing buffers. Once the accidental on-demand-build delays were gone, a
  serial fixture session could close with queued work and corrupt the next
  native session in the same process; synchronizing before free restores the
  fixture gate.

Validation:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro_runner.py
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128-iter2-run{1,2,3}.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate-iter2-fixed.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128-iter2-fixed.json
```

Result: 512/128 native prefill improved to `484.094, 482.057, 482.044` tok/s
(median `482.057`, ~10.0x vs baseline) with decode still ~101 tok/s. Fixture
gate passes after the close synchronization (`generated_match=true`,
`max_kl=0.0226`, top-1 `1.0`, native fixture prefill `1.0806s`,
`owned_device_bytes=1625645909`). 4K/128 still OOMs in grouped MoE scratch
allocation, matching the baseline blocker rather than a new regression; next
prefill-perf work should address shared/chunked prefill scratch so the 4K guard
can become active.

## 2026-05-15 — Prefill multiloop blocker: native fixture flakiness

After retaining the grouped-library preload fix, restarted the prefill loop as
`prefill-perf/run-20260515-154601` with a corrected 4K guard policy: 4K/128 is
attempted/logged every iteration but the known scratch OOM does not discard
unrelated 512 wins until a 4K-capable baseline exists.

New baseline is the committed preload state: 512/128 native prefill median
`482.057` tok/s. However, repeated fixture probes exposed that the native
single-request prefill correctness gate is flaky: some runs match the serial
resident/parent fixture, while others diverge after five decode tokens with the
native sequence repeating `[4, 220, 16, 15, 15, ...]`; failing runs show
`max_kl≈8.6-9.0` and top-1 agreement `~0.485`. Probes with `group_scatter`/`wmma`
loaded, popped back to on-demand loading, and even monkeypatched c1 MoE all show
the same pass/fail pattern, so this is a pre-existing native-prefill state/KV
or scratch determinism blocker rather than solely the preload fix. `HIP_LAUNCH_BLOCKING=1`
did not eliminate the flake.

Current loop status: blocked for further performance keeps until native prefill
fixture determinism is fixed or the gate is made repeat-deterministic. 4K/128
also remains blocked by prefill scratch OOM.

## 2026-05-15 — Native prefill determinism: 64-thread prefill attention

Fixed the native fixture flakiness blocker before resuming `prefill-perf`. The
root cause was localized to the full-attention prefill softmax kernel, not MoE,
linear-attention state, KV append, or session close ordering:

- native-vs-serial bisection showed the first pass/fail hidden divergence at
  layer 3, the first full-attention layer;
- layer-3 probes showed identical hidden input, Q/K/V/gate tensors, and appended
  BF16 KV cache between pass/fail runs;
- repeated launches of `prefill_full_attention_gqa_gate_fp16` on identical
  inputs in the same session produced different `gated_attn` outputs with the
  old 256-thread wrapper (repeat max abs roughly `0.05-0.39` depending run);
- switching the prefill wrappers to a 64-thread block (`threads=64`) made the
  repeated attention probe deterministic, and the shared-LDS size is now
  `max_context_len + threads` so short varlen/compact rows no longer
  under-allocate the per-thread reduction scratch.

Validation:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro_runner.py hipengine/runtime/qwen35_paro.py scripts/qwen35_native_prefill_fixture_gate.py
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-varlen-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
for i in $(seq 1 5); do python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/fixture-final-det-$i.json; done
for c in 2 4 8; do python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size $c --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json /tmp/packed-det-c$c.json; done
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/prefill-det-final-512.json
```

Result: smoke gates pass; fixture gate passed 5/5 repeats with native generated
IDs `[1739, 220, 16, 15, 15, 15, 15, 15, ...]`, max KL `0.00553-0.00570`, and
100% top-1 agreement; compact prompt8 gates still pass for c=2/4/8. The 512/128
prefill check after the determinism fix measured `479.755 tok/s`
(`prefill_seconds=1.06721`, decode `101.108 tok/s`), essentially flat vs the
post-preload `482.057 tok/s` baseline. The active loop remains paused until the
human asks to resume; next perf work can proceed with a repeat-stable native
prefill gate while 4K/128 scratch OOM remains a separate blocker.

## 2026-05-15 — Document native prefill determinism lesson

Added `docs/LESSONS-LEARNED.md` with the retained native-prefill flakiness
lesson from commit `4f252cf`: layer-3/full-attention localization, ruled-out
state/MoE/KV causes, repeat-launch attention probe, 64-thread prefill attention
fix, shared scratch sizing rule, validation evidence, and a checklist for future
flaky native-prefill correctness failures.

Validation: re-read `docs/LESSONS-LEARNED.md` end-to-end; docs-only change, no
GPU run needed.

## 2026-05-15 — Prefill multiloop resumed: determinism unblock verified

Resumed `prefill-perf/run-20260515-154601` after committing the native-prefill
attention determinism fix (`4f252cf`) and lesson doc (`fab7d8c`). First verify
attempt hit the expected stale-cache condition for the changed attention source
under `--require-cached-build`; prebuilt the new cached artifact outside any
profiler with:

```bash
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import build_qwen35_paged_attn_decode
compiler_version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
build_qwen35_paged_attn_decode(load=False, compiler_version=compiler_version, profile='decode')
PY
```

Verification commands:

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128-run2.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128-run3.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: 512/128 samples `485.424`, `481.047`, `479.101` tok/s (median
`481.047`, effectively flat / -0.21% vs current loop baseline `482.057`); fixture
gate passed with `native_owned_device_bytes=1625645909`, native prefill
`1.06309s`, max KL `0.00565`, top-1 `1.0`; 4K/128 remains the known baseline OOM
in grouped MoE prefill scratch allocation (`HIP error 2: out of memory`). This
iteration is a log-only correctness-unblock verification, not a primary-metric
keep; next active work should target the 4K scratch OOM / scratch reuse before
more 512 launch tuning.

## 2026-05-15 — Prefill multiloop iter 3: share native prefill scratch

Iteration `prefill-perf/run-20260515-154601` targeted the 4K/128 OOM. Root
cause was per-layer growth of native prefill scratch: each layer kept its own
4096-row linear/full-attention and grouped-MoE workspace until prefill finished,
with the OOM occurring in grouped MoE `gate_up` allocation. Implemented a
session-level transient `prefill_workspace` and shared linear/full/MoE prefill
scratch across layers; `_restore_decode_scratch_after_prefill()` frees this
transient workspace before recreating the per-layer c=1 decode scratch.

Validation commands:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro_runner.py hipengine/runtime/qwen35_paro.py
python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size 2 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json /tmp/shared-prefill-packed-c2.json
python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size 4 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json /tmp/shared-prefill-packed-c4.json
python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size 8 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json /tmp/shared-prefill-packed-c8.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-iter3-512-run1.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-iter3-512-run2.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-iter3-512-run3.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: 512/128 samples `509.518`, `508.554`, `507.502`, `508.086` tok/s
(median `508.320`, +5.45% vs current loop baseline `482.057`). Fixture gate
passed with `native_owned_device_bytes=1625645909`, native prefill `1.00693s`,
max KL `0.00534`, top-1 `1.0`; c=2/4/8 compact prompt8 gates passed. 4K/128 is
now runnable rather than OOM: `318.037 tok/s`, prefill `12.8790s`, decode
`101.714 tok/s`. This establishes the first 4K guard baseline; future active
keeps should preserve at least 95% of `318.037 tok/s` until a faster 4K baseline
is retained.

## 2026-05-15 — Prefill multiloop iter 4: skip decode-scratch restore loop rejected

Tried to remove the `_restore_decode_scratch_after_prefill()` per-layer c=1
scratch reserve loop now that native prefill scratch lives in a transient shared
workspace. The intent was to reduce post-prefill orchestration overhead while
leaving decode scratch untouched.

Validation commands:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro_runner.py hipengine/runtime/qwen35_paro.py
python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/iter4-fixture.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter4-512-run1.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter4-512-run2.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter4-512-run3.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter4-4k-128.json
```

Results: correctness stayed clean (`max_kl=0.00567`, top-1 `1.0`, fixture
`native_owned_device_bytes=1625645909`) and 4K stayed runnable at `317.711 tok/s`
(>=95% of the new `318.037` guard baseline), but 512/128 samples were
`505.890`, `504.797`, `505.787` tok/s (median `505.787`), below the retained
`508.320` current. Decision: reject/revert this micro-cleanup; the reserve loop
is not the current bottleneck.

## 2026-05-15 — Prefill multiloop iter 5 audit: retained path launch counts

Audit-only iteration after retaining shared prefill scratch. Wrapped hipENGINE
Python kernel-wrapper callables and ran `prefill_native(..., sample=False)` at
T=512 and T=4096 to count host-side launches on the current retained path:

```bash
python3 - <<'PY'
# wrapper-count audit; writes /tmp/prefill-audit-counts-{512,4096}.json
PY
```

Results: both T=512 and T=4096 issue `1113` wrapper calls for 40 layers. Top
families/counts are unchanged by sequence length: 80 transposed pack8 GEMV, 80
rotate1, 80 W8A16 shared-expert linears, 60 dense GEMV, 50 strided pack8 GEMV,
40 RMSNorm, 40 fp16->f32 casts, 40 add+rmsnorm, 40 router, 40 each grouped-MoE
metadata/scatter/WMMA/up/down/combine family, 30 linear-attention conv/GDN, and
10 full-attention prefill GQA gate calls. This confirms the remaining large 4K
slowdown is not launch-count growth; it is shape cost inside the 10 full-attention
prefill kernels and/or token-scaled MoE/projection work. Next code iteration
should target a parent-proven full-attention/projection path or profiling of
those 10 full-attention kernels rather than more Python restore-loop cleanup.

## 2026-05-15 — Prefill multiloop iter 6 skipped: restore audit metric

Skipped the planned full-attention thread-count trial before code changes because
iteration 5's audit-only `multiloop_log` incorrectly supplied the wrapper-call
count (`1113`) as the loop metric, which set `currentMetric` away from the
retained 512/128 throughput. Logged a skip with metric `508.320074` to restore
`currentMetric`/`bestMetric` before continuing optimization. No repo code change.

## 2026-05-15 — Prefill multiloop iter 7: shape-split prefill attention threads

Tuned the deterministic full-attention prefill softmax wrapper. A fixed
32-thread block improved 512/128 (`~515-517 tok/s`) but regressed 4K to
`236.212 tok/s`, below the new 95% 4K guard. Revised to a shape split:
`threads=32` for `max_context_len <= 1024`, otherwise retain the deterministic
64-thread path for long contexts. This keeps short-row one-wave overhead lower
without sacrificing 4K value-loop throughput.

Validation commands:

```bash
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import build_qwen35_paged_attn_decode
compiler_version=Path('/tmp/hipengine-hipcc-version.txt').read_text()
build_qwen35_paged_attn_decode(load=False, compiler_version=compiler_version, profile='decode')
PY
python3 -m py_compile hipengine/runtime/qwen35_paro_runner.py hipengine/runtime/qwen35_paro.py
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-varlen-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
for c in 2 4 8; do python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size $c --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json /tmp/iter7-packed-c$c.json; done
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter7-threshold-512-run1.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter7-threshold-512-run2.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter7-threshold-512-run3.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: 512/128 samples `514.232`, `516.042`, `516.431`, `516.961` tok/s
(median `516.236`, +1.56% vs retained `508.320`). Fixture gate passed with
`native_owned_device_bytes=1625645909`, native prefill `1.00033s`, max KL
`0.00487`, top-1 `1.0`; compact c=2/4/8 gates and attention smokes passed.
4K/128 stayed within guard at `316.586 tok/s` (99.54% of `318.037` baseline),
prefill `12.9381s`, decode `101.419 tok/s`.

## 2026-05-15 — Prefill multiloop iter 8: 16-thread short attention rejected

Tried changing the short-context full-attention prefill threshold from 32 threads
to 16 threads for `max_context_len <= 1024` while keeping 64 threads for long
contexts. Correctness stayed clean, but the primary metric regressed.

Validation commands:

```bash
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import build_qwen35_paged_attn_decode
compiler_version=Path('/tmp/hipengine-hipcc-version.txt').read_text()
build_qwen35_paged_attn_decode(load=False, compiler_version=compiler_version, profile='decode')
PY
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter8-512-run1.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter8-512-run2.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter8-512-run3.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: fixture passed (`max_kl=0.00584`, top-1 `1.0`,
`native_owned_device_bytes=1625645909`) and 4K stayed within guard at
`316.729 tok/s`, but 512/128 samples `508.483`, `507.023`, `507.921` tok/s
(median `507.921`) regressed below retained `516.236`. Decision: revert to the
32/64 shape split from iter 7; half-wave short blocks are too small.

## 2026-05-15 — Prefill multiloop iter 9: strided pack8 prefill projections rejected

Tried switching multi-token FP16 prefill projections from transposed
`qweight_pack8_decode` to original strided `.qweight` for full-attention Q/K and
linear-attention QKV/Z, following the parent optimal note that transposed pack8
is disabled on W7900. In hipENGINE this regressed badly, so the change is
rejected.

Validation commands:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter9-512-run1.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter9-512-run2.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter9-512-run3.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: fixture correctness stayed clean (`max_kl=0.00487`, top-1 `1.0`,
`native_owned_device_bytes=1625645909`) but native fixture prefill doubled to
`~2.03s`. 512/128 samples were `251.369`, `254.548`, `254.051`, `253.555` tok/s
(median `253.803`), far below retained `516.236`; 4K/128 also regressed to
`192.753 tok/s`, below the 95% guard floor (`302.135 tok/s`). Decision: revert;
hipENGINE's transposed projection kernels remain faster for this path despite
the parent engine's different optimal flag stack.

## 2026-05-15 — Prefill multiloop iter 10: 64-thread prefill projection GEMV

Tuned multi-token transposed pack8 prefill projections. The previous strided
layout trial showed transposed qweights are the right layout in hipENGINE, so
this iteration kept that layout and changed only the thread count for the
multi-token FP16 full-attention Q/K and linear-attention QKV/Z projections from
default `128` to `64`. These projections are one of the largest launch families
(80 calls at 512/4K), and 64-thread blocks are faster for the retained prefill
path.

Validation commands:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
for c in 2 4 8; do python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size $c --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json /tmp/iter10-packed-c$c.json; done
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter10-512-run1.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter10-512-run2.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter10-512-run3.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: 512/128 samples `530.738`, `530.202`, `529.673`, `529.768` tok/s
(median `529.985`, +2.66% vs retained `516.236`). Fixture gate passed with
`native_owned_device_bytes=1625645909`, native prefill `0.97720s`, max KL
`0.02207` (still below 0.05), top-1 `1.0`; compact c=2/4/8 prompt8 gates passed.
4K/128 improved to `322.359 tok/s` (new 4K guard baseline), prefill `12.7063s`,
decode `101.695 tok/s`.

## 2026-05-15 — Prefill multiloop iter 11: 64-thread full-attention V projection

Extended the retained 64-thread projection tuning to the remaining multi-token
full-attention V projection. Decode/c=1 keeps the default 128-thread strided
GEMV; only `tokens > 1` full-attention V prefill uses `threads=64`.

Validation commands:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
for c in 2 4 8; do python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size $c --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json /tmp/iter11-packed-c$c.json; done
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter11-512-run1.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter11-512-run2.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter11-512-run3.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: 512/128 samples `531.279`, `531.162`, `530.465`, `530.006` tok/s
(median `530.814`, +0.16% vs retained `529.985`). Fixture gate passed with
`native_owned_device_bytes=1625645909`, native prefill `0.97775s`, max KL
`0.01734`, top-1 `1.0`; compact c=2/4/8 prompt8 gates passed. 4K/128 stayed at
`322.359 tok/s`, preserving the iter-10 4K baseline.

## 2026-05-15 — Prefill multiloop iter 12: 64-thread output projection GEMVs

Extended the 64-thread prefill GEMV policy to multi-token output projections:
full-attention `o_proj` and linear-attention `out_proj` now pass `threads=64`
when `tokens > 1`, while c=1/decode remains at 128 threads. This targets the
remaining strided pack8 prefill projection launches after iter 10/11 handled Q/K,
V, and QKV/Z.

Validation commands:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
for c in 2 4 8; do python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size $c --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json /tmp/iter12-packed-c$c.json; done
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter12-512-run1.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter12-512-run2.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter12-512-run3.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: 512/128 samples `553.164`, `554.600`, `554.301`, `552.499` tok/s
(median `553.732`, +4.32% vs retained `530.814`). Fixture gate passed with
`native_owned_device_bytes=1625645909`, native prefill `0.94034s`, max KL
`0.01743`, top-1 `1.0`; compact c=2/4/8 prompt8 gates passed. 4K/128 improved
to `330.082 tok/s`, prefill `12.4090s`, decode `101.790 tok/s`.

## 2026-05-15 — Prefill multiloop iter 13: 128-thread shared-expert W8A16 rejected

Tried increasing shared-expert W8A16 prefill GEMVs from 64 to 128 threads for
`tokens > 1` while leaving c=1/decode at the existing 64-thread default. This
covers the shared-expert gate/up and down projections (80 launches for 40
layers), but it regressed the retained path.

Validation commands:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter13-512-run1.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter13-512-run2.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter13-512-run3.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: fixture passed (`max_kl=0.02090`, top-1 `1.0`,
`native_owned_device_bytes=1625645909`) and 4K remained above the 95% guard at
`324.879 tok/s`, but 512/128 samples `533.023`, `535.455`, `532.551`,
`533.145` tok/s (median `533.084`) regressed below retained `553.732`. Decision:
revert; shared-expert W8A16 should stay at 64 threads for prefill.

## 2026-05-15 — Prefill multiloop iter 14: 128-thread linear A/B dense rejected

Tried increasing the multi-token linear-attention A/B dense GEMVs from 64 to 128
threads while preserving c=1/decode defaults. These are the remaining 60 dense
GEMV launches in the 40-layer prefill path.

Validation commands:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter14-512-run1.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter14-512-run2.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter14-512-run3.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: fixture passed (`max_kl=0.02058`, top-1 `1.0`,
`native_owned_device_bytes=1625645909`) and 4K improved slightly to
`330.524 tok/s`, but 512/128 samples `552.942`, `553.533`, `553.068`,
`552.682` tok/s (median `553.005`) were below retained `553.732`. Decision:
revert; A/B dense GEMV stays at 64 threads.

## 2026-05-15 — Prefill multiloop iter 15: 256-thread router prefill

Tuned native router/top-k shared-gate prefill from 512 to 256 threads for
`tokens > 1` while leaving c=1/decode at the existing 512-thread default. Router
is one launch per layer, and the 257-output router/shared-gate shape benefits
from the smaller prefill block.

Validation commands:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
for c in 2 4 8; do python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size $c --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json /tmp/iter15-packed-c$c.json; done
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter15-512-run1.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter15-512-run2.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter15-512-run3.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: 512/128 samples `559.937`, `561.426`, `560.773`, `562.091` tok/s
(median `561.099`, +1.33% vs retained `553.732`). Fixture gate passed with
`native_owned_device_bytes=1625645909`, native prefill `0.92294s`, max KL
`0.01743`, top-1 `1.0`; compact c=2/4/8 prompt8 gates passed. 4K/128 improved
to `333.172 tok/s`, prefill `12.2940s`, decode `101.896 tok/s`.

## 2026-05-15 — Prefill multiloop iter 16: 128-thread router rejected (correctness)

Tried continuing the router/top-k shared-gate thread sweep by lowering multi-token
prefill from 256 to 128 threads while leaving c=1/decode at the existing default.
This looked faster in the repeated-token benchmark but broke the fixture gate,
so performance is invalid.

Validation commands:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: `py_compile` passed, but fixture failed (`generated_match=false`,
`kl_pass=false`, `top1_pass=false`, max KL `8.6823`, top-1 agreement `48.5%`,
`native_owned_device_bytes=1625645909`). Diagnostic-only benchmark output was
512/128 `568.846 tok/s` and 4K/128 `334.454 tok/s`, but both are invalid due to
correctness failure. Decision: revert to the retained 256-thread router prefill.

## 2026-05-15 — Prefill multiloop iter 17: FP16-input linear conv prefill

Added an FP16-input native linear-attention prefill convolution wrapper/kernel so
the FP16 qkv projection rows are converted inside the convolution kernel instead
of first materializing `scratch.qkv_f32` via a separate cast launch. The kernel
uses the same F32 conv math and writes the F32 conv state directly; c=1 decode
and segment prefill paths are unchanged.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.linear_attn.conv import build_qwen35_linear_attn_conv
build_qwen35_linear_attn_conv(load=False, require_cached=False)
PY
python3 -m py_compile hipengine/kernels/hip_gfx1100/linear_attn/conv.py hipengine/runtime/qwen35_paro.py scripts/smoke.py
python3 -m pytest tests/test_qwen35_linear_attn_conv_plan.py -q
python3 scripts/smoke.py --mode qwen35-linear-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
for c in 2 4 8; do python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size $c --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json /tmp/iter17-packed-c$c.json; done
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter17-512-run1.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter17-512-run2.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter17-512-run3.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: 512/128 samples `565.146`, `565.408`, `565.281`, `564.646` tok/s
(median `565.213`, +0.73% vs retained `561.099`). Fixture gate passed with
`native_owned_device_bytes=1625645909`, native prefill `0.91936s`, max KL
`0.01743`, top-1 `1.0`; compact c=2/4/8 prompt8 gates passed. The linear-attn
prefill smoke covered the new FP16 conv variant (`fp16_conv_out_max_abs=1.49e-08`,
`fp16_conv_state_max_abs=0`). 4K/128 stayed above guard at `333.328 tok/s`,
prefill `12.2882s`, decode `101.962 tok/s`.

## 2026-05-15 — Prefill multiloop iter 18: 512-thread FP16 conv rejected

Tried changing the new FP16-input linear-attention prefill conv wrapper from 256
to 512 threads/block. Correctness held and the first few runs looked neutral,
but the expanded 512/128 sample set regressed below the retained 256-thread
baseline.

Validation commands:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/linear_attn/conv.py hipengine/runtime/qwen35_paro.py scripts/smoke.py
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.linear_attn.conv import build_qwen35_linear_attn_conv
build_qwen35_linear_attn_conv(load=False, require_cached=False)
PY
python3 scripts/smoke.py --mode qwen35-linear-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter18-512-run{1,2,3,4,5,6}.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: fixture passed (`max_kl=0.01743`, top-1 `1.0`,
`native_owned_device_bytes=1625645909`) and 4K was runnable at `334.200 tok/s`,
but 512/128 samples `566.285`, `566.742`, `565.331`, `564.287`, `558.782`,
`557.391`, `558.201` tok/s (median `564.287`) were below retained `565.213`.
Decision: revert to 256-thread FP16 prefill conv.

## 2026-05-15 — Prefill multiloop iter 19: 128-thread FP16 conv rejected

Tried changing the retained FP16-input linear-attention prefill conv wrapper from
256 to 128 threads/block after the 512-thread variant regressed. Correctness
held, but 512/128 throughput regressed below the retained 256-thread baseline.

Validation commands:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/linear_attn/conv.py hipengine/runtime/qwen35_paro.py scripts/smoke.py
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.linear_attn.conv import build_qwen35_linear_attn_conv
build_qwen35_linear_attn_conv(load=False, require_cached=False)
PY
python3 scripts/smoke.py --mode qwen35-linear-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter19-512-run{1,2,3}.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: fixture passed (`max_kl=0.01743`, top-1 `1.0`,
`native_owned_device_bytes=1625645909`) and 4K was runnable at `333.775 tok/s`,
but 512/128 samples `563.882`, `560.443`, `562.072`, `560.937` tok/s (median
`561.504`) were below retained `565.213`. The linear-attn prefill smoke still
passed for the FP16 conv variant. Decision: revert to 256-thread FP16 prefill
conv.

## 2026-05-15 — Prefill multiloop iter 20: qkv_f32 scratch elision rejected

Tried skipping the now-unused `linear_attn.qkv_f32` workspace allocation for
single-request FP16 native prefill. Packed segment prefill still requested the
F32 qkv scratch because it still casts before the segment conv path.

Validation commands:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter20-512-run{1,2,3}.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: tests passed (`56 passed`) and fixture passed (`max_kl=0.01743`, top-1
`1.0`, `native_owned_device_bytes=1625645909`; diagnostic accounting did not
change). 4K was runnable at `333.237 tok/s`, but 512/128 samples `560.143`,
`562.544`, `560.982`, `561.091` tok/s (median `561.036`) regressed below
retained `565.213`. Decision: revert; the allocation elision does not help the
timed path and complicates scratch reuse.

## 2026-05-15 — Prefill multiloop iter 21: direct layer output rejected

Tried letting grouped MoE combine write directly into the runner's `next_hidden`
ping-pong prefill buffer for single-request native prefill, skipping the explicit
per-layer device-to-device copy from `scratch.moe_out` to `next_hidden`. This
removes the copy operation mechanically but did not improve the timed path.

Validation commands:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter21-512-run{1,2,3}.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: tests passed (`56 passed`) and fixture passed (`max_kl=0.01743`, top-1
`1.0`, `native_owned_device_bytes=1625645909`). 4K was runnable at
`332.699 tok/s`, but 512/128 samples `559.527`, `565.015`, `563.030`,
`560.252` tok/s (median `561.641`) regressed below retained `565.213`.
Decision: revert; the D2D copy is not a useful short-depth bottleneck.

## 2026-05-15 — Prefill multiloop iter 22: 128-thread head rotary rejected

After four consecutive orchestration/copy failures, switched back to a kernel-side
full-attention prefill target. Tried changing the multi-token
`qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16` wrapper from 256 to 128
threads/block. The row width is 128, so the hypothesis was that 256 threads did
extra shared reduction work with half the lanes idle.

Validation commands:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/rotary/qwen35_rotary.py hipengine/runtime/qwen35_paro.py
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import build_qwen35_rotary
build_qwen35_rotary(load=False, require_cached=False)
PY
python3 scripts/smoke.py --mode qwen35-rotary-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter22-512-run{1,2,3}.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: rotary smoke passed (`vector_position_max_abs=2.38e-07`) and fixture
passed (`max_kl=0.01743`, top-1 `1.0`, `native_owned_device_bytes=1625645909`).
4K was runnable at `332.419 tok/s`, but 512/128 samples `563.779`, `558.622`,
`556.209`, `559.374` tok/s (median `558.998`) regressed below retained
`565.213`. Decision: revert; the 256-thread head rotary path stays retained.

## 2026-05-15 — Prefill multiloop iter 23: AWQ prefill-profile library rejected

After the iter-22 plateau pivot, re-reviewed the retained docs/ROOFLINE.md and
parent docs/OPTIMAL.md notes instead of doing another local thread sweep. The
qualitatively different hypothesis was build-profile mismatch: docs/KERNELS.md
says prefill-phase kernels should use the `prefill` profile (`-mllvm
-amdgpu-unroll-threshold-local=600`, no `-mcumode`), while the resident runtime
loaded the hot AWQ projection library with the decode profile for both prompt
prefill and decode. Tried adding a separate prefill-profile AWQ `.so` and
routing native prefill layer execution through it while leaving decode on the
existing decode-profile library.

Validation commands:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro_runner.py
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import build_paro_awq_gemv
build_paro_awq_gemv(profile='prefill', load=False, require_cached=False)
PY
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter23-512-run{1,2,3}.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: targeted tests passed (`56 passed`) and fixture passed (`max_kl=0.01743`,
top-1 `1.0`, `native_owned_device_bytes=1625645909`). 4K was runnable and above
the retained guard at `333.398 tok/s`, but 512/128 samples `562.284`, `560.307`,
`557.130`, `560.647` tok/s (median `560.477`) regressed below retained
`565.213`. Decision: revert; the AWQ decode-profile build remains faster for
this resident prefill path despite the nominal phase-profile mismatch.

## 2026-05-15 — Prefill multiloop iter 24: linear-attn prefill-profile libraries rejected

Continued the build-profile pivot, but moved away from the AWQ projection family
that regressed in iter 23. Parent profiling notes call out GDN recurrent prefill
as a real 512-prefill bucket, and docs/KERNELS.md classifies linear-attention
conv/GDN as prefill-phase kernels. Tried adding prefill-profile
`linear_conv`/`linear_gdn` library overrides for native prefill layer execution
while keeping decode on the retained decode-profile libraries.

Validation commands:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro_runner.py
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.linear_attn.conv import build_qwen35_linear_attn_conv
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import build_qwen35_linear_attn_gdn
build_qwen35_linear_attn_conv(profile='prefill', load=False, require_cached=False)
build_qwen35_linear_attn_gdn(profile='prefill', load=False, require_cached=False)
PY
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
python3 scripts/smoke.py --mode qwen35-linear-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter24-512-run{1,2,3}.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: targeted tests passed (`56 passed`), the linear-attention prefill smoke
passed (`fp16_gdn_k2_out_max_abs=1.4e-09`, `fp16_gated_mismatch=0`), and fixture
passed (`max_kl=0.01743`, top-1 `1.0`, `native_owned_device_bytes=1625645909`).
4K was runnable and above the retained guard at `333.168 tok/s`, but 512/128
samples `558.703`, `559.778`, `557.489`, `559.987` tok/s (median `559.240`)
regressed below retained `565.213`. Decision: revert; the decode-profile linear
conv/GDN builds remain faster for the resident native prefill path.

## 2026-05-15 — Prefill multiloop iter 25: fused split+key cast rejected

Tried a small full-attention prefill launch fusion: added
`qwen35_split_qgate_key_fp16`, which splits the interleaved FP16 Q/gate rows and
casts the FP16 K projection to FP32 in one `qwen35_rotary` launch. The prefill
path used it only for `tokens > 1`; c=1/decode kept the existing separate
`qwen35_split_qgate_fp16` + `fp16_to_f32` sequence. Unit tests were updated to
monkeypatch the fused wrapper so fake-pointer tests do not accidentally launch a
real GPU kernel.

Validation commands:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/rotary/qwen35_rotary.py hipengine/runtime/qwen35_paro.py
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import build_qwen35_rotary
build_qwen35_rotary(load=False, require_cached=False)
PY
python3 scripts/smoke.py --mode qwen35-rotary-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 -m pytest tests/test_qwen35_rotary_plan.py tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter25-512-run{1,2,3}.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: targeted tests passed (`59 passed`), qwen35 rotary smoke passed
(`vector_position_max_abs=2.38e-07`, existing split FP16 checks clean), and
fixture passed (`max_kl=0.01743`, top-1 `1.0`,
`native_owned_device_bytes=1625645909`). 4K was runnable and above the retained
guard at `333.491 tok/s`, but 512/128 samples `562.365`, `560.989`, `562.071`,
`559.839` tok/s (median `561.530`) regressed below retained `565.213`.
Decision: revert; eliminating this small full-attention cast launch does not help
short prefill enough to retain.

## 2026-05-15 — Prefill multiloop iter 26: interleaved Q/K prefill fusion rejected

Refined iter 25's failed split+key-cast fusion by eliminating a larger part of
the full-attention prefill launch sequence. The trial used the existing dual
transposed Q/K GEMV for `tokens > 1` to write row-major `[q+gate,k]` rows into
`scratch.q_proj_key`, then added `qwen35_split_qgate_interleaved_key_fp16` to
split/cast that interleaved buffer into query/gate/key_raw. c=1/decode stayed on
the retained dual output plus separate split/cast path. This replaced two Q/K
GEMV launches plus split/cast with one dual GEMV plus one splitter in each of
the 10 full-attention prefill layers.

Validation commands:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/rotary/qwen35_rotary.py hipengine/runtime/qwen35_paro.py
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import build_qwen35_rotary
build_qwen35_rotary(load=False, require_cached=False)
PY
python3 scripts/smoke.py --mode qwen35-rotary-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 -m pytest tests/test_qwen35_rotary_plan.py tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter26-512-run{1,2,3}.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: targeted tests passed (`59 passed`), qwen35 rotary smoke passed
(`vector_position_max_abs=2.38e-07`, split FP16 checks clean), and fixture passed
(`max_kl=0.01743`, top-1 `1.0`, `native_owned_device_bytes=1625645909`). 4K was
runnable and above the retained guard at `333.557 tok/s`, but 512/128 samples
`563.252`, `561.828`, `563.680`, `559.307` tok/s (median `562.540`) regressed
below retained `565.213`. Decision: revert; even the larger full-attention Q/K
launch fusion is not enough to improve short prefill and should not complicate
layout semantics.

## 2026-05-15 — Prefill multiloop iter 27: fused grouped lane/shared combine rejected

Pivoted away from more full-attention Q/K launch fusion after four consecutive
negative trials and tried a grouped-MoE tail fusion for the FP16 native prefill
path. Added a temporary `weighted_lanes_shared_gate_combine_residual_batch_out_*`
PARO combine specialization that kept the existing sorted-lane inverse map but
merged the grouped weighted-lane accumulation with the shared-gate residual
combine. `run_moe_grouped_compact_fp16()` computed the shared expert first and
then launched the fused lane-aware combine into `scratch.moe_out`.

Validation commands:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/fused/paro_combine.py hipengine/kernels/hip_gfx1100/fused/__init__.py hipengine/runtime/qwen35_paro.py
python3 -m pytest tests/test_paro_combine_plan.py -q
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.fused.paro_combine import build_paro_combine
lib = build_paro_combine(load=True, require_cached=False)
for name in (
    'hipengine_weighted_lanes_shared_gate_combine_residual_batch_out_bf16_f32w',
    'hipengine_weighted_lanes_shared_gate_combine_residual_batch_out_fp16_f32w',
):
    getattr(lib, name)
print('paro_combine rebuilt with fused lane shared symbols')
PY
python3 -m pytest tests/test_paro_combine_plan.py tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
for i in 1 2 3; do python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter27-512-run${i}.json >/tmp/iter27-512-run${i}.stdout; done
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
```

Results: targeted tests passed (`59 passed`) and the temporary HIP combine
library rebuilt with the fused symbols. Fixture correctness passed
(`max_kl=0.01743`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`). 4K/128 remained runnable and above the
retained guard at `333.792 tok/s` (`prefill_seconds=12.2711`). The primary
512/128 samples were `564.284`, `559.974`, `559.840`, and `560.922` tok/s
(median `560.448`), below the retained `565.213` tok/s. Decision: revert; the
extra combined arithmetic/order did not offset the launch reduction, so keep the
existing separate weighted-lane sum plus shared/residual combine on the default
path.

## 2026-05-15 — Prefill multiloop iter 28: fused W4 prefill projections kept

After five failed launch/copy micro-fusion trials and the parent WORKLOG/ROOFLINE
review, pivoted to the major algorithmic gap versus the parent route: hipENGINE
multi-token prefill was still running non-expert transposed W4 pack8 projections
through row-wise GEMV kernels. Ported the parent dense fused W4→WMMA prefill
kernel from `nano-vllm-amd@55fede9` (`nanovllm/native/qwen35/paroquant_fusedw4.py`)
into the raw-pointer `paro_awq_gemv` family as `awq_fusedw4_prefill_fp16`, then
routed FP16 multi-token full-attention Q/K and linear-attention QKV/Z prompt
projections through it. c=1/decode and strided V/O/out projections retain the
existing GEMV fallback path.

Validation commands:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.py hipengine/kernels/hip_gfx1100/quant/__init__.py hipengine/runtime/qwen35_paro.py
python3 -m pytest tests/test_paro_awq_gemv_plan.py tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import build_paro_awq_gemv
lib = build_paro_awq_gemv(load=True, require_cached=False)
getattr(lib, 'hipengine_awq_fusedw4_prefill_fp16')
print('awq fusedw4 prefill symbol loaded')
PY
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
for i in 1 2 3; do python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter28-512-run${i}.json >/tmp/iter28-512-run${i}.stdout; done
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter28-fusedw4-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 1 --warmup-decode-tokens 0 --max-layers 1 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter28-fusedw4-trace-smoke.json
sqlite3 -header -csv /tmp/iter28-fusedw4-trace/trace_results.db "select name, count(*) as n, min(duration) as min_ns, max(duration) as max_ns, avg(duration) as avg_ns, min(workgroup_x) as wg from kernels where name like '%fusedw4%' group by name;"
python3 scripts/check_lineage.py --kind kernel --diff stat
```

Results: targeted tests passed (`59 passed`) and the AWQ library rebuilt with the
new symbol. 512/128 samples were `745.087`, `743.684`, `743.163`, and
`739.965` tok/s (median `743.424`), +31.5% over retained `565.213`. Fixture gate
passed (`max_kl=0.01005`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.6925s`). 4K/128 was
runnable and improved to `395.308 tok/s` (`prefill_seconds=10.3615`, decode
`102.127 tok/s`), above the retained guard. The profiler smoke confirmed
`awq_fusedw4_prefill_fp16_kernel<32, 32>` launched twice for a one-layer prompt
run (`avg_ns=267743.5`, `Workgroup_Size_X=32`). Decision: keep and update the
kernel catalog/source lineage plus benchmark diagnostic rollup. Remaining gap:
strided V/O/out W4 projections and W8/dense auxiliaries still use GEMV-style
paths, so continue with fused-W4/WMMA projection coverage rather than small
launch fusions.

## 2026-05-15 — Prefill multiloop iter 29: strided fused W4 prefill projections kept

Continued the fused-W4/WMMA projection pivot from iter 28. The transposed Q/K
and QKV/Z route was fast, but full-attention V/O and linear-attention out_proj
still called strided pack8 GEMV for every prompt row. Generalized the
`awq_fusedw4_prefill_fp16` kernel with a `qweight_transposed` template, added a
raw-pointer `hipengine_awq_fusedw4_prefill_strided_fp16` wrapper, and routed
`project_pack8_fp16(..., rows>1)` through it when the group size is compatible.
The old pack8 GEMV remains the c=1/small-group fallback and no extra transposed
weight copies were added.

Validation commands:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.py hipengine/kernels/hip_gfx1100/quant/__init__.py hipengine/runtime/qwen35_paro.py
python3 -m pytest tests/test_paro_awq_gemv_plan.py tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import build_paro_awq_gemv
lib = build_paro_awq_gemv(load=True, require_cached=False)
for name in ('hipengine_awq_fusedw4_prefill_fp16','hipengine_awq_fusedw4_prefill_strided_fp16'):
    getattr(lib, name)
print('awq fusedw4 transposed+strided symbols loaded')
PY
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
for i in 1 2 3; do python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter29-512-run${i}.json >/tmp/iter29-512-run${i}.stdout; done
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter29-fusedw4-strided-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 1 --warmup-decode-tokens 0 --max-layers 1 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter29-fusedw4-strided-trace-smoke.json
sqlite3 -header -csv /tmp/iter29-fusedw4-strided-trace/trace_results.db "select name, count(*) as n, min(duration) as min_ns, max(duration) as max_ns, avg(duration) as avg_ns, min(workgroup_x) as wg from kernels where name like '%awq_fusedw4_prefill_fp16_kernel%' group by name;"
```

Results: targeted tests passed (`59 passed`) and the AWQ library rebuilt with both
fused-W4 symbols. 512/128 samples were `1693.328`, `1680.357`, `1675.910`, and
`1674.902` tok/s (median `1678.133`), +125.7% over retained `743.424`. Fixture
gate passed (`max_kl=0.02233`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.3073s`), so the change
did not grow the recorded owned device footprint. 4K/128 was runnable at
`564.616 tok/s` (`prefill_seconds=7.2545`, decode `102.145 tok/s`), above the
retained guard. The profiler smoke confirmed the new strided instantiation ran:
`awq_fusedw4_prefill_fp16_kernel<32,32,false>` once (`327164 ns`,
`Workgroup_Size_X=32`) alongside the existing transposed instantiation. Decision:
keep. Remaining obvious projection gap is dense/W8 auxiliary prefill work and any
remaining non-W4 projection buckets, not the W4 prompt projection family.

## 2026-05-15 — Prefill multiloop iter 30: fused W8A16 shared gate/up SiLU kept

After the W4 projection family moved to fused-W4/WMMA, a prefill-only profiler
run showed W8A16 shared expert work as a top bucket: before this change,
`w8a16_linear_lowp_out_kernel<_Float16>` ran 80 times for 40-layer 512-token
prefill and totaled `50.934 ms`, with separate shared SiLU launches. Ported the
parent `w8a16_shared_gate_up_bulk4_kernel` idea from `nano-vllm-amd@59195ed`
into hipENGINE's raw-pointer W8A16 library as
`hipengine_w8a16_shared_gate_up_silu_fp16`: four intermediate columns per block,
FP16 lowp rounding before SiLU to match the existing gate/up → SiLU staging, and
output into the existing `shared_intermediate` scratch. Runtime routing uses this
only for `tokens > 1`; c=1/decode retains the old W8A16 gate/up +
`silu_mul_dual_out_fp16` fallback, and the down projection remains unchanged.

Validation commands:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/quant/w8a16_linear.py hipengine/kernels/hip_gfx1100/quant/__init__.py hipengine/runtime/qwen35_paro.py
python3 -m pytest tests/test_w8a16_linear_plan.py tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import build_w8a16_linear
lib = build_w8a16_linear(load=True, require_cached=False)
for name in ('hipengine_w8a16_linear_fp16_lowp_out','hipengine_w8a16_shared_gate_up_silu_fp16'):
    getattr(lib, name)
print('w8a16 symbols loaded')
PY
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
for i in 1 2 3; do python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter30-512-run${i}.json >/tmp/iter30-512-run${i}.stdout; done
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter30-w8-fused-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter30-w8-fused-trace.json
sqlite3 -header -csv /tmp/iter30-w8-fused-trace/trace_results.db "select name, count(*) n, sum(duration)/1e6 total_ms, avg(duration)/1e3 avg_us, max(duration)/1e3 max_us from kernels where name like '%w8a16%' or name like '%silu_mul_dual_out%' group by name order by sum(duration) desc;"
python3 scripts/check_lineage.py --kind kernel --diff stat || true
```

Results: tests passed (`60 passed`) and the W8A16 library rebuilt with the new
symbol. 512/128 samples were `1763.007`, `1757.240`, `1748.107`, and `1743.355`
tok/s (median `1752.674`), +4.4% over retained `1678.133`. Fixture gate passed
(`max_kl=0.03406`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.2953s`), so the owned
memory accounting is unchanged. 4K/128 stayed runnable and improved to
`572.269 tok/s` (`prefill_seconds=7.1575`, decode `102.614 tok/s`), above the
95% guard. The profiler confirmed the new shared gate/up+SiLU kernel ran 40
times (`14.805 ms` total, avg `370.114 us`) and the remaining W8A16 down ran 40
times (`25.714 ms` total), replacing the previous 80 generic W8A16 launches plus
standalone shared SiLU launches. Decision: keep. The remaining top buckets in
the 512 prefill trace are now full-attention prefill GQA, GDN recurrent prefill,
selected MoE WMMA, W4 prompt projections, and W8A16 down.

## 2026-05-15 — Prefill multiloop iter 31: fused W8A16 shared down combine kept

After iter 30, the W8A16 shared gate/up+SiLU bucket was fused, but the grouped
multi-token MoE path still launched a generic shared down projection plus a
separate shared-gate/residual combine. Added
`hipengine_w8a16_shared_down_combine_residual_fp16`, a raw-pointer FP16 helper
that computes four hidden columns per block, rounds the shared down value to FP16
to match the previous `w8a16_linear_fp16_lowp_out` staging, then combines the
existing rounded `selected_out`, shared gate, and residual in the same kernel.
`run_moe_grouped_compact_fp16` now uses shared gate/up+SiLU followed by this
fused down/combine helper for `tokens > 1`; c=1 and non-grouped paths keep the
unfused fallback.

Validation commands:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/quant/w8a16_linear.py hipengine/kernels/hip_gfx1100/quant/__init__.py hipengine/runtime/qwen35_paro.py
python3 -m pytest tests/test_w8a16_linear_plan.py tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import build_w8a16_linear
lib = build_w8a16_linear(load=True, require_cached=False)
for name in ('hipengine_w8a16_shared_gate_up_silu_fp16','hipengine_w8a16_shared_down_combine_residual_fp16'):
    getattr(lib, name)
print('w8a16 fused shared symbols loaded')
PY
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
for i in 1 2 3; do python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter31-512-run${i}.json >/tmp/iter31-512-run${i}.stdout; done
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter31-w8-down-combine-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter31-w8-down-combine-trace.json
sqlite3 -header -csv /tmp/iter31-w8-down-combine-trace/trace_results.db "select name, count(*) n, sum(duration)/1e6 total_ms, avg(duration)/1e3 avg_us, max(duration)/1e3 max_us from kernels where name like '%w8a16%' or name like '%shared_gate_combine%' group by name order by sum(duration) desc;"
```

Results: tests passed (`61 passed`) and the W8A16 library rebuilt with both fused
shared symbols. 512/128 samples were `1811.035`, `1797.715`, `1791.571`, and
`1794.850` tok/s (median `1796.282`), +2.5% over retained `1752.674`. Fixture
gate passed (`max_kl=0.03406`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.2874s`), so the owned
memory accounting is unchanged. 4K/128 stayed runnable and improved to
`578.288 tok/s` (`prefill_seconds=7.0830`, decode `102.490 tok/s`), above the
95% guard. Profiler evidence: `w8a16_shared_down_combine_residual_fp16_kernel`
ran 40 times (`18.913 ms` total, avg `472.832 us`) and
`w8a16_shared_gate_up_silu_fp16_kernel` ran 40 times (`15.129 ms` total),
replacing the previous W8A16 down plus separate shared-combine path. Decision:
keep. Remaining high buckets are now full-attention prefill GQA, GDN recurrent
prefill, selected MoE WMMA, and W4 prompt projections.

## 2026-05-16 — Prefill multiloop iter 32: rejected 128-thread W8A16 shared reducers

Tried a narrow W8A16 thread-count retune: grouped multi-token FP16 MoE prefill
called `w8a16_shared_gate_up_silu_fp16` and
`w8a16_shared_down_combine_residual_fp16` with `threads=128` instead of the
retained 64-thread default. The idea was to speed the 4096/768 reduction-heavy
shared expert helpers while leaving c=1/decode and non-grouped fallbacks
unchanged.

Validation commands:

```bash
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_w8a16_linear_plan.py -q
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
```

Results: targeted tests passed (`37 passed`) and fixture gate passed
(`max_kl=0.03121`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.3036s`). However the
primary 512/128 prefill sample regressed to `1699.782 tok/s` versus retained
median `1796.282`. The 4K/128 guard remained runnable but lower than retained at
`567.685 tok/s` (`prefill_seconds=7.2153`, decode `102.910 tok/s`). Decision:
reject/revert. The 64-thread retained setting is better for the composed prefill
path despite the larger reduction size; avoid retuning these W8A16 shared helpers
without direct kernel-trace evidence of a positive end-to-end signal.

## 2026-05-16 — Prefill multiloop iter 33: rejected 64-thread short full-attn prefill

Tried changing the full-attention GQA prefill wrapper so short rows
(`max_context_len <= 1024`, including the 512-token primary shape) use 64
threads instead of the retained 32-thread wave32 policy. The motivation was the
trace top bucket: all-layer 512-token full-attention prefill took ~56 ms, and
64 threads might reduce per-lane score/value work while still avoiding the known
unsafe 256-thread launch.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import build_qwen35_paged_attn_decode
lib = build_qwen35_paged_attn_decode(load=True, require_cached=False)
getattr(lib, 'hipengine_qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans')
getattr(lib, 'hipengine_qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans')
print('paged attention prefill symbols loaded')
PY
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
```

Results: paged-attention library rebuilt and targeted tests passed (`37 passed`).
Fixture gate passed (`max_kl=0.04303`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.3011s`). Primary
512/128 regressed to `1709.874 tok/s` versus retained median `1796.282`; 4K/128
remained runnable at `576.765 tok/s` but did not improve the retained best
(`578.288`). Decision: reject/revert. Keep the one-wave 32-thread short-row
attention policy; 64 threads hurts the composed 512 path despite the larger top
bucket.

## 2026-05-16 — Prefill multiloop iter 34: shared query cache for full-attn prefill

After two negative thread-count retunes, re-read the relevant optimization notes
before changing code: `docs/ROOFLINE.md`, `docs/KERNELS.md`, parent
`~/amd-gpu-tuning/docs/OPTIMAL.md`, plus parent `WORKLOG.md` / `LESSONS-LEARNED.md`
entries on compact WMMA, GDN recurrent rejections, and the retained full-shape
PARO route. The current retained trace still had full-attention GQA prefill as
the largest 512 bucket (`56.377 ms` across 10 launches). Unlike the optimized
paged decode context kernel, the prefill kernel reread the same FP32 query head
from global memory for every visible token. Changed the single-request and
varlen full-attention prefill kernels to cache the per-block query head in LDS
and use the existing vec8 BF16 key-dot helper when `head_dim` is divisible by 8.
The retained 32-thread short-row / 64-thread 4K launch policy is unchanged.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import build_qwen35_paged_attn_decode
lib = build_qwen35_paged_attn_decode(load=True, require_cached=False)
getattr(lib, 'hipengine_qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans')
getattr(lib, 'hipengine_qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans')
print('paged attention prefill symbols loaded')
PY
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter34-attn-qshared-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter34-attn-qshared-trace.json
sqlite3 -header -csv /tmp/iter34-attn-qshared-trace/trace_results.db "select name, count(*) n, sum(duration)/1e6 total_ms, avg(duration)/1e3 avg_us, max(duration)/1e3 max_us from kernels where name like '%prefill_gqa_gate%' or name like '%gdn_prefill%' or name like '%awq_fusedw4%' group by name order by sum(duration) desc limit 20;"
```

Results: paged-attention library rebuilt and targeted tests passed (`37 passed`).
512/128 samples were `1887.964`, `1879.829`, `1884.834`, and `1883.046` tok/s
(median `1883.940`), +4.9% over retained `1796.282`. Fixture gate passed
(`max_kl=0.03406`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.2731s`). 4K/128
remained runnable and improved to `658.418 tok/s` (`prefill_seconds=6.2210`,
decode `102.086 tok/s`), well above the 95% guard from retained `578.288`.
Profiler evidence: `qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel` ran 10
times with `39.981 ms` total / `3998.1 us` average, down from the prior retained
trace's `56.377 ms` total. Decision: keep. Updated benchmark artifact/rollup and
`docs/KERNELS.md`; still below the parent OPTIMAL target, with current top
buckets now GDN recurrent prefill, compact WMMA selected MoE, and W4 prompt
projections.

## 2026-05-16 — Prefill multiloop iter 35: rejected full-attn token-offset cache

Tried extending the iter-34 full-attention GQA prefill query-cache win by caching
per-token physical KV offsets in LDS for short rows (`max_context_len <= 1024`)
and using a contiguous-block fast path before the value loop. The goal was to
avoid recomputing logical block/block offset and rereading the row block table
for every output dimension. The first all-context version improved a 512 sample
but regressed 4K/128 badly (`343.696 tok/s`), so the trial was shape-gated to
short rows only before final measurement.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import build_qwen35_paged_attn_decode
lib = build_qwen35_paged_attn_decode(load=True, require_cached=False)
getattr(lib, 'hipengine_qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans')
getattr(lib, 'hipengine_qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans')
print('paged attention prefill symbols loaded')
PY
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
```

Results after short-row gating: targeted tests passed (`37 passed`) and fixture
gate passed (`max_kl=0.03406`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.2742s`). 512/128
samples were `1878.021`, `1848.587`, `1867.779`, and `1876.125` tok/s (median
`1871.952`), a regression versus retained `1883.940`. 4K/128 recovered above
the guard at `655.091 tok/s` (`prefill_seconds=6.2526`, decode `102.522 tok/s`)
but was slightly below retained `658.418`. Decision: reject/revert. The extra
short-row LDS/branching does not beat the simpler iter-34 query-cache path.

## 2026-05-16 — Prefill multiloop iter 36: two-wave GDN k2 reduction specialization

After iter 34 made full-attention prefill faster, the current 512/0 trace showed
`qwen35_gdn_prefill_recurrent_k2_kernel` as the largest remaining bucket
(`45.089 ms` across 30 launches). The wrapper validates `head_k_dim == 128` and
launches `head_k_dim / 2 == 64` threads, i.e. exactly two wave32 warps. Specialized
the k2 kernel's per-token cross-warp reductions from a runtime `num_warps` loop
over `partial[]` to direct `partial[0] + partial[1]`, preserving the accumulation
order while reducing scalar loop overhead. Did not port parent-discarded k4 or
value-tiled variants.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import build_qwen35_linear_attn_gdn
lib = build_qwen35_linear_attn_gdn(load=True, require_cached=False)
getattr(lib, 'hipengine_qwen35_gdn_prefill_recurrent_k2_f32')
print('gdn k2 symbol loaded')
PY
python3 -m pytest tests/test_qwen35_linear_attn_gdn_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter36-gdn-twowarp-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter36-gdn-twowarp-trace.json
sqlite3 -header -csv /tmp/iter36-gdn-twowarp-trace/trace_results.db "select name, count(*) n, sum(duration)/1e6 total_ms, avg(duration)/1e3 avg_us, max(duration)/1e3 max_us from kernels where name like '%gdn_prefill_recurrent%' or name like '%prefill_gqa_gate%' or name like '%awq_fusedw4%' group by name order by sum(duration) desc limit 20;"
```

Results: targeted tests passed (`37 passed`). 512/128 samples were `1916.075`,
`1910.982`, `1901.536`, and `1893.917` tok/s (median `1906.259`), +1.2% over
retained `1883.940`. Fixture gate passed (`max_kl=0.03406`, top-1 agreement
`1.0`, `native_owned_device_bytes=1625645909`, native prefill `0.2710s`).
4K/128 remained runnable and slightly improved to `661.506 tok/s`
(`prefill_seconds=6.1919`, decode `101.966 tok/s`), above the 95% guard.
Profiler evidence: `qwen35_gdn_prefill_recurrent_k2_kernel` ran 30 times with
`40.993 ms` total / `1366.4 us` average, down from the prior retained trace's
`45.089 ms` total. Decision: keep. Updated benchmark artifact/rollup and
`docs/KERNELS.md`.

## 2026-05-16 — Prefill multiloop iter 37: rejected GDN value2 tiling

Tried to reduce the remaining GDN recurrent prefill bucket by processing two
adjacent value columns per `qwen35_gdn_prefill_recurrent_k2_kernel` block. The
trial changed the launch grid from `head_v_dim` value blocks to
`ceil(head_v_dim / 2)` and kept separate `partial[0]+partial[1]` /
`partial[2]+partial[3]` reductions so each value column's k2 accumulation order
matched the retained kernel. Rationale: adjacent value columns reuse the same
Q/K/decay/beta rows and have contiguous recurrent-state columns.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import build_qwen35_linear_attn_gdn
lib = build_qwen35_linear_attn_gdn(load=True, require_cached=False)
getattr(lib, 'hipengine_qwen35_gdn_prefill_recurrent_k2_f32')
print('gdn k2 symbol loaded')
PY
python3 -m pytest tests/test_qwen35_linear_attn_gdn_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/smoke.py --mode qwen35-linear-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter37-gdn-v2-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter37-gdn-v2-trace.json
```

Results: targeted tests passed (`37 passed`) and the linear-attention prefill
smoke remained numerically equivalent (`gdn_k2_out_max_abs=9.31e-10`,
`fp16_gdn_k2_out_max_abs=1.4e-09`). 512/128 samples were `1898.937`,
`1905.544`, `1892.329`, and `1888.308` tok/s (median `1895.633`), below
retained `1906.259`. Fixture gate still passed (`max_kl=0.03406`, top-1
agreement `1.0`, `native_owned_device_bytes=1625645909`, native prefill
`0.2702s`) and 4K/128 remained above guard at `662.689 tok/s`
(`prefill_seconds=6.1809`). Profiler showed the tiled GDN kernel regressed to
`43.842 ms` total / `1461.4 us` average across 30 launches versus retained
`40.993 ms`. Decision: reject/revert. The reduced block count does not overcome
extra registers/dual reductions in this kernel.

## 2026-05-16 — Prefill multiloop iter 38: rejected fused W4 tile_m=64 default

Tried increasing the default fused W4 prefill output tile from `tile_m=32` to
`tile_m=64` for large projections while keeping `tile_n=32`. The goal was to
halve output-tile block count for Qwen projection shapes without changing math,
public ABI, or call sites. The HIP kernel already has a `<64,32>` instantiation,
so the trial only changed the Python wrapper's default tile selection.

Validation commands:

```bash
python3 -m pytest tests/test_paro_awq_gemv_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter38-fusedw4-tile64-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter38-fusedw4-tile64-trace.json
```

Results: targeted tests passed (`37 passed`). 512/128 samples were `1806.684`,
`1792.522`, `1785.408`, and `1781.828` tok/s (median `1788.965`), -6.2% below
retained `1906.259`. Fixture gate passed (`max_kl=0.03406`, top-1 agreement
`1.0`, `native_owned_device_bytes=1625645909`, native prefill `0.2877s`).
4K/128 stayed runnable and above guard at `659.614 tok/s`
(`prefill_seconds=6.2097`). Profiler showed the fused W4 kernels switched to
`<64,32>` but regressed badly: transposed total `31.160 ms` and strided total
`26.200 ms` (combined `57.36 ms`) versus retained `<32,32>` combined about
`37.39 ms`. Decision: reject/revert; the larger output tile increases per-block
register/WMMA work enough to overwhelm the reduced grid count.

## 2026-05-16 — Prefill multiloop iter 39: block_size=256 attention address fast path

After two rejected GDN/W4 retunes, moved back to the remaining full-attention
prefill bucket. The runtime enforces `block_size=256`, but
`qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel` still used dynamic
`token / block_size` and `token - logical_block * block_size` in both the score
and value loops. Added a uniform `block_size == 256` fast path using `token >> 8`
and `token & 255`, preserving the fallback for any non-256 caller and leaving
`KVLiveSpans` / BF16 KV math unchanged.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import build_qwen35_paged_attn_decode
lib = build_qwen35_paged_attn_decode(load=True, require_cached=False)
getattr(lib, 'hipengine_qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans')
print('paged attention prefill symbol loaded')
PY
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter39-attn-block256-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter39-attn-block256-trace.json
```

Results: targeted tests passed (`37 passed`). 512/128 samples were `1929.499`,
`1931.127`, `1917.049`, and `1910.712` tok/s (median `1923.274`), +0.9% over
retained `1906.259`. Fixture gate passed (`max_kl=0.03406`, top-1 agreement
`1.0`, `native_owned_device_bytes=1625645909`, native prefill `0.2662s`).
4K/128 stayed runnable and improved to `672.650 tok/s`
(`prefill_seconds=6.0893`, decode `102.537 tok/s`), above the 95% guard.
Profiler evidence: `qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel` ran 10
times with `36.361 ms` total / `3636.1 us` average, down from retained
`40.102 ms` total. Decision: keep. Updated benchmark artifact/rollup and
`docs/KERNELS.md`.

## 2026-05-16 — Prefill multiloop iter 40: rejected short-row attention block-table preload

Tried extending the retained `block_size=256` address fast path in
`qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel` with a short-row path for
`visible_len <= 512`. The trial preloaded `row_table[0]` / `row_table[1]` once
per block and used those two physical blocks in the score and value loops. The
primary 512 shape improved because every causal row sees at most two 256-token
blocks, but the same extra branches/locals stayed in the kernel and hurt the
4K guard path even though it fell back to the generic address path.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import build_qwen35_paged_attn_decode
lib = build_qwen35_paged_attn_decode(load=True, require_cached=False)
getattr(lib, 'hipengine_qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans')
print('paged attention prefill symbol loaded')
PY
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128-rerun.json >/tmp/multiloop-prefill-4k-128-rerun.stdout 2>/tmp/multiloop-prefill-4k-128-rerun.stderr
rocprofv3 --kernel-trace -d /tmp/iter40-attn-short2-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter40-attn-short2-trace.json
```

Results: targeted tests passed (`37 passed`) and the fixture gate passed
(`max_kl=0.03406`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.2651s`). 512/128
samples were `1948.157`, `1942.087`, `1924.765`, and `1928.438` tok/s (median
`1935.263`), +0.6% over retained `1923.274`. The 512 profiler confirmed the
intended local win: prefill GQA total `32.941 ms` versus retained `36.361 ms`.
However 4K/128 regressed below the required 95% guard twice: `633.950` and
`635.234` tok/s versus floor `639.018` (`0.95 * 672.650`). Decision:
reject/revert. The short-row specialization is not safe as a default-path active
change until it can be separated from the long-context kernel or made neutral at
4K.

## 2026-05-16 — Prefill multiloop iter 41: split short prefill-attention block-table preload

Revisited iter 40's local 512 win after rejecting it for 4K regression. Instead
of carrying the short-row block-table preload inside the generic prefill GQA
kernel, templated `qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel` and
launches `<true>` only for `block_size == 256 && max_context_len <= 512`. The
short template preloads `row_table[0]` / `row_table[1]` once per block and uses
those two physical blocks in the score and value loops; the `<false>` template
keeps the retained generic path for 4K and longer rows. Public C ABI and
`KVLiveSpans` remain unchanged.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import build_qwen35_paged_attn_decode
lib = build_qwen35_paged_attn_decode(load=True, require_cached=False)
getattr(lib, 'hipengine_qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans')
print('paged attention prefill symbol loaded')
PY
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128-rerun.json >/tmp/multiloop-prefill-4k-128-rerun.stdout 2>/tmp/multiloop-prefill-4k-128-rerun.stderr
rocprofv3 --kernel-trace -d /tmp/iter41-attn-short-split-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter41-attn-short-split-trace.json
```

Results: targeted tests passed (`37 passed`). 512/128 samples were `1979.986`,
`1970.632`, `1978.674`, and `1959.851` tok/s (median `1974.653`), +2.7% over
retained `1923.274`. Fixture gate passed (`max_kl=0.03406`, top-1 agreement
`1.0`, `native_owned_device_bytes=1625645909`, native prefill `0.2607s`).
4K/128 stayed runnable and above the retained 95% guard twice: `649.997` and
`650.006` tok/s versus floor `639.018` (`prefill_seconds=6.3016/6.3015`).
Profiler evidence: short prefill attention launched as
`qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel<true>` 10 times with
`26.362 ms` total / `2636.2 us` average, down from the retained generic
`36.361 ms` total. Decision: keep. Updated benchmark artifact/rollup and
`docs/KERNELS.md`.

## 2026-05-16 — Prefill multiloop iter 42: rejected short-attn token-base precompute

Tried a tiny follow-up to iter 41's split short prefill-attention template:
precompute the two physical token bases (`physical_block << 8`) for the
`SHORT_BLOCK256` kernel and use `base + (token & 255)` in the score/value loops,
removing `physical_block * block_size` from the short hot path. The generic
4K/long-row template remained unchanged.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import build_qwen35_paged_attn_decode
lib = build_qwen35_paged_attn_decode(load=True, require_cached=False)
getattr(lib, 'hipengine_qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans')
print('paged attention prefill symbol loaded')
PY
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter42-attn-short-tokenbase-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter42-attn-short-tokenbase-trace.json
```

Results: targeted tests passed (`37 passed`) and fixture gate passed
(`max_kl=0.03406`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.2593s`). 4K/128 was
runnable and above guard at `666.838 tok/s`. 512/128 samples were `1987.511`,
`1969.814`, `1967.374`, and `1980.131` tok/s (median `1974.973`), only +0.016%
relative to retained `1974.653`. Profiler showed only a tiny local attention
change: short prefill GQA total `26.041 ms` versus retained `26.362 ms`.
Decision: reject/revert. The primary-metric delta is noise, so keeping this
active-path micro-change would violate the real-gain rule despite passing
correctness and 4K guards.

## 2026-05-16 — Prefill multiloop iter 43: token-tiled router logits

Kept a prefill router/shared-gate logits optimization. Current traces showed the
FP16 hidden router logits kernel at ~15.4 ms across 40 MoE launches because it
computed one token/expert dot per block, reloading each BF16 router/shared-gate
weight row for every token. Added `qwen35_router_logits_token_tile_kernel<*,4>`
and routed `tokens >= 4` through a four-token-per-expert block; decode and tiny
rows (`tokens < 4`) still use the original one-token kernel. The output logits
ABI and the existing block-parallel `qwen35_router_select_kernel` are unchanged.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.moe.router import build_qwen35_router
lib = build_qwen35_router(load=True, require_cached=False)
for name in ['hipengine_qwen35_router_logits_fp16', 'hipengine_qwen35_router_topk_shared_out_fp16']:
    getattr(lib, name)
print('router symbols loaded')
PY
python3 -m pytest tests/test_qwen35_router_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/smoke.py --mode qwen35-router-hip --rows 8 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter43-router-tile4-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter43-router-tile4-trace.json
```

Results: targeted tests passed (`37 passed`). Router smoke with `rows=8` hit the
tiled path and matched the NumPy oracle (`selected_match=True`,
`fp16_selected_match=True`, BF16 logits max abs `0.0`, FP16 logits max abs
`2.38e-07`). 512/128 samples were `2037.119`, `2036.564`, `2027.002`, and
`2019.856` tok/s (median `2031.783`), +2.9% over retained `1974.653`. Fixture
gate passed (`max_kl=0.03406`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.2519s`). 4K/128 stayed
runnable and above guard at `656.950 tok/s` (`prefill_seconds=6.2349`, floor
`639.018`). Profiler evidence: `qwen35_router_logits_token_tile_kernel<_Float16,4>`
ran 40 times with `8.683 ms` total / `217.1 us` average, down from the retained
one-token router logits trace (`15.429 ms` total / `385.7 us` average). Decision:
keep. Updated benchmark artifact/rollup and `docs/KERNELS.md`.

## 2026-05-16 — Prefill multiloop iter 44: rejected router logits tile8

Tried widening iter 43's retained router/shared-gate logits prefill tile from four
tokens to eight tokens per expert block for `tokens >= 8`, while preserving the
retained tile4 path for `4..7` tokens and the original one-token kernel for
decode/tiny rows. The goal was to reuse each BF16 router/shared-gate weight row
across more prompt tokens and halve router-logits block count again.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.moe.router import build_qwen35_router
lib = build_qwen35_router(load=True, require_cached=False)
for name in ['hipengine_qwen35_router_logits_fp16', 'hipengine_qwen35_router_topk_shared_out_fp16']:
    getattr(lib, name)
print('router symbols loaded')
PY
python3 -m pytest tests/test_qwen35_router_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/smoke.py --mode qwen35-router-hip --rows 9 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter44-router-tile8-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter44-router-tile8-trace.json
```

Results: build/tests passed (`37 passed`). Router smoke with `rows=9` exercised
the tile8 path plus tail handling and matched the oracle (`selected_match=True`,
`fp16_selected_match=True`, BF16 logits max abs `0.0`, FP16 logits max abs
`4.77e-07`). Fixture gate passed (`max_kl=0.03406`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.2510s`). 4K/128 was
runnable and above guard at `659.063 tok/s`. 512/128 samples were `2038.951`,
`2035.685`, `2031.740`, `2035.087`, `2030.906`, and `2024.218` tok/s (median
`2033.413`), only +0.08% over retained `2031.783`. Profiler showed the local
router-logits kernel did improve (`qwen35_router_logits_token_tile_kernel<_Float16,8>`
`7.279 ms` total versus tile4's retained `8.683 ms` total), but the primary
metric delta is below run noise and the wider kernel is not a real active-path
win. Decision: reject/revert; keep iter 43 tile4.

## 2026-05-16 — Prefill multiloop iter 45: rejected 128-thread shared-expert prefill

Tried routing only the grouped FP16 prefill shared-expert helpers through
128-thread W8A16 reductions while leaving c=1/decode helper defaults at 64
threads. The target was the ~35 ms retained shared-expert prefill bucket
(`w8a16_shared_gate_up_silu_fp16_kernel` +
`w8a16_shared_down_combine_residual_fp16_kernel`) seen in iter 43 traces.

Validation commands:

```bash
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter45-w8a16-shared128-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter45-w8a16-shared128-trace.json
```

Results: targeted tests passed (`58 passed`) and fixture gate passed
(`max_kl=0.03121`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.2669s`). 4K/128 was
runnable and above guard at `644.420 tok/s`. 512/128 samples were `1924.802`,
`1926.749`, and `1915.650` tok/s (median `1924.802`), a -5.3% regression versus
retained `2031.783`. Profiler confirmed the source of the regression: shared
expert down-combine worsened from retained `19.543 ms` total to `33.283 ms`, and
shared gate/up worsened from `15.633 ms` to `17.540 ms`. Decision:
reject/revert; keep the 64-thread grouped shared-expert helper launches.
