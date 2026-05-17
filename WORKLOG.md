# hipEngine Work Log

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

- **Kernel R&D lives in `~/amd-gpu-tuning/`, not here.** Micro-tuning iteration loops (rocprofv3 time-share audit, VGPR / occupancy hunting, `__launch_bounds__` sweeps, fusion experiments, device-code gotcha catalog) all stay in the parent workspace. hipEngine ingests *stable* kernels via the port pipeline in `docs/PLAN.md` "Kernel Port Strategy".
- Consequence: hipEngine's `docs/KERNELS.md` is a port playbook (copy + partition + retype + gate), not a kernel-tuning guide. Tuning guide stays at `~/amd-gpu-tuning/AGENTS.md` and `~/amd-gpu-tuning/LESSONS-LEARNED.md`.
- AGENTS.md "Handling Blockers" redirects kernel-micro-opt and ROCm-restore situations to `~/amd-gpu-tuning/` rather than duplicating the procedures here.

### Doc inventory from `~/amd-gpu-tuning/`

Surveyed 12 `.md` files in `~/amd-gpu-tuning/docs/` plus the top-level design docs. Copied or referenced as follows:

| Upstream doc | Action | Rationale |
| --- | --- | --- |
| `docs/ROOFLINE.md` (1573 lines) | **Copied** to `docs/ROOFLINE.md` | Canonical RDNA3 / W7900 hardware landscape: hardware, roofline fundamentals, regimes, decision tree, what-not-to-chase. Read by anyone planning hipEngine kernels or setting perf targets. Added provenance header; path-qualified companion-doc cross-refs to `~/amd-gpu-tuning/`. |
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

- Commits: `f2a5166` docs: add hipEngine design plan; `f33b2a8` docs: add AGENTS.md ground rules, CLAUDE.md symlink, .gitignore.
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

## 2026-05-12 — License hipEngine as AGPL-3.0-or-later

### Decision

- Selected **AGPL-3.0-or-later** for hipEngine source code.
- Rationale: project is aimed at local/home users, and we explicitly prefer copyleft over permissive/business adoption. AGPL closes the hosted-service loophole that GPLv3 leaves open for an inference engine with optional server/API paths.
- User clarified that the future `nano-vllm-amd` kernel ports are not an upstream-license concern for this decision because those kernels were authored locally by the project lead; still, model weights/checkpoints and external datasets remain under their own licenses.

### Files changed

- Added `LICENSE` containing the full GNU Affero General Public License v3 text from the system SPDX license copy (`/usr/share/licenses/spdx/AGPL-3.0-or-later.txt`).
- Updated `pyproject.toml` project metadata from `Apache-2.0` to `AGPL-3.0-or-later`.
- Updated `README.md` with a License section: hipEngine source code is AGPL-3.0-or-later; model weights, checkpoints, and external datasets keep their own licenses.
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
- That process is not owned by this hipEngine task. Pausing further GPU actions here; do not run rmsnorm port or more profiling until the GPU is explicitly clear again.

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

- User noted hipEngine is becoming "real" software and should have a proper testing story: RED/GREEN, correctness guard/gates, and especially protection against silent math mistakes.
- Goal: adopt useful testing methodology/verbiage from `~/shisad/` and `~/shisad-dev/` without importing irrelevant process (multi-reviewer lanes, release machinery, implement-driven workflow).

### Sources reviewed

- `~/shisad/AGENTS.md`:
  - Useful: Spec → Plan → Test → Implement; write tests first even for ad-hoc work; run targeted tests first; exact command evidence; claim integrity; structural tests are not enough for runtime-facing behavior.
  - Not adopted: shisad-specific security roles, multi-reviewer process, live daemon harness details.
- `~/shisad-dev/AGENTS.md`:
  - Useful: validation cadence proportional to scope; do not default to broad suites for every small change; record validation evidence in worklog; truth-scoped claims.
  - Not adopted: private/public repo split, reviewer-lane rules, release-close process.
- `~/shisad-dev/implement/TEST-COVERAGE.md`:
  - Most relevant source. Key adapted concept: structural correctness is necessary but not sufficient. For shisad the real contract is user-visible correctness; for hipEngine the real contract is numerical correctness against an oracle.
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
- `hipengine/kernels/cpu_reference/ops.py` and `hipengine/kernels/hip_gfx1100/smoke/smoke_add.py` to list kernels/oracles actually landed in hipEngine.
- `~/amd-gpu-tuning/PLAN-PAROQUANT.md` and `~/amd-gpu-tuning/docs/PARO.md` for the current Qwen3.5-35B-A3B-PARO route, shape-gated prefill MoE split, graph replay caveats, 24GB compact path, and recent rejected/alternative routes.
- `~/amd-gpu-tuning/nano-vllm-amd` source inventory:
  - Committed stable Qwen/PARO set: 95 kernels in `csrc/amd/qwen35_expert.hip` + 25 kernels in `nanovllm/native/qwen35/paroquant_kernels.py` = 120 Qwen/PARO kernels, plus separate `smoke_add`.
  - Parent repo observed at `nano-vllm-amd@22405a9` with local modifications in `paroquant.py` and `paroquant_kernels.py`; six additional PARO kernels were documented as lineage-dirty/experimental, not hipEngine defaults.

### Files changed

- `docs/BENCHMARK.md`:
  - Added a benchmark-output contract: exact run context, correctness status/commands, repeated-run statistics, profiler/kernel summary, baseline comparison, and acceptance/rejection reason.
  - Added artifact statuses: `accepted`, `rejected_correctness`, `rejected_variance`, `blocked`.
  - Expanded microbenchmark and E2E measurement statistics requirements: samples, median/p95/min/max/stdev, warmup/measured counts, variance guard.
  - Upgraded retained benchmark JSON schema from `1` to `2` with `status`, command groups, correctness pass/fail fields, measurement samples, memory, profiler top kernels, baseline/comparison, and decision fields.
  - Clarified blocked/rejected attempts are still useful evidence but not retained performance numbers.
- `docs/KERNELS.md`:
  - Renamed to a kernel catalog + port playbook.
  - Added status legend distinguishing hipEngine-landed, CPU-reference-landed, lineage-green, lineage-dirty/experimental, and planned.
  - Added authoritative hipEngine-landed list: CPU-reference oracles (`embed`, `rmsnorm`, `linear`, `qkv_proj`, `rotate`, `attention_decode`, `o_proj`, `lm_head`) and `smoke_add` gfx1100 build/runtime smoke.
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

- User asked for a way to track whether kernel or dispatch files in `~/amd-gpu-tuning/` are newer before continuing hipEngine ports.
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
- Do not advance `docs/source_lineage.json` baseline until hipEngine's catalog/port plan is intentionally refreshed and logged.

---

## 2026-05-13 — Wire OPTIMAL.md into kernel path and hygiene docs

### Prompt / concern

- User noted `~/amd-gpu-tuning/docs/OPTIMAL.md` should be up to date with the optimal PARO inference path and should likely be referenced from hipEngine's kernel catalog.
- User also asked to review `~/amd-gpu-tuning/AGENTS.md` for git/benchmark hygiene worth adopting in hipEngine.
- Follow-up explicit rule requested: before porting, check `docs/KERNELS.md` and use the lineage script to ensure the kernel catalog/path map is up to date.

### Sources reviewed

- `~/amd-gpu-tuning/docs/OPTIMAL.md`:
  - Current optimal path: compact-WMMA prefill + one-step graph-replay decode for Qwen3.5-35B-A3B-PARO.
  - Latest retained sweep: 512/128 `2557 / 115.7`, 1K/128 `2876 / 112.9`, 4K/128 `2703 / 112.0`, 32K/128 `1880 / 98.8`, 128K/128 `914 / 62.6` prefill/decode tok/s, graph/step validation true.
  - 23 base flags, long-prefill chunking overrides, graph replay caveats, and decode profiling note that AWQ/GEMV decode is the next target.
- `~/amd-gpu-tuning/AGENTS.md`:
  - Already covered by hipEngine: explicit staging rules, no destructive cleanup, WORKLOG with logical unit, audit-first kernel tuning, raw artifact exclusion.
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
  - Current fastest hipEngine table (empty until first accepted E2E `LLM.generate()` benchmark).
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

- Reviewed `https://github.com/AICL-Lab/hetero-paged-infer` at commit `a9765bd69aefd8a64591d930867d21ed3dd7fd90` as a potential reference for hipEngine's scheduler / paged-KV / tiered-memory design.
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
- Its KV abstraction is classic uniform fixed-page `block_table + context_len`. This is useful as a small scheduler/block-manager sanity reference, but it is less general than hipEngine's planned `KVLiveSpans` ABI and `KVPolicy.admission_cap()` contract for DMS / H2O / SnapKV / sliding policies.
- No architecture change adopted. If we need a future sanity check for host-only scheduler invariants, its property tests and simple `BlockPool`/`PageTable` model are a reasonable reference. For tiered/offloaded decode scheduling, APEX and Neo are more relevant research references than this repo.

### Next

- Do not port code from this repo into hipEngine.
- Optional future doc update: add it to `docs/PLAN.md` references only as a lightweight Rust host-shape / test-harness reference, not as a kernel or tiered offload source.

---

## 2026-05-13 — Port Qwen3.5 BF16 RMSNorm HIP family

### Scope

- Ported the first real model-layer gfx1100 kernel family into hipEngine: Qwen3.5 BF16 RMSNorm from `~/amd-gpu-tuning/nano-vllm-amd/csrc/amd/qwen35_expert.hip`.
- Source commit: `nano-vllm-amd@59195ed` (`gfx1100-qwen3.5`). The lineage checker reports drift vs baseline `22405a9`, but `git diff 22405a9..HEAD -- csrc/amd/qwen35_expert.hip` shows the RMSNorm region is not touched by the current compact-WMMA drift.

### Files changed

- Added `hipengine/kernels/hip_gfx1100/norm/rmsnorm.hip`:
  - Preserved Qwen kernel bodies for `qwen35_rmsnorm_kernel`, `qwen35_add_rmsnorm_kernel`, `qwen35_add_rmsnorm_f32_kernel`, and `qwen35_head_rmsnorm_kernel`.
  - Added hipEngine C ABI launch wrappers taking raw pointers, shapes, `eps`, and `hipStream_t`.
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

- User suggested using the current `~/amd-gpu-tuning/docs/OPTIMAL.md` MoE path as the next port target so hipEngine can exercise the full `docs/KERNELS.md` checklist, correctness gates, and benchmark robustness against the parent performance rows.

### Source review

- Re-read `docs/KERNELS.md`, `docs/PLAN.md` kernel port strategy, latest WORKLOG entries, and `~/amd-gpu-tuning/docs/OPTIMAL.md`.
- Ran lineage check:

```bash
python3 scripts/check_lineage.py --diff stat --evidence-limit 4
```

Current parent checkout:

- `nano-vllm-amd` branch `gfx1100-qwen3.5`, HEAD `59195ed`.
- Drift vs hipEngine baseline `22405a9` in:
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
  - Explicitly marked current hipEngine status: only Qwen BF16 RMSNorm subset is partial/landed; PARO RMSNorm out-kernels, router, selected GEMV, fused activation/down-rotation, W8A16 shared/lm-head, compact WMMA, attention/KV, model/plugin/loader, and eval harness remain missing.
- Updated `docs/IMPLEMENTATION.md`:
  - Added an OPTIMAL MoE/PARO reproduction exercise punchlist keyed to `docs/KERNELS.md`.

### Key conclusion

- We should not start by copying a random MoE kernel. The fastest path to a meaningful exercise is:
  1. add parent-baseline + hipEngine-blocked benchmark artifacts for 512/128 and 4K/128,
  2. port the MoE c=1 decode vertical slice,
  3. port the compact-WMMA prefill slice,
  4. only then close full inference with loader/model/attention/graph replay.
- Full OPTIMAL inference cannot be replicated yet because hipEngine still lacks `LLM.generate()`, `w4_paro` weight loading/layout, the Qwen3.5 model plugin, attention/KV/linear-attn/lm-head dependencies, and graph replay.

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

## 2026-05-13 — Capture OPTIMAL parent parity artifacts and blocked hipEngine row

### Scope

- Ran the parent `nano-vllm-amd` OPTIMAL Qwen3.5-35B-A3B-PARO command for `512/128` and `4K/128` on W7900 to validate the benchmark output shape and create concrete comparison artifacts before porting more kernels.
- Created a blocked hipEngine artifact for the same parity exercise so the missing dependencies are tracked in `benchmarks/results/`, not just prose.

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
| hipEngine | OPTIMAL parity | — | — | — | not reached | blocked | `benchmarks/results/2026-05-13-hipengine-qwen35-paro-optimal-blocked.json` |

Blocked hipEngine reason: `LLM.generate`, `w4_paro` loader/layout, Qwen3.5 model plugin, MoE/attention/linear/lm-head dependency kernels, and graph replay are not landed yet.

### Files changed

- Added three compact benchmark artifacts under `benchmarks/results/`.
- Updated `benchmarks/README.md` source-lineage rows for 512/128 and 4K/128 to point at artifacts and use the local rerun values.
- Updated `benchmarks/CHANGELOG.md` with lineage-measured deltas and the blocked hipEngine row.
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
- Added hipEngine raw-pointer C ABI wrappers in the existing `norm/rmsnorm.hip` family:
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

- This first hipEngine router wrapper supports BF16 hidden and BF16 combined weights. The parent accepts FP16 or BF16 hidden inputs; if the final hipEngine OPTIMAL route keeps FP16 router inputs, add an FP16 hidden specialization before claiming full router parity.
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
- Next step is a composite hipEngine shared-expert smoke chaining W8A16 gate/up → `silu_mul_dual_out` → W8A16 down, then a c=1 MoE vertical smoke that includes selected W4 experts and shared branch combine.

---

## 2026-05-13 — Add W8A16 shared-expert composite smoke

### Scope

- Added `scripts/smoke.py --mode w8a16-shared-expert-hip` to chain the current parent shared-expert lowp route with existing hipEngine kernels:
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

- Port the committed parent attention/KV decode kernels instead of inventing a new ABI, then adapt wrappers to hipEngine's `KVLiveSpans` ABI at the host boundary.

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

- Port KV append (`qwen35_write_paged_kv_mixed_value*`) and paged full-attention decode from the committed parent kernels, adapting wrappers to hipEngine `KVLiveSpans` instead of changing kernel bodies.

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

- Resume KV append and paged full-attention decode with public wrappers shaped around hipEngine `KVLiveSpans`, while preserving parent kernel bodies internally.

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
  - runs all Qwen3.5/PARO decode layers through hipEngine linear-attention/full-attention c=1 layer chains,
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

- hipEngine actual autoregressive c=1 resident path completed on W7900.
- Shape: 512 prompt tokens, 4 warmup decode tokens, 128 measured decode tokens, repeated token id `9707`.
- Load/materialization: `35.35s`.
- Token-by-token prefill: `5.54s`, `92.39 tok/s` (actual inference, but not native batched/compact prefill).
- Warmed decode: `40.68s`, `3.146 tok/s`; median step `0.3161s`.
- Generated preview repeats token `62843` (`"estring"`).

### Comparison to PLAN-MOE2 2026-05-12 512/128 row

- PLAN-MOE2 parent baseline: prefill `1300.337 tok/s`, decode `131.128 tok/s`.
- hipEngine prefill ratio: `0.071x` of parent, **not comparable** because native prefill is not implemented.
- hipEngine warmed decode ratio: `0.024x` of parent, partially comparable but no graph replay/lower-overhead dispatch yet.

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
- Current PLAN-MOE2 compact-WMMA target is `115.666 tok/s` decode at 512/128; hipEngine is now ~`75.9%` of that decode target, but remains blocked for accepted parity because graph replay and E2E correctness gates are not landed.

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
- Current PLAN-MOE2 compact-WMMA 512/128 decode target: `115.666 tok/s`; hipEngine graph diagnostic is ~`80.1%` of target.

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

This ports the parent `NANOVLLM_PARO_LINEAR_ATTN_QKV_Z_PACK8_FUSED=1` decode route for the hipEngine c=1 path.

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
- Current PLAN-MOE2 compact-WMMA 512/128 decode target is `115.666 tok/s`; hipEngine is now ~`90.0%` of that decode target.
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

This ports the parent `NANOVLLM_PARO_FULL_ATTN_QK_PACK8_FUSED=1` decode route for hipEngine c=1 full-attention layers.

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
- Current PLAN-MOE2 compact-WMMA 512/128 decode target is `115.666 tok/s`; hipEngine is now ~`93.8%` of that decode target.
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
- Current PLAN-MOE2 compact-WMMA 512/128 decode target is `115.666 tok/s`; hipEngine is now ~`95.1%` of that decode target.

Artifact: `benchmarks/results/2026-05-14-hipengine-qwen35-paro-512-128-lmhead128-qk-qkvz-fused-graph-diagnostic.json` (`status=blocked`, diagnostic/non-retained).

---

## 2026-05-14 — Fuse linear-attention A/B dense decode projection

### Scope

- Added `dense_dual_gemv_out_bf16` raw-pointer kernel/wrapper for two small BF16 dense GEMVs with shared input and contiguous output.
- Added contiguous linear-attention `ab` scratch with existing `a`/`b` views.
- Switched `project_linear_attention_ab_bf16` from two dense GEMV launches to the dual dense GEMV.

This ports the parent `NANOVLLM_PARO_LINEAR_ATTN_AB_FUSED=1` decode route for hipEngine c=1 linear-attention layers.

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
- Current PLAN-MOE2 compact-WMMA 512/128 decode target is `115.666 tok/s`; hipEngine is now ~`96.1%` of that decode target.
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
- Captured the code-review conclusion that hipEngine is a better foundation for c>1 than `nano-vllm-amd`, but current Qwen3.5/PARO runtime remains effectively c=1.
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
- Layer-1 parent fixture passed: parent expected token `84`, hipEngine produced `84`; parent prefill seed `6332` matched.
- Full 512/32 parent fixture still blocked: parent prefill seed `4403` matched, but generated-token parity missed at index 0 (`expected 1739`, hipEngine `220`), then the remaining prefix matched (`220,16,15,...`).
- Current hipEngine fixture timing remains sequential-prefill limited: ~113.65 tok/s prefill and ~96.24 tok/s decode vs parent fixture ~2682.66 tok/s prefill and ~116.26 tok/s decode.
- hipEngine memory report is currently owned device buffers (~1.51 GiB), not parent-comparable allocator/VRAM peak (~18.8 GiB), so memory parity still needs a proper process/VRAM measurement path.

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
- Re-ran the 512/32 parent fixture gate. The first decode token is still blocked: dense short-context attention changes hipEngine from `220,...` to `4096,220,16,...`, while the parent expects `1739,220,16,...`.
- Root-cause probe: the parent PARO native fixture runs FP16 activations/scales from the checkpoint (`embed`, RMSNorm, PARO scales/theta, LM head are `torch.float16`), while hipEngine's current Qwen3.5/PARO resident path materializes those runtime tensors as BF16. Layer-0 prompt probes show BF16-vs-FP16 activation drift starts at input RMSNorm/rotation and is enough to flip close top logits after full decode (`parent top first-decode: 1739=6.4487, 220=6.3479, 4096=6.3336`; HIP BF16 dense path: 4096=6.7064, 220=6.5895, 1739=5.9954`).

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
- Full 512/32 parent fixture remains blocked by FP16-vs-BF16 activation parity: parent expected `[1739, 220, 16, ...]`; hipEngine BF16 dense path produced `[4096, 220, 16, ...]` with matching prefill seed `4403`.

### Next

- Decide whether Qwen3.5/PARO parent parity should port FP16 activation variants for the resident path or recapture/define a BF16 parent oracle. Exact generated-token equality against the current parent fixture is not a pure scheduler/cache bug; it crosses the activation dtype boundary.

---

## 2026-05-15 — Make gfx1100 wave32 the documented/build default

### Scope

- Updated the W7900/gfx1100 wavefront policy after the parent workspace probe: `-mcumode` is orthogonal to wavefront size and the HIP decode profile should be treated as wave32 unless `-mwavefrontsize64` is explicitly added for an isolated experiment.
- Added a `docs/PLAN.md` caveat section near the end: RDNA3 wave64 is architecturally real, but hipEngine/nano-vllm-amd defaults to wave32 + ILP/VOPD exposure; wave64 requires separate flags, probes, correctness gates, ISA checks, and E2E benchmarks.
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

- Created `docs/DFLASH.md` as the hipEngine-side plan for a proper native
  DFlash implementation.
- Consolidated lessons from `~/amd-gpu-tuning/PLAN-DFLASH.md`,
  `docs/SPECULATIVE-DECODE.md`, `docs/DFLASH-FRESH-EYES.md`, recent
  2026-05-15 WORKLOG entries, and local references (`reference/ddtree-mlx`,
  `reference/hipfire`, `reference/lucebox-hub/dflash`).
- Main decision recorded: the current Python/PyTorch DFlash harness has proven
  correctness and the corrected tree-kernel shape, but the remaining speed gap
  is a native-runtime verifier problem. The production path belongs in
  hipEngine as a torch-free C++/HIP hot loop with stable buffers, persistent
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
assert '# hipEngine DFlash / DDTree Native Implementation Plan' in text
assert 'DDTree details to preserve' in text
assert 'First concrete hipEngine tasks' in text
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

- Use `docs/DFLASH.md` as the launch checklist when starting hipEngine DFlash:
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

- Source-lineage check reported expected parent drift from `nano-vllm-amd` after baseline `22405a9` (`qwen35_expert.hip`, `smoke.hip`, `paroquant_kernels.py`); this iteration used existing stable hipEngine bodies and added local dtype/helper variants only.
- Active-loop guard passed: 68 targeted tests passed.
- Extended local test bundle passed (`[100%]`, 80 test dots across cast/runtime/rotary/KV/generation/layout suites).
- Parent fixture correctness remains blocked but narrower: full resident c=1 fixture run was finite/deterministic, but HIP seed token was `220` and first decode token was `58` with top logit `9.434697151184082`; parent fixture expected first generated token `1739`. No performance claim retained.
- `rocprofv3` W7900 evidence:
  - rotary smoke: `partial_max_abs=0`, `head_max_abs=2.38e-07`, `position_max_abs=2.38e-07`, `split_fp16_query_max_abs=0`, `split_fp16_gate_mismatch=0`; dispatch included `qwen35_split_qgate_fp16_kernel`, `DurationNs=3720`.
  - paged-KV smoke: `mixed_mismatch=0/0`, `mixed_fp16_mismatch=0/0`, `f32_mismatch=0/0`, `untouched_nonzero=0`; dispatch included `qwen35_write_paged_kv_mixed_value_position_tensor_kernel<_Float16>`, `DurationNs=5400`.

### Loop record

- Marked Task #40 completed. Active loop iteration 9 recorded `open_or_partial_items=8` (down from 9), guard pass, prompt-verifier pass, and explicit `parent_fixture_e2e_blocker` failure with token/logit evidence for Task #41.

### Next

- Start Task #41: promote/narrow the parent fixture by bisecting the remaining c=1 parity gap at per-layer hidden/logit checkpoints. The broad BF16-vs-FP16 activation policy is no longer the only blocker; next evidence should identify the first layer or projection where hipEngine diverges from parent.
- Hot-path torch audit for touched runtime/generation paths:
  `rg -n "^\\s*import torch|^\\s*from torch" hipengine/runtime hipengine/generation hipengine/llm.py hipengine/loading/qwen35_paro.py || true` → no executable torch imports.

---

## 2026-05-15 — Narrow Qwen3.5/PARO c=1 fixture after parent-mixed switch

### Scope

- Started Task #36 after closing the activation-parity umbrella and reproduced the post-FP16-switch blocker.
- Parent-source audit found two materialization mismatches against `nano-vllm-amd`:
  - native router/shared-gate concatenates `router.weight` and `shared_expert_gate.weight` then casts the combined matrix to BF16 before `qwen35_router_logits_kernel`;
  - fused q/k head RMSNorm+RoPE receives BF16 *offset* weights computed as `(checkpoint + 1 -> FP16 -> BF16 -> -1)`, not checkpoint-direct BF16.
- Updated hipEngine runtime materialization accordingly and refreshed layout tests.
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
  - `/tmp/hipengine_prefix_probe.py` runs hipEngine resident c=1 on the 512-token fixture for selected `max_layers` prefixes.
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
  - Parent rotates the MoE gate/up input via `experts.gate_up_weight_{pairs,theta,channel_scales}` before selected gate/up pack8 GEMV; hipEngine was feeding the unrotated post-norm hidden into selected gate/up GEMV.
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

- No hipEngine throughput row was promoted. The c=N accepted gate is explicitly serial (`step_batch_serial`) and remains a correctness gate only.
- The current-fastest hipEngine throughput table still has no accepted row; Task #15 remains the native compact/c-aware path needed before retaining c>N performance claims.

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

- Clarified that `w4a16` is the broad 4-bit-weight/16-bit-activation quantization class, while `w4_paro` is hipEngine's concrete PARO AWQ packed-layout/plugin variant under that umbrella.
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

- The original pi-multiloop verify command still fails with `active hipEngine parity TaskList not found` because it searches for completed task IDs that are no longer in the active compacted TaskList file. A robust count over the active #12/#15 file still prints `2`.

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

- The original pi-multiloop verify command remains stale and exits `active hipEngine parity TaskList not found` because it requires completed task IDs no longer present in the compacted active TaskList file. A robust count over the active #12/#15 file prints `2`.
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

- The original pi-multiloop verify command remains stale and exits `active hipEngine parity TaskList not found`; robust active TaskList count over #12/#15 remains `2`.
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
- The original pi-multiloop verify command remains stale and exits `active hipEngine parity TaskList not found`; robust active TaskList count over #12/#15 remains `2`.

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

- The original pi-multiloop verify command remains stale and exits `active hipEngine parity TaskList not found`; robust active TaskList count over the compacted active file is now `1` (#15 only).

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

- The original pi-multiloop verify command remains stale and exits `active hipEngine parity TaskList not found`; robust active TaskList count remains `1` (#15 only).
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

- The original pi-multiloop verify command remains stale and exits `active hipEngine parity TaskList not found`; robust active TaskList count remains `1` (#15 only).
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
- The original pi-multiloop verify command remains stale and exits `active hipEngine parity TaskList not found`.
- No performance claim was made; the new stage-bisect artifact is explicitly rejected correctness evidence.

---

## 2026-05-15 — Rename display branding to hipEngine

### Scope

- Completed Task #42: user-facing project display name now uses `hipEngine` while the import/package remains `hipengine`.
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
  - kept `Current fastest hipEngine rows` empty;
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
- Updated `docs/DFLASH.md` with the current hipEngine status and artifact link.
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

## 2026-05-15 — Qwen3.5-0.8B-PARO hipEngine feasibility check blocked

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

hipEngine cannot currently run `z-lab/Qwen3.5-0.8B-PARO` for the requested
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

Needed before rerunning in hipEngine: tied lm-head fallback plus dense PARO MLP
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
was copied from those drifted new kernels for this hipEngine-local vector RoPE
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
`paroquant_kernels.py` from nano-vllm-amd head `5d8f496`; no additional hipEngine
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
  slab `cu_seqlens` ABI. This task implemented the hipEngine slab ABI directly
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

Audit-only iteration after retaining shared prefill scratch. Wrapped hipEngine
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
is disabled on W7900. In hipEngine this regressed badly, so the change is
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
hipEngine's transposed projection kernels remain faster for this path despite
the parent engine's different optimal flag stack.

## 2026-05-15 — Prefill multiloop iter 10: 64-thread prefill projection GEMV

Tuned multi-token transposed pack8 prefill projections. The previous strided
layout trial showed transposed qweights are the right layout in hipEngine, so
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
review, pivoted to the major algorithmic gap versus the parent route: hipEngine
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
into hipEngine's raw-pointer W8A16 library as
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

## 2026-05-16 — Prefill multiloop iter 46: rejected W8A16 shared fast exp

Tried replacing the W8A16 shared-expert helper `expf` calls with device fast
`__expf` in the FP16 shared gate/up SiLU and shared down-combine gate sigmoid.
The change was scoped to `hipengine/kernels/hip_gfx1100/quant/w8a16_linear.hip`;
ABI, dispatch, and c=1/non-grouped helper structure were unchanged.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import build_w8a16_linear
lib = build_w8a16_linear(load=True, require_cached=False)
for name in ['hipengine_w8a16_shared_gate_up_silu_fp16', 'hipengine_w8a16_shared_down_combine_residual_fp16']:
    getattr(lib, name)
print('w8a16 shared symbols loaded')
PY
python3 -m pytest tests/test_qwen35_moe_group_scatter_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/smoke.py --mode w8a16-linear-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/smoke.py --mode w8a16-shared-expert-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter46-w8a16-fastexp-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter46-w8a16-fastexp-trace.json
```

Results: targeted tests passed (`37 passed`), W8A16 linear smoke passed, and the
shared-expert smoke stayed bit-exact (`gate_up_mismatch=0`,
`intermediate_mismatch=0`, `out_mismatch=0`). Fixture gate passed
(`max_kl=0.01970`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.2506s`). 4K/128 was
runnable and above guard at `658.373 tok/s`. 512/128 samples were `2030.564`,
`2036.835`, `2030.852`, `2036.902`, `2017.649`, and `2014.217` tok/s (median
`2030.708`), -0.05% versus retained `2031.783`. Profiler showed a real local
bucket improvement — shared down-combine `19.543 -> 18.989 ms` and shared gate/up
`15.633 -> 14.801 ms` total — but the primary metric did not improve and the
sample spread overlaps retained noise. Decision: reject/revert; keep precise
`expf` in the default path unless a later fused/shared-expert change turns this
into a clear end-to-end win.

## 2026-05-16 — Prefill multiloop iter 47: rejected compact WMMA launch_bounds 32x4

After three consecutive rejected helper/fast-math micro-tunes, switched to the
next large profiler bucket: compact WMMA selected-MoE gate/up+down. Tried raising
both compact WMMA kernels from `__launch_bounds__(32, 2)` to
`__launch_bounds__(32, 4)` to test whether stronger occupancy guidance improved
W7900 residency without changing math, ABI, or dispatch.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.wmma.paro_awq_wmma import build_paro_awq_wmma
lib = build_paro_awq_wmma(load=True, require_cached=False)
for name in ['hipengine_gemm_awq_selected_dual_pack8_wmma_compact_fp16', 'hipengine_gemm_awq_selected_pack8_wmma_compact_fp16']:
    getattr(lib, name)
print('wmma compact symbols loaded')
PY
python3 -m pytest tests/test_qwen35_moe_group_scatter_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/smoke.py --mode paro-awq-wmma-compact-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter47-wmma-lb32x4-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter47-wmma-lb32x4-trace.json
```

Results: targeted tests passed (`37 passed`) and compact WMMA smoke remained
correct (`dual_mismatch=0`, `single_mismatch=0`, FP16 mismatches `0`). Fixture
gate passed (`max_kl=0.03406`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.2531s`). 4K/128 was
runnable and above guard at `655.964 tok/s`. 512/128 samples were `2036.300`,
`2017.026`, `1998.419`, `2009.627`, and `2002.845` tok/s (median `2009.627`),
-1.1% versus retained `2031.783`. Profiler showed no meaningful WMMA bucket win:
dual compact WMMA `34.245 -> 33.827 ms`, single compact WMMA `19.404 -> 19.346`
ms, while whole-prefill timing regressed. Decision: reject/revert; keep
`__launch_bounds__(32, 2)`.

## 2026-05-16 — Prefill multiloop iter 48: retained shared-gate sigmoid precompute

After four consecutive rejected micro-tunes, refined the shared-expert lane by
removing redundant work instead of changing thread count or math. The grouped
FP16 shared down+combine kernel was recomputing the same shared-gate sigmoid once
per hidden-row tile (`ceil(hidden_size/4)` times per token). Added
`hipengine_w8a16_shared_gate_sigmoid_fp32`, which overwrites the existing router
shared-gate logit column in place with `sigmoid(logit)` after top-k/routing
weights are already materialized, then changed fused shared down+combine to
consume the cached FP32 gate. This adds no persistent scratch allocation and
keeps the grouped-MoE ABI/dispatch path unchanged.

Validation commands:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/quant/w8a16_linear.py hipengine/runtime/qwen35_paro.py
python3 -m pytest tests/test_w8a16_linear_plan.py tests/test_qwen35_decode_state.py -q
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import build_w8a16_linear
lib = build_w8a16_linear(load=True, require_cached=False)
for name in ['hipengine_w8a16_shared_gate_up_silu_fp16', 'hipengine_w8a16_shared_gate_sigmoid_fp32', 'hipengine_w8a16_shared_down_combine_residual_fp16']:
    getattr(lib, name)
print('w8a16 shared symbols loaded')
PY
python3 scripts/smoke.py --mode w8a16-linear-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter48-shared-gate-sigmoid-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter48-shared-gate-sigmoid-trace.json
```

Results: targeted tests passed (`37 passed`) and W8A16 smoke stayed green
(`bf16_f32_max_abs=0.0`, `f32_f32_max_abs=4.77e-07`, `lowp_mismatch=0`,
`fp16_lowp_mismatch=0`). 512/128 samples were `2061.288`, `2040.598`,
`2037.803`, `2042.010`, `2038.149`, `2039.177`, and `2025.894` tok/s (median
`2039.177`), +0.36% over retained `2031.783`. Fixture gate passed
(`max_kl=0.03406`, top-1 agreement `1.0`, `native_owned_device_bytes=1625645909`,
native prefill `0.2519s`). 4K/128 remained above the no-regression guard at
`656.630 tok/s` (`prefill_seconds=6.2379`, decode `101.859 tok/s`). Profiler
evidence: `shared_gate_sigmoid_fp32_kernel` ran 40 times for only `0.091 ms`
total, while `w8a16_shared_down_combine_residual_fp16_kernel` ran 40 times for
`18.768 ms` total / `469.2 us` avg versus the prior retained bucket orientation
around `19.5 ms`. Decision: keep. Updated the benchmark diagnostic artifact,
rollup, changelog, and kernel catalog.

## 2026-05-16 — Prefill multiloop iter 49: rejected fused-W4 launch_bounds 32x16

Tried the parent/hipfire-inspired RDNA3 occupancy lever on the fused W4->WMMA
prefill projection bucket. The hot `awq_fusedw4_prefill_fp16_kernel<32,32,*>()`
reported `VGPR=120` with `__launch_bounds__(32, 8)`, so the trial raised only
that kernel to `__launch_bounds__(32, 16)` to see whether stronger residency
pressure improved the ~37 ms fused-W4 bucket without changing math, ABI, tile
shape, or call sites.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import build_paro_awq_gemv
lib = build_paro_awq_gemv(load=True, require_cached=False)
for name in ['hipengine_awq_fusedw4_prefill_fp16', 'hipengine_awq_fusedw4_prefill_strided_fp16']:
    getattr(lib, name)
print('awq fusedw4 symbols loaded')
PY
python3 -m pytest tests/test_paro_awq_gemv_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/smoke.py --mode paro-pack8-gemv-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter49-fusedw4-lb32x16-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter49-fusedw4-lb32x16-trace.json
```

Results: targeted tests passed (`37 passed`) and pack8 smoke stayed bit-exact
(`single_mismatch=0/0`, `dual_mismatch=0/0`, FP16 mismatches `0/0`). Fixture
gate passed (`max_kl=0.03406`, top-1 agreement `1.0`,
`native_owned_device_bytes=1625645909`, native prefill `0.2753s`) and 4K/128 was
still above guard at `652.284 tok/s`. However 512/128 regressed badly:
`1872.188`, `1853.464`, and `1829.515` tok/s (median `1853.464`), -9.1% versus
retained `2039.177`. Profiler explained the regression: forcing `(32,16)`
lowered fused-W4 VGPR to `96` but introduced scratch (`20 B`) in the transposed
kernel and made the strided `<32,32,false>` bucket explode to `37.590 ms` versus
retained `14.616 ms`; the transposed bucket also worsened to `28.006 ms` versus
`22.999 ms`. Decision: reject/revert; keep `__launch_bounds__(32, 8)` for fused
W4 prefill and do not revisit a stricter occupancy bound without a parent-side
kernel rewrite that avoids spills.

## 2026-05-16 — Prefill multiloop iter 50: retained shared-down tile8

Retuned the grouped FP16 W8A16 shared down+combine kernel after the retained
shared-gate sigmoid precompute. The kernel still spent ~18.8 ms per 512 prefill
trace computing four hidden output rows per block. Changed only the fused shared
down+combine tile from 4 to 8 hidden rows per block, doubling the per-block
partials/LDS (`1024 -> 2048 B` at 64 threads) but halving the hidden-row grid
and reusing each `shared_intermediate` load across eight output rows. The
precomputed shared gate and per-row FP32 accumulation order are unchanged.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import build_w8a16_linear
lib = build_w8a16_linear(load=True, require_cached=False)
for name in ['hipengine_w8a16_shared_gate_sigmoid_fp32', 'hipengine_w8a16_shared_down_combine_residual_fp16']:
    getattr(lib, name)
print('w8a16 shared symbols loaded')
PY
python3 -m pytest tests/test_w8a16_linear_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/smoke.py --mode w8a16-linear-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter50-shared-down-tile8-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter50-shared-down-tile8-trace.json
```

Results: targeted tests passed (`37 passed`) and W8A16 smoke stayed green
(`bf16_f32_max_abs=0.0`, `f32_f32_max_abs=4.77e-07`, `lowp_mismatch=0`,
`fp16_lowp_mismatch=0`). 512/128 samples were `2063.545`, `2077.639`,
`2061.730`, `2055.635`, and `2048.585` tok/s (median `2061.730`), +1.1% over
retained `2039.177`. Fixture gate passed (`max_kl=0.03406`, top-1 agreement
`1.0`, `native_owned_device_bytes=1625645909`, native prefill `0.2511s`).
4K/128 remained runnable and slightly improved to `659.356 tok/s`
(`prefill_seconds=6.2121`, decode `102.647 tok/s`), above the 95% guard.
Profiler evidence: `w8a16_shared_down_combine_residual_fp16_kernel` ran 40 times
with `16.047 ms` total / `401.2 us` avg, down from the retained tile4 trace's
`18.768 ms` total / `469.2 us`; grid_x halved `32768 -> 16384`, LDS doubled
`1024 -> 2048 B`, VGPR stayed `32`, scratch stayed `0`. Decision: keep. Updated
benchmark artifact/rollup and `docs/KERNELS.md`.

## 2026-05-16 — Prefill multiloop iter 52: 4K profile pivot and rejected GQA2 prefill attention

Pivoted from the planned iter-51 full-attention `__expf` micro-tune after user
feedback correctly pointed at the much larger 4K/128 gap. First profiled the
current retained code at 4K prefill (no code changes):

```bash
rocprofv3 --kernel-trace -d /tmp/iter52-4k-profile-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter52-4k-profile-trace.json
sqlite3 -header -csv /tmp/iter52-4k-profile-trace/trace_results.db "select name, vgpr_count, scratch_size, lds_size, grid_x, grid_y, workgroup_x, count(*) n, sum(duration)/1e6 total_ms, avg(duration)/1e3 avg_us, max(duration)/1e3 max_us from kernels group by name order by sum(duration) desc limit 30;"
```

The profile confirmed the 4K problem is dominated by full-attention prefill, not
launch overhead or MoE: total traced kernel time was `6171.1 ms` and
`qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel<false>` alone consumed
`4572.4 ms` across 10 launches (`457.2 ms/layer`, `grid=(1024,4096)`, 64
threads, `LDS=17664`). The current 512 trace has the same bucket at only
`26.158 ms`, so attention scaled ~175x when prompt length scaled 8x; that
accounts for most of the 4K/128 gap. Other buckets scaled roughly linearly or
sub-quadratically: GDN `41.217 -> 391.644 ms`, compact WMMA dual `33.959 ->
199.959 ms`, fused W4 true `23.213 -> 170.159 ms`, shared down `16.047 ->
124.891 ms`.

Tried a structural GQA-pair prefill-attention kernel for the long 4K path:
compute two adjacent query heads sharing one KV head in a block, with two score
arrays in LDS, to reuse K/V reads and halve the q-head grid. The first build was
mistakenly guarded for `head_dim==128` and did not launch; after correcting the
Qwen full-attn head dim to `256`, the new kernel did launch.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import build_qwen35_paged_attn_decode
lib = build_qwen35_paged_attn_decode(load=True, require_cached=False)
getattr(lib, 'hipengine_qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans')
print('attention prefill symbol loaded')
PY
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
rocprofv3 --kernel-trace -d /tmp/iter52-gqa2h256-4k-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter52-gqa2h256-4k-trace.json
```

Results: attention smoke remained bit-exact for the short fixture
(`prefill_gate_fp16_max_abs=0`, mismatch `0`) and the 512 fixture gate passed
(`max_kl=0.03406`, top-1 `1.0`, owned bytes `1625645909`). However the GQA2
long path regressed badly: 4K/128 prefill fell to `505.058 tok/s`
(`prefill_seconds=8.1100`) versus retained `659.356`, violating the 95% 4K
no-regression guard. The profiler explains why: the new
`qwen35_paged_full_attn_prefill_gqa2_gate_fp16_kernel<false>` did halve the
q-head grid to `(512,4096)` and ran with no scratch, but VGPR rose `32 -> 48`,
LDS doubled `17664 -> 35328`, and the attention bucket worsened to `6431.6 ms`
from `4572.4 ms`. Decision: reject/revert the GQA2 kernel. Lesson: for this
one-block-per-row prefill kernel, pairing query heads loses more occupancy/LDS
than it saves in KV reuse; the 4K fix likely needs a different attention
algorithm (split/tiled context, grouped-GQA with smaller score tiles, or a
FlashAttention-style online softmax) rather than fatter per-row LDS blocks.

## 2026-05-16 — Prefill multiloop iter 53/54: rejected long-attention 128 threads under 512 loop metric

After iter 52 showed 4K/128 is dominated by the single-request full-attention
prefill kernel (`4572.4 ms` / 10 launches at 64 threads), tried the smallest
shape-targeted long-row change before a larger tiled/online-softmax rewrite:
raise only `max_context_len > 1024` full-attention prefill launches from 64 to
128 threads. The 512 short-row path remains the existing 32-thread one-wave
branch, so this is a 4K-focused launch-geometry change rather than another 512
micro-tune. The earlier 256-thread launch remains avoided because it was
repeat-nondeterministic on gfx1100.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import build_qwen35_paged_attn_decode
lib = build_qwen35_paged_attn_decode(load=True, require_cached=False)
getattr(lib, 'hipengine_qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans')
getattr(lib, 'hipengine_qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans')
print('attention prefill symbols loaded')
PY
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 -m pytest tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter53-attn128-4k-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter53-attn128-4k-trace.json
```

Results: attention smoke remained bit-exact (`prefill_gate_fp16_max_abs=0`,
`prefill_gate_fp16_mismatch=0`) and targeted tests passed (`37 passed`). The
512 verification samples during this hot/long-run session were `2036.091`,
`2022.316`, `2020.019`, and exact-verify `2012.584` tok/s (median `2021.167`),
below the retained 512 median `2061.730`; however the edited branch is
`max_context_len > 1024` only, and 512 still uses the same 32-thread short-row
kernel as iter 50. Fixture gate passed unchanged (`max_kl=0.03406`, top-1
`1.0`, `native_owned_device_bytes=1625645909`, native prefill `0.2527s`).
4K/128 improved from retained `659.356 tok/s` to `1021.001` and `1017.177`
tok/s samples (`prefill_seconds=4.0268`, decode `102.329 tok/s` on the guard
run), a +54.3% 4K prefill gain and well above the 95% guard. Profiler evidence:
`qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel<false>` ran 10 times with
`2416.235 ms` total / `241.624 ms` average at `workgroup_x=128`, `LDS=17920`,
`VGPR=32`, scratch `0`; the retained 64-thread 4K profile was `4572.375 ms` /
`457.238 ms` average. Initial decision intent was to keep for the user-requested
4K pivot, but the active multiloop's recorded primary metric is still 512/128;
action had to be recorded as revert because 512 measurements failed that legacy
acceptance gate. Benchmark rollup/catalog edits were reverted, and this note is
the retained evidence for a future 4K-primary re-scope. 4K is still far below
the parent target, so next 4K work should continue on attention algorithm
structure (not GQA2 fat LDS, which iter 52 rejected).

### Iter 54 rerun note — GPU contention check for long-attention 128-thread trial

User reported the GPU was in use during the iter-53 512 verification, so before
reverting the 4K-focused long-attention 128-thread trial I reran the same code
with `rocm-smi` showing GPU use `0%` and VRAM `0%` at the start. No additional
code changes were made. Commands repeated the standard verify/guard plus two 4K
runs:

```bash
rocm-smi --showuse --showmemuse --showtemp
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter54-rerun-512-128-N.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter54-rerun-4k-128-N.json
```

Rerun 512/128 samples were `2039.063`, `2041.361`, `2032.537`, `2034.592`, and
`2031.609` tok/s (median `2034.592`), still below the retained 512 median
`2061.730`, although the edited branch is `max_context_len > 1024` only and the
512 short-row kernel still launches at 32 threads. Fixture gate remained green
(`max_kl=0.03406`, top-1 `1.0`, owned bytes `1625645909`). The 4K/128 rerun
confirmed the large improvement: `1020.076` and `1018.439` tok/s (median
`1019.258`, +54.6% versus retained `659.356`), with decode `102.405 tok/s` on
the representative run. Because the active loop still accepts/rejects on 512/128,
the trial code was reverted despite the 4K evidence. This should be the first
candidate to restore if/when the loop is re-scoped with 4K/128 as the primary
metric.

## 2026-05-16 — Prefill multiloop iter 55: rejected shared-down tile16

After three attention-focused failures under the still-512-primary loop, returned
to a 512-visible bucket with the same structural idea as iter 50. The retained
W8A16 shared down+combine tile8 kernel still cost `16.047 ms` across 40 launches
at 512 and `~125 ms` at 4K. Changed the fused FP16 shared down+combine tile from
8 to 16 hidden rows per block using a `ROW_TILE` constexpr, halving the
hidden-row grid again while preserving the precomputed shared gate, grouped-MoE
ABI, and exact per-row FP32 accumulation order. This doubles partial LDS from
`2048 -> 4096 B` at 64 threads and increases accumulator pressure, so profiler
checks were required.

Validation commands:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import build_w8a16_linear
lib = build_w8a16_linear(load=True, require_cached=False)
getattr(lib, 'hipengine_w8a16_shared_down_combine_residual_fp16')
print('w8a16 shared down symbol loaded')
PY
python3 scripts/smoke.py --mode w8a16-linear-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 -m pytest tests/test_w8a16_linear_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json >/tmp/multiloop-prefill-4k-128.stdout 2>/tmp/multiloop-prefill-4k-128.stderr
rocprofv3 --kernel-trace -d /tmp/iter55-tile16-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter55-tile16-trace.json
```

Results: W8A16 smoke stayed green (`bf16_f32_max_abs=0.0`, `f32_f32_max_abs=4.77e-07`,
`lowp_mismatch=0`, `fp16_lowp_mismatch=0`) and targeted tests passed
(`37 passed`). 512/128 samples were `2089.537`, `2072.115`, `2059.141`,
`2074.663`, and `2052.684` tok/s (median `2072.115`), +0.50% over retained
`2061.730` but smaller than run-to-run variance (`MAD=12.974`). Fixture gate
passed unchanged (`max_kl=0.03406`, top-1 `1.0`, `native_owned_device_bytes=1625645909`).
4K/128 remained above guard at `660.088 tok/s` (`prefill_seconds=6.2052`, decode
`102.411 tok/s`). Profiler evidence: `w8a16_shared_down_combine_residual_fp16_kernel`
ran 40 times with `14.742 ms` total / `368.6 us` average, down from tile8
`16.047 ms` / `401.2 us`; grid_x halved `16384 -> 8192`, LDS doubled
`2048 -> 4096 B`, VGPR rose `32 -> 40`, scratch stayed `0`. Decision: reject/revert
because the active loop judged the median gain insufficient relative to variance;
retain tile8 and do not revisit tile16 unless a broader 4K/attention re-scope
needs the local shared-down improvement.

## 2026-05-16 — Prefill multiloop iter 56: dual fused-W4 prefill projection trial

After four consecutive failed geometry/micro-tune iterations, switched back to a
parent-proven structural lever: paired transposed W4 prompt projections. Added a
transposed `hipengine_awq_fusedw4_prefill_dual_fp16` path that launches one WMMA
prefill kernel over the concatenated output-tile grid for two independent
projections while still writing separate contiguous outputs. Runtime routing now
uses it for full-attention Q/K and linear-attention QKV/Z FP16 prefill pairs;
V/O and linear-attention out_proj stay on the existing strided/single paths. The
intent is launch reduction and better single-launch occupancy without changing
per-output math or tensor layout.

Validation commands/results before loop decision:

```bash
git diff --check
python3 -m py_compile hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.py hipengine/kernels/hip_gfx1100/quant/__init__.py hipengine/runtime/qwen35_paro.py
python3 -m pytest tests/test_paro_awq_gemv_plan.py tests/test_qwen35_decode_state.py -q
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import build_paro_awq_gemv
lib = build_paro_awq_gemv(load=True, require_cached=False)
for name in ('hipengine_awq_fusedw4_prefill_fp16','hipengine_awq_fusedw4_prefill_dual_fp16','hipengine_awq_fusedw4_prefill_strided_fp16'):
    getattr(lib, name)
print('fusedw4 symbols loaded')
PY
python3 scripts/smoke.py --mode paro-pack8-gemv-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter56-dual-fusedw4-512-128-N.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter56-dual-fusedw4-4k-128-N.json
rocprofv3 --kernel-trace -d /tmp/iter56-dual-fusedw4-trace -o trace -- python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/iter56-dual-fusedw4-trace.json
```

GPU was idle at the start (`rocm-smi` GPU use `0%`, VRAM `0%`). Targeted tests
passed (`37 passed`), fused-W4 symbols loaded, and pack8 smoke stayed bit-exact
(`single_mismatch=0/0`, `dual_mismatch=0/0`, FP16 mismatches `0/0`). 512/128
samples were `2080.018`, `2074.859`, `2084.484`, `2079.664`, `2060.165`, and exact-verify `2041.730`
tok/s (median `2077.262`, +0.75% over retained `2061.730`, `MAD=4.989`). Fixture
gate passed unchanged (`max_kl=0.03406`, top-1 `1.0`, `native_owned_device_bytes=1625645909`).
4K/128 samples were `660.922` and `658.977` tok/s (median `659.950`, +0.09% vs
retained `659.356`, above the 95% guard but effectively neutral). Profiler
confirmed the structural change: transposed fused-W4 Q/K and QKV/Z now launch
`awq_fusedw4_prefill_dual_fp16_kernel<32,32>` 40 times (`21.957 ms` total,
`548.9 us` avg, `grid_x=12288`, `VGPR=120`, scratch `0`) instead of 80 single
transposed fused-W4 launches totaling about `22.999-23.640 ms` in nearby retained
traces; strided fused-W4 remains separate at 50 calls / `14.795 ms`.

## 2026-05-16 — docs/PREFILL.md: AOTriton diagnosis + pinning strategy

Added a standalone "Optimization diagnosis (2026-05-16): the 4K gap is one
kernel" section to `docs/PREFILL.md` (between "Compact c>N prefill done" and
"References"). Captures:

- Where we stand: hipEngine 2039 tok/s @ 512 / 659 tok/s @ 4K vs nano-vllm-amd
  2589 / 1681 (+27 % / +155 %). 4K is the load-bearing gap.
- Trace comparison: top-10 kernel buckets at hipEngine 512/0 (229.77 ms total
  kernel time) and 4K/0 (6171.07 ms total), summed across 40 layers. Headline:
  `qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel<false>` is 4572 ms /
  10 launches / 457 ms per layer at 4K, 74 % of all 4K kernel time. 8× tokens
  → 175× the kernel time vs the `<true>` split-K branch at 512. Super-quadratic
  scaling; this is missing Flash-Attention, not under-tuned.
- Why the existing kernel mis-scales: `<false>` materializes the full T×T
  score tile / reloads K/V per Q sweep; no online softmax, no LDS-tiled K/V
  stream. The multiloop missed it because its primary metric is 512/128, where
  the `<true>` branch masks the issue, and the 4K guard only requires
  not-OOM.
- Options matrix for fast prefill attention without `torch` on the hot path:
  AOTriton 0.8 standalone C++ ABI (2–3 days, ≈ 1700 tok/s at 4K), hand-rolled
  HIP FA-2 (3–6 weeks, 1300–1900 tok/s), CK ck_tile/01_fmha (no gfx11), vLLM
  CK FA fork (builds gfx1100 but unproven), Dao-AILab FA-2 (CDNA-only),
  patching `<false>` in place (rejected: it is an algorithmic gap, not a
  tuning gap).
- "Why surely native HIP beats Triton is not a fast path": Triton lowers to
  AMDGPU LLVM IR through MLIR and emits the same instruction class (`v_wmma_*`,
  `ds_read_b128`, `v_dual_*`); the 5–15 % Triton tax on gfx1100 is real but
  small. The current kernel is slow because it is not FA, not because it is
  HIP.
- Recommended phased plan: Phase 1 wrap AOTriton as the long-T attention
  prefill variant (registered, not branched); Phase 2 wrap hipBLASLt for MoE
  projection at T ≥ 1K; Phase 3 optional native HIP FA-2 port with AOTriton as
  the perf/correctness oracle.

Also added an "AOTriton distribution and pinning strategy" subsection
documenting concretely how to bring AOTriton in without taking on its ABI
churn:

- Pin a manifest (`hipengine/kernels/hip_gfx1100/attention/aotriton_release.toml`)
  recording version `0.8.2b`, git SHA `33fb6bd5290b2e9e9bc71dbcf91f92c6ba7689b1`,
  SONAME `libaotriton_v2.so.0.8.0`, ROCm-minor band, tarball URL, SHA256, and
  prune rules (`amd-gfx110x` × `attn_fwd` × bf16/fp16 × head_dim 128 ×
  causal=true). Pruned footprint ≈ 30 MB; the unpruned tarball is 374 MB.
- Fetch-on-install via `scripts/fetch_aotriton.sh` (and Python twin
  `hipengine.aotriton.ensure_installed()`) into
  `~/.cache/hipengine/aotriton/<version>/`. Not run by `pip install`.
- Lookup chain at module load: `HIPENGINE_AOTRITON_LIB` env override →
  `~/.cache/hipengine/aotriton/<version>/` → optional `/opt/rocm/lib/` →
  fallback to the existing hand-rolled kernel with a one-shot diagnostic so
  generation never crashes.
- No git submodule (build is hours, needs AMDGPU Triton fork, not in our
  surface area), no vendored binary in the repo (AGENTS.md git rules + clone
  bloat), no PyTorch-bundled AOTriton (couples us to a torch we explicitly do
  not import).
- Stable-ABI shim, not raw `ctypes` against mangled C++ symbols: build a
  `hipengine/kernels/hip_gfx1100/attention/aotriton_wrap.{cc,py}` that
  includes `<aotriton/flash.h>`, links `libaotriton_v2.so` at JIT-build time,
  and exposes a small `extern "C"` ABI we own. Bumping AOTriton then touches
  one wrapper file plus the manifest; the kernel registry key, dispatch path,
  and Python ABI stay unchanged.

Evidence inspected during write-up:

- `~/Downloads/aotriton/aotriton/include/aotriton/{flash,runtime,config}.h`
  for ABI surface, namespace, `TensorView`, `Stream` typedef.
- `nm -D --defined-only ~/Downloads/aotriton/aotriton/lib/libaotriton_v2.so |
  grep flash` → confirmed `aotriton::v2::flash::attn_fwd`,
  `attn_fwd_compact_varlen`, `_attn_fwd_common`, `autotune::Autotune_attn_fwd*`
  symbols are exported; no torch include, no torch ABI.
- `readelf -d` → SONAME `libaotriton_v2.so.0.8.0`, NEEDED `libamdhip64.so.6`.
  System ROCm here is `/opt/rocm/.info/version = 7.2.2`; ROCm ships a
  `libamdhip64.so.6 -> .so.7` ABI compat shim, so 0.8 loads on 7.x in
  practice — manifest pin records the build-target ROCm minor explicitly.
- `ls ~/Downloads/aotriton/aotriton/lib/aotriton.images/amd-gfx110x/flash/
  attn_fwd/ | wc -l` → 480 pretuned binaries (Navi3x); 49 MB raw, ≈ 30 MB
  after pruning to causal × bf16/fp16 × head_dim 128.
- `/opt/rocm/lib/`: no `libaotriton_v2*` present — relying on system ROCm is
  not viable, fetch-on-install is required.
- `~/amd-gpu-tuning/reference/composable_kernel/example/ck_tile/01_fmha/
  script/known_fails_*.txt` → only `gfx90a, gfx942, gfx950`. CK FMHA is CDNA;
  not a fallback on W7900.

Multiloop `prefill-perf/run-20260515-154601` is paused/detached pending the
Phase 1 AOTriton spike. No measurements changed; this commit is docs-only.

Files: `docs/PREFILL.md`, `WORKLOG.md`.

## 2026-05-16 — Prefill multiloop iter 57: AOTriton source/runtime scaffold

User flagged that the local `~/Downloads/aotriton` 0.8 dump may be stale and
recommended a recent checkout before wrapping attention. After the peer
`docs/PREFILL.md` production decision landed, aligned the spike with Option 1:
source-reference submodule plus fetched runtime tarball. Added
`third_party/aotriton` as a git submodule pinned to release tag `0.8.2b`
(commit `b24f43a9771622faa157155568b9a200c3b49e41`) and removed the `branch = main`
tracking line from `.gitmodules`. The nested AOTriton Triton submodule is not
initialized; initialize recursively only for an explicit from-source AOTriton
build, not for normal hipEngine wrapper work.

Also added a torch-free discovery scaffold at
`hipengine/kernels/hip_gfx1100/attention/aotriton.py`. It resolves source
headers from `HIPENGINE_AOTRITON_SOURCE_ROOT` or the pinned source submodule,
and separately resolves an extracted runtime from `HIPENGINE_AOTRITON_RUNTIME_ROOT`
or a fetch-on-install cache. It intentionally does **not** fall back to the older
`~/Downloads` runtime unless the developer points `HIPENGINE_AOTRITON_RUNTIME_ROOT`
there. The pinned 0.8.2b source header is the v2 API with
`aotriton::v2::flash::attn_fwd_compact_varlen(...)`, matching the release
tarball ABI. Added `aotriton_release.toml` with the 0.8.2b ROCm 6.3 tarball URL
and SHA256 (`16356dc1813cf3e60da23eb2f29440cb35eedd3a2fbf81e6093a0bc42056ad08`),
plus `scripts/fetch_aotriton.sh` to download, verify, extract, optionally prune,
and write `MANIFEST.local.json` under `~/.cache/hipengine/aotriton/0.8.2b/`.

Validation commands:

```bash
git submodule status --recursive
python3 -m py_compile hipengine/kernels/hip_gfx1100/attention/aotriton.py
python3 -m pytest tests/test_aotriton_discovery.py -q
scripts/fetch_aotriton.sh --dry-run
HIPENGINE_AOTRITON_RUNTIME_ROOT=/home/lhl/Downloads/aotriton/aotriton python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.attention.aotriton import aotriton_runtime_tree, aotriton_source_tree
print('source', aotriton_source_tree())
print('runtime', aotriton_runtime_tree())
PY
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: discovery tests passed (`3 passed`), `fetch_aotriton.sh --dry-run`
printed the expected 0.8.2b install plan, source discovery resolved
`/home/lhl/hipengine/third_party/aotriton/include/aotriton/flash.h`, and runtime
discovery resolved the explicitly provided `/home/lhl/Downloads/aotriton/aotriton`
tree. The default hot path is unchanged: 512/128 exact verify was
`2093.475 tok/s`; fixture gate passed unchanged (`max_kl=0.03406`, top-1 `1.0`,
`native_owned_device_bytes=1625645909`); 4K/128 stayed above guard at
`661.422 tok/s` (`prefill_seconds=6.1927`, decode `102.306 tok/s`). This
iteration is setup/log-only for the AOTriton attention plugin, not a retained
performance optimization.

## 2026-05-16 — docs/PREFILL.md: kernel-source verification + production decision

Folded a peer-agent confirming analysis into the PREFILL.md AOTriton section
after reading the existing kernel source directly to verify every claim.

What I verified from `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip`:

- Line 1083 — LDS `scores` buffer is `max_context_len * 4 B` per block.
  2 KiB at T=512, 16 KiB at T=4K, ~128 KiB at T=32K. RDNA3 CU has ~64 KiB
  LDS, so resident blocks/CU collapse from 8+ at 512 to ≤3 at 4K to
  single-block-per-CU at 32K. Occupancy collapse compounds the T² cost.
- Line 1170 — V@scores epilogue is a fully serial T-deep loop per output dim;
  no LDS V staging, no GQA KV sharing. 4096 sequential FMAs + HBM loads per
  (thread, dim) at T=4K.
- Line 1084 — `kv_head = q_head / kv_group` computed per block; 16 Q-heads /
  2 KV-heads = 8× redundant K/V re-streaming from HBM across the GQA group.
- Lines 1191, 1350, 1410 — kernel epilogue is
  `out[...] = static_cast<half_t>(acc * sigmoid_f32(gate_v))`. The attention
  output is multiplied by `sigmoid(gate)` in-kernel; AOTriton has no gate
  input so a Phase 1 AOTriton wrapper must add a trivial post-pass
  `out *= sigmoid(gate)` elementwise kernel to preserve semantics. Expected
  cost ≤ 0.2 ms at T=4K, head_dim=128, num_q_heads=16.
- The `<true>` / `<false>` template flip toggles `SHORT_BLOCK256` for
  short-context block-table inlining (line 1090–1097); both branches share
  the same inner attention algorithm. The split-K story for `<true>` was
  the wrong mental model; both branches are pre-Flash-Attention.

What I verified on-disk for AOTriton 0.8 (recorded as reference, not as a
decision lever):

- `libaotriton_v2.so.0.8.0` = 28 MB.
- `aotriton.images/amd-gfx110x/flash/attn_fwd/` = 49 MB across 480 forward
  variants.
- Backward + debug subdirs total ~50 MB (drop for inference).
- Inference-only ship: 76 MB.
- Aggressive prune to causal × {bf16,fp16} × head_dim 128 only: 32 binaries,
  3 MB images + 28 MB .so = 31 MB. `ls FONLY__^{bf16,fp16}@16,False,128,*.aks2
  | wc -l` confirmed 32.

Edits to `docs/PREFILL.md`:

- "Why our kernel mis-scales" — replaced the previous summary with a
  source-line-referenced enumeration of the four structural issues
  (LDS-scales-with-T, serial V loop, missing GQA KV sharing, `<true>/<false>`
  is a red herring) and an explicit "preserve gate fusion" callout.
- Phase 1 plan — added a gate-fusion post-pass kernel as an explicit substep
  with the line references for the math source.
- "Concrete version on this host" — replaced the earlier approximate footprint
  numbers with the verified `du`/`ls` figures and added the inference-only
  vs aggressive-prune breakdown.
- "What not to do" — rewrote the "do not vendor binaries" bullet so it stands
  on AGENTS.md git rules + binary-diff review friction; footprint is recorded
  as fact, not as the argument.
- Added a `pytorch/pytorch#166397` (Nov 2025) side note clarifying that
  PyTorch marking gfx1100 SDPA "experimental" is a PyTorch QA policy and not
  a kernel correctness statement; hipengine calls AOTriton directly and is
  unaffected.
- Added a "Production decision (2026-05-16)" subsection: the fetch-on-install
  + pinned-manifest scheme is the production target; the in-flight spike may
  use any pattern (submodule, vendored binary, etc.) but cleanup must
  converge on the manifest + fetcher + stable-ABI wrapper, with graceful
  fallback to the existing hand-rolled kernel when AOTriton is absent.

Coordination context: a parallel spike has staged a git-submodule approach
(`?? .gitmodules`, `?? third_party/aotriton`, `?? hipengine/kernels/hip_gfx1100/
attention/aotriton.py`, `?? tests/test_aotriton_discovery.py`). Their working
state is untouched; this commit unstages only what I did not create. The
"Production decision" subsection makes the convergence target explicit for
the eventual cleanup PR.

Files: `docs/PREFILL.md`, `WORKLOG.md`. Docs-only; no code or measurement
change.

## 2026-05-16 — Marlin-K port analysis doc

Wrote `docs/MARLIN.md` as the hipEngine intake analysis for the parent Marlin-K/qweight-neutral work from `~/amd-gpu-tuning`.

Source/evidence reviewed:

- `python3 scripts/check_lineage.py --file '*paroquant*' --diff stat` from hipEngine: parent `nano-vllm-amd` branch `gfx1100-qwen3.5`, HEAD `1522293`, with Marlin-related drift in `paroquant.py` and `paroquant_kernels.py` since the current hipEngine lineage baseline `22405a9`.
- Parent source anchors:
  - `/home/lhl/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant.py` (`_repack_awq_to_marlin_k_v0`, qweight-neutral buffer/view setup, rows==1 dispatch).
  - `/home/lhl/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py` (`_MARLIN_K_FMA_SRC`, `gemv_paro_marlin_k_fma_kernel`, wrapper shape checks/thread selection).
- Parent docs/worklog:
  - `/home/lhl/amd-gpu-tuning/docs/OPTIMAL.md` latest retained implementation update.
  - `/home/lhl/amd-gpu-tuning/PLAN-PAROQUANT2.md` §11.11 and §12.
  - `/home/lhl/amd-gpu-tuning/WORKLOG.md` entries `2026-05-15 20:10 UTC — Marlin-K qweight-neutral replacement`, `2026-05-15 20:45 UTC — §12 Marlin-K roadmap reconciliation after qweight-neutral work`, and `2026-05-16 05:15 UTC — OPTIMAL.md refreshed for qweight-neutral Marlin-K`.

Main documented conclusion: porting is worth doing now, but first hipEngine port should be narrow/conservative: standalone `paro_marlin_k.{hip,py}` + NumPy repack/oracle tests, rows==1 non-expert GEMV only, no rejected §12 experiments, and runtime promotion only after hipEngine reproduces the parent hybrid memory story (Marlin-K rows==1, zero-copy pack8 view for fused paths, no duplicate large W4 qweight buffer).

Validation: docs/process change only. Re-read `docs/MARLIN.md`; no Python compile step was applicable because no Python files changed. No GPU benchmark rerun; all speed/memory numbers are explicitly cited as parent `~/amd-gpu-tuning` evidence, not new hipEngine measurements.

## 2026-05-16 — PARO Marlin-K host repack helper

Confirmed hipEngine was runnable before touching PARO code:

```bash
python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
python3 -m pytest tests/test_build.py tests/test_smoke_add_plan.py tests/test_paro_awq_gemv_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/smoke.py --mode registry
python3 scripts/smoke.py --mode smoke-add-plan
python3 scripts/smoke.py --mode smoke-add-hip --n 1024
python3 scripts/smoke.py --mode paro-pack8-gemv-hip --rows 2 --hidden-size 16
```

Smoke passed: `44 passed`, registry produced the expected missing-kernel plan diagnostic, smoke-add returned `max_abs=0.0`, and PARO pack8 GEMV returned `single_mismatch=0/0`, `dual_mismatch=0/0`, `max_abs=0.0`, plus FP16 parity `fp16_max_abs=0.0`.

Then made the first scoped PARO edit toward the Marlin-K port: added torch-free host helpers in `hipengine/loading/qwen35_paro.py`:

- `repack_paro_awq_to_marlin_k_host(qweight, qzeros, scales, bits=4, group_size=128)` converts checkpoint/PARO layout `qweight [K,N/8]`, `qzeros [K/128,N/8]`, `scales [K/128,N]` into parent Marlin-K v0 layout `qweight_mk [N/8,K/128,128]`, `qzeros_mk [N/8,K/128]`, `scales_mk [N/8,K/128,8]`.
- `paro_marlin_k_pack8_decode_view(qweight_mk)` returns the zero-copy pack8 decode view over `qweight_mk`, documenting the ownership/aliasing caveat for later device materialization.
- Exported both helpers from `hipengine.loading`.

Added RED/GREEN coverage in `tests/test_qwen35_paro_marlin_k.py`: deterministic layout parity, zero-copy `np.shares_memory` pack8 view, and validation errors for unsupported/mismatched shapes. The initial RED failed on missing exports, then GREEN after implementation.

Validation after the edit:

```bash
python3 -m compileall -q hipengine tests
python3 -m pytest tests/test_qwen35_paro_marlin_k.py tests/test_qwen35_paro_layout.py tests/test_paro_awq_gemv_plan.py tests/test_qwen35_decode_state.py -q
python3 scripts/smoke.py --mode registry
python3 scripts/smoke.py --mode paro-pack8-gemv-hip --rows 2 --hidden-size 16
```

Result: `54 passed`; registry and PARO pack8 smoke stayed green (`max_abs=0.0`, all mismatch counters `0/0`). No GPU Marlin-K performance claim yet: this is host layout plumbing only, preparing for the raw-pointer kernel port described in `docs/MARLIN.md`.

## 2026-05-16 — Prefill multiloop iter 58: AOTriton stable C-ABI wrapper

Implemented the next AOTriton vertical slice without changing the default
prefill dispatch path. Added `hipengine/kernels/hip_gfx1100/attention/
aotriton_wrap.cc` as a torch-free C++ shim around
`aotriton::v2::flash::attn_fwd_compact_varlen(...)`, exposing two stable
`extern "C"` symbols:

- `hipengine_aotriton_check_gpu(void* stream)`
- `hipengine_aotriton_attn_fwd_compact_varlen(...)`

Added `aotriton_wrap.py` with ctypes tensor descriptors, dtype mapping,
`plan_aotriton_wrap_build`, `build_aotriton_wrap`, and a registered but
unselected plugin variant:
`KernelKey("hip_gfx1100", "full_attn_prefill", "w4_paro", "aotriton_attn_fwd")`.
The existing `qwen35_causal_gqa_gate_fp16`/varlen kernels remain the active
runtime path; this iteration only provides the wrapper/oracle path for the next
dispatch spike.

Important build/load finding: compiling the shim with `hipcc` initially pulled
`libamdhip64.so.7` into the wrapper while the 0.8.2b AOTriton library needs
`libamdhip64.so.6`, which caused mixed-ROCm load failures. The final link flags
bracket only `-laotriton_v2` with `--no-as-needed` and restore `--as-needed`
before hipcc's implicit runtime libraries. `readelf -d` now shows the wrapper
NEEDED list contains only `libaotriton_v2.so.0.8.0` plus system C++ libs, not
`libamdhip64.so.7`; the AOTriton runtime supplies the HIP dependency. For the
local smoke I used an explicit ROCm 6.4 compat path in `LD_LIBRARY_PATH` because
this host's `/opt/rocm/lib` has `libamdhip64.so.7` but no `libamdhip64.so.6`.

Validation commands run so far:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/attention/aotriton.py hipengine/kernels/hip_gfx1100/attention/aotriton_wrap.py
python3 -m pytest tests/test_aotriton_discovery.py tests/test_qwen35_paged_attn_decode_plan.py -q
HIPENGINE_AOTRITON_RUNTIME_ROOT=/home/lhl/Downloads/aotriton/aotriton HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import build_aotriton_wrap
artifact = build_aotriton_wrap(load=False)
print(artifact.output_path)
PY
readelf -d /home/lhl/.cache/hipengine/build/aotriton_wrap-6fe002e375d8db33/hipengine_aotriton_wrap.so | grep -E 'NEEDED|RUNPATH'
LD_LIBRARY_PATH=/home/lhl/framework-cluster/lhl/therock/rocm-6.4/lib:${LD_LIBRARY_PATH:-} HIPENGINE_AOTRITON_RUNTIME_ROOT=/home/lhl/Downloads/aotriton/aotriton HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
# allocates a 1x1x1x16 fp32 compact-varlen attention problem and checks output
# equals V for the single causal row
PY
```

Results: unit tests passed (`9 passed`); wrapper build artifact was
`/home/lhl/.cache/hipengine/build/aotriton_wrap-6fe002e375d8db33/
hipengine_aotriton_wrap.so`; `readelf` showed RUNPATH to the explicit AOTriton
lib directory and NEEDED `libaotriton_v2.so.0.8.0`; the GPU smoke returned
`aotriton_wrap_smoke_out8=[0,1,2,3,4,5,6,7]`. The smoke also confirmed that
AOTriton compact-varlen wants `cu_seqlens` as `int32` in practice, matching our
current `CompactPromptSlab` tensors, despite the 0.8 header comment saying
`i64`.

Next step after loop verification: add the runtime dispatch adapter that lays
out Q/K/V/out descriptors for real Qwen3.5 full-attention prefill, allocates
softmax_lse, calls this wrapper for T >= 1024, then applies the existing gate
post-pass semantics.

Loop verify/guard after the wrapper change:

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: 512/128 exact verify `2100.627 tok/s`; fixture gate passed unchanged
(`max_kl=0.0340584589`, top-1 `1.0`, `native_owned_device_bytes=1625645909`);
4K/128 guard remained runnable at `661.341 tok/s` (`prefill_seconds=6.19347`,
decode `102.103 tok/s`). Default hot path remains the hand-rolled prefill
kernel; the AOTriton wrapper is registered only under the explicit
`aotriton_attn_fwd` variant.

## 2026-05-16 — Prefill multiloop iter 59: AOTriton GQA adapter blocked

Started wiring the real Qwen3.5 AOTriton prefill adapter, but stopped before
changing runtime dispatch after a minimal GPU semantic probe showed AOTriton
0.8.2b `attn_fwd_compact_varlen` does **not** handle our GQA shape the way the
planning note assumed.

Probe setup: Q heads = 2, KV heads = 1, T = 1, D = 16, q/k FP32, v/out FP16,
`cu_seqlens=[0,1]` int32, causal. With descriptor shapes
`q=(1,2,1,16)`, `k=(1,1,1,16)`, `v=(1,1,1,16)`, AOTriton returned the correct
V vector for q-head 0 but all zeros for q-head 1. Repeating the same physical
K/V head with a zero head-stride and shape `(1,2,1,16)` also left q-head 1 zero.
The H=1 smoke from iter 58 remains correct, so the wrapper itself is viable;
the missing piece is GQA fanout.

Exact probes run (with the local ROCm 6.4 compat library path because the
standalone AOTriton 0.8.2b library needs `libamdhip64.so.6`):

```bash
LD_LIBRARY_PATH=/home/lhl/framework-cluster/lhl/therock/rocm-6.4/lib:$LD_LIBRARY_PATH \
HIPENGINE_AOTRITON_RUNTIME_ROOT=/home/lhl/Downloads/aotriton/aotriton \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
python3 - <<'PY'
# q=(1,2,1,16), k/v=(1,1,1,16) -> q-head 0 OK, q-head 1 zero
PY

LD_LIBRARY_PATH=/home/lhl/framework-cluster/lhl/therock/rocm-6.4/lib:$LD_LIBRARY_PATH \
HIPENGINE_AOTRITON_RUNTIME_ROOT=/home/lhl/Downloads/aotriton/aotriton \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
python3 - <<'PY'
# q=(1,2,1,16), k/v=(1,2,1,16) with stride_head=0 -> q-head 0 OK, q-head 1 zero
PY
```

Observed output for both GQA probes began:
`[0,1,2,3,4,5,6,7,...,15,0,0,0,0,...]`. Conclusion: a direct single-call
Qwen3.5 adapter cannot be retained yet. Viable next options are:

1. Call AOTriton per q-head (16 launches/layer) using H=1 descriptors. This
   avoids K/V expansion and should be measured; launch overhead may still be
   tolerable at T=4K.
2. Expand K/V to Q-head count in a small HIP fanout kernel, then call AOTriton
   once with Hq==Hkv==16. This trades memory/bandwidth for fewer launches.
3. Check whether a newer AOTriton API/build has explicit GQA support before
   carrying either workaround forward.

No runtime code was changed in this iteration; default prefill remains on the
hand-rolled kernel.

Loop verify/guard after the blocked adapter probe:

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: 512/128 `2084.676 tok/s` (no code-path change; variance below the
iter-58 retained sample), fixture gate passed unchanged (`max_kl=0.0340584589`,
top-1 `1.0`, `native_owned_device_bytes=1625645909`), and 4K/128 remained
runnable at `661.442 tok/s` (`prefill_seconds=6.19253`, decode `102.300 tok/s`).
Iteration outcome: blocked/log-only; do not route Qwen3.5 to AOTriton in a
single compact-varlen call until the GQA fanout strategy is chosen and proven.

## 2026-05-16 — Prefill multiloop iter 60: AOTriton per-Q-head GQA wrapper

Implemented the first GQA workaround identified in iter 59: a wrapper-level
`hipengine_aotriton_attn_fwd_compact_varlen_gqa_per_q_head(...)` entry that
slices the Q/K/V/LSE/output descriptors and issues one H=1 AOTriton
`attn_fwd_compact_varlen` call per Q head. This preserves Qwen3.5-style GQA
semantics without expanding K/V to Q-head count. The entry is exposed in
`aotriton_wrap.py` and registered under the explicit, still-unselected variant
`KernelKey("hip_gfx1100", "full_attn_prefill", "w4_paro", "aotriton_attn_fwd_gqa_per_q_head")`.
Default runtime dispatch remains unchanged.

Validation commands:

```bash
git diff --check
python3 -m py_compile hipengine/kernels/hip_gfx1100/attention/aotriton_wrap.py
python3 -m pytest tests/test_aotriton_discovery.py tests/test_qwen35_paged_attn_decode_plan.py -q
HIPENGINE_AOTRITON_RUNTIME_ROOT=/home/lhl/Downloads/aotriton/aotriton HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import build_aotriton_wrap
artifact = build_aotriton_wrap(load=False)
print(artifact.output_path)
PY
readelf -d /home/lhl/.cache/hipengine/build/aotriton_wrap-731a06998fc8c232/hipengine_aotriton_wrap.so | grep -E 'NEEDED|RUNPATH'
LD_LIBRARY_PATH=/home/lhl/framework-cluster/lhl/therock/rocm-6.4/lib:$LD_LIBRARY_PATH HIPENGINE_AOTRITON_RUNTIME_ROOT=/home/lhl/Downloads/aotriton/aotriton HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 - <<'PY'
# q=(1,2,1,16), k/v=(1,1,1,16), fp16 out; call gqa_per_q_head wrapper
PY
```

Results: tests passed (`9 passed`). Wrapper built at
`/home/lhl/.cache/hipengine/build/aotriton_wrap-731a06998fc8c232/
hipengine_aotriton_wrap.so`; `readelf` still shows only RUNPATH to the explicit
AOTriton lib dir and NEEDED `libaotriton_v2.so.0.8.0` (no direct
`libamdhip64.so.7`). The GPU GQA smoke returned both Q heads correctly:
`[0,1,2,...,15,0,1,2,...,15]`, fixing the iter-59 direct-call zero-head
failure. Next runtime iteration can wire this opt-in variant for real Qwen3.5
prefill and measure whether 16 launches/layer are tolerable at T=4K.

Loop verify/guard after the per-Q-head wrapper change:

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-512-128-rerun.json
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json /tmp/multiloop-fixture-gate.json >/tmp/multiloop-fixture-gate.stdout
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/multiloop-prefill-4k-128.json
```

Results: 512/128 default-path samples were `2068.065` and `2055.817 tok/s`.
The per-Q-head wrapper remains unselected by default, so these lower samples are
recorded as default-path variance rather than an active-path regression.
Fixture gate passed unchanged (`max_kl=0.0340584589`, top-1 `1.0`,
`native_owned_device_bytes=1625645909`). 4K/128 remained runnable at
`661.201 tok/s` (`prefill_seconds=6.19479`, decode `101.998 tok/s`).

## 2026-05-16 — AOTriton cleanup: pin to 0.11.2b, drop submodule, fix discovery

Foundational cleanup before the next prefill-perf multiloop iteration.  The
previous iterations (58/59/60) committed the AOTriton wrapper and a
submodule simultaneously and used `HIPENGINE_AOTRITON_RUNTIME_ROOT=
/home/lhl/Downloads/aotriton/aotriton` + `LD_LIBRARY_PATH=
/home/lhl/framework-cluster/lhl/therock/rocm-6.4/lib` to load a hand-staged
0.8.2b binary on this ROCm-7.2.2 host.  That worked but was non-reproducible
on a fresh checkout; `docs/PREFILL.md` "Production decision" prescribed a
fetch-on-install + pinned-manifest scheme.  This commit lands that scheme
and bumps the pin to the latest stable AOTriton release.

Version bump rationale (recorded so the next agent does not re-derive it):

- 0.11.2b ships `aotriton-0.11.2b-manylinux_2_28_x86_64-rocm7.0-shared.tar.gz`
  (4.9 MB runtime tarball) plus a separate `aotriton-0.11.2b-images-amd-
  gfx11xx.tar.gz` (475 MB compressed kernel images).  The runtime tarball is
  the only ROCm 7.0 build in the current release matrix.  Both tarballs share
  the same top-level `aotriton/` directory and merge cleanly into one cache
  tree.
- `readelf -d libaotriton_v2.so.0.11.2` shows NEEDED `libamdhip64.so.7`
  directly.  The host has `/opt/rocm/lib/libamdhip64.so.7` (ROCm 7.2.2).  The
  iter-58 `libamdhip64.so.6` workaround is gone.
- The 0.11b release notes warned that gfx1100 is "experimental" with
  "massive accuracy problems".  Verified that warning is **training-only**:
  `test/adiffs/gfx1100.txt` in 0.11.2b lists 436 failing tests, 100% of
  which are in `test_backward.py` (`awk -F'::' '{print $1}' | sort -u`
  returns only `test/test_backward.py`).  Zero forward-pass failures are
  listed.  hipEngine is inference-only.  0.11.1b restored Navi31 support and
  0.11.2b updated the gfx11xx image tarball; gfx11xx images have 60k+
  downloads on GitHub.
- The V2 API (`attn_fwd_compact_varlen`) is still present in 0.11.x, but the
  signature shifted: in 0.8.x, the `bias` T4 came after `cu_seqlens_k`; in
  0.11.x it moved between `v` and `cu_seqlens_q`, and a new `atomic_for_causal`
  T0 parameter was added before the stream.  `aotriton_wrap.cc` updated to
  match.  Verified by reading
  `https://raw.githubusercontent.com/ROCm/aotriton/0.11.2b/include/aotriton/v2/flash.h`.
- The new `atomic_for_causal` parameter is the "Persistent Dynamic for Causal"
  dispatch atomic introduced in AOTriton 0.9b.  For `is_causal=true` it must
  point at a zero-initialized 1-element int32 device buffer; null returns
  `hipErrorInvalidValue`.  The shim allocates+zeros+frees per call (~5 us
  overhead, negligible against the attention launch itself).
- Non-targets:
  - 0.11.210b is a gfx942-only ASAN debug build.
  - 0.11.52b is a gfx1250-only tech preview.
  - 0.10b has a rocm7.0 build available but offers no advantage over 0.11.2b
    for our forward-only use case.
  - 0.8.x requires `libamdhip64.so.6` and is now unsupported on this host.

Changes (one logical commit unit):

- `hipengine/kernels/hip_gfx1100/attention/aotriton_release.toml`: bump to
  0.11.2b, two `[[aotriton.archives]]` entries (runtime + gfx11xx images),
  recorded sha256s, set so_name to `libaotriton_v2.so.0.11.2`, rocm_min=7.0.
- `scripts/fetch_aotriton.sh`: iterate manifest archives, download both
  tarballs, verify SHA256 against the manifest, extract merged into
  `~/.cache/hipengine/aotriton/<version>/`, prune to `flash/attn_fwd`
  (159 MB pruned vs ~700 MB unpruned), write `MANIFEST.local.json`.
- `hipengine/kernels/hip_gfx1100/attention/aotriton.py`: rewrite discovery
  to the PREFILL-spec'd lookup chain:
  1. `HIPENGINE_AOTRITON_LIB` (developer override)
  2. `${HIPENGINE_AOTRITON_HOME:-~/.cache/hipengine/aotriton}/<version>/`
  3. `/opt/rocm/lib/libaotriton_v2.so` (SONAME-gated)
  4. `AotritonNotInstalledError` pointing at `scripts/fetch_aotriton.sh`
  Drop `HIPENGINE_AOTRITON_SOURCE_ROOT` / `HIPENGINE_AOTRITON_RUNTIME_ROOT`
  env vars and the `aotriton_source_tree()` helper.  Reads `<version>`
  from the manifest.
- `hipengine/kernels/hip_gfx1100/attention/aotriton_wrap.{cc,py}`: update
  the V2 call signature for 0.11.x (bias position + atomic_for_causal),
  rename `runtime_root` kwarg to `home_root` to match the new env var name.
- Generic AOTriton gate post-pass: `hipengine_aotriton_gate_mul_fp16_inplace`
  is a small HIP kernel that applies `out *= sigmoid(gate)` over an FP16
  buffer.  AOTriton's `attn_fwd*` API has no gate input; any caller using
  AOTriton attention needs this post-pass to preserve Qwen3.5-style gate
  semantics.  Lives in `aotriton_wrap.cc` next to the AOTriton extern "C"
  surface, exposed via `aotriton_gate_mul_fp16_inplace` in `aotriton_wrap.py`,
  exported from `__init__.py`.  No caller wires it yet; that lands with the
  runtime wiring.
- `tests/test_aotriton_discovery.py`: replace submodule-fallback tests with
  HIPENGINE_AOTRITON_HOME + HIPENGINE_AOTRITON_LIB tests, add
  AotritonNotInstalledError assertion, add manifest-pin assertion.
- `.gitmodules`, `third_party/aotriton`: removed.
- `.gitignore`: add `third_party/`.
- `docs/PREFILL.md`: replace "AOTriton distribution and pinning strategy"
  section to reflect 0.11.2b reality (the V2 ABI shift, the two-tarball
  layout, the new discovery chain, the gfx1100 "training-only" caveat,
  removal of all submodule language).  Drop the libamdhip64.so.6 / ROCm
  6→7 compat-shim discussion (gone with rocm7.0-shared build).  Add a
  clearly-scoped FUTURE-WORK stub for per-GPU kernel streaming/caching
  (do NOT implement: just write down the idea so a future iteration can
  pick it up).

Validation:

```bash
# Manifest + fetcher dry run
bash scripts/fetch_aotriton.sh --dry-run

# Install pinned 0.11.2b into the cache
scripts/fetch_aotriton.sh
# Output: Installed AOTriton 0.11.2b at /home/lhl/.cache/hipengine/aotriton/0.11.2b
#         (159 MB after default prune to flash/attn_fwd)

# Wrapper builds, NEEDED list verified
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
  python3 -c 'from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import build_aotriton_wrap; print(build_aotriton_wrap(load=False).output_path)'
readelf -d ~/.cache/hipengine/build/aotriton_wrap-*/hipengine_aotriton_wrap.so | grep -E 'NEEDED|RUNPATH'
# NEEDED libaotriton_v2.so.0.11.2
# NEEDED libamdhip64.so.7   <-- present on host, no LD_LIBRARY_PATH workaround
# RUNPATH /home/lhl/.cache/hipengine/aotriton/0.11.2b/lib

# GPU smokes (all PASS)
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 <<'PY'
# H=1 causal (out == v):           PASS
# GQA per-Q-head, H_q=2, H_kv=1:   PASS (both heads match v)
# T=4 causal multi-row:            PASS (row 0 = v[0]; later rows are causal mix)
PY

# Unit tests
python3 -m pytest tests/test_aotriton_discovery.py \
  tests/test_qwen35_paged_attn_decode_plan.py \
  tests/test_qwen35_resident_batch_layout.py -q
# 36 passed
```

Pre-existing test failures (`test_hip_runtime.py`, `test_llm_generate.py`)
are unrelated to AOTriton; verified by `git stash` + re-run on `main`.

Next:

- Land the in-flight AOTriton runtime wiring (`hipengine/runtime/prefill.py`,
  `qwen35_paro.py`, `qwen35_paro_runner.py`, `scripts/qwen35_paro_bench.py`,
  `tests/test_qwen35_resident_batch_layout.py`) as a separate commit.  Those
  files are intentionally left modified in the working tree.
- Re-run the prefill-perf multiloop on the cleaned foundation.  Target: 4K
  attention via the AOTriton per-Q-head variant, threshold-tuned to beat the
  hand-rolled kernel.

## 2026-05-16 — Post-rebase review smoke: AOTriton runtime wiring and Marlin-K loader

Reviewed the post-pull state after the AOTriton 0.11.2b cleanup and Marlin-K
host-repack commits.  The committed Marlin-K work is loader/host-layout only
(`docs/MARLIN.md`, `hipengine/loading/qwen35_paro.py`,
`tests/test_qwen35_paro_marlin_k.py`); it does not yet wire a Marlin-K decode
kernel, so a decode throughput gain should not be expected from the current
runtime path.

Targeted validation stayed green after the rebase plus local review fixes:

```bash
python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
rocminfo | grep -E 'Name:|gfx' | head -40
python3 -m pytest tests/test_qwen35_paro_marlin_k.py tests/test_aotriton_discovery.py \
  tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_resident_batch_layout.py -q
# 39 passed
```

During the opt-in AOTriton runtime smoke, the first attempt hit a missing
`_check_positive` helper, and the second attempt faulted the GPU when passing
FP32 Q/K scratch descriptors into AOTriton.  For measurement only, added the
missing positive validator and cast FP32 Q/K scratch to FP16 before the
AOTriton call.  Also added an `--attn-aotriton-min-tokens` knob to the native
prefill fixture gate so this route can be tested explicitly.

Smoke commands/results on W7900/gfx1100, Qwen3.5-35B-A3B-PARO w4_paro,
`--max-layers 40`, repeated token id 9707, no graph replay:

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/hipengine-review-default-512-128.json
# default path: prefill 2123.798 tok/s, decode 101.811 tok/s

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/hipengine-review-default-4k-128.json
# default path: prefill 662.401 tok/s, decode 102.389 tok/s

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-review-aotriton-512-128.json
# AOTriton at 512: prefill 1866.301 tok/s, decode 101.869 tok/s (slower than default)

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 4096 \
  --json /tmp/hipengine-review-th4096-4k-128.json
# AOTriton at 4K: prefill 2203.038 tok/s, decode 102.007 tok/s
```

Correctness gates:

```bash
python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 \
  --json /tmp/hipengine-review-fixture-gate.json
# default path passed: max_kl=0.0340584589, top1=1.0, generated_match=true

python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 \
  --attn-aotriton-min-tokens 512 --json /tmp/hipengine-review-aotriton-fixture-gate.json
# AOTriton path FAILED: max_kl=8.8731804708, top1=0.484848, generated_match=false
```

Conclusion: the rebased default path is healthy and roughly in the expected
range (512/128 improved vs the latest retained diagnostic sample by variance;
4K/128 remains ~662 tok/s without AOTriton).  The opt-in AOTriton path is
promising for long prefill speed (4K/128 ~2203 tok/s) but is **not retainable**
currently because the 512 fixture correctness gate fails badly after the FP16
Q/K cast.  Multiloop iter 61 was recorded as `revert`/blocked, and the loop was
paused.  Do not promote or commit AOTriton runtime dispatch until the Q/K dtype
or descriptor semantics are corrected and the fixture gate passes.

## 2026-05-16 — Task #3 rerun: Qwen3.5/PARO 512/128 and 4K/128 default smokes

Reran the requested default-path Qwen3.5-35B-A3B-PARO smoke commands after the
targeted AOTriton/Marlin validation.  AOTriton dispatch stayed disabled
(`attn_aotriton_min_tokens=0`) for both runs.

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/task3-qwen35-paro-512-128.json
# prefill_tok_s=2059.8899402533602, warmed_decode_tok_s=101.33408660385875
# path=single_request_native_full, aotriton_attention=false, generated preview token ids all 9707

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/task3-qwen35-paro-4k-128.json
# prefill_tok_s=662.2098953663426, warmed_decode_tok_s=102.37353841024634
# path=single_request_native_full, aotriton_attention=false, generated preview token ids all 9707
```

Comparison to `benchmarks/README.md` retained diagnostic row
(512 prefill median 2077.262 tok/s, decode ~101.2 tok/s; 4K prefill median
659.950 tok/s, decode 102.146 tok/s): 512 prefill is ~0.8% below the retained
median (within recent variance), 4K prefill is ~0.3% above, and decode is
essentially unchanged.  Default smoke health is good; this does not validate the
opt-in AOTriton route, which remains blocked by the separate fixture-gate
failure recorded above.

## 2026-05-16 — Task #5 AOTriton mismatch localized

Localized the opt-in AOTriton prefill mismatch before attempting the BF16 fix.
The short version: the descriptors/GQA fanout are not the main problem.  The
current runtime wiring violates AOTriton's same-dtype contract when trying to
mirror native FP32-query/BF16-cache semantics, and the runnable all-FP16
workaround diverges from the BF16 KV cache representation used by both the
native HIP path and the parent torch SDPA path.

Reference checked in parent nano-vllm-amd:

```python
# /home/lhl/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/mtp.py:407-415
q = query.transpose(0, 1).unsqueeze(0).to(torch.bfloat16)
k = key.transpose(0, 1).unsqueeze(0).to(torch.bfloat16)
v = value.transpose(0, 1).unsqueeze(0).to(torch.bfloat16)
attn = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                      scale=float(head_dim) ** -0.5,
                                      enable_gqa=True)
gated = attn.float() * torch.sigmoid(gate.float())
# cache stores BF16 key/value
```

Localization evidence:

- Synthetic AOTriton GQA compact-varlen with **uniform FP16** Q/K/V at real
  shape family (`num_q_heads=16`, `num_kv_heads=2`, `head_dim=256`) matches a
  host softmax reference at rows up to 512 (`max_abs=0.000244`, no GPU fault).
  This supports the GQA slice/stride math.
- Synthetic mixed dtype (`Q=FP32`, `K/V=FP16` or `BF16`) can run at small rows
  but faults at larger rows/head_dim (`Memory access fault ... Page not
  present`) around rows 128/512.  This points to an AOTriton image dispatch
  same-dtype requirement rather than a row/stride bug.
- AOTriton all-FP16 on the full Qwen3.5 fixture is close for the prefill seed
  only but diverges during decode:

```bash
python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 \
  --max-new-tokens 1 --attn-aotriton-min-tokens 512 \
  --json /tmp/aot-qkvfp16-l40.json
# passed=true, max_kl=0.0001395008, top1=1.0, seed token 4403 matches

python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 \
  --attn-aotriton-min-tokens 512 --json /tmp/aot-qkvfp16-fixture32.json
# passed=false, generated_match=false, max_kl=8.8731804708, top1=0.484848
```

- Restricting AOTriton to only the final full-attention layer kept the 32-token
  fixture passing (`/tmp/aot-lastonly-fixture32.json`: `max_kl=0.0340584589`,
  top1 `1.0`, generated IDs match).  This confirms the long-run failure is
  cache/hidden-state drift accumulating across layers, not an immediate causal
  mask or gate-shape failure.

Conclusion / next fix target: switch the runtime AOTriton path to the parent
semantics: cast Q/K/V to BF16, call AOTriton with uniform BF16 descriptors and a
BF16 output scratch, then apply the Qwen gate in FP32 semantics while writing
FP16 output (`gate_mul_bf16_to_fp16` or equivalent).  Do not use mixed FP32
Q with BF16/FP16 K/V; it is undefined for AOTriton's prebuilt images and caused
the page faults.  Hoisting the per-head `atomic_for_causal` allocation is a
later perf cleanup after correctness.

## 2026-05-16 — Task #7 validation: BF16-native fallback and AOTriton match

After the AOTriton BF16 fix, aligned the native HIP prefill fallback with the
same Qwen3.5/PARO prefill semantics used by parent nano-vllm-amd SDPA:
BF16-rounded Q, BF16 K/V cache, BF16-rounded attention output, then FP32
sigmoid gate to FP16 output.  The native HIP prefill kernels now round loaded Q
through BF16 and round the pre-gate attention accumulator through BF16 before
applying the FP16 gate.  CPU reference prefill/varlen oracles were updated to
mirror that BF16 pre-gate contract.

Kernel/oracle smokes after the native fallback alignment:

```bash
python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt
# rows=3 ... prefill_gate_fp16_max_abs=0 prefill_gate_fp16_mismatch=0

python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-varlen-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt
# rows=4 ... varlen_prefill_gate_fp16_max_abs=0 varlen_prefill_gate_fp16_mismatch=0
```

Targeted unit tests:

```bash
python3 -m pytest tests/test_qwen35_paro_marlin_k.py tests/test_aotriton_discovery.py \
  tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_resident_batch_layout.py \
  tests/test_qwen35_decode_state.py -q
# 73 passed
```

Fixture gates against the serial resident path:

```bash
python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 \
  --json /tmp/task7-bf16native-fixture-default.json
# default native: passed=true, generated_match=true, expected_match=true,
# max_kl=0.0452046868, top1=1.0

python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 \
  --attn-aotriton-min-tokens 512 --json /tmp/task7-bf16native-fixture-aotriton.json
# AOTriton: passed=true, generated_match=true, expected_match=true,
# max_kl=0.0395688706, top1=1.0
```

Direct default-native vs AOTriton comparison on the same 512/32 fixture:

```bash
python3 - <<'PY' >/tmp/task7-default-vs-aotriton.json
# Runs _run_once(... prefill_mode='native') once with default prefill_config and
# once with PrefillConfig(attn_aotriton_min_tokens=512), then compares full
# lm-head logits with _compare_logits.
PY
# generated_match=true, max_kl=0.0172678504, top1=1.0
```

Smoke performance, repeated token id 9707, max_layers=40, no graph replay:

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/task7-bf16native-default-512-128.json
# default 512/128: prefill=2045.535 tok/s, decode=101.246 tok/s

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/task7-bf16native-default-4k-128.json
# default 4K/128: prefill=662.630 tok/s, decode=102.163 tok/s

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --json /tmp/task7-bf16native-aotriton-512-128.json
# AOTriton 512/128: prefill=1836.089 tok/s, decode=101.864 tok/s

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --json /tmp/task7-bf16native-aotriton-4k-128.json
# AOTriton 4K/128: prefill=2185.173 tok/s, decode=102.296 tok/s
```

Conclusion: native HIP fallback and AOTriton now share the parent-style BF16
Q/K/V + BF16 pre-gate attention semantics and match each other at the fixture
level (direct max KL 0.0173, top1 100%, generated IDs identical).  AOTriton is
still slower at 512 due to per-Q-head launches/casts, but it is correctness-clean
and ~3.3x faster than native fallback at 4K prefill.

## 2026-05-16 — Task #8 AOTriton debug wrap-up / readiness

AOTriton correctness blocker is resolved for the explicit opt-in path.  The
substantive fix was to match parent nano-vllm-amd SDPA prefill semantics end to
end: uniform BF16 Q/K/V/Out for attention, then FP32 sigmoid gate semantics to
FP16 output.  The native HIP fallback was aligned to the same BF16 pre-gate
contract so fallback, AOTriton, CPU reference, and parent SDPA are testing the
same math.

Files changed in this logical unit:

- `hipengine/kernels/hip_gfx1100/attention/aotriton_wrap.{cc,py}` and
  `attention/__init__.py`: add `hipengine_aotriton_gate_mul_bf16_to_fp16` C/Python
  wrapper for BF16 attention output + FP16 gate -> FP16 gated output.
- `hipengine/runtime/qwen35_paro.py`, `qwen35_paro_runner.py`,
  `runtime/prefill.py`, `scripts/qwen35_paro_bench.py`,
  `tests/test_qwen35_resident_batch_layout.py`: opt-in runtime dispatch via
  `PrefillConfig.attn_aotriton_min_tokens`, disabled by default.
- `scripts/qwen35_native_prefill_fixture_gate.py`: fixture-gate knob for
  exercising AOTriton prefill.
- `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip`: native
  fallback now rounds Q and pre-gate attention output through BF16 in prefill
  kernels, matching parent SDPA/AOTriton semantics.
- `hipengine/kernels/cpu_reference/ops.py`: CPU prefill references mirror the
  BF16 Q and BF16 pre-gate attention contract.

Final validation evidence is in the previous Task #7 entry.  Compact summary:

```bash
python3 -m pytest tests/test_qwen35_paro_marlin_k.py tests/test_aotriton_discovery.py \
  tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_resident_batch_layout.py \
  tests/test_qwen35_decode_state.py -q
# 73 passed

python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 \
  --json /tmp/task7-bf16native-fixture-default.json
# default native passed; max_kl=0.0452046868, top1=1.0

python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 \
  --attn-aotriton-min-tokens 512 --json /tmp/task7-bf16native-fixture-aotriton.json
# AOTriton passed; max_kl=0.0395688706, top1=1.0

# Direct default-native vs AOTriton on the same 512/32 fixture:
# generated_match=true, max_kl=0.0172678504, top1=1.0
```

Performance summary from Task #7 smokes (W7900/gfx1100, Qwen3.5-35B-A3B-PARO
w4_paro, max_layers=40, token id 9707):

- Default 512/128: `2045.535 tok/s` prefill, `101.246 tok/s` decode.
- Default 4K/128: `662.630 tok/s` prefill, `102.163 tok/s` decode.
- AOTriton 512/128: `1836.089 tok/s` prefill, `101.864 tok/s` decode.
- AOTriton 4K/128: `2185.173 tok/s` prefill, `102.296 tok/s` decode.

Readiness / blockers:

- Correctness: ready for commit.  Default and AOTriton fixture gates pass, and
  direct default-vs-AOTriton comparison passes the KL/top-1 contract.
- Perf: AOTriton should remain threshold-gated; it is slower at 512 due to
  per-Q-head launches plus cast/gate passes, but 4K is ~3.3x faster.  A sensible
  default threshold is not chosen yet; keep the default disabled (`0`) until a
  threshold sweep lands.
- Follow-up perf cleanup: hoist/reuse the per-head `atomic_for_causal` buffer in
  `aotriton_wrap.cc` instead of allocating/freeing inside each head call, and
  evaluate a K/V fanout or native GQA-capable AOTriton API if available.

## 2026-05-16 — GGUF intake analysis doc

Wrote `docs/GGUF.md` as the first planning/intake document for GGUF support in hipEngine.

Source/evidence reviewed:

- hipEngine state was clean before the docs change.
- Local llama.cpp GGUF references:
  - `/home/lhl/llama.cpp/llama.cpp-hip-therock/ggml/include/gguf.h` for GGUF file structure, magic/version/alignment, KV table, tensor info, and tensor offsets.
  - `/home/lhl/llama.cpp/llama.cpp-hip-therock/gguf-py/gguf/gguf_reader.py` for Python reader behavior.
  - `/home/lhl/llama.cpp/llama.cpp-hip-therock/gguf-py/gguf/constants.py` for `GGMLQuantizationType` and `GGML_QUANT_SIZES` (`Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`, IQ types, etc.).
  - `/home/lhl/llama.cpp/llama.cpp-hip-therock/gguf-py/gguf/quants.py` and `ggml/src/ggml-quants.c` for CPU quant/dequant references.
  - `/home/lhl/llama.cpp/llama.cpp-hip-therock/ggml/src/ggml-common.h` for GGML block structs and static sizes.
- Parent `~/amd-gpu-tuning` docs that explain how GGUF/Q4_K informed the PARO Marlin-K work: `PLAN-PAROQUANT2.md`, `PLAN-LONGCONTEXT.md`, `PR_COMMENT-llamacpp-hip-unroll600.md`, and hipEngine's new `docs/MARLIN.md`.

Main conclusions documented:

- GGUF scanner/metadata intake is easy and should be first.
- FP16 fallback load is straightforward and useful for correctness/model plumbing, but not a performance/memory path.
- Native GGUF quant execution is not drop-in from PARO Marlin-K: GGUF tensors are GGML block tensors (`Q4_0`, `Q8_0`, `Q4_K`, etc.) with embedded scale/min metadata and different quant math.
- Recommended implementation order: scanner -> `inspect_gguf.py` tensor census -> quant-layout table/oracles -> Qwen dense name-map smoke -> FP16 fallback -> native `Q8_0` or `Q4_K` GEMV.
- Keep GGUF as a first-class loader/quant family and preserve hipEngine invariants: torch-free runtime, plugin registry keys, raw-pointer kernels, CPU-reference correctness gates, and benchmark artifact policy.

Validation: docs-only change. Re-read `docs/GGUF.md` and ran `git diff --check -- docs/GGUF.md WORKLOG.md`. No GPU run; no new performance claim.

## 2026-05-16 — V2 direct-GQA AOTriton shortcut replaces per-Q-head fanout

Before migrating to the AOTriton V3 params ABI, tested the suggested V2 shortcut:
call `v2::flash::attn_fwd_compact_varlen` once with true GQA-shaped tensors
instead of slicing to one Q head at a time.  The harness allocated BF16 compact
varlen tensors with Q shape `(1, 16, rows, 256)`, K/V shape `(1, 2, rows, 256)`,
`cu_seqlens=[0, rows]`, and compared the direct V2 call against the existing
`aotriton_attn_fwd_compact_varlen_gqa_per_q_head` wrapper output.

```bash
python3 - <<'PY'
# one-off harness: random BF16 Q/K/V, rows in (64, 512), call
# aotriton_attn_fwd_compact_varlen(...) directly with Q heads=16, K/V heads=2,
# then call aotriton_attn_fwd_compact_varlen_gqa_per_q_head(...) on the same
# buffers and compare BF16 output bits.
PY
# {'rows': 64, 'bits_equal': True, 'max_abs': 0.0, 'mismatch': 0,
#  'nonzero_direct': 262144, 'nonzero_fan': 262144}
# {'rows': 512, 'bits_equal': True, 'max_abs': 0.0, 'mismatch': 0,
#  'nonzero_direct': 2097152, 'nonzero_fan': 2097152}
```

Result: the installed 0.11.2b V2 binary already handles GQA-shaped BF16
compact-varlen inputs, even though the V2 header documents a single `num_heads`.
Made the smallest runtime change: the opt-in Qwen3.5/PARO AOTriton prefill path
now calls `aotriton_attn_fwd_compact_varlen` directly with Q heads=16 and K/V
heads=2, preserving the committed BF16 Q/K/V/Out + BF16-to-FP16 gate semantics.
The per-Q-head shim stays available for tests/debugging but is no longer on the
runtime prefill path.

Validation:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro.py
# ok

python3 -m pytest tests/test_qwen35_paro_marlin_k.py tests/test_aotriton_discovery.py \
  tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_resident_batch_layout.py \
  tests/test_qwen35_decode_state.py -q
# passed (73 targeted tests, exit 0)

python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 \
  --attn-aotriton-min-tokens 512 --json /tmp/task11-v2-direct-gqa-fixture-aotriton.json
# passed=true, generated_match=true, expected_match=true,
# max_kl=0.0395688706, top1=1.0
```

Smoke performance on W7900/gfx1100, Qwen3.5-35B-A3B-PARO w4_paro, max_layers=40,
repeated token id 9707, no graph replay, AOTriton opt-in threshold 512:

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --json /tmp/task11-v2-direct-gqa-aotriton-512-128.json
# prefill=2164.732 tok/s, decode=101.604 tok/s
# previous BF16 per-Q-head AOTriton 512/128 smoke: 1836.089 tok/s
# previous BF16 native fallback 512/128 smoke: 2045.535 tok/s

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --json /tmp/task11-v2-direct-gqa-aotriton-4k-128.json
# prefill=2374.341 tok/s, decode=102.127 tok/s
# previous BF16 per-Q-head AOTriton 4K/128 smoke: 2185.173 tok/s
# previous BF16 native fallback 4K/128 smoke: 662.630 tok/s
```

Conclusion: V3 is not required to remove the 16x per-Q-head launch fanout on the
current pinned binary.  The V2 direct-GQA shortcut is correctness-clean and flips
512-token AOTriton from slower than native to faster in this smoke.  Remaining
follow-ups are now threshold sweep and optional V3/atomic-counter cleanup; keep
default dispatch disabled (`attn_aotriton_min_tokens=0`) until the threshold
sweep lands.

## 2026-05-16 — AOTriton V3 params ABI for opt-in Qwen3.5/PARO prefill

Moved the opt-in AOTriton prefill path from the V2 positional
`attn_fwd_compact_varlen` ABI to the top-level V3 `attn_fwd_params` ABI while
preserving the already-validated BF16 semantics: FP32 Q/K scratch -> BF16, FP16
V scratch -> BF16, V3 attention writes BF16, then BF16 attention output is gated
with FP32 sigmoid semantics into FP16.  Default runtime dispatch remains disabled
unless `PrefillConfig.attn_aotriton_min_tokens > 0`.

V3 binding details:

- Added `hipengine_aotriton_attn_fwd_v3_compact_varlen` in
  `aotriton_wrap.cc` and `aotriton_attn_fwd_v3_compact_varlen` in Python.
- `attn_fwd_params` uses Q shape `(1, 16, rows, 256)` and K/V shape
  `(1, 2, rows, 256)` so GQA is inferred from `Q.size(1)` vs `K.size(1)`.
- `cu_seqlens_q/k=[0, rows]`, `Max_seqlen_q/k=rows`,
  `varlen_type=CompactVarlen`, `causal_type=WindowedAttention`, and
  `window_left=window_right=WindowValue::TopLeftAligned`, matching AOTriton's
  own V2 compatibility implementation.
- Replaced per-call V2 `hipMalloc/hipFree` atomic behavior with a reusable
  workspace tensor `attn.aotriton_atomic` (`int32[1]`).  The C shim resets it via
  `hipMemsetAsync(..., 0, 4)` before each causal V3 call.

Header/source verification:

```bash
# Installed header: ~/.cache/hipengine/aotriton/0.11.2b/include/aotriton/flash.h
# V3 exposes attn_fwd_params + attn_fwd(params, kVersion, stream, options).
# Parent source checked in ~/amd-gpu-tuning/reference/aotriton/v3src/flash/attn_fwd.cc:
# V2 compatibility path lowers causal=True to CausalType::WindowedAttention with
# WindowValue::TopLeftAligned for both window_left/right, and V3 infers
# Num_head_q / Num_head_k from Q/K shapes.
```

One-off V2 direct-GQA vs V3 params equivalence harness:

```bash
python3 - <<'PY'
# random BF16 compact-varlen GQA tensors, Q=(1,16,rows,256), K/V=(1,2,rows,256)
# compare aotriton_attn_fwd_compact_varlen(...) against
# aotriton_attn_fwd_v3_compact_varlen(...), then run V3 again with the same
# persistent atomic pointer to verify reset/reuse behavior.
PY
# {'rows': 64, 'v2_v3_bits_equal': True, 'v3_repeat_bits_equal': True,
#  'max_abs': 0.0, 'mismatch': 0, 'atomic_after': 0}
# {'rows': 512, 'v2_v3_bits_equal': True, 'v3_repeat_bits_equal': True,
#  'max_abs': 0.0, 'mismatch': 0, 'atomic_after': 0}
```

Build/targeted tests:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/attention/aotriton_wrap.py \
  hipengine/runtime/qwen35_paro.py tests/test_aotriton_discovery.py
# ok

python3 - <<'PY'
from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import build_aotriton_wrap
lib = build_aotriton_wrap(load=True)
print('built', lib)
PY
# built ~/.cache/hipengine/build/aotriton_wrap-f055fc0f2aacfe03/hipengine_aotriton_wrap.so

python3 -m pytest tests/test_qwen35_paro_marlin_k.py tests/test_aotriton_discovery.py \
  tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_resident_batch_layout.py \
  tests/test_qwen35_decode_state.py -q
# 73 passed
```

Correctness fixture gate:

```bash
python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 \
  --attn-aotriton-min-tokens 512 --json /tmp/task14-v3-fixture-aotriton.json
# passed=true, generated_match=true, expected_match=true,
# max_kl=0.0395688706, top1=1.0
```

Diagnostic smoke performance on W7900/gfx1100, Qwen3.5-35B-A3B-PARO w4_paro,
max_layers=40, repeated token id 9707, no graph replay, AOTriton opt-in threshold
512:

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --json /tmp/task14-v3-aotriton-512-128.json
# prefill=2183.260 tok/s, decode=101.462 tok/s
# previous V2 direct-GQA smoke: prefill=2164.732 tok/s
# previous BF16 native fallback smoke: prefill=2045.535 tok/s

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --json /tmp/task14-v3-aotriton-4k-128.json
# prefill=2377.940 tok/s, decode=102.860 tok/s
# previous V2 direct-GQA smoke: prefill=2374.341 tok/s
# previous BF16 native fallback smoke: prefill=662.630 tok/s
```

Retained diagnostic artifact/rollup update:

- `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-v3-prefill-diagnostic.json`
- `benchmarks/README.md` blocked/diagnostic row updated; no current-fastest row
  promoted because AOTriton remains opt-in pending threshold sweep/full
  `LLM.generate()` protocol.
- `benchmarks/CHANGELOG.md` one-liner added.

Conclusion: V3 is correctness-equivalent to the V2 direct-GQA shortcut and now
puts hipEngine on the richer long-term AOTriton API with a reusable persistent
atomic counter.  The next useful step is a threshold sweep; default remains
`attn_aotriton_min_tokens=0` until that lands.

## 2026-05-16 — Task #15 real hipEngine memory reporting

Added process-local hipEngine device allocation high-water tracking and benchmark
memory snapshots so Qwen3.5/PARO smokes can report comparable memory numbers.

Implementation details:

- `hipengine.core.memory.malloc/free` now update tracked allocation counters:
  current bytes, peak bytes, total allocated/freed bytes, active allocation
  count, and peak allocation count.  `reset_memory_stats()` preserves currently
  live tracked allocations while resetting high-water counters.
- `HipRuntime.mem_get_info()` wraps `hipMemGetInfo` for phase-bound sampled HIP
  free/total/used bytes.
- `scripts/qwen35_paro_bench.py` now emits:
  - `memory`: summary with tracked high-water, before/after close current bytes,
    sampled HIP used peak, and session-owned peak.
  - `memory_snapshots`: before-load, after-load, after-prefill,
    after-warmup-decode, after-decode, before-close, and after-close snapshots.
- Caveat documented in JSON: tracked high-water covers hipEngine-owned allocations
  made through `hipengine.core.memory.malloc`; sampled HIP used is not a
  continuous device-wide peak.

Validation:

```bash
python3 -m py_compile hipengine/core/hip.py hipengine/core/memory.py scripts/qwen35_paro_bench.py
# ok

python3 -m pytest tests/test_memory_stats.py tests/test_qwen35_resident_batch_layout.py -q
# 25 passed

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 8 --decode-tokens 1 \
  --warmup-decode-tokens 0 --max-layers 1 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/task15-memory-smoke.json
# emitted memory + memory_snapshots; tracked_peak_allocated_gib=1.8499,
# tracked_current_allocated_bytes_after_close=0, sampled HIP used peak=2.0625 GiB
```

AOTriton V3 diagnostic reruns with memory reporting on W7900/gfx1100,
Qwen3.5-35B-A3B-PARO w4_paro, max_layers=40, repeated token id 9707,
no graph replay, `--attn-aotriton-min-tokens 512`:

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --json /tmp/task15-memory-aotriton-512-128.json
# prefill=2333.435 tok/s, decode=101.302 tok/s
# tracked_peak_allocated=18.6299 GiB, tracked_current_before_close=18.4216 GiB,
# sampled_hip_used_peak=18.6508 GiB, after_close_tracked=0

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 \
  --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --json /tmp/task15-memory-aotriton-4k-128.json
# prefill=2379.676 tok/s, decode=102.407 tok/s
# tracked_peak_allocated=20.8056 GiB, tracked_current_before_close=19.1388 GiB,
# sampled_hip_used_peak=19.4021 GiB, after_close_tracked=0
```

Parent comparison using tracked peak allocated as the closest analogue to parent
Torch `peak_allocated_gib`:

| Workload | hipEngine tracked peak | parent peak allocated | Delta |
| --- | ---: | ---: | ---: |
| 512/128 | 18.63 GiB | 18.80 GiB | -0.17 GiB (-0.9%) |
| 4K/128 | 20.81 GiB | 21.64 GiB | -0.84 GiB (-3.9%) |

Retained diagnostic artifact/rollup update:

- `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-v3-memory-diagnostic.json`
- Updated the AOTriton V3 diagnostic row in `benchmarks/README.md` with tracked
  peak and sampled HIP used memory.
- Added a `benchmarks/CHANGELOG.md` one-liner.  Still no current-fastest row;
  AOTriton remains opt-in until the threshold sweep/full protocol lands.

## 2026-05-16 — Task #16 parent prefill gap audit

Audited the current `nano-vllm-amd` Qwen3.5/PARO parent prefill source against
hipEngine's AOTriton V3 path and documented the prioritized residual-gap table in
`docs/PREFILL.md`.

Read-only source/evidence commands:

```bash
git status -sb && git log --oneline -4
# ## main...origin/main [ahead 1]
# a00c244 feat: report hipEngine memory peaks
# 3252b93 perf: move AOTriton prefill to V3 params
# 454d684 perf: use direct AOTriton GQA prefill
# c7e3bc1 docs: add gguf intake plan

python3 - <<'PY'
from pathlib import Path
from collections import Counter
from hipengine.loading.safetensors import load_weight_index
from hipengine.loading.qwen35_paro import qwen35_paro_config_from_hf
p = Path('/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd')
cfg = qwen35_paro_config_from_hf(load_weight_index(p).config)
print(cfg.num_hidden_layers, Counter(cfg.layer_types))
print('hidden', cfg.hidden_size, 'heads', cfg.num_attention_heads, cfg.num_key_value_heads, 'head_dim', cfg.head_dim)
print('linear heads', cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim, 'conv', cfg.linear_conv_kernel_dim)
PY
# 40 Counter({'linear_attention': 30, 'full_attention': 10})
# hidden 2048 heads 16 2 head_dim 256
# linear heads 16 32 128 128 conv 4
```

Parent files audited:

- `/home/lhl/amd-gpu-tuning/docs/OPTIMAL.md`
- `/home/lhl/amd-gpu-tuning/scripts/bench_paro_native_engine.py`
- `/home/lhl/amd-gpu-tuning/nano-vllm-amd/nanovllm/native/qwen35/paroquant.py`
- Parent artifacts: `benchmarks/results/2026-05-13-source-lineage-qwen35-paro-optimal-512-128.json`,
  `benchmarks/results/2026-05-13-source-lineage-qwen35-paro-optimal-4k-128.json`
- hipEngine artifact: `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-v3-memory-diagnostic.json`

Main finding:

- AOTriton V3 closed the old 4K full-attention cliff; residual prefill gap is now
  nearly shape-invariant: 512/128 is `2333.4` vs parent `2696.4` tok/s (-13.5%),
  and 4K/128 is `2379.7` vs parent `2741.5` tok/s (-13.2%).
- That shape signature points away from the quadratic attention core and toward
  per-layer bulk projection / shared-expert work plus AOTriton dtype/post-pass
  glue.
- Highest-priority gap: parent multi-row prefill uses `F.linear(...)`/rocBLAS-like
  bulk GEMM for `ParoQuantDenseLinear` and `ParoQuantSharedExpert` (parent ledgers
  show `native_aux_dense_linear_calls=280`, `native_shared_expert_dense_calls=80`),
  while hipEngine uses row-wise `dense_gemv_out_fp16(...)` for linear-attention
  A/B and custom scalar W8A16 shared-expert kernels for all `tokens > 1`.
- Next likely prefill lever: avoid/fuse AOTriton-side Q/K/V casts and the BF16
  attention-output gate post-pass.

No GPU benchmark was run for this audit; it is a source/ledger comparison.  The
first follow-up before invasive kernel work should be a matched ROCTX +
`rocprofv3` profile to confirm dense/shared kernels dominate the residual gap.

## 2026-05-16 - Task #17 AOTriton prefill cast-glue reduction

Goal: reduce the explicit Q/K/V cast kernels around opt-in AOTriton prefill while preserving BF16 Q/K/V/Out semantics and fixture correctness.

Implementation:

- Added `hipengine_qwen35_head_rmsnorm_partial_rotary_positions_q_bf16_key_f32`, a prefill-only head-RMSNorm/RoPE variant that writes BF16 query rows directly and keeps key rows as FP32 for the existing paged-KV append.
- Updated the single-request AOTriton prefill path to pass that BF16 Q tensor directly to AOTriton.
- Reused the already-appended BF16 paged KV cache as AOTriton K/V, so the old scratch K F32→BF16 and V FP16→BF16 casts are skipped on the cache-contiguous c=1 prompt path.
- Left fallback scratch casts in `prefill_full_attention_aotriton_varlen_gqa_gate_fp16(...)` for callers that do not provide cache tensors.

Validation:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/rotary/qwen35_rotary.py hipengine/kernels/hip_gfx1100/rotary/__init__.py hipengine/runtime/qwen35_paro.py tests/test_qwen35_rotary_plan.py
python3 -m pytest tests/test_qwen35_rotary_plan.py -q
# 3 passed
python3 -m pytest tests/test_aotriton_discovery.py tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
# 67 passed
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --attn-aotriton-min-tokens 512 --json /tmp/task17-fused-qkv-fixture-aotriton.json
# passed=true, expected_match=true, max_kl=0.039568870612619614, top1_agreement=1.0
```

Note: running `tests/test_qwen35_rotary_plan.py` immediately before `tests/test_aotriton_discovery.py` in one pytest process still leaves the global registry cleared by the pre-existing rotary test setup; the targeted tests above were run as separate pytest processes.

Diagnostic benchmark commands (performance_claim=false; AOTriton still opt-in at threshold 512):

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --attn-aotriton-min-tokens 512 --json /tmp/task17-fused-qkv-512-128.json
# prefill=2317.387 tok/s, decode=100.998 tok/s, tracked_peak=18.620 GiB
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --attn-aotriton-min-tokens 512 --json /tmp/task17-fused-qkv-4k-128.json
# prefill=2378.627 tok/s, decode=102.020 tok/s, tracked_peak=20.728 GiB
```

Compared with `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-v3-memory-diagnostic.json`, throughput is neutral/noisy (512 -0.7%, 4K ~0.0%) while tracked peak memory drops by ~0.010 GiB at 512 and ~0.078 GiB at 4K.  This suggests cast glue is worth keeping cleaned up but is not the main residual prefill gap; P0 bulk dense/shared-expert work remains the likely next win.

Retained artifact: `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-cast-glue-diagnostic.json`.

Lineage hygiene for the kernel touch:

```bash
python3 scripts/check_lineage.py --kind kernel --diff stat
# DRIFT remains on parent qwen35_expert.hip/smoke.hip/paroquant_kernels.py/paroquant_fusedw4.py vs baseline 22405a9; no code copied from the drifted parent for this task. qwen35_rotary.hip change is a hipEngine-only BF16-Q output specialization of the already-ported vector-position prelude.
```

## 2026-05-16 — Task 18 AOTriton gate+rotate fusion diagnostic

Implemented the smallest downstream fusion for the AOTriton single-request prefill tail: `paro_rotate1_bf16_gate_fp16(...)` reads BF16 AOTriton attention output plus FP16 gate, rounds `attention * sigmoid(gate)` to FP16, then applies the same PARO rotate1 math before the FP16 O projection.  The runtime AOTriton path now requests raw BF16 attention output, aliases the unused FP16 `scratch.gated_attn` bytes as the BF16 AOTriton output tensor, and skips the separate `aotriton_gate_mul_bf16_to_fp16` launch/intermediate on that path.  The older gate method remains available as a fallback helper.

Validation:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/rotary/paro_rotate.py hipengine/kernels/hip_gfx1100/rotary/__init__.py hipengine/runtime/qwen35_paro.py tests/test_paro_rotate_plan.py
python3 -m pytest tests/test_paro_rotate_plan.py -q
# 3 passed
python3 -m pytest tests/test_aotriton_discovery.py tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q
# 67 passed
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --attn-aotriton-min-tokens 512 --json /tmp/task18-fused-gate-rotate-alias-fixture-aotriton.json
# passed=true, expected_match=true, max_kl=0.039568870612619614, top1_agreement=1.0
```

Diagnostic benchmark commands (performance_claim=false; AOTriton still opt-in at threshold 512):

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --attn-aotriton-min-tokens 512 --json /tmp/task18-fused-gate-rotate-alias-512-128.json
# prefill=2312.857 tok/s, decode=101.703 tok/s, tracked_peak=18.581 GiB, owned_peak=1.554 GiB
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --attn-aotriton-min-tokens 512 --json /tmp/task18-fused-gate-rotate-alias-4k-128.json
# prefill=2371.534 tok/s, decode=102.211 tok/s, tracked_peak=20.415 GiB, owned_peak=1.930 GiB
```

Compared with `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-cast-glue-diagnostic.json`, throughput is neutral/slightly negative (512 -0.2%, 4K -0.3%), but tracked peak memory drops by 0.039 GiB at 512 and 0.3125 GiB at 4K.  Retained as a launch/memory cleanup diagnostic, not a promoted throughput win; the P0 dense/shared-expert prefill gap remains the next likely target.

Retained artifact: `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-gate-rotate-diagnostic.json`.

Lineage hygiene:

```bash
python3 scripts/check_lineage.py --kind kernel --diff stat
# DRIFT remains on parent qwen35_expert.hip/smoke.hip/paroquant_kernels.py/paroquant_fusedw4.py vs baseline 22405a9; no parent kernel code was copied for this task. The new rotate gate helper is a hipEngine-only fusion of existing gate and rotate semantics.
```

## 2026-05-16 — Task 19 easy prefill fusion source audit

Reviewed hipEngine and nano-vllm-amd Qwen3.5/PARO prefill call structure for small launch/materialization fusions after the AOTriton cast and gate+rotate cleanups.  No GPU benchmark was run; this is a source/call-graph audit only.  Added a ranked candidate table to `docs/PREFILL.md`.

Top candidates identified:

1. Linear-attention output tail: fuse `qwen35_gdn_prefill_rmsnorm_gate_fp16(...)` with `paro_rotate1_fp16(...)` for the linear-attention out projection.  This should remove one launch and `recurrent_bf16` materialization on 30 linear-attention layers when `head_v_dim == group_size` (Qwen3.5/PARO natural 128-wide groups).
2. MoE shared gate: add a prefill-only router variant that writes `sigmoid(shared_gate_logit)` in the shared-gate column after top-k selection, allowing grouped prefill to skip `w8a16_shared_gate_sigmoid_fp32(...)`.  Keep c=1 unchanged because its combine kernel expects raw logits.
3. Full-attention prelude: AOTriton-first fused Q/gate split + K cast + head RMSNorm/RoPE kernel reading FP16 q_proj/K projection directly and writing gate FP16, BF16 Q, and FP32/BF16 K outputs.
4. Packed c>N linear-attention: add a lowp-input segment conv to remove the `fp16_to_f32(qkv)` cast before `qwen35_linear_attn_conv_prefill_segments_f32(...)`.
5. MoE metadata: combine small group-prefix/tile-map/metadata-zero launches if profiler shows they matter.

Deferred as not easy: input RMSNorm+PARO input rotation, rotate fused into generic W4 WMMA projections, and folding sorted-lane selected-output accumulation into shared down combine.

## 2026-05-16 — Task 20 decode graph replay validation

Implemented the task-20 graph-replay closure around the existing c=1 Qwen3.5/PARO resident decode graph path:

- Added `record_i64_scalar_indexed(...)` to `hipengine/kernels/hip_gfx1100/runtime/state.hip` so replay can append each generated token id to a device buffer without host work inside the graph.
- Extended `Qwen35ParoResidentSession.capture_decode_graph(...)` with `max_replay_steps` (bakes split-K attention capacity for the whole measured replay span) and optional `record_steps` for correctness gates.
- Added `scripts/qwen35_decode_graph_fixture_gate.py`, which runs native prefill twice, compares eager decode vs HIP graph replay generated IDs, checks fixture expected IDs, and compares final logits/top-1/KL.
- Updated `scripts/qwen35_paro_bench.py --graph-replay-decode` to pass `max_replay_steps=args.decode_tokens`.

Validation:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/runtime/state.py hipengine/kernels/hip_gfx1100/runtime/__init__.py hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_paro_bench.py scripts/qwen35_decode_graph_fixture_gate.py tests/test_runtime_state_plan.py
python3 -m pytest tests/test_runtime_state_plan.py -q
# 3 passed

python3 scripts/qwen35_decode_graph_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --attn-aotriton-min-tokens 512 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/task20-decode-graph-fixture-gate.json
# passed=true, generated_match=true, expected_match=true, final_kl=0.0, final_top1_match=true
```

Diagnostic benchmark commands (performance_claim=false; AOTriton still opt-in at threshold 512):

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --attn-aotriton-min-tokens 512 --graph-replay-decode --json /tmp/task20-graph-512-128.json
# prefill=2312.754 tok/s, decode=109.340 tok/s, tracked_peak=18.581 GiB

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --attn-aotriton-min-tokens 512 --graph-replay-decode --json /tmp/task20-graph-4k-128.json
# prefill=2372.725 tok/s, decode=110.303 tok/s, tracked_peak=20.415 GiB
```

Compared with the prior no-graph AOTriton gate-rotate diagnostic:

| Workload | Previous decode | Graph decode | Delta | Parent step-graph decode | Gap vs parent |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 101.703 tok/s | 109.340 tok/s | +7.5% | 116.050 tok/s | -5.8% |
| 4K/128 | 102.211 tok/s | 110.303 tok/s | +7.9% | 113.049 tok/s | -2.4% |

Retained artifact: `benchmarks/results/2026-05-16-hipengine-qwen35-decode-graph-replay-diagnostic.json`.

Caveat: this closes the c=1 reusable-step decode replay gap for the opt-in AOTriton single-request path.  It does not make the compact c>N serial decode bridge c-aware; that remains separate future work.

Additional targeted regression bundle after the artifact write:

```bash
python3 -m pytest tests/test_runtime_state_plan.py tests/test_qwen35_decode_state.py tests/test_aotriton_discovery.py -q
# 46 passed
```

Lineage hygiene for the runtime-state kernel touch:

```bash
python3 scripts/check_lineage.py --kind kernel --diff stat
# DRIFT remains on parent qwen35_expert.hip/smoke.hip/paroquant_kernels.py/paroquant_fusedw4.py vs baseline 22405a9.
# No parent kernel code was copied for this task; `record_i64_scalar_indexed` is a hipEngine-only graph-gate helper.
```

## 2026-05-16 — Task 21 AOTriton threshold sweep

Ran the AOTriton opt-in threshold sweep after real memory reporting and decode graph replay landed.  Hardware/env smoke before the sweep:

```bash
python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
# hip OK
rocminfo | grep -E 'Name:|gfx'
# AMD Radeon Pro W7900 / gfx1100 present
```

Short-prompt sweep commands used cached HIP builds, repeated token id `9707`, `max_layers=40`, and `decode_tokens=0` to isolate prefill.  For each prompt length `P in {32,64,128,256,512,1024,4096}` I ran:

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length P --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/task21-p${P}-native-prefill.json
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length P --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --attn-aotriton-min-tokens 32 --json /tmp/task21-p${P}-aot32-prefill.json
```

Results (single-run diagnostics, `performance_claim=false`):

| Prompt | Native prefill tok/s | Forced AOTriton tok/s | Delta | Native peak GiB | AOTriton peak GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 605.657 | 504.397 | -16.7% | 18.331 | 18.334 |
| 64 | 994.069 | 829.395 | -16.6% | 18.345 | 18.350 |
| 128 | 1464.792 | 1304.824 | -10.9% | 18.371 | 18.381 |
| 256 | 1892.317 | 1826.457 | -3.5% | 18.429 | 18.449 |
| 512 | 2146.479 | 2284.584 | +6.4% | 18.541 | 18.580 |
| 1024 | 1815.743 | 2498.659 | +37.6% | 18.763 | 18.842 |
| 4096 | 662.419 | 2356.051 | +255.7% | 20.099 | 20.414 |

Crossover is between 256 and 512 prompt tokens.  Simulating thresholds `{0,32,64,128,256,512}` from the disabled/forced rows picks threshold `512` as the only tested policy that avoids short-prompt regressions while still selecting AOTriton at the first winning length.

Graph-replay workload rows (current parent-comparison protocol):

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --graph-replay-decode --json /tmp/task21-p512-native_graph.json
# prefill=2125.642 tok/s, decode=109.225 tok/s, tracked_peak=18.542 GiB
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --graph-replay-decode --attn-aotriton-min-tokens 512 --json /tmp/task21-p512-aot512_graph.json
# prefill=2270.750 tok/s, decode=109.123 tok/s, tracked_peak=18.581 GiB
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --graph-replay-decode --json /tmp/task21-p4096-native_graph.json
# prefill=662.873 tok/s, decode=109.980 tok/s, tracked_peak=20.100 GiB
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --graph-replay-decode --attn-aotriton-min-tokens 512 --json /tmp/task21-p4096-aot512_graph.json
# prefill=2345.670 tok/s, decode=110.091 tok/s, tracked_peak=20.415 GiB
```

Correctness gates for the selected threshold:

```bash
python3 scripts/qwen35_native_prefill_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --attn-aotriton-min-tokens 512 --json /tmp/task21-aot512-prefill-fixture-gate.json
# passed=true, expected_match=true, max_kl=0.039568870612619614, top1_agreement=1.0
python3 scripts/qwen35_decode_graph_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --attn-aotriton-min-tokens 512 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/task21-aot512-decode-graph-fixture-gate.json
# passed=true, generated_match=true, expected_match=true, final_kl=0.0, final_top1_match=true
```

Decision: recommend `--attn-aotriton-min-tokens 512` for deployments with the pinned AOTriton runtime installed.  Keep `PrefillConfig.attn_aotriton_min_tokens=0` as the code default because AOTriton is an optional fetched runtime; flipping the default before an absent-runtime fallback exists would make baseline sessions fail.

Retained artifact: `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-threshold-sweep-diagnostic.json`.

## 2026-05-16 — Long checkpoint: 4K/4K, 32K/128, attempted 128K/128

Ran a long-shape checkpoint for the current hipEngine Qwen3.5/PARO path: opt-in AOTriton threshold 512 plus one-step decode graph replay.  Hardware/env smoke stayed green (`libamdhip64.so` loads, W7900/gfx1100 visible).  These rows are diagnostic (`performance_claim=false`) because no new long-context oracle fixture was run; correctness context is inherited from the threshold-512 fixture gates in `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-threshold-sweep-diagnostic.json`.

hipEngine commands/results:

```bash
python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 4096 --decode-tokens 4096 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --attn-aotriton-min-tokens 512 --graph-replay-decode --json /tmp/task-long-checkpoint-4k-4k.json
# prefill=2379.818 tok/s, decode=108.930 tok/s, tracked_peak=20.529 GiB, sampled_hip_peak=19.131 GiB

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 32768 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --attn-aotriton-min-tokens 512 --graph-replay-decode --json /tmp/task-long-checkpoint-32k-128.json
# prefill=1718.308 tok/s, decode=93.933 tok/s, tracked_peak=35.100 GiB, sampled_hip_peak=22.052 GiB

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 131072 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --attn-aotriton-min-tokens 512 --graph-replay-decode --json /tmp/task-long-checkpoint-128k-128.json
# blocked: HIP error 2 out of memory while reserving reserve_linear_attention_scratch -> linear_attn.out_rot
```

Also reran the nano-vllm-amd parent at 4K/4K for comparison context using the short/mid OPTIMAL env from `~/amd-gpu-tuning/docs/OPTIMAL.md`:

```bash
cd /home/lhl/amd-gpu-tuning && <OPTIMAL env> PYTHONPATH=nano-vllm-amd:paroquant mamba run -n therock --no-capture-output \
  python3 scripts/bench_paro_native_engine.py --prompt-len 4096 --decode-len 4096 --decode-use-step-graph-replay --output /tmp/task-long-checkpoint-parent-4k-4k.json --json
# prefill=2728.305 tok/s, decode=104.963 tok/s, peak_allocated=21.719 GiB
# caveat: decode_graph_replay_match=false at token 581; graph-compatible replay matches. docs/OPTIMAL.md already records this 4K/4K divergence caveat.
```

Comparison:

| Workload | hipEngine prefill | hipEngine decode | Parent/source | Gap / status |
| --- | ---: | ---: | --- | --- |
| 4K/4K | 2379.818 | 108.930 | local parent rerun 2728.305 / 104.963 | prefill -12.8%, decode +3.8%; parent 4K/4K has known graph/eager divergence caveat |
| 32K/128 | 1718.308 | 93.933 | `~/amd-gpu-tuning/docs/OPTIMAL.md` 1880 / 98.8 | prefill -8.6%, decode -4.9%; hipEngine tracked peak 35.10 GiB vs parent 21.37 GiB |
| 128K/128 | blocked OOM | — | `~/amd-gpu-tuning/docs/OPTIMAL.md` 914 / 62.6 | hipEngine lacks wired long-context chunking and reserves unchunked linear-attn scratch |

Takeaway: the next long-context blocker is not AOTriton thresholding; it is wiring long-context prefill chunking.  Parent 32K/128 and 128K/128 rows add `NANOVLLM_PARO_PREFILL_LINEAR_CHUNK_SIZE=1024`, `NANOVLLM_PARO_MOE_CHUNK_SIZE=1024`, and full-attention post/RoPE/query chunk sizes.  hipEngine `PrefillConfig` already has analogous fields, but the current single-request native prefill path does not apply them, so 128K cannot run on W7900 and 32K uses much higher peak memory than parent.

Retained artifact: `benchmarks/results/2026-05-16-hipengine-qwen35-long-checkpoint-diagnostic.json`.

## 2026-05-16 — Task 25: wire long-context prefill chunking

Implemented internal chunking for the Qwen3.5/PARO single-request native prefill path.  `PrefillConfig.linear_chunk_size` now chunks linear-attention layers, `full_attn_query_chunk_size` chunks full-attention query rows, and `moe_chunk_size` is available as a config/CLI field (currently limiting layer chunks when it is the only smaller configured bound).  Chunked full-attention AOTriton required changing the v3 wrapper causal window from top-left to bottom-right alignment so a query chunk at positions `[start:end)` can attend to cached keys `[0:end)` with the correct causal mask.

Validation / smoke commands:

```bash
python3 -m py_compile hipengine/runtime/prefill.py hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_paro_bench.py scripts/qwen35_native_prefill_fixture_gate.py scripts/qwen35_decode_graph_fixture_gate.py tests/test_qwen35_resident_batch_layout.py
python3 -m pytest tests/test_qwen35_resident_batch_layout.py -q
# 24 passed

python3 scripts/qwen35_native_prefill_fixture_gate.py --max-layers 40 --max-new-tokens 8 --attn-aotriton-min-tokens 512 --json /tmp/task25-nochunk-fixture-v2.json
# passed=true, max_kl=0.039568870612619614, top1_agreement=1.0

python3 scripts/qwen35_native_prefill_fixture_gate.py --max-layers 40 --max-new-tokens 8 --attn-aotriton-min-tokens 512 --prefill-linear-chunk-size 128 --prefill-moe-chunk-size 128 --prefill-full-attn-query-chunk-size 128 --prefill-full-attn-post-chunk-size 128 --prefill-full-attn-rope-chunk-size 128 --json /tmp/task25-chunk-fixture-gate-v2.json
# passed=true, max_kl=0.039568870612619614, top1_agreement=1.0

python3 scripts/qwen35_native_prefill_fixture_gate.py --max-layers 40 --max-new-tokens 8 --attn-aotriton-min-tokens 512 --prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 --prefill-full-attn-query-chunk-size 4096 --prefill-full-attn-post-chunk-size 1024 --prefill-full-attn-rope-chunk-size 1024 --json /tmp/task25-parentchunk-fixture-gate.json
# passed=true, max_kl=0.039568870612619614, top1_agreement=1.0

python3 scripts/qwen35_paro_bench.py --token-id 9707 --prompt-length 131072 --decode-tokens 0 --warmup-decode-tokens 0 --max-layers 4 --attn-aotriton-min-tokens 512 --prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 --prefill-full-attn-query-chunk-size 4096 --prefill-full-attn-post-chunk-size 1024 --prefill-full-attn-rope-chunk-size 1024 --json /tmp/task25-128k-maxlayers4-chunk-smoke.json
# no OOM through first 3 linear-attention layers + first full-attention layer; prefill=12.846s, tracked_peak=5.844 GiB for max_layers=4 diagnostic smoke
```

Full 40-layer 128K/128 and with/without chunk tables remain the follow-up benchmark task.

## 2026-05-16 — Task 26: chunked prefill validation sweep

Validated task 25 chunking changes with narrow CPU/Python checks and fixture gates.  No benchmark rows retained; all timings below are correctness-gate diagnostics only.

```bash
python3 -m py_compile hipengine/runtime/prefill.py hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_paro_bench.py scripts/qwen35_native_prefill_fixture_gate.py scripts/qwen35_decode_graph_fixture_gate.py tests/test_qwen35_resident_batch_layout.py
python3 -m pytest tests/test_qwen35_resident_batch_layout.py -q
# 25 passed

python3 scripts/qwen35_native_prefill_fixture_gate.py --max-layers 40 --max-new-tokens 8 --json /tmp/task26-default-unchunked-fixture.json
# default unchunked/no-AOT path: passed=true, max_kl=0.04520468681522189, top1_agreement=1.0, expected_match=true

python3 scripts/qwen35_native_prefill_fixture_gate.py --max-layers 40 --max-new-tokens 8 --prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 --prefill-full-attn-query-chunk-size 4096 --prefill-full-attn-post-chunk-size 1024 --prefill-full-attn-rope-chunk-size 1024 --json /tmp/task26-short-largechunk-noop-fixture.json
# short/no-op chunk knobs: passed=true, max_kl=0.04520468681522189, top1_agreement=1.0, expected_match=true

python3 scripts/qwen35_native_prefill_fixture_gate.py --max-layers 40 --max-new-tokens 8 --attn-aotriton-min-tokens 512 --prefill-linear-chunk-size 128 --prefill-moe-chunk-size 128 --prefill-full-attn-query-chunk-size 128 --prefill-full-attn-post-chunk-size 128 --prefill-full-attn-rope-chunk-size 128 --json /tmp/task26-chunked-128-fixture.json
# actual chunked path: passed=true, max_kl=0.039568870612619614, top1_agreement=1.0, expected_match=true

python3 scripts/qwen35_decode_graph_fixture_gate.py --max-layers 40 --attn-aotriton-min-tokens 512 --prefill-linear-chunk-size 128 --prefill-moe-chunk-size 128 --prefill-full-attn-query-chunk-size 128 --prefill-full-attn-post-chunk-size 128 --prefill-full-attn-rope-chunk-size 128 --json /tmp/task26-chunked-decode-graph-fixture.json
# chunked prefill + decode graph: passed=true, generated_match=true, expected_match=true, final_kl=0.0, final_top1_match=true
```

Default behavior remains unchanged for short/no-op chunk prompts: all chunk sizes default to 0 in `PrefillConfig`, and the 512-token fixture with parent long-context chunk sizes larger than the prompt produced the same native top-1 sequence and KL gate as the default unchunked/no-AOT run.

## 2026-05-16 — Task 28: retained chunking benchmark artifact

Recorded the task 27 chunked-vs-unchunked long-context benchmark as a retained diagnostic artifact: `benchmarks/results/2026-05-16-hipengine-qwen35-prefill-chunking-diagnostic.json`.  Hardware: W7900/gfx1100.  Model: Qwen3.5-35B-A3B-PARO `w4_paro`.  Common hipEngine policy: `--attn-aotriton-min-tokens 512 --graph-replay-decode --warmup-decode-tokens 1 --max-layers 40 --require-cached-build`.  Chunked policy mirrors parent long-context knobs: linear/MoE/post/RoPE chunks `1024`, full-attn query chunk `4096`.  Correctness context comes from task 26: default unchunked fixture gate passed (`max_kl=0.04520468681522189`, top-1 `1.0`), chunked 128-row+AOTriton fixture gate passed (`max_kl=0.039568870612619614`, top-1 `1.0`), and chunked prefill + decode graph fixture passed (`final_kl=0.0`, generated IDs match eager/fixture).  `performance_claim=false`: these are single-run resident-session diagnostics, not public `LLM.generate()` accepted rows.

Exact commands:

```bash
COMMON="--token-id 9707 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --attn-aotriton-min-tokens 512 --graph-replay-decode"
CHUNK="--prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 --prefill-full-attn-query-chunk-size 4096 --prefill-full-attn-post-chunk-size 1024 --prefill-full-attn-rope-chunk-size 1024"
python3 scripts/qwen35_paro_bench.py $COMMON --prompt-length 4096 --json /tmp/task27-4k128-unchunked.json
python3 scripts/qwen35_paro_bench.py $COMMON $CHUNK --prompt-length 4096 --json /tmp/task27-4k128-chunked.json
python3 scripts/qwen35_paro_bench.py $COMMON --prompt-length 32768 --json /tmp/task27-32k128-unchunked.json
python3 scripts/qwen35_paro_bench.py $COMMON $CHUNK --prompt-length 32768 --json /tmp/task27-32k128-chunked.json
python3 scripts/qwen35_paro_bench.py $COMMON $CHUNK --prompt-length 131072 --json /tmp/task27-128k128-chunked.json
python3 scripts/qwen35_paro_bench.py $COMMON --prompt-length 131072 --json /tmp/task27-128k128-unchunked.json
# unchunked 128K exits 1 with HIP error 2 OOM while reserving linear_attn.out_rot
```

Measured hipEngine rows:

| Workload | Unchunk prefill | Chunk prefill | Delta | Unchunk decode | Chunk decode | Tracked peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K/128 | 2370.229 | 2504.959 | +5.7% | 110.168 | 110.117 | 20.415 → 19.875 |
| 32K/128 | 1731.976 | 1886.344 | +8.9% | 93.867 | 93.923 | 35.100 → 20.688 |
| 128K/128 | OOM | 1002.409 | unblocked | — | 61.051 | OOM → 23.656 |

Chunked hipEngine vs parent `~/amd-gpu-tuning/docs/OPTIMAL.md` rows:

| Workload | hipEngine chunked | Parent | Delta |
| --- | ---: | ---: | ---: |
| 4K/128 prefill | 2504.959 | 2703.0 | -7.3% |
| 4K/128 decode | 110.117 | 112.0 | -1.7% |
| 32K/128 prefill | 1886.344 | 1880.0 | +0.3% |
| 32K/128 decode | 93.923 | 98.8 | -4.9% |
| 128K/128 prefill | 1002.409 | 914.0 | +9.7% |
| 128K/128 decode | 61.051 | 62.6 | -2.5% |

Takeaway: chunking fixes the 128K OOM and removes the 32K memory cliff.  Long-context prefill is now at/above parent docs for 32K/128 and 128K/128 in this resident-runner diagnostic; decode remains slightly behind parent.

## 2026-05-16 — Qwen3.5 comparison table script + 512/128 row

Added a retained comparison-table snapshot and `scripts/qwen35_compare_tables.py` so the current hipEngine resident-runner rows can be printed as separate prefill/decode/memory tables against `nano-vllm-amd`, llama.cpp HIP, or llama.cpp Vulkan.  The script is hardcoded on purpose; it is not a benchmark runner.

New 512/128 current row, using the same installed-AOTriton + decode-graph + parent-style chunk policy as the chunking checkpoint (chunk sizes exceed the 512 prompt, so this is a no-op chunk row):

```bash
COMMON="--token-id 9707 --decode-tokens 128 --warmup-decode-tokens 1 --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --attn-aotriton-min-tokens 512 --graph-replay-decode"
CHUNK="--prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 --prefill-full-attn-query-chunk-size 4096 --prefill-full-attn-post-chunk-size 1024 --prefill-full-attn-rope-chunk-size 1024"
python3 scripts/qwen35_paro_bench.py $COMMON $CHUNK --prompt-length 512 --json /tmp/task29-512k128-chunked.json
# prefill=2216.487043410022 tok/s, decode=109.10539462615208 tok/s, tracked_peak=18.58110980410129 GiB, hip_used_peak_sampled=18.60193634033203 GiB
```

Retained artifact: `benchmarks/results/2026-05-16-hipengine-qwen35-comparison-tables-diagnostic.json`.

Primary script commands:

```bash
python3 scripts/qwen35_compare_tables.py nano-vllm-amd
python3 scripts/qwen35_compare_tables.py 'llama.cpp HIP'
python3 scripts/qwen35_compare_tables.py vulkan
python3 scripts/qwen35_compare_tables.py all
```

Current `nano-vllm-amd` view:

| Workload | hipEngine prefill | nano parent prefill | Prefill delta | hipEngine decode | nano parent decode | Decode delta | hipEngine peak | nano parent peak | Peak delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 2216.487 | 2557.000 | -13.3% | 109.105 | 115.700 | -5.7% | 18.581 | 18.860 | -0.28 GiB |
| 4K/128 | 2504.959 | 2703.000 | -7.3% | 110.117 | 112.000 | -1.7% | 19.875 | 21.640 | -1.77 GiB |
| 32K/128 | 1886.344 | 1880.000 | +0.3% | 93.923 | 98.800 | -4.9% | 20.688 | 21.370 | -0.68 GiB |
| 128K/128 | 1002.409 | 914.000 | +9.7% | 61.051 | 62.600 | -2.5% | 23.656 | 27.420 | -3.76 GiB |

llama.cpp HIP and Vulkan script views use the split rows from `~/amd-gpu-tuning/PLAN-LONGCONTEXT.md`; those rows do not have retained memory values, so the memory tables intentionally print `—` for baseline and delta.

Validation:

```bash
python3 -m py_compile scripts/qwen35_compare_tables.py
python3 -m json.tool benchmarks/results/2026-05-16-hipengine-qwen35-comparison-tables-diagnostic.json >/tmp/task31-comparison-artifact.pretty.json
python3 scripts/qwen35_compare_tables.py all >/tmp/task31-comparison-all.md
python3 - <<'PY'
# imported the script and asserted hardcoded rows match the retained artifact
PY
# artifact/script data match

git diff --check
```

## 2026-05-16 — Optimization plan doc for beating parent/llama.cpp

Reviewed the current hipEngine docs and benchmark rollup plus parent references under `~/amd-gpu-tuning/` to create `docs/OPTIMIZE.md`, the batch-1 Qwen3.5/PARO grind plan.

Inputs reviewed included:

- hipEngine: `docs/PREFILL.md`, `docs/KERNELS.md`, `docs/ROOFLINE.md`, `docs/BENCHMARK.md`, `docs/IMPLEMENTATION.md`, `docs/MARLIN.md`, `docs/GGUF.md`, `docs/DFLASH.md`, `docs/MTP.md`, `benchmarks/README.md`, and the comparison-table artifact/script.
- Parent workspace: `docs/OPTIMAL.md`, `PLAN-PAROQUANT.md`, `PLAN-PAROQUANT2.md`, `PLAN-LONGCONTEXT.md`, `docs/LLAMACPP-VULKAN.md`, `PR_COMMENT-llamacpp-hip-unroll600.md`, `LESSONS-LEARNED.md`, and recent Marlin-K WORKLOG entries.

Current board captured in the new plan:

- vs `nano-vllm-amd`: memory already wins all rows; prefill needs +15.4% at 512/128 and +7.9% at 4K/128; decode needs +6.0%, +1.7%, +5.2%, +2.5% at 512/4K/32K/128K.
- vs llama.cpp HIP: only 512/128 prefill is behind, needing about +9.9%; decode already wins the retained split rows.
- vs llama.cpp Vulkan: prefill already wins all retained split rows; decode needs +16.9%, +9.1%, +4.4%, +5.6% at 512/4K/32K/128K.

Plan decision: do an audit-first grind rather than another blind kernel loop.  Lane 0 is benchmark/protocol promotion; Lane 1 is matched ROCTX/rocprof profiles; Lane 2 attacks short/mid prefill through bulk dense/shared-expert GEMM-shaped paths; Lane 3 attacks decode through replay-only dispatch/rotation/W4/profile-driven attention work; Lane 4 preserves the memory advantage; Lane 5 defers c>N decode until batch-1 board closure.  `docs/PLAN.md` project-structure docs list now points at `docs/OPTIMIZE.md`.

## 2026-05-17 — Reorganize `docs/OPTIMIZE.md` into per-category candidate tables

Reviewed every parent and hipEngine doc named in `AGENTS.md` "Key Files" plus the parent
optimization references (`~/amd-gpu-tuning/docs/OPTIMAL.md`, `PLAN-PAROQUANT.md`,
`PLAN-PAROQUANT2.md` including the §11 post-mortem and the live §12 punchlist,
`PLAN-LONGCONTEXT.md`, `docs/LLAMACPP-VULKAN.md`, `PR_COMMENT-llamacpp-hip-unroll600.md`,
`LESSONS-LEARNED.md`).  Rewrote `docs/OPTIMIZE.md` so each optimization lane is a table with
ID / candidate / source-lineage / expected prefill Δ / expected decode Δ / memory / risk / status /
result columns, per the user request to be able to grind through candidates and fill in results.

New layout:

- §1 Scoreboard preserved (3 comparison tables + compact goal).
- §2 Strategy paragraph.
- §3 Non-negotiable promotion gates (correctness / no torch / registry-only / memory / rollup /
  generated-sample equality).
- §4 Lane M — measurement and protocol promotion (blocks everything: accepted
  `LLM.generate()` rows, comparison-table auto-refresh, matched rocprof captures, per-bucket
  Amdahl, local Vulkan calibration).
- §5 Lane P — prefill (P1 bulk dense / shared-expert; P2 AOTriton glue; P3 boundary fusion;
  P4 native FA-2; P5 long-context chunking).
- §6 Lane D — decode (D1 dispatch / boundary fusion; D2 Marlin-K vec8 + qweight-neutral port plus
  informational rows for every parent §12 rejection; D3 long-context attention split-K + grouped-GQA;
  D4 launch-floor / replay hygiene; D5 decode glue ledger; D6 DFlash/MTP).
- §7 Lane A — memory guardrails.
- §8 Lane W — compiler / build profile sweeps.
- §9 Lane S — c>N / serving (deferred).
- §10 Lane K — other quant formats / models (deferred).
- §11 Do-not-chase list (curated from parent rejections + LESSONS-LEARNED).
- §12 First concrete punchlist (M.1/M.2 → M.3/M.4 → M.5 → P1.1/P1.2/P1.4 → W.1 →
  D1.1/D1.4/D2.1 → D3.1/D3.2/D3.3).
- §13 Reference map.

Status legend explicitly distinguishes `pending` / `in-progress` / `accepted` / `rejected` /
`parked` / `deferred`, with `parked (parent rejected ...)` rows kept as informational so future
agents do not re-run the same dead-end experiments (Marlin-K B1-B7, C1-C5, E1-E6, F1/F2, naive
sudot4, LDS staging, wave32-only sweeps over AWQ layout, multi-step graph replay, etc.).

Validation:

```bash
wc -l docs/OPTIMIZE.md
# 457
grep -c "^| " docs/OPTIMIZE.md
# 155 table rows
grep -E "^### " docs/OPTIMIZE.md
# Section headings: 1.1-1.4 scoreboard; 5.1-5.5 P1-P5; 6.1-6.6 D1-D6
```

No code, kernel, or benchmark changes in this commit; the doc rewrite is the logical unit.
Unrelated untracked `hipengine/util/__init__.py` is owned by another agent per AGENTS.md
coordination rules and is left alone.

## 2026-05-17 — Tool: llama.cpp peak VRAM via amdgpu sysfs sampler

The `qwen35_compare_tables` rows for `llama.cpp-hip` and `llama.cpp-vulkan` currently carry
`peak_gib = null` because `llama-bench` does not log peak GPU memory anywhere — its `-o json`
emits only tok/s. The user requested an external watcher we can configure in milliseconds and
run in `-r 1` sweep mode where token-rate perturbation is acceptable.

Chosen mechanism: poll `/sys/class/drm/card*/device/mem_info_vram_used` from a background
thread. Reads take a few microseconds, the amdgpu kernel driver byte-accurately accounts for
VRAM committed by **any** userspace backend (HIP, Vulkan, OpenCL), so the same code captures
both llama.cpp HIP and llama.cpp Vulkan on a single scale. No HIP context, no `hipMemGetInfo`,
no `rocm-smi` subprocess churn, and no llama.cpp patch.

Added:

- `hipengine/util/__init__.py` — package marker. (The pre-existing untracked stub from the
  previous logical unit had no content referenced anywhere; the docstring now matches the
  module-style we use elsewhere.)
- `hipengine/util/amdgpu_vram.py` — `list_amdgpu_cards()`, `select_card()`, `read_vram_used()`,
  `VramSampler` (context-manager, configurable `interval_ms`, optional full-trace capture),
  `VramSamples` result struct. Standalone `python3 -m hipengine.util.amdgpu_vram --list /
  --poll/--duration/--json` CLI for sanity checks.
- `scripts/llamacpp_bench_with_peak.py` — wraps `llama-bench` per workload, splits each
  `<prompt>/<gen>` token (e.g. `512/128 4K/128 32K/128 128K/128`) into two invocations using
  the canonical PLAN-LONGCONTEXT split protocol (`-p P -n 0 -d 0` for prefill, `-p 0 -n N
  -d P` for decode-at-offset). Polls VRAM during each invocation, parses llama-bench JSON for
  tok/s, parses stderr `*_buffer size = X MiB` lines as a sanity cross-check, emits a
  benchmarks/results-shaped JSON artifact, and prints a Markdown table. Defaults to `-r 1`,
  `-fa 1`, `f16` KV, `-ngl 99`, `--poll 50` (ms). `--backend auto` detects HIP vs Vulkan from
  the llama-bench stderr banner.

Smoke validation (W7900, this host):

```bash
python3 -m py_compile hipengine/util/amdgpu_vram.py scripts/llamacpp_bench_with_peak.py
python3 -m hipengine.util.amdgpu_vram --list
# card1\tpci=0000:c3:00.0\ttotal=44.984 GiB

python3 -m hipengine.util.amdgpu_vram --poll 5 --duration 1 --json
# baseline_gib=0.026, peak_gib=0.026, samples_count=203, interval=5 ms (idle)

python3 scripts/llamacpp_bench_with_peak.py \
  --llama-bench /home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-bench \
  --model /models/gguf/llama-2-7b.Q4_0.gguf \
  --workloads 512/32 --poll 10 --backend hip
# row peak 4.331 GiB, prefill 3411 tok/s, decode 105 tok/s — sane for Llama-2-7B Q4_0.

python3 scripts/llamacpp_bench_with_peak.py \
  --llama-bench /home/lhl/llama.cpp/llama.cpp-vulkan/build/bin/llama-bench \
  --model /models/gguf/llama-2-7b.Q4_0.gguf \
  --workloads 512/32 --poll 10 --backend vulkan
# row peak 4.106 GiB, prefill 884 tok/s, decode 130 tok/s — sane for Llama-2-7B Q4_0 Vulkan.

python3 scripts/llamacpp_bench_with_peak.py \
  --llama-bench /home/lhl/amd-gpu-tuning/llama.cpp/build/bin/llama-bench \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --workloads 512/128 --poll 10 --backend hip
# row peak 21.125 GiB, prefill 2388 tok/s, decode 88 tok/s — matches PLAN-LONGCONTEXT split
# rows (2436.049 / 85.487) within run variance and confirms the peak measurement works on
# the Qwen3.6 GGUF llama.cpp was built against.
```

Next step (separate logical unit): run the full 4-workload sweep on both backends with
`--poll 10`, retain the artifacts under `benchmarks/results/`, and wire the result into
`scripts/qwen35_compare_tables.py` so the `Memory / peak GiB` table fills its blanks.

## 2026-05-16 — AOTriton vendored baseline + default graph policy

User clarified that AOTriton is mandatory for gfx1100 Qwen3.5/PARO, not an optional
fetch-only optimization.  Implemented the baseline packaging/default flip without touching
GPU sweeps or unrelated benchmark artifacts in the shared worktree.

Changes:

- Initialized Git LFS locally (`git lfs install --local`) and added `.gitattributes` for
  vendored AOTriton `.so*` and `.aks2` payloads.
- Pruned and vendored AOTriton `0.11.2b` under
  `hipengine/kernels/hip_gfx1100/attention/aotriton_runtime/0.11.2b/`: runtime headers,
  `libaotriton_v2.so.0.11.2`, symlink `libaotriton_v2.so`, and the 12 BF16 head-dim-256
  gfx11xx forward-attention images required by Qwen3.5/PARO.  `MANIFEST.vendor.json`
  records SHA256s/counts.
- Added `scripts/vendor_aotriton.sh` as the reproducible refresh path: fetch/cache via the
  pinned manifest, copy only `[aotriton.vendor]` images, and regenerate the vendor manifest.
- Changed AOTriton discovery to prefer the vendored package tree by default, keep explicit
  env/cache overrides for developers, and reject unpulled Git LFS pointer files with a
  clear `git lfs pull` hint.
- Promoted `PrefillConfig.attn_aotriton_min_tokens=512`, benchmark/fixture default
  `--attn-aotriton-min-tokens 512`, and `scripts/qwen35_paro_bench.py` decode graph replay
  default (`--no-graph-replay-decode` remains the eager diagnostic override).
- Updated `hipengine.generation.qwen35_paro` so the public `LLM.generate()` Qwen3.5/PARO path
  uses `Qwen35ParoResidentSession.capture_decode_graph(..., record_steps=...)` for generated
  tokens after native prefill.
- Updated README, `docs/PREFILL.md`, `docs/PLAN.md`, `docs/OPTIMIZE.md`, benchmark rollup text,
  changelog policy note, and package build artifacts so the vendored runtime is documented as
  baseline package data.

Validation (CPU/no-GPU only because GPU is in use by sweeps):

```bash
scripts/vendor_aotriton.sh --skip-fetch --force
# Vendored AOTriton 0.11.2b ... Images: 12

git check-attr filter diff merge text -- \
  hipengine/kernels/hip_gfx1100/attention/aotriton_runtime/0.11.2b/lib/libaotriton_v2.so.0.11.2 \
  hipengine/kernels/hip_gfx1100/attention/aotriton_runtime/0.11.2b/lib/aotriton.images/amd-gfx11xx/flash/attn_fwd/FONLY__＊bf16@16_256_F_F_0_0___gfx11xx.aks2
# both report filter/diff/merge=lfs

python3 -m pytest tests/test_aotriton_discovery.py tests/test_generation_qwen35_paro.py tests/test_qwen35_resident_batch_layout.py -q
# 39 passed

python3 scripts/qwen35_paro_bench.py --help
# exposes --graph-replay-decode / --no-graph-replay-decode and --attn-aotriton-min-tokens

scripts/vendor_aotriton.sh --help
# prints refresh usage

git diff --check
# clean
```

Unrelated untracked `benchmarks/results/2026-05-17-llamacpp-hip-qwen36-peak.json` was left
unstaged/untouched.

## 2026-05-17 — Retain llama.cpp HIP/Vulkan peak VRAM in the comparison tables

Ran the new `scripts/llamacpp_bench_with_peak.py` against the canonical `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
on both backends, `-r 1`, `--poll 10` ms, four canonical workloads (512/128, 4K/128, 32K/128,
128K/128), splitting each into prefill (`-p P -n 0 -d 0`) and decode-at-offset (`-p 0 -n 128 -d P`)
llama-bench invocations. Measured peak VRAM per row via
`/sys/class/drm/card1/device/mem_info_vram_used` sampled in a background thread.

Throughput from these `-r 1` instrumented runs is **not** retained: the polling overhead and
single-shot timing make the tok/s noisy relative to the historical PLAN-LONGCONTEXT split-row
numbers. Throughput rows in the comparison stay as they were; only the previously-null `peak_gib`
fields are populated.

Results (peak GiB, this host, W7900):

| Workload | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: |
| 512/128 | 21.125 | 20.844 |
| 4K/128 | 21.197 | 20.969 |
| 32K/128 | 21.738 | 21.533 |
| 128K/128 | 23.605 | 23.596 |

Resulting deltas vs hipEngine tracked peak (negative = hipEngine wins, positive = llama.cpp
lower):

- vs HIP: 512 `-2.54`, 4K `-1.32`, 32K `-1.05`, 128K `+0.05` GiB.
- vs Vulkan: 512 `-2.26`, 4K `-1.09`, 32K `-0.85`, 128K `+0.06` GiB.

hipEngine wins memory at all four contexts vs HIP and Vulkan, with the 128K rows now effectively
tied within run noise (≈50 MiB).

Changes:

- `scripts/qwen35_compare_tables.py`: filled the `peak_gib` field of every llama.cpp HIP/Vulkan
  row from the new artifacts; trimmed the notes to one short sentence pointing at the source
  JSON; changed the default `baseline` argument from `nano-vllm-amd` to `all` so a bare
  `python3 scripts/qwen35_compare_tables.py` now prints all three tables in one shot.
- `benchmarks/results/2026-05-16-hipengine-qwen35-comparison-tables-diagnostic.json`: same
  minimal change — only the `baselines["llama.cpp-*"]["rows"][*].peak_gib` and one note
  reference were edited.
- `benchmarks/results/2026-05-17-llamacpp-hip-qwen36-peak.json` and
  `benchmarks/results/2026-05-17-llamacpp-vulkan-qwen36-peak.json`: retained the sweep
  artifacts as the canonical source for the peak rows.
- `benchmarks/README.md`: added the llama.cpp HIP/Vulkan rows under “External comparison
  baselines” with the new peak numbers and a footnote pointing at the instrumentation, and
  bumped `Last updated` to 2026-05-17.
- `benchmarks/CHANGELOG.md`: dated one-liners for each baseline `peak_gib` update plus the
  comparison-table artifact / default-arg change.

Validation:

```bash
python3 -m json.tool benchmarks/results/2026-05-17-llamacpp-hip-qwen36-peak.json >/dev/null
python3 -m json.tool benchmarks/results/2026-05-17-llamacpp-vulkan-qwen36-peak.json >/dev/null
python3 -m json.tool benchmarks/results/2026-05-16-hipengine-qwen35-comparison-tables-diagnostic.json >/dev/null
python3 -m py_compile scripts/qwen35_compare_tables.py scripts/llamacpp_bench_with_peak.py hipengine/util/amdgpu_vram.py
python3 scripts/qwen35_compare_tables.py >/tmp/task32-all.md
python3 scripts/qwen35_compare_tables.py llama.cpp-hip >/tmp/task32-hip.md
python3 scripts/qwen35_compare_tables.py llama.cpp-vulkan >/tmp/task32-vulkan.md
# all four memory tables now render with both columns populated; delta column shows GiB
# instead of '—'.
```

Unrelated untracked `scripts/strip_paro_safetensors.py` is owned by another agent per AGENTS.md
coordination rules and is left alone.

---

## 2026-05-17 — Packed PARO shared-expert format end-to-end (commits b46339c..a70929b + bench driver)

Single canonical artifact for hipEngine-consumable PARO checkpoints is now the **packed**
format. All three dense shared-expert projections ship the same six-tensor PARO
suite (`qweight/qzeros/scales/theta/pairs/channel_scales`) that the dense attention
projections already use; the duplicate fp16 `mlp.shared_expert.*.weight` fallback path is
gone from the loader and runtime.

Commits in order:

- `b46339c` — `scripts/strip_paro_safetensors.py`: packed-only mode. Drops `--mode` and
  the hipengine-compat branch. Always removes every `.weight` whose module prefix also
  has a `.qweight`, including `mlp.shared_expert.*` when paroquant has quantized it. Dry
  run against upstream `z-lab/Qwen3.5-35B-A3B-PARO` returns 0 duplicates because that
  checkpoint never packed the shared expert.
- `1eb9b42` — loader: `required_moe_c1_tensor_names` now requires the 18 packed
  shared-expert tensor names per layer (6 per projection × 3 projections); shape
  validation drops the legacy fp16 shared-expert entries. `prepare_qwen35_paro_moe_c1_host_tensors`
  emits only the three `shared_expert.{proj}.qweight_pack8_decode` transposed views;
  raw qweight/qzeros/scales/theta/pairs/channel_scales are loaded direct by the runtime
  materializer. `_quantize_w8a16_host` removed from the loader (runner keeps its own
  copy for LM head). Existing layout tests rewritten; 14/14 pass.
- `7cd2e17` — new `silu_mul_separate_out_{fp16,bf16}` kernel variant in
  `kernels/hip_gfx1100/fused/paro_silu.hip`. Takes two separate `[rows, features]`
  pointers and writes `silu(gate) * up` to a third buffer. Needed because the W4 PARO
  prefill kernel `awq_fusedw4_prefill_dual_fp16` writes gate and up to *separate*
  output pointers, while the existing `silu_mul_dual_out_kernel` expects a packed
  `[rows, 2*features]` layout. Registered for `bf16/fp16/w4_paro`; plan tests updated;
  GPU smoke against existing `silu_mul_dual_out_fp16` shows `max_abs_diff = 0.0` on
  synthetic `[3, 16]` fp16 data (bit-exact).
- `a70929b` — runtime: replaced `shared_expert_w8a16_{fp16,bf16}` and the fused
  `shared_expert_gate_up_silu_fp16` / `shared_expert_down_combine_residual_fp16` with
  `shared_expert_paro_w4_{fp16,bf16}`. Decode (`tokens=1`) uses the existing
  `gemv_awq_dual_pack8_transposed_*` (separate inputs, packed gate||up output) feeding
  `silu_mul_dual_out_*`. FP16 prefill (`tokens>1`) uses `awq_fusedw4_prefill_dual_fp16`
  (separate inputs, separate outputs) feeding the new `silu_mul_separate_out_fp16`. BF16
  prefill falls back to the same dual GEMV used for decode (no fused W4 prefill kernel
  exists for bf16; documented as known suboptimal for large BF16 prefill batches).
  `run_moe_grouped_compact_fp16` now uses the split shared-expert + combine pattern that
  the bf16 grouped path already used; the fused W8A16+sigmoid+combine kernel is gone.
  Scratch dataclasses gained `shared_gate_input` / `shared_up_input` / `shared_gate_out`
  / `shared_up_out` / `shared_down_input` fields. 34/34 decode-state tests pass; full
  suite still has 6 pre-existing failures (test_cpu_reference fixture-format drift,
  test_hip_runtime FakeRuntime attribute differences, test_llm_generate WeightIndex.model_path
  attribute) — all present on main before this work.

Discovered while planning, recorded so it's not lost again:

- **Paroquant treats the shared expert as three independent `nn.Linear` modules** (it
  goes through `get_named_linears` → `_quantize_layer`), not as a fused `gate_up_proj`
  like routed experts. Consequence: gate/up have *separate* rotation parameters
  (`shared_expert.gate_proj.{theta,pairs,channel_scales}` ≠ `shared_expert.up_proj.{...}`),
  so the routed-expert "single rotated input → dual GEMV" pattern does not apply. The
  dense shared expert needs two rotates and a dual GEMV that consumes two distinct
  inputs.
- **The existing `gemv_awq_dual_pack8_transposed_{fp16,bf16}` already supports separate
  inputs with a packed `[rows, out_packed_a + out_packed_b]` output** (template
  `<scalar_t, qweight_transposed=true, separate_inputs=true>`). Decode and small batches
  use this directly into `scratch.shared_up` followed by `silu_mul_dual_out_*`. Only the
  fused-W4 prefill path needed the new separate-output silu_mul.

Limitations and gaps:

- **No packed-shared-expert checkpoint exists yet.** The upstream
  `z-lab/Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd`
  ships only fp16 `mlp.shared_expert.{gate,up,down}_proj.weight` for the shared expert
  (`shared_expert_intermediate_size=512`, `hidden_size=2048` per the config); none of
  the 18 packed PARO tensors per layer are present. Probing it with
  `validate_qwen35_paro_linear_attention_moe_c1_layout(layer_id=0)` reports
  `Missing count: 18`, all of them in the `mlp.shared_expert.*` family.
- **Correctness gate not yet run on real data.** All validation so far is unit-level
  (call-ordering tests with monkeypatched kernels, layout fixtures, plan tests, one
  bit-exact GPU smoke for `silu_mul_separate_out_fp16`). The AGENTS.md gate
  (KL ≤ 0.05, top-1 ≥ 90% vs `kernels/cpu_reference/`) for the dense shared-expert
  PARO chain has *not* been added; it is the next thing to land once a packed checkpoint
  is available.
- **No perf comparison run.** Decode-tps / prefill-tps / peak-VRAM all unmeasured for
  the new path. `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and `benchmarks/results/`
  intentionally not updated by this work — there is no retained measurement to record.

To unblock the gate + benchmark:

1. Regenerate paroquant outputs against the base Qwen3.5 MoE *without* `mlp.shared_expert`
   in `--skipped-modules` (the script `~/amd-gpu-tuning/paroquant/experiments/optimize/4bit_moe.sh`
   already excludes only `mlp.gate`, `mlp.shared_expert_gate`, and the linear-attn input
   projections — i.e. it already permits packing the shared expert; just re-run it).
2. `python -m paroquant.cli.convert --mode real ...` to merge the resulting per-module
   `.pt` files into a packed safetensors checkpoint.
3. `python scripts/strip_paro_safetensors.py --input-dir <packed> --output-dir <stripped>`
   to drop the now-duplicate fp16 fallbacks.
4. Add a CPU-reference fixture for the dense shared-expert W4 PARO chain
   (rotate → gemv_awq → silu*mul → rotate → gemv_awq), run the correctness gate.
5. Run `python scripts/qwen35_paro_packed_bench.py --checkpoint packed=<stripped> --baseline packed`
   for a single-row baseline; once a second variant exists (e.g. a hypothetical fp16
   shared-expert reconstruction, or a different paroquant configuration), the same
   driver does the comparison.

New tool committed alongside this WORKLOG entry: `scripts/qwen35_paro_packed_bench.py`,
a thin wrapper around the existing `qwen35_paro_bench.py` that validates packed-shared-expert
presence up-front, runs the bench across N checkpoints with identical settings, and
emits a side-by-side markdown + JSON comparison. `--how-to-pack` prints the paroquant
runbook. Smoke-tested by pointing it at the upstream z-lab checkpoint: it correctly
flags `missing packed shared-expert tensors` for all 18 names and exits without
launching the bench.

### Addendum 2026-05-17 — explicit BPW and on-device memory delta for the packed shared expert

The body of the section above describes the structural change but leaves the
memory math implicit. Recording it here so it's not lost:

**Disk BPW (Qwen3.5-35B-A3B):** the upstream `z-lab/Qwen3.5-35B-A3B-PARO`
artifact lands at roughly `4.73 BPW` against a 35B-parameter denominator (the
`scripts/strip_paro_safetensors.py --dry-run` against `dca2736` reports
`estimated_output_bpw_from_tensor_bytes = 4.7184`). Packing the shared expert
to the W4 PARO format saves the per-layer fp16 fallback weights:

- per-layer fp16 fallback bytes (z-lab snapshot, layer 0 shapes):
  `shared_expert.gate_proj.weight = [512, 2048] fp16 = 2 MiB`,
  `shared_expert.up_proj.weight   = [512, 2048] fp16 = 2 MiB`,
  `shared_expert.down_proj.weight = [2048, 512] fp16 = 2 MiB`  → **6 MiB/layer × 48 layers ≈ 288 MiB ≈ 0.281 GiB**
- W4 + PARO sidecars added per shared-expert projection (group_size=128, hidden=2048, shared_int=512):
  `qweight [K, N/8] int32` (2 MiB / 8 → ~256 KiB per proj, plus a `qweight_pack8_decode` transposed view of the same size)
  `qzeros [K/128, N/8] int32` (~512 B), `scales [K/128, N] fp16` (~16 KiB)
  `theta [krot, K/128] fp16`, `pairs [krot, K] int16`, `channel_scales [K] fp16` (~few KiB each)
  → packed shared expert footprint is roughly **20–25 % of the fp16 fallback** per projection on disk.
- Net BPW delta vs the legacy `4.73` baseline (already without duplicate
  fallbacks for the routed experts): the planning estimate documented in
  the original conversation that motivated this work was **~0.058 BPW** (~0.234 GiB),
  matching the rough back-of-the-envelope above and what the
  `strip_paro_safetensors.py --mode packed` math implies once paroquant
  emits the packed shared expert. Concrete artifact-side numbers will land
  in `benchmarks/results/qwen35_paro_packed_compare/comparison.json`
  the moment a packed checkpoint exists.

**On-device memory delta (decode, per shared-expert layer):**

| Path                | bytes per projection K=hidden=2048, N=shared_int=512 |
|---------------------|------------------------------------------------------|
| W8A16-from-fp16 (legacy, fully hipEngine-side quantized) | int8 weight ≈ 1.0 MiB + fp32 per-row scale ≈ 2 KiB ≈ **1.00 MiB/proj** |
| W4 PARO + sidecars (this work)                            | qweight + transposed view ≈ 0.5 MiB + fp16 scales/theta/channel_scales/qzeros/pairs ≈ 30 KiB ≈ **0.53 MiB/proj** |

So the shared expert weight footprint on device is roughly **halved**
(W4+sidecars vs W8A16). That is consistent with the disk delta and is the
*only* memory claim made for this work; no per-token bandwidth, decode-tps,
or prefill-tps measurement has been taken yet, by design (see the gap section
above).

**Speed expectation, explicitly:** decode tps for the shared expert alone
should be at parity or slightly better than the W8A16 path because (a) the
input bandwidth per projection drops by roughly 2×, and (b) the routed-expert
prefix already absorbed the rotation cost. The kernel count goes from
W8A16's `linear + silu + linear` (3 launches) to the new `rotate + rotate +
dual_GEMV + silu_dual + rotate + single_GEMV` (6 launches) at decode — same
count after the dual-GEMV optimization replaces what would have been 7 if
the literal spec-style two-single-GEMVs path had been used. The two extra
launches vs W8A16 are the input rotations, which are cheap relative to the
GEMVs. Net is expected to be parity-or-better but unmeasured; **not promoted
to any benchmark table** until a packed checkpoint and a real run exist.

Per AGENTS.md "Evidence Policy", `benchmarks/README.md` and
`benchmarks/CHANGELOG.md` are intentionally *not* touched by this work.
`benchmarks/results/` already contains the comparison driver's output
directory schema (`scripts/qwen35_paro_packed_bench.py` will populate
`benchmarks/results/qwen35_paro_packed_compare/comparison.json` the first
time it runs successfully), but no artifact has been emitted yet because no
packed checkpoint exists to run against.

## 2026-05-17 — OPTIMIZE.md: cut process ceremony + already-landed rows

Trimmed `docs/OPTIMIZE.md` to remove three categories of clutter the prior session left in:

1. **Process ceremony gating real work.** Killed Lane M.1 (promote first `LLM.generate()`
   accepted row), M.2 (auto-refresh hook for `qwen35_compare_tables.py`), and M.5 (rerun
   llama.cpp Vulkan Q4_K_M locally to second-guess the published baseline). These were
   doc-process tasks, not measurement tasks. Dropped the "Until M.1-M.5 land, no row below
   can move to `accepted`" footer and the `parked, blocked-by: M.5` dependency on P2.2.
   Lane M is now just M.3 (rocprof + ROCTX) and M.4 (per-bucket Amdahl) — the actual
   profile capture we still need.

2. **Already-attempted rows still marked `pending`.**
   - P2.1 (AOTriton Q/gate + K prelude fusion) → `rejected (perf); accepted (memory cleanup)`
     per `benchmarks/results/2026-05-16-hipengine-qwen35-aotriton-cast-glue-diagnostic.json`
     and `…-aotriton-gate-rotate-diagnostic.json`.
   - P2.2 (AOTriton V3 compact-varlen ABI) → `accepted (ABI landed)` per
     `…-aotriton-v3-prefill-diagnostic.json`.
   - D5.4 (linear-attn A/B decode same-input fusion) → `accepted` (live as
     `dense_dual_gemv_out_fp16` in `hipengine/runtime/qwen35_paro.py:2795` on the
     `tokens == 1` decode path).
   - W.2 (`-mcumode` default flag) → `accepted` (already in
     `hipengine/core/build.py:47` default flags + WMMA `extra_flags`).

3. **Padding rows that triple-counted Do-Not-Chase guardrails.**
   - §6.2 D2.4–D2.8 ("informational" rows listing parent's rejected B1–B7, C1–C5, E1–E6,
     F1/F2 Marlin-K experiments). The paragraph above the table and §11 "Marlin-K B1-B7
     / C1-C5 / E1-E6 / F1/F2 inner-loop experiments" line already cover this. Killed
     5 rows.
   - §6.4 D4.1 (replay-only profile harness) folded into M.3 — same data product.
   - §6.4 D4.3 (do-not-revisit multi-step graph replay) moved to §11's Do-Not-Chase list;
     it was a guardrail row, not a candidate.
   - §6.5 D5.2 (lm-head fused argmax) parent already audited and rejected — pure
     "don't redo" filler.
   - §7 A.5 (AOTriton vendored runtime dep) — not a memory guardrail; duplicates P2.3.
   - §5.5 P5.1 (chunked long-context prefill) — landed work; converted to a one-paragraph
     "already landed" note rather than a full candidate row.

Also trimmed §1 ("performance_claim=false" / "not accepted public LLM.generate() throughput
rows" ceremony) and §2 ("First promote the measurement harness") to match the leaner Lane M.

Net: 152 → 142 table rows. Live punchlist now reads as 28 pending / 11 accepted /
5 parked / 12 deferred / 9 rejected. No new candidates added; nothing measurement-related
removed.

```bash
wc -l docs/OPTIMIZE.md
# 458
grep -c '^| ' docs/OPTIMIZE.md
# 142
```

Next concrete step on the doc punchlist is M.3 (rocprofv3 --kernel-trace + ROCTX) on
512/128, 4K/128, 32K/128 with the comparison-table flags, producing per-bucket Amdahl
tables to replace the parent-borrowed §6 decode block. No GPU runs in this commit.

## 2026-05-17 — Restore dual shared-expert format support for z-lab PARO

Root cause review: the packed-PARO shared-expert series accidentally made the new
packed sidecar representation exclusive. Commit `1eb9b42` replaced the original
z-lab `mlp.shared_expert.{gate_proj,up_proj,down_proj}.weight` requirements with
18 packed PARO tensors, and `a70929b` replaced the runtime shared-expert dispatch
with the W4 PARO path only. That broke the public/original
`z-lab/Qwen3.5-35B-A3B-PARO` checkpoint, which contains only:

- `layers.0.mlp.shared_expert.gate_proj.weight`
- `layers.0.mlp.shared_expert.up_proj.weight`
- `layers.0.mlp.shared_expert.down_proj.weight`
- `layers.0.mlp.shared_expert_gate.weight`

Fix in progress/landed in the working tree: loader validation now detects the
shared-expert family per layer. If all packed sidecars are present, it uses
`packed_paro_w4`; otherwise, if the three fp16 `.weight` tensors are present, it
uses `legacy_fp16`. Both required-name and runtime-prepared-name helpers accept a
`shared_expert_format` selector. The legacy path restores host-side W8A16 prep
(`gate_up_weight_w8a16{,_scale}`, `down_weight_w8a16{,_scale}`), while packed
checkpoints keep the W4 `qweight_pack8_decode` prep.

Runtime dispatch now checks the materialized tensor family: original z-lab
checkpoints use the restored W8A16 shared-expert methods, and packed checkpoints
use `shared_expert_paro_w4_{fp16,bf16}`. For the original fp16 grouped/prefill
path, the fused `w8a16_shared_gate_up_silu_fp16` +
`w8a16_shared_down_combine_residual_fp16` chain is restored, preserving the old
kernel sequence and avoiding a performance regression for the existing format.

CPU-only validation run while the GPU is reserved for profiling:

```bash
python3 -m pytest tests/test_qwen35_paro_layout.py tests/test_qwen35_decode_state.py -q --tb=short
# 55 passed

python3 - <<'PY'
from pathlib import Path
from hipengine.loading import load_weight_index
from hipengine.loading.qwen35_paro import validate_qwen35_paro_linear_attention_moe_c1_layout
p = Path('/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd')
idx = load_weight_index(p)
val = validate_qwen35_paro_linear_attention_moe_c1_layout(idx, layer_id=0)
print(val.passed, val.shared_expert_format, len(val.missing), len(val.shape_errors))
PY
# True legacy_fp16 0 0
```

No GPU benchmark or end-to-end model run was attempted per the current profiling
reservation. Next GPU step, once clear: run the original z-lab benchmark through
`scripts/qwen35_paro_bench.py` and compare against the pre-packed baseline to
confirm no legacy-format throughput regression; packed-format perf still requires
a real packed checkpoint artifact.

## 2026-05-17 — Qwen3.5/PARO rocprof Amdahl baseline (M.3/M.4)

Ran the OPTIMIZE.md Lane M profiling baseline for Qwen3.5-35B-A3B-PARO `w4_paro` on W7900/gfx1100.

Important profiler gotchas hit and resolved:

- A first tiny rocprof smoke without `--compiler-version-file` / `--require-cached-build` could hang
  with GPU at 0% because a profiled Python process may spawn `hipcc`/clang. Verified cache-only smoke
  works:
  ```bash
  rocprofv3 --kernel-trace --marker-trace --output-format csv -d /tmp/rocprof-probe -o probe -- \
    python3 scripts/smoke.py --mode qwen35-rmsnorm-hip --rows 2 --hidden-size 16 \
      --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
  # produced probe_kernel_trace.csv in <1s
  ```
- Plain `/opt/rocm/lib/libroctx64.so` markers did not produce marker trace CSVs under this
  rocprofv3. A temporary `LD_LIBRARY_PATH` override that symlinks `libroctx64.so` to
  `librocprofiler-sdk-roctx.so.1` does produce marker traces.
- Full marker tracing around HIP graph replay and full selected-region tracing of 64/128 graph
  replays both hit a rocprofiler-sdk finalization assert:
  `retired dangling correlation IDs`. A 16-replay selected-region decode graph trace is stable,
  so the decode Amdahl uses 16 one-step graph replays and scales per token. The throughput
  scoreboard in `docs/OPTIMIZE.md` remains the true 128-token comparison-table run.

Added profiling support:

- `scripts/qwen35_paro_bench.py --rocprof-selected-region {prefill,measured_decode_graph,measured_decode}`
  wraps the chosen phase with `roctxProfilerResume/Pause` for `rocprofv3 --selected-regions`.
  This is profiler-only and does not change benchmark semantics.
- New `scripts/qwen35_rocprof_audit.py` runs selected-region rocprof for prefill and measured
  decode graph, forces the SDK ROCTX library via `/tmp/hipengine-roctx-sdk-override/libroctx64.so`,
  parses kernel trace CSVs into family/top-kernel Amdahl summaries, and writes a compact artifact.

Command:

```bash
python3 scripts/qwen35_rocprof_audit.py \
  --workloads 512/128 4096/128 32768/128 \
  --out benchmarks/results/2026-05-17-hipengine-qwen35-rocprof-amdahl-diagnostic.json
```

Raw traces/logs stay under `/tmp/hipengine-rocprof-qwen35-audit/`. Committed compact artifact:
`benchmarks/results/2026-05-17-hipengine-qwen35-rocprof-amdahl-diagnostic.json`.

Measured selected-region totals:

| Workload | Prefill kernel ms | Prefill kernel/host | Decode profile kernel ms | Decode host s (16 replays) | Dispatches/token | Kernel ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 196.9 | 82.4% | 116.2 | 0.1688 | 877 | 7.27 |
| 4K/128 | 1576.4 | 96.0% | 115.6 | 0.1681 | 877 | 7.23 |
| 32K/128 | 16973.3 | 98.1% | 137.6 | 0.1906 | 877 | 8.60 |

Prefill top families:

| Bucket | 512 | 4K | 32K |
| --- | ---: | ---: | ---: |
| MoE selected compact WMMA | 26.2% | 21.5% | 15.6% |
| Linear-attention GDN prefill | 20.5% | 21.1% | 15.7% |
| W4 prefill GEMM | 17.9% | 17.1% | 12.7% |
| Shared-expert W8A16 | 15.3% | 15.5% | 11.6% |
| AOTriton prefill attention | 1.9% | 5.7% | 30.2% |

Decode top families:

| Bucket | 512 | 4K | 32K | Calls/token |
| --- | ---: | ---: | ---: | ---: |
| Selected-MoE W4 GEMV | 17.9% | 18.3% | 15.5% | 80 |
| W8A16 linear/lm-head/dense | 15.7% | 15.7% | 13.4% | 81 |
| W4 single pack8 GEMV | 13.4% | 13.6% | 11.6% | 50 |
| W4 dual pack8 GEMV | 11.8% | 11.7% | 10.0% | 40 |
| Decode attention | 11.4% | 10.5% | 22.9% | 10-20 |
| Rotation/RoPE | 9.4% | 9.6% | 8.4% | 160 |

Updated `docs/OPTIMIZE.md`:

- M.3 and M.4 are `accepted` with the artifact path and rocprof caveat.
- Replaced the old hand-narrated/parent-borrowed §5/§6 Amdahl blocks with measured hipEngine tables.
- Added data-backed D5.2: audit W8A16 decode kernels (`w8a16_linear_kernel`,
  `w8a16_linear_lowp_out_kernel`) for tile/occupancy headroom, explicitly **not** fused argmax.
- Reordered §12 punchlist from measured buckets: P1 bulk dense/threshold first, W.1, D2.1+D5.2,
  then D1 fusion and D3 attention.

Validation:

```bash
python3 -m py_compile scripts/qwen35_paro_bench.py scripts/qwen35_rocprof_audit.py
python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-rocprof-amdahl-diagnostic.json >/tmp/rocprof-artifact.valid
python3 scripts/qwen35_paro_bench.py --help | grep -A5 rocprof-selected-region
python3 scripts/qwen35_rocprof_audit.py --dry-run --workloads 512/128 >/tmp/qwen35_rocprof_audit.dryrun
```

Next optimization step from the measured profile: P1.1/P1.2/P1.3/P1.4 (bulk dense rocBLAS for
linear-attn A/B and shared-expert prefill paths) plus W.1 as the cheap flag probe.

## 2026-05-17 — Fix packed shared-expert runtime sidecar materialization

While starting the shisa legacy/packed benchmark comparison, the first shisa run
failed during native prefill with:

```text
KeyError: 'layers.0.mlp.shared_expert.gate_proj.pairs'
```

Root cause: the dual-format loader correctly detected the shisa checkpoint as
`packed_paro_w4`, and prepared `shared_expert.{proj}.qweight_pack8_decode`, but
`runtime_{full,linear}_attention_moe_c1_tensor_names()` did not include the
packed shared-expert sidecars consumed directly by `shared_expert_paro_w4_*`:
`qzeros`, `scales`, `theta`, `pairs`, and `channel_scales`. The tests' manual
runtime weight fixture had those tensors, so the materialized-runtime path was
not covered.

Fix: packed runtime tensor-name helpers now include those sidecars while still
omitting raw `qweight` (only used at load time to build `qweight_pack8_decode`).
Legacy W8A16 format still includes no packed sidecars.

CPU validation:

```bash
python3 -m pytest tests/test_qwen35_paro_layout.py tests/test_qwen35_decode_state.py -q --tb=short
# 55 passed
```

The failed shisa benchmark will be rerun after this fix. GPU timing before the
failure is discarded; no throughput result was produced.

## 2026-05-17 — Dual-format PARO benchmark results: z-lab legacy + shisa packed/unstripped

Ran the restored dual shared-expert loader/runtime through the resident benchmark
on W7900/gfx1100 using the same diagnostic protocol as prior Qwen3.5/PARO rows:

```bash
python3 scripts/qwen35_paro_bench.py \
  --model <checkpoint> \
  --token-id 9707 \
  --prompt-length {512,4096} \
  --decode-tokens 128 \
  --warmup-decode-tokens 1 \
  --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-paro-dual-bench-20260517/<label>.json
```

Checkpoint format detection:

- `z-lab/Qwen3.5-35B-A3B-PARO` (`dca2736`) -> true `legacy_fp16`; only shared-expert fp16 `.weight` tensors are present.
- `shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5` (`1492d9a`) -> unstripped: both fp16 fallback `.weight` tensors and packed sidecars are present; loader prefers packed sidecars.
- `shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed` (`501ef86`) -> stripped packed sidecars only.

Median timings (two runs for 512 rows and shisa 4K rows; one run for z-lab 4K):

| checkpoint | format used | workload | prefill tok/s | decode tok/s | tracked peak GiB | sampled HIP peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| z-lab Qwen3.5 PARO | legacy_fp16/W8A16 shared expert | 512/128 | 2185.814 | 109.288 | 18.587 | 18.604 |
| z-lab Qwen3.5 PARO | legacy_fp16/W8A16 shared expert | 4096/128 | 2371.502 | 110.364 | 20.458 | 19.013 |
| shisa Qwen3.6 unstripped | packed_paro_w4 sidecars | 512/128 | 2417.232 | 101.547 | 18.535 | 18.551 |
| shisa Qwen3.6 stripped-packed | packed_paro_w4 sidecars | 512/128 | 2418.130 | 101.634 | 18.535 | 18.551 |
| shisa Qwen3.6 unstripped | packed_paro_w4 sidecars | 4096/128 | 2655.734 | 102.795 | 20.406 | 18.959 |
| shisa Qwen3.6 stripped-packed | packed_paro_w4 sidecars | 4096/128 | 2653.591 | 103.057 | 20.406 | 18.959 |

Takeaways:

1. The original z-lab PARO model runs again on the restored legacy path. Decode is
   effectively unchanged vs the previous graph-replay diagnostic row (`109.340 ->
   109.288 tok/s`, -0.05% at 512; `110.303 -> 110.364`, +0.06% at 4K). The 512
   prefill median is lower than that older single row (`2312.754 -> 2185.814`,
   -5.5%); the current comparison-table/no-op-chunk row was `2216.487`, so this
   is closer to -1.4% vs the recent table baseline. No correctness gate rerun was
   done in this benchmark-only pass.
2. shisa unstripped vs stripped-packed has identical on-device peaks and timing is
   within run noise because both variants use packed sidecars at runtime. The
   stripped checkpoint mainly helps disk/package size: safetensors size
   `21.686 GiB -> 19.068 GiB` (-12.1%).
3. Packed shared-expert sidecar materialization is now covered by layout tests;
   the first shisa run exposed that missing materialization and was discarded.

Committed compact artifact:
`benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-paro-dual-format-diagnostic.json`.
Rollup/changelog updated as diagnostic retained, `performance_claim=false`.

## 2026-05-17 — shisa unstripped forced legacy shared-expert diagnostic

Added a diagnostic-only `--shared-expert-format {auto,legacy_fp16,packed_paro_w4}`
override to `scripts/qwen35_paro_bench.py` and threaded it through validation and
runtime materialization. Default remains `auto`; checkpoints with both formats
still prefer `packed_paro_w4`. CPU-only validation before/after the GPU run:

```bash
python3 -m py_compile hipengine/loading/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py scripts/qwen35_paro_bench.py
python3 -m pytest tests/test_qwen35_paro_layout.py tests/test_qwen35_decode_state.py -q --tb=short
# 57 passed
```

Benchmark protocol on W7900/gfx1100, same checkpoint for every row:
`/models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5/snapshots/1492d9ae108682763e67b28ff4aad660d7e19cd4`.

```bash
python3 scripts/qwen35_paro_bench.py \
  --model <shisa-unstripped> \
  --shared-expert-format {packed_paro_w4,legacy_fp16} \
  --token-id 9707 \
  --prompt-length {512,4096} \
  --decode-tokens 128 \
  --warmup-decode-tokens 1 \
  --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-shisa-force-legacy-20260517/<label>.json
```

Results (single run per workload/format, no shisa KL/top-1 gate, diagnostic only):

| workload | packed prefill | forced legacy prefill | delta | packed decode | forced legacy decode | delta | tracked peak packed -> legacy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 2425.637 | 2196.276 | -9.46% | 101.717 | 109.230 | +7.39% | 18.535 -> 18.587 GiB |
| 4096/128 | 2653.477 | 2359.452 | -11.08% | 103.041 | 110.412 | +7.15% | 20.406 -> 20.458 GiB |

Conclusion: yes, on the shisa unstripped checkpoint the forced legacy shared
expert path has lower prefill but higher decode than the packed shared-expert
sidecar path. Artifact:
`benchmarks/results/2026-05-17-hipengine-qwen36-shisa-force-legacy-diagnostic.json`.

## 2026-05-17 — W.1 unroll-600 ablation across legacy and packed PARO

User asked to run the W.1 compiler flag probe and include both shared-expert
formats. Current build profiles already include `-mllvm
-amdgpu-unroll-threshold-local=600`, so this was an ablation rather than an
enablement. Added an env-only diagnostic knob in `hipengine/core/build.py`:
`HIPENGINE_DISABLE_UNROLL600=1` strips only the `-mllvm`/unroll-600 pair while
preserving other profile flags such as `-mcumode`.

Model paths used:

- legacy local path: `/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd` (`legacy_fp16`). The requested `z-lab/Qwen3.6-35B-A3B-PARO` 35B snapshot is not present locally.
- packed path: `/models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e` (`packed_paro_w4`).

Benchmark protocol (W7900/gfx1100):

```bash
python3 scripts/qwen35_paro_bench.py \
  --model <model> \
  --token-id 9707 \
  --prompt-length {512,4096} \
  --decode-tokens 128 \
  --warmup-decode-tokens 1 \
  --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-w1-unroll-20260517/<label>.json
# no-unroll rows add env: HIPENGINE_DISABLE_UNROLL600=1
```

No-unroll `.so` files were prebuilt outside the timed runs, then every measured
run used `--require-cached-build`. Two runs/profile/workload were captured;
results below are medians.

| model | workload | default prefill | no-unroll prefill | no-unroll Δ | default decode | no-unroll decode | no-unroll Δ | peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| z-lab legacy | 512/128 | 2179.915 | 2185.043 | +0.24% | 109.060 | 109.143 | +0.08% | 18.587 |
| z-lab legacy | 4096/128 | 2364.642 | 2361.915 | -0.12% | 110.212 | 110.227 | +0.01% | 20.458 |
| shisa packed | 512/128 | 2400.079 | 2405.569 | +0.23% | 101.638 | 101.609 | -0.03% | 18.535 |
| shisa packed | 4096/128 | 2653.642 | 2648.661 | -0.19% | 102.869 | 102.663 | -0.20% | 20.406 |

Generated preview sanity matched exactly for each model/workload/profile (first
two token IDs `9707, 9707`, logits equal to 6 decimals). Resource metadata audit
via `.hip_fatbin` extraction + `llvm-readobj --notes` on hot libraries
(`linear_gdn`, `moe_awq_wmma`, `paro_awq_gemv`, `w8a16_linear`, `kv_write`) found
`private_segment_fixed_size=0`, SGPR/VGPR spill counts 0, and identical max VGPR
between default and no-unroll.

Validation:

```bash
python3 -m py_compile hipengine/core/build.py scripts/qwen35_paro_bench.py
python3 -m pytest tests/test_build.py tests/test_qwen35_paro_layout.py tests/test_qwen35_decode_state.py -q --tb=short
# 63 passed
```

Conclusion: W.1 is neutral/noisy across both local legacy and packed paths. Keep
the current default unroll-600 flag, but remove it from the active optimization
queue. Artifact:
`benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-w1-unroll600-ablation-diagnostic.json`.

## 2026-05-17 — KV cache roadmap: INT8 no-shadow then FastDMS-derived DMS

User asked for a focused KV-cache plan covering dense INT8 KV first and a
FastDMS-derived compact-DMS variant second. Added `docs/KVCACHE.md` as the
source plan for this lane.

Key decisions recorded:

- Dense paged INT8 KV is the immediate 256K capacity path. For Qwen3.5/PARO,
  BF16 KV is ~20 KiB/token across the 10 full-attention layers, so 256K INT8
  has roughly the same raw KV footprint as 128K BF16. This only counts if there
  is no persistent BF16 shadow/staging arena.
- INT8 KV is treated as a capacity feature first, not a guaranteed speed win;
  parent evidence remains neutral/negative at 32K and marginal at 128K.
- Compact DMS follows after INT8 and should reuse the `KVLiveSpans` ABI plus
  storage-dtype plumbing. The implementation map points at `~/FastDMS` files
  for DMS metadata extraction, compact allocator/admission, streaming prefill
  pack, fused decode preprocess, and compact grouped split-K attention.
- DMS quality claims require a DMS-retrofitted checkpoint; no silent DMS on an
  arbitrary checkpoint. HIGGS/AQUA remain research-only because FastDMS serving
  speed did not justify HIGGS as a current RDNA3 target.

Updated `docs/PLAN.md` and `docs/OPTIMIZE.md` with links/guardrails pointing to
`docs/KVCACHE.md`.

Validation:

```bash
python3 - <<'PY'
from pathlib import Path
for p in ['docs/KVCACHE.md','docs/PLAN.md','docs/OPTIMIZE.md']:
    text=Path(p).read_text()
    assert '\\t' not in text
    assert 'TODO' not in text
print('markdown sanity ok')
PY
# markdown sanity ok
```

## 2026-05-17 packed shared-expert W4 decode recovery

Worked on the true packed-sidecar recovery path for the shisa Qwen3.6
shared-expert decode gap. Profiling the packed d4 decode tail showed the shared
path still paid separate gate rotate, up rotate, SiLU, down rotate, and W4 GEMV
launches per MoE layer. Implemented the low-risk packed-W4 fusions already in the
kernel catalog:

- use `paro_rotate2_{fp16,bf16}` for shared `gate_proj` + `up_proj` input
  rotations when their `krot` matches (fallback remains two `rotate1` calls),
- use `silu_mul_dual_rotate_out_{fp16,bf16}` for packed shared c=1 SiLU +
  down-input rotation.

This keeps packed sidecars as the runtime format and does not materialize a
W8A16 shadow.

Validation:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro.py
python3 -m pytest tests/test_qwen35_paro_layout.py tests/test_qwen35_decode_state.py -q --tb=short
# 57 passed
python3 scripts/smoke.py --mode paro-silu-hip --rows 2 --hidden-size 128 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
# dual/dual_rotate/pair_rotate BF16+FP16 mismatches 0
python3 scripts/smoke.py --mode paro-rotate-hip --rows 2 --hidden-size 128 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
# BF16/FP16 rotate mismatches 0
```

Benchmark command shape (two runs per workload; W7900/gfx1100, cached builds,
AOTriton threshold 512):

```bash
python3 scripts/qwen35_paro_bench.py \
  --model /models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5/snapshots/1492d9ae108682763e67b28ff4aad660d7e19cd4 \
  --shared-expert-format packed_paro_w4 \
  --token-id 9707 \
  --prompt-length {512,4096} \
  --decode-tokens 128 \
  --warmup-decode-tokens 1 \
  --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build \
  --attn-aotriton-min-tokens 512
```

Two-run medians vs the previous shisa packed/forced-legacy diagnostic:

| workload | packed before | packed after | packed delta | forced legacy | remaining packed gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 decode | 101.717 tok/s | 105.636 tok/s | +3.9% | 109.230 tok/s | -3.3% |
| 4096/128 decode | 103.041 tok/s | 106.777 tok/s | +3.6% | 110.412 tok/s | -3.3% |
| 512/128 prefill | 2425.637 tok/s | 2451.213 tok/s | +1.1% | 2196.276 tok/s | n/a |
| 4096/128 prefill | 2653.477 tok/s | 2666.743 tok/s | +0.5% | 2359.452 tok/s | n/a |

Tracked peak stayed unchanged at 18.535 GiB (512) and 20.406 GiB (4K). This
recovers about half of the decode gap while keeping packed sidecars. Remaining
likely levers are deeper packed down-W4+combine fusion or precomputed rotation
sin/cos; still diagnostic only until shisa KL/top-1 gate exists.

Artifact:
`benchmarks/results/2026-05-17-hipengine-qwen36-packed-shared-decode-fusion-diagnostic.json`.

## 2026-05-17 stale unit-test expectations review

Reviewed the current non-passing full unit suite after the packed shared-expert
commit. Failures were stale tests, not a new runtime regression:

- `tests/test_cpu_reference.py::test_cpu_reference_full_attn_prefill_causal_gqa_gate`
  still expected FP32 attention values, but `fix: align AOTriton prefill with
  BF16 SDPA` intentionally changed the CPU reference to BF16-round attention
  before applying the sigmoid gate. Updated the inline expected values and the
  committed `full_attn_prefill_causal_gqa_gate.json` fixture.
- `tests/test_hip_runtime.py` fake HIP library missed `hipMemset`,
  `hipMemsetAsync`, and `hipMemGetInfo`, which are now configured by
  `HipRuntime._configure()`.
- `tests/test_llm_generate.py` fake weight indices lacked `model_path`, now
  required because `LLM._load_model_metadata()` normalizes HF IDs to resolved
  filesystem paths.

Validation:

```bash
python3 -m pytest -q --tb=short
# 254 passed
```

## 2026-05-17 — P1.1 rocBLAS A/B prefill prototype rejected

Task #4 tested the highest-impact prefill hypothesis P1.1: replacing the
multi-token linear-attention A/B row-GEMV pair in
`project_linear_attention_ab_fp16(...)` with torch-free rocBLAS bulk GEMM. Added
an env-off diagnostic path:

- `hipengine/core/rocblas.py` lazy-loads `librocblas.so` and wraps row-major NT
  FP16 GEMM as `rocblas_gemm_ex` with FP32 accumulation.
- `HIPENGINE_LINEAR_AB_PREFILL_ROCBLAS_MIN_TOKENS=2` routes only `tokens > 1`
  A/B prefill projections through rocBLAS. The `tokens == 1` decode path remains
  `dense_dual_gemv_out_fp16`.

Microcheck:

```bash
# 5x7 @ 3x7 row-major NT rocblas_gemm_ex FP16/F32-accum
# max_abs=0 vs NumPy FP32 reference rounded to FP16
```

Validation:

```bash
python3 -m py_compile hipengine/core/rocblas.py hipengine/runtime/qwen35_paro.py scripts/qwen35_paro_bench.py
python3 -m pytest tests/test_qwen35_decode_state.py -q --tb=short
# 38 passed
```

Benchmark protocol on W7900/gfx1100, all rows used cache-only HIP builds:

```bash
python3 scripts/qwen35_paro_bench.py \
  --model <zlab-legacy-or-shisa-packed> \
  --token-id 9707 \
  --prompt-length {512,4096} \
  --decode-tokens 128 \
  --warmup-decode-tokens 1 \
  --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p11-rocblas-ab-20260517/<label>.json
# rocBLAS rows add env: HIPENGINE_LINEAR_AB_PREFILL_ROCBLAS_MIN_TOKENS=2
```

Primary `rocblas_gemm_ex` FP32-accum results (single run per row; diagnostic
only, no KL/top-1 gate because performance regressed):

| model | workload | default prefill | rocBLAS prefill | Δ | default decode | rocBLAS decode | Δ | peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| z-lab legacy | 512/128 | 2227.421 | 930.273 | -58.24% | 109.354 | 109.033 | -0.29% | 18.587 |
| z-lab legacy | 4096/128 | 2391.477 | 1996.481 | -16.52% | 110.678 | 110.450 | -0.21% | 20.458 |
| shisa packed | 512/128 | 2468.194 | 941.780 | -61.84% | 105.717 | 105.606 | -0.10% | 18.535 |
| shisa packed | 4096/128 | 2677.951 | 2204.678 | -17.67% | 106.966 | 106.719 | -0.23% | 20.406 |

Generated token IDs stayed `9707` for the first two sampled outputs, but logits
differed from default due rocBLAS/Tensile reduction order. An earlier
`rocblas_hgemm` pilot also regressed similarly and was less numerically close;
the artifact keeps those rows as `pilot_hgemm_measurements`.

Conclusion: reject P1.1 rocBLAS A/B. The linear-attention A/B dense shape is
skinny (`out_features=32`), and on this W7900 stack the current custom row-GEMV
kernels beat rocBLAS/Tensile for both 512 and 4K. Keep the env-off rocBLAS bridge
only as a diagnostic/prototype surface for wider future shapes. Artifact:
`benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p11-rocblas-ab-rejected.json`.

## 2026-05-17 — P1.2 legacy W8A16 shared gate/up token tiling retained

Task #5 prototyped a torch-free bulk-ish prefill replacement for the legacy
W8A16 shared-expert gate/up+SiLU path without materializing fp16 shadow weights.
Instead of rocBLAS/dense fp16 weights, the retained path groups adjacent prompt
tokens inside the existing quantized dot-product kernel:

- Added `w8a16_shared_gate_up_silu_fp16_token_tiled_kernel<2/4>` in
  `hipengine/kernels/hip_gfx1100/quant/w8a16_linear.hip` plus ctypes wrappers.
- Runtime default is conservative: legacy W8A16 shared gate/up uses
  `token_tile=2` only when `tokens >= 1024`.
- Existing `w8a16_shared_gate_up_silu_fp16(...)` remains fallback for smaller
  prompts and opt-out (`HIPENGINE_SHARED_GATE_UP_PREFILL_TOKEN_TILE=0`).
- Shisa packed sidecars are intentionally unaffected; they continue to use the
  packed PARO W4 prefill kernels.

Microcheck:

```bash
# random small shape tokens=5 hidden=32 intermediate=8
# tile2 max_abs 0.0 vs original kernel
# tile4 max_abs 0.0 vs original kernel
```

Validation / correctness:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/quant/w8a16_linear.py hipengine/runtime/qwen35_paro.py
python3 -m pytest tests/test_qwen35_decode_state.py -q --tb=short
# 39 passed

HIPENGINE_SHARED_GATE_UP_PREFILL_MIN_TOKENS=2 \
python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json \
  --max-layers 40 \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p12-shared-gate-up-token-tile-20260517/p12_tile2_fixture_gate.json
# passed=true, max_kl=0.0395688706, top1_agreement=1.0, generated IDs match fixture
```

Profiler smoke (cache-only build, one layer, prompt 1024) confirmed the new
kernel launched:

```bash
rocprofv3 --kernel-trace --output-format csv \
  --output-file /tmp/hipengine-p12-shared-gate-up-token-tile-20260517/rocprof/p12_tile2 -- \
python3 scripts/qwen35_paro_bench.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --token-id 9707 --prompt-length 1024 --decode-tokens 1 --warmup-decode-tokens 0 \
  --max-layers 1 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p12-shared-gate-up-token-tile-20260517/rocprof/p12_tile2_bench.json
# trace includes void (anonymous namespace)::w8a16_shared_gate_up_silu_fp16_token_tiled_kernel<2>(...), DurationNs 737130
```

Benchmark protocol on W7900/gfx1100, all rows used cache-only HIP builds:

```bash
python3 scripts/qwen35_paro_bench.py \
  --model <zlab-legacy-or-shisa-packed> \
  --token-id 9707 \
  --prompt-length {512,4096} \
  --decode-tokens 128 \
  --warmup-decode-tokens 1 \
  --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p12-shared-gate-up-token-tile-20260517/<label>.json
# candidate rows set HIPENGINE_SHARED_GATE_UP_PREFILL_TOKEN_TILE=2 or 4
```

Two-run medians, `HIPENGINE_SHARED_GATE_UP_PREFILL_TOKEN_TILE=2` vs previous
no-token-tile default:

| model/path | workload | prefill Δ | decode Δ | peak GiB | generated/logit sanity |
| --- | ---: | ---: | ---: | ---: | --- |
| z-lab legacy W8A16 | 512/128 | +0.52% | -0.07% | 18.587 | first two generated IDs/logits match |
| z-lab legacy W8A16 | 4096/128 | +2.16% | +0.15% | 20.458 | first two generated IDs/logits match |
| shisa stripped packed | 512/128 | -0.10% | -0.06% | 18.535 | packed path unaffected/noise |
| shisa stripped packed | 4096/128 | -0.19% | -0.07% | 20.406 | packed path unaffected/noise |

Tile4 was rejected after a one-run probe: it regressed 512 and did not improve
4K. A final no-env post-change pass showed the default policy is wired; 512
continues to use the old fallback, while 4K uses tile2. Because this is still a
resident-runner diagnostic rather than a promoted public `LLM.generate()` row,
`performance_claim=false` in the artifact.

Artifact:
`benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p12-shared-gate-up-token-tile-diagnostic.json`.

## 2026-05-17 — P1.3 legacy W8A16 shared down+combine token tiling retained

Task #6 prototyped the shared-expert down projection plus fused combine/residual
prefill path. To preserve the W8A16 memory advantage and the existing fused tail,
the retained implementation token-tiles the quantized down+combine kernel rather
than materializing fp16 dense down weights or a separate shared-output scratch:

- Added `w8a16_shared_down_combine_residual_fp16_token_tiled_kernel<2/4>` in
  `hipengine/kernels/hip_gfx1100/quant/w8a16_linear.hip` plus ctypes wrappers.
- Runtime default: legacy W8A16 shared down+combine uses `token_tile=2` when
  `tokens >= 2`; decode `tokens == 1` stays untiled.
- Existing `w8a16_shared_down_combine_residual_fp16(...)` remains fallback and
  opt-out (`HIPENGINE_SHARED_DOWN_COMBINE_PREFILL_TOKEN_TILE=0`).
- The tail semantics are unchanged: precompute shared sigmoid in-place, compute
  lowp W8A16 shared down, then write `residual + selected + gate * shared`.
- Shisa packed sidecars are intentionally unaffected; they continue through W4
  shared expert plus `shared_gate_combine_residual_batch_out_fp16`.

Microcheck:

```bash
# random small shape tokens=5 hidden=17 intermediate=32 gate_stride=9
# tile2 max_abs 0.0 vs original fused down+combine kernel
# tile4 max_abs 0.0 vs original fused down+combine kernel
```

Validation / correctness:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/quant/w8a16_linear.py hipengine/runtime/qwen35_paro.py
python3 -m pytest tests/test_qwen35_decode_state.py -q --tb=short
# 40 passed

python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json \
  --max-layers 40 \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p13-shared-down-combine-token-tile-20260517/p13_down_tile2_fixture_gate.json
# passed=true, max_kl=0.0395688706, top1_agreement=1.0, generated IDs match fixture
```

Profiler smoke (cache-only build, one layer, prompt 512) confirmed the new kernel
launched:

```bash
rocprofv3 --kernel-trace --output-format csv \
  --output-file /tmp/hipengine-p13-shared-down-combine-token-tile-20260517/rocprof/p13_down_tile2 -- \
python3 scripts/qwen35_paro_bench.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --token-id 9707 --prompt-length 512 --decode-tokens 1 --warmup-decode-tokens 0 \
  --max-layers 1 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p13-shared-down-combine-token-tile-20260517/rocprof/p13_down_tile2_bench.json
# trace includes void (anonymous namespace)::w8a16_shared_down_combine_residual_fp16_token_tiled_kernel<2>(...), DurationNs 309524
```

Benchmark protocol on W7900/gfx1100, all rows used cache-only HIP builds:

```bash
python3 scripts/qwen35_paro_bench.py \
  --model <zlab-legacy-or-shisa-packed> \
  --token-id 9707 \
  --prompt-length {512,4096} \
  --decode-tokens 128 \
  --warmup-decode-tokens 1 \
  --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p13-shared-down-combine-token-tile-20260517/<label>.json
# candidate rows set HIPENGINE_SHARED_DOWN_COMBINE_PREFILL_TOKEN_TILE=2 or 4
```

Two-run medians, `HIPENGINE_SHARED_DOWN_COMBINE_PREFILL_TOKEN_TILE=2` vs the
previous no-down-token-tile default (which already included P1.2 gate/up tiling):

| model/path | workload | prefill Δ | decode Δ | peak GiB | generated/logit sanity |
| --- | ---: | ---: | ---: | ---: | --- |
| z-lab legacy W8A16 | 512/128 | +0.93% | +0.09% | 18.587 | first two generated IDs/logits match |
| z-lab legacy W8A16 | 4096/128 | +0.91% | -0.14% | 20.458 | first two generated IDs/logits match |
| shisa stripped packed | 512/128 | -0.17% | +0.25% | 18.535 | packed path unaffected/noise |
| shisa stripped packed | 4096/128 | -0.01% | -0.00% | 20.406 | packed path unaffected/noise |

Tile4 was rejected after a one-run probe because it regressed z-lab legacy 512
and 4K. A final no-env post-change pass confirmed the default path is wired.
Because this remains a resident-runner diagnostic rather than a promoted public
`LLM.generate()` row, `performance_claim=false` in the artifact.

Artifact:
`benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p13-shared-down-combine-token-tile-diagnostic.json`.

## 2026-05-17 — P1.4 selected-MoE compact WMMA threshold retained

Task #7 swept selected-MoE multi-token prefill routing after the P1.1-P1.3
prototypes. Added an env-controlled token-count threshold without backend/quant
branches in the hot layer dispatch:

- `HIPENGINE_MOE_PREFILL_COMPACT_WMMA_MIN_TOKENS` defaults to `2`.
- `tokens == 1` keeps the c1 GEMV/decode path.
- single-request FP16 prefill chunks with `tokens >= threshold` use grouped
  compact WMMA; chunks below threshold use the existing `run_moe_c1_fp16` GEMV
  fallback.
- A large diagnostic threshold (tested with `999999`) forces the GEMV fallback.
- Packed cN/varlen prefill remains grouped compact WMMA.
- Runner scratch selection now follows the same token-count helper, avoiding a
  grouped scratch allocation when the diagnostic c1 fallback is forced.

Validation:

```bash
python3 -m py_compile hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py
python3 -m pytest tests/test_qwen35_decode_state.py -q --tb=short
# 41 passed

python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json \
  --max-layers 40 \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p14-moe-threshold-sweep-20260517/p14_default_fixture_gate.json
# passed=true, max_kl=0.0395688706, top1_agreement=1.0, generated IDs match fixture
```

Benchmark protocol on W7900/gfx1100, all rows used cache-only HIP builds:

```bash
python3 scripts/qwen35_paro_bench.py \
  --model <zlab-legacy-or-shisa-packed> \
  --token-id 9707 \
  --prompt-length {128,256,512,4096} \
  --decode-tokens 128 \
  --warmup-decode-tokens 1 \
  --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p14-moe-threshold-sweep-20260517/<label>.json
# forced fallback rows add HIPENGINE_MOE_PREFILL_COMPACT_WMMA_MIN_TOKENS=999999
```

Required 512/128 and 4K/128 rows, retained default compact WMMA vs forced GEMV
fallback:

| model/path | workload | compact WMMA prefill | forced GEMV prefill | compact Δ | decode Δ | peak GiB default/fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| z-lab legacy | 512/128 | 2210.372 | 811.378 | +172.4% | -0.15% | 18.587 / 18.556 |
| z-lab legacy | 4096/128 | 2452.941 | 813.924 | +201.4% | +0.14% | 20.458 / 20.208 |
| shisa packed | 512/128 | 2403.645 | 850.397 | +182.6% | -0.13% | 18.535 / 18.503 |
| shisa packed | 4096/128 | 2664.070 | 849.221 | +213.7% | -0.24% | 20.406 / 20.155 |

Small z-lab probes also favored compact WMMA: 128/128 `1453.816` vs `703.792`
(+106.6%) and 256/128 `1897.076` vs `767.725` (+147.1%). Forced GEMV previews
kept the first generated token IDs but logits differed due the different
selected-MoE ordering/reduction; the retained default fixture gate above is the
correctness gate for the policy. The GEMV fallback uses slightly less scratch
(+0.008 to +0.251 GiB for compact WMMA), but the prefill regression is too large.

Conclusion: retain `HIPENGINE_MOE_PREFILL_COMPACT_WMMA_MIN_TOKENS=2`; no useful
GEMV crossover above decode token count one. P1.1-P1.4 are now closed. Artifact:
`benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p14-moe-wmma-threshold-diagnostic.json`.

## 2026-05-17 — D2.1 Marlin-K qweight-neutral decode retained

Task #8 ported the documented parent Marlin-K v0 path for non-expert PARO W4
rows==1 single GEMV decode. Pre-port checks:

- Re-read `docs/KERNELS.md` and `docs/MARLIN.md`.
- Ran `python3 scripts/check_lineage.py --kind kernel --diff stat`; parent
  `nano-vllm-amd` still reports drift vs the old `22405a9` baseline.
- Attempted `git -C /home/lhl/amd-gpu-tuning/nano-vllm-amd show 7718fff` and
  `1522293`; those short SHAs are not present in the current parent checkout.
  Used the committed evidence in `docs/MARLIN.md` and
  `/home/lhl/amd-gpu-tuning/PLAN-PAROQUANT2.md` §§11.10-11.11, which document
  `7718fff` (vec8 FMA) and `1522293` (qweight-neutral replacement).

Implementation:

- Added `hipengine/kernels/hip_gfx1100/quant/paro_marlin_k.{hip,py}` with a
  separate `paro_marlin_k.so` build family, registry key
  `(hip_gfx1100, marlin_k_gemv, w4_paro, fma_fp16)`, and the retained vec8
  FP32-FMA loop over `qweight_mk [N/8,K/128,128]`, `qzeros_mk [N/8,K/128]`,
  `scales_mk [N/8,K/128,8]`.
- Added torch-free NumPy repack helpers plus non-owning device allocation aliases
  so `qweight_pack8_decode [N/8,K]` is a zero-copy view over `qweight_mk`; this
  preserves fused pack8 prefill/QK/QKVZ paths without duplicate W4 qweight
  residency.
- `HIPENGINE_PARO_MARLIN_K_REPLACE` defaults on; setting it to `0` restores the
  old pack8/raw-qweight materialization for diagnostics.
- Runtime dispatch uses Marlin-K only for `rows == 1` FP16 single projections;
  rows>1 and fused pair projections continue to use the existing pack8/fusedw4
  paths through the alias.

Validation commands:

```bash
python3 -m py_compile \
  hipengine/kernels/hip_gfx1100/quant/paro_marlin_k.py \
  hipengine/loading/materialize.py hipengine/loading/qwen35_paro.py \
  hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py \
  scripts/smoke.py
python3 -m pytest \
  tests/test_qwen35_paro_marlin_k.py tests/test_qwen35_decode_state.py \
  tests/test_qwen35_paro_layout.py tests/test_build.py -q --tb=short
# 77 passed

python3 scripts/smoke.py --mode paro-marlin-k-hip --rows 2 --hidden-size 128 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
# mismatch=0 max_abs=0.0

rocprofv3 --kernel-trace --output-format csv \
  --output-directory /tmp/hipengine-d21-marlin-k-20260517/rocprof_marlin_smoke -- \
  python3 scripts/smoke.py --mode paro-marlin-k-hip --rows 2 --hidden-size 128 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
# gemv_paro_marlin_k_fma_kernel<_Float16>, DurationNs=6720, VGPR=104,
# Scratch_Size=0, LDS_Block_Size=512

python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json \
  --max-layers 40 --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-d21-marlin-k-20260517/native_prefill_fixture_gate.json
# passed=true, max_kl=0.0395688706, top1_agreement=1.0

python3 scripts/qwen35_decode_graph_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json \
  --max-layers 40 --graph-steps-per-replay 1 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-d21-marlin-k-20260517/decode_graph_fixture_gate.json
# passed=true, generated_match=true, expected_match=true, final_kl=0.0
```

Benchmark protocol on W7900/gfx1100 (ROCm HIP `7.2.53211-d40244d`), all rows
cache-only HIP builds:

```bash
# baseline rows add: HIPENGINE_PARO_MARLIN_K_REPLACE=0
# candidate rows add: HIPENGINE_PARO_MARLIN_K_REPLACE=1
python3 scripts/qwen35_paro_bench.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --token-id 9707 \
  --prompt-length {512,4096} \
  --decode-tokens 128 \
  --warmup-decode-tokens 1 \
  --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-d21-marlin-k-20260517/{off,on}_{prompt}_128_run{1,2,3}.json
```

Three-run medians, Marlin-K qweight-neutral default vs pack8/raw fallback:

| workload | fallback decode tok/s | Marlin-K decode tok/s | decode Δ | fallback prefill tok/s | Marlin-K prefill tok/s | peak GiB fallback/Marlin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 109.061 | 115.137 | +5.57% | 2209.588 | 2231.018 | 18.587 / 18.176 |
| 4096/128 | 110.088 | 116.263 | +5.61% | 2436.294 | 2468.623 | 20.458 / 20.047 |

Conclusion: retain Marlin-K qweight-neutral replacement as the default. It is a
resident-runner diagnostic row (`performance_claim=false`) because public
`LLM.generate()` throughput is still not promoted, but the required 512/128 and
4K/128 decode rows both improve with low variance, correctness gates pass, the
kernel is visible under rocprof, and peak memory remains below the 24 GiB guardrail
while dropping by ~0.411 GiB. Artifact:
`benchmarks/results/2026-05-17-hipengine-qwen35-d21-marlin-k-qweight-neutral-diagnostic.json`.

## 2026-05-17 — D5.2 W8A16 decode kernel audit stop condition

Task #9 audited the W8A16 decode family called out by the M.4 Amdahl profile:
`w8a16_linear_kernel` for lm-head BF16→FP32 and
`w8a16_linear_lowp_out_kernel<_Float16>` for the legacy shared-expert gate/up
and down projections. No fused lm-head/argmax work was attempted.

M.4 source evidence (`benchmarks/results/2026-05-17-hipengine-qwen35-rocprof-amdahl-diagnostic.json`):

- 512/128 decode graph window: W8A16 `15.67%`, `1.138 ms/token`, `81` calls/token.
- 4K/128: W8A16 `15.72%`, `1.136 ms/token`.
- 32K/128: W8A16 `13.36%`, `1.149 ms/token`.
- Split at 512/128: lm-head W8A16 `0.694 ms/token`, shared lowp W8A16 `0.445 ms/token`; argmax is only `0.0069 ms/token`.

ISA/resource audit on the cached decode build (`w8a16_linear-033698c919656936`):

- `w8a16_linear_kernel`: SGPR 32, VGPR 23, no SGPR/VGPR spills, private segment 0, fixed LDS 0, wave32; dynamic LDS is `threads * sizeof(float)` for the reduction.
- `w8a16_linear_lowp_out_kernel<hip_bfloat16>`: SGPR 32, VGPR 23, no spills/scratch.
- `w8a16_linear_lowp_out_kernel<_Float16>`: SGPR 32, VGPR 20, no spills/scratch.
- Static disassembly is the expected vec8 loop plus LDS reduction (`global_load`, `v_cvt`, `v_fma_mix`/`v_fmac`, `ds_*`, `s_barrier`); no scratch path appeared.

Added `scripts/w8a16_decode_probe.py`, a synthetic decode-shape profiler helper,
and ran it under rocprof:

```bash
rocprofv3 --kernel-trace --output-format csv \
  --output-directory /tmp/hipengine-d52-w8a16-probe-repo \
  --output-file w8a16_decode_probe -- \
  python3 scripts/w8a16_decode_probe.py \
    --threads 64,128,256,512 \
    --lm-reps 5 \
    --shared-reps 40 \
    --include-fused-shared \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Profiler medians after dropping warmup rows:

| case | best / current | probes |
| --- | --- | --- |
| lm-head `w8a16_linear_kernel`, 2048→248320 | 128 threads, `674.285 us` | 64 `688.765`, 256 `913.247`, 512 `1753.732` us |
| shared gate/up `w8a16_linear_lowp_out_kernel<_Float16>`, 2048→1024 | 64 threads, `3.220 us` | 128 `3.640`, 256 `4.760`, 512 `7.340` us |
| shared down `w8a16_linear_lowp_out_kernel<_Float16>`, 512→2048 | 64 threads, `3.440 us` | 128 `4.240`, 256 `7.400`, 512 `12.320` us |
| existing fused shared gate/up+SiLU helper, c=1 probe | rejected | best `~5.040 us`, slower than generic lowp gate/up plus measured graph shared SiLU (~`3.22 + 1.28 us`) |

Conclusion / stop condition:

- Keep `HIPENGINE_QWEN35_LM_HEAD_THREADS=128` and the shared lowp W8A16 default
  at 64 threads. Larger workgroups lose to reduction/wave overhead; 64 is not
  better for lm-head.
- No code-path change retained. The remaining W8A16 lm-head cost is dominated by
  the unavoidable 248320×2048 int8 weight stream (~509 MB/token before cache and
  output/scales), while the shared-expert W8A16 linears are already low-VGPR,
  no-spill, and best at the existing small workgroup.
- Legacy-vs-packed shared-expert decode tradeoff remains a policy/format issue,
  not a W8A16 occupancy bug: forced legacy W8A16 was faster for shisa decode but
  worse for prefill/memory, and after packed-W4 recovery the remaining packed
  decode gap is ~3.3%; D5.2 found no W8A16 micro-tune that changes the packed
  checkpoint default.

Validation:

```bash
python3 -m py_compile scripts/w8a16_decode_probe.py
python3 -m pytest tests/test_w8a16_linear_plan.py -q --tb=short
python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-d52-w8a16-decode-audit.json >/tmp/d52-json-check.json
# validation ok
```

Recorded artifact and rollup:

- `benchmarks/results/2026-05-17-hipengine-qwen35-d52-w8a16-decode-audit.json`
- `docs/OPTIMIZE.md` D5.2 marked accepted(stop)
- `benchmarks/README.md` diagnostic row + `benchmarks/CHANGELOG.md` one-liner

## 2026-05-17 — D1.1 rotation-into-W4 producer decode fusion rejected

Task #10 implemented an opt-in generic transposed rotate-staged dual pack8 GEMV
surface for the decode projection pairs that share an input:

- Kernel wrappers: `gemv_awq_dual_pack8_transposed_rotate_staged_{bf16,fp16}`.
- Runtime gate: `HIPENGINE_PARO_ROTATE_DUAL_PACK8_FUSED=1` (default remains off).
- Covered decode pairs: linear-attention `in_proj_qkv + in_proj_z`; full-attention
  `q_proj + k_proj` with `v_proj` still rotated by the existing single-rotate path.
- Design: rotate each input once into the existing pack8/repacked scratch, then run
  the dual transposed pack8 GEMV after a device-side barrier; no per-output-pack
  rotation recompute, preserving the parent D4 rejection lesson.

Correctness / visibility:

```bash
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py -q --tb=short
python3 -m py_compile hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.py hipengine/runtime/qwen35_paro.py tests/test_qwen35_decode_state.py scripts/smoke.py
python3 scripts/smoke.py --mode paro-pack8-rotate-staged-hip --rows 1 --hidden-size 128 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/check_lineage.py --kind kernel --diff stat  # known drift: qwen35_expert, smoke, paroquant_kernels, paroquant_fusedw4
python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-d11-rotate-dual-pack8-fusion-rejected.json >/tmp/d11-json-check.json
git diff --check
```

The opt-in graph fixture gate matched the known generated sample/logits (`final_kl=0`,
expected/generated match true), and rocprof confirmed the fused FP16 kernel path:
`gemv_awq_dual_pack8_transposed_rotate_staged_kernel<_Float16,true>` with two
observed dispatches in the small smoke trace (`DurationNs` 36320 / 33761,
`VGPR_Count=104`, `Scratch_Size=0`, `LDS_Block_Size=512`).

Performance decision:

- 512/128 graph decode default/off median: `115.450 tok/s`, prefill `2268.369 tok/s`,
  peak `18.176 GiB`.
- 512/128 opt-in fused median: `110.457 tok/s`, prefill `2262.721 tok/s`, peak
  `18.176 GiB`.
- Delta: decode `-4.32%`, prefill `-0.25%`, memory neutral.

Rejected as a default. The removed rotate launches are cheaper under one-step graph
replay than the barrier/spin/global-staging overhead; full-attention Q/K also leaves
V on a separate rotate, so it does not remove the whole rotation launch family. The
opt-in code path and kernel catalog entry are kept as evidence/diagnostic surface;
`docs/OPTIMIZE.md` marks D1.1 rejected and the artifact is
`benchmarks/results/2026-05-17-hipengine-qwen35-d11-rotate-dual-pack8-fusion-rejected.json`.

## 2026-05-17 — D1.4 selected-MoE post-op fold stop/reject

Task #11 audited the selected-MoE post-op fold. The safe c=1 decode fold is
already the default hipEngine path: `run_moe_c1_fp16()` calls
`combine_moe_c1_shared_residual_fp16()`, which launches
`weighted_sum_shared_gate_combine_residual_out_fp16_f32w` for `tokens == 1`.
That one kernel already performs selected-expert weighted sum, shared-gate
sigmoid/combine, and residual add. The remaining combine bucket in M.4 is the
one launch per MoE layer (`~40` calls/token), not a split post-op chain.

Validation / probes:

```bash
python3 -m pytest tests/test_paro_combine_plan.py tests/test_qwen35_decode_state.py -q --tb=short
python3 scripts/smoke.py --mode paro-combine-hip --rows 8 --hidden-size 2048 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_decode_graph_fixture_gate.py --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd --max-new-tokens 16 --json /tmp/hipengine-d14/graph-fixture-current.json
PYTHONPATH=. rocprofv3 --kernel-trace --output-format csv --output-directory /tmp/hipengine-d14-combine-probe --output-file combine_threads -- python3 /tmp/d14_combine_probe.py --features 2048 --rows 8 --reps 50 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_paro_bench.py --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 --graph-replay-decode --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/hipengine-d14/512x128-current.json
```

Results:

- Combine smoke at `rows=8`, `hidden=2048` passed BF16/FP16 weighted/fused/batch/
  shared/weighted-lane checks with all mismatches `0`.
- Graph fixture gate passed (`generated_match=true`, `expected_match=true`,
  `final_kl=0`, top-1 match true).
- Combine thread probe medians for
  `weighted_sum_shared_gate_combine_residual_out_kernel<_Float16,float>` were
  effectively identical: 64 threads `2440.0 ns`, 128 threads `2440.5 ns`,
  256 threads `2440.0 ns`; no thread-count lever.
- Current 512/128 default diagnostic: prefill `2284.652 tok/s`, decode
  `115.755 tok/s`, peak `18.176 GiB`. This is current baseline/no candidate
  delta, not a retained improvement.

Decision: reject further D1.4 default changes. The only direct way to remove the
remaining combine launch is to fuse selected down-projection with weighted sum /
shared gate / residual, but parent target-shape evidence already rejected that
exact design: bit-correct, yet `13.38 us -> 16.52 us` (`0.81x`) because it
collapses grid parallelism from `out_pack * active_experts` blocks to `out_pack`
blocks. Reopen only for a design that preserves per-expert/out-pack parallelism
without atomics or a graph-level fusion that removes the dispatch without
changing selected-down GEMV layout.

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-d14-selected-moe-postop-fold-rejected.json`.

## 2026-05-17 — D1.5 router cooperative fold rejected

Task #12 prototyped an opt-in decode-only router cooperative fold behind
`HIPENGINE_PARO_ROUTER_TOPK_COOP=1`:

- `qwen35_router_topk_shared_coop_out_kernel` keeps the one-block-per-router-row
  logits producer grid (`num_experts + shared_gate` blocks) and uses a global
  atomic counter so the last producer block runs the existing block-parallel
  top-k/softmax tail.
- The default remains the separate `qwen35_router_logits_*` +
  `qwen35_router_select_kernel` path. The cooperative wrapper must reset the
  counter with `hipMemsetAsync` before each decode launch, so graph replay sees
  a memset node plus the folded kernel rather than a pure launch removal.

Validation / probes:

```bash
python3 -m py_compile hipengine/kernels/hip_gfx1100/moe/router.py hipengine/runtime/qwen35_paro.py scripts/smoke.py tests/test_qwen35_router_plan.py tests/test_qwen35_decode_state.py
python3 -m pytest tests/test_qwen35_router_plan.py tests/test_qwen35_decode_state.py -q --tb=short
python3 scripts/smoke.py --mode qwen35-router-hip --rows 1 --hidden-size 256 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
HIPENGINE_PARO_ROUTER_TOPK_COOP=1 python3 scripts/qwen35_decode_graph_fixture_gate.py --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd --max-new-tokens 16 --json /tmp/hipengine-d15/graph-fixture-coop.json
PYTHONPATH=. rocprofv3 --kernel-trace --output-format csv --output-directory /tmp/hipengine-d15-router-probe --output-file router_probe -- python3 /tmp/d15_router_probe.py --reps 80 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_paro_bench.py --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 --graph-replay-decode --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/hipengine-d15/512x128-default-1.json
HIPENGINE_PARO_ROUTER_TOPK_COOP=1 python3 scripts/qwen35_paro_bench.py --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 --graph-replay-decode --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/hipengine-d15/512x128-coop-1.json
python3 scripts/qwen35_paro_bench.py --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 --graph-replay-decode --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/hipengine-d15/4k128-default-1.json
HIPENGINE_PARO_ROUTER_TOPK_COOP=1 python3 scripts/qwen35_paro_bench.py --prompt-length 4096 --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 --graph-replay-decode --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/hipengine-d15/4k128-coop-1.json
python3 scripts/check_lineage.py --kind kernel --diff stat  # known parent drift in qwen35_expert/smoke/paroquant kernels
python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-d15-router-coop-fold-rejected.json >/tmp/d15-json-check.json
git diff --check
```

Correctness passed: router smoke reported BF16/FP16 `selected_match=True` and
cooperative `coop_selected_match=True`; graph fixture matched generated tokens
and logits (`final_kl=0`, expected/generated match true); unit tests passed
`48/48`.

Micro-profile at the target router shape (`hidden=2048`, `num_rows=257`,
`top_k=8`, `threads=512`) showed the kernel-only fold was real but too small:

- Default logits kernel median `3080 ns` (`VGPR=24`, no LDS) plus select median
  `5080 ns` (`VGPR=40`, `LDS=512`) = `8160 ns` kernel-only.
- Cooperative producer median `7080 ns` (`VGPR=40`, `LDS=512`) while preserving
  the 257-row producer grid; this excludes the required counter memset.

End-to-end graph replay regressed, so reject as a default:

- 512/128 default: prefill `2274.284 tok/s`, decode `115.931 tok/s`, peak
  `18.176 GiB`.
- 512/128 coop: prefill `2236.431 tok/s`, decode `114.856 tok/s`, peak
  `18.176 GiB` (`-0.93%` decode).
- 4K/128 default: prefill `2491.236 tok/s`, decode `116.887 tok/s`, peak
  `20.047 GiB`.
- 4K/128 coop: prefill `2490.971 tok/s`, decode `116.106 tok/s`, peak
  `20.047 GiB` (`-0.67%` decode).

Decision: keep the cooperative path only as an opt-in diagnostic. Reopen D1.5 /
D5.3 only for a graph-level fusion or persistent initialized counter design that
removes the extra memset node without racing; the naive one-block collapse remains
invalid because it abandons the router producer occupancy.

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-d15-router-coop-fold-rejected.json`.

## 2026-05-17 — D3.1-D3.3 grouped-GQA long-context decode retained

Task #13 ported and retained the parent grouped-GQA paged split-K decode
producer for Qwen3.5 full-attention decode, then swept the split cap and split
threshold defaults:

- Added `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_kernel<8,16,2>`
  launch coverage through `qwen35_paged_full_attn_decode_split_k_gqa_*_spans`.
- Added gated warp split wrappers so short/mid split decode can use the parent
  warp-cooperative producer while long Qwen3.5 GQA rows switch to grouped-GQA.
- Default policy: split decode from context `>=1024`, grouped-GQA when
  `num_splits >= 64` or context `>=4096`, and `HIPENGINE_PAGED_ATTN_MAX_SPLITS=4096`
  (no effective 128K cap). Opt-outs remain env-controlled.

Correctness / visibility:

```bash
python3 -m pytest \
  tests/test_qwen35_paged_attn_decode_plan.py \
  tests/test_qwen35_decode_state.py \
  tests/test_qwen35_resident_batch_layout.py -q --tb=short
# passed before benchmark sweep

python3 scripts/smoke.py --mode qwen35-paged-attn-gqa-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
# context_len=512, warp_max_abs=4.1e-08, gqa_max_abs=4.1e-08,
# warp/GQA BF16 gated mismatches 0

python3 scripts/qwen35_decode_graph_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json \
  --max-layers 40 --graph-steps-per-replay 1 --max-new-tokens 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --attn-aotriton-min-tokens 512 --json /tmp/hipengine-d31/graph-fixture-512.json
# passed=true, generated_match=true, expected_match=true, final_kl=0

rocprofv3 --kernel-trace --output-format csv \
  --output-directory /tmp/hipengine-d31/rocprof --output-file paged_attn_gqa_smoke -- \
  python3 scripts/smoke.py --mode qwen35-paged-attn-gqa-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
# trace includes qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_kernel<8,16,2>
# (DurationNs 109281/99761 in the two smoke launches, VGPR=80, scratch=0)
```

Benchmark protocol on W7900/gfx1100, cache-only HIP builds, graph replay decode:

```bash
python3 scripts/qwen35_paro_bench.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --token-id 9707 \
  --prompt-length {512,4096,32768,131072} \
  --decode-tokens 128 --warmup-decode-tokens 4 --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-d31/<label>.json
# long rows add prefill chunk flags: linear=1024, moe=1024,
# full-attn query=4096, post=1024, rope=1024
# opt-out rows set HIPENGINE_PAGED_ATTN_GQA_GROUPED_CTX=0
# cap row sets HIPENGINE_PAGED_ATTN_MAX_SPLITS=512
# old threshold row sets HIPENGINE_PARO_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT=4096
```

Results:

| sweep | workload | baseline | retained/default | decode Δ | peak default |
| --- | ---: | ---: | ---: | ---: | ---: |
| grouped-GQA | 32K/128 | opt-out `70.064 tok/s` | `99.560 tok/s` | `+42.1%` | `20.320 GiB` |
| grouped-GQA | 128K/128 | opt-out `30.789 tok/s` | `63.368 tok/s` | `+105.8%` | `23.288 GiB` |
| split cap | 128K/128 | default `63.368 tok/s` | cap512 `62.647 tok/s` | `-1.14%` | cap saves only `0.034 GiB` |
| split threshold | 1K/128 | threshold4096 `92.486 tok/s` | threshold1024 `113.242 tok/s` | `+22.4%` | `18.443 GiB` |

Short-context guard rows stayed within the unchanged-behavior band versus the prior
D1.5 warmup-4 baseline: 512/128 `115.931 -> 115.627 tok/s` (-0.26%)
and 4K/128 `116.887 -> 116.263 tok/s` (-0.53%). Memory guardrails
remain satisfied: 32K peak `20.320 GiB` (A.3 <=20.69) and 128K peak
`23.288 GiB` (A.4 <24).

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-d31-d33-grouped-gqa-long-context-diagnostic.json`.

Post-artifact verification:

```bash
python3 -m py_compile \
  hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.py \
  hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py \
  scripts/smoke.py

python3 -m pytest \
  tests/test_qwen35_paged_attn_decode_plan.py \
  tests/test_qwen35_decode_state.py \
  tests/test_qwen35_resident_batch_layout.py -q --tb=short
# 76 passed
```

## 2026-05-17 — Shisa Qwen3.6 packed-vs-legacy PARO refresh and compare tables

Refreshed the shisa-ai Qwen3.6 PARO packed-vs-legacy diagnostic after the latest
approved decode/prefill defaults, and updated `scripts/qwen35_compare_tables.py`
so packed PARO sidecars are the least-surprising default comparison A:

- `python3 scripts/qwen35_compare_tables.py --target shisa --against-target`
  prints packed A vs legacy B.
- `python3 scripts/qwen35_compare_tables.py --target legacy --against-target`
  flips the A/B direction for decode-focused diagnostics.
- `python3 scripts/qwen35_compare_tables.py --target shisa all` compares the
  packed shisa row against external baselines.

Benchmark protocol on W7900/gfx1100, cache-only HIP builds, graph replay decode:

```bash
# short rows
python3 scripts/qwen35_paro_bench.py \
  --model /models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5/snapshots/1492d9ae108682763e67b28ff4aad660d7e19cd4 \
  --shared-expert-format {packed_paro_w4,legacy_fp16} \
  --prompt-length {512,4096} --token-id 9707 --decode-tokens 128 \
  --warmup-decode-tokens 4 --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --attn-aotriton-min-tokens 512 --json /tmp/hipengine-shisa36-packed-legacy-20260517/<label>.json

# long rows add parent-style chunks
python3 scripts/qwen35_paro_bench.py ... \
  --prompt-length {32768,131072} \
  --prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 \
  --prefill-full-attn-query-chunk-size 4096 \
  --prefill-full-attn-post-chunk-size 1024 \
  --prefill-full-attn-rope-chunk-size 1024

# packed-only stripped checkpoint sanity at 512/4K
python3 scripts/qwen35_paro_bench.py \
  --model /models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e \
  --shared-expert-format auto --prompt-length {512,4096} ...
```

Results (packed A vs legacy B):

| workload | packed prefill | legacy prefill | packed Δ | packed decode | legacy decode | packed Δ | packed peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512/128 | `2518.836` | `2272.088` | `+10.9%` | `111.738` | `115.324` | `-3.1%` | `18.123 GiB` |
| 4K/128 | `2711.013` | `2487.298` | `+9.0%` | `113.231` | `116.688` | `-3.0%` | `19.995 GiB` |
| 32K/128 | `2130.562` | `1974.833` | `+7.9%` | `97.779` | `99.746` | `-2.0%` | `20.267 GiB` |
| 128K/128 | `1048.543` | `1002.841` | `+4.6%` | `62.014` | `63.190` | `-1.9%` | `23.235 GiB` |

Packed saves ~`0.052 GiB` tracked peak at every shape relative to legacy by
omitting legacy W8A16 shared-expert buffers. Generated previews for all rows
remain repeated token `9707`. Stripped packed-only auto rows at 512/4K match the
forced-packed token previews and memory, with timing differing only by single-run
noise.

Packed-path approved-optimization audit:

- Applicable and integrated: P1.4 compact WMMA prefill threshold=2, P2.3
  AOTriton threshold=512, D2.1 Marlin-K non-expert decode default,
  D3.1-D3.3 grouped-GQA long decode/default threshold, and packed shared-expert
  decode fusion (`paro_rotate2` gate/up + fused SiLU+down-rotate).
- Accepted legacy-only optimizations P1.2/P1.3 are deliberately not applicable
  to packed sidecars; the refresh shows packed still wins prefill and memory.
- `docs/OPTIMIZE.md` was corrected so D2.1 is no longer stale/pending.

Created follow-up task entries for the remaining pending `docs/OPTIMIZE.md`
items to process as accept/reject/defer decisions: P1.6, P3.1-P3.3, P5.2,
D1.2, D1.3, D1.6, D4.2, D4.4, and D5.1.

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen36-shisa-packed-vs-legacy-refresh-diagnostic.json`.

Verification for the shisa refresh/doc unit:

```bash
python3 -m py_compile scripts/qwen35_compare_tables.py
python3 scripts/qwen35_compare_tables.py --target shisa --against-target >/tmp/compare-shisa.md
python3 scripts/qwen35_compare_tables.py nano-vllm-amd >/tmp/compare-qwen35.md
python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen36-shisa-packed-vs-legacy-refresh-diagnostic.json >/tmp/shisa-artifact-check.json
python3 -m pytest tests/test_qwen35_paro_layout.py -q --tb=short
# 20 passed
```

## 2026-05-17 — P1.6 selective prefill `-mcumode` build-profile sweep

Task #17 evaluated `docs/OPTIMIZE.md` P1.6 after the P1.5 unroll-600 default. I
added a diagnostic-only build knob, `HIPENGINE_PREFILL_MCUMODE=1`, which appends
`-mcumode` only to artifacts built with the `prefill` profile and does not
duplicate the flag on libraries that already request it. Dry-run audit of the
surface with `/tmp/hipengine-hipcc-version.txt`:

- default `aotriton_wrap`: `-mllvm -amdgpu-unroll-threshold-local=600` plus
  include/linker flags, no `-mcumode`.
- default `qwen35_moe_group_scatter`: `-mllvm -amdgpu-unroll-threshold-local=600`.
- default `paro_awq_wmma`: already includes `-mcumode`.
- with `HIPENGINE_PREFILL_MCUMODE=1`, only `aotriton_wrap` and
  `qwen35_moe_group_scatter` gain `-mcumode`; compact WMMA and most dual-use
  decode/prefill libraries were already CU-mode builds.

Benchmark protocol on W7900/gfx1100, current `HEAD=5336924`, uncommitted P1.6
knob/tests/artifact docs pending. All measured rows used cache-only HIP builds
after this prebuild:

```bash
HIPENGINE_PREFILL_MCUMODE=1 python3 scripts/qwen35_paro_bench.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --token-id 9707 --prompt-length 512 --decode-tokens 1 --warmup-decode-tokens 0 \
  --max-layers 4 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p16-mcumode-20260517/prebuild-prefill-mcumode-l4.json
```

Measured commands were two repetitions of default vs `HIPENGINE_PREFILL_MCUMODE=1`
for z-lab Qwen3.5 legacy and shisa Qwen3.6 forced-packed at 512/128 and 4K/128:

```bash
python3 scripts/qwen35_paro_bench.py \
  --model <zlab-or-shisa-unstripped> --shared-expert-format {auto,packed_paro_w4} \
  --prompt-length {512,4096} --token-id 9707 --decode-tokens 128 \
  --warmup-decode-tokens 4 --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p16-mcumode-20260517/<mode>-<model>-<prompt>-128-runN.json

HIPENGINE_PREFILL_MCUMODE=1 python3 scripts/qwen35_paro_bench.py ...same args...
```

Two-run median results, `HIPENGINE_PREFILL_MCUMODE=1` vs default:

| model | workload | default prefill | mcumode prefill | Δ | default decode | mcumode decode | Δ | peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| z-lab Qwen3.5 legacy | 512/128 | `2213.547` | `2218.591` | `+0.23%` | `115.429` | `115.295` | `-0.12%` | `18.176` |
| z-lab Qwen3.5 legacy | 4K/128 | `2467.088` | `2463.313` | `-0.15%` | `116.718` | `116.701` | `-0.01%` | `20.047` |
| shisa Qwen3.6 packed | 512/128 | `2423.833` | `2437.094` | `+0.55%` | `111.547` | `111.634` | `+0.08%` | `18.123` |
| shisa Qwen3.6 packed | 4K/128 | `2675.662` | `2681.342` | `+0.21%` | `112.463` | `112.630` | `+0.15%` | `19.995` |

Correctness/sanity:

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
HIPENGINE_PREFILL_MCUMODE=1 \
python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --max-layers 40 --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p16-mcumode-20260517/prefill-mcumode-native-prefill-fixture-gate.json
# passed=True, max_kl=0.039568870612619614, top1_agreement=1.0, generated_match=True
```

Default and P1.6 rows produced identical first-two generated token IDs and logits
(max logit delta `0.0`) across every repetition. Decision: reject P1.6 as a
default build-profile change because the measured surface is neutral/noisy
(prefill `-0.15%..+0.55%`, decode `-0.12%..+0.15%`, memory unchanged). Keep
`HIPENGINE_PREFILL_MCUMODE=1` only as a future compiler diagnostic knob.

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p16-prefill-mcumode-rejected.json`.

Post-update validation:

```bash
python3 -m py_compile hipengine/core/build.py scripts/qwen35_paro_bench.py
python3 -m pytest tests/test_build.py tests/test_qwen35_paro_layout.py -q --tb=short
# 27 passed
python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p16-prefill-mcumode-rejected.json >/tmp/p16-artifact-check.json
git diff --check
```

## 2026-05-17 — P3.1 GDN prefill RMSNorm+gate+rotate fusion diagnostic

Task #18 evaluated `docs/OPTIMIZE.md` P3.1. I added a diagnostic opt-in
`HIPENGINE_LINEAR_GDN_PREFILL_ROTATE_FUSED=1` that keeps the existing fallback as
default and routes only safe Qwen3.5/PARO FP16 single-request prefill shapes
(`tokens > 1`, `head_v_dim == group_size`) through a new
`qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16` kernel. The fused kernel computes
per-value-head RMSNorm + SiLU gate from `recurrent_out`, rounds the gated value to
FP16 to match the old materialized path, applies the PARO rotate1 group, and
writes `out_rot` directly before the unchanged `awq_fusedw4_prefill_strided_fp16`
out projection.

Correctness/sanity:

```bash
python3 scripts/smoke.py --mode qwen35-linear-attn-prefill-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
# ... fp16_gated_mismatch=0 fused_rotate_mismatch=0

HIPENGINE_LINEAR_GDN_PREFILL_ROTATE_FUSED=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --max-layers 40 --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p31-gdn-rotate-20260517/fused-native-prefill-fixture-gate.json
# passed=True, max_kl=0.039568870612619614, top1_agreement=1.0, generated_match=True
```

Benchmark protocol on W7900/gfx1100 used cache-only HIP builds, graph replay
decode, two repetitions per row, and current `HEAD=9fecaa0` plus the uncommitted
P3.1 diagnostic kernel/runtime/docs/artifact updates:

```bash
python3 scripts/qwen35_paro_bench.py \
  --model <zlab-or-shisa-unstripped> --shared-expert-format {auto,packed_paro_w4} \
  --prompt-length {4096,32768} --token-id 9707 --decode-tokens 128 \
  --warmup-decode-tokens 4 --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p31-gdn-rotate-20260517/<mode>-<model>-<prompt>-128-runN.json

# 32K rows also used the current long-context chunk policy:
--prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 \
--prefill-full-attn-query-chunk-size 4096 --prefill-full-attn-post-chunk-size 1024 \
--prefill-full-attn-rope-chunk-size 1024

HIPENGINE_LINEAR_GDN_PREFILL_ROTATE_FUSED=1 python3 scripts/qwen35_paro_bench.py ...same args...
```

Two-run median results, fused vs default:

| model | workload | default prefill | fused prefill | Δ | default decode | fused decode | Δ | peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| z-lab Qwen3.5 legacy | 4K/128 | `2454.597` | `2467.638` | `+0.53%` | `116.473` | `116.464` | `-0.01%` | `20.047` |
| z-lab Qwen3.5 legacy | 32K/128 | `1950.331` | `1942.608` | `-0.40%` | `98.923` | `98.488` | `-0.44%` | `20.320` |
| shisa Qwen3.6 packed | 4K/128 | `2658.331` | `2672.125` | `+0.52%` | `112.400` | `112.576` | `+0.16%` | `19.995` |
| shisa Qwen3.6 packed | 32K/128 | `2067.609` | `2061.326` | `-0.30%` | `96.219` | `95.700` | `-0.54%` | `20.267` |

Default and fused rows generated identical first-two token IDs and logits (max
logit delta `0.0`) across every repetition. Decision: reject P3.1 as a default
fusion. The kernel is correct, but the retained surface is neutral-to-negative:
4K gains are only ~`+0.5%`, 32K regresses, decode is not improved, and tracked
peak memory is unchanged because the resident scratch allocator still reserves
`recurrent_bf16` for the fallback path. The env knob remains off by default as a
future diagnostic/prototype surface.

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p31-gdn-rotate-rejected.json`.

Post-update validation:

```bash
python3 -m py_compile \
  hipengine/kernels/hip_gfx1100/linear_attn/gdn.py \
  hipengine/kernels/hip_gfx1100/linear_attn/__init__.py \
  hipengine/runtime/qwen35_paro.py scripts/smoke.py \
  tests/test_qwen35_linear_attn_gdn_plan.py
python3 -m pytest tests/test_qwen35_linear_attn_gdn_plan.py tests/test_qwen35_paro_layout.py -q --tb=short
# 23 passed
python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p31-gdn-rotate-rejected.json >/tmp/p31-artifact-check.json
git diff --check
```

---

## 2026-05-17 — P3.2 prefill router shared-gate sigmoid diagnostic rejected

Task #19 evaluated `docs/OPTIMIZE.md` P3.2. I added a diagnostic opt-in
`HIPENGINE_PREFILL_ROUTER_SHARED_GATE_SIGMOID_FUSED=1` that keeps the existing
fallback as default and routes only legacy-FP16 shared-expert prefill
(`tokens > 1`, `legacy_fp16` shared expert) through a new router select variant.
The variant overwrites the shared-gate logit column with `sigmoid(logit)` inside
`qwen35_router_select_sigmoid_shared_kernel`, so legacy grouped prefill can skip
`w8a16_shared_gate_sigmoid_fp32`. c=1 decode and packed PARO shared-expert paths
continue to preserve raw shared-gate logits for their combine kernels.

Correctness/profiler evidence:

```bash
python3 scripts/smoke.py --mode qwen35-router-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
# selected_match=True, routing_max_abs=1.49e-08, fp16_selected_match=True,
# sigmoid_logits_max_abs=0.0, sigmoid_selected_match=True,
# sigmoid_fp16_logits_max_abs=4.77e-07, sigmoid_fp16_selected_match=True

rocprofv3 --kernel-trace --output-directory /tmp/hipengine-rocprof-router-p32-sigmoid \
  --output-file router-p32 --output-format csv -- \
  python3 scripts/smoke.py --mode qwen35-router-hip --rows 2 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
# qwen35_router_select_sigmoid_shared_kernel rows: duration 11840 ns and 3920 ns,
# Scratch_Size=0, VGPR_Count=40, LDS_Block_Size=512, Workgroup_Size_X=64.

HIPENGINE_PREFILL_ROUTER_SHARED_GATE_SIGMOID_FUSED=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --max-layers 40 --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p32-router-sigmoid-20260517/fused-native-prefill-fixture-gate.json
# passed=True, max_kl=0.039568870612619614, top1_agreement=1.0, generated_match=True
```

Benchmark protocol on W7900/gfx1100 used cache-only HIP builds, graph replay
decode, two repetitions per row, and current `HEAD=ca4796d` plus the uncommitted
P3.2 diagnostic kernel/runtime/smoke/docs/artifact updates:

```bash
python3 scripts/qwen35_paro_bench.py \
  --model <zlab-or-shisa-unstripped> --shared-expert-format {auto,packed_paro_w4} \
  --prompt-length {512,4096} --token-id 9707 --decode-tokens 128 \
  --warmup-decode-tokens 4 --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p32-router-sigmoid-20260517/<mode>-<model>-<prompt>-128-runN.json

HIPENGINE_PREFILL_ROUTER_SHARED_GATE_SIGMOID_FUSED=1 python3 scripts/qwen35_paro_bench.py ...same args...
```

Two-run median results, fused vs default:

| model | workload | default prefill | fused prefill | Δ | default decode | fused decode | Δ | peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| z-lab Qwen3.5 legacy | 512/128 | `2220.024` | `2224.583` | `+0.21%` | `115.308` | `115.242` | `-0.06%` | `18.176` |
| z-lab Qwen3.5 legacy | 4K/128 | `2467.279` | `2461.592` | `-0.23%` | `116.797` | `117.024` | `+0.19%` | `20.047` |
| shisa Qwen3.6 packed | 512/128 | `2429.718` | `2445.832` | `+0.66%` | `111.795` | `111.707` | `-0.08%` | `18.123` |
| shisa Qwen3.6 packed | 4K/128 | `2677.880` | `2672.998` | `-0.18%` | `112.622` | `112.945` | `+0.29%` | `19.995` |

Default and fused rows generated identical first-two token IDs (`9707`, `9707`)
across every repetition. Decision: reject P3.2 as a default fusion. Removing the
extra legacy shared-gate sigmoid launch is correct, but the retained E2E surface
is neutral/noisy: Qwen3.5 legacy is only `+0.21%` at 512 and regresses `-0.23%`
at 4K, decode does not materially improve, and tracked peak memory is unchanged.
The shisa packed rows are intentionally not routed through the sigmoid variant;
their deltas are noise checks for raw-logit preservation. The env knob remains
off by default for future diagnostics only.

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p32-router-sigmoid-rejected.json`.

Post-update validation:

```bash
python3 -m py_compile scripts/smoke.py hipengine/kernels/hip_gfx1100/moe/router.py \
  hipengine/runtime/qwen35_paro.py tests/test_qwen35_router_plan.py tests/test_qwen35_decode_state.py
python3 -m pytest tests/test_qwen35_router_plan.py tests/test_qwen35_decode_state.py -q --tb=short
# 52 passed
python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-qwen36-p32-router-sigmoid-rejected.json >/tmp/p32-artifact.pretty
git diff --check
```

---

## 2026-05-17 — P3.3 MoE metadata fanout collapse deferred by profile gate

Task #20 evaluated `docs/OPTIMIZE.md` P3.3. I did **not** implement the
proposed fused metadata kernel because the prerequisite M.3 rocprof evidence
already shows the target is below material payoff for the current c=1 prefill
path. The proposed change would combine `qwen35_moe_group_prefix` +
`qwen35_moe_wmma_tile_map` and initialize `scatter_offsets`/`tile_expert` in the
same small metadata kernel, but a fused kernel would still perform the same
prefix/tile-map work and write the same metadata.

Source evidence is the retained selected-region profile
`benchmarks/results/2026-05-17-hipengine-qwen35-rocprof-amdahl-diagnostic.json`
on W7900/gfx1100, Qwen3.5-35B-A3B-PARO, `w4_paro`, max_layers=40,
cache-only builds, graph replay decode, and the 512/128 + 4K/128 + 32K/128
M.3 workload set:

```bash
python3 scripts/qwen35_rocprof_audit.py --workloads 512/128 4096/128 32768/128 \
  --out benchmarks/results/2026-05-17-hipengine-qwen35-rocprof-amdahl-diagnostic.json
```

Profile-derived prefill upper-bound summary:

| workload | MoE metadata family share of prefill kernel time | prefix + tile-map + two average fills optimistic bound |
| --- | ---: | ---: |
| 512/128 | `0.84%` | `0.27%` kernel time / `0.22%` wall time |
| 4K/128 | `0.77%` | `0.12%` kernel time / `0.12%` wall time |
| 32K/128 | `0.57%` | `0.09%` kernel time / `0.09%` wall time |

Decision: defer/no-op. P3.3 should not add another diagnostic kernel for the
current c=1 prefill path. Revisit only if c>N batching or future scheduler
profiling makes MoE metadata a multi-percent prefill bucket.

Current fallback correctness/sanity was rechecked without code changes:

```bash
python3 scripts/smoke.py --mode qwen35-moe-group-scatter-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
# tokens=3 top_k=2 num_experts=4 hidden_size=5 prefix_match=True lane_match=True \
# expert_match=True weight_match=True packed_match=True tile_match=True

python3 -m pytest tests/test_qwen35_moe_group_scatter_plan.py -q --tb=short
# 3 passed

python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-p33-moe-metadata-fanout-deferred.json >/tmp/p33-artifact-check.json
```

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-p33-moe-metadata-fanout-deferred.json`.

---

## 2026-05-17 — P5.2 long-context chunk-size autotuner retained

Task #21 evaluated `docs/OPTIMIZE.md` P5.2. I replaced the static long-prefill
chunk defaults with a default-on, memory-budget-aware resolver in
`PrefillConfig`: manual non-zero chunk sizes still override, prompts below 32K
stay unchunked, 32K resolves to the retained static `1024/1024/4096/1024/1024`
policy, and ≥128K raises only `full_attn_query_chunk_size` to `8192` when the
budget is at least `24.5 GiB`. The default budget is derived as 55% of device
VRAM from `hipMemGetInfo`; callers can set `chunk_tune_memory_budget_gib` or use
`--no-prefill-chunk-autotune` for diagnostics.

Candidate sweep on W7900/gfx1100, Qwen3.5-35B-A3B-PARO, `w4_paro`, max_layers=40,
cache-only builds, AOTriton threshold 512, graph replay decode, warmup 1:

| workload | candidate | chunks `(linear, moe, q, post, rope)` | prefill tok/s | decode tok/s | peak GiB |
| --- | --- | --- | ---: | ---: | ---: |
| 32K/128 | static | `(1024,1024,4096,1024,1024)` | `1983.834` | `100.476` | `20.320` |
| 32K/128 | q8192 | `(1024,1024,8192,1024,1024)` | `1964.969` | `99.885` | `21.624` |
| 32K/128 | all2048/q8192 | `(2048,2048,8192,2048,2048)` | `1923.982` | `99.183` | `21.804` |
| 32K/128 | all512/q4096 | `(512,512,4096,512,512)` | `1890.300` | `98.927` | `20.230` |
| 128K/128 | static | `(1024,1024,4096,1024,1024)` | `1013.420` | `63.238` | `23.288` |
| 128K/128 | q6144 | `(1024,1024,6144,1024,1024)` | `1020.364` | `63.151` | `23.938` |
| 128K/128 | q8192 | `(1024,1024,8192,1024,1024)` | `1050.368` | `63.368` | `24.592` |
| 128K/128 | q12288 | `(1024,1024,12288,1024,1024)` | `1033.352` | `63.182` | `25.894` |
| 128K/128 | q16384 | `(1024,1024,16384,1024,1024)` | `1040.739` | `63.181` | `27.201` |
| 128K/128 | q32768 | `(1024,1024,32768,1024,1024)` | `1022.645` | `63.125` | `32.419` |

Post-implementation A/B used the same harness. Short/mid contexts keep zero
chunks, so their differences are run-to-run noise; 32K auto and static use the
same chunks; 128K auto selects q8192 and improves prefill while staying under
the derived `24.74 GiB` budget:

| workload | baseline | auto | prefill Δ | decode Δ | peak Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 no-auto -> auto | `2179.935` | `2188.194` | `+0.38%` | `+0.41%` | `+0.000 GiB` |
| 4K/128 no-auto -> auto | `2434.652` | `2453.793` | `+0.79%` | `+0.36%` | `+0.000 GiB` |
| 32K/128 static -> auto | `1937.989` | `1950.955` | `+0.67%` | `+1.24%` | `+0.000 GiB` |
| 128K/128 static -> auto | `1017.796` | `1042.600` | `+2.44%` | `-0.40%` | `+1.304 GiB` |

Decision: accept P5.2 as a default auto policy. This is a budget-aware
performance tune, not a memory-saving tune: 128K spends ~`+1.30 GiB` tracked
peak to recover `+2.44%` prefill, while 512/4K/32K do not regress because their
resolved chunks are unchanged from the intended policies.

Correctness/validation:

```bash
python3 -m py_compile hipengine/runtime/prefill.py hipengine/runtime/qwen35_paro_runner.py \
  scripts/qwen35_paro_bench.py scripts/qwen35_native_prefill_fixture_gate.py \
  scripts/qwen35_decode_graph_fixture_gate.py scripts/qwen35_rocprof_audit.py
python3 -m pytest tests/test_qwen35_resident_batch_layout.py -q --tb=short
# 27 passed
python3 -m pytest tests/test_qwen35_resident_batch_layout.py tests/test_qwen35_paro_layout.py -q --tb=short
# 47 passed

python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --max-layers 40 --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p52-chunk-tuner-20260517-post/auto-native-prefill-fixture-gate.json
# passed=True, max_kl=0.039568870612619614, top1_agreement=1.0, generated_match=True

python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --max-layers 40 --attn-aotriton-min-tokens 512 \
  --prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 \
  --prefill-full-attn-query-chunk-size 8192 --prefill-full-attn-post-chunk-size 1024 \
  --prefill-full-attn-rope-chunk-size 1024 \
  --json /tmp/hipengine-p52-chunk-tuner-20260517-post/q8192-native-prefill-fixture-gate.json
# passed=True, max_kl=0.039568870612619614, top1_agreement=1.0, generated_match=True

python3 scripts/qwen35_decode_graph_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-p52-chunk-tuner-20260517-post/auto-decode-graph-fixture-gate.json
# passed=True, final_kl=0.0, generated_match=True

python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-p52-prefill-chunk-autotune-accepted.json >/tmp/p52-artifact-check.json
```

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-p52-prefill-chunk-autotune-accepted.json`.

---

## 2026-05-17 — D1.2 RMSNorm/add-RMSNorm producer fusion deferred

Task #22 evaluated `docs/OPTIMIZE.md` D1.2. I did **not** implement a new
producer-into-projection kernel because the M.4 decode profile and the current
Qwen3.5/PARO dataflow do not expose a material single-use RMSNorm/add-RMSNorm
producer while preserving the pack8/repacked projection layout.

M.4 selected-region decode evidence (`benchmarks/results/2026-05-17-hipengine-qwen35-rocprof-amdahl-diagnostic.json`):

| workload | RMSNorm bucket share | calls/token | ms/token |
| --- | ---: | ---: | ---: |
| 512/128 | `3.303%` | `91` | `0.239940` |
| 4K/128 | `3.365%` | `91` | `0.243179` |
| 32K/128 | `2.968%` | `91` | `0.255358` |

Breakdown from the same M.4 top-kernel rows:

| producer family | calls/token | avg us/call at 512 | dataflow result |
| --- | ---: | ---: | --- |
| `paro_rmsnorm_out` | `41` | `2.525` | 40 input layernorms feed multiple attention projections; one final norm feeds lm-head |
| `paro_add_rmsnorm_out` | `40` | `2.657` | post-attention MLP input fans out to router, selected-MoE, shared expert, and residual combine |
| head RMSNorm+RoPE | `10` | `3.015` | already a fused full-attention helper; not a `paro_rmsnorm`/`add_rmsnorm` producer |

Static dataflow conclusion:

- Input RMSNorm is not single-use: linear-attention layers consume it through
  QKV/Z rotate+pack8 plus dense A/B, while full-attention layers consume it
  through Q/K/V rotations/projections.
- Post-attention add-RMSNorm is not single-use: `mlp_input` feeds router,
  selected-MoE gate/up, and shared-expert gate/up; residual output feeds combine.
- The only clear single-use producer is final RMSNorm -> lm-head, but that is
  one tiny `paro_rmsnorm_out` call per token (~`0.04%` of 512/128 kernel time).
  Folding it into W8A16 lm-head would need a new row-staged design to avoid
  recomputing RMS per vocab tile.

Decision: defer/no-op. Revisit only after D1.3/D1.6 if there is a row-staged
multi-consumer pack8/W8A16 design that stages RMS once per row without the D1.1
rotate-staged barrier regression. Defaults remain unchanged.

Validation (no math/runtime code changed):

```bash
python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
# hip OK
python3 scripts/smoke.py --mode paro-rmsnorm-hip --rows 2 --hidden-size 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt
# BF16/FP16 norm/add_norm/residual mismatches all 0
python3 -m pytest tests/test_qwen35_rmsnorm_plan.py tests/test_qwen35_paro_layout.py -q --tb=short
# 23 passed
python3 -m pytest tests/test_qwen35_decode_state.py -q --tb=short
# 49 passed
python3 -m py_compile scripts/qwen35_rocprof_audit.py scripts/smoke.py
python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-d12-rmsnorm-producer-fusion-deferred.json >/tmp/d12.json
```

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-d12-rmsnorm-producer-fusion-deferred.json`.

---

## 2026-05-17 — D1.3 same-input c=1 projection fusions rejected/no-op

Task #23 evaluated `docs/OPTIMIZE.md` D1.3. I did **not** implement a new
same-input projection fusion because static decode inventory plus the M.4 profile
show no standalone D1.3 candidate with arithmetic/data-reuse upside while
preserving pack8/repacked layouts.

M.4 selected-region decode profile (`benchmarks/results/2026-05-17-hipengine-qwen35-rocprof-amdahl-diagnostic.json`):

| workload | W4 single GEMV share | W4 single calls/token | W4 dual GEMV share | W4 dual calls/token |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | `13.379%` | `50` | `11.812%` | `40` |
| 4K/128 | `13.557%` | `50` | `11.710%` | `40` |
| 32K/128 | `11.592%` | `50` | `9.963%` | `40` |

Static inventory result:

- Already fused and retained in hipEngine:
  - linear-attention `in_proj_qkv + in_proj_z` via `gemv_awq_dual_pack8_transposed_fp16` (`30` layers/token);
  - full-attention `q_proj + k_proj` via `gemv_awq_dual_pack8_transposed_fp16` (`10` layers/token);
  - linear-attention dense `in_proj_a + in_proj_b` via `dense_dual_gemv_out_fp16` (`30` layers/token);
  - selected-MoE `gate + up` via selected dual pack8 (`40` layers/token);
  - shared-expert `gate + up` via packed dual W4 sidecars or legacy precombined W8A16 (`40` layers/token).
- The only material unfused same-input slice is full-attention `v_proj` beside
  the retained Q/K dual path. It accounts for only `10` of the `50` generic W4
  single GEMV calls/token; the remaining single pack8 calls are `o_proj` and
  linear-attention output projections, which have no adjacent same-input peer.
- Down projections and `lm_head` are not D1.3 candidates; they consume post-op
  inputs and belong to other producer/post-op rows.

Parent/source-lineage evidence:

- `/home/lhl/amd-gpu-tuning/docs/PARO.md:1414-1423` reports the full-attention
  triple Q/K/V pack8 prototype was correct (`24/24`) and graph-safe but slower:
  512/128 `116.357` and 4K/128 `107.412` decode tok/s versus the retained Q/K
  path `116.721` and `107.703`.
- The same parent note says the 2026-05-11 graph-stack recheck was only noise
  (`512/128 115.569 vs 115.258`, `4K/128 120.622 vs 120.655`) and diagnostic
  prefill wiring regressed (`~905` tok/s control to `836` Q/K-only and `822`
  Q/K/V).
- `/home/lhl/amd-gpu-tuning/LESSONS-LEARNED.md:38` generalizes the lesson:
  do not widen a fused projection family unless it preserves kernel efficiency
  or adds real data reuse; graph replay makes pure launch-count wins very small.

Decision: reject/no-op for D1.3. Defaults remain unchanged. Any narrower
`k_proj + v_proj` retest belongs to D1.6 and should inherit the parent Q/K/V
rejection as its cautionary baseline.

Validation (no math/runtime code changed):

```bash
python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
# hip OK
python3 -m py_compile hipengine/runtime/qwen35_paro.py scripts/qwen35_rocprof_audit.py
python3 -m pytest tests/test_qwen35_paro_layout.py tests/test_qwen35_decode_state.py -q --tb=short
# 69 tests passed (command exit 0)
python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-d13-same-input-projection-fusions-rejected.json >/tmp/d13.json
```

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-d13-same-input-projection-fusions-rejected.json`.

---

## 2026-05-17 — D1.6 decode K/V dual-pack8 route rejected as default

Task #24 evaluated `docs/OPTIMIZE.md` D1.6. I implemented an opt-in diagnostic
route, but did **not** promote it to default.

Implementation:

- Added `HIPENGINE_PARO_FULL_ATTN_KV_PACK8_FUSED=1` (default off).
- Default full-attention decode projection route remains:
  `rotate3 -> dual pack8 q_proj+k_proj -> single v_proj`.
- Opt-in route is:
  `rotate3 -> single q_proj -> dual pack8 k_proj+v_proj`.
- The opt-in scratch aliases key and value into a contiguous `attn.kv_proj`
  buffer so the existing dual-pack8 wrapper writes `K||V` without copy kernels.
- Pack8/repacked qweight layout is preserved; no HIP kernel body changed.
- The D1.1 rotate-staged full-attn path is disabled when K/V fusion is enabled,
  because rotate-staged only fills V scratch before the fused Q/K kernel.

Correctness:

```bash
HIPENGINE_PARO_FULL_ATTN_KV_PACK8_FUSED=1 python3 scripts/qwen35_decode_graph_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --max-new-tokens 16 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-d16/graph-fixture-optin.json
# passed=True, generated_match=True, expected_match=True, final_kl=0.0, final_top1_match=True
```

Graph replay benchmark (single run per A/B point, resident runner, max_layers=40):

```bash
python3 scripts/qwen35_paro_bench.py \
  --prompt-length {512,4096} --decode-tokens 128 --warmup-decode-tokens 4 --token-id 9707 \
  --graph-replay-decode --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --attn-aotriton-min-tokens 512 --json <path>
# opt-in adds HIPENGINE_PARO_FULL_ATTN_KV_PACK8_FUSED=1
```

| workload | route | prefill tok/s | decode tok/s | tracked peak GiB |
| --- | --- | ---: | ---: | ---: |
| 512/128 | default Q/K+V | `2276.276` | `115.495` | `18.175625` |
| 512/128 | opt-in Q+K/V | `2244.508` | `115.627` | `18.176122` |
| 4K/128 | default Q/K+V | `2487.220` | `117.301` | `20.047133` |
| 4K/128 | opt-in Q+K/V | `2487.114` | `117.053` | `20.051049` |

Deltas:

- 512/128 decode: `+0.11%` (noise), prefill `-1.40%`, peak `+0.0005 GiB`.
- 4K/128 decode: `-0.21%`, prefill ~neutral, peak `+0.0039 GiB`.

Decision: reject as default. D1.6 changes projection pairing, not launch count:
default Q/K+V and opt-in Q+K/V both use two projection launches per
full-attention layer. The hoped-for benefit from running Q as a single-projection
path and pairing the two small KV projections does not survive graph replay.
`HIPENGINE_PARO_FULL_ATTN_KV_PACK8_FUSED=1` remains only as a diagnostic surface;
default runtime behavior is unchanged.

Validation:

```bash
python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
# hip OK
python3 -m py_compile hipengine/runtime/qwen35_paro.py scripts/qwen35_paro_bench.py scripts/qwen35_decode_graph_fixture_gate.py
python3 -m pytest tests/test_qwen35_decode_state.py tests/test_qwen35_paro_layout.py -q --tb=short
# 71 passed
python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-d16-kv-pack8-fusion-rejected.json >/tmp/d16.json
```

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-d16-kv-pack8-fusion-rejected.json`.

---

## 2026-05-17 — D4.2 dispatch/token reduction plan rejected/no-op

Task #25 evaluated `docs/OPTIMIZE.md` D4.2 using the M.4 decode
Amdahl/dispatch profile plus the closed D1.1-D1.6 decisions. I did **not**
implement a new batched fusion or graph rewrite.

Baseline / target math:

- M.4 selected-region graph replay reports `877 dispatches/token` at
  512/128, 4K/128, and 32K/128.
- The D4.2 cap `<700 dispatches/token` therefore needs at least `178` fewer
  dispatches/token (`20.3%` of dispatches).
- Kernel time/token in the same profile is `7.265 ms` at 512, `7.226 ms` at
  4K, and `8.603 ms` at 32K; this task is a dispatch-plan decision, not a new
  throughput claim.

D1 ledger for D4.2:

| row | usable dispatch-count reduction | evidence |
| --- | ---: | --- |
| D1.1 rotate-staged dual pack8 | rejected; countable piece is only ~30/tok for linear-attn rotate2 | opt-in regressed 512/128 graph decode `115.450 -> 110.457 tok/s` (`-4.32%`) |
| D1.2 RMSNorm/add-RMSNorm producer fusion | `0` | no material single-use producer; input/add RMSNorm are multi-consumer fanout, final RMSNorm -> lm-head is ~`0.04%` kernel-time upper bound |
| D1.3 same-input projection sweep | `0` | material pairs are already fused; only full-attn V remains, and parent full Q/K/V widening was slower/no-win |
| D1.4 selected-MoE post-op fold | `0` for safe path | safe combine fold is already default; direct selected-down+combine could remove 40/tok but parent microbench regressed `13.38 -> 16.52 us` |
| D1.5 router cooperative fold | `0` in current implementation | logits+select becomes counter memset+cooperative kernel; graph decode regressed `-0.93%` at 512 and `-0.67%` at 4K |
| D1.6 full-attn K/V dual pack8 | `0` | changes Q/K+V to Q+K/V, so projection launch count stays two per full-attn layer; 4K decode regressed `-0.21%` |

Scenario accounting:

- Accepted safe D1 changes remove `0`, leaving `877 dispatches/token`.
- Counting rejected D1.1 plus an ideal no-memset router removes only ~`70`,
  leaving ~`807 dispatches/token`.
- Adding the parent-rejected direct selected-combine shape removes only ~`110`,
  leaving ~`767 dispatches/token`, still above the `<700` target.
- Crossing `<700` would require fusing multi-consumer RMSNorm/rotation/W4/MoE
  families, a multi-layer/megakernel schedule, or scheduler-level graph
  compaction. That is outside the D1.1-D1.6 batched-fusion plan and currently
  lacks the required real data-flow evidence.

Decision: reject/no-op for D4.2 as written. Defaults remain unchanged. Reopen
only as a new major data-flow / graph-compaction design with fresh
`dispatches/token` profiling before implementation.

Validation:

```bash
python3 /tmp/create_d42_artifact.py
python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-d42-dispatch-cap-rejected.json >/tmp/d42.json
python3 - <<'PY'
import json
from pathlib import Path
p = Path('benchmarks/results/2026-05-17-hipengine-qwen35-d42-dispatch-cap-rejected.json')
data = json.loads(p.read_text())
assert data['dispatch_cap_math']['baseline_dispatches_per_token'] == 877.0
assert data['dispatch_cap_math']['minimum_dispatches_to_remove'] == 178.0
assert data['dispatch_scenarios']['include_direct_selected_combine_too']['projected_dispatches_per_token'] == 767.0
assert data['decision']['default_changed'] is False
PY
```

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-d42-dispatch-cap-rejected.json`.

---

## 2026-05-17 — D4.4 launch_bounds retune deferred/no-op

Task #26 evaluated `docs/OPTIMIZE.md` D4.4. I did **not** retune any
`__launch_bounds__` or thread-count defaults.

Reasoning:

- D4.4 was scoped to retuning after retained rotation/RMSNorm/W4 fusion changes.
  The D1.1-D1.6 sweep did not retain a default fusion that changes the default
  kernel resource envelope: D1.1 rotate-staged regressed, D1.2 deferred/no-op,
  D1.3 rejected/no-op, and D1.6 is launch-count neutral and rejected as default.
- D4.2 also rejected the stacked D1 dispatch plan, so there is no new batched
  fusion surface whose source bounds need retuning.
- Static source/wrapper audit confirms current launch sites do not bypass source
  bounds:
  - pack8/selected pack8 and Marlin-K launch at `32/64/128` or `64/128` threads
    under `__launch_bounds__(128,4)`;
  - compact WMMA and fusedW4 prefill launch at `32` threads under
    `__launch_bounds__(32,*)`.
- Existing evidence does not identify a safe retune:
  - D1.1 rotate-staged and D2.1 Marlin-K traces both show workgroup `128`,
    `VGPR=104`, scratch `0`, LDS `512`; rotate-staged regressed and remains
    opt-in/off, while Marlin-K is already the retained default.
  - D5.2 W8A16 thread probes found current decode thread choices best
    (`lm_head=128`, shared lowp `64`) and larger workgroups regressed.
  - Earlier WORKLOG launch-bound trials rejected compact-WMMA
    `__launch_bounds__(32,2)->(32,4)` and fusedW4 prefill
    `__launch_bounds__(32,8)->(32,16)` because they regressed/spilled.
- Per the project boundary, fresh kernel micro-tuning loops belong in
  `~/amd-gpu-tuning/`; hipEngine should port only stable parent evidence.

Decision: defer/no-op. Defaults remain unchanged. Reopen only after a default
fusion is retained or parent kernel R&D produces a source-level launch-bound
retune with resource metadata and throughput evidence.

Validation:

```bash
python3 /tmp/create_d44_artifact.py
python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-d44-launch-bounds-deferred.json >/tmp/d44.json
python3 -m py_compile hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.py hipengine/kernels/hip_gfx1100/quant/paro_marlin_k.py hipengine/runtime/qwen35_paro.py
```

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-d44-launch-bounds-deferred.json`.

---

## 2026-05-17 — D5.1 GDN recurrent decode audit stop/no-op

Task #27 evaluated `docs/OPTIMIZE.md` D5.1 for
`qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel` vec8 / occupancy headroom on
c=1 decode. I added a retained diagnostic probe script,
`scripts/gdn_decode_probe.py`, and did **not** change any kernel/runtime default.

Findings:

- M.4 makes GDN decode visible but bounded: `linear_attention_gdn_decode` is
  `5.23%/5.39%/4.84%` of decode kernel time at 512/4K/32K with
  `30` calls/token.
- Static source audit shows the requested local vec8 pattern is already present:
  Q/K RMS load uses `idx = threadIdx.x * 8` / `idx += blockDim.x * 8`, and both
  KV-memory accumulation plus recurrent-state update are 8-way unrolled over
  `head_k_dim`.
- For the real Qwen3.5 c=1 shape (`num_k_heads=16`, `num_v_heads=32`,
  `head_k_dim=128`, `head_v_dim=128`), the wrapper launches 128 threads and the
  kernel maps `value_idx = threadIdx.x`, exactly one value lane per thread. A
  64-thread retune would miss lanes without a new ownership scheme; 256 threads
  add idle lanes and reduction overhead.
- The wrapper's dynamic shared formula is
  `(2 * head_k_dim + 3 * threads + head_v_dim) * sizeof(float)`, i.e. `3072 B`
  for the actual shape. `rocprofv3` reports `LDS_Block_Size=0` for this dynamic
  launch, so the artifact records the wrapper-computed value separately.
- Barrier removal or multi-value/thread recurrence rewrites cross into kernel
  R&D and need a new correctness proof; prior GDN barrier-removal attempts are a
  documented do-not-chase item because they corrupted recurrent state.

Profiler evidence (W7900/gfx1100, cache-only build, `reps=100`, first four
warmups dropped):

```bash
rocprofv3 --kernel-trace --output-format csv \
  --output-directory /tmp/hipengine-d51-gdn-probe \
  --output-file gdn_decode_probe -- \
  python3 scripts/gdn_decode_probe.py \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build --reps 100 --warmup 4
```

Target kernel rows:

- BF16 lowp: median `8760 ns`, mean `8803 ns`, min/max `8600/10681 ns`,
  `VGPR_Count=56`, `Scratch_Size=0`, `SGPR_Count=128`, workgroup `128`, grid
  work-items `4096`.
- FP16 lowp: median `8720 ns`, mean `8747 ns`, min/max `8600/10360 ns`,
  `VGPR_Count=56`, `Scratch_Size=0`, `SGPR_Count=128`, workgroup `128`, grid
  work-items `4096`.

Correctness/validation:

```bash
python3 -m py_compile scripts/gdn_decode_probe.py
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.linear_attn import build_qwen35_linear_attn_gdn
cv = Path('/tmp/hipengine-hipcc-version.txt').read_text()
build_qwen35_linear_attn_gdn(load=True, compiler_version=cv, require_cached=False)
print('gdn build OK')
PY
python3 scripts/gdn_decode_probe.py --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --reps 2 --warmup 1
python3 scripts/smoke.py --mode qwen35-linear-attn-gdn-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
# out_max_abs=2.98e-08, state_max_abs=1.49e-08; FP16 same
python3 -m pytest tests/test_qwen35_linear_attn_gdn_plan.py -q --tb=short
# 3 passed
python3 scripts/qwen35_decode_graph_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --max-new-tokens 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --json /tmp/hipengine-d51-gdn/graph-fixture-default.json
# passed=True, generated_match=True, expected_match=True, final_kl=0.0
python3 -m json.tool benchmarks/results/2026-05-17-hipengine-qwen35-d51-gdn-decode-audit.json >/tmp/d51.json
```

Decision: accepted stop/no-op. Keep the current GDN recurrent decode kernel.
Reopen only if parent kernel R&D produces a correct multi-value/thread or
barrier-reduced GDN design with resource metadata and E2E decode improvement, or
if a future M.4-style profile shows GDN growing materially after other lanes
change.

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-d51-gdn-decode-audit.json`.

2026-05-17 — Added optional FastAPI / OpenAI-compatible server layer for v0.1.

Implemented `hipengine.server` as an optional `[server]` surface that adapts
OpenAI-style requests to the existing torch-free `LLM.generate()` API. The app
factory is `hipengine.server.create_app(ServerConfig(...))`; the CLI is
`python -m hipengine.server` / `hipengine-server`. Endpoints landed:

- `GET /health`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

Current behavior/limits are explicit: single-process requests are serialized
behind an async lock because the runnable runtime is still c=1; streaming is a
compatibility one-chunk SSE plus `[DONE]`; `n>1`, `logprobs`, and non-text chat
content parts are rejected; token `usage` is exact only for an injected engine
with `count_tokens`, otherwise the public `LLM` path returns zero placeholders
until tokenizer accounting is exposed.

Docs updated: root README server quickstart, new `docs/API.md`, docs index, and
`docs/IMPLEMENTATION.md` Phase 1 server checkbox. Packaging updated with
`hipengine-server` console script and dev extra server-test deps.

Validation:

```bash
python3 -m py_compile hipengine/server/api.py hipengine/server/__init__.py hipengine/server/__main__.py
python3 -m pytest tests/test_server_api.py tests/test_llm_generate.py tests/test_model_quant_and_imports.py -q --tb=short
# 15 passed
python3 -m hipengine.server --help
python3 - <<'PY'
from pathlib import Path
import re, urllib.parse, sys
fail=[]
for md in [Path('README.md'), Path('docs/README.md'), Path('docs/API.md')]:
    text=md.read_text()
    for m in re.finditer(r'\[[^\]]+\]\(([^)]+)\)', text):
        target=m.group(1).split('#',1)[0]
        if not target or '://' in target or target.startswith('mailto:'):
            continue
        path=(md.parent / urllib.parse.unquote(target)).resolve()
        if not path.exists():
            fail.append((str(md), target))
if fail:
    raise SystemExit(fail)
print('markdown links OK')
PY
python3 -m pytest -q
# full suite passed
```

## 2026-05-17 — Initial gfx1151 backend port

Ported the current gfx11 Qwen3.5/PARO implementation to a Strix Halo backend key without forking kernel bodies.

### Scope

- Reviewed `/home/lhl/github/shisa-ai/amd-gpu-tuning/docs/ROOFLINE-gfx1151.md`; relevant constraints are 40 CUs, native `gfx1151` code objects, lower LPDDR bandwidth, and no assumption that W7900 wrapper defaults are optimal.
- Fixed `docs/source_lineage.json` external paths from `/home/lhl/amd-gpu-tuning/...` to `/home/lhl/github/shisa-ai/amd-gpu-tuning/...`, then ran:
  `python3 scripts/check_lineage.py --kind kernel --diff stat`
  - Result: expected DRIFT vs baseline `22405a9` in parent `qwen35_expert.hip`, `smoke.hip`, `paroquant_kernels.py`, and `paroquant_fusedw4.py`; no new kernel bodies were copied for this gfx1151 port.
- Added explicit HIP target-arch plumbing in `hipengine/core/build.py`:
  - `target_arch=` API and `HIPENGINE_HIP_ARCH` / `HIPENGINE_HIP_OFFLOAD_ARCH` env default.
  - Emits `--offload-arch=gfx1151` and includes it in the build-cache key.
  - Converts `HIPENGINE_ROCM_DEVICE_LIB_PATH` / `HIP_DEVICE_LIB_PATH` into explicit `--rocm-device-lib-path=...`, also included in the cache key. This avoids reusing artifacts built with a mismatched ROCm device-lib path.
- Added `hipengine/kernels/backends.py` for backend→native-arch metadata.
- Added `hipengine/kernels/hip_gfx1151/__init__.py` to register `hip_gfx1151` aliases for the current `hip_gfx1100` kernel key space. This is a bring-up baseline using the same proven kernel bodies compiled as native `gfx1151`, not a tuning claim.
- Wired Qwen3.5/PARO generation and benchmark entrypoints:
  - `hipengine.generation.qwen35_paro` now registers `backend="hip_gfx1151"` for `w4_paro`.
  - `Qwen35ParoNextTokenRunner` carries `backend` and `target_arch`; resident session library builds run under the backend target-arch environment.
  - `scripts/qwen35_paro_bench.py`, `scripts/smoke.py --mode qwen35-paro-generate-hip`, and `scripts/qwen35_paro_next_token.py` accept `--backend hip_gfx1151`.
- AOTriton note: current vendored 0.11.2b payload is the pruned `amd-gfx11xx` forward-attention image set. If we need additional gfx1151-specific AOTriton images later, use the existing Git-LFS vendor/fetch flow rather than runtime downloads.
- Added `scripts/__init__.py` so pytest imports repo scripts instead of the unrelated third-party `scripts` package in this Python environment.

### Validation

ROCm visibility:

```bash
python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
rocminfo | grep -E 'Name:|gfx' | head -n 80
```

Result: HIP loads; visible GPU is `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`, target `gfx1151`.

Build dry-run shows native arch in command:

```bash
HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
  python3 scripts/smoke.py --mode smoke-add-plan
```

Result command includes `--offload-arch=gfx1151 --rocm-device-lib-path=/opt/rocm/amdgcn/bitcode`; output cache key `smoke-8106da2da6b3d257`.

Small gfx1151 GPU smokes:

```bash
HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
  python3 scripts/smoke.py --mode smoke-add-hip --n 1024
# n=1024 max_abs=0.0

HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
  python3 scripts/smoke.py --mode qwen35-rmsnorm-hip --rows 2 --hidden-size 16
# rows=2 hidden_size=16 max_abs=0.0 bit_mismatch=0
```

Profiler smoke with cached build:

```bash
hipcc --version > /tmp/hipengine-gfx1151-hipcc-version.txt
HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
  python3 scripts/smoke.py --mode qwen35-rmsnorm-hip --rows 2 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt --require-cached-build
HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
  rocprofv3 --kernel-trace -f csv -o /tmp/hipengine-gfx1151-rmsnorm-trace -- \
    python3 scripts/smoke.py --mode qwen35-rmsnorm-hip --rows 2 --hidden-size 16 \
      --compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt --require-cached-build
```

Trace file `/tmp/hipengine-gfx1151-rmsnorm-trace_kernel_trace.csv` shows expected kernel:
`(anonymous namespace)::qwen35_rmsnorm_kernel(...)`, `End_Timestamp - Start_Timestamp = 4088 ns`.

Test suite:

```bash
python3 -m py_compile hipengine/core/build.py hipengine/kernels/backends.py \
  hipengine/kernels/hip_gfx1151/__init__.py hipengine/runtime/qwen35_paro_runner.py \
  hipengine/generation/qwen35_paro.py scripts/qwen35_paro_bench.py scripts/smoke.py \
  scripts/qwen35_paro_next_token.py tests/test_gfx1151_backend.py
python3 -m pytest tests/test_build.py tests/test_gfx1151_backend.py tests/test_llm_generate.py -q --tb=short
# 13 passed
python3 -m pytest -q --tb=short
# all tests passed
```

### Notes

- First raw build without a ROCm device-lib path failed with `cannot find ROCm device library`; building with a mismatched mambaforge device-lib path produced a smoke-launch segfault. Retained validation uses `/opt/rocm/amdgcn/bitcode`, and the build cache now keys explicit `--rocm-device-lib-path` so these artifacts do not collide.
- No throughput benchmark was run or claimed in this entry; this is an initial correctness/build/backend port.

## 2026-05-17 — Dense Qwen3.5 0.8B PARO bring-up on gfx1151

Added the missing dense-text PARO path needed by `z-lab/Qwen3.5-0.8B-PARO` (`qwen3_5_text`, `num_experts=0`, hidden 1024, 24 layers) so the gfx1151 backend can run the model instead of requiring the 35B MoE layout.

### Implementation

- Added dense Qwen3.5/PARO layout/runtime materialization helpers for linear-attention and full-attention layers:
  - `runtime_linear_attention_dense_c1_tensor_names`
  - `runtime_full_attention_dense_c1_tensor_names`
  - dense validators and runtime materializers
  - dense MLP `gate_proj`/`up_proj`/`down_proj` `qweight_pack8_decode` preparation.
- Added resident runtime dense MLP scratch and FP16 dense PARO MLP execution by reusing the existing PARO rotate/AWQ/SwiGLU kernels, then residual-adds the MLP output with the existing combine kernel using a zero shared branch.
- Resident runner now dispatches dense MLP when `num_experts <= 0` and MoE otherwise.
- Added tied-lm-head fallback: if `lm_head.weight` is absent, resident and one-token paths use `embed_tokens.weight` for the output head. This is required by the 0.8B snapshot.
- Full-attention decode now uses the generic split-K gated kernel when the parent Qwen3.5 GQA specialization shape (`16q/2kv/head256`) does not match; this unblocks 0.8B (`8q/2kv/head256`) at 4K+ decode lengths.

### Validation

```bash
python3 - <<'PY'
from pathlib import Path
from hipengine.loading import load_weight_index
from hipengine.loading.qwen35_paro import validate_qwen35_paro_linear_attention_dense_c1_layout, validate_qwen35_paro_full_attention_dense_c1_layout
model=Path('/models/huggingface/hub/models--z-lab--Qwen3.5-0.8B-PARO/snapshots/da941f4fd3fa72763c398db6cb14b2bef1ee961f')
idx=load_weight_index(model)
print(validate_qwen35_paro_linear_attention_dense_c1_layout(idx, layer_id=0).passed)
print(validate_qwen35_paro_full_attention_dense_c1_layout(idx, layer_id=3).passed)
PY
# True / True

HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
  python3 scripts/qwen35_paro_bench.py \
    --model /models/huggingface/hub/models--z-lab--Qwen3.5-0.8B-PARO/snapshots/da941f4fd3fa72763c398db6cb14b2bef1ee961f \
    --backend hip_gfx1151 --prompt-length 8 --decode-tokens 1 --warmup-decode-tokens 0 \
    --max-layers 1 --attn-aotriton-min-tokens 512 \
    --compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt \
    --json /tmp/qwen35-08b-gfx1151-smoke.json
# max_layers=1 smoke completed with finite generated_preview logits.

python3 -m pytest tests/test_qwen35_resident_batch_layout.py::test_qwen35_resident_linear_prefill_restores_decode_scratch_token1 tests/test_qwen35_paro_layout.py -q --tb=short
# 24 passed
python3 -m pytest -q --tb=short
# 262 passed
```

### Diagnostic timing probes before commit

Single-run diagnostic probes on the shared/busy gfx1151 GPU (not retained as accepted perf claims; no KL/top-1 CPU oracle yet):

- 512/128: prefill `1172.036 tok/s`, decode `135.650 tok/s`, tracked peak `1.322 GiB`, hipMemGetInfo peak `1.445 GiB`.
- 4K/128: prefill `1929.462 tok/s`, decode `135.915 tok/s`, tracked peak `2.214 GiB`, hipMemGetInfo peak `1.665 GiB`.
- 4K/4K: prefill `1897.360 tok/s`, decode `128.785 tok/s`, tracked peak `2.280 GiB`, hipMemGetInfo peak `1.736 GiB`.

Exact retained post-commit artifact/rollup will follow as a separate benchmark evidence commit so the benchmark can record a clean `hipengine_commit`.

## 2026-05-17 — Qwen3.5-0.8B-PARO gfx1151 diagnostic benchmark

Ran post-commit dense 0.8B diagnostic benchmarks on clean commit `35e61cff95fe81e6f9aee39f5b18bb324aa9bd47`.

### Environment

- GPU: `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`, target `gfx1151` (`rocminfo` also exposes `amdgcn-amd-amdhsa--gfx11-generic`).
- HIP compiler: `HIP version: 7.2.53211-e1a6bc5663`, ROCm SDK clang from `/home/lhl/mambaforge/lib/python3.11/site-packages/_rocm_sdk_core/lib/llvm/bin`.
- Model: `/models/huggingface/hub/models--z-lab--Qwen3.5-0.8B-PARO/snapshots/da941f4fd3fa72763c398db6cb14b2bef1ee961f`.
- Env used for benchmark commands: `HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151`.

### Results

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-08b-gfx1151-dense-diagnostic.json`.

These are **diagnostic only** (`performance_claim=false`): dense layout validation passed, `pytest -q` passed before the code commit, and generated logits were finite, but there is no committed KL/top-1 CPU-reference oracle for the dense 0.8B path yet. The gfx1151 GPU was shared/busy, so rows are single-run.

| Workload | Decode mode | Prefill tok/s | Decode tok/s | Tracked peak | hipMemGetInfo sampled peak |
| --- | --- | ---: | ---: | ---: | ---: |
| 512/128 | eager (`--no-graph-replay-decode`) | 703.238 | 122.141 | 1.322 GiB | 1.445 GiB |
| 4K/128 | eager (`--no-graph-replay-decode`) | 1556.392 | 113.403 | 2.214 GiB | 1.665 GiB |
| 4K/4K | eager (`--no-graph-replay-decode`) | 1969.160 | 105.103 | 2.280 GiB | 1.735 GiB |

Exact command template:

```bash
HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
  python3 scripts/qwen35_paro_bench.py \
    --model /models/huggingface/hub/models--z-lab--Qwen3.5-0.8B-PARO/snapshots/da941f4fd3fa72763c398db6cb14b2bef1ee961f \
    --backend hip_gfx1151 --prompt-length {512|4096} --decode-tokens {128|4096} \
    --warmup-decode-tokens 4 --attn-aotriton-min-tokens 512 --no-graph-replay-decode \
    --compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt --require-cached-build \
    --json /tmp/qwen35-08b-gfx1151-...json
```

### Graph replay note

- 512/128 graph replay completed diagnostically but was slower under this shared-GPU run (`817.879` prefill / `103.403` decode tok/s).
- 4K/1 graph replay smoke completed (`2029.344` prefill / `117.796` decode tok/s).
- 4K/128 graph replay blocked/hung in two clean post-commit attempts (1200s/3600s timeouts, no JSON). Eager decode is the reported path until generic split-K graph replay is debugged.

## 2026-05-17 — gfx1151 large PARO diagnostic benchmarks

Ran the requested follow-up diagnostics for the larger local PARO checkpoints on clean commit `37c61ad3ade38cb0b44ac6d8668ec051e795f83b`.

### Models and environment

- `z-lab/Qwen3.5-35B-A3B-PARO`: `/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd`, MoE config (`40` layers, hidden `2048`, `16q/2kv`, `256` experts, top-k `8`).
- `z-lab/Qwen3.6-27B-PARO`: `/models/huggingface/hub/models--z-lab--Qwen3.6-27B-PARO/snapshots/f0797088d8e0312aac0b5969bec1e6e5c6fb3ff3`, dense config (`64` layers, hidden `5120`, `24q/4kv`, intermediate `17408`).
- GPU: `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`, target `gfx1151`.
- HIP compiler: `HIP version: 7.2.53211-e1a6bc5663`.
- Benchmark env: `HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151`.

### Smoke

Before full rows, both checkpoints passed `--max-layers 1 --prompt-length 8 --decode-tokens 1 --no-graph-replay-decode` smokes with finite generated previews:

- 35B-A3B: prefill `94.787 tok/s`, decode `257.412 tok/s`, tracked peak `1.850 GiB`.
- 27B: prefill `73.513 tok/s`, decode `107.022 tok/s`, tracked peak `3.913 GiB`.

### Results

Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-35b-qwen36-27b-gfx1151-diagnostic.json`.

These rows are **diagnostic only** (`performance_claim=false`): single measured run on shared/busy gfx1151, generated previews finite, but no current KL/top-1 gate was run for these benchmark rows and 27B has no committed oracle fixture yet.

#### Qwen3.5-35B-A3B-PARO

| Workload | Decode mode | Prefill tok/s | Decode tok/s | Tracked peak | hipMemGetInfo sampled peak |
| --- | --- | ---: | ---: | ---: | ---: |
| 512/128 | graph replay | 450.436 | 44.393 | 18.587 GiB | 18.564 GiB |
| 4K/128 | graph replay | 495.460 | 49.458 | 20.458 GiB | 19.034 GiB |
| 4K/4K | eager (`--no-graph-replay-decode`) | 494.467 | 45.522 | 20.572 GiB | 19.152 GiB |

35B 4K/4K graph replay attempt timed out after 3600s with no JSON (`/tmp/qwen35-35b-a3b-paro-gfx1151-4096-4096-graph.stdout` was empty), so the retained 4K/4K row uses eager decode.

#### Qwen3.6-27B-PARO

| Workload | Decode mode | Prefill tok/s | Decode tok/s | Tracked peak | hipMemGetInfo sampled peak |
| --- | --- | ---: | ---: | ---: | ---: |
| 512/128 | eager | 86.568 | 10.385 | 26.484 GiB | 27.258 GiB |
| 4K/128 | eager + 1024-row prefill chunks | 98.190 | 9.724 | 27.223 GiB | 27.692 GiB |
| 4K/4K | eager + 1024-row prefill chunks | 96.988 | 8.777 | 27.557 GiB | 28.032 GiB |

27B unchunked 4K rows aborted with GPU memory-access faults / `HSA_STATUS_ERROR_EXCEPTION`; 1024-row chunks completed for both 4K/128 and 4K/4K.

### Command templates

35B graph rows:

```bash
HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
  python3 scripts/qwen35_paro_bench.py \
    --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
    --backend hip_gfx1151 --prompt-length {512|4096} --decode-tokens 128 \
    --warmup-decode-tokens 4 --attn-aotriton-min-tokens 512 --graph-replay-decode \
    --compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt --require-cached-build \
    --json /tmp/qwen35-35b-a3b-paro-gfx1151-...json
```

27B 4K chunked rows:

```bash
HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
  python3 scripts/qwen35_paro_bench.py \
    --model /models/huggingface/hub/models--z-lab--Qwen3.6-27B-PARO/snapshots/f0797088d8e0312aac0b5969bec1e6e5c6fb3ff3 \
    --backend hip_gfx1151 --prompt-length 4096 --decode-tokens {128|4096} \
    --warmup-decode-tokens 4 --attn-aotriton-min-tokens 512 --no-graph-replay-decode \
    --prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 \
    --prefill-full-attn-query-chunk-size 1024 --prefill-full-attn-post-chunk-size 1024 \
    --prefill-full-attn-rope-chunk-size 1024 \
    --compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt --require-cached-build \
    --json /tmp/qwen36-27b-paro-gfx1151-...json
```

## 2026-05-17 — Merge latest main into gfx1151

Merged `origin/main` (`a679944`) into `gfx1151` to bring the Strix Halo branch up to the latest retained Qwen3.5/Qwen3.6 performance work before canonical gfx1151 benchmarking.

Conflict-resolution notes:

- Kept gfx1151 build plumbing in `hipengine/core/build.py` (`target_arch`, `HIPENGINE_HIP_ARCH` / `HIPENGINE_HIP_OFFLOAD_ARCH`, explicit `--offload-arch`, `HIPENGINE_ROCM_DEVICE_LIB_PATH` / `HIP_DEVICE_LIB_PATH`) and also kept main's diagnostic `HIPENGINE_PREFILL_MCUMODE` build knob.
- Kept `hip_target_arch_environment(self.target_arch)` around resident kernel-library builds and added main's new `marlin_k` library load.
- Combined dense PARO MLP materialization/runtime support from `gfx1151` with main's Marlin-K qweight-neutral materialization and grouped-MoE threshold/default logic.
- Preserved gfx1151 diagnostic benchmark rows/artifacts while taking main's shisa packed, grouped-GQA, chunk autotuner, Marlin-K, and related benchmark rollup entries.
- Kept `docs/source_lineage.json` pointing at the actual local parent workspace `/home/lhl/github/shisa-ai/amd-gpu-tuning/...` while adding main's `PLAN-PAROQUANT2.md` evidence path.

Validation after conflict resolution:

```bash
python3 -m py_compile hipengine/core/build.py hipengine/loading/qwen35_paro.py \
  hipengine/runtime/qwen35_paro.py hipengine/runtime/qwen35_paro_runner.py \
  scripts/qwen35_paro_bench.py scripts/smoke.py
python3 -m pytest tests/test_build.py tests/test_gfx1151_backend.py \
  tests/test_qwen35_paro_layout.py tests/test_qwen35_paro_marlin_k.py \
  tests/test_qwen35_decode_state.py -q --tb=short
python3 -m pytest -q --tb=short
HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
  python3 scripts/smoke.py --mode smoke-add-plan
HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
  python3 scripts/smoke.py --mode smoke-add-hip --n 8
hipcc --version > /tmp/hipengine-gfx1151-hipcc-version.txt
HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
  python3 scripts/smoke.py --mode qwen35-rmsnorm-hip --rows 2 --hidden-size 16 \
    --compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt --require-cached-build
```

Results: targeted pytest passed, full pytest passed, smoke-add plan emitted `--offload-arch=gfx1151 --rocm-device-lib-path=/opt/rocm/amdgcn/bitcode`, `smoke-add-hip --n 8` reported `max_abs=0.0`, and cached gfx1151 RMSNorm smoke reported `max_abs=0.0` / `bit_mismatch=0`.

## 2026-05-17 — gfx1151 TUI shisa Qwen3.6 packed canonical sweep

After rebooting into TUI/headless mode, ran the canonical shisa packed Qwen3.6 comparison sweep on merged `gfx1151` commit `22b0b2eadadf5e8a43a84dfdbaa985a270a0478a`.

### Environment

- GPU: `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`, target `gfx1151`.
- ROCm/HIP: `hipcc` reports `HIP version: 7.13.26154-ca4b97ef2c`; Python `3.12.12`; torch installed but not used by hipEngine hot path (`2.10.0+rocm7.13.0a20260417`).
- hipEngine env: `HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151`.
- Model: `/models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e` (`model.safetensors` 19.068 GiB). HF `refs/main` currently points to `176e57c1a5d823bd0f41605420d04e3441465bb4`, but both local snapshots share the same packed tensor payload.
- GGUF comparison model: `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (file 21 GiB; llama.cpp record `model_size=20.604 GiB`).
- APU memory caveat: `rocm-smi`/sysfs expose only a 512 MiB VRAM aperture. hipEngine tracked allocator peak is meaningful; hipMemGetInfo sampled peaks were negative on this runtime and are not used. llama.cpp sysfs peak sampling is also aperture-limited and not a useful memory peak; the artifact records llama model size instead.

Artifact: `benchmarks/results/2026-05-17-hipengine-gfx1151-shisa-qwen36-packed-canonical-sweep-diagnostic.json`.

Status: **diagnostic retained**, `performance_claim=false`. The max-layer smoke and all benchmark rows produced finite repeated-token previews (`9707` on full rows), but no shisa KL/top-1 E2E gate or repeated-run stats were run yet.

### hipEngine sweep

Common command shape:

```bash
HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
  python3 scripts/qwen35_paro_bench.py \
    --model /models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e \
    --backend hip_gfx1151 --shared-expert-format packed_paro_w4 \
    --token-id 9707 --prompt-length {P} --decode-tokens {D} \
    --warmup-decode-tokens 4 --max-layers 40 --attn-aotriton-min-tokens 512 \
    --graph-replay-decode --compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt \
    --require-cached-build [long-context chunk flags] --json /tmp/hipengine-gfx1151-shisa36-packed-20260517/...
```

Long-context chunk flags for 32K/128 and 128K/128: `--prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 --prefill-full-attn-query-chunk-size 4096 --prefill-full-attn-post-chunk-size 1024 --prefill-full-attn-rope-chunk-size 1024`.

| Workload | Prefill tok/s | Decode tok/s | Tracked peak | Decode seconds | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 512/128 | 881.143 | 61.915 | 18.123 GiB | 2.067 | graph replay, no chunks |
| 4K/128 | 630.585 | 63.364 | 19.995 GiB | 2.020 | graph replay, no chunks |
| 32K/128 | 598.663 | 50.546 | 20.267 GiB | 2.532 | graph replay, parent-style chunks |
| 128K/128 | 371.722 | 30.220 | 23.235 GiB | 4.236 | graph replay, parent-style chunks |
| 4K/4K | 621.551 | 62.245 | 20.108 GiB | 65.805 | graph replay, no chunks |

### llama.cpp HIP GGUF comparison

Command shape via wrapper:

```bash
python3 scripts/llamacpp_bench_with_peak.py \
  --llama-bench /home/lhl/llama.cpp/llama.cpp-lhl/build-gfx1151-unroll600/bin/llama-bench \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --backend hip --workloads 512/128 4K/128 32K/128 128K/128 \
  --poll 10 --repetitions 1 --output /tmp/llamacpp-gfx1151-hip-qwen36-ud-q4km-20260517.json
```

and a separate `4K/4K` wrapper run. Binary reports llama.cpp build `e828394c2` / build number `8975`, backend `ROCm`, GPU `Radeon 8060S Graphics`.

| Workload | Prefill tok/s | Decode tok/s | Memory note |
| --- | ---: | ---: | --- |
| 512/128 | 1039.909 | 51.018 | sysfs peak invalid on 512 MiB aperture; model size 20.604 GiB |
| 4K/128 | 1014.702 | 49.074 | same |
| 32K/128 | 728.519 | 43.474 | same |
| 128K/128 | 376.264 | 31.322 | same |
| 4K/4K | 1003.175 | 49.079 | same |

On this Strix Halo run, hipEngine decode beats local llama.cpp HIP on every matched row, but hipEngine prefill trails llama.cpp at 512/4K/32K and is roughly tied/slightly behind at 128K. Keep as diagnostic until correctness/repetition gates are added.

## 2026-05-17 — upstream llama.cpp HIP/Vulkan gfx1151 rerun

Reran the GGUF side of the Strix Halo comparison after pulling/building the latest upstream llama.cpp trees:

- HIP binary: `/home/lhl/llama.cpp/llama.cpp-hip/build-gfx1151-unroll600/bin/llama-bench` (`libggml-hip`, ROCm gfx1151 libs, binary mtime 2026-05-17 19:48).
- Vulkan binary: `/home/lhl/llama.cpp/llama.cpp-vulkan/build/bin/llama-bench` (`libggml-vulkan`, AMD open-source Vulkan driver, binary mtime 2026-05-17 19:43).
- GGUF model: `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- Wrapper: `scripts/llamacpp_bench_with_peak.py --card-name card1 --poll 10 --repetitions 1`.

Artifact: `benchmarks/results/2026-05-17-llamacpp-upstream-gfx1151-qwen36-gguf-rerun-diagnostic.json`.

Memory note: we intentionally skip memory diagnostics for the compare tables because Strix Halo sysfs/rocm-smi expose only a 512 MiB VRAM aperture; row-level throughput is the relevant comparison.

### Upstream llama.cpp HIP rerun

```bash
python3 scripts/llamacpp_bench_with_peak.py \
  --llama-bench /home/lhl/llama.cpp/llama.cpp-hip/build-gfx1151-unroll600/bin/llama-bench \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --backend hip --workloads 512/128 4K/128 32K/128 128K/128 4K/4K \
  --poll 10 --repetitions 1 --card-name card1 \
  --output /tmp/llamacpp-upstream-gfx1151-qwen36-20260517/llamacpp-upstream-hip-gfx1151-unroll600.json
```

| Workload | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| 512/128 | 1058.738 | 50.537 |
| 4K/128 | 1004.220 | 49.379 |
| 32K/128 | 735.534 | 43.435 |
| 128K/128 | 376.070 | 31.286 |
| 4K/4K | 990.726 | 49.071 |

### Upstream llama.cpp Vulkan rerun

```bash
python3 scripts/llamacpp_bench_with_peak.py \
  --llama-bench /home/lhl/llama.cpp/llama.cpp-vulkan/build/bin/llama-bench \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --backend vulkan --workloads 512/128 4K/128 32K/128 128K/128 4K/4K \
  --poll 10 --repetitions 1 --card-name card1 \
  --output /tmp/llamacpp-upstream-gfx1151-qwen36-20260517/llamacpp-upstream-vulkan-gfx1151.json
```

| Workload | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| 512/128 | 638.008 | 57.615 |
| 4K/128 | 595.400 | 55.027 |
| 32K/128 | 407.984 | 44.576 |
| 128K/128 | 181.453 | 26.935 |
| 4K/4K | 590.391 | 54.241 |

Updated `scripts/qwen35_compare_tables.py` with `shisa-packed-gfx1151`, `llama.cpp-hip-gfx1151`, `llama.cpp-vulkan-gfx1151`, aliases (`--target gfx1151`, `hip-gfx1151`, `vulkan-gfx1151`), and `--no-memory` for throughput-only tables. Example commands:

```bash
python3 scripts/qwen35_compare_tables.py --target gfx1151 hip-gfx1151 --no-memory
python3 scripts/qwen35_compare_tables.py --target gfx1151 vulkan-gfx1151 --no-memory
```

## 2026-05-17 — gfx1151 4K prefill gap diagnosis

Investigated why `hip_gfx1151` shisa packed prefill trails upstream llama.cpp HIP at 4K. Baseline comparison from the retained TUI sweep:

- hipEngine shisa packed 4K/128: `630.585 prefill tok/s` / `63.364 decode tok/s`.
- upstream llama.cpp HIP GGUF 4K/128: `1004.220 prefill tok/s` / `49.379 decode tok/s`.

### Profiling command

Used cached kernels and profiled a 4K/1 run to keep decode noise out:

```bash
HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
rocprofv3 --kernel-trace --memory-copy-trace --stats --summary --summary-units msec \
  --output-directory /tmp/gfx1151-prefill-diagnostics-20260517/profile-full-4k/out \
  --output-file trace --output-format csv -- \
  python3 scripts/qwen35_paro_bench.py \
    --model /models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e \
    --backend hip_gfx1151 --shared-expert-format packed_paro_w4 --token-id 9707 \
    --prompt-length 4096 --decode-tokens 1 --warmup-decode-tokens 0 --max-layers 40 \
    --attn-aotriton-min-tokens 512 --graph-replay-decode \
    --compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt --require-cached-build
```

Baseline 4K/1 reproduced the issue: `6.524s prefill`, `627.865 tok/s`. Kernel trace total was `6.439s`, so the gap is real GPU work, not Python timing overhead. Top kernels:

| Kernel group | Time | Calls | Share |
| --- | ---: | ---: | ---: |
| linear GDN recurrent K2 | 1.568s | 30 | 24.3% |
| linear conv prefill lowp | 0.945s | 30 | 14.7% |
| PARO rotate1 | 0.939s | 190 | 14.6% |
| grouped MoE selected dual/down WMMA | 0.821s | 80 | 12.8% |
| AWQ prefill dual/single | 0.834s | 170 | 12.9% |
| linear prepare | 0.319s | 30 | 5.0% |
| AOTriton full attention | 0.157s | 10 | 2.4% |

Conclusion: 4K prefill is not attention-bound. Full-attention AOTriton is only ~2.4% of kernel time. The loss is dominated by unchunked 4096-row linear-attention and rotate/MLP surfaces on the 30 linear-attention layers.

### Chunk-size A/B

Repeated 4K/1 with prefill chunk flags:

| Variant | Prefill seconds | Prefill tok/s | Notes |
| --- | ---: | ---: | --- |
| default unchunked | 6.441s | 635.884 | no manual chunks |
| AOTriton disabled | 17.163s | 238.656 | confirms AOTriton is necessary, not the bottleneck |
| fused linear GDN rotate env | 6.311s | 649.045 | small +2% only |
| 1024 all chunks | 5.013s | 817.076 | +28% |
| 512 all chunks | 4.381s | 934.927 | +47% |
| 384 all chunks | 4.028s | 1016.899 | +60% |
| 256 all chunks | 3.977s | 1029.808 | +62%, slightly above upstream HIP 4K/128 prefill |

4K/128 and 4K/4K confirmation with 256-row chunks:

| Workload | Prefill tok/s | Decode tok/s | Tracked peak |
| --- | ---: | ---: | ---: |
| 4K/128 chunk256 | 1026.369 | 63.512 | 18.097 GiB |
| 4K/4K chunk256 | 1018.157 | 62.477 | 18.210 GiB |

Additional spot checks:

| Workload | Prefill tok/s with chunk256 | Decode tok/s | Tracked peak |
| --- | ---: | ---: | ---: |
| 512/128 | 966.510 | 62.139 | 17.997 GiB |
| 2048/128 | 970.225 | 62.396 | 18.040 GiB |
| 8192/128 | 981.441 | 62.035 | 18.211 GiB |
| 32K/1 | 794.933 | 48.624 | 18.908 GiB |

Chunk256 profile (`/tmp/gfx1151-prefill-diagnostics-20260517/profile-chunk256-4k/out`) reduced kernel total from `6.439s` to `3.987s`. The biggest wins were linear conv (`0.945s -> 0.091s`) and rotate1 (`0.939s -> 0.181s`); GDN recurrent also improved (`1.568s -> 0.931s`). MoE/AWQ/attention kernel time increases from more chunks/launches, but the linear-attention gains dominate.

Working hypothesis: Strix Halo/gfx1151 dislikes the current unchunked 4096-row prefill surfaces for linear-attention/rotate kernels. The existing long-context chunk machinery fixes the shape. Next optimization should promote a mid-context gfx1151/default autotune profile around `256` rows for `linear/moe/full_attn_*` chunks, then rerun correctness + retained 512/4K/32K/128K sweeps.

## 2026-05-17 — gfx1151 shisa packed chunk256 sweep

After the 4K prefill diagnosis, reran the shisa packed Qwen3.6 sweep with 256-row chunks promoted to all prefill surfaces:

```bash
HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151 \
  python3 scripts/qwen35_paro_bench.py \
    --model /models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e \
    --backend hip_gfx1151 --shared-expert-format packed_paro_w4 --token-id 9707 \
    --prompt-length {P} --decode-tokens {D} --warmup-decode-tokens 4 --max-layers 40 \
    --attn-aotriton-min-tokens 512 --graph-replay-decode \
    --prefill-linear-chunk-size 256 --prefill-moe-chunk-size 256 \
    --prefill-full-attn-query-chunk-size 256 --prefill-full-attn-post-chunk-size 256 \
    --prefill-full-attn-rope-chunk-size 256 \
    --compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt --require-cached-build \
    --json /tmp/hipengine-gfx1151-shisa36-packed-chunk256-20260517/packed-chunk256-{P}-{D}.json
```

Artifact: `benchmarks/results/2026-05-17-hipengine-gfx1151-shisa-qwen36-packed-chunk256-sweep-diagnostic.json`.

Status remains **diagnostic retained** (`performance_claim=false`): generated previews are finite/repeated token `9707`, but there is no shisa KL/top-1 gate or repeated-run statistic yet.

| Workload | Prefill tok/s | Decode tok/s | Tracked peak | Prefill vs default | Prefill vs upstream HIP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 983.206 | 62.060 | 17.997 GiB | +11.6% | -7.1% |
| 4K/128 | 1029.402 | 63.605 | 18.097 GiB | +63.2% | +2.5% |
| 32K/128 | 792.296 | 50.629 | 18.909 GiB | +32.3% | +7.7% |
| 128K/128 | 413.489 | 30.245 | 21.877 GiB | +11.2% | +9.9% |
| 4K/4K | 1001.266 | 62.438 | 18.210 GiB | +61.1% | +1.1% |

Compared to the upstream llama.cpp HIP rerun (`1058.738 / 1004.220 / 735.534 / 376.070 / 990.726` prefill and `50.537 / 49.379 / 43.435 / 31.286 / 49.071` decode), chunk256 fixes the 4K prefill gap and gives better prefill at 4K/32K/128K/4K4K while preserving the decode win except at 128K. It also reduces tracked peak memory vs the default sweep (`18.123/19.995/20.267/23.235/20.108 GiB` -> `17.997/18.097/18.909/21.877/18.210 GiB`).

Updated `scripts/qwen35_compare_tables.py` so `--target gfx1151` now points at the chunk256 sweep. Throughput-only comparison commands:

```bash
python3 scripts/qwen35_compare_tables.py --target gfx1151 hip-gfx1151 --no-memory
python3 scripts/qwen35_compare_tables.py --target gfx1151 vulkan-gfx1151 --no-memory
```

## 2026-05-17 — README gfx1100/gfx1151 performance split

Updated `README.md` performance presentation to split the top-level tables into:

- `gfx1100 (Radeon RX 7900 XTX / Radeon Pro W7900)` using the existing W7900/gfx1100 shisa packed vs llama.cpp HIP/Vulkan rows.
- `gfx1151 (AMD Ryzen AI MAX+ 395 / Radeon 8060S)` using the latest chunk256 shisa packed sweep (`983.206 / 1029.402 / 792.296 / 413.489 / 1001.266` prefill tok/s and `62.060 / 63.605 / 50.629 / 30.245 / 62.438` decode tok/s) against the upstream llama.cpp HIP/Vulkan rerun.

Before editing, confirmed chunk256 has no long-context regression versus the prior default/parent-style chunk sweep: 32K prefill improved `598.663 -> 792.296 tok/s` (+32.3%), 128K prefill improved `371.722 -> 413.489 tok/s` (+11.2%), decode was effectively flat/slightly positive, and tracked peak memory dropped by ~1.36 GiB on both 32K and 128K rows.

## 2026-05-17 — Additional chunk-size probe below/around 256

After the chunk256 sweep, ran quick 4K shisa packed probes to answer whether smaller chunks (`128`/`64`) or per-surface chunk sizes are worth pursuing. Commands used cached gfx1151 kernels with `--prompt-length 4096`; most probes used `--decode-tokens 1 --warmup-decode-tokens 0` to isolate prefill.

All-surfaces smaller-than-256 was worse:

| Variant | Prefill tok/s | Notes |
| --- | ---: | --- |
| all128 | 935.871 | slower than all256 (`~1029 tok/s`) |
| all64 | 716.446 | too much launch/overhead |

Per-surface 4K/1 probes around chunk256:

| Variant | Linear | MoE | Full-attn q/post/rope | Prefill tok/s |
| --- | ---: | ---: | ---: | ---: |
| linear128-rest256 | 128 | 256 | 256 | 964.648 |
| moe128-rest256 | 256 | 128 | 256 | 962.279 |
| full128-rest256 | 256 | 256 | 128 | 1003.372 |
| linear384-moe256-full256 | 384 | 256 | 256 | 1037.376 |
| linear256-moe384-full256 | 256 | 384 | 256 | 972.969 |
| linear256-moe256-full384 | 256 | 256 | 384 | 1051.281 |
| linear384-moe256-full384 | 384 | 256 | 384 | 1037.383 |
| linear512-moe256-full384 | 512 | 256 | 384 | 1019.392 |
| linear384-moe256-full512 | 384 | 256 | 512 | 1018.663 |
| linear512-moe256-full512 | 512 | 256 | 512 | 1001.862 |
| linear384-moe512-full384 | 384 | 512 | 384 | 1032.709 |

4K/128 confirmation probes:

| Variant | Prefill tok/s | Decode tok/s | Tracked peak |
| --- | ---: | ---: | ---: |
| all256 retained sweep | 1029.402 | 63.605 | 18.097 GiB |
| linear256-moe256-full384 | 986.433 | 63.774 | 18.137 GiB |
| linear384-moe256-full384 | 1044.611 | 63.710 | 18.137 GiB |

Takeaway: do **not** go below 256 globally on gfx1151. Per-surface tuning is promising but the only confirmed 4K/128 improvement over all256 is modest (`linear=384, moe=256, full=384`, +1.5% prefill, same decode, +0.04 GiB). Keep all256 as the robust default for now; use a small autotune matrix (`linear in {256,384}`, `moe=256`, `full in {256,384}`) if we want to squeeze another percent and verify at 512/4K/32K/128K before changing the retained row.

## 2026-05-17 — gfx1151 chunk autotune OFAT + 1024 extension

Extended the shisa packed chunk-size tuning sweep after the retained all256 chunk sweep. Goal: decide whether to go below 256, include 1024, or tune individual surfaces separately before changing the retained/default gfx1151 setting.

Method:

- Model: `shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed` snapshot `501ef8635e5cfb5a7497d232358ca8d1afc0c66e`.
- Backend/env: `HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode HIPENGINE_HIP_ARCH=gfx1151`, cached builds.
- Prompts: `512`, `1024`, `4096`, `8192`.
- Fast probe rows: `--decode-tokens 1 --warmup-decode-tokens 0` for OFAT and combinations.
- Full candidate confirmation rows: `512/128`, `4K/128`, `32K/128`, `128K/128`, `4K/4K`.
- Raw probe CSV: `/tmp/gfx1151-chunk-ofat-20260517/results.csv`.
- Full candidate JSONs: `/tmp/hipengine-gfx1151-shisa36-packed-candidates-20260517/`.

### Round 1: OFAT with {64,128,256,512,1024}

Held all surfaces at all256, then varied one surface/group at a time. Results are noisy single runs, but the pattern is clear enough:

- Global smaller chunks are bad: all64/all128 consistently trail all256.
- Global larger chunks are also bad: all512/all1024 trail all256, often badly.
- Individual `linear=512` or `linear=1024` is the most robust OFAT improvement across short/mid prompts.
- `moe=1024` helps 1024/4096 OFAT but hurts 8192.
- `full_q=512` helps 8192 but hurts 4096; `full_q=1024` is not a clear win and raises peak memory.
- `full_post`/`full_rope` changes are small/noisy and not additive in combinations.

Top OFAT rows by prompt after adding 1024:

| Prompt | Best OFAT config | Prefill tok/s | all256 tok/s |
| --- | --- | ---: | ---: |
| 512 | fullpost128 | 997.908 | 968.884 |
| 1024 | moe1024 | 1029.245 | 1017.095 |
| 4096 | linear512 | 1043.240 | 1035.458 |
| 8192 | fullq512 | 1003.324 | 938.318 |

### Round 2: combination probes

Tried a small combination set from OFAT signals and prior 384 hints:

| Config | 512 | 1024 | 4096 | 8192 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| all256 | 968.884 | 1017.095 | 1035.458 | 938.318 | retained robust baseline |
| linear512 only | 989.920 | 1010.456 | 1043.240 | 991.128 | best cross-prompt mean in fast probe |
| linear1024 only | 995.003 | 1019.336 | 1032.191 | 993.306 | similar but weaker at 4K |
| combo linear384/full384 | 944.933 | 1022.440 | 1003.943 | 1008.363 | good at 8192, bad at 512/4K |
| combo linear512/full512 | 990.637 | 1008.563 | 1020.993 | 971.531 | not additive |
| combo linear512/q512/post1024/rope512 | 968.330 | 996.322 | 1025.865 | 948.627 | not additive |

Conclusion from fast probes: per-surface tuning exists, but additive combinations are not obviously better than the simple all256 row. The only robust single-knob candidate is `linear=512, everything else=256`.

### Round 3: full candidate confirmation

Confirmed `linear512 only` and `linear1024 only` on the full retained workload set against the current all256 artifact.

| Workload | all256 prefill | linear512 prefill | Delta | linear512 decode | all256 decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512/128 | 983.206 | 971.270 | -1.2% | 62.625 | 62.060 |
| 4K/128 | 1029.402 | 1040.008 | +1.0% | 63.958 | 63.605 |
| 32K/128 | 792.296 | 800.872 | +1.1% | 51.265 | 50.629 |
| 128K/128 | 413.489 | 417.931 | +1.1% | 30.223 | 30.245 |
| 4K/4K | 1001.266 | 1009.313 | +0.8% | 63.121 | 62.438 |

`linear1024 only` was weaker overall: `-1.5% / -1.4% / -0.1% / +1.4% / -0.2%` prefill vs all256 for `512/128`, `4K/128`, `32K/128`, `128K/128`, `4K/4K`.

### Decision

Keep **all256** as the retained/simple gfx1151 setting for now. It is within ~1% of `linear512 only` on most retained rows, wins at 512, is easier to explain, and avoids overfitting single-run noise. If we do another optimization pass, compare repeated runs for:

1. all256: `linear=256, moe=256, full_q/post/rope=256`.
2. linear512-only: `linear=512, moe=256, full_q/post/rope=256`.

Do not spend more time on global 64/128/512/1024 or broad full-factorial combinations unless W7900 shows a different architecture-specific pattern.

## 2026-05-18 — Branding casing update to hipEngine

### Scope

- Replaced the legacy project-name casing with `hipEngine` across tracked repository files using the tracked-file match list and a Python `str.replace` pass.
- Verification used a constructed legacy-casing search string so this log entry does not reintroduce the old contiguous spelling: `OLD=$(printf 'hip%sNGINE' E); if git grep -n "$OLD" -- .; then exit 1; else echo "No legacy casing remains in tracked files"; fi`.
- No runtime tests were run; this was a docs/comments/metadata/string-only rename.

## 2026-05-18 — Backend auto-detection selector

### Scope

- Added `backend="auto"` selection in `hipengine.kernels.backends`, mapping detected ROCm `gfx1100`/`gfx1151` targets to `hip_gfx1100`/`hip_gfx1151` before registry lookup.
- Added `HIPENGINE_BACKEND` as a force override for nearby targets; unknown/no HIP detections warn with `gfx1101`/`gfx1102` guidance and select `cpu_reference` where a CPU implementation exists.
- Updated `LLM`, server defaults, Qwen3.5/PARO runner/script defaults, model metadata, and docs/API/README to use the selector without adding backend branches to dispatch/model code.

### Validation

- `python3 -m compileall -q hipengine tests/test_gfx1151_backend.py tests/test_llm_generate.py` passed.
- `uv run pytest tests/test_gfx1151_backend.py tests/test_llm_generate.py tests/test_server_api.py -q` failed during collection because the transient uv environment did not include pytest/dev extras.
- `uv run --extra dev python -m pytest tests/test_gfx1151_backend.py tests/test_llm_generate.py tests/test_server_api.py -q` passed: 19 tests.
- `git diff --check` passed.

## 2026-05-18 — v0.1.0 packaging and publish checklist prep

### Scope

- Bumped package metadata from `0.0.0` to `0.1.0` and added PyPI project URLs/classifiers.
- Added a Hatch custom build hook so wheels that bundle the x86-64 Linux AOTriton runtime build as `py3-none-manylinux_2_39_x86_64` with `Root-Is-Purelib: false` instead of `py3-none-any`; ROCm libraries remain external system dependencies.
- Added a top-level package release `CHANGELOG.md`; kept benchmark/performance history in `benchmarks/CHANGELOG.md`.
- Added `docs/PUBLISH.md` using the prior Outline/textguard checklists as references, tailored to hipEngine GitHub/PyPI release steps, platform-wheel checks, LFS, server extra smoke, and evidence policy.

### Validation

- `python3 -m compileall -q hipengine scripts tests hatch_build.py` passed.
- `uv run --extra dev python -m pytest -q` passed.
- `uv run --extra dev hipengine-server --help` passed.
- `python3 -m build --outdir /tmp/hipengine-dist-check` built `hipengine-0.1.0.tar.gz` and `hipengine-0.1.0-py3-none-manylinux_2_39_x86_64.whl`.
- Wheel metadata check confirmed `Root-Is-Purelib: false`, `Tag: py3-none-manylinux_2_39_x86_64`, bundled `libaotriton_v2.so.0.11.2`, and no `hipengine.libs/` vendored ROCm payload.
- `uvx --from auditwheel auditwheel show /tmp/hipengine-dist-check/hipengine-0.1.0-py3-none-manylinux_2_39_x86_64.whl` reported the current native payload constrains the wheel to `manylinux_2_39_x86_64`.
- `uvx --from twine twine check /tmp/hipengine-dist-check/*` passed.
- `(cd /tmp && uv run --isolated --with '/tmp/hipengine-dist-check/hipengine-0.1.0-py3-none-manylinux_2_39_x86_64.whl[server]' hipengine-server --help)` passed.
- `git diff --check` passed.

## 2026-05-18 — Qwen3.5 >1K prefill chunk policy default

Decision from W7900/gfx1100 chunk sweeps: keep the policy simple and make the
manual long-context-equivalent chunks the default above the 1K prompt seam.
`PrefillConfig.auto_tune_chunk_sizes=True` now leaves actual prompt lengths
`<=1024` unchunked and resolves prompts `>1024` to:

- linear attention: `1024`
- MoE: `1024`
- full-attn query: `4096`
- full-attn post: `1024`
- full-attn RoPE: `1024`

Manual non-zero chunk sizes still override the resolver.  I changed the resident
runner to re-resolve chunk sizes from the actual prefill prompt length (not only
session capacity), so a 1K prompt remains unchunked even when the decode-capacity
headroom makes `max_sequence_length > 1024`.

Evidence sources from the sweep series:

- `/tmp/hipengine-chunk-sweep-stage7-short-single-20260518-013706`: 512/128 has
  no useful chunk gain; 1K is noise-level.
- `/tmp/hipengine-chunk-sweep-stage8-1k-seam-20260518-014326`: three repeats show
  1K median manual-long `-0.04%` vs baseline/noise, while 1.5K manual-long is
  `+0.64%` and saves `0.09 GiB` tracked peak.
- `/tmp/hipengine-chunk-sweep-stage6-sub8k-single-20260518-012148`: manual-long
  wins 2K-4K; linear-only wins 6K/7K but by small margins and higher memory.
- `/tmp/hipengine-chunk-sweep-stage5-low-mid-single-20260518-011119`: manual-long
  wins 8K/12K/15K/16K, stays near `19.6-19.9 GiB` tracked peak, and keeps 12K+
  under the 24 GiB guardrail where unchunked baseline crosses it.
- `/tmp/hipengine-chunk-sweep-stage4-binary-seam-20260518-005115`: linear-only is
  rejected from 17K upward vs full long chunks because memory is `+4.24 GiB` or
  worse for equal/slower prefill.

Selected retained measurements (Qwen3.5-35B-A3B-PARO w4_paro, max_layers=40,
token id 9707, W7900/gfx1100):

- 1.5K/128 median manual-long `2594.862 tok/s`, `+0.64%` vs baseline, tracked
  peak `18.620 GiB`.
- 2K/128 manual-long `2627.75 tok/s`, `+3.01%`, peak `18.80 GiB`.
- 4K/128 manual-long `2620.14 tok/s`, `+6.55%`, peak `19.51 GiB`.
- 8K/128 manual-long `2546.37 tok/s`, `+8.46%`, peak `19.62 GiB`.
- 12K/128 manual-long `2419.67 tok/s`, `+9.96%`, peak `19.74 GiB` vs baseline
  peak `24.33 GiB`.
- 16K/128 manual-long `2280.31 tok/s`, `+11.04%`, peak `19.85 GiB` vs baseline
  peak `26.47 GiB`.

Post-code validation:

```bash
python3 -m py_compile hipengine/runtime/prefill.py hipengine/runtime/qwen35_paro_runner.py \
  scripts/qwen35_paro_bench.py scripts/qwen35_native_prefill_fixture_gate.py
# passed

python3 -m pytest tests/test_qwen35_resident_batch_layout.py -q --tb=short
# 27 passed

python3 scripts/qwen35_native_prefill_fixture_gate.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 \
  --attn-aotriton-min-tokens 512 \
  --prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 \
  --prefill-full-attn-query-chunk-size 4096 \
  --prefill-full-attn-post-chunk-size 1024 \
  --prefill-full-attn-rope-chunk-size 1024 \
  --json /tmp/hipengine-default-gt1k-manual-long-fixture.json
# passed=True, generated_match=True, expected_match=True, max_kl=0.0395688706, top1=1.0

python3 scripts/qwen35_paro_bench.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --prompt-length 512 --token-id 9707 --decode-tokens 16 --warmup-decode-tokens 1 \
  --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 --graph-replay-decode \
  --json /tmp/hipengine-default-gt1k-512.json
# chunks all zero, prefill=2289.648 tok/s, decode=115.768 tok/s, peak=18.175 GiB

python3 scripts/qwen35_paro_bench.py \
  --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd \
  --prompt-length 4096 --token-id 9707 --decode-tokens 128 --warmup-decode-tokens 1 \
  --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 --graph-replay-decode \
  --json /tmp/hipengine-default-gt1k-4k.json
# chunks 1024/1024/4096/1024/1024, prefill=2674.598 tok/s, decode=117.367 tok/s, peak=19.507 GiB
```

Benchmark artifact and rollup update:

- `benchmarks/results/2026-05-18-hipengine-qwen35-gt1k-prefill-chunk-policy-diagnostic.json`
- `benchmarks/README.md` P5.3 row
- `benchmarks/CHANGELOG.md` 2026-05-18 entry

## 2026-05-18 — gfx1100 shisa packed compare-table refresh after >1K chunks

Refreshed the W7900/gfx1100 shisa packed Qwen3.6 rows after the default prefill
policy change (`<=1024` unchunked, `>1024` resolves to
`1024/1024/4096/1024/1024`).  The compare-table A-side now uses the packed-only
checkpoint and default auto chunking rather than old manual/no-chunk short-row
assumptions.

Benchmark command template (one single run per workload; diagnostic only, no
shisa KL/top-1 gate):

```bash
python3 scripts/qwen35_paro_bench.py \
  --model /models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e \
  --backend hip_gfx1100 --shared-expert-format packed_paro_w4 \
  --token-id 9707 --warmup-decode-tokens 4 --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --attn-aotriton-min-tokens 512 --graph-replay-decode \
  --prompt-length {512|4096|32768|131072} --decode-tokens 128 \
  --json /tmp/hipengine-gfx1100-shisa36-packed-gt1k-default-20260518-025325/...
```

A cached-build warmup/prebuild was run once without `--require-cached-build`:

```bash
python3 scripts/qwen35_paro_bench.py \
  --model /models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e \
  --backend hip_gfx1100 --shared-expert-format packed_paro_w4 --token-id 9707 \
  --warmup-decode-tokens 0 --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --attn-aotriton-min-tokens 512 --graph-replay-decode \
  --prompt-length 8 --decode-tokens 1 \
  --json /tmp/hipengine-gfx1100-shisa36-packed-prebuild.json
# prebuild_ok
```

Measured rows (AMD Radeon Pro W7900 / gfx1100, graph replay decode):

| Workload | Prefill tok/s | Decode tok/s | Tracked peak | Resolved chunks |
| --- | ---: | ---: | ---: | --- |
| 512/128 | 2500.565 | 111.516 | 18.123 GiB | all zero |
| 4K/128 | 2899.685 | 113.094 | 19.455 GiB | 1024/1024/4096/1024/1024 |
| 32K/128 | 2115.050 | 97.594 | 20.267 GiB | 1024/1024/4096/1024/1024 |
| 128K/128 | 1054.291 | 62.027 | 23.235 GiB | 1024/1024/4096/1024/1024 |

Compared to the prior packed refresh, 4K prefill improves `2711.013 ->
2899.685 tok/s` (+6.96%) and tracked peak drops `19.995 -> 19.455 GiB`; 512,
32K, and 128K are within single-run noise.  Updated:

- `scripts/qwen35_compare_tables.py` (`--target shisa` rows and source)
- top-level `README.md` gfx1100 tables
- `benchmarks/README.md` rollup row
- `benchmarks/CHANGELOG.md`
- `benchmarks/results/2026-05-18-hipengine-gfx1100-shisa-qwen36-packed-gt1k-default-diagnostic.json`

Validation:

```bash
python3 -m py_compile scripts/qwen35_compare_tables.py
python3 scripts/qwen35_compare_tables.py --target shisa llama.cpp-hip >/tmp/qwen35-compare-shisa-hip.md
python3 scripts/qwen35_compare_tables.py --target shisa --against-target >/tmp/qwen35-compare-shisa-legacy.md
python3 -m json.tool benchmarks/results/2026-05-18-hipengine-gfx1100-shisa-qwen36-packed-gt1k-default-diagnostic.json >/tmp/gfx1100-shisa-packed.json
python3 - <<'PY'
from pathlib import Path
import re, urllib.parse
fail=[]
for md in [Path('README.md'), Path('benchmarks/README.md'), Path('benchmarks/CHANGELOG.md')]:
    text=md.read_text()
    for m in re.finditer(r'\[[^\]]+\]\(([^)]+)\)', text):
        target=m.group(1).split('#',1)[0]
        if not target or '://' in target or target.startswith('mailto:'):
            continue
        path=(md.parent / urllib.parse.unquote(target)).resolve()
        if not path.exists():
            fail.append((str(md), target))
if fail:
    raise SystemExit(fail)
print('markdown links OK')
PY
# all passed
```

## 2026-05-18 - K1 INT8 KV lineage preflight path repair

Started dense INT8 KV bring-up on branch `kvcache-int8` with task #1.

Changed `docs/source_lineage.json` paths from stale `/home/lhl/github/shisa-ai/amd-gpu-tuning/...`
to the live `/home/lhl/amd-gpu-tuning/...` workspace so the required pre-port lineage check can run.

Validation / evidence:

```bash
python3 -m json.tool docs/source_lineage.json >/tmp/hipengine-source-lineage.json
python3 scripts/check_lineage.py --kind kernel --diff stat
```

Result: JSON validation passed; lineage check now runs without path errors. It reports expected parent drift:

- repo `nano-vllm-amd`: `/home/lhl/amd-gpu-tuning/nano-vllm-amd`, branch `gfx1100-qwen3.5`, head `5d8f496`.
- `csrc/amd/qwen35_expert.hip` drift since baseline `22405a9`: commits `6e2b19b`, `5fde418`, `b95eaa5`; diffstat `1011` lines. These are compact WMMA / tree recurrence / GDN tloop updates, not a new INT8 KV replacement.
- Kernel catalog still lists the INT8 KV symbols under `docs/KERNELS.md`: `qwen35_paged_full_attn_decode_split_k_int8_kernel`, `qwen35_paged_full_attn_decode_split_k_ctx_tensor_int8_kernel`, `qwen35_write_paged_kv_int8_kernel`, and `qwen35_write_paged_kv_position_tensor_int8_kernel`.

Parent INT8 KV source evidence:

```bash
git -C /home/lhl/amd-gpu-tuning/nano-vllm-amd log --oneline --reverse -S 'qwen35_write_paged_kv_int8_kernel' -- csrc/amd/qwen35_expert.hip
git -C /home/lhl/amd-gpu-tuning/nano-vllm-amd log --oneline --reverse -S 'qwen35_paged_full_attn_decode_split_k_int8_kernel' -- csrc/amd/qwen35_expert.hip
```

Both identify `nano-vllm-amd@2751f2f` (`feat(amd): INT8 per-token KV cache with vec16 read and parallel write`) as the source commit. Parent WORKLOG entry `/home/lhl/amd-gpu-tuning/WORKLOG.md:20081` records the INT8 KV implementation, scale shapes `[num_blocks, block_size, num_kv_heads]`, opt-in `--kv-cache-dtype int8`, and parent results: INT8 attention faster at 4K microbench but only ~+0.2% E2E at 4K/128 and 4K/D4K.

Porting caution for K1: the parent writer must be audited against `docs/KVCACHE.md` before copying. The source claims per-head scales, but the current writer reduces max-abs over `total_size = num_kv_heads * head_dim` and then writes the same reduced max for each head; it also needs explicit zero-scale handling for all-zero rows. Task #2 will pin the hipEngine oracle semantics before any kernel port.
