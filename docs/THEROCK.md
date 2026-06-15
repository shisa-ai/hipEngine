# TheRock ROCm Environment

Last updated: 2026-06-15

This page is the retained setup for hipEngine W7900 / gfx1100 benchmark runs
that use AMD TheRock Python wheels, plus the local Strix Halo / gfx1151 ROCm
dev and PyTorch bootstrap recipe. It records the install recipes, package flavor
choices, verification commands, and the ROCm 7.14 nightly regression diagnostics
that keep ROCm 7.13 as the canonical stack for current W7900 topline rows. The
upstream release-package reference is TheRock
[`RELEASES.md`](https://github.com/ROCm/TheRock/blob/main/RELEASES.md).

## Retained Stack

Use the `therock` Python 3.12 environment:

```bash
PY=/home/lhl/mambaforge/envs/therock/bin/python3.12
```

The retained W7900 benchmark stack is:

| Package | Version |
| --- | --- |
| `rocm` | `7.13.0a20260423` |
| `rocm-sdk-core` | `7.13.0a20260423` |
| `rocm-sdk-devel` | `7.13.0a20260423` |
| `rocm-sdk-libraries-gfx110X-all` | `7.13.0a20260423` |
| `torch` | `2.11.0+rocm7.13.0a20260423` |
| `torchvision` | `0.26.0+rocm7.13.0a20260423` |
| `torchaudio` | `2.11.0+rocm7.13.0a20260423` |
| `triton` | `3.6.0+rocm7.13.0a20260423` |
| `numpy` | `2.1.3` |
| `fsspec` | `2026.2.0` |

Expected compiler/runtime identity:

```text
HIP version: 7.13.26162-1140233ffe
torch.version.hip 7.13.26162
```

Current local host metadata for the retained stack:

| Component | Value |
| --- | --- |
| Kernel | `Linux epyc 7.0.10-1-cachyos #1 SMP PREEMPT_DYNAMIC Sun, 24 May 2026 14:29:40 +0000 x86_64` |
| ROCm driver reported by `rocm-smi` | `7.0.10-1-cachyos` |
| GPU0 | AMD Radeon Pro W7900 / gfx1100, VBIOS `113-D7070100-138`, 44.984 GiB VRAM |
| GPU1 | AMD Radeon RX 7900 XTX / gfx1100, VBIOS `113-EXT89622-001`, 23.985 GiB VRAM |

If the kernel, firmware, or TheRock package set changes, re-run at least the
README PARO/GGUF sweep and the 24GB startup/headroom smoke before promoting the
stack.

## Package Flavor

Use the `gfx110X-all` index:

```text
https://rocm.nightlies.amd.com/v2/gfx110X-all/
```

This is the package family used by the retained W7900 rows. It contains the
gfx1100-family tuned library assets and installs
`rocm-sdk-libraries-gfx110X-all`.

Do not use `gfx1100-dgpu` for the Linux W7900 benchmark environment; it is not
the validated package/index name for this repo. Do not substitute
`gfx1100-all`; the installed retained package name is
`rocm-sdk-libraries-gfx110X-all`. Avoid the multi-arch wheel index for retained
performance rows unless it is revalidated, because the current evidence is tied
to the explicit `gfx110X-all` package set.

## gfx1151 / Strix Halo Local Nightly Install

For local Strix Halo APUs (`gfx1151`, e.g. Ryzen AI Max / Radeon 8060S), use the
per-architecture gfx1151 TheRock index, not the retained W7900 `gfx110X-all`
index:

```bash
mamba create -n therock python=3.12
mamba activate therock

INDEX=https://rocm.nightlies.amd.com/v2/gfx1151/
```

The known-good local gfx1151 ROCm dev + PyTorch stack from June 2026 is pinned to
one nightly tag throughout:

| Package | Version |
| --- | --- |
| `rocm-sdk-libraries-gfx1151` | `7.13.0a20260411` |
| `rocm-sdk-devel` | `7.13.0a20260411` |
| `torch` | `2.12.0a0+rocm7.13.0a20260411` |
| `torchvision` | `0.27.0a0+rocm7.13.0a20260411` |
| `torchaudio` | `2.11.0a0+rocm7.13.0a20260411` |
| `triton` | `3.7.0+git18f89f64.rocm7.13.0a20260411` |

Install with exact pins:

```bash
pip install --pre --no-cache-dir \
  --index-url "$INDEX" \
  "rocm-sdk-libraries-gfx1151==7.13.0a20260411" \
  "rocm-sdk-devel==7.13.0a20260411" \
  "torch==2.12.0a0+rocm7.13.0a20260411" \
  "torchvision==0.27.0a0+rocm7.13.0a20260411" \
  "torchaudio==2.11.0a0+rocm7.13.0a20260411" \
  "triton==3.7.0+git18f89f64.rocm7.13.0a20260411"
```

Expected result on the local Ryzen AI Max / Radeon 8060S box:

```text
torch 2.12.0a0+rocm7.13.0a20260411
hip   7.13.60980
torch.cuda.is_available() -> True
device -> Radeon 8060S Graphics
```

### Why Every Package Is Pinned

Do not run a floating install like this on gfx1151:

```bash
pip install --pre --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  "rocm[libraries,devel]" "rocm-sdk-libraries-gfx1151" \
  torch torchvision torchaudio triton
```

The gfx1151 nightlies publish ROCm SDK and PyTorch wheels independently. If the
floating `rocm[libraries,devel]` meta advances to a newer ROCm tag than the
newest torch wheel supports, pip tries to satisfy both constraints by
backtracking through older torch wheels. The symptom is repeated multi-hundred-MB
`torch` downloads with messages like `pip is looking at multiple versions of
torch` and `This is taking longer than usual`. This can burn tens of GB and still
not converge.

Use torch's embedded ROCm tag as the source of truth:

```text
torch==2.12.0a0+rocm7.13.0a20260411
                    └────────────┘  pin rocm-sdk-* to 7.13.0a20260411
```

Then pin `rocm-sdk-libraries-gfx1151`, `rocm-sdk-devel`, torch, torchvision,
torchaudio, and triton to one matching nightly. Dropping the floating
`rocm[libraries,devel]` meta avoids accidentally selecting a newer ROCm SDK than
torch was built against; `rocm-sdk-devel` still provides HIP headers, compiler,
and device libraries for hipEngine kernel builds.

To discover the newest torch candidate without doing a full install:

```bash
pip index versions torch --pre --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
```

Bump all packages together when moving to a newer nightly.

### gfx1151 Verification And Cleanup

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

ROOT=$(python -m rocm_sdk path --root)
echo "$ROOT"
"$ROOT/bin/hipcc" --version
"$ROOT/bin/hipcc" --version > /tmp/hipengine-hipcc-version.txt
rocminfo | grep -E 'Name:|gfx1151'
```

ROCm/HIP PyTorch builds expose AMD GPUs through the CUDA compatibility API, so
`torch.cuda.is_available()` and `torch.cuda.get_device_name()` are the right
checks. There is no separate `torch.hip` device namespace.

If converting an existing gfx1100/gfx110X environment, remove stale device or
library wheels before reinstalling or after the pinned install:

```bash
pip uninstall -y \
  rocm-sdk-libraries-gfx110X-all \
  rocm-sdk-libraries-gfx110X-dgpu \
  rocm-sdk-libraries-gfx1100 \
  rocm-sdk-libraries \
  amd-torch-device-gfx1100 \
  amd-torch-device-gfx11 \
  amd-torchvision-device-gfx1100
```

Reclaim disk after failed floating installs or successful setup:

```bash
pip cache purge
mamba clean --all --yes
```

For clean hipEngine benchmark or profiling processes on gfx1151, adapt the
wrapper below by replacing `_rocm_sdk_libraries_gfx110X_all` with
`_rocm_sdk_libraries_gfx1151` and setting `HIPENGINE_HIP_ARCH=gfx1151` when a
native gfx1151 JIT build is required. Do not set `HSA_OVERRIDE_GFX_VERSION` for a
real gfx1151 local device unless debugging a specific compatibility issue.

Strix Halo uses shared system memory for the integrated GPU. If large model runs
fail from memory pressure even though host RAM is available, check the platform's
GTT / TTM configuration (for example `ttm.pages_limit` in the host modprobe or
kernel-command-line setup); the default cap may be much lower than installed RAM.
That host-level tuning is outside TheRock itself, but it is a prerequisite for
large local LLM runs.

## Install Or Repair

Start with the pinned W7900 / gfx1100 reinstall:

```bash
PY=/home/lhl/mambaforge/envs/therock/bin/python3.12
INDEX=https://rocm.nightlies.amd.com/v2/gfx110X-all/

"$PY" -m pip install --upgrade --force-reinstall --no-cache-dir \
  --index-url "$INDEX" \
  "rocm[libraries,devel]==7.13.0a20260423" \
  "rocm-sdk-libraries-gfx110X-all==7.13.0a20260423" \
  "torch==2.11.0+rocm7.13.0a20260423" \
  "torchvision==0.26.0+rocm7.13.0a20260423" \
  "torchaudio==2.11.0+rocm7.13.0a20260423" \
  "triton==3.6.0+rocm7.13.0a20260423"
```

Then remove stale 7.14 helper/device wheels if this environment was previously
upgraded to ROCm 7.14. These wheels can remain installed after the main
downgrade and make the environment internally inconsistent:

```bash
"$PY" -m pip uninstall -y \
  amd-torch-device-gfx1100 \
  amd-torch-device-gfx11 \
  amd-torchvision-device-gfx1100 \
  rocm-sdk-device-gfx1100 \
  rocm-sdk-libraries
```

Restore the package versions expected by local Quark/datasets tooling:

```bash
"$PY" -m pip install --upgrade --force-reinstall --no-cache-dir \
  "numpy==2.1.3" \
  "fsspec==2026.2.0"
```

## Verify

Run all checks before a retained benchmark:

```bash
PY=/home/lhl/mambaforge/envs/therock/bin/python3.12

"$PY" -m pip list | rg '^(amd-|rocm|torch|torchvision|torchaudio|triton|numpy|fsspec|hipengine)'

ROOT=$("$PY" -m rocm_sdk path --root)
echo "$ROOT"
"$ROOT/bin/hipcc" --version
"$ROOT/bin/hipcc" --version > /tmp/hipengine-hipcc-version.txt

"$PY" - <<'PY'
import ctypes
import torch
ctypes.CDLL("libamdhip64.so")
print("hip OK")
print("torch", torch.__version__)
print("torch.version.hip", torch.version.hip)
print("cuda_available", torch.cuda.is_available())
PY
```

`pip check` is still useful for catching stale AMD device-package conflicts, but
the shared `therock` environment currently has unrelated `minisgl` dependency
conflicts. Treat `amd-torch-device-*`, `amd-torchvision-device-*`, or mixed
`rocm7.14` package conflicts as blockers for retained hipEngine benchmarks;
the existing `minisgl` conflicts are not part of the hipEngine ROCm stack.

## Clean Process Wrapper

For long benchmark runs, prefer an explicit TheRock process environment so the
process does not mix TheRock libraries with system `/opt/rocm` libraries:

```bash
PY=/home/lhl/mambaforge/envs/therock/bin/python3.12
CONDA_PREFIX=/home/lhl/mambaforge/envs/therock
ROOT=$("$PY" -m rocm_sdk path --root)

env -i HOME=$HOME USER=$USER LOGNAME=$LOGNAME SHELL=$SHELL TERM=${TERM:-xterm} \
  PATH="$ROOT/bin:$ROOT/lib/llvm/bin:$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="$ROOT/lib:$ROOT/lib64:$ROOT/lib/llvm/lib:$CONDA_PREFIX/lib/python3.12/site-packages/_rocm_sdk_core/lib:$CONDA_PREFIX/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx110X_all/lib" \
  HIP_PATH="$ROOT" ROCM_PATH="$ROOT" HIP_LIB_PATH="$ROOT/lib" HIP_INCLUDE_PATH="$ROOT/include" \
  HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode" \
  HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
  PYTHONPATH=. \
  python3 <script> ...
```

Only add `HSA_OVERRIDE_GFX_VERSION=11.0.0` as a local compatibility workaround
after rechecking the attached device. It is not a general hipEngine default.

## ROCm 7.14 Diagnostic

ROCm 7.14 nightly was tested on 2026-06-14 and is **not promoted** for retained
W7900 toplines. The result is mixed for PARO, negative for GGUF prefill, and
negative for the retained MTP wall metric.

| Workload | ROCm 7.14 vs retained 7.13 | Verdict |
| --- | --- | --- |
| PARO packed 512/128 | prefill `+0.61%`, decode `+1.09%` | small win only at short context |
| PARO packed 4K/128 | prefill `-0.01%`, decode `+0.60%` | effectively neutral |
| PARO packed 32K/128 | prefill `-1.59%`, decode `-0.23%` | regression |
| PARO packed 128K/128 | prefill `-4.49%`, decode `-1.06%` | regression |
| GGUF Q4_K_S 512/128 | prefill `-14.18%`, decode `-1.01%` | clear regression |
| GGUF Q4_K_S 4K/128 | prefill `-12.92%`, decode `-1.42%` | clear regression |
| GGUF Q4_K_S 32K/128 | prefill `-9.78%`, decode `-0.08%` | clear prefill regression |
| GGUF Q4_K_S 128K/128 | prefill `-4.38%`, decode `+0.66%` | mixed, prefill negative |
| MTP B=1 retained old artifact | cycle `14.134 -> 14.595 ms`, prompt-mean `1.023x -> 0.991x` | regression |
| Concurrency c1/c2/c4/c8 | aggregate `-0.33% / +0.39% / +2.11% / +0.15%` | diagnostic only |

Artifacts:

- PARO 7.14 diagnostic:
  [`../benchmarks/results/2026-06-14-w7900-rocm714-hipengine-paro-packed-readme-persistent-5run-diagnostic.json`](../benchmarks/results/2026-06-14-w7900-rocm714-hipengine-paro-packed-readme-persistent-5run-diagnostic.json)
- GGUF 7.14 diagnostic:
  [`../benchmarks/results/2026-06-14-w7900-rocm714-hipengine-gguf-q4ks-readme-persistent-5run-diagnostic.json`](../benchmarks/results/2026-06-14-w7900-rocm714-hipengine-gguf-q4ks-readme-persistent-5run-diagnostic.json)
- MTP 7.14 diagnostic:
  [`../benchmarks/results/2026-06-14-hipengine-mtp-b1-oldartifact-rocm714-3run-diagnostic.json`](../benchmarks/results/2026-06-14-hipengine-mtp-b1-oldartifact-rocm714-3run-diagnostic.json)
- Final-packed MTP 7.14 no-hold:
  [`../benchmarks/results/2026-06-14-hipengine-mtp-finalpacked-rocm714-exactness-nohold.json`](../benchmarks/results/2026-06-14-hipengine-mtp-finalpacked-rocm714-exactness-nohold.json)
- Concurrency 7.14 diagnostic:
  [`../benchmarks/results/2026-06-14-hipengine-qwen35-concurrency-decode-rocm714-w7900/summary.json`](../benchmarks/results/2026-06-14-hipengine-qwen35-concurrency-decode-rocm714-w7900/summary.json)

## Benchmark Policy

- Retained W7900 README/PARO/GGUF rows stay on TheRock ROCm 7.13 until a newer
  stack beats the relevant retained rows with the same correctness and evidence
  gates.
- Record package versions and `hipcc --version` in every benchmark artifact.
- Use `/tmp/hipengine-hipcc-version.txt` or a run-specific compiler-version file
  with `--require-cached-build` for profiled or repeated JIT benchmarks.
- Do not promote a new ROCm stack from one favorable shape. The update must be
  checked across PARO, GGUF, and any active MTP/DFlash rows affected by compiler
  or runtime behavior.
