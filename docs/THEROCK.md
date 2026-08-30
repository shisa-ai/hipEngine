# TheRock ROCm Environment

Last updated: 2026-08-30

This page is the source of truth for hipEngine development and benchmark
processes that use AMD TheRock Python packages. It covers two independent lanes:

| Lane | Package stack | Status |
| --- | --- | --- |
| HP ZBook Ultra G1a / Radeon 8060S (`gfx1151`) | Stable TheRock ROCm `10.0.0` in `/home/lhl/miniforge3` | Current local development stack; SDK, fresh HIP JIT, PyTorch, graph replay, and profiler smoke validated |
| Radeon Pro W7900 / RX 7900 XTX (`gfx1100`) | Legacy per-family TheRock ROCm `7.13.0a20260423` in `/home/lhl/mambaforge/envs/therock` | Retained benchmark stack; do not reinterpret old rates as ROCm 10 results |

A setup being operational does not promote old performance rows to a new ROCm
version. Benchmark promotion still requires the same-host, same-suite protocol in
[`BENCHMARK.md`](BENCHMARK.md). Upstream packaging instructions live in TheRock
[`RELEASES.md`](https://github.com/ROCm/TheRock/blob/main/RELEASES.md); the
stable release used here is
[`therock-10.0`](https://github.com/ROCm/TheRock/releases/tag/therock-10.0).

## Current gfx1151 Stack: Stable ROCm 10

The validated local prefix and interpreter are:

```bash
ENV_PREFIX=/home/lhl/miniforge3
PY=$ENV_PREFIX/bin/python
```

The current package set is:

| Package | Version |
| --- | --- |
| Python | `3.13.13` |
| `rocm` | `10.0.0` |
| `rocm-sdk-core` | `10.0.0` |
| `rocm-sdk-devel` | `10.0.0` |
| `rocm-sdk-libraries` | `10.0.0` |
| `rocm-sdk-device-gfx1151` | `10.0.0` |
| `torch` | `2.13.0+rocm10.0.0` |
| `torchvision` | `0.28.0+rocm10.0.0` |
| `torchaudio` | `2.11.0.2+rocm10.0.0` |
| `triton` | `3.8.0+git4cff872c.rocm10.0.0` |
| `amd-torch-device-gfx1151` | `2.13.0+rocm10.0.0` |
| `amd-torch-device-gfx115x` | `2.13.0+rocm10.0.0` |
| `amd-torchvision-device-gfx1151` | `0.28.0+rocm10.0.0` |

The packages come from the stable multi-architecture aggregate index:

```text
https://stable.repo.amd.com/rocm/whl-next/
```

The `device-gfx1151` extras select the Radeon 8060S device packs. Do not install
`device-all` on this host unless a multi-GPU/multi-architecture environment is
actually required; it consumes substantially more disk and weakens environment
provenance.

### ROCm Platform Version Versus HIP Component Version

Use `rocm-sdk version`, the ROCm core header, `rocprofv3`, or AMD SMI for the
ROCm platform release. The validated identities are:

```text
rocm-sdk version       10.0.0
ROCM_VERSION_*         10.0.0
rocprofv3 ROCm version 10.0.0
HIP version            7.15.26333-0000000
torch.version.hip      7.15.26333
AMD clang              23.0.0git, commit 8f497e0992f...
```

The stable ROCm 10 package intentionally contains a HIP component whose own
version is `7.15.26333`. Seeing `hipcc --version` print HIP 7.15 is therefore not
by itself evidence of a mixed or failed ROCm 10 installation. Check the SDK and
component package versions together.

## Fresh gfx1151 Install

Prefer an isolated environment on a new machine. The current host uses the
Miniforge base prefix shown above, but a dedicated prefix is easier to replace or
roll back:

```bash
ENV_PREFIX=$HOME/miniforge3/envs/therock-rocm10
$HOME/miniforge3/bin/mamba create -y -p "$ENV_PREFIX" python=3.13 pip
PY=$ENV_PREFIX/bin/python
INDEX=https://stable.repo.amd.com/rocm/whl-next/

"$PY" -m pip install \
  --index-url "$INDEX" \
  --upgrade --upgrade-strategy eager \
  "rocm[libraries,devel,device-gfx1151]==10.0.0" \
  "torch[device-gfx1151]==2.13.0+rocm10.0.0" \
  "torchvision[device-gfx1151]==0.28.0+rocm10.0.0" \
  "torchaudio==2.11.0.2+rocm10.0.0"

"$PY" -m rocm_sdk init
"$PY" -m rocm_sdk test
"$PY" -m pip check
```

Triton and the `amd-torch-device-*` packages are resolved by the matching
PyTorch extras; they do not need separate floating specs. hipEngine itself does
not import PyTorch on its generation hot path. The framework packages are kept
in this developer environment for reference checks and tools that require them.

For a source checkout with development tests:

```bash
git clone https://github.com/shisa-ai/hipEngine.git
cd hipEngine
git lfs install --local
git lfs pull
"$PY" -m pip install -e '.[dev]'
```

Do not use `scripts/update-therock-torch.sh` for this stable install. That helper
currently resolves dated **nightly** package versions from the legacy nightly
multi-arch index. Use the exact stable command above until the helper has an
explicit stable-release mode and corresponding tests.

## Upgrade An Existing gfx1151 Environment

Create a rollback prefix and package manifests before an in-place upgrade. The
following pattern preserves both conda-managed and pip-installed state:

```bash
ENV_PREFIX=/home/lhl/miniforge3
PY=$ENV_PREFIX/bin/python
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
STATE=$HOME/.local/state/therock-upgrades/$STAMP-rocm10
BACKUP=$HOME/.local/share/therock-backups/miniforge-before-rocm10-$STAMP
mkdir -p "$STATE" "$(dirname "$BACKUP")"

"$PY" -m pip freeze --all > "$STATE/pip-freeze-before.txt"
"$HOME/miniforge3/bin/conda" list --explicit -p "$ENV_PREFIX" \
  > "$STATE/conda-explicit-before.txt"
"$HOME/miniforge3/bin/conda" create -y -p "$BACKUP" --clone "$ENV_PREFIX"
"$BACKUP/bin/python" -m rocm_sdk version
"$BACKUP/bin/python" -m pip check
```

Remove the obsolete ROCm 7 nightly family pack before installing the stable
ROCm 10 package family. It is replaced by `amd-torch-device-gfx115x`:

```bash
"$PY" -m pip uninstall -y amd-torch-device-gfx11

INDEX=https://stable.repo.amd.com/rocm/whl-next/
"$PY" -m pip install \
  --index-url "$INDEX" \
  --upgrade --upgrade-strategy eager \
  "rocm[libraries,devel,device-gfx1151]==10.0.0" \
  "torch[device-gfx1151]==2.13.0+rocm10.0.0" \
  "torchvision[device-gfx1151]==0.28.0+rocm10.0.0" \
  "torchaudio==2.11.0.2+rocm10.0.0"

"$PY" -m rocm_sdk init
"$PY" -m rocm_sdk test
"$PY" -m pip check
"$PY" -m pip freeze --all > "$STATE/pip-freeze-after.txt"
```

The 2026-08-30 host upgrade retained its known-good ROCm 7.15 clone at:

```text
/home/lhl/.local/share/therock-backups/miniforge-rocm715-20260727
```

Its package/install records are under:

```text
/home/lhl/.local/state/therock-upgrades/2026-08-30-rocm715-to-rocm10
```

Keep that clone until a representative full-model campaign passes on ROCm 10.
The upgrade smoke proves the SDK and hipEngine JIT path, not every model and
workload shape.

## gfx1151 Shell Setup

TheRock installs libraries inside the Python environment rather than
`/opt/rocm`. The current host's interactive-shell setup is equivalent to:

```bash
export HIPENGINE_HIP_ARCH=gfx1151
_ROCM_SITE=/home/lhl/miniforge3/lib/python3.13/site-packages
_ROCM_LIBS="$_ROCM_SITE/_rocm_sdk_core/lib:$_ROCM_SITE/_rocm_sdk_devel/lib:$_ROCM_SITE/_rocm_sdk_libraries/lib"
export LD_LIBRARY_PATH="$_ROCM_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _ROCM_SITE _ROCM_LIBS
```

`/home/lhl/miniforge3/bin` is already on this host's `PATH`. For another prefix,
construct paths from its interpreter rather than copying the Python 3.13 path
literally:

```bash
ENV_PREFIX=$HOME/miniforge3/envs/therock-rocm10
PY=$ENV_PREFIX/bin/python
ROOT=$("$PY" -m rocm_sdk path --root)
SITE=$("$PY" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

export PATH="$ENV_PREFIX/bin:$ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIP_PATH="$ROOT"
export ROCM_PATH="$ROOT"
export HIP_LIB_PATH="$ROOT/lib"
export HIP_INCLUDE_PATH="$ROOT/include"
export HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode"
export HIPENGINE_HIP_ARCH=gfx1151
```

Do not set `HSA_OVERRIDE_GFX_VERSION` for a real gfx1151 device. Strix Halo uses
shared system memory; large-model runs also require an adequate host GTT/TTM
configuration. That host-level memory policy is separate from TheRock.

### Clean Process Wrapper

Use a clean process for benchmarks and profiling so system ROCm libraries cannot
leak into the run:

```bash
ENV_PREFIX=/home/lhl/miniforge3
PY=$ENV_PREFIX/bin/python
ROOT=$("$PY" -m rocm_sdk path --root)
SITE=$ENV_PREFIX/lib/python3.13/site-packages
ROCM_LIBS="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib"

"$ROOT/bin/hipcc" --version > /tmp/hipengine-gfx1151-rocm10-hipcc-version.txt

env -i HOME="$HOME" USER="$USER" LOGNAME="$LOGNAME" \
  SHELL="$SHELL" TERM="${TERM:-xterm}" \
  PATH="$ENV_PREFIX/bin:$ROOT/bin:$ROOT/lib/llvm/bin:/usr/local/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="$ROCM_LIBS" \
  HIP_PATH="$ROOT" ROCM_PATH="$ROOT" HIP_LIB_PATH="$ROOT/lib" \
  HIP_INCLUDE_PATH="$ROOT/include" \
  HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode" \
  HIPENGINE_HIP_ARCH=gfx1151 \
  HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-gfx1151-rocm10-hipcc-version.txt \
  PYTHONPATH=. \
  "$PY" <script> ...
```

## Verify gfx1151 ROCm 10

Run these checks after installation and before a benchmark campaign:

```bash
ENV_PREFIX=/home/lhl/miniforge3
PY=$ENV_PREFIX/bin/python
ROOT=$("$PY" -m rocm_sdk path --root)

"$PY" -m pip check
"$PY" -m rocm_sdk version
"$PY" -m rocm_sdk targets | tr ';' '\n' | grep -x gfx1151
"$PY" -m rocm_sdk test
"$ROOT/bin/hipcc" --version
rocminfo | grep -m 8 -E 'Name:|gfx'
amd-smi version
amd-smi list

"$PY" - <<'PY'
import ctypes
hip = ctypes.CDLL("libamdhip64.so")
count = ctypes.c_int()
rc = hip.hipGetDeviceCount(ctypes.byref(count))
print("hipGetDeviceCount", rc, count.value)
assert rc == 0 and count.value >= 1
PY
```

ROCm/HIP PyTorch builds expose AMD devices through the CUDA compatibility API;
there is no separate `torch.hip` device namespace. A minimal framework check is:

```bash
"$PY" - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.version.hip", torch.version.hip)
print("available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
y = x @ x.T
torch.cuda.synchronize()
assert y[3, 3].item() == 734.0
PY
```

For a real compiler/device launch independent of PyTorch:

```bash
cat > /tmp/therock-smoke.hip <<'HIP'
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void add_one(int* x) { x[threadIdx.x] += 1; }
int main() {
  int host[4] = {1, 2, 3, 4};
  int* device = nullptr;
  if (hipMalloc(&device, sizeof(host)) != hipSuccess) return 1;
  if (hipMemcpy(device, host, sizeof(host), hipMemcpyHostToDevice) != hipSuccess) return 2;
  hipLaunchKernelGGL(add_one, dim3(1), dim3(4), 0, 0, device);
  if (hipDeviceSynchronize() != hipSuccess) return 3;
  if (hipMemcpy(host, device, sizeof(host), hipMemcpyDeviceToHost) != hipSuccess) return 4;
  if (hipFree(device) != hipSuccess) return 5;
  std::printf("%d %d %d %d\n", host[0], host[1], host[2], host[3]);
  return !(host[0] == 2 && host[1] == 3 && host[2] == 4 && host[3] == 5);
}
HIP

"$ROOT/bin/hipcc" --offload-arch=gfx1151 \
  /tmp/therock-smoke.hip -o /tmp/therock-smoke
/tmp/therock-smoke
# Expected: 2 3 4 5
```

From a hipEngine source checkout, force one clean ROCm 10 JIT cache and graph
replay smoke:

```bash
HIPENGINE_HIP_ARCH=gfx1151 \
HIPENGINE_BUILD_CACHE_ROOT=/tmp/hipengine-rocm10-smoke-cache \
PYTHONPATH=. \
"$PY" -m pytest -q \
  tests/test_hip_runtime.py \
  tests/test_hip_graph_capture_replay.py
```

The 2026-08-30 upgrade result was `27` SDK tests and `6` hipEngine tests passed.
`rocprofv3 --version` reported profiler `1.3.5` and ROCm `10.0.0`. The exact
host upgrade and validation record is
[`20260830T135118.708318Z-pi-therock-rocm10-upgrade-9bad64.md`](../worklog/entries/20260830T135118.708318Z-pi-therock-rocm10-upgrade-9bad64.md).

## JIT Cache And Profiler Migration

A ROCm/compiler update changes hipEngine JIT cache keys. After upgrading:

1. restart all processes that loaded old ROCm libraries;
2. regenerate every compiler-version file with the new `hipcc --version`;
3. run the intended build once **outside** `rocprofv3` without
   `--require-cached-build`;
4. rerun with the new compiler-version file and `--require-cached-build`;
5. only then profile the final child process.

Do not reuse a ROCm 7.15 compiler-version file with ROCm 10. Keep old cache
directories while rollback remains possible; the compiler identity already
separates old and new artifacts. Do not wrap MTP parent harnesses that launch
nested Python children in `rocprofv3`; use the dedicated final-child profiler
scripts documented in [`KERNELS.md`](KERNELS.md).

## Rollback

For the current host, use the preserved ROCm 7.15 clone directly rather than
mixing its libraries into the ROCm 10 shell:

```bash
OLD=/home/lhl/.local/share/therock-backups/miniforge-rocm715-20260727
OLD_PY=$OLD/bin/python
OLD_ROOT=$("$OLD_PY" -m rocm_sdk path --root)
OLD_SITE=$OLD/lib/python3.13/site-packages

"$OLD_PY" -m rocm_sdk version
"$OLD_PY" -m pip check

env -i HOME="$HOME" USER="$USER" LOGNAME="$LOGNAME" \
  SHELL="$SHELL" TERM="${TERM:-xterm}" \
  PATH="$OLD/bin:$OLD_ROOT/bin:/usr/local/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="$OLD_SITE/_rocm_sdk_core/lib:$OLD_SITE/_rocm_sdk_devel/lib:$OLD_SITE/_rocm_sdk_libraries/lib" \
  HIP_PATH="$OLD_ROOT" ROCM_PATH="$OLD_ROOT" \
  HIP_DEVICE_LIB_PATH="$OLD_ROOT/lib/llvm/amdgcn/bitcode" \
  HIPENGINE_HIP_ARCH=gfx1151 PYTHONPATH=. \
  "$OLD_PY" <script> ...
```

Do not point the normal ROCm 10 shell at the backup's libraries. If ROCm 10 must
be removed from the default prefix entirely, recreate or clone a replacement
prefix and switch shell configuration atomically; do not partially downgrade
individual wheels in place.

## Retained W7900 / gfx1100 ROCm 7.13 Stack

The W7900 benchmark lane remains separate from the current gfx1151 setup:

```bash
PY=/home/lhl/mambaforge/envs/therock/bin/python3.12
```

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

Expected identity:

```text
HIP version: 7.13.26162-1140233ffe
torch.version.hip 7.13.26162
```

Retained host identity:

| Component | Value |
| --- | --- |
| Kernel | `Linux epyc 7.0.10-1-cachyos #1 SMP PREEMPT_DYNAMIC Sun, 24 May 2026 14:29:40 +0000 x86_64` |
| ROCm driver reported by `rocm-smi` | `7.0.10-1-cachyos` |
| GPU0 | AMD Radeon Pro W7900 / gfx1100, VBIOS `113-D7070100-138`, 44.984 GiB VRAM |
| GPU1 | AMD Radeon RX 7900 XTX / gfx1100, VBIOS `113-EXT89622-001`, 23.985 GiB VRAM |

This lane uses the legacy per-family index and package:

```text
https://rocm.nightlies.amd.com/v2/gfx110X-all/
rocm-sdk-libraries-gfx110X-all
```

Do not substitute `gfx1100-dgpu`, `gfx1100-all`, the stable ROCm 10 multi-arch
index, or a different host for a retained-row A/B. A new stack is a new evidence
lane until both baseline and candidate are measured under the same declared
protocol.

### Install Or Repair The Retained W7900 Stack

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

"$PY" -m pip uninstall -y \
  amd-torch-device-gfx1100 \
  amd-torch-device-gfx11 \
  amd-torchvision-device-gfx1100 \
  rocm-sdk-device-gfx1100 \
  rocm-sdk-libraries

"$PY" -m pip install --upgrade --force-reinstall --no-cache-dir \
  "numpy==2.1.3" "fsspec==2026.2.0"
```

The uninstall step removes stale multi-arch/7.14 helpers that can survive the
main reinstall and leave this legacy environment internally inconsistent.

### Verify And Run The Retained W7900 Stack

```bash
PY=/home/lhl/mambaforge/envs/therock/bin/python3.12
CONDA_PREFIX=/home/lhl/mambaforge/envs/therock
ROOT=$("$PY" -m rocm_sdk path --root)
SITE=$CONDA_PREFIX/lib/python3.12/site-packages

"$ROOT/bin/hipcc" --version > /tmp/hipengine-w7900-hipcc-version-713.txt
"$PY" -m pip list | grep -E '^(amd-|rocm|torch|torchvision|torchaudio|triton|numpy|fsspec|hipengine)'

env -i HOME="$HOME" USER="$USER" LOGNAME="$LOGNAME" \
  SHELL="$SHELL" TERM="${TERM:-xterm}" \
  PATH="$ROOT/bin:$ROOT/lib/llvm/bin:$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="$ROOT/lib:$ROOT/lib64:$ROOT/lib/llvm/lib:$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_libraries_gfx110X_all/lib" \
  HIP_PATH="$ROOT" ROCM_PATH="$ROOT" HIP_LIB_PATH="$ROOT/lib" \
  HIP_INCLUDE_PATH="$ROOT/include" \
  HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode" \
  HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-w7900-hipcc-version-713.txt \
  PYTHONPATH=. \
  "$PY" <script> ...
```

Only add `HSA_OVERRIDE_GFX_VERSION=11.0.0` as a measured local compatibility
workaround after rechecking the attached device. It is not a general hipEngine
default.

## Historical ROCm 7.14 W7900 Diagnostic

ROCm 7.14 nightly was tested on 2026-06-14 and was **not promoted** for retained
W7900 toplines. The result was mixed for PARO, negative for GGUF prefill, and
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

- [PARO 7.14 diagnostic](../benchmarks/results/2026-06-14-w7900-rocm714-hipengine-paro-packed-readme-persistent-5run-diagnostic.json)
- [GGUF 7.14 diagnostic](../benchmarks/results/2026-06-14-w7900-rocm714-hipengine-gguf-q4ks-readme-persistent-5run-diagnostic.json)
- [MTP B1 7.14 diagnostic](../benchmarks/results/2026-06-14-hipengine-mtp-b1-oldartifact-rocm714-3run-diagnostic.json)
- [Final-packed MTP 7.14 no-hold](../benchmarks/results/2026-06-14-hipengine-mtp-finalpacked-rocm714-exactness-nohold.json)
- [Concurrency 7.14 diagnostic](../benchmarks/results/2026-06-14-hipengine-qwen35-concurrency-decode-rocm714-w7900/summary.json)

## Benchmark Policy

- Current ROCm 10 gfx1151 smoke results establish environment health only; they
  do not replace ROCm 7.15 model-performance artifacts.
- Retained W7900 README/PARO/GGUF rows stay attributed to TheRock ROCm 7.13 until
  a newer stack measures both baseline and candidate on the same physical host
  with the complete applicable correctness gate.
- Record platform package versions, HIP component/compiler identity, GPU,
  physical host, and exact environment in every benchmark artifact.
- Generate a run-specific compiler-version file and use `--require-cached-build`
  only after the matching JIT artifacts were built outside the profiler.
- Do not promote a ROCm update from one favorable shape. Check all affected
  model/quant/workload categories, including active MTP/DFlash and long-context
  gates where applicable.
