---
status: current
owns: Stable ROCm 10.1 gfx1151 setup, upgrade, and rollback procedure, plus the retained W7900 ROCm 7.13 benchmark environment.
---
# TheRock ROCm Environment

Last updated: 2026-10-19

This page is the source of truth for hipEngine development and benchmark
processes that use AMD TheRock Python packages. It covers two independent lanes:

| Lane | Package stack | Status |
| --- | --- | --- |
| Strix Halo `gfx1151` (Framework Desktop, HP ZBook Ultra G1a) | Stable TheRock ROCm `10.1.0` in `~/mambaforge/envs/therock` | Current local development stack; SDK, fresh HIP JIT, PyTorch, torchvision, graph replay, and profiler smoke validated |
| Radeon Pro W7900 / RX 7900 XTX (`gfx1100`) | Legacy per-family TheRock ROCm `7.13.0a20260423` | Retained benchmark stack on a separate physical host; do not reinterpret old rates as ROCm 10 results |

A setup being operational does not promote old performance rows to a new ROCm
version. Benchmark promotion still requires the same-host, same-suite protocol in
[`BENCHMARK.md`](../BENCHMARK.md). Upstream packaging instructions live in TheRock
[`RELEASES.md`](https://github.com/ROCm/TheRock/blob/main/RELEASES.md); the
stable releases used here are
[`therock-10.1`](https://github.com/ROCm/TheRock/releases/tag/therock-10.1) and
[`therock-10.0`](https://github.com/ROCm/TheRock/releases/tag/therock-10.0).

## Current gfx1151 Stack: Stable ROCm 10.1

The validated local prefix and interpreter are:

```bash
ENV_PREFIX=~/mambaforge/envs/therock
PY=$ENV_PREFIX/bin/python
```

`therock` is a real environment directory in the `mambaforge` prefix, not a
symlink into a dated directory. Confirm the interpreter before trusting any path
in this page; a stale `~/miniforge3/envs/therock` reference from older notes does
not exist on this host.

The current package set is:

| Package | Version |
| --- | --- |
| Python | `3.12.12` |
| `rocm` | `10.1.0` |
| `rocm-sdk-core` | `10.1.0` |
| `rocm-sdk-devel` | `10.1.0` |
| `rocm-sdk-libraries` | `10.1.0` |
| `rocm-sdk-device-gfx1151` | `10.1.0` |
| `rocm-bootstrap` | `0.3.0` |
| `torch` | `2.14.0+rocm10.1.0` |
| `torchvision` | `0.29.0+rocm10.0.0` (see below) |
| `torchaudio` | `2.11.0.3+rocm10.1.0` |
| `triton` | `3.8.0+git669b31ac.rocm10.1.0` |
| `amd-torch-device-gfx1151` | `2.14.0+rocm10.1.0` |
| `amd-torch-device-gfx115x` | `2.14.0+rocm10.1.0` |
| `amd-torchvision-device-gfx1151` | `0.29.0+rocm10.0.0` (see below) |
| `numpy` | `1.26.4` (unchanged) |
| `pillow` | `10.4.0` (unchanged) |

The packages come from the stable multi-architecture aggregate index:

```text
https://stable.repo.amd.com/rocm/whl-next/
```

The `device-gfx1151` extras select the Radeon 8060S device packs. Do not install
`device-all` on this host unless a multi-GPU/multi-architecture environment is
actually required; it consumes substantially more disk and weakens environment
provenance.

### torchvision Runs One ROCm Point Release Behind

AMD never published `torchvision 0.29.0+rocm10.1.0` for any Python version. The
stable index carries `0.29.0` only as `+rocm10.0.0`, and `10.1.0` only as
`torchvision 0.28.0`. Because `0.28.0` ships a classic `_C.so` extension built
against torch `2.13`, while `0.29.0` ships the libtorch stable-ABI
`_C_stable.so` built against torch `2.14`, the correct pairing for a `2.14.0`
stack is `0.29.0+rocm10.0.0`.

Two verified facts make that cross-minor pairing safe here:

- The HIP runtime soname is `libamdhip64.so.7` in both ROCm `10.0.0` and
  `10.1.0`, so no dynamic-link break occurs.
- torchvision device kernels are not in the wheel. They ship in
  `torchvision/.kpack/torchvision_gfx1151.kpack` from
  `amd-torchvision-device-gfx1151` and are loaded by `kpack_load_code_object`
  inside `libamdhip64`.

The catch is packaging: `amd-torchvision-device-gfx1151==0.29.0+rocm10.0.0`
declares `rocm-sdk-device-gfx1151 ~= 10.0.0`, which excludes `10.1.0`. That pin
must be overridden with `--no-deps`; the exact command is in the upgrade section
below. `pip check` will keep reporting this one line, which is expected.

### ROCm Platform Version Versus HIP Component Version

Use `rocm-sdk version`, the ROCm core header, `rocprofv3`, or AMD SMI for the
ROCm platform release. The validated identities are:

```text
rocm-sdk version       10.1.0
ROCM_VERSION_*         10.1.0
rocprofv3 ROCm version 10.1.0 (profiler 1.4.1)
amd-smi ROCm version   10.1.0 (AMDSMI 27.1.0)
HIP version            7.16.26385-0000000
torch.version.hip      7.16.26385
AMD clang              24.0.0git, commit d6f6cb69186...
```

The stable ROCm 10 package intentionally contains a HIP component whose own
version tracks the HIP component release, not the platform release. Seeing
`hipcc --version` print HIP 7.16 is therefore not by itself evidence of a mixed
or failed ROCm 10.1 installation. Check the SDK and component package versions
together.

The upstream component table lists `LLVM 24.0.0` for both `10.0.0` and `10.1.0`,
but the shipped driver reported `AMD clang version 23.0.0git` under `10.0.0` and
`24.0.0git` under `10.1.0`. Trust `hipcc --version` and the
`lib/llvm/lib/clang/<N>` directory over the release-note table.

## Fresh gfx1151 Install

Prefer an isolated environment on a new machine. This host keeps ROCm 10 in the
`envs/therock` prefix, but a freshly created dated prefix is easier to replace
or roll back:

```bash
# create a dated prefix, then point the stable symlink at it
ENV_PREFIX=$HOME/mambaforge/envs/therock101-staging-$(date -u +%Y%m%d)
$HOME/mambaforge/bin/mamba create -y -p "$ENV_PREFIX" python=3.12 pip
ln -sfn "$ENV_PREFIX" "$HOME/mambaforge/envs/therock"
PY=$HOME/mambaforge/envs/therock/bin/python
INDEX=https://stable.repo.amd.com/rocm/whl-next/

"$PY" -m pip install \
  --index-url "$INDEX" \
  "rocm[libraries,devel,device-gfx1151]==10.1.0" \
  "torch[device-gfx1151]==2.14.0+rocm10.1.0" \
  "torchvision==0.29.0+rocm10.0.0" \
  "torchaudio==2.11.0.3+rocm10.1.0"

# torchvision's gfx1151 device pack is pinned to the 10.0.x SDK; override it.
"$PY" -m pip install --index-url "$INDEX" --no-deps \
  "amd-torchvision-device-gfx1151==0.29.0+rocm10.0.0"

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

Create a rollback clone and package manifests before an in-place upgrade. The
following pattern preserves both conda-managed and pip-installed state:

```bash
ENV_PREFIX=~/mambaforge/envs/therock
PY=$ENV_PREFIX/bin/python
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
STATE=$HOME/.local/state/therock-upgrades/$STAMP-rocm10
BACKUP=$HOME/mambaforge/envs/therock-rocm10.0-backup-$STAMP
mkdir -p "$STATE"

"$PY" -m pip freeze --all > "$STATE/pip-freeze-before.txt"
"$HOME/mambaforge/bin/conda" list --explicit -p "$ENV_PREFIX" \
  > "$STATE/conda-explicit-before.txt"
"$HOME/mambaforge/bin/conda" create -y -p "$BACKUP" --clone "$ENV_PREFIX"
"$BACKUP/bin/python" -m rocm_sdk version
"$BACKUP/bin/python" -m pip check
```

Dry-run the resolution first. The plan is the review artifact, and it is the only
cheap way to see what an upgrade will touch:

```bash
"$PY" -m pip install --dry-run --report /tmp/plan.json \
  --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "rocm[libraries,devel,device-gfx1151]==10.1.0" \
  "torch[device-gfx1151]==2.14.0+rocm10.1.0" \
  "torchvision==0.29.0+rocm10.0.0" \
  "torchaudio==2.11.0.3+rocm10.1.0"
```

Do **not** pass `--upgrade-strategy eager`. On this environment eager resolution
pulls `numpy 1.26.4 -> 2.5.3` plus `pillow`, `setuptools`, and `networkx` bumps,
which is a much larger blast radius than a ROCm update needs and puts
`onnxruntime-rocm`, `scipy`, and `transformers` at risk. The default
`only-if-needed` strategy moves only the ROCm/torch stack, Triton, and
`rocm-bootstrap`.

Then install, reinitialize the expanded devel tree, and add the torchvision
device pack:

```bash
INDEX=https://stable.repo.amd.com/rocm/whl-next/

"$PY" -m pip install --index-url "$INDEX" \
  "rocm[libraries,devel,device-gfx1151]==10.1.0" \
  "torch[device-gfx1151]==2.14.0+rocm10.1.0" \
  "torchvision==0.29.0+rocm10.0.0" \
  "torchaudio==2.11.0.3+rocm10.1.0"

"$PY" -m rocm_sdk init

"$PY" -m pip install --index-url "$INDEX" --no-deps \
  "amd-torchvision-device-gfx1151==0.29.0+rocm10.0.0"

"$PY" -m rocm_sdk test
"$PY" -m pip freeze --all > "$STATE/pip-freeze-after.txt"
```

`rocm_sdk init` only expands devel contents and device links inside
`site-packages`. It does not write `/opt/rocm`.

Keep the backup prefix until a representative full-model campaign passes on the
new ROCm release. The upgrade smoke proves the SDK, the JIT path, and the
framework packages, not every model and workload shape.

## gfx1151 Shell Setup

TheRock installs libraries inside the Python environment rather than
`/opt/rocm`. The current host's interactive-shell setup is equivalent to:

```bash
export HIPENGINE_HIP_ARCH=gfx1151
_ROCM_SITE=~/mambaforge/envs/therock/lib/python3.12/site-packages
_ROCM_LIBS="$_ROCM_SITE/_rocm_sdk_core/lib:$_ROCM_SITE/_rocm_sdk_devel/lib:$_ROCM_SITE/_rocm_sdk_libraries/lib"
export LD_LIBRARY_PATH="$_ROCM_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _ROCM_SITE _ROCM_LIBS
```

`~/mambaforge/envs/therock/bin` is the canonical prefix for this host.
For another prefix, construct paths from its interpreter rather than copying the
Python 3.12 path literally:

```bash
ENV_PREFIX=$HOME/mambaforge/envs/therock
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
ENV_PREFIX=~/mambaforge/envs/therock
PY=$ENV_PREFIX/bin/python
ROOT=$("$PY" -m rocm_sdk path --root)
SITE=$ENV_PREFIX/lib/python3.12/site-packages
ROCM_LIBS="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib"

"$ROOT/bin/hipcc" --version > /tmp/hipengine-gfx1151-rocm101-hipcc-version.txt

env -i HOME="$HOME" USER="$USER" LOGNAME="$LOGNAME" \
  SHELL="$SHELL" TERM="${TERM:-xterm}" \
  PATH="$ENV_PREFIX/bin:$ROOT/bin:$ROOT/lib/llvm/bin:/usr/local/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="$ROCM_LIBS" \
  HIP_PATH="$ROOT" ROCM_PATH="$ROOT" HIP_LIB_PATH="$ROOT/lib" \
  HIP_INCLUDE_PATH="$ROOT/include" \
  HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode" \
  HIPENGINE_HIP_ARCH=gfx1151 \
  HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-gfx1151-rocm101-hipcc-version.txt \
  PYTHONPATH=. \
  "$PY" <script> ...
```

The wrapper's fixed `PATH` is also what keeps the stale `/opt/rocm` tree out of a
benchmark run. See "Known Issues On ROCm 10.1.0" for why that matters.

## Verify gfx1151 ROCm 10.1

Run these checks after installation and before a benchmark campaign. The SDK
keeps its CLI tools (`hipcc`, `hipconfig`, `rocminfo`, `amd-smi`, `rocprofv3`)
and its shared libraries inside the prefix, so this block establishes `PATH` and
`LD_LIBRARY_PATH` first — the same exports the shell setup below installs. Both
matter: without `$ROOT/bin` on `PATH`, `rocm_sdk test` reports a `hipconfig`
`FileNotFoundError`, and without `LD_LIBRARY_PATH` the `libamdhip64.so` load at
the end fails.

```bash
ENV_PREFIX=~/mambaforge/envs/therock
PY=$ENV_PREFIX/bin/python
ROOT=$("$PY" -m rocm_sdk path --root)
SITE=$ENV_PREFIX/lib/python3.12/site-packages

export PATH="$ENV_PREFIX/bin:$ROOT/bin:$ROOT/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PY" -m pip check
"$PY" -m rocm_sdk version
"$PY" -m rocm_sdk targets | tr ';' '\n' | grep -x gfx1151
"$PY" -m rocm_sdk test
hipcc --version
rocminfo | grep -m 8 -E 'Name:|gfx'
amd-smi version
amd-smi list
rocprofv3 --version

"$PY" - <<'PY'
import ctypes
hip = ctypes.CDLL("libamdhip64.so")
count = ctypes.c_int()
rc = hip.hipGetDeviceCount(ctypes.byref(count))
print("hipGetDeviceCount", rc, count.value)
assert rc == 0 and count.value >= 1
PY
```

`pip check` is expected to report the `amd-torchvision-device-gfx1151`
`rocm-sdk-device-gfx1151~=10.0.0` line described above. On the 2026-10-19 host it
also reports two `marker-pdf` conflicts against `regex` and `transformers` that
predate the ROCm upgrade and are not caused by it.

The `PATH` and `LD_LIBRARY_PATH` exports above apply to the remaining checks in
this section; run them in the same shell. Compiled binaries such as the HIP
smoke test below need `LD_LIBRARY_PATH` at run time as well as compile time.

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

A torchvision check must exercise a device kernel, not only the import. `nms` and
`roi_align` both run compiled device code through the device pack:

```bash
"$PY" - <<'PY'
import torch
import torchvision
import torchvision.ops as ops

print("torchvision", torchvision.__version__)
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [20., 20., 30., 30.]], device="cuda")
scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
keep = ops.nms(boxes, scores, 0.5)
img = torch.rand(1, 3, 64, 64, device="cuda")
rois = [torch.tensor([[0., 0., 64., 64.], [10., 10., 50., 50.]], device="cuda")]
out = ops.roi_align(img, rois, spatial_scale=0.25, output_size=(8, 8), sampling_ratio=2, aligned=False)
torch.cuda.synchronize()
assert keep.tolist() == [0, 2]
assert tuple(out.shape) == (2, 3, 8, 8) and bool(torch.isfinite(out).all())
print("torchvision device kernels ok")
PY
```

Compile and launch a raw HIP kernel through the SDK toolchain:

```bash
cat > /tmp/therock-smoke.hip <<'HIP'
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void fill(int* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 4) out[i] = 2 + i;
}
int main() {
  int* d = nullptr;
  hipMalloc(&d, 4 * sizeof(int));
  fill<<<1, 4>>>(d);
  hipDeviceSynchronize();
  int host[4] = {0, 0, 0, 0};
  hipMemcpy(host, d, 4 * sizeof(int), hipMemcpyDeviceToHost);
  printf("%d %d %d %d\n", host[0], host[1], host[2], host[3]);
  hipFree(d);
  return !(host[0] == 2 && host[1] == 3 && host[2] == 4 && host[3] == 5);
}
HIP

"$ROOT/bin/hipcc" --offload-arch=gfx1151 \
  /tmp/therock-smoke.hip -o /tmp/therock-smoke
/tmp/therock-smoke
# Expected: 2 3 4 5
```

From a hipEngine source checkout, force one clean ROCm 10.1 JIT cache and graph
replay smoke:

```bash
HIPENGINE_HIP_ARCH=gfx1151 \
HIPENGINE_BUILD_CACHE_ROOT=/tmp/hipengine-rocm101-smoke-cache \
PYTHONPATH=. \
"$PY" -m pytest -q \
  tests/test_gpu_hip_runtime.py \
  tests/test_gpu_hip_graph_capture_replay.py
```

The 2026-10-19 upgrade result was `27` SDK tests passed in 25 s, `7` hipEngine
tests passed, `scripts/smoke.py --mode smoke-add-hip` returning `max_abs=0.0`
with `first5= [1.0, 4.0, 7.0, 10.0, 13.0]`, and `hipengine.LLM.generate()`
producing `' Paris.\nThe capital of France is Paris.\nThe capital of France is'`
for the prompt `The capital of France is` through `backend="hip_gfx1151"` on
`Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`. That cold generation took 234.7 s because it
includes the full 21 GB model load plus first-run JIT compiles; it is a
correctness smoke, not a throughput measurement. `rocprofv3 --version` reported
profiler `1.4.1`. The host upgrade and validation record is
[`20261018T225635.905692Z-lhl-therock-rocm-10-1-upgrade-2ce1e6.md`](../../worklog/entries/20261018T225635.905692Z-lhl-therock-rocm-10-1-upgrade-2ce1e6.md).
The previous ROCm 10.0.0 upgrade record is
[`20260830T135118.708318Z-pi-therock-rocm10-upgrade-9bad64.md`](../../worklog/entries/20260830T135118.708318Z-pi-therock-rocm10-upgrade-9bad64.md).

## Known Issues On ROCm 10.1.0

### `rocm-smi` Is Gone From The Stack

ROCm 10.1.0 removed ROCm SMI from the standard build. The `10.1.0`
`rocm-sdk-core` wheel installs `amd-smi` and no `rocm-smi` entry point; the
previous `10.0.0` prefix had both.

On this host the removal is masked rather than visible: a system ROCm `7.2.3`
install still provides `/opt/rocm/bin/rocm-smi` (ROCM-SMI `4.0.0`, SMI-LIB
`7.8.0`), which stays on the default `PATH`. Commands that call `rocm-smi`
therefore still succeed while reporting a library version that has nothing to do
with the 10.1.0 stack. Treat any `rocm-smi` output as untrusted provenance and
prefer `amd-smi` for device identity, clocks, and memory.

Effect by call site on this host:

| Call site | Behavior |
| --- | --- |
| `benchmarks/micro/collect_env.py` | Degrades safely. `run_command` checks `shutil.which` and records `available: false` with empty output when the binary is absent. With `/opt/rocm` on `PATH` it instead records the stale 7.2.3 tool, so `rocm_smi_lines` is populated but does not describe the active stack. `amd_smi_*` fields are the reliable ones. |
| `scripts/gemma4_campaign_bench.py` | Degrades safely; its `_run` catches `OSError` and returns an empty string. |
| `scripts/gemma4_scoreboard_snapshot.py` | **Hard failure.** Its `_run` does not catch `OSError`, so a missing `rocm-smi` raises and provenance capture aborts. On this host it resolves the stale `/opt/rocm` binary instead, so the snapshot records 7.2.3-era identity. |
| `scripts/gguf_c1_decode_attribution.py` | Degrades safely; guarded by `shutil.which`. |
| `scripts/gguf_pack8_dual_wmma_row_microbench.py` | Calls `rocm-smi` for GPU identity; same stale-binary caveat. |

### The cn Diagnostic Artifact Schema Still Requires `rocm-smi`

`scripts/qwen35_batch_artifact_schema.py` requires a `rocm-smi --showmeminfo vram
--showuse --showtemp` record in `commands.environment` and validates that
`hardware.rocm_smi.output` contains `GPU` and `VRAM` markers. A new c>N/A4
diagnostic artifact therefore cannot be produced honestly from the ROCm 10.1.0
prefix alone. Until that schema grows an `amd-smi` path, produce those artifacts
on a stack that still ships ROCm SMI, or extend the schema deliberately rather
than relaxing the marker check.

### Marker-PDF Conflicts Are Pre-Existing

`pip check` reports `marker-pdf 1.10.2` conflicting with the installed `regex`
`2026.5.9` and `transformers 5.8.1`. Both packages had those versions before the
ROCm upgrade; the conflict is unrelated to it and was left alone.

## JIT Cache And Profiler Migration

A ROCm/compiler update changes hipEngine JIT cache keys. After upgrading:

1. restart all processes that loaded old ROCm libraries;
2. regenerate every compiler-version file with the new `hipcc --version`;
3. run the intended build once **outside** `rocprofv3` without
   `--require-cached-build`;
4. rerun with the new compiler-version file and `--require-cached-build`;
5. only then profile the final child process.

Do not reuse a ROCm 7.15 or 10.0 compiler-version file with ROCm 10.1. Keep old
cache directories while rollback remains possible; the compiler identity already
separates old and new artifacts. Do not wrap MTP parent harnesses that launch
nested Python children in `rocprofv3`; use the dedicated final-child profiler
scripts documented in [`KERNELS.md`](../KERNELS.md).

ROCm 10.1.0 ships ROCprofiler-SDK `1.4.1` with kernel replay (beta), HIP event
tracing (`--hip-event-trace`), late tool attach, and reworked signal handling.
Existing `--kernel-trace` and `--selected-regions` invocations keep working, but
captured artifacts record a different profiler version and must not be compared
byte-for-byte against `1.3.5` captures.

## Rollback

For the current host, use the preserved ROCm 10.0.0 clone directly rather than
mixing its libraries into the ROCm 10.1 shell:

```bash
OLD=~/mambaforge/envs/therock-rocm10.0-backup-20261019
OLD_PY=$OLD/bin/python
OLD_ROOT=$("$OLD_PY" -m rocm_sdk path --root)
OLD_SITE=$OLD/lib/python3.12/site-packages

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

Do not point the normal ROCm 10.1 shell at the backup's libraries. If ROCm 10.1
must be removed from the default prefix entirely, recreate or clone a replacement
prefix and switch shell configuration atomically; do not partially downgrade
individual wheels in place.

## Retained W7900 / gfx1100 ROCm 7.13 Stack

The W7900 benchmark lane is a **different physical host**. The prefix below is
that machine's; on the gfx1151 host `~/mambaforge/envs/therock` is the ROCm 10.1
development stack documented above, so do not run these commands here.

```bash
PY=~/mambaforge/envs/therock/bin/python3.12
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
PY=~/mambaforge/envs/therock/bin/python3.12
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
PY=~/mambaforge/envs/therock/bin/python3.12
CONDA_PREFIX=~/mambaforge/envs/therock
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

- [PARO 7.14 diagnostic](../../benchmarks/results/2026-06-14-w7900-rocm714-hipengine-paro-packed-readme-persistent-5run-diagnostic.json)
- [GGUF 7.14 diagnostic](../../benchmarks/results/2026-06-14-w7900-rocm714-hipengine-gguf-q4ks-readme-persistent-5run-diagnostic.json)
- [MTP B1 7.14 diagnostic](../../benchmarks/results/2026-06-14-hipengine-mtp-b1-oldartifact-rocm714-3run-diagnostic.json)
- [Final-packed MTP 7.14 no-hold](../../benchmarks/results/2026-06-14-hipengine-mtp-finalpacked-rocm714-exactness-nohold.json)
- [Concurrency 7.14 diagnostic](../../benchmarks/results/2026-06-14-hipengine-qwen35-concurrency-decode-rocm714-w7900/summary.json)

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
