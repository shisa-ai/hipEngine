# TheRock ROCm Environment

Last updated: 2026-06-15

This page is the retained setup for hipEngine W7900 / gfx1100 benchmark runs
that use AMD TheRock Python wheels. It records the install recipe, the package
flavor choice, verification commands, and the ROCm 7.14 nightly regression
diagnostics that keep ROCm 7.13 as the canonical stack for current topline rows.
The upstream release-package reference is TheRock
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

## Install Or Repair

Start with the pinned reinstall:

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
