# vLLM on RDNA3 / gfx1100

This note tracks the local setup path for running vLLM on RDNA3 (`gfx1100`,
W7900 / RX 7900 XTX class GPUs) and the Qwen3.6-35B-A3B Q4/MTP model
candidates used for comparison against hipEngine and llama.cpp.

Status as of 2026-06-13: the **native TheRock source build is the known-good
path on this host**. The Docker images remain useful for reproduction, but they
were not the best path here: no-MTP serving worked in the pinned image, while
Qwen3.6 MTP loading failed inside the container, and the images do not use our
local TheRock torch/ROCm stack or the local `gfx1100` GPTQ build patch.

For the detailed build log/recipe, also see `/home/lhl/vllm/BUILD-gfx1100.md`.
This file keeps the hipEngine-facing summary, model table, and benchmark notes.

## TL;DR recommendation

Use the local `vllm` conda env with TheRock torch and the custom vLLM source
build:

- env: `/home/lhl/mambaforge/envs/vllm`
- Python: 3.12
- torch: `2.12.0a0+rocm7.13.0a20260416`
- Triton: `3.7.0+git4089ddfa.rocm7.13.0a20260416`
- TheRock ROCm package: `rocm 7.13.0a20260416`
- vLLM source: `/home/lhl/vllm/vllm-main`
- tested source commit: `470229c37efa` from upstream `origin/main`
- installed wheel:
  `vllm-0.22.1rc1.dev499+g470229c37.d20260613.rocm713`
- local patch needed for tested HEAD on `gfx1100`:
  `/home/lhl/vllm/patches/vllm-gfx1100-gptq-half-atomicadd.patch`

The source build imported both `vllm._C` and `vllm._rocm_C`, detected
`PlatformEnum.ROCM`, passed a small `Qwen/Qwen3-0.6B` offline/server smoke test,
and served `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` without MTP for the benchmark
below.

## Why native source build beat the Docker images here

The official/AMD ROCm containers are convenient, but they are self-contained
software stacks. On this host we already have a working TheRock torch/ROCm stack
for `gfx110X`, and the successful build depended on matching vLLM extensions to
that exact stack.

Observed differences:

- The native build uses current vLLM `main`/HEAD instead of the older pinned
  container baseline.
- The native build compiles only the local arch via `PYTORCH_ROCM_ARCH=gfx1100`.
- The native build uses TheRock's initialized `_rocm_sdk_devel` tree, not a
  container ROCm install and not a mismatched `/opt/rocm`.
- The tested HEAD needed a local patch for generic GPTQ `atomicAdd(half*)` /
  `atomicAdd(half2*)` compilation on `gfx1100`.
- Docker `vllm/vllm-openai-rocm:v0.19.1` could run no-MTP GPTQ, but MTP startup
  failed with `KeyError: layers.0.mlp.experts.w2_weight` in the Qwen MTP drafter
  loader.

Bottom line: use Docker to reproduce container behavior, but use the TheRock
source build as the reference path for hipEngine comparisons on this machine.

## Native TheRock source-build recipe

Run these commands in the `vllm` env. Keep this section aligned with
`/home/lhl/vllm/BUILD-gfx1100.md`.

### 1. Verify env and GPU arch

```bash
conda activate vllm

python - <<'PY'
import sys
import torch
print("python:", sys.version.split()[0], sys.executable)
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("gpu:", torch.cuda.get_device_name(0))
    print("gcn:", props.gcnArchName)
PY
```

Expected local arch: `gfx1100`.

### 2. Use TheRock `rocm[devel]`, not system `/opt/rocm`

`_rocm_sdk_core` was not enough for this vLLM build because it lacked CMake's
`hip-lang-config.cmake`. Install the matching TheRock devel package and
initialize it once:

```bash
ROCM_PY_VERSION="$(python - <<'PY'
from importlib.metadata import version
print(version("rocm"))
PY
)"

python -m pip --isolated install --pre \
  --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ \
  "rocm[devel]==${ROCM_PY_VERSION}"

rocm-sdk init
```

Then export these in every build/run shell:

```bash
export THEROCK_ROCM_HOME="$(rocm-sdk path --root)"
export ROCM_HOME="$THEROCK_ROCM_HOME"
export ROCM_PATH="$THEROCK_ROCM_HOME"
export HIP_PATH="$THEROCK_ROCM_HOME"
export PATH="$(rocm-sdk path --bin):$THEROCK_ROCM_HOME/bin:$THEROCK_ROCM_HOME/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$THEROCK_ROCM_HOME/lib:$(python -c 'import sys; print(sys.prefix)')/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$THEROCK_ROCM_HOME:$(python -c 'import sys; print(sys.prefix)')${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

hipcc --version
test -f "$THEROCK_ROCM_HOME/lib/cmake/hip-lang/hip-lang-config.cmake"
```

### 3. Install build prerequisites and matching AMDSMI

```bash
python -m pip install -U pip
python -m pip install -U \
  "cmake<4" ninja wheel pybind11 Cython packaging \
  "setuptools>=77.0.3,<80.0.0" setuptools-scm setuptools-rust jinja2

python -m pip install --no-build-isolation --no-deps "$THEROCK_ROCM_HOME/share/amd_smi"

PY_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
ln -sfn "$THEROCK_ROCM_HOME/lib/libamd_smi.so.26" "$PY_PREFIX/lib/libamd_smi.so.26"
ln -sfn libamd_smi.so.26 "$PY_PREFIX/lib/libamd_smi.so"
```

`amdsmi` is required by current vLLM ROCm platform detection.

### 4. Install vLLM deps without replacing TheRock torch/triton

```bash
cd /home/lhl/vllm/vllm-main

python - <<'PY' >/tmp/keep-therock-torch.txt
from importlib.metadata import PackageNotFoundError, version
for dist in ("torch", "torchvision", "torchaudio", "triton", "rocm"):
    try:
        print(f"{dist}=={version(dist)}")
    except PackageNotFoundError:
        pass
PY
cat /tmp/keep-therock-torch.txt

python -m pip install \
  -r requirements/rocm.txt \
  -c /tmp/keep-therock-torch.txt \
  --upgrade-strategy only-if-needed
```

If pip tries to uninstall or replace TheRock `torch`/`triton`, stop and fix the
resolver inputs. The final vLLM wheel should also be installed with `--no-deps`.

### 5. Patch, build, and install the vLLM wheel

```bash
cd /home/lhl/vllm/vllm-main

export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx1100
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export HSA_NO_SCRATCH_RECLAIM=1

PATCH=/home/lhl/vllm/patches/vllm-gfx1100-gptq-half-atomicadd.patch
if grep -q "gfx11 does not expose native atomicAdd overloads" \
    csrc/libtorch_stable/quantization/gptq/q_gemm.cu; then
  echo "gfx1100 GPTQ atomicAdd patch already applied"
elif git apply --check "$PATCH"; then
  git apply "$PATCH"
else
  echo "Patch did not apply; upstream may have changed. Only continue if q_gemm.cu builds on gfx1100."
fi

rm -rf build dist
python setup.py clean --all
python setup.py bdist_wheel --dist-dir=dist

wheel="$(ls -t dist/vllm-*.whl | head -1)"
python -m pip uninstall -y vllm
python -m pip install --no-deps --force-reinstall "$wheel"
```

The tested unpatched HEAD failed on `gfx1100` while compiling
`csrc/libtorch_stable/quantization/gptq/q_gemm.cu` with missing
`atomicAdd(half*)` / `atomicAdd(half2*)` overloads.

### 6. Validate install

```bash
python - <<'PY'
import torch
import vllm
from vllm.platforms import current_platform

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("vllm:", vllm.__version__)
print("platform:", current_platform._enum)

import vllm._C
import vllm._rocm_C
print("_C:", vllm._C.__file__)
print("_rocm_C:", vllm._rocm_C.__file__)
PY
```

Expected: `PlatformEnum.ROCM`, and both extension imports succeed.

Small server smoke:

```bash
export TORCHINDUCTOR_AUTOGRAD_CACHE=0

vllm serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 8008 \
  --served-model-name qwen-test \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.55 \
  --enforce-eager
```

Then:

```bash
curl -s http://127.0.0.1:8008/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-test","prompt":"Say hello from vLLM on ROCm.","max_tokens":12,"temperature":0}' \
  | python -m json.tool
```

Notes:

- `TORCHINDUCTOR_AUTOGRAD_CACHE=0` works around a PyTorch AOTAutograd cache
  pickling issue observed with this TheRock stack in compiled mode.
- Use `--enforce-eager` for quick smoke tests. For non-eager runs, keep
  `TORCHINDUCTOR_AUTOGRAD_CACHE=0` and expect compile/CUDAGraph warmup.
- vLLM logs may say `device_config=cuda`; HIP still uses the PyTorch `cuda`
  namespace.
- On RDNA3 it is normal to see fallback from ROCm custom paged attention to
  Triton attention.

## Serving Qwen3.6-35B-A3B on the native build

No-MTP baseline first:

```bash
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export HSA_NO_SCRATCH_RECLAIM=1

vllm serve palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 \
  --host 127.0.0.1 --port 8008 \
  --served-model-name qwen36-gptq-int4 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

If a specific GPTQ/AWQ checkpoint rejects `bfloat16`, retry with
`--dtype float16`. Older Docker/model-card recipes used `float16` for ROCm GPTQ.

MTP is separate from checkpoint MTP availability. Many Qwen3.6 checkpoints have
MTP weights, but the vLLM MTP loader path must still be validated per checkpoint
and vLLM revision. Start with one speculative token on HEAD even if older model
cards show two:

```bash
vllm serve palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 \
  --host 127.0.0.1 --port 8008 \
  --served-model-name qwen36-gptq-int4-mtp \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --spec-method mtp \
  --spec-tokens 1
```

Older CLI/model-card equivalent:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Always keep a no-MTP baseline because MTP can lose on small batches or hit loader
layout issues.

## Qwen3.6-35B-A3B model candidates and sizes

Sizes below are approximate GiB of model tensor files (`.safetensors`/`.bin`) as
reported by the Hugging Face API. They do **not** include runtime allocator
fragmentation, compiled kernels, activations, KV cache, or server overhead.

This matters for 24GB cards: most useful Q4 Qwen3.6-35B-A3B checkpoints are
already ~22-24 GiB on disk. They are W7900/48GB-friendly, but a 24GB RX 7900 XTX
will usually have too little memory left for KV cache after loading the model.
The few ~19-21 GiB derivative checkpoints may fit only at short context and are
lower-confidence for MTP.

### Primary vLLM candidates

| Model | Format | Tensor size | MTP checkpoint weights? | 24GB card outlook | Notes |
|---|---:|---:|---|---|---|
| `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` | GPTQ int4, g128 | ~22.74 GiB | Yes; clean `mtp.safetensors`, 785 `mtp.*` keys | Very tight / likely no | Best first GPTQ target; MTP excluded from GPTQ quantization. Native no-MTP benchmark used this model. |
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | AWQ int4, g32 | ~23.25 GiB | Yes; 2321 MTP-related packed keys | Very tight / likely no | Major AWQ candidate we initially missed; validate loader because MTP appears in packed AWQ layout. |
| `QuantTrio/Qwen3.6-35B-A3B-AWQ` | AWQ int4, g128 | ~23.71 GiB | Yes; 785 `mtp.*` keys | No | Good AWQ fallback; quant config excludes MTP/attention/shared expert from AWQ. |
| `Qwen/Qwen3.6-35B-A3B-FP8` | FP8 | ~34.89 GiB | Yes; `mtp.safetensors`, 1560 MTP keys | No | Official Qwen FP8; useful to test Qwen3.6+MTP independent of Q4, but not a Q4 comparison. |
| `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | compressed-tensors NVFP4 | ~23.32 GiB | Yes; fused MTP layout, `model_mtp.safetensors` | Very tight / likely no | Interesting FP4 route; vLLM supports compressed-tensors/modelopt families, but validate on `gfx1100`. |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | ModelOpt NVFP4 | ~21.82 GiB | Yes; fused/minimal MTP keys | Tight | NVIDIA-oriented quantization; lower priority on ROCm/RDNA3. |
| `unsloth/Qwen3.6-35B-A3B-NVFP4` | compressed-tensors NVFP4 | ~22.99 GiB | Config says MTP; index not available in API check | Very tight / likely no | Single safetensors file; validate before relying on MTP. |
| `z-lab/Qwen3.6-35B-A3B-DFlash` | DFlash draft model | ~0.88 GiB | Draft/spec model, not main model | N/A | Potential drafter/speculative component, not a standalone 35B target. |

### Other discovered 3.6 Q4-ish/derivative safetensor variants

| Model | Format | Tensor size | MTP checkpoint weights? | 24GB card outlook | Notes |
|---|---:|---:|---|---|---|
| `btbtyler09/Qwen3.6-35B-A3B-GPTQ-4bit` | GPTQ int4, g32 | ~24.50 GiB | Fused/minimal MTP keys only | No | Larger than the 24GB card budget before KV cache. |
| `btbtyler09/Qwen3.6-35B-A3B-GPTQ-8bit` | GPTQ int8, g32 | ~39.97 GiB | Fused/minimal MTP keys only | No | W7900-only fallback; Docker no-MTP loaded, MTP failed with same drafter key error. |
| `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-GPTQ-Int4` | GPTQ int4, g128 | ~20.81 GiB | Fused/minimal MTP keys | Tight / possible short ctx | Derivative; lower confidence than palmfuture/QuantTrio. |
| `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GPTQ-Int4` | GPTQ int4, g128 | ~19.23 GiB | No actual MTP keys found | Possible short ctx | Smaller, but not useful for MTP comparison. |
| `groxaxo/Qwen3.6-35B-A3B-GPTQ-Pro-FOEM-4bit-g128` | GPTQ-like | ~20.81 GiB | Fused/minimal MTP keys | Tight / possible short ctx | Quant config missing in API check; risky. |
| `Sociopacific/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-GPTQ-Int4` | GPTQ int4, g128 | ~21.17 GiB | No actual MTP keys found | Tight / possible short ctx | Distilled derivative; no MTP despite config fields. |
| `Sociopacific/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-GPTQ-Int4` | GPTQ int4, g128 | ~21.17 GiB | No actual MTP keys found | Tight / possible short ctx | Same caveat as 4.7 derivative. |
| `abhishekchohan/Qwen3.6-35B-A3B-Abliterated-AWQ` | compressed-tensors int4 | ~21.90 GiB | Fused/minimal MTP keys | Tight | Single-file derivative; lower priority. |
| `genevera/Qwen3.6-35B-A3B-Abliterated-Heretic-AWQ-4bit` | AWQ int4, g128 | ~19.31 GiB | No actual MTP keys found | Possible short ctx | Smaller derivative, but not an MTP target. |
| `Civitai/Qwen3.6-35B-A3B-Abliterated-AWQ` | compressed-tensors int4 | ~20.32 GiB | Not confirmed | Tight / possible short ctx | API header read timed out; validate before use. |
| `feanors/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-AWQ-INT4` | AWQ int4, g128 | ~23.67 GiB | Fused/minimal MTP keys | No | Distilled derivative; too large for 24GB with KV cache. |
| `mattbucci/Qwen3.6-35B-A3B-AWQ` | AWQ int4, g128 | ~19.05 GiB | No actual MTP keys found | Possible short ctx | Smaller but not an MTP target. |
| `mattbucci/Qwen3.6-35B-A3B-AWQ-CT` | compressed-tensors int4 | ~18.93 GiB | No actual MTP keys found | Possible short ctx | One of the more 24GB-plausible sizes, but no MTP. |
| `Chunity/Qwen3.6-35B-A3B-AutoRound-AWQ-4bit` | AWQ int4, g128 | ~22.56 GiB | Fused/minimal MTP keys | Very tight / likely no | AutoRound derivative. |
| `selode-ai/Qwen-3.6-35B-A3B-VRAP-4-bit-AWQ-21.2GB` | AWQ int4, g128 | ~19.71 GiB | 617 MTP keys | Possible short ctx | Interesting smaller AWQ, but derivative and less proven. |
| `FenomAI/Qwen3.6-35B-A3B-AWQ-4bit` | compressed-tensors int4, g32 | ~22.78 GiB | MTP-related packed keys | Very tight / likely no | Similar packed layout to cyankiwi, lower downloads. |

For a 24GB card, llama.cpp/GGUF Q4 variants remain more realistic for local
experiments because llama.cpp has lower runtime overhead and more explicit
context/offload controls. For vLLM, treat W7900/48GB as the target for 35B-A3B
Q4+MTP work.

## Observed W7900 native vLLM results, 2026-06-13

Local source build: `v0.22.1rc1.dev499+g470229c37.d20260613`, served on
`http://127.0.0.1:8008`, `--max-model-len 128000 --gpu-memory-utilization 0.90
--enforce-eager`, no MTP.

Concurrency sweep used exact 512-token prompt-id rows from
`/tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json`, 128 output
tokens, OpenAI `/v1/completions`, 3 reps.

| c | aggregate tok/s, wall median | per-seq tok/s | aggregate tok/s, post-TTFT approx |
|---:|---:|---:|---:|
| 1 | 19.39 | 19.39 | 19.93 |
| 2 | 37.53 | 18.77 | 39.07 |
| 4 | 72.96 | 18.24 | 77.48 |
| 8 | 115.96 | 14.49 | 125.98 |

Artifact:

```text
benchmarks/results/2026-06-13-vllm-localbuild-gptq-int4-concurrency-c1-c8-w7900.json
```

Caveat: OpenAI responses do not include llama.cpp-style pure decode timings, so
the wall metric includes prompt prefill and HTTP scheduling. The post-TTFT value
is derived from vLLM Prometheus histogram deltas.

## Observed W7900 Docker smoke results, 2026-06-13

Image: `vllm/vllm-openai-rocm:v0.19.1`, `HIP_VISIBLE_DEVICES=0`,
`--max-model-len 8192`, `--max-num-seqs 8`, `--max-num-batched-tokens 8192`.

| model | MTP | startup | prompt-suite agg tok/s | notes |
|---|---:|---|---:|---|
| `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` | off | pass | 44.10 | model load 21.06 GiB, KV cache 198,528 tokens |
| `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4` | n=2 | fail | - | `KeyError: layers.0.mlp.experts.w2_weight` in MTP drafter load |
| `btbtyler09/Qwen3.6-35B-A3B-GPTQ-8bit` | off | pass | 39.72 | model load 37.46 GiB, KV cache 40,128 tokens |
| `btbtyler09/Qwen3.6-35B-A3B-GPTQ-8bit` | n=1 | fail | - | same MTP drafter loader key error |

Artifacts:

```text
benchmarks/results/2026-06-13-vllm-rocm-w7900-smoke-summary.json
benchmarks/results/2026-06-13-vllm-pinned-gptq-int4-nomtp-smoke.json
benchmarks/results/2026-06-13-vllm-pinned-gptq8-nomtp-smoke-rerun.json
```

Bottom line: the pinned Docker image could serve these GPTQ checkpoints without
MTP, but its Qwen3.6 MTP path was blocked by the vLLM `Qwen3_5MoeMTP` loader
expecting `layers.0.mlp.experts.w2_weight` keys. This affected both Int4 and
GPTQ8, so it was not solved by moving to Q8.

## Docker helper status

The helper remains in the repo for reproducing container behavior:

```bash
# Inspect the exact command.
scripts/vllm_rocm_gfx1100_docker.sh print

# Official latest image.
scripts/vllm_rocm_gfx1100_docker.sh pull
scripts/vllm_rocm_gfx1100_docker.sh serve

# Pinned official baseline.
scripts/vllm_rocm_gfx1100_docker.sh pull-pinned
scripts/vllm_rocm_gfx1100_docker.sh serve-pinned

# AMD explicit gfx110X image.
scripts/vllm_rocm_gfx1100_docker.sh pull-amd
scripts/vllm_rocm_gfx1100_docker.sh serve-amd
```

Useful overrides:

```bash
HIP_VISIBLE_DEVICES=0 \
VLLM_MODEL=palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 \
VLLM_MAX_MODEL_LEN=8192 \
VLLM_MAX_NUM_SEQS=8 \
VLLM_MAX_NUM_BATCHED_TOKENS=8192 \
VLLM_GPU_MEMORY_UTILIZATION=0.88 \
VLLM_DTYPE=float16 \
VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}' \
scripts/vllm_rocm_gfx1100_docker.sh serve-pinned
```

Disable MTP for baseline/repro:

```bash
VLLM_SPECULATIVE_CONFIG= \
scripts/vllm_rocm_gfx1100_docker.sh serve-pinned
```

The helper uses `sudo docker` by default because this host's Docker socket is
not accessible to the unprivileged user. Override with `DOCKER_BIN=docker` if
your user is in the docker group.

## Failure signatures already observed

Older/stale local and container attempts hit these issues:

- Old `/home/lhl/mambaforge/envs/vllm` state: torch import could SIGILL in
  `libhipsparselt.so.0` with `vmovups %zmm0`, which requires AVX512 on a Ryzen
  5950X that only has AVX2. The current TheRock env no longer follows that
  stale path.
- `/home/lhl/vllm/vllm` with the sglang torch stack: `_rocm_C` failed with
  `libamdhip64.so.7: undefined symbol: hsa_amd_memory_get_preferred_copy_engine`.
- Stale local vLLM extensions with TheRock torch: `_rocm_C` failed with
  `undefined symbol: c10::hip::getCurrentHIPStream`. Rebuilding vLLM extensions
  against the active TheRock torch fixed this class of mismatch.
- Unprivileged Docker access currently fails with permission denied on
  `/var/run/docker.sock`; use `sudo docker` or add the user to the docker group.
- Docker `v0.19.1` Qwen3.6 MTP drafter load failed with
  `KeyError: layers.0.mlp.experts.w2_weight`.
- Native unpatched vLLM HEAD on `gfx1100` failed building generic GPTQ kernels
  with missing `atomicAdd(half*)` / `atomicAdd(half2*)`; apply the local patch
  before building that revision.

## Troubleshooting native build/runtime

- If `hipcc --version` does not match TheRock torch, re-export `ROCM_HOME`,
  `ROCM_PATH`, `HIP_PATH`, `PATH`, and `LD_LIBRARY_PATH` from the build recipe.
- If vLLM reports `UnspecifiedPlatform`, reinstall/verify AMDSMI from
  `$THEROCK_ROCM_HOME/share/amd_smi` and ensure `libamd_smi.so*` is visible.
- If CMake cannot find `hip-lang-config.cmake`, you are pointing at
  `_rocm_sdk_core` or system ROCm instead of TheRock `_rocm_sdk_devel`.
- If `torch.compile` fails with a launcher pickling error, set
  `TORCHINDUCTOR_AUTOGRAD_CACHE=0` or use `--enforce-eager`.
- If pip wants to replace `torch`/`triton`, stop; use the constraints file from
  the recipe and install the final vLLM wheel with `--no-deps`.
