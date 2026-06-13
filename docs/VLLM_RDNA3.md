# vLLM on RDNA3 / gfx1100

This note tracks the local setup path for running vLLM on the W7900 / RX 7900
XTX class GPUs and the Q4 + MTP model candidates for comparison against
hipEngine and llama.cpp.

## Recommendation

Use Docker first. The local conda vLLM environment is not a good baseline on
this host: importing torch in `/home/lhl/mambaforge/envs/vllm` SIGILLs inside
its bundled `libhipsparselt.so`, and the local source checkouts have ROCm
extension ABI mismatches against the available torch stacks.

Preferred image order:

1. `vllm/vllm-openai-rocm:latest` - official vLLM ROCm image. This is the
   helper default so we pick up fast-moving ROCm/RDNA3 and Qwen MTP fixes.
2. `vllm/vllm-openai-rocm:v0.19.1` - pinned official baseline. Use this only
   when we need to reproduce the version named by the recommended Q4+MTP
   checkpoint's model card.
3. `rocm/vllm:rocm7.13.0_gfx110X-all_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1`
   - AMD image with an explicit gfx110X build. Current vLLM docs mark AMD
   images deprecated in favor of official vLLM images, but this is a useful
   RDNA3 fallback.
4. TheRock source build - only if Docker fails or we need local development.
   Rebuild vLLM extensions against the TheRock torch stack; do not reuse the
   current stale local extensions.

Run through the helper:

```bash
# Inspect the exact command.
scripts/vllm_rocm_gfx1100_docker.sh print

# Pull and run the latest official vLLM image on W7900 / HIP_VISIBLE_DEVICES=0.
scripts/vllm_rocm_gfx1100_docker.sh pull
scripts/vllm_rocm_gfx1100_docker.sh serve

# If latest regresses the Qwen3.6 GPTQ MTP loader, use the pinned image named
# by the model card.
scripts/vllm_rocm_gfx1100_docker.sh pull-pinned
scripts/vllm_rocm_gfx1100_docker.sh serve-pinned

# If the official images miss gfx1100 kernels or have a ROCm mismatch, try AMD's
# explicit gfx110X image.
scripts/vllm_rocm_gfx1100_docker.sh pull-amd
scripts/vllm_rocm_gfx1100_docker.sh serve-amd
```

The helper uses `sudo docker` by default because this host's Docker socket is
not accessible to the unprivileged user. Override with `DOCKER_BIN=docker` if
your user is in the docker group.

To pin a known-good vLLM image instead of using latest:

```bash
scripts/vllm_rocm_gfx1100_docker.sh serve-pinned

# Equivalent explicit override:
VLLM_ROCM_IMAGE=vllm/vllm-openai-rocm:v0.19.1 \
scripts/vllm_rocm_gfx1100_docker.sh serve
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
VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":2}' \
scripts/vllm_rocm_gfx1100_docker.sh serve
```

For the AWQ candidate, switch model and, if needed, the older method alias:

```bash
VLLM_MODEL=QuantTrio/Qwen3.6-35B-A3B-AWQ \
VLLM_SPECULATIVE_CONFIG='{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
scripts/vllm_rocm_gfx1100_docker.sh serve
```

Recent vLLM normalizes older Qwen MTP method aliases to `mtp`, so try the
default `{"method":"mtp","num_speculative_tokens":2}` first unless the image
is exactly following the AWQ model card command.

MTP settings to test:

```bash
# Default candidate, from the GPTQ+MTP model card.
VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":2}' \
scripts/vllm_rocm_gfx1100_docker.sh serve

# Lower-overhead MTP. Try this if repeated MTP layer forwards are slower.
VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}' \
scripts/vllm_rocm_gfx1100_docker.sh serve

# No-MTP baseline for measuring speculative speedup.
VLLM_SPECULATIVE_CONFIG= \
scripts/vllm_rocm_gfx1100_docker.sh serve
```

The helper defaults `VLLM_MAX_NUM_BATCHED_TOKENS=8192`. vLLM's OpenAI server
uses 2048 by default on this class of GPU, which is too low for the 8 x 512
prompt concurrency sweep and produces a speculative scheduling warning.

Known latest-image failure seen on 2026-06-13: GPTQ+MTP can fail while loading
the MTP drafter with `KeyError: 'layers.0.mlp.experts.w2_weight'`. That is a
vLLM/model-layout compatibility issue, not an MTP tuning issue. Try the pinned
`v0.19.1` image for MTP, and separately test no-MTP on pinned/latest:

```bash
# Pinned image, MTP enabled.
scripts/vllm_rocm_gfx1100_docker.sh serve-pinned

# Pinned image, no-MTP smoke test.
VLLM_SPECULATIVE_CONFIG= \
scripts/vllm_rocm_gfx1100_docker.sh serve-pinned

# Latest image, no-MTP baseline.
VLLM_SPECULATIVE_CONFIG= \
scripts/vllm_rocm_gfx1100_docker.sh serve
```

## Observed W7900 local vLLM results, 2026-06-13

Local source build: `v0.22.1rc1.dev499+g470229c37.d20260613`, served on
`http://127.0.0.1:8008`, `--dtype bfloat16 --max-model-len 128000
--gpu-memory-utilization 0.90 --enforce-eager`, no MTP.

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

Bottom line: vLLM works on W7900 for these GPTQ checkpoints without MTP. The
native Qwen3.6 MTP path is blocked by the vLLM `Qwen3_5MoeMTP` loader expecting
`layers.0.mlp.experts.w2_weight` keys. This affects both Int4 and GPTQ8, so it
is not solved by moving to Q8.

## Candidate Q4 + MTP models

### Primary: `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`

Use this first for vLLM MTP comparison.

Evidence from the model card / Hugging Face API:

- Base: `Qwen/Qwen3.6-35B-A3B`.
- Quantization: GPTQ Int4, group size 128, symmetric.
- MTP weights are included as BF16 and exposed as `mtp.safetensors`.
- The card says the MTP layout is verified with vLLM 0.19.1 and SGLang 0.5.10.
- The card's vLLM MTP command uses:
  `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`.
- Quantized size is listed as 24.4 GB including MTP weights, but plan for W7900
  rather than RX 7900 XTX because runtime overhead and KV cache make 24 GB tight.

Recommended first server command is the helper default:

```bash
scripts/vllm_rocm_gfx1100_docker.sh serve
```

The helper defaults to `VLLM_DTYPE=float16` for this model because vLLM rejects
GPTQ with `bfloat16` activations.

Then run the existing llama.cpp-style prompt suite against vLLM:

```bash
PYTHONPATH=. python3 scripts/mtp-bench.py \
  --url http://127.0.0.1:8000 \
  --temperature 0 --top-p 1 --no-cache-prompt --ignore-eos \
  --max-tokens 32 \
  --out /tmp/vllm-qwen36-gptq-mtp-d32.json
```

### Fallback Q4: `QuantTrio/Qwen3.6-35B-A3B-AWQ`

Use this if GPTQ loader or kernels fail on ROCm.

Evidence:

- Base: `Qwen/Qwen3.6-35B-A3B`.
- Quantization: AWQ 4-bit.
- Model card requires `vllm>=0.19.0` and includes a vLLM command with MTP:
  `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`.
- Hugging Face API reports `quantization_config.quant_method = awq`.
- vLLM ROCm platform code supports `awq` and automatically enables
  `VLLM_USE_TRITON_AWQ=1`.

Caveat: the AWQ repo does not expose a separate `mtp.safetensors` sibling in the
API listing, unlike the GPTQ repo. Treat it as a fallback until the loader proves
MTP is found at startup.

### Q8 / higher-quality fallback: `btbtyler09/Qwen3.6-35B-A3B-GPTQ-8bit`

Use this if Q4 loader paths are broken and W7900 memory is available. It is not
a Q4 apples-to-apples comparison, but it is a useful vLLM fallback.

Evidence from the model card / Hugging Face API:

- Base: `Qwen/Qwen3.6-35B-A3B`.
- Quantization: GPTQ 8-bit, group size 32, symmetric.
- The model card says the MTP module is preserved at BF16.
- Listed model size is about 40 GB, so this is W7900-only for our machines.
- Use `float16`; the card notes this is required for ROCm GPTQ kernels.

Start with no MTP to avoid the MTP drafter loader path:

```bash
VLLM_MODEL=btbtyler09/Qwen3.6-35B-A3B-GPTQ-8bit \
VLLM_SERVED_MODEL_NAME=qwen36-35b-a3b-gptq8 \
VLLM_GPU_MEMORY_UTILIZATION=0.95 \
VLLM_SPECULATIVE_CONFIG= \
scripts/vllm_rocm_gfx1100_docker.sh serve-pinned --skip-mm-profiling
```

If no-MTP starts cleanly, test MTP separately:

```bash
VLLM_MODEL=btbtyler09/Qwen3.6-35B-A3B-GPTQ-8bit \
VLLM_SERVED_MODEL_NAME=qwen36-35b-a3b-gptq8-mtp \
VLLM_GPU_MEMORY_UTILIZATION=0.95 \
VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}' \
scripts/vllm_rocm_gfx1100_docker.sh serve-pinned --skip-mm-profiling
```

### Non-Q4 fallback: `Qwen/Qwen3.6-35B-A3B-FP8`

Not a Q4 comparison, but useful for checking whether vLLM's Qwen3.6 + MTP path
works independent of AWQ/GPTQ quantization. The repo includes `mtp.safetensors`
and is an official Qwen FP8 release. vLLM's Qwen3.6 recipe targets H100/H200 or
MI300-class GPUs for FP8, so W7900 may run this slowly or fail if kernels assume
FP8 hardware paths.

A second FP8-like option is `RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic`, which uses
`compressed-tensors`, exposes `model_mtp.safetensors`, and is tagged as tested
against vLLM main.

### llama.cpp-only reference: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`

Good for llama.cpp MTP, not the first choice for vLLM. vLLM ROCm advertises GGUF
as a supported quantization family in current source, but the MTP GGUF path is
primarily a llama.cpp workflow and is not the cleanest vLLM comparison.

## Local failure signatures already observed

- `/home/lhl/mambaforge/envs/vllm`: torch import can SIGILL in
  `libhipsparselt.so.0` with `vmovups %zmm0`, which requires AVX512 on a Ryzen
  5950X that only has AVX2.
- `/home/lhl/vllm/vllm` with the sglang torch stack: `_rocm_C` fails with
  `libamdhip64.so.7: undefined symbol: hsa_amd_memory_get_preferred_copy_engine`.
- `/home/lhl/vllm/vllm` with TheRock torch: `_rocm_C` fails with
  `undefined symbol: c10::hip::getCurrentHIPStream`.
- Unprivileged Docker access currently fails with permission denied on
  `/var/run/docker.sock`; use `sudo docker` or add the user to the docker group.

## If we need a native TheRock build

Only do this after Docker validation. Start from a clean env with TheRock torch
and build vLLM extensions for gfx1100:

```bash
cd /home/lhl/vllm/vllm
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx1100
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
python -m pip install -U 'cmake<4' ninja wheel pybind11 Cython
python -m pip install -r requirements/common.txt
python -m pip install -e . --no-build-isolation --no-deps
```

This should be treated as a rebuild, not a fix-up of the existing stale env.
