# hipEngine

hipEngine is a ROCm-native local inference engine built primarily for AMD 
Radeon GPUs. It pairs a small Python host with custom HIP kernels for torch-free 
model loading, generation, and OpenAI-compatible serving on supported hardware.

**Current release: v0.4.0 alpha.** Besides the Qwen 3.6 PARO and GGUF models,
the latest version of hipEngine now supports GGUF inference of more model
families. This includes [Laguna S 2.1](https://poolside.ai/blog/introducing-laguna-s-2-1),
[Maple ternary](https://github.com/deepgrove-ai/mlx-lm-deepgrove) and [Moonshine ASR](https://github.com/moonshine-ai/moonshine).

## Why use hipEngine?

- **Native AMD support.** HIP-first kernels directly target and tune for specific
  RDNA 3 (gfx1100) and Strix Halo RDNA 3.5 (gfx1151) instead of bein CUDA ports.
- **No PyTorch runtime required.** there is no PyTorch dependency, keeping 
  hipEngine lightweight (also, while Python-packaged, almost all of the hotpath
  is C++.
- **Agent and multi-request concurrency optimized.** Besides extensive tuning for
  fast single-request performance (especially for prefill), hipEngine also has
  tuned c=N support. It's significantly faster than llama.cpp or vLLM when it
  comes to c=8 results.
- **Drop in with existing clients.** The included OpenAI-compatible server 
  supports completion, chat, token-level SSE, logprobs, tools, structured-output
  validation, Qwen thinking controls, logprob-biased efforc contro, and 
  request diagnoticts.

hipEngine is new/small software project and is focused on make a select list of
models perform well (particularly Qwen 3.x variants/fine tunes).

## Supported models

`Yes` means that public text generation has been tested. A dash means that the
combination is not supported. Features such as batching, sampling, tools, and
long context can differ by model.

| Model | Formats | RX 7900 XTX / W7900 (`gfx1100`) | Radeon 8060S (`gfx1151`) | NVIDIA Blackwell (`sm_120a`) |
| --- | --- | :---: | :---: | :---: |
| [Qwen3.5 0.8B](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF) | `Q4_K_M`, `Q8_0`, `Q4_1`, `UD-Q4_K_XL` | Yes | Yes | — |
| [Qwen3.5 35B-A3B](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) and [Qwen3.6 35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | `Q4_K_M`, `Q4_K_S`, `UD-Q3_K_M`, `UD-Q4_K_M` | Yes | Yes | — |
| [Qwen3.6 35B-A3B ParoQuant](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-packed) | ParoQuant W4 | Yes | Yes | — |
| [Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1-GGUF) | `Q4_K_M` | — | Yes | — |
| [Maple-Preview 20B-A1B](https://huggingface.co/deepgrove/maple-preview-2bit-mlx) | 2-bit MLX | Yes | Yes | Python API only |

CPU model generation is not supported. The CPU backend is used for correctness
tests. On NVIDIA, load Maple with `backend="cuda_sm120a"`; automatic hardware
selection currently covers AMD only.

Support is specific to the listed model families and formats. hipEngine does
not yet run every GGUF model. See the [GGUF](docs/GGUF.md),
[Laguna](docs/LAGUNA.md), and [Maple](docs/MAPLE.md) guides for model-specific
limits.

### GGUF or ParoQuant for Qwen?

For Qwen3.6 35B-A3B, the optimized ParoQuant W4 checkpoint is slightly faster
and uses less memory in our AMD tests. It is a good choice when that exact model
meets your needs.

GGUF has a much larger model and quantization ecosystem. Current development is
therefore focused on GGUF compatibility, while the optimized ParoQuant path
remains supported.

## Installation

### Requirements

| Platform | Requirements |
| --- | --- |
| AMD | Linux x86-64, Python 3.10+ and ROCm with `hipcc` and `libamdhip64.so` |
| NVIDIA Blackwell | Linux x86-64, Python 3.10+ and the CUDA toolkit with `nvcc`; Maple only |
| Published wheel | glibc 2.39 or newer, such as Ubuntu 24.04 |

ROCm 7.x is the safest choice for the current wheel. The first model load
compiles and caches kernels, so it takes longer than later starts.

Install from PyPI:

```bash
pip install hipengine huggingface_hub
```

Or install a source checkout:

```bash
git clone https://github.com/shisa-ai/hipEngine.git
cd hipEngine
git lfs install
git lfs pull
pip install -e .
```

Confirm that the command is available:

```bash
hipengine --help
hipengine serve --help
```

## Start a local server

hipEngine does not download model weights during startup. Download a supported
model first, or use a GGUF file that is already on disk.

For the ParoQuant Qwen checkpoint:

```bash
hf download shisa-ai/Qwen3.6-35B-A3B-PARO-packed

hipengine serve \
  --model shisa-ai/Qwen3.6-35B-A3B-PARO-packed \
  --served-model-name qwen-paro
```

For GGUF, pass the path to the model file:

```bash
hipengine serve \
  --model /path/to/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --served-model-name qwen
```

The server listens on `http://127.0.0.1:8000` by default. Test it with:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Why is the sky blue?"}],
    "max_tokens": 128
  }'
```

Point any client that accepts a custom OpenAI base URL at
`http://127.0.0.1:8000/v1`. See the [server guide](docs/API.md) for API keys,
streaming, tools, structured output, and model capability checks.

## Use the Python API

```python
from hipengine import LLM, SamplingParams

llm = LLM("shisa-ai/Qwen3.6-35B-A3B-PARO-packed")
outputs = llm.generate(
    ["Hello, hipEngine."],
    SamplingParams(max_tokens=64, temperature=0.0),
)
print(outputs[0])
llm.close()
```

`LLM(...)` detects a supported AMD GPU and chooses the model format
automatically. You can also pass a local GGUF or Maple path. Advanced users can
override the choice with `backend=` and `quant=`.

## Performance highlights

These are measured results, not estimates. Prompt processing is the speed of
reading the input. Text generation is the speed of producing new tokens.

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
| Model and format | GPU | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B ParoQuant W4 | Radeon Pro W7900 | 512 input tokens, 128 output tokens | **2917.732** | **115.599** |
| Qwen3.6-35B-A3B GGUF `Q4_K_M` | Radeon Pro W7900 | 512 input tokens, 128 output tokens | **2716.648** | **92.833** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` | Radeon 8060S | 512 input tokens, 128 output tokens | **1369.489** | **54.330** |
| Laguna S 2.1 GGUF `Q4_K_M` | Radeon 8060S | 512 input tokens, 128 output tokens | **654.249** | **23.221** |
| Laguna S 2.1 GGUF `UD-Q2_K_XL` | Radeon Pro W7900 | 4,096 input tokens; prompt processing only | **440.893** | — |
| Maple-Preview 2-bit | Radeon 8060S | 512-token prompt test; varied prompts for generation | **754.458** | **153.201** |
| Maple-Preview 2-bit | RTX PRO 6000 Blackwell | 512-token prompt test; varied prompts for generation | **1917.492** | **402.361** |

These rows use different hardware and tests. Do not compare one row directly
with another.

### Multiple requests

Each value is the total tokens per second across all active requests:

| Model and interface | GPU | 1 request | 2 requests | 4 requests | 8 requests | 9 requests | 13 requests |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 GGUF, low-level engine test | Radeon Pro W7900 | **98.263** | **148.944** | **209.304** | **266.479** | — | — |
| Qwen3.6 GGUF, OpenAI streaming server | Radeon Pro W7900 | **72.169** | — | — | **158.542** | **137.001** | **129.507** |
| Maple, public generation API | Radeon 8060S | **123.131** | **165.697** | **202.038** | **214.788** | — | — |

### Optional speculative modes

| Model and mode | GPU | Total text generation | Speed compared with normal generation |
| --- | --- | ---: | ---: |
| Qwen3.6 GGUF, optional compatibility mode | Radeon Pro W7900 | **122.67 tok/s** | **1.2679x** |
| Qwen3.6 GGUF, optional native mode | Radeon 8060S | **80.10 tok/s** | **1.4282x** |

These speculative modes are opt-in because their output can differ from normal
generation.
<!-- END TOPLINE:README_HIGHLIGHTS -->

Full commands, software versions, model hashes, memory use, and correctness
checks are in the [benchmark report](benchmarks/README.md).

## Status and limits

v0.4.0 is a large alpha release. It adds or expands:

- Qwen3.5 and Qwen3.6 GGUF model support on both AMD backends.
- Native parallel request handling for supported Qwen and Maple paths.
- Laguna S 2.1 generation and serving on Radeon 8060S systems.
- Maple-Preview 2-bit generation on AMD, plus an experimental native CUDA path
  for NVIDIA Blackwell.
- Faster prompt processing and generation across the supported AMD paths.
- OpenAI-compatible streaming, sampling, tools, structured-output validation,
  request cancellation, and an endpoint that reports available features.

Important limits:

- hipEngine uses one GPU. Multi-GPU inference is not implemented.
- There is no desktop GUI, model catalog, or automatic model download.
- CPU model inference is not implemented.
- NVIDIA support is limited to single-request Maple generation through the
  Python API. CUDA server and multi-request support are not ready.
- Maple currently uses greedy generation only.
- Advertised model context lengths are not a promise that hipEngine supports the
  same length. Use the model guide and set a conservative server context limit.
- Speculative generation is optional and off by default when it changes output
  or does not provide a reliable speed benefit.
- APIs and supported combinations can still change before 1.0.

## Hardware detection

`backend="auto"` recognizes `gfx1100` and `gfx1151`. These cover the tested
Radeon RX 7900 XTX / Pro W7900 and Ryzen AI MAX+ 395 / Radeon 8060S systems.
Other AMD architecture numbers are not automatically treated as compatible.

You can force a nearby backend, but do so only after checking output quality and
performance. hipEngine will not silently use PyTorch when a GPU is unsupported.

## Documentation

### User guides

| Guide | Contents |
| --- | --- |
| [Server API](docs/API.md) | OpenAI-compatible endpoints, clients, authentication, and limits |
| [GGUF models](docs/GGUF.md) | Supported Qwen formats and model-specific behavior |
| [Laguna S 2.1](docs/LAGUNA.md) | Hardware, memory, context, and serving limits |
| [Maple-Preview](docs/MAPLE.md) | AMD and NVIDIA support, memory use, and current limits |
| [Environment settings](docs/ENVS.md) | Runtime settings and overrides |
| [Changelog](CHANGELOG.md) | User-facing changes by release |

### Development and benchmark details

| Guide | Contents |
| --- | --- |
| [Architecture and roadmap](docs/PLAN.md) | Engine design and planned work |
| [Kernel catalog](docs/KERNELS.md) | Kernel implementations and source history |
| [Testing](docs/TESTING.md) | Correctness tests and release checks |
| [Benchmark methods](docs/BENCHMARK.md) | Rules used for performance claims |
| [Benchmark results](benchmarks/README.md) | Full result tables and evidence |
| [Contributor guide](AGENTS.md) | Repository workflow |

## Project lineage

hipEngine is an independent project that builds on ideas and software from
[ROCm](https://github.com/ROCm/rocm),
[HIP](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip),
[Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm),
[ParoQuant](https://github.com/z-lab/paroquant),
[FastDMS](https://github.com/shisa-ai/FastDMS),
[llama.cpp](https://github.com/ggml-org/llama.cpp), and other open-source
projects. See the source and model guides for detailed attribution.

## License

hipEngine source code is licensed under **AGPL-3.0-or-later**. Model weights,
checkpoints, and external datasets remain under their own licenses.
