# hipEngine

hipEngine is a ROCm-native local inference engine built primarily for AMD
Radeon GPUs. It pairs a small Python host with custom HIP kernels for torch-free
model loading, generation, and OpenAI-compatible serving on supported hardware.

**Current release: v0.4.0 alpha.** Besides the Qwen 3.6 PARO and GGUF models,
the latest version of hipEngine now supports GGUF inference for more model
families. These include [Laguna S 2.1](https://poolside.ai/blog/introducing-laguna-s-2-1),
[Maple ternary](https://github.com/deepgrove-ai/mlx-lm-deepgrove), and [Moonshine ASR](https://github.com/moonshine-ai/moonshine).

## Why use hipEngine?

- **Native AMD support.** HIP-first kernels directly target and tune for specific
  RDNA 3 (gfx1100) and Strix Halo RDNA 3.5 (gfx1151) instead of being CUDA ports.
- **No PyTorch runtime required.** There is no PyTorch dependency, which keeps
  hipEngine lightweight. Although it is packaged for Python, almost all of the
  hot path is C++.
- **Optimized for agents and concurrent requests.** Besides extensive tuning for
  fast single-request performance (especially for prefill), hipEngine also has
  tuned support for c=N. It is significantly faster than llama.cpp or vLLM for
  c=8 workloads.
- **Drop-in support for existing clients.** The included OpenAI-compatible server
  supports completion, chat, token-level SSE, logprobs, tools, structured-output
  validation, Qwen thinking controls, logprob-biased effort control, and
  request diagnostics.

hipEngine is a new, small software project focused on making a select list of
models perform well, particularly Qwen 3.x variants and fine-tunes.

## Supported models

`Yes` means that public text generation has been tested. A dash means that the
combination is not supported. The Qwen rows group closely related model
versions, with size-specific format coverage shown explicitly. Features such as
batching, sampling, tools, and long context can differ by model.

| Model family | Tested models and formats | RX 7900 XTX / W7900 (`gfx1100`) | Radeon 8060S (`gfx1151`) | NVIDIA Blackwell (`sm_120a`) |
| --- | --- | :---: | :---: | :---: |
| Qwen3.x Dense | **0.8B:** [GGUF](docs/GGUF.md) `Q4_K_M`, `Q8_0`, `Q4_1`, `UD-Q4_K_XL`<br>**27B:** [GGUF](docs/GGUF.md) `Q4_K_M`; Qwen3.8-27B `Q4_K_S` on `gfx1151` | Yes | Yes | — |
| Qwen3.x MoE | **35B-A3B:** [GGUF](docs/GGUF.md) `Q4_K_M`, `Q4_K_S`, `UD-Q3_K_M`, `UD-Q4_K_M`<br>[ParoQuant W4](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-packed) | Yes | Yes | — |
| Laguna S 2.1 | [GGUF `Q4_K_M`](https://huggingface.co/poolside/Laguna-S-2.1-GGUF) | — | Yes | — |
| Maple-Preview 20B-A1B | [2-bit MLX](https://huggingface.co/deepgrove/maple-preview-2bit-mlx) | Yes | Yes | Python API only |

CPU model generation is not supported. The CPU backend is used for correctness
tests. On NVIDIA, load Maple with `backend="cuda_sm120a"`; automatic hardware
selection currently covers AMD only.

Support is specific to the listed model families and formats. hipEngine does
not yet run every GGUF model. See the [GGUF](docs/GGUF.md),
[Laguna](docs/LAGUNA.md), and [Maple](docs/MAPLE.md) guides for model-specific
limits.

### GGUF or ParoQuant for Qwen?

For Qwen3.6 35B-A3B on W7900, the optimized ParoQuant W4 checkpoint currently
leads short-context generation and uses less memory. GGUF leads prompt
processing from 1K tokens onward in the current six-shape sweep.

GGUF has a much larger model and quantization ecosystem. Current development is
therefore focused on GGUF compatibility. Choose PARO for this exact optimized
checkpoint or GGUF for broader compatibility.

## Installation

### Requirements

| Platform | Requirements |
| --- | --- |
| AMD | Linux x86-64, Python 3.11+ and ROCm with `hipcc` and `libamdhip64.so` |
| NVIDIA Blackwell | Linux x86-64, Python 3.11+ and the CUDA toolkit with `nvcc`; Maple only |
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
override the choice with `backend=` and `quant=`. The opt-in
`execution_profile="strict"|"production"|"batch_invariant"` selector is
fail-closed to registered kernel plans with exact fallbacks; omitting it
preserves the migration default until profile calibration completes.

## Performance highlights

These are measured results, not estimates. Prompt processing is the speed of
reading the input. Text generation is the speed of producing new tokens.

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
### Radeon Pro W7900 (`gfx1100`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B ParoQuant W4 | 512 input tokens, 128 output tokens | **2852.100** | **115.804** |
| Qwen3.6-35B-A3B GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **2763.590** | **94.603** |
| Qwen3.6-27B Dense GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **875.364** | **28.681** |
| Laguna S 2.1 GGUF `UD-Q2_K_XL` | 4,096 input tokens; prompt processing only | **440.893** | — |

#### Multiple requests (total tok/s across all active requests)
| Model and interface | 1 request | 2 requests | 4 requests | 8 requests | 9 requests | 13 requests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` (engine) | **98.263** | **148.944** | **209.304** | **266.479** | — | — |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` (server) | **72.169** | — | — | **158.542** | **137.001** | **129.507** |

#### MTP
| Model and mode | Text generation | Speed compared with AR |
| --- | ---: | ---: |
| Qwen3.6-27B Dense GGUF `Q4_K_M` — Generation-2 C1/K3 D24 | **32.076 tok/s** | **1.4382x** |
| Qwen3.6-27B Dense GGUF `Q4_K_M` — Generation-2 production C2/K2 D24 automatic | **34.341 tok/s** | **1.1173x** |
| Qwen3.8-27B Dense GGUF `Q4_K_M` — Generation-2 production C2/K2 D24 automatic | **36.726 tok/s** | **1.1970x** |
| Qwen3.8-27B Dense GGUF `Q4_K_M` — Generation-2 production C2/K3 D24 explicit | **51.769 tok/s** | **1.3376x** |
| Qwen3.8-27B Dense GGUF `Q4_K_M` — Generation-2 production C8/K3 D24 automatic | **98.643 tok/s** | **1.1178x** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` — Generation-2 production C2/K2 D24 automatic | **93.644 tok/s public** / **98.505 tok/s three-run** | **1.1565x** / **1.1368x** |
### RX 7900 XTX (`gfx1100`) — Qwen3.8-27B `Q4_K_M` prefill

| Workload | hipEngine | llama.cpp HIP | HE vs HIP | llama.cpp Vulkan | HE vs Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | **959.4** | 965.0 | -0.6% | 865.7 | +10.8% |
| 1K | **999.7** | 979.5 | +2.1% | 832.6 | +20.1% |
| 4K | **981.8** | 945.7 | +3.8% | 836.5 | +17.4% |

#### Dedicated-server context
| KV route | Server shape | Measured context | Peak / headroom |
| --- | --- | ---: | ---: |
| BF16 default | c1 operational | **32K** | 21.869 / 2.115 GiB |
| Pure INT8 explicit | c1 repeated natural soak | **112K** | 23.323 / 0.661 GiB |
| Pure INT8 explicit | c1 one-request physical ceiling | **126K** | 23.963 / 0.022 GiB |

#### Decode / MTP

| Metric | hipEngine | llama.cpp HIP | HE vs HIP | llama.cpp Vulkan | HE vs Vulkan |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR decode 512 | **34.06** | 32.86 | +3.6% | 13.39 | 2.54x |
| AR decode 1K | **34.91** | 32.75 | +6.6% | 13.38 | 2.61x |
| AR decode 4K | **31.79** | 32.41 | -1.9% | 13.31 | 2.39x |
| MTP natural | **62.44 B3** | 44.33 B2 | +40.9% | 73.33 B2 | -14.8% |

### Strix Halo / Radeon 8060S (`gfx1151`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` | 512 input tokens, 128 output tokens | **1369.489** | **54.330** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` (GEMV lib hoist) | sync'd eager, per-token | — | **38.9** |
| Qwen3.8-27B Dense GGUF `Q4_K_S` | 512 input tokens, 128 output tokens | **396.091** | **13.069** |
| Laguna S 2.1 GGUF `Q4_K_M` | 512 input tokens, 128 output tokens | **654.249** | **23.221** |
| Maple-Preview 2-bit | 512-token prompt test; varied prompts for generation | **754.458** | **153.201** |
#### Multiple requests

| Model and interface | 1 request | 2 requests | 4 requests | 8 requests |
| --- | ---: | ---: | ---: | ---: |
| Maple-Preview 2-bit (engine) | **123.131** | **165.697** | **202.038** | **214.788** |

#### MTP

| Model and mode | Text generation | Speed compared with AR |
| --- | ---: | ---: |
| Qwen3.8-27B Dense GGUF `Q4_K_S` — MTP-3 | **23.853 tok/s** | **1.7845x** |
| Qwen3.8-27B Dense GGUF `Q4_K_M` — normal-owner C1 MTP-3 automatic | **15.609 tok/s** | **1.5916x** |
| Qwen3.8-27B Dense GGUF `Q4_K_M` — production C2 MTP-3 automatic | **17.031 tok/s** | **1.1441x** |
| Qwen3.8-27B Dense GGUF `Q4_K_M` — production C1 MTP-3 c68-128 explicit | **13.088 tok/s** | **1.3998x** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` — MTP-2 | **80.10 tok/s** | **1.4282x** |

### RTX PRO 6000 Blackwell (`sm_120a`)

| Model and format | Test | Prompt processing (tok/s) | Text generation (tok/s) |
| --- | --- | ---: | ---: |
| Maple-Preview 2-bit | 512-token prompt test; varied prompts for generation | **1917.492** | **402.361** |

Rows use different models and tests; compare only matching protocols. The RX 7900 XTX cross-engine rows use the same Qwen3.8 file and timing boundary. llama.cpp Vulkan MTP is speed-only because its ledger differs from Vulkan AR; hipEngine and llama.cpp HIP match their controls. MTP-2/MTP-3 use two/three draft tokens. The W7900 C8/K3 row is automatic only for its exact production Qwen3.8 `Q4_K_M` capacity-8 D24 key; misses use K0. The 35B-A3B MTP-2 path matches llama.cpp MTP on the validated suite and remains opt-in because it can differ from normal AR.
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
- Dense-Qwen server MTP defaults to fail-closed `auto`; explicit MTP uses native B3 and may differ from AR.
  `HIPENGINE_GGUF_MTP_VERIFY_MODE=serial_exact` restores token-exact control; see [Server API](docs/API.md).
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
