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
Every number below is measured on the named hardware and links to a
reproducible artifact. **Prompt processing** is how fast hipEngine reads your
input; **text generation** is how fast it writes new tokens. **With MTP** is
speculative decoding, which is enabled only where it is qualified for that
model and shape. Rows use different models and protocols — compare within a
row, not across them.

### At a glance — one request

#### Radeon Pro W7900 — 48 GB (`gfx1100`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | ParoQuant W4 | **2852.1** | **115.8** | **115.8** | — |
| Qwen3.6-35B-A3B | GGUF `Q4_K_M` | **2763.6** | **94.6** | 122.7 (opt-in) | — |
| Qwen3.6-27B Dense | GGUF `Q4_K_M` | **875.4** | **28.7** | **32.1** | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` | **678.8** | **29.6** | — | — |
| Laguna S 2.1 | GGUF `UD-Q2_K_XL` | **440.9** (4K) | — | — | — |

#### Strix Halo / Radeon 8060S — 120 GB (`gfx1151`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Maple-Preview | 2-bit | **754.5** | **153.2** | — | — |
| Qwen3.6-35B-A3B | GGUF `UD-Q4_K_M` | **1369.5** | **54.3** | 80.1 (opt-in) | — |
| Laguna S 2.1 | GGUF `Q4_K_M` | **654.2** | **23.2** | — | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_S` | **396.1** | **13.1** | **23.9** | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` | — | — | **15.6** | — |

#### NVIDIA RTX PRO 6000 Blackwell — 96 GB (`sm_120a`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Maple-Preview | 2-bit | **1917.5** | **402.4** | — | — |

Blank cells are shapes we have not measured yet, not failures. Max context is
published only where a dedicated ceiling run exists.

- **On a 24 GB card such as the RX 7900 XTX**, Qwen3.8-27B `Q4_K_M` with
  512-token prompts fits **four to five concurrent requests**. Peak HIP usage is
  19.6 GiB at one request and rises about 0.9 GiB per added request: c4 needs
  22.4 GiB, c5 needs 23.3 GiB, and c6 needs 24.2 GiB, which does not fit. The
  weights alone are 15.9 GiB, so concurrency and context compete for the same
  ~8 GiB. At one request the same card has measured 32K context on BF16 KV and
  112K on INT8 KV.

### Serving several requests at once

This is where hipEngine pulls furthest ahead. Aggregate tokens per second
across all active requests, Qwen3.8-27B `Q4_K_M` on the W7900 under one server
protocol; the peers use F16 KV where hipEngine uses BF16.

| Requests | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | **23.6** | **39.1** | **53.1** | **63.9** | **72.8** | **79.5** | **83.2** | **85.9** |
| llama.cpp HIP | 21.0 | 34.4 | 30.6 | 27.7 | 36.7 | 46.4 | 52.1 | 58.4 |
| hipEngine advantage | +12% | +14% | +74% | +130% | +99% | +71% | +60% | **+47%** |

Direct engine route on the same card and model, 512-token prompts and 128
generated tokens per request, showing what each added request costs in memory:

| Requests | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text generation (total) | 29.6 | 54.0 | 75.2 | 92.3 | 105.9 | 117.6 | 123.9 | **131.3** |
| Prompt processing (total) | **678.8** | 368.9 | 362.6 | 380.0 | 378.3 | 403.6 | 385.3 | 376.6 |
| Peak memory (GiB) | 19.4 | 20.3 | 21.1 | 22.0 | 22.8 | 23.7 | 24.5 | 25.4 |

Eight concurrent requests need about 25 GiB, so this shape wants a 32 GB or
larger card; a 24 GB card runs the same model comfortably at one or two.

On Strix Halo, Maple-Preview 2-bit scales to **214.788** tok/s across eight
requests (123.131 at one, 165.697 at two, 202.038 at four). Where speculative
decoding runs automatically in production it is scoped to a qualified shape:
Qwen3.6-35B-A3B GGUF reaches **93.644 tok/s public** — 1.1565x its own AR — at
two concurrent requests on the W7900.
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
