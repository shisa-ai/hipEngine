# hipEngine

hipEngine is a ROCm-native local inference engine built primarily for AMD
Radeon GPUs. It pairs a small Python host with custom HIP kernels for torch-free
model loading, generation, and OpenAI-compatible serving on supported hardware.

**Current release: v0.5.0.** In addition to the Qwen 3.6 PARO and GGUF MoE models,
hipEngine now includes extensive tuning for [Qwen 3.8 27B Q4_K_M](#performance), 
including long-context modes that hold up to 232K tokens of context on a 24 GB GPU.

There is also initial support for Qwen 3.8 Flash Next,
[Laguna S 2.1](https://poolside.ai/blog/introducing-laguna-s-2-1), and [Maple ternary](https://github.com/deepgrove-ai/mlx-lm-deepgrove). 

Additional modalities now include:
- [Moonshine ASR](https://github.com/moonshine-ai/moonshine) (real-time ASR)
- [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) and [3.0](https://huggingface.co/google/timesfm-3.0-pytorch) (time series)
- [EVIE 4.5B](https://huggingface.co/tencent/EVIE-4.5B) and [8B](https://huggingface.co/tencent/EVIE-8B) (visual document retrieval)

## Why use hipEngine?

- **Native AMD support.** HIP-first kernels are written from scratch and tuned specifically for
  RDNA 3 (gfx1100) and Strix Halo RDNA 3.5 (gfx1151) instead of being CUDA ports.
- **No PyTorch runtime required.** There is no PyTorch dependency, which keeps
  hipEngine lightweight. Although it is packaged for Python, almost all of the
  hot path is C++.
- **Optimized for agents and concurrent requests.** Besides extensive tuning for
  fast single-request performance, hipEngine is specifically tuned for multiple concurrent request performance.
  The engine is built with continuous batching, prefix caching, and a shared KV pool.
- **Drop-in support for existing clients.** The included OpenAI-compatible server
  supports completion, chat, token-level SSE, logprobs, tools, structured-output
  validation, Qwen thinking controls, logprob-biased effort control, and
  request diagnostics. There is also a simple built-in `chat` interface.
- **Rigorous correctness.** All implementations are checked against a CPU-side
  oracle. The *strict* profile requires exact or parent-parity
  results, while *production* defaults must pass correctness gates. Optimizations or routes that
  fail these gates are rejected or made explicitly opt-in with measured costs stated.

hipEngine is a from-scratch project and does not inherit any unvetted code or legacy design. It is AGPL 3.0 licensed.

## Supported models

| Model family | Tested models and formats | AMD Radeon (`gfx1100`) | Radeon 8060S (`gfx1151`) | NVIDIA Blackwell (`sm_120a`) |
| --- | --- | :---: | :---: | :---: |
| Qwen3.x Dense | **0.8B:** [GGUF](docs/GGUF.md) `Q4_K_M`, `Q8_0`, `Q4_1`, `UD-Q4_K_XL`<br>**27B:** [GGUF](docs/GGUF.md) [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/65ca473/Qwen3.8-27B-Q4_K_M.gguf); Qwen3.8-27B `Q4_K_S` on `gfx1151` | Yes | Yes | — |
| Qwen3.x MoE | **35B-A3B:** [GGUF](docs/GGUF.md) `Q4_K_M`, `Q4_K_S`, `UD-Q3_K_M`, `UD-Q4_K_M`<br>[ParoQuant W4](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-packed) | Yes | Yes | — |
| Qwen3.8 Flash-Next | **125B-A6B + sparse PLE:** GGUF `UD-Q4_K_XL`; optional Q8 MTP and BF16 mmproj | — | Yes — text/QSA, opt-in MTP, ≤1K image/video, c2 serving | — |
| Laguna S 2.1 | [GGUF `Q4_K_M`](https://huggingface.co/poolside/Laguna-S-2.1-GGUF) | — | Yes | — |
| EVIE 4.5B / 8B | [Safetensors](docs/MODEL-EVIE.md) `fp32`, `fp16`; multimodal retrieval encoding | — | Yes | — |
| TimesFM 2.5 / 3.0 | [Safetensors](docs/MODEL-TIMESFM.md) `fp32`; [time-series forecasting](docs/MODEL-TIMESFM3.md) | — | Yes | — |
| Maple-Preview 20B-A1B | [2-bit MLX](https://huggingface.co/deepgrove/maple-preview-2bit-mlx) | Yes | Yes | Python API only |

**Qwen3.8-27B** is the dense model to start with: GGUF `Q4_K_M` on both AMD
backends, and should provide top-tier performance for both single and multiple requests.
An independent [survey of Qwen3.8-27B implementations on Strix Halo](docs/QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md)
compares hipEngine against other engines on a single [Framework Desktop](https://frame.work/desktop) host.

hipEngine includes [DMS](https://arxiv.org/abs/2506.05345) support and training code,
along with a published [DMS checkpoint for Qwen3.8-27B Q4_K_M](https://huggingface.co/shisa-ai/Qwen3.8-27B-Q4_K_M-DMS-W8192).
The 24 GB context and quality results below show DMS INT8 reaching about 5.7x the BF16 context capacity,
with quality tests reportng 100% top-1 agreement and 0.001 mean row-KL:

| KV configuration | Max context | Top-1 agreement vs BF16 | Mean row-KL |
| --- | ---: | ---: | ---: |
| BF16 KV | 40,960 | — | — |
| DMS BF16 | 73,728 | — | — |
| Direct-INT8 KV | 131,072 | 91.4% | 0.188 |
| DMS INT8 | 232,448 | 100% | 0.001 |

CPU model generation is not supported. The CPU backend is used for correctness
tests. On NVIDIA, load Maple with `backend="cuda_sm120a"`; automatic hardware
selection currently covers AMD only.

Support is specific to the listed model families and formats. hipEngine does
not yet run every GGUF model. See the [GGUF](docs/GGUF.md),
[Laguna](docs/LAGUNA.md), and [Maple](docs/MAPLE.md) guides for model-specific
limits.

### GGUF or ParoQuant for Qwen?

For Qwen3.6 35B-A3B on RDNA3, the optimized ParoQuant W4 checkpoint currently
slightly leads short-context generation and uses less memory, but GGUF is now extensively optimized.

GGUF has a much larger model and quantization ecosystem. Current development is
therefore focused on GGUF compatibility. Choose PARO for this exact optimized
checkpoint or GGUF for broader compatibility.

We've [tested alternative quantization formats](benchmarks/quant/README.md), including ROCmFP4/ROCmFPX variants. They trade quality against traditional GGUF Q formats and our testing/analysis did not reveal any intrinsic performance benefits.

## Performance highlights

These are measured results, not estimates. Prompt processing is the speed of
reading the input. Text generation is the speed of producing new tokens.

The benchmark summary below is synchronized from the [benchmark report](benchmarks/README.md).

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
Measured tokens/s on each named host. **Prompt processing** measures input;
**text generation** measures output. **MTP** is speculative decoding within
qualified scopes. Dashes indicate unmeasured results; context limits come
from separate capacity tests.

### Performance

#### Radeon Pro W7900 — 48 GB (`gfx1100`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | ParoQuant W4 | **2852.1** | **115.8** | **115.8** | — |
| Qwen3.6-35B-A3B | GGUF `Q4_K_M` | **2763.6** | **94.6** | 122.7 | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` | **868.6** | **27.9** | 39.7 | **176,128** |
| Laguna S 2.1 | GGUF `UD-Q2_K_XL` | **440.9** | — | — | — |

Laguna: 4K prompts. 35B-A3B GGUF MTP: explicitly enabled.
Qwen3.8 MTP: legacy BF16/K3, one request, 24 outputs;
**1.63x versus its matched 24.36 tok/s AR**, not the INT8 column.

#### Strix Halo / Radeon 8060S — 120 GB (`gfx1151`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Maple-Preview | 2-bit | **754.5** | **153.2** | — | — |
| Qwen3.6-35B-A3B | GGUF `UD-Q4_K_M` | **1369.5** | **54.3** | 80.1 | — |
| Laguna S 2.1 | GGUF `Q4_K_M` | **654.2** | **23.2** | — | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_S` | **396.1** | **13.1** | **23.9** | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` | **404.5** | **12.2** | 21.0 | — |

Qwen3.8 `Q4_K_M` defaults to production AR. MTP uses strict/K3 in one
qualified single-request scope: **1.88x its matched 11.15 tok/s AR baseline**,
not the 512/128 generation column.
[Measurements](https://github.com/shisa-ai/hipEngine/blob/main/benchmarks/results/2026-09-12-gfx1151-qwen38-final-headline-refresh.json).

**Time-series forecasting (TimesFM 2.5 200M).** hipEngine decodes batch=8,
context 8192, horizon 512 forecasts in **0.082 s** on the HP ZBook Strix Halo
host (8.6x the official torch reference there) and **0.062 s** on a Framework
Desktop host — the same `gfx1151` GPU on two physical machines, so the gap is
host power/thermal headroom, not a code change.

**VibeVoice-ASR 9B** runs torch-free on Strix Halo. Correctness and matched-input
performance qualification are in progress; no speedup is claimed.
[Results](https://github.com/shisa-ai/hipEngine/blob/main/benchmarks/results/2026-09-14-gfx1151-vibevoice-asr-full-content-audit.json).

#### NVIDIA RTX PRO 6000 Blackwell — 96 GB (`sm_120a`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Maple-Preview | 2-bit | **1917.5** | **402.4** | — | — |

### Long context on a 24 GB GPU

Qwen3.8-27B `Q4_K_M` can hold up to 232K tokens on a 24 GB `gfx1100` GPU
using [DMS](https://arxiv.org/abs/2506.05345), a trained KV eviction policy.
In these tests, DMS INT8 agreed more closely with BF16 than the direct-INT8 route:

| KV configuration | Max context | Top-1 agreement vs BF16 | Mean row-KL |
| --- | ---: | ---: | ---: |
| BF16 KV | 40,960 | — | — |
| DMS BF16 | 73,728 | — | — |
| Direct-INT8 KV | 131,072 | 91.4% | 0.188 |
| DMS INT8 | 232,448 | 100% | 0.001 |

The full 262,144-token context needs a predicted 24.8 GiB and does not fit.
The direct-INT8 route shown here failed 9 of 11 quality prompts and is not a
default. [Capacity evidence](https://github.com/shisa-ai/hipEngine/blob/main/benchmarks/results/2026-09-09-rx7900xtx-gguf-int8-direct-prefill-capacity.json)

### Serving several requests at once

Aggregate tokens per second across all active requests, Qwen3.8-27B `Q4_K_M`
on the W7900, measured September 4, 2026 under one server protocol.
The peers use F16 KV where hipEngine uses BF16.

| Requests | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | **23.6** | **39.1** | **53.1** | **63.9** | **72.8** | **79.5** | **83.2** | **85.9** |
| llama.cpp HIP | 21.0 | 34.4 | 30.6 | 27.7 | 36.7 | 46.4 | 52.1 | 58.4 |
| hipEngine advantage | +12% | +14% | +74% | +130% | +99% | +71% | +60% | **+47%** |

Direct engine measurements, September 6, 2026, same card and model,
512-token prompts and 128 outputs per request. They precede the shared-pool
changes; memory figures are not current serving estimates:

| Requests | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text generation (total) | 29.6 | 54.0 | 75.2 | 92.3 | 105.9 | 117.6 | 123.9 | **131.3** |
| Prompt processing (total) | **678.8** | 368.9 | 362.6 | 380.0 | 378.3 | 403.6 | 385.3 | 376.6 |
| Peak memory (GiB) | 19.4 | 20.3 | 21.1 | 22.0 | 22.8 | 23.7 | 24.5 | 25.4 |

Eight requests need about 25 GiB (32 GB or larger card); 24 GB shapes
are not qualified for concurrency yet.
On Strix Halo, Maple-Preview 2-bit scales to **214.788** tok/s across eight
requests (123.131 at one, 202.038 at four). Where speculative
decoding runs automatically in production it is scoped to a qualified shape:
Qwen3.6-35B-A3B GGUF reaches **93.644 tok/s public** — 1.1565x its own AR — at
two concurrent requests on the W7900.
<!-- END TOPLINE:README_HIGHLIGHTS -->

Full commands, software versions, model hashes, memory use, and correctness
checks are in the [benchmark report](benchmarks/README.md).

## Status and limits

v0.5.0 adds the dense Qwen models, and hipEngine now picks some performance
routes on its own, but only where they have been measured as safe:

- Qwen3.6-27B and Qwen3.8-27B GGUF generation and serving on both AMD backends.
  Both can speculate with the model's own multi-token prediction head, as can
  Qwen3.6-35B-A3B.
- The server turns speculative decoding on only for the model, GPU, and request
  shapes it has measured, reports why when it skips speculation, and can be
  switched off for all new requests by one endpoint call.
- An optional execution-profile selector (`strict`, `production`,
  `batch_invariant`) that runs a registered kernel plan, checks that its
  fallbacks are installed, and rejects a combination hipEngine has not been
  shown to complete.
- Scheduling and memory defaults: the `fair` prefill/decode policy, a smaller
  per-process GPU memory reserve on Radeon RDNA 3, and FP16 recurrent state for
  Qwen3.8 `Q4_K_S` on Strix Halo.
- Still supported: Qwen3.5/3.6 GGUF and ParoQuant, Laguna S 2.1, Maple-Preview,
  several requests at once on one resident model, and OpenAI-compatible
  streaming, sampling, tools, structured-output validation, and cancellation.

Full user-facing change history is in the [changelog](CHANGELOG.md).

Important limits:

- hipEngine uses one GPU. Multi-GPU inference is not yet implemented.
- There is no desktop GUI, model catalog, or automatic model download.
- CPU model inference is not implemented.
- The concurrency memory figures come from a 48 GB W7900. Single-request
  context on a 24 GB card is qualified to 232,448 tokens on the DMS and
  opt-in direct-INT8 routes; concurrent-request shapes on 24 GB are not
  qualified yet, so keep a conservative context limit there.
- Automatic speculative decoding covers only narrow measured shapes. On
  gfx1151, Qwen3.8 `Q4_K_M` requires the strict/K3 settings and request scope
  shown above; production-profile requests use AR. Requesting speculation
  does not guarantee engagement. See [Server API](docs/API.md).
- APIs and supported combinations can still change before 1.0.

## Hardware detection

`backend="auto"` recognizes `gfx1100` and `gfx1151`. These cover the tested
Radeon Pro W7900 and Ryzen AI MAX+ 395 / Radeon 8060S systems.
Other AMD architecture numbers are not automatically treated as compatible.

You can force a nearby backend, but do so only after checking output quality and
performance. hipEngine will not silently use PyTorch when a GPU is unsupported.

## Installation

### Requirements

| Platform | Requirements |
| --- | --- |
| AMD | Linux x86-64, Python 3.11+ and ROCm with `hipcc` and `libamdhip64.so` |
| NVIDIA Blackwell | Linux x86-64, Python 3.11+ and the CUDA toolkit with `nvcc`; Maple only |
| Published wheel | glibc 2.39 or newer, such as Ubuntu 24.04 |

ROCm 7.x is the safest choice for the current wheel (ROCm 10.0 has been tested and works fine as well).
See the [TheRock setup guide](docs/THEROCK.md) for retained ROCm 7.13 and gfx1151 ROCm 10 setup/JIT validation.
The first model load compiles and caches kernels, so it takes longer than later starts.

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

## Chat in your terminal

With the server running, open another terminal:

```bash
pip install 'hipengine[chat]'
hipengine chat
```

The client connects to `http://127.0.0.1:8000` and discovers the served model.
No model path is needed, and it does not start another server. For a different
address, use `hipengine chat --server http://127.0.0.1:8001`.

Replies stream as Markdown, with optional reasoning display and per-turn stats.
Use `/status` for server limits, `/usage` for conversation token counts,
`/think off` to disable reasoning, `/retry` to regenerate, and `/clear` to start
over. `/help` lists all commands; `/quit`, Ctrl-C, or Ctrl-D exits.
Use `hipengine chat --plain` for plain-text output.

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
override the choice with `backend=` and `quant=`. The
`execution_profile="strict"|"production"|"batch_invariant"` selector is
fail-closed to registered kernel plans with exact fallbacks; omitting it selects
`production` for models with a certified plan and keeps the previous behaviour
otherwise.

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
| [DMS analysis](docs/DMS-ANALYSIS.md) | DMS quality bar, paper-matched tests, and the 8K–232K evidence ladder |
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

## Thanks

Special thanks to [Framework](https://frame.work/) and [AMD](https://www.amd.com/) for providing Strix Halo test hardware.

## License

hipEngine source code is licensed under **AGPL-3.0-or-later**. Model weights,
checkpoints, and external datasets remain under their own licenses.
