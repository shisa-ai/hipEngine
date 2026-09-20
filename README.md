# hipEngine

hipEngine is a ROCm-native local inference engine built primarily for AMD
Radeon GPUs. It pairs a small Python host with custom HIP kernels for torch-free
model loading, generation, and OpenAI-compatible serving on supported hardware.

**Current release: v0.6.0.** This alpha adds OCR and speech runtimes, dynamic
GGUF quantization support, and serving improvements.
See the [release notes](CHANGELOG.md#v060---2026-09-20) for scope and limitations.

[Supported models](#supported-models) lists the current model families and
hardware; the [model support reference](docs/MODELS.md) gives exact formats,
checkpoints, and limitations for each.

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

- **Language models:** Qwen3.5/3.6/3.8 dense and mixture-of-experts models,
  Qwen3.8 Flash-Next, [Laguna S 2.1](docs/campaigns/LAGUNA.md), and
  [Maple-Preview](docs/campaigns/MAPLE.md).
- **Document understanding:** [Surya OCR 2](docs/model-cards/MODEL-SURYA.md) for full-page
  OCR and [EVIE 4.5B / 8B](docs/model-cards/MODEL-EVIE.md) for visual document retrieval.
- **Speech:** [VibeVoice ASR](docs/model-cards/MODEL-VIBEVOICE-ASR.md) for transcription
  (early support) and [VibeVoice TTS](docs/model-cards/MODEL-VIBEVOICE-TTS.md) for synthesis
  (experimental). [Moonshine ASR](docs/model-cards/MOONSHINE.md) has an internal runtime;
  public audio API support is planned.
- **Time-series forecasting:** [TimesFM 2.5](docs/model-cards/MODEL-TIMESFM.md) and
  [TimesFM 3.0](docs/model-cards/MODEL-TIMESFM3.md).

See the [model support reference](docs/MODELS.md) for exact checkpoints,
GGUF quantizations, ParoQuant, MLX, and safetensors formats, plus hardware
and API limits. Qwen3.8-27B GGUF `Q4_K_M` is the dense model to start with
on either AMD backend. Most other modalities are tested on Strix Halo;
NVIDIA Blackwell (`sm_120a`) support is limited to Maple's Python API.

An independent [survey of Qwen3.8-27B implementations on Strix Halo](docs/campaigns/QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md)
compares hipEngine against other engines on a single [Framework Desktop](https://frame.work/desktop) host.

hipEngine includes [DMS](https://arxiv.org/abs/2506.05345) support and training
code, with a published [DMS checkpoint for Qwen3.8-27B Q4_K_M](https://huggingface.co/shisa-ai/Qwen3.8-27B-Q4_K_M-DMS-W8192).
The [DMS analysis](docs/reference/DMS-ANALYSIS.md) records the quality bar and the
8K–232K evidence ladder; measured capacity is in
[Long context on a 24 GB GPU](#long-context-on-a-24-gb-gpu) below.

CPU model generation is not supported. The CPU backend is used for correctness
tests. On NVIDIA, load Maple with `backend="cuda_sm120a"`; automatic hardware
selection currently covers AMD only.

Choose the [ParoQuant W4 checkpoint](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-packed)
for the optimized Qwen3.6 35B-A3B path, or [GGUF](docs/reference/GGUF.md) for the
broader model and quantization ecosystem. See the
[quantization comparison](benchmarks/quant/README.md) for quality and speed
trade-offs, including ROCmFP4/ROCmFPX.

## Performance highlights

These are measured results, not estimates. Prompt processing is the speed of
reading the input. Text generation is the speed of producing new tokens.

The benchmark summary below is synchronized from the [benchmark report](benchmarks/README.md).

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
Tokens/s on each named host: **prompt processing** is input, **text generation**
is output. **MTP** is speculative decoding in qualified scopes. Dashes are
unmeasured; context limits are capacity tests.

### Performance

#### Radeon Pro W7900 — 48 GB (`gfx1100`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | ParoQuant W4 | **2879.4** | **113.3** | 115.8 | — |
| Qwen3.6-35B-A3B | GGUF `UD-Q4_K_M` | **2924.5** | **95.0** | 122.7 | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` (BF16 KV) | **898.6** | **30.8** | 39.7 | — |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` (INT8 KV) | 868.6 | 27.9 | — | **176,128** |
| Laguna S 2.1 | GGUF `UD-Q2_K_XL` | **440.9** | — | — | — |

BF16 rows: September 20, `epyc`, 512/128 tokens, warmup + three runs.
INT8: September 13. MTP/capacity not refreshed. Laguna: 4K prompts.
35B GGUF MTP is opt-in; 27B MTP is **1.63x** its matched 24.36 tok/s AR,
not either decode column.

#### Strix Halo / Radeon 8060S — 120 GB (`gfx1151`)

| Model | Quant | Prompt processing | Text generation | With MTP | AR measured |
| --- | --- | ---: | ---: | ---: | --- |
| Maple-Preview | 2-bit | **754.5** | **153.2** | — | 2026-08-08 |
| Qwen3.6-35B-A3B | GGUF `UD-Q4_K_M` | **1418.1** | **56.7** | 80.1 | 2026-09-20 |
| Laguna S 2.1 | GGUF `Q4_K_M` | **654.2** | **23.2** | — | 2026-07/08 |
| Qwen3.8-27B Dense | GGUF `Q4_K_S` | **396.1** | **13.1** | **23.9** | 2026-08-17 |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` | **404.5** | **12.15** | **22.4** | 2026-09-20 |

September 20 rows: Framework Desktop. Qwen3.8 `Q4_K_M` MTP uses three drafts
and 25 output tokens per request: **1.98x** its matched 11.30 tok/s AR,
not the 512/128 decode column. It declines above 1,023 tokens.
35B MTP is the July 19 opt-in result.
[Measurements](https://github.com/shisa-ai/hipEngine/blob/main/benchmarks/results/2026-09-20-gfx1151-v060-headline-refresh.json).

**Time-series forecasting (TimesFM 2.5 200M).** hipEngine decodes batch=8,
context 8192, horizon 512 forecasts in **0.082 s** on the HP ZBook Strix Halo
host (8.6x the official torch reference) and **0.062 s** on a Framework Desktop
host — the same GPU on two machines, so the gap is thermal headroom.

**VibeVoice-ASR 9B** is torch-free on Strix Halo. Q4_K_M beats bf16 — **1.52x**
prefill, **1.55x** decode — at 6.1 vs 16.7 GB, WER 2.01% vs 2.81%; **RTF 0.35**.
[Results](https://github.com/shisa-ai/hipEngine/blob/main/benchmarks/results/2026-09-14-gfx1151-vibevoice-q4-e2e-rtf.json).

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

Direct-INT8 failed 9 of 11 quality prompts and is not a default.
[Capacity evidence](https://github.com/shisa-ai/hipEngine/blob/main/benchmarks/results/2026-09-09-rx7900xtx-gguf-int8-direct-prefill-capacity.json)

### Prefix caching

Multi-turn conversations resend the whole transcript, so hipEngine reuses the
KV pages and hybrid state of any 256-token-aligned prefix a later request
repeats. Radix is the default for both the direct engine and HTTP serving;
`--prefix-cache off` rolls it back. Different prefill routes can change generated tokens.
Qwen3.6-35B-A3B `UD-Q4_K_M` on Strix Halo (`gfx1151`), 14 multi-turn lanes of
three turns, reusing 42,496 of 87,582 prompt tokens:

| Lane | Cache off | Cache on | Wall time |
| --- | ---: | ---: | ---: |
| Coding, transcript resent | 10.09 tok/s | **16.49** tok/s | **-38.8%** |
| Coding, transcript rebuilt | 10.13 tok/s | **12.67** tok/s | **-20.0%** |
| Chat (ShareGPT) | 29.50 tok/s | 29.15 tok/s | +1.2% |
| All lanes | 16.93 tok/s | **20.81** tok/s | **-18.6%** |

### Serving several requests at once

Aggregate tokens per second across all active requests, Qwen3.8-27B `Q4_K_M`
on the W7900, September 4, 2026. Peers use F16 KV where hipEngine uses BF16.

| Requests | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | **23.6** | **39.1** | **53.1** | **63.9** | **72.8** | **79.5** | **83.2** | **85.9** |
| llama.cpp HIP | 21.0 | 34.4 | 30.6 | 27.7 | 36.7 | 46.4 | 52.1 | 58.4 |
| hipEngine advantage | +12% | +14% | +74% | +130% | +99% | +71% | +60% | **+47%** |

On Strix Halo, Maple-Preview 2-bit scales to **214.788** tok/s across eight
requests (123.131 at one, 202.038 at four). Where speculative
decoding runs automatically in production it is scoped to a qualified shape:
Qwen3.6-35B-A3B GGUF reaches **93.644 tok/s public** — 1.1565x its own AR — at
two concurrent requests on the W7900.
<!-- END TOPLINE:README_HIGHLIGHTS -->

Full commands, software versions, model hashes, memory use, and correctness
checks are in the [benchmark report](benchmarks/README.md).

### Coding-Agent Session Replay

Replaying a recorded coding session shows the benefit of avoiding repeated
prompt processing: 32 sequential requests, 128 output tokens per request,
and a cumulative transcript growing from 740 to 13,762 tokens.
Qwen3.6-35B-A3B `UD_Q4_K_M` on the HP ZBook / Radeon 8060S (`gfx1151`),
using the in-process engine and server chat renderer:

| Prefix cache | End-to-end output | Average turn | Cache hits |
| --- | ---: | ---: | ---: |
| Off | 7.05 tok/s | 18.2 s | 0/32 |
| Radix, default retention | 12.61 tok/s | 10.2 s | 19/32 |
| Radix, 16 retained snapshots | **19.69 tok/s** | **6.5 s** | **30/32** |

**2.79x end-to-end throughput**, with 91% of prompt tokens reused rather
than recomputed. This is one adapted transcript and one run per setting,
not a general agent-quality benchmark; outputs were not compared across arms.
System/tool definitions are substituted and long tool results truncated.
The 16-snapshot setting uses more memory and is opt-in.
[Results and retention guidance](benchmarks/README.md#current-default-notes).

## Status and limits

v0.6.0 is alpha. Automatic performance routes remain scoped to qualified
model, hardware, and workload combinations:

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
  gfx1151, Qwen3.8 `Q4_K_M` speculates at one active request with the default
  candidate budget and bf16 KV, and declines to autoregressive decoding above
  the speculative head's 1,023-token context window. Requesting speculation
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
See the [TheRock setup guide](docs/reference/THEROCK.md) for retained ROCm 7.13 and gfx1151 ROCm 10 setup/JIT validation.
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
| [Model support](docs/MODELS.md) | Exact model families, formats, checkpoints, and hardware |
| [GGUF models](docs/reference/GGUF.md) | Supported Qwen formats and model-specific behavior |
| [Laguna S 2.1](docs/campaigns/LAGUNA.md) | Hardware, memory, context, and serving limits |
| [Maple-Preview](docs/campaigns/MAPLE.md) | AMD and NVIDIA support, memory use, and current limits |
| [Environment settings](docs/ENVS.md) | Runtime settings and overrides |
| [Changelog](CHANGELOG.md) | User-facing changes by release |

### Development and benchmark details

| Guide | Contents |
| --- | --- |
| [Architecture and roadmap](docs/PLAN.md) | Engine design and planned work |
| [Kernel catalog](docs/KERNELS.md) | Kernel implementations and source history |
| [DMS analysis](docs/reference/DMS-ANALYSIS.md) | DMS quality bar, paper-matched tests, and the 8K–232K evidence ladder |
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
