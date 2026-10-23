# hipEngine

hipEngine is a ROCm-native local inference engine built primarily for AMD
Radeon GPUs. It pairs a small Python host with custom HIP kernels for torch-free
model loading, generation, and OpenAI-compatible serving on supported hardware.

**Current release: v0.8.3.** Fixes Gemma 4 serving and GPU resource cleanup,
corrects Flash-Next Q4 decode selection, and reduces redundant Flash-Next work.
See the [release notes](CHANGELOG.md#v083---2026-10-23) for scope and limitations.

## Why use hipEngine?

- **Native AMD support.** HIP-first kernels are written from scratch and tuned specifically for
  RDNA 3 (gfx1100) and Strix Halo RDNA 3.5 (gfx1151) instead of being CUDA ports.
- **No PyTorch runtime required.** There is no PyTorch dependency, which keeps
  hipEngine lightweight.
- **Optimized for agents and concurrent requests.** hipEngine is tuned for both
  single-request latency and multiple concurrent requests, with continuous
  batching, prefix caching, and a shared KV pool.
- **Drop-in support for existing clients.** The included OpenAI-compatible server
  supports completion, chat, token-level SSE, logprobs, tools, structured-output
  validation, Qwen thinking controls, logprob-biased effort control, and
  request diagnostics. There is also a simple built-in `chat` interface.
- **Correct output first, then speed.** Some engines trade output accuracy for
  benchmark numbers. hipEngine does not: every kernel is checked against a CPU
  reference implementation, and a faster kernel becomes the default only after
  its output passes those checks.

## Supported models

- **Language models:** Qwen3.5/3.6/3.8 dense and mixture-of-experts models,
  Qwen3.8 Flash-Next (GGUF `UD-Q4_K_XL`: text/QSA, opt-in MTP, ≤1K image/video,
  32K served context, c4 serving), [Gemma 4 26B-A4B](docs/model-cards/MODEL-GEMMA4.md),
  [Laguna S 2.1](docs/campaigns/LAGUNA.md), and
  [Maple-Preview](docs/campaigns/MAPLE.md).
- **Document understanding:** [Surya OCR 2](docs/model-cards/MODEL-SURYA.md) for full-page
  OCR and [EVIE 4.5B / 8B](docs/model-cards/MODEL-EVIE.md) for visual document retrieval.
- **Speech:** [VibeVoice ASR](docs/model-cards/MODEL-VIBEVOICE-ASR.md) for transcription
  (early support) and [VibeVoice TTS](docs/model-cards/MODEL-VIBEVOICE-TTS.md) for synthesis
  (experimental). [Moonshine ASR](docs/model-cards/MOONSHINE.md) has an internal runtime;
  public audio API support is planned.
- **Music generation:** [YuE2](docs/model-cards/MODEL-YUE2.md) for songs from lyrics
  and a style prompt, through `LLM.generate_song()` or `POST /v1/audio/songs`
  (early support).
- **Time-series forecasting:** [TimesFM 2.5](docs/model-cards/MODEL-TIMESFM.md) and
  [TimesFM 3.0](docs/model-cards/MODEL-TIMESFM3.md).

See the [model support reference](docs/MODELS.md) for exact checkpoints,
GGUF quantizations, ParoQuant, MLX, and safetensors formats, plus hardware
and API limits. Qwen3.8-27B GGUF `Q4_K_M` is the dense model to start with
on either AMD backend. Most other modalities are tested on Strix Halo.
NVIDIA Blackwell (`sm_120a`) support is limited to Maple's Python API, which
you load explicitly with `backend="cuda_sm120a"`.

An independent [survey of Qwen3.8-27B implementations on Strix Halo](docs/campaigns/QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md)
compares hipEngine against other engines on a single [Framework Desktop](https://frame.work/desktop) host.

hipEngine includes [DMS](https://arxiv.org/abs/2506.05345) support and training
code, with a published [DMS checkpoint for Qwen3.8-27B Q4_K_M](https://huggingface.co/shisa-ai/Qwen3.8-27B-Q4_K_M-DMS-W8192).
The [DMS analysis](docs/reference/DMS-ANALYSIS.md) has the quality results;
measured capacity is in
[Long context on a 24 GB GPU](#long-context-on-a-24-gb-gpu) below.

CPU model generation is not supported. The CPU backend is used for correctness
tests.

Choose the [ParoQuant W4 checkpoint](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-packed)
for the optimized Qwen3.6 35B-A3B path, or [GGUF](docs/reference/GGUF.md) for the
broader model and quantization ecosystem. See the
[quantization comparison](benchmarks/quant/README.md) for quality and speed
trade-offs, including ROCmFP4/ROCmFPX.

## Performance highlights

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
Tokens/s on each named host: **prompt processing** is input, **text generation**
is output. **MTP** is speculative decoding. Dashes are unmeasured; **Max
context** is a capacity limit, not a full-length prefill.

### Radeon Pro W7900 — 48 GB (`gfx1100`)

| Model | Quant | Prompt processing | Text generation | With MTP |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | ParoQuant W4 | **2859.0** | **116.0** | — |
| Qwen3.6-35B-A3B | GGUF `UD-Q4_K_M` | **2924.5** | **95.0** | 122.7 |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` (BF16 KV) | **898.6** | **30.8** | 39.7 |
| Gemma 4 26B-A4B | GGUF `UD-Q4_K_XL` | **3805.6** | **75.8** | **120.2** |
| Laguna S 2.1 | GGUF `UD-Q2_K_XL` | **440.9** | — | — |

### Strix Halo / Radeon 8060S — 120 GB (`gfx1151`)

| Model | Quant | Prompt processing | Text generation | With MTP |
| --- | --- | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | ParoQuant W4 | 1200.9 | **66.4** | 64.3 |
| Qwen3.6-35B-A3B | GGUF `UD-Q4_K_M` | **1418.1** | **56.7** | 80.1 |
| Qwen3.8-27B Dense | GGUF `Q4_K_M` | **404.5** | **12.15** | **22.4** |
| Qwen3.8 Flash-Next | GGUF `UD-Q4_K_XL` | **1036.2** | **23.7** | **28.9** |
| Gemma 4 26B-A4B | GGUF `UD-Q4_K_XL` | **1865.6** | **43.3** | **69.4** |
| Laguna S 2.1 | GGUF `Q4_K_M` | **654.2** | **23.2** | — |
| Maple-Preview | 2-bit | **754.5** | **153.2** | — |

### NVIDIA RTX PRO 6000 Blackwell — 96 GB (`sm_120a`)

| Model | Quant | Prompt processing | Text generation | With MTP | Max context |
| --- | --- | ---: | ---: | ---: | ---: |
| Maple-Preview | 2-bit | **1917.5** | **402.4** | — | — |

### Long context on a 24 GB GPU

Using a source checkout: [Qwen3.8-27B `Q4_K_M`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/65ca473/Qwen3.8-27B-Q4_K_M.gguf)
on RX 7900 XTX, with `--max-active-requests 1`. Add the
[DMS checkpoint](https://huggingface.co/shisa-ai/Qwen3.8-27B-Q4_K_M-DMS-W8192)
for long context. Omit `--max-context-tokens` to let the server calculate the
context from available GPU memory.

| KV configuration | Auto context, MTP on | Auto context, MTP off |
| --- | ---: | ---: |
| BF16 KV | 61,440 | 83,456 |
| INT8 KV | 71,680 | 98,304 |
| DMS INT8 | 262,144 | 262,144 |

### Prefix caching

Multi-turn conversations resend the whole transcript, so hipEngine reuses the
KV pages and hybrid state of any 256-token-aligned prefix a later request
repeats. Radix is the default for both the direct engine and HTTP serving
(`--prefix-cache off` to disable). Qwen3.6-35B-A3B `UD-Q4_K_M` on Strix Halo:

| Lane | Cache off | Cache on | Wall time |
| --- | ---: | ---: | ---: |
| Coding, transcript resent | 10.09 tok/s | **16.49** tok/s | **-38.8%** |
| Coding, transcript rebuilt | 10.13 tok/s | **12.67** tok/s | **-20.0%** |
| Chat (ShareGPT) | 29.50 tok/s | 29.15 tok/s | +1.2% |
| All lanes | 16.93 tok/s | **20.81** tok/s | **-18.6%** |

### Serving several requests at once

Aggregate tokens/s across all active requests, Qwen3.8-27B `Q4_K_M` on the W7900:

| Requests | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hipEngine | **23.6** | **39.1** | **53.1** | **63.9** | **72.8** | **79.5** | **83.2** | **85.9** |
| llama.cpp HIP | 21.0 | 34.4 | 30.6 | 27.7 | 36.7 | 46.4 | 52.1 | 58.4 |
| hipEngine advantage | +12% | +14% | +74% | +130% | +99% | +71% | +60% | **+47%** |
<!-- END TOPLINE:README_HIGHLIGHTS -->

The summary above is synchronized from the [benchmark report](benchmarks/README.md),
which has the full commands, software versions, model hashes, memory use, and
correctness checks.

### Coding-Agent Session Replay

Replaying a recorded coding session shows the benefit of avoiding repeated
prompt processing. Qwen3.6-35B-A3B `UD_Q4_K_M` on Strix Halo (`gfx1151`):

| Prefix cache | End-to-end output | Average turn | Cache hits |
| --- | ---: | ---: | ---: |
| Off | 7.05 tok/s | 18.2 s | 0/32 |
| Radix, default retention | 12.61 tok/s | 10.2 s | 19/32 |
| Radix, 16 retained snapshots | **19.69 tok/s** | **6.5 s** | **30/32** |

**2.79x end-to-end throughput**, with 91% of prompt tokens reused rather than
recomputed. The 16-snapshot setting uses more memory and is opt-in.
[Results and retention guidance](benchmarks/README.md#current-default-notes).

## Limits

- hipEngine uses one GPU per process; multi-GPU inference is not implemented.
- Speculative decoding does not support packed INT8 or compact-DMS KV. See
  [Server API](docs/API.md) for admission and fallback behavior.
- Concurrent long-context requests on a 24 GB card are untested; keep the
  context limit conservative there.

## Hardware detection

`backend="auto"` recognizes `gfx1100` and `gfx1151`. These cover the tested
Radeon Pro W7900 and Ryzen AI MAX+ 395 / Radeon 8060S systems.
Other AMD architecture numbers are not automatically treated as compatible.

You can force a nearby backend, but do so only after checking output quality and
performance.

## Installation

### Requirements

| Platform | Requirements |
| --- | --- |
| AMD | Linux x86-64, Python 3.11+ and ROCm with `hipcc` and `libamdhip64.so` |
| NVIDIA Blackwell | Linux x86-64, Python 3.11+ and the CUDA toolkit with `nvcc`; Maple only |
| Published wheel | glibc 2.39 or newer, such as Ubuntu 24.04 |

ROCm 7.x or newer is recommended. See the [TheRock setup guide](docs/reference/THEROCK.md)
for ROCm 7.13 and gfx1151 ROCm 10 setup and JIT validation. The first model load
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

## Chat in your terminal

With the server running, open another terminal:

```bash
pip install 'hipengine[chat]'
hipengine chat
```

The client connects to `http://127.0.0.1:8000` and discovers the served model.
No model path is needed, and it does not start another server. For a different
address, use `hipengine chat --server http://127.0.0.1:8001`.

Replies stream as Markdown, with optional reasoning display and per-turn stats
(prompt prefill rate, decode rate, and tokens served from the prefix cache).
Use `/status` for server limits, `/usage` for conversation token counts,
`/bench` for a quick prefill, decode, and prefix-cache measurement of the
running server (with and without speculative MTP), `/think off` to disable
reasoning, `/retry` to regenerate, and `/clear` to start over. `/help` lists all
commands; `/quit`, Ctrl-C, or Ctrl-D exits.
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
override the choice with `backend=` and `quant=`. Pass
`execution_profile="strict"` for results that match the reference kernels
exactly, or `"batch_invariant"` for output that does not change with other
concurrent requests. The default is `"production"`, the fastest kernels that
pass hipEngine's correctness checks. If a model has no kernels for the requested
profile, loading fails instead of falling back.

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
| [DMS analysis](docs/reference/DMS-ANALYSIS.md) | DMS quality tests and long-context results |
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
