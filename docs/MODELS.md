---
status: current
owns: Models, quantizations, and backends hipEngine has implemented and measured.
---
# hipEngine model support

This page lists the model families hipEngine has implemented and measured, and
what each one was measured on. Per-model guides name the measured performance
and known limitations.

**It is a coverage record, not an admission list.** hipEngine loads any model
its kernels support and runs any feature they can execute. A model, revision,
or quantization missing from this page is not blocked — it loads if its
architecture, geometry, and quantization fit a registered kernel, and fails
with a named capability error if they do not. What absence from this page means
is that we have not measured it, so we publish no performance or quality
numbers for it.

CPU model generation is not supported. The CPU backend is used for correctness
tests. Automatic hardware selection (`backend="auto"`) covers AMD `gfx1100` and
`gfx1151` only.

## Language models

| Model family | Tested models and formats | `gfx1100` | `gfx1151` | `sm_120a` | Guide |
| --- | --- | :---: | :---: | :---: | --- |
| Qwen3.x Dense | 0.8B: GGUF `Q4_K_M`, `Q8_0`, `Q4_1`, `UD-Q4_K_XL`<br>27B (Qwen3.6 / Qwen3.8): GGUF [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/65ca473/Qwen3.8-27B-Q4_K_M.gguf); Qwen3.8-27B `Q4_K_S` on `gfx1151` | Yes | Yes | — | [GGUF](reference/GGUF.md) |
| Qwen3.8 Dense, dynamic GGUF | 27B: `UD-Q4_K_M`, `UD-Q4_K_S` | RX 7900 XTX tested; W7900 refresh pending | Not covered by the integration packet | — | [Dynamic-GGUF integration](campaigns/UD-MAIN-INTEGRATION.md) |
| Qwen3.x MoE | 35B-A3B: GGUF `Q4_K_M`, `Q4_K_S`, `UD-Q3_K_M`, `UD-Q4_K_M`<br>35B-A3B: [ParoQuant W4](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-packed) | Yes | Yes | — | [GGUF](reference/GGUF.md) |
| Qwen3.8 Flash-Next | 125B-A6B + sparse PLE: GGUF `UD-Q4_K_XL`; optional Q8 MTP and BF16 mmproj | — | Yes: text/QSA, opt-in MTP, ≤1K image/video, c2 serving | — | [Campaign](campaigns/QWEN3.8-FLASH-NEXT.md) |
| Laguna S 2.1 | [GGUF `Q4_K_M`](https://huggingface.co/poolside/Laguna-S-2.1-GGUF); BF16 DFlash drafter is an explicit opt-in | — | Yes | — | [Laguna](campaigns/LAGUNA.md) |
| Maple-Preview 20B-A1B | [2-bit MLX](https://huggingface.co/deepgrove/maple-preview-2bit-mlx) | Yes | Yes | Python API only | [Maple](campaigns/MAPLE.md) |

Dynamic-GGUF coverage is artifact-specific: the integration packet measured
single-request execution on the RX 7900 XTX and records known
speculative-output divergences for `UD-Q4_K_S`. Other `UD-` files and other
backends run on the same kernels; they are simply not covered by that packet's
measurements. Laguna also has a W7900
`UD-Q2_K_XL` prefill measurement, not a published decode result.

Qwen3.8-27B is the dense model to start with: GGUF `Q4_K_M` on both AMD
backends. An independent [survey of Qwen3.8-27B implementations on Strix
Halo](campaigns/QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md) compares hipEngine against other
engines on a single [Framework Desktop](https://frame.work/desktop) host.

## Speech, OCR, retrieval, and forecasting

| Model family | Checkpoint and format | Hardware | Status | Guide |
| --- | --- | --- | --- | --- |
| Surya OCR 2 | `datalab-to/surya-ocr-2`, BF16 safetensors loaded into the FP32 runtime; GGUF is a cross-check only | `gfx1151` | Implemented | [Surya](model-cards/MODEL-SURYA.md) |
| VibeVoice ASR | `microsoft/VibeVoice-ASR-HF`, bf16 safetensors; standalone [Q4_K_M GGUF](https://huggingface.co/shisa-ai/VibeVoice-ASR-Q4_K_M) | `gfx1151` | Native `LLM.transcribe()`; production measurements incomplete | [VibeVoice ASR](model-cards/MODEL-VIBEVOICE-ASR.md) |
| VibeVoice TTS | `microsoft/VibeVoice-1.5B`, bf16 safetensors | `gfx1151` | Experimental `LLM.synthesize()`; measurements partial | [VibeVoice TTS](model-cards/MODEL-VIBEVOICE-TTS.md) |
| EVIE 4.5B / 8B | `tencent/EVIE-4.5B` / `EVIE-8B`, Safetensors `fp32`, `fp16` | `gfx1151` | Complete; multimodal document retrieval encoding | [EVIE](model-cards/MODEL-EVIE.md) |
| TimesFM 2.5 200M | `google/timesfm-2.5-200m-pytorch`, Safetensors `fp32` | `gfx1151` | Complete; GPU decode | [TimesFM 2.5](model-cards/MODEL-TIMESFM.md) |
| TimesFM 3.0 500M | `google/timesfm-3.0-pytorch`, safetensors; `fp16` production / `fp32` strict execution | `gfx1151` | Complete; GPU decode | [TimesFM 3.0](model-cards/MODEL-TIMESFM3.md) |

## In development

These families do not have a working implementation yet — a capability gap,
not a policy one.

| Model family | Checkpoint | State |
| --- | --- | --- |
| Moonshine ASR | `shisa-ai/shisa-realtime-asr-0.92b`, F32 source with an FP16 deployment artifact | Internal `gfx1151` runtime and benchmark surface; a public audio API surface is planned. See [Moonshine](model-cards/MOONSHINE.md). |
| GLM-OCR | `zai-org/GLM-OCR` | Source reviewed; implementation pending. See [GLM-OCR](model-cards/MODEL-GLM-OCR.md). |
| MinerU 2.5 | `opendatalab/MinerU2.5-2509-1.2B` | Source reviewed; implementation pending. See [MinerU](model-cards/MODEL-MINERU.md). |

## Long-context KV (DMS)

hipEngine includes [DMS](https://arxiv.org/abs/2506.05345) support and training
code, with a published [DMS checkpoint for Qwen3.8-27B
Q4_K_M](https://huggingface.co/shisa-ai/Qwen3.8-27B-Q4_K_M-DMS-W8192).
The [DMS analysis](reference/DMS-ANALYSIS.md) records the quality bar and the 8K–232K
evidence ladder. Measured 24 GB capacity is in the root README's [Long context
on a 24 GB GPU](../README.md#long-context-on-a-24-gb-gpu) section.

## Platform limits

- hipEngine uses one GPU. Multi-GPU inference is not yet implemented.
- NVIDIA support is Maple-Preview only, through `backend="cuda_sm120a"`.
- GPU results are host-specific. See the [benchmark report](../benchmarks/README.md)
  for the exact host and protocol behind every published number.
- APIs and measured coverage can still change before 1.0.
