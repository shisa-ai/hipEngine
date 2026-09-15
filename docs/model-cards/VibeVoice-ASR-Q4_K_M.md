---
license: mit
base_model: microsoft/VibeVoice-ASR-HF
library_name: hipengine
pipeline_tag: automatic-speech-recognition
tags:
  - gguf
  - quantized
  - vibevoice
  - rocm
  - hipengine
---

# VibeVoice-ASR Q4_K_M for hipEngine

A quantized version of Microsoft's [VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF)
for [hipEngine](https://github.com/shisa-ai/hipEngine), a native HIP inference
engine for AMD GPUs. The GGUF is **6.75 GB**, about **59.5% smaller** than the
original 16.66 GB BF16 checkpoint weights.

## What's compressed?

The language model's attention and feed-forward weights use Q4_K with selected
weights in Q6_K. Token embeddings are also stored in Q4_K. The audio encoders,
audio connectors, output head, norms and biases remain BF16.

| Weight files | Size |
| --- | ---: |
| Original BF16 checkpoint | 16.66 GB |
| This Q4_K_M GGUF | **6.75 GB** |

Sizes are decimal GB on disk, not total runtime memory. hipEngine expands token
embeddings to BF16 when loading and also needs memory for audio processing,
caches and temporary buffers.

## Accuracy check

Evaluated on **200 LibriSpeech test-clean clips across all 40 speakers**,
covering 29 minutes of English read speech. Both hipEngine lanes received the
same prepared requests and used greedy decoding. Lower WER is better.

| hipEngine weights | Word error rate | Clips scored |
| --- | ---: | ---: |
| BF16 | 2.097% | 200/200 |
| Q4_K_M | **2.011%** | 200/200 |

Neither lane produced malformed transcripts. This subset showed no aggregate
WER degradation from quantization; it is not the full test-clean benchmark or
a qualification of other languages, noisy audio or speaker diarization.
WER uses Whisper English normalization and total word errors divided by total
reference words. Tested on Radeon 8060S (gfx1151).
[Evaluation details](https://github.com/shisa-ai/hipEngine/blob/main/benchmarks/results/2026-09-14-gfx1151-vibevoice-asr-200clip-comparative.json).

## Use with hipEngine

Use hipEngine's [VibeVoice-ASR Q4 evaluation runner](https://github.com/shisa-ai/hipEngine/blob/main/scripts/vibevoice_asr_wer_full.py)
with `--lanes hipq4 --gguf /path/to/vibevoice-asr-q4km.gguf`.
The current runner also requires the original HF checkpoint for the audio
frontend, configuration and processor; the GGUF alone is not a standalone
transcription package. Its tensor layout is specific to hipEngine and has not
been validated with llama.cpp or other GGUF runtimes.

See the [original model card](https://huggingface.co/microsoft/VibeVoice-ASR-HF)
for model capabilities, training, intended use and limitations.
