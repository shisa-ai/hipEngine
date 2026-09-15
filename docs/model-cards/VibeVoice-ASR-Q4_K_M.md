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

## Comparison with another public GGUF

[cstr/vibevoice-asr-GGUF](https://huggingface.co/cstr/vibevoice-asr-GGUF)
offers a smaller quant for CrispASR. Inspection of its
[`vibevoice-asr-q4_k.gguf` tensor directory](https://huggingface.co/cstr/vibevoice-asr-GGUF/tree/f316299d4b39b2990d751c55b3dd6daab61677c7)
shows a different compression policy:

| Component | This hipEngine quant | cstr quant |
| --- | --- | --- |
| File size | 6.75 GB | 4.81 GB |
| LM layer matrices | Q4_K + 28 Q6_K tensors | Q4_K throughout |
| Token embeddings | Q4_K | Q4_K |
| Output head | BF16 | Q4_K |
| Audio encoders/connectors | BF16 | Mixed Q4_K/Q4_0/F16/F32 |
| Embedded tokenizer | No | Yes |
| Runtime layout | hipEngine | CrispASR |

Both files contain 901 tensors totaling 8.33 billion parameters. Of the
1.94 GB size difference, approximately 0.91 GB comes from the audio frontend
and 0.78 GB from the output head; most of the remainder comes from selected
LM weights kept in Q6_K here. Although cstr's card labels its file Q4_K_M,
the inspected file contains no Q6_K tensors.

This is a storage comparison, not a matched accuracy or speed benchmark.
Our higher precision does not establish better quality, and the files are
not interchangeable between runtimes.

See the [original model card](https://huggingface.co/microsoft/VibeVoice-ASR-HF)
for model capabilities, training, intended use and limitations.
