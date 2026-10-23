---
status: current
owns: Gemma 4 GGUF image input, vision sidecar contract, public usage, and implementation limits.
---
# Gemma 4 image input

Gemma 4 GGUF image generation uses a separate `mmproj` GGUF containing the vision
encoder and its projection into the text model. Pass it as `vision_model` to
`hipengine.LLM`, or as `--vision-model` to `hipengine serve`. The vision kernels
are registered for `hip_gfx1100` and `hip_gfx1151`; hardware validation in this
change covers gfx1100 only.

## Python

```python
import numpy as np
from PIL import Image
from hipengine import LLM, SamplingParams

llm = LLM(
    model="/path/to/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
    vision_model="/path/to/mmproj-F16.gguf",
    backend="hip_gfx1100",
    max_sequence_length=4096,
)
try:
    with Image.open("picture.png") as image:
        pixels = np.asarray(image.convert("RGB"))
    result = llm.generate_multimodal_detailed(
        "Describe this image.", pixels, SamplingParams(max_tokens=256, temperature=0),
    )
    print(result.text)
finally:
    llm.close()
```

Plain text is rendered through the GGUF chat template. A preformatted Gemma
conversation is preserved, including its thinking policy. Without an image
marker, images are inserted before the text. To place images between text
segments, use one `<|image|>` marker per image, in input order. The server's
ordered image-item mapping supports multiple images; video and audio are not
implemented by this encoder.

## Server

```bash
hipengine serve --model /path/to/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --vision-model /path/to/mmproj-F16.gguf --backend hip_gfx1100 \
  --max-context-tokens 4096 --served-model-name gemma4
```

Send an OpenAI-compatible chat request with `text` and `image_url` content
items. Image loading, URL policy and request size limits use the shared
[server multimodal interface](../API.md). The response's prompt-token count
includes the expanded image tokens. When no output limit is supplied, the
server budgets from that expanded prompt length. An explicit output limit is
kept as requested and checked against the model's context capacity.

## Encoder and text integration

- RGB images are resized to a patch grid divisible by the spatial pooling
  factor. The default budget is 280 pooled tokens per image, with dimensions
  derived from aspect ratio. Pillow supplies bicubic resizing; it is not
  bit-identical to the Transformers antialias implementation.
- The loader reads dimensions and tensor shapes from the artifact. It accepts
  the `clip` / `gemma4v` vision contract, without model-name or artifact-hash
  admission lists. The projection width must match the text embedding width.
- The tower uses FP32 weights and rocBLAS contractions, learned X/Y positions,
  per-head Q/K norms, scale-free V normalization, two-axis rotary embeddings,
  unscaled attention scores, gated GELU blocks, spatial pooling and projection.
- Projected image rows replace text embeddings **after** text embedding
  scaling. They are not scaled a second time. Sliding-window text layers allow
  bidirectional attention within each image span; global layers remain causal.
  Images do not attend to other images in the vision tower.
- Image spans cannot be split across prefill blocks. A span larger than the
  runner's prefill block is rejected with a capacity error. Total text plus
  image tokens and requested generation must fit the configured context.
- Image prefill requires BF16 KV storage. The INT8 attention consumer cannot
  execute the bidirectional image mask and is refused before vision encoding.
- Decoding uses the ordinary Gemma sampler, including sampling controls,
  stop sequences and log probabilities. Image generation is autoregressive;
  image-aware assistant-head speculative decoding is not implemented.
- The HIP tower implements equal query/KV head counts and unclipped linear
  layers. Sidecars requiring grouped-query vision attention or clipped
  linears fail with a named capability error. The NumPy oracle also models
  those arithmetic contracts.
- Vision weights are loaded lazily and remain resident until `LLM.close()`.
  Attention scratch belongs to one image forward and is freed on success or
  failure. No torch import is required.

This is image support, not a general multimodal checkpoint loader: Safetensors
vision weights, audio, video, and batched image-prefill scheduling are outside
this interface.
