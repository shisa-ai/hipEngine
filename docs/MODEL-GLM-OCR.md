# MODEL-GLM-OCR.md — GLM-OCR on hipEngine

Status: **reviewed; implementation pending** (2026-09-11).

Target `zai-org/GLM-OCR` is a 0.9B-class OCR VLM with a CogViT-derived vision
tower, convolutional downsampling connector and small GLM decoder. Its primitives
overlap existing vision and text kernels, but its normalization, connector and
rotary semantics require their own model contract. Start with image/crop
recognition; add the document-layout pipeline and MTP as separately validated
milestones. [Card][card]

## Model and geometry

| Component | Pinned config / implementation |
| --- | --- |
| Architecture | `GlmOcrForConditionalGeneration`, `model_type=glm_ocr` |
| Text width / layers / FFN | 1536 / 16 / 4608, SiLU gated MLP |
| Attention | 16 query / 8 KV heads × 128; query projection width 2048, not hidden size 1536 |
| Decoder normalization | Four RMSNorms per layer around attention/MLP; eps 1e-5 |
| Vocabulary / output | 59392; untied LM head |
| Context / EOS | Config ceiling 131072; EOS IDs `[59246,59253]`, pad 59246 |
| Rotary config | Theta 10000; sections `[16,24,24]`, partial factor 1.0 |
| Vision | 24 blocks, width 1024, 16 heads × 64, FFN 4096, SiLU |
| Vision normalization | RMSNorm including Q/K head normalization |
| Patches / merge | Spatial 14, temporal 2, merge 2; image_size 336 is not a mandatory input size |
| Downsampling | Post-tower RMSNorm, then stride-2 2×2 Conv2d, 1024→1536 |
| Connector after downsampling | Projection → LayerNorm → erf GELU → gated SiLU MLP, output 1536 |
| Image IDs | Start 59256, end 59257, placeholder 59280 |
| MTP | One nextn prediction layer declared |

[Config][config], [Transformers implementation][model]. The connector is not
EVIE's concatenate-four-patches GELU merger. The decoder normalizes each
attention/MLP result before residual addition as well as normalizing its input;
a two-norm Llama/Qwen block is not equivalent. Head count × head dimension
must be independent of hidden width in all projections and scratch sizing.

The processor config names `Glm46VImageProcessor` / `Glm46VProcessor`, patch
14, merge 2, CLIP mean/std and area bounds 12544–9633792 pixels. Preserve actual
processor behavior and template expansion, including final `grid_thw` and
continuation positions. A still image nominally contributes H*W/784 merged
features at aligned processor dimensions. Large allowed image area is not a
validated runtime memory budget. [Processor config][processor]

## Oracles and a rotary discrepancy to resolve

Use pinned Transformers for boundary fixtures and the upstream SDK for task
prompts/postprocessing. The SDK supports vLLM and other image-capable serving
backends; its layout detector is a separate model. llama.cpp has explicit GLM-OCR
conversion and vision handling, and `ggml-org/GLM-OCR-GGUF` provides artifacts.
These are useful independent references, not proof of exact cross-engine parity.
[SDK][sdk], [llama.cpp conversion][conversion], [GGUF repository][gguf]

**Do not flatten the reference differences.** The reviewed Transformers path
uses three-axis position tensors and recomposes rotary sections, consistent
with the pinned HF config's factor 1.0. The reviewed llama.cpp `GlmOCRModel`
converter explicitly sets `use_mrope=False` and `partial_rotary_factor=0.5`,
with GLM weight-layout conversion inherited from its parent. This is an observed
implementation/configuration discrepancy; its numerical effect was not tested.
Capture the exact GGUF metadata and effective model positions, compare Q/K after
rotary and full logits on identical image inputs, and establish which reference
matches the desired checkpoint semantics. Do not silently adopt scalar/half-RoPE
as equivalent or claim the two engines already constitute an exact oracle pair.
[Transformers rotary source][model], [converter source][conversion]

## Model-only recognition versus complete document parsing

```text
model-only: image/crop + task prompt → CogViT → connector → GLM → text output

SDK pipeline: page → PP-DocLayoutV3 → region crops + task labels
                  → parallel GLM-OCR calls → coordinate/order restoration
                  → formatted document result
```

The SDK uses task prompts for text, formula and table recognition. Start with
those prompts rather than generic chat examples. PP-DocLayoutV3 is required for
the SDK's local document-layout stage, not for recognizing a supplied crop.
Current SDK config selects `PaddlePaddle/PP-DocLayoutV3_safetensors`; the Paddle
export with a similar name is a different loader artifact. Keep external layout
at the application boundary initially, then decide separately whether to port
it into hipEngine. Preserve box/polygon crop behavior, label routing, reading
order and failed-region reporting. [SDK config][sdkconfig], [layout loader][layout]

The model/SDK declare MIT; the integrated PP-DocLayoutV3 model is described by
the SDK as Apache-2.0. Pin each artifact's notices separately. [SDK license section][sdk]

### MTP is optional inference work

The model was trained with an MTP objective and config declares an extra layer.
The current SDK documents explicit vLLM speculative configuration with three
speculative tokens, so there is an inference implementation to study. However,
MTP is not required to run the ordinary Transformers generation path and is not
a prerequisite for the first hipEngine port. Inventory nextn tensors and validate
that the selected checkpoint/conversion preserves them before planning reuse.
[SDK serving example][sdk], [config][config]

Existing Qwen3.5 MTP code supplies transaction/verification patterns, not a
compatible GLM head. First establish true no-MTP autoregressive correctness and
performance. Later qualify every rejection/acceptance depth, rollback, EOS,
request isolation, task quality and a true same-protocol no-MTP baseline. Follow
the repository's full category-suite plus held-out policy and add OCR documents;
OCR-only acceptance numbers or verifier-derived off rows cannot replace it.

## Reuse and implementation sequence

| Piece | In-tree reference | Required adaptation |
| --- | --- | --- |
| Vision primitives/oracles | `hipengine/runtime/evie.py`, `hipengine/kernels/cpu_reference/evie.py` | CogViT RMSNorm/QK norm/SiLU, patch 14 and new connector |
| Multimodal controls | `hipengine/generation/qwen4_exp_multimodal.py`, `hipengine/server/multimodal.py` | GLM template, token IDs, positions and page resource limits |
| Dense decode primitives | Existing linear/norm/attention/rotary kernels | Four-norm GLM block, asymmetric projection widths and confirmed rotary layout |
| MTP lifecycle examples | `hipengine/generation/qwen35_gguf_mtp2.py` | Separate GLM nextn weights/head and numerical qualification |

1. Add proposed `hipengine/models/glm_ocr.py`, weight/spec validation and
   `scripts/glm_ocr_oracle_torch.py`. Pin processor/tokenizer/library and compare
   actual checkpoint shapes, including nextn, with config-derived expectations.
2. Prove the text block with small fixtures: all four norms, 2048-wide Q,
   rotary pairing and tied/untied head semantics. Resolve the reference rotary
   discrepancy before accepting a GGUF oracle.
3. Port the vision blocks and downsampler/connector against common processed
   patches. A 2×2 Conv2d can be represented by a packed linear operation only
   after channel/patch ordering and bias equivalence are tested.
4. Connect image embeddings to prefill/cached generation and validate each task
   prompt. Publish model-only recognition as the first bounded capability.
5. Connect an external SDK layout stage to the image-capable server, validating
   formats, limits and scheduling. Treat a native detector as an additional port.
6. Qualify production, profile page/crop workloads, then evaluate MTP separately.

## Task and performance gates

Freeze mixed document fixtures and held-outs: printed/scanned text, Japanese and
English, math, tables, seals, columns, rotation, blank images and dense small
text. Capture patch embeddings, normalized Q/K, downsampled features, connector
output, logits and cached steps. Add exact tests for placeholder counts, both
EOS IDs, image positions, chunked prefill, cancellation and inter-request state.

Measure recognition error, formula edit distance, table structure, layout and
reading order, malformed output and omissions. Compare supplied-crop recognition
separately from predicted-layout pipeline quality; use a pinned OmniDocBench
protocol for full-pipeline assessment. Do not attribute detector failures or
postprocessor repairs to VLM arithmetic alone.

Report resolution/patch tokens, regions per page, task/output lengths, layout
CPU/GPU time, preprocessing, vision, prefill, decode, assembly, complete latency,
pages/s and peak memory. Include layout placement and retries in the protocol.
Do not compare a model-only crop run with upstream document-pipeline rates.
MTP wins must survive the same correctness suite and true AR baseline on the
same physical host.

## Repository contracts and evidence

This is a source review, not an implemented plugin or a measured performance
result. Proposed implementation paths are future work. Keep runtime imports
torch-free, resolve kernels through `(backend, layer, quant, variant)`, retain
registered strict fallbacks, and preserve `KVLiveSpans` for cached attention.
Before a kernel port, read [KERNELS.md](KERNELS.md) and run
`python3 scripts/check_lineage.py --kind kernel --diff stat`; develop in-tree
and record upstream file/commit provenance. Add HIP-availability guards to GPU
tests and verify new kernels with a prebuilt-cache `rocprofv3` trace.

[EXECUTION-PROFILES.md](EXECUTION-PROFILES.md) governs promotion: exact control
and ownership, calibrated mean/p95/p99/max KL and category top-1 where logits
apply, determinism, isolation, BF16-relative and task-quality gates. Continuous
features need their own calibrated error measures plus downstream task checks.
The KL ≤ 0.05 / top-1 ≥ 90% outer smoke floor alone is insufficient. Assess
non-bit-identical candidates under production gates before rejecting them.

Benchmark matched artifacts, inputs, settings and workloads on the same physical
host. Record model/quant, shape, host identity, hardware, exact commands, result
and correctness gate. W7900/gfx1100 is the default lane; gfx1151 measurements
are independent. Follow [BENCHMARK.md](BENCHMARK.md) and update the result
artifact, scoreboard and changelog for any accepted measurement. No speedup or
validated capacity is claimed here. Update `PLAN.md` if implementation changes
architecture and track temporary loaders/flags in `REFACTOR.md`.

## Source pins

Reviewed on 2026-09-11 against hipEngine
`0dacb0df29864c622a787f8e42f64b4e36be9894`. Links below pin the inspected
config/code revisions where available. No weights were downloaded and no model
was executed. Library sources describe those revisions, not a tested dependency
set; the first implementation milestone must freeze a working oracle environment.

[card]: https://huggingface.co/zai-org/GLM-OCR/blob/ca5d8b3e287e52589e37c28385d9655ee4372f9d/README.md
[config]: https://huggingface.co/zai-org/GLM-OCR/blob/ca5d8b3e287e52589e37c28385d9655ee4372f9d/config.json
[processor]: https://huggingface.co/zai-org/GLM-OCR/blob/ca5d8b3e287e52589e37c28385d9655ee4372f9d/preprocessor_config.json
[model]: https://github.com/huggingface/transformers/blob/177e90dd2d51273fa235dd8bacee7c80f1eef067/src/transformers/models/glm_ocr/modeling_glm_ocr.py
[sdk]: https://github.com/zai-org/GLM-OCR/blob/cef4d0ea120d1741f5cefe8985eee45f6c8eff1d/README.md
[sdkconfig]: https://github.com/zai-org/GLM-OCR/blob/cef4d0ea120d1741f5cefe8985eee45f6c8eff1d/glmocr/config.yaml
[layout]: https://github.com/zai-org/GLM-OCR/blob/cef4d0ea120d1741f5cefe8985eee45f6c8eff1d/glmocr/layout/layout_detector.py
[conversion]: https://github.com/ggml-org/llama.cpp/blob/451b89bae0c4b1dd612eb503ceace906c01ddcc9/conversion/glm.py
[gguf]: https://huggingface.co/ggml-org/GLM-OCR-GGUF
