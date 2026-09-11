# MODEL-SURYA.md — Surya OCR 2 on hipEngine

Status: **implemented — CPU reference and gfx1151 HIP lane** (2026-09-11). The
per-model architecture and bring-up plan below is followed by an
implementation-status section. Torch-free preprocessing, vision tower, text
decoder, and the public OCR API run end to end; the HIP lane executes the
vision tower and text decoder on gfx1151.

### Implementation status (2026-09-11)

| Area | State | Evidence |
| --- | --- | --- |
| Model/weight/tokenizer contracts | done | `hipengine/models/surya.py`, `hipengine/loading/surya.py`, `tests/test_surya_model_contract.py` |
| Torch oracle fixtures | done | `scripts/surya_oracle_torch.py`, `tests/fixtures/surya/*.npz`, `tests/test_surya_oracle_fixtures.py` |
| CPU-reference text decoder (GDN state, causal cache, mRoPE) | done | `hipengine/kernels/cpu_reference/surya.py`, `tests/test_surya_text_decoder.py` |
| CPU-reference vision tower + merger | done | `tests/test_surya_vision.py` |
| Public OCR API, greedy oracle parity | done | `hipengine/loading/surya.py:run_surya_ocr`, `tests/test_surya_e2e.py`, `tests/test_surya_public_api.py` |
| HIP text decoder on gfx1151 | done | `hipengine/runtime/surya.py`, `tests/test_surya_gpu.py` |
| HIP vision tower on gfx1151 | done | `tests/test_surya_gpu.py::test_gpu_vision_matches_oracle_features` |
| Registered HIP generator (`surya_ocr2`/`hip_gfx1151`/fp32) | done | `hipengine/generation/surya_gpu.py` |
| Request/resource contracts (greedy-only, capacity, EOS) | done | `hipengine/generation/surya_contract.py`, `tests/test_surya_generation_contract.py` |
| GPU `rocprofv3` kernel-trace evidence | done | `benchmarks/results/2026-09-11-gfx1151-surya-kernel-trace.json` |
| Multi-prompt `generate` isolation | done | `tests/test_surya_gpu.py::test_gpu_generator_multi_prompt_isolation` |
| Resize / grid-geometry branches | done | `tests/test_surya_gpu.py::test_gpu_vision_geometry_sweep_matches_cpu_reference` (4 cases; 3:1 aspect ratios measured but uncommitted) |
| Rectangular-page coverage | done (vision + isolation) | `tests/test_surya_gpu.py` rect vision vs `oracle_rect` `vision_merged` |
| Full-page coverage (1024x1024, real text) | done | `tests/fixtures/surya/page_full.png`, `scripts/surya_oracle_greedy.py`, `tests/test_surya_gpu.py::test_gpu_full_page_layout_matches_oracle` |
| OCR output quality check (labels + bboxes vs drawn geometry) | done | `tests/test_surya_gpu.py::test_gpu_full_page_layout_matches_oracle` second half |
| Greedy reference fixtures are reproducible | done | `scripts/surya_oracle_greedy.py --case all` regenerates `oracle_greedy.json`, `oracle_fullpage_greedy.json` and `oracle_corpus.json` byte-for-byte |
| Multi-page held-out OCR corpus | done | `page_columns.png` (two-column) and `page_list.png` (numbered list) via `tests/test_surya_gpu.py::test_gpu_ocr_corpus_matches_oracle` |
| Registry migration + `KVLiveSpans` KV ABI | pending | tracked in `docs/REFACTOR.md` |
| Quantized (GGUF) Surya decoder | pending | safetensors fp32 is the implementation target |
| MTP / speculative decoding | pending | 15 MTP tensors inventoried, unused |

Correctness gates in place: CPU reference matches the torch fp32 oracle on
greedy token IDs (`tests/fixtures/surya/oracle_greedy.json`), the HIP vision
tower sits inside the CPU-reference noise band against `vision_merged`, and the
full HIP pipeline (GPU vision features + GPU prefill/decode) reproduces the
oracle greedy IDs exactly. On the full-size page the torch fp32 oracle, the
NumPy CPU reference, and the gfx1151 HIP lane all produce the same 78-token
layout-JSON output for `page_full.png` (grid 1x64x64, 1024 merged visual
tokens). All HIP tests carry a ROCm-availability guard.

The full-page gate also checks the decoded text is the *right answer* for the
page that was drawn, not merely that the lanes agree: the output parses as
JSON, the label sequence is `Section-Header, Text, Text, Caption, Table`, every
bbox is in range and ordered, and the header/caption/table boxes match the
drawn geometry (heading at y=50, caption at y=458, table below it).

Two held-out layouts cover structure the single-column fixtures cannot
produce. `page_columns.png` (512x512) has a spanning title and two prose
columns; the oracle keeps them apart as a left-column pair (x 65-419) and a
right-column pair (x 543-897), so column detection is exercised rather than
just text finding. `page_list.png` (512x512) decodes to
`Section-Header` + `List-Group` + `Text`, and is the only fixture that reaches
the list label. Both are 1x32x32 grids and both reach a natural EOS (94 and 46
tokens).

Decode is one row per step, and rocBLAS SGEMM is tuned for a wide `n`: at
`n == 1` it reads the weights at roughly a third of the bandwidth SGEMV
reaches. `SuryaGpuRunner._gemm` therefore routes `rows == 1` to
`Rocblas.sgemv_rowmajor_nt`. Both accumulate in fp32, so only the summation
order changes; the greedy oracle gates are unaffected. End-to-end this took a
full OCR request from 2.051 s to 1.276 s on gfx1151 (decode 32.9 to 55.5
tok/s). The same fp32 `rows == 1` pattern exists in `runtime/evie.py`,
`runtime/timesfm_decode.py` and `runtime/timesfm3_decode.py`, which have not
been converted.

### Measured inventory (2026-09-11, revision `3b3d4cdf`)

Checkpoint downloaded and inventoried; the geometry table above matches the
actual config and tensors on every value checked.

- **Safetensors (canonical):** single BF16 `model.safetensors`, 1.37 GB,
  488 tensors, 686.2M parameters (language 565.1M + visual 100.6M + MTP
  20.5M; advertised "650M" excludes MTP). Tied LM head confirmed: no
  `lm_head` tensor; output uses `model.language_model.embed_tokens.weight`.
- **Weight namespaces:** `model.language_model.layers.N.*` (GDN layers use
  `linear_attn.in_proj_qkv/a/b/z`, `conv1d (6144,1,4)`, `A_log (16,)`,
  `dt_bias (16,)`, `norm (128,)`, `out_proj (1024,2048)`); full-attention
  layers use fused `q/k/v_proj` with per-head `q_norm`/`k_norm (256,)`;
  vision uses `model.visual.*` with fused `attn.qkv (2304,768)` **with
  biases** (EVIE has no vision biases), `patch_embed.proj (768,3,2,16,16)`,
  learned `pos_embed.weight (2304,768)`, `merger.norm (768,)` +
  `linear_fc1 (3072,3072)` / `linear_fc2 (1024,3072)` with biases.
- **MTP tensors present:** 15 tensors (`mtp.fc`, one full-attention layer,
  `pre_fc_norm_{embedding,hidden}`, `mtp.norm`); shared embeddings
  (`mtp_use_dedicated_embeddings: false`). Optional follow-up, as above.
- **Token IDs (trap 1 resolved):** tokenizer + generation_config agree —
  EOS **2** (`<|im_end|>`), pad 0, vision_start 9 / end 10, image_pad 11,
  video_pad 12, unk 65424. `text_config.eos_token_id: 248044` is stale
  metadata outside the 65,425 vocab; never used. Chat template renders
  `<|im_start|>assistant\n` with no thinking prefix, as documented.
- **GGUF pair (same repo family, `6a3a4c30`):** `surya-2.gguf` 1.27 GB +
  `surya-2-mmproj.gguf` 0.20 GB. **Unquantized** (file_type F16/F32 only),
  so a same-precision cross-oracle rather than a quant question. Metadata
  matches the safetensors config on all geometry keys
  (`qwen35.*`, `ssm.*`, `clip.*`, projector `qwen3vl_merger`).
  Anomalies: `tokenizer.ggml.tokens` holds **130,854** entries against
  65,425 `token_type` values (doubled vocab array — do not trust GGUF
  tokenizer contents; use the HF tokenizer files), `merges` is a stub
  `[1 x 8]`, and `general.finetune = '_patched_ckpt'` records a
  llama.cpp-conversion patch. EOS=2 / pad=0 confirmed in GGUF metadata.
- The safetensors path is the implementation target; GGUF is a cross-check
  oracle only.

Surya needs a vision-to-generation path, but most of its computation already
has close relatives here: EVIE supplies a Qwen3.5 vision encoder and GDN
prefill, the Qwen3.5 GGUF generator supplies autoregressive decoding, and
Qwen4Exp supplies multimodal serving and position-control examples. The work
is to adapt and connect these pieces with Surya's weights, tokenizer, image
processing, and OCR protocol. Existing family support does not establish that
this checkpoint already loads or generates correctly.

## Model and scope

**Checkpoint:** [datalab-to/surya-ocr-2][hf], advertised as 650M parameters,
BF16 safetensors. It is `Qwen3_5ForConditionalGeneration`: an image encoder
feeding a causal hybrid language decoder. Eighteen Gated DeltaNet layers are
normal for this Qwen3.5 configuration; they are not a new kernel class for
hipEngine. [Configuration][config]

**First target:** one page image → generated HTML with layout labels and
bounding boxes. Current upstream `RecognitionPredictor` chooses full-page
mode when no layout results are supplied. Layout JSON, block OCR, and table
structure are additional prompts to the same VLM. Failed full-page output
can fall back to VLM layout followed by block OCR. [Recognition source][recognition]

The separate text-line detector and RF-DETR `FastLayoutPredictor` are optional
adjacent capabilities, not prerequisites for this path. Porting the complete
Surya/Marker toolkit would be a larger task. The older TODO claim that two
auxiliary torch models block Surya OCR support does not describe the current
full-page path. [Fast-layout source][fastlayout]

The upstream code is Apache-2.0; the weights have the **modified AI Pubs
OpenRAIL-M** license. Keep the weight license distinct from the runtime's
license and preserve its notices for any distributed conversion. [Weight license][license]

### Geometry

Values below are from the pinned checkpoint config, rather than inferred
from the Qwen family name. Validate actual tensor names/shapes before loading.

| Component | Surya OCR 2 |
| --- | --- |
| Text width / layers / FFN | 1024 / 24 / 3584, SiLU gated MLP |
| Layer schedule | Three GDN then one full attention; 18 GDN + 6 full |
| Full attention | 8 query heads, 2 KV heads, head dim 256; sigmoid output gate |
| GDN | 16 key heads × 128, 16 value heads × 128; causal conv width 4 |
| Text normalization | RMSNorm, epsilon 1e-6 |
| Rotary | 64 rotated dimensions per 256-d head; theta 10,000,000; interleaved mRoPE sections `[11,11,10]` |
| Vocabulary / head | 65,425; tied input/output embeddings |
| Configured context ceiling | 262,144; not a validated OCR serving budget |
| Vision | 12 blocks, width 768, 12 heads × 64, FFN width 3072 |
| Patches | RGB, spatial 16×16, temporal 2, spatial merge 2×2 |
| Learned vision positions | 2304 entries = 48×48 table, interpolated to the patch grid |
| Vision activations | Block MLP: tanh-approximate GELU; merger: erf GELU |
| Merger | Per-patch LayerNorm → concatenate four patches → 3072→3072→1024 MLP |
| DeepStack | Empty visual-index list; no intermediate feature injection |
| MTP metadata | One next-token prediction layer declared; optional follow-up |

Sources: [checkpoint config][config] and [Transformers Qwen3.5 implementation][transformers].
MTP tensor availability and runner compatibility still need an inventory;
metadata alone does not establish usable speculative decoding.

### Forward path and positional semantics

```text
RGB page → resize/normalize → packed temporal/spatial patches
         → vision patch projection + interpolated learned positions
         → 12 bidirectional vision blocks with 2D rotary positions
         → 2×2 merger → 1024-d image features

Surya chat template → token IDs + expanded image placeholders
                   → token embeddings with image-feature substitution
                   → 24 causal hybrid decoder layers → tied LM head
                   → cached autoregressive decode → HTML/JSON → page results
```

At final processor dimensions H×W (multiples of 32), a still image produces
`P = (H/16)*(W/16)` vision rows and `P/4 = H*W/1024` decoder image tokens.
The temporal pair duplicates the still image; it does not double the decoder
token count. A **processor-output** size of 1024×1024 therefore gives 4096
vision rows and 1024 image tokens. This is geometry, not a measured workload.

Vision attention is bidirectional within each image; text attention is causal.
Decoder mRoPE coordinates and physical KV/token offsets are different objects.
Image coordinates compress spatial progress, so the next generated token must
use the processor's continuation offset while cache writes advance through
physical sequence slots. Do not replace mRoPE with scalar positions without a
separate production-quality evaluation. [Transformers implementation][transformers]

## Existing implementations and reuse

| Reference | What to use it for | Limitation |
| --- | --- | --- |
| Upstream Surya + vLLM | Exact task prompts, client preprocessing, parsing, retries, and full-page/block orchestration | External oracle/application; do not import its torch dependency tree into generation |
| Official GGUF + llama.cpp multimodal path | Same-artifact quantized decoder comparison and independent end-to-end image oracle | Requires both `surya-2.gguf` and `surya-2-mmproj.gguf`; inspect tensor types and metadata rather than guessing quant from filename |
| Transformers Qwen3.5 | FP32 teacher, BF16 reference, intermediate tensors, processor and cached-position semantics | Keep torch in fixture/oracle tools only; pin a working revision |
| MLX-VLM Qwen3.5 + community Surya conversion | Independent shape/weight adaptation cross-check; its vision module reuses Qwen3-VL | Apple-specific, not a ROCm performance baseline; conversion-specific quantization is not a correctness contract |

Reviewed [GGUF inventory][gguf], [Surya OpenAI client][client],
[llama.cpp backend wrapper][llamacpp], and [MLX conversion card][mlxcard].
The [MLX vision adapter][mlxvision] is a secondary implementation reference;
its moving `main` source must be commit-pinned before copying any code.
No external implementation was executed during this review.

### In-tree candidates

| Piece | Existing path | Adaptation needed |
| --- | --- | --- |
| Vision math and orchestration | `hipengine/runtime/evie.py`, `hipengine/kernels/cpu_reference/evie.py`, `hipengine/kernels/hip_gfx1100/evie/evie_ops.{hip,py}` | Extract shared configuration-driven encoder; Surya shapes and independent weight/spec contract |
| Model/weight validation pattern | `hipengine/models/evie.py`, `hipengine/loading/evie.py` | No EVIE retrieval projection or MaxSim; map Surya checkpoint namespaces and tied LM head |
| Cached text generation | `hipengine/models/qwen35.py`, `hipengine/loading/qwen35_gguf.py`, `hipengine/runtime/qwen35_gguf_runner.py`, `hipengine/generation/qwen35_gguf.py` | Validate small decoder geometry and quant coverage; add image embedding injection and three-axis rotary/continuation control |
| GGUF vision loading example | `hipengine/loading/qwen4_exp_vision_gguf.py`, `hipengine/loading/qwen4_exp_vision_materialize.py` | Surya mmproj tensor inventory; parameterize validated geometry |
| Multimodal control/serving | `hipengine/generation/qwen4_exp_multimodal.py`, `hipengine/server/multimodal.py` | Surya template/token IDs; capability-based integration and realistic page limits |
| Alternative encoder reference | `hipengine/runtime/qwen4_exp_vision.py` | Currently fixes width 1152 and FFN 4304; cannot directly load Surya |

EVIE already distinguishes tanh GELU in blocks from erf GELU in the merger.
Its recent 8B repair also derives merger width from `merge² * vision_width`,
not the block FFN width. Preserve that distinction even though both happen to
be 3072 for Surya. See the [8B repair handoff](../worklog/entries/20260910T113713.093689Z-lhl-evie-cbf1b8.md).

Prefer the existing GGUF decoder as the initial generation route if inventory
confirms coverage. Safetensors vision + GGUF text is a reasonable temporary
bridge if mmproj loading is slower to implement; record both source hashes
and remove duplicate loaders through `REFACTOR.md` once the final route works.
Do not turn EVIE's prefill-only retrieval runtime into a second independent
language-generation engine.

## Correctness traps to address first

1. **Special tokens and EOS disagree across files.** The tokenizer and
   `generation_config.json` identify EOS as ID **2** (`<|im_end|>`), padding as
   **0**, and image/start/end as **11/9/10**. `text_config.eos_token_id` is
   **248044**, outside this 65,425-token vocabulary. Resolve effective stopping
   from validated tokenizer/generation metadata (and GGUF equivalents), with a
   regression test; never carry stock Qwen token IDs into Surya. Use the supplied
   chat template, which ends at the assistant prefix without a thinking prefix.
   [Tokenizer][tokenizer], [generation config][generation], [chat template][template]
2. **There are two resizing stages in upstream serving.** Surya's client
   `scale_to_fit` uses Lanczos and a 28-pixel grid with default area bounds
   `1792*28` to `3072*2048`; the model processor subsequently uses patch 16 /
   merge 2, bicubic resampling, RGB rescale 1/255, mean/std 0.5. Processor size
   fields are pixel-area bounds (65,536–16,777,216), not literal side lengths.
   Capture both stages, patch packing order, temporal duplication, interpolation,
   and final `grid_thw`. A cleaner one-resize implementation would change inputs.
   [Client resizing][util], [processor configuration][processor]
3. **Embedding injection alone is insufficient.** Validate placeholder counts,
   feature order, mRoPE axes, decode offset, and per-request GDN convolution and
   recurrent state. Preserve state across prefill chunks; reset between requests.
   EVIE prefill parity is useful evidence, not a cached-decoding gate.
4. **Existing server limits are too small to assume page compatibility.** The
   bounded PNG helper defaults to 1024-pixel sides; the Qwen4Exp vision runner
   defaults to only 256 patch rows. Audit actual callers and resource budgets.
   Avoid silent image reduction or truncation. Use configured limits with clear
   errors and account for image tokens, prompt tokens, output, and concurrency.
5. **Upstream defaults contain a context-budget mismatch.** Pinned settings use
   a 12,288-token full-page output cap and 12,288 context tokens per llama.cpp
   slot; the comment still budgets an 8192-token output. Copy neither budget
   blindly: image and prompt tokens also occupy context. [Settings][settings]
6. **Task behavior is more than raw generation.** Preserve upstream prompt
   strings, normalized 0–1000 box coordinates, original-page coordinate mapping,
   reading order, label mapping, HTML/table structure, and errors. The client
   requests token logprobs and computes mean token probability. Guided layout
   defaults on; guided table decoding defaults off. Check hipEngine's schema
   subset against the actual layout schema, including the bbox string pattern.
   [Prompts/schema][prompts], [parsers][parsers], [client][client]

## Suggested implementation sequence

These were proposed milestones and paths. Steps 1–4 are implemented (see the
status table above); step 5 is partially implemented (public OCR API and
parsing, no OpenAI-compatible service path); step 6 is pending.

1. **Freeze contracts and generate oracle fixtures.** Add
   `hipengine/models/surya.py` and targeted contract tests; inventory safetensors
   and both GGUF files, including tied weights, token IDs, and optional MTP.
   Add `scripts/surya_oracle_torch.py` plus small fixtures under
   `tests/fixtures/surya/`. Capture identical processed inputs, vision boundary
   tensors, merged features, first logits, and teacher-forced cached steps.
2. **Prove the text checkpoint independently.** Exercise the existing Qwen3.5
   decoder with Surya tokenization and known text prefixes, including EOS and
   chunked prefill. This isolates weight/quant/head/state failures before images.
3. **Adapt the shared vision encoder.** Implement a Surya loader and a reusable
   encoder interface; expose merged features and grid metadata without EVIE's
   retrieval head. Validate tiny, rectangular, odd-input-size, and full-page
   grids against common processor outputs. Add CPU-reference coverage before
   any new kernel math. Re-run affected EVIE gates if shared code changes.
4. **Connect image prefill to generation.** Register the Surya model/generator
   through existing plugin interfaces; add image embedding overrides and mRoPE
   control to the reused runner. First acceptance: one image generates valid
   full-page OCR HTML through a torch-free public path, with correct cached state.
5. **Expose the OCR protocol.** Add parsing and test the OpenAI-compatible image
   request path. Upstream `SURYA_INFERENCE_URL` is a useful eventual integration
   point after model naming, images, schema support, and logprobs are verified.
   Add layout/table tasks and full-page→block fallback as explicit subsequent
   coverage. PDF rendering can stay at the application boundary.
6. **Qualify production and optimize.** Measure full-page costs first; only then
   choose batching, tiled vision attention, GEMM precision, or decode tuning.
   Auxiliary detector/fast-layout ports and MTP are independent follow-ups.

Keep kernels behind `(backend, layer, quant, variant)` registrations and retain
registered strict fallbacks. `KVLiveSpans` remains the attention ABI; mRoPE
coordinates do not replace cache ownership metadata. The current HIP runtime
does not yet meet this: it imports the `hip_gfx1100` JIT libraries directly and
uses a bespoke dense KV scatter offset scheme, exactly as the existing EVIE
runtime does. Both deviations are recorded in `docs/REFACTOR.md` with concrete
removal conditions; migrating them is a repo-wide refactor, not a Surya-only
change. Read `KERNELS.md` and run
`python3 scripts/check_lineage.py --kind kernel --diff stat` before any kernel
port.

## Validation and performance plan

Use three independent layers of evidence:

- **Exact control:** token/template/patch counts, feature placement, box scaling,
  EOS, positions, cache ownership, GDN reset, cancellation, chunk boundaries,
  same-schedule repeats, and neighboring-request isolation. Include text-only,
  image-plus-text, mixed-size batches, empty/blank pages, and truncated outputs.
- **Numerics:** common-input CPU/FP32 oracle at patch, block, merger, logits,
  and cached-step boundaries. For new kernels the outer floor is KL ≤ 0.05
  and top-1 ≥ 90% where output distributions apply; continuous vision features
  also need calibrated feature-error gates and downstream-logit evaluation.
  Production promotion requires the complete mean/p95/p99/max KL, category
  top-1, determinism/isolation, BF16-relative and task gates in
  [EXECUTION-PROFILES.md](EXECUTION-PROFILES.md). Calibrate the OCR task envelope
  before candidate tuning. Non-bit-identical candidates get production review.
- **OCR quality:** a frozen multi-page corpus plus held-outs covering English,
  Japanese/mixed scripts, dense small text, scans, multi-column reading order,
  tables with spans, math, and blank pages. Report CER/WER where appropriate,
  layout/box and order accuracy, table structure, malformed output, omissions,
  loops, and truncation; add olmOCR-bench for broader task validation. Compare
  raw one-pass inference separately from retries/fallback-assisted results.

Benchmark the same image bytes, processor settings, prompts, output budgets,
quantized artifacts, concurrency, and retry policy on the **same physical host**.
Report cold load, preprocessing, vision, prefill/TTFT, decode, parsing, complete
page latency, pages/s, output tokens, peak memory, and quality. Dense vision
attention scales quadratically in unmerged patch rows: profile scratch usage
at document resolution before adopting EVIE's full attention-score allocation.

No Surya rate or memory target is established here. EVIE retrieval throughput
and upstream NVIDIA/Apple results are not hipEngine OCR baselines. Begin on
the declared ROCm host (default W7900/gfx1100); gfx1151 is an independent lane.
New kernels require guarded GPU tests and a prebuilt-cache `rocprofv3` trace.
Any accepted benchmark must update the result artifact, scoreboard, and
changelog under the repository's evidence policy.

## Source pins

Reviewed against hipEngine `d64eb6d10d0d31d762d6df7943776fe46804f652`.
External references were read on 2026-09-11; weights were not downloaded.

- HF safetensors/config/tokenizer: `3b3d4cdf88d6928b0acdc75181b13206ea67c4a3`.
- Official GGUF repository: `6a3a4c30e5e74446d4f8b6afd05b2f2da970f470`.
- Surya source: `c0377481c10527ec8c87bec5a10a6567fdf613fd`.
- Transformers source: tag `v5.2.0`, matching checkpoint metadata; oracle execution
  still needs a tested environment pin.
- MLX conversion card: `dd513812189b8d9bfbe76f32e84f7a496c31fac1`.

[hf]: https://huggingface.co/datalab-to/surya-ocr-2/tree/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3
[config]: https://huggingface.co/datalab-to/surya-ocr-2/blob/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3/config.json
[processor]: https://huggingface.co/datalab-to/surya-ocr-2/blob/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3/processor_config.json
[tokenizer]: https://huggingface.co/datalab-to/surya-ocr-2/blob/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3/tokenizer.json
[generation]: https://huggingface.co/datalab-to/surya-ocr-2/blob/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3/generation_config.json
[template]: https://huggingface.co/datalab-to/surya-ocr-2/blob/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3/chat_template.jinja
[license]: https://huggingface.co/datalab-to/surya-ocr-2/blob/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3/LICENSE
[gguf]: https://huggingface.co/datalab-to/surya-ocr-2-gguf/tree/6a3a4c30e5e74446d4f8b6afd05b2f2da970f470
[transformers]: https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
[recognition]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/recognition/__init__.py
[fastlayout]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/fast_layout/__init__.py
[client]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/inference/backends/openai_client.py
[llamacpp]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/inference/backends/llamacpp.py
[util]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/inference/util.py
[settings]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/settings.py
[prompts]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/inference/prompts.py
[parsers]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/inference/parsers.py
[mlxcard]: https://huggingface.co/aglaia-models/surya-ocr-2-mlx/blob/dd513812189b8d9bfbe76f32e84f7a496c31fac1/README.md
[mlxvision]: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5/vision.py
