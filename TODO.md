# TODO

## Model support matrix (downloaded 2026-08-29)

llama.cpp column checked against a 2026-08-29 llama.cpp tree; hipEngine column from `hipengine.models.registered_models()` (plugins today: evie_4p5b [also serves EVIE-8B], laguna_gguf, maple, moonshine_asr, qwen3_5_gguf, qwen3_5_moe_gguf, qwen3_5_moe_paro, qwen4_exp_gguf, timesfm_2p5_200m, timesfm_3p0, toy_one_layer, vibevoice_asr).

Goal: basic hipEngine support for all of these. Start with models that have **no llama.cpp oracle either** (EVIE, VibeVoice ×2, shisa-asr) ordered by least new code; models with llama.cpp support get an independent GGUF comparison after checking model, processor and sampling semantics.

| Model | Architecture | llama.cpp | hipEngine |
|---|---|---|---|
| google/timesfm-2.5-200m | TimesFmModelForPrediction | ❌ (not a token LM) | ✅ `timesfm_2p5_200m` |
| google/timesfm-3.0-pytorch | TimesFM3Torch | ❌ | ✅ `timesfm_3p0` (GPU decode; 4.17x torch fp32 on gfx1151) |
| shisa-ai/shisa-realtime-asr-0.92b | MoonshineForConditionalGeneration | ❌ (runtime is sherpa-onnx) | ✅ `moonshine_asr` (native HIP decoder) |
| datalab-to/surya-ocr-2 | Qwen3_5 (gated DeltaNet) | ✅ official GGUF, documented llama.cpp backend | ✅ `surya_ocr2` — torch-free CPU-reference OCR pipeline end to end, every stage oracle-gated; OCR output == torch greedy reference ([plan](docs/model-cards/MODEL-SURYA.md); GPU kernels open) |
| tencent/EVIE-4.5B / 8B | ColQwen3_5 | ❌ (no multi-vector/MaxSim) | ✅ `evie_4p5b` (both sizes; 4.5B: batched fp16 encode 1.13 s, 1.11x torch bf16; 8B: fp32 parity-gated, fp32 recommended) |
| microsoft/VibeVoice-ASR | VibeVoiceForASRTraining | ❌ (custom audio tokenizers) | ✅ `vibevoice_asr` — `LLM.transcribe` on the HF checkpoint; production qualification incomplete ([plan](docs/model-cards/MODEL-VIBEVOICE-ASR.md)) |
| microsoft/VibeVoice-1.5B | VibeVoiceForConditionalGeneration | ❌ (TTS diffusion head) | ❌ (torch oracle frozen; [plan](docs/model-cards/MODEL-VIBEVOICE-TTS.md)) |
| shisa-ai/shisa-asr-v0.95b (+FP8) | Phi4MMForCausalLM | ❌ (no phi4mm audio) | ❌ |
| opendatalab/MinerU2.5-2509-1.2B | Qwen2VL | ✅ qwen2vl + mtmd; community GGUF | ❌ ([review/plan](docs/model-cards/MODEL-MINERU.md)) |
| PaddlePaddle/PaddleOCR-VL | PaddleOCRVL | ✅ dedicated `paddleocr` arch + mtmd vision | ❌ |
| zai-org/GLM-OCR | GlmOcr | ✅ glm4/glm4v explicit GLM-OCR handling, incl. MTP head | ❌ ([review/plan](docs/model-cards/MODEL-GLM-OCR.md)) |
| google/medgemma-1.5-4b-it | Gemma3 | ✅ gemma3 + mmproj | ❌ |
| google/translategemma-4b-it | Gemma3 | ✅ gemma3 + mmproj | ❌ |
| BAAI/bge-m3 | XLM-RoBERTa | ✅ encoder embedding/pooling | ❌ |
| Qwen/Qwen3-Embedding-8B | Qwen3 | ✅ qwen3 `--embeddings` | ❌ |
| shisa-ai/chotto-e4b-20260515 | Gemma4 (not 3n) | ✅ `gemma4` + gemma4v mtmd | ❌ |
| Qwen/Qwen-Image-2.1 | `QwenImage21Pipeline`: single-stream DiT (32 layers) + Qwen3-VL text encoder + VAE, flow-match Euler | ❌ (image-gen DiT; `qwen_image` GGUF rejected, ComfyUI / stable-diffusion.cpp are the runtimes) | ❌ (added 2026-09-20; new capability class — diffusion image generation/editing) |

No llama.cpp model oracle (framework/community references exist; historical priority, easiest first): ~~EVIE (prefill-only encoder + MaxSim)~~ ✅ done → ~~shisa-asr (Phi4MM audio front end)~~ ✅ done → ~~VibeVoice-ASR~~ ✅ done → VibeVoice-TTS (diffusion head, hardest; torch oracle frozen, hipEngine port open).
llama.cpp-oracle-available, likely cheapest: Qwen3-Embedding-8B / bge-m3 (encoder, no decode loop) → medgemma/translategemma (Gemma3 plugin) → chotto-e4b (Gemma4 plugin) → OCR VLMs (MinerU → PaddleOCR-VL → GLM-OCR → surya vision).


## Machines

### zbook
- autocalc context based on memory available
- new model implementations

### gfx1151
- qwen38-27b final integration
- qwen38fn tuning
- prefix caching testing

### gfx1100
- TP=2 support



## Campaigns

### Prefix Caching
• Yes, we have benchmarked prefix caching, but that is separate from the MTP qualification in your handoff. We do not yet have evidence there of
  faster end-to-end coding-agent sessions.

  Caveat: the named campaign worktree is absent on this machine. This assessment uses the available ~/hipEngine checkout and your
  handoff, not the unpushed campaign implementation.

  What We Have

  - Explicit cache-off/radix agentic A/B: the July 21 W7900 campaign tested Qwen3.6-35B-A3B GGUF Q4_K_M/BF16-KV through a live server, using
    small-repository, medium-repository, and growing-history workloads. Correctness passed, but sparse useful hits and expensive miss/suffix
    handling caused performance regressions. C4/C8 promotion testing was consequently skipped. See the decision artifact (benchmarks/
    results/2026-07-21-w7900-agentic-a2-prefix-decision.json).

  - Actual defaults are off: both the server (hipengine/server/api.py:328) and generation loop (hipengine/generation/engine_loop.py:84). The
    statement that radix is default in docs/PLAN.md:1390 conflicts with these defaults and the recorded decision.

  - Your K1–K7 captures establish numerical properties, not repeated-request cache economics. Prefilling a context, retaining KV during decode,
    and warming compiled kernels are not evidence of cross-turn prefix-cache hits.

  The previous rejection is specific to that implementation/model/workload, not proof that caching cannot help Qwen3.8. The documented issue was
  that the retained snapshot boundary rarely matched the next request, while exact miss/tail processing cost more than the hits saved. Analysis
  (docs/campaigns/AGENTIC-OPT.md:243)

  How I’d Benchmark Real Agent Loads
  I would extend the existing live agentic protocol (docs/BENCHMARK.md:986), with two separate lanes:

  1. Reproducible transcript replay: freeze representative coding sessions, including system instructions, tool schemas, repository text,
     assistant turns, and tool results. Replay identical requests and arrival schedules across configurations. This isolates engine/cache
     performance.

  2. Actual closed-loop tasks: let the model choose tools, inspect files, edit, and run tests in isolated repositories. Score independently
     verified task completion and total elapsed time, including failed attempts and tool execution. This establishes usefulness, not just fast
     replay.

  The initial paired matrix should be:

   Configuration              Purpose
  ━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━
   True AR, cache off         Baseline
  ─────────────────────────  ───────────────────────────
   True AR, cache on          Isolate caching
  ─────────────────────────  ───────────────────────────
   Explicit MTP, cache off    Isolate speculation
  ─────────────────────────  ───────────────────────────
   Explicit MTP, cache on     Measure their interaction

  Keep MTP experimental until its binding gates pass; do not substitute verifier-derived K0 for true AR.

  For each configuration, I’d include:

  - Cold first turns, warm continuations, growing tool output, edited history, and context compaction.
  - Representative context lengths and both short tool calls and long code responses.
  - One agent first, then overlapping agents, staggered arrivals, and cache-pressure/eviction cases.
  - The same physical host, fixed memory budget, balanced repeated runs, and held-out sessions. Oracle requests must not accidentally prewarm
    the measured cache.

  The headline should be time to a correct completed task, supported by p50/p95 tool-ready latency, queue time, prefill time, decode time,
  reused versus executed prompt tokens, cache/snapshot bytes, and successful-task goodput. Report model-serving time separately from tool-
  inclusive wall time.

  My recommendation: add this as a distinct public-path workstream under #7, not as another teacher-capture gate. First establish whether
  representative next-turn requests actually hit useful cached state; then measure whether those hits reduce total turn latency. That is the
  missing bridge between the current kernel/MTP campaign and a credible coding-agent performance claim.

### New Models

**Concretely:** one consolidated table of every model we've landed on, ordered by how much new code hipEngine needs, with repo, license, type, size, and the specific work each requires.

#### Tier 0 — weight swaps, no engine work

If hipEngine already runs Gemma 3, **MedGemma 1.5** (4B multimodal; 27B text and multimodal still on MedGemma 1, HAI-DEF terms) and **TranslateGemma** (4B/12B/27B, Gemma 3 arch) are checkpoint loads. TranslateGemma's only friction is its non-standard `source_lang_code`/`target_lang_code` params, which is a template fix. I haven't verified their exact HF paths in this session — both are under `google/`.

#### Ranked port list

| # | Model | License | Type | Size |
|---|---|---|---|---|
| 1 | ✅ [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch) · [-transformers](https://huggingface.co/google/timesfm-2.5-200m-transformers) · [repo](https://github.com/google-research/timesfm) — **plus 3.0-500M done** (`timesfm_3p0`) | Apache-2.0 | Forecasting | 200M / 331M |
| 2 | ✅ [tencent/EVIE-4.5B](https://huggingface.co/tencent/EVIE-4.5B) · [8B](https://huggingface.co/tencent/EVIE-8B) · [repo](https://github.com/Tencent/EVIE) | Apache-2.0 | Retrieval encoder | 4.61B / 8.41B |
| 3 | [opendatalab/MinerU2.5-2509-1.2B](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) · [GGUF](https://huggingface.co/Mungert/MinerU2.5-2509-1.2B-GGUF) | AGPL-3.0 (checkpoint card) | OCR + layout | 1.2B |
| 4 | [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) · [1.6 LiteRT](https://huggingface.co/litert-community/PaddleOCR-VL-1.6) | Apache-2.0 | OCR + layout | 0.9B |
| 5 | [datalab-to/surya-ocr-2](https://huggingface.co/datalab-to/surya-ocr-2) · [GGUF](https://huggingface.co/datalab-to/surya-ocr-2-gguf) · [repo](https://github.com/datalab-to/surya) | AI Pubs OpenRAIL-M | OCR + layout + table | 650M |
| 6 | [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) · [repo](https://github.com/zai-org/GLM-OCR) | MIT | OCR + layout | 0.9B |
| 7 | [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) · [HF port](https://huggingface.co/microsoft/VibeVoice-ASR-HF) · [repo](https://github.com/microsoft/VibeVoice) | MIT | ASR | 9B |
| 8 | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) | MIT; card describes research-only intended use | TTS | 1.5B LM + audio modules |
| 9 | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) · [repo](https://github.com/QwenLM/Qwen-Image-2.1) · [blog](https://qwen.ai/blog?id=qwen-image-2.1) | Qwen Research License — non-commercial only; commercial use needs a separate Tongyi license | Image gen + editing (flow-match DiT) | 7B DiT + Qwen3-VL text encoder + VAE |

**Work required:**

1. ✅ **TimesFM 2.5 + 3.0** — done. 2.5: GPU decode 0.082 s (13.3x over first path). 3.0: non-autoregressive multivariate port, GPU decode 0.319 s (4.17x torch fp32); records in `docs/MODEL-TIMESFM{,3}.md`.
2. ✅ **EVIE-4.5B + 8B** — done. Batched fp16 encode 1.13 s for 8p+8q (1.11x torch bf16 with better retrieval fidelity); 8B merger-geometry fix landed with fp32 parity gates. Records in `docs/model-cards/MODEL-EVIE.md`.
3. **MinerU 2.5** — Qwen2-VL: 32-block, 1280-wide vision tower and
   24-layer, 896-wide Qwen2 decoder. Adapt shared vision primitives and add the
   Qwen2-VL model contract, then reproduce layout → native-resolution crops →
   recognition. Current client sampling differs across backends; GGUF alone
   does not give pipeline parity. See [MODEL-MINERU.md](docs/model-cards/MODEL-MINERU.md).

4. **PaddleOCR-VL** — ERNIE-4.5-0.3B decoder is trivial; NaViT variable-resolution patching and packing is the real work. The LiteRT port is a gift: it documents that the decoder is a standalone Llama-layout model, bit-exact fp32 vs original, and that 1-D positions substitute fine for M-RoPE on OCR with no quality loss. That's most of your bring-up validation done.

5. **Surya OCR 2 — next model implementation.** Qwen3.5 hybrid decoder
   (18 GDN + 6 full-attention layers) plus a 12-block vision tower. Reuse the
   existing Qwen3.5 generation path and adapt EVIE's vision encoder; remaining
   work includes weight/tokenizer validation, image preprocessing, image-feature
   injection, mRoPE continuation, and OCR output handling. The current full-page
   path uses the VLM alone; text-line detection and RF-DETR fast layout are
   separate optional capabilities. See [docs/model-cards/MODEL-SURYA.md](docs/model-cards/MODEL-SURYA.md)
   for the 2026-09-11 source review, pitfalls, and implementation gates.

6. **GLM-OCR** — CogViT-derived tower, convolutional downsampling connector,
   and four-norm GLM decoder. Start with supplied-crop recognition; the SDK's
   PP-DocLayoutV3 stage is separate. MTP is an optional later milestone. Resolve
   the reviewed Transformers/llama.cpp rotary discrepancy before using GGUF as
   a numerical oracle. See [MODEL-GLM-OCR.md](docs/model-cards/MODEL-GLM-OCR.md).
7. **VibeVoice-ASR** — two 24 kHz causal audio encoders feeding a Qwen2 decoder;
   no synthesis diffusion loop. The official ASR-HF artifact provides a complete
   processor/tokenizer reference. Acoustic latents are sampled at inference,
   so fixtures need common noise inputs. See
   [MODEL-VIBEVOICE-ASR.md](docs/model-cards/MODEL-VIBEVOICE-ASR.md).
8. **VibeVoice TTS** — Qwen2-controlled diffusion plus acoustic decoding and
   semantic audio feedback. Microsoft's original TTS inference entry is disabled;
   a pinned community loop is available for oracle qualification. The model card's
   disclosure/watermark statements do not establish that weights alone implement
   them. See [MODEL-VIBEVOICE-TTS.md](docs/model-cards/MODEL-VIBEVOICE-TTS.md).
9. **Qwen-Image-2.1** — text-to-image generation and image editing (up to 10
   reference images, masks/circle annotations, native RGBA transparency), released
   2026-09-20 under the non-commercial Qwen Research License. New capability class
   for hipEngine — no autoregressive decode at generation time. Components (from the
   checkpoint's `model_index.json`): `QwenImage21Transformer2DModel` (32-layer
   single-stream DiT, mixed-granularity attention, prefix KV cache reuse),
   `Qwen3VLForConditionalGeneration` text encoder (also consumes reference images;
   no separate vision tower), `AutoencoderKLQwenImage21` VAE,
   `FlowMatchEulerDiscreteScheduler`. Prompt/reference encoding is prefill-only
   through the Qwen3-VL encoder; the new machinery is the flow-match DiT sampling
   loop and VAE decode. Oracles: diffusers `QwenImage21Pipeline` (torch; currently
   needs `transformers>=5.17` + git diffusers) for parity, stable-diffusion.cpp as
   the C/C++ reference for Qwen-Image-family GGUF.

#### Supporting index stack (use as-is, don't port)

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) for hybrid dense + sparse + ColBERT in one pass over 8192 tokens; Qwen3-Embedding-8B for text and code. Both run fine through existing paths.
