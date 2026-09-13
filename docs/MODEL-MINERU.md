# MODEL-MINERU.md — MinerU 2.5 on hipEngine

Status: **reviewed; implementation pending** (2026-09-11).

Target `opendatalab/MinerU2.5-2509-1.2B` is a Qwen2-VL model with a large
vision encoder and small causal text decoder. Its document parser applies the
same VLM first to page layout and then to high-resolution crops. The initial
hipEngine target should be a correct image/crop recognition model, followed by
the two-stage application pipeline. It is neither Surya's Qwen3.5 tower nor a
pair of independently served layout and OCR language models. [Card][card]

## Geometry and model semantics

| Component | Pinned checkpoint/configuration |
| --- | --- |
| Architecture | `Qwen2VLForConditionalGeneration`, `model_type=qwen2_vl` |
| Text width / layers / FFN | 896 / 24 / 4864; SiLU gated MLP |
| Attention | 14 query / 2 KV heads, head dim 64; full causal GQA, no GDN |
| Text normalization / RoPE | RMSNorm eps 1e-6; theta 1,000,000; mRoPE sections `[8,12,12]` |
| Vocabulary / head / context | 151936 / tied embeddings / 16384; sliding window disabled |
| Vision | 32 blocks, width 1280, 16 heads × 80, MLP ratio 4 |
| Patch geometry | Spatial 14, temporal 2, spatial merge 2 |
| Merger | Per-patch LayerNorm → 2×2 concat → 5120→5120→896, erf GELU |
| Vision block activation | Qwen2-VL default `quick_gelu`; not EVIE's tanh GELU |
| Vision position handling | 2D rotary; no Qwen3.5 learned position table |
| Image tokens | Start/end 151652/151653; image placeholder 151655 |
| Preprocessor | Qwen2VLImageProcessor; CLIP mean/std, min 3136 / max 1605632 pixels |

[Config][config], [processor][processor], [Qwen2-VL implementation][model],
[vision defaults][defaults]. Resolve omitted defaults with the pinned library
and inspect actual weight shapes; `vision_config.hidden_size=896` is the
connector output, while `embed_dim=1280` is the tower width.

For a still image at final dimensions H×W, patch rows are `H*W/196` and
merged image tokens `H*W/784` (H,W multiples of 28). The processor cap corresponds
to 2048 image tokens. Learned positions and patch-16 preprocessing from EVIE
must not be carried over. Preserve temporal duplication, patch packing,
per-image bidirectional masking, decoder mRoPE and cached continuation offsets.

## The two-stage application contract

```text
page render → layout-size image → VLM layout sequence
            → parse boxes/types/rotation/reading order
original-resolution page → crop/rotate regions → same VLM, task-specific prompts
                        → restore region order → assemble text/math/tables
```

The reviewed `mineru-vl-utils` client defaults to a **1036×1036 bicubic layout
resize**, then the model processor runs. Do not assume a paper's or an older
version's layout dimensions. Layout uses box/reference/rotation special tokens,
not Surya's JSON or HTML layout syntax. Content prompts differ for text,
formula, table and image analysis. Freeze client and checkpoint together;
current utilities also contain newer optional postprocessors that are not
necessarily required for this older checkpoint. [Client source][client]

Scheduling must preserve page/region identities when crops complete out of
order. Record which boxes were skipped, rejected or failed. Keep original-page
coordinates through resize, crop and rotation, and preserve reading order.
Full PDF parsing, cross-page tables and Markdown formatting can remain outside
the torch-free model runtime.

**Sampling is part of pipeline parity.** Current client defaults include a
100-token no-repeat n-gram constraint, with lower frequency penalty for tables
than general text. Its llama.cpp adapter explicitly lacks an equivalent
no-repeat-ngram parameter. Therefore identical prompts and GGUF weights do not
automatically make llama.cpp and vLLM task outputs directly comparable. Use
matched raw-generation settings for math comparisons, then qualify each declared
application protocol. Do not introduce table-specific exceptions or repetition
heuristics based on benchmark prompts. [Sampling client][client],
[llama.cpp adapter][llamaclient]

## Existing implementations and reuse

| Reference/piece | Role and limits |
| --- | --- |
| Transformers Qwen2-VL | Full-precision model/processor oracle; version-pin omitted defaults |
| Official `mineru-vl-utils` | Task prompts, two-step scheduling, parser and multiple backend adapters |
| Community Mungert GGUF + mmproj | Independent quantized model comparison; inspect both artifacts and conversion metadata |
| EVIE vision runtime/CPU reference | Patch GEMM, attention, LayerNorm, merger and fixture patterns; new patch size, head dim, activation and positions |
| Qwen4Exp multimodal code | Feature injection, image transport and mRoPE lifecycle examples; model-specific controls must change |
| Existing dense primitives | GEMM, RMSNorm, SiLU, GQA, KV storage; new Qwen2-VL spec/loader/dispatch route |

Useful paths: `hipengine/runtime/evie.py`,
`hipengine/kernels/cpu_reference/evie.py`,
`hipengine/generation/qwen4_exp_multimodal.py`,
`hipengine/server/multimodal.py` and `hipengine/models/qwen35.py` as a plugin
pattern, not a drop-in decoder. Validate Qwen2 QKV biases and ungated attention
output; do not retain Qwen3.5 attention gates or GDN scheduling.
[GGUF inventory][gguf], [utility repository][utils]

**License provenance:** the pinned 2509 checkpoint card declares **AGPL-3.0**.
The current MinerU application's `LICENSE.md` instead specifies Apache-2.0 plus
additional terms. These are distinct artifacts; the current application license
does not by itself establish relicensing of the older weights. Record licenses
for the exact checkpoint, utility package and any copied source. [Card][card],
[current application license][license]

## Suggested implementation sequence

1. Add proposed `hipengine/models/mineru.py` and loader contracts; inventory
   safetensors and GGUF/mmproj. Pin tokenizer, template, processor and client.
   Add `scripts/mineru_oracle_torch.py` for processed patches, visual features,
   decoder positions, logits and cached steps.
2. Prove the small Qwen2 text decoder, then adapt shared vision primitives.
   Add quick-GELU and shape-specific work only with CPU fixtures and registry
   fallbacks. Test rectangular grids and head dim 80, not only square toy pages.
3. Connect model-only image/text generation. Validate crop text, equations and
   tables independently before adding predicted layout errors to the test.
4. Reproduce pinned layout parsing and crop preprocessing in an application
   adapter. Start serial two-step extraction, then introduce bounded concurrent
   crop scheduling with stable output order and cancellation cleanup.
5. Qualify the complete pipeline on real multi-page documents; then profile
   vision at layout/crop resolutions, batching and long table generation.

## Task and performance gates

Use separate fixtures for model math, fixed-oracle crops and predicted-layout
end-to-end parsing. This distinguishes recognition regressions from crop/reading
order regressions. Include English, Japanese/mixed text, rotated and scanned
pages, columns, formulas, spanning-cell tables, blank pages and held-outs.
Measure CER/WER, layout/box accuracy, reading order, formula edit distance,
table structure (e.g. TEDS), Markdown assembly errors and malformed/truncated
outputs; use a frozen OmniDocBench protocol for broader pipeline evaluation.

Record page render/DPI, layout resolution, crop count and dimensions, visual
and output token counts, both stage latencies, complete page latency, pages/s,
peak memory and failed/retried regions. Same-host baseline must match sampling
and parsing settings. Do not compare one crop against another engine's complete
page pipeline, or reduce crop resolution without a quality gate. The 32-layer
1280-wide vision stack may dominate despite the small text decoder; profile
before choosing decode-only optimization.

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

[card]: https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B/blob/1aa090b41282e64fadd79c10572221f91ec21924/README.md
[config]: https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B/blob/1aa090b41282e64fadd79c10572221f91ec21924/config.json
[processor]: https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B/blob/1aa090b41282e64fadd79c10572221f91ec21924/preprocessor_config.json
[model]: https://github.com/huggingface/transformers/blob/177e90dd2d51273fa235dd8bacee7c80f1eef067/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
[defaults]: https://github.com/huggingface/transformers/blob/177e90dd2d51273fa235dd8bacee7c80f1eef067/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py
[client]: https://github.com/opendatalab/mineru-vl-utils/blob/1e56064864559d6fc643f423a14e69053c06eee5/mineru_vl_utils/mineru_client.py
[llamaclient]: https://github.com/opendatalab/mineru-vl-utils/blob/1e56064864559d6fc643f423a14e69053c06eee5/mineru_vl_utils/vlm_client/llama_cpp_engine_client.py
[utils]: https://github.com/opendatalab/mineru-vl-utils/blob/1e56064864559d6fc643f423a14e69053c06eee5/README.md
[gguf]: https://huggingface.co/Mungert/MinerU2.5-2509-1.2B-GGUF/blob/9886338941b18153b5c5f22809edce055206b440/README.md
[license]: https://github.com/opendatalab/MinerU/blob/4fe4bde114a23ee5dd637eae99b767f4669bf58c/LICENSE.md
