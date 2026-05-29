# StepFun Step 3.7 Flash bring-up notes

> Status: planning note, 2026-05-29. No hipEngine performance or correctness
> claim is made here. Public StepFun / Hugging Face text and code are treated as
> architecture references only; do not execute remote `trust_remote_code` in the
> hipEngine runtime.

## Sources and local assets reviewed

- Blog: <https://static.stepfun.com/blog/step-3.7-flash/>
- Hugging Face repo: `stepfun-ai/Step-3.7-Flash-NVFP4`, snapshot
  `36afbf6e15100cdc2d7a5b79d7e95d276ed33679`.
- HF cache snapshot present at
  `~/.cache/huggingface/hub/models--stepfun-ai--Step-3.7-Flash-NVFP4/snapshots/36afbf6e15100cdc2d7a5b79d7e95d276ed33679`.
  The cache currently has metadata/code/tokenizer files (`config.json`,
  `configuration_step3p7.py`, `modeling_step3p7.py`, `hf_quant_config.json`,
  `README.md`, `chat_template.jinja`, `generation_config.json`,
  `model.safetensors.index.json`, etc.) and all 13 safetensors shards resolved
  in the snapshot; no `.incomplete` blobs were present when checked on
  2026-05-29. Resolved shard bytes sum to 124,385,256,328 bytes; the index
  metadata's `total_size=124,385,012,840` appears to count tensor payload bytes.
- GGUF language-model shards found under `/data/models/gguf/` (the shorter
  `/data/gguf/` path was not present on this machine):
  - `Step-3.7-flash-Q3_K_L-00001-of-00003.gguf` — 46,544,161,344 bytes
  - `Step-3.7-flash-Q3_K_L-00002-of-00003.gguf` — 46,401,560,832 bytes
  - `Step-3.7-flash-Q3_K_L-00003-of-00003.gguf` — 9,558,706,368 bytes
  - total: 102,504,428,544 bytes (102.50 GB / 95.46 GiB)
- No local Step 3.7 GGUF multimodal projector file was found. Text-only GGUF
  bring-up should not depend on vision assets.

## Public model profile

The model card describes Step 3.7 Flash as a sparse MoE vision-language model:
196B language parameters plus a 1.8B vision encoder, about 11B active parameters
per token, and a 256k context window. The HF NVFP4 index reports
103,810,330,432 stored parameters and `total_size=124,385,012,840` bytes
(124.39 GB / 115.84 GiB tensor payload) across 13 safetensors shards; the local
resolved files sum to 124,385,256,328 bytes including container/header overhead.

Text config facts from `config.json` / `configuration_step3p7.py`:

| Field | Value / implication |
| --- | --- |
| Top-level architecture | `Step3p7ForConditionalGeneration`, `model_type=step3p7` |
| Text architecture | `Step3p5ForCausalLM`, `model_type=step3p5` |
| Hidden size | 4096 |
| Layers | 45 decoder layers |
| Context | 262,144 tokens |
| Vocab | 128,896 tokens |
| Head dim | 128 |
| Attention pattern | `full_attention`, then three `sliding_attention` layers, repeating; layer 44 is full |
| Full attention | 64 query heads, 8 KV heads, partial RoPE factor 0.5 |
| Sliding attention | 96 query heads, 8 KV heads, sliding window 512, full RoPE factor 1.0 |
| RoPE | full layers use theta 5,000,000 with llama3 scaling; sliding layers use theta 10,000 without llama3 scaling |
| Dense layers | layers 0-2 use dense MLP, intermediate 11,264 |
| MoE layers | layers 3-44 use MoE, 288 experts, top-k 8, expert intermediate 1,280, shared expert dim 1,280 |
| Router | sigmoid routing, router bias, routing scale 3.0, normalized expert weights, FP32 gate requested |
| Attention gate | `use_head_wise_attn_gate=true`; per-head `g_proj` gates attention output with sigmoid |
| Norm | RMSNorm with epsilon `1e-5`; HF reference applies `(weight + 1)` as the scale |
| MTP/speculation | config advertises `num_nextn_predict_layers=3`; public serving examples enable speculative MTP/EAGLE, but the current NVFP4 safetensors index only contains decoder layers 0-44 |
| Vision | separate 47-layer `perception_encoder` width 1536, patch size 14, image size 728, projector to hidden 4096 |

Tokenizer / chat facts:

- BOS token id 0: `<｜begin▁of▁sentence｜>`.
- EOS ids in generation config: `[1, 2, 128007]`; `special_tokens_map.json`
  names `<|im_end|>` as the EOS token and `<｜end▁of▁sentence｜>` as pad.
- Chat template uses `<|im_start|>role\n...<|im_end|>` blocks, optional
  `Reasoning: low|medium|high`, `<think>...</think>` assistant reasoning, and
  XML-like `<tool_call>` / `<tool_response>` blocks.
- GGUF metadata reports `tokenizer.ggml.model='gpt2'` and
  `tokenizer.ggml.pre='deepseek-v3'`, not the existing hipEngine Qwen3.5 GGUF
  tokenizer pre-tokenizer key.

## Public local-serving hints

The model card lists vLLM, SGLang, Transformers, and llama.cpp support. These
examples are useful runtime references, not hipEngine design mandates:

- vLLM FP8/BF16 examples use tensor parallel size 8, expert parallelism,
  `--disable-cascade-attn`, and the `step3p5` reasoning parser.
- vLLM NVFP4 example uses tensor parallel size 4, expert parallelism,
  `--quantization modelopt`, `--kv-cache-dtype fp8`, and `--max-model-len 8192`.
- SGLang NVFP4 example uses `--tp 4 --ep 4`, `--quantization modelopt_fp4`, and
  `--kv-cache-dtype fp8_e4m3`.
- Transformers is presented as a debug/verification path and requires
  Transformers 5.0 or later.
- llama.cpp deployment notes list Q3_K_L language weights at 102.5 GB,
  multimodal projector FP16 at 3.97 GB, about 7 GB runtime overhead, minimum
  120 GB unified memory/VRAM, and 128 GB unified memory recommended.

## Weight formats observed

### GGUF Q3_K_L

The local GGUF header reports:

- `general.architecture='step35'`
- `general.name='Step-3.7'`
- `general.size_label='288x7.4B'`
- `general.file_type=13`, i.e. `MOSTLY_Q3_K_L`
- `split.count=3`, `split.tensors.count=754`
- layer metadata under the `step35.*` prefix, including per-layer attention
  head-count arrays and a boolean sliding-window pattern.

Tensor layout is close to llama.cpp's Step 3.5/3.7 support, not hipEngine's
current `qwen35moe` GGUF layout:

- root tensors: `token_embd.weight`, `output.weight`, `output_norm.weight`,
  `rope_freqs.weight`.
- dense layers 0-2: attention tensors plus `ffn_gate/up/down.weight`.
- MoE layers 3-44: attention tensors plus `ffn_gate_inp.weight`,
  `exp_probs_b.bias`, expert `ffn_{gate,up,down}_exps.weight`, and shared
  expert `ffn_{gate,up,down}_shexp.weight`.
- observed quant mix in shard 1: embeddings and output are `Q8_0`; many gate,
  up, query, key, and attention-gate matrices are `Q3_K`; value, output, and
  down projections are often `Q5_K`; norms and router bias are `F32`.

### HF ModelOpt NVFP4

`config.json` and `hf_quant_config.json` declare:

- `quant_method='modelopt'`, `quant_algo='NVFP4'`
- Linear target group uses 4-bit floating-point weights and 4-bit floating-point
  input activations, group size 16.
- KV cache scheme is 8-bit float; model-card vLLM/SGLang examples require
  `--kv-cache-dtype fp8` / `fp8_e4m3` for NVFP4.
- Ignore list keeps `lm_head`, layers 0-2, every layer's attention, router gate,
  shared expert, vision model, and projector out of the NVFP4 target set. The
  13-shard index shows dense MLP tensors for layers 0-2, MoE expert tensors for
  layers 3-44, attention tensors for every layer, shared-expert tensors for MoE
  layers, vision tensors, and the projector.

Implication for hipEngine on AMD gfx1100/gfx1151: NVFP4 is not the fastest
first bring-up path. Strix Halo/gfx1151 also has no NVIDIA FP4/FP8 tensor-core
path, so a retained NVFP4 implementation would need a ModelOpt safetensors
loader plus software FP4/FP8 dequant/GEMV or load-time dequantization. Load-time
dequantization would balloon resident memory. Treat the NVFP4 cache as
authoritative config/weight-name reference until we deliberately add a
`modelopt_nvfp4` quant plugin.

## hipEngine gap analysis

The recommended first target is **text-only GGUF Q3_K_L**, because it is already
local and closer to hipEngine's existing GGUF/Qwen MoE work than ModelOpt NVFP4.
It still needs several first-class plugin additions; do not special-case StepFun
inside Qwen3.5 dispatch.

### 1. Model plugin

Current hipEngine model registrations cover `qwen35`, `qwen35moe`, and
PARO-Qwen names. Step GGUF uses `general.architecture='step35'`; HF uses
`model_type='step3p7'` with text `model_type='step3p5'`.

Needed:

- Add a `step3_7_flash` / `step35` model plugin that owns the layer sequence,
  weight-name map, chat template, special tokens, per-layer attention metadata,
  and optional vision/speculative capabilities.
- Keep the plugin text-first initially. Vision encoder/projector and MTP can be
  advertised as unsupported capabilities until text generation is correct.
- Preserve plugin axes: model code should select `layer='full_attention'` or
  `layer='sliding_attention'`; quant remains `gguf_q3_k_l` / `modelopt_nvfp4`;
  backend remains `hip_gfx1100` / `hip_gfx1151` / `cpu_reference`.

### 2. GGUF split loader and Step tensor map

Current `loading/gguf.py` scans one GGUF file, and `loading/qwen35_gguf.py`
rejects architectures outside `{'qwen35', 'qwen35moe'}`. The local Step model is
split across three files and reports `step35.*` metadata.

Needed:

- Add a split-aware GGUF index/reader that merges tensor tables across the three
  shards while retaining each tensor's source file and data offset.
- Add `loading/stepfun_gguf.py` (or similarly named) with:
  - config parser for `step35.*` metadata;
  - per-layer attention head counts and sliding-window pattern;
  - dense-vs-MoE layer map (`leading_dense_block_count=3` in GGUF);
  - tensor slot validation for Step tensor names and shapes;
  - clear errors when the multimodal projector is requested but absent.
- Keep this separate from `qwen35_gguf.py`; the attention pattern is sliding
  window, not Qwen3.5 linear/GDN attention.

### 3. Tokenization and chat template

Current `Qwen35GGUFTokenizer` requires `tokenizer.ggml.pre == 'qwen35'`; Step
GGUF uses `deepseek-v3`.

Needed:

- Add a torch-free Step/DeepSeek-V3 BPE pre-tokenizer path using GGUF tokens and
  merges.
- Load and apply `tokenizer.chat_template` / `chat_template.jinja` without
  importing Transformers on the runtime hot path.
- Preserve the model's multi-EOS behavior (`1`, `2`, `128007`) and assistant
  generation prefix (`<|im_start|>assistant\n<think>\n`).

### 4. Quant kernels and CPU references

Existing native GGUF support includes Q4_K/Q5_K/Q6_K/Q8_0 families, but Step
Q3_K_L uses Q3_K heavily. CPU dequant helpers also do not currently implement
Q3_K dequantization.

Needed for GGUF Q3_K_L:

- CPU-reference Q3_K dequantization for fixture/oracle work.
- HIP `Q3_K` GEMV/prefill kernels, or a lossless replacement layout that maps
  Step Q3_K tensors into an existing kernel family.
- Mixed-quant linear dispatch: per tensor, Step may use Q3_K, Q5_K, Q8_0, and
  F32 in the same layer. The quant plugin should expose this without engine-wide
  `if quant == ...` branches.
- Correctness fixtures against CPU reference or llama.cpp/Transformers for at
  least tiny tensor slices before running the full model.

Needed for NVFP4 later:

- Safetensors index/metadata loader for ModelOpt packed tensors and scale
  sidecars (`input_scale`, `weight_scale`, `weight_scale_2`).
- `modelopt_nvfp4` quant plugin with explicit RDNA3 fallback semantics.
- FP8 KV policy/kernels before matching the public NVFP4 serving setup.

### 5. Attention kernels

Step attention is full/sliding GQA with head-wise gating, not Qwen3.5 linear
attention. Existing full-attention paged decode kernels are a useful starting
point but need shape/generalization work.

Needed:

- Full attention decode/prefill for 64 query heads, 8 KV heads, head dim 128,
  partial RoPE factor 0.5, theta 5e6, llama3 scaling.
- Sliding attention decode/prefill for 96 query heads, 8 KV heads, head dim 128,
  window 512, theta 1e4, no llama3 scaling.
- KV policy integration where sliding layers expose only the live window through
  `KVLiveSpans` while full layers use the full live prefix.
- Head-wise attention gate kernel/fusion: `attn_output[head] *= sigmoid(g_proj(x)[head])`
  before `o_proj`.
- Per-layer RoPE tables/cache keyed by theta, partial factor, and scaling mode.

### 6. MoE and dense MLP

Step uses dense MLP only for layers 0-2 and MoE for layers 3-44. Existing Qwen
MoE kernels are relevant, but Step's top-k, router-bias, tensor names, and GGUF
quant mix need an explicit Step layer plugin/map.

Needed:

- Router: FP32 matmul, sigmoid probabilities, add router bias for top-k
  selection, gather unbiased probabilities, normalize selected weights, then
  multiply by router scale 3.0.
- Top-k 8 selected expert path over 288 experts; batch paths should group by
  expert later, but c=1 decode can start with selected lanes.
- Shared expert path and sum with routed MoE output.
- SwiGLU clamp handling for the last layers where `swiglu_limits` /
  `swiglu_limits_shared` are non-zero.
- Dense layers 0-2 use the same attention block but dense `gate/up/down` MLP,
  not routed experts.

### 7. Strix Halo memory and serving target

**Active target for this branch:** text-only Step 3.7 Flash GGUF Q3_K_L on a
high-memory Strix Halo / gfx1151 machine. This makes the GGUF path the right
first implementation track, but full-model load is still a measured precondition,
not an assumption:

- GGUF Q3_K_L language weights: 95.46 GiB before KV/runtime overhead.
- HF NVFP4 repo size: 115.84 GiB before KV/runtime overhead; this is not the
  first runtime target.
- Public llama.cpp docs list about 7 GB runtime overhead and a minimum 120 GB
  unified memory/VRAM for local GGUF deployment, with 128 GB recommended.
- A Strix Halo configured with 100 GB+ visible UMA may be close enough for a
  short text-only smoke, but the first retained full-model run must record exact
  HIP-visible total/free memory, UMA configuration, context length, KV dtype,
  prompt length, and generated-token count.
- `backend='auto'` should resolve Strix Halo to `hip_gfx1151`; if it does not,
  force `HIPENGINE_BACKEND=hip_gfx1151` only after recording `amdgpu-arch` /
  `rocminfo` evidence and validating correctness.
- hipEngine currently has many Qwen GGUF runner imports hard-coded to
  `hip_gfx1100`; Step bring-up should register/resolve kernels for
  `hip_gfx1151` as a peer backend instead of adding model/runtime backend
  branches.

A Strix Halo full-model smoke is acceptable only after parser/tokenizer/quant
slice tests pass. A throughput claim additionally needs the benchmark artifact
and rollup updates required by `docs/BENCHMARK.md`.

## Multiloop-ready GGUF punchlist

Use this as the canonical Step 3.7 Flash GGUF bring-up backlog. Each checkbox is
intended to be a logical unit that can be implemented, validated, logged in
`WORKLOG.md`, and committed before moving on. Keep the first implementation
text-only; defer vision, MTP/speculative decode, and NVFP4 until base greedy
decode is correct.

### P0 — Hardware, assets, and oracle preflight

- [ ] Confirm the target machine reports Strix Halo/gfx1151:
  `python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"`,
  `amdgpu-arch` or `/opt/rocm/bin/amdgpu-arch`, and
  `rocminfo | grep -E 'Name:|gfx'`.
- [ ] Record HIP-visible total/free memory after a clean boot and after loading
  the GGUF shards. If full-model load fails, keep the failure as evidence and
  fall back to slice/layer correctness until offload/tiering exists.
- [ ] Record exact GGUF paths and byte sizes for all three shards; do not copy or
  rewrite the 102.50 GB assets into the repo.
- [ ] Establish a llama.cpp oracle command for tokenization and short greedy
  next-token checks. If llama.cpp cannot run the full model on the same machine,
  use it for metadata/tokenizer/slice or a smaller exported fixture.
- [ ] For any kernel port/tuning, read `docs/KERNELS.md` and run
  `python3 scripts/check_lineage.py --kind kernel --diff stat` before copying
  code.

**Acceptance:** WORKLOG preflight entry with hardware, memory, paths, and oracle
plan; no runtime correctness or performance claim yet.

### P1 — Metadata-only parser fixtures

- [x] Add tests that read only GGUF headers/KV metadata from the three local
  shards and cached HF `config.json` / `hf_quant_config.json`.
- [x] Verify `step35` architecture, split count 3, tensor count 754, vocab
  128,896, 45 layers, context 262,144, dense layers 0-2, MoE layers 3-44,
  288 experts, top-k 8, full/sliding attention pattern, head counts, RoPE modes,
  and tokenizer pre `deepseek-v3`.
- [x] Ensure metadata tests do not mmap/read full tensor payloads.

**Acceptance:** `python3 -m pytest -q tests/test_stepfun_metadata.py` passes on
this no-GPU metadata path; tests skip cleanly if external Step assets are absent.

### P2 — Split GGUF index and Step tensor map

- [x] Add a split-aware GGUF reader/index that merges shard tensor tables while
  retaining each tensor's source path, offset, type, shape, and byte span.
- [x] Add a Step-specific loader/config module for `step35.*` metadata and tensor
  naming; keep it separate from `qwen35_gguf.py`.
- [x] Validate all required text tensors for layers 0-44 and produce a clear
  unsupported error for missing multimodal projector assets.
- [x] Add tests for split discovery, duplicate/missing tensor errors, and tensor
  shape/type validation.

**Acceptance:** `python3 -m pytest -q tests/test_stepfun_gguf_loader.py` builds a
logical Step text weight index from the three GGUF shards without loading all
weights into memory; tests skip cleanly if external Step assets are absent.

### P3 — Model plugin and capability registration

- [x] Register a text-first `step35` / `step3p7` model plugin with aliases for
  GGUF and HF metadata names.
- [x] Encode capabilities explicitly: text decode supported-in-progress; vision,
  projector, MTP, and NVFP4 unsupported/deferred until their tracks land.
- [x] Keep dispatch on the existing axes `(backend, layer, quant, variant)`;
  avoid engine/model branches such as `if backend == ...` or `if quant == ...`.
- [x] Add registry tests showing Step resolves to the plugin and unsupported
  capability requests fail clearly.

**Acceptance:** `python3 -m pytest -q tests/test_stepfun_model_plugin.py` passes;
`LLM(..., quant='gguf_q3_k_l')` resolves split Step metadata without importing
torch.

### P4 — DeepSeek-V3 GGUF tokenizer and chat template

- [ ] Implement a torch-free tokenizer path for GGUF `tokenizer.ggml.model='gpt2'`
  with `tokenizer.ggml.pre='deepseek-v3'`.
- [ ] Render the Step chat template locally, including `<|im_start|>`,
  `<|im_end|>`, optional `Reasoning: low|medium|high`, assistant
  `<think>` prefix, and tool-call blocks.
- [ ] Preserve BOS id 0 and EOS ids `[1, 2, 128007]` in generation stop logic.
- [ ] Compare token IDs for representative prompts against llama.cpp or cached
  HF tokenizer output.

**Acceptance:** tokenizer/chat tests pass and runtime hot-path imports remain
torch-free.

### P5 — Q3_K CPU reference and mixed GGUF quant metadata

- [ ] Implement CPU-reference Q3_K dequantization and add block-level fixtures.
- [ ] Validate existing Q5_K/Q8_0/F32 handling against Step tensor metadata; Step
  layers mix quant types within the same layer.
- [ ] Add slice fixtures from real Step tensors for Q3_K, Q5_K, Q8_0, and F32
  without checking large binary fixtures into git.
- [ ] Expose per-tensor quant keys through the loader/quant plugin so mixed
  dispatch does not require engine-wide quant branches.

**Acceptance:** CPU dequant/slice tests pass against known-good llama.cpp or
independent GGUF reference values.

### P6 — HIP Q3_K linear kernels on gfx1151

- [ ] Add/register HIP Q3_K GEMV and selected-expert variants needed by Step
  dense, attention, MoE expert, and shared-expert paths.
- [ ] Build for `hip_gfx1151` with `HIPENGINE_HIP_ARCH=gfx1151`; reuse gfx1100
  source only through peer backend registration/build metadata, not hard-coded
  imports in Step runtime code.
- [ ] Add smoke tests comparing HIP Q3_K outputs to CPU reference for small
  tensors, then real Step tensor slices.
- [ ] Run a `rocprofv3 --kernel-trace` smoke once kernels exist and record the
  expected kernel names/durations.

**Acceptance:** HIP Q3_K slice correctness passes on Strix Halo and profiler
shows the expected kernels, with no full-model claim yet.

### P7 — Step norms, RoPE, and attention-gate primitives

- [ ] Add RMSNorm variant with Step scale semantics `(weight + 1)` and epsilon
  `1e-5`.
- [ ] Add RoPE table/cache support for full-attention layers: theta 5e6,
  llama3 scaling, partial factor 0.5.
- [ ] Add RoPE support for sliding-attention layers: theta 1e4, no llama3
  scaling, full factor 1.0.
- [ ] Add head-wise attention gate primitive/fusion:
  `attn_output[head] *= sigmoid(g_proj(x)[head])` before `o_proj`.
- [ ] Keep unfused CPU/reference fallbacks for every fused primitive.

**Acceptance:** primitive tests pass against CPU reference for representative
full and sliding layers.

### P8 — Full and sliding GQA attention

- [ ] Implement full-attention decode/prefill for 64 query heads, 8 KV heads,
  head dim 128, and partial-RoPE full layers.
- [ ] Implement sliding-attention decode/prefill for 96 query heads, 8 KV heads,
  head dim 128, and window 512.
- [ ] Represent both policies through `KVLiveSpans`: full layers expose the live
  prefix, sliding layers expose only the live window.
- [ ] Validate one-token decode and short prefill against CPU attention fixtures.
- [ ] Only after correctness, profile whether AOTriton or native kernels are the
  right Strix Halo path.

**Acceptance:** full/sliding attention tests pass for both layer types and KV
window boundaries.

### P9 — Dense MLP and Step MoE

- [ ] Wire dense MLP for layers 0-2 with `ffn_gate/up/down.weight` names.
- [ ] Implement router semantics for MoE layers 3-44: FP32 gate matmul, sigmoid,
  add router bias for top-k selection, gather unbiased probabilities, normalize
  selected weights, then multiply by routing scale 3.0.
- [ ] Implement top-k 8 over 288 experts, expert gate/up/down projections,
  shared expert gate/up/down path, and routed+shared sum.
- [ ] Handle non-zero `swiglu_limits` / `swiglu_limits_shared` in the last layers.
- [ ] Start with c=1 decode; batch/expert grouping optimization can follow after
  correctness.

**Acceptance:** router/top-k/shared-expert tests pass against CPU reference or a
llama.cpp/HF activation fixture.

### P10 — One-layer and block replay

- [ ] Build a deterministic replay harness for a dense layer, a full-attention
  MoE layer, and a sliding-attention MoE layer.
- [ ] Capture or derive reference activations/logits from llama.cpp, HF
  Transformers, or CPU-reference code without committing large blobs.
- [ ] Gate each block on numerical tolerances appropriate for quantized GGUF
  inference before integrating all 45 layers.

**Acceptance:** representative dense/full/sliding block replays pass and failures
identify the exact substage.

### P11 — Text-only c=1 decode runner

- [ ] Add a Step GGUF runner that streams one-token decode for short prompts with
  the Step tokenizer, split weight index, mixed GGUF quant dispatch, full/sliding
  attention, and Step MoE.
- [ ] Use short contexts first (for example <= 512) before exercising long
  context and sliding-window boundaries.
- [ ] Compare greedy next tokens and/or logits against llama.cpp for a small set
  of deterministic prompts.
- [ ] Preserve multi-EOS stopping and the chat assistant prefix.

**Acceptance:** deterministic short-prompt next-token parity is demonstrated;
record command, prompt shape, oracle, and result in `WORKLOG.md`.

### P12 — Full-model Strix Halo smoke

- [ ] Load all three GGUF shards on the Strix Halo target with a small context
  and `max_new_tokens` (for example 1-8).
- [ ] Record HIP-visible memory before load, after load, after KV allocation, and
  after generation; include UMA setting and backend (`hip_gfx1151`).
- [ ] If the model does not fit, keep the failure artifact and decide between
  offload/tiering, lower context/KV footprint, or slice-only correctness.
- [ ] If it fits, run a tiny text-only prompt and confirm no vision/projector/MTP
  path is required.

**Acceptance:** full-model smoke produces token(s) or a documented fit failure.
This is still not a throughput benchmark.

### P13 — Benchmark and rollup only after correctness

- [ ] Define the exact benchmark shape: model format, prompt length, generated
  tokens, context/KV policy, hardware, ROCm version, backend, and command.
- [ ] Run correctness gate first: KL <= 0.05 and top-1 agreement >= 90% vs the
  configured CPU/llama.cpp oracle on fixture inputs.
- [ ] Record benchmark artifact under `benchmarks/results/`, update
  `benchmarks/README.md`, and add a dated `benchmarks/CHANGELOG.md` entry for
  any retained performance result.

**Acceptance:** evidence policy in `docs/BENCHMARK.md` is satisfied before any
Step throughput claim appears in docs or chat.

### Deferred tracks

- [ ] NVFP4 ModelOpt safetensors loader and `modelopt_nvfp4` quant plugin.
- [ ] Vision encoder and multimodal projector loading/inference.
- [ ] MTP/EAGLE speculative decode and `num_nextn_predict_layers=3` support.
- [ ] Multi-user/server batching and expert grouping optimizations.

## Suggested multiloop lanes

These are starting points if we launch pi-multiloop later. Do not start a loop
until the setup guide has scanned the repo, asked clarifying questions, and the
user has explicitly approved the run.

| Lane | Goal | Verify command |
| --- | --- | --- |
| `step-metadata` | Header/config parser, split index metadata, tensor map validation | `python -m pytest -q tests/test_stepfun_gguf_metadata.py` |
| `step-tokenizer` | DeepSeek-V3 GGUF tokenizer and chat rendering | `python -m pytest -q tests/test_stepfun_tokenizer.py` |
| `step-q3k-cpu` | CPU Q3_K dequant and mixed-quant slice fixtures | `python -m pytest -q tests/test_stepfun_q3k_cpu.py` |
| `step-q3k-hip` | gfx1151 Q3_K HIP GEMV slice correctness | `HIPENGINE_BACKEND=hip_gfx1151 python -m pytest -q tests/test_stepfun_q3k_hip.py` |
| `step-attn` | Full/sliding GQA + KVLiveSpans + RoPE/gate primitives | `python -m pytest -q tests/test_stepfun_attention.py` |
| `step-moe` | Dense MLP, router/top-k, experts, shared expert | `python -m pytest -q tests/test_stepfun_moe.py` |
| `step-decode` | Text-only c=1 short-prompt next-token parity | `HIPENGINE_BACKEND=hip_gfx1151 python -m pytest -q tests/test_stepfun_decode.py` |
| `step-smoke` | Full GGUF load/generate on Strix Halo | command TBD after P11; record memory and oracle in `WORKLOG.md` |

## Open risks and mitigations

| Risk | Why it matters | Mitigation / decision gate |
| --- | --- | --- |
| Strix Halo UMA may be too tight for full GGUF + KV + runtime overhead | GGUF weights are 95.46 GiB and public llama.cpp notes recommend 128 GB UMA | P0/P12 must record HIP-visible free memory and fit/fail evidence before full-model claims; keep slice correctness unblocked if full load fails |
| gfx1151 kernel coverage may lag gfx1100 | Existing hipEngine code and lineage are gfx1100-centered | Treat `hip_gfx1151` as a peer backend registration/build target; run minimal kernel smoke and `rocprofv3` before relying on copied gfx1100 kernels |
| Q3_K is the dominant missing quant path | Step Q3_K_L heavily uses Q3_K and hipEngine's existing GGUF work is Q4_K/Q5_K/Q6_K/Q8_0 focused | Land CPU Q3_K first, then HIP Q3_K slice correctness; do not integrate decode through lossy ad-hoc dequant shortcuts |
| Tokenizer/chat mismatch can invalidate every oracle comparison | Step GGUF uses DeepSeek-V3 GPT-2 BPE metadata, not Qwen3.5 tokenizer metadata | Make tokenizer parity a separate lane with llama.cpp/HF token-id fixtures before logits/decode work |
| Sliding/full mixed attention can hide KV bugs until long prompts | Step alternates full and 512-window sliding layers with different head counts and RoPE settings | Exercise both layer types through `KVLiveSpans` boundary tests before full-model smoke |
| llama.cpp/HF oracle may be unavailable for full-model comparisons on the same box | The model is large and HF Transformers may require more memory than GGUF | Use llama.cpp for tokenization/short greedy where possible and derive small activation/tensor-slice fixtures for earlier lanes |
| NVFP4 completion may distract from GGUF bring-up | NVFP4 files are now cached but require ModelOpt FP4/FP8 support with unclear RDNA3/RDNA3.5 upside | Keep NVFP4 explicitly deferred until P11/P12 text-only GGUF correctness is established |

## Remaining open questions

- What exact Strix Halo UMA size and HIP-visible free memory do we have after a
  clean boot?
- Is there a smaller Step 3.7/3.5 fixture available for RED/GREEN tests, or
  should we derive slice fixtures from the large local GGUF files?
- Which llama.cpp command and commit should be the initial oracle for tokenizer,
  next-token, and full-smoke comparison?
- Are vision/projector and MTP definitely deferred until after base greedy decode
  correctness, or do they need separate early proof-of-concept loops?
