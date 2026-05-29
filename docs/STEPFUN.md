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

Implication for hipEngine on RDNA3: NVFP4 is not the fastest first bring-up
path. gfx1100 has no native NVIDIA FP4/FP8 tensor-core path, so a retained
NVFP4 implementation would need a ModelOpt safetensors loader plus software
FP4/FP8 dequant/GEMV or load-time dequantization. Load-time dequantization would
balloon resident memory and is unlikely to fit a single W7900. Treat the NVFP4
cache as authoritative config/weight-name reference until we deliberately add a
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
  backend remains `hip_gfx1100` / `cpu_reference`.

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

### 7. Memory and serving constraints

Both local formats exceed a single W7900's resident weight capacity:

- GGUF Q3_K_L language weights: 95.46 GiB before KV/runtime overhead.
- HF NVFP4 repo size: 115.84 GiB before KV/runtime overhead.
- Public llama.cpp docs list about 7 GB runtime overhead and a minimum 120 GB
  unified memory/VRAM for local GGUF deployment.

For hipEngine performance work, this means one of the following must land before
a real W7900 claim:

- tensor/expert parallelism across enough GPUs;
- tiered/offloaded weights with an explicit performance target; or
- a smaller Step-compatible fixture/checkpoint for correctness-only bring-up.

Do not claim Step 3.7 throughput on W7900 until the exact hardware, format,
workload shape, command, and correctness gate are recorded per `docs/BENCHMARK.md`.

## Proposed implementation order

1. **Metadata-only tests (no GPU).** Add fixtures using the local GGUF header and
   cached HF `config.json` / `hf_quant_config.json`. Verify parser outputs layer
   counts, full/sliding pattern, heads, MoE parameters, tokens, and format facts.
2. **Step model plugin + tokenizer.** Register `step35` / `step3p7` text plugin,
   implement DeepSeek-V3 GGUF tokenization, and render the chat template for a
   simple text-only prompt.
3. **Split GGUF loader + tensor map.** Merge the three Step GGUF shards into one
   logical index and validate all required text tensors for layers 0-44.
4. **Q3_K correctness path.** Implement CPU dequant for Q3_K, then native Q3_K
   GEMV/prefill or a documented replacement-layout route. Gate with slice-level
   numeric tests.
5. **Text decode c=1.** Wire dense layer, full-attention layer, sliding-attention
   layer, router/top-k/shared-expert layer, final norm, and LM head. Validate
   next-token logits against llama.cpp or HF Transformers on a tiny prompt.
6. **Memory strategy.** Decide whether the first retained run is multi-GPU,
   tiered/offloaded, or correctness-only. Only then benchmark.
7. **NVFP4 track.** After GGUF text works, add ModelOpt safetensors inspection
   and a `modelopt_nvfp4` quant plugin if RDNA3 software FP4 is still desired.

## Open questions

- Is the intended first run text-only, or do we need the multimodal projector and
  vision encoder immediately?
- Is there a smaller Step 3.7/3.5 fixture available for RED/GREEN tests, or
  should we derive slice fixtures from the large local GGUF files?
- What hardware target should define the first performance milestone: single
  W7900 with offload, multiple W7900s, or a high-memory unified-memory box?
- Should MTP be part of initial parity, or deferred until base greedy decode is
  correct?
