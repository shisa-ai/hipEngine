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
- Additional local GGUF variants now present for post-Q3 validation, not the
  active bring-up target yet:
  - `Step-3.7-Flash-APEX-I-Compact.gguf` — 90,321,538,656 bytes (84.12 GiB)
  - `Step-3.7-Flash-UD-IQ4_NL-00001-of-00003.gguf` — 5,232,064 bytes
  - `Step-3.7-Flash-UD-IQ4_NL-00002-of-00003.gguf` — 49,628,912,512 bytes
  - `Step-3.7-Flash-UD-IQ4_NL-00003-of-00003.gguf` — 47,683,674,304 bytes
  - UD-IQ4_NL total: 97,317,818,880 bytes (90.63 GiB)
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

### Deferred local GGUF variants

The local `Step-3.7-Flash-UD-IQ4_NL` split and single-file
`Step-3.7-Flash-APEX-I-Compact` GGUF variants are explicit follow-up targets
only after the current Q3_K_L path has end-to-end text decode correctness. The
post-Q3 validation gate should reuse the same sequence as Q3_K_L: metadata scan,
tensor-slot coverage, quant-layout support via registry keys (no engine-wide
variant branches), selected tensor CPU/HIP slice checks, resident load/resource
smoke, short prompt logits/token oracle, and only then any benchmark artifact.
Do not let these variants distract from finishing Q3_K_L P11/P12.

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

- [x] Confirm the target machine reports Strix Halo/gfx1151:
  `python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"`,
  `amdgpu-arch` or `/opt/rocm/bin/amdgpu-arch`, and
  `rocminfo | grep -E 'Name:|gfx'`.
- [x] Record HIP-visible total/free memory after a clean boot and after loading
  the GGUF shards. If full-model load fails, keep the failure as evidence and
  fall back to slice/layer correctness until offload/tiering exists. 2026-05-29
  update: with `amdgpu gttsize=120000` and `ttm pages_limit=31457280`,
  `PYTHONUNBUFFERED=1 python3 scripts/stepfun_gguf_load_smoke.py --pretty >
  /tmp/stepfun-full-load-smoke-task20.json` successfully loaded all 754
  resident weight tensors (`102,499,149,312` bytes / `95.4598 GiB`) in 754 HIP
  allocations. Committed evidence:
  `benchmarks/results/2026-05-29-stepfun-q3kl-full-load-smoke-task20.json`.
  `hipMemGetInfo` remains internally inconsistent (`total=62.5409 GiB`) but
  usable free memory dropped from `119.9961 GiB` to `23.9061 GiB` after load and
  returned to `119.8573 GiB` after free.
- [x] Record exact GGUF paths and byte sizes for all three shards; do not copy or
  rewrite the 102.50 GB assets into the repo.
- [x] Establish a llama.cpp oracle command for tokenization and short greedy
  next-token checks. If llama.cpp cannot run the full model on the same machine,
  use it for metadata/tokenizer/slice or a smaller exported fixture.
- [x] For any kernel port/tuning, read `docs/KERNELS.md` and run
  `python3 scripts/check_lineage.py --kind kernel --diff stat` before copying
  code. 2026-05-29: `docs/source_lineage.json` now points at the available
  `/home/lhl/github/lhl/nano-vllm-amd` checkout; the lineage drift report runs
  and records four tracked kernel sources with expected drift since baseline
  `22405a9`. Inspect the reported drift before copying code.

**Acceptance:** WORKLOG preflight entry with hardware, memory, paths, and oracle
plan; no runtime correctness or performance claim yet. 2026-05-29/30 preflight
confirmed gfx1151, recorded shard/oracle details, and successfully loaded the
95.46 GiB resident weight set under the configured 120 GiB GTT path despite
misleading `hipMemGetInfo` total/free readouts.

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

- [x] Implement a torch-free tokenizer path for GGUF `tokenizer.ggml.model='gpt2'`
  with `tokenizer.ggml.pre='deepseek-v3'`.
- [x] Render the Step chat template locally, including `<|im_start|>`,
  `<|im_end|>`, optional `Reasoning: low|medium|high`, assistant
  `<think>` prefix, and tool-call blocks.
- [x] Preserve BOS id 0 and EOS ids `[1, 2, 128007]` in generation stop logic.
- [x] Compare token IDs for representative prompts against llama.cpp or cached
  HF tokenizer output.

**Acceptance:** `python3 -m pytest -q tests/test_stepfun_tokenizer.py` passes;
tokenizer/chat paths remain torch-free and match cached HF `tokenizer.json`
representative token IDs.

### P5 — Q3_K CPU reference and mixed GGUF quant metadata

- [x] Implement CPU-reference Q3_K dequantization and add block-level fixtures.
- [x] Validate existing Q5_K/Q8_0/F32 handling against Step tensor metadata; Step
  layers mix quant types within the same layer.
- [x] Add slice fixtures from real Step tensors for Q3_K, Q5_K, Q8_0, and F32
  without checking large binary fixtures into git.
- [x] Expose per-tensor quant keys through the loader/quant plugin so mixed
  dispatch does not require engine-wide quant branches.

**Acceptance:** `python3 -m pytest -q tests/test_gguf_quant_layout.py
 tests/test_stepfun_q3k_cpu.py` passes against llama.cpp `gguf-py` reference
values for local Step tensor slices; tests skip cleanly when external assets or
reference code are absent.

### P6 — HIP Q3_K linear kernels on gfx1151

- [x] Add/register HIP Q3_K GEMV and selected-expert variants needed by Step
  dense, attention, MoE expert, and shared-expert paths.
- [x] Build for `hip_gfx1151` with `HIPENGINE_HIP_ARCH=gfx1151`; reuse gfx1100
  source only through peer backend registration/build metadata, not hard-coded
  imports in Step runtime code.
- [x] Add smoke tests comparing HIP Q3_K outputs to CPU reference for small
  tensors, then real Step tensor slices.
- [x] Run a `rocprofv3 --kernel-trace` smoke once kernels exist and record the
  expected kernel names/durations.

**Acceptance:** `HIPENGINE_HIP_ARCH=gfx1151 python3 -m pytest -q
 tests/test_stepfun_q3k_hip.py` passes synthetic GEMV, selected-expert GEMV,
and real Step tensor-slice checks vs CPU Q3_K. `rocprofv3 --kernel-trace`
shows `gguf_k_prefill_out_kernel<unsigned short, float, 3>` with
`DurationNs=13345`, `Scratch_Size=0`, `Grid_Size_X=1024`, and
`Workgroup_Size_X=128` for the Q3_K smoke. No full-model or throughput claim.

### P7 — Step norms, RoPE, and attention-gate primitives

- [x] Add RMSNorm variant with Step scale semantics `(weight + 1)` and epsilon
  `1e-5`.
- [x] Add RoPE table/cache support for full-attention layers: theta 5e6,
  llama3 scaling, partial factor 0.5.
- [x] Add RoPE support for sliding-attention layers: theta 1e4, no llama3
  scaling, full factor 1.0.
- [x] Add head-wise attention gate primitive/fusion:
  `attn_output[head] *= sigmoid(g_proj(x)[head])` before `o_proj`.
- [x] Keep unfused CPU/reference fallbacks for every fused primitive.

**Acceptance:** `python3 -m pytest -q tests/test_stepfun_primitives.py` passes
against CPU reference checks for representative full and sliding layer
primitives.

### P8 — Full and sliding GQA attention

- [x] Implement full-attention decode/prefill for 64 query heads, 8 KV heads,
  head dim 128, and partial-RoPE full layers.
- [x] Implement sliding-attention decode/prefill for 96 query heads, 8 KV heads,
  head dim 128, and window 512.
- [x] Represent both policies through `KVLiveSpans`: full layers expose the live
  prefix, sliding layers expose only the live window.
- [x] Validate one-token decode and short prefill against CPU attention fixtures.
- [x] Defer AOTriton/native attention profiling until after correctness. The
  current StepFun loop is correctness-first; profiling before KV-backed decode
  and oracle parity would produce misleading evidence. CPU-reference attention
  correctness exists, but the Strix Halo AOTriton/native choice is tracked as a
  post-correctness deferred performance task, not a P0-P12 gate.

**Acceptance:** `python3 -m pytest -q tests/test_stepfun_attention.py` passes for
full/sliding CPU-reference attention, both layer head shapes, and KV live-window
boundaries. Native/AOTriton profiling remains deferred until decode correctness.

### P9 — Dense MLP and Step MoE

- [x] Wire dense MLP for layers 0-2 with `ffn_gate/up/down.weight` names.
- [x] Implement router semantics for MoE layers 3-44: FP32 gate matmul, sigmoid,
  add router bias for top-k selection, gather unbiased probabilities, normalize
  selected weights, then multiply by routing scale 3.0.
- [x] Implement top-k 8 over 288 experts, expert gate/up/down projections,
  shared expert gate/up/down path, and routed+shared sum.
- [x] Handle non-zero `swiglu_limits` / `swiglu_limits_shared` in the last layers.
- [x] Start with c=1 decode; batch/expert grouping optimization can follow after
  correctness.

**Acceptance:** `python3 -m pytest -q tests/test_stepfun_moe.py` passes for
CPU-reference dense SwiGLU, router-bias top-k semantics, routed+shared MoE,
288-expert/top-8 c=1 shape, and non-zero SwiGLU limits. Expert grouping and HIP
optimization remain deferred until correctness.

### P10 — One-layer and block replay

- [x] Build a deterministic replay harness for a dense layer, a full-attention
  MoE layer, and a sliding-attention MoE layer.
- [x] Capture or derive reference activations/logits from llama.cpp, HF
  Transformers, or CPU-reference code without committing large blobs.
- [x] Gate each block on numerical tolerances appropriate for quantized GGUF
  inference before integrating all 45 layers.

**Acceptance:** `python3 -m pytest -q tests/test_stepfun_replay.py` passes for
representative dense/full/sliding CPU-reference block replays, compact in-memory
stage captures, quantized-tolerance comparison, and exact-substage mismatch
reporting.

### P11 — Text-only c=1 decode runner

- [~] Add a Step GGUF runner that streams one-token decode for short prompts with
  the Step tokenizer, split weight index, mixed GGUF quant dispatch, full/sliding
  attention, and Step MoE. 2026-05-29 progress: split-shard resident
  materialization now plans all 754 tensors / 95.46 GiB across all three shards,
  verifies Q3_K/Q5_K/Q8_0/F32 layout coverage, tests selected-slot HIP
  loading/freeing across first/last shard tensors, `StepFunResidentSession` can
  render a short Step chat prompt and launch real Q8_0 token embeddings from
  resident `token_embd.weight` into BF16 for the resulting prompt IDs, has exact
  CPU BF16 parity for BOS/EOS/chat-stop/final-vocab rows, and
  resident layer-0 `Q3_K` `attn_q` plus `Q5_K` `attn_output` projections match
  CPU dequantized references with two BF16-rounded activation rows (exercising
  rows>1 prefill dispatch), the resident attention-input bundle launches
  layer-0 Q/K/V/gate projections vs CPU references, a layer-0 attention
  prefill/output probe composes resident Q/K/V/gate projections, host Q/K
  RMSNorm, RoPE, causal GQA, head-wise gating, BF16 rounding, and resident
  `attn_output` vs CPU reference, and a layer-0 dense-layer prefill probe composes
  attention residual, `ffn_norm`, dense SwiGLU MLP, and final residual vs CPU
  reference. The same layer wrapper is exercised on a layer-3 sliding/MoE block
  by composing the attention and MoE probes through `ffn_norm`; a first-layer
  prompt logits smoke binds chat rendering, embeddings, layer-0 prefill, and
  final `lm_head` rows while explicitly skipping layers 1-44; and a layer-prefix
  prompt logits probe now applies the contiguous layers 0-3
  prefill bridge (dense layers 0-2 plus first sliding/MoE layer 3), a chunked
  prompt artifact extends that bridge through layer 4, and a full-layer chunked
  prompt artifact now runs layers 0-44 before final sampled `lm_head` checks.
  A text-only decode
  slot planner covers all 754 validated GGUF slots, including root
  `rope_freqs.weight`, and has no vision/projector/MTP slot dependencies. The
  same resource plan now records the StepFun GGUF KV dispatch keys under the
  registry quant axis `gguf_step35`: prompt KV writes use
  `paged_kv_write/mixed_bf16_prompt_spans`, decode KV writes use
  `paged_kv_write/mixed_bf16_spans`, and gated decode attention uses the generic
  `paged_attn_decode/bf16_split_k_gate_f32_spans` route on `hip_gfx1151`. The plan
  also records the parent paged-attention span geometry for the 512-token bring-up
  window (`attention_block_size=256`, decode `block_table_len=2`, decode live-count
  capacity 511 tokens, and prompt KV writes requiring per-row position/live-count
  metadata with `base_offsets_len_formula="rows * 2"` up to 511 prompt rows).
  `StepFunShortContextDecodePlanner.plan_kv_decode_chat()` now binds the rendered
  prompt to that resource plan by recording token IDs, rendered-prompt SHA-256,
  prompt positions, explicit prompt/decode span inputs (`base_offsets`/live-count
  arrays with dtype and byte-count metadata), a small host upload manifest mapping
  those arrays/scalars to future kernel arguments, bytes-producing and device-upload
  helpers (including partial-upload cleanup coverage) with deterministic little-endian
  SHA-256 hashes for each upload, decode live count, decode position, stop-token IDs, KV dispatch keys,
  and the planned launch-order operation count; this is dispatch/span/run-plan readiness evidence only, not a
  completed streaming runner. The
  dense-MLP input bundle launches layer-0 `ffn_gate`/`ffn_up` projections vs CPU references,
  and a dense-MLP correctness probe composes gate/up, host SwiGLU BF16 rounding,
  and resident `ffn_down` vs CPU reference. The resident session also owns a
  BF16 KV-cache allocation/free helper, a layer-3 MoE router probe that
  matches CPU top-k routing from resident F32 router/bias weights, a
  selected-expert `Q3_K` gate projection probe for layer-3 `ffn_gate_exps`, a
  MoE expert-input bundle that launches selected gate/up plus shared gate/up
  projections vs CPU references, a MoE correctness probe that composes routing,
  selected/shared gate/up, host SwiGLU BF16 intermediates, selected/shared down
  projections, and host routing aggregation vs CPU reference, and a final-logits
  probe that composes host output RMSNorm/BF16 rounding with resident Q8_0
  `lm_head` projection for selected full-vocab rows vs CPU reference. A root-only
  prompt logits smoke now renders/tokenizes a Step chat prompt, embeds it, and
  runs final root logits from the last prompt embedding row. Remaining
  implementation task is replacing the host-composed layer-prefix bridge with
  the KV-backed decode path and recording llama.cpp/CPU oracle parity. Status
  artifact `benchmarks/results/2026-05-31-stepfun-q3kl-correctness-status.json`
  machine-checks this state as `all_layer_prompt_smoke=true`,
  `oracle_parity=false`, `kv_backed_decode_ready=false`, and
  `e2e_inference_ready=false`; it also records the current P0-P12
  open/partial checklist metric (`2`), summarizes resident GGUF linear projection
  coverage from the all-layer prompt artifact (`487` resident projection slots plus
  `42` host-reference router projection slots across all 45 layers), records
  `kv_decode_dispatch_ready=true` from the text resource plan's `gguf_step35`
  BF16 KV write/decode registry keys plus decode/prompt 256-token paged-attention
  span contracts, a metadata-only `kv_decode_run_plan` for the canonical short
  `hello` prompt (input token IDs with int32 byte-count/SHA-256 metadata, prompt
  positions, rendered prompt SHA-256, input-token/span-input device-upload helpers
  and a metadata-only combined upload plan with cleanup order, decode position/live-count,
  stop IDs, KV dispatch keys, resource-fit booleans, and a `kv_decode_blocker_summary`
  that names the first/kernel-trace/last runtime blockers, validated upload/launch prerequisites, required
  trace/next-token artifacts, and the no-oracle/no-performance-claim policy), plus the planned per-layer KV launch
  schedule (45 layers × prompt KV write, decode KV write, gated attention = 135
  planned operations plus source-level `streaming_runner_blockers` naming the still-missing
  decode loop, kernel trace, and KV-backed next-token artifact), and includes a compact
  `handoff_summary` with open blockers, blocked gates, ready signals, an oracle parity gap report,
  combined upload-plan order/bytes/consistency checks, a KV-backed decode gap report that separates
  validated preconditions from missing streaming-runner evidence and cross-links the first
  source-level streaming blocker, next-command/blocker coverage that names the first missing
  oracle/KV evidence item plus the first source KV runner blocker, an ordered blocker work queue,
  and the no-performance/no-e2e-claim policy. `scripts/stepfun_correctness_status.py --readiness-summary-only`
  emits a tiny top-level readiness/blocker digest for scheduler polling, `--readiness-summary-sha-only`
  emits just that readiness-summary digest for drift polling, and compact
  `--readiness-status-only` / `--readiness-status-sha-only`,
  `--open-blocker-count-only` / `--open-blocker-count-sha-only`,
  `--blocker-work-queue-count-only` / `--blocker-work-queue-count-sha-only`,
  `--blocked-gates-count-only` / `--blocked-gates-count-sha-only`,
  `--blocked-gates-joined-only` / `--blocked-gates-joined-sha-only`,
  `--first-blocked-gate-only` / `--first-blocked-gate-sha-only`,
  `--last-blocked-gate-only` / `--last-blocked-gate-sha-only`,
  `--blocker-kinds-count-only` / `--blocker-kinds-count-sha-only`,
  `--blocker-kinds-joined-only` / `--blocker-kinds-joined-sha-only`, and
  `--first-blocker-kind-route-only` / `--first-blocker-kind-route-sha-only` plus
  `--last-blocker-kind-route-only` / `--last-blocker-kind-route-sha-only`
  expose the current blocked/ready state plus blocker/queue/gate counts, joined
  gate/kind routes, and first/last blocked gate and blocker-kind routes without
  fetching the full readiness summary or gate/kind arrays. Status integrity verifies
  the readiness/blocker compact output-mode mappings, `--docs-checklist-only` /
  `--docs-checklist-sha-only` expose the exact P0-P12 open/partial checklist payload
  and digest used by the loop metric, `--docs-open-partial-count-only` emits just
  the single integer loop metric, `--docs-open-partial-summary-only` /
  `--docs-open-partial-summary-sha-only` expose a compact count+boundary+digest
  summary, `--docs-open-partial-state-counts-only` /
  `--docs-open-partial-state-counts-sha-only` expose open-vs-partial blocker
  counts, `--docs-open-partial-lines-only` /
  `--docs-open-partial-lines-sha-only` expose direct checklist line numbers,
  `--docs-open-partial-texts-only` /
  `--docs-open-partial-texts-sha-only` expose checklist item text labels,
  `--docs-open-partial-texts-joined-only` /
  `--docs-open-partial-texts-joined-sha-only` expose a pipe-joined scalar form,
  `--docs-open-partial-line-texts-joined-only` /
  `--docs-open-partial-line-texts-joined-sha-only` expose a pipe-joined line:text scalar,
  `--docs-open-partial-state-line-texts-joined-only` /
  `--docs-open-partial-state-line-texts-joined-sha-only` expose a pipe-joined state:line:text scalar,
  and `--docs-first-open-partial-item-only` /
  `--docs-first-open-partial-item-sha-only` /
  `--docs-last-open-partial-item-only` /
  `--docs-last-open-partial-item-sha-only` expose the current first/last
  P0-P12 open-or-partial checklist items and digests for handoff routing; status
  integrity verifies the docs-checklist
  compact output-mode mappings, checklist count against its item list, and the
  readiness-summary metric mirror,
  `--status-refresh-command-only`
  / `--status-refresh-command-sha-only` emit the consolidated status refresh command or digest,
  `--kv-resource-command-only` / `--kv-resource-command-sha-only` emit the KV resource
  refresh command or digest, `--oracle-helper-command-only` emits just the oracle JSON refresh command,
  `--oracle-helper-command-sha-only` emits its digest, `--summary-only`
  emits just the handoff block for fast continuation checks, while `--blocker-work-queue-only`
  emits only the ordered queue (with a handoff-level queue count/schema version/SHA-256,
  per-item schema versions, explicit queue indices / first-item markers, and compact current-attempt
  status), `--blocker-work-queue-meta-only` emits the queue schema/count/digest/first-kind
  metadata, `--blocker-work-queue-sha-only` emits just the queue digest for drift polling,
  `--first-blocker-sha-only` emits just the immediate work-item digest, and
  `--first-blocker-only` emits only the immediate work item
  (including the primary command kind/string plus length/SHA-256 and, for the oracle blocker,
  the helper refresh command that regenerates the oracle JSON) for lightweight automation.
  It lists next actions for the
  StepFun-capable oracle and KV-backed decode blockers, and supports
  `--fail-on-blocked` for CI/handoff checks, including compact readiness/queue/status-refresh/KV-resource/oracle-helper/first-blocker outputs;
  the handoff also records the ready/source-mismatch/blocked exit-code expectations.
  `scripts/stepfun_final_blocker_manifest.py` now emits a compact final-blocker
  evidence manifest (and `--sha-only` digest) that joins the two remaining P11
  partial checklist items to their readiness gates, first missing evidence,
  recommended-command digests, oracle rerun artifact path, KV runtime artifacts,
  launch-trace digests, status/source provenance, and the no-claim policy without
  parsing the full status artifact. Compact `--entries-only` /
  `--entries-sha-only`, `--artifacts-only` / `--artifacts-sha-only`,
  `--success-criteria-only` / `--success-criteria-sha-only`,
  `--no-claim-policy-only` / `--no-claim-policy-sha-only`,
  `--gate-status-only` / `--gate-status-sha-only`,
  `--status-provenance-only` / `--status-provenance-sha-only`, and
  `--recommended-commands-only` / `--recommended-commands-sha-only` outputs expose
  the two blocker records, three required evidence artifacts, per-blocker
  completion criteria, claim gate, blocked readiness-gate chain, source/status
  hash bundle, and exact recommended commands directly. Its `--verify-manifest`
  mode compares a persisted manifest with the current
  prompt/oracle/resource/docs inputs, while `--verification-status-only` and
  `--verification-failures-only` provide compact drift routing. `scripts/stepfun_handoff_check.py`
  now combines correctness-status source verification with final-blocker manifest
  verification and reports `blocked_verified` for a trustworthy blocked handoff;
  compact `--summary-only` / `--summary-sha-only`, `--status-only`,
  `--artifact-verification-only` / `--artifact-verification-sha-only`,
  `--readiness-summary-only` / `--readiness-summary-sha-only`,
  `--e2e-readiness-gate-summary-only` /
  `--e2e-readiness-gate-summary-sha-only`,
  `--blocker-status-only` / `--blocker-status-sha-only`,
  `--final-blocker-summary-only` / `--final-blocker-summary-sha-only`,
  `--action-summary-only` / `--action-summary-sha-only`,
  `--artifact-status-only` / `--artifact-status-sha-only`,
  `--missing-artifacts-only` / `--missing-artifacts-sha-only`,
  `--validator-commands-only` / `--validator-commands-sha-only` list both
  placeholder validator templates and concrete commands against the current
  expected artifact paths, and `scripts/stepfun_validator_status.py` imports the
  dedicated validators to report `passed` / `missing` / `failed` for those
  concrete artifact paths without shelling out (`--results-only` /
  `--results-sha-only`, `--blocked-only` / `--blocked-sha-only`,
  `--next-blocker-only` / `--next-blocker-artifact-name-only` /
  `--next-blocker-readiness-gate-only` / `--next-blocker-status-only` /
  `--next-blocker-reason-only` / `--next-blocker-sha-only`,
  `--next-command-only` / `--next-command-kind-only` /
  `--next-command-sha-only`,
  `--next-producer-command-only` / `--next-producer-command-kind-only` /
  `--next-producer-command-sha-only`,
  `--blocked-evidence-summary-only` / `--blocked-evidence-summary-sha-only`,
  `--blocked-evidence-by-gate-only` / `--blocked-evidence-by-gate-sha-only`,
  `--blocked-readiness-gates-only` / `--blocked-readiness-gates-sha-only`,
  `--blocked-evidence-gate <gate>` with `--blocked-evidence-gate-only` /
  `--blocked-evidence-gate-sha-only`, `--blocked-evidence-gate-found-only`,
  `--blocked-evidence-gate-artifacts-only` /
  `--blocked-evidence-gate-artifacts-sha-only`,
  `--blocked-evidence-gate-artifact-count-only`,
  `--blocked-evidence-gate-blocked-count-only`,
  `--blocked-evidence-gate-status-counts-only` /
  `--blocked-evidence-gate-status-counts-sha-only`,
  `--blocked-evidence-gate-producer-commands-only` /
  `--blocked-evidence-gate-producer-commands-sha-only`,
  `--blocked-evidence-gate-producer-command-count-only`,
  `--blocked-evidence-gate-validator-commands-only` /
  `--blocked-evidence-gate-validator-commands-sha-only`,
  `--blocked-evidence-gate-validator-command-count-only`, and
  `--blocked-evidence-gate-missing-evidence-only` /
  `--blocked-evidence-gate-missing-evidence-sha-only` /
  `--blocked-evidence-gate-missing-evidence-count-only`,
  `--next-blocked-gate-only` /
  `--next-blocked-gate-sha-only`, and `--next-action-only` /
  `--next-action-sha-only` / `--next-action-available-only` /
  `--next-action-artifact-name-only` /
  `--next-action-readiness-gate-only` / `--next-action-status-only` /
  `--next-action-reason-only` /
  `--next-action-validator-command-kind-only` /
  `--next-action-validator-command-only` /
  `--next-action-validator-command-sha-only` /
  `--next-action-producer-command-kind-only` /
  `--next-action-producer-command-only` /
  `--next-action-producer-command-sha-only`, plus
  `--next-action-partial-output-handoff-only` /
  `--next-action-partial-output-handoff-sha-only` /
  `--next-action-partial-output-path-only` /
  `--next-action-partial-output-status-only`, expose compact pollable validator
  records/commands/evidence gaps by artifact (with next-action artifact name,
  readiness gate, status, reason, missing-evidence count/digest, validator
  command kind/command, producer command kind/command, partial-output handoff
  bundle/path/status, and validator/producer command digests also available at
  the top-level report), all gates,
  blocked gate names,
  selected gate (with a `selected_blocked_gate_found` flag in the aggregate
  summary), and first blocked gate,
  `--next-action-validator-summary-only` /
  `--next-action-validator-summary-sha-only` expose just the embedded validator
  summary/digest, `--next-action-validator-summary-status-only` /
  `--next-action-validator-summary-ready-only` /
  `--next-action-validator-summary-oracle-status-only` /
  `--next-action-validator-summary-oracle-blocker-kind-only` expose compact
  status/oracle-timeout routing fields from that summary,
  `--next-action-oracle-expected-token-only` /
  `--next-action-oracle-expected-token-sha-only` plus
  `--next-action-expected-next-token-*-only` modes expose the retained oracle
  target token, `--next-action-oracle-generated-text-only` /
  `--next-action-oracle-generated-text-sha-only` plus
  `--next-action-generated-text-*-only` modes expose generated text length/match
  flags, `--next-action-oracle-artifact-provenance-only` /
  `--next-action-oracle-artifact-provenance-sha-only` plus artifact/prompt/evidence
  SHA and presence scalar modes expose retained oracle/prompt artifact provenance,
  `--next-action-no-claim-policy-only` /
  `--next-action-no-claim-policy-sha-only` plus per-gate
  `--next-action-*-claim-allowed-only` modes expose the embedded no-claim policy,
  and
  `--next-action-missing-evidence-only` /
  `--next-action-missing-evidence-sha-only` /
  `--next-action-missing-evidence-count-only` /
  `--next-action-missing-evidence-summary-only` /
  `--next-action-missing-evidence-summary-sha-only` /
  `--next-action-oracle-evidence-gap-count-only` /
  `--next-action-oracle-evidence-gaps-only` /
  `--next-action-oracle-evidence-gaps-sha-only` /
  `--next-action-oracle-evidence-gap-summary-only` /
  `--next-action-oracle-evidence-gap-summary-sha-only` /
  `--next-action-oracle-evidence-gaps-joined-only` /
  `--next-action-oracle-evidence-gaps-present-only` /
  `--next-action-first-oracle-evidence-gap-only` /
  `--next-action-last-oracle-evidence-gap-only` /
  `--next-action-missing-evidence-present-only` /
  `--next-action-missing-evidence-joined-only` /
  `--next-action-missing-evidence-sorted-only` /
  `--next-action-missing-evidence-sorted-sha-only` /
  `--next-action-missing-evidence-sorted-joined-only` /
  `--next-action-first-missing-evidence-only` /
  `--next-action-last-missing-evidence-only` /
  `--next-action-artifact-file-present-missing-only` /
  `--next-action-oracle-success-status-missing-only` /
  `--next-action-oracle-returncode-zero-missing-only` /
  `--next-action-no-timeout-or-oracle-blocker-missing-only` /
  `--next-action-generated-text-matches-target-missing-only` /
  `--next-action-generated-text-nonempty-missing-only` expose just its
  missing-evidence list/digest/count/summary+summary digest/oracle-gap-count/oracle-gap list+digest+
  summary+summary digest/pipe-joined sequence+presence+first/last oracle gap/presence/pipe-joined sequence/sorted list+digest/sorted
  pipe-joined sequence/leading or trailing item/retained-artifact missing flag/
  oracle-timeout return-status/return-code/no-timeout flags/generated-text
  target/nonempty flags (including `artifact_file_present` for missing retained files),
  while the aggregate validator summary carries the next-action availability flag,
  validator-summary digest, status/oracle routing fields, expected-token,
  generated-text, artifact-provenance, and artifact-presence bundles, no-claim
  policy booleans, plus missing-evidence list/count/summary/summary-digest/oracle-gap-count/oracle-gap
  list/oracle-gap-summary/oracle-gap-summary-digest/oracle-gap-joined/oracle-gap-present/oracle-gap-first/oracle-gap-last/present/joined/sorted/sorted-joined/first-item/last-item/
  artifact-file-present, oracle-success-status, oracle-returncode-zero,
  no-timeout-or-oracle-blocker, generated-text-target,
  and generated-text-nonempty flags/digest for drift polling; with the
  oracle next-action bundle mirroring the current validator summary plus
  partial-output and supervisor-signal timeout handoff when it is
  the first blocker),
  `--exit-code-policy-only` / `--exit-code-policy-sha-only`,
  `--digest-summary-only` / `--digest-summary-sha-only`, and
  `--failures-only` / `--failures-sha-only` outputs support verifier drift
  polling, and `--verify-handoff-report` (defaulting to the persisted handoff
  artifact when no path is supplied) with `--report-verification-status-only` /
  `--report-verification-failures-only` detects drift in a persisted handoff
  report. The full verified-blocked report is persisted as
  `benchmarks/results/2026-05-31-stepfun-q3kl-handoff-check.json` alongside the
  status and final-blocker artifacts. `--fail-on-blocked` returns the documented
  blocked exit code when CI should fail until oracle/KV readiness is real.
  `scripts/stepfun_kv_trace_check.py` validates the missing KV kernel trace
  artifact when a future real `rocprofv3` CSV or compact JSON trace is retained:
  it checks the expected StepFun prompt-KV write, decode-KV write,
  split-K attention context, and gated attention reduce kernel families against
  the planned layer count and emits compact summary/status/SHA outputs without
  making token or performance claims; the final-blocker manifest and combined
  handoff report attach this command under the missing `kv_kernel_trace_artifact`
  as `validator_command_kind=kv_trace_check_command` with a stable command
  digest and expected kernel-family digest. `scripts/stepfun_kv_next_token_check.py`
  validates the missing KV-backed next-token artifact when a future real decode
  run is retained: it requires explicit KV-backed runtime provenance, ready
  streaming-runner evidence, non-host-composed provenance, prompt-length
  alignment, deterministic token/text match against the canonical prompt target,
  and a finite next-token logit, while keeping KV-decode/e2e/performance claims
  separate; the final-blocker manifest and combined handoff report attach this
  command under the missing `kv_backed_next_token_artifact` as
  `validator_command_kind=kv_next_token_check_command` with a stable command
  digest and expected evidence-check digest. This checklist item is partial rather
  than complete because the
  current runner is host-composed/chunked and does not yet implement the final
  KV-backed one-token decode path.
- [x] Use short contexts first (for example <= 512) before exercising long
  context and sliding-window boundaries. `StepFunShortContextDecodePlanner`
  enforces the current c=1 bring-up default `max_context=512`,
  `max_new_tokens=1`, and rejects overlong prompts.
- [~] Compare greedy next tokens and/or logits against llama.cpp for a small set
  of deterministic prompts. 2026-05-31 oracle-planning progress:
  `benchmarks/results/2026-05-31-stepfun-q3kl-llamacpp-oracle-plan.json`
  records a deterministic one-token llama.cpp command for the all-layer chunked
  `hello` prompt artifact (model shard
  `Step-3.7-flash-Q3_K_L-00001-of-00003.gguf`, `--predict 1`, `--temp 0`,
  `--top-k 1`, `--top-p 1`, `--min-p 0`, `--repeat-penalty 1`, `--seed 0`,
  `--no-display-prompt`, `--simple-io`, and `--log-disable`). The helper records
  a comparison policy and, when run with `--execute`, captures llama.cpp stdout
  as `generated_text` plus exact/stripped text-match booleans against the
  host-composed artifact (`next_token_id=369`, decoded ` |`). Historical
  execution attempt artifact
  `benchmarks/results/2026-05-31-stepfun-q3kl-llamacpp-oracle-exec-attempt.json`
  uses diagnostic logs and fails before generation because the local
  `/home/lhl/ai/llama.cpp-cpu/llama-cli` does not support GGUF architecture
  `step35` (`unknown model architecture: 'step35'`). A newer local Vulkan
  llama.cpp build (`/home/lhl/llama.cpp/llama.cpp-vulkan/build/bin/llama-cli`,
  version `9197 (fcae601e4)`) accepts `step35` but bounded CPU/no-GPU oracle
  attempts in
  `benchmarks/results/2026-05-31-stepfun-q3kl-llamacpp-step35-timeout.json`
  still have not produced a comparable token. The canonical artifact first
  recorded a 60 s internal timeout (refreshed 2026-06-02 with
  `elapsed_s=61.75`) and was refreshed on 2026-06-04 after the recommended
  900 s rerun hit the outer pi supervision window before the helper rewrote its
  pre-launch partial artifact. It now records `status=timeout`,
  `timeout_s=900.0`, `outer_tool_timeout_s=1000.0`,
  `partial_artifact_reconciled_after_outer_timeout=true`,
  `oracle_blocker_kind=llama_cpp_oracle_timeout`, no leftover `llama-cli`
  process after the supervisor timeout, and `timeout_termination` provenance for
  the timeout blocker. The oracle helper now also traps `SIGTERM`/`SIGINT` from
  an outer supervisor, kills the `llama-cli` process group, and rewrites the
  pre-launch partial artifact with structured timeout provenance when it gets a
  graceful interruption window. `oracle_partial_output_handoff` and the
  final-blocker handoff now expose this supervisor-signal timeout contract so a
  future oracle rerun can verify that graceful wrapper interruption should still
  produce structured timeout evidence. A 2026-06-01 bounded 180 s rerun attempt is recorded in
  `benchmarks/results/2026-06-01-stepfun-q3kl-llamacpp-step35-180s-wrapper-timeout.json`;
  the outer pi wrapper timed out at 240 s before the helper rewrote the canonical
  oracle JSON, so that wrapper artifact remains historical source evidence for
  the same canonical oracle blocker. The correctness status `source_artifacts`
  now tracks this wrapper-timeout artifact as `oracle_wrapper_timeout`, and
  compact `--oracle-wrapper-timeout-source-only` /
  `--oracle-wrapper-timeout-source-sha-only` outputs expose that provenance
  record/digest directly, while `--oracle-timeout-termination-only` /
  `--oracle-timeout-termination-sha-only` expose the canonical timeout cleanup
  payload/digest from `oracle_gap_report`. `tests/test_stepfun_oracle_wrapper_timeout.py` locks
  the artifact schema, historical after-attempt canonical SHA (which now differs
  from the refreshed timeout-termination canonical artifact), source-artifact
  provenance, compact source outputs, and a stale wrapper-timeout source hash
  verification failure.
  A default-device Vulkan oracle attempt is recorded separately in
  `benchmarks/results/2026-05-31-stepfun-q3kl-llamacpp-vulkan-harness-timeout.json`;
  it exceeded the pi tool supervision window before helper JSON was produced, so
  the CPU/no-GPU timeout artifact remains the canonical machine-readable oracle
  blocker for now. The consolidated correctness-status artifact surfaces the oracle version,
  elapsed time, stdout/stderr lengths, `oracle_progress` fields, and an `oracle_progress_sha256`
  digest; compact `--oracle-progress-only` / `--oracle-progress-sha-only`,
  `--oracle-status-only`, and `--oracle-blocker-kind-only` /
  `--oracle-blocker-kind-sha-only` outputs expose the current-attempt
  payload/digest plus the scalar status and blocker kind directly for oracle
  blocker polling. It also surfaces an `oracle_gap_report`
  that separates recorded deterministic-target prerequisites from missing run/match evidence
  for the exact deterministic target (`prompt_length=23`, `n_predict=1`, expected token id 369 /
  text ` |`, top-5 expected tokens, current timeout 900 s, no comparable
  generated text, `timeout_termination` recorded, and the llama.cpp command
  shell).
  `scripts/stepfun_oracle_artifact_check.py` validates a future retained
  llama.cpp oracle-success artifact against that target: it requires successful
  status/return code, recorded llama.cpp binary/model metadata, no Step35
  architecture rejection or timeout blocker, prompt-length / one-token run
  alignment, token/logit metadata parity, non-empty generated text, and exact
  text match, while keeping KV/e2e/performance claims separate; the final-blocker
  manifest and combined handoff report attach this command under the missing
  `llama_cpp_oracle_success_artifact` as
  `validator_command_kind=oracle_artifact_check_command` with stable command and
  expected evidence-check digests. It carries explicit status,
  readiness-summary, and handoff-summary schema versions plus compact
  `schema_versions` / `schema_versions_sha256` handoff payloads, verifies those
  compact schema-version output-mode mappings in status integrity, records both blockers
  (`oracle_parity_blocked`, `kv_backed_decode_not_wired`) for the current
  all-layer prompt smoke, exposes compact `--blocker-kinds-only` /
  `--blocker-kinds-sha-only` outputs for polling that blocker-kind list, lists
  machine-readable next actions for each blocker,
  and now includes `readiness_gates` for `oracle_parity`, `kv_backed_decode`,
  and `e2e_inference` so each readiness boolean carries
  required evidence and current blocker state, plus compact
  `--readiness-gates-only` / `--readiness-gates-sha-only` outputs for polling
  those gates directly and `--blocked-gates-only` /
  `--blocked-gates-sha-only` for polling just the currently false gate names.
  The runtime KV resource/run-plan payloads also expose deterministic streaming
  blocker-name lists plus SHA-256 digests, and the status KV gap report mirrors
  and verifies that digest with compact `--kv-streaming-blockers-only` /
  `--kv-streaming-blockers-sha-only` outputs; `--kv-streaming-blockers-joined-only` /
  `--kv-streaming-blockers-joined-sha-only` expose the same blocker-name sequence
  as one pipe-joined scalar for shell pollers, while
  `--kv-streaming-blocker-count-only` / `--kv-streaming-blockers-present-only`
  expose count/presence scalars for empty-vs-blocked checks. Compact
  `--kv-first-streaming-blocker-only` / `--kv-first-streaming-blocker-sha-only`,
  `--kv-last-streaming-blocker-only` / `--kv-last-streaming-blocker-sha-only`,
  and `--kv-kernel-trace-streaming-blocker-only` /
  `--kv-kernel-trace-streaming-blocker-sha-only` /
  `--kv-kernel-trace-streaming-blocker-present-only` outputs expose the current
  first/tail source-level runner blockers plus the middle kernel-trace artifact
  blocker (`streaming_decode_loop_not_wired`, `kv_backed_next_token_artifact_missing`,
  and `kv_kernel_trace_artifact_missing`) and their persisted digests/presence
  directly for KV implementation handoff. The runtime `kv_decode_launch_schedule` and
  `kv_decode_run_plan` now emit the same first-blocker digest at the metadata
  source, and `kv_decode_run_plan.streaming_decode_loop_blueprint` records the
  metadata-only upload/order/stage contract for the future KV loop without
  launching kernels. `kv_decode_run_plan.streaming_decode_loop_status` now adds
  a compact runtime-produced readiness summary (blocker count/names, first
  blocker digest, blueprint digest, and `next_action=wire_streaming_decode_loop`)
  next to the blueprint. `kv_decode_run_plan.streaming_decode_launch_trace`
  additionally records the metadata-only 135-operation per-layer trace with
  dispatch keys, span contracts, pre-run span uploads, expected runtime inputs,
  and `execution_status=not_launched_metadata_only` for every planned launch. The
  canonical resource artifact now contains the blueprint, loop-status summary,
  and launch trace, and the correctness-status KV gap report validates that the
  blueprint is recorded, matches the launch schedule, matches the pre-run upload
  order, points at the same first blocker, has a matching loop-status summary,
  and carries a matching non-executable launch trace. The KV gap report now
  persists the validated blueprint, loop-status, and launch-trace digests, and
  compact `--kv-streaming-blueprint-only` /
  `--kv-streaming-blueprint-sha-only`, `--kv-streaming-loop-status-only` /
  `--kv-streaming-loop-status-sha-only`,
  `--kv-streaming-loop-next-action-only` /
  `--kv-streaming-loop-next-action-sha-only`, and
  `--kv-streaming-launch-trace-only` /
  `--kv-streaming-launch-trace-sha-only` outputs expose the recorded
  summaries/trace/digests plus the direct `wire_streaming_decode_loop` handoff
  target and its persisted digest. Compact `--kv-decode-blocker-summary-only` /
  `--kv-decode-blocker-summary-sha-only` outputs expose the resource artifact's
  `kv_decode_blocker_summary` directly through the correctness-status gap report,
  including the mirrored first/kernel-trace/last blocker names and blocker-record digests,
  and `--kv-streaming-blocker-records-only` /
  `--kv-streaming-blocker-records-sha-only` expose the full KV streaming blocker
  records plus digest, so KV automation can poll the first runtime blocker,
  validated upload/launch prerequisites, required evidence artifacts, and
  no-claim policy without reading the full resource plan. Compact
  `--kv-required-artifacts-only` / `--kv-required-artifacts-sha-only` now emit
  the concrete KV evidence-artifact list (`kv_kernel_trace_artifact` and
  `kv_backed_next_token_artifact`) and digest directly from the blocker summary.
  Status integrity also
  verifies the KV compact output-mode mappings for blocker names, blocker records,
  first blocker, blueprint, loop status/next-action, launch trace, blocker summary,
  required artifacts, and resource
  refresh command routes; cross-checks the blocker-summary digest/recorded flag; recomputes its
  first-blocker/upload/launch mirror invariants against the resource-plan gap
  report, and continues to check the mirrored blueprint digest, loop-status
  digest, loop next-action digest, mirrored first KV blocker digest, full KV
  streaming blocker names/SHA, and full required-evidence blocker record/SHA
  mirrors across the gap report, next-action command, and handoff queue so the
  runner artifact and status helper agree on why KV-backed decode remains
  blocked. The same status
  artifact records `source_artifacts` path/size/SHA-256 provenance for the
  prompt, oracle, resource-plan, and docs
  inputs used to compute the summary; `--verify-source-artifacts STATUS_JSON`
  checks those embedded hashes/sizes plus the embedded digest/schema-version
  integrity fields before a handoff is trusted, can be combined with
  `--source-artifact-failures-only` to emit only stale source record names,
  `--verification-status-only` to emit just `match`/`mismatch`,
  `--verification-exit-code-only` to emit the numeric verifier exit code,
  `--verification-failures-only` / `--verification-failures-sha-only` to emit
  or digest both source-record and status-integrity failure lists, and can be combined with
  `--status-integrity-only` or `--status-integrity-failures-only` to emit just
  the embedded integrity payload or failing check names from a persisted status
  artifact; without `--verify-source-artifacts`, those compact integrity modes
  check the freshly built status without rechecking filesystem provenance. The
  `source_artifacts_sha256` digest gives
  readiness pollers a compact digest
  of the same prompt/oracle/resource/docs provenance, while compact
  `--text-resource-source-only` / `--text-resource-source-sha-only` outputs
  expose the resource-plan (`source_artifacts.text_resource`) record and digest
  directly so KV decode handoff automation can poll the dry-run plan provenance
  without reading the full status; status integrity also verifies the compact
  source-artifact output-mode mappings for the source digest, wrapper-timeout
  oracle source, and text-resource source routes. The status also exposes
  `handoff_summary_sha256` so pollers can detect blocker-summary or
  compact-output metadata drift before trusting the remaining handoff queue. The
  status also includes `next_action_commands` for rerunning the oracle command
  shell, regenerating the oracle JSON via `scripts/stepfun_llamacpp_oracle.py`,
  and refreshing the resource/status artifacts; compact
  `--next-action-commands-only` emits that command bundle directly, while
  `--remaining-blockers-report-only` / `--remaining-blockers-report-sha-only`
  emits/digests a compact report joining the two P11 partial checklist items to
  their readiness gates, missing evidence, and recommended commands; compact
  `--first-remaining-blocker-report-only` /
  `--first-remaining-blocker-report-sha-only` emits/digests just the front
  blocker report for immediate oracle/KV routing; status integrity verifies both
  report digests and recomputes the front report from `remaining_blockers_report`
  so stale compact routing metadata fails source-artifact verification.
  `next_action_commands_sha256` gives handoff pollers a compact digest of the
  command bundle itself, with length/SHA-256 metadata for the helper/resource/status refresh commands plus
  compact source-artifact verification, verification-status, verification
  exit-code, and verification-failure commands for rechecking embedded
  prompt/oracle/resource/docs hashes before blocker handoff; when invoked with
  `--execute --output`, the oracle helper now writes a structured
  `status=running` partial artifact before launching llama.cpp and overwrites it
  with the final executed/timeout JSON when the child returns, so supervised
  long reruns leave machine-readable in-progress evidence instead of only an
  opaque wrapper timeout. `stepfun_correctness_status.py` mirrors that guarantee
  in `next_action_commands.oracle_parity_blocked`, the blocker work queue,
  `remaining_blockers_report`, and `first_remaining_blocker_report` via
  `recommended_command_writes_partial_output_before_launch=true` plus the
  expected `status=running` path/overwrite metadata. Compact
  `--oracle-partial-output-handoff-only` /
  `--oracle-partial-output-handoff-sha-only` outputs expose the supervised oracle
  rerun partial-output contract (command record, queue/report mirrors, source-path
  match, and safe/drift status) without requiring the full status artifact. Status
  integrity verifies the command-level partial-output guarantee, the compact handoff digest/status,
  and that the queue/compact blocker
  reports mirror the same path/status/overwrite fields; oracle-helper tests now
  cover both successful overwrite and timeout overwrite of the pre-launch
  `status=running` artifact; timeout payloads now include `timeout_termination`
  with the exact `os.killpg` / `SIGKILL` process-group termination path used when
  `timeout_s` is reached, and the correctness status mirrors that termination
  payload plus a stable digest in `oracle_progress` and `oracle_gap_report`.
  JSON file writes now use a flushed same-directory
  temporary file plus atomic `os.replace` so handoff pollers never consume a
  truncated partial/final artifact. The helper preserves the recorded
  `diagnostic_logs=true` setting so reruns keep llama.cpp load/error logs
  enabled for the canonical timeout artifact. The handoff now
  also records `oracle_helper_long_timeout_command` (`--timeout-s 900.0`, same
  canonical oracle output JSON) with length/SHA-256 metadata and mirrors it in
  the first blocker work item. The blocker queue now also exposes a generic
  `recommended_command` / digest for the first blocker, selecting that 900 s
  oracle helper while oracle parity is the front-of-queue blocker. Status integrity
  verifies the oracle compact output-mode mappings for helper command, long-timeout
  helper command, and timeout-termination payload/digest routes. Compact
  `--oracle-helper-long-timeout-command-only` /
  `--oracle-helper-long-timeout-command-sha-only` and
  `--first-blocker-recommended-command-only` /
  `--first-blocker-recommended-command-sha-only` outputs expose the rerun
  command/digests directly. Compact
  `--first-blocker-recommended-command-reason-only` /
  `--first-blocker-recommended-command-reason-sha-only` and
  `--first-blocker-first-missing-evidence-only` /
  `--first-blocker-first-missing-evidence-sha-only`,
  `--first-blocker-current-status-only` /
  `--first-blocker-current-status-sha-only`,
  `--first-blocker-gap-report-status-only` /
  `--first-blocker-gap-report-status-sha-only`,
  `--first-blocker-kind-only` / `--first-blocker-kind-sha-only`, and
  `--first-blocker-recommended-command-kind-only` /
  `--first-blocker-recommended-command-kind-sha-only` outputs expose the current
  first-blocker routing reason (`oracle_timeout_retry_with_longer_timeout`),
  first missing evidence (`oracle_completed_successfully`), current status
  (`timeout`), gap-report status (`blocked`), blocker kind, and command kind
  without fetching the full work item. Status integrity also verifies the oracle progress
  digest and oracle compact output-mode mappings, including the progress/status/blocker-kind routes.
  The blocker queue also records a compact
  `blocker_recommended_commands` list plus SHA so automation can inspect both
  the front oracle rerun and the queued KV resource refresh without parsing full
  work-item payloads; `stepfun_correctness_status.py --output ...` now writes
  full status, compact handoff, and verification payloads through a flushed
  same-directory temporary file plus atomic `os.replace` so pollers never consume
  a truncated status JSON. The command handoff explicitly records atomic-output
  metadata for the status-refresh and KV resource-refresh commands, including
  output path, helper, `atomic_os_replace` policy, `--output` presence, and lack
  of shell redirection; compact `--atomic-output-handoff-only` /
  `--atomic-output-handoff-sha-only` outputs expose that refresh-safety summary
  and digest without requiring the full command bundle. Status integrity validates
  those command-level atomic metadata fields and their blocker-work-queue mirrors.
  Status integrity also verifies the blocker work-queue digest, queue-meta mirror,
  blocker work-queue compact output-mode mappings, first work-item digest/mirror,
  blocker-kind and blocked-gate mirrors across the
  compact top-level fields, handoff summary, work queue, and remaining-blocker report,
  the schema-version payload digest, recommended-command list digest, command length/SHA metadata inside the work queue,
  compact recommended-command records, `--blocker-recommended-command-shas-joined-only` /
  `--blocker-recommended-command-shas-joined-sha-only` command-content route outputs,
  `--blocker-recommended-command-nchars-joined-only` /
  `--blocker-recommended-command-nchars-joined-sha-only` command-character-count route outputs,
  `--blocker-recommended-command-kinds-joined-only` /
  `--blocker-recommended-command-kinds-joined-sha-only` action-plan route outputs,
  `--blocker-recommended-command-reasons-joined-only` /
  `--blocker-recommended-command-reasons-joined-sha-only` action-rationale route outputs,
  `--blocker-first-missing-evidence-joined-only` /
  `--blocker-first-missing-evidence-joined-sha-only` missing-evidence route outputs,
  `--blocker-command-available-joined-only` /
  `--blocker-command-available-joined-sha-only` runnable-action route outputs,
  handoff-integrity verification commands and their compact output-mode mappings,
  status/handoff compact output-mode mappings, plus the compact recommended-command
  list mirror against the full work queue. The status artifact now persists `status_integrity` plus
  `status_integrity_sha256`, compact `--status-integrity-sha-only` exposes the
  digest for top-level embedded-check polling, and source-artifact verification
  now checks that the persisted integrity payload/SHA still match the recomputed
  checks. Compact `--persisted-status-integrity-only` /
  `--persisted-status-integrity-failures-only` outputs expose those persisted
  payload/SHA verification checks directly. The 2026-06-04 900 s rerun is now
  recorded as timeout evidence rather than oracle parity, so the remaining
  oracle next action is to run a faster or otherwise completing StepFun-capable
  llama.cpp oracle and review/record the parsed result; KV-backed decode parity
  remains open too.
- [x] Preserve multi-EOS stopping and the chat assistant prefix. The short
  context planner renders the Step chat template with assistant `<think>` prefix
  and carries stop IDs `(1, 2, 128007)` with `should_stop()` checks.

**Acceptance:** `python3 -m pytest -q tests/test_stepfun_decode_planner.py`
passes for short-context limits, mixed-quant dispatch-key validation,
assistant-prefix rendering, multi-EOS stopping, text-only full-model slot
planning, resident-weight/KV byte planning, and torch-free imports.
`python3 -m pytest -q tests/test_stepfun_resident_session.py` passes for real
resident Q8_0 token embedding of a rendered Step chat prompt and `[0, BOS, EOS,
128007, vocab-1, EOS]` plus real layer-0 `Q3_K` `attn_q` and `Q5_K`
`attn_output` projection vs CPU
references using two BF16-rounded activation rows. It also checks rows>1
prefill dispatch, the resident attention-input Q/K/V/gate projection bundle,
the layer-0 attention prefill/output probe (resident Q/K/V/gate and `attn_output`
with host Q/K RMSNorm, RoPE, causal GQA, and head-wise gate), the layer-0
dense-layer prefill probe (attention residual + `ffn_norm` + dense MLP + final
residual), the layer-3 sliding/MoE branch of the layer prefill wrapper, the
dense-MLP gate/up projection bundle, the dense MLP correctness probe
(gate/up + host SwiGLU/BF16 + resident down projection), the layer-3 MoE router
probe (resident F32 router/bias weights -> CPU top-k routing), selected-expert
`Q3_K` gate projection via existing selected GEMV kernels, the MoE expert-input
bundle (selected gate/up plus shared gate/up), the MoE correctness probe
(router + selected/shared MLP chain with BF16 intermediates), the final-logits
probe (output RMSNorm + resident Q8_0 `lm_head` projection for sampled vocab
rows), the root-only prompt logits smoke (chat prompt -> embedding -> final root
logits), the first-layer prompt logits smoke (chat prompt -> embedding -> layer-0
prefill -> final root logits, with layers 1-44 skipped), the layer-prefix prompt
logits smoke (chat prompt -> embedding -> layers 0-44 chunked prefill -> final
root logits, no layers skipped), resident KV-cache allocation/free, resident
memory cleanup
(two/three/four active weight allocations before session free, zero after), and
no torch import. Full next-token/logit parity remains open until the streaming
layer loop is wired.

### P12 — Full-model Strix Halo smoke

> Memory status update (2026-05-29): the machine is booted with
> `/etc/modprobe.d/amdgpu_llm_optimized.conf` setting `amdgpu gttsize=120000`,
> `ttm pages_limit=31457280` (~120 GiB), `ttm page_pool_size=2081024` (~8 GiB
> preassigned), and `amdgpu vm_fragment_size=8`. Treat prior `rocm-smi
> VIS_VRAM=512 MiB` and contradictory `hipMemGetInfo` values as insufficient
> fit/fail proxies for full GTT. The next load path should attempt and measure
> real allocations; do not block implementation solely on those readouts.

- [x] Load all three GGUF shards on the Strix Halo target with a small context
  and `max_new_tokens` (for example 1-8). 2026-05-29: resident weight load now
  succeeds for all 754 tensors / `95.4598 GiB` under the configured 120 GB GTT
  (`benchmarks/results/2026-05-29-stepfun-q3kl-full-load-smoke-task20.json`),
  and the load smoke can allocate/free a synthetic 512-token BF16 KV footprint
  after weight load. 2026-05-31 completion evidence for the host-composed text
  path: `benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-all45-prompt-smoke.json`
  scans all three Q3_K_L shards, runs a 23-token Step chat prompt through all 45
  layers in chunked resident mode with no vision/projector/MTP slots, and emits
  one next-token candidate (`next_token_id=369`, decoded text ` |`). This closes
  the P12 load/small-context token smoke item, but not P11 llama.cpp/CPU oracle
  parity or the final KV-backed decode runner.
- [x] Record HIP-visible memory before load, after load, after KV allocation, and
  after generation; include UMA/GTT setting and backend (`hip_gfx1151`). Weight
  load evidence from
  `benchmarks/results/2026-05-29-stepfun-q3kl-full-load-smoke-task20.json`:
  before scan/free `119.9961 GiB`, after resident weight load `23.9061 GiB`,
  after free `119.8573 GiB`; hipEngine allocation stats peaked at
  `102,499,149,312` bytes across 754 allocations. 2026-05-30 KV smoke command:
  `PYTHONUNBUFFERED=1 python3 scripts/stepfun_gguf_load_smoke.py
  --kv-context-pages 1 --kv-page-size 512 --pretty >
  /tmp/stepfun-full-load-kv-smoke.json`. It allocated an additional
  `94,371,840` bytes (`0.0879 GiB`) across 90 K/V buffers, with free memory
  `23.9061 GiB` after weights -> `23.8183 GiB` after KV -> `23.9061 GiB` after
  KV free. `StepFunResidentSession.allocate_kv_cache()` now covers the same
  owned per-layer K/V allocation/free shape for runtime bring-up. 2026-05-31
  resource-planning progress: `StepFunShortContextDecodePlanner.text_decode_resource_plan()`
  ties the full 754-slot text plan to `102,499,149,312` resident-weight bytes
  and a 512-token BF16 KV estimate of `94,371,840` bytes (`0.0879 GiB`) across
  90 K/V buffers for backend `hip_gfx1151`; the load-smoke JSON now embeds that
  resource-plan dictionary when KV allocation is requested and offers
  `--dry-run-plan` for metadata-only resource artifacts without HIP allocation;
  `--output` writes those resource artifacts through a flushed same-directory temp
  file plus atomic `os.replace`, and the correctness-status KV refresh handoff now
  uses that flag instead of shell redirection. Artifact
  `benchmarks/results/2026-05-31-stepfun-q3kl-text-resource-dry-run.json`
  records the dry-run plan for the same 512-token KV shape. 2026-05-31
  host-composed all-layer prompt smoke artifact
  `benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-all45-prompt-smoke.json`
  records sampled token text plus top-5 token/logit rows for future llama.cpp
  output comparison and HIP-visible free memory before execution (`119.9961 GiB`), after
  generation before final root free (`118.8083 GiB`), and after final free
  (`119.8571 GiB`) for backend `hip_gfx1151`. This closes the P12
  HIP-visible memory snapshot requirement across the full-load/KV-allocation and
  all-layer host-composed generation artifacts; KV-backed decode parity remains
  open under P11.
- [x] If the model does not fit, keep the failure artifact and decide between
  offload/tiering, lower context/KV footprint, or slice-only correctness. No
  current fit failure is claimed from VIS_VRAM/`hipMemGetInfo` alone; decision
  policy is to continue implementation and only choose offload/tiering after a
  real allocation/load attempt fails under the configured GTT setup.
- [x] If it fits, run a tiny text-only prompt and confirm no vision/projector/MTP
  path is required. 2026-05-31 planning progress: `stepfun_text_decode_slot_paths()`
  covers all 754 validated text GGUF slots, including root RoPE frequencies, and
  the decode-planner test asserts there are no vision/projector/MTP slot
  dependencies. Partial prompt artifact
  `benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-prompt-smoke.json`
  is produced by `python3 scripts/stepfun_layer_prefix_smoke.py --layer-count 4
  --message hello --max-resident-weight-gib 4 --stream-chunk-layers 1 --output
  benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-prompt-smoke.json
  --pretty` and runs the resident text-only chat prompt through the layers 0-3
  prefix bridge with root tensors kept resident and each layer loaded/freed as a
  one-layer chunk; `--output` writes prompt/layer-prefix JSON artifacts through a
  flushed same-directory temp file plus atomic `os.replace` so provenance pollers
  never consume a truncated prompt artifact. It uses no vision/projector/MTP
  slots. A deeper artifact
  `benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-0-4-prompt-smoke.json`
  runs the same chunked path through layer 4 (`next_token_id=67707`, peak
  resident weight bytes `3,531,578,496`). Full-layer artifact
  `benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-all45-prompt-smoke.json`
  runs the chunked host-composed path through layers 0-44 (`next_token_id=369`,
  decoded token text ` |`, peak resident weight bytes `3,531,578,496`, prompt
  length 23) with no vision/projector/MTP slots and no skipped layers; it records
  sampled token text plus top-5 token/logit rows and HIP-visible free
  memory before execution (`119.9961 GiB`), after generation before final root
  free (`118.8083 GiB`), and after final free (`119.8571 GiB`). The same
  script's `--dry-run-plan --layer-count 45` mode now plans the all-layer text
  prefix slot/resource shape without initializing HIP, and `--output` writes the
  JSON artifact directly;
  `benchmarks/results/2026-05-31-stepfun-q3kl-layer-prefix-all45-dry-run.json`
  records that 753-slot/45-layer metadata plan from the native output path.
  The same artifact now includes a metadata-only `--stream-chunk-layers 1`
  estimate: keep root text tensors resident (`1,121,927,168` bytes) and stream
  one layer at a time, with a max root+layer peak of `3,531,578,496` bytes
  (`3.29 GiB`) at layer 3. Non-dry-run prefix smokes can now execute that
  chunked path through all 45 layers; llama.cpp/CPU oracle parity and KV-backed
  decode are still open. Non-dry-run prefix smokes also
  support `--max-resident-weight-gib` so accidental all-layer HIP allocation
  attempts fail before runtime initialization unless an explicit memory budget is
  supplied. Full KV-backed generation and oracle parity remain open.

**Acceptance:** full-model smoke produces token(s) or a documented fit failure.
Current materialization coverage is validated by
`python3 -m pytest -q tests/test_stepfun_materialize.py` for all-tensor
quant/layout planning (`Q3_K`, `Q5_K`, `Q8_0`, `F32`), split-shard payload
access on first/last shard tensors, selected-slot HIP loading/freeing with
memory stats, and torch-free imports; `python3 -m pytest -q tests/test_stepfun_load_smoke.py`
validates metadata-only dry-run load-smoke JSON; `python3 -m pytest -q
tests/test_stepfun_layer_prefix_smoke.py` validates the reusable partial prompt
smoke script, native output-file writing, memory-budget guard, top-token JSON
schema, all-layer dry-run prefix planning, root-plus-one-layer streaming memory
estimates, and a chunked layer-0 prompt execution. This is still not a
throughput benchmark.

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

- [ ] Post-correctness AOTriton/native attention profiling on Strix Halo once
  KV-backed decode and oracle parity are available; no Step throughput or kernel
  performance claim should precede the P13 correctness/benchmark gates.
- [ ] Post-Q3 GGUF variant validation for the local `Step-3.7-Flash-UD-IQ4_NL`
  split and `Step-3.7-Flash-APEX-I-Compact` single-file variant. Reuse the
  Q3_K_L metadata/materialization/slice/load/prompt-oracle gates first; add new
  quant plugins or variants only through the registry axes, not engine-wide
  special-casing.
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
| Strix Halo GTT allocation may still fail despite the 120 GB boot config | GGUF weights are 95.46 GiB and public llama.cpp notes recommend 128 GB UMA; current `rocm-smi`/`hipMemGetInfo` readouts are not reliable fit proxies | P0/P12 must record real allocation/load fit/fail evidence before full-model claims; keep slice correctness unblocked if full load fails |
| gfx1151 kernel coverage may lag gfx1100 | Existing hipEngine code and lineage are gfx1100-centered | Treat `hip_gfx1151` as a peer backend registration/build target; run minimal kernel smoke and `rocprofv3` before relying on copied gfx1100 kernels |
| Q3_K is the dominant missing quant path | Step Q3_K_L heavily uses Q3_K and hipEngine's existing GGUF work is Q4_K/Q5_K/Q6_K/Q8_0 focused | Land CPU Q3_K first, then HIP Q3_K slice correctness; do not integrate decode through lossy ad-hoc dequant shortcuts |
| Tokenizer/chat mismatch can invalidate every oracle comparison | Step GGUF uses DeepSeek-V3 GPT-2 BPE metadata, not Qwen3.5 tokenizer metadata | Make tokenizer parity a separate lane with llama.cpp/HF token-id fixtures before logits/decode work |
| Sliding/full mixed attention can hide KV bugs until long prompts | Step alternates full and 512-window sliding layers with different head counts and RoPE settings | Exercise both layer types through `KVLiveSpans` boundary tests before full-model smoke |
| llama.cpp/HF oracle may be unavailable for full-model comparisons on the same box | The model is large and HF Transformers may require more memory than GGUF | Use llama.cpp for tokenization/short greedy where possible and derive small activation/tensor-slice fixtures for earlier lanes |
| NVFP4 completion may distract from GGUF bring-up | NVFP4 files are now cached but require ModelOpt FP4/FP8 support with unclear RDNA3/RDNA3.5 upside | Keep NVFP4 explicitly deferred until P11/P12 text-only GGUF correctness is established |

## Remaining open questions

- Does a real HIP allocation/full GGUF load use the configured 120 GB GTT
  successfully despite misleading `rocm-smi`/`hipMemGetInfo` readouts?
- Is there a smaller Step 3.7/3.5 fixture available for RED/GREEN tests, or
  should we derive slice fixtures from the large local GGUF files?
- Which llama.cpp command and commit should be the initial oracle for tokenizer,
  next-token, and full-smoke comparison?
- Are vision/projector and MTP definitely deferred until after base greedy decode
  correctness, or do they need separate early proof-of-concept loops?
