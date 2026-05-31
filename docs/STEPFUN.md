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
  stop IDs, KV dispatch keys, and resource-fit booleans), plus the planned per-layer KV launch
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
  emits just that readiness-summary digest for drift polling, `--oracle-helper-command-only`
  emits just the oracle JSON refresh command, `--oracle-helper-command-sha-only` emits its
  digest, `--summary-only`
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
  `--fail-on-blocked` for CI/handoff checks, including compact readiness/queue/oracle-helper/first-blocker outputs;
  the handoff also records the ready/source-mismatch/blocked exit-code expectations.
  This checklist item is partial rather than complete because the current runner
  is host-composed/chunked and does not yet implement the final KV-backed
  one-token decode path.
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
  version `9197 (fcae601e4)`) accepts `step35` but the bounded CPU/no-GPU oracle
  attempt in
  `benchmarks/results/2026-05-31-stepfun-q3kl-llamacpp-step35-timeout.json`
  timed out after a bounded 60 s attempt (recorded `elapsed_s=62.44`) before
  producing a comparable token (`oracle_blocker_kind=llama_cpp_oracle_timeout`).
  A default-device Vulkan oracle attempt is recorded separately in
  `benchmarks/results/2026-05-31-stepfun-q3kl-llamacpp-vulkan-harness-timeout.json`;
  it exceeded the pi tool supervision window before helper JSON was produced, so
  the CPU/no-GPU timeout artifact remains the canonical machine-readable oracle
  blocker for now. The consolidated correctness-status artifact surfaces the oracle version,
  elapsed time, stdout/stderr lengths, `oracle_progress` fields, and an `oracle_gap_report`
  that separates recorded deterministic-target prerequisites from missing run/match evidence
  for the exact deterministic target (`prompt_length=23`, `n_predict=1`, expected token id 369 /
  text ` |`, top-5 expected tokens, timeout 60 s, elapsed 62.44 s, generated text
  length 0, and the llama.cpp command shell). It records both blockers
  (`oracle_parity_blocked`, `kv_backed_decode_not_wired`) for the current
  all-layer prompt smoke, lists machine-readable next actions for each blocker,
  and now includes `readiness_gates` for `oracle_parity`, `kv_backed_decode`,
  and `e2e_inference` so each false readiness boolean carries required evidence
  and current blocker state. The same status artifact records `source_artifacts`
  path/size/SHA-256 provenance for the prompt, oracle, resource-plan, and docs
  inputs used to compute the summary; `--verify-source-artifacts STATUS_JSON`
  checks those embedded hashes/sizes against the current files before a handoff is
  trusted. The status also includes `next_action_commands` for rerunning the
  oracle command shell, regenerating the oracle JSON via `scripts/stepfun_llamacpp_oracle.py`,
  and refreshing the resource/status artifacts, with length/SHA-256 metadata for
  the KV resource-plan refresh command and the shared status-refresh command.
  Remaining implementation task: run a longer/faster
  StepFun-capable llama.cpp oracle and review/record the parsed result;
  KV-backed decode parity remains open too.
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
  `--dry-run-plan` for metadata-only resource artifacts without HIP allocation.
  Artifact `benchmarks/results/2026-05-31-stepfun-q3kl-text-resource-dry-run.json`
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
  one-layer chunk; it uses no vision/projector/MTP slots. A deeper artifact
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
