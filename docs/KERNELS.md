---
status: current
owns: Kernel source catalog, model/quant and registry mappings, arithmetic variants, and fused fallback map.
---
# Kernel catalog

Kernel families, source locations, and what they do. Keep each description to
one short sentence. Implementation history, measurements, tuning decisions,
and work-in-progress notes belong in `worklog/entries/` or `benchmarks/results/`,
not here. General porting/runtime guidance lives in
[LESSONS-LEARNED.md](LESSONS-LEARNED.md); architecture-specific guidance lives in
[RDNA3-TUNING-GUIDE.md](RDNA3-TUNING-GUIDE.md). Validation rules live in
[OPTIMIZATION.md](OPTIMIZATION.md).

Paths are relative to the backend directory shown in each section. Device
sources generally have same-name Python build and launch wrappers. Exact
variants and backend availability are defined by the kernel registry and
backend packages, not this catalog.

## Backend and source map

```text
hipengine/kernels/
├── registry.py                 # (backend, layer, quant, variant)
├── backends.py                 # Backend package loading and selection
├── cpu_reference/              # NumPy oracles
├── hip_gfx1100/                # HIP device sources and Python wrappers
│   ├── attention/              # Attention and key/value cache operations
│   ├── convert/                # Casts and row gathers
│   ├── dispatch/               # Native launch dispatch
│   ├── fused/                  # Composite and elementwise operations
│   ├── linear/                 # Dense projections and output heads
│   ├── linear_attn/            # Convolution and gated delta recurrence
│   ├── moe/                    # Routing, grouping, and expert combination
│   ├── norm/                   # Normalization
│   ├── quant/                  # Quantized projections and format conversion
│   ├── rotary/                 # Rotary transforms
│   ├── runtime/                # Device state and launch batching
│   ├── sampling/               # Token sampling
│   ├── speculative/            # Drafting, acceptance, and state commit
│   ├── evie/, surya/, vision/  # Vision and OCR
│   ├── vibevoice/              # Speech encoders, decoders, and diffusion
│   ├── yue2/                   # Music-generation attention, solver, and audio decoder
│   ├── timesfm/, timesfm3/     # Forecasting
│   ├── wmma/                   # Matrix-tiled PARO projections
│   └── smoke/                  # Build/runtime probes
├── hip_gfx1151/                # Peer registrations for shared gfx11 sources
├── cuda_sm120a/                # Independent CUDA sources and wrappers
│   ├── attention/, encoder/   # Maple/Moonshine attention and speech encoder
│   ├── fused/, linear/, norm/ # Decoder operations
│   ├── moe/, quant/           # Expert operations and packed projections
│   └── smoke/                 # Build/runtime probes
└── cuda_sm86/                  # Scaffold
```

| Backend | Source ownership | Model/family coverage |
| --- | --- | --- |
| `cpu_reference` | Python/NumPy oracles | Shared primitives and model references |
| `hip_gfx1100` | Native HIP sources | Qwen/PARO/GGUF, Laguna, Maple, Moonshine, speech, vision, forecasting, speculation |
| `hip_gfx1151` | Shared HIP sources, peer registrations | Supported subsets of the gfx11 families |
| `cuda_sm120a` | Native CUDA sources | Maple, Moonshine, and shared helpers |
| `cuda_sm86` | Package scaffold | No device kernels |

## Registry-layer map

Keys are `(backend, layer, quant, variant)`. This table groups layer names;
individual variants and storage formats remain defined in their wrappers.
Paths below refer to the HIP source families unless a backend is named.

| Layer family | Source family | Operation |
| --- | --- | --- |
| `cast_*`, `gather_f32_rows_by_i32id` | `convert/` | Convert or gather rows. |
| `rmsnorm`, `add_rmsnorm`, `head_rmsnorm` | `norm/rmsnorm`, `fused/gguf_ops` | Normalize activations. |
| `paro_rotate1/2/3`, `partial_rotary`, `split_qgate` | `rotary/` | Rotate activations and split query/gate planes. |
| `dense_gemv`, `linear`, `linear_pair/triple/quad` | `linear/`, `quant/` | Project dense or quantized weights. |
| `pack8_gemv`, `selected_*pack8_gemv`, `pack8_gemm` | `quant/paro_awq_gemv` | Project PARO packed weights. |
| `lm_head`, `lm_head_argmax`, `argmax`, `topk` | `linear/lm_head` | Project or select output tokens. |
| `router_logits`, `router_select`, `router_topk_*` | `moe/router` | Select experts and route weights. |
| `moe_group_*`, `moe_gather_packed_hidden`, `moe_*tile_map` | `moe/group_scatter` | Group and pack expert work. |
| `moe_linear`, `moe_linear+weighted_sum` | `quant/gguf_*` | Project selected experts and combine outputs. |
| `moe_ffn_selected` | `quant/paro_moe_ffn_fused`, `quant/gguf_q4_k_moe_ffn_fused` | Execute a fused expert feed-forward chain. |
| `weighted_sum`, `shared_gate_combine` | `fused/paro_combine` | Combine expert outputs. |
| `embedding` | `quant/gguf_q6_k_embedding`, `quant/gguf_iq_dense` | Look up quantized token rows. |
| `activation_quant`, `weight_pack` | `quant/gguf_*` | Prepare packed projection operands. |
| `paged_kv_write`, `paged_kv_copy` | `attention/paged_kv_write` | Update paged caches. |
| `full_attn_*`, `paged_attn_*` | `attention/paged_attn_decode` | Compute attention. |
| `dms_*` | `attention/dms_compact*` | Maintain and attend over compact caches. |
| `linear_attn_*conv_*`, `gdn_*recurrent*` | `linear_attn/` | Update convolutional and recurrent state. |
| `qsa_*` | `attention/qwen4_exp_qsa` | Select blocks and compute sparse attention. |
| `sampler`, `mtp_draft_topk` | `sampling/sampler` | Sample tokens or select draft candidates. |
| `dflash_*`, `speculative_accept_commit`, `mtp_nextn_*` | `speculative/` | Propose, verify, and commit draft tokens. |

## CPU reference

Source: `hipengine/kernels/cpu_reference/`. These are NumPy reference operations.

| Source | Purpose |
| --- | --- |
| `ops.py` | Shared projection, normalization, rotary, attention, quantization, recurrent-state, and expert operations. |
| `dflash2.py` | Dynamic convolution, candidate selection, attention, and rotary operations for DFlash2. |
| `dms.py` | Compact key/value cache packing, eviction, and attention. |
| `evie.py` | Evie vision and language operations. |
| `laguna.py` | Laguna attention, routing, feed-forward, and draft-model operations. |
| `maple.py` | Ternary/affine4 projections, attention, and expert operations. |
| `moonshine.py` | Moonshine decoder projections, normalization, attention, and cache operations. |
| `moonshine_encoder.py` | Moonshine encoder convolution, normalization, and attention. |
| `qwen2.py` | Qwen2 decoder operations. |
| `qwen4_exp.py` | Qwen4Exp branch mixing, positional embeddings, sparse attention, recurrence, and experts. |
| `surya.py` | Surya OCR vision and text operations. |
| `timesfm.py` | TimesFM 2.5 normalization, attention, patch processing, and forecasting. |
| `timesfm3.py` | TimesFM 3.0 sequence/variate attention and multivariate forecasting. |
| `vibevoice_asr.py` | VibeVoice speech-recognition operations. |
| `vibevoice_tts.py` | VibeVoice speech-generation operations. |
| `vibevoice_tts_diffusion.py` | VibeVoice diffusion-head operations. |
| `yue2.py` | YuE2 normalization, rotary, attention, solver, and audio-decoder reference operations. |

## HIP gfx11

Device sources: `hipengine/kernels/hip_gfx1100/`.
`hipengine/kernels/hip_gfx1151/` registers supported shared sources for native
gfx1151 compilation; it does not enable every gfx1100 variant.

### Conversion, normalization, and rotary

| Source | Purpose |
| --- | --- |
| `convert/cast.hip` | Convert floating-point storage formats and scale rows. |
| `convert/gather.hip` | Gather rows by integer index. |
| `norm/rmsnorm.hip` | Root-mean-square normalization and residual/head variants. |
| `rotary/paro_rotate.hip` | PARO rotations and fused normalization/rotation. |
| `rotary/qwen35_rotary.hip` | Qwen partial rotary embeddings and query/gate splitting. |
| `fused/gguf_ops.hip` | GGUF normalization, head rotary, and attention-gate composites. |

### Dense projections and expert operations

| Source | Purpose |
| --- | --- |
| `linear/dense_gemv.hip` | Dense matrix-vector projections and paired/residual variants. |
| `linear/lm_head.hip` | Vocabulary projection, argmax, and top-k reductions. |
| `quant/paro_awq_gemv.hip` | PARO packed 4-bit projections and selected-expert variants. |
| `quant/paro_marlin_k.hip` | PARO decode projection using the Marlin-K layout. |
| `wmma/paro_awq_wmma.hip` | Matrix-tiled PARO prefill projections. |
| `quant/w8a16_linear.hip` | 8-bit-weight, 16-bit-activation projections and shared-expert helpers. |
| `quant/paro_moe_ffn_fused.hip` | Fused PARO selected-expert feed-forward chain. |
| `moe/router.hip` | Expert router logits, top-k selection, and shared gates. |
| `moe/group_scatter.hip` | Group expert assignments and pack rows and tile metadata. |
| `moe/prefill.py` | Compose selected-expert prefill operations. |
| `dispatch/moe_c1_dispatch.hip` | Dispatch single-token expert operations through native function pointers. |
| `fused/paro_silu.hip` | SiLU activation/product and fused down-projection rotation. |
| `fused/paro_combine.hip` | Combine routed/shared experts with residual and normalization variants. |

### Attention and state

| Source | Purpose |
| --- | --- |
| `attention/paged_kv_write.hip` | Write and copy paged key/value caches. |
| `attention/paged_attn_decode.hip` | Dense/paged attention for decode and prefill, including quantized caches. |
| `attention/aotriton.py`, `attention/aotriton_wrap.py` | Adapt the optional AOTriton attention library. |
| `attention/dms_compact.hip` | Select, pack, append, and attend over compact key/value caches. |
| `attention/dms_compact_int8.hip` | Pack, append, and attend over compact INT8 key/value caches. |
| `linear_attn/conv.hip` | Causal convolution with prefill, decode, and state snapshots. |
| `linear_attn/gdn.hip` | Gated delta recurrence, output normalization, and state snapshots. |
| `runtime/state.hip` | Device token, position, graph, and commit bookkeeping. |
| `sampling/sampler.hip` | Greedy and probabilistic token sampling and draft top-k selection. |
| `smoke/smoke_add.hip` | Vector addition for build/runtime smoke tests. |

### GGUF projections and quantization

| Source | Purpose |
| --- | --- |
| `quant/gguf_k_gemv.hip` | Raw Q5_K, Q6_K, and Q8_0 projections, including selected experts. |
| `quant/gguf_q3_k_gemv.hip` | Raw Q3_K selected-expert projections. |
| `quant/gguf_q4_k_gemv.hip` | Q4_K projections with paired, activation, and residual composites. |
| `quant/gguf_q4_k_moe_ffn_fused.hip` | Fused Q4_K selected-expert feed-forward chain. |
| `quant/gguf_q4_k_prefill.hip` | Matrix-tiled Q4_K/Q6_K prefill projections. |
| `quant/gguf_q4_k_selected_prefill.hip` | Q4_K selected-expert prefill projections. |
| `quant/gguf_k_selected_prefill.hip` | Raw Q5_K/Q6_K selected-expert prefill projections. |
| `quant/gguf_q8_0_prefill.hip` | Grouped Q8_0 expert-down prefill projections. |
| `quant/gguf_expert_pack8_gemv.hip` | Packed selected-expert projections. |
| `quant/gguf_k_selected_pack8_gemv.hip` | Packed Q5_K/Q6_K selected-expert projections. |
| `quant/gguf_q4_k_selected_pack8_gemv.hip` | Packed Q4_K selected-expert projections. |
| `quant/gguf_q4_k_pack8_gemv.hip` | Packed Q4_K matrix-vector projections. |
| `quant/gguf_q6_k_pack8_gemv.hip` | Packed Q6_K matrix-vector projections. |
| `quant/gguf_q8_0_pack8_gemv.hip` | Packed Q8_0 matrix-vector projections. |
| `quant/gguf_q6_k_t16_gemv.hip` | T16/qmicro Q6_K projections and head/residual composites. |
| `quant/gguf_t16_selected_gemv.hip` | T16 selected-expert projections and weighted/residual composites. |
| `quant/gguf_k_t16_selected_prefill.hip` | T16 Q5_K/Q6_K selected-expert prefill projections. |
| `quant/gguf_q4_k_t16_selected_prefill.hip` | T16 Q4_K selected-expert prefill projections. |
| `quant/gguf_q5_k_qmicro_planar_gemv.hip` | Planar qmicro Q5_K selected-expert projections. |
| `quant/gguf_q8_0_t16_gemv.hip` | T16 Q8_0 decode projections. |
| `quant/gguf_q8_0_t16_prefill.hip` | T16 Q8_0 prefill projections. |
| `quant/gguf_q8_0_raw_to_t16.hip` | Repack raw Q8_0 weights into T16 storage. |
| `quant/gguf_iq_dense.hip` | Raw IQ/Q3 dense projections and Q3_K embedding lookup. |
| `quant/gguf_iq_gemv.hip` | Raw IQ selected-expert projections. |
| `quant/gguf_iq_selected_prefill.hip` | IQ selected-expert prefill projections. |
| `quant/gguf_iq_wmma_prefill.hip` | Matrix-tiled raw IQ dense prefill projections. |
| `quant/gguf_k_mmq_prefill.hip` | Activation quantization and integer Q5_K/Q6_K prefill projections. |
| `quant/gguf_iq_source_mmq_prefill.hip` | Integer IQ selected-expert prefill projections. |
| `quant/gguf_iq2_xs_mmq_prefill.hip` | Integer IQ2_XS prefill projections. |
| `quant/gguf_q4_k_q8_1_mmq_prefill.hip` | Diagnostic integer Q4_K prefill projections. |
| `quant/gguf_q4_k_q8_1_dp4a_vdr_gemv.hip` | Diagnostic Q4_K decode projections using packed integer dot products. |
| `quant/gguf_q4_k_q8_1_selected_prefill.hip` | Q8_1 activation packing and integer Q4_K/Q6_K projections. |
| `quant/gguf_q4_k_qmicro_dp4a_grouped.hip` | Grouped qmicro Q4_K projections using packed integer dot products. |
| `quant/gguf_q5_1_mmq_selected_prefill.hip` | Integer Q5_1 selected-expert prefill projections. |
| `quant/gguf_q5_k_q8_1_selected_prefill.hip` | Integer Q5_K selected-expert prefill projections. |
| `quant/gguf_q8_0_mmq_prefill.hip` | Integer Q8_0 prefill projections and weight packing. |
| `quant/gguf_q8_0_dp4a_gemv.hip` | Q8_0 projections using packed integer dot products. |
| `quant/gguf_q5_k_f32_rocblas_prefill.hip` | Expand quantized weights to FP32 for prefill consumers. |
| `quant/gguf_q6_k_f16_rocblas_prefill.hip` | Dequantize Q4/Q5/Q6 tiles for FP16 rocBLAS projections. |
| `quant/gguf_q6_k_embedding.hip` | Raw GGUF embedding lookup. |
| `quant/gguf_x8_selected_gemv.hip` | X8 packed selected-expert projections and head helpers. |
| `fused/gguf_q6_q4_pair.hip` | Paired Q6_K/Q4_K projections. |

### Qwen4Exp

| Source | Purpose |
| --- | --- |
| `fused/qwen4_exp_gr.hip` | Gated branch reads, writes, and mixing. |
| `fused/qwen4_exp_ple.hip` | Positional-embedding gating, convolution, and addition. |
| `linear_attn/qwen4_exp_gdn.hip` | Gated delta recurrence with sigmoid output gating. |
| `attention/qwen4_exp_qsa.hip` | Sparse-attention rotary transforms, block pooling/scoring/selection, and attention. |
| `attention/qwen4_exp_qsa_flash.hip` | Flash-style dense prefill attention. |
| `quant/qwen4_exp_q5_1.hip` | Raw Q5_1 selected-expert projections. |
| `vision/qwen4_exp_vision.hip` | Vision normalization, activation, residual, and attention operations. |

### Laguna

| Source | Purpose |
| --- | --- |
| `linear/laguna_f16_projection.hip` | FP16-weight projections and fused residual/normalization. |
| `moe/laguna_router.hip` | Expert routing and weighted route combination. |
| `attention/laguna_kv_attention.hip` | Key/value writes, rotary transforms, and global/sliding-window attention. |
| `attention/laguna_flash_attention_prefill.hip` | Matrix-tiled prefill attention. |
| `fused/laguna_attention.hip` | Softplus/sigmoid attention output gating. |
| `runtime/laguna_launch_batch.hip` | Batch projection and expert-tail launches in native code. |

### Maple

| Source | Purpose |
| --- | --- |
| `quant/maple_ternary.hip` | Ternary projections and affine4 embedding/head operations. |
| `attention/maple_attention.hip` | Query/key normalization, rotary transforms, cache writes, and attention. |
| `moe/maple_moe.hip` | Expert selection, clamped SwiGLU, and weighted residuals. |

### Moonshine

| Source | Purpose |
| --- | --- |
| `linear/moonshine_projection.hip` | FP16 decoder projections and fused projection boundaries. |
| `linear/moonshine_w8a16.hip` | 8-bit-weight decoder projections. |
| `norm/moonshine_layernorm.hip` | Layer normalization and residual/normalization. |
| `fused/moonshine_glue.hip` | Embedding, residual, rotary, cache, and argmax operations. |
| `fused/moonshine_mlp.hip` | Gated SiLU activation. |
| `attention/moonshine_attention.hip` | Decoder self-attention and cross-attention. |

### Speech, vision, OCR, and forecasting

| Source | Purpose |
| --- | --- |
| `vibevoice/encoder.hip` | Speech encoders/connectors and Qwen2 attention/cache operations. |
| `vibevoice/decoder.hip` | Streaming speech-decoder convolutions and upsampling. |
| `vibevoice/diffusion.hip` | Diffusion-head normalization, modulation, and solver steps. |
| `evie/evie_ops.hip` | Vision patch embedding, normalization, attention, and merger operations. |
| `surya/surya_ops.hip` | Surya normalization, query/gate splitting, cache writes, and attention. |
| `timesfm/timesfm.hip` | TimesFM normalization, rotary transforms, attention, and layout operations. |
| `timesfm3/timesfm3.hip` | TimesFM 3.0 variate attention, query/key normalization, and ReLU. |

### YuE2 music generation

| Source / shared family | Purpose |
| --- | --- |
| `yue2/nar.hip` | Non-autoregressive attention, rotary transforms, cache gathering, step embeddings, and midpoint solver updates. |
| `yue2/nar_wmma.hip` | Matrix-tiled non-autoregressive attention with changed arithmetic. |
| `yue2/vae.hip` | FP32 audio-decoder convolutions, transposed convolutions, Snake activation, and residuals. |
| `vibevoice/encoder.hip` | Shared normalization, rotary, span-cache writes, attention, and residual operations. |
| `linear/dense_gemv.hip` | Shared autoregressive projections and output head, including exact paired-branch row tiles and phase-windowed output. |
| `rotary/qwen35_rotary.hip`, `fused/paro_silu.hip` | Shared decode rotary and gated activation operations. |

### Speculative decoding

| Source | Purpose |
| --- | --- |
| `speculative/dflash_drafter.hip` | DFlash draft-model projections, normalization, attention, and metadata. |
| `speculative/dflash2.hip` | DFlash2 dynamic convolution, top-k, and candidate selection. |
| `speculative/dflash_accept.hip` | Draft-chain acceptance and commit summaries. |
| `speculative/dflash_commit.hip` | Commit selected recurrent states and cursors. |
| `speculative/mtp.hip` | Multi-token prediction proposal, routing, and acceptance helpers. |
| `speculative/mtp_nextn.hip` | NextN draft-layer projections, attention, and expert operations. |
| `speculative/sampled_accept.hip` | Probabilistic draft-chain acceptance and residual sampling. |

## Exact and production implementation map

`strict` names an exact or parent-parity contract; `production` is a selection
profile, not a synonym for approximate math. Production can select exact
kernels too. The profile manifest identifies each selected variant and its
strict fallback; backend packages supply shape-specific choices. See
[EXECUTION-PROFILES.md](EXECUTION-PROFILES.md) for the numerical contracts.

| Model / quant family | Strict or unfused implementation | Alternate / production implementation | Source family |
| --- | --- | --- | --- |
| Qwen/PARO `w4_paro` | Pack8 projections, separate rotation/SiLU/combine | Fused rotation/projection and selected feed-forward chains; matrix-tiled prefill | `quant/paro_awq_gemv`, `quant/paro_moe_ffn_fused`, `wmma/paro_awq_wmma` |
| GGUF Q4/Q5/Q6 T16 | Scalar/row-tiled projections and primitive residual/weighted sums | Matrix-tiled prefill, dual+SiLU, weighted-down and residual composites | `quant/gguf_t16_selected_gemv`, `quant/gguf_k_t16_selected_prefill`, `quant/gguf_q4_k_t16_selected_prefill` |
| GGUF Q4/Q5/Q6 library routes | Exact raw/T16 projections | Dequantized FP16 rocBLAS and activation-quantized integer prefill | `quant/gguf_q6_k_f16_rocblas_prefill`, `quant/gguf_k_mmq_prefill`, `quant/gguf_q4_k_q8_1_selected_prefill` |
| GGUF Q8_0 | Raw/T16 GEMV and exact row-batched variants | Matrix-tiled and integer prefill; packed-integer verifier projections | `quant/gguf_k_gemv`, `quant/gguf_q8_0_t16_*`, `quant/gguf_q8_0_mmq_prefill`, `quant/gguf_q8_0_dp4a_gemv` |
| Qwen4Exp Q5_1 experts | Selected GEMV and exact grouped projections | `selected_grouped_wmma_prefill_compact_bf16_bf16_out` | `quant/qwen4_exp_q5_1` |
| Qwen4Exp branch mixing | `strict_unfused` | Fused gated branch mean | `fused/qwen4_exp_gr` |
| Qwen4Exp recurrent attention | `qwen4exp_sigmoid_strict_prefill` | `qwen4exp_sigmoid_peer_prefill` | `linear_attn/qwen4_exp_gdn` |
| Qwen4Exp sparse attention | `strict_rows_spans`, exact ordered decode variants | `production_wave32_h128_spans`, `production_rows_wave32_h128_spans` | `attention/qwen4_exp_qsa` |
| Laguna FP16 | Scalar/tiled projections and strict attention | Matrix-tiled projections, flash prefill, and fused attention/output gate | `linear/laguna_f16_projection`, `attention/laguna_kv_attention`, `attention/laguna_flash_attention_prefill` |
| Moonshine FP16 | Separate projection, activation, residual, and norm | Fused MLP/residual/norm; optional CUDA CUTLASS attention | `linear/moonshine_projection`, `fused/moonshine_*`, `norm/moonshine_layernorm`, CUDA `attention/moonshine_attention_cutlass` |
| VibeVoice BF16 | `strict` primitives and incremental prefill | Fused depthwise convolution, matrix-tiled frontend, and library prefill | `vibevoice/encoder`, `vibevoice/registered.py` |
| TimesFM | FP32 attention | FP16 matrix-tiled flash attention | `timesfm/timesfm` |
| YuE2 autoregressive BF16 | Row-by-row dense GEMV prefill | FP16-converted hipBLASLt batched prefill | Shared `linear/dense_gemv`, `hipengine/runtime/yue2_ar.py` |
| YuE2 non-autoregressive attention | `nar_attention_f32` | `nar_attention_wmma` with changed arithmetic | `yue2/nar`, `yue2/nar_wmma` |

These rows identify related implementations, not blanket profile assignments:
exactness, supported inputs, and selection scope belong to each variant.

## Fused and composite fallback map

Each fused composite has a registered strict unfused chain. Exact variants and
rounding boundaries are defined in the wrappers and execution profiles.

| Composite family | Paths | Unfused chain |
| --- | --- | --- |
| `add+rmsnorm`, `add_rmsnorm` | Qwen/GGUF, CUDA Maple | Residual add → RMSNorm |
| `head_rmsnorm+partial_rotary` | PARO/GGUF/Laguna | Head RMSNorm → rotary |
| `head_rmsnorm+partial_rotary+kv_write` | Laguna | Head RMSNorm → rotary → cache write |
| Projection + head norm + rotary + cache write | Laguna | Projection → head RMSNorm → rotary → cache write |
| `rotate+dual_pack8_gemv` | PARO | Input rotation → two projections |
| `rotate+selected_dual_pack8_gemv` | PARO | Selected projections and rotation in variant order |
| `silu_rotate+selected_pack8_gemv` | PARO | SiLU/product → rotation → down projection |
| `split_qgate+key_cast` | PARO | Query/gate split → key cast |
| `weighted_lanes_sum+shared_add` | PARO | Weighted reduction → shared add |
| `shared_gate_combine+residual` | PARO/GGUF | Shared-gate combine → residual add |
| `weighted_sum+shared_gate+residual` | PARO/GGUF | Weighted sum → shared-gate combine → residual add |
| Expert tail + RMSNorm | PARO/GGUF/Laguna | Expert combine → residual → RMSNorm |
| `moe_linear+weighted_sum` | GGUF | Selected down projection → weighted reduction |
| `linear+residual` | GGUF | Projection → rounded residual add |
| `linear+add+rmsnorm` | Laguna | Projection → residual add → RMSNorm |
| Linear-attention snapshot composites | GGUF/DFlash | Convolution or recurrence → named cast → state snapshot |
| `laguna_attention_decode+attention_gate` | Laguna | Attention → output gate |
| `moonshine_partial_rope+moonshine_self_cache` | HIP/CUDA Moonshine | Rotary → cache append |
| `moonshine_residual+moonshine_layernorm` | HIP/CUDA Moonshine | Rounded residual add → LayerNorm |
| Moonshine MLP projection composites | HIP/CUDA Moonshine | Bias projection → gated SiLU; projection → rounded residual |
| Selected-expert feed-forward composite | GGUF Q4_K | Gate/up projections → SiLU/product → down projection |
| Selected-expert rotation/feed-forward composite | PARO | Input rotation → gate/up → SiLU/down rotation → down projection |

## CUDA sm_120a

Device sources: `hipengine/kernels/cuda_sm120a/`. These are independent CUDA
implementations. `cuda_sm86` is a scaffold with no device kernels.

| Source | Purpose |
| --- | --- |
| `quant/maple_ternary.cu` | Maple ternary projections and affine4 embedding/head operations. |
| `attention/maple_attention.cu` | Maple normalization, rotary transforms, cache writes, and attention. |
| `moe/maple_moe.cu` | Maple routing, SwiGLU, and weighted residuals. |
| `moe/group_scatter.cu` | Group expert assignments and pack tile metadata. |
| `norm/maple_rmsnorm.cu` | Maple root-mean-square normalization and residual/head variants. |
| `linear/maple_lm_head.cu` | Maple vocabulary projection, argmax, and top-k. |
| `linear/moonshine_projection.cu` | Moonshine FP16 decoder projections. |
| `linear/lm_head.cu` | Vocabulary projection and final reductions. |
| `norm/moonshine_layernorm.cu` | Moonshine layer normalization and residual/normalization. |
| `fused/moonshine_mlp.cu` | Moonshine gated SiLU activation. |
| `fused/moonshine_glue.cu` | Moonshine embedding, residual, rotary, cache, and decode control. |
| `attention/moonshine_attention.cu` | Moonshine decoder self-attention and cross-attention. |
| `attention/moonshine_attention_cutlass.cu` | Moonshine self-attention through CUTLASS. |
| `encoder/moonshine_encoder.cu` | Moonshine encoder convolution, normalization, activation, and attention. |
| `encoder/moonshine_encoder_lt.cu` | Moonshine encoder projections/attention through cuBLASLt. |
| `encoder/moonshine_encoder_cudnn.cu` | Moonshine encoder convolution through cuDNN. |
| `smoke/smoke_add.cu` | Vector addition for build/runtime smoke tests. |
