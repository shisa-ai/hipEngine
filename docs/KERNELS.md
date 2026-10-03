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
│   ├── gemma4/                 # Gemma 4 layers, attention, rotary, router, experts
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

| Functional family | Source / wrapper | Principal registry layers and quants | Stable notes |
| --- | --- | --- | --- |
| Cast and gather | `convert/cast.{hip,py}`, `convert/gather.{hip,py}` | `cast_*` (`bf16`, `fp16`, `fp32`, scaled rows); `gather_f32_rows_by_i32id` | Explicit low-precision boundaries and row gathers; no framework tensors in device ABI. |
| RMSNorm | `norm/rmsnorm.{hip,py}`, `fused/gguf_ops.{hip,py}` | `rmsnorm`, `add_rmsnorm`, `add_rmsnorm_f32`, `head_rmsnorm` (`bf16`, `w4_paro`, `gguf_f32_weight`) | Qwen weights use delta semantics; PARO out variants use direct norm weights; GGUF F32-weight variants retain exact generic fallbacks. |
| Rotary/prelude | `rotary/paro_rotate.{hip,py}`, `rotary/qwen35_rotary.{hip,py}` | `paro_rotate1/2/3`, `paro_rmsnorm_rotate2`, `partial_rotary`, `head_rmsnorm+partial_rotary`, `split_qgate` | BF16/FP16 PARO rotation and Qwen partial-RoPE/head-normalization families. |
| Dense projection and head | `linear/dense_gemv.{hip,py}`, `linear/lm_head.{hip,py}` | `dense_gemv`, `dense_dual_gemv`, `linear_pair`, `linear+residual`; `lm_head`, `lm_head_argmax`, `argmax`, `topk` | Dense fallback/auxiliary projection plus deterministic final reductions. BF16 hidden/weight GEMV has both BF16 and unrounded F32 outputs, including the strict full-logit BF16-GGUF head route. |
| PARO AWQ projection | `quant/paro_awq_gemv.{hip,py}` | `pack8_gemv`, `dual_pack8_gemv`, `selected_*pack8_gemv`, `pack8_gemm`, rotate/SiLU composites (`w4_paro`) | Strided/transposed, BF16/FP16, selected-expert, fused-W4 prefill, and small-row routes. |
| PARO Marlin-K | `quant/paro_marlin_k.{hip,py}` | `marlin_k_gemv` (`w4_paro`) | c=1 replacement layout; pack8 alias remains available to prefill/fused projections. |
| PARO compact WMMA | `wmma/paro_awq_wmma.{hip,py}` | `awq_wmma` (`w4_paro`, `bf16`) | Compact/non-compact selected gate/up and down prefill; exact GEMV routes remain fallback. |
| W8A16 projection/shared expert | `quant/w8a16_linear.{hip,py}` | `w8a16_linear` (`w8a16`, `w4_paro`) | Single/multi-row lowp projection and shared-expert helper variants. |
| Router/select | `moe/router.{hip,py}` | `router_logits`, `router_select`, `router_topk_shared`, `router_topk_split_shared` | BF16/FP16/F32 hidden/weight combinations; deterministic top-k and shared-gate routes. The Qwen4Exp multirow F32 owner preserves dense FMA/reduction order while reusing each weight across four rows: rows508 primitive wall is 3.767→1.911 ms and clean p508 is 89.689→91.121 tok/s with exact logits/state/tasks; c1 and the registered `f32_hidden` route remain fallback. Library handle is hoisted into a module cache (`_router_library()`) so per-launch host cost stays a plain ctypes call (~15 us) instead of re-running `build_qwen35_router(load=True)` (~34 us/call with a pinned session compiler version). |
| MoE grouping and packing | `moe/group_scatter.{hip,py}` | `moe_group_count/prefix/scatter`, `moe_group_compact`, `moe_gather_packed_hidden`, `moe_wmma_tile_map`, `moe_mmq_tile_map` | Stable count/prefix/scatter and compact tile metadata; generic and `w4_paro`. |
| MoE prefill orchestration leaf | `moe/prefill.py` | `moe_prefill` (`w4_paro`) | Registered wrapper composition for selected-expert prefill. |
| Whole selected-expert FFN | `quant/paro_moe_ffn_fused.{hip,py}` | `moe_ffn_selected` (`w4_paro`) | Rotate → gate/up → SiLU → down-rotate → down projection megakernel; primitive chain remains fallback. |
| c1 native dispatcher | `dispatch/moe_c1_dispatch.{hip,py}` | C function-table dispatcher (not a registry layer) | Contracts Python launch overhead while invoking registered/raw function pointers; does not replace component kernels. |
| SiLU/rotation primitives | `fused/paro_silu.{hip,py}` | `silu_mul_dual`, `silu_mul_separate`, `silu_mul_dual_rotate`, `silu_mul_pair_rotate` | Primitive and fused activation/down-rotation boundaries coexist; separate BF16 SiLU permits exact in-place replacement of its gate plane. |
| MoE combine/tail | `fused/paro_combine.{hip,py}` | `weighted_lanes_sum`, `weighted_sum`, `shared_gate_combine`, residual/RMSNorm composites | BF16/FP16/F32 values with FP32 route weights/gates; explicit primitive fallbacks are registered. Qwen4Exp prefill uses exact token-local BF16 batch siblings for compact top-10 weighted sum and shared-gate combine (gfx1151 reduced three-row traces: 1,963/1,403 ns); c1 primitives remain unchanged. |
| Paged KV write/copy | `attention/paged_kv_write.{hip,py}` | `paged_kv_write`, `paged_kv_copy` (`bf16`, PARO/GGUF, INT8 layouts) | All attention-visible writes consume complete `KVLiveSpans`; includes BF16 and supported INT8 storage formats. The FP32→BF16 family includes shared-cache prompt rows with one explicit logical position/table per row for Qwen4Exp prefill; a reversed-page gfx1151 fixture traces at 8,376 ns. |
| Gemma BF16 attention | `gemma4/gemma4_attention.{hip,py}` | masked ungated prefill/decode, strict class and owned global-score class | Head dimensions 256/512 move logits to stream-owned global scratch when the resident row exceeds 64 KiB. Query batches of at most four bound per-layer global scratch; the existing 256-lane denominator and ordered V chains are unchanged. The strict class/block routes remain the short-context arithmetic oracle. Other head geometries keep their resident LDS capability bound. |
| Full/paged attention | `attention/paged_attn_decode.{hip,py}` | `full_attn_decode/prefill`, `paged_attn_decode/prefill`, `full_attn_gate_mul` | Contiguous and paged, batched, GQA, split-K, gated reduce, and supported INT8 KV variants. Per-token/head INT8 includes a row-batched 24Q/4KV/D256 split-K producer plus explicitly strided BF16 gated reducer; the c1 leaf remains registered as its numerical fallback, and the gfx1100 Qwen3.8-27B artifact qualifies the batch variant to physical c4 with the c1 leaf as the registered fallback above that width. gfx1151 Qwen3.5-0.8B rows1/8Q/2KV/D256 selects generic split-K3+fused BF16 gate at cap514-641. The private-c1 exact leaf is the fixed256 body at 256 threads (strict exact default) with a parameterized `fixed256_threads_spans` probe at runtime block width; gfx1151 promotes 1024 threads (T2 non-exact, execution-profile gate-passed) via `GGUF_SHORT_C1_BATCH_ATTN_THREADS`. Dense H5120/L64/24Q/4KV/D256 selects the BF16 grouped-GQA split producer from context 4096; shorter contexts and unsupported shapes/backends retain the generic producer. | Native BF16-gated prefill uses owned global score scratch when its context-sized shared allocation would exceed 64 KiB; bounded query batches reuse the split partial-output arena without changing the parent reduction order. The explicit `causal_gqa_gate_bf16_global_scores` variant also permits parent-parity checks at short contexts. INT8 per-token/head includes a row-batched 24Q/4KV/D256 split-K producer with an explicitly strided BF16 gated reducer; the c1 leaf remains its registered numerical fallback. |
| AOTriton adapter | `attention/aotriton_wrap.py`, `attention/aotriton.py` | `full_attn_prefill` (`w4_paro`, `gguf_qwen35`) | Optional library adapter; native raw-pointer paths remain available. |
| Linear-attention Conv | `linear_attn/conv.{hip,py}` | `linear_attn_*conv_decode/prefill`, chain/tree and snapshot composites | Decode, segmented prefill, verifier tree/chain, and state-snapshot variants. |
| Linear-attention GDN | `linear_attn/gdn.{hip,py}` | `linear_attn_prefill_prepare`, `gdn_*recurrent*`, RMSNorm/gate/rotate/cast/snapshot composites | Exact schedules retain FP32 recurrent state; segmented, chain/tree, snapshot, and decode-order writers cover prefill, verifier, and multi-request selected commit, with optional FP32 state-row journals, direct BF16 handoffs, and an exact FP32 output tap. FP16-state (FP32 accumulation) and gfx1151 cluster/chunked compact-peer variants are explicit opt-ins or capability selections that always retain an FP32 fallback. |
| Runtime state | `runtime/state.{hip,py}` | token embedding, positions/metadata, graph record/commit, scalar state, profiling wall-clock marker | Device-side graph/verify bookkeeping, indexed row state, token publication, and profiling-only steady-clock boundaries. |
| Sampling | `sampling/sampler.{hip,py}` | `sampler`, `mtp_draft_topk` | Greedy/temperature/top-k helpers and bounded draft top-k. Full-vocabulary `sorted_rows_i32` uses 256-key tile sorting, parallel merging and FP64 scans of FP32 weights; the original temperature/top-p variants remain registered strict fallbacks. Caller-owned scratch is `rows * (24*vocab + 8*ceil(vocab/256))` bytes. gfx1151 correctness and trace evidence are in `tests/test_gpu_sampler_full_vocab.py` and the fast-sampling worklog; gfx1100 hardware transfer is unverified. |
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
| `attention/dms_compact.hip` | Select, pack, append, and attend over compact key/value caches. BF16 split attention uses the generic grouped producer on gfx1151 and the wave-group6 producer on gfx1100 at supported geometry. |
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
| `quant/gguf_q8_0_t16_prefill.hip` | T16 Q8_0 prefill projections with FP32 repair of non-finite WMMA accumulators. |
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
| Gemma 4 GGUF `gguf_q4_k_m` prefill attention | `gemma4_plain`: scalar strict attention with an exact decode twin | `gemma4_staged`: strict-order FP32 score, softmax and P*V stages with global score workspace and bounded LDS; selected by production for head_dim 256/512. Explicit WMMA candidates remain registered for arithmetic evaluation but are not defaults after multicategory KL failures | `gemma4/gemma4_attention`, `gemma4/gemma4_attention_staged`, `gemma4/gemma4_attention_prefill_wmma`, `gemma4/gemma4_attention_prefill_wmma_full` |
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
| `add+rmsnorm`, `add_rmsnorm` | HIP Qwen/GGUF; CUDA Maple helper | add/residual boundary → RMSNorm |
| `head_rmsnorm+partial_rotary` | HIP PARO/GGUF/Laguna | head RMSNorm → partial rotary |
| `head_rmsnorm+partial_rotary+kv_write` | HIP Laguna | head RMSNorm → partial rotary → KV write |
| `attention_projection+head_rmsnorm+partial_rotary+kv_write` | HIP Laguna | projection (pair/triple/quad as applicable) → head RMSNorm → partial rotary → KV write |
| `rotate+dual_pack8_gemv` | HIP PARO | rotate input(s) → two pack8 GEMVs |
| `rotate+selected_dual_pack8_gemv` | HIP PARO | selected dual pack8 GEMV → output rotate, or explicit rotate and projection primitives matching the variant |
| `silu_rotate+selected_pack8_gemv` | HIP PARO | SiLU/product → rotate → selected down pack8 GEMV |
| `split_qgate+key_cast` | HIP PARO | split query/gate → key cast |
| `weighted_lanes_sum+shared_add` | HIP PARO | weighted lane reduction → shared add |
| `shared_gate_combine+residual` | HIP PARO/GGUF | shared-gate combine → residual add |
| `weighted_sum+shared_gate+residual` | HIP PARO/GGUF | selected weighted sum → shared-gate combine → residual add |
| MoE tail + RMSNorm composites | HIP PARO/GGUF/Laguna | weighted/shared combine → residual/tail → next RMSNorm |
| `moe_linear+weighted_sum` | HIP GGUF selected down | selected down projection → slot-order weighted reduction |
| `linear+residual` | HIP GGUF | linear projection → rounded residual add |
| `linear+add+rmsnorm` | HIP Laguna | source-F16 projection → add/residual → RMSNorm |
| Linear-attention snapshot composites | HIP GGUF/DFlash | Conv or GDN primitive → cast if named → state snapshot |
| `laguna_attention_decode+attention_gate` | HIP Laguna | attention decode → softplus/sigmoid gate publication |
| `moonshine_partial_rope+moonshine_self_cache` | HIP and CUDA Moonshine | partial RoPE → fixed self-cache append |
| `moonshine_residual+moonshine_layernorm` | HIP and CUDA Moonshine | rounded residual add → LayerNorm |
| Moonshine MLP projection composites | HIP/CUDA Moonshine | bias projection → gated SiLU; projection → rounded residual |
| `moe_ffn_selected/fused_dual_silu_down_*` | HIP GGUF Q4_K | selected dual gate/up projection → SiLU/product → selected down projection |
| `router_topk` (Gemma 4 rows=1) | HIP gfx1100 Gemma 4 decode | router_prescale → router_logits → router_select → expert_weight_scale |
| `moe_ffn_selected/fused_rotate_dual_silu_rotate_down_*` | HIP PARO | rotate1 → selected dual pack8 gate/up → SiLU/down-rotate → selected down pack8 projection |

Fallback requirements:

- Strict fused and unfused paths share exact/parent-parity fixtures at every
  published low-precision boundary. Production fused paths share the strict
  fixture plus the full strict-teacher profile gate; free-running ID equality is
  diagnostic unless strict/batch-invariant says otherwise.
- Removing a strict fallback is an architectural change and requires updating this table plus `PLAN.md` if the invariant changes.
- A library call can be one stage of an unfused chain, but it does not waive the independent primitive/oracle route.

## Source-lineage audit

R7 compiler-version override-identity cache (2026-09-10): the env compiler-
version cache in `hipengine/core/build.py` is keyed by override identity - compiler
plus the raw values of all four override vars - not compiler alone (f70c7b75d,
prefix-memoized public-API resolution refined in bb0c78afa after a bytes-key
`_data` fast path silently missed str keys). Fixes order-dependent stale
resolution where a compiler-only key reused the first version after environment
changes and selected the wrong build artifact (tests/test_unit_build.py 11/12 ->
12/12, order-dependent state leakage). Corrected cache resolves in 1.32us
(~7x cheaper than uncached; 0.03us with the original fix), order-independent.
No measurable TG effect: the earlier -1.9 ms/token claim was measured with the
buggy compiler-only cache and is WITHDRAWN (TRUE all-off-arm attribution shows
~0 ms; retained as correctness/robustness). Evidence:
`2026-09-10-r7-wrapper-host-screen.json` (supersessions_and_corrections),
`2026-09-10-r7-combined-tg-attribution.json`.

R7 PLE page-cache warm sweep, opt-in `HIPENGINE_QWEN4_EXP_PLE_WARM` (2026-09-10):
one-time 15.1s mmap-touch sweep (chunked uint64 sum through the byte view) of
the 28.8 GB PLE weight mapping at runner init, eliminating cold-cache PLE
staging gather faults (1.4-27.7 ms/step, case-dependent, ~10 major faults/step
on btrfs-compressed NVMe; stage 28.2 -> 0.11 ms/step on code-p512). fadvise/
madvise WILLNEED measured ineffective, POPULATE_READ EINVAL on this kernel.
Gates: bit-exact (A/B-harness digest ea0412231532bc7b, cold-cache A/B with
POSIX_FADV_DONTNEED reset); TG median 57.14 -> 52.33 ms/token (-4.8 ms,
~8.4%) standalone; -7.35 ms (-12.2% latency = +13.9% throughput) in the TRUE
all-off-arm attribution. Resource profile (demotion trigger): +15.4 s and
+6,015 major faults PER RUNNER CONSTRUCTION (28.8 GB read from storage);
amortization 577-15,400 tokens (median ~2,050 - canonical benches at ~576
tokens per construction never amortize); 28.8 GB page-cache residency
unqualified on constrained shared hosts (comparator-confound retraction
recorded: shared page-cache pressure, not immunity). DEFAULT OFF since
0211fe125; enable with `HIPENGINE_QWEN4_EXP_PLE_WARM=1` for long-lived
serving. Evidence: `2026-09-10-r7-wrapper-host-screen.json`
(ple_page_cache_promotion, ple_resource_qualification),
`2026-09-10-r7-baseline-retention-v8.md`, `2026-09-10-r7-combined-tg-attribution.json`.

R7 wrapper-host promote (2026-09-10): `HIPENGINE_QWEN4_EXP_BATCHED_POSITION`
default ON replaces 24 per-layer 8B blocking `set_position` H2D copies per
decode step (12 QSA layers x position+context, unique states) with one shared
interleaved [position,context] int64 region and a single 192B H2D via
`position_prepared`. Bit-exact token streams (digest ea0412231532bc7b, all
fixture cases, 3 reps/arm - the A/B-harness digest per the 2026-09-10
digest-mismatch investigation; canonical-contract digest 19045b7c9fd442e5,
arm-equality unaffected), TG median -0.77ms/token (~1.3%) at the original screen; -0.54 ms (-0.9%
latency = +0.9% throughput) in the TRUE all-off-arm attribution
(`2026-09-10-r7-combined-tg-attribution.json`); copy surface
27->4 per step. Opt-out via `=0`. Evidence: `2026-09-10-r7-wrapper-host-screen.json`,
retention v7 `2026-09-10-r7-baseline-retention-v7.json` (TG +1.6/+1.4/+3.6%
at p512/p1024/p4096; packet PP deltas environmental, direct interleaved PP A/B
within +/-0.85%). Promote commit b7e9f19b9.

H256 QSA exposes kernel-only `strict_h256_head_quad_rows_spans`,
four-head specialization of exact K/V-sharing body,page256 and GQA
divisible by4. Synthetic selected2051 attention512/1024 ratios
1.033x/1.063x versus two-head candidate,20 pairs exact.23 tests
including extreme query scales and poisoned unused KV pass.
VGPR72->112,no LDS/scratch. No runtime/default or CPU-recovery claim.
Evidence: `2026-09-07-framework-qwen4exp-qsa-head-quad.json`.
Default-off `HEAD_PAIR=quad` model admission passes six full-logit/state/
KV cases including all four p4096 categories,24 sparse calls each,
zero dense/decode calls. Page256-parent Hq24/Hkv2 only;profile binders0.
Admission: `2026-09-07-framework-qwen4exp-qsa-head-quad-state.json`.
Production now bindsquad/strict0 after staged72 exact trajectories;
all4 p4096 request means improve including transition,PP+3.571%,
TG-3.531% explicitly retained. Dense/decode kernels unchanged;not a
statistical all-case non-regression or CPU-recovery fix.
Production: `2026-09-07-framework-qwen4exp-qsa-head-quad-production.json`.

H256 QSA exposes the promoted ordered-v2 c1 decode rewrite
`strict_ordered_three_pass_v2_spans` in the source lineage: warp-tree scores on an
eight-token grid, an exact `fmaxf` block scan with pointwise `expf` and a prefetched
serial denominator, and staged-tile values with clamped unconditional loads after a
per-load select was shown to defeat memory-level parallelism (406 versus122us).
Named kernel medians at clean source, cache-only build: scores33.5us, coefficients
14.3us, values100.7us (VGPR32/40/96,scratch0) versus parent399.7/176.1/533.0us;
leaf route1.154->0.179ms/layer,6.45x,bit-exact. Six-case off/on/off full
logits/4-step/state/full-KV gate passes at committed source with exact engagement
accounting; canonical A/B 72 trajectories exact, p4096 weighted TG+14.705%
(arithmetic-mean-rate ratio+16.516%). Evidence:
`2026-09-08-framework-qwen4exp-qsa-ordered-v2-kernels.json`.

H256 QSA exposes kernel-only `strict_h256_head_pair_rows_spans`,
page256 with even GQA ratio;two adjacent query heads share each K/V
load but retain independent score/online-softmax state. Explicit
`fma(acc,old_scale,score_scale*value)` preserves parent's contraction;
initial alternative failed strict equality.23 tests pass including
poisoned unused KV and parent CPU-reference chain. Synthetic2051-selected
attention512/1024 rows2.544x/2.610x,30 pairs exact.
VGPR40->72,no scratch/LDS,half head blocks. No runtime default.
Evidence: `2026-09-07-framework-qwen4exp-qsa-head-pair.json`.
Default-off model admission via `HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR=1`
passes eight full-logit/state/KV cases at chunk1024,all four p4096
categories included.24 calls per sparse p4096 arm,zero decode,dense
short paths unchanged. Hq24/Hkv2 page256 parent only;both binders0.
Admission: `2026-09-07-framework-qwen4exp-qsa-head-pair-state.json`.
Staged full-model qualification preserves72 trajectories and improves
p4096 PP3.560%,but TG drops6.526%;two long-request averages regress.
Candidate retained default-off pending causal decode/phase investigation.
Model evidence: `2026-09-07-framework-qwen4exp-qsa-head-pair-model.json`.

Raw-vector64x64 MMQ tested against current128x64 on Framework:
actual Q layers3/7 at512/1024 rows0.959-0.985x,80 pairs exact,
27 tests pass. VGPR160/scratch0 both,threads256->128,
dynamic LDS48384->28928B. Candidate removed;production128x64 stays.
Evidence: `2026-09-07-framework-qwen4exp-mmq-square64-rejected.json`.

Prepacked token64 MMQ specialization tested and removed:actual qkv/SSM
at512/1024 rows0.931-0.969x,80 pairs exact,27 tests pass.
VGPR144->112,scratch0,dynamic LDS57856->48384B. Prepacked128 and
promoted raw-Q64 remain independently selected;no blanket tile policy.
Evidence: `2026-09-07-framework-qwen4exp-prepacked-token64-rejected.json`.

Raw Q8 MMQ exposes kernel-only
`mmq128_token64_q8_1_d4x3_guarded_f32_f32_out`:128-output/64-token
tile with existing vector activation staging. Exact per-output arithmetic,
risk set and repaired outputs;23 tests pass. Large Q projection screens
win1.109x/1.138x at1024 rows on layers3/7;all200 screen pairs exact.
VGPR184->160,dynamic LDS57856->48384B,scratch0. Short-Q/long-GR
mixed order results excluded from proposed runtime scope;prepacked path
unchanged. No production default change.
Evidence: `2026-09-07-framework-qwen4exp-mmq-token64.json`.
Default-off `HIPENGINE_QWEN4_EXP_MMQ_TOKEN64=1` model admission passes
five chunk1024 full-logit/state/KV cases,12/48 calls,zero decode/final
owners. Parent raw-vector K2560/N12288 rows>=512 only;prepacked unchanged.
Production binder1/strict0 after full72 exact trajectories and all12
prefill means improve. One request-case loss0.183% explicitly retained
under prefill-first policy;prepacked/GR unchanged.
Admission: `2026-09-07-framework-qwen4exp-mmq-token64-state.json`.
Production: `2026-09-07-framework-qwen4exp-mmq-token64-production.json`.

Q8 MMQ bank-first compute experiment removed:actual qkv/Q512 medians
1.000x/0.998x,1024 medians1.013x/1.020x but opposite-order means
regress for all four shapes.80 pairs exact,22 tests pass,
VGPR184->192,scratch0. No model admission or production change.
Evidence: `2026-09-07-framework-qwen4exp-mmq-bank-first-rejected.json`.

Q8 raw-vector MMQ output-scale hoist tested and removed:actual qkv/Q
projections at512/1024 rows yield0.319-0.338x operation-complete speedup,
80 pairs exact,22 tests pass. VGPR184->256,scratch0->1132B.
Original raw-vector kernel/wrapper/registry/harness restored.
Evidence: `2026-09-07-framework-qwen4exp-mmq-scale-cache-rejected.json`.

Q5_1 per-row publication row16 specialization was tested and removed:
actual two-bank512/1024 synthetic-routing screen0.429x/0.412x,
40 pairs exact,9 tests pass. Dynamic LDS8672->16864B,VGPR96->88,
scratch0 both. Production row8 unchanged; no runtime candidate remains.
Evidence: `2026-09-07-framework-qwen4exp-q51-row16-rejected.json`.

GDN exposes kernel-only `qwen4exp_sigmoid_wave_norm_prefill`: exact128
normalization reduction, stride64/32 grouped as parent then wave shuffle
16/8/4/2/1; serial recurrence unchanged. Dk=Dv128 only. Synthetic
Hk16/Hv48 at512/1024 tokens2.812->2.699ms /5.574->5.358ms.
Output/state exact including split execution; VGPR256 unchanged,
private scratch24->36B,dynamic LDS2560B unchanged. No runtime default.
Evidence: `2026-09-07-framework-qwen4exp-gdn-wave-norm.json`.
Default-off model admission via `HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM=1`
passes five full-logit/state/KV cases at chunk1024. Existing serial
prefix21 layers only,21/84 prefill calls,zero decode; tiled suffix unchanged.
Both binders0 pending throughput gate.
Admission: `2026-09-07-framework-qwen4exp-gdn-wave-norm-state.json`.
Full canonical72 trajectories exact; model means nearzero and mixed,
not a statistical non-regression result. Keep default-off pending
actual-model owner timing; kernel saving is retained,not discarded.
Model evidence: `2026-09-07-framework-qwen4exp-gdn-wave-norm-model.json`.
Superseding owner qualification:378 actual-model calls exact,all paired
means faster across six cases; p4096 serial GDN~487->467ms. Production
now binds1/strict0 under sub-window-retention policy. Prior mixed model
means stay explicit; no headline speedup claimed.
Owner evidence: `2026-09-07-framework-qwen4exp-gdn-wave-norm-owner.json`.

Q4 pair2 two-block weight-pipeline experiment removed: actual layer3
gate/up+SiLU screen0.910x/0.909x at512/1024 tokens,40 exact pairs,
14 tests pass. VGPR88->112,LDS4608B/scratch0 unchanged. Original
pair2 kernel/wrapper/registry/screen restored; no runtime admission.
Evidence: `2026-09-07-framework-qwen4exp-q4-block-pair-rejected.json`.

Q5_1 promotes
`selected_grouped_prefill_pair2_row_publish_bf16_bf16_out`: K640
register-cache pair2, per-row folded partial publication with identical
original LDS tree. Actual two-bank512/1024 synthetic-routing screen
1.413x/1.523x; captured mixed512 routing1.498x. All60 pairs exact,
22 tests pass. VGPR96/dynamic LDS8672B unchanged,scratch36->0B.
`HIPENGINE_QWEN4_EXP_Q51_ROW_PUBLISH=1` selects only existing
register-cache parent at rows>=512/K640. Five chunk1024 full-logit/
state/KV cases exact,25/100 prefill calls,zero decode calls/final owners.
Production binder1/strict0 after72 exact canonical trajectories and all12
prefill/request averages improve;PP gains4.296%/4.732%/4.323%.
Evidence: `2026-09-07-framework-qwen4exp-q51-row-publish.json`.
Admission: `2026-09-07-framework-qwen4exp-q51-row-publish-state.json`.
Production: `2026-09-07-framework-qwen4exp-q51-row-publish-production.json`.

Q5_1 register-cache paired-down first-wave reduction experiment removed:
exact stride64/32 LDS reads followed by shuffle tail is0.809x/0.798x
at512/1024 synthetic-routing tokens with actual layer0/1 weights.
All40 pairs exact,21 tests pass; VGPR96 unchanged,scratch36->24B,
dynamic LDS8672B unchanged. Production register-cache tree remains.
Evidence: `2026-09-07-framework-qwen4exp-q51-wave-tail-rejected.json`.

Q8 coltile paired-load candidate is removed after full12-case model A/B:
one prefill/nine request cases regress, aggregate p4096 decode -14.822%,
despite72 exact trajectories and kernel-only gate wins. Wave-scale
production stays. Kernel/state evidence at e23029b4c/067a9bcc0 is historical;
no candidate registry key or runtime route remains.
Evidence: `2026-09-07-framework-qwen4exp-q8-prefetch2-rejected.json`.

Q8 down promotes `selected_grouped_row4_register_gemv_bf16_bf16_out`,
restricted to K640/128 threads. Five decoded weights/thread are reused
across the expert row loop;row4 bundled reduction and mapped output order
unchanged. Compact/mapped screens37.373->32.127ms /37.228->30.808ms,
40 pairs exact,both orders positive.18 tests pass,VGPR24->32/LDS512B/
scratch0. `HIPENGINE_QWEN4_EXP_Q8_DOWN_REGISTER=1` replaces only the
compact/mapped bundle at rows>=512/K640; existing bundle remains fallback.
Five chunk1024 model state/full-KV cases pass exactly, zero decode calls;
production binder1/strict0 after canonical72 exact trajectories and all12
prefill/request averages improve. PP gains0.851%/0.701%/0.647%.
Evidence: `2026-09-07-framework-qwen4exp-q8-down-register.json`.
Admission: `2026-09-07-framework-qwen4exp-q8-down-register-state.json`.
Production: `2026-09-07-framework-qwen4exp-q8-down-register-production.json`.

Router shuffle-tail candidate was removed after full-model A/B failed
retention:one prefill/four request cases lose despite72 exact trajectories.
Earlier kernel/state evidence below is historical,not an available variant.
Original shared-tree router restored;21 focused regression tests pass.
Evidence: `2026-09-07-framework-qwen4exp-router-shuffle-rejected.json`.

F32 router exposes kernel-only `f32_hidden_token_tile4_shuffle_exact`.
Per-thread dense accumulation and shared128/64 reduction stay unchanged;
wave0 handles32..1 with the same tree,saving six block barriers.
Actual layer0/27 router1024 rows improve3.043->2.635ms /
3.051->2.648ms,exact.16 tests pass,80 timed pairs exact,both orders
positive;32 VGPR/scratch0/dynamic LDS4096B unchanged. Existing tile4
shared-tree owner remains strict fallback;model gates pending.
Evidence: `2026-09-07-framework-qwen4exp-router-shuffle.json`.

Default-off router model admission passes five chunk1024 full-logit/
routing/state/KV cases.48/192 enabled prefill calls,zero decode/final
owners;18 CPU tests. Both binders0;only existing exact multirow router
eligible. Canonical A/B remains. Evidence:
`2026-09-07-framework-qwen4exp-router-shuffle-state.json`.

Existing Q8 `selected_grouped_row4_bundle_gemv_bf16_bf16_out` supports
non-null sorted-lane-to-original-row maps,not just compact buffers. Actual
layer2 weights with borrowed layer0 counts screen2.126x/2.153x over
selected GEMV,exact. Potential integration reuses the map from Q5_K row4
gate/up and preserves token-major outputs;requires explicit map ownership
and model gates. No new kernel/default. Evidence:
`2026-09-06-framework-qwen4exp-q8-mapped-down.json`.

Earlier mapped-down model admission passed five full-logit/state/KV cases,
four decode steps,0/1/0 calls at512 or0/8/0 at4096,zero decode/final
owners. Current-call map-ready guard prevents stale scratch use.22 CPU
tests pass;both binders0 at admission. Counter hooks distinguish
mapped versus compact calls to the shared bundled kernel.
Evidence: `2026-09-06-framework-qwen4exp-q8-mapped-down-state.json`.
Clean8740dc13f canonical12-case A/B now passes72 exact trajectories and
all prefill/request cases. Production selects mapped-down only with a
current-call map,rows>=512;strict selects original GEMV. PP gains
1.284%/1.427%/1.594%,no new memory. Manifest and counter scope distinguish
token-major mapped calls from existing compact calls to the same kernel.
Evidence: `2026-09-06-framework-qwen4exp-q8-mapped-down-production.json`.

Q8 MMQ exposes `mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out`.
It retains raw-weight staging and uses the previously proven aligned
activation copies. Actual GR/query/output512 complete chains improve
1.891->1.371 /6.452->5.293 /2.891->2.229ms,all120 pairs exact and
both orders positive.25 tests pass;trace184 VGPR/scratch0 unchanged.
No packed-weight sidecar. Existing raw and strict fallbacks remain;
subsequent model state/KV and canonical A/B qualify promotion below.
Evidence: `2026-09-06-framework-qwen4exp-mmq-raw-vector.json`.

Earlier default-off raw-vector admission passed five full-logit/state/KV cases,
four decode steps:0/242/0 at512 and0/1936/0 at4096,zero decode/final
owners.32 CPU tests pass. Both binders0;existing raw MMQ rows>=64 only,
prepacked path independent.
Evidence: `2026-09-06-framework-qwen4exp-mmq-raw-vector-state.json`.
Cleancea077722 full12-case A/B now passes72 exact trajectories and all
prefill/request cases. Production binds raw-vector1,strict0;PP gains
3.367%/2.849%/2.969%,total request1.01730x,no memory increase.
Small decode losses retained explicitly. Manifest raw/prepacked roles
now name their respective vector kernels and strict fallbacks.
Evidence: `2026-09-06-framework-qwen4exp-mmq-raw-vector-production.json`.

Four-wave K64 Q4 residual staging is rejected/removed:61.357ms versus
exact17.403ms,224 VGPR/18432B LDS/no scratch. No-unroll59.887ms and
padded-stride57.152ms still lose. Numerical equality with residual reference
does not imply model qualification. Evidence:
`2026-09-06-framework-qwen4exp-q4-cooperative-rejected.json`.

`selected_dual_wmma_f16x2_bf16_bf16_out` was a **T2 diagnostic reference**,
now removed after the cooperative follow-up also failed. It reconstructed raw Q4_K
weights with FP16 high/residual planes, accumulates each separately and
adds before the BF16 gate/up boundary.16-row tiles with16/32/64 output
columns preserve reference outputs across tile widths. On one captured
actual-weight screen, gate/up BF16 agreement improves89.66%->99.65% and
post-SiLU82.46%->99.37%,but all tested widths lose to exact pair2.
The template branches,export/wrapper/key,harness and candidate tests are
removed;original Q4 files match24934b692. No production route ever used
the reference. Reproduction source20e39e32e retains the implementation.
25 existing Q4 regression tests pass after removal. Evidence:
`2026-09-06-framework-qwen4exp-q4-residual-wmma-reference.json`.

Q4 pair2 paired-BF16 input layout was rejected and removed. Bitwise
2x128->128x2 packing allowed32-bit pair loads but consumer17.333->
22.657ms regressed,with only0.184ms packing cost. VGPR88->72 did not
translate to throughput. Original pair2 layout/producer remain unchanged.
Recipe: `2026-09-06-framework-qwen4exp-q4-input-pair-rejected.json`.

Q4 pair2 wave-metadata scalarization was rejected and removed: readfirstlane
on wave-uniform scale/min operands reduced VGPR88->80 but actual gate/up+
SiLU remained flat/order-sensitive (17.478->17.490ms). No tile/arithmetic
change;original pair2 production retained. Recipe:
`2026-09-06-framework-qwen4exp-q4-wave-meta-rejected.json`.

Q5_1 exposes
`selected_grouped_prefill_pair2_register_cache_bf16_bf16_out` for K640.
It predecodes ten weights per thread across an output pair, reusing them
without growing LDS. Captured code/mixed two-bank512 projections improve
34.450->28.195ms /33.756->27.819ms (1.222x/1.213x), all40 pairs exact
and both orders positive.17 GPU tests pass. Trace VGPR72->96 and
private scratch0->36B, dynamic LDS8672B unchanged: not spill-free.
Full-residency model gates subsequently passed; original folded-pair and M1 fallbacks
stay registered. Evidence: `2026-09-06-framework-qwen4exp-q51-register-cache.json`.

Earlier default-off model admission passed five full-logit/state/KV cases with
four decode steps. Invocation0/25/0 at512 or0/200/0 at4096,zero decode/
final allocations.18 CPU tests passed. Both binders0 at admission,only existing folded-pair
rows>=512,K640.
Evidence: `2026-09-06-framework-qwen4exp-q51-register-cache-state.json`.
Clean67fdfccb4 full12-case A/B now passes72 exact trajectories and all
prefill/request cases. Production binds register-cache1,strict0;manifest
explicitly names rows>=512/K640 and original strict fallback. PP gains
2.892%/2.936%/2.716%,total request1.01790x;no extra tracked allocation,
private scratch36B remains. Evidence:
`2026-09-06-framework-qwen4exp-q51-register-cache-production.json`.

The Q5_1 folded-pair decoded-LDS weight-cache experiment is rejected and
removed: captured actual two-bank projection34.449->58.987ms (0.584x),
exact. Dynamic LDS8672->13792B,VGPR72/scratch0 unchanged. Original
folded-pair production and strict fallback remain. Do not repeat unchanged
LDS materialization;see `2026-09-06-framework-qwen4exp-q51-weight-cache-rejected.json`.

Q8 MMQ also exposes
`mmq128_prepacked_vec4_q8_1_d4x3_guarded_f32_f32_out`. It copies aligned
four-word activation groups instead of scalar words, preserving all three
planes, tail clamping, WMMA accumulation and risk repair. Actual QKV/SSM
512-row complete chains improve1.292x/1.325x, both orders positive.
All80 pairs exact;25 tests pass including CPU-reference floors and exact
risk sets. Cached trace VGPR144/LDS57856B/scratch0 unchanged. Subsequent
model admission and promotion are recorded below.
Evidence: `2026-09-06-framework-qwen4exp-mmq-activation-vec4.json`.

Earlier default-off model admission passed five full-logit/state/KV cases,
four decode steps each, calls0/72/0 at512 and0/576/0 at4096, zero decode
calls/final owners. Both profile binders pinned0; only existing prepacked
rows>=64 were eligible.23 CPU tests passed.
Evidence: `2026-09-06-framework-qwen4exp-mmq-vec4-state.json`.
Subsequent clean `9b0d14cec` canonical A/B passes all72 exact trajectories
and all12 prefill/request-wall cases. Production now binds vec4=1,strict=0;
PP512/1024/4096 improves2.021%/2.173%/2.056%. Scope remains existing
prepacked rows>=64. No new memory;small decode losses are explicitly
retained. Evidence: `2026-09-06-framework-qwen4exp-mmq-vec4-production.json`.

Q8 MMQ registers a separate T0
`mmq128_prepacked_q8_1_d4x3_guarded_f32_f32_out` candidate. It consumes
K-major `[K/256, ceil(N/128)*128, 76]` int32 words:64 quant words,8 exact
FP32 scales and4 padding words. Padded output rows repeat the last real row.
The aligned weight tile copies directly to the unchanged57856-byte dynamic
LDS arena; activation planes, integer dots, F32 accumulation and risk detection
are unchanged. Exact repair still consumes **raw Q8**, not packed weights.
Framework real QKV rows512 confirms1.094/1.095x on layers0/4, SSM output1.022x;
58 tests pass, trace VGPR184->144 and scratch0. GR remains excluded:
its near-flat/order-sensitive evidence is retained. Production now selects
prepacked MMQ only for measured GDN QKV/SSM shapes after full72-trajectory
exact A/B with every prefill case faster (+0.58-0.66% weighted).
Mixed512 request wall loses0.14%; retention follows prefill-first direction.
Raw MMQ remains production-parent rollback, strict coltile the declared
numerical fallback. Evidence: `2026-09-06-framework-qwen4exp-mmq-prepack-production.json`.
Evidence: `2026-09-06-framework-qwen4exp-mmq-prepack.json`.
The registered `weight_pack/gguf_q8_0/mmq_kmajor76` packer now builds this
layout directly from resident raw device weights. Nine focused tests cover
byte-exact CPU layout, signed-zero/subnormal scales, output tails and repeats;
cached trace shows24 VGPR, no scratch/LDS.
Evidence: `2026-09-06-framework-qwen4exp-mmq-gpu-pack.json`.
Qwen4Exp runtime admission passes five full logits/state/KV
cases with live calls0/72/0 at512 and0/576/0 at4096. Runner-owned72 sidecars
add1,793,064,960 bytes and close cleanly; session maps only substitute the
matmul weight pointer, never decode/repair. The full12-case performance gate
is retained above. Evidence: `2026-09-06-framework-qwen4exp-mmq-prepack-state.json`.

The GR up+sigmoid+branch-mean composite also registers the separate T0
`coltile2_branch4_rowbatch4_wave_scale_f32_exact` sibling. At rows512,
actual layer0 attention/FFN up banks improve1.032/1.039x without memory
preconditioning, with means and both order strata positive; layer4 confirms
1.036/1.032x. Gate and mixed F32 bits are exact,20 focused tests pass,
and both trace variants use72 VGPR/512-byte LDS/zero scratch. Small-row
unconditioned order reversals remain disclosed. Production selects the wave-scale
composite only in existing rows>256/four-branch scope after all72 model
trajectories pass exactly and all12 cases improve prefill and request wall.
Weighted prefill improves0.34-0.44%. The registered
parent and unfused projection/sigmoid/mean chain remain strict fallbacks.
Evidence: `2026-09-06-framework-qwen4exp-gr-wave-production.json`.

Grouped Q8 expert down has a T0 production rows>=512
`selected_grouped_row4_gemv_bf16_bf16_out` owner. Four independent rows
reuse decoded weights with the original reduction order. Actual layer4
weights/counts improve1.107x; layer30 weights with explicitly borrowed
layer4 counts improve1.137x, not actual layer30 routing.19 GPU tests pass;
trace remains24 VGPR/512B LDS/scratch0. Row1 remains the small-chunk owner/
rollback and strict selected GEMV remains the numerical fallback.
Evidence: `2026-09-06-framework-qwen4exp-q8-down-row4.json`.

Earlier default-off runtime admission selected row4 only within existing grouped
Q8 down for rows>=512. Five category/shape cases pass full logits/four decode
steps/state/full KV exactly, candidate calls4 or32 only in prefill, zero final
allocations.31 CPU route/profile/harness tests pass. Clean12-case throughput
A/B was the promotion blocker; both binders pinned0 at that stage.
Evidence: `2026-09-06-framework-qwen4exp-q8-down-row4-state.json`.

The `selected_grouped_row4_bundle_gemv_bf16_bf16_out` sibling
bundles the four original row reductions through one shared publication phase.
Actual layer4 weights/counts and layer30 weights/explicit borrowed layer4 counts
improve1.152x/1.153x versus row4; means/both orders positive.
Twenty-four GPU tests pass; trace24 VGPR/512B LDS/no spills unchanged.
Full-model engagement/state/KV and12-case A/B subsequently pass.
Evidence: `2026-09-06-framework-qwen4exp-q8-down-bundle.json`.

Earlier default-off bundle admission passes five full-model logits/state/KV cases,
four decode steps each; candidate calls4/32 only in prefill and final owners0.
Only existing row4 rows>=512 is eligible,both binders0 at admission.
Twenty-eight CPU tests pass.
Evidence: `2026-09-06-framework-qwen4exp-q8-down-bundle-state.json`.

Clean c0fc635b5 full12-case A/B passes72 exact trajectories and every
prefill/request wall. PP512/1024/4096 improves0.829%/0.671%/0.768%.
Production now binds bundle1,strict0; no new memory. Tiny aggregate decode
losses remain explicit in `2026-09-06-framework-qwen4exp-q8-down-bundle-production.json`.

Clean c7dd804cb full12-case A/B now passes72 exact trajectories and all
prefill/request-wall cases. PP512/1024/4096 improves0.822%/0.635%/0.705%.
Production binds1 for rows>=512; strict0. No new allocation or intrinsic
decode change. Evidence: `2026-09-06-framework-qwen4exp-q8-down-row4-production.json`.

The raw Q8 coltile family has a separate T0
`coltile8_rowbatch4_wave_scale_f32_f32_out` sibling. It makes the Q8 scale
block index wave-uniform while preserving original F32 FMA and reduction
order. On Framework gfx1151, under symmetric256MiB device-fill preconditioning,
actual attention-gate rows512 improves6.299->5.678ms (1.109x),
independently1.111x on layer4; shared-down improves1.089x. Both orders and
mean timings are positive under that condition. Unconditioned attention-gate
timing reverses by order; the original odd-pair1.20x claim is superseded.
Both kernels use72 VGPR,
512-byte LDS and zero scratch in the cached trace. Normal-model full72-trajectory
A/B now passes exactly and improves prefill0.75-1.29%, with every case's request
wall positive. Production selects the wave-scale sibling only for the exact
Q8 F32 coltile prefill route; strict retains original coltile. MMQ/WMMA and
decode stay unchanged. The adverse unconditioned microbench is preserved,
not relabeled. Evidence: `2026-09-05-framework-qwen4exp-q8-wave-scale-production.json`.

The Qwen4Exp serial GDN family registers the T0
`qwen4exp_sigmoid_register_prefill` candidate for Dk=Dv128.
It retains serial arithmetic and the FP32 state boundary while keeping state
across tokens. Framework gfx1151 complete-kernel tokens512 is 21.072->2.808 ms;
256 VGPR and 24-byte scratch are an explicit reviewed tradeoff.
Production now selects it at Hk16/Hv48/D128 and rows>=2 in the serial
branch only; strict retains the original registered serial kernel.
The full12-case A/B preserves all72 trajectories, improves prefill9.50-10.55%,
and speeds every complete request3.28-6.79%. Decode p4096 loses3.63%;
retention follows the owner's prefill-first direction, with decode followup.
Suffix tile16 scope is unchanged. Evidence:
`2026-09-05-framework-qwen4exp-gdn-register-production.json`.

The Qwen4Exp Q5_1 selected family registers an exact output-pair prefill
candidate that reuses BF16 activation loads across two output columns.
It preserves separate logical256 accumulations and reuses one LDS reduction
arena sequentially. Qwen4Exp production selects pair2 at rows>=64; M1 remains
the small-row and opt-out parent, and strict retains its original exact owner.

Its `selected_grouped_prefill_pair2_fold128_bf16_bf16_out`
owner folds the original first stride128 addition in registers and halves
dynamic reduction scratch, leaving the LDS64..1 tree unchanged.
Nine GPU tests pass (including K4096/CPU KL/top1), and captured layer0/1
code/mixed routing screens improve1.031x/1.036x with both orders positive.
Trace72 VGPR/no spills, dynamic LDS8672->4576B at K640. Existing pair2/M1
remain rollback/strict fallback routes.
Evidence: `2026-09-06-framework-qwen4exp-q51-fold128.json`.

Production rows>=512 `selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out`
combines folded partials for both output columns in one exact LDS reduction.
Actual captured code/mixed512-row screens improve1.163x/1.160x versus fold128,
both orders positive.13 GPU tests pass,trace72 VGPR/no spills;dynamic LDS
4576->8672B at K640 (not the rejected unfolded dual's16864B).
Small64-row candidate-first loses0.936x,so only rows>=512 is eligible for
admission;smaller rows retain sequential fold128.
Evidence: `2026-09-06-framework-qwen4exp-q51-fold-pair.json`.

Earlier default-off folded-pair admission passes five full-model logits/state/KV
cases,four decode steps each;25/200 candidate calls only in enabled prefill,
zero final allocations. Only fold128-selected rows>=512 eligible,both binders0.
Twenty-eight CPU route/profile/harness tests pass at that stage.
Evidence: `2026-09-06-framework-qwen4exp-q51-fold-pair-state.json`.

Clean f24130796 full12-case A/B now passes72 exact trajectories and all
prefill/request walls. PP512/1024/4096 improves1.950%/1.543%/1.896%.
Production binds folded-pair1 only at rows>=512,strict0;smaller folded rows
remain sequential. Adverse TG and increased dynamic LDS remain explicit.
Evidence: `2026-09-06-framework-qwen4exp-q51-fold-pair-production.json`.

Earlier default-off model admission passes five full logits/state/full-KV cases,
four decode steps each, with25/200 candidate calls only in enabled prefill
and zero final allocations.38 CPU route/profile/harness tests pass.
Only existing pair2 rows>=64 is eligible; both binders pinned0 at admission.
Evidence: `2026-09-06-framework-qwen4exp-q51-fold128-state.json`.

Clean318e0ad26 full12-case A/B passes72 exact trajectories and all prefill/
request walls. PP512/1024/4096 improves0.716%/0.978%/0.915%; production
now binds fold128=1, strict0, existing pair2 scope only. No new allocation;
tiny adverse decode rows preserved. Evidence:
`2026-09-06-framework-qwen4exp-q51-fold128-production.json`.

The Q4_K selected-prefill family has a separately registered exact bundled
publication sibling of its row8/output4/expertgrid64 owner. It preserves
per-row FMA and wave/serial-wave reduction order while publishing all
gate/up row sums in one LDS phase. The existing owner remains strict
fallback. Qwen4Exp production selects bundled publication in its exact
grouped-Q4 prefill branch; the WMMA suffix and c1 decode remain separate.

Its `selected_dual_grouped_pair2_bf16_bf16_out` sibling concurrently accumulates
two output columns with shared original-BF16 activation loads, preserving
each128-lane K sequence, wave reduction and BF16 gate/up boundary. The actual
Q4 gate/up+SiLU screen at tokens512 measures25.155->18.696 ms (1.345x);
an independent layer4 skewed map measures26.412->18.344 ms (1.440x).
gfx1151 trace:88 VGPR,4608-byte LDS, zero scratch. Production now selects pair2
at rows>=64 and supported K; smaller rows keep bundled output4, and strict
keeps its original exact owner. Full72-trajectory A/B is exact and improves
prefill5.44-5.90%; all12 request walls improve. Decode p4096 loses1.63% amid
drift, retained under the owner's prefill-first direction. Evidence:
`2026-09-05-framework-qwen4exp-q4-pair-production.json`.

The Qwen4Exp `attention/qwen4_exp_qsa.{hip,py}` family registers a separate
`strict_h256_wave_rows_spans` candidate for D=256 paged BF16 sparse rows.
Eight coordinates per lane preserve the parent 256-element score tree;
an explicit product register boundary prevents compiler contraction across
the parent's rounding point. `strict_rows_spans` remains its fallback.
Qwen4Exp production selects its qualified page256 sibling for sparse rows;
strict retains the original rows owner.
Its separately registered `strict_h256_page256_wave_rows_spans` sibling
specializes 256-token page addressing to remove runtime division/modulo;
it rejects other page sizes and retains the generic H256 wave parent.

External repositories are references, never the development tree. Before porting an externally derived family:

```bash
python3 scripts/check_lineage.py --kind kernel --diff stat
```

Useful filters:

```bash
python3 scripts/check_lineage.py --file '*paroquant*' --diff patch
python3 scripts/check_lineage.py --file '*DFlash*' --diff stat
python3 scripts/check_lineage.py --file '*MTP*' --diff stat
python3 scripts/check_lineage.py --fail-on-drift
```

`docs/source_lineage.json` is authoritative for baseline commits and external artifacts. If a source reports **DRIFT**, inspect the commit/diff and relevant worklog evidence before copying code. Update the baseline only as part of an intentional, logged source refresh.

Stable porting rules:

1. Develop and profile in this repository under `hipengine/kernels/<backend>/`.
2. Cite external source file and commit in the port commit message and worklog.
3. Preserve kernel math, launch bounds, and storage ABI during a mechanical port; do optimization as a later unit.
4. Replace `torch::Tensor`/framework bindings with raw pointers and explicit shapes/strides/dtypes.
5. Extract embedded source strings into real `.hip`/`.cu` files.
6. Register through `(backend, layer, quant, variant)`; do not add backend/quant branches to engine/model dispatch.

## Build layer

`hipengine.core.build` calls `hipcc` or `nvcc`, links a shared object, loads it with `ctypes.CDLL`, and caches by source/flags/compiler/target metadata under `~/.cache/hipengine/build/`. It does not use `torch.utils.cpp_extension`.

### Host uploads and reusable device state

Use `hipengine.core.memory.copy_host_array_to_device` for synchronous uploads
of temporary NumPy arrays. It retains the source through the copy, requires
C-contiguous storage, and checks both source and destination bounds. Passing
`host_array_ptr(np.asarray(...))` to the pointer-only copy API erases the owner
before the copy starts. A named local held through a synchronous copy is also
valid; no additional device-wide synchronization is required. Async uploads
need ownership through stream completion and are outside this helper's contract.

Reset mutable scratch at its use boundary when a kernel reads unwritten slots,
including masked slots in full-capacity GEMMs. Use stream-ordered byte clears
for zero initialization: multiplication by zero preserves NaNs. Do not poison
weights or initialized, read-only constants when testing scratch hygiene; test
request history separately from deliberate scratch corruption.

### HIP build profiles

| Profile | Important flags | Wavefront | Typical use |
| --- | --- | --- | --- |
| `decode` | local unroll threshold plus `-mcumode` | 32 | paged attention, GEMV, decode MoE |
| `prefill` | local unroll threshold, WGP mode | 32 | GEMM/WMMA and multi-row prefill |
| `baseline` | minimal flags | 32 | debug and fallback |

Wave32 is the gfx11 default. Use wave32 shuffles within a wave and LDS for cross-wave exchange. Wave64 is an isolated experiment only and requires explicit flags, probes, ISA checks, correctness fixtures, and end-to-end evidence.

### JIT cache and profiling

The env compiler-version cache in `hipengine/core/build.py` is keyed by override
identity (compiler plus the raw values of all four override vars), not compiler
alone: later environment changes re-resolve instead of reusing the first
version's build artifact. Resolution is order-independent and ~7x cheaper than
uncached.

A stale object can present as a kernel call hanging with the GPU idle. Remove only the affected family cache when known:

```bash
rm -rf ~/.cache/hipengine/build/<family>-<hash>*
```

Clearing the complete cache is acceptable when diagnosis cannot identify the family:

```bash
rm -rf ~/.cache/hipengine/build/
```

When profiling Python/ctypes JIT kernels, prebuild outside `rocprofv3` and make the profiled process cache-only. Do not let a profiler-injected child spawn `hipcc`/clang.

```bash
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 scripts/smoke.py --mode smoke-add-hip --n 1024 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-smoke -- \
  python3 scripts/smoke.py --mode smoke-add-hip --n 1024 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

To attribute cost across wave widths, trace two runs of the same workload at different row counts
and diff them per kernel with `scripts/gguf_rocprof_width_scale_diff.py`. It separates the two
signatures that look alike in a single trace: `per_row_launches` (launch count scales with rows,
i.e. one launch per row) and `per_row_inside_launch` (launch count flat, each launch longer).
Kernels present in only one run are reported as `only_in_base` / `only_in_candidate` rather than
dropped, which matters because an MTP verifier that engages only at rows >= 2 otherwise reads as
row scaling - the reason a rows-scaling trace must be taken with speculation removed from both
runs, not just from the summary. `scripts/gguf_packed_ar_rocprof.py` profiles this model but
builds two warmups (`c1` and `c4`) regardless of `--concurrency`, and `--skip-warmbuild` fails
inside `rocprofv3`, so budget 40-45 min per configuration or trace a narrower driver instead. Its
`_default_roctx_sdk` now falls back to the legacy `/opt/rocm/lib/libroctx64.so.4`, which is what
images without the pip ROCm SDK packages actually ship.

For Generation-2 GGUF owner profiling, use the mechanical isolated-cache
workflow instead of mutating the shared cache:

```bash
python3 scripts/gguf_continuous_owner_rocprof.py \
  --source-root /path/to/clean/source \
  --model /models/gguf/model.gguf --backend hip_gfx1151 \
  --compiler-version-file /tmp/hipcc-version.txt \
  --cache-root /tmp/lane/cache/<commit>/<compiler>/<profile> \
  --run-root /tmp/lane/profiles --run-tag c8-owner \
  --gpu-max-hw-queues 2 --rebuild --profile \
  --out /tmp/lane/profiles/c8-owner.json
```

`--rebuild` requires a new/empty scoped cache and never deletes the shared cache.
The workflow runs an unprofiled build child, snapshots every cache file and
build manifest, runs an unprofiled `HIPENGINE_REQUIRE_CACHED_BUILD=1` warm child,
then wraps only the final direct child in rocprof. A PATH compiler guard,
descendant-process monitor, and pre/post content/mode/mtime tree hashes reject
compiler activity or cache mutation. `HIPENGINE_BUILD_CACHE_ROOT` and
`HIPENGINE_REQUIRE_CACHED_BUILD` apply this policy to all HIP/CUDA builders in
the child, including lazy libraries that do not expose per-call cache flags.

Check expected kernel identity, plausible duration, workgroup/grid, VGPR, LDS, and scratch. `Scratch_Size > 0` on a hot path is a review trigger. Some profiler versions expose start/end timestamps instead of `DurationNs`; subtract them. Raw profiler dumps stay outside Git.

For MTP, profile the final child (`scripts/mtp_verifier_rocprof.py` or the final smoke), not the parent economics/prompt-suite harness that launches nested Python processes. Wrapper defaults, flag syntax, padding, compiler-probe, and PMC-counter traps for `rocprofv3` on this toolchain are cataloged in [`RDNA3-TUNING-GUIDE.md`](RDNA3-TUNING-GUIDE.md), section 4.9.

## Device-memory hygiene

`hipMalloc` returns zeroed pages for a fresh allocation, but a *recycled* block
keeps its previous contents at small sizes (measured on gfx1151: a 1 MiB block
comes back holding what was written to it, 4 MiB and above come back zeroed).
Three rules follow, and the first two were violated by the Surya and Evie
runners:

- **Re-zero device state with a memset, never with a scale-by-zero kernel.**
  `x * 0.0` is a no-op for a NaN or an Inf (`NaN * 0 == NaN`), so a recurrent
  state "cleared" that way keeps whatever the previous owner of the block left
  in it and turns every later output into NaN. `runtime.memset(ptr, 0, nbytes)`
  is the correct re-zero. That form is free in a hot loop: `hipMemset` acts on
  the NULL stream, which is the stream the runners launch their kernels on, and
  it only enqueues (measured on gfx1151: a 1 MiB memset behind 2.47 ms of
  queued work returns in 28 us, against 15 us for `hipMemsetAsync`), so a
  per-layer re-zero costs one enqueue and no host synchronization. This is why
  the Surya text decoder returned all-NaN
  logits after a long test suite and finite logits in isolation, and why the
  failure looked like device-state poisoning:
  `tests/test_live_surya_gpu.py::test_gpu_state_rezero_clears_recycled_nan` and
  `tests/test_evie_gpu_runtime.py::test_state_rezero_clears_recycled_nan` pin
  it by poisoning the state buffers with `0xFF` and requiring an unchanged
  result.
- **A kernel that reads a buffer before writing it is correct only by accident
  of allocation.** Any per-call buffer whose unread region is assumed to be
  zero is a latent full-suite failure. Poison it with `0xFF` in a test and
  require the result to be bit-identical. Re-zero it **where it is used, not
  where it is allocated** when the buffer is mutable request state. TimesFM
  2.5's full-capacity attention can read masked V slots left nonfinite by an
  earlier request; clearing the cache at use prevents `0 * NaN` contamination.
  TimesFM 3.0's `q_offset` is instead an initialized, read-only zero constant:
  poisoning it does not establish a read-before-write defect. Exclude initialized
  constants and weights from scratch probes. `tests/_poison_probe.py` snapshots
  reference output arrays, collects after warmup, rejects empty coverage, and
  raises when traversal limits prevent complete collection. Callers must still
  identify the expected mutable buffer families explicitly.
- **Bind the source of an H2D copy to a local, because the pointer is a bare
  address.** ``copy_host_to_device`` takes an ``int``, so the array has to
  outlive the call on its own. ``copy_host_to_device(buf,
  host_array_ptr(np.zeros_like(x)))`` does not: CPython drops the temporary's
  last reference when ``host_array_ptr`` returns, so the array is already freed
  and reusable *before* the copy is entered -- measured on gfx1151, the freed
  block is handed straight back to the next same-size allocation (same address),
  and the copy then reads whatever that allocation wrote. ``SuryaGpuRunner._upload``
  states the contract; hoist the temporary into a local. This made
  `tests/test_gpu_surya_kv_spans.py::test_scatter_f32_spans_honors_page_table_and_eviction`
  pass or fail depending on which Surya test ran before it -- 4 denormal
  values in the slots the scatter never writes. Hoisting is not optional and is
  not a style question: the same statement can pass alone and fail in a suite,
  because whether the freed block is recycled before the copy runs depends on
  what the *next* allocation does.
- **The transfer itself is complete when the copy returns, so a named local
  needs no synchronization.** Overwriting the source in place immediately after
  ``copy_host_to_device`` returns leaves the destination untouched at 1, 16, 64,
  and 128 MiB on gfx1151, so an unpinned source does not have to outlive the
  call and a per-call ``device_synchronize()`` buys nothing for source lifetime.
  Surya's upload helper uses that synchronous contract during loading, prefill,
  and decode and does not add a device-wide synchronization. An async upload
  must explicitly retain its source until stream completion; changing the copy
  API to async requires updating its callers. Do not add a defensive device-wide
  drain to a synchronous upload. `tests/test_gpu_device_memory_hygiene.py`
  enforces the always-allocating forms (`np.zeros*`, `np.ones*`, `np.full*`,
  `np.array`, `np.asarray`, `np.tile`, `.astype(...)`, `.copy()`, `.flatten()`)
  by AST scan over `hipengine/`, `tests/`, `scripts/`, and `benchmarks/`.

## Registering a kernel

Wrappers register explicit keys:

```python
from hipengine.kernels.registry import KernelKey, register

register(
    KernelKey(
        backend="hip_gfx1100",
        layer="paged_attn_decode",
        quant="w4_paro",
        variant="gqa_splitk_spans",
    ),
    paged_attn_decode,
)
```

The resolver tries exact variant, no variant, same-backend FP16 fallback, then CPU-reference candidates. Code that needs to know whether a *specific* optimized key exists must use `is_registered()`, not broad fallback resolution.

Execution profile does not change `KernelKey`. Model/session construction
resolves `strict`, `production`, or `batch_invariant` to an immutable selection
of existing variant keys plus a strict fallback for each production selection.
Dispatch consumes that plan; do not add profile branches or a fifth registry
axis. Artifacts record the selected and strict manifest hashes.

Backend packages may refresh missing keys after test isolation. `hip_gfx1151` aliases only allowed gfx11 registrations; `cuda_sm120a` registers only independent CUDA implementations.

## Correctness and profiler gate

A new or ported kernel lands only when all applicable checks pass:

1. **Declaration:** name execution profile, T0/T1/T2/T3 source, supported
   backend/model/quant/shape envelope, and strict fallback.
2. **RED fixture/oracle:** write or identify the strict/CPU/primitive oracle
   before implementation when math or storage changes.
3. **Registry:** exact intended and strict-fallback keys resolve under the correct backend, layer, quant, and variant; manifest selection adds no fifth axis.
4. **Numerics:** the CPU-reference KL ≤ 0.05 / top-1 ≥ 90% outer floor passes.
   Strict preserves its exact/parent boundary. Production additionally passes
   calibrated strict-teacher mean/tail/max KL and top-1 by category/shape/
   transition, same-schedule determinism, isolation, BF16-relative, and task
   gates.
5. **Fallback:** every fused/production composite retains its registered strict unfused chain.
6. **Profiler:** cache-only `rocprofv3 --kernel-trace` or Nsight trace names the expected kernel with plausible resources/duration.
7. **Integration:** run the narrowest applicable strict, production, or batch-invariant model/dynamic gate from `TESTING.md`.
8. **Evidence:** performance claims follow `BENCHMARK.md` and record profile/schema and selected/fallback manifest hashes in artifact/rollup/changelog/worklog; do not add the narrative here.

## Per-family port checklist

1. Audit `source_lineage.json` and run the narrow lineage check.
2. Declare the execution profile/arithmetic class and add the strict exact/
   parent-parity or production numerical fixture (RED), plus the CPU-reference
   outer oracle.
3. Copy one functional family into `hipengine/kernels/<backend>/<family>/`; do not mix unrelated families.
4. Retype launch wrappers to raw pointers and explicit metadata.
5. Preserve or document storage layout, low-precision boundaries, `KVLiveSpans`, launch bounds, and build profile.
6. Register exact four-axis keys and the required strict unfused/fallback keys;
   profile selection remains outside the key.
7. Update the relevant catalog row and fused fallback map without benchmark commentary.
8. Run registry, declared-profile numerical/control, profiler, and narrow integration gates.
9. Record decisions/results in a new immutable worklog entry; write compact benchmark artifacts only when making a performance claim.
10. Commit the validated family as one logical unit with source commit provenance when ported.
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
