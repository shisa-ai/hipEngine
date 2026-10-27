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

The gfx1151 Flash-Next UD-Q4_K_XL production profile selects guarded Q8
block-scale WMMA, exact QSA-prefill grouping and verified ordered QSA decode
through `hipengine/generation/qwen4_exp_profiles.py`. Exact grouped/risk-repair owners,
register-state GDN and mapping-only PLE advice remain active. Legacy FP16 Q8
selected-WMMA, dense MMQ/iu8, GR iu8, tiled GDN and DP4A remain registered
but are not defaults for this profile.
Q4_K_M profile selection is unchanged; raw-pointer and KVLiveSpans ABIs are
unchanged.

Dense gfx1151 Q4T16 gate/up prefill selects the existing fused row48 owner
at rows33-48 (`GGUF_Q4_DUAL_SILU_PREFILL_ROW48_MAX_ROWS=48`), row64/row128
above that band, and the registered unfused chain as fallback. The
Qwen3.8-27B Q4_K_M qualification is in
`benchmarks/results/2026-09-12-gfx1151-qwen38-row48-prefill-retained.json`.

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
| `gemma4_vision.py` | Gemma 4 image encoder, spatial pooling, standardization and text projection. |
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
| Gemma BF16 attention | `gemma4/gemma4_attention.{hip,py}` | masked ungated prefill/decode, strict class, owned global-score class, and key-sliced flash decode | Head dimensions 256/512 move logits to stream-owned global scratch when the resident row exceeds 64 KiB. Query batches of at most four bound per-layer global scratch; the existing 256-lane denominator and ordered V chains are unchanged. The strict class/block routes remain the short-context arithmetic oracle. Flash decode splits the key range into per-slice `(m, l, o)` partials and combines them; the slice count is the smallest one whose per-block LDS logits fit 64 KiB, raised to the occupancy target when that already fits. A one-token decode takes flash from 1024 keys, and from 256 keys when the plan selects the `gemma4_flash` decode variant. The `gemma4_flash_long` variant raises the count to one slice per 128 keys (at most 1024) past 8192 keys and equals `gemma4_flash` below. Under either flash variant a 2-16 row verify block is one flash launch over the block's key range, each row bounded by its own mask row; outside a graph capture a group kernel serves up to 16 query rows per block (eight sliding or two global tokens) sharing each slice's K and V loads, with one thread per query-key dot product (production arithmetic). Other head geometries keep their resident LDS capability bound. |
| Full/paged attention | `attention/paged_attn_decode.{hip,py}` | `full_attn_decode/prefill`, `paged_attn_decode/prefill`, `full_attn_gate_mul` | Contiguous and paged, batched, GQA, split-K, gated reduce, and supported INT8 KV variants. Per-token/head INT8 includes a row-batched 24Q/4KV/D256 split-K producer plus explicitly strided BF16 gated reducer; the c1 leaf remains registered as its numerical fallback, and the gfx1100 Qwen3.8-27B artifact qualifies the batch variant to physical c4 with the c1 leaf as the registered fallback above that width. gfx1151 Qwen3.5-0.8B rows1/8Q/2KV/D256 selects generic split-K3+fused BF16 gate at cap514-641. The private-c1 exact leaf is the fixed256 body at 256 threads (strict exact default) with a parameterized `fixed256_threads_spans` probe at runtime block width; gfx1151 promotes 1024 threads (T2 non-exact, execution-profile gate-passed) via `GGUF_SHORT_C1_BATCH_ATTN_THREADS`. Dense H5120/L64/24Q/4KV/D256 selects the BF16 grouped-GQA split producer from context 4096; shorter contexts and unsupported shapes/backends retain the generic producer. | Native BF16-gated prefill uses owned global score scratch when its context-sized shared allocation would exceed 64 KiB; bounded query batches reuse the split partial-output arena without changing the parent reduction order. The explicit `causal_gqa_gate_bf16_global_scores` variant also permits parent-parity checks at short contexts. INT8 per-token/head includes a row-batched 24Q/4KV/D256 split-K producer with an explicitly strided BF16 gated reducer; the c1 leaf remains its registered numerical fallback. |
| AOTriton adapter | `attention/aotriton_wrap.py`, `attention/aotriton.py` | `full_attn_prefill` (`w4_paro`, `gguf_qwen35`) | Optional library adapter; native raw-pointer paths remain available. |
| Linear-attention Conv | `linear_attn/conv.{hip,py}` | `linear_attn_*conv_decode/prefill`, chain/tree and snapshot composites | Decode, segmented prefill, verifier tree/chain, and state-snapshot variants. |
| Linear-attention GDN | `linear_attn/gdn.{hip,py}` | `linear_attn_prefill_prepare`, `gdn_*recurrent*`, RMSNorm/gate/rotate/cast/snapshot composites | Exact schedules retain FP32 recurrent state; segmented, chain/tree, snapshot, and decode-order writers cover prefill, verifier, and multi-request selected commit, with optional FP32 state-row journals, direct BF16 handoffs, and an exact FP32 output tap. FP16-state (FP32 accumulation) and gfx1151 cluster/chunked compact-peer variants are explicit opt-ins or capability selections that always retain an FP32 fallback. |
| Runtime state | `runtime/state.{hip,py}` | token embedding, positions/metadata, graph record/commit, scalar state, profiling wall-clock marker, `mtp_draft_token_index/w4_paro/remap_prefix_tail_i64` | Device-side graph/verify bookkeeping, indexed row state, token publication, and profiling-only steady-clock boundaries. The int64 draft-index map keeps prefix IDs and maps compact extra rows to global IDs in place; invalid indices become −1. |
| Sampling | `sampling/sampler.{hip,py}` | `sampler`, `mtp_draft_topk` | Greedy/temperature/top-k helpers and bounded draft top-k. Full-vocabulary `sorted_rows_i32` uses 256-key tile sorting, parallel merging and FP64 scans of FP32 weights; the original temperature/top-p variants remain registered strict fallbacks. Caller-owned scratch is `rows * (24*vocab + 8*ceil(vocab/256))` bytes. gfx1151 correctness and trace evidence are in `tests/test_gpu_sampler_full_vocab.py` and the fast-sampling worklog; gfx1100 hardware transfer is unverified. |

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
| `quant/gguf_t16_selected_gemv.hip` | T16 selected-expert projections and weighted/residual composites; `selected_chunk8_gemv` is the Q4T16 production sibling (eight-k runs per thread, one wave per tile). |
| `quant/gguf_k_t16_selected_prefill.hip` | T16 Q5_K/Q6_K selected-expert prefill projections. |
| `quant/gguf_q4_k_t16_selected_prefill.hip` | T16 Q4_K selected-expert prefill, including an LDS-transposed fused expert-major gate/up consumer. |
| `quant/gguf_q5_k_qmicro_planar_gemv.hip` | Planar qmicro Q5_K selected-expert projections. |
| `quant/gguf_q8_0_t16_gemv.hip` | T16 Q8_0 decode projections. |
| `quant/gguf_q8_0_t16_prefill.hip` | T16 Q8_0 prefill projections with FP32 repair of non-finite WMMA accumulators. The two-wave and four-wave launches also have an `f16w` form that loads a pre-converted half weight operand instead of decoding it, bit-identical to the decoding form; `q8_0_t16_tiles_to_f16w_kernel` writes that operand from the T16 tiles. |
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
| `quant/qwen4_exp_q5_1.hip` | Raw Q5_1 selected-expert projections; `selected_pack8_chunk8_gemv` is the pack8 production sibling (eight-input runs per thread, one wave per pack). |
| `vision/qwen4_exp_vision.hip` | Vision normalization, activation, residual, and attention operations. |

| Quant/layout family | Source / wrapper | Principal registry layers | Stable notes |
| --- | --- | --- | --- |
| Q4_K selected prefill | `quant/gguf_q4_k_selected_prefill.{hip,py}` | `moe_linear` (`gguf_q4_k`) | Raw selected-MoE compact/tile-map ABI, paired gate/up, three-plane iu8 risk collection and sparse strict-order repair. The checked-index `selected_dual_wmma_iu8_risk_narrow_prefill_bf16_bf16_out` sibling is diagnostic: int64 host/map ABI, signed-int32 device intermediates only after complete activation/output/weight/map/risk byte-range checks. Wide iu8 and strict grouped chains remain registered; no production selector changes. |
| Q4_K selected FFN megakernel | `quant/gguf_q4_k_moe_ffn_fused.{hip,py}` | `moe_ffn_selected` (`gguf_q4_k`) | Whole selected gate/up → SiLU → down projection; primitive selected projections remain fallback. |
| Qwen4Exp GR/PLE/GDN | `fused/qwen4_exp_gr.{hip,py}`, `fused/qwen4_exp_ple.{hip,py}`, `linear_attn/qwen4_exp_gdn.{hip,py}` | grouped GR read/write, sparse PLE gate/Conv/add, `gdn_recurrence_norm_gate` (`f32_state`) | Strict raw-pointer primitives for four authoritative BF16 branches, FP32 PLE history/compute, and FP32 recurrent state with sigmoid output gate. Grouped norm registers `f32_f16_publish` and strict `f32_f16_publish_unfused` variants: the publisher emits the same F32 norm and existing-cast F16 bytes into disjoint caller-owned buffers on one stream. Six-case/420-call complete norm+cast+HC-down timing wins at512/4096 but loses at1024. The named gfx1151 UD-Q4_K_XL production binder enables `HIPENGINE_QWEN4_EXP_GR_NORM_F16_PUBLISH` with a manifested producer selection and explicit parent rollback; strict and Q4_K_M bind it off. Registry-axis policy admits exactly4096 rows at K10240/N320 in layers13–47 and lends the existing session workspace to one immediate matching consumer, with alias-safe fallback and no added device allocation. Fourteen native canonical/heldout state/KV/logit cases and72 exact canonical engine trajectories pass; weighted4096 prefill improves0.154%, aggregate request wall0.9970x and worst request regression2.77%, so no all-request/decode-speed claim follows. Cached native trace confirms two publishers replace two casts, four padded GEMMs and scratch0; the parent norm+cast chain stays available. Evidence: `benchmarks/results/2026-10-06-qwen4exp-norm-f16-publish/`. The retained `gr_gated_mean_sigmoid` owner preserves both materialized F32 gate and mixed output bit-for-bit and removes one launch through rows<=256, with registered `strict_unfused` fallback. For rows>256, the retained raw-Q8 up composite preserves each coltile8 reduction while grouping two hidden columns across four branches and emits both gate and mean: clean p508 is 91.158→91.600 tok/s and code-p1024 is 88.754→89.239 tok/s with 450/450 logits and 18/18 state/tasks exact. The primitive coltile plus GR epilogue remains fallback. GDN has c1 decode plus a row-bulk sibling that is bit-exact to serial recurrence. The GDN family also registers a T0 tile-16 raw-Q/K staging sibling for Hk16/Hv32-or-48/D128 prefill; the columnwarp parent and serial strict route remain registered fallbacks. Qwen4Exp K4 Conv now has a separately registered bulk prefill owner that emits the same contraction sequence as serial decode per row; output/state are F32-bit exact, p508 Conv compute launches fall 18,432→72 (plus 72 final-state launches), and the serial owner remains fallback. The gfx1151 recurrence trace records the bulk symbol at 17,474 ns for a five-row reduced fixture. The registered `qwen4exp_sigmoid_peer_prefill` host composite chains Qwen4Exp prepare, compact peer-wave32 recurrence, and sigmoid gate. All-layer arithmetic fails the full numerical envelope, but the named gfx1151 production profile certifies global layers 35–47 (actual GDN layers 36/37/38/40/41/42/44/45/46): the complete stack passes 448/450 top-1 with no scope failures. At p508 it replaces nine exact fused launches with 26.77 ms total peer work, reducing the traced GDN family 992.16→750.68 ms; `qwen4exp_sigmoid_strict_prefill` and c1 remain fallbacks/oracles. |
| Qwen4Exp QSA | `attention/qwen4_exp_qsa.{hip,py}` | `qsa_split_norm_rope`, `qsa_norm_rope`, `qsa_pool_norm_rope`, `qsa_index_score`, `qsa_select_blocks`, `qsa_sparse_attention` | Split-half partial RoPE, FP32 raw-key complete-block pooling, deterministic lower-start tie break, and sparse original-BF16-K/V GQA. The exact c1 index append has a registered device-position sibling for graph-owned decode control; scalar/row append remains fallback. c1 plus explicit-position row-bulk Q/K/gate and index-query transforms are registered; reduced gfx1151 three-row traces are 2,204/2,124 ns and bit-exact to c1. Variable-selection sparse rows consume complete paged spans and trace at 7,213 ns on a reversed-page fixture; non-flash multirow dense rows use the exact fixed256/precomputed-offset/vector2 owner (real primitive 6.846→2.485 ms, clean p508 91.529→92.442 tok/s, code-p1024 89.150→90.634 tok/s), with generic FP32 batch context fallback. A bounded prompt-chunk mixer composes bulk quant projections, exact row transforms, shared K/V writes, dense batch context, and variable-selection sparse context. Its block-table-aware raw index-key scatter replaces p508's 6,096 per-row D2D copies with 24 chunk kernels and cuts p512 trace launches 11,053→4,933 with bit-exact logits; c1 append remains fallback. Its reduced six-row dense→sparse boundary matches independent c1 output/state and traces the dense/sparse leaves at 3,927/6,132 ns on gfx1151. The corrected exact chunk path uses chunk-batched PLE staging, batched projections, decode-order-exact bulk causal Conv, and exact grouped Q5_1 down pass all 687 teacher-forced rows bit-for-bit and improve the natural suite 5.265→12.117 tok/s (2.301x); warm p512 is 16.555 tok/s. The former size-2 smoke remains historical (`KL_teacher=0.00510`, `KL_serial=0.00410`), while approximate size 9 is rejected (`KL_serial=0.09754`; artifact: `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-chunked-prefill-smoke.json`). The first real sparse row at token 2,052 also passes; promoted chunk64 is bit-exact to serial, both have teacher KL `7.65e-5` and top-1 264, teardown is clean, and prefill improves 370.565→136.129 s (2.722x), as does a repeated-token structural 4K checkpoint (`KL_teacher→serial=4.40e-5`, `KL_teacher→chunk=4.78e-5`, diagnostic 854.982→574.759 s). A chunk-only repeated-token 16K checkpoint further passes teacher KL `7.55e-5`, top-1 264 exact, and clean teardown in 2,434.172 s; strict remains measured through 4K. A chunk-only repeated-token 64K checkpoint also passes teacher KL `5.74e-6`, top-1 264 exact, and clean teardown in 10,336.580 s. Real full-capacity ownership allocates and tears down at 262,144 tokens (91,126,119,496 tracked bytes, 38,915,162,112 physical bytes still free, zero tracked bytes after close), but this is not a 262K inference result. Natural 4K retrieval and Transformers index-reference control pass exactly. Persistent compressed-key preparation reduces pool launches 24,540→384 and block work 18,849,792→12,288; exact device radix top-512 removes 24,540 score D2H synchronizations and 403.341 MB metadata H2D, reducing natural 4K 303.528→294.434 s with unchanged output/control. Production wave32 H128 sparse attention improves its real 2,048-token primitive 1,982→1,796 us and paired natural 4K 298.078→290.941 s; four sparse categories have bit-exact final logits/control and strict spans remain fallback. Exact chunk-batched score/top-k reduces launches 49,080→768 and paired natural 4K 295.706→290.971 s; exact grouped rowbatch8 Q4_K gate/up then gives 291.624→231.798 s, and output4 scheduling cuts full-shape CTAs 75% plus paired wall 235.774→228.569 s, all with bit-exact logits/control. The exact owner now also covers Q8_0-down layers, removing 64 direct gate/up launches and improving paired p508 12.021→11.189 s (45.404 tok/s). Its current sibling predecodes exact `d*scale`/`dmin*min` metadata once into 2 KiB LDS; with chunk256 this reaches 51.220 tok/s first-run / 58.466 tok/s steady p508 and 55.046 tok/s p1006, all bit-exact. Natural 16K/64K now pass at 17.301/17.099 tok/s with retrieval/control/CPU-oracle/lifecycle exact; 262K execution and broader lifecycle gates remain open (`benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-2052-transition.json`, `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-4k.json`, `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-16k.json`, `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-64k.json`, `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-262k-capacity.json`). The complete runner mirrors paged K/V physical ownership, uses dense equivalence through 2,051 tokens, then runs native projections/pool/score/sparse attention with device radix top-k selection and expansion; the CPU reference preserves lexicographic tie ordering, and the single-thread device selector is a reduced-fixture oracle rather than the long-context route. For gfx1151 c1 H256 indexed-sparse decode, production selects an exact ordered three-pass owner: parallel QK scores preserve the strict reduction tree, one global selected-order recurrence emits online-softmax coefficients, and output-column recurrences consume them in the same order. The serialized strict owner remains the registered fallback; the promoted ordered-v2 rewrite (`strict_ordered_three_pass_v2_spans`) preserves that arithmetic and operand order while de-latencying each pass: warp-tree scores on an eight-token grid reproduce the strict reduction tree, an exact `fmaxf` block scan supplies the coefficient max trajectory with pointwise `expf` off the critical path and a prefetched serial denominator, and staged-tile values use clamped unconditional loads after a per-load select was shown to defeat memory-level parallelism (406 versus 122us). Named kernel medians at clean source, cache-only build: scores 33.5us, coefficients 14.3us, values 100.7us (VGPR 32/40/96, scratch 0) versus parent 399.7/176.1/533.0us; leaf route 1.154->0.179ms/layer, 6.45x, bit-exact; six-case off/on/off full logits/4-step/state/full-KV gate passes at committed source; canonical A/B 72 trajectories exact with p4096 weighted TG +14.705% (arithmetic-mean-rate ratio +16.516%). Evidence: `2026-09-08-framework-qwen4exp-qsa-ordered-v2-kernels.json`. |
| Qwen4Exp vision | `vision/qwen4_exp_vision.{hip,py}` | `vision_layernorm`, `vision_add_bias_residual`, `vision_gelu`, `vision_attention` | <=1K Qwen3-VL-compatible images/videos: merge-compatible RGB grids up to 256 patches/temporal pair, 2×2 block-major order, align-corners learned-position interpolation, frame-pair attention isolation, multiple images, odd-frame duplication, and typed placeholders. FP32 attention uses explicit vision H/W RoPE. Full 32×64 encoder matches Transformers at relative L2 1.48e-6/cosine 1.0; text QSA's registered MRoPE sibling applies interleaved T/H/W `[11,11,10]` and traces at 12,143 ns. Bounded PNG data URLs work through non-streaming chat; remote URLs/SSE/>1K remain open. |
| Qwen4Exp raw Q5_1 experts | `quant/qwen4_exp_q5_1.{hip,py}` | selected `linear`/`moe_linear` (`gguf_q5_1`) | Strict selected-expert consumer plus exact grouped rowbatch8 and grouped-WMMA down projections for the pinned Unsloth UD-Q4_K_XL mixed quant. Exact output8 scheduling cuts full-shape grouped-down CTAs 1,310,720→163,840 and paired natural 4K 237.131→222.228 s with exact logits/control; output1 remains fallback. The current short-prefill owner iterates 512 experts through 64 worker CTAs and uses 128 physical threads to materialize the same 256 logical partials before the original reduction tree; its p512 bucket is 3.470→2.534 s with exact bits. Q5_1 grouped WMMA is not the strict owner; explicit gfx1151 Qwen4Exp `production` selects it with cooperative Q4 gate/up on the definitive maximal suffix layers 27–47. Every layer 0–26 fails final-prompt mean or p95; the 27–47 450-row/three-repeat manifest passes at mean/p95/p99/max KL 1.05e-4/3.81e-4/1.52e-3/5.59e-3 and 99.556% top-1, improving the MoE-only p508/p1012 59.401→67.243 / 58.723→66.268 tok/s. The same explicit profile adds dense-Q8 WMMA on certified layers 32–47; the combined 450-row gate passes mean/p95/p99/max KL 1.20e-4/4.93e-4/1.72e-3/8.69e-3 and 99.778% top-1, reaching 73.361/71.834 tok/s. Exact grouped/coltile fallbacks remain registered. The strict selected decode default now uses 64 physical threads to materialize the same 256 logical partials before reconstructing the original shared strides 128/64/32 and wave32 tail. The first exact t128 contraction cuts Q5 cycle-wall 692.930→410.364 ms and graph decode 11.380→12.140 tok/s; t64 is BF16-bit exact to both registered t128/t256 fallbacks, cuts its matched Q5 trace 444.699→362.525 ms, and improves graph decode 13.077→13.302 tok/s (+1.69%). The c1 default also fuses selected down with routed weighted sum: one CTA per H=2560 output preserves every route BF16 result and the original ordered `fmaf`, removes 1,806 traced launches, contracts target cycle-wall 369.241→313.535 ms, and improves 13.379→13.523 tok/s (+1.06%); the separate exact chain remains fallback. A default-off 64-thread sibling improves warm decode further but is rejected for production mean/p95 KL (`0.002565/0.007202`). |
| Raw Q5_K/Q6_K/Q8_0 | `quant/gguf_k_gemv.{hip,py}` | `linear`, `linear_pair`, `attention_projection_quad` | Decode/prefill, BF16/F32 output, pair/quad launch contractions, rowbatch/coltile variants. The gfx1151 Qwen4Exp exact Q8/F32 owner first cut p508 26.264→14.718 s with coltile4/rowbatch8, then promotes coltile8/rowbatch4 alongside exact expert scheduling to reach 42.376 tok/s; p512 Q8 kernel wall falls 3.121→2.482 s with bit-exact full logits. Its c1 F32/F32 output-pack8 sibling reuses each activation across eight columns without changing per-output arithmetic, cuts the traced Q8 bucket 2.620→1.171 s and paired decode 5.698→6.305 tok/s; registered scalar raw Q8 remains fallback. A separately registered diagnostic `coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out` sibling narrows the pipelined DPP leaf's offsets to int32 and drops per-row control for complete four-row groups after signed-int32 float activation/output/raw-weight byte-range and grid admission (scalar dimensions validated before the packed row byte count is formed). No production selector points at it; callers retain the generic DPP parent outside the admitted domain. gfx1151 Q5/Q6 W7900 policies remain disabled. |
| Q5_K/Q6_K selected prefill WMMA | `quant/gguf_k_selected_prefill.{hip,py}` | `moe_linear` | Raw-byte compact selected-MoE f16-WMMA consumers with strict raw selected-gemv fallbacks. The gfx1151 Qwen4Exp layer-2 Q5_K/Q5_K route is production-rejected/default-off: p508 Q5_K gate/up falls 279.86→16.66 ms and 20/20 category-balanced p512 pairs win by about 5%, but the complete 450-row gate fails prefill-last mean KL at 0.001179 > 0.001. Do not rescreen unchanged T2 arithmetic; the older optimized metadata-hoist sibling is a separate rejected path. |
| Raw Q3_K selected | `quant/gguf_q3_k_gemv.{hip,py}` | `moe_linear` | Q3 selected-expert projection family. |
| Q4_K pack8/raw | `quant/gguf_q4_k_gemv.{hip,py}` | `linear`, `linear_pair`, `linear_pair_silu`, `linear+residual` | Raw GGUF math and lossless pack8 layouts; pair/SiLU and exact rounded-BF16 residual composites where registered. Qwen4Exp c1 now resolves raw selected dual gate/up by registry capability, halves Q4 launches 94→47/token, and improves paired decode 6.065→6.223 tok/s. Its operation-complete sibling preserves both BF16 projection boundaries and the standalone SiLU/product bits, removes another 47 launches/token, and improves 6.400→6.420 tok/s. The selected default now maps logical lanes `tid`/`tid+64` onto 64 physical threads while publishing the same four strict wave sums; it contracts Q4 cycle-wall 1,076.767→814.906 ms across 1,974 launches and improves counterbalanced graph decode 12.003→13.167 tok/s (+8.84%). IDs/full logits are exact and the physical128 dual/singleton chains remain fallbacks. Above these kernels, gfx1151 now captures each complete stateless Qwen4Exp MoE chain in one self-validating request-owned graph: 48 captures/zero rejects, 192 full-logit rows exact, eager 6.511→11.515 tok/s, then exact Q5/Q4/Q5 contractions and Q5 down+weighted fusion reach 12.140/13.167/13.302/13.523 tok/s; c2 is exact and stateful GDN/QSA remain outside replay. Earlier gfx1151 production measurements used one-plane Q8_1 DP4A Q4 dual+SiLU on static layers `0,2,5,6,8,9,10,11,13–47`; measured-failing layers `1,3,4,7,12` remain exact. The physical64 owner preserves the candidate's 128 logical partials and BF16 boundaries; combined production passes 447/450 top-1 with mean/p95/p99/max KL 2.72e-4/1.40e-3/4.00e-3/5.77e-3, improves decode 13.880→15.543 tok/s, and contracts Q4 target cycle-wall 825.340→397.755 ms. Direct suffix13→calibrated43 is +0.37%. Suffix12 and all-layer DP4A are rejected at 445/450; UD-Q4_K_XL production and fast now select strict logical128/t64; the one-plane DP4A variants remain registered diagnostics (`benchmarks/results/20261022-qwen38fn-q4-decode-repair.json`). A Qwen4Exp one-layout expert replacement is rejected and removed: sampled layer-0 bits are exact and micro speed is 4.47x, but uncached load is 979 s and full-model mean/p95 KL fail at 0.002089/0.006529. Primitive projection+add fallbacks remain available. |
| Q4_K/Q6_K prefill WMMA | `quant/gguf_q4_k_prefill.{hip,py}` | `linear` | Resident pack8/raw prefill consumers; exact scalar/pack8 routes remain fallbacks. The p512 pack8-Q4 rounded-residual output-store sibling is rejected (0.958x core / 0.952x public complete-model prefill) and is not registered. |
| Q8_0 T16 prefill | `quant/gguf_q8_0_t16_prefill.{hip,py}` | `linear`, `linear_pair` | WMMA/T16 Q8 prefill and architecture-specific wave schedules. gfx1151 rows512/K1024/N16+N16 alpha/beta uses the exact two-wave dual owner; singleton WMMA remains the fallback. |
| Q8_0 block-scale selected prefill | `quant/gguf_q8_0_prefill.{hip,py}` | `linear` | gfx1151 UD-Q4_K_XL selects `selected_grouped_blockscale_guarded_prefill_bf16_bf16_out`: raw Q8 codes and BF16 activations feed BF16 WMMA, followed by FP32 block scaling and sparse strict-order correction of rounding-risk outputs. Uses compact/tile-expert maps and request-owned full-capacity risk buffers; grouped GEMV remains the strict fallback. Unguarded block scaling is diagnostic only. |
| Q8_0 wide-row dense prefill | `quant/gguf_q8_0_dense_wide.{hip,py}` | `linear` | F16-operand/F32-accumulator raw-Q8 tile family (`dense_wide256`, `dense_wide64x256`, `dense_wide128x128`, `dense_wide64x128`) with F32 or preconverted F16 activation ABI. Both operands are staged in padded LDS; the 128-column x 256-row tile uses 55,296 bytes. Source lineage: comparator `mmb_dense_kernel<128,256,64,64,1>` in `ggml/mmb.cu` at `c4aa3022`. Registered under gfx1100 and independently aliased to gfx1151; model-plan layer scopes select production use. Exact raw-F32 coltile remains the strict fallback. The diagnostic `dense_split64x128_f32_f32_out` sibling splits F32 activations and dequantized F32 weights into F16 high/scaled-low planes, averages opposite-sign high-high WMMA dots and uses compensated F32 summation, storing TwoSum corrections in the scaled cross accumulator. Both cross products start from zero WMMA accumulators and sum in F32. Its doubled operand LDS is 55,296 bytes plus a range flag (55,808 allocated); it omits the low-low term and recomputes any block with an out-of-range operand through the strict F32 in-kernel fallback. The failed model composition's runtime selector and full-model gate modes are removed. The leaf remains independently registered for operation/F64 and natural-input diagnostics, with no generator binding; its numerical RED coverage and failure packets are preserved. The identical-product `dense_wide256_split3` export is a timing-only probe, not a numerical split. The `dense_wide256_padded` F32/F16-input siblings compute all zero-padded WMMA row fragments inside the K loop while preserving bounded publication and valid-row accumulation; full tiles keep the tail-free parent. Only the F16-input leaf has production admission: gfx1151 Qwen4Exp UD-Q4_K_XL, existing dense-wide layers13–47, K10240/N320 and rows512–4096, after successful caller-owned F16 staging. Backend registry-axis capability metadata bounds the rewrite; missing registration keeps the F16 parent, while strict keeps raw-F32 coltile. The named profile manifests the bounded selection and `HIPENGINE_QWEN4_EXP_DENSE_WIDE_TAIL_VARIANT` parent rollback. Natural owner,20-case full-state/KV and clean12-case engine A/B evidence: `benchmarks/results/2026-10-06-qwen4exp-dense-wide-padded-gate/`. F32-input padded and other shapes remain diagnostic. |
| Q8_0 T16 decode | `quant/gguf_q8_0_t16_gemv.{hip,py}` | `linear`, `linear_pair`, `linear_triple` | Exact T16 Q8 decode GEMV for Qwen3.5-family attention projections (in 2048; fused qkv 8192 + gate 4096). Per-row dual/split owners run at all widths, with an exact 128-thread dual-split rowtile col8 pair owner admitted at rows >= the backend-package floor `GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS`. |
| Q4/Q5/Q6 T16 selected | `quant/gguf_t16_selected_gemv.{hip,py}`, `quant/gguf_k_t16_selected_prefill.{hip,py}` | `linear`, `linear_pair_silu`, `moe_linear`, `moe_linear+weighted_sum`, `linear+residual` | c=1 and selected-prefill T16/qmicro/interleaved consumers, including weighted/residual composites. Exact one-wave/shared-B WMMA rowtile owners cover physical shapes, with grouped-grid siblings, fused dual+SiLU prefill owners, and input-F16 activation siblings (`*_fp16_in_bf16_out`) registered per backend; every shape/row miss and env-disabled path retains the strict one-wave/shared-B or primitive fallback, and current per-shape ownership is backend-package capability data. gfx1100/W7900 additionally retains the exact c8 Q4 selected gate/up pair-reuse dual owner through the package floor `GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS` (2026-09-05 audit packet D1: native-c8 +4.97% with exact repeatable trajectories, arm-identical state differentials on steady c2/c4/c8 and c8 shrink-sparse, natural-prompt duplicate-lane fraction 0.616 versus the direct-fixture 0.5, and a c8 census showing `q4_k_t16_selected_dual_pairreuse_direct_gemv_kernel` x80 with zero scalar fallbacks); the route's geometry gate pins it to x_rows=8/rows=64, lower widths and env-0 keep the per-row dual owner, and the selected-down/Q6-down pair-reuse packets stay unqualified on gfx1100 (floors 0). |
| Q6/Q4 mixed and narrow K/V grids | `fused/gguf_q6_q4_pair.{hip,py}` | `linear_pair` (standard-Q6+Q4, Q4, Q4+planar-Q6) | Exact block-parallel rows1 pairs; gfx1151 qualifies Qwen3.8 recurrent K5120/N10240+N6144 and full-attention K/V K5120/N1024+N1024 while primitive projections remain fallbacks. |
| Dense Q6_K T16/qmicro | `quant/gguf_q6_k_t16_gemv.{hip,py}` | `linear`, `linear+argmax`, `linear+residual` | Exact dense Q6 decode/prefill/root families. gfx1100 planar row8 uses the exact DPP reduction (VGPR136→112, bpermute320→0), admitted on all 55 actual-operation rows and retained by a 1.634% complete-owner wall win; rows1-7 keep the generic reduction. gfx1151 rows>=512 uses 128-thread/four-wave shared-weight WMMA for standard K5120/N10240 QKV (2.96-3.55x) and planar K17408/N5120 FFN-down (1.42-1.50x); both use 24 KiB LDS / 248 VGPR. Rows<512, narrow V, root, shape misses, and peer backends retain exact one-wave/16x16 primitives. |
| Dense planar-Q6 integer MMQ | `quant/gguf_q4_k_q8_1_selected_prefill.{hip,py}` | `activation_quant`, `linear` | gfx1151 production-profile T2 composite for rows17-48 on sole-resident planar K17408/N5120 down and K5120/N1024 narrow-V: session-owned BF16-to-Q8_1 packing feeding the integer `mmq64x64` consumer; exact A owners remain registered for strict/profile fallback. |
| Dense Q4 q8_1-dp4a VDR screen | `quant/gguf_q4_k_q8_1_dp4a_vdr_gemv.{hip,py}` | `linear` (leaf screen, not dispatched) | nasone32 k-quant load-reuse port (efa4e8641): subblock-hoisted metadata with activation packs reused across 8 columns, plus an unamortized control with identical thread mapping and f32 order (bit-exact RED contract). 2026-09-09 four-arm leaf on both gfx1100 cards: -50..-76% vs the control inside the dp4a class, but 1.11-1.44x slower than the retained T16 rows=1 owners at every production shape (T16 sits at the DRAM floor), so rejected as a decode replacement; retained as evidence for future integer decode routes. |
| Dense Q4 int-MMQ prefill screen | `quant/gguf_q4_k_q8_1_mmq_prefill.{hip,py}` | `linear` (leaf screen, not dispatched) | Raw-Q4_K x DS4-Q8_1 bulk-prefill integer MMQ (PP8192-attribution candidate): staged-dp4a 32x32-tile and direct-global iu8-WMMA 32x16-tile consumers, each with bit-exact ctl/vdr siblings (block-header/subblock-metadata hoisting) and a DS4 CPU oracle. 2026-09-09 W7900 six-arm leaf on real Qwen3.8-27B Q4_K_M weights, rows 512/1024/4096: best integer totals (pack + consumes) reach only 0.205-0.542x the retained float T16 prefill owners (9/9 case-rows), load-reuse deltas within +-3%, pack negligible — rejected as a bulk-prefill replacement; retained as leaf evidence for future integer prefill routes. |

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
| `vibevoice/encoder.hip` | Speech encoders/connectors and Qwen2 attention/cache operations. Its `vv_attention_spans` is the strict span-attention parent. |
| `vibevoice/attention_spans_splitk.hip` | Split-K decode span attention: the production sibling of `vv_attention_spans`, one block per (KV head, KV split) with the query group folded in. |
| `vibevoice/decoder.hip` | Streaming speech-decoder convolutions and upsampling. |
| `vibevoice/diffusion.hip` | Diffusion-head normalization, modulation, and solver steps. |
| `evie/evie_ops.hip` | Vision patch embedding, normalization, attention, and merger operations. |
| `vision/gemma4_vision.{hip,py}` | FP32 Gemma 4 image RMSNorm, elementwise stages, X/Y rotary head layouts, softmax and spatial pooling; `vision_*` / `fp32` / `gemma4` registrations share gfx11 sources. The runtime composes these primitives with rocBLAS linear and attention contractions. |
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
| `vibevoice/attention_spans_splitk.hip` | Decode span attention split over the KV range; the `production` variant of `vv_attention_spans`. The host plan sizes the split count from the context and the device compute-unit count. |
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
| GGUF Q4 T16 fused expert slab | Registered selected-row projection primitives and split-parent WMMA | `moe_linear` / `gguf_q4_k_t16_v1` / `selected_dual_wmma_prefill_fused_bf16_bf16_out`: one expert-major gate/up slab, explicit full expert stride, transient LDS transpose with split-parent arithmetic | `quant/gguf_q4_k_t16_selected_prefill` |
| GGUF Q4/Q5/Q6 library routes | Exact raw/T16 projections | Dequantized FP16 rocBLAS and activation-quantized integer prefill | `quant/gguf_q6_k_f16_rocblas_prefill`, `quant/gguf_k_mmq_prefill`, `quant/gguf_q4_k_q8_1_selected_prefill` |
| GGUF Q8_0 | Raw/T16 GEMV and exact row-batched variants | Matrix-tiled and integer prefill; packed-integer verifier projections | `quant/gguf_k_gemv`, `quant/gguf_q8_0_t16_*`, `quant/gguf_q8_0_mmq_prefill`, `quant/gguf_q8_0_dp4a_gemv` |
| Qwen4Exp Q5_1 experts | Selected GEMV and exact grouped projections | `selected_grouped_wmma_prefill_compact_bf16_bf16_out` | `quant/qwen4_exp_q5_1` |
| Qwen4Exp branch mixing | `strict_unfused` | Fused gated branch mean | `fused/qwen4_exp_gr` |
| Qwen4Exp recurrent attention | `qwen4exp_sigmoid_strict_prefill` | `qwen4exp_sigmoid_peer_prefill` | `linear_attn/qwen4_exp_gdn` |
| Qwen4Exp sparse attention | `strict_rows_spans`, exact ordered decode variants | `production_wave32_h128_spans`, `production_rows_wave32_h128_spans` | `attention/qwen4_exp_qsa` |
| Laguna FP16 | Scalar/tiled projections and strict attention | Matrix-tiled projections, flash prefill, and fused attention/output gate | `linear/laguna_f16_projection`, `attention/laguna_kv_attention`, `attention/laguna_flash_attention_prefill` |
| Moonshine FP16 | Separate projection, activation, residual, and norm | Fused MLP/residual/norm; optional CUDA CUTLASS attention | `linear/moonshine_projection`, `fused/moonshine_*`, `norm/moonshine_layernorm`, CUDA `attention/moonshine_attention_cutlass` |
| Gemma 4 GGUF `gguf_q4_k_m` prefill attention | `gemma4_plain`: scalar strict attention with an exact decode twin | `gemma4_staged`: strict-order FP32 score, softmax and P*V stages with global score workspace and bounded LDS, the exact-chain variant for head_dim 256/512 and the rollback lever for the decode scope. `gemma4_flash`: key-sliced flash decode with per-slice partials and a combine. It serves one-token decode wherever the shape is admitted and the split policy asks for more than one slice (from about 1,024 keys), with no plan selection needed, and the production plan selects it explicitly for the untrimmed global step (head_dim 512, GQA ratio 8) past the resident LDS bound, where the launcher's flash request is honoured ahead of that bound and the native path would otherwise use its owned-global-logits class kernel. Below the bound the resident logits row still fits and the native path stays available. The slice count is the larger of the occupancy target and the smallest count whose per-block LDS logits fit. `gemma4_flash_long` is the same kernel with one slice per 128 keys past 8192 keys (at most 1024 slices); the gfx1100 production plan selects it. Staged score CTAs group up to eight query heads per KV head with exact-row variants, and the singleton PV is pipelined. `gemma4_wmma_flash` (sliding) and `gemma4_wmma_flash_full` (head_dim 512) are the production multi-row prefill defaults under the section 2.11 envelope (EXECUTION-PROFILES); strict `gemma4_plain` remains the registered fallback | `gemma4/gemma4_attention`, `gemma4/gemma4_attention_staged`, `gemma4/gemma4_attention_prefill_wmma`, `gemma4/gemma4_attention_prefill_wmma_full` |
| VibeVoice BF16 | `strict` primitives and incremental prefill | Fused depthwise convolution, matrix-tiled frontend, and library prefill | `vibevoice/encoder`, `vibevoice/registered.py` |
| TimesFM | FP32 attention | FP16 matrix-tiled flash attention | `timesfm/timesfm` |
| YuE2 autoregressive BF16 | Row-by-row dense GEMV prefill; `vv_attention_spans` decode attention | FP16-converted hipBLASLt batched prefill; paired two-row projection row tiles; `vv_attention_spans` `production` variant, split over the KV range | Shared `linear/dense_gemv`, `vibevoice/encoder`, `vibevoice/attention_spans_splitk`, `hipengine/runtime/yue2_ar.py` |
| YuE2 non-autoregressive attention | `nar_attention_f32` | `nar_attention_wmma` with changed arithmetic | `yue2/nar`, `yue2/nar_wmma` |

These rows identify related implementations, not blanket profile assignments:
exactness, supported inputs, and selection scope belong to each variant.

### CUDA shared support

`smoke/smoke_add.{cu,py}` validates the CUDA build/runtime path. There is no CUDA PARO or general GGUF/Laguna catalog yet; adding one requires peer `.cu` implementations or an explicit architecture-qualified library integration, not a backend branch in engine code.

## Device translation-unit inventory

This is the mechanical inventory of every in-tree HIP/CUDA device translation unit. The semantic catalogs above are the primary organization; this tree is the completeness check. A translation unit may implement many registry keys and template instantiations.

```text
hipengine/kernels/hip_gfx1100/
├── attention/
│   ├── dms_compact.hip
│   ├── laguna_flash_attention_prefill.hip
│   ├── laguna_kv_attention.hip
│   ├── maple_attention.hip
│   ├── moonshine_attention.hip
│   ├── paged_attn_decode.hip
│   ├── paged_kv_write.hip
│   ├── qwen4_exp_qsa.hip
│   └── qwen4_exp_qsa_wmma.hip
├── convert/
│   ├── cast.hip
│   └── gather.hip
├── dispatch/
│   └── moe_c1_dispatch.hip
├── fused/
│   ├── gguf_ops.hip
│   ├── laguna_attention.hip
│   ├── moonshine_glue.hip
│   ├── moonshine_mlp.hip
│   ├── paro_combine.hip
│   ├── paro_silu.hip
│   ├── qwen4_exp_gr.hip
│   └── qwen4_exp_ple.hip
├── linear/
│   ├── dense_f16_gemm.hip
│   ├── dense_gemv.hip
│   ├── laguna_f16_projection.hip
│   ├── lm_head.hip
│   ├── moonshine_projection.hip
│   └── moonshine_w8a16.hip
├── linear_attn/
│   ├── conv.hip
│   ├── gdn.hip
│   └── qwen4_exp_gdn.hip
├── moe/
│   ├── group_scatter.hip
│   ├── laguna_router.hip
│   ├── maple_moe.hip
│   └── router.hip
├── norm/
│   ├── moonshine_layernorm.hip
│   └── rmsnorm.hip
├── quant/
│   ├── gguf_expert_pack8_gemv.hip
│   ├── gguf_iq2_xs_mmq_prefill.hip
│   ├── gguf_iq_gemv.hip
│   ├── qwen4_exp_q5_1.hip
│   ├── qwen4_exp_routed_f16.hip
│   ├── gguf_iq_selected_prefill.hip
│   ├── gguf_iq_source_mmq_prefill.hip
│   ├── gguf_k_gemv.hip
│   ├── gguf_k_mmq_prefill.hip
│   ├── gguf_k_selected_pack8_gemv.hip
│   ├── gguf_k_selected_prefill.hip
│   ├── gguf_k_t16_selected_prefill.hip
│   ├── gguf_q3_k_gemv.hip
│   ├── gguf_q4_k_gemv.hip
│   ├── gguf_q4_k_moe_ffn_fused.hip
│   ├── gguf_q4_k_pack8_gemv.hip
│   ├── gguf_q4_k_prefill.hip
│   ├── gguf_q4_k_q8_1_mmq_prefill.hip
│   ├── gguf_q4_k_q8_1_selected_prefill.hip
│   ├── gguf_q4_k_selected_pack8_gemv.hip
│   ├── gguf_q4_k_selected_prefill.hip
│   ├── gguf_q4_k_t16_selected_prefill.hip
│   ├── gguf_q5_k_f32_rocblas_prefill.hip
│   ├── gguf_q6_k_embedding.hip
│   ├── gguf_q6_k_f16_rocblas_prefill.hip
│   ├── gguf_q6_k_pack8_gemv.hip
│   ├── gguf_q6_k_t16_gemv.hip
│   ├── gguf_q8_0_dense_wide.hip
│   ├── gguf_q8_0_dp4a_gemv.hip
│   ├── gguf_q8_0_mmq_prefill.hip
│   ├── gguf_q8_0_pack8_gemv.hip
│   ├── gguf_q8_0_prefill.hip
│   ├── gguf_q8_0_raw_to_t16.hip
│   ├── gguf_q8_0_t16_gemv.hip
│   ├── gguf_q8_0_t16_prefill.hip
│   ├── gguf_t16_selected_gemv.hip
│   ├── gguf_x8_selected_gemv.hip
│   ├── maple_ternary.hip
│   ├── paro_awq_gemv.hip
│   ├── paro_marlin_k.hip
│   ├── paro_moe_ffn_fused.hip
│   └── w8a16_linear.hip
├── rotary/
│   ├── paro_rotate.hip
│   └── qwen35_rotary.hip
├── runtime/
│   ├── laguna_launch_batch.hip
│   └── state.hip
├── sampling/
│   └── sampler.hip
├── smoke/
│   └── smoke_add.hip
├── speculative/
│   ├── dflash2.hip
│   ├── dflash_accept.hip
│   ├── dflash_commit.hip
│   ├── dflash_drafter.hip
│   ├── mtp.hip
│   └── mtp_nextn.hip
└── wmma/
    └── paro_awq_wmma.hip

hipengine/kernels/cuda_sm120a/
├── attention/
│   ├── maple_attention.cu
│   ├── moonshine_attention.cu
│   └── moonshine_attention_cutlass.cu
├── encoder/
│   ├── moonshine_encoder.cu
│   ├── moonshine_encoder_cudnn.cu
│   └── moonshine_encoder_lt.cu
├── fused/
│   ├── moonshine_glue.cu
│   └── moonshine_mlp.cu
├── linear/
│   ├── lm_head.cu
│   ├── maple_lm_head.cu
│   └── moonshine_projection.cu
├── moe/
│   ├── group_scatter.cu
│   └── maple_moe.cu
├── norm/
│   ├── maple_rmsnorm.cu
│   └── moonshine_layernorm.cu
├── quant/
│   └── maple_ternary.cu
└── smoke/
    └── smoke_add.cu
```

`hipengine/kernels/cpu_reference/` is cataloged separately above because it contains Python/NumPy oracles rather than device translation units. `hipengine/kernels/cuda_sm86/` is an empty backend scaffold.

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

## Qwen4Exp router ordered DPP sibling

`moe/router.{hip,py}` registers
`router_logits/f32/f32_hidden_token_tile4_dpp_exact`, aliased to gfx1151
alongside the unchanged `f32_hidden_token_tile4_dense_exact` strict parent.
The candidate keeps tile4's per-thread FMA assignment and 256-way reduction
tree: one LDS publication, register-folded inter-wave pairs, ordered DPP
wave finish. Raw-pointer signatures, output strides and selection ABI are
unchanged. The gfx1151 `gguf_ud_q4_k_xl` production profile selects the
F16 WMMA GEMM producer; DPP serves small multirow widths and named rollback.
Its historical canonical D128 full-logit/state/KV gate,
576 natural-router captures and row-isolation/ownership fixtures passed. The
tracked-clean same-residency twelve-case A/B improves weighted prefill
+0.553%/+0.486%/+0.589% at p512/p1024/p4096, all cases positive. Strict
and Q4_K_M keep tile4; no gfx1100 execution or decode-speed claim.
The tile4 parent remains registered as strict fallback and temporary rollback.
Evidence: `benchmarks/results/2026-10-05-qwen4exp-router-dpp/README.md`.

## Qwen4Exp routed-expert F16 GEMMs

`quant/qwen4_exp_routed_f16.{hip,py}` registers two routes, aliased to
gfx1151, that read the **raw** GGUF expert tensors (no repacked copy):

- `moe_linear/{gguf_q4_k,gguf_q5_k}/selected_routed_f16_dual_silu_wmma_prefill_bf16_bf16_out`
  (Q5_K also under `linear`): gate and up in one launch with the SwiGLU
  epilogue, macro tiles of 64 or 128 grouped rows.
- `moe_linear/{gguf_q5_1,gguf_q8_0,gguf_q4_k,gguf_q5_k}/selected_routed_f16_wmma_prefill_bf16_bf16_out`
  (Q8_0 also under `linear`): one projection, macro tiles of 16, 48 or 64
  grouped rows (16 or 48 for the K-quants).

Each block stages two 32-element K blocks of packed codes for 128 output
rows in LDS, decodes them to F16 in registers (`1024 + q` through a byte
permute into the half mantissa, then one packed FMA with the block scale and
bias), and runs F16 WMMA with F32 accumulation over the whole K extent
against up to 128 grouped activation rows. A weight tile is therefore decoded
once per macro tile rather than once per 16 rows, and there is no integer
risk tracking or repair pass. Outputs are BF16 rows in the grouped
(expert-sorted) order. The paired gate/up kernel reads token-major BF16
activations through the grouped lane map (`row_lanes[r] // top_k`), so the
runner launches the metadata-only `qwen35_moe_group_scatter` and no grouped
copy of the activations is written; without a lane map it reads grouped rows
as `qwen35_moe_group_scatter_gather_lowp` lays them out. Activations are
converted to F16 while staged. `qwen4_exp_routed_f16_tile_map` expands
the int64 grouped prefix into `(expert | tile << 16)` int32 entries on the
device and publishes the live count, so the GEMM grid is sized to a static
bound with no device read.

The kernel body is ported from gufo v0.12.0 (`88090afb`, MIT),
`src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp`
`RoutedF16GEMMKernel`; the file header records the hipEngine changes.

This is a reduced-precision operand route: it changes arithmetic relative to
the exact grouped owners. The `production` plan for `qwen4_exp_gguf` /
`hip_gfx1151` / `gguf_ud_q4_k_xl` selects it
(`HIPENGINE_QWEN4_EXP_ROUTED_F16`, published by that plan's binder) as part
of the F16 route composition gated in
`benchmarks/results/2026-10-14-qwen4exp-fast-composition-gate-2/README.md`.
The strict fallbacks are the registered exact grouped chains. Correctness test:
`tests/test_gpu_qwen4exp_routed_f16.py` (CPU-reference outer floor plus a
relative-error bound, all four quants, every tile width, ragged buckets and
non-tile-multiple output widths). Evidence:
`benchmarks/results/2026-10-13-qwen4exp-fast-routed-f16/README.md`.

## Qwen4Exp tiled QSA block scoring

`attention/qwen4_exp_qsa.hip` `qsa_score_h4_d128_tiled_f32_kernel` scores a
16-query x 16-block tile per workgroup from operands staged in LDS, one
thread per (query, block) pair, for four index heads of 128 dimensions and
at least 16 queries. `qwen4_exp_qsa_score_f32` selects it inside the launcher;
other geometries, and decode's single query, keep
`qsa_score_f32_kernel` (one 32-thread workgroup per pair, four threads doing
work). Each head's dot product and the four-term ReLU sum keep the parent's
order, so the scores are bit-identical; `parent=True` on the wrapper launches
the parent on any geometry. Correctness test:
`tests/test_gpu_qwen4exp_qsa_score_tiled.py` (bitwise parity over ragged
tiles and ReLU/zero edges, plus the CPU reference). It runs in every profile.

## Qwen4Exp fused sigmoid gated mean without a gate store

`fused/qwen4_exp_gr.hip` `gated_mean_sigmoid_f32_kernel<STORE_GATE>`: the
`false` instantiation (`qwen4_exp_gated_mean_sigmoid_f32(store_gate=False)`)
applies the sigmoid inside the gated-mean pass and leaves the logits buffer
untouched. The wide hyper-connection up route uses it in place of
`sigmoid_f32` followed by `gated_mean_f32`; no caller reads the gate after a
hyper-connection read, and on that route `scratch.gate` holds logits. The
mixed output is bit-identical to the pair. Correctness test:
`tests/test_gpu_qwen4exp_gr_sigmoid_mean_no_gate.py`.

## Qwen4Exp hyper-connection read kernels (prefill)

A read of more than 256 rows in the `gguf_ud_q4_k_xl` production plan is
three launches plus the down GEMM, none of which writes the FP32 normalized
plane or the gate logits:

- `fused/qwen4_exp_gr.hip` `grouped_rmsnorm_bf16_f16_scale_kernel`: the
  grouped RMS norm, storing the F16 plane the down projection reads and
  `inv_rms[rows, branches]`. Same summation tree as
  `grouped_rmsnorm_bf16_f32_kernel`, so both are the bits that kernel stores.
- `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_dense_wide.hip` `q8_0_gr_up_mean_kernel`: the K320/N10240
  up projection with the sigmoid gated mean in its epilogue. A block's 128
  weight rows are 32 hidden columns of each of the four branches, so a wave
  holds all four logits of every output it owns. The normalized value is
  recomputed as `bf16(residual) * inv_rms * weight`. It can also write
  `half(mixed)`. Bit-identical to the wide256 F16 projection followed by
  `gated_mean_sigmoid_f32_kernel`.
- `fused/qwen4_exp_gr.hip` `gr_inject_bf16_kernel`: the four inject logits
  from the BF16 residual, one block per row, in the scalar FP32 GEMV's
  summation order.

Tests: `tests/test_gpu_qwen4exp_gr_f16_direct.py`,
`tests/test_gpu_qwen4exp_gr_up_mean_fused.py`,
`tests/test_gpu_qwen4exp_gr_read_residual.py`. No external kernel body is
ported. Strict fallback: the unfused chain (`grouped_rmsnorm_bf16_f32`,
`dense_wide*`, `gated_mean_sigmoid`, scalar GEMV), restored by
`HIPENGINE_QWEN4_EXP_GR_F16_DIRECT=0`.

## F16 activation image registry

`hipengine/kernels/hip_gfx1100/linear/f16_mirror.py` holds one context-local declaration: "this F16 buffer
is `half()` of that FP32 buffer, for exactly this extent". The fused
hyper-connection up kernel and the row-split GDN norm/gate epilogue write
such an image; the F16 wide dispatch (`hipengine/runtime/gguf_linear.py`) and the
small-matrix F16 GEMM (`hipengine/kernels/hip_gfx1100/linear/dense_f16_gemm.py`) look it up by pointer and
extent and read it instead of converting. Producers' owners invalidate the
declaration before either buffer can be rewritten. Tests:
`tests/test_unit_f16_mirror.py`, `tests/test_gpu_dense_f16_gemm.py`.

## Small-matrix F16 GEMM staged kernel

`hipengine/kernels/hip_gfx1100/linear/dense_f16_gemm.hip` `dense_f16_gemm_staged_kernel` is the launch
default for uncompressed FP32/BF16 weight matrices of at least 16 columns at
64 rows or more (router logits, indexer projections, the 48-column GDN gate
projections): four-element vector staging converted in registers, the next K
tile fetched while the current one multiplies, coalesced row stores,
128x128 tile. Bit-identical to the scalar kernel in the same file, which
serves K extents that are not a multiple of 64. F16-activation entry points
skip the conversion when an image is declared.

## Qwen4Exp row-split GDN prefill recurrence

`linear_attn/qwen4_exp_gdn.hip`
`qwen4_exp_gdn_prefill_rowsplit_f32_kernel<4, 4>`, registered for
`hip_gfx1151` as `gdn_recurrence_norm_gate / f32_state /
qwen4exp_gdn_rowsplit_prefill` and selected by the `gguf_ud_q4_k_xl`
production binder. Ported from gufo `GdnRowSplitKernel` (commit 9cad139,
MIT); the file header records the hipEngine changes (state layout, in-block
norms, decay and beta). Four lanes own one value column of the 128x128
state, blocks are 128 threads, tokens are staged four at a time through LDS
with three windows of loads in flight. It applies the key and query scales
to the two dot products and sums in a different FP32 order than the
column-warp owners (relative L2 1.5e-7 over 4096 tokens). Its norm/gate
epilogue can publish `half(core)` for the output projection. Test:
`tests/test_gpu_qwen4exp_gdn_rowsplit_prefill.py`. Strict fallback:
`qwen4exp_sigmoid_strict_prefill`; rollback
`HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT=qwen4exp_gdn_tiled16_multi_prefill`.

## Qwen4Exp indexed GDN decode primitive

`linear_attn/qwen4_exp_gdn.{hip,py}` provides
`qwen4_exp_gdn_decode_rowsplit_indexed_f32_kernel`, registered as
`gdn_recurrence_norm_gate / f32_state / qwen4exp_gdn_indexed_decode`.
Each row advances one independent recurrent-state slot, rather than one token
in a shared causal chain. Inputs and outputs have explicit row strides; the
state slab is `[state_slots, value_heads, 128, 128]` FP32. Slot indices must be
unique and in range. The raw-pointer caller remains responsible for allocation
sizes and lifetime. This variant is available through the gfx1100 kernel package
and its gfx1151 registration; source location is not device qualification.

The single-row `qwen4exp_gdn_rowsplit_decode` and
`qwen4exp_sigmoid_strict` variants remain available as fallbacks/oracles.
Focused tests are `tests/test_unit_qwen4exp_gdn_indexed_decode.py` and
`tests/test_gpu_qwen4exp_gdn_indexed_decode.py`. Evidence and hardware scope:
[primitive worklog](../worklog/entries/20261018T000834.117311Z-lhl-qwen4exp-gdn-indexed-decode-380030.md).
This primitive is not selected by serving. Conv/PLE history, QSA index/KV
ownership, runner integration and C2/C4 serving qualification are separate
requirements; no model-level numerical or throughput claim follows from it.

## Qwen4Exp indexed F32 convolution decode primitive

`linear_attn/conv.{hip,py}` registers
`linear_attn_conv_decode / f32_state / qwen4exp_conv_indexed_decode` in the
gfx1100 package and its gfx1151 mirror. One packed input row advances one unique
slot in a `[state_slots, channels, kernel_size]` FP32 history slab. Duplicate
or out-of-range slots reject the entire batch before any state/output write;
host index validation provides the diagnostic. Raw-pointer callers own buffer
sizes, lifetimes and validation before using output. The single-row
`qwen35_linear_attn_conv_decode_f32` remains the arithmetic oracle/fallback.

Tests: `tests/test_unit_qwen4exp_conv_indexed_decode.py` and
`tests/test_gpu_qwen4exp_conv_indexed_decode.py`. The
[worklog](../worklog/entries/20261018T002058.635279Z-lhl-qwen4exp-conv-indexed-decode-6eb613.md)
records zbook/gfx1151 primitive correctness and dispatch evidence, not gfx1100
qualification or serving performance. Serving integration remains open.

## Qwen4Exp indexed PLE decode primitive

`fused/qwen4_exp_ple.{hip,py}` registers
`ple_dilated_depthwise_conv / f32 / qwen4exp_ple_indexed_decode` in the gfx1100
package and its gfx1151 mirror. Each packed row is one token for a distinct
request, selecting a unique history slot in a
`[state_slots, (kernel_size - 1) * dilation, channels]` FP32 slab. Duplicate or
out-of-range slot vectors cause whole-batch no-write rejection. Callers must
validate indices on the host before relying on output and own buffer sizes and
lifetimes. The `strict` single-request variant remains the fallback.

Tests: `tests/test_unit_qwen4exp_ple_indexed_decode.py` and
`tests/test_gpu_qwen4exp_ple_indexed_decode.py`. This is a primitive, not a serving
batch path; QSA/KV ownership and runner integration remain separate requirements.

## Workgroup size and CU mode on gfx1151

Two measured facts that decide tile and build choices on this backend:

- A kernel whose blocks run for the whole launch (the GDN recurrence) is
  sensitive to workgroup size: the same row-split code measures 5.58 ms in
  128-thread blocks and 7.93 ms in 256-thread blocks at 4096 tokens.
- `-mcumode` (the `decode` build profile) is not uniformly better or worse.
  Dense wide Q8_0 launches prefer the build without it up to 1024 rows and
  for the long-K 128x128 tile, and the CU-mode build otherwise; the
  routed-expert kernels prefer no CU mode; the hyper-connection elementwise
  and GDN conv kernels show no difference. Measure in one process with both
  builds loaded: cross-process runs differ by more than these effects.

## Qwen4Exp GDN eight-wave norm/gate

`linear_attn/qwen4_exp_gdn.hip` gains
`qwen4_exp_gdn_core_norm_gate_waves_f32_kernel`, the per-head RMS norm,
weight and sigmoid output gate that follows the column-warp and tiled GDN
recurrences. The parent `qwen4_exp_gdn_core_norm_gate_f32_kernel` launches one
32-thread workgroup per (token, value head): 196,608 workgroups for a
4096-token chunk at 48 heads, about 10 ms per layer, nearly all of it launch
overhead. The eight-wave kernel runs the same per-lane partial sums, the same
wave reduction and the same epilogue with one (token, head) pair per wave,
eight pairs per 256-thread workgroup, and reads the sum from lane 0 as the
parent does. The result is **bit-identical** (T0), so all four internal
launch sites use it in every profile. The parent stays reachable through
`qwen4_exp_gdn_core_norm_gate_f32` as the strict reference. Parity test:
`tests/test_gpu_qwen4exp_gdn_norm_gate_waves.py`. Evidence:
`benchmarks/results/2026-10-14-qwen4exp-gdn-norm-gate-waves/README.md`.

## Qwen4Exp WMMA QSA prefill attention

`attention/qwen4_exp_qsa_wmma.{hip,py}` registers
`qsa_sparse_attention/bf16_kv/wmma_f16_masked_prefill_f32` and the same
function under `paged_attn_decode/w4_paro`, aliased to gfx1151, plus two
support launches:

- `WmmaCausalAttentionKernel`: 24 query heads of 256 dims over two KV heads.
  Queries (scaled by 1/16), keys and values are F16 fragments on the matrix
  units with F32 score and output accumulation and an online softmax; the
  next 16-key tile's K/V is prefetched into registers behind the current
  tile's score, softmax and PV phases. The dense causal window packs 16
  queries x 2 heads per workgroup. The masked sparse window packs 4 queries
  x 12 heads and gathers key tiles from the union of those queries' selected
  four-token blocks, so a sweep costs the union of the selections rather
  than the context.
- `PagedKvToF16Kernel` (hipEngine): gathers the paged BF16 K/V caches for
  positions `[first, context)` into contiguous F16 rows. Every BF16 value in
  F16 range converts exactly.
- `SelectionMaskKernel` (hipEngine): folds each query's selected-position
  list into a bitmask over four-token blocks. The attention kernel treats a
  query's own incomplete tail block as always visible, matching the tail
  positions the selection lists carry.

The attention kernel is ported unchanged from gufo v0.12.0 (`88090afb`, MIT),
`src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp`.

The gather wrapper takes a `KVLiveSpans` and accepts uniform BF16 spans
only, like the scalar sparse-attention wrappers beside it; it reads the page
table the spans carry over one request's contiguous positions. It is not a
route for DMS/H2O/SnapKV policies, which need a variable-span gather. Decode
attention is unchanged.

Selected by the `production` plan for `qwen4_exp_gguf` /
`hip_gfx1151` / `gguf_ud_q4_k_xl` (`HIPENGINE_QWEN4_EXP_QSA_WMMA`, published
by that plan's binder); strict fallbacks are the scalar FP32 span kernels.
Correctness test: `tests/test_gpu_qwen4exp_qsa_wmma.py` (float64 reference,
dense windows with and without prior context, masked sparse windows with
shuffled selection lists, a permuted block table). Evidence:
`benchmarks/results/2026-10-14-qwen4exp-fast-qsa-wmma/README.md`.

## Small-matrix F16 WMMA GEMM

`linear/dense_f16_gemm.{hip,py}` registers
`dense_gemv/{f32,bf16}/f16_wmma_f32_hidden_f32_out` and
`router_logits/f32/f32_hidden_f16_wmma_gemm`, aliased to gfx1151. FP32
activations and FP32 or BF16 weights are rounded to F16 while staged in LDS
(128 output columns x 64 rows x BK 32 per workgroup), multiplied with F16
WMMA and F32 accumulation, and written as unrounded FP32; any output width is
accepted and waves whose columns are past the matrix skip the matrix work.
The staging and fragment layout follow `dense_prefill_wmma_out_kernel` in
`dense_gemv.hip`. It is a reduced-precision operand route selected by
the `production` plan for `qwen4_exp_gguf` / `hip_gfx1151` /
`gguf_ud_q4_k_xl`: `HIPENGINE_QWEN4_EXP_DENSE_F16_GEMM` moves FP32
projections with at least 64 rows and 96 output columns (the indexer
projections), and the plan's `router_logits` row selects the router
variant. Strict fallbacks are the scalar FP32 GEMV and the tile4 router.
Correctness test: `tests/test_gpu_dense_f16_gemm.py` (against the FP32 product
of the F16-rounded operands at 2e-5 of the output scale, and against the
unrounded product at 2e-3 relative). Evidence:
`benchmarks/results/2026-10-13-qwen4exp-fast-indexer-f16-gemm/README.md`.

## Qwen4Exp GR-up ordered DPP sibling

`quant/gguf_k_gemv.{hip,py}` registers
`linear+gr_gated_mean/gguf_q8_0/coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact`
for gfx1100 and its gfx1151 peer. It keeps the parent's FMA, four-wave
publication, sigmoid and branch-mean order, changing only the published
lane-zero reduction to `permlanex16` plus `row_shl8/4/2/1` data-parallel
primitive (DPP) operations. The wave-scale fused parent, unscaled fused
parent and projection/sigmoid/mean primitive chain remain available.

During admission `HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION=1` selects the
sibling only within the existing rows>256/four-branch route. Raw-bit tests
cover tails through 4096 rows; real Q8_0 banks at rows512/1024/4096 improve
1.54-1.57x on physical gfx1151 host
`55ea6c509d0b49eea8de7094a1023668`. The compiler reduces VGPR72->57 and
static instructions311->243. These are operation-level results, not an
engine-rate claim; gfx1100 execution is unverified.
Evidence: `benchmarks/results/2026-10-05-qwen4exp-gr-dpp-reduction/`.

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

H256 QSA batch decode selects `strict_h256_head_pair_rows_spans`
for page256 with an even GQA ratio, with `strict_rows_spans` as fallback.
Two adjacent query heads share each K/V
load but retain independent score/online-softmax state. Explicit
`fma(acc,old_scale,score_scale*value)` preserves parent's contraction;
initial alternative failed strict equality.23 tests pass including
poisoned unused KV and parent CPU-reference chain. Synthetic2051-selected
attention512/1024 rows2.544x/2.610x,30 pairs exact.
VGPR40->72,no scratch/LDS,half head blocks.
Evidence: `2026-09-07-framework-qwen4exp-qsa-head-pair.json`.
Default-off private-runner prefill admission via `HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR=1`
passes eight full-logit/state/KV cases at chunk1024,all four p4096
categories included.24 calls per sparse p4096 arm,zero decode,dense
short paths unchanged. Hq24/Hkv2 page256 parent only;both binders0.
Admission: `2026-09-07-framework-qwen4exp-qsa-head-pair-state.json`.
Staged full-model qualification preserves72 trajectories and improves
p4096 PP3.560%,but TG drops6.526%;two long-request averages regress.
The private-runner prefill candidate stays default-off pending causal decode/phase investigation.
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

GDN registers `qwen4exp_sigmoid_wave_norm_narrow_prefill` as the gfx1151
UD production serial owner at Hk16/Hv48/D128, 2–4096 rows. It preserves FP32
recurrence and reduction association, with int32 element indexing checked in
Python and the int64 host ABI. Wide-index remains outside the bound and for
explicit plan rollback; strict primitives stay registered. Strict and Q4_K_M
production do not select the narrow leaf. The UD manifests name disjoint
bounded/outside serial scopes rather than inactive peer/tiled layer scopes.
On host `55ea6c509d0b49eea8de7094a1023668`, synthetic complete-operation
speedup is 1.0443x/1.0419x/1.0424x at 512/1024/4096; scratch 36→28B,
spills 8→6, VGPR 256 unchanged. Natural-model owner timing improves
1.027–1.044x; 20 full-state/KV cases and 72 engine trajectories pass exactly.
Engine weighted prefill gains +0.149%/+0.303%/+0.227%; one case -0.008%
and mixed request walls remain explicit. Native binder smoke verifies
selected counts and state/KV parity. Evidence:
`benchmarks/results/2026-10-06-qwen4exp-gdn-narrow-index/README.md`.

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
For private-runner prefill, Qwen4Exp production selects its page256 sibling
for sparse rows; strict retains the original rows owner.
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

Journey Q8 selected-down residual-weight experiment is rejected and removed:
the full594-row gate improves max KL to0.019447 but fails top1 at587/594
(98.82%) and prefill-last mean KL0.001256. CPU weight error improvement
does not certify the model. Original WMMA/strict fallback bodies are restored;
the corrected GPU oracle and fresh-process parent registration remain.

Journey multi-column GDN remains experimental:
`qwen4exp_gdn_tiled16_multi_prefill` uses eight columns per warp, eight
warps and token-tile16 with DPP. The single-column DPP default remains the
fallback. Parent parity is not exact; CPU-reference/numerical smoke,
carried-state and deterministic repeats pass. Full production envelope
qualification is required before any default selection.

The multi-column full594-row gate has now run: overall maxKL0.048222 and
590/594 top1, but Japanese/prefill-last scopes fail. It remains unpromoted
with a concrete numerical blocker; the incumbent also fails related scopes.

Journey GDN DPP experiment: gfx1151 registers
`gdn_recurrence_norm_gate/f32_state/qwen4exp_gdn_tiled16_dpp_prefill`.
It compiles `linear_attn/qwen4_exp_gdn.hip` in the separately hashed
`qwen4_exp_gdn_dpp` family with `HIPENGINE_QWEN4_GDN_DPP=1`; the ordinary
tile16 and serial strict keys remain unchanged fallbacks. Source:
pwilkin/llama.cpp `964c6f2`, `ggml/src/ggml-cuda/gated_delta_net.cu`,
inspected at `f5daaa3c`. The named gfx1151 UD-Q4_K_XL production profile now
selects this key within its existing tiled16 suffix; other quant/strict scopes
retain their prior owners.
Complete output/carried-state parent parity, CPU-reference outer checks
and cached profiler smoke pass; full72-sample whole-model A/B is exact and
improves weighted PP0.125%/0.265%/0.262% at512/1K/4K. Evidence:
`benchmarks/results/2026-09-14-journey-gdn-dpp/artifact.json`.

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
