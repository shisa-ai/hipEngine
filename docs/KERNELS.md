# hipEngine Kernel Catalog and Port Playbook

This document is the durable catalog of kernel families implemented in hipEngine and the stable mechanics for adding or porting one. It is intentionally **not** an experiment log.

Keep here:

- what kernel and oracle families exist;
- where their source and Python registrations live;
- which backends and model/format paths use them;
- which fused/composite families exist and what their unfused fallback is;
- stable ABI, build, profiling, and port rules.

Do not put here:

- benchmark results, tuning chronology, candidate ladders, campaign codes, or "next target" notes;
- rejected experiments or transient selectors;
- running status reports.

Those belong in immutable `worklog/entries/`, compact `benchmarks/results/` artifacts, `benchmarks/CHANGELOG.md`, and focused design/status docs. Current defaults are code: backend package capabilities and registry registrations, not prose copied into this catalog.

Related documents:

- [`PLAN.md`](PLAN.md) — architecture and roadmap.
- [`TESTING.md`](TESTING.md) — RED/GREEN workflow, fixtures, and correctness gates.
- [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) — strict/production/
  batch-invariant arithmetic, ownership, fallback, and manifest contracts.
- [`BENCHMARK.md`](BENCHMARK.md) — benchmark protocols and evidence policy.
- [`REFACTOR.md`](REFACTOR.md) — temporary flags and fallback-removal ledger.
- [`source_lineage.json`](source_lineage.json) — external source baselines.
- Model/path notes: [`GGUF.md`](GGUF.md), [`MAPLE.md`](MAPLE.md), [`MOONSHINE.md`](MOONSHINE.md), [`DFLASH.md`](DFLASH.md), and [`MTP.md`](MTP.md).

## How to read and maintain the catalog

hipEngine's registry key is:

```text
(backend, layer, quant, variant)
```

The catalog is organized in the same direction a user or maintainer selects a path:

1. **backend** — CPU oracle, HIP gfx1100/gfx1151, or CUDA sm_120a;
2. **model/format path** — shared Qwen/PARO, GGUF/Laguna/Qwen, Maple, Moonshine, or speculative support;
3. **functional family** — conversion, norm/rotary, projection, attention/KV, linear attention, MoE, sampling/state;
4. **variant** — exact registered keys remain authoritative in source.

A row catalogs a source/wrapper family, not every C++ template instantiation. Many families intentionally register dozens or hundreds of shape/layout variants. Enumerating those variants by hand here would duplicate the registry and drift quickly.

To inspect exact live keys:

```python
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import registered_keys

load_backend_kernel_package("hip_gfx1100")
for key in registered_keys():
    if key.backend == "hip_gfx1100":
        print(key.display())
```

Catalog maintenance rules:

- Add or remove the relevant family row in the same commit as a landed/removed kernel family.
- Name the `.hip`/`.cu` and `.py` owner; use registry layer/quant names rather than campaign labels.
- Put only stable constraints in Notes (ABI, storage layout, fallback, backend relationship).
- Link detailed performance/correctness evidence from worklogs or benchmark artifacts; do not reproduce it here.
- A rejected candidate that leaves no registered kernel does not get a catalog row.
- A retained diagnostic primitive may be marked **diagnostic**, but its experiment narrative stays elsewhere.

## Backend matrix

| Backend | Native target | Source ownership | Cataloged paths |
| --- | --- | --- | --- |
| `cpu_reference` | NumPy/host | `hipengine/kernels/cpu_reference/` | Shared primitive oracles, Qwen/PARO/GGUF, Laguna, Maple, Moonshine, Moonshine encoder |
| `hip_gfx1100` | RDNA3 `gfx1100` | `hipengine/kernels/hip_gfx1100/` | Qwen/PARO, GGUF/Qwen/Laguna, Maple, Moonshine, MTP/DFlash, shared state/sampling |
| `hip_gfx1151` | RDNA3.5 `gfx1151` | Shared gfx11 device sources plus peer registrations/capabilities in `hip_gfx1151/__init__.py` | Independently admitted subsets of the gfx11 families above |
| `cuda_sm120a` | CUDA `sm_120a` | `hipengine/kernels/cuda_sm120a/` | Maple and Moonshine peer implementations plus smoke/shared helpers |
| `cuda_sm86` | CUDA `sm_86` | package scaffold only | No implemented device family yet |

### gfx1151 source sharing is not backend equivalence

`hip_gfx1151` compiles shared gfx11 `.hip` bodies as native `gfx1151` code objects and registers a peer backend key. `hipengine/kernels/hip_gfx1151/__init__.py` controls aliases, exclusions, thresholds, and architecture-specific defaults. A gfx1100 variant is not a gfx1151 default merely because the source compiles there; each promotion needs its own correctness and performance gate.

### CUDA is a peer backend

`cuda_sm120a` has independent `.cu` bodies and Python wrappers. It does not alias HIP launch wrappers. CUDA-specific CUTLASS, cuDNN, cuBLASLt, graph, or thread-geometry choices are not selection evidence for either gfx11 backend.

## CPU-reference oracle catalog

CPU oracles favor clarity and deterministic boundaries over speed. They are the required comparison path for net-new kernels.

| Model/path | Source | Oracle families |
| --- | --- | --- |
| Shared primitives and Qwen/PARO/GGUF | `cpu_reference/ops.py` | embedding, linear/QKV/O/lm-head, RMSNorm, rotate, full/paged attention, KV quant/dequant/write, GDN and Conv prefill, GGUF Q4/Q5/Q6/Q8 dequant/GEMV, PARO AWQ pack8, MoE selected/tail, MTP/NextN helpers |
| Laguna | `cpu_reference/laguna.py` | YaRN/plain RoPE, head RMSNorm, global/SWA attention, dense and sparse FFN/MoE, routing, DFlash layer/model, target-hidden projection |
| DFlash2 | `cpu_reference/dflash2.py` | grouped dynamic conv (prepare/finish), top-16 bilinear candidate selector + greedy walk, q/k-norm sliding-window attention, Qwen3 block-repeat RoPE | DFlash2DraftModel exact-math oracles; fixtures generated from the z-lab/dflash torch reference (test-time torch only). |
| Maple | `cpu_reference/maple.py` | ternary and affine4 pack/dequant, BF16 boundaries, projections, attention/KV spans, routing/MoE, complete model semantics |
| Moonshine decoder | `cpu_reference/moonshine.py` | projection, LayerNorm, partial RoPE, self/cross attention, fixed cache, MLP, residual, tied head/argmax |
| Moonshine encoder | `cpu_reference/moonshine_encoder.py` | convolution, group norm, encoder attention/RoPE, GELU, layout transformations |
| Fixtures | `cpu_reference/fixtures.py` | fixture load/save/run and tolerance contracts |

`register_cpu_reference_kernels()` registers the primitive subset exposed through the four-axis registry. Additional plain NumPy functions remain direct test oracles even when they do not have a registry key.

## HIP gfx11 catalog

Unless a row says otherwise, source is under `hipengine/kernels/hip_gfx1100/`, registration is for `hip_gfx1100`, and the independently allowed subset is aliased under `hip_gfx1151`.

### Shared Qwen / PARO path

These families implement Qwen3.5/Qwen3.6 PARO W4A16, shared W8A16, full-attention, linear-attention, MoE, and common runtime glue. Some are also reused by GGUF paths.

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
| Router/select | `moe/router.{hip,py}` | `router_logits`, `router_select`, `router_topk_shared`, `router_topk_split_shared` | BF16/FP16/F32 hidden/weight combinations; deterministic top-k and shared-gate routes. The compiled library handle is cached at module scope to keep per-launch host cost a plain ctypes call. |
| MoE grouping and packing | `moe/group_scatter.{hip,py}` | `moe_group_count/prefix/scatter`, `moe_group_compact`, `moe_gather_packed_hidden`, `moe_wmma_tile_map`, `moe_mmq_tile_map` | Stable count/prefix/scatter and compact tile metadata; generic and `w4_paro`. |
| MoE prefill orchestration leaf | `moe/prefill.py` | `moe_prefill` (`w4_paro`) | Registered wrapper composition for selected-expert prefill. |
| Whole selected-expert FFN | `quant/paro_moe_ffn_fused.{hip,py}` | `moe_ffn_selected` (`w4_paro`) | Rotate → gate/up → SiLU → down-rotate → down projection megakernel; primitive chain remains fallback. |
| c1 native dispatcher | `dispatch/moe_c1_dispatch.{hip,py}` | C function-table dispatcher (not a registry layer) | Contracts Python launch overhead while invoking registered/raw function pointers; does not replace component kernels. |
| SiLU/rotation primitives | `fused/paro_silu.{hip,py}` | `silu_mul_dual`, `silu_mul_separate`, `silu_mul_dual_rotate`, `silu_mul_pair_rotate` | Primitive and fused activation/down-rotation boundaries coexist; separate BF16 SiLU permits exact in-place replacement of its gate plane. |
| MoE combine/tail | `fused/paro_combine.{hip,py}` | `weighted_lanes_sum`, `weighted_sum`, `shared_gate_combine`, residual/RMSNorm composites | BF16/FP16/F32 values with FP32 route weights/gates; explicit primitive fallbacks are registered. |
| Paged KV write/copy | `attention/paged_kv_write.{hip,py}` | `paged_kv_write`, `paged_kv_copy` (`bf16`, PARO/GGUF, INT8 layouts) | All attention-visible writes consume complete `KVLiveSpans`; includes BF16 and supported INT8 storage formats. |
| Full/paged attention | `attention/paged_attn_decode.{hip,py}` | `full_attn_decode/prefill`, `paged_attn_decode/prefill`, `full_attn_gate_mul` | Contiguous and paged, batched, GQA, split-K, gated reduce, and supported INT8 KV variants. INT8 per-token/head includes a row-batched 24Q/4KV/D256 split-K producer with an explicitly strided BF16 gated reducer; the c1 leaf remains its registered numerical fallback. |
| AOTriton adapter | `attention/aotriton_wrap.py`, `attention/aotriton.py` | `full_attn_prefill` (`w4_paro`, `gguf_qwen35`) | Optional library adapter; native raw-pointer paths remain available. |
| Linear-attention Conv | `linear_attn/conv.{hip,py}` | `linear_attn_*conv_decode/prefill`, chain/tree and snapshot composites | Decode, segmented prefill, verifier tree/chain, and state-snapshot variants. |
| Linear-attention GDN | `linear_attn/gdn.{hip,py}` | `linear_attn_prefill_prepare`, `gdn_*recurrent*`, RMSNorm/gate/rotate/cast/snapshot composites | Exact schedules retain FP32 recurrent state; segmented, chain/tree, snapshot, and decode-order writers cover prefill, verifier, and multi-request selected commit, with optional FP32 state-row journals, direct BF16 handoffs, and an exact FP32 output tap. FP16-state (FP32 accumulation) and gfx1151 cluster/chunked compact-peer variants are explicit opt-ins or capability selections that always retain an FP32 fallback. |
| Runtime state | `runtime/state.{hip,py}` | token embedding, positions/metadata, graph record/commit, scalar state, profiling wall-clock marker | Device-side graph/verify bookkeeping, indexed row state, token publication, and profiling-only steady-clock boundaries. |
| Sampling | `sampling/sampler.{hip,py}` | `sampler`, `mtp_draft_topk` | Greedy/temperature/top-k helpers and bounded draft top-k. |

**Compact DMS attention** — `attention/dms_compact.{hip,py}` registers `dms_extract_decision`, `dms_decision_source`, `dms_streaming_pack`, `dms_append_decode`, and `dms_compact_attn_decode` (grouped GQA fallback plus bounded-LDS split-K) for the compact-KV path. The CPU-reference oracles in `cpu_reference/dms.py` are the registered strict fallbacks for every key; the kernels are wired into `DMSCompactBackend` behind explicit device-payload selection, and no model package defaults to DMS.

### GGUF / Qwen / Laguna path

GGUF is not a PARO alias. Raw GGML blocks, pack8/T16/qmicro/X8 replacement layouts, exact expanded planes, and source-F16 Laguna tensors have distinct storage and registry keys.

#### GGUF projection and quant families

| Quant/layout family | Source / wrapper | Principal registry layers | Stable notes |
| --- | --- | --- | --- |
| Q4_K selected FFN megakernel | `quant/gguf_q4_k_moe_ffn_fused.{hip,py}` | `moe_ffn_selected` (`gguf_q4_k`) | Whole selected gate/up → SiLU → down projection; primitive selected projections remain fallback. |
| Raw Q5_K/Q6_K/Q8_0 | `quant/gguf_k_gemv.{hip,py}` | `linear`, `linear_pair`, `attention_projection_quad` | Decode/prefill, BF16/F32 output, pair/quad launch contractions, rowbatch/coltile variants. |
| Raw Q3_K selected | `quant/gguf_q3_k_gemv.{hip,py}` | `moe_linear` | Q3 selected-expert projection family. |
| Q4_K pack8/raw | `quant/gguf_q4_k_gemv.{hip,py}` | `linear`, `linear_pair`, `linear_pair_silu`, `linear+residual` | Raw GGUF math and lossless pack8 layouts; pair/SiLU and exact rounded-BF16 residual composites where registered. Primitive projection+add fallbacks remain available. |
| Q4_K/Q6_K prefill WMMA | `quant/gguf_q4_k_prefill.{hip,py}` | `linear` | Resident pack8/raw prefill consumers; exact scalar/pack8 routes remain fallbacks. |
| Q8_0 T16 prefill | `quant/gguf_q8_0_t16_prefill.{hip,py}` | `linear`, `linear_pair` | WMMA/T16 Q8 prefill and architecture-specific wave schedules. gfx1151 rows512/K1024/N16+N16 alpha/beta uses the exact two-wave dual owner; singleton WMMA remains the fallback. |
| Q8_0 T16 decode | `quant/gguf_q8_0_t16_gemv.{hip,py}` | `linear`, `linear_pair`, `linear_triple` | Exact T16 Q8 decode GEMV for Qwen3.5-family attention projections (in 2048; fused qkv 8192 + gate 4096). Per-row dual/split owners run at all widths, with an exact 128-thread dual-split rowtile col8 pair owner admitted at rows >= the backend-package floor `GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS`. |
| Q4/Q5/Q6 T16 selected | `quant/gguf_t16_selected_gemv.{hip,py}`, `quant/gguf_k_t16_selected_prefill.{hip,py}` | `linear`, `linear_pair_silu`, `moe_linear`, `moe_linear+weighted_sum`, `linear+residual` | c=1 and selected-prefill T16/qmicro/interleaved consumers, including weighted/residual composites. Exact one-wave/shared-B WMMA rowtile owners cover physical shapes, with grouped-grid siblings, fused dual+SiLU prefill owners, and input-F16 activation siblings (`*_fp16_in_bf16_out`) registered per backend; every shape/row miss and env-disabled path retains the strict one-wave/shared-B or primitive fallback, and current per-shape ownership is backend-package capability data. gfx1100/W7900 additionally retains the exact c8 Q4 selected gate/up pair-reuse dual owner through the package floor `GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS` (2026-09-05 audit packet D1: native-c8 +4.97% with exact repeatable trajectories, arm-identical state differentials on steady c2/c4/c8 and c8 shrink-sparse, natural-prompt duplicate-lane fraction 0.616 versus the direct-fixture 0.5, and a c8 census showing `q4_k_t16_selected_dual_pairreuse_direct_gemv_kernel` x80 with zero scalar fallbacks); the route's geometry gate pins it to x_rows=8/rows=64, lower widths and env-0 keep the per-row dual owner, and the selected-down/Q6-down pair-reuse packets stay unqualified on gfx1100 (floors 0). |
| Q6/Q4 mixed and narrow K/V grids | `fused/gguf_q6_q4_pair.{hip,py}` | `linear_pair` (standard-Q6+Q4, Q4, Q4+planar-Q6) | Exact block-parallel rows1 pairs for standard-Q6+Q4, Q4, and Q4+planar-Q6 layouts; primitive projections remain the registered fallbacks. |
| Dense Q6_K T16/qmicro | `quant/gguf_q6_k_t16_gemv.{hip,py}` | `linear`, `linear+argmax`, `linear+residual` | Exact dense Q6 decode/prefill/root families with wave-shuffle/DPP reductions, grouped-grid and shared-weight WMMA owners for larger rows, and exact col8 rowtiles for small packed rows; `HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK` caps root-logit chunking and registered primitives remain fallbacks. |
| Dense planar-Q6 integer MMQ | `quant/gguf_q4_k_q8_1_selected_prefill.{hip,py}` | `activation_quant`, `linear` | gfx1151 production-profile T2 composite for rows17-48 on sole-resident planar K17408/N5120 down and K5120/N1024 narrow-V: session-owned BF16-to-Q8_1 packing feeding the integer `mmq64x64` consumer; exact A owners remain registered for strict/profile fallback. |
| IQ2/IQ3/IQ4 decode | `quant/gguf_iq_gemv.{hip,py}` | `moe_linear` | Raw IQ selected-expert projection families; IQ3 tile4 is scoped to the gfx1100 explicit-DFlash route and gfx1151 keeps tile1. |
| IQ selected prefill | `quant/gguf_iq_selected_prefill.{hip,py}` | `moe_linear` | Grouped/expert-major, active-expert, rowbatch, and output-ownership variants. |
| Raw-K activation MMQ | `quant/gguf_k_mmq_prefill.{hip,py}` | `activation_quant`, `linear` | Q8_1 producer layouts plus Q5/Q6 MMQ consumers; retained diagnostics may not be runtime defaults. The gfx1100 C8 Q5 owner choice between K-major source MMQ and raw MMQ is capability/env data (`HIPENGINE_GGUF_C8_Q5_SOURCE_MMQ`, `HIPENGINE_GGUF_C8_Q5_RAW_MMQ`) in the backend package. |
| Raw-IQ source MMQ | `quant/gguf_iq_source_mmq_prefill.{hip,py}` | `moe_linear` | Source-faithful IQ MMQ diagnostic/alternative consumers. |
| Exact expanded F32 planes | `quant/gguf_q5_k_f32_rocblas_prefill.{hip,py}` | `linear` and raw-quant composites | Raw Q5/Q6 producers plus ordered exact consumers; library SGEMM variants are distinct diagnostic paths. |
| Source-F16 Q4/Q5/Q6 library route | `quant/gguf_q6_k_f16_rocblas_prefill.{hip,py}` | dequant/cast/`linear` composites | Bounded tile producers feeding F16 rocBLAS for Q4T16/Q5T16 and raw/sole-planar-Q6T16, with scalar, pair-, and octet-owned producer variants. Changed arithmetic is model/shape gated; scalar producers and exact T16 kernels remain registered fallbacks, and decode, verifier, peer backends, and unqualified shapes stay exact. |
| Embedding | `quant/gguf_q6_k_embedding.{hip,py}` | `embedding` (`gguf_q4_k/q5_k/q6_k/q8_0`) | Raw GGUF row lookup for root/token tables. |
| X8 sidecars/replacements | `quant/gguf_x8_selected_gemv.{hip,py}` and pack8 modules | selected `moe_linear` / top-1 helpers | GGML-style packed selected-expert and head diagnostics/qualified lanes. |
| Q8 dp4a verifier | `quant/gguf_q8_0_dp4a_gemv.{hip,py}` | `linear` pair/triple/rowtile variants | q8_1+sudot4 verifier/draft families; selection is route-specific. The Q6 X8 direct-top1 consumer is c1-only for shared-slot AR; multi-row uses Q6 rowtile logits plus GPU argmax. |
| Selected pack8/T16 support files | `quant/gguf_*selected*.{hip,py}`, `quant/gguf_*pack8*.{hip,py}`, `quant/gguf_*t16*.{hip,py}` | `linear`, `linear_pair_silu`, `moe_linear`, producer/metadata variants | Build/registration partitions for selected-expert storage layouts; exact ownership stays in each wrapper. |

Model-, quant-, and shape-specific owner selection for the dense Qwen3.6, Qwen3.8, and Qwen3.5-0.8B GGUF paths (payload plans, decode rowtiles, c=N decode maps, source-F16 library admissions, fused c1 composites, and norm/KV capability keys) is capability/policy data in `hip_gfx1100/__init__.py`, `hip_gfx1151/__init__.py`, and the GGUF dispatch wrappers — not catalog prose. Performance and correctness evidence for each selection lives in `benchmarks/results/` and the corresponding immutable worklog entries.

The numerous small files named `gguf_*selected*`, `gguf_*pack8*`, `gguf_*t16*`, and `gguf_*prefill*` are registration/build partitions of these storage families. The exact per-variant inventory is the registry plus the source directory, not old campaign prose.

#### Laguna model families

| Functional family | Source / wrapper | Principal registry layers/quants | Stable notes |
| --- | --- | --- | --- |
| Source-F16 projections | `linear/laguna_f16_projection.{hip,py}` | `linear`, `linear_pair/triple/quad`, `linear+add+rmsnorm` (`fp16_weight`) | Decode GEMV, exact tiled prefill, compensated WMMA diagnostics/qualified routes, projection-boundary composites. |
| Router and route combine | `moe/laguna_router.{hip,py}` | `laguna_router_topk`, `laguna_sigmoid_router_topk`, `weighted_sum` | Stable sigmoid correction/top-k and route-weight reductions. |
| KV write and attention | `attention/laguna_kv_attention.{hip,py}` | `laguna_kv_write`, `laguna_attention_decode`, `laguna_attention_prefill` | Global and SWA, scalar/bulk, exact qrow, online/changed-association, split/fused GQA, dense-prefix/ring, and long-context variants; complete `KVLiveSpans` ABI throughout. |
| Source F16-WMMA attention | `attention/laguna_flash_attention_prefill.{hip,py}` | `laguna_attention_prefill` diagnostic variant | Source-faithful changed-association leaf; does not replace exact attention without the full quality gate. |
| Head/RoPE/KV composites | `attention/laguna_kv.{py}` over `laguna_kv_attention.hip` | `head_rmsnorm+partial_rotary+kv_write`, projection+head+KV | Registered fused boundaries retain primitive head norm/RoPE and writer fallbacks. |
| Norm/RoPE/glue | `fused/gguf_ops.{hip,py}` | `rmsnorm`, `add_rmsnorm`, `head_rmsnorm+partial_rotary`, attention gate helpers | GGUF F32-weight norm and Qwen/Laguna head-prelude primitives/composites. |
| Softplus attention gate | `fused/laguna_attention.{hip,py}` | `attention_gate` (`f32`) | Generic and mixed-layout/prefill-tile output gates. |
| Host-batched kernel launches | `runtime/laguna_launch_batch.{hip,py}` | `linear+moe_tail+next_rmsnorm_host_batch` | Native launch contraction over already registered exact component kernels. |

Architecture-specific Laguna route choices live in `hip_gfx1100/__init__.py` and `hip_gfx1151/__init__.py`. Keep rationale/results in worklogs and benchmark artifacts; keep only family existence here.

### Maple path

The Maple path uses ternary projection weights, affine4 embedding/head weights, dense BF16 routing, and the common `KVLiveSpans` contract.

| Functional family | Source / wrapper | Principal registry layers | Notes |
| --- | --- | --- | --- |
| Ternary/affine4 projections | `quant/maple_ternary.{hip,py}` | `maple_ternary_gemv/gemm/qkv`, `maple_selected_ternary(_dual)`, `maple_affine4_embed/gemv` | 2-bit ternary and group-64 affine4 storage; grouped expert-major and c1/batched head variants. |
| Attention/KV | `attention/maple_attention.{hip,py}` | `maple_kv_span_update`, `maple_qknorm_rope_kv_write`, `maple_attention_decode/prefill` | Standard QK RMSNorm, partial RoPE, BF16 ring KV, GQA decode and prefill; all readers/writers use complete spans. |
| Router/MoE tail | `moe/maple_moe.{hip,py}` | `maple_router_topk`, `maple_clamped_swiglu`, `maple_weighted_residual` | Stable top-k, clamp-7 SwiGLU, and selected weighted residual. |
| Shared norm/head helpers | `norm/rmsnorm`, `linear/lm_head`, `moe/group_scatter` | norm, argmax/top-k, compact metadata | Reused through Maple's registered backend/quant keys. |

### Moonshine path

| Functional family | Source / wrapper | Principal registry layers | Notes |
| --- | --- | --- | --- |
| FP16 projections | `linear/moonshine_projection.{hip,py}` | projection single/rows/bias/pair/QKV/cross-KV/lm-head and MLP boundaries | Decoder projections and direct head-major cross-KV output. |
| W8A16 projections | `linear/moonshine_w8a16.{hip,py}` | Moonshine projection/QKV/cross-KV/MLP/lm-head (`w8a16`) | Quantized peer family with FP16 path as fallback. |
| LayerNorm | `norm/moonshine_layernorm.{hip,py}` | `moonshine_layernorm`, residual+LayerNorm | FP32 statistics with explicit rounded FP16 boundary. |
| Glue primitives | `fused/moonshine_glue.{hip,py}` | embedding, residual, partial RoPE, self-cache, RoPE+cache, argmax | Fixed-cache decoder glue and deterministic lowest-ID selection. |
| MLP activation | `fused/moonshine_mlp.{hip,py}` | `moonshine_gated_silu` | FP16 value/gate split with FP32 activation math. |
| Self/cross attention | `attention/moonshine_attention.{hip,py}` | `moonshine_self_attention`, `moonshine_cross_attention` | Logical-dim-52 self/cross attention, cache buckets, and parallel-token variants. |

Encoder kernels are currently CUDA-only; see the CUDA catalog below.

### Speculative decoding path

| Functional family | Source / wrapper | Principal registry layers/quants | Notes |
| --- | --- | --- | --- |
| DFlash drafter | `speculative/dflash_drafter.{hip,py}` | `dflash_*` projection, norm, attention, activation, metadata layers (`w4_paro`) | Raw-pointer drafter primitives; target verification remains transaction-shaped. |
| DFlash2 drafter reference | `speculative/dflash2_drafter.py` + `cpu_reference/dflash2.py` | `dflash2_grouped_conv`, `dflash2_selector`, `dflash2_selector_path`, `dflash2_attention_forward`, `dflash2_rope_tables` (`fp32`) | Torch-free NumPy DFlash2 exactness reference (grouped dynamic conv, top-16 bilinear selector, q/k-norm sliding attention). Golden fixtures from z-lab/dflash @ 07ebd93; native kernels land in D2. Source lineage: `docs/source_lineage.json` (repo `dflash`). |
| DFlash2 native kernels | `speculative/dflash2.{hip,py}` | `dflash2_grouped_conv`, `dflash2_top16_rows`, `dflash2_selector` (`bf16`/`fp32`) | Native grouped dynamic conv (strided side views over the 1280-wide projection), top-16 logits, and the low-rank bilinear candidate-selector greedy walk. Strict RED vs `cpu_reference/dflash2.py` (BF16 round-trip modeled); registered for `hip_gfx1100` + `hip_gfx1151`. D2a. |
| DFlash acceptance | `speculative/dflash_accept.{hip,py}` | `dflash_accept_chain`, `speculative_accept_commit` | GGUF/PARO acceptance and bounded commit summaries. |
| DFlash commit/state | `speculative/dflash_commit.{hip,py}` | `dflash_commit_chain`, `linear_state_pair_*` | Transactional selected-state and cursor commit helpers. gfx1100 target verification reads initial Conv/GDN state from a resident multi-slot slab with the strict chunked pointer-table import as rollback; gfx1151 keeps the packed-state route, and a per-layer HIP D2D chain remains a lower strict fallback on gfx1100. |
| MTP core | `speculative/mtp.{hip,py}` | MTP norm/fuse/router/top-k/gate/finalize/route accumulation | Provider-neutral proposal/acceptance primitives. |
| MTP NextN | `speculative/mtp_nextn.{hip,py}` | `mtp_nextn_*`, quant GEMVs, shared head | GGUF NextN layer, attention, MoE, and projection helpers. The exact K/V-only full-attention branch owns prompt priming and accepted-tail repair by default; `HIPENGINE_GGUF_NEXTN_ACCEPT_KV_WRITE_ONLY=0` restores the complete NextN block. |

Detailed provider/runtime status belongs in `MTP.md`, `DFLASH.md`, worklogs, and benchmark artifacts.

## CUDA sm_120a catalog

CUDA families are implemented independently under `hipengine/kernels/cuda_sm120a/` and registered only by that backend package.

### Maple

| Functional family | Source / wrapper | Principal registry layers | Notes |
| --- | --- | --- | --- |
| Ternary/affine4 projections | `quant/maple_ternary.{cu,py}` | Maple ternary, selected expert, affine4 embedding/head layers | CUDA peer of the Maple packed storage contract. |
| Attention/KV | `attention/maple_attention.{cu,py}` | Maple span update, QK/RoPE/KV write, decode/prefill | Complete spans and CUDA warp32-specific implementations. |
| Router/MoE | `moe/maple_moe.{cu,py}`, `moe/group_scatter.{cu,py}` | Maple router/SwiGLU/weighted residual; compact metadata | Stable selection and grouped native-prefill support. |
| Norm and final reductions | `norm/maple_rmsnorm.{cu,py}`, `linear/maple_lm_head.{cu,py}` | RMSNorm/add/head norm, lm-head/argmax/top-k | Independent CUDA launch/runtime wrappers. |

### Moonshine

| Functional family | Source / wrapper | Principal registry layers | Notes |
| --- | --- | --- | --- |
| Decoder projections | `linear/moonshine_projection.{cu,py}`, `linear/lm_head.{cu,py}` | single/rows/bias/pair/QKV/cross-KV/MLP/lm-head | FP16 projection families and bounded fused head/top-1 routes. |
| LayerNorm and MLP | `norm/moonshine_layernorm.{cu,py}`, `fused/moonshine_mlp.{cu,py}` | LayerNorm, residual+LayerNorm, gated SiLU | CUDA warp reductions and explicit FP16 boundaries. |
| Decoder glue | `fused/moonshine_glue.{cu,py}` | embedding/residual/RoPE/cache/argmax plus position/result publication | Includes device-owned decode control helpers. |
| Self/cross attention | `attention/moonshine_attention.{cu,py}` | self/cross attention variants | CUDA-native scalar/batched cache routes. |
| CUTLASS attention | `attention/moonshine_attention_cutlass.{cu,py}` | `moonshine_self_attention` AOT variants | Optional architecture-qualified library path; native attention remains fallback. |
| Encoder core | `encoder/moonshine_encoder.{cu,py}` | conv1/2/3, group norm, GELU, encoder RoPE/attention/transpose | Torch-free CUDA encoder primitives. |
| Encoder library adapters | `encoder/moonshine_encoder_lt.{cu,py}`, `encoder/moonshine_encoder_cudnn.{cu,py}` | projection/attention/conv alternatives | CUDA-only cuBLASLt/cuDNN candidates or selected routes. |

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
│   └── paged_kv_write.hip
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
│   └── paro_silu.hip
├── linear/
│   ├── dense_gemv.hip
│   ├── laguna_f16_projection.hip
│   ├── lm_head.hip
│   ├── moonshine_projection.hip
│   └── moonshine_w8a16.hip
├── linear_attn/
│   ├── conv.hip
│   └── gdn.hip
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
│   ├── gguf_q4_k_q8_1_selected_prefill.hip
│   ├── gguf_q4_k_selected_pack8_gemv.hip
│   ├── gguf_q4_k_selected_prefill.hip
│   ├── gguf_q4_k_t16_selected_prefill.hip
│   ├── gguf_q5_k_f32_rocblas_prefill.hip
│   ├── gguf_q6_k_embedding.hip
│   ├── gguf_q6_k_f16_rocblas_prefill.hip
│   ├── gguf_q6_k_pack8_gemv.hip
│   ├── gguf_q6_k_t16_gemv.hip
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

A `+` in a registry layer name denotes a composite boundary. Every fused
composite must have a registered strict unfused route. Strict composites satisfy
their declared exact/parent-parity boundary; production composites may
reassociate only under a certified profile manifest and still fall back to the
strict chain. The table groups registered composites by semantic family; exact
variants/dtypes remain in source.

| Composite family | Backends / paths | Required unfused chain |
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
| `moe_ffn_selected/fused_rotate_dual_silu_rotate_down_*` | HIP PARO | rotate1 → selected dual pack8 gate/up → SiLU/down-rotate → selected down pack8 projection |

Fallback requirements:

- Strict fused and unfused paths share exact/parent-parity fixtures at every
  published low-precision boundary. Production fused paths share the strict
  fixture plus the full strict-teacher profile gate; free-running ID equality is
  diagnostic unless strict/batch-invariant says otherwise.
- Removing a strict fallback is an architectural change and requires updating this table plus `PLAN.md` if the invariant changes.
- A library call can be one stage of an unfused chain, but it does not waive the independent primitive/oracle route.

## Source-lineage audit

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

### HIP build profiles

| Profile | Important flags | Wavefront | Typical use |
| --- | --- | --- | --- |
| `decode` | local unroll threshold plus `-mcumode` | 32 | paged attention, GEMV, decode MoE |
| `prefill` | local unroll threshold, WGP mode | 32 | GEMM/WMMA and multi-row prefill |
| `baseline` | minimal flags | 32 | debug and fallback |

Wave32 is the gfx11 default. Use wave32 shuffles within a wave and LDS for cross-wave exchange. Wave64 is an isolated experiment only and requires explicit flags, probes, ISA checks, correctness fixtures, and end-to-end evidence.

### JIT cache and profiling

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
