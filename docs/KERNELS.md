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
| RMSNorm | `norm/rmsnorm.{hip,py}`, `fused/gguf_ops.{hip,py}` | `rmsnorm`, `add_rmsnorm`, `add_rmsnorm_f32`, `head_rmsnorm` (`bf16`, `w4_paro`, `gguf_f32_weight`) | Qwen weights use delta semantics; PARO out variants use direct norm weights. GGUF includes exact generic fallbacks plus fixed c1/hidden-1024 wave-shuffle candidates for standalone and unrounded add+norm boundaries. |
| Rotary/prelude | `rotary/paro_rotate.{hip,py}`, `rotary/qwen35_rotary.{hip,py}` | `paro_rotate1/2/3`, `paro_rmsnorm_rotate2`, `partial_rotary`, `head_rmsnorm+partial_rotary`, `split_qgate` | BF16/FP16 PARO rotation and Qwen partial-RoPE/head-normalization families. |
| Dense projection and head | `linear/dense_gemv.{hip,py}`, `linear/lm_head.{hip,py}` | `dense_gemv`, `dense_dual_gemv`, `linear_pair`, `linear+residual`; `lm_head`, `lm_head_argmax`, `argmax`, `topk` | Dense fallback/auxiliary projection plus deterministic final reductions. Includes an exact rounded-BF16 residual sibling; the unfused projection+add chain remains registered. |
| PARO AWQ projection | `quant/paro_awq_gemv.{hip,py}` | `pack8_gemv`, `dual_pack8_gemv`, `selected_*pack8_gemv`, `pack8_gemm`, rotate/SiLU composites (`w4_paro`) | Strided/transposed, BF16/FP16, selected-expert, fused-W4 prefill, and small-row routes. |
| PARO Marlin-K | `quant/paro_marlin_k.{hip,py}` | `marlin_k_gemv` (`w4_paro`) | c=1 replacement layout; pack8 alias remains available to prefill/fused projections. |
| PARO compact WMMA | `wmma/paro_awq_wmma.{hip,py}` | `awq_wmma` (`w4_paro`, `bf16`) | Compact/non-compact selected gate/up and down prefill; exact GEMV routes remain fallback. |
| W8A16 projection/shared expert | `quant/w8a16_linear.{hip,py}` | `w8a16_linear` (`w8a16`, `w4_paro`) | Single/multi-row lowp projection and shared-expert helper variants. |
| Router/select | `moe/router.{hip,py}` | `router_logits`, `router_select`, `router_topk_shared`, `router_topk_split_shared` | BF16/FP16/F32 hidden/weight combinations; deterministic top-k and shared-gate routes. |
| MoE grouping and packing | `moe/group_scatter.{hip,py}` | `moe_group_count/prefix/scatter`, `moe_group_compact`, `moe_gather_packed_hidden`, `moe_wmma_tile_map`, `moe_mmq_tile_map` | Stable count/prefix/scatter and compact tile metadata; generic and `w4_paro`. |
| MoE prefill orchestration leaf | `moe/prefill.py` | `moe_prefill` (`w4_paro`) | Registered wrapper composition for selected-expert prefill. |
| Whole selected-expert FFN | `quant/paro_moe_ffn_fused.{hip,py}` | `moe_ffn_selected` (`w4_paro`) | Rotate → gate/up → SiLU → down-rotate → down projection megakernel; primitive chain remains fallback. |
| c1 native dispatcher | `dispatch/moe_c1_dispatch.{hip,py}` | C function-table dispatcher (not a registry layer) | Contracts Python launch overhead while invoking registered/raw function pointers; does not replace component kernels. |
| SiLU/rotation primitives | `fused/paro_silu.{hip,py}` | `silu_mul_dual`, `silu_mul_separate`, `silu_mul_dual_rotate`, `silu_mul_pair_rotate` | Primitive and fused activation/down-rotation boundaries coexist; separate BF16 SiLU permits exact in-place replacement of its gate plane. |
| MoE combine/tail | `fused/paro_combine.{hip,py}` | `weighted_lanes_sum`, `weighted_sum`, `shared_gate_combine`, residual/RMSNorm composites | BF16/FP16/F32 values with FP32 route weights/gates; explicit primitive fallbacks are registered. |
| Paged KV write/copy | `attention/paged_kv_write.{hip,py}` | `paged_kv_write`, `paged_kv_copy` (`bf16`, PARO/GGUF, INT8 layouts) | All attention-visible writes consume complete `KVLiveSpans`; includes BF16 and supported INT8 storage formats. |
| Full/paged attention | `attention/paged_attn_decode.{hip,py}` | `full_attn_decode/prefill`, `paged_attn_decode/prefill`, `full_attn_gate_mul` | Contiguous and paged, batched, GQA, split-K, gated reduce, and supported INT8 KV variants. gfx1151 Qwen3.5-0.8B rows1/8Q/2KV/D256 selects generic split-K3+fused BF16 gate at cap514-641; fixed256 and unsupported shapes/backends remain fallbacks. |
| AOTriton adapter | `attention/aotriton_wrap.py`, `attention/aotriton.py` | `full_attn_prefill` (`w4_paro`, `gguf_qwen35`) | Optional library adapter; native raw-pointer paths remain available. |
| Linear-attention Conv | `linear_attn/conv.{hip,py}` | `linear_attn_*conv_decode/prefill`, chain/tree and snapshot composites | Decode, segmented prefill, verifier tree/chain, and state-snapshot variants. |
| Linear-attention GDN | `linear_attn/gdn.{hip,py}` | `linear_attn_prefill_prepare`, `gdn_*recurrent*`, RMSNorm/gate/rotate/cast/snapshot composites | Exact and quality-gated schedules; recurrent state remains FP32. gfx1151 Q4 and Q8 `(16K,16V,128,128)` select cluster8; all other gfx1151 shapes retain exact nonvolatile LDS32. |
| Runtime state | `runtime/state.{hip,py}` | token embedding, positions/metadata, graph record/commit, scalar state, profiling wall-clock marker | Device-side graph/verify bookkeeping, indexed row state, token publication, and profiling-only steady-clock boundaries. |
| Sampling | `sampling/sampler.{hip,py}` | `sampler`, `mtp_draft_topk` | Greedy/temperature/top-k helpers and bounded draft top-k. |

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
| Q8_0 T16 prefill | `quant/gguf_q8_0_t16_prefill.{hip,py}` | `linear` | WMMA/T16 Q8 prefill and architecture-specific wave schedules. |
| Q4/Q5/Q6 T16 selected | `quant/gguf_t16_selected_gemv.{hip,py}` | `linear`, `linear_pair_silu`, `moe_linear`, `moe_linear+weighted_sum`, `linear+residual` | c=1 and selected-prefill T16/qmicro/interleaved consumers, including weighted/residual composites. |
| IQ2/IQ3/IQ4 decode | `quant/gguf_iq_gemv.{hip,py}` | `moe_linear` | Raw IQ selected-expert projection families. |
| IQ selected prefill | `quant/gguf_iq_selected_prefill.{hip,py}` | `moe_linear` | Grouped/expert-major, active-expert, rowbatch, and output-ownership variants. |
| Raw-K activation MMQ | `quant/gguf_k_mmq_prefill.{hip,py}` | `activation_quant`, `linear` | Q8_1 producer layouts plus Q5/Q6 MMQ consumers; retained diagnostics may not be runtime defaults. |
| Raw-IQ source MMQ | `quant/gguf_iq_source_mmq_prefill.{hip,py}` | `moe_linear` | Source-faithful IQ MMQ diagnostic/alternative consumers. |
| Exact expanded F32 planes | `quant/gguf_q5_k_f32_rocblas_prefill.{hip,py}` | `linear` and raw-quant composites | Raw Q5/Q6 producers plus ordered exact consumers; library SGEMM variants are distinct diagnostic paths. |
| Source-F16 Q4/Q5/Q6 library route | `quant/gguf_q6_k_f16_rocblas_prefill.{hip,py}` | dequant/cast/`linear` composites | Sole Q4T16/Q5T16 and raw/sole-planar-Q6T16 bounded tile producers feeding F16 rocBLAS; Q4T16 includes scalar column-owned and exact adjacent-pair-owned producers, while Q5T16 includes scalar plus exact pair- and natural-octet-owned producer leaves. Changed arithmetic is model/shape gated, while exact T16 remains the small-row and miss fallback. Qwen3.6-27B admits Q5T16 recurrent output with its natural-octet producer at M512-M4096, bounded Q4T16 full-attention Q with its adjacent-pair producer at M512-M2047, and Q4T16 linear-attention gate only as the second operand behind the already-admitted Q6T16 QKV peer at M512-M2047 after complete category and cross-board full-engine qualification. A generic ordered-pair policy prevents that gate shape from claiming standalone or Q4/Q4 pair dispatch, while request-row filtering and a per-shape ceiling keep M2048/4K exact. Scalar producer and exact T16 kernels remain registered policy-miss fallbacks; decode, verifier, peer backends, and every unqualified shape remain exact. |
| Embedding | `quant/gguf_q6_k_embedding.{hip,py}` | `embedding` (`gguf_q4_k/q5_k/q6_k/q8_0`) | Raw GGUF row lookup for root/token tables. |
| X8 sidecars/replacements | `quant/gguf_x8_selected_gemv.{hip,py}` and pack8 modules | selected `moe_linear` / top-1 helpers | GGML-style packed selected-expert and head diagnostics/qualified lanes. |
| Q8 dp4a verifier | `quant/gguf_q8_0_dp4a_gemv.{hip,py}` | `linear` pair/triple/rowtile variants | q8_1+sudot4 verifier/draft families; selection is route-specific. |
| Selected pack8/T16 support files | `quant/gguf_*selected*.{hip,py}`, `quant/gguf_*pack8*.{hip,py}`, `quant/gguf_*t16*.{hip,py}` | `linear`, `linear_pair_silu`, `moe_linear`, producer/metadata variants | Build/registration partitions for selected-expert storage layouts; exact ownership stays in each wrapper. |

For dense Qwen3.6-27B on gfx1100, the package-default rank-2 Q4 map is one
`gguf_q4_k_t16_v1/tiles` payload across all 288 tensors. Its c1/rows-2-4 owners
live in `gguf_t16_selected_gemv.{hip,py}` and its M16-through-M4096/tail
shared-B owner lives in `gguf_k_t16_selected_prefill.{hip,py}`; the output-major
K256 LDS slab is the retained implementation. For the model's dense
K=5,120/N=17,408 FFN gate/up pair at M>=512, the same prefill family also owns
an operation-complete dual-output WMMA+SiLU variant: one four-wave block reuses
each activation fragment across independent gate/up weights, rounds both
projection outputs to BF16 in LDS, then applies the existing SiLU boundary.
The model's Q4/Q4 linear-attention K=5,120/N=10,240+6,144 pair has a separate
exact unequal-output owner at M>=512: a dual-WMMA shared 6,144-column prefix plus
a singleton-geometry QKV tail, with two direct BF16 outputs and no new storage.
Raw/pack8 Q4 bodies remain registered for other layouts and diagnostics, not as
dense-27B sidecars. Evidence: [`XTX first fit`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-sole-t16-first-fit.json),
[`output-major LDS keep`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-q4-t16-output-major-lds.json),
[`dual-WMMA SiLU keep`](../benchmarks/results/2026-08-13-qwen36-27b-q4-dual-wmma-silu-prefill-retained.json),
[`unequal Q4 pair keep`](../benchmarks/results/2026-08-13-qwen36-27b-q4-unequal-dual-prefill-retained.json),
and [`live residency/correctness`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-correctness-residency.json).

For dense Qwen3.5-0.8B Q4_K_M on gfx1151, exact role/shape plugin policy also
keeps one compact Q4T16 payload for the six full-attention Q projections at
K=1,024/N=4,096. The existing direct leaf owns c1, exact rowtile owns c2-c4,
physical c8 is split into two exact c4 launches by backend capability, and the
existing T16 WMMA owner handles bulk rows. Every other Q4 role, Qwen3.6-27B,
and peer backends retain their prior residents. No attention kernel or
`KVLiveSpans` ABI changes. Evidence:
[`0.8B Q4T16 attention-Q route`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-q4t16-attn-q-route.json).

The same model/quant/backend also owns one operation-complete p512 dense-FFN
prefill route over the sole resident pack8 gate/up weights. A 128-thread,
32-column x 256-row WMMA body decodes both matrices into one 32-KiB LDS union,
reuses each activation fragment across gate and up, rounds both projection
boundaries to BF16 in LDS, and emits the existing BF16 SiLU product directly.
The route is qualified only for rows512/K1024/N3584 by model/quant plugin policy;
two registered singleton WMMAs plus standalone SiLU remain the exact fallback.
No resident bytes are added. Evidence:
[`0.8B operation-complete pack8 prefill`](../benchmarks/results/2026-08-15-gfx1151-qwen35-08b-pack8-dual-wmma-silu-prefill.json).

The same model/quant/backend has one separately qualified decode-only composite:
`(hip_gfx1151, linear_pair_silu, gguf_q4_k,
pack8_dual_decode_t128_bf16_bf16_out)` for c1 K=1,024/N=3,584 dense gate/up.
It binds the existing sole-pack8 dual-SiLU body to 128 threads, replacing 24
dual-t32 plus 24 standalone SiLU launches with 24 fused launches. Model/file
identity and exact shape are backend capability data; Q8, other models/shapes,
rows >1, and peer backends retain prior registered owners and the unfused
fallback. No persistent bytes or hot scratch are added. Evidence:
[`0.8B fused dense decode route`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-dense-fused-decode-retained.json).

Its dense-down boundary has a second exact decode-only policy for c1
K=3,584/N=1,024. Twelve existing Q4-pack8 t32 owners derive
`linear+residual/gguf_q4_k/pack8_bf16_residual_bf16_out`; twelve existing
dense-BF16 t256 owners derive
`linear+residual/bf16/out_bf16_residual_bf16_out`. Both round the projection to
BF16 before adding the BF16 residual in FP32 and rounding the sum to BF16. The
model/file/shape policy is resolved once when the resident runner initializes.
It adds no layout, persistent bytes, or hot scratch; Q8, rows >1, other models,
and peer backends retain the primitive projection+add or their prior registered
small-row composites. Evidence:
[`0.8B fused dense-down residual route`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-dense-down-residual-retained.json).

The model's separately screened D5 norm boundary has two fixed c1/hidden-1,024
registry candidates:
`rmsnorm/gguf_f32_weight/bf16_out_fixed1024_wave256` and
`add_rmsnorm/gguf_f32_weight/bf16_out_fixed1024_wave256`. Each caches four
values per local256 thread, reduces within wave32 using HIP shuffles, and uses
eight shared wave sums plus two block barriers. The add form preserves the
existing unrounded-F32 normalization and rounded-BF16 residual contract. Generic
t256 primitives remain registered fallbacks; no layout, persistent bytes, hot
scratch, or node count is added. The production capability selects both keys as
one C route only for gfx1151 Qwen3.5-0.8B Q4_K_M c1/hidden-1,024
attention/post-attention owners. Resident dispatch caches a prevalidated
registry partial after exact capability resolution; public wrappers and HIP
entry points retain their full validation contract. Q8, output norm, verifier
F32, rows>1, other shapes/models, and peer backends stay generic. Evidence:
[`0.8B retained norm/residual route`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-norm-residual-retained.json),
[`screen`](../benchmarks/results/2026-08-14-gfx1151-qwen35-08b-norm-residual-screen.json).

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
| DFlash acceptance | `speculative/dflash_accept.{hip,py}` | `dflash_accept_chain`, `speculative_accept_commit` | GGUF/PARO acceptance and bounded commit summaries. |
| DFlash commit/state | `speculative/dflash_commit.{hip,py}` | `dflash_commit_chain`, `linear_state_pair_*` | Transactional selected-state and cursor commit helpers. |
| MTP core | `speculative/mtp.{hip,py}` | MTP norm/fuse/router/top-k/gate/finalize/route accumulation | Provider-neutral proposal/acceptance primitives. |
| MTP NextN | `speculative/mtp_nextn.{hip,py}` | `mtp_nextn_*`, quant GEMVs, shared head | GGUF NextN layer, attention, MoE, and projection helpers. |

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

A `+` in a registry layer name denotes a composite boundary. Every fused composite must have a numerically equivalent unfused route. The table groups registered composites by semantic family; exact variants/dtypes remain in source.

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

- Fused and unfused paths must share fixtures or direct equality tests at every published low-precision boundary.
- Removing a fallback is an architectural change and requires updating this table plus `PLAN.md` if the invariant changes.
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

Check expected kernel identity, plausible duration, workgroup/grid, VGPR, LDS, and scratch. `Scratch_Size > 0` on a hot path is a review trigger. Some profiler versions expose start/end timestamps instead of `DurationNs`; subtract them. Raw profiler dumps stay outside Git.

For MTP, profile the final child (`scripts/mtp_verifier_rocprof.py` or the final smoke), not the parent economics/prompt-suite harness that launches nested Python processes.

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

Backend packages may refresh missing keys after test isolation. `hip_gfx1151` aliases only allowed gfx11 registrations; `cuda_sm120a` registers only independent CUDA implementations.

## Correctness and profiler gate

A new or ported kernel lands only when all applicable checks pass:

1. **RED fixture/oracle:** write or identify the CPU/primitive oracle before implementation when math or storage changes.
2. **Registry:** exact intended keys resolve under the correct backend, layer, quant, and variant.
3. **Numerics:** KL ≤ 0.05 and top-1 agreement ≥ 90% versus `cpu_reference` for net-new math; a mechanical split/port also preserves its parent or prior in-tree boundary.
4. **Fallback:** fused composites match their registered unfused chain.
5. **Profiler:** cache-only `rocprofv3 --kernel-trace` or Nsight trace names the expected kernel with plausible resources/duration.
6. **Integration:** run the narrowest applicable deterministic/model gate from `TESTING.md`.
7. **Evidence:** performance claims follow `BENCHMARK.md` and update artifact/rollup/changelog/worklog; do not add the narrative here.

## Per-family port checklist

1. Audit `source_lineage.json` and run the narrow lineage check.
2. Add the CPU reference or bit-exact fixture (RED).
3. Copy one functional family into `hipengine/kernels/<backend>/<family>/`; do not mix unrelated families.
4. Retype launch wrappers to raw pointers and explicit metadata.
5. Preserve or document storage layout, low-precision boundaries, `KVLiveSpans`, launch bounds, and build profile.
6. Register exact four-axis keys and any required unfused fallback keys.
7. Update the relevant catalog row and fused fallback map without benchmark commentary.
8. Run registry, numerical, profiler, and narrow integration gates.
9. Record decisions/results in a new immutable worklog entry; write compact benchmark artifacts only when making a performance claim.
10. Commit the validated family as one logical unit with source commit provenance when ported.
