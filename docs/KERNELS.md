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
| RMSNorm | `norm/rmsnorm.{hip,py}`, `fused/gguf_ops.{hip,py}` | `rmsnorm`, `add_rmsnorm`, `add_rmsnorm_f32`, `head_rmsnorm` (`bf16`, `w4_paro`, `gguf_f32_weight`) | Qwen weights use delta semantics; PARO out variants use direct norm weights. GGUF includes exact generic fallbacks plus gfx1151-qualified fixed c1/hidden-1024 and hidden-5120 wave-shuffle candidates for standalone and unrounded add+norm boundaries. |
| Rotary/prelude | `rotary/paro_rotate.{hip,py}`, `rotary/qwen35_rotary.{hip,py}` | `paro_rotate1/2/3`, `paro_rmsnorm_rotate2`, `partial_rotary`, `head_rmsnorm+partial_rotary`, `split_qgate` | BF16/FP16 PARO rotation and Qwen partial-RoPE/head-normalization families. |
| Dense projection and head | `linear/dense_gemv.{hip,py}`, `linear/lm_head.{hip,py}` | `dense_gemv`, `dense_dual_gemv`, `linear_pair`, `linear+residual`; `lm_head`, `lm_head_argmax`, `argmax`, `topk` | Dense fallback/auxiliary projection plus deterministic final reductions. BF16 hidden/weight GEMV has both BF16 and unrounded F32 outputs, including the strict full-logit BF16-GGUF head route. gfx1151 rows512/K3584/N1024 dense-BF16 FFN down uses the WMMA exact rounded-residual sibling; the unfused projection+add chain remains registered. |
| PARO AWQ projection | `quant/paro_awq_gemv.{hip,py}` | `pack8_gemv`, `dual_pack8_gemv`, `selected_*pack8_gemv`, `pack8_gemm`, rotate/SiLU composites (`w4_paro`) | Strided/transposed, BF16/FP16, selected-expert, fused-W4 prefill, and small-row routes. |
| PARO Marlin-K | `quant/paro_marlin_k.{hip,py}` | `marlin_k_gemv` (`w4_paro`) | c=1 replacement layout; pack8 alias remains available to prefill/fused projections. |
| PARO compact WMMA | `wmma/paro_awq_wmma.{hip,py}` | `awq_wmma` (`w4_paro`, `bf16`) | Compact/non-compact selected gate/up and down prefill; exact GEMV routes remain fallback. |
| W8A16 projection/shared expert | `quant/w8a16_linear.{hip,py}` | `w8a16_linear` (`w8a16`, `w4_paro`) | Single/multi-row lowp projection and shared-expert helper variants. |
| Router/select | `moe/router.{hip,py}` | `router_logits`, `router_select`, `router_topk_shared`, `router_topk_split_shared` | BF16/FP16/F32 hidden/weight combinations; deterministic top-k and shared-gate routes. Library handle is hoisted into a module cache (`_router_library()`) so per-launch host cost stays a plain ctypes call (~15 us) instead of re-running `build_qwen35_router(load=True)` (~34 us/call with a pinned session compiler version). |
| MoE grouping and packing | `moe/group_scatter.{hip,py}` | `moe_group_count/prefix/scatter`, `moe_group_compact`, `moe_gather_packed_hidden`, `moe_wmma_tile_map`, `moe_mmq_tile_map` | Stable count/prefix/scatter and compact tile metadata; generic and `w4_paro`. |
| MoE prefill orchestration leaf | `moe/prefill.py` | `moe_prefill` (`w4_paro`) | Registered wrapper composition for selected-expert prefill. |
| Whole selected-expert FFN | `quant/paro_moe_ffn_fused.{hip,py}` | `moe_ffn_selected` (`w4_paro`) | Rotate → gate/up → SiLU → down-rotate → down projection megakernel; primitive chain remains fallback. |
| c1 native dispatcher | `dispatch/moe_c1_dispatch.{hip,py}` | C function-table dispatcher (not a registry layer) | Contracts Python launch overhead while invoking registered/raw function pointers; does not replace component kernels. |
| SiLU/rotation primitives | `fused/paro_silu.{hip,py}` | `silu_mul_dual`, `silu_mul_separate`, `silu_mul_dual_rotate`, `silu_mul_pair_rotate` | Primitive and fused activation/down-rotation boundaries coexist; separate BF16 SiLU permits exact in-place replacement of its gate plane. |
| MoE combine/tail | `fused/paro_combine.{hip,py}` | `weighted_lanes_sum`, `weighted_sum`, `shared_gate_combine`, residual/RMSNorm composites | BF16/FP16/F32 values with FP32 route weights/gates; explicit primitive fallbacks are registered. |
| Paged KV write/copy | `attention/paged_kv_write.{hip,py}` | `paged_kv_write`, `paged_kv_copy` (`bf16`, PARO/GGUF, INT8 layouts) | All attention-visible writes consume complete `KVLiveSpans`; includes BF16 and supported INT8 storage formats. |
| Full/paged attention | `attention/paged_attn_decode.{hip,py}` | `full_attn_decode/prefill`, `paged_attn_decode/prefill`, `full_attn_gate_mul` | Contiguous and paged, batched, GQA, split-K, gated reduce, and supported INT8 KV variants. Per-token/head INT8 includes a row-batched 24Q/4KV/D256 split-K producer plus explicitly strided BF16 gated reducer; the c1 leaf remains registered as its numerical fallback. gfx1151 Qwen3.5-0.8B rows1/8Q/2KV/D256 selects generic split-K3+fused BF16 gate at cap514-641. The private-c1 exact leaf is the fixed256 body at 256 threads (strict exact default) with a parameterized `fixed256_threads_spans` probe at runtime block width; gfx1151 promotes 1024 threads (T2 non-exact, execution-profile gate-passed) via `GGUF_SHORT_C1_BATCH_ATTN_THREADS`. Dense H5120/L64/24Q/4KV/D256 selects the BF16 grouped-GQA split producer from context 4096; shorter contexts and unsupported shapes/backends retain the generic producer. |
| AOTriton adapter | `attention/aotriton_wrap.py`, `attention/aotriton.py` | `full_attn_prefill` (`w4_paro`, `gguf_qwen35`) | Optional library adapter; native raw-pointer paths remain available. |
| Linear-attention Conv | `linear_attn/conv.{hip,py}` | `linear_attn_*conv_decode/prefill`, chain/tree and snapshot composites | Decode, segmented prefill, verifier tree/chain, and state-snapshot variants. |
| Linear-attention GDN | `linear_attn/gdn.{hip,py}` | `linear_attn_prefill_prepare`, `gdn_*recurrent*`, RMSNorm/gate/rotate/cast/snapshot composites | Exact schedules retain FP32 recurrent state. The ordinary segmented lowp owner also has optional strict-order FP32 state-row journals and a direct BF16 handoff for physical multi-request selected commit; its FP32/BF16 outputs, every row state, and final segment states are bit-identical to scalar fused GDN+cast at Qwen geometry. The no-copy decode-order sibling additionally exposes an exact FP32 output tap so immutable initial state and selected row journals can feed FP32 `ssm_out` without in-place state traffic. The gfx1151 Qwen3.8 Q4_K_S production experiment may select FP16 state (FP32 accumulation) through explicit `_fp16state` scalar, chain, segmented, indexed-singleton, compact-peer prefill, and decode-order writers; every supported leaf retains an FP32 fallback. SPECDEC2 P8 resolves the existing FP16 chain row writer through a non-fallback production manifest while retaining consumer-owned dtype-sized rollback snapshots and the strict unfused cast; the complete P9 product gate retains explicit compatibility but promotes no automatic cell, so K0 remains default. gfx1151 Q4 and Q8 `(16K,16V,128,128)` select cluster8; Q4 `(16K,48V,128,128)` selects 1K-chunked compact-peer wave32; all other gfx1151 shapes retain exact nonvolatile LDS32. |
| Runtime state | `runtime/state.{hip,py}` | token embedding, positions/metadata, graph record/commit, scalar state, profiling wall-clock marker | Device-side graph/verify bookkeeping, indexed row state, token publication, and profiling-only steady-clock boundaries. |
| Sampling | `sampling/sampler.{hip,py}` | `sampler`, `mtp_draft_topk` | Greedy/temperature/top-k helpers and bounded draft top-k. |

Compact DMS (C2-7 device port) now has a hip_gfx1100 family in addition to the CPU-reference registrations under `hipengine/kernels/cpu_reference/dms.py` (the `dms_compact_attn_decode` CPU oracle plus INT8 payload encode/decode): `attention/dms_compact.{hip,py}` registers `dms_extract_decision` (`corrected_mask`, bit-exact vs the schema-v1 CPU oracle), `dms_decision_source` (`external_linear_sidecar_v1`, resident BF16 schema-v2 projection for gfx1100/gfx1151 with exact observed retained-candidate decisions), `dms_streaming_pack` (`count_rank_scatter`, bit-exact, chunked 256-token scan for arbitrary prompt lengths), `dms_append_decode` (`compact_append_evict`, bit-exact parent keep-recompute, fail closed on overflow), and `dms_compact_attn_decode` (`grouped_gqa` small-live fallback plus bounded-LDS `grouped_gqa_splitk`; KL ≤ 0.05 / top-1 ≥ 90% vs `compact_attention_reference`, bit-exact at live 0/1). The cpu_reference siblings are the registered strict fallbacks for every key. The kernels are wired into `DMSCompactBackend` behind explicit device-payload selection; the host parent remains the strict fallback and no model package defaults to DMS. The retained W7900 fixture qualification passes the complete 53-test host/device bundle and cached-only rocprof records the original four expected kernel identities with scratch0; the gfx1151 schema-v2 extension adds the external-linear identity at 19.396 us, LDS 1,024 bytes, scratch0, and 16 VGPR; the repaired gfx1151 provisional-shrink fixtures also pass. The exact Qwen3.8 schema-v2 W8192 sidecar is source-disjoint-quality-qualified at 32K/128K. Explicit c1 resident sessions call the schema-v2 GPU projector, direct pack/append, and bounded split-K owner when matching metadata is supplied. Dense BF16 KV remains a temporary correctness-first prefill owner and is released after compact pack; decode has no dense mirror. The production-profile grouped split-K producer handles GQA groups up to eight with one K/V scan per KV head/split while preserving independent query-head outputs; larger groups retain the per-query-head producer and CPU reference is the strict fallback. On a native gfx1151 build, exact 24Q/4KV/D256 group-6 geometry selects the T1 wave-cooperative successor: one wave scores one compact token over 32 four-dimension lanes while sharing each K load across six query heads. gfx1100 and unsupported shapes retain the generic grouped predecessor. Integrated W8192 c1 drains to zero and passes the long-context production category and non-regressive c1 performance gates. Public `LLM.generate()`/real c>N selection, streaming prefill without the temporary dense peak, sampled memory controls, and long soak remain open. Dense paging therefore remains the general product default.

Qwen3.8-27B Q4_K_M is the independent 16K/48V/128x128 gfx1151 GDN
exception. `chain_compact_peer_wave32` materializes normalized Q/K once per K
head and carries one recurrent state across at most 1,024 rows per launch. The
strict route stores that state as FP32; the Q4_K_S R2 production experiment may
store it as FP16 while retaining FP32 register accumulation and an FP32 strict
fallback. Packed c>N prefill uses the registered compact normalized-segments
sibling: one wave32 keeps four state rows per lane in FP32 registers for each
indexed slot and performs only the declared state-store conversion at a packed
chunk boundary. The FP32 compact sibling is the registered strict-storage
fallback; the older decode-order segmented writer remains a generic diagnostic
fallback, not the Q4_K_S compact-peer production owner. The complete c>N
numerical/dynamic/isolation hard gate passes. The serving screen is also
non-regressive (+0.38% c1 and +1.33% exact c8 throughput); both modes share an
absolute ITL-p99 SLO failure. With the fixed percentage threshold removed,
FP16 is the validated gfx1151 Q4_K_S default with FP32 env rollback;
runtime-manifest, BF16-relative/task, and gfx1100 named-profile packets remain
unavailable.
Prepare and RMSNorm still cover the complete prefill once. This chunk is
required because unchunked 4K loses 8.26% to direct LDS32, while the
repaired route is peer-bit-exact and wins the production complete chain
1.517x/1.479x/1.422x at 512/1K/4K. Scalar-exact output/state deltas are bounded
at 0.001953125/2.24e-8. Integrated pp512 improves 316.258 to 330.069
tok/s and drops 24 MiB; 512/1K/4K peak falls 24/128/128 MiB. rocprof confirms
48/192 compact recurrence launches at pp512/pp4K (local128, 40 VGPR, zero LDS
or scratch). Exact direct LDS32 remains the explicit rollback. Evidence:
[`Qwen3.8 compact-peer GDN`](../benchmarks/results/2026-08-15-gfx1151-qwen38-27b-p3-compact-peer-gdn.json).

The independent gfx1100 Qwen3.8 physical C5-C8 row-state path selects
`decode_order_bf16_segments_state_rows_no_copy_wave_reduce` only for FP32
state, K128/V128, and three- or four-token segments. It reconstructs the
parent local128 Q/K and norm reduction trees with wave32 shuffles, preserving
BF16 output and every FP32 state row bit-for-bit while removing redundant
workgroup barriers. Strict, FP16-state, peer, non-physical, and shape misses
retain the registered parent; `HIPENGINE_GGUF_GDN_STATE_ROWS_WAVE_REDUCE=0`
is the same-build rollback. Evidence:
[`gfx1100 segmented GDN wave reduction`](../benchmarks/results/2026-09-01-w7900-q4km-k3-c5c8-segmented-gdn-wave-reduce-retained.json).

### GGUF / Qwen / Laguna path

GGUF is not a PARO alias. Raw GGML blocks, pack8/T16/qmicro/X8 replacement layouts, exact expanded planes, and source-F16 Laguna tensors have distinct storage and registry keys.

#### GGUF projection and quant families

| Quant/layout family | Source / wrapper | Principal registry layers | Stable notes |
| --- | --- | --- | --- |
| Q4_K selected FFN megakernel | `quant/gguf_q4_k_moe_ffn_fused.{hip,py}` | `moe_ffn_selected` (`gguf_q4_k`) | Whole selected gate/up → SiLU → down projection; primitive selected projections remain fallback. |
| Raw Q5_K/Q6_K/Q8_0 | `quant/gguf_k_gemv.{hip,py}` | `linear`, `linear_pair`, `attention_projection_quad` | Decode/prefill, BF16/F32 output, pair/quad launch contractions, rowbatch/coltile variants. |
| Raw Q3_K selected | `quant/gguf_q3_k_gemv.{hip,py}` | `moe_linear` | Q3 selected-expert projection family. |
| Q4_K pack8/raw | `quant/gguf_q4_k_gemv.{hip,py}` | `linear`, `linear_pair`, `linear_pair_silu`, `linear+residual` | Raw GGUF math and lossless pack8 layouts; pair/SiLU and exact rounded-BF16 residual composites where registered. Primitive projection+add fallbacks remain available. |
| Q4_K/Q6_K prefill WMMA | `quant/gguf_q4_k_prefill.{hip,py}` | `linear` | Resident pack8/raw prefill consumers; exact scalar/pack8 routes remain fallbacks. The p512 pack8-Q4 rounded-residual output-store sibling is rejected (0.958x core / 0.952x public complete-model prefill) and is not registered. |
| Q8_0 T16 prefill | `quant/gguf_q8_0_t16_prefill.{hip,py}` | `linear`, `linear_pair` | WMMA/T16 Q8 prefill and architecture-specific wave schedules. gfx1151 rows512/K1024/N16+N16 alpha/beta uses the exact two-wave dual owner; singleton WMMA remains the fallback. |
| Q4/Q5/Q6 T16 selected | `quant/gguf_t16_selected_gemv.{hip,py}`, `quant/gguf_k_t16_selected_prefill.{hip,py}` | `linear`, `linear_pair_silu`, `moe_linear`, `moe_linear+weighted_sum`, `linear+residual` | c=1 and selected-prefill T16/qmicro/interleaved consumers, including weighted/residual composites. gfx1151 Qwen3.8 standard-Q4 physical rows6/8/9/12/16 use the strict one-wave/one-16-row-tile WMMA owner for K/N 5120/6144, 5120/10240, 5120/12288, 5120/17408, 6144/5120, and 17408/5120; production-profile target verification may select the scoped T2 singleton/pair rowtiles at C2/K3 R8 or C3/K1-K3 R6/R9/R12 after their independent D24 numerical gates, while narrow K5120/N1024 and misses retain shared-B and the strict one-wave/shared-B variants remain registered fallbacks. gfx1100 dense-H5120 physical rows6 uses the C1-equivalent rowtile for K/N 5120/1024, 5120/6144, 5120/10240, 5120/12288, and 17408/5120, plus the exact single-wave parent for 5120/17408. The exact four-wave/32-column extension is rejected and removed: it is bit-exact but loses six of seven actual Q4 role shapes (0.759–0.989x), while FFN-gate is noise-flat 1.0046x with 11/40 wins. The retained grouped-rows6 sibling keeps the WG64/two-wave block unchanged but maps consecutive physical R6 chunks onto `grid.y`; all 21 actual Q4 role×R24/R30/R36 cells are BF16-bit exact and improve HIP-event time 1.066–2.549x (minimum 29/30 wins). Its complete C5-C8 gate improves every width/category/heldout scope by 1.15–2.89%; same-build C8 tracing reduces Q4 launches 1,152→720 and Q4 wall 83.555→79.608 ms. `HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2_GROUPED_ROWS6=0` keeps repeated R6 rollback. The exact physical-R24 grouped-R8 alternative is rejected and removed: all seven actual Qwen3.8 Q4 roles improve 1.039-1.105x and same-build C8 target wall improves 174.277→172.119 ms, but the counterbalanced product gate has reverse-order losses in every C7/C8 prompt and pooled C7 code regresses 0.17%; grouped R6 remains optimal. The retained pair-seam extension then consolidates the remaining equal-width physical fallback into two full grouped projections: complete C5-C8 improves 0.72–3.53%, while same-build C8 tracing removes all 576 remaining repeated calls, cuts target launches 2,266→1,832, Q4 wall 79.794→74.394 ms, and target wall 188.588→178.704 ms. `HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2_GROUPED_PAIR_ROWS6=0` is its explicit repeated-R6 rollback. The attempted R24/R30 grouped dual+SiLU compositions are rejected and removed. A WG128/four-wave layout lost 1.1-4.9% in all six actual layer×row cells. Mapping the retained exact WG64 dual R6 block over `grid.y` then improved those isolated leaves 1.013-1.035x, but same-build C8 tracing worsened operation-complete Q4 wall 75.826→77.222 ms (+1.84%) and target wall 173.633→174.305 ms (+0.39%) despite reducing target launches 1,560→1,427; the retained grouped projections plus standalone SiLU remain optimal. The exact permlanex16+DPP transfer to those grouped K5120/N17408 gate/up projections is also rejected and removed: all 18 actual layer×gate/up×R24/R30/R36 cells lose at 0.959-0.998x (minimum 1/30 wins), so the shuffle reduction remains optimal. The exact two-wave/16-column sibling is now the gfx1100 physical-wrapper default after correcting an unrouted first attempt: its initial standard-shape gate improved every C5-C8 width/category/heldout scope by 1.00–2.14%. Recurrent-QKV K5120/N10240 is also qualified: marker-scoped C8 evidence transfers the predicted 128 calls/cycle to **853.3 candidate / 170.7 remaining-parent**, reduces same-commit Q4-family wall 101.785→101.202 ms/cycle and target device-union 192.867→192.173 ms, while its complete C5-C8 gate improves every width/category/heldout scope (aggregate +0.46–1.08%). `HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2=0` retains the WG32 strict parent. K17408/N5120 FFN-down is also qualified after correcting an earlier provenance error from an aborted screen: the valid leaf is exact and 1.033–1.054x faster, marker evidence transfers the final 170.7 WG32 calls/cycle, and its complete C5-C8 gate improves every width/category/heldout slice by 0.66–1.43%. Outside the row6 precedence, tracked-clean counterbalanced W7900 evidence gives single-wave K/N 5120/17408 and 5120/10240 through row128, and 5120/12288 through its shape-specific row112 cap. gfx1100 K/N 17408/5120 rows33-192 use the exact four-wave shared-B row64 sibling (one 16-row tile/wave); row193+ and misses keep the separately registered 256-row strict fallback; all other rows/shapes keep their explicitly registered owners. gfx1100 dense H5120 gate/up (K5120/N17408) fuses the dual+SiLU prefill owner from rows33: the fused kernel is bit-identical to the two-singleton+`silu_mul` chain at that shape for rows 45/96/192/511/512 (and on the small fixture from rows 2), and W7900 Qwen3.8-27B-Q4_K_M measured **+4.2%/+4.2%/+4.9%** prefill at 45/96/192 rows with 512 unchanged; rows<=8 keep their dedicated small-B rowtile/GEMV owners and the unfused chain remains the registered fallback. Exact row48 (three active waves), row64, and row128 fused siblings are additionally registered: row48 is the physical-R36 default after its complete gate improved C8 3.13% in both orders and every C8 category/heldout slice by 2.81–3.27%; `HIPENGINE_GGUF_Q4_T16_DUAL_SILU_ROW48=0` keeps row64 rollback. Actual-weight row64/row128 results are bit-identical and select row64 at rows33-64 (**1.517-1.611x** over the parent on measured production/crossover rows) and row128 at rows65-128 (**1.454-1.492x**); row129 immediately crosses back to the 256-row parent. They are the gfx1100 default after the complete counterbalanced category+heldout gate passed; `HIPENGINE_GGUF_Q4_T16_DUAL_SILU_RETILE=0` restores the 256-row parent on the same build. Exact C2/K3 R8 now uses the existing Q4/Q5/Q6 rows8 rowtiles instead of padding to R12 after a counterbalanced full-suite gate improved 42.350→51.769 tok/s (+22.24%); every other width keeps the rows6-multiple fallback, and `HIPENGINE_GGUF_SPECDEC2_EXACT_TARGET_ROWS=0` restores padded R12. A Q4-only R24/R30/R36 singleton-projection composition using mixed rows8/rows6 leaves is rejected and removed: isolated actual-weight leaves won, but the binding counterbalanced C5-C8 suite regressed C5-C7 by 0.23-0.56% and was noise-flat at C8. The distinct fused dual+SiLU rowtile composition is also rejected and removed: its R36 8/8/8/8/4 actual-weight leaf was 1.487x faster than fused WMMA and passed the C8 full-logit T2 gate, but launch multiplication regressed the complete C8 product route by 3.63% in both process orders; fused WMMA remains the owner. Planar Q6 now defaults to the same mixed partitions only at actual recurrent-QKV, full-attention-V, and FFN-down shapes after 12/12 exact actual-weight cells improved 1.079-1.234x and the complete counterbalanced C5-C8 gate improved every width/category/heldout scope; R18 and shape misses retain repeated R6, and `HIPENGINE_GGUF_SPECDEC2_Q6_MIXED_TARGET_ROWTILES=0` restores that fallback. The retained grouped-grid siblings preserve the mixed route's R8 DPP and R6 shuffle block bodies while consolidating identical chunks: all nine actual role×R24/R30/R36 cells are BF16-bit exact and 1.108-2.002x faster (minimum 28/30 wins). The complete C5-C8 gate improves every width/category/heldout slice by 0.70-1.91% (aggregate 1.18-1.75%); same-build C8 tracing reduces BF16 Q6 launches 192→64, Q6 wall 29.669→25.693 ms, and target wall 178.464→174.248 ms. `HIPENGINE_GGUF_Q6_T16_GROUPED_TARGET_ROWTILES=0` retains the prior mixed launch sequence. The analogous Q5 R24/R30/R36 mixed-row composition is rejected and removed: exact actual-weight leaves improved 1.085-1.159x, but the binding counterbalanced suite regressed C5/C6 by 0.07%/0.38%. The retained grouped-grid sibling instead preserves each existing R6 block verbatim while mapping independent chunks to `grid.y`; the actual recurrent-output weight is BF16-bit exact and 1.184-1.257x faster across R18/R24/R30/R36 (30/30 paired wins). Its complete C5-C8 gate improves every width/category/heldout slice by 0.48-1.77% (aggregate 0.95-1.49%); same-build C8 tracing reduces Q5 launches 192→48, Q5 wall 15.448→13.912 ms, and target wall 178.457→178.044 ms. `HIPENGINE_GGUF_Q5_T16_GROUPED_TARGET_ROWS6=0` keeps repeated-R6 rollback. The floor stops at 33 because this same shared FFN stage also serves captured target verification in 16/32-row physical groups (C4-C8 at K3), where the fused owner measured +11.9% wall over 7/7 same-cycle prompts; at rows33 MTP C8 instead improved 2.6% on the same 7/7 prompts. The same gfx1151 model's Q5 K6144/N5120, K17408/N5120, and K5120/N10240 rows2-8 use the exact col8 rowtile; packed target verification explicitly admits the measured K6144/N5120 recurrent-output shape. Registered parents remain strict fallbacks. |
| Q6/Q4 mixed and narrow K/V grids | `fused/gguf_q6_q4_pair.{hip,py}` | `linear_pair` (standard-Q6+Q4, Q4, Q4+planar-Q6) | Exact block-parallel rows1 pairs; gfx1151 qualifies Qwen3.8 recurrent K5120/N10240+N6144 and full-attention K/V K5120/N1024+N1024 while primitive projections remain fallbacks. |
| Dense Q6_K T16/qmicro | `quant/gguf_q6_k_t16_gemv.{hip,py}` | `linear`, `linear+argmax`, `linear+residual` | Exact dense Q6 decode/prefill/root families. gfx1100 planar row8 uses the exact DPP reduction (VGPR136→112, bpermute320→0), admitted on all 55 actual-operation rows and retained by a 1.634% complete-owner wall win; rows1-7 keep the generic reduction. The exact R6 DPP analogue is rejected: recurrent-QKV and FFN-down actual-weight leaves are 0.953x/0.967x, while full-V is only 1.005x. gfx1100 planar K17408/N5120 prompt prefill uses the exact four-wave row64 owner at rows33-128 and the existing four-wave shared256 owner at rows129-511; rows<=32, rows>=512, shape misses, the non-WMMA physical verifier, and peer backends retain their prior owners, with the one-wave primitive separately registered. gfx1151 packed target verification routes only actual Q5 K6144/N5120, standard Q6 K5120/N10240, and planar-qmicro Q6 K5120/N1024 plus K17408/N5120 at rows2-8 through their exact col8 rowtiles; production-profile C3 additionally composes the same row-independent owners at physical R9/R12. The verifier-local capability does not broaden the global native-batch scope or peer-backend ownership. gfx1151 rows>=512 uses 128-thread/four-wave shared-weight WMMA for standard K5120/N10240 QKV (2.96-3.55x) and planar K17408/N5120 FFN-down (1.42-1.50x); both use 24 KiB LDS / 248 VGPR. Other non-verifier rows<512, root paths, capability misses, and peer backends retain exact one-wave/16x16 primitives. Qwen3.8-27B `Q4_K_M` production NextN now defaults to a packaged, model-bound 131,072-row CJK-aware planar-Q6 proposal head: arbitrary selected rows are repacked into compact T16 storage, physical batches score only that head, and `proposal_top1_mapped_bf16`/the mapped batch reduction return full token IDs on device. The complete counterbalanced C5-C8 gate improves 3.59-4.49%, cuts proposal wall 29.19-33.44%, preserves acceptance and all 520 cross-arm target rows, and wins all 80 prompt cells in each pair. Strict, capability/identity misses, and `HIPENGINE_GGUF_MTP_HOT_VOCAB=0` retain the full-vocabulary exact head; malformed or model-mismatched explicit maps fail closed. |
| IQ2/IQ3/IQ4 decode | `quant/gguf_iq_gemv.{hip,py}` | `moe_linear` | Raw IQ selected-expert projection families. IQ3 tile4 remains scoped to the retained gfx1100 explicit-DFlash route; gfx1151 excludes it after a complete-route rejection and keeps tile1. |
| IQ selected prefill | `quant/gguf_iq_selected_prefill.{hip,py}` | `moe_linear` | Grouped/expert-major, active-expert, rowbatch, and output-ownership variants. |
| Raw-K activation MMQ | `quant/gguf_k_mmq_prefill.{hip,py}` | `activation_quant`, `linear` | Q8_1 producer layouts plus Q5/Q6 MMQ consumers; retained diagnostics may not be runtime defaults. |
| Raw-IQ source MMQ | `quant/gguf_iq_source_mmq_prefill.{hip,py}` | `moe_linear` | Source-faithful IQ MMQ diagnostic/alternative consumers. |
| Exact expanded F32 planes | `quant/gguf_q5_k_f32_rocblas_prefill.{hip,py}` | `linear` and raw-quant composites | Raw Q5/Q6 producers plus ordered exact consumers; library SGEMM variants are distinct diagnostic paths. |
| Source-F16 Q4/Q5/Q6 library route | `quant/gguf_q6_k_f16_rocblas_prefill.{hip,py}` | dequant/cast/`linear` composites | Sole Q4T16/Q5T16 and raw/sole-planar-Q6T16 bounded tile producers feeding F16 rocBLAS; Q4T16 includes scalar column-owned and exact adjacent-pair-owned producers, while Q5T16 includes scalar plus exact pair- and natural-octet-owned producer leaves. Changed arithmetic is model/shape gated, while exact T16 remains the small-row and miss fallback. Qwen3.6-27B admits Q5T16 recurrent output with its natural-octet producer at M512-M4096, bounded Q4T16 full-attention Q with its adjacent-pair producer at M512-M2047, and Q4T16 linear-attention gate only as the second operand behind the already-admitted Q6T16 QKV peer at M512-M2047 after complete category and cross-board full-engine qualification. A generic ordered-pair policy prevents that gate shape from claiming standalone or Q4/Q4 pair dispatch, while request-row filtering and a per-shape ceiling keep M2048/4K exact. Scalar producer and exact T16 kernels remain registered policy-miss fallbacks; decode, verifier, peer backends, and every unqualified shape remain exact. |
| Embedding | `quant/gguf_q6_k_embedding.{hip,py}` | `embedding` (`gguf_q4_k/q5_k/q6_k/q8_0`) | Raw GGUF row lookup for root/token tables. |
| X8 sidecars/replacements | `quant/gguf_x8_selected_gemv.{hip,py}` and pack8 modules | selected `moe_linear` / top-1 helpers | GGML-style packed selected-expert and head diagnostics/qualified lanes. |
| Q8 dp4a verifier | `quant/gguf_q8_0_dp4a_gemv.{hip,py}` | `linear` pair/triple/rowtile variants | q8_1+sudot4 verifier/draft families; selection is route-specific. The Q6 X8 direct-top1 consumer is c1-only for shared-slot AR; multi-row uses Q6 rowtile logits plus GPU argmax after the physical-cN shortcut emitted an invalid second-row sentinel. |
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
On gfx1100 dense H5120 `Q4_K_M` that floor is **M>=16** after the 2026-08-30
re-qualification: bit-identical to the two singletons at the dispatched shape for
rows 16/24/32/45/96/512 and +1.7-2.1% W7900 Qwen3.8 prefill across rows 16-192.
The route is ContextVar-scoped to the resident prefill entry, so this differs from
the shared `linear_pair_silu` gate and cannot reach captured target verification.
Raw/pack8 Q4 bodies remain registered for other layouts and diagnostics, not as
dense-27B sidecars. Evidence: [`XTX first fit`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-sole-t16-first-fit.json),
[`output-major LDS keep`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-q4-t16-output-major-lds.json),
[`dual-WMMA SiLU keep`](../benchmarks/results/2026-08-13-qwen36-27b-q4-dual-wmma-silu-prefill-retained.json),
[`unequal Q4 pair keep`](../benchmarks/results/2026-08-13-qwen36-27b-q4-unequal-dual-prefill-retained.json),
and [`live residency/correctness`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-correctness-residency.json).

For dense Qwen3.8-27B Q4_K_M on gfx1151, the capability-driven H=5,120
plan is qualified as sole Q4 ownership: all 288 rank-2 Q4 tensors use only
`gguf_q4_k_t16_v1/tiles`; pack8, decode-tile, raw, and alternate-Q4 sidecars are
absent. The serial rows1 H=5,120/N=17,408 gate/up pair defaults to the exact
local32 dual+SiLU owner. Its former changed-arithmetic Q8_1x2 split-weight route
passes the strict-teacher gate but loses the current seven-pair ZBook timing at
`0.998071x` with one win, so it remains diagnostic. Native B1 retains its
separately qualified non-split Q8_1x2 owner. Native verifier rows2-4 use the
exact standard-Q4 two-wave/16-column owner only for full-attention Q
K5,120/N12,288 (rows2/3/4) and recurrent QKV K5,120/N10,240 (rows3/4). Each
wave preserves the parent WG32/eight-column K/FMA/reduction/store sequence;
the parent stays the registered strict fallback for every shape/row miss.

The selected Q4_K_S representation independently replaces only its 128
H=5,120/N=17,408 gate/up weights with
`gguf_q4_k_qmicro_t16_v1/tiles`; its other rank-2 Q4 weights remain standard
T16. Each qmicro K256/N16 tile is 2,304 rather than 2,368 bytes, removing 170
MiB. Serial c1 uses the exact split-weight Q8_1x2 owner over compact metadata;
native rows2-4 use the exact shared-weight rowtile8 sibling because direct-BF16
association can change greedy trajectories. Bulk 512/1K uses direct-metadata
WMMA; from 4K, one bounded expansion writes only compact coefficients into the
already-dead FFN scratch plane before the same exact dual WMMA. This adds no
workspace or persistent bytes. Rows5-4095 and misses retain the qualified
singleton/primitive fallbacks. The raw token embedding remains raw GGUF, peer
geometries retain prior policy, and no `KVLiveSpans` ABI changes are involved.
Evidence:
[`current Q4_K_M strict requalification`](../benchmarks/results/2026-08-16-gfx1151-qwen38-dense-pair-requalification.json),
[`current Q4_K_M counterbalanced A/B`](../benchmarks/results/2026-08-16-gfx1151-qwen38-dense-pair-strict-default.json),
[`Q4_K_S split-weight decode`](../benchmarks/results/2026-08-15-gfx1151-qwen38-27b-q4-q8x2-split-weight-decode.json), and
[`Q4_K_S sole qmicro gate/up`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-qmicro-sole-retained.json).
The serial-c1 K=5,120/N=1,024 full-attention K/V subset independently selects
the exact four-column Q4T16 owner; native sessions, peers, and all shape misses
retain local32 direct. Evidence:
[`Qwen3.8 Q4T16 c1 col4 full-K/V`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4-single-col4-c1-decode.json).
The same gfx1151 model policy also gives the 48 exact
K=6,144/N=5,120 recurrent `ssm_out` Q5_K tensors one sole
`gguf_q5_k_t16_v1/tiles` payload each. Serial c1 uses the exact eight-column
output-ownership sibling after five actual layers and every repeated/natural AR
scope improve with BF16-bit identity; the registered local128 direct owner
remains the policy-miss and `native_batch_decode_session` fallback. Exact
rows-2-4 rowtile, rows-5+ direct fallback, and dense WMMA consumers cover the
rest of the role; GDN and residual boundaries remain separate registered
primitives. Dense BF16 stays available as a numerical oracle but is not a
resident shadow for this qualified shape. The smaller 0.8B Q5T16 role remains
independently shape-qualified. Evidence:
[`Qwen3.8 Q5T16 serial-c1 tile8`](../benchmarks/results/2026-08-15-gfx1151-qwen38-27b-q5-dense-tile8-decode.json).

Qwen3.8-27B Q4_K_S is qualified on gfx1151 using the same operation-complete
Q5T16 family. Its 60 rank-2 Q5 owners consist of the 48 existing recurrent
outputs plus eight K17,408/N5,120 FFN-down, three K5,120/N10,240 recurrent-QKV,
and one K5,120/N1,024 full-attention-V tensor; all own only `tiles`, with no
dense-BF16 Q5 shadow. The exact 16K/48V/128x128 GDN geometry also selects
`chain_compact_peer_wave32`. True AR independently transfers three exact
Q4_K_M-derived policies whose representation and math are unchanged: qmicro Q4
split-weight gate+up+SiLU, Q4 down+residual (Q5 down remains unfused), and
quant-independent fixed-H5120 norms. The transfers improve matched 512/128 AR
**12.42932 -> 13.06854 tok/s (+5.143%)**. Clean commit `3118943eb` publishes
**13.03883/12.86679/13.02544 tok/s** at 512/1K/4K, above both frozen clean
llama backends at every shape. Natural true AR is **13.33276 tok/s**,
repeat-exact across 30 requests.
Native Q4_K_S MTP uses a separately qualified rows2-8 qmicro Q8_1x2
rowtile8 owner for H5120/N17408 gate/up+SiLU. It shares each compact-weight
traversal across rows while preserving c1's dp4a/FMA/reduction and BF16
association independently per row. Rows2-8 are BF16-bit exact to serial c1;
the complete ten-prompt AR/B1/B2/B3 gate is exact with GPU/CPU acceptance
agreement, and B3 reaches **24.19347 tok/s / 1.8228x** own AR. Cache-only
`rocprofv3` records rows3 at local128, 120 VGPR, 512-byte LDS, zero scratch,
and 0.462-0.465 ms on an actual layer-0 pair. The policy adds no bytes, and the
direct-BF16 rowtile plus primitive chain remain registered fallbacks. On
2026-08-18 the owner was extended to ROW_TILE 5..8 and the packed-AR decode
step (`_enqueue_packed_decode_model_step`) now enters
`native_batch_decode_session(True)`, so eager `step_batch_native` and graph
capture route c2..c8 gate/up through the rowtile8 owner instead of WMMA
prefill (c8 step 408.9 -> 312.4 ms; c4 42.5 tok/s agg 3.31x c1 with no
regression; native_c8 25.2 tok/s agg, rows exact vs c4). On 2026-08-18 the
single Q4/Q5 projections (attn_qkv, attn_q/k/v/o, attn_gate, ffn_down,
ssm_out) were also extended to rows 2..8: `launch_q4_dense_rowtile` (8-col),
`launch_q4_dense_rowtile_col4`, and `launch_q5_dense_rowtile_col4` now
instantiate ROW_TILE 5..8, `_q4_t16_dense_native_dispatch`/`_q4_t16_
sidecar_decode_variants` cover rows 2..8, and gfx1151's
`GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT` for Q5 is 8. c8 WMMA prefill is
eliminated: packed-AR c8 step 312.4 -> 139.8 ms, native_c8 aggregate
25.2 -> 56.6 tok/s (4.40x c1, 7.1/stream), c4 unchanged (42.5 tok/s agg),
RED rows 2..8 bit-exact vs c1. The Q6 lm_head rowtile was also extended to
rows 2..8 (`launch_q6_t16_rowtile`/`_col8` ROW_TILE 5..8), so c8 lm_head is
one launch instead of the prior 4+4 chunk; `GGUF_Q6_LM_HEAD_MAX_CHUNK` is now
8 (was 5/4).

### c=N decode combination map (gfx1151 Qwen3.8 Q4_K_S)

No decode concurrency below 512 silently falls to WMMA prefill:

| rows | Q4 single proj | Q4 gate/up | Q5 single | Q6 lm_head |
| --- | --- | --- | --- | --- |
| 1 | `dense_single_local32` | `dense_dual_local32` | `t16_gemv_decode` (direct) | `t16_gemv_decode` (direct) |
| 2-8 | `dense_rowtile`/`_col4` (gfx1100 qualified Qwen3.8 R6 shapes and gfx1151 qualified small-M shapes use exact `dense_rowtile16_w2`) | `dense_dual_q8_1x2_rowtile8` | `t16_gemv_rowtile` (per-shape cap; qualified gfx1151 Qwen3.8 shapes use col8 ownership) | `t16_gemv_rowtile` |
| 9-511 | rowtile8 chunked (8+2, 8+8, ...) | dual rowtile8 chunked | `t16_gemv_decode` (direct grid.y=rows) | chunked (max 8) |
| >=512 | WMMA prefill (bulk) | WMMA prefill | WMMA prefill | WMMA prefill |

Mechanism: in a `native_batch_decode_session` the single Q4/Q5 projections at
rows 9..511 are decomposed by `_native_rowtile_chunk_groups` into
`_rowtile8_row_chunks` groups (all groups 2..8 rows, tail-1 folded), so each
group lands on the native rowtile owner; the gate only fires for quants that
would resolve to a `t16_wmma_prefill` leaf (Q4) and that have a registered
rowtile owner. Q5 keeps its native direct grid.y=rows leaf (it never hits
WMMA), and rows >= 512 stay on WMMA prefill. The gate/up dual rowtile8 policy
(`GGUF_DENSE_PAIR_SILU_NATIVE_DECODE_POLICIES`) admits rows 2..511. c>8 chunks
into <=8-row rowtile8 groups; c>=512 stays on WMMA. Evidence:
[`Qwen3.8 Q4_K_S qualification checkpoint`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4ks-qualification-checkpoint.json),
[`Qwen3.8 Q4_K_S true-AR policies`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4ks-decode-policies-retained.json),
[`clean Q4_K_S publication`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q4ks-clean-publication.json), and
[`exact Q4_K_S native B3`](../benchmarks/results/2026-08-17-gfx1151-qwen38-27b-q4ks-exact-native-b3.json).

The same Qwen3.8/gfx1151 policy role-qualifies byte-neutral Q6 ownership rather
than forcing the losing all-planar route. The 32 FFN-down tensors, eight narrow
attention-V tensors, and untied root own one
`gguf_q6_k_t16_qmicro_planar_v1/tiles` payload each. The 24 recurrent
K=5,120/N=10,240 QKV tensors retain one `gguf_q6_k_t16_v1/tiles` payload each
because planar c1 loses 8.72% on actual weights; no tensor retains both layouts
and dense-BF16 Q6 bytes are zero. Exact native c1, rows2-4, WMMA, residual,
and top-1 leaves remain registered. On gfx1151, rows2 F32 uses a dedicated
planar col16 owner, while native rows2-4 FFN down+residual deliberately uses
planar projection plus the primitive BF16 add: the exact native fused sibling
loses 17.35%/11.44%/11.15% at rows2/3/4 and remains a
peer-backend/diagnostic leaf. Complete actual-weight,
512/1K/4K, graph, NextN, natural AR/B1-B3, CPU quality, memory, and teardown
gates retain the role-qualified route. Evidence:
[`Qwen3.8 role-qualified Q6`](../benchmarks/results/2026-08-15-gfx1151-qwen38-27b-p2a-role-qualified-q6.json).

Qwen3.8/gfx1151 P4 enables changed-arithmetic source-F16 only for the 48
K=6,144/N=5,120 sole-Q5T16 recurrent outputs at M512-M4096. The byte-exact
octet producer expands four bounded tiles per layer into temporary F16,
zero-workspace rocBLAS publishes BF16, and exact Q5T16 WMMA remains the policy
miss/rollback. Q4 and Q6 source-F16 are explicitly empty after pp512 wall and
memory losses. Q5 improves prefill 3.978%/2.498%/2.650% at 512/1K/4K while
adding 24.375/65/65 MiB temporary peak, no duplicate weight payload, and zero
teardown. All natural tokens/acceptance are identical; every full/train/
heldout/category scope stays within the frozen 0.5% decode guard. Evidence:
[`Qwen3.8 Q5 source-F16`](../benchmarks/results/2026-08-15-gfx1151-qwen38-27b-p4-q5-source-f16.json).
On 2026-08-17 the same admission was extended to the byte-identical Q4_K_S
model (MOSTLY_Q4_K_S added to the dense H5120 F16 policy; its 48 recurrent
outputs are byte-identical to K_M), and the Q4_K_S bulk-prefill scratch row cap
became capacity-conditional: 4K-class requests grow to the natural 4,096-row
full-attention plateau so the source-F16 route stays active there (+2.95%),
while 8K and larger keep 1,024-row chunks (4,096-row chunks measured -2.3%
slower at 8K regardless of source-F16), keeping memory flat past 8K. Evidence:
[`Q4_K_S Q5 source-F16 retention`](../benchmarks/results/2026-08-17-gfx1151-qwen38-27b-q4ks-q5-source-f16-prefill-retention.json).

For dense Qwen3.5-0.8B Q4_K_M on gfx1151, exact role/shape plugin policy also
keeps one compact Q4T16 payload for the six full-attention Q projections at
K=1,024/N=4,096. The existing direct leaf owns c1, exact rowtile owns c2-c4,
physical c8 is split into two exact c4 launches by backend capability, and the
existing T16 WMMA owner handles bulk rows. Every other 0.8B Q4 role and peer
geometry retains its prior residents. No attention kernel or
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

Dense-H5120 has an independent exact c1 K=17,408/N=5,120 policy for its 32
Q4T16 and 32 planar-qmicro-Q6 FFN-down owners. Each same-resident
`linear+residual` sibling freezes the direct projection's K/FMA/reduction tree,
rounds that projection to BF16, then folds only the BF16 residual read/add/final
round into the producer store. The mixed prefill/decode selector treats this
explicit rows1 policy independently of the WMMA-prefill axis, and the T16 ABI
launches from the existing sole `tiles` allocation. A selected-region graph
trace removes exactly **64 launches/token (934 -> 870)** and changes profiled
host decode **82.46295 -> 82.31707 ms/token (-0.177%)** while selected kernel
wall is flat within **0.005%**. Rows>1, Q8, other shapes/models, and peer
backends retain the registered primitive chain; no payload, workspace, or
tracked peak changes. Evidence:
[`Qwen3.8 c1 down-residual graph contraction`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-c1-down-residual.json).

For scalar gfx1151 Q5T16 recurrent-output ownership, the registered
`gdn_recurrent_rmsnorm_gate+cast/gguf_q5_k_t16_v1/bf16_lowp_f32_bf16_out`
producer writes both the unchanged FP32 recurrent output/state and the exact RNE
BF16 handoff consumed by `ssm_out`. It removes one standalone cast per recurrent
layer without changing payload, scratch, or math. Registry misses and other
quants retain ordinary GDN plus explicit cast; the verifier-chain sibling stays
excluded pending its independent MTP gate. Evidence:
[`Qwen3.8 scalar GDN BF16 handoff`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-gdn-bf16-handoff.json).

The same gfx1151 scalar graph independently admits the existing exact
`linear_pair/f32/bf16_hidden_bf16_out` body only at rows1 K5120/N48+N48. One
local256 grid assigns independent alpha/beta output blocks while preserving each
singleton K/FMA/reduction tree. Capability or registry misses, gfx1100, and
rows2-4 retain two singleton dense-F32 projections. gfx1100 physical verifier
rows15/18/21/24 also retain those singletons: the exact flat pair contracted the
family 96→48 launches and 2.057→1.469 ms, but regressed operation-complete target
wall 173.633→173.816 ms and target kernel sum 159.009→160.442 ms. No payload or
scratch is added. Evidence:
[`gfx1151 Qwen3.8 dense-F32 alpha/beta pair`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-dense-f32-alpha-beta-pair.json),
[`gfx1100 high-row rejection`](../benchmarks/results/2026-09-02-w7900-q4km-k3-dense-f32-high-row-pair-rejected.json).

A second gfx1151-only rows1 capability joins those pair blocks to the independent
C10240/K4 in-place Conv blocks under
`linear_attn_alpha_beta+conv_decode/f32/bf16_k5120_n48_c10240_k4_c1`.
The local256 mixed grid preserves both dense-F32 reduction trees and every Conv
state/output bit, needs 32 VGPR, 1 KiB LDS, zero scratch, and adds no bytes.
Capability/registry/shape misses, verifier rows, and peers retain pair plus
ordinary Conv. Evidence:
[`Qwen3.8 serial alpha/beta+Conv`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-alpha-beta-serial-conv.json).

The gfx1151-only serial 24Q/4KV/D256 full-attention route also registers
`split_qgate+head_rmsnorm+partial_rotary/gguf_f32_weight/qwen35_position_qk_bf16_f32`.
One local256 grid reads packed BF16 Q/gate and BF16 K directly, reproduces the
existing FP32 head RMSNorm reduction and partial-RoPE expression bit for bit,
and copies gate BF16 bits. It removes the standalone split and K-cast nodes;
the complete primitive chain remains registered for shape/capability misses,
native rows, prefill, and peers. No payload or scratch is added. Evidence:
[`Qwen3.8 Q/K postprocess contraction`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-qk-postprocess-contraction.json).

The gfx1151 rows1 recurrent QKV/gate boundary independently registers
`linear_pair/gguf_q6_k_t16_v1+gguf_q4_k_t16_v1/mixed_grid_bf16_bf16_out` for
K5120/N10240+N6144. One local128 grid assigns the established Q6 local128
blocks and four independent Q4 local32 waves per Q4 workgroup, preserving both
primitive arithmetic trees without serializing either owner. It removes 24
launches/token, uses 96 VGPR, 512-byte LDS, zero scratch, and adds no payload or
workspace. Native rows, prefill, capability/registry misses, and peers retain
the two primitive projections. Evidence:
[`Qwen3.8 Q6/Q4 mixed grid`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-q6-q4-mixed-grid.json).

The same gfx1151 rows1 family independently registers two narrow full-attention
K/V keys for K5120/N1024+N1024:
`linear_pair/gguf_q4_k_t16_v1/narrow_col4_pair_bf16_bf16_out` and
`linear_pair/gguf_q4_k_t16_v1+gguf_q6_k_t16_qmicro_planar_v1/narrow_col4_planar_pair_bf16_bf16_out`.
Each local128 grid preserves the qualified Q4-col4 K owner and either the
Q4-col4 or planar-qmicro-Q6 V owner. The target's eight Q4/Q4 and eight Q4/Q6
pairs remove 16 launches/token; both kernels use zero scratch and add no
payload or workspace. Native rows/MTP, prefill, NextN, capability/registry
misses, and peer backends retain the two primitives. Evidence:
[`Qwen3.8 narrow K/V pair`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-narrow-kv-pair.json).

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

Dense-H5120 has an independently qualified pair under variants
`bf16_out_fixed5120_wave256`. Each local256 thread caches 20 values, preserves
the generic per-thread accumulation and complete FP32 reduction tree, replaces
nine tree barriers with shared-partial and inverse-RMS publication barriers
plus five wave32 exchanges, and reuses the cached values for output. On gfx1151 Qwen3.8-27B,
all 128 actual-weight norm/residual outputs are BF16-bit exact; the package
improves **1.23268 -> 0.35870 ms/token (3.4365x, 15/15)** and complete graph AR
**1.37-1.46%** across 512/1K/4K. `rocprofv3` records local256/grid256, 56/80
VGPR, 1,536-byte LDS, and zero scratch. The model/backend/shape capability is
rows1 Q4_K_M only; generic kernels remain the rows>1, Q8, output-norm,
other-model, and peer-backend fallbacks. Evidence:
[`dense-H5120 norm route`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-fixed5120-norm-decode.json).

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
| DFlash commit/state | `speculative/dflash_commit.{hip,py}` | `dflash_commit_chain`, `linear_state_pair_*` | Transactional selected-state and cursor commit helpers. The strict chunked pointer-table pair-copy is retained for gfx1100 physical C5-C8 and gfx1151 packed-state transfer; gfx1100 keeps the per-layer HIP D2D chain as an explicit rollback. |
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

## MTP verify profiling traps (W7900, measured 2026-08-30)

Four invocations failed before producing data; each is a property of the tool, not of the
measurement, so record it instead of rediscovering it.

1. **The self-contained wrappers cannot run on this host.** `gguf_mtp_verifier_rocprof.py`,
   `gguf_decode_rocprof.py`, `gguf_packed_ar_rocprof.py`, `gguf_mtp_draft_rocprof.py`,
   `gguf_continuous_owner_rocprof.py`, `gguf_sh_c0_profile.py`, `qwen35_rocprof_audit.py` and
   `mtp_verifier_rocprof.py` all call `_prepare_roctx_override`, which raises unless a pip
   ROCm SDK `librocprofiler-sdk-roctx.so.1` exists under `site-packages/_rocm_sdk_*/lib`.
   **That blocker was wrong and is retracted.** The sentence above came from a `find`
   over the project venv, `/opt/rocm*` and `/usr/lib`, which misses where this host keeps
   the SDK: **12 copies exist**, all under `~/mambaforge/envs/*/lib/python3.12/site-packages/`
   in `_rocm_sdk_core/lib` and `_rocm_sdk_devel/lib` (including `.so.1.3.2`), in the
   `therock` and `vllm` envs. The default only probes `sys.prefix`, and the wrappers are run
   with `.venv/bin/python`, whose prefix has no `_rocm_sdk_core` at all - while
   `shutil.which('rocprofv3')` is `/home/lhl/mambaforge/envs/therock/bin/rocprofv3`, so the
   matching library sits in the very env that provides the profiler. Passing it works in the
   sense that `_prepare_roctx_override(<therock _rocm_sdk_core path>)` returns an override
   directory instead of raising (verified in plain Python 2026-08-30, no GPU); what is *not*
   yet verified is that marker tracing then succeeds end to end under `rocprofv3`. The
   permanent fix is for `_default_roctx_sdk` to also probe the prefix of `which(rocprofv3)`.
   **Done in one wrapper on 2026-08-30** (commit 7716ccf87):
   `gguf_packed_ar_rocprof.py` globs
   `lib/python3*/site-packages/_rocm_sdk_{core,devel}/lib/librocprofiler-sdk-roctx.so*` under the
   `which(rocprofv3)` prefix, newest python first, then falls back to legacy `libroctx64`. On this
   host it resolves to the therock copy matching `rocprofv3 1.3.2` with no flag passed. The other
   seven wrappers still carry the old copy - see `docs/REFACTOR.md`.
   Kernel tracing needs no ROCTX shim - only markers do - so the
   working route is the wrapper's own `--child` mode driven under a direct
   `rocprofv3 --kernel-trace`, rolled up with `gguf_kernel_trace_rollup.py TRACE_DIR`.
2. **`rocprofv3` flags are not what they look like.** `-i/--input` is an input file and
   `-d/--output_dir` is the directory; `-o/--output_file` is the name prefix and the format is
   `-F`. Passing `-i <dir> -d csv` fails with "does not have a recognized extension" and then
   writes into a stray `csv/` directory.
3. **Defaults profile the wrong path.** `mtp_verifier_rocprof.py` defaults to
   `--model /models/hipengine/Qwen3.6-35B-A3B-PARO-...-MTP-BF16`, a safetensors model, so it
   cannot describe the GGUF path at all. `gguf_mtp_verifier_rocprof.py --mode` defaults to
   `serial-step`, the historical verifier, not the native block verifier the resident route
   reaches. Either default silently measures a path we do not ship.
   Worse, the *GGUF* wrapper is not safe either: its baseline child is
   `scripts/mtp_chain_e2e_smoke.py`, which mentions GGUF zero times and is
   safetensors-only, so `--model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf` is accepted by
   argparse and then dies in `_run_ar_baseline` with
   `MissingConfigError: config.json not found under /models/gguf/...`
   (`mtp_chain_e2e_smoke.py:1768`). Four K1-K4 budget arms were lost to this on
   2026-08-30; all four logs differ only in the budget string. That wrapper needs a
   GGUF baseline child before any GGUF verify claim can be profiled.

4. **`--mode block-verify` rejects its own padding.** With `--block-rows` beyond the prompt
   tail the child raises `ValueError: token_id 2147483647 outside [0, 248320)` from
   `qwen35_gguf_runner.py:19000`, so sweeping rows (the direct way to separate launch
   overhead from real work, since K3 is 4 rows per lane and C8 is 32) needs the padding
   contract fixed first - see `docs/REFACTOR.md`.
5. **A profiler-injected tree poisons every compiler probe it spawns.** Under `rocprofv3`, the
   injected libraries and `ROCPROF_*=1` environment are inherited by children, so even the JIT
   path's `clang++ --version` probe loads the profiler, deadlocks in `futex_wait` on a control
   channel whose owner is gone, and never exits. Measured on this host 2026-08-30: **77 orphaned
   `clang++ --version` processes**, ages 11 to 46 days, states `SN`/`S<`, ppid 1 (about 1.7 per day
   over 46 days). Six sampled across that range were identical: 15 profiler libraries in
   `/proc/<pid>/maps`, `ROCPROF_KERNEL_TRACE=1` in `/proc/<pid>/environ`, `wchan=futex_wait`; one
   also held an `anon_inode:kfd_smi_ev` fd, which is why `rocm-smi --showpids` lists a `clang++` as
   a GPU process. The parents are gone because they were killed mid-profile - an interrupted
   profile is what strands the child. Consequences: a live waiter on that probe hangs forever with
   the GPU idle, which looks exactly like the stale-JIT-cache symptom above and is a second,
   different cause of it; and the leftovers pollute the SMI pid table. The AGENTS rule - prebuild
   the `.so` and pass a precomputed compiler-version file with `require_cached` instead of letting
   the profiled process spawn `hipcc`/`clang` - is what avoids it. Treat that as a **pair, not an
   option**: measured again on 2026-08-30, `HIPENGINE_REQUIRE_CACHED_BUILD=1` on its own still lets
   the builder run `hipcc --version` to compute the cache key, so the run deadlocks anyway with the
   parent blocked in `wchan=anon_pipe_read` and the GPU idle - `HIPENGINE_COMPILER_VERSION_FILE` (or
   an explicit `compiler_version`) is what removes the probe. In this symptom the *parent* waits in
   `anon_pipe_read` rather than `futex_wait`. A permanent fix is for the probe
   in the JIT build path to refuse to spawn when `ROCPROF_*`/`ROCT_*` is in the environment and use
   the cached file instead. Stranding is host-wide, not MTP-specific, so clean up outside a
   profiling session rather than during one.

6. **The verify wrapper does not accept `--backend`, and callers pass it.** All four
   `b1..b4` arms in one sweep died before reaching the GPU with
   `gguf_mtp_verifier_rocprof.py: error: unrecognized arguments: --backend hip_gfx1100` - the
   wrapper's parser takes `--mode`, `--roctx-sdk`, `--out`, `--top` and friends, but not
   `--backend`. The four logs are byte-identical at 1907 bytes, so a whole budget ladder can
   vanish with nothing but a usage error in it. It is single-backend by construction, so the
   durable fix is either to accept and ignore the flag or to reject it with a message that names
   the accepted flags; meanwhile do not pass it from a driver script, and treat an argparse exit
   code 2 in a sweep log as "every arm in this file failed", never as a null result.

7. **`block-verify` cannot run on a Q4_K_M repack without one specific env, and a wrapper flag
   implies otherwise.** On a `q5_k_t16_v1` repack the verify path picks F32 activations unless
   `use_prefill_gdn_capture` or `prefill_score_ready` (`qwen35_gguf_runner.py:8000`); with F32 both
   dense-Q8 escapes require `quant_key == "gguf_q8_0_t16_v1"` (`_dense_q8_raw_ptr`, :12710), so they
   are structurally unavailable and the plain fallback raises
   `unsupported GGUF linear dispatch: layout='gguf_q5_k_t16_v1', activation='f32', output='bf16'`
   (`gguf_linear.py:2181`). Four arms died identically that way. `--block-wmma-prefill` alone does
   not clear it; `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1` does, by selecting bf16 activations.
   Do not substitute `--verify-dense-q8-dp4a-f32`: nothing under `hipengine/` reads a
   `verify_dense_q8_dp4a_f32` argument - the switch is the env var
   `HIPENGINE_GGUF_DENSE_Q8_DP4A_F32` - and that route stays closed to a q5 repack anyway.

For the server matrix itself: `gguf_mtp_c1c8_server_bench.py` accepts neither
`--require-mtp`, `--tag` nor `--max-run-seconds`, and its `--model` default is
`Qwen3.6-27B-Q4_K_M.gguf` - omitting `--model` benchmarks the wrong model and still produces
a clean-looking packet, so always diff the written `protocol` block against a retained packet
before trusting a comparison.

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
