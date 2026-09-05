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

The raw `quant/gguf_k_gemv.{hip,py}` family also exposes an exact
Q5_K selected grouped-row4 owner. It accepts exclusive expert starts and an
optional sorted-lane-to-original-row map, preserves selected GEMV reduction
order, and keeps `selected_gemv_bf16_bf16_out` as its strict fallback.
Qwen4Exp gfx1151 production selects it for ungrouped gate/up rows>=64;
strict, short rows and missing registry capabilities keep selected GEMV.

`hip_gfx1151` compiles shared gfx11 `.hip` bodies as native `gfx1151` code objects and registers a peer backend key. `hipengine/kernels/hip_gfx1151/__init__.py` controls aliases, exclusions, thresholds, and architecture-specific defaults. A gfx1100 variant is not a gfx1151 default merely because the source compiles there; each promotion needs its own correctness and performance gate.

### CUDA is a peer backend

`cuda_sm120a` has independent `.cu` bodies and Python wrappers. It does not alias HIP launch wrappers. CUDA-specific CUTLASS, cuDNN, cuBLASLt, graph, or thread-geometry choices are not selection evidence for either gfx11 backend.

## CPU-reference oracle catalog

CPU oracles favor clarity and deterministic boundaries over speed. They are the required comparison path for net-new kernels.

| Model/path | Source | Oracle families |
| --- | --- | --- |
| Shared primitives and Qwen/PARO/GGUF | `cpu_reference/ops.py` | embedding, linear/QKV/O/lm-head, RMSNorm, rotate, full/paged attention, KV quant/dequant/write, GDN and Conv prefill, GGUF Q4/Q5/Q6/Q8 dequant/GEMV, PARO AWQ pack8, MoE selected/tail, MTP/NextN helpers |
| Qwen4Exp | `cpu_reference/qwen4_exp.py` | four-branch GR, PLE hash/gate/dilated Conv, QSA split-half partial RoPE/block pooling/scoring/selection/sparse GQA, sigmoid-gated GDN boundary, 512/top-10 MoE, and reduced complete layer/model semantics |
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
| Router/select | `moe/router.{hip,py}` | `router_logits`, `router_select`, `router_topk_shared`, `router_topk_split_shared` | BF16/FP16/F32 hidden/weight combinations; deterministic top-k and shared-gate routes. The Qwen4Exp multirow F32 owner preserves dense FMA/reduction order while reusing each weight across four rows: rows508 primitive wall is 3.767→1.911 ms and clean p508 is 89.689→91.121 tok/s with exact logits/state/tasks; c1 and the registered `f32_hidden` route remain fallback. Library handle is hoisted into a module cache (`_router_library()`) so per-launch host cost stays a plain ctypes call (~15 us) instead of re-running `build_qwen35_router(load=True)` (~34 us/call with a pinned session compiler version). |
| MoE grouping and packing | `moe/group_scatter.{hip,py}` | `moe_group_count/prefix/scatter`, `moe_group_compact`, `moe_gather_packed_hidden`, `moe_wmma_tile_map`, `moe_mmq_tile_map` | Stable count/prefix/scatter and compact tile metadata; generic and `w4_paro`. |
| MoE prefill orchestration leaf | `moe/prefill.py` | `moe_prefill` (`w4_paro`) | Registered wrapper composition for selected-expert prefill. |
| Whole selected-expert FFN | `quant/paro_moe_ffn_fused.{hip,py}` | `moe_ffn_selected` (`w4_paro`) | Rotate → gate/up → SiLU → down-rotate → down projection megakernel; primitive chain remains fallback. |
| c1 native dispatcher | `dispatch/moe_c1_dispatch.{hip,py}` | C function-table dispatcher (not a registry layer) | Contracts Python launch overhead while invoking registered/raw function pointers; does not replace component kernels. |
| SiLU/rotation primitives | `fused/paro_silu.{hip,py}` | `silu_mul_dual`, `silu_mul_separate`, `silu_mul_dual_rotate`, `silu_mul_pair_rotate` | Primitive and fused activation/down-rotation boundaries coexist; separate BF16 SiLU permits exact in-place replacement of its gate plane. |
| MoE combine/tail | `fused/paro_combine.{hip,py}` | `weighted_lanes_sum`, `weighted_sum`, `shared_gate_combine`, residual/RMSNorm composites | BF16/FP16/F32 values with FP32 route weights/gates; explicit primitive fallbacks are registered. Qwen4Exp prefill uses exact token-local BF16 batch siblings for compact top-10 weighted sum and shared-gate combine (gfx1151 reduced three-row traces: 1,963/1,403 ns); c1 primitives remain unchanged. |
| Paged KV write/copy | `attention/paged_kv_write.{hip,py}` | `paged_kv_write`, `paged_kv_copy` (`bf16`, PARO/GGUF, INT8 layouts) | All attention-visible writes consume complete `KVLiveSpans`; includes BF16 and supported INT8 storage formats. The FP32→BF16 family includes shared-cache prompt rows with one explicit logical position/table per row for Qwen4Exp prefill; a reversed-page gfx1151 fixture traces at 8,376 ns. |
| Full/paged attention | `attention/paged_attn_decode.{hip,py}` | `full_attn_decode/prefill`, `paged_attn_decode/prefill`, `full_attn_gate_mul` | Contiguous and paged, batched, GQA, split-K, gated reduce, and supported INT8 KV variants. Per-token/head INT8 includes a row-batched 24Q/4KV/D256 split-K producer plus explicitly strided BF16 gated reducer; the c1 leaf remains registered as its numerical fallback. gfx1151 Qwen3.5-0.8B rows1/8Q/2KV/D256 selects generic split-K3+fused BF16 gate at cap514-641. The private-c1 exact leaf is the fixed256 body at 256 threads (strict exact default) with a parameterized `fixed256_threads_spans` probe at runtime block width; gfx1151 promotes 1024 threads (T2 non-exact, execution-profile gate-passed) via `GGUF_SHORT_C1_BATCH_ATTN_THREADS`. Dense H5120/L64/24Q/4KV/D256 selects the BF16 grouped-GQA split producer from context 4096; shorter contexts and unsupported shapes/backends retain the generic producer. |
| AOTriton adapter | `attention/aotriton_wrap.py`, `attention/aotriton.py` | `full_attn_prefill` (`w4_paro`, `gguf_qwen35`) | Optional library adapter; native raw-pointer paths remain available. |
| Linear-attention Conv | `linear_attn/conv.{hip,py}` | `linear_attn_*conv_decode/prefill`, chain/tree and snapshot composites | Decode, segmented prefill, verifier tree/chain, and state-snapshot variants. |
| Linear-attention GDN | `linear_attn/gdn.{hip,py}` | `linear_attn_prefill_prepare`, `gdn_*recurrent*`, RMSNorm/gate/rotate/cast/snapshot composites | Exact schedules retain FP32 recurrent state. The gfx1151 Qwen3.8 Q4_K_S production experiment may select FP16 state (FP32 accumulation) through explicit `_fp16state` scalar, chain, segmented, indexed-singleton, compact-peer prefill, and decode-order writers; every supported leaf retains an FP32 fallback. SPECDEC2 P8 resolves the existing FP16 chain row writer through a non-fallback production manifest while retaining consumer-owned dtype-sized rollback snapshots and the strict unfused cast; the complete P9 product gate retains explicit compatibility but promotes no automatic cell, so K0 remains default. gfx1151 Q4 and Q8 `(16K,16V,128,128)` select cluster8; Q4 `(16K,48V,128,128)` selects 1K-chunked compact-peer wave32; all other gfx1151 shapes retain exact nonvolatile LDS32. |
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

### GGUF / Qwen / Laguna path

GGUF is not a PARO alias. Raw GGML blocks, pack8/T16/qmicro/X8 replacement layouts, exact expanded planes, and source-F16 Laguna tensors have distinct storage and registry keys.

#### GGUF projection and quant families

| Quant/layout family | Source / wrapper | Principal registry layers | Stable notes |
| --- | --- | --- | --- |
| Q4_K selected FFN megakernel | `quant/gguf_q4_k_moe_ffn_fused.{hip,py}` | `moe_ffn_selected` (`gguf_q4_k`) | Whole selected gate/up → SiLU → down projection; primitive selected projections remain fallback. |
| Qwen4Exp GR/PLE/GDN | `fused/qwen4_exp_gr.{hip,py}`, `fused/qwen4_exp_ple.{hip,py}`, `linear_attn/qwen4_exp_gdn.{hip,py}` | grouped GR read/write, sparse PLE gate/Conv/add, `gdn_recurrence_norm_gate` (`f32_state`) | Strict raw-pointer primitives for four authoritative BF16 branches, FP32 PLE history/compute, and FP32 recurrent state with sigmoid output gate. The retained `gr_gated_mean_sigmoid` owner preserves both materialized F32 gate and mixed output bit-for-bit and removes one launch through rows<=256, with registered `strict_unfused` fallback. For rows>256, the retained raw-Q8 up composite preserves each coltile8 reduction while grouping two hidden columns across four branches and emits both gate and mean: clean p508 is 91.158→91.600 tok/s and code-p1024 is 88.754→89.239 tok/s with 450/450 logits and 18/18 state/tasks exact. The primitive coltile plus GR epilogue remains fallback. GDN has c1 decode plus a row-bulk sibling that is bit-exact to serial recurrence. The GDN family also registers a T0 tile-16 raw-Q/K staging sibling for Hk16/Hv32-or-48/D128 prefill; the columnwarp parent and serial strict route remain registered fallbacks. Qwen4Exp K4 Conv now has a separately registered bulk prefill owner that emits the same contraction sequence as serial decode per row; output/state are F32-bit exact, p508 Conv compute launches fall 18,432→72 (plus 72 final-state launches), and the serial owner remains fallback. The gfx1151 recurrence trace records the bulk symbol at 17,474 ns for a five-row reduced fixture. The registered `qwen4exp_sigmoid_peer_prefill` host composite chains Qwen4Exp prepare, compact peer-wave32 recurrence, and sigmoid gate. All-layer arithmetic fails the full numerical envelope, but the named gfx1151 production profile certifies global layers 35–47 (actual GDN layers 36/37/38/40/41/42/44/45/46): the complete stack passes 448/450 top-1 with no scope failures. At p508 it replaces nine exact fused launches with 26.77 ms total peer work, reducing the traced GDN family 992.16→750.68 ms; `qwen4exp_sigmoid_strict_prefill` and c1 remain fallbacks/oracles. |
| Qwen4Exp QSA | `attention/qwen4_exp_qsa.{hip,py}` | `qsa_split_norm_rope`, `qsa_norm_rope`, `qsa_pool_norm_rope`, `qsa_index_score`, `qsa_select_blocks`, `qsa_sparse_attention` | Split-half partial RoPE, FP32 raw-key complete-block pooling, deterministic lower-start tie break, and sparse original-BF16-K/V GQA. The exact c1 index append has a registered device-position sibling for graph-owned decode control; scalar/row append remains fallback. c1 plus explicit-position row-bulk Q/K/gate and index-query transforms are registered; reduced gfx1151 three-row traces are 2,204/2,124 ns and bit-exact to c1. Variable-selection sparse rows consume complete paged spans and trace at 7,213 ns on a reversed-page fixture; non-flash multirow dense rows use the exact fixed256/precomputed-offset/vector2 owner (real primitive 6.846→2.485 ms, clean p508 91.529→92.442 tok/s, code-p1024 89.150→90.634 tok/s), with generic FP32 batch context fallback. A bounded prompt-chunk mixer composes bulk quant projections, exact row transforms, shared K/V writes, dense batch context, and variable-selection sparse context. Its block-table-aware raw index-key scatter replaces p508's 6,096 per-row D2D copies with 24 chunk kernels and cuts p512 trace launches 11,053→4,933 with bit-exact logits; c1 append remains fallback. Its reduced six-row dense→sparse boundary matches independent c1 output/state and traces the dense/sparse leaves at 3,927/6,132 ns on gfx1151. The corrected exact chunk path uses chunk-batched PLE staging, batched projections, decode-order-exact bulk causal Conv, and exact grouped Q5_1 down pass all 687 teacher-forced rows bit-for-bit and improve the natural suite 5.265→12.117 tok/s (2.301x); warm p512 is 16.555 tok/s. The former size-2 smoke remains historical (`KL_teacher=0.00510`, `KL_serial=0.00410`), while approximate size 9 is rejected (`KL_serial=0.09754`; artifact: `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-chunked-prefill-smoke.json`). The first real sparse row at token 2,052 also passes; promoted chunk64 is bit-exact to serial, both have teacher KL `7.65e-5` and top-1 264, teardown is clean, and prefill improves 370.565→136.129 s (2.722x), as does a repeated-token structural 4K checkpoint (`KL_teacher→serial=4.40e-5`, `KL_teacher→chunk=4.78e-5`, diagnostic 854.982→574.759 s). A chunk-only repeated-token 16K checkpoint further passes teacher KL `7.55e-5`, top-1 264 exact, and clean teardown in 2,434.172 s; strict remains measured through 4K. A chunk-only repeated-token 64K checkpoint also passes teacher KL `5.74e-6`, top-1 264 exact, and clean teardown in 10,336.580 s. Real full-capacity ownership allocates and tears down at 262,144 tokens (91,126,119,496 tracked bytes, 38,915,162,112 physical bytes still free, zero tracked bytes after close), but this is not a 262K inference result. Natural 4K retrieval and Transformers index-reference control pass exactly. Persistent compressed-key preparation reduces pool launches 24,540→384 and block work 18,849,792→12,288; exact device radix top-512 removes 24,540 score D2H synchronizations and 403.341 MB metadata H2D, reducing natural 4K 303.528→294.434 s with unchanged output/control. Production wave32 H128 sparse attention improves its real 2,048-token primitive 1,982→1,796 us and paired natural 4K 298.078→290.941 s; four sparse categories have bit-exact final logits/control and strict spans remain fallback. Exact chunk-batched score/top-k reduces launches 49,080→768 and paired natural 4K 295.706→290.971 s; exact grouped rowbatch8 Q4_K gate/up then gives 291.624→231.798 s, and output4 scheduling cuts full-shape CTAs 75% plus paired wall 235.774→228.569 s, all with bit-exact logits/control. The exact owner now also covers Q8_0-down layers, removing 64 direct gate/up launches and improving paired p508 12.021→11.189 s (45.404 tok/s). Its current sibling predecodes exact `d*scale`/`dmin*min` metadata once into 2 KiB LDS; with chunk256 this reaches 51.220 tok/s first-run / 58.466 tok/s steady p508 and 55.046 tok/s p1006, all bit-exact. Natural 16K/64K now pass at 17.301/17.099 tok/s with retrieval/control/CPU-oracle/lifecycle exact; 262K execution and broader lifecycle gates remain open (`benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-2052-transition.json`, `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-4k.json`, `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-16k.json`, `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-64k.json`, `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-262k-capacity.json`). The complete runner mirrors paged K/V physical ownership, uses dense equivalence through 2,051 tokens, then runs native projections/pool/score/sparse attention with an exact host lexicographic top-512 control fallback; the single-thread device selector remains a reduced-fixture oracle and is not the long-context route. For gfx1151 c1 H256 indexed-sparse decode, production selects an exact ordered three-pass owner: parallel QK scores preserve the strict reduction tree, one global selected-order recurrence emits online-softmax coefficients, and output-column recurrences consume them in the same order. The serialized strict owner remains the registered fallback. |
| Qwen4Exp vision | `vision/qwen4_exp_vision.{hip,py}` | `vision_layernorm`, `vision_add_bias_residual`, `vision_gelu`, `vision_attention` | <=1K Qwen3-VL-compatible images/videos: merge-compatible RGB grids up to 256 patches/temporal pair, 2×2 block-major order, align-corners learned-position interpolation, frame-pair attention isolation, multiple images, odd-frame duplication, and typed placeholders. FP32 attention uses explicit vision H/W RoPE. Full 32×64 encoder matches Transformers at relative L2 1.48e-6/cosine 1.0; text QSA's registered MRoPE sibling applies interleaved T/H/W `[11,11,10]` and traces at 12,143 ns. Bounded PNG data URLs work through non-streaming chat; remote URLs/SSE/>1K remain open. |
| Qwen4Exp raw Q5_1 experts | `quant/qwen4_exp_q5_1.{hip,py}` | selected `linear`/`moe_linear` (`gguf_q5_1`) | Strict selected-expert consumer plus exact grouped rowbatch8 and grouped-WMMA down projections for the pinned Unsloth UD-Q4_K_XL mixed quant. Exact output8 scheduling cuts full-shape grouped-down CTAs 1,310,720→163,840 and paired natural 4K 237.131→222.228 s with exact logits/control; output1 remains fallback. The current short-prefill owner iterates 512 experts through 64 worker CTAs and uses 128 physical threads to materialize the same 256 logical partials before the original reduction tree; its p512 bucket is 3.470→2.534 s with exact bits. Q5_1 grouped WMMA is not the strict owner; explicit gfx1151 Qwen4Exp `production` selects it with cooperative Q4 gate/up on the definitive maximal suffix layers 27–47. Every layer 0–26 fails final-prompt mean or p95; the 27–47 450-row/three-repeat manifest passes at mean/p95/p99/max KL 1.05e-4/3.81e-4/1.52e-3/5.59e-3 and 99.556% top-1, improving the MoE-only p508/p1012 59.401→67.243 / 58.723→66.268 tok/s. The same explicit profile adds dense-Q8 WMMA on certified layers 32–47; the combined 450-row gate passes mean/p95/p99/max KL 1.20e-4/4.93e-4/1.72e-3/8.69e-3 and 99.778% top-1, reaching 73.361/71.834 tok/s. Exact grouped/coltile fallbacks remain registered. The strict selected decode default now uses 64 physical threads to materialize the same 256 logical partials before reconstructing the original shared strides 128/64/32 and wave32 tail. The first exact t128 contraction cuts Q5 cycle-wall 692.930→410.364 ms and graph decode 11.380→12.140 tok/s; t64 is BF16-bit exact to both registered t128/t256 fallbacks, cuts its matched Q5 trace 444.699→362.525 ms, and improves graph decode 13.077→13.302 tok/s (+1.69%). The c1 default also fuses selected down with routed weighted sum: one CTA per H=2560 output preserves every route BF16 result and the original ordered `fmaf`, removes 1,806 traced launches, contracts target cycle-wall 369.241→313.535 ms, and improves 13.379→13.523 tok/s (+1.06%); the separate exact chain remains fallback. A default-off 64-thread sibling improves warm decode further but is rejected for production mean/p95 KL (`0.002565/0.007202`). |
| Raw Q5_K/Q6_K/Q8_0 | `quant/gguf_k_gemv.{hip,py}` | `linear`, `linear_pair`, `attention_projection_quad` | Decode/prefill, BF16/F32 output, pair/quad launch contractions, rowbatch/coltile variants. The gfx1151 Qwen4Exp exact Q8/F32 owner first cut p508 26.264→14.718 s with coltile4/rowbatch8, then promotes coltile8/rowbatch4 alongside exact expert scheduling to reach 42.376 tok/s; p512 Q8 kernel wall falls 3.121→2.482 s with bit-exact full logits. Its c1 F32/F32 output-pack8 sibling reuses each activation across eight columns without changing per-output arithmetic, cuts the traced Q8 bucket 2.620→1.171 s and paired decode 5.698→6.305 tok/s; registered scalar raw Q8 remains fallback. gfx1151 Q5/Q6 W7900 policies remain disabled. |
| Q5_K/Q6_K selected prefill WMMA | `quant/gguf_k_selected_prefill.{hip,py}` | `moe_linear` | Raw-byte compact selected-MoE f16-WMMA consumers with strict raw selected-gemv fallbacks. The gfx1151 Qwen4Exp layer-2 Q5_K/Q5_K route is production-rejected/default-off: p508 Q5_K gate/up falls 279.86→16.66 ms and 20/20 category-balanced p512 pairs win by about 5%, but the complete 450-row gate fails prefill-last mean KL at 0.001179 > 0.001. Do not rescreen unchanged T2 arithmetic; the older optimized metadata-hoist sibling is a separate rejected path. |
| Raw Q3_K selected | `quant/gguf_q3_k_gemv.{hip,py}` | `moe_linear` | Q3 selected-expert projection family. |
| Q4_K pack8/raw | `quant/gguf_q4_k_gemv.{hip,py}` | `linear`, `linear_pair`, `linear_pair_silu`, `linear+residual` | Raw GGUF math and lossless pack8 layouts; pair/SiLU and exact rounded-BF16 residual composites where registered. Qwen4Exp c1 now resolves raw selected dual gate/up by registry capability, halves Q4 launches 94→47/token, and improves paired decode 6.065→6.223 tok/s. Its operation-complete sibling preserves both BF16 projection boundaries and the standalone SiLU/product bits, removes another 47 launches/token, and improves 6.400→6.420 tok/s. The selected default now maps logical lanes `tid`/`tid+64` onto 64 physical threads while publishing the same four strict wave sums; it contracts Q4 cycle-wall 1,076.767→814.906 ms across 1,974 launches and improves counterbalanced graph decode 12.003→13.167 tok/s (+8.84%). IDs/full logits are exact and the physical128 dual/singleton chains remain fallbacks. Above these kernels, gfx1151 now captures each complete stateless Qwen4Exp MoE chain in one self-validating request-owned graph: 48 captures/zero rejects, 192 full-logit rows exact, eager 6.511→11.515 tok/s, then exact Q5/Q4/Q5 contractions and Q5 down+weighted fusion reach 12.140/13.167/13.302/13.523 tok/s; c2 is exact and stateful GDN/QSA remain outside replay. Explicit gfx1151 `production` adds one-plane Q8_1 DP4A Q4 dual+SiLU on calibrated static layers `0,2,5,6,8,9,10,11,13–47`; measured-failing layers `1,3,4,7,12` remain exact. The physical64 owner preserves the candidate's 128 logical partials and BF16 boundaries; combined production passes 447/450 top-1 with mean/p95/p99/max KL 2.72e-4/1.40e-3/4.00e-3/5.77e-3, improves decode 13.880→15.543 tok/s, and contracts Q4 target cycle-wall 825.340→397.755 ms. Direct suffix13→calibrated43 is +0.37%. Suffix12 and all-layer DP4A are rejected at 445/450; exact logical128/t64 remains fallback and omitted-profile default. A Qwen4Exp one-layout expert replacement is rejected and removed: sampled layer-0 bits are exact and micro speed is 4.47x, but uncached load is 979 s and full-model mean/p95 KL fail at 0.002089/0.006529. Primitive projection+add fallbacks remain available. |
| Q4_K/Q6_K prefill WMMA | `quant/gguf_q4_k_prefill.{hip,py}` | `linear` | Resident pack8/raw prefill consumers; exact scalar/pack8 routes remain fallbacks. The p512 pack8-Q4 rounded-residual output-store sibling is rejected (0.958x core / 0.952x public complete-model prefill) and is not registered. |
| Q8_0 T16 prefill | `quant/gguf_q8_0_t16_prefill.{hip,py}` | `linear`, `linear_pair` | WMMA/T16 Q8 prefill and architecture-specific wave schedules. gfx1151 rows512/K1024/N16+N16 alpha/beta uses the exact two-wave dual owner; singleton WMMA remains the fallback. |
| Q8_0 grouped down (P1) | `quant/gguf_q8_0_prefill.{hip,py}` | `moe_linear` | P1 device-driven grouped Q8_0 down owner (`gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out`) for the layer-2/4/30/46/47 Q8_0 expert-down family. Reads `expert_start` on device and iterates experts via a fixed worker grid, replacing the `group_expert_start` D2H copy + Python loop over 512 experts. BF16-exact to `gguf_q8_0_gemv` per grouped row (RED test `test_qwen4_exp_q8_0_grouped_down.py`). Strict per-expert selected gemv remains default; `HIPENGINE_QWEN4_EXP_Q8_0_GROUPED=1` selects it. **Perf-negative as of 2026-08-30** (microbench 20260830T202256): grouped owner ~3-12x slower than strict `selected_gemv` on layer-2 shape due to a 1.31M-block grid with OUT_BATCH=1 and no weight reuse; not promoted. |
| Q4/Q5/Q6 T16 selected | `quant/gguf_t16_selected_gemv.{hip,py}` | `linear`, `linear_pair_silu`, `moe_linear`, `moe_linear+weighted_sum`, `linear+residual` | c=1 and selected-prefill T16/qmicro/interleaved consumers, including weighted/residual composites. A Qwen4Exp one-layout replacement profile is rejected and removed: optimized p512 is neutral (213.52 vs 211.76 tok/s), paired decode regresses 5.925→3.615 tok/s, and mean/p95 KL fail at 0.003010/0.008338. gfx1151 Qwen3.8 standard-Q4 physical rows6/8/12/16 use the exact single-wave WMMA parent for K/N 5120/6144, 5120/10240, 5120/12288, and 6144/5120; narrow V, wide-K down, and misses retain shared-B. gfx1100 Qwen3.6 physical rows6 instead uses the C1-equivalent rowtile for K/N 5120/1024, 5120/6144, 5120/10240, 5120/12288, and 17408/5120, plus the exact single-wave parent for 5120/17408; all other rows/shapes keep explicitly registered shared-B. The same gfx1151 model's Q5 K6144/N5120, K17408/N5120, and K5120/N10240 rows2-8 use the exact col8 rowtile; registered parents remain strict fallbacks. |
| Q6/Q4 mixed and narrow K/V grids | `fused/gguf_q6_q4_pair.{hip,py}` | `linear_pair` (standard-Q6+Q4, Q4, Q4+planar-Q6) | Exact block-parallel rows1 pairs; gfx1151 qualifies Qwen3.8 recurrent K5120/N10240+N6144 and full-attention K/V K5120/N1024+N1024 while primitive projections remain fallbacks. |
| Dense Q6_K T16/qmicro | `quant/gguf_q6_k_t16_gemv.{hip,py}` | `linear`, `linear+argmax`, `linear+residual` | Exact dense Q6 decode/prefill/root families. gfx1100 planar row8 uses the exact DPP reduction (VGPR136→112, bpermute320→0), admitted on all 55 actual-operation rows and retained by a 1.634% complete-owner wall win; rows1-7 keep the generic reduction. gfx1151 rows>=512 uses 128-thread/four-wave shared-weight WMMA for standard K5120/N10240 QKV (2.96-3.55x) and planar K17408/N5120 FFN-down (1.42-1.50x); both use 24 KiB LDS / 248 VGPR. Rows<512, narrow V, root, shape misses, and peer backends retain exact one-wave/16x16 primitives. |
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
| 2-8 | `dense_rowtile`/`_col4` (gfx1151 qualified H5120-package row8 shapes use `dense_rowtile16_w2`) | `dense_dual_q8_1x2_rowtile8` | `t16_gemv_rowtile` (per-shape cap; qualified gfx1151 Qwen3.8 shapes use col8 ownership) | `t16_gemv_rowtile` |
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
rows2-4 retain two singleton dense-F32 projections. No payload or scratch is
added. Evidence:
[`Qwen3.8 dense-F32 alpha/beta pair`](../benchmarks/results/2026-08-16-gfx1151-qwen38-27b-dense-f32-alpha-beta-pair.json).

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
│   ├── dms_compact.hip
│   ├── laguna_flash_attention_prefill.hip
│   ├── laguna_kv_attention.hip
│   ├── maple_attention.hip
│   ├── moonshine_attention.hip
│   ├── paged_attn_decode.hip
│   ├── paged_kv_write.hip
│   └── qwen4_exp_qsa.hip
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
