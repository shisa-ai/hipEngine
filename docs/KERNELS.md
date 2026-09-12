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

The experimental Q5_K bundled row-reduction sibling was removed after the
clean `4b39fbfa5` canonical A/B: all72 trajectories exact, but five prefill
and six request-wall cases regress. Earlier kernel1.015x/1.021x screens
and five full-logit/state/KV cases did not establish a whole-model win.
The original row4 implementation remains; its64/128/256-thread and
CPU-reference coverage is retained. This removal does not affect the
separate production Q8 bundled variant.
Evidence: `2026-09-06-framework-qwen4exp-q5k-bundle-rejected.json`.

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
| TimesFM 2.5 | `cpu_reference/timesfm.py` | multiplicative-scale RMSNorm, ResidualBlock heads, fused-QKV RoPE (pre-norm), QK norm, per-dim softplus scaling, unscaled masked attention, patch running stats/revin, AR patch decode; oracle fixture from the vendored torch reference |
| TimesFM 3.0 | `cpu_reference/timesfm3.py` | non-autoregressive multivariate forward+decode: seq + non-causal variate attention (SDPA semantics: scores x sqrt(head_dim), fully-masked rows -> zeros), 192-dim ReLU tokenizer, stitching, linear detrending, CPM iterative RevIN refine, freeze_after post-hoc mean/std; nn.RMSNorm eps = finfo(float32).eps; oracle fixtures (base/edge/covmask incl. long horizon) from the vendored torch reference |
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
| Router/select | `moe/router.{hip,py}` | `router_logits`, `router_select`, `router_topk_shared`, `router_topk_split_shared` | BF16/FP16/F32 hidden/weight combinations; deterministic top-k and shared-gate routes. The Qwen4Exp multirow F32 owner preserves dense FMA/reduction order while reusing each weight across four rows: rows508 primitive wall is 3.767→1.911 ms and clean p508 is 89.689→91.121 tok/s with exact logits/state/tasks; c1 and the registered `f32_hidden` route remain fallback. Library handle is hoisted into a module cache (`_router_library()`) so per-launch host cost stays a plain ctypes call (~15 us) instead of re-running `build_qwen35_router(load=True)` (~34 us/call with a pinned session compiler version). |
| MoE grouping and packing | `moe/group_scatter.{hip,py}` | `moe_group_count/prefix/scatter`, `moe_group_compact`, `moe_gather_packed_hidden`, `moe_wmma_tile_map`, `moe_mmq_tile_map` | Stable count/prefix/scatter and compact tile metadata; generic and `w4_paro`. |
| MoE prefill orchestration leaf | `moe/prefill.py` | `moe_prefill` (`w4_paro`) | Registered wrapper composition for selected-expert prefill. |
| Whole selected-expert FFN | `quant/paro_moe_ffn_fused.{hip,py}` | `moe_ffn_selected` (`w4_paro`) | Rotate → gate/up → SiLU → down-rotate → down projection megakernel; primitive chain remains fallback. |
| c1 native dispatcher | `dispatch/moe_c1_dispatch.{hip,py}` | C function-table dispatcher (not a registry layer) | Contracts Python launch overhead while invoking registered/raw function pointers; does not replace component kernels. |
| SiLU/rotation primitives | `fused/paro_silu.{hip,py}` | `silu_mul_dual`, `silu_mul_separate`, `silu_mul_dual_rotate`, `silu_mul_pair_rotate` | Primitive and fused activation/down-rotation boundaries coexist; separate BF16 SiLU permits exact in-place replacement of its gate plane. |
| MoE combine/tail | `fused/paro_combine.{hip,py}` | `weighted_lanes_sum`, `weighted_sum`, `shared_gate_combine`, residual/RMSNorm composites | BF16/FP16/F32 values with FP32 route weights/gates; explicit primitive fallbacks are registered. Qwen4Exp prefill uses exact token-local BF16 batch siblings for compact top-10 weighted sum and shared-gate combine (gfx1151 reduced three-row traces: 1,963/1,403 ns); c1 primitives remain unchanged. |
| Paged KV write/copy | `attention/paged_kv_write.{hip,py}` | `paged_kv_write`, `paged_kv_copy` (`bf16`, PARO/GGUF, INT8 layouts) | All attention-visible writes consume complete `KVLiveSpans`; includes BF16 and supported INT8 storage formats. The FP32→BF16 family includes shared-cache prompt rows with one explicit logical position/table per row for Qwen4Exp prefill; a reversed-page gfx1151 fixture traces at 8,376 ns. |
| Full/paged attention | `attention/paged_attn_decode.{hip,py}` | `full_attn_decode/prefill`, `paged_attn_decode/prefill`, `full_attn_gate_mul` | Contiguous and paged, batched, GQA, split-K, gated reduce, and supported INT8 KV variants. Per-token/head INT8 includes a row-batched 24Q/4KV/D256 split-K producer plus explicitly strided BF16 gated reducer; the c1 leaf remains registered as its numerical fallback, and the gfx1100 Qwen3.8-27B artifact qualifies the batch variant to physical c4 with the c1 leaf as the registered fallback above that width. gfx1151 Qwen3.5-0.8B rows1/8Q/2KV/D256 selects generic split-K3+fused BF16 gate at cap514-641. The private-c1 exact leaf is the fixed256 body at 256 threads (strict exact default) with a parameterized `fixed256_threads_spans` probe at runtime block width; gfx1151 promotes 1024 threads (T2 non-exact, execution-profile gate-passed) via `GGUF_SHORT_C1_BATCH_ATTN_THREADS`. Dense H5120/L64/24Q/4KV/D256 selects the BF16 grouped-GQA split producer from context 4096; shorter contexts and unsupported shapes/backends retain the generic producer. | Native BF16-gated prefill uses owned global score scratch when its context-sized shared allocation would exceed 64 KiB; bounded query batches reuse the split partial-output arena without changing the parent reduction order. The explicit `causal_gqa_gate_bf16_global_scores` variant also permits parent-parity checks at short contexts. INT8 per-token/head includes a row-batched 24Q/4KV/D256 split-K producer with an explicitly strided BF16 gated reducer; the c1 leaf remains its registered numerical fallback. |
| AOTriton adapter | `attention/aotriton_wrap.py`, `attention/aotriton.py` | `full_attn_prefill` (`w4_paro`, `gguf_qwen35`) | Optional library adapter; native raw-pointer paths remain available. |
| Linear-attention Conv | `linear_attn/conv.{hip,py}` | `linear_attn_*conv_decode/prefill`, chain/tree and snapshot composites | Decode, segmented prefill, verifier tree/chain, and state-snapshot variants. |
| Linear-attention GDN | `linear_attn/gdn.{hip,py}` | `linear_attn_prefill_prepare`, `gdn_*recurrent*`, RMSNorm/gate/rotate/cast/snapshot composites | Exact schedules retain FP32 recurrent state; segmented, chain/tree, snapshot, and decode-order writers cover prefill, verifier, and multi-request selected commit, with optional FP32 state-row journals, direct BF16 handoffs, and an exact FP32 output tap. FP16-state (FP32 accumulation) and gfx1151 cluster/chunked compact-peer variants are explicit opt-ins or capability selections that always retain an FP32 fallback. |
| Runtime state | `runtime/state.{hip,py}` | token embedding, positions/metadata, graph record/commit, scalar state, profiling wall-clock marker | Device-side graph/verify bookkeeping, indexed row state, token publication, and profiling-only steady-clock boundaries. |
| Sampling | `sampling/sampler.{hip,py}` | `sampler`, `mtp_draft_topk` | Greedy/temperature/top-k helpers and bounded draft top-k. |

**Compact DMS attention** — `attention/dms_compact.{hip,py}` registers `dms_extract_decision`, `dms_decision_source`, `dms_streaming_pack`, `dms_append_decode`, and `dms_compact_attn_decode` (grouped GQA fallback plus bounded-LDS split-K) for the compact-KV path. The split-K family includes the `dms_compact_attn_splitk_group6_wave_producer_kernel` for the Qwen3.8 24Q/4KV/D256 geometry (one compact token scored per wave, Q shared across the GQA group), compiled for gfx1151 and gfx1100; the grouped and scalar split-K producers remain registered fallbacks. The CPU-reference oracles in `cpu_reference/dms.py` are the registered strict fallbacks for every key; the kernels are wired into `DMSCompactBackend` behind explicit device-payload selection, and no model package defaults to DMS.

The explicit gfx1100 `attention/dms_compact_int8.{hip,py}` family adds compact
INT8 pack/append and bounded split-K attention with FP32 per-token/head scales:
the append path is the chunked keep-scan (`dms_int8_append_kernel`, one launch per
chunk rather than a serial per-token loop), and the attention path registers the
wave-grouped GQA producer `dms_int8_attn_split_wave_kernel` (in-register int8
loads with scale dequant, Q shared across the GQA group) with the generic
`dms_int8_attn_split_kernel` as fallback. Device fixtures cover exact codec
bytes/scales and ownership, above-window retention, fail-closed overflow,
attention numerics, and snapshot restoration. BF16 kernels remain unchanged
fallbacks; model-serving INT8 DMS qualification is separate and is not
established by these device fixtures. Speed and correctness evidence for the
wave6 producer, chunked keep-scan, and wave-grouped INT8 producer lives in
`benchmarks/results/2026-09-08-w7900-dense-vs-dms-speed-probe-final.json` and
the capacity lane's XTX verification
(`benchmarks/results/2026-09-08-rx7900xtx-dms-int8-merged-lane-capacity.json`).

### GGUF / Qwen / Laguna path

GGUF is not a PARO alias. Raw GGML blocks, pack8/T16/qmicro/X8 replacement layouts, exact expanded planes, and source-F16 Laguna tensors have distinct storage and registry keys.

#### Qwen3.8-27B dense GGUF route map

Qwen3.8-27B dense is served by the shared Qwen3.5/3.6/3.8 dense plugin; there is no separate Qwen3.8 model plugin. The chain from `LLM(...)` to a launch is:

| Stage | Module | Key |
| --- | --- | --- |
| Model plugin | `models/qwen35.py` (`QWEN35_GGUF`) | `name="qwen3_5_gguf"`, `architectures=("qwen35",)`, `default_quant="gguf_q4_k_m"` |
| Backend admission | `kernels/backends.py` `select_backend` | explicit arg → `HIPENGINE_BACKEND` → detected arch → `cpu_reference` |
| Generator factory | `generation/registry.py` `resolve_text_generator` | `(model, backend, quant, mode="greedy_one_token")`, exact match |
| Execution profile | `execution_profiles.py` `resolve_runtime_profile` | `(model, backend, quant, profile)`; an omitted profile selects a certified production plan where registered, and the migration route otherwise |
| Materialize / quant | `loading/qwen35_gguf_materialize.py` | file quant `gguf_q4_k_m` → layout/registry quant `gguf_q4_k_t16_v1`; GDN family uses `gguf_qwen35` |
| Dense linear dispatch | `runtime/gguf_linear.py` `resolve_gguf_linear_dispatch` | `(layout, activation, output)` template, backend from the resolved weight |
| Kernel resolution | `kernels/registry.py` `resolve` | exact → no-variant → `fp16` → `cpu_reference`; no cross-HIP-backend fallback |

Default route: bulk WMMA prefill (`use_bulk_prefill`, `bulk_prefill_attention_mode=bulk`, `use_wmma_prefill` default True) and GEMV decode (`use_gemv_decode=True`). `HIPENGINE_GGUF_DECODE_GRAPH` is enabled by default; graph replay additionally requires an admitted layout and the backend's published replay horizon. Decode remains eager when those conditions are not met. An unset sweep graph option now follows engine admission rather than forcing eager execution.

Backend-specific knobs are read through `backend_package_capability(backend, NAME, default)` against module-level constants in `kernels/<backend>/`. Process-start HIP defaults live in `HIP_BACKEND_PROCESS_ENV_DEFAULTS` (gfx1100 `HSA_SCRATCH_SINGLE_LIMIT=8388608`; gfx1151 `GPU_MAX_HW_QUEUES=2`) and never overwrite explicit user values.

`runtime/qwen35_gguf_runner.py` declares 80 module-level `KernelKey("hip_gfx1100", layer, quant, variant)` constants (46 on the dense `gguf_qwen35` GDN/linear-attention families, 34 on the MoE path). **These are nominal source markers, not backend pins:** every consumer discards `key.backend` and substitutes the active backend, for example through a local `_resolve` closure calling `resolve(backend=backend, layer=key.layer, quant=key.quant, variant=key.variant)`. Resolving the dense GDN keys on gfx1151 returns the gfx1151 bodies (`qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16` and its `_fp16state` sibling), not the gfx1100 ones. Grep hits on that literal in this file are not gfx1100-only surfaces. The live pin of this class is PARO's `_PAGED_KV_REGISTRY_BACKEND` in `runtime/qwen35_paro.py`; see `docs/REFACTOR.md`.

#### GGUF projection and quant families

| Quant/layout family | Source / wrapper | Principal registry layers | Stable notes |
| --- | --- | --- | --- |
| Q4_K selected FFN megakernel | `quant/gguf_q4_k_moe_ffn_fused.{hip,py}` | `moe_ffn_selected` (`gguf_q4_k`) | Whole selected gate/up → SiLU → down projection; primitive selected projections remain fallback. |
| Qwen4Exp GR/PLE/GDN | `fused/qwen4_exp_gr.{hip,py}`, `fused/qwen4_exp_ple.{hip,py}`, `linear_attn/qwen4_exp_gdn.{hip,py}` | grouped GR read/write, sparse PLE gate/Conv/add, `gdn_recurrence_norm_gate` (`f32_state`) | Strict raw-pointer primitives for four authoritative BF16 branches, FP32 PLE history/compute, and FP32 recurrent state with sigmoid output gate. The retained `gr_gated_mean_sigmoid` owner preserves both materialized F32 gate and mixed output bit-for-bit and removes one launch through rows<=256, with registered `strict_unfused` fallback. For rows>256, the retained raw-Q8 up composite preserves each coltile8 reduction while grouping two hidden columns across four branches and emits both gate and mean: clean p508 is 91.158→91.600 tok/s and code-p1024 is 88.754→89.239 tok/s with 450/450 logits and 18/18 state/tasks exact. The primitive coltile plus GR epilogue remains fallback. GDN has c1 decode plus a row-bulk sibling that is bit-exact to serial recurrence. The GDN family also registers a T0 tile-16 raw-Q/K staging sibling for Hk16/Hv32-or-48/D128 prefill; the columnwarp parent and serial strict route remain registered fallbacks. Qwen4Exp K4 Conv now has a separately registered bulk prefill owner that emits the same contraction sequence as serial decode per row; output/state are F32-bit exact, p508 Conv compute launches fall 18,432→72 (plus 72 final-state launches), and the serial owner remains fallback. The gfx1151 recurrence trace records the bulk symbol at 17,474 ns for a five-row reduced fixture. The registered `qwen4exp_sigmoid_peer_prefill` host composite chains Qwen4Exp prepare, compact peer-wave32 recurrence, and sigmoid gate. All-layer arithmetic fails the full numerical envelope, but the named gfx1151 production profile certifies global layers 35–47 (actual GDN layers 36/37/38/40/41/42/44/45/46): the complete stack passes 448/450 top-1 with no scope failures. At p508 it replaces nine exact fused launches with 26.77 ms total peer work, reducing the traced GDN family 992.16→750.68 ms; `qwen4exp_sigmoid_strict_prefill` and c1 remain fallbacks/oracles. |
| Qwen4Exp QSA | `attention/qwen4_exp_qsa.{hip,py}` | `qsa_split_norm_rope`, `qsa_norm_rope`, `qsa_pool_norm_rope`, `qsa_index_score`, `qsa_select_blocks`, `qsa_sparse_attention` | Split-half partial RoPE, FP32 raw-key complete-block pooling, deterministic lower-start tie break, and sparse original-BF16-K/V GQA. The exact c1 index append has a registered device-position sibling for graph-owned decode control; scalar/row append remains fallback. c1 plus explicit-position row-bulk Q/K/gate and index-query transforms are registered; reduced gfx1151 three-row traces are 2,204/2,124 ns and bit-exact to c1. Variable-selection sparse rows consume complete paged spans and trace at 7,213 ns on a reversed-page fixture; non-flash multirow dense rows use the exact fixed256/precomputed-offset/vector2 owner (real primitive 6.846→2.485 ms, clean p508 91.529→92.442 tok/s, code-p1024 89.150→90.634 tok/s), with generic FP32 batch context fallback. A bounded prompt-chunk mixer composes bulk quant projections, exact row transforms, shared K/V writes, dense batch context, and variable-selection sparse context. Its block-table-aware raw index-key scatter replaces p508's 6,096 per-row D2D copies with 24 chunk kernels and cuts p512 trace launches 11,053→4,933 with bit-exact logits; c1 append remains fallback. Its reduced six-row dense→sparse boundary matches independent c1 output/state and traces the dense/sparse leaves at 3,927/6,132 ns on gfx1151. The corrected exact chunk path uses chunk-batched PLE staging, batched projections, decode-order-exact bulk causal Conv, and exact grouped Q5_1 down pass all 687 teacher-forced rows bit-for-bit and improve the natural suite 5.265→12.117 tok/s (2.301x); warm p512 is 16.555 tok/s. The former size-2 smoke remains historical (`KL_teacher=0.00510`, `KL_serial=0.00410`), while approximate size 9 is rejected (`KL_serial=0.09754`; artifact: `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-chunked-prefill-smoke.json`). The first real sparse row at token 2,052 also passes; promoted chunk64 is bit-exact to serial, both have teacher KL `7.65e-5` and top-1 264, teardown is clean, and prefill improves 370.565→136.129 s (2.722x), as does a repeated-token structural 4K checkpoint (`KL_teacher→serial=4.40e-5`, `KL_teacher→chunk=4.78e-5`, diagnostic 854.982→574.759 s). A chunk-only repeated-token 16K checkpoint further passes teacher KL `7.55e-5`, top-1 264 exact, and clean teardown in 2,434.172 s; strict remains measured through 4K. A chunk-only repeated-token 64K checkpoint also passes teacher KL `5.74e-6`, top-1 264 exact, and clean teardown in 10,336.580 s. Real full-capacity ownership allocates and tears down at 262,144 tokens (91,126,119,496 tracked bytes, 38,915,162,112 physical bytes still free, zero tracked bytes after close), but this is not a 262K inference result. Natural 4K retrieval and Transformers index-reference control pass exactly. Persistent compressed-key preparation reduces pool launches 24,540→384 and block work 18,849,792→12,288; exact device radix top-512 removes 24,540 score D2H synchronizations and 403.341 MB metadata H2D, reducing natural 4K 303.528→294.434 s with unchanged output/control. Production wave32 H128 sparse attention improves its real 2,048-token primitive 1,982→1,796 us and paired natural 4K 298.078→290.941 s; four sparse categories have bit-exact final logits/control and strict spans remain fallback. Exact chunk-batched score/top-k reduces launches 49,080→768 and paired natural 4K 295.706→290.971 s; exact grouped rowbatch8 Q4_K gate/up then gives 291.624→231.798 s, and output4 scheduling cuts full-shape CTAs 75% plus paired wall 235.774→228.569 s, all with bit-exact logits/control. The exact owner now also covers Q8_0-down layers, removing 64 direct gate/up launches and improving paired p508 12.021→11.189 s (45.404 tok/s). Its current sibling predecodes exact `d*scale`/`dmin*min` metadata once into 2 KiB LDS; with chunk256 this reaches 51.220 tok/s first-run / 58.466 tok/s steady p508 and 55.046 tok/s p1006, all bit-exact. Natural 16K/64K now pass at 17.301/17.099 tok/s with retrieval/control/CPU-oracle/lifecycle exact; 262K execution and broader lifecycle gates remain open (`benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-2052-transition.json`, `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-4k.json`, `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-16k.json`, `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-qsa-64k.json`, `benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-262k-capacity.json`). The complete runner mirrors paged K/V physical ownership, uses dense equivalence through 2,051 tokens, then runs native projections/pool/score/sparse attention with an exact host lexicographic top-512 control fallback; the single-thread device selector remains a reduced-fixture oracle and is not the long-context route. For gfx1151 c1 H256 indexed-sparse decode, production selects an exact ordered three-pass owner: parallel QK scores preserve the strict reduction tree, one global selected-order recurrence emits online-softmax coefficients, and output-column recurrences consume them in the same order. The serialized strict owner remains the registered fallback; the promoted ordered-v2 rewrite (`strict_ordered_three_pass_v2_spans`) preserves that arithmetic and operand order while de-latencying each pass: warp-tree scores on an eight-token grid reproduce the strict reduction tree, an exact `fmaxf` block scan supplies the coefficient max trajectory with pointwise `expf` off the critical path and a prefetched serial denominator, and staged-tile values use clamped unconditional loads after a per-load select was shown to defeat memory-level parallelism (406 versus 122us). Named kernel medians at clean source, cache-only build: scores 33.5us, coefficients 14.3us, values 100.7us (VGPR 32/40/96, scratch 0) versus parent 399.7/176.1/533.0us; leaf route 1.154->0.179ms/layer, 6.45x, bit-exact; six-case off/on/off full logits/4-step/state/full-KV gate passes at committed source; canonical A/B 72 trajectories exact with p4096 weighted TG +14.705% (arithmetic-mean-rate ratio +16.516%). Evidence: `2026-09-08-framework-qwen4exp-qsa-ordered-v2-kernels.json`. |
| Qwen4Exp vision | `vision/qwen4_exp_vision.{hip,py}` | `vision_layernorm`, `vision_add_bias_residual`, `vision_gelu`, `vision_attention` | <=1K Qwen3-VL-compatible images/videos: merge-compatible RGB grids up to 256 patches/temporal pair, 2×2 block-major order, align-corners learned-position interpolation, frame-pair attention isolation, multiple images, odd-frame duplication, and typed placeholders. FP32 attention uses explicit vision H/W RoPE. Full 32×64 encoder matches Transformers at relative L2 1.48e-6/cosine 1.0; text QSA's registered MRoPE sibling applies interleaved T/H/W `[11,11,10]` and traces at 12,143 ns. Bounded PNG data URLs work through non-streaming chat; remote URLs/SSE/>1K remain open. |
| Qwen4Exp raw Q5_1 experts | `quant/qwen4_exp_q5_1.{hip,py}` | selected `linear`/`moe_linear` (`gguf_q5_1`) | Strict selected-expert consumer plus exact grouped rowbatch8 and grouped-WMMA down projections for the pinned Unsloth UD-Q4_K_XL mixed quant. Exact output8 scheduling cuts full-shape grouped-down CTAs 1,310,720→163,840 and paired natural 4K 237.131→222.228 s with exact logits/control; output1 remains fallback. The current short-prefill owner iterates 512 experts through 64 worker CTAs and uses 128 physical threads to materialize the same 256 logical partials before the original reduction tree; its p512 bucket is 3.470→2.534 s with exact bits. Q5_1 grouped WMMA is not the strict owner; explicit gfx1151 Qwen4Exp `production` selects it with cooperative Q4 gate/up on the definitive maximal suffix layers 27–47. Every layer 0–26 fails final-prompt mean or p95; the 27–47 450-row/three-repeat manifest passes at mean/p95/p99/max KL 1.05e-4/3.81e-4/1.52e-3/5.59e-3 and 99.556% top-1, improving the MoE-only p508/p1012 59.401→67.243 / 58.723→66.268 tok/s. The same explicit profile adds dense-Q8 WMMA on certified layers 32–47; the combined 450-row gate passes mean/p95/p99/max KL 1.20e-4/4.93e-4/1.72e-3/8.69e-3 and 99.778% top-1, reaching 73.361/71.834 tok/s. Exact grouped/coltile fallbacks remain registered. The strict selected decode default now uses 64 physical threads to materialize the same 256 logical partials before reconstructing the original shared strides 128/64/32 and wave32 tail. The first exact t128 contraction cuts Q5 cycle-wall 692.930→410.364 ms and graph decode 11.380→12.140 tok/s; t64 is BF16-bit exact to both registered t128/t256 fallbacks, cuts its matched Q5 trace 444.699→362.525 ms, and improves graph decode 13.077→13.302 tok/s (+1.69%). The c1 default also fuses selected down with routed weighted sum: one CTA per H=2560 output preserves every route BF16 result and the original ordered `fmaf`, removes 1,806 traced launches, contracts target cycle-wall 369.241→313.535 ms, and improves 13.379→13.523 tok/s (+1.06%); the separate exact chain remains fallback. A default-off 64-thread sibling improves warm decode further but is rejected for production mean/p95 KL (`0.002565/0.007202`). |
| Raw Q5_K/Q6_K/Q8_0 | `quant/gguf_k_gemv.{hip,py}` | `linear`, `linear_pair`, `attention_projection_quad` | Decode/prefill, BF16/F32 output, pair/quad launch contractions, rowbatch/coltile variants. The gfx1151 Qwen4Exp exact Q8/F32 owner first cut p508 26.264→14.718 s with coltile4/rowbatch8, then promotes coltile8/rowbatch4 alongside exact expert scheduling to reach 42.376 tok/s; p512 Q8 kernel wall falls 3.121→2.482 s with bit-exact full logits. Its c1 F32/F32 output-pack8 sibling reuses each activation across eight columns without changing per-output arithmetic, cuts the traced Q8 bucket 2.620→1.171 s and paired decode 5.698→6.305 tok/s; registered scalar raw Q8 remains fallback. gfx1151 Q5/Q6 W7900 policies remain disabled. |
| Q5_K/Q6_K selected prefill WMMA | `quant/gguf_k_selected_prefill.{hip,py}` | `moe_linear` | Raw-byte compact selected-MoE f16-WMMA consumers with strict raw selected-gemv fallbacks. The gfx1151 Qwen4Exp layer-2 Q5_K/Q5_K route is production-rejected/default-off: p508 Q5_K gate/up falls 279.86→16.66 ms and 20/20 category-balanced p512 pairs win by about 5%, but the complete 450-row gate fails prefill-last mean KL at 0.001179 > 0.001. Do not rescreen unchanged T2 arithmetic; the older optimized metadata-hoist sibling is a separate rejected path. |
| Raw Q3_K selected | `quant/gguf_q3_k_gemv.{hip,py}` | `moe_linear` | Q3 selected-expert projection family. |
| Q4_K pack8/raw | `quant/gguf_q4_k_gemv.{hip,py}` | `linear`, `linear_pair`, `linear_pair_silu`, `linear+residual` | Raw GGUF math and lossless pack8 layouts; pair/SiLU and exact rounded-BF16 residual composites where registered. Qwen4Exp c1 now resolves raw selected dual gate/up by registry capability, halves Q4 launches 94→47/token, and improves paired decode 6.065→6.223 tok/s. Its operation-complete sibling preserves both BF16 projection boundaries and the standalone SiLU/product bits, removes another 47 launches/token, and improves 6.400→6.420 tok/s. The selected default now maps logical lanes `tid`/`tid+64` onto 64 physical threads while publishing the same four strict wave sums; it contracts Q4 cycle-wall 1,076.767→814.906 ms across 1,974 launches and improves counterbalanced graph decode 12.003→13.167 tok/s (+8.84%). IDs/full logits are exact and the physical128 dual/singleton chains remain fallbacks. Above these kernels, gfx1151 now captures each complete stateless Qwen4Exp MoE chain in one self-validating request-owned graph: 48 captures/zero rejects, 192 full-logit rows exact, eager 6.511→11.515 tok/s, then exact Q5/Q4/Q5 contractions and Q5 down+weighted fusion reach 12.140/13.167/13.302/13.523 tok/s; c2 is exact and stateful GDN/QSA remain outside replay. Explicit gfx1151 `production` adds one-plane Q8_1 DP4A Q4 dual+SiLU on calibrated static layers `0,2,5,6,8,9,10,11,13–47`; measured-failing layers `1,3,4,7,12` remain exact. The physical64 owner preserves the candidate's 128 logical partials and BF16 boundaries; combined production passes 447/450 top-1 with mean/p95/p99/max KL 2.72e-4/1.40e-3/4.00e-3/5.77e-3, improves decode 13.880→15.543 tok/s, and contracts Q4 target cycle-wall 825.340→397.755 ms. Direct suffix13→calibrated43 is +0.37%. Suffix12 and all-layer DP4A are rejected at 445/450; exact logical128/t64 remains fallback and omitted-profile default. A Qwen4Exp one-layout expert replacement is rejected and removed: sampled layer-0 bits are exact and micro speed is 4.47x, but uncached load is 979 s and full-model mean/p95 KL fail at 0.002089/0.006529. Primitive projection+add fallbacks remain available. |
| Q4_K/Q6_K prefill WMMA | `quant/gguf_q4_k_prefill.{hip,py}` | `linear` | Resident pack8/raw prefill consumers; exact scalar/pack8 routes remain fallbacks. The p512 pack8-Q4 rounded-residual output-store sibling is rejected (0.958x core / 0.952x public complete-model prefill) and is not registered. |
| Q8_0 T16 prefill | `quant/gguf_q8_0_t16_prefill.{hip,py}` | `linear`, `linear_pair` | WMMA/T16 Q8 prefill and architecture-specific wave schedules. gfx1151 rows512/K1024/N16+N16 alpha/beta uses the exact two-wave dual owner; singleton WMMA remains the fallback. |
| Q8_0 T16 decode | `quant/gguf_q8_0_t16_gemv.{hip,py}` | `linear`, `linear_pair`, `linear_triple` | Exact T16 Q8 decode GEMV for Qwen3.5-family attention projections (in 2048; fused qkv 8192 + gate 4096). Per-row dual/split owners run at all widths, with an exact 128-thread dual-split rowtile col8 pair owner admitted at rows >= the backend-package floor `GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS`. |
| Q4/Q5/Q6 T16 selected | `quant/gguf_t16_selected_gemv.{hip,py}`, `quant/gguf_k_t16_selected_prefill.{hip,py}` | `linear`, `linear_pair_silu`, `moe_linear`, `moe_linear+weighted_sum`, `linear+residual` | c=1 and selected-prefill T16/qmicro/interleaved consumers, including weighted/residual composites. Exact one-wave/shared-B WMMA rowtile owners cover physical shapes, with grouped-grid siblings, fused dual+SiLU prefill owners, and input-F16 activation siblings (`*_fp16_in_bf16_out`) registered per backend; every shape/row miss and env-disabled path retains the strict one-wave/shared-B or primitive fallback, and current per-shape ownership is backend-package capability data. gfx1100/W7900 additionally retains the exact c8 Q4 selected gate/up pair-reuse dual owner through the package floor `GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS` (2026-09-05 audit packet D1: native-c8 +4.97% with exact repeatable trajectories, arm-identical state differentials on steady c2/c4/c8 and c8 shrink-sparse, natural-prompt duplicate-lane fraction 0.616 versus the direct-fixture 0.5, and a c8 census showing `q4_k_t16_selected_dual_pairreuse_direct_gemv_kernel` x80 with zero scalar fallbacks); the route's geometry gate pins it to x_rows=8/rows=64, lower widths and env-0 keep the per-row dual owner, and the selected-down/Q6-down pair-reuse packets stay unqualified on gfx1100 (floors 0). |
| Q6/Q4 mixed and narrow K/V grids | `fused/gguf_q6_q4_pair.{hip,py}` | `linear_pair` (standard-Q6+Q4, Q4, Q4+planar-Q6) | Exact block-parallel rows1 pairs; gfx1151 qualifies Qwen3.8 recurrent K5120/N10240+N6144 and full-attention K/V K5120/N1024+N1024 while primitive projections remain fallbacks. |
| Dense Q6_K T16/qmicro | `quant/gguf_q6_k_t16_gemv.{hip,py}` | `linear`, `linear+argmax`, `linear+residual` | Exact dense Q6 decode/prefill/root families. gfx1100 planar row8 uses the exact DPP reduction (VGPR136→112, bpermute320→0), admitted on all 55 actual-operation rows and retained by a 1.634% complete-owner wall win; rows1-7 keep the generic reduction. gfx1151 rows>=512 uses 128-thread/four-wave shared-weight WMMA for standard K5120/N10240 QKV (2.96-3.55x) and planar K17408/N5120 FFN-down (1.42-1.50x); both use 24 KiB LDS / 248 VGPR. Rows<512, narrow V, root, shape misses, and peer backends retain exact one-wave/16x16 primitives. |
| Dense planar-Q6 integer MMQ | `quant/gguf_q4_k_q8_1_selected_prefill.{hip,py}` | `activation_quant`, `linear` | gfx1151 production-profile T2 composite for rows17-48 on sole-resident planar K17408/N5120 down and K5120/N1024 narrow-V: session-owned BF16-to-Q8_1 packing feeding the integer `mmq64x64` consumer; exact A owners remain registered for strict/profile fallback. |
| Dense Q4 q8_1-dp4a VDR screen | `quant/gguf_q4_k_q8_1_dp4a_vdr_gemv.{hip,py}` | `linear` (leaf screen, not dispatched) | nasone32 k-quant load-reuse port (efa4e8641): subblock-hoisted metadata with activation packs reused across 8 columns, plus an unamortized control with identical thread mapping and f32 order (bit-exact RED contract). 2026-09-09 four-arm leaf on both gfx1100 cards: -50..-76% vs the control inside the dp4a class, but 1.11-1.44x slower than the retained T16 rows=1 owners at every production shape (T16 sits at the DRAM floor), so rejected as a decode replacement; retained as evidence for future integer decode routes. |
| Dense Q4 int-MMQ prefill screen | `quant/gguf_q4_k_q8_1_mmq_prefill.{hip,py}` | `linear` (leaf screen, not dispatched) | Raw-Q4_K x DS4-Q8_1 bulk-prefill integer MMQ (PP8192-attribution candidate): staged-dp4a 32x32-tile and direct-global iu8-WMMA 32x16-tile consumers, each with bit-exact ctl/vdr siblings (block-header/subblock-metadata hoisting) and a DS4 CPU oracle. 2026-09-09 W7900 six-arm leaf on real Qwen3.8-27B Q4_K_M weights, rows 512/1024/4096: best integer totals (pack + consumes) reach only 0.205-0.542x the retained float T16 prefill owners (9/9 case-rows), load-reuse deltas within +-3%, pack negligible — rejected as a bulk-prefill replacement; retained as leaf evidence for future integer prefill routes. |

| IQ2/IQ3/IQ4 decode | `quant/gguf_iq_gemv.{hip,py}` | `moe_linear` | Raw IQ selected-expert projection families. IQ3 tile4 remains scoped to the retained gfx1100 explicit-DFlash route; gfx1151 excludes it after a complete-route rejection and keeps tile1. |
| Q8_0 grouped down (P1) | `quant/gguf_q8_0_prefill.{hip,py}` | `moe_linear` | P1 device-driven grouped Q8_0 down owner (`gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out`) for the layer-2/4/30/46/47 Q8_0 expert-down family. Reads `expert_start` on device and iterates experts via a fixed worker grid, replacing the `group_expert_start` D2H copy + Python loop over 512 experts. BF16-exact to `gguf_q8_0_gemv` per grouped row (RED test `test_gpu_qwen4_exp_q8_0_grouped_down.py`). Strict per-expert selected gemv remains default; `HIPENGINE_QWEN4_EXP_Q8_0_GROUPED=1` selects it. **Perf-negative as of 2026-08-30** (microbench 20260830T202256): grouped owner ~3-12x slower than strict `selected_gemv` on layer-2 shape due to a 1.31M-block grid with OUT_BATCH=1 and no weight reuse; not promoted. |
| Q4/Q5/Q6 T16 selected | `quant/gguf_t16_selected_gemv.{hip,py}` | `linear`, `linear_pair_silu`, `moe_linear`, `moe_linear+weighted_sum`, `linear+residual` | c=1 and selected-prefill T16/qmicro/interleaved consumers, including weighted/residual composites. A Qwen4Exp one-layout replacement profile is rejected and removed: optimized p512 is neutral (213.52 vs 211.76 tok/s), paired decode regresses 5.925→3.615 tok/s, and mean/p95 KL fail at 0.003010/0.008338. gfx1151 Qwen3.8 standard-Q4 physical rows6/8/12/16 use the exact single-wave WMMA parent for K/N 5120/6144, 5120/10240, 5120/12288, and 6144/5120; narrow V, wide-K down, and misses retain shared-B. gfx1100 Qwen3.6 physical rows6 instead uses the C1-equivalent rowtile for K/N 5120/1024, 5120/6144, 5120/10240, 5120/12288, and 17408/5120, plus the exact single-wave parent for 5120/17408; all other rows/shapes keep explicitly registered shared-B. The same gfx1151 model's Q5 K6144/N5120, K17408/N5120, and K5120/N10240 rows2-8 use the exact col8 rowtile; registered parents remain strict fallbacks. |
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

### TimesFM path

| Functional family | Source / wrapper | Principal registry layers | Notes |
| --- | --- | --- | --- |
| Fused norm/elementwise | `timesfm/timesfm.{hip,py}` | rmsnorm (multiplicative scale, eps inside rsqrt), norm+add post-norm residual, bias, bias+swish, swish, add | Templated `<T>` `_f16`/`_f32` variants; FP16 storage with FP32 math. |
| RoPE + QK norm + scatter | `timesfm/timesfm.{hip,py}` | `timesfm_rope` (timescale table), `timesfm_qkv_norm_scatter` | One block per (b, n, h) head vector: in-kernel non-interleaved RoPE, query/key RMSNorm, per-dim softplus query scaling, head-major `[B, H, S, D]` k/v cache scatter, `[B, H, Q, D]` q transpose. |
| Masked softmax | `timesfm/timesfm.{hip,py}` | `timesfm_mask_softmax` | Register-resident two-pass row softmax; masked keys are `-INFINITY`, all-masked rows emit uniform 1/S (reference parity). |
| Attention | `timesfm/timesfm.{hip,py}` + rocBLAS `gemm_strided_batched` | `timesfm_attention` (strict FP32 fallback), `timesfm_flash_attention` (FP16 production) | Naive block-per-row kernel for the strict path; production path is a WMMA flash kernel (w32 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`): online softmax with width-16 shuffle row reductions, LDS-staged P, causal kv-tile skipping, uniform-1/S all-masked fallback, ragged-S bounds guards, 4 q-tiles/block for Q>=64. Attention sub-window: 0.73 ms/layer prefill at b8/ctx8192. |
| Head transpose | `timesfm/timesfm.{hip,py}` | `timesfm_transpose_heads` | `[B, H, Q, D]` back to row-major `[B*Q, H*D]` for the out projection; one block per row, half4-vectorized coalesced (6.8 us at b8/n=256). |

Model contract, loader, NumPy oracle, and GPU orchestration live in
`models/timesfm.py`, `loading/timesfm.py`, `kernels/cpu_reference/timesfm.py`,
and `runtime/timesfm_decode.py` respectively. The TimesFM RMSNorm is a
different contract from the Qwen family (multiplicative `scale`, no +1).

### TimesFM 3.0 path

Single non-autoregressive forward pass (no AR loop, no persistent KV cache);
sequence attention runs over `batch * variates` independent sequences with one
scratch cache pair reused across layers.

| Functional family | Source / wrapper | Principal registry layers | Notes |
| --- | --- | --- | --- |
| Variate attention | `timesfm3/timesfm3.{hip,py}` | `timesfm3_var_attention` | Non-causal attention across up to 32 variates, one block per (b, n, h) with per-warp query rows; per-(b, v) leading-mask counts exclude variate keys; scores x sqrt(head_dim); fully-masked query rows -> zeros (CPU SDPA semantics). FP16/FP32 templated. |
| QK norm + scatter (3.0) | `timesfm3/timesfm3.{hip,py}` | `timesfm3_qkv_norm_scatter_f16` | 2.5's fused kernel with runtime epsilon (torch `nn.RMSNorm` finfo(float32).eps, not 1e-6). |
| ReLU elementwise | `timesfm3/timesfm3.{hip,py}` | `timesfm3_relu` | 3.0 FFN/tokenizer activation (2.5 uses Swish). |
| Everything else | `timesfm/timesfm.{hip,py}` | reused 2.5 layers | rmsnorm/norm_add/bias/add (eps-parameterized), rope (absolute patch positions, host-supplied), flash attention, head rmsnorm/per-dim scale (also on var q/k via B=rows, N=1), scatter, transpose. The SDPA sqrt(head_dim) score scale is folded into the K-side norm weight; the 2.5 uniform fully-masked-row fallback differs from the oracle zeros only at leading-pad rows that decode() slices away (verified end-to-end on all fixtures). |

Model contract, loader, NumPy oracle, GPU orchestration, and bench live in
`models/timesfm3.py`, `loading/timesfm3.py`, `kernels/cpu_reference/timesfm3.py`,
`runtime/timesfm3_decode.py`, and `scripts/timesfm3_gpu_bench.py`; the
per-model record is `docs/MODEL-TIMESFM3.md`.

### Speculative decoding path

| Functional family | Source / wrapper | Principal registry layers/quants | Notes |
| --- | --- | --- | --- |
| DFlash drafter | `speculative/dflash_drafter.{hip,py}` | `dflash_*` projection, norm, attention, activation, metadata layers (`w4_paro`) | Raw-pointer drafter primitives; target verification remains transaction-shaped. |
| DFlash2 drafter reference | `speculative/dflash2_drafter.py` + `cpu_reference/dflash2.py` | `dflash2_grouped_conv`, `dflash2_selector`, `dflash2_selector_path`, `dflash2_attention_forward`, `dflash2_rope_tables` (`fp32`) | Torch-free NumPy DFlash2 exactness reference (grouped dynamic conv, top-16 bilinear selector, q/k-norm sliding attention). Golden fixtures from z-lab/dflash @ 07ebd93; native kernels land in D2. Source lineage: `docs/source_lineage.json` (repo `dflash`). |
| DFlash2 native kernels | `speculative/dflash2.{hip,py}` | `dflash2_grouped_conv`, `dflash2_top16_rows`, `dflash2_selector` (`bf16`/`fp32`) | Native grouped dynamic conv (strided side views over the 1280-wide projection), top-16 logits, and the low-rank bilinear candidate-selector greedy walk. Strict RED vs `cpu_reference/dflash2.py` (BF16 round-trip modeled); registered for `hip_gfx1100` + `hip_gfx1151`. D2a. |
| DFlash acceptance | `speculative/dflash_accept.{hip,py}` | `dflash_accept_chain`, `speculative_accept_commit` | GGUF/PARO acceptance and bounded commit summaries. |
| DFlash commit/state | `speculative/dflash_commit.{hip,py}` | `dflash_commit_chain`, `linear_state_pair_*` | Transactional selected-state and cursor commit helpers. gfx1100 target verification reads initial Conv/GDN state from a resident multi-slot slab with the strict chunked pointer-table import as rollback; gfx1151 keeps the packed-state route, and a per-layer HIP D2D chain remains a lower strict fallback on gfx1100. |
| MTP core | `speculative/mtp.{hip,py}` | MTP norm/fuse/router/top-k/gate/finalize/route accumulation | Provider-neutral proposal/acceptance primitives. Dense H5120 Q4_K_M gfx1100 native C1 verification uses allocated cache capacity (BF16 KV/FP32 state); scalarized rows snapshot initial Conv/GDN state before mutation. Native graph metadata is independent of bulk-prefill metadata thresholds; topology transitions may select eager native execution. |
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
