"""gfx1151 / Strix Halo backend registration.

The initial gfx1151 backend intentionally reuses the proven gfx11 HIP kernel
bodies from ``hip_gfx1100`` and compiles them as native ``gfx1151`` code objects
through ``HIPENGINE_HIP_ARCH=gfx1151`` / ``--offload-arch=gfx1151``.  This gives
Strix Halo a peer backend key while keeping tuning changes separate from the
source-lineage port.
"""

from __future__ import annotations

from importlib import import_module

from hipengine.kernels.backends import hip_target_arch_for_backend
from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
    laguna_global_f16_projection_head_kv_nontemporal_tile2_bf16_spans,
    laguna_swa_f16_projection_head_kv_nontemporal_tile2_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans,
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_logits_bf16_f32w_auto_256,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
    gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
    gguf_q4_k_pack8_wmma_prefill_gfx1151_bf16_bf16_out,
    gguf_q6_k_wmma_prefill_16x32_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    gguf_q4_k_pack8_dual_prefill_bf16_bf16_out,
    gguf_q4_k_pack8_dual_silu_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    gguf_q4_k_t16_dense_dual_interleaved_tile2_local32_silu_bf16_bf16_out,
    gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out,
)
from hipengine.kernels.registry import (
    KernelKey,
    is_registered,
    register,
    registered_keys,
    resolve,
)

BACKEND = "hip_gfx1151"
TARGET_ARCH = hip_target_arch_for_backend(BACKEND)
# Clean AR-O2 three-repeat category/quality gates admit compensated source-F16
# WMMA only for SWA QKV/gate/O from 16 rows. Full-attention layers and M2-15
# retain the exact LPF-1 tile; decode retains the separately registered GEMV.
LAGUNA_F16_PREFILL_STRATEGY = "wmma_comp_swa"
LAGUNA_F16_PREFILL_MIN_ROWS = 16
# Exact rows==1 source-F16 single/triple siblings keep the local256 grid and
# reduction order while removing the generic reducer's second broadcast
# barrier. All six natural roles improve at the gfx1151 leaf; exact production
# A/B and cache-only tracing admit automatic selection.
LAGUNA_F16_DECODE_ONEBARRIER = True
# Compile-time K3072/K6144/K9216 specializations retain the one-barrier
# arithmetic/grid while removing dynamic loop/address machinery. Seven exact
# p512/d128 pairs and cache-only tracing admit them over the generic
# one-barrier owner; the latter remains the explicit rollback.
LAGUNA_F16_DECODE_FIXEDK = True
# Exact c=1 source-F16 Q/K/V/gate launch contraction preserves every
# output column's fixed-K reduction order. Seven same-resident p512/d128 pairs
# are exact and all positive; triple-plus-single remains the unfused fallback.
LAGUNA_F16_ATTENTION_QUAD_DECODE = True
# Exact last-producer projection/head/KV fusion keeps every fixed-K dot and
# head RMSNorm/RoPE association, removes another 48 launches/token, and is
# byte-exact on global/SWA state. The seven-pair gate is throughput-flat but
# mechanically positive; the quad plus registered head/KV remains rollback.
LAGUNA_F16_PROJECTION_HEAD_KV_DECODE = True
# Exact last-producer attention-output fusion preserves the fixed-K projection
# and standalone local256 add/RMSNorm trees while removing 48 launches/token.
# Seven exact same-resident pairs are all positive.
LAGUNA_F16_OUTPUT_ADD_RMSNORM_DECODE = True
# Exact source-F16 decode streams each weight once. Cache-bypassing loads cut
# the complete six-role leaf family 3.36%, and all seven same-resident
# p512/d128 pairs improve without changing arithmetic or resident bytes.
# Explicit false and peer backends retain the cached-load owner.
LAGUNA_F16_NONTEMPORAL_DECODE = True
# The greedy argmax winner also owns the next embedding token and scratch/KV
# positions. The following exact autoregressive step validates the host token
# and consumes those device-published scalars without three synchronous H2D
# copies; forced-token and peer-backend paths retain ordinary publication.
LAGUNA_ARGMAX_CONTROL_PUBLISH = True
# Exact Q6T16 LM-head producer tile maxima remove the full-logit argmax scan
# and one model launch. The actual-weight leaf, seven-pair p512/d128 gate, and
# cache-only trace admit it on gfx1151; constructor false keeps exact rollback.
LAGUNA_Q6_T16_LM_HEAD_TOP1_STAGE1 = True
# Exact c=1 router projection wave-level reduction. Seven same-resident
# p512/d128 pairs are exact and win 6/7; the scalar local256 tree remains the
# registered rollback.
LAGUNA_ROUTER_PROJECTION_WAVE0_TREE = True
# gfx11 any-order dispatch lets the exact router projection wait on the ordered
# output/add/RMSNorm producer without holding the queue at that boundary.
LAGUNA_OUTPUT_ROUTER_ANYORDER_DECODE_AVAILABLE = True
LAGUNA_OUTPUT_ROUTER_ANYORDER_DECODE = True
# Exact D9 wave-0 RMS reduction preserves the local256 partials and complete
# FP32 addition tree while removing seven workgroup barriers. The gfx1151
# production-shape leaf improves 3.66%; the scalar tree remains rollback.
LAGUNA_MOE_TAIL_WAVE0_TREE = True
# Exact K3072/N1024 gate/up and K1024/N3072 down siblings preserve the
# production local128 grid and reduction order while compile-time-specializing
# only Laguna's c=1/top-10 shape. All three actual-weight roles improve, and
# seven exact p512/d128 pairs admit the combined owner.
LAGUNA_SELECTED_NATURAL_DECODE = True
# The exact selected-down sibling distributes the final 16 ordered wave sums
# across lanes 0..15. All seven resident p512/d128 pairs are exact and
# positive; the serial owner remains registered rollback.
LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_DECODE = True
# Exact tile-local completion preserves all ten route-parallel weight streams,
# every BF16 projection boundary, and the slot-order weighted FMA chain while
# removing the standalone reducer launch. Seven same-resident pairs all win.
LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_WEIGHTED_DECODE = True
# Reuse each adjacent Q4 selected-down nibble byte and aligned coefficient
# pair. Exact p512/d128 wins 5/7 pairs at +0.1365%; explicit false retains the
# preceding scalar-payload weighted route for rollback.
LAGUNA_SELECTED_DOWN_Q4_PAIRCOEFF_WEIGHTED_DECODE = True
# Preserve the production Q4/Q6 shared-down grids and D9 local256 tree while
# enqueueing both unchanged launch wrappers inside one native host call. Seven
# exact same-resident p512/d128 pairs win 7/7; peer backends retain separate
# Python calls until independently measured.
LAGUNA_SHARED_DOWN_MOE_TAIL_HOST_BATCH = True
# Exact gate/up owner that splits each resident T16 tile across two 8-column
# workgroups, halving live accumulators. The actual-weight leaf improves
# 5.35-7.13%; seven exact p512/d128 pairs are all positive.
LAGUNA_SELECTED_NATURAL_TILE8_DECODE = True
# The exact tile8 parallel-tail sibling preserves every column's arithmetic.
# All seven resident p512/d128 pairs are exact and positive.
LAGUNA_SELECTED_NATURAL_TILE8_PARALLEL_DECODE = True
# Promoted exact fusion: the qualified parallel tile8 owner materializes the
# BF16 SiLU intermediate directly; all seven resident pairs are positive.
LAGUNA_SELECTED_NATURAL_TILE8_PARALLEL_SILU_DECODE = True
# Convert BF16 activations and dequantized Q4 gate/up pairs to FP16 in
# registers, then use gfx1151's native adjacent-K dot2 with FP32 accumulation.
# The byte-neutral interleaved resident layout is unchanged. The actual-weight
# leaf improves 17.99%, all seven p512/d128 pairs improve, and recurrent
# candidate-vs-exact quality stays at max KL 0.00820 / top-1 93.75%.
# Constructor false retains the exact pair-coefficient owner.
LAGUNA_SELECTED_HALFDOT_DECODE = True
# Exact dense/shared Q4 pair fusion preserves the two BF16 projection
# boundaries in registers before applying the existing SiLU-product
# expression. Seven resident p512/d128 pairs are exact and all positive.
LAGUNA_Q4_PACK8_DUAL_SILU_DECODE = True
# Exact decode-only T16 consumers stream compact Q4 coefficients beside the
# retained pack8 prefill layout. Actual shared/dense leaves are BF16 bit-exact
# and improve 19.0%/60.2%; admission accounts for the bounded auxiliary bytes.
LAGUNA_Q4_DENSE_DECODE_T16_SIDECAR = True
# Exact byte-neutral gate/up interleave plus two-column wave ownership cuts
# the actual dense/shared leaf family by 6.50% before the resident gate.
LAGUNA_Q4_DENSE_DECODE_T16_DUAL_INTERLEAVED = True
# Exact standalone Q4T16 shared-down owner streams 22.9% fewer resident bytes
# than expanded pack8 and is bit-identical across all 24 actual matrices.
LAGUNA_Q4_SHARED_DOWN_T16_DECODE = True
# The selected gate/up decoder and the production D8 prefill owner now share
# one exact byte-neutral paired T16 allocation. Natural prefill leaves improve
# at M55+ and are effectively flat at M32; c=1 decode improves 5.70%.
LAGUNA_Q4_EXPERT_T16_DUAL_INTERLEAVED = True
# Clean post-350 repeated M512/M1024/M2048 timing and full-logit quality admit
# 2048-row projection/MoE transactions while attention and physical KV writes
# remain independently tiled at 128. M2048 is byte-identical at pp512, keeps
# top-1 at 512/1K/4K, and has max relative KL 1.25e-5 versus M512.
# Other backends retain the 128-row runtime fallback until measured independently.
LAGUNA_PREFILL_MATRIX_ROWS = 2048
# The post-350 LAP-7 screen reuses each streamed BF16 K/V row across four
# adjacent queries. It is byte-identical to the admitted online-qrow2 arithmetic
# on the wrap/eviction oracle and improves matched pp512 production by 3.23%.
# Qrow2/exact variants remain explicit rollback; unmeasured backends are unchanged.
LAGUNA_GLOBAL_PREFILL_VARIANT = "global_context_rows_qrow4_m128_online_spans"
LAGUNA_SWA_PREFILL_VARIANT = "swa_context_rows_qrow4_m128_online_spans"
# WPF-H5M source-qualified exact qrow4 ownership is W7900-only pending an
# independent gfx1151 exactness/resource/performance gate.
LAGUNA_SWA_PREFILL_ROLE_VARIANTS = {}
# Exact pre-append scheduling lets complete M128 global tiles and pre-wrap SWA
# tiles consume one BF16 cache source. Wrapped SWA, residual rows, verifier
# transactions, and other backends retain attend-then-append.
LAGUNA_PREFILL_KV_PREAPPEND = True
# Once a safe tile is pre-appended, complete KVLiveSpans metadata can decide
# visibility without selecting between current-row and cache sources inside
# the dot-product loop. Measured admission keeps the slower global start-0
# slice on the source-qualified cached kernel while enabling metadata-only SWA
# and global tiles beginning at position 128.
LAGUNA_PREFILL_CACHED_META = True
# Exact global-only qrow6 reuses each streamed BF16 K/V row across six adjacent
# queries. Leaf admission is limited to complete preappended global M128 tiles
# beginning at position 128; global start 0 and every SWA tile retain qrow4.
LAGUNA_PREFILL_GLOBAL_QROW6 = True
# Complete initial no-wrap preappended tiles have identity token positions and
# no eviction. The separately registered dense-initial attention variants
# preserve the full KVLiveSpans ABI while skipping per-token metadata loads.
# Partial, wrapped, explicitly evicted, and verifier routes retain exact
# cached-metadata/current-source fallbacks.
LAGUNA_PREFILL_DENSE_INITIAL = True
# Dense-initial M128 tiles beginning at position 128 widen the resident BF16
# K/V prefix exactly once, then use zero-workspace F32 hipBLASLt QK/PV
# contractions around a KVLiveSpans-qualified causal softmax. The complete
# pp512 route wins 6/7 paired runs (602.52 versus 576.08 tok/s median), keeps
# top-1 2930, and lowers all-exact full-logit KL from 0.003246 to 0.002214.
# Start-0, partial, wrapped, evicted, verifier, and decode routes stay on the
# established attention kernels.
LAGUNA_PREFILL_ATTENTION_HIPBLASLT = True
# Packing the 4.7-MB query/output tile into head-major order allows one
# eight-way wide QK and one wide PV batch without replicating K/V. It improves
# the qualified 48-layer leaf model 5.08% and the seven-pair pp512 median
# 0.87%, while all-exact KL improves from 0.002214 to 0.002097.
LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERIES = True
# Write the three qualified M128 query tiles directly in head-major order from
# the fused RMSNorm/RoPE producer. This removes 144 standalone query-transpose
# launches at pp512; eleven complete-state pairs improve the median 0.532% and
# every token/logit/hidden/KV hash remains exact.
LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERY_PRODUCER = True
# One wave32 owns one causal-score row, replacing the former 256-thread
# block reduction and its LDS barriers. The qualified 48-layer attention leaf
# improves 13.72%; paired pp512 improves 0.574% and all-exact KL falls from
# 0.002097 to 0.001796.
LAGUNA_PREFILL_ATTENTION_HIPBLASLT_WAVE_ROWS_SOFTMAX = True
# Keep the three qualified library-attention output tiles in their native
# head-major order and consume that mixed layout directly in the exact
# softplus gate. This removes 144 standalone output-transpose launches at
# pp512. Eleven complete-state pairs improve the median 0.338%; the stronger
# admission signal is the exact removal of the traced transpose sub-window.
LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_OUTPUT_GATE = True
# Contexts above 4K route only the 12 global-attention layers through a
# capacity-sized 48-head packed-F32 owner. Same-session complete-model gates
# preserve the 4K path and improve 16K/64K/128K by 7.93%/16.94%/22.09%;
# SWA, decode, partial, wrapped, evicted, and verifier paths remain unchanged.
LAGUNA_PREFILL_LONG_ATTENTION_HIPBLASLT = True
# Exact 4K key blocks carry online row max/sum/output state across tensorized
# QK/PV calls. This cuts 128K scratch 4.298 GB -> 143.753 MB and improves the
# complete-model long route another 12.52% while preserving its >4K gate.
LAGUNA_PREFILL_BLOCK_ATTENTION_HIPBLASLT = True
# Dense-initial global cache blocks are allocated in identity physical order.
# Direct addressing removes per-element span checks/remaps from the exact 4K
# BF16 widen; the full KVLiveSpans route remains the rollback/fallback.
LAGUNA_PREFILL_DENSE_CONTIGUOUS_CACHE = True
# Reuse each exact 4K global K/V block across a complete M2048 matrix chunk.
# SWA and partial matrix tails remain on the independently retained M128
# routes. LC-3 complete-model gates improve 4K/16K/64K/128K by
# 4.89%/20.48%/39.11%/44.59%.
LAGUNA_PREFILL_GLOBAL_ATTENTION_ROWS = 2_048
# Rolling M128 SWA gathers the exact 511 historical BF16 ring rows plus 128
# current BF16-rounded rows into one 639-key tensorized QK/PV union. The
# complete 4K/16K/64K/128K gate improves every shape and remains bounded.
LAGUNA_PREFILL_SWA_ATTENTION_HIPBLASLT = True
# Clean LAP-3/LAP-4 full-category admission quantizes gate/up in same-byte
# 16-value groups and uses the resident-T16 128x32 integer-dot consumer.
# The post-350 wave-column screen keeps row-vector D8 activation staging, maps
# one 32-column output slice to each wave, holds decoded T16 weights in
# registers, and ping-pongs the activation tile to remove one barrier per K32.
# For chunks of at least 512 rows, prefetch the next K32's eight raw nibble
# words while current packed dots execute; smaller chunks keep the rollback.
# Packed-dot arithmetic and K order remain bit-for-bit unchanged.
# Other backends retain exact.
LAGUNA_SELECTED_GATE_UP_MODE = (
    "mmq128x32_d8_f32_wavecols_direct_doublebuf_rawprefetch_ge512"
)
# Exact eight-token router tiling preserves every token/expert's K traversal
# and reduction tree while reusing each F32 weight row twice as long.
LAGUNA_ROUTER_LOGITS_MODE = "token_tile_8"
# The post-350 down screen maps Q4 output columns across two wave32s and lets
# the Q6 row-vector consumer reuse one decoded tile across 64 routed rows.
# Range-safe D4 resident-T16 integer-dot arithmetic is unchanged; 32-row Q6,
# scalar-staged, and exact routes remain rollbacks. At producer rows >=512,
# Q4 down also carries the next K32 raw nibble payload in registers.
LAGUNA_SELECTED_DOWN_MODE = (
    "mmq64x64_d4_f32_q6_wavecols_direct_rawprefetch_q4_ge512"
)
# Exact scratch reuse writes packed gate/up into the larger selected-down
# output allocation, then folds the standalone BF16 SiLU boundary into the
# range-safe down pack. Seven paired pp512 runs are exact and win 7/7; the
# standalone SiLU plus ordinary pack remain the explicit rollback chain.
LAGUNA_FUSED_SELECTED_SILU_PACK = True
# Byte-neutral Q6 qmicro keeps the resident T16 metadata but groups each
# four-column K4 quant quartet into one aligned 12-byte record. Exact c1 and
# selected-prefill gates both improve on gfx1151; peer backends retain legacy
# Q6 T16 bytes until independently measured.
LAGUNA_Q6_QMICRO = True
# Q6 has no minimum term, so selected-down never consumes the Q8_1 activation
# sum metadata. The compact activation tile also narrows each bounded K16
# quant sum to int16, reducing the production 64-row kernel's LDS footprint
# from 5,632 to 5,120 bytes without changing dot or accumulation order.
LAGUNA_Q6_COMPACT_ACTIVATION = True
# Split each compact Q6 activation row across two threads so all 128 threads
# stage one 16-byte half and one K16 quant sum. The exact all-layer screen
# improves 21/23 real Q6 layers without changing resources or output bytes.
LAGUNA_Q6_HALF_ROW_ACTIVATION = True
# Padded rows are never consumed by the guarded dot/store loops. Avoid writing
# zero Q8 bytes and recomputing zero K16 sums for those slots.
LAGUNA_Q6_SKIP_PADDED_ACTIVATION = True
# Two byte-permute gathers replace the scalar four-column qmicro unpack while
# preserving the byte-neutral resident layout and exact integer-dot order.
# The actual-weight leaf improves 2.67%, the complete model wins 5/7 matched
# pairs, and cached tracing reduces the selected Q6 body with no spills.
LAGUNA_Q6_QMICRO_PERMUTE = True
# Byte-neutral planar qmicro removes two prefill byte gathers and lowers exact
# decode register pressure. The actual leaf wins, while two owner-order
# full-model blocks are aggregate-neutral with complete state exact.
LAGUNA_Q6_QMICRO_PLANAR = True
# Prefetch the next planar-qmicro weight record into registers while the
# current integer-WMMA fragment consumes LDS, then recycle the same 5,120-byte
# shared tile. This preserves every Q6 dot/FP32 boundary and adds no sidecar.
LAGUNA_Q6_WMMA_PREFETCH_WEIGHT = True
# Pipeline the next compact Q8 activation half-row beside the retained next
# weight record. The current K32 WMMA hides the global read, and the following
# iteration publishes the exact bytes into the unchanged shared activation
# tile. The complete-state pp512 A/B is exact and improves the median.
LAGUNA_Q6_WMMA_PREFETCH_ACTIVATION = True
# Reuse Q6's otherwise-unused D4 sum field for two exact K16 quant sums.
# Computing them once in the packer removes repeated sum dots from every
# selected-down output-column workgroup without changing activation bytes.
LAGUNA_Q6_PRECOMPUTED_ACTIVATION_SUMS = True
# D8 gate/up stores exact K16 activation sums in a bounded scratch sidecar so
# each output-column workgroup can skip rebuilding them with integer dots.
# Resident weights, D8 bytes/scales, arithmetic, and BF16 output are unchanged.
LAGUNA_Q4_PRECOMPUTED_ACTIVATION_SUMS = True
# Stable expert-major count/prefix/scatter uses one workgroup per expert instead
# of one workgroup serially scanning all 5,120 routed lanes twice.
LAGUNA_MOE_GROUP_COMPACT_MODE = "parallel"
# The always-on shared expert is independent of router selection and routed
# gate/up/down until the final combine. A nonblocking secondary stream plus
# two dependency events overlaps 99.16% of its measured pp512 kernel time;
# complete-state A/B is exact and wins all seven queue-matched pairs.
LAGUNA_MOE_BRANCH_CONCURRENCY = True
# The same exact event schedule is profitable at c=1 once the specialized
# T16 selected and shared decode kernels are active. Seven counterbalanced
# p512/d128 pairs win 7/7, and tracing observes 94 secondary kernels/token.
LAGUNA_MOE_DECODE_BRANCH_CONCURRENCY = True
# Protect router logits/selection before releasing the concurrent shared
# branch. Matched complete-state pp512 is +0.073% with 5/7 wins, and cached
# tracing verifies a 0.310-ms kernel-span reduction.
LAGUNA_MOE_SHARED_AFTER_ROUTER = True
# gfx1151 exposes least/greatest HIP stream priorities +1/-1. Running the
# after-router shared branch at +1 improves exact pp512 0.494% (6/7 wins) and
# cuts cached kernel span 7.255 ms while keeping 99.75% of shared work hidden.
LAGUNA_MOE_SHARED_LOW_PRIORITY = True
# Decode benefits from equal scheduling priority after its exact selected and
# shared paths became closely balanced. Keep prefill at +1, but use a separate
# priority-0 shared stream for c=1; the same-session gate wins all seven pairs.
LAGUNA_MOE_DECODE_SHARED_NORMAL_PRIORITY = True
# Clean LAP-5 admission selects resident pack8-Q4/raw-Q6 64x16 WMMA consumers
# for dense/shared rows while preserving the exact low-row fallback.
LAGUNA_DENSE_Q4_PREFILL_MODE = "wmma_pack8"
# The attention-RMSNorm source range is statically bounded from resident F32
# norm weights, so Q/K/V/gate use direct BF16-to-FP16 and omit identity output
# restores. Attention output retains power-of-two row scaling; decode is
# unchanged.
LAGUNA_F16_PREFILL_MODE = "hipblaslt_range_direct"
# Exact producer-boundary variants write FP16(BF16(value)) directly from the
# attention RMSNorm and softplus-gate kernels. This removes the two standalone
# BF16-to-FP16 casts per layer while preserving the established source-F16
# input bits; the runtime setter remains the explicit rollback.
LAGUNA_F16_BOUNDARY_FUSION = True
# The gfx1100 current-P4 body is shape-identical for Laguna S 2.1 and compiles
# from the shared gfx11 source as a native gfx1151 code object. The
# architecture-local bit-exact and p512/d128 gates admit automatic selection;
# explicit False retains the registered fallback chain for rollback.
LAGUNA_HEAD_KV_FUSION = True
# The complete gfx11 exact split-attention bundle wins the clean gfx1151
# p512/d128 gate. Keep the thresholds and reducer capabilities inseparable:
# explicit use_split_attention=False retains serial global/SWA attention.
LAGUNA_GLOBAL_SPLIT_MIN_LIVE = 127
# The exact natural 48Q/8KV/D128/capacity-4096 reducer preserves the retained
# dynamic-live score ABI and local256 arithmetic. Three production live points
# and seven exact p512/d128 pairs admit it on gfx1151 only.
LAGUNA_GLOBAL_SPLIT_FIXEDSHAPE_REDUCE = True
# Above the fused kernel's LDS-resident score limit, replace the per-query score
# producer and scalar sixfold-V reducer with an exact GQA6 producer, exp32
# normalizer, and dimension-sharded shared-V owner. The live16,448 leaf is
# byte-exact and 80.94% faster than the generic split path.
LAGUNA_GLOBAL_SPLIT_GQA6_DIM32_VSTAGE64 = True
# Defer the exact normalization product into each shared-probability load,
# removing one full score-plane pass without changing its F32 bits.
LAGUNA_GLOBAL_SPLIT_GQA6_DEFERREDNORM_DIM32_VSTAGE64 = True
# Keep six query vectors resident while one exact wave scores four consecutive
# KV tokens. The natural-depth leaf is byte-exact and 8.6-9.6% faster from
# 4K through 128K on gfx1151.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE64 = True
# V80 keeps the same chronological FMA chain while reducing exact-PV staging
# barriers by 20%. It wins all four natural depths; V92/V96 lose at long
# context, so the V64 specialization remains the explicit rollback.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80 = True
# Dense initial caches do not need the physical-slot plane in the exact V80
# owner. Preserve the complete spans ABI and fall back after explicit eviction.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX = True
# Keep the old cache-policy crossover for the prefetch4 rollback. Deeper
# operand-prefetch owners hide the bypass latency and select combined
# non-temporal K/V across the dense V80 band above the 6,000-live fused owner.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX_NONTEMPORAL_MIN_LIVE = 65_536
# Once the long-context cache bypass is active, bypass K as well as V. The
# incremental exact leaf wins 3.26-3.58% at the admitted 64K/128K shapes.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE = True
# Deeper ordered operand prefetch becomes a 6.84-6.94% exact leaf win after
# non-temporal K/V increases the latency each output wave must cover.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_PREFETCH8_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE = True
# Prefetch16 preserves the chronological FMA chain and wins a further
# 1.69-2.25% over prefetch8 from 4K through 128K on gfx1151.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE = True
# V128 keeps two 512-thread workgroups resident while reducing exact-PV stage
# rounds by 37.5%. It wins 2.15-2.37% over V80 at every natural long depth;
# V160 loses the gain at 64K/128K and is not retained.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE128_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE = True
# Vectorize contiguous FP32 probability loads/stores into the shared V128
# stage. This remains byte-exact and improves every natural long depth.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE128_PROBABILITY_VEC4_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE = True
LAGUNA_GLOBAL_SPLIT_GQA6_ALLWAVE_TILE1024_DENSE_PREFIX = True
LAGUNA_GLOBAL_SPLIT_GQA6_ALLWAVE_TILE1024_DENSE_PREFIX_MIN_LIVE = 32_768
# Ordered-prefetch4 moved the exact/context-PV crossover above 64K. Keep the
# quality-scoped layer schedule only at the measured deep-context band; shorter
# long contexts use exact deferred-normalization GQA6.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_MIN_LAYER = 32
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_COMPENSATED_LAYER = 28
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_MIN_LIVE = 98_304
# Share each staged probability/value tile across two output-dimension waves.
# D64 is byte-identical to D32 and reduces the active long-context leaf by
# 17.3%/10.7%/10.9% at 16K/64K/128K on gfx1151.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DIM_TILE = 64
# The quality-scoped D64 context-split owners reuse the same exact score plane
# as the V80 path. Four-token ownership cuts their score grid by 4x while the
# admitted partial-PV/merge arithmetic remains unchanged.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_TOKENLOOP4 = True
# Keep the exact exp32 reciprocal but apply it in the context-PV probability
# loader, avoiding one normalized-score write/read pass.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DEFERREDNORM = True
# Dense initial caches have identity logical-to-physical metadata. Let the
# long-context token-loop owner bypass those metadata streams while retaining
# the full KVLiveSpans ABI and the generic fallback after any explicit eviction.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_SCORE = True
# The deep ctx4096 route streams each K/V line once. Bypass the cache only for
# that dense-prefix owner; the formal 128K leaf wins 8.47% ordinary and 6.13%
# compensated without changing F32 or BF16 output bits.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE = True
# Ordinary PV reaches its 128K minimum with four ordered operands in flight;
# deeper prefetch is tied but consumes more registers.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE_PREFETCH = 4
# Kahan PV has a longer dependent chain and continues improving through p16.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_COMPENSATED_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE_PREFETCH = 16
# Reuse the exact 1,024-token all-wave QK owner in the five quality-scoped
# ctx4096 layers. The partial-PV and merge association remain unchanged; the
# serial formal leaf improves 128K by 2.16% ordinary / 2.07% compensated.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_ALLWAVE_SCORE = True
# The exact dynamic-scan fused one-head owner keeps all 48 workgroups and
# removes the score plane/launch while preserving reduction association.
LAGUNA_GLOBAL_FUSED_FIXEDSHAPE = True
# Pair adjacent global query heads and reuse each staged 64-slot V tile.
# The exact natural-live leaf and seven resident-model pairs admit it.
LAGUNA_GLOBAL_GQA2_VSTAGE64_FIXEDSHAPE = True
# Preserve global GQA2 arithmetic while widening each padded V-stage copy to
# one aligned 16-byte transaction.
LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_FIXEDSHAPE = True
# Avoid the compiler-generated 32-byte private scratch aggregate used by the
# retained vec16 copy and write each valid vector directly into the V tile.
LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_DIRECT_FIXEDSHAPE = True
# Exact score-domain sibling passes the seven-pair resident-model wall gate.
LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Issue each wave's independent exact global-softmax exponentials across
# wave32 while retaining lane-0 token-order summation.
LAGUNA_GLOBAL_GQA2_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Raise the exact global-attention grid from 24 to 32 workgroups by assigning
# each 6-query GQA group as 2+2+1+1 owners. Singleton-owner idle waves retain
# staged-V barrier participation while active heads preserve every operation.
LAGUNA_GLOBAL_MIXED32_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Exact score-producer maxima remove the global score reread and one barrier.
# Seven resident p512/d128 pairs admit the qualified gfx1151 route.
LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Preserve the producer-max QK tree while replacing ds_bpermute transport with
# permlanex16 plus DPP moves. Seven resident pairs admit the exact sibling.
LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Exact aligned float4 replay of the normalized global probability plane.
# All seven resident p512/d128 pairs win with exact generated state.
LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Normalize each global probability once in LDS before the exact PV replay.
# Seven resident p512/d128 pairs admit the exact gfx1151 sibling.
LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Preserve the local256 eight-wave denominator tree while all sixteen waves
# share independent QK and value transport. Every natural global leaf wins.
LAGUNA_GLOBAL_MIXED32_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Use all 40 gfx1151 CUs by assigning each six-query GQA group as
# 2+1+1+1+1 exact owners. All three natural global leaves and all seven
# resident p512/d128 model pairs win against mixed32-local512.
LAGUNA_GLOBAL_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Sequential global decode before an explicit eviction has a visible identity
# prefix. Compile out token/base/eviction metadata and the physical-slot LDS
# plane while preserving the exact mixed40 arithmetic and the full span ABI.
# All natural leaves and seven resident p512/d128 pairs win byte-exactly.
LAGUNA_GLOBAL_DENSE_PREFIX = True
# Overlap the next dense-prefix global V64 load with exact scalar PV work by
# assigning it to otherwise-idle query-owner waves. All natural leaves and
# seven resident p512/d128 pairs win byte-exactly.
LAGUNA_GLOBAL_DENSE_PREFIX_IDLE_DOUBLE_BUFFER = True
# Preserve the eight-wave denominator while widening dense-prefix QK/value
# transport. All seven exact same-resident p512/d128 candidate runs win.
LAGUNA_GLOBAL_LOCAL1024 = True
LAGUNA_SWA_SPLIT_MIN_LIVE = 65
LAGUNA_SWA_SPLIT_TILE16_MIN_LIVE = 257
LAGUNA_SPLIT_GATE_FUSION = True
LAGUNA_SWA_SPLIT_WAVE_LOCAL = True
# Exact GQA3 score ownership reuses each streamed SWA K row across three query
# heads while retaining the 72-head score plane and wave-local value reducer.
# The 24 x token/tile grid preserves gfx1151 breadth; peer backends keep the
# one-query score owners until independently measured.
LAGUNA_SWA_SPLIT_GQA3_SCORES = True
# The saturated 512-slot SWA reducer preserves the retained 72-workgroup /
# 288-wave grid and every scalar/FMA operation while specializing the natural
# 72Q/8KV/D128 ring. Exact leaf and seven-pair production gates admit it.
LAGUNA_SWA_SPLIT_FIXED512_REDUCE = True
# The exact local256 GQA2 fused owner keeps 320 waves, reuses each K row across
# adjacent query heads, and removes the global score plane. Seven resident
# p512/d128 pairs admit it only at the saturated natural 512-slot shape.
LAGUNA_SWA_FUSED_FIXED512 = True
# The exact local384 GQA3 sibling keeps all 288 query/dimension waves active
# while reducing saturated K-cache owners per KV head from five to three.
# Seven resident p512/d128 pairs admit it only for the natural gfx1151 shape.
LAGUNA_SWA_GQA3_LOCAL384_FIXED512 = True
# The exact local384 sibling stages 64 contiguous V rows in LDS and reuses
# each load across the three owned query heads. The seven-pair p512/d128 gate
# is bit-identical and promotes it only at the saturated natural shape.
LAGUNA_SWA_GQA3_VSTAGE64_FIXED512 = True
# Replace scalar BF16 staging copies with aligned 16-byte transactions while
# preserving the retained local384 compute and every output operation.
LAGUNA_SWA_GQA3_VSTAGE64_VEC16_FIXED512 = True
# Avoid the compiler-generated per-thread LDS aggregate used by the retained
# vec16 copy and write each valid vector directly into the real V tile.
LAGUNA_SWA_GQA3_VSTAGE64_VEC16_DIRECT_FIXED512 = True
# The exact compiler-expf sibling exposes the finite non-positive
# score-minus-maximum domain and removes generic exponential guard work.
LAGUNA_SWA_GQA3_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = True
# Balance each KV head's nine queries as 2+2+2+3 across 32 local384 blocks.
# Exact seven-pair resident decode admits the one-phase mixed owner.
LAGUNA_SWA_MIXED32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = True
# Issue each four-slot exact softmax batch across lanes 0..3, then shuffle the
# weights back into the unchanged ordered denominator/PV chains. Seven
# resident pairs admit the resource-neutral sibling at saturated SWA512.
LAGUNA_SWA_MIXED32_EXP4_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = True
# Extend the same exact lane-parallel issue schedule to eight softmax weights.
# The leaf, cached trace, and all seven resident pairs improve without a
# resource or arithmetic-order change.
LAGUNA_SWA_MIXED32_EXP8_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = True
# Extend the exact lane-parallel schedule to sixteen softmax weights. The
# default remains subject to the resident p512/d128 gate.
LAGUNA_SWA_MIXED32_EXP16_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = True
# Complete the bounded issue-width screen with one exact softmax weight per
# wave32 lane. Resident decode decides whether this becomes the final owner.
LAGUNA_SWA_MIXED32_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = True
# Exact score-producer partial maxima remove four redundant 512-score scans
# per query. Seven exact resident p512/d128 pairs admit the specialization.
LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Compute each owned query's softplus gate once. All seven byte-exact resident
# p512/d128 pairs improve, so gfx1151 promotes the specialization.
LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Reuse each exact softmax weight across all four V-output waves through the
# V-stage publication barrier already paid by production. Seven exact resident
# pairs improve with unchanged VGPRs, so gfx1151 promotes the specialization.
LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Replay the published K64 probability tile through sixteen aligned float4 LDS
# reads while preserving the 64 ordered denominator adds. All seven exact
# resident p512/d128 pairs improve at unchanged kernel resources.
LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VEC4_DENOM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Read each published K64 probability row through sixteen aligned float4 LDS
# vectors while preserving the 64 ordered PV FMAs. All seven resident pairs
# improve at unchanged kernel resources.
LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# On pair-owner blocks, move the unchanged vectorized denominator replay onto
# idle waves 8/9 so all eight active output waves can execute PV concurrently.
# Seven exact resident p512/d128 pairs improve with complete separation.
LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Fill all 40 gfx1151 CUs with one 2+2+2+2+1 owner grid. Seven exact resident
# pairs improve with complete separation despite 25% more K/V-owner traffic.
LAGUNA_SWA_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Separate mixed40 tail exp producers from idle denominator and active PV
# waves. Six of seven exact resident pairs improve and the sole loss is
# smaller than the median paired gain.
LAGUNA_SWA_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Raise the exact mixed40 workgroup from 12 to 16 wave32s while retaining all
# 40 owners. All seven resident p512/d128 pairs improve with identical
# 128-token trajectories; the kernel also drops from 104 to 32 VGPRs.
LAGUNA_SWA_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Let the two exact tail-probability waves copy the final 64 staged-V vectors.
# The local512 combination wins all seven resident p512/d128 pairs while
# preserving the complete generated trajectory and allocation lifecycle.
LAGUNA_SWA_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_VALUE_TAIL_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Replace the retained wave32 QK shuffle transport with the association-
# identical permlanex16/DPP sequence inside the final local512/V128 tile.
# The leaf improves 5.35% and all seven resident p512/d128 pairs win.
LAGUNA_SWA_OUTPUT_SHARDED_PROBABILITY_DPP_QK = True
# Saturated sequential SWA has an identity physical ring with every slot
# visible. The exact dense-ring sibling compiles out token/base/eviction
# metadata traffic and its 2-KiB LDS physical-slot plane. Explicit eviction,
# pre-saturation, and non-standard states retain the generic DPP owner.
# The byte-exact leaf improves 25.55% and all seven resident pairs win.
LAGUNA_SWA_DENSE_RING = True
# gfx1151 exposes 32 wave32 slots and 1024 work-items per CU. The exact
# local1024 dense-ring owner fills that natural limit while preserving all 40
# workgroups; all seven same-resident p512/d128 pairs improve.
LAGUNA_SWA_LOCAL1024 = True
# Clean SOL-G5 p512/d128 evidence admits the state-bound composite GGUF graph
# only when at least 128 decode transitions amortize capture/instantiate/close.
GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS = 128
# Clean LCP-2A six-case exactness, balanced-wall, and 250-transition natural
# gates admit compiler-cacheable compact-scale direct LDS32 GDN on gfx1151.
GGUF_GDN_PREFILL_AUTO_MODE = "chain_lds32_direct_nonvolatile"
# The architecture-scoped strict-exact selector resolves to the same proven
# nonvolatile direct route as gfx1151 production.
GGUF_GDN_PREFILL_EXACT_MODE = "chain_lds32_direct_nonvolatile"
# F3's independent-c1 and physical-width gates admit the one-token-per-row
# indexed GDN sibling for packed AR while retaining segmented GDN as fallback.
GGUF_GDN_INDEXED_SINGLETON_DECODE = True
# F3's canonical p512/d128 gate rejects automatic Q8T16 row amortization:
# one non-repeated prompt trajectory diverges consistently at c2/c4/c8 even
# though the shorter d64 screen passed. Keep the env-only diagnostic available.
GGUF_Q8_T16_DECODE_ROWTILE_ALL = False
# The repaired 128-thread pair-only route preserves production reduction order.
# Scope its small repeatable win to the independently gated physical-c8 shape;
# c2/c4 stay on their faster per-row schedule.
GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS = 8
# Exact dynamic expert-ID pairing removes duplicate C8 Q4T16 gate/up weight
# reads while keeping each row's production 128-thread reduction order.
# Physical widths below C8 remain on the established kernel.
GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS = 8
# The same exact dynamic expert-ID pairing is retained for Q5T16 selected-down
# only at physical C8; lower widths preserve the established kernel.
GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS = 8
# Three Q6T16 down layers use the independently gated exact sibling at C8.
GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS = 8
# Dense Qwen3.6 Q5T16 recurrent-output ownership is W7900-only until gfx1151
# receives independent rotating-cache, quality, and complete-model gates.
GGUF_DENSE_Q5_T16_SSM_OUT = False
GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT = {
    "gguf_q5_k_t16_v1": 0,
}
# Dense Qwen3.6 planar-qmicro projection/root ownership is W7900-only until
# gfx1151 receives independent c1/small-row/top-1/prefill and complete-model
# gates.
GGUF_DENSE_Q6_T16_QMICRO_PLANAR = False
# Physical-C8 Q6T16 lm-head uses the exact 5+3 rowtile partition.
GGUF_Q6_LM_HEAD_MAX_CHUNK = 5
# F4's clean all-candidate, all-workload production gate selects fair:256 at
# +5.90% exact mixed-load SLO goodput over fair:128. Scope the default to the
# measured Q4_K_M generator registry entry; other quants/backends retain their
# prior engine-loop defaults until independently gated.
GGUF_Q4_K_M_PREFILL_DECODE_POLICY = "fair"
GGUF_Q4_K_M_MAX_PREFILL_CHUNK_TOKENS = 256
# Bound fair scheduling to two consecutive 256-token chunks so one p512 row
# becomes decode-ready per interruption instead of paying two partial-width
# decode ticks. The package selector keeps other quants/backends at one chunk.
GGUF_Q4_K_M_FAIR_PREFILL_BURST_CHUNKS = 2
# F3/F2 prove true physical-c8 GGUF AR and exact live ownership. The OpenAI
# coalescer may therefore submit eight plain-AR requests to this registry entry;
# speculative MTP keeps its separately certified four-request cap.
GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS = 8
# Clean LCP-M2 512/1K/4K full-state and balanced-wall gates admit stream-ordered
# device metadata through 4K. Explicit opt-in remains available for diagnosis;
# the 128K one-queue escalation still enters the low-power GPU-active state.
GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS = 4096
# Clean LCP-1 primitive/full-state, same-stream trace, and fresh-process wall
# gates admit the exact 32-token shared-memory convolution schedule on gfx1151.
GGUF_LINEAR_ATTN_CONV_PREFILL_AUTO_MODE = "tile32x128"
# Clean GPF-3A full-model 512/1K/4K evidence admits the byte-exact shared-X
# selected-dual Q4T16 prefill schedule on gfx1151.
GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE = "shared_x"
# LCP-4's exact router primitive and full-model gates admit the 256-thread
# reduction geometry for BF16-hidden/F32-weight GGUF router logits on gfx1151.
GGUF_ROUTER_F32_BF16_HIDDEN_THREADS = 256
# Post-LCP-4B profile and full-state gates admit 128 threads for bulk-prefill
# top-k selection. Decode keeps its independently selected 256-thread launch.
GGUF_PREFILL_ROUTER_SELECT_THREADS = 128
# Clean LCP-3 exactness plus balanced 512/4K wall admits four-wave activation
# sharing for covered dense Q8T16 WMMA prefill shapes on gfx1151. Two-wave stays
# available as the first rollback schedule during its release window.
GGUF_Q8_T16_PREFILL_FOUR_WAVE = True
GGUF_Q8_T16_PREFILL_TWO_WAVE = True
# Same-commit production-protocol 128K A/B rejects predecessor two-wave
# (382.041 vs 392.219 tok/s), so LCP-3 conservatively inherits its 64K ceiling.
GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS = 65536
# WPF-1/WPF-1T are W7900 raw-Q5/Q6 candidates. gfx1151 retains its
# independently admitted Q4_K_M/T16 matrix schedules and must not inherit their
# rowbatch or output-column selectors.
GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED = False
GGUF_RAW_K_PREFILL_ROWBATCH = 0
GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED = False
GGUF_RAW_K_PREFILL_COLTILE2_SHAPES = frozenset()
GGUF_RAW_K_PREFILL_VARIANT = "rowbatch"
# WPF-H5I exact-F32 ordered raw-K ownership is W7900-only until an
# independently measured gfx1151 package policy exists.
GGUF_Q6_F32_ORDERED_PREFILL = False
GGUF_Q6_F32_ORDERED_PREFILL_POLICY = {}
GGUF_F32_ORDERED_PREFILL_QUANTS = frozenset()
GGUF_F32_ORDERED_PREFILL_POLICIES = {}
# WPF-H5J/H5Q's K1024 IQ row owners are W7900-only until an independent gfx1151
# actual-layer and complete-runtime transfer gate exists.
LAGUNA_GROUPED_IQ_DOWN_VARIANTS = {}
LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS = {}
# LCP-2B is admitted only on W7900/gfx1100. gfx1151 keeps the exact scalar
# compact-WMMA row read until its independent post-merge transfer gate.
GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS = 0
# LCP-D2 is admitted only on W7900/gfx1100. gfx1151 keeps the serial reduction
# until it receives an independent long-context correctness/performance gate.
GGUF_PAGED_ATTN_PARALLEL_REDUCE = False
GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT = 32768
# Clean PARO G3/G5 physical-width and server gates certify c4/c8 with whole-row
# full-attention execution. Diagnostic c2 row chunking changes row-local
# numerics at these widths and must therefore remain an explicit override.
PARO_FULL_ATTN_NATIVE_EXACT_WIDTHS = frozenset({4, 8})
# G5 retains p512/d128 blocking and SSE c1/c2/c4/c8 scaling, delayed c4->c8
# admission, serial-c8 control, and repeated c8 exactness. Package capabilities
# select those identity-matched widths by default without branching in model or
# engine code; the legacy env flags remain explicit rollback opt-outs.
PARO_RETAINED_BATCH_DEFAULTS = True
PARO_NATIVE_BATCH_DECODE_DEFAULT = True
_SOURCE_BACKEND = "hip_gfx1100"
# Native speculative-cycle providers use dedicated backend registrations rather
# than this generic shared-body alias refresh. The GGUF target launcher has an
# independent gfx1151 parity gate; the proposal graph remains unadmitted here.
_GFX1151_ALIAS_EXCLUSIONS = frozenset(
    {
        # Qwen3.6 narrow/selective Q4T16 col4 rowtiles are W7900-only until
        # gfx1151 receives an independent shape crossover and full-model gate.
        (
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_rowtile_col4_bf16_bf16_out",
        ),
        # Dense Q5T16 ssm_out is W7900-only pending a separate gfx1151 gate.
        *(
            (
                "linear",
                "gguf_q5_k_t16_v1",
                variant,
            )
            for variant in (
                "t16_gemv_decode_bf16_bf16_out",
                "t16_gemv_rowtile_bf16_bf16_out",
                "t16_wmma_prefill_bf16_bf16_out",
            )
        ),
        (
            "gdn_recurrent_rmsnorm_gate+cast",
            "gguf_q5_k_t16_v1",
            "bf16_lowp_f32_bf16_out",
        ),
        (
            "gdn_chain_recurrent_rmsnorm_gate+cast",
            "gguf_q5_k_t16_v1",
            "bf16_c1_exact_state_rows_tloop_f32_bf16_out",
        ),
        # Dense planar-qmicro Q6 is W7900-only pending a separate gfx1151 gate.
        *(
            (
                layer,
                "gguf_q6_k_t16_qmicro_planar_v1",
                variant,
            )
            for layer, variant in (
                ("linear", "t16_gemv_decode_bf16_bf16_out"),
                ("linear", "t16_gemv_decode_bf16_f32_out"),
                ("linear", "t16_gemv_rowtile_bf16_bf16_out"),
                ("linear", "t16_gemv_rowtile_col8_bf16_bf16_out"),
                ("linear", "t16_gemv_rowtile_bf16_f32_out"),
                ("linear", "t16_wmma_prefill_bf16_bf16_out"),
                ("linear+argmax", "t16_gemv_decode_bf16_f32_top1_stage1"),
                ("linear+argmax", "proposal_top1_exact_bf16"),
            )
        ),
        # Exact single-page and P2 split attention are W7900-only until gfx1151
        # receives independent crossover, full-state, and performance gates.
        (
            "laguna_attention_decode",
            "bf16",
            "global_context_single_page_spans",
        ),
        (
            "laguna_attention_decode+attention_gate",
            "bf16",
            "global_single_page_softplus_bf16_spans",
        ),
        # The exact split-attention producer/reducer bundle is registered for
        # an independent gfx1151 threshold and full-model screen. Automatic
        # selection remains off until that architecture-local gate passes.
        # The global-only wave-0 tree remains W7900-only. The retained scalar
        # current-P4 global/SWA bodies are independently gated on gfx1151.
        (
            "head_rmsnorm+partial_rotary+kv_write",
            "laguna_f32_weight",
            "global_wave0_tree_f32_bf16_spans",
        ),
        # D9's wave-0 RMS tree has an independent gfx1151 gate. The exact
        # top-10 split sibling remains W7900-only.
        (
            "weighted_sum+moe_tail",
            "bf16",
            "laguna_top10_routed_hidden_out",
        ),
        # Staged unrounded-F32 Laguna add+RMSNorm is W7900-only until an
        # independent gfx1151 correctness and performance gate.
        (
            "add_rmsnorm",
            "gguf_f32_weight",
            "bf16_out_staged_f32_local256",
        ),
        # IQ2 fixed-local64 DPP reduction is W7900-only pending an independent gate.
        (
            "moe_linear",
            "gguf_iq2_xs",
            "selected_dual_silu_gemv_decode_tile2_grid64_local64_reduce_bf16_bf16_out",
        ),
        # IQ3 selected-down tiling is gfx1100-only pending independent gfx1151 gates.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_gemv_decode_tile4_bf16_bf16_out",
        ),
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_gemv_decode_k1024_wave4_signbit_bf16_bf16_out",
        ),
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_weighted_down_gemv_decode_k1024_wave10_bf16_bf16_out",
        ),
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_weighted_down_gemv_decode_k1024_wave10_signbit_bf16_bf16_out",
        ),
        # Laguna top-10/K1024 IQ4 weighted ownership is gfx1100-only pending an
        # independent gfx1151 correctness and performance gate.
        (
            "moe_linear",
            "gguf_iq4_xs",
            "selected_weighted_down_gemv_decode_bf16_bf16_out",
        ),
        # WPF-1 fixed-grid-Y raw Q5/Q6 row reuse is W7900-only pending an
        # independent gfx1151 gate. Keep every output/slab key unaliased.
        *(
            ("linear", quant, f"rowbatch{row_batch}_bf16_{output_dtype}_out")
            for quant in ("gguf_q5_k", "gguf_q6_k")
            for row_batch in (4, 8, 16, 32)
            for output_dtype in ("bf16", "f32")
        ),
        *(
            (
                "linear",
                quant,
                f"coltile{col_tile}_rowbatch{row_batch}_bf16_{output_dtype}_out",
            )
            for quant in ("gguf_q5_k", "gguf_q6_k")
            for col_tile, row_batch in ((2, 16), (4, 8))
            for output_dtype in ("bf16", "f32")
        ),
        # WPF-H7C transfers the exact H6U reduction instruction form to two
        # W7900 raw-Q6 leaves and remains absent without a gfx1151 screen.
        (
            "linear",
            "gguf_q6_k",
            "dpp_wave_reduction_coltile4_rowbatch8_bf16_bf16_out",
        ),
        (
            "linear",
            "gguf_q6_k",
            "dpp_wave_reduction_coltile2_rowbatch16_bf16_f32_out",
        ),
        # WPF-H7I removes H7C's inner live-row predicate only for exactly-full
        # W7900 natural-M512 role groups; gfx1151 remains unscreened.
        (
            "linear",
            "gguf_q6_k",
            "dpp_wave_reduction_full_group_compute_"
            "coltile4_rowbatch8_bf16_bf16_out",
        ),
        (
            "linear",
            "gguf_q6_k",
            "dpp_wave_reduction_full_group_compute_"
            "coltile2_rowbatch16_bf16_f32_out",
        ),
        # WPF-H5U and H6A local256 cached-only global leaves are W7900-only
        # pending independent gfx1151 resource/performance gates.
        (
            "laguna_attention_prefill",
            "bf16",
            "global_context_rows_cached_exact_spans",
        ),
        (
            "laguna_attention_prefill",
            "bf16",
            "global_context_rows_dense_initial_cached_exact_spans",
        ),
        # WPF-H6N's retained fixed-arena leaf is W7900-only pending independent
        # gfx1151 resource/performance evidence and any bounded runtime owner.
        (
            "laguna_attention_prefill",
            "bf16",
            "global_context_rows_dense_initial_fixed512_cached_exact_spans",
        ),
        # WPF-H6Z exact late-start global qrow4 score/weight replay is W7900-
        # only pending independent gfx1151 resource/performance evidence.
        (
            "laguna_attention_prefill",
            "bf16",
            "global_context_rows_qrow4_dense_initial_global_score_weight_replay_exact_spans",
        ),
        # WPF-H5R and H6A exact cached-only two-pass SWA qrow4 leaves are
        # W7900-only pending independent gfx1151 resource/performance gates.
        (
            "laguna_attention_prefill",
            "bf16",
            "swa_context_rows_qrow4_cached_exact_spans",
        ),
        (
            "laguna_attention_prefill",
            "bf16",
            "swa_context_rows_qrow4_dense_initial_cached_exact_spans",
        ),
        # WPF-H6W exact late-start global score replay is W7900-only pending
        # independent gfx1151 resource/performance evidence.
        (
            "laguna_attention_prefill",
            "bf16",
            "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans",
        ),
        # WPF-H7Y's lane-major SWA cache consumer and fused mirror writer are
        # W7900-only pending an independent gfx1151 resource/performance gate.
        (
            "laguna_attention_prefill",
            "bf16",
            "swa_context_rows_qrow4_dense_initial_lane_major_global_score_replay_exact_spans",
        ),
        (
            "laguna_kv_write",
            "bf16",
            "swa_f32_rows_natural_lane_major_spans",
        ),
        # WPF-H5M exact source-qualified qrow4 is W7900-only pending an
        # independent gfx1151 resource/performance gate.
        (
            "laguna_attention_prefill",
            "bf16",
            "swa_context_rows_qrow4_sourcequal_exact_spans",
        ),
        # WPF-H5N's identity/no-wrap exact specialization is likewise scoped to
        # gfx1100 until it has independent gfx1151 evidence.
        (
            "laguna_attention_prefill",
            "bf16",
            "swa_context_rows_qrow4_dense_first_fill_exact_spans",
        ),
        # WPF-H2 copies llama.cpp's gfx1100 F16-WMMA FlashAttention geometry
        # and remains excluded until gfx1151 receives an independent gate.
        (
            "laguna_attention_prefill",
            "bf16",
            "source_f16_wmma_q8_gqa8_spans",
        ),
        # WPF-H1 copies the gfx1100/RDNA3 source geometry and remains excluded
        # until gfx1151 receives an independent resource/correctness gate.
        ("activation_quant", "q8_1_ds4", "bf16_kmajor"),
        (
            "linear",
            "gguf_q5_k",
            "mmq_i128_j128_k256_q8_1_ds4_bf16_bf16_out",
        ),
        (
            "linear",
            "gguf_q5_k",
            "mmq_i128_j128_k256_q8_1_ds4_bf16_f32_out",
        ),
        # WPF-H6L's exact K3072/N1024/E256 pair16 rowbatch16 leaf is
        # W7900-only pending independent gfx1151 resource/performance gates.
        (
            "moe_linear",
            "gguf_iq2_xs",
            "selected_dual_silu_grouped_prefill_compact_"
            "k3072_n1024_e256_pair16_rowbatch16_bf16_bf16_out",
        ),
        # WPF-H6C's K3072/N1024/E256 expert-major fused-SiLU leaf is
        # W7900-only pending independent gfx1151 resource/performance gates.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_dual_silu_grouped_prefill_compact_"
            "k3072_n1024_e256_rowbatch4_bf16_bf16_out",
        ),
        # WPF-H5J's K1024 resident-segment IQ3 and one-wave IQ4 schedules are
        # W7900-only pending independent gfx1151 resource/performance gates.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_resident_rowbatch8_bf16_bf16_out",
        ),
        (
            "moe_linear",
            "gguf_iq4_xs",
            "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out",
        ),
        # H5Q's active-expert persistent traversal is likewise gfx1100-only
        # until gfx1151 receives independent resource/performance evidence.
        *(
            (
                "moe_linear",
                "gguf_iq3_xxs",
                "selected_grouped_prefill_compact_k1024_active_expert_p"
                f"{partition}_resident_rowbatch8_bf16_bf16_out",
            )
            for partition in (64,)
        ),
        # H5Z's activation-resident output sweep keeps H5Q expert P64 but is
        # gfx1100-only until independently screened on gfx1151.
        *(
            (
                "moe_linear",
                "gguf_iq3_xxs",
                "selected_grouped_prefill_compact_k1024_active_expert_p64_"
                "activation_resident_out_p"
                f"{output_partition}_rowbatch8_bf16_bf16_out",
            )
            for output_partition in (256,)
        ),
        # H6D interleaves independent H5Z row accumulators specifically for
        # gfx1100 VOPD and remains absent without a gfx1151 screen.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_"
            "activation_resident_out_p256_row_interleaved_vopd_rowbatch8_"
            "bf16_bf16_out",
        ),
        # H6F pairs two independent H6D output reductions to amortize gfx1100
        # barriers and remains absent without an independent gfx1151 screen.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_"
            "activation_resident_out_p256_row_interleaved_vopd_paired_output_"
            "rowbatch8_bf16_bf16_out",
        ),
        # H6I groups three independent H6F output reductions and remains
        # gfx1100-only until an independent gfx1151 screen.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_"
            "activation_resident_out_p256_row_interleaved_vopd_triple_output_"
            "rowbatch8_bf16_bf16_out",
        ),
        # H6P changes H6I accumulator liveness with gfx1100 wave publication
        # and remains absent without an independent gfx1151 screen.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_"
            "activation_resident_out_p256_row_interleaved_vopd_"
            "staged_wave_publication_triple_output_rowbatch8_bf16_bf16_out",
        ),
        # H6Q changes only H6P's gfx1100 shuffle-loop code footprint and
        # remains absent without an independent gfx1151 screen.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_"
            "activation_resident_out_p256_row_interleaved_vopd_"
            "staged_wave_publication_compact_shuffle_loop_triple_output_"
            "rowbatch8_bf16_bf16_out",
        ),
        # H6R changes only H6Q's gfx1100 wave peer-exchange instructions and
        # remains absent without an independent gfx1151 screen.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_"
            "activation_resident_out_p256_row_interleaved_vopd_"
            "staged_wave_publication_dpp_peer_exchange_triple_output_"
            "rowbatch8_bf16_bf16_out",
        ),
        # H6T changes only H6R's final gfx1100 DPP move/add instruction form
        # and remains absent without an independent gfx1151 screen.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_"
            "activation_resident_out_p256_row_interleaved_vopd_staged_wave_"
            "publication_dpp_peer_exchange_fused_add_triple_output_"
            "rowbatch8_bf16_bf16_out",
        ),
        # WPF-H3 reuses the DS4 producer but has independently qualified raw-IQ
        # consumers. Both remain gfx1100-only pending a gfx1151 gate.
        *(
            (
                "moe_linear",
                quant,
                "selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out",
            )
            for quant in ("gguf_iq3_xxs", "gguf_iq4_xs")
        ),
        # H7E's residual-D4x2 IQ3 consumer remains gfx1100-only until an
        # independently qualified gfx1151 residual-plane gate exists.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_mmq_i128_j128_k256_q8_1_ds4x2_"
            "prefill_compact_bf16_bf16_out",
        ),
        # WPF-H4 copies llama.cpp's gfx1100 Q6-to-F16/rocBLAS ownership and
        # remains excluded until gfx1151 receives an independent gate.
        ("dequant", "gguf_q6_k", "raw_f16_source_local64"),
        (
            "dequant_cast",
            "gguf_q6_k",
            "raw_f16_bf16_input_source_local64",
        ),
        (
            "linear",
            "gguf_q6_k",
            "f16_rocblas_source_bf16_bf16_out",
        ),
        (
            "linear",
            "gguf_q6_k",
            "f16_rocblas_source_bf16_f32_out",
        ),
        # WPF-H5A/H5I transient exact-F32 Q5/Q6 producers are gfx1100-only
        # until gfx1151 receives independent resource, timing, and quality gates.
        ("dequant", "gguf_q5_k", "raw_f32_exact_local64"),
        ("dequant", "gguf_q6_k", "raw_f32_exact_local64"),
        (
            "dequant_cast",
            "gguf_q5_k",
            "raw_f32_bf16_input_exact_local64",
        ),
        (
            "linear",
            "gguf_q5_k",
            "f32_rocblas_exact_values_bf16_bf16_out",
        ),
        (
            "linear",
            "gguf_q5_k",
            "f32_rocblas_exact_values_bf16_f32_out",
        ),
        # WPF-H5C/H5I production-ordered and H5L weight-major F32 consumers
        # plus raw-Q5/Q6 composites stay gfx1100-only pending a gfx1151 gate.
        *(
            (
                "linear",
                quant,
                f"{prefix}coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for quant, prefix in (
                ("f32_weight", "ordered_"),
                ("gguf_q5_k", "f32_ordered_"),
                ("gguf_q6_k", "f32_ordered_"),
            )
            for col_tile, row_batch in (
                (4, 8),
                (8, 4),
                (4, 16),
                (8, 8),
                (16, 4),
                (12, 4),
                (8, 10),
                (16, 5),
                (8, 12),
                (12, 8),
            )
            if quant != "gguf_q6_k"
            or (col_tile, row_batch) in {(8, 4), (16, 4), (16, 5)}
            for output_dtype in ("bf16", "f32")
        ),
        *(
            (
                "linear",
                quant,
                f"{prefix}coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for quant, prefix in (
                ("f32_weight", "ordered_weight_major_"),
                ("gguf_q5_k", "f32_ordered_weight_major_"),
            )
            for col_tile, row_batch, output_dtype in (
                (8, 4, "bf16"),
                (8, 12, "bf16"),
                (16, 5, "bf16"),
                (12, 8, "bf16"),
                (16, 4, "bf16"),
                (16, 5, "f32"),
                (8, 10, "f32"),
            )
        ),
        *(
            (
                "linear",
                "gguf_q6_k",
                f"f32_ordered_weight_major_coltile{col_tile}_"
                f"rowbatch{row_batch}_bf16_{output_dtype}_out",
            )
            for col_tile, row_batch, output_dtype in (
                (16, 5, "bf16"),
                (16, 4, "bf16"),
                (16, 5, "f32"),
            )
        ),
        # H5X exact tile-K-col Q5 producers/consumers are W7900-only until an
        # independent gfx1151 layout/resource/performance gate qualifies them.
        *(
            (
                layer,
                quant,
                f"{prefix}coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for layer, quant, prefix in (
                ("dequant", "gguf_q5_k", "raw_f32_exact_tile_k_col_"),
                (
                    "linear",
                    "f32_weight",
                    "ordered_weight_major_tile_k_col_",
                ),
                (
                    "linear",
                    "gguf_q5_k",
                    "f32_ordered_weight_major_tile_k_col_",
                ),
            )
            for col_tile, row_batch, output_dtype in (
                (8, 4, "bf16"),
                (16, 5, "bf16"),
                (16, 5, "f32"),
                (8, 10, "f32"),
            )
        ),
        # H5Y exact activation-tile-K-row packs/consumers are likewise
        # W7900-only pending an independent gfx1151 transfer gate.
        *(
            (
                "activation_pack",
                "bf16",
                f"tile_k_row_coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for col_tile, row_batch, output_dtype, _weight_layout in (
                (8, 4, "bf16", "tile_k_col"),
                (8, 12, "bf16", "row_major"),
                (16, 5, "bf16", "tile_k_col"),
                (12, 8, "bf16", "row_major"),
                (16, 5, "f32", "tile_k_col"),
                (8, 10, "f32", "tile_k_col"),
            )
        ),
        *(
            (
                "linear",
                quant,
                f"{prefix}{weight_layout}_activation_tile_k_row_"
                f"coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for quant, prefix in (
                ("f32_weight", "ordered_weight_major_"),
                ("gguf_q5_k", "f32_ordered_weight_major_"),
            )
            for col_tile, row_batch, output_dtype, weight_layout in (
                (8, 4, "bf16", "tile_k_col"),
                (8, 12, "bf16", "row_major"),
                (16, 5, "bf16", "tile_k_col"),
                (12, 8, "bf16", "row_major"),
                (16, 5, "f32", "tile_k_col"),
                (8, 10, "f32", "tile_k_col"),
            )
        ),
        # H7G exact padded-row Q5 consumers are gfx1100-only pending an
        # independent gfx1151 physical/performance transfer gate.
        *(
            (
                "linear",
                quant,
                f"{prefix}{weight_layout}_activation_tile_k_row_"
                f"padded_compute_coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for quant, prefix in (
                ("f32_weight", "ordered_weight_major_"),
                ("gguf_q5_k", "f32_ordered_weight_major_"),
            )
            for col_tile, row_batch, output_dtype, weight_layout in (
                (8, 12, "bf16", "row_major"),
                (16, 5, "bf16", "tile_k_col"),
                (16, 5, "f32", "tile_k_col"),
                (8, 10, "f32", "tile_k_col"),
            )
        ),
        # H8A's resident-plane composites remain gfx1100-only; gfx1151 has no
        # package capability or independently qualified owner/cache policy.
        *(
            (
                "linear",
                "gguf_q5_k",
                "f32_resident_ordered_weight_major_tile_k_col_"
                "activation_tile_k_row_padded_compute_coltile16_rowbatch5_"
                f"bf16_{output_dtype}_out",
            )
            for output_dtype in ("bf16", "f32")
        ),
        # H7H exact full-group Q5 consumers are separately scoped to gfx1100.
        *(
            (
                "linear",
                quant,
                f"{prefix}{weight_layout}_activation_tile_k_row_"
                f"full_group_compute_coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for quant, prefix in (
                ("f32_weight", "ordered_weight_major_"),
                ("gguf_q5_k", "f32_ordered_weight_major_"),
            )
            for col_tile, row_batch, output_dtype, weight_layout in (
                (8, 4, "bf16", "tile_k_col"),
                (12, 8, "bf16", "row_major"),
            )
        ),
        # H6E exact Q6 activation-row consumers are separately scoped to
        # gfx1100 pending their standalone physical/performance gate.
        *(
            (
                "linear",
                quant,
                f"{prefix}weight_major_row_major_activation_tile_k_row_"
                f"coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for quant, prefix in (
                ("f32_weight", "ordered_"),
                ("gguf_q6_k", "f32_ordered_"),
            )
            for col_tile, row_batch, output_dtype in (
                (16, 5, "bf16"),
                (16, 4, "bf16"),
                (16, 5, "f32"),
            )
        ),
        # H6U exact DPP-reduction candidates remain gfx1100-only unless their
        # standalone W7900 screen and a later gfx1151 transfer both pass.
        *(
            (
                "linear",
                quant,
                f"{prefix}weight_major_row_major_activation_tile_k_row_"
                f"dpp_wave_reduction_coltile{col_tile}_"
                f"rowbatch{row_batch}_bf16_{output_dtype}_out",
            )
            for quant, prefix in (
                ("f32_weight", "ordered_"),
                ("gguf_q6_k", "f32_ordered_"),
            )
            for col_tile, row_batch, output_dtype in (
                (16, 5, "bf16"),
                (16, 4, "bf16"),
                (16, 5, "f32"),
            )
        ),
        # Rejected WPF-1B producer/MMQ primitives remain gfx1100-only
        # diagnostic evidence, with no runtime policy owner on either backend.
        ("activation_quant", "q8_1_d4s4_f32", "bf16"),
        ("activation_quant", "q8_1_d8s8_f32", "bf16"),
        ("activation_quant", "q8_1_d8r8s8_f32", "bf16"),
        *(
            (
                "linear",
                quant,
                f"mmq32_q8_1_{producer}_f32_bf16_{output_dtype}_out",
            )
            for quant in ("gguf_q5_k", "gguf_q6_k")
            for producer in ("d4s4", "d8s8", "d8r8s8")
            for output_dtype in ("bf16", "f32")
        ),
        # Q4 local32 LM-head ownership is W7900-only pending an independent gate.
        (
            "linear",
            "gguf_q4_k",
            "local32_fixed_meta_gemv_decode_bf16_f32_out",
        ),
        # Q6 local32 standalone ownership is likewise W7900-only pending a gate.
        (
            "linear",
            "gguf_q6_k",
            "standalone_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out",
        ),
        # Paired-output SWAR Q5 reconstruction is W7900-only pending an independent gate.
        (
            "linear",
            "gguf_q5_k",
            "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out",
        ),
        (
            "linear_pair",
            "gguf_q5_k",
            "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out",
        ),
        (
            "attention_projection_quad",
            "gguf_q5_k+gguf_q6_k+gguf_q6_k+gguf_q5_k",
            "mixed_local32_q5_swar_pair_fixed_meta_gemv_decode_bf16_f32_out",
        ),
        # Heterogeneous Q5/Q6 pair reuse is W7900-only pending an independent gate.
        (
            "attention_projection_quad",
            "gguf_q5_k+gguf_q6_k+gguf_q6_k+gguf_q5_k",
            "mixed_pair_reuse_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out",
        ),
        # Laguna compact/persistent routing is W7900-only pending independent gates.
        (
            "laguna_sigmoid_router_topk",
            "f32",
            "correction_bias_compact_wave32",
        ),
        (
            "laguna_router_topk",
            "f32",
            "bf16_hidden_correction_bias_persistent_wave_top10",
        ),
        (
            "speculative_cycle",
            "w4_gguf",
            "native_v1_b2_target_graph",
        ),
        (
            "speculative_cycle",
            "w4_gguf",
            "native_v1_b2_proposal_graph",
        ),
    }
)
_GFX1151_OVERRIDES = {
    # Source-faithful Vulkan DMMV output-row reuse, adapted without Vulkan's
    # rejected wave64/local64 geometry: one retained local256 block owns two
    # adjacent F16 columns, shares the BF16 activation load/conversion, and
    # reduces both exact accumulator trees through one barrier.
    (
        "attention_projection+head_rmsnorm+partial_rotary+kv_write",
        "fp16_weight+laguna_f32_weight",
        "global_fixedk_nontemporal_bf16_f32_spans",
    ): laguna_global_f16_projection_head_kv_nontemporal_tile2_bf16_spans,
    (
        "attention_projection+head_rmsnorm+partial_rotary+kv_write",
        "fp16_weight+laguna_f32_weight",
        "swa_fixedk_nontemporal_bf16_f32_spans",
    ): laguna_swa_f16_projection_head_kv_nontemporal_tile2_bf16_spans,
    # F3Q caches 24 of 128 FP32 state rows across the GDN dependency barrier.
    # Its 15 KiB LDS footprint preserves four resident blocks on gfx1151.
    (
        "gdn_recurrent_rmsnorm_gate",
        "gguf_qwen35",
        "bf16_indexed_singleton",
    ): qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16,
    # The scalar-tree c1-exact kernel retained for gfx1100/PARO diverges from
    # gfx1151's established paged-c1 arithmetic at model scale. Keep gfx1151 on
    # the generic reduction, but pin its geometry to the c4/c8-proven 256-thread
    # shape: the generic rows<=2 1024-thread fast path diverges over p512/d128.
    (
        "paged_attn_decode",
        "w4_paro",
        "bf16_context_batch_c1_exact_spans",
    ): qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans,
    (
        "router_logits",
        "f32",
        "bf16_hidden",
    ): qwen35_router_logits_bf16_f32w_auto_256,
    (
        "linear",
        "gguf_q8_0_t16_v1",
        "wmma_prefill_bf16_bf16_out",
    ): gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out,
    (
        "linear",
        "gguf_q8_0_t16_v1",
        "t16_wmma_prefill_bf16_bf16_out",
    ): gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out,
    (
        "linear",
        "gguf_q4_k",
        "pack8_wmma_prefill_bf16_bf16_out",
    ): gguf_q4_k_pack8_wmma_prefill_gfx1151_bf16_bf16_out,
    (
        "linear",
        "gguf_q6_k",
        "wmma_prefill_bf16_bf16_out",
    ): gguf_q6_k_wmma_prefill_16x32_bf16_bf16_out,
}
_GFX1100_MODULES = (
    "hipengine.kernels.hip_gfx1100.attention",
    "hipengine.kernels.hip_gfx1100.convert",
    "hipengine.kernels.hip_gfx1100.fused",
    "hipengine.kernels.hip_gfx1100.linear",
    "hipengine.kernels.hip_gfx1100.linear_attn",
    "hipengine.kernels.hip_gfx1100.moe",
    "hipengine.kernels.hip_gfx1100.norm",
    "hipengine.kernels.hip_gfx1100.quant",
    "hipengine.kernels.hip_gfx1100.rotary",
    "hipengine.kernels.hip_gfx1100.runtime",
    "hipengine.kernels.hip_gfx1100.sampling",
    "hipengine.kernels.hip_gfx1100.smoke",
    "hipengine.kernels.hip_gfx1100.speculative",
    "hipengine.kernels.hip_gfx1100.wmma",
)


def register_gfx1151_kernels(*, replace: bool = False) -> None:
    """Register gfx1151 aliases for the current gfx1100 kernel key space."""

    for module_name in _GFX1100_MODULES:
        import_module(module_name)
    source_keys = [key for key in registered_keys() if key.backend == _SOURCE_BACKEND]
    for key in source_keys:
        if (key.layer, key.quant, key.variant) in _GFX1151_ALIAS_EXCLUSIONS:
            continue
        target_key = KernelKey(BACKEND, key.layer, key.quant, key.variant)
        if not replace and is_registered(target_key):
            continue
        source_fn = resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        register(
            target_key,
            _GFX1151_OVERRIDES.get((key.layer, key.quant, key.variant), source_fn),
            replace=replace,
        )
    q4_pack8_decode_pair_key = KernelKey(
        BACKEND,
        "linear_pair",
        "gguf_q4_k",
        "pack8_dual_decode_bf16_bf16_out",
    )
    if replace or not is_registered(q4_pack8_decode_pair_key):
        register(
            q4_pack8_decode_pair_key,
            gguf_q4_k_pack8_dual_prefill_bf16_bf16_out,
            replace=replace,
        )
    q4_pack8_decode_pair_silu_key = KernelKey(
        BACKEND,
        "linear_pair_silu",
        "gguf_q4_k",
        "pack8_dual_decode_bf16_bf16_out",
    )
    if replace or not is_registered(q4_pack8_decode_pair_silu_key):
        register(
            q4_pack8_decode_pair_silu_key,
            gguf_q4_k_pack8_dual_silu_bf16_bf16_out,
            replace=replace,
        )
    q4_t16_sidecar_pair_silu_key = KernelKey(
        BACKEND,
        "linear_pair_silu",
        "gguf_q4_k",
        "t16_sidecar_dual_decode_bf16_bf16_out",
    )
    if replace or not is_registered(q4_t16_sidecar_pair_silu_key):
        register(
            q4_t16_sidecar_pair_silu_key,
            gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out,
            replace=replace,
        )
    q4_t16_dual_interleaved_pair_silu_key = KernelKey(
        BACKEND,
        "linear_pair_silu",
        "gguf_q4_k",
        "t16_dual_interleaved_sidecar_decode_bf16_bf16_out",
    )
    if replace or not is_registered(
        q4_t16_dual_interleaved_pair_silu_key
    ):
        register(
            q4_t16_dual_interleaved_pair_silu_key,
            gguf_q4_k_t16_dense_dual_interleaved_tile2_local32_silu_bf16_bf16_out,
            replace=replace,
        )


register_gfx1151_kernels()
register_backend_kernels = register_gfx1151_kernels

__all__ = [
    "BACKEND",
    "GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS",
    "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS",
    "GGUF_GDN_INDEXED_SINGLETON_DECODE",
    "GGUF_GDN_PREFILL_AUTO_MODE",
    "GGUF_GDN_PREFILL_EXACT_MODE",
    "GGUF_LINEAR_ATTN_CONV_PREFILL_AUTO_MODE",
    "GGUF_PAGED_ATTN_PARALLEL_REDUCE",
    "GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT",
    "GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS",
    "GGUF_PREFILL_ROUTER_SELECT_THREADS",
    "GGUF_Q4_K_M_FAIR_PREFILL_BURST_CHUNKS",
    "GGUF_Q4_K_M_MAX_PREFILL_CHUNK_TOKENS",
    "GGUF_Q4_K_M_PREFILL_DECODE_POLICY",
    "GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS",
    "GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE",
    "GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_DENSE_Q5_T16_SSM_OUT",
    "GGUF_DENSE_Q6_T16_QMICRO_PLANAR",
    "GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT",
    "GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q6_LM_HEAD_MAX_CHUNK",
    "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS",
    "GGUF_Q8_T16_DECODE_ROWTILE_ALL",
    "GGUF_Q8_T16_PREFILL_FOUR_WAVE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS",
    "GGUF_F32_ORDERED_PREFILL_POLICIES",
    "GGUF_F32_ORDERED_PREFILL_QUANTS",
    "GGUF_Q6_F32_ORDERED_PREFILL",
    "GGUF_Q6_F32_ORDERED_PREFILL_POLICY",
    "GGUF_RAW_K_PREFILL_COLTILE2_SHAPES",
    "GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED",
    "GGUF_RAW_K_PREFILL_ROWBATCH",
    "GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED",
    "GGUF_RAW_K_PREFILL_VARIANT",
    "GGUF_ROUTER_F32_BF16_HIDDEN_THREADS",
    "LAGUNA_DENSE_Q4_PREFILL_MODE",
    "LAGUNA_F16_BOUNDARY_FUSION",
    "LAGUNA_F16_ATTENTION_QUAD_DECODE",
    "LAGUNA_F16_NONTEMPORAL_DECODE",
    "LAGUNA_F16_OUTPUT_ADD_RMSNORM_DECODE",
    "LAGUNA_F16_PROJECTION_HEAD_KV_DECODE",
    "LAGUNA_F16_DECODE_FIXEDK",
    "LAGUNA_F16_DECODE_ONEBARRIER",
    "LAGUNA_Q4_PACK8_DUAL_SILU_DECODE",
    "LAGUNA_Q4_DENSE_DECODE_T16_SIDECAR",
    "LAGUNA_Q4_DENSE_DECODE_T16_DUAL_INTERLEAVED",
    "LAGUNA_Q4_SHARED_DOWN_T16_DECODE",
    "LAGUNA_Q4_EXPERT_T16_DUAL_INTERLEAVED",
    "LAGUNA_SELECTED_NATURAL_DECODE",
    "LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_DECODE",
    "LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_WEIGHTED_DECODE",
    "LAGUNA_SELECTED_DOWN_Q4_PAIRCOEFF_WEIGHTED_DECODE",
    "LAGUNA_SELECTED_NATURAL_TILE8_DECODE",
    "LAGUNA_SELECTED_NATURAL_TILE8_PARALLEL_DECODE",
    "LAGUNA_SELECTED_NATURAL_TILE8_PARALLEL_SILU_DECODE",
    "LAGUNA_SELECTED_HALFDOT_DECODE",
    "LAGUNA_F16_PREFILL_MIN_ROWS",
    "LAGUNA_F16_PREFILL_MODE",
    "LAGUNA_F16_PREFILL_STRATEGY",
    "LAGUNA_GLOBAL_PREFILL_VARIANT",
    "LAGUNA_GLOBAL_SPLIT_MIN_LIVE",
    "LAGUNA_GLOBAL_SPLIT_FIXEDSHAPE_REDUCE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_DIM32_VSTAGE64",
    "LAGUNA_GLOBAL_SPLIT_GQA6_DEFERREDNORM_DIM32_VSTAGE64",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE64",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX_NONTEMPORAL_MIN_LIVE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_PREFETCH8_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE128_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE128_PROBABILITY_VEC4_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_ALLWAVE_TILE1024_DENSE_PREFIX",
    "LAGUNA_GLOBAL_SPLIT_GQA6_ALLWAVE_TILE1024_DENSE_PREFIX_MIN_LIVE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_COMPENSATED_LAYER",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_COMPENSATED_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE_PREFETCH",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_ALLWAVE_SCORE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DIM_TILE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DEFERREDNORM",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE_PREFETCH",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_SCORE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_MIN_LIVE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_MIN_LAYER",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_TOKENLOOP4",
    "LAGUNA_HEAD_KV_FUSION",
    "LAGUNA_GROUPED_IQ_DOWN_VARIANTS",
    "LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS",
    "LAGUNA_MOE_BRANCH_CONCURRENCY",
    "LAGUNA_MOE_DECODE_BRANCH_CONCURRENCY",
    "LAGUNA_MOE_DECODE_SHARED_NORMAL_PRIORITY",
    "LAGUNA_MOE_TAIL_WAVE0_TREE",
    "LAGUNA_MOE_GROUP_COMPACT_MODE",
    "LAGUNA_MOE_SHARED_AFTER_ROUTER",
    "LAGUNA_MOE_SHARED_LOW_PRIORITY",
    "LAGUNA_GLOBAL_FUSED_FIXEDSHAPE",
    "LAGUNA_GLOBAL_GQA2_VSTAGE64_FIXEDSHAPE",
    "LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_FIXEDSHAPE",
    "LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_DIRECT_FIXEDSHAPE",
    "LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_GQA2_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_MIXED32_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_MIXED32_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_DENSE_PREFIX",
    "LAGUNA_GLOBAL_DENSE_PREFIX_IDLE_DOUBLE_BUFFER",
    "LAGUNA_GLOBAL_LOCAL1024",
    "LAGUNA_PREFILL_ATTENTION_HIPBLASLT",
    "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_OUTPUT_GATE",
    "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERY_PRODUCER",
    "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERIES",
    "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_WAVE_ROWS_SOFTMAX",
    "LAGUNA_PREFILL_BLOCK_ATTENTION_HIPBLASLT",
    "LAGUNA_PREFILL_DENSE_CONTIGUOUS_CACHE",
    "LAGUNA_PREFILL_GLOBAL_ATTENTION_ROWS",
    "LAGUNA_PREFILL_LONG_ATTENTION_HIPBLASLT",
    "LAGUNA_PREFILL_SWA_ATTENTION_HIPBLASLT",
    "LAGUNA_PREFILL_CACHED_META",
    "LAGUNA_PREFILL_KV_PREAPPEND",
    "LAGUNA_PREFILL_MATRIX_ROWS",
    "LAGUNA_Q6_QMICRO",
    "LAGUNA_Q6_QMICRO_PLANAR",
    "LAGUNA_Q6_QMICRO_PERMUTE",
    "LAGUNA_ROUTER_PROJECTION_WAVE0_TREE",
    "LAGUNA_ROUTER_LOGITS_MODE",
    "LAGUNA_SELECTED_DOWN_MODE",
    "LAGUNA_SELECTED_GATE_UP_MODE",
    "LAGUNA_SPLIT_GATE_FUSION",
    "LAGUNA_SWA_SPLIT_MIN_LIVE",
    "LAGUNA_SWA_SPLIT_GQA3_SCORES",
    "LAGUNA_SWA_FUSED_FIXED512",
    "LAGUNA_SWA_GQA3_LOCAL384_FIXED512",
    "LAGUNA_SWA_GQA3_VSTAGE64_FIXED512",
    "LAGUNA_SWA_GQA3_VSTAGE64_VEC16_FIXED512",
    "LAGUNA_SWA_GQA3_VSTAGE64_VEC16_DIRECT_FIXED512",
    "LAGUNA_SWA_GQA3_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP4_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP8_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP16_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VEC4_DENOM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_VALUE_TAIL_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_DENSE_RING",
    "LAGUNA_SWA_LOCAL1024",
    "LAGUNA_SWA_OUTPUT_SHARDED_PROBABILITY_DPP_QK",
    "LAGUNA_SWA_SPLIT_FIXED512_REDUCE",
    "LAGUNA_SWA_SPLIT_TILE16_MIN_LIVE",
    "LAGUNA_SWA_SPLIT_WAVE_LOCAL",
    "LAGUNA_SWA_PREFILL_ROLE_VARIANTS",
    "LAGUNA_SWA_PREFILL_VARIANT",
    "PARO_FULL_ATTN_NATIVE_EXACT_WIDTHS",
    "PARO_NATIVE_BATCH_DECODE_DEFAULT",
    "PARO_RETAINED_BATCH_DEFAULTS",
    "TARGET_ARCH",
    "register_backend_kernels",
    "register_gfx1151_kernels",
]
