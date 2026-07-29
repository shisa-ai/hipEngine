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
# Protect router logits/selection before releasing the concurrent shared
# branch. Matched complete-state pp512 is +0.073% with 5/7 wins, and cached
# tracing verifies a 0.310-ms kernel-span reduction.
LAGUNA_MOE_SHARED_AFTER_ROUTER = True
# gfx1151 exposes least/greatest HIP stream priorities +1/-1. Running the
# after-router shared branch at +1 improves exact pp512 0.494% (6/7 wins) and
# cuts cached kernel span 7.255 ms while keeping 99.75% of shared work hidden.
LAGUNA_MOE_SHARED_LOW_PRIORITY = True
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
        (
            "laguna_attention_decode",
            "bf16",
            "global_context_split_exact_spans",
        ),
        (
            "laguna_attention_decode",
            "bf16",
            "global_context_split_exact_gated_spans",
        ),
        (
            "laguna_attention_decode",
            "bf16",
            "swa_context_split_exact_spans",
        ),
        (
            "laguna_attention_decode",
            "bf16",
            "swa_context_split_exact_gated_spans",
        ),
        (
            "laguna_attention_decode",
            "bf16",
            "swa_context_split_exact_gated_wave_local_spans",
        ),
        (
            "laguna_attention_decode",
            "bf16",
            "swa_context_split_exact_gated_wave_local_dim2_spans",
        ),
        (
            "laguna_attention_decode",
            "bf16",
            "swa_context_split_tile16_exact_spans",
        ),
        (
            "laguna_attention_decode",
            "bf16",
            "swa_context_split_tile16_exact_gated_spans",
        ),
        (
            "laguna_attention_decode",
            "bf16",
            "swa_context_split_tile16_exact_gated_wave_local_spans",
        ),
        (
            "laguna_attention_decode",
            "bf16",
            "swa_context_split_tile16_exact_gated_wave_local_dim2_spans",
        ),
        # Current-P4 head/KV fusion and the global-only wave-0 tree are
        # W7900-only until independently gated.
        (
            "head_rmsnorm+partial_rotary+kv_write",
            "laguna_f32_weight",
            "global_f32_bf16_spans",
        ),
        (
            "head_rmsnorm+partial_rotary+kv_write",
            "laguna_f32_weight",
            "global_wave0_tree_f32_bf16_spans",
        ),
        (
            "head_rmsnorm+partial_rotary+kv_write",
            "laguna_f32_weight",
            "swa_f32_bf16_spans",
        ),
        # D9, its wave-0 RMS tree, and its exact top-10 split sibling are
        # W7900-only until gfx1151 receives independent correctness and
        # performance gates.
        (
            "moe_tail+next_rmsnorm",
            "bf16",
            "laguna_aggregate_gguf_f32_weight_out",
        ),
        (
            "moe_tail+next_rmsnorm",
            "bf16",
            "laguna_aggregate_wave0_tree_gguf_f32_weight_out",
        ),
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
        # Laguna c=1 F32-router wave-0 reduction is W7900-only pending an
        # independent gfx1151 correctness and performance gate.
        (
            "router_logits",
            "f32",
            "bf16_hidden_wave0_tree",
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
    "GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q6_LM_HEAD_MAX_CHUNK",
    "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS",
    "GGUF_Q8_T16_DECODE_ROWTILE_ALL",
    "GGUF_Q8_T16_PREFILL_FOUR_WAVE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS",
    "GGUF_RAW_K_PREFILL_COLTILE2_SHAPES",
    "GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED",
    "GGUF_RAW_K_PREFILL_ROWBATCH",
    "GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED",
    "GGUF_RAW_K_PREFILL_VARIANT",
    "GGUF_ROUTER_F32_BF16_HIDDEN_THREADS",
    "LAGUNA_DENSE_Q4_PREFILL_MODE",
    "LAGUNA_F16_BOUNDARY_FUSION",
    "LAGUNA_F16_PREFILL_MIN_ROWS",
    "LAGUNA_F16_PREFILL_MODE",
    "LAGUNA_F16_PREFILL_STRATEGY",
    "LAGUNA_GLOBAL_PREFILL_VARIANT",
    "LAGUNA_MOE_BRANCH_CONCURRENCY",
    "LAGUNA_MOE_GROUP_COMPACT_MODE",
    "LAGUNA_MOE_SHARED_AFTER_ROUTER",
    "LAGUNA_MOE_SHARED_LOW_PRIORITY",
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
    "LAGUNA_ROUTER_LOGITS_MODE",
    "LAGUNA_SELECTED_DOWN_MODE",
    "LAGUNA_SELECTED_GATE_UP_MODE",
    "LAGUNA_SWA_PREFILL_VARIANT",
    "PARO_FULL_ATTN_NATIVE_EXACT_WIDTHS",
    "PARO_NATIVE_BATCH_DECODE_DEFAULT",
    "PARO_RETAINED_BATCH_DEFAULTS",
    "TARGET_ARCH",
    "register_backend_kernels",
    "register_gfx1151_kernels",
]
