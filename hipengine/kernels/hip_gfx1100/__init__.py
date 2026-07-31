"""gfx1100 / RDNA3 backend capabilities."""

# Clean W7900 context/category gates retain the exact token4 score-parallel SWA
# decode default. The wider token8 screen failed the every-category h16 gate and
# was removed; other backends retain the separately registered baseline.
LAGUNA_SWA_DECODE_VARIANT = "swa_context_token4_exact_spans"
# WPF-3's exact adjacent-row SWA policy keeps wave32 below the measured
# crossover and selects qrow4 only for complete M128 tiles at position 256+.
# Complete M512 state is KL0 and both 512/1K same-weight gates are positive;
# explicit local128/wave32/qrow2/qrow4 variants remain registered rollbacks.
LAGUNA_SWA_PREFILL_VARIANT = "swa_context_rows_qrow4_m128_c256_exact_spans"
# WPF-H5M promotes exact source-qualified qrow4 loads after KL0 complete state,
# all 72 integrated role calls, and positive clean 512/1K/4K timing. Every
# explicit route plus shape, registration, and backend misses retain WPF-3.
LAGUNA_SWA_PREFILL_ROLE_VARIANTS = {
    "qrow4_m128_c256_exact": (
        "swa_context_rows_qrow4_sourcequal_exact_spans"
    ),
}
# WPF-H6A promotes both exact dense-initial cached-only roles after KL0 complete
# state, exact 144-SWA/48-global integrated topology, and positive default-off
# 512/1K/4K gates. H5R/H5U remain registered rollback leaves; explicit global
# and SWA selectors plus shape/metadata/registration/backend misses fail closed.
LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS = {
    "global_m128_c4096_first_fill_exact": (
        "global_context_rows_dense_initial_cached_exact_spans"
    ),
    "swa_qrow4_m128_c512_no_wrap_exact": (
        "swa_context_rows_qrow4_dense_initial_cached_exact_spans"
    ),
}
# WPF-H5R promotes exact cached-only SWA attention after KL0 complete state,
# all 144 physical append-before-attention calls, and positive default-off clean
# 512/1K/4K timing. The package-only role map restricts schedule reordering to
# complete no-wrap M128 starts; global, partial, wrapped, explicit, missing, and
# unsupported routes retain attend-before-append fallbacks.
LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS = {
    "swa_qrow4_m128_c512_no_wrap_exact": (
        "swa_context_rows_qrow4_cached_exact_spans"
    ),
}
LAGUNA_PREFILL_KV_PREAPPEND = True
# Clean W7900 D12 leaf/profile/category evidence admits the exact local32
# two-output Q5 schedules for c=1 attention output and query/gate projection.
# Other backends retain the separately registered pack8 fallbacks.
LAGUNA_Q5_WAVE32X2_OUTPUT = True
LAGUNA_Q5_WAVE32X2_QUERY_GATE = True
# Exact wave-uniform Q5 metadata loads pass first/last actual layers, full state,
# both clean context orders, and both complete 18-prompt category orders.
# Explicit role-scoped disables retain the registered coefficient-publication path.
LAGUNA_Q5_FIXED_METADATA = True
# The exact local32 BF16 pair for shared-Q5 gate/up passes first/last actual
# weights, full state, both clean context orders, and both complete category
# orders. Explicit disable restores the registered local128 pack8 pair.
LAGUNA_Q5_SHARED_FIXED_METADATA = True
# Exact mixed Q5/Q6 and corrected Q6/Q8 projection quads pass actual layers,
# full state, both clean context orders, and both complete category orders.
# Explicit disable retains the registered Q5/Q6 pair and Q8 singleton chain.
LAGUNA_MIXED_ATTENTION_PROJECTIONS = True
# Cooperative Q6 metadata publication inside the retained mixed quad is exact,
# improves all clean contexts, and passes both complete category process orders.
# Explicit disable restores the generic-Q6 mixed projection variant.
LAGUNA_MIXED_Q6_FIXED_METADATA = True
# Exact all-local32 Q5/Q6 ownership passes production bits, full state, both
# clean context orders, and both complete category orders. Explicit disable
# restores the retained local128 fixed-Q6 mixed projection; layer 47 stays there.
LAGUNA_MIXED_LOCAL32_FIXED_METADATA = True
# The exact Q4_K c=1 BF16-to-F32 LM-head sibling passes production bits,
# full state, cached one-call tracing, both clean context orders, and both
# complete category process orders. Explicit False restores registered local128.
LAGUNA_Q4_LM_HEAD_LOCAL32_FIXED_METADATA = True
# The registered wave4 producer plus slot-order reducer remains the explicit
# rollback and exact-key-miss fallback for the retained fused schedule below.
# Other backends retain their independently selected schedules.
LAGUNA_IQ3_C1_DOWN_SCHEDULE = "wave4_reduce"
# Exact local320 ten-wave IQ3 producer/reducer fusion passes full-state,
# clean-context, and both complete 18-prompt category orders. Explicit wave4
# or serial requests and exact-key misses retain the registered unfused chain.
LAGUNA_IQ3_WAVE10_FUSED = True
# The exact expanded-magnitude IQ2 grid contracts selector reconstruction and
# passes actual-layer, full-state, clean-context, and complete-category gates.
# Rows>1 and other backends retain the compact-grid tile2 fallback.
LAGUNA_IQ2_GRID64 = True
# Clean P2.1 exact split profiles and the complete category/heldout gate admit
# independent global/SWA crossovers. Registered single-block readers remain the
# below-threshold and explicit-disable fallback on gfx1100; other backends do
# not inherit these architecture-local thresholds.
LAGUNA_GLOBAL_SPLIT_MIN_LIVE = 127
LAGUNA_SWA_SPLIT_MIN_LIVE = 65
# P2's exact post-online refinement reduces the retained SWA score-producer
# sub-window at 257/511/512 without changing dispatches, memory, or arithmetic.
LAGUNA_SWA_SPLIT_TILE16_MIN_LIVE = 257
# P4.1 exact actual-layer screens admit folding the existing softplus gate and
# BF16 store into each retained split reducer. The unfused registered chain is
# the explicit rollback and remains the below-threshold fallback.
LAGUNA_SPLIT_GATE_FUSION = True
# The retained SWA reducer replays exact scalar softmax statistics independently
# in each logical wave. This removes all block barriers/LDS and improves every
# clean context plus the complete two-order category suite.
LAGUNA_SWA_SPLIT_WAVE_LOCAL = True
# Current-P4 first/last-layer, clean context, full-state, and complete two-order
# category evidence admits exact c=1 head RMSNorm+RoPE+BF16 KV append fusion.
# Rows/prefill and other backends retain the registered two-launch fallback.
LAGUNA_HEAD_KV_FUSION = True

# Clean W7900 SOL-G5 p512/d24 evidence admits the state-bound composite GGUF
# graph when at least 24 decode transitions amortize capture/instantiate/close.
GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS = 24
# LCP-5A's clean peer-aligned semantic/decode contract and 512/4K floors admit
# the llama.cpp-HIP-shaped normalized-Q/K wave32 recurrence on gfx1100.
# Scalar-exact direct LDS32 remains available through the explicit selector.
GGUF_GDN_PREFILL_AUTO_MODE = "chain_peer_wave32"
# Strict-exact rollback/oracle stays architecture-scoped and does not replace
# the quality-admitted peer-wave production default.
GGUF_GDN_PREFILL_EXACT_MODE = "chain_lds32_direct_nonvolatile"
# The singleton-indexed packed-AR recurrence is retained only on independently
# measured backends. gfx1100 keeps the arbitrary-segment fallback by default.
GGUF_GDN_INDEXED_SINGLETON_DECODE = False
# Exact Q8T16 row-amortized decode remains explicit on gfx1100 until an
# independent native-AR width gate passes on W7900.
GGUF_Q8_T16_DECODE_ROWTILE_ALL = False
# The exact 128-thread c8 pair schedule is admitted only on independently
# measured backends. Zero disables automatic pair rowtiling on gfx1100.
GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS = 0
# Dynamic selected-expert pair reuse remains disabled until independently
# measured on W7900. Zero preserves the existing gfx1100 route.
GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS = 0
# Q5T16 selected-down pair reuse also requires an independent W7900 gate.
GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS = 0
GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS = 0
GGUF_Q6_LM_HEAD_MAX_CHUNK = 6
# Clean W7900 GPF-3A full-model 512/4K evidence admits byte-exact shared-X
# selected-dual Q4T16 prefill after the predeclared borderline-decode repeat.
GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE = "shared_x"
# Clean W7900 GPF-5A 512/4K transfer evidence admits the exact two-wave dense
# Q8T16 schedule only through the independently measured request scope.
GGUF_Q8_T16_PREFILL_TWO_WAVE = True
GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS = 4096
# WPF-2b's exact expert-major IQ2 gate/up route compounds grouped IQ3/IQ4 down
# at M512 with both attention capacities fixed at 128. Complete state is KL0 and
# dirty paired 512/1K admission improves 19.65%/18.01%. Explicit grouped_exact
# gate/up preserves the preceding route-major rollback; direct/M128/M256 remain
# exact fallbacks for unsupported quant/shape keys.
LAGUNA_PREFILL_MATRIX_ROWS = 512
LAGUNA_SELECTED_GATE_UP_MODE = "grouped_pair16"
LAGUNA_SELECTED_DOWN_MODE = "grouped_exact"
# WPF-H6C promotes the exact K3072/N1024/E256 fused-SiLU rowbatch4 leaf for
# only the qualified layer-47 IQ3 role after exact state, physical topology,
# clean 512/1K/4K, and matched C4096/M512 wins. An empty role map remains the
# scoped rollback. Wrong role/shape/quant/registration/backend and c=1 retain
# the exact grouped route-major chain, while IQ2 keeps promoted pair16.
_H6C_IQ3_GATE_UP_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_"
    "k3072_n1024_e256_rowbatch4_bf16_bf16_out"
)
LAGUNA_GROUPED_GATE_UP_ROLE_VARIANTS = {
    "layer47_iq3_k3072_n1024_e256": _H6C_IQ3_GATE_UP_VARIANT
}
LAGUNA_GROUPED_GATE_UP_VARIANT_ABIS = {
    _H6C_IQ3_GATE_UP_VARIANT: "grouped_raw_iq_dual_silu"
}
# WPF-H5Z promotes exact K1024/N3072 activation-resident P256 IQ3 ownership
# after KL0 complete state, exact 45-call integration, and positive selector-
# unset 512/1K/4K timing. H5Q remains registered rollback through the same
# bounded active-expert ABI. H6D promotes exact row-interleaved VOPD source
# ownership through that unchanged ABI after complete state, paired topology,
# and both short/matched gates; H5Z stays registered rollback. H5J is unchanged.
_H5Q_IQ3_ACTIVE_EXPERT_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "resident_rowbatch8_bf16_bf16_out"
)
_H5Z_IQ3_ACTIVATION_RESIDENT_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_rowbatch8_bf16_bf16_out"
)
_H6D_IQ3_ROW_INTERLEAVED_VOPD_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_rowbatch8_bf16_bf16_out"
)
LAGUNA_GROUPED_IQ_DOWN_VARIANTS = {
    "gguf_iq3_xxs": _H6D_IQ3_ROW_INTERLEAVED_VOPD_VARIANT,
    "gguf_iq4_xs": (
        "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out"
    ),
}
LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS = {
    _H5Q_IQ3_ACTIVE_EXPERT_VARIANT: "grouped_raw_iq_active_experts",
    _H5Z_IQ3_ACTIVATION_RESIDENT_VARIANT: "grouped_raw_iq_active_experts",
    _H6D_IQ3_ROW_INTERLEAVED_VOPD_VARIANT: "grouped_raw_iq_active_experts",
}
# WPF-1 established exact Q5/Q6 rowbatch8 after bit-exact full-state and short
# admission. WPF-1W promotes rowbatch32 after clean paired gains at both short
# shapes. WPF-1T's exact constant-accumulator screen admits four adjacent output
# columns for full RB32 slabs, then qualifies `(2,16)` only for four measured
# `(quant, output variant, K, N)` configurations. Explicit rowbatch preserves
# the preceding exact owner; zero/4/8/16/32 and unsupported widths remain
# fallbacks.
GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED = True
GGUF_RAW_K_PREFILL_ROWBATCH = 32
GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED = True
GGUF_RAW_K_PREFILL_COLTILE2_SHAPES = frozenset(
    {
        ("gguf_q5_k", "bf16_bf16_out", 3072, 12288),
        ("gguf_q5_k", "bf16_f32_out", 3072, 6144),
        ("gguf_q5_k", "bf16_f32_out", 3072, 9216),
        ("gguf_q6_k", "bf16_f32_out", 3072, 9216),
    }
)
GGUF_RAW_K_PREFILL_VARIANT = "coltile"
# WPF-H5D promotes the exact H5C producer/ordered-consumer chain after KL0,
# byte-identical complete state, and clean +7.235%/+6.519% M512/M1024 gates.
# H5E doubles output ownership on six role-qualified geometries, reducing the
# producer-inclusive weighted Q5 family another 12.32% by events / 7.52% by
# synchronized wall while remaining byte-exact. H5F's exact 12x4 N48 role saves
# another 4.224/1.989 us per M512 request. H5G's exact constant-80/96 tiles
# improve clean 512/1K/4K another 2.192%/2.055%/1.329% on five roles. H5L's
# exact weight-major traversal cuts the six-role/235-call leaf 44.857%/46.544%
# by event/wall. H5X promotes tile-K-col layout on 151 calls after KL0 complete
# state. H5Y then adds an exact tile-K-row BF16 activation plane on all 188
# ordered-Q5 calls, preserving H5X/H5L weight layouts. Selector-unset one-queue
# 512/1K/4K improves 10.862%/8.969%/5.829%; N48/N72 and every miss retain
# H5G/raw exact routes.
GGUF_Q5_F32_ORDERED_PREFILL = True
GGUF_Q5_F32_ORDERED_PREFILL_POLICY = {
    ("bf16", 3072, 1024): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile8_rowbatch4"
    ),
    ("bf16", 3072, 12288): (
        "weight_major_row_major_activation_tile_k_row_coltile8_rowbatch12"
    ),
    ("bf16", 6144, 3072): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile16_rowbatch5"
    ),
    ("bf16", 9216, 3072): (
        "weight_major_row_major_activation_tile_k_row_coltile12_rowbatch8"
    ),
    ("f32", 3072, 48): "coltile12_rowbatch4",
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 6144): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile16_rowbatch5"
    ),
    ("f32", 3072, 9216): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile8_rowbatch10"
    ),
}
# WPF-H5I introduced the shared serial F32 plane and ordered Q6 consumers.
# WPF-H5W retains exact weight-major rollback. WPF-H6E promotes activation-row
# traversal on 142/143 selected calls after KL0 complete-state, exact integrated
# topology, and positive one-queue 512/1K/4K gates at unchanged workspace. F32
# N72 and long-K/wide-N misses retain the H5I/raw exact fallbacks.
GGUF_Q6_F32_ORDERED_PREFILL = True
GGUF_Q6_F32_ORDERED_PREFILL_POLICY = {
    ("bf16", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch5"
    ),
    ("bf16", 1024, 3072): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch4"
    ),
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch5"
    ),
}
GGUF_Q6_F32_ORDERED_PREFILL_H6E_POLICY = {
    ("bf16", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch5"
    ),
    ("bf16", 1024, 3072): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch4"
    ),
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch5"
    ),
}
GGUF_F32_ORDERED_PREFILL_QUANTS = frozenset(("gguf_q5_k", "gguf_q6_k"))
GGUF_F32_ORDERED_PREFILL_POLICIES = {
    "gguf_q5_k": GGUF_Q5_F32_ORDERED_PREFILL_POLICY,
    "gguf_q6_k": GGUF_Q6_F32_ORDERED_PREFILL_POLICY,
}
# LCP-2B removes the 512-token compact-MoE scheduler's per-layer scalar D2H
# boundary using a routing-independent tight padded-row upper bound. Larger
# selected-row shapes keep the exact scalar read until independently measured.
GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS = 4096
# Clean W7900 LCP-4A transfer evidence admits the exact 256-thread
# BF16-hidden/F32-weight router-logits geometry for bulk prefill and decode.
GGUF_ROUTER_F32_BF16_HIDDEN_THREADS = 256
# Clean W7900 LCP-4B transfer evidence admits 128 threads for the bulk-prefill
# top-k selector. Decode keeps its independently selected launch geometry.
GGUF_PREFILL_ROUTER_SELECT_THREADS = 128
# Clean W7900 LCP-M2 transfer evidence admits one stream-ordered metadata
# preparation kernel in place of six synchronous H2D copies through 4K.
GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS = 4096
# Clean W7900 LCP-D2 correctness and 32K/64K/128K graph-decode gates admit the
# split-parallel gated reduction from 32K onward. Shorter contexts retain the
# single-launch serial reducer because the extra prepare launch is neutral/down.
GGUF_PAGED_ATTN_PARALLEL_REDUCE = True
GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT = 32768
# LCP-M1 validates an aligned phase-liveness arena for the production Qwen3.6
# MoE prefill route. Diagnostic/F32 layouts retain dedicated allocations.
GGUF_PREFILL_SCRATCH_LIVENESS_ALIAS = True
# LCP-1 remains a separately registered diagnostic on gfx1100 because its
# architecture-local full-state and wall gate rejected automatic promotion.
GGUF_LINEAR_ATTN_CONV_PREFILL_AUTO_MODE = "baseline"

__all__ = [
    "LAGUNA_GLOBAL_SPLIT_MIN_LIVE",
    "LAGUNA_HEAD_KV_FUSION",
    "LAGUNA_GROUPED_GATE_UP_ROLE_VARIANTS",
    "LAGUNA_GROUPED_GATE_UP_VARIANT_ABIS",
    "LAGUNA_GROUPED_IQ_DOWN_VARIANTS",
    "LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS",
    "LAGUNA_IQ2_GRID64",
    "LAGUNA_IQ3_C1_DOWN_SCHEDULE",
    "LAGUNA_IQ3_WAVE10_FUSED",
    "LAGUNA_MIXED_ATTENTION_PROJECTIONS",
    "LAGUNA_MIXED_LOCAL32_FIXED_METADATA",
    "LAGUNA_MIXED_Q6_FIXED_METADATA",
    "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS",
    "LAGUNA_PREFILL_KV_PREAPPEND",
    "LAGUNA_PREFILL_MATRIX_ROWS",
    "LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS",
    "LAGUNA_Q4_LM_HEAD_LOCAL32_FIXED_METADATA",
    "LAGUNA_Q5_FIXED_METADATA",
    "LAGUNA_Q5_SHARED_FIXED_METADATA",
    "LAGUNA_Q5_WAVE32X2_OUTPUT",
    "LAGUNA_Q5_WAVE32X2_QUERY_GATE",
    "LAGUNA_SELECTED_DOWN_MODE",
    "LAGUNA_SELECTED_GATE_UP_MODE",
    "LAGUNA_SPLIT_GATE_FUSION",
    "LAGUNA_SWA_DECODE_VARIANT",
    "LAGUNA_SWA_PREFILL_ROLE_VARIANTS",
    "LAGUNA_SWA_PREFILL_VARIANT",
    "LAGUNA_SWA_SPLIT_MIN_LIVE",
    "LAGUNA_SWA_SPLIT_TILE16_MIN_LIVE",
    "LAGUNA_SWA_SPLIT_WAVE_LOCAL",
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
    "GGUF_PREFILL_SCRATCH_LIVENESS_ALIAS",
    "GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE",
    "GGUF_F32_ORDERED_PREFILL_POLICIES",
    "GGUF_F32_ORDERED_PREFILL_QUANTS",
    "GGUF_Q5_F32_ORDERED_PREFILL",
    "GGUF_Q5_F32_ORDERED_PREFILL_POLICY",
    "GGUF_Q6_F32_ORDERED_PREFILL",
    "GGUF_Q6_F32_ORDERED_PREFILL_H6E_POLICY",
    "GGUF_Q6_F32_ORDERED_PREFILL_POLICY",
    "GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q6_LM_HEAD_MAX_CHUNK",
    "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS",
    "GGUF_Q8_T16_DECODE_ROWTILE_ALL",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS",
    "GGUF_RAW_K_PREFILL_COLTILE2_SHAPES",
    "GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED",
    "GGUF_RAW_K_PREFILL_ROWBATCH",
    "GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED",
    "GGUF_RAW_K_PREFILL_VARIANT",
    "GGUF_ROUTER_F32_BF16_HIDDEN_THREADS",
]
