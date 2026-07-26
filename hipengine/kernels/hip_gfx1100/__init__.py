"""gfx1100 / RDNA3 backend capabilities."""

# Clean W7900 context/category gates retain the exact token4 score-parallel SWA
# decode default. The wider token8 screen failed the every-category h16 gate and
# was removed; other backends retain the separately registered baseline.
LAGUNA_SWA_DECODE_VARIANT = "swa_context_token4_exact_spans"
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
# Clean P0 leaf/full-state/context/category evidence admits one exact local32
# wave per (route, output), followed by the registered slot-order reducer.
# Other backends retain the serial weighted composite.
LAGUNA_IQ3_C1_DOWN_SCHEDULE = "wave4_reduce"
# The exact expanded-magnitude IQ2 grid contracts selector reconstruction and
# passes actual-layer, full-state, clean-context, and complete-category gates.
# Rows>1 and other backends retain the compact-grid tile2 fallback.
LAGUNA_IQ2_GRID64 = True
# The fixed-local64 DPP reduction is admitted only as an explicit W7900 c=1
# screen. No-argument sessions retain the measured grid64 reduction owner.
LAGUNA_IQ2_LOCAL64_REDUCTION = False
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
    "LAGUNA_IQ2_GRID64",
    "LAGUNA_IQ2_LOCAL64_REDUCTION",
    "LAGUNA_IQ3_C1_DOWN_SCHEDULE",
    "LAGUNA_MIXED_ATTENTION_PROJECTIONS",
    "LAGUNA_MIXED_LOCAL32_FIXED_METADATA",
    "LAGUNA_MIXED_Q6_FIXED_METADATA",
    "LAGUNA_Q4_LM_HEAD_LOCAL32_FIXED_METADATA",
    "LAGUNA_Q5_FIXED_METADATA",
    "LAGUNA_Q5_SHARED_FIXED_METADATA",
    "LAGUNA_Q5_WAVE32X2_OUTPUT",
    "LAGUNA_Q5_WAVE32X2_QUERY_GATE",
    "LAGUNA_SPLIT_GATE_FUSION",
    "LAGUNA_SWA_DECODE_VARIANT",
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
    "GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q6_LM_HEAD_MAX_CHUNK",
    "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS",
    "GGUF_Q8_T16_DECODE_ROWTILE_ALL",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS",
    "GGUF_ROUTER_F32_BF16_HIDDEN_THREADS",
]
