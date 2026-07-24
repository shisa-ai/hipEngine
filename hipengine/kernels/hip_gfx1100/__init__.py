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
# D17's token8 head/KV and attention/gate boundaries remain explicit opt-in
# until the frozen clean-profile and full-suite retention gates pass.
LAGUNA_ATTENTION_BOUNDARY_FUSION = False

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
    "LAGUNA_ATTENTION_BOUNDARY_FUSION",
    "LAGUNA_Q5_WAVE32X2_OUTPUT",
    "LAGUNA_Q5_WAVE32X2_QUERY_GATE",
    "LAGUNA_SWA_DECODE_VARIANT",
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
