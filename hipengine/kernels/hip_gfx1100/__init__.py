"""gfx1100 / RDNA3 backend capabilities."""

# Clean W7900 SOL-G5 p512/d24 evidence admits the state-bound composite GGUF
# graph when at least 24 decode transitions amortize capture/instantiate/close.
GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS = 24
# LCP-5A's clean peer-aligned semantic/decode contract and 512/4K floors admit
# the llama.cpp-HIP-shaped normalized-Q/K wave32 recurrence on gfx1100.
# Scalar-exact direct LDS32 remains available through the explicit selector.
GGUF_GDN_PREFILL_AUTO_MODE = "chain_peer_wave32"
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
# Clean W7900 LCP-D2 correctness and 32K/64K/128K graph-decode gates admit the
# split-parallel gated reduction from 32K onward. Shorter contexts retain the
# single-launch serial reducer because the extra prepare launch is neutral/down.
GGUF_PAGED_ATTN_PARALLEL_REDUCE = True
GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT = 32768
# LCP-M1 validates an aligned phase-liveness arena for the production Qwen3.6
# MoE prefill route. Diagnostic/F32 layouts retain dedicated allocations.
GGUF_PREFILL_SCRATCH_LIVENESS_ALIAS = True

__all__ = [
    "GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS",
    "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS",
    "GGUF_GDN_PREFILL_AUTO_MODE",
    "GGUF_PAGED_ATTN_PARALLEL_REDUCE",
    "GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT",
    "GGUF_PREFILL_SCRATCH_LIVENESS_ALIAS",
    "GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS",
]
