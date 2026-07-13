"""gfx1100 / RDNA3 backend capabilities."""

# Clean W7900 SOL-G5 p512/d24 evidence admits the state-bound composite GGUF
# graph when at least 24 decode transitions amortize capture/instantiate/close.
GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS = 24
# GPF-2D has not received an independent W7900 transfer gate. Keep the proven
# fused GDN prefill route as gfx1100's architecture-scoped automatic policy.
GGUF_GDN_PREFILL_AUTO_MODE = "fused"
# GPF-3A has only gfx1151 family-local replay evidence so far. Keep the
# established Q4T16 compact32 kernel as the automatic policy on gfx1100.
GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE = "baseline"

__all__ = [
    "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS",
    "GGUF_GDN_PREFILL_AUTO_MODE",
    "GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE",
]
