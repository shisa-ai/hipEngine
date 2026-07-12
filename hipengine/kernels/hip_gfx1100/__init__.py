"""gfx1100 / RDNA3 backend capabilities."""

# Clean W7900 SOL-G5 p512/d24 evidence admits the state-bound composite GGUF
# graph when at least 24 decode transitions amortize capture/instantiate/close.
GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS = 24

__all__ = ["GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS"]
