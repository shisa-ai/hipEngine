# Q3_K per-tensor bisection (2026-09-11, review round 4)

Prompt: code_lru_cache (the failing position of the round-3 tokenized
category gate). Candidate arm = shipped prefill with the named Q3_K slot(s)
pinned to the strict GEMV, decode strict (isolates the Q3_K prefill delta
against the 4-quant incumbent). All seven Q3_K tensors of UD-Q4_K_M:

    blk.0.ffn_up   (5120, 17408)   blk.14.ffn_up  (5120, 17408)
    blk.13.ffn_down(17408, 5120)   blk.15.ffn_gate (5120, 17408)
    blk.13.ffn_gate (5120, 17408)  blk.16.ffn_gate (5120, 17408)

Leave-one-out (that slot strict, the other six routed):

    strict slot          mean      max
    (none - all routed)  2.21e-3   5.33e-2
    blk.0.ffn_up         3.47e-4   4.24e-3   <- dominant carrier
    blk.13.ffn_down      1.44e-3   2.82e-2
    blk.13.ffn_gate      1.50e-3   1.57e-2
    blk.14.ffn_gate      1.52e-3   3.29e-2
    blk.14.ffn_up        1.94e-3   2.23e-2
    blk.15.ffn_gate      1.67e-3   2.72e-2
    blk.16.ffn_gate      2.26e-3   3.29e-2
    (all strict)         0.0       0.0

No single tensor is the sole carrier - the 1-ULP accumulation-order class is
distributed - but blk.0.ffn_up (the earliest Q3_K tensor; its flips get the
most downstream amplification steps) accounts for ~85% of the mean and ~92%
of the max. The shipped policy pins exactly that slot
(GGUF_IQ_DENSE_PREFILL_STRICT_SLOTS); every other Q3_K tensor keeps the
W4A16 route. Full 18-prompt tokenized category gate with the pin: K_M
passes on seeds 7/11/23 (pooled max 2.81e-2, p99 1.86e-3, top-1 99.74%).
Cost: 936.4 -> 904.1 prefill tok/s (~3.4%); ~92% of the route's gain
retained. The next lever is a precision-preserving Q3_K owner (strict
accumulation association with shared decoded weights, or a
higher-precision split representation); cooperative geometry alone cannot
help - coop32 vs one-wave is bit-exact.
