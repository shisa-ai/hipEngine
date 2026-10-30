"""Backend identity and capability policy declarations for hip_gfx1151.

This module owns the gfx1151 backend key, target architecture, admission layer
surface, and the non-GGUF capability/policy defaults (Qwen4Exp graph and PARO
batch width). It is a move-only extraction from
``hipengine.kernels.hip_gfx1151``: declaration values, provenance comments, and
default/fallback semantics are unchanged. The backend package re-exports every
declaration explicitly for import compatibility.

The declarations are data only. They import no kernel bodies, touch no registry,
and select no variant at import time; backend registration remains owned by
``hipengine.kernels.hip_gfx1151``.
"""

from __future__ import annotations

from hipengine.kernels.backends import hip_target_arch_for_backend


BACKEND = "hip_gfx1151"

# The registry variant name for the LDS-staged f32 dense Q8_0 tile. Exported so
# the runtime dispatch and the gate scripts name the same string instead of
# repeating it.
STAGED_F32_VARIANT = "staged_f32_bm128_bn64_bk32"
TARGET_ARCH = hip_target_arch_for_backend(BACKEND)

# Concrete GGUF consumer layers this backend registers (UD-U1 F3 admission
# metadata; see the hip_gfx1100 declaration for the contract).  The gfx1151
# alias registrar covers the full certified consumer surface (its exclusion
# list removes only non-admission tuning variants), verified by the parity
# test that checks every certified consumer key is registered here too.
GGUF_CONSUMER_LAYERS: frozenset[str] = frozenset(
    {
        "linear",
        "dense_gemv",
        "embedding",
        "rmsnorm",
        "router_logits",
        "gdn_recurrent_rmsnorm_gate",
        "gdn_prefill_recurrent",
        "linear_attn_conv_decode",
        "linear_attn_conv_prefill",
        "moe_linear",
    }
)


# Qwen4Exp MoE is stateless between layer boundaries. Per-layer graph replay
# self-validates BF16 output before admission and leaves GDN/QSA eager.
QWEN4_EXP_MOE_GRAPH = True
# Profile-certified T2 cooperative MoE prefill is confined to late layers where
# the full 450-row/three-repeat envelope and task/c2 gates pass.
QWEN4_EXP_PRODUCTION_MOE_PREFILL_LAYERS = tuple(range(27, 48))
QWEN4_EXP_PRODUCTION_Q8_PREFILL_LAYERS = tuple(range(32, 48))
QWEN4_EXP_PRODUCTION_Q8_PREFILL_TILE = (64, 32)


# Clean PARO G3/G5 physical-width and server gates certify c4/c8 with whole-row
# full-attention execution. Diagnostic c2 row chunking changes row-local
# numerics at these widths and must therefore remain an explicit override.
PARO_FULL_ATTN_NATIVE_EXACT_WIDTHS = frozenset({4, 8})
# The shared packed-PARO adapter implements one request with one candidate.
# It uses an eager verifier and a spare resident slot for transactional restore.
PARO_SPECDEC2_MTP2_C1 = True
# G5 retains p512/d128 blocking and SSE c1/c2/c4/c8 scaling, delayed c4->c8
# admission, serial-c8 control, and repeated c8 exactness. Package capabilities
# select those identity-matched widths by default without branching in model or
# engine code; the legacy env flags remain explicit rollback opt-outs.
PARO_RETAINED_BATCH_DEFAULTS = True
PARO_NATIVE_BATCH_DECODE_DEFAULT = True


__all__ = [
    "BACKEND",
    "GGUF_CONSUMER_LAYERS",
    "PARO_FULL_ATTN_NATIVE_EXACT_WIDTHS",
    "PARO_NATIVE_BATCH_DECODE_DEFAULT",
    "PARO_RETAINED_BATCH_DEFAULTS",
    "PARO_SPECDEC2_MTP2_C1",
    "QWEN4_EXP_MOE_GRAPH",
    "QWEN4_EXP_PRODUCTION_MOE_PREFILL_LAYERS",
    "QWEN4_EXP_PRODUCTION_Q8_PREFILL_LAYERS",
    "QWEN4_EXP_PRODUCTION_Q8_PREFILL_TILE",
    "STAGED_F32_VARIANT",
    "TARGET_ARCH",
]
