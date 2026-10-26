"""Qwen3.5 MoE GGUF gfx1100 execution-profile plans.

Registers the architecture-scoped ``qwen3_5_moe_gguf`` plugin key for
``hip_gfx1100``. The strict plan is the exact fallback; the production plan is
the explicit non-default MoE candidate. Both reuse the gfx1100 dense plan's
binder environment and GDN chain, so a rename of this module cannot change the
resolved keys, selections, or runtime policy.
"""

from __future__ import annotations

from hipengine.execution_profiles import (
    ExecutionProfile,
    registered_runtime_profile_keys,
)
from hipengine.generation.qwen35_dense_gfx1100_profiles import (
    FP16_RECURRENT_STATE_ENV,
    Q4_FUSED_R28_ENV,
    Q6_DP4A_GROUPED_ENV,
    QWEN36_DENSE_GGUF_BACKEND,
    QWEN36_DENSE_GGUF_QUANT,
    VERIFY_CAPTURE_PREFILL_GDN_ENV,
    VERIFY_F32_POST_NORM_ENV,
    VERIFY_F32_RESIDUAL_ENV,
    _key,
    _production_binder,
    _register_profile,
)

QWEN36_MOE_GGUF_MODEL = "qwen3_5_moe_gguf"

_MOE_STRICT_EVIDENCE = (
    "benchmarks/results/2026-08-27-w7900-dual-concurrency2-mtp-current-state-audit.json"
)
_MOE_CANDIDATE_EVIDENCE = (
    "benchmarks/results/2026-08-27-w7900-35b-moe-generation2-mtp-c1-owner.json"
)


def register_qwen36_moe_gguf_gfx1100_profiles() -> bool:
    """Register strict fallback and the explicit non-default MoE candidate."""

    strict = _register_profile(
        model=QWEN36_MOE_GGUF_MODEL,
        profile=ExecutionProfile.STRICT,
        evidence=_MOE_STRICT_EVIDENCE,
        graph_policy="specdec2_eager_c1",
    )
    production = _register_profile(
        model=QWEN36_MOE_GGUF_MODEL,
        profile=ExecutionProfile.PRODUCTION,
        evidence=_MOE_CANDIDATE_EVIDENCE,
        graph_policy="specdec2_moe_bulk_f32_k2_strict_k1_candidate",
        binder=_production_binder,
    )
    return bool(strict or production)


def qwen36_moe_gguf_gfx1100_strict_registered() -> bool:
    return _key(
        ExecutionProfile.STRICT,
        model=QWEN36_MOE_GGUF_MODEL,
    ) in registered_runtime_profile_keys()


__all__ = [
    "FP16_RECURRENT_STATE_ENV",
    "QWEN36_DENSE_GGUF_BACKEND",
    "QWEN36_DENSE_GGUF_QUANT",
    "Q4_FUSED_R28_ENV",
    "Q6_DP4A_GROUPED_ENV",
    "QWEN36_MOE_GGUF_MODEL",
    "VERIFY_CAPTURE_PREFILL_GDN_ENV",
    "VERIFY_F32_POST_NORM_ENV",
    "VERIFY_F32_RESIDUAL_ENV",
    "qwen36_moe_gguf_gfx1100_strict_registered",
    "register_qwen36_moe_gguf_gfx1100_profiles",
]
