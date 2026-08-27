"""Qwen3.6 dense GGUF gfx1100 strict SPECDEC2 profile."""

from __future__ import annotations

import os
from typing import Any

from hipengine.execution_profiles import (
    ExecutionProfile,
    ResolvedRuntimeProfile,
    RuntimeProfileKey,
    RuntimeProfilePlan,
    VariantSelection,
    register_runtime_profile_plan,
    registered_runtime_profile_keys,
)

QWEN36_DENSE_GGUF_MODEL = "qwen3_5_gguf"
QWEN36_MOE_GGUF_MODEL = "qwen3_5_moe_gguf"
QWEN36_DENSE_GGUF_BACKEND = "hip_gfx1100"
QWEN36_DENSE_GGUF_QUANT = "gguf_q4_k_m"
FP16_RECURRENT_STATE_ENV = "HIPENGINE_GGUF_FP16_RECURRENT_STATE"
VERIFY_CAPTURE_PREFILL_GDN_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"
VERIFY_F32_RESIDUAL_ENV = "HIPENGINE_GGUF_VERIFY_F32_RESIDUAL"
VERIFY_F32_POST_NORM_ENV = "HIPENGINE_GGUF_VERIFY_F32_POST_NORM"

_GDN_CHAIN_VARIANT = "bf16_c1_exact_state_rows_tloop"
_GDN_REGISTRY_QUANT = "gguf_qwen35"
_DENSE_STRICT_EVIDENCE = (
    "benchmarks/results/2026-08-23-w7900-qwen36-27b-current-default-publication.json"
)
_MOE_STRICT_EVIDENCE = (
    "benchmarks/results/2026-08-27-w7900-dual-concurrency2-mtp-current-state-audit.json"
)
_MOE_CANDIDATE_EVIDENCE = (
    "benchmarks/results/2026-08-27-w7900-35b-moe-generation2-mtp-c1-owner.json"
)


def _strict_binder(generator: Any, resolved: ResolvedRuntimeProfile) -> None:
    del generator, resolved
    os.environ[FP16_RECURRENT_STATE_ENV] = "0"
    os.environ[VERIFY_CAPTURE_PREFILL_GDN_ENV] = "1"
    os.environ[VERIFY_F32_RESIDUAL_ENV] = "0"
    os.environ[VERIFY_F32_POST_NORM_ENV] = "0"


def _production_binder(generator: Any, resolved: ResolvedRuntimeProfile) -> None:
    _strict_binder(generator, resolved)
    os.environ[VERIFY_F32_RESIDUAL_ENV] = "1"
    os.environ[VERIFY_F32_POST_NORM_ENV] = "1"


def _key(profile: ExecutionProfile, *, model: str) -> RuntimeProfileKey:
    return RuntimeProfileKey(
        model=str(model),
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile=profile,
    )


def _register_profile(
    *,
    model: str,
    profile: ExecutionProfile,
    evidence: str,
    graph_policy: str,
    binder: Any = _strict_binder,
) -> bool:
    key = _key(profile, model=model)
    if key in registered_runtime_profile_keys():
        return False
    register_runtime_profile_plan(
        model=model,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile=profile,
        plan=RuntimeProfilePlan(
            selections=(
                VariantSelection(
                    layer="linear_attn_chain_conv_decode",
                    scope="specdec2_mtp2_c1",
                    selected_variant=_GDN_CHAIN_VARIANT,
                    strict_fallback_variant=_GDN_CHAIN_VARIANT,
                    registry_quant=_GDN_REGISTRY_QUANT,
                    evidence_artifact=evidence,
                ),
            ),
            kv_policy="paged_bf16",
            graph_policy=str(graph_policy),
            binder=binder,
        ),
    )
    return True


def register_qwen36_dense_gguf_gfx1100_profiles() -> bool:
    """Register the W7900 strict FP32-state dense NextN control once."""

    return _register_profile(
        model=QWEN36_DENSE_GGUF_MODEL,
        profile=ExecutionProfile.STRICT,
        evidence=_DENSE_STRICT_EVIDENCE,
        graph_policy="specdec2_eager_c1",
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


def qwen36_dense_gguf_gfx1100_strict_registered() -> bool:
    return _key(
        ExecutionProfile.STRICT,
        model=QWEN36_DENSE_GGUF_MODEL,
    ) in registered_runtime_profile_keys()


def qwen36_moe_gguf_gfx1100_strict_registered() -> bool:
    return _key(
        ExecutionProfile.STRICT,
        model=QWEN36_MOE_GGUF_MODEL,
    ) in registered_runtime_profile_keys()


__all__ = [
    "FP16_RECURRENT_STATE_ENV",
    "QWEN36_DENSE_GGUF_BACKEND",
    "QWEN36_DENSE_GGUF_MODEL",
    "QWEN36_DENSE_GGUF_QUANT",
    "QWEN36_MOE_GGUF_MODEL",
    "VERIFY_CAPTURE_PREFILL_GDN_ENV",
    "VERIFY_F32_POST_NORM_ENV",
    "VERIFY_F32_RESIDUAL_ENV",
    "qwen36_dense_gguf_gfx1100_strict_registered",
    "qwen36_moe_gguf_gfx1100_strict_registered",
    "register_qwen36_dense_gguf_gfx1100_profiles",
    "register_qwen36_moe_gguf_gfx1100_profiles",
]
