"""Qwen3.8 dense GGUF gfx1151 strict execution-profile plan."""

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

QWEN38_GGUF_MODEL = "qwen3_5_gguf"
QWEN38_GGUF_BACKEND = "hip_gfx1151"
QWEN38_GGUF_QUANT = "gguf_q4_k_m"
FP16_RECURRENT_STATE_ENV = "HIPENGINE_GGUF_FP16_RECURRENT_STATE"
VERIFY_CAPTURE_PREFILL_GDN_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"

_GDN_CHAIN_VARIANT = "bf16_c1_exact_state_rows_tloop"
_GDN_REGISTRY_QUANT = "gguf_qwen35"


def _strict_binder(
    generator: Any,
    resolved: ResolvedRuntimeProfile,
) -> None:
    del generator, resolved
    os.environ[FP16_RECURRENT_STATE_ENV] = "0"
    os.environ[VERIFY_CAPTURE_PREFILL_GDN_ENV] = "1"


def _key(profile: ExecutionProfile) -> RuntimeProfileKey:
    return RuntimeProfileKey(
        model=QWEN38_GGUF_MODEL,
        backend=QWEN38_GGUF_BACKEND,
        quant=QWEN38_GGUF_QUANT,
        profile=profile,
    )


def register_qwen38_gguf_gfx1151_profiles() -> bool:
    """Register the strict FP32-state MTP2 control once."""

    if _key(ExecutionProfile.STRICT) in registered_runtime_profile_keys():
        return False
    register_runtime_profile_plan(
        model=QWEN38_GGUF_MODEL,
        backend=QWEN38_GGUF_BACKEND,
        quant=QWEN38_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
        plan=RuntimeProfilePlan(
            selections=(
                VariantSelection(
                    layer="linear_attn_chain_conv_decode",
                    scope="specdec2_mtp2_c1",
                    selected_variant=_GDN_CHAIN_VARIANT,
                    strict_fallback_variant=_GDN_CHAIN_VARIANT,
                    registry_quant=_GDN_REGISTRY_QUANT,
                    evidence_artifact=(
                        "benchmarks/results/"
                        "2026-08-25-gfx1151-specdec2-s3-c1.json"
                    ),
                ),
            ),
            kv_policy="paged_bf16",
            graph_policy="specdec2_eager_c1",
            binder=_strict_binder,
        ),
    )
    return True


def qwen38_gguf_gfx1151_strict_registered() -> bool:
    return _key(ExecutionProfile.STRICT) in registered_runtime_profile_keys()


__all__ = [
    "FP16_RECURRENT_STATE_ENV",
    "QWEN38_GGUF_BACKEND",
    "QWEN38_GGUF_MODEL",
    "QWEN38_GGUF_QUANT",
    "VERIFY_CAPTURE_PREFILL_GDN_ENV",
    "qwen38_gguf_gfx1151_strict_registered",
    "register_qwen38_gguf_gfx1151_profiles",
]
