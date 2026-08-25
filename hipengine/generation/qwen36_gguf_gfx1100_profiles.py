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
QWEN36_DENSE_GGUF_BACKEND = "hip_gfx1100"
QWEN36_DENSE_GGUF_QUANT = "gguf_q4_k_m"
FP16_RECURRENT_STATE_ENV = "HIPENGINE_GGUF_FP16_RECURRENT_STATE"
VERIFY_CAPTURE_PREFILL_GDN_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"

_GDN_CHAIN_VARIANT = "bf16_c1_exact_state_rows_tloop"
_GDN_REGISTRY_QUANT = "gguf_qwen35"
_STRICT_EVIDENCE = (
    "benchmarks/results/2026-08-23-w7900-qwen36-27b-current-default-publication.json"
)


def _strict_binder(generator: Any, resolved: ResolvedRuntimeProfile) -> None:
    del generator, resolved
    os.environ[FP16_RECURRENT_STATE_ENV] = "0"
    os.environ[VERIFY_CAPTURE_PREFILL_GDN_ENV] = "1"


def _key(profile: ExecutionProfile) -> RuntimeProfileKey:
    return RuntimeProfileKey(
        model=QWEN36_DENSE_GGUF_MODEL,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile=profile,
    )


def register_qwen36_dense_gguf_gfx1100_profiles() -> bool:
    """Register the W7900 strict FP32-state dense NextN control once."""

    if _key(ExecutionProfile.STRICT) in registered_runtime_profile_keys():
        return False
    register_runtime_profile_plan(
        model=QWEN36_DENSE_GGUF_MODEL,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
        plan=RuntimeProfilePlan(
            selections=(
                VariantSelection(
                    layer="linear_attn_chain_conv_decode",
                    scope="specdec2_mtp2_c1",
                    selected_variant=_GDN_CHAIN_VARIANT,
                    strict_fallback_variant=_GDN_CHAIN_VARIANT,
                    registry_quant=_GDN_REGISTRY_QUANT,
                    evidence_artifact=_STRICT_EVIDENCE,
                ),
            ),
            kv_policy="paged_bf16",
            graph_policy="specdec2_eager_c1",
            binder=_strict_binder,
        ),
    )
    return True


def qwen36_dense_gguf_gfx1100_strict_registered() -> bool:
    return _key(ExecutionProfile.STRICT) in registered_runtime_profile_keys()


__all__ = [
    "FP16_RECURRENT_STATE_ENV",
    "QWEN36_DENSE_GGUF_BACKEND",
    "QWEN36_DENSE_GGUF_MODEL",
    "QWEN36_DENSE_GGUF_QUANT",
    "VERIFY_CAPTURE_PREFILL_GDN_ENV",
    "qwen36_dense_gguf_gfx1100_strict_registered",
    "register_qwen36_dense_gguf_gfx1100_profiles",
]
