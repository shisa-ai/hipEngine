"""Qwen3.8 dense GGUF gfx1151 strict/production profile plans.

The strict plan owns FP32 recurrent-state SPECDEC2 target journals.  The
production candidate selects the typed FP16-state siblings (FP32 accumulation,
FP16 state round-trip) and keeps every strict FP32 variant as an explicit
fallback.  Registration is cold-path only; P8 qualification still determines
whether any automatic SPECDEC2 policy cell may promote.
"""

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
PRODUCTION_Q4_VERIFIER_ROWTILE_ENV = (
    "HIPENGINE_GGUF_VERIFY_PRODUCTION_Q4_ROWTILE"
)

_GDN_REGISTRY_QUANT = "gguf_qwen35"
_Q4_REGISTRY_QUANT = "gguf_q4_k_t16_v1"
_STRICT_CHAIN_VARIANT = "bf16_c1_exact_state_rows_tloop"
_PRODUCTION_CHAIN_VARIANT = f"{_STRICT_CHAIN_VARIANT}_fp16state"
_FP16_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-20-gfx1151-qwen38-27b-r2-fp16-state-repaired-production.json"
)
_Q4_ROWTILE_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-28-gfx1151-qwen38-c2-production-q4-rowtile-retained.json"
)
_C3_R12_ROWTILE_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-28-gfx1151-qwen38-c3-production-r12-rowtiles-retained.json"
)


def _binder(
    generator: Any,
    resolved: ResolvedRuntimeProfile,
    *,
    fp16_state: bool,
    q4_verifier_rowtile: bool,
) -> None:
    del generator
    os.environ[FP16_RECURRENT_STATE_ENV] = "1" if fp16_state else "0"
    os.environ[VERIFY_CAPTURE_PREFILL_GDN_ENV] = "1"
    os.environ[PRODUCTION_Q4_VERIFIER_ROWTILE_ENV] = (
        "1" if q4_verifier_rowtile else "0"
    )
    os.environ["HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256"] = (
        resolved.manifest_sha256
    )


def _strict_binder(generator: Any, resolved: ResolvedRuntimeProfile) -> None:
    _binder(
        generator,
        resolved,
        fp16_state=False,
        q4_verifier_rowtile=False,
    )


def _production_binder(generator: Any, resolved: ResolvedRuntimeProfile) -> None:
    _binder(
        generator,
        resolved,
        fp16_state=True,
        q4_verifier_rowtile=True,
    )


def _key(profile: ExecutionProfile) -> RuntimeProfileKey:
    return RuntimeProfileKey(
        model=QWEN38_GGUF_MODEL,
        backend=QWEN38_GGUF_BACKEND,
        quant=QWEN38_GGUF_QUANT,
        profile=profile,
    )


def _selection(
    *,
    layer: str,
    scope: str,
    selected: str,
    fallback: str,
    quant: str,
    evidence: str | None = None,
) -> VariantSelection:
    return VariantSelection(
        layer=layer,
        scope=scope,
        selected_variant=selected,
        strict_fallback_variant=fallback,
        registry_quant=quant,
        evidence_artifact=evidence,
    )


def _strict_selections() -> tuple[VariantSelection, ...]:
    return (
        _selection(
            layer="linear",
            scope="specdec2_mtp2_c2_k3_r8_q4_single",
            selected="t16_wmma_prefill_smallm_bf16_bf16_out",
            fallback="t16_wmma_prefill_smallm_bf16_bf16_out",
            quant=_Q4_REGISTRY_QUANT,
        ),
        _selection(
            layer="linear_pair_silu",
            scope="specdec2_mtp2_c2_k3_r8_q4_gate_up",
            selected="dense_dual_wmma_prefill_bf16_bf16_out",
            fallback="dense_dual_wmma_prefill_bf16_bf16_out",
            quant=_Q4_REGISTRY_QUANT,
        ),
        _selection(
            layer="linear",
            scope="specdec2_mtp2_c3_k3_r12_q4_single",
            selected="t16_wmma_prefill_smallm_bf16_bf16_out",
            fallback="t16_wmma_prefill_smallm_bf16_bf16_out",
            quant=_Q4_REGISTRY_QUANT,
        ),
        _selection(
            layer="linear_pair_silu",
            scope="specdec2_mtp2_c3_k3_r12_q4_gate_up",
            selected="dense_dual_wmma_prefill_bf16_bf16_out",
            fallback="dense_dual_wmma_prefill_bf16_bf16_out",
            quant=_Q4_REGISTRY_QUANT,
        ),
        _selection(
            layer="linear_attn_chain_conv_decode",
            scope="specdec2_mtp2_c1",
            selected=_STRICT_CHAIN_VARIANT,
            fallback=_STRICT_CHAIN_VARIANT,
            quant=_GDN_REGISTRY_QUANT,
        ),
        _selection(
            layer="gdn_chain_recurrent_rmsnorm_gate",
            scope="specdec2_mtp2_target_state_rows",
            selected=_STRICT_CHAIN_VARIANT,
            fallback=_STRICT_CHAIN_VARIANT,
            quant=_GDN_REGISTRY_QUANT,
        ),
    )


def _production_selections() -> tuple[VariantSelection, ...]:
    return (
        _selection(
            layer="linear",
            scope="specdec2_mtp2_c2_k3_r8_q4_single",
            selected="dense_rowtile_bf16_bf16_out",
            fallback="t16_wmma_prefill_smallm_bf16_bf16_out",
            quant=_Q4_REGISTRY_QUANT,
            evidence=_Q4_ROWTILE_EVIDENCE,
        ),
        _selection(
            layer="linear_pair_silu",
            scope="specdec2_mtp2_c2_k3_r8_q4_gate_up",
            selected="dense_dual_rowtile_bf16_bf16_out",
            fallback="dense_dual_wmma_prefill_bf16_bf16_out",
            quant=_Q4_REGISTRY_QUANT,
            evidence=_Q4_ROWTILE_EVIDENCE,
        ),
        _selection(
            layer="linear",
            scope="specdec2_mtp2_c3_k3_r12_q4_single",
            selected="dense_rowtile_bf16_bf16_out",
            fallback="t16_wmma_prefill_smallm_bf16_bf16_out",
            quant=_Q4_REGISTRY_QUANT,
            evidence=_C3_R12_ROWTILE_EVIDENCE,
        ),
        _selection(
            layer="linear_pair_silu",
            scope="specdec2_mtp2_c3_k3_r12_q4_gate_up",
            selected="dense_dual_rowtile_bf16_bf16_out",
            fallback="dense_dual_wmma_prefill_bf16_bf16_out",
            quant=_Q4_REGISTRY_QUANT,
            evidence=_C3_R12_ROWTILE_EVIDENCE,
        ),
        _selection(
            layer="linear_attn_chain_conv_decode",
            scope="specdec2_mtp2_c1",
            selected=_STRICT_CHAIN_VARIANT,
            fallback=_STRICT_CHAIN_VARIANT,
            quant=_GDN_REGISTRY_QUANT,
            evidence=_FP16_EVIDENCE,
        ),
        _selection(
            layer="gdn_chain_recurrent_rmsnorm_gate",
            scope="specdec2_mtp2_target_state_rows",
            selected=_PRODUCTION_CHAIN_VARIANT,
            fallback=_STRICT_CHAIN_VARIANT,
            quant=_GDN_REGISTRY_QUANT,
            evidence=_FP16_EVIDENCE,
        ),
    )


def register_qwen38_gguf_gfx1151_profiles() -> bool:
    """Register strict FP32 and production-candidate FP16 plans once."""

    wanted = {_key(ExecutionProfile.STRICT), _key(ExecutionProfile.PRODUCTION)}
    existing = set(registered_runtime_profile_keys())
    if wanted <= existing:
        return False
    if existing & wanted:
        raise RuntimeError("Qwen3.8 gfx1151 profile registry is partially populated")
    register_runtime_profile_plan(
        model=QWEN38_GGUF_MODEL,
        backend=QWEN38_GGUF_BACKEND,
        quant=QWEN38_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
        plan=RuntimeProfilePlan(
            selections=_strict_selections(),
            kv_policy="paged_bf16",
            graph_policy="specdec2_eager_c1",
            binder=_strict_binder,
        ),
    )
    register_runtime_profile_plan(
        model=QWEN38_GGUF_MODEL,
        backend=QWEN38_GGUF_BACKEND,
        quant=QWEN38_GGUF_QUANT,
        profile=ExecutionProfile.PRODUCTION,
        plan=RuntimeProfilePlan(
            selections=_production_selections(),
            kv_policy="paged_bf16",
            graph_policy="specdec2_eager_c1",
            binder=_production_binder,
        ),
    )
    return True


def qwen38_gguf_gfx1151_plans_registered() -> bool:
    wanted = {_key(ExecutionProfile.STRICT), _key(ExecutionProfile.PRODUCTION)}
    return wanted <= set(registered_runtime_profile_keys())


def qwen38_gguf_gfx1151_strict_registered() -> bool:
    """Compatibility alias for older strict-only callers."""

    return _key(ExecutionProfile.STRICT) in registered_runtime_profile_keys()


__all__ = [
    "FP16_RECURRENT_STATE_ENV",
    "PRODUCTION_Q4_VERIFIER_ROWTILE_ENV",
    "QWEN38_GGUF_BACKEND",
    "QWEN38_GGUF_MODEL",
    "QWEN38_GGUF_QUANT",
    "VERIFY_CAPTURE_PREFILL_GDN_ENV",
    "qwen38_gguf_gfx1151_plans_registered",
    "qwen38_gguf_gfx1151_strict_registered",
    "register_qwen38_gguf_gfx1151_profiles",
]
