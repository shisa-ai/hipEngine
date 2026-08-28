"""Qwen4Exp gfx1151 strict and late-layer production prefill plans."""

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

QWEN4_EXP_MODEL = "qwen4_exp_gguf"
QWEN4_EXP_BACKEND = "hip_gfx1151"
QWEN4_EXP_QUANTS = ("gguf_q4_k_m", "gguf_ud_q4_k_xl")
PRODUCTION_MOE_PREFILL_ENV = "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL"
PRODUCTION_Q8_PREFILL_LAYERS = tuple(range(32, 48))
_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-29-gfx1151-qwen38-flash-next-moe27-q8-32-production.json"
)


def _selection(
    layer: str,
    scope: str,
    selected: str,
    fallback: str,
    quant: str,
    *,
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
            "moe_linear",
            "prefill_rows_ge16_layers27_47_gate_up",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "gguf_q4_k",
        ),
        _selection(
            "moe_linear",
            "prefill_rows_ge16_layers27_47_down",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
        ),
        _selection(
            "linear",
            "prefill_rows_ge16_layers32_47_q8",
            "coltile8_rowbatch4_f32_f32_out",
            "coltile8_rowbatch4_f32_f32_out",
            "gguf_q8_0",
        ),
    )


def _production_selections() -> tuple[VariantSelection, ...]:
    return (
        _selection(
            "moe_linear",
            "prefill_rows_ge16_layers27_47_gate_up",
            "selected_dual_wmma_prefill_compact_bf16_bf16_out",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "gguf_q4_k",
            evidence=_EVIDENCE,
        ),
        _selection(
            "moe_linear",
            "prefill_rows_ge16_layers27_47_down",
            "selected_grouped_wmma_prefill_compact_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
            evidence=_EVIDENCE,
        ),
        _selection(
            "linear",
            "prefill_rows_ge16_layers32_47_q8",
            "wmma_prefill_f32_f32_out",
            "coltile8_rowbatch4_f32_f32_out",
            "gguf_q8_0",
            evidence=_EVIDENCE,
        ),
    )


def _bind(generator: Any, resolved: ResolvedRuntimeProfile, *, production: bool) -> None:
    del generator
    os.environ[PRODUCTION_MOE_PREFILL_ENV] = "1" if production else "0"
    # Freeze all neighboring experiments so the manifest selects only the
    # certified late-layer cooperative MoE arithmetic.
    for key, value in {
        "HIPENGINE_GGUF_WMMA_PREFILL": "0",
        "HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL": "0",
        "HIPENGINE_QWEN4_EXP_Q5_1_WMMA": "0",
        "HIPENGINE_QWEN4_EXP_Q8_0_GROUPED_WMMA": "0",
        "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS": (
            ",".join(map(str, PRODUCTION_Q8_PREFILL_LAYERS))
            if production
            else ""
        ),
        "HIPENGINE_GGUF_Q8_0_WMMA_TILE_M": "64",
        "HIPENGINE_GGUF_Q8_0_WMMA_TILE_N": "32",
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN": "1",
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4": "1",
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4_ALL": "1",
        "HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256": resolved.manifest_sha256,
    }.items():
        os.environ[key] = value


def _strict_binder(generator: Any, resolved: ResolvedRuntimeProfile) -> None:
    _bind(generator, resolved, production=False)


def _production_binder(generator: Any, resolved: ResolvedRuntimeProfile) -> None:
    _bind(generator, resolved, production=True)


def _key(quant: str, profile: ExecutionProfile) -> RuntimeProfileKey:
    return RuntimeProfileKey(QWEN4_EXP_MODEL, QWEN4_EXP_BACKEND, quant, profile)


def register_qwen4_exp_gfx1151_profiles() -> bool:
    wanted = {
        _key(quant, profile)
        for quant in QWEN4_EXP_QUANTS
        for profile in (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION)
    }
    existing = set(registered_runtime_profile_keys())
    if wanted <= existing:
        return False
    if existing & wanted:
        raise RuntimeError("Qwen4Exp gfx1151 profile registry is partially populated")
    for quant in QWEN4_EXP_QUANTS:
        register_runtime_profile_plan(
            model=QWEN4_EXP_MODEL,
            backend=QWEN4_EXP_BACKEND,
            quant=quant,
            profile=ExecutionProfile.STRICT,
            plan=RuntimeProfilePlan(
                selections=_strict_selections(),
                kv_policy="paged_bf16_qsa_index_f32",
                graph_policy="request_owned_exact_moe_graph_c1",
                binder=_strict_binder,
            ),
        )
        register_runtime_profile_plan(
            model=QWEN4_EXP_MODEL,
            backend=QWEN4_EXP_BACKEND,
            quant=quant,
            profile=ExecutionProfile.PRODUCTION,
            plan=RuntimeProfilePlan(
                selections=_production_selections(),
                kv_policy="paged_bf16_qsa_index_f32",
                graph_policy="request_owned_exact_moe_graph_c1",
                binder=_production_binder,
            ),
        )
    return True


def qwen4_exp_gfx1151_profiles_registered() -> bool:
    wanted = {
        _key(quant, profile)
        for quant in QWEN4_EXP_QUANTS
        for profile in (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION)
    }
    return wanted <= set(registered_runtime_profile_keys())


__all__ = [
    "PRODUCTION_MOE_PREFILL_ENV",
    "QWEN4_EXP_BACKEND",
    "QWEN4_EXP_MODEL",
    "QWEN4_EXP_QUANTS",
    "PRODUCTION_Q8_PREFILL_LAYERS",
    "qwen4_exp_gfx1151_profiles_registered",
    "register_qwen4_exp_gfx1151_profiles",
]
