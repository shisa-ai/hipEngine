"""Qwen4Exp gfx1151 strict and certified production prefill/decode plans."""

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
PRODUCTION_GDN_PEER_PREFILL_LAYERS = tuple(range(35, 48))
PRODUCTION_GDN_COLWARPS_PREFILL_LAYERS = tuple(range(27, 48))
PRODUCTION_QSA_FLASH_PREFILL_LAYERS = tuple(range(35, 48))
PRODUCTION_WMMA_MOE_PREFILL_LAYERS = tuple(range(27, 48))
PRODUCTION_Q4_IU8_PREFILL_LAYERS = tuple(range(35, 48))
PRODUCTION_Q5_1_MMQ_PREFILL_LAYERS = tuple(range(32, 48))
PRODUCTION_Q4_K_MMQ_PREFILL_LAYERS = tuple(range(35, 48))
PRODUCTION_Q4_DP4A_DECODE_LAYERS = (
    0, 2, 5, 6, 8, 9, 10, 11
) + tuple(range(13, 48))
_MOE_WMMA_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-29-gfx1151-qwen38-flash-next-wmma-moe27-production.json"
)
_STACK_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-29-gfx1151-qwen38-flash-next-production-mmq-prefill-dp4a43-stack.json"
)
_DECODE_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-29-gfx1151-qwen38-flash-next-production-dp4a-safe43-decode.json"
)
_GDN_PEER_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-29-gfx1151-qwen38-flash-next-production-gdn-peer35.json"
)
_GDN_COLWARPS_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps27-production.json"
)
_GR_SIGMOID_MEAN_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-31-gfx1151-qwen38-flash-next-p3-gr-sigmoid-mean.json"
)
_ROUTER_F32_TILE4_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-31-gfx1151-qwen38-flash-next-p3-router-f32-tile4.json"
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
            "router_logits",
            "qwen4exp_multirow_f32_router",
            "f32_hidden_token_tile4_dense_exact",
            "f32_hidden",
            "f32",
            evidence=_ROUTER_F32_TILE4_EVIDENCE,
        ),
        _selection(
            "gr_gated_mean_sigmoid",
            "all_gr_reads_rows_le256",
            "strict",
            "strict_unfused",
            "f32",
            evidence=_GR_SIGMOID_MEAN_EVIDENCE,
        ),
        _selection(
            "moe_linear",
            "prefill_rows_ge2_layers27_47_gate_up",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "gguf_q4_k",
        ),
        _selection(
            "moe_linear",
            "prefill_rows_ge2_layers27_47_down",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
        ),
        _selection(
            "linear",
            "prefill_policy_qwen4exp_dense_q8_shapes",
            "coltile8_rowbatch4_f32_f32_out",
            "coltile8_rowbatch4_f32_f32_out",
            "gguf_q8_0",
        ),
        _selection(
            "gdn_recurrence_norm_gate",
            "prefill_rows_ge2_layers27_47_gdn",
            "qwen4exp_sigmoid_strict_prefill",
            "qwen4exp_sigmoid_strict_prefill",
            "f32_state",
        ),
        _selection(
            "linear",
            "decode_c1_calibrated_q4_dp4a_43_layers",
            "selected_dual_silu_logical128_t64_gemv_bf16_bf16_out",
            "selected_dual_silu_logical128_t64_gemv_bf16_bf16_out",
            "gguf_q4_k",
        ),
    )


def _production_selections() -> tuple[VariantSelection, ...]:
    return (
        _selection(
            "router_logits",
            "qwen4exp_multirow_f32_router",
            "f32_hidden_token_tile4_dense_exact",
            "f32_hidden_token_tile4_dense_exact",
            "f32",
            evidence=_ROUTER_F32_TILE4_EVIDENCE,
        ),
        _selection(
            "gr_gated_mean_sigmoid",
            "all_gr_reads_rows_le256",
            "strict",
            "strict",
            "f32",
            evidence=_GR_SIGMOID_MEAN_EVIDENCE,
        ),
        _selection(
            "moe_linear",
            "prefill_rows_ge2_layers27_47_gate_up",
            "selected_dual_wmma_prefill_compact_bf16_bf16_out",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "gguf_q4_k",
            evidence=_MOE_WMMA_EVIDENCE,
        ),
        _selection(
            "moe_linear",
            "prefill_rows_ge2_layers27_47_down",
            "selected_grouped_wmma_prefill_compact_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
            evidence=_MOE_WMMA_EVIDENCE,
        ),
        _selection(
            "linear",
            "prefill_policy_qwen4exp_dense_q8_shapes",
            "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out",
            "coltile8_rowbatch4_f32_f32_out",
            "gguf_q8_0",
            evidence=_STACK_EVIDENCE,
        ),
        _selection(
            "gdn_recurrence_norm_gate",
            "prefill_rows_ge2_layers27_47_gdn",
            "qwen4exp_gdn_columnwarps_prefill",
            "qwen4exp_sigmoid_strict_prefill",
            "f32_state",
            evidence=_GDN_COLWARPS_EVIDENCE,
        ),
        _selection(
            "linear",
            "decode_c1_calibrated_q4_dp4a_43_layers",
            "selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out",
            "selected_dual_silu_logical128_t64_gemv_bf16_bf16_out",
            "gguf_q4_k",
            evidence=_DECODE_EVIDENCE,
        ),
    )


def _bind(generator: Any, resolved: ResolvedRuntimeProfile, *, production: bool) -> None:
    os.environ[PRODUCTION_MOE_PREFILL_ENV] = "1" if production else "0"
    # Freeze neighboring experiments and select only the complete certified
    # WMMA-MoE27 prefill + Q8-MMQ dense + DP4A-decode + peer-GDN composition.
    # The WMMA route covers MoE layers 27-47 via the backend capability
    # constant; the ds4-MMQ envs stay off so they cannot preempt it, and the
    # exact-grouped guards (`not production_grouped_moe`) keep layers 0-26 on
    # the strict owners.
    for key, value in {
        "HIPENGINE_GGUF_WMMA_PREFILL": "0",
        "HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL": "0",
        "HIPENGINE_QWEN4_EXP_Q5_1_WMMA": "0",
        "HIPENGINE_QWEN4_EXP_Q8_0_GROUPED_WMMA": "0",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE": "0",
        "HIPENGINE_QWEN4_EXP_SHARED_DOWN_COMBINE": "0",
        # ds4-MMQ MoE suffixes are superseded by the certified WMMA-MoE27
        # routing on layers 27-47.
        "HIPENGINE_QWEN4_EXP_Q5_1_MMQ_PREFILL": "0",
        "HIPENGINE_QWEN4_EXP_Q5_1_MMQ_LAYERS": "",
        "HIPENGINE_QWEN4_EXP_Q4_K_MMQ_PREFILL": "0",
        "HIPENGINE_QWEN4_EXP_Q4_K_MMQ_LAYERS": "",
        "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL_LAYERS": (
            ",".join(map(str, PRODUCTION_GDN_PEER_PREFILL_LAYERS))
            if production
            else ""
        ),
        "HIPENGINE_QWEN4_EXP_Q4_DP4A64": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS": (
            ",".join(map(str, PRODUCTION_Q4_DP4A_DECODE_LAYERS))
            if production
            else ""
        ),
        "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS": "",
        "HIPENGINE_GGUF_Q8_0_WMMA_TILE_M": "64",
        "HIPENGINE_GGUF_Q8_0_WMMA_TILE_N": "32",
        "HIPENGINE_QWEN4_EXP_Q4_TILE_M": "16",
        "HIPENGINE_QWEN4_EXP_Q4_TILE_N": "16",
        # Certified iu8-WMMA gate/up suffix within the WMMA-MoE27 route.
        "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS": (
            ",".join(map(str, PRODUCTION_Q4_IU8_PREFILL_LAYERS))
            if production
            else ""
        ),
        # Certified column-warp GDN prefill suffix (supersedes peer-GDN).
        # Decode retains its separately tuned rows==1 owner.
        "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS": (
            ",".join(map(str, PRODUCTION_GDN_COLWARPS_PREFILL_LAYERS))
            if production
            else ""
        ),
        "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS": "",
        # Certified QSA flash prefill suffix.
        "HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS": (
            ",".join(map(str, PRODUCTION_QSA_FLASH_PREFILL_LAYERS))
            if production
            else ""
        ),
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN": "1",
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4": "1",
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4_ALL": "1",
        "HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256": resolved.manifest_sha256,
    }.items():
        os.environ[key] = value
    if production:
        configure = getattr(
            getattr(generator, "runner", None),
            "configure_mmq_prefill_resources",
            None,
        )
        if callable(configure):
            configure()


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
    "PRODUCTION_GDN_PEER_PREFILL_LAYERS",
    "PRODUCTION_MOE_PREFILL_ENV",
    "PRODUCTION_Q4_DP4A_DECODE_LAYERS",
    "PRODUCTION_Q4_K_MMQ_PREFILL_LAYERS",
    "PRODUCTION_Q5_1_MMQ_PREFILL_LAYERS",
    "QWEN4_EXP_BACKEND",
    "QWEN4_EXP_MODEL",
    "QWEN4_EXP_QUANTS",
    "qwen4_exp_gfx1151_profiles_registered",
    "register_qwen4_exp_gfx1151_profiles",
]
