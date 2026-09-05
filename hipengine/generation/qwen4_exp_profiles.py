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
PROFILE_Q5_1_DOWN_M1_ENV = "HIPENGINE_QWEN4_EXP_PROFILE_Q5_1_DOWN_M1"
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
_GR_UP_SIGMOID_MEAN_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-31-gfx1151-qwen38-flash-next-p3-gr-up-sigmoid-mean.json"
)
_QSA_DENSE_FIXED256_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-31-gfx1151-qwen38-flash-next-p4-qsa-dense-fixed256.json"
)
_DEVICE_ARGMAX_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-31-gfx1151-qwen38-flash-next-p5-device-argmax.json"
)
_QSA_ORDERED_DECODE_EVIDENCE = (
    "benchmarks/results/"
    "2026-09-02-gfx1151-qwen38-flash-next-p6-qsa-ordered-decode.json"
)
_HALO_PF13_EVIDENCE = (
    "benchmarks/results/"
    "2026-09-04-gfx1151-qwen38-flash-next-halo-pf13-production-refresh.json"
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
            "gdn_recurrence_norm_gate", "prefill_rows_ge2_serial_gdn_hk16_hv48_d128",
            "qwen4exp_sigmoid_strict_prefill", "qwen4exp_sigmoid_strict_prefill",
            "f32_state",
        ),
        _selection(
            "moe_linear", "prefill_rows_ge64_exact_grouped_q5_1_down",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
        ),
        _selection(
            "moe_linear", "prefill_rows_ge2_exact_grouped_q4_gate_up",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "gguf_q4_k",
        ),
        _selection(
            "qsa_sparse_attention", "prefill_h256_page256_sparse_rows_ge16",
            "strict_rows_spans", "strict_rows_spans", "bf16_kv",
        ),
        _selection(
            "linear",
            "ungrouped_prefill_rows_ge64_q5k_gate_up",
            "selected_gemv_bf16_bf16_out",
            "selected_gemv_bf16_bf16_out",
            "gguf_q5_k",
        ),
        _selection(
            "argmax",
            "qwen4exp_normal_greedy_output",
            "top1_i64",
            "top1_i64",
            "f32",
            evidence=_DEVICE_ARGMAX_EVIDENCE,
        ),
        _selection(
            "paged_attn_decode",
            "qwen4exp_multirow_dense_qsa",
            "bf16_context_batch_paged_c1_exact_spans",
            "bf16_context_batch_spans",
            "w4_paro",
            evidence=_QSA_DENSE_FIXED256_EVIDENCE,
        ),
        _selection(
            "qsa_sparse_attention",
            "qwen4exp_c1_h256_indexed_sparse_decode",
            "strict_spans",
            "strict_spans",
            "bf16_kv",
        ),
        _selection(
            "linear+gr_gated_mean",
            "qwen4exp_rows_gt256_gr_up",
            "coltile2_branch4_rowbatch4_f32_exact",
            "coltile8_rowbatch4_f32_f32_out",
            "gguf_q8_0",
            evidence=_GR_UP_SIGMOID_MEAN_EVIDENCE,
        ),
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
            "moe_linear",
            "prefill_rows_ge2_lt64_exact_grouped_q5_1_down",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
        ),
        _selection(
            "linear",
            "grouped_prefill_q8_0_expert_down",
            "selected_gemv_bf16_bf16_out",
            "selected_gemv_bf16_bf16_out",
            "gguf_q8_0",
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
            "gdn_recurrence_norm_gate", "prefill_rows_ge2_serial_gdn_hk16_hv48_d128",
            "qwen4exp_sigmoid_register_prefill", "qwen4exp_sigmoid_strict_prefill",
            "f32_state",
            evidence="benchmarks/results/2026-09-05-framework-qwen4exp-gdn-register-production.json",
        ),
        _selection(
            "moe_linear", "prefill_rows_ge64_exact_grouped_q5_1_down",
            "selected_grouped_prefill_pair2_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
            evidence="benchmarks/results/2026-09-05-framework-qwen4exp-q51-pair-production.json",
        ),
        _selection(
            "moe_linear", "prefill_rows_ge2_exact_grouped_q4_gate_up",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "gguf_q4_k",
            evidence="benchmarks/results/2026-09-05-framework-qwen4exp-prefill-promotion.json",
        ),
        _selection(
            "qsa_sparse_attention", "prefill_h256_page256_sparse_rows_ge16",
            "strict_h256_page256_wave_rows_spans", "strict_rows_spans", "bf16_kv",
            evidence="benchmarks/results/2026-09-05-framework-qwen4exp-prefill-promotion.json",
        ),
        _selection(
            "linear",
            "ungrouped_prefill_rows_ge64_q5k_gate_up",
            "selected_grouped_row4_gemv_bf16_bf16_out",
            "selected_gemv_bf16_bf16_out",
            "gguf_q5_k",
            evidence="benchmarks/results/2026-09-05-framework-qwen4exp-row4-production.json",
        ),
        _selection(
            "argmax",
            "qwen4exp_normal_greedy_output",
            "top1_i64",
            "top1_i64",
            "f32",
            evidence=_DEVICE_ARGMAX_EVIDENCE,
        ),
        _selection(
            "paged_attn_decode",
            "qwen4exp_multirow_dense_qsa",
            "bf16_context_batch_paged_c1_exact_spans",
            "bf16_context_batch_paged_c1_exact_spans",
            "w4_paro",
            evidence=_QSA_DENSE_FIXED256_EVIDENCE,
        ),
        _selection(
            "qsa_sparse_attention",
            "qwen4exp_c1_h256_indexed_sparse_decode",
            "strict_ordered_three_pass_spans",
            "strict_spans",
            "bf16_kv",
            evidence=_QSA_ORDERED_DECODE_EVIDENCE,
        ),
        _selection(
            "linear+gr_gated_mean",
            "qwen4exp_rows_gt256_gr_up",
            "coltile2_branch4_rowbatch4_f32_exact",
            "coltile2_branch4_rowbatch4_f32_exact",
            "gguf_q8_0",
            evidence=_GR_UP_SIGMOID_MEAN_EVIDENCE,
        ),
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
            "moe_linear",
            "prefill_rows_ge2_lt64_exact_grouped_q5_1_down",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
            evidence=_HALO_PF13_EVIDENCE,
        ),
        _selection(
            "linear",
            "grouped_prefill_q8_0_expert_down",
            "selected_grouped_gemv_bf16_bf16_out",
            "selected_gemv_bf16_bf16_out",
            "gguf_q8_0",
            evidence=_HALO_PF13_EVIDENCE,
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
    # their separately manifested exact grouped owners.
    for key, value in {
        "HIPENGINE_GGUF_WMMA_PREFILL": "0",
        "HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL": "0",
        "HIPENGINE_QWEN4_EXP_Q5_1_WMMA": "0",
        "HIPENGINE_QWEN4_EXP_Q8_0_GROUPED_WMMA": "0",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE": "0",
        # PF-3 Q5_1 M1 and PF-1 grouped Q8_0 down are T0 exact production
        # owners after the one-process/one-residency canonical gate. Strict
        # keeps the preceding owners as registered fallbacks.
        PROFILE_Q5_1_DOWN_M1_ENV: "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL": "page256" if production else "0",
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
        "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE": "1" if production else "0",
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
    "PROFILE_Q5_1_DOWN_M1_ENV",
    "PRODUCTION_Q4_DP4A_DECODE_LAYERS",
    "PRODUCTION_Q4_K_MMQ_PREFILL_LAYERS",
    "PRODUCTION_Q5_1_MMQ_PREFILL_LAYERS",
    "QWEN4_EXP_BACKEND",
    "QWEN4_EXP_MODEL",
    "QWEN4_EXP_QUANTS",
    "qwen4_exp_gfx1151_profiles_registered",
    "register_qwen4_exp_gfx1151_profiles",
]
