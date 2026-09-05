from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile,
    clear_runtime_profile_registry_for_tests,
    resolve_runtime_profile,
)
from hipengine.generation.qwen4_exp_profiles import (
    PRODUCTION_GDN_COLWARPS_PREFILL_LAYERS,
    PRODUCTION_QSA_FLASH_PREFILL_LAYERS,
    PRODUCTION_GDN_PEER_PREFILL_LAYERS,
    PRODUCTION_MOE_PREFILL_ENV,
    PROFILE_Q5_1_DOWN_M1_ENV,
    PRODUCTION_Q4_DP4A_DECODE_LAYERS,
    PRODUCTION_Q4_IU8_PREFILL_LAYERS,
    PRODUCTION_Q4_K_MMQ_PREFILL_LAYERS,
    PRODUCTION_Q5_1_MMQ_PREFILL_LAYERS,
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    QWEN4_EXP_QUANTS,
    qwen4_exp_gfx1151_profiles_registered,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime.qwen4_exp_runner import (
    _qwen4_exp_production_moe_prefill_enabled,
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    clear_runtime_profile_registry_for_tests()
    environment_names = (
        PRODUCTION_MOE_PREFILL_ENV,
        PROFILE_Q5_1_DOWN_M1_ENV,
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE",
        "HIPENGINE_QWEN4_EXP_Q5_1_MMQ_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q5_1_MMQ_LAYERS",
        "HIPENGINE_QWEN4_EXP_Q4_K_MMQ_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q4_K_MMQ_LAYERS",
        "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL",
        "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL_LAYERS",
        "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS",
        "HIPENGINE_QWEN4_EXP_Q4_DP4A64",
        "HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS",
        "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS",
        "HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN",
        "HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL",
        "HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256",
    )
    for name in environment_names:
        monkeypatch.delenv(name, raising=False)
    yield
    for name in environment_names:
        os.environ.pop(name, None)
    clear_runtime_profile_registry_for_tests()


def _resolve(profile: ExecutionProfile, *, quant: str = QWEN4_EXP_QUANTS[1]):
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=quant,
        profile=profile,
    )


def _selection_map(resolved):
    return {
        (row["layer"], row["scope"]): row
        for row in resolved.manifest["selections"]
    }


def test_qwen4_exp_strict_and_production_manifests_resolve() -> None:
    register_gfx1151_kernels(replace=True)
    assert register_qwen4_exp_gfx1151_profiles()
    assert qwen4_exp_gfx1151_profiles_registered()

    strict = _resolve(ExecutionProfile.STRICT)
    production = _resolve(ExecutionProfile.PRODUCTION)
    strict_argmax = _selection_map(strict)[
        ("argmax", "qwen4exp_normal_greedy_output")
    ]
    assert strict_argmax["selected_variant"] == "top1_i64"
    assert strict_argmax["strict_fallback_variant"] == "top1_i64"
    strict_qsa = _selection_map(strict)[
        ("paged_attn_decode", "qwen4exp_multirow_dense_qsa")
    ]
    assert strict_qsa["selected_variant"] == "bf16_context_batch_paged_c1_exact_spans"
    assert strict_qsa["strict_fallback_variant"] == "bf16_context_batch_spans"
    strict_gr_up = _selection_map(strict)[
        ("linear+gr_gated_mean", "qwen4exp_rows_gt256_gr_up")
    ]
    assert strict_gr_up["selected_variant"] == "coltile2_branch4_rowbatch4_f32_exact"
    assert strict_gr_up["strict_fallback_variant"] == "coltile8_rowbatch4_f32_f32_out"
    strict_router = _selection_map(strict)[
        ("router_logits", "qwen4exp_multirow_f32_router")
    ]
    assert strict_router["selected_variant"] == "f32_hidden_token_tile4_dense_exact"
    assert strict_router["strict_fallback_variant"] == "f32_hidden"
    strict_gr = _selection_map(strict)[
        ("gr_gated_mean_sigmoid", "all_gr_reads_rows_le256")
    ]
    assert strict_gr["selected_variant"] == "strict"
    assert strict_gr["strict_fallback_variant"] == "strict_unfused"
    strict_q5_m1 = _selection_map(strict)[
        ("moe_linear", "prefill_rows_ge2_exact_grouped_q5_1_down")
    ]
    assert strict_q5_m1["selected_variant"].endswith(
        "expertgrid64_bf16_bf16_out"
    )
    assert strict_q5_m1["strict_fallback_variant"].endswith(
        "expertgrid64_bf16_bf16_out"
    )
    strict_q8_grouped = _selection_map(strict)[
        ("linear", "grouped_prefill_q8_0_expert_down")
    ]
    assert strict_q8_grouped["selected_variant"] == (
        "selected_gemv_bf16_bf16_out"
    )
    assert strict_q8_grouped["strict_fallback_variant"] == (
        "selected_gemv_bf16_bf16_out"
    )
    assert strict.manifest_sha256 == strict.strict_manifest_sha256
    assert production.manifest_sha256 != production.strict_manifest_sha256
    assert not production.fell_back_to_strict
    assert production.manifest["kv_policy"] == "paged_bf16_qsa_index_f32"
    assert production.manifest["graph_policy"] == "request_owned_exact_moe_graph_c1"
    selections = _selection_map(production)
    argmax = selections[("argmax", "qwen4exp_normal_greedy_output")]
    assert argmax["selected_variant"] == "top1_i64"
    assert argmax["strict_fallback_variant"] == "top1_i64"
    assert argmax["evidence_artifact"].endswith("p5-device-argmax.json")
    qsa = selections[("paged_attn_decode", "qwen4exp_multirow_dense_qsa")]
    assert qsa["selected_variant"] == "bf16_context_batch_paged_c1_exact_spans"
    assert qsa["strict_fallback_variant"] == "bf16_context_batch_paged_c1_exact_spans"
    assert qsa["evidence_artifact"].endswith("p4-qsa-dense-fixed256.json")
    gr_up = selections[("linear+gr_gated_mean", "qwen4exp_rows_gt256_gr_up")]
    assert gr_up["selected_variant"] == "coltile2_branch4_rowbatch4_f32_exact"
    assert gr_up["strict_fallback_variant"] == "coltile2_branch4_rowbatch4_f32_exact"
    assert gr_up["evidence_artifact"].endswith("p3-gr-up-sigmoid-mean.json")
    router = selections[("router_logits", "qwen4exp_multirow_f32_router")]
    assert router["selected_variant"] == "f32_hidden_token_tile4_dense_exact"
    assert router["strict_fallback_variant"] == "f32_hidden_token_tile4_dense_exact"
    assert router["evidence_artifact"].endswith("p3-router-f32-tile4.json")
    gr = selections[("gr_gated_mean_sigmoid", "all_gr_reads_rows_le256")]
    assert gr["selected_variant"] == "strict"
    assert gr["strict_fallback_variant"] == "strict"
    assert gr["evidence_artifact"].endswith("p3-gr-sigmoid-mean.json")
    gate = selections[("moe_linear", "prefill_rows_ge2_layers27_47_gate_up")]
    assert gate["selected_variant"] == (
        "selected_dual_wmma_prefill_compact_bf16_bf16_out"
    )
    assert gate["strict_fallback_variant"].startswith(
        "selected_dual_grouped_rowbatch8"
    )
    down = selections[("moe_linear", "prefill_rows_ge2_layers27_47_down")]
    assert down["selected_variant"] == (
        "selected_grouped_wmma_prefill_compact_bf16_bf16_out"
    )
    assert down["evidence_artifact"].endswith(
        "wmma-moe27-production.json"
    )
    q5_m1 = selections[
        ("moe_linear", "prefill_rows_ge2_exact_grouped_q5_1_down")
    ]
    assert q5_m1["selected_variant"].endswith(
        "expertgrid64_m1_bf16_bf16_out"
    )
    assert q5_m1["strict_fallback_variant"].endswith(
        "expertgrid64_bf16_bf16_out"
    )
    assert q5_m1["evidence_artifact"].endswith(
        "halo-pf13-production-refresh.json"
    )
    q8_grouped = selections[("linear", "grouped_prefill_q8_0_expert_down")]
    assert q8_grouped["selected_variant"] == (
        "selected_grouped_gemv_bf16_bf16_out"
    )
    assert q8_grouped["strict_fallback_variant"] == (
        "selected_gemv_bf16_bf16_out"
    )
    assert q8_grouped["evidence_artifact"].endswith(
        "halo-pf13-production-refresh.json"
    )
    q8 = selections[("linear", "prefill_policy_qwen4exp_dense_q8_shapes")]
    assert q8["selected_variant"] == (
        "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out"
    )
    assert q8["strict_fallback_variant"] == "coltile8_rowbatch4_f32_f32_out"
    gdn = selections[
        ("gdn_recurrence_norm_gate", "prefill_rows_ge2_layers27_47_gdn")
    ]
    assert gdn["selected_variant"] == "qwen4exp_gdn_columnwarps_prefill"
    assert gdn["strict_fallback_variant"] == "qwen4exp_sigmoid_strict_prefill"
    assert gdn["evidence_artifact"].endswith("gdn-colwarps27-production.json")
    dp4a = selections[("linear", "decode_c1_calibrated_q4_dp4a_43_layers")]
    assert dp4a["selected_variant"] == (
        "selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out"
    )
    assert dp4a["strict_fallback_variant"] == (
        "selected_dual_silu_logical128_t64_gemv_bf16_bf16_out"
    )
    assert dp4a["evidence_artifact"].endswith(
        "production-dp4a-safe43-decode.json"
    )


def test_qwen4_exp_profile_binders_select_only_certified_late_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    production = _resolve(ExecutionProfile.PRODUCTION)
    assert production.binder is not None
    configure_calls: list[bool] = []
    fake_runner = SimpleNamespace(
        configure_mmq_prefill_resources=lambda: configure_calls.append(True)
    )
    production.binder(SimpleNamespace(runner=fake_runner), production)
    assert configure_calls == [True]
    assert os.environ[PRODUCTION_MOE_PREFILL_ENV] == "1"
    assert os.environ["HIPENGINE_GGUF_WMMA_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"] == ""
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE"] == "0"
    assert os.environ[PROFILE_Q5_1_DOWN_M1_ENV] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL"] == "1"
    row4 = _selection_map(production)[("linear", "ungrouped_prefill_rows_ge64_q5k_gate_up")]
    assert row4["selected_variant"] == "selected_grouped_row4_gemv_bf16_bf16_out"
    assert row4["strict_fallback_variant"] == "selected_gemv_bf16_bf16_out"
    # The ds4-MMQ MoE suffixes are superseded by the certified WMMA-MoE27
    # route; their envs stay off so they cannot preempt it.
    assert os.environ["HIPENGINE_QWEN4_EXP_Q5_1_MMQ_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q5_1_MMQ_LAYERS"] == ""
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_K_MMQ_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_K_MMQ_LAYERS"] == ""
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_TILE_M"] == "16"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_TILE_N"] == "16"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL"] == "1"
    assert tuple(
        int(value)
        for value in os.environ["HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS"].split(",")
    ) == PRODUCTION_Q4_IU8_PREFILL_LAYERS
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"] == "1"
    assert tuple(
        int(value)
        for value in os.environ[
            "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS"
        ].split(",")
    ) == PRODUCTION_GDN_COLWARPS_PREFILL_LAYERS
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS"] == ""
    assert os.environ["HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL"] == "1"
    assert tuple(
        int(value)
        for value in os.environ[
            "HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS"
        ].split(",")
    ) == PRODUCTION_QSA_FLASH_PREFILL_LAYERS
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL"] == "1"
    assert tuple(
        int(value)
        for value in os.environ[
            "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL_LAYERS"
        ].split(",")
    ) == PRODUCTION_GDN_PEER_PREFILL_LAYERS
    assert os.environ["HIPENGINE_GGUF_Q8_0_WMMA_TILE_M"] == "64"
    assert os.environ["HIPENGINE_GGUF_Q8_0_WMMA_TILE_N"] == "32"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_DP4A64"] == "1"
    assert tuple(
        int(value)
        for value in os.environ["HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS"].split(",")
    ) == PRODUCTION_Q4_DP4A_DECODE_LAYERS

    def weight(layer: int):
        return SimpleNamespace(
            backend="hip_gfx1151",
            spec=SimpleNamespace(slot_path=f"layers.{layer}.expert_gate"),
        )

    assert not _qwen4_exp_production_moe_prefill_enabled(weight(26), rows=256)
    assert _qwen4_exp_production_moe_prefill_enabled(weight(27), rows=256)
    assert _qwen4_exp_production_moe_prefill_enabled(weight(47), rows=16)
    assert not _qwen4_exp_production_moe_prefill_enabled(weight(47), rows=15)

    strict = _resolve(ExecutionProfile.STRICT)
    assert strict.binder is not None
    strict.binder(SimpleNamespace(), strict)
    assert os.environ[PRODUCTION_MOE_PREFILL_ENV] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"] == ""
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE"] == "0"
    assert os.environ[PROFILE_Q5_1_DOWN_M1_ENV] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q5_1_MMQ_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q5_1_MMQ_LAYERS"] == ""
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_K_MMQ_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_K_MMQ_LAYERS"] == ""
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL_LAYERS"] == ""
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS"] == ""
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_DP4A64"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS"] == ""
    assert not _qwen4_exp_production_moe_prefill_enabled(weight(47), rows=256)


def test_qwen4_exp_profiles_cover_both_registered_quant_names() -> None:
    register_qwen4_exp_gfx1151_profiles()
    for quant in QWEN4_EXP_QUANTS:
        for profile in (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION):
            resolved = _resolve(profile, quant=quant)
            assert resolved.manifest["quant"] == quant
