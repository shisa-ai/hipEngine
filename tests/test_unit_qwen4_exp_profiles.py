from __future__ import annotations

import os
import warnings
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import (
    ExecutionProfile,
    clear_runtime_profile_registry_for_tests,
    resolve_runtime_profile,
)
from hipengine.generation.qwen4_exp_profiles import (
    PREBINDER_CONFLICT_ENV,
    PRODUCTION_GDN_COLWARPS_PREFILL_LAYERS,
    PRODUCTION_QSA_FLASH_PREFILL_LAYERS,
    PRODUCTION_GDN_PEER_PREFILL_LAYERS,
    PRODUCTION_MOE_PREFILL_ENV,
    PROFILE_Q5_1_DOWN_M1_ENV,
    PRODUCTION_Q4_DP4A_DECODE_LAYERS,
    PRODUCTION_Q4_IU8_PREFILL_LAYERS,
    PRODUCTION_Q4_K_MMQ_PREFILL_LAYERS,
    PRODUCTION_Q5_1_MMQ_PREFILL_LAYERS,
    PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS,
    PRODUCTION_Q8_WMMA_PREFILL_LAYERS,
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    QWEN4_EXP_QUANTS,
    NORM_F16_PUBLICATION_ENV,
    last_prebinder_conflicts,
    qwen4_exp_gfx1151_profiles_registered,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear
from hipengine.runtime.qwen4_exp_runner import (
    _qwen4_exp_production_moe_prefill_enabled,
)

# DP-2's shared-cast loan is a request-local runner decision. The gfx1151 pair
# policy still owns which rows, layers and ordered pairs may use it.
DP2_PAIR_ENV = "HIPENGINE_QWEN4_EXP_DENSE_F16_PAIR"


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
        "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE",
        "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS",
        "HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN",
        "HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL",
        "HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE",
        "HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE",
        "HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_VEC4",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_RAW_VECTOR",
        "HIPENGINE_QWEN4_EXP_Q8_DOWN_ROW4_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q8_DOWN_BUNDLE_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q8_MAPPED_DOWN",
        "HIPENGINE_QWEN4_EXP_Q8_DOWN_REGISTER",
        "HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q51_FOLD128_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q51_FOLD_PAIR_PREFILL",
        "HIPENGINE_QWEN4_EXP_Q51_REGISTER_CACHE",
        "HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL",
        "HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256",
        "HIPENGINE_QWEN4_EXP_ROUTED_F16",
        "HIPENGINE_QWEN4_EXP_GR_WIDE",
        "HIPENGINE_QWEN4_EXP_DENSE_F16_GEMM",
        "HIPENGINE_QWEN4_EXP_QSA_WMMA",
        NORM_F16_PUBLICATION_ENV,
        DP2_PAIR_ENV,
    )
    for name in environment_names:
        monkeypatch.delenv(name, raising=False)
    yield
    for name in environment_names:
        os.environ.pop(name, None)
    clear_runtime_profile_registry_for_tests()


def _resolve(profile: ExecutionProfile, *, quant: str = QWEN4_EXP_QUANTS[0]):
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=quant,
        profile=profile,
    )


@pytest.mark.parametrize("profile", [ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION])
def test_removed_dense_split_env_has_no_profile_or_manifest_binding(profile, monkeypatch):
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = _resolve(profile, quant="gguf_ud_q4_k_xl")
    original_manifest = resolved.manifest_sha256
    name = "HIPENGINE_QWEN4_EXP_Q8_DENSE_SPLIT_LAYERS"
    monkeypatch.setenv(name, "0,1,2")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved.construct_generator(lambda: SimpleNamespace())
    assert not any("DENSE_SPLIT_LAYERS" in str(w.message) for w in caught)
    assert os.environ[name] == "0,1,2"  # inert legacy value; no runtime reader
    assert name not in last_prebinder_conflicts()
    assert _resolve(profile, quant="gguf_ud_q4_k_xl").manifest_sha256 == original_manifest


def _selection_map(resolved):
    return {
        (row["layer"], row["scope"]): row
        for row in resolved.manifest["selections"]
    }


def _enable_historical_q4km_plan_for_structure_test():
    """Inspect quarantined implementation structure, never certify its arithmetic."""
    from dataclasses import replace
    from hipengine import execution_profiles as profiles

    key = profiles.RuntimeProfileKey(
        QWEN4_EXP_MODEL, QWEN4_EXP_BACKEND, "gguf_q4_k_m", ExecutionProfile.PRODUCTION
    )
    plan = profiles._RUNTIME_PROFILE_PLANS[key]
    assert plan.qualification_failure is not None
    assert _resolve(ExecutionProfile.PRODUCTION).fell_back_to_strict
    profiles.register_runtime_profile_plan(
        model=key.model, backend=key.backend, quant=key.quant, profile=key.profile,
        plan=replace(plan, qualification_failure=None), replace=True,
    )


def test_historical_q4km_manifest_structure_not_qualification() -> None:
    register_gfx1151_kernels(replace=True)
    assert register_qwen4_exp_gfx1151_profiles()
    assert qwen4_exp_gfx1151_profiles_registered()
    _enable_historical_q4km_plan_for_structure_test()

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
        ("moe_linear", "prefill_rows_ge2_lt64_exact_grouped_q5_1_down")
    ]
    assert strict_q5_m1["selected_variant"].endswith(
        "expertgrid64_bf16_bf16_out"
    )
    assert strict_q5_m1["strict_fallback_variant"].endswith(
        "expertgrid64_bf16_bf16_out"
    )
    strict_q8_grouped = _selection_map(strict)[
        ("linear", "grouped_prefill_rows_lt512_q8_0_expert_down")
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
    packed = selections[("linear", "prefill_gdn_qkv_ssm_prepacked_mmq")]
    assert packed["selected_variant"] == "mmq128_prepacked_vec4_q8_1_d4x3_guarded_f32_f32_out"
    assert packed["strict_fallback_variant"] == "coltile8_rowbatch4_f32_f32_out"
    argmax = selections[("argmax", "qwen4exp_normal_greedy_output")]
    assert argmax["selected_variant"] == "top1_i64"
    assert argmax["strict_fallback_variant"] == "top1_i64"
    assert argmax["evidence_artifact"].endswith("p5-device-argmax.json")
    qsa = selections[("paged_attn_decode", "qwen4exp_multirow_dense_qsa")]
    assert qsa["selected_variant"] == "bf16_context_batch_paged_c1_exact_spans"
    assert qsa["strict_fallback_variant"] == "bf16_context_batch_paged_c1_exact_spans"
    assert qsa["evidence_artifact"].endswith("p4-qsa-dense-fixed256.json")
    gr_up = selections[("linear+gr_gated_mean", "qwen4exp_rows_gt256_gr_up")]
    assert gr_up["selected_variant"] == "coltile2_branch4_rowbatch4_wave_scale_f32_exact"
    assert gr_up["strict_fallback_variant"] == "coltile2_branch4_rowbatch4_f32_exact"
    assert gr_up["evidence_artifact"].endswith("gr-wave-production.json")
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
        ("moe_linear", "prefill_rows_ge2_lt64_exact_grouped_q5_1_down")
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
    q8_grouped = selections[("linear", "grouped_prefill_rows_lt512_q8_0_expert_down")]
    q8_row4 = selections[("linear", "grouped_prefill_rows_ge512_q8_0_expert_down")]
    assert q8_row4["selected_variant"] == "selected_grouped_row4_register_gemv_bf16_bf16_out"
    assert q8_row4["strict_fallback_variant"] == "selected_gemv_bf16_bf16_out"
    assert q8_grouped["selected_variant"] == (
        "selected_grouped_gemv_bf16_bf16_out"
    )
    assert q8_grouped["strict_fallback_variant"] == (
        "selected_gemv_bf16_bf16_out"
    )
    assert q8_grouped["evidence_artifact"].endswith(
        "halo-pf13-production-refresh.json"
    )
    q8 = selections[("linear", "prefill_policy_qwen4exp_raw_q8_shapes")]
    assert q8["selected_variant"] == (
        "mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out"
    )
    assert q8["strict_fallback_variant"] == "coltile8_rowbatch4_f32_f32_out"
    gdn = selections[
        ("gdn_recurrence_norm_gate", "prefill_rows_ge2_layers27_47_gdn")
    ]
    assert gdn["selected_variant"] == "qwen4exp_gdn_tiled16_multi_prefill"
    assert gdn["strict_fallback_variant"] == "qwen4exp_sigmoid_strict_prefill"
    assert gdn["evidence_artifact"].endswith(
        "2026-10-08-qwen4exp-gdn-multicolumn-admission/gate.json"
    )
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


def test_ud_production_binds_the_promoted_gr_dpp_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    production = _resolve(
        ExecutionProfile.PRODUCTION, quant="gguf_ud_q4_k_xl"
    )
    production.binder(SimpleNamespace(), production)
    gr_up = _selection_map(production)[
        ("linear+gr_gated_mean", "qwen4exp_rows_gt256_gr_up")
    ]
    assert gr_up["selected_variant"] == (
        "coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact"
    )
    assert gr_up["strict_fallback_variant"] == (
        "coltile2_branch4_rowbatch4_f32_exact"
    )
    assert gr_up["evidence_artifact"].endswith(
        "qwen4exp-gr-dpp-reduction/README.md"
    )
    assert os.environ["HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION"] == "1"


def test_historical_q4km_binder_structure_not_qualification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    _enable_historical_q4km_plan_for_structure_test()
    production = _resolve(ExecutionProfile.PRODUCTION)
    assert production.binder is not None
    configure_calls: list[bool] = []
    fake_runner = SimpleNamespace(
        configure_mmq_prefill_resources=lambda: configure_calls.append(True)
    )
    production_generator = SimpleNamespace(runner=fake_runner)
    # A leftover value for the retired process switch may neither select the
    # leaf nor be overwritten by the binder.
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q51_ROW_PUBLISH", "0")
    production.binder(production_generator, production)
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
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_VEC4"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_RAW_VECTOR"] == "1"
    wave = _selection_map(production)[("linear", "prefill_exact_q8_f32_coltile")]
    assert wave["selected_variant"] == (
        "coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out"
    )
    assert wave["strict_fallback_variant"] == "coltile8_rowbatch4_f32_f32_out"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL"] == "1"
    # The GDN serial owner is request-local runner state published from the
    # plan's own selection, so the binder writes no process-wide switch for it.
    # This historical Q4_K_M production plan carries the non-recovery wide
    # serial scope, so it publishes the wide leaf (the shipped Q4_K_M plan fails
    # its envelope and resolves through the strict binder, which publishes the
    # parent instead).
    assert production_generator.gdn_prefill_variant == (
        "qwen4exp_sigmoid_wave_norm_prefill")
    assert fake_runner.gdn_prefill_variant == (
        "qwen4exp_sigmoid_wave_norm_prefill")
    assert "HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM" not in os.environ
    assert "HIPENGINE_QWEN4_EXP_GDN_PREFILL_VARIANT" not in os.environ
    assert fake_runner.dense_mmq_token64_enabled is True
    assert production_generator.qsa_prefill_head_mode == "quad"
    assert fake_runner.qsa_prefill_head_mode == "quad"
    assert "HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR" not in os.environ
    raw_q = _selection_map(production)[("linear","prefill_rows_ge512_k2560_n12288_raw_q")]
    assert raw_q["selected_variant"] == "mmq128_token64_q8_1_d4x3_guarded_f32_f32_out"
    assert raw_q["strict_fallback_variant"] == "coltile8_rowbatch4_f32_f32_out"
    gdn = _selection_map(production)[(
        "gdn_recurrence_norm_gate", "prefill_rows_ge2_serial_gdn_hk16_hv48_d128")]
    assert gdn["selected_variant"] == "qwen4exp_sigmoid_wave_norm_prefill"
    assert gdn["strict_fallback_variant"] == "qwen4exp_sigmoid_strict_prefill"
    pair = _selection_map(production)[("moe_linear", "prefill_rows_ge64_lt512_exact_grouped_q5_1_down")]
    assert pair["selected_variant"] == "selected_grouped_prefill_pair2_fold128_bf16_bf16_out"
    assert pair["strict_fallback_variant"] == (
        "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out")
    large_pair = _selection_map(production)[("moe_linear", "prefill_rows_ge512_k640_exact_grouped_q5_1_down")]
    assert large_pair["selected_variant"] == "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out"
    assert large_pair["strict_fallback_variant"] == pair["strict_fallback_variant"]
    assert os.environ["HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL"] == "page256"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_DOWN_ROW4_PREFILL"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_DOWN_BUNDLE_PREFILL"] == "1"
    # The token-major mapped Q8_0 down owner is request-local runner state
    # published from the binder's own fallback policy, so the binder writes no
    # process-wide switch for it.
    assert "HIPENGINE_QWEN4_EXP_Q8_MAPPED_DOWN" not in os.environ
    assert production_generator.q8_mapped_down is True
    assert fake_runner.q8_mapped_down is True
    # The Q8_0 down register selection is request-local runner state published
    # from the binder's own fallback policy, so the binder writes no
    # process-wide switch for it.
    assert "HIPENGINE_QWEN4_EXP_Q8_DOWN_REGISTER" not in os.environ
    assert production_generator.q8_down_register is True
    assert fake_runner.q8_down_register is True
    mapped = _selection_map(production)[("linear","prefill_rows_ge512_token_major_mapped_q8_down")]
    assert mapped["selected_variant"] == "selected_grouped_row4_register_gemv_bf16_bf16_out"
    assert mapped["strict_fallback_variant"] == "selected_gemv_bf16_bf16_out"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q51_FOLD128_PREFILL"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q51_FOLD_PAIR_PREFILL"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q51_REGISTER_CACHE"] == "1"
    # The Q5_1 row-publish selection is request-local runner state published
    # from the plan's own rows>=512 K640 row, so the binder writes no
    # process-wide switch for it.
    assert production_generator.q51_row_publish is True
    assert fake_runner.q51_row_publish is True
    assert os.environ["HIPENGINE_QWEN4_EXP_Q51_ROW_PUBLISH"] == "0"
    selected = _selection_map(production)
    assert selected[("moe_linear", "prefill_rows_ge2_lt64_exact_grouped_q4_gate_up")]["selected_variant"] == (
        "selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out")
    assert selected[("moe_linear", "prefill_rows_ge64_exact_grouped_q4_gate_up")]["selected_variant"] == (
        "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out")
    assert selected[("moe_linear", "prefill_rows_ge64_exact_grouped_q4_gate_up")]["strict_fallback_variant"] == (
        "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out")
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT"] == "1"
    assert selected[("qsa_sparse_attention", "prefill_h256_page256_sparse_rows_ge16")]["selected_variant"] == (
        "strict_h256_head_quad_rows_spans")
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
    strict_generator = SimpleNamespace()
    strict.binder(strict_generator, strict)
    assert strict_generator.dense_mmq_token64_enabled is False
    assert os.environ[PRODUCTION_MOE_PREFILL_ENV] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"] == ""
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE"] == "0"
    assert os.environ[PROFILE_Q5_1_DOWN_M1_ENV] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_VEC4"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_RAW_VECTOR"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_DOWN_ROW4_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_DOWN_BUNDLE_PREFILL"] == "0"
    assert "HIPENGINE_QWEN4_EXP_Q8_MAPPED_DOWN" not in os.environ
    assert strict_generator.q8_mapped_down is False
    assert "HIPENGINE_QWEN4_EXP_Q8_DOWN_REGISTER" not in os.environ
    assert strict_generator.q8_down_register is False
    assert os.environ["HIPENGINE_QWEN4_EXP_Q51_FOLD128_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q51_FOLD_PAIR_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q51_REGISTER_CACHE"] == "0"
    assert strict_generator.q51_row_publish is False
    assert os.environ["HIPENGINE_QWEN4_EXP_Q51_ROW_PUBLISH"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL"] == "0"
    assert strict_generator.gdn_prefill_variant is None
    assert "HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM" not in os.environ
    assert "HIPENGINE_QWEN4_EXP_GDN_PREFILL_VARIANT" not in os.environ
    assert strict_generator.qsa_prefill_head_mode == "parent"
    assert "HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR" not in os.environ
    assert os.environ["HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL"] == "0"
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


@pytest.mark.parametrize("profile", [ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION, ExecutionProfile.FAST])
def test_ud_q4_decode_binding_matches_the_manifest_after_numerical_repair(profile):
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = _resolve(profile, quant="gguf_ud_q4_k_xl")
    resolved.construct_generator(lambda: SimpleNamespace())
    decode = _selection_map(resolved)[("linear", "decode_c1_calibrated_q4_dp4a_43_layers")]
    assert decode["selected_variant"] == "selected_dual_silu_logical128_t64_gemv_bf16_bf16_out"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_DP4A64"] == "0"
    if profile != ExecutionProfile.STRICT:
        # Repair the decode operation, not the complete production composition.
        assert os.environ["HIPENGINE_QWEN4_EXP_ROUTED_F16"] == "1"
        assert os.environ["HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE"] == "1"
        assert os.environ["HIPENGINE_QWEN4_EXP_QSA_WMMA"] == "1"
        # The Q5_1 row-publish publication is the fallback selection for its
        # scope, not the owner that runs: the routed F16 WMMA down owns it, the
        # exact iu8 down sits behind it, and the pair2 chain's own gate is on.
        # A diagnostic arm closes the first two for the counted leaf to launch.
        assert os.environ["HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT"] == "1"
        assert os.environ["HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN"] == "1"


def test_shipped_ud_plan_publishes_the_production_row_publish_fallback() -> None:
    """The row-publish publication must not narrow to the manifest row.

    The shipped F16 plan replaced the Q5_1 row-publish row with the newer
    routed F16 owner, so the row no longer names the fallback. The retired
    switch was bound production-only, so the effective fallback publication
    stays True on a production plan; otherwise the pair2 chain's row-publish
    sibling is unreachable and the arm measures the register-cache parent.
    """

    from hipengine.kernels.registry import KernelKey
    from hipengine.runtime import qwen4_exp_runner as runner

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = _resolve(ExecutionProfile.PRODUCTION, quant="gguf_ud_q4_k_xl")
    generator = resolved.construct_generator(lambda: SimpleNamespace())
    row = next(
        selection for selection in resolved.manifest["selections"]
        if selection["layer"] == "moe_linear"
        and selection["scope"]
        == "prefill_rows_ge512_k640_exact_grouped_q5_1_down"
    )
    assert row["selected_variant"] == "selected_routed_f16_wmma_prefill_bf16_bf16_out"
    assert generator.q51_row_publish is True
    # Through the runtime pair2 selector that publication is the row-publish
    # sibling of the register-cache owner.
    parent = KernelKey(
        "hip_gfx1151", "moe_linear", "gguf_q5_1",
        "selected_grouped_prefill_pair2_register_cache_bf16_bf16_out")
    assert runner._qwen4_exp_q51_row_publish_key(
        parent, rows=1024, in_features=640,
        published=generator.q51_row_publish,
    ).variant == "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out"
    # And a c1 runner cloned from that generator carries the same decision.
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    clone = _inherit_lane_decisions(generator, SimpleNamespace())
    assert clone.q51_row_publish is True


def test_shipped_ud_plan_publishes_the_production_down_register_fallback() -> None:
    """The down-register publication must not narrow to either manifest row.

    The shipped plan replaced the sorted chain's row
    (``grouped_prefill_rows_ge512_q8_0_expert_down``) with the routed F16 owner
    while the mapped row still names the register leaf, so the manifest cannot
    carry one decision for both paths. The retired switch was bound
    production-only, so the effective fallback publication stays True on a
    production plan and both MoE paths select the register owner.
    """

    from hipengine.kernels.registry import KernelKey
    from hipengine.runtime import qwen4_exp_runner as runner

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = _resolve(ExecutionProfile.PRODUCTION, quant="gguf_ud_q4_k_xl")
    generator = resolved.construct_generator(lambda: SimpleNamespace())
    selections = _selection_map(resolved)
    sorted_row = selections[
        ("linear", "grouped_prefill_rows_ge512_q8_0_expert_down")]
    mapped_row = selections[
        ("linear", "prefill_rows_ge512_token_major_mapped_q8_down")]
    # The sorted chain's row now names its preemptor; the mapped row still names
    # the register leaf. One published value has to serve both.
    assert sorted_row["selected_variant"] == (
        "selected_routed_f16_wmma_prefill_bf16_bf16_out")
    assert mapped_row["selected_variant"] == (
        "selected_grouped_row4_register_gemv_bf16_bf16_out")
    assert generator.q8_down_register is True
    assert generator.q8_mapped_down is True
    assert "HIPENGINE_QWEN4_EXP_Q8_DOWN_REGISTER" not in os.environ
    assert "HIPENGINE_QWEN4_EXP_Q8_MAPPED_DOWN" not in os.environ
    bundle = KernelKey(
        "hip_gfx1151", "linear", "gguf_q8_0",
        "selected_grouped_row4_bundle_gemv_bf16_bf16_out")
    assert runner._qwen4_exp_q8_down_register_key(
        bundle, rows=1024, in_features=640,
        published=generator.q8_down_register,
    ).variant == "selected_grouped_row4_register_gemv_bf16_bf16_out"
    # The token-major mapped path resolves the same bundle key from the map it
    # built on this call, so the one mapped publication admits it.
    assert runner._qwen4_exp_mapped_down_key(
        backend="hip_gfx1151", quant="gguf_q8_0", rows=1024, map_ready=True,
        published=generator.q8_mapped_down,
    ).variant == "selected_grouped_row4_bundle_gemv_bf16_bf16_out"
    # And a c1 runner cloned from that generator carries the same decisions.
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    clone = _inherit_lane_decisions(generator, SimpleNamespace())
    assert clone.q8_down_register is True
    assert clone.q8_mapped_down is True


def test_qwen4_exp_profiles_cover_both_registered_quant_names() -> None:
    register_qwen4_exp_gfx1151_profiles()
    for quant in QWEN4_EXP_QUANTS:
        for profile in (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION):
            resolved = _resolve(profile, quant=quant)
            assert resolved.manifest["quant"] == quant


def test_binder_reports_a_pre_binder_value_it_discards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A route variable set before construction is inert, so say so loudly.

    The binder owns every variable it writes. Setting one before the generator
    exists means the intended route never runs and the experiment reports a
    clean null that looks like a real negative result.
    """

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    production = _resolve(ExecutionProfile.PRODUCTION)
    assert production.binder is not None
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS", "32,33,34")

    with pytest.warns(RuntimeWarning, match="HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"):
        production.binder(SimpleNamespace(runner=None), production)

    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"] == ""
    assert last_prebinder_conflicts()["HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"] == (
        "32,33,34",
        "",
    )


def test_binder_can_be_made_to_raise_on_a_discarded_pre_binder_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    production = _resolve(ExecutionProfile.PRODUCTION)
    assert production.binder is not None
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS", "0-47")
    monkeypatch.setenv(PREBINDER_CONFLICT_ENV, "error")

    with pytest.raises(RuntimeError, match="inert"):
        production.binder(SimpleNamespace(runner=None), production)


def test_binder_does_not_report_its_own_previous_bind_as_a_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict-then-production binding in one process is normal."""

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    strict = _resolve(ExecutionProfile.STRICT)
    production = _resolve(ExecutionProfile.PRODUCTION)
    assert strict.binder is not None and production.binder is not None

    strict.binder(SimpleNamespace(runner=None), strict)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        production.binder(SimpleNamespace(runner=None), production)
    assert last_prebinder_conflicts() == {}


def test_production_binds_the_certified_q8_dense_wide_prefill_scope() -> None:
    """UD-Q4_K_XL binds the dense-wide route at every layer.

    Layers 13-47 were certified on their own in
    benchmarks/results/2026-10-04-qwen4exp-dense-wide-13-47-frontier/; the
    complete F16 route composition, which extends the window to layer 0, is
    gated in benchmarks/results/2026-10-14-qwen4exp-fast-composition-gate-2/.
    The previous f16 WMMA scope stays disabled rather than competing with
    dense-wide on the same shapes.
    """

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    production = _resolve(ExecutionProfile.PRODUCTION, quant="gguf_ud_q4_k_xl")
    assert production.binder is not None

    production.binder(SimpleNamespace(runner=None), production)

    assert PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS == tuple(range(13, 48))
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS"] == ",".join(
        str(n) for n in range(48)
    )
    assert os.environ[NORM_F16_PUBLICATION_ENV] == "1"
    # The f16 WMMA dense route is retired at this scope for this quant; the
    # selector and its env var stay available for explicit opt-in and gating.
    assert PRODUCTION_Q8_WMMA_PREFILL_LAYERS == tuple(range(16, 48))
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"] == ""


@pytest.mark.parametrize("profile,quant,selected,fallback,enabled", [
    (ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl", "f32_f16_publish", "strict", "1"),
    (ExecutionProfile.STRICT, "gguf_ud_q4_k_xl", "strict", "strict", "0"),
    (ExecutionProfile.PRODUCTION, "gguf_q4_k_m", None, None, "0"),
])
def test_norm_publication_manifest_and_binding(profile, quant, selected, fallback, enabled):
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = _resolve(profile, quant=quant)
    rows = [row for row in resolved.manifest["selections"]
            if row["layer"] == "gr_grouped_rmsnorm"]
    if selected is None:
        assert rows == []
    else:
        assert len(rows) == 1
        assert rows[0]["scope"] == "prefill_rows4096_branches4_h2560_hc_down_layers13_47"
        assert rows[0]["selected_variant"] == selected
        assert rows[0]["strict_fallback_variant"] == fallback
        assert rows[0]["registry_quant"] == "bf16_f32"
    resolved.binder(SimpleNamespace(runner=None), resolved)
    assert os.environ[NORM_F16_PUBLICATION_ENV] == enabled


def test_norm_publication_explicit_rollback_is_manifested(monkeypatch):
    monkeypatch.setenv(NORM_F16_PUBLICATION_ENV, "0")
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = _resolve(ExecutionProfile.PRODUCTION, quant="gguf_ud_q4_k_xl")
    row = next(row for row in resolved.manifest["selections"]
               if row["layer"] == "gr_grouped_rmsnorm")
    assert row["selected_variant"] == row["strict_fallback_variant"] == "strict"
    resolved.binder(SimpleNamespace(runner=None), resolved)
    assert os.environ[NORM_F16_PUBLICATION_ENV] == "0"


def test_norm_publication_rebinding_cannot_leak_into_q4_or_strict():
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    for profile, quant, enabled in (
        (ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl", "1"),
        (ExecutionProfile.PRODUCTION, "gguf_q4_k_m", "0"),
        (ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl", "1"),
        (ExecutionProfile.STRICT, "gguf_ud_q4_k_xl", "0"),
    ):
        resolved = _resolve(profile, quant=quant)
        resolved.binder(SimpleNamespace(runner=None), resolved)
        assert os.environ[NORM_F16_PUBLICATION_ENV] == enabled


def test_strict_leaves_the_q8_dense_prefill_scopes_empty() -> None:
    """Strict keeps the exact coltile chain on every layer."""

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    strict = _resolve(ExecutionProfile.STRICT, quant="gguf_ud_q4_k_xl")
    assert strict.binder is not None

    strict.binder(SimpleNamespace(runner=None), strict)

    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"] == ""
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS"] == ""
    assert os.environ[NORM_F16_PUBLICATION_ENV] == "0"


# DP-2 loans one F16 staging plane to two immediate consumers of the wide route.
# Binding cannot widen the scope: admission still reads the backend policy.
_DP2_K, _DP2_NA, _DP2_NB = 2560, 10240, 6144
_DP2_KEY = KernelKey("hip_gfx1151", "linear", "gguf_q8_0", "dense_wide256_f16in_f32_f32_out")


def _dp2_weight(role: str, ptr: int, n: int, *, layer: int = 24):
    return SimpleNamespace(
        backend=_DP2_KEY.backend,
        spec=SimpleNamespace(
            layout="raw_gguf",
            quant_key=_DP2_KEY.quant,
            slot_path=f"layers.{layer}.{role}",
            source=SimpleNamespace(shape=(n, _DP2_K)),
        ),
        allocation=lambda _: SimpleNamespace(
            tensor=SimpleNamespace(ptr=ptr),
            buffer=SimpleNamespace(nbytes=n * (_DP2_K // 32) * 34),
        ),
    )


def _dp2_pair_admitted(monkeypatch, decision: bool, *, layer: int = 24, rows: int = 4096,
                       pair=("attn_qkv", "attn_gate", _DP2_NA, _DP2_NB)) -> bool:
    """Open the scope the runner opens around one ordered projection pair."""

    monkeypatch.setattr(linear, "_wide_f16_consumer_key", lambda *a, **kw: _DP2_KEY)
    weight_a = _dp2_weight(pair[0], 5 << 32, pair[2], layer=layer)
    weight_b = _dp2_weight(pair[1], 6 << 32, pair[3], layer=layer)
    with linear.wide_f16_pair_session(decision), linear.wide_f16_activation_session(
        workspace_ptr=1 << 32, workspace_nbytes=4096 * _DP2_K * 2,
    ):
        with linear.wide_f16_activation_pair(
            weight_a, weight_b, 2 << 32, 3 << 32, 4 << 32,
            rows=rows, in_features=_DP2_K, out_features_a=pair[2],
            out_features_b=pair[3], stream=7, runtime=object(),
        ) as loan:
            return loan is not None


@pytest.mark.parametrize("profile,quant,enabled", [
    (ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl", True),
    (ExecutionProfile.STRICT, "gguf_ud_q4_k_xl", False),
    (ExecutionProfile.BATCH_INVARIANT, "gguf_ud_q4_k_xl", False),
    (ExecutionProfile.PRODUCTION, "gguf_q4_k_m", False),
    (ExecutionProfile.FAST, "gguf_ud_q4_k_xl", False),
])
def test_dp2_pair_is_bound_only_for_ud_production(profile, quant, enabled):
    """The wide route's lane owns the loan; every other profile keeps parent."""

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = _resolve(profile, quant=quant)
    runner = SimpleNamespace()
    generator = SimpleNamespace(runner=runner)
    resolved.binder(generator, resolved)
    assert runner.dense_f16_pair_enabled is enabled
    assert generator.dense_f16_pair_enabled is enabled
    # The decision is request-local: no process-wide switch may appear for it.
    assert DP2_PAIR_ENV not in os.environ


def test_dp2_pair_binding_reaches_a_runner_built_after_the_bind():
    """The generator carries the decision; late c1 runners inherit it."""

    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    production = _resolve(ExecutionProfile.PRODUCTION, quant="gguf_ud_q4_k_xl")
    generator = SimpleNamespace(runner=None)
    production.binder(generator, production)
    assert generator.dense_f16_pair_enabled is True
    assert generator.qsa_prefill_head_mode == "quad"
    late = _inherit_lane_decisions(generator, SimpleNamespace())
    assert late.dense_f16_pair_enabled is True
    assert late.dense_mmq_token64_enabled is True
    assert late.qsa_prefill_head_mode == "quad"

    strict = _resolve(ExecutionProfile.STRICT, quant="gguf_ud_q4_k_xl")
    strict.binder(generator, strict)
    assert generator.dense_f16_pair_enabled is False
    assert generator.qsa_prefill_head_mode == "parent"
    strict_late = _inherit_lane_decisions(generator, SimpleNamespace())
    assert strict_late.dense_f16_pair_enabled is False
    assert strict_late.dense_mmq_token64_enabled is False
    assert strict_late.qsa_prefill_head_mode == "parent"

    # An unbound generator keeps the parent route for anything cloned from it.
    unbound = _inherit_lane_decisions(SimpleNamespace(), SimpleNamespace())
    assert unbound.dense_f16_pair_enabled is False
    assert unbound.dense_mmq_token64_enabled is False
    assert unbound.qsa_prefill_head_mode == "parent"


@pytest.mark.parametrize("profile,quant,expected", [
    # Production lanes publish the four-head fallback grouping, and the fast
    # route shares this binder.
    (ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl", "quad"),
    (ExecutionProfile.FAST, "gguf_ud_q4_k_xl", "quad"),
    # This plan falls back to the strict binder, whose selections keep the
    # single-head parent.
    (ExecutionProfile.PRODUCTION, "gguf_q4_k_m", "parent"),
    (ExecutionProfile.STRICT, "gguf_ud_q4_k_xl", "parent"),
    (ExecutionProfile.BATCH_INVARIANT, "gguf_ud_q4_k_xl", "parent"),
])
def test_qsa_prefill_head_mode_follows_the_binder_scope(profile, quant, expected):
    """The binder publishes the retired switch's own scope, not a manifest row.

    The resolved manifest names the WMMA owner for this scope, so deriving the
    mode from it would publish ``parent`` on the lane that publishes ``quad``.
    """

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = _resolve(profile, quant=quant)
    runner = SimpleNamespace()
    generator = SimpleNamespace(runner=runner)
    resolved.binder(generator, resolved)
    assert generator.qsa_prefill_head_mode == expected
    assert runner.qsa_prefill_head_mode == expected
    assert "HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR" not in os.environ


def test_non_ud_production_bind_keeps_the_original_publication():
    """The retired switch published quad for every production bind.

    The arithmetic-recovery sweep that clears the flag is guarded by the UD
    quant and the restoration pass republishes the same value for it, so a
    production bind whose plan names another quant also published quad. Pinning
    the non-UD production scope here keeps a future narrowing to "UD only" a
    deliberate decision instead of a silent widening or regression.
    """

    import dataclasses

    from hipengine.generation.qwen4_exp_profiles import _production_binder

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = _resolve(ExecutionProfile.PRODUCTION, quant="gguf_ud_q4_k_xl")
    assert resolved.manifest["quant"] == "gguf_ud_q4_k_xl"
    non_ud = dataclasses.replace(
        resolved, manifest={**resolved.manifest, "quant": "gguf_q4_k_m"})
    for plan in (resolved, non_ud):
        runner = SimpleNamespace()
        generator = SimpleNamespace(runner=runner)
        _production_binder(generator, plan)
        assert generator.qsa_prefill_head_mode == "quad"
        assert runner.qsa_prefill_head_mode == "quad"
        assert "HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR" not in os.environ


def test_no_plan_declares_the_diagnostic_pair_leaf():
    """The four-head leaf is declared; ``pair`` is diagnostic-only.

    The UD lane's resolved manifest names the WMMA owner for the same scope, so
    this checks the selection sets rather than the resolved row. ``pair`` must
    stay undeclared: publishing it would silently change what production
    measures.
    """

    from hipengine.generation.qwen4_exp_profiles import (
        _production_selections,
        _strict_selections,
    )

    declared = {
        selection.selected_variant
        for selections in (
            _production_selections(),
            _production_selections(recovery=True),
            _strict_selections(),
            _strict_selections(recovery=True),
        )
        for selection in selections
    }
    assert "strict_h256_head_quad_rows_spans" in declared
    assert "strict_h256_head_pair_rows_spans" not in declared


@pytest.mark.parametrize("profile,quant,enabled", [
    (ExecutionProfile.PRODUCTION, "gguf_ud_q4_k_xl", True),
    # This plan falls back to the strict binder, which binds the parent leg.
    (ExecutionProfile.PRODUCTION, "gguf_q4_k_m", False),
    (ExecutionProfile.FAST, "gguf_ud_q4_k_xl", True),
    (ExecutionProfile.STRICT, "gguf_ud_q4_k_xl", False),
    (ExecutionProfile.BATCH_INVARIANT, "gguf_ud_q4_k_xl", False),
])
def test_token64_dense_leg_follows_the_q8_mmq_prefill_lane(profile, quant, enabled):
    """The leg rides the profile's Q8 raw-MMQ lane, not a process-wide switch."""

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = _resolve(profile, quant=quant)
    runner = SimpleNamespace()
    generator = SimpleNamespace(runner=runner)
    resolved.binder(generator, resolved)
    assert runner.dense_mmq_token64_enabled is enabled
    assert generator.dense_mmq_token64_enabled is enabled
    assert "HIPENGINE_QWEN4_EXP_MMQ_TOKEN64" not in os.environ


def test_dp2_pair_decision_is_owned_by_the_runner_it_was_bound_to():
    """Interleaved binds must not overwrite another runner's decision."""

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    first, second = SimpleNamespace(), SimpleNamespace()
    production = _resolve(ExecutionProfile.PRODUCTION, quant="gguf_ud_q4_k_xl")
    strict = _resolve(ExecutionProfile.STRICT, quant="gguf_ud_q4_k_xl")
    production.binder(SimpleNamespace(runner=first), production)
    strict.binder(SimpleNamespace(runner=second), strict)
    assert first.dense_f16_pair_enabled is True
    assert second.dense_f16_pair_enabled is False
    # A later bind on the same runner retracts it; the other runner is untouched.
    strict.binder(SimpleNamespace(runner=first), strict)
    assert first.dense_f16_pair_enabled is False
    assert second.dense_f16_pair_enabled is False
    production.binder(SimpleNamespace(runner=second), production)
    assert second.dense_f16_pair_enabled is True
    assert DP2_PAIR_ENV not in os.environ


def test_dp2_pair_binding_reaches_the_route_only_for_eligible_pairs(monkeypatch):
    """Binding opens the loan for the measured pair and nothing else."""

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    production = _resolve(ExecutionProfile.PRODUCTION, quant="gguf_ud_q4_k_xl")
    runner = SimpleNamespace()
    production.binder(SimpleNamespace(runner=runner), production)

    assert _dp2_pair_admitted(monkeypatch, runner.dense_f16_pair_enabled)
    assert _dp2_pair_admitted(monkeypatch, runner.dense_f16_pair_enabled, layer=13)
    assert _dp2_pair_admitted(
        monkeypatch, runner.dense_f16_pair_enabled,
        pair=("shared_gate", "shared_up", 640, 640))
    assert not _dp2_pair_admitted(monkeypatch, runner.dense_f16_pair_enabled, layer=12)
    assert not _dp2_pair_admitted(monkeypatch, runner.dense_f16_pair_enabled, rows=1024)
    assert not _dp2_pair_admitted(
        monkeypatch, runner.dense_f16_pair_enabled,
        pair=("shared_gate", "shared_down", 640, 2560))

    strict = _resolve(ExecutionProfile.STRICT, quant="gguf_ud_q4_k_xl")
    strict_runner = SimpleNamespace()
    strict.binder(SimpleNamespace(runner=strict_runner), strict)
    assert not _dp2_pair_admitted(monkeypatch, strict_runner.dense_f16_pair_enabled)


def test_the_shipped_default_routes_the_certified_window_to_the_wide_kernel() -> None:
    """A caller that names no profile must still get the wide kernel at 0-47.

    Promotion is only real if the shipped default reaches the route, and that
    guarantee is split across three places: the default profile resolution, the
    production binder, and the selector's own precedence. Each was pinned
    separately, so this test joins them. Removing any one link -- the registered
    production plan, the binding, or the route's claim on the f16 parent --
    fails here rather than silently reverting prefill to the exact chain.

    The scope the default must leave alone is checked in the same place:
    every row count the sub-256-row owner covers.
    """

    from hipengine.execution_profiles import resolve_default_execution_profile
    from hipengine.kernels.registry import KernelKey, register
    from hipengine.runtime.gguf_linear import (
        GGUFLinearDispatch,
        _q8_dense_wide_dispatch,
    )

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()

    # Unbound: this is the migration path, which keeps the exact chain.
    assert os.environ.get("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE") is None

    requested = resolve_default_execution_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant="gguf_ud_q4_k_xl",
    )
    assert requested is ExecutionProfile.PRODUCTION

    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant="gguf_ud_q4_k_xl",
        profile=requested,
    )
    assert resolved.binder is not None
    resolved.binder(SimpleNamespace(runner=None), resolved)
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE"] == "1"

    candidate = KernelKey(
        QWEN4_EXP_BACKEND, "linear", "gguf_q8_0", "dense_wide256_f32_f32_out"
    )
    register(candidate, lambda *args, **kwargs: None, replace=True)
    parent = GGUFLinearDispatch(
        KernelKey(
            QWEN4_EXP_BACKEND,
            "linear",
            "gguf_q8_0",
            "coltile8_rowbatch4_wave_scale_f32_f32_out",
        ),
        "raw",
    )

    def weight(layer: int):
        # ffn_up_shexp on this pack is 2560 -> 640 in Q8_0, the shape the route
        # was certified on.
        return SimpleNamespace(
            spec=SimpleNamespace(
                slot_path=f"layers.{layer}.ffn_up_shexp", quant_key="gguf_q8_0"
            )
        )

    for layer in (0, 12, 13, 16, 47):
        assert _q8_dense_wide_dispatch(
            parent, rows=1024, in_features=2560, out_features=640, weight=weight(layer)
        ) == GGUFLinearDispatch(candidate, "raw"), layer

    for rows in (1, 256):
        assert (
            _q8_dense_wide_dispatch(
                parent,
                rows=rows,
                in_features=2560,
                out_features=640,
                weight=weight(16),
            )
            == parent
        ), rows


_ROUTED_F16_ENV = "HIPENGINE_QWEN4_EXP_ROUTED_F16"
_ROUTED_GATE_UP = "selected_routed_f16_dual_silu_wmma_prefill_bf16_bf16_out"
_ROUTED_LINEAR = "selected_routed_f16_wmma_prefill_bf16_bf16_out"
_ROUTED_SCOPES = {
    ("moe_linear", "prefill_rows_ge2_lt64_exact_grouped_q4_gate_up"): _ROUTED_GATE_UP,
    ("moe_linear", "prefill_rows_ge64_exact_grouped_q4_gate_up"): _ROUTED_GATE_UP,
    ("linear", "ungrouped_prefill_rows_ge64_q5k_gate_up"): _ROUTED_GATE_UP,
    ("moe_linear", "prefill_rows_ge2_lt64_exact_grouped_q5_1_down"): _ROUTED_LINEAR,
    ("moe_linear", "prefill_rows_ge64_lt512_exact_grouped_q5_1_down"): _ROUTED_LINEAR,
    ("moe_linear", "prefill_rows_ge512_k640_exact_grouped_q5_1_down"): _ROUTED_LINEAR,
    ("linear", "grouped_prefill_rows_lt512_q8_0_expert_down"): _ROUTED_LINEAR,
    ("linear", "grouped_prefill_rows_ge512_q8_0_expert_down"): _ROUTED_LINEAR,
    ("router_logits", "qwen4exp_multirow_f32_router"): "f32_hidden_f16_wmma_gemm",
    ("qsa_sparse_attention", "prefill_h256_page256_sparse_rows_ge16"):
        "wmma_f16_masked_prefill_f32",
    ("paged_attn_decode", "qwen4exp_multirow_dense_qsa"): "wmma_f16_masked_prefill_f32",
}


def test_production_plan_carries_the_f16_route_composition() -> None:
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    strict = _selection_map(_resolve(ExecutionProfile.STRICT, quant="gguf_ud_q4_k_xl"))
    production = _resolve(ExecutionProfile.PRODUCTION, quant="gguf_ud_q4_k_xl")
    assert not production.fell_back_to_strict

    rows = _selection_map(production)
    assert set(rows) == set(strict)
    for key, variant in _ROUTED_SCOPES.items():
        assert rows[key]["selected_variant"] == variant
        assert rows[key]["strict_fallback_variant"] == strict[key]["selected_variant"]
        assert rows[key]["selected_variant"] != strict[key]["selected_variant"]

    # The composition is certified for this quant only.
    other = _selection_map(_resolve(ExecutionProfile.PRODUCTION, quant="gguf_q4_k_m"))
    assert not {row["selected_variant"] for row in other.values()} & set(
        _ROUTED_SCOPES.values()
    )


def test_fast_carries_the_production_plan_until_a_route_lands_in_it() -> None:
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    production = _resolve(ExecutionProfile.PRODUCTION, quant="gguf_ud_q4_k_xl")
    fast = _resolve(ExecutionProfile.FAST, quant="gguf_ud_q4_k_xl")

    assert fast.profile is ExecutionProfile.FAST
    assert fast.source_profile is ExecutionProfile.FAST
    assert fast.manifest["execution_profile"] == "fast"
    assert fast.strict_manifest_sha256 == production.strict_manifest_sha256
    assert _selection_map(fast) == _selection_map(production)


def test_fast_is_opt_in_and_registered_only_for_the_in_envelope_quant() -> None:
    from hipengine.execution_profiles import resolve_default_execution_profile

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    for quant in QWEN4_EXP_QUANTS:
        assert (
            resolve_default_execution_profile(
                model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND, quant=quant
            )
            is (ExecutionProfile.STRICT if quant == "gguf_q4_k_m" else ExecutionProfile.PRODUCTION)
        )
    unregistered = _resolve(ExecutionProfile.FAST, quant="gguf_q4_k_m")
    assert unregistered.fell_back_to_strict
    assert unregistered.source_profile is ExecutionProfile.STRICT


def test_production_binder_publishes_the_f16_routes_and_strict_retracts_them() -> None:
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    production = _resolve(ExecutionProfile.PRODUCTION, quant="gguf_ud_q4_k_xl")
    other = _resolve(ExecutionProfile.PRODUCTION, quant="gguf_q4_k_m")
    strict = _resolve(ExecutionProfile.STRICT, quant="gguf_ud_q4_k_xl")
    selectors = (
        _ROUTED_F16_ENV,
        "HIPENGINE_QWEN4_EXP_GR_WIDE",
        "HIPENGINE_QWEN4_EXP_DENSE_F16_GEMM",
        "HIPENGINE_QWEN4_EXP_QSA_WMMA",
    )
    all_layers = ",".join(map(str, range(48)))

    production.binder(SimpleNamespace(), production)
    for key in selectors:
        assert os.environ[key] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS"] == all_layers
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS"] == all_layers
    assert (
        os.environ["HIPENGINE_QWEN4_EXP_ROUTER_VARIANT"]
        == "f32_hidden_f16_wmma_gemm"
    )

    # A later bind without the composition restores its own routes.
    other.binder(SimpleNamespace(), other)
    for key in selectors:
        assert os.environ[key] == "0"
    assert other.fell_back_to_strict
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS"] == ""

    production.binder(SimpleNamespace(), production)
    strict.binder(SimpleNamespace(), strict)
    for key in selectors:
        assert os.environ[key] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS"] == ""


@pytest.mark.parametrize("quant,requested", [
    ("gguf_ud_q4_k_xl", ExecutionProfile.STRICT),
    ("gguf_q4_k_m", ExecutionProfile.PRODUCTION),
])
def test_strict_binder_clears_caller_set_composition_selectors(
    monkeypatch: pytest.MonkeyPatch, quant, requested,
) -> None:
    from hipengine.generation import qwen4_exp_profiles as profiles

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    strict = _resolve(requested, quant=quant)
    monkeypatch.setattr(profiles, "_binder_written", {})
    for key in profiles._F16_ROUTE_SELECTORS:
        monkeypatch.setenv(key, "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        strict.binder(SimpleNamespace(), strict)
    for key in profiles._F16_ROUTE_SELECTORS:
        assert os.environ[key] == "0", key
        assert any(key in str(w.message) for w in caught), key
