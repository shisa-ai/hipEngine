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
    PRODUCTION_MOE_PREFILL_ENV,
    PRODUCTION_Q4_DP4A_DECODE_LAYERS,
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
    for name in (
        PRODUCTION_MOE_PREFILL_ENV,
        "HIPENGINE_QWEN4_EXP_Q4_DP4A64",
        "HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS",
        "HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
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
    assert strict.manifest_sha256 == strict.strict_manifest_sha256
    assert production.manifest_sha256 != production.strict_manifest_sha256
    assert not production.fell_back_to_strict
    assert production.manifest["kv_policy"] == "paged_bf16_qsa_index_f32"
    assert production.manifest["graph_policy"] == "request_owned_exact_moe_graph_c1"
    selections = _selection_map(production)
    gate = selections[("moe_linear", "prefill_rows_ge16_layers27_47_gate_up")]
    assert gate["selected_variant"] == "selected_dual_wmma_prefill_compact_bf16_bf16_out"
    assert gate["strict_fallback_variant"].startswith(
        "selected_dual_grouped_rowbatch8"
    )
    down = selections[("moe_linear", "prefill_rows_ge16_layers27_47_down")]
    assert down["selected_variant"] == "selected_grouped_wmma_prefill_compact_bf16_bf16_out"
    assert down["evidence_artifact"].endswith("moe27-q8-32-production.json")
    q8 = selections[("linear", "prefill_rows_ge16_layers32_47_q8")]
    assert q8["selected_variant"] == "wmma_prefill_f32_f32_out"
    assert q8["strict_fallback_variant"] == "coltile8_rowbatch4_f32_f32_out"
    dp4a = selections[("linear", "decode_c1_layers24_47_q4_gate_up")]
    assert dp4a["selected_variant"] == (
        "selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out"
    )
    assert dp4a["strict_fallback_variant"] == (
        "selected_dual_silu_logical128_t64_gemv_bf16_bf16_out"
    )
    assert dp4a["evidence_artifact"].endswith("production-dp4a24-decode.json")


def test_qwen4_exp_profile_binders_select_only_certified_late_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    production = _resolve(ExecutionProfile.PRODUCTION)
    assert production.binder is not None
    production.binder(SimpleNamespace(), production)
    assert os.environ[PRODUCTION_MOE_PREFILL_ENV] == "1"
    assert os.environ["HIPENGINE_GGUF_WMMA_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"].split(",")[0] == "32"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"].split(",")[-1] == "47"
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
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_DP4A64"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS"] == ""
    assert not _qwen4_exp_production_moe_prefill_enabled(weight(47), rows=256)


def test_qwen4_exp_profiles_cover_both_registered_quant_names() -> None:
    register_qwen4_exp_gfx1151_profiles()
    for quant in QWEN4_EXP_QUANTS:
        for profile in (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION):
            resolved = _resolve(profile, quant=quant)
            assert resolved.manifest["quant"] == quant
