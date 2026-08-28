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
    gate = selections[("moe_linear", "prefill_rows_ge16_layers32_47_gate_up")]
    assert gate["selected_variant"] == "selected_dual_wmma_prefill_compact_bf16_bf16_out"
    assert gate["strict_fallback_variant"].startswith(
        "selected_dual_grouped_rowbatch8"
    )
    down = selections[("moe_linear", "prefill_rows_ge16_layers32_47_down")]
    assert down["selected_variant"] == "selected_grouped_wmma_prefill_compact_bf16_bf16_out"
    assert down["evidence_artifact"].endswith("late-moe-production.json")


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

    def weight(layer: int):
        return SimpleNamespace(
            backend="hip_gfx1151",
            spec=SimpleNamespace(slot_path=f"layers.{layer}.expert_gate"),
        )

    assert not _qwen4_exp_production_moe_prefill_enabled(weight(31), rows=256)
    assert _qwen4_exp_production_moe_prefill_enabled(weight(32), rows=256)
    assert _qwen4_exp_production_moe_prefill_enabled(weight(47), rows=16)
    assert not _qwen4_exp_production_moe_prefill_enabled(weight(47), rows=15)

    strict = _resolve(ExecutionProfile.STRICT)
    assert strict.binder is not None
    strict.binder(SimpleNamespace(), strict)
    assert os.environ[PRODUCTION_MOE_PREFILL_ENV] == "0"
    assert not _qwen4_exp_production_moe_prefill_enabled(weight(47), rows=256)


def test_qwen4_exp_profiles_cover_both_registered_quant_names() -> None:
    register_qwen4_exp_gfx1151_profiles()
    for quant in QWEN4_EXP_QUANTS:
        for profile in (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION):
            resolved = _resolve(profile, quant=quant)
            assert resolved.manifest["quant"] == quant
