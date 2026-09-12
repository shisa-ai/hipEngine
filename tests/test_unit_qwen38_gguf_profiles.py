from __future__ import annotations

import os

import pytest

from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.execution_profiles import (
    ExecutionProfile,
    clear_runtime_profile_registry_for_tests,
    resolve_runtime_profile,
)
from hipengine.generation.qwen38_gguf_profiles import (
    FP16_RECURRENT_STATE_ENV,
    PRODUCTION_Q4_VERIFIER_ROWTILE_ENV,
    QWEN38_GGUF_BACKEND,
    QWEN38_GGUF_MODEL,
    QWEN38_GGUF_QUANT,
    VERIFY_CAPTURE_PREFILL_GDN_ENV,
    qwen38_gguf_gfx1151_plans_registered,
    register_qwen38_gguf_gfx1151_profiles,
)


@pytest.fixture(autouse=True)
def _isolate_profile_registry(monkeypatch: pytest.MonkeyPatch):
    clear_runtime_profile_registry_for_tests()
    for name in (
        FP16_RECURRENT_STATE_ENV,
        PRODUCTION_Q4_VERIFIER_ROWTILE_ENV,
        VERIFY_CAPTURE_PREFILL_GDN_ENV,
        "HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    clear_runtime_profile_registry_for_tests()


def _resolve(profile: ExecutionProfile):
    return resolve_runtime_profile(
        model=QWEN38_GGUF_MODEL,
        backend=QWEN38_GGUF_BACKEND,
        quant=QWEN38_GGUF_QUANT,
        profile=profile,
    )


def _selection_map(resolved):
    return {
        (row["layer"], row["scope"]): row
        for row in resolved.manifest["selections"]
    }


def test_qwen38_strict_profile_resolves_and_disables_fp16_state() -> None:
    register_gfx1151_kernels(replace=True)
    register_qwen38_gguf_gfx1151_profiles()
    assert qwen38_gguf_gfx1151_plans_registered()

    resolved = _resolve(ExecutionProfile.STRICT)
    generator = object.__new__(type("Generator", (), {}))
    assert resolved.binder is not None
    resolved.binder(generator, resolved)

    assert os.environ[FP16_RECURRENT_STATE_ENV] == "0"
    assert os.environ[VERIFY_CAPTURE_PREFILL_GDN_ENV] == "1"
    assert os.environ[PRODUCTION_Q4_VERIFIER_ROWTILE_ENV] == "0"
    assert resolved.profile is ExecutionProfile.STRICT
    assert resolved.manifest_sha256 == resolved.strict_manifest_sha256
    assert resolved.manifest["graph_policy"] == "specdec2_eager_c1"
    assert resolved.manifest["kv_policy"] == "paged_bf16"
    selections = _selection_map(resolved)
    assert selections[
        ("gdn_chain_recurrent_rmsnorm_gate", "specdec2_mtp2_target_state_rows")
    ]["selected_variant"] == "bf16_c1_exact_state_rows_tloop"
    for budget, physical_rows in ((1, 6), (2, 9), (3, 12)):
        assert selections[
            ("linear", f"specdec2_mtp2_c3_k{budget}_r{physical_rows}_q4_single")
        ]["selected_variant"] == "t16_wmma_prefill_smallm_bf16_bf16_out"
        assert selections[
            (
                "linear_pair_silu",
                f"specdec2_mtp2_c3_k{budget}_r{physical_rows}_q4_gate_up",
            )
        ]["selected_variant"] == "dense_dual_wmma_prefill_bf16_bf16_out"


def test_qwen38_production_profile_resolves_fp16_state_with_strict_fallbacks() -> None:
    register_gfx1151_kernels(replace=True)
    register_qwen38_gguf_gfx1151_profiles()
    resolved = _resolve(ExecutionProfile.PRODUCTION)
    generator = object.__new__(type("Generator", (), {}))
    assert resolved.binder is not None
    resolved.binder(generator, resolved)

    assert resolved.profile is ExecutionProfile.PRODUCTION
    assert not resolved.fell_back_to_strict
    assert resolved.manifest_sha256 != resolved.strict_manifest_sha256
    assert os.environ[FP16_RECURRENT_STATE_ENV] == "1"
    assert os.environ[VERIFY_CAPTURE_PREFILL_GDN_ENV] == "1"
    assert os.environ[PRODUCTION_Q4_VERIFIER_ROWTILE_ENV] == "1"
    selections = _selection_map(resolved)
    expected = {
        ("linear", "specdec2_mtp2_c2_k3_r8_q4_single"): (
            "dense_rowtile_bf16_bf16_out",
            "t16_wmma_prefill_smallm_bf16_bf16_out",
        ),
        ("linear_pair_silu", "specdec2_mtp2_c2_k3_r8_q4_gate_up"): (
            "dense_dual_rowtile_bf16_bf16_out",
            "dense_dual_wmma_prefill_bf16_bf16_out",
        ),
        **{
            ("linear", f"specdec2_mtp2_c3_k{budget}_r{rows}_q4_single"): (
                "dense_rowtile_bf16_bf16_out",
                "t16_wmma_prefill_smallm_bf16_bf16_out",
            )
            for budget, rows in ((1, 6), (2, 9), (3, 12))
        },
        **{
            (
                "linear_pair_silu",
                f"specdec2_mtp2_c3_k{budget}_r{rows}_q4_gate_up",
            ): (
                "dense_dual_rowtile_bf16_bf16_out",
                "dense_dual_wmma_prefill_bf16_bf16_out",
            )
            for budget, rows in ((1, 6), (2, 9), (3, 12))
        },
        ("gdn_chain_recurrent_rmsnorm_gate", "specdec2_mtp2_target_state_rows"): (
            "bf16_c1_exact_state_rows_tloop_fp16state",
            "bf16_c1_exact_state_rows_tloop",
        ),
    }
    for key, variants in expected.items():
        assert (
            selections[key]["selected_variant"],
            selections[key]["strict_fallback_variant"],
        ) == variants
