from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import resolve_runtime_profile
from hipengine.generation import qwen35_dense_gfx1100_profiles as _profiles
from hipengine.runtime import qwen35_gguf_runner as _qwen35_gguf_runner  # noqa: F401
from hipengine.generation.qwen35_dense_gfx1100_profiles import (
    FP16_RECURRENT_STATE_ENV,
    Q4_FUSED_R28_ENV,
    Q6_DP4A_GROUPED_ENV,
    QWEN36_DENSE_GGUF_BACKEND,
    QWEN36_DENSE_GGUF_MODEL,
    QWEN36_DENSE_GGUF_QUANT,
    VERIFY_CAPTURE_PREFILL_GDN_ENV,
    VERIFY_F32_POST_NORM_ENV,
    VERIFY_F32_RESIDUAL_ENV,
    qwen36_dense_gguf_gfx1100_strict_registered,
    register_qwen36_dense_gguf_gfx1100_profiles,
)
from hipengine.generation.qwen35_moe_gfx1100_profiles import (
    QWEN36_MOE_GGUF_MODEL,
    qwen36_moe_gguf_gfx1100_strict_registered,
    register_qwen36_moe_gguf_gfx1100_profiles,
)


_PROFILE_MATRIX = (
    ("dense_strict", QWEN36_DENSE_GGUF_MODEL, "strict", (False, False, False)),
    (
        "dense_production",
        QWEN36_DENSE_GGUF_MODEL,
        "production",
        (False, False, True),
    ),
    (
        "moe_production",
        QWEN36_MOE_GGUF_MODEL,
        "production",
        (True, True, False),
    ),
)
_PROFILE_EXPECTED = {
    name: expected for name, _model, _profile, expected in _PROFILE_MATRIX
}
_R28_ACCEPTED = (
    ("1", True),
    ("true", True),
    (" YES ", True),
    ("on", True),
    ("0", False),
    ("false", False),
    (" No ", False),
    ("off", False),
)
_R28_REJECTED = ("2", "maybe", "", "truthy", "1.0")


@pytest.fixture(autouse=True)
def _isolate_profile_environment(monkeypatch):
    for name in (
        FP16_RECURRENT_STATE_ENV,
        VERIFY_CAPTURE_PREFILL_GDN_ENV,
        VERIFY_F32_RESIDUAL_ENV,
        VERIFY_F32_POST_NORM_ENV,
        Q4_FUSED_R28_ENV,
        Q6_DP4A_GROUPED_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    for name in (
        FP16_RECURRENT_STATE_ENV,
        VERIFY_CAPTURE_PREFILL_GDN_ENV,
        VERIFY_F32_RESIDUAL_ENV,
        VERIFY_F32_POST_NORM_ENV,
        Q4_FUSED_R28_ENV,
        Q6_DP4A_GROUPED_ENV,
    ):
        __import__("os").environ.pop(name, None)


def test_qwen36_dense_gfx1100_strict_profile_resolves_exact_c1_route(
    monkeypatch,
) -> None:
    monkeypatch.setenv(FP16_RECURRENT_STATE_ENV, "1")
    register_qwen36_dense_gguf_gfx1100_profiles()
    assert qwen36_dense_gguf_gfx1100_strict_registered()

    resolved = resolve_runtime_profile(
        model=QWEN36_DENSE_GGUF_MODEL,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile="strict",
    )
    generator = resolved.construct_generator(lambda: SimpleNamespace())

    assert generator.execution_profile == "strict"
    assert resolved.fell_back_to_strict is False
    assert resolved.manifest["kv_policy"] == "paged_bf16"
    assert resolved.manifest["graph_policy"] == "specdec2_eager_c1"
    selection = resolved.manifest["selections"][0]
    assert selection["layer"] == "linear_attn_chain_conv_decode"
    assert selection["scope"] == "specdec2_mtp2_c1"
    assert selection["selected_variant"] == "bf16_c1_exact_state_rows_tloop"
    assert selection["strict_fallback_variant"] == selection["selected_variant"]
    assert __import__("os").environ[FP16_RECURRENT_STATE_ENV] == "1"
    assert generator.target_fp16_recurrent_state is False
    assert generator.target_verifier_prefill_gdn_capture is True
    assert VERIFY_CAPTURE_PREFILL_GDN_ENV not in __import__("os").environ
    assert generator.target_verify_f32_residual is False
    assert generator.target_verify_f32_post_norm is False
    assert generator.target_q4_t16_dual_silu_r28 is False
    assert VERIFY_F32_RESIDUAL_ENV not in __import__("os").environ
    assert VERIFY_F32_POST_NORM_ENV not in __import__("os").environ
    assert generator.use_q6_dp4a_grouped is False
    assert Q6_DP4A_GROUPED_ENV not in __import__("os").environ


def test_qwen36_moe_gfx1100_strict_fallback_and_production_candidate(
    monkeypatch,
) -> None:
    monkeypatch.setenv(FP16_RECURRENT_STATE_ENV, "1")
    monkeypatch.setenv(VERIFY_F32_RESIDUAL_ENV, "0")
    monkeypatch.setenv(VERIFY_F32_POST_NORM_ENV, "0")
    register_qwen36_moe_gguf_gfx1100_profiles()
    assert qwen36_moe_gguf_gfx1100_strict_registered()

    strict = resolve_runtime_profile(
        model=QWEN36_MOE_GGUF_MODEL,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile="strict",
    )
    production = resolve_runtime_profile(
        model=QWEN36_MOE_GGUF_MODEL,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile="production",
    )

    assert strict.fell_back_to_strict is False
    assert production.fell_back_to_strict is False
    assert production.source_profile.value == "production"
    strict_generator = strict.construct_generator(lambda: SimpleNamespace())
    generator = production.construct_generator(lambda: SimpleNamespace())
    assert production.manifest["graph_policy"] == (
        "specdec2_moe_bulk_f32_k2_strict_k1_candidate"
    )
    assert generator.execution_profile == "production"
    assert strict_generator.target_fp16_recurrent_state is False
    assert generator.target_fp16_recurrent_state is False
    assert __import__("os").environ[FP16_RECURRENT_STATE_ENV] == "1"
    assert generator.target_verify_f32_residual is True
    assert generator.target_verify_f32_post_norm is True
    assert generator.target_q4_t16_dual_silu_r28 is False
    assert __import__("os").environ[VERIFY_F32_RESIDUAL_ENV] == "0"
    assert __import__("os").environ[VERIFY_F32_POST_NORM_ENV] == "0"
    assert strict.manifest["selections"][0]["strict_fallback_variant"] == (
        "bf16_c1_exact_state_rows_tloop"
    )


def test_qwen36_dense_gfx1100_production_resolves_periodic_strict_r28_and_c8_q6(
    monkeypatch,
) -> None:
    monkeypatch.delenv(Q4_FUSED_R28_ENV, raising=False)
    monkeypatch.delenv(Q6_DP4A_GROUPED_ENV, raising=False)
    register_qwen36_dense_gguf_gfx1100_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN36_DENSE_GGUF_MODEL,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile="production",
    )

    assert resolved.fell_back_to_strict is False
    assert resolved.source_profile.value == "production"
    assert resolved.manifest["execution_profile"] == "production"
    assert resolved.manifest["graph_policy"] == (
        "specdec2_eager_c1_exact_qwen38_c8_q6_dp4a"
    )
    generator = resolved.construct_generator(lambda: SimpleNamespace())
    selections = resolved.manifest["selections"]
    assert selections[0]["selected_variant"] == "bf16_c1_exact_state_rows_tloop"
    r28 = next(row for row in selections if "c7_k3_r28" in row["scope"])
    assert r28["selected_variant"] == (
        "dense_dual_wmma_prefill_row32_bf16_bf16_out"
    )
    assert r28["strict_fallback_variant"] == (
        "dense_dual_rowtile_bf16_bf16_out"
    )
    assert "periodic_strict_mod8_0" in r28["scope"]
    assert generator.target_verify_f32_residual is False
    assert generator.target_verify_f32_post_norm is False
    assert generator.target_q4_t16_dual_silu_r28 is True
    assert Q4_FUSED_R28_ENV not in __import__("os").environ
    assert generator.use_q6_dp4a_grouped is True
    assert Q6_DP4A_GROUPED_ENV not in __import__("os").environ


def test_profile_owned_strict_values_do_not_mask_dense_production_defaults() -> None:
    register_qwen36_dense_gguf_gfx1100_profiles()
    strict = resolve_runtime_profile(
        model=QWEN36_DENSE_GGUF_MODEL,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile="strict",
    )
    production = resolve_runtime_profile(
        model=QWEN36_DENSE_GGUF_MODEL,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile="production",
    )

    strict_generator = strict.construct_generator(lambda: SimpleNamespace())
    assert Q4_FUSED_R28_ENV not in __import__("os").environ
    assert strict_generator.target_q4_t16_dual_silu_r28 is False
    assert strict_generator.use_q6_dp4a_grouped is False
    production_generator = production.construct_generator(lambda: SimpleNamespace())
    assert Q4_FUSED_R28_ENV not in __import__("os").environ
    assert production_generator.target_q4_t16_dual_silu_r28 is True
    assert production_generator.use_q6_dp4a_grouped is True
    strict.construct_generator(lambda: SimpleNamespace())
    assert production_generator.use_q6_dp4a_grouped is True
    assert strict_generator.use_q6_dp4a_grouped is False
    assert Q6_DP4A_GROUPED_ENV not in __import__("os").environ


@pytest.mark.parametrize("order", ("forward", "reverse"))
def test_actual_profile_binder_matrix_preserves_f32_and_r28_owners_in_both_orders(
    monkeypatch,
    order,
) -> None:
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator

    monkeypatch.setenv(VERIFY_F32_RESIDUAL_ENV, "1")
    monkeypatch.setenv(VERIFY_F32_POST_NORM_ENV, "1")
    monkeypatch.setenv(Q4_FUSED_R28_ENV, "1")
    register_qwen36_dense_gguf_gfx1100_profiles()
    register_qwen36_moe_gguf_gfx1100_profiles()
    specs = (
        _PROFILE_MATRIX
        if order == "forward"
        else tuple(reversed(_PROFILE_MATRIX))
    )
    before = dict(os.environ)
    built = {}
    for name, model, profile, expected in specs:
        resolved = resolve_runtime_profile(
            model=model,
            backend=QWEN36_DENSE_GGUF_BACKEND,
            quant=QWEN36_DENSE_GGUF_QUANT,
            profile=profile,
        )
        generator = resolved.construct_generator(
            lambda: object.__new__(Qwen35GGUFBringupGenerator)
        )
        built[name] = generator
        for other_name, other_generator in built.items():
            other_expected = _PROFILE_EXPECTED[other_name]
            assert (
                other_generator.target_verify_f32_residual,
                other_generator.target_verify_f32_post_norm,
                other_generator.target_q4_t16_dual_silu_r28,
            ) == other_expected
        assert generator.execution_profile == profile
        assert generator._target_verify_f32_session_kwargs() == {
            "target_verify_f32_residual": expected[0],
            "target_verify_f32_post_norm": expected[1],
        }
        assert generator._target_q4_t16_dual_silu_r28_session_kwargs() == {
            "target_q4_t16_dual_silu_r28": expected[2]
        }
        assert dict(os.environ) == before
    assert dict(os.environ) == before


@pytest.mark.parametrize("raw,expected", _R28_ACCEPTED)
def test_r28_profile_construction_uses_original_selector_boolean_syntax(
    monkeypatch,
    raw,
    expected,
) -> None:
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator

    monkeypatch.setenv(Q4_FUSED_R28_ENV, raw)
    register_qwen36_dense_gguf_gfx1100_profiles()
    before = dict(os.environ)
    resolved = resolve_runtime_profile(
        model=QWEN36_DENSE_GGUF_MODEL,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile="production",
    )
    generator = resolved.construct_generator(
        lambda: object.__new__(Qwen35GGUFBringupGenerator)
    )
    assert generator.target_q4_t16_dual_silu_r28 is expected
    assert dict(os.environ) == before


@pytest.mark.parametrize("raw", _R28_REJECTED)
def test_r28_profile_construction_rejects_non_selector_boolean_strings(
    monkeypatch,
    raw,
) -> None:
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator

    monkeypatch.setenv(Q4_FUSED_R28_ENV, raw)
    register_qwen36_dense_gguf_gfx1100_profiles()
    before = dict(os.environ)
    resolved = resolve_runtime_profile(
        model=QWEN36_DENSE_GGUF_MODEL,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile="production",
    )
    with pytest.raises(ValueError) as exc:
        resolved.construct_generator(
            lambda: object.__new__(Qwen35GGUFBringupGenerator)
        )
    assert str(exc.value) == f"{Q4_FUSED_R28_ENV} must be a boolean value"
    assert dict(os.environ) == before


@pytest.mark.parametrize("value,expected", (("0", False), ("1", True), ("", False),
                                            (" off ", False), ("yes", True)))
def test_q6_profile_snapshots_override_without_mutating_environment(monkeypatch, value, expected):
    import os

    monkeypatch.setenv(Q6_DP4A_GROUPED_ENV, value)
    register_qwen36_dense_gguf_gfx1100_profiles()
    def profile(name):
        return resolve_runtime_profile(
            model=QWEN36_DENSE_GGUF_MODEL, backend=QWEN36_DENSE_GGUF_BACKEND,
            quant=QWEN36_DENSE_GGUF_QUANT, profile=name,
        ).construct_generator(lambda: SimpleNamespace())

    production = profile("production")
    strict = profile("strict")
    assert production.use_q6_dp4a_grouped is expected
    assert strict.use_q6_dp4a_grouped is False
    assert os.environ[Q6_DP4A_GROUPED_ENV] == value
    monkeypatch.setenv(Q6_DP4A_GROUPED_ENV, "1" if not expected else "0")
    assert production.use_q6_dp4a_grouped is expected


def test_qwen36_dense_gfx1100_production_honors_explicit_rollbacks(
    monkeypatch,
) -> None:
    monkeypatch.setenv(Q4_FUSED_R28_ENV, "0")
    monkeypatch.setenv(Q6_DP4A_GROUPED_ENV, "0")
    register_qwen36_dense_gguf_gfx1100_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN36_DENSE_GGUF_MODEL,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile="production",
    )
    generator = resolved.construct_generator(lambda: SimpleNamespace())
    assert __import__("os").environ[Q4_FUSED_R28_ENV] == "0"
    assert generator.target_q4_t16_dual_silu_r28 is False
    assert generator.use_q6_dp4a_grouped is False
    assert __import__("os").environ[Q6_DP4A_GROUPED_ENV] == "0"
