from __future__ import annotations

from types import SimpleNamespace

from hipengine.execution_profiles import resolve_runtime_profile
from hipengine.generation.qwen36_gguf_gfx1100_profiles import (
    FP16_RECURRENT_STATE_ENV,
    QWEN36_DENSE_GGUF_BACKEND,
    QWEN36_DENSE_GGUF_MODEL,
    QWEN36_DENSE_GGUF_QUANT,
    QWEN36_MOE_GGUF_MODEL,
    VERIFY_CAPTURE_PREFILL_GDN_ENV,
    qwen36_dense_gguf_gfx1100_strict_registered,
    qwen36_moe_gguf_gfx1100_strict_registered,
    register_qwen36_dense_gguf_gfx1100_profiles,
    register_qwen36_moe_gguf_gfx1100_profiles,
)


def test_qwen36_dense_gfx1100_strict_profile_resolves_exact_c1_route(
    monkeypatch,
) -> None:
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
    assert __import__("os").environ[FP16_RECURRENT_STATE_ENV] == "0"
    assert __import__("os").environ[VERIFY_CAPTURE_PREFILL_GDN_ENV] == "1"


def test_qwen36_moe_gfx1100_strict_fallback_and_production_candidate() -> None:
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
    assert production.manifest["graph_policy"] == (
        "specdec2_moe_native_complete_c1_candidate"
    )
    assert strict.manifest["selections"][0]["strict_fallback_variant"] == (
        "bf16_c1_exact_state_rows_tloop"
    )


def test_qwen36_dense_gfx1100_production_fails_closed_to_strict() -> None:
    register_qwen36_dense_gguf_gfx1100_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN36_DENSE_GGUF_MODEL,
        backend=QWEN36_DENSE_GGUF_BACKEND,
        quant=QWEN36_DENSE_GGUF_QUANT,
        profile="production",
    )

    assert resolved.fell_back_to_strict is True
    assert resolved.source_profile.value == "strict"
    assert resolved.manifest["execution_profile"] == "production"
    selection = resolved.manifest["selections"][0]
    assert selection["selected_variant"] == "bf16_c1_exact_state_rows_tloop"
