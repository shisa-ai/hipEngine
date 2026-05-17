from __future__ import annotations

import sys

from hipengine.models import resolve_model
from hipengine.quant import resolve_quant


def test_importing_hipengine_does_not_import_torch() -> None:
    had_torch = "torch" in sys.modules

    import hipengine  # noqa: F401

    if not had_torch:
        assert "torch" not in sys.modules


def test_builtin_toy_model_plugin_is_registered() -> None:
    plugin = resolve_model("HipEngineToyForCausalLM")

    assert plugin.name == "toy_one_layer"
    assert plugin.layer_sequence()[1:4] == ("rmsnorm", "rotate", "qkv_proj")


def test_builtin_qwen35_paro_model_plugin_is_registered() -> None:
    plugin = resolve_model("Qwen3_5MoeForConditionalGeneration")

    assert plugin.name == "qwen3_5_moe_paro"
    assert plugin.default_quant == "w4_paro"
    assert "selected_dual_pack8_gemv" in plugin.layer_sequence()
    assert plugin.decode_layer_sequence(attention_kind="linear_attention")[:2] == (
        "rmsnorm",
        "linear_attention_qkvz_proj",
    )
    assert "model.layers.{layer}.mlp.experts.{expert}.{proj}.qweight" in plugin.weight_name_templates


def test_builtin_fp16_quant_plugin_is_registered() -> None:
    plugin = resolve_quant("fp16")

    assert plugin.weight_storage == "fp16"
    assert plugin.compute_dtype == "fp16"
    assert plugin.kernel_family == "fp16"


def test_builtin_bf16_quant_plugin_is_registered() -> None:
    plugin = resolve_quant("bf16")

    assert plugin.weight_storage == "bf16"
    assert plugin.compute_dtype == "bf16"
    assert plugin.kernel_family == "bf16"


def test_builtin_w4_paro_quant_plugin_is_registered() -> None:
    plugin = resolve_quant("w4_paro")

    assert plugin.weight_storage == "uint4_pack8_awq"
    assert plugin.activation_preprocess == "bf16_pairwise_rotation"
    assert plugin.compute_dtype == "bf16"
    assert plugin.scale_granularity == "group128_per_output_channel"
    assert plugin.calibration_artifact == "paroquant_theta_pairs_scales"
    assert plugin.kernel_family == "paro_awq_pack8"


def test_builtin_gguf_q4_k_quant_plugin_is_registered() -> None:
    plugin = resolve_quant("gguf_q4_k")

    assert plugin.weight_storage == "gguf_block_q4_k"
    assert plugin.activation_preprocess == "none"
    assert plugin.compute_dtype == "fp32_accum"
    assert plugin.scale_granularity == "block256_subblock32_scale_min"
    assert plugin.calibration_artifact == "gguf"
    assert plugin.kernel_family == "gguf_q4_k_gemv"


def test_builtin_mixed_gguf_quant_plugins_are_registered() -> None:
    expected = {
        "gguf_q8_0": ("gguf_block_q8_0", "block32_scale", "gguf_k_gemv"),
        "gguf_q4_1": ("gguf_block_q4_1", "block32_scale_min", "gguf_dense_bf16_fallback"),
        "gguf_q5_k": ("gguf_block_q5_k", "block256_subblock32_scale_min", "gguf_k_gemv"),
        "gguf_q6_k": ("gguf_block_q6_k", "block256_subblock16_scale", "gguf_k_gemv"),
        "gguf_iq4_xs": ("gguf_iq4_xs", "block256_iq4_xs", "gguf_dense_bf16_fallback"),
    }
    for name, (storage, granularity, kernel_family) in expected.items():
        plugin = resolve_quant(name)

        assert plugin.weight_storage == storage
        assert plugin.activation_preprocess == "none"
        assert plugin.compute_dtype == "fp32_accum"
        assert plugin.scale_granularity == granularity
        assert plugin.calibration_artifact == "gguf"
        assert plugin.kernel_family == kernel_family
