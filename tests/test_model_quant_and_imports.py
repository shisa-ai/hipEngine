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
