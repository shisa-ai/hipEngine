from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.linear_attn import (
    plan_qwen35_linear_attn_conv_build,
    qwen35_linear_attn_conv_decode_bf16,
    qwen35_linear_attn_conv_decode_f32,
    qwen35_linear_attn_conv_prefill_f32,
    register_qwen35_linear_attn_conv_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_qwen35_linear_attn_conv_registers_decode_and_prefill_variants() -> None:
    register_qwen35_linear_attn_conv_kernels()

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_attn_conv_decode",
            quant="w4_paro",
            variant="f32",
        )
        is qwen35_linear_attn_conv_decode_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_attn_conv_decode",
            quant="w4_paro",
            variant="bf16",
        )
        is qwen35_linear_attn_conv_decode_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_attn_conv_prefill",
            quant="w4_paro",
            variant="f32",
        )
        is qwen35_linear_attn_conv_prefill_f32
    )


def test_qwen35_linear_attn_conv_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_qwen35_linear_attn_conv_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc qwen35 linear attn conv test version",
    )

    assert artifact.family == "qwen35_linear_attn_conv"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "qwen35_linear_attn_conv.so"
    assert artifact.compiler_version == "hipcc qwen35 linear attn conv test version"
    assert any(str(path).endswith("conv.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_qwen35_linear_attn_conv_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="channels must be positive"):
        qwen35_linear_attn_conv_decode_f32(0, 0, 0, 0, 0, 4)
    with pytest.raises(ValueError, match="kernel_size must be positive"):
        qwen35_linear_attn_conv_decode_bf16(0, 0, 0, 0, 4, 0)
    with pytest.raises(ValueError, match="tokens >= kernel_size"):
        qwen35_linear_attn_conv_prefill_f32(0, 0, 0, 0, 2, 4, 4)
