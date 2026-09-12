from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.quant import (
    plan_w8a16_linear_build,
    register_w8a16_linear_kernels,
    w8a16_linear_bf16_f32_out,
    w8a16_linear_bf16_lowp_out,
    w8a16_linear_f32_f32_out,
    w8a16_linear_fp16_lowp_out,
    w8a16_shared_down_combine_residual_fp16,
    w8a16_shared_gate_sigmoid_fp32,
    w8a16_shared_gate_up_silu_fp16,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_w8a16_linear_registers_w8a16_and_w4_paro_variants() -> None:
    register_w8a16_linear_kernels()

    for quant in ("w8a16", "w4_paro"):
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="w8a16_linear",
                quant=quant,
                variant="bf16_f32_out",
            )
            is w8a16_linear_bf16_f32_out
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="w8a16_linear",
                quant=quant,
                variant="bf16_lowp_out",
            )
            is w8a16_linear_bf16_lowp_out
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="w8a16_linear",
                quant=quant,
                variant="fp16_lowp_out",
            )
            is w8a16_linear_fp16_lowp_out
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="w8a16_linear",
                quant=quant,
                variant="shared_gate_up_silu_fp16",
            )
            is w8a16_shared_gate_up_silu_fp16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="w8a16_linear",
                quant=quant,
                variant="shared_gate_sigmoid_fp32",
            )
            is w8a16_shared_gate_sigmoid_fp32
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="w8a16_linear",
                quant=quant,
                variant="shared_down_combine_residual_fp16",
            )
            is w8a16_shared_down_combine_residual_fp16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="w8a16_linear",
                quant=quant,
                variant="f32_f32_out",
            )
            is w8a16_linear_f32_f32_out
        )


def test_w8a16_linear_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_w8a16_linear_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc w8a16 linear test version",
    )

    assert artifact.family == "w8a16_linear"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "w8a16_linear.so"
    assert artifact.compiler_version == "hipcc w8a16 linear test version"
    assert any(str(path).endswith("w8a16_linear.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_w8a16_linear_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="tokens must be positive"):
        w8a16_linear_bf16_f32_out(0, 0, 0, 0, 0, 16, 8)
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        w8a16_linear_bf16_lowp_out(0, 0, 0, 0, 1, 0, 8)
    with pytest.raises(ValueError, match="out_features must be positive"):
        w8a16_linear_f32_f32_out(0, 0, 0, 0, 1, 16, 0)
    with pytest.raises(ValueError, match="threads must be one of"):
        w8a16_linear_bf16_f32_out(0, 0, 0, 0, 1, 16, 8, threads=32)
    with pytest.raises(ValueError, match="tokens must be positive"):
        w8a16_linear_fp16_lowp_out(0, 0, 0, 0, 0, 16, 8)
    with pytest.raises(ValueError, match="out_features must be positive"):
        w8a16_shared_gate_up_silu_fp16(0, 0, 0, 0, 1, 16, 0)
    with pytest.raises(ValueError, match="gate_stride must be positive"):
        w8a16_shared_gate_sigmoid_fp32(0, 0, 1, 0)
    with pytest.raises(ValueError, match="gate_stride must be positive"):
        w8a16_shared_down_combine_residual_fp16(0, 0, 0, 0, 0, 0, 0, 1, 16, 8, 0)
