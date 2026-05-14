from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.norm import (
    paro_add_rmsnorm_out_bf16,
    paro_add_rmsnorm_out_fp16,
    paro_rmsnorm_out_bf16,
    paro_rmsnorm_out_fp16,
    plan_qwen35_rmsnorm_build,
    qwen35_add_rmsnorm_bf16,
    qwen35_add_rmsnorm_f32_bf16,
    qwen35_head_rmsnorm_f32_bf16,
    qwen35_rmsnorm_bf16,
    register_qwen35_rmsnorm_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_qwen35_rmsnorm_registers_bf16_family() -> None:
    register_qwen35_rmsnorm_kernels()

    assert resolve(backend="hip_gfx1100", layer="rmsnorm", quant="bf16") is qwen35_rmsnorm_bf16
    assert (
        resolve(backend="hip_gfx1100", layer="add_rmsnorm", quant="bf16")
        is qwen35_add_rmsnorm_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="add_rmsnorm_f32", quant="bf16")
        is qwen35_add_rmsnorm_f32_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="head_rmsnorm", quant="bf16")
        is qwen35_head_rmsnorm_f32_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="rmsnorm", quant="bf16", variant="paro_out")
        is paro_rmsnorm_out_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="rmsnorm", quant="w4_paro", variant="paro_out")
        is paro_rmsnorm_out_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="add_rmsnorm",
            quant="w4_paro",
            variant="paro_out",
        )
        is paro_add_rmsnorm_out_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="rmsnorm",
            quant="w4_paro",
            variant="paro_out_fp16",
        )
        is paro_rmsnorm_out_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="add_rmsnorm",
            quant="w4_paro",
            variant="paro_out_fp16",
        )
        is paro_add_rmsnorm_out_fp16
    )


def test_qwen35_rmsnorm_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_qwen35_rmsnorm_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc rmsnorm test version",
    )

    assert artifact.family == "qwen35_rmsnorm"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "qwen35_rmsnorm.so"
    assert artifact.compiler_version == "hipcc rmsnorm test version"
    assert any(str(path).endswith("rmsnorm.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_qwen35_rmsnorm_wrapper_validates_shape_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="rows must be positive"):
        qwen35_rmsnorm_bf16(0, 0, 0, 0, 16)
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        qwen35_rmsnorm_bf16(0, 0, 0, 1, 0)
    with pytest.raises(ValueError, match="rows must be positive"):
        paro_rmsnorm_out_bf16(0, 0, 0, 0, 16)
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        paro_add_rmsnorm_out_bf16(0, 0, 0, 0, 0, 1, 0)
    with pytest.raises(ValueError, match="rows must be positive"):
        paro_rmsnorm_out_fp16(0, 0, 0, 0, 16)
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        paro_add_rmsnorm_out_fp16(0, 0, 0, 0, 0, 1, 0)
