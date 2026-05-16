from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.fused import (
    plan_paro_silu_build,
    register_paro_silu_kernels,
    silu_mul_dual_out_bf16,
    silu_mul_dual_out_fp16,
    silu_mul_dual_rotate_out_bf16,
    silu_mul_dual_rotate_out_fp16,
    silu_mul_pair_rotate_out_bf16,
    silu_mul_pair_rotate_out_fp16,
    silu_mul_separate_out_bf16,
    silu_mul_separate_out_fp16,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_paro_silu_registers_bf16_fp16_and_w4_paro_variants() -> None:
    register_paro_silu_kernels()

    for quant in ("bf16", "w4_paro"):
        assert (
            resolve(backend="hip_gfx1100", layer="silu_mul_dual", quant=quant, variant="out")
            is silu_mul_dual_out_bf16
        )
        assert (
            resolve(backend="hip_gfx1100", layer="silu_mul_dual", quant=quant, variant="out_fp16")
            is silu_mul_dual_out_fp16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="silu_mul_dual_rotate",
                quant=quant,
                variant="out",
            )
            is silu_mul_dual_rotate_out_bf16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="silu_mul_dual_rotate",
                quant=quant,
                variant="out_fp16",
            )
            is silu_mul_dual_rotate_out_fp16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="silu_mul_pair_rotate",
                quant=quant,
                variant="out",
            )
            is silu_mul_pair_rotate_out_bf16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="silu_mul_pair_rotate",
                quant=quant,
                variant="out_fp16",
            )
            is silu_mul_pair_rotate_out_fp16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="silu_mul_separate",
                quant=quant,
                variant="out",
            )
            is silu_mul_separate_out_bf16
        )
        assert (
            resolve(
                backend="hip_gfx1100",
                layer="silu_mul_separate",
                quant=quant,
                variant="out_fp16",
            )
            is silu_mul_separate_out_fp16
        )
    assert resolve(backend="hip_gfx1100", layer="silu_mul_dual", quant="fp16", variant="out") is silu_mul_dual_out_fp16
    assert (
        resolve(backend="hip_gfx1100", layer="silu_mul_separate", quant="fp16", variant="out")
        is silu_mul_separate_out_fp16
    )


def test_paro_silu_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_paro_silu_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc paro silu test version",
    )

    assert artifact.family == "paro_silu"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "paro_silu.so"
    assert artifact.compiler_version == "hipcc paro silu test version"
    assert any(str(path).endswith("paro_silu.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_paro_silu_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="rows must be positive"):
        silu_mul_dual_out_bf16(0, 0, 0, 8)
    with pytest.raises(ValueError, match="threads must be one of"):
        silu_mul_dual_out_bf16(0, 0, 1, 8, threads=32)
    with pytest.raises(ValueError, match="rows must be positive"):
        silu_mul_dual_out_fp16(0, 0, 0, 8)
    with pytest.raises(ValueError, match="group_size must be even"):
        silu_mul_dual_rotate_out_bf16(0, 0, 0, 0, 0, 1, 8, 3, 1)
    with pytest.raises(ValueError, match="features must be divisible"):
        silu_mul_dual_rotate_out_bf16(0, 0, 0, 0, 0, 1, 10, 8, 1)
    with pytest.raises(ValueError, match="krot must be positive"):
        silu_mul_pair_rotate_out_bf16(0, 0, 0, 0, 0, 0, 1, 8, 8, 0)
    with pytest.raises(ValueError, match="krot must be positive"):
        silu_mul_pair_rotate_out_fp16(0, 0, 0, 0, 0, 0, 1, 8, 8, 0)
    with pytest.raises(ValueError, match="rows must be positive"):
        silu_mul_separate_out_bf16(0, 0, 0, 0, 8)
    with pytest.raises(ValueError, match="threads must be one of"):
        silu_mul_separate_out_bf16(0, 0, 0, 1, 8, threads=32)
    with pytest.raises(ValueError, match="features must be positive"):
        silu_mul_separate_out_fp16(0, 0, 0, 1, 0)
