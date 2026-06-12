from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.linear import (
    dense_dual_gemv_out_bf16,
    dense_dual_gemv_out_bf16_wmma,
    dense_dual_gemv_out_fp16,
    dense_dual_gemv_out_fp16_wmma,
    dense_dual_gemv_separate_out_bf16,
    dense_dual_gemv_separate_out_fp16,
    dense_gemv_out_bf16,
    dense_gemv_out_bf16_wmma,
    dense_gemv_out_fp16,
    dense_gemv_out_fp16_wmma,
    plan_dense_gemv_build,
    register_dense_gemv_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_dense_gemv_registers_bf16_fp16_and_w4_paro_variants() -> None:
    register_dense_gemv_kernels()

    assert (
        resolve(backend="hip_gfx1100", layer="dense_gemv", quant="bf16", variant="out")
        is dense_gemv_out_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_dual_gemv", quant="bf16", variant="out")
        is dense_dual_gemv_out_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_gemv", quant="w4_paro", variant="out")
        is dense_gemv_out_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_gemv", quant="w4_paro", variant="out_fp16")
        is dense_gemv_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_dual_gemv", quant="w4_paro", variant="out_fp16")
        is dense_dual_gemv_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_dual_gemv", quant="w4_paro", variant="separate_out")
        is dense_dual_gemv_separate_out_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_dual_gemv", quant="w4_paro", variant="separate_out_fp16")
        is dense_dual_gemv_separate_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_gemv", quant="fp16", variant="out")
        is dense_gemv_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_dual_gemv", quant="fp16", variant="separate_out")
        is dense_dual_gemv_separate_out_fp16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_gemv", quant="bf16", variant="out_wmma")
        is dense_gemv_out_bf16_wmma
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_dual_gemv", quant="bf16", variant="out_wmma")
        is dense_dual_gemv_out_bf16_wmma
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_gemv", quant="w4_paro", variant="out_fp16_wmma")
        is dense_gemv_out_fp16_wmma
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_dual_gemv", quant="fp16", variant="out_wmma")
        is dense_dual_gemv_out_fp16_wmma
    )


def test_dense_gemv_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_dense_gemv_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc dense gemv test version",
    )

    assert artifact.family == "dense_gemv"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "dense_gemv.so"
    assert artifact.compiler_version == "hipcc dense gemv test version"
    assert any(str(path).endswith("dense_gemv.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_dense_gemv_wrapper_validates_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="rows must be positive"):
        dense_gemv_out_bf16(0, 0, 0, 0, 16, 8)
    with pytest.raises(ValueError, match="in_features must be positive"):
        dense_gemv_out_bf16(0, 0, 0, 1, 0, 8)
    with pytest.raises(ValueError, match="out_features must be positive"):
        dense_gemv_out_bf16(0, 0, 0, 1, 16, 0)
    with pytest.raises(ValueError, match="threads must be one of"):
        dense_gemv_out_bf16(0, 0, 0, 1, 16, 8, threads=32)
    with pytest.raises(ValueError, match="rows must be positive"):
        dense_gemv_out_fp16(0, 0, 0, 0, 16, 8)
    with pytest.raises(ValueError, match="out_features must be positive"):
        dense_dual_gemv_out_fp16(0, 0, 0, 0, 1, 16, 8, 0)
    with pytest.raises(ValueError, match="out_features must be positive"):
        dense_dual_gemv_separate_out_fp16(0, 0, 0, 0, 0, 1, 16, 8, 0)
    with pytest.raises(ValueError, match="multiple of 16"):
        dense_gemv_out_fp16_wmma(0, 0, 0, 1, 17, 8)
