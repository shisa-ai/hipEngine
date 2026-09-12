from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.quant import (
    awq_fusedw4_prefill_dual_fp16,
    awq_fusedw4_prefill_fp16,
    awq_fusedw4_prefill_strided_fp16,
    gemv_awq_dual_pack8_strided_bf16,
    gemv_awq_dual_pack8_strided_fp16,
    gemv_awq_dual_pack8_transposed_bf16,
    gemv_awq_dual_pack8_transposed_fp16,
    gemv_awq_pack8_strided_bf16,
    gemv_awq_pack8_strided_fp16,
    gemv_awq_pack8_transposed_bf16,
    gemv_awq_pack8_transposed_fp16,
    gemv_awq_selected_dual_pack8_strided_bf16,
    gemv_awq_selected_dual_pack8_strided_fp16,
    gemv_awq_selected_dual_pack8_strided_rotate_out_bf16,
    gemv_awq_selected_dual_pack8_strided_rotate_out_fp16,
    gemv_awq_selected_dual_pack8_transposed_bf16,
    gemv_awq_selected_dual_pack8_transposed_fp16,
    gemv_awq_selected_pack8_strided_bf16,
    gemv_awq_selected_pack8_strided_fp16,
    gemv_awq_selected_pack8_transposed_bf16,
    gemv_awq_selected_pack8_transposed_fp16,
    plan_paro_awq_gemv_build,
    register_paro_awq_gemv_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_paro_awq_gemv_registers_pack8_variants() -> None:
    register_paro_awq_gemv_kernels()

    assert (
        resolve(backend="hip_gfx1100", layer="pack8_gemv", quant="w4_paro", variant="strided")
        is gemv_awq_pack8_strided_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="pack8_gemv",
            quant="w4_paro",
            variant="transposed",
        )
        is gemv_awq_pack8_transposed_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="dual_pack8_gemv",
            quant="w4_paro",
            variant="strided",
        )
        is gemv_awq_dual_pack8_strided_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="dual_pack8_gemv",
            quant="w4_paro",
            variant="transposed",
        )
        is gemv_awq_dual_pack8_transposed_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="pack8_gemv",
            quant="w4_paro",
            variant="strided_fp16",
        )
        is gemv_awq_pack8_strided_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="pack8_gemv",
            quant="w4_paro",
            variant="transposed_fp16",
        )
        is gemv_awq_pack8_transposed_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="pack8_gemm",
            quant="w4_paro",
            variant="fusedw4_prefill_fp16",
        )
        is awq_fusedw4_prefill_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="pack8_gemm",
            quant="w4_paro",
            variant="fusedw4_prefill_dual_fp16",
        )
        is awq_fusedw4_prefill_dual_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="pack8_gemm",
            quant="w4_paro",
            variant="fusedw4_prefill_strided_fp16",
        )
        is awq_fusedw4_prefill_strided_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="dual_pack8_gemv",
            quant="w4_paro",
            variant="strided_fp16",
        )
        is gemv_awq_dual_pack8_strided_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="dual_pack8_gemv",
            quant="w4_paro",
            variant="transposed_fp16",
        )
        is gemv_awq_dual_pack8_transposed_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="rotate+selected_dual_pack8_gemv",
            quant="w4_paro",
            variant="strided",
        )
        is gemv_awq_selected_dual_pack8_strided_rotate_out_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="selected_dual_pack8_gemv",
            quant="w4_paro",
            variant="strided",
        )
        is gemv_awq_selected_dual_pack8_strided_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="selected_dual_pack8_gemv",
            quant="w4_paro",
            variant="transposed",
        )
        is gemv_awq_selected_dual_pack8_transposed_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="selected_pack8_gemv",
            quant="w4_paro",
            variant="strided",
        )
        is gemv_awq_selected_pack8_strided_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="selected_pack8_gemv",
            quant="w4_paro",
            variant="transposed",
        )
        is gemv_awq_selected_pack8_transposed_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="rotate+selected_dual_pack8_gemv",
            quant="w4_paro",
            variant="strided_fp16",
        )
        is gemv_awq_selected_dual_pack8_strided_rotate_out_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="selected_dual_pack8_gemv",
            quant="w4_paro",
            variant="strided_fp16",
        )
        is gemv_awq_selected_dual_pack8_strided_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="selected_dual_pack8_gemv",
            quant="w4_paro",
            variant="transposed_fp16",
        )
        is gemv_awq_selected_dual_pack8_transposed_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="selected_pack8_gemv",
            quant="w4_paro",
            variant="strided_fp16",
        )
        is gemv_awq_selected_pack8_strided_fp16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="selected_pack8_gemv",
            quant="w4_paro",
            variant="transposed_fp16",
        )
        is gemv_awq_selected_pack8_transposed_fp16
    )


def test_paro_awq_gemv_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_paro_awq_gemv_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc paro awq gemv test version",
    )

    assert artifact.family == "paro_awq_gemv"
    assert artifact.profile.name == "decode"
    assert artifact.profile.wavefront == 32
    assert artifact.flags[:2] == ("-mllvm", "-amdgpu-unroll-threshold-local=600")
    assert "-mcumode" in artifact.flags
    assert artifact.output_path.name == "paro_awq_gemv.so"
    assert artifact.compiler_version == "hipcc paro awq gemv test version"
    assert any(str(path).endswith("paro_awq_gemv.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_paro_awq_gemv_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="rows must be positive"):
        gemv_awq_pack8_strided_bf16(0, 0, 0, 0, 0, 0, 16, 1, 8)
    with pytest.raises(ValueError, match="in_features must be divisible"):
        gemv_awq_pack8_transposed_bf16(0, 0, 0, 0, 0, 2, 18, 1, 8)
    with pytest.raises(ValueError, match="group_size must be a multiple of 8"):
        gemv_awq_dual_pack8_strided_bf16(0, 0, 0, 0, 0, 0, 0, 0, 2, 16, 1, 1, 4)
    with pytest.raises(ValueError, match="threads must be one of 64 or 128"):
        gemv_awq_dual_pack8_transposed_bf16(0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 16, 1, 1, 8, threads=256)
    with pytest.raises(ValueError, match="rows must be positive"):
        gemv_awq_pack8_strided_fp16(0, 0, 0, 0, 0, 0, 16, 1, 8)
    with pytest.raises(ValueError, match="group_size must be a multiple of 16"):
        awq_fusedw4_prefill_fp16(0, 0, 0, 0, 0, 2, 16, 1, 8)
    with pytest.raises(ValueError, match="tile_m must be one of"):
        awq_fusedw4_prefill_fp16(0, 0, 0, 0, 0, 2, 16, 1, 16, tile_m=48)
    with pytest.raises(ValueError, match="out_packed_b must be positive"):
        awq_fusedw4_prefill_dual_fp16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 16, 1, 0, 16)
    with pytest.raises(ValueError, match="tile_n must be one of"):
        awq_fusedw4_prefill_strided_fp16(0, 0, 0, 0, 0, 2, 16, 1, 16, tile_n=48)
    with pytest.raises(ValueError, match="in_features must be divisible"):
        gemv_awq_pack8_transposed_fp16(0, 0, 0, 0, 0, 2, 18, 1, 8)
    with pytest.raises(ValueError, match="group_size must be a multiple of 8"):
        gemv_awq_dual_pack8_strided_fp16(0, 0, 0, 0, 0, 0, 0, 0, 2, 16, 1, 1, 4)
    with pytest.raises(ValueError, match="threads must be one of 64 or 128"):
        gemv_awq_dual_pack8_transposed_fp16(0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 16, 1, 1, 8, threads=256)
    with pytest.raises(ValueError, match="krot must be non-negative"):
        gemv_awq_selected_dual_pack8_strided_rotate_out_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 16, 1, 1, 2, 8, -1
        )
    with pytest.raises(ValueError, match="x_rows must be positive"):
        gemv_awq_selected_dual_pack8_strided_bf16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 16, 1, 1, 2, 8)
    with pytest.raises(ValueError, match="rows must be divisible by x_rows"):
        gemv_awq_selected_dual_pack8_strided_bf16(0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 16, 1, 1, 2, 8)
    with pytest.raises(ValueError, match="in_features must be divisible"):
        gemv_awq_selected_dual_pack8_transposed_bf16(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 18, 1, 1, 2, 8)
    with pytest.raises(ValueError, match="group_size must be a multiple of 8"):
        gemv_awq_selected_pack8_strided_bf16(0, 0, 0, 0, 0, 0, 2, 16, 1, 2, 4)
    with pytest.raises(ValueError, match="threads must be one of 64 or 128"):
        gemv_awq_selected_pack8_transposed_bf16(0, 0, 0, 0, 0, 0, 2, 16, 1, 2, 8, threads=256)
    with pytest.raises(ValueError, match="krot must be non-negative"):
        gemv_awq_selected_dual_pack8_strided_rotate_out_fp16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 16, 1, 1, 2, 8, -1
        )
    with pytest.raises(ValueError, match="x_rows must be positive"):
        gemv_awq_selected_dual_pack8_strided_fp16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 16, 1, 1, 2, 8)
    with pytest.raises(ValueError, match="in_features must be divisible"):
        gemv_awq_selected_dual_pack8_transposed_fp16(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 18, 1, 1, 2, 8)
    with pytest.raises(ValueError, match="group_size must be a multiple of 8"):
        gemv_awq_selected_pack8_strided_fp16(0, 0, 0, 0, 0, 0, 2, 16, 1, 2, 4)
    with pytest.raises(ValueError, match="threads must be one of 64 or 128"):
        gemv_awq_selected_pack8_transposed_fp16(0, 0, 0, 0, 0, 0, 2, 16, 1, 2, 8, threads=256)
