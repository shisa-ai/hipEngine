from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.wmma import (
    gemm_awq_selected_dual_pack8_wmma_compact_bf16,
    gemm_awq_selected_dual_pack8_wmma_compact_fp16,
    gemm_awq_selected_pack8_wmma_compact_bf16,
    gemm_awq_selected_pack8_wmma_compact_fp16,
    plan_paro_awq_wmma_build,
    register_paro_awq_wmma_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_paro_awq_wmma_registers_compact_variants() -> None:
    register_paro_awq_wmma_kernels()

    for quant in ("bf16", "w4_paro"):
        assert (
            resolve(backend="hip_gfx1100", layer="awq_wmma", quant=quant, variant="selected_dual_pack8_compact")
            is gemm_awq_selected_dual_pack8_wmma_compact_bf16
        )
        assert (
            resolve(backend="hip_gfx1100", layer="awq_wmma", quant=quant, variant="selected_pack8_compact")
            is gemm_awq_selected_pack8_wmma_compact_bf16
        )
        assert (
            resolve(backend="hip_gfx1100", layer="awq_wmma", quant=quant, variant="selected_dual_pack8_compact_fp16")
            is gemm_awq_selected_dual_pack8_wmma_compact_fp16
        )
        assert (
            resolve(backend="hip_gfx1100", layer="awq_wmma", quant=quant, variant="selected_pack8_compact_fp16")
            is gemm_awq_selected_pack8_wmma_compact_fp16
        )


def test_paro_awq_wmma_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_paro_awq_wmma_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc paro awq wmma test version",
    )

    assert artifact.family == "paro_awq_wmma"
    assert artifact.profile.name == "prefill"
    assert artifact.output_path.name == "paro_awq_wmma.so"
    assert "-mcumode" in artifact.flags
    assert artifact.compiler_version == "hipcc paro awq wmma test version"
    assert any(str(path).endswith("paro_awq_wmma.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_paro_awq_wmma_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="compact_rows"):
        gemm_awq_selected_dual_pack8_wmma_compact_bf16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 2, 2, 1, 16, 16)
    with pytest.raises(ValueError, match="group_size"):
        gemm_awq_selected_dual_pack8_wmma_compact_fp16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 16, 2, 2, 1, 8, 16)
    with pytest.raises(ValueError, match="wmma_total_rows"):
        gemm_awq_selected_pack8_wmma_compact_bf16(0, 0, 0, 0, 0, 0, 0, 0, 1, 16, 2, 1, 16, 15)
    with pytest.raises(ValueError, match="out_packed"):
        gemm_awq_selected_pack8_wmma_compact_fp16(0, 0, 0, 0, 0, 0, 0, 0, 1, 16, 0, 1, 16, 16)
