from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.moe import (
    plan_qwen35_moe_group_scatter_build,
    qwen35_moe_gather_packed_hidden_lowp,
    qwen35_moe_group_count,
    qwen35_moe_group_prefix,
    qwen35_moe_group_scatter,
    qwen35_moe_group_scatter_gather_lowp,
    qwen35_moe_prefill_grouped_compact,
    qwen35_moe_prefill_selected_c1_rows,
    qwen35_moe_wmma_tile_map,
    register_qwen35_moe_group_scatter_kernels,
    register_qwen35_moe_prefill_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_qwen35_moe_group_scatter_registers_prefill_metadata_variants() -> None:
    register_qwen35_moe_group_scatter_kernels()
    register_qwen35_moe_prefill_kernels()

    assert resolve(backend="hip_gfx1100", layer="moe_group_count", quant="w4_paro", variant="qwen35") is qwen35_moe_group_count
    assert resolve(backend="hip_gfx1100", layer="moe_group_prefix", quant="w4_paro", variant="qwen35") is qwen35_moe_group_prefix
    assert resolve(backend="hip_gfx1100", layer="moe_wmma_tile_map", quant="w4_paro", variant="qwen35") is qwen35_moe_wmma_tile_map
    assert resolve(backend="hip_gfx1100", layer="moe_group_scatter", quant="w4_paro", variant="qwen35") is qwen35_moe_group_scatter
    assert (
        resolve(backend="hip_gfx1100", layer="moe_group_scatter_gather", quant="w4_paro", variant="qwen35_lowp")
        is qwen35_moe_group_scatter_gather_lowp
    )
    assert (
        resolve(backend="hip_gfx1100", layer="moe_gather_packed_hidden", quant="w4_paro", variant="qwen35_lowp")
        is qwen35_moe_gather_packed_hidden_lowp
    )
    assert (
        resolve(backend="hip_gfx1100", layer="moe_prefill", quant="w4_paro", variant="qwen35_grouped_compact")
        is qwen35_moe_prefill_grouped_compact
    )
    assert (
        resolve(backend="hip_gfx1100", layer="moe_prefill", quant="w4_paro", variant="qwen35_selected_c1_rows")
        is qwen35_moe_prefill_selected_c1_rows
    )


def test_qwen35_moe_group_scatter_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_qwen35_moe_group_scatter_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc qwen35 moe group scatter test version",
    )

    assert artifact.family == "qwen35_moe_group_scatter"
    assert artifact.profile.name == "prefill"
    assert artifact.profile.wavefront == 32
    assert artifact.output_path.name == "qwen35_moe_group_scatter.so"
    assert artifact.compiler_version == "hipcc qwen35 moe group scatter test version"
    assert any(str(path).endswith("group_scatter.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_qwen35_moe_group_scatter_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="total_lanes"):
        qwen35_moe_group_count(0, 0, 0, 1)
    with pytest.raises(ValueError, match="num_experts"):
        qwen35_moe_group_prefix(0, 0, 0, 0, 0, 1)
    with pytest.raises(ValueError, match="pad_multiple"):
        qwen35_moe_group_prefix(0, 0, 0, 0, 1, 0)
    with pytest.raises(ValueError, match="num_experts"):
        qwen35_moe_wmma_tile_map(0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="total_lanes"):
        qwen35_moe_group_scatter(0, 0, 0, 0, 0, 0, 0, 0, 1)
    with pytest.raises(ValueError, match="top_k"):
        qwen35_moe_group_scatter_gather_lowp(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 4)
    with pytest.raises(ValueError, match="total_elements"):
        qwen35_moe_gather_packed_hidden_lowp(0, 0, 0, 0, 1, 1, 4)
