from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.convert import (
    bf16_to_f32,
    f32_to_bf16,
    f32_to_fp16,
    fp16_to_bf16,
    fp16_to_bf16_strided_rows,
    fp16_to_f32,
    plan_cast_build,
    register_cast_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve


def setup_function() -> None:
    clear_registry_for_tests()


def test_cast_registers_bf16_and_fp16_variants() -> None:
    register_cast_kernels()

    assert resolve(backend="hip_gfx1100", layer="cast_f32_to_bf16", quant="bf16") is f32_to_bf16
    assert resolve(backend="hip_gfx1100", layer="cast_bf16_to_f32", quant="fp32") is bf16_to_f32
    assert resolve(backend="hip_gfx1100", layer="cast_f32_to_fp16", quant="fp16") is f32_to_fp16
    assert resolve(backend="hip_gfx1100", layer="cast_fp16_to_f32", quant="fp32") is fp16_to_f32
    assert resolve(backend="hip_gfx1100", layer="cast_fp16_to_bf16", quant="bf16") is fp16_to_bf16
    assert resolve(backend="hip_gfx1100", layer="cast_fp16_to_bf16_strided_rows", quant="bf16") is fp16_to_bf16_strided_rows


def test_cast_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_cast_build(cache_root=tmp_path / "cache", compiler_version="hipcc cast test version")

    assert artifact.family == "cast"
    assert artifact.output_path.name == "cast.so"
    assert any(str(path).endswith("cast.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()


def test_cast_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="count"):
        f32_to_bf16(0, 0, 0)
    with pytest.raises(ValueError, match="count"):
        bf16_to_f32(0, 0, 0)
    with pytest.raises(ValueError, match="count"):
        f32_to_fp16(0, 0, 0)
    with pytest.raises(ValueError, match="count"):
        fp16_to_f32(0, 0, 0)
    with pytest.raises(ValueError, match="count"):
        fp16_to_bf16(0, 0, 0)
    with pytest.raises(ValueError, match="rows"):
        fp16_to_bf16_strided_rows(0, 0, 0, 4, 8, 0)
    with pytest.raises(ValueError, match="dst_col_offset"):
        fp16_to_bf16_strided_rows(0, 0, 2, 4, 8, 6)
