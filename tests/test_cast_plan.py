from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.convert import (
    bf16_to_fp16_scaled_rows,
    bf16_to_f32,
    f32_to_bf16,
    f32_to_fp16,
    f32_scale_rows,
    fp16_to_bf16,
    fp16_to_bf16_strided_rows,
    fp16_to_f32,
    f32_scale_rows_to_bf16,
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
    assert resolve(
        backend="hip_gfx1100",
        layer="cast_bf16_to_fp16",
        quant="scaled_rows",
    ) is bf16_to_fp16_scaled_rows
    assert resolve(
        backend="hip_gfx1100",
        layer="cast_f32_to_bf16",
        quant="scaled_rows",
    ) is f32_scale_rows_to_bf16
    assert resolve(
        backend="hip_gfx1100",
        layer="cast_f32_scale_rows",
        quant="fp32",
    ) is f32_scale_rows


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
    with pytest.raises(ValueError, match="rows"):
        bf16_to_fp16_scaled_rows(0, 0, 0, 0, 4)
    with pytest.raises(ValueError, match="cols"):
        f32_scale_rows_to_bf16(0, 0, 0, 2, 0)
    with pytest.raises(ValueError, match="rows"):
        f32_scale_rows(0, 0, 0, 4)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_scaled_row_cast_preserves_finite_bf16_range_and_restores_scale() -> None:
    values = np.asarray(
        [
            [1.0, -2.0, 3.5, 0.0],
            [131_072.0, -65_536.0, 32.0, -0.5],
        ],
        dtype=np.float32,
    )
    bits = (values.view(np.uint32) >> 16).astype(np.uint16)
    expected = (bits.astype(np.uint32) << 16).view(np.float32)
    buffers = []
    try:
        source = malloc(bits.nbytes)
        cast = malloc(bits.nbytes)
        scales = malloc(values.shape[0] * np.dtype(np.float32).itemsize)
        cast_f32 = malloc(values.nbytes)
        restored = malloc(bits.nbytes)
        buffers.extend((source, cast, scales, cast_f32, restored))
        copy_host_to_device(source, host_array_ptr(bits), bits.nbytes)
        bf16_to_fp16_scaled_rows(
            source.ptr,
            cast.ptr,
            scales.ptr,
            values.shape[0],
            values.shape[1],
        )
        fp16_to_f32(cast.ptr, cast_f32.ptr, values.size)
        f32_scale_rows_to_bf16(
            cast_f32.ptr,
            scales.ptr,
            restored.ptr,
            values.shape[0],
            values.shape[1],
        )
        f32_scale_rows(
            cast_f32.ptr,
            scales.ptr,
            values.shape[0],
            values.shape[1],
        )
        cast_host = np.empty_like(bits)
        scale_host = np.empty(values.shape[0], dtype=np.float32)
        scaled_f32_host = np.empty_like(values)
        restored_host = np.empty_like(bits)
        copy_device_to_host(host_array_ptr(cast_host), cast, cast_host.nbytes)
        copy_device_to_host(host_array_ptr(scale_host), scales, scale_host.nbytes)
        copy_device_to_host(host_array_ptr(scaled_f32_host), cast_f32, scaled_f32_host.nbytes)
        copy_device_to_host(host_array_ptr(restored_host), restored, restored_host.nbytes)
        assert np.isfinite(cast_host.view(np.float16)).all()
        np.testing.assert_array_equal(scale_host, np.asarray([1.0, 4.0], dtype=np.float32))
        np.testing.assert_array_equal(scaled_f32_host, expected)
        restored_values = (restored_host.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(restored_values, expected)
    finally:
        for buffer in reversed(buffers):
            free(buffer)
