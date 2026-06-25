from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.linear import (
    dense_dual_gemv_out_bf16,
    dense_dual_gemv_out_bf16_wmma,
    dense_dual_gemv_out_fp16,
    dense_dual_gemv_out_fp16_wmma,
    dense_dual_gemv_separate_out_bf16,
    dense_dual_gemv_separate_out_fp16,
    build_dense_gemv,
    dense_gemv_bf16_f32w_bf16_out,
    dense_gemv_out_bf16,
    dense_gemv_out_bf16_wmma,
    dense_gemv_out_fp16,
    dense_gemv_out_fp16_wmma,
    plan_dense_gemv_build,
    register_dense_gemv_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


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
        resolve(backend="hip_gfx1100", layer="dense_gemv", quant="f32", variant="bf16_hidden_bf16_out")
        is dense_gemv_bf16_f32w_bf16_out
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


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_dense_gemv_bf16_hidden_f32_weight_matches_cpu_reference() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_dense_gemv(load=True)
    x_f32 = np.asarray(
        [
            [0.25, -0.5, 1.0, -1.5, 2.0, -2.5, 3.0, -3.5, 0.75, -0.875, 1.125, -1.375, 1.625, -1.875, 2.125, -2.375],
            [-0.125, 0.375, -0.625, 0.875, -1.125, 1.375, -1.625, 1.875, -2.125, 2.375, -2.625, 2.875, -3.125, 3.375, -3.625, 3.875],
        ],
        dtype=np.float32,
    )
    weight_f32 = (np.arange(48, dtype=np.float32).reshape(3, 16) - 23.5) / 17.0
    x_bf16 = float_array_to_bf16_bits(x_f32)
    out_bf16 = np.empty((2, 3), dtype=np.uint16)
    bufs = []
    try:
        dx = malloc(x_bf16.nbytes, runtime=runtime)
        dw = malloc(weight_f32.nbytes, runtime=runtime)
        dout = malloc(out_bf16.nbytes, runtime=runtime)
        bufs.extend((dx, dw, dout))
        copy_host_to_device(dx, host_array_ptr(x_bf16), runtime=runtime)
        copy_host_to_device(dw, host_array_ptr(weight_f32), runtime=runtime)
        dense_gemv_bf16_f32w_bf16_out(
            dx.ptr,
            dw.ptr,
            dout.ptr,
            rows=2,
            in_features=16,
            out_features=3,
            threads=64,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_bf16), dout, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    x_cpu = bf16_to_float32(x_bf16)
    expected = x_cpu @ weight_f32.T
    expected_bf16 = float_array_to_bf16_bits(expected.astype(np.float32, copy=False))
    np.testing.assert_array_equal(out_bf16, expected_bf16)


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
    with pytest.raises(ValueError, match="threads must be one of"):
        dense_gemv_bf16_f32w_bf16_out(0, 0, 0, 1, 16, 8, threads=32)
    with pytest.raises(ValueError, match="out_features must be positive"):
        dense_dual_gemv_out_fp16(0, 0, 0, 0, 1, 16, 8, 0)
    with pytest.raises(ValueError, match="out_features must be positive"):
        dense_dual_gemv_separate_out_fp16(0, 0, 0, 0, 0, 1, 16, 8, 0)
    with pytest.raises(ValueError, match="multiple of 16"):
        dense_gemv_out_fp16_wmma(0, 0, 0, 1, 17, 8)
