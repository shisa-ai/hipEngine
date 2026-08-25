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
    dense_gemv_bf16_f32_out,
    dense_gemv_bf16_f32w_bf16_out,
    dense_gemv_out_bf16,
    dense_gemv_out_bf16_wmma,
    dense_gemv_out_f32,
    dense_gemv_out_fp16,
    dense_gemv_out_fp16_wmma,
    dense_gemv_rowtile_out_bf16,
    dense_gemv_virtual256_out_bf16,
    dense_gemv_virtual256_rowtile_out_bf16,
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
        resolve(backend="hip_gfx1100", layer="dense_gemv", quant="bf16", variant="f32_out")
        is dense_gemv_bf16_f32_out
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
        resolve(backend="hip_gfx1100", layer="dense_gemv", quant="bf16", variant="rowtile_out")
        is dense_gemv_rowtile_out_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_gemv", quant="bf16", variant="virtual256_out")
        is dense_gemv_virtual256_out_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="dense_gemv",
            quant="bf16",
            variant="virtual256_rowtile_out",
        )
        is dense_gemv_virtual256_rowtile_out_bf16
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_gemv", quant="f32", variant="bf16_hidden_bf16_out")
        is dense_gemv_bf16_f32w_bf16_out
    )
    assert (
        resolve(backend="hip_gfx1100", layer="dense_gemv", quant="f32", variant="f32_hidden_f32_out")
        is dense_gemv_out_f32
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
def test_dense_gemv_bf16_hidden_weight_f32_output_matches_cpu_reference() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_dense_gemv(load=True)
    x_bits = float_array_to_bf16_bits(
        np.asarray(
            [[0.25, -0.5, 1.0, -1.5, 2.0, -2.5, 3.0, -3.5] * 8],
            dtype=np.float32,
        )
    )
    weight_bits = float_array_to_bf16_bits(
        (np.arange(192, dtype=np.float32).reshape(3, 64) - 95.5) / 31.0
    )
    out = np.empty((1, 3), dtype=np.float32)
    bufs = []
    try:
        dx = malloc(x_bits.nbytes, runtime=runtime)
        dw = malloc(weight_bits.nbytes, runtime=runtime)
        dout = malloc(out.nbytes, runtime=runtime)
        bufs.extend((dx, dw, dout))
        copy_host_to_device(dx, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(dw, host_array_ptr(weight_bits), runtime=runtime)
        dense_gemv_bf16_f32_out(
            dx.ptr,
            dw.ptr,
            dout.ptr,
            rows=1,
            in_features=64,
            out_features=3,
            threads=64,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), dout, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    expected = bf16_to_float32(x_bits) @ bf16_to_float32(weight_bits).T
    np.testing.assert_allclose(out, expected, rtol=2e-6, atol=2e-6)


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


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_dense_gemv_f32_hidden_f32_weight_matches_cpu_reference() -> None:
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
    out_f32 = np.empty((2, 3), dtype=np.float32)
    bufs = []
    try:
        dx = malloc(x_f32.nbytes, runtime=runtime)
        dw = malloc(weight_f32.nbytes, runtime=runtime)
        dout = malloc(out_f32.nbytes, runtime=runtime)
        bufs.extend((dx, dw, dout))
        copy_host_to_device(dx, host_array_ptr(x_f32), runtime=runtime)
        copy_host_to_device(dw, host_array_ptr(weight_f32), runtime=runtime)
        dense_gemv_out_f32(
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
        copy_device_to_host(host_array_ptr(out_f32), dout, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    expected = x_f32 @ weight_f32.T
    np.testing.assert_allclose(out_f32, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", (2, 3, 4))
def test_dense_gemv_rowtile_is_bit_exact_to_c1_reduction_and_matches_cpu(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_dense_gemv(load=True)
    in_features = 1024
    out_features = 128
    rng = np.random.default_rng(9100 + rows)
    x_bits = float_array_to_bf16_bits(
        (rng.standard_normal((rows, in_features), dtype=np.float32) * 0.125).astype(np.float32)
    )
    weight_bits = float_array_to_bf16_bits(
        (rng.standard_normal((out_features, in_features), dtype=np.float32) * 0.0625).astype(np.float32)
    )
    baseline_bits = np.empty((rows, out_features), dtype=np.uint16)
    candidate_bits = np.empty_like(baseline_bits)
    buffers = []
    try:
        x_buf = malloc(x_bits.nbytes, runtime=runtime)
        weight_buf = malloc(weight_bits.nbytes, runtime=runtime)
        baseline_buf = malloc(baseline_bits.nbytes, runtime=runtime)
        candidate_buf = malloc(candidate_bits.nbytes, runtime=runtime)
        buffers.extend((x_buf, weight_buf, baseline_buf, candidate_buf))
        copy_host_to_device(x_buf, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(weight_buf, host_array_ptr(weight_bits), runtime=runtime)
        dense_gemv_out_bf16(
            x_buf.ptr,
            weight_buf.ptr,
            baseline_buf.ptr,
            rows,
            in_features,
            out_features,
            threads=256,
            library=library,
            runtime=runtime,
        )
        dense_gemv_rowtile_out_bf16(
            x_buf.ptr,
            weight_buf.ptr,
            candidate_buf.ptr,
            rows,
            in_features,
            out_features,
            threads=256,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(baseline_bits), baseline_buf, runtime=runtime)
        copy_device_to_host(host_array_ptr(candidate_bits), candidate_buf, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(candidate_bits, baseline_bits)
    expected = bf16_to_float32(x_bits) @ bf16_to_float32(weight_bits).T
    np.testing.assert_allclose(
        bf16_to_float32(candidate_bits),
        expected,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", (2, 3, 4))
@pytest.mark.parametrize("in_features", (1024, 5120, 6144, 10240, 17408))
def test_dense_gemv_virtual256_rowtile_is_bit_exact_to_local256_rowtile(
    rows: int,
    in_features: int,
) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_dense_gemv(load=True)
    out_features = 64
    rng = np.random.default_rng(0xD270 + rows + in_features)
    x_bits = float_array_to_bf16_bits(
        (rng.standard_normal((rows, in_features), dtype=np.float32) * 0.125).astype(np.float32)
    )
    weight_bits = float_array_to_bf16_bits(
        (rng.standard_normal((out_features, in_features), dtype=np.float32) * 0.0625).astype(np.float32)
    )
    baseline_bits = np.empty((rows, out_features), dtype=np.uint16)
    candidate_bits = np.empty_like(baseline_bits)
    buffers = []
    try:
        x_buf = malloc(x_bits.nbytes, runtime=runtime)
        weight_buf = malloc(weight_bits.nbytes, runtime=runtime)
        baseline_buf = malloc(baseline_bits.nbytes, runtime=runtime)
        candidate_buf = malloc(candidate_bits.nbytes, runtime=runtime)
        buffers.extend((x_buf, weight_buf, baseline_buf, candidate_buf))
        copy_host_to_device(x_buf, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(weight_buf, host_array_ptr(weight_bits), runtime=runtime)
        dense_gemv_rowtile_out_bf16(
            x_buf.ptr,
            weight_buf.ptr,
            baseline_buf.ptr,
            rows,
            in_features,
            out_features,
            threads=256,
            library=library,
            runtime=runtime,
        )
        dense_gemv_virtual256_rowtile_out_bf16(
            x_buf.ptr,
            weight_buf.ptr,
            candidate_buf.ptr,
            rows,
            in_features,
            out_features,
            threads=128,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(baseline_bits), baseline_buf, runtime=runtime)
        copy_device_to_host(host_array_ptr(candidate_bits), candidate_buf, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(candidate_bits, baseline_bits)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "in_features,out_features",
    ((1024, 96), (5120, 128), (6144, 64), (10240, 32)),
)
def test_dense_gemv_virtual256_is_bit_exact_to_local256_and_passes_cpu_gate(
    in_features: int,
    out_features: int,
) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_dense_gemv(load=True)
    rng = np.random.default_rng(0xD27 + in_features + out_features)
    x_bits = float_array_to_bf16_bits(
        (rng.standard_normal((1, in_features), dtype=np.float32) * 0.125).astype(np.float32)
    )
    weight_bits = float_array_to_bf16_bits(
        (rng.standard_normal((out_features, in_features), dtype=np.float32) * 0.0625).astype(np.float32)
    )
    baseline_bits = np.empty((1, out_features), dtype=np.uint16)
    candidate_bits = np.empty_like(baseline_bits)
    buffers = []
    try:
        x_buf = malloc(x_bits.nbytes, runtime=runtime)
        weight_buf = malloc(weight_bits.nbytes, runtime=runtime)
        baseline_buf = malloc(baseline_bits.nbytes, runtime=runtime)
        candidate_buf = malloc(candidate_bits.nbytes, runtime=runtime)
        buffers.extend((x_buf, weight_buf, baseline_buf, candidate_buf))
        copy_host_to_device(x_buf, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(weight_buf, host_array_ptr(weight_bits), runtime=runtime)
        dense_gemv_out_bf16(
            x_buf.ptr,
            weight_buf.ptr,
            baseline_buf.ptr,
            1,
            in_features,
            out_features,
            threads=256,
            library=library,
            runtime=runtime,
        )
        dense_gemv_virtual256_out_bf16(
            x_buf.ptr,
            weight_buf.ptr,
            candidate_buf.ptr,
            1,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(baseline_bits), baseline_buf, runtime=runtime)
        copy_device_to_host(host_array_ptr(candidate_bits), candidate_buf, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(candidate_bits, baseline_bits)
    expected = bf16_to_float32(x_bits) @ bf16_to_float32(weight_bits).T
    actual = bf16_to_float32(candidate_bits)
    p = np.exp(expected - np.max(expected, axis=-1, keepdims=True), dtype=np.float64)
    p /= np.sum(p, axis=-1, keepdims=True)
    q = np.exp(actual - np.max(actual, axis=-1, keepdims=True), dtype=np.float64)
    q /= np.sum(q, axis=-1, keepdims=True)
    kl = np.sum(p * (np.log(p + 1.0e-30) - np.log(q + 1.0e-30)), axis=-1)
    assert float(np.max(kl)) <= 0.05
    assert float(np.mean(np.argmax(expected, axis=-1) == np.argmax(actual, axis=-1))) >= 0.90


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
    with pytest.raises(ValueError, match="rows must be between 2 and 4"):
        dense_gemv_rowtile_out_bf16(0, 0, 0, 1, 16, 8)
    with pytest.raises(ValueError, match="rows must be between 2 and 4"):
        dense_gemv_rowtile_out_bf16(0, 0, 0, 5, 16, 8)
    with pytest.raises(ValueError, match="rows must equal 1"):
        dense_gemv_virtual256_out_bf16(0, 0, 0, 2, 16, 8)
    with pytest.raises(ValueError, match="threads must equal 128"):
        dense_gemv_virtual256_out_bf16(0, 0, 0, 1, 16, 8, threads=32)
    with pytest.raises(ValueError, match="rows must be between 2 and 4"):
        dense_gemv_virtual256_rowtile_out_bf16(0, 0, 0, 1, 16, 8)
    with pytest.raises(ValueError, match="threads must equal 128"):
        dense_gemv_virtual256_rowtile_out_bf16(0, 0, 0, 2, 16, 8, threads=256)
    with pytest.raises(ValueError, match="threads must be one of"):
        dense_gemv_out_f32(0, 0, 0, 1, 16, 8, threads=32)
    with pytest.raises(ValueError, match="threads must be one of"):
        dense_gemv_bf16_f32w_bf16_out(0, 0, 0, 1, 16, 8, threads=32)
    with pytest.raises(ValueError, match="out_features must be positive"):
        dense_dual_gemv_out_fp16(0, 0, 0, 0, 1, 16, 8, 0)
    with pytest.raises(ValueError, match="out_features must be positive"):
        dense_dual_gemv_separate_out_fp16(0, 0, 0, 0, 0, 1, 16, 8, 0)
    with pytest.raises(ValueError, match="multiple of 16"):
        dense_gemv_out_fp16_wmma(0, 0, 0, 1, 17, 8)
