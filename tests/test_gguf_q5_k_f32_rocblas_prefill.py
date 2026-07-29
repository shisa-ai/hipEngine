"""Exact-value raw-Q5_K F32 dequantization plus rocBLAS SGEMM contracts."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.core.rocblas import Rocblas
from hipengine.kernels.cpu_reference import gguf_q5_k_gemv
from hipengine.kernels.hip_gfx1100.convert.cast import build_cast
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q5_raw
from hipengine.kernels.hip_gfx1100.quant import gguf_q5_k_f32_rocblas_prefill as q5_f32
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from tests.test_gguf_k_gemv import make_q5_k_weight

_QK_K = 256
_Q5_BLOCK_BYTES = 176
_SOURCE = (
    Path(__file__).parents[1]
    / "hipengine"
    / "kernels"
    / "hip_gfx1100"
    / "quant"
    / "gguf_q5_k_f32_rocblas_prefill.hip"
)
_ORDERED_GEOMETRIES = ((4, 8), (8, 4))


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        ctypes.CDLL("librocblas.so")
    except OSError:
        return False
    return True


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    bits = contiguous.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def _edge_q5_weight(out_features: int, in_features: int) -> np.ndarray:
    raw = make_q5_k_weight(out_features, in_features)
    d_values = (np.float16(0.5), np.float16(-0.25), np.float16(0.0))
    dmin_values = (np.float16(-0.125), np.float16(0.75), np.float16(0.0))
    scale_patterns = (
        np.asarray([0x00, 0xFF, 0x55, 0xAA] * 3, dtype=np.uint8),
        np.asarray([0x3F, 0xC0, 0x0F, 0xF0] * 3, dtype=np.uint8),
        np.arange(12, dtype=np.uint8) * np.uint8(17),
    )
    blocks_per_row = in_features // _QK_K
    for row in range(out_features):
        for block_id in range(blocks_per_row):
            start = block_id * _Q5_BLOCK_BYTES
            raw[row, start : start + 2] = np.asarray(
                [d_values[(row + block_id) % len(d_values)]], dtype=np.float16
            ).view(np.uint8)
            raw[row, start + 2 : start + 4] = np.asarray(
                [dmin_values[(2 * row + block_id) % len(dmin_values)]],
                dtype=np.float16,
            ).view(np.uint8)
            raw[row, start + 4 : start + 16] = scale_patterns[
                (row + 2 * block_id) % len(scale_patterns)
            ]
    return raw


def _device(array: np.ndarray, runtime):
    contiguous = np.ascontiguousarray(array)
    result = malloc(contiguous.nbytes, runtime=runtime)
    copy_host_to_device(
        result,
        host_array_ptr(contiguous),
        contiguous.nbytes,
        runtime=runtime,
    )
    return result


def test_q5_f32_rocblas_registry_build_scope_and_workspace_contract() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)

    assert q5_f32.q5_k_f32_weight_nbytes(3_072, 12_288) == 150_994_944
    assert q5_f32.q5_k_f32_input_nbytes(512, 9_216) == 18_874_368
    assert q5_f32.q5_k_f32_output_nbytes(512, 12_288) == 25_165_824
    assert q5_f32.q5_k_f32_rocblas_session_nbytes(512) == 195_035_136
    assert q5_f32.q5_k_f32_rocblas_workspace_nbytes(17, 256, 72) == (
        (17 * 256 + 256 * 72 + 17 * 72) * 4
    )
    assert q5_f32.q5_k_f32_ordered_workspace_nbytes(256, 72) == 256 * 72 * 4

    dequant_key = KernelKey(
        "hip_gfx1100", "dequant", "gguf_q5_k", "raw_f32_exact_local64"
    )
    assert resolve(
        backend=dequant_key.backend,
        layer=dequant_key.layer,
        quant=dequant_key.quant,
        variant=dequant_key.variant,
    ) is q5_f32.gguf_q5_k_dequantize_f32_exact
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            dequant_key.layer,
            dequant_key.quant,
            dequant_key.variant,
        )
    )
    fused_key = KernelKey(
        "hip_gfx1100",
        "dequant_cast",
        "gguf_q5_k",
        "raw_f32_bf16_input_exact_local64",
    )
    assert resolve(
        backend=fused_key.backend,
        layer=fused_key.layer,
        quant=fused_key.quant,
        variant=fused_key.variant,
    ) is q5_f32.gguf_q5_k_dequantize_bf16_to_f32_exact_fused
    assert not is_registered(
        KernelKey(
            "hip_gfx1151", fused_key.layer, fused_key.quant, fused_key.variant
        )
    )

    for output_dtype, function in (
        ("bf16", q5_f32.gguf_q5_k_f32_rocblas_bf16_bf16_out),
        ("f32", q5_f32.gguf_q5_k_f32_rocblas_bf16_f32_out),
    ):
        key = KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k",
            f"f32_rocblas_exact_values_bf16_{output_dtype}_out",
        )
        assert resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        ) is function
        assert not is_registered(
            KernelKey("hip_gfx1151", key.layer, key.quant, key.variant)
        )

    for col_tile, row_batch in _ORDERED_GEOMETRIES:
        for output_dtype in ("bf16", "f32"):
            suffix = (
                f"coltile{col_tile}_rowbatch{row_batch}_bf16_"
                f"{output_dtype}_out"
            )
            primitive = getattr(
                q5_f32, f"gguf_q5_k_f32_weight_ordered_{suffix}"
            )
            composite = getattr(q5_f32, f"gguf_q5_k_f32_ordered_{suffix}")
            primitive_key = KernelKey(
                "hip_gfx1100", "linear", "f32_weight", f"ordered_{suffix}"
            )
            composite_key = KernelKey(
                "hip_gfx1100", "linear", "gguf_q5_k", f"f32_ordered_{suffix}"
            )
            for key, function in (
                (primitive_key, primitive),
                (composite_key, composite),
            ):
                assert resolve(
                    backend=key.backend,
                    layer=key.layer,
                    quant=key.quant,
                    variant=key.variant,
                ) is function
                assert not is_registered(
                    KernelKey("hip_gfx1151", key.layer, key.quant, key.variant)
                )

    artifact = q5_f32.plan_gguf_q5_k_f32_rocblas_prefill_build(
        compiler_version="test"
    )
    assert artifact.output_path.name == "gguf_q5_k_f32_rocblas_prefill.so"
    assert any(
        path.name == "gguf_q5_k_f32_rocblas_prefill.hip"
        for path in artifact.sources
    )
    source = _SOURCE.read_text()
    assert "torch::Tensor" not in source
    assert "__global__ void gguf_q5_k_dequantize_f32_exact_kernel" in source
    assert "gguf_q5_k_f32_weight_ordered_coltile_kernel" in source
    assert "COL_TILE * ROW_BATCH == 32" in source


def test_q5_f32_rocblas_rejects_invalid_shapes_before_loading_libraries() -> None:
    with pytest.raises(ValueError, match="multiple of 256"):
        q5_f32.gguf_q5_k_dequantize_f32_exact(1, 2, 192, 7)
    with pytest.raises(ValueError, match="rows must be positive"):
        q5_f32.gguf_q5_k_dequantize_bf16_to_f32_exact_fused(
            1, 2, 3, 4, 0, 256, 7
        )
    with pytest.raises(ValueError, match="rows must be positive"):
        q5_f32.gguf_q5_k_f32_rocblas_bf16_bf16_out(
            1, 2, 3, 4, 5, 6, 0, 256, 64
        )
    with pytest.raises(ValueError, match="multiple of 256"):
        q5_f32.gguf_q5_k_f32_rocblas_bf16_f32_out(
            1, 2, 3, 4, 5, 6, 17, 384, 64
        )
    with pytest.raises(ValueError, match="multiple of 256"):
        q5_f32.gguf_q5_k_f32_ordered_coltile4_rowbatch8_bf16_f32_out(
            1, 2, 3, 4, 17, 384, 64
        )
    with pytest.raises(ValueError, match="divisible by 4"):
        q5_f32.gguf_q5_k_f32_ordered_coltile4_rowbatch8_bf16_bf16_out(
            1, 2, 3, 4, 17, 512, 66
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP/rocBLAS is not available")
def test_q5_exact_f32_dequant_matches_independent_cpu_values() -> None:
    from hipengine.core.hip import get_hip_runtime

    in_features = 512
    raw = _edge_q5_weight(out_features=5, in_features=in_features)
    expected = dequantize_gguf_data(raw, GGMLQuantizationType.Q5_K).astype(
        np.float32
    )
    actual = np.empty_like(expected)
    fused_actual = np.empty_like(expected)
    x_bf16 = _bf16_bits(
        np.arange(17 * in_features, dtype=np.float32).reshape(17, in_features)
        / np.float32(257.0)
    )
    expected_x_f32 = _bf16_to_f32(x_bf16)
    fused_x_f32 = np.empty_like(expected_x_f32)
    runtime = get_hip_runtime()
    library = q5_f32.build_gguf_q5_k_f32_rocblas_prefill(load=True)
    before = memory_stats()
    buffers = []
    try:
        raw_dev = _device(raw, runtime)
        x_dev = _device(x_bf16, runtime)
        out_dev = malloc(actual.nbytes, runtime=runtime)
        fused_out_dev = malloc(fused_actual.nbytes, runtime=runtime)
        fused_x_dev = malloc(fused_x_f32.nbytes, runtime=runtime)
        buffers.extend((raw_dev, x_dev, out_dev, fused_out_dev, fused_x_dev))
        q5_f32.gguf_q5_k_dequantize_f32_exact(
            raw_dev.ptr,
            out_dev.ptr,
            in_features,
            raw.shape[0],
            library=library,
            runtime=runtime,
        )
        q5_f32.gguf_q5_k_dequantize_bf16_to_f32_exact_fused(
            raw_dev.ptr,
            fused_out_dev.ptr,
            x_dev.ptr,
            fused_x_dev.ptr,
            x_bf16.shape[0],
            in_features,
            raw.shape[0],
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(actual), out_dev, actual.nbytes, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(fused_actual),
            fused_out_dev,
            fused_actual.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(fused_x_f32),
            fused_x_dev,
            fused_x_f32.nbytes,
            runtime=runtime,
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(
        fused_actual.view(np.uint32), expected.view(np.uint32)
    )
    np.testing.assert_array_equal(
        fused_x_f32.view(np.uint32), expected_x_f32.view(np.uint32)
    )
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


def _run_candidate(
    *, rows: int, in_features: int, out_features: int, output_dtype: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from hipengine.core.hip import get_hip_runtime

    rng = np.random.default_rng(rows * 97 + in_features + out_features)
    x = rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _bf16_bits(x)
    qweight = make_q5_k_weight(out_features, in_features)
    reference = gguf_q5_k_gemv(_bf16_to_f32(x_bf16), qweight)
    host_dtype = np.uint16 if output_dtype == "bf16" else np.float32
    actual_raw = np.empty((rows, out_features), dtype=host_dtype)
    x_f32_host = np.empty((rows, in_features), dtype=np.float32)

    runtime = get_hip_runtime()
    dequant_library = q5_f32.build_gguf_q5_k_f32_rocblas_prefill(load=True)
    cast_library = build_cast(load=True)
    rocblas = Rocblas.load()
    before = memory_stats()
    buffers = []
    try:
        x_dev = _device(x_bf16, runtime)
        weight_dev = _device(qweight, runtime)
        out_dev = malloc(actual_raw.nbytes, runtime=runtime)
        x_f32_dev = malloc(
            q5_f32.q5_k_f32_input_nbytes(rows, in_features), runtime=runtime
        )
        weight_f32_dev = malloc(
            q5_f32.q5_k_f32_weight_nbytes(in_features, out_features),
            runtime=runtime,
        )
        out_f32_dev = malloc(
            q5_f32.q5_k_f32_output_nbytes(rows, out_features), runtime=runtime
        )
        buffers.extend(
            (x_dev, weight_dev, out_dev, x_f32_dev, weight_f32_dev, out_f32_dev)
        )
        function = (
            q5_f32.gguf_q5_k_f32_rocblas_bf16_bf16_out
            if output_dtype == "bf16"
            else q5_f32.gguf_q5_k_f32_rocblas_bf16_f32_out
        )
        function(
            x_dev.ptr,
            weight_dev.ptr,
            out_dev.ptr,
            x_f32_dev.ptr,
            weight_f32_dev.ptr,
            out_f32_dev.ptr,
            rows,
            in_features,
            out_features,
            dequant_library=dequant_library,
            cast_library=cast_library,
            rocblas=rocblas,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(actual_raw),
            out_dev,
            actual_raw.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(x_f32_host),
            x_f32_dev,
            x_f32_host.nbytes,
            runtime=runtime,
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        rocblas.close()
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
    actual = _bf16_to_f32(actual_raw) if output_dtype == "bf16" else actual_raw
    return actual, reference, x_f32_host


@pytest.mark.skipif(not _hip_available(), reason="HIP/rocBLAS is not available")
@pytest.mark.parametrize("rows", [17, 33])
def test_q5_f32_ordered_geometries_match_raw_coltile_bytes(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    in_features, out_features = 512, 48
    rng = np.random.default_rng(20260730 + rows)
    x_bf16 = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    qweight = _edge_q5_weight(out_features, in_features)
    runtime = get_hip_runtime()
    raw_library = q5_raw.build_gguf_k_gemv(load=True)
    ordered_library = q5_f32.build_gguf_q5_k_f32_rocblas_prefill(load=True)
    before = memory_stats()
    buffers = []
    try:
        x_dev = _device(x_bf16, runtime)
        weight_dev = _device(qweight, runtime)
        weight_f32_dev = malloc(
            q5_f32.q5_k_f32_ordered_workspace_nbytes(
                in_features, out_features
            ),
            runtime=runtime,
        )
        buffers.extend((x_dev, weight_dev, weight_f32_dev))
        for output_dtype in ("bf16", "f32"):
            host_dtype = np.uint16 if output_dtype == "bf16" else np.float32
            expected = np.empty((rows, out_features), dtype=host_dtype)
            actual = np.empty_like(expected)
            expected_dev = malloc(expected.nbytes, runtime=runtime)
            actual_dev = malloc(actual.nbytes, runtime=runtime)
            buffers.extend((expected_dev, actual_dev))
            control = getattr(
                q5_raw,
                f"gguf_q5_k_gemv_coltile4_rowbatch8_bf16_"
                f"{output_dtype}_out",
            )
            control(
                x_dev.ptr,
                weight_dev.ptr,
                expected_dev.ptr,
                rows,
                in_features,
                out_features,
                library=raw_library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(
                host_array_ptr(expected),
                expected_dev,
                expected.nbytes,
                runtime=runtime,
            )
            for col_tile, row_batch in _ORDERED_GEOMETRIES:
                candidate = getattr(
                    q5_f32,
                    f"gguf_q5_k_f32_ordered_coltile{col_tile}_"
                    f"rowbatch{row_batch}_bf16_{output_dtype}_out",
                )
                candidate(
                    x_dev.ptr,
                    weight_dev.ptr,
                    actual_dev.ptr,
                    weight_f32_dev.ptr,
                    rows,
                    in_features,
                    out_features,
                    library=ordered_library,
                    runtime=runtime,
                )
                runtime.device_synchronize()
                copy_device_to_host(
                    host_array_ptr(actual),
                    actual_dev,
                    actual.nbytes,
                    runtime=runtime,
                )
                np.testing.assert_array_equal(actual, expected)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


@pytest.mark.skipif(not _hip_available(), reason="HIP/rocBLAS is not available")
@pytest.mark.parametrize(
    ("rows", "in_features", "out_features", "output_dtype"),
    [(17, 256, 72, "f32"), (32, 512, 128, "bf16")],
)
def test_q5_f32_rocblas_outputs_are_finite_and_pass_exact_path_quality(
    rows: int,
    in_features: int,
    out_features: int,
    output_dtype: str,
) -> None:
    actual, reference, x_f32 = _run_candidate(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        output_dtype=output_dtype,
    )
    rng = np.random.default_rng(rows * 97 + in_features + out_features)
    x = rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    expected_x_f32 = _bf16_to_f32(_bf16_bits(x))
    np.testing.assert_array_equal(x_f32.view(np.uint32), expected_x_f32.view(np.uint32))
    assert np.all(np.isfinite(actual))
    result = evaluate_logits(reference, actual)
    assert result.kl_mean <= 0.05, result
    assert result.top1_agreement >= 0.90, result
    assert result.passed, result
