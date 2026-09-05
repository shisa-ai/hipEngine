"""Source-faithful llama.cpp-shaped Q6_K F16/rocBLAS prefill contracts."""

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
from hipengine.kernels.cpu_reference import gguf_q5_k_gemv, gguf_q6_k_gemv
from hipengine.kernels.hip_gfx1100.convert.cast import build_cast
from hipengine.kernels.hip_gfx1100.quant import gguf_q6_k_f16_rocblas_prefill as q6_f16
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
from hipengine.quant.gguf_t16 import (
    repack_gguf_q5_k_tile16,
    repack_gguf_q6_k_tile16_qmicro_planar,
)
from tests.test_gguf_k_gemv import make_q6_k_weight
from tests.test_gguf_q4_k_gemv import make_q4_k_weight

_QK_K = 256
_Q6_BLOCK_BYTES = 210
_SOURCE = (
    Path(__file__).parents[1]
    / "hipengine"
    / "kernels"
    / "hip_gfx1100"
    / "quant"
    / "gguf_q6_k_f16_rocblas_prefill.hip"
)


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


def _source_q6_f16_cpu(qweight: np.ndarray, in_features: int) -> np.ndarray:
    """Independently reproduce llama.cpp's raw-Q6-to-F16 conversion."""

    raw = np.ascontiguousarray(qweight, dtype=np.uint8)
    if raw.ndim != 2 or in_features % _QK_K:
        raise ValueError("invalid Q6 fixture")
    out_features = raw.shape[0]
    blocks_per_row = in_features // _QK_K
    if raw.shape[1] != blocks_per_row * _Q6_BLOCK_BYTES:
        raise ValueError("invalid Q6 fixture bytes")
    out = np.empty((out_features, in_features), dtype=np.float16)
    for row in range(out_features):
        for block_id in range(blocks_per_row):
            block = raw[
                row,
                block_id * _Q6_BLOCK_BYTES : (block_id + 1) * _Q6_BLOCK_BYTES,
            ]
            ql = block[:128]
            qh = block[128:192]
            scales = block[192:208].view(np.int8)
            d = np.float32(block[208:210].view(np.float16)[0])
            for k in range(_QK_K):
                group32 = k >> 5
                lane = k & 31
                base64 = 64 if group32 >= 4 else 0
                ql_index = base64 + (group32 & 1) * 32 + lane
                low = (
                    ql[ql_index] & np.uint8(0x0F)
                    if (group32 & 2) == 0
                    else ql[ql_index] >> np.uint8(4)
                )
                qh_index = (32 if group32 >= 4 else 0) + lane
                high = (qh[qh_index] >> np.uint8(2 * (group32 & 3))) & np.uint8(3)
                quant = np.int32(int(low) | (int(high) << 4)) - np.int32(32)
                value = np.float32(d * np.float32(scales[k >> 4]))
                value = np.float32(value * np.float32(quant))
                out[row, block_id * _QK_K + k] = np.float16(value)
    # AMD clang canonicalizes signed zero in the source device expression;
    # preserve byte-exact comparison without assigning meaning to -0 payloads.
    out[out == np.float16(0.0)] = np.float16(0.0)
    return out


def _edge_q6_weight(out_features: int, in_features: int) -> np.ndarray:
    raw = make_q6_k_weight(out_features, in_features)
    scales = np.asarray(
        [-128, -64, -1, 0, 1, 2, 7, 15, 31, 63, 96, 127, -7, 0, 11, -32],
        dtype=np.int8,
    )
    d_values = (np.float16(0.5), np.float16(-0.25), np.float16(0.0))
    blocks_per_row = in_features // _QK_K
    for row in range(out_features):
        for block_id in range(blocks_per_row):
            start = block_id * _Q6_BLOCK_BYTES
            raw[row, start + 192 : start + 208] = np.roll(
                scales, row + block_id
            ).view(np.uint8)
            raw[row, start + 208 : start + 210] = np.asarray(
                [d_values[(row + block_id) % len(d_values)]], dtype=np.float16
            ).view(np.uint8)
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


def test_q6_f16_rocblas_registry_build_scope_and_workspace_contract() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    q6_f16.register_gguf_q6_k_f16_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)

    assert q6_f16.q6_k_f16_weight_nbytes(12_288, 3_072) == 75_497_472
    assert q6_f16.q6_k_f16_input_nbytes(512, 12_288) == 12_582_912
    assert q6_f16.q6_k_f16_output_nbytes(512, 9_216) == 9_437_184
    assert q6_f16.q6_k_f16_rocblas_session_nbytes(512) == 97_517_568
    assert q6_f16.q6_k_f16_rocblas_workspace_nbytes(17, 256, 72) == (
        (17 * 256 + 256 * 72 + 17 * 72) * 2
    )

    dequant_key = KernelKey(
        "hip_gfx1100", "dequant", "gguf_q6_k", "raw_f16_source_local64"
    )
    assert resolve(
        backend=dequant_key.backend,
        layer=dequant_key.layer,
        quant=dequant_key.quant,
        variant=dequant_key.variant,
    ) is q6_f16.gguf_q6_k_dequantize_f16_source
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
        "gguf_q6_k",
        "raw_f16_bf16_input_source_local64",
    )
    assert resolve(
        backend=fused_key.backend,
        layer=fused_key.layer,
        quant=fused_key.quant,
        variant=fused_key.variant,
    ) is q6_f16.gguf_q6_k_dequantize_bf16_to_f16_source_fused
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            fused_key.layer,
            fused_key.quant,
            fused_key.variant,
        )
    )

    for output_dtype, function in (
        ("bf16", q6_f16.gguf_q6_k_f16_rocblas_bf16_bf16_out),
        ("f32", q6_f16.gguf_q6_k_f16_rocblas_bf16_f32_out),
    ):
        key = KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k",
            f"f16_rocblas_source_bf16_{output_dtype}_out",
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

    artifact = q6_f16.plan_gguf_q6_k_f16_rocblas_prefill_build(
        compiler_version="test"
    )
    assert artifact.output_path.name == "gguf_q6_k_f16_rocblas_prefill.so"
    assert any(
        path.name == "gguf_q6_k_f16_rocblas_prefill.hip"
        for path in artifact.sources
    )
    source = _SOURCE.read_text()
    assert "torch::Tensor" not in source
    assert "__global__ void gguf_q6_k_dequantize_f16_source_kernel" in source


def test_q4_q5_q6_t16_f16_rocblas_registry_and_bounded_workspace_contract() -> None:
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    q6_f16.register_gguf_q6_k_f16_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)

    q4_dequant = getattr(
        q6_f16,
        "gguf_q4_k_t16_dequantize_f16_tile",
        None,
    )
    q5_dequant = getattr(
        q6_f16,
        "gguf_q5_k_t16_dequantize_f16_tile",
        None,
    )
    q5_linear = getattr(
        q6_f16,
        "gguf_q5_k_t16_f16_rocblas_bf16_bf16_out",
        None,
    )
    q4_linear = getattr(
        q6_f16,
        "gguf_q4_k_t16_f16_rocblas_bf16_bf16_out",
        None,
    )
    dequant = getattr(
        q6_f16,
        "gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile",
        None,
    )
    linear = getattr(
        q6_f16,
        "gguf_q6_k_t16_qmicro_planar_f16_rocblas_bf16_bf16_out",
        None,
    )
    workspace_nbytes = getattr(
        q6_f16,
        "q6_k_t16_f16_rocblas_workspace_nbytes",
        None,
    )
    assert callable(q4_dequant)
    assert callable(q4_linear)
    assert callable(q5_dequant)
    assert callable(q5_linear)
    assert callable(dequant)
    assert callable(linear)
    assert callable(workspace_nbytes)
    assert workspace_nbytes(512, 17_408, 5_120, tile_out_features=512) == (
        512 * 17_408 * 2 + 512 * 17_408 * 2 + 512 * 512 * 2
    )
    assert workspace_nbytes(1024, 5_120, 10_240, tile_out_features=512) == (
        1024 * 5_120 * 2 + 512 * 5_120 * 2 + 1024 * 512 * 2
    )

    q4_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q4_k_t16_v1",
        "f16_rocblas_t16_bf16_bf16_out",
    )
    assert resolve(
        backend=q4_key.backend,
        layer=q4_key.layer,
        quant=q4_key.quant,
        variant=q4_key.variant,
    ) is q4_linear
    assert resolve(
        backend="hip_gfx1151",
        layer=q4_key.layer,
        quant=q4_key.quant,
        variant=q4_key.variant,
    ) is q4_linear

    q5_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q5_k_t16_v1",
        "f16_rocblas_t16_bf16_bf16_out",
    )
    assert resolve(
        backend=q5_key.backend,
        layer=q5_key.layer,
        quant=q5_key.quant,
        variant=q5_key.variant,
    ) is q5_linear
    assert resolve(
        backend="hip_gfx1151",
        layer=q5_key.layer,
        quant=q5_key.quant,
        variant=q5_key.variant,
    ) is q5_linear

    for variant, function_name in (
        ("t16_f16_tile_pair_local64", "gguf_q5_k_t16_dequantize_f16_tile_pair"),
        (
            "t16_f16_tile_octet_local256",
            "gguf_q5_k_t16_dequantize_f16_tile_octet",
        ),
    ):
        function = getattr(q6_f16, function_name, None)
        assert callable(function)
        dequant_key = KernelKey(
            "hip_gfx1100", "dequant", "gguf_q5_k_t16_v1", variant
        )
        assert resolve(
            backend=dequant_key.backend,
            layer=dequant_key.layer,
            quant=dequant_key.quant,
            variant=dequant_key.variant,
        ) is function
        assert resolve(
            backend="hip_gfx1151",
            layer=dequant_key.layer,
            quant=dequant_key.quant,
            variant=dequant_key.variant,
        ) is function

    key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "f16_rocblas_t16_qmicro_planar_bf16_bf16_out",
    )
    assert resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    ) is linear
    assert not is_registered(
        KernelKey("hip_gfx1151", key.layer, key.quant, key.variant)
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP/rocBLAS is not available")
def test_q5_t16_dequantizes_to_source_f16_bytes() -> None:
    from hipengine.core.hip import get_hip_runtime
    from tests.test_gguf_k_gemv import make_q5_k_weight

    in_features = 512
    out_features = 32
    raw = make_q5_k_weight(out_features, in_features)
    tiles = repack_gguf_q5_k_tile16(raw[None, ...]).tiles[0]
    expected = np.empty((out_features, in_features), dtype=np.float16)
    for row in range(out_features):
        for block_id in range(in_features // _QK_K):
            block = raw[row, block_id * 176 : (block_id + 1) * 176]
            d = np.float32(block[:2].view(np.float16)[0])
            dmin = np.float32(block[2:4].view(np.float16)[0])
            scales = block[4:16]
            qh = block[16:48]
            qs = block[48:176]
            for k in range(_QK_K):
                subblock = k >> 5
                lane = k & 31
                if subblock < 4:
                    scale = int(scales[subblock] & np.uint8(0x3F))
                    minimum = int(scales[4 + subblock] & np.uint8(0x3F))
                else:
                    index = subblock - 4
                    scale = int(
                        (scales[8 + index] & np.uint8(0x0F))
                        | ((scales[index] >> np.uint8(2)) & np.uint8(0x30))
                    )
                    minimum = int(
                        (scales[8 + index] >> np.uint8(4))
                        | ((scales[4 + index] >> np.uint8(2)) & np.uint8(0x30))
                    )
                packed = qs[(subblock >> 1) * 32 + lane]
                low = int(packed >> np.uint8(4)) if subblock & 1 else int(packed & np.uint8(0x0F))
                high = int((qh[lane] >> np.uint8(subblock)) & np.uint8(1))
                value = np.float32(d * np.float32(scale) * np.float32(low | (high << 4)))
                value = np.float32(value - dmin * np.float32(minimum))
                expected[row, block_id * _QK_K + k] = np.float16(value)
    expected[expected == np.float16(0.0)] = np.float16(0.0)

    actual = np.empty_like(expected)
    runtime = get_hip_runtime()
    library = q6_f16.build_gguf_q6_k_f16_rocblas_prefill(load=True)
    before = memory_stats()
    buffers = []
    try:
        tiles_dev = _device(tiles, runtime)
        actual_dev = malloc(actual.nbytes, runtime=runtime)
        buffers.extend((tiles_dev, actual_dev))
        q6_f16.gguf_q5_k_t16_dequantize_f16_tile(
            tiles_dev.ptr,
            actual_dev.ptr,
            in_features,
            out_features,
            col_start=0,
            col_count=out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(actual), actual_dev, actual.nbytes, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    assert np.array_equal(expected.view(np.uint16), actual.view(np.uint16))
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


@pytest.mark.skipif(not _hip_available(), reason="HIP/rocBLAS is not available")
@pytest.mark.parametrize(
    "function_name",
    [
        "gguf_q5_k_t16_dequantize_f16_tile_pair",
        "gguf_q5_k_t16_dequantize_f16_tile_octet",
    ],
)
def test_q5_t16_packed_column_dequant_matches_scalar_source_f16_bytes(
    function_name: str,
) -> None:
    """Packed-column owners must preserve a nonzero tile's scalar bytes."""

    from hipengine.core.hip import get_hip_runtime
    from tests.test_gguf_k_gemv import make_q5_k_weight

    candidate_dequant = getattr(q6_f16, function_name, None)
    assert callable(candidate_dequant)
    in_features = 512
    out_features = 32
    col_start = 16
    col_count = 16
    raw = make_q5_k_weight(out_features, in_features)
    tiles = repack_gguf_q5_k_tile16(raw[None, ...]).tiles[0]
    scalar = np.empty((col_count, in_features), dtype=np.float16)
    candidate = np.empty_like(scalar)
    runtime = get_hip_runtime()
    library = q6_f16.build_gguf_q6_k_f16_rocblas_prefill(load=True)
    before = memory_stats()
    buffers = []
    try:
        tiles_dev = _device(tiles, runtime)
        scalar_dev = malloc(scalar.nbytes, runtime=runtime)
        candidate_dev = malloc(candidate.nbytes, runtime=runtime)
        buffers.extend((tiles_dev, scalar_dev, candidate_dev))
        q6_f16.gguf_q5_k_t16_dequantize_f16_tile(
            tiles_dev.ptr,
            scalar_dev.ptr,
            in_features,
            out_features,
            col_start=col_start,
            col_count=col_count,
            library=library,
            runtime=runtime,
        )
        candidate_dequant(
            tiles_dev.ptr,
            candidate_dev.ptr,
            in_features,
            out_features,
            col_start=col_start,
            col_count=col_count,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(scalar), scalar_dev, scalar.nbytes, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(candidate),
            candidate_dev,
            candidate.nbytes,
            runtime=runtime,
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    np.testing.assert_array_equal(candidate.view(np.uint16), scalar.view(np.uint16))
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


@pytest.mark.skipif(not _hip_available(), reason="HIP/rocBLAS is not available")
def test_q5_t16_f16_rocblas_passes_cpu_gate() -> None:
    from hipengine.core.hip import get_hip_runtime
    from tests.test_gguf_k_gemv import make_q5_k_weight

    rows = 17
    in_features = 512
    out_features = 32
    tile_out_features = 16
    rng = np.random.default_rng(0x55F16)
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    raw = make_q5_k_weight(out_features, in_features)
    tiles = repack_gguf_q5_k_tile16(raw[None, ...]).tiles[0]
    candidate = np.empty((rows, out_features), dtype=np.uint16)
    runtime = get_hip_runtime()
    library = q6_f16.build_gguf_q6_k_f16_rocblas_prefill(load=True)
    cast_library = build_cast(load=True)
    rocblas = Rocblas.load()
    before = memory_stats()
    buffers = []
    try:
        x_dev = _device(x_bits, runtime)
        tiles_dev = _device(tiles, runtime)
        candidate_dev = malloc(candidate.nbytes, runtime=runtime)
        x_f16_dev = malloc(
            q6_f16.q6_k_f16_input_nbytes(rows, in_features), runtime=runtime
        )
        weight_tile_dev = malloc(
            q6_f16.q6_k_f16_weight_nbytes(in_features, tile_out_features),
            runtime=runtime,
        )
        out_tile_dev = malloc(
            q6_f16.q6_k_f16_output_nbytes(rows, tile_out_features),
            runtime=runtime,
        )
        buffers.extend(
            (
                x_dev,
                tiles_dev,
                candidate_dev,
                x_f16_dev,
                weight_tile_dev,
                out_tile_dev,
            )
        )
        q6_f16.gguf_q5_k_t16_f16_rocblas_bf16_bf16_out(
            x_dev.ptr,
            tiles_dev.ptr,
            candidate_dev.ptr,
            x_f16_dev.ptr,
            weight_tile_dev.ptr,
            out_tile_dev.ptr,
            rows,
            in_features,
            out_features,
            tile_out_features=tile_out_features,
            dequant_library=library,
            cast_library=cast_library,
            rocblas=rocblas,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(candidate),
            candidate_dev,
            candidate.nbytes,
            runtime=runtime,
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        rocblas.close()
    expected = gguf_q5_k_gemv(_bf16_to_f32(x_bits), raw)
    result = evaluate_logits(expected, _bf16_to_f32(candidate))
    assert result.passed, result
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


@pytest.mark.skipif(not _hip_available(), reason="HIP/rocBLAS is not available")
def test_q4_t16_pair_owned_dequant_matches_scalar_source_f16_bytes() -> None:
    """Pair ownership must preserve the complete scalar producer byte stream."""

    from hipengine.core.hip import get_hip_runtime

    pair_dequant = getattr(q6_f16, "gguf_q4_k_t16_dequantize_f16_tile_pair", None)
    assert callable(pair_dequant)
    in_features = 512
    out_features = 32
    col_start = 16
    col_count = 16
    raw = make_q4_k_weight(out_features, in_features)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles[0]
    scalar = np.empty((col_count, in_features), dtype=np.float16)
    pair_owned = np.empty_like(scalar)
    runtime = get_hip_runtime()
    library = q6_f16.build_gguf_q6_k_f16_rocblas_prefill(load=True)
    before = memory_stats()
    buffers = []
    try:
        tiles_dev = _device(tiles, runtime)
        scalar_dev = malloc(scalar.nbytes, runtime=runtime)
        pair_dev = malloc(pair_owned.nbytes, runtime=runtime)
        buffers.extend((tiles_dev, scalar_dev, pair_dev))
        q6_f16.gguf_q4_k_t16_dequantize_f16_tile(
            tiles_dev.ptr,
            scalar_dev.ptr,
            in_features,
            out_features,
            col_start=col_start,
            col_count=col_count,
            library=library,
            runtime=runtime,
        )
        pair_dequant(
            tiles_dev.ptr,
            pair_dev.ptr,
            in_features,
            out_features,
            col_start=col_start,
            col_count=col_count,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(scalar), scalar_dev, scalar.nbytes, runtime=runtime
        )
        copy_device_to_host(
            host_array_ptr(pair_owned),
            pair_dev,
            pair_owned.nbytes,
            runtime=runtime,
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    np.testing.assert_array_equal(
        pair_owned.view(np.uint16), scalar.view(np.uint16)
    )
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


@pytest.mark.skipif(not _hip_available(), reason="HIP/rocBLAS is not available")
def test_q4_t16_f16_rocblas_passes_cpu_gate() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.cpu_reference import gguf_q4_k_gemv

    rows = 17
    in_features = 512
    out_features = 32
    tile_out_features = 16
    rng = np.random.default_rng(0x34F16)
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    raw = make_q4_k_weight(out_features, in_features)
    tiles = repack_gguf_q4_k_tile16(raw[None, ...]).tiles[0]
    candidate = np.empty((rows, out_features), dtype=np.uint16)
    runtime = get_hip_runtime()
    library = q6_f16.build_gguf_q6_k_f16_rocblas_prefill(load=True)
    cast_library = build_cast(load=True)
    rocblas = Rocblas.load()
    before = memory_stats()
    buffers = []
    try:
        x_dev = _device(x_bits, runtime)
        tiles_dev = _device(tiles, runtime)
        candidate_dev = malloc(candidate.nbytes, runtime=runtime)
        x_f16_dev = malloc(
            q6_f16.q6_k_f16_input_nbytes(rows, in_features), runtime=runtime
        )
        weight_tile_dev = malloc(
            q6_f16.q6_k_f16_weight_nbytes(in_features, tile_out_features),
            runtime=runtime,
        )
        out_tile_dev = malloc(
            q6_f16.q6_k_f16_output_nbytes(rows, tile_out_features),
            runtime=runtime,
        )
        buffers.extend(
            (
                x_dev,
                tiles_dev,
                candidate_dev,
                x_f16_dev,
                weight_tile_dev,
                out_tile_dev,
            )
        )
        q6_f16.gguf_q4_k_t16_f16_rocblas_bf16_bf16_out(
            x_dev.ptr,
            tiles_dev.ptr,
            candidate_dev.ptr,
            x_f16_dev.ptr,
            weight_tile_dev.ptr,
            out_tile_dev.ptr,
            rows,
            in_features,
            out_features,
            tile_out_features=tile_out_features,
            dequant_library=library,
            cast_library=cast_library,
            rocblas=rocblas,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(candidate), candidate_dev, candidate.nbytes, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        rocblas.close()
    expected = gguf_q4_k_gemv(_bf16_to_f32(x_bits), raw)
    result = evaluate_logits(expected, _bf16_to_f32(candidate))
    assert result.passed, result
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


@pytest.mark.skipif(not _hip_available(), reason="HIP/rocBLAS is not available")
def test_q6_t16_f16_rocblas_matches_raw_source_route_on_tiled_output() -> None:
    from hipengine.core.hip import get_hip_runtime

    rows = 17
    in_features = 512
    out_features = 32
    rng = np.random.default_rng(0x36F16)
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    raw = make_q6_k_weight(out_features, in_features)
    tiles = repack_gguf_q6_k_tile16_qmicro_planar(raw[None, ...]).tiles[0]
    source = np.empty((rows, out_features), dtype=np.uint16)
    tiled = np.empty_like(source)
    runtime = get_hip_runtime()
    library = q6_f16.build_gguf_q6_k_f16_rocblas_prefill(load=True)
    cast_library = build_cast(load=True)
    rocblas = Rocblas.load()
    before = memory_stats()
    buffers = []
    try:
        x_dev = _device(x_bits, runtime)
        raw_dev = _device(raw, runtime)
        tiles_dev = _device(tiles, runtime)
        source_dev = malloc(source.nbytes, runtime=runtime)
        tiled_dev = malloc(tiled.nbytes, runtime=runtime)
        x_f16_dev = malloc(q6_f16.q6_k_f16_input_nbytes(rows, in_features), runtime=runtime)
        source_weight_dev = malloc(
            q6_f16.q6_k_f16_weight_nbytes(in_features, out_features), runtime=runtime
        )
        source_out_f16_dev = malloc(
            q6_f16.q6_k_f16_output_nbytes(rows, out_features), runtime=runtime
        )
        tiled_weight_dev = malloc(
            q6_f16.q6_k_f16_weight_nbytes(in_features, 16), runtime=runtime
        )
        tiled_out_f16_dev = malloc(
            q6_f16.q6_k_f16_output_nbytes(rows, 16), runtime=runtime
        )
        buffers.extend(
            (
                x_dev,
                raw_dev,
                tiles_dev,
                source_dev,
                tiled_dev,
                x_f16_dev,
                source_weight_dev,
                source_out_f16_dev,
                tiled_weight_dev,
                tiled_out_f16_dev,
            )
        )
        q6_f16.gguf_q6_k_f16_rocblas_bf16_bf16_out(
            x_dev.ptr,
            raw_dev.ptr,
            source_dev.ptr,
            x_f16_dev.ptr,
            source_weight_dev.ptr,
            source_out_f16_dev.ptr,
            rows,
            in_features,
            out_features,
            dequant_library=library,
            cast_library=cast_library,
            rocblas=rocblas,
            runtime=runtime,
        )
        q6_f16.gguf_q6_k_t16_qmicro_planar_f16_rocblas_bf16_bf16_out(
            x_dev.ptr,
            tiles_dev.ptr,
            tiled_dev.ptr,
            x_f16_dev.ptr,
            tiled_weight_dev.ptr,
            tiled_out_f16_dev.ptr,
            rows,
            in_features,
            out_features,
            tile_out_features=16,
            dequant_library=library,
            cast_library=cast_library,
            rocblas=rocblas,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(source), source_dev, source.nbytes, runtime=runtime)
        copy_device_to_host(host_array_ptr(tiled), tiled_dev, tiled.nbytes, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        rocblas.close()
    after = memory_stats()
    np.testing.assert_array_equal(tiled, source)
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


@pytest.mark.skipif(not _hip_available(), reason="HIP/rocBLAS is not available")
@pytest.mark.parametrize(
    ("rows", "in_features", "out_features", "tile_out_features"),
    [(512, 17_408, 5_120, 512), (512, 5_120, 10_240, 512)],
)
def test_q6_t16_f16_rocblas_actual_shapes_use_bounded_workspace_and_pass_quality(
    rows: int,
    in_features: int,
    out_features: int,
    tile_out_features: int,
) -> None:
    """Actual dense-27B shapes: tiled candidate versus production T16 WMMA."""

    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        build_gguf_q6_k_t16_gemv,
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out,
    )

    rng = np.random.default_rng(rows + in_features + out_features)
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    model = next(
        (
            path
            for path in (
                Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf"),
                Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"),
            )
            if path.is_file()
        ),
        None,
    )
    if model is None:
        pytest.skip(
            "local Qwen3.6/Qwen3.8 dense Q4_K_M GGUF fixture is not present"
        )
    reader = __import__(
        "hipengine.loading.gguf", fromlist=["GGUFReader"]
    ).GGUFReader(model)
    tensor_name = (
        "blk.0.ffn_down.weight"
        if (out_features, in_features) == (5_120, 17_408)
        else "blk.0.attn_qkv.weight"
    )
    raw = np.asarray(reader.tensor_data(tensor_name))
    tiles = repack_gguf_q6_k_tile16_qmicro_planar(raw[None, ...]).tiles[0]
    exact = np.empty((rows, out_features), dtype=np.uint16)
    candidate = np.empty_like(exact)
    runtime = get_hip_runtime()
    t16_library = build_gguf_q6_k_t16_gemv(load=True)
    library = q6_f16.build_gguf_q6_k_f16_rocblas_prefill(load=True)
    cast_library = build_cast(load=True)
    rocblas = Rocblas.load()
    before = memory_stats()
    buffers = []
    try:
        x_dev = _device(x_bits, runtime)
        tiles_dev = _device(tiles, runtime)
        exact_dev = malloc(exact.nbytes, runtime=runtime)
        candidate_dev = malloc(candidate.nbytes, runtime=runtime)
        x_f16_dev = malloc(
            q6_f16.q6_k_f16_input_nbytes(rows, in_features), runtime=runtime
        )
        weight_tile_dev = malloc(
            q6_f16.q6_k_f16_weight_nbytes(in_features, tile_out_features),
            runtime=runtime,
        )
        out_tile_dev = malloc(
            q6_f16.q6_k_f16_output_nbytes(rows, tile_out_features), runtime=runtime
        )
        buffers.extend(
            (
                x_dev,
                tiles_dev,
                exact_dev,
                candidate_dev,
                x_f16_dev,
                weight_tile_dev,
                out_tile_dev,
            )
        )
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out(
            x_dev.ptr,
            tiles_dev.ptr,
            exact_dev.ptr,
            rows,
            in_features,
            out_features,
            library=t16_library,
            runtime=runtime,
        )
        q6_f16.gguf_q6_k_t16_qmicro_planar_f16_rocblas_bf16_bf16_out(
            x_dev.ptr,
            tiles_dev.ptr,
            candidate_dev.ptr,
            x_f16_dev.ptr,
            weight_tile_dev.ptr,
            out_tile_dev.ptr,
            rows,
            in_features,
            out_features,
            tile_out_features=tile_out_features,
            dequant_library=library,
            cast_library=cast_library,
            rocblas=rocblas,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(exact), exact_dev, exact.nbytes, runtime=runtime)
        copy_device_to_host(
            host_array_ptr(candidate), candidate_dev, candidate.nbytes, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        rocblas.close()
    after = memory_stats()
    assert np.all(np.isfinite(_bf16_to_f32(candidate)))
    result = evaluate_logits(_bf16_to_f32(exact), _bf16_to_f32(candidate))
    assert result.passed, result
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


@pytest.mark.skipif(not _hip_available(), reason="HIP/rocBLAS is not available")
def test_q6_t16_planar_direct_tile_dequant_matches_independent_source_f16_bytes() -> None:
    """The record-owned direct producer must preserve every source-F16 bit."""

    from hipengine.core.hip import get_hip_runtime

    direct_dequant = getattr(
        q6_f16,
        "gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile_direct",
        None,
    )
    assert callable(direct_dequant)
    in_features = 512
    out_features = 32
    col_start = 16
    col_count = 16
    raw = _edge_q6_weight(out_features=out_features, in_features=in_features)
    tiles = repack_gguf_q6_k_tile16_qmicro_planar(raw[None, ...]).tiles[0]
    expected = _source_q6_f16_cpu(raw, in_features)[
        col_start : col_start + col_count
    ]
    actual = np.empty_like(expected)
    runtime = get_hip_runtime()
    library = q6_f16.build_gguf_q6_k_f16_rocblas_prefill(load=True)
    before = memory_stats()
    buffers = []
    try:
        tiles_dev = _device(tiles, runtime)
        out_dev = malloc(actual.nbytes, runtime=runtime)
        buffers.extend((tiles_dev, out_dev))
        direct_dequant(
            tiles_dev.ptr,
            out_dev.ptr,
            in_features,
            out_features,
            col_start=col_start,
            col_count=col_count,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(actual), out_dev, actual.nbytes, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


@pytest.mark.skipif(not _hip_available(), reason="HIP/rocBLAS is not available")
def test_q6_t16_planar_dequant_matches_independent_source_f16_bytes() -> None:
    from hipengine.core.hip import get_hip_runtime

    in_features = 512
    out_features = 16
    raw = _edge_q6_weight(out_features=out_features, in_features=in_features)
    tiles = repack_gguf_q6_k_tile16_qmicro_planar(raw[None, ...]).tiles[0]
    expected = _source_q6_f16_cpu(raw, in_features)
    actual = np.empty_like(expected)
    runtime = get_hip_runtime()
    library = q6_f16.build_gguf_q6_k_f16_rocblas_prefill(load=True)
    before = memory_stats()
    buffers = []
    try:
        tiles_dev = _device(tiles, runtime)
        out_dev = malloc(actual.nbytes, runtime=runtime)
        buffers.extend((tiles_dev, out_dev))
        q6_f16.gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile(
            tiles_dev.ptr,
            out_dev.ptr,
            in_features,
            out_features,
            col_start=0,
            col_count=out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(actual), out_dev, actual.nbytes, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


def test_q4_t16_f16_rocblas_pair_composite_uses_pair_owned_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[object, ...]] = []

    def capture(*args, **kwargs) -> None:
        captured.append(args)

    pair_composite = getattr(
        q6_f16,
        "gguf_q4_k_t16_f16_rocblas_pair_bf16_bf16_out",
        None,
    )
    assert callable(pair_composite)
    monkeypatch.setattr(q6_f16, "_launch_t16_f16_rocblas", capture)
    pair_composite(1, 2, 3, 4, 5, 6, 512, 17_408, 5_120)
    assert len(captured) == 1
    assert captured[0][0] == "bf16"
    assert captured[0][1] is q6_f16.gguf_q4_k_t16_dequantize_f16_tile_pair


@pytest.mark.parametrize(
    ("composite_name", "producer_name"),
    [
        (
            "gguf_q5_k_t16_f16_rocblas_pair_bf16_bf16_out",
            "gguf_q5_k_t16_dequantize_f16_tile_pair",
        ),
        (
            "gguf_q5_k_t16_f16_rocblas_octet_bf16_bf16_out",
            "gguf_q5_k_t16_dequantize_f16_tile_octet",
        ),
    ],
)
def test_q5_t16_f16_rocblas_packed_column_composite_uses_candidate_producer(
    monkeypatch: pytest.MonkeyPatch,
    composite_name: str,
    producer_name: str,
) -> None:
    captured: list[tuple[object, ...]] = []

    def capture(*args, **kwargs) -> None:
        captured.append(args)

    composite = getattr(q6_f16, composite_name, None)
    producer = getattr(q6_f16, producer_name, None)
    assert callable(composite)
    assert callable(producer)
    monkeypatch.setattr(q6_f16, "_launch_t16_f16_rocblas", capture)
    composite(1, 2, 3, 4, 5, 6, 512, 6_144, 5_120)
    assert len(captured) == 1
    assert captured[0][0] == "bf16"
    assert captured[0][1] is producer


def test_q6_t16_f16_rocblas_composite_uses_record_owned_direct_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[object, ...]] = []

    def capture(*args, **kwargs) -> None:
        captured.append(args)

    monkeypatch.setattr(q6_f16, "_launch_t16_f16_rocblas", capture)
    q6_f16.gguf_q6_k_t16_qmicro_planar_f16_rocblas_bf16_bf16_out(
        1, 2, 3, 4, 5, 6, 512, 5_120, 1_024
    )
    assert len(captured) == 1
    assert captured[0][0] == "bf16"
    assert (
        captured[0][1]
        is q6_f16.gguf_q6_k_t16_qmicro_planar_dequantize_f16_tile_direct
    )


def test_q6_f16_rocblas_rejects_invalid_shapes_before_loading_libraries() -> None:
    with pytest.raises(ValueError, match="multiple of 256"):
        q6_f16.gguf_q6_k_dequantize_f16_source(1, 2, 192, 7)
    with pytest.raises(ValueError, match="rows must be positive"):
        q6_f16.gguf_q6_k_dequantize_bf16_to_f16_source_fused(
            1, 2, 3, 4, 0, 256, 7
        )
    with pytest.raises(ValueError, match="rows must be positive"):
        q6_f16.gguf_q6_k_f16_rocblas_bf16_bf16_out(
            1, 2, 3, 4, 5, 6, 0, 256, 64
        )
    with pytest.raises(ValueError, match="multiple of 256"):
        q6_f16.gguf_q6_k_f16_rocblas_bf16_f32_out(
            1, 2, 3, 4, 5, 6, 17, 384, 64
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP/rocBLAS is not available")
def test_q6_source_dequant_matches_independent_f16_bytes() -> None:
    from hipengine.core.hip import get_hip_runtime

    in_features = 512
    raw = _edge_q6_weight(out_features=5, in_features=in_features)
    expected = _source_q6_f16_cpu(raw, in_features)
    actual = np.empty_like(expected)
    fused_actual = np.empty_like(expected)
    x_bf16 = _bf16_bits(
        np.arange(17 * in_features, dtype=np.float32).reshape(17, in_features)
        / np.float32(257.0)
    )
    expected_x_f16 = _bf16_to_f32(x_bf16).astype(np.float16)
    fused_x_f16 = np.empty_like(expected_x_f16)
    runtime = get_hip_runtime()
    library = q6_f16.build_gguf_q6_k_f16_rocblas_prefill(load=True)
    before = memory_stats()
    buffers = []
    try:
        raw_dev = _device(raw, runtime)
        x_dev = _device(x_bf16, runtime)
        out_dev = malloc(actual.nbytes, runtime=runtime)
        fused_out_dev = malloc(fused_actual.nbytes, runtime=runtime)
        fused_x_dev = malloc(fused_x_f16.nbytes, runtime=runtime)
        buffers.extend((raw_dev, x_dev, out_dev, fused_out_dev, fused_x_dev))
        q6_f16.gguf_q6_k_dequantize_f16_source(
            raw_dev.ptr,
            out_dev.ptr,
            in_features,
            raw.shape[0],
            library=library,
            runtime=runtime,
        )
        q6_f16.gguf_q6_k_dequantize_bf16_to_f16_source_fused(
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
            host_array_ptr(fused_x_f16),
            fused_x_dev,
            fused_x_f16.nbytes,
            runtime=runtime,
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    np.testing.assert_array_equal(
        fused_actual.view(np.uint16), expected.view(np.uint16)
    )
    np.testing.assert_array_equal(
        fused_x_f16.view(np.uint16), expected_x_f16.view(np.uint16)
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
    qweight = make_q6_k_weight(out_features, in_features)
    reference = gguf_q6_k_gemv(_bf16_to_f32(x_bf16), qweight)
    host_dtype = np.uint16 if output_dtype == "bf16" else np.float32
    actual_raw = np.empty((rows, out_features), dtype=host_dtype)
    x_f16_host = np.empty((rows, in_features), dtype=np.float16)

    runtime = get_hip_runtime()
    dequant_library = q6_f16.build_gguf_q6_k_f16_rocblas_prefill(load=True)
    cast_library = build_cast(load=True)
    rocblas = Rocblas.load()
    before = memory_stats()
    buffers = []
    try:
        x_dev = _device(x_bf16, runtime)
        weight_dev = _device(qweight, runtime)
        out_dev = malloc(actual_raw.nbytes, runtime=runtime)
        x_f16_dev = malloc(
            q6_f16.q6_k_f16_input_nbytes(rows, in_features), runtime=runtime
        )
        weight_f16_dev = malloc(
            q6_f16.q6_k_f16_weight_nbytes(in_features, out_features),
            runtime=runtime,
        )
        out_f16_dev = malloc(
            q6_f16.q6_k_f16_output_nbytes(rows, out_features), runtime=runtime
        )
        buffers.extend(
            (
                x_dev,
                weight_dev,
                out_dev,
                x_f16_dev,
                weight_f16_dev,
                out_f16_dev,
            )
        )
        function = (
            q6_f16.gguf_q6_k_f16_rocblas_bf16_bf16_out
            if output_dtype == "bf16"
            else q6_f16.gguf_q6_k_f16_rocblas_bf16_f32_out
        )
        function(
            x_dev.ptr,
            weight_dev.ptr,
            out_dev.ptr,
            x_f16_dev.ptr,
            weight_f16_dev.ptr,
            out_f16_dev.ptr,
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
            host_array_ptr(x_f16_host),
            x_f16_dev,
            x_f16_host.nbytes,
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
    return actual, reference, x_f16_host


@pytest.mark.skipif(not _hip_available(), reason="HIP/rocBLAS is not available")
@pytest.mark.parametrize(
    ("rows", "in_features", "out_features", "output_dtype"),
    [(17, 256, 72, "f32"), (32, 512, 128, "bf16")],
)
def test_q6_f16_rocblas_outputs_are_finite_and_pass_exact_path_quality(
    rows: int,
    in_features: int,
    out_features: int,
    output_dtype: str,
) -> None:
    actual, reference, x_f16 = _run_candidate(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        output_dtype=output_dtype,
    )
    rng = np.random.default_rng(rows * 97 + in_features + out_features)
    x = rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    expected_x_f16 = _bf16_to_f32(_bf16_bits(x)).astype(np.float16)
    np.testing.assert_array_equal(x_f16.view(np.uint16), expected_x_f16.view(np.uint16))
    assert np.all(np.isfinite(actual))
    result = evaluate_logits(reference, actual)
    assert result.kl_mean <= 0.05, result
    assert result.top1_agreement >= 0.90, result
    assert result.passed, result
