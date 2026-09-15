"""The low-row WMMA twin must be a scheduling change, not a numerical one.

``dense_prefill_wmma_out_bf16_m64`` narrows the column tile from 128 to 64 so the
semantic encoder's deep stages -- which collapse to a handful of rows while their
width grows to 2048 -- put twice as many workgroups in flight. Per-element
accumulation is k0-major then kk-major and does not depend on the column tile, so
every output must be bit-identical to the BM=128 path. That equality is the whole
contract: if it ever fails, the variant is a numerical change and has to clear the
generated-audio quality gate instead of being treated as free scheduling.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import host_array_ptr
from hipengine.core.runtime import MemcpyKind
from hipengine.kernels.hip_gfx1100.linear import dense_gemv


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + ((bits >> 16) & 1) + 0x7FFF) >> 16
    return rounded.astype(np.uint16)


def _upload(array: np.ndarray):
    array = np.ascontiguousarray(array, dtype=np.uint16)
    runtime = get_hip_runtime()
    buffer = runtime.malloc(array.nbytes)
    runtime.memcpy(buffer, host_array_ptr(array), array.nbytes, MemcpyKind.HOST_TO_DEVICE)
    return buffer


# The shapes the semantic encoder actually issues, plus a wide-K case.
_SHAPES = (
    (1, 2048, 8192),
    (1, 8192, 2048),
    (1, 16384, 2048),
    (8, 1024, 4096),
    (8, 4096, 1024),
    (40, 2048, 512),
)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows,in_features,out_features", _SHAPES)
def test_low_row_wmma_twin_is_bit_identical(rows, in_features, out_features) -> None:
    library = dense_gemv._dense_gemv_library()
    runtime = get_hip_runtime()
    rng = np.random.default_rng(20260915)
    x = _bf16(rng.standard_normal((rows, in_features)) * 0.3)
    weight = _bf16(rng.standard_normal((out_features, in_features)) * 0.3)
    x_dev, weight_dev = _upload(x), _upload(weight)
    wide_out, narrow_out = runtime.malloc(rows * out_features * 2), runtime.malloc(rows * out_features * 2)
    try:
        dense_gemv.dense_prefill_wmma_out_bf16(
            x_dev, weight_dev, wide_out, rows, in_features, out_features, library=library, runtime=runtime
        )
        dense_gemv.dense_prefill_wmma_out_bf16_m64(
            x_dev, weight_dev, narrow_out, rows, in_features, out_features, library=library, runtime=runtime
        )
        runtime.device_synchronize()
        wide = np.empty((rows, out_features), dtype=np.uint16)
        narrow = np.empty((rows, out_features), dtype=np.uint16)
        runtime.memcpy(host_array_ptr(wide), wide_out, wide.nbytes, MemcpyKind.DEVICE_TO_HOST)
        runtime.memcpy(host_array_ptr(narrow), narrow_out, narrow.nbytes, MemcpyKind.DEVICE_TO_HOST)
    finally:
        for buffer in (x_dev, weight_dev, wide_out, narrow_out):
            runtime.free(buffer)

    assert np.array_equal(wide, narrow), (
        f"{int((wide != narrow).sum())} of {wide.size} outputs differ; the narrow column tile "
        "is not the pure scheduling change it is documented to be"
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_low_row_wmma_twin_rejects_shapes_it_cannot_tile() -> None:
    library = dense_gemv._dense_gemv_library()
    runtime = get_hip_runtime()
    buffer = runtime.malloc(4096)
    try:
        with pytest.raises(ValueError):
            dense_gemv.dense_prefill_wmma_out_bf16_m64(buffer, buffer, buffer, 1, 64, 32, library=library, runtime=runtime)
        with pytest.raises(ValueError):
            dense_gemv.dense_prefill_wmma_out_bf16_m64(buffer, buffer, buffer, 1, 48, 64, library=library, runtime=runtime)
        with pytest.raises(ValueError):
            dense_gemv.dense_prefill_wmma_out_bf16_m64(buffer, buffer, buffer, 0, 64, 64, library=library, runtime=runtime)
    finally:
        runtime.free(buffer)
