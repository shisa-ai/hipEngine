from __future__ import annotations

import ctypes

import numpy as np
import pytest


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")


def test_unpack_verify_chain_dynamic_metadata_matches_reference() -> None:
    from hipengine.core.dtype import DType
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.runtime import build_runtime_state, unpack_verify_chain_dynamic_metadata_i64

    runtime = get_hip_runtime()
    rows = 5
    tokens = np.asarray([1, 250000, 17, 42, 999], dtype=np.int64)
    positions = np.asarray([0, 31, 1024, 4095, 127999], dtype=np.int64)
    packed = np.empty((rows, 5), dtype=np.int64)
    packed[:, 0] = tokens
    packed[:, 1] = tokens
    packed[:, 2] = positions
    packed[:, 3] = positions
    packed[:, 4] = positions + 1

    buffers = [
        malloc(packed.nbytes, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
        malloc(rows * DType.INT32.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
        malloc(rows * DType.INT32.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
    ]
    try:
        packed_buf, token_i64_buf, token_i32_buf, pos_i64_buf, pos_i32_buf, context_i64_buf = buffers
        copy_host_to_device(packed_buf, host_array_ptr(packed), packed.nbytes, runtime=runtime)
        unpack_verify_chain_dynamic_metadata_i64(
            packed_buf.ptr,
            token_i64_buf.ptr,
            token_i32_buf.ptr,
            pos_i64_buf.ptr,
            pos_i32_buf.ptr,
            context_i64_buf.ptr,
            rows,
            library=build_runtime_state(load=True),
            runtime=runtime,
        )
        runtime.device_synchronize()

        out_token_i64 = np.empty(rows, dtype=np.int64)
        out_token_i32 = np.empty(rows, dtype=np.int32)
        out_pos_i64 = np.empty(rows, dtype=np.int64)
        out_pos_i32 = np.empty(rows, dtype=np.int32)
        out_context_i64 = np.empty(rows, dtype=np.int64)
        copy_device_to_host(host_array_ptr(out_token_i64), DeviceBuffer(token_i64_buf.ptr, out_token_i64.nbytes), runtime=runtime)
        copy_device_to_host(host_array_ptr(out_token_i32), DeviceBuffer(token_i32_buf.ptr, out_token_i32.nbytes), runtime=runtime)
        copy_device_to_host(host_array_ptr(out_pos_i64), DeviceBuffer(pos_i64_buf.ptr, out_pos_i64.nbytes), runtime=runtime)
        copy_device_to_host(host_array_ptr(out_pos_i32), DeviceBuffer(pos_i32_buf.ptr, out_pos_i32.nbytes), runtime=runtime)
        copy_device_to_host(host_array_ptr(out_context_i64), DeviceBuffer(context_i64_buf.ptr, out_context_i64.nbytes), runtime=runtime)

        np.testing.assert_array_equal(out_token_i64, tokens)
        np.testing.assert_array_equal(out_token_i32, tokens.astype(np.int32))
        np.testing.assert_array_equal(out_pos_i64, positions)
        np.testing.assert_array_equal(out_pos_i32, positions.astype(np.int32))
        np.testing.assert_array_equal(out_context_i64, positions + 1)
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)
