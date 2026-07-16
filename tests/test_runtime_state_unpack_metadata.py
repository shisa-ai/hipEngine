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


def test_prepare_prefill_chunk_metadata_matches_reference() -> None:
    from hipengine.core.dtype import DType
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.runtime import build_runtime_state, prepare_prefill_chunk_metadata

    runtime = get_hip_runtime()
    start = 127 * 1024
    rows = 1024
    buffers = [
        malloc(2 * DType.INT32.itemsize, runtime=runtime),
        malloc(2 * DType.INT32.itemsize, runtime=runtime),
        malloc(DType.INT32.itemsize, runtime=runtime),
        malloc(2 * DType.INT32.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
    ]
    try:
        cu_q, cu_k, atomic, gdn_cu, positions, contexts = buffers
        prepare_prefill_chunk_metadata(
            cu_q.ptr,
            cu_k.ptr,
            atomic.ptr,
            gdn_cu.ptr,
            positions.ptr,
            contexts.ptr,
            start,
            rows,
            library=build_runtime_state(load=True),
            runtime=runtime,
        )
        runtime.device_synchronize()

        outputs = [
            np.empty(2, dtype=np.int32),
            np.empty(2, dtype=np.int32),
            np.empty(1, dtype=np.int32),
            np.empty(2, dtype=np.int32),
            np.empty(rows, dtype=np.int64),
            np.empty(rows, dtype=np.int64),
        ]
        for host, buffer in zip(outputs, buffers, strict=True):
            copy_device_to_host(
                host_array_ptr(host),
                DeviceBuffer(buffer.ptr, host.nbytes),
                runtime=runtime,
            )

        np.testing.assert_array_equal(outputs[0], np.asarray([0, rows], dtype=np.int32))
        np.testing.assert_array_equal(outputs[1], np.asarray([0, start + rows], dtype=np.int32))
        np.testing.assert_array_equal(outputs[2], np.asarray([0], dtype=np.int32))
        np.testing.assert_array_equal(outputs[3], np.asarray([0, rows], dtype=np.int32))
        np.testing.assert_array_equal(outputs[4], np.arange(start, start + rows, dtype=np.int64))
        np.testing.assert_array_equal(outputs[5], np.arange(start + 1, start + rows + 1, dtype=np.int64))
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)


def test_prepare_packed_decode_metadata_matches_ragged_c4_reference() -> None:
    from hipengine.core.dtype import DType
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.runtime import (
        build_runtime_state,
        prepare_packed_decode_metadata,
    )

    runtime = get_hip_runtime()
    positions_host = (513, 517, 521, 525)
    rows = len(positions_host)
    blocks_per_slot = 4
    buffers = [
        malloc(rows * blocks_per_slot * DType.INT32.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
        malloc(2 * DType.INT32.itemsize, runtime=runtime),
        malloc(2 * DType.INT32.itemsize, runtime=runtime),
        malloc(DType.INT32.itemsize, runtime=runtime),
        malloc((rows + 1) * DType.INT32.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
    ]
    try:
        block_table, positions, contexts, cu_q, cu_k, atomic, gdn_cu, state_indices = buffers
        prepare_packed_decode_metadata(
            block_table.ptr,
            positions.ptr,
            contexts.ptr,
            cu_q.ptr,
            cu_k.ptr,
            atomic.ptr,
            gdn_cu.ptr,
            state_indices.ptr,
            positions_host,
            blocks_per_slot,
            library=build_runtime_state(load=True),
            runtime=runtime,
        )
        runtime.device_synchronize()

        outputs = [
            np.empty((rows, blocks_per_slot), dtype=np.int32),
            np.empty(rows, dtype=np.int64),
            np.empty(rows, dtype=np.int64),
            np.empty(2, dtype=np.int32),
            np.empty(2, dtype=np.int32),
            np.empty(1, dtype=np.int32),
            np.empty(rows + 1, dtype=np.int32),
            np.empty(rows, dtype=np.int64),
        ]
        for host, buffer in zip(outputs, buffers, strict=True):
            copy_device_to_host(
                host_array_ptr(host),
                DeviceBuffer(buffer.ptr, host.nbytes),
                runtime=runtime,
            )

        np.testing.assert_array_equal(
            outputs[0],
            np.arange(rows * blocks_per_slot, dtype=np.int32).reshape(
                rows, blocks_per_slot
            ),
        )
        np.testing.assert_array_equal(outputs[1], np.asarray(positions_host, dtype=np.int64))
        np.testing.assert_array_equal(outputs[2], np.asarray(positions_host, dtype=np.int64) + 1)
        np.testing.assert_array_equal(outputs[3], np.asarray([0, rows], dtype=np.int32))
        np.testing.assert_array_equal(outputs[4], np.asarray([0, max(positions_host) + 1], dtype=np.int32))
        np.testing.assert_array_equal(outputs[5], np.asarray([0], dtype=np.int32))
        np.testing.assert_array_equal(outputs[6], np.arange(rows + 1, dtype=np.int32))
        np.testing.assert_array_equal(outputs[7], np.arange(rows, dtype=np.int64))
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)


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
