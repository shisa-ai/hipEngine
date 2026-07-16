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


def test_packed_decode_graph_control_kernels_match_two_step_reference() -> None:
    from hipengine.core.dtype import DType
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.runtime import (
        build_runtime_state,
        commit_packed_decode_graph_step,
        prepare_packed_decode_metadata_from_positions,
        record_u16_rows_indexed,
    )

    runtime = get_hip_runtime()
    library = build_runtime_state(load=True)
    rows = 8
    blocks_per_slot = 4
    record_steps = 2
    hidden_elements = 8
    hidden_layers = 2
    hidden_step_stride = hidden_layers * hidden_elements
    positions_host = np.asarray([513, 517, 521, 525, 529, 533, 537, 541], dtype=np.int64)
    token_steps = (
        np.asarray([11, 22, 33, 44, 55, 66, 77, 88], dtype=np.int32),
        np.asarray([12, 23, 34, 45, 56, 67, 78, 89], dtype=np.int32),
    )
    hidden_steps = (
        np.arange(100, 100 + hidden_elements, dtype=np.uint16),
        np.arange(200, 200 + hidden_elements, dtype=np.uint16),
    )
    buffers = [
        malloc(rows * blocks_per_slot * DType.INT32.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
        malloc(2 * DType.INT32.itemsize, runtime=runtime),
        malloc(2 * DType.INT32.itemsize, runtime=runtime),
        malloc(DType.INT32.itemsize, runtime=runtime),
        malloc((rows + 1) * DType.INT32.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
        malloc(rows * DType.INT32.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
        malloc(record_steps * rows * DType.INT32.itemsize, runtime=runtime),
        malloc(DType.INT64.itemsize, runtime=runtime),
        malloc(hidden_elements * DType.BF16.itemsize, runtime=runtime),
        malloc(record_steps * hidden_step_stride * DType.BF16.itemsize, runtime=runtime),
    ]
    try:
        (
            block_table,
            positions,
            contexts,
            cu_q,
            cu_k,
            atomic,
            gdn_cu,
            state_indices,
            token_i32,
            token_i64,
            recorded_tokens,
            record_index,
            hidden,
            recorded_hidden,
        ) = buffers
        copy_host_to_device(positions, host_array_ptr(positions_host), positions_host.nbytes, runtime=runtime)
        zero_index = np.zeros((1,), dtype=np.int64)
        copy_host_to_device(record_index, host_array_ptr(zero_index), zero_index.nbytes, runtime=runtime)

        for step, (tokens, hidden_values) in enumerate(zip(token_steps, hidden_steps, strict=True)):
            copy_host_to_device(token_i32, host_array_ptr(tokens), tokens.nbytes, runtime=runtime)
            copy_host_to_device(hidden, host_array_ptr(hidden_values), hidden_values.nbytes, runtime=runtime)
            prepare_packed_decode_metadata_from_positions(
                block_table.ptr,
                positions.ptr,
                contexts.ptr,
                cu_q.ptr,
                cu_k.ptr,
                atomic.ptr,
                gdn_cu.ptr,
                state_indices.ptr,
                rows,
                blocks_per_slot,
                library=library,
                runtime=runtime,
            )
            for layer_id in range(hidden_layers):
                record_u16_rows_indexed(
                    hidden.ptr,
                    recorded_hidden.ptr + layer_id * hidden_elements * DType.BF16.itemsize,
                    record_index.ptr,
                    hidden_elements,
                    hidden_step_stride,
                    record_steps,
                    library=library,
                    runtime=runtime,
                )
            commit_packed_decode_graph_step(
                token_i32.ptr,
                token_i64.ptr,
                positions.ptr,
                contexts.ptr,
                rows,
                recorded_token_ids_i32_ptr=recorded_tokens.ptr,
                record_index_i64_ptr=record_index.ptr,
                record_capacity=record_steps,
                library=library,
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
            np.empty(rows, dtype=np.int64),
            np.empty((record_steps, rows), dtype=np.int32),
            np.empty(1, dtype=np.int64),
            np.empty((record_steps, hidden_layers, hidden_elements), dtype=np.uint16),
        ]
        output_buffers = (
            block_table,
            positions,
            contexts,
            cu_q,
            cu_k,
            atomic,
            gdn_cu,
            state_indices,
            token_i64,
            recorded_tokens,
            record_index,
            recorded_hidden,
        )
        for host, buffer in zip(outputs, output_buffers, strict=True):
            copy_device_to_host(
                host_array_ptr(host),
                DeviceBuffer(buffer.ptr, host.nbytes),
                runtime=runtime,
            )

        np.testing.assert_array_equal(outputs[0], np.arange(32, dtype=np.int32).reshape(8, 4))
        np.testing.assert_array_equal(outputs[1], positions_host + 2)
        np.testing.assert_array_equal(outputs[2], positions_host + 3)
        np.testing.assert_array_equal(outputs[3], np.asarray([0, rows], dtype=np.int32))
        np.testing.assert_array_equal(outputs[4], np.asarray([0, 543], dtype=np.int32))
        np.testing.assert_array_equal(outputs[5], np.asarray([0], dtype=np.int32))
        np.testing.assert_array_equal(outputs[6], np.arange(rows + 1, dtype=np.int32))
        np.testing.assert_array_equal(outputs[7], np.arange(rows, dtype=np.int64))
        np.testing.assert_array_equal(outputs[8], token_steps[-1].astype(np.int64))
        np.testing.assert_array_equal(outputs[9], np.stack(token_steps))
        np.testing.assert_array_equal(outputs[10], np.asarray([2], dtype=np.int64))
        expected_hidden = np.stack(
            [np.stack([values, values]) for values in hidden_steps]
        )
        np.testing.assert_array_equal(outputs[11], expected_hidden)
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)


def test_packed_decode_graph_control_kernels_keep_masked_c8_lanes_inert() -> None:
    from hipengine.core.dtype import DType
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.runtime import (
        build_runtime_state,
        commit_packed_decode_graph_step,
        prepare_packed_decode_metadata_from_positions,
    )

    runtime = get_hip_runtime()
    library = build_runtime_state(load=True)
    rows = 8
    blocks_per_slot = 4
    record_steps = 2
    positions_host = np.asarray([513, -1, 521, -1, -1, 533, -1, 541], dtype=np.int64)
    active_mask = np.asarray([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
    token_steps = (
        np.asarray([11, 0, 33, 0, 0, 66, 0, 88], dtype=np.int32),
        np.asarray([12, 0, 34, 0, 0, 67, 0, 89], dtype=np.int32),
    )
    token_i64_initial = np.full((rows,), -7, dtype=np.int64)
    recorded_initial = np.full((record_steps, rows), -1, dtype=np.int32)
    buffers = [
        malloc(rows * blocks_per_slot * DType.INT32.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
        malloc(2 * DType.INT32.itemsize, runtime=runtime),
        malloc(2 * DType.INT32.itemsize, runtime=runtime),
        malloc(DType.INT32.itemsize, runtime=runtime),
        malloc((rows + 1) * DType.INT32.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
        malloc(active_mask.nbytes, runtime=runtime),
        malloc(rows * DType.INT32.itemsize, runtime=runtime),
        malloc(rows * DType.INT64.itemsize, runtime=runtime),
        malloc(recorded_initial.nbytes, runtime=runtime),
        malloc(DType.INT64.itemsize, runtime=runtime),
    ]
    try:
        (
            block_table,
            positions,
            contexts,
            cu_q,
            cu_k,
            atomic,
            gdn_cu,
            state_indices,
            mask_device,
            token_i32,
            token_i64,
            recorded_tokens,
            record_index,
        ) = buffers
        copy_host_to_device(positions, host_array_ptr(positions_host), runtime=runtime)
        copy_host_to_device(mask_device, host_array_ptr(active_mask), runtime=runtime)
        copy_host_to_device(token_i64, host_array_ptr(token_i64_initial), runtime=runtime)
        copy_host_to_device(recorded_tokens, host_array_ptr(recorded_initial), runtime=runtime)
        zero_index = np.zeros((1,), dtype=np.int64)
        copy_host_to_device(record_index, host_array_ptr(zero_index), runtime=runtime)

        for tokens in token_steps:
            copy_host_to_device(token_i32, host_array_ptr(tokens), runtime=runtime)
            prepare_packed_decode_metadata_from_positions(
                block_table.ptr,
                positions.ptr,
                contexts.ptr,
                cu_q.ptr,
                cu_k.ptr,
                atomic.ptr,
                gdn_cu.ptr,
                state_indices.ptr,
                rows,
                blocks_per_slot,
                active_mask_u8_ptr=mask_device.ptr,
                library=library,
                runtime=runtime,
            )
            commit_packed_decode_graph_step(
                token_i32.ptr,
                token_i64.ptr,
                positions.ptr,
                contexts.ptr,
                rows,
                active_mask_u8_ptr=mask_device.ptr,
                recorded_token_ids_i32_ptr=recorded_tokens.ptr,
                record_index_i64_ptr=record_index.ptr,
                record_capacity=record_steps,
                library=library,
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
            np.empty(rows, dtype=np.int64),
            np.empty((record_steps, rows), dtype=np.int32),
            np.empty(1, dtype=np.int64),
        ]
        output_buffers = (
            block_table,
            positions,
            contexts,
            cu_q,
            cu_k,
            atomic,
            gdn_cu,
            state_indices,
            token_i64,
            recorded_tokens,
            record_index,
        )
        for host, buffer in zip(outputs, output_buffers, strict=True):
            copy_device_to_host(
                host_array_ptr(host),
                DeviceBuffer(buffer.ptr, host.nbytes),
                runtime=runtime,
            )

        expected_blocks = np.arange(32, dtype=np.int32).reshape(8, 4)
        expected_blocks[active_mask == 0] = -1
        expected_positions = positions_host.copy()
        expected_positions[active_mask != 0] += 2
        expected_contexts = np.zeros((rows,), dtype=np.int64)
        expected_contexts[active_mask != 0] = positions_host[active_mask != 0] + 3
        expected_token_i64 = token_i64_initial.copy()
        expected_token_i64[active_mask != 0] = token_steps[-1][active_mask != 0]
        expected_recorded = recorded_initial.copy()
        for step, tokens in enumerate(token_steps):
            expected_recorded[step, active_mask != 0] = tokens[active_mask != 0]

        np.testing.assert_array_equal(outputs[0], expected_blocks)
        np.testing.assert_array_equal(outputs[1], expected_positions)
        np.testing.assert_array_equal(outputs[2], expected_contexts)
        np.testing.assert_array_equal(outputs[3], np.asarray([0, rows], dtype=np.int32))
        np.testing.assert_array_equal(outputs[4], np.asarray([0, 543], dtype=np.int32))
        np.testing.assert_array_equal(outputs[5], np.asarray([0], dtype=np.int32))
        np.testing.assert_array_equal(outputs[6], np.arange(rows + 1, dtype=np.int32))
        np.testing.assert_array_equal(outputs[7], np.arange(rows, dtype=np.int64))
        np.testing.assert_array_equal(outputs[8], expected_token_i64)
        np.testing.assert_array_equal(outputs[9], expected_recorded)
        np.testing.assert_array_equal(outputs[10], np.asarray([2], dtype=np.int64))
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
