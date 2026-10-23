"""Real-device cursor publication and rollback for compact Qwen4Exp metadata."""

import ctypes
from types import MappingProxyType

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpDecodeState,
    Qwen4ExpDenseAttentionState,
    Qwen4ExpGGUFResidentModelRunner,
    Qwen4ExpQSAIndexDeviceState,
)
from tests._rocm_guard import hip_runtime_available


@pytest.fixture
def runtime():
    if not hip_runtime_available():
        pytest.skip("HIP runtime unavailable")
    result = get_hip_runtime()
    count = ctypes.c_int()
    if result.library.hipGetDeviceCount(ctypes.byref(count)) != 0 or count.value == 0:
        pytest.skip("HIP device unavailable")
    return result


def download(buffer, runtime, dtype=np.uint8):
    host = np.empty(buffer.nbytes // np.dtype(dtype).itemsize, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), buffer, runtime=runtime)
    return host


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("depth", [0, 17, 63])
def test_cursor_upload_and_transaction_rollback_match_device_state(runtime, compact, depth):
    runner = object.__new__(Qwen4ExpGGUFResidentModelRunner)
    runner.closed = False
    runner.runtime = runtime
    runner.position = 0
    runner.max_sequence_length = 64
    runner._ple_hash_states = {}
    runner._device_transaction_snapshot = None
    runner._device_transaction_lease = False
    runner.attention_states = []
    runner.index_states = []
    runner._shared_position_context = malloc(12 * 16, runtime=runtime) if compact else None
    runner._shared_position_context_host = np.zeros(24, dtype=np.int64) if compact else None
    runner.state = Qwen4ExpDecodeState.allocate(
        gdn_layers=2, gdn_value_heads=2, gdn_head_dim=4,
        gdn_conv_channels=8, gdn_conv_kernel=4, residual_branches=2,
        hidden=8, ple_conv_kernel=3, ple_dilation=1, runtime=runtime,
    )
    try:
        for index in range(12):
            views = {} if not compact else {
                "position_view": DeviceBuffer(runner._shared_position_context.ptr + 16 * index, 8),
                "context_view": DeviceBuffer(runner._shared_position_context.ptr + 16 * index + 8, 8),
            }
            attention = Qwen4ExpDenseAttentionState.allocate(
                max_positions=64, block_size=256, kv_heads=2, head_dim=4,
                runtime=runtime, **views,
            )
            runner.attention_states.append(attention)
            runner.index_states.append(Qwen4ExpQSAIndexDeviceState.allocate(
                attention_state=attention, index_heads=2, index_dim=4,
                compression_ratio=4, block_budget=2, runtime=runtime,
            ))
        originals = {}
        for index, (name, buffer) in enumerate(runner.state.owned_buffers.items()):
            originals[name] = (np.arange(buffer.nbytes, dtype=np.uint8) + index).astype(np.uint8)
            copy_host_to_device(buffer, host_array_ptr(originals[name]), runtime=runtime)
        hashes = MappingProxyType({7: (1, 2, 3)})
        runner.adopt_prefix_cursor(depth, hashes)
        for index in runner.index_states:
            index.pooled_count = depth // 4
        checkpoint = runner.begin_device_transaction(reuse_snapshot=True)
        for buffer in runner.state.owned_buffers.values():
            runtime.memset(buffer.ptr, 0xDC, buffer.nbytes)
        runner.adopt_prefix_cursor(depth + 1, {})
        for index in runner.index_states:
            index.pooled_count = (depth + 1) // 4
        runner.rollback_device_transaction(checkpoint)
        assert checkpoint.rolled_back and not runner._device_transaction_lease
        assert runner.position == depth and runner.ple_hash_states == hashes
        cursor = max(depth - 1, 0)
        for attention, index in zip(runner.attention_states, runner.index_states):
            assert download(attention.position, runtime, np.int64).tolist() == [cursor]
            assert download(attention.context, runtime, np.int64).tolist() == [cursor + 1]
            assert attention.position_host.tolist() == [cursor]
            assert attention.context_host.tolist() == [cursor + 1]
            assert index.count == depth and index.pooled_count == depth // 4
        for name, buffer in runner.state.owned_buffers.items():
            np.testing.assert_array_equal(download(buffer, runtime), originals[name])
        # A second lease must be usable after rollback, and commit keeps state.
        checkpoint = runner.begin_device_transaction(reuse_snapshot=True)
        runner.commit_device_transaction(checkpoint)
        assert checkpoint.committed and not runner._device_transaction_lease
    finally:
        if runner._device_transaction_snapshot is not None:
            runner._device_transaction_snapshot.close()
        for index in reversed(runner.index_states):
            index.close()
        for attention in reversed(runner.attention_states):
            attention.close()
        if runner._shared_position_context is not None:
            free(runner._shared_position_context, runtime=runtime)
        runner.state.close()
