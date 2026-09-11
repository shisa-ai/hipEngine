from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
)
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpDecodeState


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_decode_state_snapshot_restore_reset_and_close() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    state = Qwen4ExpDecodeState.allocate(
        gdn_layers=2,
        gdn_value_heads=3,
        gdn_head_dim=4,
        gdn_conv_channels=5,
        gdn_conv_kernel=4,
        residual_branches=2,
        hidden=4,
        ple_conv_kernel=4,
        ple_dilation=3,
        runtime=runtime,
    )
    expected_sizes = {
        "gdn_matrix": 2 * 3 * 4 * 4 * 4,
        "gdn_conv": 2 * 4 * 5 * 4,
        "ple_conv": 9 * 8 * 4,
        "residual": 2 * 4 * 2,
    }
    assert state.nbytes_by_owner == expected_sizes

    values = {}
    for index, (name, buffer) in enumerate(state.owned_buffers.items(), 1):
        value = np.full(buffer.nbytes, index, dtype=np.uint8)
        values[name] = value
        copy_host_to_device(buffer, host_array_ptr(value), runtime=runtime)
    snapshot = state.snapshot()
    device_snapshot = state.device_snapshot()
    assert device_snapshot.nbytes_by_owner == expected_sizes
    state.zero()
    for buffer in state.owned_buffers.values():
        actual = np.empty(buffer.nbytes, dtype=np.uint8)
        copy_device_to_host(host_array_ptr(actual), buffer, runtime=runtime)
        np.testing.assert_array_equal(actual, 0)
    state.restore(snapshot)
    for name, buffer in state.owned_buffers.items():
        actual = np.empty(buffer.nbytes, dtype=np.uint8)
        copy_device_to_host(host_array_ptr(actual), buffer, runtime=runtime)
        np.testing.assert_array_equal(actual, values[name])

    state.zero()
    state.restore_device_snapshot(device_snapshot)
    for name, buffer in state.owned_buffers.items():
        actual = np.empty(buffer.nbytes, dtype=np.uint8)
        copy_device_to_host(host_array_ptr(actual), buffer, runtime=runtime)
        np.testing.assert_array_equal(actual, values[name])
    device_snapshot.close()
    device_snapshot.close()
    assert device_snapshot.closed

    state.close()
    state.close()
    assert state.closed
    with pytest.raises(RuntimeError, match="closed"):
        state.snapshot()
