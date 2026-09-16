"""Exact H2D bytes/ownership on each physical GPU; no kernels or model load."""
import ctypes

import numpy as np
import pytest

from hipengine.core.device import scoped_current_device
from hipengine.core import memory
from hipengine.core.hip import get_hip_runtime
from hipengine.core.runtime import MemcpyKind


@pytest.mark.parametrize('device', [0, 1])
def test_resident_derived_view_upload_on_each_device(device):
    try:
        ctypes.CDLL('libamdhip64.so')
        runtime = get_hip_runtime()
        count = runtime.device_count()
    except (OSError, RuntimeError) as error:
        pytest.skip(f'HIP unavailable: {error}')
    if device >= count:
        pytest.skip(f'device {device} unavailable')
    source = np.array([248045, 846, 198], dtype=np.int64)
    expected = np.full(source.nbytes + 16, 0xA5, dtype=np.uint8)
    expected[8:-8] = source.view(np.uint8)
    before = runtime.get_device()
    with scoped_current_device(runtime, device):
        owner = memory.malloc(expected.nbytes, runtime=runtime)
        poison = np.full_like(expected, 0xA5)
        runtime.memcpy(owner.ptr, poison.ctypes.data, poison.nbytes, MemcpyKind.HOST_TO_DEVICE)
        view = memory.DeviceBuffer(owner.ptr + 8, source.nbytes)
        memory.copy_host_array_to_device(view, source, runtime=runtime)
        runtime.device_synchronize()
        actual = np.empty_like(expected)
        runtime.memcpy(actual.ctypes.data, owner.ptr, actual.nbytes, MemcpyKind.DEVICE_TO_HOST)
        memory.free(owner, runtime=runtime)
        np.testing.assert_array_equal(actual, expected)
    assert runtime.get_device() == before
