"""Real-device coverage for the synchronous array-owning upload API."""
import ctypes

import numpy as np
import pytest


def test_temporary_upload_and_source_reuse():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("ROCm/HIP runtime unavailable")
    from hipengine.core.memory import (
        copy_device_to_host, copy_host_array_to_device, free, host_array_ptr, malloc,
    )

    expected = np.arange(262144, dtype=np.int32)
    buffer = malloc(expected.nbytes + 512)
    try:
        copy_host_array_to_device(buffer, expected.astype(np.float32))
        actual = np.empty(expected.size, dtype=np.float32)
        copy_device_to_host(host_array_ptr(actual), buffer, actual.nbytes)
        np.testing.assert_array_equal(actual, expected)
        source = expected.copy()
        copy_host_array_to_device(buffer, source)
        source[:] = -1
        output = np.empty_like(expected)
        copy_device_to_host(host_array_ptr(output), buffer, output.nbytes)
        np.testing.assert_array_equal(output, expected)
        # This conversion allocates: retaining only its address would lose the
        # source before entering memcpy, unlike the contiguous no-copy case.
        copy_host_array_to_device(buffer, np.ascontiguousarray(expected[::2]))
        strided_output = np.empty(expected.size // 2, dtype=expected.dtype)
        copy_device_to_host(host_array_ptr(strided_output), buffer, strided_output.nbytes)
        np.testing.assert_array_equal(strided_output, expected[::2])
    finally:
        free(buffer)
