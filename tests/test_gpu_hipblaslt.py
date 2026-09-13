from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.hipblaslt import (
    HipblasLt,
    HipblasLtAlgo,
    HipblasLtHeuristicResult,
)
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)


def _hipblaslt_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        ctypes.CDLL("libhipblaslt.so")
    except OSError:
        return False
    return True


def test_hipblaslt_ctypes_abi_sizes_match_rocm_headers() -> None:
    assert ctypes.sizeof(HipblasLtAlgo) == 24
    assert ctypes.sizeof(HipblasLtHeuristicResult) == 56


@pytest.mark.skipif(not _hipblaslt_available(), reason="HIP/hipBLASLt is not available")
def test_hipblaslt_fp16_weight_and_input_produce_fp32_matmul() -> None:
    rows, in_features, out_features = 16, 32, 24
    generator = np.random.default_rng(20260725)
    x = generator.normal(0.0, 0.2, (rows, in_features)).astype(np.float16)
    weight = generator.normal(0.0, 0.2, (out_features, in_features)).astype(np.float16)
    expected = x.astype(np.float32) @ weight.astype(np.float32).T
    output = np.empty((rows, out_features), dtype=np.float32)
    buffers = []
    owner = None
    try:
        x_device = malloc(x.nbytes)
        weight_device = malloc(weight.nbytes)
        output_device = malloc(output.nbytes)
        buffers.extend((x_device, weight_device, output_device))
        copy_host_to_device(x_device, host_array_ptr(x))
        copy_host_to_device(weight_device, host_array_ptr(weight))

        owner = HipblasLt()
        problem = owner.problem(rows, in_features, out_features)
        algorithm = problem.algorithm()
        assert algorithm.state == 0
        assert algorithm.workspace_size == 0
        problem.launch(
            algorithm,
            x_device.ptr,
            weight_device.ptr,
            output_device.ptr,
        )
        get_hip_runtime().device_synchronize()
        copy_device_to_host(host_array_ptr(output), output_device)

        np.testing.assert_allclose(output, expected, rtol=3.0e-3, atol=3.0e-4)
    finally:
        if owner is not None:
            owner.close()
        for buffer in reversed(buffers):
            free(buffer)
