"""Pure-ctypes cuBLASLt FP16/FP32 GEMM surface tests (``sm_120a``).

Mirrors ``tests/test_hipblaslt.py`` for the CUDA backend.  Requires
``libcublasLt`` and a live CUDA device; skipped when the library is absent.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.cublaslt import (
    CUDA_R_16F,
    CUDA_R_32F,
    CublasLt,
    CublasLtAlgo,
    CublasLtHeuristicResult,
)
from hipengine.core.cuda import get_cuda_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)


def _cublaslt_available() -> bool:
    try:
        ctypes.CDLL("libcublasLt.so.13")
    except OSError:
        return False
    return True


def _run_problem(rows, in_features, out_features, output_dtype, x, weight):
    runtime = get_cuda_runtime()
    output_nbytes = (
        rows * out_features * 2 if output_dtype == CUDA_R_16F else rows * out_features * 4
    )
    output = np.empty((rows, out_features), dtype=np.float16 if output_dtype == CUDA_R_16F else np.float32)
    buffers = []
    owner = None
    try:
        x_device = malloc(x.nbytes, runtime=runtime)
        weight_device = malloc(weight.nbytes, runtime=runtime)
        output_device = malloc(output.nbytes, runtime=runtime)
        buffers.extend((x_device, weight_device, output_device))
        copy_host_to_device(x_device, host_array_ptr(x), runtime=runtime)
        copy_host_to_device(weight_device, host_array_ptr(weight), runtime=runtime)
        owner = CublasLt(runtime=runtime)
        problem = owner.problem(
            rows,
            in_features,
            out_features,
            output_dtype=output_dtype,
        )
        problem.launch(
            x_device.ptr,
            weight_device.ptr,
            output_device.ptr,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(output), output_device, runtime=runtime)
        return output
    finally:
        if owner is not None:
            owner.close()
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def test_cublaslt_ctypes_abi_sizes_match_headers() -> None:
    # cublasLtMatmulAlgo_t = uint64_t data[8]; heuristic result = algo(64) +
    # size_t(8) + cublasStatus_t(4) + float(4) + int[4](16) = 96.
    assert ctypes.sizeof(CublasLtAlgo) == 64
    assert ctypes.sizeof(CublasLtHeuristicResult) == 96


@pytest.mark.skipif(not _cublaslt_available(), reason="cuBLASLt is not available")
def test_cublaslt_fp16_weights_and_input_produce_fp16_matmul() -> None:
    rows, in_features, out_features = 64, 416, 416
    generator = np.random.default_rng(20260806)
    x = generator.normal(0.0, 0.2, (rows, in_features)).astype(np.float16)
    weight = generator.normal(0.0, 0.2, (out_features, in_features)).astype(np.float16)
    expected = x.astype(np.float32) @ weight.astype(np.float32).T
    output = _run_problem(rows, in_features, out_features, CUDA_R_16F, x, weight)
    assert output.dtype == np.float16
    # FP16 output rounding, so a few ULP of the fp16 magnitude is expected.
    np.testing.assert_allclose(output.astype(np.float32), expected, rtol=6.0e-3, atol=6.0e-3)


@pytest.mark.skipif(not _cublaslt_available(), reason="cuBLASLt is not available")
def test_cublaslt_fp32_output_boundary() -> None:
    rows, in_features, out_features = 64, 416, 416
    generator = np.random.default_rng(20260807)
    x = generator.normal(0.0, 0.2, (rows, in_features)).astype(np.float16)
    weight = generator.normal(0.0, 0.2, (out_features, in_features)).astype(np.float16)
    expected = x.astype(np.float32) @ weight.astype(np.float32).T
    output = _run_problem(rows, in_features, out_features, CUDA_R_32F, x, weight)
    assert output.dtype == np.float32
    np.testing.assert_allclose(output, expected, rtol=3.0e-3, atol=3.0e-4)
