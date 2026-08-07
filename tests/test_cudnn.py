"""Pure-ctypes cuDNN FP16 convolution surface tests (``sm_120a``).

Mirrors ``tests/test_cublaslt.py`` for the cuDNN conv front end.  Requires
``libcudnn.so.9`` and a live CUDA device; skipped when the library is absent.
The wrapper only exposes the production Moonshine conv shape family, so the
reference is a straightforward numpy NCHW fp32-accumulate convolution.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.cuda import get_cuda_runtime
from hipengine.core.cudnn import (
    CUDNN_DATA_FLOAT,
    CUDNN_DATA_HALF,
    Cudnn,
    CudnnError,
)
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)


def _cudnn_available() -> bool:
    try:
        ctypes.CDLL("libcudnn.so.9")
    except OSError:
        return False
    return True


def _numpy_conv(x: np.ndarray, weight: np.ndarray, stride: int) -> np.ndarray:
    """NCHW [1, C_in, 1, L] fp16 in; [K, C_in, 1, k] fp16 weight -> fp16 out.

    Naive fp32-accumulate conv matching the cuDNN fp16-in/fp16-out math
    (products in fp32, single fp16 output rounding), so the wrapper output
    must agree within the accepted FP32-reassociation envelope.
    """
    x = x.astype(np.float32)[0, :, 0, :]  # [C_in, L]
    w = weight.astype(np.float32)[:, :, 0, :].reshape(weight.shape[0], -1)  # [K, C_in * k]
    k = weight.shape[3]
    c_in, length = x.shape
    out_len = (length - k) // stride + 1
    out = np.zeros((weight.shape[0], out_len), dtype=np.float32)
    for pos in range(out_len):
        window = x[:, pos * stride : pos * stride + k].reshape(-1)  # [C_in*k]
        out[:, pos] = w @ window
    return out.astype(np.float16)


def _run_conv(batch, in_channels, out_channels, kernel, stride, in_length):
    runtime = get_cuda_runtime()
    out_length = (in_length - kernel) // stride + 1
    generator = np.random.default_rng(20260806)
    x = generator.normal(0.0, 0.3, (batch, in_channels, 1, in_length)).astype(np.float16)
    weight = generator.normal(0.0, 0.3, (out_channels, in_channels, 1, kernel)).astype(np.float16)
    buffers = []
    owner = None
    try:
        x_device = malloc(x.nbytes, runtime=runtime)
        w_device = malloc(weight.nbytes, runtime=runtime)
        y_device = malloc(batch * out_channels * out_length * 2, runtime=runtime)
        buffers.extend((x_device, w_device, y_device))
        copy_host_to_device(x_device, host_array_ptr(x), runtime=runtime)
        copy_host_to_device(w_device, host_array_ptr(weight), runtime=runtime)
        owner = Cudnn(runtime=runtime)
        conv = owner.conv(
            batch, in_channels, out_channels, kernel, stride, in_length
        )
        assert conv.required_workspace_bytes == 0
        conv.forward(x_device.ptr, w_device.ptr, y_device.ptr)
        runtime.device_synchronize()
        output = np.empty((batch, out_channels, out_length), dtype=np.float16)
        copy_device_to_host(host_array_ptr(output), y_device, runtime=runtime)
        reference = np.stack(
            [_numpy_conv(x[i : i + 1], weight, stride) for i in range(batch)]
        )
        return output, reference
    finally:
        if owner is not None:
            owner.close()
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


@pytest.mark.skipif(not _cudnn_available(), reason="cuDNN is not available")
def test_cudnn_conv_fp16_matches_numpy_reference() -> None:
    """A small fp16 conv reproduces a numpy fp32-accumulate reference."""
    output, reference = _run_conv(2, 8, 16, 3, 2, 64)
    assert output.dtype == np.float16
    diff = output.astype(np.float32) - reference.astype(np.float32)
    max_abs = float(np.max(np.abs(diff)))
    # fp16 output rounding + reduction order: a few ULP of the fp16 magnitude.
    assert max_abs <= 0.05, f"max-abs {max_abs} exceeded the FP32-reassociation bound"


@pytest.mark.skipif(not _cudnn_available(), reason="cuDNN is not available")
def test_cudnn_production_conv_shapes_have_no_workspace() -> None:
    """The three production convs use IMPLICIT_GEMM and need no workspace."""
    runtime = get_cuda_runtime()
    owner = None
    try:
        owner = Cudnn(runtime=runtime)
        shapes = [
            (1, 1, 416, 127, 64, 480_000),
            (1, 416, 832, 7, 3, 7499),
            (1, 832, 416, 3, 2, 2498),
        ]
        for batch, inc, outc, k, s, in_len in shapes:
            conv = owner.conv(batch, inc, outc, k, s, in_len)
            assert conv.required_workspace_bytes == 0, (
                f"conv {inc}->{outc} k={k} s={s} should need no workspace"
            )
    finally:
        if owner is not None:
            owner.close()


@pytest.mark.skipif(not _cudnn_available(), reason="cuDNN is not available")
def test_cudnn_fp32_output_is_not_supported_for_production_shapes() -> None:
    """cuDNN 9.25 rejects fp16-in/fp32-out conv for these shapes (NOT_SUPPORTED).

    This documents the reason the cuDNN conv route cannot recover the custom
    kernels' fp32-accumulator-before-activation rounding contract: the only
    available legacy-API output is fp16, so the activation runs on the fp16
    rounded conv output and the route stays numerically divergent (opt-in,
    non-default).
    """
    runtime = get_cuda_runtime()
    owner = None
    try:
        owner = Cudnn(runtime=runtime)
        with pytest.raises(CudnnError, match="cudnnGetConvolutionForwardWorkspaceSize"):
            owner.conv(
                1, 416, 832, 7, 3, 7499,
                output_dtype=CUDNN_DATA_FLOAT,
            )
    finally:
        if owner is not None:
            owner.close()
