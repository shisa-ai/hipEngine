"""M5 kernel gate: YuE2 VAE FP32 conv/snake kernels vs the NumPy reference.

The released Oobleck decoder is FP32 end to end. These kernels keep FP32
accumulation and the reference's own loop order (kernel tap outside, input
channel inside), so agreement with ``hipengine.kernels.cpu_reference.yue2`` is
expected to FP32 rounding rather than bit-exactly: the reference's own tiled and
full decodes already differ by ~1e-6 absolute, and the torch reference computes
its channel sums as GEMMs with an implementation-defined order.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_array_to_device,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import yue2 as reference


def _has_hip() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


requires_hip = pytest.mark.skipif(not _has_hip(), reason="ROCm/HIP runtime is not available")


def _upload(array: np.ndarray):
    host = np.ascontiguousarray(array, dtype=np.float32)
    buffer = malloc(max(host.nbytes, 8))
    copy_host_array_to_device(buffer, host)
    return buffer


def _download(buffer, shape) -> np.ndarray:
    out = np.empty(shape, dtype=np.float32)
    copy_device_to_host(host_array_ptr(out), buffer)
    return out


def _close(got: np.ndarray, expected: np.ndarray, *, atol: float = 2e-5) -> None:
    assert got.shape == expected.shape
    delta = np.abs(got.astype(np.float64) - expected.astype(np.float64))
    scale = max(1.0, float(np.abs(expected).max()))
    assert float(delta.max()) <= atol * scale, f"max abs {delta.max()} vs {atol * scale}"


@requires_hip
def test_conv1d_matches_the_numpy_reference():
    from hipengine.kernels.hip_gfx1100.yue2 import vae

    rng = np.random.default_rng(3)
    channels_in, channels_out, length, kernel = 6, 5, 23, 7
    x = rng.standard_normal((channels_in, length)).astype(np.float32)
    weight = rng.standard_normal((channels_out, channels_in, kernel)).astype(np.float32) * 0.3
    bias = rng.standard_normal(channels_out).astype(np.float32) * 0.1
    for stride, dilation, padding in ((1, 1, 3), (1, 3, 9), (2, 1, 0), (1, 9, 27)):
        out_length = (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
        if out_length < 1:
            continue
        expected = reference.conv1d(
            x, weight, bias, stride=stride, dilation=dilation, padding=padding
        )
        x_buf = _upload(x)
        w_buf = _upload(weight)
        b_buf = _upload(bias)
        out_buf = _upload(np.zeros((channels_out, out_length), dtype=np.float32))
        vae.vae_conv1d_f32(
            x_buf.ptr, w_buf.ptr, b_buf.ptr, out_buf.ptr, channels_in, channels_out,
            length, out_length, kernel, stride, dilation, padding,
        )
        _close(_download(out_buf, (channels_out, out_length)), expected)


@requires_hip
def test_conv1d_without_bias_matches_the_numpy_reference():
    from hipengine.kernels.hip_gfx1100.yue2 import vae

    rng = np.random.default_rng(4)
    channels_in, channels_out, length, kernel = 4, 2, 17, 7
    x = rng.standard_normal((channels_in, length)).astype(np.float32)
    weight = rng.standard_normal((channels_out, channels_in, kernel)).astype(np.float32) * 0.3
    out_length = length
    expected = reference.conv1d(x, weight, None, padding=3)
    x_buf = _upload(x)
    w_buf = _upload(weight)
    out_buf = _upload(np.zeros((channels_out, out_length), dtype=np.float32))
    vae.vae_conv1d_f32(
        x_buf.ptr, w_buf.ptr, 0, out_buf.ptr, channels_in, channels_out, length,
        out_length, kernel, 1, 1, 3,
    )
    _close(_download(out_buf, (channels_out, out_length)), expected)


@requires_hip
def test_conv_transpose1d_matches_the_numpy_reference():
    from hipengine.kernels.hip_gfx1100.yue2 import vae

    rng = np.random.default_rng(5)
    channels_in, channels_out, length = 5, 3, 11
    x = rng.standard_normal((channels_in, length)).astype(np.float32)
    bias = rng.standard_normal(channels_out).astype(np.float32) * 0.1
    # The released decoder's own upsample shapes: kernel = 2 * stride.
    for stride in (2, 4, 5, 6):
        kernel = 2 * stride
        padding = -(-stride // 2)
        weight = (
            rng.standard_normal((channels_in, channels_out, kernel)).astype(np.float32) * 0.3
        )
        out_length = (length - 1) * stride - 2 * padding + kernel
        expected = reference.conv_transpose1d(x, weight, bias, stride=stride, padding=padding)
        assert expected.shape == (channels_out, out_length)
        x_buf = _upload(x)
        w_buf = _upload(weight)
        b_buf = _upload(bias)
        out_buf = _upload(np.zeros((channels_out, out_length), dtype=np.float32))
        vae.vae_conv_transpose1d_f32(
            x_buf.ptr, w_buf.ptr, b_buf.ptr, out_buf.ptr, channels_in, channels_out,
            length, out_length, kernel, stride, padding,
        )
        _close(_download(out_buf, (channels_out, out_length)), expected)


@requires_hip
def test_snake_beta_matches_the_numpy_reference():
    from hipengine.kernels.hip_gfx1100.yue2 import vae

    rng = np.random.default_rng(6)
    channels, length = 5, 19
    x = rng.standard_normal((channels, length)).astype(np.float32) * 2
    # Log-scale parameters, as the released decoder stores them.
    alpha = rng.standard_normal(channels).astype(np.float32) * 0.5
    beta = rng.standard_normal(channels).astype(np.float32) * 0.5
    expected = reference.snake_beta(x, alpha, beta)
    x_buf = _upload(x)
    a_buf = _upload(alpha)
    b_buf = _upload(beta)
    out_buf = _upload(np.zeros((channels, length), dtype=np.float32))
    vae.vae_snake_beta_f32(x_buf.ptr, a_buf.ptr, b_buf.ptr, out_buf.ptr, channels, length)
    _close(_download(out_buf, (channels, length)), expected)


@requires_hip
def test_add_is_exact():
    from hipengine.kernels.hip_gfx1100.yue2 import vae

    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 11)).astype(np.float32)
    y = rng.standard_normal((3, 11)).astype(np.float32)
    x_buf = _upload(x)
    y_buf = _upload(y)
    out_buf = _upload(np.zeros((3, 11), dtype=np.float32))
    vae.vae_add_f32(x_buf.ptr, y_buf.ptr, out_buf.ptr, x.size)
    assert np.array_equal(_download(out_buf, x.shape), x + y)


@requires_hip
def test_snake_beta_is_finite_for_extreme_parameters():
    """A large negative log-beta must not divide by zero."""

    from hipengine.kernels.hip_gfx1100.yue2 import vae

    channels, length = 2, 4
    x = np.full((channels, length), 3.0, dtype=np.float32)
    alpha = np.zeros(channels, dtype=np.float32)
    beta = np.full(channels, -30.0, dtype=np.float32)
    expected = reference.snake_beta(x, alpha, beta)
    x_buf = _upload(x)
    a_buf = _upload(alpha)
    b_buf = _upload(beta)
    out_buf = _upload(np.zeros((channels, length), dtype=np.float32))
    vae.vae_snake_beta_f32(x_buf.ptr, a_buf.ptr, b_buf.ptr, out_buf.ptr, channels, length)
    got = _download(out_buf, (channels, length))
    assert np.isfinite(got).all()
    _close(got, expected, atol=1e-6)
