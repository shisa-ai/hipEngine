"""GPU tests for the VibeVoice codec decoder kernels (gfx1100).

Each primitive is compared against the CPU reference
(hipengine/kernels/cpu_reference/vibevoice_codec.py) on the same fp32 inputs.
Continuous fp32 outputs across different accumulation orders are gated with
calibrated tolerances, not bit equality. Skips without a ROCm device.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import vibevoice_codec as ref

hip = pytest.importorskip("hipengine.core.hip")

ctypes.CDLL("libamdhip64.so")


def _runtime():
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    return runtime


def _library():
    from hipengine.kernels.hip_gfx1100.vibevoice.codec_ops import build_codec_ops

    return build_codec_ops(load=True)


def _ptr(buffer) -> int:
    return ctypes.c_void_p(buffer.ptr)


def _launch(fn, args: list, runtime) -> None:
    err = fn(*args)
    assert err == 0, f"kernel launch failed with {err}"
    runtime.device_synchronize()


def _to_device(runtime, array: np.ndarray) -> int:
    from hipengine.core.memory import copy_host_to_device, malloc

    host = np.ascontiguousarray(array, dtype=np.float32)
    buffer = malloc(host.nbytes, runtime=runtime)
    copy_host_to_device(
        buffer, host.ctypes.data, host.nbytes, runtime=runtime
    )
    return buffer


def _to_host(runtime, buffer, shape: tuple[int, ...]) -> np.ndarray:
    from hipengine.core.memory import copy_device_to_host

    out = np.empty(shape, dtype=np.float32)
    copy_device_to_host(out.ctypes.data, buffer, out.nbytes, runtime=runtime)
    return out


pytestmark = pytest.mark.skipif(
    _runtime() is None, reason="no HIP runtime available"
)


class TestCodecKernelsAgainstReference:
    def test_dense_conv1d(self):
        from hipengine.core.memory import free

        runtime = _runtime()
        lib = _library()
        rng = np.random.default_rng(41)
        in_c, out_c, k, l_in = 8, 12, 3, 9
        x = rng.standard_normal((in_c, l_in)).astype(np.float32)
        w = (rng.standard_normal((out_c, in_c, k)) * 0.2).astype(np.float32)
        b = (rng.standard_normal(out_c) * 0.1).astype(np.float32)
        l_out = l_in - k + 1
        expected = ref.conv1d_valid(x, w, b, "expected")

        x_buf = _to_device(runtime, x)
        w_buf = _to_device(runtime, w)
        b_buf = _to_device(runtime, b)
        out_buf = _to_device(runtime, np.zeros((out_c, l_out), np.float32))
        try:
            _launch(
                lib.hipengine_vv_conv1d_valid_dense,
                [
                    _ptr(x_buf), _ptr(w_buf), _ptr(b_buf), _ptr(out_buf),
                    ctypes.c_int64(in_c), ctypes.c_int64(out_c),
                    ctypes.c_int64(k), ctypes.c_int64(l_out),
                    ctypes.c_void_p(0),
                ],
                runtime,
            )
            got = _to_host(runtime, out_buf, (out_c, l_out))
        finally:
            for buf in (x_buf, w_buf, b_buf, out_buf):
                free(buf, runtime=runtime)
        assert np.abs(got - expected).max() < 1e-5

    def test_depthwise_conv1d(self):
        from hipengine.core.memory import free

        runtime = _runtime()
        lib = _library()
        rng = np.random.default_rng(43)
        channels, k, l_in = 32, 3, 9
        x = rng.standard_normal((channels, l_in)).astype(np.float32)
        w = (rng.standard_normal((channels, 1, k)) * 0.2).astype(np.float32)
        b = (rng.standard_normal(channels) * 0.1).astype(np.float32)
        l_out = l_in - k + 1
        expected = ref.depthwise_conv1d_valid(x, w, b, "expected")

        x_buf = _to_device(runtime, x)
        w_buf = _to_device(runtime, w)
        b_buf = _to_device(runtime, b)
        out_buf = _to_device(runtime, np.zeros((channels, l_out), np.float32))
        try:
            _launch(
                lib.hipengine_vv_conv1d_valid_depthwise,
                [
                    _ptr(x_buf), _ptr(w_buf), _ptr(b_buf), _ptr(out_buf),
                    ctypes.c_int64(channels), ctypes.c_int64(k),
                    ctypes.c_int64(l_out), ctypes.c_void_p(0),
                ],
                runtime,
            )
            got = _to_host(runtime, out_buf, (channels, l_out))
        finally:
            for buf in (x_buf, w_buf, b_buf, out_buf):
                free(buf, runtime=runtime)
        assert np.abs(got - expected).max() < 1e-6

    @pytest.mark.parametrize("stride,k", [(2, 4), (4, 8), (5, 10), (8, 16)])
    def test_conv_transpose1d_causal(self, stride, k):
        from hipengine.core.memory import free

        runtime = _runtime()
        lib = _library()
        rng = np.random.default_rng(47 + stride)
        in_c, out_c, l_in = 6, 5, 7
        x = rng.standard_normal((in_c, l_in)).astype(np.float32)
        w = (rng.standard_normal((in_c, out_c, k)) * 0.1).astype(np.float32)
        b = (rng.standard_normal(out_c) * 0.1).astype(np.float32)
        l_out = l_in * stride
        expected = ref.conv_transpose1d_causal(x, w, b, stride, "expected")

        x_buf = _to_device(runtime, x)
        w_buf = _to_device(runtime, w)
        b_buf = _to_device(runtime, b)
        out_buf = _to_device(runtime, np.zeros((out_c, l_out), np.float32))
        try:
            _launch(
                lib.hipengine_vv_conv_transpose1d_causal,
                [
                    _ptr(x_buf), _ptr(w_buf), _ptr(b_buf), _ptr(out_buf),
                    ctypes.c_int64(in_c), ctypes.c_int64(out_c),
                    ctypes.c_int64(k), ctypes.c_int64(stride),
                    ctypes.c_int64(l_out), ctypes.c_void_p(0),
                ],
                runtime,
            )
            got = _to_host(runtime, out_buf, (out_c, l_out))
        finally:
            for buf in (x_buf, w_buf, b_buf, out_buf):
                free(buf, runtime=runtime)
        assert got.shape == expected.shape
        assert np.abs(got - expected).max() < 1e-4

    def test_rmsnorm_channels(self):
        from hipengine.core.memory import free

        runtime = _runtime()
        lib = _library()
        rng = np.random.default_rng(53)
        channels, length = 64, 5
        x = rng.standard_normal((channels, length)).astype(np.float32)
        w = (rng.standard_normal(channels) * 0.5 + 1.0).astype(np.float32)
        expected = ref.rms_norm_channels(x, w, 1e-5, "expected")

        x_buf = _to_device(runtime, x)
        w_buf = _to_device(runtime, w)
        out_buf = _to_device(runtime, np.zeros_like(x))
        try:
            _launch(
                lib.hipengine_vv_rmsnorm_channels,
                [
                    _ptr(x_buf), _ptr(w_buf), _ptr(out_buf),
                    ctypes.c_int64(channels), ctypes.c_int64(length),
                    ctypes.c_float(1e-5), ctypes.c_void_p(0),
                ],
                runtime,
            )
            got = _to_host(runtime, out_buf, x.shape)
        finally:
            for buf in (x_buf, w_buf, out_buf):
                free(buf, runtime=runtime)
        assert np.abs(got - expected).max() < 1e-6

    def test_gelu_erf(self):
        from hipengine.core.memory import free

        runtime = _runtime()
        lib = _library()
        x = np.linspace(-4, 4, 257, dtype=np.float32)
        expected = ref.gelu_exact(x, "expected")

        x_buf = _to_device(runtime, x)
        out_buf = _to_device(runtime, np.zeros_like(x))
        try:
            _launch(
                lib.hipengine_vv_gelu_erf,
                [
                    _ptr(x_buf), _ptr(out_buf),
                    ctypes.c_int64(x.size), ctypes.c_void_p(0),
                ],
                runtime,
            )
            got = _to_host(runtime, out_buf, x.shape)
        finally:
            for buf in (x_buf, out_buf):
                free(buf, runtime=runtime)
        assert np.abs(got - expected).max() < 1e-6

    def test_linear(self):
        from hipengine.core.memory import free

        runtime = _runtime()
        lib = _library()
        rng = np.random.default_rng(59)
        in_f, out_f, length = 32, 48, 3
        x = rng.standard_normal((in_f, length)).astype(np.float32)
        w = (rng.standard_normal((out_f, in_f)) * 0.1).astype(np.float32)
        b = (rng.standard_normal(out_f) * 0.1).astype(np.float32)
        expected = ref.linear(x, w, b, "expected")

        x_buf = _to_device(runtime, x)
        w_buf = _to_device(runtime, w)
        b_buf = _to_device(runtime, b)
        out_buf = _to_device(runtime, np.zeros((out_f, length), np.float32))
        try:
            _launch(
                lib.hipengine_vv_linear,
                [
                    _ptr(x_buf), _ptr(w_buf), _ptr(b_buf), _ptr(out_buf),
                    ctypes.c_int64(in_f), ctypes.c_int64(out_f),
                    ctypes.c_int64(length), ctypes.c_void_p(0),
                ],
                runtime,
            )
            got = _to_host(runtime, out_buf, (out_f, length))
        finally:
            for buf in (x_buf, w_buf, b_buf, out_buf):
                free(buf, runtime=runtime)
        assert np.abs(got - expected).max() < 1e-4

    def test_add_and_channel_scale(self):
        from hipengine.core.memory import free

        runtime = _runtime()
        lib = _library()
        rng = np.random.default_rng(61)
        channels, length = 16, 7
        x = rng.standard_normal((channels, length)).astype(np.float32)
        y = rng.standard_normal((channels, length)).astype(np.float32)
        scale = (rng.standard_normal(channels) * 0.1).astype(np.float32)

        x_buf = _to_device(runtime, x)
        y_buf = _to_device(runtime, y)
        scale_buf = _to_device(runtime, scale)
        add_buf = _to_device(runtime, np.zeros_like(x))
        scaled_buf = _to_device(runtime, np.zeros_like(x))
        try:
            _launch(
                lib.hipengine_vv_add,
                [
                    _ptr(x_buf), _ptr(y_buf), _ptr(add_buf),
                    ctypes.c_int64(x.size), ctypes.c_void_p(0),
                ],
                runtime,
            )
            _launch(
                lib.hipengine_vv_channel_scale,
                [
                    _ptr(add_buf), _ptr(scale_buf), _ptr(scaled_buf),
                    ctypes.c_int64(channels), ctypes.c_int64(length),
                    ctypes.c_void_p(0),
                ],
                runtime,
            )
            got_add = _to_host(runtime, add_buf, x.shape)
            got_scaled = _to_host(runtime, scaled_buf, x.shape)
        finally:
            for buf in (x_buf, y_buf, scale_buf, add_buf, scaled_buf):
                free(buf, runtime=runtime)
        np.testing.assert_allclose(got_add, x + y, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(
            got_scaled, (x + y) * scale[:, None], atol=1e-6, rtol=1e-6
        )
