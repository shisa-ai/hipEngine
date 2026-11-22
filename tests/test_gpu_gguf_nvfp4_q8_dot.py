import ctypes
from contextlib import ExitStack

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind as Copy, get_hip_runtime
from hipengine.kernels.backends import resolve_backend
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType as Q, dequantize_gguf_data


@pytest.mark.parametrize('width,scaled', [(1152, True), (2304, False)])
def test_selected_nvfp4_q8_dot_matches_quantized_reference(width, scaled):
    try:
        ctypes.CDLL('libamdhip64.so')
    except OSError as exc:
        pytest.skip(f'HIP unavailable: {exc}')
    backend = resolve_backend('auto', execution_device=True)
    if backend not in ('hip_gfx1100', 'hip_gfx1151'):
        pytest.skip('test requires gfx1100 or gfx1151')
    from hipengine.kernels.hip_gfx1100.quant import gguf_nvfp4_q8_dot  # noqa: F401

    launch = resolve(backend=backend, layer='moe_linear', quant='gguf_nvfp4', variant='selected_q8_dot_f32_f32_out')
    rt = get_hip_runtime()
    rng = np.random.default_rng(7393)
    routes, experts, outputs = 8, 2, 2304
    ids = np.array([1, 0, 1, 0, 0, 1, 0, -1], np.int32)
    scales = rng.uniform(-0.2, 0.2, routes).astype(np.float32)
    x = rng.normal(0, 0.02, (routes, width)).astype(np.float32)
    x[0, :32] = 0
    packed = rng.integers(0, 256, (experts, outputs, width // 64, 36), dtype=np.uint8)
    packed[:, :, :, :4] &= 63
    packed[1, 0, 0, :4] = (0, 1, 0x7f, 0xff)
    scratch = np.empty(x.size + x.size // 32 * 4, np.uint8)
    actual = np.empty((routes, outputs), np.float32)
    with ExitStack() as owned:
        def alloc(data, upload=True):
            ptr = rt.malloc(data.nbytes)
            owned.callback(rt.free, ptr)
            if upload:
                rt.memcpy(ptr, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
            return ptr

        xp, wp, ip = map(alloc, (x, packed, ids))
        sp = alloc(scales) if scaled else 0
        yp, qp = alloc(actual, False), alloc(scratch, False)
        launch(xp, wp, yp, ip, sp, routes, experts, width, outputs, scratch=qp)
        rt.memcpy(actual.ctypes.data, yp, actual.nbytes, Copy.DEVICE_TO_HOST)
        rt.memcpy(scratch.ctypes.data, qp, scratch.nbytes, Copy.DEVICE_TO_HOST)
    group = x.reshape(routes, -1, 32)
    d = np.max(np.abs(group), axis=-1, keepdims=True) / np.float32(127)
    normalized = np.divide(group, d, out=np.zeros_like(group), where=d != 0)
    q = np.copysign(np.floor(np.abs(normalized) + np.float32(0.5)), normalized).astype(np.int8)
    rounded_d = d.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(scratch[:x.size].view(np.int8), q.ravel())
    np.testing.assert_array_equal(scratch[x.size:].view(np.float32), rounded_d.ravel())
    restored = (q * rounded_d).reshape(x.shape).astype(np.float64)
    dense = [dequantize_gguf_data(bank.reshape(outputs, -1), Q.NVFP4).astype(np.float64) for bank in packed]
    for i, e in enumerate(ids[:-1]):
        outer = float(scales[i]) if scaled else 1.0
        expected = (dense[e] @ restored[i]) * outer
        bound = (np.abs(dense[e]) @ np.abs(restored[i])) * abs(outer)
        assert (np.abs(actual[i] - expected) <= 4e-6 * bound + 1e-10).all()
        assert actual[i].argmax() == expected.argmax()
    assert not actual[-1].any()
