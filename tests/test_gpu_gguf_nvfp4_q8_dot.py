import ctypes
from contextlib import ExitStack

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind as Copy, get_hip_runtime
from hipengine.kernels.backends import resolve_backend
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType as Q, dequantize_gguf_data


@pytest.mark.parametrize('width,scaled,shared', [(1152, True, False), (2304, False, False),
                                                (1152, True, True), (2304, False, True)])
def test_selected_nvfp4_q8_dot_matches_quantized_reference(width, scaled, shared):
    try:
        ctypes.CDLL('libamdhip64.so')
    except OSError as exc:
        pytest.skip(f'HIP unavailable: {exc}')
    backend = resolve_backend('auto', execution_device=True)
    if backend not in ('hip_gfx1100', 'hip_gfx1151'):
        pytest.skip('test requires gfx1100 or gfx1151')
    from hipengine.kernels.hip_gfx1100.quant import gguf_nvfp4_q8_dot  # noqa: F401

    variant = 'selected_shared_input_q8_dot_f32_f32_out' if shared else 'selected_q8_dot_f32_f32_out'
    launch = resolve(backend=backend, layer='moe_linear', quant='gguf_nvfp4', variant=variant)
    rt = get_hip_runtime()
    rng = np.random.default_rng(7393)
    routes, experts, outputs = 8, 2, 2304
    ids = np.array([1, 0, 1, 0, 0, 1, 0, -1], np.int32)
    scales = rng.uniform(-0.2, 0.2, routes).astype(np.float32)
    x = rng.normal(0, 0.02, (routes, width)).astype(np.float32)
    x[0, :32] = 0
    if shared:
        x[1:] = x[0]
    inputs = x[:1] if shared else x
    packed = rng.integers(0, 256, (experts, outputs, width // 64, 36), dtype=np.uint8)
    packed[:, :, :, :4] &= 63
    packed[1, 0, 0, :4] = (0, 1, 0x7f, 0xff)
    scratch = np.empty(inputs.size + inputs.size // 32 * 4, np.uint8)
    actual = np.empty((routes, outputs), np.float32)
    with ExitStack() as owned:
        def alloc(data, upload=True):
            ptr = rt.malloc(data.nbytes)
            owned.callback(rt.free, ptr)
            if upload:
                rt.memcpy(ptr, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
            return ptr

        xp, wp, ip = map(alloc, (inputs, packed, ids))
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
    np.testing.assert_array_equal(scratch[:inputs.size].view(np.int8), q[:len(inputs)].ravel())
    np.testing.assert_array_equal(scratch[inputs.size:].view(np.float32), rounded_d[:len(inputs)].ravel())
    restored = (q * rounded_d).reshape(x.shape).astype(np.float64)
    dense = [dequantize_gguf_data(bank.reshape(outputs, -1), Q.NVFP4).astype(np.float64) for bank in packed]
    for i, e in enumerate(ids[:-1]):
        outer = float(scales[i]) if scaled else 1.0
        expected = (dense[e] @ restored[i]) * outer
        bound = (np.abs(dense[e]) @ np.abs(restored[i])) * abs(outer)
        assert (np.abs(actual[i] - expected) <= 4e-6 * bound + 1e-10).all()
        assert actual[i].argmax() == expected.argmax()
    assert not actual[-1].any()


@pytest.mark.parametrize('rows,width,scaled,split', [(2, 1152, True, False), (3, 2304, True, False),
                                                       (4, 1152, False, False), (2, 1152, False, True),
                                                       (3, 1152, True, True), (4, 1152, True, True)])
def test_counted_nvfp4_q8_matches_independent_rows(rows, width, scaled, split):
    try:
        ctypes.CDLL('libamdhip64.so')
    except OSError as exc:
        pytest.skip(f'HIP unavailable: {exc}')
    backend = resolve_backend('auto', execution_device=True)
    if backend not in ('hip_gfx1100', 'hip_gfx1151'):
        pytest.skip('test requires gfx1100 or gfx1151')
    from hipengine.kernels.hip_gfx1100.quant import gguf_nvfp4_q8_dot  # noqa: F401
    from hipengine.loading.inkling2_device_experts import build_expert_route_tiles

    key = dict(backend=backend, layer='moe_linear', quant='gguf_nvfp4')
    kind = 'split_activation' if split else 'shared_input'
    scalar = resolve(**key, variant=f'selected_{kind}_q8_dot_f32_f32_out')
    grouped = resolve(**key, variant=f'selected_counted_{kind}_q8_dot_f32_f32_out')
    caps = (4.0, 25.0) if split else ()
    rt = get_hip_runtime()
    rng = np.random.default_rng(83 + rows)
    experts, outputs = 12, 2304
    x = rng.normal(0, 0.7, (rows, width * (16 if split else 1))).astype(np.float32)
    x[0, :32] = 0
    ids = np.concatenate([rng.permutation(experts)[:8] for _ in range(rows)]).astype(np.int32)
    tiles = build_expert_route_tiles(ids, 4)
    scales = rng.uniform(-0.2, 0.2, ids.size).astype(np.float32)
    raw = rng.integers(0, 256, (experts, outputs, width // 64, 36), dtype=np.uint8)
    raw[..., :4] &= 63
    actual = np.empty((ids.size, outputs), np.float32)
    expected = np.empty_like(actual)
    with ExitStack() as owned:
        def alloc(data, upload=True):
            ptr = rt.malloc(data.nbytes)
            owned.callback(rt.free, ptr)
            if upload:
                rt.memcpy(ptr, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
            return ptr

        xp, wp, ip, tp = map(alloc, (x, raw, ids, tiles))
        sp = alloc(scales) if scaled else 0
        yp = alloc(actual, False)
        packed_values = rows * width * (8 if split else 1)
        scratch = alloc(np.empty(packed_values * 9 // 8, np.uint8), False)
        for i in range(rows):
            scalar(xp + i * x.strides[0], wp, yp + i * 8 * actual.strides[0], ip + i * 32,
                   sp + i * 32 if sp else 0, 8, experts, width, outputs, *caps, scratch=scratch)
        rt.memcpy(expected.ctypes.data, yp, expected.nbytes, Copy.DEVICE_TO_HOST)
        grouped(xp, wp, yp, tp, sp, len(tiles), rows, experts, width, outputs, *caps, scratch=scratch)
        rt.memcpy(actual.ctypes.data, yp, actual.nbytes, Copy.DEVICE_TO_HOST)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("caps", [(8.0, 8.0), (4.0, 25.0)])
def test_nvfp4_fused_split_activation_pack_matches_unfused(caps):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    from hipengine.kernels.hip_gfx1100.quant import gguf_nvfp4_q8_dot  # noqa: F401
    from hipengine.kernels.hip_gfx1100.quant.inkling2_experts import launch_split_expert_activation

    key = dict(backend=backend, layer="moe_linear", quant="gguf_nvfp4")
    launch = resolve(**key, variant="selected_q8_dot_f32_f32_out")
    fused = resolve(**key, variant="selected_split_activation_q8_dot_f32_f32_out")
    rt = get_hip_runtime()
    rng = np.random.default_rng(81357)
    routes, experts, width, outputs = 8, 3, 1152, 2304
    paired = np.empty((routes, 2 * width), np.float32)
    ids = np.array([1, 0, 1, 0, 2, 1, 0, 2], np.int32)
    scales = rng.uniform(-0.2, 0.2, routes).astype(np.float32)
    packed = rng.integers(0, 256, (experts, outputs, width // 64, 36), dtype=np.uint8)
    packed[..., :4] &= 63
    values = routes * width
    buffers = [np.empty(n, np.uint8) for n in (paired.nbytes, values * 4, routes * outputs * 4, values * 9 // 8)]
    with ExitStack() as owned:
        def alloc(data, upload=False):
            ptr = rt.malloc(data.nbytes)
            owned.callback(rt.free, ptr)
            if upload:
                rt.memcpy(ptr, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
            return ptr

        gp, xp, yp, qp = map(alloc, buffers)
        wp, ip, sp = [alloc(x, upload=True) for x in (packed, ids, scales)]

        def read_outputs():
            observed = [np.empty_like(arr) for arr in buffers[2:]]
            for host, ptr in zip(observed, (yp, qp), strict=True):
                rt.memcpy(host.ctypes.data, ptr, host.nbytes, Copy.DEVICE_TO_HOST)
            return observed

        for magnitude in (1e-7, 0.01, 0.7, 5, 100):
            paired[:] = rng.normal(0, magnitude, paired.shape)
            paired[0, :32] = -0.0
            rt.memcpy(gp, paired.ctypes.data, paired.nbytes, Copy.HOST_TO_DEVICE)
            launch_split_expert_activation(gp, xp, routes, width, *caps, target_arch=backend.removeprefix("hip_"))
            launch(xp, wp, yp, ip, sp, routes, experts, width, outputs, scratch=qp)
            expected = read_outputs()
            fused(gp, wp, yp, ip, sp, routes, experts, width, outputs, *caps, scratch=qp)
            for actual, control in zip(read_outputs(), expected, strict=True):
                np.testing.assert_array_equal(actual, control)
