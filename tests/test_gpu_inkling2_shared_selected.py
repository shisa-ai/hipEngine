"""Selected NVFP4 projections whose routes share one input row, and the row replication (HIP)."""

import ctypes
from contextlib import ExitStack

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.kernels.backends import hip_target_arch_for_backend, resolve_backend
from hipengine.kernels.hip_gfx1100.quant.gguf_nvfp4_gemv import _launch_selected, _launch_shared_selected
from hipengine.kernels.hip_gfx1100.quant.inkling2_experts import launch_replicate_row


def _available():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    return hip_target_arch_for_backend(backend)


def _decode(blocks):
    """Values of packed NVFP4 blocks shaped (blocks, 36), in kernel order (group, low half, high half)."""
    scales = blocks[:, :4].view(np.uint8)
    exponent = (scales >> 3) & 15
    significand = (scales & 7) + np.where(exponent, 8, 0)
    scale = significand * np.exp2(np.where(exponent, exponent, 1).astype(np.float64) - 11)
    scale = np.where(scales == 0x7F, 0.0, scale).astype(np.float32)
    codes = blocks[:, 4:].view(np.uint8)
    magnitude = np.array([0, 1, 2, 3, 4, 6, 8, 12], np.float32)
    low = magnitude[codes & 7] * np.where(codes & 8, -1.0, 1.0).astype(np.float32)
    high = magnitude[(codes >> 4) & 7] * np.where(codes & 128, -1.0, 1.0).astype(np.float32)
    # lane l covers group l // 8 and lanes 8..15 take the high half of each group's 16 values
    values = np.empty((len(blocks), 64), np.float32)
    values[:, 0:8], values[:, 8:16] = low[:, 0:8], high[:, 0:8]
    values[:, 16:24], values[:, 24:32] = low[:, 8:16], high[:, 8:16]
    values[:, 32:40], values[:, 40:48] = low[:, 16:24], high[:, 16:24]
    values[:, 48:56], values[:, 56:64] = low[:, 24:32], high[:, 24:32]
    group = np.repeat(np.arange(4), 16)
    return values * scale[:, group]


def _reference(x, weights, ids, scales, experts, n, m):
    out = np.zeros((len(ids), m), np.float64)
    for route, expert in enumerate(ids):
        rows = weights[expert].reshape(m, n // 64, 36)
        values = _decode(rows.reshape(-1, 36)).reshape(m, n)
        out[route] = values.astype(np.float64) @ x[route].astype(np.float64)
    return (out * (1.0 if scales is None else scales[:, None])).astype(np.float32)


@pytest.mark.parametrize("register_fp4", [False, True])
@pytest.mark.parametrize("n,m", [(2304, 2304), (1152, 2304)])
def test_shared_input_selected_matches_the_selected_projection_over_replicated_rows(register_fp4, n, m):
    """Route r reads row r // routes_per_input; the arithmetic is the selected projection's."""
    from hipengine.core.hip import HipMemcpyKind as Copy

    arch = _available()
    runtime = get_hip_runtime()
    rng = np.random.default_rng(325)
    experts, routes = 16, 8
    weights = rng.integers(0, 256, (experts, m, n // 64, 36), dtype=np.uint8)
    weights[..., :4] = rng.integers(0, 120, (experts, m, n // 64, 4), dtype=np.uint8)
    hidden = rng.standard_normal((2, n), dtype=np.float32)
    ids = np.array([3, 3, 0, 15, 7, 3, 1, 1], np.int32)
    scales = rng.uniform(0.5, 1.5, routes).astype(np.float32)
    x = np.repeat(hidden, routes // 2, axis=0)
    expected, actual, unscaled = (np.empty((routes, m), np.float32) for _ in range(3))
    with ExitStack() as owned:
        def upload(data):
            ptr = runtime.malloc(data.nbytes)
            owned.callback(runtime.free, ptr)
            runtime.memcpy(ptr, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
            return ptr

        wp, xp, hp, ip, sp = map(upload, (weights, x, hidden, ids, scales))
        out = runtime.malloc(expected.nbytes)
        owned.callback(runtime.free, out)
        common = dict(target_arch=arch, register_fp4=register_fp4)
        _launch_selected(xp, wp, out, ip, sp, routes, experts, n, m, **common)
        runtime.memcpy(expected.ctypes.data, out, expected.nbytes, Copy.DEVICE_TO_HOST)
        _launch_shared_selected(hp, wp, out, ip, sp, routes, experts, n, m, routes_per_input=routes // 2, **common)
        runtime.memcpy(actual.ctypes.data, out, actual.nbytes, Copy.DEVICE_TO_HOST)
        # Null scales are a factor of one; a route count that does not divide by the rows is refused.
        _launch_shared_selected(hp, wp, out, ip, 0, routes, experts, n, m, routes_per_input=routes // 2, **common)
        runtime.memcpy(unscaled.ctypes.data, out, unscaled.nbytes, Copy.DEVICE_TO_HOST)
        from hipengine.core.hip import HipError

        with pytest.raises(HipError, match="shared-input selected projection failed"):
            _launch_shared_selected(hp, wp, out, ip, sp, routes, experts, n, m, routes_per_input=3, **common)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_allclose(actual, _reference(x, weights, ids, scales, experts, n, m), rtol=2e-5, atol=0.02)
    np.testing.assert_allclose(unscaled, _reference(x, weights, ids, None, experts, n, m), rtol=2e-5, atol=0.02)


def test_replicated_rows_equal_the_source_row():
    from hipengine.core.hip import HipMemcpyKind as Copy

    arch = _available()
    runtime = get_hip_runtime()
    rng = np.random.default_rng(25)
    row = rng.standard_normal(2304, dtype=np.float32)
    actual = np.full((8, 2304), np.nan, np.float32)
    with ExitStack() as owned:
        sp = runtime.malloc(row.nbytes)
        dp = runtime.malloc(actual.nbytes)
        owned.callback(runtime.free, dp)
        owned.callback(runtime.free, sp)
        runtime.memcpy(sp, row.ctypes.data, row.nbytes, Copy.HOST_TO_DEVICE)
        launch_replicate_row(sp, dp, 2304, 8, target_arch=arch)
        runtime.memcpy(actual.ctypes.data, dp, actual.nbytes, Copy.DEVICE_TO_HOST)
    np.testing.assert_array_equal(actual, np.repeat(row[None], 8, axis=0))
