"""Grouped WMMA projection of selected raw NVFP4 experts against a dequantized NumPy reference."""

import ctypes
from contextlib import ExitStack

import numpy as np
import pytest

from hipengine.core.hip import HipError, HipMemcpyKind as Copy, get_hip_runtime
from hipengine.kernels.backends import resolve_backend
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType as Q, dequantize_gguf_data


def _runnable():
    try:
        ctypes.CDLL('libamdhip64.so')
    except OSError as exc:
        pytest.skip(f'HIP unavailable: {exc}')
    backend = resolve_backend('auto', execution_device=True)
    if backend not in ('hip_gfx1100', 'hip_gfx1151'):
        pytest.skip('test requires gfx1100 or gfx1151')
    from hipengine.kernels.hip_gfx1100.quant import gguf_nvfp4_selected_wmma  # noqa: F401

    return backend


def _groups(ids, experts):
    """Compact order and 64-row groups: the schedule the projection owner builds for a block."""
    order = np.argsort(ids, kind='stable').astype(np.int32)
    count = np.bincount(ids, minlength=experts)
    start = np.r_[0, np.cumsum(count)].astype(np.int64)
    group_expert, group_first = [], []
    for expert in range(experts):
        for first in range(0, count[expert], 64):
            group_expert.append(expert)
            group_first.append(first)
    return order, start, np.asarray(group_expert, np.int32), np.asarray(group_first, np.int32)


@pytest.mark.parametrize('parts', [1, 2])
@pytest.mark.parametrize('n,m,shared,by_route', [(128, 96, True, True), (128, 96, False, False),
                                                 (2304, 2304, True, True)])
def test_selected_wmma_matches_dequantized_reference(parts, n, m, shared, by_route):
    backend = _runnable()
    variant = 'selected_wmma_f16_parts_f32_out' if parts == 2 else 'selected_wmma_f16_f32_out'
    launch = resolve(backend=backend, layer='moe_linear', quant='gguf_nvfp4', variant=variant)
    rt = get_hip_runtime()
    rng = np.random.default_rng(83 + parts + n)
    experts, rows, top = 7, 150, 8
    ids = np.concatenate([rng.permutation(experts)[:top] for _ in range(rows)]).astype(np.int32)
    routes = ids.size
    order, start, group_expert, group_first = _groups(ids, experts)
    outer = rng.uniform(-0.5, 0.5, experts).astype(np.float32)
    source = (order // top).astype(np.int32) if shared else order.astype(np.int32)
    x_rows = rows if shared else routes
    x = rng.normal(0, 0.7, (x_rows, n)).astype(np.float32)
    x[0, :32] = 0
    hi = x.astype(np.float16)
    lo = (x - hi.astype(np.float32)).astype(np.float16)
    raw = rng.integers(0, 256, (experts, m, n // 64, 36), dtype=np.uint8)
    raw[..., :4] &= 63
    raw[1, 0, 0, :4] = (0, 1, 0x7f, 0xff)
    # Three rows past the last route stay NaN: padding rows and other experts never write there.
    actual = np.full((routes + 3, m), np.nan, np.float32)
    with ExitStack() as owned:
        def alloc(data, upload=True):
            ptr = rt.malloc(max(data.nbytes, 16))
            owned.callback(rt.free, ptr)
            if upload:
                rt.memcpy(ptr, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
            return ptr

        hi_ptr, raw_ptr, out_ptr = alloc(hi), alloc(raw), alloc(actual)
        lo_ptr = alloc(lo) if parts == 2 else 0
        start_ptr, group_ptr, first_ptr = (alloc(a) for a in (start, group_expert, group_first))
        source_ptr = alloc(source) if shared else 0
        route_ptr = alloc(order) if by_route else 0
        outer_ptr = alloc(outer)
        launch(hi_ptr, lo_ptr, start_ptr, group_ptr, first_ptr, source_ptr, route_ptr, outer_ptr, raw_ptr, out_ptr,
               len(group_expert), n, m)
        rt.memcpy(actual.ctypes.data, out_ptr, actual.nbytes, Copy.DEVICE_TO_HOST)
    dense = np.stack([dequantize_gguf_data(bank.reshape(m, -1), Q.NVFP4).astype(np.float64) for bank in raw])
    value = hi.astype(np.float64) + (lo.astype(np.float64) if parts == 2 else 0)
    got = np.empty((routes, m), np.float64)
    if by_route:
        got[:] = actual[order]
    else:
        got[:] = actual[:routes]
    for expert in range(experts):
        first, stop = start[expert], start[expert + 1]
        rows = source[first:stop] if shared else np.arange(first, stop)
        scale = float(outer[expert])
        expected = (dense[expert] @ value[rows].T).T * scale
        bound = (np.abs(dense[expert]) @ np.abs(value[rows]).T).T * abs(scale)
        assert (np.abs(got[first:stop] - expected) <= 1e-4 * bound + 1e-30).all()
    assert np.isfinite(got).all()
    untouched = np.ones(routes + 3, bool)
    untouched[order] = False
    assert np.isnan(actual[untouched]).all()


@pytest.mark.parametrize('parts', [1, 2])
def test_fp16_parts_match_the_host_split(parts):
    backend = _runnable()
    split = resolve(backend=backend, layer='inkling2_expert_pointwise', quant='inkling2_mixed', variant='fp16_parts')
    rt = get_hip_runtime()
    rng = np.random.default_rng(5 + parts)
    x = (rng.normal(0, 1, 5_000) * 10.0 ** rng.integers(-12, 4, 5_000)).astype(np.float32)
    hi = np.full(x.size, np.nan, np.float16)
    lo = np.full(x.size, np.nan, np.float16)
    with ExitStack() as owned:
        ptr = []
        for data in (x, hi, lo):
            p = rt.malloc(data.nbytes)
            owned.callback(rt.free, p)
            rt.memcpy(p, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
            ptr.append(p)
        split(ptr[0], ptr[1], ptr[2] if parts == 2 else 0, x.size, parts)
        rt.stream_synchronize(0)
        rt.memcpy(hi.ctypes.data, ptr[1], hi.nbytes, Copy.DEVICE_TO_HOST)
        if parts == 2:
            rt.memcpy(lo.ctypes.data, ptr[2], lo.nbytes, Copy.DEVICE_TO_HOST)
    expected_hi = x.astype(np.float16)
    np.testing.assert_array_equal(hi.view(np.uint16), expected_hi.view(np.uint16))
    if parts == 2:
        expected_lo = (x - expected_hi.astype(np.float32)).astype(np.float16)
        np.testing.assert_array_equal(lo.view(np.uint16), expected_lo.view(np.uint16))


def test_selected_wmma_refuses_geometry_it_cannot_compute():
    """A shape the kernel cannot tile, or two parts without a residual input, is refused before any launch."""
    backend = _runnable()
    key = dict(backend=backend, layer='moe_linear', quant='gguf_nvfp4')
    one = resolve(**key, variant='selected_wmma_f16_f32_out')
    two = resolve(**key, variant='selected_wmma_f16_parts_f32_out')
    rt = get_hip_runtime()
    with ExitStack() as owned:
        def alloc(data):
            ptr = rt.malloc(max(data.nbytes, 16))
            owned.callback(rt.free, ptr)
            rt.memcpy(ptr, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
            return ptr

        page = alloc(np.zeros(4096, np.float32))
        start = alloc(np.array([0, 1], np.int64))
        group = alloc(np.array([0, 0], np.int32))
        weights = alloc(np.zeros(96 * 36, np.uint8))
        # 80 output columns are not a whole number of 32-column tiles.
        with pytest.raises(HipError):
            one(page, 0, start, group, group + 4, 0, 0, 0, weights, page, 1, 64, 80)
        with pytest.raises(HipError):
            two(page, 0, start, group, group + 4, 0, 0, 0, weights, page, 1, 64, 96)
        host = np.empty(64, np.float32)
        rt.memcpy(host.ctypes.data, page, host.nbytes, Copy.DEVICE_TO_HOST)
    assert not host.any()
