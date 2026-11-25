"""Grouped WMMA projection of selected raw NVFP4 experts against a dequantized NumPy reference."""

import ctypes
from contextlib import ExitStack

import numpy as np
import pytest

from hipengine.core.hip import HipError, HipMemcpyKind as Copy, get_hip_runtime
from hipengine.kernels.backends import hip_target_arch_for_backend, resolve_backend
from hipengine.kernels.hip_gfx1100.quant.gguf_nvfp4_selected_wmma import launch_route_schedule
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


def _allocate(owned, rt, data, upload=True):
    ptr = rt.malloc(max(data.nbytes, 16))
    owned.callback(rt.free, ptr)
    if upload:
        rt.memcpy(ptr, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
    return ptr


def _slot_order(ids, top):
    """The slot of each route when its row's routes are laid out in expert order."""
    rows = ids.reshape(-1, top)
    rank = np.argsort(np.argsort(rows, axis=1, kind='stable'), axis=1)
    return (rank + np.arange(rows.shape[0])[:, None] * top).astype(np.int32).ravel()


def _route_of_slot(ids, top, expert_in_slot):
    """The route (row and position in it) that holds each slot's expert."""
    rows = ids.reshape(-1, top)
    lookup = np.empty((rows.shape[0], int(ids.max()) + 1), np.int64)
    lookup[np.arange(rows.shape[0])[:, None], rows] = np.arange(top)
    token = np.arange(expert_in_slot.size) // top
    return token * top + lookup[token, expert_in_slot]


@pytest.mark.parametrize('experts,routes,skew,top', [(128, 1536, False, 8), (128, 2048, False, 8),
                                                     (128, 2048, True, 1), (2, 1536, True, 1),
                                                     (128, 65, False, 1)])
def test_route_schedule_groups_routes_like_the_host(experts, routes, skew, top):
    """The schedule built on the device is the host's: the same expert starts and the same 64-row
    groups. Every route gets one compact row on its token row, and its slot in its row follows the
    expert order, with its own mix scale. Routes of one expert may trade places among themselves."""
    backend = _runnable()
    rt = get_hip_runtime()
    rng = np.random.default_rng(routes + experts + top)
    tokens = routes // top
    if skew:
        # One expert takes 97% of the routes; the others share the rest.
        heavy = int(rng.integers(1, experts))
        probs = np.full(experts, 0.03 / max(experts - 1, 1))
        probs[heavy] = 0.97
        ids = rng.choice(experts, routes, p=probs)
    else:
        # Distinct experts inside a row, in the device router's order rather than expert order.
        ids = np.stack([rng.choice(experts, top, replace=False) for _ in range(tokens)]).ravel()
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    scaled = rng.uniform(0.0, 1.0, routes).astype(np.float32)
    _, start, group_expert, group_first = _groups(ids, experts)
    capacity = min(experts, routes) + -(-routes // 64)
    total = np.zeros(1, np.int32)
    got = {'start': np.full(experts + 1, -1, np.int64), 'group': np.full(capacity, -1, np.int32),
           'first': np.full(capacity, -1, np.int32), 'source': np.full(routes, -1, np.int32),
           'route': np.full(routes, -1, np.int32), 'mix': np.full(routes, np.nan, np.float32)}
    with ExitStack() as owned:
        device = {name: _allocate(owned, rt, data) for name, data in got.items()}
        device['total'] = _allocate(owned, rt, total, False)
        launch_route_schedule(_allocate(owned, rt, ids), _allocate(owned, rt, scaled), routes, experts, top,
                              device['start'], device['group'], device['first'], device['source'],
                              device['route'], device['mix'], device['total'], capacity,
                              target_arch=hip_target_arch_for_backend(backend))
        rt.stream_synchronize(0)
        for name, data in {**got, 'total': total}.items():
            rt.memcpy(data.ctypes.data, device[name], data.nbytes, Copy.DEVICE_TO_HOST)
    count = int(total[0])
    assert count == len(group_expert) <= capacity
    np.testing.assert_array_equal(got['start'], start)
    np.testing.assert_array_equal(got['group'][:count], group_expert)
    np.testing.assert_array_equal(got['first'][:count], group_first)
    assert (got['group'][count:] == -1).all() and (got['first'][count:] == -1).all()
    assert sorted(got['route'].tolist()) == list(range(routes))
    expert_in_slot = np.empty(routes, np.int64)
    expert_in_slot[got['route']] = np.searchsorted(got['start'], np.arange(routes), side='right') - 1
    # A compact row sits on the token of its slot, its scale is that route's, and each row's
    # slots follow the expert order.
    np.testing.assert_array_equal(got['source'], got['route'] // top)
    np.testing.assert_array_equal(got['mix'], scaled[_route_of_slot(ids, top, expert_in_slot)])
    assert (np.diff(expert_in_slot.reshape(-1, top), axis=1) > 0).all()


@pytest.mark.parametrize('parts', [1, 2])
def test_grouped_projection_reads_a_device_schedule_the_same(parts):
    """With a schedule built on the device (its group count, its token rows, its slots) a route's
    gate/up row and its down row are bitwise what the host schedule gives that route."""
    backend = _runnable()
    variant = 'selected_wmma_f16_parts_f32_out' if parts == 2 else 'selected_wmma_f16_f32_out'
    launch = resolve(backend=backend, layer='moe_linear', quant='gguf_nvfp4', variant=variant)
    rt = get_hip_runtime()
    rng = np.random.default_rng(611 + parts)
    experts, tokens, top, n, m = 24, 250, 8, 128, 96
    # Eight distinct experts per token in the device router's order.
    ids = np.stack([rng.choice(experts, top, replace=False) for _ in range(tokens)]).astype(np.int32).ravel()
    routes = ids.size
    order, start, group_expert, group_first = _groups(ids, experts)
    slot = _slot_order(ids, top)
    outer = rng.uniform(-0.5, 0.5, experts).astype(np.float32)
    scaled = rng.uniform(0.0, 1.0, routes).astype(np.float32)

    def halves(x):
        hi = x.astype(np.float16)
        return hi, (x - hi.astype(np.float32)).astype(np.float16)

    x_hi, x_lo = halves(rng.normal(0, 0.7, (tokens, n)).astype(np.float32))
    # The down stage reads one row per route, indexed by the route's slot in its row.
    y_hi, y_lo = halves(rng.normal(0, 0.7, (routes, n)).astype(np.float32))
    raw = rng.integers(0, 256, (experts, m, n // 64, 36), dtype=np.uint8)
    raw[..., :4] &= 63
    capacity = min(experts, routes) + -(-routes // 64)
    zeros = {'start': np.zeros(experts + 1, np.int64), 'group': np.zeros(capacity, np.int32),
             'first': np.zeros(capacity, np.int32), 'source': np.zeros(routes, np.int32),
             'route': np.zeros(routes, np.int32), 'mix': np.zeros(routes, np.float32)}
    total, route = np.zeros(1, np.int32), np.zeros(routes, np.int32)
    # Gate/up rows are compact (one per route); down rows are stored per route. Three rows past
    # the last one stay NaN.
    out = {(side, stage): np.full((routes, m) if stage == 0 else (routes + 3, m), np.nan, np.float32)
           for side in ('host', 'device') for stage in (0, 1)}
    with ExitStack() as owned:

        def up(data, upload=True):
            return _allocate(owned, rt, data, upload)

        x = (up(x_hi), up(x_lo) if parts == 2 else 0)
        y = (up(y_hi), up(y_lo) if parts == 2 else 0)
        raw_ptr, outer_ptr = up(raw), up(outer)
        ptr = {key: up(data) for key, data in out.items()}
        host = {'start': up(start), 'group': up(group_expert), 'first': up(group_first)}
        # Host schedule: compact row c is route order[c], on token row order[c] // top.
        launch(*x, host['start'], host['group'], host['first'], up((order // top).astype(np.int32)), 0, outer_ptr,
               raw_ptr, ptr['host', 0], len(group_expert), n, m)
        launch(*y, host['start'], host['group'], host['first'], up(slot[order]), up(order), outer_ptr, raw_ptr,
               ptr['host', 1], len(group_expert), n, m)
        device = {name: up(data, False) for name, data in zeros.items()}
        device['total'] = up(total, False)
        launch_route_schedule(up(ids), up(scaled), routes, experts, top, device['start'], device['group'],
                              device['first'], device['source'], device['route'], device['mix'],
                              device['total'], capacity, target_arch=hip_target_arch_for_backend(backend))
        # Device schedule: the grid is as tall as the capacity allows; the groups past the device's
        # count do nothing. The down stage reads and stores by the compact row's slot.
        launch(*x, device['start'], device['group'], device['first'], device['source'], 0, outer_ptr, raw_ptr,
               ptr['device', 0], capacity, n, m, group_count=device['total'])
        launch(*y, device['start'], device['group'], device['first'], device['route'], device['route'], outer_ptr,
               raw_ptr, ptr['device', 1], capacity, n, m, group_count=device['total'])
        rt.stream_synchronize(0)
        for key, data in out.items():
            rt.memcpy(data.ctypes.data, ptr[key], data.nbytes, Copy.DEVICE_TO_HOST)
        for data in (total, route):
            rt.memcpy(data.ctypes.data, device['total' if data is total else 'route'], data.nbytes,
                      Copy.DEVICE_TO_HOST)
    assert int(total[0]) == len(group_expert)
    # Down: the device stores a route's row in its slot, the host at the route's index.
    np.testing.assert_array_equal(out['device', 1][slot], out['host', 1][:routes])
    assert np.isnan(out['device', 1][routes:]).all() and np.isnan(out['host', 1][routes:]).all()
    # Gate/up: both read the token row of each route. The device's compact row c is the route
    # whose slot is route[c]; the host's row for that route is its position in the sorted order.
    position = np.empty(routes, np.int64)
    position[order] = np.arange(routes)
    expert_of_compact = np.searchsorted(start, np.arange(routes), side='right') - 1
    expert_in_slot = np.empty(routes, np.int64)
    expert_in_slot[route] = expert_of_compact
    np.testing.assert_array_equal(out['device', 0],
                                  out['host', 0][position[_route_of_slot(ids, top, expert_in_slot)[route]]])


@pytest.mark.parametrize('fmt,quant,qtype,block_size,block_bytes', [('q5_k', 'gguf_q5_k', Q.Q5_K, 256, 176),
                                                                     ('q8_0', 'gguf_q8_0', Q.Q8_0, 32, 34)])
@pytest.mark.parametrize('n,m', [(512, 96), (2304, 64)])
def test_kquant_selected_wmma_matches_dequantized_reference(fmt, quant, qtype, block_size, block_bytes, n, m):
    """Q5_K and Q8_0 weights come in two FP16 parts: the result is the FP32 reference to rounding."""
    backend = _runnable()
    launch = resolve(backend=backend, layer='moe_linear', quant=quant, variant='selected_wmma_f16_parts_f32_out')
    rt = get_hip_runtime()
    rng = np.random.default_rng(97 + n + block_bytes)
    experts, rows, top = 7, 150, 8
    ids = np.concatenate([rng.permutation(experts)[:top] for _ in range(rows)]).astype(np.int32)
    routes = ids.size
    order, start, group_expert, group_first = _groups(ids, experts)
    outer = rng.uniform(-0.5, 0.5, experts).astype(np.float32)
    source = (order // top).astype(np.int32)
    x = rng.normal(0, 0.7, (rows, n)).astype(np.float32)
    hi = x.astype(np.float16)
    lo = (x - hi.astype(np.float32)).astype(np.float16)
    blocks = n // block_size
    raw = rng.integers(0, 256, (experts, m, blocks, block_bytes), dtype=np.uint8)
    # The scales are FP16 values of a usable size; every packed field stays random.
    pairs = (experts, m, blocks, 2 if block_size == 256 else 1)
    scales = (rng.normal(0, 0.02, pairs) * 2.0 ** rng.integers(-6, 3, pairs)).astype(np.float16)
    raw[..., :scales.shape[-1] * 2] = scales.view(np.uint8).reshape(experts, m, blocks, -1)
    actual = np.full((routes + 3, m), np.nan, np.float32)
    with ExitStack() as owned:
        def alloc(data):
            ptr = rt.malloc(max(data.nbytes, 16))
            owned.callback(rt.free, ptr)
            rt.memcpy(ptr, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
            return ptr

        hi_ptr, lo_ptr, raw_ptr, out_ptr = (alloc(a) for a in (hi, lo, raw, actual))
        start_ptr, group_ptr, first_ptr, source_ptr, route_ptr, outer_ptr = (
            alloc(a) for a in (start, group_expert, group_first, source, order, outer))
        launch(hi_ptr, lo_ptr, start_ptr, group_ptr, first_ptr, source_ptr, route_ptr, outer_ptr, raw_ptr, out_ptr,
               len(group_expert), n, m)
        rt.memcpy(actual.ctypes.data, out_ptr, actual.nbytes, Copy.DEVICE_TO_HOST)
    dense = np.stack([dequantize_gguf_data(bank.reshape(m, -1), qtype).astype(np.float64) for bank in raw])
    assert np.isfinite(dense).all()
    value = hi.astype(np.float64) + lo.astype(np.float64)
    got = actual[order].astype(np.float64)
    for expert in range(experts):
        first, stop = start[expert], start[expert + 1]
        rows_of = source[first:stop]
        scale = float(outer[expert])
        expected = (dense[expert] @ value[rows_of].T).T * scale
        # The weight's residual part is FP16 too, and the product of the two residual parts is left
        # out: both are below 2^-21 of the product magnitudes. What remains is the FP32 rounding of
        # the sum, the same as for a reference that accumulated in FP32.
        bound = (np.abs(dense[expert]) @ np.abs(value[rows_of]).T).T * abs(scale) + 1e-30
        worst = float((np.abs(got[first:stop] - expected) / bound).max())
        assert worst <= 2e-5, worst
    untouched = np.ones(routes + 3, bool)
    untouched[order] = False
    assert np.isnan(actual[untouched]).all()
