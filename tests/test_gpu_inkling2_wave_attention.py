"""Inkling 2 block attention with one 32-lane group per (row, KV head) (HIP): the tiled kernel's results."""

import ctypes
from contextlib import ExitStack, closing
from functools import partial

import numpy as np
import pytest

from hipengine.runtime.inkling2_device_attention import Inkling2DeviceAttention


def _device():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.backends import hip_target_arch_for_backend, resolve_backend

    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    return get_hip_runtime(), hip_target_arch_for_backend(backend)


# Rows of the middle block, ring capacity, query heads, KV heads, relative width: three, two and
# one query head per KV head, small capacities that wrap, and the widths the kernel takes.
_CASES = [
    (130, 1024, 24, 8, 32),
    (64, 130, 24, 8, 16),
    (33, 64, 24, 8, 32),
    (17, 40, 6, 2, 8),
    (5, 7, 8, 8, 4),
    (3, 5, 4, 1, 12),
    (1, 5, 3, 1, 20),
]


def _inputs(rows, capacity, nq, nk, rel_width, seed, count=None, dim=128):
    """Query, key, value, sink, relative score rows, lag table and gate logits of one case."""
    extent = min(capacity, 50)          # lags at and beyond it carry no table bias
    n = count or max(capacity, 2 + rows) + 13
    rng = np.random.default_rng(seed)
    query, key, value, sink = (rng.standard_normal((n, *shape), dtype=np.float32) for shape in
                               ((nq, dim), (nk, dim), (nk, dim), (nq,)))
    rel = rng.standard_normal((n, nq, rel_width), dtype=np.float32)
    table = rng.standard_normal((extent, rel_width), dtype=np.float32)
    logits = rng.uniform(-6.0, 6.0, (n, nq * dim)).astype(np.float32)
    return dim, extent, n, (query, key, value, sink, rel, table, logits)


def _blocks_of(rows, n):
    """A first block of two rows, the middle block, and the rest."""
    return [(0, 2), (2, 2 + rows), (2 + rows, n)]


def _run(variants, capacity, nq, nk, dim, extent, arrays, blocks, rel_width, prepare=None):
    """Every block on one owner per named block kernel; each owner's outputs per block.

    `variants` names the `launch_attention_block_<name>` of each owner; `prepare` may change an
    owner's ring before the first block, and receives the owner.
    """
    from hipengine.core.hip import HipMemcpyKind
    from hipengine.core.memory import free, malloc
    from hipengine.kernels.hip_gfx1100.attention import inkling2_target_attention as attn

    runtime, arch = _device()
    query, key, value, sink, rel, table, logits = arrays
    widest = max(j - i for i, j in blocks)
    results = {name: [] for name in variants}
    with ExitStack() as owned:
        owners = {}
        for name, variant in variants.items():
            state = owned.enter_context(closing(Inkling2DeviceAttention(
                runtime=runtime, geometry=[(capacity, nq, nk, dim, extent)],
                launch=partial(attn.launch_tiled_attention, target_arch=arch),
                write=partial(attn.launch_cache_write, target_arch=arch),
                block_table=partial(getattr(attn, f"launch_attention_block_{variant}"), target_arch=arch))))
            owners[name] = state
            if prepare is not None:
                prepare(state)
        pointers = [malloc(array.nbytes, runtime=runtime) for array in arrays]
        outputs = {name: malloc(query[0].nbytes * widest, runtime=runtime) for name in owners}
        try:
            for buffer, array in zip(pointers, arrays, strict=True):
                runtime.memcpy(buffer.ptr, array.ctypes.data, array.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            device = [buffer.ptr for buffer in pointers]

            def at(index, i):
                return device[index] + i * arrays[index][0].nbytes

            for i, j in blocks:
                for name, state in owners.items():
                    state.step_block_resident(0, i, at(0, i), at(1, i), at(2, i), at(4, i), at(3, i), at(6, i),
                                              outputs[name].ptr, device[5], rel_width, j - i)
                runtime.stream_synchronize(0)
                for name in owners:
                    host = np.full((j - i, nq, dim), np.nan, np.float32)
                    runtime.memcpy(host.ctypes.data, outputs[name].ptr, host.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
                    results[name].append(host)
        finally:
            for buffer in (*pointers, *outputs.values()):
                free(buffer, runtime=runtime)
        results["positions"] = sorted(state.positions[0] for state in owners.values())
    return results



@pytest.mark.parametrize("rows,capacity,nq,nk,rel_width", _CASES)
def test_wave_block_attention_matches_the_tiled_kernel(rows, capacity, nq, nk, rel_width):
    """Every block equals the tiled kernel's up to rounding, and a second owner repeats it bit for bit."""
    dim, extent, n, arrays = _inputs(rows, capacity, nq, nk, rel_width, seed=711 + rows + capacity)
    results = _run({"tiled": "table_sigmoid", "waves": "table_waves", "again": "table_waves"}, capacity, nq, nk, dim,
                   extent, arrays, _blocks_of(rows, n), rel_width)
    for expected, actual, repeat in zip(results["tiled"], results["waves"], results["again"], strict=True):
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(repeat.view(np.uint32), actual.view(np.uint32))
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=5e-5)
    assert results["positions"] == [n] * 3           # every owner stands after the last row


def test_wave_block_attention_matches_float64_causal_attention():
    """Without a wrap the blocks are the causal attention over the lag table, the sinks and the gate."""
    nq, nk, rel_width, rows = 24, 8, 32, 130
    dim, extent, n, arrays = _inputs(rows, 1024, nq, nk, rel_width, seed=99, count=150)
    query, key, value, sink, rel, table, logits = arrays
    blocks = _blocks_of(rows, n)
    results = _run({"tiled": "table_sigmoid", "waves": "table_waves"}, 1024, nq, nk, dim, extent, arrays, blocks,
                   rel_width)
    groups = nq // nk
    for (i, j), expected, actual in zip(blocks, results["tiled"], results["waves"], strict=True):
        for r in range(i, j):
            # The query heads of one KV head share its keys.
            qh = query[r].reshape(nk, groups, dim).astype(np.float64)
            kh = key[:r + 1].reshape(r + 1, nk, dim).astype(np.float64).transpose(1, 0, 2)
            score = np.einsum("hgd,htd->hgt", qh, kh) / np.sqrt(dim)
            bias = rel[r].astype(np.float64) @ table.astype(np.float64).T
            lag = r - np.arange(r + 1)
            score[:, :, lag < extent] += bias.reshape(nk, groups, extent)[:, :, lag[lag < extent]]
            joined = np.concatenate((score, sink[r].reshape(nk, groups, 1).astype(np.float64)), axis=-1)
            joined -= joined.max(axis=-1, keepdims=True)
            probability = np.exp(joined)
            probability /= probability.sum(axis=-1, keepdims=True)
            vh = value[:r + 1].reshape(r + 1, nk, dim).astype(np.float64).transpose(1, 0, 2)
            out = np.einsum("hgt,htd->hgd", probability[..., :-1], vh).reshape(nq, dim)
            out *= 1.0 / (1.0 + np.exp(-logits[r].reshape(nq, dim).astype(np.float64)))
            np.testing.assert_allclose(actual[r - i], out, rtol=1e-4, atol=2e-6)
            np.testing.assert_allclose(expected[r - i], out, rtol=1e-4, atol=2e-6)


def test_wave_block_attention_skips_evicted_slots_through_a_permuted_slot_map():
    """Evicted slots are left out, and both kernels read the rows through the same slot map."""
    nq, nk, rel_width, rows, capacity, dim = 24, 8, 32, 33, 64, 128
    _, extent, n, arrays = _inputs(rows, capacity, nq, nk, rel_width, seed=4242)
    rng = np.random.default_rng(4711)
    offsets = rng.permutation(capacity).astype(np.int32)
    evicted = (rng.random(capacity) < 0.3).astype(np.uint8)
    evicted[:3] = 0                                  # the ring keeps its newest slots
    assert 0 < int(evicted.sum()) < capacity

    def prepare(state):
        from hipengine.core.hip import HipMemcpyKind

        state.banks[0] = state._make_bank(capacity, nk, dim)
        spans = state.banks[0][2]
        runtime = state.rt
        runtime.memcpy(spans.base_offsets.ptr, offsets.ctypes.data, offsets.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        runtime.memcpy(spans.evict_mask.ptr, evicted.ctypes.data, evicted.nbytes, HipMemcpyKind.HOST_TO_DEVICE)

    results = _run({"tiled": "table_sigmoid", "waves": "table_waves"}, capacity, nq, nk, dim, extent, arrays,
                   _blocks_of(rows, n), rel_width, prepare=prepare)
    for expected, actual in zip(results["tiled"], results["waves"], strict=True):
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=5e-5)


def test_unserved_shapes_take_the_tiled_kernel_bit_for_bit():
    from hipengine.kernels.hip_gfx1100.attention import inkling2_target_attention as attn

    nq, nk, dim, rel_width, rows, capacity = 4, 2, 67, 5, 9, 17
    assert not attn.waves_serve(nq, nk, dim, rel_width)
    extent, n = 15, capacity + 13
    rng = np.random.default_rng(196)
    query, key, value, sink = (rng.standard_normal((n, *shape), dtype=np.float32) for shape in
                               ((nq, dim), (nk, dim), (nk, dim), (nq,)))
    rel = rng.standard_normal((n, nq, rel_width), dtype=np.float32)
    table = rng.standard_normal((extent, rel_width), dtype=np.float32)
    logits = rng.uniform(-6.0, 6.0, (n, nq * dim)).astype(np.float32)
    results = _run({"tiled": "table_sigmoid", "routed": "table_waves"}, capacity, nq, nk, dim, extent,
                   (query, key, value, sink, rel, table, logits), _blocks_of(rows, n), rel_width)
    for expected, actual in zip(results["tiled"], results["routed"], strict=True):
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
