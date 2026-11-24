"""gfx1100: one-row decode attention from a lag table is bit-exact with the block kernel."""

import ctypes
from contextlib import closing
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


@pytest.mark.parametrize("capacity,nq,nk,dim,extent,rel_width", [
    (7, 4, 2, 67, 15, 5), (33, 24, 8, 128, 33, 16), (64, 3, 3, 128, 32, 1), (5, 6, 2, 16, 9, 32),
    (130, 8, 4, 96, 64, 7)])
def test_one_row_decode_from_the_lag_table_matches_the_block_kernel(capacity, nq, nk, dim, extent, rel_width):
    runtime, arch = _device()
    from hipengine.core.hip import HipMemcpyKind
    from hipengine.core.memory import free, malloc
    from hipengine.kernels.hip_gfx1100.attention import inkling2_target_attention as attn

    rng = np.random.default_rng(196 + capacity)
    n = capacity + 13  # the ring wraps, and the positions cross the bias extent
    query, key, value, sink = (rng.standard_normal((n, *shape), dtype=np.float32) for shape in
                               ((nq, dim), (nk, dim), (nk, dim), (nq,)))
    rel = rng.standard_normal((n, nq, rel_width), dtype=np.float32)
    table = rng.standard_normal((extent, rel_width), dtype=np.float32)
    logits = rng.uniform(-32.0, 32.0, (n, nq * dim)).astype(np.float32)
    owner = dict(runtime=runtime, geometry=[(capacity, nq, nk, dim, extent)],
                 launch=partial(attn.launch_target_attention, target_arch=arch),
                 write=partial(attn.launch_cache_write, target_arch=arch),
                 block_table=partial(attn.launch_attention_block_table_sigmoid, target_arch=arch))
    with (
        closing(Inkling2DeviceAttention(**owner)) as control,
        closing(Inkling2DeviceAttention(
            **owner, decode_table=partial(attn.launch_decode_table_attention, target_arch=arch))) as decoded,
    ):
        buffers = [malloc(array.nbytes, runtime=runtime) for array in (query, key, value, rel, sink, logits, table)]
        outs = [malloc(query[0].nbytes, runtime=runtime) for _ in range(2)]
        try:
            for buffer, array in zip(buffers, (query, key, value, rel, sink, logits, table), strict=True):
                runtime.memcpy(buffer.ptr, array.ctypes.data, array.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            q, k, v, r, s, g, t = (buffer.ptr for buffer in buffers)

            def step(state, i, out):
                def offset(ptr, array):
                    return ptr + i * array[0].nbytes

                state.step_block_resident(0, i, offset(q, query), offset(k, key), offset(v, value), offset(r, rel),
                                          offset(s, sink), offset(g, logits), out, t, rel_width, 1)

            for i in range(n):
                step(control, i, outs[0].ptr)
                step(decoded, i, outs[1].ptr)
                runtime.stream_synchronize(0)
                expected, actual = (np.empty((nq, dim), np.float32) for _ in range(2))
                for out, host in zip(outs, (expected, actual), strict=True):
                    runtime.memcpy(host.ctypes.data, out.ptr, host.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            # Both owners stand after the last row (position n - 1 plus one row).
            assert decoded.positions[0] == control.positions[0] == n
        finally:
            for buffer in (*buffers, *outs):
                free(buffer, runtime=runtime)
