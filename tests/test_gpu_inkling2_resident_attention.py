import ctypes
from contextlib import closing
from functools import partial

import numpy as np
import pytest

from hipengine.runtime.inkling2_device_attention import Inkling2DeviceAttention
from hipengine.runtime.inkling2_mini_reference import _clip, _rms


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


@pytest.mark.parametrize("capacity", [3, 17, 49])
def test_lag_table_block_attention_matches_dense_bias_blocks(capacity):
    runtime, arch = _device()
    from hipengine.core.hip import HipMemcpyKind
    from hipengine.core.memory import free, malloc
    from hipengine.kernels.hip_gfx1100.attention import inkling2_target_attention as attn

    nq, nk, dim, extent, rel_width = 4, 2, 67, 15, 5
    rng = np.random.default_rng(196)
    n = capacity + 13
    query, key, value, sink = (rng.standard_normal((n, *shape), dtype=np.float32) for shape in
                               ((nq, dim), (nk, dim), (nk, dim), (nq,)))
    rel = rng.standard_normal((n, nq, rel_width), dtype=np.float32)
    table = rng.standard_normal((extent, rel_width), dtype=np.float32)
    logits = rng.uniform(-32.0, 32.0, (n, nq * dim)).astype(np.float32)
    bias = (rel @ table.T)[..., :extent]
    owner = dict(runtime=runtime, geometry=[(capacity, nq, nk, dim, extent)],
                 launch=partial(attn.launch_tiled_attention, target_arch=arch),
                 write=partial(attn.launch_cache_write, target_arch=arch),
                 block=partial(attn.launch_attention_block, target_arch=arch),
                 block_sigmoid=partial(attn.launch_attention_block_sigmoid, target_arch=arch))
    with (
        closing(Inkling2DeviceAttention(**owner)) as control,
        closing(Inkling2DeviceAttention(
            **owner, block_table=partial(attn.launch_attention_block_table_sigmoid, target_arch=arch))) as tabled,
    ):
        buffers = [malloc(array.nbytes, runtime=runtime) for array in (query, key, value, rel, sink, logits, table)]
        out = malloc(query.nbytes, runtime=runtime)
        try:
            for buffer, array in zip(buffers, (query, key, value, rel, sink, logits, table), strict=True):
                runtime.memcpy(buffer.ptr, array.ctypes.data, array.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
            q, k, v, r, s, g, t = (buffer.ptr for buffer in buffers)
            for i, j in ((0, 2), (2, capacity + 6), (capacity + 6, n - 1)):
                expected = control.step_block(0, i, query[i:j], key[i:j], value[i:j], bias[i:j], sink[i:j],
                                              gate_logits=logits[i:j])
                offset = lambda ptr, array: ptr + i * array[0].nbytes
                tabled.step_block_resident(0, i, offset(q, query), offset(k, key), offset(v, value), offset(r, rel),
                                           offset(s, sink), offset(g, logits), out.ptr, t, rel_width, j - i)
                runtime.stream_synchronize(0)
                actual = np.empty_like(expected)
                runtime.memcpy(actual.ctypes.data, out.ptr, actual.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
                np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-6)
            assert tabled.positions[0] == control.positions[0] == n - 1
        finally:
            for buffer in (*buffers, out):
                free(buffer, runtime=runtime)


@pytest.mark.parametrize("dim,heads", [(67, 4), (128, 3), (16, 7)])
def test_prepare_kernels_match_host_clip_and_rms(dim, heads):
    runtime, arch = _device()
    from hipengine.core.hip import HipMemcpyKind
    from hipengine.core.memory import free, malloc
    from hipengine.kernels.hip_gfx1100.norm import inkling2_attention_prepare as prep

    rng = np.random.default_rng(7)
    rows, threshold, eps = 9, 1.7, 1e-6
    x = (rng.standard_normal((rows, heads, dim)) * rng.uniform(0.2, 3.0, (heads, dim))).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, (heads, dim)).astype(np.float32)
    x[0, 0] *= 20.0  # one head far past the threshold, so the clip factor is not 1 everywhere
    source = malloc(x.nbytes, runtime=runtime)
    scales = malloc(scale.nbytes, runtime=runtime)
    target = malloc(x.nbytes, runtime=runtime)
    runtime.memcpy(source.ptr, x.ctypes.data, x.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
    runtime.memcpy(scales.ptr, scale.ctypes.data, scale.nbytes, HipMemcpyKind.HOST_TO_DEVICE)

    def run(launch):
        launch()
        runtime.stream_synchronize(0)
        actual = np.empty_like(x)
        runtime.memcpy(actual.ctypes.data, target.ptr, actual.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        return actual

    try:
        clipped = run(lambda: prep.launch_prepare_clip(source.ptr, target.ptr, scales.ptr, rows, dim, heads, threshold,
                                                       target_arch=arch))
        np.testing.assert_allclose(clipped, _clip(x * scale, threshold), rtol=3e-6, atol=2e-6)
        # Log position scale: 1 + alpha * log(max(1, position / floor)), with the first row at position 6.
        positions = 6 + np.arange(rows)
        factor = (1 + 0.1 * np.log(np.maximum(1, positions / 8))).astype(np.float32)
        scaled = run(lambda: prep.launch_prepare_clip(source.ptr, target.ptr, scales.ptr, rows, dim, heads, threshold,
                                                      alpha=0.1, floor=8, first=6, logscale=True, target_arch=arch))
        np.testing.assert_allclose(scaled, _clip(x * scale, threshold) * factor[:, None, None], rtol=3e-6, atol=2e-6)
        unscaled = run(lambda: prep.launch_prepare_clip(source.ptr, target.ptr, 0, rows, dim, heads, threshold,
                                                        target_arch=arch))
        np.testing.assert_allclose(unscaled, _clip(x, threshold), rtol=3e-6, atol=2e-6)
        normalized = run(lambda: prep.launch_prepare_rms(source.ptr, target.ptr, rows, dim, heads, eps, target_arch=arch))
        np.testing.assert_allclose(normalized, _rms(x, eps), rtol=3e-6, atol=2e-6)
    finally:
        for buffer in (source, scales, target):
            free(buffer, runtime=runtime)
