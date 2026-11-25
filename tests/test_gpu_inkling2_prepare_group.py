"""Fused attention preparation for Inkling 2 rows (HIP): one launch for four separate kernels' arithmetic."""

import ctypes
from contextlib import ExitStack

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.kernels.backends import hip_target_arch_for_backend, resolve_backend
from hipengine.kernels.hip_gfx1100.norm.inkling2_attention_prepare import (
    launch_prepare_clip,
    launch_prepare_group,
    launch_prepare_rms,
)

_DIM, _HEADS, _KV_HEADS, _REL_WIDTH = 128, 24, 8, 16


def _available():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    return hip_target_arch_for_backend(backend)


@pytest.mark.parametrize("rows", [1, 5])
@pytest.mark.parametrize("logscale", [False, True])
def test_prepare_group_equals_the_four_separate_launches(rows, logscale):
    from hipengine.core.hip import HipMemcpyKind as Copy

    arch = _available()
    runtime = get_hip_runtime()
    rng = np.random.default_rng(47 + rows)
    sources = (rng.standard_normal((rows, _HEADS * _DIM)), rng.standard_normal((rows, _KV_HEADS * _DIM)),
               rng.standard_normal((rows, _KV_HEADS * _DIM)), rng.standard_normal((rows, _HEADS * _REL_WIDTH)))
    sources = [x.astype(np.float32) for x in sources]
    query_scale = rng.uniform(0.5, 1.5, _HEADS * _DIM).astype(np.float32)
    relative_scale = rng.uniform(0.5, 1.5, _HEADS * _REL_WIDTH).astype(np.float32)
    threshold, relative_threshold, eps = 22.6274166, 8.0, 1e-6
    alpha, floor, first = 0.1, 8192.0, 1718
    shapes = [(x.size,) for x in sources]
    separate = [np.full(shape, np.nan, np.float32) for shape in shapes]
    grouped = [np.full(shape, np.nan, np.float32) for shape in shapes]
    with ExitStack() as owned:
        def upload(data):
            ptr = runtime.malloc(data.nbytes)
            owned.callback(runtime.free, ptr)
            runtime.memcpy(ptr, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
            return ptr

        src = [upload(x) for x in sources]
        dst = [runtime.malloc(x.nbytes) for x in separate]
        for ptr in dst:
            owned.callback(runtime.free, ptr)
        qs, rs = upload(query_scale), upload(relative_scale)
        common = dict(target_arch=arch, alpha=alpha, floor=floor, first=first, logscale=logscale)
        launch_prepare_clip(src[0], dst[0], qs, rows, _DIM, _HEADS, threshold, **common)
        launch_prepare_clip(src[1], dst[1], 0, rows, _DIM, _KV_HEADS, threshold, target_arch=arch)
        launch_prepare_rms(src[2], dst[2], rows, _DIM, _KV_HEADS, eps, target_arch=arch)
        launch_prepare_clip(src[3], dst[3], rs, rows, _REL_WIDTH, _HEADS, relative_threshold, **common)
        for out, ptr in zip(separate, dst, strict=True):
            runtime.memcpy(out.ctypes.data, ptr, out.nbytes, Copy.DEVICE_TO_HOST)
        launch_prepare_group(src[0], dst[0], qs, _HEADS, threshold, key=src[1], key_out=dst[1],
                             key_heads=_KV_HEADS, key_threshold=threshold, value=src[2], value_out=dst[2],
                             value_heads=_KV_HEADS, eps=eps, relative=src[3], relative_out=dst[3],
                             relative_scale=rs, relative_heads=_HEADS, relative_threshold=relative_threshold,
                             relative_width=_REL_WIDTH, dim=_DIM, rows=rows, target_arch=arch,
                             alpha=alpha, floor=floor, first=first, logscale=logscale)
        for out, ptr in zip(grouped, dst, strict=True):
            runtime.memcpy(out.ctypes.data, ptr, out.nbytes, Copy.DEVICE_TO_HOST)
    for index, (one, many) in enumerate(zip(separate, grouped, strict=True)):
        assert np.isfinite(one).all(), index
        np.testing.assert_array_equal(many.view(np.uint32), one.view(np.uint32))
    # The group is not a copy of its input: the query rows come out clipped and scaled.
    assert not np.array_equal(grouped[0], sources[0])
