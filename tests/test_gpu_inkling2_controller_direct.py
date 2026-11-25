"""In-place Inkling 2 loop-controller kernels (HIP): the packed kernels' results over separate rows."""

import ctypes
from contextlib import ExitStack

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind as Copy, get_hip_runtime
from hipengine.kernels.backends import hip_target_arch_for_backend, resolve_backend
from hipengine.kernels.hip_gfx1100.norm.inkling2_controller import (
    launch_controller_blend_norm,
    launch_controller_blend_norm_direct,
    launch_controller_input,
    launch_controller_input_direct,
)

_WIDTH, _EPS = 2304, 1e-6


def _available():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    return hip_target_arch_for_backend(backend)


@pytest.mark.parametrize("count,rows", [(1, 1), (3, 1), (1, 2), (2, 3), (3, 5)])
def test_in_place_controller_kernels_equal_the_packed_kernels(count, rows):
    """Inputs, messages and norm weights in separate allocations per controller: the packed kernels' bits."""
    arch = _available()
    runtime = get_hip_runtime()
    rng = np.random.default_rng(401 + count * 8 + rows)
    inputs = rng.standard_normal((count, rows, _WIDTH)).astype(np.float32)
    messages = rng.standard_normal((count, rows, _WIDTH)).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, (count, _WIDTH)).astype(np.float32)
    # the gate projections write `rows` rows per controller, consecutively
    forget = rng.standard_normal((count, rows, _WIDTH)).astype(np.float32)
    gain = rng.standard_normal((count, rows, _WIDTH)).astype(np.float32)
    down = np.empty((count * rows, 2 * _WIDTH), np.float32)
    out = np.empty((count * rows, _WIDTH), np.float32)
    results = []
    with ExitStack() as owned:

        def upload(data):
            ptr = runtime.malloc(data.nbytes)
            owned.callback(runtime.free, ptr)
            runtime.memcpy(ptr, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
            return ptr

        packed = [upload(inputs), upload(messages), upload(weights)]
        # separate allocations, as the resident rows are: no two controllers share one buffer
        separate = [[upload(block[i]) for i in range(count)] for block in (inputs, messages, weights)]
        fp, gp, dp, op = upload(forget), upload(gain), upload(down), upload(out)
        for direct in (False, True):
            runtime.memset(dp, 0xFF, down.nbytes)
            runtime.memset(op, 0xFF, out.nbytes)
            if direct:
                launch_controller_input_direct(separate[0], separate[1], dp, rows, _WIDTH, count, _EPS,
                                               target_arch=arch)
                launch_controller_blend_norm_direct(separate[0], separate[1], fp, gp, separate[2], op, rows, _WIDTH,
                                                    count, _EPS, target_arch=arch)
            else:
                launch_controller_input(*packed[:2], dp, rows, _WIDTH, count, _EPS, target_arch=arch)
                launch_controller_blend_norm(packed[0], packed[1], fp, gp, packed[2], op, rows, _WIDTH, count, _EPS,
                                             target_arch=arch)
            got = []
            for ptr, host in ((dp, down), (op, out)):
                runtime.memcpy(host.ctypes.data, ptr, host.nbytes, Copy.DEVICE_TO_HOST)
                got.append(host.view(np.uint32).copy())
            results.append(got)
    for index, (packed_result, direct_result) in enumerate(zip(*results, strict=True)):
        assert np.isfinite(packed_result.view(np.float32)).all(), index
        np.testing.assert_array_equal(direct_result, packed_result)
    # The controllers read different rows: a controller that read another's input would show here.
    if count > 1:
        blended = out.view(np.uint32).reshape(count, rows, -1)
        assert not np.array_equal(blended[0], blended[-1])
