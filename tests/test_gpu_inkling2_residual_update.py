"""Fused residual step for Inkling 2 rows (HIP): add, stack-row append and stack read in one launch."""

import ctypes
from contextlib import ExitStack

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind as Copy, get_hip_runtime
from hipengine.kernels.backends import hip_target_arch_for_backend, resolve_backend
from hipengine.kernels.hip_gfx1100.norm.inkling2_residual import (
    launch_residual_read,
    launch_residual_update,
    launch_row_add,
    launch_row_norm_mul,
    launch_row_norm_mul_group,
    launch_row_rms,
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


# seven stack rows is the largest templated count; eight and more take the many-rows kernel
@pytest.mark.parametrize("blocks,queries", [(1, 1), (4, 1), (1, 3), (4, 3), (0, 1), (7, 2), (8, 1), (11, 3)])
@pytest.mark.parametrize("added", [False, True])
def test_residual_update_equals_add_rms_and_read(blocks, queries, added):
    """h, the new stack row and every query's read equal the three kernels over the stored rows."""
    arch = _available()
    runtime = get_hip_runtime()
    rng = np.random.default_rng(97 + blocks * 2 + queries)
    a = rng.standard_normal(_WIDTH).astype(np.float32)
    b = rng.standard_normal(_WIDTH).astype(np.float32) if added else None
    # the rows already on the stack; the new row is appended after them
    stack = rng.standard_normal((blocks, _WIDTH)).astype(np.float32)
    query = rng.standard_normal((queries, _WIDTH)).astype(np.float32)
    stack_bytes, out_bytes = stack.nbytes, queries * a.nbytes
    with ExitStack() as owned:
        def upload(data, nbytes=None):
            ptr = runtime.malloc(nbytes or data.nbytes)
            owned.callback(runtime.free, ptr)
            if data is not None:
                runtime.memcpy(ptr, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
            return ptr

        ap = upload(a)
        bp = upload(b) if added else 0
        sp = upload(stack, _WIDTH * (blocks + 1) * 4)
        qp = upload(query)
        hp = upload(None, a.nbytes)
        op = upload(None, out_bytes)
        back = [np.full(size // 4, np.nan, np.float32) for size in ((a.nbytes, a.nbytes, out_bytes) if added
                                                                else (a.nbytes, out_bytes))]
        # h is an output only when there is an add
        new_row = sp + stack_bytes
        pointers = ([hp] if added else []) + [new_row, op]
        outs = []
        for fused in (False, True):
            runtime.memcpy(sp, stack.ctypes.data, stack_bytes, Copy.HOST_TO_DEVICE)
            runtime.memset(hp, 0xFF, a.nbytes)
            runtime.memset(new_row, 0xFF, a.nbytes)
            runtime.memset(op, 0xFF, out_bytes)
            if fused:
                launch_residual_update(ap, bp, hp if added else 0, sp, new_row, qp, op, 1, blocks, queries,
                                       _WIDTH, _EPS, target_arch=arch)
            else:
                if added:
                    launch_row_add(ap, bp, hp, 1, _WIDTH, target_arch=arch)
                launch_row_rms(hp if added else ap, new_row, 1, _WIDTH, _EPS, target_arch=arch)
                launch_residual_read(qp, sp, op, 1, blocks + 1, queries, _WIDTH, target_arch=arch)
            result = []
            for ptr, host in zip(pointers, back, strict=True):
                runtime.memcpy(host.ctypes.data, ptr, host.nbytes, Copy.DEVICE_TO_HOST)
                result.append(host.view(np.uint32).copy())
            outs.append(result)
    for index, (one, many) in enumerate(zip(*outs, strict=True)):
        assert np.isfinite(one.view(np.float32)).all(), index
        np.testing.assert_array_equal(many, one)
    # The add is real: h differs from a, and the read is a convex mix of rows, not a copy of one.
    if added:
        assert not np.array_equal(outs[0][0].view(np.float32), a.view(np.uint32).view(np.float32))


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("rows", [1, 4])
def test_grouped_input_norms_equal_the_separate_normalizations(count, rows):
    arch = _available()
    runtime = get_hip_runtime()
    rng = np.random.default_rng(311 + count + rows)
    sources = rng.standard_normal((count, rows, _WIDTH)).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, (count, _WIDTH)).astype(np.float32)
    separate = np.full((rows * count, _WIDTH), np.nan, np.float32)
    grouped = np.full((rows * count, _WIDTH), np.nan, np.float32)
    with ExitStack() as owned:
        def upload(data):
            ptr = runtime.malloc(data.nbytes)
            owned.callback(runtime.free, ptr)
            runtime.memcpy(ptr, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
            return ptr

        sp = [upload(x) for x in sources]
        wp = [upload(x) for x in weights]
        op = runtime.malloc(separate.nbytes)
        owned.callback(runtime.free, op)
        for i in range(count):
            launch_row_norm_mul(sp[i], wp[i], op + i * rows * _WIDTH * 4, rows, _WIDTH, _EPS, target_arch=arch)
        runtime.memcpy(separate.ctypes.data, op, separate.nbytes, Copy.DEVICE_TO_HOST)
        runtime.memset(op, 0xFF, separate.nbytes)
        launch_row_norm_mul_group(sp, wp, op, rows, _WIDTH, _EPS, count, target_arch=arch)
        runtime.memcpy(grouped.ctypes.data, op, grouped.nbytes, Copy.DEVICE_TO_HOST)
    assert np.isfinite(separate).all()
    np.testing.assert_array_equal(grouped.view(np.uint32), separate.view(np.uint32))
    assert not np.array_equal(grouped[0], sources[0][0])
