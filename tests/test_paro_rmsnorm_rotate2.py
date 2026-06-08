"""GPU bit-exactness test for the M15.4 fused RMSNorm + paro_rotate2 kernel.

The fused kernel must be byte-identical to ``paro_rmsnorm_out_fp16`` followed by
``paro_rotate2_fp16`` so the verifier output (and thus exact-AR) is unchanged.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest


def _has_gfx1100() -> bool:
    try:
        from hipengine.core.hip import get_hip_runtime
    except Exception:
        return False
    try:
        get_hip_runtime()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gfx1100(), reason="gfx1100 HIP runtime not available")


@pytest.fixture(scope="module")
def _libs():
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.norm.rmsnorm import build_qwen35_rmsnorm
    from hipengine.kernels.hip_gfx1100.rotary.paro_rotate import build_paro_rotate

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        return (
            build_paro_rotate(load=True, compiler_version=compiler_version),
            build_qwen35_rmsnorm(load=True, compiler_version=compiler_version),
        )


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()


def _upload(runtime, bufs, array):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    arr = np.ascontiguousarray(array)
    buf = malloc(max(arr.nbytes, 4), runtime=runtime)
    bufs.append(buf)
    if arr.nbytes:
        copy_host_to_device(buf, host_array_ptr(arr), runtime=runtime)
    return buf


def _alloc(runtime, bufs, nbytes):
    from hipengine.core.memory import malloc

    buf = malloc(max(nbytes, 4), runtime=runtime)
    bufs.append(buf)
    return buf


def _download(runtime, buf, shape, dtype):
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    arr = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(arr), buf, runtime=runtime)
    return arr


def _free(runtime, bufs) -> None:
    from hipengine.core.memory import free

    for buf in reversed(bufs):
        free(buf, runtime=runtime)


@pytest.mark.parametrize("tokens", [1, 3, 4, 6, 8])
@pytest.mark.parametrize("hidden", [512, 1024, 2048, 4096])
def test_fused_rmsnorm_rotate2_matches_unfused(_libs, _runtime, tokens: int, hidden: int) -> None:
    from hipengine.kernels.hip_gfx1100.norm.rmsnorm import paro_rmsnorm_out_fp16
    from hipengine.kernels.hip_gfx1100.rotary.paro_rotate import (
        paro_rmsnorm_rotate2_fp16,
        paro_rotate2_fp16,
    )

    rotate_lib, rms_lib = _libs
    rng = np.random.default_rng(0x15A4 + tokens * 31 + hidden)
    group_size = 128
    krot = 3
    eps = 1e-6
    half = hidden // 2
    fp16 = np.dtype(np.float16).itemsize

    x = (rng.standard_normal((tokens, hidden)).astype(np.float32) * 0.1).astype(np.float16)
    ln_w = (rng.standard_normal(hidden).astype(np.float32) * 0.2).astype(np.float16)

    def _disjoint_pairs() -> np.ndarray:
        # Real PARO rotations are Givens butterflies: per (round, group) the 64
        # lanes' (i, j) pairs form a perfect matching of [0, group_size), so no
        # two lanes write the same buffer slot (no race).  Random pairs would
        # race and make even the unfused kernel nondeterministic.
        groups = hidden // group_size
        out = np.empty((krot, hidden), dtype=np.int16)
        for r in range(krot):
            for g in range(groups):
                perm = rng.permutation(group_size).astype(np.int16)
                out[r, g * group_size : (g + 1) * group_size] = perm
        return out

    pairs0 = _disjoint_pairs()
    pairs1 = _disjoint_pairs()
    theta0 = (rng.standard_normal((krot, half)).astype(np.float32) * 0.5).astype(np.float16)
    theta1 = (rng.standard_normal((krot, half)).astype(np.float32) * 0.5).astype(np.float16)
    sc0 = rng.uniform(0.5, 1.5, size=hidden).astype(np.float16)
    sc1 = rng.uniform(0.5, 1.5, size=hidden).astype(np.float16)

    bufs = []
    try:
        xd = _upload(_runtime, bufs, x)
        lnd = _upload(_runtime, bufs, ln_w)
        p0 = _upload(_runtime, bufs, pairs0)
        p1 = _upload(_runtime, bufs, pairs1)
        t0 = _upload(_runtime, bufs, theta0)
        t1 = _upload(_runtime, bufs, theta1)
        s0 = _upload(_runtime, bufs, sc0)
        s1 = _upload(_runtime, bufs, sc1)
        normed = _alloc(_runtime, bufs, tokens * hidden * fp16)
        ref0 = _alloc(_runtime, bufs, tokens * hidden * fp16)
        ref1 = _alloc(_runtime, bufs, tokens * hidden * fp16)
        fast_norm = _alloc(_runtime, bufs, tokens * hidden * fp16)
        fast0 = _alloc(_runtime, bufs, tokens * hidden * fp16)
        fast1 = _alloc(_runtime, bufs, tokens * hidden * fp16)

        # Unfused reference: rmsnorm -> rotate2.
        paro_rmsnorm_out_fp16(
            xd.ptr, lnd.ptr, normed.ptr, tokens, hidden, eps,
            library=rms_lib, runtime=_runtime,
        )
        paro_rotate2_fp16(
            normed.ptr, ref0.ptr, ref1.ptr, p0.ptr, p1.ptr, t0.ptr, t1.ptr, s0.ptr, s1.ptr,
            tokens, hidden, group_size, krot, library=rotate_lib, runtime=_runtime,
        )
        # Fused (also writes the unrotated RMSNorm output to fast_norm).
        paro_rmsnorm_rotate2_fp16(
            xd.ptr, lnd.ptr, fast_norm.ptr, fast0.ptr, fast1.ptr, p0.ptr, p1.ptr, t0.ptr, t1.ptr, s0.ptr, s1.ptr,
            eps, tokens, hidden, group_size, krot, library=rotate_lib, runtime=_runtime,
        )
        _runtime.stream_synchronize(0)

        rn = _download(_runtime, normed, (tokens, hidden), np.float16)
        r0 = _download(_runtime, ref0, (tokens, hidden), np.float16)
        r1 = _download(_runtime, ref1, (tokens, hidden), np.float16)
        fn_ = _download(_runtime, fast_norm, (tokens, hidden), np.float16)
        f0 = _download(_runtime, fast0, (tokens, hidden), np.float16)
        f1 = _download(_runtime, fast1, (tokens, hidden), np.float16)
        assert np.array_equal(rn.view(np.uint16), fn_.view(np.uint16))
        assert np.array_equal(r0.view(np.uint16), f0.view(np.uint16))
        assert np.array_equal(r1.view(np.uint16), f1.view(np.uint16))
    finally:
        _free(_runtime, bufs)
