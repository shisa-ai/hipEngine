"""GPU correctness tests for the R3.6 C1 fused DFlash add+rmsnorm kernel.

Skipped when the ROCm/HIP runtime is unavailable.  Compares the fused kernel's
two outputs (``hidden_out`` and ``norm_out``) against the unfused chain
``dflash_add_bf16 -> dflash_rmsnorm_bf16`` on the actual drafter shapes.
``hidden_out`` must be **bit-identical**; ``norm_out`` must be bit-identical
because the fused kernel rounds the residual sum to BF16 before the RMS
reduction (matching the unfused HBM round-trip).
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


def _to_bf16(x_f32: np.ndarray) -> np.ndarray:
    bits = x_f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _from_bf16(x_u16: np.ndarray) -> np.ndarray:
    return (x_u16.astype(np.uint32) << 16).view(np.float32)


@pytest.fixture(scope="module")
def _drafter_lib():
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.speculative import build_dflash_drafter

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        return build_dflash_drafter(load=True, compiler_version=compiler_version)


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime
    return get_hip_runtime()


def _upload(runtime, bufs, array):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc
    arr = np.ascontiguousarray(array)
    buf = malloc(arr.nbytes, runtime=runtime)
    bufs.append(buf)
    copy_host_to_device(buf, host_array_ptr(arr), runtime=runtime)
    return buf


def _download(runtime, buf, shape, dtype):
    from hipengine.core.memory import copy_device_to_host, host_array_ptr
    arr = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(arr), buf, runtime=runtime)
    return arr


def _free(runtime, bufs):
    from hipengine.core.memory import free
    for buf in reversed(bufs):
        free(buf, runtime=runtime)


_SHAPES = (
    (16, 2048),  # actual drafter
    (16, 1024),
    (8, 2048),
    (32, 4096),
    (1, 256),
)


@pytest.mark.parametrize("rows,hidden_size", _SHAPES)
def test_dflash_add_rmsnorm_bf16_matches_unfused(_drafter_lib, _runtime, rows, hidden_size):
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
        dflash_add_bf16,
        dflash_add_rmsnorm_bf16,
        dflash_rmsnorm_bf16,
    )

    rng = np.random.default_rng(0xADD7)
    x_f32 = (rng.standard_normal((rows, hidden_size), dtype=np.float32) * 0.05)
    r_f32 = (rng.standard_normal((rows, hidden_size), dtype=np.float32) * 0.07)
    w_f32 = (rng.standard_normal((hidden_size,), dtype=np.float32) * 1.0)
    x_bf = _to_bf16(x_f32)
    r_bf = _to_bf16(r_f32)
    w_bf = _to_bf16(w_f32)
    bufs = []
    try:
        x_dev = _upload(_runtime, bufs, x_bf)
        r_dev = _upload(_runtime, bufs, r_bf)
        w_dev = _upload(_runtime, bufs, w_bf)
        h_unfused = _upload(_runtime, bufs, np.zeros((rows, hidden_size), np.uint16))
        n_unfused = _upload(_runtime, bufs, np.zeros((rows, hidden_size), np.uint16))
        h_fused = _upload(_runtime, bufs, np.zeros((rows, hidden_size), np.uint16))
        n_fused = _upload(_runtime, bufs, np.zeros((rows, hidden_size), np.uint16))
        dflash_add_bf16(x_dev.ptr, r_dev.ptr, h_unfused.ptr, rows * hidden_size, library=_drafter_lib, runtime=_runtime)
        dflash_rmsnorm_bf16(h_unfused.ptr, w_dev.ptr, n_unfused.ptr, rows, hidden_size, library=_drafter_lib, runtime=_runtime)
        dflash_add_rmsnorm_bf16(x_dev.ptr, r_dev.ptr, w_dev.ptr, h_fused.ptr, n_fused.ptr, rows, hidden_size, library=_drafter_lib, runtime=_runtime)
        _runtime.device_synchronize()
        h_u = _download(_runtime, h_unfused, (rows, hidden_size), np.uint16)
        n_u = _download(_runtime, n_unfused, (rows, hidden_size), np.uint16)
        h_f = _download(_runtime, h_fused, (rows, hidden_size), np.uint16)
        n_f = _download(_runtime, n_fused, (rows, hidden_size), np.uint16)
        # hidden_out is the BF16 residual sum: must be bit-identical because both
        # paths use ``float_to_bf16_bits(a + b)`` once per element.
        assert np.array_equal(h_u, h_f), "fused hidden_out must match unfused dflash_add_bf16 bitwise"
        # norm_out is bit-identical too because the fused kernel rounds the
        # residual sum to BF16 before the RMS reduction reads it.
        assert np.array_equal(n_u, n_f), "fused norm_out must match unfused dflash_rmsnorm_bf16 bitwise"
    finally:
        _free(_runtime, bufs)


def test_dflash_add_rmsnorm_bf16_rejects_oversize_hidden(_drafter_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import dflash_add_rmsnorm_bf16

    with pytest.raises(ValueError, match="hidden_size <= 4096"):
        dflash_add_rmsnorm_bf16(0, 0, 0, 0, 0, 16, 8192, library=_drafter_lib, runtime=_runtime)


def test_drafter_dense_use_add_rmsnorm_env(monkeypatch):
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import _drafter_dense_use_add_rmsnorm

    monkeypatch.delenv("HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM", raising=False)
    assert _drafter_dense_use_add_rmsnorm() is False
    monkeypatch.setenv("HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM", "off")
    assert _drafter_dense_use_add_rmsnorm() is False
    monkeypatch.setenv("HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM", "on")
    assert _drafter_dense_use_add_rmsnorm() is True
    monkeypatch.setenv("HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM", "bogus")
    with pytest.raises(ValueError):
        _drafter_dense_use_add_rmsnorm()
