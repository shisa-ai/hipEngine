"""GPU correctness tests for the R3.7 fused W8A16 LM-head + argmax-rows kernel.

Skipped when the ROCm/HIP runtime is unavailable.  Compares the fused output
against the unfused chain ``w8a16_linear_bf16_f32_multi_row -> argmax_f32_rows_i32``
on representative verifier shapes.  Fused indices must be **bit-identical** to
the unfused indices and fused values must match elementwise (FP32 reduction
order is preserved).
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


@pytest.fixture(scope="module")
def _lm_head_lib():
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.linear.lm_head import build_lm_head

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        return build_lm_head(load=True, compiler_version=compiler_version)


@pytest.fixture(scope="module")
def _w8a16_lib():
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import build_w8a16_linear

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        return build_w8a16_linear(load=True, compiler_version=compiler_version)


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


def _free(runtime, bufs):
    from hipengine.core.memory import free
    for buf in reversed(bufs):
        free(buf, runtime=runtime)


# Representative verifier shapes.  hidden_size=2048 + vocab_size=2048..16384 keep
# the test fast.  The actual W7900 verifier path uses
# (rows=B+1=5, hidden=2048, vocab=248320), validated separately via the e2e bench.
_SHAPES = (
    (5, 2048, 4096),
    (4, 2048, 8192),
    (1, 2048, 16384),
    (3, 1024, 2048),
)


@pytest.mark.parametrize("rows,hidden_size,vocab_size", _SHAPES)
def test_w8a16_lm_head_argmax_rows_matches_unfused(_lm_head_lib, _w8a16_lib, _runtime, rows, hidden_size, vocab_size):
    from hipengine.kernels.hip_gfx1100.linear.lm_head import (
        argmax_f32_rows_i32,
        lm_head_argmax_stage1_blocks,
        w8a16_lm_head_argmax_rows_bf16,
    )
    from hipengine.kernels.hip_gfx1100.quant.w8a16_linear import (
        w8a16_linear_bf16_f32_multi_row,
    )

    rng = np.random.default_rng(0xC1DA7A)
    hidden_f32 = (rng.standard_normal((rows, hidden_size), dtype=np.float32) * 0.05)
    weight_int8 = rng.integers(-127, 128, size=(vocab_size, hidden_size), dtype=np.int8)
    weight_scale = (rng.uniform(0.005, 0.02, size=(vocab_size,)).astype(np.float32))
    hidden_bf = _to_bf16(hidden_f32)

    bufs = []
    try:
        h_dev = _upload(_runtime, bufs, hidden_bf)
        w_dev = _upload(_runtime, bufs, weight_int8)
        s_dev = _upload(_runtime, bufs, weight_scale)
        # Unfused path: full-vocab logits + argmax_rows
        logits_dev = _alloc(_runtime, bufs, rows * vocab_size * 4)
        s1_blocks = lm_head_argmax_stage1_blocks(vocab_size)
        bv_unfused = _alloc(_runtime, bufs, rows * s1_blocks * 4)
        bi_unfused = _alloc(_runtime, bufs, rows * s1_blocks * 4)
        idx_unfused = _alloc(_runtime, bufs, rows * 4)
        val_unfused = _alloc(_runtime, bufs, rows * 4)
        w8a16_linear_bf16_f32_multi_row(
            h_dev.ptr, w_dev.ptr, s_dev.ptr, logits_dev.ptr,
            rows, hidden_size, vocab_size,
            library=_w8a16_lib, runtime=_runtime,
        )
        argmax_f32_rows_i32(
            logits_dev.ptr, bv_unfused.ptr, bi_unfused.ptr,
            idx_unfused.ptr, val_unfused.ptr,
            rows, vocab_size,
            library=_lm_head_lib, runtime=_runtime,
        )

        # Fused path
        bv_fused = _alloc(_runtime, bufs, rows * s1_blocks * 4)
        bi_fused = _alloc(_runtime, bufs, rows * s1_blocks * 4)
        idx_fused = _alloc(_runtime, bufs, rows * 4)
        val_fused = _alloc(_runtime, bufs, rows * 4)
        w8a16_lm_head_argmax_rows_bf16(
            h_dev.ptr, w_dev.ptr, s_dev.ptr,
            bv_fused.ptr, bi_fused.ptr,
            idx_fused.ptr, val_fused.ptr,
            rows, hidden_size, vocab_size,
            library=_lm_head_lib, runtime=_runtime,
        )
        _runtime.device_synchronize()

        idx_u = _download(_runtime, idx_unfused, (rows,), np.int32)
        val_u = _download(_runtime, val_unfused, (rows,), np.float32)
        idx_f = _download(_runtime, idx_fused, (rows,), np.int32)
        val_f = _download(_runtime, val_fused, (rows,), np.float32)

        # Indices must match bit-exact (same tiebreak rule, same per-vocab-row dot order).
        assert np.array_equal(idx_u, idx_f), (
            f"fused indices must match unfused: rows={rows} hidden={hidden_size} "
            f"vocab={vocab_size} unfused={idx_u} fused={idx_f}"
        )
        # Values are also bit-exact: the fused stage 1 reuses the same 256-way
        # cooperative dot-product reduction as the unfused multi_row kernel
        # (each thread strides hidden_size with 8x unroll, then LDS reduce),
        # and the scale is applied at the end identically.
        assert np.array_equal(val_u, val_f), (
            f"fused top-1 values must match unfused: rows={rows} hidden={hidden_size} "
            f"vocab={vocab_size} unfused={val_u} fused={val_f}"
        )
    finally:
        _free(_runtime, bufs)


def test_w8a16_lm_head_argmax_rows_rejects_invalid(_lm_head_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.linear.lm_head import w8a16_lm_head_argmax_rows_bf16

    with pytest.raises(ValueError):
        w8a16_lm_head_argmax_rows_bf16(0, 0, 0, 0, 0, 0, 0, 0, 2048, 16384, library=_lm_head_lib, runtime=_runtime)
    with pytest.raises(ValueError):
        w8a16_lm_head_argmax_rows_bf16(0, 0, 0, 0, 0, 0, 0, 4, 2048, 16384, threads=64, library=_lm_head_lib, runtime=_runtime)


def test_dflash_verify_fused_lm_head_env(monkeypatch):
    from hipengine.runtime.qwen35_paro_runner import _dflash_verify_fused_lm_head_enabled

    monkeypatch.delenv("HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD", raising=False)
    assert _dflash_verify_fused_lm_head_enabled() is False
    monkeypatch.setenv("HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD", "off")
    assert _dflash_verify_fused_lm_head_enabled() is False
    monkeypatch.setenv("HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD", "0")
    assert _dflash_verify_fused_lm_head_enabled() is False
    monkeypatch.setenv("HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD", "on")
    assert _dflash_verify_fused_lm_head_enabled() is True
    monkeypatch.setenv("HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD", "1")
    assert _dflash_verify_fused_lm_head_enabled() is True
    monkeypatch.setenv("HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD", "bogus")
    with pytest.raises(ValueError):
        _dflash_verify_fused_lm_head_enabled()
