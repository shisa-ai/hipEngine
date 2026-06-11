from __future__ import annotations

import ctypes
import os
import pathlib

import numpy as np
import pytest


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime

    return get_hip_runtime()


@pytest.fixture(scope="module")
def _rotary_lib():
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import build_qwen35_rotary

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        return build_qwen35_rotary(load=True, compiler_version=compiler_version)


@pytest.fixture(scope="module")
def _cast_lib():
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.convert import build_cast

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        return build_cast(load=True, compiler_version=compiler_version)


def _upload(runtime, bufs, array):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    arr = np.ascontiguousarray(array)
    buf = malloc(max(arr.nbytes, 4), runtime=runtime)
    bufs.append(buf)
    copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes, runtime=runtime)
    return buf


def _alloc(runtime, bufs, nbytes):
    from hipengine.core.memory import malloc

    buf = malloc(max(nbytes, 4), runtime=runtime)
    bufs.append(buf)
    return buf


def _download(runtime, buf, shape, dtype):
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    arr = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(arr), buf, arr.nbytes, runtime=runtime)
    return arr


def _free(runtime, bufs) -> None:
    from hipengine.core.memory import free

    for buf in reversed(bufs):
        free(buf, runtime=runtime)


@pytest.mark.parametrize(
    "tokens,num_q_heads,num_kv_heads,head_dim",
    (
        (2, 4, 2, 16),
        (4, 16, 2, 32),
    ),
)
def test_qwen35_split_qgate_fp16_key_f32_matches_split_plus_cast(
    _runtime,
    _rotary_lib,
    _cast_lib,
    tokens,
    num_q_heads,
    num_kv_heads,
    head_dim,
) -> None:
    from hipengine.kernels.hip_gfx1100.convert import fp16_to_f32
    from hipengine.kernels.hip_gfx1100.rotary import (
        qwen35_split_qgate_fp16,
        qwen35_split_qgate_fp16_key_f32,
    )

    rng = np.random.default_rng(0x35504C17)
    q_proj = rng.standard_normal((tokens, num_q_heads, 2 * head_dim)).astype(np.float16)
    key_in = rng.standard_normal((tokens, num_kv_heads, head_dim)).astype(np.float16)
    query_shape = (tokens, num_q_heads, head_dim)
    key_shape = (tokens, num_kv_heads, head_dim)

    bufs = []
    try:
        q_proj_dev = _upload(_runtime, bufs, q_proj)
        key_in_dev = _upload(_runtime, bufs, key_in)

        query_old = _alloc(_runtime, bufs, int(np.prod(query_shape)) * 4)
        key_old = _alloc(_runtime, bufs, int(np.prod(key_shape)) * 4)
        gate_old = _alloc(_runtime, bufs, int(np.prod(query_shape)) * 2)
        query_new = _alloc(_runtime, bufs, int(np.prod(query_shape)) * 4)
        key_new = _alloc(_runtime, bufs, int(np.prod(key_shape)) * 4)
        gate_new = _alloc(_runtime, bufs, int(np.prod(query_shape)) * 2)

        qwen35_split_qgate_fp16(
            q_proj_dev.ptr,
            query_old.ptr,
            gate_old.ptr,
            tokens,
            num_q_heads,
            head_dim,
            library=_rotary_lib,
            runtime=_runtime,
        )
        fp16_to_f32(
            key_in_dev.ptr,
            key_old.ptr,
            int(np.prod(key_shape)),
            library=_cast_lib,
            runtime=_runtime,
        )
        qwen35_split_qgate_fp16_key_f32(
            q_proj_dev.ptr,
            key_in_dev.ptr,
            query_new.ptr,
            key_new.ptr,
            gate_new.ptr,
            tokens,
            num_q_heads,
            num_kv_heads,
            head_dim,
            library=_rotary_lib,
            runtime=_runtime,
        )
        _runtime.device_synchronize()

        np.testing.assert_array_equal(
            _download(_runtime, query_new, query_shape, np.float32),
            _download(_runtime, query_old, query_shape, np.float32),
        )
        np.testing.assert_array_equal(
            _download(_runtime, key_new, key_shape, np.float32),
            _download(_runtime, key_old, key_shape, np.float32),
        )
        np.testing.assert_array_equal(
            _download(_runtime, gate_new, query_shape, np.float16),
            _download(_runtime, gate_old, query_shape, np.float16),
        )
    finally:
        _free(_runtime, bufs)
