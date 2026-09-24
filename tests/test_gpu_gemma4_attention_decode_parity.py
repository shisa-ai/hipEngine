"""Decode-kernel parity with the block kernel on gfx1100.

The warp-32 decode kernel must be *bit-exact* with the original block kernel:
same products, same stride-tree order, same per-lane logit partitions, same
weighted-sum order. These tests launch both exported symbols on identical
inputs and compare outputs bitwise (uint16 for BF16, float equality for F32).
"""

import ctypes

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")

_SHAPES = [
    # (num_heads, num_kv_heads, head_dim, keys)
    (1, 1, 128, 1),  # single live key
    (1, 1, 128, 1040),  # metric-shape decode context
    (4, 2, 128, 1152),  # GQA
    (8, 4, 64, 300),  # head_dim 64 -> two chains
    (2, 1, 192, 64),  # non-power-of-two head_dim (M = 256, zero leaves)
    (2, 2, 512, 96),  # head_dim 512 -> kTile edge over the 256 thread cap
    (1, 1, 768, 64),  # head_dim above the 256-wide thread cap -> multi-term dots
    (1, 1, 6, 40),  # head_dim < 32 -> sub-wave workgroup (shfl-only tree)
]


def _bf16_bits(array: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(array, dtype=np.float32).view(np.uint32) >> 16).astype(
        np.uint16
    )


@pytest.fixture(scope="module")
def attention_library():
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        build_gemma4_attention,
    )

    return build_gemma4_attention(load=True)


def _raw_launch(library, symbol, buffers, *, tokens, keys, num_heads, num_kv_heads,
                head_dim, scale):
    from hipengine.core.ctypes_cache import signed_kernel_fn
    from hipengine.core.hip import HIP_SUCCESS
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import _ARGTYPES_PREFILL

    fn = signed_kernel_fn(library, symbol, _ARGTYPES_PREFILL, ctypes.c_int)
    err = fn(
        *(b.ptr for b in buffers[:5]),
        tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        ctypes.c_float(scale),
        0,
        keys,
    )
    assert int(err) == HIP_SUCCESS, f"{symbol} returned {err}"


def _run_both_symbols(library, *, dtype, num_heads, num_kv_heads, head_dim, keys,
                      mask_mode="keep", scale=1.0, seed=20260924):
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        _SYMBOL_DECODE_BF16,
        _SYMBOL_DECODE_F32,
        _SYMBOL_PREFILL_BF16,
        _SYMBOL_PREFILL_F32,
    )

    rng = np.random.default_rng(seed)
    storage = np.float32 if dtype == "f32" else np.uint16

    def cast(array):
        return np.ascontiguousarray(array, dtype=np.float32) if dtype == "f32" else _bf16_bits(array)

    query = cast(rng.standard_normal((1, num_heads, head_dim)) * 0.7)
    key = cast(rng.standard_normal((keys, num_kv_heads, head_dim)) * 0.7)
    value = cast(rng.standard_normal((keys, num_kv_heads, head_dim)) * 0.7)
    if mask_mode == "keep":
        mask = np.ones((1, keys), dtype=np.uint8)
    elif mask_mode == "holes":
        mask = (rng.random((1, keys)) < 0.7).astype(np.uint8)
        mask[0, 0] = 1  # at least one live key, as decode always keeps position 0's row
    elif mask_mode == "none":
        mask = np.zeros((1, keys), dtype=np.uint8)
    else:  # pragma: no cover - defensive
        raise ValueError(mask_mode)

    arrays = [query, key, value, mask]
    out_prefill = np.zeros_like(query, dtype=storage)
    out_decode = np.zeros_like(query, dtype=storage)
    arrays += [out_prefill, out_decode]

    buffers = []
    try:
        for array in arrays:
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            if array is not out_prefill and array is not out_decode:
                copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)

        shape = dict(
            tokens=1,
            keys=keys,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
        )
        prefill_symbol = _SYMBOL_PREFILL_F32 if dtype == "f32" else _SYMBOL_PREFILL_BF16
        decode_symbol = _SYMBOL_DECODE_F32 if dtype == "f32" else _SYMBOL_DECODE_BF16

        _raw_launch(library, prefill_symbol, buffers[:5], **shape)
        copy_device_to_host(host_array_ptr(out_prefill), buffers[4], out_prefill.nbytes)

        _raw_launch(library, decode_symbol,
                    [buffers[0], buffers[1], buffers[2], buffers[3], buffers[5]], **shape)
        copy_device_to_host(host_array_ptr(out_decode), buffers[5], out_decode.nbytes)

        return out_prefill, out_decode
    finally:
        for buffer in buffers:
            free(buffer)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@pytest.mark.parametrize("mask_mode", ["keep", "holes"])
@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim,keys", _SHAPES)
def test_decode_symbol_bit_matches_prefill_symbol(
    attention_library, dtype, mask_mode, num_heads, num_kv_heads, head_dim, keys
):
    reference, candidate = _run_both_symbols(
        attention_library,
        dtype=dtype,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        keys=keys,
        mask_mode=mask_mode,
    )
    if dtype == "bf16":
        np.testing.assert_array_equal(reference, candidate)
    else:
        np.testing.assert_array_equal(reference, candidate)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_all_masked_row_matches(attention_library, dtype):
    # No live key: row_max is -inf and both kernels propagate NaN the same way.
    reference, candidate = _run_both_symbols(
        attention_library, dtype=dtype, num_heads=2, num_kv_heads=2, head_dim=128,
        keys=64, mask_mode="none",
    )
    np.testing.assert_array_equal(reference, candidate)


@pytest.mark.parametrize("scale", [1.0, 0.375])
def test_scaled_decode_matches(attention_library, scale):
    reference, candidate = _run_both_symbols(
        attention_library, dtype="f32", num_heads=1, num_kv_heads=1, head_dim=128,
        keys=517, mask_mode="holes", scale=scale,
    )
    np.testing.assert_array_equal(reference, candidate)


def test_public_wrapper_tokens_one_routes_to_decode_result(attention_library):
    """The wrapper a caller reaches must produce the block kernel's exact output.

    The unit tier proves ``tokens == 1`` selects the decode symbol; here we
    confirm the launched result through that wrapper is still bit-identical to
    the original block kernel on the same inputs.
    """
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        _SYMBOL_PREFILL_F32,
        _ARGTYPES_PREFILL,
        gemma4_attention_prefill_f32,
    )
    from hipengine.core.ctypes_cache import signed_kernel_fn
    from hipengine.core.hip import HIP_SUCCESS

    rng = np.random.default_rng(7)
    keys, heads, head_dim = 777, 2, 128
    query = np.ascontiguousarray(rng.standard_normal((1, heads, head_dim)), dtype=np.float32)
    key = np.ascontiguousarray(rng.standard_normal((keys, heads, head_dim)), dtype=np.float32)
    value = np.ascontiguousarray(rng.standard_normal((keys, heads, head_dim)), dtype=np.float32)
    mask = np.ones((1, keys), dtype=np.uint8)
    out_reference = np.zeros((1, heads, head_dim), dtype=np.float32)
    out_wrapper = np.zeros_like(out_reference)

    buffers = []
    try:
        for array in (query, key, value, mask, out_reference, out_wrapper):
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes)

        shape = dict(tokens=1, keys=keys, num_heads=heads, num_kv_heads=heads,
                     head_dim=head_dim, scale=1.0)
        fn = signed_kernel_fn(attention_library, _SYMBOL_PREFILL_F32, _ARGTYPES_PREFILL,
                              ctypes.c_int)
        err = fn(*(b.ptr for b in buffers[:5]), 1, heads, heads, head_dim,
                 ctypes.c_float(1.0), 0, keys)
        assert int(err) == HIP_SUCCESS
        copy_device_to_host(host_array_ptr(out_reference), buffers[4], out_reference.nbytes)

        gemma4_attention_prefill_f32(buffers[0].ptr, buffers[1].ptr, buffers[2].ptr,
                                     buffers[3].ptr, buffers[5].ptr, **shape)
        copy_device_to_host(host_array_ptr(out_wrapper), buffers[5], out_wrapper.nbytes)

        np.testing.assert_array_equal(out_reference, out_wrapper)
    finally:
        for buffer in buffers:
            free(buffer)