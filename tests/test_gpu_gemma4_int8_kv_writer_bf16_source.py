"""BF16-source per-token/head INT8 paged KV writer (Gemma D12 prerequisite).

The FP32-source INT8 writer is the numerical contract. This module exercises the
BF16-source sibling that Gemma needs: BF16 K/V rows are decoded to FP32 and then
run through the identical per-token/head INT8 codec. Expected payloads and
scales come from the CPU oracle applied to the same decoded BF16 values.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.cpu_reference import quantize_kv_int8_per_token_head
from hipengine.kernels.hip_gfx1100.attention import paged_kv_write
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata
from hipengine.loading.materialize import float_array_to_bf16_bits

_SENTINEL_I8 = np.int8(-128)
_SENTINEL_SCALE = 4096.0
_GEOMETRIES = ((8, 256), (2, 512))
_SCALE_DTYPES = ("fp32", "fp16")

_DTYPE_ENUM = {"fp32": DType.FP32, "fp16": DType.FP16}
_NP_DTYPE = {"fp32": np.float32, "fp16": np.float16}


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits_to_float(bits: np.ndarray) -> np.ndarray:
    """Mirror the kernel's ``bf16_bits_to_float`` (shift-left-16 bit pattern)."""

    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _tensor(ptr: int, shape: tuple[int, ...], dtype: DType) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _upload(buffers: list, array: np.ndarray, *, runtime) -> object:
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes, runtime=runtime)
    buffers.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(contiguous), contiguous.nbytes, runtime=runtime)
    return buffer


def _download(buffer, array: np.ndarray, *, runtime) -> np.ndarray:
    copy_device_to_host(host_array_ptr(array), buffer, array.nbytes, runtime=runtime)
    return array


def _decode_rows(num_kv_heads: int, head_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build distinct K/V rows with zero, tie, and extremum coverage.

    Head 0 is all zero (zero row -> zero scale, zero payload). Head 1 is the
    discriminator: K holds extrema at ``+/-127`` with a round-half-to-even tie
    at ``+/-63.5`` (scale ``1.0``), while V holds ``+/-254`` with a tie at
    ``+/-125`` (scale ``2.0``), so a K/V swap changes both scale and payload.
    The remaining heads are distinct wide-range random K and V.
    """

    rng = np.random.default_rng(seed)
    key = rng.uniform(-100.0, 100.0, size=(1, num_kv_heads, head_dim)).astype(np.float32)
    value = rng.uniform(-100.0, 100.0, size=(1, num_kv_heads, head_dim)).astype(np.float32)
    key[0, 0] = 0.0
    value[0, 0] = 0.0
    key[0, 1, 0] = 127.0
    key[0, 1, 1] = -127.0
    key[0, 1, 2] = 63.5
    key[0, 1, 3] = -63.5
    value[0, 1, 0] = 254.0
    value[0, 1, 1] = -254.0
    value[0, 1, 2] = 125.0
    value[0, 1, 3] = -125.0
    return key, value


def _expected_caches(
    decoded_key: np.ndarray,
    decoded_value: np.ndarray,
    positions: np.ndarray,
    block_table: np.ndarray,
    *,
    block_size: int,
    scale_dtype: str,
    cache_blocks: int,
    row_major: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scatter oracle payloads into sentinel-filled caches.

    ``row_major`` mirrors the kernel's ``row_major_cache`` flag: per-row private
    arenas (``row*table_len + physical``) versus one shared physical arena.
    """

    rows, num_kv_heads, head_dim = decoded_key.shape
    table = np.asarray(block_table, dtype=np.int64).reshape(rows, -1)
    table_len = table.shape[1]
    qk, qv, ks, vs = quantize_kv_int8_per_token_head(
        decoded_key, decoded_value, scale_dtype=_NP_DTYPE[scale_dtype]
    )
    key_cache = np.full((cache_blocks, block_size, num_kv_heads, head_dim), _SENTINEL_I8, np.int8)
    value_cache = np.full_like(key_cache, _SENTINEL_I8)
    k_scale = np.full(
        (cache_blocks, block_size, num_kv_heads),
        np.asarray(_SENTINEL_SCALE, dtype=_NP_DTYPE[scale_dtype]),
        _NP_DTYPE[scale_dtype],
    )
    v_scale = np.full_like(k_scale, np.asarray(_SENTINEL_SCALE, dtype=_NP_DTYPE[scale_dtype]))
    for row in range(rows):
        position = int(positions[row])
        logical_block, block_offset = divmod(position, block_size)
        physical_block = int(table[row, logical_block])
        cache_block = row * table_len + physical_block if row_major else physical_block
        key_cache[cache_block, block_offset] = qk[row]
        value_cache[cache_block, block_offset] = qv[row]
        k_scale[cache_block, block_offset] = ks[row].astype(_NP_DTYPE[scale_dtype])
        v_scale[cache_block, block_offset] = vs[row].astype(_NP_DTYPE[scale_dtype])
    assert not np.array_equal(key_cache, value_cache), "fixture K/V payloads must differ"
    assert not np.array_equal(k_scale, v_scale), "fixture K/V scales must differ"
    return key_cache, value_cache, k_scale, v_scale


def _run_case(
    *,
    key: np.ndarray,
    value: np.ndarray,
    positions: np.ndarray,
    block_table: np.ndarray,
    block_size: int,
    cache_blocks: int,
    scale_dtype: str,
    row_major: bool,
    launcher,
    rows_arg: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Upload BF16 inputs, launch one writer route, return downloaded caches."""

    runtime = get_hip_runtime()
    rows, num_kv_heads, head_dim = key.shape
    key_bits = float_array_to_bf16_bits(key)
    value_bits = float_array_to_bf16_bits(value)
    decoded_key = _bf16_bits_to_float(key_bits)
    decoded_value = _bf16_bits_to_float(value_bits)

    expected = _expected_caches(
        decoded_key,
        decoded_value,
        positions,
        block_table,
        block_size=block_size,
        scale_dtype=scale_dtype,
        cache_blocks=cache_blocks,
        row_major=row_major,
    )

    np_scale = _NP_DTYPE[scale_dtype]
    key_cache_host = np.full_like(expected[0], _SENTINEL_I8)
    value_cache_host = np.full_like(expected[1], _SENTINEL_I8)
    k_scale_host = np.full(expected[2].shape, np.asarray(_SENTINEL_SCALE, np_scale), np_scale)
    v_scale_host = np.full(expected[3].shape, np.asarray(_SENTINEL_SCALE, np_scale), np_scale)

    buffers: list = []
    try:
        key_d = _upload(buffers, key_bits, runtime=runtime)
        value_d = _upload(buffers, value_bits, runtime=runtime)
        table_d = _upload(buffers, block_table.astype(np.int32), runtime=runtime)
        live_d = _upload(buffers, positions.astype(np.int64), runtime=runtime)
        key_cache_d = _upload(buffers, key_cache_host, runtime=runtime)
        value_cache_d = _upload(buffers, value_cache_host, runtime=runtime)
        k_scale_d = _upload(buffers, k_scale_host, runtime=runtime)
        v_scale_d = _upload(buffers, v_scale_host, runtime=runtime)

        metadata = KVScaleMetadata(
            k_scale=_tensor(k_scale_d.ptr, k_scale_host.shape, _DTYPE_ENUM[scale_dtype]),
            v_scale=_tensor(v_scale_d.ptr, v_scale_host.shape, _DTYPE_ENUM[scale_dtype]),
            scale_dtype=_DTYPE_ENUM[scale_dtype],
        )
        spans = KVLiveSpans.paged_uniform(
            block_table=_tensor(table_d.ptr, block_table.shape, DType.INT32),
            live_counts=_tensor(live_d.ptr, positions.shape, DType.INT64),
            max_live_count=int(positions.max()),
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            scale_metadata=metadata,
        )
        args = [
            key_d.ptr,
            value_d.ptr,
            key_cache_d.ptr,
            value_cache_d.ptr,
            k_scale_d.ptr,
            v_scale_d.ptr,
            spans,
        ]
        if rows_arg:
            args.append(rows)
        args.extend([block_size, num_kv_heads, head_dim])
        launcher(*args, runtime=runtime)

        out_key = _download(key_cache_d, key_cache_host, runtime=runtime).copy()
        out_value = _download(value_cache_d, value_cache_host, runtime=runtime).copy()
        out_k_scale = _download(k_scale_d, k_scale_host, runtime=runtime).copy()
        out_v_scale = _download(v_scale_d, v_scale_host, runtime=runtime).copy()
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(out_key, expected[0], err_msg="key payload mismatch")
    np.testing.assert_array_equal(out_value, expected[1], err_msg="value payload mismatch")
    np.testing.assert_array_equal(out_k_scale, expected[2], err_msg="k scale mismatch")
    np.testing.assert_array_equal(out_v_scale, expected[3], err_msg="v scale mismatch")
    return out_key, out_value, out_k_scale, out_v_scale


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
@pytest.mark.parametrize(
    ("num_kv_heads", "head_dim"), _GEOMETRIES, ids=("hkv8-d256", "hkv2-d512")
)
def test_bf16_source_int8_decode_matches_cpu_oracle(
    num_kv_heads: int, head_dim: int, scale_dtype: str
) -> None:
    """Decode route: one BF16 row, shuffled pages, page crossing, sentinel slots."""

    key, value = _decode_rows(num_kv_heads, head_dim, seed=0xD12 + num_kv_heads + head_dim)
    block_size = 4
    positions = np.asarray([6], dtype=np.int64)  # logical block 1, offset 2
    block_table = np.asarray([2, 0], dtype=np.int32)  # shuffled; physical 1/3 untouched
    _run_case(
        key=key,
        value=value,
        positions=positions,
        block_table=block_table,
        block_size=block_size,
        cache_blocks=4,
        scale_dtype=scale_dtype,
        row_major=False,
        launcher=paged_kv_write.qwen35_write_paged_kv_int8_per_token_head_bf16_spans,
        rows_arg=False,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_bf16_source_int8_batch_matches_cpu_oracle(scale_dtype: str) -> None:
    """Batch route: per-row private arenas with one zero row and a page crossing."""

    num_kv_heads, head_dim = 2, 512
    key = np.zeros((2, num_kv_heads, head_dim), dtype=np.float32)
    value = np.zeros((2, num_kv_heads, head_dim), dtype=np.float32)
    row0_key, row0_value = _decode_rows(num_kv_heads, head_dim, seed=0xB47C4)
    key[0] = row0_key[0]
    value[0] = row0_value[0]
    # Row 1 stays all zero -> zero scale, zero payload.
    block_size = 4
    positions = np.asarray([5, 1], dtype=np.int64)
    block_table = np.asarray([[1, 0], [0, 1]], dtype=np.int32)
    _run_case(
        key=key,
        value=value,
        positions=positions,
        block_table=block_table,
        block_size=block_size,
        cache_blocks=4,
        scale_dtype=scale_dtype,
        row_major=True,
        launcher=paged_kv_write.qwen35_write_paged_kv_int8_per_token_head_bf16_batch_spans,
        rows_arg=True,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
def test_bf16_source_int8_prompt_matches_cpu_oracle(scale_dtype: str) -> None:
    """Prompt route: shared physical arena, distinct per-row page tables."""

    num_kv_heads, head_dim = 8, 256
    key = np.zeros((2, num_kv_heads, head_dim), dtype=np.float32)
    value = np.zeros((2, num_kv_heads, head_dim), dtype=np.float32)
    row0_key, row0_value = _decode_rows(num_kv_heads, head_dim, seed=0x9E0E)
    row1_key, row1_value = _decode_rows(num_kv_heads, head_dim, seed=0x9E0F)
    key[0] = row0_key[0]
    value[0] = row0_value[0]
    key[1] = row1_key[0]
    value[1] = row1_value[0]
    block_size = 4
    positions = np.asarray([5, 2], dtype=np.int64)
    # Shared arena; distinct physical blocks per row so no aliasing occurs.
    block_table = np.asarray([[3, 1], [0, 2]], dtype=np.int32)
    _run_case(
        key=key,
        value=value,
        positions=positions,
        block_table=block_table,
        block_size=block_size,
        cache_blocks=4,
        scale_dtype=scale_dtype,
        row_major=False,
        launcher=paged_kv_write.qwen35_write_paged_kv_int8_per_token_head_bf16_prompt_spans,
        rows_arg=True,
    )


def test_bf16_source_int8_writer_refuses_invalid_metadata() -> None:
    """Validation happens before any GPU load, so this runs without HIP."""

    block_size, num_kv_heads, head_dim = 4, 2, 8

    def spans(*, scale_shape=(2, 4, 2), scale_dtype="fp32", live_dtype="int64", k_offset=0):
        metadata = KVScaleMetadata(
            k_scale=_tensor(0x3000 + k_offset, scale_shape, scale_dtype),
            v_scale=_tensor(0x4000, scale_shape, scale_dtype),
            scale_dtype=scale_dtype,
        )
        return KVLiveSpans.paged_uniform(
            block_table=_tensor(0x1000, (2,), "int32"),
            live_counts=_tensor(0x2000, (1,), live_dtype),
            max_live_count=1,
            storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            scale_metadata=metadata,
        )

    writer = paged_kv_write.qwen35_write_paged_kv_int8_per_token_head_bf16_spans
    good = spans()
    with pytest.raises(ValueError, match="int64 live_counts"):
        bad = spans(live_dtype="int32")
        writer(
            0, 0, 0, 0,
            bad.scale_metadata.k_scale.ptr,
            bad.scale_metadata.v_scale.ptr,
            bad,
            block_size,
            num_kv_heads,
            head_dim,
        )
    with pytest.raises(ValueError, match="k_scale_ptr"):
        writer(
            0, 0, 0, 0,
            good.scale_metadata.k_scale.ptr + 1,
            good.scale_metadata.v_scale.ptr,
            good,
            block_size,
            num_kv_heads,
            head_dim,
        )
    with pytest.raises(ValueError, match="scale tensor shape"):
        bad_shape = spans(scale_shape=(2, 3, 2))
        writer(
            0, 0, 0, 0,
            bad_shape.scale_metadata.k_scale.ptr,
            bad_shape.scale_metadata.v_scale.ptr,
            bad_shape,
            block_size,
            num_kv_heads,
            head_dim,
        )
