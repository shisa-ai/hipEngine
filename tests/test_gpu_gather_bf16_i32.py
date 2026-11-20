"""Unit tests for the device-side BF16 row-gather and I32 pick kernels.

These feed the device-resident EAGLE-4 efficient head: the top-K candidate ids
never leave the device — a BF16 gather selects the shortlist ``lm_head`` rows by
device id, the row-wise argmax picks a shortlist index, and an I32 pick maps that
index back to the token id. Model-free; GPU cases skip cleanly on no-ROCm runners.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.convert.gather import (
    build_gather,
    gather_bf16_rows_by_i32id,
    gather_i32_by_i32id,
    register_gather_kernels,
)
from hipengine.kernels.registry import resolve


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_device_gathers_register() -> None:
    register_gather_kernels()
    assert resolve(
        backend="hip_gfx1100", layer="gather_bf16_rows_by_i32id", quant="bf16"
    ) is gather_bf16_rows_by_i32id
    assert resolve(
        backend="hip_gfx1100", layer="gather_i32_by_i32id", quant="i32"
    ) is gather_i32_by_i32id


def test_device_gather_wrappers_validate_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="rows"):
        gather_bf16_rows_by_i32id(0, 0, 0, 0, 8, 4)
    with pytest.raises(ValueError, match="hidden"):
        gather_bf16_rows_by_i32id(0, 0, 0, 1, 0, 4)
    with pytest.raises(ValueError, match="vocab"):
        gather_bf16_rows_by_i32id(0, 0, 0, 1, 8, 0)
    with pytest.raises(ValueError, match="rows"):
        gather_i32_by_i32id(0, 0, 0, 0, 4)
    with pytest.raises(ValueError, match="vocab"):
        gather_i32_by_i32id(0, 0, 0, 1, 0)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_bf16_gather_selects_rows_by_device_id_bit_exact() -> None:
    vocab, hidden = 7, 16
    # Arbitrary bf16 bit patterns (incl. negatives / denormals): the gather must
    # be a bit-exact copy, not a float conversion.
    rng = np.random.default_rng(7)
    table = rng.integers(0, 2**16, size=(vocab, hidden), dtype=np.uint16)
    ids = np.asarray([5, 0, 6, 2], dtype=np.int32)
    rows = int(ids.shape[0])

    library = build_gather(load=True)
    table_buf = malloc(table.nbytes)
    ids_buf = malloc(ids.nbytes)
    out_buf = malloc(rows * hidden * 2)
    try:
        copy_host_to_device(table_buf, host_array_ptr(np.ascontiguousarray(table)), table.nbytes)
        copy_host_to_device(ids_buf, host_array_ptr(np.ascontiguousarray(ids)), ids.nbytes)
        gather_bf16_rows_by_i32id(
            table_buf.ptr, ids_buf.ptr, out_buf.ptr, rows, hidden, vocab, library=library
        )
        out = np.empty((rows, hidden), dtype=np.uint16)
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        free(out_buf)
        free(ids_buf)
        free(table_buf)

    np.testing.assert_array_equal(out, table[ids])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_bf16_gather_out_of_range_id_zeros_row() -> None:
    vocab, hidden = 3, 8
    table = np.full((vocab, hidden), 0x3F80, dtype=np.uint16)  # bf16 1.0
    ids = np.asarray([9, -2], dtype=np.int32)

    library = build_gather(load=True)
    table_buf = malloc(table.nbytes)
    ids_buf = malloc(ids.nbytes)
    out_buf = malloc(2 * hidden * 2)
    try:
        copy_host_to_device(table_buf, host_array_ptr(np.ascontiguousarray(table)), table.nbytes)
        copy_host_to_device(ids_buf, host_array_ptr(np.ascontiguousarray(ids)), ids.nbytes)
        gather_bf16_rows_by_i32id(
            table_buf.ptr, ids_buf.ptr, out_buf.ptr, 2, hidden, vocab, library=library
        )
        out = np.empty((2, hidden), dtype=np.uint16)
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        free(out_buf)
        free(ids_buf)
        free(table_buf)

    np.testing.assert_array_equal(out, np.zeros((2, hidden), dtype=np.uint16))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_i32_pick_maps_shortlist_index_to_token_by_device_id() -> None:
    # Mirrors the efficient head: cand_ids holds the top-K token ids; the argmax
    # writes a shortlist index; the pick reads cand_ids[index] without host I/O.
    cand = np.asarray([101, 42, 7, 999, 3], dtype=np.int32)
    picks = np.asarray([3, 0, 4], dtype=np.int32)

    library = build_gather(load=True)
    cand_buf = malloc(cand.nbytes)
    picks_buf = malloc(picks.nbytes)
    out_buf = malloc(picks.nbytes)
    try:
        copy_host_to_device(cand_buf, host_array_ptr(np.ascontiguousarray(cand)), cand.nbytes)
        copy_host_to_device(picks_buf, host_array_ptr(np.ascontiguousarray(picks)), picks.nbytes)
        # One pick per call via byte-offset id pointer (the per-step draft pattern).
        for i, expected in enumerate((999, 101, 3)):
            gather_i32_by_i32id(
                cand_buf.ptr, picks_buf.ptr + i * 4, out_buf.ptr + i * 4, 1,
                int(cand.shape[0]), library=library,
            )
        out = np.empty((3,), dtype=np.int32)
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        free(out_buf)
        free(picks_buf)
        free(cand_buf)

    np.testing.assert_array_equal(out, np.asarray([999, 101, 3], dtype=np.int32))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_i32_pick_out_of_range_id_zero() -> None:
    table = np.asarray([11, 22, 33], dtype=np.int32)
    ids = np.asarray([5, -1], dtype=np.int32)

    library = build_gather(load=True)
    table_buf = malloc(table.nbytes)
    ids_buf = malloc(ids.nbytes)
    out_buf = malloc(ids.nbytes)
    try:
        copy_host_to_device(table_buf, host_array_ptr(np.ascontiguousarray(table)), table.nbytes)
        copy_host_to_device(ids_buf, host_array_ptr(np.ascontiguousarray(ids)), ids.nbytes)
        gather_i32_by_i32id(table_buf.ptr, ids_buf.ptr, out_buf.ptr, 2, 3, library=library)
        out = np.empty((2,), dtype=np.int32)
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        free(out_buf)
        free(ids_buf)
        free(table_buf)

    np.testing.assert_array_equal(out, np.zeros((2,), dtype=np.int32))
