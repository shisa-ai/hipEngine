"""Unit tests for the device-side FP32 row-gather kernel.

The gather feeds the resident MTP NextN draft chain: a device-resident int32
row id (top-k argmax of the prior depth) selects the next depth's embedding row
with no host round-trip.  These tests are model-free; the GPU cases skip cleanly
on no-ROCm runners.
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
    gather_f32_rows_by_i32id,
    plan_gather_build,
    register_gather_kernels,
)
from hipengine.kernels.registry import resolve


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_gather_registers_and_build_plan() -> None:
    register_gather_kernels()
    assert resolve(
        backend="hip_gfx1100", layer="gather_f32_rows_by_i32id", quant="fp32"
    ) is gather_f32_rows_by_i32id

    artifact = plan_gather_build(compiler_version="gather-test-version")
    assert artifact.family == "gather"
    assert artifact.output_path.name == "gather.so"
    assert any(path.name == "gather.hip" for path in artifact.sources)


def test_gather_wrapper_validates_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="rows"):
        gather_f32_rows_by_i32id(0, 0, 0, 0, 8, 4)
    with pytest.raises(ValueError, match="hidden"):
        gather_f32_rows_by_i32id(0, 0, 0, 1, 0, 4)
    with pytest.raises(ValueError, match="vocab"):
        gather_f32_rows_by_i32id(0, 0, 0, 1, 8, 0)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_gather_selects_rows_by_device_id() -> None:
    vocab, hidden = 6, 8
    table = (np.arange(vocab * hidden, dtype=np.float32).reshape(vocab, hidden) + 0.5)
    ids = np.asarray([4, 0, 2], dtype=np.int32)
    rows = int(ids.shape[0])

    library = build_gather(load=True)
    table_buf = malloc(table.nbytes)
    ids_buf = malloc(ids.nbytes)
    out_buf = malloc(rows * hidden * 4)
    try:
        copy_host_to_device(table_buf, host_array_ptr(np.ascontiguousarray(table)), table.nbytes)
        copy_host_to_device(ids_buf, host_array_ptr(np.ascontiguousarray(ids)), ids.nbytes)
        gather_f32_rows_by_i32id(
            table_buf.ptr, ids_buf.ptr, out_buf.ptr, rows, hidden, vocab, library=library
        )
        out = np.empty((rows, hidden), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        free(out_buf)
        free(ids_buf)
        free(table_buf)

    np.testing.assert_array_equal(out[0], table[4])
    np.testing.assert_array_equal(out[1], table[0])
    np.testing.assert_array_equal(out[2], table[2])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_gather_offset_id_pointer_reads_top1_of_topk_row() -> None:
    # Mirrors the draft chain: ids buffer holds [top_k] indices per depth and the
    # gather reads the depth's top-1 via a byte-offset id pointer.
    vocab, hidden, top_k = 5, 4, 8
    table = (np.arange(vocab * hidden, dtype=np.float32).reshape(vocab, hidden) - 1.25)
    topk_all = np.zeros((2, top_k), dtype=np.int32)
    topk_all[0, 0] = 3  # depth 0 top-1
    topk_all[1, 0] = 1  # depth 1 top-1

    library = build_gather(load=True)
    table_buf = malloc(table.nbytes)
    topk_buf = malloc(topk_all.nbytes)
    out_buf = malloc(hidden * 4)
    try:
        copy_host_to_device(table_buf, host_array_ptr(np.ascontiguousarray(table)), table.nbytes)
        copy_host_to_device(topk_buf, host_array_ptr(np.ascontiguousarray(topk_all)), topk_all.nbytes)
        for depth, expected in ((0, 3), (1, 1)):
            ids_ptr = topk_buf.ptr + depth * top_k * 4
            gather_f32_rows_by_i32id(
                table_buf.ptr, ids_ptr, out_buf.ptr, 1, hidden, vocab, library=library
            )
            out = np.empty((hidden,), dtype=np.float32)
            copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
            np.testing.assert_array_equal(out, table[expected])
    finally:
        free(out_buf)
        free(topk_buf)
        free(table_buf)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_gather_out_of_range_id_zeros_row() -> None:
    vocab, hidden = 3, 4
    table = np.ones((vocab, hidden), dtype=np.float32)
    ids = np.asarray([5, -1], dtype=np.int32)  # both out of [0, vocab)

    library = build_gather(load=True)
    table_buf = malloc(table.nbytes)
    ids_buf = malloc(ids.nbytes)
    out_buf = malloc(2 * hidden * 4)
    try:
        copy_host_to_device(table_buf, host_array_ptr(np.ascontiguousarray(table)), table.nbytes)
        copy_host_to_device(ids_buf, host_array_ptr(np.ascontiguousarray(ids)), ids.nbytes)
        gather_f32_rows_by_i32id(
            table_buf.ptr, ids_buf.ptr, out_buf.ptr, 2, hidden, vocab, library=library
        )
        out = np.empty((2, hidden), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
    finally:
        free(out_buf)
        free(ids_buf)
        free(table_buf)

    np.testing.assert_array_equal(out, np.zeros((2, hidden), dtype=np.float32))
