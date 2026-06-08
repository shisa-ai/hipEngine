"""Unit gate for the persistent c>1 decode segment-metadata cache (C3.0b-1).

``Qwen35ParoResidentSession._batch_decode_segment_metadata`` caches the
``cu_seqlens`` (``arange(rows+1)``) and ``state_indices`` (physical slot ids)
device buffers keyed on ``(rows, slots)`` so the batch layer pass performs no
per-step ``malloc``/``free`` or host->device copy when the active batch is
unchanged.  This is the last per-step device allocation removed from the batch
layer pass and is a capture-safety prerequisite for c>1 decode graph replay.

These tests exercise the caching logic against a duck-typed stand-in (real HIP
runtime + device buffers, no 35B model) and assert:

* the device-resident ``cu``/``state`` values are correct,
* an unchanged ``(rows, slots)`` key reuses the same device tensors with no new
  allocation,
* a changed ``slots`` key refreshes ``state_indices`` while reusing the
  pre-sized device buffers (no realloc), and
* the third return element is an empty buffer tuple (no per-step release).
"""

from __future__ import annotations

import ctypes
import types

import numpy as np
import pytest


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")


def _make_stub(max_batch_size: int = 8):
    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime

    return types.SimpleNamespace(
        runtime=get_hip_runtime(),
        device=Device("hip", 0),
        max_batch_size=int(max_batch_size),
        buffers=[],
    )


def _read_i32(tensor) -> np.ndarray:
    from hipengine.core.dtype import DType
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    rows = int(np.prod(tensor.shape))
    out = np.empty((rows,), dtype=np.int32)
    copy_device_to_host(
        host_array_ptr(out),
        DeviceBuffer(tensor.ptr, rows * DType.INT32.itemsize),
        runtime=None,
    )
    return out


def _read_i64(tensor) -> np.ndarray:
    from hipengine.core.dtype import DType
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    rows = int(np.prod(tensor.shape))
    out = np.empty((rows,), dtype=np.int64)
    copy_device_to_host(
        host_array_ptr(out),
        DeviceBuffer(tensor.ptr, rows * DType.INT64.itemsize),
        runtime=None,
    )
    return out


def test_segment_metadata_values_and_cache_reuse() -> None:
    from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession

    meta = Qwen35ParoResidentSession._batch_decode_segment_metadata
    stub = _make_stub(max_batch_size=8)

    cu0, state0, temp0 = meta(stub, rows=2, slots=(0, 1))
    assert temp0 == ()  # no per-step buffers to release
    assert cu0.shape == (3,)
    assert state0.shape == (2,)
    np.testing.assert_array_equal(_read_i32(cu0), np.asarray([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(_read_i64(state0), np.asarray([0, 1], dtype=np.int64))

    buffers_after_first = len(stub.buffers)
    cu_buf_ptr = stub._decode_segment_cu_buf.ptr
    state_buf_ptr = stub._decode_segment_state_buf.ptr

    # Same (rows, slots) key: cached tensors reused, no new allocation.
    cu1, state1, temp1 = meta(stub, rows=2, slots=(0, 1))
    assert temp1 == ()
    assert cu1.ptr == cu0.ptr
    assert state1.ptr == state0.ptr
    assert len(stub.buffers) == buffers_after_first
    assert stub._decode_segment_cu_buf.ptr == cu_buf_ptr
    assert stub._decode_segment_state_buf.ptr == state_buf_ptr


def test_segment_metadata_refreshes_on_slot_change_without_realloc() -> None:
    from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession

    meta = Qwen35ParoResidentSession._batch_decode_segment_metadata
    stub = _make_stub(max_batch_size=8)

    meta(stub, rows=2, slots=(0, 1))
    cu_buf_ptr = stub._decode_segment_cu_buf.ptr
    state_buf_ptr = stub._decode_segment_state_buf.ptr
    buffers_after_first = len(stub.buffers)

    # Changed slots: state_indices must refresh; pre-sized buffers are reused.
    cu2, state2, _ = meta(stub, rows=2, slots=(0, 3))
    np.testing.assert_array_equal(_read_i32(cu2), np.asarray([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(_read_i64(state2), np.asarray([0, 3], dtype=np.int64))
    assert stub._decode_segment_cu_buf.ptr == cu_buf_ptr
    assert stub._decode_segment_state_buf.ptr == state_buf_ptr
    assert len(stub.buffers) == buffers_after_first

    # Wider batch within capacity: cu grows to arange(rows+1), still no realloc.
    cu4, state4, _ = meta(stub, rows=4, slots=(0, 1, 2, 3))
    np.testing.assert_array_equal(_read_i32(cu4), np.asarray([0, 1, 2, 3, 4], dtype=np.int32))
    np.testing.assert_array_equal(_read_i64(state4), np.asarray([0, 1, 2, 3], dtype=np.int64))
    assert stub._decode_segment_cu_buf.ptr == cu_buf_ptr
    assert stub._decode_segment_state_buf.ptr == state_buf_ptr
    assert len(stub.buffers) == buffers_after_first
