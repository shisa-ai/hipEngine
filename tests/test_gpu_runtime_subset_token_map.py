"""Device contract for the request-local MTP draft-head token-index map.

``remap_prefix_tail_i64`` rewrites one int64 draft row index in place with a
single device thread. These cases pin the prefix/tail boundaries, the extra-only
(zero prefix) map, the prefix-only null table, invalid indices that must become
``-1`` without reading out of bounds, and row ids above the int32 range.
"""

from __future__ import annotations

import ctypes

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
def hip_runtime_and_library():
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.runtime.state import build_runtime_state

    return get_hip_runtime(), build_runtime_state(load=True)


def _reference(value: int, extra_ids: tuple[int, ...], prefix_size: int) -> int:
    """Host reference for one draft-head row index."""

    if 0 <= value < prefix_size:
        return value
    tail = value - prefix_size
    if value >= prefix_size and 0 <= tail < len(extra_ids):
        return extra_ids[tail]
    return -1


def _remap(
    runtime,
    library,
    index_value: int,
    extra_ids: tuple[int, ...] | None,
    prefix_size: int,
    extra_count: int,
) -> int:
    from hipengine.core.dtype import DType
    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.runtime.state import remap_prefix_tail_i64

    index_host = np.asarray([index_value], dtype=np.int64)
    index_device = malloc(DType.INT64.itemsize, runtime=runtime)
    extra_device = None
    try:
        copy_host_to_device(
            index_device,
            host_array_ptr(index_host),
            index_host.nbytes,
            runtime=runtime,
        )
        extra_ptr = None
        if extra_count > 0:
            assert extra_ids is not None
            assert len(extra_ids) == extra_count
            extra_host = np.asarray(extra_ids, dtype=np.int64)
            extra_device = malloc(extra_host.nbytes, runtime=runtime)
            copy_host_to_device(
                extra_device,
                host_array_ptr(extra_host),
                extra_host.nbytes,
                runtime=runtime,
            )
            extra_ptr = extra_device.ptr
        remap_prefix_tail_i64(
            index_device.ptr,
            extra_ptr,
            prefix_size,
            extra_count,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        out = np.empty(1, dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(out),
            DeviceBuffer(index_device.ptr, out.nbytes),
            runtime=runtime,
        )
        return int(out[0])
    finally:
        free(index_device, runtime=runtime)
        if extra_device is not None:
            free(extra_device, runtime=runtime)


PREFIX = 5
EXTRA_IDS = (70, 71, 72)


@pytest.mark.parametrize(
    "index_value",
    [-1, 0, PREFIX - 1, PREFIX, PREFIX + len(EXTRA_IDS) - 1, PREFIX + len(EXTRA_IDS), 2**40],
)
def test_prefix_and_tail_boundaries_match_the_reference_map(
    hip_runtime_and_library, index_value: int
) -> None:
    runtime, library = hip_runtime_and_library

    actual = _remap(runtime, library, index_value, EXTRA_IDS, PREFIX, len(EXTRA_IDS))

    assert actual == _reference(index_value, EXTRA_IDS, PREFIX)


def test_zero_prefix_maps_the_extra_table_only(hip_runtime_and_library) -> None:
    runtime, library = hip_runtime_and_library
    extra_ids = (900, 901, 902)

    for index_value in (-3, 0, 1, 2, 3, 4096):
        actual = _remap(runtime, library, index_value, extra_ids, 0, len(extra_ids))
        assert actual == _reference(index_value, extra_ids, 0)


def test_prefix_only_map_accepts_a_null_table(hip_runtime_and_library) -> None:
    runtime, library = hip_runtime_and_library

    for index_value in (-1, 0, 5, 6, 2**40):
        actual = _remap(runtime, library, index_value, None, 6, 0)
        assert actual == _reference(index_value, (), 6)


def test_invalid_indices_become_the_sentinel_without_reading_past_the_table(
    hip_runtime_and_library,
) -> None:
    runtime, library = hip_runtime_and_library
    extra_ids = (42,)

    # The one-element table is exactly as long as the tail, so ``prefix + 1`` is
    # both the first out-of-range row and the first row a read would overrun.
    assert _remap(runtime, library, PREFIX, extra_ids, PREFIX, 1) == 42
    for index_value in (-(2**40), -1, PREFIX + 1, PREFIX + 2, 2**40):
        assert _remap(runtime, library, index_value, extra_ids, PREFIX, 1) == -1


def test_row_ids_above_the_int32_range_are_preserved(hip_runtime_and_library) -> None:
    runtime, library = hip_runtime_and_library
    prefix_size = 2**32 + 7
    extra_ids = (2**32 + 1000, 2**40 + 3)

    assert _remap(runtime, library, 2**32 + 6, extra_ids, prefix_size, 2) == 2**32 + 6
    assert _remap(runtime, library, prefix_size, extra_ids, prefix_size, 2) == 2**32 + 1000
    assert _remap(runtime, library, prefix_size + 1, extra_ids, prefix_size, 2) == 2**40 + 3
    assert _remap(runtime, library, prefix_size + 2, extra_ids, prefix_size, 2) == -1
    assert _remap(runtime, library, 2**33, extra_ids, prefix_size, 2) == -1
