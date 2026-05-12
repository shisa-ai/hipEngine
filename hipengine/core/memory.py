"""Torch-free HIP memory helpers.

No HIP library is loaded on import. Allocation/copy helpers load the runtime lazily only when
called.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime


@dataclass(frozen=True)
class DeviceBuffer:
    ptr: int
    nbytes: int

    def __post_init__(self) -> None:
        if self.ptr < 0:
            raise ValueError("device pointer must be non-negative")
        if self.nbytes < 0:
            raise ValueError("buffer size must be non-negative")


def malloc(nbytes: int, *, runtime: HipRuntime | None = None) -> DeviceBuffer:
    runtime = runtime or get_hip_runtime()
    return DeviceBuffer(ptr=runtime.malloc(nbytes), nbytes=nbytes)


def free(buffer: DeviceBuffer, *, runtime: HipRuntime | None = None) -> None:
    runtime = runtime or get_hip_runtime()
    runtime.free(buffer.ptr)


def copy_host_to_device(
    buffer: DeviceBuffer,
    host_ptr: int,
    nbytes: int | None = None,
    *,
    runtime: HipRuntime | None = None,
) -> None:
    runtime = runtime or get_hip_runtime()
    count = buffer.nbytes if nbytes is None else nbytes
    _check_copy_size(count, buffer.nbytes)
    runtime.memcpy(buffer.ptr, host_ptr, count, HipMemcpyKind.HOST_TO_DEVICE)


def copy_device_to_host(
    host_ptr: int,
    buffer: DeviceBuffer,
    nbytes: int | None = None,
    *,
    runtime: HipRuntime | None = None,
) -> None:
    runtime = runtime or get_hip_runtime()
    count = buffer.nbytes if nbytes is None else nbytes
    _check_copy_size(count, buffer.nbytes)
    runtime.memcpy(host_ptr, buffer.ptr, count, HipMemcpyKind.DEVICE_TO_HOST)


def host_array_ptr(array: object) -> int:
    """Return a ctypes pointer address for contiguous array-like objects.

    NumPy arrays expose ``ctypes.data``; this helper avoids importing NumPy in core modules.
    """

    ctypes_view = getattr(array, "ctypes", None)
    data = getattr(ctypes_view, "data", None)
    if data is None:
        raise TypeError("object does not expose a ctypes.data pointer")
    return int(data)


def host_buffer_ptr(buffer: ctypes.Array) -> int:
    return int(ctypes.addressof(buffer))


def _check_copy_size(nbytes: int, capacity: int) -> None:
    if nbytes < 0:
        raise ValueError("nbytes must be non-negative")
    if nbytes > capacity:
        raise ValueError(f"copy size {nbytes} exceeds device buffer size {capacity}")
