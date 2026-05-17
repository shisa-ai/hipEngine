"""Torch-free HIP memory helpers.

No HIP library is loaded on import. Allocation/copy helpers load the runtime lazily only when
called.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from threading import Lock

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
    ptr = runtime.malloc(nbytes)
    buffer = DeviceBuffer(ptr=ptr, nbytes=nbytes)
    _MEMORY_STATS.record_malloc(buffer)
    return buffer


def free(buffer: DeviceBuffer, *, runtime: HipRuntime | None = None) -> None:
    runtime = runtime or get_hip_runtime()
    runtime.free(buffer.ptr)
    _MEMORY_STATS.record_free(buffer)


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


def memory_stats() -> dict[str, int]:
    """Return process-local hipEngine device allocation counters.

    The counters cover allocations made through :func:`malloc`, which is the
    torch-free path used by hipEngine runtime/model buffers.  They do not include
    allocations made internally by HIP/AOTriton libraries, but they do preserve a
    real high-water mark for hipEngine-owned buffers even after temporary
    workspaces are released.
    """

    return _MEMORY_STATS.snapshot()


def reset_memory_stats() -> None:
    """Reset counters while preserving currently live tracked allocations."""

    _MEMORY_STATS.reset()


@dataclass
class _MemoryStats:
    current_allocated_bytes: int = 0
    peak_allocated_bytes: int = 0
    total_allocated_bytes: int = 0
    total_freed_bytes: int = 0
    active_allocations: int = 0
    peak_allocations: int = 0


class _MemoryStatsTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._live: dict[int, int] = {}
        self._stats = _MemoryStats()

    def record_malloc(self, buffer: DeviceBuffer) -> None:
        if buffer.ptr == 0 or buffer.nbytes <= 0:
            return
        with self._lock:
            old_nbytes = self._live.get(buffer.ptr)
            if old_nbytes is not None:
                self._stats.current_allocated_bytes -= old_nbytes
                self._stats.active_allocations -= 1
            self._live[buffer.ptr] = int(buffer.nbytes)
            self._stats.current_allocated_bytes += int(buffer.nbytes)
            self._stats.total_allocated_bytes += int(buffer.nbytes)
            self._stats.active_allocations += 1
            self._stats.peak_allocated_bytes = max(
                self._stats.peak_allocated_bytes,
                self._stats.current_allocated_bytes,
            )
            self._stats.peak_allocations = max(self._stats.peak_allocations, self._stats.active_allocations)

    def record_free(self, buffer: DeviceBuffer) -> None:
        if buffer.ptr == 0:
            return
        with self._lock:
            nbytes = self._live.pop(buffer.ptr, None)
            if nbytes is None:
                return
            self._stats.current_allocated_bytes -= nbytes
            self._stats.total_freed_bytes += nbytes
            self._stats.active_allocations -= 1
            if self._stats.current_allocated_bytes < 0 or self._stats.active_allocations < 0:
                # Defensive clamp; this should not happen with tracked pointers.
                self._stats.current_allocated_bytes = max(0, self._stats.current_allocated_bytes)
                self._stats.active_allocations = max(0, self._stats.active_allocations)

    def reset(self) -> None:
        with self._lock:
            current = sum(self._live.values())
            active = len(self._live)
            self._stats = _MemoryStats(
                current_allocated_bytes=current,
                peak_allocated_bytes=current,
                total_allocated_bytes=0,
                total_freed_bytes=0,
                active_allocations=active,
                peak_allocations=active,
            )

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats.__dict__)


_MEMORY_STATS = _MemoryStatsTracker()


def _check_copy_size(nbytes: int, capacity: int) -> None:
    if nbytes < 0:
        raise ValueError("nbytes must be non-negative")
    if nbytes > capacity:
        raise ValueError(f"copy size {nbytes} exceeds device buffer size {capacity}")
