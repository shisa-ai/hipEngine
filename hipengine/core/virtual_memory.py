"""Lazy HIP virtual-memory growable buffer for stable-address device arenas.

Importing this module does not load ``libamdhip64.so`` and does not call the GPU. The HIP
runtime and its virtual-memory entry points are touched only when
:meth:`VirtualMemoryBuffer.reserve` is called.

The buffer reserves one contiguous virtual address range up front and maps physical device
memory into it on demand. Because the base address never moves, a caller can hand out a stable
pointer while growing the committed region in granularity-sized steps -- the property a KV
arena needs so that cached pointers into the arena stay valid across growth.

The reserved virtual address range is *not* physical memory and is deliberately not recorded
in :mod:`hipengine.core.memory` allocation counters; only the committed (mapped) segments are
backed by real device memory. Precise physical accounting is the caller's responsibility.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from hipengine.core.hip import HipError, HipRuntime, get_hip_runtime

# hipMemAllocationType (hip_runtime_api.h)
HIP_MEM_ALLOCATION_TYPE_PINNED: int = 0x01
# hipMemLocationType (driver_types.h)
HIP_MEM_LOCATION_TYPE_DEVICE: int = 0x01
# hipMemAllocationGranularity_flags (hip_runtime_api.h)
HIP_MEM_ALLOCATION_GRANULARITY_MINIMUM: int = 0x00
# hipMemAccessFlags (hip_runtime_api.h)
HIP_MEM_ACCESS_FLAGS_PROT_READWRITE: int = 0x03
# hipError_t codes used for locally-detected failures.
HIP_ERROR_INVALID_VALUE: int = 1
HIP_ERROR_OUT_OF_MEMORY: int = 2
HIP_ERROR_NOT_SUPPORTED: int = 801


class HipMemLocation(ctypes.Structure):
    """ctypes layout of ``hipMemLocation`` from ``driver_types.h``."""

    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class HipMemAccessDesc(ctypes.Structure):
    """ctypes layout of ``hipMemAccessDesc`` from ``hip_runtime_api.h``."""

    _fields_ = [("location", HipMemLocation), ("flags", ctypes.c_int)]


class _HipMemAllocationFlags(ctypes.Structure):
    """ctypes layout of the anonymous ``allocFlags`` member of ``hipMemAllocationProp``."""

    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
    ]


class HipMemAllocationProp(ctypes.Structure):
    """ctypes layout of ``hipMemAllocationProp`` from ``hip_runtime_api.h``.

    The ``requestedHandleType``/``requestedHandleTypes`` union is represented by a single
    ``c_int`` field; both names share the same storage, and the VMM path leaves it zero.
    """

    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", HipMemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _HipMemAllocationFlags),
    ]


_VMM_SIGNATURES: dict[str, tuple[list[object], object]] = {
    "hipMemAddressReserve": (
        [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_ulonglong,
        ],
        ctypes.c_int,
    ),
    "hipMemAddressFree": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
    "hipMemCreate": (
        [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.POINTER(HipMemAllocationProp),
            ctypes.c_ulonglong,
        ],
        ctypes.c_int,
    ),
    "hipMemRelease": ([ctypes.c_void_p], ctypes.c_int),
    "hipMemMap": (
        [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_ulonglong,
        ],
        ctypes.c_int,
    ),
    "hipMemUnmap": ([ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int),
    "hipMemSetAccess": (
        [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(HipMemAccessDesc),
            ctypes.c_size_t,
        ],
        ctypes.c_int,
    ),
    "hipMemGetAllocationGranularity": (
        [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(HipMemAllocationProp),
            ctypes.c_int,
        ],
        ctypes.c_int,
    ),
}


@dataclass
class _MappedSegment:
    """One physical allocation mapped at ``offset`` into the reserved address range.

    ``mapped`` and ``handle_released`` track the two independently-ownable resources of a
    segment so a failed unmap or release can be retried without double-freeing. ``committed``
    is False for a segment retained by a failed :meth:`VirtualMemoryBuffer.commit_to` that
    never entered the published committed prefix.
    """

    offset: int
    size: int
    handle: int
    mapped: bool = True
    handle_released: bool = False
    committed: bool = True

    @property
    def fully_released(self) -> bool:
        return not self.mapped and self.handle_released


def virtual_memory_supported(runtime: HipRuntime | None = None) -> bool:
    """Return True when ``runtime`` exports the HIP virtual-memory API.

    Used by callers that must decide between a stable-address VMM arena and a
    plain ``hipMalloc`` chunk without hard-coding a device or backend. An
    unsupported runtime returns ``False``; no exception escapes and no GPU work
    happens beyond reading the loaded library's symbol table.
    """

    selected = runtime or get_hip_runtime()
    try:
        _configure_vmm(getattr(selected, "library", None))
    except HipError:
        return False
    return True


def _configure_vmm(library: object) -> None:
    """Declare the typed ABI of every required VMM entry point on ``library``.

    Raises :class:`HipError` with ``hipErrorNotSupported`` when the loaded runtime does not
    export one of the functions the primitive needs, instead of leaking ``AttributeError``.
    """

    missing = [name for name in _VMM_SIGNATURES if getattr(library, name, None) is None]
    if missing:
        raise HipError(
            HIP_ERROR_NOT_SUPPORTED,
            "HIP runtime does not export the virtual-memory API: " + ", ".join(missing),
        )
    for name, (argtypes, restype) in _VMM_SIGNATURES.items():
        function = getattr(library, name)
        function.argtypes = argtypes
        function.restype = restype


def _validate_capacity(capacity_bytes: int) -> int:
    value = int(capacity_bytes)
    if value <= 0:
        raise ValueError("capacity_bytes must be positive")
    return value


def _validate_commit_size(nbytes: int) -> int:
    value = int(nbytes)
    if value < 0:
        raise ValueError("commit size must be non-negative")
    return value


def _align_up(value: int, alignment: int) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


def _make_allocation_prop(device: int) -> HipMemAllocationProp:
    prop = HipMemAllocationProp()
    prop.type = HIP_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = 0
    prop.location.type = HIP_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = int(device)
    prop.win32HandleMetaData = None
    return prop


def _make_access_desc(device: int) -> HipMemAccessDesc:
    desc = HipMemAccessDesc()
    desc.location.type = HIP_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = int(device)
    desc.flags = HIP_MEM_ACCESS_FLAGS_PROT_READWRITE
    return desc


def _query_granularity(
    runtime: HipRuntime,
    library: object,
    prop: HipMemAllocationProp,
) -> int:
    granularity = ctypes.c_size_t()
    runtime.check(
        library.hipMemGetAllocationGranularity(
            ctypes.byref(granularity),
            ctypes.byref(prop),
            ctypes.c_int(HIP_MEM_ALLOCATION_GRANULARITY_MINIMUM),
        )
    )
    value = int(granularity.value)
    if value <= 0:
        raise HipError(
            HIP_ERROR_INVALID_VALUE,
            "hipMemGetAllocationGranularity returned a non-positive granularity",
        )
    return value


class VirtualMemoryBuffer:
    """Stable-address device buffer backed by lazily mapped HIP VMM granules.

    Reserve once with :meth:`reserve`, grow the committed region with :meth:`commit_to`, and
    release everything with :meth:`close`. ``ptr`` is constant for the whole lifetime of the
    buffer; committed bytes always trail the pointer and are a multiple of ``granularity``.

    The class is not thread-safe. Committing is transactional: if a step fails, the segment it
    created is unmapped and released, and the committed state is left exactly as it was before
    the call.
    """

    def __init__(
        self,
        *,
        runtime: HipRuntime,
        library: object,
        device: int,
        capacity_bytes: int,
        granularity: int,
        ptr: int,
        prop: HipMemAllocationProp,
        access: HipMemAccessDesc,
    ) -> None:
        self._runtime = runtime
        self._library = library
        self._device = int(device)
        self._capacity_bytes = int(capacity_bytes)
        self._granularity = int(granularity)
        self._ptr = int(ptr)
        self._prop = prop
        self._access = access
        self._committed_bytes = 0
        self._segments: list[_MappedSegment] = []
        self._closed = False

    @classmethod
    def reserve(
        cls,
        capacity_bytes: int,
        *,
        runtime: HipRuntime | None = None,
        device: int | None = None,
    ) -> "VirtualMemoryBuffer":
        """Reserve a stable virtual address range of at least ``capacity_bytes`` bytes.

        ``capacity_bytes`` is rounded up to the device allocation granularity, which is the
        value exposed by :attr:`capacity_bytes`. ``device`` defaults to the runtime's current
        device; no device id is hard-coded. If construction fails after the reservation was
        made, the reservation is freed before the exception propagates.
        """

        selected_runtime = runtime or get_hip_runtime()
        library = selected_runtime.library
        _configure_vmm(library)
        requested = _validate_capacity(capacity_bytes)
        selected_device = (
            selected_runtime.current_device() if device is None else int(device)
        )
        prop = _make_allocation_prop(selected_device)
        granularity = _query_granularity(selected_runtime, library, prop)
        reserved_bytes = _align_up(requested, granularity)

        base = ctypes.c_void_p()
        selected_runtime.check(
            library.hipMemAddressReserve(
                ctypes.byref(base),
                ctypes.c_size_t(reserved_bytes),
                ctypes.c_size_t(0),
                None,
                ctypes.c_ulonglong(0),
            )
        )
        base_ptr = 0 if base.value is None else int(base.value)
        if base_ptr == 0:
            raise HipError(
                HIP_ERROR_OUT_OF_MEMORY,
                "hipMemAddressReserve returned a null virtual address",
            )
        try:
            access = _make_access_desc(selected_device)
            return cls(
                runtime=selected_runtime,
                library=library,
                device=selected_device,
                capacity_bytes=reserved_bytes,
                granularity=granularity,
                ptr=base_ptr,
                prop=prop,
                access=access,
            )
        except BaseException:
            library.hipMemAddressFree(
                ctypes.c_void_p(base_ptr),
                ctypes.c_size_t(reserved_bytes),
            )
            raise

    @property
    def ptr(self) -> int:
        """Base virtual address of the reservation; stable until :meth:`close`."""

        return self._ptr

    @property
    def capacity_bytes(self) -> int:
        """Reserved address-space size, rounded up to :attr:`granularity`."""

        return self._capacity_bytes

    @property
    def committed_bytes(self) -> int:
        """Bytes of physical device memory currently mapped into the reservation."""

        return self._committed_bytes

    @property
    def granularity(self) -> int:
        """Device allocation granularity in bytes."""

        return self._granularity

    @property
    def device(self) -> int:
        """Device id the physical segments are allocated on."""

        return self._device

    @property
    def closed(self) -> bool:
        return self._closed

    def commit_to(self, nbytes: int) -> int:
        """Ensure at least ``nbytes`` bytes are backed by physical device memory.

        Only the additional bytes, rounded up to a granularity multiple, are mapped; a call
        that does not exceed the current committed size is a no-op. Returns the new committed
        byte count. Raises ``MemoryError`` when ``nbytes`` exceeds :attr:`capacity_bytes`.

        Committing is transactional. If the new segment cannot be mapped, its cleanup is
        checked: a mapping or handle that cannot be released is retained for a later
        :meth:`rollback_to`/:meth:`close`, no capacity is published for it, and the cleanup
        error is raised with the original failure as its cause. A buffer holding such a
        segment refuses further commits until it is recovered.
        """

        self._require_open()
        requested = _validate_commit_size(nbytes)
        if requested > self._capacity_bytes:
            raise MemoryError(
                f"virtual memory buffer capacity {self._capacity_bytes} cannot back "
                f"{requested} committed bytes"
            )
        target = _align_up(requested, self._granularity)
        if target <= self._committed_bytes:
            return self._committed_bytes
        if self._has_unreleased_failure():
            raise RuntimeError(
                "virtual memory buffer has an unpublished failed mapping; call "
                "rollback_to() or close() before committing again"
            )

        offset = self._committed_bytes
        size = target - offset
        address = self._ptr + offset

        handle = ctypes.c_void_p()
        self._runtime.check(
            self._library.hipMemCreate(
                ctypes.byref(handle),
                ctypes.c_size_t(size),
                ctypes.byref(self._prop),
                ctypes.c_ulonglong(0),
            )
        )
        handle_value = 0 if handle.value is None else int(handle.value)
        segment = _MappedSegment(offset=offset, size=size, handle=handle_value, mapped=False)
        try:
            self._runtime.check(
                self._library.hipMemMap(
                    ctypes.c_void_p(address),
                    ctypes.c_size_t(size),
                    ctypes.c_size_t(0),
                    handle,
                    ctypes.c_ulonglong(0),
                )
            )
            segment.mapped = True
            self._runtime.check(
                self._library.hipMemSetAccess(
                    ctypes.c_void_p(address),
                    ctypes.c_size_t(size),
                    ctypes.byref(self._access),
                    ctypes.c_size_t(1),
                )
            )
        except BaseException as commit_error:
            cleanup_error = self._cleanup_failed_commit(segment)
            if cleanup_error is not None:
                raise cleanup_error from commit_error
            raise

        self._segments.append(segment)
        self._committed_bytes = target
        return self._committed_bytes

    def _cleanup_failed_commit(self, segment: _MappedSegment) -> HipError | None:
        """Undo a partially mapped segment after a failed commit.

        Returns the first cleanup :class:`HipError`, or ``None`` when the segment was fully
        released. A segment that could not be fully released is retained with
        ``committed=False`` so a later :meth:`rollback_to`/:meth:`close` can finish and no
        capacity is falsely published for it.
        """

        error: HipError | None = None
        if segment.mapped:
            try:
                self._runtime.check(
                    self._library.hipMemUnmap(
                        ctypes.c_void_p(self._ptr + segment.offset),
                        ctypes.c_size_t(segment.size),
                    )
                )
                segment.mapped = False
            except HipError as exc:
                error = error or exc
        if not segment.mapped:
            try:
                self._runtime.check(
                    self._library.hipMemRelease(ctypes.c_void_p(segment.handle))
                )
                segment.handle_released = True
            except HipError as exc:
                error = error or exc
        if not segment.fully_released:
            segment.committed = False
            self._segments.append(segment)
        return error

    def rollback_to(self, committed_bytes: int) -> int:
        """Release every mapping above ``committed_bytes`` and return the new size.

        This is the transactional half of a multi-buffer growth: a caller that
        commits several buffers in sequence and fails partway uses it to undo the
        buffers it already committed, so the failed growth maps no physical
        memory. ``committed_bytes`` must be a granularity multiple no larger than
        the current committed size; a no-op target returns immediately.

        Every segment above the target is attempted even if an earlier one fails.
        A segment whose unmap fails is retained and keeps its bytes committed, so
        :attr:`committed_bytes` never drops below a still-mapped region; a
        segment whose handle release fails is retained for a later retry. The
        first :class:`HipError` is raised once all segments have been visited.
        """

        self._require_open()
        target = _validate_commit_size(committed_bytes)
        if target % self._granularity:
            raise ValueError("rollback target must be a multiple of the granularity")
        if target > self._committed_bytes:
            raise ValueError("rollback target exceeds the committed size")
        if target == self._committed_bytes and not self._has_pending_above(target):
            return self._committed_bytes
        boundaries = {0}
        for segment in self._segments:
            boundaries.add(segment.offset)
        if target not in boundaries:
            raise ValueError("rollback target must fall on a mapped-segment boundary")

        error: HipError | None = None
        remaining: list[_MappedSegment] = []
        new_committed = target
        # A mapped segment that belongs to the published prefix blocks any
        # lower unmap: releasing below it would leave a hole and make a single
        # committed byte count lie. Retain it and everything under it.
        stop = False
        for segment in reversed(self._segments):
            if segment.offset < target:
                remaining.append(segment)
                continue
            if stop:
                remaining.append(segment)
                if segment.mapped and segment.committed:
                    new_committed = max(new_committed, segment.offset + segment.size)
                continue
            if segment.mapped:
                try:
                    self._runtime.check(
                        self._library.hipMemUnmap(
                            ctypes.c_void_p(self._ptr + segment.offset),
                            ctypes.c_size_t(segment.size),
                        )
                    )
                    segment.mapped = False
                except HipError as exc:
                    error = error or exc
                    stop = True
                    remaining.append(segment)
                    if segment.committed:
                        new_committed = max(
                            new_committed, segment.offset + segment.size
                        )
                    continue
            if not segment.handle_released:
                try:
                    self._runtime.check(
                        self._library.hipMemRelease(ctypes.c_void_p(segment.handle))
                    )
                    segment.handle_released = True
                except HipError as exc:
                    error = error or exc
            if not segment.fully_released:
                remaining.append(segment)
        remaining.reverse()
        self._segments = remaining
        self._committed_bytes = new_committed
        if error is not None:
            raise error
        return self._committed_bytes

    def close(self) -> None:
        """Unmap and release every segment, then free the reservation.

        Every step is attempted even if an earlier one fails. A mapping or handle that cannot
        be released is retained, :attr:`committed_bytes` keeps reflecting the still-mapped
        region, and the address reservation is not freed, so a later :meth:`close` can retry
        without double-freeing. The buffer is marked closed only once every segment is
        released and the reservation is freed; the first :class:`HipError` is then re-raised.
        """

        if self._closed:
            return
        error: HipError | None = None
        remaining: list[_MappedSegment] = []
        for segment in reversed(self._segments):
            if segment.mapped:
                try:
                    self._runtime.check(
                        self._library.hipMemUnmap(
                            ctypes.c_void_p(self._ptr + segment.offset),
                            ctypes.c_size_t(segment.size),
                        )
                    )
                    segment.mapped = False
                except HipError as exc:
                    error = error or exc
            if segment.mapped:
                remaining.append(segment)
                continue
            if not segment.handle_released:
                try:
                    self._runtime.check(
                        self._library.hipMemRelease(ctypes.c_void_p(segment.handle))
                    )
                    segment.handle_released = True
                except HipError as exc:
                    error = error or exc
            if not segment.fully_released:
                remaining.append(segment)
        remaining.reverse()
        self._segments = remaining
        self._committed_bytes = max(
            (
                segment.offset + segment.size
                for segment in self._segments
                if segment.mapped and segment.committed
            ),
            default=0,
        )
        if self._segments:
            # A mapping or handle is still owned; keep the reservation so a
            # later close() can finish it rather than leaking silently.
            if error is not None:
                raise error
            return
        try:
            self._runtime.check(
                self._library.hipMemAddressFree(
                    ctypes.c_void_p(self._ptr),
                    ctypes.c_size_t(self._capacity_bytes),
                )
            )
        except HipError as exc:
            error = error or exc
        if error is None:
            self._closed = True
            self._committed_bytes = 0
        if error is not None:
            raise error

    def _has_pending_above(self, target: int) -> bool:
        """True when a segment at or above ``target`` still owns a mapping or handle."""

        return any(
            segment.offset >= target and not segment.fully_released
            for segment in self._segments
        )

    def _has_unreleased_failure(self) -> bool:
        """True when a failed commit left an unpublished mapping or handle behind."""

        return any(
            (not segment.committed)
            or (not segment.mapped and not segment.handle_released)
            for segment in self._segments
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("virtual memory buffer is closed")

    def __enter__(self) -> "VirtualMemoryBuffer":
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False


def rollback_all(
    commits: "list[tuple[VirtualMemoryBuffer, int]] | tuple[tuple[VirtualMemoryBuffer, int], ...]",
) -> None:
    """Roll back several buffers (planes) to their pre-growth committed sizes.

    A caller that commits multiple buffers for one growth uses this to undo them
    together. Every plane is attempted even if an earlier one fails, so a failed
    growth releases as much physical memory as the runtime allows instead of
    stopping at the first error. The first :class:`HipError` is re-raised after
    every plane has been visited; a plane that could not be rolled back retains
    its records for a later retry.
    """

    first_error: HipError | None = None
    for buffer, target in reversed(tuple(commits)):
        try:
            buffer.rollback_to(target)
        except HipError as exc:
            first_error = first_error or exc
    if first_error is not None:
        raise first_error
