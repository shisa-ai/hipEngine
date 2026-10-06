"""Mocked-HIP unit tests for :mod:`hipengine.core.virtual_memory`.

No GPU and no real ``libamdhip64.so`` are involved: a fake library records the VMM calls and
can inject failures so the transactional cleanup paths are exercised deterministically.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import hipengine.core.virtual_memory as virtual_memory
from hipengine.core.hip import HipError, HipRuntime
from hipengine.core.virtual_memory import VirtualMemoryBuffer

_REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeFunction:
    def __init__(self, func) -> None:
        self.func = func
        self.argtypes = None
        self.restype = None
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.func(*args)


class FakeVmmLibrary:
    """Minimal HIP VMM surface that records calls and can fail on demand."""

    GRANULARITY = 4096

    def __init__(self, device: int = 1) -> None:
        self.device = device
        self.next_address = 0x100000000
        self.next_handle = 0x9000
        self.reservations: dict[int, int] = {}
        self.reserved_sizes: list[int] = []
        self.created: list[tuple[int, int]] = []
        self.mapped: list[tuple[int, int, int]] = []
        self.unmapped: list[tuple[int, int]] = []
        self.released: list[int] = []
        self.accesses: list[tuple[int, int, int, int]] = []
        self.freed_addresses: list[tuple[int, int]] = []
        self.prop_devices: list[int] = []
        self.fail: dict[str, int] = {}

        self.hipGetDevice = FakeFunction(self._get_device)
        self.hipMemGetAllocationGranularity = FakeFunction(self._granularity)
        self.hipMemAddressReserve = FakeFunction(self._address_reserve)
        self.hipMemAddressFree = FakeFunction(self._address_free)
        self.hipMemCreate = FakeFunction(self._create)
        self.hipMemRelease = FakeFunction(self._release)
        self.hipMemMap = FakeFunction(self._map)
        self.hipMemUnmap = FakeFunction(self._unmap)
        self.hipMemSetAccess = FakeFunction(self._set_access)
        self.hipGetErrorString = FakeFunction(
            lambda code: f"fake hip error {int(code)}".encode()
        )

    def _fail(self, name: str) -> int | None:
        return self.fail.get(name)

    def _get_device(self, out_device):
        out_device._obj.value = self.device
        return 0

    def _granularity(self, out_granularity, prop, option):
        code = self._fail("hipMemGetAllocationGranularity")
        if code is not None:
            return code
        self.prop_devices.append(int(prop._obj.location.id))
        out_granularity._obj.value = self.GRANULARITY
        return 0

    def _address_reserve(self, out_ptr, size, alignment, address, flags):
        code = self._fail("hipMemAddressReserve")
        if code is not None:
            return code
        base = self.next_address
        self.next_address += int(size.value) + 0x10000
        out_ptr._obj.value = base
        self.reservations[base] = int(size.value)
        self.reserved_sizes.append(int(size.value))
        return 0

    def _address_free(self, ptr, size):
        code = self._fail("hipMemAddressFree")
        if code is not None:
            return code
        self.freed_addresses.append((ptr.value, size.value))
        self.reservations.pop(ptr.value, None)
        return 0

    def _create(self, out_handle, size, prop, flags):
        code = self._fail("hipMemCreate")
        if code is not None:
            return code
        self.prop_devices.append(int(prop._obj.location.id))
        handle = self.next_handle
        self.next_handle += 0x10
        out_handle._obj.value = handle
        self.created.append((handle, int(size.value)))
        return 0

    def _release(self, handle):
        code = self._fail("hipMemRelease")
        if code is not None:
            return code
        self.released.append(handle.value)
        return 0

    def _map(self, ptr, size, offset, handle, flags):
        code = self._fail("hipMemMap")
        if code is not None:
            return code
        self.mapped.append((ptr.value, int(size.value), handle.value))
        return 0

    def _unmap(self, ptr, size):
        code = self._fail("hipMemUnmap")
        if code is not None:
            return code
        self.unmapped.append((ptr.value, int(size.value)))
        return 0

    def _set_access(self, ptr, size, desc, count):
        code = self._fail("hipMemSetAccess")
        if code is not None:
            return code
        descriptor = desc._obj
        self.accesses.append(
            (ptr.value, int(size.value), int(descriptor.flags), int(descriptor.location.id))
        )
        return 0


def _runtime(library: FakeVmmLibrary) -> HipRuntime:
    return HipRuntime(library)  # type: ignore[arg-type]


def test_import_does_not_load_runtime_or_torch() -> None:
    script = (
        "import sys;"
        "import hipengine.core.virtual_memory;"
        "from hipengine.core.hip import is_default_runtime_loaded;"
        "print(is_default_runtime_loaded());"
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split() == ["False", "False"]


def test_reserve_rounds_capacity_and_reports_granularity() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(5000, runtime=_runtime(library))

    assert buffer.granularity == 4096
    assert buffer.capacity_bytes == 8192
    assert buffer.committed_bytes == 0
    assert buffer.ptr == 0x100000000
    assert library.reserved_sizes == [8192]
    assert library.reservations == {0x100000000: 8192}


def test_reserve_uses_current_device_and_explicit_override() -> None:
    library = FakeVmmLibrary(device=3)
    runtime = _runtime(library)

    buffer = VirtualMemoryBuffer.reserve(4096, runtime=runtime)
    assert buffer.device == 3
    assert library.prop_devices[0] == 3

    override = VirtualMemoryBuffer.reserve(4096, runtime=runtime, device=5)
    assert override.device == 5
    assert library.prop_devices[-1] == 5


def test_three_grows_keep_stable_pointer_and_map_adjacent_granules() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(3 * 4096, runtime=_runtime(library))
    base = buffer.ptr

    assert buffer.commit_to(1) == 4096
    assert buffer.ptr == base
    assert buffer.commit_to(4096) == 4096  # idempotent no-op
    assert buffer.commit_to(5000) == 8192
    assert buffer.commit_to(3 * 4096) == 3 * 4096
    assert buffer.ptr == base

    assert library.mapped == [
        (base, 4096, 0x9000),
        (base + 4096, 4096, 0x9010),
        (base + 8192, 4096, 0x9020),
    ]
    assert library.accesses == [
        (base, 4096, 3, 1),
        (base + 4096, 4096, 3, 1),
        (base + 8192, 4096, 3, 1),
    ]


def test_commit_to_zero_is_a_noop() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(8192, runtime=_runtime(library))

    assert buffer.commit_to(0) == 0
    assert buffer.commit_to(0) == 0
    assert library.created == []


def test_commit_beyond_capacity_raises_without_touching_state() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(8192, runtime=_runtime(library))
    buffer.commit_to(4096)

    with pytest.raises(MemoryError, match="capacity"):
        buffer.commit_to(8193)

    assert buffer.committed_bytes == 4096
    assert len(library.mapped) == 1


def test_negative_commit_size_is_rejected() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(4096, runtime=_runtime(library))
    with pytest.raises(ValueError, match="non-negative"):
        buffer.commit_to(-1)


def test_reserve_rejects_non_positive_capacity() -> None:
    library = FakeVmmLibrary()
    with pytest.raises(ValueError, match="positive"):
        VirtualMemoryBuffer.reserve(0, runtime=_runtime(library))


def test_create_failure_leaves_no_partial_state() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(8192, runtime=_runtime(library))
    library.fail["hipMemCreate"] = 2

    with pytest.raises(HipError) as excinfo:
        buffer.commit_to(4096)

    assert excinfo.value.code == 2
    assert buffer.committed_bytes == 0
    assert library.created == []
    assert library.mapped == []
    assert library.released == []


def test_map_failure_releases_created_handle() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(8192, runtime=_runtime(library))
    library.fail["hipMemMap"] = 1

    with pytest.raises(HipError) as excinfo:
        buffer.commit_to(4096)

    assert excinfo.value.code == 1
    assert buffer.committed_bytes == 0
    assert len(library.created) == 1
    assert library.released == [library.created[0][0]]
    assert library.mapped == []
    assert library.unmapped == []


def test_set_access_failure_unmaps_and_releases() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(8192, runtime=_runtime(library))
    library.fail["hipMemSetAccess"] = 1

    with pytest.raises(HipError):
        buffer.commit_to(4096)

    base = buffer.ptr
    assert buffer.committed_bytes == 0
    assert library.mapped == [(base, 4096, 0x9000)]
    assert library.unmapped == [(base, 4096)]
    assert library.released == [0x9000]


def test_prior_committed_state_survives_later_failure() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(3 * 4096, runtime=_runtime(library))
    buffer.commit_to(4096)

    library.fail["hipMemCreate"] = 2
    with pytest.raises(HipError):
        buffer.commit_to(8192)

    assert buffer.committed_bytes == 4096
    assert len(library.mapped) == 1
    assert library.mapped[0][1] == 4096


def test_close_unmaps_releases_and_frees_in_reverse_order() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(3 * 4096, runtime=_runtime(library))
    buffer.commit_to(4096)
    buffer.commit_to(8192)
    base = buffer.ptr

    buffer.close()

    assert buffer.closed
    assert buffer.committed_bytes == 0
    assert library.unmapped == [(base + 4096, 4096), (base, 4096)]
    assert library.released == [0x9010, 0x9000]
    assert library.freed_addresses == [(base, 3 * 4096)]


def test_close_is_idempotent() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(8192, runtime=_runtime(library))
    buffer.commit_to(4096)

    buffer.close()
    buffer.close()

    assert len(library.unmapped) == 1
    assert len(library.released) == 1
    assert len(library.freed_addresses) == 1


def test_close_without_commit_only_frees_reservation() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(8192, runtime=_runtime(library))

    buffer.close()

    assert library.unmapped == []
    assert library.released == []
    assert library.freed_addresses == [(buffer.ptr, 8192)]


def test_operations_after_close_raise() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(8192, runtime=_runtime(library))
    buffer.close()

    with pytest.raises(RuntimeError, match="closed"):
        buffer.commit_to(4096)
    with pytest.raises(RuntimeError, match="closed"):
        buffer.__enter__()


def test_context_manager_closes_buffer() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(8192, runtime=_runtime(library))
    with buffer as entered:
        assert entered is buffer
        entered.commit_to(4096)
    assert buffer.closed
    assert library.freed_addresses == [(buffer.ptr, 8192)]


def test_unsupported_api_raises_named_hip_error() -> None:
    library = FakeVmmLibrary()
    del library.hipMemCreate

    with pytest.raises(HipError) as excinfo:
        VirtualMemoryBuffer.reserve(8192, runtime=_runtime(library))

    assert excinfo.value.code == 801
    assert "hipMemCreate" in str(excinfo.value)
    assert library.reserved_sizes == []


def test_partial_construction_frees_reservation(monkeypatch) -> None:
    library = FakeVmmLibrary()

    def boom(device: int) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(virtual_memory, "_make_access_desc", boom)

    with pytest.raises(RuntimeError, match="boom"):
        VirtualMemoryBuffer.reserve(8192, runtime=_runtime(library))

    assert library.reservations == {}
    assert len(library.freed_addresses) == 1
    assert library.freed_addresses[0][1] == 8192


def test_virtual_reservation_is_not_counted_as_physical_memory() -> None:
    from hipengine.core.memory import memory_stats

    library = FakeVmmLibrary()
    before = memory_stats()
    buffer = VirtualMemoryBuffer.reserve(3 * 4096, runtime=_runtime(library))
    buffer.commit_to(8192)
    buffer.close()

    assert memory_stats() == before


def test_reserve_failure_before_address_reservation_leaves_nothing() -> None:
    library = FakeVmmLibrary()
    library.fail["hipMemGetAllocationGranularity"] = 801

    with pytest.raises(HipError) as excinfo:
        VirtualMemoryBuffer.reserve(8192, runtime=_runtime(library))

    assert excinfo.value.code == 801
    assert library.reserved_sizes == []
    assert library.freed_addresses == []


# ---------------------------------------------------------------------------
# Ownership retention and retry on rollback / close / commit cleanup.
# ---------------------------------------------------------------------------


def test_rollback_unmap_failure_keeps_commit_and_retains_handle() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(3 * 4096, runtime=_runtime(library))
    base = buffer.ptr
    buffer.commit_to(4096)
    buffer.commit_to(2 * 4096)
    library.fail["hipMemUnmap"] = 1

    with pytest.raises(HipError) as excinfo:
        buffer.rollback_to(0)

    assert excinfo.value.code == 1
    # The top segment is still mapped, so its bytes must stay committed and its
    # handle must not be released while the mapping is alive.
    assert buffer.committed_bytes == 2 * 4096
    assert library.unmapped == []
    assert library.released == []

    del library.fail["hipMemUnmap"]
    assert buffer.rollback_to(0) == 0
    assert buffer.committed_bytes == 0
    assert library.unmapped == [(base + 4096, 4096), (base, 4096)]
    assert library.released == [0x9010, 0x9000]
    buffer.close()


def test_rollback_release_failure_retains_handle_for_retry() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(2 * 4096, runtime=_runtime(library))
    base = buffer.ptr
    buffer.commit_to(4096)
    library.fail["hipMemRelease"] = 1

    with pytest.raises(HipError):
        buffer.rollback_to(0)

    # The mapping is gone, so no bytes are committed, but the handle is retained
    # for a later release rather than leaked silently.
    assert buffer.committed_bytes == 0
    assert library.unmapped == [(base, 4096)]
    assert library.released == []

    del library.fail["hipMemRelease"]
    assert buffer.rollback_to(0) == 0
    assert library.released == [0x9000]
    buffer.close()


def test_commit_cleanup_failure_is_retained_and_blocks_further_commits() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(2 * 4096, runtime=_runtime(library))
    library.fail["hipMemSetAccess"] = 1
    library.fail["hipMemUnmap"] = 1

    with pytest.raises(HipError) as excinfo:
        buffer.commit_to(4096)

    assert excinfo.value.code == 1
    # No capacity is published and the still-mapped handle is not released.
    assert buffer.committed_bytes == 0
    assert library.released == []
    assert library.unmapped == []

    with pytest.raises(RuntimeError, match="failed mapping"):
        buffer.commit_to(4096)

    del library.fail["hipMemUnmap"]
    assert buffer.rollback_to(0) == 0
    assert library.unmapped == [(buffer.ptr, 4096)]
    assert library.released == [0x9000]
    buffer.close()
    assert library.freed_addresses == [(buffer.ptr, 2 * 4096)]


def test_close_unmap_failure_is_retryable_without_double_free() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(2 * 4096, runtime=_runtime(library))
    base = buffer.ptr
    buffer.commit_to(2 * 4096)
    library.fail["hipMemUnmap"] = 1

    with pytest.raises(HipError):
        buffer.close()

    assert buffer.closed is False
    assert buffer.committed_bytes == 2 * 4096
    assert library.freed_addresses == []

    del library.fail["hipMemUnmap"]
    buffer.close()
    assert buffer.closed is True
    assert buffer.committed_bytes == 0
    assert library.unmapped == [(base, 2 * 4096)]
    assert library.released == [0x9000]
    assert library.freed_addresses == [(base, 2 * 4096)]


def test_close_release_failure_is_retryable() -> None:
    library = FakeVmmLibrary()
    buffer = VirtualMemoryBuffer.reserve(2 * 4096, runtime=_runtime(library))
    base = buffer.ptr
    buffer.commit_to(2 * 4096)
    library.fail["hipMemRelease"] = 1

    with pytest.raises(HipError):
        buffer.close()

    assert buffer.closed is False
    assert buffer.committed_bytes == 0
    assert library.unmapped == [(base, 2 * 4096)]
    assert library.freed_addresses == []

    del library.fail["hipMemRelease"]
    buffer.close()
    assert buffer.closed is True
    assert library.released == [0x9000]
    assert library.freed_addresses == [(base, 2 * 4096)]


def test_rollback_all_rolls_back_every_plane_and_reports_first_error() -> None:
    library = FakeVmmLibrary()
    planes = [
        VirtualMemoryBuffer.reserve(2 * 4096, runtime=_runtime(library))
        for _ in range(3)
    ]
    for plane in planes:
        plane.commit_to(4096)
        plane.commit_to(2 * 4096)
    failing_top = planes[1].ptr + 4096
    real_unmap = library.hipMemUnmap.func

    def unmap(ptr, size):
        if int(ptr.value) == failing_top:
            return 1
        return real_unmap(ptr, size)

    library.hipMemUnmap.func = unmap
    with pytest.raises(HipError) as excinfo:
        virtual_memory.rollback_all([(plane, 0) for plane in planes])

    assert excinfo.value.code == 1
    # The two healthy planes rolled back even though the middle one failed.
    assert planes[0].committed_bytes == 0
    assert planes[2].committed_bytes == 0
    assert planes[1].committed_bytes == 2 * 4096

    library.hipMemUnmap.func = real_unmap
    virtual_memory.rollback_all([(planes[1], 0)])
    assert planes[1].committed_bytes == 0
    for plane in planes:
        plane.close()
    assert library.reservations == {}
