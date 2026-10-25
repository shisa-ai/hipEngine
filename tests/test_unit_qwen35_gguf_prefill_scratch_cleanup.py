"""Partial-failure cleanup contracts for the GGUF prefill scratch allocator.

``_GGUFFullAttentionPrefillScratch.allocate`` acquires device owners in two
stages (the bulk planes or liveness arenas, then the dedicated metadata
buffers) and uploads host metadata in a nested ``try``. Whenever any step
fails it must release every owner it acquired exactly once - no leak, no
double free - and propagate the original error unchanged.

These tests drive the real allocator against a recording malloc / free /
upload boundary, so an owner released twice is observable as a free-count
mismatch rather than as source text. Three layouts are covered: the dedicated
per-field fallback, the single-arena liveness plan, and the owner-slot
liveness plan (several arena owners).
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import _GGUFFullAttentionPrefillScratch
from tests.test_unit_qwen35_gguf_prefill_scratch_liveness import (
    _clear_diagnostic_environment,
    _fake_runner,
)

# layout name -> (backend, rows, capacity, expected allocation_mode)
_LAYOUTS: dict[str, tuple[str, int, int, str]] = {
    # Unvalidated backend: no liveness policy, one owner per scratch field.
    "dedicated": ("cpu_reference", 8, 8, "dedicated"),
    # hip_gfx1100 MoE: one bulk arena owner plus dedicated metadata owners.
    "arena": ("hip_gfx1100", 256, 256, "liveness_aliased"),
    # hip_gfx1151 MoE: owner-slot grouping, i.e. several arena owners.
    "owner_slots": ("hip_gfx1151", 768, 768, "liveness_aliased"),
}


class _RecordingDevice:
    """Device boundary that records acquisitions, releases, and uploads.

    ``fail_alloc_at`` / ``fail_upload_at`` are 1-based attempt ordinals: the
    n-th allocation or upload raises the synthetic error before touching any
    recording, so a failed attempt never looks like an acquired owner.
    """

    def __init__(
        self,
        *,
        fail_alloc_at: int | None = None,
        fail_upload_at: int | None = None,
    ) -> None:
        self._next_ptr = 0x1000_0000
        self.allocations: list[DeviceBuffer] = []
        self.frees: list[int] = []
        self.uploads: list[int] = []
        self.alloc_attempts = 0
        self.upload_attempts = 0
        self.fail_alloc_at = fail_alloc_at
        self.fail_upload_at = fail_upload_at
        self.alloc_error = RuntimeError("synthetic device allocation failure")
        self.upload_error = RuntimeError("synthetic host upload failure")

    def malloc(self, nbytes: int, *, runtime: object) -> DeviceBuffer:
        self.alloc_attempts += 1
        if self.fail_alloc_at is not None and self.alloc_attempts == self.fail_alloc_at:
            raise self.alloc_error
        size = int(nbytes)
        buffer = DeviceBuffer(ptr=self._next_ptr, nbytes=size)
        self._next_ptr += max(256, ((size + 255) // 256) * 256 + 256)
        self.allocations.append(buffer)
        return buffer

    def free(self, buffer: DeviceBuffer, *, runtime: object) -> None:
        self.frees.append(int(buffer.ptr))

    def copy_host_to_device(
        self,
        buffer: DeviceBuffer,
        host_ptr: int,
        nbytes: int | None = None,
        *,
        runtime: object,
    ) -> None:
        self.upload_attempts += 1
        if self.fail_upload_at is not None and self.upload_attempts == self.fail_upload_at:
            raise self.upload_error
        self.uploads.append(int(buffer.ptr))


def _install_recording_device(
    monkeypatch: pytest.MonkeyPatch, device: _RecordingDevice
) -> None:
    monkeypatch.setattr(gguf_runner, "malloc", device.malloc)
    monkeypatch.setattr(gguf_runner, "free", device.free)
    monkeypatch.setattr(gguf_runner, "copy_host_to_device", device.copy_host_to_device)


def _allocate(monkeypatch: pytest.MonkeyPatch, layout: str, device: _RecordingDevice):
    _clear_diagnostic_environment(monkeypatch)
    _install_recording_device(monkeypatch, device)
    backend, rows, capacity, _mode = _LAYOUTS[layout]
    return _GGUFFullAttentionPrefillScratch.allocate(
        _fake_runner(backend),
        rows=rows,
        capacity=capacity,
        allocate_kv_cache=False,
        runtime=SimpleNamespace(),
    )


def _control_counts(monkeypatch: pytest.MonkeyPatch, layout: str) -> tuple[int, int]:
    """Owner and upload counts for a clean allocation of ``layout``."""

    device = _RecordingDevice()
    scratch = _allocate(monkeypatch, layout, device)
    assert scratch.allocation_mode == _LAYOUTS[layout][3]
    return len(device.allocations), len(device.uploads)


def _assert_released_exactly_once(device: _RecordingDevice) -> None:
    acquired = [int(buffer.ptr) for buffer in device.allocations]
    counts = Counter(device.frees)
    duplicated = sorted(ptr for ptr, count in counts.items() if count > 1)
    assert not duplicated, f"device owners released more than once: {duplicated}"
    unknown = sorted(set(counts) - set(acquired))
    assert not unknown, f"released device pointers that were never acquired: {unknown}"
    leaked = sorted(set(acquired) - set(counts))
    assert not leaked, f"acquired device owners leaked on failure: {leaked}"


@pytest.mark.parametrize("layout", tuple(_LAYOUTS))
def test_scratch_success_hands_every_owner_to_the_caller(
    monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    device = _RecordingDevice()

    scratch = _allocate(monkeypatch, layout, device)

    assert scratch.allocation_mode == _LAYOUTS[layout][3]
    assert device.uploads, "successful allocation must upload its host metadata"
    assert device.frees == [], "a successful allocation must not release its owners"
    assert sorted((b.ptr, b.nbytes) for b in scratch.buffers) == sorted(
        (b.ptr, b.nbytes) for b in device.allocations
    )
    assert len({b.ptr for b in scratch.buffers}) == len(scratch.buffers)


@pytest.mark.parametrize("layout", tuple(_LAYOUTS))
@pytest.mark.parametrize("failure_point", ("first", "middle", "last"))
def test_scratch_allocation_failure_releases_acquired_owners_once(
    monkeypatch: pytest.MonkeyPatch, layout: str, failure_point: str
) -> None:
    owners_total, _uploads_total = _control_counts(monkeypatch, layout)
    ordinal = {"first": 1, "middle": owners_total // 2, "last": owners_total}[failure_point]
    device = _RecordingDevice(fail_alloc_at=ordinal)

    with pytest.raises(RuntimeError) as excinfo:
        _allocate(monkeypatch, layout, device)

    assert excinfo.value is device.alloc_error
    assert device.alloc_attempts == ordinal
    assert len(device.allocations) == ordinal - 1
    assert device.uploads == []
    _assert_released_exactly_once(device)


@pytest.mark.parametrize("layout", tuple(_LAYOUTS))
@pytest.mark.parametrize("failure_point", ("first", "middle", "last"))
def test_scratch_upload_failure_releases_every_owner_exactly_once(
    monkeypatch: pytest.MonkeyPatch, layout: str, failure_point: str
) -> None:
    owners_total, uploads_total = _control_counts(monkeypatch, layout)
    ordinal = {
        "first": 1,
        "middle": max(1, uploads_total // 2),
        "last": uploads_total,
    }[failure_point]
    device = _RecordingDevice(fail_upload_at=ordinal)

    with pytest.raises(RuntimeError) as excinfo:
        _allocate(monkeypatch, layout, device)

    assert excinfo.value is device.upload_error
    assert device.upload_attempts == ordinal
    assert len(device.uploads) == ordinal - 1
    # The upload stage runs after every owner has been acquired, so this is the
    # partial-failure case where all owners must be released.
    assert len(device.allocations) == owners_total
    assert set(device.uploads) <= {int(buffer.ptr) for buffer in device.allocations}
    _assert_released_exactly_once(device)
