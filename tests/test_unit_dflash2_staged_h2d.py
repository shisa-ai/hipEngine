"""CPU behavior coverage for the DFlash2 drafter's persistent H2D staging.

``DFlash2NativeDrafter._staged_h2d`` preserves the existing persistent-staging
workaround for host-to-device uploads: every upload is first copied into one
persistent host staging array, then copied from that array. The staging array
owns the bytes
being copied, so the copy must hand the helper an owning contiguous byte view
instead of re-deriving a bare integer pointer from ``ctypes.data``.

Everything here is observable without a device: the fake runtime records the
leaf ``memcpy`` (pointer, byte count, kind), reads the source bytes through the
pointer the real helper passes down, and writes them into a byte mirror of the
destination. That pins the staging storage, the chunk offsets and counts, the
byte payload, the synchronous kind and the absence of extra synchronization.
"""

from __future__ import annotations

import ast
import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.core import memory
from hipengine.speculative import dflash2_native as mod


class _FakeRuntime:
    """Records the leaf copies and mirrors the destination bytes."""

    def __init__(self) -> None:
        self.memcpy_calls: list[tuple[int, int, int, object]] = []
        self.async_calls: list[tuple[int, int, int, object, int]] = []
        self.syncs = 0
        self.regions: dict[int, bytearray] = {}

    def region(self, base_ptr: int, nbytes: int) -> bytearray:
        self.regions[base_ptr] = bytearray(nbytes)
        return self.regions[base_ptr]

    def memcpy(self, dst: int, src: int, nbytes: int, kind: object) -> None:
        self.memcpy_calls.append((dst, src, int(nbytes), kind))
        base = max(candidate for candidate in self.regions if candidate <= dst)
        region = self.regions[base]
        start = dst - base
        assert start + nbytes <= len(region), "copy writes past the destination region"
        region[start : start + nbytes] = ctypes.string_at(src, nbytes)

    def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
        self.async_calls.append((dst, src, int(nbytes), kind, stream))

    def device_synchronize(self) -> None:
        self.syncs += 1

    def stream_synchronize(self, stream: int) -> None:
        self.syncs += 1


def _install(monkeypatch, runtime: _FakeRuntime) -> list:
    """Record owning uploads, then run the real helper against ``runtime``."""
    calls: list = []
    real = memory.copy_host_array_to_device

    def spy(buffer, array, nbytes=None, *, runtime=None):
        calls.append((buffer, array, nbytes, runtime))
        return real(buffer, array, nbytes, runtime=runtime)

    # ``raising=False``: the module-level import lands with the migration, and
    # the harness has to describe the same boundary on both sides of it.
    monkeypatch.setattr(mod, "copy_host_array_to_device", spy, raising=False)
    return calls


def _drafter(runtime: _FakeRuntime, staging_bytes: int):
    drafter = object.__new__(mod.DFlash2NativeDrafter)
    drafter.runtime = runtime
    drafter._staging_np = np.empty(staging_bytes // 4, dtype=np.float32)
    drafter._staging_nbytes = staging_bytes
    if "_staging_ptr" in Path(mod.__file__).read_text():
        # Pre-migration support only: the old code copies from this bare
        # address. Kept so RED fails on the recorded traffic (no owning
        # upload) instead of on a missing harness attribute.
        drafter._staging_ptr = int(drafter._staging_np.ctypes.data)
    return drafter


def _owner_of(array: np.ndarray) -> np.ndarray:
    """Walk ``.base`` to the array that actually owns the buffer."""
    owner = array
    while owner.base is not None:
        owner = owner.base
    return owner


def _payload(nbytes: int, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, nbytes, dtype=np.uint8)


def test_smaller_than_capacity_uploads_once(monkeypatch) -> None:
    runtime = _FakeRuntime()
    calls = _install(monkeypatch, runtime)
    drafter = _drafter(runtime, 4096)
    payload = _payload(1000)
    destination = runtime.region(0x9000, payload.nbytes)

    drafter._staged_h2d(0x9000, payload)

    assert len(calls) == 1
    buffer, array, nbytes, call_runtime = calls[0]
    assert (buffer.ptr, buffer.nbytes) == (0x9000, 1000)
    assert nbytes == 1000
    assert call_runtime is runtime
    # The copied bytes are a view of the persistent staging array, which stays
    # the owner of the buffer the leaf copy reads.
    assert array.nbytes == 1000
    assert _owner_of(array) is drafter._staging_np
    assert bytes(destination) == payload.tobytes()

    assert len(runtime.memcpy_calls) == 1
    dst, src, count, kind = runtime.memcpy_calls[0]
    staging_ptr = int(drafter._staging_np.ctypes.data)
    assert (dst, count, kind) == (0x9000, 1000, HipMemcpyKind.HOST_TO_DEVICE)
    assert staging_ptr <= src < staging_ptr + drafter._staging_nbytes
    assert runtime.async_calls == []
    assert runtime.syncs == 0


def test_equal_to_capacity_uploads_once(monkeypatch) -> None:
    runtime = _FakeRuntime()
    calls = _install(monkeypatch, runtime)
    drafter = _drafter(runtime, 2048)
    payload = _payload(2048, seed=11)
    destination = runtime.region(0x4000, payload.nbytes)

    drafter._staged_h2d(0x4000, payload)

    assert len(calls) == 1
    buffer, array, nbytes, _ = calls[0]
    assert (buffer.ptr, buffer.nbytes, nbytes) == (0x4000, 2048, 2048)
    assert array.nbytes == 2048
    assert bytes(destination) == payload.tobytes()
    assert len(runtime.memcpy_calls) == 1
    assert runtime.syncs == 0


def test_larger_than_capacity_chunks_with_exact_offsets(monkeypatch) -> None:
    runtime = _FakeRuntime()
    calls = _install(monkeypatch, runtime)
    drafter = _drafter(runtime, 1024)
    payload = _payload(2500, seed=13)
    destination = runtime.region(0x20000, payload.nbytes)

    drafter._staged_h2d(0x20000, payload)

    # Full chunks, then the short tail: same chunking and offsets as before.
    assert [(buffer.ptr, buffer.nbytes, nbytes) for buffer, _, nbytes, _ in calls] == [
        (0x20000, 1024, 1024),
        (0x20000 + 1024, 1024, 1024),
        (0x20000 + 2048, 452, 452),
    ]
    assert bytes(destination) == payload.tobytes()
    # The destination view is the written region only, so the helper's bound
    # check covers exactly the bytes this chunk may touch.
    for buffer, array, nbytes, call_runtime in calls:
        assert call_runtime is runtime
        assert array.nbytes == nbytes
        assert buffer.ptr + buffer.nbytes <= 0x20000 + payload.nbytes
    assert [count for _, _, count, _ in runtime.memcpy_calls] == [1024, 1024, 452]
    assert [kind for _, _, _, kind in runtime.memcpy_calls] == [
        HipMemcpyKind.HOST_TO_DEVICE
    ] * 3
    assert runtime.async_calls == []
    assert runtime.syncs == 0


def test_tail_chunk_uploads_only_its_own_bytes(monkeypatch) -> None:
    """A one-byte tail must not drag the stale prefix of the staging buffer."""
    runtime = _FakeRuntime()
    calls = _install(monkeypatch, runtime)
    drafter = _drafter(runtime, 256)
    payload = _payload(257, seed=17)
    destination = runtime.region(0x6000, payload.nbytes)

    drafter._staged_h2d(0x6000, payload)

    assert [(buffer.ptr, nbytes) for buffer, _, nbytes, _ in calls] == [
        (0x6000, 256),
        (0x6000 + 256, 1),
    ]
    assert [array.nbytes for _, array, _, _ in calls] == [256, 1]
    assert bytes(destination) == payload.tobytes()
    assert runtime.syncs == 0


def test_noncontiguous_input_is_compacted(monkeypatch) -> None:
    runtime = _FakeRuntime()
    calls = _install(monkeypatch, runtime)
    drafter = _drafter(runtime, 1024)
    source = np.arange(300, dtype=np.int32)[::2]  # strided, 150 elements
    expected = np.ascontiguousarray(source)
    destination = runtime.region(0x8000, expected.nbytes)

    drafter._staged_h2d(0x8000, source)

    assert expected.nbytes == 600
    # The strided input views a 1200-byte buffer; only the compacted 600 bytes
    # may be uploaded, so a raw strided upload would copy the wrong bytes.
    assert source.base.nbytes == 1200 and not source.flags.c_contiguous
    assert [(buffer.nbytes, nbytes) for buffer, _, nbytes, _ in calls] == [(600, 600)]
    assert bytes(destination) == expected.tobytes()


def test_module_has_no_bare_staging_pointer_upload() -> None:
    """The staging bytes must reach the device through the owning helper.

    ``_staging_ptr`` only existed to feed ``runtime.memcpy`` a bare address;
    with the owning view it has no remaining use, and a direct H2D
    ``runtime.memcpy`` would bypass the source ownership again.
    """
    source = Path(mod.__file__).read_text()
    assert "_staging_ptr" not in source

    tree = ast.parse(source)
    hazards = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "memcpy"
        ):
            continue
        kinds = [ast.unparse(argument) for argument in node.args]
        if any("HOST_TO_DEVICE" in kind for kind in kinds):
            hazards.append(node.lineno)
    assert hazards == [], f"direct H2D memcpy at lines {hazards}"

    owning = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "copy_host_array_to_device"
    ]
    assert owning, "no owning upload in the module"
    for call in owning:
        assert len(call.args) >= 3, (
            f"line {call.lineno}: upload must pass the copied byte count explicitly"
        )
        destination = call.args[0]
        assert (
            isinstance(destination, ast.Call)
            and isinstance(destination.func, ast.Name)
            and destination.func.id == "DeviceBuffer"
        ), f"line {call.lineno}: destination must be an explicit DeviceBuffer view"


def test_staged_h2d_does_not_synchronize(monkeypatch) -> None:
    """The staged path stays a plain synchronous ``hipMemcpy``."""
    runtime = _FakeRuntime()
    _install(monkeypatch, runtime)
    drafter = _drafter(runtime, 64)
    runtime.region(0x1000, 128)

    drafter._staged_h2d(0x1000, _payload(128, seed=19))

    assert runtime.syncs == 0
    assert runtime.async_calls == []
    assert {kind for _, _, _, kind in runtime.memcpy_calls} == {
        HipMemcpyKind.HOST_TO_DEVICE
    }


@pytest.mark.parametrize("nbytes", [1, 63, 64, 65, 128])
def test_chunk_boundaries_around_capacity(monkeypatch, nbytes) -> None:
    runtime = _FakeRuntime()
    calls = _install(monkeypatch, runtime)
    drafter = _drafter(runtime, 64)
    payload = _payload(nbytes, seed=nbytes)
    destination = runtime.region(0x3000, nbytes)

    drafter._staged_h2d(0x3000, payload)

    assert sum(nbytes for _, _, nbytes, _ in calls) == nbytes
    assert [buffer.ptr for buffer, _, _, _ in calls] == [
        0x3000 + offset for offset in range(0, nbytes, 64)
    ]
    assert bytes(destination) == payload.tobytes()
    assert runtime.syncs == 0
