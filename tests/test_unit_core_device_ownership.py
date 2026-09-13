"""CPU-only contracts for explicit HIP device ownership.

These tests use a fake device runtime so they run on hosts without ROCm. Real
peer-access and copy behavior is covered by the guarded GPU tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.core.device import Device, scoped_current_device
from hipengine.core.hip import format_hip_uuid
from hipengine.core.memory import (
    DeviceBuffer,
    DeviceMemoryArena,
    copy_device_to_device,
    free,
    malloc,
)
from hipengine.core.runtime import MemcpyKind


class FakeDeviceRuntime:
    """Minimal DeviceRuntime that records device selection and copy calls."""

    def __init__(self, *, current: int = 0) -> None:
        self._current = current
        self.selection: list[int] = []
        self.calls: list[tuple[str, tuple]] = []
        self._next_ptr = 0x1000

    def get_device(self) -> int:
        return self._current

    def set_device(self, device: int) -> None:
        self._current = int(device)
        self.selection.append(int(device))

    def malloc(self, nbytes: int) -> int:
        self.calls.append(("malloc", (nbytes, self._current)))
        ptr = self._next_ptr
        self._next_ptr += max(1, nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.calls.append(("free", (ptr, self._current)))

    def memcpy(self, dst: int, src: int, nbytes: int, kind: int) -> None:
        self.calls.append(("memcpy", (dst, src, nbytes, int(kind), self._current)))

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: int, stream: int) -> None:
        raise NotImplementedError

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        raise NotImplementedError

    def memset_async(self, dst: int, value: int, nbytes: int, stream: int) -> None:
        raise NotImplementedError

    def stream_create(self, *, nonblocking: bool = True) -> int:
        raise NotImplementedError

    def stream_destroy(self, stream: int) -> None:
        raise NotImplementedError

    def stream_synchronize(self, stream: int) -> None:
        raise NotImplementedError

    def event_create(self, *, flags: int = 0) -> int:
        raise NotImplementedError

    def event_destroy(self, event: int) -> None:
        raise NotImplementedError

    def event_record(self, event: int, stream: int = 0) -> None:
        raise NotImplementedError

    def event_synchronize(self, event: int) -> None:
        raise NotImplementedError

    def event_elapsed_time_ms(self, start: int, stop: int) -> float:
        raise NotImplementedError


def test_format_hip_uuid_canonical() -> None:
    raw = bytes(range(16))
    assert format_hip_uuid(raw) == "00010203-0405-0607-0809-0a0b0c0d0e0f"


def test_format_hip_uuid_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        format_hip_uuid(b"\x00" * 15)


def test_device_parse_roundtrip_and_errors() -> None:
    assert Device.parse("hip:3") == Device("hip", 3)
    assert Device.parse(Device("cuda", 1)) == Device("cuda", 1)
    assert str(Device("hip", 2)) == "hip:2"
    with pytest.raises(ValueError):
        Device.parse("hip:")
    with pytest.raises(ValueError):
        Device("hip", -1)


def test_scoped_current_device_restores_on_success() -> None:
    runtime = FakeDeviceRuntime(current=0)
    with scoped_current_device(runtime, 1) as selected:
        assert selected == 1
        assert runtime.get_device() == 1
    assert runtime.get_device() == 0
    assert runtime.selection == [1, 0]


def test_scoped_current_device_restores_on_exception() -> None:
    runtime = FakeDeviceRuntime(current=1)
    with pytest.raises(RuntimeError):
        with scoped_current_device(runtime, 0):
            raise RuntimeError("boom")
    assert runtime.get_device() == 1


def test_malloc_attributes_current_device() -> None:
    runtime = FakeDeviceRuntime(current=1)
    buffer = malloc(64, runtime=runtime)
    assert buffer.device == Device("hip", 1)
    assert runtime.calls == [("malloc", (64, 1))]


def test_malloc_explicit_device_scopes_and_restores() -> None:
    runtime = FakeDeviceRuntime(current=0)
    buffer = malloc(128, runtime=runtime, device=1)
    assert buffer.device == Device("hip", 1)
    assert runtime.get_device() == 0
    assert runtime.selection == [1, 0]
    assert runtime.calls == [("malloc", (128, 1))]


def test_malloc_rejects_non_hip_device() -> None:
    runtime = FakeDeviceRuntime()
    with pytest.raises(ValueError):
        malloc(16, runtime=runtime, device=Device("cpu"))


def test_free_selects_buffer_device() -> None:
    runtime = FakeDeviceRuntime(current=0)
    buffer = DeviceBuffer(ptr=0x2000, nbytes=16, device=Device("hip", 1))
    free(buffer, runtime=runtime)
    assert runtime.calls == [("free", (0x2000, 1))]
    assert runtime.get_device() == 0


def test_free_unattributed_buffer_uses_current_device() -> None:
    runtime = FakeDeviceRuntime(current=1)
    free(DeviceBuffer(ptr=0x2000, nbytes=16), runtime=runtime)
    assert runtime.calls == [("free", (0x2000, 1))]


def test_copy_device_to_device_requires_attribution() -> None:
    runtime = FakeDeviceRuntime()
    with pytest.raises(ValueError):
        copy_device_to_device(DeviceBuffer(1, 8), DeviceBuffer(2, 8, Device("hip", 0)), runtime=runtime)


def test_copy_device_to_device_rejects_cross_device() -> None:
    runtime = FakeDeviceRuntime()
    dst = DeviceBuffer(ptr=1, nbytes=8, device=Device("hip", 0))
    src = DeviceBuffer(ptr=2, nbytes=8, device=Device("hip", 1))
    with pytest.raises(ValueError):
        copy_device_to_device(dst, src, runtime=runtime)


def test_copy_device_to_device_validates_sizes_and_selects_device() -> None:
    runtime = FakeDeviceRuntime(current=0)
    dst = DeviceBuffer(ptr=1, nbytes=8, device=Device("hip", 1))
    src = DeviceBuffer(ptr=2, nbytes=8, device=Device("hip", 1))
    copy_device_to_device(dst, src, runtime=runtime)
    assert runtime.calls == [("memcpy", (1, 2, 8, int(MemcpyKind.DEVICE_TO_DEVICE), 1))]
    assert runtime.get_device() == 0
    with pytest.raises(ValueError):
        copy_device_to_device(dst, src, nbytes=9, runtime=runtime)


def test_device_arena_tags_owner_and_views() -> None:
    runtime = FakeDeviceRuntime(current=0)
    arena = DeviceMemoryArena.create(8192, runtime=runtime, device=1)
    assert arena.device == Device("hip", 1)
    view = arena.allocate(1024)
    assert view.device == Device("hip", 1)
    assert arena.owns(view)
    arena.release(view)
    arena.close()
    assert arena.closed
    assert runtime.get_device() == 0


def test_device_buffer_rejects_non_hip_device() -> None:
    with pytest.raises(ValueError):
        DeviceBuffer(ptr=1, nbytes=8, device=Device("cuda", 0))


# -- Packet 1 ownership audit fence -----------------------------------------


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_kernel_launch_surface_stays_visible_to_the_audit() -> None:
    """Kernel launches carry no device argument, so the audit counts matter.

    Every HIP kernel host wrapper takes an explicit stream and no device, which
    means the thread's current device decides where the launch lands. This test
    keeps the size of that surface visible so a rank-bound runner cannot quietly
    grow new launch paths that no device binding covers.
    """

    root = _repository_root()
    kernel_root = root / "hipengine" / "kernels"
    launch_modules = set()
    check_definitions = 0
    check_calls = 0
    for path in kernel_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "getattr(library, symbol)" in text or "fn(*arguments)" in text:
            launch_modules.add(path)
        check_definitions += text.count("def _check_launch(")
        check_calls += text.count("_check_launch(") - text.count("def _check_launch(")

    # Recorded Packet 1 audit numbers. They may grow; they must not silently
    # change, and the fence exists so Packet 3 sees the growth.
    assert len(launch_modules) == 51, sorted(str(p.relative_to(root)) for p in launch_modules)
    assert check_definitions == 31
    assert check_calls == 452

    # The wrappers themselves take a stream and no device, which is the reason
    # the thread's current device is the only thing that decides placement.
    source = (kernel_root / "hip_gfx1100" / "attention" / "laguna_kv_attention.hip").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "hipLaunchKernelGGL(" in source
    assert "<<<" in source
    wrapper = source.split("hipengine_laguna_global_head_rmsnorm_rope_write_kv_f32_bf16_spans(", 1)[1]
    signature = wrapper.split(") {", 1)[0]
    assert "hipStream_t stream" in signature
    assert "hipDevice_t" not in signature
    assert "int device" not in signature


def test_distributed_package_binds_devices_through_one_mechanism() -> None:
    """Rank binding must go through ``scoped_current_device`` and nothing else."""

    root = _repository_root()
    package = root / "hipengine" / "distributed"
    binders = set()
    for path in sorted(package.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "scoped_current_device(" in text:
            binders.add(path.name)
        for banned in ("hipSetDevice(", "set_device("):
            assert banned not in text, f"{path.name} selects a device directly with {banned}"

    # ``context.py`` binds a rank through ``RankRuntime.activate``; ``rccl.py``
    # selects the rank's device around communicator calls. Both are the same
    # primitive, which is the point: there is one mechanism, not three.
    assert binders == {"context.py", "rccl.py"}
