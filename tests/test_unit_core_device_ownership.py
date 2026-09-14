"""CPU-only contracts for explicit HIP device ownership.

These tests use a fake device runtime so they run on hosts without ROCm. Real
peer-access and copy behavior is covered by the guarded GPU tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ctypes

import pytest

from hipengine.core.device import Device, scoped_current_device
from hipengine.core.hip import HipRuntime, decode_hip_uuid, format_hip_uuid
from hipengine.core.memory import (
    DeviceBuffer,
    DeviceMemoryArena,
    copy_device_to_host,
    copy_device_to_device,
    copy_host_array_to_device,
    copy_host_to_device,
    free,
    host_buffer_ptr,
    malloc,
)
from hipengine.core.runtime import MemcpyKind


class FakeDeviceRuntime:
    """Minimal DeviceRuntime that records device selection and copy calls.

    It stands in for a HIP runtime, so it declares the same device kind the real
    one does; attribution is never guessed from a missing declaration.
    """

    device_kind = "hip"

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


def test_device_buffer_rejects_a_host_device() -> None:
    """Only a device kind may own a device pointer.

    This test previously asserted that a CUDA label was rejected. That encoded
    the HIP-only assumption this module no longer makes: ``CudaRuntime``
    allocates through the same helpers, so its buffers carry a CUDA label.
    """

    with pytest.raises(ValueError):
        DeviceBuffer(ptr=1, nbytes=8, device=Device("cpu", 0))
    assert DeviceBuffer(ptr=1, nbytes=8, device=Device("cuda", 0)).device == Device("cuda", 0)


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


def test_free_accepts_unattributed_and_duck_typed_buffers() -> None:
    """``free`` must not require the ``device`` attribute.

    ``DeviceBuffer`` gained ``device`` for multi-rank allocation, but callers and
    tests also pass buffers that expose only ``ptr``/``nbytes`` - either an
    unattributed ``DeviceBuffer`` or a stand-in object. Those predate attribution
    and must keep the previous current-device semantics instead of raising
    ``AttributeError``.
    """

    class _StandIn:
        def __init__(self, ptr: int, nbytes: int) -> None:
            self.ptr = ptr
            self.nbytes = nbytes

    runtime = FakeDeviceRuntime(current=0)
    free(DeviceBuffer(ptr=4096, nbytes=64), runtime=runtime)
    free(_StandIn(ptr=8192, nbytes=64), runtime=runtime)
    free(DeviceBuffer(ptr=12288, nbytes=64, device=Device("hip", 1)), runtime=runtime)

    # Each free records the device that was current at the call: the first two
    # unattributed buffers free on the ambient device, the third on its own.
    assert runtime.calls == [
        ("free", (4096, 0)),
        ("free", (8192, 0)),
        ("free", (12288, 1)),
    ]
    # The attributed buffer selected its own device and restored the previous one.
    assert runtime.get_device() == 0

    with pytest.raises(ValueError):
        copy_device_to_device(
            _StandIn(ptr=1, nbytes=8), DeviceBuffer(ptr=2, nbytes=8, device=Device("hip", 0)),
            runtime=runtime,
        )


def test_every_buffer_entry_point_handles_both_attribution_states() -> None:
    """One table over the whole buffer-taking surface of ``core.memory``.

    The ``free`` regression happened because a single function grew a
    ``buffer.device`` read and only a caller's test noticed. This covers the
    surface systematically: every entry point that takes a buffer must accept an
    unattributed one (old current-device semantics), and every entry point that
    touches device memory must select the owning device when the buffer is
    attributed. ``host_buffer_ptr`` is listed as device-agnostic because it only
    returns an address.
    """

    import inspect

    from hipengine.core import memory as memory_module

    def call_free(buffer, runtime):
        free(buffer, runtime=runtime)

    def call_h2d(buffer, runtime):
        copy_host_to_device(buffer, 0x9000, 8, runtime=runtime)

    def call_h2d_array(buffer, runtime):
        copy_host_array_to_device(buffer, np.zeros(2, dtype=np.float32), runtime=runtime)

    def call_d2h(buffer, runtime):
        copy_device_to_host(0x9000, buffer, 8, runtime=runtime)

    def call_d2d(buffer, runtime):
        copy_device_to_device(
            DeviceBuffer(ptr=0x7000, nbytes=8, device=Device("hip", 1)),
            buffer,
            runtime=runtime,
        )

    # entry point -> (call, selects the owning device?)
    cases = {
        "free": (call_free, True),
        "copy_host_to_device": (call_h2d, True),
        "copy_host_array_to_device": (call_h2d_array, True),
        "copy_device_to_host": (call_d2h, True),
    }
    # A device-to-device copy cannot infer a device from two unattributed
    # buffers, so it refuses instead of guessing.
    attribution_required = {"copy_device_to_device": call_d2d}
    # Host-side helpers whose parameter happens to be named ``buffer``: these
    # take a ctypes array and never touch device memory.
    host_side = {"host_buffer_ptr"}

    # Every public function that takes a buffer by name is in the table above.
    taken = set()
    for name, function in vars(memory_module).items():
        if name.startswith("_") or not inspect.isfunction(function):
            continue
        parameters = set(inspect.signature(function).parameters)
        if parameters & {"buffer", "dst", "src"}:
            taken.add(name)
    expected = set(cases) | set(attribution_required) | host_side
    assert taken == expected, f"buffer-taking entry points changed: {sorted(taken ^ expected)}"

    for name, (call, selects_device) in cases.items():
        unattributed = FakeDeviceRuntime(current=0)
        call(DeviceBuffer(ptr=0x1000, nbytes=8), unattributed)
        assert unattributed.get_device() == 0, f"{name} left the current device changed"

        attributed = FakeDeviceRuntime(current=0)
        call(DeviceBuffer(ptr=0x2000, nbytes=8, device=Device("hip", 1)), attributed)
        assert attributed.get_device() == 0, f"{name} left the current device changed"
        used = {entry[1][-1] for entry in attributed.calls if entry[1]}
        if selects_device:
            assert used == {1}, f"{name} did not select the owning device: {attributed.calls}"
    for name, call in attribution_required.items():
        unattributed = FakeDeviceRuntime(current=0)
        with pytest.raises(ValueError):
            call(DeviceBuffer(ptr=0x1000, nbytes=8), unattributed)
        attributed = FakeDeviceRuntime(current=0)
        call(DeviceBuffer(ptr=0x2000, nbytes=8, device=Device("hip", 1)), attributed)
        assert attributed.get_device() == 0, f"{name} left the current device changed"
        assert {entry[1][-1] for entry in attributed.calls if entry[1]} == {1}, name


def test_buffer_entry_points_tolerate_duck_typed_buffers() -> None:
    """A stand-in exposing only ptr/nbytes must not raise AttributeError."""

    from hipengine.core.memory import copy_host_to_device, copy_device_to_host

    class _StandIn:
        def __init__(self, ptr: int, nbytes: int) -> None:
            self.ptr = ptr
            self.nbytes = nbytes

    runtime = FakeDeviceRuntime(current=0)
    stand_in = _StandIn(ptr=0x3000, nbytes=8)
    free(stand_in, runtime=runtime)
    copy_host_to_device(stand_in, 0x9000, 8, runtime=runtime)
    copy_device_to_host(0x9000, stand_in, 8, runtime=runtime)
    assert [entry[0] for entry in runtime.calls] == ["free", "memcpy", "memcpy"]
    assert runtime.get_device() == 0


# -- backend-neutral attribution ---------------------------------------------


def test_cuda_runtime_buffers_are_labelled_cuda_not_hip() -> None:
    """Shared allocation helpers must not hardcode a HIP device label.

    ``CudaRuntime`` exposes ``get_device`` and uses the same ``malloc``/``free``
    helpers, so a hardcoded ``Device("hip", ...)`` mislabels every CUDA buffer -
    and ``DeviceBuffer`` then refuses the correct CUDA label as well.
    """

    class CudaLikeRuntime(FakeDeviceRuntime):
        device_kind = "cuda"

    runtime = CudaLikeRuntime(current=0)
    buffer = malloc(64, runtime=runtime)
    assert buffer.device == Device("cuda", 0)

    free(buffer, runtime=runtime)
    assert runtime.get_device() == 0

    explicit = malloc(64, runtime=runtime, device=Device("cuda", 0))
    assert explicit.device == Device("cuda", 0)


def test_device_buffer_accepts_a_cuda_label_and_refuses_a_cpu_one() -> None:
    buffer = DeviceBuffer(ptr=1, nbytes=8, device=Device("cuda", 1))
    assert buffer.device == Device("cuda", 1)
    with pytest.raises(ValueError):
        DeviceBuffer(ptr=1, nbytes=8, device=Device("cpu", 0))


def test_runtime_without_a_declared_kind_stays_unattributed() -> None:
    """An unknown runtime is left unattributed rather than labelled by guess."""

    class AnonymousRuntime:
        """A runtime with no declared device kind, and no inheritance to hide one."""

        def __init__(self, current: int = 0) -> None:
            self._current = current

        def get_device(self) -> int:
            return self._current

        def set_device(self, device: int) -> None:
            self._current = int(device)

        def malloc(self, nbytes: int) -> int:
            return 0x2000

        def free(self, ptr: int) -> None:
            return None

    assert not hasattr(AnonymousRuntime, "device_kind")
    runtime = AnonymousRuntime(current=1)
    buffer = malloc(64, runtime=runtime)
    assert buffer.device is None
    free(buffer, runtime=runtime)


def test_device_uuid_survives_embedded_nul_bytes() -> None:
    """A ``hipUUID`` is 16 binary bytes, not a NUL-terminated C string.

    ``HipUuid.bytes`` is a ``c_char`` array, so reading it as a Python value
    truncates at the first zero byte. A real device UUID commonly starts with
    one, which made ``device_get_uuid()`` raise instead of returning an identity.
    """

    class UuidLibrary:
        def hipGetDevice(self, out) -> int:
            out._obj.value = 0
            return 0

        def hipDeviceGetUuid(self, out, device) -> int:
            target = getattr(out, "_obj", out)
            ctypes.memmove(ctypes.addressof(target), bytes(range(16)), 16)
            return 0

        def hipGetErrorString(self, code) -> bytes:
            return b"fake hip error"

    runtime = HipRuntime(UuidLibrary())
    raw = runtime.device_get_uuid_bytes()
    assert len(raw) == 16
    assert raw == bytes(range(16))
    assert runtime.device_get_uuid() == "00010203-0405-0607-0809-0a0b0c0d0e0f"


def test_device_uuid_text_form_is_not_double_encoded() -> None:
    """ROCm writes the ASCII unique ID; hex-encoding it invents a fake UUID.

    The raw bytes are ``e282895b62c2b295`` - the same value sysfs and rocm-smi
    report. Hex-encoding those ASCII characters produced
    ``65323832-3839-3562-3632-633262323935``, which matches no other tool.
    """

    raw = b"e282895b62c2b295"
    assert decode_hip_uuid(raw) == "e282895b62c2b295"
    assert format_hip_uuid(raw) == "65323832-3839-3562-3632-633262323935"


def test_device_uuid_text_shorter_than_the_field_drops_nul_padding() -> None:
    """A shorter unique ID is NUL-padded; the padding is not part of the ID."""

    assert decode_hip_uuid(b"abc123" + b"\x00" * 10) == "abc123"


def test_device_info_reports_the_real_device_identity() -> None:
    """The identity snapshot must agree with what the runtime wrote."""

    class UuidLibrary:
        def hipGetDevice(self, out) -> int:
            out._obj.value = 0
            return 0

        def hipDeviceGetUuid(self, out, device) -> int:
            target = getattr(out, "_obj", out)
            ctypes.memmove(ctypes.addressof(target), b"e282895b62c2b295", 16)
            return 0

        def hipDeviceGetName(self, out, length, device) -> int:
            out.value = b"AMD Radeon Pro W7900"
            return 0

        def hipDeviceGetPCIBusId(self, out, length, device) -> int:
            out.value = b"0000:0d:00.0"
            return 0

        def hipGetErrorString(self, code) -> bytes:
            return b"fake hip error"

    info = HipRuntime(UuidLibrary()).device_info(0)
    assert info.uuid == "e282895b62c2b295"
    assert info.uuid_hex == "65323832383935623632633262323935"
    assert info.name == "AMD Radeon Pro W7900"
    assert info.pci_bus_id == "0000:0d:00.0"


def test_device_uuid_all_zero_bytes_are_a_valid_identity() -> None:
    class UuidLibrary:
        def hipGetDevice(self, out) -> int:
            out._obj.value = 0
            return 0

        def hipDeviceGetUuid(self, out, device) -> int:
            target = getattr(out, "_obj", out)
            ctypes.memmove(ctypes.addressof(target), b"\x00" * 16, 16)
            return 0

        def hipGetErrorString(self, code) -> bytes:
            return b"fake hip error"

    runtime = HipRuntime(UuidLibrary())
    assert runtime.device_get_uuid() == "00000000-0000-0000-0000-000000000000"
