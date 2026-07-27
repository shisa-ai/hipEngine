"""Default-off runtime contract for pinned async Laguna argmax readback."""

from __future__ import annotations

import ctypes
import inspect
import mmap
import struct
from types import SimpleNamespace

import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_gguf_runner as runner
from hipengine.runtime.laguna_gguf_runner import (
    LagunaEagerScratch,
    LagunaGGUFResidentSession,
)
from tests._laguna_synthetic import make_laguna_info


class _FakeRuntime:
    def __init__(
        self,
        *,
        fail_malloc_at: int | None = None,
        fail_host_register: bool = False,
    ) -> None:
        self.next_ptr = 0x7A000000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.malloc_calls = 0
        self.fail_malloc_at = fail_malloc_at
        self.fail_host_register = fail_host_register
        self.registered: list[tuple[int, int, int]] = []
        self.unregistered: list[int] = []
        self.host_get_device_pointer_calls = 0
        self.device_payloads: dict[int, bytes] = {}
        self.events: list[tuple[object, ...]] = []

    def malloc(self, nbytes: int) -> int:
        self.malloc_calls += 1
        if self.fail_malloc_at == self.malloc_calls:
            raise MemoryError("synthetic pinned-async device allocation failure")
        ptr = self.next_ptr
        self.next_ptr += max(0x1000, int(nbytes) + 0x100)
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))
        self.allocations.pop(int(ptr), None)

    def host_register(self, ptr: int, nbytes: int, *, flags: int = 0) -> None:
        if self.fail_host_register:
            raise RuntimeError("synthetic pinned-async host-register failure")
        self.registered.append((int(ptr), int(nbytes), int(flags)))
        self.events.append(("host_register", int(ptr), int(nbytes), int(flags)))

    def host_unregister(self, ptr: int) -> None:
        self.unregistered.append(int(ptr))
        self.events.append(("host_unregister", int(ptr)))

    def host_get_device_pointer(self, ptr: int, *, flags: int = 0) -> int:
        self.host_get_device_pointer_calls += 1
        raise AssertionError("pinned async readback must not map host memory")

    def memcpy_async(
        self,
        dst: int,
        src: int,
        nbytes: int,
        kind: HipMemcpyKind,
        stream: int,
    ) -> None:
        assert kind is HipMemcpyKind.DEVICE_TO_HOST
        payload = self.device_payloads[int(src)]
        assert len(payload) >= int(nbytes)
        ctypes.memmove(int(dst), payload, int(nbytes))
        self.events.append(
            ("memcpy_async", int(dst), int(src), int(nbytes), kind, int(stream))
        )

    def stream_synchronize(self, stream: int) -> None:
        self.events.append(("stream_synchronize", int(stream)))

    def device_synchronize(self) -> None:
        self.events.append(("device_synchronize",))


def _config():
    return laguna_gguf_config_from_metadata(make_laguna_info())


def test_pinned_async_capability_is_explicit_default_off_and_fail_closed() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    assert gfx1100.LAGUNA_PINNED_ASYNC_ARGMAX_READBACK is False
    assert not hasattr(gfx1151, "LAGUNA_PINNED_ASYNC_ARGMAX_READBACK")
    resolver = getattr(runner, "resolve_laguna_pinned_async_argmax_readback", None)
    assert resolver is not None
    assert not resolver("hip_gfx1100")
    assert resolver("hip_gfx1100", True)
    assert not resolver("hip_gfx1100", False)
    assert not resolver("hip_gfx1151", True)


def test_pinned_async_owner_uses_two_copies_then_one_exact_final_fence() -> None:
    owner_type = getattr(runner, "LagunaPinnedArgmaxReadback", None)
    assert owner_type is not None
    runtime = _FakeRuntime()
    owner = owner_type.allocate(runtime=runtime)
    argmax_id = DeviceBuffer(0x7B000000, 8)
    argmax_value = DeviceBuffer(0x7B001000, 4)
    runtime.device_payloads[argmax_id.ptr] = struct.pack("<q", 69_452)
    runtime.device_payloads[argmax_value.ptr] = struct.pack("<I", 0x80000000)

    assert owner.host_nbytes == mmap.PAGESIZE
    assert runtime.registered == [(owner.host_ptr, mmap.PAGESIZE, 0)]
    assert runtime.host_get_device_pointer_calls == 0
    runtime.events.clear()
    token_id, value = owner.read(argmax_id, argmax_value, stream=7)
    assert token_id == 69_452
    assert struct.pack("<f", value) == struct.pack("<I", 0x80000000)
    assert runtime.events == [
        (
            "memcpy_async",
            owner.host_ptr,
            argmax_id.ptr,
            8,
            HipMemcpyKind.DEVICE_TO_HOST,
            7,
        ),
        (
            "memcpy_async",
            owner.host_ptr + 8,
            argmax_value.ptr,
            4,
            HipMemcpyKind.DEVICE_TO_HOST,
            7,
        ),
        ("stream_synchronize", 7),
    ]

    runtime.events.clear()
    assert owner.read(argmax_id, argmax_value, stream=0)[0] == 69_452
    assert [event[0] for event in runtime.events] == [
        "memcpy_async",
        "memcpy_async",
        "device_synchronize",
    ]
    owner.free()
    assert owner.closed
    assert runtime.unregistered == [owner.host_ptr]
    owner.free()
    assert runtime.unregistered == [owner.host_ptr]


def test_pinned_async_scratch_preserves_all_device_owners_and_cleans_failures() -> None:
    assert "pinned_async_argmax_readback" in inspect.signature(
        LagunaEagerScratch.allocate
    ).parameters
    runtime = _FakeRuntime()
    scratch = LagunaEagerScratch.allocate(
        _config(),
        pinned_async_argmax_readback=True,
        runtime=runtime,
    )
    owner = scratch.pinned_argmax_readback
    assert owner is not None
    assert scratch.argmax_id in scratch.buffers
    assert scratch.argmax_value in scratch.buffers
    assert len(scratch.buffers) == len(runtime.allocations) == 24
    assert scratch.nbytes == sum(buffer.nbytes for buffer in scratch.buffers)
    assert runtime.registered == [(owner.host_ptr, mmap.PAGESIZE, 0)]

    fallback_runtime = _FakeRuntime()
    fallback = LagunaEagerScratch.allocate(
        _config(),
        pinned_async_argmax_readback=False,
        runtime=fallback_runtime,
    )
    assert fallback.pinned_argmax_readback is None
    assert len(fallback.buffers) == len(fallback_runtime.allocations) == 24
    assert fallback.nbytes == scratch.nbytes

    owned = tuple(buffer.ptr for buffer in scratch.buffers)
    scratch.free(runtime=runtime)
    assert runtime.freed == list(reversed(owned))
    assert runtime.allocations == {}
    assert runtime.unregistered == [owner.host_ptr]
    scratch.free(runtime=runtime)
    assert runtime.unregistered == [owner.host_ptr]
    fallback.free(runtime=fallback_runtime)
    assert fallback_runtime.allocations == {}

    failing_device = _FakeRuntime(fail_malloc_at=17)
    with pytest.raises(MemoryError, match="pinned-async"):
        LagunaEagerScratch.allocate(
            _config(),
            pinned_async_argmax_readback=True,
            runtime=failing_device,
        )
    assert failing_device.allocations == {}
    assert failing_device.registered == []

    failing_register = _FakeRuntime(fail_host_register=True)
    with pytest.raises(RuntimeError, match="host-register"):
        LagunaEagerScratch.allocate(
            _config(),
            pinned_async_argmax_readback=True,
            runtime=failing_register,
        )
    assert failing_register.allocations == {}
    assert failing_register.unregistered == []


def test_common_result_parser_owns_async_and_blocking_fences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = getattr(runner, "_read_laguna_argmax_result", None)
    assert helper is not None

    class _Owner:
        def __init__(self) -> None:
            self.calls: list[tuple[DeviceBuffer, DeviceBuffer, int]] = []

        def read(
            self,
            argmax_id: DeviceBuffer,
            argmax_value: DeviceBuffer,
            *,
            stream: int,
        ) -> tuple[int, float]:
            self.calls.append((argmax_id, argmax_value, int(stream)))
            return 7, -0.0

    owner = _Owner()
    scratch = SimpleNamespace(
        pinned_argmax_readback=owner,
        argmax_id=DeviceBuffer(0x7C000000, 8),
        argmax_value=DeviceBuffer(0x7C001000, 4),
    )
    runtime = _FakeRuntime()
    assert helper(scratch, runtime, stream=9) == (7, -0.0)
    assert owner.calls == [(scratch.argmax_id, scratch.argmax_value, 9)]
    assert runtime.events == []

    scalar_calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "_read_i64",
        lambda buffer, runtime: (scalar_calls.append("id") or 7),
    )
    monkeypatch.setattr(
        runner,
        "_read_f32",
        lambda buffer, runtime: (scalar_calls.append("value") or -0.0),
    )
    scratch.pinned_argmax_readback = None
    assert helper(scratch, runtime, stream=9) == (7, -0.0)
    assert runtime.events == [("stream_synchronize", 9)]
    assert scalar_calls == ["id", "value"]
    runtime.events.clear()
    scalar_calls.clear()
    assert helper(scratch, runtime, stream=0) == (7, -0.0)
    assert runtime.events == [("device_synchronize",)]
    assert scalar_calls == ["id", "value"]

    for method in (
        LagunaGGUFResidentSession._project_rows_last,
        LagunaGGUFResidentSession._project_and_sample,
    ):
        source = inspect.getsource(method)
        assert source.count("_read_laguna_argmax_result") == 1
        assert "_read_i64" not in source
        assert "_read_f32" not in source
        assert "stream_synchronize" not in source
        assert "device_synchronize" not in source


def test_pinned_async_session_and_benchmark_opt_in_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    option = "use_pinned_async_argmax_readback"
    assert option in inspect.signature(LagunaGGUFResidentSession).parameters
    constructor_source = inspect.getsource(LagunaGGUFResidentSession.__init__)
    assert (
        "pinned_async_argmax_readback=self.use_pinned_async_argmax_readback"
        in constructor_source
    )

    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert benchmark._parse_args().enable_pinned_async_argmax_readback is False
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-pinned-async-argmax-readback"],
    )
    assert benchmark._parse_args().enable_pinned_async_argmax_readback is True
    assert option in inspect.getsource(benchmark._session)
    run_source = inspect.getsource(benchmark.run)
    assert option in run_source
    assert '"use_pinned_async_argmax_readback"' in run_source
