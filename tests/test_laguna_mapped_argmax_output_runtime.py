"""Default-off runtime contract for mapped-host Laguna argmax output."""

from __future__ import annotations

import ctypes
import inspect
import mmap
import struct
from types import SimpleNamespace

import pytest

from hipengine.core.hip import HIP_HOST_REGISTER_MAPPED
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
        fail_host_get: bool = False,
    ) -> None:
        self.next_ptr = 0x6A000000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.malloc_calls = 0
        self.fail_malloc_at = fail_malloc_at
        self.fail_host_register = fail_host_register
        self.fail_host_get = fail_host_get
        self.registered: list[tuple[int, int, int]] = []
        self.unregistered: list[int] = []
        self.device_offset = 0x100000

    def malloc(self, nbytes: int) -> int:
        self.malloc_calls += 1
        if self.fail_malloc_at == self.malloc_calls:
            raise MemoryError("synthetic mapped-argmax device allocation failure")
        ptr = self.next_ptr
        self.next_ptr += max(0x1000, int(nbytes) + 0x100)
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))
        self.allocations.pop(int(ptr), None)

    def host_register(self, ptr: int, nbytes: int, *, flags: int = 0) -> None:
        if self.fail_host_register:
            raise RuntimeError("synthetic mapped-argmax host-register failure")
        self.registered.append((int(ptr), int(nbytes), int(flags)))

    def host_get_device_pointer(self, ptr: int, *, flags: int = 0) -> int:
        assert flags == 0
        if self.fail_host_get:
            raise RuntimeError("synthetic mapped-argmax device-pointer failure")
        return int(ptr) + self.device_offset

    def host_unregister(self, ptr: int) -> None:
        self.unregistered.append(int(ptr))


def _config():
    return laguna_gguf_config_from_metadata(make_laguna_info())


def test_mapped_argmax_capability_is_explicit_default_off_and_fail_closed() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    assert gfx1100.LAGUNA_MAPPED_ARGMAX_OUTPUT is False
    assert not hasattr(gfx1151, "LAGUNA_MAPPED_ARGMAX_OUTPUT")
    assert not runner.resolve_laguna_mapped_argmax_output("hip_gfx1100")
    assert runner.resolve_laguna_mapped_argmax_output("hip_gfx1100", True)
    assert not runner.resolve_laguna_mapped_argmax_output("hip_gfx1100", False)
    assert not runner.resolve_laguna_mapped_argmax_output("hip_gfx1151", True)


def test_mapped_argmax_owner_exposes_exact_host_and_device_views() -> None:
    owner_type = getattr(runner, "LagunaMappedArgmaxOutput", None)
    assert owner_type is not None
    runtime = _FakeRuntime()
    owner = owner_type.allocate(runtime=runtime)

    assert owner.host_nbytes == mmap.PAGESIZE
    assert owner.argmax_id == DeviceBuffer(owner.device_ptr, 8)
    assert owner.argmax_value == DeviceBuffer(owner.device_ptr + 8, 4)
    assert runtime.registered == [
        (owner.host_ptr, mmap.PAGESIZE, HIP_HOST_REGISTER_MAPPED),
    ]
    ctypes.c_int64.from_address(owner.host_ptr).value = 69_452
    ctypes.c_uint32.from_address(owner.host_ptr + 8).value = 0x80000000
    token_id, value = owner.read()
    assert token_id == 69_452
    assert struct.pack("<f", value) == struct.pack("<I", 0x80000000)

    owner.free()
    assert owner.closed
    assert runtime.unregistered == [owner.host_ptr]
    owner.free()
    assert runtime.unregistered == [owner.host_ptr]


def test_mapped_argmax_scratch_owns_mapping_not_device_views_and_cleans_failures() -> None:
    assert "mapped_argmax_output" in inspect.signature(
        LagunaEagerScratch.allocate
    ).parameters
    runtime = _FakeRuntime()
    scratch = LagunaEagerScratch.allocate(
        _config(),
        mapped_argmax_output=True,
        runtime=runtime,
    )
    owner = scratch.mapped_argmax_output
    assert owner is not None
    assert scratch.argmax_id == owner.argmax_id
    assert scratch.argmax_value == owner.argmax_value
    assert scratch.argmax_id not in scratch.buffers
    assert scratch.argmax_value not in scratch.buffers
    assert len(scratch.buffers) == len(runtime.allocations) == 22
    assert scratch.nbytes == sum(buffer.nbytes for buffer in scratch.buffers)

    fallback_runtime = _FakeRuntime()
    fallback = LagunaEagerScratch.allocate(
        _config(),
        mapped_argmax_output=False,
        runtime=fallback_runtime,
    )
    assert fallback.mapped_argmax_output is None
    assert fallback.argmax_id in fallback.buffers
    assert fallback.argmax_value in fallback.buffers
    assert len(fallback.buffers) == len(fallback_runtime.allocations) == 24
    assert fallback.nbytes == scratch.nbytes + 12

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
    with pytest.raises(MemoryError, match="mapped-argmax"):
        LagunaEagerScratch.allocate(
            _config(),
            mapped_argmax_output=True,
            runtime=failing_device,
        )
    assert failing_device.allocations == {}
    assert failing_device.registered == []

    failing_register = _FakeRuntime(fail_host_register=True)
    with pytest.raises(RuntimeError, match="host-register"):
        LagunaEagerScratch.allocate(
            _config(),
            mapped_argmax_output=True,
            runtime=failing_register,
        )
    assert failing_register.allocations == {}
    assert failing_register.unregistered == []

    failing_pointer = _FakeRuntime(fail_host_get=True)
    with pytest.raises(RuntimeError, match="device-pointer"):
        LagunaEagerScratch.allocate(
            _config(),
            mapped_argmax_output=True,
            runtime=failing_pointer,
        )
    assert failing_pointer.allocations == {}
    assert len(failing_pointer.registered) == 1
    assert failing_pointer.unregistered == [failing_pointer.registered[0][0]]


def test_mapped_argmax_result_parser_and_both_scalar_sites_keep_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = getattr(runner, "_read_laguna_argmax_result", None)
    assert helper is not None

    class _Owner:
        def __init__(self) -> None:
            self.calls = 0

        def read(self) -> tuple[int, float]:
            self.calls += 1
            return 7, -0.0

    owner = _Owner()
    scratch = SimpleNamespace(
        mapped_argmax_output=owner,
        argmax_id=DeviceBuffer(0x70000000, 8),
        argmax_value=DeviceBuffer(0x70001000, 4),
    )
    assert helper(scratch, object()) == (7, -0.0)
    assert owner.calls == 1

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
    scratch.mapped_argmax_output = None
    assert helper(scratch, object()) == (7, -0.0)
    assert scalar_calls == ["id", "value"]

    for method in (
        LagunaGGUFResidentSession._project_rows_last,
        LagunaGGUFResidentSession._project_and_sample,
    ):
        source = inspect.getsource(method)
        assert source.count("_read_laguna_argmax_result") == 1
        assert "_read_i64" not in source
        assert "_read_f32" not in source


def test_mapped_argmax_session_and_benchmark_opt_in_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    option = "use_mapped_argmax_output"
    assert option in inspect.signature(LagunaGGUFResidentSession).parameters
    constructor_source = inspect.getsource(LagunaGGUFResidentSession.__init__)
    assert "mapped_argmax_output=self.use_mapped_argmax_output" in constructor_source

    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert benchmark._parse_args().enable_mapped_argmax_output is False
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-mapped-argmax-output"],
    )
    assert benchmark._parse_args().enable_mapped_argmax_output is True
    assert option in inspect.getsource(benchmark._session)
    assert option in inspect.getsource(benchmark.run)
