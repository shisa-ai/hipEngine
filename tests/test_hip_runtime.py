from __future__ import annotations

import ctypes

import pytest

from hipengine.core.hip import (
    HipError,
    HipMemcpyKind,
    HipRuntime,
    is_default_runtime_loaded,
    reset_default_runtime_for_tests,
)
from hipengine.core.memory import DeviceBuffer, host_buffer_ptr


class FakeFunction:
    def __init__(self, func):
        self.func = func
        self.argtypes = None
        self.restype = None
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.func(*args)


class FakeHipLibrary:
    def __init__(self):
        self.next_ptr = 0x1000
        self.freed = []
        self.copied = []
        self.hipMalloc = FakeFunction(self._malloc)
        self.hipFree = FakeFunction(self._free)
        self.hipMemcpy = FakeFunction(self._memcpy)
        self.hipDeviceSynchronize = FakeFunction(lambda: 0)
        self.hipGetErrorString = FakeFunction(lambda code: b"fake hip error")

    def _malloc(self, out_ptr, nbytes):
        out_ptr._obj.value = self.next_ptr
        self.next_ptr += int(nbytes.value)
        return 0

    def _free(self, ptr):
        self.freed.append(ptr.value)
        return 0

    def _memcpy(self, dst, src, nbytes, kind):
        self.copied.append((dst.value, src.value, nbytes.value, kind))
        return 0


def setup_function() -> None:
    reset_default_runtime_for_tests()


def test_importing_runtime_module_does_not_load_default_runtime() -> None:
    assert not is_default_runtime_loaded()


def test_fake_runtime_malloc_free_memcpy_and_sync() -> None:
    lib = FakeHipLibrary()
    runtime = HipRuntime(lib)  # type: ignore[arg-type]
    runtime._configure()

    ptr = runtime.malloc(16)
    runtime.memcpy(ptr, 0x2000, 16, HipMemcpyKind.HOST_TO_DEVICE)
    runtime.device_synchronize()
    runtime.free(ptr)

    assert ptr == 0x1000
    assert lib.copied == [(0x1000, 0x2000, 16, int(HipMemcpyKind.HOST_TO_DEVICE))]
    assert lib.freed == [0x1000]


def test_runtime_error_uses_error_string() -> None:
    lib = FakeHipLibrary()
    runtime = HipRuntime(lib)  # type: ignore[arg-type]
    runtime._configure()

    with pytest.raises(HipError, match="fake hip error"):
        runtime.check(7)


def test_device_buffer_and_host_pointer_helpers() -> None:
    buffer = DeviceBuffer(ptr=1234, nbytes=16)
    host = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)

    assert buffer.ptr == 1234
    assert host_buffer_ptr(host) == ctypes.addressof(host)
