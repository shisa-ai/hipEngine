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
        self.sets = []
        self.hipMalloc = FakeFunction(self._malloc)
        self.hipFree = FakeFunction(self._free)
        self.hipMemcpy = FakeFunction(self._memcpy)
        self.hipMemcpyAsync = FakeFunction(self._memcpy_async)
        self.hipMemset = FakeFunction(self._memset)
        self.hipMemsetAsync = FakeFunction(self._memset_async)
        self.hipMemGetInfo = FakeFunction(self._mem_get_info)
        self.hipStreamCreateWithFlags = FakeFunction(self._stream_create_with_flags)
        self.hipStreamDestroy = FakeFunction(lambda stream: 0)
        self.hipStreamSynchronize = FakeFunction(lambda stream: 0)
        self.hipStreamBeginCapture = FakeFunction(lambda stream, mode: 0)
        self.hipStreamEndCapture = FakeFunction(self._stream_end_capture)
        self.hipGraphInstantiate = FakeFunction(self._graph_instantiate)
        self.hipGraphLaunch = FakeFunction(lambda graph_exec, stream: 0)
        self.hipGraphExecDestroy = FakeFunction(lambda graph_exec: 0)
        self.hipGraphDestroy = FakeFunction(lambda graph: 0)
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

    def _memcpy_async(self, dst, src, nbytes, kind, stream):
        self.copied.append((dst.value, src.value, nbytes.value, kind, stream.value))
        return 0

    def _memset(self, dst, value, nbytes):
        self.sets.append((dst.value, value.value, nbytes.value))
        return 0

    def _memset_async(self, dst, value, nbytes, stream):
        self.sets.append((dst.value, value.value, nbytes.value, stream.value))
        return 0

    def _mem_get_info(self, free_bytes, total_bytes):
        free_bytes._obj.value = 0x9000
        total_bytes._obj.value = 0xA000
        return 0

    def _stream_create_with_flags(self, out_stream, flags):
        out_stream._obj.value = 0x5000 + int(flags.value)
        return 0

    def _stream_end_capture(self, stream, out_graph):
        out_graph._obj.value = 0x6000
        return 0

    def _graph_instantiate(self, out_exec, graph, error_node, log_buffer, buffer_size):
        out_exec._obj.value = 0x7000
        return 0


def setup_function() -> None:
    reset_default_runtime_for_tests()


def test_importing_runtime_module_does_not_load_default_runtime() -> None:
    assert not is_default_runtime_loaded()


def test_fake_runtime_malloc_free_memcpy_stream_and_graph_helpers() -> None:
    lib = FakeHipLibrary()
    runtime = HipRuntime(lib)  # type: ignore[arg-type]
    runtime._configure()

    ptr = runtime.malloc(16)
    runtime.memcpy(ptr, 0x2000, 16, HipMemcpyKind.HOST_TO_DEVICE)
    runtime.memset(ptr, 0, 16)
    free_bytes, total_bytes = runtime.mem_get_info()
    stream = runtime.stream_create()
    runtime.memcpy_async(ptr, 0x3000, 8, HipMemcpyKind.DEVICE_TO_DEVICE, stream)
    runtime.memset_async(ptr, 0xAB, 8, stream)
    runtime.stream_begin_capture(stream)
    graph = runtime.stream_end_capture(stream)
    graph_exec = runtime.graph_instantiate(graph)
    runtime.graph_launch(graph_exec, stream)
    runtime.stream_synchronize(stream)
    runtime.graph_exec_destroy(graph_exec)
    runtime.graph_destroy(graph)
    runtime.stream_destroy(stream)
    runtime.device_synchronize()
    runtime.free(ptr)

    assert ptr == 0x1000
    assert stream == 0x5001
    assert graph == 0x6000
    assert graph_exec == 0x7000
    assert (free_bytes, total_bytes) == (0x9000, 0xA000)
    assert lib.copied == [
        (0x1000, 0x2000, 16, int(HipMemcpyKind.HOST_TO_DEVICE)),
        (0x1000, 0x3000, 8, int(HipMemcpyKind.DEVICE_TO_DEVICE), stream),
    ]
    assert lib.sets == [(0x1000, 0, 16), (0x1000, 0xAB, 8, stream)]
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
