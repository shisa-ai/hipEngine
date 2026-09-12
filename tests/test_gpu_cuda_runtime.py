from __future__ import annotations

import pytest

from hipengine.core.cuda import (
    CudaError,
    CudaMemcpyKind,
    CudaRuntime,
    is_default_cuda_runtime_loaded,
    reset_default_cuda_runtime_for_tests,
)
from hipengine.core.device import Device
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.runtime import MemcpyKind
from hipengine.runtime.workspace import RuntimeWorkspace


class FakeFunction:
    def __init__(self, func):
        self.func = func
        self.argtypes = None
        self.restype = None
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.func(*args)


class FakeCudaLibrary:
    def __init__(self):
        self.next_ptr = 0x1000
        self.current_device = 0
        self.freed: list[int] = []
        self.copied: list[tuple[int, ...]] = []
        self.sets: list[tuple[int, ...]] = []
        self.waits: list[tuple[int, ...]] = []
        self.records: list[tuple[int, ...]] = []
        self.cudaSetDevice = FakeFunction(self._set_device)
        self.cudaGetDevice = FakeFunction(self._get_device)
        self.cudaGetDeviceCount = FakeFunction(self._get_device_count)
        self.cudaMalloc = FakeFunction(self._malloc)
        self.cudaFree = FakeFunction(self._free)
        self.cudaMemcpy = FakeFunction(self._memcpy)
        self.cudaMemcpyAsync = FakeFunction(self._memcpy_async)
        self.cudaMemset = FakeFunction(self._memset)
        self.cudaMemsetAsync = FakeFunction(self._memset_async)
        self.cudaMemGetInfo = FakeFunction(self._mem_get_info)
        self.cudaStreamCreateWithFlags = FakeFunction(self._stream_create_with_flags)
        self.cudaStreamDestroy = FakeFunction(lambda stream: 0)
        self.cudaStreamSynchronize = FakeFunction(lambda stream: 0)
        self.cudaStreamWaitEvent = FakeFunction(self._stream_wait_event)
        self.cudaStreamBeginCapture = FakeFunction(lambda stream, mode: 0)
        self.cudaStreamEndCapture = FakeFunction(self._stream_end_capture)
        self.cudaGraphInstantiate = FakeFunction(self._graph_instantiate)
        self.cudaGraphLaunch = FakeFunction(lambda graph_exec, stream: 0)
        self.cudaGraphExecDestroy = FakeFunction(lambda graph_exec: 0)
        self.cudaGraphDestroy = FakeFunction(lambda graph: 0)
        self.cudaEventCreateWithFlags = FakeFunction(self._event_create_with_flags)
        self.cudaEventDestroy = FakeFunction(lambda event: 0)
        self.cudaEventRecord = FakeFunction(self._event_record)
        self.cudaEventSynchronize = FakeFunction(lambda event: 0)
        self.cudaEventElapsedTime = FakeFunction(self._event_elapsed_time)
        self.cudaDeviceSynchronize = FakeFunction(lambda: 0)
        self.cudaGetErrorString = FakeFunction(lambda code: b"fake cuda error")

    def _set_device(self, device):
        self.current_device = int(device.value)
        return 0

    def _get_device(self, out_device):
        out_device._obj.value = self.current_device
        return 0

    def _get_device_count(self, out_count):
        out_count._obj.value = 2
        return 0

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

    def _stream_wait_event(self, stream, event, flags):
        self.waits.append((stream.value, event.value, flags.value))
        return 0

    def _stream_end_capture(self, stream, out_graph):
        out_graph._obj.value = 0x6000
        return 0

    def _graph_instantiate(self, out_exec, graph, flags):
        out_exec._obj.value = 0x7000 + int(flags.value)
        return 0

    def _event_create_with_flags(self, out_event, flags):
        out_event._obj.value = 0x8000 + int(flags.value)
        return 0

    def _event_record(self, event, stream):
        self.records.append((event.value, stream.value))
        return 0

    def _event_elapsed_time(self, out_elapsed, start, stop):
        out_elapsed._obj.value = 1.25
        return 0


def setup_function() -> None:
    reset_default_cuda_runtime_for_tests()


def test_importing_runtime_module_does_not_load_default_cuda_runtime() -> None:
    assert not is_default_cuda_runtime_loaded()


def test_fake_cuda_runtime_covers_moonshine_lifecycle_operations() -> None:
    lib = FakeCudaLibrary()
    runtime = CudaRuntime(lib)  # type: ignore[arg-type]
    runtime._configure()

    assert runtime.device_count() == 2
    runtime.set_device(1)
    assert runtime.get_device() == 1
    ptr = runtime.malloc(16)
    runtime.memcpy(ptr, 0x2000, 16, CudaMemcpyKind.HOST_TO_DEVICE)
    runtime.memset(ptr, 0, 16)
    free_bytes, total_bytes = runtime.mem_get_info()
    stream = runtime.stream_create()
    runtime.memcpy_async(ptr, 0x3000, 8, CudaMemcpyKind.DEVICE_TO_DEVICE, stream)
    runtime.memset_async(ptr, 0xAB, 8, stream)
    event = runtime.event_create(flags=2)
    runtime.event_record(event, stream)
    runtime.stream_wait_event(stream, event, flags=3)
    runtime.event_synchronize(event)
    assert runtime.event_elapsed_time_ms(event, event) == pytest.approx(1.25)
    runtime.stream_begin_capture(stream)
    graph = runtime.stream_end_capture(stream)
    graph_exec = runtime.graph_instantiate(graph, flags=4)
    runtime.graph_launch(graph_exec, stream)
    runtime.stream_synchronize(stream)
    runtime.graph_exec_destroy(graph_exec)
    runtime.graph_destroy(graph)
    runtime.event_destroy(event)
    runtime.stream_destroy(stream)
    runtime.device_synchronize()
    runtime.free(ptr)

    assert ptr == 0x1000
    assert stream == 0x5001
    assert graph == 0x6000
    assert graph_exec == 0x7004
    assert event == 0x8002
    assert (free_bytes, total_bytes) == (0x9000, 0xA000)
    assert lib.copied == [
        (0x1000, 0x2000, 16, int(CudaMemcpyKind.HOST_TO_DEVICE)),
        (0x1000, 0x3000, 8, int(CudaMemcpyKind.DEVICE_TO_DEVICE), stream),
    ]
    assert lib.sets == [(0x1000, 0, 16), (0x1000, 0xAB, 8, stream)]
    assert lib.records == [(event, stream)]
    assert lib.waits == [(stream, event, 3)]
    assert lib.freed == [0x1000]


def test_cuda_runtime_rejects_negative_sizes_and_uses_error_string() -> None:
    lib = FakeCudaLibrary()
    runtime = CudaRuntime(lib)  # type: ignore[arg-type]
    runtime._configure()

    with pytest.raises(ValueError, match="non-negative"):
        runtime.malloc(-1)
    with pytest.raises(ValueError, match="non-negative"):
        runtime.memcpy(0, 0, -1, CudaMemcpyKind.DEFAULT)
    with pytest.raises(CudaError, match="fake cuda error"):
        runtime.check(7)


def test_memory_kinds_and_workspace_are_backend_neutral() -> None:
    assert CudaMemcpyKind is MemcpyKind
    assert HipMemcpyKind is MemcpyKind
    lib = FakeCudaLibrary()
    runtime = CudaRuntime(lib)  # type: ignore[arg-type]
    runtime._configure()
    workspace = RuntimeWorkspace(device=Device("cuda", 0), runtime=runtime)

    tensor = workspace.reserve_tensor("hidden", (8, 52), "fp16")
    try:
        assert tensor.device == Device("cuda", 0)
        assert tensor.numel * tensor.dtype.itemsize == 8 * 52 * 2
    finally:
        workspace.free()
    assert lib.freed == [tensor.ptr]
