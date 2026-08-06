"""Lazy ctypes wrapper for the CUDA runtime.

Importing this module does not load ``libcudart`` or initialize CUDA. The shared
library is loaded only when :func:`get_cuda_runtime` or :meth:`CudaRuntime.load`
is called.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from typing import Final

from hipengine.core.runtime import MemcpyKind

CUDA_SUCCESS: Final[int] = 0
DEFAULT_CUDA_LIBRARY: Final[str] = "libcudart.so.13"


CudaMemcpyKind = MemcpyKind


class CudaError(RuntimeError):
    """Raised when a CUDA runtime call returns a non-success code."""

    def __init__(self, code: int, message: str):
        self.code = int(code)
        super().__init__(f"CUDA error {self.code}: {message}")


@dataclass
class CudaRuntime:
    """Loaded CUDA runtime library with the Moonshine lifecycle operations."""

    library: ctypes.CDLL

    @classmethod
    def load(cls, path: str | None = None) -> "CudaRuntime":
        runtime = cls(ctypes.CDLL(path or _default_cuda_library()))
        runtime._configure()
        return runtime

    def set_device(self, device: int) -> None:
        self.check(self.library.cudaSetDevice(ctypes.c_int(device)))

    def get_device(self) -> int:
        device = ctypes.c_int()
        self.check(self.library.cudaGetDevice(ctypes.byref(device)))
        return int(device.value)

    def device_count(self) -> int:
        count = ctypes.c_int()
        self.check(self.library.cudaGetDeviceCount(ctypes.byref(count)))
        return int(count.value)

    def malloc(self, nbytes: int) -> int:
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        ptr = ctypes.c_void_p()
        self.check(self.library.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)))
        return 0 if ptr.value is None else int(ptr.value)

    def free(self, ptr: int) -> None:
        self.check(self.library.cudaFree(ctypes.c_void_p(ptr)))

    def memcpy(
        self,
        dst: int,
        src: int,
        nbytes: int,
        kind: CudaMemcpyKind | int,
    ) -> None:
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        self.check(
            self.library.cudaMemcpy(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                ctypes.c_size_t(nbytes),
                int(kind),
            )
        )

    def memcpy_async(
        self,
        dst: int,
        src: int,
        nbytes: int,
        kind: CudaMemcpyKind | int,
        stream: int,
    ) -> None:
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        self.check(
            self.library.cudaMemcpyAsync(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                ctypes.c_size_t(nbytes),
                int(kind),
                ctypes.c_void_p(stream),
            )
        )

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        self.check(
            self.library.cudaMemset(
                ctypes.c_void_p(dst),
                ctypes.c_int(value),
                ctypes.c_size_t(nbytes),
            )
        )

    def memset_async(self, dst: int, value: int, nbytes: int, stream: int) -> None:
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        self.check(
            self.library.cudaMemsetAsync(
                ctypes.c_void_p(dst),
                ctypes.c_int(value),
                ctypes.c_size_t(nbytes),
                ctypes.c_void_p(stream),
            )
        )

    def mem_get_info(self) -> tuple[int, int]:
        free_bytes = ctypes.c_size_t()
        total_bytes = ctypes.c_size_t()
        self.check(
            self.library.cudaMemGetInfo(
                ctypes.byref(free_bytes),
                ctypes.byref(total_bytes),
            )
        )
        return int(free_bytes.value), int(total_bytes.value)

    def stream_create(self, *, nonblocking: bool = True) -> int:
        stream = ctypes.c_void_p()
        flags = 0x01 if nonblocking else 0x00
        self.check(
            self.library.cudaStreamCreateWithFlags(
                ctypes.byref(stream),
                ctypes.c_uint(flags),
            )
        )
        return 0 if stream.value is None else int(stream.value)

    def stream_destroy(self, stream: int) -> None:
        self.check(self.library.cudaStreamDestroy(ctypes.c_void_p(stream)))

    def stream_synchronize(self, stream: int) -> None:
        self.check(self.library.cudaStreamSynchronize(ctypes.c_void_p(stream)))

    def stream_wait_event(self, stream: int, event: int, *, flags: int = 0) -> None:
        self.check(
            self.library.cudaStreamWaitEvent(
                ctypes.c_void_p(stream),
                ctypes.c_void_p(event),
                ctypes.c_uint(flags),
            )
        )

    def stream_begin_capture(self, stream: int, mode: int = 2) -> None:
        self.check(
            self.library.cudaStreamBeginCapture(
                ctypes.c_void_p(stream),
                ctypes.c_int(mode),
            )
        )

    def stream_end_capture(self, stream: int) -> int:
        graph = ctypes.c_void_p()
        self.check(
            self.library.cudaStreamEndCapture(
                ctypes.c_void_p(stream),
                ctypes.byref(graph),
            )
        )
        return 0 if graph.value is None else int(graph.value)

    def graph_instantiate(self, graph: int, *, flags: int = 0) -> int:
        graph_exec = ctypes.c_void_p()
        self.check(
            self.library.cudaGraphInstantiate(
                ctypes.byref(graph_exec),
                ctypes.c_void_p(graph),
                ctypes.c_ulonglong(flags),
            )
        )
        return 0 if graph_exec.value is None else int(graph_exec.value)

    def graph_launch(self, graph_exec: int, stream: int) -> None:
        self.check(
            self.library.cudaGraphLaunch(
                ctypes.c_void_p(graph_exec),
                ctypes.c_void_p(stream),
            )
        )

    def graph_exec_destroy(self, graph_exec: int) -> None:
        self.check(self.library.cudaGraphExecDestroy(ctypes.c_void_p(graph_exec)))

    def graph_destroy(self, graph: int) -> None:
        self.check(self.library.cudaGraphDestroy(ctypes.c_void_p(graph)))

    def event_create(self, *, flags: int = 0) -> int:
        event = ctypes.c_void_p()
        self.check(
            self.library.cudaEventCreateWithFlags(
                ctypes.byref(event),
                ctypes.c_uint(flags),
            )
        )
        return 0 if event.value is None else int(event.value)

    def event_destroy(self, event: int) -> None:
        self.check(self.library.cudaEventDestroy(ctypes.c_void_p(event)))

    def event_record(self, event: int, stream: int = 0) -> None:
        self.check(
            self.library.cudaEventRecord(
                ctypes.c_void_p(event),
                ctypes.c_void_p(stream),
            )
        )

    def event_synchronize(self, event: int) -> None:
        self.check(self.library.cudaEventSynchronize(ctypes.c_void_p(event)))

    def event_elapsed_time_ms(self, start: int, stop: int) -> float:
        elapsed = ctypes.c_float()
        self.check(
            self.library.cudaEventElapsedTime(
                ctypes.byref(elapsed),
                ctypes.c_void_p(start),
                ctypes.c_void_p(stop),
            )
        )
        return float(elapsed.value)

    def device_synchronize(self) -> None:
        self.check(self.library.cudaDeviceSynchronize())

    def error_string(self, code: int) -> str:
        raw = self.library.cudaGetErrorString(int(code))
        if not raw:
            return "<unknown>"
        return raw.decode("utf-8", errors="replace")

    def check(self, code: int) -> None:
        if int(code) != CUDA_SUCCESS:
            raise CudaError(int(code), self.error_string(int(code)))

    def _configure(self) -> None:
        self.library.cudaSetDevice.argtypes = [ctypes.c_int]
        self.library.cudaSetDevice.restype = ctypes.c_int
        self.library.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.library.cudaGetDevice.restype = ctypes.c_int
        self.library.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.library.cudaGetDeviceCount.restype = ctypes.c_int
        self.library.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.library.cudaMalloc.restype = ctypes.c_int
        self.library.cudaFree.argtypes = [ctypes.c_void_p]
        self.library.cudaFree.restype = ctypes.c_int
        self.library.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.library.cudaMemcpy.restype = ctypes.c_int
        self.library.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.library.cudaMemcpyAsync.restype = ctypes.c_int
        self.library.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.library.cudaMemset.restype = ctypes.c_int
        self.library.cudaMemsetAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        self.library.cudaMemsetAsync.restype = ctypes.c_int
        self.library.cudaMemGetInfo.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.library.cudaMemGetInfo.restype = ctypes.c_int
        self.library.cudaStreamCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        self.library.cudaStreamCreateWithFlags.restype = ctypes.c_int
        self.library.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.library.cudaStreamDestroy.restype = ctypes.c_int
        self.library.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.library.cudaStreamSynchronize.restype = ctypes.c_int
        self.library.cudaStreamWaitEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self.library.cudaStreamWaitEvent.restype = ctypes.c_int
        self.library.cudaStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.library.cudaStreamBeginCapture.restype = ctypes.c_int
        self.library.cudaStreamEndCapture.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.library.cudaStreamEndCapture.restype = ctypes.c_int
        self.library.cudaGraphInstantiate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_ulonglong,
        ]
        self.library.cudaGraphInstantiate.restype = ctypes.c_int
        self.library.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.library.cudaGraphLaunch.restype = ctypes.c_int
        self.library.cudaGraphExecDestroy.argtypes = [ctypes.c_void_p]
        self.library.cudaGraphExecDestroy.restype = ctypes.c_int
        self.library.cudaGraphDestroy.argtypes = [ctypes.c_void_p]
        self.library.cudaGraphDestroy.restype = ctypes.c_int
        self.library.cudaEventCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        self.library.cudaEventCreateWithFlags.restype = ctypes.c_int
        self.library.cudaEventDestroy.argtypes = [ctypes.c_void_p]
        self.library.cudaEventDestroy.restype = ctypes.c_int
        self.library.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.library.cudaEventRecord.restype = ctypes.c_int
        self.library.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
        self.library.cudaEventSynchronize.restype = ctypes.c_int
        self.library.cudaEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.library.cudaEventElapsedTime.restype = ctypes.c_int
        self.library.cudaDeviceSynchronize.argtypes = []
        self.library.cudaDeviceSynchronize.restype = ctypes.c_int
        self.library.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.library.cudaGetErrorString.restype = ctypes.c_char_p


_DEFAULT_RUNTIME: CudaRuntime | None = None


def get_cuda_runtime(path: str | None = None) -> CudaRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = CudaRuntime.load(path)
    return _DEFAULT_RUNTIME


def is_default_cuda_runtime_loaded() -> bool:
    return _DEFAULT_RUNTIME is not None


def reset_default_cuda_runtime_for_tests() -> None:
    global _DEFAULT_RUNTIME
    _DEFAULT_RUNTIME = None


def _default_cuda_library() -> str:
    return ctypes.util.find_library("cudart") or DEFAULT_CUDA_LIBRARY
