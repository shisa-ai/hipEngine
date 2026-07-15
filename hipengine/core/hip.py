"""Lazy ctypes wrapper for the HIP runtime.

Importing this module does not load ``libamdhip64.so`` and does not call the GPU. The shared
library is loaded only when ``get_hip_runtime()`` or ``HipRuntime.load()`` is called.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

HIP_SUCCESS: Final[int] = 0
HIP_HOST_REGISTER_MAPPED: Final[int] = 0x02
DEFAULT_HIP_LIBRARY: Final[str] = "libamdhip64.so"


class HipMemcpyKind(IntEnum):
    HOST_TO_HOST = 0
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2
    DEVICE_TO_DEVICE = 3
    DEFAULT = 4


class HipError(RuntimeError):
    """Raised when a HIP runtime call returns a non-success code."""

    def __init__(self, code: int, message: str):
        self.code = int(code)
        super().__init__(f"HIP error {self.code}: {message}")


@dataclass
class HipRuntime:
    """Loaded HIP runtime library with typed entry points."""

    library: ctypes.CDLL

    @classmethod
    def load(cls, path: str = DEFAULT_HIP_LIBRARY) -> "HipRuntime":
        runtime = cls(ctypes.CDLL(path))
        runtime._configure()
        return runtime

    def malloc(self, nbytes: int) -> int:
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        ptr = ctypes.c_void_p()
        self.check(self.library.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)))
        return 0 if ptr.value is None else int(ptr.value)

    def free(self, ptr: int) -> None:
        self.check(self.library.hipFree(ctypes.c_void_p(ptr)))

    def host_register(self, ptr: int, nbytes: int, *, flags: int = 0) -> None:
        """Page-lock a host range and optionally map it into the device address space."""

        if ptr <= 0:
            raise ValueError("ptr must be positive")
        if nbytes <= 0:
            raise ValueError("nbytes must be positive")
        self.check(
            self.library.hipHostRegister(
                ctypes.c_void_p(ptr),
                ctypes.c_size_t(nbytes),
                ctypes.c_uint(flags),
            )
        )

    def host_unregister(self, ptr: int) -> None:
        if ptr <= 0:
            raise ValueError("ptr must be positive")
        self.check(self.library.hipHostUnregister(ctypes.c_void_p(ptr)))

    def host_get_device_pointer(self, ptr: int, *, flags: int = 0) -> int:
        """Return the device-visible address for a registered mapped host range."""

        if ptr <= 0:
            raise ValueError("ptr must be positive")
        device_ptr = ctypes.c_void_p()
        self.check(
            self.library.hipHostGetDevicePointer(
                ctypes.byref(device_ptr),
                ctypes.c_void_p(ptr),
                ctypes.c_uint(flags),
            )
        )
        return 0 if device_ptr.value is None else int(device_ptr.value)

    def memcpy(self, dst: int, src: int, nbytes: int, kind: HipMemcpyKind | int) -> None:
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        self.check(
            self.library.hipMemcpy(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                ctypes.c_size_t(nbytes),
                int(kind),
            )
        )

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: HipMemcpyKind | int, stream: int) -> None:
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        self.check(
            self.library.hipMemcpyAsync(
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
        self.check(self.library.hipMemset(ctypes.c_void_p(dst), ctypes.c_int(value), ctypes.c_size_t(nbytes)))

    def memset_async(self, dst: int, value: int, nbytes: int, stream: int) -> None:
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        self.check(
            self.library.hipMemsetAsync(
                ctypes.c_void_p(dst),
                ctypes.c_int(value),
                ctypes.c_size_t(nbytes),
                ctypes.c_void_p(stream),
            )
        )

    def mem_get_info(self) -> tuple[int, int]:
        """Return ``(free_bytes, total_bytes)`` from ``hipMemGetInfo``."""

        free_bytes = ctypes.c_size_t()
        total_bytes = ctypes.c_size_t()
        self.check(self.library.hipMemGetInfo(ctypes.byref(free_bytes), ctypes.byref(total_bytes)))
        return int(free_bytes.value), int(total_bytes.value)

    def stream_create(self, *, nonblocking: bool = True) -> int:
        stream = ctypes.c_void_p()
        flags = 0x01 if nonblocking else 0x00
        self.check(self.library.hipStreamCreateWithFlags(ctypes.byref(stream), ctypes.c_uint(flags)))
        return 0 if stream.value is None else int(stream.value)

    def stream_destroy(self, stream: int) -> None:
        self.check(self.library.hipStreamDestroy(ctypes.c_void_p(stream)))

    def stream_synchronize(self, stream: int) -> None:
        self.check(self.library.hipStreamSynchronize(ctypes.c_void_p(stream)))

    def stream_wait_event(self, stream: int, event: int, *, flags: int = 0) -> None:
        self.check(
            self.library.hipStreamWaitEvent(
                ctypes.c_void_p(stream),
                ctypes.c_void_p(event),
                ctypes.c_uint(flags),
            )
        )

    def stream_begin_capture(self, stream: int, mode: int = 2) -> None:
        self.check(self.library.hipStreamBeginCapture(ctypes.c_void_p(stream), ctypes.c_int(mode)))

    def stream_end_capture(self, stream: int) -> int:
        graph = ctypes.c_void_p()
        self.check(self.library.hipStreamEndCapture(ctypes.c_void_p(stream), ctypes.byref(graph)))
        return 0 if graph.value is None else int(graph.value)

    def graph_instantiate(self, graph: int) -> int:
        graph_exec = ctypes.c_void_p()
        error_node = ctypes.c_void_p()
        log_buffer = ctypes.create_string_buffer(4096)
        self.check(
            self.library.hipGraphInstantiate(
                ctypes.byref(graph_exec),
                ctypes.c_void_p(graph),
                ctypes.byref(error_node),
                log_buffer,
                ctypes.c_size_t(len(log_buffer)),
            )
        )
        return 0 if graph_exec.value is None else int(graph_exec.value)

    def graph_launch(self, graph_exec: int, stream: int) -> None:
        self.check(self.library.hipGraphLaunch(ctypes.c_void_p(graph_exec), ctypes.c_void_p(stream)))

    def graph_exec_destroy(self, graph_exec: int) -> None:
        self.check(self.library.hipGraphExecDestroy(ctypes.c_void_p(graph_exec)))

    def graph_destroy(self, graph: int) -> None:
        self.check(self.library.hipGraphDestroy(ctypes.c_void_p(graph)))

    def event_create(self, *, flags: int = 0) -> int:
        event = ctypes.c_void_p()
        self.check(self.library.hipEventCreateWithFlags(ctypes.byref(event), ctypes.c_uint(flags)))
        return 0 if event.value is None else int(event.value)

    def event_destroy(self, event: int) -> None:
        self.check(self.library.hipEventDestroy(ctypes.c_void_p(event)))

    def event_record(self, event: int, stream: int = 0) -> None:
        self.check(self.library.hipEventRecord(ctypes.c_void_p(event), ctypes.c_void_p(stream)))

    def event_synchronize(self, event: int) -> None:
        self.check(self.library.hipEventSynchronize(ctypes.c_void_p(event)))

    def event_elapsed_time_ms(self, start: int, stop: int) -> float:
        elapsed = ctypes.c_float()
        self.check(
            self.library.hipEventElapsedTime(
                ctypes.byref(elapsed),
                ctypes.c_void_p(start),
                ctypes.c_void_p(stop),
            )
        )
        return float(elapsed.value)

    def device_synchronize(self) -> None:
        self.check(self.library.hipDeviceSynchronize())

    def error_string(self, code: int) -> str:
        raw = self.library.hipGetErrorString(int(code))
        if not raw:
            return "<unknown>"
        return raw.decode("utf-8", errors="replace")

    def check(self, code: int) -> None:
        if int(code) != HIP_SUCCESS:
            raise HipError(int(code), self.error_string(int(code)))

    def _configure(self) -> None:
        self.library.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.library.hipMalloc.restype = ctypes.c_int
        self.library.hipFree.argtypes = [ctypes.c_void_p]
        self.library.hipFree.restype = ctypes.c_int
        self.library.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self.library.hipHostRegister.restype = ctypes.c_int
        self.library.hipHostUnregister.argtypes = [ctypes.c_void_p]
        self.library.hipHostUnregister.restype = ctypes.c_int
        self.library.hipHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self.library.hipHostGetDevicePointer.restype = ctypes.c_int
        self.library.hipMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.library.hipMemcpy.restype = ctypes.c_int
        self.library.hipMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.library.hipMemcpyAsync.restype = ctypes.c_int
        self.library.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.library.hipMemset.restype = ctypes.c_int
        self.library.hipMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
        self.library.hipMemsetAsync.restype = ctypes.c_int
        self.library.hipMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.library.hipMemGetInfo.restype = ctypes.c_int
        self.library.hipStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self.library.hipStreamCreateWithFlags.restype = ctypes.c_int
        self.library.hipStreamDestroy.argtypes = [ctypes.c_void_p]
        self.library.hipStreamDestroy.restype = ctypes.c_int
        self.library.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.library.hipStreamSynchronize.restype = ctypes.c_int
        self.library.hipStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        self.library.hipStreamWaitEvent.restype = ctypes.c_int
        self.library.hipStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.library.hipStreamBeginCapture.restype = ctypes.c_int
        self.library.hipStreamEndCapture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.library.hipStreamEndCapture.restype = ctypes.c_int
        self.library.hipGraphInstantiate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.library.hipGraphInstantiate.restype = ctypes.c_int
        self.library.hipGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.library.hipGraphLaunch.restype = ctypes.c_int
        self.library.hipGraphExecDestroy.argtypes = [ctypes.c_void_p]
        self.library.hipGraphExecDestroy.restype = ctypes.c_int
        self.library.hipGraphDestroy.argtypes = [ctypes.c_void_p]
        self.library.hipGraphDestroy.restype = ctypes.c_int
        self.library.hipEventCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self.library.hipEventCreateWithFlags.restype = ctypes.c_int
        self.library.hipEventDestroy.argtypes = [ctypes.c_void_p]
        self.library.hipEventDestroy.restype = ctypes.c_int
        self.library.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.library.hipEventRecord.restype = ctypes.c_int
        self.library.hipEventSynchronize.argtypes = [ctypes.c_void_p]
        self.library.hipEventSynchronize.restype = ctypes.c_int
        self.library.hipEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.library.hipEventElapsedTime.restype = ctypes.c_int
        self.library.hipDeviceSynchronize.argtypes = []
        self.library.hipDeviceSynchronize.restype = ctypes.c_int
        self.library.hipGetErrorString.argtypes = [ctypes.c_int]
        self.library.hipGetErrorString.restype = ctypes.c_char_p


_DEFAULT_RUNTIME: HipRuntime | None = None


def get_hip_runtime(path: str = DEFAULT_HIP_LIBRARY) -> HipRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        # Runtime queue policy is read during HIP/HSA initialization, so apply
        # backend metadata before loading libamdhip64 rather than in a runner.
        from hipengine.kernels.backends import configure_hip_process_environment

        configure_hip_process_environment()
        _DEFAULT_RUNTIME = HipRuntime.load(path)
    return _DEFAULT_RUNTIME


def is_default_runtime_loaded() -> bool:
    return _DEFAULT_RUNTIME is not None


def reset_default_runtime_for_tests() -> None:
    global _DEFAULT_RUNTIME
    _DEFAULT_RUNTIME = None
