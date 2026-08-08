"""Lazy ctypes wrapper for the HIP runtime.

Importing this module does not load ``libamdhip64.so`` and does not call the GPU. The shared
library is loaded only when ``get_hip_runtime()`` or ``HipRuntime.load()`` is called.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Final

from hipengine.core.runtime import MemcpyKind

HIP_SUCCESS: Final[int] = 0
HIP_HOST_REGISTER_MAPPED: Final[int] = 0x02
DEFAULT_HIP_LIBRARY: Final[str] = "libamdhip64.so"
HIP_GRAPH_NODE_TYPE_KERNEL: Final[int] = 0


class HipDim3(ctypes.Structure):
    """ctypes layout of HIP's three-uint ``dim3`` value."""

    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]


class HipKernelNodeParams(ctypes.Structure):
    """ctypes layout of ``hipKernelNodeParams`` from ``hip_runtime_api.h``."""

    # Keep the declaration order from ROCm's hip_runtime_api.h. It differs
    # from CUDA's historical presentation and is ABI-significant on ROCm.
    _fields_ = [
        ("blockDim", HipDim3),
        ("extra", ctypes.POINTER(ctypes.c_void_p)),
        ("func", ctypes.c_void_p),
        ("gridDim", HipDim3),
        ("kernelParams", ctypes.POINTER(ctypes.c_void_p)),
        ("sharedMemBytes", ctypes.c_uint),
    ]


HipMemcpyKind = MemcpyKind


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

    def stream_priority_range(self) -> tuple[int, int]:
        """Return HIP's ``(least, greatest)`` stream scheduling priorities."""

        least = ctypes.c_int()
        greatest = ctypes.c_int()
        self.check(
            self.library.hipDeviceGetStreamPriorityRange(
                ctypes.byref(least),
                ctypes.byref(greatest),
            )
        )
        return int(least.value), int(greatest.value)

    def stream_create(
        self,
        *,
        nonblocking: bool = True,
        priority: int | None = None,
    ) -> int:
        stream = ctypes.c_void_p()
        flags = 0x01 if nonblocking else 0x00
        if priority is None:
            self.check(
                self.library.hipStreamCreateWithFlags(
                    ctypes.byref(stream),
                    ctypes.c_uint(flags),
                )
            )
        else:
            self.check(
                self.library.hipStreamCreateWithPriority(
                    ctypes.byref(stream),
                    ctypes.c_uint(flags),
                    ctypes.c_int(priority),
                )
            )
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

    def graph_nodes(self, graph: int) -> tuple[int, ...]:
        """Return a stable copied snapshot of all native HIP graph node handles."""

        count = ctypes.c_size_t()
        function = self._inspection_function("hipGraphGetNodes")
        self.check(function(ctypes.c_void_p(graph), None, ctypes.byref(count)))
        if count.value == 0:
            return ()
        capacity = int(count.value)
        nodes = (ctypes.c_void_p * capacity)()
        filled = ctypes.c_size_t(capacity)
        self.check(function(ctypes.c_void_p(graph), nodes, ctypes.byref(filled)))
        if int(filled.value) != capacity:
            raise RuntimeError("HIP graph node count changed during inspection")
        return tuple(int(node or 0) for node in nodes)

    def graph_edges(self, graph: int) -> tuple[tuple[int, int], ...]:
        """Return copied ``(from, to)`` dependency edges for a HIP graph."""

        count = ctypes.c_size_t()
        function = self._inspection_function("hipGraphGetEdges")
        self.check(function(ctypes.c_void_p(graph), None, None, ctypes.byref(count)))
        if count.value == 0:
            return ()
        capacity = int(count.value)
        sources = (ctypes.c_void_p * capacity)()
        destinations = (ctypes.c_void_p * capacity)()
        filled = ctypes.c_size_t(capacity)
        self.check(
            function(
                ctypes.c_void_p(graph),
                sources,
                destinations,
                ctypes.byref(filled),
            )
        )
        if int(filled.value) != capacity:
            raise RuntimeError("HIP graph edge count changed during inspection")
        return tuple(
            (int(source or 0), int(destination or 0))
            for source, destination in zip(sources, destinations, strict=True)
        )

    def graph_node_type(self, node: int) -> int:
        node_type = ctypes.c_int()
        function = self._inspection_function("hipGraphNodeGetType")
        self.check(function(ctypes.c_void_p(node), ctypes.byref(node_type)))
        return int(node_type.value)

    def graph_kernel_node_params(self, node: int) -> HipKernelNodeParams:
        params = HipKernelNodeParams()
        function = self._inspection_function("hipGraphKernelNodeGetParams")
        self.check(function(ctypes.c_void_p(node), ctypes.byref(params)))
        return params

    def kernel_name_ref_by_ptr(self, function_ptr: int, stream: int = 0) -> str:
        function = self._inspection_function("hipKernelNameRefByPtr")
        raw = function(ctypes.c_void_p(function_ptr), ctypes.c_void_p(stream))
        if not raw:
            raise RuntimeError(f"HIP returned no kernel name for function {function_ptr:#x}")
        return raw.decode("utf-8", errors="strict")

    def current_device(self) -> int:
        device = ctypes.c_int()
        function = self._inspection_function("hipGetDevice")
        self.check(function(ctypes.byref(device)))
        return int(device.value)

    def device_pci_bus_id(self, device: int | None = None) -> str:
        selected = self.current_device() if device is None else int(device)
        output = ctypes.create_string_buffer(32)
        function = self._inspection_function("hipDeviceGetPCIBusId")
        self.check(function(output, ctypes.c_int(len(output)), ctypes.c_int(selected)))
        return output.value.decode("ascii", errors="strict")

    def _inspection_function(self, name: str):
        function = getattr(self.library, name, None)
        if function is None:
            raise RuntimeError(f"HIP runtime does not export required graph inspection API {name}")
        return function

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
        self.library.hipDeviceGetStreamPriorityRange.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.library.hipDeviceGetStreamPriorityRange.restype = ctypes.c_int
        self.library.hipStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self.library.hipStreamCreateWithFlags.restype = ctypes.c_int
        self.library.hipStreamCreateWithPriority.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        self.library.hipStreamCreateWithPriority.restype = ctypes.c_int
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
        inspection_signatures = {
            "hipGraphGetNodes": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int),
            "hipGraphGetEdges": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int),
            "hipGraphNodeGetType": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            "hipGraphKernelNodeGetParams": ([ctypes.c_void_p, ctypes.POINTER(HipKernelNodeParams)], ctypes.c_int),
            "hipKernelNameRefByPtr": ([ctypes.c_void_p, ctypes.c_void_p], ctypes.c_char_p),
            "hipGetDevice": ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            "hipDeviceGetPCIBusId": ([ctypes.c_char_p, ctypes.c_int, ctypes.c_int], ctypes.c_int),
        }
        for name, (argtypes, restype) in inspection_signatures.items():
            function = getattr(self.library, name, None)
            if function is not None:
                function.argtypes = argtypes
                function.restype = restype
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
