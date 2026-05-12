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
        self.library.hipMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.library.hipMemcpy.restype = ctypes.c_int
        self.library.hipDeviceSynchronize.argtypes = []
        self.library.hipDeviceSynchronize.restype = ctypes.c_int
        self.library.hipGetErrorString.argtypes = [ctypes.c_int]
        self.library.hipGetErrorString.restype = ctypes.c_char_p


_DEFAULT_RUNTIME: HipRuntime | None = None


def get_hip_runtime(path: str = DEFAULT_HIP_LIBRARY) -> HipRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = HipRuntime.load(path)
    return _DEFAULT_RUNTIME


def is_default_runtime_loaded() -> bool:
    return _DEFAULT_RUNTIME is not None


def reset_default_runtime_for_tests() -> None:
    global _DEFAULT_RUNTIME
    _DEFAULT_RUNTIME = None
