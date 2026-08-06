"""Backend-neutral runtime protocol shared by HIP and CUDA owners."""

from __future__ import annotations

from enum import IntEnum
from typing import Protocol


class MemcpyKind(IntEnum):
    HOST_TO_HOST = 0
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2
    DEVICE_TO_DEVICE = 3
    DEFAULT = 4


class DeviceRuntime(Protocol):
    """Small structural protocol required by resident model/runtime ownership."""

    def malloc(self, nbytes: int) -> int: ...

    def free(self, ptr: int) -> None: ...

    def memcpy(self, dst: int, src: int, nbytes: int, kind: MemcpyKind | int) -> None: ...

    def memcpy_async(
        self,
        dst: int,
        src: int,
        nbytes: int,
        kind: MemcpyKind | int,
        stream: int,
    ) -> None: ...

    def memset(self, dst: int, value: int, nbytes: int) -> None: ...

    def memset_async(self, dst: int, value: int, nbytes: int, stream: int) -> None: ...

    def stream_create(self, *, nonblocking: bool = True) -> int: ...

    def stream_destroy(self, stream: int) -> None: ...

    def stream_synchronize(self, stream: int) -> None: ...

    def event_create(self, *, flags: int = 0) -> int: ...

    def event_destroy(self, event: int) -> None: ...

    def event_record(self, event: int, stream: int = 0) -> None: ...

    def event_synchronize(self, event: int) -> None: ...

    def event_elapsed_time_ms(self, start: int, stop: int) -> float: ...
