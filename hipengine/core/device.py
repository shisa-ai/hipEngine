"""Torch-free device identifiers and scoped current-device selection.

Device enumeration via HIP/CUDA APIs lands later; this scaffold only defines the value
objects used by registries and tests.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass(frozen=True, order=True)
class Device:
    kind: str
    index: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"hip", "cuda", "cpu"}:
            raise ValueError("device kind must be one of: hip, cuda, cpu")
        if self.index < 0:
            raise ValueError("device index must be non-negative")

    def __str__(self) -> str:
        return self.kind if self.kind == "cpu" else f"{self.kind}:{self.index}"

    @classmethod
    def parse(cls, value: "Device | str") -> "Device":
        if isinstance(value, Device):
            return value
        text = str(value).strip()
        if text in {"cpu", "hip", "cuda"}:
            return cls(text)
        kind, _, index = text.partition(":")
        if not index:
            raise ValueError(f"device string {value!r} must be '<kind>' or '<kind>:<index>'")
        return cls(kind, int(index))


class CurrentDeviceRuntime(Protocol):
    """Minimal structural contract for scoped current-device selection."""

    def set_device(self, device: int) -> None: ...

    def get_device(self) -> int: ...


@contextmanager
def scoped_current_device(runtime: CurrentDeviceRuntime, device: int) -> Iterator[int]:
    """Select ``device`` for the duration of the block and restore the previous one.

    The HIP runtime keeps current-device state in thread-local storage, so this
    restores the caller's device even when the block raises. It intentionally does
    not synchronize: callers own completion and teardown ordering.
    """

    selected = int(device)
    if selected < 0:
        raise ValueError("device index must be non-negative")
    previous = runtime.get_device()
    runtime.set_device(selected)
    try:
        yield selected
    finally:
        runtime.set_device(previous)
