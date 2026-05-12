"""Torch-free device identifiers.

Device enumeration via HIP/CUDA APIs lands later; this scaffold only defines the value
objects used by registries and tests.
"""

from __future__ import annotations

from dataclasses import dataclass


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
