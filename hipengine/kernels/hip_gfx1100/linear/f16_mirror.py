"""A producer-declared F16 image of an FP32 activation buffer.

A kernel that writes an FP32 activation can write its F16 rounding next to
it for free. Declaring the pair here lets the F16-operand GEMM launchers read
the image instead of converting the FP32 buffer again. The image must hold
exactly ``half(value)`` for every FP32 value, so a launcher that uses it
feeds its kernel the bytes its own conversion would have produced.

The declaration is context-local and single-slot: a new declaration replaces
the previous one, and the producer's owner invalidates it before the FP32
buffer or the image can be rewritten.
"""

from __future__ import annotations

import contextvars

_mirror: contextvars.ContextVar[tuple[int, int, int] | None] = contextvars.ContextVar(
    "hipengine_f16_activation_mirror", default=None
)


def declare_f16_mirror(source_ptr: int, count: int, f16_ptr: int) -> None:
    """Declare ``f16_ptr[0:count]`` as the F16 image of ``source_ptr[0:count]``."""

    if source_ptr <= 0 or f16_ptr <= 0 or count <= 0:
        raise ValueError("F16 mirror needs positive pointers and count")
    _mirror.set((int(source_ptr), int(count), int(f16_ptr)))


def invalidate_f16_mirror() -> None:
    _mirror.set(None)


def lookup_f16_mirror(source_ptr: int, count: int) -> int:
    """Return the image pointer for exactly this buffer and extent, else 0."""

    mirror = _mirror.get()
    if mirror is None or mirror[0] != int(source_ptr) or mirror[1] != int(count):
        return 0
    return mirror[2]


__all__ = ["declare_f16_mirror", "invalidate_f16_mirror", "lookup_f16_mirror"]
