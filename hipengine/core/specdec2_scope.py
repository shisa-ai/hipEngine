"""Request-local SPECDEC2 execution scopes shared by adapters and kernels."""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator


_q4_t16_physical_extra_rowtiles: ContextVar[bool] = ContextVar(
    "q4_t16_physical_extra_rowtiles",
    default=False,
)


@contextlib.contextmanager
def q4_t16_physical_extra_rowtiles_session(enabled: bool) -> Iterator[None]:
    """Select production-qualified extra Q4 rowtiles for one physical target."""

    token = _q4_t16_physical_extra_rowtiles.set(bool(enabled))
    try:
        yield
    finally:
        _q4_t16_physical_extra_rowtiles.reset(token)


def q4_t16_physical_extra_rowtiles_enabled() -> bool:
    """Return whether the current target selected its production extra rowtiles."""

    return bool(_q4_t16_physical_extra_rowtiles.get())


__all__ = [
    "q4_t16_physical_extra_rowtiles_enabled",
    "q4_t16_physical_extra_rowtiles_session",
]
