"""Request-local SPECDEC2 execution scopes shared by adapters and kernels."""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator


_q4_t16_physical_gate_up_rowtile: ContextVar[bool] = ContextVar(
    "q4_t16_physical_gate_up_rowtile",
    default=False,
)


@contextlib.contextmanager
def q4_t16_physical_gate_up_rowtile_session(enabled: bool) -> Iterator[None]:
    """Select the production-qualified physical gate/up rowtile for one target."""

    token = _q4_t16_physical_gate_up_rowtile.set(bool(enabled))
    try:
        yield
    finally:
        _q4_t16_physical_gate_up_rowtile.reset(token)


def q4_t16_physical_gate_up_rowtile_enabled() -> bool:
    """Return whether the current target owner selected the production route."""

    return bool(_q4_t16_physical_gate_up_rowtile.get())


__all__ = [
    "q4_t16_physical_gate_up_rowtile_enabled",
    "q4_t16_physical_gate_up_rowtile_session",
]
