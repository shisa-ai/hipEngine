"""Request-local SPECDEC2 execution scopes shared by adapters and kernels."""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator


_q4_t16_physical_extra_rowtiles: ContextVar[bool] = ContextVar(
    "q4_t16_physical_extra_rowtiles",
    default=False,
)
_q5_t16_physical_rowtile: ContextVar[bool] = ContextVar(
    "q5_t16_physical_rowtile",
    default=False,
)
_q6_t16_physical_rowtile: ContextVar[bool] = ContextVar(
    "q6_t16_physical_rowtile",
    default=False,
)
_moe_physical_c2_disable_f32_residual: ContextVar[bool] = ContextVar(
    "moe_physical_c2_disable_f32_residual",
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


@contextlib.contextmanager
def q5_t16_physical_rowtile_session(enabled: bool) -> Iterator[None]:
    """Select the production-qualified Q5 rows6 physical target rowtile."""

    token = _q5_t16_physical_rowtile.set(bool(enabled))
    try:
        yield
    finally:
        _q5_t16_physical_rowtile.reset(token)


def q5_t16_physical_rowtile_enabled() -> bool:
    """Return whether the current physical target selected the Q5 rowtile."""

    return bool(_q5_t16_physical_rowtile.get())


@contextlib.contextmanager
def q6_t16_physical_rowtile_session(enabled: bool) -> Iterator[None]:
    """Select the production-qualified Q6 rows6 physical target rowtile."""

    token = _q6_t16_physical_rowtile.set(bool(enabled))
    try:
        yield
    finally:
        _q6_t16_physical_rowtile.reset(token)


def q6_t16_physical_rowtile_enabled() -> bool:
    """Return whether the current physical target selected the Q6 rowtile."""

    return bool(_q6_t16_physical_rowtile.get())


@contextlib.contextmanager
def moe_physical_c2_numerics_session(enabled: bool) -> Iterator[None]:
    """Select the independently gated packed-MoE C2 numerical boundary."""

    token = _moe_physical_c2_disable_f32_residual.set(bool(enabled))
    try:
        yield
    finally:
        _moe_physical_c2_disable_f32_residual.reset(token)


def moe_physical_c2_f32_residual_disabled() -> bool:
    """Return whether packed MoE C2 replaces the C1 F32-residual diagnostic."""

    return bool(_moe_physical_c2_disable_f32_residual.get())


__all__ = [
    "moe_physical_c2_f32_residual_disabled",
    "moe_physical_c2_numerics_session",
    "q4_t16_physical_extra_rowtiles_enabled",
    "q4_t16_physical_extra_rowtiles_session",
    "q5_t16_physical_rowtile_enabled",
    "q5_t16_physical_rowtile_session",
    "q6_t16_physical_rowtile_enabled",
    "q6_t16_physical_rowtile_session",
]
