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
_q5_t16_physical_mixed_rowtiles: ContextVar[bool] = ContextVar(
    "q5_t16_physical_mixed_rowtiles",
    default=False,
)
_q6_t16_physical_rowtile: ContextVar[bool] = ContextVar(
    "q6_t16_physical_rowtile",
    default=False,
)
_q6_t16_physical_mixed_rowtiles: ContextVar[bool] = ContextVar(
    "q6_t16_physical_mixed_rowtiles",
    default=False,
)
_physical_exact_rowtiles: ContextVar[bool] = ContextVar(
    "physical_exact_rowtiles",
    default=False,
)
_moe_physical_c2_disable_f32_residual: ContextVar[bool] = ContextVar(
    "moe_physical_c2_disable_f32_residual",
    default=False,
)
_moe_physical_c2_pairreuse: ContextVar[bool] = ContextVar(
    "moe_physical_c2_pairreuse",
    default=False,
)
_moe_physical_c2_exact_linear: ContextVar[bool] = ContextVar(
    "moe_physical_c2_exact_linear",
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
def q5_t16_physical_mixed_rowtiles_session(enabled: bool) -> Iterator[None]:
    """Select measured mixed R8/R6 chunks for one physical Q5 target."""

    token = _q5_t16_physical_mixed_rowtiles.set(bool(enabled))
    try:
        yield
    finally:
        _q5_t16_physical_mixed_rowtiles.reset(token)


def q5_t16_physical_mixed_rowtiles_enabled() -> bool:
    """Return whether this target selected mixed Q5 rowtile chunks."""

    return bool(_q5_t16_physical_mixed_rowtiles.get())


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
def q6_t16_physical_mixed_rowtiles_session(enabled: bool) -> Iterator[None]:
    """Select measured mixed R8/R6 chunks for one physical Q6 target."""

    token = _q6_t16_physical_mixed_rowtiles.set(bool(enabled))
    try:
        yield
    finally:
        _q6_t16_physical_mixed_rowtiles.reset(token)


def q6_t16_physical_mixed_rowtiles_enabled() -> bool:
    """Return whether this target selected mixed Q6 rowtile chunks."""

    return bool(_q6_t16_physical_mixed_rowtiles.get())


@contextlib.contextmanager
def physical_exact_rowtiles_session(enabled: bool) -> Iterator[None]:
    """Admit independently screened exact-row physical target rowtiles."""

    token = _physical_exact_rowtiles.set(bool(enabled))
    try:
        yield
    finally:
        _physical_exact_rowtiles.reset(token)


def physical_exact_rowtiles_enabled() -> bool:
    """Return whether this physical target selected exact-row rowtiles."""

    return bool(_physical_exact_rowtiles.get())


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


@contextlib.contextmanager
def moe_physical_c2_pairreuse_session(enabled: bool) -> Iterator[None]:
    """Select independently qualified R5/R6 MoE target weight reuse."""

    token = _moe_physical_c2_pairreuse.set(bool(enabled))
    try:
        yield
    finally:
        _moe_physical_c2_pairreuse.reset(token)


def moe_physical_c2_pairreuse_enabled() -> bool:
    """Return whether the current physical MoE C2 target selected pair reuse."""

    return bool(_moe_physical_c2_pairreuse.get())


@contextlib.contextmanager
def moe_physical_c2_exact_linear_session(enabled: bool) -> Iterator[None]:
    """Select the strict row-exact linear owner for one MoE physical C2 target."""

    token = _moe_physical_c2_exact_linear.set(bool(enabled))
    try:
        yield
    finally:
        _moe_physical_c2_exact_linear.reset(token)


def moe_physical_c2_exact_linear_enabled() -> bool:
    """Return whether packed MoE C2 uses row-exact linear layers."""

    return bool(_moe_physical_c2_exact_linear.get())


__all__ = [
    "moe_physical_c2_exact_linear_enabled",
    "moe_physical_c2_exact_linear_session",
    "moe_physical_c2_f32_residual_disabled",
    "moe_physical_c2_numerics_session",
    "moe_physical_c2_pairreuse_enabled",
    "moe_physical_c2_pairreuse_session",
    "physical_exact_rowtiles_enabled",
    "physical_exact_rowtiles_session",
    "q4_t16_physical_extra_rowtiles_enabled",
    "q4_t16_physical_extra_rowtiles_session",
    "q5_t16_physical_mixed_rowtiles_enabled",
    "q5_t16_physical_mixed_rowtiles_session",
    "q5_t16_physical_rowtile_enabled",
    "q5_t16_physical_rowtile_session",
    "q6_t16_physical_mixed_rowtiles_enabled",
    "q6_t16_physical_mixed_rowtiles_session",
    "q6_t16_physical_rowtile_enabled",
    "q6_t16_physical_rowtile_session",
]
