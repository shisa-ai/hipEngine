"""Four-axis kernel registry.

Kernels are keyed by ``(backend, layer, quant, variant)``. Dispatch code should resolve keys
through this registry instead of branching on backend or quant names.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

Kernel = Callable[..., Any]
MissingPolicy = Literal["error", "none"]


@dataclass(frozen=True, order=True)
class KernelKey:
    backend: str
    layer: str
    quant: str
    variant: str = ""

    def __post_init__(self) -> None:
        for field_name in ("backend", "layer", "quant"):
            if not getattr(self, field_name):
                raise ValueError(f"KernelKey.{field_name} must be non-empty")

    def display(self) -> str:
        suffix = f", variant={self.variant!r}" if self.variant else ""
        return f"backend={self.backend!r}, layer={self.layer!r}, quant={self.quant!r}{suffix}"


class DuplicateKernelError(ValueError):
    """Raised when registering a key that already has an implementation."""


class MissingKernelError(LookupError):
    """Raised when no registered kernel can satisfy a requested key."""

    def __init__(self, requested: KernelKey, attempted: Iterable[KernelKey]):
        self.requested = requested
        self.attempted = tuple(attempted)
        attempted_text = "; ".join(key.display() for key in self.attempted)
        super().__init__(
            f"no kernel implementation for {requested.display()}; attempted: {attempted_text}"
        )


_KERNELS: dict[KernelKey, Kernel] = {}


def register(key: KernelKey, kernel: Kernel, *, replace: bool = False) -> Kernel:
    """Register ``kernel`` under ``key`` and return the callable.

    ``replace=False`` catches accidental duplicate self-registration during package import.
    Tests may pass ``replace=True`` when deliberately overriding a fixture kernel.
    """

    if key in _KERNELS and not replace:
        raise DuplicateKernelError(f"kernel already registered for {key.display()}")
    _KERNELS[key] = kernel
    return kernel


def unregister(key: KernelKey) -> None:
    _KERNELS.pop(key, None)


def registered_keys() -> tuple[KernelKey, ...]:
    return tuple(sorted(_KERNELS))


def is_registered(key: KernelKey) -> bool:
    """Return ``True`` iff ``key`` has a non-None entry in the registry.

    Unlike :func:`resolve`, this performs an exact-key lookup with no
    backend/quant/variant fallbacks. It is the right primitive for dispatch
    rewrites that need to ask "is the *specific* P9 kernel in tree?" before
    rewriting to it (the resolve-style fallback would otherwise pick up a
    cpu_reference fp16 catch-all and trick the rewrite into firing).
    """

    return _KERNELS.get(key) is not None


def _candidate_keys(requested: KernelKey) -> tuple[KernelKey, ...]:
    """Return resolution candidates from narrowest to broadest.

    Order:
    1. exact backend/layer/quant/variant
    2. same without variant
    3. fp16 quant fallback on same backend
    4. cpu_reference backend fallback with the same quant/variant narrowing
    """

    candidates: list[KernelKey] = []

    def add(key: KernelKey) -> None:
        if key not in candidates:
            candidates.append(key)

    backends = [requested.backend]
    if requested.backend != "cpu_reference":
        backends.append("cpu_reference")

    for backend in backends:
        quant_options = [requested.quant]
        if requested.quant != "fp16":
            quant_options.append("fp16")
        for quant in quant_options:
            if requested.variant:
                add(KernelKey(backend, requested.layer, quant, requested.variant))
            add(KernelKey(backend, requested.layer, quant, ""))

    return tuple(candidates)


def resolve(
    *,
    backend: str,
    layer: str,
    quant: str,
    variant: str = "",
    missing: MissingPolicy = "error",
) -> Kernel | None:
    """Resolve a kernel implementation.

    The resolver applies generic fallback rules; callers do not branch on backend/quant.
    """

    requested = KernelKey(backend=backend, layer=layer, quant=quant, variant=variant)
    candidates = _candidate_keys(requested)
    for candidate in candidates:
        kernel = _KERNELS.get(candidate)
        if kernel is not None:
            return kernel
    if missing == "none":
        return None
    raise MissingKernelError(requested, candidates)


def can_resolve(*, backend: str, layer: str, quant: str, variant: str = "") -> bool:
    return (
        resolve(backend=backend, layer=layer, quant=quant, variant=variant, missing="none")
        is not None
    )


def clear_registry_for_tests() -> None:
    """Clear all registered kernels.

    Test-only helper. Production code should not call this.
    """

    _KERNELS.clear()
