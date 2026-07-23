"""Registry for public speculative text-generation providers.

Provider selection is concrete across provider, target model, backend, and quant.
The engine resolves this key before generation so model/backend-specific branches
do not leak into the public API or server scheduler.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class SpeculativeProviderKey:
    """Concrete public speculative-provider implementation key."""

    provider: str
    target_model: str
    backend: str
    quant: str

    def __post_init__(self) -> None:
        for name in ("provider", "target_model", "backend", "quant"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"speculative provider {name} must be non-empty")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class SpeculativeProviderConfig:
    """User-selected sidecar and fixed provider shape."""

    provider: str
    draft_model: str | Path
    candidate_budget: int = 4

    def __post_init__(self) -> None:
        provider = str(self.provider).strip()
        draft_model = str(self.draft_model).strip()
        budget = int(self.candidate_budget)
        if not provider:
            raise ValueError("speculative provider must be non-empty")
        if not draft_model:
            raise ValueError("speculative draft_model must be non-empty")
        if budget <= 0:
            raise ValueError("speculative candidate_budget must be positive")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "draft_model", Path(draft_model).expanduser())
        object.__setattr__(self, "candidate_budget", budget)


class SpeculativeTextProvider(Protocol):
    """Public provider adapter owned by one target text generator."""

    provider_name: str

    def generate_detailed(self, request: Any) -> list[Any]: ...

    def stream_detailed(self, request: Any): ...

    def capabilities(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


SpeculativeProviderFactory = Callable[..., SpeculativeTextProvider]

_REGISTRY: dict[SpeculativeProviderKey, SpeculativeProviderFactory] = {}
_BUILTINS_REGISTERED = False


def register_speculative_provider(
    key: SpeculativeProviderKey,
    factory: SpeculativeProviderFactory,
    *,
    replace: bool = False,
) -> None:
    """Register one concrete provider factory."""

    if not isinstance(key, SpeculativeProviderKey):
        raise TypeError("key must be a SpeculativeProviderKey")
    if not callable(factory):
        raise TypeError("speculative provider factory must be callable")
    if key in _REGISTRY and not replace:
        raise KeyError(f"speculative provider already registered: {key}")
    _REGISTRY[key] = factory


def resolve_speculative_provider(
    *,
    provider: str,
    target_model: str,
    backend: str,
    quant: str,
) -> SpeculativeProviderFactory:
    """Resolve one exact provider implementation or fail closed."""

    key = SpeculativeProviderKey(
        provider=provider,
        target_model=target_model,
        backend=backend,
        quant=quant,
    )
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"unregistered speculative provider: {key}") from exc


def registered_speculative_providers() -> tuple[SpeculativeProviderKey, ...]:
    return tuple(sorted(_REGISTRY, key=lambda key: (key.provider, key.target_model, key.backend, key.quant)))


def register_builtin_speculative_providers() -> None:
    """Import built-in adapters once without eagerly loading model weights."""

    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from hipengine.generation import laguna_dflash as _laguna_dflash  # noqa: F401

    _BUILTINS_REGISTERED = True


__all__ = [
    "SpeculativeProviderConfig",
    "SpeculativeProviderFactory",
    "SpeculativeProviderKey",
    "SpeculativeTextProvider",
    "register_builtin_speculative_providers",
    "register_speculative_provider",
    "registered_speculative_providers",
    "resolve_speculative_provider",
]
