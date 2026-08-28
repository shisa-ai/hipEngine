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

from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.speculative.provider import (
    StagedSpeculativeProvider,
    validate_staged_speculative_provider,
)


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
class SpeculativeProviderCapabilities:
    """Truthful provider ownership, shape, and fallback declaration."""

    provider_name: str
    artifact_fingerprint: str
    attachment_mode: str
    supported_modes: tuple[str, ...]
    max_verifier_rows: int
    transaction_mode: str
    provider_state_key: str
    provider_kv_key: str
    fixed_transaction_units: tuple[tuple[str, int], ...] = ()
    per_candidate_units: tuple[tuple[str, int], ...] = ()
    strict_fallback: str = "target_ar"

    def __post_init__(self) -> None:
        for field in (
            "provider_name", "artifact_fingerprint", "transaction_mode",
            "provider_state_key", "provider_kv_key", "strict_fallback",
        ):
            value = str(getattr(self, field)).strip()
            if not value:
                raise ValueError(f"{field} must be non-empty")
            object.__setattr__(self, field, value)
        if self.attachment_mode not in {"model_attached", "independent"}:
            raise ValueError("attachment_mode must be model_attached or independent")
        modes = tuple(str(mode) for mode in self.supported_modes)
        if not modes or len(set(modes)) != len(modes):
            raise ValueError("supported_modes must be non-empty and unique")
        if any(mode not in {"verify_chain", "verify_tree"} for mode in modes):
            raise ValueError("supported_modes may contain verify_chain/verify_tree only")
        if int(self.max_verifier_rows) <= 0:
            raise ValueError("max_verifier_rows must be positive")
        object.__setattr__(self, "supported_modes", modes)
        object.__setattr__(self, "max_verifier_rows", int(self.max_verifier_rows))
        for field in ("fixed_transaction_units", "per_candidate_units"):
            values = tuple((str(pool), int(units)) for pool, units in getattr(self, field))
            pools = tuple(pool for pool, _units in values)
            if len(pools) != len(set(pools)) or any(
                not pool or units <= 0 for pool, units in values
            ):
                raise ValueError(f"{field} must contain unique positive pool units")
            object.__setattr__(self, field, values)

    def supports(self, mode: str) -> bool:
        return str(mode) in self.supported_modes

    def require_mode(self, mode: str) -> None:
        selected = str(mode)
        if not self.supports(selected):
            raise NotImplementedError(
                f"provider {self.provider_name} does not support {selected}"
            )

    def resource_claims(
        self,
        *,
        request_id: int,
        candidate_rows: int,
        claim_id: str,
    ) -> ResourceClaimSet:
        rows = int(candidate_rows)
        if rows <= 0 or rows > self.max_verifier_rows:
            raise ValueError("candidate_rows exceed provider max_verifier_rows")
        units: dict[str, int] = {}
        for pool, amount in self.fixed_transaction_units:
            units[pool] = units.get(pool, 0) + amount
        for pool, amount in self.per_candidate_units:
            units[pool] = units.get(pool, 0) + amount * rows
        return ResourceClaimSet.from_mapping(
            claim_id,
            units,
            request_id=int(request_id),
            lifetime=ClaimLifetime.TRANSACTION,
        )


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
StagedSpeculativeProviderFactory = Callable[..., StagedSpeculativeProvider]

_REGISTRY: dict[SpeculativeProviderKey, SpeculativeProviderFactory] = {}
_STAGED_REGISTRY: dict[SpeculativeProviderKey, StagedSpeculativeProviderFactory] = {}
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


def register_staged_speculative_provider(
    key: SpeculativeProviderKey,
    factory: StagedSpeculativeProviderFactory,
    *,
    replace: bool = False,
) -> None:
    """Register one exact staged-provider factory without aliasing legacy routes."""

    if not isinstance(key, SpeculativeProviderKey):
        raise TypeError("key must be a SpeculativeProviderKey")
    if not callable(factory):
        raise TypeError("staged speculative provider factory must be callable")
    if key in _STAGED_REGISTRY and not replace:
        raise KeyError(f"staged speculative provider already registered: {key}")
    _STAGED_REGISTRY[key] = factory


def resolve_staged_speculative_provider(
    *,
    provider: str,
    target_model: str,
    backend: str,
    quant: str,
) -> StagedSpeculativeProviderFactory:
    """Resolve one exact staged factory or fail closed."""

    key = SpeculativeProviderKey(
        provider=provider,
        target_model=target_model,
        backend=backend,
        quant=quant,
    )
    try:
        return _STAGED_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"unregistered staged speculative provider: {key}") from exc


def registered_staged_speculative_providers() -> tuple[SpeculativeProviderKey, ...]:
    return tuple(
        sorted(
            _STAGED_REGISTRY,
            key=lambda key: (key.provider, key.target_model, key.backend, key.quant),
        )
    )


def construct_staged_speculative_provider(
    *,
    provider: str,
    target_model: str,
    backend: str,
    quant: str,
    **factory_kwargs: Any,
) -> StagedSpeculativeProvider:
    """Construct and validate one bounded provider at the cold boundary."""

    factory = resolve_staged_speculative_provider(
        provider=provider,
        target_model=target_model,
        backend=backend,
        quant=quant,
    )
    return validate_staged_speculative_provider(factory(**factory_kwargs))


def register_builtin_speculative_providers() -> None:
    """Import built-in adapters once without eagerly loading model weights."""

    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from hipengine.generation import laguna_dflash as _laguna_dflash  # noqa: F401
    from hipengine.generation import qwen4_exp_mtp as _qwen4_exp_mtp  # noqa: F401

    _BUILTINS_REGISTERED = True


__all__ = [
    "SpeculativeProviderCapabilities",
    "SpeculativeProviderConfig",
    "SpeculativeProviderFactory",
    "StagedSpeculativeProviderFactory",
    "SpeculativeProviderKey",
    "SpeculativeTextProvider",
    "construct_staged_speculative_provider",
    "register_builtin_speculative_providers",
    "register_speculative_provider",
    "register_staged_speculative_provider",
    "registered_speculative_providers",
    "registered_staged_speculative_providers",
    "resolve_speculative_provider",
    "resolve_staged_speculative_provider",
]
