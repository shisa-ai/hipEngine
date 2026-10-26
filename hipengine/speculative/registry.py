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


# Generic fallback for a provider registration or a directly constructed
# SpeculativeProviderConfig that declares no depth. Built-in providers declare
# their own value at registration; server and LLM resolution reads that
# declaration instead of this constant.
DEFAULT_PROVIDER_CANDIDATE_BUDGET = 4

# The ``quant`` axis value a provider registers under when its behaviour does not
# depend on the target's quantization. ``resolve_speculative_provider`` falls back
# to it after an exact-key miss, so a provider that genuinely has no quantization
# dependence never has to enumerate the quantizations it would accept.
QUANT_AGNOSTIC = "*"


@dataclass(frozen=True, slots=True)
class SpeculativeProviderConfig:
    """User-selected sidecar and fixed provider shape."""

    provider: str
    draft_model: str | Path
    candidate_budget: int = DEFAULT_PROVIDER_CANDIDATE_BUDGET

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


@dataclass(frozen=True, slots=True)
class SpeculativeProviderRegistration:
    """One provider factory plus the shape it declares before model load.

    ``candidate_budget`` is the depth the provider owns when the caller pins
    none. Keeping it on the registration rather than on a loaded adapter lets
    server capability reads report the provider's own default without scanning
    or materializing the target artifact.
    """

    factory: SpeculativeProviderFactory
    candidate_budget: int = DEFAULT_PROVIDER_CANDIDATE_BUDGET

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError("speculative provider factory must be callable")
        budget = int(self.candidate_budget)
        if budget <= 0:
            raise ValueError("speculative provider candidate_budget must be positive")
        object.__setattr__(self, "candidate_budget", budget)


class SpeculativeProviderBudgetAmbiguousError(ValueError):
    """A provider declares different depths for different registry axes.

    The caller supplied too few axes to select one registration. This is a
    distinct condition from an unregistered provider, which remains a
    ``KeyError``: the provider is known and runnable, but the pre-load caller
    cannot yet choose its declaration.
    """


_REGISTRY: dict[SpeculativeProviderKey, SpeculativeProviderRegistration] = {}
_BUILTINS_REGISTERED = False


def register_speculative_provider(
    key: SpeculativeProviderKey,
    factory: SpeculativeProviderFactory,
    *,
    replace: bool = False,
    candidate_budget: int | None = None,
) -> None:
    """Register one concrete provider factory and its declared default depth."""

    if not isinstance(key, SpeculativeProviderKey):
        raise TypeError("key must be a SpeculativeProviderKey")
    registration = SpeculativeProviderRegistration(
        factory=factory,
        candidate_budget=(
            DEFAULT_PROVIDER_CANDIDATE_BUDGET
            if candidate_budget is None
            else candidate_budget
        ),
    )
    if key in _REGISTRY and not replace:
        raise KeyError(f"speculative provider already registered: {key}")
    _REGISTRY[key] = registration


def resolve_speculative_provider(
    *,
    provider: str,
    target_model: str,
    backend: str,
    quant: str,
) -> SpeculativeProviderFactory:
    """Resolve one provider implementation, or fail closed.

    An exact ``(provider, target_model, backend, quant)`` key wins. Failing that,
    a provider registered with ``quant=QUANT_AGNOSTIC`` is accepted: some
    providers genuinely do not depend on the target's quantization -- a
    model-attached sidecar whose verify reads whatever logits the backbone
    produces is the case this exists for -- and without this an exact-key
    registry would force every such provider to carry a written list of
    known-good quantizations. That list would refuse a newly resolvable
    quantization for a reason that is not a capability miss, which is the
    failure mode this registry must not have.

    A provider that *does* depend on the quantization still registers exact keys
    and is never matched by the wildcard.
    """

    key = SpeculativeProviderKey(
        provider=provider,
        target_model=target_model,
        backend=backend,
        quant=quant,
    )
    try:
        return _REGISTRY[key].factory
    except KeyError:
        pass
    agnostic = SpeculativeProviderKey(
        provider=provider,
        target_model=target_model,
        backend=backend,
        quant=QUANT_AGNOSTIC,
    )
    try:
        return _REGISTRY[agnostic].factory
    except KeyError as exc:
        raise KeyError(f"unregistered speculative provider: {key}") from exc


def resolve_speculative_provider_candidate_budget(
    provider: str,
    *,
    target_model: str | None = None,
    backend: str | None = None,
    quant: str | None = None,
) -> int:
    """Return the candidate depth a provider declares before model load.

    The declaration lives on the registration, so this reads registry metadata
    only: it never constructs a provider, loads weights, or inspects an
    artifact. With all registry axes supplied it follows the same
    exact-then-quantization-agnostic lookup as ``resolve_speculative_provider``.
    With an axis unknown, every matching registration must agree; otherwise the
    caller gets an error naming the axis it must supply instead of a guessed
    depth. An unregistered provider is a loud miss, not an unresolved value.
    """

    register_builtin_speculative_providers()
    name = str(provider).strip()
    if not name:
        raise ValueError("speculative provider must be non-empty")
    if target_model is not None and backend is not None and quant is not None:
        key = SpeculativeProviderKey(
            provider=name,
            target_model=target_model,
            backend=backend,
            quant=quant,
        )
        registration = _REGISTRY.get(key)
        if registration is None:
            agnostic = SpeculativeProviderKey(
                provider=name,
                target_model=target_model,
                backend=backend,
                quant=QUANT_AGNOSTIC,
            )
            registration = _REGISTRY.get(agnostic)
        if registration is None:
            raise KeyError(f"unregistered speculative provider: {key}")
        return registration.candidate_budget
    target = None if target_model is None else str(target_model).strip()
    host = None if backend is None else str(backend).strip()
    storage = None if quant is None else str(quant).strip()
    matching = [
        registration
        for key, registration in _REGISTRY.items()
        if key.provider == name
        and (target is None or key.target_model == target)
        and (host is None or key.backend == host)
        and (storage is None or key.quant in {storage, QUANT_AGNOSTIC})
    ]
    if not matching:
        raise KeyError(f"unregistered speculative provider: {name}")
    declared = {registration.candidate_budget for registration in matching}
    if len(declared) == 1:
        return next(iter(declared))
    missing = [
        axis
        for axis, value in (
            ("target_model", target),
            ("backend", host),
            ("quant", storage),
        )
        if value is None
    ]
    raise SpeculativeProviderBudgetAmbiguousError(
        f"speculative provider {name} declares multiple candidate budgets "
        f"({sorted(declared)}); supply {', '.join(missing)} to select one"
    )


def registered_speculative_providers() -> tuple[SpeculativeProviderKey, ...]:
    return tuple(sorted(_REGISTRY, key=lambda key: (key.provider, key.target_model, key.backend, key.quant)))


def register_builtin_speculative_providers() -> None:
    """Import built-in adapters once without eagerly loading model weights."""

    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from hipengine.generation import gemma4_mtp as _gemma4_mtp  # noqa: F401
    from hipengine.generation import laguna_dflash as _laguna_dflash  # noqa: F401
    from hipengine.generation import qwen4_exp_mtp as _qwen4_exp_mtp  # noqa: F401

    _BUILTINS_REGISTERED = True


__all__ = [
    "DEFAULT_PROVIDER_CANDIDATE_BUDGET",
    "QUANT_AGNOSTIC",
    "SpeculativeProviderBudgetAmbiguousError",
    "SpeculativeProviderCapabilities",
    "SpeculativeProviderConfig",
    "SpeculativeProviderFactory",
    "SpeculativeProviderKey",
    "SpeculativeProviderRegistration",
    "SpeculativeTextProvider",
    "register_builtin_speculative_providers",
    "register_speculative_provider",
    "registered_speculative_providers",
    "resolve_speculative_provider",
    "resolve_speculative_provider_candidate_budget",
]
