"""Bounded staged-provider SPI for scheduler-owned SPECDEC2 cycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from hipengine.kvcache import ResourceClaimSet
from hipengine.speculative.frontier import (
    CandidateGraph,
    SpecRequestPlan,
    SpeculativeCapability,
)
from hipengine.speculative.transaction import SpecCycleResult, SpecCycleTransaction


def _required_text(value: object, label: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return text


@dataclass(frozen=True, slots=True)
class SpeculativeRequestSemantics:
    """Cold request semantics used for capability and K/K0 planning."""

    request_id: int
    sampling_mode: str
    mode: str
    context_tokens: int
    remaining_decode: int
    grammar_key: str | None = None
    stop_policy_key: str = "token_eos_length"

    def __post_init__(self) -> None:
        request_id = int(self.request_id)
        context_tokens = int(self.context_tokens)
        remaining_decode = int(self.remaining_decode)
        if request_id < 0:
            raise ValueError("request_id must be non-negative")
        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if remaining_decode < 0:
            raise ValueError("remaining_decode must be non-negative")
        sampling_mode = _required_text(self.sampling_mode, "sampling_mode")
        mode = str(self.mode)
        if mode not in {"verify_chain", "verify_tree"}:
            raise ValueError("mode must be verify_chain or verify_tree")
        grammar_key = self.grammar_key
        if grammar_key is not None:
            grammar_key = _required_text(grammar_key, "grammar_key")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "sampling_mode", sampling_mode)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "context_tokens", context_tokens)
        object.__setattr__(self, "remaining_decode", remaining_decode)
        object.__setattr__(self, "grammar_key", grammar_key)
        object.__setattr__(
            self,
            "stop_policy_key",
            _required_text(self.stop_policy_key, "stop_policy_key"),
        )


@runtime_checkable
class StagedSpeculativeProvider(Protocol):
    """One bounded provider subordinate to the Generation-2 scheduler.

    The protocol intentionally has no ``generate`` or ``stream`` method.  Every
    call covers one cold-plan query or one bounded stage of one engine cycle.
    """

    provider_name: str

    def capability(
        self,
        target_key: str,
        request_semantics: Sequence[SpeculativeRequestSemantics],
    ) -> SpeculativeCapability: ...

    def resource_claims(
        self,
        plan: SpecRequestPlan,
    ) -> Mapping[str, ResourceClaimSet]: ...

    def prepare_requests(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> None: ...

    def propose_batch(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> CandidateGraph: ...

    def commit_batch(
        self,
        result: SpecCycleResult,
        *,
        stream: int | None = None,
    ) -> None: ...

    def rollback_batch(
        self,
        transaction: SpecCycleTransaction,
        *,
        stream: int | None = None,
    ) -> None: ...

    def close_requests(self, request_ids: Sequence[int]) -> None: ...


_STAGED_METHODS = (
    "capability",
    "resource_claims",
    "prepare_requests",
    "propose_batch",
    "commit_batch",
    "rollback_batch",
    "close_requests",
)
_WHOLE_REQUEST_METHODS = (
    "generate_detailed",
    "stream_detailed",
    "generate_speculative_mtp_detailed",
)


def validate_staged_speculative_provider(
    provider: object,
) -> StagedSpeculativeProvider:
    """Fail construction if an object is not a bounded staged provider."""

    missing = tuple(
        name for name in _STAGED_METHODS if not callable(getattr(provider, name, None))
    )
    if missing:
        raise TypeError(
            "staged speculative provider is missing staged methods: "
            + ", ".join(missing)
        )
    provider_name = _required_text(
        getattr(provider, "provider_name", ""), "provider_name"
    )
    if any(callable(getattr(provider, name, None)) for name in _WHOLE_REQUEST_METHODS):
        raise TypeError(
            f"staged provider {provider_name} cannot expose whole-request generation"
        )
    return provider  # type: ignore[return-value]


__all__ = [
    "SpeculativeRequestSemantics",
    "StagedSpeculativeProvider",
    "validate_staged_speculative_provider",
]
