"""Bounded staged-provider SPI for scheduler-owned SPECDEC2 cycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from hipengine.kvcache import ResourceClaimSet
from hipengine.speculative.frontier import (
    CandidateGraph,
    SpecRequestPlan,
    SpeculativeCapability,
    TargetFrontier,
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


@runtime_checkable
class StagedSpeculativeRunner(Protocol):
    """Scheduler-facing staged target/provider lifecycle of a resident runner.

    The Generation-2 loop owns planning, complete claim reservation, and the
    transaction verdict.  A resident runner that advertises staged speculative
    support exposes this bounded lifecycle: every call is one cold-plan query or
    one stage of one engine cycle.  The runner owns target-frontier execution,
    provider proposal, and the frontier's KV-liveness identity; it cannot
    publish tokens, await a client, or run repeated cycles on its own.

    Optional lifecycle hooks the loop consults when present are
    ``speculative_claims_fit``, ``speculative_frontier_available``,
    ``speculative_kv_live_spans_owner``, and ``rollback_speculative_cycle``.
    They are not part of this protocol because the loop falls back cleanly when
    they are absent, so requiring them would reject a working owner.
    """

    def speculative_capability(
        self,
        request_semantics: Sequence[SpeculativeRequestSemantics],
    ) -> SpeculativeCapability | None: ...

    def speculative_component_claims(
        self,
        plan: SpecRequestPlan,
    ) -> Mapping[str, ResourceClaimSet]: ...

    def reserve_speculative_claims(self, claims: ResourceClaimSet) -> object: ...

    def release_speculative_claims(self, reservation: object) -> None: ...

    def prepare_speculative_requests(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> None: ...

    def propose_speculative_batch(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> CandidateGraph: ...

    def execute_target_frontier(
        self,
        plan: SpecRequestPlan,
        frontier: TargetFrontier,
        complete_claims: ResourceClaimSet,
        *,
        commit: bool,
        cancelled_request_ids: object,
    ) -> SpecCycleResult: ...


# The staged-frontier lifecycle the loop must be able to drive end to end.  A
# runner missing any of these cannot own one bounded target/provider cycle, so
# ``validate_staged_speculative_runner`` refuses it by name.
_STAGED_RUNNER_METHODS = (
    "speculative_capability",
    "speculative_component_claims",
    "reserve_speculative_claims",
    "release_speculative_claims",
    "prepare_speculative_requests",
    "propose_speculative_batch",
    "execute_target_frontier",
)
_STAGED_OWNER_ACCESSOR = "staged_speculative_owner"


def validate_staged_speculative_runner(
    runner: object,
) -> StagedSpeculativeRunner:
    """Fail construction if an object is not a bounded staged runner owner."""

    missing = tuple(
        name
        for name in _STAGED_RUNNER_METHODS
        if not callable(getattr(runner, name, None))
    )
    if missing:
        raise TypeError(
            "staged speculative runner is missing staged methods: "
            + ", ".join(missing)
        )
    return runner  # type: ignore[return-value]


def resolve_staged_speculative_runner(
    runner: object,
) -> StagedSpeculativeRunner | None:
    """Resolve a runner's staged lifecycle owner, or None for target-only AR.

    A runner opts into the seam by implementing ``staged_speculative_owner``;
    the returned owner is validated, so an incomplete opt-in fails loudly
    instead of silently falling back to autoregressive decoding.  A runner
    without the accessor is accepted only when it already exposes the complete
    staged lifecycle, which keeps a legacy duck-typed owner working without an
    explicit opt-in.
    """

    accessor = getattr(runner, _STAGED_OWNER_ACCESSOR, None)
    if callable(accessor):
        return validate_staged_speculative_runner(accessor())
    if all(
        callable(getattr(runner, name, None)) for name in _STAGED_RUNNER_METHODS
    ):
        return validate_staged_speculative_runner(runner)
    return None


_STAGED_ADAPTER_METHODS = (
    "capability",
    "claims_fit",
    "component_claims",
    "reserve_claims",
    "release_claims",
    "prepare_requests",
    "propose_batch",
    "execute_target_frontier",
    "rollback_cycle",
    "recover_cycle_failure",
    "prepare_k0",
    "register_request",
    "close",
)


@runtime_checkable
class StagedTargetAdapter(Protocol):
    """The runner-to-adapter face every staged target/provider adapter owns.

    The resident runners (dense Qwen GGUF, MoE Qwen GGUF, PARO) forward the
    scheduler's staged lifecycle to an adapter through exactly this surface.
    The declared methods are the ones every production adapter implements;
    hooks the runners consult only when present are deliberately absent so an
    adapter without them keeps working: ``execute_cycle`` (opaque
    complete-cycle execution), ``note_speculative_ar_commit``,
    ``kv_live_spans_owner``, ``staged_frontier``, ``release_request``,
    ``close_requests``, and ``containment_evidence``.
    """

    def capability(self, request_semantics) -> SpeculativeCapability | None: ...

    def claims_fit(self, plan) -> bool: ...

    def component_claims(self, plan) -> Mapping[str, ResourceClaimSet]: ...

    def reserve_claims(self, claims: ResourceClaimSet) -> object: ...

    def release_claims(self, reservation: object) -> None: ...

    def prepare_requests(
        self, plan, request_semantics, *, stream: int | None = None
    ) -> None: ...

    def propose_batch(
        self, plan, request_semantics, *, stream: int | None = None
    ) -> CandidateGraph: ...

    def execute_target_frontier(
        self,
        plan,
        frontier: TargetFrontier,
        complete_claims: ResourceClaimSet,
        *,
        commit: bool,
        cancelled_request_ids,
    ) -> object: ...

    def rollback_cycle(self, plan, *args) -> None: ...

    def recover_cycle_failure(self, plan, error) -> bool: ...

    def prepare_k0(
        self, plan, request_semantics, *, stream: int | None = None
    ) -> None: ...

    def register_request(self, request_id, candidate_budget) -> None: ...

    def close(self) -> None: ...


def validate_staged_target_adapter(adapter: object) -> StagedTargetAdapter:
    """Fail construction if an object is not a complete staged adapter.

    Called once at adapter construction, so an incomplete owner fails loudly
    with the names it is missing instead of surfacing as a mid-cycle
    ``AttributeError`` the first time the scheduler reaches the absent stage.
    """

    missing = tuple(
        name
        for name in _STAGED_ADAPTER_METHODS
        if not callable(getattr(adapter, name, None))
    )
    if missing:
        raise TypeError(
            "staged target adapter is missing staged lifecycle methods: "
            + ", ".join(missing)
        )
    return adapter  # type: ignore[return-value]


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
    "StagedSpeculativeRunner",
    "StagedTargetAdapter",
    "resolve_staged_speculative_runner",
    "validate_staged_speculative_provider",
    "validate_staged_speculative_runner",
    "validate_staged_target_adapter",
]
