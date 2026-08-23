"""Deterministic SPEC-C0 host transaction and resource simulator."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping, Sequence

from hipengine.generation.concurrency2_simulator import SimulatedResourceLedger
from hipengine.kvcache import ClaimConfidence, ResourceClaim, ResourceClaimSet
from hipengine.speculative.interfaces import AcceptResult, DraftBatch


def _required_text(value: object, label: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return text


@dataclass(frozen=True, slots=True)
class SpeculativeRequestState:
    """Canonical per-request SPEC-C0 ownership visible to the target service."""

    method_key: str
    provider_key: str
    policy_fingerprint: str
    target_request_id: int
    resident_slot: int
    target_cursor: int
    provider_cursor: int
    provider_state_lease: str
    cycle_id: int = 0
    pending_transaction_id: int | None = None
    rng_counter: int = 0
    output_limit: int = 0
    visible_tokens: tuple[int, ...] = ()
    holdback_tokens: tuple[int, ...] = ()
    stopped: bool = False
    finished: bool = False
    cancelled: bool = False

    def __post_init__(self) -> None:
        for field in ("method_key", "provider_key", "policy_fingerprint", "provider_state_lease"):
            object.__setattr__(self, field, _required_text(getattr(self, field), field))
        for field in (
            "target_request_id", "resident_slot", "target_cursor", "provider_cursor",
            "cycle_id", "rng_counter", "output_limit",
        ):
            value = int(getattr(self, field))
            if value < 0:
                raise ValueError(f"{field} must be non-negative")
            object.__setattr__(self, field, value)
        if self.pending_transaction_id is not None and int(self.pending_transaction_id) < 0:
            raise ValueError("pending_transaction_id must be non-negative")
        if any(token < 0 for token in (*self.visible_tokens, *self.holdback_tokens)):
            raise ValueError("visible/holdback token ids must be non-negative")
        if self.finished and self.pending_transaction_id is not None:
            raise ValueError("finished request cannot retain a pending transaction")


class SpecCycleStage(str, Enum):
    NEW = "new"
    RESERVED = "reserved"
    TARGET_OPEN = "target_open"
    PROVIDER_OPEN = "provider_open"
    DRAFTED = "drafted"
    VERIFIED = "verified"
    ACCEPTED = "accepted"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SpecTransaction:
    """Both provisional state owners plus their pre-transaction checkpoints."""

    operation_id: str
    transaction_id: int
    cycle_id: int
    request_ids: tuple[int, ...]
    reserved_claims: ResourceClaimSet
    pre_target_cursors: tuple[int, ...]
    pre_provider_cursors: tuple[int, ...]
    pre_rng_counters: tuple[int, ...]
    target_open: bool = False
    provider_open: bool = False
    target_committed: bool = False
    provider_committed: bool = False
    rolled_back: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _required_text(self.operation_id, "operation_id"))
        if min(int(self.transaction_id), int(self.cycle_id)) < 0:
            raise ValueError("transaction_id/cycle_id must be non-negative")
        if not self.request_ids or len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("transaction request_ids must be non-empty and unique")
        lengths = (
            len(self.pre_target_cursors), len(self.pre_provider_cursors), len(self.pre_rng_counters)
        )
        if any(length != len(self.request_ids) for length in lengths):
            raise ValueError("transaction checkpoints must align with request_ids")
        if self.rolled_back and (self.target_committed or self.provider_committed):
            raise ValueError("rolled-back transaction cannot be committed")
        if self.target_committed != self.provider_committed:
            raise ValueError("target/provider commit outcomes must match")


@dataclass(frozen=True, slots=True)
class SpeculativeCycleResult:
    stage: SpecCycleStage
    transaction: SpecTransaction
    accept_result: AcceptResult | None = None
    cancelled_request_ids: tuple[int, ...] = ()


def compose_speculative_claims(
    claim_id: str,
    components: Mapping[str, ResourceClaimSet],
) -> ResourceClaimSet:
    """Atomically compose provider, target, and transient ownership vectors."""

    identity = _required_text(claim_id, "claim_id")
    if not components:
        raise ValueError("speculative claim composition requires components")
    entries: dict[tuple[str, object], ResourceClaim] = {}
    confidence_order = {
        ClaimConfidence.EXACT: 0,
        ClaimConfidence.BOUNDED: 1,
        ClaimConfidence.UNKNOWN: 2,
    }
    request_ids = {claims.request_id for claims in components.values() if claims.request_id is not None}
    if len(request_ids) > 1:
        raise ValueError("speculative claim components belong to different requests")
    for component, claims in sorted(components.items()):
        _required_text(component, "component name")
        if not isinstance(claims, ResourceClaimSet):
            raise TypeError("speculative claim components must be ResourceClaimSet")
        for claim in claims.claims:
            current = entries.get(claim.key)
            if current is None:
                entries[claim.key] = claim
                continue
            confidence = max(
                (current.confidence, claim.confidence), key=confidence_order.__getitem__
            )
            entries[claim.key] = ResourceClaim(
                claim.pool_id,
                current.units + claim.units,
                claim.lifetime,
                confidence,
            )
    names = tuple(sorted(str(name) for name in components))
    return ResourceClaimSet(
        claim_id=identity,
        request_id=next(iter(request_ids), None),
        claims=tuple(entries[key] for key in sorted(entries, key=lambda item: (item[0], str(item[1])))),
        metadata=(("component_count", len(names)), ("components", ",".join(names))),
    )


class SpeculativeCycleSimulator:
    """Fake target/provider transaction coordinator with exact rollback semantics."""

    def __init__(
        self,
        ledger: SimulatedResourceLedger,
        states: Sequence[SpeculativeRequestState],
    ) -> None:
        if not isinstance(ledger, SimulatedResourceLedger):
            raise TypeError("ledger must be SimulatedResourceLedger")
        normalized = tuple(states)
        request_ids = tuple(state.target_request_id for state in normalized)
        if not normalized or len(set(request_ids)) != len(request_ids):
            raise ValueError("speculative states must be non-empty with unique request ids")
        slots = tuple(state.resident_slot for state in normalized)
        if len(set(slots)) != len(slots):
            raise ValueError("resident slots must be unique")
        self.ledger = ledger
        self._states = {state.target_request_id: state for state in normalized}
        self._transaction_sequence = 0
        self._active_owner: str | None = None

    def state(self, request_id: int) -> SpeculativeRequestState:
        return self._states[int(request_id)]

    def assert_conserved(self) -> None:
        self.ledger.assert_conserved()
        if self._active_owner is not None:
            raise AssertionError("speculative simulator retained an active cycle owner")
        if any(state.pending_transaction_id is not None for state in self._states.values()):
            raise AssertionError("speculative simulator retained pending transaction state")

    def run_cycle(
        self,
        draft: DraftBatch,
        *,
        component_claims: Mapping[str, ResourceClaimSet],
        accepted_counts: Sequence[int],
        correction_or_bonus_tokens: Sequence[int | None],
        cancel_at: SpecCycleStage | None = None,
        cancel_request_id: int | None = None,
    ) -> SpeculativeCycleResult:
        if self._active_owner is not None:
            raise RuntimeError("a speculative cycle is already active")
        request_ids = tuple(int(request_id) for request_id in draft.request_ids)
        states = tuple(self.state(request_id) for request_id in request_ids)
        if any(state.finished for state in states):
            raise ValueError("speculative cycle cannot include finished requests")
        expected_cycle = max(state.cycle_id for state in states) + 1
        if draft.cycle_id != expected_cycle:
            raise ValueError("draft cycle_id must advance request cycle state exactly once")
        if draft.resident_slots and draft.resident_slots != tuple(
            self.state(request_id).resident_slot for request_id in draft.row_to_request
        ):
            raise ValueError("draft resident_slots must match request ownership")
        counts = tuple(int(count) for count in accepted_counts)
        corrections = tuple(correction_or_bonus_tokens)
        if len(counts) != len(request_ids) or len(corrections) != len(request_ids):
            raise ValueError("accept/correction rows must align with request_ids")
        available = tuple(
            sum(1 for owner, active in zip(draft.row_to_request, draft.active_mask or (True,) * draft.draft_rows, strict=True) if owner == request_id and active)
            for request_id in request_ids
        )
        if any(count < 0 or count > maximum for count, maximum in zip(counts, available, strict=True)):
            raise ValueError("accepted counts exceed active draft candidates")
        if any(token is not None and int(token) < 0 for token in corrections):
            raise ValueError("correction/bonus tokens must be non-negative")

        self._transaction_sequence += 1
        transaction_id = self._transaction_sequence
        operation_id = f"spec-cycle:{draft.cycle_id}:{transaction_id}"
        owner = operation_id
        claims = compose_speculative_claims(operation_id, component_claims)
        self.ledger.reserve(owner, claims)
        self._active_owner = owner
        transaction = SpecTransaction(
            operation_id=operation_id,
            transaction_id=transaction_id,
            cycle_id=draft.cycle_id,
            request_ids=request_ids,
            reserved_claims=claims,
            pre_target_cursors=tuple(state.target_cursor for state in states),
            pre_provider_cursors=tuple(state.provider_cursor for state in states),
            pre_rng_counters=tuple(state.rng_counter for state in states),
        )
        for state in states:
            self._states[state.target_request_id] = replace(
                state, pending_transaction_id=transaction_id
            )

        stage = SpecCycleStage.RESERVED
        cancelled = self._cancel_if_requested(
            stage, cancel_at, cancel_request_id, transaction, request_ids
        )
        if cancelled is not None:
            return cancelled
        transaction = replace(transaction, target_open=True)
        stage = SpecCycleStage.TARGET_OPEN
        cancelled = self._cancel_if_requested(stage, cancel_at, cancel_request_id, transaction, request_ids)
        if cancelled is not None:
            return cancelled
        transaction = replace(transaction, provider_open=True)
        stage = SpecCycleStage.PROVIDER_OPEN
        cancelled = self._cancel_if_requested(stage, cancel_at, cancel_request_id, transaction, request_ids)
        if cancelled is not None:
            return cancelled
        stage = SpecCycleStage.DRAFTED
        cancelled = self._cancel_if_requested(stage, cancel_at, cancel_request_id, transaction, request_ids)
        if cancelled is not None:
            return cancelled
        stage = SpecCycleStage.VERIFIED
        cancelled = self._cancel_if_requested(stage, cancel_at, cancel_request_id, transaction, request_ids)
        if cancelled is not None:
            return cancelled

        accepted_tokens = self._accepted_tokens(draft, counts)
        visible = tuple(
            (*tokens, *(() if correction is None else (int(correction),)))
            for tokens, correction in zip(accepted_tokens, corrections, strict=True)
        )
        target_deltas = tuple(len(tokens) for tokens in visible)
        provider_deltas = counts
        visible_ranges = tuple(
            (len(state.visible_tokens), len(state.visible_tokens) + len(tokens))
            for state, tokens in zip(states, visible, strict=True)
        )
        finish_reasons = tuple(
            "length" if state.output_limit and end >= state.output_limit else None
            for state, (_start, end) in zip(states, visible_ranges, strict=True)
        )
        accept_result = AcceptResult(
            request_ids=request_ids,
            accepted_counts=counts,
            accepted_tokens=accepted_tokens,
            transaction_id=transaction_id,
            correction_or_bonus_tokens=corrections,
            visible_token_ranges=visible_ranges,
            target_cursor_deltas=target_deltas,
            provider_cursor_deltas=provider_deltas,
            finish_reasons=finish_reasons,
        )
        stage = SpecCycleStage.ACCEPTED
        cancelled = self._cancel_if_requested(
            stage, cancel_at, cancel_request_id, transaction, request_ids, accept_result
        )
        if cancelled is not None:
            return cancelled

        for state, tokens, target_delta, provider_delta, finish_reason in zip(
            states, visible, target_deltas, provider_deltas, finish_reasons, strict=True
        ):
            self._states[state.target_request_id] = replace(
                state,
                target_cursor=state.target_cursor + target_delta,
                provider_cursor=state.provider_cursor + provider_delta,
                cycle_id=draft.cycle_id,
                pending_transaction_id=None,
                rng_counter=state.rng_counter + target_delta,
                visible_tokens=(*state.visible_tokens, *tokens),
                finished=finish_reason is not None,
            )
        transaction = replace(
            transaction, target_committed=True, provider_committed=True
        )
        self._release_active_owner()
        return SpeculativeCycleResult(
            stage=SpecCycleStage.COMMITTED,
            transaction=transaction,
            accept_result=accept_result,
        )

    def _accepted_tokens(
        self, draft: DraftBatch, accepted_counts: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...]:
        active = draft.active_mask or (True,) * draft.draft_rows
        result: list[tuple[int, ...]] = []
        for request_id, count in zip(draft.request_ids, accepted_counts, strict=True):
            selected: list[int] = []
            for depth in range(1, count + 1):
                rows = [
                    index for index, (owner, row_depth, enabled) in enumerate(
                        zip(draft.row_to_request, draft.draft_depths, active, strict=True)
                    )
                    if owner == request_id and row_depth == depth and enabled
                ]
                if len(rows) != 1:
                    raise ValueError("accepted tree depth is ambiguous in fake simulator")
                selected.append(draft.candidate_tokens[rows[0]])
            result.append(tuple(selected))
        return tuple(result)

    def _cancel_if_requested(
        self,
        stage: SpecCycleStage,
        cancel_at: SpecCycleStage | None,
        cancel_request_id: int | None,
        transaction: SpecTransaction,
        request_ids: tuple[int, ...],
        accept_result: AcceptResult | None = None,
    ) -> SpeculativeCycleResult | None:
        if cancel_at is None or SpecCycleStage(cancel_at) is not stage:
            return None
        if cancel_request_id is None or int(cancel_request_id) not in request_ids:
            raise ValueError("cancel_request_id must belong to the active cycle")
        cancelled_id = int(cancel_request_id)
        for request_id in request_ids:
            state = self.state(request_id)
            self._states[request_id] = replace(
                state,
                pending_transaction_id=None,
                cancelled=request_id == cancelled_id,
                finished=request_id == cancelled_id,
            )
        rolled_back = replace(
            transaction,
            target_open=False,
            provider_open=False,
            rolled_back=True,
        )
        self._release_active_owner()
        return SpeculativeCycleResult(
            stage=SpecCycleStage.CANCELLED,
            transaction=rolled_back,
            accept_result=accept_result,
            cancelled_request_ids=(cancelled_id,),
        )

    def _release_active_owner(self) -> None:
        if self._active_owner is None:
            raise AssertionError("speculative cycle has no active resource owner")
        self.ledger.release(self._active_owner)
        self._active_owner = None
