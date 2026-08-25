"""Deterministic SPEC-C0 host transaction and resource simulator."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from hipengine.generation.concurrency2_simulator import SimulatedResourceLedger
from hipengine.kvcache import ResourceClaimSet
from hipengine.speculative.frontier import SpecPlanReason, SpecTransactionMode
from hipengine.speculative.interfaces import AcceptResult, DraftBatch
from hipengine.speculative.transaction import (
    SpecCycleResult,
    SpecCycleStage,
    SpecCycleTelemetry,
    SpecCycleTransaction,
    compose_speculative_claims,
)

# Compatibility aliases retained while callers migrate to the production names.
SpecTransaction = SpecCycleTransaction
SpeculativeCycleResult = SpecCycleResult


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

    def reclaim_finished(self, request_id: int) -> SpeculativeRequestState:
        """Remove one terminal request so its resident slot can be refilled."""

        if self._active_owner is not None:
            raise RuntimeError("cannot reclaim during an active speculative cycle")
        state = self.state(request_id)
        if not state.finished or state.pending_transaction_id is not None:
            raise ValueError("only transaction-free finished requests may be reclaimed")
        del self._states[state.target_request_id]
        return state

    def admit_states(self, states: Sequence[SpeculativeRequestState]) -> None:
        """Admit new request owners between cycles for refill simulation."""

        if self._active_owner is not None:
            raise RuntimeError("cannot admit during an active speculative cycle")
        incoming = tuple(states)
        if not incoming:
            raise ValueError("admit_states requires at least one request")
        request_ids = tuple(state.target_request_id for state in incoming)
        slots = tuple(state.resident_slot for state in incoming)
        if len(request_ids) != len(set(request_ids)) or any(
            request_id in self._states for request_id in request_ids
        ):
            raise ValueError("admitted request ids must be new and unique")
        occupied = {state.resident_slot for state in self._states.values()}
        if len(slots) != len(set(slots)) or any(slot in occupied for slot in slots):
            raise ValueError("admitted resident slots must be free and unique")
        self._states.update(
            (state.target_request_id, state) for state in incoming
        )

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
        fail_at: SpecCycleStage | None = None,
        failure_message: str = "injected speculative cycle failure",
    ) -> SpeculativeCycleResult:
        if self._active_owner is not None:
            raise RuntimeError("a speculative cycle is already active")
        if cancel_at is not None and fail_at is not None:
            raise ValueError("cancel_at and fail_at are mutually exclusive")
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
        planned_counts = tuple(
            sum(1 for owner in draft.row_to_request if owner == request_id)
            for request_id in request_ids
        )
        if not any(available):
            raise ValueError("speculative cycle requires at least one active candidate")
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
        provider_states = tuple(
            state
            for state, candidate_count in zip(states, planned_counts, strict=True)
            if candidate_count > 0
        )
        transaction = SpecCycleTransaction(
            operation_id=operation_id,
            transaction_id=transaction_id,
            cycle_id=draft.cycle_id,
            request_ids=request_ids,
            reserved_claims=claims,
            pre_target_cursors=tuple(state.target_cursor for state in states),
            pre_provider_cursors=tuple(
                state.provider_cursor for state in provider_states
            ),
            pre_rng_counters=tuple(state.rng_counter for state in states),
            target_transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
            provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            target_owner=f"{operation_id}:target",
            provider_owner=f"{operation_id}:provider",
            provider_request_ids=tuple(
                state.target_request_id for state in provider_states
            ),
            target_checkpoint_ids=tuple(
                f"target:{state.target_request_id}:{state.target_cursor}"
                for state in states
            ),
            provider_checkpoint_ids=tuple(
                f"provider:{state.target_request_id}:{state.provider_cursor}"
                for state in provider_states
            ),
        )
        for state in states:
            self._states[state.target_request_id] = replace(
                state, pending_transaction_id=transaction_id
            )

        stage = SpecCycleStage.RESERVED
        interrupted = self._interrupt_if_requested(
            stage,
            cancel_at,
            cancel_request_id,
            fail_at,
            failure_message,
            transaction,
            request_ids,
        )
        if interrupted is not None:
            return interrupted
        transaction = replace(transaction, target_open=True)
        stage = SpecCycleStage.TARGET_OPEN
        interrupted = self._interrupt_if_requested(
            stage,
            cancel_at,
            cancel_request_id,
            fail_at,
            failure_message,
            transaction,
            request_ids,
        )
        if interrupted is not None:
            return interrupted
        transaction = replace(transaction, provider_open=True)
        stage = SpecCycleStage.PROVIDER_OPEN
        interrupted = self._interrupt_if_requested(
            stage,
            cancel_at,
            cancel_request_id,
            fail_at,
            failure_message,
            transaction,
            request_ids,
        )
        if interrupted is not None:
            return interrupted
        stage = SpecCycleStage.DRAFTED
        interrupted = self._interrupt_if_requested(
            stage,
            cancel_at,
            cancel_request_id,
            fail_at,
            failure_message,
            transaction,
            request_ids,
        )
        if interrupted is not None:
            return interrupted
        stage = SpecCycleStage.VERIFIED
        interrupted = self._interrupt_if_requested(
            stage,
            cancel_at,
            cancel_request_id,
            fail_at,
            failure_message,
            transaction,
            request_ids,
        )
        if interrupted is not None:
            return interrupted

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
        interrupted = self._interrupt_if_requested(
            stage,
            cancel_at,
            cancel_request_id,
            fail_at,
            failure_message,
            transaction,
            request_ids,
            accept_result,
        )
        if interrupted is not None:
            return interrupted
        for stage in (SpecCycleStage.READBACK, SpecCycleStage.COMMITTING):
            interrupted = self._interrupt_if_requested(
                stage,
                cancel_at,
                cancel_request_id,
                fail_at,
                failure_message,
                transaction,
                request_ids,
                accept_result,
            )
            if interrupted is not None:
                return interrupted

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
        candidate_counts = planned_counts
        telemetry = SpecCycleTelemetry(
            operation_id=operation_id,
            request_ids=request_ids,
            candidate_counts=candidate_counts,
            plan_reasons=tuple(
                SpecPlanReason.SPECULATIVE_QUALIFIED
                if candidate_count > 0
                else SpecPlanReason.POLICY_SELECTED_AR
                for candidate_count in candidate_counts
            ),
            proposal_widths=(len(provider_states),),
            target_row_decomposition=(len(request_ids) + draft.draft_rows,),
            execution_route="eager",
        )
        return SpecCycleResult.committed(
            transaction,
            accept_result,
            telemetry=telemetry,
        )

    def run_k0_cycle(
        self,
        request_ids: Sequence[int],
        *,
        component_claims: Mapping[str, ResourceClaimSet],
        output_tokens: Sequence[int],
        reasons: Sequence[SpecPlanReason] | None = None,
        cancel_at: SpecCycleStage | None = None,
        cancel_request_id: int | None = None,
        fail_at: SpecCycleStage | None = None,
        failure_message: str = "injected K0 cycle failure",
    ) -> SpecCycleResult:
        """Run one target-only AR transition through production K0 ownership."""

        if self._active_owner is not None:
            raise RuntimeError("a speculative cycle is already active")
        if cancel_at is not None and fail_at is not None:
            raise ValueError("cancel_at and fail_at are mutually exclusive")
        ids = tuple(int(request_id) for request_id in request_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("K0 request_ids must be non-empty and unique")
        states = tuple(self.state(request_id) for request_id in ids)
        if any(state.finished for state in states):
            raise ValueError("K0 cycle cannot include finished requests")
        tokens = tuple(int(token) for token in output_tokens)
        if len(tokens) != len(ids) or any(token < 0 for token in tokens):
            raise ValueError("output_tokens must be non-negative and align with request_ids")
        plan_reasons = (
            (SpecPlanReason.POLICY_SELECTED_AR,) * len(ids)
            if reasons is None
            else tuple(SpecPlanReason(reason) for reason in reasons)
        )
        if len(plan_reasons) != len(ids) or any(
            reason is SpecPlanReason.SPECULATIVE_QUALIFIED for reason in plan_reasons
        ):
            raise ValueError("K0 reasons must align and remain non-speculative")
        cycle_id = max(state.cycle_id for state in states) + 1
        self._transaction_sequence += 1
        transaction_id = self._transaction_sequence
        operation_id = f"k0-cycle:{cycle_id}:{transaction_id}"
        claims = compose_speculative_claims(operation_id, component_claims)
        self.ledger.reserve(operation_id, claims)
        self._active_owner = operation_id
        transaction = SpecCycleTransaction(
            operation_id=operation_id,
            transaction_id=transaction_id,
            cycle_id=cycle_id,
            request_ids=ids,
            reserved_claims=claims,
            pre_target_cursors=tuple(state.target_cursor for state in states),
            pre_rng_counters=tuple(state.rng_counter for state in states),
            target_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
            target_owner=f"{operation_id}:target",
            target_checkpoint_ids=tuple(
                f"target:{state.target_request_id}:{state.target_cursor}"
                for state in states
            ),
        )
        for state in states:
            self._states[state.target_request_id] = replace(
                state, pending_transaction_id=transaction_id
            )
        for stage in (SpecCycleStage.RESERVED,):
            interrupted = self._interrupt_if_requested(
                stage,
                cancel_at,
                cancel_request_id,
                fail_at,
                failure_message,
                transaction,
                ids,
            )
            if interrupted is not None:
                return interrupted
        transaction = replace(transaction, target_open=True)
        for stage in (
            SpecCycleStage.TARGET_OPEN,
            SpecCycleStage.VERIFIED,
            SpecCycleStage.ACCEPTED,
            SpecCycleStage.READBACK,
            SpecCycleStage.COMMITTING,
        ):
            interrupted = self._interrupt_if_requested(
                stage,
                cancel_at,
                cancel_request_id,
                fail_at,
                failure_message,
                transaction,
                ids,
            )
            if interrupted is not None:
                return interrupted
        finish_reasons = tuple(
            "length"
            if state.output_limit and len(state.visible_tokens) + 1 >= state.output_limit
            else None
            for state in states
        )
        accept_result = AcceptResult(
            request_ids=ids,
            accepted_counts=(0,) * len(ids),
            accepted_tokens=((),) * len(ids),
            transaction_id=transaction_id,
            correction_or_bonus_tokens=tokens,
            target_cursor_deltas=(1,) * len(ids),
            provider_cursor_deltas=(0,) * len(ids),
            finish_reasons=finish_reasons,
        )
        for state, token, finish_reason in zip(
            states, tokens, finish_reasons, strict=True
        ):
            self._states[state.target_request_id] = replace(
                state,
                target_cursor=state.target_cursor + 1,
                cycle_id=cycle_id,
                pending_transaction_id=None,
                rng_counter=state.rng_counter + 1,
                visible_tokens=(*state.visible_tokens, token),
                finished=finish_reason is not None,
            )
        transaction = replace(transaction, target_committed=True)
        self._release_active_owner()
        telemetry = SpecCycleTelemetry(
            operation_id=operation_id,
            request_ids=ids,
            candidate_counts=(0,) * len(ids),
            plan_reasons=plan_reasons,
            proposal_widths=(),
            target_row_decomposition=(len(ids),),
            execution_route="ar",
        )
        return SpecCycleResult.committed(
            transaction,
            accept_result,
            telemetry=telemetry,
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

    def _interrupt_if_requested(
        self,
        stage: SpecCycleStage,
        cancel_at: SpecCycleStage | None,
        cancel_request_id: int | None,
        fail_at: SpecCycleStage | None,
        failure_message: str,
        transaction: SpecTransaction,
        request_ids: tuple[int, ...],
        accept_result: AcceptResult | None = None,
    ) -> SpecCycleResult | None:
        cancelled = self._cancel_if_requested(
            stage,
            cancel_at,
            cancel_request_id,
            transaction,
            request_ids,
            accept_result,
        )
        if cancelled is not None:
            return cancelled
        return self._fail_if_requested(
            stage,
            fail_at,
            failure_message,
            transaction,
            request_ids,
            accept_result,
        )

    def _fail_if_requested(
        self,
        stage: SpecCycleStage,
        fail_at: SpecCycleStage | None,
        failure_message: str,
        transaction: SpecTransaction,
        request_ids: tuple[int, ...],
        accept_result: AcceptResult | None = None,
    ) -> SpecCycleResult | None:
        if fail_at is None or SpecCycleStage(fail_at) is not stage:
            return None
        for request_id in request_ids:
            state = self.state(request_id)
            self._states[request_id] = replace(
                state,
                pending_transaction_id=None,
            )
        rolled_back = replace(
            transaction,
            target_open=False,
            provider_open=False,
            rolled_back=True,
        )
        self._release_active_owner()
        return SpecCycleResult(
            stage=SpecCycleStage.FAILED,
            transaction=rolled_back,
            accept_result=accept_result,
            error=str(failure_message),
        )

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
        return SpecCycleResult(
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
