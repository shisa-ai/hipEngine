"""Production SPECDEC2 transaction, result, and telemetry records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

from hipengine.kvcache import (
    ClaimConfidence,
    ResourceClaim,
    ResourceClaimSet,
)
from hipengine.speculative.frontier import SpecPlanReason, SpecTransactionMode
from hipengine.speculative.interfaces import AcceptResult


def _required_text(value: object, label: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return text


def _request_ids(values: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if not normalized or any(value < 0 for value in normalized):
        raise ValueError("request_ids must be non-empty and non-negative")
    if len(normalized) != len(set(normalized)):
        raise ValueError("request_ids must be unique")
    return normalized


def _aligned_nonnegative(
    values: Sequence[int],
    request_ids: Sequence[int],
    label: str,
) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if len(normalized) != len(request_ids) or any(value < 0 for value in normalized):
        raise ValueError(f"{label} must be non-negative and align with request_ids")
    return normalized


def _aligned_checkpoints(
    values: Sequence[str],
    request_ids: Sequence[int],
    label: str,
) -> tuple[str, ...]:
    normalized = tuple(_required_text(value, label) for value in values)
    if len(normalized) != len(request_ids):
        raise ValueError(f"{label} must align with request_ids")
    return normalized


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
    request_ids = {
        claims.request_id
        for claims in components.values()
        if claims.request_id is not None
    }
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
                (current.confidence, claim.confidence),
                key=confidence_order.__getitem__,
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
        claims=tuple(
            entries[key]
            for key in sorted(entries, key=lambda item: (item[0], str(item[1])))
        ),
        metadata=(("component_count", len(names)), ("components", ",".join(names))),
    )


class SpecCycleStage(StrEnum):
    """One scheduler-visible stage in a bounded SPECDEC2 cycle."""

    NEW = "new"
    RESERVED = "reserved"
    TARGET_OPEN = "target_open"
    PROVIDER_OPEN = "provider_open"
    DRAFTED = "drafted"
    VERIFIED = "verified"
    ACCEPTED = "accepted"
    READBACK = "readback"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SpecCycleTransaction:
    """Atomic target plus optional provider ownership and checkpoints."""

    operation_id: str
    transaction_id: int
    cycle_id: int
    request_ids: tuple[int, ...]
    reserved_claims: ResourceClaimSet
    pre_target_cursors: tuple[int, ...]
    pre_rng_counters: tuple[int, ...]
    target_transaction_mode: SpecTransactionMode
    target_owner: str
    target_checkpoint_ids: tuple[str, ...]
    pre_provider_cursors: tuple[int, ...] = ()
    provider_transaction_mode: SpecTransactionMode | None = None
    provider_owner: str | None = None
    provider_request_ids: tuple[int, ...] = ()
    provider_checkpoint_ids: tuple[str, ...] = ()
    target_open: bool = False
    provider_open: bool = False
    target_committed: bool = False
    provider_committed: bool = False
    rolled_back: bool = False

    def __post_init__(self) -> None:
        operation_id = _required_text(self.operation_id, "operation_id")
        transaction_id = int(self.transaction_id)
        cycle_id = int(self.cycle_id)
        if min(transaction_id, cycle_id) < 0:
            raise ValueError("transaction_id/cycle_id must be non-negative")
        request_ids = _request_ids(self.request_ids)
        if not isinstance(self.reserved_claims, ResourceClaimSet):
            raise TypeError("reserved_claims must be a ResourceClaimSet")
        if self.reserved_claims.claim_id != operation_id:
            raise ValueError("reserved_claims claim_id must equal operation_id")
        target_cursors = _aligned_nonnegative(
            self.pre_target_cursors, request_ids, "pre_target_cursors"
        )
        rng_counters = _aligned_nonnegative(
            self.pre_rng_counters, request_ids, "pre_rng_counters"
        )
        target_checkpoints = _aligned_checkpoints(
            self.target_checkpoint_ids, request_ids, "target_checkpoint_ids"
        )
        target_owner = _required_text(self.target_owner, "target_owner")
        target_mode = SpecTransactionMode(self.target_transaction_mode)
        provider_mode = (
            None
            if self.provider_transaction_mode is None
            else SpecTransactionMode(self.provider_transaction_mode)
        )
        if provider_mode is None:
            if (
                self.provider_owner is not None
                or self.pre_provider_cursors
                or self.provider_request_ids
                or self.provider_checkpoint_ids
                or self.provider_open
                or self.provider_committed
            ):
                raise ValueError("K0 transaction cannot retain provider ownership")
            provider_owner = None
            provider_request_ids: tuple[int, ...] = ()
            provider_cursors: tuple[int, ...] = ()
            provider_checkpoints: tuple[str, ...] = ()
        else:
            provider_owner = _required_text(self.provider_owner, "provider_owner")
            provider_request_ids = (
                request_ids
                if not self.provider_request_ids
                else _request_ids(self.provider_request_ids)
            )
            if any(request_id not in request_ids for request_id in provider_request_ids):
                raise ValueError("provider_request_ids must be a transaction request subset")
            provider_cursors = _aligned_nonnegative(
                self.pre_provider_cursors,
                provider_request_ids,
                "pre_provider_cursors",
            )
            provider_checkpoints = _aligned_checkpoints(
                self.provider_checkpoint_ids,
                provider_request_ids,
                "provider_checkpoint_ids",
            )
            if self.target_committed != self.provider_committed:
                raise ValueError("target/provider commit must be atomic")
        if self.rolled_back and (self.target_committed or self.provider_committed):
            raise ValueError("rolled-back transaction cannot be committed")
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "transaction_id", transaction_id)
        object.__setattr__(self, "cycle_id", cycle_id)
        object.__setattr__(self, "request_ids", request_ids)
        object.__setattr__(self, "pre_target_cursors", target_cursors)
        object.__setattr__(self, "pre_provider_cursors", provider_cursors)
        object.__setattr__(self, "pre_rng_counters", rng_counters)
        object.__setattr__(self, "target_transaction_mode", target_mode)
        object.__setattr__(self, "provider_transaction_mode", provider_mode)
        object.__setattr__(self, "target_owner", target_owner)
        object.__setattr__(self, "provider_owner", provider_owner)
        object.__setattr__(self, "provider_request_ids", provider_request_ids)
        object.__setattr__(self, "target_checkpoint_ids", target_checkpoints)
        object.__setattr__(self, "provider_checkpoint_ids", provider_checkpoints)

    @property
    def has_provider(self) -> bool:
        return self.provider_transaction_mode is not None

    @property
    def committed(self) -> bool:
        return self.target_committed and (
            not self.has_provider or self.provider_committed
        )


@dataclass(frozen=True, slots=True)
class SpecCycleTelemetry:
    """Bounded logical/physical ownership and complete-wall attribution."""

    operation_id: str
    request_ids: tuple[int, ...]
    candidate_counts: tuple[int, ...]
    plan_reasons: tuple[SpecPlanReason, ...]
    proposal_widths: tuple[int, ...]
    target_row_decomposition: tuple[int, ...]
    execution_route: str
    proposal_seconds: float = 0.0
    target_seconds: float = 0.0
    accept_commit_seconds: float = 0.0
    provider_update_seconds: float = 0.0
    scheduler_readback_seconds: float = 0.0
    weight_sweeps: int = 1
    result_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _required_text(self.operation_id, "operation_id"))
        request_ids = _request_ids(self.request_ids)
        counts = _aligned_nonnegative(
            self.candidate_counts, request_ids, "candidate_counts"
        )
        reasons = tuple(SpecPlanReason(reason) for reason in self.plan_reasons)
        if len(reasons) != len(request_ids):
            raise ValueError("plan_reasons must align with request_ids")
        for count, reason in zip(counts, reasons, strict=True):
            if count > 0 and reason is not SpecPlanReason.SPECULATIVE_QUALIFIED:
                raise ValueError("positive candidate counts require speculative-qualified reason")
            if count == 0 and reason is SpecPlanReason.SPECULATIVE_QUALIFIED:
                raise ValueError("speculative-qualified reason requires candidate rows")
        proposal_widths = tuple(int(width) for width in self.proposal_widths)
        speculative_requests = sum(1 for count in counts if count > 0)
        if speculative_requests:
            if not proposal_widths or any(width <= 0 for width in proposal_widths):
                raise ValueError("proposal_widths must contain positive physical groups")
            if sum(proposal_widths) != speculative_requests:
                raise ValueError("proposal_widths must cover speculative requests")
        elif proposal_widths:
            raise ValueError("K0 telemetry cannot contain proposal_widths")
        target_rows = tuple(int(rows) for rows in self.target_row_decomposition)
        if not target_rows or any(rows <= 0 for rows in target_rows):
            raise ValueError("target_row_decomposition must contain positive rows")
        if sum(target_rows) != len(request_ids) + sum(counts):
            raise ValueError("target_row_decomposition must sum to logical frontier rows")
        route = str(self.execution_route)
        if route not in {"ar", "graph", "eager"}:
            raise ValueError("execution_route must be ar, graph, or eager")
        if speculative_requests and route == "ar":
            raise ValueError("speculative telemetry cannot use ar route")
        if not speculative_requests and route != "ar":
            raise ValueError("K0 telemetry must use ar route")
        for field in (
            "proposal_seconds",
            "target_seconds",
            "accept_commit_seconds",
            "provider_update_seconds",
            "scheduler_readback_seconds",
        ):
            value = float(getattr(self, field))
            if value < 0.0:
                raise ValueError(f"{field} must be non-negative")
            object.__setattr__(self, field, value)
        weight_sweeps = int(self.weight_sweeps)
        result_bytes = int(self.result_bytes)
        if weight_sweeps <= 0 or result_bytes < 0:
            raise ValueError("weight_sweeps must be positive and result_bytes non-negative")
        object.__setattr__(self, "request_ids", request_ids)
        object.__setattr__(self, "candidate_counts", counts)
        object.__setattr__(self, "plan_reasons", reasons)
        object.__setattr__(self, "proposal_widths", proposal_widths)
        object.__setattr__(self, "target_row_decomposition", target_rows)
        object.__setattr__(self, "execution_route", route)
        object.__setattr__(self, "weight_sweeps", weight_sweeps)
        object.__setattr__(self, "result_bytes", result_bytes)

    @property
    def logical_request_count(self) -> int:
        return len(self.request_ids)

    @property
    def logical_frontier_rows(self) -> int:
        return len(self.request_ids) + sum(self.candidate_counts)

    @property
    def complete_seconds(self) -> float:
        return (
            self.proposal_seconds
            + self.target_seconds
            + self.accept_commit_seconds
            + self.provider_update_seconds
            + self.scheduler_readback_seconds
        )


_TERMINAL_STAGES = {
    SpecCycleStage.COMMITTED,
    SpecCycleStage.ROLLED_BACK,
    SpecCycleStage.CANCELLED,
    SpecCycleStage.FAILED,
}


@dataclass(frozen=True, slots=True)
class SpecCycleResult:
    """One terminal scheduler-owned cycle result."""

    stage: SpecCycleStage
    transaction: SpecCycleTransaction
    accept_result: AcceptResult | None = None
    committed_output_ids: tuple[tuple[int, ...], ...] = ()
    committed_output_lengths: tuple[int, ...] = ()
    selected_rows: tuple[int | None, ...] = ()
    target_cursor_deltas: tuple[int, ...] = ()
    provider_cursor_deltas: tuple[int, ...] = ()
    finish_reasons: tuple[str | None, ...] = ()
    cancelled_request_ids: tuple[int, ...] = ()
    telemetry: SpecCycleTelemetry | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        stage = SpecCycleStage(self.stage)
        if stage not in _TERMINAL_STAGES:
            raise ValueError("SpecCycleResult stage must be terminal")
        if not isinstance(self.transaction, SpecCycleTransaction):
            raise TypeError("transaction must be SpecCycleTransaction")
        request_ids = self.transaction.request_ids
        if stage is SpecCycleStage.COMMITTED:
            if not self.transaction.committed:
                raise ValueError("committed result requires committed transaction")
        elif not self.transaction.rolled_back:
            raise ValueError("non-committed result requires rolled-back transaction")
        if stage is SpecCycleStage.FAILED:
            object.__setattr__(self, "error", _required_text(self.error, "error"))
        elif self.error is not None:
            raise ValueError("only failed cycle results may carry error")
        if self.accept_result is not None:
            if self.accept_result.request_ids != request_ids:
                raise ValueError("accept_result request_ids must match transaction")
            if self.accept_result.transaction_id != self.transaction.transaction_id:
                raise ValueError("accept_result transaction_id must match transaction")
        aligned = (
            ("committed_output_ids", self.committed_output_ids),
            ("committed_output_lengths", self.committed_output_lengths),
            ("selected_rows", self.selected_rows),
            ("target_cursor_deltas", self.target_cursor_deltas),
            ("provider_cursor_deltas", self.provider_cursor_deltas),
            ("finish_reasons", self.finish_reasons),
        )
        for label, values in aligned:
            if values and len(values) != len(request_ids):
                raise ValueError(f"{label} must align with request_ids")
        if self.committed_output_ids:
            normalized_outputs = tuple(
                tuple(int(token) for token in tokens)
                for tokens in self.committed_output_ids
            )
            if any(token < 0 for tokens in normalized_outputs for token in tokens):
                raise ValueError("committed output token ids must be non-negative")
            if self.committed_output_lengths and tuple(map(len, normalized_outputs)) != tuple(
                int(length) for length in self.committed_output_lengths
            ):
                raise ValueError("committed_output_lengths must match committed_output_ids")
            object.__setattr__(self, "committed_output_ids", normalized_outputs)
        for label, values in (
            ("committed_output_lengths", self.committed_output_lengths),
            ("target_cursor_deltas", self.target_cursor_deltas),
            ("provider_cursor_deltas", self.provider_cursor_deltas),
        ):
            if any(int(value) < 0 for value in values):
                raise ValueError(f"{label} must be non-negative")
            object.__setattr__(self, label, tuple(int(value) for value in values))
        selected = tuple(
            None if row is None else int(row) for row in self.selected_rows
        )
        if any(row is not None and row < 0 for row in selected):
            raise ValueError("selected_rows must be non-negative")
        object.__setattr__(self, "selected_rows", selected)
        cancelled = tuple(int(request_id) for request_id in self.cancelled_request_ids)
        if any(request_id not in request_ids for request_id in cancelled):
            raise ValueError("cancelled_request_ids must belong to transaction")
        if len(cancelled) != len(set(cancelled)):
            raise ValueError("cancelled_request_ids must be unique")
        object.__setattr__(self, "cancelled_request_ids", cancelled)
        if self.telemetry is not None:
            if self.telemetry.operation_id != self.transaction.operation_id:
                raise ValueError("telemetry operation_id must match transaction")
            if self.telemetry.request_ids != request_ids:
                raise ValueError("telemetry request_ids must match transaction")
        object.__setattr__(self, "stage", stage)

    @classmethod
    def committed(
        cls,
        transaction: SpecCycleTransaction,
        accept_result: AcceptResult,
        *,
        telemetry: SpecCycleTelemetry | None = None,
    ) -> "SpecCycleResult":
        corrections = accept_result.correction_or_bonus_tokens or (
            (None,) * len(accept_result.request_ids)
        )
        outputs = tuple(
            (*tokens, *(() if correction is None else (int(correction),)))
            for tokens, correction in zip(
                accept_result.accepted_tokens, corrections, strict=True
            )
        )
        target_deltas = accept_result.target_cursor_deltas or tuple(map(len, outputs))
        provider_deltas = accept_result.provider_cursor_deltas or accept_result.accepted_counts
        selected = accept_result.selected_candidate_rows or (
            (None,) * len(accept_result.request_ids)
        )
        finish_reasons = accept_result.finish_reasons or (
            (None,) * len(accept_result.request_ids)
        )
        return cls(
            stage=SpecCycleStage.COMMITTED,
            transaction=transaction,
            accept_result=accept_result,
            committed_output_ids=outputs,
            committed_output_lengths=tuple(map(len, outputs)),
            selected_rows=selected,
            target_cursor_deltas=target_deltas,
            provider_cursor_deltas=provider_deltas,
            finish_reasons=finish_reasons,
            telemetry=telemetry,
        )


__all__ = [
    "compose_speculative_claims",
    "SpecCycleResult",
    "SpecCycleStage",
    "SpecCycleTelemetry",
    "SpecCycleTransaction",
]
