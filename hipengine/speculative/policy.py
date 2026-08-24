"""Deterministic pre-mutation K/K0 planner for SPECDEC2."""

from __future__ import annotations

from typing import Sequence

from hipengine.speculative.frontier import (
    SpecPlanReason,
    SpecRequestPlan,
    SpecTransactionMode,
    SpeculativeCapability,
)
from hipengine.speculative.provider import SpeculativeRequestSemantics


def _physical_groups(total: int, installed_widths: Sequence[int]) -> tuple[int, ...]:
    """Decompose real rows deterministically under installed width ceilings.

    Returned values are real rows in each physical group.  A backend may lower a
    final real-width remainder into its narrowest fitting padded graph; graph
    bucket identity is a later backend capability, not hidden in logical R.
    """

    remaining = int(total)
    if remaining <= 0:
        return ()
    widths = tuple(sorted({int(width) for width in installed_widths if int(width) > 0}, reverse=True))
    if not widths:
        raise ValueError("installed_widths must contain a positive width")
    groups: list[int] = []
    maximum = widths[0]
    while remaining:
        if remaining > maximum:
            groups.append(maximum)
            remaining -= maximum
            continue
        exact_or_smaller = next((width for width in widths if width <= remaining), None)
        if exact_or_smaller is None:
            groups.append(remaining)
            break
        groups.append(exact_or_smaller)
        remaining -= exact_or_smaller
    return tuple(groups)


def _execution_route(
    capability: SpeculativeCapability,
    *,
    prefer_graph: bool,
    graph_available: bool,
    target_physical_available: bool,
) -> str | None:
    if not target_physical_available:
        return None
    if prefer_graph and capability.graph_supported and graph_available:
        return "graph"
    if capability.eager_supported:
        return "eager"
    if capability.graph_supported and graph_available:
        return "graph"
    return None


def plan_speculative_requests(
    capability: SpeculativeCapability | None,
    request_semantics: Sequence[SpeculativeRequestSemantics],
    *,
    resident_slots: Sequence[int],
    desired_candidate_counts: Sequence[int],
    operation_id: str,
    cycle_id: int,
    context_bucket_size: int,
    claims_fit: bool = True,
    circuit_breaker_open: bool = False,
    graph_available: bool = True,
    target_physical_available: bool = True,
    prefer_graph: bool = True,
    ar_target_widths: Sequence[int] = (1, 2, 4, 8),
) -> SpecRequestPlan:
    """Choose K or K0 for every request before opening any mutable owner."""

    semantics = tuple(request_semantics)
    if not semantics:
        raise ValueError("request_semantics must be non-empty")
    request_ids = tuple(row.request_id for row in semantics)
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request_semantics request ids must be unique")
    slots = tuple(int(slot) for slot in resident_slots)
    if len(slots) != len(semantics) or any(slot < 0 for slot in slots) or len(slots) != len(set(slots)):
        raise ValueError("resident_slots must be non-negative, unique, and aligned")
    desired = tuple(int(count) for count in desired_candidate_counts)
    if len(desired) != len(semantics) or any(count < 0 for count in desired):
        raise ValueError("desired_candidate_counts must be non-negative and aligned")
    quantum = int(context_bucket_size)
    if quantum <= 0:
        raise ValueError("context_bucket_size must be positive")
    context_bucket = (
        (max(row.context_tokens for row in semantics) + quantum - 1) // quantum
    ) * quantum

    if capability is None:
        counts = (0,) * len(semantics)
        reasons = tuple(
            SpecPlanReason.POLICY_SELECTED_AR if count == 0 else SpecPlanReason.NO_PROVIDER
            for count in desired
        )
        return SpecRequestPlan(
            operation_id=operation_id,
            cycle_id=cycle_id,
            request_ids=request_ids,
            resident_slots=slots,
            candidate_counts=counts,
            reasons=reasons,
            mode="decode",
            capability_key=None,
            provider_key=None,
            target_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
            provider_transaction_mode=None,
            proposal_widths=(),
            target_row_decomposition=_physical_groups(len(semantics), ar_target_widths),
            context_bucket_size=context_bucket,
            execution_route="ar",
        )

    route = _execution_route(
        capability,
        prefer_graph=bool(prefer_graph),
        graph_available=bool(graph_available),
        target_physical_available=bool(target_physical_available),
    )
    selected_mode = semantics[0].mode
    counts_list: list[int] = []
    reasons_list: list[SpecPlanReason] = []
    for row, desired_count in zip(semantics, desired, strict=True):
        if desired_count == 0:
            counts_list.append(0)
            reasons_list.append(SpecPlanReason.POLICY_SELECTED_AR)
            continue
        if circuit_breaker_open:
            counts_list.append(0)
            reasons_list.append(SpecPlanReason.CIRCUIT_BREAKER_OPEN)
            continue
        if not claims_fit:
            counts_list.append(0)
            reasons_list.append(SpecPlanReason.RESOURCE_CLAIM_MISS)
            continue
        if not capability.supports_sampling(row.sampling_mode):
            counts_list.append(0)
            reasons_list.append(SpecPlanReason.UNSUPPORTED_SAMPLING)
            continue
        if row.mode != selected_mode or row.mode not in capability.supported_modes:
            counts_list.append(0)
            reasons_list.append(SpecPlanReason.TARGET_PHYSICAL_BUCKET_MISS)
            continue
        if route is None:
            counts_list.append(0)
            reasons_list.append(SpecPlanReason.TARGET_PHYSICAL_BUCKET_MISS)
            continue
        output_room = row.remaining_decode - 1
        if output_room <= 0:
            counts_list.append(0)
            reasons_list.append(SpecPlanReason.TARGET_GRAPH_OUTPUT_ROOM_MISS)
            continue
        context_room = capability.max_candidates_per_request
        if capability.max_context_tokens is not None:
            context_room = capability.max_context_tokens - row.context_tokens - 1
            if context_room <= 0:
                counts_list.append(0)
                reasons_list.append(SpecPlanReason.TARGET_GRAPH_CONTEXT_BUCKET_MISS)
                continue
        selected = min(
            desired_count,
            capability.max_candidates_per_request,
            output_room,
            context_room,
        )
        if selected <= 0:
            counts_list.append(0)
            reasons_list.append(SpecPlanReason.TARGET_PHYSICAL_BUCKET_MISS)
            continue
        counts_list.append(selected)
        reasons_list.append(SpecPlanReason.SPECULATIVE_QUALIFIED)

    counts = tuple(counts_list)
    reasons = tuple(reasons_list)
    has_spec = any(counts)
    if has_spec and not capability.supports_shape(
        request_count=len(semantics),
        candidate_counts=counts,
        mode=selected_mode,
    ):
        counts = tuple(0 for _count in counts)
        reasons = tuple(
            SpecPlanReason.POLICY_SELECTED_AR
            if desired_count == 0
            else SpecPlanReason.TARGET_PHYSICAL_BUCKET_MISS
            for desired_count in desired
        )
        has_spec = False

    if not has_spec:
        return SpecRequestPlan(
            operation_id=operation_id,
            cycle_id=cycle_id,
            request_ids=request_ids,
            resident_slots=slots,
            candidate_counts=counts,
            reasons=reasons,
            mode="decode",
            capability_key=None,
            provider_key=None,
            target_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
            provider_transaction_mode=None,
            proposal_widths=(),
            target_row_decomposition=_physical_groups(len(semantics), ar_target_widths),
            context_bucket_size=context_bucket,
            execution_route="ar",
        )

    speculative_requests = sum(1 for count in counts if count > 0)
    logical_rows = len(semantics) + sum(counts)
    return SpecRequestPlan(
        operation_id=operation_id,
        cycle_id=cycle_id,
        request_ids=request_ids,
        resident_slots=slots,
        candidate_counts=counts,
        reasons=reasons,
        mode=selected_mode,
        capability_key=capability.capability_key,
        provider_key=capability.provider_key,
        target_transaction_mode=capability.target_transaction_mode,
        provider_transaction_mode=capability.provider_transaction_mode,
        proposal_widths=_physical_groups(
            speculative_requests, capability.proposal_widths
        ),
        target_row_decomposition=_physical_groups(
            logical_rows, capability.target_row_buckets
        ),
        context_bucket_size=context_bucket,
        execution_route=route or "eager",
    )


__all__ = ["plan_speculative_requests"]
