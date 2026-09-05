"""Deterministic pre-mutation K/K0 planner for SPECDEC2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

from hipengine.speculative.frontier import (
    SpecK0Class,
    SpecPlanReason,
    SpecRequestPlan,
    SpecTransactionMode,
    SpeculativeCapability,
)
from hipengine.speculative.provider import SpeculativeRequestSemantics


@dataclass(frozen=True, slots=True)
class OfflineSpeculativeDepthCell:
    """Prompt-independent automatic K/K0 cell selected before mutation."""

    cell_key: str
    min_concurrency: int
    max_concurrency: int
    selected_k: int
    reason: str
    evidence: str

    def __post_init__(self) -> None:
        key = str(self.cell_key).strip()
        reason = str(self.reason).strip()
        evidence = str(self.evidence).strip()
        lower = int(self.min_concurrency)
        upper = int(self.max_concurrency)
        selected = int(self.selected_k)
        if not key or not reason or not evidence:
            raise ValueError("offline depth cell text fields must be non-empty")
        if lower <= 0 or upper < lower:
            raise ValueError("offline depth cell concurrency range is invalid")
        if selected < 0:
            raise ValueError("offline depth cell selected_k must be non-negative")
        object.__setattr__(self, "cell_key", key)
        object.__setattr__(self, "min_concurrency", lower)
        object.__setattr__(self, "max_concurrency", upper)
        object.__setattr__(self, "selected_k", selected)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "evidence", evidence)

    def matches(self, concurrency: int) -> bool:
        value = int(concurrency)
        return self.min_concurrency <= value <= self.max_concurrency


@dataclass(frozen=True, slots=True)
class OfflineSpeculativeDepthPolicy:
    """Ordered immutable cells with a canonical content fingerprint."""

    policy_key: str
    cells: tuple[OfflineSpeculativeDepthCell, ...]

    def __post_init__(self) -> None:
        key = str(self.policy_key).strip()
        cells = tuple(self.cells)
        if not key or not cells:
            raise ValueError("offline depth policy requires a key and cells")
        if len({cell.cell_key for cell in cells}) != len(cells):
            raise ValueError("offline depth policy cell keys must be unique")
        prior = 0
        for cell in cells:
            if cell.min_concurrency <= prior:
                raise ValueError("offline depth policy cells must be ordered and disjoint")
            prior = cell.max_concurrency
        object.__setattr__(self, "policy_key", key)
        object.__setattr__(self, "cells", cells)

    @property
    def fingerprint(self) -> str:
        payload = {
            "policy_key": self.policy_key,
            "cells": [
                {
                    "cell_key": cell.cell_key,
                    "min_concurrency": cell.min_concurrency,
                    "max_concurrency": cell.max_concurrency,
                    "selected_k": cell.selected_k,
                    "reason": cell.reason,
                    "evidence": cell.evidence,
                }
                for cell in self.cells
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class OfflineSpeculativeDepthDecision:
    selected_k: int
    cell_key: str
    reason: str
    evidence: str
    policy_fingerprint: str
    concurrency: int
    output_horizon_tokens: int


_P9_FIXED_POLICY_EVIDENCE = (
    "benchmarks/results/2026-08-26-gfx1151-specdec2-perf-p9-fixed-policy.json"
)


DEFAULT_AUTO_DEPTH_POLICY = OfflineSpeculativeDepthPolicy(
    policy_key="specdec2:auto:qwen38-q4ks:production:p9-fixed-reseed:v4",
    cells=(
        OfflineSpeculativeDepthCell(
            "auto-c1-product-pending-k0",
            1,
            1,
            0,
            "product_qualification_pending",
            _P9_FIXED_POLICY_EVIDENCE,
        ),
        OfflineSpeculativeDepthCell(
            "auto-c2-measured-k0",
            2,
            2,
            0,
            "measured_speedup_below_1p10",
            _P9_FIXED_POLICY_EVIDENCE,
        ),
        OfflineSpeculativeDepthCell(
            "auto-c3-unqualified-k0",
            3,
            3,
            0,
            "no_qualified_physical_frontier",
            _P9_FIXED_POLICY_EVIDENCE,
        ),
        OfflineSpeculativeDepthCell(
            "auto-c4-measured-k0",
            4,
            4,
            0,
            "measured_speedup_below_1p10",
            _P9_FIXED_POLICY_EVIDENCE,
        ),
        OfflineSpeculativeDepthCell(
            "auto-c5-c8-unqualified-k0",
            5,
            8,
            0,
            "no_qualified_physical_frontier",
            _P9_FIXED_POLICY_EVIDENCE,
        ),
        OfflineSpeculativeDepthCell(
            "auto-c9-c17-unqualified-k0",
            9,
            17,
            0,
            "no_qualified_physical_frontier",
            _P9_FIXED_POLICY_EVIDENCE,
        ),
        OfflineSpeculativeDepthCell(
            "auto-c18-c32-unqualified-k0",
            18,
            32,
            0,
            "no_qualified_physical_frontier",
            _P9_FIXED_POLICY_EVIDENCE,
        ),
    ),
)


def select_offline_speculative_depth(
    policy: OfflineSpeculativeDepthPolicy,
    *,
    concurrency: int,
    output_horizon_tokens: int,
) -> OfflineSpeculativeDepthDecision:
    """Select one deterministic cell using shape only, never prompt content."""

    count = int(concurrency)
    horizon = int(output_horizon_tokens)
    if count <= 0 or horizon < 0:
        raise ValueError("concurrency must be positive and output horizon non-negative")
    cell = next((candidate for candidate in policy.cells if candidate.matches(count)), None)
    if cell is None:
        return OfflineSpeculativeDepthDecision(
            selected_k=0,
            cell_key="auto-outside-qualified-concurrency-k0",
            reason="outside_qualified_concurrency",
            evidence=policy.cells[-1].evidence,
            policy_fingerprint=policy.fingerprint,
            concurrency=count,
            output_horizon_tokens=horizon,
        )
    selected = min(int(cell.selected_k), max(0, horizon - 1))
    return OfflineSpeculativeDepthDecision(
        selected_k=selected,
        cell_key=cell.cell_key,
        reason=cell.reason,
        evidence=cell.evidence,
        policy_fingerprint=policy.fingerprint,
        concurrency=count,
        output_horizon_tokens=horizon,
    )


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
    suppress_speculation: Sequence[bool] = (),
    declared_logical_c: int = 0,
) -> SpecRequestPlan:
    """Choose K or K0 for every request before opening any mutable owner.

    ``suppress_speculation`` optionally forces a request to a K0-transitional
    cycle (desired count preserved, planned count zero) so its provider state
    is repaired through the target-hidden catchup instead of immediately
    re-speculating.
    """

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
    suppression = tuple(bool(flag) for flag in suppress_speculation)
    if suppression and len(suppression) != len(semantics):
        raise ValueError("suppress_speculation must align with request_semantics")
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
            k0_classes=tuple(
                SpecK0Class.PURE
                if desired_count == 0
                else SpecK0Class.TRANSITIONAL
                for desired_count in desired
            ),
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
    for row_index, (row, desired_count) in enumerate(zip(semantics, desired, strict=True)):
        if desired_count == 0:
            counts_list.append(0)
            reasons_list.append(SpecPlanReason.POLICY_SELECTED_AR)
            continue
        if len(suppression) > row_index and suppression[row_index]:
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
            if bool(capability.terminal_zero_accept_supported):
                output_room = 1
            else:
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
            k0_classes=tuple(
                SpecK0Class.PURE
                if desired_count == 0
                else SpecK0Class.TRANSITIONAL
                for desired_count in desired
            ),
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
        k0_classes=tuple(
            SpecK0Class.NOT_K0
            if count > 0
            else SpecK0Class.PURE
            if desired_count == 0
            else SpecK0Class.TRANSITIONAL
            for count, desired_count in zip(counts, desired, strict=True)
        ),
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
        declared_logical_c=declared_logical_c,
    )


__all__ = [
    "DEFAULT_AUTO_DEPTH_POLICY",
    "OfflineSpeculativeDepthCell",
    "OfflineSpeculativeDepthDecision",
    "OfflineSpeculativeDepthPolicy",
    "plan_speculative_requests",
    "select_offline_speculative_depth",
]
