"""Exact compatibility grouping and physical-width lowering for C2 rounds."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter_ns

from hipengine.dispatch.batch import (
    PhysicalBatchGroup,
    WorkItem,
    plan_physical_batch_groups,
)


@dataclass(frozen=True, slots=True)
class ExecutionCompatibilityKey:
    backend_key: str
    layout_key: str
    kernel_bundle_key: str
    work_class: str
    context_bucket: str
    workspace_key: str
    physical_widths: tuple[int, ...]
    supports_masked_rows: bool = True
    supports_dense_compaction: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "backend_key",
            "layout_key",
            "kernel_bundle_key",
            "work_class",
            "context_bucket",
            "workspace_key",
        ):
            value = str(getattr(self, field_name))
            if not value or value != value.strip():
                raise ValueError(f"{field_name} must be a non-empty trimmed string")
            object.__setattr__(self, field_name, value)
        widths = tuple(int(width) for width in self.physical_widths)
        if not widths or widths[0] != 1:
            raise ValueError("physical_widths must retain a c1 route")
        if tuple(sorted(set(widths))) != widths:
            raise ValueError("physical_widths must be unique and strictly increasing")
        object.__setattr__(self, "physical_widths", widths)


@dataclass(frozen=True, slots=True)
class PlannedExecutionGroup:
    compatibility_key: ExecutionCompatibilityKey
    work: WorkItem
    physical_groups: tuple[PhysicalBatchGroup, ...]
    execution_path: str

    def __post_init__(self) -> None:
        if self.execution_path not in {
            "registered_masked_or_exact",
            "registered_dense_compaction",
            "serial_c1_fallback",
        }:
            raise ValueError("unknown execution_path")
        flattened = tuple(
            request_id
            for group in self.physical_groups
            for request_id in group.request_ids
        )
        if flattened != self.work.request_ids:
            raise ValueError("physical groups must preserve grouped request order exactly once")


@dataclass(frozen=True, slots=True)
class TokenBudgetSLO:
    prefill_token_budget: int
    decode_row_budget: int
    ttft_target_ms: float
    itl_target_ms: float

    def __post_init__(self) -> None:
        if self.prefill_token_budget <= 0 or self.decode_row_budget <= 0:
            raise ValueError("token-budget round capacities must be positive")
        if self.ttft_target_ms <= 0 or self.itl_target_ms <= 0:
            raise ValueError("TTFT and ITL targets must be positive")

    @classmethod
    def from_targets(
        cls,
        *,
        ttft_target_ms: float,
        itl_target_ms: float,
        measured_prefill_tokens_per_ms: float,
        measured_decode_rows_per_ms: float,
        max_prefill_chunk_tokens: int,
        resident_capacity: int,
    ) -> "TokenBudgetSLO":
        if measured_prefill_tokens_per_ms <= 0 or measured_decode_rows_per_ms <= 0:
            raise ValueError("measured scheduler rates must be positive")
        if max_prefill_chunk_tokens <= 0 or resident_capacity <= 0:
            raise ValueError("scheduler capacity inputs must be positive")
        prefill_budget = min(
            int(max_prefill_chunk_tokens),
            max(1, int(float(ttft_target_ms) * measured_prefill_tokens_per_ms)),
        )
        decode_budget = min(
            int(resident_capacity),
            max(1, int(float(itl_target_ms) * measured_decode_rows_per_ms)),
        )
        return cls(
            prefill_token_budget=prefill_budget,
            decode_row_budget=decode_budget,
            ttft_target_ms=float(ttft_target_ms),
            itl_target_ms=float(itl_target_ms),
        )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    groups: tuple[PlannedExecutionGroup, ...]
    logical_rows: int
    planner_duration_ns: int

    def __post_init__(self) -> None:
        if self.logical_rows <= 0:
            raise ValueError("logical_rows must be positive")
        if self.planner_duration_ns < 0:
            raise ValueError("planner_duration_ns must be non-negative")
        planned_rows = sum(len(group.work.request_ids) for group in self.groups)
        if planned_rows != self.logical_rows:
            raise ValueError("execution plan must cover every logical row once")


def plan_execution_groups(
    work: WorkItem,
    *,
    key_resolver: Callable[[int], ExecutionCompatibilityKey],
) -> ExecutionPlan:
    """Partition by exact key, then lower stable slots into certified widths."""

    started = perf_counter_ns()
    grouped: OrderedDict[ExecutionCompatibilityKey, list[int]] = OrderedDict()
    for request_id in work.request_ids:
        key = key_resolver(int(request_id))
        if not isinstance(key, ExecutionCompatibilityKey):
            raise TypeError("key_resolver must return ExecutionCompatibilityKey")
        if key.work_class != work.kind.value:
            raise ValueError("execution compatibility work_class does not match work item")
        grouped.setdefault(key, []).append(int(request_id))

    plans: list[PlannedExecutionGroup] = []
    for key, request_ids in grouped.items():
        subset = _subset_work(work, tuple(request_ids))
        if key.supports_masked_rows:
            physical = plan_physical_batch_groups(
                subset,
                physical_bucket_widths=key.physical_widths,
            )
            path = "registered_masked_or_exact"
        elif key.supports_dense_compaction:
            physical = plan_physical_batch_groups(
                subset,
                physical_bucket_widths=key.physical_widths,
                compact_active_rows=True,
            )
            path = "registered_dense_compaction"
        else:
            serial_groups: list[PhysicalBatchGroup] = []
            for request_id in subset.request_ids:
                singleton = _subset_work(subset, (request_id,))
                serial_groups.extend(
                    plan_physical_batch_groups(
                        singleton,
                        physical_bucket_widths=(1,),
                        compact_active_rows=True,
                    )
                )
            physical = tuple(
                replace(group, group_index=index, group_count=len(serial_groups))
                for index, group in enumerate(serial_groups)
            )
            path = "serial_c1_fallback"
        plans.append(
            PlannedExecutionGroup(
                compatibility_key=key,
                work=subset,
                physical_groups=physical,
                execution_path=path,
            )
        )
    return ExecutionPlan(
        groups=tuple(plans),
        logical_rows=len(work.request_ids),
        planner_duration_ns=perf_counter_ns() - started,
    )


def _subset_work(work: WorkItem, request_ids: tuple[int, ...]) -> WorkItem:
    wanted = set(request_ids)
    if not wanted or len(wanted) != len(request_ids):
        raise ValueError("execution group request_ids must be non-empty and unique")
    if not wanted.issubset(work.request_ids):
        raise KeyError("execution group references a request outside the work item")
    row_indices = tuple(
        index
        for index, request_id in enumerate(work.row_to_request)
        if request_id in wanted
    )
    row_to_request = tuple(work.row_to_request[index] for index in row_indices)
    token_rows = (
        tuple(work.token_rows[index] for index in row_indices)
        if work.token_rows
        else ()
    )
    slot_by_request = dict(zip(work.request_ids, work.slot_ids, strict=True)) if work.slot_ids else {}
    slot_ids = tuple(slot_by_request[request_id] for request_id in request_ids) if slot_by_request else ()
    if work.active_mask and slot_ids:
        selected_slots = set(slot_ids)
        active_mask = tuple(
            slot in selected_slots for slot in range(len(work.active_mask))
        )
    else:
        active_mask = ()
    return WorkItem(
        kind=work.kind,
        request_ids=request_ids,
        row_to_request=row_to_request,
        token_rows=token_rows,
        draft_depth=work.draft_depth,
        tree_parents=work.tree_parents,
        slot_ids=slot_ids,
        active_mask=active_mask,
        declared_logical_c=work.declared_logical_c,
    )


__all__ = [
    "ExecutionCompatibilityKey",
    "ExecutionPlan",
    "PlannedExecutionGroup",
    "TokenBudgetSLO",
    "plan_execution_groups",
]
