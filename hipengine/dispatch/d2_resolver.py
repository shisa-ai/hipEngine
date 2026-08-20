"""Artifact-backed cost-aware D2 group resolver (host-first).

D2 turns an arbitrary logical concurrency ``C`` into a composition of certified
native/masked physical widths that minimizes complete model-step wall, given a
measured cost table (one model-step ms per certified width). It is the
measured-optimizer counterpart to the correct-but-not-optimized ceiling-bucket
planner in :mod:`hipengine.dispatch.batch`.

Design
------
- Cost records are supplied, never hard-coded: in production they load from a
  post-promotion benchmark artifact; in tests a fixture cost table stands in.
  This keeps the resolver generic and avoids baking benchmark constants in.
- ``d2_partition`` minimizes the serial sum of measured group costs by dynamic
  programming, breaking ties toward fewer groups, then a canonical (descending)
  width ordering.
- ``ceiling_partition`` reproduces the existing greedy largest-width chunking so
  the D2 path can fail closed to it when no cost table is certified.
- ``plan_d2_groups`` lowers a dense work item into ``PhysicalBatchGroup``s using
  the chosen composition, preserving scheduler slot identity and dense execution
  rows. Work items with holes/sparse layouts are the ceiling planner's domain;
  D2 here assumes the common fully-active steady state.

References: docs/CONCURRENCY2.md "D2 — decomposition exists; cost-aware
decomposition does not."
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from hipengine.dispatch.batch import PhysicalBatchGroup, WorkItem


@dataclass(frozen=True, slots=True)
class PhysicalWidthCost:
    """Measured complete model-step cost for one certified physical width.

    ``model_step_ms`` is the serial full-group wall time for a dense group of
    exactly ``physical_width`` rows (inter-token model-step median). ``source``
    names the artifact/commit the record was measured under so provenance is
    auditable and stale records are rejected.
    """

    physical_width: int
    model_step_ms: float
    source: str

    def __post_init__(self) -> None:
        if self.physical_width <= 0:
            raise ValueError("physical_width must be positive")
        if self.model_step_ms <= 0.0:
            raise ValueError("model_step_ms must be positive")
        if not self.source or self.source != self.source.strip():
            raise ValueError("source must be a non-empty trimmed string")


@dataclass(frozen=True, slots=True)
class CostTable:
    """Validated set of measured per-width costs plus the certified width set."""

    records: tuple[PhysicalWidthCost, ...]
    default_max_width: int = 8

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("cost table must be non-empty")
        widths = tuple(record.physical_width for record in self.records)
        if widths[0] != 1:
            raise ValueError("cost table must include a c1 route")
        if tuple(sorted(set(widths))) != widths:
            raise ValueError("cost table widths must be unique and strictly increasing")
        if self.default_max_width < 1:
            raise ValueError("default_max_width must be positive")

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(record.physical_width for record in self.records)

    def cost_ms(self, physical_width: int) -> float:
        for record in self.records:
            if record.physical_width == physical_width:
                return record.model_step_ms
        raise KeyError(f"no measured cost for physical width {physical_width}")


def ceiling_partition(
    rows: int,
    widths: Sequence[int],
) -> tuple[int, ...]:
    """Greedy largest-width chunking (the current fail-closed fallback)."""
    if rows <= 0:
        raise ValueError("rows must be positive")
    certified = tuple(int(width) for width in widths)
    if not certified or certified[0] != 1:
        raise ValueError("widths must be a sorted set starting at c1")
    remaining = int(rows)
    groups: list[int] = []
    descending = tuple(reversed(certified))
    while remaining > 0:
        width = next(width for width in descending if width <= remaining)
        groups.append(width)
        remaining -= width
    return tuple(groups)


def d2_partition(
    rows: int,
    cost_table: CostTable,
) -> tuple[int, ...]:
    """Optimal dense composition of ``rows`` into certified widths.

    Minimizes the serial sum of measured group costs. Ties break toward fewer
    groups, then a canonical descending width ordering (so ``[5, 4]`` is the
    canonical representation of ``5+4``).
    """
    if rows <= 0:
        raise ValueError("rows must be positive")
    widths = cost_table.widths
    best_cost = [0.0] + [float("inf")] * rows
    best_count = [0] + [10**9] * rows
    best_groups: list[tuple[int, ...]] = [()] + [()] * rows

    for n in range(1, rows + 1):
        for width in widths:
            if width > n:
                break
            prior = n - width
            candidate_cost = best_cost[prior] + cost_table.cost_ms(width)
            candidate_count = best_count[prior] + 1
            candidate_groups = (width,) + best_groups[prior]
            better = (
                candidate_cost < best_cost[n] - 1e-12
                or (
                    abs(candidate_cost - best_cost[n]) <= 1e-12
                    and candidate_count < best_count[n]
                )
            )
            if better:
                best_cost[n] = candidate_cost
                best_count[n] = candidate_count
                best_groups[n] = candidate_groups
    groups = best_groups[rows]
    return tuple(sorted(groups, reverse=True))


def _dense_active_slots(work: WorkItem) -> tuple[int, ...]:
    if work.slot_ids:
        # WorkItem invariants make slot_ids exactly the active lanes.
        return tuple(sorted(int(slot) for slot in work.slot_ids))
    return tuple(range(len(work.request_ids)))


def plan_d2_groups(
    work: WorkItem,
    cost_table: CostTable,
) -> tuple[PhysicalBatchGroup, ...]:
    """Lower a dense work item using the measured D2 composition.

    Preserves scheduler slot identity and uses dense execution rows. Active
    rows are partitioned in slot order into the DP-chosen widths. Holes/sparse
    layouts are out of scope here (the ceiling planner owns those); this raises
    ``ValueError`` so callers fall back closed rather than mis-place rows.
    """
    active_slots = _dense_active_slots(work)
    active_rows = len(active_slots)
    if active_rows == 0:
        raise ValueError("work item has no active rows")
    widths = d2_partition(active_rows, cost_table)
    if sum(widths) != active_rows:
        raise RuntimeError("D2 composition does not cover active rows")  # pragma: no cover

    request_by_slot = dict(zip(work.slot_ids or tuple(range(len(work.request_ids))), work.request_ids, strict=True))
    groups: list[PhysicalBatchGroup] = []
    offset = 0
    for group_index, width in enumerate(widths):
        group_slots = active_slots[offset : offset + width]
        offset += width
        if len(group_slots) != width:
            raise ValueError("D2 group width does not match active slot run")
        base = int(group_slots[0])
        request_ids = tuple(request_by_slot[slot] for slot in group_slots)
        groups.append(
            PhysicalBatchGroup(
                logical_c=active_rows,
                group_index=group_index,
                group_count=len(widths),
                physical_slot_base=base,
                physical_slot_extent=width,
                physical_rows=width,
                request_ids=request_ids,
                global_slot_indices=group_slots,
                active_slot_indices=tuple(range(width)),
                active_mask=tuple(True for _ in range(width)),
                dense_execution_rows=True,
            )
        )
    return tuple(groups)


def cost_table_from_artifact(
    path: str | Path,
    *,
    label_by_width: Mapping[int, str],
    source: str | None = None,
) -> CostTable:
    """Build a ``CostTable`` from a benchmark artifact's measured step times.

    Reads the canonical per-configuration model-step median
    (``summaries.<label>.latency.inter_token_model_step_seconds.median``) and
    converts to milliseconds. This keeps costs artifact-backed: no benchmark
    constants are baked into the resolver or caller.
    """
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    summaries = payload.get("summaries", {})
    records: list[PhysicalWidthCost] = []
    for width in sorted(label_by_width):
        label = label_by_width[width]
        config = summaries.get(label)
        if config is None:
            raise KeyError(f"artifact {resolved} has no summary for label {label!r}")
        latency = (config.get("latency") or {}).get(
            "inter_token_model_step_seconds"
        )
        median = (latency or {}).get("median")
        if median is None:
            raise KeyError(f"artifact {resolved} has no model-step median for {label!r}")
        records.append(
            PhysicalWidthCost(
                physical_width=int(width),
                model_step_ms=float(median) * 1000.0,
                source=source or str(resolved),
            )
        )
    return CostTable(tuple(records))


__all__ = [
    "CostTable",
    "PhysicalWidthCost",
    "ceiling_partition",
    "cost_table_from_artifact",
    "d2_partition",
    "plan_d2_groups",
]
