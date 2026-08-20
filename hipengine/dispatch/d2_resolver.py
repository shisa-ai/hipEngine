"""Artifact-backed cost-aware D2 physical-group resolver.

D2 partitions ready decode rows into certified physical groups while minimizing
measured serial model-step wall.  Cost maps are immutable evidence records: the
loader validates clean measurement provenance and exact hardware/model/profile
identity before a table can reach the scheduler.  Missing or mismatched evidence
must fall back to :func:`hipengine.dispatch.batch.plan_physical_batch_groups`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from hipengine.dispatch.batch import PhysicalBatchGroup, WorkItem

D2_COST_ARTIFACT_KIND = "hipengine_d2_physical_group_cost_map"
D2_COST_ARTIFACT_SCHEMA = 1
DENSE_MASK_CLASS = "dense_all_active"
_SHA256_HEX_LENGTH = 64


def _required_text(value: object, name: str) -> str:
    text = str(value) if isinstance(value, str) else ""
    if not text or text != text.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return text


def _sha256(value: object, name: str) -> str:
    text = _required_text(value, name).lower()
    if len(text) != _SHA256_HEX_LENGTH or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return text


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or int(value) <= 0:
        raise ValueError(f"{name} must be a positive int")
    return int(value)


def _non_negative_int(value: object, name: str) -> int:
    if type(value) is not int or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return int(value)


@dataclass(frozen=True, slots=True)
class CostTableExpectation:
    """Exact runtime identity required to consume one D2 cost map."""

    backend: str
    target_arch: str
    host_name: str
    device_name: str
    model_fingerprint: str
    quant: str
    kv_dtype: str
    execution_profile: str
    graph_mode: str
    physical_widths: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "backend",
            "target_arch",
            "host_name",
            "device_name",
            "quant",
            "kv_dtype",
            "execution_profile",
            "graph_mode",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "model_fingerprint",
            _sha256(self.model_fingerprint, "model_fingerprint"),
        )
        widths = tuple(int(width) for width in self.physical_widths)
        if not widths or widths[0] != 1 or tuple(sorted(set(widths))) != widths:
            raise ValueError("physical_widths must be unique, increasing, and start at c1")
        object.__setattr__(self, "physical_widths", widths)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "target_arch": self.target_arch,
            "host_name": self.host_name,
            "device_name": self.device_name,
            "model_fingerprint": self.model_fingerprint,
            "quant": self.quant,
            "kv_dtype": self.kv_dtype,
            "execution_profile": self.execution_profile,
            "graph_mode": self.graph_mode,
            "physical_widths": list(self.physical_widths),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> "CostTableExpectation":
        return cls(
            backend=str(payload.get("backend", "")),
            target_arch=str(payload.get("target_arch", "")),
            host_name=str(payload.get("host_name", "")),
            device_name=str(payload.get("device_name", "")),
            model_fingerprint=str(payload.get("model_fingerprint", "")),
            quant=str(payload.get("quant", "")),
            kv_dtype=str(payload.get("kv_dtype", "")),
            execution_profile=str(payload.get("execution_profile", "")),
            graph_mode=str(payload.get("graph_mode", "")),
            physical_widths=tuple(int(width) for width in payload.get("physical_widths", ())),
        )


@dataclass(frozen=True, slots=True)
class PrimitiveCostRecord:
    """D1 primitive-cost schema; D2 consumes the complete-group records below."""

    backend: str
    layer: str
    quant: str
    variant: str
    operation: str
    role: str
    k: int
    n: int
    active_rows: int
    physical_width: int
    mask_class: str
    graph_mode: str
    latency_ms: float
    workspace_bytes: int
    strict_fallback: str
    correctness_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "backend",
            "layer",
            "quant",
            "variant",
            "operation",
            "role",
            "mask_class",
            "graph_mode",
            "strict_fallback",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        for name in ("k", "n", "active_rows", "physical_width"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.active_rows > self.physical_width:
            raise ValueError("primitive active_rows cannot exceed physical_width")
        if not math.isfinite(self.latency_ms) or self.latency_ms <= 0.0:
            raise ValueError("primitive latency_ms must be finite and positive")
        object.__setattr__(
            self,
            "workspace_bytes",
            _non_negative_int(self.workspace_bytes, "workspace_bytes"),
        )
        object.__setattr__(
            self,
            "correctness_sha256",
            _sha256(self.correctness_sha256, "correctness_sha256"),
        )


@dataclass(frozen=True, slots=True)
class PhysicalWidthCost:
    """Measured complete-model cost for one active/physical/mask route."""

    active_rows: int
    physical_width: int
    mask_class: str
    model_step_ms: float
    workspace_bytes: int
    route_manifest_sha256: str
    correctness_sha256: str
    source: str
    workspace_scope: str = "preallocated_shared_union"
    sample_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_rows", _positive_int(self.active_rows, "active_rows"))
        object.__setattr__(
            self,
            "physical_width",
            _positive_int(self.physical_width, "physical_width"),
        )
        if self.active_rows > self.physical_width:
            raise ValueError("active_rows cannot exceed physical_width")
        object.__setattr__(self, "mask_class", _required_text(self.mask_class, "mask_class"))
        if not math.isfinite(self.model_step_ms) or self.model_step_ms <= 0.0:
            raise ValueError("model_step_ms must be finite and positive")
        object.__setattr__(
            self,
            "workspace_bytes",
            _non_negative_int(self.workspace_bytes, "workspace_bytes"),
        )
        object.__setattr__(
            self,
            "route_manifest_sha256",
            _sha256(self.route_manifest_sha256, "route_manifest_sha256"),
        )
        object.__setattr__(
            self,
            "correctness_sha256",
            _sha256(self.correctness_sha256, "correctness_sha256"),
        )
        object.__setattr__(self, "source", _required_text(self.source, "source"))
        object.__setattr__(
            self,
            "workspace_scope",
            _required_text(self.workspace_scope, "workspace_scope"),
        )
        object.__setattr__(self, "sample_count", _positive_int(self.sample_count, "sample_count"))


@dataclass(frozen=True, slots=True)
class CostTable:
    """Validated complete-group map plus optional exact evidence identity."""

    records: tuple[PhysicalWidthCost, ...]
    default_max_width: int = 8
    identity: CostTableExpectation | None = None
    primitive_records: tuple[PrimitiveCostRecord, ...] = ()

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("cost table must be non-empty")
        max_width = _positive_int(self.default_max_width, "default_max_width")
        widths = tuple(record.physical_width for record in self.records)
        if widths[0] != 1:
            raise ValueError("cost table must include a c1 route")
        if tuple(sorted(set(widths))) != widths:
            raise ValueError("cost table widths must be unique and strictly increasing")
        if widths[-1] > max_width:
            raise ValueError("cost table width exceeds default_max_width")
        sources = {record.source for record in self.records}
        if len(sources) != 1:
            raise ValueError("cost table records must share the same source")
        for record in self.records:
            if record.active_rows != record.physical_width or record.mask_class != DENSE_MASK_CLASS:
                raise ValueError("current D2 table requires exact dense physical-width records")
        if self.identity is not None and self.identity.physical_widths != widths:
            raise ValueError("cost table identity physical widths do not match records")

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(record.physical_width for record in self.records)

    def record(self, physical_width: int) -> PhysicalWidthCost:
        for record in self.records:
            if record.physical_width == int(physical_width):
                return record
        raise KeyError(f"no measured cost for physical width {physical_width}")

    def cost_ms(self, physical_width: int) -> float:
        return self.record(physical_width).model_step_ms


def _validated_widths(widths: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(width) for width in widths)
    if not values or values[0] != 1 or tuple(sorted(set(values))) != values:
        raise ValueError("widths must be unique, strictly increasing, and start at c1")
    return values


def ceiling_partition(rows: int, widths: Sequence[int]) -> tuple[int, ...]:
    """Physical widths selected by the existing masked ceiling-bucket planner."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    certified = _validated_widths(widths)
    max_width = certified[-1]
    remaining = int(rows)
    groups: list[int] = []
    while remaining > 0:
        active_rows = min(max_width, remaining)
        physical_width = next(width for width in certified if width >= active_rows)
        groups.append(physical_width)
        remaining -= active_rows
    return tuple(groups)


def d2_partition(
    rows: int,
    cost_table: CostTable,
    *,
    max_workspace_bytes: int | None = None,
    max_group_model_step_ms: float | None = None,
) -> tuple[int, ...]:
    """Optimal exact dense serial composition under optional resource/SLO caps."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if max_workspace_bytes is not None and max_workspace_bytes < 0:
        raise ValueError("max_workspace_bytes must be non-negative")
    if max_group_model_step_ms is not None and (
        not math.isfinite(max_group_model_step_ms) or max_group_model_step_ms <= 0.0
    ):
        raise ValueError("max_group_model_step_ms must be finite and positive")
    records = tuple(
        record
        for record in cost_table.records
        if (max_workspace_bytes is None or record.workspace_bytes <= max_workspace_bytes)
        and (
            max_group_model_step_ms is None
            or record.model_step_ms <= max_group_model_step_ms
        )
    )
    if not records or records[0].physical_width != 1:
        raise ValueError("D2 constraints leave no complete c1 fallback")

    best_cost = [0.0] + [float("inf")] * rows
    best_count = [0] + [10**9] * rows
    best_groups: list[tuple[int, ...]] = [()] + [()] * rows
    for n in range(1, rows + 1):
        for record in records:
            width = record.physical_width
            if width > n:
                break
            prior = n - width
            candidate_cost = best_cost[prior] + record.model_step_ms
            candidate_count = best_count[prior] + 1
            candidate_groups = tuple(sorted((width, *best_groups[prior]), reverse=True))
            current_groups = best_groups[n]
            better = (
                candidate_cost < best_cost[n] - 1e-12
                or (
                    abs(candidate_cost - best_cost[n]) <= 1e-12
                    and (
                        candidate_count < best_count[n]
                        or (
                            candidate_count == best_count[n]
                            and candidate_groups > current_groups
                        )
                    )
                )
            )
            if better:
                best_cost[n] = candidate_cost
                best_count[n] = candidate_count
                best_groups[n] = candidate_groups
    groups = best_groups[rows]
    if not groups:
        raise ValueError("D2 constraints cannot cover the requested rows")
    return groups


def _dense_active_slots(work: WorkItem) -> tuple[int, ...]:
    if work.slot_ids:
        return tuple(sorted(int(slot) for slot in work.slot_ids))
    return tuple(range(len(work.request_ids)))


def plan_d2_groups(work: WorkItem, cost_table: CostTable) -> tuple[PhysicalBatchGroup, ...]:
    """Densify active execution rows while preserving stable scheduler slots."""

    active_slots = _dense_active_slots(work)
    active_rows = len(active_slots)
    if active_rows == 0:
        raise ValueError("work item has no active rows")
    widths = d2_partition(active_rows, cost_table)
    request_by_slot = dict(
        zip(
            work.slot_ids or tuple(range(len(work.request_ids))),
            work.request_ids,
            strict=True,
        )
    )
    groups: list[PhysicalBatchGroup] = []
    offset = 0
    for group_index, width in enumerate(widths):
        group_slots = active_slots[offset : offset + width]
        offset += width
        groups.append(
            PhysicalBatchGroup(
                logical_c=active_rows,
                group_index=group_index,
                group_count=len(widths),
                physical_slot_base=offset - width,
                physical_slot_extent=width,
                physical_rows=width,
                request_ids=tuple(request_by_slot[slot] for slot in group_slots),
                global_slot_indices=group_slots,
                active_slot_indices=tuple(range(width)),
                active_mask=(True,) * width,
                dense_execution_rows=True,
            )
        )
    return tuple(groups)


def _mapping(payload: object, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"D2 cost artifact {name} must be an object")
    return payload


def cost_table_from_artifact(
    path: str | Path,
    *,
    expected: CostTableExpectation | Mapping[str, object] | None = None,
) -> CostTable:
    """Load a retained clean D2 cost map and enforce exact runtime identity."""

    resolved = Path(path).expanduser().resolve()
    payload = _mapping(json.loads(resolved.read_text(encoding="utf-8")), "root")
    if payload.get("kind") != D2_COST_ARTIFACT_KIND or payload.get("schema") != D2_COST_ARTIFACT_SCHEMA:
        raise ValueError("D2 cost artifact kind/schema is unsupported")
    if not (
        payload.get("status") == "accepted"
        and payload.get("passed") is True
        and payload.get("measurement_valid") is True
    ):
        raise ValueError("D2 cost artifact must be an accepted passed measurement")

    identity = CostTableExpectation.from_json_dict(_mapping(payload.get("identity"), "identity"))
    if expected is not None:
        wanted = (
            expected.to_json_dict()
            if isinstance(expected, CostTableExpectation)
            else dict(expected)
        )
        observed = identity.to_json_dict()
        mismatches = {
            key: {"expected": value, "observed": observed.get(key)}
            for key, value in wanted.items()
            if observed.get(key) != value
        }
        if mismatches:
            raise ValueError(f"D2 cost artifact identity mismatch: {mismatches!r}")
    source_measurement = _mapping(payload.get("source_measurement"), "source_measurement")
    if not (
        source_measurement.get("status") == "measurement_complete"
        and source_measurement.get("passed") is True
        and source_measurement.get("complete_packet") is True
    ):
        raise ValueError("D2 cost artifact source measurement is incomplete or failed")
    source_correctness = _mapping(
        source_measurement.get("cross_configuration_correctness"),
        "source_measurement.cross_configuration_correctness",
    )
    if not (
        source_correctness.get("passed") is True
        and source_correctness.get("all_direct_c1_c8_exact") is True
        and source_correctness.get("all_measured_runs_repeatable") is True
    ):
        raise ValueError("D2 cost artifact source correctness is incomplete")
    measurement_sha = _sha256(source_measurement.get("sha256"), "source_measurement.sha256")
    _sha256(
        source_measurement.get("summary_artifact_sha256"),
        "source_measurement.summary_artifact_sha256",
    )
    provenance = _mapping(source_measurement.get("provenance"), "source_measurement.provenance")
    if provenance.get("dirty") is not False:
        raise ValueError("D2 cost artifact requires clean source measurement provenance")
    provenance_checks = {
        "resolved_backend": identity.backend,
        "target_arch": identity.target_arch,
        "host_name": identity.host_name,
        "device_name": identity.device_name,
        "quant": identity.quant,
        "kv_dtype": identity.kv_dtype,
    }
    for field, wanted in provenance_checks.items():
        if provenance.get(field) != wanted:
            raise ValueError(f"D2 source provenance {field} does not match identity")
    fingerprint = _mapping(provenance.get("model_fingerprint"), "model_fingerprint")
    if fingerprint.get("value") != identity.model_fingerprint:
        raise ValueError("D2 source provenance model fingerprint does not match identity")

    correctness = _mapping(payload.get("correctness"), "correctness")
    _sha256(correctness.get("quality_artifact_sha256"), "quality_artifact_sha256")
    _sha256(correctness.get("lifecycle_artifact_sha256"), "lifecycle_artifact_sha256")
    source = f"{resolved}#{measurement_sha}"
    records_payload = payload.get("physical_group_records")
    if not isinstance(records_payload, list) or not records_payload:
        raise ValueError("D2 cost artifact physical_group_records must be non-empty")
    records = tuple(
        PhysicalWidthCost(
            active_rows=_positive_int(row.get("active_rows"), "active_rows"),
            physical_width=_positive_int(row.get("physical_width"), "physical_width"),
            mask_class=str(row.get("mask_class", "")),
            model_step_ms=float(row.get("model_step_ms", float("nan"))),
            workspace_bytes=_non_negative_int(row.get("workspace_bytes"), "workspace_bytes"),
            workspace_scope=str(row.get("workspace_scope", "")),
            route_manifest_sha256=str(row.get("route_manifest_sha256", "")),
            correctness_sha256=str(row.get("correctness_sha256", "")),
            sample_count=_positive_int(row.get("sample_count"), "sample_count"),
            source=source,
        )
        for row in (_mapping(item, "physical_group_record") for item in records_payload)
    )
    primitive_payload = payload.get("primitive_records", [])
    if not isinstance(primitive_payload, list):
        raise ValueError("D2 cost artifact primitive_records must be an array")
    primitives = tuple(PrimitiveCostRecord(**dict(_mapping(item, "primitive_record"))) for item in primitive_payload)
    table = CostTable(
        records,
        default_max_width=identity.physical_widths[-1],
        identity=identity,
        primitive_records=primitives,
    )
    if table.widths != identity.physical_widths:
        raise ValueError("D2 cost artifact physical widths are incomplete")
    return table


__all__ = [
    "D2_COST_ARTIFACT_KIND",
    "CostTable",
    "CostTableExpectation",
    "PhysicalWidthCost",
    "PrimitiveCostRecord",
    "ceiling_partition",
    "cost_table_from_artifact",
    "d2_partition",
    "plan_d2_groups",
]
