"""Plan-time and physical-range residency audits for Qwen3.5/3.6 GGUF weights."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q4_K_X8,
    LAYOUT_GGUF_Q5_K_X8,
    LAYOUT_GGUF_Q6_K_X8,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    Q4_T16_DECODE_TILES,
    Q4_T16_DECODE_TILES_R3PLUS,
    Qwen35GGUFDeviceWeight,
    Qwen35GGUFWeightSpec,
    planned_qwen35_gguf_weight_allocation_nbytes,
)


@dataclass(frozen=True)
class Qwen35GGUFPlannedAllocationCensus:
    name: str
    layout: str
    nbytes: int
    is_alternate: bool


@dataclass(frozen=True)
class Qwen35GGUFPlannedWeightCensus:
    source_name: str
    source_nbytes: int
    slot_paths: tuple[str, ...]
    canonical_layout: str
    quant_key: str
    allocations: tuple[Qwen35GGUFPlannedAllocationCensus, ...]

    @property
    def resident_nbytes(self) -> int:
        return sum(allocation.nbytes for allocation in self.allocations)

    @property
    def alternate_layout_nbytes(self) -> int:
        return sum(
            allocation.nbytes
            for allocation in self.allocations
            if allocation.is_alternate
        )


@dataclass(frozen=True)
class Qwen35GGUFPlanResidencyCensus:
    logical_weights: tuple[Qwen35GGUFPlannedWeightCensus, ...]
    issues: tuple[str, ...] = ()

    @property
    def logical_tensor_count(self) -> int:
        return len(self.logical_weights)

    @property
    def alias_count(self) -> int:
        return sum(max(0, len(weight.slot_paths) - 1) for weight in self.logical_weights)

    @property
    def source_nbytes(self) -> int:
        return sum(weight.source_nbytes for weight in self.logical_weights)

    @property
    def resident_nbytes(self) -> int:
        return sum(weight.resident_nbytes for weight in self.logical_weights)

    @property
    def alternate_layout_nbytes(self) -> int:
        return sum(weight.alternate_layout_nbytes for weight in self.logical_weights)

    def assert_single_layout(self) -> None:
        failures = list(self.issues)
        if self.alternate_layout_nbytes:
            failures.append(
                f"alternate-layout planned bytes={self.alternate_layout_nbytes}"
            )
        if failures:
            raise ValueError(
                "single-layout GGUF residency invariant failed: "
                + "; ".join(failures)
            )


@dataclass(frozen=True)
class Qwen35GGUFResidentWeightRef:
    """One named use of a resident weight, including its memory class."""

    memory_class: str
    owner: str
    weight: Qwen35GGUFDeviceWeight

    def __post_init__(self) -> None:
        if not self.memory_class:
            raise ValueError("resident weight memory_class must be non-empty")
        if not self.owner:
            raise ValueError("resident weight owner must be non-empty")


@dataclass(frozen=True)
class Qwen35GGUFPhysicalRangeCensus:
    device: str
    ptr: int
    nbytes: int
    source_names: tuple[str, ...]
    owners: tuple[str, ...]
    memory_classes: tuple[str, ...]
    allocation_names: tuple[str, ...]
    layouts: tuple[str, ...]
    is_alternate: bool


@dataclass(frozen=True)
class Qwen35GGUFRuntimeResidencyCensus:
    physical_ranges: tuple[Qwen35GGUFPhysicalRangeCensus, ...]
    duplicate_allocation_roles: tuple[tuple[str, str, str], ...]
    duplicate_payload_nbytes: int
    alternate_layout_nbytes: int
    issues: tuple[str, ...] = ()

    @property
    def physical_nbytes(self) -> int:
        return sum(record.nbytes for record in self.physical_ranges)

    @property
    def physical_allocation_count(self) -> int:
        return len(self.physical_ranges)

    @property
    def memory_class_nbytes(self) -> Mapping[str, int]:
        totals: dict[str, int] = {}
        for record in self.physical_ranges:
            # A borrowed exact range can appear as both target and root_shared.
            # Charge it to root_shared once instead of inflating both classes.
            memory_class = (
                "root_shared"
                if "root_shared" in record.memory_classes
                else record.memory_classes[0]
            )
            totals[memory_class] = totals.get(memory_class, 0) + record.nbytes
        return MappingProxyType(dict(sorted(totals.items())))

    def assert_single_layout(self) -> None:
        failures = list(self.issues)
        if self.alternate_layout_nbytes:
            failures.append(
                f"alternate-layout physical bytes={self.alternate_layout_nbytes}"
            )
        if self.duplicate_payload_nbytes:
            failures.append(
                f"duplicate physical payload bytes={self.duplicate_payload_nbytes}"
            )
        if failures:
            raise ValueError(
                "single-layout GGUF residency invariant failed: "
                + "; ".join(failures)
            )


def census_qwen35_gguf_weight_specs(
    specs: Iterable[Qwen35GGUFWeightSpec],
) -> Qwen35GGUFPlanResidencyCensus:
    """Build a device-free byte census, preserving aliases by source identity."""

    grouped: dict[str, list[Qwen35GGUFWeightSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.source.name, []).append(spec)

    logical_weights: list[Qwen35GGUFPlannedWeightCensus] = []
    issues: list[str] = []
    for source_name, aliases in grouped.items():
        owner = aliases[0]
        signatures = {
            (spec.layout, spec.quant_key, spec.allocation_names)
            for spec in aliases
        }
        if len(signatures) != 1:
            issues.append(
                f"source {source_name!r} has incompatible alias plans: "
                f"{sorted((layout, quant, names) for layout, quant, names in signatures)!r}"
            )
        allocation_records = tuple(
            Qwen35GGUFPlannedAllocationCensus(
                name=name,
                layout=_allocation_layout(owner, name),
                nbytes=nbytes,
                is_alternate=_allocation_layout(owner, name) != owner.layout,
            )
            for name, nbytes in planned_qwen35_gguf_weight_allocation_nbytes(owner)
        )
        logical_weights.append(
            Qwen35GGUFPlannedWeightCensus(
                source_name=source_name,
                source_nbytes=int(owner.source.nbytes),
                slot_paths=tuple(spec.slot_path for spec in aliases),
                canonical_layout=owner.layout,
                quant_key=owner.quant_key,
                allocations=allocation_records,
            )
        )
    return Qwen35GGUFPlanResidencyCensus(
        logical_weights=tuple(logical_weights),
        issues=tuple(issues),
    )


def qwen35_gguf_target_weight_refs(resident) -> tuple[Qwen35GGUFResidentWeightRef, ...]:
    """Enumerate target root/layer uses without hiding tied aliases."""

    refs = [
        Qwen35GGUFResidentWeightRef("target", f"root.{slot}", weight)
        for slot, weight in resident.root_weights.items()
    ]
    refs.extend(
        Qwen35GGUFResidentWeightRef(
            "target", f"layers.{layer.layer_id}.{slot}", weight
        )
        for layer in resident.layers
        for slot, weight in layer.weights.items()
    )
    return tuple(refs)


def qwen35_gguf_nextn_weight_refs(resident) -> tuple[Qwen35GGUFResidentWeightRef, ...]:
    """Enumerate NextN-owned weights and borrowed target-root aliases."""

    owned = {id(weight) for weight in resident.owned_weights}
    refs = [
        Qwen35GGUFResidentWeightRef("nextn", f"draft.layer.{slot}", weight)
        for slot, weight in resident.layer_weights.items()
    ]
    refs.extend(
        Qwen35GGUFResidentWeightRef("nextn", f"draft.nextn.{slot}", weight)
        for slot, weight in resident.nextn_weights.items()
    )
    refs.extend(
        Qwen35GGUFResidentWeightRef(
            "nextn" if id(weight) in owned else "root_shared",
            f"draft.fallback.{slot}",
            weight,
        )
        for slot, weight in resident.fallback_weights.items()
    )
    return tuple(refs)


def census_qwen35_gguf_resident_weight_refs(
    refs: Iterable[Qwen35GGUFResidentWeightRef],
) -> Qwen35GGUFRuntimeResidencyCensus:
    """Deduplicate resident allocations by exact ``(device, ptr, nbytes)``."""

    uses_by_range: dict[tuple[str, int, int], list[_PhysicalUse]] = {}
    ranges_by_role: dict[tuple[str, str, str], set[tuple[str, int, int]]] = {}
    for ref in refs:
        spec = ref.weight.spec
        for allocation_name, allocation in ref.weight.allocations.items():
            device = str(allocation.tensor.device)
            key = (device, int(allocation.buffer.ptr), int(allocation.buffer.nbytes))
            layout = _allocation_layout(spec, allocation_name)
            use = _PhysicalUse(
                source_name=spec.source.name,
                owner=ref.owner,
                memory_class=ref.memory_class,
                allocation_name=allocation_name,
                layout=layout,
                is_alternate=layout != spec.layout,
            )
            uses_by_range.setdefault(key, []).append(use)
            ranges_by_role.setdefault(
                (spec.source.name, allocation_name, layout), set()
            ).add(key)

    issues: list[str] = []
    physical_ranges: list[Qwen35GGUFPhysicalRangeCensus] = []
    alternate_layout_nbytes = 0
    for (device, ptr, nbytes), uses in sorted(
        uses_by_range.items(), key=lambda item: item[0]
    ):
        source_names = tuple(sorted({use.source_name for use in uses}))
        if len(source_names) > 1:
            issues.append(
                f"physical range {device}:{ptr:#x}+{nbytes} has undeclared logical owners "
                f"{source_names!r}"
            )
        is_alternate = any(use.is_alternate for use in uses)
        if is_alternate:
            alternate_layout_nbytes += nbytes
        physical_ranges.append(
            Qwen35GGUFPhysicalRangeCensus(
                device=device,
                ptr=ptr,
                nbytes=nbytes,
                source_names=source_names,
                owners=tuple(sorted({use.owner for use in uses})),
                memory_classes=tuple(sorted({use.memory_class for use in uses})),
                allocation_names=tuple(sorted({use.allocation_name for use in uses})),
                layouts=tuple(sorted({use.layout for use in uses})),
                is_alternate=is_alternate,
            )
        )

    duplicate_roles = tuple(
        sorted(role for role, ranges in ranges_by_role.items() if len(ranges) > 1)
    )
    duplicate_payload_nbytes = 0
    for role in duplicate_roles:
        ranges = sorted(ranges_by_role[role])
        duplicate_payload_nbytes += sum(nbytes for _device, _ptr, nbytes in ranges[1:])

    return Qwen35GGUFRuntimeResidencyCensus(
        physical_ranges=tuple(physical_ranges),
        duplicate_allocation_roles=duplicate_roles,
        duplicate_payload_nbytes=duplicate_payload_nbytes,
        alternate_layout_nbytes=alternate_layout_nbytes,
        issues=tuple(issues),
    )


@dataclass(frozen=True)
class _PhysicalUse:
    source_name: str
    owner: str
    memory_class: str
    allocation_name: str
    layout: str
    is_alternate: bool


def _allocation_layout(spec: Qwen35GGUFWeightSpec, allocation_name: str) -> str:
    if allocation_name in {Q4_T16_DECODE_TILES, Q4_T16_DECODE_TILES_R3PLUS}:
        return LAYOUT_GGUF_Q4_K_T16
    if allocation_name == "raw" and spec.allocation_names[0] != "raw":
        return LAYOUT_RAW_GGUF
    if allocation_name == "x8":
        return {
            "gguf_q4_k_t16_v1": LAYOUT_GGUF_Q4_K_X8,
            "gguf_q5_k_t16_v1": LAYOUT_GGUF_Q5_K_X8,
            "gguf_q6_k_t16_v1": LAYOUT_GGUF_Q6_K_X8,
        }.get(spec.quant_key, "gguf_x8_v1")
    return spec.layout


__all__ = [
    "Qwen35GGUFPhysicalRangeCensus",
    "Qwen35GGUFPlanResidencyCensus",
    "Qwen35GGUFPlannedAllocationCensus",
    "Qwen35GGUFPlannedWeightCensus",
    "Qwen35GGUFResidentWeightRef",
    "Qwen35GGUFRuntimeResidencyCensus",
    "census_qwen35_gguf_resident_weight_refs",
    "census_qwen35_gguf_weight_specs",
    "qwen35_gguf_nextn_weight_refs",
    "qwen35_gguf_target_weight_refs",
]
