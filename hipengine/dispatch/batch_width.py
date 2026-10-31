"""Exact row-width partition planning for resident batch decode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _normalize_widths(values: Any, *, label: str) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 1:
            raise ValueError(f"{label} must contain integers greater than one")
        if value in seen:
            raise ValueError(f"{label} contains duplicate width {value}")
        seen.add(value)
        normalized.append(int(value))
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class NativeBatchWidthProfile:
    """Executable native widths plus an optional measured cost hint.

    ``native_widths`` is the executable domain supplied by the caller from the
    backend/implementation contract. ``native_step_ms`` annotates only a subset
    with measured costs. ``blockers`` refuses native execution; identity,
    measured-position, and performance-claim mismatches belong in
    ``cost_notes`` and must not disable an implemented native path. Recorded
    negative findings apply only to matching capability and geometry.
    """

    source_artifact: str
    native_step_ms: tuple[tuple[int, float], ...]
    serial_row_step_ms: float
    native_widths: tuple[int, ...] = ()
    exact_widths: tuple[int, ...] = ()
    min_position: int | None = None
    max_position: int | None = None
    blockers: tuple[str, ...] = ()
    cost_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_artifact, str) or not self.source_artifact.strip():
            raise ValueError("source_artifact must be a non-empty string")
        if not _positive_finite(self.serial_row_step_ms):
            raise ValueError("serial_row_step_ms must be positive and finite")
        normalized: list[tuple[int, float]] = []
        seen: set[int] = set()
        for width, cost_ms in self.native_step_ms:
            if isinstance(width, bool) or not isinstance(width, int) or width <= 1:
                raise ValueError("native widths must be integers greater than one")
            if width in seen:
                raise ValueError(f"native width {width} is duplicated")
            if not _positive_finite(cost_ms):
                raise ValueError("native step costs must be positive and finite")
            seen.add(width)
            normalized.append((int(width), float(cost_ms)))
        native_widths = _normalize_widths(self.native_widths, label="native_widths")
        exact_widths = _normalize_widths(self.exact_widths, label="exact_widths")
        if native_widths:
            if exact_widths and not set(exact_widths).issubset(native_widths):
                raise ValueError("exact_widths must be a subset of native_widths")
            for width, _cost_ms in normalized:
                if width not in native_widths:
                    raise ValueError("native_step_ms widths must be a subset of native_widths")
        if not normalized and not native_widths and not self.blockers:
            raise ValueError("an unblocked profile must contain at least one native width")
        for label, value in (("min_position", self.min_position), ("max_position", self.max_position)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{label} must be a non-negative integer or None")
        if (
            self.min_position is not None
            and self.max_position is not None
            and self.min_position > self.max_position
        ):
            raise ValueError("min_position must be <= max_position")
        if any(not isinstance(blocker, str) or not blocker.strip() for blocker in self.blockers):
            raise ValueError("profile blockers must be non-empty strings")
        if any(not isinstance(note, str) or not note.strip() for note in self.cost_notes):
            raise ValueError("profile cost_notes must be non-empty strings")
        object.__setattr__(self, "source_artifact", self.source_artifact.strip())
        object.__setattr__(self, "native_step_ms", tuple(sorted(normalized)))
        object.__setattr__(self, "serial_row_step_ms", float(self.serial_row_step_ms))
        object.__setattr__(self, "native_widths", native_widths)
        object.__setattr__(self, "exact_widths", exact_widths)
        object.__setattr__(self, "blockers", tuple(self.blockers))
        object.__setattr__(self, "cost_notes", tuple(self.cost_notes))

    def native_costs(self) -> dict[int, float]:
        return dict(self.native_step_ms)

    def measured_cost_applicable(self, positions: tuple[int, ...] | None) -> bool:
        if not self.native_step_ms:
            return False
        if positions is None:
            return True
        if self.min_position is None or self.max_position is None:
            return True
        return all(self.min_position <= int(position) <= self.max_position for position in positions)

    def position_cost_note(self) -> str:
        lower = "-inf" if self.min_position is None else str(self.min_position)
        upper = "+inf" if self.max_position is None else str(self.max_position)
        return (
            f"decode positions are outside the measured range {lower}..{upper}; "
            "using nominal native costs"
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_artifact": self.source_artifact,
            "native_step_ms": {str(width): cost_ms for width, cost_ms in self.native_step_ms},
            "serial_row_step_ms": self.serial_row_step_ms,
            "native_widths": list(self.native_widths),
            "exact_widths": list(self.exact_widths),
            "min_position": self.min_position,
            "max_position": self.max_position,
            "blockers": list(self.blockers),
            "cost_notes": list(self.cost_notes),
        }


@dataclass(frozen=True, slots=True)
class BatchWidthGroup:
    """One native or serial call in an exact row-width cover."""

    mode: str
    width: int
    expected_step_ms: float

    def __post_init__(self) -> None:
        if self.mode not in {"native", "serial"}:
            raise ValueError("batch width group mode must be native or serial")
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("batch width group width must be a positive integer")
        if not _positive_finite(self.expected_step_ms):
            raise ValueError("batch width group expected_step_ms must be positive and finite")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "width": self.width,
            "expected_step_ms": self.expected_step_ms,
        }


@dataclass(frozen=True, slots=True)
class BatchWidthPartitionPlan:
    """Exact native/serial execution plan for one live row set."""

    requested_rows: int
    groups: tuple[BatchWidthGroup, ...]
    path: str
    expected_step_ms: float
    profile_source: str | None
    blockers: tuple[str, ...] = ()
    cost_source: str = "nominal"
    cost_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cost_source not in {"measured", "nominal"}:
            raise ValueError("cost_source must be measured or nominal")
        if any(not isinstance(note, str) or not note.strip() for note in self.cost_notes):
            raise ValueError("plan cost_notes must be non-empty strings")

    @property
    def group_widths(self) -> tuple[int, ...]:
        return tuple(group.width for group in self.groups)

    @property
    def native_rows(self) -> int:
        return sum(group.width for group in self.groups if group.mode == "native")

    @property
    def serial_rows(self) -> int:
        return sum(group.width for group in self.groups if group.mode == "serial")

    def signature(self) -> str:
        return "+".join(f"{group.mode}:{group.width}" for group in self.groups)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "requested_rows": self.requested_rows,
            "groups": [group.to_json_dict() for group in self.groups],
            "group_widths": list(self.group_widths),
            "path": self.path,
            "expected_step_ms": self.expected_step_ms,
            "profile_source": self.profile_source,
            "blockers": list(self.blockers),
            "cost_source": self.cost_source,
            "cost_notes": list(self.cost_notes),
        }


def _serial_plan(
    rows: int,
    *,
    serial_row_step_ms: float,
    profile_source: str | None,
    blockers: tuple[str, ...],
    cost_notes: tuple[str, ...] = (),
) -> BatchWidthPartitionPlan:
    expected = float(rows) * float(serial_row_step_ms)
    return BatchWidthPartitionPlan(
        requested_rows=rows,
        groups=(BatchWidthGroup("serial", rows, expected),),
        path="serial_exact",
        expected_step_ms=expected,
        profile_source=profile_source,
        blockers=blockers,
        cost_source="nominal",
        cost_notes=cost_notes,
    )


def plan_batch_width_partition(
    rows: int,
    *,
    profile: NativeBatchWidthProfile | None,
    positions: tuple[int, ...] | None = None,
) -> BatchWidthPartitionPlan:
    """Choose the minimum-cost exact cover from executable native widths.

    ``profile.blockers`` is the only native refusal. Measured costs are used
    when they apply to this capability/geometry and position envelope;
    otherwise nominal per-row weights choose among executable widths and the
    plan reports ``cost_source="nominal"``. Positions never disable native.
    """

    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("rows must be a positive integer")
    pos = tuple(positions or ())
    if positions is not None:
        if len(pos) != rows:
            raise ValueError("positions must match rows")
        if any(isinstance(position, bool) or not isinstance(position, int) or position < 0 for position in pos):
            raise ValueError("positions must contain non-negative integers")
    if profile is None:
        return _serial_plan(
            rows,
            serial_row_step_ms=1.0,
            profile_source=None,
            blockers=("no native batch width profile",),
            cost_notes=("no native batch width profile",),
        )
    if profile.blockers:
        return _serial_plan(
            rows,
            serial_row_step_ms=profile.serial_row_step_ms,
            profile_source=profile.source_artifact,
            blockers=profile.blockers,
            cost_notes=profile.cost_notes,
        )

    domain = profile.native_widths or tuple(width for width, _cost_ms in profile.native_step_ms)
    if not domain:
        return _serial_plan(
            rows,
            serial_row_step_ms=profile.serial_row_step_ms,
            profile_source=profile.source_artifact,
            blockers=("native batch width profile has no executable widths",),
            cost_notes=profile.cost_notes,
        )

    measured = profile.measured_cost_applicable(pos if positions is not None else None)
    measured_costs = profile.native_costs()
    costs: dict[int, float] = {}
    nominal_widths: set[int] = set()
    for width in domain:
        measured_cost = measured_costs.get(width) if measured else None
        if measured_cost is None:
            costs[width] = profile.serial_row_step_ms * width
            nominal_widths.add(width)
        else:
            costs[width] = measured_cost
    candidates = [(1, profile.serial_row_step_ms), *costs.items()]
    exact_set = set(profile.exact_widths) if profile.exact_widths else set(domain)
    best: list[tuple[float, int, tuple[int, ...]] | None] = [None] * (rows + 1)
    best[0] = (0.0, 0, ())
    for covered in range(1, rows + 1):
        options: list[tuple[float, int, tuple[int, ...]]] = []
        for width, cost_ms in candidates:
            if width > covered or best[covered - width] is None:
                continue
            previous_cost, previous_row_chunks, previous_widths = best[covered - width]
            widths = tuple(sorted((*previous_widths, width), reverse=True))
            row_chunks = previous_row_chunks + (0 if width == 1 or width in exact_set else 1)
            options.append((previous_cost + cost_ms, row_chunks, widths))
        if not options:
            raise RuntimeError(f"failed to construct an exact width cover for rows={rows}")
        lowest = min(option[0] for option in options)
        tied = [
            option
            for option in options
            if math.isclose(option[0], lowest, rel_tol=1e-12, abs_tol=1e-12)
        ]
        best[covered] = min(
            tied,
            key=lambda item: (
                item[1],
                len(item[2]),
                tuple(-width for width in item[2]),
            ),
        )

    resolved = best[rows]
    assert resolved is not None
    expected_step_ms, _row_chunks, widths = resolved
    serial_rows = widths.count(1)
    native_widths = tuple(width for width in widths if width > 1)
    groups = tuple(BatchWidthGroup("native", width, costs[width]) for width in native_widths)
    if serial_rows:
        groups += (
            BatchWidthGroup(
                "serial",
                serial_rows,
                serial_rows * profile.serial_row_step_ms,
            ),
        )
    if len(groups) == 1 and groups[0].mode == "native":
        path = "direct_native"
    elif native_widths:
        path = "partitioned_native"
    else:
        path = "serial_exact"
    selected_nominal = any(width in nominal_widths for width in native_widths)
    cost_source = "measured" if measured and not selected_nominal else "nominal"
    cost_notes = list(profile.cost_notes)
    if positions is not None and not measured and profile.native_step_ms:
        cost_notes.append(profile.position_cost_note())
    if selected_nominal and measured:
        cost_notes.append("selected widths use nominal native costs")
    return BatchWidthPartitionPlan(
        requested_rows=rows,
        groups=groups,
        path=path,
        expected_step_ms=float(expected_step_ms),
        profile_source=profile.source_artifact,
        cost_source=cost_source,
        cost_notes=tuple(dict.fromkeys(cost_notes)),
    )


__all__ = [
    "BatchWidthGroup",
    "BatchWidthPartitionPlan",
    "NativeBatchWidthProfile",
    "plan_batch_width_partition",
]
