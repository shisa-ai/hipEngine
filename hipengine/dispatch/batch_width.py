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


@dataclass(frozen=True, slots=True)
class NativeBatchWidthProfile:
    """Evidence-backed native widths and their measured decode-step costs."""

    source_artifact: str
    native_step_ms: tuple[tuple[int, float], ...]
    serial_row_step_ms: float
    min_position: int | None = None
    max_position: int | None = None
    blockers: tuple[str, ...] = ()

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
        if not normalized and not self.blockers:
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
        object.__setattr__(self, "source_artifact", self.source_artifact.strip())
        object.__setattr__(self, "native_step_ms", tuple(sorted(normalized)))
        object.__setattr__(self, "serial_row_step_ms", float(self.serial_row_step_ms))
        object.__setattr__(self, "blockers", tuple(self.blockers))

    def native_costs(self) -> dict[int, float]:
        return dict(self.native_step_ms)

    def position_blockers(self, positions: tuple[int, ...]) -> tuple[str, ...]:
        if self.min_position is not None and any(position < self.min_position for position in positions):
            return (self._position_blocker(),)
        if self.max_position is not None and any(position > self.max_position for position in positions):
            return (self._position_blocker(),)
        return ()

    def _position_blocker(self) -> str:
        lower = "-inf" if self.min_position is None else str(self.min_position)
        upper = "+inf" if self.max_position is None else str(self.max_position)
        return f"decode positions are outside the evidenced range {lower}..{upper}"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_artifact": self.source_artifact,
            "native_step_ms": {str(width): cost_ms for width, cost_ms in self.native_step_ms},
            "serial_row_step_ms": self.serial_row_step_ms,
            "min_position": self.min_position,
            "max_position": self.max_position,
            "blockers": list(self.blockers),
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
        }


def _serial_plan(
    rows: int,
    *,
    serial_row_step_ms: float,
    profile_source: str | None,
    blockers: tuple[str, ...],
) -> BatchWidthPartitionPlan:
    expected = float(rows) * float(serial_row_step_ms)
    return BatchWidthPartitionPlan(
        requested_rows=rows,
        groups=(BatchWidthGroup("serial", rows, expected),),
        path="serial_exact",
        expected_step_ms=expected,
        profile_source=profile_source,
        blockers=blockers,
    )


def plan_batch_width_partition(
    rows: int,
    *,
    profile: NativeBatchWidthProfile | None,
    positions: tuple[int, ...] | None = None,
) -> BatchWidthPartitionPlan:
    """Choose the minimum-cost exact cover from evidenced native widths and serial rows."""

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
        )
    blockers = profile.blockers + (profile.position_blockers(pos) if positions is not None else ())
    if blockers:
        return _serial_plan(
            rows,
            serial_row_step_ms=profile.serial_row_step_ms,
            profile_source=profile.source_artifact,
            blockers=blockers,
        )

    costs = profile.native_costs()
    candidates = [(1, profile.serial_row_step_ms), *costs.items()]
    best: list[tuple[float, tuple[int, ...]] | None] = [None] * (rows + 1)
    best[0] = (0.0, ())
    for covered in range(1, rows + 1):
        options: list[tuple[float, tuple[int, ...]]] = []
        for width, cost_ms in candidates:
            if width > covered or best[covered - width] is None:
                continue
            previous_cost, previous_widths = best[covered - width]
            widths = tuple(sorted((*previous_widths, width), reverse=True))
            options.append((previous_cost + cost_ms, widths))
        if not options:
            raise RuntimeError(f"failed to construct an exact width cover for rows={rows}")
        best[covered] = min(options, key=lambda item: (item[0], len(item[1]), tuple(-width for width in item[1])))

    resolved = best[rows]
    assert resolved is not None
    expected_step_ms, widths = resolved
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
    return BatchWidthPartitionPlan(
        requested_rows=rows,
        groups=groups,
        path=path,
        expected_step_ms=float(expected_step_ms),
        profile_source=profile.source_artifact,
    )


__all__ = [
    "BatchWidthGroup",
    "BatchWidthPartitionPlan",
    "NativeBatchWidthProfile",
    "plan_batch_width_partition",
]
