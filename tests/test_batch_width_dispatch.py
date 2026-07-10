from __future__ import annotations

import pytest

from hipengine.dispatch import NativeBatchWidthProfile, plan_batch_width_partition


@pytest.fixture
def gfx1151_profile() -> NativeBatchWidthProfile:
    return NativeBatchWidthProfile(
        source_artifact="benchmarks/results/gfx1151-paro-shapes.json",
        native_step_ms=((2, 25.465), (4, 40.158), (6, 54.568), (8, 69.254)),
        serial_row_step_ms=14.969,
        min_position=512,
        max_position=647,
    )


@pytest.mark.parametrize(
    ("rows", "expected_widths", "expected_path"),
    [
        (1, (1,), "serial_exact"),
        (2, (2,), "direct_native"),
        (3, (2, 1), "partitioned_native"),
        (5, (4, 1), "partitioned_native"),
        (7, (6, 1), "partitioned_native"),
        (9, (8, 1), "partitioned_native"),
        (12, (6, 6), "partitioned_native"),
        (16, (8, 8), "partitioned_native"),
    ],
)
def test_batch_width_partition_uses_lowest_cost_exact_cover(
    gfx1151_profile: NativeBatchWidthProfile,
    rows: int,
    expected_widths: tuple[int, ...],
    expected_path: str,
) -> None:
    plan = plan_batch_width_partition(
        rows,
        profile=gfx1151_profile,
        positions=(512,) * rows,
    )

    assert plan.group_widths == expected_widths
    assert plan.path == expected_path
    assert sum(group.width for group in plan.groups) == rows
    assert plan.serial_rows == expected_widths.count(1)


def test_batch_width_partition_falls_back_to_serial_outside_profile_context(
    gfx1151_profile: NativeBatchWidthProfile,
) -> None:
    plan = plan_batch_width_partition(
        4,
        profile=gfx1151_profile,
        positions=(648, 648, 648, 648),
    )

    assert plan.path == "serial_exact"
    assert plan.group_widths == (4,)
    assert plan.groups[0].mode == "serial"
    assert plan.blockers == ("decode positions are outside the evidenced range 512..647",)


def test_batch_width_partition_falls_back_to_serial_for_blocked_profile() -> None:
    profile = NativeBatchWidthProfile(
        source_artifact="benchmarks/results/blocked.json",
        native_step_ms=((2, 20.0),),
        serial_row_step_ms=15.0,
        blockers=("model fingerprint mismatch",),
    )

    plan = plan_batch_width_partition(3, profile=profile, positions=(10, 10, 10))

    assert plan.path == "serial_exact"
    assert plan.group_widths == (3,)
    assert plan.blockers == ("model fingerprint mismatch",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"native_step_ms": ((1, 10.0),)},
        {"native_step_ms": ((2, 0.0),)},
        {"native_step_ms": ((2, 10.0), (2, 11.0))},
        {"serial_row_step_ms": 0.0},
        {"min_position": 20, "max_position": 10},
    ],
)
def test_native_batch_width_profile_rejects_invalid_costs(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "source_artifact": "benchmarks/results/profile.json",
        "native_step_ms": ((2, 10.0),),
        "serial_row_step_ms": 15.0,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        NativeBatchWidthProfile(**values)
