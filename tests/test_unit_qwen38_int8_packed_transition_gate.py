"""Deterministic invariants for the IKV-C2 packed-transition gate schedule.

The GPU gate replays ``_TRANSITION_SCHEDULE`` against independent c1
references.  These tests pin the schedule's shape so a later edit cannot
silently turn the transition gate into another steady-state gate (for example
by dropping the retirement step or by reusing a lane twice in one step).
"""

from __future__ import annotations

import pytest

from scripts.qwen38_int8_packed_transition_gate import (
    _RETIRED_ROLES,
    _RETIRE_AFTER_STEP,
    _TRANSITION_SCHEDULE,
)

_PHYSICAL_ROWS = 4


def test_transition_schedule_lanes_are_unique_and_in_range() -> None:
    for step, lane_map in enumerate(_TRANSITION_SCHEDULE):
        lanes = [lane for lane, _role in lane_map]
        assert lanes, f"step {step} has no active lane"
        assert len(set(lanes)) == len(lanes), f"step {step} reuses a lane: {lanes}"
        assert all(0 <= lane < _PHYSICAL_ROWS for lane in lanes), f"step {step} lane out of range"


def test_transition_schedule_retires_and_admits_within_the_fixed_width() -> None:
    roles_by_step = [[role for _lane, role in lane_map] for lane_map in _TRANSITION_SCHEDULE]

    # The schedule must actually narrow the group before it widens it again.
    narrowed = [index for index, roles in enumerate(roles_by_step) if len(roles) < _PHYSICAL_ROWS]
    assert narrowed, "schedule never retires a lane, so it is not a transition gate"
    assert narrowed == [_RETIRE_AFTER_STEP + 1], (
        "the retired lanes must be absent on exactly the step after the retirement step"
    )

    for role in _RETIRED_ROLES:
        active = [index for index, roles in enumerate(roles_by_step) if role in roles]
        assert active, f"retired role {role} is never active"
        assert max(active) == _RETIRE_AFTER_STEP, f"retired role {role} outlives its retirement"

    newcomers = [role for role in roles_by_step[-1] if role.startswith("new")]
    assert newcomers, "schedule never admits a newcomer"
    for role in newcomers:
        first = next(index for index, roles in enumerate(roles_by_step) if role in roles)
        assert first > _RETIRE_AFTER_STEP, f"newcomer {role} joins before a lane is freed"


def test_transition_schedule_keeps_a_steady_lane_across_the_transition() -> None:
    roles_by_step = [[role for _lane, role in lane_map] for lane_map in _TRANSITION_SCHEDULE]
    steady = set(roles_by_step[0]) - set(_RETIRED_ROLES)
    assert steady, "schedule has no lane that survives the transition"
    for index, roles in enumerate(roles_by_step):
        missing = steady - set(roles)
        assert not missing, f"steady lane(s) {sorted(missing)} dropped at step {index}"


def test_transition_schedule_reuses_the_freed_lanes_for_newcomers() -> None:
    retired_lanes = {
        lane
        for lane, role in _TRANSITION_SCHEDULE[_RETIRE_AFTER_STEP]
        if role in _RETIRED_ROLES
    }
    admitted = {lane for lane, role in _TRANSITION_SCHEDULE[-1] if role.startswith("new")}
    assert admitted == retired_lanes, (
        "newcomers must occupy exactly the lanes the retired roles released"
    )


@pytest.mark.parametrize("step", range(len(_TRANSITION_SCHEDULE)))
def test_transition_schedule_step_width_never_exceeds_physical_rows(step: int) -> None:
    assert len(_TRANSITION_SCHEDULE[step]) <= _PHYSICAL_ROWS
