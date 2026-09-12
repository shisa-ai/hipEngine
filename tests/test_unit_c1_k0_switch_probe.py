"""RED/GREEN tests for the K0<->MTP switch probe planning and validation."""
from __future__ import annotations

import pytest

from scripts.c1_k0_switch_probe import plan_legs, summarize_switches, validate_sequence


def test_plan_legs_alternates_and_counterbalances() -> None:
    assert plan_legs(0) == ("mtp", "k0", "mtp", "k0")
    assert plan_legs(1) == ("k0", "mtp", "k0", "mtp")
    assert plan_legs(2) == ("mtp", "k0", "mtp", "k0")
    assert plan_legs(3) == ("k0", "mtp", "k0", "mtp")


def test_plan_legs_rejects_odd_counts() -> None:
    with pytest.raises(ValueError):
        plan_legs(0, legs_per_prompt=3)


def test_validate_sequence_accepts_clean_switches() -> None:
    legs = plan_legs(0)
    ids = [10, 11, 12]
    rows = [
        {
            "generated_ids": ids,
            "mtp": (
                {"used": True, "draft_cycles": 9, "draft_tokens": 24}
                if leg == "mtp"
                else {"used": False, "draft_cycles": 0, "draft_tokens": 0}
            ),
        }
        for leg in legs
    ]
    assert validate_sequence(legs, rows, budget=3) == []


def test_validate_sequence_rejects_disengaged_mtp_leg() -> None:
    legs = plan_legs(0)
    rows = [
        {
            "generated_ids": [10, 11, 12],
            "mtp": {"used": False, "draft_cycles": 0, "draft_tokens": 0},
        },
        {"generated_ids": [10, 11, 12], "mtp": {"used": False}},
        {"generated_ids": [10, 11, 12], "mtp": {"used": True, "draft_cycles": 9, "draft_tokens": 24}},
        {"generated_ids": [10, 11, 12], "mtp": {"used": False}},
    ]
    reasons = validate_sequence(legs, rows, budget=3)
    assert "leg0_mtp_mtp_leg_not_engaged" in reasons


def test_validate_sequence_rejects_engaged_k0_leg() -> None:
    legs = plan_legs(1)  # starts with a K0 leg
    rows = [
        {
            "generated_ids": [10, 11, 12],
            "mtp": {"used": True, "draft_cycles": 9, "draft_tokens": 24},
        },
        {"generated_ids": [10, 11, 12], "mtp": {"used": True, "draft_cycles": 9, "draft_tokens": 24}},
        {"generated_ids": [10, 11, 12], "mtp": {"used": True, "draft_cycles": 9, "draft_tokens": 24}},
        {"generated_ids": [10, 11, 12], "mtp": {"used": False}},
    ]
    reasons = validate_sequence(legs, rows, budget=3)
    assert "leg0_k0_leg_engaged" in reasons


def test_validate_sequence_rejects_diverged_mtp_leg() -> None:
    legs = plan_legs(0)
    rows = [
        {"generated_ids": [10, 11, 12], "mtp": {"used": True, "draft_cycles": 9, "draft_tokens": 24}},
        {"generated_ids": [10, 11, 12], "mtp": {"used": False}},
        # The MTP leg after a K0 switch diverges: broken provider catch-up.
        {"generated_ids": [10, 11, 99], "mtp": {"used": True, "draft_cycles": 9, "draft_tokens": 24}},
        {"generated_ids": [10, 11, 12], "mtp": {"used": False}},
    ]
    reasons = validate_sequence(legs, rows, budget=3)
    assert "divergent_legs_2" in reasons


def test_validate_sequence_rejects_budget_overflow() -> None:
    legs = plan_legs(0)
    rows = [
        {
            "generated_ids": [10, 11, 12],
            "mtp": {"used": True, "draft_cycles": 9, "draft_tokens": 40},
        },
        {"generated_ids": [10, 11, 12], "mtp": {"used": False}},
        {"generated_ids": [10, 11, 12], "mtp": {"used": True, "draft_cycles": 9, "draft_tokens": 24}},
        {"generated_ids": [10, 11, 12], "mtp": {"used": False}},
    ]
    reasons = validate_sequence(legs, rows, budget=3)
    assert "leg0_mtp_mtp_leg_budget_exceeded" in reasons


def test_summarize_switches_counts_both_directions() -> None:
    assert summarize_switches(plan_legs(0)) == {"mtp_to_k0": 2, "k0_to_mtp": 1}
    assert summarize_switches(plan_legs(1)) == {"mtp_to_k0": 1, "k0_to_mtp": 2}
