"""MMQ plane gate: worktree-provenance classification and blocker naming.

The gate writes its own schema-3 provenance, so the classified flag is always
present and the coarse ``dirty`` fallback was removed. These tests pin the
fail-closed contract without a GPU or a model.
"""

from __future__ import annotations

import pytest

from scripts.execution_profile_q8_mmq_plane_gate import (
    _classified_worktree_clean,
    gate_claims,
    qualification_blockers,
)

_PROVENANCE_BLOCKER = "execution-affecting worktree provenance"


def test_prose_only_dirty_worktree_still_qualifies() -> None:
    """Docs and worklog edits cannot change what executed."""

    assert _classified_worktree_clean(
        {
            "dirty": True,
            "staged_dirty": False,
            "unstaged_dirty": True,
            "dirty_paths": ["docs/REFACTOR.md"],
            "untracked_paths": ["worklog/entries/20261027T000000.000000Z-note.md"],
            "execution_affecting_dirty": False,
            "execution_affecting_paths": [],
        }
    )


def test_clean_collector_record_qualifies() -> None:
    assert _classified_worktree_clean(
        {
            "dirty": False,
            "staged_dirty": False,
            "unstaged_dirty": False,
            "untracked_dirty": False,
            "execution_affecting_dirty": False,
            "execution_affecting_paths": [],
        }
    )


@pytest.mark.parametrize(
    "provenance",
    [
        {"execution_affecting_dirty": True, "dirty": True},
        {"execution_affecting_dirty": True, "dirty": False},
        {"dirty": False},
        {},
        {"execution_affecting_dirty": None, "dirty": False},
        {"execution_affecting_dirty": "false", "dirty": False},
        {"execution_affecting_dirty": 0, "dirty": False},
    ],
)
def test_unclassified_or_dirty_provenance_refuses(provenance: dict) -> None:
    """No coarse fallback: only the collector's exact False clears a capture."""

    assert not _classified_worktree_clean(provenance)


def test_blockers_name_the_refused_provenance() -> None:
    assert qualification_blockers(
        quality_passed=True, deterministic=True, clean=False
    ) == [_PROVENANCE_BLOCKER]


def test_all_passing_conditions_leave_no_blocker() -> None:
    assert qualification_blockers(
        quality_passed=True, deterministic=True, clean=True
    ) == []


def test_every_failed_condition_is_named() -> None:
    assert qualification_blockers(
        quality_passed=False, deterministic=False, clean=False
    ) == [
        "calibrated quality thresholds",
        "same-schedule repeat determinism",
        _PROVENANCE_BLOCKER,
    ]


def test_clean_capture_claims_qualification() -> None:
    claims = gate_claims(quality_passed=True, deterministic=True, clean=True)

    assert claims == {
        "status": "complete",
        "measurement_valid": True,
        "performance_claim": False,
        "profile_qualification_claim": True,
        "qualification_blockers": [],
    }


def test_dirty_capture_suppresses_the_qualification_claim() -> None:
    """A passing quality gate cannot carry a claim on an invalid measurement."""

    claims = gate_claims(quality_passed=True, deterministic=True, clean=False)

    assert claims["status"] == "invalid_or_screen_only"
    assert claims["measurement_valid"] is False
    assert claims["performance_claim"] is False
    assert claims["profile_qualification_claim"] is False
    assert claims["qualification_blockers"] == [_PROVENANCE_BLOCKER]


@pytest.mark.parametrize(
    "provenance",
    [
        {"execution_affecting_dirty": True},
        {"dirty": False},
        {},
        {"execution_affecting_dirty": None},
        {"execution_affecting_dirty": "false"},
    ],
)
def test_unclassified_capture_artifact_claims_nothing(provenance: dict) -> None:
    """The artifact-level regression: helper, claim, and blocker agree."""

    clean = _classified_worktree_clean(provenance)
    claims = gate_claims(quality_passed=True, deterministic=True, clean=clean)

    assert claims["profile_qualification_claim"] is False
    assert claims["measurement_valid"] is False
    assert claims["qualification_blockers"] == [_PROVENANCE_BLOCKER]
