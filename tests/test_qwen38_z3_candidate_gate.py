from __future__ import annotations

import pytest

from scripts.qwen38_z3_candidate_gate import _REQUIRED, evaluate_candidate_evidence


def _passing(candidate_id: str, declared_class: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "declared_class": declared_class,
        "checks": {name: True for name in _REQUIRED[candidate_id]},
    }


@pytest.mark.parametrize(
    ("candidate_id", "declared_class"),
    [
        ("P1_F16_ACTIVATION_B", "T1"),
        ("M1_C1_ACCEPT_ROUTE_PARITY", "T2"),
        ("M2_C2_DRAFT_DEPTH", "T3"),
        ("M3_ACCEPT_BOUNDARY_DATAFLOW", "T0"),
    ],
)
def test_candidate_gate_accepts_complete_class_packet(candidate_id: str, declared_class: str) -> None:
    assert evaluate_candidate_evidence(_passing(candidate_id, declared_class))["passed"] is True


def test_candidate_gate_fails_closed_on_missing_checks() -> None:
    result = evaluate_candidate_evidence(
        {"candidate_id": "M3_ACCEPT_BOUNDARY_DATAFLOW", "declared_class": "T0", "checks": {}}
    )
    assert result["passed"] is False
    assert "check_failed:cancellation" in result["failures"]
    assert "check_failed:complete_wall_improved" in result["failures"]


def test_candidate_gate_rejects_class_drift() -> None:
    payload = _passing("M1_C1_ACCEPT_ROUTE_PARITY", "T0")
    result = evaluate_candidate_evidence(payload)
    assert result["passed"] is False
    assert "declared_class_must_be_T2" in result["failures"]


def test_t3_gate_forbids_ordinary_production_promotion() -> None:
    payload = _passing("M2_C2_DRAFT_DEPTH", "T3")
    payload["checks"]["ordinary_production_default"] = True  # type: ignore[index]
    result = evaluate_candidate_evidence(payload)
    assert result["passed"] is False
    assert "t3_ordinary_production_default_forbidden" in result["failures"]
