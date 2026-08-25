from __future__ import annotations

from scripts.specdec2_s6_failure_gate import _recent_rows


def test_failure_gate_selects_only_requested_completed_rows() -> None:
    snapshot = {
        "runner": {
            "routes": {
                "recent_completed": [
                    {"request_id": 1},
                    {"request_id": 2},
                    {"request_id": 3},
                ]
            }
        }
    }

    assert _recent_rows(snapshot, {1, 3}) == [
        {"request_id": 1},
        {"request_id": 3},
    ]
