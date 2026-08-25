from __future__ import annotations

import pytest

from scripts.specdec2_s4_prefix_gate import _completed_row


def test_prefix_gate_selects_one_completed_request_row() -> None:
    snapshot = {
        "runner": {
            "routes": {
                "recent_completed": [
                    {"request_id": 1, "value": "old"},
                    {"request_id": 2, "value": "peer"},
                    {"request_id": 1, "value": "new"},
                ]
            }
        }
    }

    assert _completed_row(snapshot, 1)["value"] == "new"
    with pytest.raises(RuntimeError, match="completed request 3"):
        _completed_row(snapshot, 3)
