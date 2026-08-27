from __future__ import annotations

from copy import deepcopy

from scripts.qwen36_dense_mtp2_c2_production_gate import _repeat_verdict


def _capture(*, repeat: int, digest: str = "same") -> dict[str, object]:
    return {
        "repeat": repeat,
        "prompt_id": "code_merge_intervals",
        "start_position": 40,
        "inputs": (264, 7047, 1817),
        "candidate_logits_sha256": digest,
        "candidate_top1": (7047, 1817, 25),
    }


def test_repeat_verdict_requires_all_three_identical_physical_schedules() -> None:
    result = _repeat_verdict(tuple(_capture(repeat=index) for index in range(3)), 3)

    assert result["passed"] is True
    assert result["rows"] == [
        {
            "prompt_id": "code_merge_intervals",
            "start_position": 40,
            "repeats": [0, 1, 2],
            "inputs_equal": True,
            "logits_equal": True,
            "top1_equal": True,
            "passed": True,
        }
    ]


def test_repeat_verdict_fails_on_missing_repeat_or_logit_drift() -> None:
    missing = _repeat_verdict((_capture(repeat=0), _capture(repeat=1)), 3)
    drift_rows = [_capture(repeat=index) for index in range(3)]
    drift_rows[-1] = deepcopy(drift_rows[-1])
    drift_rows[-1]["candidate_logits_sha256"] = "drift"
    drift = _repeat_verdict(tuple(drift_rows), 3)

    assert missing["passed"] is False
    assert drift["passed"] is False
    assert drift["rows"][0]["inputs_equal"] is True
    assert drift["rows"][0]["logits_equal"] is False
