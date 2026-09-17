"""CPU tests for the prefill-schedule suite rollup.

No GPU, no model, no capture: the test synthesizes probe pass artifacts (the
same shape ``tp2_prefill_schedule_failure_probe.py`` writes) and drives the
rollup end to end, so the two numbers the sustained gate reads — the authorized
horizon and the per-prompt first breaching depth — are pinned by construction
rather than by a recorded run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.tp2_prefill_schedule_suite_rollup as rollup

STEPS = 16


def _write_teacher(tmp_path: Path, curves_by_prompt: dict[str, int]) -> Path:
    """A teacher capture whose rows are as many distinct candidates as asked."""

    arrays = tmp_path / "arrays"
    arrays.mkdir(exist_ok=True)
    entries = []
    for prompt_id, top in curves_by_prompt.items():
        rows = np.zeros((STEPS, 8), dtype=np.float32)
        for index in range(STEPS):
            rows[index, top] = 1.0
        path = arrays / f"{prompt_id}.npy"
        np.save(path, rows)
        entries.append(
            {
                "prompt_id": prompt_id,
                "path": str(path),
                "sha256": "0" * 64,
                "rows": STEPS,
                "shape": [STEPS, 8],
            }
        )
    teacher = {
        "status": "complete",
        "suite": {
            "ids": list(curves_by_prompt),
            "tokens": [[1, 2] for _ in curves_by_prompt],
        },
        "arrays": entries,
        "forced_inputs": [[3] * STEPS for _ in curves_by_prompt],
    }
    path = tmp_path / "teacher.json"
    path.write_text(json.dumps(teacher))
    return path


def _write_pass(
    tmp_path: Path,
    arm: str,
    teacher_path: Path,
    curves: dict[str, list[float]],
    *,
    schedule: str = "token-serial",
    teacher_sha: str | None = None,
) -> Path:
    import hashlib

    sha = teacher_sha or hashlib.sha256(teacher_path.read_bytes()).hexdigest()
    artifact = {
        "kind": "tp2_prefill_schedule_failure_probe",
        "schema": 2,
        "status": "diagnostic_only",
        "teacher_sha256": sha,
        "source_revision": "deadbeef",
        "source_sha256": {"scripts/tp2_resident_control.py": "0" * 64},
        "teacher_identity_diff": {},
        "arms": {arm: {"prefill_schedule": schedule}},
        "prompts": {
            prompt_id: {
                "comparisons": {
                    f"teacher_vs_{arm}": {"kl_curve": list(values)},
                }
            }
            for prompt_id, values in curves.items()
        },
        "suite_aggregate": {arm: {"prompts": len(curves)}},
    }
    path = tmp_path / f"{arm}.json"
    path.write_text(json.dumps(artifact))
    return path


def test_envelope_failures_names_each_violated_statistic() -> None:
    assert rollup._envelope_failures(np.zeros(8)) == []
    # One row over the max ceiling also drags p99 and the mean over theirs.
    spiky = np.zeros(STEPS)
    spiky[3] = 0.5
    assert rollup._envelope_failures(spiky) == [
        "mean_kl",
        "p95_kl",
        "p99_kl",
        "max_kl",
    ]
    assert rollup._envelope_failures(np.array([])) == ["no rows"]


def test_first_breaching_depth_and_horizon_follow_the_floor() -> None:
    inside = np.zeros(16)
    late = np.zeros(16)
    late[11] = 0.5
    # Below the floor a percentile over two rows is the maximum, so depths
    # inside the floor are not reported.
    assert rollup._first_breaching_depth(late, floor=8) == 12
    assert rollup._first_breaching_depth(inside, floor=8) is None
    # The floor gates the depths tried, not the rows inspected: a spike inside
    # the floor still fails the first prefix that reaches it.
    assert rollup._first_breaching_depth(late, floor=14) == 15

    horizon = rollup._authorized_horizon({"inside": inside, "late": late}, floor=8)
    assert horizon["authorized_horizon"] == 11
    assert horizon["first_failure"] == {
        "depth": 12,
        "prompts": {"late": ["mean_kl", "p95_kl", "p99_kl", "max_kl"]},
    }


def test_main_merges_passes_and_reports_identical_breaching_depths(tmp_path: Path) -> None:
    teacher_path = _write_teacher(tmp_path, {"inside": 0, "late": 1})
    inside = [0.0] * STEPS
    late = [0.0] * 11 + [0.5] * (STEPS - 11)
    paths = [
        _write_pass(tmp_path, "tp1-serial", teacher_path, {"inside": inside, "late": late}, schedule="token-serial"),
        _write_pass(tmp_path, "tp2-serial", teacher_path, {"inside": inside, "late": late}, schedule="token-serial"),
    ]
    out = tmp_path / "rollup.json"
    assert (
        rollup.main(
            [
                "--teacher",
                str(teacher_path),
                "--pass",
                f"tp1-serial={paths[0]}",
                "--pass",
                f"tp2-serial={paths[1]}",
                "--steps",
                str(STEPS),
                "--json",
                str(out),
            ]
        )
        == 0
    )
    artifact = json.loads(out.read_text())
    assert artifact["kind"] == "tp2_prefill_schedule_suite_rollup"
    assert artifact["performance_claim"] is False
    for arm in ("tp1-serial", "tp2-serial"):
        assert artifact["arms"][arm]["authorized_horizon"] == 11
        assert artifact["arms"][arm]["first_breaching_depth"] == {"inside": None, "late": 12}
        assert artifact["arms"][arm]["prefill_schedule"] == "token-serial"
    assert artifact["prompts"]["late"]["arms"]["tp2-serial"]["positions_over_max_kl"] == 5
    assert artifact["prompts"]["late"]["arms"]["tp2-serial"]["failures"] == [
        "mean_kl",
        "p95_kl",
        "p99_kl",
        "max_kl",
    ]
    assert artifact["prompts"]["inside"]["arms"]["tp1-serial"]["failures"] == []
    # The fixture's teacher rows put a logit of 1.0 on one candidate and 0.0 on
    # the other seven, so its top-1 probability is e/(e+7) and its top-2 gap 1.0.
    assert artifact["teacher_flatness"]["large_kl_rows"] == 10
    assert artifact["teacher_flatness"]["large_kl_top1_prob_median"] == pytest.approx(
        2.718281828 / (2.718281828 + 7), abs=1e-6
    )
    assert artifact["teacher_flatness"]["large_kl_top2_gap_median"] == pytest.approx(1.0)


def test_main_refuses_passes_from_a_different_source_revision(tmp_path: Path) -> None:
    teacher_path = _write_teacher(tmp_path, {"inside": 0})
    first = _write_pass(tmp_path, "tp1-serial", teacher_path, {"inside": [0.0] * STEPS})
    second = _write_pass(tmp_path, "tp2-serial", teacher_path, {"inside": [0.0] * STEPS})
    artifact = json.loads(second.read_text())
    artifact["source_sha256"] = {"scripts/tp2_resident_control.py": "1" * 64}
    second.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="different source revisions"):
        rollup.main(
            [
                "--teacher",
                str(teacher_path),
                "--pass",
                f"tp1-serial={first}",
                "--pass",
                f"tp2-serial={second}",
                "--json",
                str(tmp_path / "rollup.json"),
            ]
        )


def test_main_refuses_passes_from_a_different_teacher(tmp_path: Path) -> None:
    teacher_path = _write_teacher(tmp_path, {"inside": 0})
    good = _write_pass(tmp_path, "tp1-serial", teacher_path, {"inside": [0.0] * STEPS})
    bad = _write_pass(
        tmp_path, "tp2-serial", teacher_path, {"inside": [0.0] * STEPS}, teacher_sha="f" * 64
    )
    with pytest.raises(ValueError, match="different teacher capture"):
        rollup.main(
            [
                "--teacher",
                str(teacher_path),
                "--pass",
                f"tp1-serial={good}",
                "--pass",
                f"tp2-serial={bad}",
                "--json",
                str(tmp_path / "rollup.json"),
            ]
        )
