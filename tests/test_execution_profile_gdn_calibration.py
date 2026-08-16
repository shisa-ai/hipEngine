from __future__ import annotations

import numpy as np

from scripts.execution_profile_gdn_calibration import (
    PromptCalibrationCapture,
    _trajectory_arrays,
    build_candidate_quality,
    parse_mode_sources,
)


def _trajectory(*rows: tuple[int, list[float]]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "token_id": token_id,
            "logits": np.asarray(logits, dtype=np.float32),
        }
        for token_id, logits in rows
    )


def test_trajectory_arrays_normalizes_one_singleton_sample_axis() -> None:
    logits, token_ids = _trajectory_arrays(
        ({"token_id": 7, "logits": np.asarray([[4.0, 1.0]], dtype=np.float32)},)
    )

    assert logits.shape == (1, 2)
    assert token_ids == (7,)


def test_candidate_quality_uses_full_logits_and_checks_three_run_repeat() -> None:
    strict = _trajectory(
        (0, [4.0, 1.0, 0.0]),
        (1, [0.0, 4.0, 1.0]),
    )
    candidate = _trajectory(
        (0, [4.0, 1.0, 0.0]),
        (1, [0.0, 3.8, 1.2]),
    )
    captures = (
        PromptCalibrationCapture(
            prompt_id="code-a",
            category="code",
            strict=strict,
            candidate_runs={"peer": (candidate, candidate, candidate)},
        ),
        PromptCalibrationCapture(
            prompt_id="ja-a",
            category="general_ja",
            strict=strict,
            candidate_runs={"peer": (strict, strict, strict)},
        ),
    )

    result = build_candidate_quality(
        captures,
        candidate_mode="peer",
        scenario_id="synthetic-gdn-c1",
    )

    assert result["quality"]["summary"]["rows"] == 4
    assert result["quality"]["by_scope"]["category"]["code"]["rows"] == 2
    assert result["quality"]["summary"]["kl_max"] > 0.0
    assert result["repeat_determinism"]["runs"] == 3
    assert result["repeat_determinism"]["passed"] is True
    assert len(result["strict_logits_sha256"]) == 64
    assert len(result["candidate_logits_sha256"]) == 64


def test_candidate_quality_localizes_repeat_drift() -> None:
    strict = _trajectory((0, [4.0, 1.0]), (1, [1.0, 4.0]))
    changed = _trajectory((0, [4.0, 1.0]), (0, [1.0, 4.0]))
    capture = PromptCalibrationCapture(
        prompt_id="general-a",
        category="general_en",
        strict=strict,
        candidate_runs={"candidate": (strict, changed, strict)},
    )

    result = build_candidate_quality(
        (capture,),
        candidate_mode="candidate",
        scenario_id="synthetic-gdn-c1",
    )

    assert result["repeat_determinism"]["passed"] is False
    assert result["repeat_determinism"]["mismatches"] == [
        {
            "prompt_id": "general-a",
            "repeat_index": 1,
            "logits_exact": True,
            "selected_token_ids_exact": False,
        }
    ]


def test_parse_mode_sources_requires_one_source_per_declared_control(tmp_path) -> None:
    positive = tmp_path / "positive.json"
    negative = tmp_path / "negative.json"
    positive.write_text('{"status":"retained"}', encoding="utf-8")
    negative.write_text('{"status":"rejected"}', encoding="utf-8")

    result = parse_mode_sources(
        positive_modes=("peer",),
        negative_modes=("k2",),
        source_values=(f"peer={positive}", f"k2={negative}"),
    )

    assert result["peer"]["expected_label"] == "positive"
    assert result["peer"]["historical_status"] == "retained"
    assert result["k2"]["expected_label"] == "negative"
    assert result["k2"]["historical_status"] == "rejected"
