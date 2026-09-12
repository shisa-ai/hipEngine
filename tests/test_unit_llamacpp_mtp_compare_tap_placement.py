from __future__ import annotations

import json
import struct
from pathlib import Path

from scripts.llamacpp_mtp_compare_tap_placement import (
    build_tap_placement_artifact,
    compare_against_arrays,
    conclude,
    rank_comparisons,
)
from scripts.llamacpp_mtp_run_hidden_in_capture import sha256_bytes


def test_compare_tap_placement_ranks_hidden_in_closest(tmp_path: Path) -> None:
    capture_path = _write_capture(tmp_path, [1.0, 2.0, 3.0, 4.0])
    hip_path = _write_hip_arrays(
        tmp_path,
        {
            "hidden_in_f32": [1.0, 2.0, 3.1, 4.0],
            "residual_f32": [10.0, 2.0, 3.0, 4.0],
            "post_norm_f32": [0.0, 0.0, 0.0, 0.0],
            "moe_selected_experts_i64": [1, 2],
        },
    )

    artifact = build_tap_placement_artifact(
        capture_path=capture_path,
        hipengine_arrays_path=hip_path,
    )

    assert artifact["status"] == "mismatched"
    assert artifact["ranking"]["best_same_width"]["key"] == "hidden_in_f32"
    assert artifact["ranking"]["target_rank"] == 1
    assert artifact["conclusion"] == "target_hidden_in_is_closest_but_mismatched"
    assert artifact["skipped_arrays"][0]["key"] == "moe_selected_experts_i64"
    json.dumps(artifact)


def test_compare_tap_placement_detects_non_hidden_closest(tmp_path: Path) -> None:
    capture_path = _write_capture(tmp_path, [1.0, 2.0, 3.0, 4.0])
    hip_path = _write_hip_arrays(
        tmp_path,
        {
            "hidden_in_f32": [9.0, 9.0, 9.0, 9.0],
            "residual_f32": [1.0, 2.0, 3.0, 4.1],
        },
    )

    artifact = build_tap_placement_artifact(
        capture_path=capture_path,
        hipengine_arrays_path=hip_path,
    )

    assert artifact["ranking"]["best_same_width"]["key"] == "residual_f32"
    assert artifact["conclusion"] == "non_hidden_array_is_closest"
    assert "closest_hipengine_array" in artifact["next_action"]


def test_compare_tap_placement_detects_exact_match(tmp_path: Path) -> None:
    capture_path = _write_capture(tmp_path, [1.0, 2.0])
    hip_path = _write_hip_arrays(tmp_path, {"hidden_in_f32": [1.0, 2.0]})

    artifact = build_tap_placement_artifact(
        capture_path=capture_path,
        hipengine_arrays_path=hip_path,
    )

    assert artifact["status"] == "matched"
    assert artifact["conclusion"] == "found_exact_tap_match"
    assert artifact["ranking"]["exact_matches"][0]["key"] == "hidden_in_f32"


def test_compare_against_arrays_skips_bad_shapes_and_non_numeric() -> None:
    comparisons, skipped = compare_against_arrays(
        [1.0, 2.0],
        arrays={
            "good": [1.0, 2.0],
            "wide": [1.0, 2.0, 3.0],
            "non_numeric": ["x", 2.0],
        },
        target_key="good",
    )

    assert [item["key"] for item in comparisons] == ["good"]
    assert {item["key"] for item in skipped} == {"wide", "non_numeric"}


def test_rank_and_conclude_without_same_width_arrays() -> None:
    ranking = rank_comparisons([], target_key="hidden_in_f32")

    assert ranking["best_same_width"] is None
    assert conclude(ranking, target_key="hidden_in_f32") == "no_same_width_arrays"


def _write_capture(tmp_path: Path, values: list[float]) -> Path:
    binary = tmp_path / "llama.f32"
    binary.write_bytes(struct.pack("<" + "f" * len(values), *values))
    artifact = {
        "status": "mismatched",
        "layer": 3,
        "position": 16,
        "capture": {
            "binary_path": str(binary),
            "sha256": sha256_bytes(binary.read_bytes()),
        },
    }
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(artifact))
    return path


def _write_hip_arrays(tmp_path: Path, arrays: dict[str, object]) -> Path:
    artifact = {
        "kind": "hipengine_full_arrays",
        "iteration": 280,
        "model": "model.gguf",
        "layer_id": 3,
        "position": 16,
        "token_id": 271,
        "run_preceding_layers": True,
        "array_keys": list(arrays),
        "arrays": arrays,
    }
    path = tmp_path / "hip.json"
    path.write_text(json.dumps(artifact))
    return path
