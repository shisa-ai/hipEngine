from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gguf_mtp_compare_forced_target_paths.py"


def _artifact(*, sampled_token: int, accepted: int, margin_shift: float) -> dict[str, object]:
    return {
        "probe": {"cycle": 12},
        "result": {
            "capture_linear_state_rows": sampled_token == 539,
            "target_block_verify_mode": "bulk",
            "sampled_tokens": [15495, sampled_token, 1151],
            "accepted_draft_tokens": accepted,
            "scored_layer_boundary_captures": [
                {
                    "layer": 0,
                    "row": 1,
                    "values": {
                        "hidden_in": [1.0, 2.0, 3.0],
                        "attn_out": [4.0, 5.0, 6.0],
                        "layer_out": [7.0, 8.0, 9.0],
                        "moe_selected_experts": [1, 2],
                    },
                },
                {
                    "layer": 1,
                    "row": 1,
                    "values": {
                        "hidden_in": [7.0, 8.0, 9.0],
                        "layer_out": [10.0, 11.0, 12.0],
                    },
                },
            ],
            "rows": [
                {"row": 0},
                {
                    "row": 1,
                    "position": 73,
                    "input_token": 15495,
                    "sampled_token": sampled_token,
                    "candidate_scores": [
                        {"token_id": 539, "rank": 1 if sampled_token == 539 else 2, "logit": 10.0 + margin_shift},
                        {"token_id": 26126, "rank": 2 if sampled_token == 539 else 1, "logit": 10.25},
                    ],
                    "layer_output_hidden_values": {
                        "0": [1.0, 2.0, 3.0],
                        "1": [4.0, 5.0, 6.0],
                    },
                    "hidden_seed_values": [0.0, 1.0, 2.0],
                    "pre_output_norm_hidden_values": [2.0, 1.0, 0.0],
                },
            ],
        },
    }


def test_compare_forced_target_paths_reports_margin_and_layer_drift(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "compare.json"

    reference.write_text(json.dumps(_artifact(sampled_token=26126, accepted=1, margin_shift=0.0)), encoding="utf-8")
    candidate_artifact = _artifact(sampled_token=539, accepted=2, margin_shift=0.5)
    row = candidate_artifact["result"]["rows"][1]  # type: ignore[index]
    row["layer_output_hidden_values"]["0"] = [1.0, 2.0003, 3.0]  # type: ignore[index]
    row["layer_output_hidden_values"]["1"] = [4.0, 5.006, 6.0]  # type: ignore[index]
    row["hidden_seed_values"] = [0.0, 1.1, 2.0]  # type: ignore[index]
    candidate.write_text(json.dumps(candidate_artifact), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--reference-label",
            "noncapture",
            "--candidate-label",
            "capture",
            "--row",
            "1",
            "--candidate-tokens",
            "539,26126",
            "--layers",
            "0,1",
            "--boundary-layers",
            "0,1",
            "--ignore-boundary-values",
            "moe_selected_experts",
            "--threshold",
            "0.001",
            "--output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["schema"] == "gguf_mtp_forced_target_path_compare.v1"
    assert artifact["performance_claim"] is False
    assert artifact["sampled_token_changed"] is True
    assert artifact["accepted_draft_tokens_delta"] == 1
    assert artifact["token_margin"]["reference"]["539_minus_26126"] == pytest.approx(-0.25)
    assert artifact["token_margin"]["candidate"]["539_minus_26126"] == pytest.approx(0.25)
    assert artifact["token_margin"]["candidate_minus_reference"] == pytest.approx(0.5)
    first_layer = artifact["summary"]["first_layer_mean_abs_diff_ge_threshold"]
    assert first_layer["layer"] == 1
    assert first_layer["mean_abs_diff"] == pytest.approx(0.002, abs=1.0e-7)
    assert artifact["summary"]["hidden_seed_mean_abs_diff"] == pytest.approx(1.0 / 30.0)
    assert artifact["comparisons"][0]["delta"]["mean_abs_diff"] == pytest.approx(0.0001, abs=1.0e-7)
    assert artifact["boundary_comparisons"][0]["layer"] == 0
    assert artifact["boundary_comparisons"][0]["values"][0]["name"] == "attn_out"
    assert artifact["inputs"]["ignored_boundary_values"] == ["moe_selected_experts"]
    assert "moe_selected_experts" not in {
        value["name"]
        for comparison in artifact["boundary_comparisons"]
        for value in comparison["values"]
    }
    assert artifact["summary"]["boundary_layer_out_mean_abs_diff"] == [
        {"layer": 0, "mean_abs_diff": 0.0},
        {"layer": 1, "mean_abs_diff": 0.0},
    ]


def test_compare_forced_target_paths_can_select_cycle_keyed_artifacts(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "compare.json"

    reference.write_text(
        json.dumps({"12": _artifact(sampled_token=26126, accepted=1, margin_shift=0.0)}),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps({"12": _artifact(sampled_token=539, accepted=2, margin_shift=0.5)}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--cycle",
            "12",
            "--row",
            "1",
            "--candidate-tokens",
            "539,26126",
            "--layers",
            "0",
            "--output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["inputs"]["cycle"] == 12
    assert artifact["paths"]["candidate"]["sampled_tokens"] == [15495, 539, 1151]
