from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.llamacpp_mtp_validate_final_output_projection import (
    build_projection_validation_artifact,
)


def test_build_projection_validation_artifact_marks_unprojectable_final_output(
    tmp_path: Path,
) -> None:
    hip_path = tmp_path / "hip.json"
    llama_path = tmp_path / "llama.jsonl"
    hip_path.write_text(json.dumps(_hip_artifact()) + "\n")
    llama_path.write_text(json.dumps(_llama_cycle()) + "\n")

    def project(values: np.ndarray) -> np.ndarray:
        if np.allclose(values, np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)):
            return np.asarray([1.0, 2.0], dtype=np.float32)
        return np.asarray([9.0, 9.0], dtype=np.float32)

    artifact = build_projection_validation_artifact(
        model_path="/models/fake.gguf",
        hipengine_raw_path=hip_path,
        llamacpp_jsonl_path=llama_path,
        llamacpp_cycle=18,
        row=1,
        layer=0,
        command="fake command",
        project_fn=project,
    )

    assert artifact["status"] == "complete"
    assert artifact["performance_claim"] is False
    assert artifact["assessment"]["status"] == "final_output_trace_not_projectable"
    assert (
        artifact["reprojection_deltas"][
            "hip_recurrent_reproject_vs_hip_capture_attn_out"
        ]["mean_abs_diff"]
        == 0.0
    )
    assert (
        artifact["reprojection_deltas"]["llama_final_reproject_vs_llama_linear"][
            "mean_abs_diff"
        ]
        == 7.5
    )
    assert "final_output_trace_not_projectable" in artifact["conclusion"]
    json.dumps(artifact)


def _hip_artifact() -> dict:
    return {
        "probe": {"cycle": 12},
        "result": {
            "sampled_tokens": [15495, 539],
            "accepted_draft_tokens": 2,
            "layer_boundary_captures": [
                {
                    "layer": 0,
                    "row": 1,
                    "position": 73,
                    "input_token": 15495,
                    "trace_target_token": 539,
                    "values": {
                        "recurrent_out": [0.0, 0.0, 0.0, 0.0],
                        "attn_out": [1.0, 2.0],
                    },
                }
            ],
        },
    }


def _llama_cycle() -> dict:
    return {
        "cycle": 18,
        "accepted_draft_tokens": 1,
        "accepted_token_ids": [15495],
        "bonus_token_id": 26126,
        "draft_hidden_state_trace": [
            {
                "label": "final_output_0",
                "row_index": 1,
                "values": [1.0, 2.0, 3.0, 4.0],
            },
            {
                "label": "linear_attn_out_0",
                "row_index": 1,
                "values": [1.0, 2.0],
            },
        ],
    }
