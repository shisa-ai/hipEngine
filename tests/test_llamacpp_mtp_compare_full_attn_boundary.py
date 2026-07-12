from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.llamacpp_mtp_compare_full_attn_boundary import (
    build_full_attn_compare_artifact,
)


def test_build_full_attn_compare_artifact_reports_full_attention_deltas(
    tmp_path: Path,
) -> None:
    hip_path = tmp_path / "hip.json"
    llama_path = tmp_path / "llama.jsonl"
    hip_path.write_text(json.dumps(_hip_artifact()) + "\n")
    llama_path.write_text(json.dumps(_llama_cycle()) + "\n")

    artifact = build_full_attn_compare_artifact(
        hipengine_raw_path=hip_path,
        llamacpp_jsonl_path=llama_path,
        llamacpp_cycle=3,
        row=2,
        layer=11,
    )

    assert artifact["status"] == "complete"
    assert artifact["performance_claim"] is False
    assert artifact["kind"] == "mtp_target_full_attn_cross_engine_compare"
    assert artifact["tensor_deltas"]["attn_out"]["mean_abs_diff"] == 0.5
    assert artifact["tensor_deltas"]["attn_residual"]["mean_abs_diff"] == 0.0
    assert artifact["tensor_deltas"]["post_moe"]["mean_abs_diff"] == 0.0
    assert artifact["optional_tensor_deltas"]["layer_out_vs_verify_layer_output"][
        "mean_abs_diff"
    ] == 0.0
    assert artifact["process_h_input_context"]["status"] == "context_only"
    assert artifact["boundary_assessment"]["status"] == "full_attention_output_cliff"
    assert "Layer-11 full-attention split" in artifact["conclusion"]
    json.dumps(artifact)


def test_build_full_attn_compare_artifact_filters_task_and_checks_rmsnorm(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_text("fake")
    hip = _hip_artifact(model=str(model))
    hip_path = tmp_path / "hip.json"
    llama_path = tmp_path / "llama.jsonl"
    decoy = _llama_cycle(task_id=0)
    decoy["draft_hidden_state_trace"] = [
        {**trace, "values": [100.0 for _ in trace["values"]]}
        for trace in decoy["draft_hidden_state_trace"]
        if trace.get("row_index") == 2
    ]
    target = _llama_cycle(task_id=9)
    hip_path.write_text(json.dumps(hip) + "\n")
    llama_path.write_text(json.dumps(decoy) + "\n" + json.dumps(target) + "\n")

    def weight_loader(_model: Path, layer: int):
        assert layer == 11
        return np.ones(2, dtype=np.float32), 0.0, {"tensor_name": "fake"}

    artifact = build_full_attn_compare_artifact(
        hipengine_raw_path=hip_path,
        llamacpp_jsonl_path=llama_path,
        llamacpp_cycle=3,
        llamacpp_task_id=9,
        row=2,
        layer=11,
        attn_norm_weight_loader=weight_loader,
    )

    assert artifact["inputs"]["llamacpp_task_id"] == 9
    assert artifact["llamacpp"]["task_id"] == 9
    assert artifact["attn_norm_formula_assessment"]["classification"] == (
        "attn_norm_delta_explained_by_input_delta"
    )
    assert artifact["llamacpp"]["duplicate_value_labels"]["attn_output_11"][
        "max_abs_vs_first"
    ] == 0.0
    json.dumps(artifact)


def test_build_full_attn_compare_artifact_can_use_scored_capture(
    tmp_path: Path,
) -> None:
    hip = _hip_artifact()
    scored = dict(hip["result"]["layer_boundary_captures"][0])
    scored["capture_source"] = "scored_target_block"
    hip["result"]["scored_layer_boundary_captures"] = [scored]
    hip["result"]["layer_boundary_captures"] = []
    hip_path = tmp_path / "hip.json"
    llama_path = tmp_path / "llama.jsonl"
    hip_path.write_text(json.dumps(hip) + "\n")
    llama_path.write_text(json.dumps(_llama_cycle()) + "\n")

    artifact = build_full_attn_compare_artifact(
        hipengine_raw_path=hip_path,
        llamacpp_jsonl_path=llama_path,
        llamacpp_cycle=3,
        row=2,
        layer=11,
        hipengine_capture_source="scored",
    )

    assert artifact["inputs"]["hipengine_capture_source"] == "scored_target_block"
    assert artifact["hipengine"]["capture_source"] == "scored_target_block"
    assert artifact["tensor_deltas"]["attn_out"]["mean_abs_diff"] == 0.5
    json.dumps(artifact)


def _hip_artifact(model: str = "/models/fake.gguf") -> dict:
    return {
        "model": model,
        "source_trace": "/tmp/source.json",
        "command": "fake hip command",
        "probe": {"cycle": 3},
        "result": {
            "sampled_tokens": [11, 567, 8940],
            "accepted_draft_tokens": 2,
            "layer_boundary_captures": [
                {
                    "layer": 11,
                    "row": 2,
                    "position": 75,
                    "input_token": 567,
                    "trace_target_token": 8940,
                    "values": {
                        "hidden_in": [1.0, -1.0],
                        "attn_norm": [1.0, -1.0],
                        "attn_out": [2.0, 3.0],
                        "attn_residual": [4.0, 5.0],
                        "attn_post_norm": [6.0, 7.0],
                        "attn_post_norm_bf16": [6.0, 7.0],
                        "attn_post_norm_router_input": [6.0, 7.0],
                        "ffn_out_combined_from_components": [8.0, 9.0],
                        "post_moe_rounded_from_components": [10.0, 11.0],
                        "layer_out": [10.0, 11.0],
                    },
                }
            ],
        },
    }


def _llama_cycle(task_id: int | None = None) -> dict:
    values_by_label = {
        "process_h_input": [99.0, 99.0],
        "verify_layer_output_10": [1.0, -1.0],
        "attn_norm_11": [1.0, -1.0],
        "attn_output_11": [1.5, 2.5],
        "attn_residual_11": [4.0, 5.0],
        "attn_post_norm_11": [6.0, 7.0],
        "ffn_out_11": [8.0, 9.0],
        "post_moe_11": [10.0, 11.0],
        "verify_layer_output_11": [10.0, 11.0],
    }
    trace = [
        {"label": label, "row_index": 2, "values": values}
        for label, values in values_by_label.items()
    ]
    trace.append({"label": "attn_output_11", "row_index": 2, "values": [1.5, 2.5]})
    trace.append({"label": "attn_output_11", "row_index": 1, "values": [9.0, 9.0]})
    cycle = {
        "cycle": 3,
        "accepted_draft_tokens": 2,
        "accepted_token_ids": [11, 567],
        "bonus_token_id": 668,
        "cycle_wall_ms": 100.0,
        "draft_hidden_state_trace": trace,
    }
    if task_id is not None:
        cycle["task_id"] = task_id
    return cycle
