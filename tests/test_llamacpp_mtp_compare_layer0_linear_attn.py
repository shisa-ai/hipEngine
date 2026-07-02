from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.llamacpp_mtp_compare_layer0_linear_attn import (
    _attn_norm_formula_assessment,
    _stable_split_assessment,
    build_linear_attn_compare_artifact,
)


def test_build_linear_attn_compare_artifact_reports_early_attention_drift(
    tmp_path: Path,
) -> None:
    hip_path = tmp_path / "hip.json"
    llama_path = tmp_path / "llama.jsonl"
    hip_path.write_text(json.dumps(_hip_artifact()) + "\n")
    llama_path.write_text(json.dumps(_llama_cycle()) + "\n")

    artifact = build_linear_attn_compare_artifact(
        hipengine_raw_path=hip_path,
        llamacpp_jsonl_path=llama_path,
        llamacpp_cycle=18,
        row=1,
        layer=0,
    )

    assert artifact["status"] == "complete"
    assert artifact["performance_claim"] is False
    assert artifact["tensor_deltas"]["linear_attn_out"]["mean_abs_diff"] == 0.5
    assert artifact["tensor_deltas"]["attn_residual"]["mean_abs_diff"] == 1.0
    assert artifact["tensor_deltas"]["post_moe"]["mean_abs_diff"] == 0.0
    assert artifact["tensor_deltas"]["layer_out"]["mean_abs_diff"] == 0.0
    assert artifact["pre_ssm_stable_deltas"]["conv_output_silu"][
        "mean_abs_diff"
    ] == 0.0
    assert artifact["conv_view_deltas"]["q_conv"]["mean_abs_diff"] == 0.0
    assert artifact["conv_view_deltas"]["v_conv"]["hipengine_slice"] == [4, 6]
    assert artifact["trace_label_caveats"]["linear_attn_qkv_mixed"]["status"] == (
        "layout_or_value_extraction_ambiguous"
    )
    assert artifact["trace_label_caveats"]["alpha"]["status"] == (
        "aliases_gate_or_mutated_value"
    )
    assert artifact["stable_split_assessment"]["status"] == "no_projection_or_conv_cliff"
    assert artifact["final_output_summary"]["count"] == 4
    assert artifact["pre_ssm_out_deltas"]["recurrent_out_vs_final_output"][
        "status"
    ] == "missing"
    assert artifact["pre_ssm_out_label_assessment"]["status"] == "unavailable"
    assert artifact["llamacpp"]["duplicate_value_labels"]["linear_attn_out_0"][
        "max_abs_vs_first"
    ] == 0.0
    assert "earliest complete input/pre-SSM delta" in artifact["conclusion"]
    json.dumps(artifact)


def test_build_linear_attn_compare_artifact_marks_final_output_layout_unresolved(
    tmp_path: Path,
) -> None:
    hip_path = tmp_path / "hip.json"
    llama_path = tmp_path / "llama.jsonl"
    hip_path.write_text(json.dumps(_hip_artifact_with_recurrent()) + "\n")
    llama_path.write_text(json.dumps(_llama_cycle_with_matching_ssm_out()) + "\n")

    artifact = build_linear_attn_compare_artifact(
        hipengine_raw_path=hip_path,
        llamacpp_jsonl_path=llama_path,
        llamacpp_cycle=18,
        row=1,
        layer=0,
    )

    assert artifact["tensor_deltas"]["linear_attn_out"]["mean_abs_diff"] == 0.0
    recurrent_delta = artifact["pre_ssm_out_deltas"]["recurrent_out_vs_final_output"]
    assert recurrent_delta["status"] == "complete"
    assert recurrent_delta["mean_abs_diff"] == 2.5
    assert artifact["pre_ssm_out_label_assessment"]["status"] == (
        "unresolved_label_or_layout"
    )
    assert "label/layout unresolved" in artifact["conclusion"]
    json.dumps(artifact)


def test_build_linear_attn_compare_artifact_can_use_scored_capture(
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

    artifact = build_linear_attn_compare_artifact(
        hipengine_raw_path=hip_path,
        llamacpp_jsonl_path=llama_path,
        llamacpp_cycle=18,
        row=1,
        layer=0,
        hipengine_capture_source="scored",
    )

    assert artifact["inputs"]["hipengine_capture_source"] == "scored_target_block"
    assert artifact["hipengine"]["capture_source"] == "scored_target_block"
    assert artifact["tensor_deltas"]["linear_attn_out"]["mean_abs_diff"] == 0.5
    json.dumps(artifact)


def test_build_linear_attn_compare_artifact_filters_llamacpp_task_id(
    tmp_path: Path,
) -> None:
    hip_path = tmp_path / "hip.json"
    llama_path = tmp_path / "llama.jsonl"
    hip_path.write_text(json.dumps(_hip_artifact()) + "\n")
    decoy = _llama_cycle()
    decoy["task_id"] = 0
    decoy["draft_hidden_state_trace"] = [
        {
            **trace,
            "values": [100.0 for _ in trace["values"]],
        }
        for trace in decoy["draft_hidden_state_trace"]
        if trace.get("row_index") == 1
    ]
    target = _llama_cycle()
    target["task_id"] = 9
    llama_path.write_text(json.dumps(decoy) + "\n" + json.dumps(target) + "\n")

    artifact = build_linear_attn_compare_artifact(
        hipengine_raw_path=hip_path,
        llamacpp_jsonl_path=llama_path,
        llamacpp_cycle=18,
        llamacpp_task_id=9,
        row=1,
        layer=0,
    )

    assert artifact["inputs"]["llamacpp_task_id"] == 9
    assert artifact["llamacpp"]["task_id"] == 9
    assert artifact["tensor_deltas"]["linear_attn_out"]["mean_abs_diff"] == 0.5
    json.dumps(artifact)


def test_build_linear_attn_compare_artifact_caveats_beta_when_output_is_close(
    tmp_path: Path,
) -> None:
    hip_path = tmp_path / "hip.json"
    llama_path = tmp_path / "llama.jsonl"
    llama = _llama_cycle()
    for trace in llama["draft_hidden_state_trace"]:
        if trace.get("label") == "beta_0" and trace.get("row_index") == 1:
            trace["values"] = [100.0, 100.0]
        if trace.get("label") == "linear_attn_out_0" and trace.get("row_index") == 1:
            trace["values"] = [1.5, 2.5]
    hip_path.write_text(json.dumps(_hip_artifact()) + "\n")
    llama_path.write_text(json.dumps(llama) + "\n")

    artifact = build_linear_attn_compare_artifact(
        hipengine_raw_path=hip_path,
        llamacpp_jsonl_path=llama_path,
        llamacpp_cycle=18,
        row=1,
        layer=0,
    )

    assert artifact["tensor_deltas"]["linear_attn_out"]["mean_abs_diff"] == 0.0
    assert artifact["pre_ssm_stable_deltas"]["beta_projection"]["mean_abs_diff"] > 1.0
    assert artifact["trace_label_caveats"]["beta"]["status"] == (
        "layout_or_value_extraction_ambiguous"
    )
    json.dumps(artifact)


def test_attn_norm_formula_assessment_can_explain_input_delta(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_text("fake")

    def weight_loader(_model: Path, layer: int):
        assert layer == 14
        return np.ones(2, dtype=np.float32), 0.0, {"tensor_name": "fake"}

    artifact = _attn_norm_formula_assessment(
        hip_values={
            "hidden_in": np.asarray([1.0, 1.0], dtype=np.float32),
            "attn_norm": np.asarray([1.0, 1.0], dtype=np.float32),
        },
        llama_values={
            "verify_layer_output_13": np.asarray([1.0, -1.0], dtype=np.float32),
            "attn_norm_14": np.asarray([1.0, -1.0], dtype=np.float32),
        },
        layer=14,
        model_path=model,
        weight_loader=weight_loader,
    )

    assert artifact["status"] == "complete"
    assert artifact["classification"] == "attn_norm_delta_explained_by_input_delta"
    assert artifact["best_llamacpp_candidate"]["delta"]["exact_match"] is True
    assert artifact["best_hipengine_candidate"]["delta"]["exact_match"] is True
    json.dumps(artifact)


def test_stable_split_assessment_uses_current_layer_in_reason() -> None:
    artifact = {
        "inputs": {"layer": 9},
        "input_boundary_deltas": {
            "hidden_in_vs_prev_layer_output": {
                "status": "complete",
                "mean_abs_diff": 0.0005,
            },
            "attn_norm_input": {
                "status": "complete",
                "mean_abs_diff": 0.014,
            },
        },
        "attn_norm_formula_assessment": {
            "classification": "attn_norm_delta_explained_by_input_delta",
        },
        "pre_ssm_stable_deltas": {
            "z_projection": {"status": "complete", "mean_abs_diff": 0.012},
            "beta_projection": {"status": "complete", "mean_abs_diff": 0.012},
            "conv_output_silu": {"status": "complete", "mean_abs_diff": 0.0007},
        },
        "conv_view_deltas": {
            "q_conv": {"status": "complete", "mean_abs_diff": 0.0007},
            "k_conv": {"status": "complete", "mean_abs_diff": 0.0007},
            "v_conv": {"status": "complete", "mean_abs_diff": 0.0007},
        },
        "tensor_deltas": {
            "linear_attn_out": {"mean_abs_diff": 0.0004},
            "attn_post_norm": {"mean_abs_diff": 0.018},
            "post_moe": {"mean_abs_diff": 0.00065},
        },
    }

    assessment = _stable_split_assessment(artifact)

    assert "layer-9 norm-space drift" in assessment["reason"]
    assert "layer-8 output drift" in assessment["reason"]
    assert "layer-14" not in assessment["reason"]


def _hip_artifact() -> dict:
    return {
        "model": "/models/fake.gguf",
        "source_trace": "/tmp/source.json",
        "command": "fake hip command",
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
                    "values": _hip_values(),
                }
            ],
        },
    }


def _hip_values() -> dict:
    return {
        "linear_qkv": [10.0, 10.0],
        "linear_z": [1.0, 2.0],
        "ssm_alpha": [9.0, 9.0],
        "ssm_beta": [3.0, 4.0],
        "conv_out": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "attn_out": [1.5, 2.5],
        "attn_residual": [4.0, 5.0],
        "attn_post_norm": [6.5, 7.5],
        "ffn_out_combined_from_components": [8.0, 9.0],
        "post_moe_rounded_from_components": [10.0, 11.0],
        "layer_out": [10.0, 11.0],
    }


def _llama_cycle() -> dict:
    values_by_label = {
        "linear_attn_qkv_mixed_0": [0.0, 0.0],
        "z_0": [1.001, 2.001],
        "alpha_0": [7.0, 7.0],
        "beta_0": [3.002, 4.002],
        "gate_0": [7.0, 7.0],
        "conv_output_silu_0": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "q_conv_0": [0.0, 1.0],
        "k_conv_0": [2.0, 3.0],
        "v_conv_0": [4.0, 5.0],
        "final_output_0": [0.0, 1.0, 2.0, 3.0],
        "linear_attn_out_0": [1.0, 2.0],
        "attn_residual_0": [3.0, 4.0],
        "attn_post_norm_0": [6.0, 7.0],
        "ffn_out_0": [8.0, 9.0],
        "post_moe_0": [10.0, 11.0],
    }
    trace = [
        {"label": label, "row_index": 1, "values": values}
        for label, values in values_by_label.items()
    ]
    trace.append({"label": "linear_attn_out_0", "row_index": 1, "values": [1.0, 2.0]})
    trace.append({"label": "linear_attn_out_0", "row_index": 0, "values": [9.0, 9.0]})
    return {
        "cycle": 18,
        "accepted_draft_tokens": 1,
        "accepted_token_ids": [15495],
        "bonus_token_id": 26126,
        "cycle_wall_ms": 215.0,
        "draft_hidden_state_trace": trace,
    }


def _hip_artifact_with_recurrent() -> dict:
    artifact = _hip_artifact()
    values = artifact["result"]["layer_boundary_captures"][0]["values"]
    values.update(
        {
            "attn_out": [1.0, 2.0],
            "attn_residual": [3.0, 4.0],
            "attn_post_norm": [5.0, 6.0],
            "ffn_out_combined_from_components": [7.0, 8.0],
            "post_moe_rounded_from_components": [9.0, 10.0],
            "layer_out": [9.0, 10.0],
            "recurrent_out": [0.0, 0.0, 0.0, 0.0],
            "recurrent_bf16": [0.0, 0.0, 0.0, 0.0],
        }
    )
    return artifact


def _llama_cycle_with_matching_ssm_out() -> dict:
    values_by_label = {
        "final_output_0": [1.0, 2.0, 3.0, 4.0],
        "linear_attn_out_0": [1.0, 2.0],
        "attn_residual_0": [3.0, 4.0],
        "attn_post_norm_0": [5.0, 6.0],
        "ffn_out_0": [7.0, 8.0],
        "post_moe_0": [9.0, 10.0],
    }
    return {
        "cycle": 18,
        "accepted_draft_tokens": 1,
        "accepted_token_ids": [15495],
        "bonus_token_id": 26126,
        "cycle_wall_ms": 215.0,
        "draft_hidden_state_trace": [
            {"label": label, "row_index": 1, "values": values}
            for label, values in values_by_label.items()
        ],
    }
