from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_compare_layer0_linear_attn import (
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
    assert artifact["final_output_summary"]["count"] == 4
    assert artifact["llamacpp"]["duplicate_value_labels"]["linear_attn_out_0"][
        "max_abs_vs_first"
    ] == 0.0
    assert "layer-0 linear-attention/GDN output contract" in artifact["conclusion"]
    json.dumps(artifact)


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
        "attn_out": [1.5, 2.5],
        "attn_residual": [4.0, 5.0],
        "attn_post_norm": [6.5, 7.5],
        "ffn_out_combined_from_components": [8.0, 9.0],
        "post_moe_rounded_from_components": [10.0, 11.0],
        "layer_out": [10.0, 11.0],
    }


def _llama_cycle() -> dict:
    values_by_label = {
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
