from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_compare_target_moe_taps import build_moe_tap_compare_artifact


def test_build_moe_tap_compare_artifact_aligns_router_and_segments(tmp_path: Path) -> None:
    hip_path = tmp_path / "hip.json"
    llama_path = tmp_path / "llama.jsonl"
    hip_path.write_text(json.dumps(_hip_artifact()) + "\n")
    llama_path.write_text(json.dumps(_llama_cycle()) + "\n")

    artifact = build_moe_tap_compare_artifact(
        hipengine_raw_path=hip_path,
        llamacpp_jsonl_path=llama_path,
        llamacpp_cycle=18,
        row=1,
        layer=31,
    )

    assert artifact["status"] == "complete"
    assert artifact["router"]["topk_match"] is True
    assert artifact["selection"]["common_experts"] == [0, 1]
    assert artifact["llamacpp"]["duplicate_value_labels"]["ffn_moe_out_31"][
        "max_abs_vs_first"
    ] == 0.0
    assert artifact["segment_deltas"]["down"]["segment_width"] == 3
    assert artifact["tensor_deltas"]["ffn_out"]["mean_abs_diff"] == 0.0


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
                    "layer": 31,
                    "row": 1,
                    "position": 73,
                    "input_token": 15495,
                    "trace_target_token": 539,
                    "moe_selected_experts": [0, 1],
                    "moe_routing_weights": [0.6, 0.4],
                    "moe_shared_gate": [0.25],
                    "values": _hip_values(),
                }
            ],
        },
    }


def _hip_values() -> dict:
    return {
        "moe_router_logits": [0.9, 0.8, 0.1],
        "moe_selected_swiglu": [1.0, 2.0, 3.0, 4.0],
        "ffn_or_moe_down": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "moe_selected_down_weighted": [0.6, 1.2, 1.8, 1.6, 2.0, 2.4],
        "moe_selected_weighted_sum_f32": [2.2, 3.2, 4.2],
        "moe_selected_weighted_bf16": [2.2, 3.2, 4.2],
        "moe_shared_out": [0.1, 0.2, 0.3],
        "moe_shared_gated": [0.05, 0.1, 0.15],
        "ffn_out_combined_from_components": [2.25, 3.3, 4.35],
        "post_moe_rounded_from_components": [3.0, 4.0, 5.0],
        "layer_out": [3.0, 4.0, 5.0],
    }


def _llama_cycle() -> dict:
    values_by_label = {
        "ffn_moe_logits_31": [0.9, 0.8, 0.1],
        "ffn_moe_weights_norm_31": [0.6, 0.4],
        "ffn_moe_swiglu_31": [1.0, 2.0, 3.0, 4.0],
        "ffn_moe_down_31": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "ffn_moe_weighted_31": [0.6, 1.2, 1.8, 1.6, 2.0, 2.4],
        "ffn_moe_out_31": [2.2, 3.2, 4.2],
        "ffn_shexp_31": [0.1, 0.2, 0.3],
        "shared_expert_gate_31": [0.25],
        "shared_expert_gate_sigmoid_31": [0.5621765],
        "ffn_shexp_gated_31": [0.05, 0.1, 0.15],
        "ffn_out_31": [2.25, 3.3, 4.35],
        "post_moe_31": [3.0, 4.0, 5.0],
        "verify_layer_output_31": [3.0, 4.0, 5.0],
    }
    trace = [
        {"label": label, "row_index": 1, "values": values}
        for label, values in values_by_label.items()
    ]
    trace.append({"label": "ffn_moe_out_31", "row_index": 1, "values": [2.2, 3.2, 4.2]})
    trace.append({"label": "ffn_moe_out_31", "row_index": 0, "values": [9.0, 9.0, 9.0]})
    return {
        "cycle": 18,
        "accepted_draft_tokens": 1,
        "accepted_token_ids": [15495],
        "bonus_token_id": 26126,
        "cycle_wall_ms": 215.0,
        "draft_hidden_state_trace": trace,
    }
