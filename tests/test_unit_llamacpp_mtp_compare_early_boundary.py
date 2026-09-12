from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_compare_early_boundary import (
    build_early_boundary_compare_artifact,
)


def test_build_early_boundary_compare_artifact_finds_boundary_and_router_split(
    tmp_path: Path,
) -> None:
    hip_path = tmp_path / "hip.json"
    llama_path = tmp_path / "llama.jsonl"
    hip_path.write_text(json.dumps(_hip_artifact()) + "\n")
    llama_path.write_text(json.dumps(_llama_cycle()) + "\n")

    artifact = build_early_boundary_compare_artifact(
        hipengine_raw_path=hip_path,
        llamacpp_jsonl_path=llama_path,
        llamacpp_cycle=18,
        row=1,
    )

    assert artifact["status"] == "complete"
    assert (
        artifact["boundary_deltas"]["hip_layer0_layer_out_vs_layer1_hidden_in"][
            "mean_abs_diff"
        ]
        == 0.0
    )
    assert (
        artifact["boundary_deltas"]["hip_layer1_hidden_in_vs_llamacpp_post_moe_0"][
            "mean_abs_diff"
        ]
        == 0.5
    )
    assert artifact["router"][0]["topk_match"] is True
    assert artifact["router"][1]["topk_match"] is False
    assert artifact["router"][1]["hip_only_experts"] == [1]
    assert artifact["router"][1]["llamacpp_only_experts"] == [2]
    assert artifact["llamacpp"]["duplicate_value_labels"]["ffn_out_1"][
        "max_abs_vs_first"
    ] == 0.0
    assert artifact["shared_gate"]["layer0"] is None
    assert artifact["shared_gate"]["layer1"]["delta_hip_minus_llama"] == 0.0
    assert artifact["alignment_notes"] == []


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
                _capture(
                    layer=0,
                    logits=[0.9, 0.8, 0.1],
                    experts=[0, 1],
                    layer_out=[2.0, 3.0],
                ),
                _capture(
                    layer=1,
                    logits=[0.9, 0.8, 0.1],
                    experts=[0, 1],
                    layer_out=[4.0, 5.0],
                    hidden_in=[2.0, 3.0],
                    gate=[0.25],
                ),
            ],
        },
    }


def _capture(
    *,
    layer: int,
    logits: list[float],
    experts: list[int],
    layer_out: list[float],
    hidden_in: list[float] | None = None,
    gate: list[float] | None = None,
) -> dict:
    hidden = hidden_in if hidden_in is not None else [1.0, 2.0]
    values = {
        "hidden_in": hidden,
        "attn_norm": [0.1 + layer, 0.2 + layer],
        "attn_residual": [0.3 + layer, 0.4 + layer],
        "attn_post_norm": [0.5 + layer, 0.6 + layer],
        "moe_router_logits": logits,
        "ffn_out_combined_from_components": [0.7 + layer, 0.8 + layer],
        "post_moe_rounded_from_components": layer_out,
        "layer_out": layer_out,
    }
    result = {
        "layer": layer,
        "row": 1,
        "position": 73,
        "input_token": 15495,
        "trace_target_token": 539,
        "moe_selected_experts": experts,
        "moe_routing_weights": [0.6, 0.4],
        "values": values,
    }
    if gate is not None:
        result["moe_shared_gate"] = gate
    return result


def _llama_cycle() -> dict:
    values_by_label = {
        "attn_norm_0": [0.1, 0.2],
        "attn_residual_0": [0.3, 0.4],
        "attn_post_norm_0": [0.5, 0.6],
        "ffn_moe_logits_0": [0.9, 0.8, 0.1],
        "ffn_moe_weights_norm_0": [0.6, 0.4],
        "ffn_out_0": [0.7, 0.8],
        "post_moe_0": [1.5, 2.5],
        "attn_norm_1": [1.1, 1.2],
        "attn_residual_1": [1.3, 1.4],
        "attn_post_norm_1": [1.5, 1.6],
        "ffn_moe_logits_1": [0.9, 0.1, 0.8],
        "ffn_moe_weights_norm_1": [0.6, 0.4],
        "shared_expert_gate_1": [0.25],
        "ffn_out_1": [1.7, 1.8],
        "post_moe_1": [4.0, 5.0],
    }
    trace = [
        {"label": label, "row_index": 1, "values": values}
        for label, values in values_by_label.items()
    ]
    trace.append({"label": "ffn_out_1", "row_index": 1, "values": [1.7, 1.8]})
    trace.append({"label": "ffn_out_1", "row_index": 0, "values": [9.0, 9.0]})
    return {
        "cycle": 18,
        "accepted_draft_tokens": 1,
        "accepted_token_ids": [15495],
        "bonus_token_id": 26126,
        "cycle_wall_ms": 215.0,
        "draft_hidden_state_trace": trace,
    }
