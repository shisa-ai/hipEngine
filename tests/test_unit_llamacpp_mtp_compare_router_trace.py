from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_compare_router_trace import build_router_trace_compare_artifact


def test_build_router_trace_compare_artifact_finds_first_topk_mismatch(
    tmp_path: Path,
) -> None:
    hip_path = tmp_path / "hip.json"
    llama_path = tmp_path / "llama.jsonl"
    hip_path.write_text(json.dumps(_hip_artifact()) + "\n")
    llama_path.write_text(json.dumps(_llama_cycle()) + "\n")

    artifact = build_router_trace_compare_artifact(
        hipengine_router_trace_path=hip_path,
        llamacpp_jsonl_path=llama_path,
        llamacpp_cycle=18,
        row=1,
    )

    assert artifact["status"] == "complete"
    assert artifact["matched_topk_layers"] == [0]
    assert artifact["first_topk_mismatch_layer"] == 1
    assert artifact["layers"][1]["hip_only_experts"] == [1]
    assert artifact["layers"][1]["llamacpp_only_experts"] == [2]
    assert artifact["llamacpp"]["duplicate_value_labels"]["ffn_moe_logits_1"][
        "max_abs_vs_first"
    ] == 0.0


def _hip_artifact() -> dict:
    return {
        "model": "/models/fake.gguf",
        "source_trace": "/tmp/source.json",
        "command": "fake hip command",
        "probe": {"cycle": 12},
        "result": {
            "sampled_tokens": [15495, 539],
            "accepted_draft_tokens": 2,
            "router_trace_captures": [
                {
                    "row": 1,
                    "position": 73,
                    "input_token": 15495,
                    "trace_target_token": 539,
                    "layers": [
                        _layer(0, [0.9, 0.8, 0.1], [0, 1]),
                        _layer(1, [0.9, 0.8, 0.1], [0, 1]),
                    ],
                }
            ],
        },
    }


def _layer(layer: int, logits: list[float], experts: list[int]) -> dict:
    return {
        "layer": layer,
        "layer_type": "linear_attention",
        "moe_selected_experts": experts,
        "moe_routing_weights": [0.6, 0.4],
        "moe_shared_gate": [0.25],
        "values": {"moe_router_logits": logits},
    }


def _llama_cycle() -> dict:
    trace = []
    values = {
        "ffn_moe_logits_0": [0.9, 0.8, 0.1],
        "ffn_moe_weights_norm_0": [0.6, 0.4],
        "shared_expert_gate_0": [0.25],
        "ffn_moe_logits_1": [0.9, 0.1, 0.8],
        "ffn_moe_weights_norm_1": [0.6, 0.4],
        "shared_expert_gate_1": [0.25],
    }
    for label, raw in values.items():
        trace.append({"label": label, "row_index": 1, "values": raw})
    trace.append({"label": "ffn_moe_logits_1", "row_index": 1, "values": [0.9, 0.1, 0.8]})
    return {
        "cycle": 18,
        "accepted_draft_tokens": 1,
        "accepted_token_ids": [15495],
        "bonus_token_id": 26126,
        "draft_hidden_state_trace": trace,
    }
