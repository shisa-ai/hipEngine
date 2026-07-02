from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "llamacpp_mtp_compare_verifier_tensors.py"


def test_compare_verifier_tensors_writes_compact_artifact(tmp_path: Path) -> None:
    llama_jsonl = tmp_path / "llama.jsonl"
    hip_json = tmp_path / "hip.json"
    output = tmp_path / "compare.json"

    llama_record = {
        "task_id": 9,
        "cycle": 3,
        "draft_token_ids": [11, 567],
        "accepted_draft_tokens": 2,
        "output_token_ids": [11, 567, 668],
        "bonus_token_id": 668,
        "target_sample_trace": [
            {"candidate_scores": []},
            {"candidate_scores": []},
            {
                "sampled_token": 668,
                "candidate_scores": [
                    {"token_id": 668, "rank": 1, "logit": 10.0},
                    {"token_id": 8940, "rank": 2, "logit": 9.75},
                ],
            },
        ],
        "tensor_values": [
            {
                "label": "verify_layer_output_0",
                "token_id": 567,
                "position": 75,
                "values": [1.0, 2.0, 3.0],
            },
            {
                "label": "verify_pre_output_norm",
                "token_id": 567,
                "position": 75,
                "values": [3.0, 2.0, 1.0],
            },
            {
                "label": "llama_stage_h_nextn",
                "token_id": 567,
                "position": 75,
                "values": [7.0, 8.0, 9.0],
            },
            {
                "label": "llama_stage_h_nextn",
                "token_id": 567,
                "position": 75,
                "values": [6.0, 5.0, 4.0],
            },
            {
                "label": "verify_h",
                "token_id": 567,
                "position": 75,
                "values": [6.0, 5.0, 4.0],
            },
            {
                "label": "attn_post_norm_0",
                "token_id": 567,
                "position": 75,
                "values": [1.0, 2.0, 3.0],
            },
            {
                "label": "ffn_moe_logits_0",
                "token_id": 567,
                "position": 75,
                "values": [10.0, 20.0],
            },
        ],
    }
    llama_jsonl.write_text(json.dumps(llama_record) + "\n", encoding="utf-8")

    hip_json.write_text(
        json.dumps(
            {
                "probe": {"cycle": 3, "trace_draft_tokens": [11, 567]},
                "result": {
                    "sampled_tokens": [11, 567, 8940],
                    "accepted_draft_tokens": 2,
                    "rows": [
                        {"row": 0},
                        {"row": 1},
                        {
                            "row": 2,
                            "position": 75,
                            "input_token": 567,
                            "sampled_token": 8940,
                            "candidate_scores": [
                                {"token_id": 668, "rank": 4, "logit": 9.25},
                                {"token_id": 8940, "rank": 1, "logit": 10.25},
                            ],
                            "hidden_seed_values": [6.0, 5.0, 5.0],
                            "pre_output_norm_hidden_values": [3.0, 1.0, 1.0],
                            "layer_output_hidden_values": {"0": [1.0, 2.5, 3.0]},
                        },
                    ],
                    "scored_layer_boundary_captures": [
                        {
                            "layer": 0,
                            "row": 2,
                            "values": {
                                "attn_post_norm": [1.0, 3.0, 3.0],
                                "moe_router_logits": [9.0, 22.0],
                            },
                        }
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--llamacpp-jsonl",
            str(llama_jsonl),
            "--hipengine-json",
            str(hip_json),
            "--cycle",
            "3",
            "--task-id",
            "9",
            "--draft-tokens",
            "11,567",
            "--row",
            "2",
            "--layers",
            "0",
            "--boundary-layers",
            "0",
            "--boundary-source",
            "scored",
            "--boundary-pairs",
            "attn_post_norm=attn_post_norm_{layer},moe_router_logits=ffn_moe_logits_{layer}",
            "--candidate-tokens",
            "8940,668",
            "--output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["schema"] == "llamacpp_mtp_verifier_tensor_compare.v1"
    assert artifact["performance_claim"] is False
    assert artifact["inputs"]["token_id"] == 567
    assert artifact["available_llamacpp_value_labels"] == sorted([
        "llama_stage_h_nextn_1",
        "llama_stage_h_nextn_2",
        "attn_post_norm_0",
        "ffn_moe_logits_0",
        "verify_h",
        "verify_layer_output_0",
        "verify_pre_output_norm",
    ])
    assert artifact["token_margin"]["llamacpp"]["8940_minus_668"] == -0.25
    assert artifact["token_margin"]["hipengine"]["8940_minus_668"] == 1.0
    assert artifact["comparisons"][0]["name"] == "verify_layer_output_0"
    assert artifact["comparisons"][0]["delta"]["mean_abs_diff"] == 1.0 / 6.0
    assert artifact["boundary_comparisons"][0]["name"] == "attn_post_norm"
    assert artifact["boundary_comparisons"][0]["delta"]["mean_abs_diff"] == 1.0 / 3.0
    assert artifact["boundary_comparisons"][1]["name"] == "moe_router_logits"
    assert artifact["boundary_comparisons"][1]["delta"]["mean_abs_diff"] == 1.5
    assert artifact["summary"]["largest_boundary_mean_abs_diff"] == {
        "layer": 0,
        "name": "moe_router_logits",
        "mean_abs_diff": 1.5,
    }
    assert artifact["summary"]["first_layer_mean_abs_diff_ge_1e-3"]["name"] == "verify_layer_output_0"


def test_compare_verifier_tensors_allows_boundary_only(tmp_path: Path) -> None:
    llama_jsonl = tmp_path / "llama.jsonl"
    hip_json = tmp_path / "hip.json"
    output = tmp_path / "compare.json"

    llama_jsonl.write_text(
        json.dumps(
            {
                "task_id": 9,
                "cycle": 18,
                "draft_token_ids": [15495, 539],
                "accepted_draft_tokens": 1,
                "output_token_ids": [15495, 26126],
                "bonus_token_id": 26126,
                "target_sample_trace": [
                    {"candidate_scores": []},
                    {
                        "sampled_token": 26126,
                        "top_k": [
                            {"token_id": 539, "rank": 2, "logit": 8.75},
                            {"token_id": 26126, "rank": 1, "logit": 9.0},
                        ],
                        "candidate_scores": [
                            {"token_id": 539, "rank": 2, "logit": 8.75},
                        ],
                    },
                ],
                "tensor_values": [
                    {
                        "label": "process_h_input",
                        "token_id": 15495,
                        "position": 73,
                        "values": [1.0, 2.0, 3.0],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    hip_json.write_text(
        json.dumps(
            {
                "probe": {"cycle": 12, "trace_draft_tokens": [15495, 539]},
                "result": {
                    "sampled_tokens": [15495, 26126, 1151],
                    "accepted_draft_tokens": 1,
                    "rows": [
                        {"row": 0},
                        {
                            "row": 1,
                            "position": 73,
                            "input_token": 15495,
                            "sampled_token": 26126,
                            "candidate_scores": [
                                {"token_id": 539, "rank": 2, "logit": 8.5},
                                {"token_id": 26126, "rank": 1, "logit": 8.875},
                            ],
                        },
                    ],
                    "scored_layer_boundary_captures": [
                        {
                            "layer": 0,
                            "row": 1,
                            "values": {"hidden_in": [1.0, 2.5, 3.0]},
                        }
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--llamacpp-jsonl",
            str(llama_jsonl),
            "--hipengine-json",
            str(hip_json),
            "--cycle",
            "18",
            "--task-id",
            "9",
            "--draft-tokens",
            "15495,539",
            "--row",
            "1",
            "--boundary-layers",
            "0",
            "--boundary-source",
            "scored",
            "--boundary-pairs",
            "hidden_in=process_h_input",
            "--candidate-tokens",
            "539,26126",
            "--output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["comparisons"] == []
    assert artifact["boundary_comparisons"][0]["name"] == "hidden_in"
    assert artifact["boundary_comparisons"][0]["llamacpp_label"] == "process_h_input"
    assert artifact["boundary_comparisons"][0]["delta"]["mean_abs_diff"] == 1.0 / 6.0
    assert artifact["summary"]["first_layer_mean_abs_diff_ge_1e-3"] is None
