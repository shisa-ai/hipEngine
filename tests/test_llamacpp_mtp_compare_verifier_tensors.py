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
    assert artifact["available_llamacpp_value_labels"] == [
        "llama_stage_h_nextn_1",
        "llama_stage_h_nextn_2",
        "verify_h",
        "verify_layer_output_0",
        "verify_pre_output_norm",
    ]
    assert artifact["token_margin"]["llamacpp"]["8940_minus_668"] == -0.25
    assert artifact["token_margin"]["hipengine"]["8940_minus_668"] == 1.0
    assert artifact["comparisons"][0]["name"] == "verify_layer_output_0"
    assert artifact["comparisons"][0]["delta"]["mean_abs_diff"] == 1.0 / 6.0
    assert artifact["summary"]["first_layer_mean_abs_diff_ge_1e-3"]["name"] == "verify_layer_output_0"
