from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.llamacpp_mtp_compare_hidden_lifecycle import build_artifact


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _hip_payload(prefix: list[float], row1: list[float], margin: float) -> dict[str, object]:
    return {
        "probe": {"cycle": 12},
        "result": {
            "sampled_tokens": [10, 20, 30],
            "accepted_draft_tokens": 1,
            "prefix_state_fingerprint": {
                "position": 72,
                "current_prev": 653,
                "hidden_seed": {"values": prefix},
            },
            "rows": [
                {
                    "row": 0,
                    "input_token": 653,
                    "position": 72,
                    "hidden_seed_values": [2.0, 2.0],
                    "sampled_token": 10,
                    "candidate_scores": [
                        {"token_id": 1, "logit": 0.0, "rank": 2},
                        {"token_id": 2, "logit": 1.0, "rank": 1},
                    ],
                },
                {
                    "row": 1,
                    "input_token": 15495,
                    "position": 73,
                    "hidden_seed_values": row1,
                    "sampled_token": 20,
                    "candidate_scores": [
                        {"token_id": 1, "logit": margin, "rank": 1 if margin > 0 else 2},
                        {"token_id": 2, "logit": 0.0, "rank": 2 if margin > 0 else 1},
                    ],
                },
                {
                    "row": 2,
                    "input_token": 539,
                    "position": 74,
                    "hidden_seed_values": [4.0, 4.0],
                    "sampled_token": 30,
                    "candidate_scores": [
                        {"token_id": 1, "logit": 0.0, "rank": 2},
                        {"token_id": 2, "logit": 1.0, "rank": 1},
                    ],
                },
            ],
        },
    }


def test_build_artifact_compares_hidden_lifecycle(tmp_path: Path) -> None:
    llama_jsonl = tmp_path / "llama.jsonl"
    llama_record = {
        "task_id": 9,
        "cycle": 18,
        "draft_token_ids": [15495, 539],
        "accepted_draft_tokens": 1,
        "output_token_ids": [15495, 26126],
        "bonus_token_id": 26126,
        "rejected_draft_token_id": 539,
        "draft_hidden_state_trace": [
            {"label": "draft_seed_input", "depth": -1, "row_index": -1, "token_id": 653, "position": 72, "values": [1.0, 1.0]},
            {"label": "process_h_input", "depth": -2, "row_index": 0, "token_id": 653, "position": 72, "values": [1.0, 1.0]},
            {"label": "process_h_input", "depth": -2, "row_index": 1, "token_id": 15495, "position": 73, "values": [2.0, 2.0]},
            {"label": "process_h_input", "depth": -2, "row_index": 2, "token_id": 539, "position": 74, "values": [3.0, 3.0]},
            {"label": "verify_h", "depth": 0, "row_index": 0, "token_id": 653, "position": 72, "values": [2.0, 2.0]},
            {"label": "verify_h", "depth": 1, "row_index": 1, "token_id": 15495, "position": 73, "values": [3.0, 3.0]},
            {"label": "verify_h", "depth": 2, "row_index": 2, "token_id": 539, "position": 74, "values": [4.0, 4.0]},
        ],
        "target_sample_trace": [
            {"row": 0, "top_k": [{"token_id": 1, "logit": 0.0}, {"token_id": 2, "logit": 1.0}], "sampled_token": 10},
            {"row": 1, "top_k": [{"token_id": 2, "logit": 0.0, "rank": 1}, {"token_id": 1, "logit": -0.25, "rank": 2}], "sampled_token": 2},
        ],
    }
    llama_jsonl.write_text(json.dumps(llama_record) + "\n", encoding="utf-8")

    default_json = tmp_path / "default.json"
    prefill_json = tmp_path / "prefill.json"
    _write_json(default_json, _hip_payload(prefix=[1.5, 1.5], row1=[3.1, 3.1], margin=-0.1))
    _write_json(prefill_json, _hip_payload(prefix=[1.0, 1.0], row1=[4.0, 4.0], margin=0.2))

    artifact = build_artifact(
        argparse.Namespace(
            llamacpp_jsonl=llama_jsonl,
            hipengine_json=[f"default={default_json}", f"prefill={prefill_json}"],
            cycle=18,
            task_id=9,
            draft_tokens="15495,539",
            row=1,
            verify_rows=3,
            candidate_tokens="1,2",
        )
    )

    assert artifact["llamacpp_cycle"]["draft_token_ids"] == [15495, 539]
    assert artifact["llamacpp_handoff_checks"]["draft_seed_input_vs_process_h_input_row0"]["mean_abs_diff"] == 0.0
    assert artifact["summary"]["nearest_prefix_seed"]["label"] == "prefill"
    assert artifact["summary"]["row1_verify_h_mean_abs_diff"]["label"] == "default"
    assert artifact["llamacpp_token_margin"]["1_minus_2"] == -0.25
    assert artifact["hipengine_comparisons"][0]["token_margin"]["1_minus_2"] == -0.1
