from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.llamacpp_mtp_hidden_lifecycle_ladder import build_artifact


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _comparison_payload(default_margin: float, prefill_margin: float) -> dict[str, object]:
    return {
        "inputs": {"row": 1},
        "llamacpp_cycle": {
            "task_id": 9,
            "cycle": 18,
            "seed_position": 72,
            "draft_token_ids": [15495, 539],
            "accepted_draft_tokens": 1,
            "output_token_ids": [15495, 26126],
        },
        "llamacpp_token_margin": {
            "539_minus_26126": -0.01,
        },
        "hipengine_comparisons": [
            {
                "label": "default",
                "hipengine_cycle": {
                    "sampled_tokens": [15495, 26126],
                    "accepted_draft_tokens": 1,
                },
                "prefix_vs_llama_draft_seed_input": {"mean_abs_diff": 0.2},
                "verify_h_row_comparisons": [
                    {"delta": {"mean_abs_diff": 0.3}},
                    {"delta": {"mean_abs_diff": 0.05}},
                ],
                "token_margin": {
                    "539_minus_26126": default_margin,
                },
            },
            {
                "label": "prefill_gdn",
                "hipengine_cycle": {
                    "sampled_tokens": [15495, 539],
                    "accepted_draft_tokens": 2,
                },
                "prefix_vs_llama_draft_seed_input": {"mean_abs_diff": 0.1},
                "verify_h_row_comparisons": [
                    {"delta": {"mean_abs_diff": 0.2}},
                    {"delta": {"mean_abs_diff": 0.09}},
                ],
                "token_margin": {
                    "539_minus_26126": prefill_margin,
                },
            },
        ],
    }


def test_build_artifact_summarizes_hidden_lifecycle_ladder(tmp_path: Path) -> None:
    cycle1 = tmp_path / "cycle1.json"
    cycle2 = tmp_path / "cycle2.json"
    _write_json(cycle1, _comparison_payload(default_margin=-0.02, prefill_margin=0.2))
    _write_json(cycle2, _comparison_payload(default_margin=0.5, prefill_margin=-0.015))

    artifact = build_artifact(
        argparse.Namespace(
            comparison=[
                f"cycle1={cycle1}",
                f"cycle2={cycle2}",
            ]
        )
    )

    assert artifact["performance_claim"] is False
    assert artifact["summary"]["cycles"] == 2
    assert artifact["summary"]["nearest_prefix_counts"] == {"prefill_gdn": 2}
    assert artifact["summary"]["nearest_decision_row_counts"] == {"default": 2}
    assert artifact["summary"]["nearest_margin_counts"] == {"default": 1, "prefill_gdn": 1}
    assert artifact["rows"][0]["nearest_margin_label"] == "default"
    assert artifact["rows"][1]["nearest_margin_label"] == "prefill_gdn"
