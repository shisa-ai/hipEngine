from __future__ import annotations

import json
from pathlib import Path

from scripts import mtp_economy_reconcile as reconcile


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_reconcile_flags_stage_denominator_mismatch() -> None:
    hip_suite = {
        "mtp_by_budget": {
            "b2": {
                "decode_tok_s_weighted": 71.5,
                "total_output_tokens": 240,
                "total_accepted": 143,
                "total_drafts": 184,
                "accepted_per_output": 143 / 240,
                "draft_acceptance": 143 / 184,
                "cycle_wall_ms_per_output": 14.0,
                "target_verify_rows_per_output": 1.171,
                "target_verify_discarded_rows": 41,
                "cycle_histograms": {"visible_output_tokens": {"1": 19, "2": 13, "3": 65}},
            }
        }
    }
    llama = {
        "summary": {
            "natural": {
                "mtp_weighted_predicted_per_second": 71.9,
                "mtp_accepted_per_output": 136 / 240,
                "mtp_draft_acceptance": 136 / 169,
            }
        },
        "runs": {
            "mtp": {
                "protocols": {
                    "natural": {
                        "rows": [
                            {
                                "id": "p0",
                                "category": "code",
                                "timings": {
                                    "predicted_n": 240,
                                    "predicted_ms": 3338.0,
                                    "predicted_per_second": 71.9,
                                    "draft_n": 169,
                                    "draft_n_accepted": 136,
                                },
                                "accepted_per_output": 136 / 240,
                                "draft_acceptance": 136 / 169,
                            }
                        ]
                    }
                }
            }
        },
        "stage_timing_summary": {
            "measured_excluding_first_task": {
                "cycles": 87,
                "total_output_tokens": 223,
                "total_accepted": 136,
                "total_drafts": 169,
                "accepted_per_output": 136 / 223,
                "draft_acceptance": 136 / 169,
                "cycle_wall_ms_per_output": 14.269,
                "target_verify_rows_per_output": 1.148,
                "target_verify_discarded_rows_per_output": 0.148,
            }
        },
    }

    summary = reconcile.build_reconciliation(
        hip_suite,
        llama,
        hip_category=None,
        budget="b2",
        protocol="natural",
    )

    assert summary["gaps_hip_minus_llama"]["request_accepted_per_output"] > 0
    assert summary["gaps_hip_minus_llama"]["stage_measured_accepted_per_output"] < 0
    assert summary["denominator_readout"]["hipengine_has_full_request_acceptance_deficit"] is False
    assert summary["denominator_readout"]["stage_acceptance_gap_uses_different_denominator"] is True


def test_prompt_reconciliation_uses_hip_child_artifacts(tmp_path: Path) -> None:
    category = {
        "raw_root": str(tmp_path / "raw"),
        "prompts": [{"id": "p0", "category": "code"}],
    }
    _write(
        tmp_path / "raw" / "b2" / "p0.json",
        {
            "metrics": {
                "total_output_tokens": 24,
                "total_accepted": 15,
                "total_drafts": 17,
                "tokens_per_sec": 73.45,
                "accepted_per_output": 0.625,
                "accept_per_draft": 15 / 17,
                "target_verify_rows_per_output": 1.0833,
                "target_verify_discarded_rows": 2,
            },
            "cycles": [
                {"generated_draft_tokens": 2, "accepted_draft_tokens": 2, "visible_output_tokens": 3},
                {"generated_draft_tokens": 2, "accepted_draft_tokens": 0, "visible_output_tokens": 1},
            ],
        },
    )
    llama = {
        "runs": {
            "mtp": {
                "protocols": {
                    "natural": {
                        "rows": [
                            {
                                "id": "p0",
                                "category": "code",
                                "timings": {
                                    "predicted_n": 24,
                                    "predicted_ms": 322.0,
                                    "predicted_per_second": 74.5,
                                    "draft_n": 17,
                                    "draft_n_accepted": 14,
                                },
                                "accepted_per_output": 14 / 24,
                                "draft_acceptance": 14 / 17,
                            }
                        ]
                    }
                }
            }
        }
    }

    rows = reconcile.prompt_reconciliation(category, llama, budget="b2", protocol="natural")

    assert rows[0]["id"] == "p0"
    assert rows[0]["hipengine"]["accepted_histogram"] == {"2": 1, "0": 1}
    assert rows[0]["delta_hip_minus_llama"]["accepted_draft_tokens"] == 1
    assert rows[0]["delta_hip_minus_llama"]["tok_s"] < 0
