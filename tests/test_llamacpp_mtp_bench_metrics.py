from __future__ import annotations

from scripts import llamacpp_mtp_bench as bench


def test_llamacpp_mtp_natural_summary_reports_accepted_per_output() -> None:
    rows = [
        {
            "timings": {
                "predicted_n": 10,
                "predicted_ms": 100.0,
                "predicted_per_second": 100.0,
                "draft_n": 8,
                "draft_n_accepted": 4,
            }
        },
        {
            "timings": {
                "predicted_n": 5,
                "predicted_ms": 50.0,
                "predicted_per_second": 100.0,
                "draft_n": 2,
                "draft_n_accepted": 1,
            }
        },
    ]

    summary = bench._summarize_rows(rows)

    assert summary["draft_acceptance"] == 0.5
    assert summary["accepted_per_output"] == 5 / 15
    assert summary["denominators"] == {
        "draft_acceptance": "draft_n_accepted / draft_n",
        "accepted_per_output": "draft_n_accepted / predicted_n",
    }


def test_llamacpp_mtp_token_repeat_summary_reports_accepted_per_output() -> None:
    rows = [
        {"tokens_predicted": 10, "predicted_ms": 100.0, "draft_n": 8, "draft_n_accepted": 4},
        {"tokens_predicted": 6, "predicted_ms": 60.0, "draft_n": 4, "draft_n_accepted": 2},
    ]

    summary = bench._summarize_token_repeat(rows)

    assert summary["draft_acceptance"] == 0.5
    assert summary["accepted_per_output"] == 6 / 16
    assert summary["denominators"] == {
        "draft_acceptance": "draft_n_accepted / draft_n",
        "accepted_per_output": "draft_n_accepted / tokens_predicted",
    }


def test_llamacpp_mtp_artifact_summary_and_text_include_accepted_per_output() -> None:
    artifact = {
        "runs": {
            "base": {
                "protocols": {
                    "natural": {
                        "summary": {
                            "predicted_per_second_weighted": 50.0,
                            "accepted_per_output": None,
                        }
                    },
                    "token_repeat": {
                        "summary": {
                            "weighted_predicted_per_second": 40.0,
                            "accepted_per_output": None,
                        }
                    },
                }
            },
            "mtp": {
                "protocols": {
                    "natural": {
                        "summary": {
                            "predicted_per_second_weighted": 75.0,
                            "draft_acceptance": 0.5,
                            "accepted_per_output": 0.25,
                        }
                    },
                    "token_repeat": {
                        "summary": {
                            "weighted_predicted_per_second": 80.0,
                            "draft_acceptance": 0.75,
                            "accepted_per_output": 0.5,
                        }
                    },
                }
            },
        }
    }

    artifact["summary"] = bench._summarize_artifact(artifact)
    text = bench._summary_text(artifact)

    assert artifact["summary"]["natural"]["mtp_accepted_per_output"] == 0.25
    assert artifact["summary"]["token_repeat"]["mtp_accepted_per_output"] == 0.5
    assert "accepted/output=0.250" in text
    assert "accepted/output=0.500" in text


def test_llamacpp_mtp_row_helpers_handle_missing_denominators() -> None:
    assert bench._accepted_per_output({"predicted_n": 0, "draft_n_accepted": 1}) is None
    assert bench._summarize_rows([{"timings": {}}])["accepted_per_output"] is None
    assert bench._summarize_token_repeat([{}])["accepted_per_output"] is None
