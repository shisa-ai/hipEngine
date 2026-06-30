from __future__ import annotations

import argparse
import json

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


def test_llamacpp_mtp_stage_timing_summary_excludes_first_task(tmp_path) -> None:
    path = tmp_path / "llama-stage.jsonl"
    rows = [
        {
            "task_id": 0,
            "visible_output_tokens": 1,
            "accepted_draft_tokens": 0,
            "generated_draft_tokens": 1,
            "cycle_wall_ms": 10.0,
            "target_verify_layer_passes": 1,
            "target_verify_rows_evaluated": 2,
            "target_verify_discarded_rows": 1,
            "stage_timings_ms": {"draft_initial": 1.0, "target_block_verify_total": 9.0},
        },
        {
            "task_id": 1,
            "visible_output_tokens": 2,
            "accepted_draft_tokens": 1,
            "generated_draft_tokens": 2,
            "cycle_wall_ms": 30.0,
            "target_verify_layer_passes": 1,
            "target_verify_rows_evaluated": 3,
            "target_verify_discarded_rows": 1,
            "stage_timings_ms": {
                "draft_initial": 4.0,
                "mtp_context_replay_append": 20.0,
                "target_block_forward": 2.0,
                "target_block_verify_total": 22.0,
            },
        },
        {
            "task_id": 1,
            "visible_output_tokens": 1,
            "accepted_draft_tokens": 0,
            "generated_draft_tokens": 1,
            "cycle_wall_ms": 15.0,
            "target_verify_layer_passes": 1,
            "target_verify_rows_evaluated": 2,
            "target_verify_discarded_rows": 1,
            "stage_timings_ms": {
                "draft_initial": 2.0,
                "target_block_forward": 10.0,
                "target_block_verify_total": 10.0,
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = bench._summarize_stage_timings(path)
    measured = summary["measured_excluding_first_task"]

    assert summary["rows_total"] == 3
    assert summary["rows_measured"] == 2
    assert summary["warmup_task_id_excluded"] == 0
    assert measured["total_output_tokens"] == 3
    assert measured["accepted_per_output"] == 1 / 3
    assert measured["draft_acceptance"] == 1 / 3
    assert measured["cycle_wall_ms_per_output"] == 15.0
    assert measured["target_verify_layer_passes_per_output"] == 2 / 3
    assert measured["target_verify_rows_per_output"] == 5 / 3
    assert measured["stage_timing_totals_ms"] == {
        "draft_initial": 6.0,
        "mtp_context_replay_append": 20.0,
        "target_block_forward": 12.0,
        "target_block_verify_total": 32.0,
    }
    assert measured["stage_timing_per_output_ms"]["target_block_verify_total"] == 32 / 3


def test_llamacpp_mtp_server_command_passes_extra_args_after_mtp_flags() -> None:
    args = argparse.Namespace(
        server_bin="/tmp/llama-server",
        model="/tmp/model.gguf",
        gpu_layers=99,
        flash_attn="on",
        cache_type_k="f16",
        cache_type_v="f16",
        ctx_size=8192,
        host="127.0.0.1",
        port=8013,
        alias="qwen",
        draft_max=2,
        server_extra_arg=["--reasoning", "off"],
    )

    cmd = bench._server_command(args, "mtp")

    assert cmd[-6:] == [
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "2",
        "--reasoning",
        "off",
    ]
