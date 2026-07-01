from __future__ import annotations

import json
from pathlib import Path

from scripts import mtp_proposal_trace_compare as compare


def test_compare_proposal_traces_reports_first_divergence(tmp_path: Path) -> None:
    hip_path = tmp_path / "hip.json"
    llama_path = tmp_path / "llama.jsonl"
    hip_path.write_text(
        json.dumps(
            {
                "cycles": [
                    {
                        "cycle": 0,
                        "generated_draft_tokens": 2,
                        "accepted_draft_tokens": 2,
                        "visible_output_tokens": 3,
                        "draft_tokens": [10, 11],
                        "output_tokens": [10, 11, 12],
                    },
                    {
                        "cycle": 1,
                        "generated_draft_tokens": 2,
                        "accepted_draft_tokens": 0,
                        "visible_output_tokens": 1,
                        "draft_tokens": [20, 21],
                        "output_tokens": [99],
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    llama_rows = [
        {
            "cycle": 0,
            "generated_draft_tokens": 2,
            "accepted_draft_tokens": 2,
            "visible_output_tokens": 3,
            "draft_token_ids": [10, 11],
            "sampled_token_ids": [10, 11, 12],
            "accepted_token_ids": [10, 11],
            "output_token_ids": [10, 11, 12],
            "bonus_token_id": 12,
            "rejected_draft_token_id": None,
        },
        {
            "cycle": 1,
            "generated_draft_tokens": 2,
            "accepted_draft_tokens": 1,
            "visible_output_tokens": 2,
            "draft_token_ids": [20, 42],
            "sampled_token_ids": [20, 100],
            "accepted_token_ids": [20],
            "output_token_ids": [20, 100],
            "bonus_token_id": 100,
            "rejected_draft_token_id": 42,
        },
    ]
    llama_path.write_text("\n".join(json.dumps(row) for row in llama_rows) + "\n", encoding="utf-8")

    summary = compare.compare_rows(
        compare.load_hipengine_rows(hip_path),
        compare.load_llamacpp_rows(llama_path),
    )

    assert summary["compared_rows"] == 2
    assert summary["exact_draft_rows"] == 1
    assert summary["exact_output_rows"] == 1
    assert summary["accepted_count_match_rows"] == 1
    assert summary["hipengine_totals"]["accepted_per_output"] == 2 / 4
    assert summary["llamacpp_totals"]["accepted_per_output"] == 3 / 5
    assert summary["output_stream"]["common_prefix_tokens"] == 3
    assert summary["output_stream"]["exact_match"] is False
    assert summary["output_stream"]["first_token_divergence"] == {
        "token_index": 3,
        "hipengine_token": 99,
        "llamacpp_token": 20,
    }
    assert summary["first_divergence"]["pair_index"] == 1
    assert summary["first_divergence"]["hipengine"]["rejected_draft_token_id"] == 20
    assert summary["first_divergence"]["llamacpp"]["rejected_draft_token_id"] == 42


def test_compare_proposal_traces_separates_chunking_from_stream_match() -> None:
    hip_rows = [
        compare.ProposalRow(
            source="hipengine",
            index=0,
            cycle=0,
            generated_draft_tokens=2,
            accepted_draft_tokens=2,
            visible_output_tokens=3,
            draft_token_ids=[1, 2],
            accepted_token_ids=[1, 2],
            output_token_ids=[1, 2, 3],
            bonus_token_id=3,
            rejected_draft_token_id=None,
        ),
        compare.ProposalRow(
            source="hipengine",
            index=1,
            cycle=1,
            generated_draft_tokens=1,
            accepted_draft_tokens=0,
            visible_output_tokens=1,
            draft_token_ids=[9],
            accepted_token_ids=[],
            output_token_ids=[4],
            bonus_token_id=4,
            rejected_draft_token_id=9,
        ),
    ]
    llama_rows = [
        compare.ProposalRow(
            source="llamacpp",
            index=0,
            cycle=0,
            generated_draft_tokens=1,
            accepted_draft_tokens=0,
            visible_output_tokens=1,
            draft_token_ids=[8],
            accepted_token_ids=[],
            output_token_ids=[1],
            bonus_token_id=1,
            rejected_draft_token_id=8,
        ),
        compare.ProposalRow(
            source="llamacpp",
            index=1,
            cycle=1,
            generated_draft_tokens=2,
            accepted_draft_tokens=2,
            visible_output_tokens=3,
            draft_token_ids=[2, 3],
            accepted_token_ids=[2, 3],
            output_token_ids=[2, 3, 4],
            bonus_token_id=4,
            rejected_draft_token_id=None,
        ),
    ]

    summary = compare.compare_rows(hip_rows, llama_rows)

    assert summary["exact_output_rows"] == 0
    assert summary["output_stream"]["exact_match"] is True
    assert summary["output_stream"]["common_prefix_tokens"] == 4
    assert summary["output_stream"]["first_token_divergence"] is None


def test_load_llamacpp_rows_from_wrapper_artifact(tmp_path: Path) -> None:
    path = tmp_path / "llama-wrapper.json"
    path.write_text(
        json.dumps(
            {
                "stage_timing_summary": {
                    "measured_excluding_first_task": {
                        "proposal_trace_sample": [
                            {
                                "cycle": 7,
                                "generated_draft_tokens": 1,
                                "accepted_draft_tokens": 0,
                                "visible_output_tokens": 1,
                                "draft_token_ids": [31],
                                "sampled_token_ids": [32],
                                "accepted_token_ids": [],
                                "output_token_ids": [32],
                                "bonus_token_id": 32,
                                "rejected_draft_token_id": 31,
                            }
                        ]
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = compare.load_llamacpp_rows(path)

    assert len(rows) == 1
    assert rows[0].cycle == 7
    assert rows[0].draft_token_ids == [31]
    assert rows[0].output_token_ids == [32]
