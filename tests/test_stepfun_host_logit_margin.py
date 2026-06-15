from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import (
    SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
    _stable_json_sha256,
)
from scripts.stepfun_host_logit_margin import (
    build_host_logit_margin,
    main,
    verify_host_logit_margin,
)
from test_stepfun_oracle_rank_check import _write_prompt_with_top_tokens  # type: ignore[import-not-found]


def _write_rank_check(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": "stepfun_llamacpp_oracle_rank_check",
                "status": "failed",
                "generated_first_token_id": 671,
                "generated_token_in_host_top_tokens": False,
                "generated_token_host_rank": None,
            },
            sort_keys=True,
        )
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    prompt = tmp_path / "prompt.json"
    rank_check = tmp_path / "rank-check.json"
    _write_prompt_with_top_tokens(prompt)
    _write_rank_check(rank_check)
    return prompt, rank_check


def test_stepfun_host_logit_margin_records_clear_expected_top1(tmp_path: Path) -> None:
    prompt, rank_check = _write_inputs(tmp_path)

    report = build_host_logit_margin(
        prompt_artifact=prompt,
        rank_check_artifact=rank_check,
        artifact_date="2030-01-22",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_host_top_logit_margin"
    assert report["date"] == "2030-01-22"
    assert report["status"] == "passed"
    assert report["ready"] is True
    assert report["expected_next_token_id"] == 369
    assert report["expected_next_token_text"] == " |"
    assert report["expected_next_token_logit"] == 19.158626556396484
    assert report["host_top_token_count"] == 5
    assert report["host_top_token_ids"] == [369, 5, 15251, 223, 201]
    assert report["host_top_logits"] == [
        19.158626556396484,
        18.343582153320312,
        16.907772064208984,
        16.793684005737305,
        15.653017044067383,
    ]
    assert report["top1_record"] == {
        "rank": 1,
        "token_id": 369,
        "token_text": " |",
        "logit": 19.158626556396484,
    }
    assert report["top2_record"]["token_id"] == 5
    assert report["top5_record"]["token_id"] == 201
    assert report["top1_matches_expected_id"] is True
    assert report["top_tokens_sorted_by_logit_desc"] is True
    assert report["top1_to_top2_margin"] == 0.8150444030761719
    assert report["top1_to_top5_margin"] == 3.5056095123291016
    assert report["adjacent_visible_margins"][0] == {
        "higher_rank": 1,
        "lower_rank": 2,
        "margin": 0.8150444030761719,
    }
    assert report["generated_token_context"] == {
        "rank_check_artifact": str(rank_check),
        "rank_check_present": True,
        "generated_first_token_id": 671,
        "generated_token_in_host_top_tokens": False,
        "generated_token_host_rank": None,
    }
    assert report["missing_evidence"] == []
    assert report["conclusion"] == (
        "expected token is retained host top-1 with a positive visible logit margin"
    )
    assert report["oracle_blocker_interpretation"] == (
        "the retained host logits make the expected token a clear visible top-1; "
        "the llama.cpp generated-token mismatch should be investigated as "
        "logits/backend parity rather than a near-tie in the retained host top list"
    )
    assert report["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This artifact only summarizes retained host top-token logits; "
            "generated_text_matches_target remains unresolved."
        ),
    }


def test_stepfun_host_logit_margin_reports_missing_expected_top1(tmp_path: Path) -> None:
    prompt, rank_check = _write_inputs(tmp_path)
    payload = json.loads(prompt.read_text())
    payload["top_tokens"][0]["token_id"] = 5
    payload["top_tokens"][1]["token_id"] = 369
    prompt.write_text(json.dumps(payload, sort_keys=True))

    report = build_host_logit_margin(
        prompt_artifact=prompt,
        rank_check_artifact=rank_check,
    )

    assert report["status"] == "failed"
    assert report["ready"] is False
    assert report["top1_matches_expected_id"] is False
    assert report["top_tokens_sorted_by_logit_desc"] is True
    assert report["top1_to_top2_margin"] == 0.8150444030761719
    assert report["missing_evidence"] == ["expected_token_is_host_top1"]


def test_stepfun_host_logit_margin_reports_unsorted_logits(tmp_path: Path) -> None:
    prompt, rank_check = _write_inputs(tmp_path)
    payload = json.loads(prompt.read_text())
    payload["top_tokens"][1]["logit"] = 20.0
    prompt.write_text(json.dumps(payload, sort_keys=True))

    report = build_host_logit_margin(
        prompt_artifact=prompt,
        rank_check_artifact=rank_check,
    )

    assert report["status"] == "failed"
    assert report["ready"] is False
    assert report["top_tokens_sorted_by_logit_desc"] is False
    assert report["top1_to_top2_margin"] < 0.0
    assert report["missing_evidence"] == [
        "host_top_tokens_sorted_by_logit_desc",
        "host_top1_to_top2_margin_positive",
    ]


def test_stepfun_host_logit_margin_cli_writes_report(tmp_path: Path) -> None:
    prompt, rank_check = _write_inputs(tmp_path)
    output = tmp_path / "logit-margin.json"

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--rank-check-artifact",
            str(rank_check),
            "--artifact-date",
            "2030-01-23",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-01-23"
    assert payload["status"] == "passed"
    assert payload["top1_matches_expected_id"] is True
    assert payload["top1_to_top2_margin"] == 0.8150444030761719


def test_stepfun_host_logit_margin_cli_compact_modes(tmp_path: Path) -> None:
    prompt, rank_check = _write_inputs(tmp_path)
    output = tmp_path / "compact.json"
    base_args = [
        "--prompt-artifact",
        str(prompt),
        "--rank-check-artifact",
        str(rank_check),
    ]

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "passed"
    assert main([*base_args, "--top1-margin-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == 0.8150444030761719
    assert main([*base_args, "--expected-top1-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is True


def test_stepfun_host_logit_margin_verifies_persisted_artifact(tmp_path: Path) -> None:
    prompt, rank_check = _write_inputs(tmp_path)
    artifact = tmp_path / "host-margin.json"
    output = tmp_path / "verify.json"
    base_args = [
        "--prompt-artifact",
        str(prompt),
        "--rank-check-artifact",
        str(rank_check),
        "--artifact-date",
        "2030-01-24",
    ]
    current = build_host_logit_margin(
        prompt_artifact=prompt,
        rank_check_artifact=rank_check,
        artifact_date="2030-01-24",
    )
    artifact.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    verification = verify_host_logit_margin(artifact, current_report=current)
    assert verification["status"] == "match"
    assert verification["all_match"] is True
    assert verification["verification_failures"] == []
    assert verification["persisted_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["current_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["persisted_status"] == "passed"
    assert verification["current_status"] == "passed"
    assert verification["persisted_ready"] is True
    assert verification["current_ready"] is True
    assert verification["persisted_top1_to_top2_margin"] == 0.8150444030761719
    assert verification["current_top1_to_top2_margin"] == 0.8150444030761719

    assert (
        main(
            [
                *base_args,
                "--verify-margin",
                str(artifact),
                "--verification-status-only",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text()) == "match"

    drifted = dict(current)
    drifted["ready"] = False
    artifact.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    mismatch = verify_host_logit_margin(artifact, current_report=current)
    assert mismatch["status"] == "mismatch"
    assert mismatch["all_match"] is False
    assert mismatch["verification_failure_count"] == 1
    assert mismatch["verification_failures"][0]["name"] == "host_logit_margin_drift"
    assert (
        main(
            [
                *base_args,
                "--verify-margin",
                str(artifact),
                "--verification-failures-only",
                "--output",
                str(output),
            ]
        )
        == SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    )
    assert json.loads(output.read_text())[0]["name"] == "host_logit_margin_drift"
