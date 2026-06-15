from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import (
    SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
    _stable_json_sha256,
)
from scripts.stepfun_oracle_rank_check import (
    build_oracle_rank_check,
    main,
    verify_oracle_rank_check,
)
from test_stepfun_oracle_artifact_check import _oracle_artifact  # type: ignore[import-not-found]
from test_stepfun_oracle_token_mismatch import _write_fake_tokenize  # type: ignore[import-not-found]


def _write_prompt_with_top_tokens(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "prompt_length": 23,
                "next_token_id": 369,
                "next_token_text": " |",
                "next_token_logit": 19.158626556396484,
                "top_tokens": [
                    {
                        "rank": 1,
                        "token_id": 369,
                        "token_text": " |",
                        "logit": 19.158626556396484,
                    },
                    {
                        "rank": 2,
                        "token_id": 5,
                        "token_text": "#",
                        "logit": 18.343582153320312,
                    },
                    {
                        "rank": 3,
                        "token_id": 15251,
                        "token_text": "\ufeff",
                        "logit": 16.907772064208984,
                    },
                    {
                        "rank": 4,
                        "token_id": 223,
                        "token_text": " ",
                        "logit": 16.793684005737305,
                    },
                    {
                        "rank": 5,
                        "token_id": 201,
                        "token_text": "\n",
                        "logit": 15.653017044067383,
                    },
                ],
            },
            sort_keys=True,
        )
    )


def _write_mismatch_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    model = tmp_path / "model.gguf"
    fake_tokenize = tmp_path / "llama-tokenize"
    _write_prompt_with_top_tokens(prompt)
    mismatch = _oracle_artifact()
    mismatch.update(
        {
            "model": str(model),
            "stdout": "The\n\n",
            "generated_text": "The\n\n",
            "text_matches_expected_exact": False,
            "text_matches_expected_stripped": False,
        }
    )
    oracle.write_text(json.dumps(mismatch, sort_keys=True))
    model.write_text("fake model path for oracle rank check")
    _write_fake_tokenize(fake_tokenize)
    return prompt, oracle, model, fake_tokenize


def test_stepfun_oracle_rank_check_reports_generated_token_absent_from_top_tokens(
    tmp_path: Path,
) -> None:
    prompt, oracle, model, fake_tokenize = _write_mismatch_inputs(tmp_path)

    report = build_oracle_rank_check(
        artifact=oracle,
        prompt_artifact=prompt,
        llama_tokenize=fake_tokenize,
        tokenizer_model=model,
        artifact_date="2030-01-11",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_llamacpp_oracle_rank_check"
    assert report["date"] == "2030-01-11"
    assert report["status"] == "failed"
    assert report["ready"] is False
    assert report["oracle_status"] == "executed"
    assert report["oracle_returncode"] == 0
    assert report["expected_next_token_id"] == 369
    assert report["expected_top_record"] == {
        "rank": 1,
        "token_id": 369,
        "token_text": " |",
        "logit": 19.158626556396484,
    }
    assert report["host_top_token_count"] == 5
    assert report["host_top_token_ids"] == [369, 5, 15251, 223, 201]
    assert report["generated_text_token_ids"] == [671, 271]
    assert report["generated_text_stripped_token_ids"] == [671]
    assert report["generated_first_token_id"] == 671
    assert report["generated_first_token_matches_expected_id"] is False
    assert report["generated_token_in_host_top_tokens"] is False
    assert report["generated_token_host_top_record"] is None
    assert report["generated_token_host_rank"] is None
    assert report["missing_evidence"] == ["generated_token_in_host_top_tokens"]
    assert report["conclusion"] == "generated token is absent from host top-token list"
    assert report["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This rank check only compares retained oracle/token artifacts with "
            "the host top-token list; it does not resolve the oracle mismatch."
        ),
    }


def test_stepfun_oracle_rank_check_cli_writes_report(tmp_path: Path) -> None:
    prompt, oracle, model, fake_tokenize = _write_mismatch_inputs(tmp_path)
    output = tmp_path / "rank-check.json"

    rc = main(
        [
            "--artifact",
            str(oracle),
            "--prompt-artifact",
            str(prompt),
            "--llama-tokenize",
            str(fake_tokenize),
            "--tokenizer-model",
            str(model),
            "--artifact-date",
            "2030-01-12",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-01-12"
    assert payload["status"] == "failed"
    assert payload["generated_first_token_id"] == 671
    assert payload["generated_token_in_host_top_tokens"] is False


def test_stepfun_oracle_rank_check_cli_compact_modes(tmp_path: Path) -> None:
    prompt, oracle, model, fake_tokenize = _write_mismatch_inputs(tmp_path)
    output = tmp_path / "compact.json"
    base_args = [
        "--artifact",
        str(oracle),
        "--prompt-artifact",
        str(prompt),
        "--llama-tokenize",
        str(fake_tokenize),
        "--tokenizer-model",
        str(model),
    ]

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "failed"
    assert main([*base_args, "--generated-in-top-list-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is False
    assert main([*base_args, "--generated-rank-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is None


def test_stepfun_oracle_rank_check_verifies_persisted_artifact(tmp_path: Path) -> None:
    prompt, oracle, model, fake_tokenize = _write_mismatch_inputs(tmp_path)
    artifact = tmp_path / "rank-check.json"
    output = tmp_path / "verify.json"
    base_args = [
        "--artifact",
        str(oracle),
        "--prompt-artifact",
        str(prompt),
        "--llama-tokenize",
        str(fake_tokenize),
        "--tokenizer-model",
        str(model),
        "--artifact-date",
        "2030-01-13",
    ]
    current = build_oracle_rank_check(
        artifact=oracle,
        prompt_artifact=prompt,
        llama_tokenize=fake_tokenize,
        tokenizer_model=model,
        artifact_date="2030-01-13",
    )
    artifact.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    verification = verify_oracle_rank_check(artifact, current_report=current)
    assert verification["status"] == "match"
    assert verification["all_match"] is True
    assert verification["verification_failures"] == []
    assert verification["persisted_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["current_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["persisted_status"] == "failed"
    assert verification["current_status"] == "failed"
    assert verification["persisted_generated_token_in_host_top_tokens"] is False
    assert verification["current_generated_token_in_host_top_tokens"] is False
    assert verification["persisted_generated_token_host_rank"] is None
    assert verification["current_generated_token_host_rank"] is None

    assert (
        main(
            [
                *base_args,
                "--verify-rank-check",
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
    drifted["generated_token_in_host_top_tokens"] = True
    artifact.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    mismatch = verify_oracle_rank_check(artifact, current_report=current)
    assert mismatch["status"] == "mismatch"
    assert mismatch["all_match"] is False
    assert mismatch["verification_failure_count"] == 1
    assert mismatch["verification_failures"][0]["name"] == "oracle_rank_check_drift"
    assert (
        main(
            [
                *base_args,
                "--verify-rank-check",
                str(artifact),
                "--verification-failures-only",
                "--output",
                str(output),
            ]
        )
        == SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    )
    assert json.loads(output.read_text())[0]["name"] == "oracle_rank_check_drift"
