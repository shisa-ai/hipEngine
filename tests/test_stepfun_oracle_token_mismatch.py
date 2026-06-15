from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import (
    SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
    _stable_json_sha256,
)
from scripts.stepfun_oracle_token_mismatch import (
    build_oracle_token_mismatch_summary,
    main,
    verify_oracle_token_mismatch_summary,
)
from test_stepfun_oracle_artifact_check import (  # type: ignore[import-not-found]
    _oracle_artifact,
    _write_prompt,
)


def _write_fake_tokenize(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        "file_path = Path(sys.argv[sys.argv.index('--file') + 1])\n"
        "text = file_path.read_text()\n"
        "ids = {' |': [369], 'The': [671], 'The\\n\\n': [671, 271]}.get(text, [])\n"
        "print(ids)\n"
        "print(f'Total number of tokens: {len(ids)}')\n"
    )
    path.chmod(0o755)


def _write_mismatch_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    model = tmp_path / "model.gguf"
    fake_tokenize = tmp_path / "llama-tokenize"
    _write_prompt(prompt)
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
    model.write_text("fake model path for oracle token mismatch diagnostics")
    _write_fake_tokenize(fake_tokenize)
    return prompt, oracle, model, fake_tokenize


def test_stepfun_oracle_token_mismatch_summary_pins_generated_token(
    tmp_path: Path,
) -> None:
    prompt, oracle, model, fake_tokenize = _write_mismatch_inputs(tmp_path)

    summary = build_oracle_token_mismatch_summary(
        artifact=oracle,
        prompt_artifact=prompt,
        llama_tokenize=fake_tokenize,
        tokenizer_model=model,
    )

    assert summary["status"] == "failed"
    assert summary["ready"] is False
    assert summary["missing_evidence"] == ["generated_text_matches_target"]
    diagnostic = summary["tokenization_diagnostic"]
    assert diagnostic["status"] == "completed"
    assert diagnostic["expected_next_token_id"] == 369
    assert diagnostic["expected_next_token_text_token_ids"] == [369]
    assert diagnostic["expected_next_token_text_single_token_matches_expected_id"] is True
    assert diagnostic["generated_text_token_ids"] == [671, 271]
    assert diagnostic["generated_text_stripped_token_ids"] == [671]
    assert diagnostic["generated_first_token_id"] == 671
    assert diagnostic["generated_first_token_matches_expected_id"] is False
    assert diagnostic["conclusion"] == (
        "generated_text tokenization does not match expected_next_token_id"
    )


def test_stepfun_oracle_token_mismatch_cli_writes_summary(
    tmp_path: Path,
) -> None:
    prompt, oracle, model, fake_tokenize = _write_mismatch_inputs(tmp_path)
    output = tmp_path / "oracle-token-mismatch.json"

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
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["status"] == "failed"
    assert payload["missing_evidence"] == ["generated_text_matches_target"]
    assert payload["tokenization_diagnostic"]["expected_next_token_text_token_ids"] == [369]
    assert payload["tokenization_diagnostic"]["generated_text_token_ids"] == [671, 271]


def test_stepfun_oracle_token_mismatch_cli_token_ids_only(tmp_path: Path) -> None:
    prompt, oracle, model, fake_tokenize = _write_mismatch_inputs(tmp_path)
    output = tmp_path / "token-ids.json"

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
            "--token-ids-only",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload == {
        "status": "failed",
        "missing_evidence": ["generated_text_matches_target"],
        "expected_next_token_id": 369,
        "expected_next_token_text_token_ids": [369],
        "generated_text_token_ids": [671, 271],
        "generated_text_stripped_token_ids": [671],
        "generated_first_token_id": 671,
        "generated_first_token_matches_expected_id": False,
        "conclusion": "generated_text tokenization does not match expected_next_token_id",
    }


def test_stepfun_oracle_token_mismatch_verifies_persisted_artifact(
    tmp_path: Path,
) -> None:
    prompt, oracle, model, fake_tokenize = _write_mismatch_inputs(tmp_path)
    artifact = tmp_path / "token-mismatch.json"
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
    ]
    current = build_oracle_token_mismatch_summary(
        artifact=oracle,
        prompt_artifact=prompt,
        llama_tokenize=fake_tokenize,
        tokenizer_model=model,
    )
    artifact.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    verification = verify_oracle_token_mismatch_summary(
        artifact,
        current_summary=current,
    )
    assert verification["status"] == "match"
    assert verification["all_match"] is True
    assert verification["verification_failures"] == []
    assert verification["persisted_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["current_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["persisted_status"] == "failed"
    assert verification["current_status"] == "failed"
    assert verification["persisted_expected_next_token_id"] == 369
    assert verification["current_expected_next_token_id"] == 369
    assert verification["persisted_generated_first_token_id"] == 671
    assert verification["current_generated_first_token_id"] == 671
    assert verification["persisted_generated_first_token_matches_expected_id"] is False
    assert verification["current_generated_first_token_matches_expected_id"] is False

    assert (
        main(
            [
                *base_args,
                "--verify-token-mismatch",
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
    drifted["ready"] = True
    artifact.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    mismatch = verify_oracle_token_mismatch_summary(
        artifact,
        current_summary=current,
    )
    assert mismatch["status"] == "mismatch"
    assert mismatch["all_match"] is False
    assert mismatch["verification_failure_count"] == 1
    assert mismatch["verification_failures"][0]["name"] == (
        "oracle_token_mismatch_drift"
    )
    assert (
        main(
            [
                *base_args,
                "--verify-token-mismatch",
                str(artifact),
                "--verification-failures-only",
                "--output",
                str(output),
            ]
        )
        == SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    )
    assert json.loads(output.read_text())[0]["name"] == "oracle_token_mismatch_drift"
