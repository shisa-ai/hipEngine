from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import (
    SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
    _stable_json_sha256,
)
from scripts.stepfun_top_token_roundtrip import (
    build_top_token_roundtrip,
    main,
    verify_top_token_roundtrip,
)
from test_stepfun_oracle_rank_check import _write_prompt_with_top_tokens  # type: ignore[import-not-found]


def _write_fake_top_tokenize(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        "file_path = Path(sys.argv[sys.argv.index('--file') + 1])\n"
        "text = file_path.read_text()\n"
        "ids = {\n"
        "    ' |': [369],\n"
        "    '#': [5],\n"
        "    '\\ufeff': [15251],\n"
        "    ' ': [223],\n"
        "    '\\n': [201],\n"
        "}.get(text, [])\n"
        "print(ids)\n"
        "print(f'Total number of tokens: {len(ids)}')\n"
    )
    path.chmod(0o755)


def _write_roundtrip_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    prompt = tmp_path / "prompt.json"
    model = tmp_path / "model.gguf"
    fake_tokenize = tmp_path / "llama-tokenize"
    _write_prompt_with_top_tokens(prompt)
    model.write_text("fake model path for top-token round-trip diagnostics")
    _write_fake_top_tokenize(fake_tokenize)
    return prompt, model, fake_tokenize


def test_stepfun_top_token_roundtrip_confirms_all_host_top_tokens(
    tmp_path: Path,
) -> None:
    prompt, model, fake_tokenize = _write_roundtrip_inputs(tmp_path)

    report = build_top_token_roundtrip(
        prompt_artifact=prompt,
        llama_tokenize=fake_tokenize,
        tokenizer_model=model,
        artifact_date="2030-01-13",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_host_top_token_roundtrip"
    assert report["date"] == "2030-01-13"
    assert report["status"] == "passed"
    assert report["ready"] is True
    assert report["mode"] == "llama-tokenize --no-bos"
    assert report["host_top_token_count"] == 5
    assert report["host_top_token_ids"] == [369, 5, 15251, 223, 201]
    assert report["all_top_token_texts_roundtrip"] is True
    assert report["mismatch_count"] == 0
    assert report["mismatches"] == []
    assert report["missing_evidence"] == []
    assert report["conclusion"] == (
        "all host top-token texts round-trip to their host token ids"
    )
    assert report["oracle_blocker_interpretation"] == (
        "top-token text labels are tokenizer-coherent; the retained llama.cpp "
        "oracle mismatch should be investigated as logits/prompt/backend parity, "
        "not host top-token text-label drift"
    )
    assert report["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This artifact only checks host top-token text/token-id round-trips "
            "with llama-tokenize; it does not compare generated text to the target."
        ),
    }
    records = report["records"]
    assert [record["roundtrip_token_ids"] for record in records] == [
        [369],
        [5],
        [15251],
        [223],
        [201],
    ]
    assert all(record["single_token_matches_host_id"] is True for record in records)


def test_stepfun_top_token_roundtrip_reports_mismatch(tmp_path: Path) -> None:
    prompt, model, fake_tokenize = _write_roundtrip_inputs(tmp_path)
    payload = json.loads(prompt.read_text())
    payload["top_tokens"][1]["token_text"] = "wrong"
    prompt.write_text(json.dumps(payload, sort_keys=True))

    report = build_top_token_roundtrip(
        prompt_artifact=prompt,
        llama_tokenize=fake_tokenize,
        tokenizer_model=model,
    )

    assert report["status"] == "failed"
    assert report["ready"] is False
    assert report["all_top_token_texts_roundtrip"] is False
    assert report["mismatch_count"] == 1
    assert report["mismatches"][0]["rank"] == 2
    assert report["mismatches"][0]["token_id"] == 5
    assert report["mismatches"][0]["roundtrip_token_ids"] == []
    assert report["mismatches"][0]["single_token_matches_host_id"] is False
    assert report["missing_evidence"] == [
        "rank_2_host_top_token_text_roundtrips_to_token_id"
    ]


def test_stepfun_top_token_roundtrip_cli_writes_report(tmp_path: Path) -> None:
    prompt, model, fake_tokenize = _write_roundtrip_inputs(tmp_path)
    output = tmp_path / "top-token-roundtrip.json"

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--llama-tokenize",
            str(fake_tokenize),
            "--tokenizer-model",
            str(model),
            "--artifact-date",
            "2030-01-14",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-01-14"
    assert payload["status"] == "passed"
    assert payload["host_top_token_ids"] == [369, 5, 15251, 223, 201]
    assert payload["all_top_token_texts_roundtrip"] is True


def test_stepfun_top_token_roundtrip_cli_compact_modes(tmp_path: Path) -> None:
    prompt, model, fake_tokenize = _write_roundtrip_inputs(tmp_path)
    output = tmp_path / "compact.json"
    base_args = [
        "--prompt-artifact",
        str(prompt),
        "--llama-tokenize",
        str(fake_tokenize),
        "--tokenizer-model",
        str(model),
    ]

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "passed"
    assert main([*base_args, "--all-roundtrip-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is True
    assert main([*base_args, "--mismatches-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == []


def test_stepfun_top_token_roundtrip_verifies_persisted_artifact(
    tmp_path: Path,
) -> None:
    prompt, model, fake_tokenize = _write_roundtrip_inputs(tmp_path)
    artifact = tmp_path / "top-token-roundtrip.json"
    output = tmp_path / "verify.json"
    base_args = [
        "--prompt-artifact",
        str(prompt),
        "--llama-tokenize",
        str(fake_tokenize),
        "--tokenizer-model",
        str(model),
        "--artifact-date",
        "2030-01-14",
    ]
    current = build_top_token_roundtrip(
        prompt_artifact=prompt,
        llama_tokenize=fake_tokenize,
        tokenizer_model=model,
        artifact_date="2030-01-14",
    )
    artifact.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    verification = verify_top_token_roundtrip(artifact, current_report=current)
    assert verification["status"] == "match"
    assert verification["all_match"] is True
    assert verification["verification_failures"] == []
    assert verification["persisted_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["current_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["persisted_status"] == "passed"
    assert verification["current_status"] == "passed"
    assert verification["persisted_all_top_token_texts_roundtrip"] is True
    assert verification["current_all_top_token_texts_roundtrip"] is True
    assert verification["persisted_mismatch_count"] == 0
    assert verification["current_mismatch_count"] == 0

    assert (
        main(
            [
                *base_args,
                "--verify-roundtrip",
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
    drifted["all_top_token_texts_roundtrip"] = False
    artifact.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    mismatch = verify_top_token_roundtrip(artifact, current_report=current)
    assert mismatch["status"] == "mismatch"
    assert mismatch["all_match"] is False
    assert mismatch["verification_failure_count"] == 1
    assert mismatch["verification_failures"][0]["name"] == (
        "top_token_roundtrip_drift"
    )
    assert (
        main(
            [
                *base_args,
                "--verify-roundtrip",
                str(artifact),
                "--verification-failures-only",
                "--output",
                str(output),
            ]
        )
        == SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    )
    assert json.loads(output.read_text())[0]["name"] == "top_token_roundtrip_drift"
