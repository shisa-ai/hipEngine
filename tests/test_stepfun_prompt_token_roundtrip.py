from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import (
    SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
    _stable_json_sha256,
)
from scripts.stepfun_prompt_token_roundtrip import (
    build_prompt_token_roundtrip,
    main,
    verify_prompt_token_roundtrip,
)

PROMPT_TEXT = (
    "<｜begin▁of▁sentence｜><|im_start|>system\n"
    "Reasoning: low\n\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "hello<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n"
)
PROMPT_INPUT_IDS = [
    0,
    128006,
    27824,
    201,
    77666,
    288,
    28,
    3157,
    271,
    128007,
    201,
    128006,
    5265,
    201,
    33310,
    128007,
    201,
    128006,
    624,
    15059,
    201,
    128798,
    201,
]


def _write_prompt(path: Path, *, input_ids: list[int] | None = None) -> None:
    ids = PROMPT_INPUT_IDS if input_ids is None else input_ids
    path.write_text(
        json.dumps(
            {
                "prompt": PROMPT_TEXT,
                "prompt_length": len(ids),
                "input_ids": ids,
                "next_token_id": 369,
                "next_token_text": " |",
                "top_tokens": [
                    {"rank": 1, "token_id": 369, "token_text": " |", "logit": 19.0}
                ],
            },
            sort_keys=True,
        )
    )


def _write_fake_prompt_tokenize(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n"
        f"PROMPT_TEXT = {PROMPT_TEXT!r}\n"
        f"PROMPT_INPUT_IDS = {PROMPT_INPUT_IDS!r}\n"
        "file_path = Path(sys.argv[sys.argv.index('--file') + 1])\n"
        "text = file_path.read_text()\n"
        "ids = PROMPT_INPUT_IDS if text == PROMPT_TEXT else []\n"
        "print(json.dumps(ids))\n"
        "print(f'Total number of tokens: {len(ids)}')\n"
    )
    path.chmod(0o755)


def _write_roundtrip_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    prompt = tmp_path / "prompt.json"
    model = tmp_path / "model.gguf"
    fake_tokenize = tmp_path / "llama-tokenize"
    _write_prompt(prompt)
    model.write_text("fake model path for prompt token round-trip diagnostics")
    _write_fake_prompt_tokenize(fake_tokenize)
    return prompt, model, fake_tokenize


def test_stepfun_prompt_token_roundtrip_confirms_prompt_input_ids(tmp_path: Path) -> None:
    prompt, model, fake_tokenize = _write_roundtrip_inputs(tmp_path)

    report = build_prompt_token_roundtrip(
        prompt_artifact=prompt,
        llama_tokenize=fake_tokenize,
        tokenizer_model=model,
        artifact_date="2030-01-15",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_prompt_token_roundtrip"
    assert report["date"] == "2030-01-15"
    assert report["status"] == "passed"
    assert report["ready"] is True
    assert report["mode"] == "llama-tokenize --no-bos"
    assert report["prompt_text_len"] == len(PROMPT_TEXT)
    assert report["host_prompt_length"] == len(PROMPT_INPUT_IDS)
    assert report["host_input_id_count"] == len(PROMPT_INPUT_IDS)
    assert report["host_input_ids"] == PROMPT_INPUT_IDS
    assert report["llama_token_count"] == len(PROMPT_INPUT_IDS)
    assert report["llama_token_ids"] == PROMPT_INPUT_IDS
    assert report["prompt_tokenization_matches_host_input_ids"] is True
    assert report["first_mismatch"] is None
    assert report["missing_evidence"] == []
    assert report["conclusion"] == (
        "host prompt input_ids match llama.cpp no-BOS prompt tokenization"
    )
    assert report["oracle_blocker_interpretation"] == (
        "prompt tokenization is tokenizer-coherent; the retained llama.cpp "
        "oracle mismatch should be investigated as logits/backend parity, not "
        "prompt-token drift"
    )
    assert report["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This artifact only checks host prompt input_ids against llama-tokenize; "
            "it does not compare generated text to the target."
        ),
    }


def test_stepfun_prompt_token_roundtrip_reports_first_mismatch(tmp_path: Path) -> None:
    prompt, model, fake_tokenize = _write_roundtrip_inputs(tmp_path)
    mutated_ids = list(PROMPT_INPUT_IDS)
    mutated_ids[4] = 12345
    _write_prompt(prompt, input_ids=mutated_ids)

    report = build_prompt_token_roundtrip(
        prompt_artifact=prompt,
        llama_tokenize=fake_tokenize,
        tokenizer_model=model,
    )

    assert report["status"] == "failed"
    assert report["ready"] is False
    assert report["prompt_tokenization_matches_host_input_ids"] is False
    assert report["first_mismatch"] == {
        "index": 4,
        "host_token_id": 12345,
        "llama_token_id": 77666,
        "kind": "token_id_mismatch",
    }
    assert report["missing_evidence"] == [
        "host_prompt_input_ids_match_llama_tokenize"
    ]


def test_stepfun_prompt_token_roundtrip_cli_writes_report(tmp_path: Path) -> None:
    prompt, model, fake_tokenize = _write_roundtrip_inputs(tmp_path)
    output = tmp_path / "prompt-roundtrip.json"

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--llama-tokenize",
            str(fake_tokenize),
            "--tokenizer-model",
            str(model),
            "--artifact-date",
            "2030-01-16",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-01-16"
    assert payload["status"] == "passed"
    assert payload["host_input_ids"] == PROMPT_INPUT_IDS
    assert payload["llama_token_ids"] == PROMPT_INPUT_IDS
    assert payload["prompt_tokenization_matches_host_input_ids"] is True


def test_stepfun_prompt_token_roundtrip_cli_compact_modes(tmp_path: Path) -> None:
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
    assert main([*base_args, "--roundtrip-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is True
    assert main([*base_args, "--first-mismatch-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is None


def test_stepfun_prompt_token_roundtrip_verifies_persisted_artifact(
    tmp_path: Path,
) -> None:
    prompt, model, fake_tokenize = _write_roundtrip_inputs(tmp_path)
    artifact = tmp_path / "prompt-token-roundtrip.json"
    output = tmp_path / "verify.json"
    base_args = [
        "--prompt-artifact",
        str(prompt),
        "--llama-tokenize",
        str(fake_tokenize),
        "--tokenizer-model",
        str(model),
        "--artifact-date",
        "2030-01-17",
    ]
    current = build_prompt_token_roundtrip(
        prompt_artifact=prompt,
        llama_tokenize=fake_tokenize,
        tokenizer_model=model,
        artifact_date="2030-01-17",
    )
    artifact.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    verification = verify_prompt_token_roundtrip(artifact, current_report=current)
    assert verification["status"] == "match"
    assert verification["all_match"] is True
    assert verification["verification_failures"] == []
    assert verification["persisted_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["current_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["persisted_status"] == "passed"
    assert verification["current_status"] == "passed"
    assert verification["persisted_prompt_tokenization_matches_host_input_ids"] is True
    assert verification["current_prompt_tokenization_matches_host_input_ids"] is True
    assert verification["persisted_first_mismatch"] is None
    assert verification["current_first_mismatch"] is None

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
    drifted["prompt_tokenization_matches_host_input_ids"] = False
    artifact.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    mismatch = verify_prompt_token_roundtrip(artifact, current_report=current)
    assert mismatch["status"] == "mismatch"
    assert mismatch["all_match"] is False
    assert mismatch["verification_failure_count"] == 1
    assert mismatch["verification_failures"][0]["name"] == (
        "prompt_token_roundtrip_drift"
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
    assert json.loads(output.read_text())[0]["name"] == (
        "prompt_token_roundtrip_drift"
    )
