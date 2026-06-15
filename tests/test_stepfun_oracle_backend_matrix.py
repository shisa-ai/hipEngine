from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_oracle_backend_matrix import build_oracle_backend_matrix, main
from test_stepfun_oracle_artifact_check import (  # type: ignore[import-not-found]
    _oracle_artifact,
    _write_prompt,
)
from test_stepfun_oracle_token_mismatch import (  # type: ignore[import-not-found]
    _write_fake_tokenize,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    prompt = tmp_path / "prompt.json"
    vulkan = tmp_path / "vulkan-oracle.json"
    hip = tmp_path / "hip-oracle.json"
    model = tmp_path / "model.gguf"
    fake_tokenize = tmp_path / "llama-tokenize"
    _write_prompt(prompt)

    vulkan_payload = _oracle_artifact()
    vulkan_payload.update(
        {
            "model": str(model),
            "stdout": "The\n\n",
            "generated_text": "The\n\n",
            "text_matches_expected_exact": False,
            "text_matches_expected_stripped": False,
        }
    )
    vulkan.write_text(json.dumps(vulkan_payload, sort_keys=True))

    hip_payload = _oracle_artifact()
    hip_payload.update(
        {
            "model": str(model),
            "status": "timeout",
            "returncode": None,
            "stdout": "",
            "generated_text": "",
            "text_matches_expected_exact": False,
            "text_matches_expected_stripped": False,
            "oracle_blocker_kind": "llama_cpp_oracle_timeout",
            "oracle_blocker_detail": "timed out before generation",
        }
    )
    hip.write_text(json.dumps(hip_payload, sort_keys=True))
    model.write_text("fake model path for backend matrix diagnostics")
    _write_fake_tokenize(fake_tokenize)
    return prompt, vulkan, hip, model, fake_tokenize


def test_stepfun_oracle_backend_matrix_summarizes_vulkan_and_hip(
    tmp_path: Path,
) -> None:
    prompt, vulkan, hip, model, fake_tokenize = _write_inputs(tmp_path)

    matrix = build_oracle_backend_matrix(
        vulkan_artifact=vulkan,
        hip_artifact=hip,
        prompt_artifact=prompt,
        llama_tokenize=fake_tokenize,
        tokenizer_model=model,
        artifact_date="2030-01-04",
    )

    assert matrix["schema_version"] == 1
    assert matrix["artifact_kind"] == "stepfun_llamacpp_oracle_backend_matrix"
    assert matrix["date"] == "2030-01-04"
    assert matrix["status"] == "blocked"
    assert matrix["canonical_backend"] == "vulkan"
    assert matrix["oracle_parity_ready"] is False
    assert matrix["backend_count"] == 2
    assert matrix["backend_outcomes"] == [
        {"backend": "vulkan", "outcome": "executed_token_mismatch"},
        {"backend": "hip", "outcome": "timeout"},
    ]
    vulkan_record, hip_record = matrix["backends"]
    assert vulkan_record["backend"] == "vulkan"
    assert vulkan_record["role"] == "canonical_executed_oracle"
    assert vulkan_record["status"] == "failed"
    assert vulkan_record["oracle_status"] == "executed"
    assert vulkan_record["oracle_returncode"] == 0
    assert vulkan_record["missing_evidence"] == ["generated_text_matches_target"]
    assert vulkan_record["expected_next_token_text_token_ids"] == [369]
    assert vulkan_record["generated_text_token_ids"] == [671, 271]
    assert vulkan_record["generated_text_stripped_token_ids"] == [671]
    assert vulkan_record["generated_first_token_id"] == 671
    assert vulkan_record["generated_first_token_matches_expected_id"] is False
    assert hip_record["backend"] == "hip"
    assert hip_record["role"] == "comparison_timeout_oracle"
    assert hip_record["status"] == "failed"
    assert hip_record["outcome"] == "timeout"
    assert hip_record["oracle_status"] == "timeout"
    assert hip_record["oracle_returncode"] is None
    assert hip_record["oracle_blocker_kind"] == "llama_cpp_oracle_timeout"
    assert hip_record["generated_text"] == ""
    assert hip_record["missing_evidence"] == [
        "oracle_success_status",
        "oracle_returncode_zero",
        "no_timeout_or_oracle_blocker",
        "generated_text_nonempty",
        "generated_text_matches_target",
    ]
    assert matrix["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This matrix consolidates oracle blocker evidence only; it does not "
            "resolve the generated token mismatch or run KV-backed decode."
        ),
    }


def test_stepfun_oracle_backend_matrix_cli_writes_matrix(tmp_path: Path) -> None:
    prompt, vulkan, hip, model, fake_tokenize = _write_inputs(tmp_path)
    output = tmp_path / "backend-matrix.json"

    rc = main(
        [
            "--vulkan-artifact",
            str(vulkan),
            "--hip-artifact",
            str(hip),
            "--prompt-artifact",
            str(prompt),
            "--llama-tokenize",
            str(fake_tokenize),
            "--tokenizer-model",
            str(model),
            "--artifact-date",
            "2030-01-05",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-01-05"
    assert payload["status"] == "blocked"
    assert payload["backend_outcomes"] == [
        {"backend": "vulkan", "outcome": "executed_token_mismatch"},
        {"backend": "hip", "outcome": "timeout"},
    ]


def test_stepfun_oracle_backend_matrix_cli_backend_outcomes_only(tmp_path: Path) -> None:
    prompt, vulkan, hip, model, fake_tokenize = _write_inputs(tmp_path)
    output = tmp_path / "backend-outcomes.json"

    rc = main(
        [
            "--vulkan-artifact",
            str(vulkan),
            "--hip-artifact",
            str(hip),
            "--prompt-artifact",
            str(prompt),
            "--llama-tokenize",
            str(fake_tokenize),
            "--tokenizer-model",
            str(model),
            "--backend-outcomes-only",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    assert json.loads(output.read_text()) == [
        {"backend": "vulkan", "outcome": "executed_token_mismatch"},
        {"backend": "hip", "outcome": "timeout"},
    ]
