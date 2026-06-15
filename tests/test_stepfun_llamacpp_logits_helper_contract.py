from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import (
    SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
    _stable_json_sha256,
)
from scripts.stepfun_llamacpp_logits_helper_contract import (
    build_llamacpp_logits_helper_contract,
    main,
    verify_llamacpp_logits_helper_contract,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_prompt_artifact(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "partial_prompt_smoke",
            "prompt": "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n",
            "prompt_length": 4,
            "input_ids": [0, 128006, 201, 128798],
            "next_token_id": 369,
            "next_token_text": " |",
            "next_token_logit": 19.158626556396484,
        },
    )


def _write_artifact(path: Path, *, kind: str, status: str = "blocked") -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "artifact_kind": kind,
            "status": status,
            "ready": False,
            "missing_evidence": ["llama_cpp_same_prompt_logits_artifact_present"],
        },
    )


def _write_llamacpp_source_tree(root: Path, *, include_debug_tokenize_marker: bool = True) -> None:
    (root / "examples/debug").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    debug_lines = [
        "params.save_logits",
        "llama_get_logits_ith(ctx, tokens.size() - 1)",
    ]
    if include_debug_tokenize_marker:
        debug_lines.insert(0, "common_tokenize(ctx, params.prompt, add_bos)")
    (root / "examples/debug/debug.cpp").write_text("\n".join(debug_lines))
    (root / "common/common.h").write_text("bool   parse_special = false);\n")
    (root / "common/common.cpp").write_text(
        "llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special)\n"
    )
    (root / "common/arg.cpp").write_text(
        '{"-sp", "--special"}\n'
        '{"--parse-special"}\n'
    )


def test_stepfun_llamacpp_logits_helper_contract_records_prompt_and_hooks(tmp_path: Path) -> None:
    root = tmp_path / "llama.cpp"
    prompt = tmp_path / "prompt.json"
    probe = tmp_path / "probe.json"
    preflight = tmp_path / "preflight.json"
    _write_llamacpp_source_tree(root)
    _write_prompt_artifact(prompt)
    _write_artifact(probe, kind="stepfun_llamacpp_logits_probe")
    _write_artifact(preflight, kind="stepfun_llamacpp_logits_preflight")

    report = build_llamacpp_logits_helper_contract(
        llama_cpp_root=root,
        prompt_artifact=prompt,
        probe_artifact=probe,
        preflight_artifact=preflight,
        artifact_date="2030-04-05",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_llamacpp_logits_helper_contract"
    assert report["date"] == "2030-04-05"
    assert report["status"] == "blocked"
    assert report["implementation_ready"] is True
    assert report["source_hooks_ready"] is True
    assert report["source_hook_marker_failures"] == []
    assert report["prompt_contract"] == {
        "prompt": "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "prompt_length": 4,
        "input_id_count": 4,
        "input_ids": [0, 128006, 201, 128798],
        "expected_next_token_id": 369,
        "expected_next_token_text": " |",
        "expected_next_token_logit": 19.158626556396484,
        "generated_first_token_id": 671,
        "generated_first_token_text": "The",
    }
    assert report["helper_contract"]["must_match_retained_input_ids"] is True
    assert report["helper_contract"]["accepted_prompt_paths"] == [
        "tokenize prompt text with parse_special=true and add_bos=false for this retained artifact",
        "or bypass text tokenization by accepting the retained input_ids explicitly",
    ]
    assert report["required_code_shapes"] == [
        "common_tokenize(ctx, params.prompt, add_bos, true)",
        "llama_get_logits_ith(ctx, tokens.size() - 1)",
        "--save-logits / --logits-output-dir path",
        "common_tokenize(..., add_special, true)",
        "llama_tokenize(..., add_special, parse_special)",
        "add a parse-special/token-ids control to the logits helper instead",
        "helper-specific --parse-special or --token-ids input",
    ]
    assert report["missing_evidence"] == [
        "same_prompt_logits_helper_built",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]
    assert report["blocked_reason"] == (
        "same-prompt-capable llama.cpp logits helper has not been built and captured"
    )
    assert report["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This contract only identifies the source hooks for same-prompt logits capture; "
            "generated_text_matches_target remains unresolved."
        ),
    }


def test_stepfun_llamacpp_logits_helper_contract_reports_missing_prompt_or_hook(
    tmp_path: Path,
) -> None:
    root = tmp_path / "llama.cpp"
    _write_llamacpp_source_tree(root, include_debug_tokenize_marker=False)

    report = build_llamacpp_logits_helper_contract(
        llama_cpp_root=root,
        prompt_artifact=tmp_path / "missing-prompt.json",
        probe_artifact=tmp_path / "missing-probe.json",
        preflight_artifact=tmp_path / "missing-preflight.json",
    )

    assert report["status"] == "blocked"
    assert report["implementation_ready"] is False
    assert report["source_hooks_ready"] is False
    assert report["source_hook_marker_failures"] == ["debug_current_tokenization"]
    assert report["prompt_contract"] == {
        "prompt": None,
        "prompt_length": None,
        "input_id_count": 0,
        "input_ids": [],
        "expected_next_token_id": None,
        "expected_next_token_text": None,
        "expected_next_token_logit": None,
        "generated_first_token_id": 671,
        "generated_first_token_text": "The",
    }
    assert "same_prompt_logits_helper_contract_source_hooks_present" in report["missing_evidence"]
    assert "retained_prompt_input_ids_present" in report["missing_evidence"]
    assert report["prompt_artifact"] == {
        "path": str(tmp_path / "missing-prompt.json"),
        "exists": False,
        "artifact_kind": None,
        "status": None,
        "sha256": None,
    }


def test_stepfun_llamacpp_logits_helper_contract_cli_modes(tmp_path: Path) -> None:
    root = tmp_path / "llama.cpp"
    prompt = tmp_path / "prompt.json"
    probe = tmp_path / "probe.json"
    preflight = tmp_path / "preflight.json"
    output = tmp_path / "contract.json"
    _write_llamacpp_source_tree(root)
    _write_prompt_artifact(prompt)
    _write_artifact(probe, kind="stepfun_llamacpp_logits_probe")
    _write_artifact(preflight, kind="stepfun_llamacpp_logits_preflight")
    base_args = [
        "--llama-cpp-root",
        str(root),
        "--prompt-artifact",
        str(prompt),
        "--probe-artifact",
        str(probe),
        "--preflight-artifact",
        str(preflight),
        "--artifact-date",
        "2030-04-06",
    ]

    assert main([*base_args, "--output", str(output), "--pretty"]) == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-04-06"
    assert payload["implementation_ready"] is True
    assert payload["prompt_contract"]["input_ids"] == [0, 128006, 201, 128798]

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "blocked"
    assert main([*base_args, "--implementation-ready-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is True
    assert main([*base_args, "--missing-evidence-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == [
        "same_prompt_logits_helper_built",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]
    assert main([*base_args, "--required-code-shapes-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text())[0] == "common_tokenize(ctx, params.prompt, add_bos, true)"
    assert main([*base_args, "--sha-only", "--output", str(output)]) == 0
    assert isinstance(json.loads(output.read_text()), str)


def test_stepfun_llamacpp_logits_helper_contract_verifies_persisted_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "llama.cpp"
    prompt = tmp_path / "prompt.json"
    probe = tmp_path / "probe.json"
    preflight = tmp_path / "preflight.json"
    artifact = tmp_path / "contract.json"
    output = tmp_path / "verify.json"
    _write_llamacpp_source_tree(root)
    _write_prompt_artifact(prompt)
    _write_artifact(probe, kind="stepfun_llamacpp_logits_probe")
    _write_artifact(preflight, kind="stepfun_llamacpp_logits_preflight")
    base_args = [
        "--llama-cpp-root",
        str(root),
        "--prompt-artifact",
        str(prompt),
        "--probe-artifact",
        str(probe),
        "--preflight-artifact",
        str(preflight),
        "--artifact-date",
        "2030-04-07",
    ]
    current = build_llamacpp_logits_helper_contract(
        llama_cpp_root=root,
        prompt_artifact=prompt,
        probe_artifact=probe,
        preflight_artifact=preflight,
        artifact_date="2030-04-07",
    )
    artifact.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    verification = verify_llamacpp_logits_helper_contract(
        artifact,
        current_report=current,
    )
    assert verification["status"] == "match"
    assert verification["all_match"] is True
    assert verification["verification_failures"] == []
    assert verification["persisted_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["current_artifact_sha256"] == _stable_json_sha256(current)
    assert verification["persisted_status"] == "blocked"
    assert verification["current_status"] == "blocked"
    assert verification["persisted_implementation_ready"] is True
    assert verification["current_implementation_ready"] is True
    assert verification["persisted_source_hooks_ready"] is True
    assert verification["current_source_hooks_ready"] is True
    assert verification["persisted_source_hook_marker_failures"] == []
    assert verification["current_source_hook_marker_failures"] == []
    assert verification["persisted_missing_evidence"] == [
        "same_prompt_logits_helper_built",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]
    assert verification["current_missing_evidence"] == [
        "same_prompt_logits_helper_built",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]
    assert verification["persisted_required_code_shapes"][0] == (
        "common_tokenize(ctx, params.prompt, add_bos, true)"
    )
    assert verification["current_required_code_shapes"][0] == (
        "common_tokenize(ctx, params.prompt, add_bos, true)"
    )
    assert verification["persisted_prompt_input_ids"] == [0, 128006, 201, 128798]
    assert verification["current_prompt_input_ids"] == [0, 128006, 201, 128798]
    assert verification["persisted_expected_next_token_id"] == 369
    assert verification["current_expected_next_token_id"] == 369
    assert verification["persisted_generated_first_token_id"] == 671
    assert verification["current_generated_first_token_id"] == 671
    assert verification["persisted_prompt_sha256"] == _stable_json_sha256(
        json.loads(prompt.read_text())
    )
    assert verification["current_prompt_sha256"] == _stable_json_sha256(
        json.loads(prompt.read_text())
    )
    assert verification["persisted_probe_sha256"] == _stable_json_sha256(
        json.loads(probe.read_text())
    )
    assert verification["current_probe_sha256"] == _stable_json_sha256(
        json.loads(probe.read_text())
    )
    assert verification["persisted_preflight_sha256"] == _stable_json_sha256(
        json.loads(preflight.read_text())
    )
    assert verification["current_preflight_sha256"] == _stable_json_sha256(
        json.loads(preflight.read_text())
    )

    assert (
        main(
            [
                *base_args,
                "--verify-contract",
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
    drifted["implementation_ready"] = False
    artifact.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    mismatch = verify_llamacpp_logits_helper_contract(
        artifact,
        current_report=current,
    )
    assert mismatch["status"] == "mismatch"
    assert mismatch["all_match"] is False
    assert mismatch["verification_failure_count"] == 1
    assert mismatch["verification_failures"][0]["name"] == (
        "llamacpp_logits_helper_contract_drift"
    )
    assert (
        main(
            [
                *base_args,
                "--verify-contract",
                str(artifact),
                "--verification-failures-only",
                "--output",
                str(output),
            ]
        )
        == SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    )
    assert json.loads(output.read_text())[0]["name"] == (
        "llamacpp_logits_helper_contract_drift"
    )
