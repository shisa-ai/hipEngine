from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import (
    SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
    _stable_json_sha256,
)
from scripts.stepfun_llamacpp_logits_preflight import (
    build_llamacpp_logits_preflight,
    main,
    verify_llamacpp_logits_preflight,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_fake_llama_cli(path: Path, *, help_text: str) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"HELP = {help_text!r}\n"
        "if '--help' in sys.argv:\n"
        "    print(HELP)\n"
        "else:\n"
        "    print('fake llama-cli')\n"
    )
    path.chmod(0o755)


def _write_inputs(tmp_path: Path, *, help_text: str = "--logit-bias TOKEN_ID(+/-)BIAS") -> tuple[Path, Path, Path, Path, Path]:
    manifest = tmp_path / "next-action.json"
    source_map = tmp_path / "source-map.json"
    margin = tmp_path / "host-margin.json"
    fake_cli = tmp_path / "llama-cli"
    logits_artifact = tmp_path / "llamacpp-logits.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_oracle_next_action_manifest",
            "status": "blocked",
            "investigation_ready": True,
            "target": {
                "canonical_backend": "vulkan",
                "expected_next_token_id": 369,
                "expected_next_token_text": " |",
                "generated_first_token_id": 671,
                "generated_text": "The\n\n",
                "host_top_token_ids": [369, 5, 15251, 223, 201],
                "host_top1_to_top2_margin": 0.8150444030761719,
            },
        },
    )
    _write_json(
        source_map,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_oracle_source_map",
            "status": "mapped",
            "ready": True,
        },
    )
    _write_json(
        margin,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_host_top_logit_margin",
            "status": "passed",
            "top1_matches_expected_id": True,
            "top1_to_top2_margin": 0.8150444030761719,
        },
    )
    _write_fake_llama_cli(fake_cli, help_text=help_text)
    return manifest, source_map, margin, fake_cli, logits_artifact


def test_stepfun_llamacpp_logits_preflight_reports_missing_logits_artifact_and_entrypoint(
    tmp_path: Path,
) -> None:
    manifest, source_map, margin, fake_cli, logits_artifact = _write_inputs(tmp_path)

    report = build_llamacpp_logits_preflight(
        next_action_manifest=manifest,
        source_map_artifact=source_map,
        host_logit_margin_artifact=margin,
        llama_logits_artifact=logits_artifact,
        llama_cli=fake_cli,
        artifact_date="2030-01-29",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_llamacpp_logits_preflight"
    assert report["date"] == "2030-01-29"
    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["oracle_parity_ready"] is False
    assert report["expected_llama_logits_artifact"] == {
        "path": str(logits_artifact),
        "exists": False,
        "sha256": None,
    }
    assert report["llama_cli_probe"]["exists"] is True
    assert report["llama_cli_probe"]["executable"] is True
    assert report["llama_cli_probe"]["help_status"] == "executed"
    assert report["llama_cli_probe"]["help_mentions_logit_bias"] is True
    assert report["llama_cli_probe"]["obvious_logits_dump_flag_present"] is False
    assert report["llama_cli_probe"]["same_prompt_special_token_flag_present"] is False
    assert report["missing_evidence"] == [
        "llama_cpp_same_prompt_logits_artifact_present",
        "llama_cpp_logits_dump_entrypoint_identified",
        "llama_cpp_same_prompt_special_token_support_present",
    ]
    assert report["target"] == {
        "canonical_backend": "vulkan",
        "readiness_gate": "oracle_parity",
        "unresolved_evidence_gap": "generated_text_matches_target",
        "expected_next_token_id": 369,
        "expected_next_token_text": " |",
        "generated_first_token_id": 671,
        "generated_text": "The\n\n",
        "host_top_token_ids": [369, 5, 15251, 223, 201],
        "host_top1_to_top2_margin": 0.8150444030761719,
    }
    assert report["blocked_reason"] == (
        "same-prompt logits probe binary lacks special-token parsing for the retained prompt"
    )
    assert report["next_action"] == (
        "identify or add a llama.cpp logits dump entrypoint for the retained StepFun prompt, "
        "then retain same-prompt logits before comparing expected token 369 with generated token 671"
    )
    assert report["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This preflight only records availability of same-prompt llama.cpp logits evidence; "
            "generated_text_matches_target remains unresolved."
        ),
    }


def test_stepfun_llamacpp_logits_preflight_can_be_ready_when_artifact_and_entrypoint_exist(
    tmp_path: Path,
) -> None:
    manifest, source_map, margin, fake_cli, logits_artifact = _write_inputs(
        tmp_path, help_text="--save-logits --logits-output-dir PATH --special --logit-bias TOKEN"
    )
    _write_json(
        logits_artifact,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_llamacpp_logits_probe",
            "status": "captured",
            "ready": True,
            "same_prompt_tokens_match": True,
        },
    )

    report = build_llamacpp_logits_preflight(
        next_action_manifest=manifest,
        source_map_artifact=source_map,
        host_logit_margin_artifact=margin,
        llama_logits_artifact=logits_artifact,
        llama_cli=fake_cli,
    )

    assert report["status"] == "ready"
    assert report["ready"] is True
    assert report["expected_llama_logits_artifact"]["exists"] is True
    assert report["llama_cli_probe"]["obvious_logits_dump_flag_present"] is True
    assert report["llama_cli_probe"]["matched_logits_dump_markers"] == [
        "--save-logits",
        "--logits-output-dir",
    ]
    assert report["missing_evidence"] == []


def test_stepfun_llamacpp_logits_preflight_reports_only_missing_artifact_when_debug_entrypoint_exists(
    tmp_path: Path,
) -> None:
    manifest, source_map, margin, fake_cli, logits_artifact = _write_inputs(
        tmp_path, help_text="--save-logits --logits-output-dir PATH --special --logit-bias TOKEN"
    )

    report = build_llamacpp_logits_preflight(
        next_action_manifest=manifest,
        source_map_artifact=source_map,
        host_logit_margin_artifact=margin,
        llama_logits_artifact=logits_artifact,
        llama_cli=fake_cli,
    )

    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["llama_cli_probe"]["obvious_logits_dump_flag_present"] is True
    assert report["missing_evidence"] == ["llama_cpp_same_prompt_logits_artifact_present"]
    assert report["blocked_reason"] == "same-prompt llama.cpp logits artifact is missing"
    assert report["next_action"] == (
        "retain same-prompt logits from llama-debug using --save-logits, then compare expected "
        "token 369 with generated token 671"
    )


def test_stepfun_llamacpp_logits_preflight_reports_missing_special_token_support(
    tmp_path: Path,
) -> None:
    manifest, source_map, margin, fake_cli, logits_artifact = _write_inputs(
        tmp_path, help_text="--save-logits --logits-output-dir PATH --logit-bias TOKEN"
    )

    report = build_llamacpp_logits_preflight(
        next_action_manifest=manifest,
        source_map_artifact=source_map,
        host_logit_margin_artifact=margin,
        llama_logits_artifact=logits_artifact,
        llama_cli=fake_cli,
    )

    assert report["status"] == "blocked"
    assert report["llama_cli_probe"]["obvious_logits_dump_flag_present"] is True
    assert report["llama_cli_probe"]["same_prompt_special_token_flag_present"] is False
    assert report["missing_evidence"] == [
        "llama_cpp_same_prompt_logits_artifact_present",
        "llama_cpp_same_prompt_special_token_support_present",
    ]
    assert report["blocked_reason"] == (
        "same-prompt logits probe binary lacks special-token parsing for the retained prompt"
    )
    assert report["next_action"] == (
        "add or build a llama.cpp logits dump helper that parses special tokens or accepts explicit "
        "token IDs for the retained StepFun prompt, then retain same-prompt logits before comparing "
        "expected token 369 with generated token 671"
    )


def test_stepfun_llamacpp_logits_preflight_reports_failed_prerequisite(tmp_path: Path) -> None:
    manifest, source_map, margin, fake_cli, logits_artifact = _write_inputs(tmp_path)
    manifest_payload = json.loads(manifest.read_text())
    manifest_payload["investigation_ready"] = False
    _write_json(manifest, manifest_payload)

    report = build_llamacpp_logits_preflight(
        next_action_manifest=manifest,
        source_map_artifact=source_map,
        host_logit_margin_artifact=margin,
        llama_logits_artifact=logits_artifact,
        llama_cli=fake_cli,
    )

    assert report["status"] == "blocked"
    assert "next_action_manifest_investigation_ready" in report["missing_evidence"]


def test_stepfun_llamacpp_logits_preflight_cli_writes_report(tmp_path: Path) -> None:
    manifest, source_map, margin, fake_cli, logits_artifact = _write_inputs(tmp_path)
    output = tmp_path / "preflight.json"

    rc = main(
        [
            "--next-action-manifest",
            str(manifest),
            "--source-map-artifact",
            str(source_map),
            "--host-logit-margin-artifact",
            str(margin),
            "--llama-logits-artifact",
            str(logits_artifact),
            "--llama-cli",
            str(fake_cli),
            "--artifact-date",
            "2030-01-30",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-01-30"
    assert payload["status"] == "blocked"
    assert payload["expected_llama_logits_artifact"]["exists"] is False


def test_stepfun_llamacpp_logits_preflight_cli_compact_modes(tmp_path: Path) -> None:
    manifest, source_map, margin, fake_cli, logits_artifact = _write_inputs(tmp_path)
    output = tmp_path / "compact.json"
    base_args = [
        "--next-action-manifest",
        str(manifest),
        "--source-map-artifact",
        str(source_map),
        "--host-logit-margin-artifact",
        str(margin),
        "--llama-logits-artifact",
        str(logits_artifact),
        "--llama-cli",
        str(fake_cli),
    ]

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "blocked"
    assert main([*base_args, "--logits-artifact-present-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is False
    assert main([*base_args, "--missing-evidence-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == [
        "llama_cpp_same_prompt_logits_artifact_present",
        "llama_cpp_logits_dump_entrypoint_identified",
        "llama_cpp_same_prompt_special_token_support_present",
    ]


def test_stepfun_llamacpp_logits_preflight_verifies_persisted_artifact(
    tmp_path: Path,
) -> None:
    manifest, source_map, margin, fake_cli, logits_artifact = _write_inputs(
        tmp_path,
        help_text="--save-logits --logits-output-dir PATH --logit-bias TOKEN",
    )
    artifact = tmp_path / "preflight.json"
    output = tmp_path / "verify.json"
    base_args = [
        "--next-action-manifest",
        str(manifest),
        "--source-map-artifact",
        str(source_map),
        "--host-logit-margin-artifact",
        str(margin),
        "--llama-logits-artifact",
        str(logits_artifact),
        "--llama-cli",
        str(fake_cli),
        "--artifact-date",
        "2030-01-31",
    ]
    current = build_llamacpp_logits_preflight(
        next_action_manifest=manifest,
        source_map_artifact=source_map,
        host_logit_margin_artifact=margin,
        llama_logits_artifact=logits_artifact,
        llama_cli=fake_cli,
        artifact_date="2030-01-31",
    )
    artifact.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    verification = verify_llamacpp_logits_preflight(
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
    assert verification["persisted_ready"] is False
    assert verification["current_ready"] is False
    assert verification["persisted_missing_evidence"] == [
        "llama_cpp_same_prompt_logits_artifact_present",
        "llama_cpp_same_prompt_special_token_support_present",
    ]
    assert verification["current_missing_evidence"] == [
        "llama_cpp_same_prompt_logits_artifact_present",
        "llama_cpp_same_prompt_special_token_support_present",
    ]
    assert verification["persisted_logits_artifact_exists"] is False
    assert verification["current_logits_artifact_exists"] is False
    assert verification["persisted_cli_help_status"] == "executed"
    assert verification["current_cli_help_status"] == "executed"
    assert verification["persisted_cli_logits_dump_flag_present"] is True
    assert verification["current_cli_logits_dump_flag_present"] is True
    assert verification["persisted_cli_special_token_flag_present"] is False
    assert verification["current_cli_special_token_flag_present"] is False

    assert (
        main(
            [
                *base_args,
                "--verify-preflight",
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
    drifted["missing_evidence"] = ["llama_cpp_same_prompt_logits_artifact_present"]
    artifact.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    mismatch = verify_llamacpp_logits_preflight(
        artifact,
        current_report=current,
    )
    assert mismatch["status"] == "mismatch"
    assert mismatch["all_match"] is False
    assert mismatch["verification_failure_count"] == 1
    assert mismatch["verification_failures"][0]["name"] == (
        "llamacpp_logits_preflight_drift"
    )
    assert (
        main(
            [
                *base_args,
                "--verify-preflight",
                str(artifact),
                "--verification-failures-only",
                "--output",
                str(output),
            ]
        )
        == SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    )
    assert json.loads(output.read_text())[0]["name"] == (
        "llamacpp_logits_preflight_drift"
    )
