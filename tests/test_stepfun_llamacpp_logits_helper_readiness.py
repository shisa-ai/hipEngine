from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from pathlib import Path

from scripts.stepfun_correctness_status import SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
from scripts.stepfun_llamacpp_logits_helper_readiness import (
    PATCH_APPLIED_MARKERS,
    build_llamacpp_logits_helper_readiness,
    main,
    verify_llamacpp_logits_helper_readiness,
)


def _stable_json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def _init_git(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _write_source(root: Path, *, patched: bool) -> None:
    path = root / "examples/debug/debug.cpp"
    path.parent.mkdir(parents=True, exist_ok=True)
    if patched:
        path.write_text("\n".join(str(marker["marker"]) for marker in PATCH_APPLIED_MARKERS))
    else:
        path.write_text("static bool run(llama_context * ctx, const common_params & params) {}\n")


def _write_dry_run_artifact(path: Path, *, patch: Path) -> None:
    _write_json(
        path,
        {
            "status": "blocked",
            "patch_ready": True,
            "git_apply_check": {"status": "passed"},
            "patch_artifact": {
                "path": str(patch),
                "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                "apply_command": f"git apply --unidiff-zero {patch}",
            },
        },
    )


def _write_fake_llama_debug(path: Path, *, help_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if '--help' in sys.argv:\n"
        f"    print({help_text!r})\n"
        "elif '--version' in sys.argv:\n"
        "    print('llama-debug fake-version')\n"
        "else:\n"
        "    print('unexpected invocation')\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_stepfun_llamacpp_logits_helper_readiness_reports_current_blockers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "llama.cpp"
    patch = tmp_path / "helper.patch"
    dry_run = tmp_path / "dry-run.json"
    helper = tmp_path / "build/bin/llama-debug"
    logits = tmp_path / "logits.json"
    _write_source(root, patched=False)
    _init_git(root)
    patch.write_text("diff --git a/examples/debug/debug.cpp b/examples/debug/debug.cpp\n")
    _write_dry_run_artifact(dry_run, patch=patch)
    _write_fake_llama_debug(helper, help_text="--save-logits --logits-output-dir --special")
    _write_json(
        logits,
        {
            "status": "blocked",
            "ready": False,
            "same_prompt_tokens_match": None,
            "prompt_token_source": "retained-input-ids",
        },
    )

    report = build_llamacpp_logits_helper_readiness(
        llama_cpp_root=root,
        llama_debug=helper,
        patch_artifact=patch,
        patch_dry_run_artifact=dry_run,
        llama_logits_artifact=logits,
        artifact_date="2030-04-11",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_llamacpp_logits_helper_readiness"
    assert report["date"] == "2030-04-11"
    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["git_worktree"]["is_git_worktree"] is True
    assert report["git_worktree"]["clean_for_patch_apply"] is True
    assert report["patch_artifact"]["exists"] is True
    assert report["patch_dry_run_artifact"]["exists"] is True
    assert report["patch_dry_run_artifact"]["patch_ready"] is True
    assert report["patch_dry_run_artifact"]["git_apply_check_status"] == "passed"
    assert report["patch_dry_run_artifact"]["patch_artifact_sha256_matches"] is True
    assert report["source_patch"]["patch_applied"] is False
    assert report["source_patch"]["missing_markers"] == [
        marker["key"] for marker in PATCH_APPLIED_MARKERS
    ]
    assert report["llama_debug"]["exists"] is True
    assert report["llama_debug"]["executable"] is True
    assert report["llama_debug"]["retained_token_ids_capable"] is False
    assert report["llama_logits_artifact"] == {
        "path": str(logits),
        "exists": True,
        "sha256": report["llama_logits_artifact"]["sha256"],
        "status": "blocked",
        "ready": False,
        "same_prompt_tokens_match": None,
        "prompt_token_source": "retained-input-ids",
    }
    assert report["missing_evidence"] == [
        "llama_cpp_token_ids_helper_patch_applied",
        "llama_debug_retained_token_ids_input_present",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]
    commands = report["required_next_commands"]
    assert commands == [
        f"git -C {root} apply --unidiff-zero {patch}",
        "cmake --build /home/lhl/llama.cpp/llama.cpp-vulkan/build-vulkan-release --target llama-debug -j",
        "python3 scripts/stepfun_llamacpp_logits_probe.py --prompt-token-source retained-input-ids --execute --default-output --pretty",
    ]
    command_records = report["required_next_command_records"]
    assert [record["kind"] for record in command_records] == [
        "apply_retained_token_helper_patch",
        "build_llama_debug_helper",
        "capture_same_prompt_logits",
    ]
    assert [record["command"] for record in command_records] == commands
    assert [record["side_effect_scope"] for record in command_records] == [
        "external_llama_cpp_worktree",
        "external_llama_cpp_build_dir",
        "in_tree_benchmark_artifact",
    ]
    for record in command_records:
        payload = {key: value for key, value in record.items() if key != "sha256"}
        assert record["sha256"] == hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    assert report["required_next_command_records_sha256"] == hashlib.sha256(
        json.dumps(command_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert report["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": "Readiness metadata is not same-prompt logits parity evidence.",
    }


def test_stepfun_llamacpp_logits_helper_readiness_ready_when_all_evidence_present(
    tmp_path: Path,
) -> None:
    root = tmp_path / "llama.cpp"
    patch = tmp_path / "helper.patch"
    dry_run = tmp_path / "dry-run.json"
    helper = tmp_path / "build/bin/llama-debug"
    logits = tmp_path / "logits.json"
    _write_source(root, patched=True)
    _init_git(root)
    patch.write_text("diff --git a/examples/debug/debug.cpp b/examples/debug/debug.cpp\n")
    _write_dry_run_artifact(dry_run, patch=patch)
    _write_fake_llama_debug(
        helper,
        help_text="--save-logits --logits-output-dir --special --parse-special --token-ids",
    )
    _write_json(
        logits,
        {
            "status": "captured",
            "ready": True,
            "same_prompt_tokens_match": True,
            "prompt_token_source": "retained-input-ids",
        },
    )

    report = build_llamacpp_logits_helper_readiness(
        llama_cpp_root=root,
        llama_debug=helper,
        patch_artifact=patch,
        patch_dry_run_artifact=dry_run,
        llama_logits_artifact=logits,
    )

    assert report["status"] == "ready"
    assert report["ready"] is True
    assert report["source_patch"]["patch_applied"] is True
    assert report["source_patch"]["missing_markers"] == []
    assert report["patch_dry_run_artifact"]["patch_artifact_sha256_matches"] is True
    assert report["llama_debug"]["retained_token_ids_capable"] is True
    assert report["llama_debug"]["parse_special_capable"] is True
    assert report["llama_logits_artifact"]["status"] == "captured"
    assert report["missing_evidence"] == []
    assert report["blocked_reason"] is None


def test_stepfun_llamacpp_logits_helper_readiness_reports_dry_run_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "llama.cpp"
    patch = tmp_path / "helper.patch"
    dry_run = tmp_path / "dry-run.json"
    helper = tmp_path / "build/bin/llama-debug"
    logits = tmp_path / "logits.json"
    _write_source(root, patched=False)
    _init_git(root)
    patch.write_text("patch")
    _write_json(
        dry_run,
        {
            "status": "blocked",
            "patch_ready": True,
            "git_apply_check": {"status": "passed"},
            "patch_artifact": {"sha256": "not-the-patch-sha"},
        },
    )
    _write_fake_llama_debug(helper, help_text="--save-logits --logits-output-dir --special")
    _write_json(logits, {"status": "blocked", "ready": False})

    report = build_llamacpp_logits_helper_readiness(
        llama_cpp_root=root,
        llama_debug=helper,
        patch_artifact=patch,
        patch_dry_run_artifact=dry_run,
        llama_logits_artifact=logits,
    )

    assert report["patch_dry_run_artifact"]["patch_artifact_sha256_matches"] is False
    assert "llama_cpp_token_ids_helper_patch_artifact_matches_dry_run" in report[
        "missing_evidence"
    ]


def test_stepfun_llamacpp_logits_helper_readiness_cli_modes(tmp_path: Path) -> None:
    root = tmp_path / "llama.cpp"
    patch = tmp_path / "helper.patch"
    dry_run = tmp_path / "dry-run.json"
    helper = tmp_path / "build/bin/llama-debug"
    logits = tmp_path / "logits.json"
    output = tmp_path / "readiness.json"
    _write_source(root, patched=False)
    _init_git(root)
    patch.write_text("patch")
    _write_dry_run_artifact(dry_run, patch=patch)
    _write_fake_llama_debug(helper, help_text="--save-logits --logits-output-dir --special")
    _write_json(logits, {"status": "blocked", "ready": False})
    base_args = [
        "--llama-cpp-root",
        str(root),
        "--llama-debug",
        str(helper),
        "--patch-artifact",
        str(patch),
        "--patch-dry-run-artifact",
        str(dry_run),
        "--llama-logits-artifact",
        str(logits),
        "--artifact-date",
        "2030-04-12",
    ]

    assert main([*base_args, "--output", str(output), "--pretty"]) == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-04-12"
    assert payload["status"] == "blocked"
    assert payload["patch_dry_run_artifact"]["patch_artifact_sha256_matches"] is True

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "blocked"
    assert main([*base_args, "--patch-applied-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is False
    assert main([*base_args, "--helper-capable-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is False
    assert main([*base_args, "--ready-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is False
    assert main([*base_args, "--missing-evidence-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == [
        "llama_cpp_token_ids_helper_patch_applied",
        "llama_debug_retained_token_ids_input_present",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]
    assert main([*base_args, "--sha-only", "--output", str(output)]) == 0
    assert isinstance(json.loads(output.read_text()), str)


def test_stepfun_llamacpp_logits_helper_readiness_verifies_persisted_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "llama.cpp"
    patch = tmp_path / "helper.patch"
    dry_run = tmp_path / "dry-run.json"
    helper = tmp_path / "build/bin/llama-debug"
    logits = tmp_path / "logits.json"
    artifact = tmp_path / "readiness.json"
    output = tmp_path / "verify.json"
    _write_source(root, patched=False)
    _init_git(root)
    patch.write_text("patch")
    _write_dry_run_artifact(dry_run, patch=patch)
    _write_fake_llama_debug(helper, help_text="--save-logits --logits-output-dir --special")
    _write_json(logits, {"status": "blocked", "ready": False})
    base_args = [
        "--llama-cpp-root",
        str(root),
        "--llama-debug",
        str(helper),
        "--patch-artifact",
        str(patch),
        "--patch-dry-run-artifact",
        str(dry_run),
        "--llama-logits-artifact",
        str(logits),
        "--artifact-date",
        "2030-04-13",
    ]
    current = build_llamacpp_logits_helper_readiness(
        llama_cpp_root=root,
        llama_debug=helper,
        patch_artifact=patch,
        patch_dry_run_artifact=dry_run,
        llama_logits_artifact=logits,
        artifact_date="2030-04-13",
    )
    artifact.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    verification = verify_llamacpp_logits_helper_readiness(
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
    assert verification["persisted_missing_evidence"] == current["missing_evidence"]
    assert verification["current_missing_evidence"] == current["missing_evidence"]

    assert (
        main(
            [
                *base_args,
                "--verify-readiness",
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
    drifted["status"] = "ready"
    artifact.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    mismatch = verify_llamacpp_logits_helper_readiness(
        artifact,
        current_report=current,
    )
    assert mismatch["status"] == "mismatch"
    assert mismatch["all_match"] is False
    assert mismatch["verification_failure_count"] == 1
    assert mismatch["verification_failures"][0]["name"] == (
        "llamacpp_logits_helper_readiness_drift"
    )
    assert (
        main(
            [
                *base_args,
                "--verify-readiness",
                str(artifact),
                "--verification-failures-only",
                "--output",
                str(output),
            ]
        )
        == SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    )
    assert json.loads(output.read_text())[0]["name"] == (
        "llamacpp_logits_helper_readiness_drift"
    )
