from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import (
    SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
    _stable_json_sha256,
)
from scripts.stepfun_llamacpp_logits_entrypoint_inventory import (
    build_llamacpp_logits_entrypoint_inventory,
    main,
    verify_llamacpp_logits_entrypoint_inventory,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_exe(path: Path, *, help_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"HELP = {help_text!r}\n"
        "if '--help' in sys.argv:\n"
        "    print(HELP)\n"
        "else:\n"
        "    print('fake llama binary')\n"
    )
    path.chmod(0o755)


def _write_fake_cmake(path: Path, *, targets: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    help_text = "The following are some of the valid targets for this Makefile:\n" + "".join(
        f"... {target}\n" for target in targets
    )
    path.write_text(
        "#!/usr/bin/env python3\n"
        f"HELP = {help_text!r}\n"
        "print(HELP, end='')\n"
    )
    path.chmod(0o755)


def _write_source_tree(root: Path) -> None:
    (root / "examples/debug").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    (root / "include").mkdir(parents=True)
    (root / "examples/batched").mkdir(parents=True)
    (root / "examples/debug/debug.cpp").write_text(
        "params.save_logits\nllama_get_logits_ith\nsave_output_data\n"
    )
    (root / "common/arg.cpp").write_text(
        "--save-logits\n--logits-output-dir\n--save-all-logits\n--kl-divergence-base\n"
    )
    (root / "include/llama.h").write_text(
        "llama_get_logits\nllama_get_logits_ith\nllama_get_sampled_logits_ith\n"
    )
    (root / "examples/batched/batched.cpp").write_text(
        "batch.logits\nllama_decode\nllama_sampler_sample\n"
    )


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    preflight = tmp_path / "preflight.json"
    plan = tmp_path / "plan.json"
    _write_json(
        preflight,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_llamacpp_logits_preflight",
            "status": "blocked",
        },
    )
    _write_json(
        plan,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_llamacpp_logits_plan",
            "status": "blocked",
        },
    )
    return preflight, plan


def test_stepfun_llamacpp_logits_entrypoint_inventory_reports_source_only_candidates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "llama.cpp"
    build_dir = root / "build"
    bin_dir = build_dir / "bin"
    fake_cmake = tmp_path / "fake-cmake"
    _write_source_tree(root)
    _write_fake_cmake(fake_cmake, targets=["llama-cli", "llama-debug", "llama-batched"])
    _write_exe(bin_dir / "llama-cli", help_text="--logit-bias TOKEN")
    _write_exe(bin_dir / "llama-tokenize", help_text="tokenize help")
    preflight, plan = _write_artifacts(tmp_path)

    report = build_llamacpp_logits_entrypoint_inventory(
        llama_cpp_root=root,
        build_dir=build_dir,
        bin_dir=bin_dir,
        preflight_artifact=preflight,
        plan_artifact=plan,
        cmake=str(fake_cmake),
        artifact_date="2030-02-03",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_llamacpp_logits_entrypoint_inventory"
    assert report["date"] == "2030-02-03"
    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["built_binary_count"] == 2
    assert report["built_logits_dump_binary_present"] is False
    assert report["built_logits_dump_binaries"] == []
    assert report["build_readiness_status"] == "ready_to_build"
    assert report["build_ready_without_source_changes"] is True
    assert report["target_help"]["status"] == "executed"
    assert "llama-debug" in report["target_help"]["available_targets"]
    assert report["build_target_records"][0] == {
        "target": "llama-debug",
        "target_available": True,
        "binary_path": str(bin_dir / "llama-debug"),
        "binary_built": False,
        "build_command": f"cmake --build {build_dir} --target llama-debug",
        "next_action": f"Run `cmake --build {build_dir} --target llama-debug` and refresh this inventory",
    }
    assert report["source_candidate_count"] == 4
    assert report["source_logits_candidate_count"] == 4
    assert [record["path"] for record in report["source_logits_candidates"]] == [
        "examples/debug/debug.cpp",
        "common/arg.cpp",
        "include/llama.h",
        "examples/batched/batched.cpp",
    ]
    assert report["source_candidates"][0]["expected_binary"] == "llama-debug"
    assert report["source_candidates"][0]["expected_binary_built"] is False
    assert report["source_candidates"][0]["candidate_kind"] == "source_only_or_api"
    assert report["missing_evidence"] == [
        "built_llamacpp_logits_dump_binary_present",
        "llama_debug_binary_built",
    ]
    assert report["blocked_reason"] == (
        "llama-debug target is available but the logits-dump binary has not been built"
    )
    assert report["next_action"] == (
        f"Run `cmake --build {build_dir} --target llama-debug`, then refresh this inventory "
        "and the llama.cpp logits preflight"
    )
    assert report["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This inventory only records logits entrypoint availability; "
            "generated_text_matches_target remains unresolved."
        ),
    }


def test_stepfun_llamacpp_logits_entrypoint_inventory_can_be_ready_with_built_dump_binary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "llama.cpp"
    build_dir = root / "build"
    bin_dir = build_dir / "bin"
    fake_cmake = tmp_path / "fake-cmake"
    _write_source_tree(root)
    _write_fake_cmake(fake_cmake, targets=["llama-debug", "llama-batched"])
    _write_exe(bin_dir / "llama-debug", help_text="--save-logits --logits-output-dir")
    preflight, plan = _write_artifacts(tmp_path)

    report = build_llamacpp_logits_entrypoint_inventory(
        llama_cpp_root=root,
        build_dir=build_dir,
        bin_dir=bin_dir,
        preflight_artifact=preflight,
        plan_artifact=plan,
        cmake=str(fake_cmake),
    )

    assert report["status"] == "ready"
    assert report["ready"] is True
    assert report["built_logits_dump_binary_present"] is True
    assert report["built_logits_dump_binaries"][0]["name"] == "llama-debug"
    assert report["built_logits_dump_binaries"][0]["logits_dump_markers"] == [
        "--save-logits",
        "--logits-output-dir",
    ]
    assert report["build_readiness_status"] == "built"
    assert report["build_target_records"][0]["binary_built"] is True
    assert report["missing_evidence"] == []


def test_stepfun_llamacpp_logits_entrypoint_inventory_reports_missing_sources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "empty-llama.cpp"
    build_dir = root / "build"
    bin_dir = build_dir / "bin"
    fake_cmake = tmp_path / "fake-cmake"
    bin_dir.mkdir(parents=True)
    _write_fake_cmake(fake_cmake, targets=[])
    preflight, plan = _write_artifacts(tmp_path)

    report = build_llamacpp_logits_entrypoint_inventory(
        llama_cpp_root=root,
        build_dir=build_dir,
        bin_dir=bin_dir,
        preflight_artifact=preflight,
        plan_artifact=plan,
        cmake=str(fake_cmake),
    )

    assert report["status"] == "blocked"
    assert report["source_logits_candidate_count"] == 0
    assert report["build_readiness_status"] == "blocked"
    assert "llamacpp_source_logits_api_candidate_present" in report["missing_evidence"]
    assert "llama_debug_build_target_available" in report["missing_evidence"]


def test_stepfun_llamacpp_logits_entrypoint_inventory_cli_writes_report(tmp_path: Path) -> None:
    root = tmp_path / "llama.cpp"
    build_dir = root / "build"
    bin_dir = build_dir / "bin"
    fake_cmake = tmp_path / "fake-cmake"
    _write_source_tree(root)
    _write_fake_cmake(fake_cmake, targets=["llama-debug"])
    _write_exe(bin_dir / "llama-cli", help_text="--logit-bias TOKEN")
    preflight, plan = _write_artifacts(tmp_path)
    output = tmp_path / "inventory.json"

    rc = main(
        [
            "--llama-cpp-root",
            str(root),
            "--build-dir",
            str(build_dir),
            "--bin-dir",
            str(bin_dir),
            "--cmake",
            str(fake_cmake),
            "--preflight-artifact",
            str(preflight),
            "--plan-artifact",
            str(plan),
            "--artifact-date",
            "2030-02-04",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-02-04"
    assert payload["status"] == "blocked"
    assert payload["source_logits_candidate_count"] == 4
    assert payload["build_readiness_status"] == "ready_to_build"


def test_stepfun_llamacpp_logits_entrypoint_inventory_cli_compact_modes(tmp_path: Path) -> None:
    root = tmp_path / "llama.cpp"
    build_dir = root / "build"
    bin_dir = build_dir / "bin"
    fake_cmake = tmp_path / "fake-cmake"
    _write_source_tree(root)
    _write_fake_cmake(fake_cmake, targets=["llama-debug"])
    _write_exe(bin_dir / "llama-cli", help_text="--logit-bias TOKEN")
    preflight, plan = _write_artifacts(tmp_path)
    output = tmp_path / "compact.json"
    base_args = [
        "--llama-cpp-root",
        str(root),
        "--build-dir",
        str(build_dir),
        "--bin-dir",
        str(bin_dir),
        "--cmake",
        str(fake_cmake),
        "--preflight-artifact",
        str(preflight),
        "--plan-artifact",
        str(plan),
    ]

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "blocked"
    assert main([*base_args, "--built-logits-binary-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is False
    assert main([*base_args, "--source-candidates-only", "--output", str(output)]) == 0
    assert len(json.loads(output.read_text())) == 4
    assert main([*base_args, "--build-readiness-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "ready_to_build"


def test_stepfun_llamacpp_logits_entrypoint_inventory_verifies_persisted_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "llama.cpp"
    build_dir = root / "build"
    bin_dir = build_dir / "bin"
    fake_cmake = tmp_path / "fake-cmake"
    _write_source_tree(root)
    _write_fake_cmake(fake_cmake, targets=["llama-cli", "llama-debug", "llama-batched"])
    _write_exe(bin_dir / "llama-cli", help_text="--logit-bias TOKEN")
    preflight, plan = _write_artifacts(tmp_path)
    artifact = tmp_path / "inventory.json"
    output = tmp_path / "verify.json"
    base_args = [
        "--llama-cpp-root",
        str(root),
        "--build-dir",
        str(build_dir),
        "--bin-dir",
        str(bin_dir),
        "--cmake",
        str(fake_cmake),
        "--preflight-artifact",
        str(preflight),
        "--plan-artifact",
        str(plan),
        "--artifact-date",
        "2030-02-06",
    ]
    current = build_llamacpp_logits_entrypoint_inventory(
        llama_cpp_root=root,
        build_dir=build_dir,
        bin_dir=bin_dir,
        preflight_artifact=preflight,
        plan_artifact=plan,
        cmake=str(fake_cmake),
        artifact_date="2030-02-06",
    )
    artifact.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    verification = verify_llamacpp_logits_entrypoint_inventory(
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
    assert verification["persisted_built_logits_dump_binary_present"] is False
    assert verification["current_built_logits_dump_binary_present"] is False
    assert verification["persisted_build_readiness_status"] == "ready_to_build"
    assert verification["current_build_readiness_status"] == "ready_to_build"
    assert verification["persisted_missing_evidence"] == [
        "built_llamacpp_logits_dump_binary_present",
        "llama_debug_binary_built",
    ]
    assert verification["current_missing_evidence"] == [
        "built_llamacpp_logits_dump_binary_present",
        "llama_debug_binary_built",
    ]
    assert verification["persisted_source_logits_candidate_count"] == 4
    assert verification["current_source_logits_candidate_count"] == 4
    assert verification["persisted_source_candidate_paths"] == [
        "examples/debug/debug.cpp",
        "common/arg.cpp",
        "include/llama.h",
        "examples/batched/batched.cpp",
    ]
    assert verification["current_source_candidate_paths"] == [
        "examples/debug/debug.cpp",
        "common/arg.cpp",
        "include/llama.h",
        "examples/batched/batched.cpp",
    ]
    assert verification["persisted_built_logits_dump_binaries"] == []
    assert verification["current_built_logits_dump_binaries"] == []
    assert verification["persisted_preflight_sha256"] == _stable_json_sha256(
        json.loads(preflight.read_text())
    )
    assert verification["current_preflight_sha256"] == _stable_json_sha256(
        json.loads(preflight.read_text())
    )
    assert verification["persisted_plan_sha256"] == _stable_json_sha256(
        json.loads(plan.read_text())
    )
    assert verification["current_plan_sha256"] == _stable_json_sha256(
        json.loads(plan.read_text())
    )

    assert (
        main(
            [
                *base_args,
                "--verify-inventory",
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
    drifted["build_readiness_status"] = "blocked"
    artifact.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    mismatch = verify_llamacpp_logits_entrypoint_inventory(
        artifact,
        current_report=current,
    )
    assert mismatch["status"] == "mismatch"
    assert mismatch["all_match"] is False
    assert mismatch["verification_failure_count"] == 1
    assert mismatch["verification_failures"][0]["name"] == (
        "llamacpp_logits_entrypoint_inventory_drift"
    )
    assert (
        main(
            [
                *base_args,
                "--verify-inventory",
                str(artifact),
                "--verification-failures-only",
                "--output",
                str(output),
            ]
        )
        == SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    )
    assert json.loads(output.read_text())[0]["name"] == (
        "llamacpp_logits_entrypoint_inventory_drift"
    )
