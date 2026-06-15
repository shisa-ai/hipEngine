from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import (
    SOURCE_ARTIFACT_MISMATCH_EXIT_CODE,
    _stable_json_sha256,
)
from scripts.stepfun_llamacpp_logits_helper_patch_plan import (
    build_llamacpp_logits_helper_patch_plan,
    main,
    verify_llamacpp_logits_helper_patch_plan,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_prompt_artifact(path: Path) -> None:
    _write_json(
        path,
        {
            "status": "partial_prompt_smoke",
            "prompt": "<|im_start|>user\nhi<|im_end|>",
            "prompt_length": 3,
            "input_ids": [0, 128006, 201],
            "next_token_id": 369,
            "next_token_text": " |",
        },
    )


def _write_llamacpp_source_tree(root: Path, *, include_common_anchor: bool = True) -> None:
    (root / "examples/debug").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    (root / "examples/debug/debug.cpp").write_text(
        "\n".join(
            [
                "#include <regex>",
                "output_data(llama_context * ctx, const llama_model * model, const common_params & params) {",
                "tokens = common_tokenize(ctx, params.prompt, add_bos);",
                "static bool run(llama_context * ctx, const common_params & params) {",
                "std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);",
                "output_data output {ctx, model, params};",
                "if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DEBUG, print_usage)) {",
                "if (!run(ctx, params)) {",
            ]
        )
    )
    (root / "common/common.h").write_text(
        "bool   parse_special = false);\n" if include_common_anchor else "bool parse_special);\n"
    )


def test_stepfun_llamacpp_logits_helper_patch_plan_records_exact_recipe(
    tmp_path: Path,
) -> None:
    root = tmp_path / "llama.cpp"
    prompt = tmp_path / "prompt.json"
    _write_llamacpp_source_tree(root)
    _write_prompt_artifact(prompt)

    report = build_llamacpp_logits_helper_patch_plan(
        llama_cpp_root=root,
        prompt_artifact=prompt,
        build_dir=tmp_path / "build",
        artifact_date="2030-04-07",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_llamacpp_logits_helper_patch_plan"
    assert report["date"] == "2030-04-07"
    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["anchors_ready"] is True
    assert report["missing_anchors"] == []
    assert report["prompt_contract"] == {
        "prompt_artifact_present": True,
        "prompt": "<|im_start|>user\nhi<|im_end|>",
        "prompt_length": 3,
        "input_id_count": 3,
        "input_ids": [0, 128006, 201],
        "token_ids_csv": "0,128006,201",
        "expected_next_token_id": 369,
        "expected_next_token_text": " |",
    }
    assert [step["key"] for step in report["patch_steps"]] == [
        "add_helper_local_cli_state",
        "decode_retained_tokens",
        "save_exact_prompt_tokens",
        "wire_main_to_helper_args",
    ]
    assert report["expected_probe_token_ids_argument"] == "0,128006,201"
    assert report["expected_probe_command_shape"] == [
        "--prompt-token-source retained-input-ids",
        "--token-ids",
        "0,128006,201",
        "--save-logits",
        "--logits-output-dir",
    ]
    assert report["probe_command_after_build"] == (
        "python3 scripts/stepfun_llamacpp_logits_probe.py "
        "--prompt-token-source retained-input-ids --execute --default-output --pretty"
    )
    assert report["missing_evidence"] == [
        "llama_cpp_token_ids_helper_patch_applied",
        "same_prompt_logits_helper_built",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]
    assert report["patch_application_policy"]["edits_external_tree"] is False
    assert report["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This artifact is a patch recipe only; generated_text_matches_target remains unresolved "
            "until the retained-token helper captures same-prompt logits and oracle comparison passes."
        ),
    }


def test_stepfun_llamacpp_logits_helper_patch_plan_reports_missing_anchor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "llama.cpp"
    prompt = tmp_path / "prompt.json"
    _write_llamacpp_source_tree(root, include_common_anchor=False)
    _write_prompt_artifact(prompt)

    report = build_llamacpp_logits_helper_patch_plan(
        llama_cpp_root=root,
        prompt_artifact=prompt,
    )

    assert report["status"] == "blocked"
    assert report["anchors_ready"] is False
    assert report["missing_anchors"] == ["common_tokenize_parse_special_signature"]
    assert report["missing_evidence"] == [
        "llama_cpp_helper_patch_source_anchors_present",
        "llama_cpp_token_ids_helper_patch_applied",
        "same_prompt_logits_helper_built",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]


def test_stepfun_llamacpp_logits_helper_patch_plan_cli_modes(tmp_path: Path) -> None:
    root = tmp_path / "llama.cpp"
    prompt = tmp_path / "prompt.json"
    output = tmp_path / "patch-plan.json"
    _write_llamacpp_source_tree(root)
    _write_prompt_artifact(prompt)
    base_args = [
        "--llama-cpp-root",
        str(root),
        "--prompt-artifact",
        str(prompt),
        "--build-dir",
        str(tmp_path / "build"),
        "--artifact-date",
        "2030-04-08",
    ]

    assert main([*base_args, "--output", str(output), "--pretty"]) == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-04-08"
    assert payload["anchors_ready"] is True
    assert payload["expected_probe_token_ids_argument"] == "0,128006,201"

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "blocked"
    assert main([*base_args, "--anchors-ready-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is True
    assert main([*base_args, "--missing-evidence-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == [
        "llama_cpp_token_ids_helper_patch_applied",
        "same_prompt_logits_helper_built",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]
    assert main([*base_args, "--patch-steps-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text())[0]["key"] == "add_helper_local_cli_state"
    assert main([*base_args, "--sha-only", "--output", str(output)]) == 0
    assert isinstance(json.loads(output.read_text()), str)


def test_stepfun_llamacpp_logits_helper_patch_plan_verifies_persisted_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "llama.cpp"
    prompt = tmp_path / "prompt.json"
    build_dir = tmp_path / "build"
    artifact = tmp_path / "patch-plan.json"
    output = tmp_path / "verify.json"
    _write_llamacpp_source_tree(root)
    _write_prompt_artifact(prompt)
    base_args = [
        "--llama-cpp-root",
        str(root),
        "--prompt-artifact",
        str(prompt),
        "--build-dir",
        str(build_dir),
        "--artifact-date",
        "2030-04-09",
    ]
    current = build_llamacpp_logits_helper_patch_plan(
        llama_cpp_root=root,
        prompt_artifact=prompt,
        build_dir=build_dir,
        artifact_date="2030-04-09",
    )
    artifact.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    verification = verify_llamacpp_logits_helper_patch_plan(
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
    assert verification["persisted_anchors_ready"] is True
    assert verification["current_anchors_ready"] is True
    assert verification["persisted_missing_anchors"] == []
    assert verification["current_missing_anchors"] == []
    assert verification["persisted_missing_evidence"] == [
        "llama_cpp_token_ids_helper_patch_applied",
        "same_prompt_logits_helper_built",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]
    assert verification["current_missing_evidence"] == [
        "llama_cpp_token_ids_helper_patch_applied",
        "same_prompt_logits_helper_built",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]
    assert verification["persisted_patch_step_keys"] == [
        "add_helper_local_cli_state",
        "decode_retained_tokens",
        "save_exact_prompt_tokens",
        "wire_main_to_helper_args",
    ]
    assert verification["current_patch_step_keys"] == [
        "add_helper_local_cli_state",
        "decode_retained_tokens",
        "save_exact_prompt_tokens",
        "wire_main_to_helper_args",
    ]
    assert verification["persisted_expected_probe_token_ids_argument"] == "0,128006,201"
    assert verification["current_expected_probe_token_ids_argument"] == "0,128006,201"
    assert verification["persisted_expected_probe_command_shape"] == [
        "--prompt-token-source retained-input-ids",
        "--token-ids",
        "0,128006,201",
        "--save-logits",
        "--logits-output-dir",
    ]
    assert verification["current_expected_probe_command_shape"] == [
        "--prompt-token-source retained-input-ids",
        "--token-ids",
        "0,128006,201",
        "--save-logits",
        "--logits-output-dir",
    ]
    assert verification["persisted_prompt_input_ids"] == [0, 128006, 201]
    assert verification["current_prompt_input_ids"] == [0, 128006, 201]
    assert verification["persisted_build_command"] == f"cmake --build {build_dir} --target llama-debug -j"
    assert verification["current_build_command"] == f"cmake --build {build_dir} --target llama-debug -j"
    assert verification["persisted_probe_command_after_build"] == (
        "python3 scripts/stepfun_llamacpp_logits_probe.py "
        "--prompt-token-source retained-input-ids --execute --default-output --pretty"
    )
    assert verification["current_probe_command_after_build"] == (
        "python3 scripts/stepfun_llamacpp_logits_probe.py "
        "--prompt-token-source retained-input-ids --execute --default-output --pretty"
    )

    assert (
        main(
            [
                *base_args,
                "--verify-patch-plan",
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
    drifted["anchors_ready"] = False
    artifact.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    mismatch = verify_llamacpp_logits_helper_patch_plan(
        artifact,
        current_report=current,
    )
    assert mismatch["status"] == "mismatch"
    assert mismatch["all_match"] is False
    assert mismatch["verification_failure_count"] == 1
    assert mismatch["verification_failures"][0]["name"] == (
        "llamacpp_logits_helper_patch_plan_drift"
    )
    assert (
        main(
            [
                *base_args,
                "--verify-patch-plan",
                str(artifact),
                "--verification-failures-only",
                "--output",
                str(output),
            ]
        )
        == SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
    )
    assert json.loads(output.read_text())[0]["name"] == (
        "llamacpp_logits_helper_patch_plan_drift"
    )
