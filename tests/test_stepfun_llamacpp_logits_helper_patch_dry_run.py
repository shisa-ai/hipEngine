from __future__ import annotations

import json
from pathlib import Path

from scripts import stepfun_llamacpp_logits_helper_patch_dry_run as dry_run
from scripts.stepfun_llamacpp_logits_helper_patch_dry_run import (
    build_llamacpp_logits_helper_patch_dry_run,
    main,
)


def _write_debug_source(root: Path, *, omit_usage: bool = False) -> None:
    path = root / "examples/debug/debug.cpp"
    path.parent.mkdir(parents=True, exist_ok=True)
    pieces = [
        dry_run.INCLUDE_OLD,
        dry_run.USAGE_OLD if not omit_usage else "missing usage anchor\n",
        dry_run.HELPER_INSERT_ANCHOR,
        "}\n",
        dry_run.OUTPUT_CTOR_OLD,
        "    }\n",
        dry_run.RUN_SIGNATURE_OLD,
        dry_run.RUN_TOKENIZE_OLD,
        dry_run.OUTPUT_CALL_OLD,
        "}\n",
        "int main(int argc, char ** argv) {\n",
        dry_run.MAIN_PARSE_OLD,
        dry_run.RUN_CALL_OLD,
        "}\n",
    ]
    path.write_text("".join(pieces))


def test_stepfun_llamacpp_logits_helper_patch_dry_run_generates_applyable_recipe(
    tmp_path: Path,
) -> None:
    _write_debug_source(tmp_path)

    report = build_llamacpp_logits_helper_patch_dry_run(
        llama_cpp_root=tmp_path,
        artifact_date="2030-04-09",
        skip_apply_check=True,
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_llamacpp_logits_helper_patch_dry_run"
    assert report["date"] == "2030-04-09"
    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["target_file"] == "examples/debug/debug.cpp"
    assert report["source_exists"] is True
    assert report["missing_transforms"] == []
    assert all(result["applied"] is True for result in report["transform_results"])
    assert report["patch_ready"] is True
    assert report["git_apply_check"] == {
        "status": "skipped",
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    assert report["patch_summary"] == {
        "adds_debug_local_token_ids_flag": True,
        "adds_debug_local_parse_special_flag": True,
        "decodes_retained_token_ids_without_retokenizing": True,
        "threads_exact_tokens_to_saved_prompt_metadata": True,
    }
    assert report["missing_evidence"] == [
        "llama_cpp_token_ids_helper_patch_applied",
        "same_prompt_logits_helper_built",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]
    assert report["probe_command_after_build"] == (
        "python3 scripts/stepfun_llamacpp_logits_probe.py "
        "--prompt-token-source retained-input-ids --execute --default-output --pretty"
    )
    assert report["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": "Dry-run patch applicability is not same-prompt logits evidence.",
    }


def test_stepfun_llamacpp_logits_helper_patch_dry_run_reports_missing_transform(
    tmp_path: Path,
) -> None:
    _write_debug_source(tmp_path, omit_usage=True)

    report = build_llamacpp_logits_helper_patch_dry_run(
        llama_cpp_root=tmp_path,
        skip_apply_check=True,
    )

    assert report["status"] == "blocked"
    assert report["patch_ready"] is False
    assert report["missing_transforms"] == ["usage_helper_flags"]
    assert report["missing_evidence"] == [
        "llama_cpp_token_ids_helper_patch_transforms_apply",
        "llama_cpp_token_ids_helper_patch_applied",
        "same_prompt_logits_helper_built",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]


def test_stepfun_llamacpp_logits_helper_patch_dry_run_cli_modes(tmp_path: Path) -> None:
    _write_debug_source(tmp_path)
    output = tmp_path / "dry-run.json"
    base_args = [
        "--llama-cpp-root",
        str(tmp_path),
        "--skip-apply-check",
        "--artifact-date",
        "2030-04-10",
    ]

    assert main([*base_args, "--output", str(output), "--pretty"]) == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-04-10"
    assert payload["patch_ready"] is True
    assert payload["git_apply_check"]["status"] == "skipped"

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "blocked"
    assert main([*base_args, "--patch-ready-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is True
    assert main([*base_args, "--apply-check-status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "skipped"
    assert main([*base_args, "--missing-evidence-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == [
        "llama_cpp_token_ids_helper_patch_applied",
        "same_prompt_logits_helper_built",
        "llama_cpp_same_prompt_logits_artifact_present",
    ]
    assert main([*base_args, "--patch-sha-only", "--output", str(output)]) == 0
    assert isinstance(json.loads(output.read_text()), str)
    assert main([*base_args, "--patch-only", "--output", str(output)]) == 0
    patch_text = output.read_text()
    assert "--token-ids" in patch_text
    assert "common_tokenize(ctx, params.prompt, add_bos, extra.parse_special)" in patch_text
    assert main([*base_args, "--sha-only", "--output", str(output)]) == 0
    assert isinstance(json.loads(output.read_text()), str)
