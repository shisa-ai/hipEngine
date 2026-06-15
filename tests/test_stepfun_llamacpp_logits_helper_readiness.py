from __future__ import annotations

import json
import stat
from pathlib import Path

from scripts.stepfun_llamacpp_logits_helper_readiness import (
    PATCH_APPLIED_MARKERS,
    build_llamacpp_logits_helper_readiness,
    main,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_source(root: Path, *, patched: bool) -> None:
    path = root / "examples/debug/debug.cpp"
    path.parent.mkdir(parents=True, exist_ok=True)
    if patched:
        path.write_text("\n".join(str(marker["marker"]) for marker in PATCH_APPLIED_MARKERS))
    else:
        path.write_text("static bool run(llama_context * ctx, const common_params & params) {}\n")


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
    helper = tmp_path / "build/bin/llama-debug"
    logits = tmp_path / "logits.json"
    _write_source(root, patched=False)
    patch.write_text("diff --git a/examples/debug/debug.cpp b/examples/debug/debug.cpp\n")
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
        llama_logits_artifact=logits,
        artifact_date="2030-04-11",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_llamacpp_logits_helper_readiness"
    assert report["date"] == "2030-04-11"
    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["patch_artifact"]["exists"] is True
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
    helper = tmp_path / "build/bin/llama-debug"
    logits = tmp_path / "logits.json"
    _write_source(root, patched=True)
    patch.write_text("diff --git a/examples/debug/debug.cpp b/examples/debug/debug.cpp\n")
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
        llama_logits_artifact=logits,
    )

    assert report["status"] == "ready"
    assert report["ready"] is True
    assert report["source_patch"]["patch_applied"] is True
    assert report["source_patch"]["missing_markers"] == []
    assert report["llama_debug"]["retained_token_ids_capable"] is True
    assert report["llama_debug"]["parse_special_capable"] is True
    assert report["llama_logits_artifact"]["status"] == "captured"
    assert report["missing_evidence"] == []
    assert report["blocked_reason"] is None


def test_stepfun_llamacpp_logits_helper_readiness_cli_modes(tmp_path: Path) -> None:
    root = tmp_path / "llama.cpp"
    patch = tmp_path / "helper.patch"
    helper = tmp_path / "build/bin/llama-debug"
    logits = tmp_path / "logits.json"
    output = tmp_path / "readiness.json"
    _write_source(root, patched=False)
    patch.write_text("patch")
    _write_fake_llama_debug(helper, help_text="--save-logits --logits-output-dir --special")
    _write_json(logits, {"status": "blocked", "ready": False})
    base_args = [
        "--llama-cpp-root",
        str(root),
        "--llama-debug",
        str(helper),
        "--patch-artifact",
        str(patch),
        "--llama-logits-artifact",
        str(logits),
        "--artifact-date",
        "2030-04-12",
    ]

    assert main([*base_args, "--output", str(output), "--pretty"]) == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-04-12"
    assert payload["status"] == "blocked"

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
