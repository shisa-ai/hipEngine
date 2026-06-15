#!/usr/bin/env python3
"""Emit a compact manifest for the remaining StepFun GGUF correctness blockers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stepfun_correctness_status as status_mod
from scripts import stepfun_kv_next_token_check as kv_next_token_check_mod
from scripts import stepfun_kv_trace_check as kv_trace_check_mod
from scripts import stepfun_llamacpp_logits_helper_patch_dry_run as oracle_patch_mod
from scripts import stepfun_llamacpp_logits_helper_readiness as oracle_readiness_mod
from scripts import stepfun_oracle_artifact_check as oracle_check_mod

DEFAULT_KV_TRACE_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-kv-kernel-trace.json"
)
DEFAULT_KV_NEXT_TOKEN_ARTIFACT = Path(
    "benchmarks/results/2026-05-31-stepfun-q3kl-kv-backed-next-token.json"
)


def _string_path(path_value: object) -> str | None:
    """Return a path-like value as a string, preserving None for absent paths."""

    if path_value in (None, ""):
        return None
    return str(path_value)


def _concrete_validator_command(
    command_template: str,
    *,
    placeholder: str,
    artifact_path: object,
) -> tuple[str | None, str | None]:
    """Return a runnable validator command and digest for a concrete artifact path."""

    path = _string_path(artifact_path)
    if path is None:
        return None, None
    command = command_template.replace(placeholder, path)
    return command, status_mod._stable_json_sha256(command)


def _artifact_file_present(path_value: object) -> bool:
    """Return whether an artifact path is provided and currently exists."""

    if not isinstance(path_value, str) or not path_value:
        return False
    return Path(path_value).exists()


def _load_json_object_if_present(path: Path) -> dict[str, object] | None:
    """Return a JSON object artifact when present, otherwise None."""

    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else None


def _file_sha256_if_present(path: Path) -> str | None:
    """Return a file SHA-256 digest when a path exists."""

    if not path.exists():
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative_path(path: Path) -> Path:
    """Resolve a repo artifact path independent of caller cwd."""

    return path if path.is_absolute() else REPO_ROOT / path


def _artifact_evidence_satisfied(artifact: dict[str, object]) -> bool:
    """Return whether a required artifact currently satisfies its blocker gate."""

    name = artifact.get("name")
    if name == "llama_cpp_oracle_success_artifact":
        return (
            artifact.get("current_status") == "executed"
            and artifact.get("current_returncode") == 0
            and artifact.get("first_missing_evidence") is None
        )
    if artifact.get("current") not in (None, ""):
        return True
    required_for = str(artifact.get("required_for") or "")
    if required_for.endswith("_missing"):
        return False
    return _artifact_file_present(artifact.get("path"))


def _artifact_missing_reason(artifact: dict[str, object]) -> object:
    """Return the most specific machine-readable reason an artifact is unsatisfied."""

    for key in (
        "first_missing_evidence",
        "current_blocker_kind",
        "required_for",
        "current_status",
    ):
        reason = artifact.get(key)
        if reason not in (None, ""):
            return reason
    if not _artifact_file_present(artifact.get("path")):
        return "artifact_file_missing"
    return None


def _summarize_required_artifact(artifact: dict[str, object]) -> dict[str, object]:
    """Return a compact satisfaction record for one required artifact."""

    evidence_satisfied = _artifact_evidence_satisfied(artifact)
    return {
        "name": artifact.get("name"),
        "readiness_gate": artifact.get("readiness_gate"),
        "required_for": artifact.get("required_for"),
        "path": artifact.get("path"),
        "artifact_file_present": _artifact_file_present(artifact.get("path")),
        "evidence_satisfied": evidence_satisfied,
        "missing": not evidence_satisfied,
        "missing_reason": None
        if evidence_satisfied
        else _artifact_missing_reason(artifact),
        "current_status": artifact.get("current_status"),
        "current_returncode": artifact.get("current_returncode"),
        "current_blocker_kind": artifact.get("current_blocker_kind"),
        "first_missing_evidence": artifact.get("first_missing_evidence"),
        "recommended_command_kind": artifact.get("recommended_command_kind"),
        "recommended_command_sha256": artifact.get("recommended_command_sha256"),
        "validator_command_kind": artifact.get("validator_command_kind"),
        "validator_artifact_path": artifact.get("validator_artifact_path"),
        "validator_command": artifact.get("validator_command"),
        "validator_command_sha256": artifact.get("validator_command_sha256"),
        "validator_command_concrete": artifact.get("validator_command_concrete"),
        "validator_command_concrete_sha256": artifact.get(
            "validator_command_concrete_sha256"
        ),
        "validator_expected_kernel_families": artifact.get(
            "validator_expected_kernel_families"
        ),
        "validator_expected_kernel_families_sha256": artifact.get(
            "validator_expected_kernel_families_sha256"
        ),
        "validator_expected_evidence_checks": artifact.get(
            "validator_expected_evidence_checks"
        ),
        "validator_expected_evidence_checks_sha256": artifact.get(
            "validator_expected_evidence_checks_sha256"
        ),
        "validator_success_status": artifact.get("validator_success_status"),
        "validator_failure_exit_code": artifact.get("validator_failure_exit_code"),
    }


def _summarize_required_artifacts(
    artifacts: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return compact satisfaction records for required blocker artifacts."""

    return [_summarize_required_artifact(artifact) for artifact in artifacts]


def _summarize_validator_commands(
    artifact_status: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return compact validator commands for required evidence artifacts."""

    return [
        {
            "artifact_name": record.get("name"),
            "readiness_gate": record.get("readiness_gate"),
            "required_for": record.get("required_for"),
            "missing": record.get("missing"),
            "missing_reason": record.get("missing_reason"),
            "evidence_satisfied": record.get("evidence_satisfied"),
            "validator_command_kind": record.get("validator_command_kind"),
            "validator_artifact_path": record.get("validator_artifact_path"),
            "validator_command": record.get("validator_command"),
            "validator_command_sha256": record.get("validator_command_sha256"),
            "validator_command_concrete": record.get("validator_command_concrete"),
            "validator_command_concrete_sha256": record.get(
                "validator_command_concrete_sha256"
            ),
            "validator_expected_kernel_families_sha256": record.get(
                "validator_expected_kernel_families_sha256"
            ),
            "validator_expected_evidence_checks_sha256": record.get(
                "validator_expected_evidence_checks_sha256"
            ),
            "validator_success_status": record.get("validator_success_status"),
            "validator_failure_exit_code": record.get("validator_failure_exit_code"),
        }
        for record in artifact_status
        if record.get("validator_command_kind") not in (None, "")
    ]


def _oracle_helper_prerequisite_handoff() -> dict[str, object]:
    """Return same-prompt llama.cpp helper prerequisite handoff metadata."""

    readiness_path = oracle_readiness_mod.DEFAULT_OUTPUT
    patch_path = oracle_patch_mod.DEFAULT_PATCH_OUTPUT
    dry_run_path = oracle_patch_mod.DEFAULT_OUTPUT
    readiness_file = _repo_relative_path(readiness_path)
    patch_file = _repo_relative_path(patch_path)
    dry_run_file = _repo_relative_path(dry_run_path)
    readiness = _load_json_object_if_present(readiness_file) or {}
    patch_dry_run = readiness.get("patch_dry_run_artifact")
    patch_dry_run_record = patch_dry_run if isinstance(patch_dry_run, dict) else {}
    command_records_raw = readiness.get("required_next_command_records")
    command_records = command_records_raw if isinstance(command_records_raw, list) else []
    patch_apply_command = None
    required_commands = readiness.get("required_next_commands")
    if isinstance(required_commands, list) and required_commands:
        patch_apply_command = required_commands[0]
    if patch_apply_command is None:
        patch_apply_command = (
            "git -C /home/lhl/llama.cpp/llama.cpp-vulkan apply --unidiff-zero "
            "/home/lhl/hipEngine-stepfun-3.7-flash/benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-helper.patch"
        )
    readiness_command = (
        "python3 scripts/stepfun_llamacpp_logits_helper_readiness.py --default-output --pretty"
    )
    dry_run_command = (
        "python3 scripts/stepfun_llamacpp_logits_helper_patch_dry_run.py --default-output --pretty"
    )
    probe_command = (
        "python3 scripts/stepfun_llamacpp_logits_probe.py --prompt-token-source retained-input-ids --execute --default-output --pretty"
    )
    return {
        "schema_version": 1,
        "source": "stepfun_llamacpp_logits_helper_readiness",
        "readiness_artifact": str(readiness_path),
        "readiness_artifact_present": readiness_file.exists(),
        "readiness_artifact_sha256": _file_sha256_if_present(readiness_file),
        "readiness_status": readiness.get("status"),
        "readiness_ready": readiness.get("ready"),
        "readiness_missing_evidence": list(readiness.get("missing_evidence", []))
        if isinstance(readiness.get("missing_evidence"), list)
        else [],
        "patch_artifact": str(patch_path),
        "patch_artifact_present": patch_file.exists(),
        "patch_artifact_sha256": _file_sha256_if_present(patch_file),
        "patch_dry_run_artifact": str(dry_run_path),
        "patch_dry_run_artifact_present": dry_run_file.exists(),
        "patch_dry_run_artifact_sha256": _file_sha256_if_present(dry_run_file),
        "patch_dry_run_ready": patch_dry_run_record.get("patch_ready"),
        "patch_dry_run_apply_check_status": patch_dry_run_record.get(
            "git_apply_check_status"
        ),
        "patch_artifact_sha256_matches_dry_run": patch_dry_run_record.get(
            "patch_artifact_sha256_matches"
        ),
        "required_next_command_records": command_records,
        "required_next_command_records_sha256": readiness.get(
            "required_next_command_records_sha256"
        ),
        "patch_apply_command": patch_apply_command,
        "patch_apply_command_sha256": status_mod._stable_json_sha256(
            patch_apply_command
        ),
        "dry_run_refresh_command": dry_run_command,
        "dry_run_refresh_command_sha256": status_mod._stable_json_sha256(
            dry_run_command
        ),
        "readiness_refresh_command": readiness_command,
        "readiness_refresh_command_sha256": status_mod._stable_json_sha256(
            readiness_command
        ),
        "probe_command_after_build": probe_command,
        "probe_command_after_build_sha256": status_mod._stable_json_sha256(
            probe_command
        ),
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "These are helper readiness prerequisites; same-prompt logits/oracle "
                "parity remains blocked until the retained-token probe captures evidence."
            ),
        },
    }


def _kv_runner_prerequisite_handoff() -> dict[str, object]:
    """Return KV-backed decode runner prerequisite handoff metadata."""

    session_path = Path(
        "benchmarks/results/2026-06-15-stepfun-q3kl-kv-session-contract.json"
    )
    preflight_path = Path(
        "benchmarks/results/2026-06-15-stepfun-q3kl-kv-evidence-preflight.json"
    )
    blocker_path = Path(
        "benchmarks/results/2026-06-15-stepfun-q3kl-kv-backed-blocker-status.json"
    )
    session_file = _repo_relative_path(session_path)
    preflight_file = _repo_relative_path(preflight_path)
    blocker_file = _repo_relative_path(blocker_path)
    session = _load_json_object_if_present(session_file) or {}
    contract = session.get("contract") if isinstance(session.get("contract"), dict) else {}
    decode_entrypoint = (
        session.get("decode_entrypoint_blocker")
        if isinstance(session.get("decode_entrypoint_blocker"), dict)
        else {}
    )
    preflight = _load_json_object_if_present(preflight_file) or {}
    blocker = _load_json_object_if_present(blocker_file) or {}
    source_status = blocker.get("streaming_runner_source_status")
    source_status_record = source_status if isinstance(source_status, dict) else {}
    runtime_wiring = blocker.get("runtime_wiring_symbol_validation")
    runtime_wiring_record = runtime_wiring if isinstance(runtime_wiring, dict) else {}
    trace_artifact = Path(
        "benchmarks/results/2026-05-31-stepfun-q3kl-kv-kernel-trace.json"
    )
    next_token_artifact = Path(
        "benchmarks/results/2026-05-31-stepfun-q3kl-kv-backed-next-token.json"
    )
    session_command = (
        "python3 scripts/stepfun_kv_session_contract.py --default-output --pretty"
    )
    preflight_command = (
        "python3 scripts/stepfun_kv_evidence_preflight.py --default-output --pretty"
    )
    blocker_command = (
        "python3 scripts/stepfun_kv_blocker_status.py --default-output --pretty"
    )
    trace_command = (
        "python3 scripts/stepfun_kv_trace_check.py "
        f"--trace {trace_artifact} "
        f"--resource-artifact {status_mod.DEFAULT_RESOURCE_ARTIFACT} "
        "--summary-only --fail-on-missing --pretty"
    )
    next_token_command = (
        "python3 scripts/stepfun_kv_next_token_check.py "
        f"--artifact {next_token_artifact} "
        f"--prompt-artifact {status_mod.DEFAULT_PROMPT_ARTIFACT} "
        "--summary-only --fail-on-missing --pretty"
    )
    missing_required_paths = preflight.get("missing_required_artifact_paths")
    return {
        "schema_version": 1,
        "source": "stepfun_kv_evidence_preflight",
        "session_contract_artifact": str(session_path),
        "session_contract_artifact_present": session_file.exists(),
        "session_contract_artifact_sha256": _file_sha256_if_present(session_file),
        "session_contract_status": session.get("status"),
        "session_contract_ready": contract.get("ready"),
        "session_contract_executable": contract.get("executable"),
        "session_contract_blocked_by": contract.get("blocked_by"),
        "session_contract_pre_run_upload_checks_passed": contract.get(
            "pre_run_upload_checks_passed"
        ),
        "session_contract_launch_operation_count": contract.get(
            "launch_operation_count"
        ),
        "session_contract_all_launches_ready": contract.get("all_launches_ready"),
        "session_contract_all_launches_have_dispatch_keys": contract.get(
            "all_launches_have_dispatch_keys"
        ),
        "decode_entrypoint_source": decode_entrypoint.get("source"),
        "decode_entrypoint_executable": decode_entrypoint.get("executable"),
        "decode_entrypoint_ready": decode_entrypoint.get("ready"),
        "decode_entrypoint_blocked_by": decode_entrypoint.get("blocked_by"),
        "decode_entrypoint_next_action": decode_entrypoint.get("next_action"),
        "decode_entrypoint_no_kernel_launches": decode_entrypoint.get(
            "no_kernel_launches"
        ),
        "decode_entrypoint_kv_dispatch_key_names": decode_entrypoint.get(
            "kv_dispatch_key_names"
        ),
        "decode_entrypoint_kv_dispatch_keys_sha256": decode_entrypoint.get(
            "kv_dispatch_keys_sha256"
        ),
        "decode_entrypoint_all_kv_dispatch_keys_bound": decode_entrypoint.get(
            "all_kv_dispatch_keys_bound"
        ),
        "decode_entrypoint_rendered_prompt_sha256": decode_entrypoint.get(
            "rendered_prompt_sha256"
        ),
        "decode_entrypoint_input_ids_sha256": decode_entrypoint.get(
            "input_ids_sha256"
        ),
        "decode_entrypoint_span_input_payload_entry_names": decode_entrypoint.get(
            "span_input_payload_entry_names"
        ),
        "decode_entrypoint_span_input_payloads_sha256": decode_entrypoint.get(
            "span_input_payloads_sha256"
        ),
        "decode_entrypoint_pre_run_payload_fingerprints_sha256": decode_entrypoint.get(
            "pre_run_payload_fingerprints_sha256"
        ),
        "decode_entrypoint_pre_run_upload_plan_sha256": decode_entrypoint.get(
            "pre_run_upload_plan_sha256"
        ),
        "decode_entrypoint_pre_run_upload_entry_count": decode_entrypoint.get(
            "pre_run_upload_entry_count"
        ),
        "decode_entrypoint_pre_run_upload_total_nbytes": decode_entrypoint.get(
            "pre_run_upload_total_nbytes"
        ),
        "decode_entrypoint_pre_run_upload_checks_passed": decode_entrypoint.get(
            "pre_run_upload_checks_passed"
        ),
        "decode_entrypoint_launch_operation_sequence_sha256": decode_entrypoint.get(
            "launch_operation_sequence_sha256"
        ),
        "decode_entrypoint_launch_operation_records_sha256": decode_entrypoint.get(
            "launch_operation_records_sha256"
        ),
        "evidence_preflight_artifact": str(preflight_path),
        "evidence_preflight_artifact_present": preflight_file.exists(),
        "evidence_preflight_artifact_sha256": _file_sha256_if_present(
            preflight_file
        ),
        "evidence_preflight_status": preflight.get("status"),
        "evidence_preflight_next_action": preflight.get("next_action"),
        "required_artifacts_present": preflight.get("required_artifacts_present"),
        "missing_required_artifact_paths": list(missing_required_paths)
        if isinstance(missing_required_paths, list)
        else [],
        "kv_blocker_status_artifact": str(blocker_path),
        "kv_blocker_status_artifact_present": blocker_file.exists(),
        "kv_blocker_status_artifact_sha256": _file_sha256_if_present(blocker_file),
        "kv_blocker_status": blocker.get("status"),
        "kv_backed_decode_ready": blocker.get("kv_backed_decode_ready"),
        "kv_decode_dispatch_ready": blocker.get("kv_decode_dispatch_ready"),
        "streaming_runner_source_status": source_status_record,
        "runtime_wiring_symbol_validation": runtime_wiring_record,
        "trace_artifact": str(trace_artifact),
        "next_token_artifact": str(next_token_artifact),
        "session_contract_refresh_command": session_command,
        "session_contract_refresh_command_sha256": status_mod._stable_json_sha256(
            session_command
        ),
        "evidence_preflight_refresh_command": preflight_command,
        "evidence_preflight_refresh_command_sha256": status_mod._stable_json_sha256(
            preflight_command
        ),
        "kv_blocker_status_refresh_command": blocker_command,
        "kv_blocker_status_refresh_command_sha256": status_mod._stable_json_sha256(
            blocker_command
        ),
        "trace_validator_command": trace_command,
        "trace_validator_command_sha256": status_mod._stable_json_sha256(
            trace_command
        ),
        "next_token_validator_command": next_token_command,
        "next_token_validator_command_sha256": status_mod._stable_json_sha256(
            next_token_command
        ),
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "These are KV runner prerequisites; KV-backed decode remains blocked "
                "until the streaming loop, kernel trace, and next-token artifact exist."
            ),
        },
    }


def _oracle_validator_handoff(
    prompt_artifact: object,
    oracle_artifact: object,
) -> dict[str, object]:
    """Return the validation command for a retained llama.cpp oracle artifact."""

    artifact_placeholder = "<llama_cpp_oracle_success_artifact.json>"
    prompt_arg = str(prompt_artifact) if prompt_artifact else str(
        status_mod.DEFAULT_PROMPT_ARTIFACT
    )
    command = (
        "python3 scripts/stepfun_oracle_artifact_check.py "
        f"--artifact {artifact_placeholder} "
        f"--prompt-artifact {prompt_arg} "
        "--summary-only --fail-on-missing --pretty"
    )
    concrete_command, concrete_sha256 = _concrete_validator_command(
        command,
        placeholder=artifact_placeholder,
        artifact_path=oracle_artifact,
    )
    expected_checks = [
        "oracle_success_status",
        "oracle_returncode_zero",
        "oracle_binary_metadata_recorded",
        "step35_supported_by_oracle",
        "no_timeout_or_oracle_blocker",
        "prompt_length_matches_target",
        "n_predict_one",
        "expected_token_metadata_matches_target",
        "top_token_metadata_matches_target",
        "top_token_logit_matches_target",
        "generated_text_nonempty",
        "generated_text_matches_target",
    ]
    return {
        "validator_command_kind": "oracle_artifact_check_command",
        "validator_artifact_path": _string_path(oracle_artifact),
        "validator_command": command,
        "validator_command_sha256": status_mod._stable_json_sha256(command),
        "validator_command_concrete": concrete_command,
        "validator_command_concrete_sha256": concrete_sha256,
        "validator_expected_evidence_checks": expected_checks,
        "validator_expected_evidence_checks_sha256": status_mod._stable_json_sha256(
            expected_checks
        ),
        "validator_success_status": "passed",
        "validator_failure_exit_code": oracle_check_mod.FAILED_EXIT_CODE,
    }


def _kv_trace_validator_handoff(
    resource_artifact: object,
    trace_artifact: object,
) -> dict[str, object]:
    """Return the validation command for a retained StepFun KV trace artifact."""

    trace_placeholder = "<kv_kernel_trace_artifact.csv-or-json>"
    resource_arg = str(resource_artifact) if resource_artifact else str(
        status_mod.DEFAULT_RESOURCE_ARTIFACT
    )
    command = (
        "python3 scripts/stepfun_kv_trace_check.py "
        f"--trace {trace_placeholder} "
        f"--resource-artifact {resource_arg} "
        "--summary-only --fail-on-missing --pretty"
    )
    concrete_command, concrete_sha256 = _concrete_validator_command(
        command,
        placeholder=trace_placeholder,
        artifact_path=trace_artifact,
    )
    expected_families = [
        {
            "name": family.get("name"),
            "operation": family.get("operation"),
            "symbols": list(family.get("symbols", ())),
        }
        for family in kv_trace_check_mod.REQUIRED_KERNEL_FAMILIES
    ]
    return {
        "validator_command_kind": "kv_trace_check_command",
        "validator_artifact_path": _string_path(trace_artifact),
        "validator_command": command,
        "validator_command_sha256": status_mod._stable_json_sha256(command),
        "validator_command_concrete": concrete_command,
        "validator_command_concrete_sha256": concrete_sha256,
        "validator_expected_kernel_families": expected_families,
        "validator_expected_kernel_families_sha256": status_mod._stable_json_sha256(
            expected_families
        ),
        "validator_success_status": "passed",
        "validator_failure_exit_code": kv_trace_check_mod.FAILED_EXIT_CODE,
    }


def _kv_next_token_validator_handoff(
    prompt_artifact: object,
    token_artifact: object,
) -> dict[str, object]:
    """Return the validation command for a retained StepFun KV next-token artifact."""

    artifact_placeholder = "<kv_backed_next_token_artifact.json>"
    prompt_arg = str(prompt_artifact) if prompt_artifact else str(
        status_mod.DEFAULT_PROMPT_ARTIFACT
    )
    command = (
        "python3 scripts/stepfun_kv_next_token_check.py "
        f"--artifact {artifact_placeholder} "
        f"--prompt-artifact {prompt_arg} "
        "--summary-only --fail-on-missing --pretty"
    )
    concrete_command, concrete_sha256 = _concrete_validator_command(
        command,
        placeholder=artifact_placeholder,
        artifact_path=token_artifact,
    )
    expected_checks = [
        "artifact_success_status",
        "kv_backed_runtime_path",
        "streaming_runner_ready",
        "not_host_composed_layer_prefix",
        "prompt_length_matches_target",
        "next_token_id_matches_target",
        "next_token_text_matches_target",
        "next_token_logit_recorded_finite",
        "next_token_logit_within_tolerance",
    ]
    return {
        "validator_command_kind": "kv_next_token_check_command",
        "validator_artifact_path": _string_path(token_artifact),
        "validator_command": command,
        "validator_command_sha256": status_mod._stable_json_sha256(command),
        "validator_command_concrete": concrete_command,
        "validator_command_concrete_sha256": concrete_sha256,
        "validator_expected_evidence_checks": expected_checks,
        "validator_expected_evidence_checks_sha256": status_mod._stable_json_sha256(
            expected_checks
        ),
        "validator_success_status": "passed",
        "validator_failure_exit_code": kv_next_token_check_mod.FAILED_EXIT_CODE,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-artifact",
        type=Path,
        default=status_mod.DEFAULT_PROMPT_ARTIFACT,
        help="Prompt/layer-prefix artifact to summarize.",
    )
    parser.add_argument(
        "--oracle-artifact",
        type=Path,
        default=status_mod.DEFAULT_ORACLE_ARTIFACT,
        help="llama.cpp oracle artifact to summarize.",
    )
    parser.add_argument(
        "--resource-artifact",
        type=Path,
        default=status_mod.DEFAULT_RESOURCE_ARTIFACT,
        help="StepFun text-resource dry-run artifact to summarize.",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=status_mod.DEFAULT_DOCS_PATH,
        help="docs/STEPFUN.md checklist source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON output to this path instead of stdout.",
    )
    parser.add_argument(
        "--verify-manifest",
        type=Path,
        default=None,
        help="Compare a persisted final-blocker manifest with the current inputs.",
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-manifest, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-manifest, emit only verification_failures.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the manifest.",
    )
    parser.add_argument(
        "--entries-only",
        action="store_true",
        help="Emit only the manifest entries for compact blocker polling.",
    )
    parser.add_argument(
        "--entries-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of manifest entries.",
    )
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Emit only artifacts_to_collect for compact evidence polling.",
    )
    parser.add_argument(
        "--artifacts-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of artifacts_to_collect.",
    )
    parser.add_argument(
        "--artifact-status-only",
        action="store_true",
        help="Emit only compact satisfaction status for required artifacts.",
    )
    parser.add_argument(
        "--artifact-status-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of artifact status.",
    )
    parser.add_argument(
        "--missing-artifacts-only",
        action="store_true",
        help="Emit only required artifacts whose evidence is not satisfied.",
    )
    parser.add_argument(
        "--missing-artifacts-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of missing artifact status.",
    )
    parser.add_argument(
        "--validator-commands-only",
        action="store_true",
        help="Emit only validator commands for required evidence artifacts.",
    )
    parser.add_argument(
        "--validator-commands-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of validator commands.",
    )
    parser.add_argument(
        "--success-criteria-only",
        action="store_true",
        help="Emit only the compact success criteria for each remaining blocker.",
    )
    parser.add_argument(
        "--success-criteria-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the success-criteria handoff.",
    )
    parser.add_argument(
        "--no-claim-policy-only",
        action="store_true",
        help="Emit only the no-claim policy for compact claim-gate polling.",
    )
    parser.add_argument(
        "--no-claim-policy-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the no-claim policy.",
    )
    parser.add_argument(
        "--gate-status-only",
        action="store_true",
        help="Emit only compact readiness-gate status for remaining blockers.",
    )
    parser.add_argument(
        "--gate-status-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the gate-status handoff.",
    )
    parser.add_argument(
        "--status-provenance-only",
        action="store_true",
        help="Emit only status/source provenance for compact manifest drift polling.",
    )
    parser.add_argument(
        "--status-provenance-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of status/source provenance.",
    )
    parser.add_argument(
        "--recommended-commands-only",
        action="store_true",
        help="Emit only exact recommended commands for the remaining blockers.",
    )
    parser.add_argument(
        "--recommended-commands-sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of recommended commands.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return parser.parse_args(argv)


def build_final_blocker_manifest(status: dict[str, object]) -> dict[str, object]:
    """Return the compact evidence manifest for the two final P11 blockers."""

    remaining = dict(status.get("remaining_blockers_report", {}))
    remaining_items = [
        item for item in remaining.get("items", []) if isinstance(item, dict)
    ]
    readiness_gates = dict(status.get("readiness_gates", {}))
    source_artifacts = dict(status.get("source_artifacts", {}))
    oracle_source = dict(source_artifacts.get("oracle", {}))
    prompt_source = dict(source_artifacts.get("prompt", {}))
    text_resource_source = dict(source_artifacts.get("text_resource", {}))
    oracle_progress = dict(status.get("oracle_progress", {}))
    oracle_gap_report = dict(status.get("oracle_gap_report", {}))
    oracle_partial_handoff = dict(status.get("oracle_partial_output_handoff", {}))
    kv_gap_report = dict(status.get("kv_backed_decode_gap_report", {}))
    kv_blocker_summary = dict(kv_gap_report.get("kv_decode_blocker_summary", {}))
    kv_launch_trace_summary = dict(
        kv_gap_report.get("streaming_decode_launch_trace_summary", {})
    )
    status_provenance = {
        "source_artifacts_sha256": status.get("source_artifacts_sha256"),
        "status_integrity_sha256": status.get("status_integrity_sha256"),
        "handoff_summary_sha256": status.get("handoff_summary_sha256"),
        "readiness_summary_sha256": status.get("readiness_summary_sha256"),
        "next_action_commands_sha256": status.get("next_action_commands_sha256"),
        "oracle_progress_sha256": status.get("oracle_progress_sha256"),
        "oracle_partial_output_handoff_sha256": status.get(
            "oracle_partial_output_handoff_sha256"
        ),
        "source_artifacts": source_artifacts,
    }

    entries: list[dict[str, object]] = []
    artifacts_to_collect: list[dict[str, object]] = []
    for item in remaining_items:
        blocker_kind = str(item.get("blocker_kind"))
        gate_name = item.get("readiness_gate")
        gate = readiness_gates.get(gate_name, {}) if isinstance(gate_name, str) else {}
        gate_record = gate if isinstance(gate, dict) else {}
        missing_evidence = list(item.get("missing_evidence", []))
        entry: dict[str, object] = {
            "blocker_kind": blocker_kind,
            "checklist_item": item.get("checklist_item"),
            "queue_index": item.get("queue_index"),
            "readiness_gate": gate_name,
            "gate_ready": item.get("gate_ready") is True,
            "gate_blocked_by": item.get("gate_blocked_by"),
            "required_evidence": gate_record.get("required_evidence"),
            "first_missing_evidence": item.get("first_missing_evidence")
            or (missing_evidence[0] if missing_evidence else None),
            "missing_evidence": missing_evidence,
            "success_criteria": list(item.get("success_criteria", [])),
            "recommended_command_kind": item.get("recommended_command_kind"),
            "recommended_command_sha256": item.get("recommended_command_sha256"),
        }
        if blocker_kind == "oracle_parity_blocked":
            artifact = {
                "name": "llama_cpp_oracle_success_artifact",
                "required_for": blocker_kind,
                "readiness_gate": "oracle_parity",
                "path": oracle_source.get("path"),
                "source_sha256": oracle_source.get("sha256"),
                "current_status": oracle_progress.get("status"),
                "current_returncode": oracle_progress.get("returncode"),
                "current_blocker_kind": oracle_progress.get("oracle_blocker_kind"),
                "expected_next_token_id": oracle_progress.get("expected_next_token_id"),
                "expected_next_token_text": oracle_progress.get("expected_next_token_text"),
                "first_missing_evidence": oracle_gap_report.get(
                    "first_missing_evidence"
                ),
                "recommended_command_kind": item.get("recommended_command_kind"),
                "recommended_command_sha256": item.get("recommended_command_sha256"),
                "partial_output_handoff_safe": oracle_partial_handoff.get(
                    "all_partial_output_contracts_safe"
                )
                is True,
                "partial_output_supervisor_signal_contract": oracle_partial_handoff.get(
                    "supervisor_signal_timeout_contract"
                ),
                "partial_output_supervisor_signal_handoff_safe": oracle_partial_handoff.get(
                    "supervisor_signal_contract_safe"
                )
                is True,
            }
            artifact.update(
                _oracle_validator_handoff(
                    prompt_source.get("path"),
                    oracle_source.get("path"),
                )
            )
            entry["oracle_helper_prerequisite_handoff"] = _oracle_helper_prerequisite_handoff()
            entry["artifact_handoff"] = artifact
            artifacts_to_collect.append(artifact)
        elif blocker_kind == "kv_backed_decode_not_wired":
            kv_required_artifacts = []
            for record in kv_blocker_summary.get("artifacts_needed", []):
                if not isinstance(record, dict):
                    continue
                enriched_record = dict(record)
                enriched_record.setdefault("readiness_gate", "kv_backed_decode")
                enriched_record.setdefault(
                    "recommended_command_kind", item.get("recommended_command_kind")
                )
                enriched_record.setdefault(
                    "recommended_command_sha256", item.get("recommended_command_sha256")
                )
                if enriched_record.get("name") == "kv_kernel_trace_artifact":
                    enriched_record.setdefault("path", str(DEFAULT_KV_TRACE_ARTIFACT))
                    enriched_record.update(
                        _kv_trace_validator_handoff(
                            text_resource_source.get("path"),
                            enriched_record.get("path"),
                        )
                    )
                elif enriched_record.get("name") == "kv_backed_next_token_artifact":
                    enriched_record.setdefault(
                        "path", str(DEFAULT_KV_NEXT_TOKEN_ARTIFACT)
                    )
                    enriched_record.update(
                        _kv_next_token_validator_handoff(
                            prompt_source.get("path"),
                            enriched_record.get("path"),
                        )
                    )
                kv_required_artifacts.append(enriched_record)
            artifact = {
                "name": "kv_backed_decode_runtime_artifacts",
                "required_for": blocker_kind,
                "readiness_gate": "kv_backed_decode",
                "first_streaming_runner_blocker": kv_gap_report.get(
                    "first_streaming_runner_blocker"
                ),
                "streaming_runner_blocker_names": list(
                    kv_gap_report.get("streaming_runner_blocker_names", [])
                ),
                "streaming_decode_launch_trace_sha256": kv_gap_report.get(
                    "streaming_decode_launch_trace_sha256"
                ),
                "streaming_decode_launch_trace_summary_sha256": kv_gap_report.get(
                    "streaming_decode_launch_trace_summary_sha256"
                ),
                "launch_trace_operation_count": kv_launch_trace_summary.get(
                    "operation_count"
                ),
                "launch_trace_non_executable": kv_launch_trace_summary.get(
                    "non_executable"
                )
                is True,
                "required_artifacts": kv_required_artifacts,
                "required_artifact_names": [
                    str(record.get("name")) for record in kv_required_artifacts
                ],
                "required_artifacts_sha256": kv_blocker_summary.get(
                    "artifacts_needed_sha256"
                ),
            }
            entry["kv_runner_prerequisite_handoff"] = _kv_runner_prerequisite_handoff()
            entry["artifact_handoff"] = artifact
            artifacts_to_collect.extend(kv_required_artifacts)
        entries.append(entry)

    recommended_commands_handoff = [
        {
            "blocker_kind": str(item.get("blocker_kind")),
            "readiness_gate": item.get("readiness_gate"),
            "queue_index": item.get("queue_index"),
            "recommended_command_kind": item.get("recommended_command_kind"),
            "recommended_command_reason": item.get("recommended_command_reason"),
            "recommended_command": item.get("recommended_command"),
            "recommended_command_sha256": item.get("recommended_command_sha256"),
            "writes_partial_output_before_launch": item.get(
                "recommended_command_writes_partial_output_before_launch"
            ),
            "partial_output_path": item.get("partial_output_path"),
            "partial_output_status": item.get("partial_output_status"),
            "partial_output_overwrite_policy": item.get(
                "partial_output_overwrite_policy"
            ),
            "partial_output_supervisor_signal_handoff_safe": (
                oracle_partial_handoff.get("supervisor_signal_contract_safe") is True
                if item.get("blocker_kind") == "oracle_parity_blocked"
                else None
            ),
            "resource_artifact": item.get("resource_artifact"),
            "success_criteria": list(item.get("success_criteria", [])),
        }
        for item in remaining_items
    ]
    success_criteria_handoff = [
        {
            "blocker_kind": entry.get("blocker_kind"),
            "readiness_gate": entry.get("readiness_gate"),
            "gate_ready": entry.get("gate_ready"),
            "first_missing_evidence": entry.get("first_missing_evidence"),
            "success_criteria": list(entry.get("success_criteria", [])),
            "recommended_command_kind": entry.get("recommended_command_kind"),
            "recommended_command_sha256": entry.get("recommended_command_sha256"),
        }
        for entry in entries
    ]
    artifact_status_handoff = _summarize_required_artifacts(artifacts_to_collect)
    missing_artifacts_handoff = [
        record for record in artifact_status_handoff if record.get("missing") is True
    ]
    validator_commands_handoff = _summarize_validator_commands(
        artifact_status_handoff
    )
    entry_by_gate = {
        str(entry.get("readiness_gate")): entry
        for entry in entries
        if entry.get("readiness_gate") is not None
    }
    gate_status_handoff = []
    for gate_name in remaining.get("blocked_gates", []):
        gate_record = readiness_gates.get(gate_name, {})
        gate = gate_record if isinstance(gate_record, dict) else {}
        entry = entry_by_gate.get(str(gate_name), {})
        gate_status_handoff.append(
            {
                "readiness_gate": gate_name,
                "ready": gate.get("ready"),
                "blocked_by": gate.get("blocked_by"),
                "required_evidence": gate.get("required_evidence"),
                "blocker_kind": entry.get("blocker_kind"),
                "first_missing_evidence": entry.get("first_missing_evidence"),
                "success_criteria": list(entry.get("success_criteria", [])),
            }
        )
    no_claim_policy = dict(remaining.get("no_claim_policy", {}))
    entries_sha256 = status_mod._stable_json_sha256(entries)
    artifacts_to_collect_sha256 = status_mod._stable_json_sha256(artifacts_to_collect)
    artifact_status_handoff_sha256 = status_mod._stable_json_sha256(
        artifact_status_handoff
    )
    missing_artifacts_handoff_sha256 = status_mod._stable_json_sha256(
        missing_artifacts_handoff
    )
    validator_commands_handoff_sha256 = status_mod._stable_json_sha256(
        validator_commands_handoff
    )
    success_criteria_handoff_sha256 = status_mod._stable_json_sha256(
        success_criteria_handoff
    )
    no_claim_policy_sha256 = status_mod._stable_json_sha256(no_claim_policy)
    gate_status_handoff_sha256 = status_mod._stable_json_sha256(gate_status_handoff)
    status_provenance_sha256 = status_mod._stable_json_sha256(status_provenance)
    recommended_commands_handoff_sha256 = status_mod._stable_json_sha256(
        recommended_commands_handoff
    )
    compact_output_modes = {
        "sha_only": "manifest_sha256",
        "entries_only": "entries",
        "entries_sha_only": "entries_sha256",
        "artifacts_only": "artifacts_to_collect",
        "artifacts_sha_only": "artifacts_to_collect_sha256",
        "artifact_status_only": "artifact_status_handoff",
        "artifact_status_sha_only": "artifact_status_handoff_sha256",
        "missing_artifacts_only": "missing_artifacts_handoff",
        "missing_artifacts_sha_only": "missing_artifacts_handoff_sha256",
        "validator_commands_only": "validator_commands_handoff",
        "validator_commands_sha_only": "validator_commands_handoff_sha256",
        "success_criteria_only": "success_criteria_handoff",
        "success_criteria_sha_only": "success_criteria_handoff_sha256",
        "no_claim_policy_only": "no_claim_policy",
        "no_claim_policy_sha_only": "no_claim_policy_sha256",
        "gate_status_only": "gate_status_handoff",
        "gate_status_sha_only": "gate_status_handoff_sha256",
        "status_provenance_only": "status_provenance",
        "status_provenance_sha_only": "status_provenance_sha256",
        "recommended_commands_only": "recommended_commands_handoff",
        "recommended_commands_sha_only": "recommended_commands_handoff_sha256",
        "verification_status_only": "verification.status",
        "verification_failures_only": "verification.verification_failures",
    }

    return {
        "schema_version": 1,
        "status": remaining.get("status", "blocked"),
        "status_provenance": status_provenance,
        "status_provenance_sha256": status_provenance_sha256,
        "open_or_partial_items_p0_p12": remaining.get(
            "open_or_partial_items_p0_p12"
        ),
        "remaining_blocker_count": remaining.get("remaining_blocker_count"),
        "remaining_blocker_kinds": list(remaining.get("remaining_blocker_kinds", [])),
        "blocked_gates": list(remaining.get("blocked_gates", [])),
        "entries": entries,
        "entries_sha256": entries_sha256,
        "artifacts_to_collect": artifacts_to_collect,
        "artifacts_to_collect_sha256": artifacts_to_collect_sha256,
        "artifact_status_handoff": artifact_status_handoff,
        "artifact_status_handoff_sha256": artifact_status_handoff_sha256,
        "missing_artifacts_handoff": missing_artifacts_handoff,
        "missing_artifacts_handoff_sha256": missing_artifacts_handoff_sha256,
        "validator_commands_handoff": validator_commands_handoff,
        "validator_commands_handoff_sha256": validator_commands_handoff_sha256,
        "all_required_artifacts_satisfied": not missing_artifacts_handoff,
        "missing_artifact_count": len(missing_artifacts_handoff),
        "recommended_commands_handoff": recommended_commands_handoff,
        "recommended_commands_handoff_sha256": recommended_commands_handoff_sha256,
        "success_criteria_handoff": success_criteria_handoff,
        "success_criteria_handoff_sha256": success_criteria_handoff_sha256,
        "gate_status_handoff": gate_status_handoff,
        "gate_status_handoff_sha256": gate_status_handoff_sha256,
        "compact_output_modes": compact_output_modes,
        "artifact_count": len(artifacts_to_collect),
        "entry_count": len(entries),
        "all_entries_have_success_criteria": all(
            bool(entry.get("success_criteria")) for entry in entries
        ),
        "all_entries_have_recommended_commands": all(
            bool(entry.get("recommended_command_sha256")) for entry in entries
        ),
        "no_claim_policy": no_claim_policy,
        "no_claim_policy_sha256": no_claim_policy_sha256,
    }


def verify_final_blocker_manifest(
    manifest_path: Path,
    *,
    current_manifest: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted final-blocker manifest with the current one."""

    persisted = json.loads(manifest_path.read_text())
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_manifest)
    failures: list[dict[str, object]] = []
    if persisted != current_manifest:
        failures.append(
            {
                "name": "final_blocker_manifest_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": "Persisted final-blocker manifest differs from current prompt/oracle/resource/docs inputs.",
            }
        )
    all_match = not failures
    return {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_manifest_sha256": persisted_sha256,
        "current_manifest_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failure_count": len(failures),
        "current_status_provenance": current_manifest.get("status_provenance"),
        "persisted_status_provenance": persisted.get("status_provenance")
        if isinstance(persisted, dict)
        else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    status = status_mod.build_status(
        args.prompt_artifact,
        args.oracle_artifact,
        args.docs,
        resource_artifact=args.resource_artifact,
    )
    manifest = build_final_blocker_manifest(status)
    if args.verify_manifest is not None:
        verification = verify_final_blocker_manifest(
            args.verify_manifest,
            current_manifest=manifest,
        )
        if args.verification_status_only:
            payload: object = verification["status"]
        elif args.verification_failures_only:
            payload = verification["verification_failures"]
        else:
            payload = verification
        status_mod._emit_json(payload, pretty=args.pretty, output=args.output)
        return (
            status_mod.READY_EXIT_CODE
            if verification["all_match"] is True
            else status_mod.SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
        )
    if args.entries_sha_only:
        payload = manifest["entries_sha256"]
    elif args.entries_only:
        payload = manifest["entries"]
    elif args.artifacts_sha_only:
        payload = manifest["artifacts_to_collect_sha256"]
    elif args.artifacts_only:
        payload = manifest["artifacts_to_collect"]
    elif args.artifact_status_sha_only:
        payload = manifest["artifact_status_handoff_sha256"]
    elif args.artifact_status_only:
        payload = manifest["artifact_status_handoff"]
    elif args.missing_artifacts_sha_only:
        payload = manifest["missing_artifacts_handoff_sha256"]
    elif args.missing_artifacts_only:
        payload = manifest["missing_artifacts_handoff"]
    elif args.validator_commands_sha_only:
        payload = manifest["validator_commands_handoff_sha256"]
    elif args.validator_commands_only:
        payload = manifest["validator_commands_handoff"]
    elif args.success_criteria_sha_only:
        payload = manifest["success_criteria_handoff_sha256"]
    elif args.success_criteria_only:
        payload = manifest["success_criteria_handoff"]
    elif args.no_claim_policy_sha_only:
        payload = manifest["no_claim_policy_sha256"]
    elif args.no_claim_policy_only:
        payload = manifest["no_claim_policy"]
    elif args.gate_status_sha_only:
        payload = manifest["gate_status_handoff_sha256"]
    elif args.gate_status_only:
        payload = manifest["gate_status_handoff"]
    elif args.status_provenance_sha_only:
        payload = manifest["status_provenance_sha256"]
    elif args.status_provenance_only:
        payload = manifest["status_provenance"]
    elif args.recommended_commands_sha_only:
        payload = manifest["recommended_commands_handoff_sha256"]
    elif args.recommended_commands_only:
        payload = manifest["recommended_commands_handoff"]
    else:
        payload = status_mod._stable_json_sha256(manifest) if args.sha_only else manifest
    status_mod._emit_json(payload, pretty=args.pretty, output=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
