#!/usr/bin/env python3
"""Emit a StepFun llama.cpp logits-probe availability preflight artifact.

The preflight records whether the retained oracle next-action inputs are ready for
a same-prompt llama.cpp logits comparison and whether the required llama.cpp
logits artifact/entrypoint exists. It does not execute llama.cpp and makes no
oracle, KV, e2e, or performance claim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stepfun_correctness_status as status_mod
from scripts import stepfun_host_logit_margin as host_margin
from scripts import stepfun_llamacpp_oracle as llama_oracle
from scripts import stepfun_oracle_next_action_manifest as next_action_mod
from scripts import stepfun_oracle_source_map as source_map_mod

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-preflight.json"
)
DEFAULT_LLAMA_LOGITS_ARTIFACT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-probe.json"
)
DEFAULT_LLAMA_CLI = Path(
    "/home/lhl/llama.cpp/llama.cpp-vulkan/build-vulkan-release/bin/llama-debug"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"
LOGITS_DUMP_HELP_MARKERS = (
    "--save-logits",
    "--logits-output-dir",
    "--logits-file",
    "--dump-logits",
    "--logprobs",
    "--top-logprobs",
)
SAME_PROMPT_SPECIAL_HELP_MARKERS = ("--special", "--parse-special")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--next-action-manifest",
        type=Path,
        default=next_action_mod.DEFAULT_OUTPUT,
        help="Retained oracle next-action manifest artifact.",
    )
    parser.add_argument(
        "--source-map-artifact",
        type=Path,
        default=source_map_mod.DEFAULT_OUTPUT,
        help="Retained oracle source-map artifact.",
    )
    parser.add_argument(
        "--host-logit-margin-artifact",
        type=Path,
        default=host_margin.DEFAULT_OUTPUT,
        help="Retained host top-logit margin artifact.",
    )
    parser.add_argument(
        "--llama-logits-artifact",
        type=Path,
        default=DEFAULT_LLAMA_LOGITS_ARTIFACT,
        help="Expected retained llama.cpp same-prompt logits artifact.",
    )
    parser.add_argument(
        "--llama-cli",
        type=Path,
        default=DEFAULT_LLAMA_CLI,
        help="llama.cpp binary candidate to inspect for logits-probe capability.",
    )
    parser.add_argument(
        "--help-timeout-s",
        type=float,
        default=10.0,
        help="Timeout for inspecting the llama.cpp probe binary --help.",
    )
    parser.add_argument(
        "--artifact-date",
        default=DEFAULT_ARTIFACT_DATE,
        help="Date string to record in the artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write JSON output atomically to this path instead of stdout. Use "
            "--default-output for the canonical StepFun artifact path."
        ),
    )
    parser.add_argument(
        "--default-output",
        action="store_true",
        help=f"Write to the canonical artifact path: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--status-only", action="store_true", help="Emit only status.")
    parser.add_argument(
        "--missing-evidence-only",
        action="store_true",
        help="Emit only missing evidence keys.",
    )
    parser.add_argument(
        "--logits-artifact-present-only",
        action="store_true",
        help="Emit only whether the expected llama.cpp logits artifact exists.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the artifact payload.",
    )
    parser.add_argument(
        "--verify-preflight",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help=(
            "Compare a persisted llama.cpp logits preflight artifact with current "
            f"source/input/logits-probe metadata. If no path is supplied, uses {DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-preflight, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-preflight, emit only verification failures.",
    )
    parser.add_argument(
        "--verification-sha-only",
        action="store_true",
        help="With --verify-preflight, emit only the stable verification digest.",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _artifact_ref(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None}
    payload = json.loads(path.read_text())
    return {
        "path": str(path),
        "exists": True,
        "artifact_kind": payload.get("artifact_kind") if isinstance(payload, dict) else None,
        "status": payload.get("status") if isinstance(payload, dict) else None,
        "ready": payload.get("ready") if isinstance(payload, dict) else None,
        "same_prompt_tokens_match": payload.get("same_prompt_tokens_match") if isinstance(payload, dict) else None,
        "sha256": status_mod._stable_json_sha256(payload),
    }


def _is_executable(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return False
    return bool(mode & stat.S_IXUSR) or os.access(path, os.X_OK)


def _marker_in_help(text: str, marker: str) -> bool:
    return re.search(rf"(?<!\S){re.escape(marker)}(?![\w-])", text) is not None


def _help_info(llama_cli: Path, timeout_s: float) -> dict[str, object]:
    info: dict[str, object] = {
        "path": str(llama_cli),
        "exists": llama_cli.exists(),
        "executable": _is_executable(llama_cli),
        "help_status": "not_run",
        "help_returncode": None,
        "help_markers_checked": list(LOGITS_DUMP_HELP_MARKERS),
        "obvious_logits_dump_flag_present": False,
        "matched_logits_dump_markers": [],
        "same_prompt_special_markers_checked": list(SAME_PROMPT_SPECIAL_HELP_MARKERS),
        "same_prompt_special_token_flag_present": False,
        "matched_same_prompt_special_markers": [],
        "help_mentions_logit_bias": False,
    }
    if not info["exists"] or not info["executable"]:
        info["help_status"] = "unavailable"
        return info
    try:
        completed = subprocess.run(
            [str(llama_cli), "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        info["help_status"] = "timeout"
        return info
    except OSError as exc:
        info["help_status"] = f"error:{type(exc).__name__}"
        info["help_error"] = str(exc)
        return info
    text = (completed.stdout or "") + (completed.stderr or "")
    matched = [marker for marker in LOGITS_DUMP_HELP_MARKERS if _marker_in_help(text, marker)]
    same_prompt_matched = [
        marker for marker in SAME_PROMPT_SPECIAL_HELP_MARKERS if _marker_in_help(text, marker)
    ]
    info.update(
        {
            "help_status": "executed",
            "help_returncode": completed.returncode,
            "obvious_logits_dump_flag_present": bool(matched),
            "matched_logits_dump_markers": matched,
            "same_prompt_special_token_flag_present": bool(same_prompt_matched),
            "matched_same_prompt_special_markers": same_prompt_matched,
            "help_mentions_logit_bias": "--logit-bias" in text,
        }
    )
    return info


def build_llamacpp_logits_preflight(
    *,
    next_action_manifest: Path = next_action_mod.DEFAULT_OUTPUT,
    source_map_artifact: Path = source_map_mod.DEFAULT_OUTPUT,
    host_logit_margin_artifact: Path = host_margin.DEFAULT_OUTPUT,
    llama_logits_artifact: Path = DEFAULT_LLAMA_LOGITS_ARTIFACT,
    llama_cli: Path = DEFAULT_LLAMA_CLI,
    help_timeout_s: float = 10.0,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the llama.cpp logits-probe availability preflight."""

    manifest = _load_json_object(next_action_manifest)
    source_map = _load_json_object(source_map_artifact)
    margin = _load_json_object(host_logit_margin_artifact)
    help_record = _help_info(llama_cli, help_timeout_s)
    logits_artifact = _artifact_ref(llama_logits_artifact)
    target = manifest.get("target") if isinstance(manifest.get("target"), dict) else {}
    prerequisites = [
        {
            "name": "next_action_manifest_investigation_ready",
            "passed": manifest.get("investigation_ready") is True,
            "expected": True,
            "actual": manifest.get("investigation_ready"),
        },
        {
            "name": "source_map_mapped",
            "passed": source_map.get("status") == "mapped" and source_map.get("ready") is True,
            "expected": {"status": "mapped", "ready": True},
            "actual": {"status": source_map.get("status"), "ready": source_map.get("ready")},
        },
        {
            "name": "host_logit_margin_passed",
            "passed": margin.get("status") == "passed",
            "expected": "passed",
            "actual": margin.get("status"),
        },
        {
            "name": "llama_cli_available",
            "passed": help_record.get("exists") is True and help_record.get("executable") is True,
            "expected": {"exists": True, "executable": True},
            "actual": {"exists": help_record.get("exists"), "executable": help_record.get("executable")},
        },
    ]
    missing_evidence: list[str] = []
    logits_artifact_captured = (
        logits_artifact.get("exists") is True
        and logits_artifact.get("status") == "captured"
        and logits_artifact.get("ready") is True
        and logits_artifact.get("same_prompt_tokens_match") is True
    )
    if not logits_artifact_captured:
        missing_evidence.append("llama_cpp_same_prompt_logits_artifact_present")
    if help_record.get("obvious_logits_dump_flag_present") is not True:
        missing_evidence.append("llama_cpp_logits_dump_entrypoint_identified")
    if help_record.get("same_prompt_special_token_flag_present") is not True:
        missing_evidence.append("llama_cpp_same_prompt_special_token_support_present")
    for prereq in prerequisites:
        if prereq.get("passed") is not True:
            missing_evidence.append(str(prereq["name"]))
    ready = not missing_evidence
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_llamacpp_logits_preflight",
        "date": artifact_date,
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "oracle_parity_ready": False,
        "next_action_kind": "llamacpp_same_prompt_logits_probe",
        "expected_llama_logits_artifact": logits_artifact,
        "llama_cli_probe": help_record,
        "prerequisites": prerequisites,
        "missing_evidence": missing_evidence,
        "target": {
            "canonical_backend": target.get("canonical_backend", "vulkan"),
            "readiness_gate": "oracle_parity",
            "unresolved_evidence_gap": "generated_text_matches_target",
            "expected_next_token_id": target.get("expected_next_token_id"),
            "expected_next_token_text": target.get("expected_next_token_text"),
            "generated_first_token_id": target.get("generated_first_token_id"),
            "generated_text": target.get("generated_text"),
            "host_top_token_ids": target.get("host_top_token_ids"),
            "host_top1_to_top2_margin": target.get("host_top1_to_top2_margin"),
        },
        "required_retained_inputs": [
            _artifact_ref(next_action_manifest),
            _artifact_ref(source_map_artifact),
            _artifact_ref(host_logit_margin_artifact),
            _artifact_ref(status_mod.DEFAULT_PROMPT_ARTIFACT),
            _artifact_ref(status_mod.DEFAULT_ORACLE_ARTIFACT),
        ],
        "candidate_probe_commands": [
            (
                "python3 scripts/stepfun_llamacpp_oracle.py --artifact "
                f"{status_mod.DEFAULT_PROMPT_ARTIFACT} --llama-cli {llama_cli} "
                f"--model {llama_oracle.DEFAULT_MODEL} --n-predict 1 --timeout-s 900 --pretty"
            ),
            (
                "after a concrete logits dump entrypoint is identified, retain same-prompt "
                f"llama.cpp logits as {llama_logits_artifact} and compare token 369 vs 671"
            ),
            (
                "python3 scripts/stepfun_llamacpp_logits_probe.py --prompt-token-source retained-input-ids "
                "--execute --default-output --pretty captures the compact logits artifact only after "
                "the probe binary can parse special tokens or accept retained token IDs"
            ),
        ],
        "blocked_reason": (
            "same-prompt llama.cpp logits artifact is missing"
            if missing_evidence == ["llama_cpp_same_prompt_logits_artifact_present"]
            else "same-prompt logits probe binary lacks special-token parsing for the retained prompt"
            if "llama_cpp_same_prompt_special_token_support_present" in missing_evidence
            else "same-prompt llama.cpp logits artifact and/or logits dump entrypoint is missing"
            if not ready
            else "same-prompt llama.cpp logits preflight is ready"
        ),
        "next_action": (
            "retain same-prompt logits from llama-debug using --save-logits, then compare expected "
            "token 369 with generated token 671"
            if help_record.get("obvious_logits_dump_flag_present") is True
            and help_record.get("same_prompt_special_token_flag_present") is True
            else (
                "add or build a llama.cpp logits dump helper that parses special tokens or accepts explicit "
                "token IDs for the retained StepFun prompt, then retain same-prompt logits before comparing "
                "expected token 369 with generated token 671"
            )
            if help_record.get("obvious_logits_dump_flag_present") is True
            else "identify or add a llama.cpp logits dump entrypoint for the retained StepFun prompt, "
            "then retain same-prompt logits before comparing expected token 369 with generated token 671"
        ),
        "blocked_gates": ["oracle_parity", "kv_backed_decode", "e2e_inference"],
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This preflight only records availability of same-prompt llama.cpp logits evidence; "
                "generated_text_matches_target remains unresolved."
            ),
        },
    }


def verify_llamacpp_logits_preflight(
    preflight_artifact: Path,
    *,
    current_report: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted llama.cpp logits preflight artifact with current metadata."""

    persisted = _load_json_object(preflight_artifact)
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_report)
    failures: list[dict[str, object]] = []
    if persisted != current_report:
        failures.append(
            {
                "name": "llamacpp_logits_preflight_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted llama.cpp logits preflight artifact differs from current "
                    "next-action/source-map/host-margin/CLI/logits-artifact metadata."
                ),
            }
        )
    all_match = not failures
    persisted_logit_artifact = persisted.get("expected_llama_logits_artifact")
    current_logit_artifact = current_report.get("expected_llama_logits_artifact")
    persisted_cli = persisted.get("llama_cli_probe")
    current_cli = current_report.get("llama_cli_probe")
    return {
        "schema_version": 1,
        "artifact_path": str(preflight_artifact),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_artifact_sha256": persisted_sha256,
        "current_artifact_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "verification_failure_count": len(failures),
        "persisted_status": persisted.get("status"),
        "current_status": current_report.get("status"),
        "persisted_ready": persisted.get("ready"),
        "current_ready": current_report.get("ready"),
        "persisted_missing_evidence": persisted.get("missing_evidence"),
        "current_missing_evidence": current_report.get("missing_evidence"),
        "persisted_logits_artifact_exists": (
            persisted_logit_artifact.get("exists")
            if isinstance(persisted_logit_artifact, dict)
            else None
        ),
        "current_logits_artifact_exists": (
            current_logit_artifact.get("exists")
            if isinstance(current_logit_artifact, dict)
            else None
        ),
        "persisted_cli_help_status": (
            persisted_cli.get("help_status") if isinstance(persisted_cli, dict) else None
        ),
        "current_cli_help_status": (
            current_cli.get("help_status") if isinstance(current_cli, dict) else None
        ),
        "persisted_cli_logits_dump_flag_present": (
            persisted_cli.get("obvious_logits_dump_flag_present")
            if isinstance(persisted_cli, dict)
            else None
        ),
        "current_cli_logits_dump_flag_present": (
            current_cli.get("obvious_logits_dump_flag_present")
            if isinstance(current_cli, dict)
            else None
        ),
        "persisted_cli_special_token_flag_present": (
            persisted_cli.get("same_prompt_special_token_flag_present")
            if isinstance(persisted_cli, dict)
            else None
        ),
        "current_cli_special_token_flag_present": (
            current_cli.get("same_prompt_special_token_flag_present")
            if isinstance(current_cli, dict)
            else None
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    report = build_llamacpp_logits_preflight(
        next_action_manifest=args.next_action_manifest,
        source_map_artifact=args.source_map_artifact,
        host_logit_margin_artifact=args.host_logit_margin_artifact,
        llama_logits_artifact=args.llama_logits_artifact,
        llama_cli=args.llama_cli,
        help_timeout_s=args.help_timeout_s,
        artifact_date=args.artifact_date,
    )
    if args.verify_preflight is not None:
        verification = verify_llamacpp_logits_preflight(
            args.verify_preflight,
            current_report=report,
        )
        if args.verification_status_only:
            payload: object = verification["status"]
        elif args.verification_failures_only:
            payload = verification["verification_failures"]
        elif args.verification_sha_only:
            payload = status_mod._stable_json_sha256(verification)
        else:
            payload = verification
        output = DEFAULT_OUTPUT if args.default_output else args.output
        status_mod._emit_json(payload, pretty=args.pretty, output=output)
        return (
            0
            if verification["all_match"] is True
            else status_mod.SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
        )
    if args.status_only:
        payload: object = report["status"]
    elif args.missing_evidence_only:
        payload = report["missing_evidence"]
    elif args.logits_artifact_present_only:
        payload = report["expected_llama_logits_artifact"]["exists"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(report)
    else:
        payload = report
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
