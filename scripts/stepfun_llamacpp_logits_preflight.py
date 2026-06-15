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
    "/home/lhl/llama.cpp/llama.cpp-vulkan/build-vulkan-release/bin/llama-cli"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"
LOGITS_DUMP_HELP_MARKERS = (
    "--logits",
    "--logits-file",
    "--dump-logits",
    "--logprobs",
    "--top-logprobs",
)


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
        help="llama.cpp CLI candidate to inspect for logits-probe capability.",
    )
    parser.add_argument(
        "--help-timeout-s",
        type=float,
        default=10.0,
        help="Timeout for inspecting llama-cli --help.",
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
        "sha256": status_mod._stable_json_sha256(payload),
    }


def _is_executable(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return False
    return bool(mode & stat.S_IXUSR) or os.access(path, os.X_OK)


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
    matched = [marker for marker in LOGITS_DUMP_HELP_MARKERS if marker in text]
    info.update(
        {
            "help_status": "executed",
            "help_returncode": completed.returncode,
            "obvious_logits_dump_flag_present": bool(matched),
            "matched_logits_dump_markers": matched,
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
    if logits_artifact.get("exists") is not True:
        missing_evidence.append("llama_cpp_same_prompt_logits_artifact_present")
    if help_record.get("obvious_logits_dump_flag_present") is not True:
        missing_evidence.append("llama_cpp_logits_dump_entrypoint_identified")
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
        ],
        "blocked_reason": (
            "same-prompt llama.cpp logits artifact and/or logits dump entrypoint is missing"
            if not ready
            else "same-prompt llama.cpp logits preflight is ready"
        ),
        "next_action": (
            "identify or add a llama.cpp logits dump entrypoint for the retained StepFun prompt, "
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
