#!/usr/bin/env python3
"""Emit a compact StepFun remaining-blockers rollup artifact.

The rollup links the two remaining P0-P12 blockers to their reproducible helper
commands and retained artifacts: oracle parity (backend matrix) and KV-backed
decode (KV blocker status). It is handoff evidence only and does not claim oracle
parity, KV-backed decode readiness, e2e readiness, or performance.
"""

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
from scripts import stepfun_kv_blocker_status as kv_status
from scripts import stepfun_kv_session_contract as kv_session_contract
from scripts import stepfun_llamacpp_logits_helper_readiness as helper_readiness
from scripts import stepfun_oracle_backend_matrix as backend_matrix

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-remaining-blockers-rollup.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-artifact",
        type=Path,
        default=status_mod.DEFAULT_PROMPT_ARTIFACT,
        help="Canonical prompt/logit artifact used by all blocker checks.",
    )
    parser.add_argument(
        "--oracle-artifact",
        type=Path,
        default=status_mod.DEFAULT_ORACLE_ARTIFACT,
        help="Canonical Vulkan llama.cpp oracle artifact.",
    )
    parser.add_argument(
        "--hip-artifact",
        type=Path,
        default=backend_matrix.DEFAULT_HIP_ARTIFACT,
        help="HIP llama.cpp oracle timeout artifact.",
    )
    parser.add_argument(
        "--resource-artifact",
        type=Path,
        default=status_mod.DEFAULT_RESOURCE_ARTIFACT,
        help="StepFun text-resource dry-run artifact used by the KV blocker.",
    )
    parser.add_argument(
        "--helper-readiness-artifact",
        type=Path,
        default=helper_readiness.DEFAULT_OUTPUT,
        help="Retained llama.cpp retained-token helper readiness artifact.",
    )
    parser.add_argument(
        "--kv-session-contract-artifact",
        type=Path,
        default=kv_session_contract.DEFAULT_OUTPUT,
        help="Guarded StepFun KV session-contract artifact.",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=status_mod.DEFAULT_DOCS_PATH,
        help="docs/STEPFUN.md checklist source.",
    )
    parser.add_argument(
        "--llama-tokenize",
        type=Path,
        default=backend_matrix.token_mismatch.DEFAULT_LLAMA_TOKENIZE,
        help="llama.cpp llama-tokenize binary for oracle token diagnostics.",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=backend_matrix.token_mismatch.DEFAULT_TOKENIZER_MODEL,
        help="GGUF model to pass to llama-tokenize.",
    )
    parser.add_argument(
        "--tokenizer-timeout-s",
        type=float,
        default=60.0,
        help="Per-text timeout for tokenizer diagnostics.",
    )
    parser.add_argument(
        "--artifact-date",
        default=DEFAULT_ARTIFACT_DATE,
        help="Date string to record in the rollup artifact.",
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
        "--open-count-only",
        action="store_true",
        help="Emit only P0-P12 open/partial checklist count.",
    )
    parser.add_argument(
        "--blocker-kinds-only",
        action="store_true",
        help="Emit only remaining blocker kinds.",
    )
    parser.add_argument(
        "--generator-commands-only",
        action="store_true",
        help="Emit only generator commands for retained blocker artifacts.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the rollup payload.",
    )
    parser.add_argument(
        "--verify-rollup",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help=(
            "Compare a persisted remaining-blockers rollup with current inputs. "
            f"If no path is supplied, uses {DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-rollup, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-rollup, emit only verification failures.",
    )
    parser.add_argument(
        "--verification-sha-only",
        action="store_true",
        help="With --verify-rollup, emit only the stable verification digest.",
    )
    return parser.parse_args(argv)


def _docs_open_partial_summary(status: dict[str, object]) -> dict[str, object]:
    docs = status.get("docs_checklist")
    docs_record = docs if isinstance(docs, dict) else {}
    items = docs_record.get("open_or_partial_items_p0_p12", [])
    if not isinstance(items, list):
        items = []
    return {
        "count": docs_record.get("open_or_partial_count_p0_p12"),
        "state_counts": docs_record.get("open_or_partial_state_counts_p0_p12"),
        "items": items,
        "first_item": items[0] if items else None,
        "last_item": items[-1] if items else None,
    }


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _readiness_summary(status: dict[str, object]) -> dict[str, object]:
    return {
        "status": status.get("status"),
        "oracle_parity": status.get("oracle_parity"),
        "kv_decode_dispatch_ready": status.get("kv_decode_dispatch_ready"),
        "kv_backed_decode_ready": status.get("kv_backed_decode_ready"),
        "e2e_inference_ready": status.get("e2e_inference_ready"),
        "blocked_gates": status.get("blocked_gates"),
        "blocker_kinds": status.get("blocker_kinds"),
    }


def _generator_commands() -> dict[str, str]:
    return {
        "oracle_backend_matrix": (
            "python3 scripts/stepfun_oracle_backend_matrix.py --default-output --pretty"
        ),
        "oracle_token_mismatch": (
            "python3 scripts/stepfun_oracle_token_mismatch.py --default-output --pretty"
        ),
        "oracle_rank_check": (
            "python3 scripts/stepfun_oracle_rank_check.py --default-output --pretty"
        ),
        "host_logit_margin": (
            "python3 scripts/stepfun_host_logit_margin.py --default-output --pretty"
        ),
        "top_token_roundtrip": (
            "python3 scripts/stepfun_top_token_roundtrip.py --default-output --pretty"
        ),
        "prompt_token_roundtrip": (
            "python3 scripts/stepfun_prompt_token_roundtrip.py --default-output --pretty"
        ),
        "oracle_blocker_diagnosis": (
            "python3 scripts/stepfun_oracle_blocker_diagnosis.py --default-output --pretty"
        ),
        "oracle_evidence_consistency_check": (
            "python3 scripts/stepfun_oracle_evidence_consistency_check.py --default-output --pretty"
        ),
        "oracle_next_action_manifest": (
            "python3 scripts/stepfun_oracle_next_action_manifest.py --default-output --pretty"
        ),
        "oracle_source_map": (
            "python3 scripts/stepfun_oracle_source_map.py --default-output --pretty"
        ),
        "llamacpp_logits_preflight": (
            "python3 scripts/stepfun_llamacpp_logits_preflight.py --default-output --pretty"
        ),
        "llamacpp_logits_probe": (
            "python3 scripts/stepfun_llamacpp_logits_probe.py --prompt-token-source retained-input-ids --execute --default-output --pretty"
        ),
        "llamacpp_logits_plan": (
            "python3 scripts/stepfun_llamacpp_logits_plan.py --default-output --pretty"
        ),
        "llamacpp_logits_entrypoint_inventory": (
            "python3 scripts/stepfun_llamacpp_logits_entrypoint_inventory.py --default-output --pretty"
        ),
        "llamacpp_logits_helper_contract": (
            "python3 scripts/stepfun_llamacpp_logits_helper_contract.py --default-output --pretty"
        ),
        "llamacpp_logits_helper_patch_plan": (
            "python3 scripts/stepfun_llamacpp_logits_helper_patch_plan.py --default-output --pretty"
        ),
        "llamacpp_logits_helper_patch_dry_run": (
            "python3 scripts/stepfun_llamacpp_logits_helper_patch_dry_run.py --default-output --pretty"
        ),
        "llamacpp_logits_helper_readiness": (
            "python3 scripts/stepfun_llamacpp_logits_helper_readiness.py --default-output --pretty"
        ),
        "kv_blocker_status": (
            "python3 scripts/stepfun_kv_blocker_status.py --default-output --pretty"
        ),
        "kv_session_contract": (
            "python3 scripts/stepfun_kv_session_contract.py --default-output --pretty"
        ),
        "kv_evidence_preflight": (
            "python3 scripts/stepfun_kv_evidence_preflight.py --default-output --pretty"
        ),
        "status_refresh": (
            "python3 scripts/stepfun_correctness_status.py --pretty "
            "--output benchmarks/results/2026-05-31-stepfun-q3kl-correctness-status.json"
        ),
        "final_blocker_refresh": (
            "python3 scripts/stepfun_final_blocker_manifest.py --pretty "
            "--output benchmarks/results/2026-05-31-stepfun-q3kl-final-blocker-manifest.json"
        ),
        "handoff_refresh": (
            "python3 scripts/stepfun_handoff_check.py --pretty "
            "--output benchmarks/results/2026-05-31-stepfun-q3kl-handoff-check.json"
        ),
    }


def build_remaining_blockers_rollup(
    *,
    prompt_artifact: Path = status_mod.DEFAULT_PROMPT_ARTIFACT,
    oracle_artifact: Path = status_mod.DEFAULT_ORACLE_ARTIFACT,
    hip_artifact: Path = backend_matrix.DEFAULT_HIP_ARTIFACT,
    resource_artifact: Path = status_mod.DEFAULT_RESOURCE_ARTIFACT,
    helper_readiness_artifact: Path = helper_readiness.DEFAULT_OUTPUT,
    kv_session_contract_artifact: Path = kv_session_contract.DEFAULT_OUTPUT,
    docs: Path = status_mod.DEFAULT_DOCS_PATH,
    llama_tokenize: Path = backend_matrix.token_mismatch.DEFAULT_LLAMA_TOKENIZE,
    tokenizer_model: Path = backend_matrix.token_mismatch.DEFAULT_TOKENIZER_MODEL,
    tokenizer_timeout_s: float = 60.0,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the consolidated remaining-blockers rollup."""

    status = status_mod.build_status(
        prompt_artifact,
        oracle_artifact,
        docs,
        resource_artifact=resource_artifact,
    )
    oracle_matrix = backend_matrix.build_oracle_backend_matrix(
        vulkan_artifact=oracle_artifact,
        hip_artifact=hip_artifact,
        prompt_artifact=prompt_artifact,
        llama_tokenize=llama_tokenize,
        tokenizer_model=tokenizer_model,
        tokenizer_timeout_s=tokenizer_timeout_s,
        artifact_date=artifact_date,
    )
    kv_blocker = kv_status.build_kv_blocker_status(
        prompt_artifact=prompt_artifact,
        oracle_artifact=oracle_artifact,
        docs=docs,
        resource_artifact=resource_artifact,
        artifact_date=artifact_date,
    )
    helper_readiness_payload = _load_json_object(helper_readiness_artifact)
    kv_session_payload = _load_json_object(kv_session_contract_artifact)
    kv_session_contract_record = kv_session_payload.get("contract")
    kv_session_contract_record = (
        kv_session_contract_record
        if isinstance(kv_session_contract_record, dict)
        else {}
    )
    kv_decode_entrypoint = kv_session_payload.get("decode_entrypoint_blocker")
    kv_decode_entrypoint = (
        kv_decode_entrypoint if isinstance(kv_decode_entrypoint, dict) else {}
    )
    docs_summary = _docs_open_partial_summary(status)
    readiness = _readiness_summary(status)
    commands = _generator_commands()
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_remaining_blockers_rollup",
        "date": artifact_date,
        "status": status.get("status"),
        "readiness": readiness,
        "p0_p12_open_partial": docs_summary,
        "remaining_blocker_count": docs_summary.get("count"),
        "remaining_blockers": [
            {
                "blocker_kind": "oracle_parity_blocked",
                "readiness_gate": "oracle_parity",
                "status": oracle_matrix.get("status"),
                "primary_artifact": str(backend_matrix.DEFAULT_OUTPUT),
                "generator_command_kind": "oracle_backend_matrix",
                "generator_command": commands["oracle_backend_matrix"],
                "backend_outcomes": oracle_matrix.get("backend_outcomes"),
                "canonical_backend": oracle_matrix.get("canonical_backend"),
                "oracle_parity_ready": oracle_matrix.get("oracle_parity_ready"),
                "rank_check_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-oracle-rank-check.json",
                "rank_check_generator_command": commands["oracle_rank_check"],
                "host_logit_margin_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-host-top-logit-margin.json",
                "host_logit_margin_generator_command": commands["host_logit_margin"],
                "top_token_roundtrip_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-host-top-token-roundtrip.json",
                "top_token_roundtrip_generator_command": commands["top_token_roundtrip"],
                "prompt_token_roundtrip_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-prompt-token-roundtrip.json",
                "prompt_token_roundtrip_generator_command": commands["prompt_token_roundtrip"],
                "diagnosis_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-oracle-blocker-diagnosis.json",
                "diagnosis_generator_command": commands["oracle_blocker_diagnosis"],
                "evidence_consistency_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-oracle-evidence-consistency-check.json",
                "evidence_consistency_generator_command": commands["oracle_evidence_consistency_check"],
                "next_action_manifest_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-oracle-next-action-manifest.json",
                "next_action_manifest_generator_command": commands["oracle_next_action_manifest"],
                "source_map_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-oracle-source-map.json",
                "source_map_generator_command": commands["oracle_source_map"],
                "llamacpp_logits_preflight_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-preflight.json",
                "llamacpp_logits_preflight_generator_command": commands["llamacpp_logits_preflight"],
                "llamacpp_logits_probe_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-probe.json",
                "llamacpp_logits_probe_generator_command": commands["llamacpp_logits_probe"],
                "llamacpp_logits_plan_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-plan.json",
                "llamacpp_logits_plan_generator_command": commands["llamacpp_logits_plan"],
                "llamacpp_logits_entrypoint_inventory_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-entrypoint-inventory.json",
                "llamacpp_logits_entrypoint_inventory_generator_command": commands["llamacpp_logits_entrypoint_inventory"],
                "llamacpp_logits_helper_contract_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-helper-contract.json",
                "llamacpp_logits_helper_contract_generator_command": commands["llamacpp_logits_helper_contract"],
                "llamacpp_logits_helper_patch_plan_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-helper-patch-plan.json",
                "llamacpp_logits_helper_patch_plan_generator_command": commands["llamacpp_logits_helper_patch_plan"],
                "llamacpp_logits_helper_patch_dry_run_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-helper-patch-dry-run.json",
                "llamacpp_logits_helper_patch_dry_run_generator_command": commands["llamacpp_logits_helper_patch_dry_run"],
                "llamacpp_logits_helper_patch_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-helper.patch",
                "llamacpp_logits_helper_patch_apply_command": (
                    "git -C /home/lhl/llama.cpp/llama.cpp-vulkan apply --unidiff-zero "
                    "/home/lhl/hipEngine-stepfun-3.7-flash/benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-helper.patch"
                ),
                "llamacpp_logits_helper_readiness_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-helper-readiness.json",
                "llamacpp_logits_helper_readiness_generator_command": commands["llamacpp_logits_helper_readiness"],
                "llamacpp_logits_helper_readiness_status": helper_readiness_payload.get("status"),
                "llamacpp_logits_helper_readiness_missing_evidence": helper_readiness_payload.get("missing_evidence"),
                "llamacpp_logits_helper_required_next_command_records": helper_readiness_payload.get("required_next_command_records"),
                "llamacpp_logits_helper_required_next_command_records_sha256": helper_readiness_payload.get("required_next_command_records_sha256"),
                "blocked_reason": oracle_matrix.get("blocked_reason"),
            },
            {
                "blocker_kind": "kv_backed_decode_not_wired",
                "readiness_gate": "kv_backed_decode",
                "status": kv_blocker.get("status"),
                "primary_artifact": str(kv_status.DEFAULT_OUTPUT),
                "generator_command_kind": "kv_blocker_status",
                "generator_command": commands["kv_blocker_status"],
                "kv_decode_dispatch_ready": kv_blocker.get("kv_decode_dispatch_ready"),
                "kv_backed_decode_ready": kv_blocker.get("kv_backed_decode_ready"),
                "blocked_count": kv_blocker.get("blocked_count"),
                "missing_artifact_paths": kv_blocker.get("missing_artifact_paths"),
                "session_contract_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-kv-session-contract.json",
                "session_contract_generator_command": commands["kv_session_contract"],
                "session_contract_status": kv_session_payload.get("status"),
                "session_contract_blocked_by": kv_session_contract_record.get("blocked_by"),
                "session_contract_ready": kv_session_contract_record.get("ready"),
                "session_contract_executable": kv_session_contract_record.get("executable"),
                "decode_entrypoint_ready": kv_decode_entrypoint.get("ready"),
                "decode_entrypoint_executable": kv_decode_entrypoint.get("executable"),
                "decode_entrypoint_no_kernel_launches": kv_decode_entrypoint.get("no_kernel_launches"),
                "decode_entrypoint_kv_dispatch_key_names": kv_decode_entrypoint.get("kv_dispatch_key_names"),
                "decode_entrypoint_kv_dispatch_keys_sha256": kv_decode_entrypoint.get("kv_dispatch_keys_sha256"),
                "decode_entrypoint_input_ids_sha256": kv_decode_entrypoint.get("input_ids_sha256"),
                "decode_entrypoint_span_input_payloads_sha256": kv_decode_entrypoint.get("span_input_payloads_sha256"),
                "decode_entrypoint_pre_run_payload_fingerprints_sha256": kv_decode_entrypoint.get("pre_run_payload_fingerprints_sha256"),
                "decode_entrypoint_pre_run_upload_plan_sha256": kv_decode_entrypoint.get("pre_run_upload_plan_sha256"),
                "decode_entrypoint_launch_operation_sequence_sha256": kv_decode_entrypoint.get("launch_operation_sequence_sha256"),
                "decode_entrypoint_launch_operation_records_sha256": kv_decode_entrypoint.get("launch_operation_records_sha256"),
                "evidence_preflight_artifact": "benchmarks/results/2026-06-15-stepfun-q3kl-kv-evidence-preflight.json",
                "evidence_preflight_generator_command": commands["kv_evidence_preflight"],
                "streaming_runner_source_status": kv_blocker.get(
                    "streaming_runner_source_status"
                ),
                "runtime_wiring_symbol_validation": kv_blocker.get(
                    "runtime_wiring_symbol_validation"
                ),
            },
        ],
        "generator_commands": commands,
        "source_artifact_sha256": {
            "status": status_mod._stable_json_sha256(status),
            "oracle_backend_matrix": status_mod._stable_json_sha256(oracle_matrix),
            "kv_blocker_status": status_mod._stable_json_sha256(kv_blocker),
            "kv_session_contract": status_mod._stable_json_sha256(
                kv_session_payload
            ),
            "llamacpp_logits_helper_readiness": status_mod._stable_json_sha256(
                helper_readiness_payload
            ),
        },
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This rollup links retained blocker evidence and generator commands; "
                "it does not satisfy oracle parity or run KV-backed decode."
            ),
        },
    }


def verify_remaining_blockers_rollup(
    rollup_artifact: Path,
    *,
    current_rollup: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted rollup artifact with the current rollup payload."""

    persisted = json.loads(rollup_artifact.read_text())
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_rollup)
    failures: list[dict[str, object]] = []
    if persisted != current_rollup:
        failures.append(
            {
                "name": "remaining_blockers_rollup_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted remaining-blockers rollup differs from current "
                    "prompt/oracle/HIP/resource/helper/KV/docs inputs."
                ),
            }
        )
    all_match = not failures
    return {
        "schema_version": 1,
        "artifact_path": str(rollup_artifact),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_rollup_sha256": persisted_sha256,
        "current_rollup_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "verification_failure_count": len(failures),
        "persisted_status": persisted.get("status")
        if isinstance(persisted, dict)
        else None,
        "current_status": current_rollup.get("status"),
        "persisted_remaining_blocker_count": persisted.get("remaining_blocker_count")
        if isinstance(persisted, dict)
        else None,
        "current_remaining_blocker_count": current_rollup.get(
            "remaining_blocker_count"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    rollup = build_remaining_blockers_rollup(
        prompt_artifact=args.prompt_artifact,
        oracle_artifact=args.oracle_artifact,
        hip_artifact=args.hip_artifact,
        resource_artifact=args.resource_artifact,
        helper_readiness_artifact=args.helper_readiness_artifact,
        kv_session_contract_artifact=args.kv_session_contract_artifact,
        docs=args.docs,
        llama_tokenize=args.llama_tokenize,
        tokenizer_model=args.tokenizer_model,
        tokenizer_timeout_s=args.tokenizer_timeout_s,
        artifact_date=args.artifact_date,
    )
    if args.verify_rollup is not None:
        verification = verify_remaining_blockers_rollup(
            args.verify_rollup,
            current_rollup=rollup,
        )
        if args.verification_status_only:
            payload: object = verification["status"]
        elif args.verification_failures_only:
            payload = verification["verification_failures"]
        elif args.verification_sha_only:
            payload = status_mod._stable_json_sha256(verification)
        else:
            payload = verification
        status_mod._emit_json(payload, pretty=args.pretty, output=args.output)
        return (
            0
            if verification["all_match"] is True
            else status_mod.SOURCE_ARTIFACT_MISMATCH_EXIT_CODE
        )
    if args.status_only:
        payload: object = rollup["status"]
    elif args.open_count_only:
        payload = rollup["remaining_blocker_count"]
    elif args.blocker_kinds_only:
        payload = [
            item["blocker_kind"]
            for item in rollup["remaining_blockers"]
            if isinstance(item, dict)
        ]
    elif args.generator_commands_only:
        payload = rollup["generator_commands"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(rollup)
    else:
        payload = rollup
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
