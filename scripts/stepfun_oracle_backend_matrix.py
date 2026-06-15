#!/usr/bin/env python3
"""Emit a compact StepFun llama.cpp oracle backend matrix artifact.

The matrix consolidates the retained Vulkan executed-mismatch oracle evidence and
HIP timeout evidence. It is blocker evidence only: passing this script does not
imply oracle parity, KV-backed decode readiness, e2e readiness, or performance.
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
from scripts import stepfun_oracle_artifact_check as oracle_check
from scripts import stepfun_oracle_token_mismatch as token_mismatch

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-oracle-backend-matrix.json"
)
DEFAULT_HIP_ARTIFACT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-hip-oracle-timeout.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vulkan-artifact",
        type=Path,
        default=status_mod.DEFAULT_ORACLE_ARTIFACT,
        help="Canonical Vulkan llama.cpp oracle artifact.",
    )
    parser.add_argument(
        "--hip-artifact",
        type=Path,
        default=DEFAULT_HIP_ARTIFACT,
        help="HIP llama.cpp oracle artifact to summarize.",
    )
    parser.add_argument(
        "--prompt-artifact",
        type=Path,
        default=status_mod.DEFAULT_PROMPT_ARTIFACT,
        help="Canonical StepFun prompt/logit artifact providing expected token/text.",
    )
    parser.add_argument(
        "--llama-tokenize",
        type=Path,
        default=token_mismatch.DEFAULT_LLAMA_TOKENIZE,
        help="llama.cpp llama-tokenize binary for Vulkan token diagnostics.",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=token_mismatch.DEFAULT_TOKENIZER_MODEL,
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
        help="Date string to record in the matrix artifact.",
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
    parser.add_argument("--status-only", action="store_true", help="Emit only matrix status.")
    parser.add_argument(
        "--backend-outcomes-only",
        action="store_true",
        help="Emit only backend/outcome pairs.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the matrix payload.",
    )
    parser.add_argument(
        "--verify-matrix",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help=(
            "Compare a persisted oracle backend-matrix artifact with current "
            f"Vulkan/HIP oracle metadata. If no path is supplied, uses {DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-matrix, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-matrix, emit only verification failures.",
    )
    parser.add_argument(
        "--verification-sha-only",
        action="store_true",
        help="With --verify-matrix, emit only the stable verification digest.",
    )
    return parser.parse_args(argv)


def _vulkan_backend_record(
    *,
    artifact: Path,
    prompt_artifact: Path,
    llama_tokenize: Path,
    tokenizer_model: Path,
    tokenizer_timeout_s: float,
) -> dict[str, object]:
    summary = token_mismatch.build_oracle_token_mismatch_summary(
        artifact=artifact,
        prompt_artifact=prompt_artifact,
        llama_tokenize=llama_tokenize,
        tokenizer_model=tokenizer_model,
        tokenizer_timeout_s=tokenizer_timeout_s,
    )
    diagnostic = summary.get("tokenization_diagnostic")
    if not isinstance(diagnostic, dict):
        diagnostic = {}
    outcome = (
        "executed_match"
        if summary.get("status") == "passed"
        else "executed_token_mismatch"
        if summary.get("oracle_status") == "executed"
        else str(summary.get("oracle_status") or "unknown")
    )
    return {
        "backend": "vulkan",
        "role": "canonical_executed_oracle",
        "artifact": str(artifact),
        "artifact_sha256": summary.get("artifact_sha256"),
        "status": summary.get("status"),
        "outcome": outcome,
        "oracle_status": summary.get("oracle_status"),
        "oracle_returncode": summary.get("oracle_returncode"),
        "oracle_blocker_kind": summary.get("oracle_blocker_kind"),
        "missing_evidence": summary.get("missing_evidence"),
        "expected_next_token_id": summary.get("expected_next_token_id"),
        "expected_next_token_text": summary.get("expected_next_token_text"),
        "expected_next_token_text_token_ids": diagnostic.get(
            "expected_next_token_text_token_ids"
        ),
        "generated_text": summary.get("generated_text"),
        "generated_text_token_ids": diagnostic.get("generated_text_token_ids"),
        "generated_text_stripped_token_ids": diagnostic.get(
            "generated_text_stripped_token_ids"
        ),
        "generated_first_token_id": diagnostic.get("generated_first_token_id"),
        "generated_first_token_matches_expected_id": diagnostic.get(
            "generated_first_token_matches_expected_id"
        ),
    }


def _hip_backend_record(*, artifact: Path, prompt_artifact: Path) -> dict[str, object]:
    report = oracle_check.build_oracle_check_report(
        artifact,
        prompt_artifact=prompt_artifact,
    )
    summary = report["oracle_summary"]
    if not isinstance(summary, dict):
        raise ValueError("oracle checker returned a non-object HIP oracle_summary")
    outcome = (
        "timeout"
        if summary.get("oracle_status") == "timeout"
        else str(summary.get("oracle_status") or "unknown")
    )
    return {
        "backend": "hip",
        "role": "comparison_timeout_oracle",
        "artifact": str(artifact),
        "artifact_sha256": summary.get("artifact_sha256"),
        "status": summary.get("status"),
        "outcome": outcome,
        "oracle_status": summary.get("oracle_status"),
        "oracle_returncode": summary.get("oracle_returncode"),
        "oracle_blocker_kind": summary.get("oracle_blocker_kind"),
        "missing_evidence": summary.get("missing_evidence"),
        "expected_next_token_id": summary.get("expected_next_token_id"),
        "expected_next_token_text": summary.get("expected_next_token_text"),
        "generated_text": summary.get("generated_text"),
        "generated_text_len": summary.get("generated_text_len"),
    }


def build_oracle_backend_matrix(
    *,
    vulkan_artifact: Path = status_mod.DEFAULT_ORACLE_ARTIFACT,
    hip_artifact: Path = DEFAULT_HIP_ARTIFACT,
    prompt_artifact: Path = status_mod.DEFAULT_PROMPT_ARTIFACT,
    llama_tokenize: Path = token_mismatch.DEFAULT_LLAMA_TOKENIZE,
    tokenizer_model: Path = token_mismatch.DEFAULT_TOKENIZER_MODEL,
    tokenizer_timeout_s: float = 60.0,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the StepFun llama.cpp oracle backend matrix."""

    records = [
        _vulkan_backend_record(
            artifact=vulkan_artifact,
            prompt_artifact=prompt_artifact,
            llama_tokenize=llama_tokenize,
            tokenizer_model=tokenizer_model,
            tokenizer_timeout_s=tokenizer_timeout_s,
        ),
        _hip_backend_record(artifact=hip_artifact, prompt_artifact=prompt_artifact),
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_llamacpp_oracle_backend_matrix",
        "date": artifact_date,
        "status": "blocked",
        "canonical_backend": "vulkan",
        "oracle_parity_ready": False,
        "blocked_reason": (
            "canonical Vulkan oracle executes but generated token does not match the "
            "host-composed target; HIP comparison oracle times out before generation"
        ),
        "backend_count": len(records),
        "backend_outcomes": [
            {"backend": record["backend"], "outcome": record["outcome"]}
            for record in records
        ],
        "backends": records,
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This matrix consolidates oracle blocker evidence only; it does not "
                "resolve the generated token mismatch or run KV-backed decode."
            ),
        },
    }


def verify_oracle_backend_matrix(
    matrix_artifact: Path,
    *,
    current_matrix: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted oracle backend-matrix artifact with current metadata."""

    persisted = json.loads(matrix_artifact.read_text())
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_matrix)
    failures: list[dict[str, object]] = []
    if persisted != current_matrix:
        failures.append(
            {
                "name": "oracle_backend_matrix_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted oracle backend-matrix artifact differs from current "
                    "Vulkan/HIP oracle artifact metadata."
                ),
            }
        )
    all_match = not failures
    return {
        "schema_version": 1,
        "artifact_path": str(matrix_artifact),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_artifact_sha256": persisted_sha256,
        "current_artifact_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "verification_failure_count": len(failures),
        "persisted_status": persisted.get("status")
        if isinstance(persisted, dict)
        else None,
        "current_status": current_matrix.get("status"),
        "persisted_backend_outcomes": persisted.get("backend_outcomes")
        if isinstance(persisted, dict)
        else None,
        "current_backend_outcomes": current_matrix.get("backend_outcomes"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    matrix = build_oracle_backend_matrix(
        vulkan_artifact=args.vulkan_artifact,
        hip_artifact=args.hip_artifact,
        prompt_artifact=args.prompt_artifact,
        llama_tokenize=args.llama_tokenize,
        tokenizer_model=args.tokenizer_model,
        tokenizer_timeout_s=args.tokenizer_timeout_s,
        artifact_date=args.artifact_date,
    )
    if args.verify_matrix is not None:
        verification = verify_oracle_backend_matrix(
            args.verify_matrix,
            current_matrix=matrix,
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
        payload: object = matrix["status"]
    elif args.backend_outcomes_only:
        payload = matrix["backend_outcomes"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(matrix)
    else:
        payload = matrix
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
