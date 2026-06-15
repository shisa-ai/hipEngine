#!/usr/bin/env python3
"""Emit a compact StepFun oracle rank-check artifact.

The rank check compares llama.cpp's generated token (from the retained token
mismatch diagnostic) against the host-composed expected top-token list. It is
blocker evidence only and does not claim oracle parity, KV readiness, e2e
readiness, or performance.
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
    "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-oracle-rank-check.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=status_mod.DEFAULT_ORACLE_ARTIFACT,
        help="Retained llama.cpp oracle JSON artifact to summarize.",
    )
    parser.add_argument(
        "--prompt-artifact",
        type=Path,
        default=status_mod.DEFAULT_PROMPT_ARTIFACT,
        help="Canonical StepFun prompt/logit artifact with host top-token list.",
    )
    parser.add_argument(
        "--llama-tokenize",
        type=Path,
        default=token_mismatch.DEFAULT_LLAMA_TOKENIZE,
        help="llama.cpp llama-tokenize binary used for generated-token diagnostics.",
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
        help="Per-text timeout for llama-tokenize diagnostics.",
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
        "--generated-in-top-list-only",
        action="store_true",
        help="Emit only whether the generated token is present in the host top-token list.",
    )
    parser.add_argument(
        "--generated-rank-only",
        action="store_true",
        help="Emit only the generated token rank in the host top-token list, or null.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the artifact payload.",
    )
    parser.add_argument(
        "--verify-rank-check",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help=(
            "Compare a persisted oracle rank-check artifact with current "
            f"oracle/prompt/tokenizer metadata. If no path is supplied, uses {DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-rank-check, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-rank-check, emit only verification failures.",
    )
    parser.add_argument(
        "--verification-sha-only",
        action="store_true",
        help="With --verify-rank-check, emit only the stable verification digest.",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _host_top_tokens(prompt_artifact: Path) -> list[dict[str, object]]:
    payload = _load_json_object(prompt_artifact)
    top_tokens = payload.get("top_tokens")
    if not isinstance(top_tokens, list):
        return []
    records: list[dict[str, object]] = []
    for fallback_rank, item in enumerate(top_tokens, start=1):
        if not isinstance(item, dict):
            continue
        token_id = item.get("token_id")
        rank = item.get("rank", fallback_rank)
        records.append(
            {
                "rank": rank,
                "token_id": token_id,
                "token_text": item.get("token_text"),
                "logit": item.get("logit"),
            }
        )
    return records


def build_oracle_rank_check(
    *,
    artifact: Path = status_mod.DEFAULT_ORACLE_ARTIFACT,
    prompt_artifact: Path = status_mod.DEFAULT_PROMPT_ARTIFACT,
    llama_tokenize: Path = token_mismatch.DEFAULT_LLAMA_TOKENIZE,
    tokenizer_model: Path = token_mismatch.DEFAULT_TOKENIZER_MODEL,
    tokenizer_timeout_s: float = 60.0,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the oracle generated-token rank check."""

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
    top_tokens = _host_top_tokens(prompt_artifact)
    generated_first_token_id = diagnostic.get("generated_first_token_id")
    expected_next_token_id = summary.get("expected_next_token_id")
    top_record = top_tokens[0] if top_tokens else None
    generated_rank_record = next(
        (
            record
            for record in top_tokens
            if record.get("token_id") == generated_first_token_id
        ),
        None,
    )
    generated_rank = (
        generated_rank_record.get("rank") if generated_rank_record is not None else None
    )
    generated_in_top_tokens = generated_rank_record is not None
    expected_record = next(
        (
            record
            for record in top_tokens
            if record.get("token_id") == expected_next_token_id
        ),
        None,
    )
    missing_evidence: list[str] = []
    if not top_tokens:
        missing_evidence.append("host_top_tokens_present")
    if generated_first_token_id is None:
        missing_evidence.append("generated_first_token_id_present")
    if not generated_in_top_tokens:
        missing_evidence.append("generated_token_in_host_top_tokens")
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_llamacpp_oracle_rank_check",
        "date": artifact_date,
        "status": "failed" if missing_evidence else "passed",
        "ready": not missing_evidence,
        "artifact": str(artifact),
        "artifact_sha256": summary.get("artifact_sha256"),
        "prompt_artifact": str(prompt_artifact),
        "prompt_artifact_sha256": status_mod._stable_json_sha256(
            _load_json_object(prompt_artifact)
        ),
        "oracle_status": summary.get("oracle_status"),
        "oracle_returncode": summary.get("oracle_returncode"),
        "expected_next_token_id": expected_next_token_id,
        "expected_next_token_text": summary.get("expected_next_token_text"),
        "expected_top_record": expected_record,
        "host_top_token_count": len(top_tokens),
        "host_top_tokens": top_tokens,
        "host_top_token_ids": [record.get("token_id") for record in top_tokens],
        "host_top_record": top_record,
        "generated_text": summary.get("generated_text"),
        "generated_text_token_ids": diagnostic.get("generated_text_token_ids"),
        "generated_text_stripped_token_ids": diagnostic.get(
            "generated_text_stripped_token_ids"
        ),
        "generated_first_token_id": generated_first_token_id,
        "generated_first_token_matches_expected_id": diagnostic.get(
            "generated_first_token_matches_expected_id"
        ),
        "generated_token_in_host_top_tokens": generated_in_top_tokens,
        "generated_token_host_top_record": generated_rank_record,
        "generated_token_host_rank": generated_rank,
        "missing_evidence": missing_evidence,
        "conclusion": (
            "generated token is absent from host top-token list"
            if not generated_in_top_tokens
            else "generated token appears in host top-token list"
        ),
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This rank check only compares retained oracle/token artifacts with "
                "the host top-token list; it does not resolve the oracle mismatch."
            ),
        },
    }


def verify_oracle_rank_check(
    rank_artifact: Path,
    *,
    current_report: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted oracle rank-check artifact with current metadata."""

    persisted = _load_json_object(rank_artifact)
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_report)
    failures: list[dict[str, object]] = []
    if persisted != current_report:
        failures.append(
            {
                "name": "oracle_rank_check_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted oracle rank-check artifact differs from current "
                    "oracle/prompt/tokenizer metadata."
                ),
            }
        )
    all_match = not failures
    return {
        "schema_version": 1,
        "artifact_path": str(rank_artifact),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_artifact_sha256": persisted_sha256,
        "current_artifact_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "verification_failure_count": len(failures),
        "persisted_status": persisted.get("status"),
        "current_status": current_report.get("status"),
        "persisted_generated_token_in_host_top_tokens": persisted.get(
            "generated_token_in_host_top_tokens"
        ),
        "current_generated_token_in_host_top_tokens": current_report.get(
            "generated_token_in_host_top_tokens"
        ),
        "persisted_generated_token_host_rank": persisted.get(
            "generated_token_host_rank"
        ),
        "current_generated_token_host_rank": current_report.get(
            "generated_token_host_rank"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    report = build_oracle_rank_check(
        artifact=args.artifact,
        prompt_artifact=args.prompt_artifact,
        llama_tokenize=args.llama_tokenize,
        tokenizer_model=args.tokenizer_model,
        tokenizer_timeout_s=args.tokenizer_timeout_s,
        artifact_date=args.artifact_date,
    )
    if args.verify_rank_check is not None:
        verification = verify_oracle_rank_check(
            args.verify_rank_check,
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
    elif args.generated_in_top_list_only:
        payload = report["generated_token_in_host_top_tokens"]
    elif args.generated_rank_only:
        payload = report["generated_token_host_rank"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(report)
    else:
        payload = report
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
