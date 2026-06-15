#!/usr/bin/env python3
"""Emit the retained StepFun llama.cpp oracle token-mismatch artifact.

This is a reproducible wrapper around ``stepfun_oracle_artifact_check.py`` with
``--summary-only`` and llama.cpp no-BOS tokenizer diagnostics enabled. The output
is blocker evidence only: passing this script does not imply oracle parity, KV
readiness, e2e readiness, or any performance claim.
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

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-oracle-token-mismatch.json"
)
DEFAULT_LLAMA_TOKENIZE = Path(
    "/home/lhl/llama.cpp/llama.cpp-vulkan/build-vulkan-release/bin/llama-tokenize"
)
DEFAULT_TOKENIZER_MODEL = Path("/models/gguf/Step-3.7-flash-Q3_K_L-00001-of-00003.gguf")


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
        help="Canonical StepFun prompt/logit artifact providing expected token/text.",
    )
    parser.add_argument(
        "--llama-tokenize",
        type=Path,
        default=DEFAULT_LLAMA_TOKENIZE,
        help="llama.cpp llama-tokenize binary used for no-BOS token diagnostics.",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=DEFAULT_TOKENIZER_MODEL,
        help="GGUF model to pass to llama-tokenize.",
    )
    parser.add_argument(
        "--tokenizer-timeout-s",
        type=float,
        default=60.0,
        help="Per-text timeout for llama-tokenize diagnostics.",
    )
    parser.add_argument(
        "--logit-atol",
        type=float,
        default=oracle_check.DEFAULT_LOGIT_ATOL,
        help="Absolute tolerance for expected top-token logit metadata.",
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
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Emit only the compact summary status string.",
    )
    parser.add_argument(
        "--missing-evidence-only",
        action="store_true",
        help="Emit only the compact summary missing_evidence list.",
    )
    parser.add_argument(
        "--token-ids-only",
        action="store_true",
        help=(
            "Emit only expected/generated token ids from the tokenization diagnostic."
        ),
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the compact summary payload.",
    )
    parser.add_argument(
        "--verify-token-mismatch",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help=(
            "Compare a persisted oracle token-mismatch artifact with current "
            f"oracle/prompt/tokenizer metadata. If no path is supplied, uses {DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-token-mismatch, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-token-mismatch, emit only verification failures.",
    )
    parser.add_argument(
        "--verification-sha-only",
        action="store_true",
        help="With --verify-token-mismatch, emit only the stable verification digest.",
    )
    return parser.parse_args(argv)


def build_oracle_token_mismatch_summary(
    *,
    artifact: Path = status_mod.DEFAULT_ORACLE_ARTIFACT,
    prompt_artifact: Path = status_mod.DEFAULT_PROMPT_ARTIFACT,
    llama_tokenize: Path = DEFAULT_LLAMA_TOKENIZE,
    tokenizer_model: Path = DEFAULT_TOKENIZER_MODEL,
    tokenizer_timeout_s: float = 60.0,
    logit_atol: float = oracle_check.DEFAULT_LOGIT_ATOL,
) -> dict[str, object]:
    """Return the compact oracle summary with token-mismatch diagnostics."""

    report = oracle_check.build_oracle_check_report(
        artifact,
        prompt_artifact=prompt_artifact,
        logit_atol=logit_atol,
        llama_tokenize=llama_tokenize,
        tokenizer_model=tokenizer_model,
        tokenizer_timeout_s=tokenizer_timeout_s,
    )
    summary = report["oracle_summary"]
    if not isinstance(summary, dict):
        raise ValueError("oracle checker returned a non-object oracle_summary")
    return summary


def _token_id_payload(summary: dict[str, object]) -> dict[str, object]:
    diagnostic = summary.get("tokenization_diagnostic")
    if not isinstance(diagnostic, dict):
        return {
            "status": summary.get("status"),
            "missing_evidence": summary.get("missing_evidence"),
            "tokenization_diagnostic_present": False,
        }
    return {
        "status": summary.get("status"),
        "missing_evidence": summary.get("missing_evidence"),
        "expected_next_token_id": diagnostic.get("expected_next_token_id"),
        "expected_next_token_text_token_ids": diagnostic.get(
            "expected_next_token_text_token_ids"
        ),
        "generated_text_token_ids": diagnostic.get("generated_text_token_ids"),
        "generated_text_stripped_token_ids": diagnostic.get(
            "generated_text_stripped_token_ids"
        ),
        "generated_first_token_id": diagnostic.get("generated_first_token_id"),
        "generated_first_token_matches_expected_id": diagnostic.get(
            "generated_first_token_matches_expected_id"
        ),
        "conclusion": diagnostic.get("conclusion"),
    }


def verify_oracle_token_mismatch_summary(
    token_mismatch_artifact: Path,
    *,
    current_summary: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted token-mismatch artifact with current metadata."""

    persisted = json.loads(token_mismatch_artifact.read_text())
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_summary)
    failures: list[dict[str, object]] = []
    if persisted != current_summary:
        failures.append(
            {
                "name": "oracle_token_mismatch_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted oracle token-mismatch artifact differs from current "
                    "oracle/prompt/tokenizer metadata."
                ),
            }
        )
    all_match = not failures
    persisted_diag = (
        persisted.get("tokenization_diagnostic")
        if isinstance(persisted, dict)
        else None
    )
    persisted_diag = persisted_diag if isinstance(persisted_diag, dict) else {}
    current_diag = current_summary.get("tokenization_diagnostic")
    current_diag = current_diag if isinstance(current_diag, dict) else {}
    return {
        "schema_version": 1,
        "artifact_path": str(token_mismatch_artifact),
        "status": "match" if all_match else "mismatch",
        "all_match": all_match,
        "persisted_artifact_sha256": persisted_sha256,
        "current_artifact_sha256": current_sha256,
        "verification_failures": failures,
        "verification_failures_sha256": status_mod._stable_json_sha256(failures),
        "verification_failure_count": len(failures),
        "persisted_status": (
            persisted.get("status") if isinstance(persisted, dict) else None
        ),
        "current_status": current_summary.get("status"),
        "persisted_expected_next_token_id": persisted_diag.get(
            "expected_next_token_id"
        ),
        "current_expected_next_token_id": current_diag.get("expected_next_token_id"),
        "persisted_generated_first_token_id": persisted_diag.get(
            "generated_first_token_id"
        ),
        "current_generated_first_token_id": current_diag.get(
            "generated_first_token_id"
        ),
        "persisted_generated_first_token_matches_expected_id": persisted_diag.get(
            "generated_first_token_matches_expected_id"
        ),
        "current_generated_first_token_matches_expected_id": current_diag.get(
            "generated_first_token_matches_expected_id"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    summary = build_oracle_token_mismatch_summary(
        artifact=args.artifact,
        prompt_artifact=args.prompt_artifact,
        llama_tokenize=args.llama_tokenize,
        tokenizer_model=args.tokenizer_model,
        tokenizer_timeout_s=args.tokenizer_timeout_s,
        logit_atol=args.logit_atol,
    )
    if args.verify_token_mismatch is not None:
        verification = verify_oracle_token_mismatch_summary(
            args.verify_token_mismatch,
            current_summary=summary,
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
        payload: object = summary["status"]
    elif args.missing_evidence_only:
        payload = summary["missing_evidence"]
    elif args.token_ids_only:
        payload = _token_id_payload(summary)
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(summary)
    else:
        payload = summary
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
