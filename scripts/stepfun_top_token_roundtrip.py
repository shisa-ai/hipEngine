#!/usr/bin/env python3
"""Emit a StepFun host top-token text/tokenizer round-trip artifact.

The round-trip check tokenizes each host-composed top-token text with the same
llama.cpp ``llama-tokenize --no-bos`` mode used by the oracle token-mismatch
helpers and verifies it maps back to the host token id. This narrows oracle
blocker evidence only: it can rule out top-token text-label drift, but it does
not prove oracle parity, KV readiness, e2e readiness, or performance.
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
from scripts import stepfun_oracle_rank_check as rank_check
from scripts import stepfun_oracle_token_mismatch as token_mismatch

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-host-top-token-roundtrip.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="llama.cpp llama-tokenize binary used for host top-token diagnostics.",
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
        "--all-roundtrip-only",
        action="store_true",
        help="Emit only whether all host top-token texts round-trip to their token ids.",
    )
    parser.add_argument(
        "--mismatches-only",
        action="store_true",
        help="Emit only host top-token records whose text did not round-trip.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the artifact payload.",
    )
    parser.add_argument(
        "--verify-roundtrip",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help=(
            "Compare a persisted host top-token roundtrip artifact with current "
            f"prompt/tokenizer metadata. If no path is supplied, uses {DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-roundtrip, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-roundtrip, emit only verification failures.",
    )
    parser.add_argument(
        "--verification-sha-only",
        action="store_true",
        help="With --verify-roundtrip, emit only the stable verification digest.",
    )
    return parser.parse_args(argv)


def _host_top_tokens(prompt_artifact: Path) -> list[dict[str, object]]:
    return rank_check._host_top_tokens(prompt_artifact)


def _tokenize_host_top_record(
    record: dict[str, object],
    *,
    llama_tokenize: Path,
    tokenizer_model: Path,
    tokenizer_timeout_s: float,
) -> dict[str, object]:
    rank = record.get("rank")
    token_id = record.get("token_id")
    token_text = record.get("token_text")
    output: dict[str, object] = {
        "rank": rank,
        "token_id": token_id,
        "token_text": token_text,
        "logit": record.get("logit"),
    }
    if not isinstance(token_text, str):
        output.update(
            {
                "roundtrip_status": "failed",
                "roundtrip_token_ids": None,
                "roundtrip_token_count": None,
                "single_token_matches_host_id": False,
                "missing_evidence": ["host_top_token_text_present"],
            }
        )
        return output
    tokenization = oracle_check._tokenize_text_with_llamacpp(
        text=token_text,
        label=f"host-top-token-rank-{rank}",
        llama_tokenize=llama_tokenize,
        model=tokenizer_model,
        timeout_s=tokenizer_timeout_s,
    )
    token_ids = tokenization.get("token_ids")
    expected_ids = [token_id] if isinstance(token_id, int) else None
    single_token_matches_host_id = token_ids == expected_ids
    missing_evidence: list[str] = []
    if tokenization.get("status") != "passed":
        missing_evidence.append("host_top_token_text_tokenization_passed")
    if not single_token_matches_host_id:
        missing_evidence.append("host_top_token_text_roundtrips_to_token_id")
    output.update(
        {
            "roundtrip_status": tokenization.get("status"),
            "roundtrip_token_ids": token_ids,
            "roundtrip_token_count": tokenization.get("token_count"),
            "single_token_matches_host_id": single_token_matches_host_id,
            "missing_evidence": missing_evidence,
            "tokenization_record": tokenization,
        }
    )
    return output


def build_top_token_roundtrip(
    *,
    prompt_artifact: Path = status_mod.DEFAULT_PROMPT_ARTIFACT,
    llama_tokenize: Path = token_mismatch.DEFAULT_LLAMA_TOKENIZE,
    tokenizer_model: Path = token_mismatch.DEFAULT_TOKENIZER_MODEL,
    tokenizer_timeout_s: float = 60.0,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the host top-token text/token-id round-trip report."""

    prompt_payload = rank_check._load_json_object(prompt_artifact)
    host_top_tokens = _host_top_tokens(prompt_artifact)
    records = [
        _tokenize_host_top_record(
            record,
            llama_tokenize=llama_tokenize,
            tokenizer_model=tokenizer_model,
            tokenizer_timeout_s=tokenizer_timeout_s,
        )
        for record in host_top_tokens
    ]
    mismatches = [
        record for record in records if record.get("single_token_matches_host_id") is not True
    ]
    missing_evidence: list[str] = []
    if not host_top_tokens:
        missing_evidence.append("host_top_tokens_present")
    for record in mismatches:
        rank = record.get("rank")
        missing_evidence.extend(
            f"rank_{rank}_{item}" for item in record.get("missing_evidence", [])
        )
    all_roundtrip = bool(host_top_tokens) and not mismatches
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_host_top_token_roundtrip",
        "date": artifact_date,
        "status": "passed" if all_roundtrip else "failed",
        "ready": all_roundtrip,
        "prompt_artifact": str(prompt_artifact),
        "prompt_artifact_sha256": status_mod._stable_json_sha256(prompt_payload),
        "llama_tokenize": str(llama_tokenize),
        "tokenizer_model": str(tokenizer_model),
        "mode": "llama-tokenize --no-bos",
        "host_top_token_count": len(host_top_tokens),
        "host_top_token_ids": [record.get("token_id") for record in host_top_tokens],
        "records": records,
        "all_top_token_texts_roundtrip": all_roundtrip,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "missing_evidence": missing_evidence,
        "conclusion": (
            "all host top-token texts round-trip to their host token ids"
            if all_roundtrip
            else "one or more host top-token texts did not round-trip to their host token ids"
        ),
        "oracle_blocker_interpretation": (
            "top-token text labels are tokenizer-coherent; the retained llama.cpp "
            "oracle mismatch should be investigated as logits/prompt/backend parity, "
            "not host top-token text-label drift"
            if all_roundtrip
            else "top-token text-label/tokenizer coherence is not fully established"
        ),
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This artifact only checks host top-token text/token-id round-trips "
                "with llama-tokenize; it does not compare generated text to the target."
            ),
        },
    }


def verify_top_token_roundtrip(
    roundtrip_artifact: Path,
    *,
    current_report: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted host top-token roundtrip artifact with current metadata."""

    persisted = json.loads(roundtrip_artifact.read_text())
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_report)
    failures: list[dict[str, object]] = []
    if persisted != current_report:
        failures.append(
            {
                "name": "top_token_roundtrip_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted host top-token roundtrip artifact differs from "
                    "current prompt/tokenizer metadata."
                ),
            }
        )
    all_match = not failures
    return {
        "schema_version": 1,
        "artifact_path": str(roundtrip_artifact),
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
        "current_status": current_report.get("status"),
        "persisted_all_top_token_texts_roundtrip": persisted.get(
            "all_top_token_texts_roundtrip"
        )
        if isinstance(persisted, dict)
        else None,
        "current_all_top_token_texts_roundtrip": current_report.get(
            "all_top_token_texts_roundtrip"
        ),
        "persisted_mismatch_count": persisted.get("mismatch_count")
        if isinstance(persisted, dict)
        else None,
        "current_mismatch_count": current_report.get("mismatch_count"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    report = build_top_token_roundtrip(
        prompt_artifact=args.prompt_artifact,
        llama_tokenize=args.llama_tokenize,
        tokenizer_model=args.tokenizer_model,
        tokenizer_timeout_s=args.tokenizer_timeout_s,
        artifact_date=args.artifact_date,
    )
    if args.verify_roundtrip is not None:
        verification = verify_top_token_roundtrip(
            args.verify_roundtrip,
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
    elif args.all_roundtrip_only:
        payload = report["all_top_token_texts_roundtrip"]
    elif args.mismatches_only:
        payload = report["mismatches"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(report)
    else:
        payload = report
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
