#!/usr/bin/env python3
"""Emit a StepFun host top-logit margin artifact.

The margin check records the retained host top-token logits for the oracle prompt
and verifies the expected token is the host top-1 with a positive visible margin.
It helps prioritize the next logits/backend parity investigation, but it does
not claim oracle parity, KV readiness, e2e readiness, or performance.
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
from scripts import stepfun_oracle_rank_check as rank_check

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-host-top-logit-margin.json"
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
        "--rank-check-artifact",
        type=Path,
        default=rank_check.DEFAULT_OUTPUT,
        help="Retained rank-check artifact, used to include generated-token context.",
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
        "--top1-margin-only",
        action="store_true",
        help="Emit only the top-1 to top-2 visible logit margin.",
    )
    parser.add_argument(
        "--expected-top1-only",
        action="store_true",
        help="Emit only whether the expected token is host top-1.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the artifact payload.",
    )
    parser.add_argument(
        "--verify-margin",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        default=None,
        help=(
            "Compare a persisted host top-logit margin artifact with current "
            f"prompt/rank-check metadata. If no path is supplied, uses {DEFAULT_OUTPUT}."
        ),
    )
    parser.add_argument(
        "--verification-status-only",
        action="store_true",
        help="With --verify-margin, emit only match/mismatch status.",
    )
    parser.add_argument(
        "--verification-failures-only",
        action="store_true",
        help="With --verify-margin, emit only verification failures.",
    )
    parser.add_argument(
        "--verification-sha-only",
        action="store_true",
        help="With --verify-margin, emit only the stable verification digest.",
    )
    return parser.parse_args(argv)


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _load_optional_json_object(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _logit(record: dict[str, object] | None) -> float | None:
    if record is None:
        return None
    return _as_float(record.get("logit"))


def _margin(
    left: dict[str, object] | None, right: dict[str, object] | None
) -> float | None:
    left_logit = _logit(left)
    right_logit = _logit(right)
    if left_logit is None or right_logit is None:
        return None
    return left_logit - right_logit


def _top_tokens_sorted_desc(top_tokens: list[dict[str, object]]) -> bool:
    logits = [_logit(record) for record in top_tokens]
    if any(logit is None for logit in logits):
        return False
    return all(float(logits[i]) >= float(logits[i + 1]) for i in range(len(logits) - 1))


def build_host_logit_margin(
    *,
    prompt_artifact: Path = status_mod.DEFAULT_PROMPT_ARTIFACT,
    rank_check_artifact: Path = rank_check.DEFAULT_OUTPUT,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the retained host top-logit margin report."""

    prompt_payload = rank_check._load_json_object(prompt_artifact)
    rank_payload = _load_optional_json_object(rank_check_artifact)
    top_tokens = rank_check._host_top_tokens(prompt_artifact)
    top1 = top_tokens[0] if len(top_tokens) >= 1 else None
    top2 = top_tokens[1] if len(top_tokens) >= 2 else None
    top5 = top_tokens[4] if len(top_tokens) >= 5 else None
    expected_next_token_id = prompt_payload.get("next_token_id")
    expected_next_token_text = prompt_payload.get("next_token_text")
    expected_next_token_logit = prompt_payload.get("next_token_logit")
    top1_matches_expected_id = (
        top1 is not None and top1.get("token_id") == expected_next_token_id
    )
    sorted_desc = _top_tokens_sorted_desc(top_tokens)
    top1_to_top2_margin = _margin(top1, top2)
    top1_to_top5_margin = _margin(top1, top5)
    pairwise_margins = []
    for index in range(len(top_tokens) - 1):
        pairwise_margins.append(
            {
                "higher_rank": top_tokens[index].get("rank"),
                "lower_rank": top_tokens[index + 1].get("rank"),
                "margin": _margin(top_tokens[index], top_tokens[index + 1]),
            }
        )
    margin_positive = top1_to_top2_margin is not None and top1_to_top2_margin > 0.0
    missing_evidence: list[str] = []
    if not top_tokens:
        missing_evidence.append("host_top_tokens_present")
    if len(top_tokens) < 2:
        missing_evidence.append("at_least_two_host_top_tokens_present")
    if not top1_matches_expected_id:
        missing_evidence.append("expected_token_is_host_top1")
    if not sorted_desc:
        missing_evidence.append("host_top_tokens_sorted_by_logit_desc")
    if not margin_positive:
        missing_evidence.append("host_top1_to_top2_margin_positive")
    generated_context: dict[str, object] = {
        "rank_check_artifact": str(rank_check_artifact),
        "rank_check_present": rank_payload is not None,
    }
    if rank_payload is not None:
        generated_context.update(
            {
                "generated_first_token_id": rank_payload.get("generated_first_token_id"),
                "generated_token_in_host_top_tokens": rank_payload.get(
                    "generated_token_in_host_top_tokens"
                ),
                "generated_token_host_rank": rank_payload.get(
                    "generated_token_host_rank"
                ),
            }
        )
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_host_top_logit_margin",
        "date": artifact_date,
        "status": "passed" if not missing_evidence else "failed",
        "ready": not missing_evidence,
        "prompt_artifact": str(prompt_artifact),
        "prompt_artifact_sha256": status_mod._stable_json_sha256(prompt_payload),
        "expected_next_token_id": expected_next_token_id,
        "expected_next_token_text": expected_next_token_text,
        "expected_next_token_logit": expected_next_token_logit,
        "host_top_token_count": len(top_tokens),
        "host_top_tokens": top_tokens,
        "host_top_token_ids": [record.get("token_id") for record in top_tokens],
        "host_top_logits": [record.get("logit") for record in top_tokens],
        "top1_record": top1,
        "top2_record": top2,
        "top5_record": top5,
        "top1_matches_expected_id": top1_matches_expected_id,
        "top_tokens_sorted_by_logit_desc": sorted_desc,
        "top1_to_top2_margin": top1_to_top2_margin,
        "top1_to_top5_margin": top1_to_top5_margin,
        "adjacent_visible_margins": pairwise_margins,
        "generated_token_context": generated_context,
        "missing_evidence": missing_evidence,
        "conclusion": (
            "expected token is retained host top-1 with a positive visible logit margin"
            if not missing_evidence
            else "retained host top-logit margin evidence is incomplete"
        ),
        "oracle_blocker_interpretation": (
            "the retained host logits make the expected token a clear visible top-1; "
            "the llama.cpp generated-token mismatch should be investigated as "
            "logits/backend parity rather than a near-tie in the retained host top list"
            if not missing_evidence
            else "host top-logit evidence is not sufficient to rule out an ambiguous top-1"
        ),
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This artifact only summarizes retained host top-token logits; "
                "generated_text_matches_target remains unresolved."
            ),
        },
    }


def verify_host_logit_margin(
    margin_artifact: Path,
    *,
    current_report: dict[str, object],
) -> dict[str, object]:
    """Compare a persisted host top-logit margin artifact with current metadata."""

    persisted = rank_check._load_json_object(margin_artifact)
    persisted_sha256 = status_mod._stable_json_sha256(persisted)
    current_sha256 = status_mod._stable_json_sha256(current_report)
    failures: list[dict[str, object]] = []
    if persisted != current_report:
        failures.append(
            {
                "name": "host_logit_margin_drift",
                "expected_sha256": current_sha256,
                "actual_sha256": persisted_sha256,
                "evidence": (
                    "Persisted host top-logit margin artifact differs from "
                    "current prompt/rank-check metadata."
                ),
            }
        )
    all_match = not failures
    return {
        "schema_version": 1,
        "artifact_path": str(margin_artifact),
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
        "persisted_top1_to_top2_margin": persisted.get("top1_to_top2_margin"),
        "current_top1_to_top2_margin": current_report.get("top1_to_top2_margin"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    report = build_host_logit_margin(
        prompt_artifact=args.prompt_artifact,
        rank_check_artifact=args.rank_check_artifact,
        artifact_date=args.artifact_date,
    )
    if args.verify_margin is not None:
        verification = verify_host_logit_margin(
            args.verify_margin,
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
    elif args.top1_margin_only:
        payload = report["top1_to_top2_margin"]
    elif args.expected_top1_only:
        payload = report["top1_matches_expected_id"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(report)
    else:
        payload = report
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
