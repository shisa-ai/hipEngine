#!/usr/bin/env python3
"""Check retained StepFun oracle evidence artifacts for internal consistency.

The consistency check detects stale or contradictory blocker evidence across the
retained oracle diagnosis bundle before the next logits/backend parity
investigation. It is handoff/blocker hygiene only and does not claim oracle
parity, KV readiness, e2e readiness, or performance.
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
from scripts import stepfun_oracle_backend_matrix as backend_matrix
from scripts import stepfun_oracle_blocker_diagnosis as diagnosis_mod
from scripts import stepfun_oracle_rank_check as rank_check
from scripts import stepfun_oracle_token_mismatch as token_mismatch
from scripts import stepfun_prompt_token_roundtrip as prompt_roundtrip
from scripts import stepfun_top_token_roundtrip as top_roundtrip

DEFAULT_OUTPUT = Path(
    "benchmarks/results/2026-06-15-stepfun-q3kl-oracle-evidence-consistency-check.json"
)
DEFAULT_ARTIFACT_DATE = "2026-06-15"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnosis-artifact",
        type=Path,
        default=diagnosis_mod.DEFAULT_OUTPUT,
        help="Retained oracle blocker diagnosis artifact.",
    )
    parser.add_argument(
        "--backend-matrix-artifact",
        type=Path,
        default=backend_matrix.DEFAULT_OUTPUT,
        help="Retained oracle backend matrix artifact.",
    )
    parser.add_argument(
        "--token-mismatch-artifact",
        type=Path,
        default=token_mismatch.DEFAULT_OUTPUT,
        help="Retained oracle token-mismatch artifact.",
    )
    parser.add_argument(
        "--rank-check-artifact",
        type=Path,
        default=rank_check.DEFAULT_OUTPUT,
        help="Retained oracle rank-check artifact.",
    )
    parser.add_argument(
        "--top-token-roundtrip-artifact",
        type=Path,
        default=top_roundtrip.DEFAULT_OUTPUT,
        help="Retained host top-token text round-trip artifact.",
    )
    parser.add_argument(
        "--prompt-token-roundtrip-artifact",
        type=Path,
        default=prompt_roundtrip.DEFAULT_OUTPUT,
        help="Retained host prompt token round-trip artifact.",
    )
    parser.add_argument(
        "--artifact-date",
        default=DEFAULT_ARTIFACT_DATE,
        help="Date string to record in the consistency artifact.",
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
        "--inconsistencies-only",
        action="store_true",
        help="Emit only consistency-check failures.",
    )
    parser.add_argument(
        "--checks-only",
        action="store_true",
        help="Emit only consistency-check records.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the consistency payload.",
    )
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _artifact_ref(path: Path, payload: dict[str, object]) -> dict[str, object]:
    return {
        "path": str(path),
        "artifact_kind": payload.get("artifact_kind"),
        "status": payload.get("status"),
        "sha256": status_mod._stable_json_sha256(payload),
    }


def _diagnosis_artifact_ref(
    diagnosis: dict[str, object], artifact_key: str
) -> dict[str, object]:
    refs = diagnosis.get("evidence_artifacts")
    if not isinstance(refs, dict):
        return {}
    ref = refs.get(artifact_key)
    return ref if isinstance(ref, dict) else {}


def _first_stripped_generated_token_id(mismatch: dict[str, object]) -> object:
    diagnostic = mismatch.get("tokenization_diagnostic")
    if not isinstance(diagnostic, dict):
        return None
    stripped = diagnostic.get("generated_text_stripped_token_ids")
    if isinstance(stripped, list) and stripped:
        return stripped[0]
    generated = diagnostic.get("generated_text_token_ids")
    if isinstance(generated, list) and generated:
        return generated[0]
    return None


def _expected_token_text_ids(mismatch: dict[str, object]) -> object:
    diagnostic = mismatch.get("tokenization_diagnostic")
    if not isinstance(diagnostic, dict):
        return None
    return diagnostic.get("expected_next_token_text_token_ids")


def _finding_active(diagnosis: dict[str, object], finding_name: str) -> object:
    findings = diagnosis.get("active_findings")
    if not isinstance(findings, list):
        return None
    for item in findings:
        if isinstance(item, dict) and item.get("finding") == finding_name:
            return item.get("active")
    return None


def _cause_ruled_out(diagnosis: dict[str, object], cause_name: str) -> object:
    causes = diagnosis.get("ruled_out_causes")
    if not isinstance(causes, list):
        return None
    for item in causes:
        if isinstance(item, dict) and item.get("cause") == cause_name:
            return item.get("ruled_out")
    return None


def _add_check(
    checks: list[dict[str, object]],
    *,
    name: str,
    passed: bool,
    evidence: str,
    expected: object | None = None,
    actual: object | None = None,
) -> None:
    record: dict[str, object] = {
        "name": name,
        "passed": passed,
        "evidence": evidence,
    }
    if expected is not None:
        record["expected"] = expected
    if actual is not None:
        record["actual"] = actual
    checks.append(record)


def build_oracle_evidence_consistency_check(
    *,
    diagnosis_artifact: Path = diagnosis_mod.DEFAULT_OUTPUT,
    backend_matrix_artifact: Path = backend_matrix.DEFAULT_OUTPUT,
    token_mismatch_artifact: Path = token_mismatch.DEFAULT_OUTPUT,
    rank_check_artifact: Path = rank_check.DEFAULT_OUTPUT,
    top_token_roundtrip_artifact: Path = top_roundtrip.DEFAULT_OUTPUT,
    prompt_token_roundtrip_artifact: Path = prompt_roundtrip.DEFAULT_OUTPUT,
    artifact_date: str = DEFAULT_ARTIFACT_DATE,
) -> dict[str, object]:
    """Return the cross-artifact oracle evidence consistency report."""

    diagnosis = _load_json_object(diagnosis_artifact)
    backend = _load_json_object(backend_matrix_artifact)
    mismatch = _load_json_object(token_mismatch_artifact)
    rank = _load_json_object(rank_check_artifact)
    top = _load_json_object(top_token_roundtrip_artifact)
    prompt = _load_json_object(prompt_token_roundtrip_artifact)

    artifact_payloads = {
        "backend_matrix": (backend_matrix_artifact, backend),
        "token_mismatch": (token_mismatch_artifact, mismatch),
        "rank_check": (rank_check_artifact, rank),
        "top_token_roundtrip": (top_token_roundtrip_artifact, top),
        "prompt_token_roundtrip": (prompt_token_roundtrip_artifact, prompt),
    }
    checks: list[dict[str, object]] = []
    for artifact_key, (path, payload) in artifact_payloads.items():
        live_sha = status_mod._stable_json_sha256(payload)
        diagnosis_ref = _diagnosis_artifact_ref(diagnosis, artifact_key)
        expected_sha = diagnosis_ref.get("sha256")
        _add_check(
            checks,
            name=f"diagnosis_{artifact_key}_sha_matches_live_artifact",
            passed=expected_sha == live_sha,
            expected=expected_sha,
            actual=live_sha,
            evidence=f"{path} stable SHA matches the diagnosis evidence_artifacts reference.",
        )

    expected_id = mismatch.get("expected_next_token_id")
    rank_expected_id = rank.get("expected_next_token_id")
    _add_check(
        checks,
        name="expected_next_token_id_consistent",
        passed=expected_id == rank_expected_id,
        expected=expected_id,
        actual=rank_expected_id,
        evidence="token-mismatch and rank-check artifacts agree on expected_next_token_id.",
    )
    expected_text = mismatch.get("expected_next_token_text")
    rank_expected_text = rank.get("expected_next_token_text")
    _add_check(
        checks,
        name="expected_next_token_text_consistent",
        passed=expected_text == rank_expected_text,
        expected=expected_text,
        actual=rank_expected_text,
        evidence="token-mismatch and rank-check artifacts agree on expected_next_token_text.",
    )
    token_text_ids = _expected_token_text_ids(mismatch)
    _add_check(
        checks,
        name="expected_next_token_text_roundtrips_to_expected_id",
        passed=token_text_ids == [expected_id],
        expected=[expected_id],
        actual=token_text_ids,
        evidence="expected next-token text tokenizes as the expected token id.",
    )
    generated_first = rank.get("generated_first_token_id")
    mismatch_first = _first_stripped_generated_token_id(mismatch)
    _add_check(
        checks,
        name="generated_first_token_id_consistent",
        passed=generated_first == mismatch_first,
        expected=mismatch_first,
        actual=generated_first,
        evidence="rank-check generated_first_token_id matches token-mismatch generated text tokenization.",
    )
    host_top_ids = rank.get("host_top_token_ids")
    top_host_ids = top.get("host_top_token_ids")
    _add_check(
        checks,
        name="host_top_token_ids_consistent",
        passed=host_top_ids == top_host_ids,
        expected=top_host_ids,
        actual=host_top_ids,
        evidence="rank-check and host top-token round-trip artifacts use the same host top-token ids.",
    )
    prompt_ok = prompt.get("prompt_tokenization_matches_host_input_ids") is True
    _add_check(
        checks,
        name="prompt_token_roundtrip_ready",
        passed=prompt_ok,
        expected=True,
        actual=prompt.get("prompt_tokenization_matches_host_input_ids"),
        evidence="prompt-token round-trip artifact reports host input_ids match llama-tokenize.",
    )
    top_ok = top.get("all_top_token_texts_roundtrip") is True
    _add_check(
        checks,
        name="top_token_roundtrip_ready",
        passed=top_ok,
        expected=True,
        actual=top.get("all_top_token_texts_roundtrip"),
        evidence="host top-token round-trip artifact reports all top-token texts match their token ids.",
    )
    generated_absent = rank.get("generated_token_in_host_top_tokens") is False
    _add_check(
        checks,
        name="generated_token_absent_from_host_top_tokens_consistent",
        passed=generated_absent
        and _finding_active(diagnosis, "generated_token_absent_from_host_top_tokens")
        is True,
        expected={
            "generated_token_in_host_top_tokens": False,
            "diagnosis_finding_active": True,
        },
        actual={
            "generated_token_in_host_top_tokens": rank.get(
                "generated_token_in_host_top_tokens"
            ),
            "diagnosis_finding_active": _finding_active(
                diagnosis, "generated_token_absent_from_host_top_tokens"
            ),
        },
        evidence="rank-check absence agrees with the diagnosis active finding.",
    )
    generated_text_mismatch = mismatch.get("text_matches_expected_stripped") is False
    _add_check(
        checks,
        name="generated_text_mismatch_consistent",
        passed=generated_text_mismatch
        and _finding_active(diagnosis, "generated_text_mismatch") is True,
        expected={"text_matches_expected_stripped": False, "diagnosis_finding_active": True},
        actual={
            "text_matches_expected_stripped": mismatch.get(
                "text_matches_expected_stripped"
            ),
            "diagnosis_finding_active": _finding_active(
                diagnosis, "generated_text_mismatch"
            ),
        },
        evidence="token-mismatch generated-text failure agrees with the diagnosis active finding.",
    )
    _add_check(
        checks,
        name="hip_oracle_timeout_consistent",
        passed=diagnosis_mod._hip_outcome(backend) == "timeout"
        and _finding_active(diagnosis, "hip_oracle_timeout") is True,
        expected={"hip_outcome": "timeout", "diagnosis_finding_active": True},
        actual={
            "hip_outcome": diagnosis_mod._hip_outcome(backend),
            "diagnosis_finding_active": _finding_active(diagnosis, "hip_oracle_timeout"),
        },
        evidence="backend matrix HIP timeout agrees with the diagnosis active finding.",
    )
    _add_check(
        checks,
        name="diagnosis_ruled_out_causes_consistent",
        passed=(
            _cause_ruled_out(diagnosis, "prompt_token_drift") is True
            and _cause_ruled_out(diagnosis, "host_top_token_text_label_drift") is True
            and _cause_ruled_out(
                diagnosis, "expected_next_token_text_tokenization_drift"
            )
            is True
            and diagnosis.get("all_tokenizer_prompt_drift_causes_ruled_out") is True
        ),
        expected=True,
        actual={
            "prompt_token_drift": _cause_ruled_out(diagnosis, "prompt_token_drift"),
            "host_top_token_text_label_drift": _cause_ruled_out(
                diagnosis, "host_top_token_text_label_drift"
            ),
            "expected_next_token_text_tokenization_drift": _cause_ruled_out(
                diagnosis, "expected_next_token_text_tokenization_drift"
            ),
            "all_tokenizer_prompt_drift_causes_ruled_out": diagnosis.get(
                "all_tokenizer_prompt_drift_causes_ruled_out"
            ),
        },
        evidence="diagnosis records the three tokenizer/prompt drift causes as ruled out.",
    )
    inconsistencies = [check for check in checks if check.get("passed") is not True]
    status = "match" if not inconsistencies else "mismatch"
    return {
        "schema_version": 1,
        "artifact_kind": "stepfun_oracle_evidence_consistency_check",
        "date": artifact_date,
        "status": status,
        "ready": not inconsistencies,
        "diagnosis_artifact": _artifact_ref(diagnosis_artifact, diagnosis),
        "checked_artifacts": {
            key: _artifact_ref(path, payload)
            for key, (path, payload) in artifact_payloads.items()
        },
        "check_count": len(checks),
        "passed_check_count": len(checks) - len(inconsistencies),
        "inconsistency_count": len(inconsistencies),
        "checks": checks,
        "inconsistencies": inconsistencies,
        "blocked_gates": ["oracle_parity", "kv_backed_decode", "e2e_inference"],
        "oracle_parity_ready": False,
        "next_action": diagnosis.get("next_action"),
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "This artifact only checks retained oracle evidence consistency; "
                "generated_text_matches_target remains unresolved."
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.default_output and args.output is not None:
        raise SystemExit("Use either --output or --default-output, not both")
    report = build_oracle_evidence_consistency_check(
        diagnosis_artifact=args.diagnosis_artifact,
        backend_matrix_artifact=args.backend_matrix_artifact,
        token_mismatch_artifact=args.token_mismatch_artifact,
        rank_check_artifact=args.rank_check_artifact,
        top_token_roundtrip_artifact=args.top_token_roundtrip_artifact,
        prompt_token_roundtrip_artifact=args.prompt_token_roundtrip_artifact,
        artifact_date=args.artifact_date,
    )
    if args.status_only:
        payload: object = report["status"]
    elif args.inconsistencies_only:
        payload = report["inconsistencies"]
    elif args.checks_only:
        payload = report["checks"]
    elif args.sha_only:
        payload = status_mod._stable_json_sha256(report)
    else:
        payload = report
    output = DEFAULT_OUTPUT if args.default_output else args.output
    status_mod._emit_json(payload, pretty=args.pretty, output=output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
