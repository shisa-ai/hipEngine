#!/usr/bin/env python3
"""Validate a StepFun llama.cpp oracle-success artifact.

This checker is intentionally mechanical and conservative: it validates a
retained JSON artifact from a llama.cpp one-token oracle run against the
canonical deterministic StepFun prompt target. Passing this checker can satisfy
only the ``llama_cpp_oracle_success_artifact`` evidence item; KV-backed decode,
e2e readiness, and performance/throughput claims remain separate gates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stepfun_correctness_status as status_mod

ORACLE_CHECK_SCHEMA_VERSION = 1
PASSED_EXIT_CODE = 0
FAILED_EXIT_CODE = 2
DEFAULT_LOGIT_ATOL = 1.0e-5
_SUCCESS_STATUSES = {"executed", "passed", "success", "completed"}
_BLOCKED_ORACLE_KINDS = {
    "llama_cpp_missing_step35_architecture",
    "llama_cpp_oracle_timeout",
    "llama_cpp_oracle_returncode_nonzero",
}


def _write_text_atomic(output: Path, text: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, output)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _emit_json(payload: object, *, pretty: bool, output: Path | None) -> None:
    text = json.dumps(payload, indent=2 if pretty else None, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
        return
    _write_text_atomic(output, text)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="Retained llama.cpp oracle JSON artifact to validate.",
    )
    parser.add_argument(
        "--prompt-artifact",
        type=Path,
        default=status_mod.DEFAULT_PROMPT_ARTIFACT,
        help="Canonical StepFun prompt/logit artifact providing expected token/text.",
    )
    parser.add_argument(
        "--logit-atol",
        type=float,
        default=DEFAULT_LOGIT_ATOL,
        help="Absolute tolerance for expected top-token logit metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON output atomically to this path instead of stdout.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Emit only the compact oracle_summary payload.",
    )
    parser.add_argument(
        "--sha-only",
        action="store_true",
        help="Emit only the stable SHA-256 digest of the full report or compact summary.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Emit only passed/failed status.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Return exit code 2 when any required oracle evidence is missing or mismatched.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _expected_target(prompt: dict[str, object]) -> dict[str, object]:
    return {
        "prompt_length": _as_int(prompt.get("prompt_length")),
        "next_token_id": _as_int(prompt.get("next_token_id")),
        "next_token_text": prompt.get("next_token_text"),
        "next_token_logit": _as_float(prompt.get("next_token_logit")),
        "prompt_sha256": status_mod._stable_json_sha256(prompt),
    }


def _top_token(oracle: dict[str, object]) -> dict[str, object]:
    top_tokens = oracle.get("expected_top_tokens")
    if not isinstance(top_tokens, list) or not top_tokens:
        return {}
    first = top_tokens[0]
    return dict(first) if isinstance(first, dict) else {}


def _observed_oracle(oracle: dict[str, object]) -> dict[str, object]:
    generated_text = oracle.get("generated_text")
    if generated_text in (None, "") and oracle.get("stdout") not in (None, ""):
        generated_text = oracle.get("stdout")
    top_token = _top_token(oracle)
    return {
        "status": oracle.get("status"),
        "returncode": oracle.get("returncode"),
        "llama_cli": oracle.get("llama_cli"),
        "llama_cpp_version": oracle.get("llama_cpp_version"),
        "model": oracle.get("model"),
        "command_shell": oracle.get("command_shell"),
        "prompt_length": _as_int(oracle.get("prompt_length")),
        "n_predict": _as_int(oracle.get("n_predict")),
        "expected_next_token_id": _as_int(oracle.get("expected_next_token_id")),
        "expected_next_token_text": oracle.get("expected_next_token_text"),
        "expected_next_token_logit": _as_float(oracle.get("expected_next_token_logit")),
        "top_token_id": _as_int(top_token.get("token_id")),
        "top_token_text": top_token.get("token_text"),
        "top_token_logit": _as_float(top_token.get("logit")),
        "generated_text": generated_text,
        "generated_text_len": len(str(generated_text or "")),
        "stdout_len": len(str(oracle.get("stdout", ""))),
        "stderr_len": len(str(oracle.get("stderr", ""))),
        "text_matches_expected_exact": oracle.get("text_matches_expected_exact") is True,
        "text_matches_expected_stripped": oracle.get("text_matches_expected_stripped") is True,
        "oracle_blocker_kind": oracle.get("oracle_blocker_kind"),
        "oracle_blocker_detail": oracle.get("oracle_blocker_detail"),
        "step35_supported": oracle.get("step35_supported"),
        "timeout_termination": oracle.get("timeout_termination"),
        "artifact_sha256": status_mod._stable_json_sha256(oracle),
    }


def _check_record(
    name: str,
    ready: bool,
    *,
    required_evidence: str,
    current: object,
) -> dict[str, object]:
    return {
        "name": name,
        "ready": ready,
        "required_evidence": required_evidence,
        "current": current,
    }


def _build_evidence_checks(
    *,
    observed: dict[str, object],
    expected: dict[str, object],
    logit_atol: float,
) -> list[dict[str, object]]:
    expected_logit = expected.get("next_token_logit")
    top_logit = observed.get("top_token_logit")
    metadata_logit = observed.get("expected_next_token_logit")
    top_logit_abs_error = None
    metadata_logit_abs_error = None
    if isinstance(top_logit, float) and isinstance(expected_logit, float):
        top_logit_abs_error = abs(top_logit - expected_logit)
    if isinstance(metadata_logit, float) and isinstance(expected_logit, float):
        metadata_logit_abs_error = abs(metadata_logit - expected_logit)
    blocker_kind = observed.get("oracle_blocker_kind")
    return [
        _check_record(
            "oracle_success_status",
            str(observed.get("status")) in _SUCCESS_STATUSES,
            required_evidence="oracle artifact status must be executed/passed/success/completed",
            current=observed.get("status"),
        ),
        _check_record(
            "oracle_returncode_zero",
            observed.get("returncode") == 0,
            required_evidence="llama.cpp oracle process must complete with returncode=0",
            current=observed.get("returncode"),
        ),
        _check_record(
            "oracle_binary_metadata_recorded",
            all(
                bool(observed.get(key))
                for key in ("llama_cli", "llama_cpp_version", "model", "command_shell")
            ),
            required_evidence="llama-cli path/version, model path, and command must be recorded",
            current={
                "llama_cli": observed.get("llama_cli"),
                "llama_cpp_version": observed.get("llama_cpp_version"),
                "model": observed.get("model"),
                "command_shell": observed.get("command_shell"),
            },
        ),
        _check_record(
            "step35_supported_by_oracle",
            observed.get("step35_supported") is not False
            and blocker_kind != "llama_cpp_missing_step35_architecture",
            required_evidence="oracle artifact must not record a Step35 architecture rejection",
            current={
                "step35_supported": observed.get("step35_supported"),
                "oracle_blocker_kind": blocker_kind,
            },
        ),
        _check_record(
            "no_timeout_or_oracle_blocker",
            blocker_kind in (None, "") and observed.get("timeout_termination") in (None, {}),
            required_evidence="oracle artifact must not be a timeout or blocked oracle attempt",
            current={
                "oracle_blocker_kind": blocker_kind,
                "blocked_kind_known": blocker_kind in _BLOCKED_ORACLE_KINDS,
                "timeout_termination": observed.get("timeout_termination"),
            },
        ),
        _check_record(
            "prompt_length_matches_target",
            observed.get("prompt_length") == expected.get("prompt_length"),
            required_evidence="oracle prompt_length must match the canonical prompt artifact",
            current={
                "observed_prompt_length": observed.get("prompt_length"),
                "expected_prompt_length": expected.get("prompt_length"),
            },
        ),
        _check_record(
            "n_predict_one",
            observed.get("n_predict") == 1,
            required_evidence="oracle run must be a deterministic one-token run",
            current=observed.get("n_predict"),
        ),
        _check_record(
            "expected_token_metadata_matches_target",
            observed.get("expected_next_token_id") == expected.get("next_token_id")
            and observed.get("expected_next_token_text") == expected.get("next_token_text"),
            required_evidence="oracle artifact expected token metadata must match the canonical target",
            current={
                "observed_expected_next_token_id": observed.get("expected_next_token_id"),
                "expected_next_token_id": expected.get("next_token_id"),
                "observed_expected_next_token_text": observed.get("expected_next_token_text"),
                "expected_next_token_text": expected.get("next_token_text"),
            },
        ),
        _check_record(
            "top_token_metadata_matches_target",
            observed.get("top_token_id") == expected.get("next_token_id")
            and observed.get("top_token_text") == expected.get("next_token_text"),
            required_evidence="rank-1 oracle token metadata must match the deterministic target",
            current={
                "top_token_id": observed.get("top_token_id"),
                "expected_next_token_id": expected.get("next_token_id"),
                "top_token_text": observed.get("top_token_text"),
                "expected_next_token_text": expected.get("next_token_text"),
            },
        ),
        _check_record(
            "top_token_logit_matches_target",
            top_logit_abs_error is not None
            and math.isfinite(top_logit_abs_error)
            and top_logit_abs_error <= logit_atol
            and metadata_logit_abs_error is not None
            and math.isfinite(metadata_logit_abs_error)
            and metadata_logit_abs_error <= logit_atol,
            required_evidence="oracle expected/top-token logit metadata must match the canonical target",
            current={
                "top_token_logit": top_logit,
                "expected_next_token_logit": expected_logit,
                "top_logit_abs_error": top_logit_abs_error,
                "metadata_logit_abs_error": metadata_logit_abs_error,
                "logit_atol": logit_atol,
            },
        ),
        _check_record(
            "generated_text_nonempty",
            int(observed.get("generated_text_len") or 0) > 0,
            required_evidence="oracle artifact must capture non-empty generated text",
            current={
                "generated_text_len": observed.get("generated_text_len"),
                "stdout_len": observed.get("stdout_len"),
                "stderr_len": observed.get("stderr_len"),
            },
        ),
        _check_record(
            "generated_text_matches_target",
            observed.get("generated_text") == expected.get("next_token_text")
            and observed.get("text_matches_expected_exact") is True
            and observed.get("text_matches_expected_stripped") is True,
            required_evidence="oracle generated text must exactly match expected_next_token_text",
            current={
                "generated_text": observed.get("generated_text"),
                "expected_next_token_text": expected.get("next_token_text"),
                "text_matches_expected_exact": observed.get("text_matches_expected_exact"),
                "text_matches_expected_stripped": observed.get("text_matches_expected_stripped"),
            },
        ),
    ]


def build_oracle_check_report(
    artifact: Path,
    *,
    prompt_artifact: Path = status_mod.DEFAULT_PROMPT_ARTIFACT,
    logit_atol: float = DEFAULT_LOGIT_ATOL,
) -> dict[str, object]:
    """Return a mechanical validation report for a llama.cpp oracle artifact."""

    if logit_atol < 0 or not math.isfinite(logit_atol):
        raise ValueError("--logit-atol must be a finite non-negative number")
    oracle_payload = _load_json_object(artifact)
    prompt_payload = _load_json_object(prompt_artifact)
    expected = _expected_target(prompt_payload)
    observed = _observed_oracle(oracle_payload)
    evidence_checks = _build_evidence_checks(
        observed=observed,
        expected=expected,
        logit_atol=logit_atol,
    )
    missing_evidence = [
        str(record["name"]) for record in evidence_checks if record.get("ready") is not True
    ]
    ready = not missing_evidence
    summary = {
        "schema_version": ORACLE_CHECK_SCHEMA_VERSION,
        "status": "passed" if ready else "failed",
        "ready": ready,
        "artifact": str(artifact),
        "prompt_artifact": str(prompt_artifact),
        "prompt_artifact_sha256": status_mod._stable_json_sha256(prompt_payload),
        "artifact_sha256": observed["artifact_sha256"],
        "expected_next_token_id": expected["next_token_id"],
        "expected_next_token_text": expected["next_token_text"],
        "expected_next_token_logit": expected["next_token_logit"],
        "oracle_status": observed["status"],
        "oracle_returncode": observed["returncode"],
        "oracle_blocker_kind": observed["oracle_blocker_kind"],
        "step35_supported": observed["step35_supported"],
        "generated_text": observed["generated_text"],
        "generated_text_len": observed["generated_text_len"],
        "text_matches_expected_exact": observed["text_matches_expected_exact"],
        "text_matches_expected_stripped": observed["text_matches_expected_stripped"],
        "missing_evidence": missing_evidence,
        "missing_evidence_count": len(missing_evidence),
        "evidence_checks_sha256": status_mod._stable_json_sha256(evidence_checks),
        "no_claim_policy": {
            "llama_cpp_oracle_success_artifact_claim_allowed": ready,
            "oracle_parity_claim_allowed": ready,
            "kv_backed_decode_claim_allowed": False,
            "e2e_inference_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "Oracle validation checks only the retained llama.cpp one-token artifact; "
                "KV-backed decode and benchmark gates remain separate."
            ),
        },
    }
    report = {
        "schema_version": ORACLE_CHECK_SCHEMA_VERSION,
        "status": summary["status"],
        "oracle_summary": summary,
        "oracle_summary_sha256": status_mod._stable_json_sha256(summary),
        "expected_target": expected,
        "observed_oracle": observed,
        "evidence_checks": evidence_checks,
        "evidence_checks_sha256": status_mod._stable_json_sha256(evidence_checks),
        "readiness_impact": {
            "llama_cpp_oracle_success_artifact": ready,
            "oracle_parity": ready,
            "kv_backed_decode_ready": False,
            "e2e_inference_ready": False,
            "reason": (
                "This report can satisfy only the oracle parity side of StepFun e2e readiness; "
                "KV trace/token evidence must also pass."
            ),
        },
    }
    report["report_sha256"] = status_mod._stable_json_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_oracle_check_report(
        args.artifact,
        prompt_artifact=args.prompt_artifact,
        logit_atol=args.logit_atol,
    )
    if args.status_only:
        payload: object = report["status"]
    elif args.sha_only:
        payload = report["oracle_summary_sha256"] if args.summary_only else report["report_sha256"]
    elif args.summary_only:
        payload = report["oracle_summary"]
    else:
        payload = report
    _emit_json(payload, pretty=args.pretty, output=args.output)
    if args.fail_on_missing and report["status"] != "passed":
        return FAILED_EXIT_CODE
    return PASSED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
