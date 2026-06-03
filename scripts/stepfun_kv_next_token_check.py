#!/usr/bin/env python3
"""Validate a StepFun KV-backed next-token artifact.

The checker is intentionally mechanical and conservative: it validates a retained
JSON artifact from a real StepFun KV-backed one-token decode run against the
canonical deterministic prompt target. Passing this checker can satisfy only the
``kv_backed_next_token_artifact`` evidence item; oracle parity, KV trace evidence,
and performance/throughput claims remain separate gates.
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

NEXT_TOKEN_CHECK_SCHEMA_VERSION = 1
PASSED_EXIT_CODE = 0
FAILED_EXIT_CODE = 2
DEFAULT_LOGIT_ATOL = 1.0e9
_SUCCESS_STATUSES = {"passed", "success", "completed", "executed"}
_KV_RUNTIME_PATHS = {
    "kv_backed_decode",
    "resident_kv_backed_decode",
    "streaming_kv_decode",
    "streaming_decode_loop",
    "stepfun_kv_backed_decode",
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
        help="Retained StepFun KV-backed next-token JSON artifact to validate.",
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
        help=(
            "Absolute tolerance for optional next-token logit comparison. Default is intentionally "
            "wide because exact logit parity is a later oracle/correctness gate; this checker still "
            "requires a finite logit to be recorded."
        ),
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
        help="Emit only the compact next_token_summary payload.",
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
        help="Return exit code 2 when any required next-token evidence is missing or mismatched.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args(argv)


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _nested_get(payload: dict[str, object], path: tuple[str, ...]) -> object:
    current: object = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_present(payload: dict[str, object], paths: Sequence[tuple[str, ...]]) -> object:
    for path in paths:
        value = _nested_get(payload, path)
        if value not in (None, ""):
            return value
    return None


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


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _string_set(values: Sequence[object]) -> set[str]:
    return {str(value) for value in values if value not in (None, "")}


def _expected_target(prompt: dict[str, object]) -> dict[str, object]:
    return {
        "prompt_length": _as_int(prompt.get("prompt_length")),
        "next_token_id": _as_int(prompt.get("next_token_id")),
        "next_token_text": prompt.get("next_token_text"),
        "next_token_logit": _as_float(prompt.get("next_token_logit")),
        "prompt_sha256": status_mod._stable_json_sha256(prompt),
    }


def _observed_artifact(payload: dict[str, object]) -> dict[str, object]:
    return {
        "status": payload.get("status"),
        "prompt_length": _as_int(
            _first_present(
                payload,
                (("prompt_length",), ("prompt", "prompt_length"), ("inputs", "prompt_length")),
            )
        ),
        "next_token_id": _as_int(
            _first_present(
                payload,
                (
                    ("next_token_id",),
                    ("generated_token_id",),
                    ("token_id",),
                    ("final_token_id",),
                    ("result", "next_token_id"),
                    ("correctness_sanity", "final_token_id"),
                ),
            )
        ),
        "next_token_text": _first_present(
            payload,
            (
                ("next_token_text",),
                ("generated_token_text",),
                ("token_text",),
                ("final_token_text",),
                ("result", "next_token_text"),
            ),
        ),
        "next_token_logit": _as_float(
            _first_present(
                payload,
                (
                    ("next_token_logit",),
                    ("generated_token_logit",),
                    ("logit",),
                    ("final_logit",),
                    ("result", "next_token_logit"),
                    ("correctness_sanity", "final_logit"),
                ),
            )
        ),
        "execution_path": _first_present(
            payload,
            (("execution_path",), ("runtime_path",), ("decode_path",), ("source",), ("mode",)),
        ),
        "kv_backed_decode": _as_bool(payload.get("kv_backed_decode")),
        "kv_cache_used": _as_bool(payload.get("kv_cache_used")),
        "streaming_runner_ready": _as_bool(payload.get("streaming_runner_ready")),
        "host_composed_layer_prefix": _as_bool(payload.get("host_composed_layer_prefix")),
        "host_composed": _as_bool(payload.get("host_composed")),
        "layer_prefix_host_composed": _as_bool(payload.get("layer_prefix_host_composed")),
        "artifact_sha256": status_mod._stable_json_sha256(payload),
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
    status = observed.get("status")
    execution_path = observed.get("execution_path")
    runtime_markers = {
        "execution_path": execution_path,
        "kv_backed_decode": observed.get("kv_backed_decode"),
        "kv_cache_used": observed.get("kv_cache_used"),
    }
    host_composed_markers = {
        "host_composed_layer_prefix": observed.get("host_composed_layer_prefix"),
        "host_composed": observed.get("host_composed"),
        "layer_prefix_host_composed": observed.get("layer_prefix_host_composed"),
    }
    observed_logit = observed.get("next_token_logit")
    expected_logit = expected.get("next_token_logit")
    logit_abs_error = None
    if isinstance(observed_logit, float) and isinstance(expected_logit, float):
        logit_abs_error = abs(observed_logit - expected_logit)
    return [
        _check_record(
            "artifact_success_status",
            str(status) in _SUCCESS_STATUSES,
            required_evidence="artifact status must be passed/success/completed/executed",
            current=status,
        ),
        _check_record(
            "kv_backed_runtime_path",
            observed.get("kv_backed_decode") is True
            or observed.get("kv_cache_used") is True
            or str(execution_path) in _KV_RUNTIME_PATHS,
            required_evidence="artifact must explicitly identify a KV-backed decode/runtime path",
            current=runtime_markers,
        ),
        _check_record(
            "streaming_runner_ready",
            observed.get("streaming_runner_ready") is True,
            required_evidence="artifact must be produced by a ready streaming decode runner",
            current=observed.get("streaming_runner_ready"),
        ),
        _check_record(
            "not_host_composed_layer_prefix",
            True not in host_composed_markers.values(),
            required_evidence="artifact must not be host-composed layer-prefix output",
            current=host_composed_markers,
        ),
        _check_record(
            "prompt_length_matches_target",
            observed.get("prompt_length") == expected.get("prompt_length"),
            required_evidence="artifact prompt_length must match the canonical prompt artifact",
            current={
                "observed_prompt_length": observed.get("prompt_length"),
                "expected_prompt_length": expected.get("prompt_length"),
            },
        ),
        _check_record(
            "next_token_id_matches_target",
            observed.get("next_token_id") == expected.get("next_token_id"),
            required_evidence="artifact next token id must match the deterministic target",
            current={
                "observed_next_token_id": observed.get("next_token_id"),
                "expected_next_token_id": expected.get("next_token_id"),
            },
        ),
        _check_record(
            "next_token_text_matches_target",
            observed.get("next_token_text") == expected.get("next_token_text"),
            required_evidence="artifact next token text must match the deterministic target",
            current={
                "observed_next_token_text": observed.get("next_token_text"),
                "expected_next_token_text": expected.get("next_token_text"),
            },
        ),
        _check_record(
            "next_token_logit_recorded_finite",
            isinstance(observed_logit, float) and math.isfinite(observed_logit),
            required_evidence="artifact must record a finite next-token logit",
            current=observed_logit,
        ),
        _check_record(
            "next_token_logit_within_tolerance",
            logit_abs_error is not None and math.isfinite(logit_abs_error) and logit_abs_error <= logit_atol,
            required_evidence="artifact next-token logit must be within configured tolerance of the deterministic target",
            current={
                "observed_next_token_logit": observed_logit,
                "expected_next_token_logit": expected_logit,
                "logit_abs_error": logit_abs_error,
                "logit_atol": logit_atol,
            },
        ),
    ]


def build_next_token_check_report(
    artifact: Path,
    *,
    prompt_artifact: Path = status_mod.DEFAULT_PROMPT_ARTIFACT,
    logit_atol: float = DEFAULT_LOGIT_ATOL,
) -> dict[str, object]:
    """Return a mechanical validation report for a KV-backed next-token artifact."""

    if logit_atol < 0 or not math.isfinite(logit_atol):
        raise ValueError("--logit-atol must be a finite non-negative number")
    payload = _load_json_object(artifact)
    prompt_payload = _load_json_object(prompt_artifact)
    expected = _expected_target(prompt_payload)
    observed = _observed_artifact(payload)
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
        "schema_version": NEXT_TOKEN_CHECK_SCHEMA_VERSION,
        "status": "passed" if ready else "failed",
        "ready": ready,
        "artifact": str(artifact),
        "prompt_artifact": str(prompt_artifact),
        "prompt_artifact_sha256": status_mod._stable_json_sha256(prompt_payload),
        "artifact_sha256": observed["artifact_sha256"],
        "expected_next_token_id": expected["next_token_id"],
        "expected_next_token_text": expected["next_token_text"],
        "expected_next_token_logit": expected["next_token_logit"],
        "observed_next_token_id": observed["next_token_id"],
        "observed_next_token_text": observed["next_token_text"],
        "observed_next_token_logit": observed["next_token_logit"],
        "prompt_length": observed["prompt_length"],
        "expected_prompt_length": expected["prompt_length"],
        "execution_path": observed["execution_path"],
        "kv_backed_decode": observed["kv_backed_decode"],
        "kv_cache_used": observed["kv_cache_used"],
        "streaming_runner_ready": observed["streaming_runner_ready"],
        "missing_evidence": missing_evidence,
        "missing_evidence_count": len(missing_evidence),
        "evidence_checks_sha256": status_mod._stable_json_sha256(evidence_checks),
        "no_claim_policy": {
            "kv_backed_next_token_artifact_claim_allowed": ready,
            "kv_backed_decode_claim_allowed": False,
            "oracle_parity_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "KV next-token validation checks only the retained one-token artifact; "
                "KV trace, oracle parity, and benchmark gates remain separate."
            ),
        },
    }
    report = {
        "schema_version": NEXT_TOKEN_CHECK_SCHEMA_VERSION,
        "status": summary["status"],
        "next_token_summary": summary,
        "next_token_summary_sha256": status_mod._stable_json_sha256(summary),
        "expected_target": expected,
        "observed_artifact": observed,
        "evidence_checks": evidence_checks,
        "evidence_checks_sha256": status_mod._stable_json_sha256(evidence_checks),
        "readiness_impact": {
            "kv_backed_next_token_artifact": ready,
            "kv_backed_decode_ready": False,
            "e2e_inference_ready": False,
            "reason": (
                "This report can satisfy only the retained KV-backed next-token artifact; "
                "the streaming loop readiness, KV trace artifact, and oracle parity must also pass."
            ),
        },
    }
    report["report_sha256"] = status_mod._stable_json_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_next_token_check_report(
        args.artifact,
        prompt_artifact=args.prompt_artifact,
        logit_atol=args.logit_atol,
    )
    if args.status_only:
        payload: object = report["status"]
    elif args.sha_only:
        payload = report["next_token_summary_sha256"] if args.summary_only else report["report_sha256"]
    elif args.summary_only:
        payload = report["next_token_summary"]
    else:
        payload = report
    _emit_json(payload, pretty=args.pretty, output=args.output)
    if args.fail_on_missing and report["status"] != "passed":
        return FAILED_EXIT_CODE
    return PASSED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
