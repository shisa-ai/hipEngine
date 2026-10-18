#!/usr/bin/env python3
"""Audit that speculative counters describe the cycles that actually ran.

Checks, per response:

- ``speculative_mtp.used`` is attributable: a response that claims MTP usage
  names the provider that produced it, and one that does not claim usage does
  not carry MTP draft counts.
- ``draft_tokens``, ``accepted_draft_tokens`` and ``rejected_draft_tokens`` agree
  with the per-cycle records the backend published, and
  ``rejected = draft - accepted``.
- ``acceptance_rate`` is the ratio those counts imply, or ``null`` when no draft
  was proposed.
- ``usage.completion_tokens_details.accepted_prediction_tokens`` and
  ``rejected_prediction_tokens`` agree with the compact summary.
- A plain (non-speculative) request claims no MTP usage.

Usage::

    python scripts/qwen4exp_speculative_counter_audit.py \
        --url http://127.0.0.1:18392 --output /tmp/spec_counters.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx


def _check(checks: list[dict[str, Any]], name: str, passed: bool, **details: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), **details})


def _dig(payload: Any, *path: str) -> Any:
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _cycles(choice: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = _dig(choice, "hipengine", "diagnostics") or {}
    cycles = diagnostics.get("cycles")
    return [cycle for cycle in cycles or () if isinstance(cycle, dict)]


def _provider(choice: dict[str, Any], body: dict[str, Any]) -> str | None:
    for container in (
        _dig(choice, "hipengine", "diagnostics"),
        _dig(body, "hipengine", "diagnostics"),
    ):
        if isinstance(container, dict) and container.get("speculative_provider"):
            return str(container["speculative_provider"])
    return None


def audit_mtp(body: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    choice = (body.get("choices") or [{}])[0]
    summary = _dig(body, "hipengine", "speculative_mtp") or {}
    cycles = _cycles(choice)
    provider = _provider(choice, body)
    used = bool(summary.get("used"))
    draft = int(summary.get("draft_tokens") or 0)
    accepted = int(summary.get("accepted_draft_tokens") or 0)
    rejected = int(summary.get("rejected_draft_tokens") or 0)
    rate = summary.get("acceptance_rate")

    _check(checks, "mtp.summary_present", bool(summary), summary=summary)
    _check(checks, "mtp.used_true", used is True, used=used)
    _check(checks, "mtp.used_has_provider", (not used) or provider is not None,
           used=used, provider=provider)
    _check(checks, "mtp.used_has_cycles", (not used) or bool(cycles),
           used=used, cycle_count=len(cycles))
    cycle_draft = sum(len(cycle.get("candidates") or ()) for cycle in cycles)
    cycle_accepted = sum(int(cycle.get("accepted") or 0) for cycle in cycles)
    _check(checks, "mtp.draft_matches_cycles", cycle_draft == draft,
           cycle_draft=cycle_draft, summary_draft=draft)
    _check(checks, "mtp.accepted_matches_cycles", cycle_accepted == accepted,
           cycle_accepted=cycle_accepted, summary_accepted=accepted)
    _check(checks, "mtp.rejected_is_the_remainder", rejected == max(0, draft - accepted),
           draft=draft, accepted=accepted, rejected=rejected)
    _check(checks, "mtp.accepted_within_draft", 0 <= accepted <= draft,
           draft=draft, accepted=accepted)
    if draft:
        _check(checks, "mtp.acceptance_rate_matches_counts",
               rate is not None and abs(float(rate) - accepted / draft) < 1e-9,
               rate=rate, implied=accepted / draft)
    else:
        _check(checks, "mtp.acceptance_rate_null_without_drafts", rate is None, rate=rate)
    details = _dig(body, "usage", "completion_tokens_details") or {}
    usage_accepted = details.get("accepted_prediction_tokens")
    usage_rejected = details.get("rejected_prediction_tokens")
    _check(checks, "mtp.usage_accepted_matches_summary",
           usage_accepted is not None and int(usage_accepted) == accepted,
           usage_accepted=usage_accepted, summary_accepted=accepted)
    _check(checks, "mtp.usage_rejected_matches_summary",
           usage_rejected is not None and int(usage_rejected) == rejected,
           usage_rejected=usage_rejected, summary_rejected=rejected)
    return checks


def audit_plain(body: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    choice = (body.get("choices") or [{}])[0]
    summary = _dig(body, "hipengine", "speculative_mtp")
    # A plain request may carry the compact summary; what it must never do is
    # claim speculative work it did not do.
    if summary is None:
        _check(checks, "plain.mtp_summary_absent_or_silent", True, summary=None)
    else:
        _check(
            checks,
            "plain.mtp_summary_absent_or_silent",
            summary.get("used") is False
            and int(summary.get("draft_tokens") or 0) == 0
            and int(summary.get("accepted_draft_tokens") or 0) == 0
            and int(summary.get("draft_cycles") or 0) == 0,
            summary=summary,
        )
    details = _dig(body, "usage", "completion_tokens_details") or {}
    _check(checks, "plain.no_accepted_prediction_tokens",
           details.get("accepted_prediction_tokens") is None,
           details=details)
    _check(checks, "plain.no_mtp_cycles", not _cycles(choice), cycles=_cycles(choice))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18392")
    parser.add_argument("--model", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mtp-requests", type=int, default=3)
    parser.add_argument("--plain-requests", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=40)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_speculative_counter_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": args.url,
        "checks": [],
        "rows": [],
    }
    checks: list[dict[str, Any]] = report["checks"]
    with httpx.Client(base_url=args.url, timeout=httpx.Timeout(900.0)) as client:
        ready = client.get("/ready").json()
        model = args.model or str((ready.get("model") or {}).get("id") or "")
        report["model"] = model
        if not model:
            raise SystemExit("cannot resolve the served model id; pass --model")
        prompts = (
            "Explain in two sentences why a hash table lookup is usually constant time.",
            "Write one sentence about why batching improves GPU utilization.",
            "Name three properties of a stable sorting algorithm.",
            "Summarize what a KV cache stores during decoding.",
            "Describe one risk of long-context inference.",
        )
        for index in range(int(args.mtp_requests) + int(args.plain_requests)):
            mtp = index < int(args.mtp_requests)
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompts[index % len(prompts)]}],
                "max_tokens": int(args.tokens),
                "temperature": 0.0,
            }
            if mtp:
                payload["speculative_mtp"] = True
            response = client.post("/v1/chat/completions", json=payload)
            body = response.json()
            row_checks: list[dict[str, Any]] = []
            if mtp:
                row_checks.extend(audit_mtp(body))
            else:
                row_checks.extend(audit_plain(body))
            choice = (body.get("choices") or [{}])[0]
            report["rows"].append({
                "kind": "speculative_mtp" if mtp else "plain",
                "status": response.status_code,
                "finish_reason": choice.get("finish_reason"),
                "text": (choice.get("message") or {}).get("content"),
                "speculative_mtp": _dig(body, "hipengine", "speculative_mtp"),
                "provider": _provider(choice, body),
                "cycles": _cycles(choice),
                "completion_tokens_details": _dig(body, "usage", "completion_tokens_details"),
                "checks": row_checks,
            })
            checks.extend(
                {"row": index, **check} for check in row_checks
            )

    failures = [item for item in checks if not item["passed"]]
    report["status"] = "passed" if not failures else "failed"
    report["check_count"] = len(checks)
    report["failure_count"] = len(failures)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": len(checks),
                      "failures": len(failures)}))
    for item in failures:
        print(json.dumps({"failed_check": item}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
