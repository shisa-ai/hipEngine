#!/usr/bin/env python3
"""Audit that finish reasons and route fallbacks describe what happened.

Cases, each asserted against the response body rather than the request:

- ``max_tokens`` with ``ignore_eos`` finishes on ``length`` and reports exactly
  the budget it was given.
- A natural end finishes on EOS, with ``finish_details.reason == "eos"``.
- A requested stop string finishes on ``stop`` and names the string it matched.
- A sampled speculative request claims the speculation it ran: provider named,
  cycles present, and counts that agree with those cycles.

Usage::

    python scripts/qwen4exp_finish_route_audit.py \
        --url http://127.0.0.1:18392 --output /tmp/finish_route.json
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


def _finish(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    choice = (body.get("choices") or [{}])[0]
    return choice, (choice.get("finish_details") or {})


def audit_length(body: dict[str, Any], budget: int) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    choice, details = _finish(body)
    _check(checks, "length.finish_reason", choice.get("finish_reason") == "length",
           got=choice.get("finish_reason"))
    _check(checks, "length.finish_details_reason", details.get("reason") == "length",
           got=details.get("reason"))
    completion = _dig(body, "usage", "completion_tokens")
    _check(checks, "length.usage_equals_budget",
           completion is not None and int(completion) == int(budget),
           usage=completion, budget=budget)
    return checks


def audit_eos(body: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    choice, details = _finish(body)
    _check(checks, "eos.finish_reason", choice.get("finish_reason") == "stop",
           got=choice.get("finish_reason"))
    _check(checks, "eos.finish_details_reason", details.get("reason") == "eos",
           got=details.get("reason"))
    _check(checks, "eos.reports_the_eos_id", details.get("eos_token_id") is not None,
           finish_details=details)
    return checks


def audit_stop(body: dict[str, Any], stop: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    choice, details = _finish(body)
    _check(checks, "stop.finish_reason", choice.get("finish_reason") == "stop",
           got=choice.get("finish_reason"))
    _check(checks, "stop.finish_details_reason", details.get("reason") == "stop",
           got=details.get("reason"))
    _check(checks, "stop.names_the_matched_string", details.get("matched_stop") == stop,
           matched_stop=details.get("matched_stop"), expected=stop)
    # The server stop is the reported cause; a backend eos/length cause must not
    # survive beside it.
    _check(checks, "stop.no_contradictory_eos", "eos_token_id" not in details,
           finish_details=details)
    _check(checks, "stop.no_contradictory_length", "length_limit" not in details,
           finish_details=details)
    text = (choice.get("message") or {}).get("content") or ""
    _check(checks, "stop.text_excludes_the_stop_string", stop not in text, text=text)
    return checks


def audit_sampled_mtp(body: dict[str, Any]) -> list[dict[str, Any]]:
    """A sampled speculative request must describe the speculation it ran.

    Sampled MTP is implemented, so this case asserts that the response claims
    the work rather than inventing a fallback: a ``used:true`` claim needs a
    provider, cycles, and counts that agree with those cycles.
    """

    checks: list[dict[str, Any]] = []
    summary = _dig(body, "hipengine", "speculative_mtp") or {}
    choice = (body.get("choices") or [{}])[0]
    cycles = [
        cycle
        for cycle in (_dig(choice, "hipengine", "diagnostics", "cycles") or ())
        if isinstance(cycle, dict)
    ]
    provider = _dig(choice, "hipengine", "diagnostics", "speculative_provider")
    used = bool(summary.get("used"))
    draft = int(summary.get("draft_tokens") or 0)
    accepted = int(summary.get("accepted_draft_tokens") or 0)
    _check(checks, "sampled_mtp.used", used is True, summary=summary)
    _check(checks, "sampled_mtp.provider_named", bool(provider), provider=provider)
    _check(checks, "sampled_mtp.cycles_reported", bool(cycles), cycle_count=len(cycles))
    _check(checks, "sampled_mtp.draft_matches_cycles",
           sum(len(cycle.get("candidates") or ()) for cycle in cycles) == draft,
           draft=draft, cycles=cycles)
    _check(checks, "sampled_mtp.accepted_matches_cycles",
           sum(int(cycle.get("accepted") or 0) for cycle in cycles) == accepted,
           accepted=accepted)
    _check(checks, "sampled_mtp.no_false_fallback_reason",
           summary.get("decision_reason") in (None, "resident_dynamic_mtp"),
           decision_reason=summary.get("decision_reason"))
    return checks


def audit_ignore_eos(body: dict[str, Any]) -> list[dict[str, Any]]:
    """A raw continuation must not publish chat-template scaffolding as text."""

    checks: list[dict[str, Any]] = []
    choice, details = _finish(body)
    text = (choice.get("message") or {}).get("content") or ""
    leaked = sorted({marker for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
                     if marker in text})
    _check(checks, "ignore_eos.no_special_token_text", not leaked, leaked=leaked, tail=text[-80:])
    _check(checks, "ignore_eos.reason_is_length_or_stop",
           choice.get("finish_reason") in {"length", "stop"}, got=choice.get("finish_reason"))
    return checks


def audit_sse_usage(blocking: dict[str, Any], streaming: dict[str, Any]) -> list[dict[str, Any]]:
    """Streaming and blocking answers must report the same usage for one request."""

    checks: list[dict[str, Any]] = []
    left = blocking.get("usage") or {}
    right = streaming.get("usage") or {}
    _check(checks, "sse_usage.prompt_tokens_match",
           left.get("prompt_tokens") == right.get("prompt_tokens"),
           blocking=left.get("prompt_tokens"), streaming=right.get("prompt_tokens"))
    _check(checks, "sse_usage.completion_tokens_match",
           left.get("completion_tokens") == right.get("completion_tokens"),
           blocking=left.get("completion_tokens"), streaming=right.get("completion_tokens"))
    _check(checks, "sse_usage.total_tokens_match",
           left.get("total_tokens") == right.get("total_tokens"),
           blocking=left.get("total_tokens"), streaming=right.get("total_tokens"))
    left_text = ((blocking.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    right_text = "".join(
        ((choice.get("delta") or {}).get("content") or "")
        for choice in streaming.get("choices") or ()
    )
    _check(checks, "sse_usage.text_matches", left_text.strip() == right_text.strip(),
           blocking_text=left_text, streaming_text=right_text)
    _check(checks, "sse_usage.completion_tokens_counted",
           int(left.get("completion_tokens") or 0) > 0, completion_tokens=left.get("completion_tokens"))
    return checks


def _stream_usage(client: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Collect the terminal usage and text deltas from an SSE response."""

    usage: dict[str, Any] = {}
    choices: list[dict[str, Any]] = []
    # Usage is opt-in on a stream, exactly as the API documents.
    request = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    with client.stream("POST", "/v1/chat/completions", json=request) as response:
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            choices.extend(event.get("choices") or ())
    return {"usage": usage, "choices": choices}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18392")
    parser.add_argument("--model", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_finish_route_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": args.url,
        "cases": [],
        "checks": [],
    }
    checks: list[dict[str, Any]] = report["checks"]

    def run(name: str, payload: dict[str, Any], audit) -> None:
        with httpx.Client(base_url=args.url, timeout=httpx.Timeout(900.0)) as client:
            response = client.post("/v1/chat/completions", json=payload)
        body = response.json()
        case_checks: list[dict[str, Any]] = []
        if response.status_code == 200:
            case_checks = audit(body)
        else:
            case_checks = [{
                "name": f"{name}.http_200",
                "passed": False,
                "status": response.status_code,
                "body": body,
            }]
        choice, details = _finish(body)
        report["cases"].append({
            "case": name,
            "status": response.status_code,
            "finish_reason": choice.get("finish_reason"),
            "finish_details": details,
            "text": (choice.get("message") or {}).get("content"),
            "usage": body.get("usage"),
            "speculative_mtp": _dig(body, "hipengine", "speculative_mtp"),
            "checks": case_checks,
        })
        checks.extend({"case": name, **item} for item in case_checks)

    with httpx.Client(base_url=args.url, timeout=httpx.Timeout(900.0)) as client:
        ready = client.get("/ready").json()
    model = args.model or str((ready.get("model") or {}).get("id") or "")
    report["model"] = model
    if not model:
        raise SystemExit("cannot resolve the served model id; pass --model")

    budget = 5
    run(
        "length",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Count upward without stopping."}],
            "max_tokens": budget,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        lambda body: audit_length(body, budget),
    )
    run(
        "eos",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly one word: OMEGA"}],
            "max_tokens": 32,
            "temperature": 0.0,
        },
        audit_eos,
    )
    stop = "STOPHERE"
    run(
        "stop",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write the single line ALPHA BETA GAMMA STOPHERE DELTA and nothing else."
                    ),
                }
            ],
            "max_tokens": 48,
            "temperature": 0.0,
            "stop": [stop],
        },
        lambda body: audit_stop(body, stop),
    )
    run(
        "ignore_eos",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Count from 1 upward, one number per line."}],
            "temperature": 0.0,
            "max_tokens": 512,
            "ignore_eos": True,
        },
        audit_ignore_eos,
    )
    run(
        "sampled_mtp",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly one word: BRAVO"}],
            "max_tokens": 24,
            "temperature": 0.8,
            "top_p": 0.95,
            "speculative_mtp": True,
        },
        audit_sampled_mtp,
    )

    # Streaming and blocking must agree on what the request cost.
    usage_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 20, one number per line."}],
        "temperature": 0.0,
        "max_tokens": 96,
    }
    with httpx.Client(base_url=args.url, timeout=httpx.Timeout(900.0)) as client:
        blocking_body = client.post("/v1/chat/completions", json=usage_payload).json()
        streaming_body = _stream_usage(client, usage_payload)
    usage_checks = audit_sse_usage(blocking_body, streaming_body)
    report["cases"].append({
        "case": "sse_usage",
        "status": 200,
        "finish_reason": ((blocking_body.get("choices") or [{}])[0]).get("finish_reason"),
        "finish_details": ((blocking_body.get("choices") or [{}])[0]).get("finish_details"),
        "text": ((blocking_body.get("choices") or [{}])[0].get("message") or {}).get("content"),
        "usage": blocking_body.get("usage"),
        "streaming_usage": streaming_body.get("usage"),
        "checks": usage_checks,
    })
    checks.extend({"case": "sse_usage", **item} for item in usage_checks)

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
