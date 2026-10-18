#!/usr/bin/env python3
"""Isolation and determinism gate over the served HTTP route.

Concurrency is only useful if a request's answer does not depend on who else is
in the batch. This gate proves that on the live server:

* solo answers are recorded one request at a time,
* the same prompts are replayed concurrently at the configured width,
* every answer must match its solo run exactly (text and token ids),
* a repeated request must reproduce itself,
* requests with different budgets keep their own `max_tokens`,
* two interleaved sessions must each answer from their own history.

A mismatch is a real isolation defect, not a numerical tolerance question: the
same prompt with the same sampling must produce the same tokens either way.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PROMPTS = (
    "Name the capital of France in one word.",
    "Name the capital of Japan in one word.",
    "Write the first eight prime numbers, comma separated.",
    "Reply with exactly: ISOLATION-OK",
    "List three primary colors, comma separated.",
    "What is 13 multiplied by 17? Answer with the number only.",
    "Write the word 'stable' five times separated by spaces.",
    "Name two planets that have rings, comma separated.",
)


def _check(checks: list[dict[str, Any]], name: str, passed: bool, **detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), **detail})


def _chat(client: Any, url: str, model: str, content: Any, **extra: Any) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": extra.pop("max_tokens", 48),
        **extra,
    }
    response = client.post(f"{url}/v1/chat/completions", json=payload)
    body = response.json()
    if response.status_code != 200:
        return {"status": response.status_code, "body": body}
    choice = (body.get("choices") or [{}])[0]
    return {
        "status": 200,
        "text": ((choice.get("message") or {}).get("content") or ""),
        "ids": ((body.get("hipengine") or {}).get("generated_token_ids")),
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
    }


def _concurrent(
    calls: list[dict[str, Any]],
    url: str,
    model: str,
    timeout: float,
    call_fn: Any = None,
) -> list[dict[str, Any]]:
    import httpx

    call_fn = call_fn or _chat
    results: list[dict[str, Any]] = [{} for _ in calls]

    def worker(index: int) -> None:
        local = httpx.Client(timeout=timeout)
        try:
            call = calls[index]
            results[index] = call_fn(
                local, url, model, call["content"], **call.get("extra", {})
            )
        except Exception as exc:  # a raised client error is a gate failure
            results[index] = {"status": 0, "error": str(exc)}
        finally:
            local.close()

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(len(calls))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18400")
    parser.add_argument("--model", default="qwen38fn-capon")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    import httpx

    url = args.url.rstrip("/")
    checks: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    client = httpx.Client(timeout=args.timeout)

    capabilities = client.get(f"{url}/v1/hipengine/capabilities").json()
    width = int(
        ((capabilities.get("admission") or {}).get("active_requests") or {}).get(
            "max_active_requests"
        )
        or 1
    )
    details["configured_width"] = width

    prompts = list(DEFAULT_PROMPTS)
    solo = [
        _chat(client, url, args.model, prompt, max_tokens=args.max_tokens)
        for prompt in prompts
    ]
    details["solo"] = [{"prompt": p, "text": s.get("text"), "ids": s.get("ids")} for p, s in zip(prompts, solo)]
    _check(
        checks,
        "solo.all_served",
        all(item.get("status") == 200 for item in solo),
        statuses=[item.get("status") for item in solo],
    )

    # Replay the same prompts concurrently in waves at the configured width.
    concurrent: list[dict[str, Any]] = []
    for start in range(0, len(prompts), max(1, width)):
        wave = prompts[start : start + max(1, width)]
        concurrent.extend(
            _concurrent(
                [{"content": prompt, "extra": {"max_tokens": args.max_tokens}} for prompt in wave],
                url,
                args.model,
                args.timeout,
            )
        )
    details["concurrent"] = [
        {"prompt": p, "text": c.get("text"), "ids": c.get("ids")}
        for p, c in zip(prompts, concurrent)
    ]
    _check(
        checks,
        "concurrent.all_served",
        all(item.get("status") == 200 for item in concurrent),
        statuses=[item.get("status") for item in concurrent],
    )
    mismatched = [
        {"prompt": prompt, "solo": s.get("text"), "concurrent": c.get("text")}
        for prompt, s, c in zip(prompts, solo, concurrent)
        if s.get("text") != c.get("text")
    ]
    details["mismatched_text"] = mismatched
    _check(checks, "concurrent.text_matches_solo", not mismatched, mismatched=mismatched)
    id_mismatched = [
        {"prompt": prompt, "solo": s.get("ids"), "concurrent": c.get("ids")}
        for prompt, s, c in zip(prompts, solo, concurrent)
        if s.get("ids") is not None and c.get("ids") is not None and s.get("ids") != c.get("ids")
    ]
    details["mismatched_ids"] = id_mismatched
    _check(checks, "concurrent.ids_match_solo", not id_mismatched, mismatched=id_mismatched)

    # A repeated request reproduces itself.
    repeats = [_chat(client, url, args.model, prompts[0], max_tokens=args.max_tokens) for _ in range(3)]
    details["repeat_texts"] = [item.get("text") for item in repeats]
    _check(
        checks,
        "repeat.identical",
        len({item.get("text") for item in repeats}) == 1,
        texts=[item.get("text") for item in repeats],
    )

    # Different budgets in one batch must stay separate.
    budgets = [16, 32, 64]
    mixed = _concurrent(
        [
            {"content": "Count upward from 1, one number per line.", "extra": {"max_tokens": budget, "ignore_eos": True}}
            for budget in budgets
        ],
        url,
        args.model,
        args.timeout,
    )
    details["mixed_budget_completion_tokens"] = [
        ((item.get("usage") or {}).get("completion_tokens")) for item in mixed
    ]
    _check(
        checks,
        "mixed_budget.each_respects_its_own_limit",
        [
            ((item.get("usage") or {}).get("completion_tokens")) for item in mixed
        ]
        == budgets,
        budgets=budgets,
        got=details["mixed_budget_completion_tokens"],
    )

    # Two interleaved sessions must answer from their own history.
    facts = {"iso-session-a": "The access code is 4417.", "iso-session-b": "The access code is 9028."}
    for session_id, fact in facts.items():
        _chat(
            client,
            url,
            args.model,
            f"Remember this: {fact}",
            session={"id": session_id},
            max_tokens=16,
        )
    answers: dict[str, str] = {}
    for session_id in facts:
        result = _chat(
            client,
            url,
            args.model,
            "What is the access code? Answer with the four digits only.",
            session={"id": session_id},
            max_tokens=24,
        )
        answers[session_id] = result.get("text") or ""
    details["session_answers"] = answers
    _check(
        checks,
        "session.answers_from_own_history",
        "4417" in answers.get("iso-session-a", "") and "9028" in answers.get("iso-session-b", ""),
        answers=answers,
    )
    _check(
        checks,
        "session.no_cross_contamination",
        "9028" not in answers.get("iso-session-a", "")
        and "4417" not in answers.get("iso-session-b", ""),
        answers=answers,
    )
    for session_id in facts:
        client.delete(f"{url}/v1/hipengine/sessions/{session_id}")

    client.close()
    failures = [check for check in checks if not check["passed"]]
    summary = {
        "schema": "hipengine.qwen4exp.http_isolation_gate.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": url,
        "model": args.model,
        "checks": checks,
        "details": details,
        "check_count": len(checks),
        "failure_count": len(failures),
        "status": "passed" if not failures else "failed",
    }
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for check in checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}")
    print(json.dumps({"status": summary["status"], "checks": len(checks), "failures": len(failures)}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
