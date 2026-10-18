#!/usr/bin/env python3
"""Repeated mixed-request soak with a recovery probe after every round.

Each round issues, concurrently: blocking chat on the AR and MTP routes, a
streaming request, a streaming request the client abandons mid-stream, a refused
oversized request, a session write/read pair, a logprobs request and a
structured-output request. After the round settles, the harness checks `/ready`
and runs one plain completion. The engine must serve that probe every time: a
failed round that leaves the resident engine closed is the defect this catches.

A fixed greedy prompt is also replayed once per round and compared to round one,
so cross-request state leakage shows up as a changed answer.
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

PROBE_PROMPT = "Reply with exactly: SOAK-PROBE"
DETERMINISM_PROMPT = "Name the capital of France in one word."


def _post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """One completion call, retrying the documented retryable admission refusal.

    ``429 engine_busy`` with ``Retry-After`` is backpressure, not a failure: a
    production client waits and retries. The harness records how many attempts
    it took so the burst cost stays visible.
    """

    import httpx

    attempts = 0
    while True:
        attempts += 1
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{url}/v1/chat/completions", json=payload)
        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text[:400]}
        error = body.get("error") or {}
        retryable = (
            response.status_code == 429
            and error.get("code") == "engine_busy"
            and response.headers.get("retry-after")
            and attempts < 4
        )
        if retryable:
            time.sleep(float(response.headers["retry-after"]))
            continue
        choice = (body.get("choices") or [{}])[0]
        return {
            "status": response.status_code,
            "text": ((choice.get("message") or {}).get("content") or ""),
            "finish_reason": choice.get("finish_reason"),
            "code": error.get("code"),
            "retry_after": response.headers.get("retry-after"),
            "attempts": attempts,
            "usage": body.get("usage"),
        }


def _chat(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    return _post(url, payload, timeout)


def _stream(url: str, payload: dict[str, Any], timeout: float, *, abandon_after: int | None) -> dict[str, Any]:
    import httpx

    chunks: list[str] = []
    usage: dict[str, Any] | None = None
    events = 0
    with httpx.Client(timeout=timeout) as client:
        try:
            with client.stream("POST", f"{url}/v1/chat/completions", json=payload) as response:
                if response.status_code == 429 and response.headers.get("retry-after"):
                    response.read()
                    time.sleep(float(response.headers["retry-after"]))
                    return _stream(url, payload, timeout, abandon_after=abandon_after)
                if response.status_code != 200:
                    response.read()
                    return {"status": response.status_code, "chunks": 0, "abandoned": False}
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    events += 1
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or ():
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            chunks.append(delta)
                    if abandon_after is not None and events >= abandon_after:
                        return {
                            "status": 200,
                            "chunks": events,
                            "text": "".join(chunks),
                            "abandoned": True,
                            "usage": usage,
                        }
        except httpx.HTTPError as exc:
            return {"status": 0, "error": str(exc), "chunks": events, "abandoned": False}
    return {
        "status": 200,
        "chunks": events,
        "text": "".join(chunks),
        "abandoned": False,
        "usage": usage,
    }


def _round(url: str, model: str, round_index: int, timeout: float) -> dict[str, Any]:
    results: dict[str, Any] = {}

    def run(name: str, fn) -> None:
        try:
            results[name] = fn()
        except Exception as exc:  # a raised client error is soak evidence
            results[name] = {"status": 0, "error": str(exc)}

    session_id = f"soak-{round_index}"
    code = f"{4000 + round_index}"

    def session_write() -> dict[str, Any]:
        return _chat(
            url,
            {
                "model": model,
                "messages": [{"role": "user", "content": f"Remember: the code is {code}."}],
                "session": {"id": session_id},
                "max_tokens": 16,
            },
            timeout,
        )

    def session_read() -> dict[str, Any]:
        return _chat(
            url,
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "What is the code? Answer with digits only."}
                ],
                "session": {"id": session_id},
                "max_tokens": 24,
            },
            timeout,
        )

    tasks = {
        "ar_blocking": lambda: _chat(
            url,
            {
                "model": model,
                "messages": [{"role": "user", "content": PROBE_PROMPT}],
                "max_tokens": 16,
                "speculative_mtp": False,
            },
            timeout,
        ),
        "mtp_blocking": lambda: _chat(
            url,
            {
                "model": model,
                "messages": [{"role": "user", "content": "List three primary colors."}],
                "max_tokens": 48,
                "speculative_mtp": True,
            },
            timeout,
        ),
        "stream_mtp": lambda: _stream(
            url,
            {
                "model": model,
                "messages": [{"role": "user", "content": "Count from 1 to 12."}],
                "max_tokens": 64,
                "speculative_mtp": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout,
            abandon_after=None,
        ),
        "stream_abandoned": lambda: _stream(
            url,
            {
                "model": model,
                "messages": [{"role": "user", "content": "Write a long paragraph about rivers."}],
                "max_tokens": 256,
                "speculative_mtp": True,
                "stream": True,
            },
            timeout,
            abandon_after=2,
        ),
        "refused_oversized": lambda: _chat(
            url,
            {
                "model": model,
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 4_000_000,
            },
            timeout,
        ),
        "logprobs": lambda: _chat(
            url,
            {
                "model": model,
                "messages": [{"role": "user", "content": PROBE_PROMPT}],
                "max_tokens": 8,
                "logprobs": True,
                "top_logprobs": 2,
            },
            timeout,
        ),
        "structured": lambda: _chat(
            url,
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Return a JSON object with answer equal to 42."}
                ],
                "max_tokens": 48,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
            timeout,
        ),
        "determinism": lambda: _chat(
            url,
            {
                "model": model,
                "messages": [{"role": "user", "content": DETERMINISM_PROMPT}],
                "max_tokens": 8,
            },
            timeout,
        ),
    }

    threads = [threading.Thread(target=run, args=(name, fn)) for name, fn in tasks.items()]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started

    # A session turn is ordered by definition: the read must see the write, so
    # the pair runs after the burst instead of racing it.
    run("session_write", session_write)
    run("session_read", session_read)

    import httpx

    with httpx.Client(timeout=timeout) as client:
        ready = client.get(f"{url}/ready")
        ready_body = ready.json() if ready.headers.get("content-type", "").startswith("application/json") else {}
    recovery = _chat(
        url,
        {
            "model": model,
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
            "max_tokens": 16,
        },
        timeout,
    )
    with httpx.Client(timeout=timeout) as client:
        client.delete(f"{url}/v1/hipengine/sessions/{session_id}")
    return {
        "round": round_index,
        "session_code": code,
        "wall_s": round(elapsed, 3),
        "requests": results,
        "ready_status": ready.status_code,
        "ready": ready_body,
        "recovery": recovery,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18400")
    parser.add_argument("--model", default="qwen38fn-capon")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    url = args.url.rstrip("/")
    rounds: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, **detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **detail})

    for index in range(1, args.rounds + 1):
        result = _round(url, args.model, index, args.timeout)
        rounds.append(result)
        requests = result["requests"]
        check(
            f"round{index}.recovery",
            result["recovery"].get("status") == 200
            and "SOAK" in (result["recovery"].get("text") or ""),
            recovery=result["recovery"],
        )
        check(f"round{index}.ready", result["ready_status"] == 200, status=result["ready_status"])
        check(
            f"round{index}.ar_blocking",
            requests["ar_blocking"].get("status") == 200,
            status=requests["ar_blocking"].get("status"),
        )
        check(
            f"round{index}.mtp_blocking",
            requests["mtp_blocking"].get("status") == 200,
            status=requests["mtp_blocking"].get("status"),
        )
        check(
            f"round{index}.stream_mtp",
            requests["stream_mtp"].get("status") == 200
            and (requests["stream_mtp"].get("chunks") or 0) > 1
            and bool(requests["stream_mtp"].get("usage")),
            chunks=requests["stream_mtp"].get("chunks"),
            usage=requests["stream_mtp"].get("usage"),
        )
        check(
            f"round{index}.stream_abandoned",
            requests["stream_abandoned"].get("status") == 200
            and requests["stream_abandoned"].get("abandoned") is True,
            chunks=requests["stream_abandoned"].get("chunks"),
        )
        check(
            f"round{index}.refused_oversized",
            400 <= int(requests["refused_oversized"].get("status") or 0) < 500,
            status=requests["refused_oversized"].get("status"),
            code=requests["refused_oversized"].get("code"),
        )
        check(
            f"round{index}.logprobs",
            requests["logprobs"].get("status") == 200,
            status=requests["logprobs"].get("status"),
        )
        check(
            f"round{index}.structured",
            requests["structured"].get("status") == 200,
            status=requests["structured"].get("status"),
        )
        expected_code = result["session_code"]
        check(
            f"round{index}.session_isolated",
            expected_code in (requests["session_read"].get("text") or ""),
            expected=expected_code,
            answer=requests["session_read"].get("text"),
        )
        absorbed = {
            name: payload.get("attempts")
            for name, payload in requests.items()
            if isinstance(payload, dict) and int(payload.get("attempts") or 1) > 1
        }
        check(
            f"round{index}.backpressure_retryable",
            all(
                payload.get("status") == 200
                for name, payload in requests.items()
                if isinstance(payload, dict) and int(payload.get("attempts") or 1) > 1
            ),
            retried=absorbed,
        )
        print(
            f"round {index}: wall {result['wall_s']}s "
            f"recovery={result['recovery'].get('status')} ready={result['ready_status']} "
            f"retried={absorbed or 'none'}",
            flush=True,
        )

    # The same greedy prompt must answer identically after all the churn.
    texts = [item["requests"]["determinism"].get("text") for item in rounds]
    check(
        "churn.determinism",
        len(set(texts)) == 1 and texts[0] is not None,
        texts=texts,
    )
    # No round may leave the engine closed.
    closed = [
        name
        for item in rounds
        for name, payload in item["requests"].items()
        if isinstance(payload, dict)
        and payload.get("code") == "resident_prepare_failed"
    ]
    check("churn.engine_never_closed", not closed, closed=closed)

    failures = [item for item in checks if not item["passed"]]
    summary = {
        "schema": "hipengine.qwen4exp.http_soak.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": url,
        "model": args.model,
        "rounds": len(rounds),
        "checks": checks,
        "results": rounds,
        "check_count": len(checks),
        "failure_count": len(failures),
        "status": "passed" if not failures else "failed",
    }
    args.output.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for item in checks:
        print(f"{'PASS' if item['passed'] else 'FAIL'} {item['name']}")
    print(json.dumps({"status": summary["status"], "checks": len(checks), "failures": len(failures)}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
