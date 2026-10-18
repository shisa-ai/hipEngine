#!/usr/bin/env python3
"""Lifecycle gate for the served route: slots, cancellation, and sessions.

Capacity numbers say nothing about whether a slot comes back. This gate drives
the live server and asserts the observable lifecycle:

* sequential requests leave no live slots behind,
* concurrent requests up to the configured width all complete,
* a request beyond the width is admitted or refused with a retryable error,
  never a crash or a silent hang,
* a client that disconnects mid-stream releases its slot,
* a stateful chat session reports the commit action it actually applied and
  disappears when deleted.
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


def _check(checks: list[dict[str, Any]], name: str, passed: bool, **detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), **detail})


def _decode(client: Any, url: str) -> dict[str, Any]:
    response = client.get(f"{url}/v1/hipengine/decode")
    if response.status_code != 200:
        return {}
    return response.json().get("decode") or {}


def _live_slots(payload: dict[str, Any]) -> int | None:
    """Read live batch slots from either published decode shape.

    ``/v1/hipengine/decode`` nests the state under ``decode.ar_batch`` and the
    capabilities payload nests it under ``ar_batch.active``; a gate that only
    understood one of them would report "no leaked slots" without reading any.
    """

    ar_batch = payload.get("ar_batch")
    if not isinstance(ar_batch, dict):
        return None
    active = ar_batch.get("active")
    source = active if isinstance(active, dict) else ar_batch
    value = source.get("live_slots")
    return None if value is None else int(value)


def _chat(client: Any, url: str, model: str, text: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.0,
        "max_tokens": extra.pop("max_tokens", 24),
        **extra,
    }
    response = client.post(f"{url}/v1/chat/completions", json=payload)
    return {"status": response.status_code, "body": response.json()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18395")
    parser.add_argument("--model", default="qwen38fn-capon")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    import httpx

    url = args.url.rstrip("/")
    checks: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    client = httpx.Client(timeout=args.timeout)

    capabilities = client.get(f"{url}/v1/hipengine/capabilities").json()
    width = int((capabilities.get("admission", {}).get("active_requests") or {}).get("max_active_requests") or 1)
    details["configured_width"] = width
    before = _decode(client, url)
    details["live_slots_before"] = _live_slots(before)

    # Sequential requests must not accumulate slots.
    for index in range(3):
        result = _chat(client, url, args.model, f"Reply with exactly: OK{index}")
        _check(checks, f"sequential[{index}].status", result["status"] == 200, status=result["status"])
    after_sequential = _decode(client, url)
    details["live_slots_after_sequential"] = _live_slots(after_sequential)
    _check(
        checks,
        "sequential.no_leaked_slots",
        _live_slots(after_sequential) in (None, 0),
        live_slots=_live_slots(after_sequential),
    )

    # Concurrent requests at the configured width must all complete.
    results: list[dict[str, Any]] = [{} for _ in range(width)]
    def _worker(index: int) -> None:
        local = httpx.Client(timeout=args.timeout)
        try:
            results[index] = _chat(local, url, args.model, f"Reply with exactly: OKC{index}")
        except Exception as exc:  # a raised client error is a lifecycle failure
            results[index] = {"status": 0, "error": str(exc)}
        finally:
            local.close()

    threads = [threading.Thread(target=_worker, args=(index,)) for index in range(width)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    observed: list[int] = []
    while any(thread.is_alive() for thread in threads):
        sample = _live_slots(_decode(client, url))
        if sample is not None:
            observed.append(sample)
        time.sleep(0.05)
    for thread in threads:
        thread.join()
    concurrent_s = time.perf_counter() - started
    details["max_live_slots_during_concurrent"] = max(observed) if observed else None
    details["concurrent_s"] = round(concurrent_s, 3)
    details["concurrent_statuses"] = [result.get("status") for result in results]
    _check(
        checks,
        "concurrent.slots_were_used",
        (details.get("max_live_slots_during_concurrent") or 0) >= 1,
        max_live_slots=details.get("max_live_slots_during_concurrent"),
    )
    _check(
        checks,
        "concurrent.all_completed",
        all(result.get("status") == 200 for result in results),
        statuses=details["concurrent_statuses"],
    )
    after_concurrent = _decode(client, url)
    details["live_slots_after_concurrent"] = _live_slots(after_concurrent)
    _check(
        checks,
        "concurrent.no_leaked_slots",
        _live_slots(after_concurrent) in (None, 0),
        live_slots=_live_slots(after_concurrent),
    )

    # A request beyond the width is admitted, queued, or refused with a
    # retryable error. It must never crash the server or hang forever.
    oversubscribe: list[dict[str, Any]] = [{} for _ in range(width + 2)]
    def _over(index: int) -> None:
        local = httpx.Client(timeout=args.timeout)
        try:
            oversubscribe[index] = _chat(local, url, args.model, f"Reply with exactly: OKX{index}")
        except Exception as exc:
            oversubscribe[index] = {"status": 0, "error": str(exc)}
        finally:
            local.close()

    threads = [threading.Thread(target=_over, args=(index,)) for index in range(width + 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    over_statuses = [entry.get("status") for entry in oversubscribe]
    details["oversubscribe_statuses"] = over_statuses
    _check(
        checks,
        "oversubscribe.answered_every_request",
        all(status in (200, 429, 503) for status in over_statuses),
        statuses=over_statuses,
    )
    details["oversubscribe_codes"] = [
        ((entry.get("body") or {}).get("error") or {}).get("code")
        for entry in oversubscribe
        if entry.get("status") not in (None, 200)
    ]

    # A client that disconnects mid-stream must release its slot.
    stream_error: str | None = None
    try:
        with httpx.Client(timeout=args.timeout) as streamer:
            with streamer.stream(
                "POST",
                f"{url}/v1/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": "Count from 1 to 400, one per line."}],
                    "temperature": 0.0,
                    "max_tokens": 400,
                    "stream": True,
                },
            ) as response:
                chunks = 0
                for _line in response.iter_lines():
                    chunks += 1
                    if chunks >= 4:
                        break
    except Exception as exc:
        stream_error = str(exc)
    details["stream_chunks_before_disconnect"] = chunks if stream_error is None else None
    details["stream_error"] = stream_error
    time.sleep(2.0)
    after_cancel = _decode(client, url)
    details["live_slots_after_cancel"] = _live_slots(after_cancel)
    _check(
        checks,
        "cancel.slot_released",
        _live_slots(after_cancel) in (None, 0),
        live_slots=_live_slots(after_cancel),
    )
    recovery = _chat(client, url, args.model, "Reply with exactly: OKAFTER")
    _check(checks, "cancel.server_still_serves", recovery["status"] == 200, status=recovery["status"])

    # A stateful chat session reports the commit it applied and is deletable.
    session_id = f"lifecycle-{int(time.time())}"
    stateful = _chat(
        client,
        url,
        args.model,
        "Reply with exactly: OKSESSION",
        session={"id": session_id},
    )
    action = None
    if stateful["status"] == 200:
        choice = (stateful["body"].get("choices") or [{}])[0]
        action = ((choice.get("finish_details") or {}).get("cache_action"))
    details["session_cache_action"] = action
    _check(
        checks,
        "session.commit_action_reported",
        action in {"append_visible_only", "append_prompt_only", "append_all", "append_none"},
        cache_action=action,
    )
    listing = client.get(f"{url}/v1/hipengine/sessions").json()
    ids = [entry.get("id") for entry in listing.get("sessions") or ()]
    _check(checks, "session.listed_after_commit", session_id in ids, session_ids=ids)
    deleted = client.delete(f"{url}/v1/hipengine/sessions/{session_id}")
    _check(checks, "session.delete_ok", deleted.status_code == 200, status=deleted.status_code)
    listing_after = client.get(f"{url}/v1/hipengine/sessions").json()
    ids_after = [entry.get("id") for entry in listing_after.get("sessions") or ()]
    _check(checks, "session.removed_after_delete", session_id not in ids_after, session_ids=ids_after)

    client.close()
    failures = [check for check in checks if not check["passed"]]
    summary = {
        "schema": "hipengine.qwen4exp.http_lifecycle_gate.v1",
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
        args.output.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    for check in checks:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}")
    print(json.dumps({"status": summary["status"], "checks": len(checks), "failures": len(failures)}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
