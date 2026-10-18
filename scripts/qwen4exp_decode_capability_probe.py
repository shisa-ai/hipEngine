#!/usr/bin/env python3
"""Check that the advertised decode route matches the route the server runs.

The capabilities manifest and ``/v1/hipengine/decode`` are two surfaces over one
probe. This probe reads both before and after real concurrent generation and
fails when they disagree, when the advertised switch disagrees with the server's
own environment, or when a physical batch width is advertised that the server
never ran.

Usage (one server configuration per invocation)::

    python scripts/qwen4exp_decode_capability_probe.py \
        --url http://127.0.0.1:18391 --expect-enabled 1 --expect-width 2 \
        --output /tmp/decode_capability_batch.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx


def _check(checks: list[dict[str, Any]], name: str, passed: bool, **details: Any) -> bool:
    checks.append({"name": name, "passed": bool(passed), **details})
    return bool(passed)


async def _get(client: httpx.AsyncClient, path: str) -> dict[str, Any]:
    response = await client.get(path)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise TypeError(f"{path} returned {type(body).__name__}")
    return body


async def _wait_idle(client: httpx.AsyncClient, timeout_s: float = 180.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = await _get(client, "/v1/hipengine/decode")
        decode = last.get("decode") or {}
        batch = decode.get("ar_batch") or {}
        if int(batch.get("live_slots") or 0) == 0:
            return last
        await asyncio.sleep(0.1)
    raise TimeoutError(f"server never released its batch slots; last {last}")


async def _chat(client: httpx.AsyncClient, model: str, prompt: str, tokens: int) -> dict[str, Any]:
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(tokens),
            "temperature": 0.0,
        },
    )
    body: dict[str, Any] = {"status": response.status_code}
    try:
        body["body"] = response.json()
    except ValueError:
        body["body"] = response.text[:200]
    return body


def _summary(capabilities: dict[str, Any]) -> dict[str, Any]:
    """Flatten the capabilities ``decode`` block for the checks and the artifact."""

    decode = capabilities.get("decode") or {}
    batch = decode.get("ar_batch")
    return {
        "available": decode.get("available"),
        "path": decode.get("path"),
        "reason": decode.get("reason"),
        "enabled": None if batch is None else batch.get("enabled"),
        "configured_slots": None if batch is None else batch.get("configured_slots"),
        "qualified_max_slots": None if batch is None else batch.get("qualified_max_slots"),
        "qualified_max_positions": None if batch is None else batch.get("qualified_max_positions"),
        "mtp_rows_join_batch": None if batch is None else batch.get("mtp_rows_join_batch"),
        "invalid_configuration": None if batch is None else batch.get("invalid_configuration"),
        "active": None if batch is None else batch.get("active"),
    }


async def run(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_decode_capability_probe",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": args.url,
        "expect": {"enabled": int(args.expect_enabled), "width": args.expect_width},
        "checks": [],
    }
    checks: list[dict[str, Any]] = report["checks"]
    async with httpx.AsyncClient(base_url=args.url, timeout=httpx.Timeout(600.0)) as client:
        ready = await _get(client, "/ready")
        report["ready"] = {"ready": ready.get("ready"), "status": ready.get("status")}
        model = str((ready.get("model") or {}).get("id") or args.model)
        if not model:
            raise SystemExit("cannot resolve the served model id; pass --model")
        report["model"] = model

        before_cap = await _get(client, "/v1/hipengine/capabilities")
        before_dec = await _get(client, "/v1/hipengine/decode")
        report["before"] = {
            "capabilities": _summary(before_cap),
            "decode": before_dec,
        }
        before = report["before"]["capabilities"]
        _check(checks, "before.available", before["available"] is True, got=before["available"])
        _check(
            checks,
            "before.same_path",
            before["path"] == (before_dec.get("decode") or {}).get("decode_path"),
            capabilities_path=before["path"],
            decode_path=(before_dec.get("decode") or {}).get("decode_path"),
        )
        _check(
            checks,
            "before.envelope_matches_env",
            before["enabled"] == bool(int(args.expect_enabled)),
            advertised=before["enabled"],
            expected=bool(int(args.expect_enabled)),
        )
        _check(
            checks,
            "before.envelope_present",
            before["qualified_max_slots"] is not None,
            qualified_max_slots=before["qualified_max_slots"],
        )

        started = time.time()
        rows = await asyncio.gather(
            *(
                _chat(client, model, f"Reply with exactly one word: WORD{index}", args.tokens)
                for index in range(int(args.requests))
            )
        )
        report["requests"] = {
            "count": len(rows),
            "statuses": [row["status"] for row in rows],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        _check(
            checks,
            "requests.all_200",
            all(row["status"] == 200 for row in rows),
            statuses=[row["status"] for row in rows],
        )

        after_dec = await _wait_idle(client)
        after_cap = await _get(client, "/v1/hipengine/capabilities")
        report["after"] = {"capabilities": _summary(after_cap), "decode": after_dec}
        after = report["after"]["capabilities"]
        decode_after = after_dec.get("decode") or {}
        _check(
            checks,
            "after.same_path",
            after["path"] == decode_after.get("decode_path"),
            capabilities_path=after["path"],
            decode_path=decode_after.get("decode_path"),
        )
        _check(
            checks,
            "after.same_live_state",
            after["active"] == decode_after.get("ar_batch"),
            capabilities_active=after["active"],
            decode_active=decode_after.get("ar_batch"),
        )

        if int(args.expect_enabled):
            _check(
                checks,
                "after.batch_path",
                after["path"] == "qwen4exp_full_model_ar_batch",
                path=after["path"],
            )
            widths = (after["active"] or {}).get("step_widths") or {}
            report["after"]["step_widths"] = widths
            _check(
                checks,
                "after.physical_width_observed",
                int(widths.get(str(args.expect_width), 0)) > 0,
                expected_width=args.expect_width,
                step_widths=widths,
            )
        else:
            _check(
                checks,
                "after.c1_path",
                after["path"] == "qwen4exp_c1_request_owned_step",
                path=after["path"],
            )
            _check(
                checks,
                "after.no_batch_state",
                after["active"] is None,
                active=after["active"],
            )

    failures = [item for item in checks if not item["passed"]]
    report["status"] = "passed" if not failures else "failed"
    report["check_count"] = len(checks)
    report["failure_count"] = len(failures)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"status": report["status"], "checks": len(checks), "failures": len(failures)}
        ),
        flush=True,
    )
    for item in failures:
        print(json.dumps({"failed_check": item}, ensure_ascii=False), flush=True)
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18391")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--expect-enabled", choices=("0", "1"), default="0")
    parser.add_argument("--expect-width", type=int, default=2)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
