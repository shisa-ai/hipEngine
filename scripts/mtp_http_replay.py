#!/usr/bin/env python3
"""Replay recorded HTTP requests and verify that explicit MTP actually ran."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request


def summarize(response: dict[str, Any]) -> dict[str, Any]:
    """Read blocking or streamed output without merging reasoning into content."""
    events = [response]
    if "sse_raw" in response:
        events = [
            json.loads(line[6:])
            for line in response["sse_raw"].splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"
        ]
    metadata: dict[str, Any] = {}
    content: list[str] = []
    reasoning: list[str] = []
    errors = []
    usage = None
    finish_reason = None
    for event in events:
        if event.get("error"):
            errors.append(event["error"])
        metadata.update(event.get("hipengine") or {})
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            if int(choice.get("index", 0)) != 0:
                continue
            metadata.update(choice.get("hipengine") or {})
            message = choice.get("delta", choice.get("message", choice))
            content.append(message.get("content", message.get("text")) or "")
            reasoning.append(message.get("reasoning_content", message.get("reasoning")) or "")
            finish_reason = choice.get("finish_reason") or finish_reason
    return {
        "content": "".join(content), "reasoning_content": "".join(reasoning),
        "hipengine": metadata, "usage": usage, "finish_reason": finish_reason,
        "errors": errors,
    }


def explicit_mtp(payload: dict[str, Any]) -> bool:
    value = payload.get("speculative_mtp", payload.get("speculative"))
    return value is True or isinstance(value, dict) and value.get("enabled", True) is True


def check_row(row: dict[str, Any]) -> list[str]:
    failures = []
    summary = row["summary"]
    if row["status"] != 200 or summary["errors"]:
        failures.append("request_failed")
    if explicit_mtp(row["request"]) and row["status"] == 200 and not summary["errors"]:
        if not summary["hipengine"].get("speculative_mtp", {}).get("used"):
            failures.append("explicit_mtp_did_not_run")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Recorded JSON with a rows array containing label and request.")
    parser.add_argument("--output", type=Path, required=True, help="Write requests, responses, normalized channels and checks here.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18999")
    parser.add_argument("--labels", help="Comma-separated labels to replay; the default replays every row.")
    args = parser.parse_args()
    source = args.input.read_bytes()
    recorded = json.loads(source)
    labels = None if not args.labels else set(args.labels.split(","))
    rows = [row for row in recorded["rows"] if labels is None or row["label"] in labels]
    if labels is not None and labels != {row["label"] for row in rows}:
        parser.error("one or more requested labels are missing from the input")
    with urllib.request.urlopen(args.base_url + "/ready", timeout=10) as response:
        ready = json.load(response)
    if not ready.get("ready"):
        raise RuntimeError("server is not ready")
    requested_models = {row["request"].get("model") for row in rows}
    if requested_models != {ready.get("model", {}).get("id")}:
        raise RuntimeError("loaded model does not match the recorded requests")
    result = {
        "schema": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "host": platform.node(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "ready": ready, "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def post(row: dict[str, Any]) -> None:
        payload = row["request"]
        request = urllib.request.Request(
            args.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                status = response.status
                raw = response.read().decode()
                body = {"sse_raw": raw} if payload.get("stream") else json.loads(raw)
        except urllib.error.HTTPError as error:
            status, body = error.code, json.loads(error.read())
        item = {
            "label": row["label"], "request": payload, "status": status,
            "wall_seconds": time.perf_counter() - started, "response": body,
            "summary": summarize(body),
        }
        item["failures"] = check_row(item)
        with lock:
            result["rows"].append(item)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            spec = item["summary"]["hipengine"].get("speculative_mtp", {})
            print(row["label"], status, f"used={spec.get('used')}", item["failures"], flush=True)

    for row in rows:
        if not row["label"].startswith("c4_"):
            post(row)
    concurrent = [row for row in rows if row["label"].startswith("c4_")]
    if concurrent:
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(post, concurrent))
    result["passed"] = all(not row["failures"] for row in result["rows"])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
