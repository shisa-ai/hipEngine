#!/usr/bin/env python3
"""Execute a llama.cpp MTP trace-capture plan.

The companion plan generator emits the exact command/request/parser recipe.  This
script performs the side effects: start llama-server, wait for health, POST the
request, write metadata/response JSON, stop the server, then parse the verbose log
into a compact draft-MTP trace artifact.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_plan_metadata(plan: dict[str, Any]) -> Path:
    metadata_path = Path(plan["metadata_path"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(plan["metadata"], indent=2) + "\n")
    return metadata_path


def response_path_for_plan(plan: dict[str, Any]) -> Path:
    trace_json = Path(plan["trace_json"])
    return trace_json.with_suffix(".response.json")


def health_endpoint_for_request(endpoint: str) -> str:
    # endpoint is http://host:port/v1/chat/completions
    return endpoint.split("/v1/", 1)[0] + "/health"


def wait_for_health(url: str, *, timeout_s: float = 600.0, interval_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as exc:  # pragma: no cover - exercised via integration
            last_error = exc
        time.sleep(interval_s)
    raise TimeoutError(f"Timed out waiting for {url}; last_error={last_error!r}")


def post_json(url: str, payload: dict[str, Any], *, timeout_s: float = 600.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        data = response.read().decode("utf-8")
    return json.loads(data)


def stop_process(proc: subprocess.Popen[Any], *, timeout_s: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive integration path
        proc.kill()
        proc.wait(timeout=timeout_s)


def run_capture(plan: dict[str, Any], *, startup_timeout_s: float, request_timeout_s: float) -> dict[str, Any]:
    server_log = Path(plan["server_log"])
    server_log.parent.mkdir(parents=True, exist_ok=True)
    write_plan_metadata(plan)
    response_path = response_path_for_plan(plan)

    with server_log.open("wb") as log_file:
        proc = subprocess.Popen(plan["server_command"], stdout=log_file, stderr=subprocess.STDOUT)
        try:
            wait_for_health(
                health_endpoint_for_request(plan["request_endpoint"]),
                timeout_s=startup_timeout_s,
            )
            response = post_json(
                plan["request_endpoint"],
                plan["request_payload"],
                timeout_s=request_timeout_s,
            )
            response_path.write_text(json.dumps(response, indent=2) + "\n")
        finally:
            stop_process(proc)

    subprocess.run(plan["parser_command"], check=True)
    return {
        "server_log": str(server_log),
        "metadata_path": plan["metadata_path"],
        "response_path": str(response_path),
        "trace_json": plan["trace_json"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Trace plan JSON produced by llamacpp_mtp_greeting_trace_plan.py")
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without launching llama-server")
    args = parser.parse_args()

    plan = load_plan(args.plan)
    if args.dry_run:
        print(json.dumps({
            "server_command": plan["server_command"],
            "health_endpoint": health_endpoint_for_request(plan["request_endpoint"]),
            "request_endpoint": plan["request_endpoint"],
            "metadata_path": plan["metadata_path"],
            "response_path": str(response_path_for_plan(plan)),
            "parser_command": plan["parser_command"],
        }, indent=2))
        return

    result = run_capture(
        plan,
        startup_timeout_s=args.startup_timeout_s,
        request_timeout_s=args.request_timeout_s,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
