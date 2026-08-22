#!/usr/bin/env python3
"""RF7 zero-scope rollback/restart drill for dense GGUF MTP.

RF6 promoted no automatic scope, so this gate performs no canary expansion. It
proves an operator can disable MTP for new requests without restart, an in-flight
owned request retires safely, subsequent requests route AR, shutdown drains
resources, and restart resets the breaker while auto remains AR.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.server import ServerConfig, create_app
from scripts.gguf_mtp_long_context_gate import _atomic_write_json, _git

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")


def _post(client: TestClient, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/v1/completions", json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"completion status={response.status_code}: {response.text}")
    return response.json()


def _ids(body: dict[str, Any]) -> list[int]:
    return [int(token) for token in body["choices"][0]["hipengine"]["generated_token_ids"]]


def _config(model: Path, policy: str) -> ServerConfig:
    return ServerConfig(
        model=str(model),
        served_model_name="qwen36-rf7",
        eager_load=True,
        startup_chat_smoke=False,
        startup_scratch_probe=False,
        speculative_mtp_serving=policy,
        max_active_requests=4,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    reset_memory_stats()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    payload = {
        "model": "qwen36-rf7",
        "prompt": "Write a concise explanation of rollback safety.",
        "max_tokens": 32,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    long_payload = {**payload, "max_tokens": 128}

    app = create_app(_config(args.model, "enabled"))
    with TestClient(app) as client:
        before = _post(client, payload)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            in_flight_future = pool.submit(_post, client, long_payload)
            deadline = time.monotonic() + 10.0
            while app.state.hipengine_generation_batcher.active_requests() == 0:
                if time.monotonic() >= deadline:
                    raise RuntimeError("in-flight MTP request did not start")
                time.sleep(0.005)
            rollback = client.post("/v1/hipengine/speculative_mtp/rollback")
            in_flight = in_flight_future.result(timeout=180.0)
        after = _post(client, payload)
        capability = client.get("/v1/hipengine/capabilities").json()["sampling"][
            "speculative_mtp"
        ]
        row = {
            "id": "runtime_rollback",
            "before_route": before["hipengine"]["generation_shape"]["route"],
            "in_flight_route": in_flight["hipengine"]["generation_shape"]["route"],
            "rollback_status": rollback.status_code,
            "rollback_state": rollback.json()["circuit_breaker"]["state"],
            "after_route": after["hipengine"]["generation_shape"]["route"],
            "after_reason": after["hipengine"]["generation_shape"]["route_decision"]["reason"],
            "breaker_state": capability["circuit_breaker"]["state"],
            "in_flight_used_mtp": in_flight["hipengine"]["speculative_mtp"]["used"],
            "after_used_mtp": after["hipengine"]["speculative_mtp"]["used"],
        }
        row["passed"] = bool(
            row["before_route"] == "speculative_mtp"
            and row["in_flight_route"] == "speculative_mtp"
            and row["rollback_status"] == 200
            and row["rollback_state"] == "operator_disabled"
            and row["after_route"] == "default"
            and row["after_reason"] == "mtp_operator_rollback"
            and row["breaker_state"] == "operator_disabled"
            and row["in_flight_used_mtp"] is True
            and row["after_used_mtp"] is False
        )
        rows.append(row)
        ready_before_shutdown = client.get("/ready").status_code

    after_shutdown_memory = memory_stats()
    rows.append(
        {
            "id": "shutdown_drain",
            "ready_before_shutdown": ready_before_shutdown,
            "memory": after_shutdown_memory,
            "passed": ready_before_shutdown == 200
            and after_shutdown_memory["current_allocated_bytes"] == 0
            and after_shutdown_memory["active_allocations"] == 0,
        }
    )

    restarted_app = create_app(_config(args.model, "auto"))
    with TestClient(restarted_app) as restarted:
        capability = restarted.get("/v1/hipengine/capabilities").json()["sampling"][
            "speculative_mtp"
        ]
        auto = _post(restarted, payload)
        explicit = _post(restarted, {**payload, "speculative_mtp": True})
        row = {
            "id": "restart_zero_scope",
            "breaker_state": capability["circuit_breaker"]["state"],
            "certified_default_scopes": capability["certified_default_scopes"],
            "automatic_route_promoted": capability["automatic_route_promoted"],
            "auto_route": auto["hipengine"]["generation_shape"]["route"],
            "auto_reason": auto["hipengine"]["generation_shape"]["route_decision"]["reason"],
            "explicit_route": explicit["hipengine"]["generation_shape"]["route"],
            "auto_ids_equal_explicit": _ids(auto) == _ids(explicit),
        }
        row["passed"] = bool(
            row["breaker_state"] == "closed"
            and row["certified_default_scopes"] == []
            and row["automatic_route_promoted"] is False
            and row["auto_route"] == "default"
            and row["auto_reason"] == "automatic_mtp_scope_not_promoted"
            and row["explicit_route"] == "speculative_mtp"
            and row["auto_ids_equal_explicit"]
        )
        rows.append(row)
    final_memory = memory_stats()
    rows.append(
        {
            "id": "restart_shutdown_drain",
            "memory": final_memory,
            "passed": final_memory["current_allocated_bytes"] == 0
            and final_memory["active_allocations"] == 0,
        }
    )

    passed = all(bool(row["passed"]) for row in rows)
    artifact = {
        "schema": 1,
        "kind": "gguf_mtp_zero_scope_rollout_gate",
        "status": "passed" if passed else "failed",
        "verdict": "pass" if passed else "fail",
        "performance_claim": False,
        "rollout_scope": "none",
        "canary_started": False,
        "command": [sys.executable, *sys.argv],
        "repo": {"commit": _git(["rev-parse", "HEAD"]), "dirty": bool(_git(["status", "--porcelain"]))},
        "model": {
            "path": str(args.model.resolve()),
            "size_bytes": args.model.stat().st_size,
            "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest() if args.hash_model else None,
        },
        "rows": rows,
        "memory": final_memory,
        "summary": {
            "passed": sum(bool(row["passed"]) for row in rows),
            "total": len(rows),
            "wall_seconds": time.perf_counter() - started,
        },
        "decision": "RF6 promoted no scope. Do not canary; keep auto on AR and explicit diagnostic MTP available after restart.",
        "passed": passed,
    }
    if args.out is None:
        print(json.dumps(artifact, indent=2, sort_keys=True))
    else:
        _atomic_write_json(args.out, artifact)
        print(f"wrote {args.out}: passed={passed}")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--hash-model", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    artifact = run(args)
    return 1 if args.fail_on_fail and not artifact["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
