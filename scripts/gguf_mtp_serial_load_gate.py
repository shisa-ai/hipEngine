#!/usr/bin/env python3
"""RF5 load/isolation gate for the honest serialized dense-MTP policy.

The server may coalesce queue requests, but dense MTP owns one physical target
slot and executes request rows serially. This gate validates that declared
policy under c1/c2/c4/c8 offered load, one multi-prompt request, ragged prompts,
mixed AR/MTP traffic, deadline/re-admission, a focused 100-request lifecycle
run, bounded memory, and shutdown drain. It does not claim physical MTP
concurrency or automatic promotion.
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


def _ids(body: dict[str, Any]) -> list[list[int]]:
    return [
        [int(token) for token in choice["hipengine"]["generated_token_ids"]]
        for choice in body["choices"]
    ]


def _request(
    client: TestClient,
    prompts: str | list[str],
    *,
    mtp: bool,
    max_tokens: int = 4,
    timeout_ms: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "qwen36-rf5",
        "prompt": prompts,
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "speculative_mtp": bool(mtp),
    }
    if timeout_ms is not None:
        payload["timeout_ms"] = float(timeout_ms)
    started = time.perf_counter()
    response = client.post("/v1/completions", json=payload)
    row: dict[str, Any] = {
        "status_code": response.status_code,
        "wall_seconds": time.perf_counter() - started,
        "mtp_requested": bool(mtp),
    }
    try:
        body = response.json()
    except Exception:
        row["body"] = response.text
        return row
    if response.status_code == 200:
        row.update(
            {
                "ids": _ids(body),
                "route": body["hipengine"]["generation_shape"]["route"],
                "mtp": body["hipengine"]["speculative_mtp"],
                "usage": body["usage"],
            }
        )
    else:
        row["error"] = body.get("error")
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompts = [
        "Reply with the letter A.",
        "Reply with the letter B. " * 2,
        "Reply with the letter C. " * 4,
        "Reply with the letter D. " * 8,
        "Write one short greeting. " * 3,
        "Write one short farewell. " * 5,
        "Output the integer 7. " * 7,
        "Output the integer 9. " * 9,
    ]
    reset_memory_stats()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    def checkpoint(event: str, details: dict[str, Any]) -> None:
        print(json.dumps({"event": event, **details}, sort_keys=True), file=sys.stderr, flush=True)
        if args.out is not None:
            _atomic_write_json(
                args.out,
                {
                    "schema": 1,
                    "kind": "gguf_mtp_serial_load_checkpoint",
                    "status": "running",
                    "active_event": event,
                    "active_details": details,
                    "rows": rows,
                },
            )

    config = ServerConfig(
        model=str(args.model),
        served_model_name="qwen36-rf5",
        eager_load=True,
        startup_chat_smoke=False,
        startup_scratch_probe=False,
        speculative_mtp_serving="opt_in",
        generation_batch_window_ms=10.0,
        max_active_requests=4,
        max_queued_requests=16,
        request_timeout_ms=None,
    )
    app = create_app(config)
    with TestClient(app) as client:
        capability = client.get("/v1/hipengine/capabilities").json()["sampling"][
            "speculative_mtp"
        ]
        rows.append(
            {
                "id": "declared_policy",
                "physical_concurrency": capability.get("physical_concurrency"),
                "max_physical_target_slots": capability.get("max_physical_target_slots"),
                "route_coalescing_is_physical_concurrency": capability.get(
                    "route_coalescing_is_physical_concurrency"
                ),
                "passed": capability.get("physical_concurrency")
                == "serialized_target_slot"
                and capability.get("max_physical_target_slots") == 1
                and capability.get("route_coalescing_is_physical_concurrency") is False,
            }
        )

        checkpoint("baseline_start", {})
        baseline = [_request(client, prompt, mtp=False) for prompt in prompts]
        baseline_ids = [row["ids"][0] for row in baseline]
        sequential_mtp = [_request(client, prompt, mtp=True) for prompt in prompts]
        rows.append(
            {
                "id": "sequential_baseline",
                "ar_statuses": [row["status_code"] for row in baseline],
                "mtp_statuses": [row["status_code"] for row in sequential_mtp],
                "ids_exact": [row["ids"][0] for row in sequential_mtp] == baseline_ids,
                "passed": all(row["status_code"] == 200 for row in baseline + sequential_mtp)
                and [row["ids"][0] for row in sequential_mtp] == baseline_ids,
            }
        )
        checkpoint("baseline_complete", {"passed": rows[-1]["passed"]})

        for width in (1, 2, 4, 8):
            checkpoint("width_start", {"width": width})
            with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:
                futures = [
                    pool.submit(_request, client, prompts[index], mtp=True)
                    for index in range(width)
                ]
                results = [future.result(timeout=120.0) for future in futures]
            row = {
                "id": f"independent_c{width}",
                "width": width,
                "statuses": [result["status_code"] for result in results],
                "routes": [result.get("route") for result in results],
                "ids_exact": [result.get("ids", [[]])[0] for result in results]
                == baseline_ids[:width],
                "wall_seconds": max(result["wall_seconds"] for result in results),
            }
            row["passed"] = bool(
                all(status == 200 for status in row["statuses"])
                and all(route == "speculative_mtp" for route in row["routes"])
                and row["ids_exact"]
            )
            rows.append(row)
            checkpoint("width_complete", {"width": width, "passed": row["passed"]})

        multi = _request(client, prompts, mtp=True)
        multi_row = {
            "id": "one_multi_prompt_c8",
            "status": multi["status_code"],
            "route": multi.get("route"),
            "ids_exact": multi.get("ids") == baseline_ids,
            "choice_count": len(multi.get("ids", [])),
        }
        multi_row["passed"] = bool(
            multi_row["status"] == 200
            and multi_row["route"] == "speculative_mtp"
            and multi_row["ids_exact"]
            and multi_row["choice_count"] == 8
        )
        rows.append(multi_row)

        checkpoint("mixed_start", {})
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(_request, client, prompts[index], mtp=(index % 2 == 0))
                for index in range(8)
            ]
            mixed = [future.result(timeout=120.0) for future in futures]
        mixed_row = {
            "id": "mixed_ar_mtp_c8",
            "statuses": [result["status_code"] for result in mixed],
            "routes": [result.get("route") for result in mixed],
            "ids_exact": [result.get("ids", [[]])[0] for result in mixed] == baseline_ids,
        }
        mixed_row["passed"] = bool(
            all(status == 200 for status in mixed_row["statuses"])
            and mixed_row["ids_exact"]
            and mixed_row["routes"]
            == ["speculative_mtp" if index % 2 == 0 else "default" for index in range(8)]
        )
        rows.append(mixed_row)
        checkpoint("mixed_complete", {"passed": mixed_row["passed"]})

        timed_out = _request(
            client,
            "deadline request " * 32,
            mtp=True,
            max_tokens=8,
            timeout_ms=1.0,
        )
        health = _request(client, prompts[0], mtp=True)
        deadline_row = {
            "id": "deadline_and_readmission",
            "deadline_status": timed_out["status_code"],
            "deadline_code": (timed_out.get("error") or {}).get("code"),
            "health_status": health["status_code"],
            "health_ids_exact": health.get("ids", [[]])[0] == baseline_ids[0],
        }
        deadline_row["passed"] = bool(
            deadline_row["deadline_status"] == 408
            and deadline_row["deadline_code"] == "deadline_exceeded"
            and deadline_row["health_status"] == 200
            and deadline_row["health_ids_exact"]
        )
        rows.append(deadline_row)

        checkpoint("soak_start", {"requests": int(args.soak_requests)})
        soak_failures: list[dict[str, Any]] = []
        route_counts = {"default": 0, "speculative_mtp": 0}
        for index in range(int(args.soak_requests)):
            mtp = index % 2 == 0
            prompt_index = index % len(prompts)
            result = _request(
                client,
                prompts[prompt_index],
                mtp=mtp,
                max_tokens=2,
            )
            route = result.get("route")
            if route in route_counts:
                route_counts[route] += 1
            if (
                result["status_code"] != 200
                or result.get("ids", [[]])[0][:1] != baseline_ids[prompt_index][:1]
            ):
                soak_failures.append(
                    {
                        "index": index,
                        "status": result["status_code"],
                        "route": route,
                    }
                )
            if (index + 1) % 20 == 0:
                checkpoint("soak_progress", {"completed": index + 1, "failures": len(soak_failures)})
        soak_row = {
            "id": "focused_lifecycle_soak",
            "requests": int(args.soak_requests),
            "route_counts": route_counts,
            "failures": soak_failures,
            "passed": not soak_failures,
        }
        rows.append(soak_row)
        checkpoint("soak_complete", {"passed": soak_row["passed"]})

        ready_before_shutdown = client.get("/ready").status_code
    final_memory = memory_stats()
    rows.append(
        {
            "id": "resource_and_shutdown",
            "ready_before_shutdown": ready_before_shutdown,
            "memory": final_memory,
            "passed": ready_before_shutdown == 200
            and final_memory["current_allocated_bytes"] == 0
            and final_memory["active_allocations"] == 0,
        }
    )

    passed = all(bool(row["passed"]) for row in rows)
    payload = {
        "schema": 1,
        "kind": "gguf_mtp_serial_load_gate",
        "status": "passed" if passed else "failed",
        "verdict": "pass" if passed else "fail",
        "performance_claim": False,
        "policy": "serialized_dense_mtp_explicit_only",
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
        "soak_scope": {
            "focused_requests": int(args.soak_requests),
            "promotion_soak_required": False,
            "reason": "no automatic MTP scope; RF2/RF6 economics route production to AR",
        },
        "passed": passed,
    }
    if args.out is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _atomic_write_json(args.out, payload)
        print(f"wrote {args.out}: passed={passed} rows={len(rows)}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--soak-requests", type=int, default=100)
    parser.add_argument("--hash-model", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args(argv)
    if not args.model.is_file() or args.soak_requests <= 0:
        raise SystemExit("model must exist and soak requests must be positive")
    payload = run(args)
    return 1 if args.fail_on_fail and not payload["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
