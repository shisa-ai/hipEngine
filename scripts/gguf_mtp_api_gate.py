#!/usr/bin/env python3
"""RF4 real OpenAI API semantics gate for dense GGUF MTP.

Runs eager and lazy/restart server lifecycles. It binds completion/chat routing,
single/multi prompt behavior, implicit auto fallback, explicit MTP/AR, usage and
hipEngine extensions, thinking hint/hard policy, streaming/incompatible
rejection, app-local sessions, capabilities, and post-restart health. Direct
response objects are the evidence source; no external telemetry is required.
"""

from __future__ import annotations

import argparse
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

from hipengine.server import ServerConfig, create_app
from scripts.gguf_mtp_long_context_gate import _atomic_write_json, _git

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
AUTO_REASON = "automatic_mtp_scope_not_promoted"


def sse_payloads(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in str(text).splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        rows.append(json.loads(line[len("data: ") :]))
    return rows


def _generated_ids(body: dict[str, Any]) -> list[list[int]]:
    return [
        [int(token) for token in choice["hipengine"]["generated_token_ids"]]
        for choice in body["choices"]
    ]


def _mtp_contract(body: dict[str, Any], *, used: bool) -> bool:
    extension = body.get("hipengine", {}).get("speculative_mtp", {})
    details = body.get("usage", {}).get("completion_tokens_details", {})
    if bool(extension.get("used")) is not bool(used):
        return False
    if used:
        return (
            extension.get("effective_route") == "speculative_mtp"
            and "accepted_prediction_tokens" in details
            and "rejected_prediction_tokens" in details
            and extension.get("thinking_policy") in {"hint", "hard"}
        )
    return (
        "accepted_prediction_tokens" not in details
        and "rejected_prediction_tokens" not in details
    )


def _post(client: TestClient, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(endpoint, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"{endpoint} status={response.status_code}: {response.text}")
    return response.json()


def _run_server(model: Path, *, eager_load: bool) -> dict[str, Any]:
    config = ServerConfig(
        model=str(model),
        served_model_name="qwen36-rf4",
        eager_load=bool(eager_load),
        startup_chat_smoke=False,
        startup_scratch_probe=False,
        speculative_mtp_serving="auto",
        speculative_mtp_thinking="hint",
        max_active_requests=4,
        generation_batch_window_ms=0.0,
    )
    app = create_app(config)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with TestClient(app) as client:
        ready_before = client.get("/ready")
        model_before = client.get("/v1/models").json()["data"][0]["hipengine"]

        base_completion = {
            "model": "qwen36-rf4",
            "prompt": "Write exactly one short greeting.",
            "max_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        ar_completion = _post(client, "/v1/completions", dict(base_completion))
        mtp_completion = _post(
            client,
            "/v1/completions",
            {**base_completion, "speculative_mtp": True},
        )
        rows.append(
            {
                "id": "completion_single",
                "ar_ids": _generated_ids(ar_completion),
                "mtp_ids": _generated_ids(mtp_completion),
                "ids_exact": _generated_ids(ar_completion) == _generated_ids(mtp_completion),
                "ar_contract": _mtp_contract(ar_completion, used=False),
                "mtp_contract": _mtp_contract(mtp_completion, used=True),
                "auto_reason": ar_completion["hipengine"]["speculative_mtp"].get(
                    "decision_reason"
                ),
            }
        )

        multi = _post(
            client,
            "/v1/completions",
            {
                **base_completion,
                "prompt": ["Reply with A.", "Reply with B."],
                "speculative_mtp": True,
            },
        )
        rows.append(
            {
                "id": "completion_multi_prompt",
                "choice_count": len(multi["choices"]),
                "mtp_contract": _mtp_contract(multi, used=True),
                "route": multi["hipengine"]["generation_shape"]["route"],
            }
        )

        base_chat = {
            "model": "qwen36-rf4",
            "messages": [{"role": "user", "content": "Write exactly one short greeting."}],
            "max_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        ar_chat = _post(client, "/v1/chat/completions", dict(base_chat))
        mtp_chat = _post(
            client,
            "/v1/chat/completions",
            {**base_chat, "speculative_mtp": True},
        )
        rows.append(
            {
                "id": "chat_single",
                "ar_ids": _generated_ids(ar_chat),
                "mtp_ids": _generated_ids(mtp_chat),
                "ids_exact": _generated_ids(ar_chat) == _generated_ids(mtp_chat),
                "ar_contract": _mtp_contract(ar_chat, used=False),
                "mtp_contract": _mtp_contract(mtp_chat, used=True),
                "auto_reason": ar_chat["hipengine"]["speculative_mtp"].get(
                    "decision_reason"
                ),
            }
        )

        thinking_hint = _post(
            client,
            "/v1/chat/completions",
            {
                **base_chat,
                "reasoning_effort": "medium",
                "speculative_mtp": {"enabled": True, "thinking": "hint"},
            },
        )
        hint_extension = thinking_hint["hipengine"]["speculative_mtp"]
        hard = client.post(
            "/v1/chat/completions",
            json={
                **base_chat,
                "reasoning_effort": "medium",
                "speculative_mtp": {"enabled": True, "thinking": "hard"},
            },
        )
        rows.append(
            {
                "id": "thinking_policy",
                "hint_route": hint_extension.get("effective_route"),
                "hint_policy": hint_extension.get("thinking_policy"),
                "hint_controls": hint_extension.get("thinking_controls"),
                "hard_status": hard.status_code,
                "hard_blockers": hard.json().get("error", {})
                .get("hipengine", {})
                .get("speculative_mtp", {})
                .get("blockers"),
            }
        )

        for content in ("remember alpha", "now beta"):
            _post(
                client,
                "/v1/chat/completions",
                {
                    **base_chat,
                    "messages": [{"role": "user", "content": content}],
                    "session": {"id": "rf4_session", "commit": "append_all"},
                    "speculative_mtp": True,
                },
            )
        sessions = client.get("/v1/hipengine/sessions").json()["sessions"]
        stored = next(row for row in sessions if row["id"] == "rf4_session")
        rows.append(
            {
                "id": "app_local_session",
                "message_count": stored["message_count"],
            }
        )

        non_greedy = client.post(
            "/v1/completions",
            json={**base_completion, "temperature": 0.4, "speculative_mtp": True},
        )
        streaming = client.post(
            "/v1/completions",
            json={**base_completion, "stream": True, "speculative_mtp": True},
        )
        stream_errors = [row["error"] for row in sse_payloads(streaming.text) if row.get("error")]
        rows.append(
            {
                "id": "explicit_rejections",
                "non_greedy_status": non_greedy.status_code,
                "non_greedy_blockers": non_greedy.json().get("error", {})
                .get("hipengine", {})
                .get("speculative_mtp", {})
                .get("blockers"),
                "stream_http_status": streaming.status_code,
                "stream_error_code": stream_errors[0].get("code") if stream_errors else None,
                "stream_error_param": stream_errors[0].get("param") if stream_errors else None,
            }
        )

        capabilities_after = client.get("/v1/hipengine/capabilities").json()
        mtp_capability = capabilities_after["sampling"]["speculative_mtp"]
        ready_after = client.get("/ready")
        model_after = client.get("/v1/models").json()["data"][0]["hipengine"]
        rows.append(
            {
                "id": "capability_and_readiness",
                "ready_before_status": ready_before.status_code,
                "ready_after_status": ready_after.status_code,
                "loaded_before": model_before["loaded"],
                "loaded_after": model_after["loaded"],
                "configured_policy": mtp_capability["configured_policy"],
                "engine_supported": mtp_capability["engine_supported"],
                "certified_default_scopes": mtp_capability["certified_default_scopes"],
                "automatic_route_promoted": mtp_capability["automatic_route_promoted"],
                "auto_reason": mtp_capability["auto_route"]["reason"],
            }
        )

    for row in rows:
        if row["id"] in {"completion_single", "chat_single"}:
            row["passed"] = bool(
                row["ids_exact"]
                and row["ar_contract"]
                and row["mtp_contract"]
                and row["auto_reason"] == AUTO_REASON
            )
        elif row["id"] == "completion_multi_prompt":
            row["passed"] = bool(
                row["choice_count"] == 2
                and row["mtp_contract"]
                and row["route"] == "speculative_mtp"
            )
        elif row["id"] == "thinking_policy":
            row["passed"] = bool(
                row["hint_route"] == "speculative_mtp"
                and row["hint_policy"] == "hint"
                and row["hint_controls"] == "prompt_hint_only"
                and row["hard_status"] == 400
                and row["hard_blockers"] == ["thinking_budget"]
            )
        elif row["id"] == "app_local_session":
            row["passed"] = row["message_count"] == 4
        elif row["id"] == "explicit_rejections":
            row["passed"] = bool(
                row["non_greedy_status"] == 400
                and row["non_greedy_blockers"] == ["temperature"]
                and row["stream_http_status"] == 200
                and row["stream_error_code"] == "unsupported_parameter"
                and row["stream_error_param"] == "speculative_mtp"
            )
        elif row["id"] == "capability_and_readiness":
            row["passed"] = bool(
                row["ready_after_status"] == 200
                and row["loaded_after"] is True
                and row["configured_policy"] == "auto"
                and row["engine_supported"] is True
                and row["certified_default_scopes"] == []
                and row["automatic_route_promoted"] is False
                and row["auto_reason"] == AUTO_REASON
            )
    return {
        "eager_load": bool(eager_load),
        "rows": rows,
        "passed": all(bool(row.get("passed")) for row in rows),
        "wall_seconds": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    eager = _run_server(args.model, eager_load=True)
    lazy_restart = _run_server(args.model, eager_load=False)
    passed = bool(eager["passed"] and lazy_restart["passed"])
    payload = {
        "schema": 1,
        "kind": "gguf_mtp_api_semantics_gate",
        "status": "passed" if passed else "failed",
        "verdict": "pass" if passed else "fail",
        "performance_claim": False,
        "command": [sys.executable, *sys.argv],
        "repo": {"commit": _git(["rev-parse", "HEAD"]), "dirty": bool(_git(["status", "--porcelain"]))},
        "model": {
            "path": str(args.model.resolve()),
            "size_bytes": args.model.stat().st_size,
            "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest() if args.hash_model else None,
        },
        "eager": eager,
        "lazy_restart": lazy_restart,
        "summary": {"passed": 2 if passed else int(eager["passed"]) + int(lazy_restart["passed"]), "total": 2, "wall_seconds": time.perf_counter() - started},
        "passed": passed,
    }
    if args.out is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _atomic_write_json(args.out, payload)
        print(f"wrote {args.out}: passed={passed}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--hash-model", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    payload = run(args)
    return 1 if args.fail_on_fail and not payload["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
