#!/usr/bin/env python3
"""Exercise INT8 MTP through a running OpenAI-compatible hipEngine server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]


def blocking_result(body: dict[str, Any]) -> dict[str, Any]:
    choice = body["choices"][0]
    metadata = choice["hipengine"]
    return {
        "ids": metadata["generated_token_ids"],
        "usage": body["usage"],
        "finish_reason": choice["finish_reason"],
        "diagnostics": metadata.get("diagnostics", {}),
        "cycles": int(metadata.get("timing", {}).get("mtp_cycles_count", 0)),
    }


def stream_result(lines) -> dict[str, Any]:
    final = None
    usage = None
    done = False
    for line in lines:
        if not line.startswith("data: "):
            continue
        if done:
            raise AssertionError("SSE data after DONE")
        payload = line[6:]
        if payload == "[DONE]":
            done = True
            continue
        event = json.loads(payload)
        if "error" in event:
            raise AssertionError(f"SSE error: {event['error']}")
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("finish_reason") is not None:
                if final is not None:
                    raise AssertionError("multiple terminal choices")
                final = choice
    if not done or final is None or usage is None:
        raise AssertionError("incomplete SSE terminal/usage/DONE sequence")
    return blocking_result({"choices": [final], "usage": usage})


def assert_result(result, *, speculative, compact):
    assert result["usage"]["completion_tokens"] == len(result["ids"])
    assert result["usage"]["total_tokens"] == (
        result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]
    )
    assert result["finish_reason"] in {"stop", "length"}
    layout = result["diagnostics"]["kv_layout"]
    assert layout["storage_dtype"] == "int8_per_token_head", layout
    assert layout["scale_dtype"] == "fp32", layout
    if compact:
        assert layout["kv_attention_source"] == "int8_direct", layout
        assert layout["persistent_bf16_mirror_bytes"] == 0, layout
    if speculative:
        assert result["cycles"] > 0, result["diagnostics"]
    else:
        assert result["cycles"] == 0, result["diagnostics"]


def run(args):
    rows = []
    for filename in ("mtpbench-code-general-ja.jsonl", "gdn-prefill-category-heldouts.jsonl"):
        path = ROOT / "benchmarks" / "prompts" / filename
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if args.limit:
        rows = rows[:args.limit]
    report = {
        "command": sys.argv,
        "scope": "http_correctness",
        "performance_claim": False,
        "complete_prompt_suite": not bool(args.limit),
        "compact_required": not args.allow_mirror,
        "rows": [],
        "passed": False,
    }
    try:
        with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
            ready = client.get("/ready")
            ready.raise_for_status()
            ready = ready.json()
            capability = ready["model"]["kv_capability"]
            report["kv_capability"] = capability
            report["startup"] = ready["startup"]
            assert capability["effective_kv_storage"] == "int8_per_token_head", capability
            if capability.get("diagnostic_override"):
                assert args.allow_kv_diagnostic_override, "KV quality diagnostic override must be acknowledged"
            for row in rows:
                for endpoint in ("/v1/completions", "/v1/chat/completions"):
                    prompt = (
                        {"prompt": "\n".join(message["content"] for message in row["messages"])}
                        if endpoint == "/v1/completions"
                        else {"messages": row["messages"], "chat_template_kwargs": {"enable_thinking": False}}
                    )
                    base = {
                        "model": args.model, "max_tokens": args.max_tokens,
                        "temperature": 0, **prompt,
                    }
                    results = []
                    for speculative, streamed in ((False, False), (True, False), (True, True)):
                        payload = {**base, "speculative_mtp": speculative, "stream": streamed}
                        if streamed:
                            payload["stream_options"] = {"include_usage": True, "include_hipengine": True}
                            with client.stream("POST", endpoint, json=payload) as response:
                                response.raise_for_status()
                                result = stream_result(response.iter_lines())
                        else:
                            response = client.post(endpoint, json=payload)
                            response.raise_for_status()
                            result = blocking_result(response.json())
                        assert_result(result, speculative=speculative, compact=not args.allow_mirror)
                        results.append(result)
                    assert results[0]["ids"] == results[1]["ids"] == results[2]["ids"], {
                        "prompt": row["id"], "endpoint": endpoint, "ids": [result["ids"] for result in results],
                    }
                    assert len({result["usage"]["prompt_tokens"] for result in results}) == 1
                    report["rows"].append({
                        "id": row["id"], "category": row["category"], "endpoint": endpoint,
                        "ids": results[0]["ids"], "usage": results[0]["usage"],
                        "blocking_mtp_cycles": results[1]["cycles"],
                        "stream_mtp_cycles": results[2]["cycles"],
                    })
                    print(f"PASS {row['id']} {endpoint}", flush=True)
            final_ready = client.get("/ready")
            final_ready.raise_for_status()
            report["final_queue"] = final_ready.json()["queue"]
            assert report["final_queue"]["active_requests"] == 0
            assert report["final_queue"]["depth"] == 0
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "rows": len(report["rows"]), "error": report.get("error")}))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--model", default="int8-mtp")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--limit", type=int, default=0, help="Diagnostic subset; not complete-suite evidence")
    parser.add_argument("--allow-mirror", action="store_true")
    parser.add_argument("--allow-kv-diagnostic-override", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.max_tokens < 4 or args.limit < 0 or args.timeout <= 0:
        parser.error("max-tokens must be >=4, limit >=0, and timeout >0")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
