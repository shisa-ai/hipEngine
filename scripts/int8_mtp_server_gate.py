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


def request_payload(args, row, endpoint, *, speculative, streamed):
    prompt = (
        {"prompt": "\n".join(message["content"] for message in row["messages"])}
        if endpoint == "/v1/completions"
        else {"messages": row["messages"], "chat_template_kwargs": {"enable_thinking": False}}
    )
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "speculative_mtp": speculative,
        "stream": streamed,
        **prompt,
    }
    if streamed:
        payload["stream_options"] = {"include_usage": True, "include_hipengine": True}
    return payload


def run_triple(client, args, row, endpoint):
    """Run the AR/MTP/MTP-stream triple for one row and assert their agreement."""

    results = []
    for speculative, streamed in ((False, False), (True, False), (True, True)):
        payload = request_payload(args, row, endpoint, speculative=speculative, streamed=streamed)
        if streamed:
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
        "prompt": row["id"],
        "endpoint": endpoint,
        "ids": [result["ids"] for result in results],
    }
    assert len({result["usage"]["prompt_tokens"] for result in results}) == 1
    return results


def single_row_ids(client, args, row, endpoint, *, speculative=True, streamed=False):
    """Fetch one row's ids in one configuration, for the determinism checks."""

    payload = request_payload(args, row, endpoint, speculative=speculative, streamed=streamed)
    if streamed:
        with client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            return stream_result(response.iter_lines())["ids"]
    response = client.post(endpoint, json=payload)
    response.raise_for_status()
    return blocking_result(response.json())["ids"]


def check_repeatability(client, args, schedule, endpoint, baselines):
    """Re-run the whole schedule and require identical ids row by row.

    A second schedule over the same work must not change any row's ids. This
    catches state that leaks forward between requests -- a pooled buffer or a
    retained snapshot whose contents depend on what ran before it -- which a
    single forward pass cannot see because every row runs against the same
    history both times.

    ``schedule`` is the row order for the second pass; the caller passes the
    reverse of the first so the history each row sees is genuinely different.
    """

    mismatches = []
    for row in schedule:
        repeat = single_row_ids(client, args, row, endpoint)
        baseline = baselines[(row["id"], endpoint)]
        if repeat != baseline:
            mismatches.append({
                "id": row["id"], "endpoint": endpoint,
                "first_schedule": baseline, "second_schedule": repeat,
            })
    assert not mismatches, {"repeatability": mismatches}
    return len(schedule)


def check_isolation(client, args, target, neighbour, endpoint, baseline):
    """Run the target beside a neighbouring request and require its own ids.

    The neighbour runs in a thread on the same server, so the two requests are
    in flight together. The target must reproduce the ids it produced alone; if
    a shared pool, slot, or packed workspace leaks between concurrent requests,
    the target's ids drift and this fails. The neighbour's own ids are checked
    too, so a run where both were corrupted cannot pass by agreeing.
    """

    import threading

    neighbour_ids: list[list[int]] = []
    neighbour_error: list[BaseException] = []

    def run_neighbour():
        try:
            neighbour_ids.append(single_row_ids(client, args, neighbour, endpoint))
        except BaseException as exc:  # surfaced below, not swallowed
            neighbour_error.append(exc)

    thread = threading.Thread(target=run_neighbour)
    thread.start()
    try:
        target_ids = single_row_ids(client, args, target, endpoint)
    finally:
        thread.join(timeout=args.timeout)

    assert not thread.is_alive(), "neighbouring request did not finish"
    assert not neighbour_error, {"neighbour_error": repr(neighbour_error[0])}
    assert target_ids == baseline, {
        "isolation": {"id": target["id"], "endpoint": endpoint,
                      "alone": baseline, "with_neighbour": target_ids},
    }
    return {"id": target["id"], "endpoint": endpoint, "ids": target_ids,
            "neighbour_id": neighbour["id"], "neighbour_ids": neighbour_ids[0]}


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
        "determinism_required": not args.skip_determinism,
        "compact_required": not args.allow_mirror,
        "rows": [],
        "passed": False,
    }
    baselines: dict[tuple[str, str], list[int]] = {}
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
                    results = run_triple(client, args, row, endpoint)
                    baselines[(row["id"], endpoint)] = results[0]["ids"]
                    report["rows"].append({
                        "id": row["id"], "category": row["category"], "endpoint": endpoint,
                        "ids": results[0]["ids"], "usage": results[0]["usage"],
                        "blocking_mtp_cycles": results[1]["cycles"],
                        "stream_mtp_cycles": results[2]["cycles"],
                    })
                    print(f"PASS {row['id']} {endpoint}", flush=True)
            if args.skip_determinism:
                report["determinism"] = {"skipped": True}
            else:
                # Second schedule over the same work, in the opposite order, so
                # each row runs against a different request history than it did
                # in the first pass.
                endpoint = "/v1/completions"
                checked = check_repeatability(
                    client, args, list(reversed(rows)), endpoint, baselines
                )
                # The neighbour is a different row from the same suite, run
                # concurrently with the target rather than before it.
                target, neighbour = rows[0], rows[-1]
                isolation = check_isolation(
                    client, args, target, neighbour, endpoint,
                    baselines[(target["id"], endpoint)],
                )
                report["determinism"] = {
                    "skipped": False,
                    "repeatability_rows": checked,
                    "repeatability_endpoint": endpoint,
                    "second_schedule": "reversed",
                    "isolation": isolation,
                }
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
    parser.add_argument(
        "--skip-determinism",
        action="store_true",
        help="Skip the repeatability and isolation checks; not complete-suite evidence",
    )
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.max_tokens < 4 or args.limit < 0 or args.timeout <= 0:
        parser.error("max-tokens must be >=4, limit >=0, and timeout >0")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
