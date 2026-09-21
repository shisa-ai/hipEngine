#!/usr/bin/env python3
"""Replay a recorded agent session against any OpenAI-compatible endpoint.

This is the cross-engine agentic comparison surface. It drives an endpoint over
HTTP only, so the same requests, in the same order, can be sent to hipEngine,
llama.cpp, or any other OpenAI-compatible server and the resulting rates are
attributable to the engine rather than to the client.

The workload is a recorded coding-agent session replayed as its client actually
sent it: every request resends the whole transcript, which grows by one
assistant tool-call round plus its tool results. That cumulative-resend shape is
what makes an agent session a prefix-cache workload, and it is why the session
wall time -- not a single-request decode rate -- is the headline number here.

Request fidelity is the point of the script, so it does not flatten the
transcript. Assistant ``tool_calls``, the ``tool_call_id`` on each tool result,
and the fixture's ``tools`` array are all sent verbatim, because a flattened
prompt would prefill different tokens on each engine and the comparison would
stop being a comparison.

Every engine reports its own ``usage``, and both are asked for it explicitly.
Two rate definitions are recorded per request because they answer different
questions:

``decode_tok_s``
    ``(completion_tokens - 1) / (last_delta - first_delta)``. The first token
    arrives at the end of prefill, so it is not part of any decode interval.
    Counting it would credit the engine with a token it never decoded.
``session_wall_s``
    the whole pass: prefill, decode, HTTP, and client overhead. This is what a
    user waits for.

Usage:
    scripts/agentic_session_compare.py --base-url http://127.0.0.1:8080 \\
        --model qwen3.8-27b-q4km --label hipengine --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "benchmarks/prompts/agentic-session-replay-v1.json"


def load_fixture(path: Path) -> dict[str, Any]:
    """Return the single replayed workload plus the request boundaries.

    ``request_ends`` are prefix lengths: request ``k`` sends
    ``entries[:request_ends[k]]``. They are the client's real boundaries, so
    they are used as given rather than re-derived.
    """
    document = json.loads(path.read_text())
    workloads = document.get("workloads") or []
    if len(workloads) != 1:
        raise SystemExit(f"{path}: expected exactly one workload, found {len(workloads)}")
    workload = workloads[0]
    entries = workload.get("entries") or []
    request_ends = workload.get("request_ends") or []
    if not entries or not request_ends:
        raise SystemExit(f"{path}: workload has no entries or no request boundaries")
    if request_ends != sorted(request_ends) or len(set(request_ends)) != len(request_ends):
        raise SystemExit(f"{path}: request_ends must be strictly increasing")
    if request_ends[-1] > len(entries):
        raise SystemExit(
            f"{path}: request_ends reaches {request_ends[-1]} but only {len(entries)} entries exist"
        )
    return {"document": document, "workload": workload, "entries": entries,
            "request_ends": list(request_ends)}


def build_messages(system_prompt: str, entries: list[dict[str, Any]], end: int) -> list[dict[str, Any]]:
    """Messages for one request: the system prompt plus the transcript prefix.

    Entry fields are passed through unchanged. Only the field set the chat
    renderer consumes is kept, and nothing is dropped that would change the
    rendered tokens.
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for entry in entries[:end]:
        message: dict[str, Any] = {"role": entry["role"]}
        content = entry.get("content")
        message["content"] = "" if content is None else content
        if entry.get("tool_calls"):
            message["tool_calls"] = entry["tool_calls"]
        if entry.get("tool_call_id"):
            message["tool_call_id"] = entry["tool_call_id"]
        if entry.get("name"):
            message["name"] = entry["name"]
        messages.append(message)
    return messages


def request_body(args: argparse.Namespace, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": True,
        # Both engines are asked for usage on the stream. An engine that ignores
        # this is reported as such rather than silently measured differently.
        "stream_options": {"include_usage": True},
    }
    if tools and not args.no_tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if args.seed is not None:
        body["seed"] = args.seed
    return body


def stream_once(args: argparse.Namespace, body: dict[str, Any]) -> dict[str, Any]:
    """Send one streaming request and time it. Returns the turn record."""
    payload = json.dumps(body).encode()
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    first_delta: float | None = None
    last_delta: float | None = None
    deltas = 0
    usage: dict[str, Any] | None = None
    tool_call_chunks = 0
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        now = time.perf_counter()
                        if first_delta is None:
                            first_delta = now
                        last_delta = now
                        deltas += 1
                    if delta.get("tool_calls"):
                        tool_call_chunks += len(delta["tool_calls"])
    except urllib.error.HTTPError as exc:
        error = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}"
    except Exception as exc:  # noqa: BLE001 - any transport failure is a turn failure
        error = f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - started

    record: dict[str, Any] = {
        "total_s": round(total, 4),
        "ttft_s": round(first_delta - started, 4) if first_delta is not None else None,
        "content_deltas": deltas,
        "tool_call_chunks": tool_call_chunks,
        "usage": usage,
        "error": error,
    }
    completion_tokens = (usage or {}).get("completion_tokens")
    if first_delta is not None and last_delta is not None and last_delta > first_delta:
        decode_window = last_delta - first_delta
        # The token that arrived at TTFT was produced during prefill, so it is
        # not counted in the decode interval.
        decoded = (completion_tokens - 1) if isinstance(completion_tokens, int) else (deltas - 1)
        record["decode_window_s"] = round(decode_window, 4)
        record["decode_tokens"] = decoded
        record["decode_tok_s"] = round(decoded / decode_window, 4) if decoded > 0 else None
    else:
        record["decode_window_s"] = None
        record["decode_tokens"] = None
        record["decode_tok_s"] = None
    return record


def summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(values),
        "median": round(statistics.median(ordered), 4),
        "mean": round(statistics.fmean(ordered), 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def run_pass(args: argparse.Namespace, fixture: dict[str, Any], pass_index: int,
             tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Replay every request boundary once. One pass is one agent session."""
    entries = fixture["entries"]
    system_prompt = fixture["workload"].get("system_prompt") or ""
    turns: list[dict[str, Any]] = []
    pass_started = time.perf_counter()
    for index, end in enumerate(fixture["request_ends"]):
        messages = build_messages(system_prompt, entries, end)
        body = request_body(args, messages, tools)
        record = stream_once(args, body)
        record["turn"] = index
        record["prefix_entries"] = end
        record["request_chars"] = len(json.dumps(body["messages"]))
        record["prompt_tokens"] = (record.get("usage") or {}).get("prompt_tokens")
        turns.append(record)
        status = "ok" if record["error"] is None else "FAIL"
        rate = record["decode_tok_s"]
        print(f"  pass {pass_index} turn {index + 1}/{len(fixture['request_ends'])} "
              f"prefix={end} prompt={record['prompt_tokens']} "
              f"ttft={record['ttft_s']}s decode={rate} tok/s total={record['total_s']}s {status}",
              flush=True)
        if record["error"] is not None:
            print(f"    error: {record['error']}", flush=True)
    pass_wall = time.perf_counter() - pass_started
    return {
        "pass": pass_index,
        "session_wall_s": round(pass_wall, 4),
        "turns": turns,
        "prompt_tokens_total": sum(t["prompt_tokens"] for t in turns
                                   if isinstance(t.get("prompt_tokens"), int)),
        "completion_tokens_total": sum((t.get("usage") or {}).get("completion_tokens", 0) or 0
                                       for t in turns),
        "errors": sum(1 for t in turns if t["error"] is not None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay a recorded agent session against any OpenAI-compatible endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", required=True,
                        help="Engine root URL, without /v1 (for example http://127.0.0.1:8080)")
    parser.add_argument("--model", required=True, help="Served model name to send in each request")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--passes", type=int, default=2,
                        help="Whole-session replays. Pass 1 is cold; later passes reuse the "
                             "engine's own prefix cache and are reported separately.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-tools", action="store_true",
                        help="Omit the fixture's tools array, for engines that reject it")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the payloads that would be sent and exit without any request")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    fixture = load_fixture(args.fixture)
    tools = fixture["document"].get("tools") or None
    entries = fixture["entries"]
    system_prompt = fixture["workload"].get("system_prompt") or ""

    if args.dry_run:
        # Offline fidelity check: no server is contacted, so this runs while the
        # GPU is busy with someone else's work.
        print(f"fixture: {args.fixture}")
        print(f"entries: {len(entries)} | requests: {len(fixture['request_ends'])} | "
              f"tools: {len(tools or [])} | system prompt chars: {len(system_prompt)}")
        body = request_body(args, build_messages(system_prompt, entries,
                                                 fixture["request_ends"][-1]), tools)
        for index, end in enumerate(fixture["request_ends"]):
            messages = build_messages(system_prompt, entries, end)
            print(f"  request {index + 1}: {len(messages)} messages, prefix entries {end}, "
                  f"last role {messages[-1]['role']}")
        blob = json.dumps(body, sort_keys=True).encode()
        print(f"largest request payload: {len(blob)} bytes, sha256 {hashlib.sha256(blob).hexdigest()[:16]}")
        print(f"assistant messages carrying tool_calls: "
              f"{sum(1 for m in body['messages'] if m.get('tool_calls'))}")
        print(f"tool messages carrying tool_call_id: "
              f"{sum(1 for m in body['messages'] if m.get('tool_call_id'))}")
        return 0

    report: dict[str, Any] = {
        "schema": "hipengine.agentic_session_replay_http.v1",
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "fixture": str(args.fixture),
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "requests_per_pass": len(fixture["request_ends"]),
        "passes_requested": args.passes,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "tools_sent": bool(tools) and not args.no_tools,
        "workload_note": (
            "Recorded agent session replayed as its client sent it: every request resends the "
            "whole transcript, growing by one assistant tool-call round plus its tool results. "
            "Generated text is not compared across engines; this measures the workload shape."
        ),
    }
    passes = [run_pass(args, fixture, index + 1, tools) for index in range(args.passes)]
    report["passes"] = passes

    cold, *warm = passes
    report["result"] = {
        "cold_session_wall_s": cold["session_wall_s"],
        "cold_ttft_s_median": summarise([t["ttft_s"] for t in cold["turns"] if t["ttft_s"] is not None]).get("median"),
        "cold_decode_tok_s_median": summarise([t["decode_tok_s"] for t in cold["turns"]
                                               if t["decode_tok_s"] is not None]).get("median"),
        "warm_session_wall_s": [p["session_wall_s"] for p in warm],
        "warm_ttft_s_median": summarise([t["ttft_s"] for p in warm for t in p["turns"]
                                         if t["ttft_s"] is not None]).get("median"),
        "warm_decode_tok_s_median": summarise([t["decode_tok_s"] for p in warm for t in p["turns"]
                                               if t["decode_tok_s"] is not None]).get("median"),
        "ttft_by_turn_median_s": [
            summarise([p["turns"][i]["ttft_s"] for p in passes
                       if p["turns"][i]["ttft_s"] is not None]).get("median")
            for i in range(len(fixture["request_ends"]))
        ],
        "errors_total": sum(p["errors"] for p in passes),
    }
    print(f"\n{args.label or args.base_url}: cold session {cold['session_wall_s']}s, "
          f"warm {[p['session_wall_s'] for p in warm]}s, "
          f"errors {report['result']['errors_total']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
