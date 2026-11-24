#!/usr/bin/env python3
"""Ordinary (non-speculative) streaming A/B driver for two OpenAI-compatible servers.

Measures, for one prompt and one engine per request, the time from request
start to the first non-empty text delta (content or reasoning) and the time
to stream close, then reports medians across repetitions and both engine
orders. This replaces the off-tree driver used for the November 2026 Strix
Halo comparison; the protocol matches it: one warmup request per engine and
prompt, N measured repetitions, alternating which engine goes first.

Example
-------
    python scripts/inkling2_ordinary_pairs.py --server a=http://127.0.0.1:8080 \
        --server b=http://127.0.0.1:8081 --model inkling2 \
        --prompts prompts.jsonl --output pairs.json

The prompts file is JSONL with ``{"id": ..., "prompt": ...}`` (a
``messages`` array is also accepted). No GPU is required to import this
module or to run its tests; the measurements need two live servers.

Sampling is greedy (temperature 0) and no engine-specific options are sent,
so both sides receive byte-identical requests. Engine-side profiles, quant
and KV layout must be pinned on the servers themselves; the driver records
echoes such as the ``hipengine`` metadata block when a server returns one,
but does not enforce them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def iter_sse_events(lines):
    """Yield decoded JSON events from an SSE byte/str line iterator.

    Stops at ``data: [DONE]``. Lines that are not ``data:`` fields are ignored,
    as are blank lines. A ``data:`` payload that is not valid JSON raises
    ValueError; callers that want to tolerate broken streams must catch it.
    """
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        line = line.rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


def event_deltas(event):
    """Return ``(content, reasoning)`` text carried by one SSE event (choice 0)."""
    for choice in event.get("choices", []):
        if int(choice.get("index", 0)) != 0:
            continue
        delta = choice.get("delta", choice.get("message", choice)) or {}
        content = delta.get("content") or delta.get("text") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        return content, reasoning
    return "", ""


def carries_text(delta):
    return bool(delta) and any(not ch.isspace() for ch in delta)


def summarize_events(events, start, first_text_at, end):
    """Collapse one streamed response into a record.

    ``start``, ``first_text_at`` and ``end`` are monotonic timestamps;
    ``first_text_at`` is None when no text delta ever arrived.
    """
    content, reasoning, finish_reason, usage, metadata = [], [], None, None, {}
    for event in events:
        metadata.update(event.get("hipengine") or {})
        if event.get("usage"):
            usage = event["usage"]
        content_delta, reasoning_delta = event_deltas(event)
        content.append(content_delta)
        reasoning.append(reasoning_delta)
        for choice in event.get("choices", []):
            if int(choice.get("index", 0)) == 0 and choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    return {
        "first_text_seconds": None if first_text_at is None else first_text_at - start,
        "total_seconds": end - start,
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "finish_reason": finish_reason,
        "usage": usage,
        "hipengine": metadata or None,
    }


def median(values):
    values = list(values)
    return statistics.median(values) if values else None


def summarize_records(records):
    """Group per-request records by (prompt, engine) and report medians."""
    groups = {}
    for record in records:
        groups.setdefault((record["prompt_id"], record["engine"]), []).append(record)
    rows = []
    for (prompt_id, engine), items in sorted(groups.items()):
        ok = [item for item in items if item.get("error") is None]
        failed = len(items) - len(ok)
        rows.append({
            "prompt_id": prompt_id,
            "engine": engine,
            "requests": len(items),
            "failed": failed,
            "median_first_text_seconds": median(r["first_text_seconds"] for r in ok),
            "median_total_seconds": median(r["total_seconds"] for r in ok),
            "prompt_tokens": next((r["usage"].get("prompt_tokens") for r in ok
                                   if isinstance(r.get("usage"), dict) and r["usage"].get("prompt_tokens") is not None), None),
        })
    return rows


def _request_payload(messages, model, max_tokens, temperature):
    if not model:
        raise ValueError("no model id: pass --model or set one per prompt in the JSONL")
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }


def stream_once(url, api_key, payload, timeout, now=time.monotonic):
    """POST ``payload`` and stream the response; returns the summarize_events dict.

    ``now`` is injectable for tests. Raises urllib errors on HTTP failure.
    """
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body, headers=headers, method="POST")
    start = now()
    events = []
    first_text_at = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for event in iter_sse_events(response):
            events.append(event)
            if first_text_at is None:
                content, reasoning = event_deltas(event)
                if carries_text(content) or carries_text(reasoning):
                    first_text_at = now()
    end = now()
    result = summarize_events(events, start, first_text_at, end)
    result["error"] = None
    return result


def run_pairs(servers, prompts, *, reps, warmup, max_tokens, temperature, timeout,
              default_model=None, now=time.monotonic, opener=stream_once):
    """Run the paired protocol. ``servers`` is a list of dicts with name/url/api_key;
    ``prompts`` a list of dicts with id and messages (see load_prompts).
    Alternates engine order per repetition."""
    if len(servers) != 2:
        raise ValueError("exactly two --server entries are required")
    records = []
    for rep in range(reps):
        order = servers if rep % 2 == 0 else list(reversed(servers))
        for position, server in enumerate(order):
            for prompt in prompts:
                payload = _request_payload(prompt["messages"], prompt.get("model") or default_model,
                                           max_tokens, temperature)
                for _ in range(warmup):
                    try:
                        opener(server["url"], server.get("api_key"), payload, timeout, now=now)
                    except Exception:
                        pass  # warmup is best-effort; measured runs record their own errors
                record = {"prompt_id": prompt["id"], "engine": server["name"],
                          "order": "first" if position == 0 else "second", "rep": rep,
                          "request": {k: v for k, v in payload.items() if k != "messages"}}
                try:
                    record.update(opener(server["url"], server.get("api_key"), payload, timeout, now=now))
                except Exception as exc:  # noqa: BLE001 - record and continue the matrix
                    record.update({"first_text_seconds": None, "total_seconds": None,
                                   "content": "", "reasoning_content": "", "finish_reason": None,
                                   "usage": None, "hipengine": None, "error": f"{type(exc).__name__}: {exc}"})
                records.append(record)
    return records


def load_prompts(path):
    """JSONL with {"id", "prompt"} (wrapped as a single user message) or {"id", "messages"}.
    An optional per-prompt "model" overrides --model."""
    prompts = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "id" not in item:
                raise ValueError(f"line {lineno}: missing id")
            if "messages" in item:
                messages = item["messages"]
            elif "prompt" in item:
                messages = [{"role": "user", "content": item["prompt"]}]
            else:
                raise ValueError(f"line {lineno}: need either 'prompt' or 'messages'")
            prompts.append({"id": item["id"], "messages": messages, "model": item.get("model")})
    return prompts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", action="append", required=True, metavar="NAME=URL",
                        help="repeat exactly twice; optional KEY=VALUE api key via NAME=URL#key=KEY")
    parser.add_argument("--prompts", required=True, help="JSONL with id+prompt (or id+messages)")
    parser.add_argument("--model", default=None, help="model id to send; overrides nothing per-prompt")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", default=None, help="write JSON here (default: stdout)")
    args = parser.parse_args(argv)

    servers = []
    for spec in args.server:
        name, sep, url = spec.partition("=")
        if not sep:
            parser.error(f"--server must be NAME=URL, got {spec!r}")
        api_key = None
        if "#" in url and url.split("#", 1)[1].startswith("key="):
            url, api_key = url.split("#", 1)[0], url.split("#", 1)[1][4:]
        servers.append({"name": name, "url": url, "api_key": api_key})

    prompts = load_prompts(args.prompts)
    if args.model is None and any(prompt.get("model") is None for prompt in prompts):
        parser.error("--model is required unless every prompt carries one")

    records = run_pairs(servers, prompts, reps=args.reps, warmup=args.warmup,
                        max_tokens=args.max_tokens, temperature=args.temperature,
                        timeout=args.timeout, default_model=args.model)
    report = {"records": records, "summary": summarize_records(records)}
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
