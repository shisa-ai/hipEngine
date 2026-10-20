#!/usr/bin/env python3
"""Measure Flash-Next HTTP token latency while another request prefills.

Run against an idle server. Raw token IDs and arrival times stay in the output
artifact. Prompt rate is effective HTTP prompt time-to-first-token throughput,
not a device-only prefill measurement. Results are diagnostic until their
numerical, lifecycle and source-provenance gates are attached.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any

import httpx


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "intervals": len(values),
        "median_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "maximum_ms": max(values) if values else None,
    }


def token_intervals(row: dict, window: tuple[float, float] | None = None) -> list[float]:
    """Include gaps intersecting the prefill window, including a spanning gap."""
    times = row["token_times"]
    return [
        (end - start) * 1000
        for start, end in zip(times, times[1:])
        if window is None or (end >= window[0] and start <= window[1])
    ]


def summarize_wave(peers: list[dict], arrival: dict, wall_s: float) -> dict:
    window = (arrival["started_at"], arrival["token_times"][0])
    gaps = [gap for row in peers for gap in token_intervals(row)]
    interference = [gap for row in peers for gap in token_intervals(row, window)]
    peer_times = [stamp for row in peers for stamp in row["token_times"]]
    peer_interval = max(peer_times) - min(peer_times) if peer_times else 0
    peer_tokens = sum(len(row["generated_ids"]) - row.get("event_token_counts", [1])[0] for row in peers)
    ttft = window[1] - window[0]
    total_tokens = sum(len(row["generated_ids"]) for row in [*peers, arrival])
    return {
        "latency_scope": "client_visible_token_events; UTF-8-coalesced tokens have no invented timestamps",
        "coalesced_peer_events": sum(sum(count > 1 for count in row.get("event_token_counts", [])) for row in peers),
        "peer_inter_token_latency": latency_summary(gaps),
        "peer_interference_inter_token_latency": latency_summary(interference),
        "arrival_ttft_ms": ttft * 1000,
        "arrival_effective_prompt_tok_s": arrival["prompt_tokens"] / ttft,
        "peer_decode_interval_tok_s": peer_tokens / peer_interval if peer_interval else None,
        "group_wall_s": wall_s,
        "group_generated_tokens": total_tokens,
        "group_end_to_end_tok_s": total_tokens / wall_s,
    }


async def stream_request(client, model, prompt, max_tokens, *, token_ready=None) -> dict:
    started = time.perf_counter()
    row: dict[str, Any] = {
        "started_at": started, "prompt_tokens": len(prompt),
        "prompt_sha256": hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest(),
        "token_times": [], "event_token_counts": [], "server_elapsed_ms": [], "generated_ids": [],
    }
    payload = {
        "model": model, "prompt": prompt, "temperature": 0,
        "speculative_mtp": False, "ignore_eos": True, "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True, "include_hipengine": True},
    }
    async with client.stream("POST", "/v1/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(f"stream failed: {event}")
            for choice in event.get("choices", []):
                metadata = choice.get("hipengine") or {}
                if choice.get("finish_reason") is None:
                    count = (metadata.get("tokens") or {}).get("streamed_tokens")
                    if count is None:
                        raise RuntimeError("stream delta lacks exact generated-token count")
                    previous = sum(row["event_token_counts"])
                    if count < previous:
                        raise RuntimeError(f"AR stream token count moved backwards: {previous} -> {count}")
                    if count > previous:
                        # The text decoder may merge tokens until a complete UTF-8
                        # span exists. Keep event latency honest; never manufacture
                        # intermediate timestamps for the merged tokens.
                        row["token_times"].append(time.perf_counter())
                        row["event_token_counts"].append(count - previous)
                        row["server_elapsed_ms"].append(event["hipengine"]["timing"]["elapsed_ms"])
                        if token_ready is not None and count >= 8:
                            token_ready.set()
                else:
                    row["generated_ids"] = metadata.get("generated_token_ids") or []
                    row["finish_details"] = choice.get("finish_details")
                    row["decode_state"] = metadata.get("decode_state")
            if "usage" in event:
                row["usage"] = event["usage"]
    row["wall_s"] = time.perf_counter() - started
    if len(row["generated_ids"]) != max_tokens or sum(row["event_token_counts"]) != max_tokens:
        raise RuntimeError(f"expected {max_tokens} exact live tokens, got {row}")
    if row.get("usage", {}).get("prompt_tokens") != len(prompt):
        raise RuntimeError("server prompt-token count does not match the raw-ID request")
    return row


async def run_wave(client, args, peer_prompts, arrival_prompt) -> dict:
    started = time.perf_counter()
    ready = [asyncio.Event() for _ in peer_prompts]
    tasks = [asyncio.create_task(stream_request(
        client, args.model, prompt, args.decode_tokens, token_ready=event,
    )) for prompt, event in zip(peer_prompts, ready)]
    try:
        if ready:
            # A hard deadline detects failed peers without polling the server.
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ready)), 120)
        arrival = await stream_request(client, args.model, arrival_prompt, args.arrival_tokens)
        peers = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    wall_s = time.perf_counter() - started
    return {"peers": peers, "arrival": arrival, "summary": summarize_wave(peers, arrival, wall_s)}


async def run(args) -> None:
    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    report: dict[str, Any] = {
        "schema": 1, "kind": "qwen38fn_prefill_interference", "label": args.label,
        "performance_claim": False, "status": "running", "protocol": vars(args).copy(),
        "suite_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(), "waves": [],
        "server_source_diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "-C", str(args.server_repo), "diff", "HEAD"])).hexdigest(),
        "client_command": sys.argv,
        "physical_host": {
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "machine_id": Path("/etc/machine-id").read_text().strip(),
        },
        "server_commit": subprocess.check_output(
            ["git", "-C", str(args.server_repo), "rev-parse", "HEAD"], text=True).strip(),
        "server_git_status": subprocess.check_output(
            ["git", "-C", str(args.server_repo), "status", "--porcelain"], text=True),
    }
    report["protocol"] = {key: str(value) if isinstance(value, Path) else value
                          for key, value in report["protocol"].items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    async with httpx.AsyncClient(base_url=args.url, timeout=600) as client:
        for path in ("/ready", "/v1/hipengine/capabilities", "/v1/hipengine/decode"):
            response = await client.get(path)
            response.raise_for_status()
            report[path] = response.json()
        queue = report["/ready"]["queue"]
        if queue["depth"] or queue["active_requests"]:
            raise RuntimeError("server must be idle before this experiment")

        async def tokenize(text):
            response = await client.post("/v1/hipengine/tokenize", json={"text": text})
            response.raise_for_status()
            body = response.json()
            return body.get("token_ids", body.get("tokens"))

        peer_prompts = [await tokenize(json.dumps(row["messages"], ensure_ascii=False)) for row in prompts]
        filler = await tokenize("Archive entry: The library opens at nine and closes at five.\n" * 1024)
        for warmup in range(args.warmups):
            await run_wave(client, args, [], filler[:37])
        for repeat in range(args.repetitions):
            for length in args.prompt_lengths:
                for width in args.peers:
                    prompt_index = repeat + args.repeat_offset
                    category = prompts[prompt_index % len(prompts)]
                    head = peer_prompts[prompt_index % len(prompts)]
                    arrival = (head + filler * (1 + length // len(filler)))[:length]
                    peers = [peer_prompts[(prompt_index + index) % len(prompts)] for index in range(width)]
                    wave = await run_wave(client, args, peers, arrival)
                    wave.update(repeat=prompt_index, prompt_tokens=length, peer_count=width,
                                category=category["category"], prompt_id=category["id"])
                    wave["decode_telemetry_after"] = (
                        await client.get("/v1/hipengine/decode")
                    ).json()
                    report["waves"].append(wave)
                    save()
                    print(json.dumps({key: wave[key] for key in (
                        "repeat", "prompt_tokens", "peer_count", "summary")}), flush=True)
        report["decode_after"] = (await client.get("/v1/hipengine/decode")).json()
        report["ready_after"] = (await client.get("/ready")).json()
        report["source_unchanged"] = report["server_source_diff_sha256"] == hashlib.sha256(
            subprocess.check_output(["git", "-C", str(args.server_repo), "diff", "HEAD"])
        ).hexdigest()
    report["status"] = "complete" if report["source_unchanged"] else "invalid_source_changed"
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18388")
    parser.add_argument("--model", default="qwen38fn-followup")
    parser.add_argument("--label", required=True)
    parser.add_argument("--server-repo", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"))
    parser.add_argument("--prompt-lengths", type=lambda text: [int(item) for item in text.split(",")], default=[37, 4093, 12287])
    parser.add_argument("--peers", type=lambda text: [int(item) for item in text.split(",")], default=[0, 1, 3])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--repeat-offset", type=int, default=0,
                        help="Starting suite index when repetitions use separate server processes")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--decode-tokens", type=int, default=192)
    parser.add_argument("--arrival-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value < 1 for value in args.prompt_lengths) or any(value < 0 for value in args.peers):
        parser.error("prompt lengths must be positive and peer counts nonnegative")
    if args.repetitions < 1 or args.repeat_offset < 0 or args.warmups < 0 or args.decode_tokens < 9 or args.arrival_tokens < 1:
        parser.error("require positive repetitions/arrival tokens, nonnegative repeat offset/warmups, and at least nine peer tokens")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
