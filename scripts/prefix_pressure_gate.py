#!/usr/bin/env python3
"""Drive the served prefix cache to eviction pressure with MTP on and off.

The prefix checklist item asks for eviction under pressure with live requests,
in both MTP configurations, asserting parity with the same workload without
prefix reuse, correct hit/miss reporting in ``/ready`` and request diagnostics,
and no corrupted provider position.

Shape: more conversations than the retained budget, each about 900 prompt
tokens so the captured boundary sits strictly inside the prompt and is
reusable.  Every conversation runs cold, then the newest ones are re-requested
(they must reuse their boundary) and the oldest ones are re-requested (their
boundary was evicted, so they must miss cleanly).  The default retained budget
is ``max(1, capacity)`` entries, so six conversations against a four-request
server evict the two oldest.

Every re-request must reproduce its cold run's generated text token for token:
prefix reuse is only correct if the reused boundary produces the same
continuation, which is what rules out a corrupted provider position.  The
``/ready`` deltas must account for exactly the hits and misses the requests
reported, so a served surface that silently stops reporting reuse fails here.

Usage::

    python3 scripts/prefix_pressure_gate.py --json /tmp/prefix-pressure.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

CONVERSATIONS = 6
TARGET_PROMPT_TOKENS = 900
REPLY_TOKENS = 16
BLOCK_TOKENS = 256
SENTENCE = (
    "The prefix cache keeps block-aligned prompt state so a later request can skip "
    "prefilling tokens it has already processed. "
)
SENTENCE_TOKENS = 24


def conversation_prompt(tag: str, index: int, *, nonce: str) -> str:
    """A distinct long prompt per conversation, so each occupies its own branch."""

    repeats = max(1, round(TARGET_PROMPT_TOKENS / SENTENCE_TOKENS))
    return (
        f"[pressure {tag} conversation {index} {nonce}] "
        + SENTENCE * repeats
        + "\nReply with one short sentence."
    )


def captured_boundary(prompt_tokens: int) -> int:
    """The deepest 256-aligned boundary at or before the prompt end.

    A prompt that ends on a block boundary captures the whole prompt, which a
    same-length resend cannot reuse, so those conversations are excluded from
    the reuse expectation by ``reuse_expectation``.
    """

    return (int(prompt_tokens) // BLOCK_TOKENS) * BLOCK_TOKENS


def reuse_expectation(prompt_tokens: Sequence[int]) -> dict[int, int | None]:
    """Expected cached-token count per conversation when the boundary survives."""

    expectation: dict[int, int | None] = {}
    for index, tokens in enumerate(prompt_tokens):
        boundary = captured_boundary(tokens)
        expectation[index] = boundary if 0 < boundary < int(tokens) else None
    return expectation


def _delta(after: Mapping[str, Any], before: Mapping[str, Any], key: str) -> int | None:
    left = after.get(key)
    right = before.get(key)
    if not isinstance(left, int) or not isinstance(right, int):
        return None
    return left - right


def evaluate(record: Mapping[str, Any]) -> list[str]:
    """Every acceptance failure the record shows, as readable sentences."""

    failures: list[str] = []
    for route, block in record.get("routes", {}).items():
        conversations = block.get("conversations") or []
        if not conversations:
            failures.append(f"{route}: no conversations recorded")
            continue
        hits = 0
        misses = 0
        for row in conversations:
            index = row.get("index")
            warm = row.get("warm") or {}
            cold = row.get("cold") or {}
            cached = warm.get("cached_tokens")
            expected = row.get("expected_reuse")
            if row.get("kind") == "warm":
                hits += 1
                if cached != expected:
                    failures.append(
                        f"{route}: conversation {index} reused {cached} tokens, "
                        f"expected {expected}"
                    )
            else:
                misses += 1
                if cached not in (0, None):
                    failures.append(
                        f"{route}: evicted conversation {index} reported {cached} "
                        "cached tokens, expected a clean miss"
                    )
            if not warm.get("text"):
                failures.append(f"{route}: conversation {index} produced no text")
            elif cold.get("text") != warm.get("text"):
                failures.append(
                    f"{route}: conversation {index} output changed with prefix reuse"
                )
        before = block.get("ready_after_cold") or {}
        after = block.get("ready_after_warm") or {}
        reported_hits = _delta(after, before, "usable_hits")
        reported_misses = _delta(after, before, "misses")
        if reported_hits != hits:
            failures.append(
                f"{route}: /ready reported {reported_hits} hits for {hits} reusing "
                "requests"
            )
        if reported_misses != misses:
            failures.append(
                f"{route}: /ready reported {reported_misses} misses for {misses} "
                "re-prefilling requests"
            )
        if (before.get("stats") or {}).get("live_requests") or (
            after.get("stats") or {}
        ).get("live_requests"):
            failures.append(f"{route}: the counter sample was taken with requests live")
    return failures


def _fetch_json(url: str, *, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def prefix_counters(server: str) -> dict:
    block = _fetch_json(f"{server}/ready", timeout=60).get("prefix_cache") or {}
    stats = block.get("stats") or {}
    return {
        "mode": block.get("mode"),
        "stats": {"live_requests": stats.get("live_requests")},
        "usable_hits": block.get("usable_hits"),
        "misses": stats.get("misses"),
        "unusable_hits": block.get("unusable_hits"),
        "fallback_reasons": block.get("fallback_reasons"),
        "reused_tokens": block.get("reused_tokens"),
        "retained_snapshot_entries": block.get("retained_snapshot_entries"),
        "retained_snapshot_limit": block.get("retained_snapshot_limit"),
        "retained_snapshot_evictions_by_reason": block.get(
            "retained_snapshot_evictions_by_reason"
        ),
        "snapshot_captures": block.get("snapshot_captures"),
    }


def request(
    server: str,
    model: str,
    content: str,
    *,
    speculative_mtp: bool,
) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": REPLY_TOKENS,
            "temperature": 0.0,
            "speculative_mtp": speculative_mtp,
            "stream": True,
            "stream_options": {"include_usage": True, "include_hipengine": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"{server}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    text: list[str] = []
    usage: dict = {}
    timing: dict = {}
    started = time.perf_counter()
    first = None
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            payload = json.loads(data)
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
            block = payload.get("hipengine")
            if isinstance(block, dict):
                if isinstance(block.get("timing"), dict):
                    timing = block["timing"]
                if isinstance(block.get("usage"), dict):
                    usage = block["usage"]
            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}
                for key in ("content", "reasoning_content"):
                    value = delta.get(key)
                    if isinstance(value, str) and value:
                        if first is None:
                            first = time.perf_counter()
                        text.append(value)
    finished = time.perf_counter()
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "text": "".join(text),
        "ttft_s": None if first is None else first - started,
        "wall_s": finished - started,
        "decode_tps": timing.get("decode_tokens_per_second"),
        "elapsed_ms": timing.get("elapsed_ms"),
    }


def run(args: argparse.Namespace) -> dict:
    server = args.server.rstrip("/")
    model = _fetch_json(f"{server}/v1/models", timeout=30)["data"][0]["id"]
    nonce = f"{os.getpid():x}-{time.time_ns():x}"
    if args.warm_count + args.evicted_count > CONVERSATIONS:
        raise SystemExit(
            f"--warm-count {args.warm_count} + --evicted-count {args.evicted_count} "
            f"exceeds the {CONVERSATIONS} conversations this gate drives"
        )
    warm_indexes = tuple(
        range(CONVERSATIONS - 1, CONVERSATIONS - 1 - args.warm_count, -1)
    )
    evicted_indexes = tuple(range(args.evicted_count))

    record: dict = {
        "model": model,
        "server": server,
        "conversations": CONVERSATIONS,
        "reply_tokens": REPLY_TOKENS,
        "routes": {},
    }
    for route, speculative in (("on", True), ("off", False)):
        tag = f"mtp {route}"
        prompts = {
            index: conversation_prompt(tag, index, nonce=nonce)
            for index in range(CONVERSATIONS)
        }
        before = prefix_counters(server)
        cold = {
            index: request(server, model, prompts[index], speculative_mtp=speculative)
            for index in range(CONVERSATIONS)
        }
        after_cold = prefix_counters(server)
        warm = {
            index: request(server, model, prompts[index], speculative_mtp=speculative)
            for index in warm_indexes
        }
        evicted = {
            index: request(server, model, prompts[index], speculative_mtp=speculative)
            for index in evicted_indexes
        }
        after_warm = prefix_counters(server)

        expectation = reuse_expectation(
            [int(cold[index]["prompt_tokens"]) for index in range(CONVERSATIONS)]
        )
        rows = []
        for index in range(CONVERSATIONS):
            if index in warm:
                result, kind = warm[index], "warm"
            else:
                result, kind = evicted[index], "evicted"
            rows.append(
                {
                    "index": index,
                    "kind": kind,
                    "prompt_tokens": cold[index]["prompt_tokens"],
                    "expected_reuse": expectation.get(index),
                    "cold": {
                        "cached_tokens": cold[index]["cached_tokens"],
                        "text": cold[index]["text"],
                        "ttft_s": cold[index]["ttft_s"],
                        "wall_s": cold[index]["wall_s"],
                    },
                    "warm": {
                        "cached_tokens": result["cached_tokens"],
                        "text": result["text"],
                        "ttft_s": result["ttft_s"],
                        "wall_s": result["wall_s"],
                        "decode_tps": result["decode_tps"],
                    },
                }
            )
        record["routes"][route] = {
            "speculative_mtp": speculative,
            "ready_before": before,
            "ready_after_cold": after_cold,
            "ready_after_warm": after_warm,
            "conversations": rows,
        }
        for row in rows:
            print(
                f"mtp={route:3s} conv={row['index']} {row['kind']:8s} "
                f"prompt={row['prompt_tokens']:5d} expected={row['expected_reuse']} "
                f"warm_cached={row['warm']['cached_tokens']} "
                f"cold_ttft={row['cold']['ttft_s']:.2f} "
                f"warm_ttft={row['warm']['ttft_s']:.2f} "
                f"parity={row['cold']['text'] == row['warm']['text']}",
                flush=True,
            )
        print(f"mtp={route:3s} /ready before={json.dumps(before)}", flush=True)
        print(f"mtp={route:3s} /ready after_warm={json.dumps(after_warm)}", flush=True)

    record["failures"] = evaluate(record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8000",
        help="base URL of a running hipengine serve endpoint",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--warm-count",
        type=int,
        default=4,
        help="newest conversations re-requested for reuse (default 4)",
    )
    parser.add_argument(
        "--evicted-count",
        type=int,
        default=2,
        help="oldest conversations re-requested for a clean miss (default 2)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record = run(args)
    text = json.dumps(record, indent=1, default=str)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    failures = record["failures"]
    if failures:
        print("FAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("prefix pressure gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
