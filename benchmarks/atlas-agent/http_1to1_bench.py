#!/usr/bin/env python3
"""Engine-agnostic HTTP benchmark for head-to-head comparison.

Measures any OpenAI-compatible chat-completions endpoint from the SSE stream
alone, with no engine-specific diagnostics fields, so the same invocation runs
unchanged against two different engines and the measurement is symmetric.

The comparison is made robust to tokenizer and chunking differences by fixing
the completion length: every request sets ``max_tokens`` and the prompts are
chosen so no early stop occurs, so both engines emit exactly N tokens and the
decode rate is ``(N - 1) / (end_to_end - time_to_first_token)``. The harness
records the delta count and any ``usage`` block it receives, and reports whether
each request actually reached N, so a run that stopped early is visible rather
than silently averaged.

Arms:

``single``    one request at a time; TTFT, ITL, per-request decode rate.
``multi``     multi-turn: each turn re-sends the whole transcript, so turns 2+
              share a long prefix and exercise prefix caching on either engine.
``conc``      a concurrency ladder; adds aggregate tokens/second.

Every request is greedy (``temperature=0``) and streamed.

Usage::

    python3 scripts/http_1to1_bench.py \
        --base-url http://127.0.0.1:8097 --model qwen3.8-27b-q4km \
        --prompt-file benchmarks/prompts/mtpbench-code-general-ja.jsonl \
        --arm single --output-len 128 --repeats 4 --json /tmp/out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Result:
    """One request's timing, measured from the SSE stream only."""

    prompt_tokens_estimate: int = 0
    ttft_s: float = 0.0
    e2e_s: float = 0.0
    deltas: int = 0
    content: str = ""
    usage_completion_tokens: int | None = None
    reached_target: bool = False
    error: str | None = None

    @property
    def did_not_stop_early(self) -> bool:
        """Whether the engine produced the full requested length.

        With a usage block this is exact. Without one it is only provable in the
        direction where the observed delta count already meets the target, so a
        chunk-batching engine reads as unknown rather than as early-stopping.
        """
        if self.usage_completion_tokens is not None:
            return self.usage_completion_tokens >= 1 and self.reached_target
        return self.reached_target

    @property
    def content_chars(self) -> int:
        return len(self.content)

    @property
    def decode_s(self) -> float:
        return max(self.e2e_s - self.ttft_s, 0.0)

    def decode_tokens(self, target: int) -> int:
        """Tokens attributed to the decode phase.

        The engine's own ``usage.completion_tokens`` wins when it is reported.
        Otherwise the count is ``target`` only when the request is known not to
        have stopped early, because a fixed ``max_tokens`` with no early stop
        means the engine produced exactly that many tokens -- whereas counting
        SSE deltas would undercount any engine that batches several tokens into
        one chunk and would overstate its rate.
        """
        if self.usage_completion_tokens:
            return int(self.usage_completion_tokens)
        return target if self.did_not_stop_early else self.deltas


def stream_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_s: float,
) -> Result:
    """Issue one streamed greedy chat completion and time it from the stream."""
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    result = Result()
    started = time.perf_counter()
    first: float | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens"):
                    result.usage_completion_tokens = int(usage["completion_tokens"])
                for choice in chunk.get("choices") or ():
                    piece = (choice.get("delta") or {}).get("content")
                    if not piece:
                        continue
                    if first is None:
                        first = time.perf_counter()
                    result.deltas += 1
                    result.content += piece
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        result.error = f"HTTP {exc.code}: {detail}"
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        result.error = f"{type(exc).__name__}: {exc}"
    result.e2e_s = time.perf_counter() - started
    result.ttft_s = (first - started) if first is not None else result.e2e_s
    if result.usage_completion_tokens is not None:
        result.reached_target = result.usage_completion_tokens >= max_tokens
    else:
        result.reached_target = result.deltas >= max_tokens
    # A response with no content is a failure, not a fast generation. Without
    # this guard an engine that errors immediately is measured at a near-zero
    # decode time and reports an enormous, entirely fictional rate -- which is
    # exactly what a broken atlas MTP arm produced (~260 tok/s from requests that
    # returned nothing). Empty output must be visible as an error.
    if result.error is None and not result.content:
        result.error = "empty response: no content deltas"
    return result


def load_prompts(path: Path, category: str | None, limit: int) -> list[list[dict[str, str]]]:
    """Load prompt rows as ready-to-send chat message lists.

    ``benchmarks/prompts/*.jsonl`` rows carry a ``messages`` array, so the
    request sends the file's own roles verbatim rather than flattening them into
    one user turn -- a flattened prompt would not be the same request on both
    engines and would change the prefill being compared.
    """
    prompts: list[list[dict[str, str]]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if category and str(row.get("category", "")) != category:
                continue
            messages = row.get("messages")
            if isinstance(messages, list) and messages:
                prompts.append([{"role": str(m.get("role", "user")),
                                "content": str(m.get("content", ""))}
                               for m in messages])
            else:
                text = row.get("prompt") or row.get("text") or ""
                if text:
                    prompts.append([{"role": "user", "content": str(text)}])
            if len(prompts) >= limit:
                break
    if not prompts:
        raise SystemExit(f"no prompts selected from {path} (category={category!r})")
    return prompts


def summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": p95,
        "min": ordered[0],
        "max": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-category", default=None)
    parser.add_argument("--arm", choices=("single", "multi", "conc"), default="single")
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="conc arm: highest level; ladder is 1..N")
    parser.add_argument("--turns", type=int, default=3, help="multi arm: turns")
    parser.add_argument("--prompt-limit", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    prompts = load_prompts(args.prompt_file, args.prompt_category, args.prompt_limit)
    report: dict[str, object] = {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "arm": args.arm,
        "output_len": args.output_len,
        "prompt_file": str(args.prompt_file),
        "prompt_category": args.prompt_category,
        "prompts_used": len(prompts),
        "arms": {},
    }
    print(f"# {args.label or args.base_url} model={args.model} arm={args.arm} "
          f"prompts={len(prompts)} output_len={args.output_len}", flush=True)

    if args.arm == "single":
        results: list[Result] = []
        for repeat in range(args.repeats):
            for index, prompt in enumerate(prompts):
                res = stream_chat(args.base_url, args.model, prompt,
                                  args.output_len, args.timeout)
                results.append(res)
                status = res.error or (f"ttft={res.ttft_s*1000:.0f}ms "
                                       f"dec={res.decode_s:.2f}s deltas={res.deltas}")
                print(f"  repeat{repeat} p{index}: {status}", flush=True)
        ok = [r for r in results if not r.error]
        report["arms"]["single"] = {
            "requests": len(results),
            "errors": len(results) - len(ok),
            "reached_target": sum(1 for r in ok if r.reached_target),
            "ttft_ms": summarise([r.ttft_s * 1000 for r in ok]),
            "e2e_s": summarise([r.e2e_s for r in ok]),
            "decode_tok_s": summarise(
                [r.decode_tokens(args.output_len) / r.decode_s for r in ok if r.decode_s > 0]
            ),
            "itl_ms": summarise(
                [r.decode_s * 1000 / max(r.decode_tokens(args.output_len) - 1, 1)
                 for r in ok if r.decode_s > 0]
            ),
            "errors_detail": [r.error for r in results if r.error][:5],
        }

    elif args.arm == "multi":
        turn_rows: list[dict[str, object]] = []
        for index, prompt in enumerate(prompts):
            messages: list[dict[str, str]] = [dict(m) for m in prompt]
            for turn in range(args.turns):
                res = stream_chat(args.base_url, args.model, messages,
                                  args.output_len, args.timeout)
                turn_rows.append({
                    "prompt": index, "turn": turn, "error": res.error,
                    "ttft_ms": res.ttft_s * 1000, "e2e_s": res.e2e_s,
                    "deltas": res.deltas, "reached_target": res.reached_target,
                    "decode_tok_s": (res.decode_tokens(args.output_len) / res.decode_s
                                     if res.decode_s > 0 else 0.0),
                    "transcript_chars": sum(len(m["content"]) for m in messages),
                })
                transcript_chars = sum(len(m["content"]) for m in messages)
                print(f"  p{index} turn{turn}: "
                      f"{res.error or f'ttft={res.ttft_s*1000:.0f}ms deltas={res.deltas} '
                                       f'transcript={transcript_chars}ch'}",
                      flush=True)
                if res.error:
                    break
                # Greedy, so the reply is deterministic and turn N+1's prefix is
                # exactly the transcript turn N ended with -- which is what makes
                # a prefix cache hit reproducible across turns. The reply is the
                # text actually streamed, never a placeholder: a fabricated
                # transcript would not share the engine's real prefix at all.
                messages.append({"role": "assistant", "content": res.content})
                messages.append({"role": "user", "content": prompt[-1]["content"]})
        ok = [r for r in turn_rows if not r["error"]]
        first_turn = [r for r in ok if r["turn"] == 0]
        later = [r for r in ok if r["turn"] > 0]
        report["arms"]["multi"] = {
            "requests": len(turn_rows),
            "errors": len(turn_rows) - len(ok),
            "turn0_ttft_ms": summarise([r["ttft_ms"] for r in first_turn]),
            "later_turn_ttft_ms": summarise([r["ttft_ms"] for r in later]),
            "turn0_decode_tok_s": summarise([r["decode_tok_s"] for r in first_turn]),
            "later_turn_decode_tok_s": summarise([r["decode_tok_s"] for r in later]),
            "rows": turn_rows,
            "errors_detail": [r["error"] for r in turn_rows if r["error"]][:5],
        }

    else:  # conc
        ladder: dict[str, object] = {}
        for level in range(1, args.concurrency + 1):
            batch = [prompts[i % len(prompts)] for i in range(level)]
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=level) as pool:
                futures = [
                    pool.submit(stream_chat, args.base_url, args.model,
                                prompt, args.output_len, args.timeout)
                    for prompt in batch
                ]
                results = [f.result() for f in futures]
            wall = time.perf_counter() - started
            ok = [r for r in results if not r.error]
            produced = sum(r.decode_tokens(args.output_len) for r in ok)
            ladder[str(level)] = {
                "requests": len(results),
                "errors": len(results) - len(ok),
                "wall_s": wall,
                "aggregate_tok_s": produced / wall if wall > 0 else 0.0,
                "per_request_decode_tok_s": summarise(
                    [r.decode_tokens(args.output_len) / r.decode_s
                     for r in ok if r.decode_s > 0]
                ),
                "ttft_ms": summarise([r.ttft_s * 1000 for r in ok]),
                "errors_detail": [r.error for r in results if r.error][:3],
            }
            print(f"  c={level}: aggregate={ladder[str(level)]['aggregate_tok_s']:.2f} tok/s "
                  f"wall={wall:.2f}s errors={ladder[str(level)]['errors']}", flush=True)
        report["arms"]["conc"] = ladder

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1, default=str))
        print(f"# wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
