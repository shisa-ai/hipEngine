#!/usr/bin/env python3
"""Measure how a real ShareGPT load routes through the speculative MTP path.

The vLLM serving benchmark's ShareGPT loader is the load definition: prompt is
the first conversation turn, the expected output length is the second turn's
token count, and rows are filtered to 4..1024 prompt tokens, 4..2048 total. This
harness sends that same load to a running hipEngine server and records, for every
request, which execution mode produced its output:

* ``hipengine.speculative_mtp`` (used, MTP-committed versus autoregressive
  output tokens, the first fallback position, and the planner reason),
* ``hipengine.generation_shape.route_decision`` (the route the model plugin
  resolved, and why),
* a per-chunk decode timeline, so a request that leaves speculation mid-output
  shows the resulting latency step rather than only an aggregate rate.

Usage:
    python3 scripts/sharegpt_mtp_routing_pass.py \
        --server-url http://127.0.0.1:8030 --model qwen3.8-27b-q4km \
        --dataset /home/lhl/models/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --gguf /home/lhl/models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
        --num-prompts 24 --max-concurrency 1 --output-len 128 \
        --label c1-len128 --out /tmp/sharegpt-c1-len128.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "qwen3.8-27b-q4km"
DEFAULT_GGUF = Path("/home/lhl/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
# vLLM's ShareGPTDataset pruning criteria (benchmarks/datasets/datasets.py).
MIN_LEN = 4
MAX_PROMPT_LEN = 1024
MAX_TOTAL_LEN = 2048


def _tokenizer(gguf: Path):
    from hipengine.loading.gguf import GGUFReader
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    return Qwen35GGUFTokenizer.from_gguf_info(GGUFReader(gguf).info)


def load_samples(
    dataset: Path,
    *,
    tokenizer,
    count: int,
    seed: int,
    output_len: int | None,
    max_prompt_len: int,
    max_total_len: int,
) -> list[dict[str, Any]]:
    """Reproduce the vLLM ShareGPT sample set for one benchmark point."""

    with dataset.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    rows = [
        row
        for row in rows
        if isinstance(row.get("conversations"), list) and len(row["conversations"]) >= 2
    ]
    random.Random(seed).shuffle(rows)
    samples: list[dict[str, Any]] = []
    for row in rows:
        if len(samples) >= count:
            break
        prompt = str(row["conversations"][0].get("value") or "")
        completion = str(row["conversations"][1].get("value") or "")
        prompt_len = len(tokenizer.encode(prompt))
        expected = (
            len(tokenizer.encode(completion)) if output_len is None else output_len
        )
        if prompt_len < MIN_LEN or prompt_len > max_prompt_len:
            continue
        if expected < MIN_LEN or prompt_len + expected > max_total_len:
            continue
        samples.append(
            {
                "source_id": row.get("id"),
                "prompt": prompt,
                "prompt_tokens": prompt_len,
                "expected_output_tokens": expected,
            }
        )
    if len(samples) < count:
        raise SystemExit(
            f"dataset produced {len(samples)} valid rows, need {count}"
        )
    return samples


def _stream_request(
    url: str,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float | None,
    timeout: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": int(max_tokens),
        "stream": True,
        "stream_options": {"include_usage": True, "include_hipengine": True},
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    timeline: list[list[float]] = []
    usage: Mapping[str, Any] | None = None
    terminal: Mapping[str, Any] | None = None
    text: list[str] = []
    started = time.perf_counter()
    ttft_ms: float | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000.0
                    text.append(delta["content"])
                if choice.get("finish_reason") is not None:
                    terminal = choice
                state = ((choice.get("hipengine") or {}).get("decode_state")) or {}
                generated = state.get("generated_tokens")
                if generated is not None:
                    timeline.append(
                        [
                            (time.perf_counter() - started) * 1000.0,
                            int(generated),
                        ]
                    )
            if event.get("usage"):
                usage = event["usage"]
    e2e_ms = (time.perf_counter() - started) * 1000.0
    extension = (terminal or {}).get("hipengine") or {}
    return {
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "timeline": timeline,
        "usage": usage,
        "text_characters": len("".join(text)),
        "speculative_mtp": extension.get("speculative_mtp"),
        "generation_shape": extension.get("generation_shape"),
    }

def _request_row(sample: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    usage = result.get("usage") or {}
    if usage.get("completion_tokens") is None:
        # A stream that ends without usage is a truncated or aborted response;
        # record it as a failure instead of an empty success row.
        return {
            "source_id": sample.get("source_id"),
            "prompt_tokens_local": sample.get("prompt_tokens"),
            "expected_output_tokens": sample.get("expected_output_tokens"),
            "e2e_ms": result.get("e2e_ms"),
            "error": (
                "stream ended without usage"
                f" (events={len(result.get('timeline') or [])},"
                f" text_characters={result.get('text_characters')},"
                f" e2e_ms={result.get('e2e_ms')})"
            ),
        }
    mtp = result.get("speculative_mtp") or {}
    shape = result.get("generation_shape") or {}
    decision = shape.get("route_decision") or {}
    accounting = mtp.get("output_accounting") or {}
    return {
        "source_id": sample.get("source_id"),
        "prompt_tokens_local": sample.get("prompt_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "expected_output_tokens": sample.get("expected_output_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "ttft_ms": result.get("ttft_ms"),
        "e2e_ms": result.get("e2e_ms"),
        "text_characters": result.get("text_characters"),
        "route": shape.get("route"),
        "route_decision_reason": decision.get("reason"),
        "requested_route": decision.get("requested_route"),
        "mtp_used": mtp.get("used"),
        "mtp_output_tokens": mtp.get("mtp_output_tokens"),
        "ar_output_tokens": mtp.get("ar_output_tokens"),
        "mtp_coverage": accounting.get("mtp_coverage"),
        "first_fallback_position": mtp.get("first_fallback_position"),
        "fallback_reason": mtp.get("fallback_reason"),
        "fallback_reason_counts": mtp.get("fallback_reason_counts"),
        "selected_depth_histogram": mtp.get("selected_depth_histogram"),
        "draft_cycles": mtp.get("draft_cycles"),
        "accepted_draft_tokens": mtp.get("accepted_draft_tokens"),
        "reconciled": accounting.get("reconciled"),
        "timeline": result.get("timeline"),
    }


def _rate_windows(timeline: Sequence[Sequence[float]], *, window: int = 16) -> list[dict[str, float]]:
    """Decode rate per window of committed tokens, from the chunk timeline."""

    points = [(float(ms), int(tokens)) for ms, tokens in timeline]
    if len(points) < 2:
        return []
    windows: list[dict[str, float]] = []
    start_index = 0
    while start_index < len(points) - 1:
        start_ms, start_tokens = points[start_index]
        end_index = min(start_index + window, len(points) - 1)
        end_ms, end_tokens = points[end_index]
        tokens = end_tokens - start_tokens
        seconds = (end_ms - start_ms) / 1000.0
        if tokens > 0 and seconds > 0:
            windows.append(
                {
                    "start_token": float(start_tokens),
                    "tokens_per_second": tokens / seconds,
                }
            )
        start_index = end_index
    return windows


def _cliff(rows: Sequence[Mapping[str, Any]], *, window: int = 16) -> dict[str, Any]:
    """Quantify the latency step a request takes when it leaves speculation."""

    measured: list[dict[str, Any]] = []
    for row in rows:
        position = row.get("first_fallback_position")
        timeline = row.get("timeline") or []
        if not position or not timeline:
            continue
        windows = _rate_windows(timeline, window=window)
        before = [
            entry["tokens_per_second"]
            for entry in windows
            if entry["start_token"] < position - window
        ]
        after = [
            entry["tokens_per_second"]
            for entry in windows
            if entry["start_token"] >= position
        ]
        if not before or not after:
            continue
        measured.append(
            {
                "source_id": row.get("source_id"),
                "first_fallback_position": position,
                "completion_tokens": row.get("completion_tokens"),
                "before_tokens_per_second": statistics.median(before),
                "after_tokens_per_second": statistics.median(after),
                "ratio": statistics.median(after) / statistics.median(before),
                "rate_profile": windows,
            }
        )
    if not measured:
        return {"requests": 0}
    ratios = [entry["ratio"] for entry in measured]
    return {
        "requests": len(measured),
        "median_before_tokens_per_second": statistics.median(
            entry["before_tokens_per_second"] for entry in measured
        ),
        "median_after_tokens_per_second": statistics.median(
            entry["after_tokens_per_second"] for entry in measured
        ),
        "median_ratio": statistics.median(ratios),
        "worst_ratio": min(ratios),
        "requests_slower_than_80pct_after_crossing": sum(
            1 for ratio in ratios if ratio < 0.8
        ),
        "detail": measured,
    }


def summarize(rows: Sequence[Mapping[str, Any]], *, window: int = 16) -> dict[str, Any]:
    completion = sum(int(row.get("completion_tokens") or 0) for row in rows)
    mtp_outputs = sum(int(row.get("mtp_output_tokens") or 0) for row in rows)
    used = [row for row in rows if row.get("mtp_used")]
    reasons: Counter[str] = Counter()
    for row in rows:
        if row.get("mtp_used"):
            continue
        reason = row.get("fallback_reason") or row.get("route_decision_reason")
        reasons[str(reason or "unknown")] += 1
    prompt_buckets: dict[str, Counter[str]] = {}
    for row in rows:
        prompt_tokens = int(row.get("prompt_tokens") or row.get("prompt_tokens_local") or 0)
        bucket = (
            "0-511"
            if prompt_tokens < 512
            else "512-767"
            if prompt_tokens < 768
            else "768-1022"
            if prompt_tokens < 1023
            else "1023+"
        )
        prompt_buckets.setdefault(bucket, Counter())
        prompt_buckets[bucket]["requests"] += 1
        if row.get("mtp_used"):
            prompt_buckets[bucket]["mtp_requests"] += 1
        prompt_buckets[bucket]["mtp_output_tokens"] += int(
            row.get("mtp_output_tokens") or 0
        )
        prompt_buckets[bucket]["completion_tokens"] += int(
            row.get("completion_tokens") or 0
        )
    ttfts = [float(row["ttft_ms"]) for row in rows if row.get("ttft_ms")]
    return {
        "requests": len(rows),
        "mtp_requests": len(used),
        "mtp_request_share": (len(used) / len(rows)) if rows else 0.0,
        "completion_tokens": completion,
        "mtp_output_tokens": mtp_outputs,
        "mtp_output_share": (mtp_outputs / completion) if completion else 0.0,
        "non_mtp_reasons": dict(sorted(reasons.items())),
        "prompt_token_buckets": {
            bucket: dict(sorted(counts.items()))
            for bucket, counts in sorted(prompt_buckets.items())
        },
        "median_ttft_ms": statistics.median(ttfts) if ttfts else None,
        "cliff": _cliff(rows, window=window),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--gguf", default=DEFAULT_GGUF, type=Path)
    parser.add_argument("--num-prompts", type=int, default=24)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help=(
            "Override the dataset's expected output length (tokens). "
            "0 or a negative value uses the dataset's own answer length."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Omit to use the server default (hipEngine defaults to greedy).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompt-len", type=int, default=MAX_PROMPT_LEN)
    parser.add_argument("--max-total-len", type=int, default=MAX_TOTAL_LEN)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--label", default="sharegpt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Requests to run (and discard) before the measured load.",
    )
    args = parser.parse_args()

    tokenizer = _tokenizer(args.gguf)
    output_len = args.output_len if (args.output_len or 0) > 0 else None
    samples = load_samples(
        args.dataset,
        tokenizer=tokenizer,
        count=args.num_prompts + args.warmup,
        seed=args.seed,
        output_len=output_len,
        max_prompt_len=args.max_prompt_len,
        max_total_len=args.max_total_len,
    )
    warmup, measured = samples[: args.warmup], samples[args.warmup :]
    print(
        f"[{args.label}] {len(measured)} prompts, concurrency {args.max_concurrency}, "
        f"output_len={output_len or 'dataset'}, "
        f"temperature={args.temperature if args.temperature is not None else 'server-default'}"
    )

    def run(sample: Mapping[str, Any]) -> dict[str, Any]:
        try:
            result = _stream_request(
                args.server_url,
                model=args.model,
                prompt=str(sample["prompt"]),
                max_tokens=int(sample["expected_output_tokens"]),
                temperature=args.temperature,
                timeout=args.request_timeout,
            )
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")[:400]
            return {
                "source_id": sample.get("source_id"),
                "prompt_tokens_local": sample.get("prompt_tokens"),
                "expected_output_tokens": sample.get("expected_output_tokens"),
                "error": f"HTTP {error.code}: {body}",
            }
        return _request_row(sample, result)

    for sample in warmup:
        run(sample)

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_concurrency
    ) as pool:
        rows = list(pool.map(run, measured))
    wall = time.perf_counter() - started

    failures = [row for row in rows if row.get("error")]
    good = [row for row in rows if not row.get("error")]
    report = {
        "label": args.label,
        "server_url": args.server_url,
        "model": args.model,
        "dataset": str(args.dataset),
        "num_prompts": len(measured),
        "max_concurrency": args.max_concurrency,
        "output_len": output_len,
        "temperature": args.temperature,
        "seed": args.seed,
        "wall_seconds": wall,
        "failed_requests": len(failures),
        "summary": summarize(good, window=args.window),
        "rows": rows,
    }
    report["summary"]["decode_tokens_per_second"] = (
        sum(int(row.get("completion_tokens") or 0) for row in good) / wall
        if wall
        else None
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = report["summary"]
    print(
        f"[{args.label}] requests={summary['requests']} failures={len(failures)} "
        f"mtp_requests={summary['mtp_requests']} "
        f"({summary['mtp_request_share']:.1%}) "
        f"mtp_output_share={summary['mtp_output_share']:.1%} "
        f"decode={summary['decode_tokens_per_second']:.2f} tok/s "
        f"wall={wall:.1f}s"
    )
    print(f"[{args.label}] non-MTP reasons: {json.dumps(summary['non_mtp_reasons'])}")
    cliff = summary["cliff"]
    if cliff.get("requests"):
        print(
            f"[{args.label}] cliff over {cliff['requests']} requests: "
            f"before={cliff['median_before_tokens_per_second']:.2f} "
            f"after={cliff['median_after_tokens_per_second']:.2f} tok/s "
            f"median_ratio={cliff['median_ratio']:.2f} "
            f"slower_than_80pct={cliff['requests_slower_than_80pct_after_crossing']}"
        )
    print(f"[{args.label}] wrote {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
