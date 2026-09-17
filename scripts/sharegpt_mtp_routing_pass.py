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

The per-request ``reconciled`` field is an attribution result, not arithmetic:
``ar_output_tokens`` is ``completion - mtp`` by construction, so the row also
carries the committed cycle counts that independently bound the speculative
output, and the failed checks when it does not reconcile. Start the server with
``HIPENGINE_MTP2_OUTPUT_SPANS=1`` to have the backend record committed output
spans (execution mode, reason, emitted-token position, token count); the harness
then records whether those spans tile the emitted output, which is the strongest
form of the proof. ``fallback_event_counts`` counts non-speculative steps and
refusals (events) and is never a token count; the token-level attribution for
the same reasons is ``ar_output_tokens_by_reason`` from the spans.

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
    error_body: Mapping[str, Any] | None = None
    # The realized-route and MTP extension can arrive on a choice (the terminal
    # chunk) or on the top-level event (the usage chunk). Keep whichever channel
    # carries it so a rejected or errored stream still reports its route.
    stream_extension: dict[str, Any] = {}
    answer: list[str] = []
    thinking: list[str] = []
    started = time.perf_counter()
    first_token_ms: float | None = None
    first_answer_ms: float | None = None
    first_thinking_ms: float | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if isinstance(event.get("error"), Mapping):
                # A request rejected mid-stream (for example min_tokens without
                # eos_token_id) arrives as a normal SSE chunk with
                # finish_reason="error" and this body. Recording it is the
                # difference between "stream ended without usage" and the cause.
                error_body = event["error"]
            if isinstance(event.get("hipengine"), Mapping):
                stream_extension.update(event["hipengine"])
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                # A Qwen-style thinking reply streams its reasoning on a
                # separate channel, so the first generated token is usually a
                # reasoning token. Both channels are timed and counted.
                if delta.get("reasoning_content"):
                    if first_thinking_ms is None:
                        first_thinking_ms = (time.perf_counter() - started) * 1000.0
                    if first_token_ms is None:
                        first_token_ms = first_thinking_ms
                    thinking.append(delta["reasoning_content"])
                if delta.get("content"):
                    if first_answer_ms is None:
                        first_answer_ms = (time.perf_counter() - started) * 1000.0
                    if first_token_ms is None:
                        first_token_ms = first_answer_ms
                    answer.append(delta["content"])
                if choice.get("finish_reason") is not None:
                    terminal = choice
                if isinstance(choice.get("hipengine"), Mapping):
                    stream_extension.update(choice["hipengine"])
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
    terminal_extension = (terminal or {}).get("hipengine") or {}
    extension = {**stream_extension, **terminal_extension}
    return {
        "ttft_ms": first_token_ms,
        "first_answer_ms": first_answer_ms,
        "first_thinking_ms": first_thinking_ms,
        "first_decode_state_ms": timeline[0][0] if timeline else None,
        "e2e_ms": e2e_ms,
        "timeline": timeline,
        "usage": usage,
        "error_body": None if error_body is None else dict(error_body),
        "answer_characters": len("".join(answer)),
        "thinking_characters": len("".join(thinking)),
        "speculative_mtp": extension.get("speculative_mtp"),
        "generation_shape": extension.get("generation_shape"),
        "routing": extension.get("routing"),
    }

def _error_row(sample: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Describe one request that produced no usage, without hiding the cause."""

    error = result.get("error_body") or {}
    extension = error.get("hipengine") if isinstance(error.get("hipengine"), Mapping) else {}
    message = error.get("message") or "stream ended without usage"
    detail = [str(message)]
    if error.get("code"):
        detail.append(f"code={error['code']}")
    if error.get("type"):
        detail.append(f"type={error['type']}")
    if extension.get("status_code") is not None:
        detail.append(f"status={extension['status_code']}")
    detail.append(
        f"events={len(result.get('timeline') or [])},"
        f" answer_characters={result.get('answer_characters')},"
        f" thinking_characters={result.get('thinking_characters')},"
        f" e2e_ms={result.get('e2e_ms')}"
    )
    return {
        "source_id": sample.get("source_id"),
        "prompt_tokens_local": sample.get("prompt_tokens"),
        "expected_output_tokens": sample.get("expected_output_tokens"),
        "e2e_ms": result.get("e2e_ms"),
        "error": " ".join(detail),
        "error_code": error.get("code"),
        "error_type": error.get("type"),
        "error_status_code": extension.get("status_code"),
        "error_message": None if error.get("message") is None else str(error["message"]),
        "error_body": None if not error else dict(error),
        # Even a rejected request carries its realized route when the server
        # reported it, which is what makes a route-level rejection legible.
        **_route_fields(result),
    }


def _route_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    """Realized route, planned route, and refusal reason for one request."""

    shape = result.get("generation_shape") or {}
    decision = shape.get("route_decision") or {}
    return {
        # ``effective_route`` is the route the request actually executed, which
        # is what a reader needs to compare against ``selected_route``. Reporting
        # only the latter made an earlier pass read as "route=speculative_mtp
        # with mtp_used=false".
        "effective_route": shape.get("route"),
        "selected_route": decision.get("selected_route"),
        "requested_route": decision.get("requested_route"),
        "route_decision_reason": decision.get("reason"),
        "k0_class": decision.get("k0_class"),
        "policy_reason": decision.get("policy_reason"),
        "static_intent_allowed": decision.get("static_intent_allowed"),
        "realized_group_rows": decision.get("realized_group_rows"),
        "output_horizon_tokens": decision.get("output_horizon_tokens"),
        "selected_candidate_count": decision.get("selected_candidate_count"),
    }


def _request_row(sample: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    usage = result.get("usage") or {}
    if usage.get("completion_tokens") is None:
        # A stream that ends without usage is a truncated or aborted response;
        # record it as a failure instead of an empty success row.
        return _error_row(sample, result)
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
        "first_answer_ms": result.get("first_answer_ms"),
        "first_thinking_ms": result.get("first_thinking_ms"),
        "e2e_ms": result.get("e2e_ms"),
        "answer_characters": result.get("answer_characters"),
        "thinking_characters": result.get("thinking_characters"),
        "route": shape.get("route"),
        **_route_fields(result),
        "mtp_used": mtp.get("used"),
        "mtp_output_tokens": mtp.get("mtp_output_tokens"),
        "ar_output_tokens": mtp.get("ar_output_tokens"),
        "mtp_coverage": accounting.get("mtp_coverage"),
        "first_fallback_position": mtp.get("first_fallback_position"),
        "fallback_reason": mtp.get("fallback_reason"),
        # Events, not tokens: one entry per non-speculative step or refusal. The
        # token-level attribution for the same reasons comes from
        # ``ar_output_tokens_by_reason`` when diagnostic spans were recorded.
        "fallback_event_counts": mtp.get("fallback_event_counts"),
        "fallback_event_total": mtp.get("fallback_event_total"),
        "selected_depth_histogram": mtp.get("selected_depth_histogram"),
        "draft_cycles": mtp.get("draft_cycles"),
        "accepted_draft_tokens": mtp.get("accepted_draft_tokens"),
        "ar_output_tokens_in_cycles": accounting.get(
            "ar_output_tokens_in_cycles"
        ),
        "mtp_output_tokens_explained_by_cycles": accounting.get(
            "mtp_output_tokens_explained_by_cycles"
        ),
        "unexplained_mtp_output_tokens": accounting.get(
            "unexplained_mtp_output_tokens"
        ),
        "reconciled": accounting.get("reconciled"),
        "reconciled_reasons": accounting.get("reconciled_reasons"),
        "span_accounting": accounting.get("span_accounting"),
        "ar_output_tokens_by_reason": (
            (accounting.get("span_accounting") or {}).get("ar_tokens_by_reason")
            if isinstance(accounting.get("span_accounting"), Mapping)
            else None
        ),
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


def _speculation_case(row: Mapping[str, Any]) -> str:
    """Classify how far one request's speculation ran.

    ``first_fallback_position`` is ``0`` when speculation never covered a token,
    ``None`` when it covered every token after the prompt root (so the request
    ended inside speculation), and a middle value when the request left
    speculation mid-stream. Collapsing the last two into one empty block hid the
    difference between "never entered" and "stayed to the end".
    """

    if not row.get("mtp_used"):
        return "never_entered_speculation"
    position = row.get("first_fallback_position")
    if position is None:
        return "ended_inside_speculation"
    if int(position) <= 0:
        return "never_entered_speculation"
    return "left_mid_stream"


def _cliff(rows: Sequence[Mapping[str, Any]], *, window: int = 16) -> dict[str, Any]:
    """Quantify the latency step a request takes when it leaves speculation."""

    cases: Counter[str] = Counter(_speculation_case(row) for row in rows)
    measured: list[dict[str, Any]] = []
    for row in rows:
        if _speculation_case(row) != "left_mid_stream":
            continue
        position = int(row.get("first_fallback_position") or 0)
        timeline = row.get("timeline") or []
        if not timeline:
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
    base: dict[str, Any] = {
        "case_counts": dict(sorted(cases.items())),
        "left_mid_stream_requests": cases.get("left_mid_stream", 0),
        "never_entered_speculation_requests": cases.get(
            "never_entered_speculation", 0
        ),
        "ended_inside_speculation_requests": cases.get(
            "ended_inside_speculation", 0
        ),
    }
    if not measured:
        # Say why there is no rate step instead of reporting an empty block that
        # reads like a measurement of zero.
        base["requests"] = 0
        base["note"] = (
            "no request both left speculation mid-stream and had rate windows on "
            "both sides of the crossing"
        )
        return base
    ratios = [entry["ratio"] for entry in measured]
    return {
        **base,
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
        # Two refusals that a route-level share hides: a long prompt still
        # priming its provider refuses with prompt_activation_in_flight, and a
        # plan with no provider at all refuses with no_provider. Count both per
        # bucket, because the first pass showed long prompts losing the route
        # while mid-length ones kept it.
        events = row.get("fallback_event_counts") or {}
        if isinstance(events, Mapping):
            for reason in ("prompt_activation_in_flight", "no_provider"):
                count = int(events.get(reason) or 0)
                if count:
                    prompt_buckets[bucket][f"{reason}_requests"] += 1
                    prompt_buckets[bucket][f"{reason}_events"] += count
        reason = row.get("fallback_reason")
        if reason in {"prompt_activation_in_flight", "no_provider"}:
            prompt_buckets[bucket][f"primary_{reason}_requests"] += 1
    ttfts = [float(row["ttft_ms"]) for row in rows if row.get("ttft_ms")]
    answers = [
        float(row["first_answer_ms"])
        for row in rows
        if row.get("first_answer_ms")
    ]
    thinking = [
        float(row["first_thinking_ms"])
        for row in rows
        if row.get("first_thinking_ms")
    ]
    accounted = [row for row in rows if row.get("reconciled") is not None]
    unreconciled = [
        {
            "source_id": row.get("source_id"),
            "reconciled_reasons": row.get("reconciled_reasons"),
        }
        for row in accounted
        if row.get("reconciled") is not True
    ]
    span_proofs = [
        row for row in rows if isinstance(row.get("span_accounting"), Mapping)
    ]
    event_total = sum(int(row.get("fallback_event_total") or 0) for row in rows)
    # Events are not tokens. A planning refusal that is retried on the next step
    # costs one autoregressive step and no route, so an event count alone reads
    # as a lost route when the request was in fact almost entirely speculative.
    # The spanned token attribution is the quantity that answers "what did this
    # refusal cost", so both are reported.
    ar_tokens_by_reason: Counter[str] = Counter()
    for row in rows:
        attributed = row.get("ar_output_tokens_by_reason")
        if isinstance(attributed, Mapping):
            for reason, count in attributed.items():
                ar_tokens_by_reason[str(reason)] += int(count or 0)
    return {
        "requests": len(rows),
        "mtp_requests": len(used),
        "mtp_request_share": (len(used) / len(rows)) if rows else 0.0,
        "completion_tokens": completion,
        "mtp_output_tokens": mtp_outputs,
        "mtp_output_share": (mtp_outputs / completion) if completion else 0.0,
        "ar_output_tokens": sum(
            int(row.get("ar_output_tokens") or 0) for row in rows
        ),
        # Attribution gate. ``reconciled`` is not arithmetic consistency: it is
        # decided from the committed cycle counts and, in diagnostic mode, from
        # the committed output spans tiling the emitted output.
        "accounted_requests": len(accounted),
        "reconciled_requests": sum(
            1 for row in accounted if row.get("reconciled") is True
        ),
        "unreconciled_requests": unreconciled,
        "span_proof_requests": len(span_proofs),
        "span_proof_failures": [
            {
                "source_id": row.get("source_id"),
                "reconciled_reasons": (row.get("span_accounting") or {}).get(
                    "reconciled_reasons"
                ),
            }
            for row in span_proofs
            if (row.get("span_accounting") or {}).get("reconciled") is not True
        ],
        # Event counts stay separate from token counts: this is the number of
        # non-speculative steps and refusals, which is a different quantity from
        # ``ar_output_tokens`` and from ``mtp_output_tokens``.
        "fallback_event_total": event_total,
        "non_mtp_reasons": dict(sorted(reasons.items())),
        # Token-level attribution of the same refusals, from the committed output
        # spans (present when the server runs with HIPENGINE_MTP2_OUTPUT_SPANS=1).
        "ar_tokens_by_reason": dict(sorted(ar_tokens_by_reason.items())),
        "ar_tokens_attributed": sum(ar_tokens_by_reason.values()),
        "prompt_token_buckets": {
            bucket: dict(sorted(counts.items()))
            for bucket, counts in sorted(prompt_buckets.items())
        },
        # ttft_ms is time to the first generated token of either channel, which
        # for a thinking reply is a reasoning token.
        "median_ttft_ms": statistics.median(ttfts) if ttfts else None,
        "ttft_requests": len(ttfts),
        "median_first_answer_ms": statistics.median(answers) if answers else None,
        "answer_requests": len(answers),
        "median_first_thinking_ms": (
            statistics.median(thinking) if thinking else None
        ),
        "thinking_requests": len(thinking),
        "thinking_output_share": (
            sum(1 for row in rows if row.get("first_thinking_ms")) / len(rows)
            if rows
            else 0.0
        ),
        # Character split, so a reader can see how much of the output was
        # reasoning rather than answer text.
        "answer_characters": sum(int(row.get("answer_characters") or 0) for row in rows),
        "thinking_characters": sum(
            int(row.get("thinking_characters") or 0) for row in rows
        ),
        "thinking_character_share": (
            sum(int(row.get("thinking_characters") or 0) for row in rows)
            / max(
                1,
                sum(int(row.get("thinking_characters") or 0) for row in rows)
                + sum(int(row.get("answer_characters") or 0) for row in rows),
            )
        ),
        # Realized vs planned route, per request. A pass where every request
        # reports effective_route=ar while selected_route=speculative_mtp is a
        # route-level refusal, not a lying route.
        "effective_routes": dict(
            sorted(Counter(str(row.get("effective_route")) for row in rows).items())
        ),
        "selected_routes": dict(
            sorted(Counter(str(row.get("selected_route")) for row in rows).items())
        ),
        "route_disagreements": [
            {
                "source_id": row.get("source_id"),
                "effective_route": row.get("effective_route"),
                "selected_route": row.get("selected_route"),
                "route_decision_reason": row.get("route_decision_reason"),
                "k0_class": row.get("k0_class"),
                "fallback_reason": row.get("fallback_reason"),
            }
            for row in rows
            if row.get("selected_route")
            and row.get("effective_route")
            and row.get("selected_route") != row.get("effective_route")
        ],
        "cliff": _cliff(rows, window=window),
    }


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


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
    print(
        f"[{args.label}] refusal cost: events={summary['fallback_event_total']} "
        f"spanned_ar_tokens={summary['ar_tokens_attributed']} "
        f"by_reason={json.dumps(summary['ar_tokens_by_reason'])} "
        f"(of {summary['ar_output_tokens']} autoregressive tokens)"
    )
    print(
        f"[{args.label}] attribution: "
        f"reconciled={summary['reconciled_requests']}/{summary['accounted_requests']} "
        f"span_proofs={summary['span_proof_requests']} "
        f"fallback_events={summary['fallback_event_total']}"
    )
    if summary["unreconciled_requests"]:
        print(
            f"[{args.label}] UNRECONCILED: "
            f"{json.dumps(summary['unreconciled_requests'])}"
        )
    if summary["span_proof_failures"]:
        print(
            f"[{args.label}] SPAN PROOF FAILURES: "
            f"{json.dumps(summary['span_proof_failures'])}"
        )
    elif not summary["span_proof_requests"]:
        print(
            f"[{args.label}] note: no committed output spans were reported; "
            "start the server with HIPENGINE_MTP2_OUTPUT_SPANS=1 to record "
            "the per-span attribution proof"
        )
    print(
        f"[{args.label}] latency: ttft(first token)={_fmt_ms(summary['median_ttft_ms'])} "
        f"over {summary['ttft_requests']} requests, "
        f"first_answer={_fmt_ms(summary['median_first_answer_ms'])} "
        f"over {summary['answer_requests']}, "
        f"first_thinking={_fmt_ms(summary['median_first_thinking_ms'])} "
        f"over {summary['thinking_requests']}"
    )
    print(
        f"[{args.label}] output characters: answer={summary['answer_characters']} "
        f"thinking={summary['thinking_characters']} "
        f"({summary['thinking_character_share']:.1%} reasoning)"
    )
    print(
        f"[{args.label}] routes: effective={json.dumps(summary['effective_routes'])} "
        f"selected={json.dumps(summary['selected_routes'])}"
    )
    if summary["route_disagreements"]:
        print(
            f"[{args.label}] ROUTE DISAGREEMENTS "
            f"({len(summary['route_disagreements'])}): "
            f"{json.dumps(summary['route_disagreements'][:5])}"
        )
    buckets = summary["prompt_token_buckets"]
    refused = {
        bucket: counts
        for bucket, counts in buckets.items()
        if any(
            key.endswith("_requests") and key.startswith(("prompt_activation", "no_provider"))
            for key in counts
        )
    }
    if refused:
        print(
            f"[{args.label}] route refusals by prompt bucket: "
            f"{json.dumps(refused)}"
        )
    cliff = summary["cliff"]
    print(
        f"[{args.label}] speculation coverage: "
        f"never_entered={cliff['never_entered_speculation_requests']} "
        f"left_mid_stream={cliff['left_mid_stream_requests']} "
        f"ended_inside={cliff['ended_inside_speculation_requests']}"
    )
    if cliff.get("requests"):
        print(
            f"[{args.label}] cliff over {cliff['requests']} requests: "
            f"before={cliff['median_before_tokens_per_second']:.2f} "
            f"after={cliff['median_after_tokens_per_second']:.2f} tok/s "
            f"median_ratio={cliff['median_ratio']:.2f} "
            f"slower_than_80pct={cliff['requests_slower_than_80pct_after_crossing']}"
        )
    elif cliff.get("note"):
        print(f"[{args.label}] cliff: {cliff['note']}")
    if failures:
        print(
            f"[{args.label}] failures: "
            f"{json.dumps([{k: row.get(k) for k in ('source_id', 'error', 'error_code', 'error_status_code', 'effective_route', 'selected_route')} for row in failures[:5]])}"
        )
    print(f"[{args.label}] wrote {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
