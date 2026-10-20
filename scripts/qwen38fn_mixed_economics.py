#!/usr/bin/env python3
"""Bounded streamed HTTP economics capture for Flash-Next serving.

Two arms share one prompt suite and one resident server:

* ``ar`` — every request pins ``speculative_mtp: false``, so the wave is a true
  autoregressive baseline. Omitting the key is not the same request: the server
  may then select MTP on its own, which silently turns the AR arm into an
  unknown arm.
* ``mixed`` — exactly one request in the wave asks for MTP and the rest stay
  pinned AR. That one-MTP-plus-AR-peer shape is what the serving campaign
  prices.

Each wave streams ``/v1/completions`` with ``include_hipengine`` and
``include_usage``, so token arrival times, the final exact generated IDs and
counts, and the per-request route/acceptance diagnostics come from the server's
own counters. Timestamps are client receipt times of delivered deltas: when the
tokenizer merges tokens until a complete UTF-8 span exists, the delta carries
one timestamp and its true multi-token width, and no intermediate time is
invented. A baseline server that publishes no exact streamed count still yields
timestamped deltas with an unknown width, and one that publishes no
speculative-MTP block yields ``speculative_mtp: null``. Missing telemetry is
recorded as missing, not raised as a parse failure.

``/v1/hipengine/decode`` is read around every wave. Its ``step_widths``
histogram is the engine's physical decode width and ``mixed_target_passes`` /
``last_mixed_target_pass`` report physical target rows. Neither is the number of
HTTP requests in flight, and this capture never converts one into the other.

The protocol is bounded: ``--widths`` x ``--arms`` x ``--repeats`` waves plus
one discarded warmup wave, each wave ``--steps`` generated tokens per request.

``--prompts`` takes the suite files; the canonical pair is
``benchmarks/prompts/mtpbench-code-general-ja.jsonl`` and
``benchmarks/prompts/gdn-prefill-category-heldouts.jsonl``. Both are read
through ``scripts/gguf_mtp_category_bench.load_prompt_rows``, whose normalized
prompt text is the exact string sent to ``/v1/completions``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _candidate in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

SCHEMA = "hipengine.qwen38fn.mixed_economics.v1"
KIND = "qwen38fn_mixed_economics"
ARMS = ("ar", "mixed")
DEFAULT_PROMPTS = (
    REPO_ROOT / "benchmarks" / "prompts" / "mtpbench-code-general-ja.jsonl",
    REPO_ROOT / "benchmarks" / "prompts" / "gdn-prefill-category-heldouts.jsonl",
)
DEFAULT_WIDTHS = (1, 2, 4, 8)
DEFAULT_STEPS = 64
DEFAULT_REPEATS = 3
STREAM_METADATA_SCOPE = (
    "client receipt times of delivered stream deltas; a coalesced delta carries "
    "one timestamp and its reported multi-token width"
)
DECODE_SCOPE = "physical decode steps and target rows, not HTTP request counts"

# The speculative-MTP extension block is wide. These are the fields this capture
# prices per request; the rest of the block stays on the server.
MTP_DIAGNOSTIC_KEYS = (
    "effective_route",
    "selected_route",
    "requested_route",
    "used",
    "decision_reason",
    "selection_reason",
    "fallback_reason",
    "draft_tokens",
    "accepted_draft_tokens",
    "rejected_draft_tokens",
    "acceptance_rate",
    "draft_cycles",
    "mtp_output_tokens",
    "ar_output_tokens",
    "first_fallback_position",
)
MTP_ACCOUNTING_KEYS = (
    "completion_tokens",
    "mtp_output_tokens",
    "ar_output_tokens",
    "ar_output_tokens_in_cycles",
    "reconciled",
    "reconciled_reasons",
)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Interpolating percentile; an empty sample set has no percentile."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    """p50/p95/p99/max of delivered inter-token intervals, in milliseconds.

    No observed interval reports ``None`` for every percentile. Reporting ``0``
    would claim instantaneous decode from an empty sample.
    """

    return {
        "intervals": len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values) if values else None,
    }


def token_intervals(record: Mapping[str, Any]) -> list[float]:
    """Client-visible gaps between delivered deltas for one request, in ms."""

    times = [float(value) for value in record.get("token_times") or ()]
    return [(end - start) * 1000.0 for start, end in zip(times, times[1:])]


def arm_flags(arm: str, width: int) -> list[bool]:
    """Per-request ``speculative_mtp`` flags for one wave.

    ``mixed`` asks exactly one request for MTP and pins the rest AR. At width 1
    that is a single MTP request, and an odd or partial group still carries one
    MTP request rather than rounding the shape.
    """

    if width < 1:
        raise ValueError(f"width must be at least 1, got {width}")
    if arm == "ar":
        return [False] * width
    if arm == "mixed":
        return [True] + [False] * (width - 1)
    raise ValueError(f"unknown arm: {arm!r}; expected one of {list(ARMS)}")


def arm_order(arms: Sequence[str], repeat: int) -> list[str]:
    """Arm order for one repeat, flipped on every other repeat."""

    ordered = list(arms)
    if repeat % 2 == 1:
        ordered.reverse()
    return ordered


def wave_plan(
    *,
    arms: Sequence[str],
    widths: Sequence[int],
    repeats: int,
    row_count: int,
    waves_per_cell: int = 1,
) -> list[dict[str, Any]]:
    """Deterministic bounded wave order.

    Prompt selection is keyed on the (repeat, width, cell) triple rather than on
    the arm, so the AR and mixed arms of one cell run the same prompts in the
    same order and their rates are comparable. Cells advance one prompt at a
    time so a full-suite cell count puts every prompt in the MTP role, even when
    batch width and suite size have a common divisor.
    """

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if waves_per_cell < 1:
        raise ValueError("waves_per_cell must be at least 1")
    if row_count < 1:
        raise ValueError("row_count must be at least 1")
    if not arms or not widths:
        raise ValueError("at least one arm and one width are required")
    plan: list[dict[str, Any]] = []
    cell_index = 0
    for repeat in range(repeats):
        for width in widths:
            for cell in range(waves_per_cell):
                offset = cell_index % int(row_count)
                for arm in arm_order(arms, repeat):
                    plan.append(
                        {
                            "index": len(plan),
                            "repeat": repeat,
                            "width": int(width),
                            "cell": cell,
                            "arm": arm,
                            "prompt_offset": offset,
                        }
                    )
                cell_index += 1
    return plan


def wave_prompts(
    rows: Sequence[Mapping[str, Any]], *, width: int, offset: int
) -> list[Mapping[str, Any]]:
    """The ``width`` prompts for one wave, cycling the suite when it is shorter."""

    if not rows:
        raise ValueError("the prompt suite is empty")
    if width < 1:
        raise ValueError(f"width must be at least 1, got {width}")
    start = int(offset) % len(rows)
    return [rows[(start + index) % len(rows)] for index in range(width)]


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_payload(
    model: str, prompt: str, *, speculative_mtp: bool, steps: int
) -> dict[str, Any]:
    """The streamed completion request for one row.

    ``speculative_mtp`` is always written, including when it is ``False``. An
    omitted key lets the server's automatic selection run, which would make the
    AR arm an unknown arm instead of a baseline.
    """

    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    return {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(steps),
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True, "include_hipengine": True},
        "speculative_mtp": bool(speculative_mtp),
    }


class StreamCapture:
    """One streamed completion, accumulated from the server's own events.

    A delivered delta is timestamped when it arrives. When the server reports an
    exact cumulative token count the delta also records its true width, which
    can exceed one because the tokenizer merges tokens until a complete UTF-8
    span exists. When the server reports no count, the delta is still
    timestamped and its width stays ``None``; no per-token time is invented and
    no width is guessed.
    """

    def __init__(self, *, started_at: float) -> None:
        self.started_at = float(started_at)
        self.token_times: list[float] = []
        self.event_token_counts: list[int | None] = []
        self.event_elapsed_ms: list[float | None] = []
        self.events = 0
        self._last_count: int | None = None
        self._count_missing = False
        self.final_generated_token_ids: list[int] | None = None
        self.done_event_generated_tokens: int | None = None
        self.finish_reason: str | None = None
        self.finish_details: dict[str, Any] | None = None
        self.decode_state: dict[str, Any] | None = None
        self.route: str | None = None
        self.route_decision: dict[str, Any] | None = None
        self.speculative_mtp: dict[str, Any] | None = None
        self.diagnostics: dict[str, Any] | None = None
        self.usage: dict[str, int | None] | None = None

    def observe(self, event: Mapping[str, Any], *, now: float) -> None:
        if not isinstance(event, Mapping):
            raise RuntimeError(
                f"stream event is not an object: {type(event).__name__}"
            )
        if "error" in event:
            raise RuntimeError(
                f"stream failed: {json.dumps(event, ensure_ascii=False)[:400]}"
            )
        self.events += 1
        usage = event.get("usage")
        if isinstance(usage, Mapping):
            self.usage = {
                name: _as_int(usage.get(name))
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
        event_metadata = event.get("hipengine")
        event_metadata = event_metadata if isinstance(event_metadata, Mapping) else {}
        elapsed_ms = None
        timing = event_metadata.get("timing")
        if isinstance(timing, Mapping):
            elapsed_ms = _as_float(timing.get("elapsed_ms"))
        for choice in event.get("choices") or ():
            if not isinstance(choice, Mapping):
                continue
            metadata = choice.get("hipengine")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            finish_reason = choice.get("finish_reason")
            if finish_reason is None:
                self._observe_delta(metadata, now=now, elapsed_ms=elapsed_ms)
            else:
                self._observe_done(metadata, finish_reason=finish_reason)

    def _observe_delta(
        self, metadata: Mapping[str, Any], *, now: float, elapsed_ms: float | None
    ) -> None:
        tokens = metadata.get("tokens")
        tokens = tokens if isinstance(tokens, Mapping) else None
        count = None if tokens is None else _as_int(tokens.get("streamed_tokens"))
        if count is None:
            self._count_missing = True
            width: int | None = None
        else:
            previous = 0 if self._last_count is None else self._last_count
            if count < previous:
                raise RuntimeError(
                    f"streamed token count moved backwards: {previous} -> {count}"
                )
            if count == previous:
                # A delta that adds no token still carries timing metadata. It
                # is not a delivered token, so it must not become a latency
                # sample.
                return
            width = count - previous
            self._last_count = count
        self.token_times.append(float(now))
        self.event_token_counts.append(width)
        self.event_elapsed_ms.append(elapsed_ms)

    def _observe_done(
        self, metadata: Mapping[str, Any], *, finish_reason: Any
    ) -> None:
        self.finish_reason = str(finish_reason)
        ids = metadata.get("generated_token_ids")
        if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
            self.final_generated_token_ids = [int(value) for value in ids]
        count = _as_int(metadata.get("generated_tokens"))
        if count is None and self.final_generated_token_ids is not None:
            count = len(self.final_generated_token_ids)
        self.done_event_generated_tokens = count
        details = metadata.get("finish_details")
        self.finish_details = dict(details) if isinstance(details, Mapping) else None
        state = metadata.get("decode_state")
        self.decode_state = dict(state) if isinstance(state, Mapping) else None
        shape = metadata.get("generation_shape")
        if isinstance(shape, Mapping) and shape.get("route") is not None:
            self.route = str(shape.get("route"))
            decision = shape.get("route_decision")
            self.route_decision = (
                dict(decision) if isinstance(decision, Mapping) else None
            )
        summary = metadata.get("speculative_mtp")
        self.speculative_mtp = dict(summary) if isinstance(summary, Mapping) else None
        diagnostics = metadata.get("diagnostics")
        self.diagnostics = (
            dict(diagnostics) if isinstance(diagnostics, Mapping) else None
        )

    @property
    def event_streamed_tokens(self) -> int | None:
        """Exact token total from the deltas, or ``None`` if any delta lacked it."""

        if self._count_missing or self._last_count is None:
            return None
        return self._last_count

    def final_generated_tokens(self) -> int | None:
        """Exact final token count: done event, then usage, then the deltas."""

        if self.done_event_generated_tokens is not None:
            return int(self.done_event_generated_tokens)
        usage = self.usage or {}
        completion = _as_int(usage.get("completion_tokens"))
        if completion is not None:
            return completion
        return self.event_streamed_tokens

    def accounting(self) -> dict[str, Any]:
        """Reconcile the independent token counts the server can publish.

        ``consistent`` is ``None`` when fewer than two counts were published:
        one number cannot disagree with itself, and silence is not agreement.
        """

        event_total = self.event_streamed_tokens
        done_total = self.done_event_generated_tokens
        usage_total = _as_int((self.usage or {}).get("completion_tokens"))
        observed = [
            value
            for value in (event_total, done_total, usage_total)
            if value is not None
        ]
        return {
            "event_streamed_tokens": event_total,
            "done_event_generated_tokens": done_total,
            "usage_completion_tokens": usage_total,
            "sources": len(observed),
            "consistent": None if len(observed) < 2 else len(set(observed)) == 1,
            "coalesced_events": sum(
                1
                for count in self.event_token_counts
                if count is not None and count > 1
            ),
            "delivered_events": len(self.token_times),
            "events": self.events,
        }


def _mtp_diagnostics(summary: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(summary, Mapping):
        return None
    diagnostics = {
        key: summary.get(key) for key in MTP_DIAGNOSTIC_KEYS if key in summary
    }
    accounting = summary.get("output_accounting")
    if isinstance(accounting, Mapping):
        diagnostics["output_accounting"] = {
            key: accounting.get(key)
            for key in MTP_ACCOUNTING_KEYS
            if key in accounting
        }
    execution = summary.get("execution")
    if isinstance(execution, Mapping):
        diagnostics["execution"] = dict(execution)
    return diagnostics


def request_record(
    capture: StreamCapture,
    *,
    row: Mapping[str, Any],
    arm: str,
    requested_speculative_mtp: bool,
    steps: int,
    wall_s: float,
    status: int = 200,
    error: str | None = None,
) -> dict[str, Any]:
    """One request's streamed evidence, with absent telemetry left absent."""

    final_tokens = capture.final_generated_tokens()
    times = capture.token_times
    return {
        "prompt_id": row.get("id"),
        "category": row.get("category"),
        "prompt_sha256": prompt_sha256(str(row.get("prompt") or "")),
        "arm": arm,
        "requested_speculative_mtp": bool(requested_speculative_mtp),
        "status": int(status),
        "error": error,
        "started_at": capture.started_at,
        "wall_s": float(wall_s),
        "ttft_ms": (times[0] - capture.started_at) * 1000.0 if times else None,
        "token_times": list(times),
        "token_offsets_ms": [
            (value - capture.started_at) * 1000.0 for value in times
        ],
        "event_token_counts": list(capture.event_token_counts),
        "event_elapsed_ms": list(capture.event_elapsed_ms),
        "final_generated_token_ids": capture.final_generated_token_ids,
        "final_generated_tokens": final_tokens,
        "expected_tokens": int(steps),
        "expected_tokens_match": (
            None if final_tokens is None else final_tokens == int(steps)
        ),
        "request_tok_s": (
            (final_tokens / wall_s)
            if final_tokens is not None and wall_s > 0
            else None
        ),
        "accounting": capture.accounting(),
        "usage": capture.usage,
        "finish_reason": capture.finish_reason,
        "finish_details": capture.finish_details,
        "decode_state": capture.decode_state,
        "route": capture.route,
        "route_published": capture.route is not None,
        "route_decision": capture.route_decision,
        "diagnostics": capture.diagnostics,
        "speculative_mtp_present": capture.speculative_mtp is not None,
        "speculative_mtp": _mtp_diagnostics(capture.speculative_mtp),
    }


def summarize_wave(
    records: Sequence[Mapping[str, Any]], wall_s: float
) -> dict[str, Any]:
    """Aggregate one wave from its per-request records.

    ``wall_s`` covers first send to last completion. Inter-token latency is
    reported for the wave's AR requests only: an MTP request's deltas cover
    several tokens each, so folding them in would price a different quantity.
    """

    ok = [record for record in records if int(record.get("status") or 0) == 200]
    ar_records = [
        record for record in ok if not record.get("requested_speculative_mtp")
    ]
    intervals = [
        interval for record in ar_records for interval in token_intervals(record)
    ]
    tokens = sum(int(record.get("final_generated_tokens") or 0) for record in ok)
    acceptance = [
        float(record["speculative_mtp"]["acceptance_rate"])
        for record in ok
        if isinstance(record.get("speculative_mtp"), Mapping)
        and record["speculative_mtp"].get("acceptance_rate") is not None
    ]
    prompt_ids = [record.get("prompt_id") for record in records]
    return {
        "requests": len(records),
        "ar_requests": len(ar_records),
        "mtp_requests": sum(
            1 for record in ok if record.get("requested_speculative_mtp")
        ),
        "errors": len(records) - len(ok),
        "wall_s": float(wall_s),
        "generated_tokens": tokens,
        "wave_tok_s": (tokens / wall_s) if wall_s > 0 else 0.0,
        "latency_scope": STREAM_METADATA_SCOPE,
        "ar_peer_inter_token": latency_summary(intervals),
        "ar_peer_inter_token_intervals_ms": intervals,
        "delivered_events": sum(
            int((record.get("accounting") or {}).get("delivered_events") or 0)
            for record in ok
        ),
        "coalesced_events": sum(
            int((record.get("accounting") or {}).get("coalesced_events") or 0)
            for record in ok
        ),
        "speculative_used_requests": sum(
            1
            for record in ok
            if isinstance(record.get("speculative_mtp"), Mapping)
            and record["speculative_mtp"].get("used") is True
        ),
        "speculative_present_requests": sum(
            1 for record in ok if record.get("speculative_mtp_present")
        ),
        "speculative_absent_requests": sum(
            1 for record in ok if not record.get("speculative_mtp_present")
        ),
        "acceptance_rate_sum": sum(acceptance),
        "acceptance_rate_samples": len(acceptance),
        "expected_token_mismatches": sum(
            1 for record in ok if record.get("expected_tokens_match") is False
        ),
        "prompt_ids": prompt_ids,
        "distinct_prompts": len(set(prompt_ids)),
        "reused_prompts": len(prompt_ids) - len(set(prompt_ids)),
    }


def decode_snapshot(body: Any) -> dict[str, Any]:
    """Compact engine decode state from one ``/v1/hipengine/decode`` body.

    An engine without a decode-telemetry hook, or one whose provider has no
    ``ar_batch`` owner, keeps ``available``/``step_widths`` truthful instead of
    reporting a zero-width histogram it never measured.
    """

    if not isinstance(body, Mapping):
        return {"available": False, "reason": "decode_telemetry_not_an_object"}
    if body.get("available") is not True:
        return {
            "available": False,
            "reason": body.get("reason") or "decode_telemetry_unavailable",
        }
    decode = body.get("decode")
    if not isinstance(decode, Mapping):
        return {"available": False, "reason": "decode_telemetry_missing_decode"}
    batch = decode.get("ar_batch")
    batch = batch if isinstance(batch, Mapping) else None
    widths = None
    if batch is not None:
        raw_widths = batch.get("step_widths")
        if isinstance(raw_widths, Mapping):
            widths = {}
            for key, value in raw_widths.items():
                count = _as_int(value)
                if count is not None:
                    widths[str(key)] = count
    last_pass = decode.get("last_mixed_target_pass")
    return {
        "available": True,
        "decode_path": decode.get("decode_path"),
        "request_capacity": _as_int(decode.get("request_capacity")),
        "ar_batch_present": batch is not None,
        "steps": None if batch is None else _as_int(batch.get("steps")),
        "live_slots": None if batch is None else _as_int(batch.get("live_slots")),
        "step_widths": widths,
        "failure": None if batch is None else batch.get("failure"),
        "mixed_target_passes": _as_int(decode.get("mixed_target_passes")),
        "last_mixed_target_pass": (
            dict(last_pass) if isinstance(last_pass, Mapping) else None
        ),
    }


def _difference(after: int | None, before: int | None) -> int | None:
    if after is None or before is None:
        return None
    return int(after) - int(before)


def decode_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Physical decode evidence moved across one wave.

    ``step_width_delta`` counts engine decode steps by physical width.
    ``mixed_target_pass`` is this wave's own mixed target pass, reported only
    when the wave ran one, and its ``target_rows`` counts physical rows in that
    pass. ``last_mixed_target_pass`` is the server's snapshot at the end of the
    wave and can be older than the wave. A wave's HTTP request count is reported
    beside these by the wave summary; it is never substituted for either.
    """

    result: dict[str, Any] = {
        "scope": DECODE_SCOPE,
        "available": bool(after.get("available")),
        "reason": None if after.get("available") else after.get("reason"),
        "decode_path": after.get("decode_path"),
        "live_slots": after.get("live_slots"),
        "step_widths": after.get("step_widths"),
        "step_width_delta": None,
        "steps_delta": None,
        "mixed_target_passes_delta": None,
        "mixed_target_pass": None,
        "last_mixed_target_pass": after.get("last_mixed_target_pass"),
        "failure": after.get("failure"),
    }
    if not result["available"]:
        return result
    result["steps_delta"] = _difference(after.get("steps"), before.get("steps"))
    passes = _difference(
        after.get("mixed_target_passes"), before.get("mixed_target_passes")
    )
    result["mixed_target_passes_delta"] = passes
    # The server's ``last_mixed_target_pass`` is the most recent pass ever, so it
    # outlives the wave that produced it. Attribute the rows to this wave only
    # when this wave actually ran a pass.
    if passes is not None and passes > 0:
        result["mixed_target_pass"] = after.get("last_mixed_target_pass")
    before_widths = before.get("step_widths")
    after_widths = after.get("step_widths")
    if isinstance(before_widths, Mapping) and isinstance(after_widths, Mapping):
        result["step_width_delta"] = {
            str(width): int(after_widths.get(width, 0))
            - int(before_widths.get(width, 0))
            for width in sorted(
                {*before_widths, *after_widths}, key=lambda value: int(value)
            )
        }
    return result


def aggregate(waves: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum a set of waves; pooled intervals, never an average of rates."""

    summaries = [wave["summary"] for wave in waves]
    wall = sum(float(summary["wall_s"]) for summary in summaries)
    tokens = sum(int(summary["generated_tokens"]) for summary in summaries)
    intervals = [
        interval
        for summary in summaries
        for interval in summary["ar_peer_inter_token_intervals_ms"]
    ]
    samples = sum(int(summary["acceptance_rate_samples"]) for summary in summaries)
    widths: dict[str, int] = {}
    steps = 0
    passes = 0
    physical_available = False
    for wave in waves:
        delta = wave.get("decode_delta") or {}
        if delta.get("steps_delta") is not None:
            physical_available = True
            steps += int(delta["steps_delta"])
        if delta.get("mixed_target_passes_delta") is not None:
            passes += int(delta["mixed_target_passes_delta"])
        for width, count in (delta.get("step_width_delta") or {}).items():
            widths[str(width)] = widths.get(str(width), 0) + int(count)
    return {
        "waves": len(summaries),
        "requests": sum(int(summary["requests"]) for summary in summaries),
        "errors": sum(int(summary["errors"]) for summary in summaries),
        "wall_s": round(wall, 3),
        "generated_tokens": tokens,
        "wave_tok_s": round(tokens / wall, 3) if wall > 0 else 0.0,
        "latency_scope": STREAM_METADATA_SCOPE,
        "ar_peer_inter_token": latency_summary(intervals),
        "delivered_events": sum(
            int(summary["delivered_events"]) for summary in summaries
        ),
        "coalesced_events": sum(
            int(summary["coalesced_events"]) for summary in summaries
        ),
        "speculative_used_requests": sum(
            int(summary["speculative_used_requests"]) for summary in summaries
        ),
        "speculative_present_requests": sum(
            int(summary["speculative_present_requests"]) for summary in summaries
        ),
        "speculative_absent_requests": sum(
            int(summary["speculative_absent_requests"]) for summary in summaries
        ),
        "mean_acceptance_rate": (
            round(sum(float(summary["acceptance_rate_sum"]) for summary in summaries) / samples, 4)
            if samples
            else None
        ),
        "expected_token_mismatches": sum(
            int(summary["expected_token_mismatches"]) for summary in summaries
        ),
        "physical_decode_steps": steps if physical_available else None,
        "physical_step_widths": widths if physical_available else None,
        "mixed_target_passes": passes if physical_available else None,
        "decode_scope": DECODE_SCOPE,
    }


def arm_ratios(
    waves: Sequence[Mapping[str, Any]], arms: Sequence[str]
) -> list[dict[str, Any]]:
    """Per-cell ratio of each non-AR arm against the AR arm of the same cell."""

    cells: dict[tuple[int, int, int], dict[str, Mapping[str, Any]]] = {}
    for wave in waves:
        key = (int(wave["repeat"]), int(wave["width"]), int(wave["cell"]))
        cells.setdefault(key, {})[str(wave["arm"])] = wave
    ratios: list[dict[str, Any]] = []
    for key in sorted(cells):
        group = cells[key]
        base = group.get("ar")
        if base is None:
            continue
        base_rate = float(base["summary"]["wave_tok_s"])
        for arm in arms:
            if arm == "ar" or arm not in group:
                continue
            rate = float(group[arm]["summary"]["wave_tok_s"])
            ratios.append(
                {
                    "repeat": key[0],
                    "width": key[1],
                    "cell": key[2],
                    "arm": arm,
                    "ar_wave_tok_s": base_rate,
                    "arm_wave_tok_s": rate,
                    "ratio_vs_ar": (rate / base_rate) if base_rate else None,
                    "ar_prompt_ids": base["summary"]["prompt_ids"],
                    "arm_prompt_ids": group[arm]["summary"]["prompt_ids"],
                }
            )
    return ratios


def load_suite(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load prompt suites through the canonical loader's normalization.

    ``load_prompt_rows`` owns normalization: a ``messages[]`` row becomes the
    joined user text, so a message suite and a prompt-string suite produce the
    same prompt string and the same hash. Imported lazily so wave arithmetic
    tests do not pay for the loader's own imports.
    """

    from gguf_mtp_category_bench import BenchError, load_prompt_rows

    rows: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    for path in paths:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"prompt suite not found: {candidate}")
        try:
            loaded = load_prompt_rows(candidate)
        except BenchError as exc:
            raise ValueError(str(exc)) from exc
        for row in loaded:
            prompt_id = str(row["id"])
            if prompt_id in seen:
                raise ValueError(
                    f"duplicate prompt id {prompt_id!r} in {candidate} and {seen[prompt_id]}"
                )
            seen[prompt_id] = candidate
            rows.append({**row, "source_path": str(candidate)})
    if not rows:
        raise ValueError("no prompt rows loaded")
    return rows


def _command(argv: Sequence[str]) -> list[str]:
    try:
        script = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    except ValueError:
        script = str(Path(__file__).resolve())
    return [sys.executable, script, *argv]


def _git(server_repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(server_repo), *args], text=True
    )


def _source_diff_sha256(server_repo: Path) -> str:
    return hashlib.sha256(
        subprocess.check_output(["git", "-C", str(server_repo), "diff", "HEAD"])
    ).hexdigest()


async def _get_json(client: httpx.AsyncClient, path: str) -> Any:
    response = await client.get(path)
    response.raise_for_status()
    return response.json()


async def stream_request(
    client: httpx.AsyncClient,
    *,
    payload: Mapping[str, Any],
    row: Mapping[str, Any],
    arm: str,
    steps: int,
    timeout: float,
) -> dict[str, Any]:
    """Stream one completion and return its record."""

    capture = StreamCapture(started_at=time.perf_counter())
    async with client.stream(
        "POST", "/v1/completions", json=dict(payload), timeout=timeout
    ) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode("utf-8", "replace")
            return request_record(
                capture,
                row=row,
                arm=arm,
                requested_speculative_mtp=bool(payload["speculative_mtp"]),
                steps=steps,
                wall_s=time.perf_counter() - capture.started_at,
                status=response.status_code,
                error=body[:400],
            )
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :].strip()
            if data == "[DONE]":
                break
            capture.observe(json.loads(data), now=time.perf_counter())
    return request_record(
        capture,
        row=row,
        arm=arm,
        requested_speculative_mtp=bool(payload["speculative_mtp"]),
        steps=steps,
        wall_s=time.perf_counter() - capture.started_at,
    )


async def run_wave(
    client: httpx.AsyncClient,
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    arm: str,
    width: int,
    prompt_offset: int,
    steps: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], float]:
    """Issue one wave's requests together and return records plus wall time."""

    prompts = wave_prompts(rows, width=width, offset=prompt_offset)
    flags = arm_flags(arm, len(prompts))
    barrier = asyncio.Event()

    async def worker(index: int) -> dict[str, Any]:
        row = prompts[index]
        payload = request_payload(
            model,
            str(row["prompt"]),
            speculative_mtp=flags[index],
            steps=steps,
        )
        await barrier.wait()
        started = time.perf_counter()
        try:
            return await stream_request(
                client,
                payload=payload,
                row=row,
                arm=arm,
                steps=steps,
                timeout=timeout,
            )
        except Exception as exc:  # a broken request is recorded, not fatal
            return request_record(
                StreamCapture(started_at=started),
                row=row,
                arm=arm,
                requested_speculative_mtp=flags[index],
                steps=steps,
                wall_s=time.perf_counter() - started,
                status=0,
                error=f"{type(exc).__name__}: {exc}",
            )

    started = time.perf_counter()
    tasks = [asyncio.create_task(worker(index)) for index in range(len(prompts))]
    barrier.set()
    records = list(await asyncio.gather(*tasks))
    return records, time.perf_counter() - started


async def run(
    args: argparse.Namespace, command: Sequence[str]
) -> dict[str, Any]:
    rows = load_suite(args.prompts)
    plan = wave_plan(
        arms=args.arms,
        widths=args.widths,
        repeats=args.repeats,
        row_count=len(rows),
        waves_per_cell=args.waves_per_cell,
    )
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": KIND,
        "label": args.label,
        "performance_claim": False,
        "status": "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": list(command),
        "url": args.url,
        "model": args.model,
        "protocol": {
            "widths": list(args.widths),
            "arms": list(args.arms),
            "steps": args.steps,
            "repeats": args.repeats,
            "waves_per_cell": args.waves_per_cell,
            "warmups_discarded": args.warmups,
            "arms_note": "AR pins speculative_mtp=false; mixed asks one request for MTP",
            "order": "arm order alternates every repeat",
        },
        "suite": {
            "paths": [str(Path(path)) for path in args.prompts],
            "sha256": {
                str(Path(path)): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for path in args.prompts
            },
            "rows": len(rows),
            "prompt_sha256": {
                str(row["id"]): prompt_sha256(str(row["prompt"])) for row in rows
            },
            "normalization": "scripts/gguf_mtp_category_bench.load_prompt_rows",
        },
        "server": {
            "repo": str(args.server_repo),
            "label": args.label,
            "commit": _git(args.server_repo, "rev-parse", "HEAD").strip(),
            "git_status": _git(args.server_repo, "status", "--porcelain"),
            "source_diff_sha256": _source_diff_sha256(args.server_repo),
        },
        "physical_host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
        "warmup": [],
        "waves": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    save()
    try:
        async with httpx.AsyncClient(base_url=args.url, timeout=args.timeout) as client:
            preflight: dict[str, Any] = {}
            for path in ("/ready", "/v1/hipengine/capabilities"):
                preflight[path] = await _get_json(client, path)
            before = decode_snapshot(await _get_json(client, "/v1/hipengine/decode"))
            preflight["decode"] = before
            artifact["preflight"] = preflight
            queue = (preflight["/ready"] or {}).get("queue") or {}
            if int(queue.get("depth") or 0) or int(queue.get("active_requests") or 0):
                raise RuntimeError(
                    f"server must be idle before this capture; queue is {queue}"
                )

            warmup_width = max(args.widths)
            for index in range(args.warmups):
                records, wall = await run_wave(
                    client,
                    rows,
                    model=args.model,
                    arm="mixed",
                    width=warmup_width,
                    prompt_offset=0,
                    steps=args.steps,
                    timeout=args.timeout,
                )
                artifact["warmup"].append(
                    {
                        "index": index,
                        "arm": "mixed",
                        "width": warmup_width,
                        "discarded": True,
                        "requests": records,
                        "summary": summarize_wave(records, wall),
                    }
                )
                save()

            for entry in plan:
                before = decode_snapshot(
                    await _get_json(client, "/v1/hipengine/decode")
                )
                records, wall = await run_wave(
                    client,
                    rows,
                    model=args.model,
                    arm=entry["arm"],
                    width=entry["width"],
                    prompt_offset=entry["prompt_offset"],
                    steps=args.steps,
                    timeout=args.timeout,
                )
                after = decode_snapshot(
                    await _get_json(client, "/v1/hipengine/decode")
                )
                wave = {
                    **entry,
                    "discarded": False,
                    "steps": args.steps,
                    "arm_flags": arm_flags(entry["arm"], entry["width"]),
                    "requests": records,
                    "summary": summarize_wave(records, wall),
                    "decode_before": before,
                    "decode_after": after,
                    "decode_delta": decode_delta(before, after),
                }
                artifact["waves"].append(wave)
                save()
                print(
                    json.dumps(
                        {
                            "wave": wave["index"],
                            "repeat": wave["repeat"],
                            "width": wave["width"],
                            "arm": wave["arm"],
                            "wave_tok_s": round(wave["summary"]["wave_tok_s"], 3),
                            "ar_peer_p50_ms": wave["summary"][
                                "ar_peer_inter_token"
                            ]["p50_ms"],
                            "mtp_used": wave["summary"]["speculative_used_requests"],
                            "physical_steps": wave["decode_delta"]["steps_delta"],
                            "target_rows": (
                                wave["decode_delta"]["mixed_target_pass"] or {}
                            ).get("target_rows"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            artifact["decode_after"] = decode_snapshot(
                await _get_json(client, "/v1/hipengine/decode")
            )
            artifact["ready_after"] = await _get_json(client, "/ready")
            artifact["source_unchanged"] = (
                artifact["server"]["source_diff_sha256"]
                == _source_diff_sha256(args.server_repo)
            )
    except BaseException as exc:
        # The partial artifact on disk must say the capture stopped, not that it
        # is still running.
        artifact["status"] = "failed"
        artifact["failure"] = f"{type(exc).__name__}: {exc}"
        save()
        raise

    artifact["arms"] = {
        arm: aggregate(
            [wave for wave in artifact["waves"] if wave["arm"] == arm]
        )
        for arm in args.arms
    }
    artifact["ratios"] = arm_ratios(artifact["waves"], args.arms)
    artifact["status"] = (
        "complete" if artifact["source_unchanged"] else "invalid_source_changed"
    )
    save()
    print(
        json.dumps(
            {
                "arms": {
                    arm: {
                        "waves": totals["waves"],
                        "wave_tok_s": totals["wave_tok_s"],
                        "p50_ms": totals["ar_peer_inter_token"]["p50_ms"],
                        "mtp_used": totals["speculative_used_requests"],
                        "absent_telemetry": totals["speculative_absent_requests"],
                    }
                    for arm, totals in artifact["arms"].items()
                },
                "ratios_vs_ar": [
                    {"width": ratio["width"], "repeat": ratio["repeat"],
                     "arm": ratio["arm"], "ratio": ratio["ratio_vs_ar"]}
                    for ratio in artifact["ratios"]
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return artifact


def verdict(artifact: Mapping[str, Any]) -> list[str]:
    """Problems that make the captured numbers unusable.

    Missing mixed telemetry is not one of them: a baseline server that publishes
    no speculative-MTP block is reported as absent, and an MTP request whose
    diagnostics say speculation did not run is a failed arm.
    """

    problems: list[str] = []
    for wave in artifact.get("waves") or ():
        summary = wave["summary"]
        if summary["errors"]:
            problems.append(
                f"wave {wave['index']} ({wave['arm']} width {wave['width']}) "
                f"has {summary['errors']} failed request(s)"
            )
        if summary["expected_token_mismatches"]:
            problems.append(
                f"wave {wave['index']} ({wave['arm']} width {wave['width']}) "
                "generated a token count that differs from --steps"
            )
        for record in wave["requests"]:
            if not record.get("requested_speculative_mtp"):
                continue
            diagnostics = record.get("speculative_mtp")
            if isinstance(diagnostics, Mapping) and diagnostics.get("used") is False:
                problems.append(
                    f"wave {wave['index']} request {record['prompt_id']} asked "
                    "for MTP and the server reported it did not run"
                )
    if artifact.get("source_unchanged") is False:
        problems.append("server source changed during the capture")
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18400")
    parser.add_argument("--model", default="qwen38fn-capon")
    parser.add_argument(
        "--prompts",
        type=Path,
        nargs="+",
        default=list(DEFAULT_PROMPTS),
        help="Prompt suite files, read through the canonical loader.",
    )
    parser.add_argument(
        "--widths",
        type=lambda text: _int_list(text, name="widths", minimum=1),
        default=list(DEFAULT_WIDTHS),
        help="Comma-separated concurrent request counts per wave.",
    )
    parser.add_argument(
        "--arms",
        type=lambda text: _arm_list(text),
        default=list(ARMS),
        help="Comma-separated arms: ar, mixed.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--waves-per-cell", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--server-repo",
        type=Path,
        required=True,
        help="Server source tree, for the commit/status/diff identity block.",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Free-form label recorded beside the server source identity.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _int_list(text: str, *, name: str, minimum: int) -> list[int]:
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError(f"--{name} entries must be integers")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--{name} entries must be integers"
        ) from exc
    bad = [value for value in values if value < minimum]
    if bad:
        raise argparse.ArgumentTypeError(
            f"--{name} entries must be >= {minimum}: {bad}"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"--{name} must not repeat a value: {values}")
    return values


def _arm_list(text: str) -> list[str]:
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError("--arms entries must be arm names")
    unknown = [part for part in parts if part not in ARMS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown arm(s) {unknown}; expected one of {list(ARMS)}"
        )
    if len(set(parts)) != len(parts):
        raise argparse.ArgumentTypeError(f"--arms must not repeat a value: {parts}")
    return parts


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if args.steps < 1 or args.repeats < 1 or args.waves_per_cell < 1:
        parser.error("--steps, --repeats and --waves-per-cell must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups must be nonnegative")
    try:
        artifact = asyncio.run(run(args, command=_command(raw_argv)))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    problems = verdict(artifact)
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    print(f"artifact: {args.output}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
