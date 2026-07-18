#!/usr/bin/env python3
"""Measure exact PARO resident OpenAI SSE concurrency.

The harness owns one fixed-capacity PARO model session per configuration and
uses the real FastAPI ``/v1/completions`` streaming route.  Authoritative token
IDs are captured at resident-runner reclaim, so throughput is accepted only
when every streamed request matches an independent same-session c1 oracle.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from fastapi.testclient import TestClient

from hipengine import LLM, SamplingParams
from hipengine.benchmark.prompts import token_ids_sha256
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.memory import memory_stats
from hipengine.server import ServerConfig, create_app


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    "/home/lhl/.cache/huggingface/hub/"
    "models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/snapshots/"
    "437eba06df05aad71a4dacdcaf3fff70ae1ee8a1"
)


@dataclass(frozen=True)
class Configuration:
    name: str
    rows: int
    native: bool


CONFIGURATIONS = {
    "c1": Configuration("c1", 1, True),
    "native_c2": Configuration("native_c2", 2, True),
    "native_c4": Configuration("native_c4", 4, True),
    "native_c8": Configuration("native_c8", 8, True),
    "serial_c8": Configuration("serial_c8", 8, False),
}
DEFAULT_CONFIGURATIONS = tuple(CONFIGURATIONS)


@dataclass
class ReclaimedRow:
    request_id: int
    prompt_ids: list[int]
    generated_ids: list[int]
    finish_reason: str
    route: dict[str, Any]


@dataclass
class StreamTrace:
    row_index: int
    status_code: int
    request_id: int | None
    started_at: float
    first_delta_at: float | None
    completed_at: float
    delta_times: list[float]
    text: str
    usage: dict[str, Any] | None
    finish_reason: str | None
    done: bool
    error: str | None


def _parse_configurations(raw: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not names:
        raise argparse.ArgumentTypeError("configurations must not be empty")
    unknown = sorted(set(names) - set(CONFIGURATIONS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown configurations: {unknown!r}")
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("configurations must be unique")
    return names


def _stats(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "samples": [],
            "count": 0,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
            "stdev": None,
            "stdev_pct_of_median": None,
        }
    ordered = sorted(samples)
    median = float(statistics.median(samples))
    p95 = float(ordered[min(len(ordered) - 1, max(0, (95 * len(ordered) + 99) // 100 - 1))])
    stdev = float(statistics.stdev(samples)) if len(samples) > 1 else 0.0
    return {
        "samples": samples,
        "count": len(samples),
        "median": median,
        "p95": p95,
        "min": float(min(samples)),
        "max": float(max(samples)),
        "stdev": stdev,
        "stdev_pct_of_median": 100.0 * stdev / median if median else None,
    }


def _counter_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(key): int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(set(before) | set(after))
    }


def _fallback_reasons_ok(
    config: Configuration,
    fallback_delta: Mapping[str, Any],
) -> bool:
    active = {
        str(key): int(value)
        for key, value in fallback_delta.items()
        if int(value) != 0
    }
    if config.native:
        return not active
    return bool(active) and set(active) == {"no native batch width profile"} and all(
        value > 0 for value in active.values()
    )


def _latency_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for name in sorted(set(before) | set(after)):
        prior = before.get(name)
        current = after.get(name)
        prior_samples = list(prior.get("samples", ())) if isinstance(prior, Mapping) else []
        current_samples = list(current.get("samples", ())) if isinstance(current, Mapping) else []
        result[str(name)] = [float(value) for value in current_samples[len(prior_samples) :]]
    return result


def _prompt_rows(
    tokenizer: Any,
    rows: int,
    prompt_length: int,
    token_id: int,
) -> tuple[dict[str, Any], ...]:
    """Build exact-roundtrip text prompts for the streaming API."""

    result: list[dict[str, Any]] = []
    for row in range(int(rows)):
        token_ids = tuple(
            [int(token_id)] * max(0, int(prompt_length) - 1)
            + [int(token_id) + (row % 4)]
        )
        text = str(tokenizer.decode(list(token_ids)))
        encoded = tokenizer.encode(text)
        roundtrip = tuple(int(token) for token in getattr(encoded, "ids", encoded))
        if roundtrip != token_ids:
            raise RuntimeError(
                f"prompt row {row} failed exact tokenizer roundtrip: "
                f"expected={len(token_ids)} observed={len(roundtrip)}"
            )
        result.append(
            {
                "row_index": int(row),
                "token_ids": token_ids,
                "token_ids_sha256": token_ids_sha256(token_ids),
                "text": text,
                "roundtrip_exact": True,
            }
        )
    return tuple(result)


def _parse_sse_line(line: str | bytes) -> Any | None:
    text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else str(line)
    if not text.startswith("data: "):
        return None
    data = text[6:]
    return "[DONE]" if data == "[DONE]" else json.loads(data)


def _observed_native_widths(exact_rows: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    widths: set[int] = set()
    for exact_row in exact_rows:
        route = exact_row.get("route")
        if not isinstance(route, Mapping):
            continue
        for item in route.get("scheduler_chunks", ()):
            if not isinstance(item, Mapping):
                continue
            chunk = item.get("chunk")
            telemetry = chunk.get("telemetry") if isinstance(chunk, Mapping) else None
            if not isinstance(telemetry, Mapping):
                continue
            decode_state = telemetry.get("decode_state")
            if not isinstance(decode_state, Mapping) or decode_state.get("native_caware_decode") is not True:
                continue
            diagnostics = telemetry.get("diagnostics")
            plan = diagnostics.get("last_width_plan") if isinstance(diagnostics, Mapping) else None
            if not isinstance(plan, Mapping):
                continue
            for group in plan.get("groups", ()):
                if isinstance(group, Mapping) and group.get("mode") == "native":
                    width = int(group.get("width", 0))
                    if width > 1:
                        widths.add(width)
    return tuple(sorted(widths))


def _stream_one(
    client: TestClient,
    *,
    served_model: str,
    row_index: int,
    prompt: str,
    max_tokens: int,
    gate: threading.Barrier | threading.Event,
) -> StreamTrace:
    gate.wait(timeout=60.0)
    started = time.perf_counter()
    request_id: int | None = None
    delta_times: list[float] = []
    pieces: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    done = False
    status = 0
    error: str | None = None
    try:
        with client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": str(served_model),
                "prompt": str(prompt),
                "max_tokens": int(max_tokens),
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
                "kv_storage": "bf16",
                "stream": True,
                "stream_options": {"include_hipengine": True, "include_usage": True},
            },
        ) as response:
            status = int(response.status_code)
            if status != 200:
                error = response.read().decode("utf-8", errors="replace")
            else:
                for line in response.iter_lines():
                    observed = time.perf_counter()
                    payload = _parse_sse_line(line)
                    if payload is None:
                        continue
                    if payload == "[DONE]":
                        done = True
                        continue
                    if not isinstance(payload, Mapping):
                        continue
                    if isinstance(payload.get("usage"), Mapping):
                        usage = copy.deepcopy(dict(payload["usage"]))
                    choices = payload.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, Mapping):
                        continue
                    choice_hip = choice.get("hipengine")
                    if isinstance(choice_hip, Mapping):
                        decode_state = choice_hip.get("decode_state")
                        if isinstance(decode_state, Mapping) and decode_state.get("request_id") is not None:
                            observed_id = int(decode_state["request_id"])
                            if request_id is not None and request_id != observed_id:
                                raise RuntimeError(
                                    f"request id changed during SSE {request_id}->{observed_id}"
                                )
                            request_id = observed_id
                    if choice.get("finish_reason") is None:
                        delta_times.append(observed)
                        pieces.append(str(choice.get("text", "")))
                    else:
                        finish_reason = str(choice.get("finish_reason"))
    except Exception as exc:  # pragma: no cover - retained in hardware artifact
        error = f"{type(exc).__name__}: {exc}"
    completed = time.perf_counter()
    return StreamTrace(
        row_index=int(row_index),
        status_code=status,
        request_id=request_id,
        started_at=started,
        first_delta_at=delta_times[0] if delta_times else None,
        completed_at=completed,
        delta_times=delta_times,
        text="".join(pieces),
        usage=usage,
        finish_reason=finish_reason,
        done=done,
        error=error,
    )


def _memory_snapshot(owner: Any) -> dict[str, Any]:
    session = owner._session
    if session is None:
        return {"tracked": memory_stats(), "hip": None, "owner": owner.observability_snapshot()["kv"]}
    free_bytes, total_bytes = session.runtime.mem_get_info()
    return {
        "tracked": memory_stats(),
        "hip": {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "used_bytes": int(total_bytes - free_bytes),
        },
        "owner": copy.deepcopy(owner.observability_snapshot()["kv"]),
    }


@contextmanager
def _temporary_environment(updates: Mapping[str, str]) -> Iterator[None]:
    prior = {str(key): os.environ.get(str(key)) for key in updates}
    os.environ.update({str(key): str(value) for key, value in updates.items()})
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _wait_for_live_trigger(
    llm: LLM,
    *,
    initial_rows: int,
    native_calls_before: int,
    native_steps: int,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        snapshot = llm.live_loop_snapshot()
        counts = snapshot["runner"]["routes"]["counts"]
        if (
            int(snapshot["loop"]["requests"]["active"]) >= int(initial_rows)
            and int(counts["native_group_calls"]) >= int(native_calls_before) + int(native_steps)
        ):
            return copy.deepcopy(snapshot)
        time.sleep(0.001)
    raise TimeoutError("live admission trigger did not observe the requested native decode progress")


def _run_sample(
    *,
    client: TestClient,
    llm: LLM,
    owner: Any,
    config: Configuration,
    prompts: Sequence[Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    captured: Mapping[int, ReclaimedRow],
    max_tokens: int,
    run_index: int,
    measured: bool,
    live_initial_rows: int | None = None,
    live_trigger_native_steps: int = 8,
) -> dict[str, Any]:
    rows = int(config.rows)
    before = llm.live_loop_snapshot()
    before_ids = set(captured)
    before_counts = before["runner"]["routes"]["counts"]
    before_fallbacks = before["runner"]["routes"]["fallback_reasons"]
    before_latency = before["loop"]["latency_seconds"]
    memory_before = _memory_snapshot(owner)
    traces: list[StreamTrace]
    live_trigger: dict[str, Any] | None = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=rows) as executor:
        if live_initial_rows is None:
            gate = threading.Barrier(rows + 1)
            futures = [
                executor.submit(
                    _stream_one,
                    client,
                    served_model="paro-live",
                    row_index=index,
                    prompt=str(prompts[index]["text"]),
                    max_tokens=max_tokens,
                    gate=gate,
                )
                for index in range(rows)
            ]
            gate.wait(timeout=60.0)
        else:
            initial = int(live_initial_rows)
            initial_gate = threading.Barrier(initial + 1)
            tail_gate = threading.Event()
            futures = [
                executor.submit(
                    _stream_one,
                    client,
                    served_model="paro-live",
                    row_index=index,
                    prompt=str(prompts[index]["text"]),
                    max_tokens=max_tokens,
                    gate=initial_gate if index < initial else tail_gate,
                )
                for index in range(rows)
            ]
            initial_gate.wait(timeout=60.0)
            live_trigger = _wait_for_live_trigger(
                llm,
                initial_rows=initial,
                native_calls_before=int(before_counts["native_group_calls"]),
                native_steps=int(live_trigger_native_steps),
                timeout=120.0,
            )
            tail_gate.set()
        traces = [future.result() for future in futures]
    after = llm.live_loop_snapshot()
    memory_after = _memory_snapshot(owner)
    new_ids = sorted(set(captured) - before_ids)
    trace_ids = [trace.request_id for trace in traces if trace.request_id is not None]
    exact_rows: list[dict[str, Any]] = []
    for trace in sorted(traces, key=lambda item: item.row_index):
        reclaimed = None if trace.request_id is None else captured.get(int(trace.request_id))
        prompt = prompts[trace.row_index]
        prompt_ids = [int(token) for token in prompt["token_ids"]]
        reference = references[str(prompt["token_ids_sha256"])]
        generated = [] if reclaimed is None else list(reclaimed.generated_ids)
        exact_rows.append(
            {
                "row_index": int(trace.row_index),
                "request_id": trace.request_id,
                "prompt_ids_sha256": str(prompt["token_ids_sha256"]),
                "prompt_exact": reclaimed is not None and reclaimed.prompt_ids == prompt_ids,
                "generated_ids": generated,
                "expected_ids": list(reference["generated_ids"]),
                "generated_exact": generated == list(reference["generated_ids"]),
                "text_exact": trace.text == str(reference["text"]),
                "route": None if reclaimed is None else copy.deepcopy(reclaimed.route),
            }
        )
    route_delta = _counter_delta(before_counts, after["runner"]["routes"]["counts"])
    fallback_delta = _counter_delta(
        before_fallbacks,
        after["runner"]["routes"]["fallback_reasons"],
    )
    starts = [trace.started_at for trace in traces]
    ends = [trace.completed_at for trace in traces]
    wall = max(ends) - min(starts)
    generated_tokens = sum(len(row["generated_ids"]) for row in exact_rows)
    http_ok = all(
        trace.status_code == 200
        and trace.error is None
        and trace.done
        and trace.request_id is not None
        and len(trace.delta_times) == int(max_tokens)
        and trace.finish_reason == "length"
        and isinstance(trace.usage, Mapping)
        and int(trace.usage.get("prompt_tokens", -1))
        == len(prompts[trace.row_index]["token_ids"])
        and int(trace.usage.get("completion_tokens", -1)) == int(max_tokens)
        for trace in traces
    )
    ownership_ok = bool(
        len(trace_ids) == rows
        and len(set(trace_ids)) == rows
        and set(trace_ids) == set(new_ids)
        and int(after["loop"]["requests"]["active"]) == 0
        and int(after["loop"]["requests"]["pending"]) == 0
        and int(after["runner"]["model_runner"]["active_requests"]) == 0
    )
    last_plan = after["runner"]["routes"]["last_width_plan"]
    observed_native_widths = _observed_native_widths(exact_rows)
    native_steps_by_row = [
        int(row["route"].get("native_decode_steps", 0))
        for row in exact_rows
        if isinstance(row.get("route"), Mapping)
    ]
    serial_steps_by_row = [
        int(row["route"].get("serial_decode_steps", 0))
        for row in exact_rows
        if isinstance(row.get("route"), Mapping)
    ]
    if config.native and rows > 1:
        required_native_widths = {rows}
        if live_initial_rows is not None and int(live_initial_rows) > 1:
            required_native_widths.add(int(live_initial_rows))
        route_ok = bool(
            int(route_delta["native_group_calls"]) > 0
            and required_native_widths.issubset(observed_native_widths)
            and len(native_steps_by_row) == rows
            and all(value > 0 for value in native_steps_by_row)
        )
    elif config.native:
        route_ok = bool(
            int(route_delta["native_group_calls"]) == 0
            and int(route_delta["serial_row_calls"]) > 0
            and native_steps_by_row == [0]
        )
    else:
        route_ok = bool(
            int(route_delta["native_group_calls"]) == 0
            and int(route_delta["serial_row_calls"]) > 0
            and len(native_steps_by_row) == rows
            and all(value == 0 for value in native_steps_by_row)
            and all(value > 0 for value in serial_steps_by_row)
        )
    fallback_reasons_ok = _fallback_reasons_ok(config, fallback_delta)
    latency = _latency_delta(before_latency, after["loop"]["latency_seconds"])
    passed = bool(
        http_ok
        and ownership_ok
        and route_ok
        and fallback_reasons_ok
        and all(row["prompt_exact"] and row["generated_exact"] and row["text_exact"] for row in exact_rows)
        and generated_tokens == rows * int(max_tokens)
    )
    return {
        "configuration": config.name,
        "run_index": int(run_index),
        "measured": bool(measured),
        "live_admission": live_initial_rows is not None,
        "passed": passed,
        "workload": {
            "rows": rows,
            "prompt_tokens_per_request": len(prompts[0]["token_ids"]),
            "generated_tokens_per_request": int(max_tokens),
            "live_initial_rows": live_initial_rows,
        },
        "throughput": {
            "wall_seconds": wall,
            "aggregate_generated_tok_s": generated_tokens / wall,
            "per_request_generated_tok_s": generated_tokens / wall / rows,
        },
        "correctness": {
            "http_ok": http_ok,
            "ownership_ok": ownership_ok,
            "exact_rows": exact_rows,
            "generated_tokens": generated_tokens,
        },
        "route": {
            "native_requested": bool(config.native),
            "passed": route_ok,
            "counts_delta": route_delta,
            "fallback_reasons_delta": fallback_delta,
            "fallback_reasons_passed": fallback_reasons_ok,
            "observed_native_widths": list(observed_native_widths),
            "native_decode_steps_by_row": native_steps_by_row,
            "serial_decode_steps_by_row": serial_steps_by_row,
            "truthful_edge_serial_steps": sum(serial_steps_by_row),
            "last_width_plan": copy.deepcopy(last_plan),
            "last_execution_manifest": copy.deepcopy(
                after["runner"]["routes"]["last_execution_manifest"]
            ),
        },
        "latency": {
            "scheduler": {name: _stats(values) for name, values in latency.items()},
            "client_ttft_seconds": _stats(
                [
                    trace.first_delta_at - trace.started_at
                    for trace in traces
                    if trace.first_delta_at is not None
                ]
            ),
            "client_inter_delta_seconds": _stats(
                [
                    current - prior
                    for trace in traces
                    for prior, current in zip(trace.delta_times, trace.delta_times[1:])
                ]
            ),
            "client_note": "TestClient may buffer ASGI chunks; scheduler latency is authoritative",
        },
        "http": [asdict(trace) for trace in traces],
        "memory": {"before": memory_before, "after": memory_after},
        "live_trigger": live_trigger,
        "final_snapshot": after,
    }


def _run_configuration(args: argparse.Namespace, config: Configuration) -> dict[str, Any]:
    env = {
        "HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS": "1",
        "HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE": "1" if config.native else "0",
        "HIPENGINE_PREFILL_DECODE_POLICY": "protect_ttft",
        "HIPENGINE_MAX_PREFILL_CHUNK_TOKENS": str(args.prefill_chunk_size),
    }
    with _temporary_environment(env):
        started = time.perf_counter()
        llm = LLM(
            str(args.model),
            backend=str(args.backend),
            quant=str(args.quant),
            max_active_requests=int(config.rows),
        )
        params = SamplingParams(
            max_tokens=int(args.decode_tokens),
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            kv_storage="bf16",
        )
        try:
            llm.prepare(max_sequence_length=int(args.max_sequence_length), sampling_params=params)
            adapter = llm._get_text_generator()
            owner = adapter._runner
            session = owner._session
            if session is None or session.tokenizer is None:
                raise RuntimeError("prepared PARO owner did not expose its tokenizer")
            prompts = _prompt_rows(
                session.tokenizer,
                config.rows,
                args.prompt_length,
                args.prompt_token_id,
            )
            references: dict[str, dict[str, Any]] = {}
            for prompt in prompts:
                digest = str(prompt["token_ids_sha256"])
                if digest in references:
                    continue
                prompt_ids = tuple(int(token) for token in prompt["token_ids"])
                output = llm.generate_detailed((prompt_ids,), params)[0]
                references[digest] = {
                    "generated_ids": list(output.generated_token_ids or ()),
                    "text": output.text,
                }
                if len(references[digest]["generated_ids"]) != int(args.decode_tokens):
                    raise RuntimeError("c1 oracle did not return the requested token count")
            captured: dict[int, ReclaimedRow] = {}
            original_reclaim = owner.reclaim

            def capture_reclaim(completed: Any) -> None:
                request_id = int(completed.request_id)
                row = owner._rows.get(request_id)
                prompt_ids = [] if row is None else [int(token) for token in row.prompt_ids]
                generated_ids = (
                    [int(step.token_id) for step in row.generated_steps]
                    if row is not None
                    else [int(token) for token in completed.generated_tokens]
                )
                original_reclaim(completed)
                route = copy.deepcopy(owner._recent_completed_routes[-1])
                captured[request_id] = ReclaimedRow(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    generated_ids=generated_ids,
                    finish_reason=str(completed.finish_reason),
                    route=route,
                )

            owner.reclaim = capture_reclaim
            app = create_app(
                ServerConfig(
                    model=str(args.model),
                    served_model_name="paro-live",
                    eager_load=False,
                    generation_batch_window_ms=float(args.batch_window_ms),
                    max_context_tokens=int(args.max_sequence_length),
                    stream_queue_max_chunks=max(256, int(args.decode_tokens) + 8),
                    metrics="prometheus",
                ),
                llm=llm,
            )
            samples: list[dict[str, Any]] = []
            with TestClient(app) as client:
                for run_index in range(int(args.warmup_runs) + int(args.measured_runs)):
                    measured = run_index >= int(args.warmup_runs)
                    sample = _run_sample(
                        client=client,
                        llm=llm,
                        owner=owner,
                        config=config,
                        prompts=prompts,
                        references=references,
                        captured=captured,
                        max_tokens=int(args.decode_tokens),
                        run_index=run_index,
                        measured=measured,
                    )
                    samples.append(sample)
                    print(
                        f"{config.name} {'measured' if measured else 'warmup'} {run_index}: "
                        f"{sample['throughput']['aggregate_generated_tok_s']:.3f} tok/s "
                        f"passed={sample['passed']}",
                        flush=True,
                    )
                live = None
                if config.name == str(args.live_configuration):
                    live = _run_sample(
                        client=client,
                        llm=llm,
                        owner=owner,
                        config=config,
                        prompts=prompts,
                        references=references,
                        captured=captured,
                        max_tokens=int(args.decode_tokens),
                        run_index=len(samples),
                        measured=True,
                        live_initial_rows=int(args.live_initial_rows),
                        live_trigger_native_steps=int(args.live_trigger_native_steps),
                    )
                    print(
                        f"{config.name} live: {live['throughput']['aggregate_generated_tok_s']:.3f} "
                        f"tok/s passed={live['passed']}",
                        flush=True,
                    )
            measured_samples = [sample for sample in samples if sample["measured"]]
            rates = [sample["throughput"]["aggregate_generated_tok_s"] for sample in measured_samples]
            walls = [sample["throughput"]["wall_seconds"] for sample in measured_samples]
            final = llm.live_loop_snapshot()
            passed = bool(
                samples
                and all(sample["passed"] for sample in samples)
                and (live is None or live["passed"])
                and int(final["loop"]["requests"]["active"]) == 0
                and int(final["runner"]["model_runner"]["active_requests"]) == 0
            )
            return {
                "configuration": asdict(config),
                "passed": passed,
                "elapsed_seconds": time.perf_counter() - started,
                "references": references,
                "samples": samples,
                "summary": {
                    "aggregate_generated_tok_s": _stats(rates),
                    "wall_seconds": _stats(walls),
                },
                "live_admission": live,
                "final_snapshot": final,
                "final_memory": _memory_snapshot(owner),
            }
        finally:
            llm.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model.exists():
        raise ValueError(f"model does not exist: {args.model}")
    rows: dict[str, Any] = {}
    for name in args.configurations:
        rows[name] = _run_configuration(args, CONFIGURATIONS[name])
    passed = all(bool(row["passed"]) for row in rows.values())
    command = [sys.executable, "scripts/paro_live_server_bench.py", *sys.argv[1:]]
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=str(args.backend),
        target_arch=str(args.backend).removeprefix("hip_"),
        model_path=args.model,
        quant=str(args.quant),
        kv_dtype="bf16",
        command=command,
        environment={
            key: os.environ.get(key)
            for key in (
                "HIP_VISIBLE_DEVICES",
                "HIPENGINE_HIP_ARCH",
                "GPU_MAX_HW_QUEUES",
                "HIPENGINE_COMPILER_VERSION_FILE",
            )
        },
        build_profile="gfx1151_paro_resident_openai_sse_native_and_serial",
        timing_protocol=(
            "one fixed-capacity prepared model per width; exact-roundtrip text prompts; "
            "concurrent OpenAI SSE; resident reclaim token-ID oracle; client cycle wall"
        ),
        warmups=int(args.warmup_runs),
        repetitions=int(args.measured_runs),
        profiler={"used": False, "reason": "server cycle-wall and scheduler-latency packet"},
    )
    payload = {
        "schema": 1,
        "kind": "gfx1151_paro_resident_openai_sse_concurrency_packet",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "measurement_complete" if passed else "failed_gate",
        "passed": passed,
        "performance_claim": passed,
        "performance_claim_scope": (
            "in-process real FastAPI /v1/completions SSE cycle wall; p512/d128; "
            "generated-token numerator; greedy W4 PARO/BF16-KV; no MTP"
        ),
        "provenance": provenance,
        "workload": {
            "prompt_tokens": int(args.prompt_length),
            "decode_tokens": int(args.decode_tokens),
            "warmup_runs": int(args.warmup_runs),
            "measured_runs": int(args.measured_runs),
            "configurations": list(args.configurations),
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "batch_window_ms": float(args.batch_window_ms),
            "max_sequence_length": int(args.max_sequence_length),
            "live_configuration": str(args.live_configuration),
            "live_initial_rows": int(args.live_initial_rows),
        },
        "rows": rows,
        "limitations": [
            "TestClient can buffer transport chunks; scheduler TTFT/ITL snapshots are authoritative.",
            "Each configuration owns a separately prepared fixed-capacity model session.",
            "The serial c8 row is an exact same-loop control, not a no-concurrency server baseline.",
        ],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--quant", default="w4_paro")
    parser.add_argument(
        "--configurations",
        type=_parse_configurations,
        default=DEFAULT_CONFIGURATIONS,
    )
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--batch-window-ms", type=float, default=20.0)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--live-configuration", default="native_c8")
    parser.add_argument("--live-initial-rows", type=int, default=4)
    parser.add_argument("--live-trigger-native-steps", type=int, default=8)
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run(args)
    except Exception as exc:
        failure = {
            "schema": 1,
            "kind": "gfx1151_paro_resident_openai_sse_concurrency_packet",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed_exception",
            "passed": False,
            "performance_claim": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"FINAL status={payload['status']} passed={payload['passed']} artifact={args.json}")
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
