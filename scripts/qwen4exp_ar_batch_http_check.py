#!/usr/bin/env python3
"""Qualify the opt-in Flash-Next AR batch path over live HTTP, not performance.

Run against an already running server started with ``HIPENGINE_QWEN4_EXP_AR_BATCH=1``.
Every phase asserts a lifecycle contract and records the server's own decode
telemetry before and after, so a reported physical batch width is the owner's
step histogram rather than the number of HTTP requests in flight.

This is a bounded functional/lifecycle check. It makes no speed, quality, or
admission claim, and it does not replace the production numerical suite.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

import httpx

from qwen4exp_http_grid import compact, request, response_body

MARKERS = ("ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT")
FILLER = (
    "Archive entry: the library opens at nine and closes at five. This entry "
    "contains no access code.\n"
)


def chat_payload(model: str, prompt: str, **overrides) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "enable_thinking": False,
        "max_tokens": 32,
    }
    payload.update(overrides)
    return payload


async def get_json(client: httpx.AsyncClient, path: str) -> dict:
    response = await client.get(path)
    response.raise_for_status()
    return response.json()


async def wait_idle(client: httpx.AsyncClient, timeout_s: float = 120.0) -> dict:
    """Wait until the server reports no queued or active generation work."""

    deadline = time.perf_counter() + timeout_s
    last = {}
    while time.perf_counter() < deadline:
        response = await client.get("/ready")
        try:
            last = response.json()
        except ValueError:
            last = {"unparsed_status": response.status_code}
        queue = last.get("queue") or {}
        if int(queue.get("depth") or 0) == 0 and int(queue.get("active_requests") or 0) == 0:
            return last
        await asyncio.sleep(0.05)
    raise TimeoutError(f"server never went idle; last queue {last.get('queue')}")


def require_telemetry(body: object) -> dict:
    """Return the decode object, rejecting unavailable or malformed bodies.

    A phase must not pass because a missing ``ar_batch`` made every width
    lookup return zero; an absent or unusable payload is a harness failure.
    """

    if not isinstance(body, dict):
        raise RuntimeError(
            f"decode telemetry is not a JSON object: {type(body).__name__}"
        )
    if body.get("available") is not True:
        raise RuntimeError(f"decode telemetry unavailable: {body}")
    decode = body.get("decode")
    if not isinstance(decode, dict):
        raise RuntimeError(f"decode telemetry has no decode object: {body}")
    batch = decode.get("ar_batch")
    if batch is not None and not isinstance(batch, dict):
        raise RuntimeError(f"decode telemetry ar_batch is not an object: {batch!r}")
    return decode


async def telemetry(client: httpx.AsyncClient) -> dict:
    return require_telemetry(await get_json(client, "/v1/hipengine/decode"))


def batch_state(decode: dict) -> dict:
    return decode.get("ar_batch") or {}


def widths(decode: dict) -> dict[str, int]:
    return dict(batch_state(decode).get("step_widths") or {})


def width_delta(before: dict, after: dict) -> dict[str, int]:
    old, new = widths(before), widths(after)
    return {key: int(new.get(key, 0)) - int(old.get(key, 0)) for key in new}


async def poll_ready(client: httpx.AsyncClient, stop: asyncio.Event, samples: list[dict]) -> None:
    """Sample queue depth while a phase runs, so queueing is observed not inferred."""

    started = time.perf_counter()
    while not stop.is_set():
        try:
            body = await get_json(client, "/ready")
        except Exception:
            body = {}
        queue = body.get("queue") or {}
        samples.append(
            {
                "elapsed_s": round(time.perf_counter() - started, 4),
                "depth": queue.get("depth"),
                "active_requests": queue.get("active_requests"),
                "ready": body.get("ready"),
            }
        )
        await asyncio.sleep(0.05)


async def synchronized(client, out, specs, *, poll: bool = False):
    """Start every request on one barrier so arrivals overlap by construction."""

    barrier = asyncio.Event()

    async def run(spec):
        await barrier.wait()
        return await request(client, out, spec["name"], spec["payload"])

    tasks = [asyncio.create_task(run(spec)) for spec in specs]
    samples: list[dict] = []
    stop = asyncio.Event()
    poller = asyncio.create_task(poll_ready(client, stop, samples)) if poll else None
    await asyncio.sleep(0.1)
    barrier.set()
    rows = await asyncio.gather(*tasks)
    if poller is not None:
        stop.set()
        await poller
    return rows, samples


def stream_text(row: dict) -> str:
    """Assemble a streamed completion; the server splits tokens across chunks."""

    pieces: list[str] = []
    for event in row.get("events") or []:
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        for choice in data.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if isinstance(content, str):
                pieces.append(content)
    return "".join(pieces)


def forced(payload: dict, tokens: int) -> dict:
    """Pin the decode length so a short answer cannot hide a width-2 overlap.

    Without this the model answers ``ALPHA`` in two ticks and two requests
    serialize; the width-2 contract would then be untestable rather than false.
    """

    return payload | {"max_tokens": int(tokens), "ignore_eos": True}


def completion_text(row: dict) -> str:
    choices = (row.get("choices") or [{}])
    message = (choices[0] or {}).get("message") or {}
    return str(message.get("content") or "")


def check(checks: list[dict], name: str, passed: bool, **details) -> bool:
    checks.append({"name": name, "passed": bool(passed), **details})
    print(json.dumps({"check": name, "passed": bool(passed), **details}), flush=True)
    return passed


def terminal_stream_checks(row: dict, checks: list[dict], name: str) -> None:
    events = row.get("events") or []
    terminal = [event for event in events if event.get("data") == "[DONE]"]
    usage_events = [
        event for event in events
        if isinstance(event.get("data"), dict) and event["data"].get("usage")
    ]
    finish = [
        event for event in events
        if isinstance(event.get("data"), dict)
        and any(choice.get("finish_reason") for choice in event["data"].get("choices") or [])
    ]
    check(checks, f"{name}.http_200", row.get("status") == 200, status=row.get("status"))
    check(checks, f"{name}.sse_done", len(terminal) == 1, done_events=len(terminal))
    check(checks, f"{name}.usage_terminal", len(usage_events) == 1,
          usage=(usage_events[0]["data"].get("usage") if usage_events else None))
    check(checks, f"{name}.finish_reason", len(finish) >= 1,
          finish_reasons=[choice.get("finish_reason") for event in finish
                          for choice in event["data"].get("choices") or []])
    check(checks, f"{name}.event_count", len(events) >= 3, events=len(events))
    check(checks, f"{name}.no_transport_error", "transport_error" not in row,
          transport_error=row.get("transport_error"))


async def stream_with_cancel(
    client: httpx.AsyncClient,
    out: Path,
    name: str,
    payload: dict,
    *,
    cancel_after_events: int,
) -> dict:
    """Open an SSE stream, read a few events, then drop the connection."""

    started = time.perf_counter()
    row = {
        "name": name,
        "endpoint": "/v1/chat/completions",
        "request": payload,
        "started_unix": time.time(),
        "events": [],
        "cancelled": True,
    }
    try:
        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            row["status"] = response.status_code
            if response.status_code != 200:
                await response.aread()
                row["body"] = response_body(response)
            else:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        row["events"].append(
                            {
                                "elapsed_s": time.perf_counter() - started,
                                "data": data if data == "[DONE]" else json.loads(data),
                            }
                        )
                        if len(row["events"]) >= cancel_after_events:
                            break
    except Exception as exc:
        row["transport_error"] = f"{type(exc).__name__}: {exc}"
    row["cancel_unix"] = time.time()
    row["wall_s"] = time.perf_counter() - started
    (out / f"{name}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"name": name, "status": row.get("status"),
                      "cancelled_after_events": len(row["events"]),
                      "wall_s": row["wall_s"]}), flush=True)
    return compact(row) | {"cancel_unix": row["cancel_unix"], "event_count": len(row["events"])}


async def run(args) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    report = {
        "schema": 1,
        "kind": "qwen4exp_ar_batch_http_check",
        "performance_claim": False,
        "numerical_gate": "not_run",
        "decode_length_forced": True,
        "protocol": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "phases": [],
        "checks": checks,
    }

    def save() -> None:
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )

    async with httpx.AsyncClient(base_url=args.url, timeout=600) as client:
        ready = await wait_idle(client)
        report["ready_before"] = ready
        capabilities = await get_json(client, "/v1/hipengine/capabilities")
        report["capabilities_model"] = capabilities.get("model")
        report["capabilities_sha256"] = hashlib.sha256(
            json.dumps(capabilities, sort_keys=True).encode()
        ).hexdigest()
        before = await telemetry(client)
        report["telemetry_before"] = before
        batch = batch_state(before)
        check(checks, "preflight.ready", bool(ready.get("ready")),
              status=ready.get("status"), diagnostics=ready.get("diagnostics"))
        check(checks, "preflight.decode_path",
              str(before.get("decode_path")) == "qwen4exp_full_model_ar_batch",
              decode_path=before.get("decode_path"))
        check(checks, "preflight.state_slots", int(batch.get("state_slots") or 0) >= 1,
              state_slots=batch.get("state_slots"))
        check(checks, "preflight.no_failure", batch.get("failure") is None,
              failure=batch.get("failure"))
        check(checks, "preflight.live_slots_zero", int(batch.get("live_slots") or 0) == 0,
              live_slots=batch.get("live_slots"))
        startup = (ready.get("startup") or {}).get("checks") or {}
        scratch = startup.get("scratch_probe") or {}
        report["startup_scratch_probe"] = scratch
        scratch_decode = (scratch.get("result") or {}).get("decode") or {}
        scratch_batch = scratch_decode.get("ar_batch") or {}
        check(checks, "preflight.scratch_probe_prepared",
              str(scratch.get("status")) == "passed", scratch_probe=scratch.get("status"))
        check(checks, "preflight.scratch_probe_batch_path",
              scratch_decode.get("decode_path") == "qwen4exp_full_model_ar_batch",
              scratch_decode_path=scratch_decode.get("decode_path"))
        check(checks, "preflight.scratch_probe_state_slots",
              int(scratch_batch.get("state_slots") or 0) == int(batch.get("state_slots") or -1),
              scratch_state_slots=scratch_batch.get("state_slots"),
              live_state_slots=batch.get("state_slots"))
        save()

        if args.phases == "c4":
            await c4_phases(client, args, report, checks, save)
            return await finish(client, args, report, checks)

        model = args.model or str((report["capabilities"].get("model") or {}).get("id"))

        # -- C1 blocking + controls -------------------------------------
        phase_start = await telemetry(client)
        row = await request(client, args.output, "c1-block", chat_payload(
            model, "Reply with exactly one word: ALPHA", max_tokens=args.c1_tokens))
        await wait_idle(client)
        after = await telemetry(client)
        text = completion_text(row)
        check(checks, "c1_block.http_200", row.get("status") == 200, status=row.get("status"))
        check(checks, "c1_block.marker", "ALPHA" in text.upper(), text=text[:80])
        check(checks, "c1_block.usage",
              int(((row.get("usage") or {}).get("completion_tokens")) or 0) > 0,
              usage=row.get("usage"))
        check(checks, "c1_block.hipengine_metadata", isinstance(row.get("hipengine"), dict),
              hipengine_keys=sorted((row.get("hipengine") or {})))
        check(checks, "c1_block.width_1", width_delta(phase_start, after).get("1", 0) > 0,
              width_delta=width_delta(phase_start, after))
        report["phases"].append({
            "name": "c1_block", "row": row,
            "width_delta": width_delta(phase_start, after),
            "telemetry_after": after,
        })
        save()

        phase_start = await telemetry(client)
        stopped = await request(client, args.output, "c1-stop", chat_payload(
            model, "Repeat this exact sentence: alpha beta gamma delta epsilon.",
            max_tokens=32, stop=[" beta gamma"]))
        logprobs = await request(client, args.output, "c1-logprobs", chat_payload(
            model, "Reply with exactly one word: BRAVO", max_tokens=8,
            logprobs=True, top_logprobs=2))
        await wait_idle(client)
        stop_choice = (stopped.get("choices") or [{}])[0]
        logprob_content = ((logprobs.get("choices") or [{}])[0].get("logprobs") or {}).get("content")
        check(checks, "controls.stop_http_200", stopped.get("status") == 200,
              status=stopped.get("status"))
        check(checks, "controls.stop_finish_reason",
              stop_choice.get("finish_reason") == "stop",
              finish_reason=stop_choice.get("finish_reason"))
        check(checks, "controls.stop_text_excludes_sequence",
              "beta gamma" not in completion_text(stopped),
              text=completion_text(stopped)[:80])
        check(checks, "controls.logprobs_http_200", logprobs.get("status") == 200,
              status=logprobs.get("status"))
        check(checks, "controls.logprobs_present",
              isinstance(logprob_content, list) and len(logprob_content) > 0,
              tokens=len(logprob_content or []))
        report["phases"].append({
            "name": "c1_controls",
            "rows": [stopped, logprobs],
            "width_delta": width_delta(phase_start, await telemetry(client)),
        })
        save()

        # -- C1 streaming -----------------------------------------------
        phase_start = await telemetry(client)
        streamed = await request(client, args.output, "c1-stream", chat_payload(
            model, "Reply with exactly one word: CHARLIE", max_tokens=args.c1_tokens,
            stream=True,
            stream_options={"include_usage": True, "include_hipengine": True}))
        await wait_idle(client)
        terminal_stream_checks(streamed, checks, "c1_stream")
        check(checks, "c1_stream.marker", "CHARLIE" in stream_text(streamed).upper(),
              text=stream_text(streamed)[:80],
              event_count=streamed.get("event_count"))
        report["phases"].append({
            "name": "c1_stream", "row": streamed,
            "width_delta": width_delta(phase_start, await telemetry(client)),
        })
        save()

        # -- C2 concurrent, distinct requests ---------------------------
        phase_start = await telemetry(client)
        specs = [
            {"name": f"c2-block-{index}", "payload": forced(chat_payload(
                model, f"Reply with exactly one word: {MARKERS[index]}"), args.c2_tokens)}
            for index in range(2)
        ]
        rows, samples = await synchronized(client, args.output, specs, poll=True)
        await wait_idle(client)
        after = await telemetry(client)
        delta = width_delta(phase_start, after)
        texts = [completion_text(row).upper() for row in rows]
        check(checks, "c2_block.all_200", all(row.get("status") == 200 for row in rows),
              statuses=[row.get("status") for row in rows])
        check(checks, "c2_block.distinct_markers",
              all(MARKERS[index] in texts[index] for index in range(2))
              and texts[0] != texts[1],
              texts=[text[:40] for text in texts])
        check(checks, "c2_block.physical_width_2", delta.get("2", 0) > 0, width_delta=delta)
        check(checks, "c2_block.no_width_above_capacity",
              max((int(key) for key in delta), default=0) <= int(batch.get("state_slots") or 2),
              width_delta=delta, state_slots=batch.get("state_slots"))
        check(checks, "c2_block.imports", int((batch_state(after).get("imports") or 0))
              - int((batch_state(phase_start).get("imports") or 0)) == 2,
              imports_before=batch_state(phase_start).get("imports"),
              imports_after=batch_state(after).get("imports"))
        report["phases"].append({
            "name": "c2_block", "rows": rows, "width_delta": delta,
            "ready_samples": samples, "telemetry_after": after,
        })
        save()

        phase_start = await telemetry(client)
        specs = [
            {"name": f"c2-stream-{index}", "payload": forced(chat_payload(
                model, f"Reply with exactly one word: {MARKERS[index + 2]}",
                stream=True,
                stream_options={"include_usage": True, "include_hipengine": True}),
                args.c2_tokens)}
            for index in range(2)
        ]
        rows, _ = await synchronized(client, args.output, specs)
        await wait_idle(client)
        after = await telemetry(client)
        for index, row in enumerate(rows):
            terminal_stream_checks(row, checks, f"c2_stream_{index}")
        texts = [stream_text(row).upper() for row in rows]
        check(checks, "c2_stream.distinct_markers",
              all(MARKERS[index + 2] in texts[index] for index in range(2))
              and texts[0] != texts[1],
              texts=[text[:40] for text in texts],
              markers=[MARKERS[2], MARKERS[3]])
        delta = width_delta(phase_start, after)
        check(checks, "c2_stream.physical_width_2", delta.get("2", 0) > 0, width_delta=delta)
        report["phases"].append({
            "name": "c2_stream", "rows": rows, "width_delta": delta,
            "telemetry_after": after,
        })
        save()

        # -- C2 mixed prompt lengths ------------------------------------
        phase_start = await telemetry(client)
        long_prompt = (
            "Read this archive and reply with exactly one word: DELTA\n"
            + FILLER * args.filler_lines
            + "The access code is DELTA.\n"
        )
        specs = [
            {"name": "c2-mixed-long", "payload": forced(chat_payload(
                model, long_prompt), args.c2_tokens)},
            {"name": "c2-mixed-short", "payload": forced(chat_payload(
                model, "Reply with exactly one word: ECHO"), args.c2_tokens)},
        ]
        rows, samples = await synchronized(client, args.output, specs, poll=True)
        await wait_idle(client)
        after = await telemetry(client)
        delta = width_delta(phase_start, after)
        check(checks, "c2_mixed.all_200", all(row.get("status") == 200 for row in rows),
              statuses=[row.get("status") for row in rows])
        check(checks, "c2_mixed.physical_width_2", delta.get("2", 0) > 0, width_delta=delta)
        check(checks, "c2_mixed.prompt_token_accounting",
              all(int(((row.get("usage") or {}).get("prompt_tokens")) or 0) > 0 for row in rows),
              prompt_tokens=[(row.get("usage") or {}).get("prompt_tokens") for row in rows])
        report["phases"].append({
            "name": "c2_mixed", "rows": rows, "width_delta": delta,
            "ready_samples": samples, "telemetry_after": after,
        })
        save()

        # -- C4 arrivals over a C2 capacity: queue and admission reuse ---
        phase_start = await telemetry(client)
        specs = [
            {"name": f"c4-queue-{index}", "payload": forced(chat_payload(
                model, f"Reply with exactly one word: {MARKERS[index % len(MARKERS)]}"),
                args.c4_tokens)}
            for index in range(4)
        ]
        rows, samples = await synchronized(client, args.output, specs, poll=True)
        await wait_idle(client)
        after = await telemetry(client)
        delta = width_delta(phase_start, after)
        observed_depth = max((int(sample.get("depth") or 0) for sample in samples), default=0)
        check(checks, "c4_queue.all_200", all(row.get("status") == 200 for row in rows),
              statuses=[row.get("status") for row in rows])
        check(checks, "c4_queue.observed_queueing", observed_depth > 0,
              max_depth=observed_depth, samples=len(samples))
        check(checks, "c4_queue.no_physical_width_4",
              int(delta.get("4", 0)) == 0, width_delta=delta)
        check(checks, "c4_queue.admission_reuse",
              int((batch_state(after).get("imports") or 0))
              - int(batch.get("imports") or 0) >= 4,
              imports=batch_state(after).get("imports"))
        report["phases"].append({
            "name": "c4_queue", "rows": rows, "width_delta": delta,
            "max_observed_queue_depth": observed_depth,
            "ready_samples": samples, "telemetry_after": after,
        })
        save()

        # -- Cancellation with a surviving peer -------------------------
        phase_start = await telemetry(client)
        peer_task = asyncio.create_task(request(client, args.output, "cancel-peer", forced(
            chat_payload(model, "Reply with exactly one word: FOXTROT",
                         stream=True,
                         stream_options={"include_usage": True, "include_hipengine": True}),
            args.cancel_peer_tokens)))
        await asyncio.sleep(0.15)
        victim = await stream_with_cancel(
            client, args.output, "cancel-victim",
            forced(chat_payload(model, "Count slowly from one to two hundred.",
                                stream=True,
                                stream_options={"include_usage": True, "include_hipengine": True}),
                   args.cancel_victim_tokens),
            cancel_after_events=args.cancel_after_events,
        )
        peer = await peer_task
        await wait_idle(client)
        after = await telemetry(client)
        peer_events = peer.get("events") or []
        peer_last = max((event.get("elapsed_s") or 0 for event in peer_events), default=0)
        peer_last_abs = float(peer.get("started_unix") or 0.0) + float(peer_last)
        peer_text = stream_text(peer).upper()
        check(checks, "cancel.victim_started", int(victim.get("event_count") or 0) >= args.cancel_after_events,
              events=victim.get("event_count"), status=victim.get("status"))
        check(checks, "cancel.victim_was_streaming", victim.get("status") == 200,
              status=victim.get("status"))
        check(checks, "cancel.peer_http_200", peer.get("status") == 200, status=peer.get("status"))
        check(checks, "cancel.peer_survived", "FOXTROT" in peer_text, event_count=len(peer_events))
        check(checks, "cancel.peer_progress_after_cancel",
              peer_last_abs > float(victim.get("cancel_unix") or 0.0),
              peer_last_event_unix=round(peer_last_abs, 3),
              victim_cancel_unix=round(float(victim.get("cancel_unix") or 0.0), 3))
        check(checks, "cancel.peer_terminal", any(
            isinstance(event.get("data"), dict) and event["data"].get("usage")
            for event in peer_events), event_count=len(peer_events))
        check(checks, "cancel.no_batch_failure", batch_state(after).get("failure") is None,
              failure=batch_state(after).get("failure"))
        check(checks, "cancel.live_slots_zero", int(batch_state(after).get("live_slots") or 0) == 0,
              live_slots=batch_state(after).get("live_slots"))
        report["phases"].append({
            "name": "cancel_peer", "victim": victim, "peer": peer,
            "width_delta": width_delta(phase_start, after), "telemetry_after": after,
        })
        save()

        # -- Post-cancel health and slot reuse --------------------------
        phase_start = await telemetry(client)
        recovery = await request(client, args.output, "post-cancel", chat_payload(
            model, "Reply with exactly one word: ALPHA", max_tokens=args.c1_tokens))
        followup = await request(client, args.output, "post-cancel-2", chat_payload(
            model, "Reply with exactly one word: BRAVO", max_tokens=args.c1_tokens))
        await wait_idle(client)
        after = await telemetry(client)
        ready_after = await get_json(client, "/ready")
        check(checks, "recovery.http_200",
              recovery.get("status") == 200 and followup.get("status") == 200,
              statuses=[recovery.get("status"), followup.get("status")])
        check(checks, "recovery.markers",
              "ALPHA" in completion_text(recovery).upper()
              and "BRAVO" in completion_text(followup).upper(),
              texts=[completion_text(recovery)[:40], completion_text(followup)[:40]])
        check(checks, "recovery.ready", bool(ready_after.get("ready")),
              status=ready_after.get("status"))
        check(checks, "recovery.no_batch_failure", batch_state(after).get("failure") is None,
              failure=batch_state(after).get("failure"))
        check(checks, "recovery.live_slots_zero",
              int(batch_state(after).get("live_slots") or 0) == 0,
              live_slots=batch_state(after).get("live_slots"))
        check(checks, "recovery.release_accounting",
              int(batch_state(after).get("releases") or 0)
              >= int(batch_state(after).get("imports") or 0),
              imports=batch_state(after).get("imports"),
              releases=batch_state(after).get("releases"))
        report["phases"].append({
            "name": "recovery", "rows": [recovery, followup], "ready_after": ready_after,
            "width_delta": width_delta(phase_start, after), "telemetry_after": after,
        })
        save()

        # -- Mixed AR + MTP progress diagnostic (never a physical batch) -
        if args.with_mtp:
            phase_start = await telemetry(client)
            specs = [
                {"name": "mixed-ar", "payload": chat_payload(
                    model, "Reply with exactly one word: ECHO", max_tokens=args.c2_tokens)},
                {"name": "mixed-mtp", "payload": chat_payload(
                    model, "Reply with exactly one word: FOXTROT", max_tokens=args.c2_tokens,
                    speculative_mtp=True)},
            ]
            rows, _ = await synchronized(client, args.output, specs)
            await wait_idle(client)
            after = await telemetry(client)
            delta = width_delta(phase_start, after)
            mtp_summary = (rows[1].get("hipengine") or {}).get("speculative_mtp") or {}
            check(checks, "mixed_mtp.ar_http_200", rows[0].get("status") == 200,
                  status=rows[0].get("status"))
            check(checks, "mixed_mtp.mtp_http_200", rows[1].get("status") == 200,
                  status=rows[1].get("status"))
            check(checks, "mixed_mtp.mtp_route_reported",
                  bool(mtp_summary) and bool(mtp_summary.get("used")),
                  speculative_mtp=mtp_summary)
            check(checks, "mixed_mtp.mtp_cycles",
                  int(mtp_summary.get("draft_cycles") or 0) > 0,
                  draft_cycles=mtp_summary.get("draft_cycles"),
                  note="diagnostic only: MTP stays c1; the width histogram is the AR batch's")
            report["phases"].append({
                "name": "mixed_mtp", "rows": rows, "width_delta": delta,
                "mtp_summary": mtp_summary, "telemetry_after": after,
            })
            save()

        return await finish(client, args, report, checks)


async def finish(client, args, report, checks) -> int:
    """Final health checks plus the shared report summary."""

    final_ready = await wait_idle(client)
    final = await telemetry(client)
    report["ready_after"] = final_ready
    report["telemetry_after"] = final
    check(checks, "final.no_failure", batch_state(final).get("failure") is None,
          failure=batch_state(final).get("failure"))
    check(checks, "final.live_slots_zero",
          int(batch_state(final).get("live_slots") or 0) == 0,
          live_slots=batch_state(final).get("live_slots"))
    check(checks, "final.accepting", bool(final_ready.get("ready")),
          status=final_ready.get("status"))

    failures = [item for item in checks if not item["passed"]]
    report["status"] = "passed" if not failures else "failed"
    report["failure_count"] = len(failures)
    report["check_count"] = len(checks)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"status": report["status"], "checks": len(checks),
                      "failures": len(failures)}), flush=True)
    for item in failures:
        print(json.dumps({"failed_check": item}, ensure_ascii=False), flush=True)
    return 0 if not failures else 1

async def c4_phases(client, args, report, checks, save) -> None:
    """Physical width-4 lifecycle only: block, SSE, cancel one of four, recovery.

    Four arrivals are barrier-synchronized so the width-4 tick exists by
    construction; a queue at capacity 4 is a scheduler bug, not an expected
    phase. The cancel phase keeps three peers generating past the victim's
    disconnect, so the batch must shrink 4 -> 3 and finish without a failure.
    """

    model = args.model
    before = await telemetry(client)
    batch = batch_state(before)
    check(checks, "c4_preflight.state_slots_4",
          int(batch.get("state_slots") or 0) == 4, state_slots=batch.get("state_slots"))
    check(checks, "c4_preflight.live_slots_zero",
          int(batch.get("live_slots") or 0) == 0, live_slots=batch.get("live_slots"))

    # -- four synchronized blocking requests --------------------------
    phase_start = await telemetry(client)
    specs = [
        {"name": f"c4-block-{index}", "payload": forced(chat_payload(
            model, f"Reply with exactly one word: {MARKERS[index]}"), args.c4_tokens)}
        for index in range(4)
    ]
    rows, samples = await synchronized(client, args.output, specs, poll=True)
    await wait_idle(client)
    after = await telemetry(client)
    delta = width_delta(phase_start, after)
    texts = [completion_text(row).upper() for row in rows]
    check(checks, "c4_block.all_200", all(row.get("status") == 200 for row in rows),
          statuses=[row.get("status") for row in rows])
    check(checks, "c4_block.distinct_markers",
          all(MARKERS[index] in texts[index] for index in range(4))
          and len(set(texts)) == 4,
          texts=[text[:40] for text in texts])
    check(checks, "c4_block.physical_width_4", delta.get("4", 0) > 0, width_delta=delta)
    check(checks, "c4_block.no_width_above_capacity",
          max((int(key) for key in delta), default=0) <= 4,
          width_delta=delta, state_slots=batch.get("state_slots"))
    check(checks, "c4_block.imports",
          int(batch_state(after).get("imports") or 0)
          - int(batch_state(phase_start).get("imports") or 0) == 4,
          imports_before=batch_state(phase_start).get("imports"),
          imports_after=batch_state(after).get("imports"))
    check(checks, "c4_block.live_slots_zero",
          int(batch_state(after).get("live_slots") or 0) == 0,
          live_slots=batch_state(after).get("live_slots"))
    report["phases"].append({
        "name": "c4_block", "rows": rows, "width_delta": delta,
        "ready_samples": samples, "telemetry_after": after,
    })
    save()

    # -- four synchronized SSE streams --------------------------------
    phase_start = await telemetry(client)
    specs = [
        {"name": f"c4-stream-{index}", "payload": forced(chat_payload(
            model, f"Reply with exactly one word: {MARKERS[index]}",
            stream=True,
            stream_options={"include_usage": True, "include_hipengine": True}),
            args.c4_tokens)}
        for index in range(4)
    ]
    rows, _ = await synchronized(client, args.output, specs)
    await wait_idle(client)
    after = await telemetry(client)
    for index, row in enumerate(rows):
        terminal_stream_checks(row, checks, f"c4_stream_{index}")
    texts = [stream_text(row).upper() for row in rows]
    check(checks, "c4_stream.distinct_markers",
          all(MARKERS[index] in texts[index] for index in range(4))
          and len(set(texts)) == 4,
          texts=[text[:40] for text in texts])
    delta = width_delta(phase_start, after)
    check(checks, "c4_stream.physical_width_4", delta.get("4", 0) > 0, width_delta=delta)
    check(checks, "c4_stream.no_batch_failure", batch_state(after).get("failure") is None,
          failure=batch_state(after).get("failure"))
    report["phases"].append({
        "name": "c4_stream", "rows": rows, "width_delta": delta,
        "telemetry_after": after,
    })
    save()

    # -- cancel one of four; the other three must keep generating -----
    phase_start = await telemetry(client)
    peer_tasks = [
        asyncio.create_task(request(client, args.output, f"c4-cancel-peer-{index}", forced(
            chat_payload(model, f"Reply with exactly one word: {MARKERS[index]}",
                         stream=True,
                         stream_options={"include_usage": True, "include_hipengine": True}),
            args.cancel_peer_tokens)))
        for index in range(3)
    ]
    await asyncio.sleep(0.15)
    victim = await stream_with_cancel(
        client, args.output, "c4-cancel-victim",
        forced(chat_payload(model, "Count slowly from one to two hundred.",
                            stream=True,
                            stream_options={"include_usage": True, "include_hipengine": True}),
               args.cancel_victim_tokens),
        cancel_after_events=args.cancel_after_events,
    )
    peers = await asyncio.gather(*peer_tasks)
    await wait_idle(client)
    after = await telemetry(client)
    cancel_unix = float(victim.get("cancel_unix") or 0.0)
    for index, peer in enumerate(peers):
        peer_events = peer.get("events") or []
        peer_last = max((event.get("elapsed_s") or 0 for event in peer_events), default=0)
        peer_last_abs = float(peer.get("started_unix") or 0.0) + float(peer_last)
        check(checks, f"c4_cancel.peer_{index}_http_200", peer.get("status") == 200,
              status=peer.get("status"))
        check(checks, f"c4_cancel.peer_{index}_marker",
              MARKERS[index] in stream_text(peer).upper(),
              text=stream_text(peer)[:40])
        check(checks, f"c4_cancel.peer_{index}_progress_after_cancel",
              peer_last_abs > cancel_unix,
              peer_last_event_unix=round(peer_last_abs, 3),
              victim_cancel_unix=round(cancel_unix, 3))
        check(checks, f"c4_cancel.peer_{index}_terminal", any(
            isinstance(event.get("data"), dict) and event["data"].get("usage")
            for event in peer_events), event_count=len(peer_events))
    check(checks, "c4_cancel.victim_started",
          int(victim.get("event_count") or 0) >= args.cancel_after_events,
          events=victim.get("event_count"), status=victim.get("status"))
    check(checks, "c4_cancel.no_batch_failure", batch_state(after).get("failure") is None,
          failure=batch_state(after).get("failure"))
    check(checks, "c4_cancel.live_slots_zero",
          int(batch_state(after).get("live_slots") or 0) == 0,
          live_slots=batch_state(after).get("live_slots"))
    report["phases"].append({
        "name": "c4_cancel", "victim": victim, "peers": peers,
        "width_delta": width_delta(phase_start, after), "telemetry_after": after,
    })
    save()

    # -- post-cancel health and slot reuse ----------------------------
    phase_start = await telemetry(client)
    recovery = await request(client, args.output, "c4-post-cancel", chat_payload(
        model, "Reply with exactly one word: ALPHA", max_tokens=args.c1_tokens))
    followup = await request(client, args.output, "c4-post-cancel-2", chat_payload(
        model, "Reply with exactly one word: BRAVO", max_tokens=args.c1_tokens))
    await wait_idle(client)
    after = await telemetry(client)
    ready_after = await get_json(client, "/ready")
    check(checks, "c4_recovery.http_200",
          recovery.get("status") == 200 and followup.get("status") == 200,
          statuses=[recovery.get("status"), followup.get("status")])
    check(checks, "c4_recovery.markers",
          "ALPHA" in completion_text(recovery).upper()
          and "BRAVO" in completion_text(followup).upper(),
          texts=[completion_text(recovery)[:40], completion_text(followup)[:40]])
    check(checks, "c4_recovery.ready", bool(ready_after.get("ready")),
          status=ready_after.get("status"))
    check(checks, "c4_recovery.no_batch_failure", batch_state(after).get("failure") is None,
          failure=batch_state(after).get("failure"))
    check(checks, "c4_recovery.live_slots_zero",
          int(batch_state(after).get("live_slots") or 0) == 0,
          live_slots=batch_state(after).get("live_slots"))
    check(checks, "c4_recovery.release_accounting",
          int(batch_state(after).get("releases") or 0)
          >= int(batch_state(after).get("imports") or 0),
          imports=batch_state(after).get("imports"),
          releases=batch_state(after).get("releases"))
    report["phases"].append({
        "name": "c4_recovery", "rows": [recovery, followup], "ready_after": ready_after,
        "width_delta": width_delta(phase_start, after), "telemetry_after": after,
    })
    save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18391")
    parser.add_argument("--model", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-command", default="")
    parser.add_argument("--c1-tokens", type=int, default=24)
    parser.add_argument("--c2-tokens", type=int, default=64)
    parser.add_argument("--c4-tokens", type=int, default=64)
    parser.add_argument("--cancel-victim-tokens", type=int, default=512)
    parser.add_argument("--cancel-peer-tokens", type=int, default=48)
    parser.add_argument("--cancel-after-events", type=int, default=8)
    parser.add_argument("--filler-lines", type=int, default=40)
    parser.add_argument(
        "--phases",
        choices=("all", "c4"),
        default="all",
        help="all runs the full lifecycle matrix; c4 runs only the physical width-4 phases",
    )
    parser.add_argument("--with-mtp", action="store_true")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
