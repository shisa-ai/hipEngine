#!/usr/bin/env python3
"""Real-socket public-default AR serving and cancellation qualification."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gguf_mtp_c1c8_server_bench import (
    _generated_ids, _resident_observability, load_prompt_suite, DEFAULT_PROMPTS,
)
from scripts.gguf_p6e_cancel_refill_proof import _free_port, _metrics_values, _wait_for


def stream_summary(events):
    text, finished, done, errors, request_ids = [], False, False, [], set()
    for event in events:
        if event == "[DONE]":
            done = True
            continue
        if "error" in event:
            errors.append(event["error"])
        for choice in event.get("choices", []):
            text.append(choice.get("text", ""))
            finished |= choice.get("finish_reason") is not None
            state = choice.get("hipengine", {}).get("decode_state", {})
            if state.get("request_id") is not None:
                request_ids.add(state["request_id"])
    return {"text": "".join(text), "complete": done and finished and not errors
            and len(request_ids) <= 1, "errors": errors,
            "request_ids": sorted(request_ids)}


def validate_response(body):
    ids = _generated_ids(body)
    accounting = body.get("hipengine", {}).get("token_accounting", {})
    if (accounting.get("total_generated_tokens") != len(ids)
            or body.get("usage", {}).get("completion_tokens") != len(ids)):
        raise ValueError("missing or inconsistent exact generated-token accounting")
    return {"ids": ids, "text": body["choices"][0]["text"],
            "finish_reason": body["choices"][0].get("finish_reason"),
            "hipengine": body.get("hipengine", {})}


def payload(prompt, count, *, stream=False):
    result = {"model": "qwen38-closure", "prompt": list(prompt), "max_tokens": count,
              "temperature": 0.0, "top_p": 1.0, "ignore_eos": True,
              "stream": stream, "speculative_mtp": False}
    if stream:
        result["stream_options"] = {"include_hipengine": True, "include_usage": True}
    return result


def request(base_url, body, *, barrier=None, first_token=None, cancel=False):
    import httpx

    with httpx.Client(timeout=240) as client:
        if barrier is not None:
            barrier.wait(timeout=30)
        started = time.perf_counter()
        if not body["stream"]:
            response = client.post(base_url + "/v1/completions", json=body)
            response.raise_for_status()
            result = validate_response(response.json())
        else:
            events = []
            with client.stream("POST", base_url + "/v1/completions", json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    event = raw if raw == "[DONE]" else json.loads(raw)
                    events.append(event)
                    if isinstance(event, dict) and any(
                        c.get("finish_reason") is None and c.get("text")
                        for c in event.get("choices", [])
                    ):
                        if first_token is not None:
                            first_token.set()
                        if cancel:
                            break
            result = stream_summary(events)
    result["wall_seconds"] = time.perf_counter() - started
    return result


def run(args):
    import httpx
    import uvicorn
    from hipengine import LLM
    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.core.memory import memory_stats
    from hipengine.server.api import ServerConfig, create_app

    before_memory = memory_stats()
    llm = LLM(str(args.model), backend="hip_gfx1151", max_active_requests=8,
              max_sequence_length=8192, execution_profile=None)
    server = thread = runner = original_reclaim = None
    reclaimed = {}
    rows, cancellation = [], []
    report = {"kind": "qwen38_public_default_socket_serving", "passed": False,
              "performance_claim": False, "scope": "greedy AR; MTP checked separately",
              "rows": rows, "cancellation": cancellation}
    try:
        llm.prepare(max_sequence_length=8192)
        generator = llm._get_text_generator()
        runner = generator._runner
        original_reclaim = runner.reclaim

        def capture_reclaim(owner, completed):
            row = owner._rows.get(int(completed.request_id))
            if row is not None and row.slot is not None:
                reclaimed[int(completed.request_id)] = {
                    "prompt_ids": list(completed.prompt_tokens),
                    "generated_ids": list(row.slot.generated_ids),
                }
            original_reclaim(completed)

        runner.reclaim = MethodType(capture_reclaim, runner)
        suite = load_prompt_suite(DEFAULT_PROMPTS)
        prompts = [list(generator.tokenize(row["rendered_prompt"])) for row in suite]
        report["profile"] = {"requested": None, "resolved": llm.resolved_execution_profile,
                             "manifest_sha256": llm.execution_profile_manifest_sha256,
                             "manifest": llm.execution_profile_manifest}
        report["prompt_suite"] = str(DEFAULT_PROMPTS)
        report["prompt_ids"] = [row["id"] for row in suite]
        app = create_app(ServerConfig(
            model=str(args.model), backend="hip_gfx1151", served_model_name="qwen38-closure",
            max_active_requests=8, max_context_tokens=8192, metrics="prometheus",
            eager_load=False, stream_queue_max_chunks=256, shutdown_grace_seconds=10,
        ), llm=llm)
        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
        thread = threading.Thread(target=server.run, name="qwen38-closure-server")
        thread.start()
        if not _wait_for(lambda: server.started, timeout=120):
            raise RuntimeError("server startup timed out")
        base = f"http://127.0.0.1:{port}"
        references = [request(base, payload(prompt, args.tokens)) for prompt in prompts]
        report["references"] = references
        for streaming in (False, True):
            for width in (1, 2, 4, 8):
                for repeat in range(3):
                    indices = [(repeat * width + offset) % len(prompts) for offset in range(width)]
                    barrier = threading.Barrier(width)
                    with ThreadPoolExecutor(max_workers=width) as pool:
                        futures = [pool.submit(
                            request, base, payload(prompts[i], args.tokens, stream=streaming),
                            barrier=barrier) for i in indices]
                        actual = [future.result() for future in futures]
                    matches = [
                        (value["complete"] and value["text"] == references[i]["text"])
                        if streaming else value["ids"] == references[i]["ids"]
                        for i, value in zip(indices, actual, strict=True)
                    ]
                    if streaming:
                        for i, value in zip(indices, actual, strict=True):
                            ids = value["request_ids"]
                            if len(ids) != 1 or not _wait_for(
                                lambda: ids[0] in reclaimed, timeout=10):
                                raise ValueError("SSE response lacks a reclaimed request identity")
                            terminal = reclaimed[ids[0]]
                            value["terminal_generated_ids"] = terminal["generated_ids"]
                            value["terminal_prompt_ids"] = terminal["prompt_ids"]
                            matches.append(terminal["generated_ids"] == references[i]["ids"]
                                           and terminal["prompt_ids"] == prompts[i])
                    row = {"stream": streaming, "width": width, "repeat": repeat,
                           "prompt_indices": indices, "exact": all(matches),
                           "responses": actual, "telemetry": _resident_observability(llm, recent=width)}
                    rows.append(row)
                    print(f"stream={streaming} C{width} repeat={repeat}: exact={row['exact']}", flush=True)

        with httpx.Client(timeout=240) as client:
            cancel_metric = "hipengine_request_cancelled_total"
            for phase in ("decode", "prefill"):
                baseline = _metrics_values(client, base).get(cancel_metric, 0)
                ready = threading.Event()
                with ThreadPoolExecutor(max_workers=2) as pool:
                    survivor = pool.submit(request, base,
                        payload(prompts[0], args.tokens, stream=True), first_token=ready)
                    if not ready.wait(60):
                        raise RuntimeError("survivor never entered decode")
                    if phase == "decode":
                        victim = request(base, payload(prompts[1], 128, stream=True), cancel=True)
                        admitted = not victim["complete"] and bool(victim["text"])
                    else:
                        long_prompt = (prompts[1] * (4096 // len(prompts[1]) + 1))[:4096]
                        body = json.dumps(payload(long_prompt, 128, stream=True)).encode()
                        with socket.create_connection(("127.0.0.1", port), timeout=60) as sock:
                            sock.sendall(
                                b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                                b"Content-Type: application/json\r\nContent-Length: "
                                + str(len(body)).encode() + b"\r\n\r\n" + body)
                            sock.settimeout(60)
                            headers = b""
                            while b"\r\n\r\n" not in headers:
                                chunk = sock.recv(4096)
                                if not chunk:
                                    raise RuntimeError("prefill victim closed before response headers")
                                headers += chunk
                            admitted = b"200 OK" in headers
                            time.sleep(0.2)
                    observed = _wait_for(
                        lambda: _metrics_values(client, base).get(cancel_metric, 0) > baseline,
                        timeout=60)
                    survivor_result = survivor.result(timeout=120)
                refill = request(base, payload(prompts[0], args.tokens))
                passed = (admitted and observed and survivor_result["complete"]
                          and survivor_result["text"] == references[0]["text"]
                          and refill["ids"] == references[0]["ids"])
                cancellation.append({"phase": phase, "admitted": admitted,
                                     "cancellation_counter_increased": observed, "passed": passed,
                                     "survivor": survivor_result, "refill": refill})
                print(f"{phase} cancellation/refill: {passed}", flush=True)
            report["final_metrics"] = _metrics_values(client, base)
        report["resident_final"] = _resident_observability(llm, recent=20)
        report["drained"] = _wait_for(lambda: not runner._rows, timeout=30)
        report["passed"] = (report["drained"] and all(row["exact"] for row in rows)
                            and all(row["passed"] for row in cancellation))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=60)
            report["server_joined"] = not thread.is_alive()
        llm.close()
        report["memory_before"] = before_memory
        report["memory_after"] = memory_stats()
        report["tracked_memory_reclaimed"] = all(
            report["memory_after"][key] == before_memory[key]
            for key in ("current_allocated_bytes", "active_allocations"))
        report["passed"] = bool(report["passed"] and report.get("server_joined")
                                and report["tracked_memory_reclaimed"])
        report["provenance"] = collect_artifact_provenance(
            repo_root=ROOT, configured_backend="hip_gfx1151", resolved_backend="hip_gfx1151",
            target_arch="gfx1151", model_path=args.model, quant="gguf_q4_k_m", kv_dtype="bf16",
            command=[sys.executable, *sys.argv], environment={
                k: v for k, v in os.environ.items() if k.startswith(("HIPENGINE_", "GPU_MAX_HW_QUEUES"))},
            build_profile="public_default_socket_gate", timing_protocol="quality_only",
            warmups=0, repetitions=3, profiler={"enabled": False})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if run(args)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
