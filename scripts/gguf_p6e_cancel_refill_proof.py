#!/usr/bin/env python3
"""P6e service proof: cancel a real prefill, refill, and verify the survivors.

The in-process half of P6e (``gguf_resumable_prefill_gpu_proof.py``) proves the
yield mechanism on GPU. This half proves the *service* consequence: a long prompt
arriving while a short request is being served, cancelled mid-prefill, must not
damage the surviving request or leak the cancelled one's state.

``gguf_live_server_bench.py`` cannot do this: it has no cancel path. The server
itself already supports client-disconnect cancellation
(``hipengine/server/api.py`` exposes ``cancel()``/``cancelled`` and records
``hipengine_request_cancelled_total``).

Why a real socket instead of ``TestClient``: during a long prefill the SSE stream
carries no bytes (no keepalive, no prefill event), so a reader blocked in
``iter_lines()`` cannot observe an abort request, and closing a ``TestClient``
response only takes effect once the reader returns. A genuine mid-prefill
disconnect therefore needs a real socket to close. This harness runs the app
under ``uvicorn`` in a thread of the same process - so the scheduler stays
observable in-process - and aborts the long request by closing a raw TCP
connection while its prefill is still in flight.

Sequence:

1. **Reference** - stream a short prompt to completion; record its generated token
   IDs and inter-token gaps. This is the survivor truth.
2. **Cancel** - open a raw socket, send a multi-round long prompt, wait until the
   scheduler shows it admitted, wait a little longer so the prefill is really
   running, then close the socket. That is the disconnect the server cancels on.
3. **Refill** - immediately stream the short prompt again.
4. **Verify** - the cancellation counter moved, the refill reproduces the
   reference token IDs exactly, and the refill's acknowledgement latency and
   inter-token gap stay inside limits declared before measuring.

Eager, greedy, prefix-off, MTP-off, artifact-scoped. ``performance_claim`` is
false: this is a control-and-liveness proof, not a throughput result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import sys
import threading
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from hipengine.core import build as _hipengine_build  # noqa: E402
from hipengine.kernels.backends import hip_target_arch_for_backend  # noqa: E402
from hipengine.server import ServerConfig, create_app  # noqa: E402
from scripts.gguf_live_server_bench import (  # noqa: E402
    _EXACT_ENV,
    _parse_sse_data_line,
    _prompt_rows,
    _read_compiler_version,
    _stats,
    _temporary_environment,
)

ARTIFACT_KIND = "w7900_p6e_cancel_refill_service_proof"
DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_COMPILER_VERSION_FILE = Path("/tmp/hipengine-hipcc-version.txt")

REQUIRE_CACHED_BUILD_ENV = _hipengine_build._ENV_REQUIRE_CACHED_BUILD
COMPILER_VERSION_FILE_ENV = "HIPENGINE_COMPILER_VERSION_FILE"
# The layer-outer route that the resumable prefill lives on, and the indicator that
# only that route moves. The live oracle-owner gauge CANNOT be used: /metrics
# scrapes block on the loop lock while a synchronous prefill runs, so
# out-of-process polling never observes those buffers while live
# (qwen35_gguf_runner.py:22425). The while-live capture is the observed peak, which
# is a running max recorded at release time for exactly this reason.
PACKED_LAYER_OUTER_ENV = "HIPENGINE_GGUF_PACKED_LAYER_OUTER"
ORACLE_OWNER_BYTES_METRIC = "hipengine_resident_prefill_oracle_owner_bytes"
ORACLE_OWNER_COUNT_METRIC = "hipengine_resident_prefill_oracle_owners"
ORACLE_OBSERVED_PEAK_BYTES_METRIC = (
    "hipengine_resident_prefill_oracle_observed_peak_bytes"
)
ORACLE_OBSERVED_PEAK_OWNERS_METRIC = (
    "hipengine_resident_prefill_oracle_observed_peak_owners"
)

# Declared before measuring, so no gate can be fitted to the result.
DECLARED_ACK_P95_LIMIT_MS = 3000.0
DECLARED_ACK_P99_LIMIT_MS = 6000.0
DECLARED_GAP_MAX_FACTOR = 2.5
CANCELLATION_COUNTER = "hipengine_request_cancelled_total"
REQUEST_FAILED_METRIC = "hipengine_request_failed_total"
WORK_PREFILL_METRIC = "hipengine_resident_work_prefill_total"
ACTIVE_REQUESTS_METRIC = "hipengine_resident_requests_active"
PREFILL_OWNER_BYTES_METRIC = "hipengine_resident_prefill_hidden_owner_bytes"
ORACLE_OWNER_BYTES_METRIC = "hipengine_resident_prefill_oracle_owner_bytes"


class _StreamOutcome:
    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.delta_times: list[float] = []
        self.first_token_at: float | None = None
        self.finish_reason: str | None = None
        self.status_code = 0
        self.error: str | None = None
        self.raw_lines: list[str] = []
        self.wall_ms: float | None = None

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _completion_payload(
    *, model_name: str, prompt: str, max_tokens: int, stream: bool = True
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream": bool(stream),
    }
    if stream:
        payload["stream_options"] = {"include_hipengine": True}
    return payload


def _consume_sse_line(outcome: _StreamOutcome, raw_line: str, observed: float) -> None:
    """Fold one SSE line into an outcome. Shared by every streaming arm."""

    if len(outcome.raw_lines) < 8 and raw_line.strip():
        outcome.raw_lines.append(str(raw_line)[:200])
    payload = _parse_sse_data_line(raw_line)
    if payload is None or payload == "[DONE]" or not isinstance(payload, dict):
        return
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    choice = choices[0]
    if not isinstance(choice, dict):
        return
    if choice.get("finish_reason") is None:
        # The stream carries text deltas, not token IDs; the text is
        # the survivor truth a cancelled prefill must not perturb.
        if outcome.first_token_at is None:
            outcome.first_token_at = observed
        outcome.delta_times.append(observed)
        outcome.text_parts.append(str(choice.get("text", "")))
    else:
        outcome.finish_reason = str(choice.get("finish_reason"))


def _stream_completion(
    client: httpx.Client,
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    per_line_delay_s: float = 0.0,
) -> _StreamOutcome:
    """Stream one completion to completion and record tokens and timing.

    ``per_line_delay_s`` throttles the reader to model a slow consumer, which
    must not stall the engine for other requests.
    """

    outcome = _StreamOutcome()
    try:
        with client.stream(
            "POST",
            f"{base_url}/v1/completions",
            json=_completion_payload(
                model_name=model_name, prompt=prompt, max_tokens=max_tokens
            ),
        ) as response:
            outcome.status_code = int(response.status_code)
            if outcome.status_code != 200:
                outcome.error = response.read().decode("utf-8", errors="replace")
                return outcome
            for raw_line in response.iter_lines():
                if per_line_delay_s:
                    time.sleep(per_line_delay_s)
                _consume_sse_line(outcome, raw_line, time.perf_counter())
    except Exception as error:  # noqa: BLE001 - recorded in the artifact
        outcome.error = f"{type(error).__name__}: {error}"
    return outcome


def _blocking_completion(
    client: httpx.Client,
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
) -> _StreamOutcome:
    """POST a non-streaming completion, covering the blocking request mode.

    The resumable yield lives in the scheduler rather than the transport, so a
    blocking request must still hand control back between layer segments. This
    arm is what proves it: a short request issued while the long blocking prefill
    is in flight must be admitted and answered on time.
    """

    outcome = _StreamOutcome()
    started = time.perf_counter()
    try:
        response = client.post(
            f"{base_url}/v1/completions",
            json=_completion_payload(
                model_name=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                stream=False,
            ),
        )
        outcome.status_code = int(response.status_code)
        outcome.wall_ms = (time.perf_counter() - started) * 1e3
        if outcome.status_code != 200:
            outcome.error = response.text
            return outcome
        body = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            outcome.text_parts.append(str(choice.get("text", "")))
            outcome.finish_reason = choice.get("finish_reason")
        outcome.first_token_at = started
    except Exception as error:  # noqa: BLE001 - recorded in the artifact
        outcome.error = f"{type(error).__name__}: {error}"
    return outcome


def _inter_token_gaps_ms(outcome: _StreamOutcome) -> list[float]:
    return [
        (later - earlier) * 1e3
        for earlier, later in zip(outcome.delta_times, outcome.delta_times[1:])
    ]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _start_server(app: Any, port: int, timeout: float = 120.0) -> uvicorn.Server:
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="p6e-uvicorn")
    thread.start()
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if server.started:
            return server
        time.sleep(0.05)
    raise RuntimeError("uvicorn did not start within the timeout")


def _cancel_probe(
    *,
    port: int,
    model_name: str,
    prompt: str,
    max_tokens: int,
    admit_timeout: float,
    hold_seconds: float,
) -> dict[str, Any]:
    """Send a long prompt over a raw socket, then close it mid-prefill.

    Closing the socket is a real TCP disconnect, which the ASGI server turns into
    the cancellation this proof is about. Admission is read from the server's own
    metrics, so the close lands after the request is really running.
    """

    body = json.dumps(
        _completion_payload(model_name=model_name, prompt=prompt, max_tokens=max_tokens)
    ).encode("utf-8")
    request = (
        b"POST /v1/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: close\r\n\r\n" + body
    )
    result: dict[str, Any] = {
        "admitted": False,
        "admission_ms": None,
        "held_open_ms": None,
        "bytes_before_close": 0,
        "closed_by_server": False,
        "response_head": "",
        "error": None,
    }
    started = time.perf_counter()
    sock = socket.create_connection(("127.0.0.1", port), timeout=admit_timeout)
    try:
        sock.sendall(request)
        # Admission is the arrival of the response headers. uvicorn sends them as
        # soon as the streaming response starts, which is before the generator has
        # produced any token - so this is the point at which the prefill is about
        # to run, not after it has finished. Waiting for a token or for a request
        # gauge would miss the prefill entirely.
        buffer = b""
        deadline = time.perf_counter() + admit_timeout
        while b"\r\n\r\n" not in buffer and time.perf_counter() < deadline:
            sock.settimeout(0.2)
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
        if b"\r\n\r\n" in buffer:
            result["admitted"] = True
            result["admission_ms"] = round((time.perf_counter() - started) * 1e3, 3)
            result["response_head"] = buffer[:800].decode("utf-8", errors="replace")
        # Hold the connection open past admission so the prefill is genuinely
        # running when the socket closes.
        if result["admitted"]:
            hold_started = time.perf_counter()
            time.sleep(hold_seconds)
            result["held_open_ms"] = round((time.perf_counter() - hold_started) * 1e3, 3)
        # Count anything that arrived while holding, so the artifact shows whether
        # the prefill was still in flight (no token yet) when the socket closed.
        # The body is kept too: a rejection streams an error event instead of
        # tokens, and that distinction decides whether a cancel was even possible.
        sock.settimeout(0.2)
        try:
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    result["closed_by_server"] = True
                    break
                result["bytes_before_close"] += len(chunk)
                if len(result["response_head"]) < 1200:
                    result["response_head"] += chunk.decode("utf-8", errors="replace")
        except (socket.timeout, OSError):
            pass
    except Exception as error:  # noqa: BLE001
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        sock.close()
    return result


def _metrics_values(client: httpx.Client, base_url: str) -> dict[str, float]:
    """Scrape every ``hipengine_*`` Prometheus sample as a name->sum map."""

    try:
        response = client.get(f"{base_url}/metrics")
    except Exception:  # noqa: BLE001
        return {}
    if int(response.status_code) != 200:
        return {}
    values: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, raw = line.rpartition(" ")
        if not name or not raw:
            continue
        base = name.split("{", 1)[0]
        if not base.startswith("hipengine_"):
            continue
        try:
            values[base] = values.get(base, 0.0) + float(raw)
        except ValueError:
            continue
    return values


def _metric(values: Mapping[str, float], name: str) -> float | None:
    return values.get(name)


def _metric_delta(
    after: Mapping[str, float], before: Mapping[str, float], name: str
) -> float | None:
    new = after.get(name)
    old = before.get(name)
    if new is None or old is None:
        return None
    return new - old


def _wait_for(predicate: Any, *, timeout: float, interval: float = 0.05) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def evaluate_gates(
    *,
    reference_text: str,
    refill_text: str,
    cancellation_delta: float | None,
    acknowledgement_ms: float | None,
    refill_gaps_ms: Sequence[float],
    reference_gaps_ms: Sequence[float],
    drained: bool,
    prefill_owner_released: bool,
    blocking_survivor_text: str = "",
    slow_consumer_survivor_text: str = "",
    slow_consumer_completed: bool = False,
    oracle_observed_peak_bytes: float | None = None,
    oracle_observed_peak_owners: float | None = None,
    live_oracle_owners: float | None = None,
) -> dict[str, Any]:
    """Evaluate the service gates. Split out so the failure paths are testable."""

    gates: dict[str, Any] = {}
    gates["cancellation_reached_backend"] = {
        "passed": cancellation_delta is not None and cancellation_delta >= 1,
        "counter": CANCELLATION_COUNTER,
        "delta": cancellation_delta,
        "detail": (
            "closing the socket must reach the backend as a cancellation, not "
            "merely stop the client reading"
        ),
    }
    gates["resumable_path_engaged"] = {
        # Without this the other gates can all pass on the default chunk-outer
        # path, which does not contain the deliverable at all. The indicator must
        # be the WHILE-LIVE observed peak, not the live owner gauge: scrapes block
        # on the loop lock during a synchronous prefill, so polling the live gauge
        # from out of process always reads zero and would fail this gate no matter
        # what the engine did.
        "passed": bool(oracle_observed_peak_owners) and bool(oracle_observed_peak_bytes),
        "oracle_observed_peak_owners": oracle_observed_peak_owners,
        "oracle_observed_peak_bytes": oracle_observed_peak_bytes,
        "owners_metric": ORACLE_OBSERVED_PEAK_OWNERS_METRIC,
        "bytes_metric": ORACLE_OBSERVED_PEAK_BYTES_METRIC,
        "live_gauge_owners": live_oracle_owners,
        "detail": (
            "the layer-outer route must actually engage during a long prefill, "
            "proven by the while-live observed peak of the shared per-layer oracle "
            "owner being non-zero; otherwise every other gate here is evidence "
            "about the default route rather than about the resumable yield"
        ),
    }
    gates["survivors_exact"] = {
        "passed": bool(refill_text) and refill_text == reference_text,
        "reference_chars": len(reference_text),
        "refill_chars": len(refill_text),
        "reference_sha256": hashlib.sha256(reference_text.encode()).hexdigest(),
        "refill_sha256": hashlib.sha256(refill_text.encode()).hexdigest(),
        "detail": (
            "the refill request must reproduce the reference completion text "
            "exactly, so a cancelled prefill left no trace in the surviving path"
        ),
    }
    # The yield lives in the scheduler rather than the transport, so the same
    # guarantee must hold when the long request is blocking (non-streaming) and
    # when the concurrent reader consumes its stream slowly. Both arms submit the
    # same short prompt as the reference arm, so the same text is the expectation.
    gates["blocking_survivor_exact"] = {
        "passed": bool(blocking_survivor_text)
        and blocking_survivor_text == reference_text,
        "chars": len(blocking_survivor_text),
        "sha256": hashlib.sha256(blocking_survivor_text.encode()).hexdigest(),
        "detail": (
            "a short request issued while a long blocking (non-streaming) prefill "
            "is in flight must reproduce the reference text exactly, which is what "
            "shows the yield also serves the blocking request mode"
        ),
    }
    gates["slow_consumer_survivor_exact"] = {
        "passed": bool(slow_consumer_survivor_text)
        and slow_consumer_survivor_text == reference_text,
        "chars": len(slow_consumer_survivor_text),
        "sha256": hashlib.sha256(slow_consumer_survivor_text.encode()).hexdigest(),
        "slow_consumer_completed": slow_consumer_completed,
        "detail": (
            "a reader that consumes its stream slowly must not stall the engine: "
            "a short request issued while the slow reader is mid-stream must still "
            "reproduce the reference text exactly"
        ),
    }
    gates["bounded_acknowledgement"] = {
        "passed": acknowledgement_ms is not None
        and acknowledgement_ms <= DECLARED_ACK_P95_LIMIT_MS,
        "acknowledgement_ms": acknowledgement_ms,
        "p95_limit_ms": DECLARED_ACK_P95_LIMIT_MS,
        "p99_limit_ms": DECLARED_ACK_P99_LIMIT_MS,
        "detail": (
            "a refill admitted right after a cancelled prefill must be acknowledged "
            "within the declared limit"
        ),
    }
    gap_limit = (
        max(reference_gaps_ms) * DECLARED_GAP_MAX_FACTOR if reference_gaps_ms else None
    )
    gates["bounded_decode_gap"] = {
        "passed": bool(refill_gaps_ms)
        and gap_limit is not None
        and max(refill_gaps_ms) <= gap_limit,
        "refill_max_gap_ms": round(max(refill_gaps_ms), 3) if refill_gaps_ms else None,
        "reference_max_gap_ms": (
            round(max(reference_gaps_ms), 3) if reference_gaps_ms else None
        ),
        "limit_ms": round(gap_limit, 3) if gap_limit is not None else None,
        "declared_factor": DECLARED_GAP_MAX_FACTOR,
        "detail": (
            "the surviving request's worst inter-token gap must stay within the "
            "declared factor of its own reference run"
        ),
    }
    gates["cleanup"] = {
        "passed": bool(drained) and bool(prefill_owner_released),
        "drained": bool(drained),
        "prefill_owner_released": bool(prefill_owner_released),
        "detail": (
            "after the cancel and the refill the scheduler must drain to zero "
            "active requests and the resident prefill hidden/oracle owner bytes "
            "must return to their baseline, so the cancelled request released "
            "both its slot and its suspended state"
        ),
    }
    return gates


def run(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine import LLM, SamplingParams

    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    backend = str(args.backend)
    version_file = (
        args.compiler_version_file.expanduser().resolve()
        if args.compiler_version_file
        else None
    )
    compiler_version = _read_compiler_version(version_file)
    if args.require_cached_build and compiler_version is None:
        raise ValueError("require-cached-build requires compiler-version-file")

    long_rows = int(args.long_prompt_rows)
    short_rows = int(args.short_prompt_rows)
    max_tokens = int(args.decode_tokens)
    max_sequence_length = long_rows + max_tokens + 2
    model_name = "qwen35-p6e"

    environment = {
        **_EXACT_ENV,
        # P6's resumable yield only exists on the layer-outer route, which is
        # default OFF (HIPENGINE_GGUF_PACKED_LAYER_OUTER=0). Without this the
        # harness silently measures the default chunk-outer path and its gates
        # pass for reasons that have nothing to do with the deliverable. The
        # resumable_path_engaged gate below is what stops that from recurring.
        PACKED_LAYER_OUTER_ENV: "1" if args.packed_layer_outer else "0",
        "HIPENGINE_MAX_ACTIVE_REQUESTS": "2",
        # Sized for the long prompt, not for the live bench's 512-token contexts:
        # a 3072-token context needs far more KV pages than the live bench's
        # max_rows * 3, and an under-provisioned pool rejects the long request
        # before it ever reaches a cancellable prefill.
        "HIPENGINE_KV_POOL_INITIAL_PAGES": "256",
        "HIPENGINE_KV_POOL_LOW_WATER_PAGES": "128",
        "HIPENGINE_KV_POOL_HIGH_WATER_PAGES": "512",
        "HIPENGINE_KV_POOL_CHUNK_PAGES": "32",
        "HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS": "0",
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "0"),
        "HIPENGINE_HIP_ARCH": hip_target_arch_for_backend(backend),
    }
    if version_file is not None:
        environment[COMPILER_VERSION_FILE_ENV] = str(version_file)
    if args.require_cached_build:
        environment[REQUIRE_CACHED_BUILD_ENV] = "1"

    stack = ExitStack()
    server: uvicorn.Server | None = None
    try:
        with _temporary_environment(environment):
            # Record what the process actually sees, not just what was requested:
            # "the flag was passed" and "the engine read the flag" are different
            # claims, and the engagement gate is only meaningful for the second.
            layer_outer_env_seen = os.environ.get(PACKED_LAYER_OUTER_ENV)
            llm = LLM(model, backend=backend, max_sequence_length=max_sequence_length)
            adapter = llm._get_text_generator()
            llm.prepare(
                max_sequence_length=max_sequence_length,
                sampling_params=SamplingParams(max_tokens=max_tokens),
            )
            tokenizer = adapter._runner.generator.tokenizer
            short_row = _prompt_rows(
                tokenizer,
                rows=1,
                prompt_length=short_rows,
                prompt_token_id=int(args.prompt_token_id),
            )[0]
            long_row = _prompt_rows(
                tokenizer,
                rows=1,
                prompt_length=long_rows,
                prompt_token_id=int(args.prompt_token_id),
            )[0]
            short_text = str(short_row["text"])

            app = create_app(
                ServerConfig(
                    model=str(model),
                    backend=backend,
                    quant=str(args.quant),
                    served_model_name=model_name,
                    eager_load=False,
                    metrics="prometheus",
                    generation_batch_window_ms=float(args.batch_window_ms),
                    max_context_tokens=max_sequence_length,
                    max_active_requests=2,
                    stream_queue_max_chunks=max_tokens + 8,
                    shutdown_grace_seconds=5.0,
                ),
                llm=llm,
            )
            port = _free_port()
            server = _start_server(app, port)
            base_url = f"http://127.0.0.1:{port}"
            client = stack.enter_context(
                httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0))
            )

            # --- 1. reference run ------------------------------------------------
            reference = _stream_completion(
                client,
                base_url=base_url,
                model_name=model_name,
                prompt=short_text,
                max_tokens=max_tokens,
            )
            reference_gaps = _inter_token_gaps_ms(reference)

            baseline_metrics = _metrics_values(client, base_url)
            cancelled_before = _metric(baseline_metrics, CANCELLATION_COUNTER)
            owner_bytes_baseline = _metric(baseline_metrics, PREFILL_OWNER_BYTES_METRIC)
            oracle_bytes_baseline = _metric(baseline_metrics, ORACLE_OWNER_BYTES_METRIC)

            # --- 2. cancel a real prefill ----------------------------------------
            cancel_result = _cancel_probe(
                port=port,
                model_name=model_name,
                prompt=str(long_row["text"]),
                max_tokens=max_tokens,
                admit_timeout=float(args.admit_timeout_seconds),
                hold_seconds=float(args.cancel_delay_ms) / 1e3,
            )
            # Give the server a moment to observe the disconnect and cancel. A
            # disconnect is only noticed when the server next touches the
            # connection, which for a streaming completion is the first write
            # after the prefill, so this waits past the prefill duration.
            _wait_for(
                lambda: (
                    _metric(_metrics_values(client, base_url), CANCELLATION_COUNTER) or 0
                )
                > (cancelled_before or 0),
                timeout=float(args.cancel_observe_timeout_seconds),
            )
            after_cancel_metrics = _metrics_values(client, base_url)
            cancelled_after = _metric(after_cancel_metrics, CANCELLATION_COUNTER)
            cancellation_delta = (
                None
                if cancelled_before is None or cancelled_after is None
                else cancelled_after - cancelled_before
            )
            # If the disconnect was ignored, the cancelled request's prefill work
            # still runs to completion. These counters distinguish "cancelled"
            # from "silently kept working".
            work_delta = _metric_delta(
                after_cancel_metrics, baseline_metrics, WORK_PREFILL_METRIC
            )
            failure_delta = _metric_delta(
                after_cancel_metrics, baseline_metrics, REQUEST_FAILED_METRIC
            )

            # --- 3. refill --------------------------------------------------------
            refill_started = time.perf_counter()
            refill = _stream_completion(
                client,
                base_url=base_url,
                model_name=model_name,
                prompt=short_text,
                max_tokens=max_tokens,
            )
            refill_submit_to_first_ms = (
                (refill.first_token_at - refill_started) * 1e3
                if refill.first_token_at
                else None
            )
            refill_gaps = _inter_token_gaps_ms(refill)

            # --- 4. drain and confirm the cancelled request released its state ----
            drained = _wait_for(
                lambda: (_metric(_metrics_values(client, base_url), ACTIVE_REQUESTS_METRIC) or 0)
                == 0,
                timeout=float(args.drain_timeout_seconds),
            )
            final_metrics = _metrics_values(client, base_url)
            owner_bytes_final = _metric(final_metrics, PREFILL_OWNER_BYTES_METRIC)
            oracle_bytes_final = _metric(final_metrics, ORACLE_OWNER_BYTES_METRIC)
            prefill_owner_released = (
                owner_bytes_final is not None
                and owner_bytes_baseline is not None
                and owner_bytes_final <= owner_bytes_baseline
                and oracle_bytes_final is not None
                and oracle_bytes_baseline is not None
                and oracle_bytes_final <= oracle_bytes_baseline
            )

            # --- 5. blocking request mode ----------------------------------------
            # The yield lives in the scheduler, not the transport, so a blocking
            # (non-streaming) long prompt must also hand control back between
            # layer segments. A short request issued while that prefill is in
            # flight is what proves it.
            lead_seconds = float(args.concurrent_lead_seconds)
            join_seconds = float(args.concurrent_join_timeout_seconds)
            blocking: dict[str, Any] = {}

            def _run_blocking() -> None:
                blocking["outcome"] = _blocking_completion(
                    client,
                    base_url=base_url,
                    model_name=model_name,
                    prompt=str(long_row["text"]),
                    max_tokens=max_tokens,
                )

            blocking_thread = threading.Thread(target=_run_blocking, daemon=True)
            blocking_thread.start()
            time.sleep(lead_seconds)
            blocking_survivor = _stream_completion(
                client,
                base_url=base_url,
                model_name=model_name,
                prompt=short_text,
                max_tokens=max_tokens,
            )
            blocking_thread.join(timeout=join_seconds)
            blocking_outcome = blocking.get("outcome")

            # --- 6. slow consumer ------------------------------------------------
            # A reader that consumes its stream slowly must not stall the engine.
            slow: dict[str, Any] = {}

            def _run_slow() -> None:
                slow["outcome"] = _stream_completion(
                    client,
                    base_url=base_url,
                    model_name=model_name,
                    prompt=str(long_row["text"]),
                    max_tokens=max_tokens,
                    per_line_delay_s=float(args.slow_consumer_line_delay_ms) / 1e3,
                )

            slow_thread = threading.Thread(target=_run_slow, daemon=True)
            slow_thread.start()
            time.sleep(lead_seconds)
            slow_survivor = _stream_completion(
                client,
                base_url=base_url,
                model_name=model_name,
                prompt=short_text,
                max_tokens=max_tokens,
            )
            slow_thread.join(timeout=join_seconds)
            slow_outcome = slow.get("outcome")

            # Read the while-live observed peak from a settled scrape. The live
            # owner gauge is deliberately also recorded, because it reads zero even
            # when the route ran, which is what made an earlier version of this gate
            # a false negative.
            engagement_metrics = _metrics_values(client, base_url)
            oracle_observed_peak_bytes = _metric(
                engagement_metrics, ORACLE_OBSERVED_PEAK_BYTES_METRIC
            )
            oracle_observed_peak_owners = _metric(
                engagement_metrics, ORACLE_OBSERVED_PEAK_OWNERS_METRIC
            )
            live_oracle_owners = _metric(
                engagement_metrics, ORACLE_OWNER_COUNT_METRIC
            )
    finally:
        if server is not None:
            server.should_exit = True
        stack.close()

    gates = evaluate_gates(
        reference_text=reference.text,
        refill_text=refill.text,
        cancellation_delta=cancellation_delta,
        acknowledgement_ms=refill_submit_to_first_ms,
        refill_gaps_ms=refill_gaps,
        reference_gaps_ms=reference_gaps,
        drained=drained,
        prefill_owner_released=prefill_owner_released,
        blocking_survivor_text=blocking_survivor.text,
        slow_consumer_survivor_text=slow_survivor.text,
        slow_consumer_completed=(
            slow_outcome is not None
            and slow_outcome.error is None
            and bool(slow_outcome.text)
            and slow_outcome.finish_reason is not None
        ),
        oracle_observed_peak_bytes=oracle_observed_peak_bytes,
        oracle_observed_peak_owners=oracle_observed_peak_owners,
        live_oracle_owners=live_oracle_owners,
    )

    return {
        "schema": 1,
        "kind": ARTIFACT_KIND,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "passed": all(gate["passed"] for gate in gates.values()),
        "host": {
            "target_arch": hip_target_arch_for_backend(backend),
            "hip_visible_devices": environment.get("HIP_VISIBLE_DEVICES", ""),
        },
        "model": {
            "path": str(model),
            "quant": str(args.quant),
            "route": "int8_direct, eager, greedy, prefix-off, MTP-off",
        },
        "workload": {
            "long_prompt_rows": long_rows,
            "short_prompt_rows": short_rows,
            "decode_tokens": max_tokens,
            "cancel_hold_ms": float(args.cancel_delay_ms),
            "max_active_requests": 2,
            "transport": "uvicorn on a loopback socket; cancel is a raw TCP close",
        },
        "gates": gates,
        "measurements": {
            "reference": {
                "chars": len(reference.text),
                "text_sha256": reference.text_sha256,
                "finish_reason": reference.finish_reason,
                "error": reference.error,
                "raw_lines": reference.raw_lines,
                "gaps_ms": _stats(reference_gaps),
            },
            "cancel_probe": cancel_result,
            "refill": {
                "chars": len(refill.text),
                "text_sha256": refill.text_sha256,
                "finish_reason": refill.finish_reason,
                "error": refill.error,
                "submit_to_first_token_ms": (
                    round(refill_submit_to_first_ms, 3)
                    if refill_submit_to_first_ms is not None
                    else None
                ),
                "gaps_ms": _stats(refill_gaps),
            },
            "cancellation_counter_before": cancelled_before,
            "cancellation_counter_after": cancelled_after,
            "packed_layer_outer_enabled": bool(args.packed_layer_outer),
            "packed_layer_outer_env_seen": layer_outer_env_seen,
            "oracle_observed_peak_bytes": oracle_observed_peak_bytes,
            "oracle_observed_peak_owners": oracle_observed_peak_owners,
            "live_oracle_owners": live_oracle_owners,
            "blocking": {
                "long_request": {
                    "chars": len(blocking_outcome.text)
                    if blocking_outcome is not None
                    else 0,
                    "finish_reason": (
                        blocking_outcome.finish_reason
                        if blocking_outcome is not None
                        else None
                    ),
                    "wall_ms": (
                        round(blocking_outcome.wall_ms, 3)
                        if blocking_outcome is not None
                        and blocking_outcome.wall_ms is not None
                        else None
                    ),
                    "error": (
                        blocking_outcome.error if blocking_outcome is not None else None
                    ),
                },
                "survivor": {
                    "chars": len(blocking_survivor.text),
                    "text_sha256": blocking_survivor.text_sha256,
                    "finish_reason": blocking_survivor.finish_reason,
                    "error": blocking_survivor.error,
                },
            },
            "slow_consumer": {
                "reader_line_delay_ms": float(args.slow_consumer_line_delay_ms),
                "long_request": {
                    "chars": len(slow_outcome.text) if slow_outcome is not None else 0,
                    "finish_reason": (
                        slow_outcome.finish_reason if slow_outcome is not None else None
                    ),
                    "error": slow_outcome.error if slow_outcome is not None else None,
                },
                "survivor": {
                    "chars": len(slow_survivor.text),
                    "text_sha256": slow_survivor.text_sha256,
                    "finish_reason": slow_survivor.finish_reason,
                    "error": slow_survivor.error,
                },
            },
            "prefill_work_total_delta": work_delta,
            "request_failed_total_delta": failure_delta,
            "metrics_baseline": {
                key: value for key, value in sorted(baseline_metrics.items())
            },
            "prefill_owner_bytes_baseline": owner_bytes_baseline,
            "prefill_owner_bytes_final": owner_bytes_final,
            "oracle_owner_bytes_baseline": oracle_bytes_baseline,
            "oracle_owner_bytes_final": oracle_bytes_final,
        },
        "command": " ".join(
            shlex.quote(part)
            for part in [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        ),
        "trace_environment": {
            key: environment[key]
            for key in (COMPILER_VERSION_FILE_ENV, REQUIRE_CACHED_BUILD_ENV)
            if key in environment
        },
        "notes": [
            "Control-and-liveness proof, not a throughput claim: performance_claim is "
            "false and no benchmark scoreboard row changes.",
            "The acknowledgement and decode-gap limits were declared before "
            "measuring, so no gate could be fitted to the result.",
            "Cancellation is a raw TCP close on a real socket, not a TestClient "
            "context exit: during prefill the SSE stream carries no bytes, so a "
            "reader cannot observe an abort and a TestClient close only takes "
            "effect once the reader returns.",
            "The cancellation counter proves the disconnect reached the backend "
            "rather than only stopping the client read.",
            "The long request is confirmed admitted, then held open a little longer, "
            "so the cancel lands inside a real prefill.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument(
        "--long-prompt-rows",
        type=int,
        default=3072,
        help="long prompt length; must exceed the bulk prefill row capacity",
    )
    parser.add_argument("--short-prompt-rows", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=24)
    parser.add_argument(
        "--cancel-delay-ms",
        type=float,
        default=600.0,
        help="how long to hold the admitted long request open before closing it",
    )
    parser.add_argument("--admit-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--cancel-observe-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--drain-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--concurrent-lead-seconds",
        type=float,
        default=0.5,
        help=(
            "how long the long blocking/slow request runs before the short "
            "survivor is submitted, so the survivor lands inside the prefill"
        ),
    )
    parser.add_argument("--concurrent-join-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--slow-consumer-line-delay-ms",
        type=float,
        default=40.0,
        help="per-SSE-line reader delay, which is what models a slow consumer",
    )
    parser.add_argument("--batch-window-ms", type=float, default=20.0)
    parser.add_argument(
        "--compiler-version-file", type=Path, default=DEFAULT_COMPILER_VERSION_FILE
    )
    parser.add_argument(
        "--packed-layer-outer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "enable the layer-outer route the resumable prefill lives on; the "
            "resumable_path_engaged gate fails if it never engages, so this "
            "cannot silently fall back to the default chunk-outer path"
        ),
    )
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact = run(args)
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    text = json.dumps(artifact, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.expanduser().resolve().write_text(text, encoding="utf-8")
    print(text)
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
