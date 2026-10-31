"""OpenAI server integration tests: errors.

Split from ``tests/test_integration_server_api.py`` by RF-12; behavior and
assertions are unchanged. Shared doubles live in ``tests.support``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import anyio
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from hipengine import ResidentPreparationError, SamplingParams
from hipengine.generation import (
    GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS,
    EngineCommandTimeout,
    EngineServiceClosed,
    FinishDetails,
    GenerationAdmissionRejected,
    GenerationCancellationToken,
    GenerationCancelled,
    GenerationDeadlineExceeded,
    GenerationExecutionFailed,
    GenerationOutput,
    GenerationStreamChunk,
    GenerationTelemetry,
    PreparedPromptInput,
    TokenLogprob,
)
from hipengine.generation.qwen4_exp_ar_batch import QWEN4_EXP_MAX_BATCH_ROWS
from hipengine.generation.sampling import (
    NATIVE_GPU_SAMPLER_UNSUPPORTED_CAPABILITIES,
    SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS,
    SPECULATIVE_MTP_INCOMPATIBLE_FIELDS,
)
from hipengine.runtime.memory_admission import (
    ADMISSION_REFUSAL_CODE,
    MemoryAdmissionRefused,
    MemoryConsumer,
    price_memory_admission,
)
from hipengine.server import ServerConfig, create_app, render_chat_prompt
from hipengine.server.__main__ import build_parser
from hipengine.speculative import (
    SpeculativeMTPStaticEligibility,
    SpeculativeMTPStaticState,
)
from hipengine.models.qwen35 import Qwen35GGUFModel
from hipengine.speculative.policy import DEFAULT_AUTO_DEPTH_POLICY
from hipengine.speculative.serving import SpeculativeMTPServingKey
from hipengine.server.api import (
    OpenAIHTTPError,
    StartupFailure,
    _AGENTIC_REPLAY_FAILURE_REASONS,
    _STRUCTURED_OUTPUT_RESULT_VALIDATION_FAILURE_REASONS,
    _TOOL_RESULT_VALIDATION_FAILURE_REASONS,
    _THINKING_CLOSE_MARKER,
    _await_with_request_control,
    _backend_scheduler_token_chunks,
    _coerce_generation_output,
    _execution_route_for_static_intent,
    _generation_route_for_request,
    _GenerationBatcher,
    _MTPCircuitBreaker,
    _QueuedBatchResult,
    _QueuedGeneration,
    _SPECULATIVE_MTP_AUTO_ROUTE,
    _SPECULATIVE_MTP_BATCH_ROUTE,
    _SPECULATIVE_MTP_DEFAULT_ROUTE,
    _SPECULATIVE_MTP_K0_ROUTE,
    _prepared_context_tokens,
    _request_completion_cap,
    _request_control,
    _realized_model_serving_plan,
    _resolve_realized_generation_route,
    _sampling_for_realized_generation_route,
    _sampling_key,
    _serving_plan_route_decision,
    _log_pretty_startup_summary,
    _startup_memory_summary,
    _mtp_accepted_rejected_counts,
    _mtp_response_summary,
)
from hipengine.server.chat_output import _parse_chat_tool_calls
from hipengine.server.metrics import _ServerMetrics
from hipengine.server.schemas import ChatCompletionRequest, CompletionRequest
from tests.support.server_api_fakes import (
    AdmissionRejectedFakeLLM,
    BackendCancelledFakeLLM,
    BackendDeadlineFakeLLM,
    BackendExecutionFailedFakeLLM,
    DelayedFakeLLM,
    FakeLLM,
    PromptPreparingFakeLLM,
)
from tests.support.server_api_support import (
    _routing_metadata,
    _sse_payloads,
)


def test_closed_engine_service_answers_with_a_typed_unavailable_error() -> None:
    """The 2026-09-18 bodyless 500: a closed service must name itself.

    The resident service closed after a fatal packed-prefill failure, and the
    next requests hit ``EngineServiceClosed`` while the server prepared their
    resident context. That exception escaped the endpoint and Starlette answered
    with a plain-text ``Internal Server Error``: no ``error.code``, no message,
    and nothing the client could tell apart from a truncation.
    """

    closed_message = (
        "engine service is closed after a fatal execution failure "
        "(NotImplementedError: packed paged AR prefill currently requires "
        "context < 1024)"
    )

    class ClosedServiceFakeLLM(FakeLLM):
        def prepare(self, *, max_sequence_length=None, sampling_params):
            raise EngineServiceClosed(closed_message)

    fake = ClosedServiceFakeLLM(outputs=["never served"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
        ),
        llm=fake,
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 4,
            "stream": True,
        },
    )

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["error"]["code"] == "engine_unavailable"
    assert body["error"]["type"] == "server_error"
    assert body["error"]["message"] == closed_message
    assert body["error"]["hipengine"]["retryable"] is True
    assert body["error"]["hipengine"]["exception_type"] == "EngineServiceClosed"


def test_admission_refusal_answers_with_its_named_capacity_error() -> None:
    """A pre-mutation capacity refusal is a capacity answer, not an internal fault.

    The admission budget prices every resident consumer before anything is
    allocated, so a refusal leaves the engine healthy and the same request may
    fit once memory frees up. The refusal names the consumer that did not fit and
    carries its own error code, so a client can tell "too large right now" from
    "the engine cannot answer" -- and from its own malformed request.
    """

    decision = price_memory_admission(
        free_bytes=1024**3,
        consumers=(MemoryConsumer("kv_pool", 8 * 1024**3),),
        reserve_bytes=0,
    )
    refusal = MemoryAdmissionRefused(decision)

    class OverloadedFakeLLM(FakeLLM):
        def prepare(self, *, max_sequence_length=None, sampling_params):
            raise refusal

    fake = OverloadedFakeLLM(outputs=["never served"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
        ),
        llm=fake,
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 4,
            "stream": True,
        },
    )

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["error"]["code"] == ADMISSION_REFUSAL_CODE
    assert body["error"]["code"] != "internal_error"
    assert body["error"]["message"] == str(refusal)
    assert "kv_pool" in body["error"]["message"]


def test_engine_command_timeout_answers_with_a_typed_unavailable_error() -> None:
    """A driver that never answers is the server's problem, not the request's.

    The resident service abandons a command whose liveness budget expires, and the
    error names the method, the budget, and the activity holding the driver thread.
    Answering the non-retryable ``internal_error`` would tell a client that its own
    request was at fault for a stalled engine, so an exhausted budget is reported
    the way a closed service is: retryable ``engine_unavailable``.
    """

    timeout_message = (
        "engine service command timed out: method=tokenize budget_s=300.0 "
        "driver_command=control:prepare driver_command_s=301.2 queued_commands=2 "
        "active_requests=1 driver_alive=yes"
    )

    class UnresponsiveServiceFakeLLM(PromptPreparingFakeLLM):
        def tokenize(self, text: str) -> tuple[int, ...]:
            raise EngineCommandTimeout(timeout_message)

    fake = UnresponsiveServiceFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
        ),
        llm=fake,
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 4,
        },
    )

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["error"]["code"] == "engine_unavailable"
    assert body["error"]["type"] == "server_error"
    assert "method=tokenize" in body["error"]["message"]
    assert "budget_s=300.0" in body["error"]["message"]
    assert "driver_command=control:prepare" in body["error"]["message"]
    assert body["error"]["hipengine"]["exception_type"] == "EngineCommandTimeout"


@pytest.mark.parametrize("error_class", [EngineServiceClosed, EngineCommandTimeout])
@pytest.mark.parametrize("chat,n", [(False, 1), (True, 1), (True, 2)])
def test_closed_engine_service_mid_stream_ends_with_a_typed_error_event(error_class, chat, n) -> None:
    """A service that closes after the first chunk still names itself."""

    class ClosingStreamFakeLLM(FakeLLM):
        supports_stream_many = True

        def stream(self, prompt, sampling_params):
            yield "partial"
            raise error_class(
                "engine service is closed after a fatal execution failure "
                "(RuntimeError: device fault)"
            )

        def stream_many_detailed(self, prompts, sampling_params):
            for text in self.stream(prompts[0], sampling_params):
                yield GenerationStreamChunk(
                    text=text,
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1, generated_tokens=1, row_index=0,
                        phase="answer", sampler_mode="greedy_fast", execution_path="test",
                    ),
                )

    fake = ClosingStreamFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
        ),
        llm=fake,
    )
    client = TestClient(app, raise_server_exceptions=False)

    with client.stream(
        "POST",
        "/v1/chat/completions" if chat else "/v1/completions",
        json={
            "model": "fake-model",
            **({"messages": [{"role": "user", "content": "hi"}]} if chat else {"prompt": "hi"}),
            "max_tokens": 4,
            "stream": True,
            "n": n,
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    errors = [event["error"] for event in events if isinstance(event.get("error"), dict)]
    assert errors, body
    assert errors[-1]["code"] == "engine_unavailable"
    assert errors[-1]["hipengine"]["status_code"] == 503
    assert errors[-1]["hipengine"]["retryable"] is True


@pytest.mark.parametrize("stream", [False, True])
def test_completion_tokenize_timeout_is_engine_unavailable(stream) -> None:
    class UnresponsiveFake(PromptPreparingFakeLLM):
        def tokenize(self, text):
            raise EngineCommandTimeout("method=tokenize budget_s=300.0 driver_alive=yes")

    app = create_app(
        ServerConfig(model="fake", served_model_name="fake-model", eager_load=False),
        llm=UnresponsiveFake(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "fake-model", "prompt": "hi", "max_tokens": 4, "stream": stream},
        )
    if stream:
        assert response.status_code == 200
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        error = next(event["error"] for event in events if "error" in event)
        assert response.text.endswith("data: [DONE]\n\n")
    else:
        assert response.status_code == 503
        error = response.json()["error"]
    assert error["code"] == "engine_unavailable"
    assert error["hipengine"]["status_code"] == 503
    assert error["hipengine"]["retryable"] is True
    assert "method=tokenize" in error["message"]


def test_unexpected_server_fault_answers_with_the_openai_error_shape() -> None:
    """Any escaped fault keeps a JSON body and a code instead of a bare 500."""

    class ExplodingFakeLLM(FakeLLM):
        def prepare(self, *, max_sequence_length=None, sampling_params):
            raise RuntimeError("resident workspace bookkeeping exploded")

    fake = ExplodingFakeLLM(outputs=["never served"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
        ),
        llm=fake,
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 4,
        },
    )

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["type"] == "server_error"
    assert "resident workspace bookkeeping exploded" in body["error"]["message"]
    assert body["error"]["hipengine"]["retryable"] is False
    assert body["error"]["hipengine"]["exception_type"] == "RuntimeError"


def test_completion_timeout_returns_deadline_error_and_server_reuses() -> None:
    fake = DelayedFakeLLM(outputs=["late", "ok"], generate_delay_s=0.03)
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    timed_out = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "slow", "max_tokens": 1, "timeout_ms": 1},
    )

    assert timed_out.status_code == 408
    error = timed_out.json()["error"]
    assert error["type"] == "timeout_error"
    assert error["code"] == "deadline_exceeded"
    assert error["param"] == "timeout_ms"
    assert error["hipengine"] == {
        "code": "deadline_exceeded",
        "status_code": 408,
        "retryable": True,
    }
    assert error["finish_details"] == {"reason": "deadline_exceeded", "deadline_exceeded": True}

    # With very short deadlines the worker may time out before the threadpool
    # starts it, or it may unwind later. The public contract is server reuse.
    fake.generate_delay_s = 0.0
    fake.outputs = ["ok"]
    reused = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "after", "max_tokens": 1},
    )

    assert reused.status_code == 200
    assert reused.json()["choices"][0]["text"] == "ok"


def test_backend_deadline_exception_maps_to_completion_408() -> None:
    fake = BackendDeadlineFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "slow", "max_tokens": 4, "timeout_ms": 5000},
    )

    assert response.status_code == 408
    assert fake.calls[0][1].deadline_at is not None
    error = response.json()["error"]
    assert error["type"] == "timeout_error"
    assert error["code"] == "deadline_exceeded"
    assert error["param"] == "timeout_ms"
    assert error["finish_details"] == {"reason": "deadline_exceeded", "deadline_exceeded": True}


def test_kv_admission_rejection_maps_to_retryable_completion_429() -> None:
    fake = AdmissionRejectedFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            queue_retry_after_seconds=3,
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "pressure", "max_tokens": 4},
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "3"
    error = response.json()["error"]
    assert error["type"] == "rate_limit_error"
    assert error["code"] == "engine_busy"
    assert error["hipengine"]["retryable"] is True
    assert error["hipengine"]["routing"]["overload_source"] == "kv_pool_capacity"
    assert error["hipengine"]["routing"]["admission"] == {
        "resource": "device_kv_pool",
        "requested_units": 17,
        "current_units": 129,
        "capacity_units": 129,
    }


def test_backend_execution_failed_maps_to_request_scoped_500() -> None:
    """A contained failure is this request's failure, not the server's."""

    fake = BackendExecutionFailedFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "fail", "max_tokens": 4},
    )

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["type"] == "server_error"
    assert error["code"] == "execution_failed"
    assert error["hipengine"]["retryable"] is True
    assert error["hipengine"]["execution"] == {
        "phase": "decode",
        "request_ids": [7],
        "work_kind": "decode_batch",
        "state_mutation": "none",
        "cause_type": "ValueError",
        "reason": "sampler row 1 exceeds the packed capacity of 0",
    }
    # The engine survived, so readiness must not report a dead service.
    ready = client.get("/ready").json()
    assert ready["engine_service"] is None
    assert ready["status"] != "unhealthy"
    assert not any("engine service" in item for item in ready["diagnostics"])


def test_backend_execution_failed_streams_request_scoped_error_chunk() -> None:
    fake = BackendExecutionFailedFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/v1/completions",
        json={"model": "fake-model", "prompt": "fail", "max_tokens": 4, "stream": True},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    payloads = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
    error_chunk = next(
        json.loads(payload)
        for payload in payloads
        if payload != "[DONE]"
        and json.loads(payload)["choices"][0]["finish_reason"] == "error"
    )
    error = error_chunk["error"]
    assert error["code"] == "execution_failed"
    assert error["hipengine"]["retryable"] is True
    assert error["hipengine"]["execution"]["phase"] == "decode"
    assert error["hipengine"]["execution"]["request_ids"] == [7]
    assert error["hipengine"]["execution"]["state_mutation"] == "none"
    assert body.rstrip().endswith("data: [DONE]")


def test_backend_cancelled_exception_maps_to_completion_499() -> None:
    fake = BackendCancelledFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "cancel", "max_tokens": 4},
    )

    assert response.status_code == 499
    assert fake.calls[0][1].cancellation_token is not None
    assert fake.calls[0][1].cancellation_token.cancelled is True
    error = response.json()["error"]
    assert error["type"] == "cancelled_error"
    assert error["code"] == "cancelled"
    assert error["finish_details"] == {"reason": "cancelled", "cancelled": True}


def test_backend_deadline_finish_detail_maps_to_chat_408() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="deadline_exceeded", deadline_exceeded=True),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "slow"}],
            "max_tokens": 4,
            "timeout_ms": 5000,
        },
    )

    assert response.status_code == 408
    assert fake.calls[0][1].deadline_at is not None
    error = response.json()["error"]
    assert error["code"] == "deadline_exceeded"
    assert error["finish_details"] == {"reason": "deadline_exceeded", "deadline_exceeded": True}


def test_streaming_completion_timeout_emits_error_and_done() -> None:
    fake = DelayedFakeLLM(outputs=["ok"], stream_chunks=["late"], stream_delay_s=0.03)
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "slow", "max_tokens": 1, "stream": True, "timeout_ms": 1},
    )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    payloads = _sse_payloads(response.text)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["choices"][0]["finish_reason"] == "error"
    assert payload["choices"][0]["finish_details"] == {
        "reason": "deadline_exceeded",
        "deadline_exceeded": True,
    }
    assert payload["error"]["type"] == "timeout_error"
    assert payload["error"]["code"] == "deadline_exceeded"
    assert payload["error"]["param"] == "timeout_ms"
    assert payload["error"]["hipengine"] == {
        "code": "deadline_exceeded",
        "status_code": 408,
        "retryable": True,
    }
    assert payload["error"]["finish_details"] == payload["choices"][0]["finish_details"]


def test_streaming_completion_timeout_can_include_hipengine_error_metadata() -> None:
    fake = DelayedFakeLLM(outputs=["ok"], stream_chunks=["late"], stream_delay_s=0.03)
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "slow",
            "max_tokens": 1,
            "stream": True,
            "timeout_ms": 1,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["hipengine"]["event"] == "error"
    assert isinstance(payload["hipengine"]["timing"]["elapsed_ms"], float)
    assert payload["choices"][0]["hipengine"] == {
        "phase": "done",
        "finish_details": {"reason": "deadline_exceeded", "deadline_exceeded": True},
    }
    assert payload["error"]["finish_details"] == payload["choices"][0]["finish_details"]


def test_streaming_chat_backend_deadline_exception_emits_error_and_done() -> None:
    fake = BackendDeadlineFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "slow"}],
            "max_tokens": 4,
            "stream": True,
            "timeout_ms": 5000,
        },
    )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert fake.calls[0][1].deadline_at is not None
    payloads = _sse_payloads(response.text)
    payload = next(item for item in payloads if item.get("error"))
    assert payload["choices"][0]["finish_reason"] == "error"
    assert payload["choices"][0]["finish_details"] == {
        "reason": "deadline_exceeded",
        "deadline_exceeded": True,
    }
    assert payload["error"]["type"] == "timeout_error"
    assert payload["error"]["code"] == "deadline_exceeded"
    assert payload["error"]["param"] == "timeout_ms"


def test_streaming_kv_admission_rejection_emits_retryable_429_and_done() -> None:
    fake = AdmissionRejectedFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "pressure", "max_tokens": 4, "stream": True},
    )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    payload = next(item for item in _sse_payloads(response.text) if item.get("error"))
    assert payload["choices"][0]["finish_reason"] == "error"
    assert payload["error"]["type"] == "rate_limit_error"
    assert payload["error"]["code"] == "engine_busy"
    assert payload["error"]["hipengine"]["status_code"] == 429
    assert payload["error"]["hipengine"]["retryable"] is True
    assert payload["error"]["hipengine"]["routing"]["overload_source"] == "kv_pool_capacity"


def test_streaming_completion_backend_cancelled_exception_emits_error_and_done() -> None:
    fake = BackendCancelledFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "cancel", "max_tokens": 4, "stream": True},
    )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert fake.calls[0][1].cancellation_token is not None
    assert fake.calls[0][1].cancellation_token.cancelled is True
    payloads = _sse_payloads(response.text)
    payload = next(item for item in payloads if item.get("error"))
    assert payload["choices"][0]["finish_reason"] == "error"
    assert payload["choices"][0]["finish_details"] == {
        "reason": "cancelled",
        "cancelled": True,
    }
    assert payload["error"]["type"] == "cancelled_error"
    assert payload["error"]["code"] == "cancelled"


def test_streaming_chat_timeout_can_include_hipengine_error_metadata() -> None:
    fake = DelayedFakeLLM(outputs=["ok"], stream_chunks=["late"], stream_delay_s=0.1)
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "timeout_ms": 50,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert payloads[0]["hipengine"]["event"] == "role"
    payload = next(item for item in payloads if item.get("error"))
    assert payload["hipengine"]["event"] == "error"
    assert isinstance(payload["hipengine"]["timing"]["elapsed_ms"], float)
    assert payload["choices"][0]["hipengine"] == {
        "phase": "done",
        "finish_details": {"reason": "deadline_exceeded", "deadline_exceeded": True},
    }
    assert payload["error"]["finish_details"] == payload["choices"][0]["finish_details"]


def test_server_rejects_requests_beyond_preallocated_context() -> None:
    fake = FakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=5,
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "one two three four", "max_tokens": 2},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert error["hipengine"] == {
        "code": "context_overflow",
        "status_code": 400,
        "legacy_code": "context_length_exceeded",
        "retryable": False,
        "routing": {
            **_routing_metadata(),
            "matched": True,
            "reason": "context_overflow",
        },
    }
    assert error["fit_context"] == {
        "prompt_tokens": 4,
        "max_context_tokens": 5,
        "effective_max_tokens": 2,
        "max_allowed_max_tokens": 0,
        "recommended_max_tokens": 0,
        "required_context_tokens": 7,
        "overflow_tokens": 2,
        "fits": False,
        "clear_policy": "reject",
        "would_truncate": False,
        "would_drop": [],
    }
    fit = client.post(
        "/v1/hipengine/fit_context",
        json={"text": "one two three four", "max_tokens": 2},
    )
    assert fit.status_code == 200
    fit_body = fit.json()
    for key, value in error["fit_context"].items():
        assert fit_body[key] == value
    assert fake.calls == []


def test_streaming_completion_context_overflow_preserves_error_diagnostics() -> None:
    fake = FakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=5,
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "one two three four",
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["choices"][0]["finish_reason"] == "error"
    assert payload["error"]["code"] == "context_length_exceeded"
    assert payload["error"]["hipengine"] == {
        "code": "context_overflow",
        "status_code": 400,
        "legacy_code": "context_length_exceeded",
        "retryable": False,
        "routing": {
            **_routing_metadata(),
            "matched": True,
            "reason": "context_overflow",
        },
    }
    assert payload["error"]["fit_context"] == {
        "prompt_tokens": 4,
        "max_context_tokens": 5,
        "effective_max_tokens": 2,
        "max_allowed_max_tokens": 0,
        "recommended_max_tokens": 0,
        "required_context_tokens": 7,
        "overflow_tokens": 2,
        "fits": False,
        "clear_policy": "reject",
        "would_truncate": False,
        "would_drop": [],
    }
    assert payload["hipengine"]["event"] == "error"
    assert payload["hipengine"]["routing"] == _routing_metadata()
    assert "data: [DONE]" in response.text
    assert fake.calls == []


def test_server_rejects_request_kv_policy_mismatch() -> None:
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            kv_storage="int8_per_token_head",
        ),
        llm=FakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "kv_storage": "bf16"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_kv_policy"


def test_server_rejects_wrong_model_and_unsupported_options(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="uvicorn.error")
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    wrong_model = client.post(
        "/v1/completions",
        json={"model": "other", "prompt": "hello"},
    )
    assert wrong_model.status_code == 404
    assert wrong_model.json()["error"]["code"] == "model_not_found"
    assert wrong_model.json()["error"]["hipengine"] == {
        "code": "model_unavailable",
        "status_code": 404,
        "legacy_code": "model_not_found",
        "retryable": False,
        "routing": {
            "requested_model": "other",
            "served_model": None,
            "configured_model": "fake-model",
            "fallback_used": False,
            "policy": "single_model_exact",
            "loaded_model_count": 1,
            "multiple_models": False,
            "matched": False,
            "reason": "model_unavailable",
        },
    }

    chat_wrong_model = client.post(
        "/v1/chat/completions",
        json={"model": "other", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert chat_wrong_model.status_code == 404
    assert chat_wrong_model.json()["error"]["code"] == "model_not_found"
    assert chat_wrong_model.json()["error"]["hipengine"]["routing"] == wrong_model.json()["error"]["hipengine"]["routing"]
    assert fake.calls == []

    schema_violation = client.post(
        "/v1/completions",
        json={"model": "fake-model", "max_tokens": 1},
    )
    assert schema_violation.status_code == 400
    assert schema_violation.json()["error"]["code"] == "invalid_request"
    assert schema_violation.json()["error"]["param"] == "prompt"
    assert schema_violation.json()["error"]["hipengine"] == {
        "code": "schema_violation",
        "status_code": 400,
        "legacy_code": "invalid_request",
        "retryable": False,
    }

    unsupported_chat_top_logprobs = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "top_logprobs": 1,
        },
    )
    assert unsupported_chat_top_logprobs.status_code == 400
    assert unsupported_chat_top_logprobs.json()["error"]["param"] == "top_logprobs"

    unsupported_extra = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "typical_p": 0.9},
    )
    assert unsupported_extra.status_code == 400
    assert unsupported_extra.json()["error"]["code"] == "unsupported_parameter"
    assert unsupported_extra.json()["error"]["hipengine"] == {
        "code": "unsupported_parameter",
        "status_code": 400,
        "retryable": False,
    }
    assert unsupported_extra.json()["error"]["param"] == "typical_p"
    assert "REQUEST_FAILED: POST /v1/completions status=404 code=model_not_found" in caplog.text
    assert "REQUEST_FAILED: POST /v1/completions status=400 code=unsupported_parameter" in caplog.text
    assert "param=typical_p" in caplog.text


@pytest.mark.parametrize(
    ("endpoint", "payload", "param"),
    [
        (
            "/v1/completions",
            {
                "model": "fake-model",
                "prompt": "hello",
                "session": {"id": "session_123"},
            },
            "session.id",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "session": {"id": "session_123"},
            },
            "stream",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "n": 2,
                "session": {"id": "session_123"},
            },
            "n",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "continuation_id": "gen_missing",
                "session": {"id": "session_123"},
            },
            "messages",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "grammar": {"type": "json"},
            },
            "grammar",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "guided_grammar": "root ::= 'ok'",
            },
            "guided_grammar",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "guided_decoding_backend": "outlines",
            },
            "guided_decoding_backend",
        ),
    ],
)
def test_server_rejects_known_unsupported_agentic_fields(endpoint, payload, param) -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(endpoint, json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == param
    if param in {"grammar", "guided_grammar", "guided_decoding_backend"}:
        assert response.json()["error"]["hipengine"] == {
            "code": "unsupported_parameter",
            "status_code": 400,
            "retryable": False,
            "routing": {
                **_routing_metadata(),
                "matched": True,
                "reason": "unsupported_grammar",
                "unsupported_field": param,
                "unsupported_capability": "grammar",
            },
        }
    assert fake.calls == []


def test_capabilities_advertised_unsupported_fields_are_rejected_before_generation() -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    capabilities = client.get("/v1/hipengine/capabilities").json()

    unsupported_fields = capabilities["unsupported_fields"]
    assert unsupported_fields == capabilities["features"]["grammars"]["unsupported_fields"]

    field_values = {
        "grammar": {"type": "json"},
        "guided_grammar": "root ::= 'ok'",
        "guided_decoding_backend": "outlines",
    }
    assert set(unsupported_fields) == set(field_values)

    for endpoint in ("/v1/completions", "/v1/chat/completions"):
        for field in unsupported_fields:
            payload = (
                {"model": "fake-model", "prompt": "hello"}
                if endpoint == "/v1/completions"
                else {"model": "fake-model", "messages": [{"role": "user", "content": "hello"}]}
            )
            payload[field] = field_values[field]

            response = client.post(endpoint, json=payload)

            assert response.status_code == 400
            assert response.json()["error"]["code"] == "unsupported_parameter"
            assert response.json()["error"]["param"] == field
            assert response.json()["error"]["hipengine"] == {
                "code": "unsupported_parameter",
                "status_code": 400,
                "retryable": False,
                "routing": {
                    **_routing_metadata(),
                    "matched": True,
                    "reason": "unsupported_grammar",
                    "unsupported_field": field,
                    "unsupported_capability": "grammar",
                },
            }

    assert fake.calls == []


def test_resident_prepare_refusal_returns_a_structured_remedy() -> None:
    """An operator-level refusal must not surface as an opaque 500.

    A wrong slot/context combination is refused when the resident session is
    prepared. Every endpoint that touches the engine, including capabilities and
    chat preparation, must report the refusal and its remedy.
    """

    class RefusingLLM(FakeLLM):
        def prepare(self, *, max_sequence_length: int | None = None, sampling_params: SamplingParams) -> int:
            raise ResidentPreparationError(
                "resident capacity must be within 1..2 for max_sequence_length=262144"
            )

        @property
        def supports_speculative(self) -> bool:
            # The real engine builds the text generator to answer this, so a
            # capability probe reaches the same refusal as a request does.
            raise ResidentPreparationError(
                "resident capacity must be within 1..2 for max_sequence_length=262144"
            )

    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            kv_storage="bf16",
            kv_scale_dtype="fp32",
            eager_load=True,
            # A configured provider makes the capability probe reach the engine,
            # which is how the refusal escaped as a bare 500 in production.
            speculative_provider="qwen4_exp_mtp",
            draft_model="fake-draft.gguf",
        ),
        llm=RefusingLLM(outputs=["never"]),
    )
    client = TestClient(app, raise_server_exceptions=False)

    for path in ("/v1/hipengine/capabilities",):
        response = client.get(path)
        assert response.status_code == 503, (path, response.status_code, response.text)
        body = response.json()
        assert body["error"]["code"] == "resident_prepare_failed", body
        assert "resident capacity must be within 1..2" in body["error"]["message"], body
        assert body["error"]["hipengine"]["guidance"], body

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
    )
    assert response.status_code == 503, (response.status_code, response.text)
    assert response.json()["error"]["code"] == "resident_prepare_failed"

    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["ready"] is False
