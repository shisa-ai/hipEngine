"""OpenAI server integration tests: logprobs.

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
    _chat_session_message_copy,
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
    FakeLLM,
)
from tests.support.server_api_support import (
    _sse_payloads,
    _stateless_finish_details,
)


def test_completions_endpoint_returns_openai_logprobs() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="alpha beta",
                token_logprobs=(
                    TokenLogprob(
                        token_id=1,
                        token_text="alpha",
                        logprob=-0.25,
                        top_logprobs=((1, "alpha", -0.25), (2, "omega", -1.5)),
                    ),
                    TokenLogprob(
                        token_id=3,
                        token_text=" beta",
                        logprob=-0.5,
                        top_logprobs=((3, " beta", -0.5),),
                    ),
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 2, "logprobs": 2},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == "alpha beta"
    assert choice["logprobs"]["tokens"] == ["alpha", " beta"]
    assert choice["logprobs"]["token_logprobs"] == [-0.25, -0.5]
    assert choice["logprobs"]["top_logprobs"][0] == {"alpha": -0.25, "omega": -1.5}
    assert fake.calls[0][1].logprobs is True
    assert fake.calls[0][1].top_logprobs == 2


def test_completion_logprobs_omitted_selected_score_reports_reason() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="alpha beta",
                token_logprobs=(
                    TokenLogprob(token_id=1, token_text="alpha", logprob=None),
                    TokenLogprob(token_id=3, token_text=" beta", logprob=-0.5),
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 2, "logprobs": 1},
    )

    assert response.status_code == 200
    logprobs = response.json()["choices"][0]["logprobs"]
    assert logprobs["tokens"] == ["alpha", " beta"]
    assert logprobs["token_logprobs"] == [None, -0.5]
    assert logprobs["hipengine"] == {
        "omitted_token_logprobs": [
            {
                "index": 0,
                "token": "alpha",
                "token_id": 1,
                "reason": "backend_omitted_logprob",
            }
        ]
    }


def test_completion_logprobs_missing_backend_metadata_returns_unsupported_feature() -> None:
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=["alpha"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1, "logprobs": 1},
    )

    assert response.status_code == 501
    error = response.json()["error"]
    assert error["code"] == "unsupported_feature"
    assert error["param"] == "logprobs"
    assert error["hipengine"] == {
        "code": "unsupported_feature",
        "status_code": 501,
        "retryable": False,
    }


def test_completions_endpoint_echo_logprobs_shift_generated_offsets() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text=" world",
                token_logprobs=(
                    TokenLogprob(token_id=7, token_text=" world", logprob=-0.125),
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1, "echo": True, "logprobs": 0},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == "hello world"
    assert choice["logprobs"]["tokens"] == ["hello", " world"]
    assert choice["logprobs"]["token_logprobs"] == [None, -0.125]
    assert choice["logprobs"]["text_offset"] == [0, 5]
    assert choice["logprobs"]["hipengine"] == {
        "omitted_token_logprobs": [
            {
                "index": 0,
                "token": "hello",
                "token_id": None,
                "reason": "prompt_logprob_unavailable",
            }
        ]
    }


def test_streaming_completion_returns_logprobs_from_buffered_path() -> None:
    fake = FakeLLM(
        outputs=["should-not-stream"],
        stream_chunks=["wrong"],
        detailed_outputs=[
            GenerationOutput(
                text="alpha",
                token_logprobs=(
                    TokenLogprob(token_id=1, token_text="alpha", logprob=-0.25, top_logprobs=((1, "alpha", -0.25),)),
                ),
            )
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1, "stream": True, "logprobs": 1},
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert payloads[0]["choices"][0]["text"] == "alpha"
    assert payloads[0]["choices"][0]["logprobs"]["token_logprobs"] == [-0.25]
    assert payloads[-1]["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert fake.stream_calls == []
    assert fake.calls[0][1].logprobs is True


def test_streaming_completion_returns_live_chunk_logprobs_when_backend_supports_metadata() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "alpha",
                token_logprobs=(
                    TokenLogprob(
                        token_id=1,
                        token_text="alpha",
                        logprob=-0.25,
                        top_logprobs=((1, "alpha", -0.25),),
                    ),
                ),
            )
        ],
    )
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1, "stream": True, "logprobs": 1},
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    delta = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0].get("text"))
    choice = delta["choices"][0]
    assert choice["text"] == "alpha"
    assert choice["logprobs"] == {
        "tokens": ["alpha"],
        "token_logprobs": [-0.25],
        "top_logprobs": [{"alpha": -0.25}],
        "text_offset": [0],
    }
    assert fake.stream_calls
    assert fake.calls[0][1].logprobs is True


def test_streaming_completion_live_logprobs_omitted_selected_score_reports_reason() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "alpha",
                token_logprobs=(TokenLogprob(token_id=1, token_text="alpha", logprob=None),),
            )
        ],
    )
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1, "stream": True, "logprobs": 1},
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    delta = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0].get("text"))
    assert delta["choices"][0]["logprobs"]["token_logprobs"] == [None]
    assert delta["choices"][0]["logprobs"]["hipengine"] == {
        "omitted_token_logprobs": [
            {
                "index": 0,
                "token": "alpha",
                "token_id": 1,
                "reason": "backend_omitted_logprob",
            }
        ]
    }


def test_streaming_completion_logprobs_missing_backend_metadata_returns_unsupported_feature() -> None:
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=["alpha"], stream_chunks=["alpha"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1, "stream": True, "logprobs": 1},
    )

    assert response.status_code == 501
    error = response.json()["error"]
    assert error["code"] == "unsupported_feature"
    assert error["param"] == "logprobs"
    assert error["hipengine"] == {
        "code": "unsupported_feature",
        "status_code": 501,
        "retryable": False,
    }


def test_chat_completion_returns_openai_logprobs() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="assistant reply",
                token_logprobs=(
                    TokenLogprob(
                        token_id=4,
                        token_text="assistant",
                        logprob=-0.1,
                        top_logprobs=((4, "assistant", -0.1),),
                    ),
                    TokenLogprob(token_id=5, token_text=" reply", logprob=-0.2),
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "logprobs": True,
            "top_logprobs": 1,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"]["content"] == "assistant reply"
    assert choice["logprobs"]["content"][0]["token"] == "assistant"
    assert choice["logprobs"]["content"][0]["top_logprobs"] == [
        {"token": "assistant", "logprob": -0.1, "bytes": None}
    ]
    assert fake.calls[0][1].logprobs is True
    assert fake.calls[0][1].top_logprobs == 1


def test_chat_logprobs_use_visible_content_tokens_after_reasoning() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="<think>plan</think>answer",
                token_logprobs=(
                    TokenLogprob(token_id=3, token_text="<think>plan</think>", logprob=-0.1),
                    TokenLogprob(
                        token_id=4,
                        token_text="answer",
                        logprob=-0.2,
                        top_logprobs=((4, "answer", -0.2),),
                    ),
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "logprobs": True,
            "top_logprobs": 1,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"] == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "plan",
    }
    assert choice["logprobs"] == {
        "content": [
            {
                "token": "answer",
                "logprob": -0.2,
                "bytes": None,
                "top_logprobs": [{"token": "answer", "logprob": -0.2, "bytes": None}],
            }
        ],
        "refusal": None,
    }


def test_chat_logprobs_omitted_selected_score_reports_reason() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="assistant reply",
                token_logprobs=(
                    TokenLogprob(token_id=4, token_text="assistant", logprob=None),
                    TokenLogprob(token_id=5, token_text=" reply", logprob=-0.2),
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "logprobs": True,
        },
    )

    assert response.status_code == 200
    logprobs = response.json()["choices"][0]["logprobs"]
    assert [entry["logprob"] for entry in logprobs["content"]] == [None, -0.2]
    assert logprobs["hipengine"] == {
        "omitted_token_logprobs": [
            {
                "index": 0,
                "token": "assistant",
                "token_id": 4,
                "reason": "backend_omitted_logprob",
            }
        ]
    }


def test_chat_logprobs_missing_backend_metadata_returns_unsupported_feature() -> None:
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=["assistant"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "logprobs": True,
        },
    )

    assert response.status_code == 501
    error = response.json()["error"]
    assert error["code"] == "unsupported_feature"
    assert error["param"] == "logprobs"
    assert error["hipengine"] == {
        "code": "unsupported_feature",
        "status_code": 501,
        "retryable": False,
    }


def test_streaming_chat_completion_returns_logprobs_from_buffered_path() -> None:
    fake = FakeLLM(
        outputs=["should-not-stream"],
        stream_chunks=["wrong"],
        detailed_outputs=[
            GenerationOutput(
                text="assistant",
                token_logprobs=(
                    TokenLogprob(token_id=4, token_text="assistant", logprob=-0.1),
                ),
            )
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "stream": True,
            "logprobs": True,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content_chunks = [payload for payload in payloads if payload.get("choices") and payload["choices"][0]["delta"].get("content")]
    assert content_chunks[0]["choices"][0]["delta"] == {"content": "assistant"}
    assert content_chunks[0]["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.1
    assert payloads[-1]["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert fake.stream_calls == []
    assert fake.calls[0][1].logprobs is True


def test_streaming_chat_completion_returns_live_chunk_logprobs_when_backend_supports_metadata() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "assistant",
                token_logprobs=(
                    TokenLogprob(
                        token_id=4,
                        token_text="assistant",
                        logprob=-0.1,
                        top_logprobs=((4, "assistant", -0.1),),
                    ),
                ),
            )
        ],
    )
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "stream": True,
            "logprobs": True,
            "top_logprobs": 1,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content = next(
        payload for payload in payloads if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    )
    choice = content["choices"][0]
    assert choice["delta"] == {"content": "assistant"}
    assert choice["logprobs"] == {
        "content": [
            {
                "token": "assistant",
                "logprob": -0.1,
                "bytes": None,
                "top_logprobs": [{"token": "assistant", "logprob": -0.1, "bytes": None}],
            }
        ],
        "refusal": None,
    }
    assert fake.stream_calls
    assert fake.calls[0][1].logprobs is True
    assert fake.calls[0][1].top_logprobs == 1


def test_streaming_chat_completion_n_uses_scheduler_token_chunks_for_buffered_logprobs() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            outputs = [
                GenerationOutput(
                    text="AB",
                    token_logprobs=(
                        TokenLogprob(token_id=100, token_text="A", logprob=-0.1),
                        TokenLogprob(token_id=101, token_text="B", logprob=-0.2),
                    ),
                    finish_details=FinishDetails(reason="length", length_limit=2, sampler_mode="host_logits_sample"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=0,
                        phase="done",
                        sampler_mode="host_logits_sample",
                    ),
                ),
                GenerationOutput(
                    text="CD",
                    token_logprobs=(
                        TokenLogprob(token_id=200, token_text="C", logprob=-0.3),
                        TokenLogprob(token_id=201, token_text="D", logprob=-0.4),
                    ),
                    finish_details=FinishDetails(reason="length", length_limit=2, sampler_mode="host_logits_sample"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=1,
                        phase="done",
                        sampler_mode="host_logits_sample",
                    ),
                ),
            ]
            chunk_texts = (("A", "B"), ("C", "D"))
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": request_id,
                        "token_index": token_index,
                        "token_id": 100 + request_id * 100 + token_index,
                        "finished": token_index == 1,
                        "chunk": {
                            "text": text,
                            "token_logprobs": [
                                {
                                    "token_id": 100 + request_id * 100 + token_index,
                                    "token_text": text,
                                    "logprob": -(request_id * 2 + token_index + 1) / 10,
                                    "top_logprobs": [
                                        {
                                            "token_id": 100 + request_id * 100 + token_index,
                                            "token_text": text,
                                            "logprob": -(request_id * 2 + token_index + 1) / 10,
                                        }
                                    ],
                                }
                            ],
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=token_index + 1,
                                row_index=request_id,
                                request_id=str(request_id),
                                phase="answer",
                                sampler_mode="host_logits_sample",
                                execution_path="scheduler_native_packed_prefill_serial_host_sampler_decode",
                            ).to_json_dict(),
                        },
                    }
                    for request_id, row in enumerate(chunk_texts)
                    for token_index, text in enumerate(row)
                ]
            }
            return outputs[: len(prompt_tuple)]

    fake = SchedulerChunkFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "n": 2,
            "max_tokens": 2,
            "stream": True,
            "logprobs": True,
            "top_logprobs": 1,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    deltas = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] is None
    ]
    content_deltas = [choice for choice in deltas if "content" in choice["delta"]]
    assert [(choice["index"], choice["delta"]["content"]) for choice in content_deltas] == [
        (0, "A"),
        (0, "B"),
        (1, "C"),
        (1, "D"),
    ]
    assert [
        choice["logprobs"]["content"][0]["logprob"]
        for choice in content_deltas
    ] == pytest.approx([-0.1, -0.2, -0.3, -0.4])
    assert [
        choice["logprobs"]["content"][0]["top_logprobs"][0]["token"]
        for choice in content_deltas
    ] == ["A", "B", "C", "D"]
    assert {
        choice["hipengine"]["decode_state"]["execution_path"]
        for choice in content_deltas
    } == {"scheduler_native_packed_prefill_serial_host_sampler_decode"}
    done = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] == "length"
    ]
    assert [choice["index"] for choice in done] == [0, 1]
    assert fake.stream_calls == []


def test_streaming_chat_completion_uses_scheduler_reasoning_private_logprobs() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            text = "<think>r0</think>A"
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": 0,
                        "token_index": token_index,
                        "token_id": 300 + token_index,
                        "finished": token_index == 1,
                        "chunk": {
                            "text": chunk_text,
                            "token_logprobs": [
                                {
                                    "token_id": 300 + token_index,
                                    "token_text": chunk_text,
                                    "logprob": -(token_index + 1) / 10,
                                }
                            ],
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=token_index + 1,
                                row_index=0,
                                request_id="0",
                                phase="answer",
                                sampler_mode="host_logits_sample",
                                execution_path="scheduler_should_not_surface",
                            ).to_json_dict(),
                        },
                    }
                    for token_index, chunk_text in enumerate(("<think>r0</think>", "A"))
                ]
            }
            return [
                GenerationOutput(
                    text=text,
                    token_logprobs=(
                        TokenLogprob(token_id=300, token_text="<think>r0</think>", logprob=-0.1),
                        TokenLogprob(token_id=301, token_text="A", logprob=-0.2),
                    ),
                    finish_details=FinishDetails(reason="stop", sampler_mode="host_logits_sample"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=0,
                        phase="done",
                        sampler_mode="host_logits_sample",
                        execution_path="buffered_final_metadata",
                    ),
                )
            ][: len(prompt_tuple)]

    fake = SchedulerChunkFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "stream": True,
            "logprobs": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    deltas = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] is None
    ]
    assert [(choice["index"], choice["delta"]) for choice in deltas] == [
        (0, {"role": "assistant"}),
        (0, {"reasoning_content": "r0"}),
        (0, {"content": "A"}),
    ]
    assert "<think>" not in json.dumps([choice["delta"] for choice in deltas])
    assert {
        choice.get("hipengine", {}).get("decode_state", {}).get("execution_path")
        for choice in deltas
        if choice["delta"].get("content") or choice["delta"].get("reasoning_content")
    } == {"scheduler_should_not_surface"}
    reasoning = next(choice for choice in deltas if choice["delta"].get("reasoning_content"))
    assert "logprobs" not in reasoning
    assert reasoning["hipengine"]["decode_state"]["phase"] == "think"
    assert reasoning["hipengine"]["reasoning_logprobs"] == {
        "content": [
            {
                "token_id": 300,
                "token": "<think>r0</think>",
                "logprob": -0.1,
                "bytes": None,
                "top_logprobs": [],
            }
        ],
        "public_text": "r0",
        "refusal": None,
    }
    content = next(choice for choice in deltas if choice["delta"].get("content"))
    assert content["hipengine"]["decode_state"]["phase"] == "answer"
    assert content["logprobs"]["content"] == [
        {"token": "A", "logprob": -0.2, "bytes": None, "top_logprobs": []}
    ]
    assert fake.stream_calls == []


def test_streaming_chat_completion_uses_live_reasoning_private_logprobs() -> None:
    fake = FakeLLM(
        stream_chunks=[
            GenerationStreamChunk(
                text="<think>r0</think>",
                token_logprobs=(
                    TokenLogprob(
                        token_id=300,
                        token_text="<think>r0</think>",
                        logprob=-0.1,
                        top_logprobs=((300, "<think>r0</think>", -0.1),),
                    ),
                ),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=1,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
            GenerationStreamChunk(
                text="A",
                token_logprobs=(TokenLogprob(token_id=301, token_text="A", logprob=-0.2),),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=2,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
        ],
    )
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "stream": True,
            "logprobs": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    reasoning = next(
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("reasoning_content")
    )
    content = next(
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    )
    assert reasoning["delta"] == {"reasoning_content": "r0"}
    assert "logprobs" not in reasoning
    assert reasoning["hipengine"]["decode_state"]["phase"] == "think"
    assert reasoning["hipengine"]["decode_state"]["execution_path"] == "live_host_sampler_decode"
    assert reasoning["hipengine"]["reasoning_logprobs"] == {
        "content": [
            {
                "token_id": 300,
                "token": "<think>r0</think>",
                "logprob": -0.1,
                "bytes": None,
                "top_logprobs": [
                    {
                        "token_id": 300,
                        "token": "<think>r0</think>",
                        "logprob": -0.1,
                        "bytes": None,
                    }
                ],
            }
        ],
        "public_text": "r0",
        "refusal": None,
    }
    assert content["delta"] == {"content": "A"}
    assert content["logprobs"]["content"] == [
        {"token": "A", "logprob": -0.2, "bytes": None, "top_logprobs": []}
    ]
    assert fake.stream_calls
    assert fake.calls[0][1].logprobs is True
    assert fake.calls[0][1].top_logprobs == 0


def test_streaming_chat_completion_maps_live_middle_reasoning_span_logprobs() -> None:
    fake = FakeLLM(
        stream_chunks=[
            GenerationStreamChunk(
                text="<think>plan</think>A<thi",
                token_logprobs=(
                    TokenLogprob(token_id=330, token_text="<think>", logprob=-0.1),
                    TokenLogprob(token_id=331, token_text="plan", logprob=-0.2),
                    TokenLogprob(token_id=332, token_text="</think>", logprob=-0.3),
                    TokenLogprob(token_id=333, token_text="A", logprob=-0.4),
                    TokenLogprob(token_id=334, token_text="<thi", logprob=-0.5),
                ),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=5,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
        ],
    )
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "logprobs": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    reasoning = next(
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("reasoning_content")
    )
    content = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    ]
    assert reasoning["delta"] == {"reasoning_content": "plan"}
    assert reasoning["hipengine"]["reasoning_logprobs"] == {
        "content": [
            {
                "token_id": 331,
                "token": "plan",
                "logprob": -0.2,
                "bytes": None,
                "top_logprobs": [],
            }
        ],
        "public_text": "plan",
        "refusal": None,
    }
    assert [choice["delta"] for choice in content] == [{"content": "A"}, {"content": "<thi"}]
    assert content[0]["logprobs"]["content"] == [
        {"token": "A", "logprob": -0.4, "bytes": None, "top_logprobs": []}
    ]
    assert content[1]["logprobs"]["content"] == [
        {"token": "<thi", "logprob": -0.5, "bytes": None, "top_logprobs": []}
    ]


def test_streaming_chat_completion_maps_live_final_content_logprobs() -> None:
    fake = FakeLLM(
        stream_chunks=[
            GenerationStreamChunk(
                text="<t",
                token_logprobs=(TokenLogprob(token_id=310, token_text="<t", logprob=-0.7),),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=1,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
            GenerationStreamChunk(
                text="hi",
                token_logprobs=(TokenLogprob(token_id=311, token_text="hi", logprob=-0.8),),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=2,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
        ],
    )
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "logprobs": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content = next(
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    )
    assert content["delta"] == {"content": "<thi"}
    assert content["logprobs"]["content"] == [
        {"token": "<t", "logprob": -0.7, "bytes": None, "top_logprobs": []},
        {"token": "hi", "logprob": -0.8, "bytes": None, "top_logprobs": []},
    ]
    assert content["hipengine"]["decode_state"]["phase"] == "answer"
    assert content["hipengine"]["decode_state"]["execution_path"] == "live_host_sampler_decode"


def test_streaming_chat_completion_maps_live_final_reasoning_logprobs() -> None:
    fake = FakeLLM(
        stream_chunks=[
            GenerationStreamChunk(
                text="<think>plan",
                token_logprobs=(TokenLogprob(token_id=320, token_text="<think>plan", logprob=-0.1),),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=1,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
            GenerationStreamChunk(
                text="</t",
                token_logprobs=(TokenLogprob(token_id=321, token_text="</t", logprob=-0.2),),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=2,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
            GenerationStreamChunk(
                text="hi",
                token_logprobs=(TokenLogprob(token_id=322, token_text="hi", logprob=-0.3),),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=3,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
        ],
    )
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "logprobs": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    reasoning_deltas = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("reasoning_content")
    ]
    assert [choice["delta"] for choice in reasoning_deltas] == [
        {"reasoning_content": "plan"},
        {"reasoning_content": "</thi"},
    ]
    assert [choice["hipengine"]["decode_state"]["phase"] for choice in reasoning_deltas] == [
        "think",
        "think",
    ]
    assert reasoning_deltas[-1]["hipengine"]["reasoning_logprobs"] == {
        "content": [
            {
                "token_id": 321,
                "token": "</t",
                "logprob": -0.2,
                "bytes": None,
                "top_logprobs": [],
            },
            {
                "token_id": 322,
                "token": "hi",
                "logprob": -0.3,
                "bytes": None,
                "top_logprobs": [],
            }
        ],
        "public_text": "</thi",
        "refusal": None,
    }


def test_streaming_chat_completion_falls_back_for_unmappable_reasoning_logprobs() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            text = "<think>r0</think>A"
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": 0,
                        "token_index": 0,
                        "token_id": 300,
                        "finished": True,
                        "chunk": {
                            "text": text,
                            "token_logprobs": [
                                {
                                    "token_id": 300,
                                    "token_text": text,
                                    "logprob": -0.1,
                                }
                            ],
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=1,
                                row_index=0,
                                request_id="0",
                                phase="answer",
                                sampler_mode="host_logits_sample",
                                execution_path="scheduler_should_not_surface",
                            ).to_json_dict(),
                        },
                    }
                ]
            }
            return [
                GenerationOutput(
                    text=text,
                    token_logprobs=(TokenLogprob(token_id=300, token_text=text, logprob=-0.1),),
                    finish_details=FinishDetails(reason="stop", sampler_mode="host_logits_sample"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=1,
                        row_index=0,
                        phase="done",
                        sampler_mode="host_logits_sample",
                        execution_path="buffered_final_metadata",
                    ),
                )
            ][: len(prompt_tuple)]

    fake = SchedulerChunkFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "stream": True,
            "logprobs": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    deltas = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] is None
    ]
    assert [(choice["index"], choice["delta"]) for choice in deltas] == [
        (0, {"role": "assistant"}),
        (0, {"reasoning_content": "r0"}),
        (0, {"content": "A"}),
    ]
    assert {
        choice.get("hipengine", {}).get("decode_state", {}).get("execution_path")
        for choice in deltas
        if choice["delta"].get("content") or choice["delta"].get("reasoning_content")
    } == {"buffered_final_metadata"}
    content = next(choice for choice in deltas if choice["delta"].get("content"))
    assert content["logprobs"]["content"] == []
    done = next(
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] == "stop"
    )
    diagnostic = done["hipengine"]["withheld_scheduler_logprob_chunks"]
    assert diagnostic == {
        "surface": "chat_logprob_delta",
        "reason": "unmappable_logprobs",
        "public_delta": "buffered_without_scheduler_logprobs",
        "chunk_count": 1,
        "malformed_chunk_count": 0,
        "chunks_with_logprobs": 1,
        "token_logprob_count": 1,
        "raw_text_matches": True,
        "text_bytes": len("<think>r0</think>A".encode("utf-8")),
        "text_sha256": hashlib.sha256("<think>r0</think>A".encode("utf-8")).hexdigest(),
        "chunk_text_bytes": [len("<think>r0</think>A".encode("utf-8"))],
        "execution_paths": ["scheduler_should_not_surface"],
    }
    assert "text" not in diagnostic
    assert fake.stream_calls == []


def test_streaming_chat_live_logprobs_omitted_selected_score_reports_reason() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "assistant",
                token_logprobs=(TokenLogprob(token_id=4, token_text="assistant", logprob=None),),
            )
        ],
    )
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "stream": True,
            "logprobs": True,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content = next(
        payload for payload in payloads if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    )
    assert content["choices"][0]["logprobs"]["content"][0]["logprob"] is None
    assert content["choices"][0]["logprobs"]["hipengine"] == {
        "omitted_token_logprobs": [
            {
                "index": 0,
                "token": "assistant",
                "token_id": 4,
                "reason": "backend_omitted_logprob",
            }
        ]
    }
