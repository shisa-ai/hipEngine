"""OpenAI server integration tests: streaming.

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
    _assert_openai_stream_tool_call_delta_shape,
    _fake_kv_pool_metadata,
    _fake_kv_pool_stats,
    _routing_metadata,
    _sse_payloads,
    _stateless_finish_details,
)


def test_buffered_streaming_completion_preserves_backend_done_decode_state() -> None:
    fake = FakeLLM(
        outputs=["should-not-stream"],
        stream_chunks=["wrong"],
        detailed_outputs=[
            GenerationOutput(
                text="alpha",
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=7,
                    generated_tokens=3,
                    phase="done",
                    sampler_mode="processed_argmax",
                    forced_token_id=77,
                    forced_token_reason="thinking_hard_close",
                    forced_tokens_remaining=0,
                    active_processors=("logit_bias",),
                    sampler_fallback_reason="processed_logits_required",
                    sampler_fast_path_blockers=("logit_bias",),
                    full_vocab_logits_d2h=False,
                    logits_d2h_bytes=0,
                    execution_path="native_sampler",
                    native_compact_prefill=True,
                    native_caware_decode=True,
                    serial_decode_fallback=False,
                    native_sampler_rows=True,
                    timing={"prefill_ms": 3.5, "decode_ms": 1.25},
                ),
            )
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 1,
            "stream": True,
            "echo": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert payloads[0]["choices"][0]["text"] == "helloalpha"
    assert payloads[0]["choices"][0]["hipengine"] == {
        "phase": "answer",
        "tokens": {
            "streamed_tokens": 1,
            "delta_tokens": 1,
            "answer_tokens": 1,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 1,
            "prompt_tokens": 0,
            "generated_tokens": 1,
            "phase": "answer",
            "continuation_eligible": False,
            "answer_tokens": 1,
            "active_processors": ["logit_bias"],
            "sampler_fast_path_blockers": ["logit_bias"],
            "sampler_fallback_reason": "processed_logits_required",
            "sampler_mode": "processed_argmax",
            "full_vocab_logits_d2h": False,
            "logits_d2h_bytes": 0,
            "execution_path": "native_sampler",
            "native_compact_prefill": True,
            "native_caware_decode": True,
            "serial_decode_fallback": False,
            "native_sampler_rows": True,
        },
    }
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["hipengine"] == {
        "phase": "done",
        "decode_state": {
            "row_index": 0,
            "step_index": 3,
            "prompt_tokens": 7,
            "generated_tokens": 3,
            "phase": "done",
            "continuation_eligible": False,
            "forced_token_id": 77,
            "forced_token_reason": "thinking_hard_close",
            "forced_tokens_remaining": 0,
            "active_processors": ["logit_bias"],
            "sampler_fast_path_blockers": ["logit_bias"],
            "sampler_fallback_reason": "processed_logits_required",
            "sampler_mode": "processed_argmax",
            "full_vocab_logits_d2h": False,
            "logits_d2h_bytes": 0,
            "execution_path": "native_sampler",
            "native_compact_prefill": True,
            "native_caware_decode": True,
            "serial_decode_fallback": False,
            "native_sampler_rows": True,
        },
        "timing": {"prefill_ms": 3.5, "decode_ms": 1.25},
        "timing_scope": "choice",
        "group_rows": 1,
        "timing_owner": True,
        "finish_details": _stateless_finish_details("stop"),
    }
    assert done["hipengine"]["timing"]["backend_prefill_ms"] == 3.5
    assert done["hipengine"]["timing"]["backend_decode_ms"] == 1.25
    assert fake.stream_calls == []


def test_buffered_streaming_chat_preserves_backend_done_decode_state() -> None:
    fake = FakeLLM(
        outputs=["should-not-stream"],
        stream_chunks=["wrong"],
        detailed_outputs=[
            GenerationOutput(
                text="first",
                telemetry=GenerationTelemetry.from_decode_counts(
                    row_index=0,
                    prompt_tokens=11,
                    generated_tokens=1,
                    phase="done",
                    sampler_mode="processed_argmax",
                    sampler_fallback_reason="processed_logits_required",
                    sampler_fast_path_blockers=("logit_bias",),
                    timing={"prefill_ms": 2.0},
                ),
            ),
            GenerationOutput(
                text="second",
                telemetry=GenerationTelemetry.from_decode_counts(
                    row_index=1,
                    prompt_tokens=11,
                    generated_tokens=2,
                    phase="done",
                    sampler_mode="host_logits_sample",
                    sampler_fallback_reason="host_sampling_required",
                    sampler_fast_path_blockers=("temperature",),
                    timing={"prefill_ms": 2.5},
                ),
            ),
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "n": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content_payloads = [
        payload
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    ]
    assert [payload["choices"][0]["index"] for payload in content_payloads] == [0, 1]
    assert content_payloads[0]["choices"][0]["hipengine"] == {
        "phase": "answer",
        "tokens": {
            "streamed_tokens": 1,
            "delta_tokens": 1,
            "answer_tokens": 1,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 1,
            "prompt_tokens": 0,
            "generated_tokens": 1,
            "phase": "answer",
            "continuation_eligible": False,
            "answer_tokens": 1,
            "sampler_fast_path_blockers": ["logit_bias"],
            "sampler_fallback_reason": "processed_logits_required",
            "sampler_mode": "processed_argmax",
        },
    }
    assert content_payloads[1]["choices"][0]["hipengine"] == {
        "phase": "answer",
        "tokens": {
            "streamed_tokens": 1,
            "delta_tokens": 1,
            "answer_tokens": 1,
        },
        "decode_state": {
            "row_index": 1,
            "step_index": 1,
            "prompt_tokens": 0,
            "generated_tokens": 1,
            "phase": "answer",
            "continuation_eligible": False,
            "answer_tokens": 1,
            "sampler_fast_path_blockers": ["temperature"],
            "sampler_fallback_reason": "host_sampling_required",
            "sampler_mode": "host_logits_sample",
        },
    }
    done_payloads = [
        payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"]
    ]
    assert [payload["choices"][0]["index"] for payload in done_payloads] == [0, 1]
    assert done_payloads[0]["choices"][0]["hipengine"] == {
        "phase": "done",
        "decode_state": {
            "row_index": 0,
            "step_index": 1,
            "prompt_tokens": 11,
            "generated_tokens": 1,
            "phase": "done",
            "continuation_eligible": False,
            "sampler_fast_path_blockers": ["logit_bias"],
            "sampler_fallback_reason": "processed_logits_required",
            "sampler_mode": "processed_argmax",
        },
        "timing": {"prefill_ms": 2.0},
        "timing_scope": "choice",
        "group_rows": 1,
        "timing_owner": True,
        "finish_details": _stateless_finish_details("stop"),
        "tokens": {
            "streamed_tokens": 1,
            "answer_tokens": 1,
        },
    }
    assert done_payloads[1]["choices"][0]["hipengine"] == {
        "phase": "done",
        "decode_state": {
            "row_index": 1,
            "step_index": 2,
            "prompt_tokens": 11,
            "generated_tokens": 2,
            "phase": "done",
            "continuation_eligible": False,
            "sampler_fast_path_blockers": ["temperature"],
            "sampler_fallback_reason": "host_sampling_required",
            "sampler_mode": "host_logits_sample",
        },
        "timing": {"prefill_ms": 2.5},
        "timing_scope": "choice",
        "group_rows": 1,
        "timing_owner": True,
        "finish_details": _stateless_finish_details("stop"),
        "tokens": {
            "streamed_tokens": 1,
            "answer_tokens": 1,
        },
    }
    assert done_payloads[0]["hipengine"]["timing"]["backend_prefill_ms"] == 2.0
    assert done_payloads[1]["hipengine"]["timing"]["backend_prefill_ms"] == 2.5
    assert fake.stream_calls == []


def test_streaming_chat_completion_returns_token_sse_and_usage() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=["<think>scratch pad</think>streamed ", "reply"],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"object":"chat.completion.chunk"' in response.text
    assert "data: [DONE]" in response.text
    payloads = _sse_payloads(response.text)
    assert all("hipengine" not in payload for payload in payloads)
    assert all("hipengine" not in payload["choices"][0] for payload in payloads if payload.get("choices"))
    deltas = [payload["choices"][0]["delta"] for payload in payloads if payload.get("choices")]
    assert deltas[:4] == [
        {"role": "assistant"},
        {"reasoning_content": "scratch pad"},
        {"content": "streamed "},
        {"content": "reply"},
    ]
    assert len(fake.stream_calls) == 1
    prompt = fake.calls[0][0][0]
    completion_tokens = fake.count_tokens("<think>scratch pad</think>streamed reply")
    assert payloads[-1]["usage"] == {
        "prompt_tokens": fake.count_tokens(prompt),
        "completion_tokens": completion_tokens,
        "total_tokens": fake.count_tokens(prompt) + completion_tokens,
    }
    assert fake.calls[0][1].max_tokens == 4096


def test_streaming_chat_completion_can_include_hipengine_metadata() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=["<think>scratch pad</think>streamed ", "reply"],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert payloads[0]["hipengine"]["metadata_version"] == 1
    assert payloads[0]["hipengine"]["event"] == "role"
    assert isinstance(payloads[0]["hipengine"]["timing"]["elapsed_ms"], float)
    assert "ttft_ms" not in payloads[0]["hipengine"]["timing"]
    assert all(payload["hipengine"]["routing"] == _routing_metadata() for payload in payloads)

    reasoning = next(payload for payload in payloads if payload.get("choices") and "reasoning_content" in payload["choices"][0]["delta"])
    assert reasoning["hipengine"]["event"] == "delta"
    assert isinstance(reasoning["hipengine"]["timing"]["ttft_ms"], float)
    assert reasoning["choices"][0]["hipengine"] == {
        "phase": "think",
        "tokens": {
            "streamed_tokens": 2,
            "delta_tokens": 2,
            "reasoning_tokens": 2,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 2,
            "prompt_tokens": 0,
            "generated_tokens": 2,
            "phase": "think",
            "continuation_eligible": False,
            "reasoning_tokens": 2,
        },
    }

    content = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["delta"].get("content") == "streamed ")
    assert content["choices"][0]["hipengine"] == {
        "phase": "answer",
        "tokens": {
            "streamed_tokens": 3,
            "delta_tokens": 1,
            "reasoning_tokens": 2,
            "answer_tokens": 1,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 3,
            "prompt_tokens": 0,
            "generated_tokens": 3,
            "phase": "answer",
            "continuation_eligible": False,
            "reasoning_tokens": 2,
            "answer_tokens": 1,
        },
    }

    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"] == "stop")
    assert done["hipengine"]["event"] == "done"
    assert isinstance(done["hipengine"]["timing"]["ttft_ms"], float)
    assert isinstance(done["hipengine"]["timing"]["decode_elapsed_ms"], float)
    assert isinstance(done["hipengine"]["timing"]["decode_tokens_per_second"], float)
    assert done["choices"][0]["hipengine"] == {
        "phase": "done",
        "finish_details": _stateless_finish_details("stop"),
        "tokens": {
            "prompt_tokens": payloads[-1]["usage"]["prompt_tokens"],
            "completion_tokens": payloads[-1]["usage"]["completion_tokens"],
            "total_tokens": payloads[-1]["usage"]["total_tokens"],
            "streamed_tokens": 4,
            "reasoning_tokens": 2,
            "answer_tokens": 2,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 4,
            "prompt_tokens": payloads[-1]["usage"]["prompt_tokens"],
            "generated_tokens": payloads[-1]["usage"]["completion_tokens"],
            "phase": "done",
            "continuation_eligible": False,
            "reasoning_tokens": 2,
            "answer_tokens": 2,
        },
    }
    assert payloads[-1]["hipengine"]["event"] == "usage"
    assert payloads[-1]["hipengine"]["usage"] == payloads[-1]["usage"]
    assert isinstance(payloads[-1]["hipengine"]["timing"]["ttft_ms"], float)
    assert isinstance(payloads[-1]["hipengine"]["timing"]["decode_elapsed_ms"], float)
    assert isinstance(payloads[-1]["hipengine"]["timing"]["decode_tokens_per_second"], float)


def test_streaming_chat_completion_n_forwards_runtime_native_live_chunks() -> None:
    class LiveManyFakeLLM(FakeLLM):
        supports_stream_many = True

        def __init__(self) -> None:
            super().__init__(outputs=["should-not-buffer"])
            self.stream_many_calls: list[tuple[tuple[str, ...], SamplingParams]] = []

        def stream_many_detailed(self, prompts, sampling_params: SamplingParams):
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.stream_many_calls.append((prompt_tuple, sampling_params))
            chunks = (
                (
                    0,
                    "<think>r0</think>",
                    GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=1,
                        row_index=0,
                        request_id="0",
                        phase="think",
                        sampler_mode="greedy_fast",
                        execution_path="runtime_native_live_many",
                    ),
                    None,
                ),
                (
                    1,
                    "B",
                    GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=1,
                        row_index=1,
                        request_id="1",
                        phase="answer",
                        sampler_mode="greedy_fast",
                        execution_path="runtime_native_live_many",
                    ),
                    None,
                ),
                (
                    0,
                    "A",
                    GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=0,
                        request_id="0",
                        phase="answer",
                        sampler_mode="greedy_fast",
                        execution_path="runtime_native_live_many",
                    ),
                    FinishDetails(reason="length", length_limit=2, sampler_mode="greedy_fast"),
                ),
                (
                    1,
                    "C",
                    GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=1,
                        request_id="1",
                        phase="answer",
                        sampler_mode="greedy_fast",
                        execution_path="runtime_native_live_many",
                    ),
                    FinishDetails(reason="length", length_limit=2, sampler_mode="greedy_fast"),
                ),
            )
            assert tuple(row for row, _text, _telemetry, _finish in chunks) == (0, 1, 0, 1)
            for _row, text, telemetry, finish_details in chunks:
                yield GenerationStreamChunk(
                    text=text,
                    telemetry=telemetry,
                    finish_details=finish_details,
                )

    fake = LiveManyFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    capabilities = client.get("/v1/hipengine/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["features"]["stream_metadata"]["live_many_chunks"]["available"] is True

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "n": 2,
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    choices = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] is None
    ]
    assert [(choice["index"], choice["delta"]) for choice in choices] == [
        (0, {"role": "assistant"}),
        (1, {"role": "assistant"}),
        (0, {"reasoning_content": "r0"}),
        (1, {"content": "B"}),
        (0, {"content": "A"}),
        (1, {"content": "C"}),
    ]
    token_choices = [choice for choice in choices if "role" not in choice["delta"]]
    assert [
        (
            choice["index"],
            choice["hipengine"]["phase"],
            choice["hipengine"]["decode_state"]["row_index"],
            choice["hipengine"]["decode_state"]["execution_path"],
        )
        for choice in token_choices
    ] == [
        (0, "think", 0, "runtime_native_live_many"),
        (1, "answer", 1, "runtime_native_live_many"),
        (0, "answer", 0, "runtime_native_live_many"),
        (1, "answer", 1, "runtime_native_live_many"),
    ]
    done = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] == "length"
    ]
    assert [(choice["index"], choice["finish_details"]) for choice in done] == [
        (
            0,
            {
                "reason": "length",
                "length_limit": 2,
                "cache_action": "append_none",
                "sampler_mode": "greedy_fast",
                "phase": "answer",
                "continuation_eligible": False,
            },
        ),
        (
            1,
            {
                "reason": "length",
                "length_limit": 2,
                "cache_action": "append_none",
                "sampler_mode": "greedy_fast",
                "phase": "answer",
                "continuation_eligible": False,
            },
        ),
    ]
    prompt_tokens = sum(fake.count_tokens(prompt) for prompt in fake.stream_many_calls[0][0])
    assert payloads[-1]["usage"] == {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 2,
        "total_tokens": prompt_tokens + 2,
    }
    assert len(fake.stream_many_calls) == 1
    assert fake.calls == []
    assert fake.stream_calls == []
    assert "data: [DONE]" in response.text


def test_streaming_chat_completion_n_buffers_live_many_unsupported_surfaces() -> None:
    class LiveManyCapableFakeLLM(FakeLLM):
        supports_stream_many = True

        def __init__(self, outputs: list[str], *, include_logprobs: bool = False) -> None:
            super().__init__(outputs=outputs)
            self.include_logprobs = bool(include_logprobs)
            self.stream_many_calls: list[tuple[tuple[str, ...], SamplingParams]] = []

        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            if not self.include_logprobs:
                return super().generate_detailed(prompts, sampling_params)
            prompt_tuple = tuple(prompts)
            self.calls.append((prompt_tuple, sampling_params))
            return [
                GenerationOutput(
                    text=output,
                    token_logprobs=(
                        TokenLogprob(
                            token_id=700 + index,
                            token_text=output,
                            logprob=-0.1,
                            top_logprobs=((700 + index, output, -0.1),),
                        ),
                    ),
                )
                for index, output in enumerate(self.outputs or ())
            ][: len(prompt_tuple)]

        def stream_many_detailed(self, prompts, sampling_params: SamplingParams):
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.stream_many_calls.append((prompt_tuple, sampling_params))
            raise AssertionError("unsupported live-many surfaces must use the buffered path")

    tool_request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a command",
                    "parameters": {"type": "object"},
                },
            }
        ]
    }
    cases: tuple[tuple[str, dict[str, Any], list[str], bool], ...] = (
        (
            "tools",
            tool_request,
            [
                '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',
                '<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>',
            ],
            False,
        ),
        ("logprobs", {"logprobs": True}, ["alpha", "beta"], True),
        ("response_format", {"response_format": {"type": "json_object"}}, ['{"ok":true}', '{"ok":false}'], False),
        ("stop", {"stop": "<stop>"}, ["alpha<stop>tail", "beta<stop>tail"], False),
    )

    for label, request_extra, outputs, include_logprobs in cases:
        fake = LiveManyCapableFakeLLM(outputs, include_logprobs=include_logprobs)
        app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
        client = TestClient(app)

        capabilities = client.get("/v1/hipengine/capabilities")
        assert capabilities.status_code == 200
        assert capabilities.json()["features"]["stream_metadata"]["live_many_chunks"]["available"] is True

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": f"{label} request"}],
                "n": 2,
                "stream": True,
                "stream_options": {"include_hipengine": True},
                **request_extra,
            },
        )

        assert response.status_code == 200, (label, response.text)
        assert fake.stream_many_calls == []
        assert len(fake.calls) == 1
        assert len(fake.calls[0][0]) == 2
        assert "data: [DONE]" in response.text
        payloads = _sse_payloads(response.text)
        choices = [
            payload["choices"][0]
            for payload in payloads
            if payload.get("choices")
        ]
        done = [choice for choice in choices if choice["finish_reason"] is not None]
        assert [choice["index"] for choice in done] == [0, 1]
        if label == "tools":
            assert "<tool_call>" not in response.text
            tool_choices = [
                choice for choice in choices if choice["delta"].get("tool_calls")
            ]
            assert [choice["index"] for choice in tool_choices] == [0, 1]
            for choice, command in zip(tool_choices, ("pwd", "ls"), strict=True):
                _assert_openai_stream_tool_call_delta_shape(
                    choice,
                    name="bash",
                    arguments={"command": command},
                    index=choice["index"],
                )
            assert [choice["finish_reason"] for choice in done] == ["tool_calls", "tool_calls"]
        elif label == "logprobs":
            assert any(choice.get("logprobs") is not None for choice in choices)
            assert [choice["finish_reason"] for choice in done] == ["stop", "stop"]
        elif label == "response_format":
            assert [choice["finish_details"]["reason"] for choice in done] == ["stop", "stop"]
            content = "".join(
                choice["delta"].get("content", "")
                for choice in choices
                if choice["index"] == 0
            )
            assert json.loads(content) == {"ok": True}
        elif label == "stop":
            content_by_index = {
                index: "".join(
                    choice["delta"].get("content", "")
                    for choice in choices
                    if choice["index"] == index
                )
                for index in (0, 1)
            }
            assert content_by_index == {0: "alpha", 1: "beta"}
            assert [choice["finish_reason"] for choice in done] == ["stop", "stop"]


def test_streaming_chat_completion_can_include_kv_pool_metadata() -> None:
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=["streamed reply"])
    fake.kv_pool_stats = _fake_kv_pool_stats()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert "kv_pool" not in payloads[0]["hipengine"]
    done = next(payload for payload in payloads if payload["hipengine"]["event"] == "done")
    usage = next(payload for payload in payloads if payload["hipengine"]["event"] == "usage")
    assert done["hipengine"]["kv_pool"] == _fake_kv_pool_metadata()
    assert usage["hipengine"]["kv_pool"] == _fake_kv_pool_metadata()


def test_streaming_chat_completion_prefers_backend_chunk_decode_state() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "backend reply",
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=7,
                    generated_tokens=3,
                    phase="answer",
                    sampler_mode="processed_argmax",
                    sampler_fallback_reason="processed_logits_required",
                    forced_token_id=78,
                    forced_token_reason="tool_choice_required",
                    forced_tokens_remaining=1,
                    active_processors=("logit_bias",),
                    sampler_fast_path_blockers=("logit_bias",),
                    timing={"prefill_ms": 4.0, "decode_ms": 2.0},
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
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["delta"].get("content"))
    assert content["choices"][0]["delta"] == {"content": "backend reply"}
    assert content["choices"][0]["hipengine"] == {
        "phase": "answer",
        "decode_state": {
            "row_index": 0,
            "step_index": 3,
            "prompt_tokens": 7,
            "generated_tokens": 3,
            "phase": "answer",
            "continuation_eligible": False,
            "forced_token_id": 78,
            "forced_token_reason": "tool_choice_required",
            "forced_tokens_remaining": 1,
            "active_processors": ["logit_bias"],
            "sampler_fast_path_blockers": ["logit_bias"],
            "sampler_fallback_reason": "processed_logits_required",
            "sampler_mode": "processed_argmax",
        },
        "timing": {"prefill_ms": 4.0, "decode_ms": 2.0},
        "timing_scope": "choice",
        "group_rows": 1,
        "timing_owner": True,
        "tokens": {
            # Backend committed count, not re-tokenized display words.
            "streamed_tokens": 3,
            "delta_tokens": 3,
            "answer_tokens": 3,
        },
    }
    assert content["hipengine"]["timing"]["backend_prefill_ms"] == 4.0
    assert content["hipengine"]["timing"]["backend_decode_ms"] == 2.0
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["hipengine"]["timing"]["backend_prefill_ms"] == 4.0
    assert done["hipengine"]["timing"]["backend_decode_ms"] == 2.0


def test_streaming_chat_completion_prefers_backend_chunk_finish_details() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "backend reply",
                finish_details=FinishDetails(
                    reason="length",
                    length_limit=4,
                    sampler_mode="processed_argmax",
                    phase="answer",
                    continuation_eligible=False,
                ),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=7,
                    generated_tokens=4,
                    phase="answer",
                    sampler_mode="processed_argmax",
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
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    expected_finish = {
        "reason": "length",
        "length_limit": 4,
        "cache_action": "append_none",
        "sampler_mode": "processed_argmax",
        "phase": "answer",
        "continuation_eligible": False,
    }
    assert done["choices"][0]["finish_reason"] == "length"
    assert done["choices"][0]["finish_details"] == expected_finish
    assert done["choices"][0]["hipengine"]["finish_details"] == expected_finish
    assert done["choices"][0]["hipengine"]["decode_state"] == {
        "row_index": 0,
        "step_index": 4,
        "prompt_tokens": 7,
        "generated_tokens": 4,
        "phase": "answer",
        "continuation_eligible": False,
        "sampler_mode": "processed_argmax",
    }


def test_streaming_completion_uses_engine_stream_and_usage() -> None:
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=["alpha", " beta"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text
    payloads = _sse_payloads(response.text)
    assert all("hipengine" not in payload for payload in payloads)
    assert all("hipengine" not in payload["choices"][0] for payload in payloads if payload.get("choices"))
    text_chunks = [payload["choices"][0]["text"] for payload in payloads if payload.get("choices")]
    assert text_chunks == ["alpha", " beta", ""]
    assert payloads[-1]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert fake.stream_calls == [("hello", SamplingParams(max_tokens=2))]


def test_streaming_completion_can_include_hipengine_metadata() -> None:
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=["alpha", " beta"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert [payload["hipengine"]["event"] for payload in payloads] == ["delta", "delta", "done", "usage"]
    assert all(payload["hipengine"]["routing"] == _routing_metadata() for payload in payloads)
    assert isinstance(payloads[0]["hipengine"]["timing"]["ttft_ms"], float)
    assert payloads[0]["choices"][0]["hipengine"] == {
        "phase": "answer",
        "tokens": {
            "streamed_tokens": 1,
            "delta_tokens": 1,
            "answer_tokens": 1,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 1,
            "prompt_tokens": 0,
            "generated_tokens": 1,
            "phase": "answer",
            "continuation_eligible": False,
            "answer_tokens": 1,
        },
    }
    assert isinstance(payloads[0]["hipengine"]["timing"]["elapsed_ms"], float)
    assert isinstance(payloads[2]["hipengine"]["timing"]["ttft_ms"], float)
    assert isinstance(payloads[2]["hipengine"]["timing"]["decode_elapsed_ms"], float)
    assert isinstance(payloads[2]["hipengine"]["timing"]["decode_tokens_per_second"], float)
    assert payloads[2]["choices"][0]["hipengine"] == {
        "phase": "done",
        "finish_details": _stateless_finish_details("stop"),
        "tokens": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "streamed_tokens": 2,
            "answer_tokens": 2,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 2,
            "prompt_tokens": 1,
            "generated_tokens": 2,
            "phase": "done",
            "continuation_eligible": False,
            "answer_tokens": 2,
        },
    }
    assert payloads[-1]["hipengine"]["usage"] == payloads[-1]["usage"]
    assert isinstance(payloads[-1]["hipengine"]["timing"]["ttft_ms"], float)
    assert isinstance(payloads[-1]["hipengine"]["timing"]["decode_elapsed_ms"], float)
    assert isinstance(payloads[-1]["hipengine"]["timing"]["decode_tokens_per_second"], float)


def test_streaming_completion_can_include_kv_pool_metadata() -> None:
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=["alpha", " beta"])
    fake.kv_pool_stats = _fake_kv_pool_stats()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert "kv_pool" not in payloads[0]["hipengine"]
    done = next(payload for payload in payloads if payload["hipengine"]["event"] == "done")
    usage = next(payload for payload in payloads if payload["hipengine"]["event"] == "usage")
    assert done["hipengine"]["kv_pool"] == _fake_kv_pool_metadata()
    assert usage["hipengine"]["kv_pool"] == _fake_kv_pool_metadata()


def test_streaming_completion_prefers_backend_chunk_decode_state() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "alpha",
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=5,
                    generated_tokens=4,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    sampler_fallback_reason="host_sampling_required",
                    forced_token_id=79,
                    forced_token_reason="thinking_hard_close",
                    forced_tokens_remaining=0,
                    sampler_fast_path_blockers=("temperature",),
                    timing={"prefill_ms": 4.0, "decode_ms": 2.0},
                    usage={"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
                ),
            )
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 1,
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert payloads[0]["choices"][0]["text"] == "alpha"
    assert payloads[0]["choices"][0]["hipengine"] == {
        "phase": "answer",
        "decode_state": {
            "row_index": 0,
            "step_index": 4,
            "prompt_tokens": 5,
            "generated_tokens": 4,
            "phase": "answer",
            "continuation_eligible": False,
            "forced_token_id": 79,
            "forced_token_reason": "thinking_hard_close",
            "forced_tokens_remaining": 0,
            "sampler_fast_path_blockers": ["temperature"],
            "sampler_fallback_reason": "host_sampling_required",
            "sampler_mode": "host_logits_sample",
        },
        "timing": {"prefill_ms": 4.0, "decode_ms": 2.0},
        "timing_scope": "choice",
        "group_rows": 1,
        "timing_owner": True,
        "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        "tokens": {
            "streamed_tokens": 4,
            "delta_tokens": 4,
            "answer_tokens": 4,
        },
    }
    assert payloads[0]["hipengine"]["timing"]["backend_prefill_ms"] == 4.0
    assert payloads[0]["hipengine"]["timing"]["backend_decode_ms"] == 2.0
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["hipengine"]["timing"] == {"prefill_ms": 4.0, "decode_ms": 2.0}
    assert done["hipengine"]["timing"]["backend_prefill_ms"] == 4.0
    assert done["hipengine"]["timing"]["backend_decode_ms"] == 2.0


def test_streaming_completion_uses_backend_decode_counts_for_exact_usage() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                " ",
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=1,
                    phase="answer",
                ),
            ),
            GenerationStreamChunk(
                "\u00a0",
                finish_details=FinishDetails(reason="length", length_limit=2),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=2,
                    phase="done",
                ),
            ),
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    deltas = [payload for payload in payloads if payload.get("hipengine", {}).get("event") == "delta"]
    assert [payload["choices"][0]["text"] for payload in deltas] == [" ", "\u00a0"]
    assert [payload["choices"][0]["hipengine"]["tokens"]["streamed_tokens"] for payload in deltas] == [1, 2]
    usage = next(payload["usage"] for payload in payloads if payload.get("usage") is not None)
    assert usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    done = next(payload for payload in payloads if payload.get("hipengine", {}).get("event") == "done")
    assert done["choices"][0]["hipengine"]["tokens"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
        "streamed_tokens": 2,
        "answer_tokens": 2,
    }


def test_streaming_completion_prefers_backend_chunk_finish_details() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "alpha",
                finish_details=FinishDetails(
                    reason="length",
                    length_limit=4,
                    sampler_mode="host_logits_sample",
                ),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=5,
                    generated_tokens=4,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                ),
            )
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 1,
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    expected_finish = {
        "reason": "length",
        "length_limit": 4,
        "cache_action": "append_none",
        "sampler_mode": "host_logits_sample",
    }
    assert done["choices"][0]["finish_reason"] == "length"
    assert done["choices"][0]["finish_details"] == expected_finish
    assert done["choices"][0]["hipengine"]["finish_details"] == expected_finish
    assert done["choices"][0]["hipengine"]["decode_state"] == {
        "row_index": 0,
        "step_index": 4,
        "prompt_tokens": 5,
        "generated_tokens": 4,
        "phase": "answer",
        "continuation_eligible": False,
        "sampler_mode": "host_logits_sample",
    }


def test_streaming_chat_completion_lowers_n_to_seeded_rows() -> None:
    fake = FakeLLM(outputs=["alpha", "beta"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "n": 2,
            "seed": 5,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    indices = [payload["choices"][0]["index"] for payload in payloads]
    assert 0 in indices and 1 in indices
    assert "data: [DONE]" in response.text
    assert fake.calls[0][0] == (fake.calls[0][0][0], fake.calls[0][0][0])
    assert len(fake.calls[0][1].row_seeds) == 2
    assert len(set(fake.calls[0][1].row_seeds)) == 2
