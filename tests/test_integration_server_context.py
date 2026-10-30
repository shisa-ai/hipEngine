"""OpenAI server integration tests: context.

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
    ChatCompletionRequest,
    CompletionRequest,
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
    _parse_chat_tool_calls,
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
    _ServerMetrics,
    _log_pretty_startup_summary,
    _startup_memory_summary,
    _mtp_accepted_rejected_counts,
    _mtp_response_summary,
)
from tests.support.server_api_fakes import (
    FakeLLM,
    NoTokenizerFakeLLM,
    SequentialFakeLLM,
)


def test_token_diagnostics_endpoints_handle_text_and_chat() -> None:
    fake = FakeLLM(token_map={"hello": [10, 11], "closing now</think>\n": [42, 43, 44]})
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=512,
            chat_default_max_tokens=7,
        ),
        llm=fake,
    )
    client = TestClient(app)

    tokenize = client.post("/v1/hipengine/tokenize", json={"text": "hello"})
    assert tokenize.status_code == 200
    tokenize_body = tokenize.json()
    assert tokenize_body.pop("timing")["tokenize_ms"] >= 0.0
    assert tokenize_body == {
        "object": "hipengine.tokens",
        "text": "hello",
        "token_ids": [10, 11],
        "token_count": 2,
    }

    detokenize = client.post("/v1/hipengine/detokenize", json={"token_ids": [10, 11]})
    assert detokenize.status_code == 200
    assert detokenize.json() == {
        "object": "hipengine.text",
        "text": "T10 T11",
        "token_ids": [10, 11],
    }

    count_text = client.post("/v1/hipengine/count_tokens", json={"text": "one two three"})
    assert count_text.status_code == 200
    assert count_text.json()["token_count"] == 3
    assert count_text.json()["input_type"] == "text"
    assert count_text.json()["timing"]["tokenize_ms"] >= 0.0

    chat_payload = {
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"query":"hello"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "tool result"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "reasoning_effort": "low",
        "hard_close_sequence": "closing now</think>\n",
        "soft_close_window": 4,
        "max_tokens": 32,
    }
    count_chat = client.post("/v1/hipengine/count_tokens", json=chat_payload)
    assert count_chat.status_code == 200
    chat_body = count_chat.json()
    assert chat_body["input_type"] == "chat"
    assert "<|im_start|>user\nhello<|im_end|>" in chat_body["text"]
    assert "<tools>" in chat_body["text"]
    assert (
        "<tool_call>\n<function=lookup>\n<parameter=query>\nhello\n</parameter>\n</function>\n</tool_call>"
        in chat_body["text"]
    )
    assert "<tool_response>\ntool result\n</tool_response>" in chat_body["text"]
    assert "use 'closing now</think>\\n' as the close sequence" in chat_body["text"]
    assert chat_body["token_count"] == fake.count_tokens(chat_body["text"])
    assert chat_body["thinking_budget"]["close_text"] == "closing now</think>\n"
    assert chat_body["thinking_budget"]["close_token_ids"] == [42, 43, 44]
    assert chat_body["thinking_budget"]["initial_state"] == {
        "phase": "think",
        "reasoning_tokens": 0,
        "answer_tokens": 0,
        "hard_token_cap": 13,
        "remaining_think_tokens": 13,
        "soft_close_window": 4,
        "close_sequence": [42, 43, 44],
    }

    fit = client.post("/v1/hipengine/fit_context", json=chat_payload)
    assert fit.status_code == 200
    fit_body = fit.json()
    expected_max_tokens = 32
    assert fit_body["input_type"] == "chat"
    assert fit_body["prompt_tokens"] == chat_body["token_count"]
    assert fit_body["max_context_tokens"] == 512
    assert fit_body["requested_max_tokens"] == 32
    assert fit_body["effective_max_tokens"] == expected_max_tokens
    assert fit_body["max_allowed_max_tokens"] == 512 - chat_body["token_count"] - 1
    assert fit_body["recommended_max_tokens"] == expected_max_tokens
    assert fit_body["required_context_tokens"] == chat_body["token_count"] + expected_max_tokens + 1
    assert fit_body["overflow_tokens"] == 0
    assert fit_body["fits"] is True
    assert fit_body["chat_default_max_tokens"] == 7
    assert fit_body["clear_policy"] == "reject"
    assert fit_body["would_drop"] == []
    assert fit_body["thinking_budget"]["lowering_supported"] is True
    assert fit_body["thinking_budget"]["close_token_ids"] == [42, 43, 44]


def test_token_diagnostics_report_unbounded_nested_reasoning_control() -> None:
    fake = FakeLLM(token_map={"</think>": [42, 43]})
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=512,
            chat_default_max_tokens=128,
        ),
        llm=fake,
    )
    client = TestClient(app)
    payload = {
        "messages": [{"role": "user", "content": "think without hard cap"}],
        "max_tokens": 64,
        "reasoning": {"effort": "low", "allow_unbounded": True},
    }

    count = client.post("/v1/hipengine/count_tokens", json=payload)
    fit = client.post("/v1/hipengine/fit_context", json=payload)

    assert count.status_code == 200
    assert fit.status_code == 200
    for body in (count.json(), fit.json()):
        budget = body["thinking_budget"]
        assert budget["enabled"] is True
        assert budget["effort"] == "low"
        assert budget["allow_unbounded"] is True
        assert budget["min_answer_tokens"] == 32
        assert budget["close_text"] == "</think>"
        assert budget["close_token_ids"] == [42, 43]
        assert "hard_think_cap" not in budget
        assert budget["initial_state"] == {
            "phase": "think",
            "reasoning_tokens": 0,
            "answer_tokens": 0,
            "hard_token_cap": 30,
            "remaining_think_tokens": 30,
            "close_sequence": [42, 43],
        }


def test_token_diagnostics_use_session_prefix_for_chat() -> None:
    fake = SequentialFakeLLM(["stored answer", "follow-up answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=512,
            chat_default_max_tokens=9,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "remember alpha"}],
            "max_tokens": 2,
            "session": {"id": "diag_session"},
        },
    )
    diagnostic_payload = {
        "messages": [{"role": "user", "content": "now beta"}],
        "session": {"id": "diag_session"},
    }
    count = client.post("/v1/hipengine/count_tokens", json=diagnostic_payload)
    fit = client.post("/v1/hipengine/fit_context", json=diagnostic_payload)
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "now beta"}],
            "max_tokens": 2,
            "session": {"id": "diag_session", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert count.status_code == 200
    assert fit.status_code == 200
    assert second.status_code == 200
    count_body = count.json()
    assert count_body["input_type"] == "chat"
    assert "remember alpha" in count_body["text"]
    assert "stored answer" in count_body["text"]
    assert "now beta" in count_body["text"]
    assert count_body["session"] == {
        "id": "diag_session",
        "stateful": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "prefix_message_count": 2,
        "request_message_count": 1,
        "rendered_message_count": 3,
        "cache_action": "append_visible_only",
    }
    assert count_body["token_count"] == fake.count_tokens(count_body["text"])

    fit_body = fit.json()
    assert fit_body["prompt_tokens"] == count_body["token_count"]
    assert fit_body["effective_max_tokens"] == 9
    assert fit_body["max_allowed_max_tokens"] == 512 - count_body["token_count"] - 1
    assert fit_body["recommended_max_tokens"] == 9
    assert fit_body["required_context_tokens"] == count_body["token_count"] + 9 + 1
    assert fit_body["overflow_tokens"] == 0
    assert fit_body["session"] == count_body["session"]
    assert fake.calls[1][0][0] == count_body["text"]


def test_chat_context_overflow_reports_session_fit_context() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=20,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "remember alpha"}],
            "max_tokens": 1,
            "session": {"id": "overflow_session"},
        },
    )
    diagnostic_payload = {
        "messages": [{"role": "user", "content": "now beta"}],
        "max_tokens": 64,
        "session": {"id": "overflow_session"},
    }
    fit = client.post("/v1/hipengine/fit_context", json=diagnostic_payload)
    overflow = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", **diagnostic_payload},
    )

    assert first.status_code == 200
    assert fit.status_code == 200
    assert overflow.status_code == 400
    fit_body = fit.json()
    error = overflow.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert error["hipengine"]["code"] == "context_overflow"
    assert error["param"] == "max_tokens"
    for key in (
        "prompt_tokens",
        "max_context_tokens",
        "effective_max_tokens",
        "max_allowed_max_tokens",
        "recommended_max_tokens",
        "required_context_tokens",
        "overflow_tokens",
        "fits",
        "clear_policy",
        "would_truncate",
        "would_drop",
        "session",
    ):
        assert error["fit_context"][key] == fit_body[key]
    assert error["fit_context"]["session"] == {
        "id": "overflow_session",
        "stateful": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "prefix_message_count": 2,
        "request_message_count": 1,
        "rendered_message_count": 3,
        "cache_action": "append_visible_only",
    }
    assert len(fake.calls) == 1


def test_fit_context_new_session_policy_reports_reset_when_prefix_overflows() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=10,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "remember alpha"}],
            "max_tokens": 1,
            "session": {"id": "fit_reset_session"},
        },
    )
    fit = client.post(
        "/v1/hipengine/fit_context",
        json={
            "messages": [{"role": "user", "content": "now beta"}],
            "max_tokens": 5,
            "session": {"id": "fit_reset_session", "context_overflow_policy": "new_session"},
        },
    )

    assert first.status_code == 200
    assert fit.status_code == 200
    body = fit.json()
    assert body["fits"] is True
    assert body["prompt_tokens"] == 4
    assert body["effective_max_tokens"] == 5
    assert body["required_context_tokens"] == 10
    assert body["clear_policy"] == "new_session"
    assert body["would_truncate"] is False
    assert body["would_reset_session"] is True
    assert body["would_drop"] == [
        {
            "kind": "session_prefix",
            "session_id": "fit_reset_session",
            "storage": "app_local_transcript",
            "message_count": 2,
        }
    ]
    assert body["kept_segments"] == [{"kind": "request_messages", "message_count": 1}]
    assert body["session"] == {
        "id": "fit_reset_session",
        "stateful": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "prefix_message_count": 0,
        "request_message_count": 1,
        "rendered_message_count": 1,
        "cache_action": "append_visible_only",
    }
    serialized = json.dumps(body)
    assert "now beta" in serialized
    assert "remember alpha" not in serialized
    assert "stored answer" not in serialized


def test_chat_context_new_session_policy_resets_transcript_on_success() -> None:
    fake = SequentialFakeLLM(["stored answer", "fresh answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=10,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "remember alpha"}],
            "max_tokens": 1,
            "session": {"id": "chat_reset_session"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "now beta"}],
            "max_tokens": 5,
            "session": {"id": "chat_reset_session", "context_overflow_policy": "new_session"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(fake.calls) == 2
    prompt = fake.calls[1][0][0]
    assert "now beta" in prompt
    assert "remember alpha" not in prompt
    assert "stored answer" not in prompt
    record = app.state.hipengine_chat_sessions["chat_reset_session"]
    assert record.messages == (
        {"role": "user", "content": "now beta"},
        {"role": "assistant", "content": "fresh answer"},
    )


def test_chat_context_new_session_policy_does_not_hide_request_overflow() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=10,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "remember alpha"}],
            "max_tokens": 1,
            "session": {"id": "too_large_session"},
        },
    )
    payload = {
        "messages": [{"role": "user", "content": "one two three four five six seven eight nine ten"}],
        "max_tokens": 1,
        "session": {"id": "too_large_session", "context_overflow_policy": "new_session"},
    }
    fit = client.post("/v1/hipengine/fit_context", json=payload)
    overflow = client.post("/v1/chat/completions", json={"model": "fake-model", **payload})

    assert first.status_code == 200
    assert fit.status_code == 200
    assert overflow.status_code == 400
    fit_body = fit.json()
    error = overflow.json()["error"]
    assert fit_body["fits"] is False
    assert fit_body["clear_policy"] == "new_session"
    assert fit_body["would_reset_session"] is False
    assert fit_body["would_drop"] == []
    assert fit_body["kept_segments"] == [
        {
            "kind": "session_prefix",
            "session_id": "too_large_session",
            "storage": "app_local_transcript",
            "message_count": 2,
        },
        {"kind": "request_messages", "message_count": 1},
    ]
    assert fit_body["session"]["prefix_message_count"] == 2
    assert error["code"] == "context_length_exceeded"
    assert error["fit_context"]["clear_policy"] == "new_session"
    assert error["fit_context"]["would_reset_session"] is False
    assert len(fake.calls) == 1
    assert app.state.hipengine_chat_sessions["too_large_session"].messages == (
        {"role": "user", "content": "remember alpha"},
        {"role": "assistant", "content": "stored answer"},
    )


def test_chat_context_auto_clear_transient_preserves_committed_prefix() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=10,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "remember alpha"}],
            "max_tokens": 1,
            "session": {"id": "auto_clear_session"},
        },
    )
    payload = {
        "messages": [{"role": "user", "content": "now beta"}],
        "max_tokens": 5,
        "session": {"id": "auto_clear_session", "context_overflow_policy": "auto_clear_transient"},
    }
    fit = client.post("/v1/hipengine/fit_context", json=payload)
    overflow = client.post("/v1/chat/completions", json={"model": "fake-model", **payload})

    assert first.status_code == 200
    assert fit.status_code == 200
    assert overflow.status_code == 400
    fit_body = fit.json()
    assert fit_body["fits"] is False
    assert fit_body["clear_policy"] == "auto_clear_transient"
    assert fit_body["would_clear_transient"] is False
    assert fit_body["transient_message_count"] == 0
    assert fit_body["would_drop"] == []
    assert fit_body["kept_segments"] == [
        {
            "kind": "session_prefix",
            "session_id": "auto_clear_session",
            "storage": "app_local_transcript",
            "message_count": 2,
        },
        {"kind": "request_messages", "message_count": 1},
    ]
    assert fit_body["session"] == {
        "id": "auto_clear_session",
        "stateful": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "prefix_message_count": 2,
        "request_message_count": 1,
        "rendered_message_count": 3,
        "cache_action": "append_visible_only",
    }
    error = overflow.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert error["fit_context"]["clear_policy"] == "auto_clear_transient"
    assert error["fit_context"]["would_clear_transient"] is False
    assert error["fit_context"]["transient_message_count"] == 0
    assert error["fit_context"]["would_drop"] == []
    assert len(fake.calls) == 1
    assert app.state.hipengine_chat_sessions["auto_clear_session"].messages == (
        {"role": "user", "content": "remember alpha"},
        {"role": "assistant", "content": "stored answer"},
    )


def test_chat_context_truncate_oldest_visible_policy_keeps_fitting_suffix() -> None:
    fake = SequentialFakeLLM(["answer alpha", "answer beta", "answer gamma"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=12,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "one alpha"}],
            "max_tokens": 1,
            "session": {"id": "truncate_session"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "two beta"}],
            "max_tokens": 1,
            "session": {"id": "truncate_session"},
        },
    )
    payload = {
        "messages": [{"role": "user", "content": "now gamma"}],
        "max_tokens": 1,
        "session": {"id": "truncate_session", "context_overflow_policy": "truncate_oldest_visible"},
    }
    fit = client.post("/v1/hipengine/fit_context", json=payload)
    third = client.post("/v1/chat/completions", json={"model": "fake-model", **payload})

    assert first.status_code == 200
    assert second.status_code == 200
    assert fit.status_code == 200
    assert third.status_code == 200
    fit_body = fit.json()
    assert fit_body["fits"] is True
    assert fit_body["prompt_tokens"] == 10
    assert fit_body["effective_max_tokens"] == 1
    assert fit_body["required_context_tokens"] == 12
    assert fit_body["clear_policy"] == "truncate_oldest_visible"
    assert fit_body["would_truncate"] is True
    assert fit_body["would_reset_session"] is False
    assert fit_body["would_drop"] == [
        {
            "kind": "session_prefix",
            "session_id": "truncate_session",
            "storage": "app_local_transcript",
            "message_count": 2,
        }
    ]
    assert fit_body["kept_segments"] == [
        {
            "kind": "session_prefix",
            "session_id": "truncate_session",
            "storage": "app_local_transcript",
            "message_count": 2,
        },
        {"kind": "request_messages", "message_count": 1},
    ]
    assert fit_body["session"] == {
        "id": "truncate_session",
        "stateful": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "prefix_message_count": 2,
        "request_message_count": 1,
        "rendered_message_count": 3,
        "cache_action": "append_visible_only",
    }
    serialized = json.dumps(fit_body)
    assert "two beta" in serialized
    assert "answer beta" in serialized
    assert "now gamma" in serialized
    assert "one alpha" not in serialized
    assert "answer alpha" not in serialized
    prompt = fake.calls[2][0][0]
    assert "two beta" in prompt
    assert "answer beta" in prompt
    assert "now gamma" in prompt
    assert "one alpha" not in prompt
    assert "answer alpha" not in prompt
    assert app.state.hipengine_chat_sessions["truncate_session"].messages == (
        {"role": "user", "content": "two beta"},
        {"role": "assistant", "content": "answer beta"},
        {"role": "user", "content": "now gamma"},
        {"role": "assistant", "content": "answer gamma"},
    )


def test_chat_context_overflow_policy_requires_known_stateful_mode() -> None:
    fake = FakeLLM(outputs=["unused"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    invalid = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "session": {"id": "bad_policy", "context_overflow_policy": "compact_summary"},
        },
    )
    stateless = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "session": {"context_overflow_policy": "new_session"},
        },
    )

    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_request"
    assert invalid.json()["error"]["param"] == "session.context_overflow_policy"
    assert stateless.status_code == 400
    assert stateless.json()["error"]["code"] == "unsupported_parameter"
    assert stateless.json()["error"]["param"] == "session.context_overflow_policy"
    assert fake.calls == []


def test_token_diagnostics_report_unsupported_model_hooks() -> None:
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=NoTokenizerFakeLLM(),
    )
    client = TestClient(app)

    capabilities = client.get("/v1/hipengine/capabilities")
    tokenize = client.post("/v1/hipengine/tokenize", json={"text": "hello"})
    detokenize = client.post("/v1/hipengine/detokenize", json={"token_ids": [1, 2]})
    count = client.post("/v1/hipengine/count_tokens", json={"text": "hello"})
    fit = client.post("/v1/hipengine/fit_context", json={"text": "hello"})

    assert capabilities.status_code == 200
    tokenizer = capabilities.json()["tokenizer"]
    assert tokenizer["tokenize"] is False
    assert tokenizer["detokenize"] is False
    assert tokenizer["count_tokens"] is False
    assert capabilities.json()["features"]["token_diagnostics"] == {
        "tokenize": False,
        "detokenize": False,
        "count_tokens": False,
        "fit_context": False,
        "session_aware_chat": False,
    }
    for response, param in (
        (tokenize, "text"),
        (detokenize, "token_ids"),
        (count, "text"),
        (fit, "text"),
    ):
        assert response.status_code == 501
        assert response.json()["error"]["code"] == "unsupported_feature"
        assert response.json()["error"]["param"] == param
        assert response.json()["error"]["hipengine"] == {
            "code": "unsupported_feature",
            "status_code": 501,
            "retryable": False,
        }


def test_token_diagnostics_reject_ambiguous_inputs() -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/hipengine/count_tokens",
        json={"text": "hello", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["hipengine"] == {
        "code": "schema_violation",
        "status_code": 400,
        "legacy_code": "invalid_request",
        "retryable": False,
    }


def test_token_diagnostics_reject_session_for_raw_text() -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/hipengine/count_tokens",
        json={"text": "hello", "session": {"id": "diag_session"}},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "session"
