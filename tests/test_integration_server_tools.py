"""OpenAI server integration tests: tools.

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
    SchedulerChunkRowsFakeLLM,
    SequentialFakeLLM,
)
from tests.support.server_api_support import (
    _QWEN35_XML_TOOL_CALL,
    _WEATHER_TOOL,
    _assert_openai_stream_tool_call_delta_shape,
    _assert_openai_tool_call_shape,
    _bounded_tool_schema,
    _incomplete_duplicate_read_tool_output,
    _sse_payloads,
    _stateless_finish_details,
)


def test_chat_context_truncate_oldest_visible_policy_skips_orphan_tool_suffix() -> None:
    class WeightedToolTranscriptFakeLLM(SequentialFakeLLM):
        def count_tokens(self, text: str) -> int:
            prompt = str(text)
            if "new beta" not in prompt:
                return 1
            score = 1
            if "README.md" in prompt:
                score += 20
            if "alpha file text" in prompt:
                score += 1
            if "alpha summary" in prompt:
                score += 1
            return score

    fake = WeightedToolTranscriptFakeLLM(
        [
            '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>',
            "alpha summary",
            "after truncate",
        ]
    )
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
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
    ]

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "alpha read"}],
            "tools": tools,
            "max_tokens": 1,
            "session": {"id": "truncate_tool_session"},
        },
    )
    assert first.status_code == 200
    tool_call_id = first.json()["choices"][0]["message"]["tool_calls"][0]["id"]
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "tool", "tool_call_id": tool_call_id, "content": "alpha file text"}],
            "tools": tools,
            "max_tokens": 1,
            "session": {"id": "truncate_tool_session"},
        },
    )
    assert second.status_code == 200
    original_prefix = app.state.hipengine_chat_sessions["truncate_tool_session"].messages
    payload = {
        "messages": [{"role": "user", "content": "new beta"}],
        "max_tokens": 1,
        "session": {
            "id": "truncate_tool_session",
            "context_overflow_policy": "truncate_oldest_visible",
        },
    }

    fit = client.post("/v1/hipengine/fit_context", json=payload)
    third = client.post("/v1/chat/completions", json={"model": "fake-model", **payload})

    assert fit.status_code == 200
    assert third.status_code == 200
    fit_body = fit.json()
    assert fit_body["fits"] is True
    assert fit_body["clear_policy"] == "truncate_oldest_visible"
    assert fit_body["would_truncate"] is True
    assert fit_body["would_reset_session"] is False
    dropped_messages = fit_body["would_drop"][0]["message_count"]
    assert dropped_messages >= 3
    assert fit_body["kept_segments"] == [
        {
            "kind": "session_prefix",
            "session_id": "truncate_tool_session",
            "storage": "app_local_transcript",
            "message_count": len(original_prefix) - dropped_messages,
        },
        {"kind": "request_messages", "message_count": 1},
    ]
    serialized = json.dumps(fit_body)
    assert "new beta" in serialized
    assert "alpha file text" not in serialized
    prompt = fake.calls[2][0][0]
    assert "new beta" in prompt
    assert "alpha file text" not in prompt
    assert "<tool_response>" not in prompt
    expected_messages = (
        *original_prefix[dropped_messages:],
        {"role": "user", "content": "new beta"},
        {"role": "assistant", "content": "after truncate"},
    )
    assert app.state.hipengine_chat_sessions["truncate_tool_session"].messages == expected_messages


def test_chat_session_message_copy_deep_copies_nested_tool_calls() -> None:
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path":"README.md"}'},
            }
        ],
    }

    copied = _chat_session_message_copy(message)

    assert copied == message
    assert copied is not message
    assert copied["tool_calls"] is not message["tool_calls"]
    assert copied["tool_calls"][0] is not message["tool_calls"][0]
    assert copied["tool_calls"][0]["function"] is not message["tool_calls"][0]["function"]
    copied["tool_calls"][0]["function"]["arguments"] = '{"path":"MUTATED.md"}'
    assert message["tool_calls"][0]["function"]["arguments"] == '{"path":"README.md"}'


def test_chat_session_fork_deep_copies_nested_tool_calls() -> None:
    fake = SequentialFakeLLM(
        ['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>']
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
    ]

    created = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read it"}],
            "tools": tools,
            "session": {"id": "sess_tool_root", "commit": "append_visible_only"},
            "max_tokens": 4,
        },
    )
    forked = client.post(
        "/v1/hipengine/sessions/sess_tool_root/fork",
        json={"id": "sess_tool_branch"},
    )

    assert created.status_code == 200
    assert forked.status_code == 200
    root_call = app.state.hipengine_chat_sessions["sess_tool_root"].messages[1]["tool_calls"][0]
    branch_call = app.state.hipengine_chat_sessions["sess_tool_branch"].messages[1]["tool_calls"][0]
    assert branch_call is not root_call
    assert branch_call["function"] is not root_call["function"]

    branch_call["function"]["arguments"] = '{"path":"BRANCH.md"}'

    assert root_call["function"]["arguments"] == '{"path":"README.md"}'
    assert branch_call["function"]["arguments"] == '{"path":"BRANCH.md"}'


def test_chat_session_rollback_deep_copies_retained_tool_calls() -> None:
    fake = SequentialFakeLLM(
        [
            '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>',
            "done",
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
    ]

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read it"}],
            "tools": tools,
            "session": {"id": "sess_tool_rollback", "commit": "append_visible_only"},
            "max_tokens": 4,
        },
    )
    assert first.status_code == 200
    tool_call_id = first.json()["choices"][0]["message"]["tool_calls"][0]["id"]
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "tool", "tool_call_id": tool_call_id, "content": "file text"}],
            "tools": tools,
            "session": {"id": "sess_tool_rollback", "commit": "append_visible_only"},
            "max_tokens": 4,
        },
    )
    previous = app.state.hipengine_chat_sessions["sess_tool_rollback"]
    rollback = client.post(
        "/v1/hipengine/sessions/sess_tool_rollback/rollback",
        json={"message_count": 2},
    )

    assert second.status_code == 200
    assert rollback.status_code == 200
    assert rollback.json()["rolled_back"] is True
    rolled_back = app.state.hipengine_chat_sessions["sess_tool_rollback"]
    previous_call = previous.messages[1]["tool_calls"][0]
    rolled_back_call = rolled_back.messages[1]["tool_calls"][0]
    assert rolled_back_call is not previous_call
    assert rolled_back_call["function"] is not previous_call["function"]

    rolled_back_call["function"]["arguments"] = '{"path":"ROLLED.md"}'

    assert previous_call["function"]["arguments"] == '{"path":"README.md"}'
    assert rolled_back_call["function"]["arguments"] == '{"path":"ROLLED.md"}'


@pytest.mark.parametrize(
    ("corruption", "param"),
    [
        ("non_object", "messages[1].tool_calls[0]"),
        ("extra_key", "messages[1].tool_calls[0].unexpected"),
        ("missing_id", "messages[1].tool_calls[0].id"),
        ("wrong_type", "messages[1].tool_calls[0].type"),
        ("missing_function", "messages[1].tool_calls[0].function"),
        ("function_extra_key", "messages[1].tool_calls[0].function.unexpected"),
        ("empty_name", "messages[1].tool_calls[0].function.name"),
        ("non_string_arguments", "messages[1].tool_calls[0].function.arguments"),
        ("invalid_json_arguments", "messages[1].tool_calls[0].function.arguments"),
    ],
)
def test_chat_session_snapshot_restore_rejects_corrupted_tool_call_shape(
    corruption: str,
    param: str,
) -> None:
    fake = SequentialFakeLLM(
        ['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>']
    )
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    created = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "snapshot tool call"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "session": {"id": "sess_corrupt_tool"},
            "max_tokens": 4,
        },
    )
    snapshot = client.get("/v1/hipengine/sessions/sess_corrupt_tool/snapshot", headers=headers).json()
    tool_call = snapshot["messages"][1]["tool_calls"][0]

    if corruption == "non_object":
        snapshot["messages"][1]["tool_calls"][0] = "not-object"
    elif corruption == "extra_key":
        tool_call["unexpected"] = True
    elif corruption == "missing_id":
        tool_call["id"] = ""
    elif corruption == "wrong_type":
        tool_call["type"] = "custom"
    elif corruption == "missing_function":
        del tool_call["function"]
    elif corruption == "function_extra_key":
        tool_call["function"]["unexpected"] = True
    elif corruption == "empty_name":
        tool_call["function"]["name"] = ""
    elif corruption == "non_string_arguments":
        tool_call["function"]["arguments"] = {"path": "README.md"}
    elif corruption == "invalid_json_arguments":
        tool_call["function"]["arguments"] = '{"path":'
    else:
        raise AssertionError(f"unhandled corruption case: {corruption}")
    client.delete("/v1/hipengine/sessions/sess_corrupt_tool", headers=headers)

    restored = client.post(
        "/v1/hipengine/sessions/sess_corrupt_tool/snapshot",
        headers=headers,
        json=snapshot,
    )

    assert created.status_code == 200
    assert restored.status_code == 400
    assert restored.json()["error"]["code"] == "invalid_request"
    assert restored.json()["error"]["param"] == param
    assert "sess_corrupt_tool" not in app.state.hipengine_chat_sessions


@pytest.mark.parametrize(
    ("case", "first_output", "request_extra", "expected_reason", "rejected_text"),
    [
        (
            "invalid_tool_call",
            '<tool_call>{"name":"write","arguments":{"path":"README.md"}}</tool_call>',
            {},
            "invalid_tool_call",
            '"name":"write"',
        ),
        (
            "missing_required_tool",
            "ordinary answer",
            {"tool_choice": "required"},
            "tool_required_not_satisfied",
            "ordinary answer",
        ),
        (
            "schema_violation",
            '<tool_call>{"name":"read","arguments":{"path":7}}</tool_call>',
            {},
            "schema_violation",
            '"path":7',
        ),
    ],
)
def test_chat_session_visible_only_downgrades_strict_tool_failures_to_prompt_only(
    case: str,
    first_output: str,
    request_extra: dict[str, Any],
    expected_reason: str,
    rejected_text: str,
) -> None:
    fake = SequentialFakeLLM([first_output, "done"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    session_id = f"sess_tool_failure_{case}"

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "try tool"}],
            "tools": tools,
            "max_tokens": 4,
            "session": {"id": session_id, "commit": "append_visible_only"},
            **request_extra,
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "tools": tools,
            "max_tokens": 4,
            "session": {"id": session_id, "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_choice = first.json()["choices"][0]
    assert first_choice["finish_details"]["reason"] == expected_reason
    assert first_choice["finish_details"]["cache_action"] == "append_prompt_only"
    prompt = fake.calls[1][0][0]
    assert "try tool" in prompt
    assert "continue" in prompt
    assert rejected_text not in prompt
    record = app.state.hipengine_chat_sessions[session_id]
    assert record.messages == ({"role": "user", "content": "try tool"},)


def test_chat_session_visible_only_downgrades_unparseable_tool_markup_to_prompt_only() -> None:
    raw_tool_markup = '<tool_call>{"name":"read","arguments":</tool_call>'
    fake = SequentialFakeLLM([raw_tool_markup, "done"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
    ]

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "try tool"}],
            "tools": tools,
            "max_tokens": 4,
            "session": {"id": "sess_unparseable_tool", "commit": "append_visible_only"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "tools": tools,
            "max_tokens": 4,
            "session": {"id": "sess_unparseable_tool", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_choice = first.json()["choices"][0]
    assert first_choice["finish_details"]["reason"] == "invalid_tool_call"
    assert first_choice["finish_details"]["cache_action"] == "append_prompt_only"
    prompt = fake.calls[1][0][0]
    assert "try tool" in prompt
    assert "continue" in prompt
    assert raw_tool_markup not in prompt
    record = app.state.hipengine_chat_sessions["sess_unparseable_tool"]
    assert record.messages == ({"role": "user", "content": "try tool"},)


def test_chat_session_visible_only_commits_tool_calls_without_reasoning() -> None:
    fake = SequentialFakeLLM(
        [
            '<think>need file</think><tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>',
            "done",
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
    ]

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read it"}],
            "tools": tools,
            "max_tokens": 4,
            "session": {"id": "sess_tool", "commit": "append_visible_only"},
        },
    )
    assert first.status_code == 200
    tool_call_id = first.json()["choices"][0]["message"]["tool_calls"][0]["id"]
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "tool", "tool_call_id": tool_call_id, "content": "file text"}],
            "tools": tools,
            "max_tokens": 4,
            "session": {"id": "sess_tool", "commit": "append_none"},
        },
    )

    assert second.status_code == 200
    assert first.json()["choices"][0]["finish_reason"] == "tool_calls"
    prompt = fake.calls[1][0][0]
    assert (
        "<tool_call>\n<function=read>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>"
        in prompt
    )
    assert "<tool_response>\nfile text\n</tool_response>" in prompt
    assert "need file" not in prompt
    assert first.json()["choices"][0]["finish_details"]["cache_action"] == "append_visible_only"


@pytest.mark.parametrize(
    ("message", "param"),
    [
        (
            {
                "role": "user",
                "content": "hello",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            "messages[0].tool_calls",
        ),
        (
            {"role": "assistant", "content": "hello", "tool_call_id": "call_1"},
            "messages[0].tool_call_id",
        ),
        ({"role": "tool", "content": "tool result"}, "messages[0].tool_call_id"),
    ],
)
def test_chat_completion_rejects_invalid_role_specific_tool_fields_before_generation(
    message: dict[str, Any],
    param: str,
) -> None:
    fake = FakeLLM(outputs=["should not generate"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [message], "max_tokens": 4},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert fake.calls == []


@pytest.mark.parametrize(
    ("messages", "param"),
    [
        (
            [{"role": "tool", "tool_call_id": "call_missing", "content": "orphan tool result"}],
            "messages[0].tool_call_id",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        },
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "write", "arguments": "{}"},
                        },
                    ],
                }
            ],
            "messages[0].tool_calls[1].id",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "first result"},
                {"role": "tool", "tool_call_id": "call_1", "content": "duplicate result"},
            ],
            "messages[2].tool_call_id",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "user", "content": "skip the tool"},
            ],
            "messages[1].role",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "write", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "first result"},
                {"role": "user", "content": "skip the second tool"},
            ],
            "messages[2].role",
        ),
    ],
)
def test_chat_completion_rejects_invalid_tool_transcript_before_generation(
    messages: list[dict[str, Any]],
    param: str,
) -> None:
    fake = FakeLLM(outputs=["should not generate"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": messages, "max_tokens": 4},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert fake.calls == []


@pytest.mark.parametrize(
    ("corruption", "param"),
    [
        ("extra_key", "messages[0].tool_calls[0].unexpected"),
        ("missing_id", "messages[0].tool_calls[0].id"),
        ("wrong_type", "messages[0].tool_calls[0].type"),
        ("missing_function", "messages[0].tool_calls[0].function"),
        ("function_extra_key", "messages[0].tool_calls[0].function.unexpected"),
        ("empty_name", "messages[0].tool_calls[0].function.name"),
        ("non_string_arguments", "messages[0].tool_calls[0].function.arguments"),
        ("invalid_json_arguments", "messages[0].tool_calls[0].function.arguments"),
    ],
)
def test_chat_completion_rejects_invalid_prior_assistant_tool_call_shape_before_generation(
    corruption: str,
    param: str,
) -> None:
    tool_call: dict[str, Any] = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read", "arguments": '{"path":"README.md"}'},
    }
    if corruption == "extra_key":
        tool_call["unexpected"] = True
    elif corruption == "missing_id":
        del tool_call["id"]
    elif corruption == "wrong_type":
        tool_call["type"] = "custom"
    elif corruption == "missing_function":
        del tool_call["function"]
    elif corruption == "function_extra_key":
        tool_call["function"]["unexpected"] = True
    elif corruption == "empty_name":
        tool_call["function"]["name"] = ""
    elif corruption == "non_string_arguments":
        tool_call["function"]["arguments"] = {"path": "README.md"}
    elif corruption == "invalid_json_arguments":
        tool_call["function"]["arguments"] = '{"path":'
    else:
        raise AssertionError(f"unhandled corruption case: {corruption}")

    fake = FakeLLM(outputs=["should not generate"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "assistant", "content": "", "tool_calls": [tool_call]}],
            "max_tokens": 4,
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert fake.calls == []


def test_render_chat_prompt_includes_qwen_tool_blocks() -> None:
    prompt = render_chat_prompt(
        [
            {"role": "developer", "content": "Use tools carefully."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file text"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "read"}},
    )

    assert "<tools>" in prompt
    assert '"name":"read"' in prompt
    assert "You must call the function named 'read'." in prompt
    assert "<|im_start|>system\nUse tools carefully.<|im_end|>" in prompt
    assert '<tool_call>\n<function=read>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>' in prompt
    assert "<tool_response>\nfile text\n</tool_response>" in prompt


@pytest.mark.parametrize("result_count", [1, 2, 3])
@pytest.mark.parametrize("typed_messages", [False, True])
def test_render_chat_prompt_groups_consecutive_tool_results(result_count, typed_messages) -> None:
    from hipengine.server.api import ChatMessage

    contents = ["first result", "", "line one\nline two"][:result_count]
    calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }
        for index in range(result_count)
    ]
    messages = [{"role": "assistant", "content": None, "tool_calls": calls}]
    messages.extend(
        {"role": "tool", "tool_call_id": call["id"], "content": content}
        for call, content in zip(calls, contents)
    )
    if typed_messages:
        messages = [ChatMessage(**message) for message in messages]

    prompt = render_chat_prompt(messages)
    block = "\n".join(f"<tool_response>\n{content}\n</tool_response>" for content in contents)
    assert f"<|im_start|>user\n{block}<|im_end|>\n<|im_start|>assistant\n" in prompt
    assert prompt.count("<|im_start|>user\n") == 1


def test_render_chat_prompt_keeps_separate_tool_result_groups() -> None:
    messages = []
    for index in range(2):
        messages.extend([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": f"call_{index}", "content": f"result {index}"},
        ])
    messages.append({"role": "user", "content": "Summarize both results."})
    prompt = render_chat_prompt(messages)
    assert prompt.count("<|im_start|>user\n") == 3
    for index in range(2):
        assert f"<|im_start|>user\n<tool_response>\nresult {index}\n</tool_response><|im_end|>" in prompt
    assert "<|im_start|>user\nSummarize both results.<|im_end|>" in prompt


def test_chat_completion_returns_openai_tool_calls() -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"] == _stateless_finish_details(
        "tool_calls",
        tool_call_tokens=1,
        phase="tool_call",
    )
    message = choice["message"]
    assert message["content"] == ""
    tool_call = message["tool_calls"][0]
    assert tool_call["id"].startswith("call_")
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "read"
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "README.md"}
    assert "<tools>" in fake.calls[0][0][0]


def test_chat_completion_lifts_qwen35_family_xml_function_tool_call() -> None:
    fake = FakeLLM(outputs=[_QWEN35_XML_TOOL_CALL])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "weather in Tokyo"}],
            "tools": [_WEATHER_TOOL],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] == ""
    _assert_openai_tool_call_shape(
        message["tool_calls"][0],
        name="get_weather",
        arguments={"city": "Tokyo", "days": 3},
    )
    assert "<function=" not in json.dumps(message)


def test_chat_completion_preserves_reasoning_with_qwen35_xml_tool_call() -> None:
    fake = FakeLLM(
        outputs=["<think>The user wants Tokyo weather.</think>\n\n" + _QWEN35_XML_TOOL_CALL]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "weather in Tokyo"}],
            "tools": [_WEATHER_TOOL],
            "enable_thinking": True,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] == ""
    assert message["reasoning_content"] == "The user wants Tokyo weather."
    _assert_openai_tool_call_shape(
        message["tool_calls"][0],
        name="get_weather",
        arguments={"city": "Tokyo", "days": 3},
    )
    assert "<tool_call>" not in json.dumps(message)


def test_chat_completion_tools_prompt_matches_qwen35_family_template() -> None:
    fake = FakeLLM(outputs=[_QWEN35_XML_TOOL_CALL])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "weather in Tokyo"}],
            "tools": [_WEATHER_TOOL],
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[0][0][0]
    assert "<function=example_function_name>" in prompt
    assert "<parameter=example_parameter_1>" in prompt
    # The legacy JSON-envelope instruction must not fight the trained format.
    assert '"name":"function_name"' not in prompt


def test_parse_chat_tool_calls_lifts_xml_multiline_string_values_verbatim() -> None:
    raw = (
        "<tool_call>\n"
        "<function=record_notes>\n"
        "<parameter=notes>\n"
        "first line\nsecond line\n"
        "\n</parameter>\n"
        "<parameter=priority>\n"
        "2\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    parsed = _parse_chat_tool_calls(raw, tools=None)

    assert parsed.text == ""
    (call,) = parsed.tool_calls
    assert call.name == "record_notes"
    assert json.loads(call.arguments) == {
        "notes": "first line\nsecond line\n",
        "priority": 2,
    }


def test_render_chat_prompt_renders_assistant_tool_calls_in_family_xml_format() -> None:
    prompt = render_chat_prompt(
        [
            {"role": "user", "content": "weather in Tokyo"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Tokyo", "days": 3}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        tools=[_WEATHER_TOOL],
    )

    assert (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nTokyo\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n"
        "</function>\n</tool_call>"
    ) in prompt
    # String arguments render verbatim; non-string arguments as compact JSON.
    assert '<parameter=days>\n"3"\n</parameter>' not in prompt


def test_chat_completion_strips_qwen_terminal_residue_around_tool_call() -> None:
    fake = FakeLLM(
        outputs=[
            '<|im_end|><tool_call>{"name":"read","arguments":{"path":"README.md"}}'
            '</tool_call><|im_end|><|im_end|>'
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "read"
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "README.md"}
    assert "<|im_end|>" not in response.text


def test_chat_completion_strips_qwen_role_template_residue_around_tool_call() -> None:
    fake = FakeLLM(
        outputs=[
            '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'
            '<|endoftext|><|im_start|><|im_start|><|im_start|>user\n'
            '<|im_start|><|im_start|>'
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "read"
    assert "<|endoftext|>" not in response.text
    assert "<|im_start|>" not in response.text


def test_chat_completion_strips_incomplete_duplicate_tool_prefix_control_residue() -> None:
    fake = FakeLLM(outputs=[_incomplete_duplicate_read_tool_output()])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read pyproject"}],
            "tool_choice": {"type": "function", "function": {"name": "read"}},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    _assert_openai_tool_call_shape(
        choice["message"]["tool_calls"][0],
        name="read",
        arguments={"path": "pyproject.toml", "mode": "summary"},
    )
    assert "<|im_start|>" not in response.text


def test_chat_completion_preserves_mismatched_incomplete_duplicate_tool_prefix() -> None:
    valid_call = (
        '<tool_call>{"name":"read","arguments":{"path":"pyproject.toml",'
        '"mode":"summary"}}</tool_call>'
    )
    raw_output = _incomplete_duplicate_read_tool_output().replace(
        '"name":"read"',
        '"name":"write"',
        1,
    )
    expected_content = raw_output.replace(valid_call, "").strip()
    fake = FakeLLM(outputs=[raw_output])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read pyproject"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "read", "parameters": {"type": "object"}},
                },
                {
                    "type": "function",
                    "function": {"name": "write", "parameters": {"type": "object"}},
                },
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == expected_content
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "read"


def test_chat_completion_preserves_reasoning_with_openai_tool_call() -> None:
    fake = FakeLLM(
        outputs=['<think>need file</think><tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>']
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"] == _stateless_finish_details(
        "tool_calls",
        reasoning_tokens=2,
        tool_call_tokens=1,
        phase="tool_call",
    )
    message = choice["message"]
    assert message["content"] == ""
    assert message["reasoning_content"] == "need file"
    tool_call = message["tool_calls"][0]
    assert tool_call["function"]["name"] == "read"
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "README.md"}
    assert "<tool_call>" not in json.dumps(message)


def test_chat_completion_parses_tool_call_after_literal_marker_text() -> None:
    fake = FakeLLM(
        outputs=[
            'Mention <tool_call> literally, then call. '
            '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] == "Mention <tool_call> literally, then call."
    assert len(message["tool_calls"]) == 1
    _assert_openai_tool_call_shape(message["tool_calls"][0], name="read", arguments={"path": "README.md"})


def test_chat_completion_parses_tool_call_with_marker_text_in_arguments() -> None:
    fake = FakeLLM(
        outputs=[
            '<tool_call>{"name":"read","arguments":{"path":"README.md",'
            '"needle":"</tool_call>"}}</tool_call>'
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] == ""
    _assert_openai_tool_call_shape(
        message["tool_calls"][0],
        name="read",
        arguments={"path": "README.md", "needle": "</tool_call>"},
    )


def test_chat_completion_auto_tool_rejects_literal_paired_tool_markup() -> None:
    raw_tool_markup = "Use <tool_call> to invoke tools.</tool_call>"
    fake = FakeLLM(outputs=[raw_tool_markup])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "explain tool syntax"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert raw_tool_markup not in response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_auto_tool_recovers_duplicated_start_marker() -> None:
    fake = FakeLLM(
        outputs=['<tool_call>\n<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>']
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"] == _stateless_finish_details(
        "tool_calls",
        tool_call_tokens=2,
        phase="tool_call",
    )
    message = choice["message"]
    assert set(message) == {"role", "content", "tool_calls"}
    assert message["role"] == "assistant"
    assert message["content"] == ""
    assert "<tool_call>" not in json.dumps(message)
    assert isinstance(message["tool_calls"], list)
    assert len(message["tool_calls"]) == 1
    tool_call = message["tool_calls"][0]
    _assert_openai_tool_call_shape(tool_call, name="read", arguments={"path": "README.md"})


def test_chat_completion_auto_tool_rejects_unparseable_tool_markup() -> None:
    raw_tool_markup = '<tool_call>{"name":"read","arguments":</tool_call>'
    fake = FakeLLM(outputs=[raw_tool_markup])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert raw_tool_markup not in response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_auto_tool_rejects_undeclared_function() -> None:
    fake = FakeLLM(
        outputs=['<tool_call>{"name":"write","arguments":{"path":"README.md"}}</tool_call>']
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    assert "write" not in response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_invalid_tool_call_can_return_hard_error() -> None:
    fake = FakeLLM(
        outputs=['<tool_call>{"name":"write","arguments":{"path":"README.md"}}</tool_call>']
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "invalid_tool_call_error_mode": "hard_error",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 400
    assert "<tool_call>" not in response.text
    assert "write" not in response.text
    error = response.json()["error"]
    assert error["code"] == "invalid_tool_call"
    assert error["param"] == "tool_calls"
    assert error["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert error["hipengine"] == {
        "code": "invalid_tool_call",
        "status_code": 400,
        "retryable": False,
    }
    assert len(fake.calls) == 1


def test_chat_completion_rejects_invalid_tool_call_error_mode_before_generation() -> None:
    fake = FakeLLM(outputs=["unused"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "invalid_tool_call_error_mode": "sometimes",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["param"] == "invalid_tool_call_error_mode"
    assert fake.calls == []


def test_chat_completion_strict_validation_recovers_doubled_tool_call_tag() -> None:
    fake = FakeLLM(
        outputs=['<tool_call>\n<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>']
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "strict": True,
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"] == _stateless_finish_details(
        "tool_calls",
        tool_call_tokens=2,
        phase="tool_call",
    )
    message = choice["message"]
    assert set(message) == {"role", "content", "tool_calls"}
    assert message["role"] == "assistant"
    assert message["content"] == ""
    assert isinstance(message["tool_calls"], list)
    assert len(message["tool_calls"]) == 1
    tool_call = message["tool_calls"][0]
    _assert_openai_tool_call_shape(tool_call, name="read", arguments={"path": "README.md"})
    assert "<tool_call>" not in response.text


def test_chat_completion_required_tool_reports_missing_call() -> None:
    fake = FakeLLM(outputs=["ordinary answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("tool_required_not_satisfied")
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_required_tool_choice_rejects_missing_tools_without_generation() -> None:
    fake = FakeLLM(outputs=["ordinary answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": "required",
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "tool_choice"
    assert error["hipengine"]["code"] == "schema_violation"
    assert fake.calls == []


def test_chat_completion_required_tool_missing_call_preserves_length_context() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="ordinary answer",
                finish_details=FinishDetails(reason="length", length_limit=7),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": "required",
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "tool_required_not_satisfied",
        length_limit=7,
        phase="answer",
        continuation_eligible=False,
    )
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_required_tool_partial_call_preserves_length_context() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='<tool_call>{"name":"read"',
                finish_details=FinishDetails(reason="length", length_limit=6),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": "required",
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "invalid_tool_call",
        length_limit=6,
        phase="tool_call",
        continuation_eligible=False,
    )
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_specific_tool_rejects_wrong_function() -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"write","arguments":{"path":"README.md"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": {"type": "function", "function": {"name": "read"}},
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "write", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_specific_tool_choice_rejects_unknown_function_without_generation() -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": {"type": "function", "function": {"name": "write"}},
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "tool_choice.function.name"
    assert error["hipengine"]["code"] == "schema_violation"
    assert fake.calls == []


@pytest.mark.parametrize(
    ("tool_choice", "param"),
    [
        ({"type": "custom", "function": {"name": "read"}}, "tool_choice.type"),
        ({"type": "function"}, "tool_choice.function"),
        ({"type": "function", "function": "read"}, "tool_choice.function"),
        ({"type": "function", "function": {}}, "tool_choice.function.name"),
        ({"type": "function", "function": {"name": ""}}, "tool_choice.function.name"),
    ],
)
def test_chat_completion_malformed_tool_choice_rejects_without_generation(
    tool_choice: dict[str, Any],
    param: str,
) -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": tool_choice,
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert error["hipengine"]["code"] == "schema_violation"
    assert fake.calls == []


def test_chat_completion_tool_choice_none_rejects_tool_call() -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "answer without tools"}],
            "tool_choice": "none",
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_auto_tool_builds_xml_grammar_without_stripping_close() -> None:
    fake = FakeLLM(
        outputs=["ordinary answer"],
        token_map={"</tool_call>": [88, 89]},
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "choose the right tool"}],
            "tool_choice": "auto",
            "enable_thinking": False,
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "grep", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert fake.tokenize_calls == ["</tool_call>"]
    params = fake.calls[-1][1]
    assert params.tool_call_constraint.tool_names == ("read", "grep")
    assert params.tool_call_constraint.mode == "auto"
    assert params.tool_call_constraint.forbidden_text_prefixes == ("<think>",)
    assert params.tool_call_constraint.thinking_start_marker is None
    assert params.tool_call_constraint.thinking_end_marker is None
    assert params.tool_call_constraint.envelope == "xml"
    assert params.force_sequence_completion_token_sequences == ()
    assert params.force_sequence_completion_reason is None
    assert params.stop_token_sequences == ()


def test_chat_completion_tool_constraint_requires_tokenize_and_detokenize() -> None:
    fake = FakeLLM(outputs=["ordinary answer"], token_map={"</tool_call>": [88, 89]})
    fake.detokenize = None
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    capabilities = client.get("/v1/hipengine/capabilities")
    assert capabilities.status_code == 200
    tools_capability = capabilities.json()["features"]["tools"]
    assert tools_capability["strict_decoding"] is False
    assert tools_capability["tool_call_close_repair"] is False
    assert tools_capability["tool_call_close_repair_scope"] == "none"
    assert tools_capability["strict_tool_schema_prefix_anchor"] is False
    assert capabilities.json()["features"]["structured_outputs"]["strict_decoding"] is False

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "choose the right tool"}],
            "tool_choice": "auto",
            "enable_thinking": False,
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert fake.calls[-1][1].tool_call_constraint is None


def test_chat_completion_tool_choice_none_suppresses_tool_call_start_token() -> None:
    fake = FakeLLM(outputs=["plain answer"], token_map={"<tool_call>": [77, 78]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "answer without tools"}],
            "tool_choice": "none",
            "suppress_token_ids": [13],
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert fake.tokenize_calls == ["<tool_call>"]
    assert fake.calls[-1][1].suppress_token_ids == (13, 77)
    assert response.json()["choices"][0]["message"]["content"] == "plain answer"


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "read"}},
    ],
)
def test_chat_completion_required_tool_choice_forces_atomic_tool_xml_prefix(tool_choice) -> None:
    fake = FakeLLM(
        outputs=["ordinary answer"],
        token_map={
            "<tool_call>": [77, 78],
            '<tool_call>\n<function=read>\n': [77, 78, 90, 91, 92],
            "</tool_call>": [88, 89],
        },
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": tool_choice,
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert fake.tokenize_calls == [
        "<tool_call>",
        '<tool_call>\n<function=read>\n',
        "</tool_call>",
    ]
    params = fake.calls[-1][1]
    assert params.forced_tokens_pending == (77, 78, 90, 91, 92)
    assert params.forced_token_reason == "tool_choice_required"
    assert params.tool_call_constraint.thinking_start_marker == "<think>"
    assert params.tool_call_constraint.thinking_end_marker == "</think>"
    assert params.force_sequence_completion_token_sequences == ()
    assert params.force_sequence_completion_reason is None
    assert params.stop_token_sequences == ()
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("tool_required_not_satisfied")
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_required_tool_choice_queues_tool_start_after_thinking_budget() -> None:
    fake = FakeLLM(
        outputs=["ordinary answer"],
        token_map={
            "</think>": [91, 92],
            "<tool_call>": [77, 78],
            '<tool_call>\n<function=read>\n': [77, 78, 90, 91, 92],
            "</tool_call>": [88, 89],
        },
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": {"type": "function", "function": {"name": "read"}},
            "reasoning": {"enabled": True, "effort": "low"},
            "max_tokens": 2048,
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert fake.tokenize_calls == [
        "</think>",
        "<tool_call>",
        '<tool_call>\n<function=read>\n',
        "</tool_call>",
    ]
    params = fake.calls[-1][1]
    assert params.forced_tokens_pending == ()
    assert params.post_thinking_forced_tokens_pending == (77, 78, 90, 91, 92)
    assert params.post_thinking_forced_token_reason == "tool_choice_required"
    assert params.tool_call_constraint.thinking_start_marker == "<think>"
    assert params.tool_call_constraint.thinking_end_marker == "</think>"
    assert params.force_sequence_completion_token_sequences == ()
    assert params.force_sequence_completion_reason is None
    assert params.stop_token_sequences == ()
    assert params.thinking_close_token_ids == (91, 92)
    assert params.thinking_hard_token_cap == 512
    choice = response.json()["choices"][0]
    assert choice["finish_details"] == _stateless_finish_details("tool_required_not_satisfied")


def test_chat_completion_required_tool_choice_skips_name_prefix_with_multiple_tools() -> None:
    fake = FakeLLM(
        outputs=["ordinary answer"],
        token_map={
            "<tool_call>": [77, 78],
            "</tool_call>": [88, 89],
        },
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read or write"}],
            "tool_choice": "required",
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "write", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert fake.tokenize_calls == ["<tool_call>", "</tool_call>"]
    params = fake.calls[-1][1]
    assert params.forced_tokens_pending == (77, 78)
    assert params.force_sequence_completion_token_sequences == ()
    assert params.force_sequence_completion_reason is None
    assert params.stop_token_sequences == ()


@pytest.mark.parametrize(
    ("schema_prefix_ids", "expected_close_sequences"),
    [
        ((93, 94, 95, 96), ()),
        ((27, 12445, 13766, 13042, 29, 591), ()),
        ((27, 510, 13766, 13042, 29, 591), ()),
    ],
)
def test_chat_completion_specific_tool_choice_forces_xml_header_after_tool_result(
    schema_prefix_ids: tuple[int, ...],
    expected_close_sequences: tuple[tuple[int, ...], ...],
) -> None:
    schema_prefix = '<tool_call>\n<function=grep>\n'
    fake = FakeLLM(
        outputs=["ordinary answer"],
        token_map={
            "<tool_call>": [77, 78],
            '<tool_call>{"name":"grep","arguments":': [90, 91, 92],
            schema_prefix: list(schema_prefix_ids),
            "</tool_call>": [510, 13766, 13042, 29],
            "}}</tool_call>": [12445, 13766, 13042, 29],
        },
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [
                {"role": "user", "content": "read the readme"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_read",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_read", "content": "file text"},
                {"role": "user", "content": "grep the scheduler"},
            ],
            "tool_choice": {"type": "function", "function": {"name": "grep"}},
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
                {
                    "type": "function",
                    "function": {
                        "name": "grep",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "pattern": {"type": "string", "minLength": 1},
                                "path": {"type": "string", "minLength": 1},
                            },
                            "required": ["pattern", "path"],
                            "additionalProperties": False,
                        },
                    },
                },
            ],
        },
    )

    assert response.status_code == 200
    assert fake.tokenize_calls == [
        "<tool_call>",
        schema_prefix,
        "</tool_call>",
    ]
    params = fake.calls[-1][1]
    assert params.forced_tokens_pending == schema_prefix_ids
    assert params.forced_token_reason == "tool_choice_required"
    assert params.force_sequence_completion_token_sequences == expected_close_sequences
    assert params.force_sequence_completion_reason == ("tool_call_sequence_completion" if expected_close_sequences else None)
    assert params.stop_token_sequences == expected_close_sequences


def test_chat_completion_specific_tool_choice_falls_back_for_non_string_first_required_schema() -> None:
    fake = FakeLLM(
        outputs=["ordinary answer"],
        token_map={
            "<tool_call>": [77, 78],
            '<tool_call>\n<function=count>\n': [90, 91, 92],
            "</tool_call>": [88, 89],
        },
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "count the matches"}],
            "tool_choice": {"type": "function", "function": {"name": "count"}},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "count",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {"limit": {"type": "integer"}},
                            "required": ["limit"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert fake.tokenize_calls == [
        "<tool_call>",
        '<tool_call>\n<function=count>\n',
        "</tool_call>",
    ]
    params = fake.calls[-1][1]
    assert params.forced_tokens_pending == (90, 91, 92)
    assert params.force_sequence_completion_token_sequences == ()
    assert params.stop_token_sequences == ()


def test_chat_completion_strict_tool_schema_reports_schema_violation() -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":123,"mode":"raw"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in choice["message"]


def test_chat_completion_strict_tool_schema_rejects_unsupported_keywords() -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "$anchor": "read_args",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "tools[0].function.parameters.$anchor"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_chat_completion_strict_tool_schema_validates_string_pattern() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "pattern": r"^README[.]md$"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>']),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"WORKLOG.md"}}</tool_call>']),
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "read the readme"}],
        "tools": tools,
    }

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert json.loads(valid_choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "path": "README.md"
    }
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_rejects_invalid_pattern_without_generation() -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string", "pattern": "["}},
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "tools[0].function.parameters.properties.path.pattern"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_chat_completion_strict_tool_schema_validates_property_count() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "minProperties": 2,
                    "maxProperties": 2,
                    "properties": {
                        "path": {"type": "string"},
                        "mode": {"type": "string"},
                        "extra": {"type": "string"},
                    },
                },
            },
        }
    ]
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md","mode":"raw"}}</tool_call>']
        ),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=[
                '<tool_call>{"name":"read","arguments":{"path":"README.md","mode":"raw","extra":"x"}}</tool_call>'
            ]
        ),
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "read the readme"}],
        "tools": tools,
    }

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_additional_properties_schema() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": {"type": "string"},
                },
            },
        }
    ]
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"ok":true,"note":"done"}}</tool_call>']),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"ok":true,"note":7}}</tool_call>']),
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_pattern_properties() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "patternProperties": {r"^x-[a-z]+$": {"type": "integer"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"ok":true,"x-count":2}}</tool_call>']),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"ok":true,"x-count":"two"}}</tool_call>']),
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_object_name_keywords() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "propertyNames": {"pattern": r"^(id|name|x-[a-z]+)$"},
                    "dependentRequired": {"id": ["name"]},
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                    },
                },
            },
        }
    ]
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=['<tool_call>{"name":"record","arguments":{"id":"1","name":"doc","x-extra":"ok"}}</tool_call>']
        ),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"id":"1","x-extra":"ok"}}</tool_call>']),
    )

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_dependent_schemas() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["file", "url"]},
                        "path": {"type": "string"},
                        "href": {"type": "string", "pattern": r"^https://"},
                    },
                    "dependentSchemas": {
                        "path": {
                            "required": ["kind"],
                            "properties": {"kind": {"const": "file"}},
                        },
                        "href": {
                            "required": ["kind"],
                            "properties": {"kind": {"const": "url"}},
                        },
                    },
                    "additionalProperties": False,
                },
            },
        }
    ]
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=['<tool_call>{"name":"record","arguments":{"kind":"file","path":"README.md"}}</tool_call>']
        ),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=['<tool_call>{"name":"record","arguments":{"kind":"url","path":"README.md"}}</tool_call>']
        ),
    )

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_conditional_schemas() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["file", "url"]},
                        "path": {"type": "string"},
                        "href": {"type": "string", "pattern": r"^https://"},
                    },
                    "if": {"required": ["kind"], "properties": {"kind": {"const": "file"}}},
                    "then": {"required": ["path"], "not": {"required": ["href"]}},
                    "else": {"required": ["href"], "not": {"required": ["path"]}},
                    "additionalProperties": False,
                },
            },
        }
    ]
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=['<tool_call>{"name":"record","arguments":{"kind":"file","path":"README.md"}}</tool_call>']
        ),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=[
                '<tool_call>{"name":"record","arguments":{"kind":"file","href":"https://example.test"}}</tool_call>'
            ]
        ),
    )

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_local_refs() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "definitions": {
                        "file_ref": {"type": "string", "pattern": r"^[A-Z]+[.]md$"},
                        "result_tag": {"enum": ["ok", "skip"]},
                    },
                    "properties": {
                        "path": {"$ref": "#/definitions/file_ref"},
                        "result": {"$ref": "#/definitions/result_tag"},
                    },
                    "required": ["path", "result"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=['<tool_call>{"name":"record","arguments":{"path":"README.md","result":"ok"}}</tool_call>']
        ),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=['<tool_call>{"name":"record","arguments":{"path":"readme.md","result":"ok"}}</tool_call>']
        ),
    )

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_multiple_of() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"score": {"type": "number", "multipleOf": 0.5}},
                    "required": ["score"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"score":1.5}}</tool_call>']),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"score":1.25}}</tool_call>']),
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record score"}],
        "tools": tools,
    }

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_unique_items() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
                    },
                    "required": ["tags"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"tags":["docs","api"]}}</tool_call>']),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"tags":["docs","docs"]}}</tool_call>']),
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record tags"}],
        "tools": tools,
    }

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_array_contains() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scores": {
                            "type": "array",
                            "contains": {"type": "number", "minimum": 0.9},
                            "minContains": 1,
                            "maxContains": 2,
                        }
                    },
                    "required": ["scores"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record score"}],
        "tools": tools,
    }
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"scores":[0.1,0.95]}}</tool_call>']),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"scores":[0.1,0.2]}}</tool_call>']),
    )

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_uses_json_typed_equality() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "const_num": {"const": 1},
                        "enum_num": {"enum": [1]},
                        "mixed": {"type": "array", "uniqueItems": True},
                    },
                    "required": ["const_num", "enum_num", "mixed"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=[
                '<tool_call>{"name":"record","arguments":{"const_num":1,"enum_num":1,'
                '"mixed":[true,1,"1"]}}</tool_call>'
            ]
        ),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=[
                '<tool_call>{"name":"record","arguments":{"const_num":true,"enum_num":1,'
                '"mixed":[true,1,"1"]}}</tool_call>'
            ]
        ),
    )

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_composition_keywords() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
                        "score": {
                            "type": "number",
                            "allOf": [{"minimum": 0}, {"maximum": 1}],
                        },
                        "label": {
                            "type": "string",
                            "allOf": [{"minLength": 2}, {"maxLength": 4}],
                        },
                        "ambiguous": {"oneOf": [{"type": "number"}, {"type": "integer"}]},
                        "debug": {"not": {"const": True}},
                    },
                    "required": ["target", "score", "label", "ambiguous", "debug"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=[
                (
                    '<tool_call>{"name":"record","arguments":'
                    '{"target":null,"score":0.5,"label":"ok","ambiguous":1.5,"debug":false}}</tool_call>'
                )
            ]
        ),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=[
                (
                    '<tool_call>{"name":"record","arguments":'
                    '{"target":null,"score":0.5,"label":"x","ambiguous":1.5,"debug":false}}</tool_call>'
                )
            ]
        ),
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_accepts_annotation_keywords() -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "strict": True,
                        "parameters": {
                            "title": "Read arguments",
                            "description": "Annotation-only metadata is ignored by validation.",
                            "default": {"path": "README.md"},
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Repository path",
                                    "examples": ["README.md"],
                                    "format": "uri-reference",
                                }
                            },
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"]["reason"] == "tool_calls"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"path": "README.md"}


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"path": "README.md", "mode": "raw"},
    ],
)
def test_chat_completion_strict_tool_schema_rejects_missing_and_extra_arguments(arguments) -> None:
    fake = FakeLLM(outputs=[f'<tool_call>{{"name":"read","arguments":{json.dumps(arguments)}}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")


def test_chat_completion_strict_tool_schema_accepts_bounded_subset() -> None:
    arguments = {
        "kind": "file",
        "path": "README.md",
        "mode": "summary",
        "tags": ["docs"],
        "filters": [{"field": "ext", "value": "md"}],
        "limit": 2,
    }
    fake = FakeLLM(outputs=[f'<tool_call>{{"name":"read","arguments":{json.dumps(arguments)}}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "strict": True,
                        "parameters": _bounded_tool_schema(),
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"]["reason"] == "tool_calls"
    assert choice["finish_details"]["phase"] == "tool_call"
    assert choice["finish_details"]["tool_call_tokens"] > 0
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == arguments


@pytest.mark.parametrize(
    "arguments",
    [
        {"kind": "directory", "path": "README.md", "mode": "summary", "tags": ["docs"], "limit": 2},
        {"kind": "file", "path": "", "mode": "summary", "tags": ["docs"], "limit": 2},
        {"kind": "file", "path": "README.md", "mode": "binary", "tags": ["docs"], "limit": 2},
        {"kind": "file", "path": "README.md", "mode": "summary", "tags": [], "limit": 2},
        {"kind": "file", "path": "README.md", "mode": "summary", "tags": ["a", "b", "c"], "limit": 2},
        {"kind": "file", "path": "README.md", "mode": "summary", "tags": ["docs"], "limit": 4},
        {
            "kind": "file",
            "path": "README.md",
            "mode": "summary",
            "tags": ["docs"],
            "filters": [{"field": "suffix", "value": "md"}],
            "limit": 2,
        },
        {
            "kind": "file",
            "path": "README.md",
            "mode": "summary",
            "tags": ["docs"],
            "filters": [{"field": "ext"}],
            "limit": 2,
        },
    ],
)
def test_chat_completion_strict_tool_schema_rejects_bounded_subset_violations(arguments) -> None:
    fake = FakeLLM(outputs=[f'<tool_call>{{"name":"read","arguments":{json.dumps(arguments)}}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "strict": True,
                        "parameters": _bounded_tool_schema(),
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in choice["message"]


@pytest.mark.parametrize("parallel", [None, True])
def test_chat_completion_parallel_tool_calls_default_on_and_false_disables(parallel) -> None:
    output = (
        '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'
        '<tool_call>{"name":"read","arguments":{"path":"WORKLOG.md"}}</tool_call>'
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    rejected = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=[output]),
    )
    accepted = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=[output]),
    )

    rejected_response = TestClient(rejected).post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "read files"}], "tools": tools, "parallel_tool_calls": False},
    )
    accepted_response = TestClient(accepted).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read files"}],
            "tools": tools,
            **({} if parallel is None else {"parallel_tool_calls": parallel}),
        },
    )

    assert rejected_response.json()["choices"][0]["finish_details"] == _stateless_finish_details("invalid_tool_call")
    accepted_choice = accepted_response.json()["choices"][0]
    assert accepted_choice["finish_reason"] == "tool_calls"
    assert len(accepted_choice["message"]["tool_calls"]) == 2


@pytest.mark.parametrize("parallel", [None, True, False])
def test_xml_parallel_default_matches_sampling_and_streaming(parallel) -> None:
    chunks = [
        '<tool_call>\n<function=bash>\n<parameter=command>\necho one\n</parameter>\n</function>\n</tool_call>',
        '<tool_call>\n<function=bash>\n<parameter=command>\necho two\n</parameter>\n</function>\n</tool_call>',
    ]
    fake = FakeLLM(outputs=['unused'], stream_chunks=chunks)
    body = {
        'model': 'fake-model', 'messages': [{'role': 'user', 'content': 'run both'}],
        'stream': True, 'chat_template_kwargs': {'enable_thinking': False},
        'tools': [{'type': 'function', 'function': {'name': 'bash', 'parameters': {
            'type': 'object', 'properties': {'command': {'type': 'string'}},
            'required': ['command'],
        }}}],
        **({} if parallel is None else {'parallel_tool_calls': parallel}),
    }
    from hipengine.server.api import ChatCompletionRequest, _tool_call_sampling_constraint
    constraint = _tool_call_sampling_constraint(
        ChatCompletionRequest(**body), fake, chat_default_max_tokens=None,
    )
    assert constraint.parallel_tool_calls is (parallel is not False)
    client = TestClient(create_app(ServerConfig(model='fake-path', served_model_name='fake-model'), llm=fake))
    payloads = _sse_payloads(client.post('/v1/chat/completions', json=body).text)
    if parallel is False:
        assert any(p.get('error', {}).get('code') == 'invalid_tool_call' for p in payloads)
    else:
        calls = [c for p in payloads for choice in p.get('choices', [])
                 for c in choice.get('delta', {}).get('tool_calls', [])]
        assert len({c['id'] for c in calls}) == 2
        assert payloads[-1]['choices'][0]['finish_reason'] == 'tool_calls'


@pytest.mark.parametrize("truncated", [False, True])
def test_streaming_xml_tool_emits_incremental_arguments_and_errors_on_truncation(truncated):
    chunks = ['<tool_call>\n<function=bash>\n<parameter=command>\n', 'echo ', 'live']
    if not truncated:
        chunks.append('\n</parameter>\n</function>\n</tool_call>')
    fake = FakeLLM(outputs=['unused'], stream_chunks=chunks)
    client = TestClient(create_app(ServerConfig(model='fake-path', served_model_name='fake-model'), llm=fake))
    response = client.post('/v1/chat/completions', json={
        'model': 'fake-model', 'messages': [{'role': 'user', 'content': 'run command'}],
        'stream': True, 'chat_template_kwargs': {'enable_thinking': False},
        'tools': [{'type': 'function', 'function': {'name': 'bash', 'parameters': {
            'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command'],
        }}}],
    })
    payloads = _sse_payloads(response.text)
    calls = [call for payload in payloads for choice in payload.get('choices', [])
             for call in choice.get('delta', {}).get('tool_calls', [])]
    assert len(calls) >= 4
    assert len({call['id'] for call in calls}) == 1
    assert calls[0]['function']['name'] == 'bash'
    if truncated:
        assert any(payload.get('error', {}).get('code') == 'invalid_tool_call' for payload in payloads)
        assert not any(choice.get('finish_reason') in {'stop', 'tool_calls'}
                       for payload in payloads for choice in payload.get('choices', []))
    else:
        assert json.loads(''.join(call['function']['arguments'] for call in calls)) == {'command': 'echo live'}
        assert payloads[-1]['choices'][0]['finish_reason'] == 'tool_calls'


def test_streaming_chat_completion_returns_tool_call_deltas() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=['<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    payloads = _sse_payloads(response.text)
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    tool_delta = next(payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls"))
    tool_call = tool_delta["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["index"] == 0
    assert tool_call["id"].startswith("call_")
    assert tool_call["function"]["name"] == "bash"
    assert json.loads(tool_call["function"]["arguments"]) == {"command": "pwd"}
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert payloads[-1]["choices"][0]["finish_details"] == _stateless_finish_details(
        "tool_calls",
        tool_call_tokens=1,
        phase="tool_call",
    )


def test_streaming_chat_tool_done_exposes_response_owned_generated_ids() -> None:
    raw_tool_call = '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                raw_tool_call,
                generated_token_ids=(701, 702, 703),
                finish_details=FinishDetails(reason="stop"),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=9,
                    generated_tokens=3,
                    row_index=0,
                    phase="tool_call",
                    sampler_mode="greedy_fast",
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
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    done = next(
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] == "tool_calls"
    )
    assert done["hipengine"]["generated_token_ids"] == [701, 702, 703]
    assert done["hipengine"]["generated_tokens"] == 3
    assert payloads[-1]["usage"]["completion_tokens"] == 3


def test_streaming_chat_completion_strips_qwen_terminal_residue_around_tool_call() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            '<|im_end|><tool_call>{"name":"bash","arguments":{"command":"pwd"}}'
            '</tool_call><|im_end|><|im_end|>'
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<|im_end|>" not in response.text
    payloads = _sse_payloads(response.text)
    content = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
        if payload.get("choices")
    )
    assert content == ""
    tool_choice = next(
        payload["choices"][0]
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
    )
    _assert_openai_stream_tool_call_delta_shape(
        tool_choice,
        name="bash",
        arguments={"command": "pwd"},
    )
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_streaming_chat_completion_strips_qwen_role_template_residue_around_tool_call() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'
            '<|endoftext|><|im_start|><|im_start|><|im_start|><|im_start|><|im_start|>'
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
        if payload.get("choices")
    )
    assert content == ""
    tool_choice = next(
        payload["choices"][0]
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
    )
    _assert_openai_stream_tool_call_delta_shape(
        tool_choice,
        name="bash",
        arguments={"command": "pwd"},
    )
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert "<|endoftext|>" not in response.text
    assert "<|im_start|>" not in response.text


def test_streaming_chat_completion_strips_incomplete_duplicate_tool_prefix_control_residue() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[_incomplete_duplicate_read_tool_output()],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read pyproject"}],
            "stream": True,
            "tool_choice": {"type": "function", "function": {"name": "read"}},
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "read", "parameters": {"type": "object"}},
                }
            ],
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
        if payload.get("choices")
    )
    assert content == ""
    tool_choice = next(
        payload["choices"][0]
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
    )
    _assert_openai_stream_tool_call_delta_shape(
        tool_choice,
        name="read",
        arguments={"path": "pyproject.toml", "mode": "summary"},
    )
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert "<|im_start|>" not in response.text


def test_streaming_chat_completion_parses_tool_call_after_literal_marker_text() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            "Mention <tool_call> literally, then call. ",
            '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>',
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert '<tool_call>{"name"' not in response.text
    payloads = _sse_payloads(response.text)
    content = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
        if payload.get("choices")
    )
    assert content == "Mention <tool_call> literally, then call."
    tool_choice = next(payload["choices"][0] for payload in payloads if payload["choices"][0]["delta"].get("tool_calls"))
    _assert_openai_stream_tool_call_delta_shape(
        tool_choice,
        name="read",
        arguments={"path": "README.md"},
    )
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_streaming_chat_completion_n_uses_scheduler_chunks_for_tool_call_arguments() -> None:
    raw_outputs = (
        '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',
        '<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>',
    )
    chunk_texts = (
        (
            '<tool_call>{"name":"bash","arguments":',
            '{"command":',
            '"pwd"}',
            "}</tool_call>",
        ),
        (
            '<tool_call>{"name":"bash","arguments":',
            '{"command":',
            '"ls"}',
            "}</tool_call>",
        ),
    )

    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": request_id,
                        "token_index": token_index,
                        "token_id": 600 + request_id * 10 + token_index,
                        "finished": token_index == len(row) - 1,
                        "chunk": {
                            "text": text,
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=token_index + 1,
                                row_index=request_id,
                                request_id=str(request_id),
                                phase="answer",
                                sampler_mode="greedy_fast",
                                execution_path="scheduler_tool_call_chunks",
                            ).to_json_dict(),
                        },
                    }
                    for request_id, row in enumerate(chunk_texts)
                    for token_index, text in enumerate(row)
                ]
            }
            return [
                GenerationOutput(
                    text=raw_outputs[index],
                    generated_token_ids=tuple(
                        600 + index * 10 + token_index
                        for token_index in range(len(chunk_texts[index]))
                    ),
                    finish_details=FinishDetails(reason="stop", sampler_mode="greedy_fast"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=len(chunk_texts[index]),
                        row_index=index,
                        phase="done",
                        sampler_mode="greedy_fast",
                    ),
                )
                for index, _prompt in enumerate(prompt_tuple)
            ]

    fake = SchedulerChunkFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run shell commands"}],
            "n": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    payloads = _sse_payloads(response.text)
    tool_choices = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert [
        (
            choice["index"],
            choice["delta"]["tool_calls"][0]["function"].get("name"),
            choice["delta"]["tool_calls"][0]["function"]["arguments"],
        )
        for choice in tool_choices
    ] == [
        (0, "bash", '{"command":'),
        (0, None, '"pwd"}'),
        (1, "bash", '{"command":'),
        (1, None, '"ls"}'),
    ]
    arguments_by_choice: dict[int, str] = {}
    for choice in tool_choices:
        call = choice["delta"]["tool_calls"][0]
        arguments_by_choice.setdefault(choice["index"], "")
        arguments_by_choice[choice["index"]] += call["function"]["arguments"]
    assert {index: json.loads(arguments) for index, arguments in arguments_by_choice.items()} == {
        0: {"command": "pwd"},
        1: {"command": "ls"},
    }
    assert {
        choice["hipengine"]["phase"]
        for choice in tool_choices
    } == {"tool_call"}
    assert {
        choice["hipengine"]["decode_state"]["phase"]
        for choice in tool_choices
    } == {"tool_call"}
    assert {
        choice["hipengine"]["decode_state"]["execution_path"]
        for choice in tool_choices
    } == {"scheduler_tool_call_chunks"}
    done = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] == "tool_calls"
    ]
    assert [choice["index"] for choice in done] == [0, 1]
    assert [choice["hipengine"]["generated_token_ids"] for choice in done] == [
        [600, 601, 602, 603],
        [610, 611, 612, 613],
    ]
    assert {
        choice["hipengine"]["decode_state"]["execution_path"]
        for choice in done
    } == {"scheduler_tool_call_chunks"}
    assert fake.stream_calls == []


def test_streaming_chat_completion_n_reports_withheld_scheduler_tool_chunks_for_invalid_tool_call() -> None:
    raw_outputs = (
        '<tool_call>{"name":"write","arguments":{"path":"README.md"}}</tool_call>',
        '<tool_call>{"name":"write","arguments":{"path":"WORKLOG.md"}}</tool_call>',
    )
    fake = SchedulerChunkRowsFakeLLM(
        raw_outputs,
        (
            (
                '<tool_call>{"name":"write","arguments":',
                '{"path":"README.md"}}</tool_call>',
            ),
            (
                '<tool_call>{"name":"write","arguments":',
                '{"path":"WORKLOG.md"}}</tool_call>',
            ),
        ),
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read files"}],
            "n": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    assert "write" not in response.text
    assert "README.md" not in response.text
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0]["delta"].get("tool_calls") for payload in payloads)
    done = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"]
    ]
    assert [choice["index"] for choice in done] == [0, 1]
    for index, choice in enumerate(done):
        assert choice["finish_reason"] == "stop"
        assert choice["finish_details"] == _stateless_finish_details("invalid_tool_call")
        diagnostic = choice["hipengine"]["withheld_scheduler_tool_chunks"]
        assert diagnostic == {
            "surface": "chat_tool_argument_delta",
            "reason": "invalid_tool_call",
            "public_delta": "withheld",
            "chunk_count": 2,
            "malformed_chunk_count": 0,
            "raw_text_matches": True,
            "text_bytes": len(raw_outputs[index].encode("utf-8")),
            "text_sha256": hashlib.sha256(raw_outputs[index].encode("utf-8")).hexdigest(),
            "chunk_text_bytes": [
                len(chunk.encode("utf-8")) for chunk in fake.chunk_rows[index]
            ],
            "execution_paths": ["scheduler_tool_call_chunks"],
        }
    assert fake.stream_calls == []


def test_streaming_chat_completion_n_reports_unmappable_scheduler_tool_chunks() -> None:
    raw_output = '<tool_call>{"name":"bash","arguments":"{\\"command\\":\\"pwd\\"}"}</tool_call>'
    fake = SchedulerChunkRowsFakeLLM(
        (raw_output, raw_output),
        (
            (
                '<tool_call>{"name":"bash","arguments":',
                '"{\\"command\\":\\"pwd\\"}"}</tool_call>',
            ),
            (
                '<tool_call>{"name":"bash","arguments":',
                '"{\\"command\\":\\"pwd\\"}"}</tool_call>',
            ),
        ),
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "n": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    payloads = _sse_payloads(response.text)
    tool_choices = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert [choice["index"] for choice in tool_choices] == [0, 1]
    for choice in tool_choices:
        _assert_openai_stream_tool_call_delta_shape(
            choice,
            name="bash",
            arguments={"command": "pwd"},
            index=choice["index"],
        )
        assert choice["hipengine"]["phase"] == "tool_call"
        assert choice["hipengine"]["decode_state"]["phase"] == "tool_call"
        assert choice["hipengine"]["decode_state"].get("execution_path") != "scheduler_tool_call_chunks"

    done = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"]
    ]
    assert [choice["index"] for choice in done] == [0, 1]
    for choice in done:
        assert choice["finish_reason"] == "tool_calls"
        assert choice["finish_details"] == _stateless_finish_details(
            "tool_calls",
            tool_call_tokens=1,
            phase="tool_call",
        )
        diagnostic = choice["hipengine"]["withheld_scheduler_tool_chunks"]
        assert diagnostic["reason"] == "unmappable_tool_arguments"
        assert diagnostic["public_delta"] == "withheld"
        assert diagnostic["chunk_count"] == 2
        assert diagnostic["raw_text_matches"] is True
        assert diagnostic["text_sha256"] == hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        assert diagnostic["execution_paths"] == ["scheduler_tool_call_chunks"]
        assert "text" not in diagnostic
    assert fake.stream_calls == []


def test_streaming_chat_completion_recovers_duplicated_tool_start_marker() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=['<tool_call>\n<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    payloads = _sse_payloads(response.text)
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    tool_delta = next(payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls"))
    _assert_openai_stream_tool_call_delta_shape(
        tool_delta["choices"][0],
        name="bash",
        arguments={"command": "pwd"},
    )
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["delta"] == {}
    assert done["choices"][0]["finish_reason"] == "tool_calls"
    assert done["choices"][0]["finish_details"] == _stateless_finish_details(
        "tool_calls",
        tool_call_tokens=2,
        phase="tool_call",
    )


def test_streaming_chat_completion_rejects_undeclared_auto_tool_name() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=['<tool_call>{"name":"write","arguments":{"path":"README.md"}}</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    assert "write" not in response.text
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0]["delta"].get("tool_calls") for payload in payloads)
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_reason"] == "stop"
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("invalid_tool_call")


def test_streaming_chat_completion_invalid_tool_call_can_return_sse_error() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=['<tool_call>{"name":"write","arguments":{"path":"README.md"}}</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "invalid_tool_call_error_mode": "hard_error",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    assert "write" not in response.text
    payloads = _sse_payloads(response.text)
    error_payload = next(payload for payload in payloads if payload.get("error"))
    assert error_payload["choices"][0]["finish_reason"] == "error"
    assert error_payload["choices"][0]["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert error_payload["error"]["code"] == "invalid_tool_call"
    assert error_payload["error"]["param"] == "tool_calls"
    assert error_payload["error"]["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert error_payload["error"]["hipengine"] == {
        "code": "invalid_tool_call",
        "status_code": 400,
        "retryable": False,
    }
    assert error_payload["hipengine"]["event"] == "error"
    assert "data: [DONE]" in response.text


def test_streaming_chat_completion_auto_tool_rejects_unparseable_tool_markup() -> None:
    raw_tool_markup = '<tool_call>{"name":"bash","arguments":</tool_call>'
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=[raw_tool_markup])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert raw_tool_markup not in response.text
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0]["delta"].get("tool_calls") for payload in payloads)
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_reason"] == "stop"
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("invalid_tool_call")


def test_streaming_chat_completion_chunks_long_tool_call_arguments() -> None:
    command = "x" * 300
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[f'<tool_call>{{"name":"bash","arguments":{{"command":{json.dumps(command)}}}}}</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run a long command"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    payloads = _sse_payloads(response.text)
    chunks = [
        payload["choices"][0]["delta"]["tool_calls"][0]
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(chunks) > 1
    assert {chunk["id"] for chunk in chunks} == {chunks[0]["id"]}
    assert {chunk["index"] for chunk in chunks} == {0}
    assert chunks[0]["function"]["name"] == "bash"
    assert all("name" not in chunk["function"] for chunk in chunks[1:])
    argument_text = "".join(chunk["function"]["arguments"] for chunk in chunks)
    assert json.loads(argument_text) == {"command": command}
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_streaming_chat_completion_preserves_reasoning_with_tool_call() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            '<think>need shell</think><tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    payloads = _sse_payloads(response.text)
    reasoning = next(payload for payload in payloads if payload["choices"][0]["delta"].get("reasoning_content"))
    assert reasoning["choices"][0]["delta"] == {"reasoning_content": "need shell"}
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
    tool_delta = next(payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls"))
    tool_call = tool_delta["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "bash"
    assert json.loads(tool_call["function"]["arguments"]) == {"command": "pwd"}
    assert tool_delta["choices"][0]["hipengine"] == {
        "phase": "tool_call",
        "tokens": {
            "streamed_tokens": 3,
            "delta_tokens": 1,
            "reasoning_tokens": 2,
            "tool_call_tokens": 1,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 3,
            "prompt_tokens": 0,
            "generated_tokens": 3,
            "phase": "tool_call",
            "continuation_eligible": False,
            "reasoning_tokens": 2,
            "tool_call_tokens": 1,
        },
    }
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert payloads[-1]["choices"][0]["finish_details"] == _stateless_finish_details(
        "tool_calls",
        reasoning_tokens=2,
        tool_call_tokens=1,
        phase="tool_call",
    )
    done_hipengine = payloads[-1]["choices"][0]["hipengine"]
    assert done_hipengine["phase"] == "done"
    assert done_hipengine["finish_details"] == _stateless_finish_details(
        "tool_calls",
        reasoning_tokens=2,
        tool_call_tokens=1,
        phase="tool_call",
    )
    assert done_hipengine["tokens"]["streamed_tokens"] == 3
    assert done_hipengine["tokens"]["reasoning_tokens"] == 2
    assert done_hipengine["tokens"]["tool_call_tokens"] == 1
    assert done_hipengine["tokens"]["prompt_tokens"] == fake.count_tokens(fake.calls[0][0][0])
    assert done_hipengine["decode_state"]["step_index"] == 3
    assert done_hipengine["decode_state"]["prompt_tokens"] == done_hipengine["tokens"]["prompt_tokens"]
    assert done_hipengine["decode_state"]["generated_tokens"] == done_hipengine["tokens"]["completion_tokens"]
    assert done_hipengine["decode_state"]["phase"] == "done"
    assert done_hipengine["decode_state"]["reasoning_tokens"] == 2
    assert done_hipengine["decode_state"]["tool_call_tokens"] == 1


def test_streaming_chat_completion_strict_validation_recovers_doubled_tool_call_tag() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=['<tool_call>\n<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "strict": True,
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    payloads = _sse_payloads(response.text)
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    tool_payload = next(payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls"))
    _assert_openai_stream_tool_call_delta_shape(
        tool_payload["choices"][0],
        name="bash",
        arguments={"command": "pwd"},
    )
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["delta"] == {}
    assert done["choices"][0]["finish_reason"] == "tool_calls"
    assert done["choices"][0]["finish_details"] == _stateless_finish_details(
        "tool_calls",
        tool_call_tokens=2,
        phase="tool_call",
    )


def test_streaming_chat_completion_strict_validation_rejects_malformed_tool_json() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=['<tool_call>{"name":"bash","arguments":</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "strict": True,
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0]["delta"].get("tool_calls") for payload in payloads)
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_reason"] == "stop"
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("invalid_tool_call")


def test_streaming_chat_completion_preserves_parallel_tool_call_indexes() -> None:
    output = (
        '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'
        '<tool_call>{"name":"read","arguments":{"path":"WORKLOG.md"}}</tool_call>'
    )
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=[output])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read files"}],
            "stream": True,
            "parallel_tool_calls": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    tool_calls = [
        payload["choices"][0]["delta"]["tool_calls"][0]
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert [call["index"] for call in tool_calls] == [0, 1]
    assert [call["function"]["name"] for call in tool_calls] == ["read", "read"]
    assert [json.loads(call["function"]["arguments"]) for call in tool_calls] == [
        {"path": "README.md"},
        {"path": "WORKLOG.md"},
    ]
    assert tool_calls[0]["id"] != tool_calls[1]["id"]
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert payloads[-1]["choices"][0]["finish_details"] == _stateless_finish_details(
        "tool_calls",
        tool_call_tokens=2,
        phase="tool_call",
    )


def test_streaming_chat_completion_reports_strict_tool_schema_failure() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=['<tool_call>{"name":"bash","arguments":{"command":7}}</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    assert '"command":7' not in response.text
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0]["delta"].get("tool_calls") for payload in payloads)
    assert not any(payload["choices"][0]["delta"].get("content") for payload in payloads)
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1]["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")


def test_render_chat_prompt_accepts_plain_message_mappings() -> None:
    assert render_chat_prompt([{"role": "user", "content": "hello"}]) == (
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    )
