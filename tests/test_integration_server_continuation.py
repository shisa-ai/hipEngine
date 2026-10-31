"""OpenAI server integration tests: continuation.

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
    FakeLLM,
    SequentialFakeLLM,
)
from tests.support.server_api_support import (
    _stateless_finish_details,
)


def test_completion_continuation_resumes_buffered_length_finish_once() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="length", length_limit=1),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=2,
                    generated_tokens=1,
                    sampler_mode="greedy_fast",
                ),
            ),
            GenerationOutput(text=" beta", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert continuation_id.startswith("gen_")
    assert first_choice["text"] == "alpha"
    assert first_choice["finish_reason"] == "length"
    assert first_choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=1,
        continuation_eligible=True,
        continuation_id=continuation_id,
    )
    assert first_choice["hipengine"]["decode_state"] == {
        "row_index": 0,
        "step_index": 1,
        "prompt_tokens": 2,
        "generated_tokens": 1,
        "phase": "done",
        "continuation_eligible": True,
        "sampler_mode": "greedy_fast",
    }

    second = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["text"] == "alpha beta"
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["finish_details"] == _stateless_finish_details("eos", eos_token_id=151645)
    assert fake.calls[0][0] == ("Say: ",)
    assert fake.calls[1][0] == ("Say: alpha",)

    reused = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 1},
    )

    assert reused.status_code == 400
    assert reused.json()["error"]["code"] == "invalid_continuation"
    assert len(fake.calls) == 2


def test_completion_continuation_is_scoped_to_auth_principal() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" beta", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    first = client.post(
        "/v1/completions",
        headers=headers,
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]
    record = app.state.hipengine_continuations[continuation_id]
    assert record.auth_principal.startswith("bearer_sha256:")
    assert "secret" not in record.auth_principal

    app.state.hipengine_continuations[continuation_id] = replace(
        record,
        auth_principal="bearer_sha256:other",
    )
    rejected = client.post(
        "/v1/completions",
        headers=headers,
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "invalid_continuation"
    assert rejected.json()["error"]["param"] == "continuation_id"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    app.state.hipengine_continuations[continuation_id] = record
    resumed = client.post(
        "/v1/completions",
        headers=headers,
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["text"] == "alpha beta"
    assert continuation_id not in app.state.hipengine_continuations
    assert len(fake.calls) == 2


def test_completion_continuation_is_scoped_to_session_id() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" beta", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]
    record = app.state.hipengine_continuations[continuation_id]
    assert record.session_id is None

    app.state.hipengine_continuations[continuation_id] = replace(record, session_id="other-session")
    rejected = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "invalid_continuation"
    assert rejected.json()["error"]["param"] == "continuation_id"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    app.state.hipengine_continuations[continuation_id] = record
    resumed = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["text"] == "alpha beta"
    assert continuation_id not in app.state.hipengine_continuations
    assert len(fake.calls) == 2


def test_completion_continuation_is_scoped_to_tokenizer_metadata() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" beta", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]
    record = app.state.hipengine_continuations[continuation_id]
    assert record.tokenizer == {
        "name": "SequentialFakeLLM",
        "tokenize": True,
        "detokenize": True,
        "count_tokens": True,
    }

    app.state.hipengine_continuations[continuation_id] = replace(
        record,
        tokenizer={**record.tokenizer, "name": "OtherTokenizer"},
    )
    rejected = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "invalid_continuation"
    assert rejected.json()["error"]["param"] == "continuation_id"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    app.state.hipengine_continuations[continuation_id] = record
    resumed = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["text"] == "alpha beta"
    assert continuation_id not in app.state.hipengine_continuations
    assert len(fake.calls) == 2


def test_completion_continuation_resume_rejects_prompt_without_consuming_handle() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" beta", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    with_prompt = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "prompt": "ignored prompt",
            "max_tokens": 4,
        },
    )

    assert with_prompt.status_code == 400
    assert with_prompt.json()["error"]["code"] == "unsupported_parameter"
    assert with_prompt.json()["error"]["param"] == "prompt"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    resumed = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["text"] == "alpha beta"
    assert len(fake.calls) == 2


def test_completion_length_finish_with_stop_is_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1, continuation_eligible=True),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=2,
                    generated_tokens=1,
                    sampler_mode="greedy_fast",
                    continuation_eligible=True,
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "Say: ",
            "stop": "<end>",
            "max_tokens": 1,
            "temperature": 0.0,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert "continuation_id" not in choice
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=1,
        continuation_eligible=False,
    )
    assert choice["hipengine"]["decode_state"] == {
        "row_index": 0,
        "step_index": 1,
        "prompt_tokens": 2,
        "generated_tokens": 1,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_mode": "greedy_fast",
    }


def test_completion_length_finish_honors_backend_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1, continuation_eligible=False),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=2,
                    generated_tokens=1,
                    sampler_mode="greedy_fast",
                    continuation_eligible=True,
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == "partial"
    assert "continuation_id" not in choice
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=1,
        continuation_eligible=False,
    )
    assert choice["hipengine"]["decode_state"] == {
        "row_index": 0,
        "step_index": 1,
        "prompt_tokens": 2,
        "generated_tokens": 1,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_mode": "greedy_fast",
    }


def test_completion_length_finish_with_ignore_eos_is_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=2,
                    generated_tokens=1,
                    sampler_mode="greedy_fast",
                    continuation_eligible=True,
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "Say: ",
            "ignore_eos": True,
            "max_tokens": 1,
            "temperature": 0.0,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert "continuation_id" not in choice
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=1,
        continuation_eligible=False,
    )
    assert choice["hipengine"]["decode_state"] == {
        "row_index": 0,
        "step_index": 1,
        "prompt_tokens": 2,
        "generated_tokens": 1,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_mode": "greedy_fast",
    }


def test_completion_continuation_resume_rejects_explicit_stop_without_consuming_handle() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" beta", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    with_stop = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "stop": "<end>",
            "max_tokens": 4,
        },
    )

    assert with_stop.status_code == 400
    assert with_stop.json()["error"]["code"] == "unsupported_parameter"
    assert with_stop.json()["error"]["param"] == "stop"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    resumed = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["text"] == "alpha beta"
    assert len(fake.calls) == 2


def test_completion_continuation_resume_rejects_ignore_eos_without_consuming_handle() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" beta", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    with_ignore_eos = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "ignore_eos": True,
            "max_tokens": 4,
        },
    )

    assert with_ignore_eos.status_code == 400
    assert with_ignore_eos.json()["error"]["code"] == "unsupported_parameter"
    assert with_ignore_eos.json()["error"]["param"] == "ignore_eos"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    resumed = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["text"] == "alpha beta"
    assert len(fake.calls) == 2


def test_completion_length_finish_marks_sampled_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="sampled partial",
                finish_details=FinishDetails(reason="length", length_limit=2),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "sample", "max_tokens": 2, "temperature": 0.7},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert "continuation_id" not in choice
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=2,
        continuation_eligible=False,
    )


def test_completion_length_finish_with_n_gt_one_is_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial one",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(
                text="partial two",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "n": 2},
    )

    assert response.status_code == 200
    choices = response.json()["choices"]
    assert len(choices) == 2
    for choice in choices:
        assert "continuation_id" not in choice
        assert choice["finish_details"] == _stateless_finish_details(
            "length",
            length_limit=1,
            continuation_eligible=False,
        )


def test_chat_length_finish_with_n_gt_one_is_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial one",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(
                text="partial two",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 1,
            "n": 2,
        },
    )

    assert response.status_code == 200
    choices = response.json()["choices"]
    assert len(choices) == 2
    for choice in choices:
        assert "continuation_id" not in choice
        assert choice["finish_details"] == _stateless_finish_details(
            "length",
            length_limit=1,
            phase="answer",
            continuation_eligible=False,
        )


def test_chat_length_finish_with_reasoning_effort_is_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial answer",
                finish_details=FinishDetails(reason="length", length_limit=2),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "max_tokens": 16,
            "reasoning_effort": "low",
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert "continuation_id" not in choice
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=2,
        phase="answer",
        continuation_eligible=False,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reasoning_effort", "low"),
        ("max_think_tokens", 16),
        ("min_answer_tokens", 4),
        ("hard_think_cap", 16),
        ("soft_close_window", 4),
        ("hard_close_message", "answer now"),
        ("hard_close_sequence", "</think>\n"),
        ("thinking_token_budget", 16),
        ("chat_template_kwargs", {"thinking_budget": 16}),
        ("thinking", {"budget_tokens": 16}),
        ("reasoning", {"allow_unbounded": True}),
    ],
)
def test_chat_continuation_resume_rejects_thinking_controls_without_consuming_handle(
    field: str,
    value: Any,
) -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="partial answer",
                finish_details=FinishDetails(reason="length", length_limit=2),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 2,
        },
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    capabilities = client.get("/v1/hipengine/capabilities").json()
    assert field in capabilities["sessions"]["continuations"]["unsupported_resume_fields"]
    resumed = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "max_tokens": 4,
            field: value,
        },
    )

    assert resumed.status_code == 400
    assert resumed.json()["error"]["code"] == "unsupported_parameter"
    assert resumed.json()["error"]["param"] == field
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1


def test_chat_continuation_resume_rejects_messages_without_consuming_handle() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" answer", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 1,
        },
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    with_messages = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "messages": [{"role": "user", "content": "ignored follow-up"}],
            "max_tokens": 4,
        },
    )

    assert with_messages.status_code == 400
    assert with_messages.json()["error"]["code"] == "unsupported_parameter"
    assert with_messages.json()["error"]["param"] == "messages"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    resumed = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["message"]["content"] == "partial answer"
    assert len(fake.calls) == 2


@pytest.mark.parametrize(
    ("field", "value", "expected_param"),
    [
        ("stream", True, "stream"),
        ("n", 2, "n"),
    ],
)
def test_chat_continuation_resume_rejects_lower_loop_fields_without_consuming_handle(
    field: str,
    value: Any,
    expected_param: str,
) -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" answer", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 1,
        },
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    capabilities = client.get("/v1/hipengine/capabilities").json()
    assert expected_param in capabilities["sessions"]["continuations"]["unsupported_resume_fields"]
    rejected = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "max_tokens": 4,
            field: value,
        },
    )

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "unsupported_parameter"
    assert rejected.json()["error"]["param"] == expected_param
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    resumed = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["message"]["content"] == "partial answer"
    assert continuation_id not in app.state.hipengine_continuations
    assert len(fake.calls) == 2


def test_chat_session_continuation_resumes_buffered_length_finish_once() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" answer", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "session": {"id": "sess_continue"},
            "max_tokens": 1,
            "temperature": 0.0,
        },
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert continuation_id.startswith("gen_")
    assert first_choice["finish_reason"] == "length"
    assert first_choice["finish_details"] == {
        "reason": "length",
        "length_limit": 1,
        "cache_action": "append_prompt_only",
        "phase": "answer",
        "continuation_eligible": True,
        "continuation_id": continuation_id,
    }
    assert app.state.hipengine_continuations[continuation_id].session_id == "sess_continue"
    assert app.state.hipengine_chat_sessions["sess_continue"].messages == (
        {"role": "user", "content": "continue"},
    )

    resumed = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "session": {"id": "sess_continue"},
            "max_tokens": 4,
        },
    )

    assert resumed.status_code == 200
    resumed_choice = resumed.json()["choices"][0]
    assert resumed_choice["message"]["content"] == "partial answer"
    assert resumed_choice["finish_reason"] == "stop"
    assert resumed_choice["finish_details"]["cache_action"] == "append_visible_only"
    assert continuation_id not in app.state.hipengine_continuations
    assert app.state.hipengine_chat_sessions["sess_continue"].messages == (
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "partial answer"},
    )
    assert "partial" in fake.calls[1][0][0]

    reused = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "session": {"id": "sess_continue"},
            "max_tokens": 1,
        },
    )

    assert reused.status_code == 400
    assert reused.json()["error"]["code"] == "invalid_continuation"
    assert len(fake.calls) == 2


def test_chat_session_continuation_rejects_deleted_session_without_generation() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" should not run", finish_details=FinishDetails(reason="eos")),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "session": {"id": "sess_deleted"},
            "max_tokens": 1,
        },
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    deleted = client.delete("/v1/hipengine/sessions/sess_deleted")
    rejected = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "session": {"id": "sess_deleted"},
            "max_tokens": 4,
        },
    )

    assert deleted.json()["deleted"] is True
    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "invalid_continuation"
    assert rejected.json()["error"]["param"] == "continuation_id"
    assert continuation_id in app.state.hipengine_continuations
    assert "sess_deleted" not in app.state.hipengine_chat_sessions
    assert len(fake.calls) == 1


def test_completion_continuation_expiration_reports_stable_error() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "start", "max_tokens": 1},
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]
    record = app.state.hipengine_continuations[continuation_id]
    app.state.hipengine_continuations[continuation_id] = replace(record, expires_at=time.time() - 1)

    expired = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 1},
    )

    assert expired.status_code == 410
    assert expired.json()["error"]["code"] == "continuation_expired"
    assert expired.json()["error"]["param"] == "continuation_id"
    assert len(fake.calls) == 1


def test_chat_length_finish_honors_backend_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial answer",
                finish_details=FinishDetails(reason="length", length_limit=2, continuation_eligible=False),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=2,
                    generated_tokens=2,
                    sampler_mode="greedy_fast",
                    continuation_eligible=True,
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "continue"}]},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert "continuation_id" not in choice
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=2,
        phase="answer",
        continuation_eligible=False,
    )
    assert choice["message"] == {"role": "assistant", "content": "partial answer"}
    assert choice["hipengine"]["decode_state"] == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 2,
        "generated_tokens": 2,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_mode": "greedy_fast",
    }


def test_server_rejects_unknown_continuation_id_without_generation() -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": "gen_123"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_continuation"
    assert response.json()["error"]["param"] == "continuation_id"
    assert fake.calls == []
