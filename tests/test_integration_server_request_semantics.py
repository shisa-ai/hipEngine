"""OpenAI server integration tests: request semantics.

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
    DetailedGenerateFakeLLM,
    FakeLLM,
)
from tests.support.server_api_support import (
    _incomplete_duplicate_read_tool_output,
    _routing_metadata,
    _sse_payloads,
    _stateless_finish_details,
)


def test_chat_default_max_tokens_is_dynamic_when_omitted() -> None:
    request = ChatCompletionRequest(
        model="fake-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert request.max_tokens is None


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_min_tokens_uses_model_eos_without_explicit_request_id(chat, stream) -> None:
    fake = FakeLLM(outputs=["yes yes"])
    fake._text_generator = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=99))
    client = TestClient(create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake))
    request = {"model": "fake-model", "max_tokens": 4, "min_tokens": 2, "stream": stream}
    if chat:
        request["messages"] = [{"role": "user", "content": "Say yes"}]
    else:
        request["prompt"] = "Say yes"
    response = client.post("/v1/chat/completions" if chat else "/v1/completions", json=request)
    assert response.status_code == 200, response.text
    assert fake.calls[0][1].min_tokens == 2
    assert fake.calls[0][1].eos_token_id == 99


def test_completions_endpoint_calls_llm_and_applies_stop() -> None:
    fake = FakeLLM(outputs=["alpha<stop>tail", "beta<stop>tail"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            kv_storage="int8_per_token_head",
            kv_scale_dtype="fp32",
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": ["one", "two"],
            "max_tokens": 3,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": "<stop>",
            "suppress_token_ids": [12, 13],
            "min_tokens": 2,
            "eos_token_id": 151645,
            "kv_storage": "int8_per_token_head",
            "kv_scale_dtype": "fp32",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "fake-model"
    assert body["hipengine"]["routing"] == _routing_metadata()
    assert [choice["text"] for choice in body["choices"]] == ["alpha", "beta"]
    assert [choice["finish_reason"] for choice in body["choices"]] == ["stop", "stop"]
    # A server-side stop names the string that matched, and does not keep a
    # backend cause beside it.
    assert [choice["finish_details"] for choice in body["choices"]] == [
        _stateless_finish_details("stop", matched_stop="<stop>"),
        _stateless_finish_details("stop", matched_stop="<stop>"),
    ]
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    assert fake.calls == [
        (
            ("one", "two"),
            SamplingParams(
                max_tokens=3,
                temperature=0.0,
                top_p=1.0,
                suppress_token_ids=(12, 13),
                min_tokens=2,
                eos_token_id=151645,
                ignore_eos=False,
                kv_storage="int8_per_token_head",
                kv_scale_dtype="fp32",
            ),
        )
    ]


def test_completions_exact_token_prompt_preserves_ids_and_identity() -> None:
    class ExactTokenFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams):
            prompt_rows = tuple(tuple(int(token) for token in row) for row in prompts)
            self.calls.append((prompt_rows, sampling_params))
            return [
                GenerationOutput(text="answer", generated_token_ids=(701, 702))
                for _row in prompt_rows
            ]

    fake = ExactTokenFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": [10, 11, 12, 13],
            "max_tokens": 2,
            "ignore_eos": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert fake.calls[0][0] == ((10, 11, 12, 13),)
    assert body["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
    }
    assert body["hipengine"]["prompt_token_accounting"] == {
        "schema_version": 1,
        "input_type": "token_ids",
        "prompt_token_ids_sha256": [
            "fe52e32025ab5b0c96a9dff2cf661e34f4ce86fe760a6399212957aee39c6bde"
        ],
        "prompt_tokens": [4],
        "total_prompt_tokens": 4,
    }
    assert body["hipengine"]["token_accounting"]["choice_generated_token_ids"] == [[701, 702]]
    capabilities = client.get("/v1/hipengine/capabilities").json()
    exact = capabilities["features"]["exact_token_prompts"]
    assert exact["completions"] is True
    assert exact["streaming"] is True
    assert exact["response_identity"] == "hipengine.prompt_token_accounting"


def test_completions_exact_token_prompt_streams_without_retokenizing_and_rejects_mixed_rows() -> None:
    fake = FakeLLM(stream_chunks=["alpha", " beta"])
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=fake,
    )
    client = TestClient(app)

    streamed = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": [10, 11],
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )
    mixed = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": [[10, 11], "text"]},
    )

    assert streamed.status_code == 200
    payloads = _sse_payloads(streamed.text)
    assert "".join(
        payload["choices"][0]["text"]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] is None
    ) == "alpha beta"
    assert fake.calls[0][0] == ((10, 11),)
    usage = next(payload["usage"] for payload in payloads if payload.get("usage"))
    assert usage == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    assert mixed.status_code in {400, 422}


def test_completions_exact_token_rows_expand_n_without_id_loss() -> None:
    class ExactRowsFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams):
            rows = tuple(tuple(int(token) for token in row) for row in prompts)
            self.calls.append((rows, sampling_params))
            return [
                GenerationOutput(text=f"row-{index}", generated_token_ids=(800 + index,))
                for index, _row in enumerate(rows)
            ]

    fake = ExactRowsFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": [[10, 11, 12], [20, 21, 22]],
            "max_tokens": 1,
            "ignore_eos": True,
            "n": 2,
        },
    )

    assert response.status_code == 200
    expected_rows = ((10, 11, 12), (10, 11, 12), (20, 21, 22), (20, 21, 22))
    assert fake.calls[0][0] == expected_rows
    body = response.json()
    assert body["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }
    assert body["hipengine"]["prompt_token_accounting"]["prompt_tokens"] == [3, 3, 3, 3]
    assert body["hipengine"]["token_accounting"]["choice_generated_token_ids"] == [
        [800],
        [801],
        [802],
        [803],
    ]


@pytest.mark.parametrize(
    ("endpoint", "request_payload"),
    [
        ("/v1/completions", {"prompt": "one", "n": 6}),
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "one"}], "n": 6},
        ),
    ],
)
def test_openai_n6_uses_exact_all_choice_generated_token_accounting(
    endpoint: str,
    request_payload: dict[str, Any],
) -> None:
    token_rows = (
        (101, 102),
        (201,),
        (301, 302, 303),
        (),
        (501, 502),
        (601,),
    )
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text=f"singleword{index}",
                generated_token_ids=token_ids,
            )
            for index, token_ids in enumerate(token_rows)
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        endpoint,
        json={"model": "fake-model", "max_tokens": 3, **request_payload},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["choices"]) == 6
    assert body["usage"]["completion_tokens"] == 9
    accounting = body["hipengine"]["token_accounting"]
    assert accounting == {
        "choice_generated_token_ids": [list(row) for row in token_rows],
        "choice_generated_tokens": [2, 1, 3, 0, 2, 1],
        "total_generated_tokens": 9,
        "retokenized_visible_tokens": 6,
    }
    assert [choice["hipengine"]["generated_token_ids"] for choice in body["choices"]] == [
        list(row) for row in token_rows
    ]
    assert [choice["hipengine"]["generated_tokens"] for choice in body["choices"]] == [
        2,
        1,
        3,
        0,
        2,
        1,
    ]


def test_completions_preserve_structured_finish_details() -> None:
    fake = DetailedGenerateFakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="eos", eos_token_id=151645, sampler_mode="greedy_fast"),
            ),
            GenerationOutput(
                text="beta",
                finish_details=FinishDetails(reason="length", length_limit=2, budget_pressure="answer_budget"),
            ),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": ["one", "two"], "max_tokens": 2, "logit_bias": {"12": -1.0}},
    )

    assert response.status_code == 200
    choices = response.json()["choices"]
    assert [choice["finish_reason"] for choice in choices] == ["stop", "length"]
    assert choices[0]["finish_details"] == _stateless_finish_details(
        "eos",
        eos_token_id=151645,
        sampler_mode="greedy_fast",
    )
    assert choices[1]["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=2,
        budget_pressure="answer_budget",
        continuation_eligible=False,
    )


def test_completions_expose_backend_generation_telemetry() -> None:
    fake = DetailedGenerateFakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="answer",
                finish_details=FinishDetails(reason="eos", eos_token_id=151645, sampler_mode="greedy_fast"),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=3,
                    generated_tokens=1,
                    row_index=0,
                    phase="done",
                    sampler_mode="greedy_fast",
                    forced_token_id=42,
                    forced_token_reason="tool_choice_required",
                    forced_tokens_remaining=0,
                    active_processors=("logit_bias",),
                    sampler_fast_path_blockers=("logit_bias",),
                    timing={"prefill_ms": 2.5, "decode_ms": 1.25},
                    usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                    diagnostics={"batch_execution": {"path": "scheduler_native_compact_batch"}},
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "one two three", "max_tokens": 1},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == "answer"
    assert choice["hipengine"] == {
        "decode_state": {
            "row_index": 0,
            "step_index": 1,
            "prompt_tokens": 3,
            "generated_tokens": 1,
            "phase": "done",
            "continuation_eligible": False,
            "forced_token_id": 42,
            "forced_token_reason": "tool_choice_required",
            "forced_tokens_remaining": 0,
            "active_processors": ["logit_bias"],
            "sampler_fast_path_blockers": ["logit_bias"],
            "sampler_mode": "greedy_fast",
        },
        "timing": {"prefill_ms": 2.5, "decode_ms": 1.25},
        "timing_scope": "choice",
        "group_rows": 1,
        "timing_owner": True,
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        "diagnostics": {"batch_execution": {"path": "scheduler_native_compact_batch"}},
        "finish_details": _stateless_finish_details(
            "eos",
            eos_token_id=151645,
            sampler_mode="greedy_fast",
        ),
    }


def test_server_lowers_single_token_stop_strings_to_stop_token_ids() -> None:
    fake = FakeLLM(outputs=["alpha!tail"], token_map={"!": [99], "two tokens": [10, 11]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "one",
            "max_tokens": 4,
            "stop": ["!", "two tokens"],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "alpha"
    sampling = fake.calls[0][1]
    assert sampling.stop_token_ids == (99,)
    assert sampling.stop_token_sequences == ((10, 11),)
    assert fake.tokenize_calls == ["!", "two tokens"]


def test_chat_completion_uses_bounded_default_max_tokens() -> None:
    fake = FakeLLM(outputs=["assistant reply"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            max_context_tokens=100,
            chat_default_max_tokens=7,
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert fake.calls[0][1].max_tokens == 7


def test_chat_completion_auto_default_max_tokens_uses_remaining_context() -> None:
    fake = FakeLLM(outputs=["assistant reply"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            max_context_tokens=12,
            chat_default_max_tokens=None,
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[0][0][0]
    assert fake.calls[0][1].max_tokens == 12 - fake.count_tokens(prompt) - 1


def test_completions_endpoint_plumbs_sampling_parameters() -> None:
    fake = FakeLLM(outputs=["sampled"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "one",
            "max_tokens": 2,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.3,
            "logit_bias": {"12": -1.5},
            "seed": 123,
        },
    )

    assert response.status_code == 200
    sampling = fake.calls[0][1]
    assert sampling.temperature == 0.8
    assert sampling.top_p == 0.9
    assert sampling.top_k == 40
    assert sampling.min_p == 0.05
    assert sampling.repetition_penalty == 1.1
    assert sampling.presence_penalty == 0.2
    assert sampling.frequency_penalty == 0.3
    assert sampling.logit_bias == ((12, -1.5),)
    assert sampling.seed == 123


def test_chat_completion_renders_messages_to_prompt() -> None:
    fake = FakeLLM(outputs=["assistant reply"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
            "max_tokens": 4,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "fake-model"
    assert body["hipengine"]["routing"] == _routing_metadata()
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "assistant reply"},
            "finish_reason": "stop",
            "finish_details": _stateless_finish_details("stop"),
        }
    ]
    assert fake.calls[0][0] == (
        "<|im_start|>system\nbe concise<|im_end|>\n"
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n",
    )
    assert fake.calls[0][1].max_tokens == 4


def test_chat_completion_rejects_unsupported_message_role_before_generation() -> None:
    fake = FakeLLM(outputs=["should not generate"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "critic", "content": "hello"}],
            "max_tokens": 4,
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "messages[0].role"
    assert "assistant, developer, system, tool, user" in error["message"]
    assert fake.calls == []


def test_chat_completion_exposes_backend_generation_telemetry() -> None:
    fake = DetailedGenerateFakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="assistant reply",
                finish_details=FinishDetails(reason="eos", eos_token_id=151645, sampler_mode="processed_argmax"),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=4,
                    generated_tokens=2,
                    row_index=0,
                    phase="answer",
                    sampler_mode="processed_argmax",
                    active_processors=("min_tokens",),
                    sampler_fast_path_blockers=("min_tokens",),
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
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "assistant reply"}
    assert choice["hipengine"] == {
        "decode_state": {
            "row_index": 0,
            "step_index": 2,
            "prompt_tokens": 4,
            "generated_tokens": 2,
            "phase": "answer",
            "continuation_eligible": False,
            "active_processors": ["min_tokens"],
            "sampler_fast_path_blockers": ["min_tokens"],
            "sampler_mode": "processed_argmax",
        },
        "finish_details": _stateless_finish_details(
            "eos",
            eos_token_id=151645,
            sampler_mode="processed_argmax",
        ),
    }


def test_chat_completion_segregates_reasoning_content() -> None:
    fake = FakeLLM(outputs=["<think>scratch pad</think>assistant reply"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message == {
        "role": "assistant",
        "content": "assistant reply",
        "reasoning_content": "scratch pad",
    }


def test_chat_completion_accepts_qwen_no_think_controls() -> None:
    fake = FakeLLM(outputs=["direct answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "answer directly"}],
            "enable_thinking": False,
        },
    )

    assert response.status_code == 200
    assert fake.calls[0][0][0].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_chat_completion_accepts_reasoning_effort_controls() -> None:
    fake = FakeLLM(outputs=["reasoned answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    low = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think briefly"}],
            "reasoning_effort": "low",
        },
    )
    assert low.status_code == 200
    assert "keep it very brief" in fake.calls[-1][0][0]
    assert "close </think> before exceeding 512 hidden reasoning tokens" in fake.calls[-1][0][0]
    assert "reserve at least 512 tokens for the final answer or tool call" in fake.calls[-1][0][0]
    assert "begin closing during the final 128 hidden reasoning tokens" in fake.calls[-1][0][0]
    assert not fake.calls[-1][0][0].endswith("<think>\n\n</think>\n\n")

    none = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "do not think"}],
            "reasoning_effort": "none",
        },
    )
    assert none.status_code == 200
    assert fake.calls[-1][0][0].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_chat_completion_clamps_reasoning_effort_defaults_to_generation_budget() -> None:
    fake = FakeLLM(outputs=["reasoned answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "debug"}],
            "reasoning_effort": "medium",
            "max_tokens": 100,
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "keep it concise" in prompt
    assert "close </think> before exceeding 50 hidden reasoning tokens" in prompt
    assert "reserve at least 50 tokens for the final answer or tool call" in prompt
    assert "begin closing during the final 50 hidden reasoning tokens" in prompt


def test_chat_completion_clamps_reasoning_effort_defaults_to_remaining_context() -> None:
    fake = FakeLLM(outputs=["reasoned answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            max_context_tokens=120,
            chat_default_max_tokens=4096,
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "debug"}],
            "reasoning_effort": "medium",
        },
    )

    assert response.status_code == 200
    prompt, sampling = fake.calls[-1]
    admitted_budget = int(sampling.max_tokens)
    expected_reserved = admitted_budget // 2
    expected_think_cap = admitted_budget - expected_reserved
    assert admitted_budget < 4096
    assert f"close </think> before exceeding {expected_think_cap} hidden reasoning tokens" in prompt[0]
    assert f"reserve at least {expected_reserved} tokens for the final answer or tool call" in prompt[0]
    assert f"begin closing during the final {expected_think_cap} hidden reasoning tokens" in prompt[0]
    assert "4096 hidden reasoning tokens" not in prompt[0]
    assert "1024 tokens for the final answer" not in prompt[0]


def test_chat_completion_clamps_explicit_thinking_budget_hints() -> None:
    fake = FakeLLM(outputs=["bounded answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think within budget"}],
            "max_tokens": 100,
            "hard_think_cap": 90,
            "min_answer_tokens": 80,
            "soft_close_window": 200,
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "close </think> before exceeding 50 hidden reasoning tokens" in prompt
    assert "reserve at least 50 tokens for the final answer or tool call" in prompt
    assert "begin closing during the final 50 hidden reasoning tokens" in prompt


def test_chat_completion_accepts_thinking_budget_prompt_hints() -> None:
    fake = FakeLLM(outputs=["bounded answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think, then answer"}],
            "chat_template_kwargs": {"thinking_budget": 99, "reasoning_effort": "low"},
            "thinking_token_budget": 123,
            "max_think_tokens": 32,
            "min_answer_tokens": 8,
            "soft_close_window": 4,
            "hard_close_message": "closing now",
            "thinking": {
                "budget_tokens": 456,
                "hard_close_sequence": "closing now</think>\n",
            },
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "keep it very brief" in prompt
    assert "aim to close hidden reasoning within 32 tokens" in prompt
    assert "close </think> before exceeding 456 hidden reasoning tokens" in prompt
    assert "reserve at least 8 tokens for the final answer or tool call" in prompt
    assert "begin closing during the final 4 hidden reasoning tokens" in prompt
    assert "use the close message 'closing now' only if budget pressure requires it" in prompt
    assert "use 'closing now</think>\\n' as the close sequence if budget pressure requires it" in prompt
    assert "exceeding 99 hidden reasoning tokens" not in prompt
    assert "exceeding 123 hidden reasoning tokens" not in prompt


def test_chat_completion_lowers_thinking_budget_into_sampling_params() -> None:
    fake = FakeLLM(outputs=["bounded answer"], token_map={"</think>": [42, 43]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think, then answer"}],
            "max_tokens": 20,
            "hard_think_cap": 12,
            "soft_close_window": 3,
        },
    )

    assert response.status_code == 200
    assert fake.tokenize_calls == ["</think>"]
    sampling = fake.calls[-1][1]
    assert sampling.thinking_close_token_ids == (42, 43)
    assert sampling.thinking_hard_token_cap == 12
    assert sampling.thinking_soft_close_window == 3


def test_chat_completion_preserves_string_thinking_budget_effort_alias() -> None:
    fake = FakeLLM(outputs=["answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "chat_template_kwargs": {"thinking_budget": "high"},
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "keep it focused but complete" in prompt
    assert "close </think> before exceeding 2048 hidden reasoning tokens" in prompt
    assert "reserve at least 2048 tokens for the final answer or tool call" in prompt
    assert "begin closing during the final 1024 hidden reasoning tokens" in prompt


def test_chat_completion_allow_unbounded_skips_default_hard_thinking_cap() -> None:
    fake = FakeLLM(outputs=["answer"], token_map={"</think>": [42, 43]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "reasoning_effort": "low",
            "thinking": {"allow_unbounded": True},
            "max_tokens": 100,
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "keep it very brief" in prompt
    assert "reserve at least 50 tokens for the final answer or tool call" in prompt
    assert "close </think> before exceeding" not in prompt
    assert "begin closing during the final" not in prompt
    sampling = fake.calls[-1][1]
    # Effort cap is disabled, but the promised answer reserve remains binding.
    assert sampling.thinking_hard_token_cap == 48
    assert sampling.thinking_close_token_ids == (42, 43)
    assert fake.tokenize_calls == ["</think>"]


def test_chat_completion_allow_unbounded_preserves_explicit_hard_cap() -> None:
    fake = FakeLLM(outputs=["answer"], token_map={"</think>": [42, 43]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "reasoning_effort": "low",
            "thinking": {"allow_unbounded": True, "hard_think_cap": 12},
            "max_tokens": 100,
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "close </think> before exceeding 12 hidden reasoning tokens" in prompt
    assert "begin closing during the final 12 hidden reasoning tokens" in prompt
    sampling = fake.calls[-1][1]
    assert sampling.thinking_hard_token_cap == 12
    assert sampling.thinking_close_token_ids == (42, 43)
    assert fake.tokenize_calls == ["</think>"]


def test_chat_completion_reasoning_object_budget_overrides_top_level_controls() -> None:
    fake = FakeLLM(outputs=["answer"], token_map={"closing</think>\n": [42, 43, 44]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "max_tokens": 100,
            "chat_template_kwargs": {"thinking_budget": 99},
            "thinking_token_budget": 123,
            "hard_think_cap": 80,
            "min_answer_tokens": 70,
            "soft_close_window": 60,
            "thinking": {"budget_tokens": 50, "min_answer_tokens": 30},
            "reasoning": {
                "effort": "high",
                "max_tokens": 12,
                "min_answer_tokens": 4,
                "soft_close_window": 3,
                "hard_close_sequence": "closing</think>\n",
            },
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "keep it focused but complete" in prompt
    assert "close </think> before exceeding 12 hidden reasoning tokens" in prompt
    assert "reserve at least 4 tokens for the final answer or tool call" in prompt
    assert "begin closing during the final 3 hidden reasoning tokens" in prompt
    assert "use 'closing</think>\\n' as the close sequence if budget pressure requires it" in prompt
    assert "exceeding 99 hidden reasoning tokens" not in prompt
    assert "exceeding 123 hidden reasoning tokens" not in prompt
    assert "exceeding 80 hidden reasoning tokens" not in prompt
    assert "exceeding 50 hidden reasoning tokens" not in prompt
    sampling = fake.calls[-1][1]
    assert sampling.thinking_hard_token_cap == 12
    assert sampling.thinking_soft_close_window == 3
    assert sampling.thinking_close_token_ids == (42, 43, 44)
    assert fake.tokenize_calls == ["closing</think>\n"]


def test_chat_completion_reasoning_type_disabled_wins_over_enabled_true() -> None:
    fake = FakeLLM(outputs=["answer"], token_map={"</think>": [42, 43]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "reasoning_effort": "low",
            "reasoning": {"type": "none", "enabled": True, "hard_think_cap": 12},
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n" in prompt
    assert "Do not include hidden reasoning" in prompt
    assert "close </think> before exceeding" not in prompt
    sampling = fake.calls[-1][1]
    assert sampling.thinking_hard_token_cap is None
    assert sampling.thinking_close_token_ids == ()
    assert fake.tokenize_calls == []


def test_chat_completion_rejects_hard_close_without_think_marker() -> None:
    fake = FakeLLM(outputs=["unused"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "hard_close_sequence": "DONE\n",
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "hard_close_sequence"
    assert fake.calls == []


def test_chat_completion_rejects_invalid_thinking_budget_value() -> None:
    fake = FakeLLM(outputs=["unused"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "thinking": {"max_tokens": "soon"},
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "thinking.max_tokens"
    assert fake.calls == []


@pytest.mark.parametrize(
    ("output", "phase", "content", "reasoning_content"),
    [
        ("<think>scratch", "reasoning", "", "scratch"),
        ("<think>scratch</thi", "closing_think", "", "scratch</thi"),
        ('<tool_call>{"name":"read"', "tool_call", '<tool_call>{"name":"read"', None),
        ('{"status":', "structured", '{"status":', None),
        ("partial answer", "answer", "partial answer", None),
    ],
)
def test_chat_completion_length_finish_details_include_phase(
    output: str,
    phase: str,
    content: str,
    reasoning_content: str | None,
) -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text=output,
                finish_details=FinishDetails(reason="length", length_limit=5),
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
    assert choice["finish_reason"] == "length"
    if phase in {"answer", "structured"}:
        continuation_id = choice["continuation_id"]
        assert continuation_id.startswith("gen_")
        assert choice["finish_details"] == _stateless_finish_details(
            "length",
            length_limit=5,
            phase=phase,
            continuation_eligible=True,
            continuation_id=continuation_id,
        )
    else:
        assert "continuation_id" not in choice
        assert choice["finish_details"] == _stateless_finish_details(
            "length",
            length_limit=5,
            phase=phase,
            continuation_eligible=False,
        )
    assert choice["message"]["content"] == content
    if reasoning_content is None:
        assert "reasoning_content" not in choice["message"]
    else:
        assert choice["message"]["reasoning_content"] == reasoning_content


def test_chat_completion_thinking_budget_exhausted_maps_to_length_finish() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="<think>closed</think>",
                finish_details=FinishDetails(
                    reason="thinking_budget_exhausted",
                    length_limit=1,
                    forced_close=True,
                    reasoning_tokens=1,
                    budget_pressure="hard_close",
                    phase="answer",
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
            "messages": [{"role": "user", "content": "think briefly"}],
            "reasoning_effort": "low",
            "max_tokens": 1,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "length"
    assert "continuation_id" not in choice
    assert choice["finish_details"] == _stateless_finish_details(
        "thinking_budget_exhausted",
        length_limit=1,
        forced_close=True,
        reasoning_tokens=1,
        budget_pressure="hard_close",
        phase="answer",
        continuation_eligible=False,
    )


def test_chat_completion_preserves_incomplete_duplicate_prefix_with_ordinary_content() -> None:
    valid_call = (
        '<tool_call>{"name":"read","arguments":{"path":"pyproject.toml",'
        '"mode":"summary"}}</tool_call>'
    )
    raw_output = "Keep this literal prefix: " + _incomplete_duplicate_read_tool_output()
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
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == expected_content
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "read"


def test_chat_completion_rejects_incomplete_duplicate_prefix_without_valid_call() -> None:
    raw_output = (
        '<tool_call>{"name":"read","arguments":{<|im_end|>\n\n'
        '<|im_start|>assistant\n<|im_end|><|endoftext|>'
    )
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
                }
            ],
        },
    )

    assert response.status_code == 200
    assert raw_output not in response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_terminal_cleanup_preserves_interior_literal_content() -> None:
    fake = FakeLLM(
        outputs=[
            '<|im_end|>Keep <|im_end|> and <|im_start|>user literal. '
            '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'
            '<|im_end|>'
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
    assert choice["message"]["content"] == "Keep <|im_end|> and <|im_start|>user literal."
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "read"


def test_completions_endpoint_lowers_n_to_distinct_seeded_rows() -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": ["one", "two"], "max_tokens": 1, "n": 2, "seed": 123},
    )

    assert response.status_code == 200
    body = response.json()
    assert [choice["text"] for choice in body["choices"]] == [
        "generated:one",
        "generated:one",
        "generated:two",
        "generated:two",
    ]
    assert [choice["index"] for choice in body["choices"]] == [0, 1, 2, 3]
    assert len({choice["request_id"] for choice in body["choices"]}) == 4
    assert fake.calls[0][0] == ("one", "one", "two", "two")
    assert len(fake.calls[0][1].row_seeds) == 4
    assert len(set(fake.calls[0][1].row_seeds)) == 4


def test_chat_endpoint_lowers_n_to_distinct_seeded_rows() -> None:
    fake = FakeLLM(outputs=["first", "second"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hello"}], "n": 2, "seed": 9},
    )

    assert response.status_code == 200
    body = response.json()
    assert [choice["index"] for choice in body["choices"]] == [0, 1]
    assert [choice["message"]["content"] for choice in body["choices"]] == ["first", "second"]
    assert len({choice["request_id"] for choice in body["choices"]}) == 2
    assert fake.calls[0][0] == (fake.calls[0][0][0], fake.calls[0][0][0])
    assert len(fake.calls[0][1].row_seeds) == 2
    assert len(set(fake.calls[0][1].row_seeds)) == 2


def test_chat_endpoint_rejects_non_text_content_parts() -> None:
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=FakeLLM())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unsupported_content_type"
    assert error["hipengine"] == {
        "code": "unsupported_parameter",
        "status_code": 400,
        "legacy_code": "unsupported_content_type",
        "retryable": False,
    }


def test_chat_request_accepts_max_completion_tokens() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 7,
            "temperature": 0,
        }
    )
    assert request.max_completion_tokens == 7
    extra = getattr(request, "model_extra", None) or {}
    assert "max_completion_tokens" not in extra
    assert _request_completion_cap(request) == 7
    # max_tokens takes precedence when both are supplied.
    both = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "max_completion_tokens": 7,
        }
    )
    assert _request_completion_cap(both) == 4


def test_completion_request_accepts_max_completion_tokens() -> None:
    request = CompletionRequest.model_validate(
        {"model": "fake-model", "prompt": "hi", "max_tokens": None, "max_completion_tokens": 5}
    )
    assert request.max_completion_tokens == 5
    assert _request_completion_cap(request) == 5


def test_strip_chat_terminal_markers_removes_eos() -> None:
    from hipengine.server.api import _strip_chat_terminal_markers

    assert _strip_chat_terminal_markers("OK<|im_end|>") == "OK"
    assert _strip_chat_terminal_markers("\n\nOK<|im_end|>") == "\n\nOK"
    assert _strip_chat_terminal_markers("READY<|im_end|>") == "READY"
    assert _strip_chat_terminal_markers("plain text") == "plain text"


def test_usage_exposes_reasoning_tokens() -> None:
    from hipengine.server.api import _finish_reasoning_token_total, _usage

    details = [
        GenerationOutput(
            text="answer",
            generated_token_ids=tuple(range(12)),
            finish_details=FinishDetails(reason="eos", reasoning_tokens=9, answer_tokens=3),
        )
    ]
    usage = _usage(FakeLLM(), ("prompt",), ("answer",), details=details)
    assert usage["completion_tokens"] == 12
    assert usage["reasoning_tokens"] == 9
    assert usage["completion_tokens_details"] == {"reasoning_tokens": 9}
    assert _finish_reasoning_token_total(details) == 9
