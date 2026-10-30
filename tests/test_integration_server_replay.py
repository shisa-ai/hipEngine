"""OpenAI server integration tests: replay.

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
    BackendCancelledFakeLLM,
    BackendDeadlineFakeLLM,
    FakeLLM,
)
from tests.support.server_api_support import (
    _continuation_capability,
    _load_single_replay_artifact,
    _parallelism_capability,
    _session_commit_policy_capability,
    _session_metadata_capability,
    _sse_payloads,
    _stateless_finish_details,
)


def test_agentic_replay_failure_reasons_match_capability_contract() -> None:
    advertised = frozenset(
        (
            *_TOOL_RESULT_VALIDATION_FAILURE_REASONS,
            *_STRUCTURED_OUTPUT_RESULT_VALIDATION_FAILURE_REASONS,
        )
    )
    assert _AGENTIC_REPLAY_FAILURE_REASONS == advertised


def test_replay_artifacts_are_default_off(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=FakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "secret prompt", "typical_p": 0.9},
    )

    assert response.status_code == 400
    assert not replay_dir.exists()


@pytest.mark.parametrize(
    ("payload", "output", "expected_finish_details"),
    [
        (
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "secret no replay tool task"}],
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
                "max_tokens": 16,
            },
            "ordinary answer",
            _stateless_finish_details("tool_required_not_satisfied"),
        ),
        (
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "secret no replay structured task"}],
                "response_format": {"type": "json_object"},
                "max_tokens": 16,
            },
            "not json",
            _stateless_finish_details("schema_violation"),
        ),
    ],
)
def test_replay_artifacts_are_default_off_for_agentic_result_failures(
    tmp_path,
    payload: dict[str, Any],
    output: str,
    expected_finish_details: dict[str, Any],
) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=FakeLLM(outputs=[output]),
    )
    client = TestClient(app)

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_details"] == expected_finish_details
    assert choice["message"] == {"role": "assistant", "content": ""}
    assert not replay_dir.exists()


def test_replay_artifact_captures_streaming_completion_deadline_error(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=BackendDeadlineFakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "secret streaming deadline prompt",
            "max_tokens": 4,
            "stream": True,
            "timeout_ms": 5000,
        },
    )

    assert response.status_code == 200
    payload = next(item for item in _sse_payloads(response.text) if item.get("error"))
    assert payload["error"]["code"] == "deadline_exceeded"
    assert payload["choices"][0]["finish_details"] == {
        "reason": "deadline_exceeded",
        "deadline_exceeded": True,
    }

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["schema"] == "hipengine.replay.v1"
    assert artifact["request"]["path"] == "/v1/completions"
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": "$.prompt",
            "sha256": artifact["request"]["json"]["prompt"]["sha256"],
            "length": len("secret streaming deadline prompt"),
        }
    ]
    assert artifact["error"]["type"] == "timeout_error"
    assert artifact["error"]["code"] == "deadline_exceeded"
    assert artifact["error"]["param"] == "timeout_ms"
    assert artifact["error"]["hipengine"] == {
        "code": "deadline_exceeded",
        "status_code": 408,
        "retryable": True,
    }
    assert artifact["finish_details"] == {"reason": "deadline_exceeded", "deadline_exceeded": True}
    assert "secret streaming deadline prompt" not in serialized


def test_replay_artifact_captures_streaming_chat_cancelled_error(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=BackendCancelledFakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "secret streaming cancel task"}],
            "max_tokens": 4,
            "stream": True,
        },
    )

    assert response.status_code == 200
    payload = next(item for item in _sse_payloads(response.text) if item.get("error"))
    assert payload["error"]["code"] == "cancelled"
    assert payload["choices"][0]["finish_details"] == {
        "reason": "cancelled",
        "cancelled": True,
    }

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["schema"] == "hipengine.replay.v1"
    assert artifact["request"]["path"] == "/v1/chat/completions"
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": "$.messages[0].content",
            "sha256": artifact["request"]["json"]["messages"][0]["content"]["sha256"],
            "length": len("secret streaming cancel task"),
        }
    ]
    assert artifact["error"]["type"] == "cancelled_error"
    assert artifact["error"]["code"] == "cancelled"
    assert artifact["error"]["param"] is None
    assert artifact["error"]["hipengine"] == {
        "code": "cancelled",
        "status_code": 499,
        "retryable": True,
    }
    assert artifact["finish_details"] == {"reason": "cancelled", "cancelled": True}
    assert "secret streaming cancel task" not in serialized


def test_replay_artifact_redacts_failed_request(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "secret prompt",
            "max_tokens": 1,
            "top_k": 8,
            "logit_bias": {"12": -2.0},
            "top_logprobs": 2,
            "response_format": {"type": "json_object"},
            "seed": 123,
            "typical_p": 0.9,
        },
    )

    assert response.status_code == 400
    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["schema"] == "hipengine.replay.v1"
    assert artifact["redaction"] == {"mode": "hash", "hash": "sha256"}
    assert artifact["request"]["method"] == "POST"
    assert artifact["request"]["path"] == "/v1/completions"
    assert artifact["request"]["json"]["prompt"]["redacted"] == "sha256"
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": "$.prompt",
            "sha256": artifact["request"]["json"]["prompt"]["sha256"],
            "length": len("secret prompt"),
        }
    ]
    assert artifact["model"]["id"] == "fake-model"
    assert artifact["sampling"]["max_tokens"] == 1
    assert artifact["sampling"]["top_k"] == 8
    assert artifact["sampling"]["logit_bias"] == {"12": -2.0}
    assert artifact["sampling"]["top_logprobs"] == 2
    assert artifact["sampling"]["response_format"]["type"]["redacted"] == "sha256"
    assert artifact["sampling"]["response_format"]["type"]["length"] == len("json_object")
    assert artifact["seeds"] == {"seed": 123, "row_seeds": []}
    assert artifact["token_counts"] == {
        "prompt_tokens": 2,
        "completion_tokens": None,
        "total_tokens": None,
        "available": True,
        "source": "completion_prompt",
        "entries": [{"path": "$.prompt", "token_count": 2}],
    }
    assert artifact["error"]["code"] == "unsupported_parameter"
    assert artifact["error"]["param"] == "top_logprobs"
    assert artifact["error"]["hipengine"]["code"] == "unsupported_parameter"
    assert artifact["capabilities"]["model"]["id"] == "fake-model"
    assert artifact["capabilities"]["features"]["structured_outputs"][
        "result_validation_failure_reasons"
    ] == ["schema_violation"]
    assert artifact["capabilities"]["features"]["tools"]["result_validation_failure_reasons"] == [
        "invalid_tool_call",
        "tool_required_not_satisfied",
        "schema_violation",
    ]
    assert artifact["capabilities"]["features"]["tools"]["no_tool_start_suppression"] is True
    assert artifact["capabilities"]["features"]["tools"]["required_tool_start_forcing"] is True
    assert artifact["capabilities"]["features"]["tools"]["required_tool_start_forcing_scope"] == (
        "initial_or_after_tokenized_thinking_close"
    )
    assert artifact["capabilities"]["features"]["tools"]["specific_tool_name_prefix_forcing"] is True
    assert artifact["capabilities"]["features"]["tools"]["specific_tool_name_prefix_forcing_scope"] == (
        "atomic_xml_function_header"
    )
    assert artifact["capabilities"]["features"]["tools"]["strict_tool_schema_prefix_anchor"] is False
    assert artifact["capabilities"]["features"]["tools"]["strict_tool_schema_prefix_anchor_scope"] == "none"
    assert artifact["capabilities"]["features"]["tools"]["tool_call_close_repair"] is True
    assert artifact["capabilities"]["features"]["tools"]["tool_call_close_repair_scope"] == (
        "xml_parameter_boundary_suffix_exact_remaining_budget"
    )
    assert artifact["capabilities"]["features"]["tools"]["compatibility_parser_repairs"] == [
        "qwen35_family_xml_function_envelope",
        "legacy_json_tool_call_body",
        "duplicated_tool_call_start",
        "incomplete_duplicate_tool_prefix_control_residue",
        "outer_qwen_template_control_residue",
    ]
    assert (
        artifact["capabilities"]["features"]["tools"]["malformed_json_compatibility"]
        == "invalid_tool_call_when_tools_enabled"
    )
    assert artifact["capabilities"]["features"]["tools"]["strict_malformed_blocks_rejected"] is True
    assert artifact["capabilities"]["features"]["tools"]["declared_tool_name_validation"] is True
    assert artifact["capabilities"]["features"]["tools"]["transcript_validation"] == {
        "role_specific_fields": True,
        "assistant_tool_call_ids_unique": True,
        "tool_results_reference_prior_call_ids": True,
        "tool_results_must_resolve_pending_calls_before_non_tool_messages": True,
        "allows_pending_tool_calls_at_transcript_end": True,
        "applies_to_session_snapshots": True,
    }
    assert artifact["capabilities"]["features"]["tools"]["parallel_tool_calls_requires_opt_in"] is False
    assert artifact["capabilities"]["features"]["tools"]["streaming_argument_chunks"] is True
    assert artifact["capabilities"]["features"]["tools"]["streaming_argument_chunk_chars"] == 128
    assert artifact["capabilities"]["features"]["reasoning_controls"]["token_budget_enforced"] is True
    assert artifact["capabilities"]["features"]["reasoning_controls"]["hard_close_token_forcing"] is True
    assert artifact["capabilities"]["features"]["reasoning_controls"]["soft_close_bias"] is True
    assert artifact["capabilities"]["features"]["reasoning_controls"]["eos_suppression"] is True
    assert artifact["capabilities"]["sampling"]["speculative_mtp"] == {
        "serving_route": False,
        "configured_policy": "auto",
        "engine_supported": False,
        "model_has_mtp_tensors": False,
        "certified_default_scopes": [],
        "automatic_route_promoted": False,
        "sampling_compatible": False,
        "compatibility_guard": "supports_speculative_mtp_sampling",
        "allowed_execution_modes": ["greedy_fast"],
        "incompatible_fields": [
            "temperature",
            "logit_bias",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "suppress_token_ids",
            "min_tokens",
            "eos_token_id",
            "ignore_eos",
            "stop_token_ids",
            "stop_token_sequences",
            "forced_tokens_pending",
            "post_thinking_forced_tokens_pending",
            "force_sequence_completion_token_sequences",
            "json_object_close_forcing",
            "tool_call_constraint",
            "thinking_budget",
            "logprobs",
            "top_logprobs",
        ],
        "incompatible_conditions": {
            "temperature": "temperature > 0",
            "logit_bias": "non-empty logit_bias",
            "repetition_penalty": "repetition_penalty != 1.0",
            "presence_penalty": "presence_penalty != 0.0",
            "frequency_penalty": "frequency_penalty != 0.0",
            "suppress_token_ids": "one or more suppressed token ids",
            "min_tokens": "min_tokens > 0",
            "eos_token_id": "eos_token_id set",
            "ignore_eos": "ignore_eos=true",
            "stop_token_ids": "one or more token stop ids",
            "stop_token_sequences": "one or more multi-token stop sequences",
            "forced_tokens_pending": "one or more forced tokens pending",
            "post_thinking_forced_tokens_pending": "one or more post-thinking forced tokens pending",
            "force_sequence_completion_token_sequences": "one or more token sequence completion repairs",
            "json_object_close_forcing": "JSON object close forcing active",
            "tool_call_constraint": "tokenizer-aware tool-call grammar active",
            "thinking_budget": "thinking budget soft-close, EOS suppression, or hard-close control",
            "logprobs": "logprobs requested",
            "top_logprobs": "top_logprobs > 0",
        },
        "thinking_policy": "hint",
        "processed_target_verification": False,
        "candidate_budget": {
            "requested": None,
            "resolved": None,
            "source": "unresolved",
        },
        "context_window": {
            "policy": "target_capacity",
            "fixed_cap": None,
            "env": None,
            "exported": None,
            "resolved": None,
            "qualified_default": None,
            "error": None,
        },
    }
    assert artifact["capabilities"]["sampling"]["speculative_mtp"]["incompatible_fields"] == list(
        SPECULATIVE_MTP_INCOMPATIBLE_FIELDS
    )
    assert artifact["capabilities"]["sampling"]["speculative_mtp"]["incompatible_conditions"] == dict(
        SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS
    )
    assert artifact["capabilities"]["sessions"] == {
        "resident_context": True,
        "commit_policy": _session_commit_policy_capability(),
        "continuations": _continuation_capability(),
        "metadata": _session_metadata_capability(),
    }
    assert artifact["capabilities"]["parallelism"] == _parallelism_capability()
    assert "secret prompt" not in serialized


def test_replay_artifact_counts_chat_prompt_when_engine_loaded(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "secret chat prompt"}],
            "chat_template_kwargs": {"thinking_budget": "low"},
            "thinking": {"budget_tokens": 32},
            "reasoning": {
                "allow_unbounded": True,
                "max_tokens": 12,
                "min_answer_tokens": 4,
                "hard_close_sequence": "closing</think>\n",
            },
            "hard_close_message": "closing",
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "top_logprobs": 1,
        },
    )

    assert response.status_code == 400
    artifact, serialized = _load_single_replay_artifact(replay_dir)

    assert artifact["request"]["path"] == "/v1/chat/completions"
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": "$.messages[0].content",
            "sha256": artifact["request"]["json"]["messages"][0]["content"]["sha256"],
            "length": len("secret chat prompt"),
        }
    ]
    assert artifact["token_counts"]["available"] is True
    assert artifact["token_counts"]["source"] == "chat_prompt"
    assert artifact["token_counts"]["entries"][0]["path"] == "$.messages"
    assert artifact["token_counts"]["entries"][0]["token_count"] == artifact["token_counts"]["prompt_tokens"]
    assert artifact["token_counts"]["prompt_tokens"] > 0
    assert artifact["sampling"]["chat_template_kwargs"]["thinking_budget"]["redacted"] == "sha256"
    assert artifact["sampling"]["chat_template_kwargs"]["thinking_budget"]["length"] == len("low")
    assert artifact["sampling"]["thinking"] == {"budget_tokens": 32}
    assert artifact["sampling"]["reasoning"]["allow_unbounded"] is True
    assert artifact["sampling"]["reasoning"]["max_tokens"] == 12
    assert artifact["sampling"]["reasoning"]["min_answer_tokens"] == 4
    assert artifact["sampling"]["reasoning"]["hard_close_sequence"]["redacted"] == "sha256"
    assert artifact["sampling"]["reasoning"]["hard_close_sequence"]["length"] == len("closing</think>\n")
    assert artifact["sampling"]["hard_close_message"]["redacted"] == "sha256"
    assert artifact["sampling"]["hard_close_message"]["length"] == len("closing")
    assert artifact["sampling"]["tool_choice"]["redacted"] == "sha256"
    assert artifact["sampling"]["tool_choice"]["length"] == len("auto")
    assert artifact["sampling"]["parallel_tool_calls"] is False
    assert artifact["error"]["param"] == "top_logprobs"
    assert "secret chat prompt" not in serialized
    assert "closing</think>" not in serialized


def test_replay_artifact_captures_completion_structured_result_validation_failure(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(outputs=["not json"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "secret structured prompt",
            "response_format": {"type": "json_object"},
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert choice["text"] == ""

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["schema"] == "hipengine.replay.v1"
    assert artifact["request"]["path"] == "/v1/completions"
    assert artifact["request"]["json"]["prompt"]["redacted"] == "sha256"
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": "$.prompt",
            "sha256": artifact["request"]["json"]["prompt"]["sha256"],
            "length": len("secret structured prompt"),
        }
    ]
    assert artifact["sampling"]["response_format"]["type"]["redacted"] == "sha256"
    assert artifact["sampling"]["response_format"]["type"]["length"] == len("json_object")
    assert artifact["finish_details"] == _stateless_finish_details("schema_violation")
    assert artifact["error"] is None
    assert artifact["result"] == {
        "type": "agentic_result_validation",
        "finish_details": _stateless_finish_details("schema_violation"),
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "finish_details": _stateless_finish_details("schema_violation"),
            }
        ],
    }
    assert "secret structured prompt" not in serialized
    assert "not json" not in serialized


def test_replay_artifact_captures_agentic_result_validation_failure(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    prior_tool_arguments = '{"path":"secret previous path"}'
    prior_tool_result = "secret previous tool result"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(outputs=["ordinary answer"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [
                {"role": "user", "content": "secret tool task"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_secret_prev",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": prior_tool_arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_secret_prev",
                    "content": prior_tool_result,
                },
            ],
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
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("tool_required_not_satisfied")
    assert choice["message"] == {"role": "assistant", "content": ""}

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["schema"] == "hipengine.replay.v1"
    assert artifact["request"]["path"] == "/v1/chat/completions"
    assert artifact["request"]["json"]["messages"][0]["content"]["redacted"] == "sha256"
    assert (
        artifact["request"]["json"]["messages"][1]["tool_calls"][0]["function"]["arguments"][
            "redacted"
        ]
        == "sha256"
    )
    assert artifact["request"]["json"]["messages"][2]["content"]["redacted"] == "sha256"
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": "$.messages[0].content",
            "sha256": artifact["request"]["json"]["messages"][0]["content"]["sha256"],
            "length": len("secret tool task"),
        },
        {
            "path": "$.messages[1].tool_calls[0].function.arguments",
            "sha256": artifact["request"]["json"]["messages"][1]["tool_calls"][0]["function"][
                "arguments"
            ]["sha256"],
            "length": len(prior_tool_arguments),
        },
        {
            "path": "$.messages[2].content",
            "sha256": artifact["request"]["json"]["messages"][2]["content"]["sha256"],
            "length": len(prior_tool_result),
        },
    ]
    assert artifact["finish_details"] == _stateless_finish_details("tool_required_not_satisfied")
    assert artifact["error"] is None
    assert artifact["result"] == {
        "type": "agentic_result_validation",
        "finish_details": _stateless_finish_details("tool_required_not_satisfied"),
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "finish_details": _stateless_finish_details("tool_required_not_satisfied"),
            }
        ],
    }
    assert artifact["token_counts"]["available"] is True
    assert artifact["token_counts"]["source"] == "chat_prompt"
    assert "secret tool task" not in serialized
    assert "secret previous path" not in serialized
    assert prior_tool_arguments not in serialized
    assert prior_tool_result not in serialized
    assert "ordinary answer" not in serialized


def test_replay_artifact_captures_streaming_structured_result_validation_failure(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(stream_chunks=["not json"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "secret streaming structured task"}],
            "response_format": {"type": "json_object"},
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_reason"] == "stop"
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"]["finish_details"] == _stateless_finish_details("schema_violation")

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["request"]["path"] == "/v1/chat/completions"
    assert artifact["sampling"]["response_format"]["type"]["redacted"] == "sha256"
    assert artifact["sampling"]["response_format"]["type"]["length"] == len("json_object")
    assert artifact["finish_details"] == _stateless_finish_details("schema_violation")
    assert artifact["error"] is None
    assert artifact["result"]["choices"] == [
        {
            "index": 0,
            "finish_reason": "stop",
            "finish_details": _stateless_finish_details("schema_violation"),
        }
    ]
    assert "secret streaming structured task" not in serialized
    assert "not json" not in serialized


def test_replay_artifact_captures_streaming_agentic_result_validation_failure(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(stream_chunks=["ordinary stream answer"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "secret streaming tool task"}],
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
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_reason"] == "error"
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("tool_required_not_satisfied")
    assert done["choices"][0]["hipengine"]["finish_details"] == _stateless_finish_details(
        "tool_required_not_satisfied"
    )

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["request"]["path"] == "/v1/chat/completions"
    assert artifact["finish_details"] == _stateless_finish_details("tool_required_not_satisfied")
    assert artifact["error"]["code"] == "tool_required_not_satisfied"
    assert "result" not in artifact
    assert "secret streaming tool task" not in serialized
    assert "ordinary stream answer" not in serialized
