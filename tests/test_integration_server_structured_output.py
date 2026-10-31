"""OpenAI server integration tests: structured output.

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
    _load_single_replay_artifact,
    _response_json_schema,
    _sse_payloads,
    _stateless_finish_details,
    _unified_diff_text,
)


def test_completion_continuation_resume_rejects_explicit_response_format_override() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text='{"ok":',
                finish_details=FinishDetails(reason="length", length_limit=6),
            ),
            GenerationOutput(text="true}", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "Return JSON: ",
            "response_format": {"type": "json_object"},
            "max_tokens": 6,
        },
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    override = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "response_format": {"type": "text"},
            "max_tokens": 4,
        },
    )

    assert override.status_code == 400
    assert override.json()["error"]["code"] == "unsupported_parameter"
    assert override.json()["error"]["param"] == "response_format"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    inherited = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert inherited.status_code == 200
    assert inherited.json()["choices"][0]["text"] == '{"ok":true}'
    assert len(fake.calls) == 2


def test_completions_response_format_json_object_validates_result() -> None:
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=["not json"]),
        )
    )

    valid = valid_client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {"type": "json_object"},
        },
    )
    invalid = invalid_client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {"type": "json_object"},
        },
    )

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"ok":true}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


@pytest.mark.parametrize(
    ("payload", "output", "expected"),
    [
        ({"response_format": {"type": "json_object"}}, '{"ok":true}', True),
        (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "object_result",
                        "schema": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
            '{"ok":true}',
            True,
        ),
        (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "array_result",
                        "schema": {"type": "array", "items": {"type": "integer"}},
                    },
                },
            },
            "[1]",
            False,
        ),
        ({"guided_json": True}, '{"ok":true}', True),
        (
            {
                "guided_json": {
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                },
            },
            '{"ok":true}',
            True,
        ),
    ],
)
def test_completions_structured_json_lowers_close_forcing(
    payload: dict[str, Any],
    output: str,
    expected: bool,
) -> None:
    fake = FakeLLM(outputs=[output])
    client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=fake,
        )
    )

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            **payload,
        },
    )

    assert response.status_code == 200
    assert fake.calls[0][1].json_object_close_forcing is expected
    assert fake.calls[0][1].max_tokens == 16


def test_completions_response_format_length_rejects_invalid_json_continuation() -> None:
    client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(
                detailed_outputs=[
                    GenerationOutput(
                        text='{"ok": [1}',
                        finish_details=FinishDetails(reason="length", length_limit=9),
                    )
                ]
            ),
        )
    )

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {"type": "json_object"},
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == '{"ok": [1}'
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_completions_response_format_json_schema_validates_result() -> None:
    schema = _response_json_schema()
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"path":"README.md"}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":"yes","path":"README.md"}']),
        )
    )

    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"ok":true,"path":"README.md"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_string_pattern() -> None:
    schema = _response_json_schema()
    schema["schema"]["properties"]["path"]["pattern"] = r"^README[.]md$"
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"path":"README.md"}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"path":"WORKLOG.md"}']),
        )
    )

    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"ok":true,"path":"README.md"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_object_property_count() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "minProperties": 2,
            "maxProperties": 2,
            "properties": {
                "ok": {"type": "boolean"},
                "path": {"type": "string"},
                "extra": {"type": "string"},
            },
        },
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"path":"README.md"}']),
        )
    )
    too_few_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true}']),
        )
    )
    too_many_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"path":"README.md","extra":"x"}']),
        )
    )

    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid = valid_client.post("/v1/completions", json=payload)
    too_few = too_few_client.post("/v1/completions", json=payload)
    too_many = too_many_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"ok":true,"path":"README.md"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for response in (too_few, too_many):
        assert response.status_code == 200
        choice = response.json()["choices"][0]
        assert choice["text"] == ""
        assert choice["finish_reason"] == "stop"
        assert choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_additional_properties_schema() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": {"type": "string"},
        },
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"note":"done"}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"note":7}']),
        )
    )
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }

    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"ok":true,"note":"done"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_pattern_properties() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "patternProperties": {r"^x-[a-z]+$": {"type": "integer"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"x-count":2}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"x-count":"two"}']),
        )
    )
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }

    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"ok":true,"x-count":2}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_object_name_keywords() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "propertyNames": {"pattern": r"^(id|name|x-[a-z]+)$"},
            "dependentRequired": {"id": ["name"]},
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
            },
        },
    }
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"id":"1","name":"doc","x-extra":"ok"}']),
        )
    )
    invalid_outputs = [
        '{"id":"1","name":"doc","bad-key":"no"}',
        '{"id":"1","x-extra":"ok"}',
    ]

    valid = valid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"id":"1","name":"doc","x-extra":"ok"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for generated in invalid_outputs:
        invalid = TestClient(
            create_app(
                ServerConfig(model="fake-path", served_model_name="fake-model"),
                llm=FakeLLM(outputs=[generated]),
            )
        ).post("/v1/completions", json=payload)
        assert invalid.status_code == 200
        invalid_choice = invalid.json()["choices"][0]
        assert invalid_choice["text"] == ""
        assert invalid_choice["finish_reason"] == "stop"
        assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_dependent_schemas() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
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
    }
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"kind":"file","path":"README.md"}']),
        )
    )
    invalid_outputs = [
        '{"kind":"url","path":"README.md"}',
        '{"href":"https://example.test"}',
    ]

    valid = valid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"kind":"file","path":"README.md"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for generated in invalid_outputs:
        invalid = TestClient(
            create_app(
                ServerConfig(model="fake-path", served_model_name="fake-model"),
                llm=FakeLLM(outputs=[generated]),
            )
        ).post("/v1/completions", json=payload)
        assert invalid.status_code == 200
        invalid_choice = invalid.json()["choices"][0]
        assert invalid_choice["text"] == ""
        assert invalid_choice["finish_reason"] == "stop"
        assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_conditional_schemas() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
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
    }
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"kind":"file","path":"README.md"}']),
        )
    )
    invalid_outputs = [
        '{"kind":"file","href":"https://example.test"}',
        '{"kind":"url","path":"README.md"}',
    ]

    valid = valid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"kind":"file","path":"README.md"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for generated in invalid_outputs:
        invalid = TestClient(
            create_app(
                ServerConfig(model="fake-path", served_model_name="fake-model"),
                llm=FakeLLM(outputs=[generated]),
            )
        ).post("/v1/completions", json=payload)
        assert invalid.status_code == 200
        invalid_choice = invalid.json()["choices"][0]
        assert invalid_choice["text"] == ""
        assert invalid_choice["finish_reason"] == "stop"
        assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_local_refs() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "$defs": {
                "file_ref": {"type": "string", "pattern": r"^[A-Z]+[.]md$"},
                "result_tag": {"enum": ["ok", "skip"]},
            },
            "properties": {
                "path": {"$ref": "#/$defs/file_ref"},
                "result": {"$ref": "#/$defs/result_tag"},
            },
            "required": ["path", "result"],
            "additionalProperties": False,
        },
    }
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"path":"README.md","result":"ok"}']),
        )
    )
    invalid_outputs = [
        '{"path":"readme.md","result":"ok"}',
        '{"path":"README.md","result":"done"}',
    ]

    valid = valid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"path":"README.md","result":"ok"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for generated in invalid_outputs:
        invalid = TestClient(
            create_app(
                ServerConfig(model="fake-path", served_model_name="fake-model"),
                llm=FakeLLM(outputs=[generated]),
            )
        ).post("/v1/completions", json=payload)
        assert invalid.status_code == 200
        invalid_choice = invalid.json()["choices"][0]
        assert invalid_choice["text"] == ""
        assert invalid_choice["finish_reason"] == "stop"
        assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_numeric_multiple_of() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {"score": {"type": "number", "multipleOf": 0.25}},
            "required": ["score"],
            "additionalProperties": False,
        },
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"score":1.5}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"score":1.3}']),
        )
    )
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }

    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"score":1.5}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_unique_items() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}},
            "required": ["tags"],
            "additionalProperties": False,
        },
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"tags":["docs","api"]}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"tags":["docs","docs"]}']),
        )
    )
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }

    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"tags":["docs","api"]}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_array_contains() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
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
    }
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"scores":[0.1,0.95,0.99]}']),
        )
    )
    too_few_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"scores":[0.1,0.2]}']),
        )
    )
    too_many_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"scores":[0.9,0.95,0.99]}']),
        )
    )

    valid = valid_client.post("/v1/completions", json=payload)
    too_few = too_few_client.post("/v1/completions", json=payload)
    too_many = too_many_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"scores":[0.1,0.95,0.99]}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for response in (too_few, too_many):
        assert response.status_code == 200
        choice = response.json()["choices"][0]
        assert choice["text"] == ""
        assert choice["finish_reason"] == "stop"
        assert choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_uses_json_typed_equality() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {
                "const_num": {"const": 1},
                "enum_num": {"enum": [1]},
                "mixed": {"type": "array", "uniqueItems": True},
            },
            "required": ["const_num", "enum_num", "mixed"],
            "additionalProperties": False,
        },
    }
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"const_num":1,"enum_num":1,"mixed":[true,1,"1"]}']),
        )
    )
    invalid_outputs = [
        '{"const_num":true,"enum_num":1,"mixed":[true,1,"1"]}',
        '{"const_num":1,"enum_num":true,"mixed":[true,1,"1"]}',
        '{"const_num":1,"enum_num":1,"mixed":[1,1.0]}',
    ]

    valid = valid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"const_num":1,"enum_num":1,"mixed":[true,1,"1"]}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for generated in invalid_outputs:
        invalid = TestClient(
            create_app(
                ServerConfig(model="fake-path", served_model_name="fake-model"),
                llm=FakeLLM(outputs=[generated]),
            )
        ).post("/v1/completions", json=payload)
        assert invalid.status_code == 200
        invalid_choice = invalid.json()["choices"][0]
        assert invalid_choice["text"] == ""
        assert invalid_choice["finish_reason"] == "stop"
        assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_composition_keywords() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {
                "target": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
                "score": {"type": "number", "allOf": [{"minimum": 0}, {"maximum": 1}]},
                "label": {"type": "string", "allOf": [{"minLength": 2}, {"maxLength": 4}]},
                "ambiguous": {"oneOf": [{"type": "number"}, {"type": "integer"}]},
                "debug": {"not": {"const": True}},
            },
            "required": ["target", "score", "label", "ambiguous", "debug"],
            "additionalProperties": False,
        },
    }
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"target":null,"score":0.5,"label":"ok","ambiguous":1.5,"debug":false}']),
        )
    ).post("/v1/completions", json=payload)
    invalid_outputs = [
        '{"target":7,"score":0.5,"label":"ok","ambiguous":1.5,"debug":false}',
        '{"target":null,"score":2,"label":"ok","ambiguous":1.5,"debug":false}',
        '{"target":null,"score":0.5,"label":"x","ambiguous":1.5,"debug":false}',
        '{"target":null,"score":0.5,"label":"ok","ambiguous":1,"debug":false}',
        '{"target":null,"score":0.5,"label":"ok","ambiguous":1.5,"debug":true}',
    ]

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == (
        '{"target":null,"score":0.5,"label":"ok","ambiguous":1.5,"debug":false}'
    )
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for generated in invalid_outputs:
        invalid = TestClient(
            create_app(
                ServerConfig(model="fake-path", served_model_name="fake-model"),
                llm=FakeLLM(outputs=[generated]),
            )
        ).post("/v1/completions", json=payload)
        assert invalid.status_code == 200
        invalid_choice = invalid.json()["choices"][0]
        assert invalid_choice["text"] == ""
        assert invalid_choice["finish_reason"] == "stop"
        assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_length_rejects_invalid_json_continuation() -> None:
    client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(
                detailed_outputs=[
                    GenerationOutput(
                        text='{"ok": [1}',
                        finish_details=FinishDetails(reason="length", length_limit=9),
                    )
                ]
            ),
        )
    )

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {"type": "json_schema", "json_schema": _response_json_schema()},
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == '{"ok": [1}'
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_completions_guided_json_true_validates_object_result() -> None:
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=["[true]"]),
        )
    )
    payload = {"model": "fake-model", "prompt": "return json", "guided_json": True}

    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"ok":true}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_guided_json_length_rejects_invalid_json_continuation() -> None:
    client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(
                detailed_outputs=[
                    GenerationOutput(
                        text='{"ok": [1}',
                        finish_details=FinishDetails(reason="length", length_limit=9),
                    )
                ]
            ),
        )
    )

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "return json", "guided_json": True},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == '{"ok": [1}'
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_completions_guided_json_schema_validates_result() -> None:
    schema = _response_json_schema()["schema"]
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"path":"README.md"}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":"yes","path":"README.md"}']),
        )
    )
    payload = {"model": "fake-model", "prompt": "return json", "guided_json": schema}

    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"ok":true,"path":"README.md"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_guided_json_schema_validates_local_refs() -> None:
    schema = {
        "type": "object",
        "$defs": {
            "file_ref": {"type": "string", "pattern": r"^[A-Z]+[.]md$"},
            "result_tag": {"enum": ["ok", "skip"]},
        },
        "properties": {
            "path": {"$ref": "#/$defs/file_ref"},
            "result": {"$ref": "#/$defs/result_tag"},
        },
        "required": ["path", "result"],
        "additionalProperties": False,
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"path":"README.md","result":"ok"}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"path":"readme.md","result":"ok"}']),
        )
    )
    payload = {
        "model": "fake-model",
        "prompt": "return json",
        "guided_json": {"schema": schema},
    }

    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"path":"README.md","result":"ok"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_guided_json_schema_length_rejects_invalid_json_continuation() -> None:
    schema = _response_json_schema()["schema"]
    client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(
                detailed_outputs=[
                    GenerationOutput(
                        text='{"ok": [1}',
                        finish_details=FinishDetails(reason="length", length_limit=9),
                    )
                ]
            ),
        )
    )

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "return json", "guided_json": schema},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == '{"ok": [1}'
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_completions_response_format_rejects_unsupported_modes() -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    unsupported = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {"type": "xml"},
        },
    )
    missing_schema = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
        },
    )
    echo = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "echo": True,
            "response_format": {"type": "json_object"},
        },
    )

    assert unsupported.status_code == 400
    assert unsupported.json()["error"]["code"] == "unsupported_parameter"
    assert unsupported.json()["error"]["param"] == "response_format"
    assert missing_schema.status_code == 400
    assert missing_schema.json()["error"]["code"] == "invalid_request"
    assert missing_schema.json()["error"]["param"] == "response_format.json_schema.schema"
    assert missing_schema.json()["error"]["hipengine"]["code"] == "schema_violation"
    assert missing_schema.json()["error"]["hipengine"]["legacy_code"] == "invalid_request"
    assert echo.status_code == 400
    assert echo.json()["error"]["code"] == "invalid_request"
    assert echo.json()["error"]["param"] == "echo"
    assert echo.json()["error"]["hipengine"]["code"] == "schema_violation"
    assert echo.json()["error"]["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_unsupported_schema_keywords() -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {
                        "type": "object",
                        "$anchor": "agent_result",
                    },
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.$anchor"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_invalid_composition_schema() -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"anyOf": []}},
                    },
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.properties.ok.anyOf"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


@pytest.mark.parametrize(
    ("schema", "param"),
    [
        (
            {"type": "object", "if": "file"},
            "response_format.json_schema.schema.if",
        ),
        (
            {"type": "object", "if": {"type": "object"}, "then": "file"},
            "response_format.json_schema.schema.then",
        ),
        (
            {"type": "object", "if": {"type": "object"}, "else": "url"},
            "response_format.json_schema.schema.else",
        ),
    ],
)
def test_completions_response_format_rejects_invalid_conditional_schema(
    schema: dict[str, Any],
    param: str,
) -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": schema,
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


@pytest.mark.parametrize(
    ("schema", "param"),
    [
        (
            {"type": "object", "$ref": "https://example.test/schema.json"},
            "response_format.json_schema.schema.$ref",
        ),
        (
            {"type": "object", "$defs": {}, "$ref": "#/$defs/missing"},
            "response_format.json_schema.schema.$ref",
        ),
        (
            {
                "type": "object",
                "$defs": {"node": {"$ref": "#/$defs/node"}},
                "properties": {"node": {"$ref": "#/$defs/node"}},
            },
            "response_format.json_schema.schema.$defs.node.$ref",
        ),
        (
            {"type": "object", "$defs": {"name": "schema"}},
            "response_format.json_schema.schema.$defs.name",
        ),
    ],
)
def test_completions_response_format_rejects_invalid_schema_refs(
    schema: dict[str, Any],
    param: str,
) -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": schema,
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_invalid_schema_pattern() -> None:
    fake = FakeLLM(outputs=['{"ok":true,"path":"README.md"}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "pattern": "["}},
                    },
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.properties.path.pattern"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_invalid_property_count_bound() -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {"type": "object", "minProperties": -1},
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.minProperties"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_invalid_additional_properties_schema() -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {"type": "object", "additionalProperties": ["string"]},
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.additionalProperties"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_invalid_pattern_properties_schema() -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {"type": "object", "patternProperties": {"[": {"type": "string"}}},
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.patternProperties.["
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


@pytest.mark.parametrize(
    ("schema", "param"),
    [
        (
            {"type": "object", "propertyNames": "name"},
            "response_format.json_schema.schema.propertyNames",
        ),
        (
            {"type": "object", "dependentRequired": {"id": "name"}},
            "response_format.json_schema.schema.dependentRequired.id",
        ),
        (
            {"type": "object", "dependentSchemas": "path"},
            "response_format.json_schema.schema.dependentSchemas",
        ),
        (
            {"type": "object", "dependentSchemas": {"path": "schema"}},
            "response_format.json_schema.schema.dependentSchemas.path",
        ),
    ],
)
def test_completions_response_format_rejects_invalid_object_keyword_schema(
    schema: dict[str, Any],
    param: str,
) -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": schema,
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_invalid_multiple_of_bound() -> None:
    fake = FakeLLM(outputs=['{"score":1}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {
                        "type": "object",
                        "properties": {"score": {"type": "number", "multipleOf": 0}},
                    },
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.properties.score.multipleOf"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_invalid_unique_items_bound() -> None:
    fake = FakeLLM(outputs=['{"tags":["docs"]}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {
                        "type": "object",
                        "properties": {"tags": {"type": "array", "uniqueItems": "yes"}},
                    },
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.properties.tags.uniqueItems"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


@pytest.mark.parametrize(
    ("schema", "param"),
    [
        (
            {"type": "array", "contains": "number"},
            "response_format.json_schema.schema.contains",
        ),
        (
            {"type": "array", "contains": {"type": "number"}, "minContains": -1},
            "response_format.json_schema.schema.minContains",
        ),
    ],
)
def test_completions_response_format_rejects_invalid_contains_schema(
    schema: dict[str, Any],
    param: str,
) -> None:
    fake = FakeLLM(outputs=["[]"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": schema,
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_accepts_annotation_schema_keywords() -> None:
    fake = FakeLLM(outputs=['{"ok":true,"path":"README.md"}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {
                        "title": "Agent result",
                        "description": "Annotation-only metadata is ignored by validation.",
                        "default": {"ok": True, "path": "README.md"},
                        "type": "object",
                        "properties": {
                            "ok": {"type": "boolean", "description": "success flag"},
                            "path": {
                                "type": "string",
                                "examples": ["README.md"],
                                "format": "uri-reference",
                            },
                        },
                        "required": ["ok", "path"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == '{"ok":true,"path":"README.md"}'
    assert choice["finish_details"] == _stateless_finish_details("stop")
    assert len(fake.calls) == 1


def test_completions_guided_choice_validates_result() -> None:
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=["yes"]),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=["maybe"]),
        )
    )
    payload = {
        "model": "fake-model",
        "prompt": "answer yes or no",
        "guided_choice": ["yes", "no"],
    }

    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == "yes"
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_guided_regex_validates_result() -> None:
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=["AB-12"]),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=["AB"]),
        )
    )
    payload = {
        "model": "fake-model",
        "prompt": "return an id",
        "guided_regex": r"[A-Z]{2}-\d{2}",
    }

    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == "AB-12"
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_guided_diff_validates_unified_diff_result() -> None:
    diff_text = _unified_diff_text()
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=[diff_text]),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=["Here is the patch:\n" + diff_text]),
        )
    )

    payload = {
        "model": "fake-model",
        "prompt": "patch",
        "guided_diff": {"type": "unified_diff"},
    }
    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == diff_text
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


@pytest.mark.parametrize(
    ("guided_diff", "generated", "expected_text"),
    [
        ({"type": "unified_diff", "fenced": False}, _unified_diff_text(), _unified_diff_text()),
        ({"type": "unified_diff", "fenced": "forbidden"}, f"```diff\n{_unified_diff_text()}\n```", ""),
        ({"type": "unified_diff", "fenced": True}, f"```patch\n{_unified_diff_text()}\n```", f"```patch\n{_unified_diff_text()}\n```"),
        ({"type": "unified_diff", "fenced": "required"}, _unified_diff_text(), ""),
        ({"type": "unified_diff", "fenced": "optional"}, f"```diff\n{_unified_diff_text()}\n```", f"```diff\n{_unified_diff_text()}\n```"),
    ],
)
def test_completions_guided_diff_enforces_fenced_policy(guided_diff, generated, expected_text) -> None:
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=[generated]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "patch", "guided_diff": guided_diff},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == expected_text
    expected_reason = "stop" if expected_text else "schema_violation"
    assert choice["finish_details"] == _stateless_finish_details(expected_reason)


def test_streaming_completion_guided_diff_buffers_validation_failure() -> None:
    fake = FakeLLM(outputs=["not a diff"], stream_chunks=[_unified_diff_text()])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "patch",
            "guided_diff": True,
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0].get("text") for payload in payloads if payload.get("choices"))
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
    }
    assert fake.stream_calls == []


def test_streaming_completion_guided_choice_buffers_validation_failure() -> None:
    fake = FakeLLM(outputs=["maybe"], stream_chunks=["yes"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "answer yes or no",
            "guided_choice": ["yes", "no"],
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0].get("text") for payload in payloads if payload.get("choices"))
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
    }
    assert fake.stream_calls == []


def test_streaming_completion_guided_regex_buffers_validation_failure() -> None:
    fake = FakeLLM(outputs=["AB"], stream_chunks=["AB-12"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "return an id",
            "guided_regex": r"[A-Z]{2}-\d{2}",
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0].get("text") for payload in payloads if payload.get("choices"))
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
    }
    assert fake.stream_calls == []


def test_streaming_completion_response_format_buffers_validation() -> None:
    fake = FakeLLM(outputs=["not json"], stream_chunks=["wrong"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "stream": True,
            "response_format": {"type": "json_object"},
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0].get("text") for payload in payloads if payload.get("choices"))
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
    }
    assert fake.stream_calls == []


def test_streaming_completion_guided_json_buffers_validation() -> None:
    fake = FakeLLM(outputs=["not json"], stream_chunks=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "stream": True,
            "guided_json": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0].get("text") for payload in payloads if payload.get("choices"))
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
    }
    assert fake.stream_calls == []


def test_streaming_completion_response_format_emits_structured_metadata() -> None:
    structured_text = '{"ok": true, "path": "README.md"}'
    fake = FakeLLM(outputs=[structured_text], stream_chunks=["wrong"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "return json",
            "stream": True,
            "response_format": {"type": "json_object"},
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    delta = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0].get("text"))
    assert delta["choices"][0]["text"] == structured_text
    assert delta["choices"][0]["hipengine"] == {
        "phase": "structured",
        "tokens": {
            "streamed_tokens": 4,
            "delta_tokens": 4,
            "structured_tokens": 4,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 4,
            "prompt_tokens": 0,
            "generated_tokens": 4,
            "phase": "structured",
            "continuation_eligible": False,
            "structured_tokens": 4,
        },
    }
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("stop"),
        "tokens": {
            "prompt_tokens": 2,
            "completion_tokens": 4,
            "total_tokens": 6,
            "streamed_tokens": 4,
            "structured_tokens": 4,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 4,
            "prompt_tokens": 2,
            "generated_tokens": 4,
            "phase": "structured",
            "continuation_eligible": False,
            "structured_tokens": 4,
        },
    }
    usage = next(payload for payload in payloads if payload.get("usage"))
    assert usage["usage"] == {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}
    assert fake.stream_calls == []


def test_streaming_completion_response_format_json_schema_buffers_validation() -> None:
    fake = FakeLLM(outputs=['{"ok":"yes","path":"README.md"}'], stream_chunks=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "stream": True,
            "response_format": {"type": "json_schema", "json_schema": _response_json_schema()},
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0].get("text") for payload in payloads if payload.get("choices"))
    done = next(
        payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"]
    )
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
    }
    assert fake.stream_calls == []


def test_chat_completion_response_format_json_object_validates_visible_content() -> None:
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['<think>check</think>{"ok":true}']),
        )
    )
    invalid_fake = FakeLLM(outputs=["not json"])
    invalid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=invalid_fake)
    )

    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "return json"}],
        "response_format": {"type": "json_object"},
    }
    valid = valid_client.post("/v1/chat/completions", json=payload)
    invalid = invalid_client.post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["message"]["content"] == '{"ok":true}'
    assert valid_choice["message"]["reasoning_content"] == "check"
    assert valid_choice["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "Return only one valid JSON object" in invalid_fake.calls[0][0][0]


def test_chat_completion_response_format_json_schema_validates_visible_content() -> None:
    valid_fake = FakeLLM(outputs=['<think>check</think>{"ok":true,"path":"README.md"}'])
    invalid_fake = FakeLLM(outputs=['{"ok":true,"path":""}'])
    valid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=valid_fake)
    )
    invalid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=invalid_fake)
    )

    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "return json"}],
        "response_format": {"type": "json_schema", "json_schema": _response_json_schema()},
    }
    valid = valid_client.post("/v1/chat/completions", json=payload)
    invalid = invalid_client.post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["message"]["content"] == '{"ok":true,"path":"README.md"}'
    assert valid_choice["message"]["reasoning_content"] == "check"
    assert valid_choice["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "Return only JSON that satisfies this JSON schema" in invalid_fake.calls[0][0][0]


def test_chat_completion_response_format_json_schema_length_rejects_invalid_json_continuation() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='{"ok": [1}',
                finish_details=FinishDetails(reason="length", length_limit=9),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_schema", "json_schema": _response_json_schema()},
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": '{"ok": [1}'}
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_chat_completion_guided_json_schema_validates_visible_content() -> None:
    valid_fake = FakeLLM(outputs=['<think>check</think>{"ok":true,"path":"README.md"}'])
    invalid_fake = FakeLLM(outputs=['{"ok":true,"path":""}'])
    valid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=valid_fake)
    )
    invalid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=invalid_fake)
    )
    schema_text = json.dumps(_response_json_schema()["schema"])

    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "return json"}],
        "guided_json": schema_text,
    }
    valid = valid_client.post("/v1/chat/completions", json=payload)
    invalid = invalid_client.post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["message"]["content"] == '{"ok":true,"path":"README.md"}'
    assert valid_choice["message"]["reasoning_content"] == "check"
    assert valid_choice["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "Return only JSON that satisfies this JSON schema" in valid_fake.calls[0][0][0]


def test_chat_completion_guided_json_schema_validates_local_refs() -> None:
    schema = {
        "type": "object",
        "$defs": {
            "file_ref": {"type": "string", "pattern": r"^[A-Z]+[.]md$"},
            "result_tag": {"enum": ["ok", "skip"]},
        },
        "properties": {
            "path": {"$ref": "#/$defs/file_ref"},
            "result": {"$ref": "#/$defs/result_tag"},
        },
        "required": ["path", "result"],
        "additionalProperties": False,
    }
    valid_fake = FakeLLM(outputs=['<think>check</think>{"path":"README.md","result":"ok"}'])
    invalid_fake = FakeLLM(outputs=['{"path":"README.md","result":"done"}'])
    valid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=valid_fake)
    )
    invalid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=invalid_fake)
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "return json"}],
        "guided_json": json.dumps(schema),
    }

    valid = valid_client.post("/v1/chat/completions", json=payload)
    invalid = invalid_client.post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["message"]["content"] == '{"path":"README.md","result":"ok"}'
    assert valid_choice["message"]["reasoning_content"] == "check"
    assert valid_choice["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "Return only JSON that satisfies this JSON schema" in valid_fake.calls[0][0][0]


def test_chat_guided_json_schema_length_rejects_invalid_json_continuation() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='{"ok": [1}',
                finish_details=FinishDetails(reason="length", length_limit=9),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    schema_text = json.dumps(_response_json_schema()["schema"])

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "guided_json": schema_text,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": '{"ok": [1}'}
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_chat_completion_guided_choice_validates_visible_content() -> None:
    valid_fake = FakeLLM(outputs=["<think>choose</think>no"])
    invalid_fake = FakeLLM(outputs=["<think>choose</think>maybe"])
    valid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=valid_fake)
    )
    invalid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=invalid_fake)
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "answer yes or no"}],
        "guided_choice": ["yes", "no"],
    }

    valid = valid_client.post("/v1/chat/completions", json=payload)
    invalid = invalid_client.post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["message"]["content"] == "no"
    assert valid_choice["message"]["reasoning_content"] == "choose"
    assert valid_choice["finish_details"] == _stateless_finish_details("stop")
    assert 'Return exactly one of these choices and no other text: ["yes","no"]' in valid_fake.calls[0][0][0]
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_chat_completion_guided_regex_validates_visible_content() -> None:
    valid_fake = FakeLLM(outputs=["<think>format</think>AB-12"])
    invalid_fake = FakeLLM(outputs=["<think>format</think>AB"])
    valid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=valid_fake)
    )
    invalid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=invalid_fake)
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "return an id"}],
        "guided_regex": r"[A-Z]{2}-\d{2}",
    }

    valid = valid_client.post("/v1/chat/completions", json=payload)
    invalid = invalid_client.post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["message"]["content"] == "AB-12"
    assert valid_choice["message"]["reasoning_content"] == "format"
    assert valid_choice["finish_details"] == _stateless_finish_details("stop")
    assert (
        'Return text that fully matches this regular expression and no other text: "[A-Z]{2}-\\\\d{2}"'
        in valid_fake.calls[0][0][0]
    )
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_chat_completion_guided_patch_validates_visible_unified_diff() -> None:
    diff_text = _unified_diff_text()
    fenced_diff = f"<think>patch</think>```diff\n{diff_text}\n```"
    valid_fake = FakeLLM(outputs=[fenced_diff])
    invalid_fake = FakeLLM(outputs=[f"<think>patch</think>Here is the patch:\n{diff_text}"])
    valid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=valid_fake)
    )
    invalid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=invalid_fake)
    )

    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "edit README"}],
        "guided_patch": {"format": "unified-diff"},
    }
    valid = valid_client.post("/v1/chat/completions", json=payload)
    invalid = invalid_client.post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["message"]["content"] == f"```diff\n{diff_text}\n```"
    assert valid_choice["message"]["reasoning_content"] == "patch"
    assert valid_choice["finish_details"] == _stateless_finish_details("stop")
    assert "Return only a valid unified diff patch" in valid_fake.calls[0][0][0]
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_chat_completion_response_format_length_keeps_partial_json() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='{"ok":',
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
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_object"},
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    continuation_id = choice["continuation_id"]
    assert choice["message"] == {"role": "assistant", "content": '{"ok":'}
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=6,
        phase="structured",
        continuation_eligible=True,
        continuation_id=continuation_id,
    )


def test_chat_completion_response_format_length_rejects_invalid_json_continuation() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='{"ok": [1}',
                finish_details=FinishDetails(reason="length", length_limit=9),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_object"},
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": '{"ok": [1}'}
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_chat_completion_response_format_json_schema_length_rejects_non_object_prefix() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='<|im_end|> {"ok":true,"path":"README.md"}',
                finish_details=FinishDetails(reason="length", length_limit=16),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_schema", "json_schema": _response_json_schema()},
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"] == {
        "role": "assistant",
        "content": '<|im_end|> {"ok":true,"path":"README.md"}',
    }
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=16,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_chat_completion_response_format_length_marks_complete_json_structured() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='{"ok":true}',
                finish_details=FinishDetails(reason="length", length_limit=11),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_object"},
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    continuation_id = choice["continuation_id"]
    assert choice["message"] == {"role": "assistant", "content": '{"ok":true}'}
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=11,
        phase="structured",
        continuation_eligible=True,
        continuation_id=continuation_id,
    )


def test_chat_continuation_resumes_partial_guided_patch_and_inherits_validation() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old",
                finish_details=FinishDetails(reason="length", length_limit=12),
            ),
            GenerationOutput(text="\n+new", finish_details=FinishDetails(reason="stop")),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "edit README"}],
            "guided_patch": True,
            "max_tokens": 12,
            "temperature": 0.0,
        },
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert first_choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=12,
        phase="structured",
        continuation_eligible=True,
        continuation_id=continuation_id,
    )

    second = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["message"]["content"].endswith("-old\n+new")
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["finish_details"] == _stateless_finish_details("stop")


def test_chat_continuation_resumes_partial_json_and_inherits_response_format() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text='{"ok":',
                finish_details=FinishDetails(reason="length", length_limit=6),
            ),
            GenerationOutput(text="true}", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_object"},
            "max_tokens": 6,
            "temperature": 0.0,
        },
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert first_choice["message"] == {"role": "assistant", "content": '{"ok":'}
    assert first_choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=6,
        phase="structured",
        continuation_eligible=True,
        continuation_id=continuation_id,
    )

    second = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["message"] == {"role": "assistant", "content": '{"ok":true}'}
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["finish_details"] == _stateless_finish_details("eos", eos_token_id=151645)
    assert fake.calls[1][0][0].endswith('{"ok":')


def test_chat_continuation_resumes_partial_guided_json_and_inherits_validation() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text='{"ok":',
                finish_details=FinishDetails(reason="length", length_limit=6),
            ),
            GenerationOutput(text="true}", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "guided_json": True,
            "max_tokens": 6,
            "temperature": 0.0,
        },
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert first_choice["message"] == {"role": "assistant", "content": '{"ok":'}
    assert first_choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=6,
        phase="structured",
        continuation_eligible=True,
        continuation_id=continuation_id,
    )

    second = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["message"] == {"role": "assistant", "content": '{"ok":true}'}
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["finish_details"] == _stateless_finish_details("eos", eos_token_id=151645)
    assert fake.calls[1][0][0].endswith('{"ok":')


def test_chat_guided_json_length_rejects_invalid_json_continuation() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='{"ok": [1}',
                finish_details=FinishDetails(reason="length", length_limit=9),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "guided_json": True,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": '{"ok": [1}'}
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_chat_continuation_resumes_partial_guided_regex_and_inherits_validation() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="AB-",
                finish_details=FinishDetails(reason="length", length_limit=3),
            ),
            GenerationOutput(text="12", finish_details=FinishDetails(reason="stop")),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return an id"}],
            "guided_regex": r"[A-Z]{2}-\d{2}",
            "max_tokens": 3,
            "temperature": 0.0,
        },
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert first_choice["message"] == {"role": "assistant", "content": "AB-"}
    assert first_choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=3,
        phase="structured",
        continuation_eligible=True,
        continuation_id=continuation_id,
    )

    second = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 2},
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["message"] == {"role": "assistant", "content": "AB-12"}
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["finish_details"] == _stateless_finish_details("stop")


def test_chat_continuation_resumes_partial_guided_choice_and_inherits_validation() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="ye",
                finish_details=FinishDetails(reason="length", length_limit=2),
            ),
            GenerationOutput(text="s", finish_details=FinishDetails(reason="stop")),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "answer yes or no"}],
            "guided_choice": ["yes", "no"],
            "max_tokens": 2,
            "temperature": 0.0,
        },
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert first_choice["message"] == {"role": "assistant", "content": "ye"}
    assert first_choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=2,
        phase="structured",
        continuation_eligible=True,
        continuation_id=continuation_id,
    )

    second = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 1},
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["message"] == {"role": "assistant", "content": "yes"}
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["finish_details"] == _stateless_finish_details("stop")


def test_chat_continuation_resume_rejects_explicit_response_format_override() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text='{"ok":',
                finish_details=FinishDetails(reason="length", length_limit=6),
            ),
            GenerationOutput(text="true}", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_object"},
            "max_tokens": 6,
        },
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    override = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "response_format": {"type": "text"},
            "max_tokens": 4,
        },
    )

    assert override.status_code == 400
    assert override.json()["error"]["code"] == "unsupported_parameter"
    assert override.json()["error"]["param"] == "response_format"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    inherited = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert inherited.status_code == 200
    assert inherited.json()["choices"][0]["message"] == {"role": "assistant", "content": '{"ok":true}'}
    assert len(fake.calls) == 2


def test_streaming_chat_completion_response_format_buffers_validation() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": 0,
                        "token_index": 0,
                        "token_id": 400,
                        "finished": True,
                        "chunk": {
                            "text": "not json",
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=1,
                                row_index=0,
                                request_id="0",
                                phase="answer",
                                sampler_mode="greedy_fast",
                                execution_path="scheduler_invalid_should_not_surface",
                            ).to_json_dict(),
                        },
                    }
                ]
            }
            return [GenerationOutput(text="not json") for _prompt in prompt_tuple]

    fake = SchedulerChunkFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_object"},
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0]["delta"].get("content") for payload in payloads if payload.get("choices"))
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
    }
    assert "scheduler_invalid_should_not_surface" not in response.text


def test_streaming_chat_completion_response_format_emits_structured_metadata() -> None:
    structured_text = '{"ok": true, "path": "README.md"}'
    fake = FakeLLM(outputs=[structured_text], stream_chunks=["wrong"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_object"},
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content = next(
        payload
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    )
    assert content["choices"][0]["delta"] == {"content": structured_text}
    assert content["choices"][0]["hipengine"] == {
        "phase": "structured",
        "tokens": {
            "streamed_tokens": 4,
            "delta_tokens": 4,
            "structured_tokens": 4,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 4,
            "prompt_tokens": 0,
            "generated_tokens": 4,
            "phase": "structured",
            "continuation_eligible": False,
            "structured_tokens": 4,
        },
    }
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    done_hipengine = done["choices"][0]["hipengine"]
    assert done_hipengine["phase"] == "structured"
    assert done_hipengine["finish_details"] == _stateless_finish_details("stop")
    assert done_hipengine["tokens"] == {
        "streamed_tokens": 4,
        "structured_tokens": 4,
    }
    assert done_hipengine["decode_state"]["generated_tokens"] == 4
    assert done_hipengine["decode_state"]["phase"] == "structured"
    assert done_hipengine["decode_state"]["structured_tokens"] == 4
    usage = next(payload for payload in payloads if payload.get("usage"))
    assert usage["usage"]["completion_tokens"] == 4
    assert fake.stream_calls == []


def test_streaming_chat_completion_response_format_uses_scheduler_structured_chunks() -> None:
    structured_text = '{"ok": true}'
    chunk_texts = ('{"ok":', " true}")

    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": 0,
                        "token_index": token_index,
                        "token_id": 500 + token_index,
                        "finished": token_index == len(chunk_texts) - 1,
                        "chunk": {
                            "text": chunk_text,
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=token_index + 1,
                                row_index=0,
                                request_id="0",
                                phase="answer",
                                sampler_mode="greedy_fast",
                                execution_path="scheduler_structured_chunks",
                            ).to_json_dict(),
                        },
                    }
                    for token_index, chunk_text in enumerate(chunk_texts)
                ]
            }
            return [
                GenerationOutput(
                    text=structured_text,
                    finish_details=FinishDetails(reason="stop", sampler_mode="greedy_fast"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=len(chunk_texts),
                        row_index=0,
                        phase="done",
                        sampler_mode="greedy_fast",
                    ),
                )
                for _prompt in prompt_tuple
            ]

    fake = SchedulerChunkFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_object"},
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    ]
    assert [choice["delta"]["content"] for choice in content] == list(chunk_texts)
    assert [choice["hipengine"]["phase"] for choice in content] == ["structured", "structured"]
    assert {
        choice["hipengine"]["decode_state"]["execution_path"]
        for choice in content
    } == {"scheduler_structured_chunks"}
    assert {
        choice["hipengine"]["decode_state"]["phase"]
        for choice in content
    } == {"structured"}
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == {
        "reason": "stop",
        "cache_action": "append_none",
        "sampler_mode": "greedy_fast",
    }
    assert done["choices"][0]["hipengine"]["phase"] == "structured"
    assert done["choices"][0]["hipengine"]["decode_state"]["execution_path"] == "scheduler_structured_chunks"
    assert fake.stream_calls == []


def test_streaming_chat_completion_response_format_json_schema_buffers_validation() -> None:
    fake = FakeLLM(outputs=['{"ok":true,"path":""}'], stream_chunks=['{"ok":true,"path":"README.md"}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_schema", "json_schema": _response_json_schema()},
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(
        payload["choices"][0]["delta"].get("content")
        for payload in payloads
        if payload.get("choices")
    )
    done = next(
        payload
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"]
    )
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
    }
    assert fake.stream_calls == []


@pytest.mark.parametrize(
    ("endpoint", "prompt_text", "request_extra", "generated", "sampling_assertion"),
    [
        (
            "/v1/completions",
            "secret guided json prompt",
            {"guided_json": True},
            "[true]",
            ("guided_json", True),
        ),
        (
            "/v1/chat/completions",
            "secret guided regex task",
            {"guided_regex": r"[A-Z]{2}-\d{2}"},
            "AB",
            ("guided_regex", {"length": len(r"[A-Z]{2}-\d{2}")}),
        ),
        (
            "/v1/chat/completions",
            "secret guided choice task",
            {"guided_choice": ["yes", "no"]},
            "maybe",
            ("guided_choice", [{"length": 3}, {"length": 2}]),
        ),
    ],
)
def test_replay_artifact_captures_guided_result_validation_failure(
    tmp_path,
    endpoint: str,
    prompt_text: str,
    request_extra: dict[str, Any],
    generated: str,
    sampling_assertion: tuple[str, Any],
) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(outputs=[generated]),
    )
    client = TestClient(app)
    is_chat = endpoint.endswith("/chat/completions")
    if not is_chat:
        payload: dict[str, Any] = {"model": "fake-model", "prompt": prompt_text, "max_tokens": 16}
    else:
        payload = {
            "model": "fake-model",
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": 16,
        }
    payload.update(request_extra)

    response = client.post(endpoint, json=payload)

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("schema_violation")
    if is_chat:
        assert choice["message"] == {"role": "assistant", "content": ""}
        prompt_path = "$.messages[0].content"
    else:
        assert choice["text"] == ""
        prompt_path = "$.prompt"

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["schema"] == "hipengine.replay.v1"
    assert artifact["request"]["path"] == endpoint
    redacted_prompt = (
        artifact["request"]["json"]["messages"][0]["content"]
        if is_chat
        else artifact["request"]["json"]["prompt"]
    )
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": prompt_path,
            "sha256": redacted_prompt["sha256"],
            "length": len(prompt_text),
        }
    ]
    field, expected_sampling = sampling_assertion
    actual_sampling = artifact["sampling"][field]
    if isinstance(expected_sampling, list):
        assert [item["length"] for item in actual_sampling] == [
            item["length"] for item in expected_sampling
        ]
        assert all(item["redacted"] == "sha256" for item in actual_sampling)
    elif isinstance(expected_sampling, dict):
        assert actual_sampling["redacted"] == "sha256"
        assert actual_sampling["length"] == expected_sampling["length"]
    else:
        assert actual_sampling is expected_sampling
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
    assert prompt_text not in serialized
    assert generated not in serialized


def test_replay_artifact_captures_guided_patch_result_validation_failure(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    invalid_patch = "Here is the patch:\n" + _unified_diff_text()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(outputs=[invalid_patch]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "secret patch task"}],
            "guided_patch": True,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert choice["message"] == {"role": "assistant", "content": ""}

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["schema"] == "hipengine.replay.v1"
    assert artifact["request"]["path"] == "/v1/chat/completions"
    assert artifact["request"]["json"]["messages"][0]["content"]["redacted"] == "sha256"
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": "$.messages[0].content",
            "sha256": artifact["request"]["json"]["messages"][0]["content"]["sha256"],
            "length": len("secret patch task"),
        }
    ]
    assert artifact["sampling"]["guided_patch"] is True
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
    assert "secret patch task" not in serialized
    assert "Here is the patch" not in serialized
    assert "diff --git" not in serialized


def test_replay_artifact_captures_streaming_guided_diff_result_validation_failure(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(outputs=["not a diff"], stream_chunks=["not a diff"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "secret streaming patch prompt",
            "guided_diff": True,
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
    assert artifact["request"]["path"] == "/v1/completions"
    assert artifact["sampling"]["guided_diff"] is True
    assert artifact["finish_details"] == _stateless_finish_details("schema_violation")
    assert artifact["error"] is None
    assert artifact["result"]["choices"] == [
        {
            "index": 0,
            "finish_reason": "stop",
            "finish_details": _stateless_finish_details("schema_violation"),
        }
    ]
    assert "secret streaming patch prompt" not in serialized
    assert "not a diff" not in serialized


@pytest.mark.parametrize(
    ("payload", "param", "code"),
    [
        (
            {
                "guided_json": 7,
            },
            "guided_json",
            "invalid_request",
        ),
        (
            {
                "guided_json": "{not json}",
            },
            "guided_json",
            "invalid_request",
        ),
        (
            {
                "guided_json": {"$ref": "#/$defs/result"},
            },
            "guided_json.$ref",
            "invalid_request",
        ),
        (
            {
                "guided_json": {"schema": {"$ref": "#/$defs/result"}},
            },
            "guided_json.schema.$ref",
            "invalid_request",
        ),
        (
            {
                "guided_json": True,
                "response_format": {"type": "json_object"},
            },
            "guided_json",
            "invalid_request",
        ),
        (
            {
                "guided_json": True,
                "guided_regex": "[a-z]+",
            },
            "guided_json",
            "invalid_request",
        ),
        (
            {
                "guided_json": True,
                "guided_choice": ["yes", "no"],
            },
            "guided_json",
            "invalid_request",
        ),
        (
            {
                "guided_json": True,
                "guided_patch": True,
            },
            "guided_json",
            "invalid_request",
        ),
        (
            {
                "guided_regex": "",
            },
            "guided_regex",
            "invalid_request",
        ),
        (
            {
                "guided_regex": ["[a-z]+"],
            },
            "guided_regex",
            "invalid_request",
        ),
        (
            {
                "guided_regex": "[",
            },
            "guided_regex",
            "invalid_request",
        ),
        (
            {
                "guided_regex": "[a-z]+",
                "response_format": {"type": "json_object"},
            },
            "guided_regex",
            "invalid_request",
        ),
        (
            {
                "guided_regex": "[a-z]+",
                "guided_choice": ["yes", "no"],
            },
            "guided_regex",
            "invalid_request",
        ),
        (
            {
                "guided_regex": "[a-z]+",
                "guided_patch": True,
            },
            "guided_regex",
            "invalid_request",
        ),
        (
            {
                "guided_choice": [],
            },
            "guided_choice",
            "invalid_request",
        ),
        (
            {
                "guided_choice": ["yes", 1],
            },
            "guided_choice[1]",
            "invalid_request",
        ),
        (
            {
                "guided_choice": ["yes", "no"],
                "response_format": {"type": "json_object"},
            },
            "guided_choice",
            "invalid_request",
        ),
        (
            {
                "guided_choice": ["yes", "no"],
                "guided_patch": True,
            },
            "guided_choice",
            "invalid_request",
        ),
        (
            {
                "guided_patch": {"type": "regex"},
            },
            "guided_patch.type",
            "unsupported_parameter",
        ),
        (
            {
                "guided_patch": True,
                "guided_diff": True,
            },
            "guided_patch",
            "invalid_request",
        ),
        (
            {
                "guided_patch": True,
                "response_format": {"type": "json_object"},
            },
            "guided_patch",
            "invalid_request",
        ),
        (
            {
                "guided_patch": {"fenced": "sometimes"},
            },
            "guided_patch.fenced",
            "invalid_request",
        ),
        (
            {
                "guided_patch": {"fenced": None},
            },
            "guided_patch.fenced",
            "invalid_request",
        ),
    ],
)
def test_guided_output_request_validation_fails_before_generation(payload, param, code) -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            **payload,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["param"] == param
    assert fake.calls == []
