"""OpenAI server integration tests: sessions.

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
    BackendErrorThenSequentialFakeLLM,
    FakeLLM,
    SequentialFakeLLM,
    SpeculativeMTPFakeLLM,
)
from tests.support.server_api_support import (
    _response_json_schema,
    _routing_metadata,
    _sse_payloads,
)


def test_session_metadata_list_and_delete_are_authenticated() -> None:
    fake = SequentialFakeLLM(["stored answer"])
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

    unauthorized = client.get("/v1/hipengine/sessions")
    created = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "secret list prompt"}],
            "session": {"id": "sess_list"},
            "max_tokens": 4,
        },
    )
    listed = client.get("/v1/hipengine/sessions", headers=headers)
    deleted = client.delete("/v1/hipengine/sessions/sess_list", headers=headers)
    deleted_again = client.delete("/v1/hipengine/sessions/sess_list", headers=headers)
    listed_after_delete = client.get("/v1/hipengine/sessions", headers=headers)

    assert unauthorized.status_code == 401
    assert created.status_code == 200
    assert listed.status_code == 200
    body = listed.json()
    assert body["object"] == "hipengine.sessions"
    assert body["storage"] == "app_local_transcript"
    assert body["resident_state_reuse"] is False
    assert body["includes_transcript"] is False
    assert body["active"] == 1
    assert body["pending_creations"] == 0
    assert body["max_active"] is None
    assert body["continuations"] == {"active": 0, "ttl_seconds": 900}
    assert len(body["sessions"]) == 1
    metadata = body["sessions"][0]
    assert metadata["id"] == "sess_list"
    assert metadata["storage"] == "app_local_transcript"
    assert metadata["resident_state_reuse"] is False
    assert metadata["message_count"] == 2
    assert isinstance(metadata["created"], int)
    assert isinstance(metadata["updated"], int)
    serialized = json.dumps(body)
    assert "secret list prompt" not in serialized
    assert "stored answer" not in serialized
    assert deleted.json() == {
        "object": "hipengine.session.deleted",
        "id": "sess_list",
        "deleted": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
    }
    assert deleted_again.json()["deleted"] is False
    assert listed_after_delete.json()["active"] == 0
    assert listed_after_delete.json()["sessions"] == []


def test_chat_session_fork_branches_visible_transcript_without_cross_contamination() -> None:
    fake = SequentialFakeLLM(["base answer", "left answer", "right answer"])
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
            "messages": [{"role": "user", "content": "base prompt"}],
            "session": {"id": "sess_root"},
            "max_tokens": 4,
        },
    )
    unauthorized = client.post("/v1/hipengine/sessions/sess_root/fork", json={"id": "sess_branch"})
    forked = client.post(
        "/v1/hipengine/sessions/sess_root/fork",
        headers=headers,
        json={"id": "sess_branch"},
    )
    continued_root = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "left turn"}],
            "session": {"id": "sess_root"},
            "max_tokens": 4,
        },
    )
    continued_branch = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "right turn"}],
            "session": {"id": "sess_branch"},
            "max_tokens": 4,
        },
    )
    listed = client.get("/v1/hipengine/sessions", headers=headers)

    assert created.status_code == 200
    assert unauthorized.status_code == 401
    assert forked.status_code == 200
    assert forked.json() == {
        "object": "hipengine.session.forked",
        "source_id": "sess_root",
        "id": "sess_branch",
        "forked": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "message_count": 2,
    }
    assert continued_root.status_code == 200
    assert continued_branch.status_code == 200
    root_prompt = fake.calls[1][0][0]
    branch_prompt = fake.calls[2][0][0]
    assert "base prompt" in root_prompt
    assert "base answer" in root_prompt
    assert "left turn" in root_prompt
    assert "right turn" not in root_prompt
    assert "base prompt" in branch_prompt
    assert "base answer" in branch_prompt
    assert "right turn" in branch_prompt
    assert "left turn" not in branch_prompt
    body = listed.json()
    assert body["active"] == 2
    counts = {session["id"]: session["message_count"] for session in body["sessions"]}
    assert counts == {"sess_root": 4, "sess_branch": 4}
    assert "base prompt" not in json.dumps(body)
    assert "base answer" not in json.dumps(body)


def test_chat_session_fork_rejects_existing_target_and_session_cap() -> None:
    fake = SequentialFakeLLM(["root answer", "target answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            api_key="secret",
            max_chat_sessions=2,
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    root = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "root"}],
            "session": {"id": "sess_root"},
            "max_tokens": 4,
        },
    )
    target = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "target"}],
            "session": {"id": "sess_target"},
            "max_tokens": 4,
        },
    )
    existing = client.post(
        "/v1/hipengine/sessions/sess_root/fork",
        headers=headers,
        json={"id": "sess_target"},
    )
    full = client.post(
        "/v1/hipengine/sessions/sess_root/fork",
        headers=headers,
        json={"id": "sess_new"},
    )
    missing = client.post(
        "/v1/hipengine/sessions/sess_missing/fork",
        headers=headers,
        json={"id": "sess_other"},
    )

    assert root.status_code == 200
    assert target.status_code == 200
    assert existing.status_code == 400
    assert existing.json()["error"]["param"] == "id"
    assert full.status_code == 429
    assert full.json()["error"]["code"] == "engine_busy"
    assert full.json()["error"]["message"] == "chat session limit is full"
    assert full.json()["error"]["hipengine"] == {
        "code": "engine_busy",
        "status_code": 429,
        "retryable": True,
        "routing": {
            **_routing_metadata(),
            "matched": True,
            "reason": "engine_busy",
            "overload_source": "chat_session_cap",
            "max_active_chat_sessions": 2,
        },
    }
    assert full.headers["retry-after"] == "1"
    assert missing.status_code == 404
    assert missing.json()["error"]["param"] == "session_id"
    assert set(app.state.hipengine_chat_sessions) == {"sess_root", "sess_target"}
    assert app.state.hipengine_server_metrics.request_rejected_total == 1


def test_chat_session_rollback_trims_visible_transcript_for_next_turn() -> None:
    fake = SequentialFakeLLM(["base answer", "second answer", "after rollback"])
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

    first = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "base prompt"}],
            "session": {"id": "sess_rollback"},
            "max_tokens": 4,
        },
    )
    second = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "second turn"}],
            "session": {"id": "sess_rollback"},
            "max_tokens": 4,
        },
    )
    unauthorized = client.post(
        "/v1/hipengine/sessions/sess_rollback/rollback",
        json={"message_count": 2},
    )
    rollback = client.post(
        "/v1/hipengine/sessions/sess_rollback/rollback",
        headers=headers,
        json={"message_count": 2},
    )
    continued = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "after rollback turn"}],
            "session": {"id": "sess_rollback", "commit": "append_none"},
            "max_tokens": 4,
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert unauthorized.status_code == 401
    assert rollback.status_code == 200
    assert rollback.json() == {
        "object": "hipengine.session.rolled_back",
        "id": "sess_rollback",
        "rolled_back": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "previous_message_count": 4,
        "message_count": 2,
    }
    assert continued.status_code == 200
    record = app.state.hipengine_chat_sessions["sess_rollback"]
    assert len(record.messages) == 2
    prompt = fake.calls[2][0][0]
    assert "base prompt" in prompt
    assert "base answer" in prompt
    assert "after rollback turn" in prompt
    assert "second turn" not in prompt
    assert "second answer" not in prompt
    listed = client.get("/v1/hipengine/sessions", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["sessions"][0]["message_count"] == 2
    assert "base prompt" not in json.dumps(listed.json())
    assert "base answer" not in json.dumps(listed.json())


def test_chat_session_rollback_rejects_missing_and_out_of_range() -> None:
    fake = SequentialFakeLLM(["root answer"])
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
            "messages": [{"role": "user", "content": "root prompt"}],
            "session": {"id": "sess_rollback_errors"},
            "max_tokens": 4,
        },
    )
    too_far = client.post(
        "/v1/hipengine/sessions/sess_rollback_errors/rollback",
        headers=headers,
        json={"message_count": 3},
    )
    missing = client.post(
        "/v1/hipengine/sessions/sess_missing/rollback",
        headers=headers,
        json={"message_count": 0},
    )
    noop = client.post(
        "/v1/hipengine/sessions/sess_rollback_errors/rollback",
        headers=headers,
        json={"message_count": 2},
    )

    assert created.status_code == 200
    assert too_far.status_code == 400
    assert too_far.json()["error"]["param"] == "message_count"
    assert missing.status_code == 404
    assert missing.json()["error"]["param"] == "session_id"
    assert noop.status_code == 200
    assert noop.json()["rolled_back"] is False
    assert noop.json()["previous_message_count"] == 2
    assert noop.json()["message_count"] == 2
    assert len(app.state.hipengine_chat_sessions["sess_rollback_errors"].messages) == 2


def test_chat_session_snapshot_export_restore_round_trips_visible_transcript() -> None:
    fake = SequentialFakeLLM(["stored answer", "after restore"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            backend="test-backend",
            quant="test-quant",
            eager_load=False,
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    unauthorized = client.get("/v1/hipengine/sessions/sess_snap/snapshot")
    unauthorized_restore = client.post("/v1/hipengine/sessions/sess_snap/snapshot", json={})
    created = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "snapshot prompt"}],
            "session": {"id": "sess_snap"},
            "max_tokens": 4,
        },
    )
    exported = client.get("/v1/hipengine/sessions/sess_snap/snapshot", headers=headers)
    deleted = client.delete("/v1/hipengine/sessions/sess_snap", headers=headers)
    restored = client.post(
        "/v1/hipengine/sessions/sess_snap/snapshot",
        headers=headers,
        json=exported.json(),
    )
    continued = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "session": {"id": "sess_snap", "commit": "append_none"},
            "max_tokens": 4,
        },
    )

    assert unauthorized.status_code == 401
    assert unauthorized_restore.status_code == 401
    assert created.status_code == 200
    assert exported.status_code == 200
    snapshot = exported.json()
    assert snapshot["object"] == "hipengine.session.snapshot"
    assert snapshot["schema"] == "hipengine.chat_session_snapshot.v1"
    assert snapshot["model"] == {"id": "fake-model", "backend": "test-backend", "quant": "test-quant"}
    assert snapshot["tokenizer"] == {
        "name": "SequentialFakeLLM",
        "tokenize": True,
        "detokenize": True,
        "count_tokens": True,
    }
    assert snapshot["resident_state_reuse"] is False
    assert snapshot["session"]["id"] == "sess_snap"
    assert snapshot["session"]["storage"] == "app_local_transcript"
    assert snapshot["session"]["includes_transcript"] is True
    assert snapshot["messages"] == [
        {"role": "user", "content": "snapshot prompt"},
        {"role": "assistant", "content": "stored answer"},
    ]
    assert deleted.json()["deleted"] is True
    assert restored.json() == {
        "object": "hipengine.session.restored",
        "id": "sess_snap",
        "restored": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "message_count": 2,
    }
    assert continued.status_code == 200
    prompt = fake.calls[1][0][0]
    assert prompt.index("snapshot prompt") < prompt.index("stored answer") < prompt.index("continue")


def test_chat_session_snapshot_restore_rejects_incompatible_model() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            backend="test-backend",
            quant="test-quant",
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
            "messages": [{"role": "user", "content": "snapshot prompt"}],
            "session": {"id": "sess_bad"},
            "max_tokens": 4,
        },
    )
    snapshot = client.get("/v1/hipengine/sessions/sess_bad/snapshot", headers=headers).json()
    snapshot["model"]["quant"] = "other-quant"
    client.delete("/v1/hipengine/sessions/sess_bad", headers=headers)

    restored = client.post(
        "/v1/hipengine/sessions/sess_bad/snapshot",
        headers=headers,
        json=snapshot,
    )

    assert created.status_code == 200
    assert restored.status_code == 400
    assert restored.json()["error"]["code"] == "invalid_request"
    assert restored.json()["error"]["param"] == "model.quant"
    assert "sess_bad" not in app.state.hipengine_chat_sessions


def test_chat_session_snapshot_restore_rejects_incompatible_tokenizer() -> None:
    fake = SequentialFakeLLM(["stored answer"])
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
            "messages": [{"role": "user", "content": "snapshot prompt"}],
            "session": {"id": "sess_bad_tokenizer"},
            "max_tokens": 4,
        },
    )
    snapshot = client.get(
        "/v1/hipengine/sessions/sess_bad_tokenizer/snapshot",
        headers=headers,
    ).json()
    snapshot["tokenizer"]["name"] = "OtherTokenizer"
    client.delete("/v1/hipengine/sessions/sess_bad_tokenizer", headers=headers)

    restored = client.post(
        "/v1/hipengine/sessions/sess_bad_tokenizer/snapshot",
        headers=headers,
        json=snapshot,
    )

    assert created.status_code == 200
    assert restored.status_code == 400
    assert restored.json()["error"]["code"] == "invalid_request"
    assert restored.json()["error"]["param"] == "tokenizer.name"
    assert "sess_bad_tokenizer" not in app.state.hipengine_chat_sessions


@pytest.mark.parametrize(
    ("corruption", "param"),
    [
        ("object", "object"),
        ("resident_state_reuse", "resident_state_reuse"),
        ("session_resident_state_reuse", "session.resident_state_reuse"),
        ("session_includes_transcript", "session.includes_transcript"),
    ],
)
def test_chat_session_snapshot_restore_rejects_corrupted_envelope(
    corruption: str,
    param: str,
) -> None:
    fake = SequentialFakeLLM(["stored answer"])
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
            "messages": [{"role": "user", "content": "snapshot prompt"}],
            "session": {"id": "sess_corrupt_envelope"},
            "max_tokens": 4,
        },
    )
    snapshot = client.get(
        "/v1/hipengine/sessions/sess_corrupt_envelope/snapshot",
        headers=headers,
    ).json()
    if corruption == "object":
        snapshot["object"] = "hipengine.session.other"
    elif corruption == "resident_state_reuse":
        snapshot["resident_state_reuse"] = True
    elif corruption == "session_resident_state_reuse":
        snapshot["session"]["resident_state_reuse"] = True
    elif corruption == "session_includes_transcript":
        snapshot["session"]["includes_transcript"] = False
    else:
        raise AssertionError(f"unhandled corruption case: {corruption}")
    client.delete("/v1/hipengine/sessions/sess_corrupt_envelope", headers=headers)

    restored = client.post(
        "/v1/hipengine/sessions/sess_corrupt_envelope/snapshot",
        headers=headers,
        json=snapshot,
    )

    assert created.status_code == 200
    assert restored.status_code == 400
    assert restored.json()["error"]["code"] == "invalid_request"
    assert restored.json()["error"]["param"] == param
    assert "sess_corrupt_envelope" not in app.state.hipengine_chat_sessions


def test_chat_session_snapshot_restore_rejects_corrupted_message_shape() -> None:
    fake = SequentialFakeLLM(["stored answer"])
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
            "messages": [{"role": "user", "content": "snapshot prompt"}],
            "session": {"id": "sess_corrupt"},
            "max_tokens": 4,
        },
    )
    snapshot = client.get("/v1/hipengine/sessions/sess_corrupt/snapshot", headers=headers).json()
    snapshot["messages"][0]["unexpected"] = True
    client.delete("/v1/hipengine/sessions/sess_corrupt", headers=headers)

    restored = client.post(
        "/v1/hipengine/sessions/sess_corrupt/snapshot",
        headers=headers,
        json=snapshot,
    )

    assert created.status_code == 200
    assert restored.status_code == 400
    assert restored.json()["error"]["code"] == "invalid_request"
    assert restored.json()["error"]["param"] == "messages[0].unexpected"
    assert "sess_corrupt" not in app.state.hipengine_chat_sessions


@pytest.mark.parametrize(
    ("corruption", "param"),
    [
        ("content_object", "messages[0].content"),
        ("content_part_non_object", "messages[0].content[0]"),
        ("content_part_wrong_type", "messages[0].content[0]"),
        ("content_part_non_string_text", "messages[0].content[0].text"),
        ("unsupported_role", "messages[0].role"),
        ("user_tool_calls", "messages[0].tool_calls"),
        ("assistant_tool_call_id", "messages[1].tool_call_id"),
        ("tool_missing_tool_call_id", "messages[2].tool_call_id"),
        ("tool_with_tool_calls", "messages[2].tool_calls"),
        ("tool_unknown_tool_call_id", "messages[2].tool_call_id"),
        ("duplicate_tool_result", "messages[3].tool_call_id"),
        ("duplicate_tool_call_id", "messages[1].tool_calls[1].id"),
        ("tool_result_skipped_by_user", "messages[2].role"),
        ("empty_name", "messages[0].name"),
        ("non_string_tool_call_id", "messages[0].tool_call_id"),
    ],
)
def test_chat_session_snapshot_restore_rejects_corrupted_message_fields(
    corruption: str,
    param: str,
) -> None:
    fake = SequentialFakeLLM(["stored answer"])
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
            "messages": [
                {"role": "user", "content": "snapshot prompt"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_snap",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_snap", "content": "snapshot tool result"},
            ],
            "session": {"id": "sess_corrupt_fields"},
            "max_tokens": 4,
        },
    )
    snapshot = client.get("/v1/hipengine/sessions/sess_corrupt_fields/snapshot", headers=headers).json()

    if corruption == "content_object":
        snapshot["messages"][0]["content"] = {"text": "not an accepted content shape"}
    elif corruption == "content_part_non_object":
        snapshot["messages"][0]["content"] = [7]
    elif corruption == "content_part_wrong_type":
        snapshot["messages"][0]["content"] = [{"type": "image_url", "text": "not text"}]
    elif corruption == "content_part_non_string_text":
        snapshot["messages"][0]["content"] = [{"type": "text", "text": 7}]
    elif corruption == "unsupported_role":
        snapshot["messages"][0]["role"] = "critic"
    elif corruption == "user_tool_calls":
        snapshot["messages"][0]["tool_calls"] = snapshot["messages"][1]["tool_calls"]
    elif corruption == "assistant_tool_call_id":
        snapshot["messages"][1]["tool_call_id"] = "call_snap"
    elif corruption == "tool_missing_tool_call_id":
        del snapshot["messages"][2]["tool_call_id"]
    elif corruption == "tool_with_tool_calls":
        snapshot["messages"][2]["tool_calls"] = snapshot["messages"][1]["tool_calls"]
    elif corruption == "tool_unknown_tool_call_id":
        snapshot["messages"][2]["tool_call_id"] = "call_missing"
    elif corruption == "duplicate_tool_result":
        snapshot["messages"].insert(
            3,
            {"role": "tool", "tool_call_id": "call_snap", "content": "duplicate tool result"},
        )
    elif corruption == "duplicate_tool_call_id":
        snapshot["messages"][1]["tool_calls"].append(
            {
                "id": "call_snap",
                "type": "function",
                "function": {"name": "write", "arguments": "{}"},
            }
        )
    elif corruption == "tool_result_skipped_by_user":
        snapshot["messages"].insert(2, {"role": "user", "content": "skip the tool"})
    elif corruption == "empty_name":
        snapshot["messages"][0]["name"] = ""
    elif corruption == "non_string_tool_call_id":
        snapshot["messages"][0]["tool_call_id"] = 7
    else:
        raise AssertionError(f"unhandled corruption case: {corruption}")
    client.delete("/v1/hipengine/sessions/sess_corrupt_fields", headers=headers)

    restored = client.post(
        "/v1/hipengine/sessions/sess_corrupt_fields/snapshot",
        headers=headers,
        json=snapshot,
    )

    assert created.status_code == 200
    assert restored.status_code == 400
    assert restored.json()["error"]["code"] == "invalid_request"
    assert restored.json()["error"]["param"] == param
    assert "sess_corrupt_fields" not in app.state.hipengine_chat_sessions


def test_chat_session_snapshot_restore_rejects_new_session_when_cap_full() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            api_key="secret",
            max_chat_sessions=1,
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
            "messages": [{"role": "user", "content": "snapshot prompt"}],
            "session": {"id": "sess_one"},
            "max_tokens": 4,
        },
    )
    snapshot = client.get("/v1/hipengine/sessions/sess_one/snapshot", headers=headers).json()
    snapshot["session"]["id"] = "sess_two"

    restored = client.post(
        "/v1/hipengine/sessions/sess_two/snapshot",
        headers=headers,
        json=snapshot,
    )

    assert created.status_code == 200
    assert restored.status_code == 429
    assert restored.headers["Retry-After"] == "1"
    assert restored.json()["error"]["code"] == "engine_busy"
    assert restored.json()["error"]["message"] == "chat session limit is full"
    assert restored.json()["error"]["hipengine"] == {
        "code": "engine_busy",
        "status_code": 429,
        "retryable": True,
        "routing": {
            **_routing_metadata(),
            "matched": True,
            "reason": "engine_busy",
            "overload_source": "chat_session_cap",
            "max_active_chat_sessions": 1,
        },
    }
    assert set(app.state.hipengine_chat_sessions) == {"sess_one"}
    assert app.state.hipengine_server_metrics.request_rejected_total == 1


def test_chat_session_cap_rejects_new_sessions_before_generation() -> None:
    fake = SequentialFakeLLM(["first answer", "existing answer", "after delete answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_chat_sessions=1,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "first"}],
            "session": {"id": "sess_one"},
            "max_tokens": 4,
        },
    )
    rejected = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "second"}],
            "session": {"id": "sess_two"},
            "max_tokens": 4,
        },
    )

    assert first.status_code == 200
    assert rejected.status_code == 429
    assert rejected.headers["retry-after"] == "1"
    assert rejected.json()["error"]["code"] == "engine_busy"
    assert rejected.json()["error"]["message"] == "chat session limit is full"
    assert rejected.json()["error"]["hipengine"] == {
        "code": "engine_busy",
        "status_code": 429,
        "retryable": True,
        "routing": {
            **_routing_metadata(),
            "matched": True,
            "reason": "engine_busy",
            "overload_source": "chat_session_cap",
            "max_active_chat_sessions": 1,
        },
    }
    assert "sess_two" not in app.state.hipengine_chat_sessions
    assert app.state.hipengine_server_metrics.request_rejected_total == 1
    assert len(fake.calls) == 1

    existing = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "again"}],
            "session": {"id": "sess_one"},
            "max_tokens": 4,
        },
    )
    assert existing.status_code == 200
    assert len(fake.calls) == 2

    deleted = client.delete("/v1/hipengine/sessions/sess_one")
    assert deleted.json()["deleted"] is True

    after_delete = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "after delete"}],
            "session": {"id": "sess_two"},
            "max_tokens": 4,
        },
    )
    assert after_delete.status_code == 200
    assert len(fake.calls) == 3
    ready = client.get("/ready").json()
    assert ready["sessions"]["active"] == 1
    assert ready["sessions"]["max_active"] == 1
    assert ready["sessions"]["pending_creations"] == 0


def test_chat_session_can_use_explicit_mtp_without_losing_transcript() -> None:
    fake = SpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="opt_in",
        ),
        llm=fake,
    )
    client = TestClient(app)
    for content in ("remember alpha", "now beta"):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": content}],
                "session": {"id": "mtp_session", "commit": "append_all"},
                "max_tokens": 3,
                "temperature": 0.0,
                "speculative_mtp": True,
            },
        )
        assert response.status_code == 200
        assert response.json()["hipengine"]["speculative_mtp"]["used"] is True

    assert len(fake.mtp_calls) == 2
    assert "remember alpha" in fake.mtp_calls[0][0][0]
    assert "remember alpha" in fake.mtp_calls[1][0][0]
    assert "now beta" in fake.mtp_calls[1][0][0]
    sessions = client.get("/v1/hipengine/sessions").json()
    stored = next(row for row in sessions["sessions"] if row["id"] == "mtp_session")
    assert stored["message_count"] == 4


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        (
            "/v1/completions",
            {"model": "fake-model", "prompt": "hello", "max_tokens": 1},
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
            },
        ),
    ],
)
def test_session_append_none_reports_cache_action(endpoint, payload) -> None:
    fake = FakeLLM(outputs=["reply"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(endpoint, json={**payload, "session": {"commit": "append_none"}})

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_details"]["cache_action"] == "append_none"


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        (
            "/v1/completions",
            {"model": "fake-model", "prompt": "hello", "max_tokens": 1},
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
            },
        ),
    ],
)
def test_stateless_session_default_reports_append_none_cache_action(endpoint, payload) -> None:
    fake = FakeLLM(outputs=["reply"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(endpoint, json=payload)

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_details"]["cache_action"] == "append_none"


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        (
            "/v1/completions",
            {"model": "fake-model", "prompt": "hello", "max_tokens": 1},
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
            },
        ),
    ],
)
def test_streaming_session_append_none_reports_cache_action(endpoint, payload) -> None:
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=["reply"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        endpoint,
        json={**payload, "stream": True, "session": {"commit": "append_none"}},
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    done = next(item for item in payloads if item["choices"][0]["finish_reason"] == "stop")
    assert done["choices"][0]["finish_details"]["cache_action"] == "append_none"


def test_chat_session_visible_only_prepends_stored_transcript_and_commits_visible_answer() -> None:
    fake = SequentialFakeLLM(["first answer", "second answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "session": {"id": "sess_visible", "commit": "append_visible_only"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "again"}],
            "max_tokens": 2,
            "session": {"id": "sess_visible", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["choices"][0]["finish_details"]["cache_action"] == "append_visible_only"
    assert second.json()["choices"][0]["finish_details"]["cache_action"] == "append_none"
    prompt = fake.calls[1][0][0]
    assert prompt.index("hello") < prompt.index("first answer") < prompt.index("again")
    record = app.state.hipengine_chat_sessions["sess_visible"]
    assert record.messages == (
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "first answer"},
    )


def test_chat_session_visible_only_strips_hidden_reasoning_from_next_prompt() -> None:
    fake = SequentialFakeLLM(["<think>secret plan</think>visible answer", "done"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "max_tokens": 2,
            "session": {"id": "sess_reasoning"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 2,
            "session": {"id": "sess_reasoning", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    choice = first.json()["choices"][0]
    assert choice["message"] == {
        "role": "assistant",
        "content": "visible answer",
        "reasoning_content": "secret plan",
    }
    assert choice["finish_details"]["cache_action"] == "append_visible_only"
    prompt = fake.calls[1][0][0]
    assert "visible answer" in prompt
    assert "secret plan" not in prompt
    assert "<think>secret plan</think>" not in prompt


def test_chat_session_append_all_retains_raw_generated_text_for_debug() -> None:
    fake = SequentialFakeLLM(["<think>secret plan</think>visible answer", "done"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "debug"}],
            "max_tokens": 2,
            "session": {"id": "sess_all", "commit": "append_all"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 2,
            "session": {"id": "sess_all", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["choices"][0]["finish_details"]["cache_action"] == "append_all"
    assert "<think>secret plan</think>visible answer" in fake.calls[1][0][0]


def test_chat_session_visible_only_downgrades_length_finish_to_prompt_only() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(text="partial answer", finish_details=FinishDetails(reason="length")),
            "done",
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "start"}],
            "max_tokens": 1,
            "session": {"id": "sess_length", "commit": "append_visible_only"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "next"}],
            "max_tokens": 2,
            "session": {"id": "sess_length", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert first_choice["finish_details"]["cache_action"] == "append_prompt_only"
    assert first_choice["finish_details"]["continuation_eligible"] is True
    assert first_choice["finish_details"]["continuation_id"] == continuation_id
    prompt = fake.calls[1][0][0]
    assert "start" in prompt
    assert "next" in prompt
    assert "partial answer" not in prompt
    record = app.state.hipengine_chat_sessions["sess_length"]
    assert record.messages == ({"role": "user", "content": "start"},)


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_finish_details", "request_extra"),
    [
        (
            "deadline",
            408,
            {
                "reason": "deadline_exceeded",
                "deadline_exceeded": True,
                "cache_action": "append_prompt_only",
            },
            {"timeout_ms": 5000},
        ),
        (
            "cancelled",
            499,
            {"reason": "cancelled", "cancelled": True, "cache_action": "append_prompt_only"},
            {},
        ),
    ],
)
def test_chat_session_visible_only_downgrades_backend_errors_to_prompt_only(
    error: str,
    expected_status: int,
    expected_finish_details: dict[str, Any],
    request_extra: dict[str, Any],
) -> None:
    fake = BackendErrorThenSequentialFakeLLM(error, ["done"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    session_id = f"sess_backend_error_{error}"

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "unsafe turn"}],
            "max_tokens": 4,
            "session": {"id": session_id, "commit": "append_visible_only"},
            **request_extra,
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "follow up"}],
            "max_tokens": 4,
            "session": {"id": session_id, "commit": "append_none"},
        },
    )

    assert first.status_code == expected_status
    assert first.json()["error"]["finish_details"] == expected_finish_details
    assert second.status_code == 200
    prompt = fake.calls[1][0][0]
    assert "unsafe turn" in prompt
    assert "follow up" in prompt
    assert "done" not in prompt
    record = app.state.hipengine_chat_sessions[session_id]
    assert record.messages == ({"role": "user", "content": "unsafe turn"},)


@pytest.mark.parametrize(
    "response_format_mode",
    ["json_object", "json_schema"],
)
def test_chat_session_visible_only_downgrades_structured_output_failures_to_prompt_only(
    response_format_mode: str,
) -> None:
    response_format = (
        {"type": "json_object"}
        if response_format_mode == "json_object"
        else {"type": "json_schema", "json_schema": _response_json_schema()}
    )
    fake = SequentialFakeLLM(["not json", "done"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": response_format,
            "max_tokens": 4,
            "session": {"id": "sess_structured_failure", "commit": "append_visible_only"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 4,
            "session": {"id": "sess_structured_failure", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_choice = first.json()["choices"][0]
    assert first_choice["message"] == {"role": "assistant", "content": ""}
    assert first_choice["finish_details"]["reason"] == "schema_violation"
    assert first_choice["finish_details"]["cache_action"] == "append_prompt_only"
    prompt = fake.calls[1][0][0]
    assert "return json" in prompt
    assert "continue" in prompt
    assert "not json" not in prompt
    record = app.state.hipengine_chat_sessions["sess_structured_failure"]
    assert record.messages == ({"role": "user", "content": "return json"},)


def test_chat_session_visible_only_downgrades_synthetic_output_to_prompt_only() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="synthetic repair",
                finish_details=FinishDetails(reason="stop", synthetic_tokens=2),
            ),
            "done",
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "repair"}],
            "max_tokens": 4,
            "session": {"id": "sess_synthetic", "commit": "append_visible_only"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 4,
            "session": {"id": "sess_synthetic", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_choice = first.json()["choices"][0]
    assert first_choice["message"] == {"role": "assistant", "content": "synthetic repair"}
    assert first_choice["finish_details"]["reason"] == "stop"
    assert first_choice["finish_details"]["synthetic_tokens"] == 2
    assert first_choice["finish_details"]["cache_action"] == "append_prompt_only"
    prompt = fake.calls[1][0][0]
    assert "repair" in prompt
    assert "continue" in prompt
    assert "synthetic repair" not in prompt
    record = app.state.hipengine_chat_sessions["sess_synthetic"]
    assert record.messages == ({"role": "user", "content": "repair"},)
