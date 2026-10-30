"""Shared builders, assertions and constants for the server API integration tests.

Extracted from ``tests/test_integration_server_api.py`` by RF-12.
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


AR_BATCH_ENV = "HIPENGINE_QWEN4_EXP_AR_BATCH"


REPO_ROOT = Path(__file__).resolve().parents[2]


def _test_static_eligibility() -> SpeculativeMTPStaticEligibility:
    return SpeculativeMTPStaticEligibility(
        state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
        reason="qualified_test_singleton",
        max_candidate_count=3,
        max_realized_group_rows=1,
        automatic_eligible=True,
        strict_fallback_key="gguf_target_ar",
        evidence_key="test-singleton",
        evidence_fingerprint="sha256:test-singleton",
    )


def _api_error_taxonomy_table() -> dict[str, dict[str, Any]]:
    lines = (REPO_ROOT / "docs" / "API.md").read_text(encoding="utf-8").splitlines()
    start = lines.index("| Code | Status | Retry | Current emission |") + 2
    table: dict[str, dict[str, Any]] = {}
    for line in lines[start:]:
        if not line.startswith("| `"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        code = cells[0].strip("`")
        table[code] = {
            "status_code": int(cells[1]),
            "retryable": cells[2] == "yes",
            "current_emission": cells[3],
        }
    return table


async def _asgi_chat_post(
    app: Any,
    payload: Mapping[str, Any],
    *,
    on_body: Any = None,
) -> int:
    """POST one chat request straight into the ASGI app, returning its status.

    ``TestClient`` collects the response body before handing it back, so it cannot
    show whether a stream kept moving while other work was in flight. Calling the
    app directly reports every body message as the app produces it.
    """

    body = json.dumps(payload).encode()
    sent = {"request": False}

    async def receive() -> dict[str, Any]:
        if not sent["request"]:
            sent["request"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(0.01)
        return {"type": "http.request", "body": b"", "more_body": False}

    status = {"code": 0}

    async def send(message: Mapping[str, Any]) -> None:
        if message["type"] == "http.response.start":
            status["code"] = int(message["status"])
        elif on_body is not None:
            on_body(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    await app(scope, receive, send)
    return status["code"]


def _stateless_finish_details(reason: str, **extra: Any) -> dict[str, Any]:
    return {"reason": reason, **extra, "cache_action": "append_none"}


def _assert_openai_tool_call_shape(
    tool_call: Mapping[str, Any],
    *,
    name: str,
    arguments: Mapping[str, Any],
) -> None:
    assert set(tool_call) == {"id", "type", "function"}
    assert isinstance(tool_call["id"], str)
    assert tool_call["id"].startswith("call_")
    assert tool_call["type"] == "function"
    function = tool_call["function"]
    assert isinstance(function, Mapping)
    assert set(function) == {"name", "arguments"}
    assert function["name"] == name
    assert isinstance(function["arguments"], str)
    assert json.loads(function["arguments"]) == dict(arguments)


def _assert_openai_stream_tool_call_delta_shape(
    choice: Mapping[str, Any],
    *,
    name: str,
    arguments: Mapping[str, Any],
    index: int = 0,
    tool_index: int = 0,
) -> None:
    assert choice["index"] == index
    assert choice["finish_reason"] is None
    assert set(choice["delta"]) == {"tool_calls"}
    tool_calls = choice["delta"]["tool_calls"]
    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert set(tool_call) == {"index", "id", "type", "function"}
    assert tool_call["index"] == tool_index
    _assert_openai_tool_call_shape(
        {
            "id": tool_call["id"],
            "type": tool_call["type"],
            "function": tool_call["function"],
        },
        name=name,
        arguments=arguments,
    )


def _routing_metadata(
    *,
    requested_model: str = "fake-model",
    served_model: str = "fake-model",
    loaded_model_count: int = 1,
) -> dict[str, Any]:
    return {
        "requested_model": requested_model,
        "served_model": served_model,
        "fallback_used": False,
        "policy": "single_model_exact",
        "loaded_model_count": loaded_model_count,
        "multiple_models": False,
    }


def _fake_kv_pool_stats() -> SimpleNamespace:
    return SimpleNamespace(
        current_bytes=4096,
        high_water_observed_bytes=8192,
        current_pages=6,
        high_water_observed_pages=8,
        grow_events=2,
        grow_failures=1,
        shrink_events=3,
        free_pages=4,
        refcounted_pages=5,
        pinned_pages=1,
        retired_pages=2,
        retired_bytes=8192,
    )


def _fake_kv_pool_metadata() -> dict[str, float]:
    return {
        "current_bytes": 4096.0,
        "high_water_observed_bytes": 8192.0,
        "current_pages": 6.0,
        "high_water_observed_pages": 8.0,
        "grow_events": 2.0,
        "grow_failures": 1.0,
        "shrink_events": 3.0,
        "free_pages": 4.0,
        "refcounted_pages": 5.0,
        "pinned_pages": 1.0,
        "retired_pages": 2.0,
        "retired_bytes": 8192.0,
    }


def _continuation_capability() -> dict[str, Any]:
    return {
        "supported": True,
        "stateful": False,
        "resident_state_reuse": False,
        "single_use": True,
        "ttl_seconds": 900,
        "scoped_to": ["served_model", "endpoint", "tokenizer", "auth_principal", "session_id"],
        "supported_endpoints": ["completions", "chat_completions"],
        "supported_finishes": ["length"],
        "supported_streaming": False,
        "supported_sampling": "deterministic_buffered_only",
        "ineligible_when": [
            "non_length_finish",
            "length_phase_not_answer_or_structured",
            "stream",
            "n_not_1",
            "logprobs",
            "completion_echo",
            "non_deterministic_sampling",
            "logit_processors",
            "stop",
            "chat_tools",
            "thinking_budget_controls",
            "session_id_without_commit",
        ],
        "unsupported_resume_fields": [
            "prompt",
            "messages",
            "stream",
            "n",
            "logprobs",
            "echo",
            "temperature",
            "logit_bias",
            "suppress_token_ids",
            "min_tokens",
            "ignore_eos",
            "stop",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "guided_json",
            "guided_regex",
            "guided_choice",
            "guided_patch",
            "guided_diff",
            "reasoning_effort",
            "max_think_tokens",
            "min_answer_tokens",
            "hard_think_cap",
            "soft_close_window",
            "hard_close_message",
            "hard_close_sequence",
            "thinking_token_budget",
            "chat_template_kwargs",
            "thinking",
            "reasoning",
        ],
    }


def _session_commit_policy_capability() -> dict[str, Any]:
    return {
        "supported": True,
        "stateful": True,
        "resident_state_reuse": False,
        "storage": "app_local_transcript",
        "default": "append_none",
        "stateful_default": "append_visible_only",
        "modes": [
            "append_none",
            "append_prompt_only",
            "append_visible_only",
            "append_all",
        ],
        "supported_endpoints": ["chat_completions"],
        "supported_streaming": False,
        "resident_kv_commit": False,
        "visible_only_reprefill": False,
        "visible_only_replay": "rerender_app_local_transcript",
        "downgrade_visible_only_on": [
            "cancelled",
            "deadline_exceeded",
            "invalid_tool_call",
            "length",
            "schema_violation",
            "tool_required_not_satisfied",
        ],
        "context_overflow_policy": {
            "field": "session.context_overflow_policy",
            "default": "reject",
            "modes": ["reject", "auto_clear_transient", "new_session", "truncate_oldest_visible"],
            "aliases": {"fail": "reject"},
            "auto_clear_transient": {
                "scope": "transient_session_segments",
                "drops_committed_visible_turns": False,
                "drops_request_content": False,
                "current_transient_segment_count": 0,
                "metadata": ["clear_policy", "would_clear_transient", "transient_message_count"],
            },
            "new_session": {
                "scope": "app_local_chat_transcript_prefix",
                "drops_request_content": False,
                "requires_request_only_fit": True,
                "metadata": ["clear_policy", "would_reset_session", "would_drop", "kept_segments"],
            },
            "truncate_oldest_visible": {
                "scope": "app_local_chat_transcript_prefix",
                "drops_request_content": False,
                "requires_valid_rendered_suffix": True,
                "commit": "replace_stored_prefix_on_success",
                "metadata": ["clear_policy", "would_truncate", "would_drop", "kept_segments"],
            },
        },
    }


def _session_metadata_capability(max_active: int | None = None) -> dict[str, Any]:
    return {
        "supported": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "includes_transcript": False,
        "transcript_message_copy": "json_deep_copy",
        "max_active": max_active,
        "list_endpoint": "/v1/hipengine/sessions",
        "delete_endpoint": "/v1/hipengine/sessions/{session_id}",
        "fork_endpoint": "/v1/hipengine/sessions/{session_id}/fork",
        "fork_resident_state_reuse": False,
        "fork_deep_copies_transcript": True,
        "rollback_endpoint": "/v1/hipengine/sessions/{session_id}/rollback",
        "rollback_target": "message_count",
        "rollback_resident_state_reuse": False,
        "rollback_deep_copies_retained_transcript": True,
        "snapshot_schema": "hipengine.chat_session_snapshot.v1",
        "snapshot_export_endpoint": "/v1/hipengine/sessions/{session_id}/snapshot",
        "snapshot_restore_endpoint": "/v1/hipengine/sessions/{session_id}/snapshot",
        "snapshot_includes_transcript": True,
        "snapshot_resident_state_reuse": False,
        "snapshot_export_deep_copies_transcript": True,
        "snapshot_includes_tokenizer_metadata": True,
        "snapshot_tokenizer_validation": "when_model_loaded",
    }


def _parallelism_capability() -> dict[str, Any]:
    return {
        "tensor_parallel": {
            "enabled": False,
            "topology": {
                "mode": "single_process",
                "world_size": 1,
                "local_world_size": 1,
                "rank": 0,
                "local_rank": 0,
            },
            "collectives": {
                "available": False,
                "backend": None,
            },
            "unsupported_features": [
                "world_size_gt_1",
                "weight_sharding",
                "kv_cache_sharding",
                "collective_reduce_scatter_all_gather",
                "multi_gpu_graph_capture",
                "cross_rank_session_snapshots",
            ],
        },
    }


def _mtp_accounting_detail(
    *,
    generated_tokens: int,
    accounting: dict[str, Any],
    execution_path: str = "gguf_specdec2_mtp2",
    timing: Mapping[str, float] | None = None,
    finish_details: Any | None = None,
) -> GenerationOutput:
    """One backend output that carries a per-request execution block."""

    cycles = int(accounting.get("cycles") or 0)
    if timing is None and cycles > 0:
        timing = {
            "mtp_cycles_count": float(cycles),
            "mtp_generated_draft_tokens": float(
                accounting.get("generated_draft_tokens") or 0
            ),
            "mtp_accepted_draft_tokens": float(
                accounting.get("accepted_draft_tokens") or 0
            ),
        }
    return GenerationOutput(
        text="done",
        finish_details=finish_details,
        telemetry=GenerationTelemetry.from_decode_counts(
            prompt_tokens=8,
            generated_tokens=generated_tokens,
            row_index=0,
            execution_path=execution_path,
            timing=timing,
            timing_scope="choice" if timing is not None else None,
            timing_owner=True if timing is not None else None,
            diagnostics={"specdec2_mtp2": accounting},
        ),
    )


def _full_coverage_accounting(**overrides: Any) -> dict[str, Any]:
    accounting: dict[str, Any] = {
        "requested_budget": 3,
        "candidate_budget": 3,
        "prompt_streaming": False,
        "prompt_fallback_reason": None,
        "cycles": 8,
        "generated_draft_tokens": 24,
        "accepted_draft_tokens": 15,
        "selected_depth_histogram": {"3": 8},
        "mtp_output_tokens": 23,
        "ar_output_tokens_in_cycles": 0,
        "first_fallback_position": None,
        "ar_step_reason_counts": {},
        "recoverable_failures": 0,
        "failure_reason_counts": {},
        "k0_catchups": 0,
    }
    accounting.update(overrides)
    return accounting


def _response_json_schema() -> dict[str, Any]:
    return {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "path": {"type": "string", "minLength": 1},
            },
            "required": ["ok", "path"],
            "additionalProperties": False,
        },
    }


def _unified_diff_text() -> str:
    return "\n".join(
        [
            "diff --git a/README.md b/README.md",
            "index 1111111..2222222 100644",
            "--- a/README.md",
            "+++ b/README.md",
            "@@ -1,2 +1,2 @@",
            " hello",
            "-old",
            "+new",
        ]
    )


_QWEN35_XML_TOOL_CALL = (
    "<tool_call>\n"
    "<function=get_weather>\n"
    "<parameter=city>\n"
    "Tokyo\n"
    "</parameter>\n"
    "<parameter=days>\n"
    "3\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    },
}


def _incomplete_duplicate_read_tool_output() -> str:
    return (
        '<tool_call>{"name":"read","arguments":{<|im_end|>\n\n'
        '<|im_start|>assistant\n'
        '<tool_call>{"name":"read","arguments":{"path":"pyproject.toml",'
        '"mode":"summary"}}</tool_call>'
        '<|im_end|><|endoftext|>'
        '<|im_start|><|im_start|><|im_start|><|im_start|><|im_start|>'
    )


def _bounded_tool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {"const": "file"},
            "path": {"type": "string", "minLength": 1, "maxLength": 64},
            "mode": {"type": "string", "enum": ["raw", "summary"]},
            "tags": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {"type": "string", "minLength": 1, "maxLength": 16},
            },
            "filters": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string", "enum": ["ext", "name"]},
                        "value": {"type": "string", "minLength": 1},
                    },
                    "required": ["field", "value"],
                    "additionalProperties": False,
                },
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 3},
        },
        "required": ["kind", "path", "mode", "tags", "limit"],
        "additionalProperties": False,
    }


def _load_single_replay_artifact(replay_dir: Path) -> tuple[dict[str, Any], str]:
    artifacts = list(replay_dir.glob("*.json"))
    assert len(artifacts) == 1
    artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
    serialized = json.dumps(artifact, sort_keys=True, allow_nan=False)
    return artifact, serialized


def _metric_value(text: str, name: str) -> float:
    prefix = f"{name} "
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line.removeprefix(prefix))
    raise AssertionError(f"metric {name} not found in:\n{text}")


def _labeled_metric_value(text: str, name: str, **labels: str) -> float:
    encoded_labels = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
    prefix = f"{name}{{{encoded_labels}}} "
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line.removeprefix(prefix))
    raise AssertionError(f"metric {name} with labels {labels} not found in:\n{text}")


def _sse_payloads(text: str) -> list[dict]:
    payloads = []
    for line in text.splitlines():
        if line == "data: [DONE]" or not line.startswith("data: "):
            continue
        payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads
