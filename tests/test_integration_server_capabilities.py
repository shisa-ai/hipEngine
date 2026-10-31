"""OpenAI server integration tests: capabilities.

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
    SpeculativeMTPFakeLLM,
)
from tests.support.server_api_support import (
    AR_BATCH_ENV,
    REPO_ROOT,
    _api_error_taxonomy_table,
    _continuation_capability,
    _parallelism_capability,
    _session_commit_policy_capability,
    _session_metadata_capability,
)


def test_tensor_parallel_design_doc_matches_capability_manifest() -> None:
    doc = (REPO_ROOT / "docs" / "reference" / "TENSOR_PARALLEL.md").read_text(encoding="utf-8")
    tensor_parallel = _parallelism_capability()["tensor_parallel"]

    assert "`parallelism.tensor_parallel.enabled=false`" in doc
    assert '`topology.mode="single_process"`' in doc
    assert "`world_size=1`" in doc
    assert "Rank 0 owns sampling." in doc
    assert "Phase 1 replicates KV cache on every rank." in doc
    assert "No TP code may affect the default single-GPU path without hardware validation" in doc
    for feature_id in tensor_parallel["unsupported_features"]:
        assert f"`{feature_id}`" in doc


def test_coerce_generation_output_preserves_telemetry() -> None:
    raw = SimpleNamespace(
        text="answer",
        telemetry=GenerationTelemetry.from_decode_counts(
            prompt_tokens=3,
            generated_tokens=2,
            sampler_mode="host_logits_sample",
        ),
    )

    output = _coerce_generation_output(raw)

    assert output.text == "answer"
    assert output.telemetry is not None
    assert output.telemetry.to_json_dict()["decode_state"] == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 3,
        "generated_tokens": 2,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_mode": "host_logits_sample",
    }


def test_models_endpoint_reports_served_model_name_and_auth() -> None:
    fake = FakeLLM()
    app = create_app(
        ServerConfig(model="/models/fake", served_model_name="fake-model", api_key="secret"),
        llm=fake,
    )
    client = TestClient(app)

    unauthorized = client.get("/v1/models")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["type"] == "authentication_error"

    response = client.get("/v1/models", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    model = body["data"][0]
    assert model["id"] == "fake-model"
    assert model["object"] == "model"
    assert model["created"] == app.state.hipengine_config.created
    assert model["owned_by"] == "hipengine"
    assert model["hipengine"] == {
        "path": "/models/fake",
        "backend": "auto",
        "quant": "auto",
        "execution_profile": {
            "requested": None,
            "resolved": None,
            "manifest_sha256": None,
            "strict_manifest_sha256": None,
            "fell_back_to_strict": None,
            "migration_default_preserved": True,
        },
        "loaded": True,
        "resident_context": True,
        "context": {
            "configured_max_context_tokens": None,
            "effective_max_context_tokens": None,
            "chat_default_max_tokens": 4096,
        },
        "kv_capacity": {
            "storage": "auto",
            "scale_dtype": "fp16",
            "scale_granularity": "per_token_head",
            "estimate": None,
            "capability": {
                "schema_version": 1,
                "status": "unavailable",
                "runtime_action": "not_applicable",
                "promotion_eligible": False,
                "diagnostic_override": False,
                "requested": None,
                "effective_kv_storage": None,
                "artifact": None,
                "evidence": None,
                "reason": (
                    "loaded engine does not expose artifact-scoped KV capability"
                    " provenance"
                ),
            },
        },
        "capabilities": {
            "surfaces": ["text"],
            "completions": True,
            "chat_completions": True,
            "streaming": True,
            "tools": True,
            "reasoning_controls": True,
            "structured_outputs": True,
            "continuations": True,
            "sessions": True,
            "grammars": False,
            "song_generation": False,
            "speculative_mtp": False,
            "speculative": False,
            "tensor_parallel": False,
            "multiple_models": False,
        },
        "song_generation": {
            "enabled": False,
            "reason": "the served model declares no song surface",
            "surfaces": ["text"],
        },
        "capabilities_url": "/v1/hipengine/capabilities",
        "routing": {"loaded_model_count": 1, "multiple_models": False},
    }


def test_models_endpoint_reports_lazy_model_not_loaded() -> None:
    app = create_app(
        ServerConfig(model="/models/fake", served_model_name="fake-model", eager_load=False),
        llm=None,
    )
    client = TestClient(app)

    response = client.get("/v1/models")

    assert response.status_code == 200
    model = response.json()["data"][0]
    assert model["id"] == "fake-model"
    assert model["hipengine"]["loaded"] is False
    assert model["hipengine"]["capabilities"]["chat_completions"] is True
    assert model["hipengine"]["capabilities"]["tools"] is True
    assert model["hipengine"]["capabilities"]["multiple_models"] is False
    assert model["hipengine"]["routing"] == {"loaded_model_count": 0, "multiple_models": False}
    assert model["hipengine"]["kv_capacity"]["estimate"] is None


def test_models_endpoint_reports_resolved_auto_backend_and_quant() -> None:
    fake = FakeLLM()
    fake._resolved_backend = "hip_gfx1151"
    fake._resolved_quant = "gguf_q4_k_m"
    fake.resolved_execution_profile = "production"
    fake.execution_profile_manifest_sha256 = "a" * 64
    fake.execution_profile_strict_manifest_sha256 = "b" * 64
    fake.execution_profile_fell_back_to_strict = False
    app = create_app(
        ServerConfig(
            model="/models/fake.gguf",
            served_model_name="fake-model",
            execution_profile="production",
            eager_load=False,
        ),
        llm=fake,
    )
    client = TestClient(app)

    model = client.get("/v1/models").json()["data"][0]["hipengine"]

    assert model["backend"] == "hip_gfx1151"
    assert model["quant"] == "gguf_q4_k_m"
    assert model["execution_profile"] == {
        "requested": "production",
        "resolved": "production",
        "manifest_sha256": "a" * 64,
        "strict_manifest_sha256": "b" * 64,
        "fell_back_to_strict": False,
        "migration_default_preserved": False,
    }


def test_decode_telemetry_endpoint_reports_physical_state_and_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(AR_BATCH_ENV, raising=False)
    app = create_app(
        ServerConfig(model="/models/fake", served_model_name="fake-model", api_key="secret", eager_load=False),
        llm=FakeLLM(),
    )
    client = TestClient(app)
    assert client.get("/v1/hipengine/decode").status_code == 401
    unavailable = client.get(
        "/v1/hipengine/decode", headers={"Authorization": "Bearer secret"}
    )
    assert unavailable.status_code == 200
    assert unavailable.json() == {
        "object": "hipengine.decode_telemetry",
        "available": False,
        "reason": "engine_has_no_decode_telemetry",
    }

    # The serving engine reaches the hook through its generator, so both the
    # ``_text_generator`` hop and a direct engine hook must be probed.
    class ServiceLLM(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self._text_generator = SimpleNamespace(
                decode_telemetry=lambda: {
                    "decode_path": "qwen4exp_full_model_ar_batch",
                    "ar_batch": {"state_slots": 2, "steps": 3, "step_widths": {"2": 3}},
                }
            )

    service_app = create_app(
        ServerConfig(model="/models/fake", served_model_name="fake-model", api_key="secret", eager_load=False),
        llm=ServiceLLM(),
    )
    body = TestClient(service_app).get(
        "/v1/hipengine/decode", headers={"Authorization": "Bearer secret"}
    ).json()
    assert body["available"] is True
    assert body["decode"]["ar_batch"]["step_widths"] == {"2": 3}

    # Capabilities must describe the same route the telemetry endpoint reports,
    # not a second opinion about it.
    advertised = TestClient(service_app).get(
        "/v1/hipengine/capabilities", headers={"Authorization": "Bearer secret"}
    ).json()["decode"]
    assert advertised["available"] is True
    assert advertised["telemetry_url"] == "/v1/hipengine/decode"
    assert advertised["path"] == body["decode"]["decode_path"]
    assert advertised["ar_batch"]["active"] == body["decode"]["ar_batch"]
    assert advertised["ar_batch"]["mtp_rows_join_batch"] is False
    assert advertised["ar_batch"]["enabled"] is True
    assert "rollback_env" not in advertised["ar_batch"]
    assert advertised["ar_batch"]["max_rows"] == QWEN4_EXP_MAX_BATCH_ROWS

    # An engine without the hook advertises no path and no envelope.
    advertised_unavailable = TestClient(app).get(
        "/v1/hipengine/capabilities", headers={"Authorization": "Bearer secret"}
    ).json()["decode"]
    assert advertised_unavailable == {
        "telemetry_url": "/v1/hipengine/decode",
        "available": False,
        "reason": "engine_has_no_decode_telemetry",
        "ar_batch": None,
    }


def test_decode_telemetry_endpoint_sanitizes_hook_failures_and_bad_payloads() -> None:
    class FailingLLM(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self._text_generator = SimpleNamespace(
                decode_telemetry=lambda: (_ for _ in ()).throw(
                    RuntimeError("slot 3 at 0x7fdeadbeef leaked")
                )
            )

    failing_app = create_app(
        ServerConfig(model="/models/fake", served_model_name="fake-model", api_key="secret", eager_load=False),
        llm=FailingLLM(),
    )
    failed = TestClient(failing_app).get(
        "/v1/hipengine/decode", headers={"Authorization": "Bearer secret"}
    ).json()
    assert failed == {
        "object": "hipengine.decode_telemetry",
        "available": False,
        "reason": "decode_telemetry_unavailable",
    }
    assert "0x7fdeadbeef" not in str(failed)

    class NonMappingLLM(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self._text_generator = SimpleNamespace(decode_telemetry=lambda: ["nope"])

    bad_app = create_app(
        ServerConfig(model="/models/fake", served_model_name="fake-model", api_key="secret", eager_load=False),
        llm=NonMappingLLM(),
    )
    bad = TestClient(bad_app).get(
        "/v1/hipengine/decode", headers={"Authorization": "Bearer secret"}
    ).json()
    assert bad == {
        "object": "hipengine.decode_telemetry",
        "available": False,
        "reason": "decode_telemetry_malformed",
    }


def test_capabilities_endpoint_reports_manifest_and_auth(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", raising=False)
    # The MTP context window is an environment resolution, so pin it: a value
    # exported in the developer's shell must not change this expectation.
    monkeypatch.delenv("HIPENGINE_MTP2_MAX_CONTEXT_TOKENS", raising=False)
    fake = FakeLLM()
    app = create_app(
        ServerConfig(
            model="/models/fake",
            served_model_name="fake-model",
            api_key="secret",
            eager_load=False,
            max_context_tokens=2048,
            request_timeout_ms=250.0,
            max_queued_requests=5,
            max_active_requests=2,
            max_chat_sessions=3,
        ),
        llm=fake,
    )
    client = TestClient(app)

    unauthorized = client.get("/v1/hipengine/capabilities")
    assert unauthorized.status_code == 401

    response = client.get("/v1/hipengine/capabilities", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "hipengine.capabilities"
    assert body["model"] == {
        "id": "fake-model",
        "path": "/models/fake",
        "backend": "auto",
        "quant": "auto",
        "execution_profile": {
            "requested": None,
            "resolved": None,
            "manifest_sha256": None,
            "strict_manifest_sha256": None,
            "fell_back_to_strict": None,
            "migration_default_preserved": True,
        },
        "kv_capability": {
            "schema_version": 1,
            "status": "unavailable",
            "runtime_action": "not_applicable",
            "promotion_eligible": False,
            "diagnostic_override": False,
            "requested": None,
            "effective_kv_storage": None,
            "artifact": None,
            "evidence": None,
            "reason": (
                "loaded engine does not expose artifact-scoped KV capability"
                " provenance"
            ),
        },
    }
    assert body["context"] == {
        "configured_max_context_tokens": 2048,
        "effective_max_context_tokens": 2048,
        "chat_default_max_tokens": 4096,
        "chat_default_mode": "bounded",
        "overflow_policy_field": "session.context_overflow_policy",
        "default_overflow_policy": "reject",
        "overflow_policies": ["reject", "auto_clear_transient", "new_session", "truncate_oldest_visible"],
    }
    assert body["tokenizer"]["tokenize"] is True
    assert body["tokenizer"]["detokenize"] is True
    assert body["tokenizer"]["count_tokens"] is True
    assert body["features"]["stream_options"] == {"include_usage": True, "include_hipengine": True}
    assert body["features"]["stream_metadata"] == {
        "metadata_version": 1,
        "events": ["role", "delta", "tool_call", "done", "usage", "error"],
        "timing": [
            "elapsed_ms",
            "ttft_ms",
            "decode_elapsed_ms",
            "decode_tokens_per_second",
            "backend_*",
        ],
        "server_wall_timing": True,
        "backend_prefill_timing": "GenerationTelemetry.timing_when_available",
        "choice_phase": True,
        "choice_finish_details": True,
        "choice_generated_token_ids": "done_event_when_backend_supplies_exact_ids",
        "choice_token_accounting": True,
        "choice_token_accounting_scopes": ["live_delta", "buffered_delta", "final_choice"],
        "choice_decode_state": True,
        "choice_decode_state_scopes": ["live_delta", "buffered_delta", "final_choice"],
        "backend_telemetry_scopes": [
            "live_chunk",
            "buffered_delta_safe_decode_state",
            "buffered_done",
        ],
        "live_many_chunks": {
            "available": False,
            "source": "engine.stream_many_detailed",
            "capability": "engine.supports_stream_many",
            "requires_row_index": "GenerationTelemetry.decode_state.row_index",
            "public_surfaces": [
                "chat_answer_delta",
                "chat_reasoning_delta",
            ],
            "safe_request_shape": {
                "chat_n_gt_1": True,
                "tools": False,
                "structured_outputs": False,
                "logprobs": False,
                "stop": False,
                "continuation": False,
            },
            "fallback": "buffered_scheduler_chunks_or_buffered_choice",
        },
        "buffered_scheduler_chunks": {
            "source": "engine_or_wrapped_generator.last_batch_generation.scheduler_token_chunks",
            "requires_single_http_request_batch": True,
            "text_must_reconstruct_public_choice": True,
            "public_surfaces": [
                "completion_delta",
                "chat_answer_delta",
                "chat_reasoning_delta_without_logprobs",
                "chat_reasoning_delta_with_private_logprobs",
                "chat_content_logprob_delta",
                "chat_structured_delta_validated",
                "chat_tool_argument_delta_validated",
            ],
            "fallback_surfaces": [
                "coalesced_multi_request_batch",
                "raw_text_mismatch",
                "invalid_tool_call",
                "unmappable_tool_arguments",
                "structured_validation_failure",
                "unmappable_logprobs",
            ],
            "fallback_diagnostics": [
                "choices[].hipengine.withheld_scheduler_tool_chunks",
                "choices[].hipengine.withheld_scheduler_logprob_chunks",
            ],
        },
        "routing": "stream_options.include_hipengine",
        "kv_pool": "done_and_usage_events_when_engine_exposes_kv_pool_stats",
    }
    assert body["features"]["choice_telemetry"] == {
        "non_streaming": True,
        "streaming": "stream_options.include_hipengine",
        "decode_state": True,
        "decode_state_fields": [
            "row_index",
            "step_index",
            "prompt_tokens",
            "generated_tokens",
            "phase",
            "continuation_eligible",
            "request_id",
            "reasoning_tokens",
            "answer_tokens",
            "tool_call_tokens",
            "structured_tokens",
            "stop_suffix_state",
            "forced_tokens_pending",
            "forced_token_id",
            "forced_token_reason",
            "forced_tokens_remaining",
            "post_thinking_forced_tokens_pending",
            "post_thinking_forced_token_reason",
            "force_sequence_completion_token_sequences",
            "force_sequence_completion_reason",
            "active_processors",
            "sampler_fast_path_blockers",
            "sampler_fallback_reason",
            "budget_pressure",
            "sampler_mode",
            "full_vocab_logits_d2h",
            "logits_d2h_bytes",
            "execution_path",
            "native_compact_prefill",
            "native_caware_decode",
            "serial_decode_fallback",
            "native_sampler_rows",
        ],
        "timing": "backend_generation_telemetry_when_available",
        "usage": "backend_generation_telemetry_when_available",
        "diagnostics": "backend_generation_telemetry_when_available",
        "source": "backend_generation_telemetry_when_available",
    }
    assert body["features"]["structured_outputs"] == {
        "response_format": True,
        "json_object": True,
        "json_schema": True,
        "guided_json": True,
        "guided_json_modes": ["json_object", "json_schema"],
        "guided_regex": True,
        "guided_regex_match": "fullmatch_after_strip",
        "guided_choice": True,
        "guided_patch": True,
        "guided_diff": True,
        "guided_patch_formats": ["unified_diff"],
        "guided_diff_formats": ["unified_diff"],
        "guided_patch_fenced_policies": ["optional", "required", "forbidden"],
        "guided_diff_fenced_policies": ["optional", "required", "forbidden"],
        "guided_patch_default_fenced_policy": "optional",
        "guided_diff_default_fenced_policy": "optional",
        "guided_patch_fence_labels": ["diff", "patch"],
        "guided_diff_fence_labels": ["diff", "patch"],
        "strict_decoding": True,
        "strict_decoding_scope": "root_object_json_syntax",
        "strict_result_validation": True,
        "decode_time_close_forcing": "host_json_object_parse_validated_suffix",
        "length_finish_structural_validation": "root_object_json_prefix",
        "result_validation_failure_reasons": ["schema_violation"],
        "schema_validation": "json_schema_subset",
        "schema_subset": [
            "type",
            "enum",
            "const",
            "references.local_ref",
            "references.$defs",
            "references.definitions",
            "composition.allOf",
            "composition.anyOf",
            "composition.oneOf",
            "composition.not",
            "conditional.if",
            "conditional.then",
            "conditional.else",
            "object.properties",
            "object.patternProperties",
            "object.propertyNames",
            "object.required",
            "object.dependentRequired",
            "object.dependentSchemas",
            "object.additionalProperties=false",
            "object.additionalProperties=schema",
            "object.minProperties",
            "object.maxProperties",
            "array.items",
            "array.contains",
            "array.minItems",
            "array.maxItems",
            "array.minContains",
            "array.maxContains",
            "array.uniqueItems",
            "string.minLength",
            "string.maxLength",
            "string.pattern",
            "number.minimum",
            "number.maximum",
            "number.exclusiveMinimum",
            "number.exclusiveMaximum",
            "number.multipleOf",
        ],
        "unsupported_schema_keywords_rejected": True,
        "annotation_keywords_ignored": [
            "title",
            "description",
            "default",
            "examples",
            "deprecated",
            "readOnly",
            "writeOnly",
            "format",
        ],
    }
    assert body["features"]["grammars"] == {
        "enabled": False,
        "strict_decoding": False,
        "supported": [],
        "unsupported_fields": [
            "grammar",
            "guided_grammar",
            "guided_decoding_backend",
        ],
        "result_validation_only": [
            "json_object",
            "json_schema",
            "guided_json",
            "guided_regex",
            "guided_choice",
            "guided_patch",
            "guided_diff",
        ],
    }
    assert body["features"]["token_diagnostics"] == {
        "tokenize": True,
        "detokenize": True,
        "count_tokens": True,
        "fit_context": True,
        "session_aware_chat": True,
    }
    assert body["features"]["tools"] == {
        "enabled": True,
        "strict_decoding": True,
        "strict_decoding_scope": "tokenizer_aware_xml_function_envelope",
        "strict_result_validation": True,
        "result_validation_failure_reasons": [
            "invalid_tool_call",
            "tool_required_not_satisfied",
            "schema_violation",
        ],
        "invalid_tool_call_error_mode_field": "invalid_tool_call_error_mode",
        "invalid_tool_call_error_modes": ["finish_details", "hard_error"],
        "default_invalid_tool_call_error_mode": "finish_details",
        "hard_error_surfaces": ["http", "sse"],
        "schema_validation": "function_strict",
        "schema_subset": [
            "type",
            "enum",
            "const",
            "references.local_ref",
            "references.$defs",
            "references.definitions",
            "composition.allOf",
            "composition.anyOf",
            "composition.oneOf",
            "composition.not",
            "conditional.if",
            "conditional.then",
            "conditional.else",
            "object.properties",
            "object.patternProperties",
            "object.propertyNames",
            "object.required",
            "object.dependentRequired",
            "object.dependentSchemas",
            "object.additionalProperties=false",
            "object.additionalProperties=schema",
            "object.minProperties",
            "object.maxProperties",
            "array.items",
            "array.contains",
            "array.minItems",
            "array.maxItems",
            "array.minContains",
            "array.maxContains",
            "array.uniqueItems",
            "string.minLength",
            "string.maxLength",
            "string.pattern",
            "number.minimum",
            "number.maximum",
            "number.exclusiveMinimum",
            "number.exclusiveMaximum",
            "number.multipleOf",
        ],
        "unsupported_schema_keywords_rejected": True,
        "annotation_keywords_ignored": [
            "title",
            "description",
            "default",
            "examples",
            "deprecated",
            "readOnly",
            "writeOnly",
            "format",
        ],
        "format": "qwen_tool_call_xml_with_json_fallback",
        "compatibility_parser_repairs": [
            "qwen35_family_xml_function_envelope",
            "legacy_json_tool_call_body",
            "duplicated_tool_call_start",
            "incomplete_duplicate_tool_prefix_control_residue",
            "outer_qwen_template_control_residue",
        ],
        "malformed_json_compatibility": "invalid_tool_call_when_tools_enabled",
        "strict_malformed_blocks_rejected": True,
        "declared_tool_name_validation": True,
        "transcript_validation": {
            "role_specific_fields": True,
            "assistant_tool_call_ids_unique": True,
            "tool_results_reference_prior_call_ids": True,
            "tool_results_must_resolve_pending_calls_before_non_tool_messages": True,
            "allows_pending_tool_calls_at_transcript_end": True,
            "applies_to_session_snapshots": True,
        },
        "parallel_tool_calls_requires_opt_in": False,
        "parallel_tool_calls": True,
        "streaming_argument_chunks": True,
        "streaming_argument_chunk_chars": 128,
        "no_tool_start_suppression": True,
        "required_tool_start_forcing": True,
        "required_tool_start_forcing_scope": "initial_or_after_tokenized_thinking_close",
        "specific_tool_name_prefix_forcing": True,
        "specific_tool_name_prefix_forcing_scope": "atomic_xml_function_header",
        "strict_tool_schema_prefix_anchor": False,
        "strict_tool_schema_prefix_anchor_scope": "none",
        "tool_call_close_repair": True,
        "tool_call_close_repair_scope": "xml_parameter_boundary_suffix_exact_remaining_budget",
    }
    assert body["features"]["reasoning_controls"] == {
        "enabled": True,
        "fields": [
            "reasoning_effort",
            "enable_thinking",
            "max_think_tokens",
            "min_answer_tokens",
            "hard_think_cap",
            "soft_close_window",
            "hard_close_message",
            "hard_close_sequence",
            "thinking_token_budget",
            "thinking.allow_unbounded",
            "reasoning.allow_unbounded",
            "chat_template_kwargs",
            "thinking",
            "reasoning",
        ],
        "budget_policy": "prompt_hint_plus_tokenized_soft_and_hard_close",
        "token_budget": True,
        "token_budget_enforced": True,
        "effort_defaults": {
            "minimal": {"hard_think_cap": 256, "soft_close_window": 64, "min_answer_tokens": 256},
            "low": {"hard_think_cap": 512, "soft_close_window": 128, "min_answer_tokens": 512},
            "medium": {"hard_think_cap": 4096, "soft_close_window": 512, "min_answer_tokens": 1024},
            "high": {"hard_think_cap": 16384, "soft_close_window": 1024, "min_answer_tokens": 2048},
            "xhigh": {"hard_think_cap": 32768, "soft_close_window": 2048, "min_answer_tokens": 4096},
            "max": {"hard_think_cap": 32768, "soft_close_window": 2048, "min_answer_tokens": 4096},
        },
        "effort_default_clamp": "request_max_tokens_chat_default_or_remaining_context",
        "hard_close_validation": True,
        "hard_close_token_forcing": True,
        "soft_close_bias": True,
        "eos_suppression": True,
        "hard_close_marker": "</think>",
        "diagnostic_close_token_lowering": True,
        "diagnostic_initial_state": True,
    }
    assert body["features"]["logprobs"] == {
        "completions": True,
        "chat": True,
        "top_logprobs_max": 20,
        "streaming": "buffered",
        "live_chunk_metadata": False,
        "live_chunk_metadata_capability": "engine.supports_stream_logprobs",
        "chat_reasoning_private_stream_metadata": "choices[].hipengine.reasoning_logprobs",
        "requires_backend_token_metadata": True,
        "omission_reasons": ["backend_omitted_logprob", "prompt_logprob_unavailable"],
        "missing_backend_metadata_error": {
            "code": "unsupported_feature",
            "status_code": 501,
            "param": "logprobs",
        },
    }
    assert body["features"]["request_timeouts"] == {
        "timeout_ms": True,
        "default_timeout_ms": 250.0,
        "client_disconnect": True,
        "cooperative_backend_deadline": True,
        "cooperative_backend_cancel": True,
        "preemptive_decode_cancel": False,
    }
    assert body["admission"] == {
        "queue": {
            "max_queued_requests": 5,
            "retry_after_seconds": 1,
            "rejects_when_full": True,
        },
        "active_requests": {
            "max_active_requests": 2,
            "limits_backend_batch_width": True,
        },
        "chat_sessions": {
            "max_active": 3,
            "rejects_new_sessions_when_full": True,
        },
        "streaming": {
            "controlled_model_loop": False,
            "queue_max_chunks": 16,
            "backpressure_scope": "per_request",
            "shutdown_grace_seconds": 5.0,
        },
        "scheduler_fairness": {
            "policy": "fifo_compatible_sampling_key",
            "compatible_sampling_coalescing": True,
            "continuous_decode": False,
            "preemptive_fairness": False,
        },
    }
    assert body["errors"]["schema"] == "hipengine.error_taxonomy.v1"
    errors_by_code = {item["code"]: item for item in body["errors"]["codes"]}
    for code in (
        "unsupported_parameter",
        "unsupported_feature",
        "invalid_tool_call",
        "schema_violation",
        "invalid_continuation",
        "continuation_expired",
        "context_overflow",
        "deadline_exceeded",
        "cancelled",
        "engine_busy",
        "model_unavailable",
        "routing_failed",
    ):
        assert code in errors_by_code
    assert errors_by_code["engine_busy"]["emitted"] is True
    assert errors_by_code["engine_busy"]["status_code"] == 429
    assert errors_by_code["unsupported_feature"]["status_code"] == 501
    assert errors_by_code["invalid_continuation"]["status_code"] == 400
    assert errors_by_code["continuation_expired"]["status_code"] == 410
    assert errors_by_code["routing_failed"]["status_code"] == 502
    assert errors_by_code["routing_failed"]["retryable"] is True
    assert errors_by_code["routing_failed"]["emitted"] is False
    assert errors_by_code["invalid_tool_call"]["emitted"] is True
    assert "finish_details.reason" in errors_by_code["invalid_tool_call"]["description"]
    assert "structured-output result" in errors_by_code["schema_violation"]["description"]
    assert {
        "legacy_code": "model_not_found",
        "code": "model_unavailable",
    } in body["errors"]["aliases"]
    assert {
        "legacy_code": "unsupported_content_type",
        "code": "unsupported_parameter",
    } in body["errors"]["aliases"]
    assert {
        "legacy_code": "invalid_request",
        "code": "schema_violation",
    } in body["errors"]["aliases"]
    assert {
        "legacy_code": "validation_error",
        "code": "schema_violation",
    } in body["errors"]["aliases"]
    assert body["sampling"]["execution_modes"] == [
        "greedy_fast",
        "processed_argmax",
        "host_logits_sample",
        "gpu_sample",
    ]
    assert "suppress_token_ids" in body["sampling"]["parameters"]
    assert "min_tokens" in body["sampling"]["parameters"]
    assert "eos_token_id" in body["sampling"]["parameters"]
    assert "json_object_close_forcing" in body["sampling"]["parameters"]
    assert body["sampling"]["native_gpu"] == {
        "enabled": True,
        "env": "HIPENGINE_QWEN35_NATIVE_SAMPLER",
        "disable_env": "HIPENGINE_QWEN35_NATIVE_SAMPLER=0",
        "scope": "paro_c1_and_serial_per_slot_c_gt_1",
        "c_gt_1": "serial_per_slot_when_all_rows_supported",
        "true_batched_c_gt_1": False,
        "default_path": True,
        "top_k_max": 64,
        "top_p_min_p": "exact_full_vocab_top_k_0_and_bounded_top_k",
        "selected_logprobs": True,
        "top_logprobs": {
            "full_vocab_top_k_0": True,
            "bounded_top_k": True,
            "max": 64,
            "constraint": "top_k=0 or top_logprobs <= top_k <= 64",
        },
        "processors": [
            "logit_bias",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "suppress_token_ids",
            "min_tokens",
        ],
        "post_selection_controls": [
            "stop_token_ids",
            "stop_token_sequences",
        ],
        "unsupported": list(NATIVE_GPU_SAMPLER_UNSUPPORTED_CAPABILITIES),
    }
    assert body["sampling"]["speculative_mtp"] == {
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
    assert body["sampling"]["speculative_mtp"]["incompatible_fields"] == list(
        SPECULATIVE_MTP_INCOMPATIBLE_FIELDS
    )
    assert body["sampling"]["speculative_mtp"]["incompatible_conditions"] == dict(
        SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS
    )
    mtp_incompatible = set(body["sampling"]["speculative_mtp"]["incompatible_fields"])
    assert {
        "response_format",
        "guided_json",
        "guided_regex",
        "guided_choice",
        "guided_patch",
        "guided_diff",
    }.isdisjoint(mtp_incompatible)
    assert body["sessions"] == {
        "resident_context": True,
        "commit_policy": _session_commit_policy_capability(),
        "continuations": _continuation_capability(),
        "metadata": _session_metadata_capability(3),
    }
    assert body["routing"] == {"loaded_model_count": 1, "multiple_models": False}
    assert body["parallelism"] == _parallelism_capability()
    assert "session.id" not in body["unsupported_fields"]
    assert "timeout_ms" not in body["unsupported_fields"]
    assert "parallel_tool_calls" not in body["unsupported_fields"]
    assert "response_format" not in body["unsupported_fields"]
    assert "continuation_id" not in body["unsupported_fields"]
    assert "grammar" in body["unsupported_fields"]
    assert "guided_json" not in body["unsupported_fields"]
    assert "guided_json" in body["features"]["grammars"]["result_validation_only"]
    assert "guided_patch" not in body["unsupported_fields"]
    assert "guided_diff" not in body["unsupported_fields"]


def test_capabilities_endpoint_reports_default_gguf_native_sampler(
    monkeypatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", raising=False)
    config = ServerConfig(
        model="/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        served_model_name="fake-model",
        eager_load=False,
    )
    disabled = TestClient(create_app(config, llm=FakeLLM())).get(
        "/v1/hipengine/capabilities"
    )
    assert disabled.status_code == 200
    native = disabled.json()["sampling"]["native_gpu"]
    assert native["enabled"] is True
    assert native["enable_env"] == "HIPENGINE_QWEN35_NATIVE_SAMPLER=1"
    assert native["scope"] == "gguf_resident_c1_and_dense_compatible_c_gt_1"
    assert native["c_gt_1"] == (
        "single_batched_launch_when_compatible_else_native_rows"
    )
    assert native["true_batched_c_gt_1"] is True
    assert native["default_path"] is True
    assert "forced_tokens_pending" in native["unsupported"]
    assert "gguf" not in native["unsupported"]
    assert "true_batched_c_gt_1" not in native["unsupported"]

    monkeypatch.setenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", "1")
    enabled = TestClient(create_app(config, llm=FakeLLM())).get(
        "/v1/hipengine/capabilities"
    )
    assert enabled.status_code == 200
    assert enabled.json()["sampling"]["native_gpu"]["enabled"] is True

    monkeypatch.setenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", "0")
    rollback = TestClient(create_app(config, llm=FakeLLM())).get("/v1/hipengine/capabilities")
    assert rollback.json()["sampling"]["native_gpu"]["enabled"] is False


def test_capabilities_endpoint_reports_speculative_mtp_when_config_and_engine_support(monkeypatch) -> None:
    # The MTP context window is an environment resolution, so pin it: a value
    # exported in the developer's shell must not change this expectation.
    monkeypatch.delenv("HIPENGINE_MTP2_MAX_CONTEXT_TOKENS", raising=False)
    fake = SpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            speculative_mtp_serving="opt_in",
        ),
        llm=fake,
    )
    client = TestClient(app)

    body = client.get("/v1/hipengine/capabilities").json()

    assert body["sampling"]["speculative_mtp"] == {
        "serving_route": True,
        "configured_policy": "opt_in",
        "engine_supported": True,
        "model_has_mtp_tensors": True,
        "certified_default_scopes": [],
        "automatic_route_promoted": False,
        "sampling_compatible": True,
        "compatibility_guard": "supports_speculative_mtp_sampling",
        "allowed_execution_modes": ["greedy_fast"],
        "incompatible_fields": list(SPECULATIVE_MTP_INCOMPATIBLE_FIELDS),
        "incompatible_conditions": dict(SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS),
        "thinking_policy": "hint",
        "processed_target_verification": False,
        "candidate_budget": {
            "requested": None,
            "resolved": 3,
            "source": "model_plugin_evidence",
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
        "policy": "opt_in",
        "request_field": "speculative_mtp",
        "default_enabled": False,
        "streaming_compatible": True,
        "batch_route": "speculative_mtp",
        "physical_concurrency": "generation2_target_frontier",
        "max_physical_target_slots": 4,
        "route_coalescing_is_physical_concurrency": True,
        "circuit_breaker": {
            "state": "closed",
            "operator_disabled": False,
            "operator_reason": None,
            "threshold": 3,
            "open_scopes": [],
            "failures": [],
            "transitions": [],
            "reset_policy": "server_restart",
        },
    }
    assert client.get("/v1/models").json()["data"][0]["hipengine"]["capabilities"]["speculative_mtp"] is True


def test_capabilities_endpoint_defaults_to_auto_exact_fallback() -> None:
    fake = SpeculativeMTPFakeLLM()
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        eager_load=False,
    )
    assert config.speculative_mtp_serving == "auto"
    app = create_app(
        config,
        llm=fake,
    )
    client = TestClient(app)

    payload = client.get("/v1/hipengine/capabilities").json()["sampling"]["speculative_mtp"]

    assert payload["policy"] == "auto"
    assert payload["default_enabled"] is False
    assert payload["auto_route"] == {
        "selected_route": "default",
        "reason": "automatic_mtp_scope_not_promoted",
        "selected_candidate_count": 0,
        "policy_key": DEFAULT_AUTO_DEPTH_POLICY.policy_key,
        "policy_fingerprint": DEFAULT_AUTO_DEPTH_POLICY.fingerprint,
        "exact_default_required": True,
        "compatibility_mtp_explicit_only": True,
        "evidence": "benchmarks/results/2026-08-26-gfx1151-specdec2-perf-p9-fixed-policy.json",
    }


def test_mtp_capability_reports_capability_resolved_candidate_budget() -> None:
    """An omitted budget reports the depth capability data resolved."""

    fake = SpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            speculative_mtp_serving="opt_in",
        ),
        llm=fake,
    )

    payload = TestClient(app).get("/v1/hipengine/capabilities").json()

    assert payload["sampling"]["speculative_mtp"]["candidate_budget"] == {
        "requested": None,
        "resolved": 3,
        "source": "model_plugin_evidence",
    }


def test_mtp_capability_reports_an_explicit_candidate_budget() -> None:
    """A pinned depth is reported as pinned, never rewritten."""

    fake = SpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            speculative_mtp_serving="opt_in",
            speculative_candidate_budget=4,
        ),
        llm=fake,
    )

    payload = TestClient(app).get("/v1/hipengine/capabilities").json()

    assert payload["sampling"]["speculative_mtp"]["candidate_budget"] == {
        "requested": 4,
        "resolved": 4,
        "source": "explicit",
    }


def test_candidate_budget_reads_the_providers_declared_default_before_load() -> None:
    """A configured provider owns its depth before the target model loads."""

    from hipengine.server.api import _candidate_budget_resolution
    from hipengine.speculative.registry import (
        SpeculativeProviderKey,
        register_speculative_provider,
    )

    register_speculative_provider(
        SpeculativeProviderKey(
            provider="test_server_declared_budget",
            target_model="test_model",
            backend="test_backend",
            quant="test_quant",
        ),
        lambda **kwargs: None,
        replace=True,
        candidate_budget=7,
    )
    config = ServerConfig(
        model="target",
        speculative_provider="test_server_declared_budget",
        draft_model="drafter",
    )

    assert _candidate_budget_resolution(config) == {
        "requested": None,
        "resolved": 7,
        "source": "provider_default",
    }


def test_candidate_budget_reports_pending_for_ambiguous_provider_declarations() -> None:
    """Pre-load capability reports pending, then the loaded engine resolves it."""

    from hipengine.server.api import _candidate_budget_resolution
    from hipengine.speculative.registry import (
        SpeculativeProviderKey,
        register_speculative_provider,
    )

    for quant, budget in (("quant_a", 3), ("quant_b", 7)):
        register_speculative_provider(
            SpeculativeProviderKey(
                provider="test_server_ambiguous_budget",
                target_model="test_model",
                backend="test_backend",
                quant=quant,
            ),
            lambda **kwargs: None,
            replace=True,
            candidate_budget=budget,
        )
    config = ServerConfig(
        model="target",
        speculative_provider="test_server_ambiguous_budget",
        draft_model="drafter",
    )

    assert _candidate_budget_resolution(config) == {
        "requested": None,
        "resolved": None,
        "source": "pending",
    }

    class LoadedEngine:
        @property
        def speculative_candidate_budget_resolution(self) -> dict[str, Any]:
            return {
                "requested_candidate_budget": None,
                "candidate_budget": 3,
                "candidate_budget_source": "provider_default",
            }

    assert _candidate_budget_resolution(config, engine=LoadedEngine()) == {
        "requested": None,
        "resolved": 3,
        "source": "provider_default",
    }


def test_candidate_budget_rejects_an_unregistered_configured_provider() -> None:
    """A configured provider with no declaration is a miss, not a guess."""

    from hipengine.server.api import _candidate_budget_resolution

    config = ServerConfig(
        model="target",
        speculative_provider="test_server_unregistered_provider",
        draft_model="drafter",
    )

    with pytest.raises(KeyError, match="unregistered speculative provider"):
        _candidate_budget_resolution(config)


def test_candidate_budget_pending_engine_precedes_the_provider_default() -> None:
    """An engine that has not loaded its artifact is pending, not provider-default."""

    from hipengine.server.api import _candidate_budget_resolution

    class UnloadedEngine:
        @property
        def speculative_candidate_budget_resolution(self) -> dict[str, Any]:
            return {
                "requested_candidate_budget": None,
                "candidate_budget": None,
                "candidate_budget_source": "pending",
            }

    config = ServerConfig(
        model="target",
        speculative_provider="dflash",
        draft_model="drafter",
    )

    assert _candidate_budget_resolution(config, engine=UnloadedEngine()) == {
        "requested": None,
        "resolved": None,
        "source": "pending",
    }


def test_candidate_budget_explicit_precedes_the_provider_default() -> None:
    """An operator-pinned depth wins over the provider's declaration."""

    from hipengine.server.api import _candidate_budget_resolution

    config = ServerConfig(
        model="target",
        speculative_provider="dflash",
        draft_model="drafter",
        speculative_candidate_budget=5,
    )

    assert _candidate_budget_resolution(config) == {
        "requested": 5,
        "resolved": 5,
        "source": "explicit",
    }


def test_candidate_budget_reports_pending_before_the_engine_loads() -> None:
    """An engine that has not loaded its artifact reports pending, not a depth."""

    from hipengine.server.api import _candidate_budget_resolution

    class UnloadedEngine:
        @property
        def speculative_candidate_budget_resolution(self) -> dict[str, Any]:
            return {
                "requested_candidate_budget": None,
                "candidate_budget": None,
                "candidate_budget_source": "pending",
            }

    assert _candidate_budget_resolution(
        ServerConfig(model="fake-path"), engine=UnloadedEngine()
    ) == {"requested": None, "resolved": None, "source": "pending"}


def test_server_config_accepts_an_omitted_candidate_budget() -> None:
    """Omitting the budget is a supported configuration, not a default of 4."""

    assert ServerConfig(model="fake-path").speculative_candidate_budget is None
    assert (
        ServerConfig(model="fake-path", speculative_candidate_budget=2)
        .speculative_candidate_budget
        == 2
    )
    with pytest.raises(ValueError, match="must be positive"):
        ServerConfig(model="fake-path", speculative_candidate_budget=0)


def test_capabilities_endpoint_does_not_infer_default_mtp_from_dense_boolean() -> None:
    fake = SpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            speculative_mtp_serving="enabled",
        ),
        llm=fake,
    )
    client = TestClient(app)

    payload = client.get("/v1/hipengine/capabilities").json()["sampling"]["speculative_mtp"]

    assert payload["policy"] == "enabled"
    assert payload["default_enabled"] is False
    assert payload["automatic_route_promoted"] is False
    assert payload["certified_default_scopes"] == []
    assert "auto_route" not in payload


def test_capabilities_endpoint_advertises_live_stream_logprobs_when_engine_supports_metadata() -> None:
    fake = FakeLLM(outputs=["ok"])
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False), llm=fake)
    client = TestClient(app)

    response = client.get("/v1/hipengine/capabilities")

    assert response.status_code == 200
    assert response.json()["features"]["logprobs"] == {
        "completions": True,
        "chat": True,
        "top_logprobs_max": 20,
        "streaming": "live_chunk_metadata",
        "live_chunk_metadata": True,
        "live_chunk_metadata_capability": "engine.supports_stream_logprobs",
        "chat_reasoning_private_stream_metadata": "choices[].hipengine.reasoning_logprobs",
        "requires_backend_token_metadata": True,
        "omission_reasons": ["backend_omitted_logprob", "prompt_logprob_unavailable"],
        "missing_backend_metadata_error": {
            "code": "unsupported_feature",
            "status_code": 501,
            "param": "logprobs",
        },
    }


def test_api_error_taxonomy_table_matches_capabilities_manifest() -> None:
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False), llm=FakeLLM())
    client = TestClient(app)

    response = client.get("/v1/hipengine/capabilities")

    assert response.status_code == 200
    api_table = _api_error_taxonomy_table()
    manifest_by_code = {item["code"]: item for item in response.json()["errors"]["codes"]}
    assert set(api_table) == set(manifest_by_code)
    for code, item in manifest_by_code.items():
        assert api_table[code]["status_code"] == item["status_code"]
        assert api_table[code]["retryable"] == item["retryable"]
        assert api_table[code]["current_emission"]


def test_capabilities_endpoint_reports_auto_chat_default_and_cache_config() -> None:
    fake = FakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            chat_default_max_tokens=None,
            kv_storage="int8_per_token_head",
            kv_scale_dtype="fp32",
            prefix_cache="radix",
            max_chat_sessions=6,
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.get("/v1/hipengine/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["context"]["chat_default_max_tokens"] is None
    assert body["context"]["chat_default_mode"] == "auto"
    assert body["cache"]["prefix_cache"] == "radix"
    assert body["cache"]["kv_storage"] == "int8_per_token_head"
    assert body["cache"]["kv_scale_dtype"] == "fp32"
    assert body["sessions"]["metadata"]["max_active"] == 6


def test_capabilities_report_controlled_streaming_backpressure_and_continuous_membership() -> None:
    fake = FakeLLM()
    fake.supports_controlled_streaming = True
    kv_capability = {
        "schema_version": 1,
        "capability_id": "qualified-capability",
        "status": "qualified",
        "runtime_action": "admit",
        "promotion_eligible": True,
        "diagnostic_override": False,
        "requested": {"kv_storage": "int8_per_token_head"},
        "effective_kv_storage": "int8_per_token_head",
        "artifact": {"sha256": "abc", "content_verified": True},
        "evidence": {"quality_artifact": "benchmarks/results/quality.json"},
        "reason": "qualified fixture",
    }
    fake._text_generator = SimpleNamespace(kv_capability_provenance=kv_capability)
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            stream_queue_max_chunks=7,
            shutdown_grace_seconds=1.5,
            metrics="prometheus",
        ),
        llm=fake,
    )

    client = TestClient(app)
    body = client.get("/v1/hipengine/capabilities").json()

    assert body["admission"]["streaming"] == {
        "controlled_model_loop": True,
        "queue_max_chunks": 7,
        "backpressure_scope": "per_request",
        "shutdown_grace_seconds": 1.5,
    }
    assert body["admission"]["scheduler_fairness"]["continuous_decode"] is True
    assert body["model"]["kv_capability"] == kv_capability
    ready = client.get("/ready").json()
    assert ready["queue"]["scheduler_fairness"]["continuous_decode"] is True
    assert ready["model"]["kv_capability"] == kv_capability
    models = client.get("/v1/models").json()
    assert models["data"][0]["hipengine"]["kv_capacity"]["capability"] == kv_capability
    metrics = client.get("/metrics").text
    assert 'continuous_decode="true"' in metrics
