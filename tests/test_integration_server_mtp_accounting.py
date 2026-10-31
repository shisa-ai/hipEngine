"""OpenAI server integration tests: mtp accounting.

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
    AccountingSpeculativeMTPFakeLLM,
    SpeculativeMTPFakeLLM,
)
from tests.support.server_api_support import (
    _full_coverage_accounting,
    _metric_value,
    _mtp_accounting_detail,
    _sse_payloads,
)


@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
def test_explicit_mtp_streaming_uses_committed_speculative_chunks(endpoint: str) -> None:
    fake = SpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="opt_in",
        ),
        llm=fake,
    )
    payload = {
        "model": "fake-model",
        "max_tokens": 3,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True, "include_hipengine": True},
        "speculative_mtp": True,
    }
    if endpoint.endswith("chat/completions"):
        payload["messages"] = [{"role": "user", "content": "hello"}]
    else:
        payload["prompt"] = "hello"

    response = TestClient(app).post(endpoint, json=payload)

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(item.get("error") for item in payloads)
    if endpoint.endswith("chat/completions"):
        text = "".join(
            item["choices"][0]["delta"].get("content", "")
            for item in payloads
            if item.get("choices")
        )
        assert text.startswith("mtp:")
    else:
        text = "".join(
            item["choices"][0].get("text", "")
            for item in payloads
            if item.get("choices")
        )
        assert text == "mtp:hello"
    assert fake.calls == []
    assert len(fake.mtp_calls) == 1
    done = next(
        item["choices"][0]
        for item in payloads
        if item.get("choices")
        and item["choices"][0].get("finish_reason") is not None
    )
    assert done["hipengine"]["generation_shape"]["route"] == "speculative_mtp"
    assert done["hipengine"]["speculative_mtp"] == {
        "effective_route": "speculative_mtp",
        "used": True,
        "draft_tokens": 0,
        "accepted_draft_tokens": 0,
        "rejected_draft_tokens": 0,
        "acceptance_rate": None,
        "draft_cycles": 0,
        "thinking_policy": "hint",
        "thinking_controls": "prompt_hint_only",
    }


@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
def test_explicit_mtp_streaming_done_reports_output_accounting(endpoint: str) -> None:
    """A streamed MTP request reports the same split as the blocking response."""

    fake = AccountingSpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="opt_in",
        ),
        llm=fake,
    )
    payload: dict[str, Any] = {
        "model": "fake-model",
        "max_tokens": 12,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True, "include_hipengine": True},
        "speculative_mtp": True,
    }
    if endpoint.endswith("chat/completions"):
        payload["messages"] = [{"role": "user", "content": "hello"}]
    else:
        payload["prompt"] = "hello"

    response = TestClient(app).post(endpoint, json=payload)

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(item.get("error") for item in payloads)
    done = next(
        item["choices"][0]
        for item in payloads
        if item.get("choices")
        and item["choices"][0].get("finish_reason") is not None
    )
    summary = done["hipengine"]["speculative_mtp"]
    assert summary["used"] is True
    assert summary["effective_route"] == "speculative_mtp"
    assert summary["draft_cycles"] == 3
    assert summary["accepted_draft_tokens"] == 6
    assert summary["mtp_output_tokens"] == 9
    assert summary["ar_output_tokens"] == 3
    assert summary["output_accounting"] == {
        "completion_tokens": 12,
        "mtp_output_tokens": 9,
        "ar_output_tokens": 3,
        "ar_output_tokens_in_cycles": 0,
        "mtp_coverage": 0.75,
        # Three committed depth-3 cycles emit three verified tokens plus six
        # accepted drafts, which is exactly the reported speculative output.
        "mtp_output_tokens_explained_by_cycles": 9,
        "unexplained_mtp_output_tokens": 0,
        "speculative_cycles": 3,
        "reconciled": True,
        "reconciled_reasons": [],
    }
    assert summary["selected_depth_histogram"] == {"3": 3}
    assert summary["fallback_event_counts"] == {}
    assert summary["first_fallback_position"] is None
    # The same stream also reports draft acceptance through OpenAI usage.
    usage_event = next(item for item in payloads if item.get("usage"))
    assert usage_event["usage"]["completion_tokens"] == 12
    assert usage_event["usage"]["completion_tokens_details"] == {
        "accepted_prediction_tokens": 6,
        "rejected_prediction_tokens": 3,
    }


def test_mtp_summary_honors_batch_timing_ownership() -> None:
    details = []
    for row_index, timing_owner in enumerate((True, False)):
        details.append(
            GenerationOutput(
                text="done",
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=4,
                    generated_tokens=3,
                    row_index=row_index,
                    execution_path="speculative_mtp_server",
                    timing={
                        "mtp_cycles_count": 2,
                        "mtp_generated_draft_tokens": 6,
                        "mtp_accepted_draft_tokens": 4,
                    },
                    timing_scope="batch",
                    batch_id="mtp-batch",
                    group_rows=2,
                    timing_owner=timing_owner,
                ),
            )
        )

    assert _mtp_accepted_rejected_counts(details) == (4, 2)
    expected = {
        "effective_route": "speculative_mtp",
        "used": True,
        "draft_tokens": 6,
        "accepted_draft_tokens": 4,
        "rejected_draft_tokens": 2,
        "acceptance_rate": 4 / 6,
        "draft_cycles": 2,
    }
    assert _mtp_response_summary("speculative_mtp", details) == expected
    # Frontend grouping may predict K0 before the resident due work selects K>0.
    # Committed-cycle telemetry owns the effective route and records that change.
    assert _mtp_response_summary("default", details) == {
        **expected,
        "selected_route": "default",
        "decision_reason": "resident_dynamic_mtp",
    }


def test_mtp_usage_reads_execution_accounting_without_timing_mirror() -> None:
    detail = _mtp_accounting_detail(
        generated_tokens=24,
        accounting=_full_coverage_accounting(),
        timing={},
    )

    assert _mtp_accepted_rejected_counts([detail]) == (15, 9)
    summary = _mtp_response_summary("speculative_mtp", [detail])
    assert summary["used"] is True
    assert summary["draft_cycles"] == 8
    assert summary["draft_tokens"] == 24
    assert summary["accepted_draft_tokens"] == 15


def test_mtp_usage_reads_zero_cycle_execution_accounting() -> None:
    detail = _mtp_accounting_detail(
        generated_tokens=1,
        accounting=_full_coverage_accounting(
            cycles=0,
            generated_draft_tokens=0,
            accepted_draft_tokens=0,
            selected_depth_histogram={},
            mtp_output_tokens=0,
            ar_output_tokens_in_cycles=0,
            ar_step_reason_counts={"no_provider": 1},
        ),
        execution_path="gguf_packed_ar_server_decode",
        timing={},
    )

    assert _mtp_accepted_rejected_counts([detail]) == (0, 0)
    summary = _mtp_response_summary("speculative_mtp", [detail])
    assert summary["used"] is False
    assert summary["draft_cycles"] == 0


def test_mtp_usage_reads_fallback_accounting_without_timing_mirror() -> None:
    detail = _mtp_accounting_detail(
        generated_tokens=24,
        accounting=_full_coverage_accounting(
            cycles=3,
            generated_draft_tokens=6,
            accepted_draft_tokens=3,
            selected_depth_histogram={"0": 1, "3": 2},
            mtp_output_tokens=7,
            ar_output_tokens_in_cycles=2,
            first_fallback_position=8,
            ar_step_reason_counts={"target_graph_output_room_miss": 1},
        ),
        timing={},
    )

    assert _mtp_accepted_rejected_counts([detail]) == (3, 3)
    summary = _mtp_response_summary("speculative_mtp", [detail])
    assert summary["draft_cycles"] == 3
    assert summary["draft_tokens"] == 6
    assert summary["accepted_draft_tokens"] == 3
    assert summary["first_fallback_position"] == 8


def test_mtp_summary_uses_provider_diagnostics_when_no_normalized_timing_exists() -> None:
    details = [
        GenerationOutput(
            text="done",
            telemetry=GenerationTelemetry.from_decode_counts(
                prompt_tokens=4,
                generated_tokens=7,
                execution_path="gemma4_mtp_batched_exact_verify",
                diagnostics={
                    "speculative_provider": "gemma4_mtp",
                    "proposed_draft_tokens": 7,
                    "accepted_draft_tokens": 4,
                    "cycles": [{"accepted": 2}, {"accepted": 2}],
                },
            ),
        )
    ]

    assert _mtp_accepted_rejected_counts(details) == (4, 3)
    assert _mtp_response_summary("speculative_mtp", details) == {
        "effective_route": "speculative_mtp",
        "used": True,
        "draft_tokens": 7,
        "accepted_draft_tokens": 4,
        "rejected_draft_tokens": 3,
        "acceptance_rate": 4 / 7,
        "draft_cycles": 2,
    }


def test_mtp_summary_treats_owned_zero_draft_timing_as_realized_mtp() -> None:
    details = [
        GenerationOutput(
            text="done",
            telemetry=GenerationTelemetry.from_decode_counts(
                prompt_tokens=4,
                generated_tokens=1,
                row_index=0,
                execution_path="resident_verify_chain",
                timing={
                    "mtp_cycles_count": 0,
                    "mtp_generated_draft_tokens": 0,
                    "mtp_accepted_draft_tokens": 0,
                },
                timing_scope="choice",
                timing_owner=True,
            ),
        )
    ]

    assert _mtp_response_summary("speculative_mtp", details) == {
        "effective_route": "speculative_mtp",
        "used": True,
        "draft_tokens": 0,
        "accepted_draft_tokens": 0,
        "rejected_draft_tokens": 0,
        "acceptance_rate": None,
        "draft_cycles": 0,
    }


def test_mtp_summary_reports_resident_dynamic_mtp_after_frontend_k0() -> None:
    details = [
        GenerationOutput(
            text="done",
            telemetry=GenerationTelemetry.from_decode_counts(
                prompt_tokens=4,
                generated_tokens=3,
                row_index=0,
                execution_path="resident_verify_chain",
                timing={
                    "mtp_cycles_count": 1,
                    "mtp_generated_draft_tokens": 2,
                    "mtp_accepted_draft_tokens": 1,
                },
                timing_scope="choice",
                timing_owner=True,
            ),
        )
    ]

    assert _mtp_response_summary(
        "default",
        details,
        route_decision={
            "requested_route": "speculative_mtp_auto",
            "reason": "physical_group_not_qualified",
        },
    ) == {
        "effective_route": "speculative_mtp",
        "selected_route": "default",
        "used": True,
        "draft_tokens": 2,
        "accepted_draft_tokens": 1,
        "rejected_draft_tokens": 1,
        "acceptance_rate": 0.5,
        "draft_cycles": 1,
        "decision_reason": "resident_dynamic_mtp",
        "requested_route": "speculative_mtp_auto",
        "selection_reason": "physical_group_not_qualified",
    }


def test_mtp_summary_reports_selected_mtp_backend_k0_fallback_truthfully() -> None:
    details = [
        GenerationOutput(
            text="done",
            telemetry=GenerationTelemetry.from_decode_counts(
                prompt_tokens=4,
                generated_tokens=3,
                row_index=0,
                execution_path="gguf_packed_ar_server_decode",
                timing={"request_total_ms": 1.0},
                timing_scope="choice",
                timing_owner=True,
            ),
        )
    ]

    assert _mtp_accepted_rejected_counts(details) is None
    assert _mtp_response_summary("speculative_mtp", details) == {
        "effective_route": "default",
        "selected_route": "speculative_mtp",
        "used": False,
        "draft_tokens": 0,
        "accepted_draft_tokens": 0,
        "rejected_draft_tokens": 0,
        "acceptance_rate": None,
        "draft_cycles": 0,
        "decision_reason": "backend_k0_fallback",
    }


def test_mtp_summary_separates_the_four_refusal_facts_from_the_folded_reason() -> None:
    """A refused row and an engaged row must not fold to the same story.

    Both rows here emit autoregressive steps attributed to ``no_provider``, so
    both report the same ``fallback_reason``. What separates them is the
    activation reason, the provider readiness, the realized group width, and the
    group's autoregressive decision -- each published on its own, because the
    first row was refused at admission for its whole life while the second only
    waited one cycle for its provider to prime.
    """

    refused = _mtp_accounting_detail(
        generated_tokens=3,
        accounting=_full_coverage_accounting(
            cycles=0,
            generated_draft_tokens=0,
            accepted_draft_tokens=0,
            selected_depth_histogram={},
            mtp_output_tokens=0,
            ar_output_tokens_in_cycles=0,
            first_fallback_position=0,
            ar_step_reason_counts={"no_provider": 3},
            activation_reason="prefix_reuse_k0",
            provider_readiness="absent",
            provider_decline_reason="provider_state_absent",
            provider_state_present=False,
            plan_group_rows=1,
            plan_ar_only=True,
            plan_reason="no_provider",
        ),
    )

    summary = _mtp_response_summary("speculative_mtp", [refused])

    assert summary["fallback_reason"] == "prefix_reuse_k0"
    assert summary["execution"] == {
        "activation_reason": "prefix_reuse_k0",
        "provider_readiness": "absent",
        "provider_decline_reason": "provider_state_absent",
        # Reported beside readiness because the two answer different questions:
        # readiness says whether the capability was granted, this says whether
        # the row still holds provider state. A reused prefix never primes, so
        # there is no state here even before the refusal.
        "provider_state_present": False,
        "plan_group_rows": 1,
        "plan_ar_only": True,
        "plan_reason": "no_provider",
    }

    engaged = _mtp_accounting_detail(
        generated_tokens=24,
        accounting=_full_coverage_accounting(
            ar_step_reason_counts={"no_provider": 1},
            provider_readiness="ready",
            plan_group_rows=1,
            plan_ar_only=False,
            plan_reason="speculative_qualified",
            provider_state_present=True,
            cycle_timing_ms={
                "proposal": 1.0,
                "target": 40.0,
                "provider_update": 2.0,
                "accept": 3.0,
                "candidate_readback": 0.0,
                "target_readback": 1.0,
                "accept_upload": 0.0,
                "accept_tail": 0.0,
                "accept_enqueue": 0.5,
                "selected_commit": 0.5,
            },
        ),
    )

    engaged_summary = _mtp_response_summary("speculative_mtp", [engaged])

    # The phase split is reported with the cycle count it must be divided by:
    # these windows are host wall, so a sum alone is not a per-cycle cost.
    assert engaged_summary["draft_cycles"] == 8
    assert engaged_summary["cycle_timing_ms"]["target"] == 40.0
    assert engaged_summary["cycle_timing_ms"]["proposal"] == 1.0

    # A ready provider with no fallback position must not publish the planner's
    # root bookkeeping reason as if the request fell back.
    assert "fallback_reason" not in engaged_summary
    assert engaged_summary["execution"] == {
        "activation_reason": None,
        "provider_readiness": "ready",
        "provider_decline_reason": None,
        "provider_state_present": True,
        "plan_group_rows": 1,
        "plan_ar_only": False,
        "plan_reason": "speculative_qualified",
    }


def test_mtp_summary_reports_full_speculative_output_coverage() -> None:
    """``used`` alone must not read as full coverage: report the output split."""

    detail = _mtp_accounting_detail(
        generated_tokens=24,
        accounting=_full_coverage_accounting(),
    )

    summary = _mtp_response_summary("speculative_mtp", [detail])

    assert summary["used"] is True
    assert summary["draft_tokens"] == 24
    assert summary["accepted_draft_tokens"] == 15
    assert summary["draft_cycles"] == 8
    assert summary["mtp_output_tokens"] == 23
    # The prompt root token is produced by the target prefill, not a cycle.
    assert summary["ar_output_tokens"] == 1
    assert summary["output_accounting"] == {
        "completion_tokens": 24,
        "mtp_output_tokens": 23,
        "ar_output_tokens": 1,
        "ar_output_tokens_in_cycles": 0,
        "mtp_coverage": 23 / 24,
        "mtp_output_tokens_explained_by_cycles": 23,
        "unexplained_mtp_output_tokens": 0,
        "speculative_cycles": 8,
        "reconciled": True,
        "reconciled_reasons": [],
    }
    assert summary["selected_depth_histogram"] == {"3": 8}
    assert summary["fallback_event_counts"] == {}
    assert "fallback_reason" not in summary
    # No mid-generation fallback: speculation covered everything after the root.
    assert summary["first_fallback_position"] is None


def test_mtp_summary_reports_mixed_speculation_with_the_fallback_position() -> None:
    detail = _mtp_accounting_detail(
        generated_tokens=24,
        accounting=_full_coverage_accounting(
            cycles=3,
            generated_draft_tokens=8,
            # Two committed depth-3 cycles emit one verified token each plus
            # four accepted drafts in total, which is the six speculative
            # output tokens this summary reports.
            accepted_draft_tokens=4,
            selected_depth_histogram={"0": 1, "3": 2},
            mtp_output_tokens=6,
            ar_output_tokens_in_cycles=1,
            first_fallback_position=7,
            ar_step_reason_counts={"target_graph_context_bucket_miss": 16},
        ),
    )

    summary = _mtp_response_summary("speculative_mtp", [detail])

    assert summary["used"] is True
    assert summary["mtp_output_tokens"] == 6
    assert summary["ar_output_tokens"] == 18
    assert summary["output_accounting"]["mtp_coverage"] == 0.25
    assert summary["output_accounting"]["mtp_output_tokens_explained_by_cycles"] == 6
    assert summary["output_accounting"]["unexplained_mtp_output_tokens"] == 0
    assert summary["output_accounting"]["reconciled"] is True
    assert summary["output_accounting"]["reconciled_reasons"] == []
    assert summary["selected_depth_histogram"] == {"0": 1, "3": 2}
    assert summary["fallback_event_counts"] == {
        "target_graph_context_bucket_miss": 16
    }
    assert summary["fallback_reason"] == "target_graph_context_bucket_miss"
    assert summary["first_fallback_position"] == 7


def test_mtp_summary_surfaces_the_prompt_fallback_reason() -> None:
    """A refused activation reports why, not just ``backend_k0_fallback``."""

    detail = _mtp_accounting_detail(
        generated_tokens=24,
        execution_path="gguf_packed_ar_server_decode",
        accounting=_full_coverage_accounting(
            candidate_budget=0,
            cycles=0,
            generated_draft_tokens=0,
            accepted_draft_tokens=0,
            selected_depth_histogram={},
            mtp_output_tokens=0,
            prompt_fallback_reason="target_context_k0",
            ar_step_reason_counts={"no_provider": 23},
        ),
    )

    summary = _mtp_response_summary("speculative_mtp", [detail])

    assert summary["used"] is False
    assert summary["effective_route"] == "default"
    assert summary["selected_route"] == "speculative_mtp"
    # The compatibility field is preserved and the specific reason is added.
    assert summary["decision_reason"] == "backend_k0_fallback"
    assert summary["fallback_reason"] == "target_context_k0"
    assert summary["fallback_event_counts"] == {
        "no_provider": 23,
        "target_context_k0": 1,
    }
    assert summary["mtp_output_tokens"] == 0
    assert summary["ar_output_tokens"] == 24
    assert summary["output_accounting"]["mtp_coverage"] == 0.0
    assert summary["first_fallback_position"] == 0


def test_mtp_summary_leaves_a_request_without_intent_unchanged() -> None:
    detail = GenerationOutput(
        text="done",
        telemetry=GenerationTelemetry.from_decode_counts(
            prompt_tokens=4,
            generated_tokens=3,
            row_index=0,
            execution_path="gguf_packed_ar_server_decode",
            timing={"request_total_ms": 1.0},
            timing_scope="choice",
            timing_owner=True,
            diagnostics={"prefix_cache": {}},
        ),
    )

    assert _mtp_response_summary("default", [detail]) == {
        "effective_route": "default",
        "used": False,
        "draft_tokens": 0,
        "accepted_draft_tokens": 0,
        "rejected_draft_tokens": 0,
        "acceptance_rate": None,
        "draft_cycles": 0,
    }


def test_mtp_summary_reconciles_a_short_output_tail() -> None:
    """The final cycle of a truncated run emits fewer tokens than its depth."""

    detail = _mtp_accounting_detail(
        generated_tokens=4,
        accounting=_full_coverage_accounting(
            cycles=2,
            generated_draft_tokens=6,
            accepted_draft_tokens=3,
            selected_depth_histogram={"3": 2},
            mtp_output_tokens=3,
            first_fallback_position=None,
        ),
    )

    summary = _mtp_response_summary("speculative_mtp", [detail])

    assert summary["mtp_output_tokens"] == 3
    assert summary["ar_output_tokens"] == 1
    assert summary["output_accounting"] == {
        "completion_tokens": 4,
        "mtp_output_tokens": 3,
        "ar_output_tokens": 1,
        "ar_output_tokens_in_cycles": 0,
        "mtp_coverage": 0.75,
        # The truncated tail emits fewer tokens than the cycles could explain,
        # so the committed cycles bound the speculative output rather than
        # matching it exactly.
        "mtp_output_tokens_explained_by_cycles": 5,
        # Negative: the committed cycles could have explained two more tokens
        # than the truncated tail emitted. Only a positive value is
        # over-attribution.
        "unexplained_mtp_output_tokens": -2,
        "speculative_cycles": 2,
        "reconciled": True,
        "reconciled_reasons": [],
    }


def test_mtp_summary_reports_partial_accounting_for_a_cancelled_request() -> None:
    """A cancelled request reports what it emitted, without a false fallback."""

    detail = _mtp_accounting_detail(
        generated_tokens=5,
        finish_details={"reason": "cancel", "cancelled": True},
        accounting=_full_coverage_accounting(
            cycles=1,
            generated_draft_tokens=3,
            accepted_draft_tokens=2,
            selected_depth_histogram={"3": 1},
            mtp_output_tokens=3,
            first_fallback_position=None,
        ),
    )

    summary = _mtp_response_summary("speculative_mtp", [detail])

    assert summary["used"] is True
    assert summary["mtp_output_tokens"] == 3
    assert summary["ar_output_tokens"] == 2
    assert summary["output_accounting"]["reconciled"] is True
    assert summary["first_fallback_position"] is None
    assert "decision_reason" not in summary


def test_mtp_summary_flags_an_unreconciled_accounting_block() -> None:
    """More speculative output than completion tokens is a backend bug."""

    detail = _mtp_accounting_detail(
        generated_tokens=4,
        accounting=_full_coverage_accounting(mtp_output_tokens=9),
    )

    summary = _mtp_response_summary("speculative_mtp", [detail])

    assert summary["output_accounting"]["reconciled"] is False
    assert summary["ar_output_tokens"] == 0


def test_mtp_metrics_count_owned_zero_draft_request() -> None:
    metrics = _ServerMetrics()

    metrics.record_success(
        {
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        }
    )

    assert metrics.mtp_requests_total == 1
    assert metrics.mtp_draft_tokens_accepted_total == 0
    assert metrics.mtp_draft_tokens_rejected_total == 0


def test_metrics_reports_mtp_serving_and_draft_acceptance() -> None:
    fake = SpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="opt_in",
            metrics="prometheus",
            max_active_requests=4,
        ),
        llm=fake,
    )
    client = TestClient(app)

    before = client.get("/metrics")
    assert before.status_code == 200
    # opt-in MTP with an engine that supports the route: effective serving is on.
    assert _metric_value(before.text, "hipengine_mtp_serving") == 1
    assert _metric_value(before.text, "hipengine_mtp_requests_total") == 0
    assert _metric_value(before.text, "hipengine_mtp_draft_tokens_accepted_total") == 0
    assert _metric_value(before.text, "hipengine_mtp_draft_tokens_rejected_total") == 0
    assert _metric_value(before.text, "hipengine_mtp_draft_acceptance_rate") == 0

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": ["one", "two"],
            "max_tokens": 3,
            "temperature": 0.0,
            "top_p": 1.0,
            "speculative_mtp": True,
        },
    )
    assert response.status_code == 200

    after = client.get("/metrics")
    assert _metric_value(after.text, "hipengine_mtp_requests_total") == 1
    assert _metric_value(after.text, "hipengine_mtp_draft_tokens_accepted_total") == 4
    assert _metric_value(after.text, "hipengine_mtp_draft_tokens_rejected_total") == 2
    assert _metric_value(after.text, "hipengine_mtp_draft_acceptance_rate") == 4 / 6


def test_mtp_summary_refuses_a_split_that_only_reconciles_arithmetically() -> None:
    """Arithmetic consistency is not attribution.

    ``ar_output_tokens`` is defined as ``completion - mtp``, so a response
    reporting five completion tokens, three speculative ones and ninety-nine
    in-cycle autoregressive ones satisfies ``mtp + ar == completion``. It is
    still impossible: the request emitted five tokens, so ninety-nine of them
    cannot have come from inside its cycles.
    """

    detail = _mtp_accounting_detail(
        generated_tokens=5,
        accounting=_full_coverage_accounting(
            cycles=1,
            generated_draft_tokens=3,
            accepted_draft_tokens=2,
            selected_depth_histogram={"3": 1},
            mtp_output_tokens=3,
            ar_output_tokens_in_cycles=99,
            ar_step_reason_counts={"no_provider": 99},
        ),
    )

    summary = _mtp_response_summary("speculative_mtp", [detail])

    accounting = summary["output_accounting"]
    # The arithmetic still holds, which is exactly why it proves nothing.
    assert accounting["mtp_output_tokens"] + accounting["ar_output_tokens"] == 5
    assert accounting["reconciled"] is False
    assert accounting["reconciled_reasons"] == [
        "in_cycle_ar_output_exceeds_ar_output"
    ]


def test_mtp_summary_refuses_speculative_output_its_cycles_cannot_explain() -> None:
    """Over-attribution: more speculative output than the cycles emitted."""

    detail = _mtp_accounting_detail(
        generated_tokens=9,
        accounting=_full_coverage_accounting(
            cycles=1,
            generated_draft_tokens=3,
            accepted_draft_tokens=1,
            selected_depth_histogram={"3": 1},
            mtp_output_tokens=7,
        ),
    )

    summary = _mtp_response_summary("speculative_mtp", [detail])

    accounting = summary["output_accounting"]
    assert accounting["mtp_output_tokens_explained_by_cycles"] == 2
    assert accounting["unexplained_mtp_output_tokens"] == 5
    assert accounting["reconciled"] is False
    assert accounting["reconciled_reasons"] == [
        "mtp_output_exceeds_committed_cycles"
    ]


def test_mtp_summary_refuses_speculative_output_without_committed_cycles() -> None:
    detail = _mtp_accounting_detail(
        generated_tokens=6,
        accounting=_full_coverage_accounting(
            cycles=0,
            generated_draft_tokens=0,
            accepted_draft_tokens=0,
            selected_depth_histogram={},
            mtp_output_tokens=6,
        ),
    )

    summary = _mtp_response_summary("speculative_mtp", [detail])

    assert summary["output_accounting"]["reconciled"] is False
    # No cycle ran at all, so it also cannot explain any speculative output.
    assert summary["output_accounting"]["reconciled_reasons"] == [
        "mtp_output_without_committed_cycles",
        "mtp_output_exceeds_committed_cycles",
    ]


def test_mtp_summary_proves_attribution_with_committed_output_spans(
    monkeypatch,
) -> None:
    """Diagnostic spans must tile the emitted output and match both counters."""

    monkeypatch.setenv("HIPENGINE_MTP2_OUTPUT_SPANS", "1")
    detail = _mtp_accounting_detail(
        generated_tokens=7,
        accounting=_full_coverage_accounting(
            cycles=3,
            generated_draft_tokens=6,
            accepted_draft_tokens=3,
            selected_depth_histogram={"3": 2, "0": 1},
            mtp_output_tokens=5,
            ar_output_tokens_in_cycles=1,
            output_spans=[
                {"mode": "mtp", "reason": "speculative_qualified", "position": 1, "tokens": 3},
                {"mode": "mtp", "reason": "speculative_qualified", "position": 4, "tokens": 2},
                {"mode": "ar", "reason": "target_graph_context_bucket_miss", "position": 6, "tokens": 1},
            ],
        ),
    )

    summary = _mtp_response_summary("speculative_mtp", [detail])

    accounting = summary["output_accounting"]
    assert accounting["reconciled"] is True
    assert accounting["reconciled_reasons"] == []
    spans = accounting["span_accounting"]
    assert spans == {
        "spans": 3,
        "tokens": 6,
        "mtp_tokens": 5,
        "ar_tokens": 1,
        "ar_tokens_by_reason": {"target_graph_context_bucket_miss": 1},
        "unspanned_tokens": 1,
        "expected_unspanned_ar_tokens": 1,
        "first_position": 1,
        "last_end": 7,
        "contiguous": True,
        "mtp_tokens_match": True,
        "ar_in_cycle_tokens_match": True,
        "unspanned_tokens_match": True,
        "reconciled": True,
        "reconciled_reasons": [],
    }


def test_mtp_summary_refuses_spans_that_do_not_tile_the_output(monkeypatch) -> None:
    """A gap, an overlap, or a short span list is a failed attribution proof."""

    monkeypatch.setenv("HIPENGINE_MTP2_OUTPUT_SPANS", "1")

    def _summary(spans: list[dict[str, Any]]) -> dict[str, Any]:
        detail = _mtp_accounting_detail(
            generated_tokens=7,
            accounting=_full_coverage_accounting(
                cycles=3,
                generated_draft_tokens=6,
                accepted_draft_tokens=3,
                selected_depth_histogram={"3": 2, "0": 1},
                mtp_output_tokens=5,
                ar_output_tokens_in_cycles=1,
                output_spans=spans,
            ),
        )
        return _mtp_response_summary("speculative_mtp", [detail])[
            "output_accounting"
        ]

    # A gap between the second and third span leaves position 6 unaccounted.
    gap = _summary(
        [
            {"mode": "mtp", "reason": "r", "position": 1, "tokens": 3},
            {"mode": "mtp", "reason": "r", "position": 4, "tokens": 2},
            {"mode": "ar", "reason": "r", "position": 7, "tokens": 1},
        ]
    )
    assert gap["reconciled"] is False
    # The spans cover every token the split accounts for, so the only failure is
    # the gap itself.
    assert gap["reconciled_reasons"] == ["span:spans_not_contiguous"]

    # Spans that stop short: the remainder must be the autoregressive output
    # emitted outside the plan, and here it is not.
    short = _summary(
        [
            {"mode": "mtp", "reason": "r", "position": 1, "tokens": 3},
            {"mode": "mtp", "reason": "r", "position": 4, "tokens": 2},
        ]
    )
    assert short["reconciled"] is False
    assert short["reconciled_reasons"] == [
        "span:span_ar_tokens_do_not_match_in_cycle_ar_output",
        "span:unspanned_tokens_are_not_autoregressive",
    ]

    # Spans that exceed the emitted output entirely.
    over = _summary(
        [
            {"mode": "mtp", "reason": "r", "position": 1, "tokens": 6},
            {"mode": "ar", "reason": "r", "position": 7, "tokens": 2},
        ]
    )
    assert over["reconciled"] is False
    assert over["reconciled_reasons"] == [
        "span:spans_exceed_the_completion_count",
        "span:span_mtp_tokens_do_not_match_mtp_output",
        "span:span_ar_tokens_do_not_match_in_cycle_ar_output",
        "span:unspanned_tokens_are_not_autoregressive",
    ]

    # A span list whose modes disagree with the reported split.
    mismatched = _summary(
        [
            {"mode": "mtp", "reason": "r", "position": 1, "tokens": 4},
            {"mode": "ar", "reason": "r", "position": 5, "tokens": 2},
        ]
    )
    assert mismatched["reconciled"] is False
    assert mismatched["reconciled_reasons"] == [
        "span:span_mtp_tokens_do_not_match_mtp_output",
        "span:span_ar_tokens_do_not_match_in_cycle_ar_output",
    ]

    # Recording enabled but nothing committed: the proof cannot be made, and
    # neither counter can be matched by an empty span list.
    empty = _summary([])
    assert empty["reconciled"] is False
    assert empty["reconciled_reasons"] == [
        "span:no_committed_spans",
        "span:span_mtp_tokens_do_not_match_mtp_output",
        "span:span_ar_tokens_do_not_match_in_cycle_ar_output",
        "span:unspanned_tokens_are_not_autoregressive",
    ]


def test_mtp_summary_omits_spans_unless_diagnostic_recording_is_on(
    monkeypatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_MTP2_OUTPUT_SPANS", raising=False)
    detail = _mtp_accounting_detail(
        generated_tokens=24,
        accounting=_full_coverage_accounting(),
    )

    summary = _mtp_response_summary("speculative_mtp", [detail])

    assert "span_accounting" not in summary["output_accounting"]
    assert summary["output_accounting"]["reconciled"] is True
