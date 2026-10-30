"""OpenAI server integration tests: routing.

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
    ArtifactScopedSpeculativeMTPFakeLLM,
    C2OnlyArtifactScopedSpeculativeMTPFakeLLM,
    DeclaredCellSpeculativeMTPFakeLLM,
    DefaultMTPDeclaredCellFakeLLM,
    OverDeclaredDepthSpeculativeMTPFakeLLM,
    PromotedArtifactScopedSpeculativeMTPFakeLLM,
    SpeculativeMTPFakeLLM,
)
from tests.support.server_api_support import (
    _test_static_eligibility,
)


def test_automatic_realized_mtp_propagates_typed_static_intent() -> None:
    sampling = SamplingParams(max_tokens=25)
    eligibility = _test_static_eligibility()
    route_decision = {
        "requested_route": _SPECULATIVE_MTP_AUTO_ROUTE,
        "selected_route": _SPECULATIVE_MTP_DEFAULT_ROUTE,
        "static_intent_allowed": True,
        "static_eligibility": eligibility.as_dict(),
    }

    automatic = _sampling_for_realized_generation_route(
        sampling,
        route_decision,
    )
    pure_k0 = _sampling_for_realized_generation_route(
        automatic,
        {
            **route_decision,
            "static_intent_allowed": False,
        },
    )

    assert automatic.speculative_mtp_static_eligibility == eligibility
    assert pure_k0.speculative_mtp_static_eligibility is None
    assert _execution_route_for_static_intent(
        _SPECULATIVE_MTP_DEFAULT_ROUTE,
        route_decision,
    ) == _SPECULATIVE_MTP_BATCH_ROUTE


def test_automatic_width_above_static_bound_becomes_pure_k0() -> None:
    eligibility = _test_static_eligibility()
    plan = {
        "admitted": True,
        "selected_candidate_count": 3,
        "automatic_eligible": True,
        "reason": eligibility.reason,
        "plan_fingerprint": "sha256:test-plan",
        "evidence_key": eligibility.evidence_key,
        "evidence_fingerprint": eligibility.evidence_fingerprint,
        "strict_fallback_key": eligibility.strict_fallback_key,
        "static_intent_allowed": True,
        "static_eligibility": eligibility.as_dict(),
        "key": {"realized_group_rows": 1},
    }
    selected, automatic = _serving_plan_route_decision(
        plan,
        requested_route=_SPECULATIVE_MTP_AUTO_ROUTE,
        group_rows=2,
        output_horizon_tokens=25,
    )
    _explicit_selected, explicit = _serving_plan_route_decision(
        plan,
        requested_route=_SPECULATIVE_MTP_BATCH_ROUTE,
        group_rows=2,
        output_horizon_tokens=25,
    )

    assert selected == _SPECULATIVE_MTP_DEFAULT_ROUTE
    assert automatic["reason"] == "physical_group_not_qualified"
    assert automatic["k0_class"] == "pure_k0"
    assert automatic["static_intent_allowed"] is False
    assert _execution_route_for_static_intent(selected, automatic) == (
        _SPECULATIVE_MTP_DEFAULT_ROUTE
    )
    assert explicit["k0_class"] == "transitional_k0"
    assert explicit["static_intent_allowed"] is True
    assert _execution_route_for_static_intent(selected, explicit) == (
        _SPECULATIVE_MTP_BATCH_ROUTE
    )


def test_prepared_context_tokens_finds_resident_model_owner_session() -> None:
    session = SimpleNamespace(max_sequence_length=8192)
    generator = SimpleNamespace(
        _session=None,
        _resident_model_runner=SimpleNamespace(_session=session),
    )
    engine = SimpleNamespace(_text_generator=generator)

    assert _prepared_context_tokens(engine) == 8192


@pytest.mark.parametrize("explicit,status", [(True, 501), (False, 200), (None, 200)])
def test_explicit_unimplemented_mtp_does_not_silently_downgrade(explicit, status):
    fake = SpeculativeMTPFakeLLM()
    fake.resolve_speculative_mtp_serving_plan = lambda **kwargs: {
        "admitted": False,
        "automatic_eligible": False,
        "reason": "dense_group_above_offered_width",
        "implementation_key": "gguf_dense_int8_gfx1151_group_native_chain",
        "selected_candidate_count": 0,
        "key": {"kv_storage": "int8_per_token_head", "realized_group_rows": 8},
    }
    app = create_app(
        ServerConfig(model="fake", served_model_name="fake-model", eager_load=False),
        llm=fake,
    )
    body = {"model": "fake-model", "prompt": "hi", "max_tokens": 4}
    if explicit is not None:
        body["speculative_mtp"] = explicit
    with TestClient(app) as client:
        response = client.post("/v1/completions", json=body)
    assert response.status_code == status, response.text
    if explicit is True:
        assert response.json()["error"]["code"] == "unsupported_feature"
        assert "dense_group_above_offered_width" in response.json()["error"]["message"]
        assert fake.mtp_calls == []
        assert fake.calls == []


def test_explicit_c2_intent_survives_request_time_c1_plan_rejection() -> None:
    fake = C2OnlyArtifactScopedSpeculativeMTPFakeLLM()
    request = CompletionRequest(
        model="fake-model",
        prompt="one",
        max_tokens=25,
        temperature=0.0,
        speculative_mtp=True,
    )

    route, plan = _generation_route_for_request(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="opt_in",
        ),
        request,
        engine=fake,
        sampling=SamplingParams(max_tokens=25),
        prompts=("one",),
    )

    assert route == _SPECULATIVE_MTP_BATCH_ROUTE
    assert plan is not None
    assert plan["admitted"] is False
    assert plan["reason"] == "physical_group_not_qualified"
    deferred_route, deferred = _resolve_realized_generation_route(
        route,
        group_rows=1,
        sampling=SamplingParams(max_tokens=25),
        precomputed_decision=plan,
    )
    assert deferred_route == _SPECULATIVE_MTP_BATCH_ROUTE
    assert deferred is not None
    assert deferred["reason"] == "physical_group_deferred_to_resident_owner"
    assert deferred["deferred_key_rows"] == 1


def test_explicit_width_deferred_plan_preserves_static_intent() -> None:
    """A width-miss deferral keeps the plan's evidence-backed static intent.

    The resident owner can only re-admit a deferred explicit request through
    its fail-closed physical admission when the deferred route decision still
    carries the static eligibility override resolved at request time. Dropping
    it turns every deferred C1 explicit request into pre-mutation K0.
    """

    plan = {
        "schema_version": 1,
        "plan_fingerprint": "sha256:" + "a" * 64,
        "key": {
            "realized_group_rows": 1,
            "context_tokens": 12,
            "output_horizon_tokens": 25,
        },
        "admitted": False,
        "selected_route": "default",
        "selected_candidate_count": 0,
        "reason": "physical_group_not_qualified",
        "strict_fallback_key": "gguf_target_ar",
        "evidence_key": "fake-qwen38-q4km-c2-b3",
        "evidence_artifacts": ["benchmarks/results/fake-qwen38-q4km-c2.json"],
        "automatic_eligible": False,
        "static_max_realized_group_rows": 2,
        "static_eligibility": {
            "state": "speculative_capable",
            "eligible": True,
            "reason": "qualified_automatic_c2_b3",
            "packed_c1_target": False,
            "max_candidate_count": 3,
            "max_realized_group_rows": 2,
            "automatic_eligible": False,
            "strict_fallback_key": "gguf_target_ar",
            "evidence_key": "fake-qwen38-q4km-c2-b3",
            "evidence_fingerprint": None,
            "evidence_artifacts": ["benchmarks/results/fake-qwen38-q4km-c2.json"],
        },
        "static_intent_allowed": True,
    }
    route, deferred = _resolve_realized_generation_route(
        _SPECULATIVE_MTP_BATCH_ROUTE,
        group_rows=1,
        sampling=SamplingParams(max_tokens=25),
        precomputed_decision=plan,
    )
    assert route == _SPECULATIVE_MTP_BATCH_ROUTE
    assert deferred is not None
    assert deferred["reason"] == "physical_group_deferred_to_resident_owner"
    assert deferred["static_intent_allowed"] is True
    assert deferred["static_eligibility"]["eligible"] is True
    assert deferred["static_eligibility"]["max_realized_group_rows"] == 2
    assert deferred["k0_class"] == "not_k0"

    denied = dict(plan, static_intent_allowed=False)
    _route, denied_deferred = _resolve_realized_generation_route(
        _SPECULATIVE_MTP_BATCH_ROUTE,
        group_rows=1,
        sampling=SamplingParams(max_tokens=25),
        precomputed_decision=denied,
    )
    assert denied_deferred is not None
    assert denied_deferred["static_intent_allowed"] is False


def test_reresolved_width_deferred_plan_derives_static_intent() -> None:
    """A re-resolved raw decision payload (no intent key) still grants intent.

    The batcher replaces the request-time plan with the raw
    SpeculativeMTPServingDecision.as_dict() payload, which carries the static
    eligibility override but not the plan layer's static_intent_allowed key.
    The deferral must derive intent from the eligible payload, because this
    branch only fires for explicit batch-route width misses.
    """

    plan = {
        "schema_version": 1,
        "plan_fingerprint": "sha256:" + "b" * 64,
        "key": {
            "realized_group_rows": 1,
            "context_tokens": 12,
            "output_horizon_tokens": 25,
        },
        "admitted": False,
        "selected_route": "default",
        "selected_candidate_count": 0,
        "reason": "physical_group_not_qualified",
        "strict_fallback_key": "gguf_target_ar",
        "evidence_key": "fake-qwen38-q4km-c2-b3",
        "evidence_artifacts": ["benchmarks/results/fake-qwen38-q4km-c2.json"],
        "automatic_eligible": False,
        "static_max_realized_group_rows": 2,
        "static_eligibility": {
            "state": "speculative_capable",
            "eligible": True,
            "reason": "qualified_automatic_c2_b3",
            "packed_c1_target": False,
            "max_candidate_count": 3,
            "max_realized_group_rows": 2,
            "automatic_eligible": False,
            "strict_fallback_key": "gguf_target_ar",
            "evidence_key": "fake-qwen38-q4km-c2-b3",
            "evidence_fingerprint": None,
            "evidence_artifacts": ["benchmarks/results/fake-qwen38-q4km-c2.json"],
        },
    }
    route, deferred = _resolve_realized_generation_route(
        _SPECULATIVE_MTP_BATCH_ROUTE,
        group_rows=1,
        sampling=SamplingParams(max_tokens=25),
        precomputed_decision=plan,
    )
    assert route == _SPECULATIVE_MTP_BATCH_ROUTE
    assert deferred is not None
    assert deferred["reason"] == "physical_group_deferred_to_resident_owner"
    assert deferred["static_intent_allowed"] is True
    assert deferred["static_eligibility"]["eligible"] is True


def test_explicit_c1_plan_drops_intent_when_rejection_is_not_a_width_miss() -> None:
    """Only a physical width miss keeps typed intent for the resident owner.

    A sampling-mode rejection is decided before mutation, so the request falls
    out of the speculative route instead of being deferred.
    """

    fake = ArtifactScopedSpeculativeMTPFakeLLM()
    request = CompletionRequest(
        model="fake-model",
        prompt="one",
        max_tokens=25,
        temperature=0.4,
        speculative_mtp=True,
    )
    with pytest.raises(OpenAIHTTPError, match="temperature"):
        _generation_route_for_request(
            ServerConfig(
                model="fake-path",
                served_model_name="fake-model",
                speculative_mtp_serving="opt_in",
            ),
            request,
            engine=fake,
            sampling=SamplingParams(max_tokens=25, temperature=0.4),
            prompts=("one",),
        )


def test_explicit_unlisted_cell_resolves_through_the_declaration() -> None:
    """C3/K2 is absent from the gfx1100 table and still executes on request.

    The server no longer manufactures a screening eligibility for a rejected
    plan: the model plugin's declaration admits the cell, so typed intent comes
    from the ordinary plan and the response carries no screening marker.
    """

    fake = DeclaredCellSpeculativeMTPFakeLLM()
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        speculative_mtp_serving="opt_in",
    )
    sampling = SamplingParams(max_tokens=25)
    prompts = ("one", "two", "three")
    route, plan = _generation_route_for_request(
        config,
        CompletionRequest(
            model="fake-model", prompt="one", max_tokens=25, speculative_mtp=True
        ),
        engine=fake,
        sampling=sampling,
        prompts=prompts,
    )

    assert route == _SPECULATIVE_MTP_BATCH_ROUTE
    assert plan is not None
    assert plan["admitted"] is True
    assert plan["admission_basis"] == "implementation"
    assert plan["implementation_key"] == (
        "gguf_dense_bf16_gfx1100_group_native_chain"
    )
    assert plan["static_intent_allowed"] is True
    assert "screening" not in plan
    assert "qualification" not in plan

    realized_route, decision = _resolve_realized_generation_route(
        route,
        group_rows=len(prompts),
        sampling=sampling,
        precomputed_decision=plan,
    )
    assert realized_route == _SPECULATIVE_MTP_BATCH_ROUTE
    assert decision is not None
    assert decision["k0_class"] == "not_k0"
    assert decision["static_intent_allowed"] is True
    assert "screening" not in decision
    assert "qualification" not in decision

    summary = _mtp_response_summary(
        realized_route,
        None,
        route_decision=decision,
        thinking_policy="hint",
    )
    assert "qualification" not in summary
    assert "unqualified" not in summary


def test_explicit_capability_miss_names_the_axis_and_never_screens() -> None:
    """A group wider than the declaration is refused, not admitted by override."""

    fake = DeclaredCellSpeculativeMTPFakeLLM()
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        speculative_mtp_serving="opt_in",
    )
    sampling = SamplingParams(max_tokens=25)
    prompts = tuple(str(index) for index in range(9))

    with pytest.raises(OpenAIHTTPError, match="dense_group_above_offered_width"):
        _generation_route_for_request(
            config,
            CompletionRequest(
                model="fake-model",
                prompt="one",
                max_tokens=25,
                speculative_mtp=True,
            ),
            engine=fake,
            sampling=sampling,
            prompts=prompts,
        )


@pytest.mark.parametrize("arrival", ["explicit", "automatic"])
def test_enabled_mode_intent_never_shares_a_batch_group(arrival: str) -> None:
    """An explicit and an automatic request must not coalesce in `enabled` mode.

    The batcher applies one route decision -- and therefore one MTP intent -- to
    every row of a group, so a group holding both intents would either widen
    automatic policy onto an unlisted cell or withhold an explicit request's
    declared cell. Reproduced finding: both requests route `speculative_mtp`,
    their plan fingerprints match because the fingerprint excludes
    `request_mode`, and their sampling keys match, so arrival order alone
    decided which intent the whole group ran.
    """

    fake = DefaultMTPDeclaredCellFakeLLM()
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        speculative_mtp_serving="enabled",
    )
    sampling = SamplingParams(max_tokens=25)
    requests = {
        "explicit": CompletionRequest(
            model="fake-model", prompt="explicit", max_tokens=25, speculative_mtp=True
        ),
        "automatic": CompletionRequest(
            model="fake-model", prompt="automatic", max_tokens=25
        ),
    }
    built = {}
    for label, request in requests.items():
        route, plan = _generation_route_for_request(
            config,
            request,
            engine=fake,
            sampling=sampling,
            prompts=(request.prompt,),
        )
        assert route == _SPECULATIVE_MTP_BATCH_ROUTE
        assert plan is not None
        built[label] = _QueuedGeneration(
            prompts=(request.prompt,),
            sampling=sampling,
            route=route,
            route_decision=plan,
        )

    first = built[arrival]
    second = built["automatic" if arrival == "explicit" else "explicit"]
    # Both are interchangeable on every key the batcher used before intent:
    # same route, same plan payload (the fingerprint omits request_mode), same
    # sampling.
    assert first.route_decision["plan_fingerprint"] == second.route_decision[
        "plan_fingerprint"
    ]
    assert _sampling_key(
        first.sampling, include_cancellation_token=False
    ) == _sampling_key(second.sampling, include_cancellation_token=False)
    assert first.route_decision["request_mode"] != second.route_decision[
        "request_mode"
    ]

    batcher = _GenerationBatcher(engine_factory=lambda: fake, batch_window_seconds=0.0)
    assert batcher._group_key(first) != batcher._group_key(second)

    # The partition is intent, not request identity: a second automatic request
    # still coalesces with the automatic one.
    peer_route, peer_plan = _generation_route_for_request(
        config,
        CompletionRequest(model="fake-model", prompt="peer", max_tokens=25),
        engine=fake,
        sampling=sampling,
        prompts=("peer",),
    )
    peer = _QueuedGeneration(
        prompts=("peer",),
        sampling=sampling,
        route=peer_route,
        route_decision=peer_plan,
    )
    assert batcher._group_key(peer) == batcher._group_key(built["automatic"])

    # Each request's own decision is what its own group runs.
    for item in (first, second):
        realized = _realized_model_serving_plan(
            fake, item.prompts, item.sampling, item.route_decision
        )
        _, decision = _resolve_realized_generation_route(
            item.route,
            group_rows=len(item.prompts),
            sampling=item.sampling,
            precomputed_decision=realized,
        )
        group_sampling = _sampling_for_realized_generation_route(
            item.sampling, decision
        )
        assert group_sampling.speculative_mtp_automatic is (
            item.route_decision["request_mode"] == "automatic"
        )


def test_explicit_depth_above_the_declaration_is_a_capability_refusal() -> None:
    """K4 is a declared-shape miss, so no override may admit it."""

    fake = OverDeclaredDepthSpeculativeMTPFakeLLM()
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        speculative_mtp_serving="opt_in",
    )

    with pytest.raises(OpenAIHTTPError, match="mtp_candidate_depth_unsupported"):
        _generation_route_for_request(
            config,
            CompletionRequest(
                model="fake-model",
                prompt="one",
                max_tokens=25,
                speculative_mtp=True,
            ),
            engine=fake,
            sampling=SamplingParams(max_tokens=25),
            prompts=("one", "two"),
        )


def test_automatic_route_rejects_any_explicit_only_realized_plan() -> None:
    selected, decision = _serving_plan_route_decision(
        {
            "schema_version": 1,
            "plan_fingerprint": "sha256:" + "a" * 64,
            "key": {"realized_group_rows": 2},
            "admitted": True,
            "selected_candidate_count": 2,
            "reason": "qualified_explicit_dense_c2_k2_d24",
            "evidence_key": "fake-explicit-c2",
            "evidence_artifacts": ["benchmarks/results/fake-explicit-c2.json"],
            "automatic_eligible": False,
        },
        requested_route=_SPECULATIVE_MTP_AUTO_ROUTE,
        group_rows=2,
        output_horizon_tokens=24,
    )

    assert selected == _SPECULATIVE_MTP_DEFAULT_ROUTE
    assert decision["reason"] == "automatic_mtp_scope_not_promoted"
    assert decision["selected_candidate_count"] == 0


def test_completions_endpoint_routes_explicit_speculative_mtp_request() -> None:
    fake = SpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="opt_in",
            max_active_requests=4,
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
            "speculative_mtp": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert [choice["text"] for choice in body["choices"]] == ["mtp:one", "mtp:two"]
    assert fake.calls == []
    assert len(fake.mtp_calls) == 1
    prompts, sampling = fake.mtp_calls[0]
    assert prompts == ("one", "two")
    assert sampling.max_tokens == 3
    assert sampling.temperature == 0.0
    assert sampling.top_p == 1.0
    assert body["hipengine"]["generation_shape"] == {
        "schema_version": 1,
        "route": "speculative_mtp",
        "route_cap": {
            "scope": "queue_requests",
            "value": 4,
            "applied": True,
        },
        "queue_group": {
            "id": body["hipengine"]["generation_shape"]["queue_group"]["id"],
            "request_count": 1,
            "prompt_rows": 2,
            "item_index": 0,
            "item_prompt_offset": 0,
            "item_prompt_rows": 2,
        },
        "backend_groups": [
            {
                "id": body["hipengine"]["generation_shape"]["backend_groups"][0]["id"],
                "call_index": 0,
                "prompt_offset": 0,
                "input_rows": 2,
                "actual_group_rows": [2],
                "max_actual_group_rows": 2,
                "verifier_rows": 6,
            }
        ],
        "verifier_rows": 6,
    }
    assert body["usage"]["completion_tokens_details"] == {
        "accepted_prediction_tokens": 4,
        "rejected_prediction_tokens": 2,
    }
    assert body["hipengine"]["speculative_mtp"] == {
        "effective_route": "speculative_mtp",
        "used": True,
        "draft_tokens": 6,
        "accepted_draft_tokens": 4,
        "rejected_draft_tokens": 2,
        "acceptance_rate": 4 / 6,
        "draft_cycles": 2,
        "thinking_policy": "hint",
        "thinking_controls": "prompt_hint_only",
    }


def test_artifact_scoped_explicit_mtp_uses_one_plan_in_capability_route_and_response() -> None:
    fake = ArtifactScopedSpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="opt_in",
            max_active_requests=1,
            max_context_tokens=1024,
        ),
        llm=fake,
    )
    client = TestClient(app)

    capability = client.get("/v1/hipengine/capabilities").json()["sampling"]["speculative_mtp"]
    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 25,
            "temperature": 0.0,
            "top_p": 1.0,
            "speculative_mtp": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    plan = capability["certified_explicit_scope"]
    decision = body["hipengine"]["generation_shape"]["route_decision"]
    assert plan["admitted"] is True
    assert plan["automatic_eligible"] is False
    assert body["hipengine"]["generation_shape"]["route"] == "speculative_mtp"
    assert decision["policy_fingerprint"] == plan["plan_fingerprint"]
    assert decision["policy_cell"] == plan["evidence_key"]
    assert decision["selected_candidate_count"] == 3
    assert body["hipengine"]["speculative_mtp"]["used"] is True
    assert len(fake.mtp_calls) == 1

    rollback = client.post("/v1/hipengine/speculative_mtp/rollback").json()
    assert rollback["serving_plan"]["plan_fingerprint"] == plan["plan_fingerprint"]


def test_artifact_scoped_explicit_mtp_rejects_unsupported_sampling_before_mutation() -> None:
    fake = ArtifactScopedSpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="opt_in",
            max_active_requests=1,
            max_context_tokens=1024,
        ),
        llm=fake,
    )
    # A sampling-mode miss still fails closed before backend mutation. Shape
    # (prompt context and output horizon) is deliberately not an admission
    # axis, so a long prompt alone no longer selects K0.
    prompt = " ".join(f"token{index}" for index in range(68))

    response = TestClient(app).post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": prompt,
            "max_tokens": 25,
            "temperature": 0.4,
            "speculative_mtp": True,
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "unsupported_parameter"
    assert body["error"]["param"] == "speculative_mtp"
    assert "temperature" in body["error"]["message"]
    assert fake.mtp_calls == []
    assert fake.calls == []


def test_auto_and_enabled_share_explicit_only_plan_fingerprint_and_k0() -> None:
    fingerprints = []
    for mode in ("auto", "enabled"):
        fake = ArtifactScopedSpeculativeMTPFakeLLM()
        app = create_app(
            ServerConfig(
                model="fake-path",
                served_model_name="fake-model",
                speculative_mtp_serving=mode,
                max_active_requests=1,
                max_context_tokens=1024,
            ),
            llm=fake,
        )
        client = TestClient(app)
        capability = client.get("/v1/hipengine/capabilities").json()["sampling"]["speculative_mtp"]
        response = client.post(
            "/v1/completions",
            json={
                "model": "fake-model",
                "prompt": "hello",
                "max_tokens": 25,
                "temperature": 0.0,
            },
        )

        assert response.status_code == 200
        body = response.json()
        plan = capability["certified_explicit_scope"]
        decision = body["hipengine"]["generation_shape"]["route_decision"]
        assert capability["default_enabled"] is False
        assert capability["automatic_route_promoted"] is False
        assert body["hipengine"]["generation_shape"]["route"] == "default"
        assert decision["reason"] == "automatic_mtp_scope_not_promoted"
        assert decision["policy_fingerprint"] == plan["plan_fingerprint"]
        assert fake.mtp_calls == []
        fingerprints.append(plan["plan_fingerprint"])

    assert fingerprints[0] == fingerprints[1]


def test_promoted_artifact_plan_routes_auto_for_every_shape_in_the_cell() -> None:
    """Shape is not an admission axis, so the promoted cell covers all shapes."""

    fake = PromotedArtifactScopedSpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="auto",
            max_active_requests=1,
            max_context_tokens=1024,
        ),
        llm=fake,
    )
    client = TestClient(app)

    capability = client.get("/v1/hipengine/capabilities").json()["sampling"]["speculative_mtp"]
    auto = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 25,
            "temperature": 0.0,
        },
    ).json()
    outside = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": " ".join(f"token{index}" for index in range(68)),
            "max_tokens": 25,
            "temperature": 0.0,
        },
    ).json()
    incompatible = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 25,
            "temperature": 0.4,
        },
    ).json()
    disabled = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 25,
            "temperature": 0.0,
            "speculative_mtp": False,
        },
    ).json()

    plan = capability["certified_explicit_scope"]
    assert capability["automatic_route_promoted"] is True
    assert capability["default_enabled"] is True
    assert capability["auto_route"]["selected_route"] == "speculative_mtp"
    assert capability["auto_route"]["selected_candidate_count"] == 3
    assert capability["auto_route"]["policy_fingerprint"] == plan["plan_fingerprint"]
    assert auto["hipengine"]["generation_shape"]["route"] == "speculative_mtp"
    assert auto["hipengine"]["generation_shape"]["route_decision"]["policy_fingerprint"] == plan["plan_fingerprint"]
    assert outside["hipengine"]["generation_shape"]["route"] == "speculative_mtp"
    assert outside["hipengine"]["generation_shape"]["route_decision"]["policy_fingerprint"] == plan["plan_fingerprint"]
    assert incompatible["hipengine"]["generation_shape"]["route"] == "default"
    assert incompatible["hipengine"]["generation_shape"]["route_decision"]["reason"] == "sampling_mode_not_qualified"
    assert disabled["hipengine"]["generation_shape"]["route"] == "default"


def test_chat_endpoint_routes_explicit_mtp_and_reports_direct_usage() -> None:
    fake = SpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="opt_in",
        ),
        llm=fake,
    )
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
            "temperature": 0.0,
            "top_p": 1.0,
            "speculative_mtp": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(fake.mtp_calls) == 1
    assert body["hipengine"]["generation_shape"]["route"] == "speculative_mtp"
    assert body["usage"]["completion_tokens_details"] == {
        "accepted_prediction_tokens": 2,
        "rejected_prediction_tokens": 1,
    }
    assert body["hipengine"]["speculative_mtp"] == {
        "effective_route": "speculative_mtp",
        "used": True,
        "draft_tokens": 3,
        "accepted_draft_tokens": 2,
        "rejected_draft_tokens": 1,
        "acceptance_rate": 2 / 3,
        "draft_cycles": 1,
        "thinking_policy": "hint",
        "thinking_controls": "prompt_hint_only",
    }


def test_chat_auto_fallback_reports_stable_route_reason() -> None:
    fake = SpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=fake,
    )
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
            "temperature": 0.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert fake.mtp_calls == []
    assert body["hipengine"]["generation_shape"]["route"] == "default"
    assert body["hipengine"]["speculative_mtp"] == {
        "effective_route": "default",
        "used": False,
        "draft_tokens": 0,
        "accepted_draft_tokens": 0,
        "rejected_draft_tokens": 0,
        "acceptance_rate": None,
        "draft_cycles": 0,
        "requested_route": "speculative_mtp_auto",
        "decision_reason": "automatic_mtp_scope_not_promoted",
    }


def test_completions_default_auto_keeps_compatibility_mtp_explicit_only() -> None:
    fake = SpeculativeMTPFakeLLM()
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        max_active_requests=4,
    )
    assert config.speculative_mtp_serving == "auto"
    app = create_app(
        config,
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": ["one", "two"],
            "max_tokens": 24,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert [choice["text"] for choice in body["choices"]] == [
        "generated:one",
        "generated:two",
    ]
    assert fake.mtp_calls == []
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == ("one", "two")
    shape = body["hipengine"]["generation_shape"]
    assert shape["route"] == "default"
    assert shape["route_decision"] == {
        "requested_route": "speculative_mtp_auto",
        "selected_route": "default",
        "reason": "automatic_mtp_scope_not_promoted",
        "policy_cell": "auto-c2-measured-k0",
        "selected_candidate_count": 0,
        "policy_reason": "measured_speedup_below_1p10",
        "policy_fingerprint": DEFAULT_AUTO_DEPTH_POLICY.fingerprint,
        "realized_group_rows": 2,
        "output_horizon_tokens": 24,
        "exact_default_required": True,
        "evidence": "benchmarks/results/2026-08-26-gfx1151-specdec2-perf-p9-fixed-policy.json",
    }
    explicit_response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "three",
            "max_tokens": 24,
            "temperature": 0.0,
            "top_p": 1.0,
            "speculative_mtp": True,
        },
    )
    assert explicit_response.status_code == 200
    assert explicit_response.json()["choices"][0]["text"] == "mtp:three"
    assert len(fake.mtp_calls) == 1
    assert fake.mtp_calls[0][0] == ("three",)


def test_completions_enabled_mode_requires_model_plugin_default_evidence() -> None:
    fake = SpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="enabled",
            max_active_requests=4,
        ),
        llm=fake,
    )
    client = TestClient(app)

    # Automatic admission requires immutable model-plugin evidence.
    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": ["one", "two"],
            "max_tokens": 3,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert [choice["text"] for choice in body["choices"]] == [
        "generated:one",
        "generated:two",
    ]
    assert fake.mtp_calls == []
    assert body["hipengine"]["generation_shape"]["route"] == "default"
    assert body["hipengine"]["generation_shape"]["route_decision"]["reason"] == (
        "model_plugin_scope_not_qualified"
    )

    # Existing operator-selected explicit compatibility remains independent.
    explicit_response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "three",
            "max_tokens": 3,
            "temperature": 0.0,
            "top_p": 1.0,
            "speculative_mtp": True,
        },
    )
    assert explicit_response.status_code == 200
    assert explicit_response.json()["choices"][0]["text"] == "mtp:three"
    assert len(fake.mtp_calls) == 1


def test_completions_endpoint_rejects_unsupported_explicit_non_greedy_mtp() -> None:
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

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "one",
            "max_tokens": 3,
            "temperature": 0.4,
            "speculative_mtp": True,
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "unsupported_parameter"
    assert body["error"]["param"] == "speculative_mtp"
    assert "temperature" in body["error"]["message"]
    assert fake.calls == []
    assert fake.mtp_calls == []


@pytest.mark.parametrize(
    "payload,token_map,blocker",
    [
        ({"stop": ["\n\n"]}, {"one": [1], "\n\n": [5, 6]}, "stop_token_sequences"),
        ({"stop": ["<eos>"]}, {"one": [1], "<eos>": [9]}, "stop_token_ids"),
        ({"stop": ["zzz"]}, {"one": [1]}, None),
        ({"min_tokens": 2, "eos_token_id": 9}, None, "min_tokens"),
        ({"eos_token_id": 9}, None, "eos_token_id"),
        ({"temperature": 0.4, "eos_token_id": 9}, None, "eos_token_id"),
        ({"ignore_eos": True}, None, "ignore_eos"),
        ({"suppress_token_ids": [7]}, None, "suppress_token_ids"),
    ],
)
def test_explicit_mtp_rejects_unimplemented_finish_semantics(
    payload: dict[str, object], token_map: dict[str, list[int]] | None, blocker: str | None
) -> None:
    """A greedy-only provider refuses controls instead of substituting AR.

    An unlowerable text stop is handled after generation and keeps MTP.
    """

    fake = SpeculativeMTPFakeLLM(token_map=token_map)
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="opt_in",
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "one", "max_tokens": 3, "speculative_mtp": True}
        | payload,
    )

    if blocker is None:
        assert response.status_code == 200
        assert response.json()["hipengine"]["generation_shape"]["route"] == "speculative_mtp"
        assert fake.mtp_calls
        return
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unsupported_parameter"
    assert error["param"] == "speculative_mtp"
    assert blocker in error["message"]
    assert fake.mtp_calls == []
    assert fake.calls == []


def test_chat_completion_hint_thinking_policy_routes_through_mtp_and_relaxes_sampling() -> None:
    fake = SpeculativeMTPFakeLLM(token_map={_THINKING_CLOSE_MARKER: [42, 43]})
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="opt_in",
            speculative_mtp_thinking="hint",
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think then answer"}],
            "reasoning_effort": "medium",
            "max_tokens": 100,
            "temperature": 0.0,
            "top_p": 1.0,
            "speculative_mtp": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["hipengine"]["generation_shape"]["route"] == "speculative_mtp"
    assert len(fake.mtp_calls) == 1
    # The MTP route relaxes host-sampler thinking enforcement while keeping the
    # thinking hints in the rendered prompt.
    sampling = fake.mtp_calls[0][1]
    assert sampling.thinking_close_token_ids == ()
    assert sampling.thinking_hard_token_cap is None
    assert sampling.thinking_soft_close_window == 0
    assert "hidden reasoning tokens" in str(fake.mtp_calls[0][0][0])


def test_chat_completion_hard_thinking_policy_falls_back_to_ar_and_keeps_enforcement() -> None:
    fake = SpeculativeMTPFakeLLM(token_map={_THINKING_CLOSE_MARKER: [42, 43]})
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="enabled",
            speculative_mtp_thinking="hard",
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think then answer"}],
            "reasoning_effort": "medium",
            "max_tokens": 100,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["hipengine"]["generation_shape"]["route"] == "default"
    assert fake.mtp_calls == []
    assert len(fake.calls) == 1
    # Hard policy keeps full host-sampler thinking enforcement on the AR path.
    sampling = fake.calls[-1][1]
    assert sampling.thinking_close_token_ids == (42, 43)
    assert sampling.thinking_hard_token_cap == 48
    assert sampling.thinking_soft_close_window == 48


def test_chat_completion_thinking_policy_request_override_selects_policy() -> None:
    # Config is hard but the request opts in to hint -> MTP route.
    fake = SpeculativeMTPFakeLLM(token_map={_THINKING_CLOSE_MARKER: [42, 43]})
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="enabled",
            speculative_mtp_thinking="hard",
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think then answer"}],
            "reasoning_effort": "medium",
            "max_tokens": 100,
            "temperature": 0.0,
            "top_p": 1.0,
            "speculative_mtp": {"enabled": True, "thinking": "hint"},
        },
    )

    assert response.status_code == 200
    assert response.json()["hipengine"]["generation_shape"]["route"] == "speculative_mtp"
    assert len(fake.mtp_calls) == 1

    # Config is hint but the explicit request opts in to hard; this greedy-only
    # fake must refuse rather than discard enforcement or substitute AR.
    fake2 = SpeculativeMTPFakeLLM(token_map={_THINKING_CLOSE_MARKER: [42, 43]})
    app2 = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="enabled",
            speculative_mtp_thinking="hint",
        ),
        llm=fake2,
    )
    client2 = TestClient(app2)

    response = client2.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think then answer"}],
            "reasoning_effort": "medium",
            "max_tokens": 100,
            "temperature": 0.0,
            "top_p": 1.0,
            "speculative_mtp": {"enabled": True, "thinking": "hard"},
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "unsupported_parameter"
    assert "thinking_budget" in body["error"]["message"]
    assert fake2.mtp_calls == []
    assert fake2.calls == []
