"""OpenAI server integration tests: startup.

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
    LateResolvingMTPFakeLLM,
    NarrowCapacityFakeLLM,
    ScratchProbeFailureFakeLLM,
    SequentialFakeLLM,
    SpeculativeMTPFakeLLM,
    WideScratchProbeFakeLLM,
    _FakeCapacityEstimate,
)


def test_unprepared_engine_logs_no_mtp_state_yet(caplog) -> None:
    """A loading engine says nothing about MTP; the resolved line carries it.

    RED for the pre-resolution echo: reporting ``serving=off
    engine_supported=False`` while the generator is still being prepared
    contradicts the ``serving=enabled engine_supported=True`` line the operator
    reads once the model is resident. Reporting ``pending`` there is the same
    contradiction, softened: the engine is already loading, so the requested
    values are the only content and the resolved line follows on its own.
    """

    from hipengine.server.api import _log_effective_mtp_config

    class UnpreparedEngine:
        """An engine whose generator - and MTP capability - is not resolved."""

        _text_generator = None

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        logged = _log_effective_mtp_config(
            ServerConfig(
                model="fake-path",
                served_model_name="fake-model",
                eager_load=False,
                speculative_mtp_serving="auto",
                speculative_candidate_budget=3,
            ),
            engine=UnpreparedEngine(),
        )

    assert logged is False
    assert "EFFECTIVE_MTP" not in caplog.text


def test_missing_engine_logs_no_mtp_state(caplog) -> None:
    """No engine at all is not a resolved state either; lazy startup says pending."""

    from hipengine.server.api import _log_effective_mtp_config

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        logged = _log_effective_mtp_config(
            ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
            engine=None,
        )

    assert logged is False
    assert "EFFECTIVE_MTP" not in caplog.text


def test_lazy_startup_pending_line_defers_to_the_first_request(caplog, monkeypatch) -> None:
    """Lazy startup states the request, then the load states the outcome.

    Exactly two field lines: the pending form at startup, when no engine exists,
    and the resolved form once the first request prepares the model. The request
    itself must not repeat either one.
    """

    caplog.set_level(logging.INFO, logger="uvicorn.error")
    fake = LateResolvingMTPFakeLLM()
    monkeypatch.setattr("hipengine.server.api.LLM", lambda *args, **kwargs: fake)
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            speculative_mtp_serving="enabled",
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "fake-model",
                "prompt": "one",
                "max_tokens": 1,
                "temperature": 0.0,
                "speculative_mtp": False,
            },
        )
        second = client.post(
            "/v1/completions",
            json={
                "model": "fake-model",
                "prompt": "two",
                "max_tokens": 1,
                "temperature": 0.0,
                "speculative_mtp": False,
            },
        )

    assert response.status_code == 200
    assert second.status_code == 200
    assert fake.prepares, "the request path must prepare the model it reports on"
    field_lines = [line for line in caplog.text.splitlines() if "EFFECTIVE_MTP: serving=" in line]
    assert len(field_lines) == 2, caplog.text
    assert "serving=pending engine_supported=unknown" in field_lines[0]
    assert "serving=enabled engine_supported=True" in field_lines[1]


def test_eager_startup_logs_one_resolved_mtp_line(caplog, monkeypatch) -> None:
    """Eager startup prints the resolved state and never a pending echo.

    The engine is loading for the whole startup, so a pending line would be a
    second field line that the resolved one immediately supersedes.
    """

    caplog.set_level(logging.INFO, logger="uvicorn.error")
    fake = LateResolvingMTPFakeLLM()
    monkeypatch.setattr("hipengine.server.api.LLM", lambda *args, **kwargs: fake)
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="enabled",
        )
    )

    with TestClient(app):
        pass

    assert fake.prepares, "eager startup must prepare the model it reports on"
    field_lines = [line for line in caplog.text.splitlines() if "EFFECTIVE_MTP: serving=" in line]
    assert len(field_lines) == 1, caplog.text
    assert "serving=enabled engine_supported=True" in field_lines[0]


def test_resolved_engine_without_mtp_route_still_logs_off(caplog) -> None:
    """A resolved engine that has no MTP route reports off, as before."""

    from hipengine.server.api import _log_effective_mtp_config

    class ResolvedPlainEngine:
        _text_generator = object()

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        _log_effective_mtp_config(
            ServerConfig(
                model="fake-path",
                served_model_name="fake-model",
                eager_load=False,
                speculative_mtp_serving="auto",
            ),
            engine=ResolvedPlainEngine(),
        )

    assert "serving=off engine_supported=False" in caplog.text


def test_startup_warns_when_enabled_mtp_cannot_serve_its_budget(caplog) -> None:
    """An explicitly enabled route that cannot admit is reported, not silent."""

    from hipengine.server.api import _log_effective_mtp_config

    fake = SpeculativeMTPFakeLLM()
    fake.speculative_mtp_serving_capability = {
        "admitted": False,
        "automatic_eligible": False,
        "reason": "candidate_budget_not_qualified",
    }
    fake.speculative_candidate_budget_requested = 4
    fake.speculative_candidate_budget = 4
    fake.speculative_candidate_budget_source = "explicit"

    with caplog.at_level(logging.WARNING):
        _log_effective_mtp_config(
            ServerConfig(
                model="fake-path",
                served_model_name="fake-model",
                eager_load=False,
                speculative_mtp_serving="enabled",
                speculative_candidate_budget=4,
            ),
            engine=fake,
        )

    assert any(
        "candidate_budget_not_qualified" in record.message
        and "candidate_budget requested:4 resolved:4 source:explicit" in record.message
        for record in caplog.records
    )


def test_lazy_startup_logs_pending_then_effective_mtp_state(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    fake = SpeculativeMTPFakeLLM()
    monkeypatch.setattr("hipengine.server.api.LLM", lambda *args, **kwargs: fake)
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            speculative_mtp_serving="enabled",
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "fake-model",
                "prompt": "one",
                "max_tokens": 1,
                "temperature": 0.0,
                "speculative_mtp": False,
            },
        )

    assert response.status_code == 200
    assert (
        "EFFECTIVE_MTP: serving=pending engine_supported=unknown "
        "default_enabled=unknown policy=enabled thinking=hint"
    ) in caplog.text
    assert (
        "EFFECTIVE_MTP: serving=enabled engine_supported=True "
        "default_enabled=False policy=enabled thinking=hint"
    ) in caplog.text


def test_server_eager_loads_model_on_startup(caplog) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    fake = FakeLLM(outputs=["warm"])
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        eager_load_prompt="one two three four",
        eager_load_max_tokens=2,
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert "Config: model=fake-path" in caplog.text
    assert "MODEL_LOAD: loading model=fake-path" in caplog.text
    assert "MODEL_LOAD: engine created elapsed=" in caplog.text
    assert "max_context_tokens=131072" in caplog.text
    assert "chat_default_max_tokens=4096" in caplog.text
    assert "kv_storage=auto" in caplog.text
    assert "KVCache: storage=bf16" in caplog.text
    assert "model_max_context_tokens=262144" in caplog.text
    assert "WARMUP: prompt_tokens<=131072 max_tokens=2" in caplog.text
    assert "STARTUP_SCRATCH_PROBE: max_prompt_tokens=131071" in caplog.text
    assert "WARMUP_CHAT: prompt_tokens=" in caplog.text
    assert "LOAD_TIMING: phase=startup resident_prepare_s=" in caplog.text
    assert "LOAD_TIMING: model=fake-model engine_create_s=" in caplog.text
    assert "warmup_s=" in caplog.text
    assert "scratch_probe_s=" in caplog.text
    assert "chat_smoke_s=" in caplog.text
    assert "hipEngine is ready." in caplog.text
    assert fake.prepares == [
        (
            None,
            SamplingParams(max_tokens=2, temperature=0.0, top_p=1.0, ignore_eos=True),
        )
    ]
    assert fake.scratch_prepares == [
        {
            "max_prompt_tokens": 131071,
            "max_new_tokens": 0,
            "sampling_params": SamplingParams(max_tokens=2, temperature=0.0, top_p=1.0, ignore_eos=True, speculative_mtp=True),
            "max_batch_size": 1,
            "release_after_probe": True,
        }
    ]
    assert fake.calls == [
        (
            ("one two three four",),
            SamplingParams(max_tokens=2, temperature=0.0, top_p=1.0, ignore_eos=True),
        ),
        (
            ("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",),
            SamplingParams(max_tokens=2, temperature=0.0, top_p=1.0),
        ),
    ]


def test_startup_scratch_probe_uses_max_active_request_width() -> None:
    fake = FakeLLM(outputs=["warm"])
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        max_active_requests=4,
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert fake.scratch_prepares
    assert fake.scratch_prepares[0]["max_batch_size"] == 4


def test_startup_scratch_probe_clamps_to_registered_plain_ar_route_width() -> None:
    class CappedPlainARFakeLLM(FakeLLM):
        server_plain_ar_max_active_requests = 4

    fake = CappedPlainARFakeLLM(outputs=["warm"])
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        max_active_requests=8,
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert fake.scratch_prepares
    assert fake.scratch_prepares[0]["max_batch_size"] == 4
    assert response.json()["startup"]["checks"]["scratch_probe"]["result"]["max_batch_size"] == 4


def test_startup_scratch_probe_keeps_admission_width_for_enabled_mtp_route() -> None:
    class CappedPlainARWithMTPFakeLLM(SpeculativeMTPFakeLLM):
        server_plain_ar_max_active_requests = 4

    fake = CappedPlainARWithMTPFakeLLM()
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        max_active_requests=8,
        speculative_mtp_serving="opt_in",
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert fake.scratch_prepares
    assert fake.scratch_prepares[0]["max_batch_size"] == 8


def test_startup_scratch_probe_leaves_the_width_to_the_configured_cap() -> None:
    """An unset --max-active-requests keeps the probe at one request.

    Widening it to the route cap makes the probe re-size the resident context for a
    wider batch, which OOMs on a box whose KV budget is already committed; the probe
    is a startup gate as well as a warmup, so its width stays the configured one.
    """

    fake = FakeLLM(outputs=["warm"])
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        quant="gguf_q4_k_m",
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert fake.scratch_prepares
    assert fake.scratch_prepares[0]["max_batch_size"] == 1
    probe = response.json()["startup"]["checks"]["scratch_probe"]
    assert probe["result"]["max_batch_size"] == 1


def test_startup_scratch_probe_keeps_a_smaller_server_cap() -> None:
    fake = FakeLLM(outputs=["warm"])
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        quant="gguf_q4_k_m",
        max_active_requests=2,
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert fake.scratch_prepares[0]["max_batch_size"] == 2


def test_startup_scratch_probe_prices_the_width_before_allocating_it() -> None:
    """A probe slot is a resident session, so width multiplies the session price.

    The estimator prices one session. Asking for four of them without pricing four is
    what turned a warmup into four HIP out-of-memory attempts at startup, so the probe
    is reduced before it allocates and the reduction is reported instead of silent.
    """

    fake = NarrowCapacityFakeLLM(outputs=["warm"])
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        max_active_requests=4,
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert fake.scratch_prepares[0]["max_batch_size"] == 1
    probe = body["startup"]["checks"]["scratch_probe"]
    assert probe["status"] == "reduced"
    assert probe["requested_max_batch_size"] == 4
    assert probe["max_batch_size"] == 1
    assert probe["result"]["max_batch_size"] == 1
    reduction = probe["width_reduction"]
    assert reduction["reason"] == "insufficient_free_memory_for_width"
    assert reduction["session_context_tokens"] == 4096
    assert reduction["per_session_bytes"] == 20 * 1024**3
    assert reduction["required_bytes"] == 4 * 20 * 1024**3
    assert reduction["usable_bytes"] == 8 * 1024**3
    assert reduction["detail"] == (
        "4 concurrent sessions need 80.00 GiB against 8.00 GiB usable"
    )
    assert body["diagnostics"] == [
        "startup scratch probe ran at width 1 instead of 4: "
        "4 concurrent sessions need 80.00 GiB against 8.00 GiB usable"
    ]


def test_startup_scratch_probe_keeps_the_configured_width_when_it_fits() -> None:
    """The preflight only reduces a width the machine cannot hold."""

    class RoomyCapacityFakeLLM(FakeLLM):
        def resident_capacity_estimate(
            self,
            *,
            max_batch_size: int = 1,
            requested_context_tokens: int | None = None,
        ) -> _FakeCapacityEstimate:
            del max_batch_size, requested_context_tokens
            return _FakeCapacityEstimate(
                allocatable_context_tokens=4096,
                requested_total_bytes=1 * 1024**3,
                usable_bytes=8 * 1024**3,
            )

    fake = RoomyCapacityFakeLLM(outputs=["warm"])
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", max_active_requests=4),
        llm=fake,
    )

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert fake.scratch_prepares[0]["max_batch_size"] == 4
    probe = response.json()["startup"]["checks"]["scratch_probe"]
    assert probe["status"] == "passed"
    assert "width_reduction" not in probe


def test_startup_scratch_probe_retries_at_width_one_before_failing_startup() -> None:
    """A wider probe that cannot allocate retries narrow instead of ending startup."""

    fake = WideScratchProbeFakeLLM(outputs=["warm"])
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", max_active_requests=4),
        llm=fake,
    )

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert [payload["max_batch_size"] for payload in fake.scratch_prepares] == [4, 1]
    body = response.json()
    probe = body["startup"]["checks"]["scratch_probe"]
    assert probe["status"] == "reduced"
    assert probe["requested_max_batch_size"] == 4
    assert probe["max_batch_size"] == 1
    assert probe["result"]["max_batch_size"] == 1
    assert probe["failed_attempts"] == [{"max_batch_size": 4, "exception_type": "MemoryError"}]
    assert body["diagnostics"] == [
        "startup scratch probe ran at width 1 instead of 4: "
        "width 4 failed to allocate (MemoryError)"
    ]


def test_startup_warms_the_speculative_mtp_route_before_ready() -> None:
    """The chat smoke runs the default route; an explicit MTP request runs another."""

    fake = SpeculativeMTPFakeLLM()
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        speculative_mtp_serving="opt_in",
        max_active_requests=4,
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        body = client.get("/ready").json()

    check = body["startup"]["checks"]["mtp_smoke"]
    assert check["status"] == "passed"
    assert check["route"] == "speculative_mtp"
    assert check["realized_route"] == "speculative_mtp"
    assert check["max_tokens"] == 8
    assert check["prompt_tokens"] >= 128
    assert body["startup"]["last_timings_s"]["mtp_smoke_s"] is not None
    assert len(fake.mtp_calls) == 1
    prompts, sampling = fake.mtp_calls[0]
    assert sampling.max_tokens == 8
    assert prompts[0].count("one two three four") == 32


def test_startup_skips_the_mtp_warmup_when_the_route_is_off() -> None:
    fake = SpeculativeMTPFakeLLM()
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        speculative_mtp_serving="off",
        max_active_requests=4,
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        body = client.get("/ready").json()

    assert body["startup"]["checks"]["mtp_smoke"] == {
        "enabled": False,
        "status": "disabled",
        "reason": "mtp_route_disabled",
    }
    assert body["startup"]["last_timings_s"]["mtp_smoke_s"] is None
    assert fake.mtp_calls == []


def test_startup_mtp_warmup_failure_keeps_the_server_ready() -> None:
    """A speculative warmup failure is not a serving failure: the route falls back."""

    class FailingMTPFakeLLM(SpeculativeMTPFakeLLM):
        def generate_speculative_mtp_detailed(self, prompts, sampling_params):
            raise RuntimeError("speculative warmup failed")

    fake = FailingMTPFakeLLM()
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        speculative_mtp_serving="opt_in",
        max_active_requests=4,
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    check = body["startup"]["checks"]["mtp_smoke"]
    assert check["enabled"] is True
    assert check["status"] == "failed"
    assert isinstance(check["exception_type"], str) and check["exception_type"]


def test_startup_mtp_warmup_failure_that_closes_the_engine_ends_startup() -> None:
    """A warmup that closed the engine must not announce ready over 503s.

    On a 24 GB card the MTP warmup could OOM, close the engine service, and the
    server still printed ``hipEngine ready`` while every request returned 503.
    """

    class EngineClosingMTPFakeLLM(SpeculativeMTPFakeLLM):
        closed = False

        def generate_speculative_mtp_detailed(self, prompts, sampling_params):
            self.closed = True
            raise RuntimeError("HIP error 2: out of memory")

        def service_health(self):
            return {
                "accepting_requests": not self.closed,
                "driver_alive": not self.closed,
                "closing": False,
                "closed": False,
                "failure": "GenerationExecutionFailed" if self.closed else None,
            }

    fake = EngineClosingMTPFakeLLM()
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        speculative_mtp_serving="opt_in",
        max_active_requests=4,
    )
    app = create_app(config, llm=fake)

    with pytest.raises(StartupFailure) as raised:
        with TestClient(app):
            pass

    assert raised.value.stage == "mtp_smoke"
    assert "--speculative-mtp-serving off" in raised.value.guidance


def test_lazy_server_passes_max_active_requests_to_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    fake = FakeLLM(outputs=["ok"])

    def build_llm(
        model: str,
        *,
        backend: str,
        quant: str,
        execution_profile: str | None = None,
        max_active_requests: int | None = None,
        max_sequence_length: int | None = None,
        prefix_cache: str | None = None,
        speculative_mtp_serving: str | None = None,
        speculative_provider: str | None = None,
        draft_model: str | None = None,
        speculative_candidate_budget: int | None = None,
        kv_storage: str = "auto",
        kv_scale_dtype: str = "fp16",
        kv_scale_granularity: str = "per_token_head",
        vision_model: str | None = None,
        vae_model: str | None = None,
        dms_metadata_path: str | None = None,
    ) -> FakeLLM:
        captured.update(
            {
                "model": model,
                "backend": backend,
                "quant": quant,
                "execution_profile": execution_profile,
                "max_active_requests": max_active_requests,
                "max_sequence_length": max_sequence_length,
                "prefix_cache": prefix_cache,
                "speculative_provider": speculative_provider,
                "draft_model": draft_model,
                "speculative_candidate_budget": speculative_candidate_budget,
                "kv_storage": kv_storage,
                "kv_scale_dtype": kv_scale_dtype,
                "kv_scale_granularity": kv_scale_granularity,
                "vision_model": vision_model,
                "vae_model": vae_model,
            }
        )
        return fake

    monkeypatch.setattr("hipengine.server.api.LLM", build_llm)
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_active_requests=8,
            max_context_tokens=768,
            prefix_cache="radix",
            kv_storage="int8_per_token_head",
            kv_scale_dtype="fp32",
            kv_scale_granularity="per_token_head",
        )
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "fake-model", "prompt": "one", "max_tokens": 1},
        )

    assert response.status_code == 200
    assert captured == {
        "model": "fake-path",
        "backend": "auto",
        "quant": "auto",
        "execution_profile": None,
        "max_active_requests": 8,
        "max_sequence_length": 768,
        "prefix_cache": "radix",
        "speculative_provider": None,
        "draft_model": None,
        "speculative_candidate_budget": None,
        "kv_storage": "int8_per_token_head",
        "kv_scale_dtype": "fp32",
        "kv_scale_granularity": "per_token_head",
        "vision_model": None,
        "vae_model": None,
    }


def test_startup_scratch_probe_runs_batch_width_when_context_unknown() -> None:
    class UnknownContextFakeLLM(FakeLLM):
        def prepare(self, *, max_sequence_length: int | None = None, sampling_params: SamplingParams) -> None:
            self.prepares.append((None if max_sequence_length is None else int(max_sequence_length), sampling_params))
            self.max_sequence_length = None
            return None

    fake = UnknownContextFakeLLM(outputs=["warm"])
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        max_active_requests=4,
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert fake.scratch_prepares
    assert fake.scratch_prepares[0]["max_prompt_tokens"] == 64
    assert fake.scratch_prepares[0]["max_batch_size"] == 4
    scratch_probe = response.json()["startup"]["checks"]["scratch_probe"]
    assert scratch_probe["status"] == "passed"
    assert scratch_probe["max_prompt_tokens"] is None
    assert scratch_probe["probe_prompt_tokens"] == 64
    assert scratch_probe["context_unknown"] is True


@pytest.mark.parametrize("previous_marker", [None, "0", "1"])
def test_startup_scratch_probe_passes_mtp_warmup_sampling_without_env_mutation(
    monkeypatch, previous_marker: str | None,
) -> None:
    class EnvCaptureFakeLLM(FakeLLM):
        def prepare_request_scratch(self, **kwargs) -> dict[str, Any]:
            payload = super().prepare_request_scratch(**kwargs)
            payload["mtp_startup_warmup"] = kwargs["sampling_params"].speculative_mtp
            payload["legacy_marker"] = os.environ.get("HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP")
            return payload

    if previous_marker is None:
        monkeypatch.delenv("HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP", previous_marker)
    fake = EnvCaptureFakeLLM(outputs=["warm"])
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        max_active_requests=4,
        speculative_mtp_serving="opt_in",
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    scratch_result = response.json()["startup"]["checks"]["scratch_probe"]["result"]
    assert scratch_result["mtp_startup_warmup"] is True
    assert scratch_result["legacy_marker"] == previous_marker
    assert os.environ.get("HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP") == previous_marker
    assert all(not params.speculative_mtp for _, params in fake.prepares)
    assert all(not params.speculative_mtp for _, params in fake.calls)

    fake_off = EnvCaptureFakeLLM(outputs=["warm"])
    config_off = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        max_active_requests=4,
        speculative_mtp_serving="off",
    )
    app_off = create_app(config_off, llm=fake_off)

    with TestClient(app_off) as client:
        response_off = client.get("/ready")

    assert response_off.status_code == 200
    scratch_result_off = response_off.json()["startup"]["checks"]["scratch_probe"]["result"]
    assert scratch_result_off["mtp_startup_warmup"] is False
    assert scratch_result_off["legacy_marker"] == previous_marker
    assert os.environ.get("HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP") == previous_marker


@pytest.mark.parametrize("probe_mtp", [False, True])
def test_startup_scratch_probe_retry_preserves_explicit_warmup_intent(
    monkeypatch, probe_mtp: bool,
) -> None:
    observed = []

    class RetryFakeLLM(FakeLLM):
        def prepare_request_scratch(self, **kwargs):
            observed.append((
                kwargs["max_batch_size"], kwargs["sampling_params"].speculative_mtp,
                os.environ.get("HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP"),
            ))
            if kwargs["max_batch_size"] > 1:
                raise MemoryError("retry narrow")
            return super().prepare_request_scratch(**kwargs)

    monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP", "external")
    fake = RetryFakeLLM(outputs=["warm"])
    app = create_app(ServerConfig(
        model="fake-path", served_model_name="fake-model", max_active_requests=4,
        speculative_mtp_serving="opt_in" if probe_mtp else "off",
    ), llm=fake)
    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    assert observed == [(4, probe_mtp, "external"), (1, probe_mtp, "external")]
    assert os.environ["HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP"] == "external"


def test_startup_memory_summary_counts_live_scratch_probe_peak() -> None:
    summary = _startup_memory_summary(
        {
            "startup_begin": {"free_bytes": 900, "used_bytes": 100, "total_bytes": 1000},
            "after_raw_warmup": {"free_bytes": 500, "used_bytes": 500, "total_bytes": 1000},
            "guard": {"free_bytes": 600, "used_bytes": 400, "total_bytes": 1000},
        },
        {
            "scratch_probe": {
                "status": "passed",
                "result": {
                    "live_memory": {
                        "stage": "linear_prefill_scratch_live",
                        "free_bytes": 250,
                        "used_bytes": 750,
                        "total_bytes": 1000,
                    },
                },
            },
        },
    )

    assert summary == {
        "sample_count": 4,
        "final_stage": "guard",
        "final_free_bytes": 600,
        "final_used_bytes": 400,
        "peak_stage": "scratch_probe:linear_prefill_scratch_live",
        "peak_used_bytes": 750,
        "min_free_stage": "scratch_probe:linear_prefill_scratch_live",
        "min_free_bytes": 250,
        "total_bytes": 1000,
    }


def test_pretty_startup_summary_contains_operational_limits(monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr("hipengine.server.api._LOGGER.info", lambda message, *args: messages.append(message % args))
    config = ServerConfig(model="/models/example.gguf", served_model_name="example")
    estimate = SimpleNamespace(
        requested_context_tokens=176128,
        model_max_context_tokens=262144,
        kv_storage_dtype="int8_per_token_head",
        kv_scale_dtype="fp32",
        usable_bytes=24 * 1024**3,
    )
    engine = SimpleNamespace(
        kv_capacity_estimate=estimate,
        live_loop_snapshot=lambda: {
            "runner": {
                "max_active_requests": 4,
                "kv_pool": {"budget_bytes": 24 * 1024**3},
            }
        },
    )
    _log_pretty_startup_summary(
        config,
        engine=engine,
        memory={
            "final_used_bytes": 28 * 1024**3,
            "total_bytes": 48 * 1024**3,
        },
    )
    rendered = messages[-1]
    assert "176,128 / 262,144 tokens" in rendered
    assert "4 requests in flight" in rendered
    assert "24.00 GiB" in rendered
    assert "28.00 GiB / 48.00 GiB used" in rendered
    assert "Ready for requests" in rendered


def test_health_and_ready_report_eager_startup_diagnostics() -> None:
    fake = FakeLLM(outputs=["private warmup output"])
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        eager_load_prompt="private startup prompt",
        eager_load_max_tokens=2,
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")

    assert health.status_code == 200
    assert health.json() == {
        "object": "hipengine.health",
        "status": "ok",
        "model": "fake-model",
    }
    assert ready.status_code == 200
    body = ready.json()
    assert body["object"] == "hipengine.readiness"
    assert body["ready"] is True
    assert body["status"] == "ready"
    assert {
        key: body["model"][key]
        for key in ("id", "backend", "quant", "loaded", "loaded_model_count")
    } == {
        "id": "fake-model",
        "backend": "auto",
        "quant": "auto",
        "loaded": True,
        "loaded_model_count": 1,
    }
    assert body["model"]["kv_capability"]["status"] == "unavailable"
    assert body["model"]["kv_capability"]["promotion_eligible"] is False
    assert body["startup"]["eager_load"] is True
    assert body["startup"]["warmup_complete"] is True
    assert body["startup"]["last_timings_s"]["warmup_s"] >= 0.0
    assert body["startup"]["last_timings_s"]["scratch_probe_s"] >= 0.0
    assert body["startup"]["last_timings_s"]["chat_smoke_s"] >= 0.0
    assert body["startup"]["checks"]["scratch_probe"]["status"] == "passed"
    assert body["startup"]["checks"]["scratch_probe"]["max_prompt_tokens"] == 131071
    assert body["startup"]["checks"]["chat_smoke"]["status"] == "passed"
    for snapshot in body["startup"]["memory"].values():
        assert set(snapshot) >= {"free_bytes", "total_bytes", "used_bytes"}
    assert body["context"]["effective_max_context_tokens"] == 131072
    assert body["kv_capacity"]["estimate"]["allocatable_context_tokens"] == 131072
    assert body["kv_capacity"]["storage"] == "auto"
    assert body["prefix_cache"] == {
        "mode": "off",
        "block_size_tokens": 256,
        "snapshot_entries": 0,
        "snapshot_limit": 0,
        "retained_snapshot_entries": 0,
        "snapshot_bytes": 0,
        "retained_kv_pages": 0,
        "retained_kv_bytes": 0,
        "resident_bytes": 0,
        "resident_limit_bytes": 0,
    }
    assert body["graph_cache"]["entries"] == 0.0
    assert body["queue"]["depth"] == 0
    assert body["queue"]["max_depth"] is None
    assert body["queue"]["active_requests"] == 0
    assert body["queue"]["max_active_requests"] is None
    assert body["queue"]["scheduler_fairness"] == {
        "policy": "fifo_compatible_sampling_key",
        "compatible_sampling_coalescing": True,
        "continuous_decode": False,
        "preemptive_fairness": False,
    }
    assert body["sessions"] == {
        "resident_context": True,
        "active": 0,
        "pending_creations": 0,
        "max_active": None,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "total_messages": 0,
        "continuations": {"active": 0, "ttl_seconds": 900},
    }
    serialized = json.dumps(body)
    assert "private startup prompt" not in serialized
    assert "private warmup output" not in serialized


def test_ready_reports_selected_visible_gpu_from_rocm_env(monkeypatch) -> None:
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0, 2")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3")
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=FakeLLM(),
    )
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["device"] == {
        "backend": "auto",
        "hip_visible_devices": "0, 2",
        "rocr_visible_devices": "3",
        "visible_devices": ["0", "2"],
        "selected_visible_device": "0",
        "selection_source": "HIP_VISIBLE_DEVICES",
    }


def test_failed_eager_startup_ends_the_process_and_keeps_the_diagnosis() -> None:
    """A server that cannot start exits instead of listening unready.

    The failure is recorded before the process ends, so the same diagnosis is still
    readable from the readiness payload, and the log carries it for a supervisor that
    only sees a non-zero exit.
    """

    fake = ScratchProbeFailureFakeLLM(outputs=["private warmup output"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load_prompt="private startup prompt",
            eager_load_max_tokens=2,
        ),
        llm=fake,
    )

    with pytest.raises(StartupFailure) as raised:
        with TestClient(app):
            pass

    assert raised.value.stage == "scratch_probe"
    assert raised.value.guidance == (
        "Try a lower --max-context-tokens or a higher scratch/headroom reserve."
    )
    assert isinstance(raised.value.cause, RuntimeError)

    # The lifespan never completed, so this reads the state it recorded on the way out.
    ready = TestClient(app).get("/ready")

    assert ready.status_code == 503
    body = ready.json()
    assert body["object"] == "hipengine.readiness"
    assert body["ready"] is False
    assert body["status"] == "error"
    assert body["diagnostics"] == [
        "server startup is not ready; check startup.error and server logs",
        "startup scratch_probe failed: Try a lower --max-context-tokens or a higher scratch/headroom reserve.",
    ]
    assert body["startup"]["warmup_complete"] is False
    assert body["startup"]["error"] == {
        "stage": "scratch_probe",
        "type": "RuntimeError",
        "message": "startup scratch_probe failed",
        "guidance": "Try a lower --max-context-tokens or a higher scratch/headroom reserve.",
    }
    assert body["startup"]["checks"]["raw_warmup"] == {"status": "passed", "max_tokens": 2}
    probe = body["startup"]["checks"]["scratch_probe"]
    assert probe == {
        "enabled": True,
        "status": "failed",
        "max_prompt_tokens": 131071,
        "probe_prompt_tokens": 131071,
        "context_unknown": False,
        "requested_max_batch_size": 1,
        "max_batch_size": 1,
        "exception_type": "RuntimeError",
        "failed_attempts": [{"max_batch_size": 1, "exception_type": "RuntimeError"}],
    }
    assert body["startup"]["last_timings_s"]["warmup_s"] >= 0.0
    assert body["startup"]["last_timings_s"]["scratch_probe_s"] >= 0.0
    assert body["startup"]["last_timings_s"]["startup_total_s"] >= 0.0
    serialized = json.dumps(body)
    assert "private startup prompt" not in serialized
    assert "private warmup output" not in serialized


def test_ready_reports_chat_session_counts_without_payload_text() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "secret session prompt"}],
            "session": {"id": "sess_ready"},
            "max_tokens": 4,
        },
    )
    ready = client.get("/ready")

    assert response.status_code == 200
    assert ready.status_code == 200
    body = ready.json()
    assert body["sessions"] == {
        "resident_context": True,
        "active": 1,
        "pending_creations": 0,
        "max_active": None,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "total_messages": 2,
        "continuations": {"active": 0, "ttl_seconds": 900},
    }
    serialized = json.dumps(body)
    assert "secret session prompt" not in serialized
    assert "stored answer" not in serialized


def test_ready_reports_lazy_server_ready_without_loaded_model() -> None:
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False)
    )

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["status"] == "ready"
    assert body["startup"]["eager_load"] is False
    assert body["startup"]["warmup_complete"] is True
    assert body["startup"]["last_timings_s"]["startup_total_s"] >= 0.0
    assert body["model"]["loaded"] is False
    assert body["model"]["loaded_model_count"] == 0


def test_ready_reports_runtime_engine_service_failure_as_unhealthy() -> None:
    """A fatal resident-execution failure must be visible to a readiness probe."""

    class UnhealthyLLM(FakeLLM):
        def engine_service_health(self) -> dict[str, Any]:
            return {
                "object": "hipengine.engine_service.health",
                "status": "unhealthy",
                "serving": False,
                "unhealthy": {
                    "exception_type": "RuntimeError",
                    "message": "packed sampler row 1 exceeds capacity 0",
                    "location": "runner.py:29767 in sample_native_from_packed_logits",
                    "phase": "decode",
                    "request_ids": [3],
                },
            }

    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False)
    )
    app.state.hipengine_llm = UnhealthyLLM()

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["status"] == "unhealthy"
    assert body["engine_service"]["status"] == "unhealthy"
    diagnostic = " ".join(body["diagnostics"])
    assert "fatal execution failure" in diagnostic
    assert "packed sampler row 1 exceeds capacity 0" in diagnostic
    assert "runner.py:29767" in diagnostic
    # The client is never told to shrink its memory budget for a runtime fault.
    assert "--max-context-tokens" not in diagnostic
