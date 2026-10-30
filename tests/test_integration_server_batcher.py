"""OpenAI server integration tests: batcher.

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
    BlockingPrepareFakeLLM,
    FailingSpeculativeMTPFakeLLM,
    FakeLLM,
    PromptPreparingFakeLLM,
    RealizedGroupArtifactScopedSpeculativeMTPFakeLLM,
    ResidentSessionPromptFakeLLM,
    SpeculativeMTPFakeLLM,
)
from tests.support.server_api_support import (
    _asgi_chat_post,
    _sse_payloads,
)


def test_mtp_circuit_breaker_scope_buckets_by_prompt_context_tokens() -> None:
    """The breaker's context bucket follows the request, not the engine."""

    engine = SimpleNamespace(
        model_id="model",
        _resolved_backend="hip_gfx1151",
        resolved_execution_profile="strict",
    )
    breaker = _MTPCircuitBreaker(threshold=2)

    assert breaker.scope(engine, ((7,) * 300,)) == (
        "model",
        "hip_gfx1151",
        "strict",
        "512",
    )
    assert breaker.scope(engine, ((7,) * 4096,))[3] == "4096"
    assert breaker.scope(
        engine,
        (PreparedPromptInput(source_text="x", token_ids=(7,) * 1000),),
    )[3] == "1024"
    # Text prompts are not tokenized here, so counting characters would put
    # unrelated requests in one bucket. They stay scope-unknown instead.
    assert breaker.scope(engine, ("a text prompt",))[3] == "unknown"
    assert breaker.scope(engine, ())[3] == "unknown"


@pytest.mark.parametrize(
    "path,stream,n,info,metadata,live_many",
    [
        ("/ready", False, 1, False, False, False),
        ("/metrics", False, 1, False, False, False),
        ("/v1/completions", False, 1, True, False, False),
        ("/v1/completions", True, 1, True, False, False),
        ("/v1/completions", True, 1, False, True, False),
        ("/v1/chat/completions", False, 1, True, False, False),
        ("/v1/chat/completions", True, 1, True, False, False),
        ("/v1/chat/completions", True, 1, False, True, False),
        ("/v1/chat/completions", True, 2, True, False, False),
        ("/v1/chat/completions", True, 2, False, True, False),
        ("/v1/chat/completions", True, 2, True, False, True),
        ("/v1/chat/completions", True, 2, False, True, True),
    ],
)
def test_busy_snapshot_does_not_block_other_http_requests(path, stream, n, info, metadata, live_many):
    import httpx

    entered = threading.Event()
    release = threading.Event()
    snapshot_threads = []

    class BusySnapshotFake(FakeLLM):
        supports_stream_many = live_many

        def stream_many_detailed(self, prompts, sampling_params):
            for index, _prompt in enumerate(prompts):
                yield GenerationStreamChunk(
                    text="hello",
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1, generated_tokens=1, row_index=index,
                        phase="answer", sampler_mode="greedy_fast", execution_path="test",
                    ),
                )

        def live_loop_snapshot(self):
            snapshot_threads.append(threading.get_ident())
            entered.set()
            assert release.wait(5.0), "snapshot watchdog did not release"
            return {"runner": {"kv_pool": {"current_bytes": 1024}}}

    fake = BusySnapshotFake(outputs=["hello"] * n)
    app = create_app(
        ServerConfig(
            model="fake", served_model_name="fake-model", eager_load=False,
            metrics="prometheus", info=info,
        ),
        llm=fake,
    )

    # Only release after a real snapshot wait has started. This also lets a
    # regressed, blocked event loop unwind and fail instead of hanging the test.
    def watchdog():
        if entered.wait(5.0):
            release.wait(1.0)
        release.set()

    watcher = threading.Thread(target=watchdog)
    watcher.start()

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            body = {"model": "fake-model", "max_tokens": 4, "stream": stream, "n": n}
            if path == "/v1/chat/completions":
                body["messages"] = [{"role": "user", "content": "hi"}]
            else:
                body["prompt"] = "hi"
            if metadata:
                body["stream_options"] = {"include_hipengine": True, "include_usage": True}
            pending = asyncio.create_task(
                client.get(path) if path in {"/ready", "/metrics"}
                else client.post(path, json=body)
            )
            try:
                assert await asyncio.to_thread(entered.wait, 5.0)
                health = await client.get("/health")
                models = await client.get("/v1/models")
                assert health.status_code == models.status_code == 200
                assert not release.is_set(), "unrelated HTTP requests waited for the driver"
                assert threading.get_ident() not in snapshot_threads
            finally:
                release.set()
                response = await pending
            assert response.status_code == 200
            if path.startswith("/v1/"):
                assert '"error":' not in response.text

    try:
        asyncio.run(run())
    finally:
        release.set()
        watcher.join(5.0)


def test_generation_batcher_coalesces_compatible_submissions() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), sampling),
            batcher.submit(("two", "three"), sampling),
        )

        assert first == ["generated:one"]
        assert second == ["generated:two", "generated:three"]
        assert fake.calls == [(("one", "two", "three"), sampling)]

    asyncio.run(run())


def test_generation_batcher_dispatches_solo_submission_before_full_window() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.100,
        )

        started = time.perf_counter()
        result = await batcher.submit(("solo",), sampling)
        elapsed = time.perf_counter() - started

        assert result == ["generated:solo"]
        assert fake.calls == [(("solo",), sampling)]
        # A solo submission on an idle engine must not pay the full batch
        # window; the solo dispatch slice is 2 ms, so anything near the
        # 100 ms window is a regression of the early-dispatch contract.
        assert elapsed < 0.060, f"solo dispatch took {elapsed:.3f}s"

    asyncio.run(run())


def test_mtp_circuit_breaker_opens_by_scope_and_restart_resets() -> None:
    engine = SimpleNamespace(
        model_id="model",
        _resolved_backend="hip_gfx1151",
        resolved_execution_profile="strict",
    )
    breaker = _MTPCircuitBreaker(threshold=2)
    scope = breaker.scope(engine, ("prompt",))
    assert breaker.is_open(scope) is False

    breaker.record_failure(scope, RuntimeError("first"))
    assert breaker.is_open(scope) is False
    breaker.record_failure(scope, RuntimeError("second"))
    breaker.attach(engine)

    assert breaker.is_open(scope) is True
    assert engine.mtp_circuit_breaker_state["state"] == "open"
    assert engine.mtp_circuit_breaker_state["failures"][0]["count"] == 2
    restarted = _MTPCircuitBreaker(threshold=2)
    assert restarted.snapshot()["state"] == "closed"
    assert restarted.is_open(scope) is False


def test_generation_batcher_open_mtp_breaker_rejects_explicit_third_launch() -> None:
    async def run() -> None:
        fake = FailingSpeculativeMTPFakeLLM()
        breaker = _MTPCircuitBreaker(threshold=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
            mtp_circuit_breaker=breaker,
        )
        for _ in range(2):
            with pytest.raises(RuntimeError, match="injected MTP"):
                await batcher.submit(
                    ("prompt",),
                    SamplingParams(max_tokens=2),
                    route="speculative_mtp",
                )
        with pytest.raises(OpenAIHTTPError, match="mtp_circuit_breaker_open"):
            await batcher.submit(
                ("prompt",),
                SamplingParams(max_tokens=2),
                route="speculative_mtp",
            )

        assert len(fake.mtp_calls) == 2
        assert fake.calls == []
        assert fake.mtp_circuit_breaker_state["state"] == "open"

    asyncio.run(run())


def test_open_mtp_breaker_rejects_explicit_request_and_reports_state() -> None:
    fake = SpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="opt_in",
        ),
        llm=fake,
    )
    breaker = app.state.hipengine_mtp_circuit_breaker
    scope = breaker.scope(fake, ("hello",))
    for index in range(breaker.threshold):
        breaker.record_failure(scope, RuntimeError(f"injected {index}"))
    assert breaker.is_open(scope) is True
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 2,
            "temperature": 0.0,
            "speculative_mtp": True,
        },
    )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["param"] == "speculative_mtp"
    assert body["error"]["code"] == "engine_unavailable"
    assert body["error"]["hipengine"]["speculative_mtp"]["reason"] == "mtp_circuit_breaker_open"
    assert body["error"]["hipengine"]["speculative_mtp"]["circuit_breaker"]["state"] == "open"
    assert fake.mtp_calls == []
    assert fake.calls == []


@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/chat/completions"])
def test_open_mtp_breaker_rejects_explicit_streaming_requests(endpoint: str) -> None:
    fake = SpeculativeMTPFakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            speculative_mtp_serving="opt_in",
        ),
        llm=fake,
    )
    breaker = app.state.hipengine_mtp_circuit_breaker
    scope = breaker.scope(fake, ("hello",))
    for index in range(breaker.threshold):
        breaker.record_failure(scope, RuntimeError(f"injected {index}"))
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
    error = next(item["error"] for item in payloads if item.get("error"))
    assert error["code"] == "engine_unavailable"
    assert "mtp_circuit_breaker_open" in error["message"]
    assert not any(item.get("choices", [{}])[0].get("text") for item in payloads if item.get("choices"))
    assert fake.mtp_calls == []
    assert fake.calls == []


def test_operator_mtp_rollback_rejects_new_explicit_requests_until_restart() -> None:
    fake = SpeculativeMTPFakeLLM()
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        speculative_mtp_serving="opt_in",
    )
    app = create_app(config, llm=fake)
    client = TestClient(app)
    payload = {
        "model": "fake-model",
        "prompt": "hello",
        "max_tokens": 2,
        "temperature": 0.0,
        "speculative_mtp": True,
    }

    before = client.post("/v1/completions", json=payload)
    rollback = client.post("/v1/hipengine/speculative_mtp/rollback")
    after = client.post("/v1/completions", json=payload)

    assert before.status_code == rollback.status_code == 200
    assert after.status_code == 503
    assert before.json()["hipengine"]["generation_shape"]["route"] == "speculative_mtp"
    assert rollback.json()["new_requests_route"] == "default"
    assert rollback.json()["circuit_breaker"]["state"] == "operator_disabled"
    assert after.json()["error"]["hipengine"]["speculative_mtp"]["reason"] == "mtp_operator_rollback"
    assert after.json()["error"]["hipengine"]["speculative_mtp"]["circuit_breaker"]["state"] == "operator_disabled"
    assert len(fake.mtp_calls) == 1
    assert fake.calls == []

    restarted_fake = SpeculativeMTPFakeLLM()
    restarted = TestClient(create_app(config, llm=restarted_fake))
    restarted_response = restarted.post("/v1/completions", json=payload)
    assert restarted_response.status_code == 200
    assert restarted_response.json()["hipengine"]["generation_shape"]["route"] == "speculative_mtp"
    assert len(restarted_fake.mtp_calls) == 1


def test_generation_batcher_keeps_speculative_mtp_and_default_routes_separate() -> None:
    async def run() -> None:
        fake = SpeculativeMTPFakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        default_result, mtp_result = await asyncio.gather(
            batcher.submit(("one",), sampling),
            batcher.submit(("two", "three"), sampling, route="speculative_mtp"),
        )

        assert default_result == ["generated:one"]
        assert mtp_result == ["mtp:two", "mtp:three"]
        assert fake.calls == [(("one",), sampling)]
        assert fake.mtp_calls == [(("two", "three"), sampling)]

    asyncio.run(run())


def test_generation_batcher_coalesces_speculative_mtp_with_request_tokens() -> None:
    async def run() -> None:
        fake = SpeculativeMTPFakeLLM()
        first_token = GenerationCancellationToken()
        second_token = GenerationCancellationToken()
        first_sampling = SamplingParams(max_tokens=2, cancellation_token=first_token)
        second_sampling = SamplingParams(max_tokens=2, cancellation_token=second_token)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), first_sampling, route="speculative_mtp"),
            batcher.submit(("two",), second_sampling, route="speculative_mtp"),
        )

        assert first == ["mtp:one"]
        assert second == ["mtp:two"]
        assert fake.calls == []
        assert len(fake.mtp_calls) == 1
        prompts, grouped_sampling = fake.mtp_calls[0]
        assert prompts == ("one", "two")
        assert grouped_sampling.cancellation_token is not first_token
        assert grouped_sampling.cancellation_token is not second_token
        assert grouped_sampling.cancellation_token is not None
        assert grouped_sampling.cancellation_token.cancelled is False
        second_token.cancel(FinishDetails(reason="cancelled", cancelled=True))
        assert grouped_sampling.cancellation_token.cancelled is False
        grouped_sampling.cancellation_token.raise_if_cancelled()
        first_token.cancel(FinishDetails(reason="cancelled", cancelled=True))
        assert grouped_sampling.cancellation_token.cancelled is True
        with pytest.raises(GenerationCancelled):
            grouped_sampling.cancellation_token.raise_if_cancelled()

    asyncio.run(run())


def test_generation_batcher_returns_scheduler_chunks_for_single_metadata_submission() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            outputs = super().generate_detailed(prompts, sampling_params)
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {"request_id": index, "token_index": 0, "token_id": 100 + index, "chunk": {"text": output.text}}
                    for index, output in enumerate(outputs)
                ]
            }
            return outputs

    async def run() -> None:
        fake = SchedulerChunkFakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
        )

        result = await batcher.submit(
            ("one", "two"),
            sampling,
            detailed=True,
            include_batch_metadata=True,
        )

        assert isinstance(result, _QueuedBatchResult)
        assert [output.text for output in result.outputs] == ["generated:one", "generated:two"]
        assert result.scheduler_token_chunks == [
            {"request_id": 0, "token_index": 0, "token_id": 100, "chunk": {"text": "generated:one"}},
            {"request_id": 1, "token_index": 0, "token_id": 101, "chunk": {"text": "generated:two"}},
        ]

    asyncio.run(run())


def test_backend_scheduler_token_chunks_reads_wrapped_text_generator() -> None:
    source_chunks = [
        {
            "request_id": 0,
            "token_index": 0,
            "token_id": 101,
            "chunk": {"text": "A", "telemetry": {"decode_state": {"execution_path": "gguf_serial_host_sampler_decode"}}},
        }
    ]
    inner = SimpleNamespace(last_batch_generation={"scheduler_token_chunks": source_chunks})
    engine = SimpleNamespace(_text_generator=SimpleNamespace(inner=inner))

    chunks = _backend_scheduler_token_chunks(engine)

    assert chunks == source_chunks
    assert chunks is not None
    assert chunks[0] is not source_chunks[0]
    assert chunks[0]["chunk"] is not source_chunks[0]["chunk"]


def test_generation_batcher_drops_scheduler_chunks_for_coalesced_metadata_submissions() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            outputs = super().generate_detailed(prompts, sampling_params)
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {"request_id": index, "token_index": 0, "token_id": 100 + index, "chunk": {"text": output.text}}
                    for index, output in enumerate(outputs)
                ]
            }
            return outputs

    async def run() -> None:
        fake = SchedulerChunkFakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), sampling, detailed=True, include_batch_metadata=True),
            batcher.submit(("two",), sampling, detailed=True, include_batch_metadata=True),
        )

        assert isinstance(first, _QueuedBatchResult)
        assert isinstance(second, _QueuedBatchResult)
        assert [output.text for output in first.outputs] == ["generated:one"]
        assert [output.text for output in second.outputs] == ["generated:two"]
        assert first.scheduler_token_chunks is None
        assert second.scheduler_token_chunks is None
        assert fake.calls == [(("one", "two"), sampling)]

    asyncio.run(run())


def test_generation_batcher_default_zero_window_queues_without_lifetime_lock() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), sampling),
            batcher.submit(("two",), sampling),
        )

        streamed = [chunk async for chunk in batcher.stream(("three",), sampling)]

        assert first == ["generated:one"]
        assert second == ["generated:two"]
        assert [chunk.text for chunk in streamed] == ["generated:three"]
        assert all(isinstance(chunk, GenerationStreamChunk) for chunk in streamed)
        assert fake.calls == [(("one", "two"), sampling), (("three",), sampling)]

    asyncio.run(run())


def test_generation_batcher_stream_uses_per_request_queue_and_coalesces() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        async def collect_stream() -> list[GenerationStreamChunk]:
            return [chunk async for chunk in batcher.stream(("stream",), sampling)]

        streamed, submitted = await asyncio.gather(
            collect_stream(),
            batcher.submit(("batch",), sampling),
        )

        assert [chunk.text for chunk in streamed] == ["generated:stream"]
        assert all(isinstance(chunk, GenerationStreamChunk) for chunk in streamed)
        assert submitted == ["generated:batch"]
        assert len(fake.calls) == 1
        assert set(fake.calls[0][0]) == {"stream", "batch"}
        assert fake.calls[0][1] == sampling

    asyncio.run(run())


def test_generation_batcher_controlled_stream_admits_while_neighbor_is_live() -> None:
    class ControlledStreamLLM(FakeLLM):
        supports_controlled_streaming = True

        def __init__(self) -> None:
            super().__init__()
            self.started = {"slow": threading.Event(), "fast": threading.Event()}
            self.release_slow = threading.Event()

        def stream_detailed(self, prompt: str, sampling_params: SamplingParams):
            name = str(prompt)
            self.started[name].set()
            yield GenerationStreamChunk(text=f"{name}:0")
            if name == "slow":
                assert self.release_slow.wait(timeout=5.0)
            yield GenerationStreamChunk(text=f"{name}:1")

    async def run() -> None:
        fake = ControlledStreamLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
        )

        async def collect(name: str) -> list[str]:
            return [chunk.text async for chunk in batcher.stream((name,), sampling)]

        slow = asyncio.create_task(collect("slow"))
        assert await asyncio.to_thread(fake.started["slow"].wait, 5.0)
        fast = asyncio.create_task(collect("fast"))
        await asyncio.sleep(0.05)
        admitted_while_slow = fake.started["fast"].is_set()
        fake.release_slow.set()
        slow_text, fast_text = await asyncio.gather(slow, fast)

        assert admitted_while_slow is True
        assert slow_text == ["slow:0", "slow:1"]
        assert fast_text == ["fast:0", "fast:1"]
        assert batcher.active_requests() == 0

    asyncio.run(run())


def test_generation_batcher_limits_controlled_stream_active_requests() -> None:
    class ControlledStreamLLM(FakeLLM):
        supports_controlled_streaming = True

        def __init__(self) -> None:
            super().__init__()
            self.started = {
                "first": threading.Event(),
                "second": threading.Event(),
                "third": threading.Event(),
            }
            self.release = {
                "first": threading.Event(),
                "second": threading.Event(),
                "third": threading.Event(),
            }
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def stream_detailed(self, prompt: str, sampling_params: SamplingParams):
            del sampling_params
            name = str(prompt)
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            self.started[name].set()
            try:
                yield GenerationStreamChunk(text=f"{name}:0")
                assert self.release[name].wait(timeout=5.0)
                yield GenerationStreamChunk(text=f"{name}:1")
            finally:
                with self.lock:
                    self.active -= 1

    async def run() -> None:
        fake = ControlledStreamLLM()
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
            max_active_requests=1,
            max_queue_size=1,
        )
        sampling = SamplingParams(max_tokens=2)

        async def collect(name: str) -> list[str]:
            return [chunk.text async for chunk in batcher.stream((name,), sampling)]

        first = asyncio.create_task(collect("first"))
        assert await asyncio.to_thread(fake.started["first"].wait, 5.0)
        second = asyncio.create_task(collect("second"))
        await asyncio.sleep(0.05)

        assert fake.started["second"].is_set() is False
        assert batcher.active_requests() == 1
        assert batcher.queue_depth() == 1
        third = batcher.stream(("third",), sampling)
        with pytest.raises(OpenAIHTTPError) as exc_info:
            await anext(third)
        assert exc_info.value.status_code == 429
        assert exc_info.value.code == "engine_busy"
        assert fake.started["third"].is_set() is False

        fake.release["first"].set()
        assert await first == ["first:0", "first:1"]
        assert await asyncio.to_thread(fake.started["second"].wait, 5.0)
        fake.release["second"].set()
        assert await second == ["second:0", "second:1"]
        assert fake.max_active == 1
        assert batcher.active_requests() == 0
        assert batcher.queue_depth() == 0

    asyncio.run(run())


def test_generation_batcher_releases_prestart_cancelled_controlled_reservation() -> None:
    async def run() -> None:
        fake = FakeLLM()
        fake.supports_controlled_streaming = True
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
            max_active_requests=1,
        )
        item = _QueuedGeneration(
            prompts=("cancel-before-start",),
            sampling=SamplingParams(max_tokens=1),
            stream_queue=asyncio.Queue(maxsize=2),
        )

        batcher._launch_controlled_stream(item, fake)
        assert item.producer_task is not None
        assert batcher.active_requests() == 1
        item.producer_task.cancel()
        await asyncio.gather(item.producer_task, return_exceptions=True)
        await asyncio.sleep(0)

        assert batcher.active_requests() == 0
        assert batcher.active() is False

    asyncio.run(run())


def test_generation_batcher_drains_pending_controlled_cancellation_on_exit() -> None:
    class PendingCancellationLLM(FakeLLM):
        supports_controlled_streaming = True

        def __init__(self) -> None:
            super().__init__()
            self.release = threading.Event()
            self.token: GenerationCancellationToken | None = None
            self.drain_calls = 0
            self.drained = threading.Event()

        def stream_detailed(self, prompt: str, sampling_params: SamplingParams):
            del prompt
            token = sampling_params.cancellation_token
            assert token is not None
            token.set_cancel_dispatch(lambda _details: None)
            self.token = token
            yield GenerationStreamChunk(text="first")
            assert self.release.wait(timeout=5.0)

        def drain_generation_cancellations(self) -> int:
            self.drain_calls += 1
            assert self.token is not None
            self.token.acknowledge_cancel()
            self.drained.set()
            return 1

    async def run() -> None:
        fake = PendingCancellationLLM()
        token = GenerationCancellationToken()
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
            max_active_requests=1,
        )
        stream = batcher.stream(
            ("cancel-me",),
            SamplingParams(max_tokens=8, cancellation_token=token),
        )

        assert (await anext(stream)).text == "first"
        token.cancel(FinishDetails(reason="cancelled", cancelled=True))
        assert token.cancel_requested is True
        assert token.cancelled is False
        fake.release.set()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert await asyncio.to_thread(fake.drained.wait, 5.0)

        assert fake.drain_calls == 1
        assert token.cancelled is True
        assert batcher.active_requests() == 0

    asyncio.run(run())


def test_generation_batcher_bounds_slow_stream_queue_without_blocking_neighbor() -> None:
    class BurstStreamLLM(FakeLLM):
        supports_controlled_streaming = True

        def stream_detailed(self, prompt: str, sampling_params: SamplingParams):
            count = 20 if str(prompt) == "slow" else 2
            for index in range(count):
                yield GenerationStreamChunk(text=f"{prompt}:{index}")

    async def run() -> None:
        fake = BurstStreamLLM()
        slow_token = GenerationCancellationToken()
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
            stream_queue_max_chunks=2,
        )
        slow_stream = batcher.stream(
            ("slow",),
            SamplingParams(max_tokens=20, cancellation_token=slow_token),
        )

        assert (await anext(slow_stream)).text == "slow:0"
        await asyncio.sleep(0.05)
        fast_text = [
            chunk.text
            async for chunk in batcher.stream(("fast",), SamplingParams(max_tokens=2))
        ]

        assert fast_text == ["fast:0", "fast:1"]
        assert batcher.stream_queue_max_chunks() == 2
        assert batcher.stream_queue_depths()
        assert max(batcher.stream_queue_depths()) <= 2

        await slow_stream.aclose()
        await asyncio.sleep(0)
        assert slow_token.cancelled is True
        assert slow_token.finish_details.to_json_dict() == {
            "reason": "cancelled",
            "cancelled": True,
        }
        await batcher.shutdown(grace_seconds=0.1)
        assert batcher.active() is False

    asyncio.run(run())


def test_generation_batcher_stream_close_waits_for_cooperative_backend_cancel() -> None:
    class CooperativeCloseLLM(FakeLLM):
        supports_controlled_streaming = True

        def __init__(self) -> None:
            super().__init__()
            self.cancel_seen = threading.Event()
            self.release_cancel = threading.Event()

        def stream_detailed(self, prompt: str, sampling_params: SamplingParams):
            yield GenerationStreamChunk(text=f"{prompt}:0")
            token = sampling_params.cancellation_token
            assert token is not None
            while not token.cancelled:
                time.sleep(0.001)
            self.cancel_seen.set()
            assert self.release_cancel.wait(timeout=5.0)
            token.raise_if_cancelled()

    async def run() -> None:
        fake = CooperativeCloseLLM()
        token = GenerationCancellationToken()
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
        )
        stream = batcher.stream(
            ("cooperative",),
            SamplingParams(max_tokens=8, cancellation_token=token),
        )

        assert (await anext(stream)).text == "cooperative:0"
        release = threading.Timer(0.05, fake.release_cancel.set)
        release.start()
        try:
            await stream.aclose()
        finally:
            release.cancel()
            fake.release_cancel.set()
        assert fake.cancel_seen.is_set()
        assert token.cancelled is True
        assert batcher.active_requests() == 0
        assert batcher.active() is False

    asyncio.run(run())


def test_generation_batcher_shutdown_forces_cancel_after_grace() -> None:
    class CooperativeBlockingStreamLLM(FakeLLM):
        supports_controlled_streaming = True

        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()

        def stream_detailed(self, prompt: str, sampling_params: SamplingParams):
            del prompt
            self.started.set()
            token = sampling_params.cancellation_token
            assert token is not None
            while not token.cancelled:
                time.sleep(0.005)
            token.raise_if_cancelled()
            yield GenerationStreamChunk(text="unreachable")

    async def run() -> None:
        fake = CooperativeBlockingStreamLLM()
        token = GenerationCancellationToken()
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
            stream_queue_max_chunks=2,
        )
        stream = batcher.stream(
            ("blocked",),
            SamplingParams(max_tokens=8, cancellation_token=token),
        )
        pending = asyncio.create_task(anext(stream))
        assert await asyncio.to_thread(fake.started.wait, 5.0)

        result = await batcher.shutdown(grace_seconds=0.01)

        assert result["forced"] is True
        assert result["active_requests"] == 0
        assert token.cancelled is True
        with pytest.raises((GenerationCancelled, StopAsyncIteration, asyncio.CancelledError)):
            await pending
        await stream.aclose()
        assert batcher.active() is False

    asyncio.run(run())


def test_generation_batcher_shutdown_waits_for_model_thread_to_retire() -> None:
    class SlowCooperativeLLM(FakeLLM):
        supports_controlled_streaming = True

        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.cancel_seen = threading.Event()
            self.release = threading.Event()

        def stream_detailed(self, prompt: str, sampling_params: SamplingParams):
            del prompt
            self.started.set()
            token = sampling_params.cancellation_token
            assert token is not None
            while not token.cancelled:
                time.sleep(0.001)
            self.cancel_seen.set()
            assert self.release.wait(timeout=5.0)
            token.raise_if_cancelled()
            yield GenerationStreamChunk(text="unreachable")

    async def run() -> None:
        fake = SlowCooperativeLLM()
        token = GenerationCancellationToken()
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
        )
        stream = batcher.stream(
            ("slow",),
            SamplingParams(max_tokens=8, cancellation_token=token),
        )
        pending = asyncio.create_task(anext(stream))
        assert await asyncio.to_thread(fake.started.wait, 5.0)

        shutdown = asyncio.create_task(batcher.shutdown(grace_seconds=0.01))
        assert await asyncio.to_thread(fake.cancel_seen.wait, 5.0)
        await asyncio.sleep(0.02)
        assert shutdown.done() is False
        assert batcher.active_requests() == 1

        fake.release.set()
        result = await asyncio.wait_for(shutdown, timeout=5.0)
        assert result["forced"] is True
        assert result["active_requests"] == 0
        assert batcher.active() is False
        with pytest.raises((GenerationCancelled, StopAsyncIteration)):
            await pending
        await stream.aclose()

    asyncio.run(run())


def test_server_shutdown_drains_batcher_and_closes_long_lived_engine() -> None:
    class ClosableLLM(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake = ClosableLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            stream_queue_max_chunks=4,
            shutdown_grace_seconds=0.05,
        ),
        llm=fake,
    )

    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200

    assert fake.closed is True
    assert app.state.hipengine_shutdown == {
        "forced": False,
        "grace_seconds": 0.05,
        "active_requests": 0,
        "queued_requests": 0,
    }
    assert app.state.hipengine_readiness.status == "stopped"

    with pytest.raises(ValueError, match="stream_queue_max_chunks"):
        ServerConfig(model="fake-path", stream_queue_max_chunks=1)
    with pytest.raises(ValueError, match="shutdown_grace_seconds"):
        ServerConfig(model="fake-path", shutdown_grace_seconds=-0.1)


def test_generation_batcher_stream_generate_only_fallback_yields_chunks() -> None:
    class GenerateOnlyLLM:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], SamplingParams]] = []

        def generate(self, prompts, sampling_params: SamplingParams) -> list[str]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            return [f"generated:{prompt}" for prompt in prompt_tuple]

    async def run() -> None:
        fake = GenerateOnlyLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
        )

        streamed = [chunk async for chunk in batcher.stream(("fallback",), sampling)]

        assert [chunk.text for chunk in streamed] == ["generated:fallback"]
        assert all(isinstance(chunk, GenerationStreamChunk) for chunk in streamed)
        assert fake.calls == [(("fallback",), sampling)]

    asyncio.run(run())


def test_generation_batcher_skips_cancelled_queued_submit() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        cancelled = asyncio.create_task(batcher.submit(("cancelled",), sampling))
        await asyncio.sleep(0)
        cancelled.cancel()
        try:
            await cancelled
        except asyncio.CancelledError:
            pass
        else:  # pragma: no cover - defensive guard for cancellation semantics
            raise AssertionError("cancelled submit task did not raise CancelledError")

        live = await batcher.submit(("live",), sampling)

        assert live == ["generated:live"]
        assert fake.calls == [(("live",), sampling)]

    asyncio.run(run())


def test_generation_batcher_rejects_when_queue_cap_is_full() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.01,
            max_queue_size=1,
            retry_after_seconds=2,
        )

        first = asyncio.create_task(batcher.submit(("one",), sampling))
        await asyncio.sleep(0)
        with pytest.raises(OpenAIHTTPError) as exc_info:
            await batcher.submit(
                ("two",),
                sampling,
                error_extra={
                    "hipengine": {
                        "routing": {
                            "reason": "engine_busy",
                            "overload_source": "generation_queue_cap",
                        }
                    }
                },
            )

        exc = exc_info.value
        assert exc.status_code == 429
        assert exc.code == "engine_busy"
        assert exc.extra == {
            "hipengine": {
                "routing": {
                    "reason": "engine_busy",
                    "overload_source": "generation_queue_cap",
                }
            }
        }
        assert exc.headers == {"Retry-After": "2"}
        assert await first == ["generated:one"]
        assert fake.calls == [(("one",), sampling)]

    asyncio.run(run())


def test_generation_batcher_limits_active_request_group_size() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.01,
            max_active_requests=1,
        )

        first = asyncio.create_task(batcher.submit(("one",), sampling))
        await asyncio.sleep(0)
        second = asyncio.create_task(batcher.submit(("two",), sampling))

        first_result, second_result = await asyncio.gather(first, second)

        assert first_result == ["generated:one"]
        assert second_result == ["generated:two"]
        assert fake.calls == [(("one",), sampling), (("two",), sampling)]
        assert batcher.active_requests() == 0
        assert batcher.max_active_requests() == 1

    asyncio.run(run())


def test_generation_batcher_applies_route_specific_group_limit() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.01,
            max_active_requests=8,
            route_max_active_requests={_SPECULATIVE_MTP_DEFAULT_ROUTE: 4},
        )

        results = await asyncio.gather(
            *(batcher.submit((f"prompt-{index}",), sampling) for index in range(8))
        )

        assert results == [[f"generated:prompt-{index}"] for index in range(8)]
        assert fake.calls == [
            ((("prompt-0", "prompt-1", "prompt-2", "prompt-3")), sampling),
            ((("prompt-4", "prompt-5", "prompt-6", "prompt-7")), sampling),
        ]

    asyncio.run(run())


def test_generation_batcher_uses_registered_plain_ar_cap_without_widening_mtp() -> None:
    async def run() -> None:
        class FakeNativeC8LLM(FakeLLM):
            server_plain_ar_max_active_requests = 8

        fake = FakeNativeC8LLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.01,
            max_active_requests=8,
            route_max_active_requests={
                _SPECULATIVE_MTP_DEFAULT_ROUTE: 4,
                _SPECULATIVE_MTP_BATCH_ROUTE: 4,
                _SPECULATIVE_MTP_AUTO_ROUTE: 4,
            },
        )

        results = await asyncio.gather(
            *(batcher.submit((f"prompt-{index}",), sampling) for index in range(8))
        )

        assert results == [[f"generated:prompt-{index}"] for index in range(8)]
        assert fake.calls == [
            (tuple(f"prompt-{index}" for index in range(8)), sampling),
        ]
        assert batcher._route_request_cap(_SPECULATIVE_MTP_DEFAULT_ROUTE) == 8
        assert batcher._route_request_cap(_SPECULATIVE_MTP_AUTO_ROUTE) == 8
        assert batcher._route_request_cap(_SPECULATIVE_MTP_BATCH_ROUTE) == 4

    asyncio.run(run())


def test_server_auto_quant_keeps_gguf_route_group_limits() -> None:
    app = create_app(
        ServerConfig(
            model="/models/Qwen3.6-35B-A3B-Q4_K_M.gguf",
            eager_load=False,
            max_active_requests=8,
        ),
        llm=FakeLLM(),
    )

    batcher = app.state.hipengine_generation_batcher

    assert batcher._route_max_active_requests == {
        _SPECULATIVE_MTP_DEFAULT_ROUTE: 4,
        _SPECULATIVE_MTP_BATCH_ROUTE: 4,
        _SPECULATIVE_MTP_AUTO_ROUTE: 4,
    }


def test_server_production_gguf_uses_backend_physical_mtp_route_limit() -> None:
    async def run() -> None:
        class FakeProductionGGUF(FakeLLM):
            supports_speculative_mtp = True
            backend = "hip_gfx1100"
            resolved_execution_profile = "production"

            def generate_speculative_mtp_detailed(
                self,
                prompts,
                sampling_params: SamplingParams,
            ):
                return self.generate_detailed(prompts, sampling_params)

        llm = FakeProductionGGUF()
        app = create_app(
            ServerConfig(
                model="/models/Qwen3.6-35B-A3B-Q4_K_M.gguf",
                backend="hip_gfx1100",
                execution_profile="production",
                eager_load=False,
                generation_batch_window_ms=10.0,
                max_active_requests=8,
            ),
            llm=llm,
        )
        batcher = app.state.hipengine_generation_batcher

        assert batcher._route_max_active_requests == {
            _SPECULATIVE_MTP_DEFAULT_ROUTE: 4,
            _SPECULATIVE_MTP_BATCH_ROUTE: 8,
            _SPECULATIVE_MTP_AUTO_ROUTE: 4,
        }
        sampling = SamplingParams(max_tokens=2)
        results = await asyncio.gather(
            *(
                batcher.submit(
                    (f"prompt-{index}",),
                    sampling,
                    route=_SPECULATIVE_MTP_BATCH_ROUTE,
                )
                for index in range(8)
            )
        )
        await batcher.shutdown(grace_seconds=0.1)

        assert results == [[f"generated:prompt-{index}"] for index in range(8)]
        assert llm.calls == [
            (tuple(f"prompt-{index}" for index in range(8)), sampling),
        ]

    asyncio.run(run())


def test_server_production_gguf_width_ignores_retired_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeLLM()
    llm.resolved_execution_profile = "production"
    monkeypatch.setenv("HIPENGINE_GGUF_SPECDEC2_MTP2_MAX_REQUESTS", "4")

    app = create_app(
        ServerConfig(
            model="/models/Qwen3.6-35B-A3B-Q4_K_M.gguf",
            backend="hip_gfx1100",
            execution_profile="production",
            eager_load=False,
            max_active_requests=8,
        ),
        llm=llm,
    )

    assert app.state.hipengine_generation_batcher._route_max_active_requests == {
        _SPECULATIVE_MTP_DEFAULT_ROUTE: 4,
        _SPECULATIVE_MTP_BATCH_ROUTE: 8,
        _SPECULATIVE_MTP_AUTO_ROUTE: 4,
    }


def test_generation_batcher_applies_mtp_route_group_limit() -> None:
    async def run() -> None:
        class FakeMTPLLM(FakeLLM):
            supports_speculative_mtp = True

            def generate_speculative_mtp_detailed(self, prompts, sampling_params: SamplingParams):
                return self.generate_detailed(prompts, sampling_params)

        fake = FakeMTPLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.01,
            max_active_requests=8,
            route_max_active_requests={
                _SPECULATIVE_MTP_DEFAULT_ROUTE: 4,
                _SPECULATIVE_MTP_BATCH_ROUTE: 4,
            },
        )

        results = await asyncio.gather(
            *(
                batcher.submit((f"prompt-{index}",), sampling, route=_SPECULATIVE_MTP_BATCH_ROUTE)
                for index in range(8)
            )
        )

        assert results == [[f"generated:prompt-{index}"] for index in range(8)]
        assert fake.calls == [
            (
                ("prompt-0", "prompt-1", "prompt-2", "prompt-3"),
                sampling,
            ),
            (
                ("prompt-4", "prompt-5", "prompt-6", "prompt-7"),
                sampling,
            )
        ]

    asyncio.run(run())


def test_generation_batcher_re_resolves_model_plan_at_realized_c2_before_mutation() -> None:
    async def run() -> None:
        fake = RealizedGroupArtifactScopedSpeculativeMTPFakeLLM()
        sampling = SamplingParams(max_tokens=25)
        c1_plan = fake.resolve_speculative_mtp_serving_plan(
            realized_group_rows=1,
            sampling_mode="greedy_fast",
            kv_storage="bf16",
            memory_fit=True,
        )
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.01,
            max_active_requests=2,
            route_max_active_requests={_SPECULATIVE_MTP_AUTO_ROUTE: 2},
        )

        results = await asyncio.gather(
            *(
                batcher.submit(
                    (prompt,),
                    sampling,
                    detailed=True,
                    include_batch_metadata=True,
                    route=_SPECULATIVE_MTP_AUTO_ROUTE,
                    route_decision=c1_plan,
                )
                for prompt in ("one", "two")
            )
        )

        assert all(isinstance(result, _QueuedBatchResult) for result in results)
        assert [call["realized_group_rows"] for call in fake.plan_calls] == [1, 2]
        assert fake.calls == []
        assert len(fake.mtp_calls) == 1
        assert fake.mtp_calls[0][0] == ("one", "two")
        for result in results:
            assert result.generation_shape is not None
            assert result.generation_shape["route"] == _SPECULATIVE_MTP_BATCH_ROUTE
            decision = result.generation_shape["route_decision"]
            assert decision["reason"] == "qualified_automatic_c2_b3"
            assert decision["realized_group_rows"] == 2
            assert decision["policy_cell"] == "fake-qwen38-q4km-c2-b3"

    asyncio.run(run())


def test_generation_batcher_shape_separates_c8_queue_from_c4_verifier_groups() -> None:
    async def run() -> None:
        fake = SpeculativeMTPFakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.01,
            max_active_requests=8,
            route_max_active_requests={_SPECULATIVE_MTP_BATCH_ROUTE: 4},
        )

        results = await asyncio.gather(
            *(
                batcher.submit(
                    (f"prompt-{index}",),
                    sampling,
                    detailed=True,
                    include_batch_metadata=True,
                    route=_SPECULATIVE_MTP_BATCH_ROUTE,
                )
                for index in range(8)
            )
        )

        assert all(isinstance(result, _QueuedBatchResult) for result in results)
        shapes = [result.generation_shape for result in results]
        assert all(shape is not None for shape in shapes)
        queue_group_ids = [shape["queue_group"]["id"] for shape in shapes]
        assert len(set(queue_group_ids)) == 2
        assert sorted(queue_group_ids.count(group_id) for group_id in set(queue_group_ids)) == [4, 4]
        for shape in shapes:
            assert shape["route"] == _SPECULATIVE_MTP_BATCH_ROUTE
            assert shape["route_cap"] == {
                "scope": "queue_requests",
                "value": 4,
                "applied": True,
            }
            assert shape["queue_group"]["request_count"] == 4
            assert shape["queue_group"]["prompt_rows"] == 4
            assert shape["queue_group"]["item_prompt_rows"] == 1
            assert shape["backend_groups"][0]["input_rows"] == 4
            assert shape["backend_groups"][0]["actual_group_rows"] == [4]
            assert shape["backend_groups"][0]["max_actual_group_rows"] == 4
            assert shape["backend_groups"][0]["verifier_rows"] == 12
            assert shape["verifier_rows"] == 12

    asyncio.run(run())


def test_generation_batcher_keeps_c6_direct_when_removed_split_env_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS", "1")
    monkeypatch.setenv("HIPENGINE_QWEN35_AVOID_C6_GROUPS", "1")

    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.01,
            max_active_requests=8,
        )

        results = await asyncio.gather(
            *(batcher.submit((f"prompt-{index}",), sampling) for index in range(6))
        )

        assert results == [[f"generated:prompt-{index}"] for index in range(6)]
        assert fake.calls == [(tuple(f"prompt-{index}" for index in range(6)), sampling)]

    asyncio.run(run())


def test_generation_batcher_keeps_incompatible_sampling_separate() -> None:
    async def run() -> None:
        fake = FakeLLM()
        first_sampling = SamplingParams(max_tokens=1)
        second_sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), first_sampling),
            batcher.submit(("two",), second_sampling),
        )

        assert first == ["generated:one"]
        assert second == ["generated:two"]
        assert fake.calls == [(("one",), first_sampling), (("two",), second_sampling)]

    asyncio.run(run())


def test_generation_batcher_keeps_different_deadlines_separate() -> None:
    async def run() -> None:
        fake = FakeLLM()
        first_sampling = SamplingParams(max_tokens=2, deadline_at=100.0)
        second_sampling = SamplingParams(max_tokens=2, deadline_at=101.0)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), first_sampling),
            batcher.submit(("two",), second_sampling),
        )

        assert first == ["generated:one"]
        assert second == ["generated:two"]
        assert fake.calls == [(("one",), first_sampling), (("two",), second_sampling)]

    asyncio.run(run())


def test_generation_batcher_coalesces_default_route_without_cross_request_cancellation() -> None:
    async def run() -> None:
        fake = FakeLLM()
        first_token = GenerationCancellationToken()
        second_token = GenerationCancellationToken()
        first_sampling = SamplingParams(max_tokens=2, cancellation_token=first_token)
        second_sampling = SamplingParams(max_tokens=2, cancellation_token=second_token)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), first_sampling),
            batcher.submit(("two",), second_sampling),
        )

        assert first == ["generated:one"]
        assert second == ["generated:two"]
        assert len(fake.calls) == 1
        prompts, grouped_sampling = fake.calls[0]
        assert prompts == ("one", "two")
        assert grouped_sampling.cancellation_token is not first_token
        assert grouped_sampling.cancellation_token is not second_token
        assert grouped_sampling.cancellation_token is not None
        assert grouped_sampling.cancellation_token.cancelled is False
        first_token.cancel(FinishDetails(reason="cancelled", cancelled=True))
        assert grouped_sampling.cancellation_token.cancelled is False
        grouped_sampling.cancellation_token.raise_if_cancelled()
        second_token.cancel(FinishDetails(reason="cancelled", cancelled=True))
        assert grouped_sampling.cancellation_token.cancelled is True
        with pytest.raises(GenerationCancelled):
            grouped_sampling.cancellation_token.raise_if_cancelled()

    asyncio.run(run())


def test_generation_batcher_live_neighbor_survives_member_cancellation() -> None:
    class CancellationAwareLLM(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.resume = threading.Event()

        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            self.started.set()
            assert self.resume.wait(timeout=5.0)
            assert sampling_params.cancellation_token is not None
            sampling_params.cancellation_token.raise_if_cancelled()
            return super().generate_detailed(prompts, sampling_params)

    async def run() -> None:
        fake = CancellationAwareLLM()
        first_token = GenerationCancellationToken()
        second_token = GenerationCancellationToken()
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )
        first_task = asyncio.create_task(
            batcher.submit(("one",), SamplingParams(max_tokens=2, cancellation_token=first_token))
        )
        second_task = asyncio.create_task(
            batcher.submit(("two",), SamplingParams(max_tokens=2, cancellation_token=second_token))
        )

        assert await asyncio.to_thread(fake.started.wait, 5.0)
        first_token.cancel()
        fake.resume.set()
        first, second = await asyncio.gather(first_task, second_task)

        assert first == ["generated:one"]
        assert second == ["generated:two"]
        assert fake.calls[0][0] == ("one", "two")

    asyncio.run(run())


def test_await_with_request_control_caller_cancel_closes_child_task() -> None:
    async def run() -> None:
        token = GenerationCancellationToken()
        started = asyncio.Event()
        cancel_seen = asyncio.Event()
        closed = asyncio.Event()

        async def child() -> None:
            started.set()
            try:
                while not token.cancelled:
                    await asyncio.sleep(0.001)
                cancel_seen.set()
            finally:
                closed.set()

        control = SimpleNamespace(
            deadline_at=None,
            disconnected=None,
            poll_interval_s=0.001,
            cancellation_token=token,
        )
        waiter = asyncio.create_task(_await_with_request_control(child(), control))
        await started.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert token.cancelled
        assert token.finish_details == FinishDetails(reason="cancelled", cancelled=True)
        assert cancel_seen.is_set()
        assert closed.is_set()

    asyncio.run(run())


def test_await_with_request_control_caller_cancel_shields_child_ack_from_level_cancellation() -> None:
    async def run() -> None:
        token = GenerationCancellationToken()
        started = asyncio.Event()
        cancel_seen = asyncio.Event()
        release = asyncio.Event()
        closed = asyncio.Event()
        waiter_done = asyncio.Event()
        scopes: list[anyio.CancelScope] = []

        async def child() -> None:
            started.set()
            try:
                while not token.cancelled:
                    await asyncio.sleep(0.001)
                cancel_seen.set()
                await release.wait()
            finally:
                closed.set()

        async def waiter() -> None:
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                await _await_with_request_control(
                    child(),
                    SimpleNamespace(
                        deadline_at=None,
                        disconnected=None,
                        poll_interval_s=0.001,
                        cancellation_token=token,
                    ),
                )
            waiter_done.set()

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(waiter)
            await started.wait()
            scopes[0].cancel()
            await cancel_seen.wait()
            try:
                await asyncio.sleep(0.01)
                assert not waiter_done.is_set()
            finally:
                release.set()
            await waiter_done.wait()

        assert closed.is_set()

    anyio.run(run)


def test_await_with_request_control_deadline_waits_for_child_cancel_ack() -> None:
    async def run() -> None:
        token = GenerationCancellationToken()
        started = asyncio.Event()
        cancel_seen = asyncio.Event()

        async def child() -> None:
            started.set()
            while not token.cancelled:
                await asyncio.sleep(0.001)
            cancel_seen.set()

        control = SimpleNamespace(
            deadline_at=time.perf_counter() + 0.01,
            disconnected=None,
            poll_interval_s=0.001,
            cancellation_token=token,
        )
        with pytest.raises(OpenAIHTTPError) as error:
            await _await_with_request_control(child(), control)
        assert error.value.status_code == 408
        assert error.value.code == "deadline_exceeded"
        assert started.is_set()
        assert cancel_seen.is_set()

    asyncio.run(run())


def test_stream_completion_caller_cancel_records_one_cancelled_request() -> None:
    class CooperativeStreamLLM(FakeLLM):
        supports_controlled_streaming = True

        def stream_detailed(self, prompt: str, sampling_params: SamplingParams):
            yield GenerationStreamChunk(text=f"{prompt}:0")
            token = sampling_params.cancellation_token
            assert token is not None
            while not token.cancelled:
                time.sleep(0.001)
            token.raise_if_cancelled()

    async def run() -> None:
        fake = CooperativeStreamLLM()
        app = create_app(
            ServerConfig(
                model="fake-path",
                served_model_name="fake-model",
                eager_load=False,
                metrics="prometheus",
                generation_batch_window_ms=0.0,
            ),
            llm=fake,
        )

        receive_calls = 0

        async def connected_receive() -> dict[str, object]:
            nonlocal receive_calls
            receive_calls += 1
            await asyncio.sleep(3600)
            return {"type": "http.disconnect"}

        raw_request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/completions",
                "headers": [],
                "query_string": b"",
            },
            connected_receive,
        )
        endpoint = next(route.endpoint for route in app.routes if route.path == "/v1/completions")
        response = await endpoint(
            CompletionRequest(
                model="fake-model",
                prompt="active-disconnect",
                max_tokens=8,
                stream=True,
            ),
            raw_request,
            "anonymous",
        )
        stream = response.body_iterator
        first = await anext(stream)
        assert "active-disconnect:0" in first

        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.01)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        metrics = app.state.hipengine_server_metrics
        assert metrics.request_total == 1
        assert metrics.request_failed_total == 1
        assert metrics.request_cancelled_total == 1
        assert receive_calls == 0
        assert app.state.hipengine_generation_batcher.active_requests() == 0

    asyncio.run(run())


def test_request_control_maps_http_disconnect_to_cancelled_error() -> None:
    async def run() -> None:
        async def receive() -> dict[str, object]:
            return {"type": "http.disconnect"}

        raw_request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/completions",
                "headers": [],
                "query_string": b"",
            },
            receive,
        )
        control = _request_control(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            CompletionRequest(model="fake-model", prompt="hello"),
            raw_request,
        )

        async def work() -> str:
            await asyncio.sleep(1.0)
            return "late"

        try:
            await _await_with_request_control(work(), control)
        except OpenAIHTTPError as exc:
            assert exc.status_code == 499
            assert exc.error_type == "cancelled_error"
            assert exc.code == "cancelled"
            assert exc.finish_details == {"reason": "cancelled", "cancelled": True}
            assert control.cancellation_token.cancelled is True
            assert control.cancellation_token.finish_details.to_json_dict() == exc.finish_details
        else:  # pragma: no cover - defensive guard for cancellation semantics
            raise AssertionError("disconnect did not cancel request")

    asyncio.run(run())


@pytest.mark.parametrize(
    ("endpoint", "payload", "stream", "explicit_max_tokens"),
    [
        (
            "/v1/completions",
            {"model": "fake-model", "prompt": "hello prepared prompt", "max_tokens": 2},
            False,
            True,
        ),
        (
            "/v1/completions",
            {"model": "fake-model", "prompt": "hello prepared prompt", "stream": True},
            True,
            False,
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello prepared prompt"}],
                "max_tokens": 2,
            },
            False,
            True,
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello prepared prompt"}],
                "stream": True,
            },
            True,
            False,
        ),
    ],
)
def test_server_prepares_each_text_prompt_once_for_generation(
    endpoint: str,
    payload: dict[str, Any],
    stream: bool,
    explicit_max_tokens: bool,
) -> None:
    from hipengine.generation.registry import PreparedPromptInput

    fake = PromptPreparingFakeLLM(stream=stream)
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=64,
            chat_default_max_tokens=None,
        ),
        llm=fake,
    )
    response = TestClient(app).post(endpoint, json=payload)

    assert response.status_code == 200
    assert len(fake.calls) == 1
    prepared = fake.calls[0][0][0]
    assert isinstance(prepared, PreparedPromptInput)
    assert tuple(prepared) == prepared.token_ids
    assert str(prepared) == prepared.source_text
    prompt_tokenize_calls = [
        text for text in fake.tokenize_calls if "hello prepared prompt" in text
    ]
    prompt_count_calls = [
        text for text in fake.count_token_calls if "hello prepared prompt" in text
    ]
    assert prompt_tokenize_calls == [prepared.source_text]
    assert prompt_count_calls == []
    assert prepared.tokenize_ms >= 0.0
    assert fake.calls[0][1].max_tokens == (
        2 if explicit_max_tokens else (16 if endpoint == "/v1/completions" else 64 - len(prepared) - 1)
    )


def test_server_reuses_one_prepared_prompt_for_n_choices() -> None:
    from hipengine.generation.registry import PreparedPromptInput

    fake = PromptPreparingFakeLLM()
    fake.detailed_outputs = [
        GenerationOutput(text="first", generated_token_ids=(901,)),
        GenerationOutput(text="second", generated_token_ids=(902,)),
    ]
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=fake,
    )
    response = TestClient(app).post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "duplicate prepared prompt",
            "max_tokens": 1,
            "n": 2,
        },
    )

    assert response.status_code == 200
    prepared_rows = fake.calls[0][0]
    assert len(prepared_rows) == 2
    assert all(isinstance(row, PreparedPromptInput) for row in prepared_rows)
    assert prepared_rows[0] is prepared_rows[1]
    assert fake.tokenize_calls == ["duplicate prepared prompt"]
    assert response.json()["usage"]["prompt_tokens"] == 2 * len(prepared_rows[0])


@pytest.mark.parametrize("stream", [False, True])
def test_chat_passes_auth_scoped_resident_session_key_to_capable_generator(
    stream: bool,
) -> None:
    fake = ResidentSessionPromptFakeLLM(stream=stream)
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "stateful prompt"}],
            "max_tokens": 1,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else None,
            "session": {"id": "session-a", "commit": "append_visible_only"},
        },
    )

    assert response.status_code == 200
    sampling = fake.calls[0][1]
    assert sampling.resident_session_key == hashlib.sha256(
        b"anonymous\0session-a"
    ).hexdigest()
    assert sampling.resident_session_cache_action == "append_visible_only"
    commit = client.get("/v1/hipengine/capabilities").json()["sessions"]["commit_policy"]
    assert commit["resident_state_reuse"] is True
    assert commit["resident_kv_commit"] is True
    assert commit["resident_kv_capacity"] == 1
    assert commit["supported_streaming"] is True


def test_streaming_completion_n_uses_scheduler_token_chunks_for_buffered_deltas() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            outputs = [
                GenerationOutput(
                    text="AB",
                    finish_details=FinishDetails(reason="length", length_limit=2, sampler_mode="greedy_fast"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=0,
                        phase="done",
                        sampler_mode="greedy_fast",
                    ),
                ),
                GenerationOutput(
                    text="CD",
                    finish_details=FinishDetails(reason="length", length_limit=2, sampler_mode="greedy_fast"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=1,
                        phase="done",
                        sampler_mode="greedy_fast",
                    ),
                ),
            ]
            chunk_texts = (("A", "B"), ("C", "D"))
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": request_id,
                        "token_index": token_index,
                        "token_id": 100 + request_id * 10 + token_index,
                        "finished": token_index == 1,
                        "chunk": {
                            "text": text,
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=token_index + 1,
                                row_index=request_id,
                                request_id=str(request_id),
                                phase="answer",
                                sampler_mode="greedy_fast",
                                execution_path="scheduler_native_packed_prefill_serial_decode",
                            ).to_json_dict(),
                        },
                    }
                    for request_id, row in enumerate(chunk_texts)
                    for token_index, text in enumerate(row)
                ]
            }
            return outputs[: len(prompt_tuple)]

    fake = SchedulerChunkFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "one",
            "n": 2,
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    deltas = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] is None
    ]
    assert [(choice["index"], choice["text"]) for choice in deltas] == [
        (0, "A"),
        (0, "B"),
        (1, "C"),
        (1, "D"),
    ]
    assert [
        choice["hipengine"]["decode_state"]["generated_tokens"]
        for choice in deltas
    ] == [1, 2, 1, 2]
    assert [
        choice["hipengine"]["decode_state"]["execution_path"]
        for choice in deltas
    ] == [
        "scheduler_native_packed_prefill_serial_decode",
        "scheduler_native_packed_prefill_serial_decode",
        "scheduler_native_packed_prefill_serial_decode",
        "scheduler_native_packed_prefill_serial_decode",
    ]
    done = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] == "length"
    ]
    assert [(choice["index"], choice["finish_details"]) for choice in done] == [
        (
            0,
            {
                "reason": "length",
                "length_limit": 2,
                "cache_action": "append_none",
                "sampler_mode": "greedy_fast",
                "continuation_eligible": False,
            },
        ),
        (
            1,
            {
                "reason": "length",
                "length_limit": 2,
                "cache_action": "append_none",
                "sampler_mode": "greedy_fast",
                "continuation_eligible": False,
            },
        ),
    ]
    assert payloads[-1]["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    assert "data: [DONE]" in response.text
    assert fake.calls[0][0] == ("one", "one")
    assert fake.stream_calls == []


def test_blocked_engine_command_does_not_stall_an_in_flight_stream() -> None:
    """An engine command wait belongs to its request, not to the event loop.

    The server tokenizes prompts through the resident engine service, whose
    commands wait on a single driver thread. Running that wait on the event loop
    thread stops the server from flushing anything else: an in-flight SSE stream
    pauses and then delivers its remaining tokens in one burst after the wait, and
    health checks, metrics, and cancellation queue behind it too. The wait must run
    on a worker thread, so the stream keeps moving and the engine call never runs
    on the loop thread.
    """

    fake = BlockingPrepareFakeLLM(stream_delay_s=0.05)
    app = create_app(
        ServerConfig(model="/models/fake", served_model_name="fake-model"),
        llm=fake,
    )

    async def run() -> list[int]:
        loop_thread = threading.get_ident()
        started = time.monotonic()
        body_times: list[float] = []

        def record(message: Mapping[str, Any]) -> None:
            if message["type"] == "http.response.body" and message.get("body"):
                body_times.append(time.monotonic() - started)

        stream_task = asyncio.create_task(
            _asgi_chat_post(
                app,
                {
                    "model": "fake-model",
                    "messages": [{"role": "user", "content": "stream please"}],
                    "max_tokens": 100000,
                    "stream": True,
                },
                on_body=record,
            )
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not body_times:
            await asyncio.sleep(0.01)
        assert body_times, "the streaming request never delivered a body message"

        # A blocked event loop cannot report on itself, so the block is observed
        # from a thread. The watchdog always releases the fake, which keeps a
        # stalled loop a failed assertion instead of a hung test.
        progress_during_block: list[int] = []

        def watchdog() -> None:
            baseline = len(body_times)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if len(body_times) > baseline:
                    progress_during_block.append(len(body_times) - baseline)
                    break
                time.sleep(0.01)
            fake.tokenize_release.set()
            fake.stream_stop.set()

        slow_task = asyncio.create_task(
            _asgi_chat_post(
                app,
                {
                    "model": "fake-model",
                    "messages": [{"role": "user", "content": "SLOW-PREP prompt"}],
                    "max_tokens": 4,
                },
            )
        )
        watcher = threading.Thread(target=watchdog, daemon=True)
        watcher.start()
        assert await asyncio.to_thread(fake.tokenize_entered.wait, 20.0)
        # Joining from the loop would block the very stream this test measures.
        await asyncio.to_thread(watcher.join, 15.0)

        assert progress_during_block, (
            "the in-flight stream stalled while another request's engine command was "
            "blocked on the event loop thread"
        )
        assert await slow_task == 200
        await asyncio.wait_for(stream_task, timeout=10.0)
        # The engine wait itself must never have run on the event loop thread.
        assert fake.tokenize_threads
        assert loop_thread not in fake.tokenize_threads
        return progress_during_block

    assert asyncio.run(run())


def test_streaming_chat_completion_n_uses_scheduler_token_chunks_for_buffered_answer_deltas() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            outputs = [
                GenerationOutput(
                    text="<think>r0</think>A",
                    finish_details=FinishDetails(reason="length", length_limit=2, sampler_mode="greedy_fast"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=4,
                        row_index=0,
                        phase="done",
                        sampler_mode="greedy_fast",
                    ),
                ),
                GenerationOutput(
                    text="BC",
                    finish_details=FinishDetails(reason="length", length_limit=2, sampler_mode="greedy_fast"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=1,
                        phase="done",
                        sampler_mode="greedy_fast",
                    ),
                ),
            ]
            chunk_texts = (("<thi", "nk>r0</think>", "A"), ("B", "C"))
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": request_id,
                        "token_index": token_index,
                        "token_id": 200 + request_id * 10 + token_index,
                        "finished": token_index == len(row) - 1,
                        "chunk": {
                            "text": text,
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=token_index + 1,
                                row_index=request_id,
                                request_id=str(request_id),
                                phase="answer",
                                sampler_mode="greedy_fast",
                                execution_path="scheduler_native_packed_prefill_serial_decode",
                            ).to_json_dict(),
                        },
                    }
                    for request_id, row in enumerate(chunk_texts)
                    for token_index, text in enumerate(row)
                ]
            }
            return outputs[: len(prompt_tuple)]

    fake = SchedulerChunkFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "n": 2,
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    deltas = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] is None
    ]
    assert [(choice["index"], choice["delta"]) for choice in deltas] == [
        (0, {"role": "assistant"}),
        (0, {"reasoning_content": "r0"}),
        (0, {"content": "A"}),
        (1, {"role": "assistant"}),
        (1, {"content": "B"}),
        (1, {"content": "C"}),
    ]
    chunk_deltas = [choice for choice in deltas if "role" not in choice["delta"]]
    assert [
        choice["hipengine"]["decode_state"]["generated_tokens"]
        for choice in chunk_deltas
    ] == [2, 3, 1, 2]
    assert {
        choice["hipengine"]["decode_state"]["execution_path"]
        for choice in chunk_deltas
    } == {"scheduler_native_packed_prefill_serial_decode"}
    assert [
        choice["hipengine"]["phase"]
        for choice in chunk_deltas
    ] == ["think", "answer", "answer", "answer"]
    done = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] == "length"
    ]
    assert [(choice["index"], choice["finish_details"]) for choice in done] == [
        (
            0,
            {
                "reason": "length",
                "length_limit": 2,
                "cache_action": "append_none",
                "sampler_mode": "greedy_fast",
                "phase": "answer",
                "continuation_eligible": False,
            },
        ),
        (
            1,
            {
                "reason": "length",
                "length_limit": 2,
                "cache_action": "append_none",
                "sampler_mode": "greedy_fast",
                "phase": "answer",
                "continuation_eligible": False,
            },
        ),
    ]
    prompt_tokens = sum(fake.count_tokens(prompt) for prompt in fake.calls[0][0])
    assert payloads[-1]["usage"] == {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 2,
        "total_tokens": prompt_tokens + 2,
    }
    assert "data: [DONE]" in response.text
    assert len(fake.calls) == 1
    assert fake.stream_calls == []


def test_generation_batcher_finishes_queued_items_when_worker_raises() -> None:
    """A dispatch-loop exception must fail queued requests, not orphan them.

    Regression guard for the K4 server hang: a ValueError escaping the
    batcher's dispatch loop killed the worker while queued items waited on
    futures that nothing would ever resolve. The handler now finishes every
    in-flight item with the exception before the worker exits.
    """

    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        def exploding_route_cap(route: str) -> None:
            raise ValueError("route cap resolution failed")

        batcher._route_request_cap = exploding_route_cap  # type: ignore[method-assign]

        async def guarded_submit(*prompts: str):
            return await asyncio.wait_for(
                batcher.submit(prompts, sampling),
                timeout=10.0,
            )

        results = await asyncio.gather(
            guarded_submit("one"),
            guarded_submit("two"),
            return_exceptions=True,
        )
        assert all(isinstance(result, ValueError) for result in results), results

    asyncio.run(run())
