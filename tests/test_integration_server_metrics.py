"""OpenAI server integration tests: metrics.

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
    BackendCancelledFakeLLM,
    FakeLLM,
)
from tests.support.server_api_support import (
    _labeled_metric_value,
    _metric_value,
)


def test_ready_reports_bounded_prefix_cache_ownership() -> None:
    fake = FakeLLM()
    prefix = {
        "mode": "radix",
        "block_size_tokens": 256,
        "snapshot_entries": 2,
        "snapshot_limit": 4,
        "retained_snapshot_entries": 2,
        "snapshot_bytes": 1024,
        "retained_kv_pages": 3,
        "retained_kv_bytes": 12_288,
        "resident_bytes": 13_312,
        "resident_limit_bytes": 65_536,
    }
    fake.live_loop_snapshot = lambda: {"runner": {"prefix_cache": prefix}}
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=fake,
    )

    body = TestClient(app).get("/ready").json()

    assert body["prefix_cache"] == prefix


def test_info_mode_logs_one_request_summary_per_completed_request(caplog) -> None:
    """--info logs tokens in/out, rates, and KV memory once per request.

    The summary is the only place the server reports a request's own token
    counts, phase rates, and KV footprint together, so it must appear exactly
    once per completed request and must stay silent when the flag is off.
    """

    fake = FakeLLM(outputs=["hello there"])
    app = create_app(
        ServerConfig(model="/models/fake", served_model_name="fake-model", info=True),
        llm=fake,
    )
    client = TestClient(app)

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    usage = response.json()["usage"]
    lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("REQUEST_INFO:")
    ]
    assert len(lines) == 1
    line = lines[0]
    assert "endpoint=/v1/chat/completions" in line
    assert "stream=false" in line
    assert "model=fake-model" in line
    assert f"tokens_in={usage['prompt_tokens']}" in line
    assert f"tokens_out={usage['completion_tokens']}" in line
    assert "wall_ms=" in line

    caplog.clear()
    quiet = create_app(
        ServerConfig(model="/models/fake", served_model_name="fake-model"),
        llm=FakeLLM(outputs=["hello there"]),
    )
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        assert (
            TestClient(quiet)
            .post(
                "/v1/chat/completions",
                json={
                    "model": "fake-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4,
                },
            )
            .status_code
            == 200
        )
    assert "REQUEST_INFO:" not in caplog.text


@pytest.mark.parametrize("mode", [None, "off", "radix"])
def test_server_prefix_cache_default_and_overrides(monkeypatch, mode) -> None:
    from hipengine.kvcache import PREFIX_CACHE_DEFAULT

    monkeypatch.delenv("HIPENGINE_PREFIX_CACHE", raising=False)
    assert PREFIX_CACHE_DEFAULT == "radix"
    # The CLI preserves an omitted value so the engine can resolve its
    # backend-owned default; explicit environment and command-line values remain
    # visible at parse time.
    assert build_parser().parse_args(["--model", "fake-path"]).prefix_cache is None
    if mode is not None:
        monkeypatch.setenv("HIPENGINE_PREFIX_CACHE", mode)
    args = build_parser().parse_args(["--model", "fake-path"])
    assert args.prefix_cache == mode
    for explicit in ("off", "radix"):
        assert build_parser().parse_args(
            ["--model", "fake-path", "--prefix-cache", explicit]
        ).prefix_cache == explicit
        app = create_app(
            ServerConfig(model="fake-path", eager_load=False, prefix_cache=explicit),
            llm=FakeLLM(),
        )
        assert app.state.hipengine_prefix_cache_mode == explicit
    app = create_app(ServerConfig(model="fake-path", eager_load=False), llm=FakeLLM())
    assert app.state.hipengine_prefix_cache_mode == PREFIX_CACHE_DEFAULT


def test_metrics_prefix_cache_and_generation_batch_cli_env_defaults(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GENERATION_BATCH_WINDOW_MS", raising=False)
    monkeypatch.delenv("HIPENGINE_DEBUG", raising=False)
    monkeypatch.delenv("HIPENGINE_CHAT_DEFAULT_MAX_TOKENS", raising=False)
    monkeypatch.delenv("HIPENGINE_STARTUP_CHAT_SMOKE", raising=False)
    monkeypatch.delenv("HIPENGINE_STARTUP_SCRATCH_PROBE", raising=False)
    monkeypatch.delenv("HIPENGINE_STARTUP_MIN_FREE_MIB", raising=False)
    monkeypatch.delenv("HIPENGINE_REQUEST_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("HIPENGINE_STREAM_QUEUE_MAX_CHUNKS", raising=False)
    monkeypatch.delenv("HIPENGINE_SHUTDOWN_GRACE_SECONDS", raising=False)
    monkeypatch.delenv("HIPENGINE_MAX_QUEUED_REQUESTS", raising=False)
    monkeypatch.delenv("HIPENGINE_MAX_ACTIVE_REQUESTS", raising=False)
    monkeypatch.delenv("HIPENGINE_MAX_CHAT_SESSIONS", raising=False)
    monkeypatch.delenv("HIPENGINE_REPLAY_DIR", raising=False)
    monkeypatch.delenv("HIPENGINE_REPLAY_REDACTION", raising=False)
    monkeypatch.delenv("HIPENGINE_INFO", raising=False)
    default_args = build_parser().parse_args(["--model", "fake-path"])
    assert default_args.backend == "auto"
    assert default_args.quant == "auto"
    assert default_args.generation_batch_window_ms == 0.0
    assert default_args.debug is False
    assert default_args.info is False
    assert default_args.chat_default_max_tokens == 4096
    assert default_args.startup_chat_smoke is True
    assert default_args.startup_scratch_probe is True
    assert default_args.startup_min_free_mib is None
    assert default_args.request_timeout_ms is None
    assert default_args.stream_queue_max_chunks == 16
    assert default_args.shutdown_grace_seconds == 5.0
    assert default_args.max_queued_requests is None
    assert default_args.max_active_requests is None
    assert default_args.max_chat_sessions is None
    assert default_args.replay_dir is None
    assert default_args.replay_redaction == "hash"

    monkeypatch.setenv("HIPENGINE_METRICS", "prometheus")
    monkeypatch.setenv("HIPENGINE_PREFIX_CACHE", "radix")
    monkeypatch.setenv("HIPENGINE_GENERATION_BATCH_WINDOW_MS", "3.5")
    monkeypatch.setenv("HIPENGINE_DEBUG", "1")
    monkeypatch.setenv("HIPENGINE_CHAT_DEFAULT_MAX_TOKENS", "auto")
    monkeypatch.setenv("HIPENGINE_STARTUP_CHAT_SMOKE", "0")
    monkeypatch.setenv("HIPENGINE_STARTUP_SCRATCH_PROBE", "0")
    monkeypatch.setenv("HIPENGINE_STARTUP_MIN_FREE_MIB", "512")
    monkeypatch.setenv("HIPENGINE_REQUEST_TIMEOUT_MS", "250.5")
    monkeypatch.setenv("HIPENGINE_STREAM_QUEUE_MAX_CHUNKS", "9")
    monkeypatch.setenv("HIPENGINE_SHUTDOWN_GRACE_SECONDS", "2.5")
    monkeypatch.setenv("HIPENGINE_MAX_QUEUED_REQUESTS", "7")
    monkeypatch.setenv("HIPENGINE_MAX_ACTIVE_REQUESTS", "6")
    monkeypatch.setenv("HIPENGINE_MAX_CHAT_SESSIONS", "5")
    monkeypatch.setenv("HIPENGINE_REPLAY_DIR", "/tmp/hipengine-replay")
    monkeypatch.setenv("HIPENGINE_REPLAY_REDACTION", "none")
    monkeypatch.setenv("HIPENGINE_INFO", "1")
    env_args = build_parser().parse_args(["--model", "fake-path"])
    assert env_args.metrics == "prometheus"
    assert env_args.prefix_cache == "radix"
    assert env_args.generation_batch_window_ms == 3.5
    assert env_args.debug is True
    assert env_args.info is True
    assert env_args.chat_default_max_tokens is None
    assert env_args.startup_chat_smoke is False
    assert env_args.startup_scratch_probe is False
    assert env_args.startup_min_free_mib == 512
    assert env_args.request_timeout_ms == 250.5
    assert env_args.stream_queue_max_chunks == 9
    assert env_args.shutdown_grace_seconds == 2.5
    assert env_args.max_queued_requests == 7
    assert env_args.max_active_requests == 6
    assert env_args.max_chat_sessions == 5
    assert env_args.replay_dir == "/tmp/hipengine-replay"
    assert env_args.replay_redaction == "none"

    cli_args = build_parser().parse_args(
        [
            "--model",
            "fake-path",
            "--metrics",
            "off",
            "--prefix-cache",
            "off",
            "--generation-batch-window-ms",
            "0",
            "--request-timeout-ms",
            "123.5",
            "--stream-queue-max-chunks",
            "6",
            "--shutdown-grace-seconds",
            "1.25",
            "--max-queued-requests",
            "3",
            "--max-active-requests",
            "4",
            "--max-chat-sessions",
            "2",
            "--chat-default-max-tokens",
            "123",
            "--replay-dir",
            "/tmp/hipengine-cli-replay",
            "--replay-redaction",
            "hash",
            "--startup-chat-smoke",
            "--startup-scratch-probe",
            "--startup-min-free-mib",
            "256",
            "--no-debug",
        ]
    )
    assert cli_args.metrics == "off"
    assert cli_args.prefix_cache == "off"
    assert cli_args.generation_batch_window_ms == 0.0
    assert cli_args.request_timeout_ms == 123.5
    assert cli_args.stream_queue_max_chunks == 6
    assert cli_args.shutdown_grace_seconds == 1.25
    assert cli_args.max_queued_requests == 3
    assert cli_args.max_active_requests == 4
    assert cli_args.max_chat_sessions == 2
    assert cli_args.chat_default_max_tokens == 123
    assert cli_args.replay_dir == "/tmp/hipengine-cli-replay"
    assert cli_args.replay_redaction == "hash"
    assert cli_args.startup_chat_smoke is True
    assert cli_args.startup_scratch_probe is True
    assert cli_args.startup_min_free_mib == 256
    assert cli_args.debug is False

    app = create_app(ServerConfig(model="fake-path", eager_load=False, prefix_cache="radix"), llm=FakeLLM())
    assert app.state.hipengine_prefix_cache_mode == "radix"


def test_debug_mode_logs_full_request_and_response_payloads(caplog) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    fake = FakeLLM(outputs=["debug reply"])
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False, debug=True),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1},
    )

    assert response.status_code == 200
    assert "DEBUG_PAYLOAD REQUEST POST /v1/completions" in caplog.text
    assert '"prompt":"hello"' in caplog.text
    assert "DEBUG_PAYLOAD RESPONSE POST /v1/completions status=200" in caplog.text
    assert '"text":"debug reply"' in caplog.text


def test_metrics_endpoint_is_opt_in_and_additive() -> None:
    disabled = create_app(ServerConfig(model="fake-path", eager_load=False), llm=FakeLLM())
    assert TestClient(disabled).get("/metrics").status_code == 404

    fake = FakeLLM(outputs=["alpha beta"])
    fake.kv_pool_stats = SimpleNamespace(
        current_bytes=4096,
        high_water_observed_bytes=8192,
        grow_events=2,
        grow_failures=1,
        shrink_events=3,
        free_pages=4,
        refcounted_pages=5,
        retired_pages=2,
        retired_bytes=8192,
    )
    fake.graph_bucket_stats = SimpleNamespace(
        entries=6,
        hits=7,
        misses=8,
        replay_hit_rate=0.0,
        miss_reasons={"cache_absent": 5, "shape_changed": 3, "bool_bad": True, "nan_bad": float("nan")},
        kernel_time_histogram_ns={
            "le_10us": 2,
            "le_100us": 4,
            "le_1ms": True,
            "le_10ms": float("inf"),
            "gt_10ms": -1,
            "lt_1us": 9,
        },
    )
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            metrics="prometheus",
            max_queued_requests=3,
            max_active_requests=2,
            max_chat_sessions=4,
        ),
        llm=fake,
    )
    client = TestClient(app)

    before = client.get("/metrics")
    assert before.status_code == 200
    assert _metric_value(before.text, "hipengine_requests_total") == 0
    assert _metric_value(before.text, "hipengine_generation_queue_depth") == 0
    assert _metric_value(before.text, "hipengine_request_cancelled_total") == 0
    assert _metric_value(before.text, "hipengine_generation_queue_max_depth") == 3
    assert _metric_value(before.text, "hipengine_generation_worker_active") == 0
    assert _metric_value(before.text, "hipengine_generation_requests_active") == 0
    assert _metric_value(before.text, "hipengine_generation_requests_max_active") == 2
    assert (
        'hipengine_generation_scheduler_fairness_policy_info{policy="fifo_compatible_sampling_key",'
        'compatible_sampling_coalescing="true",continuous_decode="false",preemptive_fairness="false"} 1'
        in before.text
    )
    assert _metric_value(before.text, "hipengine_chat_sessions_active") == 0
    assert _metric_value(before.text, "hipengine_chat_sessions_pending") == 0
    assert _metric_value(before.text, "hipengine_chat_sessions_max_active") == 4

    for prompt in ["one", "two three"]:
        response = client.post(
            "/v1/completions",
            json={"model": "fake-model", "prompt": prompt, "max_tokens": 2},
        )
        assert response.status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    assert _metric_value(metrics.text, "hipengine_requests_total") == 2
    assert _metric_value(metrics.text, "hipengine_request_completed_total") == 2
    assert _metric_value(metrics.text, "hipengine_request_failed_total") == 0
    assert _metric_value(metrics.text, "hipengine_request_rejected_total") == 0
    assert _metric_value(metrics.text, "hipengine_request_cancelled_total") == 0
    assert _metric_value(metrics.text, "hipengine_generation_queue_depth") == 0
    assert _metric_value(metrics.text, "hipengine_generation_queue_max_depth") == 3
    assert _metric_value(metrics.text, "hipengine_generation_worker_active") == 0
    assert _metric_value(metrics.text, "hipengine_generation_requests_active") == 0
    assert _metric_value(metrics.text, "hipengine_generation_requests_max_active") == 2
    assert _metric_value(metrics.text, "hipengine_chat_sessions_active") == 0
    assert _metric_value(metrics.text, "hipengine_chat_sessions_pending") == 0
    assert _metric_value(metrics.text, "hipengine_chat_sessions_max_active") == 4
    assert _metric_value(metrics.text, "hipengine_prompt_tokens_total") == 3
    assert _metric_value(metrics.text, "hipengine_completion_tokens_total") == 4
    assert _metric_value(metrics.text, "hipengine_kv_pool_current_bytes") == 4096
    assert _metric_value(metrics.text, "hipengine_kv_pool_grow_events_total") == 2
    assert _metric_value(metrics.text, "hipengine_kv_pool_grow_failures_total") == 1
    assert _metric_value(metrics.text, "hipengine_kv_pool_shrink_events_total") == 3
    assert _metric_value(metrics.text, "hipengine_kv_pool_free_pages") == 4
    assert _metric_value(metrics.text, "hipengine_kv_pool_refcounted_pages") == 5
    assert _metric_value(metrics.text, "hipengine_kv_pool_retired_pages_total") == 2
    assert _metric_value(metrics.text, "hipengine_kv_pool_retired_bytes_total") == 8192
    assert _metric_value(metrics.text, "hipengine_graph_bucket_entries") == 6
    assert _metric_value(metrics.text, "hipengine_graph_bucket_hits_total") == 7
    assert _metric_value(metrics.text, "hipengine_graph_bucket_misses_total") == 8
    assert _metric_value(metrics.text, "hipengine_graph_bucket_replay_hit_rate") == 7 / 15
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_miss_reason_total", reason="cache_absent") == 5
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_miss_reason_total", reason="shape_changed") == 3
    assert 'hipengine_graph_bucket_miss_reason_total{reason="bool_bad"}' not in metrics.text
    assert 'hipengine_graph_bucket_miss_reason_total{reason="nan_bad"}' not in metrics.text
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="le_10us") == 2
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="le_100us") == 4
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="le_1ms") == 0
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="le_10ms") == 0
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="gt_10ms") == 0
    for bucket in GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS:
        assert f'hipengine_graph_bucket_kernel_time_bucket_total{{bucket="{bucket}"}}' in metrics.text
    assert 'hipengine_graph_bucket_kernel_time_bucket_total{bucket="lt_1us"}' not in metrics.text


def test_metrics_endpoint_exports_resident_loop_d5_observability() -> None:
    fake = FakeLLM()
    fake.live_loop_snapshot = lambda: {
        "schema": 1,
        "kind": "resident_engine_loop_observability",
        "loop": {
            "requests": {
                "pending": 1,
                "admitted_current": 2,
                "active": 2,
                "admitted_total": 7,
                "completed_buffered": 0,
                "reclaimed_total": 5,
            },
            "physical_bucket": {
                "capacity": 4,
                "active_c": 2,
                "occupied_slots": 2,
                "free_slots": 2,
                "active_mask": [True, False, True, False],
                "slot_to_request": [10, None, 12, None],
                "occupancy_ratio": 0.5,
            },
            "work_counts": {"prefill": 3, "decode": 9, "reclaim": 5},
            "latency_seconds": {
                "queue": {"count": 2, "sum": 0.3, "max": 0.2, "p50": 0.15, "p95": 0.2, "samples": [0.1, 0.2]},
                "time_to_first_token": {"count": 2, "sum": 1.5, "max": 1.0, "p50": 0.75, "p95": 1.0, "samples": [0.5, 1.0]},
                "inter_token": {"count": 3, "sum": 0.12, "max": 0.05, "p50": 0.04, "p95": 0.05, "samples": [0.03, 0.04, 0.05]},
                "service": {"count": 2, "sum": 2.5, "max": 1.5, "p50": 1.25, "p95": 1.5, "samples": [1.0, 1.5]},
                "completion": {"count": 2, "sum": 2.8, "max": 1.7, "p50": 1.4, "p95": 1.7, "samples": [1.1, 1.7]},
            },
            "recent_completed": [],
            "scheduler_policy": {
                "prefill_decode_policy": "protect_ttft",
                "prefill_chunk_tokens": 256,
                "last_work_kind": "decode",
            },
        },
        "runner": {
            "model_runner": {
                "capacity": 4,
                "active_request_ids": [10, 12],
                "active_requests": 2,
                "available_sessions": 2,
                "packed_workspace_current_bytes": 0,
                "packed_workspace_owner_sessions": 3,
                "packed_workspace_leased_pool_bytes": 2097152,
                "packed_workspace_release_events": 4,
                "packed_workspace_released_bytes": 3330000000,
                "prefill_transients": {
                    "oracle_owner_bytes": 1048576,
                    "oracle_owner_count_total": 16,
                    "oracle_observed_peak_bytes": 17179869184,
                    "oracle_observed_peak_owners": 16,
                    "hidden_and_bulk_owner_bytes": 5242880,
                    "last_packed_executor_modes": ["layer_outer_packed", None],
                    "kv_attention_sources": ["int8_direct", "bf16_mirror"],
                },
                "persistent_int8_payload_bytes": 8388608,
                "persistent_bf16_payload_bytes": 0,
                "persistent_scale_bytes": 65536,
                "persistent_bf16_mirror_bytes": 0,
                "persistent_kv_total_bytes": 8454144,
            },
            "kv_pool": {
                "current_bytes": 15728640,
                "high_water_observed_bytes": 31457280,
                "current_pages": 3,
                "high_water_observed_pages": 6,
                "free_pages": 0,
                "refcounted_pages": 3,
                "pinned_pages": 1,
                "grow_events": 2,
                "grow_failures": 1,
                "shrink_events": 1,
            },
            "graph_buckets": {
                "entries": 1,
                "captures_total": 2,
                "hits_total": 3,
                "replays_total": 3,
                "invalidations_total": 1,
                "buckets": {
                    "bucket-c2": {
                        "bucket_key": {"key_sha256": "bucket-c2", "active_rows": 2},
                        "entries": 1,
                        "captures": 2,
                        "hits": 3,
                        "replays": 3,
                        "invalidations": 1,
                    }
                },
            },
            "routes": {
                "counts": {
                    "native_full_prefill_rows": 2,
                    "native_incremental_prefill_chunks": 4,
                    "native_packed_decode_steps": 3,
                    "native_c1_decode_steps": 1,
                    "serial_decode_fallback_steps": 0,
                    "resident_fallback_requests": 1,
                },
                "fallback_reasons": {"host_sampling": 1},
                "last_execution_manifest": {
                    "kind": "gguf_ar_serial_fallback_execution_manifest",
                    "mode": "serial_c1_per_row",
                    "rows": 2,
                    "logical_c": 2,
                    "physical_execution_width": 1,
                    "kv_attention_source": "int8_direct",
                    "serial_decode_fallback": True,
                    "throughput_claim_eligible": False,
                },
                "recent_completed": [],
            },
        },
    }
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            metrics="prometheus",
        ),
        llm=fake,
    )

    body = TestClient(app).get("/metrics").text

    assert _metric_value(body, "hipengine_resident_requests_pending") == 1
    assert _metric_value(body, "hipengine_resident_requests_admitted") == 2
    assert _metric_value(body, "hipengine_resident_requests_active") == 2
    assert _metric_value(body, "hipengine_resident_requests_admitted_total") == 7
    assert _metric_value(body, "hipengine_resident_requests_reclaimed_total") == 5
    assert _metric_value(body, "hipengine_resident_bucket_capacity") == 4
    assert _metric_value(body, "hipengine_resident_bucket_active_rows") == 2
    assert _metric_value(body, "hipengine_resident_bucket_occupied_slots") == 2
    assert _metric_value(body, "hipengine_resident_bucket_free_slots") == 2
    assert _metric_value(body, "hipengine_resident_bucket_occupancy_ratio") == 0.5
    assert _metric_value(body, "hipengine_resident_work_prefill_total") == 3
    assert _metric_value(body, "hipengine_resident_work_decode_total") == 9
    assert _metric_value(body, "hipengine_resident_work_reclaim_total") == 5
    assert _metric_value(body, "hipengine_resident_prefill_chunk_tokens") == 256
    assert _metric_value(body, "hipengine_resident_fair_prefill_burst_chunks") == 1
    assert _metric_value(body, "hipengine_resident_consecutive_prefill_chunks") == 0
    assert _metric_value(body, "hipengine_resident_packed_workspace_current_bytes") == 0
    assert _metric_value(body, "hipengine_resident_packed_workspace_owner_sessions") == 3
    assert _metric_value(body, "hipengine_resident_packed_workspace_leased_pool_bytes") == 2097152
    assert _metric_value(body, "hipengine_resident_prefill_oracle_owner_bytes") == 1048576
    assert _metric_value(body, "hipengine_resident_prefill_oracle_owners") == 16
    assert _metric_value(body, "hipengine_resident_prefill_oracle_observed_peak_bytes") == 17179869184
    assert _metric_value(body, "hipengine_resident_prefill_oracle_observed_peak_owners") == 16
    assert _metric_value(body, "hipengine_resident_prefill_hidden_owner_bytes") == 5242880
    assert _metric_value(body, "hipengine_resident_packed_workspace_release_events_total") == 4
    assert _metric_value(body, "hipengine_resident_packed_workspace_released_bytes_total") == 3330000000
    assert _metric_value(body, "hipengine_resident_kv_int8_payload_bytes") == 8388608
    assert _metric_value(body, "hipengine_resident_kv_bf16_payload_bytes") == 0
    assert _metric_value(body, "hipengine_resident_kv_scale_bytes") == 65536
    assert _metric_value(body, "hipengine_resident_kv_bf16_mirror_bytes") == 0
    assert _metric_value(body, "hipengine_resident_kv_total_bytes") == 8454144
    assert (
        'hipengine_resident_bucket_info{active_mask="1010",fair_prefill_burst_chunks="1",'
        'last_work_kind="decode",policy="protect_ttft"} 1'
        in body
    )
    assert (
        'hipengine_resident_route_manifest_info{claim_level="unavailable",'
        'kind="gguf_ar_serial_fallback_execution_manifest",'
        'kv_attention_source="int8_direct",logical_c="2",'
        'mode="serial_c1_per_row",physical_execution_width="1",rows="2",'
        'serial_decode_fallback="true",throughput_claim_eligible="false"} 1'
        in body
    )
    assert _labeled_metric_value(
        body,
        "hipengine_resident_request_latency_seconds",
        kind="time_to_first_token",
        quantile="0.5",
    ) == 0.75
    assert _labeled_metric_value(
        body,
        "hipengine_resident_request_latency_seconds",
        kind="inter_token",
        quantile="0.95",
    ) == 0.05
    assert _labeled_metric_value(
        body,
        "hipengine_resident_request_latency_seconds_sum",
        kind="completion",
    ) == 2.8
    assert _labeled_metric_value(
        body,
        "hipengine_resident_request_latency_seconds_count",
        kind="completion",
    ) == 2
    assert _labeled_metric_value(
        body,
        "hipengine_resident_request_latency_max_seconds",
        kind="service",
    ) == 1.5
    assert _metric_value(body, "hipengine_kv_pool_current_pages") == 3
    assert _metric_value(body, "hipengine_kv_pool_high_water_observed_pages") == 6
    assert _metric_value(body, "hipengine_kv_pool_pinned_pages") == 1
    assert _metric_value(body, "hipengine_graph_bucket_captures_total") == 2
    assert _metric_value(body, "hipengine_graph_bucket_replays_total") == 3
    assert _metric_value(body, "hipengine_graph_bucket_invalidations_total") == 1
    assert _labeled_metric_value(
        body,
        "hipengine_graph_bucket_replays_by_bucket_total",
        bucket="bucket-c2",
    ) == 3
    assert _labeled_metric_value(
        body,
        "hipengine_resident_route_total",
        route="native_packed_decode_steps",
    ) == 3
    # Which packed-prefill executor ran, per session. ``route`` alone cannot tell
    # the layer-outer executor from the chunk-outer fallback, so a proof keyed on
    # route can pass while the deliverable route never ran.
    assert _labeled_metric_value(
        body,
        "hipengine_resident_prefill_executor_mode_total",
        mode="layer_outer_packed",
    ) == 1
    assert _labeled_metric_value(
        body,
        "hipengine_resident_prefill_executor_mode_total",
        mode="none",
    ) == 1
    # The session's own KV layout, which gates whether the resumable prefill is
    # attempted at all. Distinct from the route manifest's kv_attention_source.
    assert _labeled_metric_value(
        body,
        "hipengine_resident_kv_attention_source_total",
        source="int8_direct",
    ) == 1
    assert _labeled_metric_value(
        body,
        "hipengine_resident_kv_attention_source_total",
        source="bf16_mirror",
    ) == 1
    assert _labeled_metric_value(
        body,
        "hipengine_resident_fallback_total",
        reason="host_sampling",
    ) == 1


def test_metrics_endpoint_counts_cancelled_requests() -> None:
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            metrics="prometheus",
        ),
        llm=BackendCancelledFakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "cancel", "max_tokens": 1},
    )
    metrics = client.get("/metrics")

    assert response.status_code == 499
    assert _metric_value(metrics.text, "hipengine_requests_total") == 1
    assert _metric_value(metrics.text, "hipengine_request_failed_total") == 1
    assert _metric_value(metrics.text, "hipengine_request_cancelled_total") == 1
    assert _metric_value(metrics.text, "hipengine_request_rejected_total") == 0


def test_metrics_endpoint_filters_malformed_graph_bucket_scalars() -> None:
    fake = FakeLLM()
    fake.graph_bucket_stats = SimpleNamespace(
        entries=True,
        hits=float("nan"),
        misses=float("inf"),
        replay_hit_rate=1.0,
        miss_reasons={},
        kernel_time_histogram_ns={},
    )
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False, metrics="prometheus"),
        llm=fake,
    )
    client = TestClient(app)

    malformed = client.get("/metrics")

    assert malformed.status_code == 200
    assert _metric_value(malformed.text, "hipengine_graph_bucket_entries") == 0
    assert _metric_value(malformed.text, "hipengine_graph_bucket_hits_total") == 0
    assert _metric_value(malformed.text, "hipengine_graph_bucket_misses_total") == 0
    assert _metric_value(malformed.text, "hipengine_graph_bucket_replay_hit_rate") == 0

    fake.graph_bucket_stats = SimpleNamespace(
        entries=-1,
        hits=3,
        misses=-4,
        replay_hit_rate=0.0,
        miss_reasons={},
        kernel_time_histogram_ns={},
    )
    partially_valid = client.get("/metrics")

    assert partially_valid.status_code == 200
    assert _metric_value(partially_valid.text, "hipengine_graph_bucket_entries") == 0
    assert _metric_value(partially_valid.text, "hipengine_graph_bucket_hits_total") == 3
    assert _metric_value(partially_valid.text, "hipengine_graph_bucket_misses_total") == 0
    assert _metric_value(partially_valid.text, "hipengine_graph_bucket_replay_hit_rate") == 1


def test_metrics_endpoint_filters_malformed_kv_pool_scalars() -> None:
    fake = FakeLLM()
    fake.kv_pool_stats = SimpleNamespace(
        current_bytes=True,
        high_water_observed_bytes=float("nan"),
        grow_events=float("inf"),
        grow_failures=-1,
        shrink_events="bad",
        free_pages=3,
        refcounted_pages=-4,
    )
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False, metrics="prometheus"),
        llm=fake,
    )
    client = TestClient(app)

    metrics = client.get("/metrics")

    assert metrics.status_code == 200
    assert _metric_value(metrics.text, "hipengine_kv_pool_current_bytes") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_high_water_observed_bytes") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_grow_events_total") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_grow_failures_total") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_shrink_events_total") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_free_pages") == 3
    assert _metric_value(metrics.text, "hipengine_kv_pool_refcounted_pages") == 0


def test_string_value_counts_counts_paths_and_keeps_unset_visible() -> None:
    """The executor-mode export counts which path ran, not how much was used."""

    from hipengine.server.api import _string_value_counts

    assert _string_value_counts(["layer_outer_packed", "layer_outer_packed"]) == {
        "layer_outer_packed": 2.0
    }
    # An unset executor_mode is a real observation (the plan never recorded one),
    # so it must be exported rather than dropped as if the session did not exist.
    assert _string_value_counts([None, "layer_outer_packed", None]) == {
        "none": 2.0,
        "layer_outer_packed": 1.0,
    }
    # Absent or non-list values emit nothing rather than inventing a zero.
    assert _string_value_counts(None) == {}
    assert _string_value_counts("layer_outer_packed") == {}
    assert _string_value_counts([]) == {}


def test_string_value_counts_escapes_label_values_that_would_break_the_format() -> None:
    """A mode string containing a quote or newline must not corrupt the export."""

    from hipengine.server.api import (
        _append_labeled_counter_metrics,
        _string_value_counts,
    )

    lines: list[str] = []
    _append_labeled_counter_metrics(
        lines,
        "hipengine_resident_prefill_executor_mode_total",
        "Packed-prefill executor that ran per resident session, by executor mode.",
        "mode",
        _string_value_counts(['we"ird\nmode']),
    )

    samples = [
        row
        for row in lines
        if row.startswith("hipengine_resident_prefill_executor_mode_total{")
    ]
    assert len(samples) == 1
    # The newline was escaped rather than emitted, so the sample stays one line
    # and Prometheus does not read the remainder as a malformed sample.
    assert "\\n" in samples[0]
    assert '\\"' in samples[0]
    assert "\n" not in samples[0]
