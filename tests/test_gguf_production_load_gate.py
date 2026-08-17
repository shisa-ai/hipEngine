from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from hipengine.generation import EngineLoopConfig

from scripts.gguf_production_load_gate import (
    RequestResult,
    SLOThresholds,
    TuningCandidate,
    _PROVENANCE_ENV_KEYS,
    _aggregate_tuning_runs,
    _build_workload_specs,
    build_parser,
    _distribution,
    _evaluate_workload,
    _force_disconnect,
    _http_json,
    _LocalUvicorn,
    _openai_error_fields,
    _load_tuning_protocol,
    _occupancy_summary,
    _parse_workload_names,
    _poisson_arrival_offsets,
    _reconfigure_loaded_loop,
    _rotated_tuning_plan,
    _select_tuning_candidate,
    _wait_for_idle,
)


def test_local_uvicorn_uses_a_real_socket_and_stops_cleanly() -> None:
    app = FastAPI()

    @app.get("/ready")
    async def ready() -> dict[str, bool]:
        return {"ready": True}

    with _LocalUvicorn(app) as server:
        assert _http_json("127.0.0.1", server.port, "GET", "/ready") == {
            "ready": True,
        }

    assert not server.thread.is_alive()


def test_distribution_reports_nearest_rank_p50_p95_p99() -> None:
    summary = _distribution([float(value) for value in range(1, 101)])

    assert summary["count"] == 100
    assert summary["p50"] == pytest.approx(50.0)
    assert summary["p95"] == pytest.approx(95.0)
    assert summary["p99"] == pytest.approx(99.0)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(100.0)


def test_tuning_reconfiguration_uses_loaded_driver_control() -> None:
    observed: list[EngineLoopConfig] = []
    driver = SimpleNamespace(
        reconfigure_engine_loop=lambda config: observed.append(config)
    )
    llm = SimpleNamespace(_get_text_generator=lambda: driver)
    adapter = SimpleNamespace(
        _loop=SimpleNamespace(
            config=EngineLoopConfig(max_active_requests=8),
        )
    )

    _reconfigure_loaded_loop(
        llm,
        adapter,
        policy="token_budget",
        prefill_chunk_tokens=256,
        fair_prefill_burst_chunks=2,
    )

    assert len(observed) == 1
    assert observed[0].prefill_decode_policy == "token_budget"
    assert observed[0].max_prefill_chunk_tokens == 256
    assert observed[0].fair_prefill_burst_chunks == 2


def test_wait_for_idle_tolerates_long_transition_control_timeout() -> None:
    class FakeLLM:
        calls = 0

        def live_loop_snapshot(self):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("control timeout")
            return {
                "loop": {"requests": {"active": 0, "pending": 0}},
                "runner": {"model_runner": {"active_requests": 0}},
            }

    class FakeBatcher:
        @staticmethod
        def queue_depth() -> int:
            return 0

        @staticmethod
        def active_requests() -> int:
            return 0

        @staticmethod
        def active() -> bool:
            return False

    observed = _wait_for_idle(FakeLLM(), FakeBatcher(), timeout_seconds=1.0)

    assert observed["control_timeouts"] == 1
    assert observed["generation_queue_depth"] == 0


def test_occupancy_route_uses_live_snapshot_plan_when_hook_timeline_is_empty() -> None:
    plan = {
        "logical_c": 3,
        "groups": [
            {"physical_rows": 2, "active_mask": [True, True], "execution_path": "packed_native"},
            {"physical_rows": 1, "active_mask": [True], "execution_path": "native_c1_eager"},
        ],
    }

    summary = _occupancy_summary(
        [
            {
                "active": 3,
                "pending": 0,
                "occupancy_ratio": 1.0,
                "generation_queue_depth": 0,
                "stream_queue_max_depth": 1,
                "physical_group_plan": plan,
            }
        ],
        [],
        stream_queue_limit=4,
    )

    assert summary["route_passed"] is True
    assert summary["execution_paths"] == ["native_c1_eager", "packed_native"]
    assert summary["logical_physical_shapes"] == [
        {"logical_c": 3, "physical_widths": [2, 1], "active_masks": ["11", "1"]}
    ]


def test_poisson_offsets_are_seeded_monotonic_and_start_at_zero() -> None:
    first = _poisson_arrival_offsets(count=8, rate_per_second=4.0, seed=1234)
    second = _poisson_arrival_offsets(count=8, rate_per_second=4.0, seed=1234)

    assert first == second
    assert first[0] == 0.0
    assert len(first) == 8
    assert all(left < right for left, right in zip(first, first[1:]))
    with pytest.raises(ValueError, match="positive"):
        _poisson_arrival_offsets(count=2, rate_per_second=0.0, seed=1)


def test_pressure_gate_prefix_cache_cli_defaults_off_and_records_radix() -> None:
    parser = build_parser()

    defaults = parser.parse_args([])
    assert defaults.prefix_cache == "off"
    assert defaults.initial_policy == "token_budget"
    assert defaults.tuning_candidates.startswith("token_budget:128,token_budget:256")
    assert parser.parse_args(["--prefix-cache", "radix"]).prefix_cache == "radix"
    assert "HIPENGINE_PREFIX_CACHE" in _PROVENANCE_ENV_KEYS


def test_force_disconnect_shutdowns_socket_before_closing_http_wrappers() -> None:
    calls: list[object] = []

    class FakeSocket:
        def setsockopt(self, level, option, value) -> None:
            calls.append(("setsockopt", level, option, value))

        def shutdown(self, how) -> None:
            calls.append(("shutdown", how))

    class FakeResponse:
        def close(self) -> None:
            calls.append("response.close")

    class FakeConnection:
        sock = FakeSocket()

        def close(self) -> None:
            calls.append("connection.close")

    _force_disconnect(FakeResponse(), FakeConnection())

    assert calls[0][0] == "setsockopt"
    assert calls[1] == ("shutdown", 2)
    assert calls[2:] == ["response.close", "connection.close"]


def test_workload_selector_is_ordered_unique_and_fail_closed() -> None:
    available = ("static_c1", "cancellation_disconnect", "soak")

    assert _parse_workload_names(
        "cancellation_disconnect",
        available=available,
    ) == ("cancellation_disconnect",)
    with pytest.raises(ValueError, match="unknown"):
        _parse_workload_names("overload", available=available)
    with pytest.raises(ValueError, match="unique"):
        _parse_workload_names("soak,soak", available=available)


def test_workload_plan_covers_required_production_modes() -> None:
    workloads = _build_workload_specs(
        fixed_rate_per_second=4.0,
        poisson_rate_per_second=4.0,
        poisson_seed=1234,
        soak_seconds=2.0,
        soak_rate_per_second=4.0,
    )

    assert set(workloads) == {
        "static_c1",
        "static_c8",
        "ragged_burst",
        "continuous_fixed",
        "continuous_poisson",
        "cancellation_disconnect",
        "overload",
        "idle_recovery",
        "soak",
    }
    ragged = workloads["ragged_burst"]
    assert len({item.prompt_length for item in ragged}) >= 4
    assert len({item.max_tokens for item in ragged}) >= 4
    assert {item.action for item in workloads["cancellation_disconnect"]} == {
        "complete",
        "disconnect",
        "timeout",
    }
    disconnect = next(
        item
        for item in workloads["cancellation_disconnect"]
        if item.action == "disconnect"
    )
    assert disconnect.disconnect_after_tokens == 1
    assert len(workloads["overload"]) == 40
    assert workloads["continuous_fixed"][0].arrival_offset_seconds == 0.0
    assert workloads["continuous_fixed"][-1].arrival_offset_seconds > 0.0
    assert workloads["continuous_poisson"][0].arrival_offset_seconds == 0.0
    assert len(workloads["soak"]) >= 8


def _result(
    label: str,
    *,
    generated: int = 8,
    exact: bool = True,
    outcome: str = "completed",
    queue: float = 0.01,
    ttft: float = 0.2,
    itl: tuple[float, ...] = (0.02, 0.03),
    e2e: float = 0.5,
    error_code: str | None = None,
    http_protocol_exact: bool = True,
    action: str = "complete",
    cancellation_latency_seconds: float | None = None,
) -> RequestResult:
    return RequestResult(
        label=label,
        action=action,
        outcome=outcome,
        status_code=200 if outcome != "rejected" else 429,
        error_code=error_code,
        request_id=None if outcome == "rejected" else 10,
        generated_count=generated,
        exact=exact,
        queue_seconds=queue,
        ttft_seconds=ttft,
        inter_token_seconds=itl,
        end_to_end_seconds=e2e,
        finish_reason="length" if outcome == "completed" else outcome,
        http_protocol_exact=http_protocol_exact,
        cancellation_latency_seconds=cancellation_latency_seconds,
    )


def test_workload_evaluation_derives_exact_generated_token_goodput_and_slos() -> None:
    slos = SLOThresholds(
        queue_p99_seconds=1.0,
        ttft_p95_seconds=1.0,
        itl_p99_seconds=0.1,
        end_to_end_p95_seconds=2.0,
    )
    rows = [
        _result("good-a", generated=8),
        _result("good-b", generated=8),
        _result("slow", generated=8, ttft=2.0),
        _result("wrong", generated=8, exact=False),
    ]

    summary = _evaluate_workload(
        "continuous_fixed",
        rows,
        wall_seconds=2.0,
        slos=slos,
    )

    assert summary["accounting"]["exact_generated_tokens"] == 24
    assert summary["goodput"]["qualifying_generated_tokens"] == 16
    assert summary["goodput"]["generated_tokens_per_second"] == pytest.approx(8.0)
    assert summary["latency_seconds"]["ttft"]["p95"] == pytest.approx(2.0)
    assert summary["slo"]["passed"] is False
    assert summary["correctness"]["passed"] is False
    assert "generated_token_mismatch" in summary["failure_reasons"]


def test_workload_evaluation_reports_disconnect_cancellation_ack_latency() -> None:
    summary = _evaluate_workload(
        "cancellation_disconnect",
        [
            _result("survivor"),
            _result(
                "disconnected",
                generated=1,
                outcome="disconnected",
                action="disconnect",
                cancellation_latency_seconds=0.125,
            ),
        ],
        wall_seconds=1.0,
        slos=SLOThresholds(5.0, 5.0, 1.0, 20.0),
    )

    assert summary["passed"] is True
    assert summary["latency_seconds"]["cancellation_ack"] == {
        "count": 1,
        "p50": pytest.approx(0.125),
        "p95": pytest.approx(0.125),
        "p99": pytest.approx(0.125),
        "min": pytest.approx(0.125),
        "max": pytest.approx(0.125),
        "mean": pytest.approx(0.125),
        "stdev": pytest.approx(0.0),
    }


def test_openai_error_fields_reads_canonical_nested_status() -> None:
    assert _openai_error_fields(
        {
            "message": "request deadline exceeded",
            "code": "deadline_exceeded",
            "hipengine": {
                "code": "deadline_exceeded",
                "status_code": 408,
                "retryable": True,
            },
        }
    ) == ("deadline_exceeded", 408, "request deadline exceeded")


def test_http_protocol_failure_rejects_an_otherwise_exact_workload() -> None:
    summary = _evaluate_workload(
        "static_c1",
        [_result("missing-done-or-usage", http_protocol_exact=False)],
        wall_seconds=1.0,
        slos=SLOThresholds(5.0, 5.0, 1.0, 20.0),
    )

    assert summary["correctness"]["passed"] is False
    assert summary["passed"] is False
    assert "http_sse_protocol_failed" in summary["failure_reasons"]


def test_overload_requires_both_exact_accepts_and_engine_busy_rejects() -> None:
    slos = SLOThresholds(5.0, 5.0, 1.0, 20.0)
    rows = [
        _result("accepted", generated=8),
        _result(
            "rejected",
            generated=0,
            exact=True,
            outcome="rejected",
            error_code="engine_busy",
        ),
    ]

    summary = _evaluate_workload(
        "overload",
        rows,
        wall_seconds=1.0,
        slos=slos,
        require_rejects=True,
    )

    assert summary["outcomes"] == {"completed": 1, "rejected": 1}
    assert summary["overload"]["passed"] is True
    assert summary["passed"] is True

    no_reject = _evaluate_workload(
        "overload",
        rows[:1],
        wall_seconds=1.0,
        slos=slos,
        require_rejects=True,
    )
    assert no_reject["overload"]["passed"] is False
    assert no_reject["passed"] is False


def test_tuning_selection_maximizes_goodput_only_among_slo_passing_rows() -> None:
    selected = _select_tuning_candidate(
        [
            TuningCandidate("protect_ttft", 256, 50.0, 0.8, 0.08, False),
            TuningCandidate("fair", 256, 44.0, 0.5, 0.06, True),
            TuningCandidate("fair", 128, 47.0, 0.6, 0.07, True),
        ]
    )

    assert selected.prefill_decode_policy == "fair"
    assert selected.prefill_chunk_tokens == 128
    assert selected.goodput_generated_tokens_per_second == pytest.approx(47.0)

    with pytest.raises(ValueError, match="no SLO-passing"):
        _select_tuning_candidate(
            [TuningCandidate("protect_decode", 256, 80.0, 9.0, 0.1, False)]
        )


def test_a4_protocol_loader_and_rotation_preserve_predeclared_candidates() -> None:
    configurations, metadata = _load_tuning_protocol(
        "benchmarks/results/2026-07-22-w7900-agentic-a4-predeclared-protocol.json"
    )

    assert metadata["kind"] == "gfx1100_agentic_a4_predeclared_protocol"
    assert len(configurations) == 8
    assert configurations[0].candidate_id == "pd256_b1_w0_control"
    assert configurations[-1].candidate_id == "fair256_b2_w100_diagnostic"
    assert configurations[-1].batch_window_ms == 100.0
    plans = _rotated_tuning_plan(configurations, repetitions=3)
    assert [plan[0].candidate_id for plan in plans] == [
        "pd256_b1_w0_control",
        "fair256_b1_w0",
        "fair256_b2_w5",
    ]
    assert all(set(plan) == set(configurations) for plan in plans)


def test_a4_tuning_aggregation_uses_complete_medians_and_all_pass_gate() -> None:
    configurations, _metadata = _load_tuning_protocol(
        "benchmarks/results/2026-07-22-w7900-agentic-a4-predeclared-protocol.json"
    )
    configuration = configurations[0]
    runs = []
    for repetition, (goodput, ttft, itl, e2e, passed) in enumerate(
        (
            (40.0, 0.7, 0.20, 4.0, True),
            (44.0, 0.5, 0.18, 3.5, True),
            (42.0, 0.6, 0.19, 3.8, False),
        )
    ):
        runs.append(
            {
                "repetition": repetition,
                "configuration": configuration,
                "candidate": TuningCandidate(
                    configuration.prefill_decode_policy,
                    configuration.prefill_chunk_tokens,
                    goodput,
                    ttft,
                    itl,
                    passed,
                    candidate_id=configuration.candidate_id,
                    fair_prefill_burst_chunks=configuration.fair_prefill_burst_chunks,
                    batch_window_ms=configuration.batch_window_ms,
                    end_to_end_p95_seconds=e2e,
                ),
            }
        )

    aggregates, candidates = _aggregate_tuning_runs(
        runs,
        configurations=(configuration,),
        expected_repetitions=3,
    )

    assert len(aggregates) == len(candidates) == 1
    assert candidates[0].goodput_generated_tokens_per_second == pytest.approx(42.0)
    assert candidates[0].ttft_p95_seconds == pytest.approx(0.6)
    assert candidates[0].itl_p99_seconds == pytest.approx(0.19)
    assert candidates[0].end_to_end_p95_seconds == pytest.approx(3.8)
    assert candidates[0].passed is False
    assert aggregates[0]["complete"] is True
    assert aggregates[0]["all_repetitions_passed"] is False


def test_a4_tuning_aggregation_rejects_incomplete_candidate() -> None:
    configurations, _metadata = _load_tuning_protocol(
        "benchmarks/results/2026-07-22-w7900-agentic-a4-predeclared-protocol.json"
    )
    configuration = configurations[0]
    with pytest.raises(ValueError, match="incomplete"):
        _aggregate_tuning_runs(
            [
                {
                    "repetition": 0,
                    "configuration": configuration,
                    "candidate": TuningCandidate(
                        configuration.prefill_decode_policy,
                        configuration.prefill_chunk_tokens,
                        40.0,
                        0.5,
                        0.2,
                        True,
                        candidate_id=configuration.candidate_id,
                    ),
                }
            ],
            configurations=(configuration,),
            expected_repetitions=3,
        )
