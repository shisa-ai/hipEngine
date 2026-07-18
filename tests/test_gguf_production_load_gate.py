from __future__ import annotations

import pytest
from fastapi import FastAPI

from scripts.gguf_production_load_gate import (
    RequestResult,
    SLOThresholds,
    TuningCandidate,
    _build_workload_specs,
    _distribution,
    _evaluate_workload,
    _http_json,
    _LocalUvicorn,
    _openai_error_fields,
    _parse_workload_names,
    _poisson_arrival_offsets,
    _select_tuning_candidate,
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


def test_poisson_offsets_are_seeded_monotonic_and_start_at_zero() -> None:
    first = _poisson_arrival_offsets(count=8, rate_per_second=4.0, seed=1234)
    second = _poisson_arrival_offsets(count=8, rate_per_second=4.0, seed=1234)

    assert first == second
    assert first[0] == 0.0
    assert len(first) == 8
    assert all(left < right for left, right in zip(first, first[1:]))
    with pytest.raises(ValueError, match="positive"):
        _poisson_arrival_offsets(count=2, rate_per_second=0.0, seed=1)


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
    assert len(workloads["overload"]) > 16
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
) -> RequestResult:
    return RequestResult(
        label=label,
        action="complete",
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
