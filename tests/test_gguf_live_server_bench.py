from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.gguf_live_server_bench import (
    CONFIGURATIONS,
    _counter_delta,
    _latency_delta,
    _logical_shape_covers,
    _owned_physical_plans,
    _parse_configurations,
    _parse_sse_data_line,
    _prompt_rows,
    _scaling_summary,
    _stats,
    _wait_for_live_admission_trigger,
)


def test_live_server_bench_declares_honest_c13_routes() -> None:
    names = _parse_configurations(
        "c1,packed_c8,packed_c9,packed_c13,serial_c13"
    )

    assert names == (
        "c1",
        "packed_c8",
        "packed_c9",
        "packed_c13",
        "serial_c13",
    )
    assert CONFIGURATIONS["packed_c13"].logical_rows == 13
    assert CONFIGURATIONS["packed_c13"].execution_class == "grouped_exact_hybrid"
    assert CONFIGURATIONS["serial_c13"].packed_decode is False
    assert CONFIGURATIONS["serial_c13"].execution_class == "serial_bridge"

    with pytest.raises(ValueError, match="unknown"):
        _parse_configurations("c1,native_c13")
    with pytest.raises(ValueError, match="unique"):
        _parse_configurations("c1,c1")
    with pytest.raises(ValueError, match="canonical"):
        _parse_configurations(
            "packed_c13,c1,packed_c8,packed_c9,serial_c13"
        )


def test_live_server_bench_stats_and_counter_deltas_are_exact() -> None:
    stats = _stats([1.0, 2.0, 3.0, 4.0])

    assert stats["count"] == 4
    assert stats["median"] == pytest.approx(2.5)
    assert stats["p95"] == pytest.approx(4.0)
    assert _counter_delta({"packed": 2}, {"packed": 5, "serial": 1}) == {
        "packed": 3,
        "serial": 1,
    }


def test_live_server_bench_slices_append_only_scheduler_latency() -> None:
    before = {
        "time_to_first_token": {"samples": [0.1]},
        "inter_token": {"samples": [0.02, 0.03]},
    }
    after = {
        "time_to_first_token": {"samples": [0.2, 0.1, 0.3]},
        "inter_token": {"samples": [0.02, 0.04, 0.03]},
    }

    assert _latency_delta(before, after) == {
        "time_to_first_token": [0.2, 0.3],
        "inter_token": [0.04],
    }

    with pytest.raises(RuntimeError, match="history changed"):
        _latency_delta(before, {**after, "inter_token": {"samples": [9.0]}})


def test_live_server_bench_scopes_plans_to_sample_request_ownership() -> None:
    timeline = [
        {
            "physical_group_plans": [
                {
                    "logical_c": 1,
                    "groups": [
                        {
                            "request_ids": [8],
                            "physical_rows": 8,
                            "active_mask": [True] + [False] * 7,
                            "execution_path": "serial_fallback",
                        }
                    ],
                },
                {
                    "logical_c": 9,
                    "groups": [
                        {
                            "request_ids": list(range(9, 17)),
                            "physical_rows": 8,
                            "active_mask": [True] * 8,
                            "execution_path": "packed_native",
                        },
                        {
                            "request_ids": [17],
                            "physical_rows": 8,
                            "active_mask": [True] + [False] * 7,
                            "execution_path": "packed_native",
                        },
                    ],
                },
            ]
        }
    ]

    plans, foreign = _owned_physical_plans(timeline, list(range(9, 18)))

    assert len(plans) == 1
    assert plans[0]["logical_c"] == 9
    assert foreign == 1


def test_live_server_bench_accepts_truthful_sparse_c9_without_compaction() -> None:
    assert _logical_shape_covers(
        (9, (8, 8), ("11111111", "10000000")),
        logical_c=9,
        group_count=2,
    )
    assert _logical_shape_covers(
        (9, (8, 1), ("11111111", "1")),
        logical_c=9,
        group_count=2,
    )
    assert not _logical_shape_covers(
        (9, (8,), ("11111111",)),
        logical_c=9,
        group_count=2,
    )
    assert not _logical_shape_covers(
        (9, (8, 8), ("11111111", "11000000")),
        logical_c=9,
        group_count=2,
    )


def test_live_server_bench_parses_openai_sse_data() -> None:
    assert _parse_sse_data_line("") is None
    assert _parse_sse_data_line("event: message") is None
    assert _parse_sse_data_line("data: [DONE]") == "[DONE]"
    assert _parse_sse_data_line('data: {"choices": []}') == {"choices": []}


def test_live_server_bench_requires_exact_text_token_roundtrip() -> None:
    class FakeTokenizer:
        def decode(self, token_ids):
            return ",".join(str(token) for token in token_ids)

        def encode(self, text):
            return [int(token) for token in text.split(",")]

    rows = _prompt_rows(
        FakeTokenizer(),
        rows=5,
        prompt_length=3,
        prompt_token_id=100,
    )

    assert [row["token_id"] for row in rows] == [100, 101, 102, 103, 100]
    assert all(row["token_count"] == 3 for row in rows)
    assert all(row["roundtrip_exact"] is True for row in rows)


def test_live_server_bench_scaling_compares_grouped_c13_with_c1_and_serial() -> None:
    def summary(rate: float) -> dict[str, object]:
        return {
            "rates": {
                "aggregate_generated_tok_s": {"median": rate},
            }
        }

    scaling = _scaling_summary(
        {
            "c1": summary(80.0),
            "packed_c8": summary(200.0),
            "packed_c9": summary(205.0),
            "packed_c13": summary(230.0),
            "serial_c13": summary(75.0),
        }
    )

    assert scaling["grouped_c13_scaling_gate_passed"] is True
    assert scaling["ratios"]["packed_c13_vs_c1"] == pytest.approx(230.0 / 80.0)
    assert scaling["ratios"]["packed_c13_vs_serial_c13"] == pytest.approx(
        230.0 / 75.0
    )
    assert "multiple declared physical buckets" in scaling["policy"]


def test_live_server_bench_live_trigger_requires_declared_initial_group() -> None:
    timeline = [
        {
            "physical_group_plans": [
                {
                    "logical_c": 8,
                    "groups": [],
                }
            ]
        }
    ]
    llm = SimpleNamespace(
        live_loop_snapshot=lambda: {
            "loop": {"requests": {"active": 8}},
        }
    )

    trigger = _wait_for_live_admission_trigger(
        llm,
        timeline,
        timeline_start=0,
        initial_rows=8,
        timeout_seconds=0.01,
    )

    assert trigger["timeline_event"] == timeline[0]
    assert trigger["snapshot"]["loop"]["requests"]["active"] == 8
