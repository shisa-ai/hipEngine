from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.gguf_mtp_c1c8_server_bench import (
    _backend_mtp_engaged,
    _cell_correctness,
    _diagnostic_plan,
    _generated_ids,
    _memory_delta,
    _mtp_budget_conformed,
    _mtp_engaged,
    _parse_expected_mtp_widths,
    _parse_widths,
    _render_messages,
    _resident_observability,
    build_parser,
    summarize,
)


def test_mtp_c1c8_parser_defaults_to_production_profile() -> None:
    parser = build_parser()
    default = parser.parse_args(("--output", "/tmp/out.json"))
    strict = parser.parse_args(
        ("--execution-profile", "strict", "--output", "/tmp/out.json")
    )

    assert default.execution_profile == "production"
    assert strict.execution_profile == "strict"


def test_mtp_c1c8_parses_complete_widths() -> None:
    assert _parse_widths("1,2,3,4,5,6,7,8") == tuple(range(1, 9))
    assert _parse_expected_mtp_widths("none") == ()
    assert _parse_expected_mtp_widths("1,4") == (1, 4)
    with pytest.raises(Exception, match="unique subset"):
        _parse_widths("1,1")
    with pytest.raises(Exception, match="unique subset"):
        _parse_widths("9")


def test_mtp_c1c8_renders_canonical_messages() -> None:
    assert _render_messages([{"role": "user", "content": "hello"}]) == (
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    )


def test_mtp_c1c8_extracts_authoritative_ids_and_engagement() -> None:
    payload = {
        "choices": [{"hipengine": {"generated_token_ids": [9]}}],
        "hipengine": {
            "token_accounting": {"choice_generated_token_ids": [[1, 2, 3]]}
        },
    }
    assert _generated_ids(payload) == [1, 2, 3]
    assert _mtp_engaged(
        "speculative_mtp",
        {"used": True, "draft_tokens": 4, "draft_cycles": 2},
    )
    assert not _mtp_engaged(
        "speculative_mtp",
        {"used": False, "draft_tokens": 0, "draft_cycles": 0},
    )
    assert _mtp_budget_conformed(
        {"draft_tokens": 8, "draft_cycles": 4}, budget=2
    )
    assert not _mtp_budget_conformed(
        {"draft_tokens": 9, "draft_cycles": 4}, budget=2
    )
    assert not _mtp_budget_conformed(
        {"draft_tokens": 0, "draft_cycles": 0}, budget=2
    )


def test_mtp_c1c8_compacts_nested_resident_observability() -> None:
    runner = SimpleNamespace(
        observability_snapshot=lambda: {
            "resources": {"active_requests": 0},
            "routes": {"recent_completed": [{"request_id": 1}, {"request_id": 2}]},
        }
    )
    llm = SimpleNamespace(
        _get_text_generator=lambda: SimpleNamespace(
            _driver=SimpleNamespace(_runner=runner)
        )
    )

    snapshot = _resident_observability(llm, recent=1)

    assert snapshot["resources"]["active_requests"] == 0
    assert snapshot["routes"]["recent_completed"] == [{"request_id": 2}]


def test_mtp_c1c8_reports_tracked_memory_delta() -> None:
    assert _memory_delta(
        {"total_allocated_bytes": 10, "active_allocations": 2},
        {"total_allocated_bytes": 25, "active_allocations": 1},
    )["total_allocated_bytes"] == 15
    assert _memory_delta(
        {"total_allocated_bytes": 10, "active_allocations": 2},
        {"total_allocated_bytes": 25, "active_allocations": 1},
    )["active_allocations"] == -1


def test_mtp_c1c8_correctness_contract_separates_exact_and_traded_routes() -> None:
    ar = [[1, 2], [1, 2]]
    mtp = [[1, 3], [1, 3]]

    strict = _cell_correctness(ar, mtp, contract="ar_exact")
    traded = _cell_correctness(ar, mtp, contract="mtp_self_exact")

    assert strict == {"ar_self_exact": True, "mtp_self_exact": True, "ar_mtp_equal": False, "passed": False}
    assert traded == {"ar_self_exact": True, "mtp_self_exact": True, "ar_mtp_equal": False, "passed": True}


def test_mtp_c1c8_backend_telemetry_proves_legacy_engagement() -> None:
    assert _backend_mtp_engaged(
        {
            "path": "gguf_llama_compat_mtp_server",
            "batch_size": 4,
            "speculative_mtp": {
                "total_draft_tokens": 21,
                "direct_cycles": 8,
            },
        },
        width=4,
    )
    assert not _backend_mtp_engaged(
        {
            "path": "gguf_packed_ar_server_decode",
            "batch_size": 4,
            "speculative_mtp": {"total_draft_tokens": 0, "direct_cycles": 0},
        },
        width=4,
    )


def test_mtp_c1c8_diagnostic_plan_is_content_agnostic_and_bounded() -> None:
    base = {
        "realized_group_rows": 4,
        "candidate_budget": 2,
        "sampling_mode": "greedy_fast",
        "context_tokens": 64,
        "output_horizon_tokens": 24,
        "memory_fit": True,
    }
    first = _diagnostic_plan(**base)
    second = _diagnostic_plan(**base)

    assert first == second
    assert first["admitted"] is True
    assert first["selected_candidate_count"] == 2
    assert _diagnostic_plan(**{**base, "candidate_budget": 1})["admitted"] is True
    assert _diagnostic_plan(**{**base, "candidate_budget": 3})["admitted"] is True
    assert _diagnostic_plan(**{**base, "candidate_budget": 4})["admitted"] is False
    assert _diagnostic_plan(**{**base, "realized_group_rows": 5})["admitted"] is False
    assert _diagnostic_plan(**{**base, "context_tokens": 96})["admitted"] is False


def test_mtp_c1c8_summary_uses_complete_wall() -> None:
    cells = [
        {
            "width": 2,
            "exact": True,
            "mtp_engaged": True,
            "ar": {"wall_seconds": 2.0, "generated_tokens": 48},
            "mtp": {"wall_seconds": 1.5, "generated_tokens": 48},
        },
        {
            "width": 2,
            "exact": True,
            "mtp_engaged": True,
            "ar": {"wall_seconds": 2.0, "generated_tokens": 48},
            "mtp": {"wall_seconds": 1.5, "generated_tokens": 48},
        },
    ]

    row = summarize(cells, widths=(2,))["2"]
    assert row["ar"]["tok_s"] == pytest.approx(24.0)
    assert row["mtp"]["tok_s"] == pytest.approx(32.0)
    assert row["mtp_vs_ar_percent"] == pytest.approx(100.0 / 3.0)
    assert row["exact_cells"] == row["engaged_cells"] == row["cells"] == 2
    assert row["budget_conformed_cells"] == 2
    assert row["mtp_expected"] is True
    assert row["route_expectation_passed"] is True

    k0 = summarize(cells, widths=(2,), expected_mtp_widths=())["2"]
    assert k0["mtp_expected"] is False
    assert k0["route_expectation_passed"] is False


def test_mtp_c1c8_summary_accepts_expected_k0() -> None:
    cells = [
        {
            "width": 5,
            "exact": True,
            "mtp_engaged": False,
            "ar": {"wall_seconds": 2.0, "generated_tokens": 24},
            "mtp": {"wall_seconds": 2.0, "generated_tokens": 24},
        }
    ]
    row = summarize(cells, widths=(5,), expected_mtp_widths=())["5"]
    assert row["exact_cells"] == row["cells"] == 1
    assert row["engaged_cells"] == 0
    assert row["route_expectation_passed"] is True
