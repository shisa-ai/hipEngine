from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.gguf_mtp_c1c8_server_bench import (
    _backend_mtp_engaged,
    _cell_correctness,
    _diagnostic_plan,
    _generated_ids,
    _install_diagnostic_plan,
    _memory_delta,
    _mtp_budget_conformed,
    _mtp_engaged,
    _parse_expected_mtp_widths,
    _parse_widths,
    _render_messages,
    _resident_observability,
    _select_diagnostic_prompts,
    build_parser,
    summarize,
    summarize_acceptance,
)


def test_mtp_c1c8_parser_defaults_to_production_profile() -> None:
    parser = build_parser()
    default = parser.parse_args(("--output", "/tmp/out.json"))
    strict = parser.parse_args(
        ("--execution-profile", "strict", "--output", "/tmp/out.json")
    )
    normal_owner = parser.parse_args(
        (
            "--widths",
            "2",
            "--resident-capacity",
            "4",
            "--output",
            "/tmp/out.json",
        )
    )

    assert default.execution_profile == "production"
    assert default.resident_capacity is None
    assert strict.execution_profile == "strict"
    assert normal_owner.widths == (2,)
    assert normal_owner.resident_capacity == 4
    assert default.diagnostic_prompt_count == 0


def test_mtp_c1c8_diagnostic_prompt_subset_is_fail_closed() -> None:
    prompts = tuple({"id": str(index)} for index in range(10))
    assert _select_diagnostic_prompts(
        prompts, count=1, generation2_diagnostic=True
    ) == ({"id": "0"},)
    assert _select_diagnostic_prompts(
        prompts, count=0, generation2_diagnostic=False
    ) == prompts
    with pytest.raises(ValueError, match="generation2-diagnostic"):
        _select_diagnostic_prompts(prompts, count=1, generation2_diagnostic=False)
    with pytest.raises(ValueError, match="between 1 and 10"):
        _select_diagnostic_prompts(prompts, count=11, generation2_diagnostic=True)


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
    assert _mtp_engaged(
        "default",
        {
            "used": True,
            "draft_tokens": 4,
            "draft_cycles": 2,
            "requested_route": "speculative_mtp",
            "effective_route": "default",
        },
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
    adapter = SimpleNamespace(
        cycle_workspace_contract=lambda: {"allocated": True, "shape": [4, 5120]},
        _states={},
        _provider_groups={},
        _prompt_streaming_sinks={},
        _batch_accept_workspace=None,
    )
    runner = SimpleNamespace(
        _mtp2_adapter=adapter,
        observability_snapshot=lambda: {
            "resources": {"active_requests": 0},
            "routes": {"recent_completed": [{"request_id": 1}, {"request_id": 2}]},
        },
    )
    llm = SimpleNamespace(
        _get_text_generator=lambda: SimpleNamespace(
            _driver=SimpleNamespace(_runner=runner)
        )
    )

    snapshot = _resident_observability(llm, recent=1)

    assert snapshot["resources"]["active_requests"] == 0
    assert snapshot["routes"]["recent_completed"] == [{"request_id": 2}]
    assert snapshot["mtp2_adapter"] == {
        "cycle_workspace": {"allocated": True, "shape": [4, 5120]},
        "active_states": 0,
        "provider_groups": 0,
        "prompt_streaming_sinks": 0,
        "batch_accept_workspace_allocated": False,
    }


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
    frontend_c1 = _diagnostic_plan(
        **{
            **base,
            "realized_group_rows": 1,
            "candidate_budget": 1,
        }
    )
    assert frontend_c1["key"]["realized_group_rows"] == 1
    assert frontend_c1["static_eligibility"]["eligible"] is True
    assert frontend_c1["static_eligibility"]["max_candidate_count"] == 1
    assert frontend_c1["static_eligibility"]["max_realized_group_rows"] == 8
    assert frontend_c1["static_eligibility"]["automatic_eligible"] is False
    owner = SimpleNamespace(speculative_candidate_budget=1)
    _install_diagnostic_plan(owner)
    installed = owner.resolve_speculative_mtp_serving_plan(
        realized_group_rows=1,
        sampling_mode="greedy_fast",
        context_tokens=64,
        output_horizon_tokens=24,
        memory_fit=True,
    )
    assert installed["selected_candidate_count"] == 1
    assert installed["static_eligibility"]["max_candidate_count"] == 1
    assert _diagnostic_plan(**{**base, "candidate_budget": 1})["admitted"] is True
    assert _diagnostic_plan(**{**base, "candidate_budget": 3})["admitted"] is True
    assert _diagnostic_plan(**{**base, "candidate_budget": 4})["admitted"] is False
    # M1 protocol: single-group wide verify admits rows5-8 through C8/R32.
    assert _diagnostic_plan(**{**base, "realized_group_rows": 5})["admitted"] is True
    assert _diagnostic_plan(**{**base, "realized_group_rows": 8})["admitted"] is True
    assert _diagnostic_plan(**{**base, "realized_group_rows": 9})["admitted"] is False
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


def test_mtp_c1c8_summarizes_acceptance_by_scope_and_position() -> None:
    cells = [
        {
            "category": "code",
            "heldout": False,
            "width": 1,
            "mtp": {
                "resident_observability": {
                    "routes": {
                        "recent_completed": [
                            {
                                "specdec2_mtp2_candidate_counts": [3, 3, 2],
                                "specdec2_mtp2_accepted_counts": [3, 1, 0],
                            }
                        ]
                    }
                }
            },
        },
        {
            "category": "general_en",
            "heldout": True,
            "width": 1,
            "mtp": {
                "resident_observability": {
                    "routes": {
                        "recent_completed": [
                            {
                                "specdec2_mtp2_candidate_counts": [3],
                                "specdec2_mtp2_accepted_counts": [2],
                            }
                        ]
                    }
                }
            },
        },
    ]

    summary = summarize_acceptance(cells)

    assert summary["denominators"] == {
        "draft_acceptance": "accepted draft tokens / proposed draft tokens",
        "position_acceptance": "cycles accepting through this position / cycles proposing this position",
        "conditional_position_acceptance": "cycles accepting through this position / cycles accepting the preceding positions while proposing this position",
    }
    all_rows = summary["scopes"]["all"]
    assert all_rows["cycles"] == 4
    assert all_rows["proposed_draft_tokens"] == 11
    assert all_rows["accepted_draft_tokens"] == 6
    assert all_rows["draft_acceptance"] == pytest.approx(6 / 11)
    assert all_rows["positions"] == [
        {
            "position": 1,
            "proposed_cycles": 4,
            "accepted_cycles": 3,
            "position_acceptance": pytest.approx(3 / 4),
            "conditional_opportunities": 4,
            "conditional_position_acceptance": pytest.approx(3 / 4),
        },
        {
            "position": 2,
            "proposed_cycles": 4,
            "accepted_cycles": 2,
            "position_acceptance": pytest.approx(2 / 4),
            "conditional_opportunities": 3,
            "conditional_position_acceptance": pytest.approx(2 / 3),
        },
        {
            "position": 3,
            "proposed_cycles": 3,
            "accepted_cycles": 1,
            "position_acceptance": pytest.approx(1 / 3),
            "conditional_opportunities": 2,
            "conditional_position_acceptance": pytest.approx(1 / 2),
        },
    ]
    assert summary["scopes"]["train"]["draft_acceptance"] == pytest.approx(4 / 8)
    assert summary["scopes"]["heldout"]["draft_acceptance"] == pytest.approx(2 / 3)
    assert summary["categories"]["code"]["draft_acceptance"] == pytest.approx(4 / 8)
    assert summary["categories"]["general_en"]["draft_acceptance"] == pytest.approx(2 / 3)


def test_mtp_c1c8_acceptance_summary_rejects_malformed_cycle_counts() -> None:
    cells = [
        {
            "category": "code",
            "heldout": False,
            "width": 1,
            "mtp": {
                "resident_observability": {
                    "routes": {
                        "recent_completed": [
                            {
                                "specdec2_mtp2_candidate_counts": [3],
                                "specdec2_mtp2_accepted_counts": [4],
                            }
                        ]
                    }
                }
            },
        }
    ]

    with pytest.raises(ValueError, match="accepted count 4 exceeds candidate count 3"):
        summarize_acceptance(cells)


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
