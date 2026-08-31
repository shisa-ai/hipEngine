from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.gguf_mtp_c1c8_server_bench import (
    _backend_mtp_engaged,
    _cell_correctness,
    _diagnostic_plan,
    _generated_ids,
    _install_diagnostic_plan,
    _install_native_prefill_probe,
    _memory_delta,
    _mtp_budget_conformed,
    _mtp_engaged,
    _parse_expected_mtp_widths,
    _parse_widths,
    _request_mtp_value,
    _render_messages,
    _resident_observability,
    _response_attribution,
    build_parser,
    summarize,
    summarize_acceptance,
    summarize_prefill_attribution,
    verdict_reasons,
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

    automatic = parser.parse_args(
        ("--mtp-request-mode", "automatic", "--output", "/tmp/out.json")
    )
    attribution = parser.parse_args(
        ("--capture-prefill-attribution", "--output", "/tmp/out.json")
    )

    assert default.execution_profile == "production"
    assert default.mtp_request_mode == "explicit"
    assert default.resident_capacity is None
    assert strict.execution_profile == "strict"
    assert automatic.mtp_request_mode == "automatic"
    assert default.capture_prefill_attribution is False
    assert attribution.capture_prefill_attribution is True
    assert normal_owner.widths == (2,)
    assert normal_owner.resident_capacity == 4


def test_mtp_c1c8_request_mode_separates_explicit_and_automatic() -> None:
    assert _request_mtp_value(arm="ar", request_mode="explicit") is False
    assert _request_mtp_value(arm="ar", request_mode="automatic") is False
    assert _request_mtp_value(arm="mtp", request_mode="explicit") is True
    assert _request_mtp_value(arm="mtp", request_mode="automatic") is None
    with pytest.raises(ValueError, match="request mode"):
        _request_mtp_value(arm="mtp", request_mode="unknown")


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


def test_mtp_c1c8_extracts_response_timing_and_generation_shape() -> None:
    payload = {
        "choices": [
            {
                "hipengine": {
                    "timing": {
                        "render_ms": 1,
                        "tokenize_ms": 2.5,
                        "admission_prepare_ms": 3,
                        "prefill_ms": 200,
                        "request_total_ms": 240,
                    },
                    "timing_scope": "choice",
                    "batch_id": "batch-7",
                    "group_rows": 2,
                    "timing_owner": True,
                    "decode_state": {"execution_path": "gguf_packed_ar_server_decode"},
                }
            }
        ],
        "hipengine": {
            "generation_shape": {
                "queue_group": {"id": "queue-1", "request_count": 2},
                "backend_groups": [{"input_rows": 2, "actual_group_rows": [2]}],
            }
        },
    }

    attribution = _response_attribution(payload)

    assert attribution["timing"] == {
        "render_ms": 1.0,
        "tokenize_ms": 2.5,
        "admission_prepare_ms": 3.0,
        "prefill_ms": 200.0,
        "request_total_ms": 240.0,
    }
    assert attribution["timing_scope"] == "choice"
    assert attribution["batch_id"] == "batch-7"
    assert attribution["group_rows"] == 2
    assert attribution["timing_owner"] is True
    assert attribution["execution_path"] == "gguf_packed_ar_server_decode"
    assert attribution["generation_shape"]["queue_group"]["request_count"] == 2


def test_mtp_c1c8_native_prefill_probe_records_actual_handled_rows() -> None:
    class Owner:
        def _try_prefill_native_work_batch(self, work):
            return frozenset((11, 13))

    owner = Owner()
    llm = SimpleNamespace(
        _get_text_generator=lambda: SimpleNamespace(
            _driver=SimpleNamespace(_runner=owner)
        )
    )
    probe = _install_native_prefill_probe(llm)
    cursor = probe.cursor()

    handled = owner._try_prefill_native_work_batch(
        SimpleNamespace(
            request_ids=(11, 12, 13),
            token_rows=((1, 2, 3), (4,), (5, 6)),
            kind="prefill",
        )
    )

    assert handled == frozenset((11, 13))
    records = probe.since(cursor)
    assert len(records) == 1
    elapsed_ms = records[0].pop("elapsed_ms")
    assert records == [
        {
            "work_rows": 3,
            "handled_rows": 2,
            "handled_request_ids": [11, 13],
            "prompt_lengths": [3, 2],
            "work_kind": "prefill",
        }
    ]
    assert elapsed_ms >= 0.0


def test_mtp_c1c8_prefill_attribution_closes_critical_wave_wall() -> None:
    timing = {
        "render_ms": 1.0,
        "tokenize_ms": 2.0,
        "admission_prepare_ms": 3.0,
        "prefill_ms": 200.0,
        "request_total_ms": 240.0,
    }
    cells = [
        {
            "width": 2,
            "prompt_id": "p0",
            "ar": {
                "wall_seconds": 0.300,
                "rows": [
                    {
                        "started": 1.0,
                        "completed": 1.290,
                        "attribution": {"timing": dict(timing), "generation_shape": {}},
                    },
                    {
                        "started": 1.0,
                        "completed": 1.300,
                        "attribution": {
                            "timing": dict(timing),
                            "generation_shape": {
                                "queue_group": {"request_count": 2},
                            },
                        },
                    },
                ],
                "native_prefill_groups": [
                    {
                        "work_rows": 2,
                        "handled_rows": 2,
                        "handled_request_ids": [11, 12],
                        "prompt_lengths": [45, 45],
                        "work_kind": "prefill",
                        "elapsed_ms": 201.0,
                    }
                ],
                "native_full_prefill_groups_delta": 1,
            },
            "mtp": {
                "wall_seconds": 0.300,
                "rows": [],
                "native_prefill_groups": [],
                "native_full_prefill_groups_delta": 0,
            },
        }
    ]

    summary = summarize_prefill_attribution(cells, arms=("ar",))
    row = summary["by_width"]["2"]["ar"]

    assert row["cells"] == 1
    assert row["mean_ms"] == {
        "wave_wall": 300.0,
        "render": 1.0,
        "tokenize": 2.0,
        "admission_prepare": 3.0,
        "native_prefill": 200.0,
        "remaining": 94.0,
        "engine_loop_residual": 40.0,
        "server_queue_response_residual": 54.0,
        "request_total": 240.0,
    }
    assert row["native_group_size_histogram"] == {"2": 1}
    assert row["native_full_prefill_groups_delta"] == 1
    assert row["queue_request_count_histogram"] == {"2": 1}
    assert summary["refused_cells"] == []


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
    assert frontend_c1["static_eligibility"]["max_realized_group_rows"] == 4
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


# --- verdict_reasons: a `failed` status must never be silent ----------------------

def _row(
    *,
    cells: int = 10,
    exact: int | None = None,
    budget: int | None = None,
    engaged: int = 10,
    mtp_expected: bool = True,
) -> dict:
    exact = cells if exact is None else exact
    budget = cells if budget is None else budget
    return {
        "cells": cells,
        "exact_cells": exact,
        "budget_conformed_cells": budget,
        "engaged_cells": engaged,
        "mtp_expected": mtp_expected,
        "route_expectation_passed": (engaged == cells) if mtp_expected else engaged == 0,
    }


def _old_predicate(summary: dict) -> bool:
    """The pre-existing pass condition, kept here so equivalence is checked, not asserted."""
    return all(
        int(row["exact_cells"]) == int(row["budget_conformed_cells"]) == int(row["cells"]) == 10
        and bool(row["route_expectation_passed"])
        for row in summary.values()
    )


def test_verdict_reasons_is_empty_for_a_clean_run() -> None:
    summary = {str(width): _row() for width in range(1, 9)}
    assert verdict_reasons(summary, expected_cells=10, max_tokens=24) == []


def test_verdict_reasons_matches_the_original_predicate() -> None:
    grid = [
        dict(),
        dict(exact=9),
        dict(budget=9),
        dict(cells=9),
        dict(engaged=9),
        dict(engaged=0, mtp_expected=False),
        dict(engaged=1, mtp_expected=False),
        dict(exact=9, budget=8, engaged=0),
    ]
    for index, kwargs in enumerate(grid):
        for width_pair in (({"1": _row(**kwargs)}), ({"1": _row(), "4": _row(**kwargs)})):
            summary = {str(index) + k: v for k, v in width_pair.items()}
            reasons = verdict_reasons(summary, expected_cells=10, max_tokens=24)
            assert bool(reasons) == (not _old_predicate(summary)), (index, kwargs, reasons)


def test_verdict_reasons_names_the_cells_that_disagreed() -> None:
    summary = {"4": _row(exact=9, budget=8)}
    reasons = verdict_reasons(summary, expected_cells=10, max_tokens=24)
    assert len(reasons) == 1
    assert "w4" in reasons[0]
    assert "exact=9" in reasons[0]
    assert "budget_conformed=8" in reasons[0]
    assert "cells=10" in reasons[0]


def test_verdict_reasons_explains_the_max_tokens_one_route_failure() -> None:
    """Reproduces 2026-08-30: 8 widths x 10 prompts all exact, MTP never engaged, no explanation."""
    summary = {str(width): _row(engaged=0) for width in range(1, 9)}
    reasons = verdict_reasons(summary, expected_cells=10, max_tokens=1)
    assert len(reasons) == 8, reasons
    assert all(f"w{width}:" in reasons[width - 1] for width in range(1, 9))
    assert "max_tokens=1" in reasons[0]
    assert "admission overhead" in reasons[0], "the d1 mtp/ar ratio must not read as speculation"


def test_verdict_reasons_omits_the_single_token_note_when_tokens_allow_speculation() -> None:
    reasons = verdict_reasons({"1": _row(engaged=0)}, expected_cells=10, max_tokens=24)
    assert len(reasons) == 1
    assert "max_tokens=1" not in reasons[0]
    assert "engaged=0/10" in reasons[0]
