from __future__ import annotations

import pytest

from scripts.laguna_grouped_down_prefill_bench import (
    FUSED_COMBINE_MODES,
    MODES,
    _comparison_summary,
    _grouped_combine_summary,
    _mode_order,
    _mode_scope,
)


def _samples(
    direct: dict[int, list[float]],
    candidate: dict[int, list[float]],
) -> dict[str, dict[int, list[float]]]:
    return {
        "direct": direct,
        "adaptive_grouped_smallm": candidate,
    }


def _tokens(
    rows: tuple[int, ...], candidate_offset: int = 0
) -> dict[str, dict[int, list[int]]]:
    return {
        "direct": {row: [row, row] for row in rows},
        "adaptive_grouped_smallm": {
            row: [row + candidate_offset, row + candidate_offset] for row in rows
        },
    }


def test_grouped_down_mode_order_is_counterbalanced() -> None:
    assert MODES == ("direct", "adaptive_grouped_smallm")
    assert _mode_order(0, 0) == MODES
    assert _mode_order(1, 0) == tuple(reversed(MODES))
    assert _mode_order(0, 1) == tuple(reversed(MODES))


def test_grouped_combine_mode_order_and_nonregression_screen() -> None:
    rows = (16, 32)
    assert FUSED_COMBINE_MODES == (
        "adaptive_grouped_smallm",
        "adaptive_grouped_smallm_fused",
    )
    assert _mode_order(0, 0, modes=FUSED_COMBINE_MODES) == FUSED_COMBINE_MODES
    result = _grouped_combine_summary(
        {
            FUSED_COMBINE_MODES[0]: {16: [1.0, 1.0], 32: [1.0, 1.0]},
            FUSED_COMBINE_MODES[1]: {16: [1.001, 1.001], 32: [0.999, 0.999]},
        },
        {
            mode: {row: [row, row] for row in rows}
            for mode in FUSED_COMBINE_MODES
        },
        rows=rows,
    )

    assert result["correctness"]["pass"] is True
    assert result["screen"]["pass"] is True
    assert result["screen"]["effective_speedup"] == pytest.approx(1.0)


def test_grouped_combine_screen_rejects_mismatch_or_material_regression() -> None:
    rows = (16, 32)
    tokens = {
        mode: {row: [row, row] for row in rows}
        for mode in FUSED_COMBINE_MODES
    }
    tokens[FUSED_COMBINE_MODES[1]][32][-1] = 999
    result = _grouped_combine_summary(
        {
            FUSED_COMBINE_MODES[0]: {16: [1.0, 1.0], 32: [1.0, 1.0]},
            FUSED_COMBINE_MODES[1]: {16: [1.0, 1.0], 32: [1.01, 1.01]},
        },
        tokens,
        rows=rows,
    )

    assert result["correctness"]["pass"] is False
    assert result["screen"]["pass"] is False
    assert "output_ids_not_exact" in result["screen"]["failed_checks"]
    assert "candidate_below_no_regression_floor" in result["screen"]["failed_checks"]


def test_grouped_down_summary_allows_exact_direct_fallback_and_gates_grouped_gain() -> None:
    rows = (16, 32)
    result = _comparison_summary(
        _samples(
            {16: [1.0, 1.0], 32: [1.0, 1.0]},
            {16: [1.001, 1.001], 32: [0.9, 0.9]},
        ),
        _tokens(rows),
        rows=rows,
    )

    assert result["correctness"]["pass"] is True
    assert result["promotion"]["pass"] is True
    assert result["shapes"]["16"]["route"] == "direct_fallback"
    assert result["shapes"]["32"]["route"] == "grouped_smallm"


def test_grouped_down_summary_fails_mismatch_or_slow_grouped_shape() -> None:
    rows = (16, 32)
    result = _comparison_summary(
        _samples(
            {16: [1.0, 1.0], 32: [1.0, 1.0]},
            {16: [1.0, 1.0], 32: [1.1, 1.1]},
        ),
        _tokens(rows, candidate_offset=1),
        rows=rows,
    )

    assert result["correctness"]["pass"] is False
    assert result["promotion"]["pass"] is False
    assert "output_ids_not_exact" in result["promotion"]["failed_checks"]
    assert "grouped_route_not_faster" in result["promotion"]["failed_checks"]


def test_grouped_down_mode_scope_restores_session_local_selector() -> None:
    from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
    session = object.__new__(LagunaGGUFResidentSession)
    session.backend = "hip_gfx1151"
    session.selected_gate_up_mode = "grouped_exact"
    session.selected_down_mode = "direct"
    with pytest.raises(RuntimeError, match="synthetic"):
        with _mode_scope(session, "adaptive_grouped_smallm"):
            assert session.selected_down_mode == "adaptive_grouped_smallm"
            raise RuntimeError("synthetic")
    assert session.selected_down_mode == "direct"
    assert session.selected_gate_up_mode == "grouped_exact"
    with pytest.raises(ValueError, match="unknown Laguna grouped-down mode"):
        with _mode_scope(session, "bogus"):
            pytest.fail("invalid mode entered")
